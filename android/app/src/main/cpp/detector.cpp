// Item segmenter for the sorting line (YOLO26-seg through ncnn).
//
// One camera frame goes through: NV21 -> RGB -> resize -> rotate upright -> letterbox to the model
// size -> model -> items with masks -> overlap removal. This is the same math as
// src/door_counter/yolo_seg.py on the laptop; keep the two in step.
//
// Results go back to Java as one float record per item (see kValuesPerItem) plus, from a second call,
// the run-length encoded mask cells of the last frame (runs alternate off/on, starting with off).

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mat.h"
#include "net.h"

namespace {

constexpr const char* kTag = "DoorDetector";
constexpr int kInputSizePx = 320;
constexpr float kPadValue = 114.f;  // grey letterbox border, same as Ultralytics
constexpr float kPixelScale = 1.f / 255.f;
constexpr int kBoxRows = 4;         // cx, cy, w, h; then class scores; then mask weights
constexpr int kProtoStridePx = 4;   // the mask prototypes are a quarter of the input size
constexpr float kOverlapIou = 0.7f; // same as the Ultralytics default
constexpr size_t kMaxItems = 50;
// class_id, score, box x1 y1 x2 y2, mask rect x1 y1 x2 y2, mask width/height in cells, run offset, run count
constexpr int kValuesPerItem = 14;

struct Rect {
    float x1;
    float y1;
    float x2;
    float y2;

    float area() const { return std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1); }
};

struct Candidate {
    Rect box;  // letterbox pixels
    float score;
    int class_id;
    int column;
};

struct Letterbox {
    int pad_left;
    int pad_top;
    int content_w;
    int content_h;
};

float iou(const Rect& a, const Rect& b) {
    const Rect inter{std::max(a.x1, b.x1), std::max(a.y1, b.y1), std::min(a.x2, b.x2), std::min(a.y2, b.y2)};
    const float uni = a.area() + b.area() - inter.area();
    return uni > 0.f ? inter.area() / uni : 0.f;
}

// Keep the best candidate of each group of heavily overlapping ones, across classes:
// each item is made of one material.
std::vector<Candidate> remove_overlaps(std::vector<Candidate> candidates) {
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    std::vector<Candidate> kept;
    for (const Candidate& c : candidates) {
        const bool is_duplicate = std::any_of(kept.begin(), kept.end(),
                                              [&](const Candidate& k) { return iou(c.box, k.box) > kOverlapIou; });
        if (!is_duplicate) {
            kept.push_back(c);
            if (kept.size() == kMaxItems) {
                break;
            }
        }
    }
    return kept;
}

// Rotate an RGB image clockwise by 0, 90, 180 or 270 degrees.
std::vector<unsigned char> rotate_rgb(const std::vector<unsigned char>& src, int w, int h, int rotation_deg) {
    std::vector<unsigned char> dst(src.size());
    const int dst_w = (rotation_deg == 90 || rotation_deg == 270) ? h : w;
    const int dst_h = (rotation_deg == 90 || rotation_deg == 270) ? w : h;
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int sx = x;
            int sy = y;
            if (rotation_deg == 90) {
                sx = y;
                sy = h - 1 - x;
            } else if (rotation_deg == 180) {
                sx = w - 1 - x;
                sy = h - 1 - y;
            } else if (rotation_deg == 270) {
                sx = w - 1 - y;
                sy = x;
            }
            std::copy_n(&src[(sy * w + sx) * 3], 3, &dst[(y * dst_w + x) * 3]);
        }
    }
    return dst;
}

// Letterbox pixels -> 0..1 of the upright frame.
float to_frame_x(float v, const Letterbox& lb) {
    return std::clamp((v - lb.pad_left) / static_cast<float>(lb.content_w), 0.f, 1.f);
}

float to_frame_y(float v, const Letterbox& lb) {
    return std::clamp((v - lb.pad_top) / static_cast<float>(lb.content_h), 0.f, 1.f);
}

class Segmenter {
public:
    // Load the model from the APK assets. Returns false when a file is missing or broken.
    bool load(AAssetManager* assets, const std::string& param_name, const std::string& bin_name, int threads) {
        net_.opt.num_threads = threads;
        return net_.load_param(assets, param_name.c_str()) == 0 && net_.load_model(assets, bin_name.c_str()) == 0;
    }

    // Find items in one NV21 frame. Fills the records and the runs of this frame.
    void segment(const unsigned char* nv21, int w, int h, int rotation_deg, float min_score) {
        records_.clear();
        runs_.clear();
        std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
        ncnn::yuv420sp2rgb(nv21, w, h, rgb.data());

        // Resize first (cheaper to rotate a small image), keeping the aspect ratio.
        const bool is_swapped = rotation_deg == 90 || rotation_deg == 270;
        const float scale = static_cast<float>(kInputSizePx) / static_cast<float>(std::max(w, h));
        const int small_w = static_cast<int>(std::round(w * scale));
        const int small_h = static_cast<int>(std::round(h * scale));
        std::vector<unsigned char> small(static_cast<size_t>(small_w) * small_h * 3);
        ncnn::resize_bilinear_c3(rgb.data(), w, h, small.data(), small_w, small_h);
        const std::vector<unsigned char> upright = rotate_rgb(small, small_w, small_h, rotation_deg);
        Letterbox lb{0, 0, is_swapped ? small_h : small_w, is_swapped ? small_w : small_h};
        lb.pad_left = (kInputSizePx - lb.content_w) / 2;
        lb.pad_top = (kInputSizePx - lb.content_h) / 2;

        ncnn::Mat unpadded = ncnn::Mat::from_pixels(upright.data(), ncnn::Mat::PIXEL_RGB, lb.content_w, lb.content_h);
        ncnn::Mat input;
        ncnn::copy_make_border(unpadded, input, lb.pad_top, kInputSizePx - lb.content_h - lb.pad_top, lb.pad_left,
                               kInputSizePx - lb.content_w - lb.pad_left, ncnn::BORDER_CONSTANT, kPadValue);
        const float norm[3] = {kPixelScale, kPixelScale, kPixelScale};
        input.substract_mean_normalize(nullptr, norm);

        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", input);
        ncnn::Mat out;
        ncnn::Mat protos;
        ex.extract("out0", out);
        ex.extract("out1", protos);
        const int mask_dim = protos.c;
        const int class_count = out.h - kBoxRows - mask_dim;
        if (class_count < 1) {
            __android_log_print(ANDROID_LOG_ERROR, kTag, "out0 has %d rows, too few for %d mask weights", out.h,
                                mask_dim);
            return;
        }

        std::vector<Candidate> candidates;
        for (int i = 0; i < out.w; ++i) {
            int best_class = 0;
            float best_score = out.row(kBoxRows)[i];
            for (int c = 1; c < class_count; ++c) {
                const float s = out.row(kBoxRows + c)[i];
                if (s > best_score) {
                    best_score = s;
                    best_class = c;
                }
            }
            if (best_score < min_score) {
                continue;
            }
            const float cx = out.row(0)[i];
            const float cy = out.row(1)[i];
            const float bw = out.row(2)[i];
            const float bh = out.row(3)[i];
            candidates.push_back(Candidate{Rect{cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2}, best_score,
                                           best_class, i});
        }
        for (const Candidate& c : remove_overlaps(std::move(candidates))) {
            add_item(c, out, protos, class_count, lb);
        }
    }

    const std::vector<float>& records() const { return records_; }
    const std::vector<int>& runs() const { return runs_; }

private:
    // Build one item's mask on the prototype grid (cut to its box and to the picture area) and store it.
    void add_item(const Candidate& c, const ncnn::Mat& out, const ncnn::Mat& protos, int class_count,
                  const Letterbox& lb) {
        const int grid_w = protos.w;
        const int grid_h = protos.h;
        const float content_x2 = static_cast<float>(lb.pad_left + lb.content_w);
        const float content_y2 = static_cast<float>(lb.pad_top + lb.content_h);
        const int x1 = std::clamp(static_cast<int>(std::floor(std::max(c.box.x1, float(lb.pad_left)) / kProtoStridePx)),
                                  0, grid_w);
        const int y1 = std::clamp(static_cast<int>(std::floor(std::max(c.box.y1, float(lb.pad_top)) / kProtoStridePx)),
                                  0, grid_h);
        const int x2 = std::clamp(static_cast<int>(std::ceil(std::min(c.box.x2, content_x2) / kProtoStridePx)), x1,
                                  grid_w);
        const int y2 = std::clamp(static_cast<int>(std::ceil(std::min(c.box.y2, content_y2) / kProtoStridePx)), y1,
                                  grid_h);

        // sigmoid(logit) > 0.5 is the same as logit > 0, so no exp() is needed.
        const int run_offset = static_cast<int>(runs_.size());
        bool current = false;
        int length = 0;
        for (int y = y1; y < y2; ++y) {
            for (int x = x1; x < x2; ++x) {
                float logit = 0.f;
                for (int k = 0; k < protos.c; ++k) {
                    logit += out.row(kBoxRows + class_count + k)[c.column] * protos.channel(k).row(y)[x];
                }
                const bool is_on = logit > 0.f;
                if (is_on == current) {
                    ++length;
                } else {
                    runs_.push_back(length);
                    current = is_on;
                    length = 1;
                }
            }
        }
        runs_.push_back(length);

        const float cell = static_cast<float>(kProtoStridePx);
        records_.insert(records_.end(),
                        {static_cast<float>(c.class_id), c.score, to_frame_x(c.box.x1, lb), to_frame_y(c.box.y1, lb),
                         to_frame_x(c.box.x2, lb), to_frame_y(c.box.y2, lb), to_frame_x(x1 * cell, lb),
                         to_frame_y(y1 * cell, lb), to_frame_x(x2 * cell, lb), to_frame_y(y2 * cell, lb),
                         static_cast<float>(x2 - x1), static_cast<float>(y2 - y1), static_cast<float>(run_offset),
                         static_cast<float>(runs_.size() - run_offset)});
    }

    ncnn::Net net_;
    std::vector<float> records_;
    std::vector<int> runs_;
};

Segmenter* from_handle(jlong handle) {
    return reinterpret_cast<Segmenter*>(handle);
}

std::string to_string(JNIEnv* env, jstring s) {
    const char* chars = env->GetStringUTFChars(s, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(s, chars);
    return result;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_doorcounter_app_Detector_nativeCreate(JNIEnv* env, jclass, jobject asset_manager,
                                                                       jstring param_name, jstring bin_name,
                                                                       jint threads) {
    auto segmenter = std::make_unique<Segmenter>();
    AAssetManager* assets = AAssetManager_fromJava(env, asset_manager);
    if (!segmenter->load(assets, to_string(env, param_name), to_string(env, bin_name), threads)) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "could not load the model from assets");
        return 0;
    }
    __android_log_print(ANDROID_LOG_INFO, kTag, "model loaded, %d threads", threads);
    return reinterpret_cast<jlong>(segmenter.release());
}

JNIEXPORT jfloatArray JNICALL Java_com_doorcounter_app_Detector_nativeDetect(JNIEnv* env, jclass, jlong handle,
                                                                             jbyteArray nv21, jint width,
                                                                             jint height, jint rotation_deg,
                                                                             jfloat min_score) {
    Segmenter* segmenter = from_handle(handle);
    jbyte* pixels = env->GetByteArrayElements(nv21, nullptr);
    segmenter->segment(reinterpret_cast<const unsigned char*>(pixels), width, height, rotation_deg, min_score);
    env->ReleaseByteArrayElements(nv21, pixels, JNI_ABORT);  // read only: nothing to copy back
    const std::vector<float>& records = segmenter->records();
    jfloatArray result = env->NewFloatArray(static_cast<jsize>(records.size()));
    env->SetFloatArrayRegion(result, 0, static_cast<jsize>(records.size()), records.data());
    return result;
}

JNIEXPORT jintArray JNICALL Java_com_doorcounter_app_Detector_nativeLastRuns(JNIEnv* env, jclass, jlong handle) {
    const std::vector<int>& runs = from_handle(handle)->runs();
    jintArray result = env->NewIntArray(static_cast<jsize>(runs.size()));
    env->SetIntArrayRegion(result, 0, static_cast<jsize>(runs.size()), runs.data());
    return result;
}

JNIEXPORT void JNICALL Java_com_doorcounter_app_Detector_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    delete from_handle(handle);
}

}  // extern "C"
