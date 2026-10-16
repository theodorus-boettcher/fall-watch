// Person detector for the door counter.
//
// One camera frame goes through: NV21 -> RGB -> resize -> rotate upright -> letterbox to the
// model size -> YOLO26n (ncnn) -> person boxes -> overlap removal. Boxes come back to Java as a
// flat float array [x1, y1, x2, y2, score] per person, normalised to 0..1 of the upright frame.

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
// Rows of the model output: cx, cy, w, h, then one score row per COCO class. Person is class 0.
constexpr int kPersonScoreRow = 4;
constexpr float kOverlapIou = 0.7f;  // same as the Ultralytics default
constexpr size_t kMaxPersons = 50;
constexpr int kValuesPerBox = 5;

struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

// Intersection over union of two boxes, 0 when they do not touch.
float iou(const Box& a, const Box& b) {
    const float w = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float h = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = w * h;
    const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// Keep the best box of each group of heavily overlapping boxes (greedy NMS).
std::vector<Box> remove_overlaps(std::vector<Box> boxes) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
    std::vector<Box> kept;
    for (const Box& candidate : boxes) {
        const bool is_duplicate = std::any_of(kept.begin(), kept.end(),
                                              [&](const Box& k) { return iou(candidate, k) > kOverlapIou; });
        if (!is_duplicate) {
            kept.push_back(candidate);
            if (kept.size() == kMaxPersons) {
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

class PersonDetector {
public:
    // Load the model from the APK assets. Returns false when a file is missing or broken.
    bool load(AAssetManager* assets, const std::string& param_name, const std::string& bin_name, int threads) {
        net_.opt.num_threads = threads;
        return net_.load_param(assets, param_name.c_str()) == 0 && net_.load_model(assets, bin_name.c_str()) == 0;
    }

    // Find people in one NV21 frame. Boxes are normalised to the upright frame.
    std::vector<Box> detect(const unsigned char* nv21, int w, int h, int rotation_deg, float min_score) {
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
        const int up_w = is_swapped ? small_h : small_w;
        const int up_h = is_swapped ? small_w : small_h;

        const int pad_left = (kInputSizePx - up_w) / 2;
        const int pad_top = (kInputSizePx - up_h) / 2;
        ncnn::Mat unpadded = ncnn::Mat::from_pixels(upright.data(), ncnn::Mat::PIXEL_RGB, up_w, up_h);
        ncnn::Mat input;
        ncnn::copy_make_border(unpadded, input, pad_top, kInputSizePx - up_h - pad_top, pad_left,
                               kInputSizePx - up_w - pad_left, ncnn::BORDER_CONSTANT, kPadValue);
        const float norm[3] = {kPixelScale, kPixelScale, kPixelScale};
        input.substract_mean_normalize(nullptr, norm);

        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", input);
        ncnn::Mat out;
        ex.extract("out0", out);

        const float* cx = out.row(0);
        const float* cy = out.row(1);
        const float* bw = out.row(2);
        const float* bh = out.row(3);
        const float* score = out.row(kPersonScoreRow);
        std::vector<Box> candidates;
        for (int i = 0; i < out.w; ++i) {
            if (score[i] < min_score) {
                continue;
            }
            const auto to_x = [&](float v) { return std::clamp((v - pad_left) / up_w, 0.f, 1.f); };
            const auto to_y = [&](float v) { return std::clamp((v - pad_top) / up_h, 0.f, 1.f); };
            candidates.push_back(Box{to_x(cx[i] - bw[i] / 2), to_y(cy[i] - bh[i] / 2), to_x(cx[i] + bw[i] / 2),
                                     to_y(cy[i] + bh[i] / 2), score[i]});
        }
        return remove_overlaps(std::move(candidates));
    }

private:
    ncnn::Net net_;
};

PersonDetector* from_handle(jlong handle) {
    return reinterpret_cast<PersonDetector*>(handle);
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
    auto detector = std::make_unique<PersonDetector>();
    AAssetManager* assets = AAssetManager_fromJava(env, asset_manager);
    if (!detector->load(assets, to_string(env, param_name), to_string(env, bin_name), threads)) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "could not load the model from assets");
        return 0;
    }
    __android_log_print(ANDROID_LOG_INFO, kTag, "model loaded, %d threads", threads);
    return reinterpret_cast<jlong>(detector.release());
}

JNIEXPORT jfloatArray JNICALL Java_com_doorcounter_app_Detector_nativeDetect(JNIEnv* env, jclass, jlong handle,
                                                                             jbyteArray nv21, jint width,
                                                                             jint height, jint rotation_deg,
                                                                             jfloat min_score) {
    jbyte* pixels = env->GetByteArrayElements(nv21, nullptr);
    const std::vector<Box> boxes = from_handle(handle)->detect(reinterpret_cast<const unsigned char*>(pixels), width,
                                                               height, rotation_deg, min_score);
    env->ReleaseByteArrayElements(nv21, pixels, JNI_ABORT);  // read only: nothing to copy back

    std::vector<float> flat;
    flat.reserve(boxes.size() * kValuesPerBox);
    for (const Box& b : boxes) {
        flat.insert(flat.end(), {b.x1, b.y1, b.x2, b.y2, b.score});
    }
    jfloatArray result = env->NewFloatArray(static_cast<jsize>(flat.size()));
    env->SetFloatArrayRegion(result, 0, static_cast<jsize>(flat.size()), flat.data());
    return result;
}

JNIEXPORT void JNICALL Java_com_doorcounter_app_Detector_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    delete from_handle(handle);
}

}  // extern "C"
