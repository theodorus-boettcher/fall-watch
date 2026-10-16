package com.doorcounter.app;

import android.content.res.AssetManager;

/**
 * Item segmenter running on the phone (YOLO26-seg through ncnn, see {@code src/main/cpp/detector.cpp}).
 *
 * <p>Not thread safe: call {@link #detect} and {@link #lastRuns} from one thread only.
 */
final class Detector implements AutoCloseable {
    /**
     * Values per item in the array returned by {@link #detect}: class id, score, box x1 y1 x2 y2,
     * mask rect x1 y1 x2 y2 (all 0..1 of the upright frame), mask width and height in cells,
     * offset and count of the item's runs in {@link #lastRuns}.
     */
    static final int VALUES_PER_ITEM = 14;
    static final int CLASS_ID = 0;
    static final int SCORE = 1;
    static final int BOX = 2;
    static final int MASK_RECT = 6;
    static final int MASK_WIDTH = 10;
    static final int MASK_HEIGHT = 11;
    static final int RUN_OFFSET = 12;
    static final int RUN_COUNT = 13;

    static {
        System.loadLibrary("detector");
    }

    private long handle;

    /**
     * Load a model from the APK assets.
     *
     * @param assets the app's asset manager
     * @param modelName asset path without extension, e.g. {@code models/yolo26n-det-320}
     * @param threadCount CPU threads for inference
     * @throws IllegalStateException if the model files are missing or broken
     */
    Detector(AssetManager assets, String modelName, int threadCount) {
        handle = nativeCreate(assets, modelName + ".param", modelName + ".bin", threadCount);
    }

    /**
     * Find items in one camera frame.
     *
     * @param nv21 the frame in NV21 layout
     * @param widthPx frame width as delivered by the camera
     * @param heightPx frame height as delivered by the camera
     * @param rotationDeg clockwise rotation (0, 90, 180, 270) that makes the frame upright
     * @param minScore lowest confidence to keep
     * @return {@link #VALUES_PER_ITEM} values per item (see there)
     * @throws IllegalStateException if the detector was closed
     */
    float[] detect(byte[] nv21, int widthPx, int heightPx, int rotationDeg, float minScore) {
        if (handle == 0) {
            throw new IllegalStateException("detector is closed");
        }
        return nativeDetect(handle, nv21, widthPx, heightPx, rotationDeg, minScore);
    }

    /**
     * The mask runs of the frame passed to the last {@link #detect} call: per item, the patch is read
     * row by row and the runs alternate between "off" and "on" cells, starting with "off".
     *
     * @return all items' runs, one after the other
     * @throws IllegalStateException if the detector was closed
     */
    int[] lastRuns() {
        if (handle == 0) {
            throw new IllegalStateException("detector is closed");
        }
        return nativeLastRuns(handle);
    }

    @Override
    public void close() {
        if (handle != 0) {
            nativeDestroy(handle);
            handle = 0;
        }
    }

    private static native long nativeCreate(AssetManager assets, String paramName, String binName, int threadCount);

    private static native float[] nativeDetect(long handle, byte[] nv21, int widthPx, int heightPx, int rotationDeg,
                                               float minScore);

    private static native int[] nativeLastRuns(long handle);

    private static native void nativeDestroy(long handle);
}
