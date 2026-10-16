package com.doorcounter.app;

import android.content.Context;
import android.content.SharedPreferences;

/** Phone-side settings. Stored in SharedPreferences and changed from the laptop through {@link ConfigReceiver}. */
final class Settings {
    /** Reaches the laptop over USB after {@code adb reverse tcp:8000 tcp:8000}. */
    static final String DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000";
    static final int DEFAULT_ROTATION_DEG = 0;
    static final int DEFAULT_THREAD_COUNT = 4;

    private static final String PREFS_NAME = "settings";
    private static final String KEY_ROTATION_DEG = "rotation_deg";
    private static final String KEY_GATEWAY_URL = "gateway_url";
    private static final String KEY_PREVIEW = "preview";
    private static final String KEY_THREAD_COUNT = "thread_count";
    private static final String KEY_MOTION_GATE = "motion_gate";

    /** Clockwise rotation (0, 90, 180, 270) that makes the camera frame upright. */
    final int rotationDeg;
    /** Base URL of the laptop gateway. */
    final String gatewayUrl;
    /** Send a small preview picture to the gateway. Off by default: images leave the phone only on purpose. */
    final boolean isPreviewEnabled;
    /** CPU threads for the detector. */
    final int threadCount;
    /** Skip the model while nothing moves (see {@link MotionGate}). On by default. */
    final boolean isMotionGateEnabled;

    private Settings(int rotationDeg, String gatewayUrl, boolean isPreviewEnabled, int threadCount,
                     boolean isMotionGateEnabled) {
        this.rotationDeg = rotationDeg;
        this.gatewayUrl = gatewayUrl;
        this.isPreviewEnabled = isPreviewEnabled;
        this.threadCount = threadCount;
        this.isMotionGateEnabled = isMotionGateEnabled;
    }

    /**
     * Read the stored settings, with defaults for anything never set.
     *
     * @param context any context of this app
     * @return the current settings
     */
    static Settings load(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        return new Settings(
                prefs.getInt(KEY_ROTATION_DEG, DEFAULT_ROTATION_DEG),
                prefs.getString(KEY_GATEWAY_URL, DEFAULT_GATEWAY_URL),
                prefs.getBoolean(KEY_PREVIEW, false),
                prefs.getInt(KEY_THREAD_COUNT, DEFAULT_THREAD_COUNT),
                prefs.getBoolean(KEY_MOTION_GATE, true));
    }

    /**
     * Store new settings.
     *
     * @param context any context of this app
     * @param rotationDeg clockwise rotation, one of 0, 90, 180, 270
     * @param gatewayUrl base URL of the gateway
     * @param isPreviewEnabled whether to send preview pictures
     * @param threadCount detector threads, 1 to 4
     * @param isMotionGateEnabled whether to skip the model while nothing moves
     * @throws IllegalArgumentException if a value is out of range
     */
    static void save(Context context, int rotationDeg, String gatewayUrl, boolean isPreviewEnabled, int threadCount,
                     boolean isMotionGateEnabled) {
        if (rotationDeg % 90 != 0 || rotationDeg < 0 || rotationDeg > 270) {
            throw new IllegalArgumentException("rotation must be 0, 90, 180 or 270, got " + rotationDeg);
        }
        if (threadCount < 1 || threadCount > 4) {
            throw new IllegalArgumentException("threads must be 1 to 4, got " + threadCount);
        }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
                .putInt(KEY_ROTATION_DEG, rotationDeg)
                .putString(KEY_GATEWAY_URL, gatewayUrl)
                .putBoolean(KEY_PREVIEW, isPreviewEnabled)
                .putInt(KEY_THREAD_COUNT, threadCount)
                .putBoolean(KEY_MOTION_GATE, isMotionGateEnabled)
                .apply();
    }

    @Override
    public String toString() {
        return "rotation_deg=" + rotationDeg + " gateway=" + gatewayUrl + " preview=" + isPreviewEnabled
                + " threads=" + threadCount + " motion_gate=" + isMotionGateEnabled;
    }
}
