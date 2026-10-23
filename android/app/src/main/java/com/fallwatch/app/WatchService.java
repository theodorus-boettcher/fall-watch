package com.fallwatch.app;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.net.wifi.WifiManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Foreground service that finds people in the camera stream and sends postures, boxes and keypoints to the gateway.
 *
 * <p>Works with the screen off: the preview goes to an off-screen SurfaceTexture and a partial
 * wake lock keeps the CPU running. The camera delivers frames into a few reused buffers. When the
 * detector is free it takes the newest frame; frames that arrive while it is busy go straight
 * back to the camera, so the detector always works on a fresh picture and never builds a backlog.
 */
public final class WatchService extends Service {
    private static final String TAG = "FallWatch";
    private static final int NOTIFICATION_ID = 1;
    private static final float MIN_SCORE = 0.25f;
    // The model sees 320 px anyway; smaller preview pictures also get through the S5's lossy Wi-Fi faster.
    private static final int TARGET_PREVIEW_WIDTH_PX = 320;
    private static final int CAMERA_BUFFER_COUNT = 3;
    // Another app (e.g. the camera app, to aim the phone) can take the camera; keep trying to get it back.
    private static final long CAMERA_RETRY_DELAY_MS = 5000;
    private static final double STATS_WINDOW_S = 5.0;
    private static final double MOTION_HOLD_S = 2.0;
    private static final double IDLE_INTERVAL_S = 1.0;
    private static final double NANOS_PER_S = 1e9;
    private static final double MILLIS_PER_S = 1e3;
    // Any texture name works: the texture is never drawn, it only receives the preview.
    private static final int OFFSCREEN_TEXTURE_NAME = 10;

    private final AtomicBoolean isDetectorBusy = new AtomicBoolean(false);
    private final MotionGate motionGate = new MotionGate(MOTION_HOLD_S, IDLE_INTERVAL_S);
    private Settings settings;
    private PowerManager.WakeLock wakeLock;
    private WifiManager.WifiLock wifiLock;
    private GatewayClient gateway;
    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private HandlerThread detectorThread;
    private Handler detectorHandler;
    // Camera state: touched on the camera thread only.
    private Camera camera;
    private SurfaceTexture previewTexture;
    private volatile int frameWidthPx;
    private volatile int frameHeightPx;
    // Detector state: touched on the detector thread only.
    private volatile Detector detector;
    private long frameCount;
    private long windowStartMs;
    private int windowFrames;
    private double windowInferenceS;

    @Override
    public void onCreate() {
        super.onCreate();
        startForeground(NOTIFICATION_ID, buildNotification());
        settings = Settings.load(this);
        Log.i(TAG, "starting, " + settings);
        try {
            gateway = new GatewayClient(settings.gatewayUrl, System.currentTimeMillis(), settings.modelName,
                    readLabels(settings.modelName + ".labels"));
        } catch (MalformedURLException e) {
            Log.e(TAG, "bad gateway URL " + settings.gatewayUrl, e);
            stopSelf();
            return;
        } catch (IOException e) {
            Log.e(TAG, "class names missing for " + settings.modelName, e);
            stopSelf();
            return;
        }
        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, TAG);
        wakeLock.acquire();
        // Without it, Wi-Fi power saving with the screen off holds back sends for seconds at a time.
        WifiManager wifiManager = (WifiManager) getApplicationContext().getSystemService(WIFI_SERVICE);
        wifiLock = wifiManager.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, TAG);
        wifiLock.acquire();
        detectorThread = new HandlerThread("detector");
        detectorThread.start();
        detectorHandler = new Handler(detectorThread.getLooper());
        detectorHandler.post(this::loadDetector);
        cameraThread = new HandlerThread("camera");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
        // Camera callbacks arrive on the thread that opened the camera, so open it there.
        cameraHandler.post(this::openCamera);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        if (cameraHandler != null) {
            cameraHandler.post(this::releaseCamera);
            cameraThread.quitSafely();
        }
        if (detectorHandler != null) {
            detectorHandler.post(this::closeDetector);
            detectorThread.quitSafely();
        }
        if (gateway != null) {
            gateway.close();
        }
        if (wakeLock != null) {
            wakeLock.release();
        }
        if (wifiLock != null) {
            wifiLock.release();
        }
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    /**
     * Build the notification that keeps this service in the foreground.
     *
     * @return the ongoing notification
     */
    private Notification buildNotification() {
        return new Notification.Builder(this)
                .setContentTitle("Fall watch")
                .setContentText("Watching for someone lying on the floor")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setOngoing(true)
                .build();
    }

    /**
     * Read the model's class names from the APK assets, one per line.
     *
     * @param assetPath path of the labels file
     * @return the names, index = class id
     * @throws IOException if the file is missing
     */
    private List<String> readLabels(String assetPath) throws IOException {
        List<String> names = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(getAssets().open(assetPath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    names.add(line.trim());
                }
            }
        }
        return names;
    }

    /** Load the model (detector thread). Stops the service if the model is missing. */
    private void loadDetector() {
        try {
            detector = new Detector(getAssets(), settings.modelName, settings.threadCount);
        } catch (IllegalStateException e) {
            Log.e(TAG, "model load failed", e);
            stopSelf();
            return;
        }
        windowStartMs = SystemClock.elapsedRealtime();
    }

    /** Free the model (detector thread). */
    private void closeDetector() {
        if (detector != null) {
            detector.close();
            detector = null;
        }
    }

    /** Open the back camera and start an off-screen preview that feeds the reused buffers (camera thread). */
    private void openCamera() {
        try {
            camera = Camera.open(0);
        } catch (RuntimeException e) {
            Log.e(TAG, "camera open failed, trying again", e);
            cameraHandler.postDelayed(this::openCamera, CAMERA_RETRY_DELAY_MS);
            return;
        }
        camera.setErrorCallback(this::onCameraError);
        Camera.Parameters params = camera.getParameters();
        Camera.Size size = pickPreviewSize(params.getSupportedPreviewSizes());
        frameWidthPx = size.width;
        frameHeightPx = size.height;
        params.setPreviewSize(frameWidthPx, frameHeightPx);
        params.setPreviewFormat(ImageFormat.NV21);
        // Fixed far focus: at 320 px everything past about half a metre is sharp, and focus never hunts
        // when people move (continuous focus went soft every time someone walked by).
        if (params.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_INFINITY)) {
            params.setFocusMode(Camera.Parameters.FOCUS_MODE_INFINITY);
        }
        camera.setParameters(params);
        previewTexture = new SurfaceTexture(OFFSCREEN_TEXTURE_NAME);
        try {
            camera.setPreviewTexture(previewTexture);
        } catch (IOException e) {
            Log.e(TAG, "preview texture failed, trying again", e);
            releaseCamera();
            cameraHandler.postDelayed(this::openCamera, CAMERA_RETRY_DELAY_MS);
            return;
        }
        int bufferBytes = frameWidthPx * frameHeightPx * ImageFormat.getBitsPerPixel(ImageFormat.NV21) / 8;
        for (int i = 0; i < CAMERA_BUFFER_COUNT; i++) {
            camera.addCallbackBuffer(new byte[bufferBytes]);
        }
        camera.setPreviewCallbackWithBuffer(this::onPreviewFrame);
        camera.startPreview();
        Log.i(TAG, String.format(Locale.ROOT, "camera open, preview %dx%d", frameWidthPx, frameHeightPx));
    }

    /**
     * The camera was taken away or its server died (camera thread). Free it and open it again later.
     *
     * @param error the camera's error code
     * @param source the camera that failed
     */
    private void onCameraError(int error, Camera source) {
        Log.w(TAG, "camera error " + error + ", opening it again");
        releaseCamera();
        cameraHandler.postDelayed(this::openCamera, CAMERA_RETRY_DELAY_MS);
    }

    /** Stop the preview and free the camera (camera thread). */
    private void releaseCamera() {
        if (camera != null) {
            camera.setPreviewCallbackWithBuffer(null);
            camera.stopPreview();
            camera.release();
            camera = null;
        }
        if (previewTexture != null) {
            previewTexture.release();
            previewTexture = null;
        }
    }

    /**
     * A new frame from the camera (camera thread). Hand it to the detector if it is free and the
     * motion gate lets it through, otherwise give the buffer straight back.
     *
     * @param nv21 the frame in NV21 layout, one of our reused buffers
     * @param source the camera that produced it
     */
    private void onPreviewFrame(byte[] nv21, Camera source) {
        boolean isWanted = !settings.isMotionGateEnabled
                || motionGate.shouldRun(nv21, frameWidthPx, frameHeightPx, SystemClock.elapsedRealtime());
        if (!isWanted || detector == null || !isDetectorBusy.compareAndSet(false, true)) {
            source.addCallbackBuffer(nv21);
            return;
        }
        detectorHandler.post(() -> processFrame(nv21));
    }

    /** Give a buffer back to the camera (camera thread). */
    private void returnBuffer(byte[] nv21) {
        if (camera != null) {
            camera.addCallbackBuffer(nv21);
        }
    }

    /**
     * Run the detector on one frame and queue the result (detector thread).
     *
     * @param nv21 the frame in NV21 layout; given back to the camera when done
     */
    private void processFrame(byte[] nv21) {
        try {
            if (detector == null) {
                return;
            }
            int width = frameWidthPx;
            int height = frameHeightPx;
            int rotation = settings.rotationDeg;
            long frameTimeMs = System.currentTimeMillis();
            long startNs = SystemClock.elapsedRealtimeNanos();
            float[] items = detector.detect(nv21, width, height, rotation, MIN_SCORE);
            double inferenceS = (SystemClock.elapsedRealtimeNanos() - startNs) / NANOS_PER_S;
            boolean isSwapped = rotation == 90 || rotation == 270;
            FrameResult result = new FrameResult(frameCount, frameTimeMs,
                    isSwapped ? height : width, isSwapped ? width : height, inferenceS, items);
            gateway.enqueue(result);
            if (settings.isPreviewEnabled && gateway.isPreviewSlotFree()) {
                gateway.offerPreview(frameCount, nv21.clone(), width, height, rotation);
            }
            frameCount++;
            updateStats(inferenceS, result.itemCount());
        } finally {
            isDetectorBusy.set(false);
            cameraHandler.post(() -> returnBuffer(nv21));
        }
    }

    /**
     * Track speed over a short window; every few seconds log it and send a health snapshot.
     *
     * @param inferenceS detector time for the frame just processed
     * @param itemCount items found in it
     */
    private void updateStats(double inferenceS, int itemCount) {
        windowFrames++;
        windowInferenceS += inferenceS;
        long nowMs = SystemClock.elapsedRealtime();
        double elapsedS = (nowMs - windowStartMs) / MILLIS_PER_S;
        if (elapsedS < STATS_WINDOW_S) {
            return;
        }
        double fps = windowFrames / elapsedS;
        double meanInferenceS = windowInferenceS / windowFrames;
        Health health = Health.sample(this, fps, meanInferenceS);
        gateway.updateHealth(health);
        Log.i(TAG, String.format(Locale.ROOT, "fps=%.1f inference_ms=%.0f items=%d battery_c=%.1f cpu_c=%s",
                fps, meanInferenceS * MILLIS_PER_S, itemCount, health.batteryTempC, health.cpuTempC));
        windowStartMs = nowMs;
        windowFrames = 0;
        windowInferenceS = 0.0;
    }

    /**
     * Pick the supported preview size whose width is closest to the target width.
     *
     * @param sizes the sizes the camera supports
     * @return the closest size
     */
    private static Camera.Size pickPreviewSize(List<Camera.Size> sizes) {
        Camera.Size best = sizes.get(0);
        for (Camera.Size size : sizes) {
            if (Math.abs(size.width - TARGET_PREVIEW_WIDTH_PX) < Math.abs(best.width - TARGET_PREVIEW_WIDTH_PX)) {
                best = size;
            }
        }
        return best;
    }
}
