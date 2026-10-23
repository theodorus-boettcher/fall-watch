package com.doorcounter.app;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.util.Base64;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * Sends detector results to the laptop gateway on its own thread.
 *
 * <p>Every frame's people (class, score, box, keypoints) are queued and sent in order, in batches, so
 * the gateway can track people even after a network gap. While the gateway is unreachable the queue keeps growing up to
 * a limit, then the oldest frames are dropped (and counted). Preview pictures are different: only
 * the newest one is kept, and it is sent only when previews are switched on.
 */
final class GatewayClient {
    private static final String TAG = "DoorGateway";
    private static final int MAX_QUEUED_FRAMES = 3000;  // about 5 minutes at 10 frames per second
    private static final int MAX_BATCH_FRAMES = 200;
    private static final int TIMEOUT_MS = 2000;
    private static final long RETRY_DELAY_MS = 1000;
    private static final int PREVIEW_JPEG_QUALITY = 50;

    /** The newest camera frame waiting to be sent as a preview picture. */
    private static final class Preview {
        final long frameId;
        final byte[] nv21;
        final int widthPx;
        final int heightPx;
        final int rotationDeg;

        Preview(long frameId, byte[] nv21, int widthPx, int heightPx, int rotationDeg) {
            this.frameId = frameId;
            this.nv21 = nv21;
            this.widthPx = widthPx;
            this.heightPx = heightPx;
            this.rotationDeg = rotationDeg;
        }
    }

    private final Object lock = new Object();
    private final ArrayDeque<FrameResult> queue = new ArrayDeque<>();
    private final URL framesUrl;
    private final long sessionId;
    private final String modelName;
    private final List<String> classNames;
    private final Thread thread;
    private Preview pendingPreview;
    private Health latestHealth;
    private long droppedFrames;
    private boolean isRunning = true;
    private boolean isReachable = true;

    /**
     * Start the sender thread.
     *
     * @param gatewayUrl base URL of the gateway, e.g. {@code http://127.0.0.1:8000}
     * @param sessionId identifies this service run, so the gateway can tell restarts apart
     * @param modelName the model in use, sent with every batch
     * @param classNames the model's class names, index = class id
     * @throws MalformedURLException if the URL is not valid
     */
    GatewayClient(String gatewayUrl, long sessionId, String modelName, List<String> classNames)
            throws MalformedURLException {
        this.framesUrl = new URL(gatewayUrl + "/frames");
        this.sessionId = sessionId;
        this.modelName = modelName;
        this.classNames = classNames;
        this.thread = new Thread(this::run, "gateway");
        thread.start();
    }

    /**
     * Queue one frame's result for sending.
     *
     * @param result the detector result
     */
    void enqueue(FrameResult result) {
        synchronized (lock) {
            if (queue.size() >= MAX_QUEUED_FRAMES) {
                queue.pollFirst();
                droppedFrames++;
            }
            queue.addLast(result);
            lock.notifyAll();
        }
    }

    /** @return true when no preview is waiting, so a new one would be sent soon */
    boolean isPreviewSlotFree() {
        synchronized (lock) {
            return pendingPreview == null;
        }
    }

    /**
     * Offer a camera frame as the next preview picture. Replaces any preview not sent yet.
     *
     * @param frameId the frame's id, matching its queued result
     * @param nv21 a copy of the frame the caller will not touch again
     * @param widthPx frame width as delivered by the camera
     * @param heightPx frame height as delivered by the camera
     * @param rotationDeg clockwise rotation that makes it upright
     */
    void offerPreview(long frameId, byte[] nv21, int widthPx, int heightPx, int rotationDeg) {
        synchronized (lock) {
            pendingPreview = new Preview(frameId, nv21, widthPx, heightPx, rotationDeg);
            lock.notifyAll();
        }
    }

    /**
     * Replace the health snapshot sent with the next batch.
     *
     * @param health the newest snapshot
     */
    void updateHealth(Health health) {
        synchronized (lock) {
            latestHealth = health;
        }
    }

    /** Stop the sender thread. Results still queued are lost. */
    void close() {
        synchronized (lock) {
            isRunning = false;
            lock.notifyAll();
        }
        thread.interrupt();
    }

    /** Sender loop: wait for work, send one batch, drop it from the queue once the gateway took it. */
    private void run() {
        while (true) {
            List<FrameResult> batch = new ArrayList<>();
            Preview preview;
            Health health;
            long dropped;
            int queued;
            synchronized (lock) {
                try {
                    while (isRunning && queue.isEmpty() && pendingPreview == null) {
                        lock.wait();
                    }
                } catch (InterruptedException e) {
                    return;
                }
                if (!isRunning) {
                    return;
                }
                for (FrameResult result : queue) {
                    if (batch.size() == MAX_BATCH_FRAMES) {
                        break;
                    }
                    batch.add(result);
                }
                preview = pendingPreview;
                pendingPreview = null;
                health = latestHealth;
                dropped = droppedFrames;
                queued = queue.size();
            }
            boolean isTaken;
            try {
                isTaken = post(buildJson(batch, preview, health, dropped, queued));
            } catch (JSONException e) {
                isTaken = false;  // a value org.json cannot write, such as NaN: sending it again won't help
            } catch (IOException e) {
                if (isReachable) {
                    Log.w(TAG, "gateway unreachable, queueing: " + e);
                    isReachable = false;
                }
                try {
                    Thread.sleep(RETRY_DELAY_MS);
                } catch (InterruptedException interrupted) {
                    return;
                }
                continue;
            }
            if (!isReachable) {
                Log.i(TAG, "gateway reachable again, " + queued + " frames queued");
                isReachable = true;
            }
            if (!isTaken) {
                // Dropped like a sent batch, or the queue would stay stuck on it.
                Log.w(TAG, "gateway refused a batch, dropping " + batch.size() + " frames");
            }
            if (!batch.isEmpty()) {
                long lastSentId = batch.get(batch.size() - 1).frameId;
                synchronized (lock) {
                    // Remove by id, not by count: the queue may have dropped old frames meanwhile.
                    while (!queue.isEmpty() && queue.peekFirst().frameId <= lastSentId) {
                        queue.pollFirst();
                    }
                }
            }
        }
    }

    /**
     * Build the request body.
     *
     * @param batch frames to send, oldest first
     * @param preview a preview to include, or null
     * @param health the latest health snapshot, or null before the first one
     * @param dropped frames dropped so far because the queue was full
     * @param queued frames in the queue, this batch included
     * @return the JSON body
     * @throws JSONException never in practice; org.json declares it
     */
    private JSONObject buildJson(List<FrameResult> batch, Preview preview, Health health, long dropped, int queued)
            throws JSONException {
        JSONArray frames = new JSONArray();
        for (FrameResult result : batch) {
            JSONArray items = new JSONArray();
            for (int i = 0; i < result.items.length; i += Detector.VALUES_PER_ITEM) {
                items.put(itemJson(result.items, i));
            }
            frames.put(new JSONObject()
                    .put("frame_id", result.frameId)
                    .put("wall_time_ms", result.wallTimeMs)
                    .put("width_px", result.uprightWidthPx)
                    .put("height_px", result.uprightHeightPx)
                    .put("inference_s", result.inferenceS)
                    .put("items", items));
        }
        JSONObject body = new JSONObject()
                .put("session_id", sessionId)
                .put("model", modelName)
                .put("class_names", new JSONArray(classNames))
                .put("frames", frames)
                .put("queued_frames", queued)
                .put("dropped_frames", dropped);
        if (health != null) {
            body.put("health", health.toJson());
        }
        if (preview != null) {
            body.put("preview", new JSONObject()
                    .put("frame_id", preview.frameId)
                    .put("rotation_deg", preview.rotationDeg)
                    .put("jpeg_b64", Base64.encodeToString(toJpeg(preview), Base64.NO_WRAP)));
        }
        return body;
    }

    /**
     * One person as JSON: class, score, box, and x, y, visibility of each keypoint.
     *
     * @param records the frame's person records
     * @param start index of this person's first value
     * @return the item
     * @throws JSONException never in practice; org.json declares it
     */
    private static JSONObject itemJson(float[] records, int start) throws JSONException {
        JSONArray keypoints = new JSONArray();
        for (int k = 0; k < 3 * Detector.KEYPOINT_COUNT; k++) {
            keypoints.put((double) records[start + Detector.KEYPOINTS + k]);
        }
        return new JSONObject()
                .put("class_id", (int) records[start + Detector.CLASS_ID])
                .put("score", (double) records[start + Detector.SCORE])
                .put("box", rectJson(records, start + Detector.BOX))
                .put("keypoints", keypoints);
    }

    /**
     * Four values as a JSON array: x1, y1, x2, y2.
     *
     * @param values the record
     * @param start index of x1
     * @return the array
     * @throws JSONException never in practice; org.json declares it
     */
    private static JSONArray rectJson(float[] values, int start) throws JSONException {
        JSONArray rect = new JSONArray();
        for (int j = 0; j < 4; j++) {
            rect.put((double) values[start + j]);
        }
        return rect;
    }

    /**
     * Compress a preview frame to JPEG.
     *
     * @param preview the frame
     * @return JPEG bytes, not rotated
     */
    private static byte[] toJpeg(Preview preview) {
        YuvImage image = new YuvImage(preview.nv21, ImageFormat.NV21, preview.widthPx, preview.heightPx, null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        image.compressToJpeg(new Rect(0, 0, preview.widthPx, preview.heightPx), PREVIEW_JPEG_QUALITY, out);
        return out.toByteArray();
    }

    /**
     * POST a JSON body to the gateway.
     *
     * @param body the request body
     * @return true if the gateway took it, false if it refused it (4xx: sending it again won't help)
     * @throws IOException if the gateway cannot be reached or answers anything else
     */
    private boolean post(JSONObject body) throws IOException {
        byte[] bytes = body.toString().getBytes(StandardCharsets.UTF_8);
        HttpURLConnection connection = (HttpURLConnection) framesUrl.openConnection();
        try {
            connection.setConnectTimeout(TIMEOUT_MS);
            connection.setReadTimeout(TIMEOUT_MS);
            connection.setDoOutput(true);
            connection.setFixedLengthStreamingMode(bytes.length);
            connection.setRequestProperty("Content-Type", "application/json");
            try (OutputStream out = connection.getOutputStream()) {
                out.write(bytes);
            }
            int status = connection.getResponseCode();
            if (status / 100 == 4) {
                return false;
            }
            if (status / 100 != 2) {
                throw new IOException("gateway answered HTTP " + status);
            }
            return true;
        } finally {
            connection.disconnect();
        }
    }
}
