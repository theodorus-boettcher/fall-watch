package com.doorcounter.app;

/**
 * Decides whether a camera frame is worth running the model on.
 *
 * <p>An empty or still belt does not need 5 detections a second. Each frame is shrunk to a tiny
 * grid of brightness samples and compared with the previous one. While something moves (and for a
 * short hold time after), every frame may run; otherwise only one frame per idle interval runs, so
 * a newly placed item still shows up within about a second. This saves heat and battery.
 *
 * <p>Not thread safe: call it from the camera thread only.
 */
final class MotionGate {
    private static final int SAMPLE_COLUMNS = 40;
    private static final int SAMPLE_ROWS = 30;
    /** Brightness change (0..255) that counts a sample as changed. */
    private static final int SAMPLE_CHANGE = 12;
    /** Share of changed samples that counts as motion. */
    private static final double CHANGED_FRACTION = 0.01;
    private static final double MILLIS_PER_S = 1000.0;

    private final long holdMs;
    private final long idleIntervalMs;
    private int[] previous;
    private long lastMotionMs = Long.MIN_VALUE / 2;
    private long lastRunMs = Long.MIN_VALUE / 2;

    /**
     * @param holdS keep running every frame this long after the last motion
     * @param idleIntervalS when nothing moves, run once per this interval
     */
    MotionGate(double holdS, double idleIntervalS) {
        this.holdMs = Math.round(holdS * MILLIS_PER_S);
        this.idleIntervalMs = Math.round(idleIntervalS * MILLIS_PER_S);
    }

    /**
     * Look at a frame and decide whether the model should run on it.
     *
     * @param nv21 the frame in NV21 layout (only the brightness plane is read)
     * @param widthPx frame width
     * @param heightPx frame height
     * @param nowMs a monotonic clock in milliseconds
     * @return true if the model should run on this frame
     */
    boolean shouldRun(byte[] nv21, int widthPx, int heightPx, long nowMs) {
        int[] samples = sample(nv21, widthPx, heightPx);
        if (previous != null && isMoving(previous, samples)) {
            lastMotionMs = nowMs;
        }
        previous = samples;
        boolean isActive = nowMs - lastMotionMs <= holdMs;
        boolean isDue = nowMs - lastRunMs >= idleIntervalMs;
        if (isActive || isDue) {
            lastRunMs = nowMs;
            return true;
        }
        return false;
    }

    /** @return true if motion was seen within the hold time of the last call */
    boolean isActive(long nowMs) {
        return nowMs - lastMotionMs <= holdMs;
    }

    /**
     * Brightness samples on an even grid.
     *
     * @param nv21 the frame
     * @param widthPx frame width
     * @param heightPx frame height
     * @return SAMPLE_ROWS x SAMPLE_COLUMNS values in 0..255
     */
    private static int[] sample(byte[] nv21, int widthPx, int heightPx) {
        int[] samples = new int[SAMPLE_ROWS * SAMPLE_COLUMNS];
        for (int r = 0; r < SAMPLE_ROWS; r++) {
            int y = (r * heightPx + heightPx / 2) / SAMPLE_ROWS;
            for (int c = 0; c < SAMPLE_COLUMNS; c++) {
                int x = (c * widthPx + widthPx / 2) / SAMPLE_COLUMNS;
                samples[r * SAMPLE_COLUMNS + c] = nv21[y * widthPx + x] & 0xFF;
            }
        }
        return samples;
    }

    /**
     * @param before samples of the previous frame
     * @param after samples of this frame
     * @return true if enough samples changed by enough
     */
    private static boolean isMoving(int[] before, int[] after) {
        int changed = 0;
        for (int i = 0; i < after.length; i++) {
            if (Math.abs(after[i] - before[i]) >= SAMPLE_CHANGE) {
                changed++;
            }
        }
        return changed >= CHANGED_FRACTION * after.length;
    }
}
