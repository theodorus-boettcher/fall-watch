package com.doorcounter.app;

/** What the detector found in one frame. Immutable. */
final class FrameResult {
    /** Counts up from 0 for each service start. */
    final long frameId;
    /** Phone wall clock when the frame arrived, milliseconds since 1970. */
    final long wallTimeMs;
    /** Size of the upright frame, after rotation. */
    final int uprightWidthPx;
    final int uprightHeightPx;
    /** Time spent in the detector for this frame. */
    final double inferenceS;
    /** {@link Detector#VALUES_PER_ITEM} values per item (see {@link Detector#detect}). */
    final float[] items;
    /** Mask runs of all items (see {@link Detector#lastRuns}). */
    final int[] runs;

    FrameResult(long frameId, long wallTimeMs, int uprightWidthPx, int uprightHeightPx, double inferenceS,
                float[] items, int[] runs) {
        this.frameId = frameId;
        this.wallTimeMs = wallTimeMs;
        this.uprightWidthPx = uprightWidthPx;
        this.uprightHeightPx = uprightHeightPx;
        this.inferenceS = inferenceS;
        this.items = items;
        this.runs = runs;
    }

    /** @return the number of items found */
    int itemCount() {
        return items.length / Detector.VALUES_PER_ITEM;
    }
}
