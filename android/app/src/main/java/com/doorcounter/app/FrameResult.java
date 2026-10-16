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
    /** x1, y1, x2, y2, score per person, normalised to 0..1 (see {@link Detector#detect}). */
    final float[] boxes;

    FrameResult(long frameId, long wallTimeMs, int uprightWidthPx, int uprightHeightPx, double inferenceS,
                float[] boxes) {
        this.frameId = frameId;
        this.wallTimeMs = wallTimeMs;
        this.uprightWidthPx = uprightWidthPx;
        this.uprightHeightPx = uprightHeightPx;
        this.inferenceS = inferenceS;
        this.boxes = boxes;
    }

    /** @return the number of people found */
    int personCount() {
        return boxes.length / Detector.VALUES_PER_BOX;
    }
}
