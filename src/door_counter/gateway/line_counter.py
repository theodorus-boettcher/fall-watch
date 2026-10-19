"""Count items per material as they cross a line across the belt, with Roboflow's trackers and supervision.

The tracker follows items from frame to frame (ByteTrack, matching boxes by overlap), and the line
counts a tracked item once when its centre crosses it, in either direction. The line is vertical, so
the belt should run left to right (or right to left) in the upright picture; the phone's rotation
setting makes that so.
"""

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from door_counter.gateway.messages import Frame

COORD_SCALE = 1000.0  # boxes arrive as 0..1 of the upright frame; tracker and line work in these units
PHONE_FPS_HZ = 5.0
LOST_AFTER_S = 2.0  # a track survives this long without being seen
MIN_TRACK_SCORE = 0.3  # the tiny model's scores are low (0.3 to 0.6); the tracker's defaults (0.7, 0.6) ignore most
DEFAULT_LINE_X = 0.5


class LineCounter:
    """Tracks items and counts, per class, the ones crossing a vertical line. Not thread-safe:
    ``LiveState`` calls it under its lock."""

    def __init__(self, *, line_x: float = DEFAULT_LINE_X) -> None:
        """Create a counter with nothing counted.

        Args:
            line_x: where the line is, as a fraction of the upright frame's width.

        Raises:
            ValueError: if the line is not inside the frame.
        """
        if not 0.0 < line_x < 1.0:
            raise ValueError(f"line_x must be between 0 and 1, got {line_x}")
        self._tracker = ByteTrackTracker(
            frame_rate=PHONE_FPS_HZ,
            lost_track_buffer=round(PHONE_FPS_HZ * LOST_AFTER_S),
            track_activation_threshold=MIN_TRACK_SCORE,
            high_conf_det_threshold=MIN_TRACK_SCORE,
        )
        x = line_x * COORD_SCALE
        self._line = sv.LineZone(
            start=sv.Point(x=x, y=0.0), end=sv.Point(x=x, y=COORD_SCALE), triggering_anchors=(sv.Position.CENTER,)
        )

    def update(self, *, frame: Frame) -> None:
        """Feed the next frame, in order.

        Args:
            frame: the phone's result for one camera frame.
        """
        if frame.items:
            detections = sv.Detections(
                xyxy=np.array([[i.box.x1, i.box.y1, i.box.x2, i.box.y2] for i in frame.items]) * COORD_SCALE,
                confidence=np.array([i.score for i in frame.items]),
                class_id=np.array([i.class_id for i in frame.items], dtype=int),
            )
        else:
            detections = sv.Detections.empty()
        # No timestamps: the phone's clock can jump back (no internet time), and the tracker would then stall.
        self._line.trigger(self._tracker.update(detections))
        return None

    def crossings(self, *, class_count: int) -> tuple[int, ...]:
        """Items counted so far.

        Args:
            class_count: number of classes the model has.

        Returns:
            Crossings per class, both directions together.
        """
        line = self._line
        return tuple(line.in_count_per_class.get(c, 0) + line.out_count_per_class.get(c, 0) for c in range(class_count))
