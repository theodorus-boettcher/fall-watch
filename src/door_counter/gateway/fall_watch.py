"""Watch people's postures over time and raise an alert when someone lies outside the bed.

Roboflow's trackers (ByteTrack) follow each person from frame to frame. The phone's model says per
person whether they are ``lying`` or ``not lying``; the ready-made one-class pose model has no posture,
so then the keypoints decide (``is_lying``). A tracked person who lies with their box centre outside
every bed zone for ``ALERT_AFTER_S`` raises an alert. The alert ends when the same person is seen not
lying, or when staff clear it on the live page (a person lying still may stop being found).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from door_counter.gateway.messages import Frame, Item
from door_counter.yolo_pose import Keypoint, Rect

LYING_CLASS = "lying"
ALERT_AFTER_S = 5.0  # short for the demo; a care home would pick something like 30 s
TORSO_LYING_DEG = 60.0  # torso further than this from vertical counts as lying
MIN_VISIBILITY = 0.5
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12  # COCO keypoint order
PHONE_FPS_HZ = 5.0
LOST_AFTER_S = 2.0  # a track survives this long without being seen
MIN_TRACK_SCORE = 0.3
FORGET_AFTER_S = 30.0  # drop a person's timer when they have not been seen for this long


@dataclass(frozen=True)
class Alert:
    """Someone lying outside the bed. Times are the phone's clock, seconds since 1970."""

    track_id: int
    lying_since_s: float
    raised_s: float
    ended_s: float | None = None


@dataclass(frozen=True)
class PersonState:
    """One person in the newest frame, as the live page shows them."""

    track_id: int
    item: Item
    is_lying: bool
    is_in_bed: bool
    lying_for_s: float  # 0 while not lying


def _middle(*, a: Keypoint, b: Keypoint) -> tuple[float, float] | None:
    """Middle of two keypoints, or the one that is visible; None if neither is."""
    seen = [k for k in (a, b) if k.visibility >= MIN_VISIBILITY]
    if not seen:
        return None
    return sum(k.x for k in seen) / len(seen), sum(k.y for k in seen) / len(seen)


def is_lying(*, item: Item, class_names: Sequence[str], width_px: int, height_px: int) -> bool:
    """Whether a person lies, from the model's posture class or, for a one-class model, from the body.

    Args:
        item: the person.
        class_names: the model's classes.
        width_px: frame width, to measure angles in true proportions.
        height_px: frame height.

    Returns:
        True when lying: the class says so, or the torso (mid-shoulders to mid-hips) is further than
        ``TORSO_LYING_DEG`` from vertical, or, with no torso visible, the box is wider than tall.
    """
    if len(class_names) > 1:
        return class_names[item.class_id] == LYING_CLASS
    k = item.keypoints
    shoulders = _middle(a=k[LEFT_SHOULDER], b=k[RIGHT_SHOULDER])
    hips = _middle(a=k[LEFT_HIP], b=k[RIGHT_HIP])
    if shoulders is not None and hips is not None and shoulders != hips:
        dx = (shoulders[0] - hips[0]) * width_px
        dy = (shoulders[1] - hips[1]) * height_px
        return math.degrees(math.atan2(abs(dx), abs(dy))) > TORSO_LYING_DEG
    return (item.box.x2 - item.box.x1) * width_px > (item.box.y2 - item.box.y1) * height_px


def is_in_bed(*, box: Rect, beds: Sequence[Rect]) -> bool:
    """Whether the box centre lies inside one of the bed zones (all 0..1 of the upright frame)."""
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    return any(b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2 for b in beds)


class FallWatch:
    """Tracks people and keeps their lying timers and alerts. Not thread-safe: ``LiveState`` calls it
    under its lock."""

    def __init__(self, *, beds: Sequence[Rect] = (), alert_after_s: float = ALERT_AFTER_S) -> None:
        """Create a watch with nobody seen yet.

        Args:
            beds: bed zones, 0..1 of the upright frame; lying there is fine.
            alert_after_s: how long someone must lie outside a bed before the alert.

        Raises:
            ValueError: if alert_after_s is not positive.
        """
        if alert_after_s <= 0:
            raise ValueError("alert_after_s must be positive")
        self.beds = tuple(beds)
        self._alert_after_s = alert_after_s
        self._tracker = ByteTrackTracker(
            frame_rate=PHONE_FPS_HZ,
            lost_track_buffer=round(PHONE_FPS_HZ * LOST_AFTER_S),
            track_activation_threshold=MIN_TRACK_SCORE,
            high_conf_det_threshold=MIN_TRACK_SCORE,
        )
        self._lying_since_s: dict[int, float] = {}
        self._last_seen_s: dict[int, float] = {}
        self._open: dict[int, int] = {}  # track id -> index of its open alert
        self._alerts: list[Alert] = []
        self._people: tuple[PersonState, ...] = ()
        self._last_time_s: float | None = None

    def update(self, *, frame: Frame, class_names: Sequence[str]) -> None:
        """Feed the next frame, in order.

        Args:
            frame: the phone's result for one camera frame.
            class_names: the model's classes.
        """
        now = frame.wall_time_s
        if self._last_time_s is not None and now < self._last_time_s:
            self._lying_since_s.clear()  # the phone's clock jumped back (no internet time): restart the timers
        self._last_time_s = now
        items = frame.items
        if items:
            detections = sv.Detections(
                xyxy=np.array([[i.box.x1, i.box.y1, i.box.x2, i.box.y2] for i in items]),
                confidence=np.array([i.score for i in items]),
                data={"index": np.arange(len(items))},
            )
        else:
            detections = sv.Detections.empty()
        # No timestamps for the tracker either: it would stall after a clock jump.
        tracked = self._tracker.update(detections)
        track_ids = [] if tracked.tracker_id is None else tracked.tracker_id.tolist()
        people = []
        for track_id, index in zip(track_ids, tracked.data.get("index", []), strict=True):
            if track_id < 0:
                continue  # not confirmed yet
            item = items[int(index)]
            lying = is_lying(item=item, class_names=class_names, width_px=frame.width_px, height_px=frame.height_px)
            in_bed = is_in_bed(box=item.box, beds=self.beds)
            people.append(self._step(track_id=int(track_id), item=item, lying=lying, in_bed=in_bed, now=now))
        self._people = tuple(people)
        for track_id in [t for t, seen in self._last_seen_s.items() if now - seen > FORGET_AFTER_S]:
            del self._last_seen_s[track_id]
            self._lying_since_s.pop(track_id, None)
        return None

    def _step(self, *, track_id: int, item: Item, lying: bool, in_bed: bool, now: float) -> PersonState:
        """Update one person's timer and alert.

        Args:
            track_id: the tracker's id for this person.
            item: what the phone found.
            lying: whether they lie.
            in_bed: whether their box centre is in a bed zone.
            now: the frame's time.

        Returns:
            Their state for the live page.
        """
        self._last_seen_s[track_id] = now
        if not lying or in_bed:
            self._lying_since_s.pop(track_id, None)
            if track_id in self._open:
                i = self._open.pop(track_id)
                self._alerts[i] = replace(self._alerts[i], ended_s=now)
            return PersonState(track_id=track_id, item=item, is_lying=lying, is_in_bed=in_bed, lying_for_s=0.0)
        since = self._lying_since_s.setdefault(track_id, now)
        if now - since >= self._alert_after_s and track_id not in self._open:
            self._open[track_id] = len(self._alerts)
            self._alerts.append(Alert(track_id=track_id, lying_since_s=since, raised_s=now))
        return PersonState(track_id=track_id, item=item, is_lying=True, is_in_bed=False, lying_for_s=now - since)

    def clear_alerts(self, *, now_s: float) -> None:
        """End every open alert (staff have seen to it).

        Args:
            now_s: time to record as the end, phone clock if known.
        """
        for track_id, i in self._open.items():
            self._alerts[i] = replace(self._alerts[i], ended_s=now_s)
            self._lying_since_s.pop(track_id, None)
        self._open.clear()
        return None

    @property
    def last_time_s(self) -> float | None:
        """Time of the newest frame (phone clock), None before the first."""
        return self._last_time_s

    def people(self) -> tuple[PersonState, ...]:
        """People in the newest frame."""
        return self._people

    def alerts(self) -> tuple[Alert, ...]:
        """All alerts so far, oldest first; open ones have no end time."""
        return tuple(self._alerts)
