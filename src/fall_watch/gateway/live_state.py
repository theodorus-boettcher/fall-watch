"""The newest data from the phone, kept in memory for the live page."""

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from fall_watch.gateway.fall_watch import Alert, FallWatch, PersonState
from fall_watch.gateway.messages import Batch, Frame, Health, Preview
from fall_watch.yolo_pose import Rect

DEFAULT_HISTORY_FRAMES = 600


@dataclass(frozen=True)
class LiveSnapshot:
    """Everything the live page shows, taken at one moment."""

    session_id: int | None
    model: str | None
    class_names: tuple[str, ...]
    latest_frame: Frame | None
    preview: Preview | None
    preview_people: tuple[PersonState, ...]
    health: Health | None
    queued_frames: int
    dropped_frames: int
    last_batch_wall_time_s: float | None
    people: tuple[PersonState, ...]  # tracked people in the newest frame
    alerts: tuple[Alert, ...]  # since the gateway started, oldest first
    beds: tuple[Rect, ...]


class LiveState:
    """Thread-safe store of the newest batch data. The HTTP handler threads share one instance."""

    def __init__(self, *, history_frames: int = DEFAULT_HISTORY_FRAMES, beds: Sequence[Rect] = ()) -> None:
        """Create an empty store.

        Args:
            history_frames: how many recent frames to remember, so a preview can find its boxes.
            beds: bed zones (0..1 of the upright frame) where lying raises no alert.

        Raises:
            ValueError: if history_frames is not positive.
        """
        if history_frames < 1:
            raise ValueError("history_frames must be positive")
        self._history_frames = history_frames
        self._lock = threading.Lock()
        self._recent: OrderedDict[int, Frame] = OrderedDict()
        self._people: dict[int, tuple[PersonState, ...]] = {}  # per remembered frame, so a preview finds its own
        self._session_id: int | None = None
        self._model: str | None = None
        self._class_names: tuple[str, ...] = ()
        self._preview: Preview | None = None
        self._health: Health | None = None
        self._queued_frames = 0
        self._dropped_frames = 0
        self._last_batch_wall_time_s: float | None = None
        self._watch = FallWatch(beds=beds)
        return None

    def add_batch(self, *, batch: Batch, received_wall_time_s: float) -> None:
        """Take in one batch from the phone.

        A new session id (the phone app restarted) clears the remembered frames, because frame ids
        start again from 0.

        Args:
            batch: the parsed batch.
            received_wall_time_s: laptop clock when it arrived, seconds since 1970.
        """
        with self._lock:
            if batch.session_id != self._session_id:
                self._recent.clear()
                self._people.clear()
                self._preview = None
                self._session_id = batch.session_id
            self._model = batch.model
            self._class_names = batch.class_names
            for frame in batch.frames:
                if frame.frame_id not in self._recent:  # a re-sent batch must not move the tracker twice
                    self._watch.update(frame=frame, class_names=batch.class_names)
                    self._people[frame.frame_id] = self._watch.people()
                self._recent[frame.frame_id] = frame
            while len(self._recent) > self._history_frames:
                old_id, _ = self._recent.popitem(last=False)
                self._people.pop(old_id, None)
            if batch.preview is not None:
                self._preview = batch.preview
            if batch.health is not None:
                self._health = batch.health
            self._queued_frames = batch.queued_frames
            self._dropped_frames = batch.dropped_frames
            self._last_batch_wall_time_s = received_wall_time_s
        return None

    def snapshot(self) -> LiveSnapshot:
        """Copy out the current state.

        Returns:
            The snapshot. ``preview_people`` are the people of the preview's own frame, so they line
            up with the picture even when newer frames have arrived.
        """
        with self._lock:
            latest = next(reversed(self._recent.values()), None)
            return LiveSnapshot(
                session_id=self._session_id,
                model=self._model,
                class_names=self._class_names,
                latest_frame=latest,
                preview=self._preview,
                preview_people=() if self._preview is None else self._people.get(self._preview.frame_id, ()),
                health=self._health,
                queued_frames=self._queued_frames,
                dropped_frames=self._dropped_frames,
                last_batch_wall_time_s=self._last_batch_wall_time_s,
                people=self._watch.people(),
                alerts=self._watch.alerts(),
                beds=self._watch.beds,
            )

    def clear_alerts(self, *, now_wall_time_s: float) -> None:
        """End every open alert.

        Args:
            now_wall_time_s: laptop clock, used only before any frame has arrived.
        """
        with self._lock:
            last = self._watch.last_time_s
            self._watch.clear_alerts(now_s=now_wall_time_s if last is None else last)
        return None
