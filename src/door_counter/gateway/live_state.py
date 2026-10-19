"""The newest data from the phone, kept in memory for the live page."""

import threading
from collections import OrderedDict
from dataclasses import dataclass

from door_counter.gateway.line_counter import LineCounter
from door_counter.gateway.messages import Batch, Frame, Health, Item, Preview

DEFAULT_HISTORY_FRAMES = 600


@dataclass(frozen=True)
class LiveSnapshot:
    """Everything the live page shows, taken at one moment."""

    session_id: int | None
    model: str | None
    class_names: tuple[str, ...]
    latest_frame: Frame | None
    preview: Preview | None
    preview_items: tuple[Item, ...]
    health: Health | None
    queued_frames: int
    dropped_frames: int
    last_batch_wall_time_s: float | None
    crossings: tuple[int, ...]  # items per class that crossed the counting line since the gateway started


class LiveState:
    """Thread-safe store of the newest batch data. The HTTP handler threads share one instance."""

    def __init__(self, *, history_frames: int = DEFAULT_HISTORY_FRAMES) -> None:
        """Create an empty store.

        Args:
            history_frames: how many recent frames to remember, so a preview can find its boxes.

        Raises:
            ValueError: if history_frames is not positive.
        """
        if history_frames < 1:
            raise ValueError("history_frames must be positive")
        self._history_frames = history_frames
        self._lock = threading.Lock()
        self._recent: OrderedDict[int, Frame] = OrderedDict()
        self._session_id: int | None = None
        self._model: str | None = None
        self._class_names: tuple[str, ...] = ()
        self._preview: Preview | None = None
        self._health: Health | None = None
        self._queued_frames = 0
        self._dropped_frames = 0
        self._last_batch_wall_time_s: float | None = None
        self._counter = LineCounter()

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
                self._preview = None
                self._session_id = batch.session_id
            self._model = batch.model
            self._class_names = batch.class_names
            for frame in batch.frames:
                if frame.frame_id not in self._recent:  # a re-sent batch must not move the tracker twice
                    self._counter.update(frame=frame)
                self._recent[frame.frame_id] = frame
            while len(self._recent) > self._history_frames:
                self._recent.popitem(last=False)
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
            The snapshot. ``preview_items`` are the items of the preview's own frame, so they line
            up with the picture even when newer frames have arrived.
        """
        with self._lock:
            latest = next(reversed(self._recent.values()), None)
            preview_frame = None if self._preview is None else self._recent.get(self._preview.frame_id)
            return LiveSnapshot(
                session_id=self._session_id,
                model=self._model,
                class_names=self._class_names,
                latest_frame=latest,
                preview=self._preview,
                preview_items=() if preview_frame is None else preview_frame.items,
                health=self._health,
                queued_frames=self._queued_frames,
                dropped_frames=self._dropped_frames,
                last_batch_wall_time_s=self._last_batch_wall_time_s,
                crossings=self._counter.crossings(class_count=len(self._class_names)),
            )
