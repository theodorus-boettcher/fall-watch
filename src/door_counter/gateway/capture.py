"""Save the phone's preview pictures into named clips, as new training and test data.

One clip holds one thing: training pictures of one kind (``room-up-1``, ``room-lying-2``), or a test scene
(``test-fall-3``). The pictures are saved upright, the way the model sees them.
"""

import re
import threading
from pathlib import Path

import cv2
import numpy as np

from door_counter.gateway.messages import Preview

CLIP_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
JPEG_QUALITY = 95
ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class CaptureError(ValueError):
    """Raised for a bad clip name or a picture that cannot be decoded."""


def upright_jpeg(*, preview: Preview) -> bytes:
    """Turn a preview picture upright.

    Args:
        preview: the picture as the camera delivered it.

    Returns:
        JPEG bytes, unchanged when no rotation is needed.

    Raises:
        CaptureError: if the JPEG cannot be decoded or encoded.
    """
    if preview.rotation_deg == 0:
        return preview.jpeg
    pixels = cv2.imdecode(np.frombuffer(preview.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise CaptureError(f"preview of frame {preview.frame_id} is not a readable JPEG")
    is_ok, encoded = cv2.imencode(".jpg", cv2.rotate(pixels, ROTATE_CODES[preview.rotation_deg]),
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not is_ok:
        raise CaptureError(f"could not encode frame {preview.frame_id}")
    return bytes(encoded)


class Capture:
    """Writes previews to ``<root>/<clip>/`` while a clip is running. Thread-safe."""

    def __init__(self, *, root: Path) -> None:
        """Create a stopped capture.

        Args:
            root: folder that gets one subfolder per clip.
        """
        self._root = root
        self._lock = threading.Lock()
        self._clip: str | None = None
        self._saved = 0
        return None

    @property
    def clip(self) -> str | None:
        """The running clip, or None when stopped."""
        with self._lock:
            return self._clip

    @property
    def saved(self) -> int:
        """Pictures saved in the running (or last) clip."""
        with self._lock:
            return self._saved

    def start(self, *, clip: str) -> None:
        """Start saving into a clip; an existing clip gets more pictures added.

        Args:
            clip: lower-case letters, digits, ``-`` and ``_``, at most 48 characters.

        Raises:
            CaptureError: if the name is not allowed (it becomes a folder name).
        """
        if CLIP_NAME_PATTERN.fullmatch(clip) is None:
            raise CaptureError(f"clip name {clip!r} must match {CLIP_NAME_PATTERN.pattern}")
        (self._root / clip).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._clip = clip
            self._saved = 0
        return None

    def stop(self) -> None:
        """Stop saving."""
        with self._lock:
            self._clip = None
        return None

    def save(self, *, preview: Preview, session_id: int) -> Path | None:
        """Save a preview into the running clip.

        Args:
            preview: the picture.
            session_id: the phone's session, so file names stay unique when the app restarts.

        Returns:
            The file written, or None when no clip is running.
        """
        clip = self.clip
        if clip is None:
            return None
        path = self._root / clip / f"{session_id}_{preview.frame_id:08d}.jpg"
        path.write_bytes(upright_jpeg(preview=preview))
        with self._lock:
            self._saved += 1
        return path
