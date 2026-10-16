"""Messages the phone sends to the gateway, checked and turned into dataclasses.

The phone POSTs one JSON batch to ``/frames`` at a time (see ``GatewayClient.java``). Times arrive
in milliseconds and are converted to seconds here; box coordinates are normalised to 0..1 of the
upright frame.
"""

import base64
import binascii
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

MILLIS_PER_S = 1000.0
VALUES_PER_BOX = 5
ROTATIONS_DEG = (0, 90, 180, 270)


class MessageError(ValueError):
    """Raised when a message from the phone is malformed."""


@dataclass(frozen=True)
class Box:
    """One person found by the phone. Coordinates are 0..1 of the upright frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass(frozen=True)
class Frame:
    """The detector result for one camera frame."""

    frame_id: int
    wall_time_s: float
    width_px: int
    height_px: int
    inference_s: float
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class Health:
    """How the phone is doing: heat, battery and speed."""

    battery_temp_c: float
    battery_level_pct: int
    is_charging: bool
    cpu_temp_c: float | None
    fps: float
    mean_inference_s: float
    uptime_s: float


@dataclass(frozen=True)
class Preview:
    """A camera picture as JPEG, not rotated. ``rotation_deg`` (clockwise) makes it upright."""

    frame_id: int
    rotation_deg: int
    jpeg: bytes


@dataclass(frozen=True)
class Batch:
    """One POST from the phone: frames oldest first, plus status and maybe a preview."""

    session_id: int
    frames: tuple[Frame, ...]
    queued_frames: int
    dropped_frames: int
    health: Health | None
    preview: Preview | None


def parse_batch(*, payload: object) -> Batch:
    """Check a decoded JSON batch from the phone and convert it.

    Args:
        payload: the JSON body, already decoded.

    Returns:
        The batch as dataclasses.

    Raises:
        MessageError: if a field is missing, has the wrong type or is out of range.
    """
    body = _as_mapping(value=payload, name="batch")
    health = body.get("health")
    preview = body.get("preview")
    return Batch(
        session_id=_get_int(obj=body, key="session_id"),
        frames=tuple(_parse_frames(value=body.get("frames"))),
        queued_frames=_get_int(obj=body, key="queued_frames"),
        dropped_frames=_get_int(obj=body, key="dropped_frames"),
        health=None if health is None else _parse_health(value=health),
        preview=None if preview is None else _parse_preview(value=preview),
    )


def _parse_frames(*, value: object) -> Iterator[Frame]:
    """Yield the frames of a batch.

    Args:
        value: the ``frames`` field.

    Returns:
        A generator of frames, oldest first.

    Raises:
        MessageError: if the field is not a list or a frame is malformed.
    """
    if not isinstance(value, list):
        raise MessageError("frames must be a list")
    for item in value:
        frame = _as_mapping(value=item, name="frame")
        boxes = frame.get("boxes")
        if not isinstance(boxes, list):
            raise MessageError("boxes must be a list")
        yield Frame(
            frame_id=_get_int(obj=frame, key="frame_id"),
            wall_time_s=_get_int(obj=frame, key="wall_time_ms") / MILLIS_PER_S,
            width_px=_get_int(obj=frame, key="width_px"),
            height_px=_get_int(obj=frame, key="height_px"),
            inference_s=_get_float(obj=frame, key="inference_s"),
            boxes=tuple(_parse_box(value=box) for box in boxes),
        )
    return None


def _parse_box(*, value: object) -> Box:
    """Convert ``[x1, y1, x2, y2, score]``.

    Args:
        value: one entry of a frame's ``boxes`` list.

    Returns:
        The box.

    Raises:
        MessageError: if it is not 5 numbers in 0..1 with x1 <= x2 and y1 <= y2.
    """
    if not isinstance(value, list) or len(value) != VALUES_PER_BOX:
        raise MessageError(f"a box must be a list of {VALUES_PER_BOX} numbers")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
        raise MessageError("box values must be numbers")
    x1, y1, x2, y2, score = (float(v) for v in value)
    if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2, score)):
        raise MessageError("box values must be in 0..1")
    if x1 > x2 or y1 > y2:
        raise MessageError("box corners are swapped")
    return Box(x1=x1, y1=y1, x2=x2, y2=y2, score=score)


def _parse_health(*, value: object) -> Health:
    """Convert the ``health`` field.

    Args:
        value: the ``health`` field.

    Returns:
        The health snapshot.

    Raises:
        MessageError: if a field is missing or has the wrong type.
    """
    health = _as_mapping(value=value, name="health")
    cpu_temp = health.get("cpu_temp_c")
    is_charging = health.get("is_charging")
    if not isinstance(is_charging, bool):
        raise MessageError("is_charging must be true or false")
    return Health(
        battery_temp_c=_get_float(obj=health, key="battery_temp_c"),
        battery_level_pct=_get_int(obj=health, key="battery_level_pct"),
        is_charging=is_charging,
        cpu_temp_c=None if cpu_temp is None else _get_float(obj=health, key="cpu_temp_c"),
        fps=_get_float(obj=health, key="fps"),
        mean_inference_s=_get_float(obj=health, key="mean_inference_s"),
        uptime_s=_get_float(obj=health, key="uptime_s"),
    )


def _parse_preview(*, value: object) -> Preview:
    """Convert the ``preview`` field and decode its JPEG.

    Args:
        value: the ``preview`` field.

    Returns:
        The preview.

    Raises:
        MessageError: if a field is missing, the rotation is not a right angle or the image is not base64.
    """
    preview = _as_mapping(value=value, name="preview")
    rotation_deg = _get_int(obj=preview, key="rotation_deg")
    if rotation_deg not in ROTATIONS_DEG:
        raise MessageError(f"rotation_deg must be one of {ROTATIONS_DEG}")
    encoded = preview.get("jpeg_b64")
    if not isinstance(encoded, str):
        raise MessageError("jpeg_b64 must be a string")
    try:
        jpeg = base64.b64decode(encoded, validate=True)
    except binascii.Error as e:
        raise MessageError("jpeg_b64 is not valid base64") from e
    return Preview(frame_id=_get_int(obj=preview, key="frame_id"), rotation_deg=rotation_deg, jpeg=jpeg)


def _as_mapping(*, value: object, name: str) -> Mapping[str, object]:
    """Check that a JSON value is an object.

    Args:
        value: the JSON value.
        name: what it is, for the error message.

    Returns:
        The same value, typed as a mapping.

    Raises:
        MessageError: if it is not a JSON object.
    """
    if not isinstance(value, dict):
        raise MessageError(f"{name} must be a JSON object")
    return value


def _get_int(*, obj: Mapping[str, object], key: str) -> int:
    """Read an integer field.

    Args:
        obj: the JSON object.
        key: the field name.

    Returns:
        The value.

    Raises:
        MessageError: if the field is missing or not an integer.
    """
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise MessageError(f"{key} must be an integer")
    return value


def _get_float(*, obj: Mapping[str, object], key: str) -> float:
    """Read a number field (JSON integers are accepted too).

    Args:
        obj: the JSON object.
        key: the field name.

    Returns:
        The value as a float.

    Raises:
        MessageError: if the field is missing or not a number.
    """
    value = obj.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise MessageError(f"{key} must be a number")
    return float(value)
