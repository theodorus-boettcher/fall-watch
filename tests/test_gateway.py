"""Tests for the gateway's message parsing and live state."""

import base64
import copy

import pytest

from door_counter.gateway.live_state import LiveState
from door_counter.gateway.messages import MessageError, parse_batch
from door_counter.gateway.server import snapshot_to_json


def make_payload(*, session_id: int = 7, first_frame_id: int = 0, frame_count: int = 2) -> dict[str, object]:
    """Build a batch like the phone sends, with a preview of the last frame.

    Args:
        session_id: the phone's session id.
        first_frame_id: id of the first frame in the batch.
        frame_count: how many frames, one person each.

    Returns:
        The decoded JSON body.
    """
    frames = [
        {
            "frame_id": first_frame_id + i,
            "wall_time_ms": 1_760_000_000_000 + 100 * i,
            "width_px": 640,
            "height_px": 480,
            "inference_s": 0.12,
            "boxes": [[0.1, 0.2, 0.3 + 0.01 * i, 0.9, 0.88]],
        }
        for i in range(frame_count)
    ]
    return {
        "session_id": session_id,
        "frames": frames,
        "queued_frames": frame_count,
        "dropped_frames": 0,
        "health": {
            "battery_temp_c": 27.7,
            "battery_level_pct": 94,
            "is_charging": True,
            "cpu_temp_c": 74.0,
            "fps": 6.7,
            "mean_inference_s": 0.124,
            "uptime_s": 1000,
        },
        "preview": {
            "frame_id": first_frame_id + frame_count - 1,
            "rotation_deg": 180,
            "jpeg_b64": base64.b64encode(b"\xff\xd8fake").decode(),
        },
    }


def test_parse_batch_converts_units_and_types() -> None:
    batch = parse_batch(payload=make_payload())
    assert batch.session_id == 7
    assert len(batch.frames) == 2
    assert batch.frames[0].wall_time_s == pytest.approx(1_760_000_000.0)
    assert batch.frames[1].boxes[0].x2 == pytest.approx(0.31)
    assert batch.health is not None and batch.health.uptime_s == pytest.approx(1000.0)
    assert batch.preview is not None and batch.preview.jpeg == b"\xff\xd8fake"


def test_parse_batch_accepts_missing_optional_parts() -> None:
    payload = make_payload()
    del payload["health"]
    del payload["preview"]
    batch = parse_batch(payload=payload)
    assert batch.health is None
    assert batch.preview is None


@pytest.mark.parametrize(
    "box",
    [
        [0.1, 0.2, 0.3, 0.9],  # too short
        [0.1, 0.2, 0.3, 1.5, 0.9],  # outside 0..1
        [0.5, 0.2, 0.3, 0.9, 0.9],  # x1 > x2
        [0.1, 0.2, 0.3, 0.9, True],  # bool is not a number here
    ],
)
def test_parse_batch_rejects_bad_boxes(box: list[object]) -> None:
    payload = make_payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    frames[0]["boxes"] = [box]
    with pytest.raises(MessageError):
        parse_batch(payload=payload)


def test_parse_batch_rejects_bad_rotation_and_missing_fields() -> None:
    payload = make_payload()
    preview = payload["preview"]
    assert isinstance(preview, dict)
    preview["rotation_deg"] = 45
    with pytest.raises(MessageError):
        parse_batch(payload=payload)
    payload = make_payload()
    del payload["session_id"]
    with pytest.raises(MessageError):
        parse_batch(payload=payload)
    with pytest.raises(MessageError):
        parse_batch(payload=["not", "an", "object"])


def test_preview_gets_the_boxes_of_its_own_frame() -> None:
    state = LiveState()
    state.add_batch(batch=parse_batch(payload=make_payload(frame_count=3)), received_wall_time_s=10.0)
    newer = make_payload(first_frame_id=3, frame_count=2)
    del newer["preview"]  # newer frames arrive without a new preview
    state.add_batch(batch=parse_batch(payload=newer), received_wall_time_s=11.0)
    snapshot = state.snapshot()
    assert snapshot.latest_frame is not None and snapshot.latest_frame.frame_id == 4
    assert snapshot.preview is not None and snapshot.preview.frame_id == 2
    assert snapshot.preview_boxes[0].x2 == pytest.approx(0.32)  # frame 2's box, not frame 4's


def test_new_session_clears_old_frames() -> None:
    state = LiveState()
    state.add_batch(batch=parse_batch(payload=make_payload(session_id=1)), received_wall_time_s=10.0)
    restarted = make_payload(session_id=2)
    del restarted["preview"]
    state.add_batch(batch=parse_batch(payload=restarted), received_wall_time_s=11.0)
    snapshot = state.snapshot()
    assert snapshot.session_id == 2
    assert snapshot.preview is None
    assert snapshot.preview_boxes == ()


def test_history_is_bounded() -> None:
    state = LiveState(history_frames=3)
    state.add_batch(batch=parse_batch(payload=make_payload(frame_count=10)), received_wall_time_s=10.0)
    # The preview's frame (9) is among the newest 3, so it keeps its boxes.
    assert len(state.snapshot().preview_boxes) == 1
    with pytest.raises(ValueError):
        LiveState(history_frames=0)


def test_snapshot_json_leaves_out_the_picture() -> None:
    state = LiveState()
    state.add_batch(batch=parse_batch(payload=copy.deepcopy(make_payload())), received_wall_time_s=10.0)
    data = snapshot_to_json(snapshot=state.snapshot(), now_wall_time_s=12.5)
    assert data["seconds_since_last_batch"] == pytest.approx(2.5)
    assert data["preview"] == {"frame_id": 1, "rotation_deg": 180}
    assert "jpeg" not in str(data)
