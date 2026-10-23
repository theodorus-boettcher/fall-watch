"""Tests for the fall watch: postures, bed zones and alert timing."""

import pytest

from fall_watch.gateway.fall_watch import FallWatch, is_lying
from fall_watch.gateway.messages import Frame, Item
from fall_watch.yolo_pose import Keypoint, Rect

POSTURES = ("not lying", "lying")
BED = Rect(x1=0.6, y1=0.5, x2=1.0, y2=1.0)


def make_person(*, cx: float, cy: float, class_id: int = 0, torso: tuple[float, float] = (0.0, 0.0)) -> Item:
    """A person with a 0.2 x 0.2 box; shoulders and hips sit ``torso`` (dx, dy) apart.

    Args:
        cx: box centre x, 0..1 of the frame.
        cy: box centre y.
        class_id: posture class.
        torso: hips-to-shoulders offset; (0, 0) leaves the torso unseen.

    Returns:
        The person.
    """
    seen = torso != (0.0, 0.0)
    points = [Keypoint(x=cx, y=cy, visibility=0.1) for _ in range(17)]
    for i in (5, 6):
        points[i] = Keypoint(x=cx + torso[0], y=cy + torso[1], visibility=0.9 if seen else 0.1)
    for i in (11, 12):
        points[i] = Keypoint(x=cx, y=cy, visibility=0.9 if seen else 0.1)
    return Item(class_id=class_id, score=0.8, box=Rect(x1=cx - 0.1, y1=cy - 0.1, x2=cx + 0.1, y2=cy + 0.1),
                keypoints=tuple(points))


def frame_at(*, k: int, people: list[Item], start_s: float = 1000.0, step_s: float = 0.2) -> Frame:
    """Frame number k, step_s apart (5 FPS by default), 640 x 480."""
    return Frame(frame_id=k, wall_time_s=start_s + step_s * k, width_px=640, height_px=480, inference_s=0.15,
                 items=tuple(people))


def test_lying_from_class_or_from_the_torso() -> None:
    lying = make_person(cx=0.3, cy=0.5, class_id=1)
    assert is_lying(item=lying, class_names=POSTURES, width_px=640, height_px=480)
    upright = make_person(cx=0.3, cy=0.5, torso=(0.0, -0.2))  # shoulders straight above the hips
    flat = make_person(cx=0.3, cy=0.5, torso=(-0.2, 0.02))  # shoulders beside the hips
    assert not is_lying(item=upright, class_names=("person",), width_px=640, height_px=480)
    assert is_lying(item=flat, class_names=("person",), width_px=640, height_px=480)
    # No torso seen: the box decides. A square box in a 4:3 frame is wider than tall in pixels.
    assert is_lying(item=make_person(cx=0.3, cy=0.5), class_names=("person",), width_px=640, height_px=480)
    assert not is_lying(item=make_person(cx=0.3, cy=0.5), class_names=("person",), width_px=480, height_px=640)


def test_alert_after_five_seconds_on_the_floor_and_end_when_up() -> None:
    watch = FallWatch(beds=[BED])
    for k in range(10):  # 2 s standing
        watch.update(frame=frame_at(k=k, people=[make_person(cx=0.3, cy=0.4)]), class_names=POSTURES)
    for k in range(10, 34):  # 4.6 s on the floor: not yet
        watch.update(frame=frame_at(k=k, people=[make_person(cx=0.3, cy=0.5, class_id=1)]), class_names=POSTURES)
    assert watch.alerts() == ()
    assert watch.people()[0].lying_for_s == pytest.approx(4.6)
    for k in range(34, 40):  # past 5 s
        watch.update(frame=frame_at(k=k, people=[make_person(cx=0.3, cy=0.5, class_id=1)]), class_names=POSTURES)
    (alert,) = watch.alerts()
    assert alert.ended_s is None and alert.raised_s - alert.lying_since_s == pytest.approx(5.0)
    for k in range(40, 45):  # gets up
        watch.update(frame=frame_at(k=k, people=[make_person(cx=0.3, cy=0.45)]), class_names=POSTURES)
    (alert,) = watch.alerts()
    assert alert.ended_s == pytest.approx(1000.0 + 0.2 * 40)


def test_lying_in_bed_raises_no_alert() -> None:
    watch = FallWatch(beds=[BED])
    for k in range(100):  # 20 s lying in bed
        watch.update(frame=frame_at(k=k, people=[make_person(cx=0.8, cy=0.75, class_id=1)]), class_names=POSTURES)
    assert watch.alerts() == ()
    assert watch.people()[0].is_in_bed


def test_missed_frames_do_not_restart_the_timer_but_a_clock_jump_does() -> None:
    watch = FallWatch()
    lying = [make_person(cx=0.3, cy=0.5, class_id=1)]
    for k in range(15):
        watch.update(frame=frame_at(k=k, people=lying if k % 3 else []), class_names=POSTURES)  # every 3rd missed
    assert watch.people()[0].lying_for_s == pytest.approx(0.2 * (14 - 2))  # tracked from frame 2 (1 only spawns it)
    watch.update(frame=frame_at(k=15, people=lying, start_s=500.0), class_names=POSTURES)  # clock went back
    assert watch.people()[0].lying_for_s == 0.0


@pytest.mark.parametrize(("step_s", "missed"), [(0.2, 3), (1.05, 2)])  # full rate; motion gate's idle rate
def test_a_few_missed_frames_keep_the_person_and_their_timer(step_s: float, missed: int) -> None:
    watch = FallWatch()
    lying = [make_person(cx=0.3, cy=0.5, class_id=1)]
    frames = [lying] * 3 + [[]] * missed + [lying] * round(8 / step_s)
    for k, people in enumerate(frames):
        watch.update(frame=frame_at(k=k, people=people, step_s=step_s), class_names=POSTURES)
    (alert,) = watch.alerts()
    assert alert.lying_since_s < 1000.0 + 3 * step_s  # the timer started before the gap
    assert alert.raised_s - alert.lying_since_s == pytest.approx(5.0, abs=step_s)


def test_clear_ends_open_alerts() -> None:
    watch = FallWatch(alert_after_s=1.0)
    for k in range(10):
        watch.update(frame=frame_at(k=k, people=[make_person(cx=0.3, cy=0.5, class_id=1)]), class_names=POSTURES)
    assert watch.alerts()[0].ended_s is None
    watch.clear_alerts(now_s=2000.0)
    assert watch.alerts()[0].ended_s == 2000.0
    with pytest.raises(ValueError):
        FallWatch(alert_after_s=0.0)
