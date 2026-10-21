"""Tests for the CAUCAFall converter's matching and label lines."""

from pathlib import Path

import numpy as np
import pytest

from door_counter.fall_data import (TeacherPerson, activity_ranges, coco_keypoints, match_teacher, posture_at,
                                     read_yolo_box, yolo_line)
from door_counter.yolo_pose import Rect


def teacher_at(*, box: Rect, confidence: float = 0.9) -> TeacherPerson:
    """A teacher person whose keypoints all sit at the box centre."""
    centre = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
    return TeacherPerson(box=box, xy=np.tile(centre, (17, 1)), confidence=np.full(17, confidence))


def test_read_yolo_box_maps_centre_and_size(tmp_path: Path) -> None:
    label = tmp_path / "f.txt"
    label.write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    cls, box = read_yolo_box(label=label) or (None, None)
    assert cls == 1 and box == Rect(x1=pytest.approx(0.4), y1=pytest.approx(0.3), x2=pytest.approx(0.6),
                                    y2=pytest.approx(0.7))
    label.write_text("", encoding="utf-8")
    assert read_yolo_box(label=label) is None
    assert read_yolo_box(label=tmp_path / "missing.txt") is None


def test_teacher_match_needs_enough_overlap() -> None:
    box = Rect(x1=0.2, y1=0.2, x2=0.4, y2=0.6)
    near = teacher_at(box=Rect(x1=0.21, y1=0.22, x2=0.41, y2=0.6))
    far = teacher_at(box=Rect(x1=0.6, y1=0.2, x2=0.8, y2=0.6))
    assert match_teacher(box=box, people=[far, near]) is near
    assert match_teacher(box=box, people=[far]) is None


def test_keypoints_unseen_or_unmatched_are_not_labelled() -> None:
    assert coco_keypoints(person=None, width_px=640, height_px=480) == [0.0] * 51
    sure = coco_keypoints(person=teacher_at(box=Rect(x1=0.0, y1=0.0, x2=0.5, y2=0.5)), width_px=640, height_px=480)
    assert sure[:3] == [160.0, 120.0, 2.0]
    unsure = coco_keypoints(person=teacher_at(box=Rect(x1=0.0, y1=0.0, x2=0.5, y2=0.5), confidence=0.2),
                            width_px=640, height_px=480)
    assert unsure[:3] == [0.0, 0.0, 0.0]
    line = yolo_line(class_id=1, box=Rect(x1=0.0, y1=0.0, x2=0.5, y2=0.5), keypoints=sure, width_px=640,
                     height_px=480)
    assert line.split()[:8] == ["1", "0.25000", "0.25000", "0.50000", "0.50000", "0.25000", "0.25000", "1.00000"]


def test_activity_ranges_read_gmdcsa_lines() -> None:
    line = ("05.mp4,05, Day (Light On),Full T-shirt & Trousers, Walking to falling. Right side fall, "
            "Falling (SW)[1.8 to 5]:; Walking[0 to 1.8]")
    assert activity_ranges(text=line) == {"Falling": [(1.8, 5.0)], "Walking": [(0.0, 1.8)]}
    assert activity_ranges(text="Fall (FW)[2 to 4]; Standing[4.5 to 6.2; 10 to 13]") == {
        "Falling": [(2.0, 4.0)], "Standing": [(4.5, 6.2), (10.0, 13.0)]}


def test_posture_comes_from_the_time() -> None:
    fall = activity_ranges(text="Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]")
    assert posture_at(ranges=fall, time_s=1.0) == 0  # sitting
    assert posture_at(ranges=fall, time_s=4.0) is None  # on the way down
    assert posture_at(ranges=fall, time_s=5.0) == 1  # on the floor
    bed = activity_ranges(text="Sitting[0 to 1]; Sleeping[2 to 8]")
    assert posture_at(ranges=bed, time_s=5.0) == 1
    assert posture_at(ranges=bed, time_s=1.5) is None  # nothing named
    push = activity_ranges(text="Exercising[1.9 to 8.3]; Sitting[9.5 to 10.5]")
    assert posture_at(ranges=push, time_s=5.0) is None
    clash = activity_ranges(text="Sleeping[0 to 5]; Sitting[4 to 6]")
    assert posture_at(ranges=clash, time_s=4.5) is None
