"""Tests for the CAUCAFall converter's matching and label lines."""

from pathlib import Path

import numpy as np
import pytest

from door_counter.fall_data import TeacherPerson, coco_keypoints, match_teacher, read_yolo_box, yolo_line
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
