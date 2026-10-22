"""Tests for the CAUCAFall converter's matching and label lines."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from door_counter.fall_data import (TeacherPerson, activity_ranges, coco_keypoints, match_teacher,
                                     missed_room_pictures, posture_at, read_yolo_box, room_samples, sam3_samples,
                                     yolo_line)
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


def test_room_posture_comes_from_the_clip_name(tmp_path: Path) -> None:
    for clip in ("room-up-1", "room-lying-2", "room-empty-1", "test-fall-1"):
        (tmp_path / clip).mkdir()
        cv2.imwrite(str(tmp_path / clip / "1_00000001.jpg"), np.zeros((24, 32, 3), dtype=np.uint8))
    found = {s.stem: (s.split, s.class_id, s.box, s.one_person) for s in room_samples(root=tmp_path)}
    assert found == {"room-empty-1_1_00000001": ("train", None, None, True),
                     "room-lying-2_1_00000001": ("train", 1, None, True), "room-up-1_1_00000001": ("train", 0, None, True)}
    (tmp_path / "room-sitting-1").mkdir()
    with pytest.raises(ValueError, match="room-sitting-1"):
        list(room_samples(root=tmp_path))


def test_missed_room_pictures_are_the_ones_without_a_label(tmp_path: Path) -> None:
    for clip in ("room-up-1", "room-empty-1", "test-fall-1"):
        (tmp_path / "captures" / clip).mkdir(parents=True)
        for frame in ("a", "b"):
            (tmp_path / "captures" / clip / f"{frame}.jpg").write_bytes(b"jpg")
    (tmp_path / "converted" / "train" / "labels").mkdir(parents=True)
    (tmp_path / "converted" / "train" / "labels" / "room-up-1_a.txt").write_text("", encoding="utf-8")
    assert missed_room_pictures(root=tmp_path / "captures", converted=tmp_path / "converted") == [
        tmp_path / "captures" / "room-up-1" / "b.jpg"]


def test_sam3_outlines_merge_into_one_box_per_picture(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    names = ("room-lying-2_1_00000007_jpg.rf.abc.jpg", "room-up-1_1_00000008_jpg.rf.def.jpg")
    for name in names:
        cv2.imwrite(str(tmp_path / "train" / name), np.zeros((50, 100, 3), dtype=np.uint8))
    coco = {"images": [{"id": i + 1, "file_name": n, "width": 100, "height": 50} for i, n in enumerate(names)],
            "annotations": [{"id": 1, "image_id": 1, "bbox": [10, 5, 20, 10]},  # a lying body in two parts
                            {"id": 2, "image_id": 1, "bbox": [40, 20, 30, 20]}]}  # picture 2: no outline
    (tmp_path / "train" / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    found = list(sam3_samples(root=tmp_path))
    assert [(s.stem, s.split, s.class_id, s.one_person) for s in found] == [
        ("room-lying-2_1_00000007", "train", 1, True)]
    box = found[0].box
    assert box is not None and (box.x1, box.y1, box.x2, box.y2) == pytest.approx((0.1, 0.1, 0.7, 0.8))
