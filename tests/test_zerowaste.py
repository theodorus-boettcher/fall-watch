"""Tests for the ZeroWaste converter, on a tiny made-up split."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from door_counter import zerowaste


def make_split(*, root: Path) -> Path:
    """Write a split with one 40 x 20 image and two outlines (metal, soft_plastic).

    Args:
        root: temporary folder.

    Returns:
        The split folder.
    """
    split = root / "train"
    (split / "data").mkdir(parents=True)
    cv2.imwrite(str(split / "data" / "frame_1.PNG"), np.zeros((20, 40, 3), dtype=np.uint8))
    coco = {
        "images": [{"id": 7, "file_name": "frame_1.PNG", "width": 40, "height": 20}],
        "categories": [{"id": 3, "name": "metal"}, {"id": 9, "name": "soft_plastic"}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 3, "segmentation": [[0, 0, 20, 0, 20, 10, 0, 10]]},
            {"id": 2, "image_id": 7, "category_id": 9, "segmentation": [[20, 10, 40, 10, 40, 20]]},
        ],
    }
    (split / "labels.json").write_text(json.dumps(coco), encoding="utf-8")
    return split


def test_class_names_are_normalised() -> None:
    assert zerowaste.normalise_class_name(name="Soft_Plastic") == "soft plastic"
    with pytest.raises(ValueError):
        zerowaste.normalise_class_name(name="glass")


def test_convert_split_writes_both_formats(tmp_path: Path) -> None:
    split = make_split(root=tmp_path)
    stats = zerowaste.convert_split(source_dir=split, out_dir=tmp_path / "out", width_px=20)
    assert stats.images == 1 and stats.polygons == 2
    assert stats.polygons_per_class == (0, 1, 0, 1)  # soft plastic, metal
    image = cv2.imread(str(tmp_path / "out" / "images" / "frame_1.jpg"))
    assert image is not None and image.shape[:2] == (10, 20)
    lines = (tmp_path / "out" / "labels" / "frame_1.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].split()[:3] == ["3", "0.000000", "0.000000"]
    assert lines[1].split()[0] == "1"
    coco = json.loads((tmp_path / "out" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert coco["images"][0]["width"] == 20
    assert coco["annotations"][0]["segmentation"][0][:4] == [0.0, 0.0, 10.0, 0.0]  # scaled by 0.5
    assert coco["annotations"][0]["bbox"] == [0.0, 0.0, 10.0, 5.0]
    assert [c["name"] for c in coco["categories"]] == list(zerowaste.CLASS_NAMES)


def test_rle_annotations_are_refused(tmp_path: Path) -> None:
    split = make_split(root=tmp_path)
    coco = json.loads((split / "labels.json").read_text(encoding="utf-8"))
    coco["annotations"][0]["segmentation"] = {"counts": "abc", "size": [20, 40]}
    (split / "labels.json").write_text(json.dumps(coco), encoding="utf-8")
    with pytest.raises(ValueError):
        zerowaste.read_coco(labels_path=split / "labels.json")
