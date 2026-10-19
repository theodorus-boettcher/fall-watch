"""Tests for the Roboflow helpers that run offline."""

import zipfile
from pathlib import Path

import pytest

from door_counter import roboflow_io


def test_api_key_is_read_from_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nROBOFLOW_API_KEY=abc123\n", encoding="utf-8")
    assert roboflow_io.load_api_key(env_path=env) == "abc123"
    env.write_text("OTHER=1\n", encoding="utf-8")
    with pytest.raises(roboflow_io.MissingApiKeyError):
        roboflow_io.load_api_key(env_path=env)
    with pytest.raises(roboflow_io.MissingApiKeyError):
        roboflow_io.load_api_key(env_path=tmp_path / "missing.env")


def test_zip_puts_images_next_to_their_coco_file(tmp_path: Path) -> None:
    for split in roboflow_io.SPLITS:
        (tmp_path / "ds" / split / "images").mkdir(parents=True)
        (tmp_path / "ds" / split / roboflow_io.COCO_FILE).write_text("{}", encoding="utf-8")
        (tmp_path / "ds" / split / "images" / f"{split}_1.jpg").write_bytes(b"jpg")
        (tmp_path / "ds" / split / "labels").mkdir()  # YOLO labels must stay out of the zip
        (tmp_path / "ds" / split / "labels" / f"{split}_1.txt").write_text("0 0 0", encoding="utf-8")
    zip_path = tmp_path / "ds.zip"
    assert roboflow_io.pack_coco_zip(dataset_dir=tmp_path / "ds", zip_path=zip_path) == 6
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert names == {f"{s}/{n}" for s in roboflow_io.SPLITS for n in (roboflow_io.COCO_FILE, f"{s}_1.jpg")}


def test_zip_needs_every_split(tmp_path: Path) -> None:
    (tmp_path / "ds" / "train" / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        roboflow_io.pack_coco_zip(dataset_dir=tmp_path / "ds", zip_path=tmp_path / "ds.zip")


@pytest.mark.parametrize(
    ("clip", "material"),
    [("metal-1", "metal"), ("soft-plastic-2", "soft plastic"), ("rigid-plastic-1", "rigid plastic"),
     ("cardboard-3", "cardboard"), ("test-1", None), ("metal", None), ("soft-1", None)],
)
def test_material_comes_from_the_clip_name(clip: str, material: str | None) -> None:
    assert roboflow_io.material_of_clip(clip=clip) == material
