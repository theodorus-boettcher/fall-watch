"""Tests for the training helpers that need no GPU."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("ultralytics", reason="needs the train group: uv run --group train pytest")
from fall_watch.train import repeat_room  # noqa: E402


def test_room_pictures_are_listed_again(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    for name in ("room-up-1_a.jpg", "s01_b.jpg"):
        (tmp_path / "train" / "images" / name).write_bytes(b"jpg")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"path": "/elsewhere", "train": "train/images",
                                                        "val": "valid/images", "names": {0: "lying"}}),
                                        encoding="utf-8")
    out = repeat_room(data_yaml=tmp_path / "data.yaml", times=2)
    assert (tmp_path / "room-extra.txt").read_text(encoding="utf-8") == "./train/images/room-up-1_a.jpg\n" * 2
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == {
        "train": ["train/images", "room-extra.txt"], "val": "valid/images", "names": {0: "lying"}}
