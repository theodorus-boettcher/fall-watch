"""Tests for the reference YOLO-seg decoding."""

import numpy as np
import pytest

from door_counter import yolo_seg


def test_letterbox_centres_a_landscape_frame() -> None:
    lb = yolo_seg.make_letterbox(width_px=640, height_px=480, input_size_px=320)
    assert (lb.content_width_px, lb.content_height_px) == (320, 240)
    assert (lb.pad_left_px, lb.pad_top_px) == (0, 40)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = yolo_seg.letterbox_image(image_rgb=image, letterbox=lb)
    assert canvas.shape == (320, 320, 3)
    assert canvas[0, 0, 0] == yolo_seg.PAD_VALUE and canvas[160, 160, 0] == 0


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (7, 2)])
def test_runs_round_trip(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(seed=1)
    cells = rng.random(shape) > 0.5
    runs = yolo_seg.encode_runs(cells=cells)
    assert sum(runs) == cells.size
    restored = yolo_seg.decode_runs(runs=runs, width=shape[1], height=shape[0])
    assert (restored == cells).all()


def test_runs_start_with_an_off_run() -> None:
    assert yolo_seg.encode_runs(cells=np.array([[True, True, False]])) == [0, 2, 1]
    with pytest.raises(ValueError):
        yolo_seg.decode_runs(runs=[1, 1], width=3, height=1)
    with pytest.raises(ValueError):
        yolo_seg.decode_runs(runs=[4, -1], width=3, height=1)


def test_remove_overlaps_keeps_the_best_of_a_group() -> None:
    boxes = [
        yolo_seg.Rect(x1=0, y1=0, x2=10, y2=10),
        yolo_seg.Rect(x1=1, y1=1, x2=10, y2=10),  # overlaps the first a lot
        yolo_seg.Rect(x1=50, y1=50, x2=60, y2=60),
    ]
    assert yolo_seg.remove_overlaps(boxes=boxes, scores=[0.5, 0.9, 0.4], iou_threshold=0.7) == [1, 2]


def test_decode_finds_one_item_with_its_mask() -> None:
    lb = yolo_seg.make_letterbox(width_px=640, height_px=480, input_size_px=320)
    class_count, mask_dim, candidates = 3, 2, 5
    out0 = np.zeros((yolo_seg.BOX_ROWS + class_count + mask_dim, candidates), dtype=np.float32)
    # Candidate 2: a 40 x 40 px box centred at (100, 100) in letterbox pixels, class 1, mask weight on proto 0.
    out0[:4, 2] = [100, 100, 40, 40]
    out0[4 + 1, 2] = 0.9
    out0[4 + class_count, 2] = 10.0
    protos = np.full((mask_dim, 80, 80), -1.0, dtype=np.float32)
    protos[0, 22:28, 22:28] = 1.0  # "on" cells in the middle of the box
    items = yolo_seg.decode(out0=out0, protos=protos, letterbox=lb)
    assert len(items) == 1
    item = items[0]
    assert item.class_id == 1 and item.score == pytest.approx(0.9)
    assert item.box.x1 == pytest.approx(80 / 320) and item.box.y1 == pytest.approx((80 - 40) / 240)
    assert item.mask.cells.shape == (10, 10)  # box covers proto cells 20..29
    assert int(item.mask.cells.sum()) == 36


def test_decode_rejects_mismatched_outputs() -> None:
    lb = yolo_seg.make_letterbox(width_px=640, height_px=480, input_size_px=320)
    with pytest.raises(ValueError):
        yolo_seg.decode(
            out0=np.zeros((6, 4), dtype=np.float32), protos=np.zeros((32, 80, 80), dtype=np.float32), letterbox=lb
        )
