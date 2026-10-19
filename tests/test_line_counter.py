"""Tests for counting items as they cross the line."""

from door_counter.gateway.line_counter import LineCounter
from door_counter.gateway.messages import Frame, Item, Mask
from door_counter.yolo_seg import Rect


def make_frame(*, frame_id: int, boxes: list[tuple[float, float, int]]) -> Frame:
    """One frame at 5 FPS with square items.

    Args:
        frame_id: the frame number.
        boxes: (centre x, centre y, class id) per item, as fractions of the frame.

    Returns:
        The frame.
    """
    items = []
    for cx, cy, class_id in boxes:
        box = Rect(x1=cx - 0.05, y1=cy - 0.05, x2=cx + 0.05, y2=cy + 0.05)
        items.append(Item(class_id=class_id, score=0.35, box=box,  # a typical score for the tiny model
                          mask=Mask(rect=box, width_cells=1, height_cells=1, runs=(0, 1))))
    return Frame(frame_id=frame_id, wall_time_s=1000.0 + 0.2 * frame_id, width_px=640, height_px=480,
                 inference_s=0.13, items=tuple(items))


def test_counts_each_material_once_when_it_crosses() -> None:
    counter = LineCounter()
    for k in range(20):  # a can moves left to right across the line; a bag stays on the left
        counter.update(frame=make_frame(frame_id=k, boxes=[(0.1 + 0.04 * k, 0.3, 3), (0.2, 0.7, 1)]))
    assert counter.crossings(class_count=4) == (0, 0, 0, 1)


def test_counts_both_directions_and_survives_empty_frames() -> None:
    counter = LineCounter()
    frame_id = 0
    for k in range(20):  # a bottle moves right to left
        counter.update(frame=make_frame(frame_id=frame_id, boxes=[(0.9 - 0.04 * k, 0.5, 2)]))
        frame_id += 1
    for _ in range(30):  # belt empty for 6 s: the old track is dropped
        counter.update(frame=make_frame(frame_id=frame_id, boxes=[]))
        frame_id += 1
    for k in range(20):  # a second bottle, left to right
        counter.update(frame=make_frame(frame_id=frame_id, boxes=[(0.1 + 0.04 * k, 0.5, 2)]))
        frame_id += 1
    assert counter.crossings(class_count=4) == (0, 0, 2, 0)
