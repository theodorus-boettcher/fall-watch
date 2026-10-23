"""Tests for the reference YOLO-pose decoding."""

import numpy as np
import pytest

from fall_watch import yolo_pose


def test_letterbox_centres_a_landscape_frame() -> None:
    lb = yolo_pose.make_letterbox(width_px=640, height_px=480, input_size_px=320)
    assert (lb.content_width_px, lb.content_height_px) == (320, 240)
    assert (lb.pad_left_px, lb.pad_top_px) == (0, 40)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = yolo_pose.letterbox_image(image_rgb=image, letterbox=lb)
    assert canvas.shape == (320, 320, 3)
    assert canvas[0, 0, 0] == yolo_pose.PAD_VALUE and canvas[160, 160, 0] == 0


def test_remove_overlaps_keeps_the_best_of_a_group() -> None:
    boxes = [
        yolo_pose.Rect(x1=0, y1=0, x2=10, y2=10),
        yolo_pose.Rect(x1=1, y1=1, x2=10, y2=10),  # overlaps the first a lot
        yolo_pose.Rect(x1=50, y1=50, x2=60, y2=60),
    ]
    assert yolo_pose.remove_overlaps(boxes=boxes, scores=[0.5, 0.9, 0.4], iou_threshold=0.7) == [1, 2]


def test_decode_finds_one_person_with_keypoints() -> None:
    lb = yolo_pose.make_letterbox(width_px=640, height_px=480, input_size_px=320)
    class_count, candidates = 2, 5
    out0 = np.zeros((yolo_pose.BOX_ROWS + class_count + yolo_pose.KEYPOINT_ROWS, candidates), dtype=np.float32)
    # Candidate 2: a 40 x 40 px box centred at (100, 100) in letterbox pixels, class 1.
    out0[:4, 2] = [100, 100, 40, 40]
    out0[4 + 1, 2] = 0.9
    keypoints = out0[4 + class_count :, 2].reshape(yolo_pose.KEYPOINT_COUNT, 3)
    keypoints[:] = [160, 40, 0.2]  # every point at the top-middle of the picture content
    keypoints[5] = [0, 280, 0.9]  # left shoulder at the bottom-left corner of the content
    out0[4 + class_count :, 2] = keypoints.ravel()
    people = yolo_pose.decode(out0=out0, letterbox=lb)
    assert len(people) == 1
    person = people[0]
    assert person.class_id == 1 and person.score == pytest.approx(0.9)
    assert person.box.x1 == pytest.approx(80 / 320) and person.box.y1 == pytest.approx((80 - 40) / 240)
    assert len(person.keypoints) == yolo_pose.KEYPOINT_COUNT
    assert (person.keypoints[0].x, person.keypoints[0].y) == pytest.approx((0.5, 0.0))
    shoulder = person.keypoints[5]
    assert (shoulder.x, shoulder.y, shoulder.visibility) == pytest.approx((0.0, 1.0, 0.9))


def test_decode_rejects_an_output_without_keypoints() -> None:
    lb = yolo_pose.make_letterbox(width_px=640, height_px=480, input_size_px=320)
    with pytest.raises(ValueError):
        yolo_pose.decode(out0=np.zeros((4 + 1 + 32, 4), dtype=np.float32), letterbox=lb)
