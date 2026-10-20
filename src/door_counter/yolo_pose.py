"""Reference decoding of YOLO26-pose (ncnn export), the same math as ``detector.cpp`` on the phone.

The model sees a square letterboxed picture. ``out0`` has one column per candidate: cx, cy, w, h
(letterbox pixels), one score per class, then x, y and visibility for each of the 17 COCO keypoints
(x and y already in letterbox pixels, visibility already 0..1).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

BOX_ROWS = 4
KEYPOINT_COUNT = 17  # COCO person skeleton
KEYPOINT_ROWS = KEYPOINT_COUNT * 3
PAD_VALUE = 114
DEFAULT_MIN_SCORE = 0.25
DEFAULT_OVERLAP_IOU = 0.7
MAX_PEOPLE = 50


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle; units depend on the context (pixels or 0..1)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        """Area, 0 for an empty rectangle."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Keypoint:
    """One body point: x and y in 0..1 of the upright frame, visibility 0..1."""

    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class Letterbox:
    """Where the upright picture sits inside the square model input."""

    input_size_px: int
    scale: float
    pad_left_px: int
    pad_top_px: int
    content_width_px: int
    content_height_px: int


@dataclass(frozen=True)
class Person:
    """One person found by the model. ``box`` is 0..1 of the upright frame."""

    class_id: int
    score: float
    box: Rect
    keypoints: tuple[Keypoint, ...]


def make_letterbox(*, width_px: int, height_px: int, input_size_px: int) -> Letterbox:
    """Compute how a picture is scaled and padded into the square model input.

    Args:
        width_px: width of the upright picture.
        height_px: height of the upright picture.
        input_size_px: side of the square model input.

    Returns:
        The letterbox geometry, centred like Ultralytics.
    """
    scale = input_size_px / max(width_px, height_px)
    content_w = round(width_px * scale)
    content_h = round(height_px * scale)
    return Letterbox(
        input_size_px=input_size_px,
        scale=scale,
        pad_left_px=(input_size_px - content_w) // 2,
        pad_top_px=(input_size_px - content_h) // 2,
        content_width_px=content_w,
        content_height_px=content_h,
    )


def letterbox_image(*, image_rgb: npt.NDArray[np.uint8], letterbox: Letterbox) -> npt.NDArray[np.uint8]:
    """Resize and pad an RGB picture into the square model input.

    Args:
        image_rgb: upright picture, height x width x 3.
        letterbox: geometry from ``make_letterbox`` for this picture size.

    Returns:
        The square input picture, grey (114) around the content.
    """
    resized = cv2.resize(
        image_rgb, (letterbox.content_width_px, letterbox.content_height_px), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((letterbox.input_size_px, letterbox.input_size_px, 3), PAD_VALUE, dtype=np.uint8)
    top, left = letterbox.pad_top_px, letterbox.pad_left_px
    canvas[top : top + letterbox.content_height_px, left : left + letterbox.content_width_px] = resized
    return canvas


def box_iou(*, a: Rect, b: Rect) -> float:
    """Intersection over union of two rectangles.

    Args:
        a: first rectangle.
        b: second rectangle.

    Returns:
        IoU in 0..1, 0 when they do not touch.
    """
    inter = Rect(x1=max(a.x1, b.x1), y1=max(a.y1, b.y1), x2=min(a.x2, b.x2), y2=min(a.y2, b.y2)).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def remove_overlaps(*, boxes: Sequence[Rect], scores: Sequence[float], iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression across all classes (each person has one posture).

    Args:
        boxes: candidate boxes.
        scores: their scores.
        iou_threshold: boxes overlapping a better one by more than this are dropped.

    Returns:
        Indices of the kept candidates, best first, at most ``MAX_PEOPLE``.
    """
    kept: list[int] = []
    for i in sorted(range(len(boxes)), key=lambda k: -scores[k]):
        if all(box_iou(a=boxes[i], b=boxes[k]) <= iou_threshold for k in kept):
            kept.append(i)
            if len(kept) == MAX_PEOPLE:
                break
    return kept


def decode(
    *,
    out0: npt.NDArray[np.float32],
    letterbox: Letterbox,
    min_score: float = DEFAULT_MIN_SCORE,
    iou_threshold: float = DEFAULT_OVERLAP_IOU,
) -> list[Person]:
    """Turn the raw model output into people with keypoints.

    Args:
        out0: (4 + classes + 51) x candidates.
        letterbox: geometry used to build the input.
        min_score: lowest class score to keep.
        iou_threshold: overlap above which the weaker of two boxes is dropped.

    Returns:
        People, best first.

    Raises:
        ValueError: if out0 has too few rows for a class and the keypoints.
    """
    class_count = out0.shape[0] - BOX_ROWS - KEYPOINT_ROWS
    if class_count < 1:
        raise ValueError(f"out0 has {out0.shape[0]} rows, too few for {KEYPOINT_ROWS} keypoint values")
    class_scores = out0[BOX_ROWS : BOX_ROWS + class_count]
    best_class = class_scores.argmax(axis=0)
    best_score = class_scores.max(axis=0)
    candidates = np.flatnonzero(best_score >= min_score)
    cx, cy, w, h = (out0[row, candidates] for row in range(BOX_ROWS))
    letterbox_boxes = [
        Rect(x1=float(a - c / 2), y1=float(b - d / 2), x2=float(a + c / 2), y2=float(b + d / 2))
        for a, b, c, d in zip(cx, cy, w, h, strict=True)
    ]
    kept = remove_overlaps(
        boxes=letterbox_boxes, scores=[float(s) for s in best_score[candidates]], iou_threshold=iou_threshold
    )
    keypoint_values = out0[BOX_ROWS + class_count :]
    people = []
    for k in kept:
        column = int(candidates[k])
        values = keypoint_values[:, column].reshape(KEYPOINT_COUNT, 3)
        people.append(
            Person(
                class_id=int(best_class[column]),
                score=float(best_score[column]),
                box=_to_frame(rect=letterbox_boxes[k], letterbox=letterbox),
                keypoints=tuple(
                    Keypoint(x=_to_x(v=float(x), letterbox=letterbox), y=_to_y(v=float(y), letterbox=letterbox),
                             visibility=float(vis))
                    for x, y, vis in values
                ),
            )
        )
    return people


def _to_x(*, v: float, letterbox: Letterbox) -> float:
    """Letterbox pixels to 0..1 of the upright frame's width, clipped."""
    return float(np.clip((v - letterbox.pad_left_px) / letterbox.content_width_px, 0.0, 1.0))


def _to_y(*, v: float, letterbox: Letterbox) -> float:
    """Letterbox pixels to 0..1 of the upright frame's height, clipped."""
    return float(np.clip((v - letterbox.pad_top_px) / letterbox.content_height_px, 0.0, 1.0))


def _to_frame(*, rect: Rect, letterbox: Letterbox) -> Rect:
    """Map a rectangle from letterbox pixels to 0..1 of the upright frame.

    Args:
        rect: rectangle in letterbox pixels.
        letterbox: geometry used to build the input.

    Returns:
        The rectangle in 0..1, clipped to the frame.
    """
    return Rect(x1=_to_x(v=rect.x1, letterbox=letterbox), y1=_to_y(v=rect.y1, letterbox=letterbox),
                x2=_to_x(v=rect.x2, letterbox=letterbox), y2=_to_y(v=rect.y2, letterbox=letterbox))
