"""Reference decoding of YOLO26-seg (ncnn export), the same math as ``detector.cpp`` on the phone.

The model sees a square letterboxed picture. ``out0`` has one column per candidate: cx, cy, w, h
(letterbox pixels), one score per class, then the mask weights. ``out1`` holds the mask prototypes
at a quarter of the input size. An item's mask = sigmoid(weights x prototypes), cut to its box.

Masks travel as a small patch on the prototype grid plus run-length encoding (RLE): the patch is read
row by row, and the runs alternate between "off" and "on" cells, starting with "off".
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

BOX_ROWS = 4
PAD_VALUE = 114
PROTO_STRIDE_PX = 4  # prototypes are a quarter of the input size
DEFAULT_MIN_SCORE = 0.25
DEFAULT_OVERLAP_IOU = 0.7
MASK_THRESHOLD = 0.5
MAX_ITEMS = 50


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
class Letterbox:
    """Where the upright picture sits inside the square model input."""

    input_size_px: int
    scale: float
    pad_left_px: int
    pad_top_px: int
    content_width_px: int
    content_height_px: int


@dataclass(frozen=True)
class MaskPatch:
    """A mask on the prototype grid: ``cells`` covers ``rect`` (0..1 of the upright frame)."""

    rect: Rect
    cells: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Item:
    """One item found by the model. ``box`` is 0..1 of the upright frame."""

    class_id: int
    score: float
    box: Rect
    mask: MaskPatch


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
    """Greedy non-maximum suppression across all classes (each item has one material).

    Args:
        boxes: candidate boxes.
        scores: their scores.
        iou_threshold: boxes overlapping a better one by more than this are dropped.

    Returns:
        Indices of the kept candidates, best first, at most ``MAX_ITEMS``.
    """
    kept: list[int] = []
    for i in sorted(range(len(boxes)), key=lambda k: -scores[k]):
        if all(box_iou(a=boxes[i], b=boxes[k]) <= iou_threshold for k in kept):
            kept.append(i)
            if len(kept) == MAX_ITEMS:
                break
    return kept


def decode(
    *,
    out0: npt.NDArray[np.float32],
    protos: npt.NDArray[np.float32],
    letterbox: Letterbox,
    min_score: float = DEFAULT_MIN_SCORE,
    iou_threshold: float = DEFAULT_OVERLAP_IOU,
) -> list[Item]:
    """Turn the raw model outputs into items with masks.

    Args:
        out0: (4 + classes + mask weights) x candidates.
        protos: mask prototypes, weights x grid height x grid width.
        letterbox: geometry used to build the input.
        min_score: lowest class score to keep.
        iou_threshold: overlap above which the weaker of two boxes is dropped.

    Returns:
        Items, best first.

    Raises:
        ValueError: if the two outputs do not fit together.
    """
    mask_dim, grid_h, grid_w = protos.shape
    class_count = out0.shape[0] - BOX_ROWS - mask_dim
    if class_count < 1:
        raise ValueError(f"out0 has {out0.shape[0]} rows, too few for {mask_dim} mask weights")
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
    weights = out0[BOX_ROWS + class_count :]
    items = []
    for k in kept:
        column = int(candidates[k])
        patch = _mask_patch(
            weights=weights[:, column], protos=protos, box=letterbox_boxes[k], letterbox=letterbox, grid=(grid_h, grid_w)
        )
        items.append(
            Item(
                class_id=int(best_class[column]),
                score=float(best_score[column]),
                box=_to_frame(rect=letterbox_boxes[k], letterbox=letterbox),
                mask=patch,
            )
        )
    return items


def _mask_patch(
    *,
    weights: npt.NDArray[np.float32],
    protos: npt.NDArray[np.float32],
    box: Rect,
    letterbox: Letterbox,
    grid: tuple[int, int],
) -> MaskPatch:
    """Build one item's mask on the prototype grid, cut to its box and to the picture area.

    Args:
        weights: the item's mask weights.
        protos: mask prototypes.
        box: the item's box in letterbox pixels.
        letterbox: geometry used to build the input.
        grid: prototype grid height and width.

    Returns:
        The mask patch.
    """
    grid_h, grid_w = grid
    content = Rect(
        x1=letterbox.pad_left_px,
        y1=letterbox.pad_top_px,
        x2=letterbox.pad_left_px + letterbox.content_width_px,
        y2=letterbox.pad_top_px + letterbox.content_height_px,
    )
    x1 = int(np.clip(np.floor(max(box.x1, content.x1) / PROTO_STRIDE_PX), 0, grid_w))
    y1 = int(np.clip(np.floor(max(box.y1, content.y1) / PROTO_STRIDE_PX), 0, grid_h))
    x2 = int(np.clip(np.ceil(min(box.x2, content.x2) / PROTO_STRIDE_PX), x1, grid_w))
    y2 = int(np.clip(np.ceil(min(box.y2, content.y2) / PROTO_STRIDE_PX), y1, grid_h))
    logits = np.tensordot(weights, protos[:, y1:y2, x1:x2], axes=1)
    cells = (1.0 / (1.0 + np.exp(-logits))) > MASK_THRESHOLD
    cell_rect = Rect(
        x1=x1 * PROTO_STRIDE_PX, y1=y1 * PROTO_STRIDE_PX, x2=x2 * PROTO_STRIDE_PX, y2=y2 * PROTO_STRIDE_PX
    )
    return MaskPatch(rect=_to_frame(rect=cell_rect, letterbox=letterbox), cells=cells)


def _to_frame(*, rect: Rect, letterbox: Letterbox) -> Rect:
    """Map a rectangle from letterbox pixels to 0..1 of the upright frame.

    Args:
        rect: rectangle in letterbox pixels.
        letterbox: geometry used to build the input.

    Returns:
        The rectangle in 0..1, clipped to the frame.
    """

    def to_x(v: float) -> float:
        return float(np.clip((v - letterbox.pad_left_px) / letterbox.content_width_px, 0.0, 1.0))

    def to_y(v: float) -> float:
        return float(np.clip((v - letterbox.pad_top_px) / letterbox.content_height_px, 0.0, 1.0))

    return Rect(x1=to_x(rect.x1), y1=to_y(rect.y1), x2=to_x(rect.x2), y2=to_y(rect.y2))


def encode_runs(*, cells: npt.NDArray[np.bool_]) -> list[int]:
    """Run-length encode a mask patch row by row, starting with an "off" run (possibly 0).

    Args:
        cells: the patch, height x width.

    Returns:
        Alternating off/on run lengths that add up to height x width.
    """
    flat = cells.ravel()
    runs: list[int] = []
    current = False
    length = 0
    for value in flat:
        if bool(value) == current:
            length += 1
        else:
            runs.append(length)
            current = not current
            length = 1
    runs.append(length)
    return runs


def decode_runs(*, runs: Sequence[int], width: int, height: int) -> npt.NDArray[np.bool_]:
    """Undo ``encode_runs``.

    Args:
        runs: alternating off/on run lengths, starting with "off".
        width: patch width in cells.
        height: patch height in cells.

    Returns:
        The patch, height x width.

    Raises:
        ValueError: if a run is negative or the runs do not add up to width x height.
    """
    if any(run < 0 for run in runs):
        raise ValueError("runs must not be negative")
    if sum(runs) != width * height:
        raise ValueError(f"runs add up to {sum(runs)}, expected {width * height}")
    values = np.repeat(np.arange(len(runs)) % 2 == 1, runs)
    return values.reshape(height, width)

