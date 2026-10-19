"""Move capture pictures that show a hand out of the clips, so no person reaches the dataset.

An open-vocabulary detector (YOLOE) looks at every picture twice: for a "hand" anywhere, and for a
"person" touching the picture edge. Pictures with a
hand, and their neighbours (a hand coming in at the edge is easy to miss), move to
``<out>/<clip>/``, outside the capture folder the Roboflow upload reads. Running it again on a
filtered clip moves nothing.

Needs the ``train`` dependency group. Example:
``uv run --group train python -m door_counter.hand_filter --clips data/captures --out data/hands``.
"""

import argparse
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

from ultralytics.models import YOLOE

# shortcut: Ultralytics downloads this and its text model (mobileclip2_b.ts) into the working
# directory and installs its CLIP package itself on first use; pin them in pyproject if it must run offline.
HAND_MODEL = "yoloe-26s-seg.pt"
HAND_PROMPT = "hand"  # misses fingertips at the picture edge
EDGE_PROMPT = "person"  # catches most edge fingertips, but also fires on crumpled bags in the middle
EDGE_MARGIN = 0.02  # a box this close to a side (fraction of width or height) touches the edge
MIN_HAND_SCORE = 0.1  # low on purpose: real hands scored 0.22 to 0.78, edge fingertips 0.13 to 0.33
NEIGHBOURS = 4  # pictures dropped on each side of one with a hand, about 1 s at 4 pictures per second
BATCH = 16


def pictures_to_drop(*, has_hand: Sequence[bool], neighbours: int) -> set[int]:
    """Pick the pictures to drop: each one with a hand, plus its neighbours.

    Args:
        has_hand: one flag per picture, in recording order.
        neighbours: how many pictures to drop on each side of one with a hand.

    Returns:
        Indexes into ``has_hand``.
    """
    drop: set[int] = set()
    for i, is_hand in enumerate(has_hand):
        if is_hand:
            drop.update(range(max(0, i - neighbours), min(len(has_hand), i + neighbours + 1)))
    return drop


def touches_edge(*, box: Sequence[float], width_px: int, height_px: int) -> bool:
    """Tell whether a box reaches the picture edge. A real hand always does: the arm is outside the picture.

    Args:
        box: x1, y1, x2, y2 in pixels.
        width_px: picture width.
        height_px: picture height.

    Returns:
        True if any side of the box is within ``EDGE_MARGIN`` of the picture's side.
    """
    x1, y1, x2, y2 = box
    return (min(x1, width_px - x2) < EDGE_MARGIN * width_px) or (min(y1, height_px - y2) < EDGE_MARGIN * height_px)


def find_hands(*, pictures: Sequence[Path]) -> Iterator[bool]:
    """Look for a hand in each picture.

    Args:
        pictures: the pictures.

    Returns:
        A generator of flags, True where a hand was found.
    """
    hand_model, edge_model = YOLOE(HAND_MODEL), YOLOE(HAND_MODEL)
    hand_model.set_classes([HAND_PROMPT])  # separate models: with both prompts in one, a hand gets only one label
    edge_model.set_classes([EDGE_PROMPT])
    for start in range(0, len(pictures), BATCH):  # a list source is loaded whole, so feed it in parts
        chunk = [str(p) for p in pictures[start:start + BATCH]]
        hand_results = hand_model.predict(source=chunk, conf=MIN_HAND_SCORE, verbose=False)
        edge_results = edge_model.predict(source=chunk, conf=MIN_HAND_SCORE, verbose=False)
        for hand, edge in zip(hand_results, edge_results, strict=True):
            hand_boxes, edge_boxes = getattr(hand, "boxes", None), getattr(edge, "boxes", None)  # loosely typed
            height_px, width_px = edge.orig_shape
            yield (hand_boxes is not None and len(hand_boxes) > 0) or (edge_boxes is not None and any(
                touches_edge(box=box, width_px=width_px, height_px=height_px) for box in edge_boxes.xyxy.tolist()))
    return None


def main() -> None:
    """Filter every clip and print what moved."""
    parser = argparse.ArgumentParser(description="Move capture pictures that show a hand out of the clips.")
    parser.add_argument("--clips", type=Path, required=True, help="capture folder, one subfolder per clip")
    parser.add_argument("--out", type=Path, required=True, help="pictures with hands move to <out>/<clip>/")
    parser.add_argument("--clip", action="append", help="only these clips (default: all)")
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS,
                        help="pictures dropped on each side of a hand; 0 for photo sets that are not recordings")
    args = parser.parse_args()
    clip_dirs = sorted(d for d in args.clips.iterdir() if d.is_dir() and (args.clip is None or d.name in args.clip))
    for clip_dir in clip_dirs:
        pictures = sorted(clip_dir.glob("*.jpg"))  # names are <session>_<frame>, so this is recording order
        has_hand = list(find_hands(pictures=pictures))
        drop = pictures_to_drop(has_hand=has_hand, neighbours=args.neighbours)
        (args.out / clip_dir.name).mkdir(parents=True, exist_ok=True)
        for i in sorted(drop):
            shutil.move(pictures[i], args.out / clip_dir.name / pictures[i].name)
        print(f"{clip_dir.name}: {len(pictures)} pictures, {sum(has_hand)} with a hand, "
              f"{len(drop)} moved, {len(pictures) - len(drop)} kept")
    return None


if __name__ == "__main__":
    main()
