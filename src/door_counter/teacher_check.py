"""How good is the teacher? SAM 3's plant labels and a student model's, both against the human labels.

SAM 3 labels come without scores, so both are compared at one operating point: an outline counts as found
when it overlaps a human outline of the same material by ``MATCH_IOU`` or more (each human outline once).
The student runs like on the phone (``PHONE_INPUT_SIZE_PX``, ``PHONE_MIN_SCORE``). Needs the ``train`` group.
Example: ``uv run --group train python -m door_counter.teacher_check --sam3 data/rf-teacher/train
--human data/zerowaste-960/test --student v2=runs/sort-v2/weights/best.pt --out measurements/teacher.json``.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import yaml
from ultralytics.models import YOLO

from door_counter.eval_home import PHONE_INPUT_SIZE_PX, PHONE_MIN_SCORE
from door_counter.mix import class_map_from_names, clean_plant_lines, remap_lines, source_stem
from door_counter.zerowaste import CLASS_NAMES

MATCH_IOU = 0.5
RASTER_PX = (320, 180)  # outlines are compared on this grid (the plant pictures are 16:9)


@dataclass(frozen=True)
class Agreement:
    """Outlines found and missed for one labeller on one set. Per-material lists follow ``CLASS_NAMES``."""

    labeller: str
    pictures: int
    precision: float
    recall: float
    f1: float
    precision_per_material: tuple[float, ...]
    recall_per_material: tuple[float, ...]


def raster(*, line: str) -> npt.NDArray[np.bool_]:
    """Fill one YOLO segmentation line on the comparison grid."""
    width, height = RASTER_PX
    xy = (np.asarray(line.split()[1:], dtype=np.float64).reshape(-1, 2) * (width, height)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [xy], 1)
    return mask.astype(bool)


def match_counts(*, found: Sequence[str], truth: Sequence[str]) -> npt.NDArray[np.int64]:
    """Count matched, found and true outlines per material for one picture.

    Args:
        found: label lines from the labeller (largest or most confident first).
        truth: human label lines.

    Returns:
        Array (materials, 3): matched, found, true.
    """
    counts = np.zeros((len(CLASS_NAMES), 3), dtype=np.int64)
    truth_masks = [(int(t.split()[0]), raster(line=t)) for t in truth]
    used = [False] * len(truth_masks)
    for line in found:
        cls, mask = int(line.split()[0]), raster(line=line)
        counts[cls, 1] += 1
        best, best_iou = None, MATCH_IOU
        for i, (t_cls, t_mask) in enumerate(truth_masks):
            if used[i] or t_cls != cls:
                continue
            union = (mask | t_mask).sum()
            iou = (mask & t_mask).sum() / union if union else 0.0
            if iou >= best_iou:
                best, best_iou = i, iou
        if best is not None:
            used[best] = True
            counts[cls, 0] += 1
    for t_cls, _ in truth_masks:
        counts[t_cls, 2] += 1
    return counts


def agreement(*, labeller: str, counts: npt.NDArray[np.int64], pictures: int) -> Agreement:
    """Turn summed counts into precision, recall and F1."""
    matched, found, true = counts.sum(axis=0)
    precision = matched / found if found else 0.0
    recall = matched / true if true else 0.0
    per_p = tuple(round(float(m / f), 3) if f else 0.0 for m, f, _ in counts)
    per_r = tuple(round(float(m / t), 3) if t else 0.0 for m, _, t in counts)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Agreement(labeller=labeller, pictures=pictures, precision=round(float(precision), 3),
                     recall=round(float(recall), 3), f1=round(float(f1), 3), precision_per_material=per_p,
                     recall_per_material=per_r)


def read_lines(*, path: Path) -> list[str]:
    """Label lines of one picture; none if the file is missing."""
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def main() -> None:
    """Score SAM 3's labels and each student's predictions on the same pictures."""
    parser = argparse.ArgumentParser(description="Compare SAM 3 and student outlines with human ones.")
    parser.add_argument("--sam3", type=Path, required=True, help="export split with SAM 3 labels (images/, labels/)")
    parser.add_argument("--human", type=Path, required=True, help="human-labelled split (images/, labels/)")
    parser.add_argument("--student", action="append", default=[], help="name=weights")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    names = yaml.safe_load((args.sam3.parent / "data.yaml").read_text(encoding="utf-8"))["names"]
    class_map = class_map_from_names(names=list(names.values()) if isinstance(names, dict) else names)
    pictures = {source_stem(name=p.name): p for p in (args.sam3 / "images").iterdir()}
    truth = {stem: read_lines(path=args.human / "labels" / f"{stem}.txt") for stem in pictures}
    sam3 = np.zeros((len(CLASS_NAMES), 3), dtype=np.int64)
    for stem, picture in pictures.items():
        lines = remap_lines(lines=read_lines(path=args.sam3 / "labels" / f"{picture.stem}.txt"), class_map=class_map)
        sam3 += match_counts(found=clean_plant_lines(lines=lines or []), truth=truth[stem])
    results = [agreement(labeller="sam3", counts=sam3, pictures=len(pictures))]
    for spec in args.student:
        name, _, weights = spec.partition("=")
        model = YOLO(model=weights, task="segment")
        ours = {i: CLASS_NAMES.index(n) for i, n in model.names.items()}
        counts = np.zeros((len(CLASS_NAMES), 3), dtype=np.int64)
        stems = list(pictures)
        results_iter = model.predict(source=[str(args.human / "images" / f"{s}.jpg") for s in stems],
                                     imgsz=PHONE_INPUT_SIZE_PX, conf=PHONE_MIN_SCORE, stream=True, verbose=False)
        for stem, result in zip(stems, results_iter):
            boxes, masks = getattr(result, "boxes", None), getattr(result, "masks", None)  # predict() is typed loosely
            classes = [] if boxes is None else boxes.cls.tolist()
            polygons = [] if masks is None else masks.xyn
            found = [f"{ours[int(c)]} " + " ".join(f"{v:.5f}" for v in xy.ravel())
                     for c, xy in zip(classes, polygons) if len(xy) >= 3]
            counts += match_counts(found=found, truth=truth[stem])
        results.append(agreement(labeller=name, counts=counts, pictures=len(pictures)))
    print(f"{'labeller':10} {'pics':>5} {'prec':>5} {'recall':>6} {'F1':>5}   recall per material {CLASS_NAMES}")
    for r in results:
        print(f"{r.labeller:10} {r.pictures:5} {r.precision:5.3f} {r.recall:6.3f} {r.f1:5.3f}   {r.recall_per_material}")
    if args.out is not None:
        args.out.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")
    return None


if __name__ == "__main__":
    main()
