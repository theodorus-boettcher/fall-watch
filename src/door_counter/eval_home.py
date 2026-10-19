"""Score models on the home test scenes: items counted per material in each picture vs what was written down.

The test scenes are capture clips named ``test-<n>``. ``truth.json`` holds what each scene contains,
e.g. ``{"test-1": {"metal": 2, "rigid plastic": 1}}``; materials left out count as 0. The models run
as the exported ncnn files the phone uses, at the phone's input size and score threshold.

Needs the ``train`` dependency group. Example:
``uv run --group train python -m door_counter.eval_home --clips data/captures --truth data/captures/truth.json
--model v1=runs/sort-v1/weights/best_ncnn_model --model v2=runs/sort-v2/weights/best_ncnn_model``.
"""

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ultralytics.models import YOLO

from door_counter.zerowaste import CLASS_NAMES

PHONE_INPUT_SIZE_PX = 320
PHONE_MIN_SCORE = 0.25  # MIN_SCORE in CounterService.java


@dataclass(frozen=True)
class CountScore:
    """How well one model counts on one set of pictures. Per-material lists follow ``CLASS_NAMES``."""

    model: str
    scene: str
    pictures: int
    exact_pct: float  # pictures where every material's count is right
    mean_abs_error: float  # items off, per picture and material
    mean_abs_error_per_material: tuple[float, ...]
    mean_counted_per_material: tuple[float, ...]
    mean_true_per_material: tuple[float, ...]


def truth_vector(*, contents: Mapping[str, int]) -> tuple[int, ...]:
    """Turn a scene's written-down contents into counts in ``CLASS_NAMES`` order.

    Args:
        contents: material -> count.

    Returns:
        One count per material.

    Raises:
        ValueError: if a material is unknown.
    """
    unknown = set(contents) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"unknown materials {sorted(unknown)}, expected {CLASS_NAMES}")
    return tuple(int(contents.get(name, 0)) for name in CLASS_NAMES)


def score_counts(
    *, model: str, scene: str, counted: Sequence[Sequence[int]], truth: Sequence[int] | Sequence[Sequence[int]]
) -> CountScore:
    """Compare per-picture counts with the truth.

    Args:
        model: model name for the report.
        scene: scene name (or ``all``).
        counted: one row per picture, one count per material.
        truth: the true count per material, the same for every picture of the scene (or one row per
            picture when scenes are pooled: pass ``truth`` with the same length as ``counted``).

    Returns:
        The score.

    Raises:
        ValueError: if there are no pictures.
    """
    got = np.asarray(counted, dtype=np.float64).reshape(-1, len(CLASS_NAMES))
    if len(got) == 0:
        raise ValueError(f"no pictures for {scene}")
    want = np.broadcast_to(np.asarray(truth, dtype=np.float64).reshape(-1, len(CLASS_NAMES)), got.shape)
    error = np.abs(got - want)
    return CountScore(
        model=model,
        scene=scene,
        pictures=len(got),
        exact_pct=100.0 * float(np.mean(np.all(error == 0, axis=1))),
        mean_abs_error=float(error.mean()),
        mean_abs_error_per_material=tuple(round(float(v), 3) for v in error.mean(axis=0)),
        mean_counted_per_material=tuple(round(float(v), 3) for v in got.mean(axis=0)),
        mean_true_per_material=tuple(round(float(v), 3) for v in want.mean(axis=0)),
    )


def count_items(*, model_path: Path, pictures: Sequence[Path]) -> Iterator[tuple[int, ...]]:
    """Run a model the way the phone does and count items per material in each picture.

    Args:
        model_path: an exported ncnn model folder (or .pt weights).
        pictures: the pictures.

    Returns:
        A generator of counts in ``CLASS_NAMES`` order.
    """
    model = YOLO(model=str(model_path), task="segment")
    ours = {i: CLASS_NAMES.index(name) for i, name in model.names.items()}
    for result in model.predict(source=[str(p) for p in pictures], imgsz=PHONE_INPUT_SIZE_PX, conf=PHONE_MIN_SCORE,
                                stream=True, verbose=False):
        counts = [0] * len(CLASS_NAMES)
        boxes = getattr(result, "boxes", None)  # predict() is typed loosely; a Results always has boxes
        for class_id in [] if boxes is None else boxes.cls.tolist():
            counts[ours[int(class_id)]] += 1
        yield tuple(counts)
    return None


def main() -> None:
    """Score every model on every test scene, print a table and save it as JSON."""
    parser = argparse.ArgumentParser(description="Count accuracy on the home test scenes.")
    parser.add_argument("--clips", type=Path, required=True, help="capture folder with test-<n> clips")
    parser.add_argument("--truth", type=Path, required=True, help="JSON: scene -> {material: count}")
    parser.add_argument("--model", action="append", required=True, help="name=path, e.g. v1=runs/.../best_ncnn_model")
    parser.add_argument("--out", type=Path, help="write the scores here as JSON")
    args = parser.parse_args()
    with args.truth.open(encoding="utf-8") as f:
        truth = {scene: truth_vector(contents=contents) for scene, contents in json.load(f).items()}
    scores: list[CountScore] = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        all_counted: list[tuple[int, ...]] = []
        all_truth: list[tuple[int, ...]] = []
        for scene, want in sorted(truth.items()):
            pictures = sorted((args.clips / scene).glob("*.jpg"))
            counted = list(count_items(model_path=Path(path), pictures=pictures))
            scores.append(score_counts(model=name, scene=scene, counted=counted, truth=want))
            all_counted += counted
            all_truth += [want] * len(counted)
        scores.append(score_counts(model=name, scene="all", counted=all_counted, truth=all_truth))
    print(f"{'model':6} {'scene':8} {'pics':>5} {'exact %':>8} {'off/mat':>8}  off per material {CLASS_NAMES}")
    for s in scores:
        print(f"{s.model:6} {s.scene:8} {s.pictures:5d} {s.exact_pct:8.1f} {s.mean_abs_error:8.2f}  "
              f"{s.mean_abs_error_per_material}")
    if args.out is not None:
        args.out.write_text(json.dumps([asdict(s) for s in scores], indent=2) + "\n", encoding="utf-8")
    return None


if __name__ == "__main__":
    main()
