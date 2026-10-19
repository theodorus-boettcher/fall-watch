"""Score several models on several labelled picture sets, per material, at the phone's input size.

Each set is a YOLO ``images`` folder (labels next to it). Scores are mask mAP50 (how well the outlines
match, 0 to 1) overall and per material. Needs the ``train`` dependency group. Example:
``uv run --group train python -m door_counter.compare --model v1=runs/sort-v1/weights/best.pt
--model v2=runs/sort-v2/weights/best.pt --set plant=data/zerowaste-960/test/images
--set home=data/mix-v2/home/valid/images --out measurements/compare-v1-v2.json``.
"""

import argparse
import json
import tempfile
from pathlib import Path

import yaml
from ultralytics.models import YOLO

from door_counter.zerowaste import CLASS_NAMES

PHONE_INPUT_SIZE_PX = 320


def score_set(*, weights: Path, images_dir: Path) -> dict[str, object]:
    """Score one model on one picture set.

    Args:
        weights: model weights (``.pt``) or an exported ncnn folder.
        images_dir: the set's ``images`` folder.

    Returns:
        Pictures, mask mAP50 and mAP50-95 overall, and mask mAP50 per material (None where the set has none).
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = Path(tmp) / "data.yaml"
        folder = str(images_dir.resolve())
        spec = {"path": str(images_dir.parent.resolve()), "train": folder,  # Ultralytics requires train; unused
                "val": folder, "names": dict(enumerate(CLASS_NAMES))}
        data_yaml.write_text(yaml.safe_dump(spec), encoding="utf-8")
        metrics = YOLO(model=str(weights), task="segment").val(data=str(data_yaml), split="val",
                                                                imgsz=PHONE_INPUT_SIZE_PX, plots=False, verbose=False)
    per_class = dict.fromkeys(CLASS_NAMES)
    for i, class_index in enumerate(metrics.seg.ap_class_index):
        per_class[CLASS_NAMES[int(class_index)]] = round(float(metrics.seg.ap50[i]), 3)
    return {"pictures": sum(1 for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")),
            "mask_map50": round(float(metrics.seg.map50), 3), "mask_map50_95": round(float(metrics.seg.map), 3),
            "mask_map50_per_material": per_class}


def main() -> None:
    """Score every model on every set, print a table and save JSON."""
    parser = argparse.ArgumentParser(description="Compare models on labelled picture sets.")
    parser.add_argument("--model", action="append", required=True, help="name=weights")
    parser.add_argument("--set", action="append", required=True, help="name=images folder")
    parser.add_argument("--out", type=Path, help="write the scores here as JSON")
    args = parser.parse_args()
    results = []
    for model_spec in args.model:
        model, _, weights = model_spec.partition("=")
        for set_spec in args.set:
            name, _, folder = set_spec.partition("=")
            results.append({"model": model, "set": name, **score_set(weights=Path(weights), images_dir=Path(folder))})
    print(f"{'model':6} {'set':8} {'pics':>5} {'mAP50':>6}  per material {CLASS_NAMES}")
    for r in results:
        per = r["mask_map50_per_material"]
        assert isinstance(per, dict)
        print(f"{r['model']:6} {r['set']:8} {r['pictures']:5} {r['mask_map50']:6.3f}  "
              + "  ".join("  -  " if per[c] is None else f"{per[c]:.3f}" for c in CLASS_NAMES))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return None


if __name__ == "__main__":
    main()
