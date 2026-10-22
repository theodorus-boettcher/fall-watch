"""Train the phone model (YOLO26n-pose) on a GPU, score it, and export it for the phone.

Needs the ``train`` dependency group (``uv sync --group train``). Example:
``uv run --group train python -m door_counter.train --data data/fall-v1/data.yaml --name fall-v1``.
The phone files land in ``models/<name>/`` as ``<name>.param``, ``<name>.bin`` and ``<name>.labels``.
"""

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from ultralytics.models import YOLO

DEFAULT_BASE_MODEL = "yolo26n-pose.pt"
DEFAULT_IMAGE_SIZE_PX = 320  # what the phone runs
DEFAULT_EPOCHS = 80
DEFAULT_BATCH = 32
# What optimizer="auto" picked for fall-v1 and fall-v2. Fixed, because "auto" switches to MuSGD at a 6x higher
# rate once a run passes 10,000 steps, so a bigger dataset would also change the optimizer.
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.001667
MOMENTUM = 0.9
SPLIT_KEYS = {"val": "val", "test": "test"}  # our split name -> data.yaml key
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class TrainConfig:
    """One training run."""

    data_yaml: Path
    name: str
    base_model: str = DEFAULT_BASE_MODEL
    image_size_px: int = DEFAULT_IMAGE_SIZE_PX
    epochs: int = DEFAULT_EPOCHS
    batch: int = DEFAULT_BATCH
    runs_dir: Path = Path("runs")


@dataclass(frozen=True)
class Scores:
    """Accuracy on one split. mAP = mean average precision, the standard COCO-style score."""

    split: str
    images: int
    box_map50: float
    box_map50_95: float
    pose_map50: float  # keypoints: how close the predicted points are to the true ones (OKS)
    pose_map50_95: float


def train(*, config: TrainConfig) -> Path:
    """Fine-tune the base model on the dataset.

    Args:
        config: the run settings.

    Returns:
        Path to the best weights.

    Raises:
        FileNotFoundError: if training finished without writing best weights.
    """
    YOLO(model=config.base_model).train(
        data=str(config.data_yaml),
        imgsz=config.image_size_px,
        epochs=config.epochs,
        batch=config.batch,
        project=str(config.runs_dir.resolve()),
        name=config.name,
        exist_ok=True,
        patience=20,
        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        momentum=MOMENTUM,
        plots=True,
        verbose=False,
    )
    best = config.runs_dir / config.name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    return best


def count_images(*, data_yaml: Path, split: str) -> int:
    """Count the images of one split listed in an Ultralytics data.yaml.

    Args:
        data_yaml: the dataset file.
        split: ``val`` or ``test``.

    Returns:
        Number of image files in the split's folder (or folders, for a mixed dataset).

    Raises:
        KeyError: if the split is not in the file.
    """
    with data_yaml.open(encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    folders = spec[SPLIT_KEYS[split]]
    folders = folders if isinstance(folders, list) else [folders]
    return sum(1 for folder in folders for p in (Path(spec["path"]) / folder).iterdir()
               if p.suffix.lower() in IMAGE_SUFFIXES)


def score(*, weights: Path, data_yaml: Path, split: str, image_size_px: int) -> Scores:
    """Score a model on one split of the dataset.

    Args:
        weights: model weights.
        data_yaml: the dataset file.
        split: ``val`` or ``test``.
        image_size_px: input size, the same as on the phone.

    Returns:
        Box and keypoint mAP.
    """
    metrics = YOLO(model=str(weights)).val(data=str(data_yaml), split=split, imgsz=image_size_px, plots=False,
                                           verbose=False)
    return Scores(
        split=split,
        images=count_images(data_yaml=data_yaml, split=split),
        box_map50=float(metrics.box.map50),
        box_map50_95=float(metrics.box.map),
        pose_map50=float(metrics.pose.map50),
        pose_map50_95=float(metrics.pose.map),
    )


def export_for_phone(*, weights: Path, image_size_px: int, name: str, out_root: Path) -> Path:
    """Convert weights to ncnn and lay the files out the way the phone app loads them.

    Args:
        weights: model weights.
        image_size_px: input size.
        name: model name on the phone.
        out_root: folder that gets ``<name>/``.

    Returns:
        The folder with ``<name>.param``, ``<name>.bin`` and ``<name>.labels``.
    """
    model = YOLO(model=str(weights))
    exported = Path(model.export(format="ncnn", imgsz=image_size_px, verbose=False))
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported / "model.ncnn.param", out_dir / f"{name}.param")
    shutil.copy(exported / "model.ncnn.bin", out_dir / f"{name}.bin")
    names = model.names
    (out_dir / f"{name}.labels").write_text("".join(names[i] + "\n" for i in sorted(names)), encoding="utf-8")
    return out_dir


def main() -> None:
    """Train, score on val and test, export for the phone, and save the scores as JSON."""
    parser = argparse.ArgumentParser(description="Train and export the phone model.")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics data.yaml")
    parser.add_argument("--name", required=True, help="run and model name, e.g. v1")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE_PX)
    parser.add_argument("--base", default=DEFAULT_BASE_MODEL, help="starting weights")
    parser.add_argument("--weights", type=Path, help="skip training: score and export these weights")
    args = parser.parse_args()
    config = TrainConfig(data_yaml=args.data, name=args.name, base_model=args.base, image_size_px=args.imgsz,
                         epochs=args.epochs, batch=args.batch)
    best = args.weights if args.weights is not None else train(config=config)
    scores = [score(weights=best, data_yaml=config.data_yaml, split=s, image_size_px=config.image_size_px)
              for s in ("val", "test")]
    out_dir = export_for_phone(weights=best, image_size_px=config.image_size_px, name=config.name,
                               out_root=Path("models"))
    report = {"config": {k: str(v) for k, v in asdict(config).items()}, "scores": [asdict(s) for s in scores]}
    (out_dir / "scores.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for s in scores:
        print(f"{s.split}: {s.images} images, box mAP50 {s.box_map50:.3f}, keypoint mAP50 {s.pose_map50:.3f}, "
              f"keypoint mAP50-95 {s.pose_map50_95:.3f}", flush=True)
    print(f"phone files in {out_dir}", flush=True)
    return None


if __name__ == "__main__":
    main()
