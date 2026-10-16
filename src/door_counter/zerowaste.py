"""Convert ZeroWaste-f (COCO instance masks, 1920x1080 PNG) into a smaller copy for training and upload.

For each split it writes JPEGs at a smaller width plus the masks as polygons in two formats:
``_annotations.coco.json`` (what Roboflow imports) and YOLO segmentation labels (what Ultralytics
trains on), and a ``data.yaml`` at the top. Class names follow ``CLASS_NAMES``, whatever order the
source uses.

Run: ``uv run python -m door_counter.zerowaste --source data/zerowaste/zerowaste-f-final
--out data/zerowaste-960``.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ("cardboard", "soft plastic", "rigid plastic", "metal")
SPLITS = {"train": "train", "val": "valid", "test": "test"}  # source name -> Roboflow/Ultralytics name
DEFAULT_WIDTH_PX = 960
JPEG_QUALITY = 90
MIN_POLYGON_POINTS = 3


@dataclass(frozen=True)
class CocoImage:
    """One image entry of a COCO file."""

    image_id: int
    file_name: str
    width_px: int
    height_px: int


@dataclass(frozen=True)
class Polygon:
    """One outline of an object, in source pixels: x0, y0, x1, y1, ..."""

    class_index: int
    points: tuple[float, ...]


@dataclass(frozen=True)
class SplitStats:
    """What a converted split contains."""

    images: int
    polygons: int
    polygons_per_class: tuple[int, ...]


def normalise_class_name(*, name: str) -> str:
    """Map a source class name like ``soft_plastic`` to ours (``soft plastic``).

    Args:
        name: the name in the source file.

    Returns:
        Our name.

    Raises:
        ValueError: if it is not one of our classes.
    """
    ours = name.strip().lower().replace("_", " ").replace("-", " ")
    if ours not in CLASS_NAMES:
        raise ValueError(f"unknown class {name!r}, expected one of {CLASS_NAMES}")
    return ours


def read_coco(*, labels_path: Path) -> tuple[list[CocoImage], dict[int, list[Polygon]]]:
    """Read a COCO instance-segmentation file.

    Args:
        labels_path: the JSON file.

    Returns:
        The images, and each image id's polygons (with our class indices).

    Raises:
        ValueError: if an annotation uses RLE instead of polygons, or a class is unknown.
    """
    with labels_path.open(encoding="utf-8") as f:
        coco = json.load(f)
    class_index = {
        int(c["id"]): CLASS_NAMES.index(normalise_class_name(name=str(c["name"]))) for c in coco["categories"]
    }
    images = [
        CocoImage(
            image_id=int(im["id"]), file_name=str(im["file_name"]), width_px=int(im["width"]), height_px=int(im["height"])
        )
        for im in coco["images"]
    ]
    polygons: dict[int, list[Polygon]] = defaultdict(list)
    for ann in coco["annotations"]:
        segmentation = ann["segmentation"]
        if not isinstance(segmentation, list):
            raise ValueError(f"annotation {ann['id']} uses RLE; only polygons are supported")
        for outline in segmentation:
            if len(outline) >= 2 * MIN_POLYGON_POINTS:
                polygons[int(ann["image_id"])].append(
                    Polygon(class_index=class_index[int(ann["category_id"])], points=tuple(float(v) for v in outline))
                )
    return images, polygons


def yolo_label_lines(*, polygons: Sequence[Polygon], width_px: int, height_px: int) -> Iterator[str]:
    """Yield YOLO segmentation label lines: class index, then x y pairs in 0..1.

    Args:
        polygons: the image's outlines in source pixels.
        width_px: source image width.
        height_px: source image height.

    Returns:
        A generator of label lines.
    """
    for polygon in polygons:
        xy = np.asarray(polygon.points, dtype=np.float64).reshape(-1, 2)
        xy = np.clip(xy / (width_px, height_px), 0.0, 1.0)
        yield f"{polygon.class_index} " + " ".join(f"{v:.6f}" for v in xy.ravel())
    return None


def convert_split(*, source_dir: Path, out_dir: Path, width_px: int) -> SplitStats:
    """Convert one ZeroWaste split (``data/`` images + ``labels.json``).

    Args:
        source_dir: the split folder in the extracted ZeroWaste-f.
        out_dir: where to write ``images/``, ``labels/`` and ``_annotations.coco.json``.
        width_px: target width; height follows the aspect ratio.

    Returns:
        Counts of what was written.

    Raises:
        FileNotFoundError: if an image listed in the labels is missing.
    """
    images, polygons = read_coco(labels_path=source_dir / "labels.json")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    coco_images: list[dict[str, object]] = []
    coco_annotations: list[dict[str, object]] = []
    per_class = [0] * len(CLASS_NAMES)
    for image in images:
        source_path = source_dir / "data" / Path(image.file_name).name
        pixels = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if pixels is None:
            raise FileNotFoundError(source_path)
        scale = width_px / image.width_px
        height_px = round(image.height_px * scale)
        stem = Path(image.file_name).stem
        cv2.imwrite(
            str(out_dir / "images" / f"{stem}.jpg"),
            cv2.resize(pixels, (width_px, height_px), interpolation=cv2.INTER_AREA),
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        own = polygons.get(image.image_id, [])
        lines = yolo_label_lines(polygons=own, width_px=image.width_px, height_px=image.height_px)
        (out_dir / "labels" / f"{stem}.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        coco_images.append({"id": image.image_id, "file_name": f"{stem}.jpg", "width": width_px, "height": height_px})
        for polygon in own:
            per_class[polygon.class_index] += 1
            coco_annotations.append(_scaled_annotation(polygon=polygon, image_id=image.image_id, scale=scale,
                                                       annotation_id=len(coco_annotations) + 1))
    categories = [{"id": i + 1, "name": name, "supercategory": "material"} for i, name in enumerate(CLASS_NAMES)]
    with (out_dir / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump({"images": coco_images, "annotations": coco_annotations, "categories": categories}, f)
    return SplitStats(images=len(images), polygons=len(coco_annotations), polygons_per_class=tuple(per_class))


def _scaled_annotation(*, polygon: Polygon, image_id: int, scale: float, annotation_id: int) -> dict[str, object]:
    """One COCO annotation for the resized image.

    Args:
        polygon: the outline in source pixels.
        image_id: the image it belongs to.
        scale: source-to-target scale factor.
        annotation_id: id for the new annotation.

    Returns:
        The COCO annotation dict (category ids start at 1).
    """
    xy = np.asarray(polygon.points, dtype=np.float64).reshape(-1, 2) * scale
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": polygon.class_index + 1,
        "segmentation": [[round(float(v), 2) for v in xy.ravel()]],
        "bbox": [round(float(v), 2) for v in (x_min, y_min, x_max - x_min, y_max - y_min)],
        "area": round(float(cv2.contourArea(xy.astype(np.float32))), 2),
        "iscrowd": 0,
    }


def write_data_yaml(*, out_root: Path, names: Mapping[int, str]) -> None:
    """Write the Ultralytics dataset file.

    Args:
        out_root: the converted dataset folder.
        names: class index -> name.
    """
    lines = [f"path: {out_root.resolve()}", "train: train/images", "val: valid/images", "test: test/images", "names:"]
    lines += [f"  {i}: {name}" for i, name in sorted(names.items())]
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return None


def main() -> None:
    """Convert all three splits and print a summary."""
    parser = argparse.ArgumentParser(description="Convert ZeroWaste-f for training and upload.")
    parser.add_argument("--source", type=Path, required=True, help="extracted zerowaste-f-final folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH_PX, help="target width in pixels")
    args = parser.parse_args()
    for source_name, out_name in SPLITS.items():
        stats = convert_split(source_dir=args.source / source_name, out_dir=args.out / out_name, width_px=args.width)
        counts = ", ".join(f"{n} {c}" for n, c in zip(CLASS_NAMES, stats.polygons_per_class, strict=True))
        print(f"{out_name}: {stats.images} images, {stats.polygons} outlines ({counts})", flush=True)
    write_data_yaml(out_root=args.out, names=dict(enumerate(CLASS_NAMES)))
    return None


if __name__ == "__main__":
    main()
