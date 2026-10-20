"""Build the v2 training set: every source turned into our four classes, side by side.

Each source lands in ``<out>/<source>/<split>/{images,labels}`` (YOLO segmentation labels), and
``<out>/data.yaml`` lists them all. A source that is already built is kept as it is (so pictures the
hand filter moved out stay out); delete its folder to build it again:

- ``zerowaste``: the plant footage, linked from ``data/zerowaste-960`` (validation and test stay plant-only
  plus ``home`` validation).
- ``home`` and ``trashnet``: our Roboflow export with SAM 3 labels. Home labels lose outlines on the phone
  support and duplicates (several prompts outlining one item). TrashNet pictures that also have human masks
  (Universe ``trash_segmentation_train``) take those instead.
- ``bottle``: Universe ``bottle-bokxa`` (bottles, cans, polystyrene), one copy of each augmented picture.
- ``taco``: TACO litter photos, categories mapped to materials (``TACO_MATERIALS``).
- ``zerowaste-s`` and ``warp``: unlabelled plant pictures (ZeroWaste-s, WaRP-D) with SAM 3 labels from the same
  export, prompted with decoys for paper, cartons, glass and food. WaRP's human boxes set the material of the
  outlines they match.

Plain numpy and OpenCV (no ``train`` group needed). Example:
``uv run python -m door_counter.mix --rf-export data/rf-export --out data/mix-v2``.
"""

import argparse
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import yaml

from door_counter.zerowaste import CLASS_NAMES

DROP = -1  # in a class map: the whole picture is left out (its labels would teach something wrong)
DECOY = -2  # kept through remapping so plant cleaning can drop material outlines of decoy items; never written
OURS = {name: i for i, name in enumerate(CLASS_NAMES)}
CARDBOARD, SOFT, RIGID, METAL = (OURS[n] for n in CLASS_NAMES)
TACO_MATERIALS: dict[str, int | None] = {
    **dict.fromkeys(["Aluminium foil", "Food Can", "Aerosol", "Drink can", "Metal bottle cap", "Metal lid", "Pop tab",
                     "Scrap metal"], METAL),
    **dict.fromkeys(["Toilet tube", "Other carton", "Egg carton", "Corrugated carton", "Meal carton", "Pizza box"],
                    CARDBOARD),
    **dict.fromkeys(["Other plastic bottle", "Clear plastic bottle", "Plastic bottle cap", "Disposable plastic cup",
                     "Other plastic cup", "Foam cup", "Plastic lid", "Spread tub", "Tupperware",
                     "Disposable food container", "Foam food container", "Other plastic container", "Plastic utensils",
                     "Styrofoam piece"], RIGID),
    **dict.fromkeys(["Plastic film", "Six pack rings", "Garbage bag", "Other plastic wrapper", "Single-use carrier bag",
                     "Polypropylene bag", "Crisp packet", "Plastic glooves"], SOFT),
    # not ours, like paper and the belt in ZeroWaste: the object is background
    **dict.fromkeys(["Battery", "Glass bottle", "Broken glass", "Glass cup", "Glass jar", "Food waste", "Magazine paper",
                     "Tissues", "Wrapping paper", "Normal paper", "Paper bag", "Paper cup", "Rope & strings", "Shoe",
                     "Plastic straw", "Paper straw", "Cigarette"], None),
    # mixed or unknown material
    **dict.fromkeys(["Aluminium blister pack", "Carded blister pack", "Drink carton", "Other plastic",
                     "Plastified paper bag", "Squeezable tube", "Unlabeled litter"], DROP),
}
UNIVERSE_MATERIALS: dict[str, int | None] = {
    "cardboard": CARDBOARD, "metal": METAL, "can": METAL, "bottle": RIGID, "polystyrene": RIGID,
    "plastic": RIGID,  # trash_segmentation_train: TrashNet's plastic class, bottles and tubs
    "paper": None, "glass": None,
    "trash": DROP,  # mostly crisp packets and wrappers mixed with other things
    "soft plastic": SOFT, "rigid plastic": RIGID, "decoy": DECOY,
}
SUPPORT_RIGHT_EDGE = 0.14  # the phone support fills the left 12% of the upright home picture
SUPPORT_SHARE = 0.7  # an outline this much on the support is the support, not an item
DUPLICATE_IOU = 0.6
PART_SHARE = 0.8
MAX_CARDBOARD_PIECES = 4  # SAM 3 cuts a TrashNet box close-up into its panels; such pictures are left out
MASK_SIZE_PX = (160, 120)  # outlines are compared on a small raster
MAX_SIDE_PX = 960
JPEG_QUALITY = 90
HOME_NAME = re.compile(r"^\d{13}_\d{8}")  # <session>_<frame> from the capture
TRASHNET_NAME = re.compile(r"^(cardboard|glass|metal|paper|plastic|trash)\d+")
ZEROWASTE_S_NAME = re.compile(r"^\d\d_frame_\d{6}$")
DECOY_IOU = 0.5  # a material outline overlapping a decoy this much is the same item, and SAM 3 was unsure
HUMAN_BOX_IOU = 0.5  # an outline's box matching a WaRP box this much is that item
MAX_PLANT_BOX_SHARE = 0.5  # SAM 3 sometimes outlines the whole belt as a "plastic tray"
ROBOFLOW_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.IGNORECASE)


def source_stem(*, name: str) -> str:
    """Strip Roboflow's export suffix: ``cardboard109_jpg.rf.27fa.jpg`` -> ``cardboard109``.

    Args:
        name: the exported file name.

    Returns:
        The name the picture had when it was uploaded.
    """
    return ROBOFLOW_SUFFIX.sub("", Path(name).stem)


def class_map_from_names(*, names: Sequence[str]) -> dict[int, int | None]:
    """Map a dataset's class indices to ours by name.

    Args:
        names: the dataset's class names, in index order.

    Returns:
        Source index -> our index, None (background) or ``DROP``.

    Raises:
        KeyError: if a name is not in ``UNIVERSE_MATERIALS``.
    """
    return {i: UNIVERSE_MATERIALS[name.strip().lower()] for i, name in enumerate(names)}


def remap_lines(*, lines: Iterable[str], class_map: Mapping[int, int | None]) -> list[str] | None:
    """Rewrite YOLO label lines with our class indices.

    Args:
        lines: label lines, class index first.
        class_map: source index -> our index, None (drop the object) or ``DROP`` (drop the picture).

    Returns:
        The new lines, or None if the picture must be left out.
    """
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7:  # a class and at least three points
            continue
        ours = class_map[int(parts[0])]
        if ours == DROP:
            return None
        if ours is not None:
            out.append(" ".join([str(ours), *parts[1:]]))
    return out


def outline_mask(*, line: str) -> npt.NDArray[np.bool_]:
    """Rasterise one label line's outline on the small comparison grid.

    Args:
        line: a YOLO segmentation label line.

    Returns:
        The filled outline, ``MASK_SIZE_PX`` wide and high.
    """
    width, height = MASK_SIZE_PX
    xy = (np.asarray(line.split()[1:], dtype=np.float64).reshape(-1, 2) * (width, height)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [xy], 1)
    return mask.astype(bool)


def drop_duplicates(*, lines: Sequence[str]) -> list[str]:
    """Keep one outline per item: SAM 3 often outlines an item twice, or a part of it again.

    Several prompts can outline the same item (overlap above ``DUPLICATE_IOU``), and "metal lid" outlines
    a can's lid inside the outline of the whole can (``PART_SHARE`` of the smaller one inside the larger).

    Args:
        lines: SAM 3 label lines of one picture.

    Returns:
        The lines to keep, largest outline first.
    """
    masks = [(line, outline_mask(line=line)) for line in lines]
    kept: list[tuple[str, npt.NDArray[np.bool_]]] = []
    for line, mask in sorted(((line, m) for line, m in masks if m.any()), key=lambda lm: -int(lm[1].sum())):
        same_item = any(line.split()[0] == k.split()[0] and ((mask & km).sum() > DUPLICATE_IOU * (mask | km).sum()
                                                             or (mask & km).sum() > PART_SHARE * mask.sum())
                        for k, km in kept)
        if not same_item:
            kept.append((line, mask))
    return [line for line, _ in kept]


def warp_material(*, name: str) -> int | None:
    """Our material for a WaRP category.

    Args:
        name: a WaRP class name, e.g. ``bottle-blue5l-full``.

    Returns:
        Our class index, or None for drink cartons and glass (not ours).
    """
    if name == "cans":
        return METAL
    if name == "detergent-box":
        return CARDBOARD
    if name.endswith("-cardboard") or name.startswith("glass-"):  # juice and milk cartons, glass bottles
        return None
    return RIGID  # plastic bottles, detergent bottles, canisters


Box = tuple[float, float, float, float]  # left, top, right, bottom, 0 to 1


def read_warp_boxes(*, label: Path, names: Sequence[str]) -> list[tuple[int | None, Box]]:
    """Read one WaRP-D label file (YOLO boxes).

    Args:
        label: the ``.txt`` file.
        names: WaRP's class names, in index order.

    Returns:
        Our material (or None) and the box, per labelled item.
    """
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            x, y, w, h = (float(v) for v in parts[1:])
            boxes.append((warp_material(name=names[int(parts[0])]), (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return boxes


def box_iou(*, a: Box, b: Box) -> float:
    """Overlap of two boxes, intersection over union."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clean_plant_lines(*, lines: Sequence[str], human_boxes: Sequence[tuple[int | None, Box]] = ()) -> list[str]:
    """Clean SAM 3 labels of a plant picture prompted with decoys.

    Material outlines that overlap a decoy outline go (SAM 3 was unsure what the item is), and so do the decoys
    and outlines whose box covers more than ``MAX_PLANT_BOX_SHARE`` of the picture (the belt).
    Where a human box matches an outline, the box's material wins; None removes the outline. Then duplicates.

    Args:
        lines: remapped label lines, decoys still in (class ``DECOY``).
        human_boxes: WaRP's labelled items, if any.

    Returns:
        The lines to keep, largest outline first.
    """
    decoys = [outline_mask(line=line) for line in lines if int(line.split()[0]) == DECOY]
    kept = []
    for line in lines:
        parts = line.split()
        if int(parts[0]) == DECOY:
            continue
        mask = outline_mask(line=line)
        if any((mask & d).sum() > DECOY_IOU * (mask | d).sum() for d in decoys):
            continue
        xy = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
        box: Box = (*xy.min(axis=0), *xy.max(axis=0))
        if (box[2] - box[0]) * (box[3] - box[1]) > MAX_PLANT_BOX_SHARE:
            continue
        match = max(human_boxes, key=lambda hb: box_iou(a=box, b=hb[1]), default=None)
        if match is not None and box_iou(a=box, b=match[1]) >= HUMAN_BOX_IOU:
            if match[0] is None:
                continue
            line = " ".join([str(match[0]), *parts[1:]])
        kept.append(line)
    return drop_duplicates(lines=kept)


def clean_home_lines(*, lines: Sequence[str]) -> list[str]:
    """Drop outlines on the phone support, then duplicates and parts of the same item.

    Args:
        lines: SAM 3 label lines of one home picture.

    Returns:
        The lines to keep, largest outline first.
    """
    support = np.zeros((MASK_SIZE_PX[1], MASK_SIZE_PX[0]), dtype=bool)
    support[:, : round(SUPPORT_RIGHT_EDGE * MASK_SIZE_PX[0])] = True
    off_support = [line for line in lines
                   if (m := outline_mask(line=line)).any() and (m & support).sum() < SUPPORT_SHARE * m.sum()]
    return drop_duplicates(lines=off_support)


def link_or_copy(*, source: Path, target: Path) -> None:
    """Hard-link a file (same disk, no extra space), copying when linking is not possible.

    Args:
        source: the existing file.
        target: the new name.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        target.write_bytes(source.read_bytes())
    return None


def copy_split(*, images_dir: Path, out_dir: Path, class_map: Mapping[int, int | None],
               keep: Callable[[Path], bool] = lambda _: True,
               clean: Callable[[Sequence[str], Path], list[str] | None] | None = None) -> int:
    """Copy one YOLO split (``images/`` next to ``labels/``) with our classes.

    Args:
        images_dir: the split's ``images`` folder.
        out_dir: the target split folder; gets ``images/`` and ``labels/``.
        class_map: source class index -> ours.
        keep: which pictures to take.
        clean: optional extra pass over each picture's remapped lines (and its path); None leaves it out.

    Returns:
        Pictures written.
    """
    written = 0
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png") or not keep(image):
            continue
        label = images_dir.parent / "labels" / f"{image.stem}.txt"
        source_lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
        lines = remap_lines(lines=source_lines, class_map=class_map)
        if lines is None:
            continue
        if clean is not None:
            lines = clean(lines, image)
            if lines is None:
                continue
        link_or_copy(source=image, target=out_dir / "images" / image.name)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / f"{image.stem}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        written += 1
    return written


def read_taco_image(*, path: Path, width_px: int, height_px: int) -> npt.NDArray[np.uint8] | None:
    """Load a TACO photo the way its annotations see it.

    TACO's outlines follow the photo's orientation tag for most pictures, but not all; the size in
    the annotation file tells which.

    Args:
        path: the photo.
        width_px: width in the annotation file.
        height_px: height in the annotation file.

    Returns:
        The pixels, or None if neither orientation matches.
    """
    for flags in (cv2.IMREAD_COLOR, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION):
        pixels = cv2.imread(str(path), flags)
        if pixels is not None and pixels.shape[:2] == (height_px, width_px):
            return pixels.astype(np.uint8)
    return None


def convert_taco(*, taco_data: Path, out_dir: Path, valid_every: int = 10) -> tuple[int, int]:
    """Convert TACO (COCO polygons, 60 categories) to our classes, long side ``MAX_SIDE_PX``.

    Args:
        taco_data: the ``TACO/data`` folder with ``annotations.json`` and ``batch_*``.
        out_dir: gets ``train/`` and ``valid/`` with ``images/`` and ``labels/``.
        valid_every: every n-th kept picture goes to validation.

    Returns:
        Pictures written, and pictures left out (mixed material, or the size does not match).

    Raises:
        KeyError: if TACO has a category ``TACO_MATERIALS`` does not know.
    """
    with (taco_data / "annotations.json").open(encoding="utf-8") as f:
        coco = json.load(f)
    material = {int(c["id"]): TACO_MATERIALS[str(c["name"])] for c in coco["categories"]}
    per_image: dict[int, list[dict[str, Any]]] = {}  # parsed JSON
    for ann in coco["annotations"]:
        per_image.setdefault(int(ann["image_id"]), []).append(ann)
    written = skipped = 0
    for image in coco["images"]:
        anns = per_image.get(int(image["id"]), [])
        if any(material[int(a["category_id"])] == DROP for a in anns):
            skipped += 1
            continue
        width_px, height_px = int(image["width"]), int(image["height"])
        pixels = read_taco_image(path=taco_data / str(image["file_name"]), width_px=width_px, height_px=height_px)
        if pixels is None:
            skipped += 1
            continue
        scale = MAX_SIDE_PX / max(width_px, height_px)
        split = "valid" if written % valid_every == valid_every - 1 else "train"
        stem = "taco_" + str(image["file_name"]).replace("/", "_").rsplit(".", 1)[0]
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / split / "images" / f"{stem}.jpg"),
                    cv2.resize(pixels, (round(width_px * scale), round(height_px * scale)), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        lines = []
        for ann in anns:
            ours = material[int(ann["category_id"])]
            segmentation = ann["segmentation"]
            if ours is None or not isinstance(segmentation, list) or not segmentation:
                continue
            piece = max(segmentation, key=len)  # YOLO holds one outline per object
            xy = np.clip(np.asarray(piece, dtype=np.float64).reshape(-1, 2) / (width_px, height_px), 0.0, 1.0)
            if len(xy) >= 3:
                lines.append(f"{ours} " + " ".join(f"{v:.6f}" for v in xy.ravel()))
        (out_dir / split / "labels" / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        written += 1
    return written, skipped


def write_mix_yaml(*, out_root: Path, train: Sequence[Path], val: Sequence[Path], test: Sequence[Path]) -> None:
    """Write the Ultralytics dataset file listing every source's image folders.

    Args:
        out_root: the mix folder.
        train: training image folders.
        val: validation image folders.
        test: test image folders.
    """
    data = {"path": str(out_root.resolve()), "train": [str(p.resolve()) for p in train],
            "val": [str(p.resolve()) for p in val], "test": [str(p.resolve()) for p in test],
            "names": dict(enumerate(CLASS_NAMES))}
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return None


def main() -> None:
    """Build every source that is given and write ``data.yaml``."""
    parser = argparse.ArgumentParser(description="Build the v2 training set from all sources.")
    parser.add_argument("--zerowaste", type=Path, default=Path("data/zerowaste-960"))
    parser.add_argument("--rf-export", type=Path, required=True, help="our Roboflow version export (YOLO format)")
    parser.add_argument("--trash-seg", type=Path, default=Path("data/universe/trash_segmentation_train"))
    parser.add_argument("--bottle", type=Path, default=Path("data/universe/bottle-bokxa"))
    parser.add_argument("--taco", type=Path, default=Path("data/taco/TACO/data"))
    parser.add_argument("--warp", type=Path, default=Path("data/warp/Warp-D"), help="WaRP-D, for its human boxes")
    parser.add_argument("--uploads", type=Path, action="append",
                        help="folders of the plant pictures as uploaded (default: data/upload-plant, -zws, -probe)")
    parser.add_argument("--synth", type=Path, help="copy-paste scenes (door_counter.copy_paste output)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: Path = args.out
    counts: dict[str, int] = {}
    train: list[Path] = [args.zerowaste / "train" / "images"]
    val: list[Path] = [args.zerowaste / "valid" / "images"]

    human = {source_stem(name=p.name) for p in (args.trash_seg / "train" / "images").glob("*.jpg")}
    names = yaml.safe_load((args.trash_seg / "data.yaml").read_text(encoding="utf-8"))["names"]
    if not (out / "trashnet-human").is_dir():
        counts["trashnet human"] = copy_split(images_dir=args.trash_seg / "train" / "images",
                                              out_dir=out / "trashnet-human" / "train",
                                              class_map=class_map_from_names(names=names))
    train.append(out / "trashnet-human" / "train" / "images")

    names = yaml.safe_load((args.rf_export / "data.yaml").read_text(encoding="utf-8"))["names"]
    ours = class_map_from_names(names=names)
    for split in ("train", "valid"):
        images_dir = args.rf_export / split / "images"
        if not images_dir.is_dir() or (out / "home" / split).is_dir():
            continue  # trashnet-sam3 is built together with home: delete both folders to redo
        counts[f"home {split}"] = copy_split(
            images_dir=images_dir, out_dir=out / "home" / split, class_map=ours,
            keep=lambda p: HOME_NAME.match(source_stem(name=p.name)) is not None,
            clean=lambda lines, _: clean_home_lines(lines=lines))
        counts[f"trashnet sam3 {split}"] = copy_split(
            images_dir=images_dir, out_dir=out / "trashnet-sam3" / "train", class_map=ours,
            keep=lambda p: (TRASHNET_NAME.match(source_stem(name=p.name)) is not None
                            and source_stem(name=p.name) not in human),
            clean=lambda lines, _: None if sum(line.split()[0] == str(CARDBOARD) for line in lines) > MAX_CARDBOARD_PIECES
            else drop_duplicates(lines=lines))
    train += [out / "home" / "train" / "images", out / "trashnet-sam3" / "train" / "images"]
    val.append(out / "home" / "valid" / "images")

    names = yaml.safe_load((args.bottle / "data.yaml").read_text(encoding="utf-8"))["names"]
    seen: set[str] = set()

    def first_copy(path: Path) -> bool:
        stem = source_stem(name=path.name)
        if stem in seen:
            return False
        seen.add(stem)
        return True

    if not (out / "bottle").is_dir():
        counts["bottle"] = copy_split(images_dir=args.bottle / "train" / "images", out_dir=out / "bottle" / "train",
                                      class_map=class_map_from_names(names=names), keep=first_copy)
    train.append(out / "bottle" / "train" / "images")

    if not (out / "taco").is_dir():  # slow (full-size photos)
        counts["taco"], counts["taco left out"] = convert_taco(taco_data=args.taco, out_dir=out / "taco")
    train.append(out / "taco" / "train" / "images")
    val.append(out / "taco" / "valid" / "images")

    upload_dirs = args.uploads or [Path("data/upload-plant"), Path("data/upload-zws"), Path("data/upload-probe")]
    uploaded = {p.stem for d in upload_dirs for p in d.rglob("*.jpg")}  # ZeroWaste-f shares the name pattern
    warp_names = (args.warp / "classes.txt").read_text(encoding="utf-8").split()
    warp_labels = {p.stem: p for split in ("train", "test") for p in (args.warp / split / "labels").glob("*.txt")}
    build_plant = not (out / "zerowaste-s").is_dir()  # zerowaste-s and warp are built together: delete both to redo
    for split in ("train", "valid"):
        images_dir = args.rf_export / split / "images"
        if not build_plant or not images_dir.is_dir():
            continue
        counts[f"zerowaste-s {split}"] = copy_split(
            images_dir=images_dir, out_dir=out / "zerowaste-s" / "train", class_map=ours,
            keep=lambda p: (stem := source_stem(name=p.name)) in uploaded and ZEROWASTE_S_NAME.match(stem) is not None,
            clean=lambda lines, _: clean_plant_lines(lines=lines))
        counts[f"warp {split}"] = copy_split(
            images_dir=images_dir, out_dir=out / "warp" / "train", class_map=ours,
            keep=lambda p: (stem := source_stem(name=p.name)) in uploaded and stem in warp_labels,
            clean=lambda lines, image: clean_plant_lines(lines=lines, human_boxes=read_warp_boxes(
                label=warp_labels[source_stem(name=image.name)], names=warp_names)))
    train += [d / "train" / "images" for d in (out / "zerowaste-s", out / "warp") if d.is_dir()]

    if args.synth is not None:
        train.append(args.synth / "images")
    write_mix_yaml(out_root=out, train=train, val=val, test=[args.zerowaste / "test" / "images"])
    for name, n in counts.items():
        print(f"{name}: {n}")
    return None


if __name__ == "__main__":
    main()
