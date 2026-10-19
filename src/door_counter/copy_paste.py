"""Make training scenes by pasting cut-out items onto empty belt pictures (copy-paste augmentation).

Cut-outs come from labelled pictures (home SAM 3 outlines, TrashNet masks): each outline cuts its
item out. Each scene takes one empty home picture and pastes a few items at random size, angle and
place, with a soft shadow. Items pasted later cover earlier ones, whose outlines shrink to what stays
visible. Materials are drawn evenly. Labels are written from the visible masks.

Example: ``uv run python -m door_counter.copy_paste --cut-outs data/mix-v2/home/train/images
--cut-outs data/mix-v2/trashnet-human/train/images --backgrounds data/captures --out data/synth --scenes 2000``.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from door_counter.zerowaste import CLASS_NAMES

SCENE_SIZE_PX = (640, 480)  # the phone's upright preview
ITEMS_PER_SCENE = (2, 6)
ITEM_SIZE_SHARE = (0.12, 0.45)  # longest side of a pasted item, as a share of the scene width
MAX_ITEM_AREA_SHARE = 0.5  # bigger outlines fill their picture (close-ups) and make poor cut-outs
MIN_ITEM_SIDE_PX = 24
EDGE_MARGIN_PX = 2  # an outline touching the picture edge is a cut-off item
MIN_VISIBLE_SHARE = 0.15  # a covered item keeps its label while this much of it shows
BACKGROUND_MAX_ITEM_PIXELS = 0.003  # a background picture may differ from plain cloth in this share of pixels
SUPPORT_WIDTH_PX = 80  # the phone support on the left of the upright home picture
SHADOW_OFFSET_PX = 6
SHADOW_STRENGTH = 0.35
JPEG_QUALITY = 90


@dataclass(frozen=True)
class CutOut:
    """One item cut out of a labelled picture."""

    pixels: npt.NDArray[np.uint8]  # h x w x 3, BGR
    mask: npt.NDArray[np.uint8]  # h x w, 0 or 255
    class_index: int


def cut_outs_from(*, images_dir: Path) -> list[CutOut]:
    """Cut every whole, not too large item out of a labelled YOLO split.

    Args:
        images_dir: an ``images`` folder next to a ``labels`` folder.

    Returns:
        The cut-outs.
    """
    cut_outs = []
    for image in sorted(images_dir.glob("*.jpg")):
        label = images_dir.parent / "labels" / f"{image.stem}.txt"
        lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
        if not lines:
            continue
        loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if loaded is None:
            continue
        pixels = np.asarray(loaded, dtype=np.uint8)
        height, width = pixels.shape[:2]
        for line in lines:
            parts = line.split()
            xy = (np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2) * (width, height)).astype(np.int32)
            x, y, w, h = cv2.boundingRect(xy)
            touches_edge = x < EDGE_MARGIN_PX or y < EDGE_MARGIN_PX or x + w > width - EDGE_MARGIN_PX \
                or y + h > height - EDGE_MARGIN_PX
            if touches_edge or min(w, h) < MIN_ITEM_SIDE_PX or w * h > MAX_ITEM_AREA_SHARE * width * height:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [xy - (x, y)], 255)
            mask = np.asarray(cv2.erode(mask, np.ones((3, 3), np.uint8)), dtype=np.uint8)  # lose the background halo
            cut_outs.append(CutOut(pixels=pixels[y:y + h, x:x + w].copy(), mask=mask, class_index=int(parts[0])))
    return cut_outs


def is_empty_cloth(*, pixels: npt.NDArray[np.uint8]) -> bool:
    """Tell whether a home picture shows only cloth (and the support), so it can be a background.

    Args:
        pixels: an upright home picture.

    Returns:
        True if almost no pixel stands out from the cloth in brightness or colour.
    """
    cloth = pixels[:, SUPPORT_WIDTH_PX:]
    gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY).astype(np.int16)
    saturation = cv2.cvtColor(cloth, cv2.COLOR_BGR2HSV)[..., 1]
    bright = np.mean(np.abs(gray - int(np.median(gray))) > 30)
    coloured = np.mean(saturation > 60)
    return bool(bright < BACKGROUND_MAX_ITEM_PIXELS and coloured < BACKGROUND_MAX_ITEM_PIXELS)


def transformed(*, cut: CutOut, longest_side_px: int, angle_deg: float, flip: bool,
                brightness: float) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Resize, rotate, mirror and brighten a cut-out.

    Args:
        cut: the cut-out.
        longest_side_px: target length of its longest side, before rotating.
        angle_deg: rotation.
        flip: mirror left-right first.
        brightness: pixel scale factor.

    Returns:
        Pixels and mask on a canvas big enough for any rotation.
    """
    scale = longest_side_px / max(cut.mask.shape)
    pixels = cv2.resize(cut.pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(cut.mask, (pixels.shape[1], pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
    if flip:
        pixels, mask = pixels[:, ::-1], mask[:, ::-1]
    h, w = mask.shape
    side = int(np.ceil(np.hypot(h, w)))
    turn = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    turn[:, 2] += ((side - w) / 2, (side - h) / 2)
    pixels = cv2.warpAffine(np.ascontiguousarray(pixels), turn, (side, side), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), turn, (side, side), flags=cv2.INTER_NEAREST)
    pixels = np.clip(pixels.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return pixels, np.asarray(mask, dtype=np.uint8)


def mask_to_line(*, mask: npt.NDArray[np.bool_], class_index: int) -> str | None:
    """Turn a visible mask into a YOLO label line (largest connected part).

    Args:
        mask: the item's visible pixels in the scene.
        class_index: its class.

    Returns:
        The label line, or None if nothing is left.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    if len(outline) < 3:
        return None
    xy = outline / (mask.shape[1], mask.shape[0])
    return f"{class_index} " + " ".join(f"{v:.6f}" for v in xy.ravel())


def make_scene(*, background: npt.NDArray[np.uint8], cut_outs: Sequence[Sequence[CutOut]],
               rng: np.random.Generator) -> tuple[npt.NDArray[np.uint8], list[str]]:
    """Paste a few items onto a background.

    Args:
        background: an empty cloth picture, ``SCENE_SIZE_PX``.
        cut_outs: cut-outs grouped by class (a class may have none).
        rng: random source.

    Returns:
        The scene and its label lines.
    """
    scene = background.astype(np.float32)
    height, width = scene.shape[:2]
    classes = [c for c, group in enumerate(cut_outs) if group]
    placed: list[tuple[npt.NDArray[np.bool_], int, int]] = []  # visible mask, class, pixels when pasted
    for _ in range(int(rng.integers(ITEMS_PER_SCENE[0], ITEMS_PER_SCENE[1] + 1))):
        group = cut_outs[classes[int(rng.integers(len(classes)))]]
        cut = group[int(rng.integers(len(group)))]
        pixels, mask = transformed(cut=cut, longest_side_px=int(rng.uniform(*ITEM_SIZE_SHARE) * width),
                                   angle_deg=float(rng.uniform(0, 360)), flip=bool(rng.integers(2)),
                                   brightness=float(rng.uniform(0.8, 1.15)))
        side = mask.shape[0]
        if side >= min(width, height):
            continue
        x = int(rng.integers(SUPPORT_WIDTH_PX - side // 4, width - side * 3 // 4))  # may stick out of the picture
        y = int(rng.integers(-side // 4, height - side * 3 // 4))
        full = np.zeros((height, width), dtype=np.float32)
        colour = np.zeros_like(scene)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + side, width), min(y + side, height)
        full[y0:y1, x0:x1] = mask[y0 - y:y1 - y, x0 - x:x1 - x] / 255.0
        colour[y0:y1, x0:x1] = pixels[y0 - y:y1 - y, x0 - x:x1 - x]
        shifted = np.zeros_like(full)
        shifted[SHADOW_OFFSET_PX:, SHADOW_OFFSET_PX:] = full[:-SHADOW_OFFSET_PX, :-SHADOW_OFFSET_PX]
        shadow = cv2.GaussianBlur(shifted, (21, 21), 0)
        scene *= 1.0 - SHADOW_STRENGTH * shadow[..., None]
        alpha = cv2.GaussianBlur(full, (3, 3), 0)[..., None]  # soften the pasted edge
        scene = scene * (1.0 - alpha) + colour * alpha
        item = full > 0.5
        if not item.any():
            continue
        placed = [(earlier & ~item, c, area) for earlier, c, area in placed]
        placed.append((item, cut.class_index, int(item.sum())))
    lines = []
    for visible, class_index, area in placed:
        shows = visible.sum() >= MIN_VISIBLE_SHARE * area
        line = mask_to_line(mask=visible, class_index=class_index) if shows else None
        if line is not None:
            lines.append(line)
    return np.clip(scene, 0, 255).astype(np.uint8), lines


def main() -> None:
    """Cut items out, find empty backgrounds, write the scenes."""
    parser = argparse.ArgumentParser(description="Paste cut-out items onto empty belt pictures.")
    parser.add_argument("--cut-outs", type=Path, action="append", required=True, help="labelled images folder")
    parser.add_argument("--backgrounds", type=Path, required=True, help="capture folder, one subfolder per clip")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    by_class: list[list[CutOut]] = [[] for _ in CLASS_NAMES]
    for folder in args.cut_outs:
        for cut in cut_outs_from(images_dir=folder):
            by_class[cut.class_index].append(cut)
    print("cut-outs per class:", dict(zip(CLASS_NAMES, map(len, by_class), strict=True)))
    rng = np.random.default_rng(args.seed)
    candidates = sorted(p for p in args.backgrounds.glob("*-*/*.jpg") if not p.parent.name.startswith("test-"))
    candidates = [candidates[i] for i in rng.permutation(len(candidates))[:1500]]
    backgrounds = []
    for path in candidates:
        loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        pixels = None if loaded is None else np.asarray(loaded, dtype=np.uint8)
        if pixels is not None and pixels.shape[:2] == SCENE_SIZE_PX[::-1] and is_empty_cloth(pixels=pixels):
            backgrounds.append(pixels)
    print(f"backgrounds: {len(backgrounds)} of {len(candidates)} pictures checked")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    for n in range(args.scenes):
        scene, lines = make_scene(background=backgrounds[int(rng.integers(len(backgrounds)))], cut_outs=by_class,
                                  rng=rng)
        cv2.imwrite(str(args.out / "images" / f"synth_{n:05d}.jpg"), scene, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        (args.out / "labels" / f"synth_{n:05d}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    print(f"scenes: {args.scenes} in {args.out}")
    return None


if __name__ == "__main__":
    main()
