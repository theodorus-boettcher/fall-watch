"""Turn CAUCAFall into a posture + keypoint dataset: COCO keypoints for Roboflow, YOLO pose labels for training.

CAUCAFall (CC BY 4.0) has 10 people doing 10 activities each in one room, filmed from a wall corner at
2.15 m. Every frame has one person box and a class: ``fall`` when the body is on the ground after a fall,
``nofall`` otherwise. That maps to our ``lying`` / ``not lying``. There are no keypoints, so a big pose
model, RF-DETR Keypoint, labels them: its person that overlaps the CAUCAFall box by ``MATCH_IOU`` or more
gives the keypoints. With no match, the box keeps its class and its keypoints stay "not labelled".

Frames come at 23 per second and are near copies, so one in ``EVERY`` is kept. The split goes by person
(``SPLIT_OF_SUBJECT``) so the same person is never in two splits. Pictures are scaled to ``SIDE_PX``.

Needs the ``teacher`` dependency group. Example:
``uv run --group teacher python -m door_counter.fall_data --caucafall data/caucafall/raw --out data/fall-coco``.
"""

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from door_counter.yolo_pose import KEYPOINT_COUNT, Rect, box_iou

CLASS_NAMES = ("not lying", "lying")  # our class ids 0 and 1; CAUCAFall's 0 = nofall, 1 = fall
KEYPOINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                  "right_knee", "left_ankle", "right_ankle")
SKELETON = ((16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7), (6, 8), (7, 9), (8, 10),
            (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7))  # COCO, 1-based
FLIP_INDEX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)  # left <-> right for Ultralytics
EVERY = 4
SIDE_PX = 640
JPEG_QUALITY = 90
MATCH_IOU = 0.5
TEACHER_MIN_SCORE = 0.3
KEYPOINT_MIN_CONFIDENCE = 0.5
SPLIT_OF_SUBJECT = {**{s: "train" for s in range(1, 8)}, 8: "valid", 9: "test", 10: "test"}
SUBJECT_NUMBER = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Labelled:
    """One kept frame: its picture, CAUCAFall box (0..1) and our class."""

    image: Path
    box: Rect
    class_id: int


@dataclass(frozen=True)
class TeacherPerson:
    """One person from the teacher: box and keypoints in 0..1, keypoint confidences."""

    box: Rect
    xy: npt.NDArray[np.float64]  # (17, 2)
    confidence: npt.NDArray[np.float64]  # (17,)


def read_yolo_box(*, label: Path) -> tuple[int, Rect] | None:
    """Read the first box of a YOLO label file (class cx cy w h, 0..1).

    Args:
        label: the .txt file.

    Returns:
        (CAUCAFall class, box), or None for an empty or missing file.
    """
    if not label.is_file():
        return None
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
            return cls, Rect(x1=max(0.0, cx - w / 2), y1=max(0.0, cy - h / 2), x2=min(1.0, cx + w / 2),
                             y2=min(1.0, cy + h / 2))
    return None


def subject_frames(*, subject_dir: Path) -> list[Labelled]:
    """Every ``EVERY``-th labelled frame of one person, all activities.

    Args:
        subject_dir: one CAUCAFall subject folder (activity folders inside).

    Returns:
        The kept frames.
    """
    kept = []
    for activity in sorted(d for d in subject_dir.iterdir() if d.is_dir()):
        images = sorted(p for p in activity.iterdir() if p.suffix.lower() in (".png", ".jpg"))
        for image in images[::EVERY]:
            found = read_yolo_box(label=image.with_suffix(".txt"))
            if found is not None:
                kept.append(Labelled(image=image, box=found[1], class_id=1 if found[0] == 1 else 0))
    return kept


def match_teacher(*, box: Rect, people: Sequence[TeacherPerson]) -> TeacherPerson | None:
    """The teacher's person that overlaps the labelled box most, if by ``MATCH_IOU`` or more."""
    best, best_iou = None, MATCH_IOU
    for person in people:
        iou = box_iou(a=box, b=person.box)
        if iou >= best_iou:
            best, best_iou = person, iou
    return best


def coco_keypoints(*, person: TeacherPerson | None, width_px: int, height_px: int) -> list[float]:
    """Keypoints in COCO form (x, y, v per point, pixels; v=2 seen, v=0 not labelled).

    Args:
        person: the matched teacher person, or None.
        width_px: picture width.
        height_px: picture height.

    Returns:
        51 numbers.
    """
    if person is None:
        return [0.0] * (3 * KEYPOINT_COUNT)
    values: list[float] = []
    for (x, y), c in zip(person.xy, person.confidence):
        seen = c >= KEYPOINT_MIN_CONFIDENCE
        values += [round(float(x) * width_px, 1), round(float(y) * height_px, 1), 2.0] if seen else [0.0, 0.0, 0.0]
    return values


def yolo_line(*, class_id: int, box: Rect, keypoints: Sequence[float], width_px: int, height_px: int) -> str:
    """One Ultralytics pose label line: class, box (cx cy w h) and x y v per keypoint, all 0..1."""
    cx, cy, w, h = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, box.x2 - box.x1, box.y2 - box.y1
    points = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i : i + 3]
        points += [x / width_px, y / height_px, 1.0 if v > 0 else 0.0]
    return " ".join([str(class_id)] + [f"{v:.5f}" for v in (cx, cy, w, h, *points)])


def run_teacher(*, model: object, image_rgb: npt.NDArray[np.uint8]) -> list[TeacherPerson]:
    """People found by RF-DETR Keypoint, in 0..1 of the picture."""
    height, width = image_rgb.shape[:2]
    found = model.predict(image_rgb, threshold=TEACHER_MIN_SCORE)  # type: ignore[attr-defined]
    boxes = found.data.get("xyxy")
    people = []
    for i in range(len(found)):
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        people.append(TeacherPerson(box=Rect(x1=x1 / width, y1=y1 / height, x2=x2 / width, y2=y2 / height),
                                    xy=np.asarray(found.xy[i], dtype=np.float64) / (width, height),
                                    confidence=np.asarray(found.keypoint_confidence[i], dtype=np.float64)))
    return people


def main() -> None:
    """Convert CAUCAFall with teacher keypoints into ``<out>/<split>/`` (images, labels, COCO file)."""
    parser = argparse.ArgumentParser(description="CAUCAFall -> posture and keypoint dataset.")
    parser.add_argument("--caucafall", type=Path, required=True, help="folder with the 10 subject folders")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from rfdetr import RFDETRKeypointPreview  # heavy; only the converter needs it

    teacher = RFDETRKeypointPreview()
    subjects = sorted((d for d in args.caucafall.iterdir() if d.is_dir() and SUBJECT_NUMBER.search(d.name)),
                      key=lambda d: int(SUBJECT_NUMBER.findall(d.name)[-1]))
    categories = [{"id": i + 1, "name": n, "supercategory": "person", "keypoints": list(KEYPOINT_NAMES),
                   "skeleton": [list(e) for e in SKELETON]} for i, n in enumerate(CLASS_NAMES)]
    cocos = {s: {"images": [], "annotations": [], "categories": categories} for s in set(SPLIT_OF_SUBJECT.values())}
    matched = total = 0
    for subject_dir in subjects:
        number = int(SUBJECT_NUMBER.findall(subject_dir.name)[-1])
        split = SPLIT_OF_SUBJECT[number]
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)
        for frame in subject_frames(subject_dir=subject_dir):
            bgr = cv2.imread(str(frame.image))
            if bgr is None:
                continue
            scale = SIDE_PX / max(bgr.shape[:2])
            small = cv2.resize(bgr, (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
            height, width = small.shape[:2]
            rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            person = match_teacher(box=frame.box, people=run_teacher(model=teacher, image_rgb=rgb))
            matched += person is not None
            total += 1
            stem = f"s{number:02d}_{frame.image.parent.name.replace(' ', '_')}_{frame.image.stem}"
            cv2.imwrite(str(args.out / split / "images" / f"{stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            keypoints = coco_keypoints(person=person, width_px=width, height_px=height)
            (args.out / split / "labels" / f"{stem}.txt").write_text(
                yolo_line(class_id=frame.class_id, box=frame.box, keypoints=keypoints, width_px=width,
                          height_px=height) + "\n", encoding="utf-8")
            coco = cocos[split]
            image_id = len(coco["images"]) + 1
            b = frame.box
            coco["images"].append({"id": image_id, "file_name": f"{stem}.jpg", "width": width, "height": height})
            coco["annotations"].append({
                "id": image_id, "image_id": image_id, "category_id": frame.class_id + 1, "iscrowd": 0,
                "bbox": [round(b.x1 * width, 1), round(b.y1 * height, 1), round((b.x2 - b.x1) * width, 1),
                         round((b.y2 - b.y1) * height, 1)],
                "area": round((b.x2 - b.x1) * width * (b.y2 - b.y1) * height, 1),
                "keypoints": keypoints, "num_keypoints": sum(1 for v in keypoints[2::3] if v > 0),
            })
        print(f"{subject_dir.name} -> {split}: {matched} of {total} frames so far have teacher keypoints", flush=True)
    for split, coco in cocos.items():
        (args.out / split / "_annotations.coco.json").write_text(json.dumps(coco) + "\n", encoding="utf-8")
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"kpt_shape: [{KEYPOINT_COUNT}, 3]\nflip_idx: {list(FLIP_INDEX)}\n"
        f"names: {dict(enumerate(CLASS_NAMES))}\n", encoding="utf-8")
    print(f"done: {total} frames, {matched} with teacher keypoints", flush=True)
    return None


if __name__ == "__main__":
    main()
