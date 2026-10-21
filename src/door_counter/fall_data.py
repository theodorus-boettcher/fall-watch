"""Turn public fall videos into a posture + keypoint dataset: COCO keypoints for Roboflow, YOLO pose labels.

CAUCAFall (CC BY 4.0, ``--caucafall``) has 10 people doing 10 activities each in one room, filmed from a
wall corner at 2.15 m. Every frame has one person box and a class: ``fall`` when the body is on the ground
after a fall, ``nofall`` otherwise. That maps to our ``lying`` / ``not lying``. There are no keypoints, so
a big pose model, RF-DETR Keypoint, labels them: its person that overlaps the CAUCAFall box by
``MATCH_IOU`` or more gives the keypoints. With no match, the box keeps its class and its keypoints stay
"not labelled". Frames come at 23 per second and are near copies, so one in ``EVERY`` is kept.

GMDCSA-24 (MIT, ``--gmdcsa``) has 4 people in 3 rooms, filmed by a low webcam, with time ranges per
activity and no boxes. The posture comes from the time (``posture_at``) and the box and keypoints from
the teacher; frames where it does not find exactly one person are left out. About 5 pictures per second
are kept.

The split goes by person (``SPLIT_OF_SUBJECT``, ``GMDCSA_SPLIT_OF_SUBJECT``), so the same person is never
in two splits; GMDCSA-24's room 3 (person 4) is the test. Pictures are scaled to ``SIDE_PX``.

Needs the ``teacher`` dependency group. Example:
``uv run --group teacher python -m door_counter.fall_data --caucafall data/caucafall/raw --out data/fall-v1``.
"""

import argparse
import json
import re
from collections.abc import Iterator, Sequence
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
GMDCSA_SPLIT_OF_SUBJECT = {1: "train", 2: "train", 3: "valid", 4: "test"}  # person 4 is the only one in room 3
SUBJECT_NUMBER = re.compile(r"(\d+)")
ACTIVITY = re.compile(r"([A-Za-z][A-Za-z ()]*?)\s*\[([^\]]*)\]")
TIME_RANGE = re.compile(r"([\d.]+)\s*to\s*([\d.]+)")
GMDCSA_EVERY_S = 0.2
FALL_SETTLE_S = 1.0  # the first second of "Falling" is the way down: neither posture
LYING_ACTIVITIES = ("Sleeping",)
UNCLEAR_ACTIVITIES = ("Exercising",)  # push-ups and yoga: left out


@dataclass(frozen=True)
class Labelled:
    """One kept frame: its picture, CAUCAFall box (0..1) and our class."""

    image: Path
    box: Rect
    class_id: int


@dataclass(frozen=True)
class Sample:
    """One picture to label: where it goes, our class, and the box if the dataset has one (0..1)."""

    split: str
    stem: str
    bgr: npt.NDArray[np.uint8]
    class_id: int
    box: Rect | None


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


def activity_ranges(*, text: str) -> dict[str, list[tuple[float, float]]]:
    """Read GMDCSA-24 activity time ranges like ``Falling (SW)[3.4 to 6]; Standing[4.5 to 6.2; 10 to 13]``.

    Args:
        text: a CSV line or its classes field.

    Returns:
        Activity (first word; ``Fall`` is read as ``Falling``) -> its (start, end) ranges in seconds.
    """
    found: dict[str, list[tuple[float, float]]] = {}
    for name, inside in ACTIVITY.findall(text):
        word = name.split()[0]
        word = "Falling" if word.startswith("Fall") else word
        found.setdefault(word, []).extend((float(a), float(b)) for a, b in TIME_RANGE.findall(inside))
    return found


def posture_at(*, ranges: dict[str, list[tuple[float, float]]], time_s: float) -> int | None:
    """Our class at one moment of a GMDCSA-24 clip.

    Args:
        ranges: from ``activity_ranges``.
        time_s: seconds from the clip start.

    Returns:
        1 (lying) when sleeping or from ``FALL_SETTLE_S`` into a fall, 0 (not lying) during any other named
        activity, None when nothing is named, the activity is unclear, or the labels disagree.
    """
    def active(name: str) -> list[tuple[float, float]]:
        return [(a, b) for a, b in ranges.get(name, []) if a <= time_s <= b]

    if any(active(name) for name in UNCLEAR_ACTIVITIES):
        return None
    falling = active("Falling")
    if any(time_s < a + FALL_SETTLE_S for a, _ in falling):
        return None
    lying = bool(falling) or any(active(name) for name in LYING_ACTIVITIES)
    other = any(active(name) for name in ranges if name not in ("Falling", *LYING_ACTIVITIES))
    if lying and other:
        return None
    return 1 if lying else 0 if other else None


def caucafall_samples(*, root: Path) -> Iterator[Sample]:
    """CAUCAFall pictures with their boxes and classes, split by person."""
    subjects = sorted((d for d in root.iterdir() if d.is_dir() and SUBJECT_NUMBER.search(d.name)),
                      key=lambda d: int(SUBJECT_NUMBER.findall(d.name)[-1]))
    for subject_dir in subjects:
        number = int(SUBJECT_NUMBER.findall(subject_dir.name)[-1])
        for frame in subject_frames(subject_dir=subject_dir):
            bgr = cv2.imread(str(frame.image))
            if bgr is not None:
                yield Sample(split=SPLIT_OF_SUBJECT[number],
                             stem=f"s{number:02d}_{frame.image.parent.name.replace(' ', '_')}_{frame.image.stem}",
                             bgr=np.asarray(bgr, dtype=np.uint8), class_id=frame.class_id, box=frame.box)
    return None


def gmdcsa_samples(*, root: Path) -> Iterator[Sample]:
    """GMDCSA-24 pictures every ``GMDCSA_EVERY_S`` with the posture from the time ranges (no boxes)."""
    for subject in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Subject")):
        number = int(SUBJECT_NUMBER.findall(subject.name)[-1])
        for kind in ("Fall", "ADL"):
            for line in (subject / f"{kind}.csv").read_text(encoding="utf-8").splitlines()[1:]:
                file_name = line.split(",", maxsplit=1)[0].strip()
                video = subject / kind / file_name
                if not file_name or not video.is_file():
                    continue
                ranges = activity_ranges(text=line)
                capture = cv2.VideoCapture(str(video))
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                step = max(1, round(fps * GMDCSA_EVERY_S))
                index = 0
                try:
                    while True:
                        ok, bgr = capture.read()
                        if not ok:
                            break
                        class_id = posture_at(ranges=ranges, time_s=index / fps) if index % step == 0 else None
                        if class_id is not None:
                            yield Sample(split=GMDCSA_SPLIT_OF_SUBJECT[number],
                                         stem=f"g{number}_{kind}_{video.stem}_{index:04d}",
                                         bgr=np.asarray(bgr, dtype=np.uint8), class_id=class_id, box=None)
                        index += 1
                finally:
                    capture.release()
    return None


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
    """Convert one dataset with teacher keypoints into ``<out>/<split>/`` (images, labels, COCO file)."""
    parser = argparse.ArgumentParser(description="Public fall videos -> posture and keypoint dataset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--caucafall", type=Path, help="CAUCAFall folder with the 10 subject folders")
    source.add_argument("--gmdcsa", type=Path, help="GMDCSA-24 folder with the 4 subject folders")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from rfdetr import RFDETRKeypointPreview  # heavy; only the converter needs it

    teacher = RFDETRKeypointPreview()
    samples = caucafall_samples(root=args.caucafall) if args.caucafall else gmdcsa_samples(root=args.gmdcsa)
    categories = [{"id": i + 1, "name": n, "supercategory": "person", "keypoints": list(KEYPOINT_NAMES),
                   "skeleton": [list(e) for e in SKELETON]} for i, n in enumerate(CLASS_NAMES)]
    cocos: dict[str, dict[str, list[object]]] = {
        s: {"images": [], "annotations": [], "categories": list(categories)} for s in ("train", "valid", "test")}
    kept = skipped = 0
    for sample in samples:
        rgb = np.asarray(cv2.cvtColor(sample.bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        people = run_teacher(model=teacher, image_rgb=rgb)
        if sample.box is not None:
            box, person = sample.box, match_teacher(box=sample.box, people=people)
        elif len(people) == 1:
            box, person = people[0].box, people[0]
        else:
            skipped += 1  # no box in the dataset and not exactly one person found
            continue
        scale = SIDE_PX / max(sample.bgr.shape[:2])
        small = cv2.resize(sample.bgr, (round(sample.bgr.shape[1] * scale), round(sample.bgr.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        height, width = small.shape[:2]
        split_dir = args.out / sample.split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(split_dir / "images" / f"{sample.stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        keypoints = coco_keypoints(person=person, width_px=width, height_px=height)
        (split_dir / "labels" / f"{sample.stem}.txt").write_text(
            yolo_line(class_id=sample.class_id, box=box, keypoints=keypoints, width_px=width, height_px=height) + "\n",
            encoding="utf-8")
        coco = cocos[sample.split]
        image_id = len(coco["images"]) + 1
        coco["images"].append({"id": image_id, "file_name": f"{sample.stem}.jpg", "width": width, "height": height})
        coco["annotations"].append({
            "id": image_id, "image_id": image_id, "category_id": sample.class_id + 1, "iscrowd": 0,
            "bbox": [round(box.x1 * width, 1), round(box.y1 * height, 1), round((box.x2 - box.x1) * width, 1),
                     round((box.y2 - box.y1) * height, 1)],
            "area": round((box.x2 - box.x1) * width * (box.y2 - box.y1) * height, 1),
            "keypoints": keypoints, "num_keypoints": sum(1 for v in keypoints[2::3] if v > 0),
        })
        kept += 1
        if kept % 500 == 0:
            print(f"{kept} pictures kept, {skipped} left out", flush=True)
    for split, coco in cocos.items():
        (args.out / split).mkdir(parents=True, exist_ok=True)
        (args.out / split / "_annotations.coco.json").write_text(json.dumps(coco) + "\n", encoding="utf-8")
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"kpt_shape: [{KEYPOINT_COUNT}, 3]\nflip_idx: {list(FLIP_INDEX)}\n"
        f"names: {dict(enumerate(CLASS_NAMES))}\n", encoding="utf-8")
    print(f"done: {kept} pictures kept, {skipped} left out (no box and not exactly one person)", flush=True)
    return None


if __name__ == "__main__":
    main()
