"""Score fall alerts on labelled clips: falls caught, false alarms, and seconds from the fall to the alert.

The models run as the exported ncnn files the phone uses, at the phone's input size and score threshold, on
about 5 pictures per second like the phone. Their people go through ``FallWatch``, the same code as the
gateway, with no bed zones. A fall clip counts as caught when it raises an alert; a normal clip that raises
one is a false alarm. Normal clips where someone sleeps are counted apart: in a real room the bed zone
covers them.

The public clips are short (4 to 10 s), so the alert comes after ``CLIP_ALERT_AFTER_S`` here instead of the
gateway's 5 s.

Clip sets:
    ``--gmdcsa``: GMDCSA-24 (subject folders with ``Fall``/``ADL`` videos and CSVs of time ranges).

Needs the ``train`` dependency group. Example:
``uv run --group train python -m door_counter.eval_falls --model before=models/pose-coco
--gmdcsa data/gmdcsa24/raw/<folder> --out measurements/falls-before.json``.
"""

import argparse
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics.models import YOLO

from door_counter.gateway.fall_watch import FallWatch
from door_counter.gateway.messages import Frame, Item
from door_counter.yolo_pose import Keypoint, Rect

PHONE_INPUT_SIZE_PX = 320
PHONE_MIN_SCORE = 0.25  # MIN_SCORE in CounterService.java
PHONE_OVERLAP_IOU = 0.7  # kOverlapIou in detector.cpp, across classes
PHONE_FPS_HZ = 5.0
CLIP_ALERT_AFTER_S = 1.0
INTERVAL = re.compile(r"([A-Za-z][A-Za-z ()]*?)\s*\[\s*([\d.]+)\s*to\s*([\d.]+)\s*\]")


@dataclass(frozen=True)
class Clip:
    """One labelled video."""

    name: str
    path: Path
    is_fall: bool
    has_sleeping: bool
    fall_start_s: float | None


@dataclass(frozen=True)
class ClipResult:
    """What one model did on one clip."""

    model: str
    clip: str
    is_fall: bool
    has_sleeping: bool
    alerted: bool
    seconds_to_alert: float | None  # from the start of the fall


@dataclass(frozen=True)
class Summary:
    """One model on one clip set."""

    model: str
    falls: int
    falls_caught: int
    normal_clips: int
    false_alarms: int
    sleeping_clips: int
    sleeping_alerts: int
    mean_seconds_to_alert: float | None


def parse_intervals(*, text: str) -> dict[str, tuple[float, float]]:
    """Read GMDCSA-24 activity ranges like ``Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]``.

    Args:
        text: a CSV line or its classes field.

    Returns:
        Activity name (first word, e.g. ``Falling``) -> (start, end) in seconds; the first range wins.
    """
    found: dict[str, tuple[float, float]] = {}
    for name, start, end in INTERVAL.findall(text):
        found.setdefault(name.split()[0], (float(start), float(end)))
    return found


def gmdcsa_clips(*, root: Path) -> list[Clip]:
    """List the GMDCSA-24 clips with their labels.

    Args:
        root: the folder with ``Subject 1`` to ``Subject 4``.

    Returns:
        Every clip named in a ``Fall.csv`` or ``ADL.csv`` whose video exists.
    """
    clips = []
    for subject in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Subject")):
        for kind in ("Fall", "ADL"):
            for line in (subject / f"{kind}.csv").read_text(encoding="utf-8").splitlines()[1:]:
                file_name = line.split(",", maxsplit=1)[0].strip()
                video = subject / kind / file_name
                if not file_name or not video.is_file():
                    continue
                ranges = parse_intervals(text=line)
                falling = ranges.get("Falling")
                clips.append(Clip(name=f"{subject.name.replace(' ', '')}/{kind}/{file_name}", path=video,
                                  is_fall=kind == "Fall", has_sleeping="Sleeping" in ranges,
                                  fall_start_s=None if falling is None else falling[0]))
    return clips


def sampled_frames(*, video: Path) -> Iterator[tuple[float, npt.NDArray[np.uint8]]]:
    """Yield (time in seconds, BGR picture) at about the phone's rate.

    Args:
        video: the video file.

    Returns:
        A generator of timed pictures.
    """
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / PHONE_FPS_HZ))
    index = 0
    try:
        while True:
            ok, picture = capture.read()
            if not ok:
                break
            if index % step == 0:
                yield index / fps, np.asarray(picture, dtype=np.uint8)
            index += 1
    finally:
        capture.release()
    return None


def people_of(*, result: object) -> tuple[Item, ...]:
    """Turn one Ultralytics pose result into the items the phone would send."""
    boxes, keypoints = getattr(result, "boxes", None), getattr(result, "keypoints", None)  # predict() is typed loosely
    if boxes is None or keypoints is None or len(boxes) == 0:
        return ()
    xyxy = np.clip(boxes.xyxyn.cpu().numpy(), 0.0, 1.0)
    points = np.clip(keypoints.xyn.cpu().numpy(), 0.0, 1.0)
    visibility = keypoints.conf.cpu().numpy() if keypoints.conf is not None else np.ones(points.shape[:2])
    return tuple(
        Item(class_id=int(c), score=float(s), box=Rect(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])),
             keypoints=tuple(Keypoint(x=float(x), y=float(y), visibility=float(v)) for (x, y), v in zip(p, vis)))
        for c, s, b, p, vis in zip(boxes.cls.tolist(), boxes.conf.tolist(), xyxy, points, visibility)
    )


def run_clip(*, model: YOLO, name: str, clip: Clip) -> ClipResult:
    """Play one clip through the model and the fall watch.

    Args:
        model: the phone model (ncnn folder).
        name: the model's name in the report.
        clip: the clip.

    Returns:
        Whether it alerted, and how long after the fall started.
    """
    watch = FallWatch(alert_after_s=CLIP_ALERT_AFTER_S)
    class_names = [model.names[i] for i in sorted(model.names)]
    for k, (time_s, picture) in enumerate(sampled_frames(video=clip.path)):
        (result,) = list(model.predict(source=picture, imgsz=PHONE_INPUT_SIZE_PX, conf=PHONE_MIN_SCORE,
                                       iou=PHONE_OVERLAP_IOU, agnostic_nms=True, verbose=False))
        height, width = picture.shape[:2]
        watch.update(frame=Frame(frame_id=k, wall_time_s=time_s, width_px=width, height_px=height, inference_s=0.0,
                                 items=people_of(result=result)), class_names=class_names)
    alerts = watch.alerts()
    first = alerts[0].raised_s if alerts else None
    delay = None if first is None or clip.fall_start_s is None else round(first - clip.fall_start_s, 2)
    return ClipResult(model=name, clip=clip.name, is_fall=clip.is_fall, has_sleeping=clip.has_sleeping,
                      alerted=bool(alerts), seconds_to_alert=delay)


def summarise(*, model: str, results: Sequence[ClipResult]) -> Summary:
    """Count falls caught and false alarms for one model.

    Args:
        model: the model's name.
        results: its clip results.

    Returns:
        The summary; sleeping clips are kept out of the false alarms.
    """
    falls = [r for r in results if r.is_fall]
    normal = [r for r in results if not r.is_fall and not r.has_sleeping]
    sleeping = [r for r in results if not r.is_fall and r.has_sleeping]
    delays = [r.seconds_to_alert for r in falls if r.seconds_to_alert is not None]
    return Summary(model=model, falls=len(falls), falls_caught=sum(r.alerted for r in falls),
                   normal_clips=len(normal), false_alarms=sum(r.alerted for r in normal),
                   sleeping_clips=len(sleeping), sleeping_alerts=sum(r.alerted for r in sleeping),
                   mean_seconds_to_alert=round(sum(delays) / len(delays), 2) if delays else None)


def main() -> None:
    """Score every model on the clips, print a table and save JSON."""
    parser = argparse.ArgumentParser(description="Score fall alerts on labelled clips.")
    parser.add_argument("--model", action="append", required=True, help="name=ncnn folder, e.g. before=models/pose-coco")
    parser.add_argument("--gmdcsa", type=Path, required=True, help="GMDCSA-24 folder with the Subject folders")
    parser.add_argument("--out", type=Path, help="write the results here as JSON")
    args = parser.parse_args()
    clips = gmdcsa_clips(root=args.gmdcsa)
    summaries, results = [], []
    for spec in args.model:
        name, _, folder = spec.partition("=")
        model = YOLO(model=folder, task="pose")
        mine = [run_clip(model=model, name=name, clip=clip) for clip in clips]
        results += mine
        summaries.append(summarise(model=name, results=mine))
    print(f"{'model':10} {'falls caught':>13} {'false alarms':>13} {'sleeping alerts':>16} {'s to alert':>11}")
    for s in summaries:
        delay = "-" if s.mean_seconds_to_alert is None else f"{s.mean_seconds_to_alert:.1f}"
        print(f"{s.model:10} {s.falls_caught:>6} / {s.falls:<4} {s.false_alarms:>6} / {s.normal_clips:<4} "
              f"{s.sleeping_alerts:>8} / {s.sleeping_clips:<5} {delay:>11}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summaries": [asdict(s) for s in summaries],
                                        "clips": [asdict(r) for r in results]}, indent=2) + "\n", encoding="utf-8")
    return None


if __name__ == "__main__":
    main()
