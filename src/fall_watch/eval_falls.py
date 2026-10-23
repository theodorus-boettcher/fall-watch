"""Score fall alerts on labelled clips: falls caught, false alarms, and seconds from the fall to the alert.

The models run as the exported ncnn files the phone uses, at the phone's input size and score threshold, on
about 5 pictures per second like the phone. Their people go through ``FallWatch``, the same code as the
gateway, with no bed zones. A fall clip counts as caught when it raises an alert; a normal clip that raises
one is a false alarm. Normal clips where someone sleeps are counted apart: in a real room the bed zone
covers them.

The public clips are short (4 to 10 s), so the alert comes after ``CLIP_ALERT_AFTER_S`` here instead of the
gateway's 5 s. Our own clips are long enough for the gateway's time (``--captures-alert-after-s 5``).

Clip sets:
    ``--gmdcsa``: GMDCSA-24 (subject folders with ``Fall``/``ADL`` videos and CSVs of time ranges).
    ``--caucafall``: CAUCAFall's held-out people (``--caucafall-subjects``, 9 and 10 by default): one video per
    activity; ``Fall *`` activities are falls, and the first frame labelled ``fall`` is when the body is down.
    ``--captures``: our own test clips recorded with the gateway (``test-fall-*`` are falls, other ``test-*``
    clips are normal). Their pictures are named ``<session>_<frame id>.jpg``; times come from the frame ids at
    ``PHONE_FRAME_RATE_HZ``. That holds for clips recorded with the motion gate off, as all of ours are: with it
    on, a still room runs about 1 frame a second.

Needs the ``train`` dependency group. Example:
``uv run --group train python -m fall_watch.eval_falls --model before=models/pose-coco
--gmdcsa data/gmdcsa24/raw/<folder> --out measurements/falls-before.json``.
"""

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics.models import YOLO

from fall_watch.fall_data import gmdcsa_videos, video_frames
from fall_watch.gateway.fall_watch import PHONE_FPS_HZ, FallWatch, parse_bed
from fall_watch.gateway.messages import Frame, Item
from fall_watch.yolo_pose import DEFAULT_MIN_SCORE, DEFAULT_OVERLAP_IOU, Keypoint, Rect

PHONE_INPUT_SIZE_PX = 320
CLIP_ALERT_AFTER_S = 1.0
CAUCAFALL_FPS = 23.0
PHONE_FRAME_RATE_HZ = 5.5  # measured on the S5 with YOLO26n-pose at 320; turns capture frame ids into times
CAUCAFALL_TEST_SUBJECTS = (9, 10)


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
    clip_set: str
    falls: int
    falls_caught: int
    normal_clips: int
    false_alarms: int
    sleeping_clips: int
    sleeping_alerts: int
    mean_seconds_to_alert: float | None


def gmdcsa_clips(*, root: Path, subjects: Sequence[int] = (1, 2, 3, 4)) -> list[Clip]:
    """List the GMDCSA-24 clips with their labels.

    Args:
        root: the folder with ``Subject 1`` to ``Subject 4``.
        subjects: which people (person 4 is alone in room 3, the part no model trains on).

    Returns:
        Every clip named in a ``Fall.csv`` or ``ADL.csv`` whose video exists.
    """
    clips = []
    for subject in sorted(root / f"Subject {n}" for n in subjects):
        for video in gmdcsa_videos(subject=subject):
            falling = video.ranges.get("Falling")
            clips.append(Clip(name=f"{subject.name.replace(' ', '')}/{video.kind}/{video.path.name}", path=video.path,
                              is_fall=video.kind == "Fall", has_sleeping="Sleeping" in video.ranges,
                              fall_start_s=falling[0][0] if falling else None))
    return clips


def caucafall_clips(*, root: Path, subjects: Sequence[int]) -> list[Clip]:
    """List CAUCAFall activity videos of some people.

    Args:
        root: the folder with ``Subject.1`` to ``Subject.10``.
        subjects: which people.

    Returns:
        One clip per activity; for falls, ``fall_start_s`` is the first frame labelled ``fall`` (body down).
    """
    clips = []
    for number in subjects:
        for activity in sorted(d for d in (root / f"Subject.{number}").iterdir() if d.is_dir()):
            videos = sorted(activity.glob("*.avi"))
            if not videos:
                continue
            labels = sorted(activity.glob("*.txt"))
            labels = [p for p in labels if p.name != "classes.txt"]
            down = next((i for i, p in enumerate(labels) if p.read_text(encoding="utf-8").startswith("1 ")), None)
            clips.append(Clip(name=f"Subject{number}/{activity.name}", path=videos[0],
                              is_fall=activity.name.startswith("Fall"), has_sleeping=False,
                              fall_start_s=None if down is None else down / CAUCAFALL_FPS))
    return clips


def capture_clips(*, root: Path) -> list[Clip]:
    """List our recorded test clips.

    Args:
        root: the capture folder (``data/captures``).

    Returns:
        One clip per ``test-*`` folder; ``test-fall-*`` ones are falls (fall time not known).
    """
    return [Clip(name=d.name, path=d, is_fall=d.name.startswith("test-fall"), has_sleeping=False, fall_start_s=None)
            for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith("test-")]


def sampled_frames(*, video: Path) -> Iterator[tuple[float, npt.NDArray[np.uint8]]]:
    """Yield (time in seconds, BGR picture) at about the phone's rate.

    Args:
        video: a video file, or a capture folder of ``<session>_<frame id>.jpg`` pictures (already upright).

    Returns:
        A generator of timed pictures.
    """
    if video.is_dir():
        for picture_path in sorted(video.glob("*.jpg")):
            picture = cv2.imread(str(picture_path))
            if picture is not None:
                yield int(picture_path.stem.rsplit("_", maxsplit=1)[-1]) / PHONE_FRAME_RATE_HZ, np.asarray(
                    picture, dtype=np.uint8)
        return None
    for _, time_s, picture in video_frames(video=video, every_s=1 / PHONE_FPS_HZ):
        yield time_s, picture
    return None


def people_of(*, result: object) -> tuple[Item, ...]:
    """Turn one Ultralytics pose result into the items the phone would send.

    Args:
        result: one result of ``YOLO.predict``.

    Returns:
        Its people, as the phone sends them.
    """
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


def run_clip(*, model: YOLO, name: str, clip: Clip, beds: Sequence[Rect] = (),
             alert_after_s: float = CLIP_ALERT_AFTER_S) -> ClipResult:
    """Play one clip through the model and the fall watch.

    Args:
        model: the phone model (ncnn folder).
        name: the model's name in the report.
        clip: the clip.
        beds: bed zones in 0..1 of the picture, where lying raises no alert.
        alert_after_s: how long someone must lie outside a bed before the alert.

    Returns:
        Whether it alerted, and how long after the fall started.
    """
    watch = FallWatch(beds=beds, alert_after_s=alert_after_s)
    class_names = [model.names[i] for i in sorted(model.names)]
    for k, (time_s, picture) in enumerate(sampled_frames(video=clip.path)):
        (result,) = list(model.predict(source=picture, imgsz=PHONE_INPUT_SIZE_PX, conf=DEFAULT_MIN_SCORE,
                                       iou=DEFAULT_OVERLAP_IOU, agnostic_nms=True, verbose=False))
        height, width = picture.shape[:2]
        watch.update(frame=Frame(frame_id=k, wall_time_s=time_s, width_px=width, height_px=height, inference_s=0.0,
                                 items=people_of(result=result)), class_names=class_names)
    alerts = watch.alerts()
    first = alerts[0].raised_s if alerts else None
    delay = None if first is None or clip.fall_start_s is None else round(first - clip.fall_start_s, 2)
    return ClipResult(model=name, clip=clip.name, is_fall=clip.is_fall, has_sleeping=clip.has_sleeping,
                      alerted=bool(alerts), seconds_to_alert=delay)


def summarise(*, model: str, clip_set: str, results: Sequence[ClipResult]) -> Summary:
    """Count falls caught and false alarms for one model on one clip set.

    Args:
        model: the model's name.
        clip_set: the set's name.
        results: its clip results.

    Returns:
        The summary; sleeping clips are kept out of the false alarms.
    """
    falls = [r for r in results if r.is_fall]
    normal = [r for r in results if not r.is_fall and not r.has_sleeping]
    sleeping = [r for r in results if not r.is_fall and r.has_sleeping]
    delays = [r.seconds_to_alert for r in falls if r.seconds_to_alert is not None]
    return Summary(model=model, clip_set=clip_set, falls=len(falls), falls_caught=sum(r.alerted for r in falls),
                   normal_clips=len(normal), false_alarms=sum(r.alerted for r in normal),
                   sleeping_clips=len(sleeping), sleeping_alerts=sum(r.alerted for r in sleeping),
                   mean_seconds_to_alert=round(sum(delays) / len(delays), 2) if delays else None)


def main() -> None:
    """Score every model on the clips, print a table and save JSON."""
    parser = argparse.ArgumentParser(description="Score fall alerts on labelled clips.")
    parser.add_argument("--model", action="append", required=True,
                        help="name=ncnn folder, e.g. before=models/pose-coco")
    parser.add_argument("--gmdcsa", type=Path, help="GMDCSA-24 folder with the Subject folders")
    parser.add_argument("--gmdcsa-subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--caucafall", type=Path, help="CAUCAFall folder with the Subject.N folders")
    parser.add_argument("--caucafall-subjects", type=int, nargs="+", default=list(CAUCAFALL_TEST_SUBJECTS))
    parser.add_argument("--captures", type=Path, help="our capture folder with test-* clips")
    parser.add_argument("--bed", type=lambda text: parse_bed(text=text), action="append", default=[],
                        help="bed zone of the --captures room, x1,y1,x2,y2 in 0..1 (repeat for more beds)")
    parser.add_argument("--captures-alert-after-s", type=float, default=CLIP_ALERT_AFTER_S,
                        help="alert time for the --captures clips (the gateway uses 5)")
    parser.add_argument("--out", type=Path, help="write the results here as JSON")
    args = parser.parse_args()
    clip_sets = {}
    if args.gmdcsa is not None:
        clip_sets["gmdcsa24"] = gmdcsa_clips(root=args.gmdcsa, subjects=args.gmdcsa_subjects)
    if args.caucafall is not None:
        clip_sets["caucafall"] = caucafall_clips(root=args.caucafall, subjects=args.caucafall_subjects)
    if args.captures is not None:
        clip_sets["ours"] = capture_clips(root=args.captures)
    if not clip_sets:
        parser.error("give --gmdcsa, --caucafall and/or --captures")
    summaries, results = [], []
    for spec in args.model:
        name, _, folder = spec.partition("=")
        model = YOLO(model=folder, task="pose")
        for set_name, clips in clip_sets.items():
            ours = set_name == "ours"
            mine = [run_clip(model=model, name=name, clip=clip, beds=args.bed if ours else [],
                             alert_after_s=args.captures_alert_after_s if ours else CLIP_ALERT_AFTER_S)
                    for clip in clips]
            results += mine
            summaries.append(summarise(model=name, clip_set=set_name, results=mine))
    print(f"{'model':10} {'clips':10} {'falls caught':>13} {'false alarms':>13} {'sleeping alerts':>16} "
          f"{'s to alert':>11}")
    for s in summaries:
        delay = "-" if s.mean_seconds_to_alert is None else f"{s.mean_seconds_to_alert:.1f}"
        print(f"{s.model:10} {s.clip_set:10} {s.falls_caught:>6} / {s.falls:<4} {s.false_alarms:>6} / "
              f"{s.normal_clips:<4} {s.sleeping_alerts:>8} / {s.sleeping_clips:<5} {delay:>11}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summaries": [asdict(s) for s in summaries],
                                        "clips": [asdict(r) for r in results]}, indent=2) + "\n", encoding="utf-8")
    return None


if __name__ == "__main__":
    main()
