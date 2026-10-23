"""Talking to Roboflow: API key, the project, uploads and versions.

The key (the account's private API key) lives in the repo's git-ignored ``.env`` as ``ROBOFLOW_API_KEY=...``
and is never printed. The workspace is the one the key belongs to.
Run ``uv run python -m fall_watch.roboflow_io`` to create (or find) the keypoint project,
``... --upload data/fall-v1 --batch caucafall`` to upload a dataset in COCO keypoint layout (see
``fall_watch.fall_data``), and ``... --export data/rf-fall-v1`` to make a version and download it.

Auto Label does not do keypoint projects, so the room pictures the pose teacher missed get their boxes from
SAM 3 in a second, box-only project (``BOX_PROJECT_ID``). The whole room flow:

1. ``fall_data --room data/captures --out data/fall-room``: teacher labels for the room clips.
2. ``... --autolabel-room data/captures --converted data/fall-room --batch room-missed``: uploads the pictures
   it left out and starts SAM 3 on them (paid: about 1,000 pictures per credit). It prints the job id.
3. ``... --job <id>`` until it says done, then ``... --accept <id>`` and ``... --export-boxes data/rf-room-sam3``.
4. ``fall_data --sam3 data/rf-room-sam3 --out data/fall-room-sam3``: one box per picture, posture from the
   clip name.
5. Upload both converted folders to the keypoint project, ``--export`` a version, and train with
   ``fall_watch.train --repeat-room 2`` so the room counts 3x next to the public data.
"""

import argparse
import tempfile
import time
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import roboflow
import yaml
from roboflow.adapters.rfapi import RoboflowError
from roboflow.core.project import Project

from fall_watch.fall_data import FLIP_INDEX, JPEG_QUALITY, missed_room_pictures, scale_to_side

PROJECT_ID = "fall-watch"
PROJECT_TYPE = "keypoint-detection"
BOX_PROJECT_ID = "fall-watch-boxes"
BOX_PROJECT_TYPE = "object-detection"
AUTOLABEL_MODEL = "sam3-polygon"
AUTOLABEL_CONFIDENCE = 0.4
ANNOTATION_GROUP = "people"
KEY_NAME = "ROBOFLOW_API_KEY"
SPLITS = ("train", "valid", "test")
COCO_FILE = "_annotations.coco.json"
VERSION_WAIT_S = 30.0
VERSION_WAIT_TRIES = 60


class MissingApiKeyError(RuntimeError):
    """Raised when the Roboflow API key is not in the .env file."""


def load_api_key(*, env_path: Path) -> str:
    """Read the Roboflow API key from a .env file.

    Args:
        env_path: path to the .env file.

    Returns:
        The key.

    Raises:
        MissingApiKeyError: if the file or the key is missing.
    """
    if not env_path.is_file():
        raise MissingApiKeyError(f"{env_path} not found")
    with env_path.open(encoding="utf-8") as env_file:
        for line in env_file:
            name, _, value = line.strip().partition("=")
            if name == KEY_NAME and value:
                return value
    raise MissingApiKeyError(f"{KEY_NAME} not set in {env_path}")


def get_or_create_project(*, api_key: str, project_id: str = PROJECT_ID,
                          project_type: str = PROJECT_TYPE) -> Project:
    """Open the fall-watch project (or another one) in the key's workspace, creating it (private) on first use.

    Args:
        api_key: the Roboflow API key.
        project_id: the project's id.
        project_type: used only when creating it.

    Returns:
        The project.

    Raises:
        RuntimeError: if Roboflow refuses to create it.
    """
    workspace = roboflow.Roboflow(api_key=api_key).workspace()
    if any(name.split("/")[-1] == project_id for name in workspace.projects()):
        return workspace.project(project_id=project_id)
    return workspace.create_project(
        project_name=project_id,
        project_type=project_type,
        project_license="Private",
        annotation=ANNOTATION_GROUP,
    )


def iter_coco_members(*, dataset_dir: Path) -> Iterator[tuple[Path, str]]:
    """Yield (file on disk, name in the zip) in Roboflow's COCO layout: ``<split>/<image>`` and
    ``<split>/_annotations.coco.json`` side by side.

    Args:
        dataset_dir: a converted dataset with ``<split>/images/`` and ``<split>/_annotations.coco.json``.

    Returns:
        A generator of zip members.

    Raises:
        FileNotFoundError: if a split has no COCO file.
    """
    for split in SPLITS:
        coco = dataset_dir / split / COCO_FILE
        if not coco.is_file():
            raise FileNotFoundError(coco)
        yield coco, f"{split}/{COCO_FILE}"
        for image in sorted((dataset_dir / split / "images").glob("*.jpg")):
            yield image, f"{split}/{image.name}"
    return None


def pack_coco_zip(*, dataset_dir: Path, zip_path: Path) -> int:
    """Pack a converted dataset into one zip for Roboflow's server-side import.

    Args:
        dataset_dir: the converted dataset.
        zip_path: where to write the zip (JPEGs are stored, not compressed again).

    Returns:
        Number of files packed.
    """
    count = 0
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for source, name in iter_coco_members(dataset_dir=dataset_dir):
            archive.write(source, arcname=name)
            count += 1
    return count


def upload_dataset(*, api_key: str, dataset_dir: Path, batch_name: str) -> object:
    """Pack and upload a converted dataset into the project, keeping its train/valid/test splits.

    Args:
        api_key: the Roboflow API key.
        dataset_dir: the converted dataset.
        batch_name: name of the upload batch shown in Roboflow.

    Returns:
        Roboflow's final import status.
    """
    zip_path = dataset_dir.with_suffix(".roboflow.zip")
    count = pack_coco_zip(dataset_dir=dataset_dir, zip_path=zip_path)
    print(f"packed {count} files into {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)", flush=True)
    get_or_create_project(api_key=api_key)
    workspace = roboflow.Roboflow(api_key=api_key).workspace()
    return workspace.upload_dataset(
        dataset_path=str(zip_path),
        project_name=PROJECT_ID,
        project_license="Private",
        project_type=PROJECT_TYPE,
        batch_name=batch_name,
        poll_timeout=3 * 3600.0,
    )


def upload_pictures(*, project: Project, pictures: Sequence[Path], batch_name: str) -> int:
    """Upload capture pictures into one batch, scaled like the converter's and named ``<clip>_<frame>.jpg`` so the
    posture can be read back from the name.

    Args:
        project: the Roboflow project.
        pictures: capture frames, each inside its clip folder.
        batch_name: name of the upload batch shown in Roboflow.

    Returns:
        How many were uploaded; a picture that fails is reported and skipped.
    """
    uploaded = 0
    with tempfile.TemporaryDirectory() as tmp:
        for picture in pictures:
            bgr = cv2.imread(str(picture))
            if bgr is None:
                print(f"{picture}: not readable, skipped", flush=True)
                continue
            out = Path(tmp) / f"{picture.parent.name}_{picture.stem}.jpg"
            small = scale_to_side(bgr=np.asarray(bgr, dtype=np.uint8))
            cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            try:
                project.upload(image_path=str(out), batch_name=batch_name, split="train", num_retry_uploads=3)
            except (RuntimeError, RoboflowError) as e:  # keep going: one picture shouldn't stop the batch
                print(f"{out.name}: {e}", flush=True)
                continue
            uploaded += 1
    return uploaded


def start_autolabel(*, project: Project, batch_name: str) -> dict[str, object]:
    """Start SAM 3 Auto Label on one uploaded batch: an outline (and box) for every person.

    Args:
        project: the box project.
        batch_name: the batch made by ``upload_pictures``.

    Returns:
        Roboflow's reply with ``jobId``.

    Raises:
        LookupError: if the batch is not in the project.
    """
    batches = project.get_batches().get("batches", [])
    batch_id = next((b["id"] for b in batches if b.get("name") == batch_name), None)
    if batch_id is None:
        raise LookupError(f"batch {batch_name!r} not found among {[b.get('name') for b in batches]}")
    result: dict[str, object] = project.autolabel(
        batch_id, model=AUTOLABEL_MODEL, ontology={"person": "person"}, confidence=AUTOLABEL_CONFIDENCE,
        run_nms=False)  # a lying body often comes back in parts; the converter merges them
    return result


def accept_autolabel(*, project: Project, job_id: str) -> dict[str, object]:
    """Move a finished SAM 3 job's pictures into the dataset, keeping the split chosen at upload.

    Args:
        project: the box project.
        job_id: the auto-label job (``jobId`` from ``start_autolabel``).

    Returns:
        Roboflow's reply with ``numImagesAdded``.

    Raises:
        RuntimeError: if the job is not done yet.
    """
    job = project.autolabel_job(job_id)
    if job.get("status") != "done":
        raise RuntimeError(f"auto-label job {job_id} is {job.get('status')!r}, not done")
    result: dict[str, object] = project.accept_annotation_job_images(
        job_id=str(job["annotationJobId"]), split_method="preset", statuses_to_include=["annotated"],
        train_count=0, valid_count=0, test_count=0)  # counts are only used by the other split methods
    return result


def download_new_version(*, project: Project, out_dir: Path, model_format: str) -> Path:
    """Make a dataset version with no resizing or augmentation and download it once Roboflow has built it.

    Args:
        project: the Roboflow project.
        out_dir: where the export goes.
        model_format: Roboflow's export format, e.g. ``yolov8`` or ``coco``.

    Returns:
        The export folder.

    Raises:
        TimeoutError: if the version is still not ready after ``VERSION_WAIT_TRIES`` tries.
    """
    number = project.generate_version(settings={"augmentation": {}, "preprocessing": {"auto-orient": True}})
    for _ in range(VERSION_WAIT_TRIES):
        try:
            project.version(number).download(model_format, location=str(out_dir), overwrite=True)
            return out_dir
        except (RuntimeError, RoboflowError) as e:  # still being generated, or Roboflow's passing 'Internal Error'
            print(f"version {number} not ready ({e}); waiting", flush=True)
            time.sleep(VERSION_WAIT_S)
    raise TimeoutError(f"version {number} was not ready after {VERSION_WAIT_TRIES * VERSION_WAIT_S} s")


def export_version(*, project: Project, out_dir: Path) -> Path:
    """Make a version of the keypoint project and download it in YOLO format, ready to train on.

    Roboflow writes ``flip_idx`` as 0..16 (no left/right swap); for the COCO skeleton that would teach
    mirrored pictures the wrong side, so the COCO swap is written back.

    Args:
        project: the Roboflow project.
        out_dir: where the export goes (``data.yaml``, ``train/``, ``valid/``, ``test/``).

    Returns:
        The export folder.
    """
    download_new_version(project=project, out_dir=out_dir, model_format="yolov8")
    data_yaml = out_dir / "data.yaml"
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    spec.update(flip_idx=list(FLIP_INDEX), path=str(out_dir.resolve()), train="train/images", val="valid/images",
                test="test/images")
    data_yaml.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return out_dir


def main() -> None:
    """Create or find the project, optionally upload a dataset or export a version, and print what Roboflow reports."""
    parser = argparse.ArgumentParser(description="Create or find the Roboflow project, upload datasets, export.")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="file holding ROBOFLOW_API_KEY")
    parser.add_argument("--upload", type=Path, help="dataset folder in COCO keypoint layout to upload")
    parser.add_argument("--batch", help="upload batch name, for --upload and --autolabel-room")
    parser.add_argument("--export", type=Path, help="make a version and download it here (YOLO format)")
    parser.add_argument("--autolabel-room", type=Path,
                        help="capture folder: send the room pictures the converter missed to SAM 3 (costs credits)")
    parser.add_argument("--converted", type=Path, help="fall_data's --room output, for --autolabel-room")
    parser.add_argument("--job", help="print the state of a SAM 3 job")
    parser.add_argument("--accept", action="append", default=[], help="move a finished SAM 3 job into the dataset")
    parser.add_argument("--export-boxes", type=Path, help="make a version of the box project and download it (COCO)")
    args = parser.parse_args()
    if (args.upload or args.autolabel_room) and not args.batch:
        parser.error("--upload and --autolabel-room need --batch")
    if args.autolabel_room and not args.converted:
        parser.error("--autolabel-room needs --converted")
    api_key = load_api_key(env_path=args.env)
    if args.upload is not None:
        print("upload status:", upload_dataset(api_key=api_key, dataset_dir=args.upload, batch_name=args.batch))
    project = get_or_create_project(api_key=api_key)
    if args.export is not None:
        print(f"exported to {export_version(project=project, out_dir=args.export)}")
    if args.autolabel_room or args.job or args.accept or args.export_boxes:
        boxes = get_or_create_project(api_key=api_key, project_id=BOX_PROJECT_ID, project_type=BOX_PROJECT_TYPE)
        if args.autolabel_room is not None:
            missed = missed_room_pictures(root=args.autolabel_room, converted=args.converted)
            uploaded = upload_pictures(project=boxes, pictures=missed, batch_name=args.batch)
            job = start_autolabel(project=boxes, batch_name=args.batch)
            print(f"{uploaded} of {len(missed)} missed pictures uploaded; SAM 3 job {job.get('jobId')}", flush=True)
        if args.job is not None:
            print(boxes.autolabel_job(args.job), flush=True)
        for job_id in args.accept:
            print(f"{job_id}: {accept_autolabel(project=boxes, job_id=job_id)}", flush=True)
        if args.export_boxes is not None:
            out_dir = download_new_version(project=boxes, out_dir=args.export_boxes, model_format="coco")
            print(f"boxes exported to {out_dir}", flush=True)
    print(f"project {project.id} type={project.type} images={project.images} public={project.public}")
    return None


if __name__ == "__main__":
    main()
