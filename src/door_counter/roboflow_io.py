"""Talking to Roboflow: API key, the project, uploads.

The key lives in the repo's git-ignored ``.env`` as ``ROBOFLOW_API_KEY=...`` and is never printed.
Run ``uv run python -m door_counter.roboflow_io`` to create (or find) the project,
``... --upload data/zerowaste-960`` to upload a converted dataset (see ``door_counter.zerowaste``),
``... --home-clips data/captures`` to upload the home clips and start SAM 3 labeling on each, and
``... --job <id>`` to see how a labeling job is doing.
"""

import argparse
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import roboflow
from roboflow.adapters.rfapi import ImageUploadError, RoboflowError
from roboflow.core.project import Project

WORKSPACE = "<workspace>"
PROJECT_ID = "sorting-line"
PROJECT_NAME = "sorting-line"
PROJECT_TYPE = "instance-segmentation"
ANNOTATION_GROUP = "materials"
KEY_NAME = "ROBOFLOW_API_KEY"
SPLITS = ("train", "valid", "test")
COCO_FILE = "_annotations.coco.json"
# One material per home clip, so every SAM 3 text prompt of a clip maps to that clip's material.
HOME_PROMPTS = {
    "cardboard": ("cardboard", "cardboard box", "piece of cardboard", "torn cardboard"),  # the last two find torn pieces
    "soft plastic": ("plastic bag", "plastic film", "plastic wrapper"),
    "rigid plastic": ("plastic bottle", "plastic container", "plastic cup", "plastic tray", "styrofoam"),
    "metal": ("can", "metal can", "aluminium foil", "metal lid", "metal object"),
}  # the class comes from the clip, so broad prompts are safe: whatever SAM 3 finds in a metal clip is metal
AUTOLABEL_MODEL = "sam3-polygon"
VERSION_WAIT_S = 30.0
VERSION_WAIT_TRIES = 60
AUTOLABEL_CONFIDENCE = 0.4
VALID_EVERY = 5  # every 5th uploaded home picture goes to valid (only for early stopping; the test is separate)


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


def get_or_create_project(*, api_key: str) -> Project:
    """Open the sorting-line project, creating it (private) on first use.

    Args:
        api_key: the Roboflow API key.

    Returns:
        The project.

    Raises:
        RuntimeError: if Roboflow refuses to create it.
    """
    workspace = roboflow.Roboflow(api_key=api_key).workspace(the_workspace=WORKSPACE)
    if any(name.split("/")[-1] == PROJECT_ID for name in workspace.projects()):
        return workspace.project(project_id=PROJECT_ID)
    return workspace.create_project(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
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
    workspace = roboflow.Roboflow(api_key=api_key).workspace(the_workspace=WORKSPACE)
    return workspace.upload_dataset(
        dataset_path=str(zip_path),
        project_name=PROJECT_ID,
        project_license="Private",
        project_type=PROJECT_TYPE,
        batch_name=batch_name,
        poll_timeout=3 * 3600.0,
    )


def material_of_clip(*, clip: str) -> str | None:
    """Read the material from a clip name: ``soft-plastic-2`` -> ``soft plastic``.

    Args:
        clip: the clip folder name.

    Returns:
        The material, or None for clips that hold more than one (the test scenes).
    """
    for material in HOME_PROMPTS:
        if clip.startswith(material.replace(" ", "-") + "-"):
            return material
    return None


def upload_clip(*, project: Project, clip_dir: Path, every: int, source: str) -> str:
    """Upload every n-th picture of one clip into its own batch.

    Args:
        project: the Roboflow project.
        clip_dir: folder of one capture clip.
        every: keep one picture in this many (neighbours are nearly the same picture).
        source: where the pictures come from (``home``, ``trashnet``); names the batch and tags the images.

    Returns:
        The batch name.
    """
    batch_name = f"{source}-{clip_dir.name}"
    failed = 0
    for i, image in enumerate(sorted(clip_dir.glob("*.jpg"))[::every]):
        try:
            project.upload(image_path=str(image), split="valid" if i % VALID_EVERY == VALID_EVERY - 1 else "train",
                           batch_name=batch_name, tag_names=[source, clip_dir.name], num_retry_uploads=3)
        except ImageUploadError as e:  # Roboflow sometimes answers 500; one lost picture must not stop the clip
            failed += 1
            print(f"{image.name}: upload failed ({e})", flush=True)
    if failed:
        print(f"{clip_dir.name}: {failed} pictures not uploaded; running again uploads them (duplicates are skipped)")
    return batch_name


def accept_autolabel(*, project: Project, job_id: str) -> dict[str, object]:
    """Move a finished SAM 3 job's pictures into the dataset, keeping the split chosen at upload.

    Args:
        project: the Roboflow project.
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


def export_version(*, project: Project, out_dir: Path) -> Path:
    """Make a dataset version with no resizing or augmentation, and download it in YOLO format.

    Args:
        project: the Roboflow project.
        out_dir: where the export goes (``data.yaml``, ``train/``, ``valid/``).

    Returns:
        The export folder.
    """
    number = project.generate_version(settings={"augmentation": {}, "preprocessing": {"auto-orient": True}})
    for _ in range(VERSION_WAIT_TRIES):
        try:
            project.version(number).download("yolov8", location=str(out_dir), overwrite=True)
            return out_dir
        except (RuntimeError, RoboflowError) as e:  # still being generated, or Roboflow's passing 'Internal Error'
            print(f"version {number} not ready ({e}); waiting", flush=True)
            time.sleep(VERSION_WAIT_S)
    raise TimeoutError(f"version {number} was not ready after {VERSION_WAIT_TRIES * VERSION_WAIT_S} s")


def start_autolabel(*, project: Project, batch_name: str, material: str) -> dict[str, object]:
    """Start SAM 3 Auto Label on one uploaded batch, with the material's text prompts.

    Args:
        project: the Roboflow project.
        batch_name: the batch made by ``upload_clip``.
        material: the clip's material; every outline found gets this class.

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
        batch_id, model=AUTOLABEL_MODEL, ontology={prompt: material for prompt in HOME_PROMPTS[material]},
        confidence=AUTOLABEL_CONFIDENCE, run_nms=True,  # several prompts often outline the same item
    )
    return result


def main() -> None:
    """Create or find the project, optionally upload a dataset, and print what Roboflow reports."""
    parser = argparse.ArgumentParser(description="Create or find the Roboflow project, upload datasets.")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="file holding ROBOFLOW_API_KEY")
    parser.add_argument("--upload", type=Path, help="converted dataset folder to upload")
    parser.add_argument("--batch", default="zerowaste-f", help="upload batch name")
    parser.add_argument("--home-clips", type=Path, help="capture folder: upload one-material clips, start SAM 3")
    parser.add_argument("--every", type=int, default=3, help="upload one home picture in this many")
    parser.add_argument("--source", default="home", help="batch and tag prefix for --home-clips, e.g. trashnet")
    parser.add_argument("--job", help="show the status of an auto-label job")
    parser.add_argument("--accept", action="append", default=[], help="move a done auto-label job into the dataset")
    parser.add_argument("--export", type=Path, help="make a version and download it here (YOLO format)")
    args = parser.parse_args()
    api_key = load_api_key(env_path=args.env)
    if args.upload is not None:
        print("upload status:", upload_dataset(api_key=api_key, dataset_dir=args.upload, batch_name=args.batch))
    project = get_or_create_project(api_key=api_key)
    if args.home_clips is not None:
        for clip_dir in sorted(d for d in args.home_clips.iterdir() if d.is_dir()):
            material = material_of_clip(clip=clip_dir.name)
            if material is None:
                print(f"{clip_dir.name}: skipped (not a one-material clip)", flush=True)
                continue
            batch_name = upload_clip(project=project, clip_dir=clip_dir, every=args.every, source=args.source)
            job = start_autolabel(project=project, batch_name=batch_name, material=material)
            print(f"{clip_dir.name}: uploaded as {batch_name}, SAM 3 job {job.get('jobId')}", flush=True)
    if args.job is not None:
        print(project.autolabel_job(args.job))
    for job_id in args.accept:
        print(f"{job_id}: {accept_autolabel(project=project, job_id=job_id)}", flush=True)
    if args.export is not None:
        print(f"exported to {export_version(project=project, out_dir=args.export)}")
    print(f"project {project.id} type={project.type} images={project.images} public={project.public}")
    return None


if __name__ == "__main__":
    main()
