"""Talking to Roboflow: API key, the project, uploads and versions.

The key (the account's private API key) lives in the repo's git-ignored ``.env`` as ``ROBOFLOW_API_KEY=...``
and is never printed. The workspace is the one the key belongs to.
Run ``uv run python -m door_counter.roboflow_io`` to create (or find) the keypoint project,
``... --upload data/fall-coco --batch caucafall`` to upload a dataset in COCO keypoint layout (see
``door_counter.fall_data``), and ``... --export data/fall-v1`` to make a version and download it.
"""

import argparse
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import roboflow
import yaml
from roboflow.adapters.rfapi import RoboflowError
from roboflow.core.project import Project

from door_counter.fall_data import FLIP_INDEX

PROJECT_ID = "fall-watch"
PROJECT_TYPE = "keypoint-detection"
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


def get_or_create_project(*, api_key: str, project_id: str = PROJECT_ID) -> Project:
    """Open the fall-watch project (or another one) in the key's workspace, creating it (private) on first use.

    Args:
        api_key: the Roboflow API key.
        project_id: the project's id.

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
    workspace = roboflow.Roboflow(api_key=api_key).workspace()
    return workspace.upload_dataset(
        dataset_path=str(zip_path),
        project_name=PROJECT_ID,
        project_license="Private",
        project_type=PROJECT_TYPE,
        batch_name=batch_name,
        poll_timeout=3 * 3600.0,
    )


def export_version(*, project: Project, out_dir: Path) -> Path:
    """Make a dataset version with no resizing or augmentation, and download it in YOLO format.

    Roboflow writes ``flip_idx`` as 0..16 (no left/right swap); for the COCO skeleton that would teach
    mirrored pictures the wrong side, so the COCO swap is written back.

    Args:
        project: the Roboflow project.
        out_dir: where the export goes (``data.yaml``, ``train/``, ``valid/``, ``test/``).

    Returns:
        The export folder.
    """
    number = project.generate_version(settings={"augmentation": {}, "preprocessing": {"auto-orient": True}})
    for _ in range(VERSION_WAIT_TRIES):
        try:
            project.version(number).download("yolov8", location=str(out_dir), overwrite=True)
            data_yaml = out_dir / "data.yaml"
            spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            spec.update(flip_idx=list(FLIP_INDEX), path=str(out_dir.resolve()), train="train/images",
                        val="valid/images", test="test/images")
            data_yaml.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            return out_dir
        except (RuntimeError, RoboflowError) as e:  # still being generated, or Roboflow's passing 'Internal Error'
            print(f"version {number} not ready ({e}); waiting", flush=True)
            time.sleep(VERSION_WAIT_S)
    raise TimeoutError(f"version {number} was not ready after {VERSION_WAIT_TRIES * VERSION_WAIT_S} s")


def main() -> None:
    """Create or find the project, optionally upload a dataset or export a version, and print what Roboflow reports."""
    parser = argparse.ArgumentParser(description="Create or find the Roboflow project, upload datasets, export.")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="file holding ROBOFLOW_API_KEY")
    parser.add_argument("--upload", type=Path, help="dataset folder in COCO keypoint layout to upload")
    parser.add_argument("--batch", default="caucafall", help="upload batch name")
    parser.add_argument("--export", type=Path, help="make a version and download it here (YOLO format)")
    args = parser.parse_args()
    api_key = load_api_key(env_path=args.env)
    if args.upload is not None:
        print("upload status:", upload_dataset(api_key=api_key, dataset_dir=args.upload, batch_name=args.batch))
    project = get_or_create_project(api_key=api_key)
    if args.export is not None:
        print(f"exported to {export_version(project=project, out_dir=args.export)}")
    print(f"project {project.id} type={project.type} images={project.images} public={project.public}")
    return None


if __name__ == "__main__":
    main()
