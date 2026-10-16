"""Talking to Roboflow: API key, the project, uploads.

The key lives in the repo's git-ignored ``.env`` as ``ROBOFLOW_API_KEY=...`` and is never printed.
Run ``uv run python -m door_counter.roboflow_io`` to create (or find) the project.
"""

import argparse
from pathlib import Path

import roboflow
from roboflow.core.project import Project

WORKSPACE = "<workspace>"
PROJECT_ID = "sorting-line"
PROJECT_NAME = "sorting-line"
PROJECT_TYPE = "instance-segmentation"
ANNOTATION_GROUP = "materials"
KEY_NAME = "ROBOFLOW_API_KEY"


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


def main() -> None:
    """Create or find the project and print what Roboflow reports about it."""
    parser = argparse.ArgumentParser(description="Create or find the Roboflow project.")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="file holding ROBOFLOW_API_KEY")
    args = parser.parse_args()
    project = get_or_create_project(api_key=load_api_key(env_path=args.env))
    print(f"project {project.id} type={project.type} images={project.images} public={project.public}")
    return None


if __name__ == "__main__":
    main()
