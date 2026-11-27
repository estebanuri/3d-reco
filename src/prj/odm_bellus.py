"""odm_bellus.py - OdmBellusProject: OpenDroneMap Bellus dataset implementation.

Clones the ODM Bellus dataset, validates the images directory, and provides
COLMAP camera parameters tailored to this drone dataset.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.auto.logging_config import get_logger
from src.prj.base import BaseProject


logger = get_logger(__name__)


class OdmBellusProject(BaseProject):
    """Project implementation for the OpenDroneMap Bellus dataset.

    Dataset: https://github.com/OpenDroneMap/odm_data_bellus.git
    """

    GIT_URL = "https://github.com/OpenDroneMap/odm_data_bellus.git"
    DATASET_DIR_NAME = "odm_data_bellus"

    def __init__(self, scratch_dir: Path, force_download: bool = False):
        self.scratch_dir = Path(scratch_dir).resolve()
        self.force_download = force_download
        self.dataset_dir = self.scratch_dir / "datasets" / self.DATASET_DIR_NAME

    @property
    def name(self) -> str:
        return "odm_bellus"

    def get_workspace_name(self) -> str:
        return "odm_bellus"

    def fetch_data(self, force_download: bool | None = None) -> Path:
        """Clone the ODM Bellus dataset into <scratch>/datasets/odm_data_bellus.

        Skips if already present unless force_download is True.

        Returns:
            Absolute path to the dataset root directory.
        """
        force = self.force_download if force_download is None else force_download

        self.dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        if self.dataset_dir.exists() and not force:
            logger.info("[odm_bellus] Dataset already present at %s", self.dataset_dir)
            return self.dataset_dir

        if self.dataset_dir.exists() and force:
            logger.info("[odm_bellus] Removing existing dataset at %s", self.dataset_dir)
            import shutil
            shutil.rmtree(self.dataset_dir)

        logger.info("[odm_bellus] Cloning %s -> %s", self.GIT_URL, self.dataset_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", self.GIT_URL, str(self.dataset_dir)],
            check=True,
        )
        return self.dataset_dir

    def get_image_dir(self) -> Path:
        """Return the absolute path to the images/ subdirectory inside the dataset."""
        image_dir = self.dataset_dir / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Images directory not found: {image_dir}. "
                f"Run fetch_data() first to clone the dataset."
            )
        return image_dir

    def get_camera_params(self) -> dict[str, str]:
        """Return recommended COLMAP camera parameters for the Bellus drone dataset.

        Bellus is a single-camera drone dataset, so SIMPLE_RADIAL with
        single_camera=1 is appropriate.
        """
        return {
            "camera_model": "SIMPLE_RADIAL",
            "single_camera": "1",
        }