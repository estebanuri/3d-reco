"""base.py - Abstract BaseProject defining the project/dataset interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseProject(ABC):
    """Abstract base class for COLMAP project/dataset definitions.

    Subclasses implement dataset fetching, image directory resolution, and
    recommended COLMAP camera parameters for a specific dataset.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short project identifier (used for workspace and log directory naming)."""

    @abstractmethod
    def fetch_data(self, force_download: bool = False) -> Path:
        """Download/clone the dataset if needed and return its root directory path.

        Args:
            force_download: If True, re-download even if already present.

        Returns:
            Absolute path to the dataset root directory.
        """

    @abstractmethod
    def get_image_dir(self) -> Path:
        """Return the absolute path to the directory containing input images."""

    @abstractmethod
    def get_camera_params(self) -> dict[str, str]:
        """Return recommended COLMAP camera parameters for this dataset.

        Returns:
            Dict of COLMAP option -> value (e.g., {"single_camera": "1",
            "camera_model": "SIMPLE_RADIAL"}).
        """

    @abstractmethod
    def get_workspace_name(self) -> str:
        """Return a workspace-friendly name for this project."""