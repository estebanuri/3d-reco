"""validator.py - Preflight environment checks for COLMAP automation.

Validates:
  - /home storage guard (skipped if local_dev=True)
  - Docker daemon availability (docker mode) or COLMAP binary existence (local mode)
  - Optional nvidia-smi check when use_gpu=True
  - Minimum free disk space on scratch_dir
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src.auto.config import AppConfig


@dataclass
class ValidationResult:
    """Result of preflight validation."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def validate_environment(config: AppConfig) -> ValidationResult:
    """Run all preflight checks and return a ValidationResult.

    Raises nothing; callers inspect result.passed and result.errors.
    """
    result = ValidationResult()

    # 1. /home storage guard (skip if local_dev)
    if not config.local_dev:
        scratch_resolved = config.scratch_dir.resolve()
        home = Path.home()
        try:
            scratch_resolved.relative_to(home)
            result.add_error(
                f"SCRATCH_DIR ({scratch_resolved}) is inside /home ({home}). "
                "Refusing to proceed to protect storage quotas. "
                "Set SCRATCH_DIR outside /home or use --local-dev."
            )
        except ValueError:
            pass  # Not under home -- OK

        if scratch_resolved == home:
            result.add_error(
                f"SCRATCH_DIR ({scratch_resolved}) is /home itself. "
                "Refusing to proceed."
            )

    # 2. Execution environment check
    if config.execution_mode == "docker":
        if shutil.which("docker") is None:
            result.add_error(
                "Docker CLI not found in PATH. Install Docker or switch to local mode."
            )
        else:
            try:
                subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                result.add_error(
                    f"Docker daemon is not reachable: {exc}. "
                    "Ensure Docker is running and the user has permission."
                )
    else:  # local
        colmap_bin = Path(config.colmap_bin)
        if not colmap_bin.exists() and shutil.which("colmap") is None:
            result.add_error(
                f"COLMAP binary not found at {colmap_bin} and not in PATH. "
                "Install COLMAP or switch to docker mode."
            )

    # 3. GPU check (nvidia-smi) when use_gpu is requested
    if config.use_gpu:
        if shutil.which("nvidia-smi") is None:
            result.add_warning(
                "nvidia-smi not found. GPU acceleration may not be available. "
                "Consider --no-use-gpu for CPU-only execution."
            )
        else:
            try:
                subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                result.add_warning(
                    f"nvidia-smi failed: {exc}. GPU may not be available."
                )

    # 4. Disk space check on scratch_dir (require at least 5 GB free)
    scratch_resolved = config.scratch_dir.resolve()
    try:
        scratch_resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result.add_error(f"Cannot create SCRATCH_DIR {scratch_resolved}: {exc}")
        return result

    try:
        disk = shutil.disk_usage(scratch_resolved)
        free_gb = disk.free / (1024**3)
        if free_gb < 5.0:
            result.add_error(
                f"Insufficient disk space in {scratch_resolved}: "
                f"{free_gb:.1f} GB free, minimum 5 GB required."
            )
        else:
            result.warnings.append(
                f"Disk space OK: {free_gb:.1f} GB free in {scratch_resolved}"
            )
    except OSError as exc:
        result.add_warning(f"Could not check disk space: {exc}")

    return result