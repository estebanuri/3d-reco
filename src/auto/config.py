"""AppConfig - Centralized configuration for COLMAP automation.

Loads settings from environment variables / .env files, with CLI flag overrides.
Enforces the /home storage guard when running in remote (non-local-dev) mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator


class AppConfig(BaseModel):
    """Typed configuration for COLMAP execution.

    Attributes:
        execution_mode: "docker" for university server, "local" for local dev.
        docker_image: Docker image tag (used when execution_mode == "docker").
        colmap_bin: Path to COLMAP binary (used when execution_mode == "local").
        scratch_dir: Absolute path to the SCRATCH root directory.
        use_gpu: Enable GPU acceleration (default True for RTX 4090).
        local_dev: Safety switch. If False, abort if scratch_dir is inside /home.
    """

    execution_mode: Literal["docker", "local"] = Field(
        default="local",
        description="Execution mode: 'docker' or 'local'.",
    )
    docker_image: str = Field(
        default="colmap/colmap:latest",
        description="Docker image for COLMAP (docker mode only).",
    )
    colmap_bin: str = Field(
        default="/usr/local/bin/colmap",
        description="Path to COLMAP binary (local mode only).",
    )
    scratch_dir: Path = Field(
        default=Path("./scratch"),
        description="Absolute path to SCRATCH root directory.",
    )
    use_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration.",
    )
    local_dev: bool = Field(
        default=True,
        description="Allow scratch_dir inside /home for local development.",
    )

    @field_validator("scratch_dir", mode="before")
    @classmethod
    def _expand_scratch_dir(cls, v: object) -> Path:
        """Expand environment variables and user home in scratch_dir path."""
        if isinstance(v, Path):
            return v
        if not isinstance(v, str):
            return Path(v)
        expanded = os.path.expandvars(v)
        expanded = os.path.expanduser(expanded)
        return Path(expanded)

    @model_validator(mode="after")
    def _validate_scratch_not_in_home(self) -> AppConfig:
        """Abort if scratch_dir is inside /home and local_dev is False."""
        if self.local_dev:
            return self

        scratch_resolved = self.scratch_dir.resolve()
        home = Path.home()

        # Check if scratch_dir is /home or a subpath of /home
        try:
            scratch_resolved.relative_to(home)
            raise ValueError(
                f"SCRATCH_DIR ({scratch_resolved}) resides inside /home ({home}). "
                "This violates the university storage quota policy. "
                "Set SCRATCH_DIR to a path outside /home (e.g., /scratch/$USER/colmap_auto) "
                "or set LOCAL_DEV=true for local development only."
            )
        except ValueError:
            # relative_to raises ValueError if the path is NOT under home -- that's fine
            # Check if the error was our custom one or the relative_to one
            pass

        # If scratch_dir == home exactly, relative_to returns "." which doesn't raise
        if scratch_resolved == home:
            raise ValueError(
                f"SCRATCH_DIR ({scratch_resolved}) is /home itself. "
                "This violates the university storage quota policy."
            )

        return self

    @property
    def docker_uid_gid(self) -> str:
        """Return 'uid:gid' string for Docker --user flag."""
        return f"{os.getuid()}:{os.getgid()}"

    @property
    def env(self) -> dict[str, str]:
        """Environment variables to inject into subprocess (offscreen Qt, etc.)."""
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        if not self.use_gpu:
            env["COLMAP_USE_GPU"] = "0"
        return env

    @classmethod
    def load_from_env(cls, env_name: str = "local") -> AppConfig:
        """Load configuration from .env.<env_name> file, then environment variables.

        CLI overrides are applied after construction by the caller.
        """
        env_file = Path(__file__).resolve().parents[3] / f".env.{env_name}"
        if env_file.exists():
            load_dotenv(env_file, override=False)
        else:
            # Fall back to generic .env
            generic_env = Path(__file__).resolve().parents[3] / ".env"
            if generic_env.exists():
                load_dotenv(generic_env, override=False)

        # Load from system environment (already set or from .env above)
        execution_mode = os.getenv("COLMAP_MODE", "local")
        docker_image = os.getenv("DOCKER_IMAGE", "colmap/colmap:latest")
        colmap_bin = os.getenv("COLMAP_BIN", "/usr/local/bin/colmap")
        scratch_dir = os.getenv("SCRATCH_DIR", "./scratch")
        use_gpu = os.getenv("USE_GPU", "true").lower() in ("1", "true", "yes", "on")
        local_dev = os.getenv("LOCAL_DEV", "true").lower() in ("1", "true", "yes", "on")

        return cls(
            execution_mode=execution_mode,  # type: ignore[arg-type]
            docker_image=docker_image,
            colmap_bin=colmap_bin,
            scratch_dir=scratch_dir,
            use_gpu=use_gpu,
            local_dev=local_dev,
        )

    def model_copy_with_overrides(self, **kwargs) -> AppConfig:
        """Return a copy of this config with the given fields overridden."""
        return self.model_copy(update=kwargs)