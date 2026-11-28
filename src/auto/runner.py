"""runner.py - ColmapRunner: unified command builder and subprocess executor.

Transparently wraps Docker or local COLMAP binary execution. Streams stdout/stderr
to a unified pipeline log file inside the workspace log directory. Command headers
and progress are emitted via logging (console shows INFO by default, DEBUG with
--verbose). All subprocess output goes to the same pipeline.log.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from datetime import datetime

from pathlib import Path
from typing import Iterable, TextIO

# Interval (seconds) between real-time progress heartbeat messages while a
# COLMAP subprocess is running. Keeps the terminal from appearing frozen
# during long stages (e.g., mapper, patch_match_stereo).
_HEARTBEAT_INTERVAL = 20

from src.auto.config import AppConfig
from src.auto.logging_config import get_logger


logger = get_logger(__name__)

# Human-readable description for each COLMAP subcommand.
_SUBCOMMAND_DESCRIPTIONS: dict[str, str] = {
    "feature_extractor": "Extracting features",
    "exhaustive_matcher": "Matching features (exhaustive)",
    "sequential_matcher": "Matching features (sequential)",
    "vocabulary_tree_matcher": "Matching features (vocabulary tree)",
    "spatial_matcher": "Matching features (spatial verification)",
    "mapper": "Running sparse reconstruction (Mapper)",
    "model_converter": "Converting model",
    "bundle_adjuster": "Running bundle adjustment",
    "image_undistorter": "Undistorting images",
    "patch_match_stereo": "Running Patch-Match Stereo",
    "stereo_fusion": "Fusing stereo depth maps",
    "poisson_mesher": "Meshing (Poisson)",
    "delaunay_mesher": "Meshing (Delaunay)",
    "meshing_texturing": "Texturing mesh",
    "convert_model": "Converting model format",
}


def _subcommand_description(subcommand: str) -> str:
    """Return a human-readable label for a COLMAP subcommand."""
    return _SUBCOMMAND_DESCRIPTIONS.get(subcommand, subcommand)



class ColmapRunner:
    """Builds and executes COLMAP commands via Docker or local binary.

    Attributes:
        config: AppConfig instance.
        project_name: Identifier for the current project/dataset.
        timestamp: Timestamp string for workspace isolation.
        workspace_dir: Absolute path to the workspace directory.
        log_dir: Absolute path to the logs subdirectory.
        log_file: Open file handle for the main pipeline log.
    """

    def __init__(
        self,
        config: AppConfig,
        project_name: str,
        timestamp: str,
        *,
        workspace_dir: Path | None = None,
        is_resuming: bool = False,
    ):
        self.config = config
        self.project_name = project_name
        self.timestamp = timestamp
        self.is_resuming = is_resuming

        # Workspace: either an existing one to resume, or a fresh one.
        if workspace_dir is not None:
            self.workspace_dir = Path(workspace_dir).resolve()
        else:
            # Workspace: <scratch_dir>/workspaces/<project>_<timestamp>/
            self.workspace_dir = (
                config.scratch_dir.resolve() / "workspaces" / f"{project_name}_{timestamp}"
            )
        self.log_dir = self.workspace_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._log_file: TextIO | None = None

    @classmethod
    def resolve_workspace(
        cls,
        config: AppConfig,
        project_name: str,
        *,
        resume: str | None = None,
        is_new: bool = False,
    ) -> tuple[Path, bool, str]:
        """Resolve which workspace directory to use.

        Supports three selection modes:
          * ``--new``: always create a brand-new ``<project>_<timestamp>`` workspace.
          * ``--resume <name_or_path>``: target a specific existing workspace to resume.
            The value may be a directory name inside ``<scratch_dir>/workspaces/`` or a
            full path. Raises :exc:`FileNotFoundError` if the target does not exist.
          * Default (neither flag): search ``<scratch_dir>/workspaces/`` for the most
            recent ``<project>_*`` directory and reuse it; if none exist, create a new
            ``<project>_<timestamp>`` workspace.

        Args:
            config: AppConfig instance (provides scratch_dir).
            project_name: Project identifier used for workspace naming.
            resume: Name or path of an existing workspace to resume.
            is_new: If True, always create a brand-new workspace.

        Returns:
            Tuple of (workspace_dir, is_resuming, timestamp). ``is_resuming`` is True
            when reusing an existing workspace; ``timestamp`` is either parsed from an
            existing workspace name or freshly minted.

        Raises:
            FileNotFoundError: if the ``--resume`` target does not exist or is not a
                directory.
        """
        workspaces_root = config.scratch_dir.resolve() / "workspaces"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_new:
            # Brand-new workspace: <project>_<timestamp>
            workspace_dir = workspaces_root / f"{project_name}_{timestamp}"
            return workspace_dir, False, timestamp

        if resume is not None:
            # Explicit target: either a dir name inside workspaces/ or a full path.
            candidate = Path(resume)
            if not candidate.is_absolute():
                candidate = workspaces_root / candidate
            candidate = candidate.resolve()
            if not candidate.is_dir():
                raise FileNotFoundError(
                    f"Workspace to resume not found: {candidate}\n"
                    f"Expected a directory name inside {workspaces_root} or a full path "
                    f"to an existing workspace directory."
                )
            # Parse the timestamp from the directory name if it matches <project>_<ts>.
            name = candidate.name
            if "_" in name:
                ts_candidate = name.rsplit("_", 1)[-1]
                if (
                    len(ts_candidate) == 15
                    and ts_candidate[:8].isdigit()
                    and ts_candidate[9:].isdigit()
                ):
                    timestamp = ts_candidate
            return candidate, True, timestamp

        # Default: search for the most recent <project>_* workspace.
        if workspaces_root.is_dir():
            matches = sorted(
                workspaces_root.glob(f"{project_name}_*"),
                key=lambda p: p.name,
            )
            if matches:
                latest = matches[-1]
                return latest, True, timestamp

        # None exist -> create new.
        workspace_dir = workspaces_root / f"{project_name}_{timestamp}"
        return workspace_dir, False, timestamp

    # ------------------------------------------------------------------ #
    # Command building
    # ------------------------------------------------------------------ #
    def _docker_prefix(self) -> list[str]:
        """Build the Docker run prefix for transparent container execution."""
        scratch_str = str(self.config.scratch_dir.resolve())
        return [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--user",
            self.config.docker_uid_gid,
            "-e",
            "QT_QPA_PLATFORM=offscreen",
            "-v",
            f"{scratch_str}:{scratch_str}",
            self.config.docker_image,
            "colmap",
        ]

    def _local_prefix(self) -> list[str]:
        """Return the local binary prefix."""
        return [str(self.config.colmap_bin)]

    def build_command(self, subcommand: str, *args: str) -> list[str]:
        """Build the full command list for a COLMAP subcommand.

        Args:
            subcommand: COLMAP subcommand (e.g., "feature_extractor").
            *args: Additional arguments passed to the subcommand.

        Returns:
            Full command list ready for subprocess execution.
        """
        if self.config.execution_mode == "docker":
            prefix = self._docker_prefix()
        else:
            prefix = self._local_prefix()
        return prefix + [subcommand, *args]

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _open_log(self, log_name: str = "pipeline.log") -> TextIO:
        """Open (or reuse) the pipeline log file."""
        if self._log_file is None or self._log_file.closed:
            log_path = self.log_dir / log_name
            self._log_file = open(log_path, "a", encoding="utf-8")
        return self._log_file

    def _close_log(self) -> None:
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()

    def run(
        self,
        command_parts: Iterable[str],
        log_name: str = "pipeline.log",
        check: bool = True,
    ) -> int:
        """Execute a command, streaming stdout/stderr to the log file.

        Command header and progress are emitted via logging so they appear on
        the console at INFO level (default) or DEBUG level with --verbose.
        Full subprocess stdout/stderr always goes to the unified pipeline log file.

        Args:
            command_parts: Command and arguments as an iterable of strings.
            log_name: Name of the log file inside <log_dir>/.
            check: If True, raise CalledProcessError on non-zero exit.

        Returns:
            The process return code.

        Raises:
            subprocess.CalledProcessError: If check=True and the command fails.
        """
        cmd = list(command_parts)
        log_file = self._open_log(log_name)

        # Write command header to the step log file (always) and to the
        # console via logging at INFO level (visible by default).
        header = f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}\n"
        logger.info("Running: %s", " ".join(cmd))
        log_file.write(header)
        log_file.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=self.config.env,
                bufsize=1,
            )

            assert proc.stdout is not None

            # Real-time progress heartbeat: keep the terminal alive during
            # long-running stages that emit little/no stdout (e.g., mapper,
            # patch_match_stereo). Runs independently of the blocking
            # stdout-read loop below so progress is printed even while COLMAP
            # is silent.
            stop_heartbeat = threading.Event()

            def _heartbeat() -> None:
                start = time.monotonic()
                cmd_str = " ".join(cmd)
                while not stop_heartbeat.wait(_HEARTBEAT_INTERVAL):
                    elapsed = int(time.monotonic() - start)
                    logger.info(
                        "Still running (%ds elapsed): %s", elapsed, cmd_str
                    )

            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()

            try:
                for line in proc.stdout:
                    # Full subprocess output always goes to the log file.
                    log_file.write(line)
                    log_file.flush()
                    # Only echo to console at DEBUG level (--verbose).
                    logger.debug(line.rstrip("\n"))
            finally:
                stop_heartbeat.set()
                heartbeat_thread.join()

            return_code = proc.wait()

            logger.info("Command finished with return code %d.", return_code)

            if return_code != 0 and check:
                raise subprocess.CalledProcessError(return_code, cmd)

            return return_code

        except subprocess.CalledProcessError:
            raise
        finally:
            self._close_log()

    def run_colmap(
        self,
        subcommand: str,
        *args: str,
        log_name: str | None = None,
        check: bool = True,
    ) -> int:
        """Convenience: build and run a COLMAP subcommand.

        Args:
            subcommand: COLMAP subcommand name.
            *args: Arguments for the subcommand.
            log_name: Optional log file name (defaults to pipeline.log).
            check: If True, raise on non-zero exit.

        Returns:
            Process return code.
        """
        cmd = self.build_command(subcommand, *args)
        step_log = log_name or "pipeline.log"
        desc = _subcommand_description(subcommand)
        logger.info("%s ( %s )", desc, subcommand)
        return self.run(cmd, log_name=step_log, check=check)

    def __enter__(self) -> ColmapRunner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._close_log()
