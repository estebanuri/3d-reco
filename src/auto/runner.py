"""runner.py - ColmapRunner: unified command builder and subprocess executor.

Transparently wraps Docker or local COLMAP binary execution. Streams stdout/stderr
to a unified pipeline log file inside the workspace log directory. Command headers
and progress are emitted via logging (console shows INFO by default, DEBUG with
--verbose). All subprocess output goes to the same pipeline.log.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, TextIO

from src.auto.config import AppConfig
from src.auto.logging_config import get_logger


logger = get_logger(__name__)


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

    def __init__(self, config: AppConfig, project_name: str, timestamp: str):
        self.config = config
        self.project_name = project_name
        self.timestamp = timestamp

        # Workspace: <scratch_dir>/workspaces/<project>_<timestamp>/
        self.workspace_dir = (
            config.scratch_dir.resolve() / "workspaces" / f"{project_name}_{timestamp}"
        )
        self.log_dir = self.workspace_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._log_file: TextIO | None = None

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
            for line in proc.stdout:
                # Full subprocess output always goes to the log file.
                log_file.write(line)
                log_file.flush()
                # Only echo to console at DEBUG level (--verbose).
                logger.debug(line.rstrip("\n"))

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
        return self.run(cmd, log_name=step_log, check=check)

    def __enter__(self) -> ColmapRunner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._close_log()
