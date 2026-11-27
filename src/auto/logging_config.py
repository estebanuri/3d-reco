"""logging_config.py - Centralized logging setup for the COLMAP Automatizer.

Configures the root Python logger so that application-level messages
(INFO / WARNING / ERROR / tracebacks) are written to a log file.
COLMAP subprocess output is written to the same log file
by :class:`src.auto.runner.ColmapRunner`, giving a single unified log
per workspace. Console output is optional and level-controlled.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_path: Path | str | None = None,
    log_level: str | int = "INFO",
    *,
    console_level: str | int | None = None,
) -> logging.Logger:
    """Configure the root logger with optional console and file handlers.

    Any previously attached handlers are removed so this can be called
    multiple times (e.g., first with a default path, then again with the
    workspace log path once the :class:`ColmapRunner` is instantiated).

    Args:
        log_path: Path to the log file. If ``None``, no file handler is
            added (console only).
        log_level: Logging level name (e.g. ``"DEBUG"``, ``"INFO"``) or
            integer constant for the file handler and root logger.
            Defaults to ``"INFO"``.
        console_level: Level for the console stream handler. If ``None``,
            no console handler is added (nothing prints to terminal).
            If set, records at this level and above are shown on the
            terminal. Use ``"ERROR"`` to show only errors, or the same
            value as ``log_level`` to show everything (e.g. ``--verbose``).

    Returns:
        The configured root logger.
    """
    level = log_level if isinstance(log_level, int) else getattr(
        logging, str(log_level).upper(), logging.INFO
    )
    console_lvl = (
        console_level if isinstance(console_level, int)
        else getattr(logging, str(console_level).upper(), logging.ERROR)
        if console_level is not None else None
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers so re-configuration is clean.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_lvl is not None:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(console_lvl)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for the given module."""
    return logging.getLogger(name)
