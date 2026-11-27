"""main.py - CLI entry point for COLMAP Automatizer.

Selects project, execution mode, and flags; runs preflight validation and
the full SfM pipeline. All application messages are routed through Python
logging and written to a log file. By default the terminal shows
progress milestones (INFO). Full stdout/stderr goes to the log file.
Use ``--verbose`` to show everything on the terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.auto.config import AppConfig
from src.auto.logging_config import get_logger, setup_logging
from src.auto.pipeline import SfMPipeline
from src.auto.runner import ColmapRunner
from src.auto.validator import validate_environment
from src.prj.odm_bellus import OdmBellusProject

logger = get_logger(__name__)


PROJECT_REGISTRY: dict[str, type] = {
    "odm_bellus": OdmBellusProject,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COLMAP Automatizer - Automated COLMAP execution pipeline "
        "(Docker on university server, local binary for dev).",
    )
    parser.add_argument(
        "--project",
        default="odm_bellus",
        choices=list(PROJECT_REGISTRY.keys()),
        help="Project/dataset to process (default: odm_bellus).",
    )
    parser.add_argument(
        "--matcher",
        default="exhaustive",
        choices=["exhaustive", "sequential"],
        help="Feature matching strategy (default: exhaustive).",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Enable dense MVS reconstruction (image_undistorter, patch_match_stereo, stereo_fusion).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download/re-clone dataset even if already present.",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        default=None,
        help="Enable GPU acceleration.",
    )
    parser.add_argument(
        "--no-use-gpu",
        dest="use_gpu",
        action="store_false",
        default=None,
        help="Disable GPU acceleration (CPU-only).",
    )
    parser.add_argument(
        "--local-dev",
        action="store_true",
        default=False,
        help="Allow scratch_dir inside /home for local development.",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Override SCRATCH_DIR from .env.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip preflight environment validation (not recommended).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show all log messages including DEBUG on the terminal (default: INFO progress only).",
    )
    parser.add_argument(
        "--log_path",
        default=None,
        help="Path to the log file (default: <scratch_dir>/workspaces/<project>_<timestamp>/logs/pipeline.log).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level for the log file (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Console shows progress milestones by default; everything when --verbose.
    console_level = args.log_level if args.verbose else "INFO"

    # Set up console logging immediately so early errors are visible
    # even if config loading fails.
    setup_logging(log_path=None, log_level=args.log_level, console_level=console_level)

    # 1. Load configuration from .env
    try:
        config = AppConfig.load_from_env()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # 2. Apply CLI overrides
    override_kwargs: dict = {}
    if args.use_gpu is not None:
        override_kwargs["use_gpu"] = args.use_gpu
    if args.local_dev:
        override_kwargs["local_dev"] = True
    if args.scratch_dir is not None:
        override_kwargs["scratch_dir"] = args.scratch_dir

    if override_kwargs:
        config = config.model_copy_with_overrides(**override_kwargs)

    # Now that scratch_dir is known, add a file handler in scratch_dir
    # (falls back to console-only if scratch_dir is unusable).
    try:
        early_log_path = config.scratch_dir / "colmap_auto.log"
        setup_logging(
            log_path=early_log_path,
            log_level=args.log_level,
            console_level=console_level,
        )
        logger.info("Logging to %s", early_log_path)
    except (OSError, AttributeError) as exc:
        logger.warning("Could not open early log file: %s", exc)

    # 3. Preflight validation
    if not args.skip_validation:
        result = validate_environment(config)
        for warning in result.warnings:
            logger.warning("%s", warning)
        if not result.passed:
            for error in result.errors:
                logger.error("%s", error)
            logger.error("Preflight validation failed. Aborting.")
            return 1
        logger.info("Preflight validation passed.")

    # 4. Instantiate project and fetch data
    project_cls = PROJECT_REGISTRY[args.project]
    project = project_cls(scratch_dir=config.scratch_dir, force_download=args.force_download)

    try:
        dataset_root = project.fetch_data(force_download=args.force_download)
    except Exception as exc:
        logger.exception("Failed to fetch dataset: %s", exc)
        return 1

    image_dir = project.get_image_dir()
    camera_params = project.get_camera_params()

    logger.info("Project: %s", project.name)
    logger.info("Dataset: %s", dataset_root)
    logger.info("Images: %s", image_dir)
    logger.info("Camera params: %s", camera_params)
    logger.info("Execution mode: %s", config.execution_mode)
    logger.info("Scratch dir: %s", config.scratch_dir)
    logger.info("GPU: %s", config.use_gpu)

    # 5. Set up runner and pipeline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner = ColmapRunner(config, project.get_workspace_name(), timestamp)

    # Unify application logging into the same file the runner uses for
    # COLMAP subprocess output so there is a single log per workspace.
    runner_log_path = args.log_path or (runner.log_dir / "pipeline.log")
    setup_logging(
        log_path=runner_log_path,
        log_level=args.log_level,
        console_level=console_level,
    )
    logger.info("Unified logging to %s", runner_log_path)

    pipeline = SfMPipeline(
        runner=runner,
        image_dir=image_dir,
        workspace_dir=runner.workspace_dir,
    )

    # 6. Execute pipeline
    try:
        outputs = pipeline.run(
            matcher=args.matcher,
            camera_model=camera_params["camera_model"],
            single_camera=camera_params.get("single_camera", "1") == "1",
            dense=args.dense,
            use_gpu=config.use_gpu,
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    # 7. Summary
    logger.info("\n%s\nPILINE COMPLETE\n%s", "=" * 70, "=" * 70)
    for key, path in outputs.items():
        logger.info("  %s: %s", key, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
