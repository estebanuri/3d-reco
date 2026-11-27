"""main.py - CLI entry point for COLMAP Automatizer.

Selects project, execution mode, and flags; runs preflight validation and
the full SfM pipeline.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.auto.config import AppConfig
from src.auto.pipeline import SfMPipeline
from src.auto.runner import ColmapRunner
from src.auto.validator import validate_environment
from src.prj.odm_bellus import OdmBellusProject


PROJECT_REGISTRY: dict[str, type] = {
    "odm_bellus": OdmBellusProject,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COLMAP Automatizer - Automated COLMAP execution pipeline "
        "(Docker on university server, local binary for dev).",
    )
    parser.add_argument(
        "--env",
        default="local",
        choices=["local", "remote"],
        help="Environment profile: selects .env.<env> file (default: local).",
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
        default=True,
        help="Enable GPU acceleration (default: True).",
    )
    parser.add_argument(
        "--no-use-gpu",
        dest="use_gpu",
        action="store_false",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1. Load configuration from .env.<env>
    config = AppConfig.load_from_env(env_name=args.env)

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

    # 3. Preflight validation
    if not args.skip_validation:
        result = validate_environment(config)
        for warning in result.warnings:
            print(f"[WARNING] {warning}")
        if not result.passed:
            for error in result.errors:
                print(f"[ERROR] {error}", file=sys.stderr)
            print("Preflight validation failed. Aborting.", file=sys.stderr)
            return 1
        print("[OK] Preflight validation passed.")

    # 4. Instantiate project and fetch data
    project_cls = PROJECT_REGISTRY[args.project]
    project = project_cls(scratch_dir=config.scratch_dir, force_download=args.force_download)

    try:
        dataset_root = project.fetch_data(force_download=args.force_download)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch dataset: {exc}", file=sys.stderr)
        return 1

    image_dir = project.get_image_dir()
    camera_params = project.get_camera_params()

    print(f"[INFO] Project: {project.name}")
    print(f"[INFO] Dataset: {dataset_root}")
    print(f"[INFO] Images: {image_dir}")
    print(f"[INFO] Camera params: {camera_params}")
    print(f"[INFO] Execution mode: {config.execution_mode}")
    print(f"[INFO] Scratch dir: {config.scratch_dir}")
    print(f"[INFO] GPU: {config.use_gpu}")

    # 5. Set up runner and pipeline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner = ColmapRunner(config, project.get_workspace_name(), timestamp)

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
        print(f"[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        return 1

    # 7. Summary
    print("\n" + "=" * 70)
    print("PILINE COMPLETE")
    print("=" * 70)
    for key, path in outputs.items():
        print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())