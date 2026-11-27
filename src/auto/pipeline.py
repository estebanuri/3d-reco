"""pipeline.py - End-to-end SfM pipeline orchestration.

Executes: feature_extractor -> matcher -> mapper -> model_converter,
plus optional dense MVS (image_undistorter -> patch_match_stereo -> stereo_fusion).
"""

from __future__ import annotations

from pathlib import Path

from src.auto.runner import ColmapRunner


class SfMPipeline:
    """Orchestrates the COLMAP SfM (and optional MVS dense) pipeline.

    Attributes:
        runner: ColmapRunner instance for executing commands.
        image_dir: Absolute path to the input images directory.
        workspace_dir: Absolute path to the workspace/output directory.
        database_path: Path to the COLMAP database (database.db).
        sparse_dir: Path to the sparse reconstruction output directory.
        dense_dir: Path to the dense reconstruction output directory.
    """

    def __init__(
        self,
        runner: ColmapRunner,
        image_dir: Path,
        workspace_dir: Path,
    ):
        self.runner = runner
        self.image_dir = Path(image_dir).resolve()
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.database_path = self.workspace_dir / "database.db"
        self.sparse_dir = self.workspace_dir / "sparse"
        self.dense_dir = self.workspace_dir / "dense"

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #
    def feature_extractor(
        self,
        camera_model: str = "SIMPLE_RADIAL",
        single_camera: bool = True,
        sift_use_gpu: bool | None = None,
    ) -> None:
        """Run COLMAP feature_extractor on the image directory."""
        args = [
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", str(int(single_camera)),
        ]
        if sift_use_gpu is not None:
            args.extend(["--FeatureExtraction.use_gpu", str(int(sift_use_gpu))])
        elif not self.runner.config.use_gpu:
            args.extend(["--FeatureExtraction.use_gpu", "0"])

        self.runner.run_colmap("feature_extractor", *args)

    def exhaustive_matcher(self, use_gpu: bool | None = None) -> None:
        """Run COLMAP exhaustive_matcher (all-pairs matching)."""
        args = ["--database_path", str(self.database_path)]
        if use_gpu is not None:
            args.extend(["--FeatureMatching.use_gpu", str(int(use_gpu))])
        elif not self.runner.config.use_gpu:
            args.extend(["--FeatureMatching.use_gpu", "0"])

        self.runner.run_colmap("exhaustive_matcher", *args)

    def sequential_matcher(self, use_gpu: bool | None = None) -> None:
        """Run COLMAP sequential_matcher (sequential matching)."""
        args = ["--database_path", str(self.database_path)]
        if use_gpu is not None:
            args.extend(["--FeatureMatching.use_gpu", str(int(use_gpu))])
        elif not self.runner.config.use_gpu:
            args.extend(["--FeatureMatching.use_gpu", "0"])

        self.runner.run_colmap("sequential_matcher", *args)

    def mapper(self) -> Path:
        """Run COLMAP mapper to produce sparse reconstruction.

        Returns:
            Path to the sparse reconstruction directory (sparse/0/).
        """
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.runner.run_colmap(
            "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--output_path", str(self.sparse_dir),
        )
        return self.sparse_dir / "0"

    def model_converter(
        self,
        sparse_dir: Path | None = None,
        output_format: str = "TXT",
    ) -> None:
        """Convert COLMAP binary model to text and/or PLY formats.

        Args:
            sparse_dir: Path to sparse/0/ (defaults to self.sparse_dir / "0").
            output_format: "TXT", "PLY", or "BOTH".
        """
        if sparse_dir is None:
            sparse_dir = self.sparse_dir / "0"

        # Convert binary -> text (.txt)
        if output_format in ("TXT", "BOTH"):
            txt_dir = sparse_dir.parent / "sparse_txt"
            txt_dir.mkdir(parents=True, exist_ok=True)
            self.runner.run_colmap(
                "model_converter",
                "--input_path", str(sparse_dir),
                "--output_path", str(txt_dir),
                "--output_type", "TXT",
            )

        # Convert binary -> PLY (for visualization / Gaussian Splatting)
        if output_format in ("PLY", "BOTH"):
            ply_dir = sparse_dir.parent / "sparse_ply"
            ply_dir.mkdir(parents=True, exist_ok=True)
            self.runner.run_colmap(
                "model_converter",
                "--input_path", str(sparse_dir),
                "--output_path", str(ply_dir),
                "--output_type", "PLY",
            )

    # ------------------------------------------------------------------ #
    # Optional dense MVS
    # ------------------------------------------------------------------ #
    def dense_pipeline(self) -> None:
        """Run the full dense MVS pipeline:
        image_undistorter -> patch_match_stereo -> stereo_fusion.
        """
        self.dense_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: image_undistorter
        self.runner.run_colmap(
            "image_undistorter",
            "--image_path", str(self.image_dir),
            "--input_path", str(self.sparse_dir / "0"),
            "--output_path", str(self.dense_dir),
        )

        # Step 2: patch_match_stereo
        dense_stereo_dir = self.dense_dir / "stereo"
        dense_stereo_dir.mkdir(parents=True, exist_ok=True)
        self.runner.run_colmap(
            "patch_match_stereo",
            "--workspace_path", str(self.dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", str(0 if self.runner.config.use_gpu else -1),
        )

        # Step 3: stereo_fusion
        self.runner.run_colmap(
            "stereo_fusion",
            "--workspace_path", str(self.dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(self.dense_dir / "fused.ply"),
        )

    # ------------------------------------------------------------------ #
    # Full orchestration
    # ------------------------------------------------------------------ #
    def run(
        self,
        matcher: str = "exhaustive",
        camera_model: str = "SIMPLE_RADIAL",
        single_camera: bool = True,
        dense: bool = False,
        use_gpu: bool | None = None,
    ) -> dict[str, Path]:
        """Run the complete SfM pipeline.

        Args:
            matcher: "exhaustive" or "sequential".
            camera_model: Camera model for feature extraction.
            single_camera: Whether all images share one camera.
            dense: If True, also run dense MVS.
            use_gpu: Override GPU usage for matching steps.

        Returns:
            Dict of output path names to Path objects.
        """
        # 1. Feature extraction
        self.feature_extractor(
            camera_model=camera_model,
            single_camera=single_camera,
            sift_use_gpu=use_gpu,
        )

        # 2. Matching
        if matcher == "exhaustive":
            self.exhaustive_matcher(use_gpu=use_gpu)
        elif matcher == "sequential":
            self.sequential_matcher(use_gpu=use_gpu)
        else:
            raise ValueError(f"Unknown matcher: {matcher}. Use 'exhaustive' or 'sequential'.")

        # 3. Sparse reconstruction
        sparse_0 = self.mapper()

        # 4. Convert to text and PLY
        self.model_converter(sparse_dir=sparse_0, output_format="BOTH")

        outputs: dict[str, Path] = {
            "database": self.database_path,
            "sparse_bin": self.sparse_dir / "0",
            "sparse_txt": self.sparse_dir.parent / "sparse_txt",
            "sparse_ply": self.sparse_dir.parent / "sparse_ply",
        }

        # 5. Optional dense MVS
        if dense:
            self.dense_pipeline()
            outputs["dense_dir"] = self.dense_dir
            outputs["fused_ply"] = self.dense_dir / "fused.ply"

        return outputs