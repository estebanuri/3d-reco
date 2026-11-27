# COLMAP Automatizer

Production-ready Python pipeline to automate COLMAP execution on a shared
university workstation (NVIDIA RTX 4090) via Docker, with a local binary mode
for lightweight testing on a developer machine.

## ⚠️ Critical Constraints

- **No `pycolmap`**: COLMAP is always invoked via CLI commands (subprocess).
- **No heavy I/O in `/home`**: Never store datasets, images, `database.db`,
  or sparse/dense reconstructions inside `/home` or relative to the repo.
- **Mandatory SCRATCH storage**: All outputs live inside a fast SCRATCH directory.
- **Docker (server) vs. Local binary (dev)**: Docker with `--gpus all` and `--user`
  on server; local binary on dev machine.
- **Headless mode**: `QT_QPA_PLATFORM=offscreen` in both modes.

## Project Structure

```
colmap-automatizer/
├── main.py                      # CLI entry point
├── .env.example                 # Template for environment variables
├── .env.local                   # Gitignored: Mac/PC dev settings
├── .env.remote                  # Gitignored: University server settings
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore
├── pyproject.toml
└── src/
    ├── __init__.py
    ├── auto/                    # Generic COLMAP automation engine
    │   ├── __init__.py
    │   ├── config.py            # AppConfig (Docker vs Local, GPU, Scratch paths)
    │   ├── validator.py         # Preflight checks (/home guard, Docker/binary, nvidia-smi, disk)
    │   ├── runner.py            # ColmapRunner: Docker vs local command building, real-time logging
    │   └── pipeline.py          # SfM pipeline: feature_extractor -> matcher -> mapper -> model_converter -> dense MVS
    └── prj/                     # Project-specific dataset fetchers & configurations
        ├── __init__.py
        ├── base.py              # Abstract BaseProject interface
        └── odm_bellus.py        # ODM Bellus dataset project implementation
```

## Configuration

### `.env` Strategy

| File | Purpose | Gitignored |
|------|---------|------------|
| `.env.example` | Template (committed) | No |
| `.env.local` | Mac/PC local dev | Yes |
| `.env.remote` | University server | Yes |

Use `--env local` or `--env remote` to select the profile. CLI flags override env vars.

### Key Settings

| Variable | Description | Example (local) | Example (remote) |
|----------|-------------|-----------------|------------------|
| `COLMAP_MODE` | `docker` or `local` | `local` | `docker` |
| `DOCKER_IMAGE` | COLMAP Docker image | `colmap/colmap:latest` | `colmap/colmap:latest` |
| `COLMAP_BIN` | Path to COLMAP binary | `/usr/local/bin/colmap` | `/usr/local/bin/colmap` |
| `SCRATCH_DIR` | Scratch root (must be outside `/home` on server) | `./scratch` | `/scratch/$USER/colmap_auto` |
| `USE_GPU` | Enable GPU acceleration | `false` | `true` |
| `LOCAL_DEV` | Allow `/home` scratch (dev only) | `true` | `false` |

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and adjust
cp .env.example .env.local   # for Mac/PC dev
cp .env.example .env.remote  # for university server
```

## Usage

### Local Development (Mac/PC)

```bash
# Use the local COLMAP binary with CPU-only mode
python main.py --env local --no-use-gpu --scratch-dir ./scratch

# With GPU (if COLMAP built with CUDA)
python main.py --env local --use-gpu --scratch-dir ./scratch

# Sequential matching (faster for small datasets)
python main.py --env local --matcher sequential --no-use-gpu

# Enable dense MVS
python main.py --env local --dense --no-use-gpu
```

### University Server (RTX 4090, Docker)

```bash
source .venv/bin/activate
python main.py --env remote --use-gpu
```

This will:
1. Validate that `SCRATCH_DIR` is outside `/home`
2. Check Docker daemon availability

## Docker Volume Mounting & Path Consistency

When `COLMAP_MODE=docker`, the `ColmapRunner` builds a Docker command with:

```
docker run --rm --gpus all --user <uid>:<gid> \
  -e QT_QPA_PLATFORM=offscreen \
  -v <scratch_dir>:<scratch_dir> \
  colmap/colmap:latest colmap <subcommand> ...
```

Key design decisions:

1. **Single source of truth**: `AppConfig.scratch_dir` is the only directory COLMAP
   reads from or writes to. All pipeline paths (dataset, workspace, logs, database,
   sparse/dense outputs) are derived as subpaths of `scratch_dir`.

2. **1:1 path mapping**: The scratch directory is mounted at the **same absolute path**
   inside the container (`-v <scratch>:<scratch>`). Because the path string is identical
   on both sides, no path translation is needed. COLMAP commands reference only paths
   under `scratch_dir`, so they work identically inside and outside the container.

3. **`--user $(id -u):$(id -g)`**: Prevents Docker from creating root-owned files in
   the scratch directory, preserving host file ownership.

4. **`QT_QPA_PLATFORM=offscreen`**: Injected via `-e` in Docker and via `env` in local
   subprocess, ensuring headless execution on both server and dev machines.

5. **Local mode**: No volume mount needed since host and container are the same filesystem.

## `/home` Guard

When `LOCAL_DEV=false` (remote mode), `AppConfig` and `validator.py` abort immediately
if `SCRATCH_DIR` resolves inside `/home` or if `SCRATCH_DIR` is missing. This protects
the university storage quotas. Set `LOCAL_DEV=true` only for local development.

## Outputs

Each run creates a timestamped workspace under `<SCRATCH_DIR>/workspaces/<project>_<timestamp>/`:

```
<workspace>/
├── database.db              # COLMAP SQLite database
├── sparse/
│   └── 0/                   # Binary sparse reconstruction (cameras.bin, images.bin, points3D.bin)
├── sparse_txt/              # Text-format sparse reconstruction
├── sparse_ply/              # PLY-format sparse reconstruction (for Gaussian Splatting, etc.)
├── dense/                   # Dense MVS outputs (only with --dense)
│   ├── stereo/
│   └── fused.ply
└── logs/                    # Per-step and pipeline log files
```

## License

MIT
3. Check `nvidia-smi` for GPU
4. Clone the ODM Bellus dataset into `<SCRATCH_DIR>/datasets/odm_data_bellus`
5. Run feature_extractor → exhaustive_matcher → mapper → model_converter
6. Output sparse reconstruction (`.bin`, `.txt`, `.ply`) into `<SCRATCH_DIR>/workspaces/odm_bellus_<timestamp>/`

### CLI Options

| Option | Description |
|--------|-------------|
| `--env` | Environment profile: `local` or `remote` (default: `local`) |
| `--project` | Project to process (default: `odm_bellus`) |
| `--matcher` | `exhaustive` or `sequential` (default: `exhaustive`) |
| `--dense` | Enable dense MVS reconstruction |
| `--force-download` | Re-clone dataset even if already present |
| `--use-gpu` / `--no-use-gpu` | Toggle GPU acceleration |
| `--local-dev` | Allow scratch_dir inside `/home` (dev only) |
| `--scratch-dir` | Override `SCRATCH_DIR` from `.env` |
| `--skip-validation` | Skip preflight checks (not recommended) |
