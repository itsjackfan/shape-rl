#!/bin/bash
#SBATCH -J cf-data-verify
#SBATCH -p test
#SBATCH --account=wattenberg_lab
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-00:30
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TMPDIR=/scratch/$SLURM_JOB_ID/tmp; mkdir -p $TMPDIR
echo "venv: $UV_PROJECT_ENVIRONMENT"; date
export UV_NO_CACHE=1
uv run --no-sync --env-file .env python scripts/tmp_cf-data/verify_step1.py; date
uv run --no-sync --env-file .env pytest -m "not slow" -q -p no:cacheprovider 2>&1 | tail -25; date
