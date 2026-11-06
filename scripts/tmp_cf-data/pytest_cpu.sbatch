#!/bin/bash
#SBATCH -J cf-data-pytest-cpu
#SBATCH -p test
#SBATCH --account=wattenberg_lab
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-00:30
cd "$SLURM_SUBMIT_DIR"
export UV_NO_CACHE=1
uv run --no-sync --env-file .env pytest -q -p no:cacheprovider -rA "$@" 2>&1 | tail -40
