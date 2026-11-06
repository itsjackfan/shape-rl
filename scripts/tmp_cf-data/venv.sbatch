#!/bin/bash
# Venv for one worktree from the warm uv cache: sbatch --chdir <worktree> venv.sbatch (UV_PROJECT_ENVIRONMENT exported).
#SBATCH -J shape-rl-venv
#SBATCH -p test
#SBATCH --account=wattenberg_lab
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-00:40
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "venv: $UV_PROJECT_ENVIRONMENT"; date
uv sync --extra cluster --no-install-package llguidance --no-install-package deepspeed 2>&1 | tail -3
S=$(ls -d $UV_PROJECT_ENVIRONMENT/lib/python3*/site-packages)
mkdir -p $S/llguidance && cp $SCRATCHDIR/venvs/shape-rl-cf-data/lib/python3.11/site-packages/llguidance/*.py $S/llguidance/
date
