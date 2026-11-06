#!/bin/bash
# Light CPU analysis: sbatch --chdir <worktree> analysis.sbatch <script.py> [args]
#SBATCH -J shape-rl-analysis
#SBATCH -p test
#SBATCH --account=wattenberg_lab
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-00:20
cd "$SLURM_SUBMIT_DIR"
export UV_NO_CACHE=1
uv run --no-sync --env-file .env python "$@"
