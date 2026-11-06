#!/bin/bash
#SBATCH -J cf-data-pytest
#SBATCH -p kempner_requeue
#SBATCH --account=kempner_krajan_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 0-00:40
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 gcc/14.2.0-fasrc01
export TMPDIR=/scratch/$SLURM_JOB_ID/tmp TRITON_CACHE_DIR=/scratch/$SLURM_JOB_ID/triton && mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
export UV_NO_CACHE=1 VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=/scratch/$SLURM_JOB_ID
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --no-sync --env-file .env pytest -q -p no:cacheprovider -rA "$@" 2>&1 | tail -60
