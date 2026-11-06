#!/bin/bash
# One job. Submitted only by cluster/launch.py (partition/account/GPU type chosen there): run.sbatch <entry> <hydra overrides>
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=160G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 gcc/14.2.0-fasrc01
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export UV_NO_CACHE=1  # uv run --no-sync needs no cache; a stale NFS flock on the shared uv cache can block it
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=/scratch/$SLURM_JOB_ID  # no FlashInfer JIT into ~/.cache
export TMPDIR=/scratch/$SLURM_JOB_ID/tmp TRITON_CACHE_DIR=/scratch/$SLURM_JOB_ID/triton && mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
N=${SLURM_GPUS_ON_NODE:-0}
entry=$1; shift
# --no-sync: concurrent jobs must not rewrite the shared .venv (run cluster/setup.sh after dependency changes)
if [ "$entry" = train ]; then
  uv run --no-sync --env-file .env accelerate launch --config_file cluster/zero3.yaml --num_processes "$N" \
    -m shape_rl.train train.use_vllm=true "$@"
elif [ "$entry" = eval ]; then
  uv run --no-sync --env-file .env python -m shape_rl.evaluate eval.backend=vllm \
    "eval.model_args='dtype=bfloat16,tensor_parallel_size=$N,max_model_len=32768,gpu_memory_utilization=0.85'" "$@"
else  # rollout | encode | ssl: one GPU (ssl may run with --gpus 0), one shard
  uv run --no-sync --env-file .env python -m "shape_rl.$entry" "$@"
fi
