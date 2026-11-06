#!/bin/bash
#SBATCH -J cf-data-download
#SBATCH -p test
#SBATCH --account=wattenberg_lab
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-01:00
set -euo pipefail
module load python
source activate $LABDIR/conda_envs/base
date; hostname
python - <<'PY'
import time
from huggingface_hub import snapshot_download
for repo, kind in [("Qwen/Qwen3.5-4B", "model"), ("trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6", "model"),
                   ("DigitalLearningGmbH/MATH-lighteval", "dataset"), ("openai/gsm8k", "dataset")]:
    t0 = time.time()
    p = snapshot_download(repo, repo_type=kind, max_workers=16)
    print(repo, p, f"{time.time() - t0:.0f}s", flush=True)
PY
date
