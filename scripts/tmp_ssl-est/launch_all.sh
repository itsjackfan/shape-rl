#!/bin/bash
# After cf-data/full/index.parquet exists: launch B0-B3, C0-C2 on CPU (one job per ablation), the dataset report and the
# confound probes. C3 follows once C0-C2 val metrics exist (its loss is the best C by val spearman_A).
set -euo pipefail
cd /n/home03/kzheng/shape-rl-ssl-est
export UV_PROJECT_ENVIRONMENT=$SCRATCHDIR/venvs/shape-rl-ssl-est UV_NO_CACHE=1
test -f $SCRATCHDIR/shape-rl/outputs/cf-data/full/index.parquet
i=0
for e in b0 b1 b2 b3 c0 c1 c2; do
  q=$([ $i -lt 4 ] && echo test:wattenberg_lab:none || echo sapphire:wattenberg_lab:none)
  uv run --no-sync --env-file .env python cluster/launch.py ssl ablation=$e estimator=$e ssl.data=cf-data/full wandb.mode=offline \
    --gpus 0 --queues $q --time 0-02:00:00 2>&1 | tail -1 | sed "s/^/$e: /"
  i=$((i + 1))
done
S=$SCRATCHDIR/shape-rl/outputs/ssl-est/slurm
sbatch --parsable -o $S/dataset-report-%j.out ../shape-rl-cf-data/scripts/tmp_cf-data/base_env.sbatch \
  ../shape-rl-cf-data/scripts/tmp_cf-data/dataset_report.py $SCRATCHDIR/shape-rl/outputs/cf-data/full | sed "s/^/dataset_report: /"
sbatch --parsable -p sapphire -c 16 --mem=128G -t 0-00:40 -o $S/probes-%j.out ../shape-rl-cf-data/scripts/tmp_cf-data/analysis.sbatch \
  ../shape-rl-cf-data/scripts/tmp_cf-data/probes.py $SCRATCHDIR/shape-rl/outputs/cf-data/full | sed "s/^/probes: /"
