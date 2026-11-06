#!/bin/bash
# Every number for the final report: results tables, ceiling, H1, dataset stats, probes, interpretability, GPU-hours.
O=$SCRATCHDIR/shape-rl/outputs
cd /n/home03/kzheng/shape-rl-ssl-est
module load python >/dev/null 2>&1; source activate $LABDIR/conda_envs/base >/dev/null 2>&1
echo "## results (test prompts)"; python -I scripts/tmp_ssl-est/results_table.py $O/ssl-est b0 b1 b2 b3 c0 c1 c2 c3
echo; echo "## by branch kind"; python -I scripts/tmp_ssl-est/results_table.py $O/ssl-est b0 b1 b2 b3 c0 c1 c2 c3 --kinds
echo; echo "## val spearman_A (C3 choice)"; for e in c0 c1 c2; do python -I -c "import json; m=json.load(open('$O/ssl-est/$e/metrics.json')); print('$e', round(m['val_mean']['spearman_A'], 4))"; done
echo; echo "## paired bootstrap over test prompts (vs b3)"; cat $O/ssl-est/paired_boot.json 2>/dev/null
echo; echo "## dataset + H1"; cat $O/cf-data/full/analysis/dataset_stats.json
echo; echo "## probes"; cat $O/cf-data/full/analysis/probes.json
echo; echo "## interpretability"; cat $O/cf-data/full/analysis/interp_*.json 2>/dev/null
echo; echo "## GPU-hours"; bash scripts/tmp_ssl-est/gpu_hours.sh
