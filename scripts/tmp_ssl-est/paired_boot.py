"""Paired bootstrap over test prompts (B resamples, the same for every estimator) on seed-averaged predictions:
95 % interval of each metric and of each estimator's difference from a reference (default b3, the learned critic).
Resampled prompts get unique ids, so a prompt drawn twice counts as two prompts.
Usage: python paired_boot.py <output_root> [ref] [B]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shape_rl import ssl

root, ref, B = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "b3"), int(sys.argv[3]) if len(sys.argv) > 3 else 200
idx = pd.read_parquet(root / "cf-data/full/index.parquet")
A = len(idx.q_out.iloc[0]) // len(idx.v_out.iloc[0])
o = np.concatenate([np.stack(idx.v_out.values)[:, None], np.stack(idx.q_out.values).reshape(len(idx), A, -1)], 1)
succ, fin = (o == 1).sum(-1), (o >= 0).sum(-1)
trunc = (o == -1).mean(-1)
data = {"idx": idx, "rate": ssl.rate(succ, fin), "fin": fin, "trunc_q": trunc[:, 1:],
        "conf": np.c_[idx.t.values, idx.trace_len.values, idx.t.values / idx.trace_len.values, trunc[:, 0]]}
test_rows = np.nonzero((idx.split == "test").values)[0]
by_prompt = pd.Series(test_rows).groupby(idx.prompt_id.values[test_rows]).apply(list)
ests = sorted(p.name for p in (root / "ssl-est").iterdir() if p.name in {"b0", "b1", "b2", "b3", "c0", "c1", "c2", "c3"} and (p / "metrics.json").exists())
pred = {e: np.mean([np.load(f) for f in sorted((root / "ssl-est" / e).glob("pred_seed*.npy"))], 0) for e in ests}
keys = ["auroc", "succ_pcorr", "succ_pcorr_resid", "spearman_A", "spearman_A_resid", "sign_acc", "ece"]


def sub(rows, copy_ids):
    d = {k: v[rows] for k, v in data.items() if k != "idx"}
    d["idx"] = data["idx"].iloc[rows].reset_index(drop=True).assign(prompt_id=copy_ids)
    return d


rng = np.random.default_rng(0)
boot = {e: [] for e in ests}
for _ in range(B):
    pick = rng.integers(len(by_prompt), size=len(by_prompt))
    rows = np.concatenate([by_prompt.iloc[i] for i in pick])
    ids = np.concatenate([[f"{i}-{c}"] * len(by_prompt.iloc[i]) for c, i in enumerate(pick)])
    d = sub(rows, ids)
    for e in ests:
        m = ssl.metrics(pred[e][rows], d, np.ones(len(rows), bool))
        boot[e].append([m[k] for k in keys])
boot = {e: np.array(v) for e, v in boot.items()}
pct = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
out = {"B": B, "n_test_prompts": len(by_prompt), "ref": ref, "metric_ci95": {}, f"diff_vs_{ref}_ci95": {}}
for e in ests:
    out["metric_ci95"][e] = {k: pct(boot[e][:, i]) for i, k in enumerate(keys)}
    if ref in boot and e != ref:
        out[f"diff_vs_{ref}_ci95"][e] = {k: pct(boot[e][:, i] - boot[ref][:, i]) for i, k in enumerate(keys)}
print(json.dumps(out, indent=1))
(root / "ssl-est" / "paired_boot.json").write_text(json.dumps(out, indent=1))
