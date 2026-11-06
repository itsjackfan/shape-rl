"""Confound probes: ridge probes from frozen h(s_t) (standardized, train split) to position t, trace length, relative position
t / T and the truncated fraction of the V continuations; ridge strength chosen on val, R^2 on test.
Usage: python scripts/tmp_cf-data/probes.py <dataset_dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from shape_rl.ssl import load

d = Path(sys.argv[1])
idx, h_NSD, succ, fin, o = load(d)
dev = "cuda" if torch.cuda.is_available() else "cpu"
X = h_NSD[:, 0].float().to(dev)
split = idx.split.values
tr, va, te = (torch.as_tensor(split == s) for s in ("train", "val", "test"))
X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-4)
X = torch.cat([X, torch.ones(len(X), 1, device=dev)], 1)
targets = {"t": idx.t.values, "trace_len": idx.trace_len.values, "rel_pos": idx.t.values / idx.trace_len.values,
           "trunc_frac_v": (o[:, 0] == -1).mean(-1), "trace_reward": idx.reward.values}
r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
res = {}
for name, y in targets.items():
    y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    if y[tr].std() == 0:
        res[name] = {"note": "constant on train"}
        continue
    G, b = X[tr].T @ X[tr], X[tr].T @ y[tr]
    best = max((r2(y[va], X[va] @ torch.linalg.solve(G + lam * torch.eye(len(G), device=dev), b)), lam) for lam in 10.0 ** np.arange(-1, 6))
    w = torch.linalg.solve(G + best[1] * torch.eye(len(G), device=dev), b)
    res[name] = {"test_r2": r2(y[te], X[te] @ w), "val_r2": best[0], "lambda": best[1]}
print(json.dumps(res, indent=2))
(d / "analysis").mkdir(exist_ok=True)
(d / "analysis" / "probes.json").write_text(json.dumps(res, indent=2))
