"""Interpretability (seed 0): k-means (64) on train states, cluster purity vs outcome and variance explained on test states,
for z = g(h) of one contrastive loss and for raw h (PCA to 128 dims); 2-D PCA coordinates of test h(s_t) states for a plot.
Purity = sum over clusters of the majority count (successes or failures of the continuations) / all finished continuations.
Usage: python interp.py <exp>/<ablation> <estimator, e.g. c2> -> <dataset>/analysis/interp_<est>.{json,parquet}"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir

from shape_rl import ssl

with initialize_config_dir(config_dir=str(ssl.paths.ROOT / "configs"), version_base=None):
    cfg = compose("ssl", overrides=[f"estimator={sys.argv[2]}", f"ssl.data={sys.argv[1]}", "ssl.seeds=[0]"])
data = ssl.prepare(cfg)
X, tr, y, w = data["X"], data["tr"], data["y"], data["w"]
te = torch.as_tensor((data["idx"].split == "test").values, device=ssl.DEV)[:, None] & (data["w"] > 0)
g = ssl.fit_projection(X, data, cfg.estimator, 0)
with torch.no_grad():
    Z = torch.cat([g(x) for x in X.flatten(0, 1).split(65536)]).view(*X.shape[:2], -1)
U, S, V = torch.pca_lowrank(X[tr], q=128, center=True)
H = (X - X[tr].mean(0)) @ V


def cluster_stats(F, name):
    c = ssl.kmeans(F[tr], 64, 50, 0)
    out = {}
    for split, m in (("train", tr), ("test", te)):
        a = torch.cdist(F[m], c).argmin(1)
        succ, fin = (y[m] * w[m]), w[m]
        s_c = torch.zeros(64, device=ssl.DEV).index_add_(0, a, succ)
        n_c = torch.zeros(64, device=ssl.DEV).index_add_(0, a, fin)
        purity = (torch.maximum(s_c, n_c - s_c).sum() / n_c.sum()).item()
        base = max(succ.sum().item(), (fin - succ).sum().item()) / fin.sum().item()
        mean_c = s_c / n_c.clamp(min=1)
        r2 = 1 - ((w[m] * (y[m] - mean_c[a]) ** 2).sum() / (w[m] * (y[m] - (succ.sum() / fin.sum())) ** 2).sum()).item()
        out[split] = {"purity": purity, "majority_baseline": base, "var_explained": r2, "clusters_used": int((n_c > 0).sum())}
    return {name: out}


res = {**cluster_stats(Z, f"z_{cfg.estimator.loss}"), **cluster_stats(H, "h_pca128")}
print(json.dumps(res, indent=1))
d = Path(cfg.output_root) / cfg.ssl.data / "analysis"
d.mkdir(exist_ok=True)
(d / f"interp_{sys.argv[2]}.json").write_text(json.dumps(res, indent=1))
rows = torch.as_tensor((data["idx"].split == "test").values, device=ssl.DEV)
for F, name in ((Z[:, 0], "z"), (X[:, 0], "h")):
    F = F[rows]
    _, _, V2 = torch.pca_lowrank(F, q=2, center=True)
    P = ((F - F.mean(0)) @ V2).cpu().numpy()
    pd.DataFrame({"x": P[:, 0], "y": P[:, 1], "V_mc": data["rate"][rows.cpu().numpy(), 0], "reward": data["idx"].reward.values[rows.cpu().numpy()],
                  "kind": data["idx"].kind.values[rows.cpu().numpy()]}).to_parquet(d / f"interp_{sys.argv[2]}_{name}_pca2.parquet")
print("wrote", d)
