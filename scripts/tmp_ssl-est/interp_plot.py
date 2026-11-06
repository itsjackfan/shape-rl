"""2-D PCA of test h(s_t) states (raw h and z), colored by the MC value V(s_t). Usage: python interp_plot.py <analysis dir> <est>"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d, est = Path(sys.argv[1]), sys.argv[2]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, name in zip(axes, ("h", "z")):
    p = pd.read_parquet(d / f"interp_{est}_{name}_pca2.parquet").dropna(subset=["V_mc"])
    sc = ax.scatter(p.x, p.y, c=p.V_mc, cmap="viridis", s=4, alpha=0.6, vmin=0, vmax=1, linewidths=0)
    ax.set_title(f"{'raw h(s_t)' if name == 'h' else f'z = g(h), {est}'} (test, n={len(p)})", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
fig.colorbar(sc, ax=axes, label="MC value V(s_t)", shrink=0.8)
fig.savefig(d / f"interp_{est}_pca2.png", dpi=150, bbox_inches="tight")
print(d / f"interp_{est}_pca2.png")
