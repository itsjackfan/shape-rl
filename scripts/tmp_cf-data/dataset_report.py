"""Dataset stats and the H1 check for one rollout dataset dir (runs with pandas/matplotlib/scipy only, e.g. conda base).
H1 check: |Q(s_t, a') - Q(s_t, a_t)| (MC success rates, finished continuations only) at high-entropy vs random branch points,
with a noise reference: |Q_A(a_t) - Q_B(a_t)| between two random halves of the same action's M continuations.
Usage: python dataset_report.py <dataset_dir> [index.parquet subset: all|train|val|test]"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = Path(sys.argv[1])
out = d / "analysis"
out.mkdir(exist_ok=True)
P = pd.concat([pd.read_parquet(f) for f in sorted((d / "rollout").glob("s*.prompts.parquet"))], ignore_index=True)
R = pd.read_parquet(d / "index.parquet") if (d / "index.parquet").exists() else pd.concat(
    [pd.read_parquet(f) for f in sorted((d / "rollout").glob("s[0-9][0-9][0-9]_c[0-9][0-9][0-9].parquet"))], ignore_index=True)
stats = {}
rew = np.concatenate(P.sample_rewards.values)
lens = np.concatenate([[len(s) for s in ss] for ss in P.sample_ids])
stats["stageA"] = {"prompts": len(P), "samples": int(len(rew)), "truncated": float(np.mean(rew == -1)),
                   "correct": float(np.mean(rew == 1)), "finished_wrong": float(np.mean(rew == 0)),
                   "len_q50_q90_q95_q99": np.percentile(lens, [50, 90, 95, 99]).tolist(), "tokens": int(lens.sum()),
                   "kept_prompts": int(((P.pass_rate > 0) & (P.pass_rate < 1)).sum()),
                   "by_source": {s: {"n": len(g), "mean_pass": float(g.pass_rate.mean()), "kept": int(((g.pass_rate > 0) & (g.pass_rate < 1)).sum()),
                                     "p0": int((g.pass_rate == 0).sum()), "p1": int((g.pass_rate == 1).sum()), "all_truncated": int(g.pass_rate.isna().sum())}
                                 for s, g in P.groupby("source")}}
M, A = len(R.v_out.iloc[0]), len(R.actions.iloc[0])
v = np.stack(R.v_out.values)
q = np.stack(R.q_out.values).reshape(len(R), A, M)
stats["branch"] = {"rows": len(R), "kinds": R.kind.value_counts().to_dict(), "traces": int(R.trace_id.nunique()),
                   "prompts": int(R.prompt_id.nunique()), "splits_prompts": R.groupby("split").prompt_id.nunique().to_dict(),
                   "continuations": int((1 + A) * M * len(R)), "truncated_continuations": float(np.mean(np.concatenate([v.ravel(), q.ravel()]) == -1)),
                   "continuation_tokens": int(R.gen_tokens.sum()), "trace_reward_counts": R.drop_duplicates("trace_id").reward.value_counts().to_dict()}


def rate(o):
    fin = o >= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(fin.sum(-1) > 0, (o == 1).sum(-1) / fin.sum(-1), np.nan)


Q = rate(q)
rng = np.random.default_rng(0)
perm = rng.permutation(M)
QA, QB = rate(q[..., perm[:M // 2]]), rate(q[..., perm[M // 2:]])
dq = np.abs(Q[:, 1:] - Q[:, :1])
h1 = {}
for kind in ("entropy", "random"):
    x = dq[(R.kind == kind).values].ravel()
    x = x[~np.isnan(x)]
    h1[kind] = {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "q75": float(np.quantile(x, 0.75)),
                "q90": float(np.quantile(x, 0.9)), "P(>0)": float(np.mean(x > 0)), "P(>=0.25)": float(np.mean(x >= 0.25)),
                "P(>=0.5)": float(np.mean(x >= 0.5)), "P(>=0.75)": float(np.mean(x >= 0.75))}
xe, xr = [dq[(R.kind == k).values].ravel() for k in ("entropy", "random")]
xe, xr = xe[~np.isnan(xe)], xr[~np.isnan(xr)]
h1["mannwhitney_entropy_vs_random"] = {"U": float(mannwhitneyu(xe, xr).statistic), "p": float(mannwhitneyu(xe, xr).pvalue)}
nz = np.abs(QA[:, 0] - QB[:, 0])
nz = nz[~np.isnan(nz)]
h1["noise_ref_halfM_same_action"] = {"n": int(len(nz)), "mean": float(nz.mean()), "P(>=0.25)": float(np.mean(nz >= 0.25)),
                                     "P(>=0.5)": float(np.mean(nz >= 0.5))}
h1["mean_entropy_by_kind"] = R.groupby("kind").entropy.mean().to_dict()
h1["mean_logprob_alt_by_kind"] = {k: float(np.mean(np.stack(g.action_logprobs.values)[:, 1:])) for k, g in R.groupby("kind")}
stats["h1"] = h1
print(json.dumps(stats, indent=1, default=float))
(out / "dataset_stats.json").write_text(json.dumps(stats, indent=1, default=float))

bins = np.arange(0, 1.0001 + 1 / 8, 1 / 8) - 1 / 16
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for x, kind, color in ((xe, "high-entropy", "#d95f02"), (xr, "random", "#1b9e77")):
    ax.hist(x, bins=bins, weights=np.full(len(x), 1 / len(x)), histtype="step", lw=2, color=color, label=f"{kind} (n={len(x)})")
ax.hist(nz, bins=bins, weights=np.full(len(nz), 1 / len(nz)), histtype="step", lw=1.5, ls="--", color="gray",
        label=f"noise ref: same action, M/2 vs M/2 (n={len(nz)})")
ax.set_xlabel(f"|Q(s_t, a') - Q(s_t, a_t)|  (MC, M = {M})")
ax.set_ylabel("fraction of pairs")
ax.set_yscale("log")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(out / "h1_dq_hist.png", dpi=150)
print("plot:", out / "h1_dq_hist.png")
