"""Smoke report for one rollout dataset dir: truncation vs cap, pass rates, label noise, H1 preview, 20 graded outputs.
Usage: python scripts/tmp_cf-data/smoke_report.py <dataset_dir> [slurm_log]"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from shape_rl.rewards import last_boxed

d = Path(sys.argv[1])
out = d / "analysis"
out.mkdir(exist_ok=True)
P = pd.concat([pd.read_parquet(f) for f in sorted((d / "rollout").glob("s*.prompts.parquet"))], ignore_index=True)
chunks = sorted((d / "rollout").glob("s[0-9][0-9][0-9]_c[0-9][0-9][0-9].parquet"))
R = pd.concat([pd.read_parquet(f) for f in chunks], ignore_index=True) if chunks else pd.DataFrame()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

rew = np.concatenate(P.sample_rewards.values)
lens = np.concatenate([[len(s) for s in ss] for ss in P.sample_ids])
print(f"== stage A: {len(P)} prompts x {len(P.sample_rewards[0])} samples; truncated {np.mean(rew == -1):.3f}; "
      f"correct {np.mean(rew == 1):.3f}; finished-wrong {np.mean(rew == 0):.3f}")
print("length quantiles (tokens) 50/90/95/99/max:", np.percentile(lens, [50, 90, 95, 99, 100]).round())
for cap in (1024, 1536, 2048, 3072, 4096):
    print(f"  truncation rate at cap {cap}: {np.mean((lens > cap) | (rew == -1)):.3f}")
for src, g in P.groupby("source"):
    pr = g.pass_rate
    print(f"  {src}: n={len(g)} mean pass {pr.mean():.3f}; kept (0<p<1) {((pr > 0) & (pr < 1)).mean():.3f}; "
          f"p=0 {(pr == 0).mean():.3f}; p=1 {(pr == 1).mean():.3f}")

# 20 graded outputs for the manual check: 10 graded correct, 10 graded wrong or truncated
rng = np.random.default_rng(0)
flat = [(i, k) for i in range(len(P)) for k in range(len(P.sample_ids[i]))]
pick = [flat[j] for j in rng.permutation(len(flat))]
good = [x for x in pick if P.sample_rewards[x[0]][x[1]] == 1][:10]
bad = [x for x in pick if P.sample_rewards[x[0]][x[1]] != 1][:10]
with open(out / "manual_check.txt", "w") as f:
    for i, k in good + bad:
        text = tok.decode(P.sample_ids[i][k], skip_special_tokens=True)
        f.write(f"#### prompt {P.prompt_id[i]} ({P.source[i]}) sample {k}: grade {P.sample_rewards[i][k]} | gold {P.answer[i]!r} | "
                f"box {last_boxed(text)!r} | len {len(P.sample_ids[i][k])} | last token {tok.decode(P.sample_ids[i][k][-1:])!r}\n"
                f"...{text[-600:]}\n\n")
print("manual check file:", out / "manual_check.txt")
if not len(R):
    sys.exit()

M = len(R.v_out[0])
A = len(R.actions[0])
v = np.stack(R.v_out.values)  # [N, M]
q = np.stack(R.q_out.values).reshape(len(R), A, M)  # [N, A, M]


def rate(o):
    fin = o >= 0
    with np.errstate(invalid="ignore"):
        return np.where(fin.sum(-1) > 0, (o == 1).sum(-1) / fin.sum(-1), np.nan)


print(f"== branch rows {len(R)} ({R.kind.value_counts().to_dict()}), traces {R.trace_id.nunique()}, prompts {R.prompt_id.nunique()}; "
      f"continuations {(1 + A) * M * len(R)}; truncated continuations {np.mean(np.concatenate([v.ravel(), q.ravel()]) == -1):.3f}")
print(f"   gen tokens {R.gen_tokens.sum()} ({R.gen_tokens.sum() / R.prompt_id.nunique():.0f} per kept prompt)")
V, Q = rate(v), rate(q)
h = M // 2
VA, VB, QA, QB = rate(v[:, :h]), rate(v[:, h:]), rate(q[..., :h]), rate(q[..., h:])


def rel(a, b, name):
    m = ~np.isnan(a) & ~np.isnan(b)
    rs, rp = spearmanr(a[m], b[m])[0], pearsonr(a[m], b[m])[0]
    print(f"   split-half {name}: n={m.sum()} spearman {rs:.3f} (Spearman-Brown {2 * rs / (1 + rs):.3f}); pearson {rp:.3f} "
          f"(SB {2 * rp / (1 + rp):.3f})")


print("== label noise (halves of M =", M, ")")
rel(QA.ravel(), QB.ravel(), "Q(s,a), all actions")
rel(VA, VB, "V(s)")
rel((QA - VA[:, None]).ravel(), (QB - VB[:, None]).ravel(), "A = Q - V")
rel((QA[:, 1:] - QA[:, :1]).ravel(), (QB[:, 1:] - QB[:, :1]).ravel(), "dQ = Q(a') - Q(a_t)")

print("== H1 preview: |Q(a') - Q(a_t)| by branch kind (full M); noise ref = split-half |Q_A(a_t) - Q_B(a_t)|")
dq = np.abs(Q[:, 1:] - Q[:, :1])
for kind in ("entropy", "random"):
    m = (R.kind == kind).values
    x = dq[m].ravel()
    x = x[~np.isnan(x)]
    print(f"   {kind}: n={len(x)} mean {x.mean():.3f} median {np.median(x):.3f} P(>0.25) {np.mean(x > 0.25):.3f} P(>0.5) {np.mean(x > 0.5):.3f}")
x = np.abs(QA[:, 0] - QB[:, 0])
x = x[~np.isnan(x)]
print(f"   noise ref (M/2 vs M/2, same action): mean {x.mean():.3f} P(>0.25) {np.mean(x > 0.25):.3f} P(>0.5) {np.mean(x > 0.5):.3f}")
print(f"   V(s_t) mean {np.nanmean(V):.3f}; Q(a_t) mean {np.nanmean(Q[:, 0]):.3f}; Q(a') mean {np.nanmean(Q[:, 1:]):.3f}")
print(f"   logprob a' (mean) {np.mean(np.stack(R.action_logprobs.values)[:, 1:]):.2f}; entropy by kind:",
      R.groupby("kind").entropy.mean().round(3).to_dict())

if len(sys.argv) > 2:
    log = Path(sys.argv[2]).read_text()
    ch = [(int(a), int(b), float(c)) for a, b, c in re.findall(r"(\d+) branch points, (\d+) tokens, (\d+)s", log)]
    st = re.findall(r"stage A (\d+)s", log)
    if ch:
        n, tk, s = map(sum, zip(*ch))
        print(f"== throughput: stage A {st} s; {len(ch)} chunks, {n} branch points, {tk} tokens in {s:.0f}s -> {tk / s:.0f} tok/s, "
              f"{s / n:.2f} GPU-s per branch point, {s / R.prompt_id.nunique():.0f} GPU-s per kept prompt")

# H1 plot: |Q(a') - Q(a_t)| at high-entropy vs random branch points, with the split-half same-action noise reference
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bins = np.linspace(0, 1, 9 + 1) - 1 / 16
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for kind, color in (("entropy", "#d95f02"), ("random", "#1b9e77")):
    x = dq[(R.kind == kind).values].ravel()
    x = x[~np.isnan(x)]
    ax.hist(x, bins=bins, density=True, histtype="step", lw=2, color=color, label=f"{kind} (n={len(x)})")
ax.hist(np.abs(QA[:, 0] - QB[:, 0])[~np.isnan(QA[:, 0] - QB[:, 0])], bins=bins, density=True, histtype="step", lw=1.5, ls="--",
        color="gray", label=f"noise: |Q_A(a_t) - Q_B(a_t)|, M/2 each")
ax.set_xlabel("|Q(s_t, a') - Q(s_t, a_t)| (MC, M = %d)" % M)
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "h1_dq_hist.png", dpi=150)
print("plot:", out / "h1_dq_hist.png")
