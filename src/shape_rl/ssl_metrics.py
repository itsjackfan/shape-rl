"""Held-out metrics of the offline estimators (shape_rl.ssl) and the noise ceilings of the Monte Carlo (MC) labels.

p_NS = predicted P(success) per branch row and slot (slot 0 = V(s_t), slot 1 + j = Q(s_t, a_j)); A = Q - V.
Shapes: N = branch rows, S = 1 + A slots, M = continuations per (state, action).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

NAN = float("nan")


def rate(succ, fin):
    """Success rate over finished continuations; NaN where none finished."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(fin > 0, succ / fin, np.nan)


def ece(p, y, w, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    gap = np.abs(np.bincount(b, w * p, bins) - np.bincount(b, w * y, bins))
    return float(gap.sum() / np.bincount(b, w, bins).sum())


def residual(x, Z):
    Z1 = np.c_[np.ones(len(Z)), (Z - Z.mean(0)) / (Z.std(0) + 1e-9)]
    return x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]


def partial_corr(x, y, Z, groups):
    """Correlation of x and y after removing per-group means (prompt fixed effects) and then a linear fit on Z from both."""
    if len(x) < 3:
        return NAN
    code = pd.factorize(groups)[0]
    D = np.c_[x, y, Z].astype(float)
    mean = np.zeros((code.max() + 1, D.shape[1]))
    np.add.at(mean, code, D)
    d = D - (mean / np.bincount(code)[:, None])[code]
    r = d[:, :2] - d[:, 2:] @ np.linalg.lstsq(d[:, 2:], d[:, :2], rcond=None)[0] if Z.shape[1] else d[:, :2]
    return float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]) if r[:, 0].std() > 1e-12 else NAN


def auroc_within(score, label, prompt):
    a = [roc_auc_score(label[m], score[m]) for p in np.unique(prompt) if len(np.unique(label[m := prompt == p])) == 2]
    return float(np.mean(a))


def spearman(pred, ref):
    """Spearman correlation; NaN for a constant prediction (B0 predicts the same value for every slot)."""
    return float(spearmanr(pred, ref)[0]) if pred.std() > 0 else NAN


def metrics(p_NS, data, test):
    """AUROC (trace success from V_pred at h(s_t), within prompt); succ_pcorr: within-prompt correlation of V_pred with
    trace success, '_resid' with position, trace length, relative position and truncation regressed out of both;
    Spearman(A_pred, A_MC) ('_resid': ranks of both residualized on the confounds); sign accuracy of Q(a') - Q(a_t) (pairs
    share position and trace, so they need no confound control); ECE."""
    idx, R, fin, conf = data["idx"][test], data["rate"][test], data["fin"][test], data["conf"][test]
    p = p_NS[test]
    y, g = idx.reward.values.astype(float), idx.prompt_id.values
    out = {"auroc": auroc_within(p[:, 0], idx.reward.values, g),
           "succ_pcorr": partial_corr(p[:, 0], y, np.zeros((len(y), 0)), g),  # within prompt
           "succ_pcorr_resid": partial_corr(p[:, 0], y, (conf - conf.mean(0)) / (conf.std(0) + 1e-9), g)}  # and confounds out
    A_mc, A_pr = (R[:, 1:] - R[:, :1]), (p[:, 1:] - p[:, :1])
    m = ~np.isnan(A_mc)
    varied = A_pr[m].std() > 0
    out["spearman_A"] = spearman(A_pr[m], A_mc[m])
    out["pearson_A"] = float(np.corrcoef(A_pr[m], A_mc[m])[0, 1]) if varied else NAN
    Zp = np.concatenate([np.repeat(conf[:, None], A_mc.shape[1], 1), data["trunc_q"][test][..., None]], -1)[m]
    ra, rm = residual(pd.Series(A_pr[m]).rank().values, Zp), residual(pd.Series(A_mc[m]).rank().values, Zp)
    out["spearman_A_resid"] = float(np.corrcoef(ra, rm)[0, 1]) if varied else NAN
    dq_mc, dq_pr = R[:, 2:] - R[:, 1:2], p[:, 2:] - p[:, 1:2]
    m = ~np.isnan(dq_mc) & (dq_mc != 0)
    hit = np.where(dq_pr == 0, 0.5, np.sign(dq_pr) == np.sign(dq_mc))
    out["sign_acc"], out["n_sign_pairs"] = float(np.mean(hit[m])), int(m.sum())
    for kind in ("entropy", "random"):  # by branch-point kind (H1 x H2)
        k = (idx.kind.values == kind)[:, None]
        out[f"sign_acc_{kind}"] = float(np.mean(hit[m & k]))
        mk = ~np.isnan(A_mc) & k
        out[f"spearman_A_{kind}"] = spearman(A_pr[mk], A_mc[mk])
    m = fin > 0
    out["ece"] = ece(np.clip(p[m], 0, 1), R[m], fin[m])
    out["n_test_prompts"] = int(idx.prompt_id.nunique())
    return out


def split_half_reliability(o_NSM, n_splits, rng):
    """Split-half reliability of the MC labels, Spearman-Brown corrected to M: Spearman of A, Q, dQ and Pearson of A, plus
    the sign agreement of dQ between the halves (not corrected)."""
    M = o_NSM.shape[-1]
    r = {"A": [], "Q": [], "dQ": [], "agree": [], "A_pearson": []}
    for _ in range(n_splits):
        perm = rng.permutation(M)
        h1, h2 = o_NSM[..., perm[:M // 2]], o_NSM[..., perm[M // 2:]]
        R1, R2 = rate((h1 == 1).sum(-1), (h1 >= 0).sum(-1)), rate((h2 == 1).sum(-1), (h2 >= 0).sum(-1))
        for k, f in (("A", lambda R: R[:, 1:] - R[:, :1]), ("Q", lambda R: R[:, 1:]), ("dQ", lambda R: R[:, 2:] - R[:, 1:2])):
            a, b = f(R1).ravel(), f(R2).ravel()
            m = ~np.isnan(a) & ~np.isnan(b)
            r[k].append(spearmanr(a[m], b[m])[0])
            if k == "A":
                r["A_pearson"].append(np.corrcoef(a[m], b[m])[0, 1])
        a, b = (R1[:, 2:] - R1[:, 1:2]).ravel(), (R2[:, 2:] - R2[:, 1:2]).ravel()
        m = ~np.isnan(a) & ~np.isnan(b) & (a != 0) & (b != 0)
        r["agree"].append(np.mean(np.sign(a[m]) == np.sign(b[m])))
    rel = {k: float(2 * np.mean(v) / (1 + np.mean(v))) for k, v in r.items() if k != "agree"}
    return rel, float(np.mean(r["agree"]))


def ceiling(data, test, prompt_ids=None, n_splits=20, seed=0):
    """Noise ceilings from split halves of the M continuations (split_half_reliability). Ceiling for Pearson(A) =
    sqrt(r_A, Pearson) (exact under classical test theory); sqrt(r_A, Spearman) for Spearman(A) is an approximation (above
    the oracle value in simulation). sign_acc_halfM = the accuracy of the true sign against M/2-sample labels, from the
    split-half sign agreement (symmetric noise, equal effect sizes); it is not a bound at M. mc_ref_*: scores of the
    M-sample estimate V_MC itself (references, not upper bounds)."""
    rows = np.nonzero(test)[0] if np.asarray(test).dtype == bool else np.asarray(test)
    idx = data["idx"].iloc[rows]
    if prompt_ids is not None:
        idx = idx.assign(prompt_id=prompt_ids)
    rel, agree = split_half_reliability(data["o"][rows], n_splits, np.random.default_rng(seed))
    R = data["rate"][rows]
    m = ~np.isnan(R[:, 0])
    return {"reliability_A": rel["A"], "reliability_A_pearson": rel["A_pearson"], "reliability_Q": rel["Q"],
            "reliability_dQ": rel["dQ"], "pearson_A": float(np.sqrt(max(rel["A_pearson"], 0))),
            "spearman_A_approx": float(np.sqrt(max(rel["A"], 0))),
            "sign_acc_halfM": float((1 + np.sqrt(max(2 * agree - 1, 0))) / 2), "sign_agree_halves": agree,
            "mc_ref_auroc": auroc_within(R[m, 0], idx.reward.values[m], idx.prompt_id.values[m]),
            "mc_ref_succ_pcorr": partial_corr(R[m, 0], idx.reward.values[m].astype(float), np.zeros((m.sum(), 0)),
                                              idx.prompt_id.values[m])}


def ceiling_ci(data, test, n_boot, seed=0):
    """95 % percentile intervals of the ceilings over resamples of the test prompts (with replacement; a prompt drawn twice
    counts as two prompts)."""
    rng, idx = np.random.default_rng(seed), data["idx"]
    rows = [np.nonzero(test & (idx.prompt_id == p).values)[0] for p in idx.prompt_id[test].unique()]
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(len(rows), len(rows))
        sel = np.concatenate([rows[k] for k in draw])
        labels = np.concatenate([np.full(len(rows[k]), j) for j, k in enumerate(draw)])
        boots.append(ceiling(data, sel, labels))
    return {k: [float(np.nanpercentile([b[k] for b in boots], q)) for q in (2.5, 97.5)] for k in boots[0]}
