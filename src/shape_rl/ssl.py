"""Offline advantage estimators on frozen last-layer states (state indexing: encode.py; labels: rollout.py).

A prefix-value estimator f maps a state to P(success | prefix). For branch row i, slot 0 is h_v = h(s_t) and slot 1 + j is
h_q[:, j] = h(s_t, a_j), so V_pred = f(slot 0), Q_pred(a_j) = f(slot 1 + j), A_pred = Q_pred - V_pred.
Labels: success rate over finished continuations (truncated ones excluded), weight = number finished.
Shapes: N = branch rows, S = 1 + A slots, D = state dim, Z = projection dim, I = labelled items (rows x slots), B = batch.
"""
import json
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn

from shape_rl import paths
from shape_rl.tracking import WandbLogger

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(d):
    """-> index df, states h_NSD (bf16, index order), successes and finished counts [N, S]."""
    idx = pd.read_parquet(d / "index.parquet")
    h = None
    for f, g in idx.groupby("states", sort=False):
        st = load_file(d / f)
        x_NSD = torch.cat([st["h_v"][:, None], st["h_q"]], 1)[torch.as_tensor(g.row.values)]
        h = h if h is not None else torch.empty(len(idx), *x_NSD.shape[1:], dtype=torch.bfloat16)
        h[torch.as_tensor(g.index.values)] = x_NSD
    S = h.shape[1]
    o_NSM = np.concatenate([np.stack(idx.v_out.values)[:, None], np.stack(idx.q_out.values).reshape(len(idx), S - 1, -1)], 1)
    return idx, h, (o_NSM == 1).sum(-1), (o_NSM >= 0).sum(-1), o_NSM


def rate(succ, fin):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(fin > 0, succ / fin, np.nan)


def scalar_feats(idx, S):
    """B1 features per slot: entropy of the distribution that produced the prefix's last token and that token's logprob."""
    lp = np.stack(idx.action_logprobs.values)
    ent = np.repeat(idx.entropy.values[:, None], S - 1, 1)
    return np.stack([np.concatenate([idx.entropy_prev.values[:, None], ent], 1),
                     np.concatenate([idx.logprob_prev.values[:, None], lp], 1)], -1)  # [N, S, 2]


def mlp(d_in, d_hid, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_hid, d_out))


def fit_head(x_tr, y_tr, w_tr, x_va, y_va, w_va, E, hidden):
    """Weighted BCE on soft labels; linear if hidden == 0 else 1-hidden-layer MLP; early stop on val loss. E = estimator.head."""
    net = (nn.Linear(x_tr.shape[1], 1) if hidden == 0 else mlp(x_tr.shape[1], hidden, 1)).to(DEV)
    with torch.no_grad():  # start at the train base rate
        (net if hidden == 0 else net[-1]).bias.fill_(torch.logit(((y_tr * w_tr).sum() / w_tr.sum()).clamp(1e-3, 1 - 1e-3)).item())
    opt = torch.optim.AdamW(net.parameters(), lr=E.lr, weight_decay=E.weight_decay)
    loss = lambda x, y, w: (F.binary_cross_entropy_with_logits(net(x)[:, 0], y, reduction="none") * w).sum() / w.sum()
    best, state, bad = float("inf"), None, 0
    for _ in range(E.max_epochs):
        net.train()
        for b in torch.randperm(len(x_tr), device=DEV).split(E.batch):
            opt.zero_grad()
            loss(x_tr[b], y_tr[b], w_tr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss(x_va, y_va, w_va).item()
        if v < best - 1e-5:
            best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        elif (bad := bad + 1) >= E.patience:
            break
    net.load_state_dict(state)
    return net.eval()


# ---- contrastive projections z = g(h), unit norm; each loss returns a scalar for one sampled batch


def loss_trajectory(g, X_NSD, data, rng, E):
    """C0, label-free InfoNCE: positive = the next on-trace state (by position) in the same trace; negatives = other prompts."""
    tr = data["traj"]  # list of (prompt code, [(row, slot)] sorted by position), train traces only
    pick = rng.choice(len(tr), E.proj.batch, replace=len(tr) < E.proj.batch)
    a, p, pr = [], [], []
    for k in pick:
        code, st = tr[k]
        u = rng.integers(len(st) - 1)
        a.append(st[u]), p.append(st[u + 1]), pr.append(code)
    a, p = torch.tensor(a, device=DEV), torch.tensor(p, device=DEV)
    za, zp = g(X_NSD[a[:, 0], a[:, 1]]), g(X_NSD[p[:, 0], p[:, 1]])
    pr = torch.tensor(pr, device=DEV)
    logits = za @ zp.T / E.proj.tau
    logits = logits.masked_fill((pr[:, None] == pr[None]) & ~torch.eye(len(pr), dtype=torch.bool, device=DEV), float("-inf"))
    t = torch.arange(len(pr), device=DEV)
    return (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2


def supcon(z, pos, valid, tau):
    """Supervised contrastive loss: pos[i, j] marks positives of anchor i; valid[i, j] marks candidates (self excluded)."""
    s = (z @ z.T / tau).masked_fill(~valid, float("-inf"))
    logp = s - torch.logsumexp(s, 1, keepdim=True)
    n = pos.sum(1)
    keep = n > 0
    if not keep.any():
        return z.sum() * 0
    return -(logp.masked_fill(~pos, 0).sum(1)[keep] / n[keep]).mean()


def loss_bucket(g, X_NSD, data, rng, E):
    """C1, supervised contrastive: positives = same-prompt labelled states in the same success-rate bucket."""
    groups = data["prompt_items"]  # per train prompt: array of (row, slot, bucket)
    n_p = max(1, E.proj.batch // E.per_prompt)
    ps = rng.choice(len(groups), n_p, replace=len(groups) < n_p)
    it = np.concatenate([groups[p][rng.choice(len(groups[p]), min(E.per_prompt, len(groups[p])), replace=False)] for p in ps])
    code = torch.tensor(np.repeat(ps, [min(E.per_prompt, len(groups[p])) for p in ps]), device=DEV)
    z = g(X_NSD[torch.as_tensor(it[:, 0], device=DEV), torch.as_tensor(it[:, 1], device=DEV)])
    bucket = torch.as_tensor(it[:, 2], device=DEV)
    eye = torch.eye(len(it), dtype=torch.bool, device=DEV)
    return supcon(z, (code[:, None] == code[None]) & (bucket[:, None] == bucket[None]) & ~eye, ~eye, E.proj.tau)


def loss_pairs(g, X_NSD, data, rng, E):
    """C2, counterfactual pairs: anchor h(s_t, a_t); positives = replacements a' with |dQ| <= delta, hard negatives = |dQ| > delta;
    anchors of rows from other prompts are extra negatives. Candidates are indexed [anchor b, row c, action j]."""
    r = torch.as_tensor(rng.choice(data["pair_rows"], E.proj.batch, replace=False), device=DEV)
    Bn, A = len(r), X_NSD.shape[1] - 1
    z_BAZ = g(X_NSD[r, 1:].flatten(0, 1)).view(Bn, A, -1)  # action 0 = a_t
    Q_BA, ok_BA, code = data["Q"][r], data["Qok"][r], data["code"][r]
    own = torch.eye(Bn, dtype=torch.bool, device=DEV)[:, :, None]
    alt = (torch.arange(A, device=DEV) > 0)[None, None]
    lab = ok_BA[None] & ok_BA[:, None, :1]
    dq = (Q_BA[None] - Q_BA[:, None, :1]).abs()
    pos = own & alt & lab & (dq <= E.delta)
    valid = (own & alt & lab) | ((code[:, None] != code[None])[:, :, None] & ~alt)
    s = torch.einsum("bz,caz->bca", z_BAZ[:, 0], z_BAZ) / E.proj.tau
    s = s.masked_fill(~valid, float("-inf")).flatten(1)
    logp, pos = s - torch.logsumexp(s, 1, keepdim=True), pos.flatten(1)
    n = pos.sum(1)
    keep = n > 0
    if not keep.any():
        return z_BAZ.sum() * 0
    return -(logp.masked_fill(~pos, 0).sum(1)[keep] / n[keep]).mean()


LOSSES = {"trajectory": loss_trajectory, "bucket": loss_bucket, "pairs": loss_pairs}


def fit_projection(X_NSD, data, E, seed):
    """z = g(h): MLP, unit norm, trained by the estimator's contrastive loss (settings in E.proj)."""
    g_net = mlp(X_NSD.shape[-1], E.proj.hidden, E.proj.dim).to(DEV)
    g = lambda x: F.normalize(g_net(x), dim=-1)
    opt = torch.optim.AdamW(g_net.parameters(), lr=E.proj.lr, weight_decay=E.proj.weight_decay)
    rng = np.random.default_rng(seed)
    for step in range(E.proj.steps):
        opt.zero_grad()
        loss = LOSSES[E.loss](g, X_NSD, data, rng, E)
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"[ssl] {E.loss} step {step} loss {loss.item():.4f}", flush=True)
    g_net.eval()
    return lambda x: F.normalize(g_net(x), dim=-1)


def kmeans(x, K, iters, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=gen)[:K].to(x.device)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        c = torch.stack([x[a == k].mean(0) if (a == k).any() else c[k] for k in range(K)])
    return c


# ---- estimators: each returns p_NS, predicted P(success) for every row and slot


def predict(net, X_NSD):
    with torch.no_grad():
        return torch.cat([torch.sigmoid(net(x)[:, 0]) for x in X_NSD.flatten(0, 1).split(65536)]).view(X_NSD.shape[:2])


def estimate(data, E, seed):
    idx, X_NSD, y_NS, w_NS, tr, va = data["idx"], data["X"], data["y"], data["w"], data["tr"], data["va"]
    torch.manual_seed(seed)
    if E.kind == "pass_rate":  # B0: GRPO's per-prompt baseline, the same value for every prefix of a prompt
        return np.repeat(idx.pass_rate.values[:, None], X_NSD.shape[1], 1)
    if E.kind == "scalar":  # B1: entropy and logprob only
        X_NSD = data["scalar"]
    if E.kind in ("scalar", "linear", "mlp"):
        Xin = X_NSD
    else:  # contrastive projection, then a fixed readout
        g = fit_projection(X_NSD, data, E, seed)
        with torch.no_grad():
            Xin = torch.cat([g(x) for x in X_NSD.flatten(0, 1).split(65536)]).view(*X_NSD.shape[:2], -1)
    if E.kind == "kmeans":  # C3: cluster-mean value
        z_tr = Xin[tr]
        c = kmeans(z_tr, E.clusters, E.kmeans_iters, seed)
        a_tr = torch.cdist(z_tr, c).argmin(1)
        num = torch.zeros(E.clusters, device=DEV).index_add_(0, a_tr, y_NS[tr] * w_NS[tr])
        den = torch.zeros(E.clusters, device=DEV).index_add_(0, a_tr, w_NS[tr])
        prior = (y_NS[tr] * w_NS[tr]).sum() / w_NS[tr].sum()
        val = (num + prior) / (den + 1)  # one pseudo-count at the train mean
        a = torch.cat([torch.cdist(x, c).argmin(1) for x in Xin.flatten(0, 1).split(65536)])
        return val[a].view(Xin.shape[:2]).cpu().numpy()
    hidden = E.hidden if E.kind == "mlp" else 0
    net = fit_head(Xin[tr], y_NS[tr], w_NS[tr], Xin[va], y_NS[va], w_NS[va], E.head, hidden)
    return predict(net, Xin).cpu().numpy()


# ---- metrics on held-out prompts


def ece(p, y, w, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    W = np.bincount(b, w, bins)
    gap = np.abs(np.bincount(b, w * p, bins) - np.bincount(b, w * y, bins))
    return float(gap.sum() / W.sum())


def residual(x, Z):
    Z1 = np.c_[np.ones(len(Z)), (Z - Z.mean(0)) / (Z.std(0) + 1e-9)]
    return x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]


def partial_corr(x, y, Z, groups):
    """Correlation of x and y after removing per-group means (prompt fixed effects) and then a linear fit on Z from both."""
    if len(x) < 3:
        return float("nan")
    code = pd.factorize(groups)[0]
    D = np.c_[x, y, Z].astype(float)
    mean = np.zeros((code.max() + 1, D.shape[1]))
    np.add.at(mean, code, D)
    d = D - (mean / np.bincount(code)[:, None])[code]
    r = d[:, :2] - d[:, 2:] @ np.linalg.lstsq(d[:, 2:], d[:, :2], rcond=None)[0] if Z.shape[1] else d[:, :2]
    return float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]) if r[:, 0].std() > 1e-12 else float("nan")


def auroc_within(score, label, prompt):
    a = [roc_auc_score(label[m], score[m]) for p in np.unique(prompt) if len(np.unique(label[m := prompt == p])) == 2]
    return float(np.mean(a))


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
    out["spearman_A"] = float(spearmanr(A_pr[m], A_mc[m])[0]) if A_pr[m].std() > 0 else float("nan")
    Zp = np.concatenate([np.repeat(conf[:, None], A_mc.shape[1], 1), data["trunc_q"][test][..., None]], -1)[m]
    ra, rm = residual(pd.Series(A_pr[m]).rank().values, Zp), residual(pd.Series(A_mc[m]).rank().values, Zp)
    out["spearman_A_resid"] = float(np.corrcoef(ra, rm)[0, 1]) if A_pr[m].std() > 0 else float("nan")
    dq_mc, dq_pr = R[:, 2:] - R[:, 1:2], p[:, 2:] - p[:, 1:2]
    m = ~np.isnan(dq_mc) & (dq_mc != 0)
    hit = np.where(dq_pr == 0, 0.5, np.sign(dq_pr) == np.sign(dq_mc))
    out["sign_acc"], out["n_sign_pairs"] = float(np.mean(hit[m])), int(m.sum())
    for kind in ("entropy", "random"):  # by branch-point kind (H1 x H2)
        k = (idx.kind.values == kind)[:, None]
        out[f"sign_acc_{kind}"] = float(np.mean(hit[m & k]))
        mk = ~np.isnan(A_mc) & k
        out[f"spearman_A_{kind}"] = float(spearmanr(A_pr[mk], A_mc[mk])[0]) if A_pr[mk].std() > 0 else float("nan")
    m = fin > 0
    out["ece"] = ece(np.clip(p[m], 0, 1), R[m], fin[m])
    out["n_test_prompts"] = int(idx.prompt_id.nunique())
    return out


def ceiling(data, test, n_splits=20, seed=0):
    """Noise ceiling from split halves of the M continuations (Spearman-Brown corrected to M): reliability r of A_MC, Q_MC, dQ;
    ceiling for Spearman(A) = sqrt(r_A); for sign accuracy, the accuracy of the true sign against M-sample labels implied by the
    split-half sign agreement (symmetric noise); for AUROC, the AUROC of the MC estimate V_MC itself."""
    o, idx = data["o"][test], data["idx"][test]
    rng = np.random.default_rng(seed)
    M = o.shape[-1]
    r = {"A": [], "Q": [], "dQ": [], "agree": []}
    for _ in range(n_splits):
        perm = rng.permutation(M)
        h1, h2 = o[..., perm[:M // 2]], o[..., perm[M // 2:]]
        R1, R2 = rate((h1 == 1).sum(-1), (h1 >= 0).sum(-1)), rate((h2 == 1).sum(-1), (h2 >= 0).sum(-1))
        for k, f in (("A", lambda R: R[:, 1:] - R[:, :1]), ("Q", lambda R: R[:, 1:]), ("dQ", lambda R: R[:, 2:] - R[:, 1:2])):
            a, b = f(R1).ravel(), f(R2).ravel()
            m = ~np.isnan(a) & ~np.isnan(b)
            r[k].append(spearmanr(a[m], b[m])[0])
        a, b = (R1[:, 2:] - R1[:, 1:2]).ravel(), (R2[:, 2:] - R2[:, 1:2]).ravel()
        m = ~np.isnan(a) & ~np.isnan(b) & (a != 0) & (b != 0)
        r["agree"].append(np.mean(np.sign(a[m]) == np.sign(b[m])))
    sb = lambda x: 2 * x / (1 + x)
    rel = {k: float(sb(np.mean(v))) for k, v in r.items() if k != "agree"}
    agree = float(np.mean(r["agree"]))
    R = data["rate"][test]
    m = ~np.isnan(R[:, 0])
    return {"reliability_A": rel["A"], "reliability_Q": rel["Q"], "reliability_dQ": rel["dQ"],
            "spearman_A": float(np.sqrt(max(rel["A"], 0))), "sign_acc_halfM": float((1 + np.sqrt(max(2 * agree - 1, 0))) / 2),
            "sign_agree_halves": agree,
            "auroc": auroc_within(R[m, 0], idx.reward.values[m], idx.prompt_id.values[m]),
            "succ_pcorr": partial_corr(R[m, 0], idx.reward.values[m].astype(float), np.zeros((m.sum(), 0)), idx.prompt_id.values[m])}


def prepare(cfg):
    idx, h_NSD, succ, fin, o = load(Path(cfg.output_root) / cfg.ssl.data)
    N, S = succ.shape
    split = idx.split.values
    tr_rows, va_rows = split == "train", split == "val"
    w = torch.as_tensor(fin, dtype=torch.float32)
    y = torch.as_tensor(np.nan_to_num(rate(succ, fin)), dtype=torch.float32)
    lab = fin > 0
    mu = h_NSD[torch.as_tensor(tr_rows)].float().flatten(0, 1).mean(0)
    sd = h_NSD[torch.as_tensor(tr_rows)].float().flatten(0, 1).std(0) + 1e-4
    X = ((h_NSD.to(DEV).float() - mu.to(DEV)) / sd.to(DEV))
    sc = scalar_feats(idx, S)
    sc = (sc - sc[tr_rows].reshape(-1, 2).mean(0)) / sc[tr_rows].reshape(-1, 2).std(0)
    item = lambda rows: torch.as_tensor(rows[:, None] & lab, device=DEV)
    tr_i, va_i = item(tr_rows), item(va_rows)
    data = {"idx": idx, "X": X, "y": y.to(DEV), "w": w.to(DEV), "tr": tr_i, "va": va_i, "o": o, "fin": fin, "rate": rate(succ, fin),
            "scalar": torch.as_tensor(sc, dtype=torch.float32, device=DEV)}
    trunc = (o == -1).mean(-1)
    data["conf"] = np.c_[idx.t.values, idx.trace_len.values, idx.t.values / idx.trace_len.values, trunc[:, 0]]
    data["trunc_q"] = trunc[:, 1:]
    code = pd.factorize(idx.prompt_id)[0]
    data["code"] = torch.as_tensor(code, device=DEV)
    # C0: on-trace states (h_v at pos - 1, h_q[a_t] at pos) of train traces, sorted by position
    traj = []
    for tid, g in idx[tr_rows].groupby("trace_id"):
        st = sorted([(p - 1, i, 0) for i, p in zip(g.index, g.pos)] + [(p, i, 1) for i, p in zip(g.index, g.pos)])
        traj.append((code[g.index[0]], [(i, s) for _, i, s in st]))
    data["traj"] = traj
    # C1: labelled train items per prompt with their success-rate bucket
    R = data["rate"]
    data["prompt_items"] = []
    for p, g in idx[tr_rows].groupby("prompt_id"):
        ii, ss = np.nonzero(lab[g.index.values])
        rows = g.index.values[ii]
        nb = cfg.estimator.get("buckets", 4)
        data["prompt_items"].append(np.c_[rows, ss, np.minimum((R[rows, ss] * nb).astype(int), nb - 1)])
    # C2: train rows whose a_t is labelled
    data["Q"] = torch.as_tensor(np.nan_to_num(R[:, 1:]), device=DEV)
    data["Qok"] = torch.as_tensor(fin[:, 1:] > 0, device=DEV)
    data["pair_rows"] = np.nonzero(tr_rows & (fin[:, 1] > 0))[0]
    return data


def run(cfg):
    paths.check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    out = paths.job_dir(paths.run_dir(cfg))
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    data = prepare(cfg)
    test, val = ((data["idx"].split == s).values for s in ("test", "val"))
    res = {"ceiling": ceiling(data, test), "seeds": {}, "val_seeds": {}}  # val: model choices (C3's representation)
    for seed in cfg.ssl.seeds:
        t0 = time.time()
        p_NS = estimate(data, cfg.estimator, seed)
        res["seeds"][seed], res["val_seeds"][seed] = metrics(p_NS, data, test), metrics(p_NS, data, val)
        np.save(out / f"pred_seed{seed}.npy", p_NS.astype(np.float32))
        print(f"[ssl] {cfg.ablation} seed {seed}: {res['seeds'][seed]} ({time.time() - t0:.0f}s)", flush=True)
    keys = next(iter(res["seeds"].values())).keys()
    res["mean"] = {k: float(np.mean([s[k] for s in res["seeds"].values()])) for k in keys}
    res["std"] = {k: float(np.std([s[k] for s in res["seeds"].values()])) for k in keys}
    res["val_mean"] = {k: float(np.mean([s[k] for s in res["val_seeds"].values()])) for k in keys}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    logger = WandbLogger(cfg, out)
    logger.log({**{f"test/{k}": v for k, v in res["mean"].items()}, **{f"test/ceiling/{k}": v for k, v in res["ceiling"].items()},
                "test/step": 0})
    logger.finish()
    return res


@hydra.main(config_path="../../configs", config_name="ssl", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
