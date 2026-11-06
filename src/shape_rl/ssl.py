"""Offline advantage estimators on frozen last-layer states (state indexing: encode.py; labels: rollout.py).

A prefix-value estimator f maps a state to P(success | prefix). For branch row i, slot 0 is h_v = h(s_t) and slot 1 + j is
h_q[:, j] = h(s_t, a_j), so V_pred = f(slot 0), Q_pred(a_j) = f(slot 1 + j), A_pred = Q_pred - V_pred.
Labels: success rate over finished continuations (truncated ones excluded), weight = number finished.
Contrastive projections: contrastive.py. Metrics and noise ceilings: ssl_metrics.py.
Shapes: N = branch rows, S = 1 + A slots, D = state dim, Z = projection dim, I = labelled items (rows x slots).
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
from torch import nn

from shape_rl import contrastive, paths
from shape_rl.advantage import mlp
from shape_rl.ssl_metrics import ceiling, ceiling_ci, metrics, rate
from shape_rl.tracking import WandbLogger

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(d):
    """-> index df, states h_NSD (bf16, index order), successes and finished counts [N, S], outcome codes [N, S, M]."""
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


def scalar_feats(idx, S):
    """B1 features per slot: entropy of the distribution that produced the prefix's last token and that token's logprob."""
    lp = np.stack(idx.action_logprobs.values)
    ent = np.repeat(idx.entropy.values[:, None], S - 1, 1)
    return np.stack([np.concatenate([idx.entropy_prev.values[:, None], ent], 1),
                     np.concatenate([idx.logprob_prev.values[:, None], lp], 1)], -1)  # [N, S, 2]


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


def kmeans(x, K, iters, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=gen)[:K].to(x.device)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        c = torch.stack([x[a == k].mean(0) if (a == k).any() else c[k] for k in range(K)])
    return c


# ---- estimators: each returns p_NS, predicted P(success) for every row and slot


def batched(fn, X_NSD):
    """fn over all rows and slots in blocks of 65536 items -> [N, S, ...]."""
    with torch.no_grad():
        return torch.cat([fn(x) for x in X_NSD.flatten(0, 1).split(65536)]).view(*X_NSD.shape[:2], -1)


def kmeans_value(z_NSZ, y_NS, w_NS, tr, E, seed):
    """C3: k-means on the train items' z, value = cluster mean success (one pseudo-count at the train mean)."""
    c = kmeans(z_NSZ[tr], E.clusters, E.kmeans_iters, seed)
    a_tr = torch.cdist(z_NSZ[tr], c).argmin(1)
    num = torch.zeros(E.clusters, device=DEV).index_add_(0, a_tr, y_NS[tr] * w_NS[tr])
    den = torch.zeros(E.clusters, device=DEV).index_add_(0, a_tr, w_NS[tr])
    prior = (y_NS[tr] * w_NS[tr]).sum() / w_NS[tr].sum()
    return ((num + prior) / (den + 1))[batched(lambda x: torch.cdist(x, c).argmin(1), z_NSZ)[..., 0]]


def head_spec(data, readout, g_net, E):
    """The RL starting head (advantage.ValueHead layout: standardization, optional projection, readout)."""
    sd = {"mu": data["mu"].cpu(), "sd": data["sd"].cpu(), **{f"readout.{k}": v.cpu() for k, v in readout.state_dict().items()}}
    if g_net is not None:
        sd.update({f"proj.{k}": v.cpu() for k, v in g_net.state_dict().items()})
    return {"D": data["X"].shape[-1], "proj": OmegaConf.to_container(E.proj) if g_net is not None else None,
            "hidden": E.hidden if E.kind == "mlp" else 0, "state_dict": sd}


def estimate(data, E, seed):
    """-> (p_NS, RL head spec or None). B0 pass rate; B1 scalar features; B2/B3 linear/MLP head on h; C0-C2 contrastive
    projection + linear readout; C3 k-means on the projection."""
    torch.manual_seed(seed)
    if E.kind == "pass_rate":  # B0: GRPO's per-prompt baseline, the same value for every prefix of a prompt
        return np.repeat(data["idx"].pass_rate.values[:, None], data["X"].shape[1], 1), None
    X, g_net = data["scalar"] if E.kind == "scalar" else data["X"], None
    if E.kind in ("contrastive", "kmeans"):
        g_net = contrastive.fit_projection(data["X"], data, E, seed)
        X = batched(lambda x: F.normalize(g_net(x), dim=-1), data["X"])
    y, w, tr, va = data["y"], data["w"], data["tr"], data["va"]
    if E.kind == "kmeans":
        return kmeans_value(X, y, w, tr, E, seed).cpu().numpy(), None
    net = fit_head(X[tr], y[tr], w[tr], X[va], y[va], w[va], E.head, E.hidden if E.kind == "mlp" else 0)
    p_NS = batched(lambda x: torch.sigmoid(net(x)[:, 0]), X)[..., 0].cpu().numpy()
    return p_NS, None if E.kind == "scalar" else head_spec(data, net, g_net, E)


def prepare(cfg):
    """Standardized states, labels, train/val item masks, confounds and the contrastive sampling tables."""
    idx, h_NSD, succ, fin, o = load(Path(cfg.output_root) / cfg.ssl.data)
    tr_rows, va_rows = (idx.split.values == s for s in ("train", "val"))
    h_tr = h_NSD[torch.as_tensor(tr_rows)].float().flatten(0, 1)
    mu, sd = h_tr.mean(0), h_tr.std(0) + 1e-4
    sc = scalar_feats(idx, h_NSD.shape[1])
    sc = (sc - sc[tr_rows].reshape(-1, 2).mean(0)) / sc[tr_rows].reshape(-1, 2).std(0)
    item = lambda rows: torch.as_tensor(rows[:, None] & (fin > 0), device=DEV)
    R, trunc = rate(succ, fin), (o == -1).mean(-1)
    code = pd.factorize(idx.prompt_id)[0]
    data = {"idx": idx, "X": (h_NSD.to(DEV).float() - mu.to(DEV)) / sd.to(DEV), "mu": mu, "sd": sd,
            "y": torch.as_tensor(np.nan_to_num(R), dtype=torch.float32, device=DEV),
            "w": torch.as_tensor(fin, dtype=torch.float32, device=DEV), "tr": item(tr_rows), "va": item(va_rows),
            "o": o, "fin": fin, "rate": R, "scalar": torch.as_tensor(sc, dtype=torch.float32, device=DEV),
            "conf": np.c_[idx.t.values, idx.trace_len.values, idx.t.values / idx.trace_len.values, trunc[:, 0]],
            "trunc_q": trunc[:, 1:], "code": torch.as_tensor(code, device=DEV)}
    data.update(contrastive.train_data(idx, tr_rows, R, fin, code, cfg.estimator.get("buckets", 4), DEV))
    return data


def across_seeds(per_seed, fn):
    """{metric: fn over seeds} from {seed: {metric: value}}."""
    return {k: float(fn([m[k] for m in per_seed.values()])) for k in next(iter(per_seed.values()))}


def run(cfg):
    paths.guard(cfg)
    out = paths.job_dir(paths.run_dir(cfg))
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    data = prepare(cfg)
    test, val = ((data["idx"].split == s).values for s in ("test", "val"))
    res = {"ceiling": ceiling(data, test), "ceiling_ci95": ceiling_ci(data, test, cfg.ssl.ceiling_boot),
           "seeds": {}, "val_seeds": {}}  # val: model choices (C3's representation)
    for seed in cfg.ssl.seeds:
        t0 = time.time()
        p_NS, head = estimate(data, cfg.estimator, seed)
        res["seeds"][seed], res["val_seeds"][seed] = metrics(p_NS, data, test), metrics(p_NS, data, val)
        np.save(out / f"pred_seed{seed}.npy", p_NS.astype(np.float32))
        if head is not None:  # the RL estimator's starting head (advantage.HiddenValue init)
            torch.save(head, out / f"head_seed{seed}.pt")
        print(f"[ssl] {cfg.ablation} seed {seed}: {res['seeds'][seed]} ({time.time() - t0:.0f}s)", flush=True)
    res.update(mean=across_seeds(res["seeds"], np.mean), std=across_seeds(res["seeds"], np.std),
               val_mean=across_seeds(res["val_seeds"], np.mean))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    logger = WandbLogger(cfg, out)
    ceil = {f"test/{'mc_ref' if k.startswith('mc_ref') else 'ceiling'}/{k.removeprefix('mc_ref_')}": v
            for k, v in res["ceiling"].items()}
    logger.log({**{f"test/{k}": v for k, v in res["mean"].items()}, **ceil, "test/step": 0, "test/flops": 0.0})
    logger.finish()
    return res


@hydra.main(config_path="../../configs", config_name="ssl", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
