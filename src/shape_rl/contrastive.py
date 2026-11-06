"""Contrastive projections z = g(h) for the offline estimators C0-C3 (shape_rl.ssl): the training data of each loss, the
losses and the projection fit. g = MLP, then unit norm. Each loss returns a scalar for one sampled batch.

Shapes: N = branch rows, S = 1 + A slots (slot 0 = h(s_t), slot 1 + j = h(s_t, a_j)), D = state dim, Z = projection dim,
B = batch.
"""
import numpy as np
import torch
import torch.nn.functional as F

from shape_rl.advantage import mlp


def train_data(idx, tr_rows, rate_NS, fin_NS, code_N, buckets, dev):
    """Sampling tables of the three losses (train rows only; code_N = numpy prompt codes).
    traj: C0, per trace (prompt code, [(row, slot)] of its on-trace states sorted by position: h_v at pos - 1, h_q[a_t] at
      pos; branch points t and t + 1 share a state, so it is kept once).
    prompt_items: C1, per prompt an array of labelled (row, slot, success-rate bucket).
    Q, Qok, pair_rows: C2, MC Q of each action, its label mask, and the rows whose a_t is labelled."""
    traj = []
    for _, g in idx[tr_rows].groupby("trace_id"):
        st = sorted([(p - 1, i, 0) for i, p in zip(g.index, g.pos)] + [(p, i, 1) for i, p in zip(g.index, g.pos)])
        st = [x for k, x in enumerate(st) if k == 0 or x[0] != st[k - 1][0]]
        traj.append((code_N[g.index[0]], [(i, s) for _, i, s in st]))
    prompt_items = []
    for _, g in idx[tr_rows].groupby("prompt_id"):
        ii, ss = np.nonzero(fin_NS[g.index.values] > 0)
        rows = g.index.values[ii]
        prompt_items.append(np.c_[rows, ss, np.minimum((rate_NS[rows, ss] * buckets).astype(int), buckets - 1)])
    return {"traj": traj, "prompt_items": prompt_items, "Q": torch.as_tensor(np.nan_to_num(rate_NS[:, 1:]), device=dev),
            "Qok": torch.as_tensor(fin_NS[:, 1:] > 0, device=dev), "pair_rows": np.nonzero(tr_rows & (fin_NS[:, 1] > 0))[0]}


def positive_logp(s, pos):
    """Supervised-contrastive loss: minus the mean log-probability of an anchor's positives, averaged over the anchors (rows)
    with >= 1 positive. s: logits, -inf at non-candidates; pos: positives mask of the same shape."""
    logp = s - torch.logsumexp(s, 1, keepdim=True)
    n = pos.sum(1)
    keep = n > 0
    if not keep.any():
        return s[torch.isfinite(s)].sum() * 0  # no positives in the batch: 0, with a graph to g
    return -(logp.masked_fill(~pos, 0).sum(1)[keep] / n[keep]).mean()


def supcon(z, pos, valid, tau):
    """Supervised contrastive loss: pos[i, j] marks positives of anchor i; valid[i, j] marks candidates (self excluded)."""
    return positive_logp((z @ z.T / tau).masked_fill(~valid, float("-inf")), pos)


def loss_trajectory(g, X_NSD, data, rng, E):
    """C0, label-free InfoNCE: positive = the next on-trace state (by position) in the same trace; negatives = other prompts."""
    dev, tr = X_NSD.device, data["traj"]
    pick = rng.choice(len(tr), E.proj.batch, replace=len(tr) < E.proj.batch)
    a, p, pr = [], [], []
    for k in pick:
        code, st = tr[k]
        u = rng.integers(len(st) - 1)
        a.append(st[u]), p.append(st[u + 1]), pr.append(code)
    a, p, pr = (torch.tensor(x, device=dev) for x in (a, p, pr))
    logits = g(X_NSD[a[:, 0], a[:, 1]]) @ g(X_NSD[p[:, 0], p[:, 1]]).T / E.proj.tau
    logits = logits.masked_fill((pr[:, None] == pr[None]) & ~torch.eye(len(pr), dtype=torch.bool, device=dev), float("-inf"))
    t = torch.arange(len(pr), device=dev)
    return (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2


def loss_bucket(g, X_NSD, data, rng, E):
    """C1, supervised contrastive: positives = same-prompt labelled states in the same success-rate bucket."""
    dev, groups = X_NSD.device, data["prompt_items"]
    n_p = max(1, E.proj.batch // E.per_prompt)
    ps = rng.choice(len(groups), n_p, replace=len(groups) < n_p)
    it = np.concatenate([groups[p][rng.choice(len(groups[p]), min(E.per_prompt, len(groups[p])), replace=False)] for p in ps])
    code = torch.tensor(np.repeat(ps, [min(E.per_prompt, len(groups[p])) for p in ps]), device=dev)
    z = g(X_NSD[torch.as_tensor(it[:, 0], device=dev), torch.as_tensor(it[:, 1], device=dev)])
    bucket = torch.as_tensor(it[:, 2], device=dev)
    eye = torch.eye(len(it), dtype=torch.bool, device=dev)
    return supcon(z, (code[:, None] == code[None]) & (bucket[:, None] == bucket[None]) & ~eye, ~eye, E.proj.tau)


def loss_pairs(g, X_NSD, data, rng, E):
    """C2, counterfactual pairs: anchor h(s_t, a_t); positives = replacements a' with |dQ| <= delta, hard negatives = |dQ| > delta;
    anchors of rows from other prompts are extra negatives. Candidates are indexed [anchor b, row c, action j]."""
    dev = X_NSD.device
    r = torch.as_tensor(rng.choice(data["pair_rows"], E.proj.batch, replace=False), device=dev)
    Bn, A = len(r), X_NSD.shape[1] - 1
    z_BAZ = g(X_NSD[r, 1:].flatten(0, 1)).view(Bn, A, -1)  # action 0 = a_t
    Q_BA, ok_BA, code = data["Q"][r], data["Qok"][r], data["code"][r]
    own = torch.eye(Bn, dtype=torch.bool, device=dev)[:, :, None]
    alt = (torch.arange(A, device=dev) > 0)[None, None]
    lab = ok_BA[None] & ok_BA[:, None, :1]
    pos = own & alt & lab & ((Q_BA[None] - Q_BA[:, None, :1]).abs() <= E.delta)
    valid = (own & alt & lab) | ((code[:, None] != code[None])[:, :, None] & ~alt)
    s = torch.einsum("bz,caz->bca", z_BAZ[:, 0], z_BAZ) / E.proj.tau
    return positive_logp(s.masked_fill(~valid, float("-inf")).flatten(1), pos.flatten(1))


LOSSES = {"trajectory": loss_trajectory, "bucket": loss_bucket, "pairs": loss_pairs}


def fit_projection(X_NSD, data, E, seed):
    """g_net trained by the estimator's contrastive loss (settings in E.proj); z = F.normalize(g_net(h))."""
    g_net = mlp(X_NSD.shape[-1], E.proj.hidden, E.proj.dim).to(X_NSD.device)
    opt = torch.optim.AdamW(g_net.parameters(), lr=E.proj.lr, weight_decay=E.proj.weight_decay)
    rng = np.random.default_rng(seed)
    for step in range(E.proj.steps):
        opt.zero_grad()
        loss = LOSSES[E.loss](lambda x: F.normalize(g_net(x), dim=-1), X_NSD, data, rng, E)
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"[ssl] {E.loss} step {step} loss {loss.item():.4f}", flush=True)
    return g_net.eval()
