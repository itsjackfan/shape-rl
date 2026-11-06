"""Advantage estimators: one subclass per ablation; each replaces TRL's group-relative advantage.

Shapes: N = completions in the global batch (all processes, prompt-major: rows g*G .. g*G+G-1 are prompt g's group),
n = this process's rows (`local`, a slice of N), G = generations per prompt, T = completion width, D = hidden size,
K = states kept per row for the online head fit, S = step states kept per row (HiddenCluster).
State contract (as encode.py): completion token t sits at column P + t of prompt + completion (P = padded prompt width), so
h_nTD[:, t] = post-norm last-layer state at column P + t - 1 = h(s_t), the input of V(s_t) (the state that predicts r_t).
rewards_N: weighted reward sum, NaN for an unscorable row (TRL's rule); such rows get advantage 0 and are not fit targets.
"""
import torch
import torch.nn.functional as F
from torch import nn
from trl.trainer.utils import nanstd


def group_relative(rewards_N, G: int, scale: str = "group"):
    """TRL's built-in advantage (sum_then_normalize): r - group mean, divided by (group std + 1e-4) if scale == "group"."""
    r_PG = rewards_N.view(-1, G)
    a_PG = r_PG - r_PG.nanmean(1, keepdim=True)
    if scale == "group":
        a_PG = a_PG / (nanstd(r_PG, dim=1).unsqueeze(1) + 1e-4)
    return a_PG.flatten()


def all_reduce_mean_(params):
    """Average gradients over processes so that every process keeps the same head."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for p in params:
            if p.grad is not None:
                torch.distributed.all_reduce(p.grad)
                p.grad /= torch.distributed.get_world_size()


class AdvantageEstimator(nn.Module):
    """Base class. If needs_states, the trainer calls observe(h_bTD, mask_bT, ids_bT) on every sub-batch of the scoring
    forward (no grad, rows in order) and passes the row-concatenated results as `obs`. forward returns the advantages of
    this process's rows: (n,) or (n, T). `flops` counts the estimator's own FLOPs on this process; `metrics` are logged
    under policy/estimator/*."""

    needs_states = False

    def __init__(self):
        super().__init__()
        self.flops, self.metrics, self.gather = 0.0, {}, lambda x: x

    def observe(self, h_bTD, mask_bT, ids_bT) -> dict:
        return {}

    def forward(self, rewards_N, G: int, local: slice, obs: dict) -> torch.Tensor:
        raise NotImplementedError

    def extra_state(self) -> dict:
        """State that a requeued job must restore (online heads and their optimizers)."""
        return {}

    def load_extra_state(self, state: dict):
        pass


class ZeroAdvantage(AdvantageEstimator):
    def forward(self, rewards_N, G, local, obs):
        return torch.zeros_like(rewards_N[local])


class GroupRelative(AdvantageEstimator):
    """TRL's group-relative advantage (reference implementation for the hook checks)."""

    def __init__(self, scale: str = "group"):
        super().__init__()
        self.scale = scale

    def forward(self, rewards_N, G, local, obs):
        return torch.nan_to_num(group_relative(rewards_N, G, self.scale), nan=0.0)[local]


def mlp(d_in, d_hid, d_out):
    """The ssl.py head body (also used for the contrastive projection)."""
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_hid, d_out))


class ValueHead(nn.Module):
    """logit P(success | prefix) from a state: standardize with (mu, sd), optional unit-norm projection z = g(h) (the
    contrastive estimators), then a linear or MLP readout. Same layout as the heads that ssl.py saves (head_seed*.pt)."""

    def __init__(self, D: int, proj: dict | None, hidden: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(D))
        self.register_buffer("sd", torch.ones(D))
        self.proj = mlp(D, proj["hidden"], proj["dim"]) if proj else None
        d = proj["dim"] if proj else D
        self.readout = nn.Linear(d, 1) if hidden == 0 else mlp(d, hidden, 1)

    def features(self, h):
        x = (h.float() - self.mu) / self.sd
        return F.normalize(self.proj(x), dim=-1) if self.proj is not None else x

    def forward(self, h):
        return self.readout(self.features(h))[..., 0]

    @classmethod
    def load(cls, path):
        spec = torch.load(path, map_location="cpu", weights_only=True)
        head = cls(spec["D"], spec["proj"], spec["hidden"])
        head.load_state_dict(spec["state_dict"])
        return head


class HiddenValue(AdvantageEstimator):
    """The proposal's method: a value head V(s_t) on the policy's own last-layer states, used as a per-token baseline.
    A_t = sum_k lam^k delta_{t+k}, delta_t = V(s_{t+1}) - V(s_t), delta_{last} = r - V(s_last) (GAE, gamma = 1); lam = 1
    gives A_t = r - V(s_t). combine: "replace" (A = A_value) or "add" (A = group-relative A + omega * A_value).
    Online fit after each batch: weighted BCE of V on the batch's outcome r (clamped to [0, 1]) at K sampled states per
    row; `train` = "readout" (the projection stays fixed), "all" or "none". init: a head file from ssl.py, or null
    (a new head: kind "linear" or "mlp", mu/sd from the first batch)."""

    needs_states = True

    def __init__(self, init=None, kind="mlp", hidden=512, train="all", lam=1.0, combine="replace", omega=1.0, scale="group",
                 lr=1.0e-3, steps=4, tokens=64, seed=0):
        super().__init__()
        self.head = ValueHead.load(init) if init else None
        self.kind, self.hidden, self.train_mode, self.lam = kind, hidden, train, lam
        self.combine, self.omega, self.scale, self.lr, self.steps, self.K = combine, omega, scale, lr, steps, tokens
        self.gen = torch.Generator().manual_seed(seed)
        self.opt = None

    def _params(self):
        if self.train_mode == "readout":
            return list(self.head.readout.parameters())
        return list(self.head.parameters()) if self.train_mode == "all" else []

    def _setup(self, h_bTD, mask_bT):
        if self.head is None:  # new head: standardize with the first batch's states, pooled over processes
            x = h_bTD[mask_bT.bool()].float()
            stats = torch.cat([x.sum(0), (x * x).sum(0), x.new_tensor([len(x)])])
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(stats)
            D = h_bTD.shape[-1]
            n = stats[-1].clamp(min=2)
            mu = stats[:D] / n
            var = (stats[D:2 * D] - n * mu * mu) / (n - 1)
            self.head = ValueHead(D, None, self.hidden if self.kind == "mlp" else 0)
            self.head.mu.copy_(mu.cpu())
            self.head.sd.copy_(var.clamp(min=0).sqrt().cpu() + 1e-4)
        self.head.to(h_bTD.device)
        if self.opt is None and self._params():
            self.opt = torch.optim.AdamW(self._params(), lr=self.lr, weight_decay=1e-2)
            if getattr(self, "_pending_opt", None):
                self.opt.load_state_dict(self._pending_opt)

    @torch.no_grad()
    def observe(self, h_bTD, mask_bT, ids_bT):
        self._setup(h_bTD, mask_bT)
        self.head.eval()
        V_bT = torch.sigmoid(self.head(h_bTD))
        n_par = sum(p.numel() for p in self.head.parameters())
        self.flops += 2.0 * n_par * float(mask_bT.sum())
        # K states per row for the online fit, uniform over the row's real tokens (with replacement)
        u = torch.rand(mask_bT.shape[0], self.K, generator=self.gen).to(h_bTD.device)
        n_tok = mask_bT.sum(1, keepdim=True)
        idx = (u * n_tok).long().clamp(max=mask_bT.shape[1] - 1)
        h_fit = torch.gather(h_bTD, 1, idx[..., None].expand(-1, -1, h_bTD.shape[-1]))
        return {"V": V_bT.float(), "mask": mask_bT, "h_fit": h_fit, "fit_ok": (n_tok > 0).expand(-1, self.K)}

    def value_advantage(self, r_n, V_nT, mask_nT):
        m = mask_nT.float()
        last = (m.sum(1) - 1).clamp(min=0).long()
        V_next = torch.cat([V_nT[:, 1:], V_nT[:, -1:]], 1)
        delta = (V_next - V_nT) * m
        delta[torch.arange(len(r_n)), last] = r_n - V_nT[torch.arange(len(r_n)), last]
        if self.lam == 1.0:
            A = torch.flip(torch.cumsum(torch.flip(delta, [1]), 1), [1])
        else:
            A, run = torch.zeros_like(delta), torch.zeros_like(r_n)
            for t in range(delta.shape[1] - 1, -1, -1):
                run = delta[:, t] + self.lam * run
                A[:, t] = run
        return A * m

    def forward(self, rewards_N, G, local, obs):
        r_n = rewards_N[local]
        ok_n = ~torch.isnan(r_n)
        r_n = torch.nan_to_num(r_n, nan=0.0)
        V, mask = obs["V"], obs["mask"]
        A = self.value_advantage(r_n, V, mask)
        if self.combine == "add":
            A = torch.nan_to_num(group_relative(rewards_N, G, self.scale), nan=0.0)[local, None] + self.omega * A
        A = A * ok_n[:, None]
        first = V[:, 0]
        self.metrics = {"V0_mean": float(first.mean()), "abs_adv_mean": float((A.abs() * mask).sum() / mask.sum().clamp(min=1))}
        if self.opt is not None and self.training:
            self._fit(obs["h_fit"], obs["fit_ok"] & ok_n[:, None], r_n.clamp(0, 1))
        return A

    def _fit(self, h_nKD, ok_nK, y_n):
        """Online fit; gradients are averaged over processes (every process keeps the same head)."""
        x, y = h_nKD[ok_nK], y_n[:, None].expand_as(ok_nK)[ok_nK]
        (self.head.readout if self.train_mode == "readout" else self.head).train()  # a fixed projection keeps dropout off
        n_par = sum(p.numel() for p in self._params())
        for _ in range(self.steps):
            self.opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.head(x), y) if len(x) else sum(p.sum() for p in self._params()) * 0
            loss.backward()
            all_reduce_mean_(self._params())
            self.opt.step()
        self.head.eval()
        self.flops += 6.0 * sum(p.numel() for p in self.head.parameters()) * len(x) * self.steps
        self.metrics["fit_loss"] = float(loss.detach())
        self.metrics["fit_params"] = n_par

    def extra_state(self):
        return {"head": None if self.head is None else self.head.state_dict(),
                "head_shape": None if self.head is None else (self.head.mu.shape[0], self.kind, self.hidden),
                "opt": None if self.opt is None else self.opt.state_dict(), "gen": self.gen.get_state()}

    def load_extra_state(self, state):
        if state.get("head") is not None:
            if self.head is None:
                D, kind, hidden = state["head_shape"]
                self.head = ValueHead(D, None, hidden if kind == "mlp" else 0)
            self.head.load_state_dict(state["head"])
        self._pending_opt = state.get("opt")
        self.gen.set_state(state["gen"])


class HiddenCluster(AdvantageEstimator):
    """GiGPO-style step groups formed in the policy's hidden-state space (the state grouping of BiPACE; its action-
    conditioned PACE baseline is not implemented). Steps start at t = 0 and after each token that contains a blank line
    ("\\n\\n"); at most S evenly spaced step starts per row are kept, and each token belongs to the last kept start <= t.
    Within each prompt group, step-start states with cosine similarity >= tau form a cluster (greedy leader clustering in
    row-major order). Step advantage A_S = r - mean r over the cluster's members; it is 0 if all members come from one
    trajectory. A_t = group-relative A (episode) + omega * A_S(step of t), as in GiGPO."""

    needs_states = True

    def __init__(self, model_path, tau=0.9, omega=1.0, steps=64, scale="group"):
        super().__init__()
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
        vocab = torch.zeros(max(len(tok), max(tok.get_vocab().values()) + 1), dtype=torch.bool)
        for t, i in tok.get_vocab().items():
            vocab[i] = "\n\n" in tok.convert_tokens_to_string([t])
        self.register_buffer("blank", vocab, persistent=False)
        self.tau, self.omega, self.S, self.scale = tau, omega, steps, scale

    @torch.no_grad()
    def observe(self, h_bTD, mask_bT, ids_bT):
        b, T = mask_bT.shape
        brk = torch.zeros_like(mask_bT, dtype=torch.bool)
        brk[:, 0] = True
        brk[:, 1:] = self.blank.to(ids_bT.device)[ids_bT[:, :-1].clamp(0, len(self.blank) - 1)]
        brk &= mask_bT.bool()
        start_bS = torch.zeros(b, self.S, dtype=torch.long, device=h_bTD.device)
        ok_bS = torch.zeros(b, self.S, dtype=torch.bool, device=h_bTD.device)
        step_bT = torch.zeros(b, T, dtype=torch.long, device=h_bTD.device)
        for i in range(b):
            s = brk[i].nonzero()[:, 0]
            if len(s) == 0:
                continue
            s = s[torch.linspace(0, len(s) - 1, min(len(s), self.S), device=s.device).round().long().unique()]
            start_bS[i, :len(s)], ok_bS[i, :len(s)] = s, True
            step_bT[i] = (torch.searchsorted(s, torch.arange(T, device=s.device), right=True) - 1).clamp(min=0)
        z = F.normalize(torch.gather(h_bTD, 1, start_bS[..., None].expand(-1, -1, h_bTD.shape[-1])).float(), dim=-1)
        return {"z": z.to(torch.bfloat16), "ok": ok_bS, "step": step_bT, "mask": mask_bT}

    def forward(self, rewards_N, G, local, obs):
        z_NSD, ok_NS = self.gather(obs["z"]), self.gather(obs["ok"])  # step states of every process (groups can straddle)
        r_N = torch.nan_to_num(rewards_N, nan=0.0)
        A_S_NS = torch.zeros(ok_NS.shape, device=r_N.device)
        n_clusters, n_shared = 0, 0
        for g in range(len(r_N) // G):
            rows = torch.arange(g * G, (g + 1) * G, device=r_N.device)
            items = ok_NS[rows].nonzero()  # (row in group, step)
            if len(items) == 0:
                continue
            zz = z_NSD[rows[items[:, 0]], items[:, 1]].float()
            sim = zz @ zz.T
            lab = torch.full((len(items),), -1, dtype=torch.long, device=zz.device)
            for i in range(len(items)):
                if lab[i] < 0:
                    lab[(lab < 0) & (sim[i] >= self.tau)] = i
            r_items = r_N[rows[items[:, 0]]]
            for c in lab.unique():
                m = lab == c
                n_clusters += 1
                if items[m, 0].unique().numel() > 1:
                    n_shared += 1
                    A_S_NS[rows[items[m, 0]], items[m, 1]] = r_items[m] - r_items[m].mean()
        A_S = torch.gather(A_S_NS[local], 1, obs["step"])
        A_E = torch.nan_to_num(group_relative(rewards_N, G, self.scale), nan=0.0)[local]
        A = (A_E[:, None] + self.omega * A_S) * obs["mask"] * (~torch.isnan(rewards_N[local]))[:, None]
        self.metrics = {"clusters_per_group": n_clusters / max(len(r_N) // G, 1),
                        "shared_cluster_frac": n_shared / max(n_clusters, 1)}
        return A
