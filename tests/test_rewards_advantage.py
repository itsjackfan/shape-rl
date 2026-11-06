import pytest
import torch

from shape_rl.advantage import AdvantageEstimator, GroupRelative, HiddenCluster, HiddenValue, ZeroAdvantage, group_relative
from shape_rl.rewards import Correctness, last_boxed

TINY = "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6"


def test_correctness_finished_only():
    reward = Correctness(TINY)
    eos = next(iter(reward.stops))
    completions = [[{"role": "assistant", "content": c}] for c in ["so \\boxed{\\frac{1}{2}}", "\\boxed{3}", "so \\boxed{\\frac{1}{2}}"]]
    ids = [[1, 2, eos], [1, eos], [1, 2, 3]]  # the third does not end with a stop id: truncated -> 0
    assert reward(completions, answer=["0.5", "4", "0.5"], completion_ids=ids) == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("text,expected", [("a \\boxed{x{y}} b", "x{y}"), ("\\boxed {5}", "5"), ("\\boxed 5. Note {x}", "5"),
                                           ("\\boxed 9$", "9"), ("\\boxed{1} then \\boxed{2}", "2"), ("\\boxedx {3}", None),
                                           ("\\boxed{open", None), ("none", None), ("\\boxed{}", "")])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


def test_group_relative_matches_trl_rule():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    a = group_relative(r, 3)
    g = r.view(2, 3)
    ref = ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-4)).flatten()
    assert torch.allclose(a, ref)
    assert GroupRelative()(r, 3, slice(3, 6), {}).shape == (3,)


def test_zero_estimator_shapes():
    rewards_N = torch.rand(12)
    adv = ZeroAdvantage()(rewards_N, 4, slice(4, 8), {})
    assert isinstance(ZeroAdvantage(), AdvantageEstimator) and adv.shape == (4,) and (adv == 0).all()


def _obs(est, n=4, T=6, D=8):
    h, mask = torch.randn(n, T, D), torch.ones(n, T, dtype=torch.long)
    mask[0, 3:] = 0
    return est.observe(h, mask, torch.zeros(n, T, dtype=torch.long)), mask


def test_hidden_value_lam1_is_r_minus_V():
    est = HiddenValue(kind="linear", train="none", lam=1.0)
    obs, mask = _obs(est)
    r = torch.tensor([1.0, 0.0, 1.0, 0.0, 9.0, 9.0])  # rows 0..3 are this process's
    A = est(r, 2, slice(0, 4), obs)
    assert A.shape == (4, 6)
    assert torch.allclose(A, (r[:4, None] - obs["V"]) * mask, atol=1e-6)


def test_hidden_value_gae_matches_loop():
    est = HiddenValue(kind="mlp", hidden=4, train="all", lam=0.9, steps=2, tokens=3)
    obs, mask = _obs(est)
    V, r = obs["V"], torch.tensor([1.0, 0.0, 1.0, float("nan")])
    A = est(r, 2, slice(0, 4), obs)
    for i in range(3):
        L = int(mask[i].sum())
        ref, run = torch.zeros(L), 0.0
        for t in reversed(range(L)):
            run = ((r[i] if t == L - 1 else V[i, t + 1]) - V[i, t]) + 0.9 * run
            ref[t] = run
        assert torch.allclose(A[i, :L], ref, atol=1e-5) and (A[i, L:] == 0).all()
    assert (A[3] == 0).all()  # unscorable row: no advantage
    assert "fit_loss" in est.metrics and est.flops > 0
    state = est.extra_state()
    est2 = HiddenValue(kind="mlp", hidden=4, train="all", lam=0.9)
    est2.load_extra_state(state)
    est2.head.eval()
    x = torch.ones(2, 8)
    assert torch.allclose(est2.head(x), est.head(x))


def test_hidden_cluster_shapes():
    est = HiddenCluster(TINY, tau=0.5)
    n, T, D = 4, 5, 8
    h = torch.randn(n, T, D)
    obs = est.observe(h, torch.ones(n, T, dtype=torch.long), torch.zeros(n, T, dtype=torch.long))
    A = est(torch.tensor([1.0, 0.0, 1.0, 1.0]), 2, slice(0, 4), obs)
    assert A.shape == (n, T) and torch.isfinite(A).all()
