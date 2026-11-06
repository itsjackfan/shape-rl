"""Exhaustive and property checks of the pure functions behind the RL advantages, DAPO batch selection and the grader.
Each test states the property in its docstring and checks it on every case in a bounded space (or on seeded random cases)."""
import itertools

import pytest
import torch

from shape_rl.advantage import gae, group_relative, leader_labels, step_advantage
from shape_rl.rewards import last_boxed
from shape_rl.trainer import select_groups


def random_rows(n, T, seed):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, T + 1, (n,), generator=g)
    mask = (torch.arange(T)[None] < lengths[:, None]).long()  # right-padded, some rows empty
    return torch.rand(n, generator=g, dtype=torch.float64), torch.rand(n, T, generator=g, dtype=torch.float64), mask


@pytest.mark.parametrize("seed", range(5))
def test_gae_lam1_is_reward_minus_value(seed):
    """lam = 1: A_t = r - V_t on every real token (the sum telescopes), 0 on padding."""
    r, V, mask = random_rows(16, 12, seed)
    A = gae(r, V, mask, 1.0)
    torch.testing.assert_close(A, (r[:, None] - V) * mask, rtol=0, atol=1e-12)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95])
def test_gae_closed_form(lam):
    """A_t = sum_{k >= 0} lam^k delta_{t+k} over the row's real tokens, with delta_last = r - V_last; lam = 0: A_t = delta_t."""
    r, V, mask = random_rows(16, 12, 7)
    A = gae(r, V, mask, lam)
    for i in range(len(r)):
        L = int(mask[i].sum())
        delta = [float(V[i, t + 1] - V[i, t]) for t in range(L - 1)] + ([float(r[i] - V[i, L - 1])] if L else [])
        ref = [sum(lam ** k * delta[t + k] for k in range(L - t)) for t in range(L)]
        torch.testing.assert_close(A[i, :L], torch.tensor(ref, dtype=torch.float64), rtol=0, atol=1e-12)
        assert (A[i, L:] == 0).all()


def all_similarity_patterns(n):
    """Every symmetric 0/1 similarity matrix on n items with a unit diagonal."""
    pairs = list(itertools.combinations(range(n), 2))
    for bits in itertools.product([0.0, 1.0], repeat=len(pairs)):
        sim = torch.eye(n)
        for (i, j), b in zip(pairs, bits):
            sim[i, j] = sim[j, i] = b
        yield sim


@pytest.mark.parametrize("n", range(1, 6))
def test_leader_labels_exhaustive(n):
    """Every item has a label; a label is a leader (labelled by itself); members are >= tau to their leader; a leader is
    < tau to every earlier leader (it was free when they ran); leaders appear in item order."""
    for sim in all_similarity_patterns(n):
        lab = leader_labels(sim, 0.5).tolist()
        leaders = sorted(set(lab))
        assert all(lab[c] == c for c in leaders) and all(x >= 0 for x in lab)
        assert all(sim[lab[i], i] >= 0.5 for i in range(n))
        assert all(sim[a, b] < 0.5 for a, b in itertools.combinations(leaders, 2))
        assert all(lab[i] <= i for i in range(n))


def test_leader_labels_low_self_similarity():
    """An item below tau to itself (a zero state) still gets its own cluster."""
    assert leader_labels(torch.zeros(3, 3), 0.5).tolist() == [0, 1, 2]


@pytest.mark.parametrize("seed", range(20))
def test_step_advantage_properties(seed):
    """Within each cluster A_S sums to 0 (r minus the cluster mean); A_S = 0 on invalid steps and in clusters whose
    members all come from one trajectory; the cluster count is the number of distinct labels."""
    g = torch.Generator().manual_seed(seed)
    G, S, D = 4, 3, 2
    z = torch.nn.functional.normalize(torch.randn(G, S, D, generator=g), dim=-1)
    ok = torch.rand(G, S, generator=g) > 0.3
    r = torch.randint(0, 2, (G,), generator=g).float()
    A, n_c, n_s = step_advantage(z, ok, r, 0.5)
    assert (A[~ok] == 0).all() and n_s <= n_c
    items = ok.nonzero()
    if len(items):
        zz = z[items[:, 0], items[:, 1]]
        lab = leader_labels(zz @ zz.T, 0.5)
        assert n_c == len(lab.unique())
        for c in lab.unique():
            m = lab == c
            a = A[items[m, 0], items[m, 1]]
            torch.testing.assert_close(a.sum(), torch.tensor(0.0), rtol=0, atol=1e-6)
            if items[m, 0].unique().numel() == 1:
                assert (a == 0).all()


def test_select_groups_exhaustive():
    """For 1-3 rounds of 1-4 groups and every keep pattern: one round's worth of distinct groups; the kept groups first,
    as many as exist (up to the limit); both parts in (round, group) order."""
    cases = 0
    for n_groups in range(1, 5):
        for R in range(1, 4):
            for bits in itertools.product([False, True], repeat=n_groups * R):
                keep = [list(bits[r * n_groups:(r + 1) * n_groups]) for r in range(R)]
                pick = select_groups(keep)
                flags = [keep[r][g] for r, g in pick]
                n_kept = sum(bits)
                assert len(pick) == n_groups == len(set(pick))
                assert flags == sorted(flags, reverse=True) and sum(flags) == min(n_groups, n_kept)
                k = sum(flags)
                assert pick[:k] == sorted(pick[:k]) and pick[k:] == sorted(pick[k:])
                cases += 1
    assert cases == sum(2 ** (n * R) for n in range(1, 5) for R in range(1, 4))


@pytest.mark.parametrize("seed", range(5))
def test_group_relative_zero_mean(seed):
    """Group-relative advantages sum to 0 within each group (scale none and group)."""
    r = torch.rand(24, generator=torch.Generator().manual_seed(seed))
    for scale in ("none", "group"):
        torch.testing.assert_close(group_relative(r, 4, scale).view(-1, 4).sum(1), torch.zeros(6), rtol=0, atol=1e-5)


TOK = ["\\boxed", "{", "}", "x", " ", "$", "."]


def boxed_spec(toks):
    """last_boxed by tokens: the last \\boxed (not part of a longer macro), then a brace group or, after white space, the
    run of characters up to white space or `$` without trailing `.`."""
    idx = [i for i, t in enumerate(toks) if t == "\\boxed"]
    if not idx:
        return None
    rest = toks[idx[-1] + 1:]
    if rest[:1] == ["x"]:
        return None
    k = next((i for i, t in enumerate(rest) if t != " "), len(rest))
    if rest[k:k + 1] == ["{"]:
        depth = 0
        for e in range(k, len(rest)):
            depth += {"{": 1, "}": -1}.get(rest[e], 0)
            if depth == 0:
                return "".join(rest[k + 1:e])
        return None
    if k == 0:
        return None
    arg = "".join(itertools.takewhile(lambda t: t not in (" ", "$"), rest[k:])).rstrip(".,;")
    return arg or None


def test_last_boxed_exhaustive():
    """last_boxed == the token-level spec on every string of <= 5 tokens over 7 tokens (19 608 strings)."""
    bad = [s for L in range(6) for toks in itertools.product(TOK, repeat=L)
           if last_boxed(s := "".join(toks)) != boxed_spec(list(toks))]
    assert not bad, bad[:5]
