import datasets
import numpy as np
import pandas as pd
import pytest

from shape_rl import rollout
from shape_rl.rewards import grade, last_boxed

RAW = {
    "openai/gsm8k": {"question": ["1+1?", "2+2?", "3-1?"], "answer": ["so #### 2", "so #### 4", "so #### 2"]},
    "DigitalLearningGmbH/MATH-lighteval": {"problem": ["x?", "y?"], "solution": ["thus $\\boxed{\\frac{1}{2}}$.", "no box here"]},
}


def test_last_boxed_and_grade():
    assert last_boxed("a \\boxed{\\frac{1}{2}} b \\boxed{3}") == "3"
    assert last_boxed("x \\boxed{\\frac{a}{b}}") == "\\frac{a}{b}"
    assert last_boxed("the answer is 7") is None
    assert grade("so \\boxed{7}", True, "7") == 1
    assert grade("so \\boxed{8}", True, "7") == 0
    assert grade("the answer is 7", True, "7") == 0  # finished but unboxed: a failure
    assert grade("so far 3 + 4 = 7", False, "7") == -1  # truncated: unknown, never a failure


def test_topk_entropy():
    assert rollout.topk_entropy(np.log(np.full(4, 0.25))) == pytest.approx(np.log(4))
    p = np.array([0.5, 0.25])  # leftover 0.25 counts as one outcome
    assert rollout.topk_entropy(np.log(p)) == pytest.approx(-(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25)))


def test_branch_points():
    H_T = np.arange(100, dtype=float)
    ent, rnd = rollout.branch_points(H_T, np.random.default_rng(0), 8, 8, 0.8)
    assert len(ent) == 8 and len(rnd) == 8 and not set(ent) & set(rnd)
    assert ent.min() >= np.quantile(H_T[1:99], 0.8)
    assert min(ent.min(), rnd.min()) >= 1 and max(ent.max(), rnd.max()) <= 98


def test_alternatives_and_select():
    top = {5: (-0.1, 1), 7: (-2.0, 2), 2: (-3.0, 3), 9: (-4.0, 4)}
    assert rollout.alternatives(top, a_t=5, stop_ids={7}, k=2) == [2, 9]
    assert rollout.select_traces([1, 0, -1, 1, 1, 0, 0], 2, 2) == [0, 3, 1, 5]
    assert rollout.split_of("ff", [70, 15, 15]) == rollout.split_of("ff", [70, 15, 15]) in {"train", "val", "test"}


class FakeEngine:
    """Alternates a correct and a wrong boxed answer; top-k = the actual token plus 20 fixed others."""

    def __init__(self, tok):
        self.resp = [tok(f"so \\boxed{{{a}}}", add_special_tokens=False).input_ids + [tok.eos_token_id] for a in (2, 3)]

    def sample(self, prompts, n, max_tokens, temperature):
        return [[(self.resp[i % 2][:m], len(self.resp[i % 2]) <= m) for i in range(n)] for m in max_tokens]

    def topk(self, seqs, k):
        def top(a):
            return {a: (-0.5, 1), **{100 + j: (-1.0 - 0.1 * j, j + 2) for j in range(k) if 100 + j != a}}
        return [[None] + [top(a) for a in s[1:]] for s in seqs]


@pytest.mark.slow
def test_rollout_smoke(cfg_factory, monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setattr(datasets, "load_dataset", lambda path, *a, **k: datasets.Dataset.from_dict(RAW[path]))
    cfg = cfg_factory("rollout", overrides=["model=tiny", "rollout.samples=4", "rollout.M=2", "rollout.chunk_prompts=1"])
    rollout.run(cfg, engine=FakeEngine(AutoTokenizer.from_pretrained(cfg.model.path)))
    out = rollout.paths.run_dir(cfg) / "rollout"
    prompts = pd.read_parquet(out / "s000.prompts.parquet")
    assert len(prompts) == 4 and set(prompts.split) <= {"train", "val", "test"}  # MATH row without a box is dropped
    rows = pd.concat([pd.read_parquet(f) for f in sorted(out.glob("s000_c*[0-9].parquet"))])
    assert len(rows) and (out / "s000.done").exists()
    assert set(rows.kind) <= {"entropy", "random"} and (rows.pos == rows.prompt_len + rows.t).all()
    assert all(len(a) == 3 for a in rows.actions) and all(len(v) == 2 for v in rows.v_out) and all(len(q) == 6 for q in rows.q_out)
    assert set(np.concatenate(list(rows.v_out) + list(rows.q_out))) <= {-1, 0, 1}
    rollout.run(cfg, engine=None)  # resume: every chunk exists, so no engine is built
