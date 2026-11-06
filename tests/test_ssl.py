import json

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from shape_rl import ssl

ESTIMATORS = ["b0", "b1", "b2", "b3", "c0", "c1", "c2", "c3"]


def fake_dataset(d, N=240, A=3, M=8, D=16, seed=0):
    """A dataset dir like encode writes: success probability is a linear function of the state."""
    rng = np.random.default_rng(seed)
    h_NSD = rng.normal(size=(N, 1 + A, D)).astype(np.float32)
    p_NS = 1 / (1 + np.exp(-2 * h_NSD[..., 0]))
    o_NSM = (rng.random((N, 1 + A, M)) < p_NS[..., None]).astype(np.int64)
    o_NSM[rng.random((N, 1 + A, M)) < 0.05] = -1  # truncated
    trace = np.arange(N) // 8
    prompt = trace // 4
    idx = pd.DataFrame({"prompt_id": [f"p{p}" for p in prompt], "trace_id": [f"t{t}" for t in trace],
                        "split": np.array(["train", "train", "val", "test"])[prompt % 4], "reward": trace % 2,
                        "pass_rate": 0.5, "t": np.tile(np.arange(1, 9) * 5, N // 8), "trace_len": 50,
                        "kind": np.where(np.arange(N) % 2, "entropy", "random"), "entropy": rng.random(N),
                        "entropy_prev": rng.random(N), "logprob_prev": -rng.random(N),
                        "action_logprobs": list(-rng.random((N, A))), "v_out": list(o_NSM[:, 0]), "q_out": list(o_NSM[:, 1:].reshape(N, -1)),
                        "states": "states/s000_c000.safetensors", "row": np.arange(N)})
    idx["pos"] = 20 + idx.t
    (d / "states").mkdir(parents=True)
    save_file({"h_v": torch.tensor(h_NSD[:, 0]).bfloat16(), "h_q": torch.tensor(h_NSD[:, 1:]).bfloat16()}, d / "states/s000_c000.safetensors")
    idx.to_parquet(d / "index.parquet")


@pytest.mark.parametrize("est", ESTIMATORS)
def test_ssl_smoke(cfg_factory, est):
    cfg = cfg_factory("ssl", overrides=[f"estimator={est}", "ssl.data=data/fake", "ssl.seeds=[0,1]"]
                      + ([] if est in ("b0", "b1", "b2", "b3") else ["estimator.proj.steps=20", "estimator.proj.batch=32"]))
    fake_dataset(ssl.Path(cfg.output_root) / "data/fake")
    res = ssl.run(cfg)
    m = res["mean"]
    assert {"auroc", "succ_pcorr", "succ_pcorr_resid", "spearman_A", "spearman_A_resid", "sign_acc", "ece"} <= set(m)
    assert 0 <= m["sign_acc"] <= 1 and 0 <= m["ece"] <= 1 and 0 < res["ceiling"]["reliability_Q"] <= 1
    np.testing.assert_equal(json.loads((ssl.paths.run_dir(cfg) / "metrics.json").read_text())["mean"], m)  # NaN-safe (B0 is constant)
    if est == "b2":  # a linear probe recovers a linear signal
        assert m["spearman_A"] > 0.3
