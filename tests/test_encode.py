import datasets
import pandas as pd
import pytest
from safetensors.torch import load_file

from shape_rl import encode, rollout
from test_rollout import RAW, FakeEngine


@pytest.mark.slow
def test_encode_smoke(cfg_factory, monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setattr(datasets, "load_dataset", lambda path, *a, **k: datasets.Dataset.from_dict(RAW[path]))
    cfg = cfg_factory("rollout", overrides=["model=tiny", "rollout.samples=4", "rollout.M=2", "rollout.chunk_prompts=1"])
    rollout.run(cfg, engine=FakeEngine(AutoTokenizer.from_pretrained(cfg.model.path)))
    ecfg = cfg_factory("encode", overrides=["model=tiny", "encode.batch_tokens=64"], logged=False)
    encode.run(ecfg)
    out = rollout.paths.run_dir(cfg)
    index = pd.read_parquet(out / "index.parquet")
    for f, rows in index.groupby("states"):
        st = load_file(out / f)
        N, A, D = st["h_q"].shape
        assert st["h_v"].shape == (N, D) and N == len(rows) == rows.row.max() + 1 and A == 3
        # h_q[:, 0] (state at pos with the sampled token) is the next position after h_v of the same trace
        assert not (st["h_q"][:, 0] == st["h_v"]).all()
