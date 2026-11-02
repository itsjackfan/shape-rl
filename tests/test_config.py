import pytest
from hydra.utils import instantiate

from shape_rl.advantage import ZeroAdvantage

MODELS = ["qwen3.6-27b", "qwen3.6-35b-a3b", "tiny"]
OBJECTIVES = ["grpo", "dr_grpo", "dapo", "noop"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_train_config_composes(cfg_factory, model, objective):
    from trl import GRPOConfig

    cfg = cfg_factory(overrides=[f"model={model}", f"objective={objective}"])
    assert cfg.model.path
    GRPOConfig(output_dir="x", **cfg.objective.trl, **cfg.train)
    est = instantiate(cfg.objective.estimator)
    assert est is None or isinstance(est, ZeroAdvantage)


def test_eval_config_composes(cfg_factory):
    cfg = cfg_factory("eval")
    assert {b.suite for b in cfg.data.eval.values()} == {"math", "code", "reasoning"}
    assert all(b.task and b.metric for b in cfg.data.eval.values())
