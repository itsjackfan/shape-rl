import pytest
from hydra.utils import instantiate

from shape_rl.advantage import ZeroAdvantage

MODELS = ["qwen3.5-4b", "qwen3.6-27b", "qwen3.6-35b-a3b", "tiny"]
OBJECTIVES = ["grpo", "dr_grpo", "dapo", "noop", "hvalue", "hcritic", "hcluster"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_train_config_composes(cfg_factory, model, objective):
    from trl import GRPOConfig

    from shape_rl.train import grpo_kwargs

    cfg = cfg_factory(overrides=[f"model={model}", f"objective={objective}"])
    assert cfg.model.path and cfg.model.chat_template_kwargs.enable_thinking is False
    args = GRPOConfig(output_dir="x", **{**grpo_kwargs(cfg), "bf16": False})  # bf16 needs a GPU; the rest as train.py builds it
    args.to_json_string()  # plain containers only (W&B / checkpoint serialization)
    assert args.max_completion_length == cfg.model.max_new_tokens and args.chat_template_kwargs == {"enable_thinking": False}
    assert args.vllm_max_model_length == cfg.max_prompt_len + cfg.model.max_new_tokens
    if objective in ("grpo", "dr_grpo", "dapo", "noop"):
        est = instantiate(cfg.objective.estimator)
        assert est is None or isinstance(est, ZeroAdvantage)


def test_eval_config_composes(cfg_factory):
    cfg = cfg_factory("eval")
    assert {b.suite for b in cfg.data.eval.values()} == {"math", "code", "reasoning"}
    assert all(b.task and b.metric for b in cfg.data.eval.values())
    assert "enable_thinking=False" in cfg.eval.model_args and cfg.eval.gen_kwargs == f"max_gen_toks={cfg.model.max_new_tokens}"
