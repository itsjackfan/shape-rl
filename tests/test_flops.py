import pytest
import torch

from shape_rl.flops import FlopModel

TINY = "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6"


def test_flop_model_counts():
    fm = FlopModel(TINY)
    assert fm.F == 1 and fm.A == 4 * 256 and fm.P > 0
    L = torch.tensor([3, 5])
    assert fm.forward(L) == sum(2 * fm.P * x + 2 * fm.F * fm.A * x * (x + 1) for x in (3, 5))
    assert fm.forward(torch.tensor([0])) == 0


def test_flop_model_under_zero3():
    pytest.importorskip("deepspeed")
    from transformers.integrations import HfDeepSpeedConfig

    plain = FlopModel(TINY).P
    keep = HfDeepSpeedConfig({"zero_optimization": {"stage": 3}, "train_batch_size": 1})  # noqa: F841 (a live weakref enables ZeRO-3)
    from transformers.integrations import is_deepspeed_zero3_enabled

    assert is_deepspeed_zero3_enabled() and FlopModel(TINY).P == plain and is_deepspeed_zero3_enabled()
    del keep  # end ZeRO-3 for the other tests
