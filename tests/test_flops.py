import torch

from shape_rl.flops import FlopModel


def test_flop_model_counts():
    fm = FlopModel("trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6")
    assert fm.F == 1 and fm.A == 4 * 256 and fm.P > 0
    L = torch.tensor([3, 5])
    assert fm.forward(L) == sum(2 * fm.P * x + 2 * fm.F * fm.A * x * (x + 1) for x in (3, 5))
    assert fm.forward(torch.tensor([0])) == 0
