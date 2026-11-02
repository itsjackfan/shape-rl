import torch

from shape_rl.advantage import AdvantageEstimator, ZeroAdvantage
from shape_rl.rewards import correctness


def test_correctness():
    completions = [[{"role": "assistant", "content": "so \\boxed{\\frac{1}{2}}"}], [{"role": "assistant", "content": "\\boxed{3}"}]]
    assert correctness(completions, answer=["0.5", "4"]) == [1.0, 0.0]


def test_estimator_shapes():
    rewards_N = torch.rand(12)
    adv_N = ZeroAdvantage()(rewards_N, G=4)
    assert isinstance(ZeroAdvantage(), AdvantageEstimator)
    assert adv_N.shape == rewards_N.shape and (adv_N == 0).all()
