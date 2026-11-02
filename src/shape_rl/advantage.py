"""N = num completions in the global batch (num_prompts * G), G = generations per prompt."""
import torch
from torch import nn


class AdvantageEstimator(nn.Module):
    """Base class: one subclass per advantage ablation. Overrides TRL's built-in group-relative advantage."""

    def forward(self, rewards_N: torch.Tensor, G: int) -> torch.Tensor:
        raise NotImplementedError


class ZeroAdvantage(AdvantageEstimator):
    def forward(self, rewards_N, G):
        return torch.zeros_like(rewards_N)
