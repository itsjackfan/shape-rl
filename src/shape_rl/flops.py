"""Model FLOPs of the training process (the compute-matching axis): generation, scoring forwards, training forward + backward
and the advantage estimator. Dataset building (rollout.py, encode.py, ssl.py) is not counted.

Per sequence of L real tokens (padding not counted), one forward pass costs
  2 * P * L                     P = parameters in matmuls per token: decoder layers (routed experts scaled by top-k / E) and
                                the LM head (counted once, also when tied); token embeddings (a lookup) and the vision tower
                                are excluded;
  + 2 * F * A * L * (L + 1)     F = full-attention layers, A = heads * head_dim: Q K^T and attention-weighted V, causal.
Generation of a completion = one forward over prompt + completion (KV cache). Training = 3 forwards (forward + backward) per
optimizer pass; the gradient-checkpointing recompute is not counted (model FLOPs, not hardware FLOPs). Not counted:
linear-attention state updates (< 1 % for Qwen3.5-4B), norms, activations, softmax."""
import torch
import transformers
from transformers import AutoConfig


class FlopModel:
    def __init__(self, model_path: str):
        config = AutoConfig.from_pretrained(model_path)
        with torch.device("meta"):
            model = getattr(transformers, config.architectures[0])._from_config(config)
        tc = config.get_text_config()
        frac = (tc.num_experts_per_tok / tc.num_experts) if getattr(tc, "num_experts", None) else 1.0
        P = 0
        for name, p in model.named_parameters():
            if "visual" in name or "embed_tokens" in name:
                continue
            P += p.numel() * (frac if ".experts." in name and "shared" not in name else 1.0)
        if not any(n.startswith("lm_head") for n, _ in model.named_parameters()):  # tied: the head matmul still runs
            P += tc.vocab_size * tc.hidden_size
        types = getattr(tc, "layer_types", None) or ["full_attention"] * tc.num_hidden_layers
        self.P, self.F = float(P), sum(t == "full_attention" for t in types)
        self.A = tc.num_attention_heads * (getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads)

    def forward(self, lengths: torch.Tensor) -> float:
        """FLOPs of one forward pass over sequences with these real lengths."""
        L = lengths.double()
        return float((2 * self.P * L + 2 * self.F * self.A * L * (L + 1)).sum())
