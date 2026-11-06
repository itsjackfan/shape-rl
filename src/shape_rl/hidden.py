"""Last-layer hidden states. B = batch, L = sequence length, D = hidden size, P = requested positions per row."""
import torch
import transformers
from transformers import AutoConfig


def load_model(path, dtype=torch.bfloat16, **kw):
    """The checkpoint's own architecture (e.g. Qwen3_5ForConditionalGeneration), eval mode."""
    cls = getattr(transformers, AutoConfig.from_pretrained(path).architectures[0])
    return cls.from_pretrained(path, dtype=dtype, **kw).eval()


def final_norm(model) -> torch.nn.Module:
    """The final norm of the text decoder: its output is the post-norm last-layer state (the lm_head input)."""
    return model.get_decoder().norm


@torch.no_grad()
def states_at(model, input_ids_BL, attention_mask_BL, pos_BP) -> torch.Tensor:
    """Post-norm last-layer states at pos_BP -> [B, P, D], captured by a forward hook on the final norm
    (one layer only; output_hidden_states=True would return every layer). logits_to_keep=1 skips full logits."""
    cap = {}
    handle = final_norm(model).register_forward_hook(lambda m, i, o: cap.__setitem__("x_BLD", o))
    try:
        model(input_ids=input_ids_BL, attention_mask=attention_mask_BL, logits_to_keep=1, use_cache=False)
    finally:
        handle.remove()
    x_BLD = cap["x_BLD"]
    return torch.gather(x_BLD, 1, pos_BP[..., None].expand(-1, -1, x_BLD.shape[-1]))
