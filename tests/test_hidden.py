import pytest
import torch

from shape_rl import hidden

TINY = "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6"


@pytest.mark.slow
def test_states_at_matches_output_hidden_states():
    model = hidden.load_model(TINY, dtype=torch.float32)
    ids_BL = torch.randint(0, 100, (2, 12))
    mask_BL = torch.ones_like(ids_BL)
    mask_BL[1, 9:] = 0  # right padding
    pos_BP = torch.tensor([[0, 5, 11], [2, 4, 8]])
    got_BPD = hidden.states_at(model, ids_BL, mask_BL, pos_BP)
    ref_BLD = model(input_ids=ids_BL, attention_mask=mask_BL, output_hidden_states=True).hidden_states[-1]
    assert got_BPD.shape == (2, 3, ref_BLD.shape[-1])
    torch.testing.assert_close(got_BPD, torch.gather(ref_BLD, 1, pos_BP[..., None].expand(-1, -1, ref_BLD.shape[-1])))
