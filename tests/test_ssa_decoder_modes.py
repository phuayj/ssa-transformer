"""Fast forward-pass smoke tests for SSA decoder mask modes."""

from __future__ import annotations

import logging

import pytest
import torch

from universal.ssa_decoder import SSASlotDecoder


LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "mask_mode",
    [
        "selective_ssa",
        "full_causal",
        "blanket_ssa",
        "local_block_only",
        "swa_prefix",
    ],
)
def test_ssa_decoder_forward_pass_for_manuscript_mask_modes(mask_mode: str) -> None:
    torch.manual_seed(0)
    model = SSASlotDecoder(
        vocab_size=17,
        d_model=16,
        n_layers=1,
        n_heads=2,
        max_seq_len=8,
        n_slots=2,
        dropout=0.0,
    )
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    block_ids = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)

    with torch.no_grad():
        lm_logits, verify_logits = model(
            input_ids,
            attention_mask=attention_mask,
            block_ids=block_ids,
            mask_mode=mask_mode,
            window_size=3,
        )

    LOGGER.info(
        "SSA smoke forward mask_mode=%s lm_shape=%s verify_shape=%s lm_mean=%.6f",
        mask_mode,
        tuple(lm_logits.shape),
        tuple(verify_logits.shape),
        float(lm_logits.mean().item()),
    )
    assert tuple(lm_logits.shape) == (1, input_ids.size(1), 17)
    assert tuple(verify_logits.shape) == (1, input_ids.size(1), 2)
    assert torch.isfinite(lm_logits).all()
    assert torch.isfinite(verify_logits).all()
