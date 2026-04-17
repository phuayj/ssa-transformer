from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts" / "training", ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.interleaved_tokenizer import SATInterleavedTokenizer
from train_contrastive_invariance import (
    compute_contrastive_invariance_loss,
    compute_symmetric_action_kl,
)


def test_symmetric_action_kl_zero_and_positive() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)
    identical = compute_symmetric_action_kl(logits, logits)
    different = compute_symmetric_action_kl(
        logits,
        torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32),
    )

    assert torch.isclose(identical, torch.tensor(0.0), atol=1e-7)
    assert float(different.item()) > 0.0


def test_contrastive_invariance_loss_backprops_to_both_branches() -> None:
    vocab_size = int(SATInterleavedTokenizer.VOCAB_SIZE)
    input_ids = torch.tensor(
        [[1, SATInterleavedTokenizer.var_token(0), SATInterleavedTokenizer.TRUE_VAL, 2]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[False, True, True, False]], dtype=torch.bool)
    block_ids = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    source_logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32, requires_grad=True)
    pair_logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32, requires_grad=True)
    source_logits.data[0, 0, SATInterleavedTokenizer.var_token(0)] = 2.0
    source_logits.data[0, 1, SATInterleavedTokenizer.TRUE_VAL] = 1.5
    pair_logits.data[0, 0, SATInterleavedTokenizer.var_token(0)] = 0.5
    pair_logits.data[0, 1, SATInterleavedTokenizer.TRUE_VAL] = 2.5

    loss, metrics = compute_contrastive_invariance_loss(
        source_logits=source_logits,
        pair_logits=pair_logits,
        source_input_ids=input_ids,
        pair_input_ids=input_ids,
        source_attention_mask=attention_mask,
        pair_attention_mask=attention_mask,
        source_loss_mask=loss_mask,
        pair_loss_mask=loss_mask,
        source_block_ids=block_ids,
        pair_block_ids=block_ids,
        lambda_kl=1.0,
    )
    loss.backward()

    assert metrics["token_count"] == 2
    assert float(metrics["loss_kl"]) > 0.0
    assert source_logits.grad is not None
    assert pair_logits.grad is not None
    assert float(source_logits.grad.abs().sum().item()) > 0.0
    assert float(pair_logits.grad.abs().sum().item()) > 0.0


def test_contrastive_invariance_loss_returns_zero_for_empty_action_batch() -> None:
    vocab_size = int(SATInterleavedTokenizer.VOCAB_SIZE)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[False, True, True, False]], dtype=torch.bool)
    block_ids = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    source_logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32, requires_grad=True)
    pair_logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32, requires_grad=True)

    loss, metrics = compute_contrastive_invariance_loss(
        source_logits=source_logits,
        pair_logits=pair_logits,
        source_input_ids=input_ids,
        pair_input_ids=input_ids,
        source_attention_mask=attention_mask,
        pair_attention_mask=attention_mask,
        source_loss_mask=loss_mask,
        pair_loss_mask=loss_mask,
        source_block_ids=block_ids,
        pair_block_ids=block_ids,
        lambda_kl=1.0,
    )
    loss.backward()

    assert loss.requires_grad
    assert float(loss.item()) == 0.0
    assert metrics["num_action_tokens"] == 0
    assert metrics["token_count"] == 0
    assert float(metrics["ce_source"]) == 0.0
    assert float(metrics["ce_pair"]) == 0.0
    assert float(metrics["kl"]) == 0.0
    assert source_logits.grad is None
    assert pair_logits.grad is None
