from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from universal.ssa_decoder import SSASlotDecoder


def test_local_block_only_mask_blocks_prefix_and_slots() -> None:
    model = SSASlotDecoder(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=32,
        n_slots=2,
        dropout=0.0,
    )
    block_ids = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        ],
        dtype=torch.long,
    )
    padding_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    mask = model._build_ablation_mask(
        batch_size=2,
        seq_len=20,
        device=torch.device("cpu"),
        block_ids=block_ids,
        mask_mode="local_block_only",
        padding_mask=padding_mask,
    )[:, 0]

    slots = model.n_slots
    prefix_query = slots + 2
    block1_query = slots + 6
    block2_query = slots + 10

    assert not mask[0, block1_query, :slots].any()
    assert not mask[0, block1_query, slots : slots + 4].any()
    assert mask[0, block1_query, slots + 4]
    assert mask[0, block1_query, slots + 6]
    assert not mask[0, block1_query, slots + 7]
    assert not mask[0, block1_query, slots + 9]

    assert not mask[0, block2_query, :slots].any()
    assert not mask[0, block2_query, slots : slots + 4].any()
    assert mask[0, block2_query, slots + 8]
    assert mask[0, block2_query, slots + 10]
    assert not mask[0, block2_query, slots + 11]

    assert mask[0, 0, 1]
    assert mask[0, 0, slots + 0]
    assert mask[0, 1, slots + 3]

    assert mask[0, prefix_query, 0]
    assert mask[0, prefix_query, 1]
    assert mask[0, prefix_query, slots + 1]
    assert not mask[0, prefix_query, slots + 4]

    padded_key = slots + 18
    assert not mask[0, :, padded_key].any()
    padded_key_b1 = slots + 17
    assert not mask[1, :, padded_key_b1].any()
