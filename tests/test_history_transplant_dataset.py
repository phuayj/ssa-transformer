from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts" / "training", ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_history_ablation import HistoryAblationDataset


class TinyDataset:
    def __init__(self, examples):
        self.examples = list(examples)
        self.max_seq_len = 32
        self.vocab_size = 512

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def test_history_transplant_replaces_prior_history_only() -> None:
    source_seq = [101, 102, 11, 12, 21, 22, 31, 32, 33]
    source_lm = [False, False, False, False, False, False, True, True, False]
    source_blk = [0, 0, 1, 1, 2, 2, 3, 3, 3]
    donor_record = {
        "sequence": [101, 999, 111, 112, 121, 122, 77, 78],
        "loss_mask": [False, False, False, False, False, False, False, False],
        "block_ids": [0, 0, 1, 1, 2, 2, 3, 3],
    }
    dataset = HistoryAblationDataset(
        TinyDataset([(source_seq, source_lm, source_blk)]),
        history_mode="history_transplant",
        dropout_prob=0.0,
        window_size=1,
        placeholder_token=None,
        seed=7,
        transplant_prob=1.0,
        donor_pool=[donor_record],
        partial_transplant=False,
    )

    seq, loss_mask, block_ids = dataset[0]
    current_positions = [idx for idx, block_id in enumerate(block_ids) if block_id == 3]
    history_positions = [idx for idx, block_id in enumerate(block_ids) if block_id in (1, 2)]

    assert [seq[idx] for idx in current_positions] == [31, 32, 33]
    assert [loss_mask[idx] for idx in current_positions] == [True, True, False]
    assert any(seq[idx] != source_seq[idx] for idx in history_positions)
    assert all(not loss_mask[idx] for idx in history_positions)
