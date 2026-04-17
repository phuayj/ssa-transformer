"""Dataset wrappers for the r^k benchmark."""

from __future__ import annotations

import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .generator import (
    PAD_TOKEN,
    RkBenchmarkConfig,
    extract_oracle_features_from_tokens,
    generate_example,
)


class RkBenchmarkDataset(Dataset):
    """Online-generated dataset for r^k benchmark (pre-generated for reproducibility)."""

    def __init__(self, config: RkBenchmarkConfig, size: int, seed: int):
        self.config = config
        self.size = int(size)
        self.seed = int(seed)
        self.rng = random.Random(int(seed))
        self.examples = [generate_example(config, self.rng) for _ in range(int(size))]

    def __len__(self) -> int:
        return int(self.size)

    def _extract_oracle_features(self, idx: int) -> List[int]:
        tokens, _label = self.examples[int(idx)]
        features = extract_oracle_features_from_tokens(
            tokens=tokens,
            k=int(self.config.k),
            key_len=int(self.config.key_len),
        )
        return [int(x) for x in features]

    def __getitem__(self, idx: int) -> Any:
        tokens, label = self.examples[int(idx)]
        oracle_features = self._extract_oracle_features(idx)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(float(label), dtype=torch.float),
            "oracle_features": torch.tensor(oracle_features, dtype=torch.float),
        }


def rk_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length token sequences and oracle feature vectors."""
    if len(batch) == 0:
        raise ValueError("Empty batch in rk_collate_fn")

    max_seq = max(int(item["input_ids"].shape[0]) for item in batch)
    max_k = max(int(item["oracle_features"].shape[0]) for item in batch)

    padded_inputs = torch.full(
        (len(batch), max_seq),
        fill_value=int(PAD_TOKEN),
        dtype=torch.long,
    )
    padding_mask = torch.ones((len(batch), max_seq), dtype=torch.bool)
    oracle = torch.full((len(batch), max_k), fill_value=-1.0, dtype=torch.float)
    labels = torch.zeros((len(batch),), dtype=torch.float)

    for i, item in enumerate(batch):
        seq = item["input_ids"]
        feat = item["oracle_features"]
        seq_len = int(seq.shape[0])
        feat_len = int(feat.shape[0])

        padded_inputs[i, :seq_len] = seq
        padding_mask[i, :seq_len] = False
        oracle[i, :feat_len] = feat
        labels[i] = item["label"]

    return {
        "input_ids": padded_inputs,
        "padding_mask": padding_mask,
        "label": labels,
        "oracle_features": oracle,
    }
