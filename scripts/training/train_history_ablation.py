#!/usr/bin/env python3
"""Train SAT SSA models with history-removal ablations.

Extends ``train_gc_mask_ablation.py`` with training-only history transforms:

- ``full``: no history change
- ``block_dropout``: randomly replace intermediate search blocks with PAD spans
- ``sliding_window``: keep only the last N search blocks, PAD earlier blocks
- ``null_history``: replace intermediate search blocks with a learned placeholder
- ``prefix_only``: keep graph prefix + final search block only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.interleaved_tokenizer import SATInterleavedTokenizer
from sat.length_ood_common import block_truncation_cutoff, compute_length_stats
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


HISTORY_MODES = (
    "full",
    "block_dropout",
    "sliding_window",
    "null_history",
    "prefix_only",
    "history_transplant",
)
POSITION_MODES = ("auto", "standard", "block_relative")


@dataclass(frozen=True)
class BlockSpan:
    block_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return int(self.end - self.start)


def _safe_decode(tokenizer: SATInterleavedTokenizer, tok_id: int) -> str:
    """Decode token to string, falling back to raw ID if out of range."""
    try:
        return tokenizer.decode_token(int(tok_id))
    except (ValueError, KeyError, IndexError):
        return f"<{int(tok_id)}>"


def _get_state_tried_tokens(vocab_size: int) -> Tuple[int, int]:
    """Infer STATE/TRIED marker tokens from vocabulary size."""
    if int(vocab_size) <= 394:
        mask_offset = 240 + 30 * 4
    elif int(vocab_size) <= 574:
        mask_offset = 240 + 75 * 4
    elif int(vocab_size) <= 2030:
        return int(SATInterleavedTokenizer.STATE), -1
    else:
        raise ValueError(f"Unsupported vocab_size={vocab_size}")

    state_tok = int(mask_offset + 16)
    tried_tok = int(mask_offset + 32)
    return state_tok, tried_tok


def compute_block_ids_for_vocab(sequence: Sequence[int], vocab_size: int) -> List[int]:
    """Compute SSA block IDs from tokens.

    block_id=0 for the static prefix, then 1..K for search blocks.
    """
    state_tok, tried_tok = _get_state_tried_tokens(int(vocab_size))

    block_ids: List[int] = []
    current_block = 0
    in_tried_section = False

    for raw_token in sequence:
        token = int(raw_token)
        if current_block == 0:
            if token == tried_tok:
                current_block = 1
                in_tried_section = True
            elif token == state_tok:
                current_block = 1
                in_tried_section = False
        else:
            if token == tried_tok:
                current_block += 1
                in_tried_section = True
            elif token == state_tok and not in_tried_section:
                current_block += 1
            elif token == state_tok and in_tried_section:
                in_tried_section = False

        block_ids.append(int(current_block))

    return block_ids


class SSATriedDataset(Dataset[Tuple[List[int], List[bool], List[int]]]):
    """Base dataset returning sequence, loss mask, and block IDs."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        max_seq_len: int,
        vocab_size: int,
    ):
        self.records = list(records)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[bool], List[int]]:
        item = self.records[int(idx)]
        seq = [int(x) for x in item["sequence"]][: self.max_seq_len]
        loss_mask = [bool(x) for x in item["loss_mask"]][: self.max_seq_len]
        if len(seq) != len(loss_mask):
            raise ValueError("sequence/loss_mask mismatch")

        if "block_ids" in item and item["block_ids"] is not None:
            block_ids = [int(x) for x in item["block_ids"]][: self.max_seq_len]
        else:
            block_ids = compute_block_ids_for_vocab(seq, vocab_size=self.vocab_size)
        if len(block_ids) != len(seq):
            raise ValueError("sequence/block_ids mismatch")
        return seq, loss_mask, block_ids


class BlockTruncatedDataset(Dataset[Tuple[List[int], List[bool], List[int]]]):
    """Training-only wrapper that truncates examples to the first K search blocks."""

    def __init__(
        self,
        base_dataset: Dataset[Tuple[List[int], List[bool], List[int]]],
        *,
        max_train_blocks: int,
    ):
        self.base_dataset = base_dataset
        self.max_train_blocks = int(max_train_blocks)
        if self.max_train_blocks <= 0:
            raise ValueError("max_train_blocks must be positive")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[bool], List[int]]:
        sequence, loss_mask, block_ids = self.base_dataset[int(idx)]
        cutoff = block_truncation_cutoff(block_ids, int(self.max_train_blocks))
        return (
            [int(x) for x in sequence[:cutoff]],
            [bool(x) for x in loss_mask[:cutoff]],
            [int(x) for x in block_ids[:cutoff]],
        )


def _compute_block_spans(block_ids: Sequence[int]) -> List[BlockSpan]:
    """Convert per-token block IDs into contiguous spans."""
    if len(block_ids) == 0:
        return []

    spans: List[BlockSpan] = []
    start = 0
    current = int(block_ids[0])
    for idx in range(1, len(block_ids)):
        block_id = int(block_ids[idx])
        if block_id != current:
            spans.append(
                BlockSpan(block_id=int(current), start=int(start), end=int(idx))
            )
            start = idx
            current = block_id
    spans.append(
        BlockSpan(block_id=int(current), start=int(start), end=int(len(block_ids)))
    )
    return spans


def _clone_example(
    sequence: Sequence[int],
    loss_mask: Sequence[bool],
    block_ids: Sequence[int],
) -> Tuple[List[int], List[bool], List[int]]:
    return (
        [int(x) for x in sequence],
        [bool(x) for x in loss_mask],
        [int(x) for x in block_ids],
    )


PreparedDonor = Dict[str, Any]


def _unwrap_base_dataset(dataset: Dataset[Tuple[List[int], List[bool], List[int]]]) -> Any:
    current: Any = dataset
    while hasattr(current, "base_dataset"):
        current = getattr(current, "base_dataset")
    return current


def _resolve_dataset_metadata(
    dataset: Dataset[Tuple[List[int], List[bool], List[int]]],
) -> Tuple[int, int]:
    base = _unwrap_base_dataset(dataset)
    if not hasattr(base, "max_seq_len") or not hasattr(base, "vocab_size"):
        raise ValueError("history transplant requires dataset max_seq_len and vocab_size")
    return int(base.max_seq_len), int(base.vocab_size)


def _record_to_example(
    record: Dict[str, Any],
    *,
    max_seq_len: int,
    vocab_size: int,
) -> Tuple[List[int], List[bool], List[int]]:
    seq = [int(x) for x in record["sequence"]][: int(max_seq_len)]
    loss_mask = [bool(x) for x in record["loss_mask"]][: int(max_seq_len)]
    if len(seq) != len(loss_mask):
        raise ValueError("sequence/loss_mask mismatch")
    if "block_ids" in record and record["block_ids"] is not None:
        block_ids = [int(x) for x in record["block_ids"]][: int(max_seq_len)]
    else:
        block_ids = compute_block_ids_for_vocab(seq, vocab_size=int(vocab_size))
    if len(block_ids) != len(seq):
        raise ValueError("sequence/block_ids mismatch")
    return seq, loss_mask, block_ids


def prepare_history_transplant_donor_pool(
    donor_pool: Sequence[Dict[str, Any]],
    *,
    max_seq_len: int,
    vocab_size: int,
) -> List[PreparedDonor]:
    prepared: List[PreparedDonor] = []
    for record in donor_pool:
        seq, loss_mask, block_ids = _record_to_example(
            dict(record),
            max_seq_len=int(max_seq_len),
            vocab_size=int(vocab_size),
        )
        spans = _compute_block_spans(block_ids)
        search_spans = [span for span in spans if int(span.block_id) > 0]
        prepared.append(
            {
                "sequence": seq,
                "loss_mask": loss_mask,
                "block_ids": block_ids,
                "spans": spans,
                "search_spans": search_spans,
            }
        )
    return prepared


def _find_current_search_span(
    loss_mask: Sequence[bool],
    search_spans: Sequence[BlockSpan],
) -> Tuple[int, bool]:
    last_search_idx = len(search_spans) - 1
    found_supervised_search = False
    for idx in range(len(search_spans) - 1, -1, -1):
        span = search_spans[idx]
        if any(bool(loss_mask[pos]) for pos in range(int(span.start), int(span.end))):
            last_search_idx = idx
            found_supervised_search = True
            break
    return int(last_search_idx), bool(found_supervised_search)


def build_history_transplant_example(
    sequence: Sequence[int],
    loss_mask: Sequence[bool],
    block_ids: Sequence[int],
    *,
    donor_pool: Sequence[PreparedDonor],
    rng: random.Random,
    transplant_prob: float = 1.0,
    partial_transplant: bool = False,
    max_seq_len: int | None = None,
) -> Tuple[List[int], List[bool], List[int]]:
    seq, lm, blk = _clone_example(sequence, loss_mask, block_ids)
    if len(seq) == 0 or len(donor_pool) == 0:
        return seq, lm, blk

    spans = _compute_block_spans(blk)
    search_spans = [span for span in spans if int(span.block_id) > 0]
    if len(search_spans) <= 1:
        return seq, lm, blk

    last_search_idx, _found_supervised_search = _find_current_search_span(lm, search_spans)
    history_spans = list(search_spans[:last_search_idx])
    if len(history_spans) == 0:
        return seq, lm, blk

    current_span = search_spans[last_search_idx]
    prefix_seq = [int(seq[pos]) for pos in range(0, int(current_span.start)) if int(blk[pos]) == 0]
    prefix_lm = [bool(lm[pos]) for pos in range(0, int(current_span.start)) if int(blk[pos]) == 0]
    prefix_blk = [int(blk[pos]) for pos in range(0, int(current_span.start)) if int(blk[pos]) == 0]
    suffix_seq = [int(seq[pos]) for pos in range(int(current_span.start), len(seq))]
    suffix_lm = [bool(lm[pos]) for pos in range(int(current_span.start), len(seq))]
    suffix_blk = [int(blk[pos]) for pos in range(int(current_span.start), len(seq))]

    source_history_tokens = [
        tuple(int(seq[pos]) for pos in range(int(span.start), int(span.end)))
        for span in history_spans
    ]

    def _candidate_ok(donor: PreparedDonor, donor_indices: Sequence[int]) -> bool:
        projected_len = int(len(prefix_seq) + len(suffix_seq))
        for donor_idx in donor_indices:
            donor_span = donor["search_spans"][int(donor_idx)]
            projected_len += int(donor_span.length)
        return max_seq_len is None or projected_len <= int(max_seq_len)

    if not bool(partial_transplant):
        if float(transplant_prob) < 1.0 and rng.random() >= float(transplant_prob):
            return seq, lm, blk
        candidate_indices: List[int] = []
        source_matches: List[int] = []
        needed_blocks = int(len(history_spans))
        for donor_idx, donor in enumerate(donor_pool):
            donor_spans = donor["search_spans"]
            if len(donor_spans) < needed_blocks:
                continue
            chosen = list(range(needed_blocks))
            if not _candidate_ok(donor, chosen):
                continue
            donor_tokens = [
                tuple(
                    int(donor["sequence"][pos])
                    for pos in range(int(span.start), int(span.end))
                )
                for span in donor_spans[:needed_blocks]
            ]
            if donor_tokens == source_history_tokens:
                source_matches.append(int(donor_idx))
            else:
                candidate_indices.append(int(donor_idx))
        if len(candidate_indices) == 0:
            candidate_indices = source_matches
        if len(candidate_indices) == 0:
            return seq, lm, blk
        donor = donor_pool[int(rng.choice(candidate_indices))]
        new_seq = list(prefix_seq)
        new_lm = list(prefix_lm)
        new_blk = list(prefix_blk)
        for source_span, donor_span in zip(history_spans, donor["search_spans"][:needed_blocks]):
            donor_tokens = [
                int(donor["sequence"][pos])
                for pos in range(int(donor_span.start), int(donor_span.end))
            ]
            new_seq.extend(donor_tokens)
            new_lm.extend([False] * len(donor_tokens))
            new_blk.extend([int(source_span.block_id)] * len(donor_tokens))
        new_seq.extend(suffix_seq)
        new_lm.extend(suffix_lm)
        new_blk.extend(suffix_blk)
        return new_seq, new_lm, new_blk

    donor_choices: List[Tuple[List[int], List[bool], List[int]]] = []
    replaced_any = False
    projected_len = int(len(prefix_seq) + len(suffix_seq))
    for history_idx, source_span in enumerate(history_spans):
        keep_source = True
        span_seq = [int(seq[pos]) for pos in range(int(source_span.start), int(source_span.end))]
        span_lm = [bool(lm[pos]) for pos in range(int(source_span.start), int(source_span.end))]
        span_blk = [int(blk[pos]) for pos in range(int(source_span.start), int(source_span.end))]
        if rng.random() < float(transplant_prob):
            eligible = [
                donor
                for donor in donor_pool
                if len(donor["search_spans"]) > int(history_idx)
                and _candidate_ok(donor, [int(history_idx)])
            ]
            if len(eligible) > 0:
                donor = eligible[int(rng.randrange(len(eligible)))]
                donor_span = donor["search_spans"][int(history_idx)]
                donor_tokens = [
                    int(donor["sequence"][pos])
                    for pos in range(int(donor_span.start), int(donor_span.end))
                ]
                span_seq = donor_tokens
                span_lm = [False] * len(donor_tokens)
                span_blk = [int(source_span.block_id)] * len(donor_tokens)
                keep_source = False
        projected_len += len(span_seq)
        donor_choices.append((span_seq, span_lm, span_blk))
        replaced_any = replaced_any or (not keep_source)
    if not replaced_any or (max_seq_len is not None and projected_len > int(max_seq_len)):
        return seq, lm, blk

    new_seq = list(prefix_seq)
    new_lm = list(prefix_lm)
    new_blk = list(prefix_blk)
    for span_seq, span_lm, span_blk in donor_choices:
        new_seq.extend(span_seq)
        new_lm.extend(span_lm)
        new_blk.extend(span_blk)
    new_seq.extend(suffix_seq)
    new_lm.extend(suffix_lm)
    new_blk.extend(suffix_blk)
    return new_seq, new_lm, new_blk


class HistoryAblationDataset(Dataset[Tuple[List[int], List[bool], List[int]]]):
    """Training-only wrapper that removes or corrupts search history."""

    def __init__(
        self,
        base_dataset: Dataset[Tuple[List[int], List[bool], List[int]]],
        *,
        history_mode: str,
        dropout_prob: float,
        window_size: int,
        placeholder_token: int | None,
        seed: int,
        transplant_prob: float = 1.0,
        donor_pool: List[Dict[str, Any]] | None = None,
        partial_transplant: bool = False,
    ):
        self.base_dataset = base_dataset
        self.history_mode = str(history_mode)
        self.dropout_prob = float(dropout_prob)
        self.window_size = int(window_size)
        self.placeholder_token = (
            int(placeholder_token) if placeholder_token is not None else None
        )
        self.transplant_prob = float(transplant_prob)
        self.partial_transplant = bool(partial_transplant)
        self.max_seq_len, self.vocab_size = _resolve_dataset_metadata(base_dataset)
        raw_donor_pool = list(donor_pool) if donor_pool is not None else []
        self.donor_pool = prepare_history_transplant_donor_pool(
            raw_donor_pool,
            max_seq_len=int(self.max_seq_len),
            vocab_size=int(self.vocab_size),
        )
        self.rng = random.Random(int(seed))

        if self.history_mode not in HISTORY_MODES:
            raise ValueError(
                f"unknown history_mode='{self.history_mode}', expected one of {sorted(HISTORY_MODES)}"
            )
        if not (0.0 <= self.dropout_prob <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1]")
        if not (0.0 <= self.transplant_prob <= 1.0):
            raise ValueError("transplant_prob must be in [0, 1]")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.history_mode == "null_history" and self.placeholder_token is None:
            raise ValueError("null_history requires placeholder_token")
        if self.history_mode == "history_transplant" and len(self.donor_pool) == 0:
            raise ValueError("history_transplant requires donor_pool")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[bool], List[int]]:
        sequence, loss_mask, block_ids = self.base_dataset[int(idx)]
        return self._apply_history_ablation(sequence, loss_mask, block_ids)

    def _pad_replace_spans(
        self,
        sequence: List[int],
        loss_mask: List[bool],
        spans: Sequence[BlockSpan],
        fill_token: int,
    ) -> Tuple[List[int], List[bool]]:
        for span in spans:
            for pos in range(int(span.start), int(span.end)):
                sequence[pos] = int(fill_token)
                loss_mask[pos] = False
        return sequence, loss_mask

    def _apply_history_ablation(
        self,
        sequence: Sequence[int],
        loss_mask: Sequence[bool],
        block_ids: Sequence[int],
    ) -> Tuple[List[int], List[bool], List[int]]:
        seq, lm, blk = _clone_example(sequence, loss_mask, block_ids)
        if self.history_mode == "full" or len(seq) == 0:
            return seq, lm, blk

        spans = _compute_block_spans(blk)
        search_spans = [span for span in spans if int(span.block_id) > 0]
        if len(search_spans) <= 1:
            return seq, lm, blk

        last_search_idx, found_supervised_search = _find_current_search_span(lm, search_spans)
        last_search = search_spans[last_search_idx]
        intermediate = [
            span for idx, span in enumerate(search_spans) if idx != last_search_idx
        ]

        if self.history_mode == "history_transplant":
            return build_history_transplant_example(
                seq,
                lm,
                blk,
                donor_pool=self.donor_pool,
                rng=self.rng,
                transplant_prob=float(self.transplant_prob),
                partial_transplant=bool(self.partial_transplant),
                max_seq_len=int(self.max_seq_len),
            )

        if self.history_mode == "block_dropout":
            dropped = [
                span
                for span in intermediate
                if self.rng.random() < float(self.dropout_prob)
            ]
            seq, lm = self._pad_replace_spans(
                seq, lm, dropped, fill_token=int(SATInterleavedTokenizer.PAD)
            )
            return seq, lm, blk

        if self.history_mode == "sliding_window":
            keep_n = max(1, int(self.window_size))
            keep_start = max(0, last_search_idx - keep_n + 1)
            keep_block_ids = {
                int(span.block_id)
                for span in search_spans[keep_start : last_search_idx + 1]
            }
            dropped = [
                span
                for span in search_spans
                if int(span.block_id) not in keep_block_ids
            ]
            seq, lm = self._pad_replace_spans(
                seq, lm, dropped, fill_token=int(SATInterleavedTokenizer.PAD)
            )
            return seq, lm, blk

        if self.history_mode == "null_history":
            seq, lm = self._pad_replace_spans(
                seq,
                lm,
                intermediate,
                fill_token=int(self.placeholder_token),
            )
            return seq, lm, blk

        if self.history_mode == "prefix_only":
            if not found_supervised_search:
                return seq, lm, blk
            keep_block_ids = {0, int(last_search.block_id)}
            kept_positions = [
                idx
                for idx, block_id in enumerate(blk)
                if int(block_id) in keep_block_ids
            ]
            seq = [int(seq[idx]) for idx in kept_positions]
            lm = [bool(lm[idx]) for idx in kept_positions]
            blk = [int(blk[idx]) for idx in kept_positions]
            return seq, lm, blk

        raise RuntimeError(f"unhandled history_mode={self.history_mode}")


def _collate(batch: Sequence[Tuple[List[int], List[bool], List[int]]]):
    bsz = len(batch)
    max_len = max(len(item[0]) for item in batch)
    input_ids = torch.full(
        (bsz, max_len), int(SATInterleavedTokenizer.PAD), dtype=torch.long
    )
    loss_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    block_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, (seq, lm, blk) in enumerate(batch):
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        loss_mask[i, :seq_len] = torch.tensor(lm, dtype=torch.bool)
        attention_mask[i, :seq_len] = 1
        block_ids[i, :seq_len] = torch.tensor(blk, dtype=torch.long)
    return input_ids, attention_mask, loss_mask, block_ids


def _compute_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    labels = input_ids[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    supervised = loss_mask[:, 1:] & (attention_mask[:, 1:] > 0)
    token_count = int(supervised.sum().item())
    if token_count == 0:
        raise RuntimeError("no supervised tokens in batch")
    labels = labels.masked_fill(~supervised, int(SATInterleavedTokenizer.PAD))
    loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=int(SATInterleavedTokenizer.PAD),
    )
    return loss, token_count, supervised, labels


def _count_effective_blocks(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    block_ids: torch.Tensor,
    placeholder_token: int | None,
) -> float:
    total = 0.0
    batch_size = int(input_ids.size(0))
    content_mask = (attention_mask > 0) & input_ids.ne(int(SATInterleavedTokenizer.PAD))
    if placeholder_token is not None:
        content_mask = content_mask & input_ids.ne(int(placeholder_token))

    for row_idx in range(batch_size):
        row_mask = content_mask[row_idx]
        if bool(row_mask.any().item()):
            total += float(torch.unique(block_ids[row_idx, row_mask]).numel())
    return total


def _resolve_effective_position_mode(
    *,
    position_mode: str,
    has_block_ids: bool,
) -> str:
    resolved = str(position_mode)
    if resolved not in POSITION_MODES:
        raise ValueError(
            f"unknown position_mode='{resolved}', expected one of {sorted(POSITION_MODES)}"
        )
    if resolved == "auto":
        return "block_relative" if bool(has_block_ids) else "standard"
    if resolved == "block_relative" and not bool(has_block_ids):
        raise ValueError("position_mode='block_relative' requires block_ids")
    return resolved


def _run_epoch(
    *,
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    train: bool,
    device: torch.device,
    tokenizer: SATInterleavedTokenizer,
    mask_mode: str,
    position_mode: str,
    history_mode: str,
    placeholder_token: int | None,
) -> Dict[str, float]:
    if train:
        model.train()
        if optimizer is None:
            raise ValueError("optimizer is required for training")
        opt = optimizer
    else:
        model.eval()
        opt = None

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_seq = 0
    total_block_id_span = 0.0
    total_effective_blocks = 0.0
    total_seq_tokens = 0.0
    total_content_tokens = 0.0
    total_placeholder_tokens = 0.0
    total_supervised_tokens = 0.0
    logged = False

    for input_ids, attention_mask, loss_mask, block_ids in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device)
        block_ids = block_ids.to(device)

        with torch.set_grad_enabled(train):
            lm_logits, _verify_logits = model(
                input_ids,
                attention_mask,
                block_ids=block_ids,
                mask_mode=mask_mode,
                position_mode=position_mode,
            )

            lm_loss, token_count, supervised, labels = _compute_lm_loss(
                logits=lm_logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            if train:
                assert opt is not None
                opt.zero_grad()
                lm_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                if scheduler is not None:
                    scheduler.step()

        preds = lm_logits[:, :-1, :].argmax(dim=-1)
        total_correct += int(((preds == labels) & supervised).sum().item())
        total_tokens += int(token_count)
        batch_size = int(input_ids.size(0))
        total_seq += int(batch_size)
        total_loss += float(lm_loss.item()) * float(batch_size)

        valid_mask = attention_mask > 0
        content_mask = valid_mask & input_ids.ne(int(SATInterleavedTokenizer.PAD))
        if placeholder_token is not None:
            placeholder_mask = valid_mask & input_ids.eq(int(placeholder_token))
            content_mask = content_mask & input_ids.ne(int(placeholder_token))
            total_placeholder_tokens += float(placeholder_mask.sum().item())
        total_seq_tokens += float(valid_mask.sum().item())
        total_content_tokens += float(content_mask.sum().item())
        total_supervised_tokens += float((loss_mask & valid_mask).sum().item())
        total_effective_blocks += _count_effective_blocks(
            input_ids=input_ids,
            attention_mask=attention_mask,
            block_ids=block_ids,
            placeholder_token=placeholder_token,
        )

        for row_idx in range(batch_size):
            row_valid = valid_mask[row_idx]
            if bool(row_valid.any().item()):
                total_block_id_span += float(block_ids[row_idx, row_valid].max().item())

        if not logged:
            row = 0
            valid_pos = torch.nonzero(supervised[row], as_tuple=False).squeeze(-1)
            if valid_pos.numel() > 0:
                pos = int(valid_pos[0].item())
                target_tok = int(labels[row, pos].item())
                pred_tok = int(preds[row, pos].item())
                logger.info(
                    "sample_token train=%s mask_mode=%s history_mode=%s pos=%d target=%s pred=%s",
                    str(train),
                    str(mask_mode),
                    str(history_mode),
                    int(pos),
                    _safe_decode(tokenizer, target_tok),
                    _safe_decode(tokenizer, pred_tok),
                )

            row_valid = valid_mask[row]
            row_content = row_valid & input_ids[row].ne(
                int(SATInterleavedTokenizer.PAD)
            )
            if placeholder_token is not None:
                row_content = row_content & input_ids[row].ne(int(placeholder_token))
                row_placeholder_tokens = int(
                    (row_valid & input_ids[row].eq(int(placeholder_token))).sum().item()
                )
            else:
                row_placeholder_tokens = 0

            row_effective_blocks = (
                int(torch.unique(block_ids[row, row_content]).numel())
                if bool(row_content.any().item())
                else 0
            )
            logger.info(
                "sample_history train=%s mask_mode=%s history_mode=%s seq_tokens=%d content_tokens=%d supervised_tokens=%d effective_blocks=%d placeholder_tokens=%d block_id_span=%d",
                str(train),
                str(mask_mode),
                str(history_mode),
                int(row_valid.sum().item()),
                int(row_content.sum().item()),
                int((loss_mask[row] & row_valid).sum().item()),
                int(row_effective_blocks),
                int(row_placeholder_tokens),
                int(block_ids[row, row_valid].max().item())
                if bool(row_valid.any().item())
                else 0,
            )
            logged = True

    stats = {
        "loss": float(total_loss / max(float(total_seq), 1.0)),
        "token_acc": float(total_correct / max(float(total_tokens), 1.0)),
        "tokens": float(total_tokens),
        "sequences": float(total_seq),
        "mean_blocks_per_seq": float(total_block_id_span / max(float(total_seq), 1.0)),
        "mean_effective_blocks_per_seq": float(
            total_effective_blocks / max(float(total_seq), 1.0)
        ),
        "mean_seq_tokens_per_seq": float(total_seq_tokens / max(float(total_seq), 1.0)),
        "mean_content_tokens_per_seq": float(
            total_content_tokens / max(float(total_seq), 1.0)
        ),
        "mean_supervised_tokens_per_seq": float(
            total_supervised_tokens / max(float(total_seq), 1.0)
        ),
        "mean_placeholder_tokens_per_seq": float(
            total_placeholder_tokens / max(float(total_seq), 1.0)
        ),
    }
    return stats


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("traces"), list):
        records = payload["traces"]
    elif (
        isinstance(payload, dict) and "sequences" in payload and "loss_masks" in payload
    ):
        records = [
            {"sequence": seq, "loss_mask": lm}
            for seq, lm in zip(payload["sequences"], payload["loss_masks"])
        ]
    else:
        raise ValueError(
            "unsupported data format; expected list[dict], dict['traces'], or dict with sequences/loss_masks"
        )

    if not records:
        raise ValueError("empty training data")
    return [dict(record) for record in records]


def _infer_vocab_size_from_records(records: Sequence[Dict[str, Any]]) -> int:
    max_token: int | None = None
    for record in records:
        seq = record.get("sequence")
        if not isinstance(seq, Sequence) or len(seq) == 0:
            continue
        seq_max = max(int(x) for x in seq)
        max_token = seq_max if max_token is None else max(max_token, seq_max)

    if max_token is None:
        raise ValueError("cannot infer vocab_size: all sequences are empty")

    vocab_size = int(max_token + 1)
    logger.info("inferred base_vocab_size=%d from training data", int(vocab_size))
    return vocab_size


def _infer_model_init_config_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = ckpt["model_state_dict"]
    if "position_embedding.weight" not in state_dict:
        raise RuntimeError(
            f"checkpoint missing position_embedding.weight: {checkpoint_path}"
        )

    inferred: Dict[str, Any] = {
        "max_seq_len": int(state_dict["position_embedding.weight"].shape[0])
    }
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(ckpt_cfg, dict):
        for key in (
            "model_type",
            "d_model",
            "n_layers",
            "n_heads",
            "n_slots",
            "dropout",
            "hidden_size",
            "n_lstm_layers",
            "block_mode",
        ):
            if key in ckpt_cfg:
                inferred[key] = ckpt_cfg[key]

    return inferred


def _load_with_vocab_expansion(
    model: torch.nn.Module,
    checkpoint_path: Path,
    target_vocab_size: int,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = dict(ckpt["model_state_dict"])
    if "token_embedding.weight" not in state_dict:
        raise RuntimeError(
            f"checkpoint missing token_embedding.weight: {checkpoint_path}"
        )

    src_vocab = int(state_dict["token_embedding.weight"].shape[0])
    target_vocab_size = int(target_vocab_size)
    expanded = False

    if src_vocab > target_vocab_size:
        raise RuntimeError(
            f"checkpoint vocab {src_vocab} exceeds target vocab {target_vocab_size}"
        )

    if src_vocab < target_vocab_size:
        d_model = int(state_dict["token_embedding.weight"].shape[1])
        expanded_embed = torch.empty(
            (target_vocab_size, d_model),
            dtype=state_dict["token_embedding.weight"].dtype,
        )
        torch.nn.init.normal_(expanded_embed, mean=0.0, std=0.02)
        expanded_embed[:src_vocab] = state_dict["token_embedding.weight"]
        state_dict["token_embedding.weight"] = expanded_embed
        if "lm_head.weight" in state_dict:
            state_dict["lm_head.weight"] = expanded_embed.clone()
        expanded = True

    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value

    if skipped:
        logger.warning(
            "Skipped %d checkpoint keys due to shape mismatch: %s",
            len(skipped),
            skipped,
        )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.warning("Missing keys after checkpoint load: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint load: %s", unexpected)

    return {
        "source_vocab": int(src_vocab),
        "target_vocab": int(target_vocab_size),
        "expanded": bool(expanded),
        "added_tokens": int(target_vocab_size - src_vocab),
        "skipped_mismatch": len(skipped),
    }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SAT SSA decoder with history ablations"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mask_mode",
        type=str,
        required=True,
        choices=[
            "selective_ssa",
            "full_causal",
            "blanket_ssa",
            "local_block_only",
            "swa_prefix",
            "reverse_selective",
            "random_matched",
        ],
    )
    parser.add_argument(
        "--history_mode",
        type=str,
        default="full",
        choices=list(HISTORY_MODES),
    )
    parser.add_argument(
        "--position_mode",
        type=str,
        default="auto",
        choices=list(POSITION_MODES),
        help=(
            "Forward-pass positional scheme. 'auto' preserves current behavior: "
            "use block-relative positions whenever block_ids are supplied."
        ),
    )
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--transplant_prob", type=float, default=1.0)
    parser.add_argument(
        "--partial_transplant",
        action="store_true",
        help="Replace prior history blocks independently with probability --transplant_prob.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "lstm"],
    )
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument(
        "--max_train_blocks",
        type=int,
        default=0,
        help="If >0, truncate each training trace after the first K search blocks while keeping max_seq_len unchanged.",
    )
    parser.add_argument(
        "--match_token_budget",
        action="store_true",
        help="If set with --max_train_blocks>0, scale epochs to approximately match the full-training supervised token budget.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help=(
            "Validation batch size. If omitted, defaults to --batch_size, except "
            "for --max_seq_len > 4096 where it auto-uses max(1, --batch_size // 4) "
            "to reduce validation OOM risk."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_slots", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_lstm_layers", type=int, default=4)
    parser.add_argument(
        "--block_mode",
        type=str,
        default="continuous",
        choices=["continuous", "block_reset"],
    )
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one data/model forward-pass smoke test and write smoke.json without training.",
    )
    return parser


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_val_batch_size(args: argparse.Namespace, max_seq_len: int) -> int:
    if args.val_batch_size is not None:
        val_batch_size = int(args.val_batch_size)
        source = "explicit"
    elif int(max_seq_len) > 4096:
        val_batch_size = max(1, int(args.batch_size) // 4)
        source = "auto_long_context"
    else:
        val_batch_size = int(args.batch_size)
        source = "match_train"

    if val_batch_size <= 0:
        raise ValueError("--val_batch_size must be positive")

    logger.info(
        "validation_batch_size resolved=%d source=%s train_batch_size=%d max_seq_len=%d",
        int(val_batch_size),
        source,
        int(args.batch_size),
        int(max_seq_len),
    )
    return int(val_batch_size)


def main() -> None:
    args = _build_argparser().parse_args()
    _set_seed(int(args.seed))

    if not (0.0 <= float(args.dropout_prob) <= 1.0):
        raise ValueError("--dropout_prob must be in [0, 1]")
    if not (0.0 <= float(args.transplant_prob) <= 1.0):
        raise ValueError("--transplant_prob must be in [0, 1]")
    if int(args.window_size) <= 0:
        raise ValueError("--window_size must be positive")
    if int(args.max_train_blocks) < 0:
        raise ValueError("--max_train_blocks must be >= 0")
    if args.val_batch_size is not None and int(args.val_batch_size) <= 0:
        raise ValueError("--val_batch_size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SATInterleavedTokenizer()
    records = _load_records(Path(args.data_path))
    base_vocab_size = _infer_vocab_size_from_records(records)
    _get_state_tried_tokens(base_vocab_size)
    placeholder_token = (
        int(base_vocab_size) if str(args.history_mode) == "null_history" else None
    )
    effective_vocab_size = int(
        base_vocab_size + (1 if placeholder_token is not None else 0)
    )
    device = torch.device(str(args.device))
    resolved_position_mode = _resolve_effective_position_mode(
        position_mode=str(args.position_mode),
        has_block_ids=True,
    )

    model_cfg: Dict[str, Any] = {
        "model_type": str(args.model_type),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "max_seq_len": int(args.max_seq_len),
        "n_slots": int(args.n_slots),
        "dropout": float(args.dropout),
        "hidden_size": int(args.hidden_size),
        "n_lstm_layers": int(args.n_lstm_layers),
        "block_mode": str(args.block_mode),
    }
    if str(args.init_checkpoint).strip():
        checkpoint_path = Path(args.init_checkpoint)
        inferred_cfg = _infer_model_init_config_from_checkpoint(checkpoint_path)
        for key, inferred_value in inferred_cfg.items():
            if key in model_cfg and model_cfg[key] != inferred_value:
                logger.info(
                    "overriding model %s from %s to checkpoint value %s",
                    key,
                    model_cfg[key],
                    inferred_value,
                )
            model_cfg[key] = inferred_value

    val_batch_size = _resolve_val_batch_size(
        args=args,
        max_seq_len=int(model_cfg["max_seq_len"]),
    )

    random.Random(int(args.seed)).shuffle(records)
    split = int(round((1.0 - float(args.val_split)) * len(records)))
    split = max(1, min(split, len(records) - 1))
    train_records = records[:split]
    val_records = records[split:]
    donor_pool: List[Dict[str, Any]] | None = None
    if str(args.history_mode) == "history_transplant":
        donor_pool = [dict(record) for record in train_records]
        random.Random(int(args.seed) + 17).shuffle(donor_pool)
        logger.info(
            "history_transplant_setup donors=%d transplant_prob=%.3f partial=%s",
            int(len(donor_pool)),
            float(args.transplant_prob),
            str(bool(args.partial_transplant)),
        )

    all_length_stats = compute_length_stats(
        records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        max_train_blocks=int(args.max_train_blocks),
    )
    train_length_stats = compute_length_stats(
        train_records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        max_train_blocks=int(args.max_train_blocks),
    )
    val_length_stats = compute_length_stats(
        val_records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        max_train_blocks=0,
    )

    mean_full_supervised_tokens = float(
        train_length_stats["supervised_tokens_per_trace"]["mean"]
    )
    mean_truncated_supervised_tokens = float(mean_full_supervised_tokens)
    token_budget_multiplier = 1.0
    if int(args.max_train_blocks) > 0:
        trunc_stats = train_length_stats.get("truncation", {})
        mean_truncated_supervised_tokens = float(
            trunc_stats.get("supervised_tokens_per_trace", {}).get("mean", 0.0)
        )
        if mean_truncated_supervised_tokens <= 0.0:
            raise RuntimeError(
                "--max_train_blocks removed all supervised tokens; choose a larger K"
            )
        token_budget_multiplier = float(
            mean_full_supervised_tokens / max(mean_truncated_supervised_tokens, 1e-8)
        )

    effective_epochs = int(args.epochs)
    if bool(args.match_token_budget):
        if int(args.max_train_blocks) <= 0:
            logger.warning(
                "--match_token_budget requested without --max_train_blocks>0; keeping epochs=%d",
                int(args.epochs),
            )
        else:
            effective_epochs = max(
                1,
                int(round(float(args.epochs) * float(token_budget_multiplier))),
            )

    length_stats_payload: Dict[str, Any] = {
        "data_path": str(args.data_path),
        "max_seq_len": int(model_cfg["max_seq_len"]),
        "max_train_blocks": int(args.max_train_blocks),
        "all_records": all_length_stats,
        "train_records": train_length_stats,
        "val_records": val_length_stats,
        "token_budget_matching": {
            "match_token_budget": bool(args.match_token_budget),
            "base_epochs": int(args.epochs),
            "effective_epochs": int(effective_epochs),
            "mean_full_supervised_tokens_per_trace": float(mean_full_supervised_tokens),
            "mean_truncated_supervised_tokens_per_trace": float(
                mean_truncated_supervised_tokens
            ),
            "supervised_token_multiplier": float(token_budget_multiplier),
        },
    }
    with (output_dir / "length_stats.json").open("w", encoding="utf-8") as f:
        json.dump(length_stats_payload, f, indent=2)
    logger.info(
        "length_stats_saved path=%s max_train_blocks=%d truncated_p95=%.1f token_multiplier=%.3f recommended_k=%s",
        str(output_dir / "length_stats.json"),
        int(args.max_train_blocks),
        float(
            train_length_stats.get("truncation", {})
            .get("tokens_per_trace", {})
            .get("p95", train_length_stats["tokens_per_trace"]["p95"])
        ),
        float(token_budget_multiplier),
        str(
            all_length_stats.get("recommended_short_context", {}).get(
                "recommended_max_train_blocks", 0
            )
        ),
    )

    base_train_ds = SSATriedDataset(
        train_records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        vocab_size=int(base_vocab_size),
    )
    train_core_ds: Dataset[Tuple[List[int], List[bool], List[int]]] = base_train_ds
    if int(args.max_train_blocks) > 0:
        train_core_ds = BlockTruncatedDataset(
            base_train_ds,
            max_train_blocks=int(args.max_train_blocks),
        )
    train_ds: Dataset[Tuple[List[int], List[bool], List[int]]]
    if str(args.history_mode) == "full":
        train_ds = train_core_ds
    else:
        train_ds = HistoryAblationDataset(
            train_core_ds,
            history_mode=str(args.history_mode),
            dropout_prob=float(args.dropout_prob),
            window_size=int(args.window_size),
            placeholder_token=placeholder_token,
            seed=int(args.seed),
            transplant_prob=float(args.transplant_prob),
            donor_pool=donor_pool,
            partial_transplant=bool(args.partial_transplant),
        )

    val_ds = SSATriedDataset(
        val_records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        vocab_size=int(base_vocab_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(val_batch_size),
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    model: torch.nn.Module
    if str(model_cfg["model_type"]) == "lstm":
        from universal.lstm_decoder import LSTMDecoder

        model = LSTMDecoder(
            vocab_size=int(effective_vocab_size),
            d_model=int(model_cfg["d_model"]),
            hidden_size=int(model_cfg["hidden_size"]),
            n_lstm_layers=int(model_cfg["n_lstm_layers"]),
            max_seq_len=int(model_cfg["max_seq_len"]),
            n_slots=int(model_cfg["n_slots"]),
            dropout=float(model_cfg["dropout"]),
            block_mode=str(model_cfg["block_mode"]),
        )
    else:
        model = SSASlotDecoder(
            vocab_size=int(effective_vocab_size),
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            n_heads=int(model_cfg["n_heads"]),
            max_seq_len=int(model_cfg["max_seq_len"]),
            n_slots=int(model_cfg["n_slots"]),
            dropout=float(model_cfg["dropout"]),
        )

    init_meta: Dict[str, Any] = {"used": False}
    if str(args.init_checkpoint).strip():
        init_meta = _load_with_vocab_expansion(
            model=model,
            checkpoint_path=Path(args.init_checkpoint),
            target_vocab_size=int(effective_vocab_size),
        )
        init_meta["used"] = True
        logger.info("initialized_from=%s meta=%s", str(args.init_checkpoint), init_meta)

    model = model.to(device)

    config = {
        "data_path": str(args.data_path),
        "output_dir": str(output_dir),
        "mask_mode": str(args.mask_mode),
        "training_position_mode": str(args.position_mode),
        "history_mode": str(args.history_mode),
        "dropout_prob": float(args.dropout_prob),
        "window_size": int(args.window_size),
        "transplant_prob": float(args.transplant_prob),
        "partial_transplant": bool(args.partial_transplant),
        "null_placeholder_token": (
            int(placeholder_token) if placeholder_token is not None else None
        ),
        "max_train_blocks": int(args.max_train_blocks),
        "match_token_budget": bool(args.match_token_budget),
        "effective_epochs": int(effective_epochs),
        "mean_full_supervised_tokens_per_trace": float(mean_full_supervised_tokens),
        "mean_truncated_supervised_tokens_per_trace": float(
            mean_truncated_supervised_tokens
        ),
        "supervised_token_multiplier": float(token_budget_multiplier),
        "base_vocab_size": int(base_vocab_size),
        "vocab_size": int(effective_vocab_size),
        "train_history_ablation": True,
        **model_cfg,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "val_batch_size": int(val_batch_size),
        "requested_val_batch_size": (
            int(args.val_batch_size) if args.val_batch_size is not None else None
        ),
        "val_batch_size_auto_rule": (
            "If --val_batch_size is omitted: use max(1, batch_size // 4) when "
            "max_seq_len > 4096, otherwise use batch_size."
        ),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "val_split": float(args.val_split),
        "device": str(args.device),
        "seed": int(args.seed),
        "init_checkpoint": str(args.init_checkpoint),
        "null_placeholder_token": (
            int(placeholder_token) if placeholder_token is not None else None
        ),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    planned_total_steps = int(len(train_loader) * int(effective_epochs))
    warmup_steps = int(planned_total_steps * float(args.warmup_ratio))

    if bool(args.smoke):
        train_input_ids, train_attention_mask, train_loss_mask, train_block_ids = next(
            iter(train_loader)
        )
        val_input_ids, val_attention_mask, val_loss_mask, val_block_ids = next(
            iter(val_loader)
        )
        train_input_ids = train_input_ids.to(device)
        train_attention_mask = train_attention_mask.to(device)
        train_loss_mask = train_loss_mask.to(device)
        train_block_ids = train_block_ids.to(device)
        val_input_ids = val_input_ids.to(device)
        val_attention_mask = val_attention_mask.to(device)
        val_loss_mask = val_loss_mask.to(device)
        val_block_ids = val_block_ids.to(device)
        with torch.no_grad():
            train_logits, _verify_logits = model(
                train_input_ids,
                train_attention_mask,
                block_ids=train_block_ids,
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
            )
            train_loss, train_token_count, train_supervised, train_labels = (
                _compute_lm_loss(
                    logits=train_logits,
                    input_ids=train_input_ids,
                    attention_mask=train_attention_mask,
                    loss_mask=train_loss_mask,
                )
            )
            train_preds = train_logits[:, :-1, :].argmax(dim=-1)
            train_correct = int(
                ((train_preds == train_labels) & train_supervised).sum().item()
            )

            val_logits, _verify_logits = model(
                val_input_ids,
                val_attention_mask,
                block_ids=val_block_ids,
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
            )
            val_loss, val_token_count, val_supervised, val_labels = _compute_lm_loss(
                logits=val_logits,
                input_ids=val_input_ids,
                attention_mask=val_attention_mask,
                loss_mask=val_loss_mask,
            )
            val_preds = val_logits[:, :-1, :].argmax(dim=-1)
            val_correct = int(((val_preds == val_labels) & val_supervised).sum().item())
        smoke_payload = {
            "smoke": True,
            "mask_mode": str(args.mask_mode),
            "training_position_mode": str(args.position_mode),
            "resolved_position_mode": str(resolved_position_mode),
            "history_mode": str(args.history_mode),
            "transplant_prob": float(args.transplant_prob),
            "partial_transplant": bool(args.partial_transplant),
            "max_train_blocks": int(args.max_train_blocks),
            "batch_size": int(args.batch_size),
            "val_batch_size": int(val_batch_size),
            "train_batch_shape": [int(x) for x in train_input_ids.shape],
            "train_token_count": int(train_token_count),
            "train_token_acc": float(train_correct / max(int(train_token_count), 1)),
            "train_loss": float(train_loss.item()),
            "val_batch_shape": [int(x) for x in val_input_ids.shape],
            "val_token_count": int(val_token_count),
            "val_token_acc": float(val_correct / max(int(val_token_count), 1)),
            "val_loss": float(val_loss.item()),
            "planned_total_steps": int(planned_total_steps),
        }
        with (output_dir / "smoke.json").open("w", encoding="utf-8") as f:
            json.dump(smoke_payload, f, indent=2)
        logger.info(
            "smoke_complete output=%s train_batch_shape=%s train_tokens=%d train_loss=%.4f train_acc=%.4f val_batch_shape=%s val_tokens=%d val_loss=%.4f val_acc=%.4f",
            str(output_dir / "smoke.json"),
            str(tuple(int(x) for x in train_input_ids.shape)),
            int(train_token_count),
            float(train_loss.item()),
            float(smoke_payload["train_token_acc"]),
            str(tuple(int(x) for x in val_input_ids.shape)),
            int(val_token_count),
            float(val_loss.item()),
            float(smoke_payload["val_token_acc"]),
        )
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(float(step + 1) / max(float(warmup_steps), 1.0), 1.0)
            if step < warmup_steps
            else 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (step - warmup_steps)
                    / max(float(planned_total_steps - warmup_steps), 1.0)
                )
            )
        ),
    )

    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    logger.info(
        "start_training model_type=%s block_mode=%s mask_mode=%s requested_position_mode=%s resolved_position_mode=%s history_mode=%s train=%d val=%d base_vocab_size=%d effective_vocab_size=%d batch=%d epochs=%d effective_epochs=%d max_train_blocks=%d token_multiplier=%.3f planned_steps=%d",
        str(model_cfg["model_type"]),
        str(model_cfg["block_mode"]),
        str(args.mask_mode),
        str(args.position_mode),
        str(resolved_position_mode),
        str(args.history_mode),
        int(len(train_ds)),
        int(len(val_ds)),
        int(base_vocab_size),
        int(effective_vocab_size),
        int(args.batch_size),
        int(args.epochs),
        int(effective_epochs),
        int(args.max_train_blocks),
        float(token_budget_multiplier),
        int(planned_total_steps),
    )
    if placeholder_token is not None:
        logger.info(
            "null_history_placeholder_token=%d base_vocab_size=%d effective_vocab_size=%d",
            int(placeholder_token),
            int(base_vocab_size),
            int(effective_vocab_size),
        )

    for epoch in range(int(effective_epochs)):
        train_stats = _run_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            device=device,
            tokenizer=tokenizer,
            mask_mode=str(args.mask_mode),
            position_mode=str(args.position_mode),
            history_mode=str(args.history_mode),
            placeholder_token=placeholder_token,
        )
        val_stats = _run_epoch(
            loader=val_loader,
            model=model,
            optimizer=None,
            scheduler=None,
            train=False,
            device=device,
            tokenizer=tokenizer,
            mask_mode=str(args.mask_mode),
            position_mode=str(args.position_mode),
            history_mode="full",
            placeholder_token=None,
        )

        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": float(epoch + 1),
            "effective_epochs": float(effective_epochs),
            "train_loss": float(train_stats["loss"]),
            "train_token_acc": float(train_stats["token_acc"]),
            "val_loss": float(val_stats["loss"]),
            "val_token_acc": float(val_stats["token_acc"]),
            "train_mean_blocks_per_seq": float(
                train_stats.get("mean_blocks_per_seq", 0.0)
            ),
            "val_mean_blocks_per_seq": float(val_stats.get("mean_blocks_per_seq", 0.0)),
            "train_mean_effective_blocks_per_seq": float(
                train_stats.get("mean_effective_blocks_per_seq", 0.0)
            ),
            "val_mean_effective_blocks_per_seq": float(
                val_stats.get("mean_effective_blocks_per_seq", 0.0)
            ),
            "train_mean_seq_tokens_per_seq": float(
                train_stats.get("mean_seq_tokens_per_seq", 0.0)
            ),
            "train_mean_content_tokens_per_seq": float(
                train_stats.get("mean_content_tokens_per_seq", 0.0)
            ),
            "train_mean_supervised_tokens_per_seq": float(
                train_stats.get("mean_supervised_tokens_per_seq", 0.0)
            ),
            "train_mean_placeholder_tokens_per_seq": float(
                train_stats.get("mean_placeholder_tokens_per_seq", 0.0)
            ),
            "val_mean_seq_tokens_per_seq": float(
                val_stats.get("mean_seq_tokens_per_seq", 0.0)
            ),
            "val_mean_content_tokens_per_seq": float(
                val_stats.get("mean_content_tokens_per_seq", 0.0)
            ),
            "val_mean_supervised_tokens_per_seq": float(
                val_stats.get("mean_supervised_tokens_per_seq", 0.0)
            ),
            "lr": float(lr_now),
            "mask_mode": str(args.mask_mode),
            "training_position_mode": str(args.position_mode),
            "resolved_position_mode": str(resolved_position_mode),
            "history_mode": str(args.history_mode),
            "dropout_prob": float(args.dropout_prob),
            "window_size": int(args.window_size),
            "transplant_prob": float(args.transplant_prob),
            "partial_transplant": bool(args.partial_transplant),
        }
        history.append(row)

        logger.info(
            "epoch=%d/%d mask_mode=%s history_mode=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f train_blocks=%.2f train_effective_blocks=%.2f train_content_tokens=%.1f lr=%.2e",
            int(epoch + 1),
            int(args.epochs),
            str(args.mask_mode),
            str(args.history_mode),
            float(train_stats["loss"]),
            float(train_stats["token_acc"]),
            float(val_stats["loss"]),
            float(val_stats["token_acc"]),
            float(train_stats.get("mean_blocks_per_seq", 0.0)),
            float(train_stats.get("mean_effective_blocks_per_seq", 0.0)),
            float(train_stats.get("mean_content_tokens_per_seq", 0.0)),
            float(lr_now),
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_type": str(model_cfg["model_type"]),
                "base_vocab_size": int(base_vocab_size),
                "vocab_size": int(effective_vocab_size),
                "d_model": int(model_cfg["d_model"]),
                "n_layers": int(model_cfg["n_layers"]),
                "n_heads": int(model_cfg["n_heads"]),
                "n_slots": int(model_cfg["n_slots"]),
                "max_seq_len": int(model_cfg["max_seq_len"]),
                "dropout": float(model_cfg["dropout"]),
                "hidden_size": int(model_cfg["hidden_size"]),
                "n_lstm_layers": int(model_cfg["n_lstm_layers"]),
                "block_mode": str(model_cfg["block_mode"]),
                "attention_mode": "ssa",
                "mask_mode": str(args.mask_mode),
                "training_position_mode": str(args.position_mode),
                "resolved_position_mode": str(resolved_position_mode),
                "history_mode": str(args.history_mode),
                "dropout_prob": float(args.dropout_prob),
                "window_size": int(args.window_size),
                "transplant_prob": float(args.transplant_prob),
                "partial_transplant": bool(args.partial_transplant),
                "seed": int(args.seed),
                "init_checkpoint": str(args.init_checkpoint),
                "null_placeholder_token": (
                    int(placeholder_token) if placeholder_token is not None else None
                ),
            },
            "epoch": int(epoch + 1),
            "train_loss": float(train_stats["loss"]),
            "val_loss": float(val_stats["loss"]),
            "history": history,
            "init_meta": init_meta,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if float(val_stats["loss"]) < best_val:
            best_val = float(val_stats["loss"])
            torch.save(ckpt, output_dir / "best.pt")

    summary = {
        "mask_mode": str(args.mask_mode),
        "training_position_mode": str(args.position_mode),
        "history_mode": str(args.history_mode),
        "transplant_prob": float(args.transplant_prob),
        "partial_transplant": bool(args.partial_transplant),
        "max_train_blocks": int(args.max_train_blocks),
        "match_token_budget": bool(args.match_token_budget),
        "effective_epochs": int(effective_epochs),
        "planned_total_steps": int(planned_total_steps),
        "mean_full_supervised_tokens_per_trace": float(mean_full_supervised_tokens),
        "mean_truncated_supervised_tokens_per_trace": float(
            mean_truncated_supervised_tokens
        ),
        "supervised_token_multiplier": float(token_budget_multiplier),
        "base_vocab_size": int(base_vocab_size),
        "vocab_size": int(effective_vocab_size),
        "train_examples": int(len(train_ds)),
        "val_examples": int(len(val_ds)),
        "epochs": int(args.epochs),
        "best_val_loss": float(best_val),
        "history": history,
        "config": config,
        "init_meta": init_meta,
        "null_placeholder_token": (
            int(placeholder_token) if placeholder_token is not None else None
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "training_complete mask_mode=%s history_mode=%s best_val_loss=%.4f output_dir=%s",
        str(args.mask_mode),
        str(args.history_mode),
        best_val,
        output_dir,
    )


if __name__ == "__main__":
    main()
