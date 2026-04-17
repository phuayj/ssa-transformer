#!/usr/bin/env python3
"""Train CBV (Counterfactual Branch Verifier) on extracted SAT decision states.

This script builds *single-state* verification examples from full SAT traces:

    [prefix tokens incl. SEARCH] [STATE ... SEP]

Each example is paired with one verify label tuple:
    (var_id, is_extendable_T, is_extendable_F)

The training objective is BCE-with-logits on the two branch extendability labels.
When training from scratch (no --checkpoint), LM next-token loss is also applied.
When fine-tuning from a checkpoint, verify loss is used by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.interleaved_tokenizer import SATInterleavedTokenizer
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VerifyExample:
    tokens: List[int]
    prefix_len: int
    var_id: int
    ext_t: int
    ext_f: int
    swapped: bool = False


@dataclass
class PBPExample:
    tokens: List[int]
    block_ids: List[int]
    sep_pos_1: int
    sep_pos_2: int
    dead_1: int
    dead_2: int
    swapped: bool = False


class ViabilityHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(d_model), 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class CBVVerifyDataset(Dataset):
    """Dataset of (prefix + single decision-state block, verify labels)."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        max_seq_len: int,
        vocab_size: int,
        tokenizer: SATInterleavedTokenizer,
        polarity_swap: bool = False,
    ):
        self.examples: List[VerifyExample] = []
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)
        self._tok = tokenizer
        self.polarity_swap = bool(polarity_swap)

        self.stats: Dict[str, int] = {
            "records": 0,
            "records_with_verify_labels": 0,
            "records_missing_search": 0,
            "records_with_no_decision_blocks": 0,
            "decision_blocks": 0,
            "verify_labels": 0,
            "aligned_pairs": 0,
            "var_mismatch": 0,
            "skipped_too_long": 0,
            "skipped_bad_token": 0,
            "examples_original": 0,
            "examples_swapped": 0,
        }

        for rec in records:
            self.stats["records"] += 1
            seq = rec.get("sequence")
            labels = rec.get("verify_labels")
            if seq is None or labels is None:
                continue
            self.stats["records_with_verify_labels"] += 1

            prefix, decision_blocks, decision_vars = self._extract_decision_states(rec)
            if not prefix:
                self.stats["records_missing_search"] += 1
                continue
            if not decision_blocks:
                self.stats["records_with_no_decision_blocks"] += 1
                continue

            clean_labels: List[Tuple[int, int, int]] = []
            for item in labels:
                if not isinstance(item, (list, tuple)) or len(item) != 3:
                    continue
                v, t, f = item
                clean_labels.append((int(v), int(bool(t)), int(bool(f))))

            self.stats["decision_blocks"] += int(len(decision_blocks))
            self.stats["verify_labels"] += int(len(clean_labels))
            aligned_count = min(len(decision_blocks), len(clean_labels))
            self.stats["aligned_pairs"] += int(aligned_count)

            for idx in range(aligned_count):
                state_block = decision_blocks[idx]
                block_var = decision_vars[idx]
                label_var, ext_t, ext_f = clean_labels[idx]
                if int(block_var) != int(label_var):
                    self.stats["var_mismatch"] += 1

                tokens = [int(x) for x in prefix + state_block]
                if len(tokens) > int(self.max_seq_len):
                    self.stats["skipped_too_long"] += 1
                    continue
                if any(int(t) < 0 or int(t) >= int(self.vocab_size) for t in tokens):
                    self.stats["skipped_bad_token"] += 1
                    continue

                self.examples.append(
                    VerifyExample(
                        tokens=tokens,
                        prefix_len=int(len(prefix)),
                        var_id=int(label_var),
                        ext_t=int(ext_t),
                        ext_f=int(ext_f),
                        swapped=False,
                    )
                )
                self.stats["examples_original"] += 1

                if self.polarity_swap:
                    self.examples.append(
                        VerifyExample(
                            tokens=tokens,
                            prefix_len=int(len(prefix)),
                            var_id=int(label_var),
                            ext_t=int(ext_f),
                            ext_f=int(ext_t),
                            swapped=True,
                        )
                    )
                    self.stats["examples_swapped"] += 1

    def __len__(self) -> int:
        return int(len(self.examples))

    def __getitem__(self, idx: int) -> VerifyExample:
        return self.examples[int(idx)]

    def _extract_decision_states(
        self,
        rec: Dict[str, Any],
    ) -> Tuple[List[int], List[List[int]], List[int]]:
        seq_raw = rec.get("sequence", [])
        seq = [int(x) for x in seq_raw]

        search_idx = -1
        for i, tok in enumerate(seq):
            if int(tok) == int(self._tok.SEARCH_START):
                search_idx = int(i)
                break
        if search_idx < 0:
            return [], [], []

        prefix = list(seq[: search_idx + 1])

        decision_blocks: List[List[int]] = []
        decision_vars: List[int] = []
        i = int(search_idx + 1)
        while i < len(seq):
            if int(seq[i]) != int(self._tok.STATE):
                i += 1
                continue

            block_start = int(i)
            j = int(i + 1)
            while j < len(seq) and int(seq[j]) != int(self._tok.SEP):
                j += 1
            if j >= len(seq):
                break

            state_block = list(seq[block_start : j + 1])
            next_tok = int(seq[j + 1]) if (j + 1) < len(seq) else -1
            if next_tok >= int(self._tok.VAR_OFFSET) and next_tok < int(
                self._tok.VOCAB_SIZE
            ):
                decision_blocks.append(state_block)
                decision_vars.append(int(next_tok - int(self._tok.VAR_OFFSET)))

            i = int(j + 1)

        return prefix, decision_blocks, decision_vars


class PBPViabilityDataset(Dataset):
    """Dataset of paired hypothetical post-UP states for PBP viability training."""

    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        *,
        max_seq_len: int,
        vocab_size: int,
        swap_augment: bool = True,
    ):
        self.examples: List[Dict[str, Any]] = []
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)
        self.swap_augment = bool(swap_augment)
        self.stats: Dict[str, int] = {
            "raw_examples": 0,
            "kept_examples": 0,
            "skipped_too_long": 0,
            "skipped_bad_token": 0,
            "both_dead": 0,
            "both_viable": 0,
            "one_dead": 0,
        }

        for ex in examples:
            self.stats["raw_examples"] += 1
            prefix = [int(x) for x in ex.get("prefix_tokens", [])]
            st_t = [int(x) for x in ex.get("state_t_tokens", [])]
            st_f = [int(x) for x in ex.get("state_f_tokens", [])]
            dead_t = int(bool(ex.get("dead_t", False)))
            dead_f = int(bool(ex.get("dead_f", False)))

            tokens = prefix + st_t + st_f
            if len(tokens) > int(self.max_seq_len):
                self.stats["skipped_too_long"] += 1
                continue
            if any(int(t) < 0 or int(t) >= int(self.vocab_size) for t in tokens):
                self.stats["skipped_bad_token"] += 1
                continue

            self.examples.append(
                {
                    "prefix_tokens": prefix,
                    "state_t_tokens": st_t,
                    "state_f_tokens": st_f,
                    "dead_t": int(dead_t),
                    "dead_f": int(dead_f),
                }
            )
            self.stats["kept_examples"] += 1
            if dead_t and dead_f:
                self.stats["both_dead"] += 1
            elif (not dead_t) and (not dead_f):
                self.stats["both_viable"] += 1
            else:
                self.stats["one_dead"] += 1

    def __len__(self) -> int:
        return int(len(self.examples))

    def __getitem__(self, idx: int) -> PBPExample:
        ex = self.examples[int(idx)]
        prefix = [int(x) for x in ex["prefix_tokens"]]
        st_t = [int(x) for x in ex["state_t_tokens"]]
        st_f = [int(x) for x in ex["state_f_tokens"]]

        swapped = bool(self.swap_augment and (random.random() < 0.5))
        if swapped:
            tokens = prefix + st_f + st_t
            block_ids = [0] * len(prefix) + [1] * len(st_f) + [2] * len(st_t)
            sep_pos_1 = int(len(prefix) + len(st_f) - 1)
            sep_pos_2 = int(len(prefix) + len(st_f) + len(st_t) - 1)
            dead_1 = int(ex["dead_f"])
            dead_2 = int(ex["dead_t"])
        else:
            tokens = prefix + st_t + st_f
            block_ids = [0] * len(prefix) + [1] * len(st_t) + [2] * len(st_f)
            sep_pos_1 = int(len(prefix) + len(st_t) - 1)
            sep_pos_2 = int(len(prefix) + len(st_t) + len(st_f) - 1)
            dead_1 = int(ex["dead_t"])
            dead_2 = int(ex["dead_f"])

        return PBPExample(
            tokens=tokens,
            block_ids=block_ids,
            sep_pos_1=int(sep_pos_1),
            sep_pos_2=int(sep_pos_2),
            dead_1=int(dead_1),
            dead_2=int(dead_2),
            swapped=bool(swapped),
        )


def _collate_cbv_batch(
    batch: Sequence[VerifyExample],
    *,
    pad_token: int,
    block_mode: str,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = int(len(batch))
    max_len = max(int(len(item.tokens)) for item in batch)

    input_ids = torch.full((batch_size, max_len), int(pad_token), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    block_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    targets = torch.zeros((batch_size, 2), dtype=torch.float32)
    var_ids = torch.zeros((batch_size,), dtype=torch.long)
    swapped = torch.zeros((batch_size,), dtype=torch.bool)

    for i, ex in enumerate(batch):
        seq_len = int(len(ex.tokens))
        input_ids[i, :seq_len] = torch.tensor(ex.tokens, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        if str(block_mode) == "prefix_state":
            pfx = int(max(0, min(int(ex.prefix_len), seq_len)))
            block_ids[i, :pfx] = 0
            block_ids[i, pfx:seq_len] = 1
        elif str(block_mode) == "all_zero":
            block_ids[i, :seq_len] = 0
        else:
            raise ValueError(f"unknown block_mode='{block_mode}'")

        targets[i, 0] = float(ex.ext_t)
        targets[i, 1] = float(ex.ext_f)
        var_ids[i] = int(ex.var_id)
        swapped[i] = bool(ex.swapped)

    return input_ids, attention_mask, block_ids, targets, var_ids, swapped


def _collate_pbp_batch(
    batch: Sequence[PBPExample],
    *,
    pad_token: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = int(len(batch))
    max_len = max(int(len(item.tokens)) for item in batch)

    input_ids = torch.full((batch_size, max_len), int(pad_token), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    block_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    sep_pos_1 = torch.zeros((batch_size,), dtype=torch.long)
    sep_pos_2 = torch.zeros((batch_size,), dtype=torch.long)
    dead_1 = torch.zeros((batch_size,), dtype=torch.float32)
    dead_2 = torch.zeros((batch_size,), dtype=torch.float32)

    for i, ex in enumerate(batch):
        seq_len = int(len(ex.tokens))
        input_ids[i, :seq_len] = torch.tensor(ex.tokens, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        block_ids[i, :seq_len] = torch.tensor(ex.block_ids, dtype=torch.long)
        sep_pos_1[i] = int(ex.sep_pos_1)
        sep_pos_2[i] = int(ex.sep_pos_2)
        dead_1[i] = float(ex.dead_1)
        dead_2[i] = float(ex.dead_2)

    return input_ids, attention_mask, block_ids, sep_pos_1, sep_pos_2, dead_1, dead_2


def _compute_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token: int,
) -> Tuple[torch.Tensor, int]:
    labels = input_ids[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    supervised = attention_mask[:, 1:] > 0
    token_count = int(supervised.sum().item())
    if token_count == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype), 0
    labels = labels.masked_fill(~supervised, int(pad_token))
    loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=int(pad_token),
    )
    return loss, token_count


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute tie-aware AUROC for binary labels.

    Returns NaN if labels are all-positive or all-negative.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = (np.asarray(labels).reshape(-1) > 0.5).astype(np.int32)
    if scores.size != labels.size:
        raise ValueError(
            f"scores and labels must have same size; got {scores.size} vs {labels.size}"
        )

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    _, first_idx, counts = np.unique(
        sorted_scores,
        return_index=True,
        return_counts=True,
    )
    for start, count in zip(first_idx.tolist(), counts.tolist()):
        end = int(start + count)
        ranks[start:end] = (float(start + 1) + float(end)) / 2.0

    sum_pos_ranks = float(ranks[sorted_labels == 1].sum())
    return float(
        (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / float(max(n_pos * n_neg, 1))
    )


def _compute_verify_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if probs.size == 0 or targets.size == 0:
        return {
            "branch_acc": 0.0,
            "branch_t_acc": 0.0,
            "branch_f_acc": 0.0,
            "four_way_acc": 0.0,
            "dead_end_recall": 0.0,
            "dead_end_precision": 0.0,
            "dead_end_support": 0.0,
            "pred_dead_end": 0.0,
            "auroc_t": float("nan"),
            "auroc_f": float("nan"),
        }

    preds = (probs >= float(threshold)).astype(np.int32)
    labels = (targets >= 0.5).astype(np.int32)

    branch_t_acc = float((preds[:, 0] == labels[:, 0]).mean())
    branch_f_acc = float((preds[:, 1] == labels[:, 1]).mean())
    branch_acc = float((preds == labels).mean())

    pred_four = preds[:, 0] * 2 + preds[:, 1]
    true_four = labels[:, 0] * 2 + labels[:, 1]
    four_way_acc = float((pred_four == true_four).mean())

    true_dead = (labels[:, 0] == 0) & (labels[:, 1] == 0)
    pred_dead = (preds[:, 0] == 0) & (preds[:, 1] == 0)
    tp_dead = int((true_dead & pred_dead).sum())
    dead_end_recall = float(tp_dead) / max(float(int(true_dead.sum())), 1.0)
    dead_end_precision = float(tp_dead) / max(float(int(pred_dead.sum())), 1.0)

    auroc_t = _binary_auroc(probs[:, 0], labels[:, 0])
    auroc_f = _binary_auroc(probs[:, 1], labels[:, 1])

    return {
        "branch_acc": float(branch_acc),
        "branch_t_acc": float(branch_t_acc),
        "branch_f_acc": float(branch_f_acc),
        "four_way_acc": float(four_way_acc),
        "dead_end_recall": float(dead_end_recall),
        "dead_end_precision": float(dead_end_precision),
        "dead_end_support": float(int(true_dead.sum())),
        "pred_dead_end": float(int(pred_dead.sum())),
        "auroc_t": float(auroc_t),
        "auroc_f": float(auroc_f),
    }


def pbp_loss(
    logit_1: torch.Tensor,
    logit_2: torch.Tensor,
    dead_1: torch.Tensor,
    dead_2: torch.Tensor,
    *,
    margin: float = 0.5,
    rank_weight: float = 0.25,
) -> Tuple[torch.Tensor, float, float]:
    bce_1 = F.binary_cross_entropy_with_logits(logit_1, dead_1.float())
    bce_2 = F.binary_cross_entropy_with_logits(logit_2, dead_2.float())
    bce = bce_1 + bce_2

    one_dead_mask = dead_1.ne(dead_2)
    if bool(one_dead_mask.any().item()):
        dead_logit = torch.where(one_dead_mask & dead_1.bool(), logit_1, logit_2)
        live_logit = torch.where(one_dead_mask & dead_1.bool(), logit_2, logit_1)
        rank_raw = F.relu(float(margin) - (dead_logit - live_logit))
        rank_loss = rank_raw[one_dead_mask].mean()
    else:
        rank_loss = torch.zeros((), dtype=logit_1.dtype, device=logit_1.device)

    total = bce + float(rank_weight) * rank_loss
    return total, float(bce.item()), float(rank_loss.item())


def _binary_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = (np.asarray(labels).reshape(-1) > 0.5).astype(np.int32)
    if scores.size != labels.size:
        raise ValueError("scores and labels must have same size")
    if int(labels.sum()) == 0:
        return float("nan")

    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y, dtype=np.float64)
    fp = np.cumsum(1 - y, dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(float(labels.sum()), 1.0)

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _compute_pbp_metrics(
    probs_1: np.ndarray,
    probs_2: np.ndarray,
    dead_1: np.ndarray,
    dead_2: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    p1 = np.asarray(probs_1, dtype=np.float64).reshape(-1)
    p2 = np.asarray(probs_2, dtype=np.float64).reshape(-1)
    d1 = (np.asarray(dead_1).reshape(-1) > 0.5).astype(np.int32)
    d2 = (np.asarray(dead_2).reshape(-1) > 0.5).astype(np.int32)
    if p1.size == 0:
        return {
            "state_auroc": float("nan"),
            "state_auprc": float("nan"),
            "state_acc": 0.0,
            "force_precision": 0.0,
            "force_recall": 0.0,
            "force_coverage": 0.0,
            "both_viable_false_force_rate": 0.0,
            "force_pred_count": 0.0,
            "force_true_count": 0.0,
        }

    pred_d1 = (p1 >= float(threshold)).astype(np.int32)
    pred_d2 = (p2 >= float(threshold)).astype(np.int32)

    state_probs = np.concatenate([p1, p2], axis=0)
    state_labels = np.concatenate([d1, d2], axis=0)
    state_auroc = _binary_auroc(state_probs, state_labels)
    state_auprc = _binary_auprc(state_probs, state_labels)
    state_acc = float((np.concatenate([pred_d1, pred_d2]) == state_labels).mean())

    pred_force = pred_d1 ^ pred_d2
    true_force = d1 ^ d2
    pred_dead_side_is_1 = pred_d1.astype(bool)
    true_dead_side_is_1 = d1.astype(bool)
    force_correct = pred_force.astype(bool) & (
        pred_dead_side_is_1 == true_dead_side_is_1
    )

    force_precision = float(force_correct.sum()) / max(float(pred_force.sum()), 1.0)
    force_recall = float(force_correct.sum()) / max(float(true_force.sum()), 1.0)
    force_coverage = float(pred_force.sum()) / max(float(p1.size), 1.0)

    both_viable = (d1 == 0) & (d2 == 0)
    both_viable_false_force_rate = float(
        (pred_force.astype(bool) & both_viable).sum()
    ) / max(float(both_viable.sum()), 1.0)

    return {
        "state_auroc": float(state_auroc),
        "state_auprc": float(state_auprc),
        "state_acc": float(state_acc),
        "force_precision": float(force_precision),
        "force_recall": float(force_recall),
        "force_coverage": float(force_coverage),
        "both_viable_false_force_rate": float(both_viable_false_force_rate),
        "force_pred_count": float(int(pred_force.sum())),
        "force_true_count": float(int(true_force.sum())),
    }


def _run_epoch(
    *,
    loader: DataLoader,
    model: SSASlotDecoder,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    tokenizer: SATInterleavedTokenizer,
    verify_weight: float,
    lm_weight: float,
    use_lm_loss: bool,
    mask_mode: str,
    position_mode: str,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    total_verify_loss_sum = 0.0
    total_lm_loss_sum = 0.0
    total_examples = 0
    total_lm_tokens = 0

    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    logged = False
    for input_ids, attention_mask, block_ids, targets, var_ids, swapped in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        block_ids = block_ids.to(device)
        targets = targets.to(device)
        var_ids = var_ids.to(device)
        swapped = swapped.to(device=device, dtype=torch.bool)

        with torch.set_grad_enabled(train):
            lm_logits, verify_logits, _slot_states = model(
                input_ids,
                attention_mask=attention_mask,
                block_ids=block_ids,
                mask_mode=str(mask_mode),
                position_mode=str(position_mode),
                return_slot_states=True,
            )
            if bool(swapped.any().item()):
                swapped_logits = verify_logits.clone()
                swapped_logits[swapped, 0] = verify_logits[swapped, 1]
                swapped_logits[swapped, 1] = verify_logits[swapped, 0]
                verify_logits = swapped_logits

            verify_loss = F.binary_cross_entropy_with_logits(verify_logits, targets)

            lm_loss, lm_tokens = _compute_lm_loss(
                logits=lm_logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token=int(tokenizer.PAD),
            )
            loss = float(verify_weight) * verify_loss
            if bool(use_lm_loss):
                loss = loss + float(lm_weight) * lm_loss

            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required when train=True")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = int(input_ids.size(0))
        total_examples += int(batch_size)
        total_loss_sum += float(loss.item()) * float(batch_size)
        total_verify_loss_sum += float(verify_loss.item()) * float(batch_size)
        total_lm_loss_sum += float(lm_loss.item()) * float(batch_size)
        total_lm_tokens += int(lm_tokens)

        probs = torch.sigmoid(verify_logits).detach().cpu().numpy()
        t_np = targets.detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(t_np)

        if not logged:
            sample_n = min(4, batch_size)
            logger.info(
                "sample_verify train=%s vars=%s probs=%s targets=%s",
                str(train),
                [int(v) for v in var_ids[:sample_n].detach().cpu().tolist()],
                np.round(probs[:sample_n], 4).tolist(),
                t_np[:sample_n].astype(int).tolist(),
            )
            logger.info(
                "sample_verify_swap_stats train=%s swapped_in_batch=%d batch_size=%d",
                str(train),
                int(swapped.sum().item()),
                int(batch_size),
            )
            valid_tokens = int(attention_mask[0].sum().item())
            row_blocks = block_ids[0, :valid_tokens]
            logger.info(
                "sample_input_stats train=%s seq_len=%d graph_prefix_tokens=%d unique_blocks=%d",
                str(train),
                int(valid_tokens),
                int((row_blocks == 0).sum().item()),
                int(row_blocks.unique().numel()),
            )
            logged = True

    probs_all = (
        np.concatenate(all_probs, axis=0)
        if all_probs
        else np.zeros((0, 2), dtype=np.float32)
    )
    targets_all = (
        np.concatenate(all_targets, axis=0)
        if all_targets
        else np.zeros((0, 2), dtype=np.float32)
    )
    verify_metrics = _compute_verify_metrics(probs_all, targets_all)

    stats: Dict[str, float] = {
        "loss": float(total_loss_sum / max(float(total_examples), 1.0)),
        "verify_loss": float(total_verify_loss_sum / max(float(total_examples), 1.0)),
        "lm_loss": float(total_lm_loss_sum / max(float(total_examples), 1.0)),
        "examples": float(total_examples),
        "lm_tokens": float(total_lm_tokens),
    }
    stats.update(verify_metrics)
    return stats


def _run_epoch_pbp(
    *,
    loader: DataLoader,
    model: SSASlotDecoder,
    viability_head: ViabilityHead,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mask_mode: str,
    position_mode: str,
    rank_margin: float,
    rank_weight: float,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
        viability_head.train()
    else:
        model.eval()
        viability_head.eval()

    total_loss_sum = 0.0
    total_bce_sum = 0.0
    total_rank_sum = 0.0
    total_examples = 0

    all_p1: List[np.ndarray] = []
    all_p2: List[np.ndarray] = []
    all_d1: List[np.ndarray] = []
    all_d2: List[np.ndarray] = []

    logged = False
    for (
        input_ids,
        attention_mask,
        block_ids,
        sep_pos_1,
        sep_pos_2,
        dead_1,
        dead_2,
    ) in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        block_ids = block_ids.to(device)
        sep_pos_1 = sep_pos_1.to(device)
        sep_pos_2 = sep_pos_2.to(device)
        dead_1 = dead_1.to(device)
        dead_2 = dead_2.to(device)

        with torch.set_grad_enabled(train):
            lm_logits, _verify_logits, seq_out = model(
                input_ids,
                attention_mask=attention_mask,
                block_ids=block_ids,
                mask_mode=str(mask_mode),
                position_mode=str(position_mode),
                return_hidden_states=True,
            )
            del lm_logits

            bsz = int(seq_out.size(0))
            idx = torch.arange(bsz, device=device)
            h1 = seq_out[idx, sep_pos_1, :]
            h2 = seq_out[idx, sep_pos_2, :]

            logit_1 = viability_head(h1).squeeze(-1)
            logit_2 = viability_head(h2).squeeze(-1)

            loss, bce_value, rank_value = pbp_loss(
                logit_1,
                logit_2,
                dead_1,
                dead_2,
                margin=float(rank_margin),
                rank_weight=float(rank_weight),
            )

            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required when train=True")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(
                    viability_head.parameters(), max_norm=1.0
                )
                optimizer.step()

        batch_size = int(input_ids.size(0))
        total_examples += int(batch_size)
        total_loss_sum += float(loss.item()) * float(batch_size)
        total_bce_sum += float(bce_value) * float(batch_size)
        total_rank_sum += float(rank_value) * float(batch_size)

        p1 = torch.sigmoid(logit_1).detach().cpu().numpy()
        p2 = torch.sigmoid(logit_2).detach().cpu().numpy()
        d1 = dead_1.detach().cpu().numpy()
        d2 = dead_2.detach().cpu().numpy()
        all_p1.append(p1)
        all_p2.append(p2)
        all_d1.append(d1)
        all_d2.append(d2)

        if not logged:
            sample_n = min(6, batch_size)
            logger.info(
                "sample_pbp train=%s probs1=%s probs2=%s dead1=%s dead2=%s",
                str(train),
                np.round(p1[:sample_n], 4).tolist(),
                np.round(p2[:sample_n], 4).tolist(),
                d1[:sample_n].astype(int).tolist(),
                d2[:sample_n].astype(int).tolist(),
            )
            logged = True

    probs_1 = (
        np.concatenate(all_p1, axis=0) if all_p1 else np.zeros((0,), dtype=np.float32)
    )
    probs_2 = (
        np.concatenate(all_p2, axis=0) if all_p2 else np.zeros((0,), dtype=np.float32)
    )
    labels_1 = (
        np.concatenate(all_d1, axis=0) if all_d1 else np.zeros((0,), dtype=np.float32)
    )
    labels_2 = (
        np.concatenate(all_d2, axis=0) if all_d2 else np.zeros((0,), dtype=np.float32)
    )

    metrics = _compute_pbp_metrics(
        probs_1,
        probs_2,
        labels_1,
        labels_2,
        threshold=0.5,
    )
    metrics.update(
        {
            "loss": float(total_loss_sum / max(float(total_examples), 1.0)),
            "bce_loss": float(total_bce_sum / max(float(total_examples), 1.0)),
            "rank_loss": float(total_rank_sum / max(float(total_examples), 1.0)),
            "examples": float(total_examples),
        }
    )
    return metrics


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    else:
        raise ValueError(
            "unsupported payload format; expected list[dict] or dict(records=...)"
        )

    if not records:
        raise ValueError("no records found")
    return [dict(r) for r in records]


def _load_pbp_examples(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError("pbp payload must be dict with examples")
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("pbp payload missing list field 'examples'")
    if not examples:
        raise ValueError("pbp payload has empty examples list")
    meta = {
        "config": payload.get("config", {}),
        "stats": payload.get("stats", {}),
    }
    return [dict(x) for x in examples], meta


def _infer_model_cfg_from_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(f"checkpoint missing model_state_dict: {path}")
    state_dict = ckpt["model_state_dict"]
    cfg: Dict[str, Any] = {}

    if isinstance(ckpt.get("config"), dict):
        c = ckpt["config"]
        for key in (
            "d_model",
            "n_layers",
            "n_heads",
            "n_slots",
            "max_seq_len",
            "dropout",
            "vocab_size",
        ):
            if key in c:
                cfg[key] = c[key]

    if "position_embedding.weight" in state_dict and "max_seq_len" not in cfg:
        cfg["max_seq_len"] = int(state_dict["position_embedding.weight"].shape[0])
    if "token_embedding.weight" in state_dict and "vocab_size" not in cfg:
        cfg["vocab_size"] = int(state_dict["token_embedding.weight"].shape[0])
    if "token_embedding.weight" in state_dict and "d_model" not in cfg:
        cfg["d_model"] = int(state_dict["token_embedding.weight"].shape[1])
    if "slot_embedding" in state_dict and "n_slots" not in cfg:
        cfg["n_slots"] = int(state_dict["slot_embedding"].shape[1])

    block_prefixes = {
        k.split(".")[0] for k in state_dict.keys() if k.startswith("blocks.")
    }
    if "n_layers" not in cfg and block_prefixes:
        cfg["n_layers"] = int(len(block_prefixes))

    if "blocks.0.attn.qkv_proj.weight" in state_dict and "n_heads" not in cfg:
        # cannot recover n_heads reliably from qkv shape alone
        pass

    return cfg


def _load_checkpoint_weights(
    model: SSASlotDecoder,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = ckpt["model_state_dict"]
    model_state = model.state_dict()

    filtered_state: Dict[str, Any] = {}
    skipped_keys: List[str] = []
    resized_keys: List[str] = []

    for key, param in state_dict.items():
        if key.startswith("verify_head"):
            skipped_keys.append(
                f"{key} (skipped: legacy verify_head from non-CBV architecture)"
            )
            continue

        if key not in model_state:
            skipped_keys.append(str(key))
            continue

        model_shape = model_state[key].shape
        ckpt_shape = param.shape

        if model_shape == ckpt_shape:
            filtered_state[key] = param
        elif (
            key in ("token_embedding.weight", "lm_head.weight")
            and len(model_shape) == 2
            and len(ckpt_shape) == 2
            and model_shape[1] == ckpt_shape[1]
        ):
            new_param = model_state[key].clone()
            min_vocab = min(int(model_shape[0]), int(ckpt_shape[0]))
            new_param[:min_vocab] = param[:min_vocab]
            filtered_state[key] = new_param
            resized_keys.append(f"{key}: {tuple(ckpt_shape)} -> {tuple(model_shape)}")
        else:
            skipped_keys.append(
                f"{key} (shape mismatch: ckpt={tuple(ckpt_shape)}, model={tuple(model_shape)})"
            )

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    if skipped_keys:
        logger.warning("Skipped checkpoint keys (incompatible): %s", skipped_keys)
    if resized_keys:
        logger.info("Resized checkpoint keys: %s", resized_keys)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    if missing:
        logger.info(
            "Missing keys (expected for CBV init): %s",
            [k for k in missing if "verify" in k or "polarity" in k],
        )

    return {
        "checkpoint": str(checkpoint_path),
        "missing_keys": [str(k) for k in missing],
        "skipped_keys": skipped_keys,
        "resized_keys": resized_keys,
    }


def _configure_trainable_parameters(
    model: SSASlotDecoder, freeze_backbone: bool
) -> int:
    if not bool(freeze_backbone):
        count = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        logger.info("trainable_params=%d (freeze_backbone=False)", int(count))
        return int(count)

    for name, param in model.named_parameters():
        if "verify_head" in name or "polarity_embedding" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    count = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    logger.info("trainable_params=%d (freeze_backbone=True)", int(count))
    return int(count)


def _configure_trainable_parameters_pbp(
    model: SSASlotDecoder,
    viability_head: ViabilityHead,
    freeze_backbone: bool,
) -> int:
    if bool(freeze_backbone):
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    for param in viability_head.parameters():
        param.requires_grad = True

    count = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    count += sum(int(p.numel()) for p in viability_head.parameters() if p.requires_grad)
    logger.info(
        "trainable_params=%d (pbp_mode=True freeze_backbone=%s)",
        int(count),
        str(bool(freeze_backbone)),
    )
    return int(count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CBV verifier from SAT traces")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pbp_data_path", type=str, default="")
    parser.add_argument("--pbp_mode", action="store_true")
    parser.add_argument("--rank_margin", type=float, default=0.5)
    parser.add_argument("--rank_weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--verify_weight", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_slots", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=1500)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_branch_slots", type=int, default=12)
    parser.add_argument("--n_verifier_slots", type=int, default=8)
    parser.add_argument("--mask_mode", type=str, default="selective_ssa")
    parser.add_argument("--position_mode", type=str, default="auto")
    parser.add_argument(
        "--block_mode",
        type=str,
        choices=["all_zero", "prefix_state"],
        default="all_zero",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--polarity_swap", action="store_true")
    args = parser.parse_args()

    _set_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SATInterleavedTokenizer()
    tokenizer_vocab_size = int(tokenizer.VOCAB_SIZE)
    device = torch.device(str(args.device))
    use_pbp_mode = bool(args.pbp_mode or str(args.pbp_data_path).strip())

    if use_pbp_mode:
        pbp_examples, pbp_meta = _load_pbp_examples(
            Path(str(args.pbp_data_path).strip())
        )
        random.Random(int(args.seed)).shuffle(pbp_examples)
        split = int(round((1.0 - float(args.val_split)) * len(pbp_examples)))
        split = max(1, min(split, len(pbp_examples) - 1))
        train_ds = PBPViabilityDataset(
            pbp_examples[:split],
            max_seq_len=int(args.max_seq_len),
            vocab_size=int(tokenizer_vocab_size),
            swap_augment=True,
        )
        val_ds = PBPViabilityDataset(
            pbp_examples[split:],
            max_seq_len=int(args.max_seq_len),
            vocab_size=int(tokenizer_vocab_size),
            swap_augment=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=lambda b: _collate_pbp_batch(b, pad_token=int(tokenizer.PAD)),
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=lambda b: _collate_pbp_batch(b, pad_token=int(tokenizer.PAD)),
            num_workers=0,
        )
    else:
        if not str(args.data_path).strip():
            raise ValueError("--data_path is required when --pbp_mode is disabled")
        records = _load_records(Path(args.data_path))
        random.Random(int(args.seed)).shuffle(records)
        split = int(round((1.0 - float(args.val_split)) * len(records)))
        split = max(1, min(split, len(records) - 1))
        train_records = records[:split]
        val_records = records[split:]
        train_ds = CBVVerifyDataset(
            train_records,
            max_seq_len=int(args.max_seq_len),
            vocab_size=int(tokenizer_vocab_size),
            tokenizer=tokenizer,
            polarity_swap=bool(args.polarity_swap),
        )
        val_ds = CBVVerifyDataset(
            val_records,
            max_seq_len=int(args.max_seq_len),
            vocab_size=int(tokenizer_vocab_size),
            tokenizer=tokenizer,
            polarity_swap=False,
        )
        collate = lambda batch: _collate_cbv_batch(  # noqa: E731
            batch,
            pad_token=int(tokenizer.PAD),
            block_mode=str(args.block_mode),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty examples: train={len(train_ds)} val={len(val_ds)}")

    model_cfg: Dict[str, Any] = {
        "vocab_size": int(tokenizer_vocab_size),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "max_seq_len": int(args.max_seq_len),
        "n_slots": int(args.n_slots),
        "dropout": float(args.dropout),
    }
    ckpt_meta: Dict[str, Any] = {"loaded": False}

    checkpoint_path = str(args.checkpoint).strip()
    if checkpoint_path:
        inferred = _infer_model_cfg_from_checkpoint(Path(checkpoint_path))
        for key in ("d_model", "n_layers", "n_slots", "max_seq_len", "dropout"):
            if key in inferred:
                model_cfg[key] = inferred[key]
        if "vocab_size" in inferred:
            model_cfg["vocab_size"] = int(inferred["vocab_size"])

    if not use_pbp_mode:
        expected_slots = int(2 * int(args.n_branch_slots) + int(args.n_verifier_slots))
        model_cfg["n_slots"] = int(expected_slots)

    model = SSASlotDecoder(
        vocab_size=int(model_cfg["vocab_size"]),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        n_slots=int(model_cfg["n_slots"]),
        dropout=float(model_cfg["dropout"]),
        cbv_enabled=(not use_pbp_mode),
        n_branch_slots=int(args.n_branch_slots),
        n_verifier_slots=int(args.n_verifier_slots),
    )
    if checkpoint_path:
        ckpt_meta = _load_checkpoint_weights(model, Path(checkpoint_path))
        ckpt_meta["loaded"] = True
    model = model.to(device)

    viability_head: ViabilityHead | None = None
    if use_pbp_mode:
        viability_head = ViabilityHead(d_model=int(model_cfg["d_model"])).to(device)
        trainable_params = _configure_trainable_parameters_pbp(
            model, viability_head, freeze_backbone=bool(args.freeze_backbone)
        )
        opt_params = [p for p in model.parameters() if p.requires_grad] + [
            p for p in viability_head.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            opt_params, lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
    else:
        use_lm_loss = not bool(checkpoint_path)
        lm_weight = 1.0 if bool(use_lm_loss) else 0.0
        trainable_params = _configure_trainable_parameters(
            model, freeze_backbone=bool(args.freeze_backbone)
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

    logger.info("train_dataset_stats=%s", json.dumps(train_ds.stats, indent=2))
    logger.info("val_dataset_stats=%s", json.dumps(val_ds.stats, indent=2))
    if use_pbp_mode:
        logger.info(
            "pbp_input_stats=%s", json.dumps(pbp_meta.get("stats", {}), indent=2)
        )

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        if use_pbp_mode:
            assert viability_head is not None
            train_stats = _run_epoch_pbp(
                loader=train_loader,
                model=model,
                viability_head=viability_head,
                optimizer=optimizer,
                device=device,
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
                rank_margin=float(args.rank_margin),
                rank_weight=float(args.rank_weight),
                train=True,
            )
            val_stats = _run_epoch_pbp(
                loader=val_loader,
                model=model,
                viability_head=viability_head,
                optimizer=None,
                device=device,
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
                rank_margin=float(args.rank_margin),
                rank_weight=float(args.rank_weight),
                train=False,
            )
            row = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_stats["loss"]),
                "train_bce_loss": float(train_stats["bce_loss"]),
                "train_rank_loss": float(train_stats["rank_loss"]),
                "train_state_auroc": float(train_stats["state_auroc"]),
                "train_state_auprc": float(train_stats["state_auprc"]),
                "train_force_precision": float(train_stats["force_precision"]),
                "train_force_recall": float(train_stats["force_recall"]),
                "train_force_coverage": float(train_stats["force_coverage"]),
                "train_both_viable_false_force_rate": float(
                    train_stats["both_viable_false_force_rate"]
                ),
                "val_loss": float(val_stats["loss"]),
                "val_bce_loss": float(val_stats["bce_loss"]),
                "val_rank_loss": float(val_stats["rank_loss"]),
                "val_state_auroc": float(val_stats["state_auroc"]),
                "val_state_auprc": float(val_stats["state_auprc"]),
                "val_force_precision": float(val_stats["force_precision"]),
                "val_force_recall": float(val_stats["force_recall"]),
                "val_force_coverage": float(val_stats["force_coverage"]),
                "val_both_viable_false_force_rate": float(
                    val_stats["both_viable_false_force_rate"]
                ),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        else:
            use_lm_loss = not bool(checkpoint_path)
            lm_weight = 1.0 if bool(use_lm_loss) else 0.0
            train_stats = _run_epoch(
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                device=device,
                tokenizer=tokenizer,
                verify_weight=float(args.verify_weight),
                lm_weight=float(lm_weight),
                use_lm_loss=bool(use_lm_loss),
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
                train=True,
            )
            val_stats = _run_epoch(
                loader=val_loader,
                model=model,
                optimizer=None,
                device=device,
                tokenizer=tokenizer,
                verify_weight=float(args.verify_weight),
                lm_weight=float(lm_weight),
                use_lm_loss=bool(use_lm_loss),
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
                train=False,
            )
            row = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_stats["loss"]),
                "train_verify_loss": float(train_stats["verify_loss"]),
                "train_lm_loss": float(train_stats["lm_loss"]),
                "train_branch_acc": float(train_stats["branch_acc"]),
                "train_4way_acc": float(train_stats["four_way_acc"]),
                "train_dead_end_recall": float(train_stats["dead_end_recall"]),
                "train_dead_end_precision": float(train_stats["dead_end_precision"]),
                "train_auroc_t": float(train_stats["auroc_t"]),
                "train_auroc_f": float(train_stats["auroc_f"]),
                "val_loss": float(val_stats["loss"]),
                "val_verify_loss": float(val_stats["verify_loss"]),
                "val_lm_loss": float(val_stats["lm_loss"]),
                "val_branch_acc": float(val_stats["branch_acc"]),
                "val_4way_acc": float(val_stats["four_way_acc"]),
                "val_dead_end_recall": float(val_stats["dead_end_recall"]),
                "val_dead_end_precision": float(val_stats["dead_end_precision"]),
                "val_auroc_t": float(val_stats["auroc_t"]),
                "val_auroc_f": float(val_stats["auroc_f"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        history.append(row)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": int(model_cfg["vocab_size"]),
                "d_model": int(model_cfg["d_model"]),
                "n_layers": int(model_cfg["n_layers"]),
                "n_heads": int(model_cfg["n_heads"]),
                "n_slots": int(model_cfg["n_slots"]),
                "max_seq_len": int(model_cfg["max_seq_len"]),
                "dropout": float(model_cfg["dropout"]),
                "cbv_enabled": bool(not use_pbp_mode),
                "pbp_mode": bool(use_pbp_mode),
                "rank_margin": float(args.rank_margin),
                "rank_weight": float(args.rank_weight),
            },
            "epoch": int(epoch + 1),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "history": history,
            "train_dataset_stats": train_ds.stats,
            "val_dataset_stats": val_ds.stats,
            "ckpt_meta": ckpt_meta,
            "trainable_params": int(trainable_params),
        }
        if viability_head is not None:
            checkpoint["viability_head_state_dict"] = viability_head.state_dict()

        torch.save(checkpoint, output_dir / "last.pt")
        if float(val_stats["loss"]) < float(best_val_loss):
            best_val_loss = float(val_stats["loss"])
            torch.save(checkpoint, output_dir / "best.pt")

    summary: Dict[str, Any] = {
        "data_path": str(args.data_path),
        "pbp_data_path": str(args.pbp_data_path),
        "pbp_mode": bool(use_pbp_mode),
        "checkpoint": str(checkpoint_path),
        "train_examples": int(len(train_ds)),
        "val_examples": int(len(val_ds)),
        "epochs": int(args.epochs),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "train_dataset_stats": train_ds.stats,
        "val_dataset_stats": val_ds.stats,
        "model_cfg": model_cfg,
        "ckpt_meta": ckpt_meta,
        "trainable_params": int(trainable_params),
    }
    if use_pbp_mode:
        summary["pbp_input_stats"] = pbp_meta.get("stats", {})
    else:
        summary["train_records"] = int(len(train_records))
        summary["val_records"] = int(len(val_records))

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "training_complete mode=%s best_val_loss=%.4f output_dir=%s",
        "pbp" if use_pbp_mode else "cbv",
        float(best_val_loss),
        str(output_dir),
    )


if __name__ == "__main__":
    main()
