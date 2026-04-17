#!/usr/bin/env python3
"""Shared utilities for SAT length-OOD experiments.

This module is intentionally light on policy: training, evaluation, and plotting
scripts build on the parsing and aggregation helpers here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sat.interleaved_tokenizer import SATInterleavedTokenizer


logger = logging.getLogger(__name__)


DEFAULT_HELD_OUT_INTERSECTION_TRAIN_SEEDS = (42, 123, 456)
VALID_POSITION_MODES = ("auto", "standard", "block_relative")


STATE_TOKEN = int(SATInterleavedTokenizer.STATE)
SEARCH_START_TOKEN = int(SATInterleavedTokenizer.SEARCH_START)
SEP_TOKEN = int(SATInterleavedTokenizer.SEP)
COLON_TOKEN = int(SATInterleavedTokenizer.COLON)
TRUE_TOKEN = int(SATInterleavedTokenizer.TRUE_VAL)
FALSE_TOKEN = int(SATInterleavedTokenizer.FALSE_VAL)
VAR_OFFSET = int(SATInterleavedTokenizer.VAR_OFFSET)
POS_LIT_OFFSET = int(SATInterleavedTokenizer.POS_LIT_OFFSET)
NEG_LIT_OFFSET = int(SATInterleavedTokenizer.NEG_LIT_OFFSET)
CLAUSE_OFFSET = int(SATInterleavedTokenizer.CLAUSE_OFFSET)
LEVEL_OFFSET = int(SATInterleavedTokenizer.LEVEL_OFFSET)


@dataclass(frozen=True)
class BlockSpan:
    block_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return int(self.end - self.start)


@dataclass(frozen=True)
class TraceBlock:
    trace_index: int
    block_id: int
    decision_index: int
    start: int
    end: int
    target_positions: Tuple[int, ...]
    target_tokens: Tuple[int, ...]
    first_target_pos: int
    pre_action_tokens: Tuple[int, ...]
    visible_context_len: int

    @property
    def target_count(self) -> int:
        return int(len(self.target_positions))

    @property
    def is_assignment_block(self) -> bool:
        if len(self.target_tokens) < 2:
            return False
        first = int(self.target_tokens[0])
        second = int(self.target_tokens[1])
        return _is_var_token(first) and second in (TRUE_TOKEN, FALSE_TOKEN)

    @property
    def candidate_vars(self) -> Tuple[int, ...]:
        vars_in_state: List[int] = []
        tokens = list(self.pre_action_tokens)
        if not tokens or int(tokens[0]) != STATE_TOKEN:
            return tuple()
        idx = 1
        while idx < len(tokens) and int(tokens[idx]) != SEP_TOKEN:
            tok = int(tokens[idx])
            if _is_var_token(tok):
                vars_in_state.append(tok)
            idx += 1
        return tuple(vars_in_state)


@dataclass(frozen=True)
class SATTraceExample:
    trace_index: int
    instance_id: str
    clauses: Tuple[Tuple[int, ...], ...]
    num_vars: int
    sequence: Tuple[int, ...]
    loss_mask: Tuple[bool, ...]
    block_ids: Tuple[int, ...]
    prefix_len: int
    block_spans: Tuple[BlockSpan, ...]
    supervised_blocks: Tuple[TraceBlock, ...]
    meta: Dict[str, Any]

    @property
    def num_blocks(self) -> int:
        return int(max(self.block_ids)) if self.block_ids else 0

    @property
    def num_supervised_blocks(self) -> int:
        return int(len(self.supervised_blocks))

    @property
    def full_length(self) -> int:
        return int(len(self.sequence))

    def length_through_block(self, max_block_id: int) -> int:
        return int(block_truncation_cutoff(self.block_ids, int(max_block_id)))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _is_var_token(token_id: int) -> bool:
    token = int(token_id)
    return int(VAR_OFFSET) <= token < int(SATInterleavedTokenizer.VOCAB_SIZE)


def _token_to_literal(token_id: int) -> int:
    token = int(token_id)
    if POS_LIT_OFFSET <= token < NEG_LIT_OFFSET:
        return int((token - POS_LIT_OFFSET) + 1)
    if NEG_LIT_OFFSET <= token < CLAUSE_OFFSET:
        return -int((token - NEG_LIT_OFFSET) + 1)
    raise ValueError(f"token {token} is not a literal token")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_trace_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("traces"), list):
        return [dict(item) for item in payload["traces"]]
    raise ValueError(f"unsupported trace payload type at {path}: {type(payload)}")


def load_checkpoint_with_sidecar_config(
    checkpoint_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    sidecar_config: Dict[str, Any] = {}
    sidecar_path = checkpoint_path.parent / "config.json"
    if sidecar_path.exists():
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            sidecar_config = dict(payload)

    checkpoint_config = checkpoint.get("config", {})
    merged_config: Dict[str, Any] = {}
    if isinstance(sidecar_config, dict):
        merged_config.update(sidecar_config)
    if isinstance(checkpoint_config, dict):
        merged_config.update(checkpoint_config)
    return checkpoint, merged_config


def was_cli_flag_explicit(argv: Sequence[str], flag: str) -> bool:
    normalized = str(flag)
    return any(
        str(arg) == normalized or str(arg).startswith(f"{normalized}=") for arg in argv
    )


def get_training_position_mode(config: Dict[str, Any]) -> str | None:
    raw_value = config.get("training_position_mode")
    if raw_value is None:
        return None
    value = str(raw_value)
    if value not in VALID_POSITION_MODES:
        logger.warning(
            "ignoring invalid training_position_mode=%s; expected one of %s",
            value,
            sorted(VALID_POSITION_MODES),
        )
        return None
    return value


def resolve_eval_position_mode(
    *,
    checkpoint_path: Path,
    checkpoint_config: Dict[str, Any],
    cli_position_mode: str,
    cli_flag_explicit: bool,
) -> Dict[str, Any]:
    requested = str(cli_position_mode)
    if requested not in VALID_POSITION_MODES:
        raise ValueError(
            f"unknown position_mode='{requested}', expected one of {sorted(VALID_POSITION_MODES)}"
        )

    training_position_mode = get_training_position_mode(checkpoint_config)
    if training_position_mode is not None:
        logger.info(
            "checkpoint_training_position_mode checkpoint=%s training_position_mode=%s",
            str(checkpoint_path),
            str(training_position_mode),
        )

    if not bool(cli_flag_explicit) and training_position_mode is not None:
        effective = str(training_position_mode)
        source = "checkpoint_training_position_mode"
    else:
        effective = requested
        source = "cli"

    logger.info(
        "checkpoint_eval_position_mode checkpoint=%s requested=%s explicit_cli=%s effective=%s source=%s",
        str(checkpoint_path),
        requested,
        bool(cli_flag_explicit),
        effective,
        source,
    )
    return {
        "requested_position_mode": requested,
        "effective_position_mode": effective,
        "training_position_mode": training_position_mode,
        "source": source,
    }


def _resolve_checkpoint_reference(reference: str, *, anchor: Path) -> Path:
    candidate = Path(str(reference))
    if candidate.is_absolute() and candidate.exists():
        return candidate

    anchored = (anchor / candidate).resolve()
    if anchored.exists():
        return anchored

    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return candidate if candidate.is_absolute() else anchored


def instantiate_model_from_checkpoint_config(
    *,
    checkpoint_path: Path,
    config: Dict[str, Any],
    state_dict: Dict[str, Any],
) -> torch.nn.Module:
    model_type = str(config.get("model_type", "transformer"))
    vocab_size = int(
        config.get("vocab_size", state_dict["token_embedding.weight"].shape[0])
    )
    d_model = int(config.get("d_model", 256))
    max_seq_len = int(
        config.get("max_seq_len", state_dict["position_embedding.weight"].shape[0])
    )
    n_slots = int(config.get("n_slots", 32))
    dropout = float(config.get("dropout", 0.1))

    if model_type == "lstm":
        from universal.lstm_decoder import LSTMDecoder

        return LSTMDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            hidden_size=int(config.get("hidden_size", d_model)),
            n_lstm_layers=int(config.get("n_lstm_layers", config.get("n_layers", 6))),
            max_seq_len=int(max_seq_len),
            n_slots=int(n_slots),
            dropout=float(dropout),
            block_mode=str(config.get("block_mode", "continuous")),
        )

    from universal.ssa_decoder import SSASlotDecoder

    return SSASlotDecoder(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_layers=int(config.get("n_layers", 6)),
        n_heads=int(config.get("n_heads", 8)),
        max_seq_len=int(max_seq_len),
        n_slots=int(n_slots),
        dropout=float(dropout),
    )


def reconstruct_initial_position_embedding(
    checkpoint_path: Path,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    checkpoint, merged_config = load_checkpoint_with_sidecar_config(checkpoint_path)
    state_dict = checkpoint["model_state_dict"]
    seed_value = merged_config.get("seed")
    if seed_value is None:
        logger.warning(
            "checkpoint %s missing seed metadata; falling back to seed=42 for init reconstruction",
            str(checkpoint_path),
        )
        seed_value = 42
    seed = int(seed_value)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    model = instantiate_model_from_checkpoint_config(
        checkpoint_path=checkpoint_path,
        config=merged_config,
        state_dict=state_dict,
    )

    init_checkpoint = str(merged_config.get("init_checkpoint", "")).strip()
    init_source = "random_seed"
    if init_checkpoint:
        init_path = _resolve_checkpoint_reference(
            init_checkpoint, anchor=checkpoint_path.parent
        )
        if init_path.exists():
            init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
            if (
                not isinstance(init_payload, dict)
                or "model_state_dict" not in init_payload
            ):
                raise RuntimeError(
                    f"init checkpoint missing model_state_dict: {init_path}"
                )
            model_state = model.state_dict()
            filtered_state: Dict[str, torch.Tensor] = {}
            for key, value in init_payload["model_state_dict"].items():
                if key in model_state and tuple(value.shape) == tuple(
                    model_state[key].shape
                ):
                    filtered_state[key] = value
            model.load_state_dict(filtered_state, strict=False)
            init_source = str(init_path)
        else:
            logger.warning(
                "init checkpoint %s not found while reconstructing %s; using random seed init",
                init_checkpoint,
                str(checkpoint_path),
            )

    position_embedding = getattr(model, "position_embedding", None)
    if position_embedding is None:
        raise RuntimeError(
            f"model instantiated from {checkpoint_path} does not expose position_embedding"
        )
    weight = position_embedding.weight.detach().cpu().clone()
    return weight, {
        "seed": int(seed),
        "init_source": str(init_source),
        "max_seq_len": int(weight.shape[0]),
        "d_model": int(weight.shape[1]),
    }


def apply_position_row_ablation_(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    ablate_positions_above: int | None,
    ablate_mode: str,
    shuffle_seed: int,
) -> Dict[str, Any]:
    candidate = model
    if hasattr(candidate, "module") and getattr(candidate, "module") is not None:
        candidate = getattr(candidate, "module")

    position_embedding = getattr(candidate, "position_embedding", None)
    if position_embedding is None or not hasattr(position_embedding, "weight"):
        raise RuntimeError(
            "model does not expose position_embedding.weight for ablation"
        )

    total_rows = int(position_embedding.weight.shape[0])
    if ablate_positions_above is None:
        return {
            "enabled": False,
            "mode": "none",
            "threshold": None,
            "start_index": None,
            "stop_index_exclusive": None,
            "num_rows": 0,
            "suffix": "posabl_none",
        }

    threshold = int(ablate_positions_above)
    if threshold < 0:
        raise ValueError("ablate_positions_above must be >= 0")
    start_index = int(min(threshold + 1, total_rows))
    num_rows = int(max(total_rows - start_index, 0))
    metadata: Dict[str, Any] = {
        "enabled": True,
        "mode": str(ablate_mode),
        "threshold": int(threshold),
        "start_index": int(start_index),
        "stop_index_exclusive": int(total_rows),
        "num_rows": int(num_rows),
        "suffix": f"posabl_gt{int(threshold)}_{str(ablate_mode)}",
    }
    if num_rows <= 0:
        logger.warning(
            "position_ablation requested above=%d but model only has %d rows; no-op",
            int(threshold),
            int(total_rows),
        )
        metadata["enabled"] = False
        metadata["suffix"] = f"posabl_gt{int(threshold)}_noop"
        return metadata

    with torch.no_grad():
        if str(ablate_mode) == "zero":
            position_embedding.weight[start_index:].zero_()
        elif str(ablate_mode) == "shuffle":
            original = position_embedding.weight[start_index:].detach().cpu().clone()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(shuffle_seed))
            perm = torch.randperm(int(original.shape[0]), generator=generator)
            shuffled = original[perm].to(
                device=position_embedding.weight.device,
                dtype=position_embedding.weight.dtype,
            )
            position_embedding.weight[start_index:] = shuffled
        elif str(ablate_mode) == "reinit":
            init_weight, init_meta = reconstruct_initial_position_embedding(
                checkpoint_path
            )
            if int(init_weight.shape[0]) < int(total_rows):
                raise RuntimeError(
                    f"reconstructed init rows {init_weight.shape[0]} shorter than trained rows {total_rows}"
                )
            position_embedding.weight[start_index:] = init_weight[
                start_index:total_rows
            ].to(
                device=position_embedding.weight.device,
                dtype=position_embedding.weight.dtype,
            )
            metadata["reinit_source"] = dict(init_meta)
        else:
            raise ValueError(
                f"unknown ablate_mode='{ablate_mode}', expected one of ['reinit', 'shuffle', 'zero']"
            )

    logger.info(
        "position_ablation checkpoint=%s mode=%s threshold=%d rows=[%d,%d) num_rows=%d suffix=%s",
        str(checkpoint_path),
        str(ablate_mode),
        int(threshold),
        int(start_index),
        int(total_rows),
        int(num_rows),
        str(metadata["suffix"]),
    )
    return metadata


def _compute_train_val_indices(
    num_records: int,
    *,
    seed: int,
    val_split: float,
) -> Dict[str, Any]:
    if int(num_records) <= 0:
        return {
            "seed": int(seed),
            "train_indices": [],
            "val_indices": [],
            "split": 0,
        }
    if not 0.0 < float(val_split) < 1.0:
        raise ValueError(f"val_split must be in (0, 1); got {val_split}")
    shuffled = list(range(int(num_records)))
    random.Random(int(seed)).shuffle(shuffled)
    split = int(round((1.0 - float(val_split)) * float(num_records)))
    split = max(0, min(int(split), int(num_records)))
    return {
        "seed": int(seed),
        "train_indices": [int(idx) for idx in shuffled[:split]],
        "val_indices": [int(idx) for idx in shuffled[split:]],
        "split": int(split),
    }


def select_held_out_eval_records(
    records: Sequence[Dict[str, Any]],
    *,
    eval_mode: str,
    held_out_seed: int,
    val_split: float,
    max_eval_records: int = 0,
    intersection_train_seeds: Sequence[int] = DEFAULT_HELD_OUT_INTERSECTION_TRAIN_SEEDS,
    min_intersection_size: int = 100,
    tail_count: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    num_records = int(len(records))
    if num_records <= 0:
        raise RuntimeError("no trace records available for held-out evaluation")

    unique_seeds: List[int] = []
    for seed in [int(held_out_seed), *[int(x) for x in intersection_train_seeds]]:
        if int(seed) not in unique_seeds:
            unique_seeds.append(int(seed))

    split_payloads: Dict[int, Dict[str, Any]] = {
        int(seed): _compute_train_val_indices(
            num_records,
            seed=int(seed),
            val_split=float(val_split),
        )
        for seed in unique_seeds
    }
    held_out_payload = split_payloads[int(held_out_seed)]
    split_index = int(held_out_payload["split"])

    intersection_indices: List[int] = []
    if intersection_train_seeds:
        intersection_sets = [
            set(split_payloads[int(seed)]["val_indices"])
            for seed in intersection_train_seeds
        ]
        if intersection_sets:
            intersection_indices = sorted(set.intersection(*intersection_sets))

    effective_mode = str(eval_mode)
    warning_message: str | None = None
    if str(eval_mode) == "per-seed-val":
        base_eval_indices = [int(idx) for idx in held_out_payload["val_indices"]]
        num_train_excluded = int(split_index)
    elif str(eval_mode) == "intersection":
        if int(len(intersection_indices)) < int(min_intersection_size):
            effective_mode = "tail-500"
            start = max(0, int(num_records) - min(int(tail_count), int(num_records)))
            base_eval_indices = list(range(int(start), int(num_records)))
            warning_message = (
                "held_out_eval intersection too small "
                f"size={int(len(intersection_indices))} min_required={int(min_intersection_size)}; "
                f"falling back to tail-{int(len(base_eval_indices))} file-order traces"
            )
        else:
            base_eval_indices = [int(idx) for idx in intersection_indices]
        num_train_excluded = int(num_records - len(base_eval_indices))
    elif str(eval_mode) == "tail-500":
        start = max(0, int(num_records) - min(int(tail_count), int(num_records)))
        base_eval_indices = list(range(int(start), int(num_records)))
        num_train_excluded = int(num_records - len(base_eval_indices))
    else:
        raise ValueError(f"unsupported eval_mode={eval_mode}")

    if int(max_eval_records) > 0:
        eval_indices = [int(idx) for idx in base_eval_indices[: int(max_eval_records)]]
    else:
        eval_indices = [int(idx) for idx in base_eval_indices]

    eval_index_set = set(int(idx) for idx in eval_indices)
    seen_during_training_counts = {
        f"seed{int(seed)}": int(
            len(
                eval_index_set.intersection(
                    set(int(idx) for idx in split_payloads[int(seed)]["train_indices"])
                )
            )
        )
        for seed in intersection_train_seeds
    }

    metadata = {
        "requested_mode": str(eval_mode),
        "effective_mode": str(effective_mode),
        "held_out_seed": int(held_out_seed),
        "val_split": float(val_split),
        "num_total_records": int(num_records),
        "split_index": int(split_index),
        "num_train_excluded": int(num_train_excluded),
        "num_eval": int(len(eval_indices)),
        "eval_indices": [int(idx) for idx in eval_indices],
        "first_eval_indices": [int(idx) for idx in eval_indices[:10]],
        "intersection_train_seeds": [int(seed) for seed in intersection_train_seeds],
        "intersection_size": int(len(intersection_indices)),
        "min_intersection_size": int(min_intersection_size),
        "tail_count": int(min(int(tail_count), int(num_records))),
        "used_fallback": bool(
            str(eval_mode) == "intersection" and str(effective_mode) != "intersection"
        ),
        "warning": warning_message,
        "seen_during_training_counts": seen_during_training_counts,
        "train_val_sizes": {
            f"seed{int(seed)}": {
                "train": int(len(split_payloads[int(seed)]["train_indices"])),
                "val": int(len(split_payloads[int(seed)]["val_indices"])),
            }
            for seed in unique_seeds
        },
    }
    selected_records = [dict(records[int(idx)]) for idx in eval_indices]
    return selected_records, metadata


def compute_block_spans(block_ids: Sequence[int]) -> List[BlockSpan]:
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


def block_truncation_cutoff(block_ids: Sequence[int], max_block_id: int) -> int:
    if int(max_block_id) <= 0:
        return len(block_ids)
    for idx, block_id in enumerate(block_ids):
        if int(block_id) > int(max_block_id):
            return int(idx)
    return int(len(block_ids))


def truncate_record_to_max_blocks(
    record: Dict[str, Any],
    max_block_id: int,
    max_seq_len: int,
) -> Tuple[List[int], List[bool], List[int], int]:
    sequence = [int(x) for x in record["sequence"]][: int(max_seq_len)]
    loss_mask = [bool(x) for x in record["loss_mask"]][: int(max_seq_len)]
    if record.get("block_ids") is None:
        raise ValueError("truncate_record_to_max_blocks requires explicit block_ids")
    block_ids = [int(x) for x in record["block_ids"]][: int(max_seq_len)]
    cutoff = block_truncation_cutoff(block_ids, int(max_block_id))
    return (
        sequence[:cutoff],
        loss_mask[:cutoff],
        block_ids[:cutoff],
        int(cutoff),
    )


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray([float(x) for x in values], dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def recommend_max_train_blocks(
    records: Sequence[Dict[str, Any]],
    max_seq_len: int,
    target_p95_tokens: int = 2048,
    max_candidate_blocks: int = 64,
) -> Dict[str, Any]:
    best_block = 0
    best_p95 = 0.0
    rows: List[Dict[str, Any]] = []
    for block_budget in range(1, max(1, int(max_candidate_blocks)) + 1):
        lengths: List[int] = []
        for record in records:
            _seq, _lm, _blk, cutoff = truncate_record_to_max_blocks(
                record=record,
                max_block_id=int(block_budget),
                max_seq_len=int(max_seq_len),
            )
            lengths.append(int(cutoff))
        p95 = float(np.percentile(np.asarray(lengths, dtype=np.float64), 95))
        rows.append(
            {
                "max_train_blocks": int(block_budget),
                "p95_tokens": float(p95),
                "mean_tokens": float(np.mean(lengths)),
                "max_tokens": float(np.max(lengths)),
            }
        )
        if p95 <= float(target_p95_tokens):
            best_block = int(block_budget)
            best_p95 = float(p95)
    return {
        "target_p95_tokens": int(target_p95_tokens),
        "recommended_max_train_blocks": int(best_block),
        "recommended_p95_tokens": float(best_p95),
        "scan": rows,
    }


def compute_length_stats(
    records: Sequence[Dict[str, Any]],
    max_seq_len: int,
    max_train_blocks: int = 0,
) -> Dict[str, Any]:
    token_lengths: List[int] = []
    supervised_tokens: List[int] = []
    num_blocks: List[int] = []
    truncated_lengths: List[int] = []
    truncated_supervised: List[int] = []

    for record in records:
        sequence = [int(x) for x in record["sequence"]][: int(max_seq_len)]
        loss_mask = [bool(x) for x in record["loss_mask"]][: int(max_seq_len)]
        if record.get("block_ids") is None:
            raise ValueError("compute_length_stats requires explicit block_ids")
        block_ids = [int(x) for x in record["block_ids"]][: int(max_seq_len)]
        token_lengths.append(int(len(sequence)))
        supervised_tokens.append(int(sum(loss_mask)))
        num_blocks.append(int(max(block_ids)) if block_ids else 0)
        if int(max_train_blocks) > 0:
            cutoff = block_truncation_cutoff(block_ids, int(max_train_blocks))
            truncated_lengths.append(int(cutoff))
            truncated_supervised.append(int(sum(bool(x) for x in loss_mask[:cutoff])))

    payload: Dict[str, Any] = {
        "n_records": int(len(records)),
        "tokens_per_trace": summarize_numeric(token_lengths),
        "supervised_tokens_per_trace": summarize_numeric(supervised_tokens),
        "num_blocks_per_trace": summarize_numeric(num_blocks),
        "recommended_short_context": recommend_max_train_blocks(
            records=records,
            max_seq_len=int(max_seq_len),
        ),
    }
    if int(max_train_blocks) > 0:
        payload["truncation"] = {
            "max_train_blocks": int(max_train_blocks),
            "tokens_per_trace": summarize_numeric(truncated_lengths),
            "supervised_tokens_per_trace": summarize_numeric(truncated_supervised),
            "truncated_p95_length": float(
                np.percentile(np.asarray(truncated_lengths, dtype=np.float64), 95)
            )
            if truncated_lengths
            else 0.0,
        }
    return payload


def parse_clause_prefix(
    sequence: Sequence[int],
) -> Tuple[List[Tuple[int, ...]], int, int]:
    tokens = [int(x) for x in sequence]
    if len(tokens) < 3:
        raise ValueError("sequence too short to contain SAT prefix")
    if int(tokens[0]) != int(SATInterleavedTokenizer.BOS):
        raise ValueError("expected BOS at position 0")
    if int(tokens[1]) != int(SATInterleavedTokenizer.CLAUSE_START):
        raise ValueError("expected CLAUSE_START at position 1")

    clauses: List[Tuple[int, ...]] = []
    max_var = -1
    idx = 2
    while idx < len(tokens):
        token = int(tokens[idx])
        if token == SEARCH_START_TOKEN:
            prefix_len = int(idx + 1)
            return clauses, int(max_var + 1), int(prefix_len)
        if not (CLAUSE_OFFSET <= token < LEVEL_OFFSET):
            raise ValueError(f"expected clause token at position {idx}, got {token}")
        idx += 1
        if idx >= len(tokens) or int(tokens[idx]) != COLON_TOKEN:
            raise ValueError(f"expected COLON after clause token at position {idx}")
        idx += 1
        clause: List[int] = []
        while idx < len(tokens) and int(tokens[idx]) != SEP_TOKEN:
            lit = _token_to_literal(int(tokens[idx]))
            clause.append(int(lit))
            max_var = max(max_var, abs(int(lit)) - 1)
            idx += 1
        if idx >= len(tokens):
            raise ValueError("unterminated clause in SAT prefix")
        clauses.append(tuple(clause))
        idx += 1
    raise ValueError("SEARCH_START token not found in trace")


def trace_instance_id(sequence: Sequence[int]) -> str:
    prefix_end = sequence.index(SEARCH_START_TOKEN) + 1
    prefix_bytes = json.dumps([int(x) for x in sequence[:prefix_end]]).encode("utf-8")
    return hashlib.sha1(prefix_bytes).hexdigest()


def extract_trace_examples(records: Sequence[Dict[str, Any]]) -> List[SATTraceExample]:
    examples: List[SATTraceExample] = []
    for trace_index, record in enumerate(records):
        sequence = tuple(int(x) for x in record["sequence"])
        loss_mask = tuple(bool(x) for x in record["loss_mask"])
        if record.get("block_ids") is None:
            raise ValueError(f"trace {trace_index} missing block_ids")
        block_ids = tuple(int(x) for x in record["block_ids"])
        if len(sequence) != len(loss_mask) or len(sequence) != len(block_ids):
            raise ValueError(f"trace {trace_index} token/loss/block length mismatch")

        clauses, inferred_num_vars, prefix_len = parse_clause_prefix(sequence)
        meta = dict(record.get("meta", {}))
        num_vars = int(meta.get("num_vars", inferred_num_vars))
        spans = tuple(compute_block_spans(block_ids))
        supervised_blocks: List[TraceBlock] = []
        for span in spans:
            if int(span.block_id) <= 0:
                continue
            target_positions = tuple(
                int(pos)
                for pos in range(int(span.start), int(span.end))
                if bool(loss_mask[int(pos)])
            )
            if not target_positions:
                continue
            first_target_pos = int(target_positions[0])
            supervised_blocks.append(
                TraceBlock(
                    trace_index=int(trace_index),
                    block_id=int(span.block_id),
                    decision_index=int(span.block_id),
                    start=int(span.start),
                    end=int(span.end),
                    target_positions=target_positions,
                    target_tokens=tuple(int(sequence[pos]) for pos in target_positions),
                    first_target_pos=int(first_target_pos),
                    pre_action_tokens=tuple(
                        int(x)
                        for x in sequence[int(span.start) : int(first_target_pos)]
                    ),
                    visible_context_len=int(first_target_pos),
                )
            )

        examples.append(
            SATTraceExample(
                trace_index=int(trace_index),
                instance_id=str(trace_instance_id(sequence)),
                clauses=tuple(tuple(int(x) for x in clause) for clause in clauses),
                num_vars=int(num_vars),
                sequence=sequence,
                loss_mask=loss_mask,
                block_ids=block_ids,
                prefix_len=int(prefix_len),
                block_spans=spans,
                supervised_blocks=tuple(supervised_blocks),
                meta=meta,
            )
        )
    return examples


def trace_examples_to_instances(
    examples: Sequence[SATTraceExample],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                "clauses": [
                    tuple(int(x) for x in clause) for clause in example.clauses
                ],
                "num_vars": int(example.num_vars),
                "planted_solution": None,
                "instance_id": str(example.instance_id),
                "trace_index": int(example.trace_index),
                "gold_num_blocks": int(example.num_blocks),
                "gold_num_supervised_blocks": int(example.num_supervised_blocks),
            }
        )
    return rows


def block_budget_to_token_budget(
    examples: Sequence[SATTraceExample],
    block_budget: int,
    percentile: float = 95.0,
) -> Dict[str, float]:
    lengths = [
        float(example.length_through_block(int(block_budget)))
        for example in examples
        if example.num_blocks > 0
    ]
    if not lengths:
        return {
            "block_budget": float(block_budget),
            "token_budget_p95": 0.0,
            "token_budget_max": 0.0,
            "token_budget_mean": 0.0,
        }
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "block_budget": float(block_budget),
        "token_budget_p95": float(np.percentile(arr, percentile)),
        "token_budget_max": float(np.max(arr)),
        "token_budget_mean": float(np.mean(arr)),
    }


def forward_model_logits(
    *,
    model: torch.nn.Module,
    input_ids: Sequence[int],
    block_ids: Sequence[int],
    mask_mode: str,
    position_mode: str,
    device: torch.device,
) -> torch.Tensor:
    tokens = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    blocks = torch.tensor([list(block_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tokens, dtype=torch.long)
    lm_logits, _verify_logits = model(
        tokens,
        attention_mask=attention_mask,
        block_ids=blocks,
        mask_mode=str(mask_mode),
        position_mode=str(position_mode),
    )
    return lm_logits[0]


def resolve_model_max_seq_len(model: torch.nn.Module) -> int:
    candidate = model
    if hasattr(candidate, "module") and getattr(candidate, "module") is not None:
        candidate = getattr(candidate, "module")

    max_seq_len = getattr(candidate, "max_seq_len", None)
    if max_seq_len is not None:
        return int(max_seq_len)

    position_embedding = getattr(candidate, "position_embedding", None)
    if position_embedding is not None and hasattr(position_embedding, "num_embeddings"):
        return int(position_embedding.num_embeddings)

    raise AttributeError("unable to resolve model max_seq_len")


def evaluate_teacher_forced_block(
    *,
    model: torch.nn.Module,
    example: SATTraceExample,
    block: TraceBlock,
    mask_mode: str,
    position_mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    prefix_tokens = [int(x) for x in example.sequence[: int(block.end)]]
    prefix_blocks = [int(x) for x in example.block_ids[: int(block.end)]]
    input_len = int(len(prefix_tokens))
    model_max_seq_len = int(resolve_model_max_seq_len(model))
    if int(input_len) > int(model_max_seq_len):
        logger.info(
            "teacher_forced_overflow_skip trace=%d instance=%s decision=%d input_len=%d model_max_seq_len=%d visible_context_len=%d",
            int(example.trace_index),
            str(example.instance_id),
            int(block.decision_index),
            int(input_len),
            int(model_max_seq_len),
            int(block.visible_context_len),
        )
        return {
            "trace_index": int(example.trace_index),
            "instance_id": str(example.instance_id),
            "block_id": int(block.block_id),
            "decision_index": int(block.decision_index),
            "visible_context_len": int(block.visible_context_len),
            "input_len": int(input_len),
            "model_max_seq_len": int(model_max_seq_len),
            "skipped_overflow": True,
            "token_acc": float("nan"),
            "action_exact_match": float("nan"),
            "nll": float("nan"),
            "target_count": int(block.target_count),
            "target_tokens": [int(x) for x in block.target_tokens],
            "per_token": [],
        }
    logits = forward_model_logits(
        model=model,
        input_ids=prefix_tokens,
        block_ids=prefix_blocks,
        mask_mode=str(mask_mode),
        position_mode=str(position_mode),
        device=device,
    )
    log_probs = F.log_softmax(logits, dim=-1)

    token_correct = 0
    token_nll = 0.0
    per_token: List[Dict[str, Any]] = []
    block_exact = True
    for pos in block.target_positions:
        if int(pos) <= 0:
            continue
        target = int(example.sequence[int(pos)])
        pred = int(torch.argmax(logits[int(pos) - 1]).item())
        nll = float(-log_probs[int(pos) - 1, int(target)].item())
        correct = int(pred == target)
        block_exact = bool(block_exact and correct == 1)
        token_correct += int(correct)
        token_nll += float(nll)
        per_token.append(
            {
                "position": int(pos),
                "target": int(target),
                "prediction": int(pred),
                "correct": int(correct),
                "nll": float(nll),
            }
        )

    token_count = max(len(per_token), 1)
    return {
        "trace_index": int(example.trace_index),
        "instance_id": str(example.instance_id),
        "block_id": int(block.block_id),
        "decision_index": int(block.decision_index),
        "visible_context_len": int(block.visible_context_len),
        "input_len": int(input_len),
        "model_max_seq_len": int(model_max_seq_len),
        "skipped_overflow": False,
        "token_acc": float(token_correct / token_count),
        "action_exact_match": float(1.0 if block_exact else 0.0),
        "nll": float(token_nll / token_count),
        "target_count": int(len(per_token)),
        "target_tokens": [int(x) for x in block.target_tokens],
        "per_token": per_token,
    }


def symmetric_kl_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> float:
    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    eps = 1e-12
    probs_a = torch.clamp(probs_a, min=eps)
    probs_b = torch.clamp(probs_b, min=eps)
    kl_ab = torch.sum(probs_a * (torch.log(probs_a) - torch.log(probs_b)))
    kl_ba = torch.sum(probs_b * (torch.log(probs_b) - torch.log(probs_a)))
    return float(0.5 * (kl_ab.item() + kl_ba.item()))


def cosine_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            logits_a.unsqueeze(0),
            logits_b.unsqueeze(0),
            dim=-1,
            eps=1e-8,
        ).item()
    )


def build_context_bin_edges(token_budgets: Sequence[int]) -> List[Tuple[int, int, str]]:
    budgets = sorted({max(1, int(x)) for x in token_budgets})
    if not budgets:
        budgets = [2048, 4096, 8192]
    bins: List[Tuple[int, int, str]] = []
    low = 0
    for budget in budgets:
        bins.append((int(low), int(budget), f"{int(low)}-{int(budget)}"))
        low = int(budget) + 1
    bins.append((int(low), int(10**9), f">={int(low)}"))
    return bins


def assign_context_bin(
    visible_context_len: int,
    bins: Sequence[Tuple[int, int, str]],
) -> str:
    value = int(visible_context_len)
    for low, high, label in bins:
        if int(low) <= value <= int(high):
            return str(label)
    return str(bins[-1][2]) if bins else "all"


def aggregate_mean_rows(
    rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        bucket = grouped.setdefault(
            key,
            {
                **{k: row[k] for k in group_keys},
                "count": 0,
                "skipped_overflow_count": 0,
                **{f"sum::{name}": 0.0 for name in value_keys},
                **{f"valid::{name}": 0 for name in value_keys},
            },
        )
        bucket["count"] = int(bucket["count"]) + 1
        overflow_count = row.get("skipped_overflow_count")
        if overflow_count is None:
            overflow_count = int(bool(row.get("skipped_overflow", False)))
        bucket["skipped_overflow_count"] = int(bucket["skipped_overflow_count"]) + int(
            overflow_count
        )
        for name in value_keys:
            value = float(row[name])
            if math.isfinite(float(value)):
                bucket[f"sum::{name}"] = float(bucket[f"sum::{name}"]) + float(value)
                bucket[f"valid::{name}"] = int(bucket[f"valid::{name}"]) + 1

    out: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        count = max(int(bucket["count"]), 1)
        row = {k: bucket[k] for k in group_keys}
        row["count"] = int(count)
        row["skipped_overflow_count"] = int(bucket["skipped_overflow_count"])
        for name in value_keys:
            valid_count = int(bucket[f"valid::{name}"])
            if valid_count <= 0:
                row[name] = float("nan")
            else:
                row[name] = float(bucket[f"sum::{name}"] / float(valid_count))
        out.append(row)
    out.sort(key=lambda item: tuple(item[k] for k in group_keys))
    return out


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_label_group(label: str) -> str:
    raw = str(label)
    parts = raw.replace("/", "-").split("-")
    kept: List[str] = []
    for part in parts:
        lower = part.lower()
        if lower.startswith("seed"):
            continue
        kept.append(part)
    return "-".join(kept) if kept else raw


def format_step_budget_log(
    *,
    label: str,
    token_budget: int,
    solve_rate: float,
    deep_solve_rate: float,
    n_total: int,
    n_deep: int,
) -> str:
    return (
        f"label={label} token_budget={int(token_budget)} solve_rate={float(solve_rate):.4f} "
        f"deep_solve_rate={float(deep_solve_rate):.4f} n_total={int(n_total)} n_deep={int(n_deep)}"
    )


def round_token_budget(value: float, min_budget: int = 1) -> int:
    if not math.isfinite(float(value)):
        return int(min_budget)
    return int(max(int(min_budget), int(round(float(value)))))
