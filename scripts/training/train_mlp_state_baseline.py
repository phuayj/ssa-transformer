#!/usr/bin/env python3
"""Train fixed-feature SAT state-only MLP and linear baselines.

This script builds a reviewer-facing baseline that consumes the exact same
state-only SAT n=50 tokenized traces used by the transformer models. The model
never reads hidden oracle state: every feature is derived from the clause prefix,
the visible STATE block, and record metadata already present in the dataset.

Feature layout
--------------
Per-variable features are emitted for variables 0..num_vars-1 and flattened in
variable order. Each variable contributes 8 values:

1. status_unassigned      (one-hot)
2. status_true            (one-hot)
3. status_false           (one-hot)
4. status_newly_true      (one-hot)
5. status_newly_false     (one-hot)
6. status_free            (one-hot)
7. order_position_norm    (0 if absent, else normalized STATE-block position)
8. appears_in_state_block (binary)

Global features:

1. inferred_fraction_assigned
2. visible_fraction_unassigned
3. source_block_id_raw
4. source_block_id_norm
5. num_clauses_norm
6-11. normalized counts for each visible status type in the order above

Optional clause features (enabled by default here for the richer baseline):
for each clause slot 0..max_clause_features-1, append 3 values:

1. visible_satisfied_literal_fraction
2. visible_unsatisfied_literal_fraction
3. unresolved_literal_fraction

Important: clause features are computed only from *visible* STATE information.
Variables absent from the STATE block are treated as unresolved, so the model
does not receive any hidden assignment information.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.dsl import SatAction, SatActionType
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator
from sat.interleaved_tokenizer import SATInterleavedTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


STATUS_TOKENS: Tuple[int, ...] = (
    int(SATInterleavedTokenizer.UNASSIGNED),
    int(SATInterleavedTokenizer.TRUE_VAL),
    int(SATInterleavedTokenizer.FALSE_VAL),
    int(SATInterleavedTokenizer.NEWLY_TRUE),
    int(SATInterleavedTokenizer.NEWLY_FALSE),
    int(SATInterleavedTokenizer.FREE),
)
STATUS_NAMES: Tuple[str, ...] = (
    "unassigned",
    "true",
    "false",
    "newly_true",
    "newly_false",
    "free",
)
STATUS_TO_INDEX: Dict[int, int] = {
    int(tok): idx for idx, tok in enumerate(STATUS_TOKENS)
}
TRUE_LIKE_TOKENS = {
    int(SATInterleavedTokenizer.TRUE_VAL),
    int(SATInterleavedTokenizer.NEWLY_TRUE),
}
FALSE_LIKE_TOKENS = {
    int(SATInterleavedTokenizer.FALSE_VAL),
    int(SATInterleavedTokenizer.NEWLY_FALSE),
}


@dataclass(frozen=True)
class TargetInfo:
    continue_label: int  # 0=continue, 1=backtrack
    variable_label: int  # 0..num_vars-1 or -1 when not applicable
    value_label: int  # 0=False, 1=True, or -1 when not applicable
    clause_id: int  # conflict clause or -1
    backjump_level: int  # conflict level or -1
    solved_after_assignment: bool


@dataclass(frozen=True)
class SplitMetrics:
    loss: float
    continue_loss: float
    variable_loss: float
    value_loss: float
    continue_acc: float
    variable_acc: float
    value_acc: float
    joint_continue_acc: float
    backtrack_rate: float
    num_examples: int
    num_continue: int
    num_backtrack: int


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _mean_dict(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out: Dict[str, float] = {}
    for key in keys:
        out[key] = float(sum(float(row[key]) for row in rows) / len(rows))
    return out


def _parse_seed_list(seed: Optional[int], seeds: Optional[str]) -> List[int]:
    if seeds is not None and str(seeds).strip():
        return [int(part.strip()) for part in str(seeds).split(",") if part.strip()]
    if seed is not None:
        return [int(seed)]
    return [42]


def _literal_from_token(token_id: int, tokenizer: SATInterleavedTokenizer) -> int:
    tok = int(token_id)
    if tokenizer.POS_LIT_OFFSET <= tok < tokenizer.NEG_LIT_OFFSET:
        return int(tok - tokenizer.POS_LIT_OFFSET + 1)
    if tokenizer.NEG_LIT_OFFSET <= tok < tokenizer.CLAUSE_OFFSET:
        return -int(tok - tokenizer.NEG_LIT_OFFSET + 1)
    raise ValueError(f"token is not a literal token: {tok}")


def _var_from_token(token_id: int, tokenizer: SATInterleavedTokenizer) -> int:
    tok = int(token_id)
    if tokenizer.VAR_OFFSET <= tok < tokenizer.VAR_OFFSET + tokenizer.MAX_VARS:
        return int(tok - tokenizer.VAR_OFFSET)
    raise ValueError(f"token is not a variable token: {tok}")


def _clause_from_token(token_id: int, tokenizer: SATInterleavedTokenizer) -> int:
    tok = int(token_id)
    if tokenizer.CLAUSE_OFFSET <= tok < tokenizer.LEVEL_OFFSET:
        return int(tok - tokenizer.CLAUSE_OFFSET)
    raise ValueError(f"token is not a clause token: {tok}")


def _level_from_token(token_id: int, tokenizer: SATInterleavedTokenizer) -> int:
    tok = int(token_id)
    if tokenizer.LEVEL_OFFSET <= tok < tokenizer.VAR_OFFSET:
        return int(tok - tokenizer.LEVEL_OFFSET)
    raise ValueError(f"token is not a level token: {tok}")


def _parse_clause_prefix(
    sequence: Sequence[int],
    block_ids: Sequence[int],
    tokenizer: SATInterleavedTokenizer,
) -> List[Tuple[int, ...]]:
    prefix = [int(tok) for tok, block in zip(sequence, block_ids) if int(block) == 0]
    if not prefix:
        raise ValueError("empty prefix")

    try:
        start_idx = prefix.index(int(tokenizer.CLAUSE_START)) + 1
    except ValueError as exc:
        raise ValueError("prefix missing CLAUSE_START") from exc

    clauses: List[Tuple[int, ...]] = []
    i = int(start_idx)
    while i < len(prefix):
        tok = int(prefix[i])
        if tok == int(tokenizer.SEARCH_START):
            break
        _clause_id = _clause_from_token(tok, tokenizer)
        i += 1
        if i >= len(prefix) or int(prefix[i]) != int(tokenizer.COLON):
            raise ValueError("malformed clause prefix: missing COLON")
        i += 1
        clause: List[int] = []
        while i < len(prefix) and int(prefix[i]) != int(tokenizer.SEP):
            clause.append(_literal_from_token(int(prefix[i]), tokenizer))
            i += 1
        if i >= len(prefix):
            raise ValueError("malformed clause prefix: missing SEP")
        clauses.append(tuple(int(lit) for lit in clause))
        i += 1
    if not clauses:
        raise ValueError("no clauses parsed from prefix")
    return clauses


def _extract_block_tokens(
    sequence: Sequence[int],
    block_ids: Sequence[int],
) -> List[int]:
    return [int(tok) for tok, block in zip(sequence, block_ids) if int(block) == 1]


def _parse_state_block(
    block_tokens: Sequence[int],
    tokenizer: SATInterleavedTokenizer,
) -> Tuple[List[int], Dict[int, int]]:
    if not block_tokens:
        raise ValueError("empty block tokens")
    if int(block_tokens[0]) != int(tokenizer.STATE):
        raise ValueError("block does not start with STATE")
    try:
        sep_idx = list(int(tok) for tok in block_tokens).index(int(tokenizer.SEP))
    except ValueError as exc:
        raise ValueError("STATE block missing SEP") from exc

    state_tokens = [int(tok) for tok in block_tokens[1:sep_idx]]
    if len(state_tokens) % 2 != 0:
        raise ValueError("STATE block must contain var/status pairs")

    visible_order: List[int] = []
    visible_status: Dict[int, int] = {}
    for i in range(0, len(state_tokens), 2):
        var_id = _var_from_token(int(state_tokens[i]), tokenizer)
        status_tok = int(state_tokens[i + 1])
        if status_tok not in STATUS_TO_INDEX:
            raise ValueError(f"unknown status token in STATE block: {status_tok}")
        visible_order.append(int(var_id))
        visible_status[int(var_id)] = int(status_tok)
    return visible_order, visible_status


def _extract_target_info(
    sequence: Sequence[int],
    loss_mask: Sequence[bool],
    tokenizer: SATInterleavedTokenizer,
) -> TargetInfo:
    target_tokens = [int(tok) for tok, use in zip(sequence, loss_mask) if bool(use)]
    if not target_tokens:
        raise ValueError("record has no supervised target tokens")

    first = int(target_tokens[0])
    if tokenizer.VAR_OFFSET <= first < tokenizer.VAR_OFFSET + tokenizer.MAX_VARS:
        if len(target_tokens) not in {3, 4}:
            raise ValueError(
                f"unexpected assignment target length: {len(target_tokens)}"
            )
        variable_label = _var_from_token(first, tokenizer)
        value_tok = int(target_tokens[1])
        if value_tok == int(tokenizer.TRUE_VAL):
            value_label = 1
        elif value_tok == int(tokenizer.FALSE_VAL):
            value_label = 0
        else:
            raise ValueError(f"unexpected assignment value token: {value_tok}")
        if int(target_tokens[2]) != int(tokenizer.OK):
            raise ValueError(f"assignment target missing OK token: {target_tokens}")
        solved_after_assignment = len(target_tokens) == 4 and int(
            target_tokens[3]
        ) == int(tokenizer.SOLVED)
        if len(target_tokens) == 4 and not solved_after_assignment:
            raise ValueError(f"unexpected 4-token assignment target: {target_tokens}")
        return TargetInfo(
            continue_label=0,
            variable_label=int(variable_label),
            value_label=int(value_label),
            clause_id=-1,
            backjump_level=-1,
            solved_after_assignment=bool(solved_after_assignment),
        )

    if first == int(tokenizer.CONFLICT):
        if len(target_tokens) != 4:
            raise ValueError(f"unexpected conflict target length: {len(target_tokens)}")
        if int(target_tokens[2]) != int(tokenizer.BACKJUMP):
            raise ValueError(f"conflict target missing BJ token: {target_tokens}")
        return TargetInfo(
            continue_label=1,
            variable_label=-1,
            value_label=-1,
            clause_id=int(_clause_from_token(int(target_tokens[1]), tokenizer)),
            backjump_level=int(_level_from_token(int(target_tokens[3]), tokenizer)),
            solved_after_assignment=False,
        )

    raise ValueError(f"unrecognized supervised target pattern: {target_tokens}")


def _status_truth_value(status_token: Optional[int]) -> Optional[int]:
    if status_token is None:
        return None
    tok = int(status_token)
    if tok in TRUE_LIKE_TOKENS:
        return 1
    if tok in FALSE_LIKE_TOKENS:
        return -1
    return None


def feature_spec(
    num_vars: int,
    include_clause_features: bool,
    max_clause_features: int,
) -> Dict[str, Any]:
    per_variable = [
        "status_unassigned",
        "status_true",
        "status_false",
        "status_newly_true",
        "status_newly_false",
        "status_free",
        "order_position_norm",
        "appears_in_state_block",
    ]
    global_names = [
        "inferred_fraction_assigned",
        "visible_fraction_unassigned",
        "source_block_id_raw",
        "source_block_id_norm",
        "num_clauses_norm",
        "count_unassigned_norm",
        "count_true_norm",
        "count_false_norm",
        "count_newly_true_norm",
        "count_newly_false_norm",
        "count_free_norm",
    ]
    spec: Dict[str, Any] = {
        "num_vars": int(num_vars),
        "status_order": list(STATUS_NAMES),
        "per_variable": {
            "variables": int(num_vars),
            "features_per_variable": list(per_variable),
            "flattened_dim": int(num_vars) * len(per_variable),
        },
        "global": {
            "features": list(global_names),
            "dim": len(global_names),
        },
        "include_clause_features": bool(include_clause_features),
        "clause": {
            "max_clause_features": int(max_clause_features),
            "features_per_clause": [
                "visible_satisfied_literal_fraction",
                "visible_unsatisfied_literal_fraction",
                "unresolved_literal_fraction",
            ],
            "dim": int(max_clause_features) * 3 if include_clause_features else 0,
        },
    }
    spec["total_dim"] = int(
        spec["per_variable"]["flattened_dim"]
        + spec["global"]["dim"]
        + spec["clause"]["dim"]
    )
    return spec


def extract_feature_vector(
    *,
    clauses: Sequence[Tuple[int, ...]],
    visible_order: Sequence[int],
    visible_status: Dict[int, int],
    num_vars: int,
    source_block_id: int,
    include_clause_features: bool,
    max_clause_features: int,
) -> np.ndarray:
    spec = feature_spec(
        num_vars=int(num_vars),
        include_clause_features=bool(include_clause_features),
        max_clause_features=int(max_clause_features),
    )
    total_dim = int(spec["total_dim"])
    feat = np.zeros((total_dim,), dtype=np.float32)

    per_var_dim = len(spec["per_variable"]["features_per_variable"])
    visible_positions = {int(var): idx for idx, var in enumerate(visible_order)}
    status_counts = np.zeros((len(STATUS_TOKENS),), dtype=np.float32)
    denom_pos = max(len(visible_order) - 1, 1)

    for var in range(int(num_vars)):
        base = int(var) * int(per_var_dim)
        status_tok = visible_status.get(int(var))
        if status_tok is not None:
            feat[base + STATUS_TO_INDEX[int(status_tok)]] = 1.0
            feat[base + 6] = float(visible_positions[int(var)]) / float(denom_pos)
            feat[base + 7] = 1.0
            status_counts[STATUS_TO_INDEX[int(status_tok)]] += 1.0

    global_offset = int(num_vars) * int(per_var_dim)
    visible_fraction_unassigned = _safe_div(float(len(visible_order)), float(num_vars))
    inferred_fraction_assigned = max(0.0, 1.0 - visible_fraction_unassigned)
    feat[global_offset + 0] = float(inferred_fraction_assigned)
    feat[global_offset + 1] = float(visible_fraction_unassigned)
    feat[global_offset + 2] = float(source_block_id)
    feat[global_offset + 3] = _safe_div(float(source_block_id), float(num_vars))
    feat[global_offset + 4] = _safe_div(float(len(clauses)), float(max_clause_features))
    for idx in range(len(STATUS_TOKENS)):
        feat[global_offset + 5 + idx] = _safe_div(
            float(status_counts[idx]),
            float(num_vars),
        )

    if not include_clause_features:
        return feat

    clause_offset = global_offset + 11
    for clause_idx in range(min(len(clauses), int(max_clause_features))):
        sat_count = 0.0
        unsat_count = 0.0
        unresolved_count = 0.0
        clause = clauses[int(clause_idx)]
        for lit in clause:
            var_id = abs(int(lit)) - 1
            visible_truth = _status_truth_value(visible_status.get(int(var_id)))
            if visible_truth is None:
                unresolved_count += 1.0
                continue
            literal_true = visible_truth == (1 if int(lit) > 0 else -1)
            if literal_true:
                sat_count += 1.0
            else:
                unsat_count += 1.0
        base = clause_offset + int(clause_idx) * 3
        clause_len = max(float(len(clause)), 1.0)
        feat[base + 0] = float(sat_count / clause_len)
        feat[base + 1] = float(unsat_count / clause_len)
        feat[base + 2] = float(unresolved_count / clause_len)

    return feat


def _sorted_unassigned_vars(
    env: SatEnv,
    state: SatState,
    occurrence: np.ndarray,
) -> List[int]:
    _ = env
    unassigned = [
        int(v) for v in range(int(state.num_vars)) if int(state.assignment[int(v)]) == 0
    ]
    return sorted(
        unassigned,
        key=lambda v: (
            -float(state.activity[int(v)]),
            -int(occurrence[int(v)]),
            int(v),
        ),
    )


def _visible_status_from_env(
    env: SatEnv,
    state: SatState,
    var_id: int,
    tokenizer: SATInterleavedTokenizer,
) -> int:
    domain = {int(x) for x in env._effective_domain(state, int(var_id))}
    if domain == {-1, 1} or domain == {1, -1}:
        return int(tokenizer.UNASSIGNED)
    if domain == {1}:
        return int(tokenizer.TRUE_VAL)
    if domain == {-1}:
        return int(tokenizer.FALSE_VAL)
    return int(tokenizer.FREE)


def extract_feature_vector_from_env(
    *,
    env: SatEnv,
    state: SatState,
    clauses: Sequence[Tuple[int, ...]],
    occurrence: np.ndarray,
    tokenizer: SATInterleavedTokenizer,
    source_block_id: int,
    include_clause_features: bool,
    max_clause_features: int,
) -> Tuple[np.ndarray, List[int], Dict[int, int]]:
    visible_order = _sorted_unassigned_vars(env, state, occurrence)
    visible_status = {
        int(var): int(_visible_status_from_env(env, state, int(var), tokenizer))
        for var in visible_order
    }
    features = extract_feature_vector(
        clauses=clauses,
        visible_order=visible_order,
        visible_status=visible_status,
        num_vars=int(state.num_vars),
        source_block_id=int(source_block_id),
        include_clause_features=bool(include_clause_features),
        max_clause_features=int(max_clause_features),
    )
    return features, visible_order, visible_status


def _load_records(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, list):
        raise TypeError(f"expected list from {dataset_path}, got {type(raw)}")
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"record {idx} is not a dict: {type(item)}")
        records.append({str(key): value for key, value in item.items()})
    return records


def prepare_dataset_arrays(
    *,
    records: Sequence[Dict[str, Any]],
    tokenizer: SATInterleavedTokenizer,
    num_vars: int,
    include_clause_features: bool,
    max_clause_features: int,
) -> Dict[str, Any]:
    spec = feature_spec(
        num_vars=int(num_vars),
        include_clause_features=bool(include_clause_features),
        max_clause_features=int(max_clause_features),
    )
    feature_dim = int(spec["total_dim"])
    num_records = len(records)
    features = np.zeros((num_records, feature_dim), dtype=np.float32)
    continue_labels = np.zeros((num_records,), dtype=np.int64)
    variable_labels = np.full((num_records,), -1, dtype=np.int64)
    value_labels = np.full((num_records,), -1, dtype=np.int64)
    solved_after_assignment = np.zeros((num_records,), dtype=np.int64)

    clause_cache: Dict[int, List[Tuple[int, ...]]] = {}
    assignment_targets = 0
    conflict_targets = 0

    for idx, record in enumerate(records):
        sequence = [int(tok) for tok in record["sequence"]]
        block_ids = [int(x) for x in record["block_ids"]]
        loss_mask = [bool(x) for x in record["loss_mask"]]
        meta = record.get("meta", {}) if isinstance(record.get("meta"), dict) else {}
        record_num_vars = int(meta.get("num_vars", num_vars))
        if int(record_num_vars) != int(num_vars):
            raise ValueError(
                f"record {idx} num_vars mismatch: expected {num_vars}, got {record_num_vars}"
            )

        source_trace_index = int(meta.get("source_trace_index", idx))
        clauses = clause_cache.get(int(source_trace_index))
        if clauses is None:
            clauses = _parse_clause_prefix(sequence, block_ids, tokenizer)
            clause_cache[int(source_trace_index)] = list(clauses)

        block_tokens = _extract_block_tokens(sequence, block_ids)
        visible_order, visible_status = _parse_state_block(block_tokens, tokenizer)
        source_block_id = int(meta.get("source_block_id", 0))
        features[idx] = extract_feature_vector(
            clauses=clauses,
            visible_order=visible_order,
            visible_status=visible_status,
            num_vars=int(num_vars),
            source_block_id=int(source_block_id),
            include_clause_features=bool(include_clause_features),
            max_clause_features=int(max_clause_features),
        )
        target = _extract_target_info(sequence, loss_mask, tokenizer)
        continue_labels[idx] = int(target.continue_label)
        variable_labels[idx] = int(target.variable_label)
        value_labels[idx] = int(target.value_label)
        solved_after_assignment[idx] = int(target.solved_after_assignment)
        if int(target.continue_label) == 0:
            assignment_targets += 1
        else:
            conflict_targets += 1

        if idx % 25000 == 0 or idx == num_records - 1:
            logger.info(
                "prepared_features records=%d/%d assignment=%d conflict=%d",
                int(idx + 1),
                int(num_records),
                int(assignment_targets),
                int(conflict_targets),
            )

    return {
        "features": features,
        "continue_labels": continue_labels,
        "variable_labels": variable_labels,
        "value_labels": value_labels,
        "solved_after_assignment": solved_after_assignment,
        "feature_spec": spec,
        "dataset_summary": {
            "num_records": int(num_records),
            "feature_dim": int(feature_dim),
            "num_assignment_targets": int(assignment_targets),
            "num_backtrack_targets": int(conflict_targets),
            "continue_rate": _safe_div(float(assignment_targets), float(num_records)),
            "backtrack_rate": _safe_div(float(conflict_targets), float(num_records)),
            "num_vars": int(num_vars),
            "include_clause_features": bool(include_clause_features),
            "max_clause_features": int(max_clause_features),
        },
    }


class StatePolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_vars: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(int(in_dim), int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.continue_head = nn.Linear(int(in_dim), 2)
        self.variable_head = nn.Linear(int(in_dim), int(num_vars))
        self.value_head = nn.Linear(int(in_dim), 2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.continue_head(h), self.variable_head(h), self.value_head(h)


def _iterate_batches(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterable[np.ndarray]:
    order = np.array(indices, copy=True)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        yield order[start : start + int(batch_size)]


def _compute_normalization(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = features[train_indices].mean(axis=0).astype(np.float32)
    std = features[train_indices].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _batch_to_device(
    *,
    features: np.ndarray,
    continue_labels: np.ndarray,
    variable_labels: np.ndarray,
    value_labels: np.ndarray,
    batch_indices: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_features = (features[batch_indices] - feature_mean) / feature_std
    x = torch.tensor(batch_features, dtype=torch.float32, device=device)
    y_continue = torch.tensor(
        continue_labels[batch_indices], dtype=torch.long, device=device
    )
    y_var = torch.tensor(
        variable_labels[batch_indices], dtype=torch.long, device=device
    )
    y_value = torch.tensor(value_labels[batch_indices], dtype=torch.long, device=device)
    return x, y_continue, y_var, y_value


def _compute_losses_and_metrics(
    *,
    continue_logits: torch.Tensor,
    variable_logits: torch.Tensor,
    value_logits: torch.Tensor,
    y_continue: torch.Tensor,
    y_var: torch.Tensor,
    y_value: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = nn.CrossEntropyLoss()
    continue_loss = ce(continue_logits, y_continue)
    continue_mask = y_continue == 0

    if bool(torch.any(continue_mask)):
        variable_loss = ce(variable_logits[continue_mask], y_var[continue_mask])
        value_loss = ce(value_logits[continue_mask], y_value[continue_mask])
    else:
        variable_loss = continue_logits.new_zeros(())
        value_loss = continue_logits.new_zeros(())

    total_loss = continue_loss + variable_loss + value_loss

    with torch.no_grad():
        continue_pred = torch.argmax(continue_logits, dim=-1)
        variable_pred = torch.argmax(variable_logits, dim=-1)
        value_pred = torch.argmax(value_logits, dim=-1)
        continue_correct = (continue_pred == y_continue).float()
        if bool(torch.any(continue_mask)):
            variable_correct = (
                (variable_pred[continue_mask] == y_var[continue_mask]).float().mean()
            )
            value_correct = (
                (value_pred[continue_mask] == y_value[continue_mask]).float().mean()
            )
            joint_correct = (
                (
                    (continue_pred[continue_mask] == y_continue[continue_mask])
                    & (variable_pred[continue_mask] == y_var[continue_mask])
                    & (value_pred[continue_mask] == y_value[continue_mask])
                )
                .float()
                .mean()
            )
            num_continue = int(torch.sum(continue_mask).item())
        else:
            variable_correct = continue_logits.new_tensor(0.0)
            value_correct = continue_logits.new_tensor(0.0)
            joint_correct = continue_logits.new_tensor(0.0)
            num_continue = 0
        num_backtrack = int(torch.sum(y_continue == 1).item())

    metrics = {
        "loss": float(total_loss.item()),
        "continue_loss": float(continue_loss.item()),
        "variable_loss": float(variable_loss.item()),
        "value_loss": float(value_loss.item()),
        "continue_acc": float(continue_correct.mean().item()),
        "variable_acc": float(variable_correct.item()),
        "value_acc": float(value_correct.item()),
        "joint_continue_acc": float(joint_correct.item()),
        "backtrack_rate": _safe_div(float(num_backtrack), float(y_continue.shape[0])),
        "batch_size": int(y_continue.shape[0]),
        "num_continue": int(num_continue),
        "num_backtrack": int(num_backtrack),
    }
    return total_loss, metrics


def run_epoch(
    *,
    model: StatePolicyNet,
    optimizer: Optional[torch.optim.Optimizer],
    features: np.ndarray,
    continue_labels: np.ndarray,
    variable_labels: np.ndarray,
    value_labels: np.ndarray,
    indices: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    train: bool,
) -> Tuple[SplitMetrics, List[Dict[str, Any]]]:
    model.train(mode=bool(train))
    total_weight = 0
    accum = {
        "loss": 0.0,
        "continue_loss": 0.0,
        "variable_loss": 0.0,
        "value_loss": 0.0,
        "continue_acc": 0.0,
        "variable_acc": 0.0,
        "value_acc": 0.0,
        "joint_continue_acc": 0.0,
        "backtrack_rate": 0.0,
    }
    total_continue = 0
    total_backtrack = 0
    samples: List[Dict[str, Any]] = []

    for batch_idx, batch_indices in enumerate(
        _iterate_batches(
            indices=indices,
            batch_size=int(batch_size),
            rng=rng,
            shuffle=bool(train),
        )
    ):
        x, y_continue, y_var, y_value = _batch_to_device(
            features=features,
            continue_labels=continue_labels,
            variable_labels=variable_labels,
            value_labels=value_labels,
            batch_indices=batch_indices,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
        )

        with torch.set_grad_enabled(bool(train)):
            continue_logits, variable_logits, value_logits = model(x)
            loss, batch_metrics = _compute_losses_and_metrics(
                continue_logits=continue_logits,
                variable_logits=variable_logits,
                value_logits=value_logits,
                y_continue=y_continue,
                y_var=y_var,
                y_value=y_value,
            )
            if train:
                if optimizer is None:
                    raise ValueError("optimizer required for train=True")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size_actual = int(batch_metrics["batch_size"])
        total_weight += int(batch_size_actual)
        total_continue += int(batch_metrics["num_continue"])
        total_backtrack += int(batch_metrics["num_backtrack"])
        for key in accum:
            accum[key] += float(batch_metrics[key]) * float(batch_size_actual)

        if batch_idx == 0:
            with torch.no_grad():
                cont_pred = torch.argmax(continue_logits[:3], dim=-1).cpu().tolist()
                var_pred = torch.argmax(variable_logits[:3], dim=-1).cpu().tolist()
                val_pred = torch.argmax(value_logits[:3], dim=-1).cpu().tolist()
                for i in range(min(3, len(batch_indices))):
                    samples.append(
                        {
                            "dataset_index": int(batch_indices[i]),
                            "pred_continue": int(cont_pred[i]),
                            "pred_var": int(var_pred[i]),
                            "pred_value": int(val_pred[i]),
                            "target_continue": int(y_continue[i].item()),
                            "target_var": int(y_var[i].item()),
                            "target_value": int(y_value[i].item()),
                        }
                    )

    if total_weight == 0:
        raise RuntimeError("empty split encountered")

    metrics = SplitMetrics(
        loss=float(accum["loss"] / total_weight),
        continue_loss=float(accum["continue_loss"] / total_weight),
        variable_loss=float(accum["variable_loss"] / total_weight),
        value_loss=float(accum["value_loss"] / total_weight),
        continue_acc=float(accum["continue_acc"] / total_weight),
        variable_acc=float(accum["variable_acc"] / total_weight),
        value_acc=float(accum["value_acc"] / total_weight),
        joint_continue_acc=float(accum["joint_continue_acc"] / total_weight),
        backtrack_rate=float(accum["backtrack_rate"] / total_weight),
        num_examples=int(total_weight),
        num_continue=int(total_continue),
        num_backtrack=int(total_backtrack),
    )
    return metrics, samples


@torch.no_grad()
def predict_policy_logits(
    *,
    model: StatePolicyNet,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.tensor(
        ((features.astype(np.float32) - feature_mean) / feature_std)[None, :],
        dtype=torch.float32,
        device=device,
    )
    continue_logits, variable_logits, value_logits = model(x)
    return (
        continue_logits[0].detach().cpu().numpy(),
        variable_logits[0].detach().cpu().numpy(),
        value_logits[0].detach().cpu().numpy(),
    )


def _masked_argmax(logits: np.ndarray, allowed_indices: Sequence[int]) -> int:
    allowed = [int(x) for x in allowed_indices]
    if not allowed:
        raise ValueError("allowed_indices must be non-empty")
    best_idx = allowed[0]
    best_value = float(logits[best_idx])
    for idx in allowed[1:]:
        value = float(logits[int(idx)])
        if value > best_value:
            best_idx = int(idx)
            best_value = value
    return int(best_idx)


def _has_frontier_actions(actions: Sequence[SatAction]) -> Tuple[bool, bool]:
    has_assign = any(action.type == SatActionType.ASSIGN_VALUE for action in actions)
    has_select = any(action.type == SatActionType.SELECT_VAR for action in actions)
    return bool(has_assign), bool(has_select)


def _trim_tried_levels(
    tried_values_by_level: Dict[int, set[int]], max_level: int
) -> None:
    stale = [
        int(level) for level in tried_values_by_level if int(level) > int(max_level)
    ]
    for level in stale:
        tried_values_by_level.pop(int(level), None)


def _cascade_exhausted_backtracks(
    *,
    env: SatEnv,
    stats: Dict[str, Any],
    tried_values_by_level: Dict[int, set[int]],
) -> bool:
    while True:
        cascade_state = env.get_state()
        _trim_tried_levels(
            tried_values_by_level, int(len(cascade_state.decision_stack))
        )
        if cascade_state.status != SatEnvStatus.RUNNING:
            return True
        valid_actions = env.get_valid_actions()
        has_assign, has_select = _has_frontier_actions(valid_actions)
        if has_assign or has_select:
            return True
        if not cascade_state.decision_stack:
            done_res = env.step(SatAction.done())
            stats["termination_reason"] = (
                "unsat" if bool(done_res.done) else "failed_done_after_root_conflict"
            )
            return False
        backtrack_res = env.step(SatAction.backtrack())
        if not bool(backtrack_res.info.get("valid", True)):
            stats["termination_reason"] = (
                f"invalid_backtrack:{backtrack_res.info.get('reason', 'unknown')}"
            )
            return False
        stats["backtracks"] += 1
        stats["mechanical_backtracks"] += 1


def _variable_occurrence_counts(
    clauses: Sequence[Tuple[int, ...]],
    num_vars: int,
) -> np.ndarray:
    counts = np.zeros((int(num_vars),), dtype=np.int64)
    for clause in clauses:
        for lit in clause:
            var = int(abs(int(lit)) - 1)
            if 0 <= var < int(num_vars):
                counts[var] += 1
    return counts


def solve_instance_with_policy(
    *,
    model: StatePolicyNet,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    clauses: List[Tuple[int, ...]],
    num_vars: int,
    planted_solution: Optional[np.ndarray],
    tokenizer: SATInterleavedTokenizer,
    device: torch.device,
    max_steps: int,
    include_clause_features: bool,
    max_clause_features: int,
    var_select: str = "model",
) -> Dict[str, Any]:
    env = SatEnv(
        clauses=[tuple(int(x) for x in clause) for clause in clauses],
        num_vars=int(num_vars),
        planted_solution=None
        if planted_solution is None
        else np.array(planted_solution, dtype=np.int64, copy=True),
        mode="strict",
        max_steps=int(max_steps * 8 + 20),
    )
    env.reset()
    occurrence = _variable_occurrence_counts(clauses, int(num_vars))
    tried_values_by_level: Dict[int, set[int]] = {}

    stats: Dict[str, Any] = {
        "solved": False,
        "steps": 0,
        "decisions": 0,
        "conflicts": 0,
        "backtracks": 0,
        "mechanical_backtracks": 0,
        "policy_backtracks": 0,
        "invalid_continue_on_conflict": 0,
        "invalid_backtrack_at_root": 0,
        "post_backtrack_decisions": 0,
        "repeat_value_predictions": 0,
        "termination_reason": "max_steps",
    }

    with torch.no_grad():
        for step in range(int(max_steps)):
            stats["steps"] = int(step + 1)
            state = env.get_state()

            if state.status != SatEnvStatus.RUNNING:
                stats["solved"] = bool(state.status == SatEnvStatus.SUCCESS)
                stats["termination_reason"] = str(
                    state.termination_reason or "env_done"
                )
                break

            if bool(state.propagation_pending):
                prop_res = env.step(SatAction.propagate())
                if not bool(prop_res.info.get("valid", True)):
                    stats["termination_reason"] = (
                        f"invalid_propagate:{prop_res.info.get('reason', 'unknown')}"
                    )
                    break
                state = env.get_state()

            if state.status != SatEnvStatus.RUNNING:
                stats["solved"] = bool(state.status == SatEnvStatus.SUCCESS)
                stats["termination_reason"] = str(
                    state.termination_reason or "env_done"
                )
                break

            if env._all_satisfied(state):
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_solved"
                    break
                stats["solved"] = True
                stats["termination_reason"] = "solved"
                break

            features, _visible_order, _visible_status = extract_feature_vector_from_env(
                env=env,
                state=state,
                clauses=clauses,
                occurrence=occurrence,
                tokenizer=tokenizer,
                source_block_id=int(step + 1),
                include_clause_features=bool(include_clause_features),
                max_clause_features=int(max_clause_features),
            )
            continue_logits, variable_logits, value_logits = predict_policy_logits(
                model=model,
                features=features,
                feature_mean=feature_mean,
                feature_std=feature_std,
                device=device,
            )
            predicted_backtrack = int(np.argmax(continue_logits)) == 1

            if state.conflict_clause is not None:
                stats["conflicts"] += 1
                if not predicted_backtrack:
                    stats["invalid_continue_on_conflict"] += 1
                if not state.decision_stack:
                    done_res = env.step(SatAction.done())
                    if not bool(done_res.done):
                        stats["termination_reason"] = "failed_done_after_root_conflict"
                    else:
                        stats["termination_reason"] = "unsat"
                    break
                backtrack_res = env.step(SatAction.backtrack())
                if not bool(backtrack_res.info.get("valid", True)):
                    stats["termination_reason"] = (
                        f"invalid_backtrack:{backtrack_res.info.get('reason', 'unknown')}"
                    )
                    break
                stats["backtracks"] += 1
                stats["mechanical_backtracks"] += 1
                if not _cascade_exhausted_backtracks(
                    env=env,
                    stats=stats,
                    tried_values_by_level=tried_values_by_level,
                ):
                    break
                continue

            if predicted_backtrack:
                if not state.decision_stack:
                    stats["invalid_backtrack_at_root"] += 1
                else:
                    backtrack_res = env.step(SatAction.backtrack())
                    if not bool(backtrack_res.info.get("valid", True)):
                        stats["termination_reason"] = (
                            f"invalid_backtrack:{backtrack_res.info.get('reason', 'unknown')}"
                        )
                        break
                    stats["backtracks"] += 1
                    stats["policy_backtracks"] += 1
                    if not _cascade_exhausted_backtracks(
                        env=env,
                        stats=stats,
                        tried_values_by_level=tried_values_by_level,
                    ):
                        break
                    continue

            valid_actions = env.get_valid_actions()
            selectable_vars = [
                int(action.target)
                for action in valid_actions
                if action.type == SatActionType.SELECT_VAR and action.target is not None
            ]
            if selectable_vars:
                if str(var_select) == "model":
                    chosen_var = _masked_argmax(variable_logits, selectable_vars)
                elif str(var_select) == "random":
                    chosen_var = int(random.choice(selectable_vars))
                elif str(var_select) == "index":
                    chosen_var = int(min(selectable_vars))
                elif str(var_select) == "occurrence":
                    chosen_var = int(
                        max(
                            selectable_vars,
                            key=lambda v: (int(occurrence[int(v)]), -int(v)),
                        )
                    )
                else:
                    chosen_var = _masked_argmax(variable_logits, selectable_vars)
                select_res = env.step(SatAction.select_var(int(chosen_var)))
                if not bool(select_res.info.get("valid", True)):
                    stats["termination_reason"] = (
                        f"invalid_select:{select_res.info.get('reason', 'unknown')}"
                    )
                    break

            post_select_state = env.get_state()
            assign_actions = [
                int(action.target)
                for action in env.get_valid_actions()
                if action.type == SatActionType.ASSIGN_VALUE
                and action.target is not None
            ]
            if not assign_actions:
                if post_select_state.decision_stack:
                    backtrack_res = env.step(SatAction.backtrack())
                    if not bool(backtrack_res.info.get("valid", True)):
                        stats["termination_reason"] = (
                            f"invalid_backtrack:{backtrack_res.info.get('reason', 'unknown')}"
                        )
                        break
                    stats["backtracks"] += 1
                    stats["mechanical_backtracks"] += 1
                    if not _cascade_exhausted_backtracks(
                        env=env,
                        stats=stats,
                        tried_values_by_level=tried_values_by_level,
                    ):
                        break
                    continue
                stats["termination_reason"] = "no_assign_actions"
                break

            chosen_value = _masked_argmax(value_logits, assign_actions)
            decision_level = int(len(post_select_state.decision_stack))
            tried_before = tried_values_by_level.get(int(decision_level), set())
            if tried_before:
                stats["post_backtrack_decisions"] += 1
                if int(chosen_value) in tried_before:
                    stats["repeat_value_predictions"] += 1

            assign_res = env.step(SatAction.assign_value(int(chosen_value)))
            if not bool(assign_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_assign:{assign_res.info.get('reason', 'unknown')}"
                )
                break
            stats["decisions"] += 1

            prop_res = env.step(SatAction.propagate())
            if not bool(prop_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_propagate:{prop_res.info.get('reason', 'unknown')}"
                )
                break

            post_state = env.get_state()
            if post_state.conflict_clause is not None:
                stats["conflicts"] += 1
                tried_values_by_level.setdefault(
                    int(len(post_state.decision_stack)), set()
                ).add(int(chosen_value))
                if post_state.decision_stack:
                    backtrack_res = env.step(SatAction.backtrack())
                    if not bool(backtrack_res.info.get("valid", True)):
                        stats["termination_reason"] = (
                            f"invalid_backtrack:{backtrack_res.info.get('reason', 'unknown')}"
                        )
                        break
                    stats["backtracks"] += 1
                    stats["mechanical_backtracks"] += 1
                    if not _cascade_exhausted_backtracks(
                        env=env,
                        stats=stats,
                        tried_values_by_level=tried_values_by_level,
                    ):
                        break
                    continue
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_root_conflict"
                    break
                stats["termination_reason"] = "unsat"
                break

            if env._all_satisfied(post_state):
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_solved"
                    break
                stats["solved"] = True
                stats["termination_reason"] = "solved"
                break

    return stats


def evaluate_closed_loop(
    *,
    model: StatePolicyNet,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    seed: int,
    num_instances: int,
    num_vars: int,
    alpha: float,
    budget: int,
    include_clause_features: bool,
    max_clause_features: int,
    var_select: str = "model",
) -> Dict[str, Any]:
    tokenizer = SATInterleavedTokenizer()
    generator = SatGenerator(seed=int(seed))
    per_instance: List[Dict[str, Any]] = []

    for idx in range(int(num_instances)):
        instance = generator.generate_planted(
            num_vars=int(num_vars), alpha=float(alpha)
        )
        stats = solve_instance_with_policy(
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            clauses=[tuple(int(x) for x in clause) for clause in instance.clauses],
            num_vars=int(instance.num_vars),
            planted_solution=instance.planted_solution,
            tokenizer=tokenizer,
            device=device,
            max_steps=int(budget),
            include_clause_features=bool(include_clause_features),
            max_clause_features=int(max_clause_features),
            var_select=str(var_select),
        )
        per_instance.append(stats)
        if idx < 3:
            logger.info(
                "eval_sample idx=%d solved=%s reason=%s decisions=%d backtracks=%d conflicts=%d",
                int(idx),
                bool(stats["solved"]),
                str(stats["termination_reason"]),
                int(stats["decisions"]),
                int(stats["backtracks"]),
                int(stats["conflicts"]),
            )

    solve_rate = _safe_div(
        float(sum(1 for row in per_instance if bool(row["solved"]))),
        float(len(per_instance)),
    )
    false_unsat_rate = _safe_div(
        float(
            sum(1 for row in per_instance if str(row["termination_reason"]) == "unsat")
        ),
        float(len(per_instance)),
    )
    timeout_rate = _safe_div(
        float(
            sum(
                1
                for row in per_instance
                if str(row["termination_reason"]) in {"timeout", "max_steps", "budget"}
            )
        ),
        float(len(per_instance)),
    )
    invalid_rate = _safe_div(
        float(
            sum(
                1
                for row in per_instance
                if str(row["termination_reason"]).startswith("invalid")
            )
        ),
        float(len(per_instance)),
    )

    return {
        "num_instances": int(len(per_instance)),
        "num_vars": int(num_vars),
        "alpha": float(alpha),
        "budget": int(budget),
        "solve_rate": float(solve_rate),
        "false_unsat_rate": float(false_unsat_rate),
        "timeout_rate": float(timeout_rate),
        "invalid_rate": float(invalid_rate),
        "mean_decisions": float(np.mean([row["decisions"] for row in per_instance])),
        "mean_backtracks": float(np.mean([row["backtracks"] for row in per_instance])),
        "mean_conflicts": float(np.mean([row["conflicts"] for row in per_instance])),
        "mean_steps": float(np.mean([row["steps"] for row in per_instance])),
        "mean_policy_backtracks": float(
            np.mean([row["policy_backtracks"] for row in per_instance])
        ),
        "mean_mechanical_backtracks": float(
            np.mean([row["mechanical_backtracks"] for row in per_instance])
        ),
        "termination_reason_histogram": {
            str(reason): int(
                sum(
                    1
                    for row in per_instance
                    if str(row["termination_reason"]) == str(reason)
                )
            )
            for reason in sorted(
                {str(row["termination_reason"]) for row in per_instance}
            )
        },
        "sample_instances": per_instance[:10],
    }


def train_single_model(
    *,
    model_name: str,
    hidden_dims: Sequence[int],
    seed: int,
    output_dir: Path,
    features: np.ndarray,
    continue_labels: np.ndarray,
    variable_labels: np.ndarray,
    value_labels: np.ndarray,
    feature_spec_dict: Dict[str, Any],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_fraction: float,
    eval_instances: int,
    eval_num_vars: int,
    eval_alpha: float,
    eval_budget: int,
    include_clause_features: bool,
    max_clause_features: int,
) -> Dict[str, Any]:
    _set_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    num_records = int(features.shape[0])
    permutation = rng.permutation(num_records)
    split = max(1, int(math.floor((1.0 - float(val_fraction)) * num_records)))
    split = min(split, num_records - 1)
    train_indices = permutation[:split]
    val_indices = permutation[split:]
    feature_mean, feature_std = _compute_normalization(features, train_indices)

    model = StatePolicyNet(
        input_dim=int(features.shape[1]),
        num_vars=int(feature_spec_dict["num_vars"]),
        hidden_dims=list(int(x) for x in hidden_dims),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_checkpoint_path = model_dir / "best.pt"

    logger.info(
        "training model=%s seed=%d train=%d val=%d input_dim=%d hidden_dims=%s",
        str(model_name),
        int(seed),
        int(len(train_indices)),
        int(len(val_indices)),
        int(features.shape[1]),
        list(int(x) for x in hidden_dims),
    )

    for epoch in range(1, int(epochs) + 1):
        train_metrics, train_samples = run_epoch(
            model=model,
            optimizer=optimizer,
            features=features,
            continue_labels=continue_labels,
            variable_labels=variable_labels,
            value_labels=value_labels,
            indices=train_indices,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_size=int(batch_size),
            device=device,
            rng=rng,
            train=True,
        )
        val_metrics, val_samples = run_epoch(
            model=model,
            optimizer=None,
            features=features,
            continue_labels=continue_labels,
            variable_labels=variable_labels,
            value_labels=value_labels,
            indices=val_indices,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_size=int(batch_size),
            device=device,
            rng=rng,
            train=False,
        )
        entry = {
            "epoch": int(epoch),
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
            "train_samples": train_samples,
            "val_samples": val_samples,
        }
        history.append(entry)

        logger.info(
            "epoch=%d model=%s train_loss=%.4f val_loss=%.4f train_continue=%.4f val_continue=%.4f val_var=%.4f val_value=%.4f sample=%s",
            int(epoch),
            str(model_name),
            float(train_metrics.loss),
            float(val_metrics.loss),
            float(train_metrics.continue_acc),
            float(val_metrics.continue_acc),
            float(val_metrics.variable_acc),
            float(val_metrics.value_acc),
            val_samples[:1],
        )

        if float(val_metrics.loss) < float(best_val_loss):
            best_val_loss = float(val_metrics.loss)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "seed": int(seed),
                    "model_name": str(model_name),
                    "hidden_dims": [int(x) for x in hidden_dims],
                    "feature_spec": feature_spec_dict,
                    "history": history,
                    "val_loss": float(val_metrics.loss),
                    "val_continue_acc": float(val_metrics.continue_acc),
                    "val_variable_acc": float(val_metrics.variable_acc),
                    "val_value_acc": float(val_metrics.value_acc),
                },
                best_checkpoint_path,
            )

    history_path = model_dir / "history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    best_model = StatePolicyNet(
        input_dim=int(features.shape[1]),
        num_vars=int(feature_spec_dict["num_vars"]),
        hidden_dims=list(int(x) for x in hidden_dims),
    )
    best_model.load_state_dict(best_checkpoint["model_state_dict"])
    best_model = best_model.to(device).eval()

    eval_results = evaluate_closed_loop(
        model=best_model,
        feature_mean=np.asarray(best_checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(best_checkpoint["feature_std"], dtype=np.float32),
        device=device,
        seed=int(seed),
        num_instances=int(eval_instances),
        num_vars=int(eval_num_vars),
        alpha=float(eval_alpha),
        budget=int(eval_budget),
        include_clause_features=bool(include_clause_features),
        max_clause_features=int(max_clause_features),
    )
    with (model_dir / "closed_loop_eval.json").open("w") as f:
        json.dump(eval_results, f, indent=2)

    final_summary = {
        "seed": int(seed),
        "model_name": str(model_name),
        "hidden_dims": [int(x) for x in hidden_dims],
        "train_size": int(len(train_indices)),
        "val_size": int(len(val_indices)),
        "best_val_loss": float(best_val_loss),
        "history_path": str(history_path),
        "checkpoint_path": str(best_checkpoint_path),
        "final_epoch": history[-1] if history else None,
        "closed_loop_eval": eval_results,
    }
    with (model_dir / "summary.json").open("w") as f:
        json.dump(final_summary, f, indent=2)
    return final_summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MLP and linear SAT state-only baselines"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="experiments/sat-n50-state-only-from-enriched/traces.pkl",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num_vars", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--max_clause_features", type=int, default=200)
    parser.add_argument("--disable_clause_features", action="store_true")
    parser.add_argument("--eval_instances", type=int, default=200)
    parser.add_argument("--eval_num_vars", type=int, default=50)
    parser.add_argument("--eval_alpha", type=float, default=4.0)
    parser.add_argument("--eval_budget", type=int, default=4096)
    parser.add_argument(
        "--models",
        type=str,
        default="mlp,linear",
        help="Comma-separated subset of: mlp,linear",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(args.device))
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    seeds = _parse_seed_list(args.seed, args.seeds)
    requested_models = [
        str(x.strip()) for x in str(args.models).split(",") if str(x).strip()
    ]
    valid_models = {"mlp", "linear"}
    if any(model_name not in valid_models for model_name in requested_models):
        raise ValueError(f"models must be drawn from {sorted(valid_models)}")

    logger.info(
        "loading dataset path=%s seeds=%s device=%s",
        str(dataset_path),
        seeds,
        str(device),
    )
    records = _load_records(dataset_path)
    if args.max_records is not None:
        records = records[: int(args.max_records)]
        logger.info("using truncated dataset max_records=%d", int(args.max_records))

    tokenizer = SATInterleavedTokenizer()
    prepared = prepare_dataset_arrays(
        records=records,
        tokenizer=tokenizer,
        num_vars=int(args.num_vars),
        include_clause_features=not bool(args.disable_clause_features),
        max_clause_features=int(args.max_clause_features),
    )

    with (output_dir / "feature_spec.json").open("w") as f:
        json.dump(prepared["feature_spec"], f, indent=2)
    with (output_dir / "dataset_summary.json").open("w") as f:
        json.dump(prepared["dataset_summary"], f, indent=2)

    logger.info("dataset_summary=%s", prepared["dataset_summary"])

    per_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_result: Dict[str, Any] = {"seed": int(seed), "models": {}}
        for model_name in requested_models:
            hidden_dims: List[int] = [256, 256] if model_name == "mlp" else []
            summary = train_single_model(
                model_name=str(model_name),
                hidden_dims=hidden_dims,
                seed=int(seed),
                output_dir=seed_dir,
                features=prepared["features"],
                continue_labels=prepared["continue_labels"],
                variable_labels=prepared["variable_labels"],
                value_labels=prepared["value_labels"],
                feature_spec_dict=prepared["feature_spec"],
                device=device,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                val_fraction=float(args.val_fraction),
                eval_instances=int(args.eval_instances),
                eval_num_vars=int(args.eval_num_vars),
                eval_alpha=float(args.eval_alpha),
                eval_budget=int(args.eval_budget),
                include_clause_features=not bool(args.disable_clause_features),
                max_clause_features=int(args.max_clause_features),
            )
            seed_result["models"][str(model_name)] = summary
        per_seed_results.append(seed_result)

    aggregate: Dict[str, Any] = {
        "seeds": [int(seed) for seed in seeds],
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "models": {},
    }
    for model_name in requested_models:
        rows = [
            {
                "best_val_loss": float(seed_row["models"][model_name]["best_val_loss"]),
                "solve_rate": float(
                    seed_row["models"][model_name]["closed_loop_eval"]["solve_rate"]
                ),
                "false_unsat_rate": float(
                    seed_row["models"][model_name]["closed_loop_eval"][
                        "false_unsat_rate"
                    ]
                ),
                "timeout_rate": float(
                    seed_row["models"][model_name]["closed_loop_eval"]["timeout_rate"]
                ),
                "mean_decisions": float(
                    seed_row["models"][model_name]["closed_loop_eval"]["mean_decisions"]
                ),
                "mean_backtracks": float(
                    seed_row["models"][model_name]["closed_loop_eval"][
                        "mean_backtracks"
                    ]
                ),
            }
            for seed_row in per_seed_results
        ]
        aggregate["models"][str(model_name)] = {
            "per_seed": [
                seed_row["models"][model_name] for seed_row in per_seed_results
            ],
            "mean_metrics": _mean_dict(rows),
        }

    with (output_dir / "results.json").open("w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info("finished training/eval; results saved to %s", str(output_dir))


if __name__ == "__main__":
    main()
