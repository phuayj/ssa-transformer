#!/usr/bin/env python3
"""Train a SAT n=50 FactorGNN baseline on enriched traces.

This baseline encodes each SAT search state as a clause-variable factor graph
instead of a serialized token prefix. Traces are decoded into per-decision
state/action supervision tuples and then consumed by ``universal.model.FactorGNN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.dsl import SatAction
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.interleaved_tokenizer import SATInterleavedTokenizer
from universal.model import FactorGNN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


CACHE_PATH = REPO_ROOT / "experiments" / "sat-n50-factor-gnn-examples" / "examples.pkl"
DEFAULT_DATA_PATH = (
    REPO_ROOT / "experiments" / "sat-n50-enriched-traces" / "traces.pkl"
)
ACTION_ASSIGN = "ASSIGN"
ACTION_BACKTRACK = "BACKTRACK"
ACTION_DONE = "DONE"
SAT_NUM_CONSTRAINT_TYPES = 3
SAT_MAX_DOMAIN = 2


@dataclass(frozen=True)
class BlockSpan:
    block_id: int
    start: int
    end: int


@dataclass(frozen=True)
class DecodedSATExample:
    clauses: Tuple[Tuple[int, int, int], ...]
    assignment: np.ndarray  # [-1=unassigned, 0=false, 1=true]
    domain_mask: np.ndarray  # [N, 2] => [false, true]
    action: Tuple[str, int, int]
    decision_depth: int
    has_conflict: bool
    propagation_pending: bool
    source_trace_index: int
    source_block_id: int


@dataclass(frozen=True)
class ExampleDecodeSummary:
    num_traces: int
    num_examples: int
    num_assign: int
    num_backtrack: int
    num_done: int


@dataclass(frozen=True)
class EpochSummary:
    loss: float
    type_loss: float
    assign_loss: float
    type_acc: float
    assign_acc_when_assign: float
    num_examples: int
    num_assign_examples: int
    samples: List[Dict[str, Any]]


class FactorGNNSATDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        examples: Sequence[DecodedSATExample],
        *,
        max_vars: int,
        max_constraints: int,
    ):
        self.examples = list(examples)
        self.max_vars = int(max_vars)
        self.max_constraints = int(max_constraints)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return build_factor_gnn_item_from_example(
            self.examples[int(idx)],
            max_vars=int(self.max_vars),
            max_constraints=int(self.max_constraints),
        )


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _action_to_type_id(action_name: str) -> int:
    if str(action_name) == ACTION_ASSIGN:
        return 0
    if str(action_name) == ACTION_BACKTRACK:
        return 1
    if str(action_name) == ACTION_DONE:
        return 2
    raise ValueError(f"unsupported action_name={action_name}")


def _env_to_model_assignment(env_assignment: np.ndarray) -> np.ndarray:
    out = np.full(env_assignment.shape, -1, dtype=np.int64)
    out[np.asarray(env_assignment) == -1] = 0
    out[np.asarray(env_assignment) == 1] = 1
    return out


def _model_to_signed_value(value_idx: int) -> int:
    if int(value_idx) == 0:
        return -1
    if int(value_idx) == 1:
        return 1
    raise ValueError(f"SAT value_idx must be 0/1, got {value_idx}")


def _signed_value_to_model(value: int) -> int:
    if int(value) == -1:
        return 0
    if int(value) == 1:
        return 1
    raise ValueError(f"signed SAT value must be -1/+1, got {value}")


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


def _compute_block_ids_from_sequence(sequence: Sequence[int]) -> List[int]:
    tokenizer = SATInterleavedTokenizer()
    block_ids: List[int] = []
    current_block = 0
    search_started = False
    for raw_tok in sequence:
        tok = int(raw_tok)
        if tok == int(tokenizer.SEARCH_START):
            search_started = True
        elif search_started and tok == int(tokenizer.STATE):
            current_block += 1
        block_ids.append(int(current_block))
    return block_ids


def _load_trace_records(data_path: Path) -> List[Dict[str, Any]]:
    with data_path.open("rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        traces = raw.get("traces")
        if not isinstance(traces, list):
            raise TypeError("expected dict['traces'] to be a list")
        raw_records = traces
    elif isinstance(raw, list):
        raw_records = raw
    else:
        raise TypeError(f"unsupported trace container type: {type(raw)}")

    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_records):
        if not isinstance(item, dict):
            raise TypeError(f"trace {idx} is not a dict: {type(item)}")
        record = {str(k): v for k, v in item.items()}
        if "block_ids" not in record or record["block_ids"] is None:
            record["block_ids"] = _compute_block_ids_from_sequence(
                cast(Sequence[int], record["sequence"])
            )
        records.append(record)
    return records


def _compute_block_spans(block_ids: Sequence[int]) -> List[BlockSpan]:
    if not block_ids:
        return []
    spans: List[BlockSpan] = []
    start = 0
    current = int(block_ids[0])
    for idx in range(1, len(block_ids)):
        block_id = int(block_ids[idx])
        if block_id != current:
            spans.append(BlockSpan(block_id=int(current), start=int(start), end=int(idx)))
            start = idx
            current = block_id
    spans.append(BlockSpan(block_id=int(current), start=int(start), end=int(len(block_ids))))
    return spans


def _parse_clause_prefix(
    sequence: Sequence[int],
    block_ids: Sequence[int],
    tokenizer: SATInterleavedTokenizer,
) -> List[Tuple[int, int, int]]:
    prefix = [int(tok) for tok, block in zip(sequence, block_ids) if int(block) == 0]
    if not prefix:
        raise ValueError("empty clause prefix")

    try:
        start_idx = prefix.index(int(tokenizer.CLAUSE_START)) + 1
    except ValueError as exc:
        raise ValueError("prefix missing CLAUSE_START") from exc

    clauses: List[Tuple[int, int, int]] = []
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
        if len(clause) != 3:
            raise ValueError(f"expected 3-SAT clause, got len={len(clause)}")
        if i >= len(prefix):
            raise ValueError("malformed clause prefix: missing SEP")
        clauses.append((int(clause[0]), int(clause[1]), int(clause[2])))
        i += 1

    if not clauses:
        raise ValueError("no clauses parsed from prefix")
    return clauses


def _parse_state_block_visible_domains(
    block_tokens: Sequence[int],
    tokenizer: SATInterleavedTokenizer,
) -> Dict[int, Tuple[bool, bool]]:
    if not block_tokens or int(block_tokens[0]) != int(tokenizer.STATE):
        raise ValueError("block must start with STATE")
    try:
        sep_idx = list(int(tok) for tok in block_tokens).index(int(tokenizer.SEP))
    except ValueError as exc:
        raise ValueError("STATE block missing SEP") from exc

    state_tokens = [int(tok) for tok in block_tokens[1:sep_idx]]
    if len(state_tokens) % 2 != 0:
        raise ValueError("STATE block must contain var/status pairs")

    visible_domains: Dict[int, Tuple[bool, bool]] = {}
    for idx in range(0, len(state_tokens), 2):
        var_id = _var_from_token(int(state_tokens[idx]), tokenizer)
        status_tok = int(state_tokens[idx + 1])
        if status_tok == int(tokenizer.UNASSIGNED):
            dom = (True, True)
        elif status_tok == int(tokenizer.FALSE_VAL):
            dom = (True, False)
        elif status_tok == int(tokenizer.TRUE_VAL):
            dom = (False, True)
        elif status_tok == int(tokenizer.NEWLY_FALSE):
            dom = (True, False)
        elif status_tok == int(tokenizer.NEWLY_TRUE):
            dom = (False, True)
        else:
            raise ValueError(f"unsupported state-domain token: {status_tok}")
        visible_domains[int(var_id)] = dom
    return visible_domains


def _parse_block_action(
    block_tokens: Sequence[int],
    block_loss_mask: Sequence[bool],
    tokenizer: SATInterleavedTokenizer,
) -> Tuple[str, int, int]:
    target_tokens = [int(tok) for tok, use in zip(block_tokens, block_loss_mask) if bool(use)]
    if not target_tokens:
        raise ValueError("block is missing supervised action tokens")

    first = int(target_tokens[0])
    if tokenizer.VAR_OFFSET <= first < tokenizer.VAR_OFFSET + tokenizer.MAX_VARS:
        if len(target_tokens) not in {3, 4}:
            raise ValueError(f"unexpected assignment target length: {len(target_tokens)}")
        var_id = _var_from_token(first, tokenizer)
        value_tok = int(target_tokens[1])
        if value_tok == int(tokenizer.FALSE_VAL):
            value = 0
        elif value_tok == int(tokenizer.TRUE_VAL):
            value = 1
        else:
            raise ValueError(f"unexpected assignment target token: {value_tok}")
        if int(target_tokens[2]) != int(tokenizer.OK):
            raise ValueError("assignment target missing OK marker")
        if len(target_tokens) == 4 and int(target_tokens[3]) != int(tokenizer.SOLVED):
            raise ValueError(f"unexpected assignment suffix: {target_tokens}")
        return (ACTION_ASSIGN, int(var_id), int(value))

    if first == int(tokenizer.CONFLICT):
        if len(target_tokens) != 4:
            raise ValueError(f"unexpected conflict target length: {len(target_tokens)}")
        _clause_id = _clause_from_token(int(target_tokens[1]), tokenizer)
        if int(target_tokens[2]) != int(tokenizer.BACKJUMP):
            raise ValueError("conflict target missing BACKJUMP marker")
        _level = _level_from_token(int(target_tokens[3]), tokenizer)
        return (ACTION_BACKTRACK, -1, -1)

    raise ValueError(f"unrecognized supervised target pattern: {target_tokens}")


def _parse_conflict_backjump_level(
    block_tokens: Sequence[int],
    block_loss_mask: Sequence[bool],
    tokenizer: SATInterleavedTokenizer,
) -> Optional[int]:
    target_tokens = [int(tok) for tok, use in zip(block_tokens, block_loss_mask) if bool(use)]
    if not target_tokens or int(target_tokens[0]) != int(tokenizer.CONFLICT):
        return None
    if len(target_tokens) != 4:
        raise ValueError(f"unexpected conflict target length: {len(target_tokens)}")
    return int(_level_from_token(int(target_tokens[3]), tokenizer))


def _advance_propagation(env: SatEnv) -> SatState:
    state = env.get_state()
    while bool(state.propagation_pending):
        res = env.step(SatAction.propagate())
        if not bool(res.info.get("valid", True)):
            raise RuntimeError(f"invalid_propagate:{res.info.get('reason', 'unknown')}")
        state = env.get_state()
        if state.status != SatEnvStatus.RUNNING:
            return state
    return state


def _state_domain_mask(env: SatEnv, state: SatState) -> np.ndarray:
    mask = np.zeros((int(state.num_vars), SAT_MAX_DOMAIN), dtype=np.bool_)
    for var_id in range(int(state.num_vars)):
        domain = env._effective_domain(state, int(var_id))
        mask[int(var_id), 0] = -1 in domain
        mask[int(var_id), 1] = 1 in domain
    return mask


def make_decoded_example_from_state(
    *,
    clauses: Sequence[Tuple[int, int, int]],
    env: SatEnv,
    state: SatState,
    action: Tuple[str, int, int],
    source_trace_index: int,
    source_block_id: int,
) -> DecodedSATExample:
    return DecodedSATExample(
        clauses=tuple(
            cast(Tuple[int, int, int], tuple(int(x) for x in clause))
            for clause in clauses
        ),
        assignment=_env_to_model_assignment(state.assignment).astype(np.int32, copy=False),
        domain_mask=_state_domain_mask(env, state),
        action=(str(action[0]), int(action[1]), int(action[2])),
        decision_depth=int(len(state.decision_stack)),
        has_conflict=bool(state.conflict_clause is not None),
        propagation_pending=bool(state.propagation_pending),
        source_trace_index=int(source_trace_index),
        source_block_id=int(source_block_id),
    )


def _validate_visible_state_tokens(
    *,
    visible_domains: Dict[int, Tuple[bool, bool]],
    state: SatState,
    env: SatEnv,
    source_trace_index: int,
    source_block_id: int,
) -> None:
    mismatches: List[str] = []
    for var_id, domain in visible_domains.items():
        state_domain = env._effective_domain(state, int(var_id))
        expected = (-1 in state_domain, 1 in state_domain)
        if tuple(bool(x) for x in domain) != tuple(bool(x) for x in expected):
            mismatches.append(
                f"var={var_id} visible={domain} env={expected} assigned={int(state.assignment[int(var_id)])}"
            )
    if mismatches:
        logger.debug(
            "visible_state_domain_mismatch trace=%d block=%d mismatches=%s",
            int(source_trace_index),
            int(source_block_id),
            mismatches[:5],
        )


def decode_trace_record_to_examples(
    record: Dict[str, Any],
    *,
    source_trace_index: int,
    tokenizer: Optional[SATInterleavedTokenizer] = None,
) -> List[DecodedSATExample]:
    tokenizer = tokenizer or SATInterleavedTokenizer()
    sequence = [int(tok) for tok in record["sequence"]]
    loss_mask = [bool(x) for x in record["loss_mask"]]
    block_ids = [int(x) for x in record["block_ids"]]
    if not (len(sequence) == len(loss_mask) == len(block_ids)):
        raise ValueError("sequence/loss_mask/block_ids length mismatch")

    meta = record.get("meta", {}) if isinstance(record.get("meta"), dict) else {}
    clauses = _parse_clause_prefix(sequence, block_ids, tokenizer)
    num_vars = int(meta.get("num_vars", max(abs(lit) for clause in clauses for lit in clause)))

    env = SatEnv(
        clauses=[tuple(int(x) for x in clause) for clause in clauses],
        num_vars=int(num_vars),
        planted_solution=None,
        mode="strict",
        max_steps=max(1000, int(len(sequence) * 4 + 100)),
    )
    env.reset()

    spans = [span for span in _compute_block_spans(block_ids) if int(span.block_id) > 0]
    preparsed_actions = [
        _parse_block_action(
            sequence[int(span.start) : int(span.end)],
            loss_mask[int(span.start) : int(span.end)],
            tokenizer,
        )
        for span in spans
    ]
    preparsed_backjump_levels = [
        _parse_conflict_backjump_level(
            sequence[int(span.start) : int(span.end)],
            loss_mask[int(span.start) : int(span.end)],
            tokenizer,
        )
        for span in spans
    ]
    examples: List[DecodedSATExample] = []

    for span_idx, span in enumerate(spans):
        state = _advance_propagation(env)
        if state.status != SatEnvStatus.RUNNING:
            raise RuntimeError(
                f"trace terminated before block {span.block_id}: status={state.status}"
            )

        block_tokens = sequence[int(span.start) : int(span.end)]
        block_loss = loss_mask[int(span.start) : int(span.end)]
        visible_domains = _parse_state_block_visible_domains(block_tokens, tokenizer)
        _validate_visible_state_tokens(
            visible_domains=visible_domains,
            state=state,
            env=env,
            source_trace_index=int(source_trace_index),
            source_block_id=int(span.block_id),
        )

        action = preparsed_actions[int(span_idx)]
        examples.append(
            make_decoded_example_from_state(
                clauses=clauses,
                env=env,
                state=state,
                action=action,
                source_trace_index=int(source_trace_index),
                source_block_id=int(span.block_id),
            )
        )

        if str(action[0]) == ACTION_ASSIGN:
            var_id = int(action[1])
            value_idx = int(action[2])
            if state.selected_var is None:
                select_res = env.step(SatAction.select_var(int(var_id)))
                if not bool(select_res.info.get("valid", True)):
                    raise RuntimeError(
                        f"invalid_select trace={source_trace_index} block={span.block_id}: "
                        f"{select_res.info.get('reason', 'unknown')}"
                    )
            assign_res = env.step(SatAction.assign_value(int(value_idx)))
            if not bool(assign_res.info.get("valid", True)):
                raise RuntimeError(
                    f"invalid_assign trace={source_trace_index} block={span.block_id}: "
                    f"{assign_res.info.get('reason', 'unknown')}"
                )
        elif str(action[0]) == ACTION_BACKTRACK:
            expected_next_var: Optional[int] = None
            if int(span_idx) + 1 < len(preparsed_actions):
                next_action = preparsed_actions[int(span_idx) + 1]
                if str(next_action[0]) == ACTION_ASSIGN:
                    expected_next_var = int(next_action[1])
            target_backjump_level = preparsed_backjump_levels[int(span_idx)]
            for _attempt in range(int(num_vars) + 2):
                backtrack_res = env.step(SatAction.backtrack())
                if not bool(backtrack_res.info.get("valid", True)):
                    raise RuntimeError(
                        f"invalid_backtrack trace={source_trace_index} block={span.block_id}: "
                        f"{backtrack_res.info.get('reason', 'unknown')}"
                    )
                replay_state = _advance_propagation(env)
                if replay_state.status != SatEnvStatus.RUNNING:
                    break
                selectable_vars = [
                    int(candidate.target)
                    for candidate in env.get_valid_actions()
                    if candidate.type.name == "SELECT_VAR" and candidate.target is not None
                ]
                if expected_next_var is not None and int(expected_next_var) in selectable_vars:
                    break
                if expected_next_var is None:
                    if (
                        target_backjump_level is not None
                        and int(len(replay_state.decision_stack)) > int(target_backjump_level)
                    ):
                        continue
                    if replay_state.conflict_clause is not None and replay_state.decision_stack:
                        continue
                    break
            else:
                raise RuntimeError(
                    f"failed to align post-backtrack replay for trace={source_trace_index} block={span.block_id}"
                )
        else:
            raise RuntimeError(f"unexpected traced action type: {action}")

    terminal_state = _advance_propagation(env)
    if terminal_state.status == SatEnvStatus.RUNNING and (
        env._all_satisfied(terminal_state)
        or (
            terminal_state.conflict_clause is not None
            and not terminal_state.decision_stack
        )
    ):
        examples.append(
            make_decoded_example_from_state(
                clauses=clauses,
                env=env,
                state=terminal_state,
                action=(ACTION_DONE, -1, -1),
                source_trace_index=int(source_trace_index),
                source_block_id=int(spans[-1].block_id + 1) if spans else 1,
            )
        )
        done_res = env.step(SatAction.done())
        if not bool(done_res.done):
            raise RuntimeError(f"expected DONE to terminate trace {source_trace_index}")

    final_state = env.get_state()
    if final_state.status == SatEnvStatus.RUNNING:
        raise RuntimeError(f"trace replay did not terminate for trace {source_trace_index}")

    return examples


def summarize_examples(examples: Sequence[DecodedSATExample], num_traces: int) -> ExampleDecodeSummary:
    num_assign = int(sum(1 for ex in examples if ex.action[0] == ACTION_ASSIGN))
    num_backtrack = int(sum(1 for ex in examples if ex.action[0] == ACTION_BACKTRACK))
    num_done = int(sum(1 for ex in examples if ex.action[0] == ACTION_DONE))
    return ExampleDecodeSummary(
        num_traces=int(num_traces),
        num_examples=int(len(examples)),
        num_assign=int(num_assign),
        num_backtrack=int(num_backtrack),
        num_done=int(num_done),
    )


def _cache_signature(data_path: Path, version: str) -> str:
    stat = data_path.stat()
    payload = json.dumps(
        {
            "path": str(data_path.resolve()),
            "version": str(version),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_records_to_examples(
    records: Sequence[Dict[str, Any]],
    *,
    target_examples: Optional[int] = None,
) -> Tuple[List[DecodedSATExample], int]:
    tokenizer = SATInterleavedTokenizer()
    decoded: List[DecodedSATExample] = []
    decoded_traces = 0
    for trace_idx, record in enumerate(records):
        trace_examples = decode_trace_record_to_examples(
            record,
            source_trace_index=int(trace_idx),
            tokenizer=tokenizer,
        )
        decoded.extend(trace_examples)
        decoded_traces += 1
        if decoded_traces % 100 == 0:
            logger.info(
                "decoded_traces=%d decoded_examples=%d sample_action_hist={assign:%d backtrack:%d done:%d}",
                int(decoded_traces),
                int(len(decoded)),
                int(sum(1 for ex in decoded if ex.action[0] == ACTION_ASSIGN)),
                int(sum(1 for ex in decoded if ex.action[0] == ACTION_BACKTRACK)),
                int(sum(1 for ex in decoded if ex.action[0] == ACTION_DONE)),
            )
        if target_examples is not None and len(decoded) >= int(target_examples):
            break
    return decoded, int(decoded_traces)


def load_or_decode_examples(
    *,
    data_path: Path,
    version: str,
    force_rebuild_cache: bool,
    smoke: bool,
    smoke_target_examples: int,
) -> Tuple[List[Dict[str, Any]], List[DecodedSATExample], ExampleDecodeSummary]:
    records = _load_trace_records(data_path)
    cache_key = _cache_signature(data_path, version)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not smoke and not bool(force_rebuild_cache) and CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            cached = pickle.load(f)
        if (
            isinstance(cached, dict)
            and str(cached.get("cache_key", "")) == str(cache_key)
            and isinstance(cached.get("examples"), list)
        ):
            examples = list(cached["examples"])
            summary = summarize_examples(examples, num_traces=len(records))
            logger.info(
                "loaded_example_cache path=%s traces=%d examples=%d assign=%d backtrack=%d done=%d",
                str(CACHE_PATH),
                int(summary.num_traces),
                int(summary.num_examples),
                int(summary.num_assign),
                int(summary.num_backtrack),
                int(summary.num_done),
            )
            return records, examples, summary

    target_examples = int(smoke_target_examples) if smoke else None
    examples, decoded_traces = _decode_records_to_examples(
        records,
        target_examples=target_examples,
    )
    summary = summarize_examples(examples, num_traces=decoded_traces if smoke else len(records))

    if not smoke:
        payload = {
            "cache_key": str(cache_key),
            "data_path": str(data_path.resolve()),
            "version": str(version),
            "summary": asdict(summary),
            "examples": examples,
        }
        with CACHE_PATH.open("wb") as f:
            pickle.dump(payload, f)
        logger.info(
            "saved_example_cache path=%s traces=%d examples=%d assign=%d backtrack=%d done=%d",
            str(CACHE_PATH),
            int(summary.num_traces),
            int(summary.num_examples),
            int(summary.num_assign),
            int(summary.num_backtrack),
            int(summary.num_done),
        )
    else:
        logger.info(
            "smoke_decode_subset traces=%d examples=%d assign=%d backtrack=%d done=%d",
            int(summary.num_traces),
            int(summary.num_examples),
            int(summary.num_assign),
            int(summary.num_backtrack),
            int(summary.num_done),
        )

    return records, examples, summary


@lru_cache(maxsize=8192)
def _static_clause_graph(
    clauses: Tuple[Tuple[int, int, int], ...],
    num_vars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_counts = np.zeros((int(num_vars),), dtype=np.float32)
    neg_counts = np.zeros((int(num_vars),), dtype=np.float32)
    edge_con_idx: List[int] = []
    edge_var_idx: List[int] = []
    edge_features: List[List[float]] = []
    clause_len_norm = np.zeros((len(clauses),), dtype=np.float32)

    for con_idx, clause in enumerate(clauses):
        clause_len = int(len(clause))
        clause_len_norm[int(con_idx)] = float(clause_len) / 3.0
        denom = max(clause_len - 1, 1)
        for lit_pos, lit in enumerate(clause):
            var_idx = int(abs(int(lit)) - 1)
            edge_con_idx.append(int(con_idx))
            edge_var_idx.append(int(var_idx))
            sign = 1.0 if int(lit) > 0 else -1.0
            lit_pos_norm = float(lit_pos) / float(denom)
            edge_features.append([float(sign), float(lit_pos_norm)])
            if int(lit) > 0:
                pos_counts[int(var_idx)] += 1.0
            else:
                neg_counts[int(var_idx)] += 1.0

    con_type = np.zeros((len(clauses),), dtype=np.int64)
    return (
        pos_counts,
        neg_counts,
        con_type,
        np.asarray(edge_con_idx, dtype=np.int64),
        np.asarray(edge_var_idx, dtype=np.int64),
        np.asarray(edge_features, dtype=np.float32),
        clause_len_norm,
    )


def _literal_truth(assignment: np.ndarray, lit: int) -> int:
    var_idx = int(abs(int(lit)) - 1)
    value = int(assignment[int(var_idx)])
    if value == -1:
        return 0
    signed_value = -1 if value == 0 else 1
    if signed_value == (1 if int(lit) > 0 else -1):
        return 1
    return -1


def build_factor_gnn_item_from_example(
    example: DecodedSATExample,
    *,
    max_vars: int,
    max_constraints: int,
) -> Dict[str, Any]:
    num_vars = int(example.assignment.shape[0])
    num_constraints = int(len(example.clauses))
    if num_vars > int(max_vars):
        raise ValueError(f"num_vars={num_vars} exceeds max_vars={max_vars}")
    if num_constraints > int(max_constraints):
        raise ValueError(
            f"num_constraints={num_constraints} exceeds max_constraints={max_constraints}"
        )

    (
        pos_counts,
        neg_counts,
        con_type,
        edge_con_idx,
        edge_var_idx,
        edge_features,
        clause_len_norm,
    ) = _static_clause_graph(tuple(example.clauses), int(num_vars))

    clause_count_denom = max(float(num_constraints), 1.0)
    depth_norm = float(example.decision_depth) / 50.0
    var_features = np.stack(
        [
            np.full((num_vars,), depth_norm, dtype=np.float32),
            pos_counts / clause_count_denom,
            neg_counts / clause_count_denom,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    satisfied_flag = np.asarray(
        [
            1.0
            if any(_literal_truth(example.assignment, int(lit)) == 1 for lit in clause)
            else 0.0
            for clause in example.clauses
        ],
        dtype=np.float32,
    )
    con_features = np.stack([clause_len_norm, satisfied_flag], axis=-1).astype(
        np.float32,
        copy=False,
    )

    return {
        "num_vars": int(num_vars),
        "num_constraints": int(num_constraints),
        "var_features": var_features,
        "var_domain_mask": np.asarray(example.domain_mask, dtype=np.bool_),
        "var_nogood_mask": np.zeros((num_vars, SAT_MAX_DOMAIN), dtype=np.bool_),
        "var_assigned": np.asarray(example.assignment, dtype=np.int64),
        "con_type": np.asarray(con_type, dtype=np.int64),
        "con_features": con_features,
        "edge_con_idx": np.asarray(edge_con_idx, dtype=np.int64),
        "edge_var_idx": np.asarray(edge_var_idx, dtype=np.int64),
        "edge_features": np.asarray(edge_features, dtype=np.float32),
        "stack_depth": int(example.decision_depth),
        "propagation_pending": bool(example.propagation_pending),
        "has_conflict": bool(example.has_conflict),
        "propagation_mode": 1,
        "action_type": int(_action_to_type_id(example.action[0])),
        "action_var": int(example.action[1]),
        "action_value": int(example.action[2]),
        "source_trace_index": int(example.source_trace_index),
        "source_block_id": int(example.source_block_id),
    }


def collate_factor_gnn_sat(
    batch: List[Dict[str, Any]],
    *,
    max_vars: int,
    max_constraints: int,
    max_domain: int = SAT_MAX_DOMAIN,
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("empty batch")

    batch_size = len(batch)
    var_feat_dim = int(batch[0]["var_features"].shape[1])
    con_feat_dim = int(batch[0]["con_features"].shape[1])
    edge_feat_dim = int(batch[0]["edge_features"].shape[1])
    max_edges = max(int(item["edge_con_idx"].shape[0]) for item in batch)

    var_features = np.zeros((batch_size, int(max_vars), var_feat_dim), dtype=np.float32)
    var_domain_mask = np.zeros((batch_size, int(max_vars), int(max_domain)), dtype=np.bool_)
    var_nogood_mask = np.zeros((batch_size, int(max_vars), int(max_domain)), dtype=np.bool_)
    var_assigned = np.full((batch_size, int(max_vars)), -1, dtype=np.int64)
    con_type = np.zeros((batch_size, int(max_constraints)), dtype=np.int64)
    con_features = np.zeros(
        (batch_size, int(max_constraints), con_feat_dim),
        dtype=np.float32,
    )
    edge_con_idx = np.zeros((batch_size, int(max_edges)), dtype=np.int64)
    edge_var_idx = np.zeros((batch_size, int(max_edges)), dtype=np.int64)
    edge_features = np.zeros(
        (batch_size, int(max_edges), edge_feat_dim),
        dtype=np.float32,
    )
    var_mask = np.zeros((batch_size, int(max_vars)), dtype=np.bool_)
    con_mask = np.zeros((batch_size, int(max_constraints)), dtype=np.bool_)
    edge_mask = np.zeros((batch_size, int(max_edges)), dtype=np.bool_)
    global_features = np.zeros((batch_size, 4), dtype=np.float32)
    action_type = np.zeros((batch_size,), dtype=np.int64)
    action_var = np.full((batch_size,), -1, dtype=np.int64)
    action_value = np.full((batch_size,), -1, dtype=np.int64)

    for row_idx, item in enumerate(batch):
        num_vars = int(item["num_vars"])
        num_constraints = int(item["num_constraints"])
        num_edges = int(item["edge_con_idx"].shape[0])
        if num_vars > int(max_vars):
            raise ValueError(f"num_vars={num_vars} exceeds max_vars={max_vars}")
        if num_constraints > int(max_constraints):
            raise ValueError(
                f"num_constraints={num_constraints} exceeds max_constraints={max_constraints}"
            )

        var_features[row_idx, :num_vars] = item["var_features"]
        var_domain_mask[row_idx, :num_vars] = item["var_domain_mask"]
        var_nogood_mask[row_idx, :num_vars] = item["var_nogood_mask"]
        var_assigned[row_idx, :num_vars] = item["var_assigned"]
        con_type[row_idx, :num_constraints] = item["con_type"]
        con_features[row_idx, :num_constraints] = item["con_features"]
        edge_con_idx[row_idx, :num_edges] = item["edge_con_idx"]
        edge_var_idx[row_idx, :num_edges] = item["edge_var_idx"]
        edge_features[row_idx, :num_edges] = item["edge_features"]
        var_mask[row_idx, :num_vars] = True
        con_mask[row_idx, :num_constraints] = True
        edge_mask[row_idx, :num_edges] = True
        global_features[row_idx] = np.asarray(
            [
                float(item["stack_depth"]) / 50.0,
                float(item["propagation_pending"]),
                float(item["has_conflict"]),
                1.0,
            ],
            dtype=np.float32,
        )
        action_type[row_idx] = int(item["action_type"])
        action_var[row_idx] = int(item["action_var"])
        action_value[row_idx] = int(item["action_value"])

    return {
        "var_features": torch.from_numpy(var_features),
        "var_domain_mask": torch.from_numpy(var_domain_mask),
        "var_nogood_mask": torch.from_numpy(var_nogood_mask),
        "var_assigned": torch.from_numpy(var_assigned),
        "con_type": torch.from_numpy(con_type),
        "con_features": torch.from_numpy(con_features),
        "edge_con_idx": torch.from_numpy(edge_con_idx),
        "edge_var_idx": torch.from_numpy(edge_var_idx),
        "edge_features": torch.from_numpy(edge_features),
        "var_mask": torch.from_numpy(var_mask),
        "con_mask": torch.from_numpy(con_mask),
        "edge_mask": torch.from_numpy(edge_mask),
        "global_features": torch.from_numpy(global_features),
        "action_type": torch.from_numpy(action_type),
        "action_var": torch.from_numpy(action_var),
        "action_value": torch.from_numpy(action_value),
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def create_factor_gnn_model(
    *,
    max_vars: int,
    max_constraints: int,
    d_model: int,
    num_layers: int,
    dropout: float,
) -> FactorGNN:
    return FactorGNN(
        max_vars=int(max_vars),
        max_constraints=int(max_constraints),
        max_domain=SAT_MAX_DOMAIN,
        num_constraint_types=SAT_NUM_CONSTRAINT_TYPES,
        d_model=int(d_model),
        num_layers=int(num_layers),
        dropout=float(dropout),
        var_feature_dim=3,
        con_feature_dim=2,
        edge_feature_dim=2,
        global_feature_dim=4,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(int(param.numel()) for param in model.parameters()))


def compute_factor_gnn_loss(
    batch: Dict[str, torch.Tensor],
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]],
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    assign_logits, backtrack_logit, done_logit, _ = outputs
    batch_size, num_vars, max_domain = assign_logits.shape
    if int(max_domain) != SAT_MAX_DOMAIN:
        raise ValueError(f"expected SAT domain size 2, got {max_domain}")

    flat_logits = assign_logits.reshape(batch_size, num_vars * max_domain)
    best_assign_logit = flat_logits.max(dim=-1).values
    action_type_logits = torch.stack(
        [best_assign_logit, backtrack_logit.squeeze(-1), done_logit.squeeze(-1)],
        dim=-1,
    )

    action_type_target = batch["action_type"].long()
    per_example_type_loss = F.cross_entropy(
        action_type_logits,
        action_type_target,
        reduction="none",
    )

    assign_examples = action_type_target == 0
    per_example_assign_loss = torch.zeros_like(per_example_type_loss)
    assign_target_flat = batch["action_var"].long() * int(max_domain) + batch[
        "action_value"
    ].long()

    if bool(assign_examples.any()):
        per_example_assign_loss[assign_examples] = F.cross_entropy(
            flat_logits[assign_examples],
            assign_target_flat[assign_examples],
            reduction="none",
        )

    loss = (per_example_type_loss + per_example_assign_loss).mean()

    action_type_pred = action_type_logits.argmax(dim=-1)
    type_acc = float((action_type_pred == action_type_target).float().mean().item())
    if bool(assign_examples.any()):
        assign_pred_flat = flat_logits[assign_examples].argmax(dim=-1)
        assign_acc = float(
            (assign_pred_flat == assign_target_flat[assign_examples]).float().mean().item()
        )
        assign_loss_mean = float(per_example_assign_loss[assign_examples].mean().item())
    else:
        assign_acc = 0.0
        assign_loss_mean = 0.0

    metrics = {
        "loss": float(loss.item()),
        "type_loss": float(per_example_type_loss.mean().item()),
        "assign_loss": float(assign_loss_mean),
        "type_acc": float(type_acc),
        "assign_acc_when_assign": float(assign_acc),
        "num_examples": float(batch_size),
        "num_assign_examples": float(int(assign_examples.sum().item())),
    }
    aux = {
        "action_type_logits": action_type_logits.detach(),
        "action_type_pred": action_type_pred.detach(),
        "assign_pred_flat": flat_logits.argmax(dim=-1).detach(),
        "assign_target_flat": assign_target_flat.detach(),
    }
    return loss, metrics, aux


def _first_batch_samples(
    batch: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    max_samples: int = 3,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    num_rows = min(int(batch["action_type"].shape[0]), int(max_samples))
    for row_idx in range(num_rows):
        pred_flat = int(aux["assign_pred_flat"][row_idx].item())
        target_flat = int(aux["assign_target_flat"][row_idx].item())
        samples.append(
            {
                "target_action_type": int(batch["action_type"][row_idx].item()),
                "pred_action_type": int(aux["action_type_pred"][row_idx].item()),
                "target_var": int(batch["action_var"][row_idx].item()),
                "target_value": int(batch["action_value"][row_idx].item()),
                "pred_var": int(pred_flat // SAT_MAX_DOMAIN),
                "pred_value": int(pred_flat % SAT_MAX_DOMAIN),
                "target_flat": int(target_flat),
                "pred_flat": int(pred_flat),
            }
        )
    return samples


def run_epoch(
    *,
    loader: DataLoader,
    model: FactorGNN,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    train: bool,
) -> EpochSummary:
    model.train(mode=bool(train))
    accum_examples = 0
    accum_assign_examples = 0
    accum = {
        "loss": 0.0,
        "type_loss": 0.0,
        "assign_loss": 0.0,
        "type_acc": 0.0,
        "assign_acc_when_assign": 0.0,
    }
    first_samples: List[Dict[str, Any]] = []

    model_inputs = {
        "var_features",
        "var_domain_mask",
        "var_nogood_mask",
        "var_assigned",
        "con_type",
        "con_features",
        "edge_con_idx",
        "edge_var_idx",
        "edge_features",
        "var_mask",
        "con_mask",
        "edge_mask",
        "global_features",
    }

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        forward_inputs = {key: value for key, value in batch.items() if key in model_inputs}
        with torch.set_grad_enabled(bool(train)):
            outputs = model(**forward_inputs)
            loss, batch_metrics, aux = compute_factor_gnn_loss(batch, outputs)
            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required when train=True")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_examples = int(batch_metrics["num_examples"])
        batch_assign_examples = int(batch_metrics["num_assign_examples"])
        accum_examples += int(batch_examples)
        accum_assign_examples += int(batch_assign_examples)
        accum["loss"] += float(batch_metrics["loss"]) * float(batch_examples)
        accum["type_loss"] += float(batch_metrics["type_loss"]) * float(batch_examples)
        accum["assign_loss"] += float(batch_metrics["assign_loss"]) * float(
            max(batch_assign_examples, 1)
        )
        accum["type_acc"] += float(batch_metrics["type_acc"]) * float(batch_examples)
        accum["assign_acc_when_assign"] += float(
            batch_metrics["assign_acc_when_assign"]
        ) * float(batch_assign_examples)

        if batch_idx == 0:
            first_samples = _first_batch_samples(batch, aux)

    if accum_examples <= 0:
        raise RuntimeError("encountered empty split")

    return EpochSummary(
        loss=float(accum["loss"] / float(accum_examples)),
        type_loss=float(accum["type_loss"] / float(accum_examples)),
        assign_loss=float(accum["assign_loss"] / max(float(accum_assign_examples), 1.0)),
        type_acc=float(accum["type_acc"] / float(accum_examples)),
        assign_acc_when_assign=float(
            accum["assign_acc_when_assign"] / max(float(accum_assign_examples), 1.0)
        ),
        num_examples=int(accum_examples),
        num_assign_examples=int(accum_assign_examples),
        samples=first_samples,
    )


def _split_trace_indices(num_traces: int, val_split: float, seed: int) -> Tuple[set[int], set[int]]:
    if int(num_traces) < 2:
        raise ValueError("need at least 2 traces to create train/val split")
    order = list(range(int(num_traces)))
    random.Random(int(seed)).shuffle(order)
    split = int(round((1.0 - float(val_split)) * float(num_traces)))
    split = max(1, min(split, int(num_traces) - 1))
    return set(order[:split]), set(order[split:])


def _filter_examples_by_trace_ids(
    examples: Sequence[DecodedSATExample],
    allowed_trace_ids: Iterable[int],
) -> List[DecodedSATExample]:
    allowed = {int(x) for x in allowed_trace_ids}
    return [ex for ex in examples if int(ex.source_trace_index) in allowed]


def _resolve_device(device_arg: str) -> torch.device:
    device = torch.device(str(device_arg))
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAT n=50 FactorGNN baseline")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_vars", type=int, default=60)
    parser.add_argument("--max_constraints", type=int, default=300)
    parser.add_argument("--d_model", type=int, default=112)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(str(args.device))

    smoke_target_examples = max(int(args.batch_size) * 4, 64)
    records, examples, decode_summary = load_or_decode_examples(
        data_path=data_path,
        version=str(args.version),
        force_rebuild_cache=bool(args.force_rebuild_cache),
        smoke=bool(args.smoke),
        smoke_target_examples=int(smoke_target_examples),
    )
    if len(examples) < 2:
        raise RuntimeError("not enough decoded examples to train")

    effective_trace_count = max(
        int(max(ex.source_trace_index for ex in examples) + 1),
        2,
    )
    train_trace_ids, val_trace_ids = _split_trace_indices(
        num_traces=int(effective_trace_count),
        val_split=float(args.val_split),
        seed=int(args.seed),
    )
    train_examples = _filter_examples_by_trace_ids(examples, train_trace_ids)
    val_examples = _filter_examples_by_trace_ids(examples, val_trace_ids)
    if not train_examples or not val_examples:
        raise RuntimeError(
            f"empty split after trace split: train={len(train_examples)} val={len(val_examples)}"
        )

    train_ds = FactorGNNSATDataset(
        train_examples,
        max_vars=int(args.max_vars),
        max_constraints=int(args.max_constraints),
    )
    val_ds = FactorGNNSATDataset(
        val_examples,
        max_vars=int(args.max_vars),
        max_constraints=int(args.max_constraints),
    )
    collate_kwargs = {
        "max_vars": int(args.max_vars),
        "max_constraints": int(args.max_constraints),
        "max_domain": SAT_MAX_DOMAIN,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_factor_gnn_sat(batch, **collate_kwargs),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_factor_gnn_sat(batch, **collate_kwargs),
    )

    model = create_factor_gnn_model(
        max_vars=int(args.max_vars),
        max_constraints=int(args.max_constraints),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    parameter_count = count_parameters(model)

    config = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "val_split": float(args.val_split),
        "max_vars": int(args.max_vars),
        "max_constraints": int(args.max_constraints),
        "max_domain": int(SAT_MAX_DOMAIN),
        "num_constraint_types": int(SAT_NUM_CONSTRAINT_TYPES),
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "device": str(device),
        "version": str(args.version),
        "parameter_count": int(parameter_count),
        "decode_summary": asdict(decode_summary),
        "train_examples": int(len(train_examples)),
        "val_examples": int(len(val_examples)),
        "train_traces": int(len(train_trace_ids)),
        "val_traces": int(len(val_trace_ids)),
        "cache_path": str(CACHE_PATH),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(
        "factor_gnn_setup traces=%d examples=%d train_examples=%d val_examples=%d assign=%d backtrack=%d done=%d param_count=%d device=%s",
        int(decode_summary.num_traces),
        int(decode_summary.num_examples),
        int(len(train_examples)),
        int(len(val_examples)),
        int(decode_summary.num_assign),
        int(decode_summary.num_backtrack),
        int(decode_summary.num_done),
        int(parameter_count),
        str(device),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    planned_total_steps = int(len(train_loader) * int(max(int(args.epochs), 1)))
    warmup_steps = int(float(planned_total_steps) * float(args.warmup_ratio))
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
                    * (float(step) - float(warmup_steps))
                    / max(float(planned_total_steps - warmup_steps), 1.0)
                )
            )
        ),
    )

    if bool(args.smoke):
        train_batch = move_batch_to_device(next(iter(train_loader)), device)
        val_batch = move_batch_to_device(next(iter(val_loader)), device)
        model_inputs = {
            "var_features",
            "var_domain_mask",
            "var_nogood_mask",
            "var_assigned",
            "con_type",
            "con_features",
            "edge_con_idx",
            "edge_var_idx",
            "edge_features",
            "var_mask",
            "con_mask",
            "edge_mask",
            "global_features",
        }
        forward_inputs = {key: value for key, value in train_batch.items() if key in model_inputs}
        outputs = model(**forward_inputs)
        train_loss, train_metrics, train_aux = compute_factor_gnn_loss(train_batch, outputs)
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        grad_norm = float(
            torch.sqrt(
                sum(
                    torch.sum(param.grad.detach() ** 2)
                    for param in model.parameters()
                    if param.grad is not None
                )
            ).item()
        )
        with torch.no_grad():
            val_outputs = model(
                **{key: value for key, value in val_batch.items() if key in model_inputs}
            )
            val_loss, val_metrics, val_aux = compute_factor_gnn_loss(val_batch, val_outputs)

        smoke_ckpt = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "smoke": True,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(smoke_ckpt, output_dir / "smoke.pt")
        smoke_payload = {
            "smoke": True,
            "parameter_count": int(parameter_count),
            "planned_total_steps": int(planned_total_steps),
            "train_batch_shape": {
                key: [int(x) for x in tensor.shape]
                for key, tensor in train_batch.items()
                if isinstance(tensor, torch.Tensor)
            },
            "val_batch_shape": {
                key: [int(x) for x in tensor.shape]
                for key, tensor in val_batch.items()
                if isinstance(tensor, torch.Tensor)
            },
            "train_loss": float(train_loss.item()),
            "train_type_acc": float(train_metrics["type_acc"]),
            "train_assign_acc_when_assign": float(train_metrics["assign_acc_when_assign"]),
            "train_samples": _first_batch_samples(train_batch, train_aux),
            "val_loss": float(val_loss.item()),
            "val_type_acc": float(val_metrics["type_acc"]),
            "val_assign_acc_when_assign": float(val_metrics["assign_acc_when_assign"]),
            "val_samples": _first_batch_samples(val_batch, val_aux),
            "grad_norm": float(grad_norm),
            "checkpoint": str(output_dir / "smoke.pt"),
        }
        with (output_dir / "smoke.json").open("w", encoding="utf-8") as f:
            json.dump(smoke_payload, f, indent=2)
        logger.info(
            "smoke_complete output=%s train_loss=%.4f val_loss=%.4f grad_norm=%.4f",
            str(output_dir / "smoke.json"),
            float(train_loss.item()),
            float(val_loss.item()),
            float(grad_norm),
        )
        return

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        train_summary = run_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train=True,
        )
        with torch.no_grad():
            val_summary = run_epoch(
                loader=val_loader,
                model=model,
                optimizer=None,
                scheduler=None,
                device=device,
                train=False,
            )
        row = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": asdict(train_summary),
            "val": asdict(val_summary),
        }
        history.append(row)

        logger.info(
            "epoch=%d/%d train_loss=%.4f train_type_acc=%.4f train_assign_acc=%.4f val_loss=%.4f val_type_acc=%.4f val_assign_acc=%.4f sample=%s",
            int(epoch),
            int(args.epochs),
            float(train_summary.loss),
            float(train_summary.type_acc),
            float(train_summary.assign_acc_when_assign),
            float(val_summary.loss),
            float(val_summary.type_acc),
            float(val_summary.assign_acc_when_assign),
            val_summary.samples[:1],
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": int(epoch),
            "history": history,
            "train_loss": float(train_summary.loss),
            "val_loss": float(val_summary.loss),
            "val_type_acc": float(val_summary.type_acc),
            "val_assign_acc_when_assign": float(val_summary.assign_acc_when_assign),
            "parameter_count": int(parameter_count),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if float(val_summary.loss) < float(best_val_loss):
            best_val_loss = float(val_summary.loss)
            torch.save(checkpoint, output_dir / "best.pt")

    metrics = {
        "config": config,
        "best_val_loss": float(best_val_loss),
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "training_complete output=%s best_val_loss=%.4f parameter_count=%d",
        str(output_dir),
        float(best_val_loss),
        int(parameter_count),
    )


if __name__ == "__main__":
    main()
