#!/usr/bin/env python3
"""History-transplant behavioral test for SAT SSA vs causal checkpoints.

This script mirrors the graph-coloring history-transplant diagnostic but uses
the SAT n=50 enriched trace format from the training pipeline:

    STATE v_i {U|T|F} ... SEP  v_j  {T|F}  OK  v_j  {T|F}

Conflict blocks preserve the original generator structure:

    STATE ... SEP  CONFLICT  C<clause_id>  BJ  L<backjump_level>

Matched state keys include the reviewer-requested full state:
partial assignment, propagated domains, conflict status, decision level, and
the tried-alternatives set at the current decision-stack prefix.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.dsl import SatAction, SatActionType
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator
from sat.interleaved_tokenizer import SATInterleavedTokenizer
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


CanonicalStateKey = Tuple[
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, Tuple[int, ...]], ...],
    bool,
    int,
    Tuple[Tuple[int, int], ...],
]


@dataclass
class DecisionPoint:
    position: int
    block_id: int
    block_start: int
    assignment: List[int]
    domains: List[List[int]]
    selectable_vars: List[int]
    action_candidates: List[Tuple[int, int]]
    canonical_state: CanonicalStateKey
    conflict_status: bool
    decision_level: int
    tried_alternatives: List[Tuple[int, int]]


@dataclass
class OracleTrace:
    tokens: List[int]
    block_ids: List[int]
    decision_points: List[DecisionPoint]
    clause_prefix_len: int


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _append_tokens(
    sequence: List[int],
    block_ids: List[int],
    tokens: Iterable[int],
    *,
    block_id: int,
    max_seq_len: int,
) -> bool:
    chunk = [int(x) for x in tokens]
    if len(sequence) + len(chunk) > int(max_seq_len):
        return False
    sequence.extend(chunk)
    block_ids.extend([int(block_id)] * len(chunk))
    return True


def _ensure_checkpoint_exists(checkpoint_path: Path, arg_name: str) -> None:
    if checkpoint_path.exists():
        return
    raise FileNotFoundError(
        f"{arg_name} not found: {checkpoint_path}. "
        "Pass an explicit checkpoint path that exists on disk."
    )


def _as_three_lit_clauses(
    clauses: Sequence[Tuple[int, ...]],
) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for clause in clauses:
        if len(clause) != 3:
            raise ValueError(f"expected 3-literal clause, got len={len(clause)}")
        out.append((int(clause[0]), int(clause[1]), int(clause[2])))
    return out


def _variable_occurrence_counts(
    clauses: Sequence[Tuple[int, ...]],
    num_vars: int,
) -> np.ndarray:
    counts = np.zeros((int(num_vars),), dtype=np.int64)
    for clause in clauses:
        for lit in clause:
            var = int(abs(int(lit)) - 1)
            if 0 <= var < int(num_vars):
                counts[int(var)] += 1
    return counts


def _sorted_unassigned_vars(
    state: SatState,
    occurrence: np.ndarray,
    *,
    state_sort: str,
) -> List[int]:
    unassigned = [
        int(var_id)
        for var_id in range(int(state.num_vars))
        if int(state.assignment[int(var_id)]) == 0
    ]
    if str(state_sort) == "lexical":
        return sorted(unassigned)
    if str(state_sort) != "vsids":
        raise ValueError(f"unsupported state_sort={state_sort}")
    return sorted(
        unassigned,
        key=lambda var_id: (
            -float(state.activity[int(var_id)]),
            -int(occurrence[int(var_id)]),
            int(var_id),
        ),
    )


def _domain_token_for_var(
    env: SatEnv,
    state: SatState,
    var_id: int,
    tokenizer: SATInterleavedTokenizer,
) -> Optional[int]:
    domain = env._effective_domain(state, int(var_id))
    has_true = 1 in domain
    has_false = -1 in domain

    if has_true and has_false:
        return int(tokenizer.UNASSIGNED)
    if has_true and not has_false:
        return int(tokenizer.TRUE_VAL)
    if has_false and not has_true:
        return int(tokenizer.FALSE_VAL)
    return None


def _build_enriched_state_tokens(
    env: SatEnv,
    state: SatState,
    sorted_vars: Sequence[int],
    tokenizer: SATInterleavedTokenizer,
) -> List[int]:
    tokens: List[int] = [int(tokenizer.STATE)]
    for var_id in sorted_vars:
        domain_tok = _domain_token_for_var(env, state, int(var_id), tokenizer)
        if domain_tok is None:
            continue
        tokens.extend([int(tokenizer.var_token(int(var_id))), int(domain_tok)])
    tokens.append(int(tokenizer.SEP))
    return tokens


def _value_bit(val: int) -> int:
    if int(val) == -1:
        return 1
    if int(val) == 1:
        return 2
    raise ValueError(f"expected value in {{-1, +1}}, got {val}")


def _assign_target_from_signed(signed_value: int) -> int:
    if int(signed_value) == 1:
        return 1
    if int(signed_value) == -1:
        return 0
    raise ValueError(f"expected signed SAT value in {{-1,+1}}, got {signed_value}")


def _value_token_from_signed(
    signed_value: int,
    tokenizer: SATInterleavedTokenizer,
) -> int:
    if int(signed_value) == 1:
        return int(tokenizer.TRUE_VAL)
    if int(signed_value) == -1:
        return int(tokenizer.FALSE_VAL)
    raise ValueError(f"expected signed SAT value in {{-1,+1}}, got {signed_value}")


def _prefix_key_from_assignment(assignment: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    nz = np.nonzero(assignment)[0]
    return tuple(sorted((int(var_id), int(assignment[int(var_id)])) for var_id in nz))


def _extract_tried_alternatives(state: SatState) -> Tuple[Tuple[int, int], ...]:
    if not state.decision_stack:
        return ()

    top = state.decision_stack[-1]
    top_var = int(top.decision_var)
    if int(state.assignment[top_var]) != 0:
        return ()

    failed: List[Tuple[int, int]] = []
    if int(top.failed_mask) & int(_value_bit(-1)):
        failed.append((int(top_var), -1))
    if int(top.failed_mask) & int(_value_bit(1)):
        failed.append((int(top_var), 1))
    return tuple(sorted(failed))


def _propagation_domain_for_var(state: SatState, var_id: int) -> Tuple[int, ...]:
    """Propagation-only domain for canonical-state matching.

    SAT unit propagation is applied eagerly in the environment, so any variable
    forced by propagation is already assigned and therefore excluded from the
    unassigned-domain portion of the canonical key. The remaining unassigned
    variables all have the raw propagated domain {-1, +1}; branch-local retry
    restrictions are represented separately by the tried-alternatives set.
    """

    if int(state.assignment[int(var_id)]) != 0:
        return (int(state.assignment[int(var_id)]),)
    return (-1, 1)


def _canonical_state_key(env: SatEnv, state: SatState) -> CanonicalStateKey:
    assignment_key = _prefix_key_from_assignment(state.assignment)
    domains_key = tuple(
        (
            int(var_id),
            tuple(int(val) for val in _propagation_domain_for_var(state, int(var_id))),
        )
        for var_id in range(int(state.num_vars))
        if int(state.assignment[int(var_id)]) == 0
    )
    conflict_status = bool(state.conflict_clause is not None)
    decision_level = int(len(state.decision_stack))
    tried_alternatives = _extract_tried_alternatives(state)
    return (
        assignment_key,
        domains_key,
        bool(conflict_status),
        int(decision_level),
        tried_alternatives,
    )


def _selectable_vars_from_state(env: SatEnv, state: SatState) -> List[int]:
    if state.selected_var is not None:
        return [int(state.selected_var)]

    vars_out: List[int] = []
    for action in env.get_valid_actions():
        if action.type == SatActionType.SELECT_VAR and action.target is not None:
            vars_out.append(int(action.target))
    return sorted(set(vars_out))


def _action_candidates_from_state(
    env: SatEnv,
    state: SatState,
    selectable_vars: Sequence[int],
) -> List[Tuple[int, int]]:
    actions: List[Tuple[int, int]] = []
    for var_id in selectable_vars:
        domain = sorted(int(val) for val in env._effective_domain(state, int(var_id)))
        for signed_value in domain:
            actions.append((int(var_id), int(signed_value)))
    return actions


def _choose_var_with_random_tie(
    state: SatState,
    selectable_vars: Sequence[int],
    occurrence: np.ndarray,
    rng: random.Random,
) -> int:
    best_score: Optional[Tuple[float, int]] = None
    best_vars: List[int] = []
    for var_id in selectable_vars:
        score = (float(state.activity[int(var_id)]), int(occurrence[int(var_id)]))
        if best_score is None or score > best_score:
            best_score = score
            best_vars = [int(var_id)]
        elif score == best_score:
            best_vars.append(int(var_id))

    if not best_vars:
        raise RuntimeError("no selectable variables available")
    rng.shuffle(best_vars)
    return int(best_vars[0])


def _random_signed_value(
    env: SatEnv,
    state: SatState,
    selected_var: int,
    rng: random.Random,
) -> int:
    domain = sorted(int(val) for val in env._effective_domain(state, int(selected_var)))
    if not domain:
        raise RuntimeError("selected variable has empty effective domain")
    return int(rng.choice(domain))


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


def _history_difference_tokens(
    history_a: Sequence[int],
    history_b: Sequence[int],
) -> int:
    min_len = min(int(len(history_a)), int(len(history_b)))
    diffs = sum(
        1
        for idx in range(int(min_len))
        if int(history_a[int(idx)]) != int(history_b[int(idx)])
    )
    return int(diffs + abs(int(len(history_a)) - int(len(history_b))))


def generate_oracle_trace_with_random_ties(
    *,
    clauses: Sequence[Tuple[int, ...]],
    num_vars: int,
    planted_solution: Optional[np.ndarray],
    max_seq_len: int,
    max_steps: int,
    tie_seed: int,
    state_sort: str,
) -> OracleTrace:
    """Run a DPLL-ish stochastic rollout and emit enriched SAT trace tokens."""

    tokenizer = SATInterleavedTokenizer()
    clauses_3 = _as_three_lit_clauses(clauses)
    env = SatEnv(
        clauses=[tuple(int(x) for x in clause) for clause in clauses_3],
        num_vars=int(num_vars),
        planted_solution=None
        if planted_solution is None
        else np.array(planted_solution, dtype=np.int64, copy=True),
        mode="strict",
        max_steps=int(max_steps * 8 + 20),
    )
    env.reset()

    rng = random.Random(int(tie_seed))
    occurrence = _variable_occurrence_counts(clauses_3, int(num_vars))

    tokens: List[int] = tokenizer.build_clause_prefix(list(clauses_3), int(num_vars))
    clause_prefix_len = int(len(tokens))
    block_ids: List[int] = [0] * int(clause_prefix_len)
    current_block = 0
    decision_points: List[DecisionPoint] = []

    for _step in range(int(max_steps)):
        state = _advance_propagation(env)
        if state.status == SatEnvStatus.SUCCESS:
            _append_tokens(
                tokens,
                block_ids,
                [int(tokenizer.SOLVED)],
                block_id=max(int(current_block), 1),
                max_seq_len=int(max_seq_len),
            )
            break
        if state.status == SatEnvStatus.FAILURE:
            _append_tokens(
                tokens,
                block_ids,
                [int(tokenizer.FAILED)],
                block_id=max(int(current_block), 1),
                max_seq_len=int(max_seq_len),
            )
            break

        if state.conflict_clause is not None:
            if not state.decision_stack:
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    raise RuntimeError("expected DONE to terminate at root conflict")
                continue

            clause_id = int(state.conflict_clause)
            if clause_id < 0:
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    raise RuntimeError(
                        "expected DONE to terminate at sentinel conflict"
                    )
                continue

            current_block += 1
            block_start = int(len(tokens))
            sorted_vars = _sorted_unassigned_vars(
                state,
                occurrence,
                state_sort=str(state_sort),
            )
            conflict_chunk = _build_enriched_state_tokens(
                env=env,
                state=state,
                sorted_vars=sorted_vars,
                tokenizer=tokenizer,
            )
            conflict_chunk.extend(
                [
                    int(tokenizer.CONFLICT),
                    int(tokenizer.clause_token(int(clause_id))),
                    int(tokenizer.BACKJUMP),
                    int(
                        tokenizer.level_token(
                            int(max(len(state.decision_stack) - 1, 0))
                        )
                    ),
                ]
            )
            if not _append_tokens(
                tokens,
                block_ids,
                conflict_chunk,
                block_id=int(current_block),
                max_seq_len=int(max_seq_len),
            ):
                logger.info(
                    "trace budget hit on conflict block len=%d block_start=%d",
                    int(len(conflict_chunk)),
                    int(block_start),
                )
                break

            bt_res = env.step(SatAction.backtrack())
            if not bool(bt_res.info.get("valid", True)):
                raise RuntimeError(
                    f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                )
            continue

        if env._all_satisfied(state):
            done_res = env.step(SatAction.done())
            if not bool(done_res.done):
                raise RuntimeError("expected DONE to terminate solved SAT state")
            continue

        open_var = env._open_decision_var(state)
        if open_var is not None and not env._effective_domain(state, int(open_var)):
            bt_res = env.step(SatAction.backtrack())
            if not bool(bt_res.info.get("valid", True)):
                raise RuntimeError(
                    f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                )
            continue

        if state.selected_var is not None and not env._effective_domain(
            state, int(state.selected_var)
        ):
            bt_res = env.step(SatAction.backtrack())
            if not bool(bt_res.info.get("valid", True)):
                raise RuntimeError(
                    f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                )
            continue

        selectable_vars = _selectable_vars_from_state(env, state)
        if not selectable_vars:
            raise RuntimeError("no selectable SAT variable at decision point")

        selected_var = _choose_var_with_random_tie(
            state=state,
            selectable_vars=selectable_vars,
            occurrence=occurrence,
            rng=rng,
        )
        selected_value = _random_signed_value(
            env=env,
            state=state,
            selected_var=int(selected_var),
            rng=rng,
        )

        current_block += 1
        block_start = int(len(tokens))
        sorted_vars = _sorted_unassigned_vars(
            state,
            occurrence,
            state_sort=str(state_sort),
        )
        state_chunk = _build_enriched_state_tokens(
            env=env,
            state=state,
            sorted_vars=sorted_vars,
            tokenizer=tokenizer,
        )
        if not _append_tokens(
            tokens,
            block_ids,
            state_chunk,
            block_id=int(current_block),
            max_seq_len=int(max_seq_len),
        ):
            logger.info(
                "trace budget hit on decision state block len=%d block_start=%d",
                int(len(state_chunk)),
                int(block_start),
            )
            break

        decision_pos = int(len(tokens) - 1)
        full_domains = [
            sorted(int(val) for val in env._effective_domain(state, int(var_id)))
            if int(state.assignment[int(var_id)]) == 0
            else [int(state.assignment[int(var_id)])]
            for var_id in range(int(state.num_vars))
        ]
        action_candidates = _action_candidates_from_state(
            env=env,
            state=state,
            selectable_vars=selectable_vars,
        )
        canonical_state = _canonical_state_key(env, state)
        tried_alternatives = [
            (int(var_id), int(value)) for var_id, value in canonical_state[4]
        ]
        decision_points.append(
            DecisionPoint(
                position=int(decision_pos),
                block_id=int(current_block),
                block_start=int(block_start),
                assignment=[int(x) for x in state.assignment.tolist()],
                domains=full_domains,
                selectable_vars=[int(x) for x in selectable_vars],
                action_candidates=[
                    (int(var_id), int(value)) for var_id, value in action_candidates
                ],
                canonical_state=canonical_state,
                conflict_status=bool(canonical_state[2]),
                decision_level=int(canonical_state[3]),
                tried_alternatives=tried_alternatives,
            )
        )

        value_tok = int(_value_token_from_signed(int(selected_value), tokenizer))
        decision_chunk = [
            int(tokenizer.var_token(int(selected_var))),
            int(value_tok),
            int(tokenizer.OK),
            int(tokenizer.var_token(int(selected_var))),
            int(value_tok),
        ]
        if not _append_tokens(
            tokens,
            block_ids,
            decision_chunk,
            block_id=int(current_block),
            max_seq_len=int(max_seq_len),
        ):
            logger.info(
                "trace budget hit on decision suffix len=%d block=%d",
                int(len(decision_chunk)),
                int(current_block),
            )
            break

        if state.selected_var is None:
            select_res = env.step(SatAction.select_var(int(selected_var)))
            if not bool(select_res.info.get("valid", True)):
                raise RuntimeError(
                    f"invalid_select:{select_res.info.get('reason', 'unknown')}"
                )
        assign_res = env.step(
            SatAction.assign_value(int(_assign_target_from_signed(int(selected_value))))
        )
        if not bool(assign_res.info.get("valid", True)):
            raise RuntimeError(
                f"invalid_assign:{assign_res.info.get('reason', 'unknown')}"
            )

    if len(tokens) != len(block_ids):
        raise RuntimeError("trace token/block length mismatch")

    return OracleTrace(
        tokens=tokens,
        block_ids=block_ids,
        decision_points=decision_points,
        clause_prefix_len=int(clause_prefix_len),
    )


def _extract_val_metrics(
    checkpoint: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    config = checkpoint.get("config", {})
    history = checkpoint.get("history")

    val_loss: Optional[float] = None
    val_acc: Optional[float] = None

    if checkpoint.get("val_loss") is not None:
        val_loss = float(checkpoint["val_loss"])
    elif config.get("val_loss") is not None:
        val_loss = float(config["val_loss"])

    if checkpoint.get("val_acc") is not None:
        val_acc = float(checkpoint["val_acc"])
    elif checkpoint.get("val_token_acc") is not None:
        val_acc = float(checkpoint["val_token_acc"])
    elif config.get("val_acc") is not None:
        val_acc = float(config["val_acc"])
    elif config.get("val_token_acc") is not None:
        val_acc = float(config["val_token_acc"])

    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            if val_loss is None and last.get("val_loss") is not None:
                val_loss = float(last["val_loss"])
            if val_acc is None:
                if last.get("val_acc") is not None:
                    val_acc = float(last["val_acc"])
                elif last.get("val_token_acc") is not None:
                    val_acc = float(last["val_token_acc"])

    return val_loss, val_acc


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    max_seq_len_fallback: int,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})
    model_type = str(config.get("model_type", "transformer"))
    vocab_size = int(
        config.get("vocab_size", state_dict["token_embedding.weight"].shape[0])
    )
    d_model = int(config.get("d_model", 256))
    n_layers = int(config.get("n_layers", 6))
    n_heads = int(config.get("n_heads", 8))
    n_slots = int(config.get("n_slots", 32))
    hidden_size = int(config.get("hidden_size", d_model))
    n_lstm_layers = int(config.get("n_lstm_layers", n_layers))
    block_mode = str(config.get("block_mode", "continuous"))
    max_seq_len_model = int(config.get("max_seq_len", int(max_seq_len_fallback)))
    dropout = float(config.get("dropout", 0.1))
    mask_mode = str(config.get("mask_mode", "selective_ssa"))

    model: torch.nn.Module
    if model_type == "lstm":
        from universal.lstm_decoder import LSTMDecoder

        model = LSTMDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            hidden_size=int(hidden_size),
            n_lstm_layers=int(n_lstm_layers),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
            block_mode=str(block_mode),
        )
        kind = "LSTMDecoder"
    else:
        model = SSASlotDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
        )
        kind = "SSASlotDecoder"

    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(str(key))
            continue
        filtered[str(key)] = value

    if skipped:
        logger.warning(
            "Skipping %d mismatched keys from %s", len(skipped), checkpoint_path
        )

    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()
    val_loss, val_acc = _extract_val_metrics(checkpoint)

    meta = {
        "kind": kind,
        "model_type": str(model_type),
        "mask_mode": str(mask_mode),
        "config": dict(config),
        "max_seq_len_model": int(max_seq_len_model),
        "n_layers": int(n_layers),
        "n_slots": int(n_slots),
        "vocab_size": int(vocab_size),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    return model, meta


@torch.no_grad()
def _forward_logits(
    model: torch.nn.Module,
    input_ids: Sequence[int],
    block_ids: Sequence[int],
    device: torch.device,
    mask_mode: str,
    vocab_size: int,
) -> torch.Tensor:
    if len(input_ids) != len(block_ids):
        raise RuntimeError("input_ids/block_ids length mismatch")
    _ensure_tokens_within_vocab(
        input_ids,
        vocab_size=int(vocab_size),
        context="forward prefix",
    )
    input_tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    block_tensor = torch.tensor([list(block_ids)], dtype=torch.long, device=device)
    lm_logits, _verify_logits = model(
        input_tensor,
        block_ids=block_tensor,
        mask_mode=str(mask_mode),
    )
    final_logits = lm_logits[0, -1, :]
    if int(final_logits.shape[-1]) < int(vocab_size):
        raise RuntimeError(
            "model returned fewer logits than requested vocab size: "
            f"logits={int(final_logits.shape[-1])} target_vocab={int(vocab_size)}"
        )
    return final_logits[: int(vocab_size)]


def _raise_vocab_bound_error(token: int, vocab_size: int, context: str) -> None:
    raise RuntimeError(
        f"tokenizer emitted token {int(token)} but checkpoint only has vocab {int(vocab_size)} "
        f"({str(context)})"
    )


def _ensure_token_within_vocab(token: int, vocab_size: int, context: str) -> int:
    token = int(token)
    if token < 0 or token >= int(vocab_size):
        _raise_vocab_bound_error(token, int(vocab_size), str(context))
    return int(token)


def _ensure_tokens_within_vocab(
    tokens: Sequence[int],
    *,
    vocab_size: int,
    context: str,
) -> None:
    if not tokens:
        return
    max_token = max(int(tok) for tok in tokens)
    if max_token >= int(vocab_size):
        _raise_vocab_bound_error(max_token, int(vocab_size), str(context))


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-12
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    kl = torch.sum(p * (torch.log(p) - torch.log(q)))
    return float(kl.item())


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1, eps=1e-8)
    return float(sim.item())


def _joint_action_distribution(
    *,
    model: torch.nn.Module,
    prefix_tokens: Sequence[int],
    prefix_block_ids: Sequence[int],
    decision_point: DecisionPoint,
    tokenizer: SATInterleavedTokenizer,
    device: torch.device,
    mask_mode: str,
    max_seq_len: int,
    vocab_size: int,
) -> Dict[str, Any]:
    if not decision_point.action_candidates:
        raise RuntimeError("decision point has no legal action candidates")

    var_logits = _forward_logits(
        model=model,
        input_ids=prefix_tokens,
        block_ids=prefix_block_ids,
        device=device,
        mask_mode=str(mask_mode),
        vocab_size=int(vocab_size),
    )
    var_probs_full = torch.softmax(var_logits, dim=-1)

    allowed_var_tokens = [
        _ensure_token_within_vocab(
            int(tokenizer.var_token(int(var_id))),
            vocab_size=int(vocab_size),
            context=f"decision variable token var={int(var_id)}",
        )
        for var_id in decision_point.selectable_vars
    ]
    allowed_var_mass = torch.sum(
        torch.stack([var_probs_full[int(tok)] for tok in allowed_var_tokens], dim=0)
    )
    allowed_var_mass = torch.clamp(allowed_var_mass, min=1e-12)

    joint_probs_values: List[torch.Tensor] = []
    value_logits_cache: Dict[int, torch.Tensor] = {}

    for var_id in decision_point.selectable_vars:
        var_token = _ensure_token_within_vocab(
            int(tokenizer.var_token(int(var_id))),
            vocab_size=int(vocab_size),
            context=f"value probe variable token var={int(var_id)}",
        )
        var_prob = var_probs_full[int(var_token)] / allowed_var_mass

        value_prefix_tokens = list(prefix_tokens) + [int(var_token)]
        value_prefix_blocks = list(prefix_block_ids) + [int(decision_point.block_id)]
        if len(value_prefix_tokens) > int(max_seq_len):
            raise RuntimeError("value probe exceeds max_seq_len")

        value_logits = _forward_logits(
            model=model,
            input_ids=value_prefix_tokens,
            block_ids=value_prefix_blocks,
            device=device,
            mask_mode=str(mask_mode),
            vocab_size=int(vocab_size),
        )
        value_logits_cache[int(var_id)] = value_logits.detach()
        value_probs_full = torch.softmax(value_logits, dim=-1)

        allowed_values = [
            int(v) for v in decision_point.domains[int(var_id)] if int(v) in {-1, 1}
        ]
        if not allowed_values:
            raise RuntimeError(
                f"selected SAT var has no legal polarities: var={var_id}"
            )

        allowed_value_tokens = [
            _ensure_token_within_vocab(
                int(_value_token_from_signed(int(v), tokenizer)),
                vocab_size=int(vocab_size),
                context=f"value token var={int(var_id)} signed_value={int(v)}",
            )
            for v in allowed_values
        ]
        allowed_value_mass = torch.sum(
            torch.stack(
                [value_probs_full[int(tok)] for tok in allowed_value_tokens],
                dim=0,
            )
        )
        allowed_value_mass = torch.clamp(allowed_value_mass, min=1e-12)

        for signed_value in allowed_values:
            value_tok = _ensure_token_within_vocab(
                int(_value_token_from_signed(int(signed_value), tokenizer)),
                vocab_size=int(vocab_size),
                context=(
                    f"value probability token var={int(var_id)} "
                    f"signed_value={int(signed_value)}"
                ),
            )
            value_prob = value_probs_full[int(value_tok)] / allowed_value_mass
            joint_probs_values.append((var_prob * value_prob).detach())

    joint_probs = torch.stack(joint_probs_values, dim=0).to(dtype=torch.float32)
    joint_probs = joint_probs / torch.clamp(torch.sum(joint_probs), min=1e-12)

    best_idx = int(torch.argmax(joint_probs).item())
    best_action = decision_point.action_candidates[int(best_idx)]

    return {
        "action": (int(best_action[0]), int(best_action[1])),
        "joint_probs": joint_probs.detach(),
        "var_logits": var_logits.detach(),
        "argmax_var_token": int(torch.argmax(var_logits).item()),
        "value_logits_by_var": {
            int(var_id): tensor.detach()
            for var_id, tensor in value_logits_cache.items()
        },
    }


def _compare_prefix_behavior(
    *,
    model: torch.nn.Module,
    prefix_a_tokens: Sequence[int],
    prefix_a_blocks: Sequence[int],
    dp_a: DecisionPoint,
    prefix_b_tokens: Sequence[int],
    prefix_b_blocks: Sequence[int],
    dp_b: DecisionPoint,
    tokenizer: SATInterleavedTokenizer,
    max_seq_len: int,
    device: torch.device,
    mask_mode: str,
    store_full_distributions: bool,
    vocab_size: int,
) -> Dict[str, Any]:
    if list(dp_a.action_candidates) != list(dp_b.action_candidates):
        raise RuntimeError("matched SAT states disagree on legal next actions")

    probe_a = _joint_action_distribution(
        model=model,
        prefix_tokens=prefix_a_tokens,
        prefix_block_ids=prefix_a_blocks,
        decision_point=dp_a,
        tokenizer=tokenizer,
        device=device,
        mask_mode=str(mask_mode),
        max_seq_len=int(max_seq_len),
        vocab_size=int(vocab_size),
    )
    probe_b = _joint_action_distribution(
        model=model,
        prefix_tokens=prefix_b_tokens,
        prefix_block_ids=prefix_b_blocks,
        decision_point=dp_b,
        tokenizer=tokenizer,
        device=device,
        mask_mode=str(mask_mode),
        max_seq_len=int(max_seq_len),
        vocab_size=int(vocab_size),
    )

    p_a = probe_a["joint_probs"]
    p_b = probe_b["joint_probs"]
    kl_ab = _kl_divergence(p_a, p_b)
    kl_ba = _kl_divergence(p_b, p_a)
    kl_sym = float(0.5 * (kl_ab + kl_ba))
    cos = _cosine_similarity(p_a, p_b)

    record: Dict[str, Any] = {
        "action_a": {
            "var": int(probe_a["action"][0]),
            "value": int(probe_a["action"][1]),
        },
        "action_b": {
            "var": int(probe_b["action"][0]),
            "value": int(probe_b["action"][1]),
        },
        "action_agreement": bool(
            int(probe_a["action"][0]) == int(probe_b["action"][0])
            and int(probe_a["action"][1]) == int(probe_b["action"][1])
        ),
        "kl_ab": float(kl_ab),
        "kl_ba": float(kl_ba),
        "kl_symmetric": float(kl_sym),
        "cosine_sim": float(cos),
        "argmax_var_token_a": int(probe_a["argmax_var_token"]),
        "argmax_var_token_b": int(probe_b["argmax_var_token"]),
        "action_candidates": [
            {"var": int(var_id), "value": int(value)}
            for var_id, value in dp_a.action_candidates
        ],
    }

    if store_full_distributions:
        record["action_distribution_a"] = [float(x) for x in p_a.cpu().tolist()]
        record["action_distribution_b"] = [float(x) for x in p_b.cpu().tolist()]

    return record


def _aggregate_behavior(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {
            "action_agreement": 0.0,
            "mean_kl_divergence": 0.0,
            "std_kl_divergence": 0.0,
            "mean_cosine_sim": 0.0,
            "std_cosine_sim": 0.0,
        }

    agreement = [1.0 if bool(e["action_agreement"]) else 0.0 for e in entries]
    kls = [float(e["kl_symmetric"]) for e in entries]
    cosines = [float(e["cosine_sim"]) for e in entries]
    return {
        "action_agreement": float(_safe_mean(agreement)),
        "mean_kl_divergence": float(_safe_mean(kls)),
        "std_kl_divergence": float(_safe_std(kls)),
        "mean_cosine_sim": float(_safe_mean(cosines)),
        "std_cosine_sim": float(_safe_std(cosines)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="History transplant behavioral test for SAT checkpoints"
    )
    parser.add_argument(
        "--ssa_checkpoint",
        type=str,
        default=str(REPO_ROOT / "experiments/sat-n50-enriched-selective_ssa-seed42/best.pt"),
    )
    parser.add_argument(
        "--causal_checkpoint",
        type=str,
        default=str(REPO_ROOT / "experiments/sat-n50-enriched-full_causal-seed42/best.pt"),
    )
    parser.add_argument("--n_instances", type=int, default=100)
    parser.add_argument("--n_traces_per_instance", type=int, default=4)
    parser.add_argument("--num_vars", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--state-sort",
        type=str,
        choices=("vsids", "lexical"),
        default="lexical",
        help="Ordering for unassigned variables in STATE blocks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "experiments/sat-history-transplant"),
    )
    parser.add_argument(
        "--store_full_distributions",
        action="store_true",
        help="If set, stores full joint action distributions per pair.",
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))
    tokenizer = SATInterleavedTokenizer()
    device = torch.device(str(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ssa_checkpoint = Path(args.ssa_checkpoint)
    causal_checkpoint = Path(args.causal_checkpoint)
    _ensure_checkpoint_exists(ssa_checkpoint, "--ssa_checkpoint")
    _ensure_checkpoint_exists(causal_checkpoint, "--causal_checkpoint")

    ssa_model, ssa_meta = _load_checkpoint(
        checkpoint_path=ssa_checkpoint,
        device=device,
        max_seq_len_fallback=int(args.max_seq_len),
    )
    causal_model, causal_meta = _load_checkpoint(
        checkpoint_path=causal_checkpoint,
        device=device,
        max_seq_len_fallback=int(args.max_seq_len),
    )

    if str(ssa_meta["mask_mode"]) != "selective_ssa":
        raise RuntimeError(
            "SSA SAT checkpoint must declare mask_mode='selective_ssa'; "
            f"got {ssa_meta['mask_mode']}"
        )
    if str(causal_meta["mask_mode"]) != "full_causal":
        raise RuntimeError(
            "Causal SAT checkpoint must declare mask_mode='full_causal'; "
            f"got {causal_meta['mask_mode']}"
        )

    ssa_vocab_size = int(ssa_meta["vocab_size"])
    causal_vocab_size = int(causal_meta["vocab_size"])
    target_vocab_size = int(min(ssa_vocab_size, causal_vocab_size))
    if ssa_vocab_size != causal_vocab_size:
        raise RuntimeError(
            "SSA and causal checkpoints use different vocab sizes: "
            f"ssa={ssa_vocab_size} causal={causal_vocab_size}"
        )
    if int(target_vocab_size) > int(tokenizer.VOCAB_SIZE):
        raise RuntimeError(
            "SAT checkpoint vocab exceeds SATInterleavedTokenizer vocab size: "
            f"checkpoint={int(target_vocab_size)} tokenizer={int(tokenizer.VOCAB_SIZE)}"
        )

    effective_max_seq_len = int(
        min(
            int(args.max_seq_len),
            int(ssa_meta["max_seq_len_model"]),
            int(causal_meta["max_seq_len_model"]),
        )
    )
    max_steps = int(args.num_vars * 16 + 32)

    generator = SatGenerator(seed=int(args.seed))

    ssa_entries: List[Dict[str, Any]] = []
    causal_entries: List[Dict[str, Any]] = []

    total_candidate_matches = 0
    skipped_identical_history = 0
    skipped_too_long = 0
    skipped_bad_probe = 0
    instances_with_pairs = 0
    pair_id = 0

    logger.info(
        "sat_history_transplant setup num_vars=%d alpha=%.3f n_instances=%d n_traces_per_instance=%d effective_max_seq_len=%d checkpoint_vocab=%d tokenizer_vocab=%d state_sort=%s",
        int(args.num_vars),
        float(args.alpha),
        int(args.n_instances),
        int(args.n_traces_per_instance),
        int(effective_max_seq_len),
        int(target_vocab_size),
        int(tokenizer.VOCAB_SIZE),
        str(args.state_sort),
    )

    for instance_idx in range(int(args.n_instances)):
        inst = generator.generate_planted(
            num_vars=int(args.num_vars),
            alpha=float(args.alpha),
        )
        clauses = [tuple(int(x) for x in clause) for clause in inst.clauses]
        planted_solution = None
        if inst.planted_solution is not None:
            planted_solution = np.array(
                inst.planted_solution, dtype=np.int64, copy=True
            )

        traces: List[OracleTrace] = []
        for trace_idx in range(int(args.n_traces_per_instance)):
            tie_seed = int(args.seed) + int(instance_idx) * 10_007 + int(trace_idx) * 97
            trace = generate_oracle_trace_with_random_ties(
                clauses=clauses,
                num_vars=int(args.num_vars),
                planted_solution=planted_solution,
                max_seq_len=int(effective_max_seq_len),
                max_steps=int(max_steps),
                tie_seed=int(tie_seed),
                state_sort=str(args.state_sort),
            )
            _ensure_tokens_within_vocab(
                trace.tokens,
                vocab_size=int(target_vocab_size),
                context=(
                    f"generated trace instance={int(instance_idx)} trace={int(trace_idx)}"
                ),
            )
            traces.append(trace)

        state_index: Dict[CanonicalStateKey, List[Tuple[int, int]]] = {}
        for trace_idx, trace in enumerate(traces):
            for dp_idx, dp in enumerate(trace.decision_points):
                state_index.setdefault(dp.canonical_state, []).append(
                    (int(trace_idx), int(dp_idx))
                )

        local_pairs = 0
        for _state_key, matches in state_index.items():
            if len(matches) < 2:
                continue

            for i in range(len(matches) - 1):
                trace_a_idx, dp_a_idx = matches[i]
                for j in range(i + 1, len(matches)):
                    trace_b_idx, dp_b_idx = matches[j]
                    if int(trace_a_idx) == int(trace_b_idx):
                        continue
                    total_candidate_matches += 1

                    trace_a = traces[int(trace_a_idx)]
                    trace_b = traces[int(trace_b_idx)]
                    dp_a = trace_a.decision_points[int(dp_a_idx)]
                    dp_b = trace_b.decision_points[int(dp_b_idx)]

                    prefix_a_tokens = [
                        int(x) for x in trace_a.tokens[: int(dp_a.position) + 1]
                    ]
                    prefix_a_blocks = [
                        int(x) for x in trace_a.block_ids[: int(dp_a.position) + 1]
                    ]
                    prefix_b_tokens = [
                        int(x) for x in trace_b.tokens[: int(dp_b.position) + 1]
                    ]
                    prefix_b_blocks = [
                        int(x) for x in trace_b.block_ids[: int(dp_b.position) + 1]
                    ]

                    if len(prefix_a_tokens) > int(effective_max_seq_len) or len(
                        prefix_b_tokens
                    ) > int(effective_max_seq_len):
                        skipped_too_long += 1
                        continue

                    history_a = prefix_a_tokens[
                        int(trace_a.clause_prefix_len) : int(dp_a.block_start)
                    ]
                    history_b = prefix_b_tokens[
                        int(trace_b.clause_prefix_len) : int(dp_b.block_start)
                    ]
                    if history_a == history_b:
                        skipped_identical_history += 1
                        continue

                    history_diff_tokens = _history_difference_tokens(
                        history_a, history_b
                    )
                    if int(history_diff_tokens) <= 0:
                        skipped_identical_history += 1
                        continue

                    try:
                        ssa_behavior = _compare_prefix_behavior(
                            model=ssa_model,
                            prefix_a_tokens=prefix_a_tokens,
                            prefix_a_blocks=prefix_a_blocks,
                            dp_a=dp_a,
                            prefix_b_tokens=prefix_b_tokens,
                            prefix_b_blocks=prefix_b_blocks,
                            dp_b=dp_b,
                            tokenizer=tokenizer,
                            max_seq_len=int(effective_max_seq_len),
                            device=device,
                            mask_mode=str(ssa_meta["mask_mode"]),
                            store_full_distributions=bool(
                                args.store_full_distributions
                            ),
                            vocab_size=int(target_vocab_size),
                        )
                        causal_behavior = _compare_prefix_behavior(
                            model=causal_model,
                            prefix_a_tokens=prefix_a_tokens,
                            prefix_a_blocks=prefix_a_blocks,
                            dp_a=dp_a,
                            prefix_b_tokens=prefix_b_tokens,
                            prefix_b_blocks=prefix_b_blocks,
                            dp_b=dp_b,
                            tokenizer=tokenizer,
                            max_seq_len=int(effective_max_seq_len),
                            device=device,
                            mask_mode=str(causal_meta["mask_mode"]),
                            store_full_distributions=bool(
                                args.store_full_distributions
                            ),
                            vocab_size=int(target_vocab_size),
                        )
                    except RuntimeError as exc:
                        skipped_bad_probe += 1
                        logger.debug("skip pair runtime_error=%s", str(exc))
                        continue

                    shared = {
                        "pair_id": int(pair_id),
                        "instance_idx": int(instance_idx),
                        "trace_a_idx": int(trace_a_idx),
                        "trace_b_idx": int(trace_b_idx),
                        "decision_a_idx": int(dp_a_idx),
                        "decision_b_idx": int(dp_b_idx),
                        "prefix_a_len": int(len(prefix_a_tokens)),
                        "prefix_b_len": int(len(prefix_b_tokens)),
                        "history_tokens_a": int(len(history_a)),
                        "history_tokens_b": int(len(history_b)),
                        "history_tokens_differ": int(history_diff_tokens),
                        "decision_level": int(dp_a.decision_level),
                        "conflict_status": bool(dp_a.conflict_status),
                        "tried_alternatives": [
                            {"var": int(var_id), "value": int(value)}
                            for var_id, value in dp_a.tried_alternatives
                        ],
                    }
                    ssa_entries.append({**shared, **ssa_behavior})
                    causal_entries.append({**shared, **causal_behavior})

                    pair_id += 1
                    local_pairs += 1

        if local_pairs > 0:
            instances_with_pairs += 1

        if (instance_idx + 1) % 10 == 0:
            logger.info(
                "processed_instances=%d/%d pairs=%d candidates=%d skipped_identical=%d skipped_too_long=%d skipped_bad_probe=%d",
                int(instance_idx + 1),
                int(args.n_instances),
                int(len(ssa_entries)),
                int(total_candidate_matches),
                int(skipped_identical_history),
                int(skipped_too_long),
                int(skipped_bad_probe),
            )
            if ssa_entries:
                last_ssa = ssa_entries[-1]
                last_causal = causal_entries[-1]
                logger.info(
                    "sample_pair id=%d hist_diff=%d level=%d tried=%d ssa(agree=%s kl=%.4f cos=%.4f) causal(agree=%s kl=%.4f cos=%.4f)",
                    int(last_ssa["pair_id"]),
                    int(last_ssa["history_tokens_differ"]),
                    int(last_ssa["decision_level"]),
                    int(len(last_ssa["tried_alternatives"])),
                    str(bool(last_ssa["action_agreement"])),
                    float(last_ssa["kl_symmetric"]),
                    float(last_ssa["cosine_sim"]),
                    str(bool(last_causal["action_agreement"])),
                    float(last_causal["kl_symmetric"]),
                    float(last_causal["cosine_sim"]),
                )

    ssa_summary = _aggregate_behavior(ssa_entries)
    causal_summary = _aggregate_behavior(causal_entries)

    payload = {
        "config": {
            "ssa_checkpoint": str(args.ssa_checkpoint),
            "causal_checkpoint": str(args.causal_checkpoint),
            "n_instances": int(args.n_instances),
            "n_traces_per_instance": int(args.n_traces_per_instance),
            "num_vars": int(args.num_vars),
            "alpha": float(args.alpha),
            "max_seq_len": int(args.max_seq_len),
            "effective_max_seq_len": int(effective_max_seq_len),
            "target_vocab_size": int(target_vocab_size),
            "tokenizer_vocab_size": int(tokenizer.VOCAB_SIZE),
            "device": str(args.device),
            "seed": int(args.seed),
            "state_sort": str(args.state_sort),
            "output_dir": str(output_dir),
            "max_steps": int(max_steps),
            "trace_format": "enriched",
            "store_full_distributions": bool(args.store_full_distributions),
        },
        "n_pairs": int(len(ssa_entries)),
        "pair_generation": {
            "instances_with_pairs": int(instances_with_pairs),
            "candidate_matches": int(total_candidate_matches),
            "n_candidate_matches": int(total_candidate_matches),
            "skipped_identical_history": int(skipped_identical_history),
            "skipped_too_long": int(skipped_too_long),
            "skipped_bad_probe": int(skipped_bad_probe),
            "retained_ratio": float(
                _safe_div(len(ssa_entries), total_candidate_matches)
            ),
        },
        "ssa": {
            **ssa_summary,
            "per_pair": ssa_entries,
        },
        "causal": {
            **causal_summary,
            "per_pair": causal_entries,
        },
    }

    output_path = output_dir / "results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== SAT History Transplant Behavioral Test ===")
    print(f"pairs: {int(len(ssa_entries))}")
    print(
        "| model  | action_agreement | mean_kl_divergence | mean_cosine_sim | std_kl | std_cos |"
    )
    print(
        "|--------|------------------|--------------------|-----------------|--------|---------|"
    )
    print(
        "| SSA    | "
        f"{float(ssa_summary['action_agreement']):.4f}            | "
        f"{float(ssa_summary['mean_kl_divergence']):.6f}           | "
        f"{float(ssa_summary['mean_cosine_sim']):.4f}          | "
        f"{float(ssa_summary['std_kl_divergence']):.6f} | "
        f"{float(ssa_summary['std_cosine_sim']):.4f}  |"
    )
    print(
        "| Causal | "
        f"{float(causal_summary['action_agreement']):.4f}            | "
        f"{float(causal_summary['mean_kl_divergence']):.6f}           | "
        f"{float(causal_summary['mean_cosine_sim']):.4f}          | "
        f"{float(causal_summary['std_kl_divergence']):.6f} | "
        f"{float(causal_summary['std_cosine_sim']):.4f}  |"
    )
    print(f"results_json={str(output_path)}")


if __name__ == "__main__":
    main()
