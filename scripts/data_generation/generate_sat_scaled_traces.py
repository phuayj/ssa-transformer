#!/usr/bin/env python3
"""Generate SAT DPLL traces at larger scale (up to n=200).

This script keeps the existing SatEnv + SatOracle rollout logic for trace generation,
and uses PySAT Glucose4 to verify satisfiability before collecting each trace.

Trace format options:
  - enriched:             STATE includes domain annotations (v U/T/F)
  - enriched_diff:        STATE includes domain annotations with state-diff markers (v U/T/F/NT/NF)
  - minimal:              STATE includes only variable IDs (v)
  - residual_cnf:         STATE includes all residual CNF clauses under current assignment
  - residual_cnf_compact: STATE includes the 50 shortest residual CNF clauses
  - annotated:            STATE includes clause annotations (CONFLICT + UNIT + BIN)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from multiprocessing import Pool, cpu_count
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pysat.solvers import Glucose4  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "pysat is required for generate_sat_scaled_traces.py. Install python-sat."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.dsl import SatAction, SatActionType
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator
from sat.interleaved_tokenizer import SATInterleavedTokenizer, serialize_annotated_state
from sat.oracle import SatOracle
from sat.solvability_oracle import SolvabilityOracle


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> random.Random:
    random.seed(int(seed))
    np.random.seed(int(seed))
    return random.Random(int(seed))


def _glucose_is_sat(clauses: Sequence[Tuple[int, ...]]) -> bool:
    solver = Glucose4(bootstrap_with=[list(clause) for clause in clauses])
    try:
        return bool(solver.solve())
    finally:
        solver.delete()


def _check_branch_extendability(
    clauses: List[Tuple[int, ...]],
    assignment: np.ndarray,
    var_id: int,
    num_vars: int,
    timeout: float = 0.1,
) -> Tuple[bool, bool]:
    """Check if assignment + var=True and assignment + var=False are extendable.

    Returns:
        (is_extendable_true, is_extendable_false)
    """
    oracle = SolvabilityOracle(clauses, time_limit_sec=timeout)

    partial: Dict[int, bool] = {}
    for v in range(int(num_vars)):
        val = int(assignment[int(v)])
        if val == 1:
            partial[int(v)] = True
        elif val == -1:
            partial[int(v)] = False

    partial_t = dict(partial)
    partial_t[int(var_id)] = True
    ext_t = oracle.is_extendable(partial_t)

    partial_f = dict(partial)
    partial_f[int(var_id)] = False
    ext_f = oracle.is_extendable(partial_f)

    oracle.close()
    return (bool(ext_t), bool(ext_f))


def _variable_occurrence_counts(
    clauses: Sequence[Tuple[int, ...]],
    num_vars: int,
) -> np.ndarray:
    counts = np.zeros((int(num_vars),), dtype=np.int64)
    for clause in clauses:
        for lit in clause:
            var = int(abs(int(lit)) - 1)
            if var < 0 or var >= int(num_vars):
                raise ValueError(f"literal out of range: {lit}")
            counts[var] += 1
    return counts


def _sorted_unassigned_vars(
    state: SatState,
    occurrence_counts: np.ndarray,
) -> List[int]:
    unassigned = [
        int(var_id)
        for var_id in range(int(state.num_vars))
        if int(state.assignment[int(var_id)]) == 0
    ]
    return sorted(
        unassigned,
        key=lambda var_id: (
            -float(state.activity[int(var_id)]),
            -int(occurrence_counts[int(var_id)]),
            int(var_id),
        ),
    )


def _compute_domain_status(
    env: SatEnv,
    state: SatState,
    num_vars: int,
    tokenizer: SATInterleavedTokenizer,
) -> np.ndarray:
    """Compute domain status for all variables: U=0, T=1, F=-1, empty/assigned=2."""
    _ = tokenizer
    status = np.zeros((int(num_vars),), dtype=np.int8)
    for var_id in range(int(num_vars)):
        if int(state.assignment[int(var_id)]) != 0:
            # Assigned variables are not emitted in state blocks.
            status[int(var_id)] = 2
            continue

        domain = env._effective_domain(state, int(var_id))
        has_true = 1 in domain
        has_false = -1 in domain
        if has_true and has_false:
            status[int(var_id)] = 0
        elif has_true:
            status[int(var_id)] = 1
        elif has_false:
            status[int(var_id)] = -1
        else:
            status[int(var_id)] = 2
    return status


def _domain_token_for_var(
    env: SatEnv,
    state: SatState,
    var_id: int,
    tokenizer: SATInterleavedTokenizer,
    prev_domain_status: Optional[np.ndarray] = None,
) -> int | None:
    domain = env._effective_domain(state, int(var_id))
    has_true = 1 in domain
    has_false = -1 in domain

    if has_true and has_false:
        return int(tokenizer.UNASSIGNED)

    is_newly_forced = False
    if prev_domain_status is not None:
        prev_status = int(prev_domain_status[int(var_id)])
        if prev_status == 0:
            is_newly_forced = True

    if has_true:
        return int(tokenizer.NEWLY_TRUE) if is_newly_forced else int(tokenizer.TRUE_VAL)
    if has_false:
        return (
            int(tokenizer.NEWLY_FALSE) if is_newly_forced else int(tokenizer.FALSE_VAL)
        )
    return None


def _domain_token_for_var_with_diff(
    env: SatEnv,
    state: SatState,
    var_id: int,
    tokenizer: SATInterleavedTokenizer,
    prev_domain_status: Optional[np.ndarray] = None,
) -> int | None:
    """Domain token with state-diff markers for newly forced variables."""
    return _domain_token_for_var(
        env=env,
        state=state,
        var_id=int(var_id),
        tokenizer=tokenizer,
        prev_domain_status=prev_domain_status,
    )


def _build_residual_cnf_tokens(
    *,
    clauses: Sequence[Tuple[int, ...]],
    assignment: np.ndarray,
    tokenizer: SATInterleavedTokenizer,
    max_residual_clauses: Optional[int] = None,
) -> List[int]:
    residual_clauses: List[List[int]] = []

    for clause in clauses:
        residual_clause: List[int] = []
        clause_satisfied = False
        for lit in clause:
            lit_int = int(lit)
            var_id = int(SATInterleavedTokenizer._lit_var(lit_int))
            sign = int(SATInterleavedTokenizer._lit_sign(lit_int))
            value = int(assignment[var_id])
            if value * sign > 0:
                clause_satisfied = True
                break
            if value == 0:
                if sign > 0:
                    residual_clause.append(int(tokenizer.pos_lit_token(var_id)))
                else:
                    residual_clause.append(int(tokenizer.neg_lit_token(var_id)))

        if clause_satisfied:
            continue
        if not residual_clause:
            return [int(tokenizer.CONFLICT)]
        residual_clauses.append(residual_clause)

    if not residual_clauses:
        return [int(tokenizer.SAT_OK)]

    if max_residual_clauses is not None:
        residual_clauses = sorted(
            residual_clauses,
            key=lambda residual_clause: len(residual_clause),
        )[: int(max_residual_clauses)]

    tokens: List[int] = []
    for clause_index, residual_clause in enumerate(residual_clauses):
        if clause_index > 0:
            tokens.append(int(tokenizer.COLON))
        tokens.extend(residual_clause)
    return tokens


def _build_state_chunk(
    *,
    env: SatEnv,
    state: SatState,
    sorted_vars: Sequence[int],
    tokenizer: SATInterleavedTokenizer,
    trace_format: str,
    prev_domain_status: Optional[np.ndarray] = None,
    clauses: Optional[Sequence[Tuple[int, ...]]] = None,
    max_residual_clauses: Optional[int] = None,
    annotated: bool = False,
) -> List[int]:
    if bool(annotated):
        if clauses is None:
            raise ValueError("annotated state serialization requires clauses")
        state_tokens, _ = serialize_annotated_state(
            clauses=[tuple(int(x) for x in clause) for clause in clauses],
            assignment=state.assignment,
            num_vars=int(state.num_vars),
            tokenizer=tokenizer,
        )
        return [int(tok) for tok in state_tokens]

    chunk: List[int] = [int(tokenizer.STATE)]
    if trace_format == "enriched":
        for var_id in sorted_vars:
            domain_tok = _domain_token_for_var(
                env,
                state,
                int(var_id),
                tokenizer,
                prev_domain_status=None,
            )
            if domain_tok is None:
                continue
            chunk.extend([int(tokenizer.var_token(int(var_id))), int(domain_tok)])
    elif trace_format == "enriched_diff":
        for var_id in sorted_vars:
            domain_tok = _domain_token_for_var_with_diff(
                env=env,
                state=state,
                var_id=int(var_id),
                tokenizer=tokenizer,
                prev_domain_status=prev_domain_status,
            )
            if domain_tok is None:
                continue
            chunk.extend([int(tokenizer.var_token(int(var_id))), int(domain_tok)])
    elif trace_format == "minimal":
        for var_id in sorted_vars:
            chunk.append(int(tokenizer.var_token(int(var_id))))
    elif trace_format == "residual_cnf":
        if clauses is None:
            raise ValueError("residual_cnf state serialization requires clauses")
        chunk.extend(
            _build_residual_cnf_tokens(
                clauses=clauses,
                assignment=state.assignment,
                tokenizer=tokenizer,
            )
        )
    elif trace_format == "residual_cnf_compact":
        if clauses is None:
            raise ValueError(
                "residual_cnf_compact state serialization requires clauses"
            )
        chunk.extend(
            _build_residual_cnf_tokens(
                clauses=clauses,
                assignment=state.assignment,
                tokenizer=tokenizer,
                max_residual_clauses=max_residual_clauses,
            )
        )
    else:
        raise ValueError(f"unsupported trace_format: {trace_format}")

    chunk.append(int(tokenizer.SEP))
    return chunk


def _append_chunk(
    sequence: List[int],
    loss_mask: List[bool],
    chunk: Sequence[int],
    true_positions: Sequence[int],
    *,
    max_seq_len: int,
    vocab_size: int,
) -> bool:
    chunk_tokens = [int(tok) for tok in chunk]
    if len(sequence) + len(chunk_tokens) > int(max_seq_len):
        return False

    for tok in chunk_tokens:
        if tok < 0 or tok >= int(vocab_size):
            raise ValueError(f"token out of range [0,{vocab_size}): {tok}")

    true_set = {int(pos) for pos in true_positions}
    sequence.extend(chunk_tokens)
    loss_mask.extend(bool(i in true_set) for i in range(len(chunk_tokens)))
    return True


def _build_block_ids(sequence: Sequence[int], state_token: int) -> List[int]:
    block_ids: List[int] = []
    current_block = 0
    for tok in sequence:
        if int(tok) == int(state_token):
            current_block += 1
        block_ids.append(int(current_block))
    return block_ids


def _collect_single_trace(
    *,
    clauses: Sequence[Tuple[int, ...]],
    num_vars: int,
    planted_solution: np.ndarray | None,
    p_error: float,
    max_steps: int,
    max_seq_len: int,
    rng: random.Random,
    tokenizer: SATInterleavedTokenizer,
    trace_format: str,
    max_residual_clauses: Optional[int] = None,
    annotated: bool = False,
    compute_verify_labels: bool = False,
    verify_timeout: float = 0.1,
) -> Dict[str, Any]:
    clauses_list = [tuple(int(x) for x in clause) for clause in clauses]
    env = SatEnv(
        clauses=clauses_list,
        num_vars=int(num_vars),
        planted_solution=None
        if planted_solution is None
        else np.array(planted_solution, copy=True),
        mode="soft",
        max_steps=int(max_steps),
    )
    oracle = SatOracle(env)

    env.reset()
    occurrence_counts = _variable_occurrence_counts(clauses_list, int(num_vars))

    sequence = tokenizer.build_clause_prefix(clauses_list, int(num_vars))
    loss_mask = [False] * int(len(sequence))

    decisions = 0
    conflicts = 0
    backtracks = 0
    forced_errors = 0
    propagations = 0
    max_backtrack_depth = 0
    terminal: str | None = None
    prev_domain_status: Optional[np.ndarray] = None
    verify_labels: List[Tuple[int, bool, bool]] = []

    for _ in range(int(max_steps)):
        state = env.get_state()
        max_backtrack_depth = max(max_backtrack_depth, int(len(state.decision_stack)))

        if state.status == SatEnvStatus.SUCCESS:
            if not _append_chunk(
                sequence,
                loss_mask,
                [int(tokenizer.SOLVED)],
                [0],
                max_seq_len=int(max_seq_len),
                vocab_size=int(tokenizer.VOCAB_SIZE),
            ):
                return {"ok": False, "reason": "max_seq_len"}
            terminal = "sat"
            break

        if state.status == SatEnvStatus.FAILURE:
            if not _append_chunk(
                sequence,
                loss_mask,
                [int(tokenizer.FAILED)],
                [0],
                max_seq_len=int(max_seq_len),
                vocab_size=int(tokenizer.VOCAB_SIZE),
            ):
                return {"ok": False, "reason": "max_seq_len"}
            terminal = "unsat"
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

            sorted_vars = _sorted_unassigned_vars(state, occurrence_counts)
            backjump_level = int(max(len(state.decision_stack) - 1, 0))
            chunk = _build_state_chunk(
                env=env,
                state=state,
                sorted_vars=sorted_vars,
                tokenizer=tokenizer,
                trace_format=str(trace_format),
                prev_domain_status=prev_domain_status,
                clauses=clauses_list,
                max_residual_clauses=max_residual_clauses,
                annotated=bool(annotated),
            )
            action_start = int(len(chunk))
            chunk.extend(
                [
                    int(tokenizer.CONFLICT),
                    int(tokenizer.clause_token(clause_id)),
                    int(tokenizer.BACKJUMP),
                    int(tokenizer.level_token(backjump_level)),
                ]
            )

            if not _append_chunk(
                sequence,
                loss_mask,
                chunk,
                [action_start, action_start + 1, action_start + 2, action_start + 3],
                max_seq_len=int(max_seq_len),
                vocab_size=int(tokenizer.VOCAB_SIZE),
            ):
                return {"ok": False, "reason": "max_seq_len"}

            prev_domain_status = _compute_domain_status(
                env,
                state,
                int(num_vars),
                tokenizer,
            )
            conflicts += 1
            backtracks += 1
            env.step(SatAction.backtrack())
            continue

        if state.propagation_pending:
            propagations += 1
            env.step(SatAction.propagate())
            continue

        if env._all_satisfied(state):
            env.step(SatAction.done())
            continue

        action = oracle.get_action(state)

        if action.type == SatActionType.SELECT_VAR:
            if action.target is None:
                raise RuntimeError("oracle SELECT_VAR missing target")
            selected_var = int(action.target)

            res = env.step(SatAction.select_var(selected_var))
            if bool(res.done):
                continue

            state_after_select = env.get_state()
            assign_action = oracle.get_action(state_after_select)
            if assign_action.type != SatActionType.ASSIGN_VALUE:
                env.step(assign_action)
                continue

            if assign_action.target is None:
                raise RuntimeError("oracle ASSIGN_VALUE missing target")

            oracle_value = 1 if int(assign_action.target) == 1 else -1
            domain = env._effective_domain(state_after_select, selected_var)
            chosen_value = int(oracle_value)
            if len(domain) > 1 and rng.random() < float(p_error):
                alternate = -int(oracle_value)
                if int(alternate) in domain:
                    chosen_value = int(alternate)
                    forced_errors += 1

            sorted_vars = _sorted_unassigned_vars(state, occurrence_counts)
            chunk = _build_state_chunk(
                env=env,
                state=state,
                sorted_vars=sorted_vars,
                tokenizer=tokenizer,
                trace_format=str(trace_format),
                prev_domain_status=prev_domain_status,
                clauses=clauses_list,
                max_residual_clauses=max_residual_clauses,
                annotated=bool(annotated),
            )

            action_start = int(len(chunk))
            value_tok = int(
                tokenizer.TRUE_VAL if int(chosen_value) == 1 else tokenizer.FALSE_VAL
            )
            chunk.extend(
                [
                    int(tokenizer.var_token(selected_var)),
                    value_tok,
                    int(tokenizer.OK),
                    int(tokenizer.var_token(selected_var)),
                    value_tok,
                ]
            )

            if not _append_chunk(
                sequence,
                loss_mask,
                chunk,
                [action_start, action_start + 1, action_start + 2],
                max_seq_len=int(max_seq_len),
                vocab_size=int(tokenizer.VOCAB_SIZE),
            ):
                return {"ok": False, "reason": "max_seq_len"}

            if bool(compute_verify_labels):
                ext_t, ext_f = _check_branch_extendability(
                    clauses=clauses_list,
                    assignment=state.assignment,
                    var_id=int(selected_var),
                    num_vars=int(num_vars),
                    timeout=float(verify_timeout),
                )
                verify_labels.append((int(selected_var), bool(ext_t), bool(ext_f)))

            prev_domain_status = _compute_domain_status(
                env,
                state,
                int(num_vars),
                tokenizer,
            )
            decisions += 1
            env.step(SatAction.assign_value(1 if int(chosen_value) == 1 else 0))
            continue

        if action.type == SatActionType.ASSIGN_VALUE:
            if state.selected_var is None:
                raise RuntimeError("ASSIGN_VALUE with no selected_var")
            selected_var = int(state.selected_var)
            if action.target is None:
                raise RuntimeError("oracle ASSIGN_VALUE missing target")

            oracle_value = 1 if int(action.target) == 1 else -1
            domain = env._effective_domain(state, selected_var)
            chosen_value = int(oracle_value)
            if len(domain) > 1 and rng.random() < float(p_error):
                alternate = -int(oracle_value)
                if int(alternate) in domain:
                    chosen_value = int(alternate)
                    forced_errors += 1

            sorted_vars = _sorted_unassigned_vars(state, occurrence_counts)
            chunk = _build_state_chunk(
                env=env,
                state=state,
                sorted_vars=sorted_vars,
                tokenizer=tokenizer,
                trace_format=str(trace_format),
                prev_domain_status=prev_domain_status,
                clauses=clauses_list,
                max_residual_clauses=max_residual_clauses,
                annotated=bool(annotated),
            )

            action_start = int(len(chunk))
            value_tok = int(
                tokenizer.TRUE_VAL if int(chosen_value) == 1 else tokenizer.FALSE_VAL
            )
            chunk.extend(
                [
                    int(tokenizer.var_token(selected_var)),
                    value_tok,
                    int(tokenizer.OK),
                    int(tokenizer.var_token(selected_var)),
                    value_tok,
                ]
            )

            if not _append_chunk(
                sequence,
                loss_mask,
                chunk,
                [action_start, action_start + 1, action_start + 2],
                max_seq_len=int(max_seq_len),
                vocab_size=int(tokenizer.VOCAB_SIZE),
            ):
                return {"ok": False, "reason": "max_seq_len"}

            if bool(compute_verify_labels):
                ext_t, ext_f = _check_branch_extendability(
                    clauses=clauses_list,
                    assignment=state.assignment,
                    var_id=int(selected_var),
                    num_vars=int(num_vars),
                    timeout=float(verify_timeout),
                )
                verify_labels.append((int(selected_var), bool(ext_t), bool(ext_f)))

            prev_domain_status = _compute_domain_status(
                env,
                state,
                int(num_vars),
                tokenizer,
            )
            decisions += 1
            env.step(SatAction.assign_value(1 if int(chosen_value) == 1 else 0))
            continue

        if action.type == SatActionType.BACKTRACK:
            backtracks += 1
            env.step(action)
            continue

        if action.type == SatActionType.PROPAGATE:
            propagations += 1
            env.step(action)
            continue

        if action.type == SatActionType.DONE:
            env.step(action)
            continue

        raise RuntimeError(f"unhandled action type: {action.type}")

    if terminal is None:
        final_state = env.get_state()
        if final_state.status == SatEnvStatus.SUCCESS:
            terminal = "sat"
        elif final_state.status == SatEnvStatus.FAILURE:
            terminal = "unsat"
        else:
            return {"ok": False, "reason": "max_steps"}

    if not _append_chunk(
        sequence,
        loss_mask,
        [int(tokenizer.EOS)],
        [],
        max_seq_len=int(max_seq_len),
        vocab_size=int(tokenizer.VOCAB_SIZE),
    ):
        return {"ok": False, "reason": "max_seq_len"}

    block_ids = _build_block_ids(sequence, state_token=int(tokenizer.STATE))
    if len(sequence) != len(loss_mask) or len(sequence) != len(block_ids):
        raise RuntimeError("sequence/loss_mask/block_ids length mismatch")

    result: Dict[str, Any] = {
        "ok": True,
        "sequence": sequence,
        "loss_mask": loss_mask,
        "block_ids": block_ids,
        "terminal": str(terminal),
        "decisions": int(decisions),
        "conflicts": int(conflicts),
        "backtracks": int(backtracks),
        "forced_errors": int(forced_errors),
        "propagations": int(propagations),
        "trace_len": int(len(sequence)),
        "max_backtrack_depth": int(max_backtrack_depth),
    }
    if bool(compute_verify_labels):
        result["verify_labels"] = verify_labels
    return result


def _worker_fn(
    worker_args: Tuple[
        int,
        Sequence[Tuple[int, ...]],
        int,
        np.ndarray | None,
        float,
        int,
        int,
        str,
        int,
        bool,
        bool,
        float,
        int,
    ],
) -> Dict[str, Any]:
    (
        worker_index,
        clauses,
        num_vars,
        planted_solution,
        p_error,
        max_steps,
        max_seq_len,
        trace_format,
        max_residual_clauses,
        annotated,
        compute_verify_labels,
        verify_timeout,
        global_seed,
    ) = worker_args

    rng = random.Random(int(global_seed) + int(worker_index))
    tokenizer = SATInterleavedTokenizer()
    return _collect_single_trace(
        clauses=clauses,
        num_vars=int(num_vars),
        planted_solution=planted_solution,
        p_error=float(p_error),
        max_steps=int(max_steps),
        max_seq_len=int(max_seq_len),
        rng=rng,
        tokenizer=tokenizer,
        trace_format=str(trace_format),
        max_residual_clauses=int(max_residual_clauses),
        annotated=bool(annotated),
        compute_verify_labels=bool(compute_verify_labels),
        verify_timeout=float(verify_timeout),
    )


def _generate_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    _set_seed(int(args.seed))
    generator = SatGenerator(seed=int(args.seed))

    alpha = 4.26 if bool(args.phase_transition) else float(args.alpha)
    max_steps = max(5000, int(args.num_vars) * 40)
    max_workers = max(1, int(args.workers))
    selected_trace_format = (
        "annotated" if bool(args.annotated) else str(args.trace_format)
    )
    logger.info(
        "trace_generation_config trace_format=%s annotated=%s",
        str(selected_trace_format),
        bool(args.annotated),
    )

    records: List[Dict[str, Any]] = []
    kept = 0
    attempts = 0
    rejected_unsat = 0
    dropped_len = 0
    dropped_steps = 0
    worker_seed_index = 0

    trace_lengths: List[int] = []
    decision_counts: List[int] = []
    conflict_counts: List[int] = []
    backtrack_counts: List[int] = []
    max_depths: List[int] = []
    verify_step_counts: List[int] = []

    target = int(args.num_instances)
    while kept < target:
        remaining = int(target - kept)
        instances: List[Any] = []
        while len(instances) < remaining:
            attempts += 1
            if bool(args.phase_transition):
                instance = generator.generate_random(
                    num_vars=int(args.num_vars), alpha=4.26
                )
            else:
                instance = generator.generate_planted(
                    num_vars=int(args.num_vars), alpha=float(alpha)
                )

            if not _glucose_is_sat(instance.clauses):
                rejected_unsat += 1
                continue
            instances.append(instance)

        worker_args: List[
            Tuple[
                int,
                Sequence[Tuple[int, ...]],
                int,
                np.ndarray | None,
                float,
                int,
                int,
                str,
                int,
                bool,
                bool,
                float,
                int,
            ]
        ] = []
        for instance in instances:
            worker_args.append(
                (
                    int(worker_seed_index),
                    tuple(tuple(int(x) for x in clause) for clause in instance.clauses),
                    int(instance.num_vars),
                    None
                    if instance.planted_solution is None
                    else np.array(instance.planted_solution, copy=True),
                    float(args.p_error),
                    int(max_steps),
                    int(args.max_seq_len),
                    str(args.trace_format),
                    int(args.max_residual_clauses),
                    bool(args.annotated),
                    bool(args.compute_verify_labels),
                    float(args.verify_timeout),
                    int(args.seed),
                )
            )
            worker_seed_index += 1

        with Pool(processes=int(max_workers)) as pool:
            for trace in pool.imap_unordered(_worker_fn, worker_args):
                if not bool(trace.get("ok", False)):
                    if str(trace.get("reason")) == "max_seq_len":
                        dropped_len += 1
                    else:
                        dropped_steps += 1
                    continue

                if str(trace["terminal"]) != "sat":
                    logger.warning(
                        "trace_non_sat terminal=%s attempt=%d",
                        str(trace["terminal"]),
                        int(attempts),
                    )

                record = {
                    "sequence": trace["sequence"],
                    "loss_mask": trace["loss_mask"],
                    "block_ids": trace["block_ids"],
                    "label": "sat",
                    "meta": {
                        "num_vars": int(args.num_vars),
                        "alpha": float(alpha),
                        "trace_format": str(selected_trace_format),
                        "num_decisions": int(trace["decisions"]),
                        "num_conflicts": int(trace["conflicts"]),
                        "num_backtracks": int(trace["backtracks"]),
                        "trace_length": int(trace["trace_len"]),
                        "max_backtrack_depth": int(trace["max_backtrack_depth"]),
                    },
                }
                if bool(args.compute_verify_labels):
                    record["verify_labels"] = trace.get("verify_labels", [])
                records.append(record)

                kept += 1
                trace_lengths.append(int(trace["trace_len"]))
                decision_counts.append(int(trace["decisions"]))
                conflict_counts.append(int(trace["conflicts"]))
                backtrack_counts.append(int(trace["backtracks"]))
                max_depths.append(int(trace["max_backtrack_depth"]))
                if bool(args.compute_verify_labels):
                    verify_step_counts.append(len(trace.get("verify_labels", [])))

                if kept <= 3:
                    if bool(args.compute_verify_labels):
                        logger.info(
                            "sample_trace idx=%d len=%d decisions=%d verify_steps=%d conflicts=%d backtracks=%d depth=%d terminal=%s",
                            int(kept - 1),
                            int(trace["trace_len"]),
                            int(trace["decisions"]),
                            int(len(trace.get("verify_labels", []))),
                            int(trace["conflicts"]),
                            int(trace["backtracks"]),
                            int(trace["max_backtrack_depth"]),
                            str(trace["terminal"]),
                        )
                    else:
                        logger.info(
                            "sample_trace idx=%d len=%d decisions=%d conflicts=%d backtracks=%d depth=%d terminal=%s",
                            int(kept - 1),
                            int(trace["trace_len"]),
                            int(trace["decisions"]),
                            int(trace["conflicts"]),
                            int(trace["backtracks"]),
                            int(trace["max_backtrack_depth"]),
                            str(trace["terminal"]),
                        )

                if kept % 100 == 0:
                    if bool(args.compute_verify_labels):
                        logger.info(
                            "progress kept=%d/%d attempts=%d rejected_unsat=%d dropped_len=%d dropped_steps=%d mean_len=%.1f mean_conflicts=%.2f mean_depth=%.2f mean_verify_steps=%.2f",
                            int(kept),
                            int(target),
                            int(attempts),
                            int(rejected_unsat),
                            int(dropped_len),
                            int(dropped_steps),
                            float(np.mean(trace_lengths)) if trace_lengths else 0.0,
                            float(np.mean(conflict_counts)) if conflict_counts else 0.0,
                            float(np.mean(max_depths)) if max_depths else 0.0,
                            float(np.mean(verify_step_counts))
                            if verify_step_counts
                            else 0.0,
                        )
                    else:
                        logger.info(
                            "progress kept=%d/%d attempts=%d rejected_unsat=%d dropped_len=%d dropped_steps=%d mean_len=%.1f mean_conflicts=%.2f mean_depth=%.2f",
                            int(kept),
                            int(target),
                            int(attempts),
                            int(rejected_unsat),
                            int(dropped_len),
                            int(dropped_steps),
                            float(np.mean(trace_lengths)) if trace_lengths else 0.0,
                            float(np.mean(conflict_counts)) if conflict_counts else 0.0,
                            float(np.mean(max_depths)) if max_depths else 0.0,
                        )

    logger.info(
        "summary total=%d attempts=%d rejected_unsat=%d dropped_len=%d dropped_steps=%d mean_len=%.1f max_len=%d mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f mean_depth=%.2f",
        int(len(records)),
        int(attempts),
        int(rejected_unsat),
        int(dropped_len),
        int(dropped_steps),
        float(np.mean(trace_lengths)) if trace_lengths else 0.0,
        int(np.max(trace_lengths)) if trace_lengths else 0,
        float(np.mean(decision_counts)) if decision_counts else 0.0,
        float(np.mean(conflict_counts)) if conflict_counts else 0.0,
        float(np.mean(backtrack_counts)) if backtrack_counts else 0.0,
        float(np.mean(max_depths)) if max_depths else 0.0,
    )

    return records


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SAT traces for scaled SAT experiments"
    )
    parser.add_argument("--num_vars", "--num-vars", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=4.26)
    parser.add_argument(
        "--num_instances",
        "--num-instances",
        "--num_traces",
        "--num-traces",
        type=int,
        default=5000,
    )
    parser.add_argument("--max_seq_len", "--max-seq-len", type=int, default=8192)
    parser.add_argument(
        "--max_residual_clauses",
        "--max-residual-clauses",
        type=int,
        default=50,
    )
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument(
        "--trace_format",
        "--trace-format",
        type=str,
        choices=(
            "enriched",
            "enriched_diff",
            "minimal",
            "residual_cnf",
            "residual_cnf_compact",
        ),
        default="enriched",
    )
    parser.add_argument(
        "--annotated",
        action="store_true",
        help="Use clause-annotated state blocks (CONFLICT+UNIT+BIN)",
    )
    parser.add_argument(
        "--phase_transition",
        "--phase-transition",
        action="store_true",
        help="Use random 3-SAT near phase transition (alpha fixed to 4.26) and filter SAT",
    )
    parser.add_argument(
        "--compute_verify_labels",
        action="store_true",
        help="Compute per-decision oracle extendability labels using PySAT",
    )
    parser.add_argument(
        "--verify_timeout",
        type=float,
        default=0.1,
        help="PySAT timeout per extendability query (seconds)",
    )
    parser.add_argument("--p_error", "--p-error", type=float, default=0.0)
    parser.add_argument("--output_dir", "--output-dir", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional direct output path for traces.pkl; metadata is written alongside it",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if int(args.num_vars) < 3:
        raise ValueError("num_vars must be >= 3")
    if int(args.num_vars) > int(SATInterleavedTokenizer.MAX_VARS):
        raise ValueError(
            f"num_vars must be <= {SATInterleavedTokenizer.MAX_VARS} for current tokenizer"
        )
    if int(args.num_instances) <= 0:
        raise ValueError("num_instances must be > 0")
    if float(args.alpha) <= 0.0:
        raise ValueError("alpha must be > 0")
    if int(args.max_seq_len) <= 0:
        raise ValueError("max_seq_len must be > 0")
    if not (0.0 <= float(args.p_error) <= 1.0):
        raise ValueError("p_error must be in [0, 1]")
    if float(args.verify_timeout) <= 0.0:
        raise ValueError("verify_timeout must be > 0")
    if args.output_dir is None and args.output is None:
        raise ValueError("must provide either --output_dir/--output-dir or --output")
    if args.output_dir is not None and args.output is not None:
        raise ValueError("provide only one of --output_dir/--output-dir or --output")

    records = _generate_dataset(args)

    if args.output is not None:
        traces_path = Path(args.output)
        output_dir = traces_path.parent
    else:
        output_dir = Path(args.output_dir)
        traces_path = output_dir / "traces.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)

    with traces_path.open("wb") as f:
        pickle.dump(records, f)

    run_metadata = {
        "num_records": int(len(records)),
        "num_vars": int(args.num_vars),
        "alpha": float(4.26 if bool(args.phase_transition) else float(args.alpha)),
        "trace_format": "annotated" if bool(args.annotated) else str(args.trace_format),
        "annotated": bool(args.annotated),
        "phase_transition": bool(args.phase_transition),
        "max_seq_len": int(args.max_seq_len),
        "p_error": float(args.p_error),
        "compute_verify_labels": bool(args.compute_verify_labels),
        "verify_timeout": float(args.verify_timeout),
        "seed": int(args.seed),
    }
    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    logger.info(
        "saved traces path=%s count=%d metadata_path=%s",
        str(traces_path),
        int(len(records)),
        str(metadata_path),
    )


if __name__ == "__main__":
    main()
