#!/usr/bin/env python3
"""Generate SAT/UNSAT DPLL backtracking traces for SSA training.

Trace format (simplified, no PROP section):
    [BOS] [CLAUSES ...] [SEARCH]
    [STATE v* (U|T|F) ... SEP] [var] [T/F] [OK] [var] [T/F]
    [STATE v* (U|T|F) ... SEP] [CONFLICT] [clause] [BACKJUMP] [level]
    [SOLVED|FAILED] [EOS]

Each saved record is a dict:
    {
        "sequence": List[int],
        "loss_mask": List[bool],
        "block_ids": List[int],
        "label": "sat" | "unsat",
        "meta": {...}
    }
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.dsl import SatAction, SatActionType
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator
from sat.interleaved_tokenizer import SATInterleavedTokenizer
from sat.oracle import SatOracle


def _as_three_lit_clauses(
    clauses: Sequence[Tuple[int, ...]],
) -> List[Tuple[int, int, int]]:
    normalized: List[Tuple[int, int, int]] = []
    for clause in clauses:
        if len(clause) != 3:
            raise ValueError(f"expected 3-literal clause, got: {clause}")
        normalized.append((int(clause[0]), int(clause[1]), int(clause[2])))
    return normalized


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> random.Random:
    random.seed(int(seed))
    np.random.seed(int(seed))
    return random.Random(int(seed))


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
    state: SatState, occurrence_counts: np.ndarray
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


def _domain_token_for_var(
    env: SatEnv,
    state: SatState,
    var_id: int,
    tokenizer: SATInterleavedTokenizer,
) -> int | None:
    domain = env._effective_domain(state, int(var_id))
    has_true = 1 in domain
    has_false = -1 in domain

    if has_true and has_false:
        return int(tokenizer.UNASSIGNED)
    if has_true:
        return int(tokenizer.TRUE_VAL)
    if has_false:
        return int(tokenizer.FALSE_VAL)
    return None


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


def _label_instance(
    clauses: Sequence[Tuple[int, ...]],
    num_vars: int,
    max_steps: int,
) -> str:
    clauses_3 = _as_three_lit_clauses(clauses)
    env = SatEnv(
        clauses=clauses_3,
        num_vars=int(num_vars),
        planted_solution=None,
        mode="strict",
        max_steps=int(max_steps),
    )
    oracle = SatOracle(env)
    oracle.solve()
    final_state = env.get_state()

    if final_state.status == SatEnvStatus.SUCCESS:
        return "sat"
    if (
        final_state.status == SatEnvStatus.FAILURE
        and str(final_state.termination_reason) == "unsat"
    ):
        return "unsat"
    return "unknown"


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
) -> Dict[str, Any]:
    clauses_3 = _as_three_lit_clauses(clauses)
    env = SatEnv(
        clauses=clauses_3,
        num_vars=int(num_vars),
        planted_solution=None
        if planted_solution is None
        else np.array(planted_solution, copy=True),
        mode="soft",
        max_steps=int(max_steps),
    )
    oracle = SatOracle(env)

    env.reset()
    occurrence_counts = _variable_occurrence_counts(clauses, int(num_vars))

    sequence = tokenizer.build_clause_prefix(clauses_3, int(num_vars))
    loss_mask = [False] * int(len(sequence))

    decisions = 0
    conflicts = 0
    backtracks = 0
    forced_errors = 0
    propagations = 0
    terminal: str | None = None

    for _ in range(int(max_steps)):
        state = env.get_state()

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
                # Sentinel should only appear for exhausted search after pop.
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    raise RuntimeError(
                        "expected DONE to terminate at sentinel conflict"
                    )
                continue

            sorted_vars = _sorted_unassigned_vars(state, occurrence_counts)
            backjump_level = int(max(len(state.decision_stack) - 1, 0))

            chunk: List[int] = [int(tokenizer.STATE)]
            for var_id in sorted_vars:
                domain_tok = _domain_token_for_var(env, state, int(var_id), tokenizer)
                if domain_tok is None:
                    continue
                chunk.extend([int(tokenizer.var_token(int(var_id))), int(domain_tok)])
            chunk.append(int(tokenizer.SEP))
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

            chunk = [int(tokenizer.STATE)]
            for var_id in sorted_vars:
                domain_tok = _domain_token_for_var(env, state, int(var_id), tokenizer)
                if domain_tok is None:
                    continue
                chunk.extend([int(tokenizer.var_token(int(var_id))), int(domain_tok)])
            chunk.append(int(tokenizer.SEP))

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

            chunk = [int(tokenizer.STATE)]
            for var_id in sorted_vars:
                domain_tok = _domain_token_for_var(env, state, int(var_id), tokenizer)
                if domain_tok is None:
                    continue
                chunk.extend([int(tokenizer.var_token(int(var_id))), int(domain_tok)])
            chunk.append(int(tokenizer.SEP))

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

    return {
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
    }


def _generate_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = _set_seed(int(args.seed))
    generator = SatGenerator(seed=int(args.seed))
    tokenizer = SATInterleavedTokenizer()

    records: List[Dict[str, Any]] = []

    kept_sat = 0
    kept_unsat = 0
    dropped_len = 0
    dropped_steps = 0
    unsat_attempts = 0
    unsat_rejected_sat = 0
    unsat_rejected_unknown = 0

    trace_lengths: List[int] = []
    decision_counts: List[int] = []
    conflict_counts: List[int] = []
    backtrack_counts: List[int] = []
    forced_error_counts: List[int] = []

    # SAT traces (planted SAT)
    sat_target = int(args.num_sat)
    sat_attempt = 0
    while kept_sat < sat_target:
        sat_attempt += 1
        instance = generator.generate_planted(
            num_vars=int(args.num_vars), alpha=float(args.alpha_sat)
        )

        trace = _collect_single_trace(
            clauses=instance.clauses,
            num_vars=int(instance.num_vars),
            planted_solution=instance.planted_solution,
            p_error=float(args.p_error),
            max_steps=int(args.max_steps),
            max_seq_len=int(args.max_seq_len),
            rng=rng,
            tokenizer=tokenizer,
        )

        if not bool(trace.get("ok", False)):
            if str(trace.get("reason")) == "max_seq_len":
                dropped_len += 1
            else:
                dropped_steps += 1
            continue

        if str(trace["terminal"]) != "sat":
            logger.warning(
                "sat_trace_non_sat terminal=%s attempt=%d",
                str(trace["terminal"]),
                int(sat_attempt),
            )

        records.append(
            {
                "sequence": trace["sequence"],
                "loss_mask": trace["loss_mask"],
                "block_ids": trace["block_ids"],
                "label": "sat",
                "meta": {
                    "terminal": trace["terminal"],
                    "decisions": trace["decisions"],
                    "conflicts": trace["conflicts"],
                    "backtracks": trace["backtracks"],
                    "forced_errors": trace["forced_errors"],
                    "propagations": trace["propagations"],
                    "trace_len": trace["trace_len"],
                },
            }
        )

        kept_sat += 1
        trace_lengths.append(int(trace["trace_len"]))
        decision_counts.append(int(trace["decisions"]))
        conflict_counts.append(int(trace["conflicts"]))
        backtrack_counts.append(int(trace["backtracks"]))
        forced_error_counts.append(int(trace["forced_errors"]))

        if kept_sat <= 3:
            logger.info(
                "sample_sat idx=%d len=%d decisions=%d conflicts=%d backtracks=%d forced_errors=%d terminal=%s",
                int(kept_sat - 1),
                int(trace["trace_len"]),
                int(trace["decisions"]),
                int(trace["conflicts"]),
                int(trace["backtracks"]),
                int(trace["forced_errors"]),
                str(trace["terminal"]),
            )

        if kept_sat % 100 == 0:
            logger.info(
                "progress_sat kept=%d/%d dropped_len=%d dropped_steps=%d mean_len=%.1f mean_conflicts=%.2f mean_forced_errors=%.2f",
                int(kept_sat),
                int(sat_target),
                int(dropped_len),
                int(dropped_steps),
                float(np.mean(trace_lengths)) if trace_lengths else 0.0,
                float(np.mean(conflict_counts)) if conflict_counts else 0.0,
                float(np.mean(forced_error_counts)) if forced_error_counts else 0.0,
            )

    # UNSAT traces (rejection sampled random 3-SAT)
    unsat_target = int(args.num_unsat)
    max_unsat_attempts = max(int(args.max_unsat_attempts), int(unsat_target) * 100)
    while kept_unsat < unsat_target:
        if unsat_attempts >= max_unsat_attempts:
            raise RuntimeError(
                f"failed to collect enough UNSAT traces: kept={kept_unsat}/{unsat_target} attempts={unsat_attempts}"
            )
        unsat_attempts += 1

        instance = generator.generate_random(
            num_vars=int(args.num_vars), alpha=float(args.alpha_unsat)
        )
        label = _label_instance(
            clauses=instance.clauses,
            num_vars=int(instance.num_vars),
            max_steps=int(args.label_max_steps),
        )

        if label != "unsat":
            if label == "sat":
                unsat_rejected_sat += 1
            else:
                unsat_rejected_unknown += 1
            continue

        trace = _collect_single_trace(
            clauses=instance.clauses,
            num_vars=int(instance.num_vars),
            planted_solution=None,
            p_error=float(args.p_error),
            max_steps=int(args.max_steps),
            max_seq_len=int(args.max_seq_len),
            rng=rng,
            tokenizer=tokenizer,
        )
        if not bool(trace.get("ok", False)):
            if str(trace.get("reason")) == "max_seq_len":
                dropped_len += 1
            else:
                dropped_steps += 1
            continue

        records.append(
            {
                "sequence": trace["sequence"],
                "loss_mask": trace["loss_mask"],
                "block_ids": trace["block_ids"],
                "label": "unsat",
                "meta": {
                    "terminal": trace["terminal"],
                    "decisions": trace["decisions"],
                    "conflicts": trace["conflicts"],
                    "backtracks": trace["backtracks"],
                    "forced_errors": trace["forced_errors"],
                    "propagations": trace["propagations"],
                    "trace_len": trace["trace_len"],
                },
            }
        )

        kept_unsat += 1
        trace_lengths.append(int(trace["trace_len"]))
        decision_counts.append(int(trace["decisions"]))
        conflict_counts.append(int(trace["conflicts"]))
        backtrack_counts.append(int(trace["backtracks"]))
        forced_error_counts.append(int(trace["forced_errors"]))

        if kept_unsat <= 3:
            logger.info(
                "sample_unsat idx=%d len=%d decisions=%d conflicts=%d backtracks=%d forced_errors=%d terminal=%s",
                int(kept_unsat - 1),
                int(trace["trace_len"]),
                int(trace["decisions"]),
                int(trace["conflicts"]),
                int(trace["backtracks"]),
                int(trace["forced_errors"]),
                str(trace["terminal"]),
            )

        if kept_unsat % 50 == 0:
            acceptance = float(kept_unsat) / max(float(unsat_attempts), 1.0)
            logger.info(
                "progress_unsat kept=%d/%d attempts=%d acceptance=%.3f rejected_sat=%d rejected_unknown=%d",
                int(kept_unsat),
                int(unsat_target),
                int(unsat_attempts),
                float(acceptance),
                int(unsat_rejected_sat),
                int(unsat_rejected_unknown),
            )

    logger.info(
        "summary total=%d sat=%d unsat=%d mean_len=%.1f max_len=%d mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f mean_forced_errors=%.2f dropped_len=%d dropped_steps=%d unsat_attempts=%d unsat_rejected_sat=%d unsat_rejected_unknown=%d",
        int(len(records)),
        int(kept_sat),
        int(kept_unsat),
        float(np.mean(trace_lengths)) if trace_lengths else 0.0,
        int(np.max(trace_lengths)) if trace_lengths else 0,
        float(np.mean(decision_counts)) if decision_counts else 0.0,
        float(np.mean(conflict_counts)) if conflict_counts else 0.0,
        float(np.mean(backtrack_counts)) if backtrack_counts else 0.0,
        float(np.mean(forced_error_counts)) if forced_error_counts else 0.0,
        int(dropped_len),
        int(dropped_steps),
        int(unsat_attempts),
        int(unsat_rejected_sat),
        int(unsat_rejected_unknown),
    )

    return records


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate simplified SAT/UNSAT backtracking traces for SSA training"
    )
    parser.add_argument("--num-sat", type=int, default=3000)
    parser.add_argument("--num-unsat", type=int, default=3000)
    parser.add_argument("--num-vars", type=int, default=20)
    parser.add_argument("--alpha-sat", type=float, default=3.5)
    parser.add_argument("--alpha-unsat", type=float, default=5.0)
    parser.add_argument("--p-error", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--label-max-steps", type=int, default=5000)
    parser.add_argument("--max-unsat-attempts", type=int, default=200000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=str,
        default="experiments/sat-traces/traces.pkl",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if int(args.num_sat) < 0 or int(args.num_unsat) < 0:
        raise ValueError("num_sat and num_unsat must be non-negative")
    if int(args.num_sat) == 0 and int(args.num_unsat) == 0:
        raise ValueError("at least one of num_sat/num_unsat must be > 0")
    if int(args.num_vars) < 3:
        raise ValueError("num_vars must be >= 3")
    if float(args.alpha_sat) <= 0.0 or float(args.alpha_unsat) <= 0.0:
        raise ValueError("alpha values must be > 0")
    if not (0.0 <= float(args.p_error) <= 1.0):
        raise ValueError("p_error must be in [0, 1]")

    records = _generate_dataset(args)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(records, f)

    logger.info(
        "saved traces path=%s count=%d",
        str(output_path),
        int(len(records)),
    )


if __name__ == "__main__":
    main()
