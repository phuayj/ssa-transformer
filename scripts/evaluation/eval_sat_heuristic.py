#!/usr/bin/env python3
"""Evaluate rule-based heuristic baselines on planted SAT instances.

This script mirrors the closed-loop search mechanics used by
``eval_sat_autonomous.py`` but removes all tokenizer/model logic. Decisions are
made directly from the environment state using simple rule-based heuristics.

Implemented heuristics:
  - ``vsids_domain``: highest-activity variable, domain-aware polarity
  - ``occurrence_domain``: highest-occurrence variable, domain-aware polarity
  - ``random_domain``: random variable, domain-aware polarity
  - ``pure_random``: random variable and random legal polarity
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pysat.formula import CNF
    from pysat.solvers import Glucose4
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pysat is required for SAT heuristic evaluation (install python-sat)."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.dsl import SatAction, SatActionType
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _occurrence_counts(
    clauses: Sequence[Tuple[int, ...]],
    num_vars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_count = np.zeros(int(num_vars), dtype=np.int64)
    neg_count = np.zeros(int(num_vars), dtype=np.int64)
    for clause in clauses:
        for lit in clause:
            var = abs(int(lit)) - 1
            if int(lit) > 0:
                pos_count[int(var)] += 1
            else:
                neg_count[int(var)] += 1
    return pos_count, neg_count


def _trim_tried_levels(
    tried_values_by_level: Dict[int, set[int]],
    max_level: int,
) -> None:
    stale = [
        int(level) for level in tried_values_by_level if int(level) > int(max_level)
    ]
    for level in stale:
        tried_values_by_level.pop(int(level), None)


def _forced_candidates(
    env: SatEnv,
    state: SatState,
    candidates: Sequence[int],
) -> List[int]:
    forced: List[int] = []
    for var_id in candidates:
        domain = {int(v) for v in env._effective_domain(state, int(var_id))}
        if len(domain) == 1:
            forced.append(int(var_id))
    return forced


def _legalize_value_choice(
    chosen_value: int,
    valid_assign_targets: Sequence[int],
) -> int:
    if int(chosen_value) in {int(v) for v in valid_assign_targets}:
        return int(chosen_value)
    if not valid_assign_targets:
        raise RuntimeError("no legal ASSIGN_VALUE targets available")
    return int(valid_assign_targets[0])


def _generate_instances(
    num_instances: int,
    num_vars: int,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rows, _generation_stats = _generate_instances_with_mode(
        num_instances=int(num_instances),
        num_vars=int(num_vars),
        alpha=float(alpha),
        seed=int(seed),
        phase_transition=False,
    )
    return rows


def _is_satisfiable_random_instance(clauses: Sequence[Tuple[int, ...]]) -> bool:
    cnf = CNF(from_clauses=[[int(lit) for lit in clause] for clause in clauses])
    with Glucose4(bootstrap_with=cnf.clauses) as solver:
        return bool(solver.solve())


def _generate_instances_with_mode(
    num_instances: int,
    num_vars: int,
    alpha: float,
    seed: int,
    phase_transition: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    generator = SatGenerator(seed=int(seed))
    rows: List[Dict[str, Any]] = []

    if not bool(phase_transition):
        for idx in range(int(num_instances)):
            generator_any: Any = generator
            generate_planted_instance = getattr(
                generator_any, "generate_planted_instance", None
            )
            if callable(generate_planted_instance):
                clauses_raw, solution_raw = generate_planted_instance(
                    num_vars=int(num_vars),
                    alpha=float(alpha),
                    seed=int(seed) + int(idx),
                )
                clauses = [tuple(int(x) for x in clause) for clause in clauses_raw]
                planted_solution = np.array(solution_raw, dtype=np.int64, copy=True)
            else:
                inst = generator.generate_planted(
                    num_vars=int(num_vars), alpha=float(alpha)
                )
                clauses = [tuple(int(x) for x in clause) for clause in inst.clauses]
                planted_solution = None
                if inst.planted_solution is not None:
                    planted_solution = np.array(
                        inst.planted_solution, dtype=np.int64, copy=True
                    )

            rows.append(
                {
                    "clauses": clauses,
                    "num_vars": int(num_vars),
                    "planted_solution": planted_solution,
                }
            )

        generation_stats = {
            "phase_transition": False,
            "accepted_instances": int(len(rows)),
            "total_generated": int(len(rows)),
            "sat_ratio": 1.0 if int(len(rows)) > 0 else 0.0,
            "max_generation_attempts": int(len(rows)),
        }
        logger.info(
            "instance_generation mode=planted accepted=%d total=%d sat_ratio=%.3f",
            int(generation_stats["accepted_instances"]),
            int(generation_stats["total_generated"]),
            float(generation_stats["sat_ratio"]),
        )
        return rows, generation_stats

    max_generation_attempts = max(int(num_instances) * 50, int(num_instances))
    total_generated = 0
    sat_hits = 0

    while len(rows) < int(num_instances) and total_generated < int(
        max_generation_attempts
    ):
        inst = generator.generate_random(num_vars=int(num_vars), alpha=float(alpha))
        total_generated += 1
        clauses = [tuple(int(x) for x in clause) for clause in inst.clauses]
        if not bool(_is_satisfiable_random_instance(clauses)):
            continue

        sat_hits += 1
        rows.append(
            {
                "clauses": clauses,
                "num_vars": int(num_vars),
                "planted_solution": None,
            }
        )

    sat_ratio = float(_safe_div(sat_hits, total_generated))
    generation_stats = {
        "phase_transition": True,
        "accepted_instances": int(len(rows)),
        "total_generated": int(total_generated),
        "sat_ratio": float(sat_ratio),
        "max_generation_attempts": int(max_generation_attempts),
    }
    logger.info(
        "instance_generation mode=phase_transition accepted=%d requested=%d total=%d sat_ratio=%.3f cap=%d",
        int(len(rows)),
        int(num_instances),
        int(total_generated),
        float(sat_ratio),
        int(max_generation_attempts),
    )
    if len(rows) < int(num_instances):
        raise RuntimeError(
            "phase-transition SAT generation hit safety cap before collecting enough "
            f"instances: collected={int(len(rows))} requested={int(num_instances)} "
            f"total_generated={int(total_generated)} sat_ratio={float(sat_ratio):.3f} "
            f"cap={int(max_generation_attempts)}"
        )

    return rows, generation_stats


class BaseHeuristic:
    """Base class for SAT search heuristics."""

    name: str

    def select_variable(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selectable_vars: Sequence[int],
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        rng: random.Random,
    ) -> int:
        raise NotImplementedError

    def select_value(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selected_var: int,
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        valid_assign_targets: Sequence[int],
        rng: random.Random,
    ) -> int:
        raise NotImplementedError


class VSIDSDomainHeuristic(BaseHeuristic):
    """Domain-aware VSIDS baseline.

    Variable selection follows the same ordering used for the autonomous model's
    state block: descending activity, then descending total occurrence count,
    then ascending variable index. If any candidate has a singleton effective
    domain, those forced variables are prioritized.

    Value selection uses the effective domain directly: forced true/false when
    singleton, otherwise default to true polarity.
    """

    name = "vsids_domain"

    def select_variable(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selectable_vars: Sequence[int],
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        rng: random.Random,
    ) -> int:
        _ = env, rng
        total_occurrence = pos_count + neg_count
        forced = _forced_candidates(env, state, selectable_vars)
        pool = forced if forced else [int(v) for v in selectable_vars]
        return int(
            min(
                pool,
                key=lambda var_id: (
                    -float(state.activity[int(var_id)]),
                    -int(total_occurrence[int(var_id)]),
                    int(var_id),
                ),
            )
        )

    def select_value(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selected_var: int,
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        valid_assign_targets: Sequence[int],
        rng: random.Random,
    ) -> int:
        _ = pos_count, neg_count, rng
        domain = {int(v) for v in env._effective_domain(state, int(selected_var))}
        if domain == {1}:
            chosen_value = 1
        elif domain == {-1}:
            chosen_value = 0
        else:
            chosen_value = 1
        return int(_legalize_value_choice(chosen_value, valid_assign_targets))


class OccurrenceDomainHeuristic(BaseHeuristic):
    """Occurrence-count baseline with domain-aware polarity selection.

    Variable selection chooses the unassigned variable with the largest total
    literal occurrence count, breaking ties by smallest index. Singleton
    effective-domain variables are prioritized over all others.

    Value selection follows the domain when singleton; otherwise it chooses the
    polarity whose literal sign appears in more clauses.
    """

    name = "occurrence_domain"

    def select_variable(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selectable_vars: Sequence[int],
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        rng: random.Random,
    ) -> int:
        _ = state, rng
        total_occurrence = pos_count + neg_count
        forced = _forced_candidates(env, state, selectable_vars)
        pool = forced if forced else [int(v) for v in selectable_vars]
        return int(
            min(
                pool,
                key=lambda var_id: (
                    -int(total_occurrence[int(var_id)]),
                    int(var_id),
                ),
            )
        )

    def select_value(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selected_var: int,
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        valid_assign_targets: Sequence[int],
        rng: random.Random,
    ) -> int:
        _ = rng
        domain = {int(v) for v in env._effective_domain(state, int(selected_var))}
        if domain == {1}:
            chosen_value = 1
        elif domain == {-1}:
            chosen_value = 0
        else:
            chosen_value = (
                1
                if int(pos_count[int(selected_var)])
                >= int(neg_count[int(selected_var)])
                else 0
            )
        return int(_legalize_value_choice(chosen_value, valid_assign_targets))


class RandomDomainHeuristic(BaseHeuristic):
    """Random variable selection with domain-aware value choice.

    Variable selection is uniform random over unassigned candidates, except that
    singleton effective-domain variables are prioritized when available. Value
    selection follows singleton domains exactly and chooses uniformly between
    true/false when the domain remains unconstrained.
    """

    name = "random_domain"

    def select_variable(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selectable_vars: Sequence[int],
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        rng: random.Random,
    ) -> int:
        _ = state, pos_count, neg_count
        forced = _forced_candidates(env, state, selectable_vars)
        pool = forced if forced else [int(v) for v in selectable_vars]
        return int(rng.choice(pool))

    def select_value(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selected_var: int,
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        valid_assign_targets: Sequence[int],
        rng: random.Random,
    ) -> int:
        _ = pos_count, neg_count
        domain = {int(v) for v in env._effective_domain(state, int(selected_var))}
        if domain == {1}:
            chosen_value = 1
        elif domain == {-1}:
            chosen_value = 0
        else:
            chosen_value = int(rng.choice([0, 1]))
        return int(_legalize_value_choice(chosen_value, valid_assign_targets))


class PureRandomHeuristic(BaseHeuristic):
    """Pure random legal baseline.

    Variable selection is uniform random among currently selectable variables.
    Value selection is uniform random among the currently legal ASSIGN_VALUE
    actions, without consulting the environment's effective-domain annotations.
    """

    name = "pure_random"

    def select_variable(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selectable_vars: Sequence[int],
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        rng: random.Random,
    ) -> int:
        _ = state, env, pos_count, neg_count
        return int(rng.choice([int(v) for v in selectable_vars]))

    def select_value(
        self,
        *,
        state: SatState,
        env: SatEnv,
        selected_var: int,
        pos_count: np.ndarray,
        neg_count: np.ndarray,
        valid_assign_targets: Sequence[int],
        rng: random.Random,
    ) -> int:
        _ = state, env, selected_var, pos_count, neg_count
        return int(rng.choice([int(v) for v in valid_assign_targets]))


HEURISTICS: Dict[str, BaseHeuristic] = {
    VSIDSDomainHeuristic.name: VSIDSDomainHeuristic(),
    OccurrenceDomainHeuristic.name: OccurrenceDomainHeuristic(),
    RandomDomainHeuristic.name: RandomDomainHeuristic(),
    PureRandomHeuristic.name: PureRandomHeuristic(),
}


def _has_frontier_actions(actions: Sequence[Any]) -> Tuple[bool, bool]:
    has_assign = any(action.type == SatActionType.ASSIGN_VALUE for action in actions)
    has_select = any(action.type == SatActionType.SELECT_VAR for action in actions)
    return bool(has_assign), bool(has_select)


def _cascade_exhausted_backtracks(
    *,
    env: SatEnv,
    tried_values_by_level: Dict[int, set[int]],
    stats: Dict[str, Any],
) -> bool:
    """Mechanically backtrack until an assign/select frontier is available."""

    while True:
        cascade_state = env.get_state()
        _trim_tried_levels(
            tried_values_by_level, int(len(cascade_state.decision_stack))
        )

        if cascade_state.status != SatEnvStatus.RUNNING:
            return True

        cascade_actions = env.get_valid_actions()
        has_assign, has_select = _has_frontier_actions(cascade_actions)
        if has_assign or has_select:
            return True

        if not cascade_state.decision_stack:
            done_res = env.step(SatAction.done())
            if not bool(done_res.done):
                stats["termination_reason"] = "failed_done_after_root_conflict"
            else:
                stats["termination_reason"] = "unsat"
            return False

        bt_res = env.step(SatAction.backtrack())
        if not bool(bt_res.info.get("valid", True)):
            stats["termination_reason"] = (
                f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
            )
            return False

        stats["backtracks"] += 1


def solve_instance(
    *,
    clauses: List[Tuple[int, ...]],
    num_vars: int,
    planted_solution: Optional[np.ndarray],
    heuristic: BaseHeuristic,
    budget: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """Run the heuristic search loop for one SAT instance."""

    env = SatEnv(
        clauses=[tuple(int(x) for x in clause) for clause in clauses],
        num_vars=int(num_vars),
        planted_solution=None
        if planted_solution is None
        else np.array(planted_solution, dtype=np.int64, copy=True),
        mode="strict",
        max_steps=int(budget * 8 + 20),
    )
    env.reset()

    pos_count, neg_count = _occurrence_counts(clauses, int(num_vars))
    tried_values_by_level: Dict[int, set[int]] = {}

    stats: Dict[str, Any] = {
        "heuristic": str(heuristic.name),
        "solved": False,
        "steps": 0,
        "decisions": 0,
        "conflicts": 0,
        "backtracks": 0,
        "post_backtrack_decisions": 0,
        "repeat_errors": 0,
        "termination_reason": "max_steps",
    }

    for step in range(int(budget)):
        stats["steps"] = int(step + 1)
        state = env.get_state()

        if state.status != SatEnvStatus.RUNNING:
            stats["solved"] = bool(state.status == SatEnvStatus.SUCCESS)
            stats["termination_reason"] = str(state.termination_reason or "env_done")
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
            stats["termination_reason"] = str(state.termination_reason or "env_done")
            break

        if state.conflict_clause is not None:
            stats["conflicts"] += 1

            if not state.decision_stack:
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_root_conflict"
                else:
                    stats["termination_reason"] = "unsat"
                break

            bt_res = env.step(SatAction.backtrack())
            if not bool(bt_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                )
                break
            stats["backtracks"] += 1

            if not _cascade_exhausted_backtracks(
                env=env,
                tried_values_by_level=tried_values_by_level,
                stats=stats,
            ):
                break
            continue

        if env._all_satisfied(state):
            done_res = env.step(SatAction.done())
            if not bool(done_res.done):
                stats["termination_reason"] = "failed_done_after_solved"
                break
            stats["solved"] = True
            stats["termination_reason"] = "solved"
            break

        valid_actions = env.get_valid_actions()
        has_assign, has_select = _has_frontier_actions(valid_actions)
        if not has_assign and not has_select:
            stats["termination_reason"] = "no_valid_frontier_actions"
            break

        selected_var: Optional[int] = None
        decision_level: Optional[int] = None

        if has_assign:
            if state.selected_var is not None:
                selected_var = int(state.selected_var)
            elif state.decision_stack:
                selected_var = int(state.decision_stack[-1].decision_var)
            else:
                stats["termination_reason"] = "assign_available_but_no_selected_var"
                break
            decision_level = int(len(state.decision_stack))
        else:
            selectable_vars = [
                int(action.target)
                for action in valid_actions
                if action.type == SatActionType.SELECT_VAR and action.target is not None
            ]
            if not selectable_vars:
                stats["termination_reason"] = "no_valid_select_actions"
                break

            selected_var = int(
                heuristic.select_variable(
                    state=state,
                    env=env,
                    selectable_vars=selectable_vars,
                    pos_count=pos_count,
                    neg_count=neg_count,
                    rng=rng,
                )
            )

            select_res = env.step(SatAction.select_var(int(selected_var)))
            if not bool(select_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_select:{select_res.info.get('reason', 'unknown')}"
                )
                break

            decision_level = int(len(state.decision_stack) + 1)
            state = env.get_state()
            valid_actions = env.get_valid_actions()

        if selected_var is None or decision_level is None:
            stats["termination_reason"] = "selected_var_missing"
            break

        valid_assign_targets = [
            int(action.target)
            for action in valid_actions
            if action.type == SatActionType.ASSIGN_VALUE and action.target is not None
        ]
        if not valid_assign_targets:
            stats["termination_reason"] = "no_valid_assign_actions"
            break

        chosen_value = int(
            heuristic.select_value(
                state=state,
                env=env,
                selected_var=int(selected_var),
                pos_count=pos_count,
                neg_count=neg_count,
                valid_assign_targets=valid_assign_targets,
                rng=rng,
            )
        )

        prior_tried = tried_values_by_level.get(int(decision_level), set())
        if len(prior_tried) > 0:
            stats["post_backtrack_decisions"] += 1
            if int(chosen_value) in prior_tried:
                stats["repeat_errors"] += 1

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
            post_level = int(len(post_state.decision_stack))
            tried_values_by_level.setdefault(post_level, set()).add(int(chosen_value))

            stats["conflicts"] += 1
            if post_state.decision_stack:
                bt_res = env.step(SatAction.backtrack())
                if not bool(bt_res.info.get("valid", True)):
                    stats["termination_reason"] = (
                        f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                    )
                    break
                stats["backtracks"] += 1

                if not _cascade_exhausted_backtracks(
                    env=env,
                    tried_values_by_level=tried_values_by_level,
                    stats=stats,
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


def _evaluate_heuristic(
    *,
    heuristic: BaseHeuristic,
    instances: Sequence[Dict[str, Any]],
    budget: int,
    seed: int,
) -> Dict[str, Any]:
    per_instance: List[Dict[str, Any]] = []

    for idx, row in enumerate(instances):
        rng = random.Random(int(seed) + int(idx))
        stats = solve_instance(
            clauses=[tuple(int(x) for x in clause) for clause in row["clauses"]],
            num_vars=int(row["num_vars"]),
            planted_solution=None
            if row.get("planted_solution") is None
            else np.array(row["planted_solution"], dtype=np.int64, copy=True),
            heuristic=heuristic,
            budget=int(budget),
            rng=rng,
        )
        stats["instance_id"] = int(idx)
        per_instance.append(stats)

        if (idx + 1) % 25 == 0:
            solved = int(sum(int(item["solved"]) for item in per_instance))
            decisions = [float(item["decisions"]) for item in per_instance]
            conflicts = [float(item["conflicts"]) for item in per_instance]
            backtracks = [float(item["backtracks"]) for item in per_instance]
            repeats = int(sum(int(item["repeat_errors"]) for item in per_instance))
            revisits = int(
                sum(int(item["post_backtrack_decisions"]) for item in per_instance)
            )
            timeouts = int(
                sum(
                    1
                    for item in per_instance
                    if str(item["termination_reason"])
                    in {"max_steps", "budget", "timeout"}
                )
            )
            logger.info(
                "heuristic=%s processed=%d/%d solve_rate=%.3f repeat_rate=%.3f mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f timeout_rate=%.3f",
                str(heuristic.name),
                int(idx + 1),
                int(len(instances)),
                float(_safe_div(solved, len(per_instance))),
                float(_safe_div(repeats, revisits)),
                float(np.mean(decisions)) if decisions else 0.0,
                float(np.mean(conflicts)) if conflicts else 0.0,
                float(np.mean(backtracks)) if backtracks else 0.0,
                float(_safe_div(timeouts, len(per_instance))),
            )

    total = int(len(per_instance))
    solved = int(sum(int(item["solved"]) for item in per_instance))
    timeout_count = int(
        sum(
            1
            for item in per_instance
            if str(item["termination_reason"]) in {"max_steps", "budget", "timeout"}
        )
    )
    repeat_errors = int(sum(int(item["repeat_errors"]) for item in per_instance))
    post_bt_decisions = int(
        sum(int(item["post_backtrack_decisions"]) for item in per_instance)
    )

    return {
        "heuristic": str(heuristic.name),
        "solve_rate": float(_safe_div(solved, total)),
        "mean_decisions": float(
            np.mean([float(item["decisions"]) for item in per_instance])
            if total
            else 0.0
        ),
        "mean_conflicts": float(
            np.mean([float(item["conflicts"]) for item in per_instance])
            if total
            else 0.0
        ),
        "mean_backtracks": float(
            np.mean([float(item["backtracks"]) for item in per_instance])
            if total
            else 0.0
        ),
        "timeout_rate": float(_safe_div(timeout_count, total)),
        "repeat_rate": float(_safe_div(repeat_errors, post_bt_decisions)),
        "mean_repeats_per_instance": float(
            np.mean([float(item["repeat_errors"]) for item in per_instance])
            if total
            else 0.0
        ),
        "per_instance": per_instance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rule-based heuristic baselines on SAT instances"
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        choices=tuple(HEURISTICS.keys()),
        required=True,
        help="Heuristic baseline to evaluate",
    )
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--num-vars", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument(
        "--phase-transition",
        action="store_true",
        help="Use random phase-transition 3-SAT and retain only satisfiable instances.",
    )
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    _set_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_all = time.time()
    instances, generation_stats = _generate_instances_with_mode(
        num_instances=int(args.num_instances),
        num_vars=int(args.num_vars),
        alpha=float(args.alpha),
        seed=int(args.seed),
        phase_transition=bool(args.phase_transition),
    )
    logger.info(
        "generated SAT instances=%d num_vars=%d alpha=%.3f heuristic=%s budget=%d phase_transition=%s sat_ratio=%.3f",
        int(len(instances)),
        int(args.num_vars),
        float(args.alpha),
        str(args.heuristic),
        int(args.budget),
        str(bool(args.phase_transition)),
        float(generation_stats["sat_ratio"]),
    )

    heuristic = HEURISTICS[str(args.heuristic)]
    run_started = time.time()
    aggregate = _evaluate_heuristic(
        heuristic=heuristic,
        instances=instances,
        budget=int(args.budget),
        seed=int(args.seed),
    )

    result_row: Dict[str, Any] = {
        "heuristic": str(heuristic.name),
        "solve_rate": float(aggregate["solve_rate"]),
        "mean_decisions": float(aggregate["mean_decisions"]),
        "mean_conflicts": float(aggregate["mean_conflicts"]),
        "mean_backtracks": float(aggregate["mean_backtracks"]),
        "timeout_rate": float(aggregate["timeout_rate"]),
        "repeat_rate": float(aggregate["repeat_rate"]),
        "mean_repeats_per_instance": float(aggregate["mean_repeats_per_instance"]),
        "elapsed_sec": float(time.time() - run_started),
    }

    payload: Dict[str, Any] = {
        "config": {
            "heuristic": str(args.heuristic),
            "num_instances": int(args.num_instances),
            "num_vars": int(args.num_vars),
            "alpha": float(args.alpha),
            "phase_transition": bool(args.phase_transition),
            "budget": int(args.budget),
            "seed": int(args.seed),
            "elapsed_sec": float(time.time() - started_all),
        },
        "results": [result_row],
    }

    out_path = output_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info(
        "completed heuristic=%s solve_rate=%.3f repeat_rate=%.3f mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f timeout_rate=%.3f",
        str(heuristic.name),
        float(result_row["solve_rate"]),
        float(result_row["repeat_rate"]),
        float(result_row["mean_decisions"]),
        float(result_row["mean_conflicts"]),
        float(result_row["mean_backtracks"]),
        float(result_row["timeout_rate"]),
    )
    logger.info("wrote results to %s", str(out_path))


if __name__ == "__main__":
    main()
