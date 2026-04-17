#!/usr/bin/env python3
"""Generate balanced SAT verifier data with mixed rollout policies.

Usage:
    python scripts/generate_balanced_sat_data.py \
        --num_instances 1000 \
        --num_vars 20 \
        --alpha 4.0 \
        --rollouts_per_instance 10 \
        --max_depth 20 \
        --target_ratio 0.5 \
        --output data/balanced_sat.pkl \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sat.generator import SatGenerator

try:
    from pysat.solvers import Glucose4  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - optional runtime dep
    raise ImportError(
        "pysat is required for SAT data generation. Install python-sat to use it."
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEPTH_BUCKETS: List[Tuple[int, int]] = [(0, 4), (5, 9), (10, 14), (15, 20)]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _assumptions_from_partial(partial: Dict[int, bool]) -> List[int]:
    return [
        int(var) + 1 if bool(val) else -(int(var) + 1) for var, val in partial.items()
    ]


def _lit_value(partial: Dict[int, bool], lit: int) -> int:
    var = abs(int(lit)) - 1
    if int(var) not in partial:
        return 0
    val = bool(partial[int(var)])
    sign = int(lit) > 0
    return 1 if bool(val) == bool(sign) else -1


def _unit_suggestions(
    clauses: List[Tuple[int, int, int]], partial: Dict[int, bool]
) -> Dict[int, bool]:
    suggestions: Dict[int, bool | None] = {}
    for clause in clauses:
        satisfied = False
        unassigned: List[int] = []
        for lit in clause:
            val = _lit_value(partial, int(lit))
            if int(val) == 1:
                satisfied = True
                break
            if int(val) == 0:
                unassigned.append(int(lit))
        if satisfied:
            continue
        if len(unassigned) == 1:
            lit = int(unassigned[0])
            var = abs(int(lit)) - 1
            suggested = bool(int(lit) > 0)
            if int(var) in suggestions and suggestions[int(var)] != suggested:
                suggestions[int(var)] = None
            else:
                suggestions[int(var)] = suggested
    return {int(k): bool(v) for k, v in suggestions.items() if v is not None}


def _var_scores(clauses: List[Tuple[int, int, int]], num_vars: int) -> List[int]:
    scores = [0 for _ in range(int(num_vars))]
    for clause in clauses:
        for lit in clause:
            scores[abs(int(lit)) - 1] += 1
    return scores


def _select_random(unassigned: List[int], rng: random.Random) -> Tuple[int, bool]:
    var = int(rng.choice(unassigned))
    val = bool(rng.choice([True, False]))
    return int(var), bool(val)


def _select_greedy_bad(
    unassigned: List[int],
    var_scores: List[int],
    suggestions: Dict[int, bool],
    rng: random.Random,
) -> Tuple[int, bool]:
    var = max(unassigned, key=lambda v: (int(var_scores[int(v)]), -int(v)))
    if int(var) in suggestions:
        return int(var), not bool(suggestions[int(var)])
    return _select_random(unassigned, rng)


def _select_oracle(
    unassigned: List[int],
    var_scores: List[int],
    solver: Glucose4,
    partial: Dict[int, bool],
    rng: random.Random,
) -> Tuple[int, bool]:
    var = max(unassigned, key=lambda v: (int(var_scores[int(v)]), -int(v)))
    assumptions = _assumptions_from_partial(partial)
    if not bool(solver.solve(assumptions=assumptions)):
        return int(var), bool(rng.choice([True, False]))
    model = solver.get_model()
    if model is None or int(var) >= len(model):
        return int(var), bool(rng.choice([True, False]))
    return int(var), bool(int(model[int(var)]) > 0)


def _depth_bucket(depth: int) -> int:
    for idx, (_lo, hi) in enumerate(DEPTH_BUCKETS):
        if int(depth) <= int(hi):
            return int(idx)
    return int(len(DEPTH_BUCKETS) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate balanced SAT verifier data with diverse rollouts"
    )
    parser.add_argument("--num_instances", type=int, default=1000)
    parser.add_argument("--num_vars", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--rollouts_per_instance", type=int, default=10)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--target_ratio", type=float, default=0.5)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    generator = SatGenerator(seed=int(args.seed))

    states: List[Dict[int, bool]] = []
    labels: List[bool] = []
    depths: List[int] = []
    clause_lists: List[List[List[int]]] = []
    state_map: Dict[
        Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, bool], ...]], int
    ] = {}

    policy_rollout_counts = defaultdict(int)
    policy_state_counts = defaultdict(int)
    near_boundary_pairs = 0

    def maybe_add_state(
        clause_key: Tuple[Tuple[int, ...], ...],
        clause_list: List[List[int]],
        partial: Dict[int, bool],
        extendable: bool,
    ) -> bool:
        signature = (clause_key, tuple((int(k), bool(v)) for k, v in partial.items()))
        if signature in state_map:
            return False
        state_map[signature] = int(len(states))
        states.append(dict(partial))
        labels.append(bool(extendable))
        depths.append(int(len(partial)))
        clause_lists.append(clause_list)
        return True

    for inst_idx in range(int(args.num_instances)):
        instance = generator.generate_planted(
            num_vars=int(args.num_vars), alpha=float(args.alpha)
        )
        clauses = instance.clauses
        clause_list = [[int(l) for l in clause] for clause in clauses]
        clause_key = tuple(tuple(int(l) for l in clause) for clause in clauses)
        var_scores = _var_scores(clauses, int(args.num_vars))

        solver = Glucose4(bootstrap_with=[list(c) for c in clauses])

        for _ in range(int(args.rollouts_per_instance)):
            roll = float(rng.random())
            if roll < 0.4:
                policy = "random_policy"
            elif roll < 0.7:
                policy = "greedy_bad_policy"
            else:
                policy = "oracle_policy"
            policy_rollout_counts[policy] += 1

            partial: Dict[int, bool] = {}

            for _depth in range(int(args.max_depth)):
                unassigned = [
                    int(v) for v in range(int(args.num_vars)) if int(v) not in partial
                ]
                if not unassigned:
                    break

                if policy == "random_policy":
                    var, val = _select_random(unassigned, rng)
                elif policy == "greedy_bad_policy":
                    suggestions = _unit_suggestions(clauses, partial)
                    var, val = _select_greedy_bad(
                        unassigned, var_scores, suggestions, rng
                    )
                else:
                    var, val = _select_oracle(
                        unassigned, var_scores, solver, partial, rng
                    )

                if int(var) in partial:
                    raise RuntimeError("selected variable already assigned")

                partial[int(var)] = bool(val)

                assumptions = _assumptions_from_partial(partial)
                extendable = bool(solver.solve(assumptions=assumptions))

                if maybe_add_state(clause_key, clause_list, partial, extendable):
                    policy_state_counts[policy] += 1

                if not extendable and len(partial) > 0:
                    parent = dict(partial)
                    last_key = next(reversed(parent))
                    parent.pop(int(last_key))
                    parent_extendable = bool(
                        solver.solve(assumptions=_assumptions_from_partial(parent))
                    )
                    if parent_extendable:
                        added = maybe_add_state(clause_key, clause_list, parent, True)
                        if added:
                            near_boundary_pairs += 1
                    break

                if not extendable:
                    break

        solver.delete()

        if (inst_idx + 1) % 50 == 0:
            logger.info(
                "generated %d/%d instances (states=%d)",
                int(inst_idx + 1),
                int(args.num_instances),
                int(len(states)),
            )

    total_states = int(len(states))
    dead_states = int(sum(1 for x in labels if not bool(x)))
    extendable_states = int(total_states) - int(dead_states)
    dead_ratio = float(dead_states) / max(float(total_states), 1.0)

    logger.info(
        "raw states=%d extendable=%d dead_end=%d dead_ratio=%.3f near_boundary_pairs=%d",
        int(total_states),
        int(extendable_states),
        int(dead_states),
        float(dead_ratio),
        int(near_boundary_pairs),
    )
    for policy in ("random_policy", "greedy_bad_policy", "oracle_policy"):
        logger.info(
            "policy=%s rollouts=%d states=%d",
            str(policy),
            int(policy_rollout_counts[policy]),
            int(policy_state_counts[policy]),
        )

    bucket_indices: Dict[int, Dict[bool, List[int]]] = {
        int(i): {True: [], False: []} for i in range(int(len(DEPTH_BUCKETS)))
    }
    for idx, label in enumerate(labels):
        bucket = _depth_bucket(int(depths[int(idx)]))
        bucket_indices[int(bucket)][bool(label)].append(int(idx))

    selected: List[int] = []
    for bucket, label_map in bucket_indices.items():
        dead_idx = label_map[False]
        ext_idx = label_map[True]
        if not dead_idx or not ext_idx:
            logger.warning(
                "bucket %s has dead=%d extendable=%d; skipping for balance",
                str(DEPTH_BUCKETS[int(bucket)]),
                int(len(dead_idx)),
                int(len(ext_idx)),
            )
            continue

        max_total = min(
            float(len(dead_idx)) / float(args.target_ratio),
            float(len(ext_idx)) / max(1.0 - float(args.target_ratio), 1e-9),
        )
        total_bucket = int(max_total)
        if total_bucket <= 1:
            logger.warning(
                "bucket %s too small for balancing (dead=%d extendable=%d)",
                str(DEPTH_BUCKETS[int(bucket)]),
                int(len(dead_idx)),
                int(len(ext_idx)),
            )
            continue

        dead_take = int(total_bucket * float(args.target_ratio))
        dead_take = min(int(dead_take), int(len(dead_idx)))
        ext_take = int(total_bucket) - int(dead_take)
        ext_take = min(int(ext_take), int(len(ext_idx)))

        if dead_take <= 0 or ext_take <= 0:
            logger.warning(
                "bucket %s insufficient for target ratio; dead_take=%d ext_take=%d",
                str(DEPTH_BUCKETS[int(bucket)]),
                int(dead_take),
                int(ext_take),
            )
            continue

        selected.extend(rng.sample(dead_idx, int(dead_take)))
        selected.extend(rng.sample(ext_idx, int(ext_take)))

        logger.info(
            "bucket %s selected dead=%d extendable=%d",
            str(DEPTH_BUCKETS[int(bucket)]),
            int(dead_take),
            int(ext_take),
        )

    rng.shuffle(selected)

    states = [states[i] for i in selected]
    labels = [labels[i] for i in selected]
    depths = [depths[i] for i in selected]
    clause_lists = [clause_lists[i] for i in selected]

    balanced_total = int(len(states))
    balanced_dead = int(sum(1 for x in labels if not bool(x)))
    balanced_ratio = float(balanced_dead) / max(float(balanced_total), 1.0)

    logger.info(
        "balanced states=%d dead_end=%d dead_ratio=%.3f target_ratio=%.2f",
        int(balanced_total),
        int(balanced_dead),
        float(balanced_ratio),
        float(args.target_ratio),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "num_instances": int(args.num_instances),
        "num_vars": int(args.num_vars),
        "alpha": float(args.alpha),
        "rollouts_per_instance": int(args.rollouts_per_instance),
        "max_depth": int(args.max_depth),
        "target_ratio": float(args.target_ratio),
        "seed": int(args.seed),
        "raw_states": int(total_states),
        "raw_dead_ratio": float(dead_ratio),
        "balanced_states": int(balanced_total),
        "balanced_dead_ratio": float(balanced_ratio),
        "near_boundary_pairs": int(near_boundary_pairs),
        "policy_rollout_counts": dict(policy_rollout_counts),
        "policy_state_counts": dict(policy_state_counts),
        "depth_buckets": [list(x) for x in DEPTH_BUCKETS],
    }

    payload = {
        "states": states,
        "labels": labels,
        "depths": depths,
        "clause_lists": clause_lists,
        "config": config,
    }

    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    logger.info("saved dataset to %s", str(output_path))


if __name__ == "__main__":
    main()
