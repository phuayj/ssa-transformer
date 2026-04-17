#!/usr/bin/env python3
"""Generate dense UP-viability labels for SAT literals.

Usage:
    python scripts/generate_dense_viability_data.py \
        --num_instances 2000 \
        --num_vars 20 \
        --alpha 4.0 \
        --max_depth 15 \
        --seed 42 \
        --output experiments/dvp-sat/dense_viability_data.pkl
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
    from pysat.solvers import Minisat22  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pysat is required for dense viability data generation. Install python-sat."
    ) from exc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _lit_value(partial: Dict[int, bool], lit: int) -> int:
    var = abs(int(lit)) - 1
    if int(var) not in partial:
        return 0
    val = bool(partial[int(var)])
    sign = int(lit) > 0
    return 1 if bool(val) == bool(sign) else -1


def _unit_suggestions(
    clauses: List[List[int]], partial: Dict[int, bool]
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


def _var_scores(
    clauses: List[List[int]], num_vars: int
) -> Tuple[List[int], List[int], List[int]]:
    scores = [0 for _ in range(int(num_vars))]
    pos_scores = [0 for _ in range(int(num_vars))]
    neg_scores = [0 for _ in range(int(num_vars))]
    for clause in clauses:
        for lit in clause:
            var = abs(int(lit)) - 1
            scores[int(var)] += 1
            if int(lit) > 0:
                pos_scores[int(var)] += 1
            else:
                neg_scores[int(var)] += 1
    return scores, pos_scores, neg_scores


def _select_random(unassigned: List[int], rng: random.Random) -> Tuple[int, bool]:
    var = int(rng.choice(unassigned))
    val = bool(rng.choice([True, False]))
    return int(var), bool(val)


def _select_greedy(
    unassigned: List[int],
    scores: List[int],
    pos_scores: List[int],
    neg_scores: List[int],
    suggestions: Dict[int, bool],
    rng: random.Random,
) -> Tuple[int, bool]:
    var = max(unassigned, key=lambda v: (int(scores[int(v)]), -int(v)))
    if int(var) in suggestions:
        return int(var), bool(suggestions[int(var)])
    if int(pos_scores[int(var)]) > int(neg_scores[int(var)]):
        return int(var), True
    if int(neg_scores[int(var)]) > int(pos_scores[int(var)]):
        return int(var), False
    return _select_random(unassigned, rng)


def _select_adversarial(
    unassigned: List[int],
    scores: List[int],
    pos_scores: List[int],
    neg_scores: List[int],
    suggestions: Dict[int, bool],
    rng: random.Random,
) -> Tuple[int, bool]:
    var = max(unassigned, key=lambda v: (int(scores[int(v)]), -int(v)))
    if int(var) in suggestions:
        return int(var), not bool(suggestions[int(var)])
    if int(pos_scores[int(var)]) > int(neg_scores[int(var)]):
        return int(var), False
    if int(neg_scores[int(var)]) > int(pos_scores[int(var)]):
        return int(var), True
    return _select_random(unassigned, rng)


def _extract_dense_features(
    clauses: List[List[int]], trail: List[int], num_vars: int
) -> np.ndarray:
    """Compute [N,2,9] literal features with shared vectorized clause stats."""
    n_vars = int(num_vars)
    clause_arr = np.asarray(clauses, dtype=np.int64)
    n_clauses = int(clause_arr.shape[0])

    assigned_vals = np.zeros((n_vars,), dtype=np.int8)
    for lit in trail:
        v = abs(int(lit)) - 1
        assigned_vals[int(v)] = 1 if int(lit) > 0 else -1

    clause_vars = np.abs(clause_arr) - 1  # [M,3]
    clause_signs = np.where(clause_arr > 0, 1, -1).astype(np.int8)  # [M,3]

    base_var_vals = assigned_vals[clause_vars]  # [M,3]
    base_assigned = base_var_vals != 0
    base_lit_vals = np.where(
        base_assigned, np.where(base_var_vals == clause_signs, 1, -1), 0
    ).astype(np.int8)

    cand_var = np.repeat(np.arange(n_vars, dtype=np.int64), 2)[
        :, None, None
    ]  # [2N,1,1]
    cand_val = np.tile(np.array([-1, 1], dtype=np.int8), n_vars)[
        :, None, None
    ]  # [2N,1,1]

    clause_vars_b = clause_vars[None, :, :]  # [1,M,3]
    clause_signs_b = clause_signs[None, :, :]  # [1,M,3]
    base_assigned_b = base_assigned[None, :, :]  # [1,M,3]
    base_lit_vals_b = base_lit_vals[None, :, :]  # [1,M,3]

    assigned_by_candidate = (~base_assigned_b) & (clause_vars_b == cand_var)
    cand_lit_vals = np.where(clause_signs_b == cand_val, 1, -1).astype(np.int8)

    lit_vals = np.where(
        base_assigned_b,
        base_lit_vals_b,
        np.where(assigned_by_candidate, cand_lit_vals, 0),
    )

    has_true = np.any(lit_vals == 1, axis=2)
    n_unset = np.sum(lit_vals == 0, axis=2)

    n_satisfied = np.sum(has_true, axis=1).astype(np.float32)
    n_empty = np.sum((~has_true) & (n_unset == 0), axis=1).astype(np.float32)
    n_unit = np.sum((~has_true) & (n_unset == 1), axis=1).astype(np.float32)
    n_unresolved = np.sum((~has_true) & (n_unset > 1), axis=1).astype(np.float32)

    pos_occ = np.zeros((n_vars,), dtype=np.float32)
    neg_occ = np.zeros((n_vars,), dtype=np.float32)
    flat_lits = clause_arr.reshape(-1)
    flat_vars = np.abs(flat_lits) - 1
    np.add.at(pos_occ, flat_vars[flat_lits > 0], 1.0)
    np.add.at(neg_occ, flat_vars[flat_lits < 0], 1.0)

    frac_assigned = float(np.count_nonzero(assigned_vals)) / float(max(n_vars, 1))
    clause_count = float(max(n_clauses, 1))

    feats = np.zeros((2 * n_vars, 9), dtype=np.float32)
    feats[:, 0] = float(frac_assigned)
    feats[:, 1] = float(n_clauses)
    feats[:, 2] = n_satisfied / clause_count
    feats[:, 3] = n_unit / clause_count
    feats[:, 4] = n_empty / clause_count
    feats[:, 5] = n_unresolved / clause_count

    cand_var_flat = np.repeat(np.arange(n_vars, dtype=np.int64), 2)
    cand_is_pos = np.tile(np.array([0.0, 1.0], dtype=np.float32), n_vars)
    feats[:, 6] = pos_occ[cand_var_flat] / clause_count
    feats[:, 7] = neg_occ[cand_var_flat] / clause_count
    feats[:, 8] = cand_is_pos

    return feats.reshape(n_vars, 2, 9)


def _dense_labels(solver: Minisat22, trail: List[int], num_vars: int) -> np.ndarray:
    labels = np.full((int(num_vars), 2), fill_value=-1, dtype=np.int8)
    assigned = {abs(int(lit)) - 1 for lit in trail}
    assumptions = [int(l) for l in trail]
    for v in range(int(num_vars)):
        if int(v) in assigned:
            continue
        lit_false = -(int(v) + 1)
        lit_true = int(v) + 1
        ok_false, _ = solver.propagate(assumptions=list(assumptions) + [int(lit_false)])
        ok_true, _ = solver.propagate(assumptions=list(assumptions) + [int(lit_true)])
        labels[int(v), 0] = np.int8(1 if bool(ok_false) else 0)
        labels[int(v), 1] = np.int8(1 if bool(ok_true) else 0)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dense UP viability SAT dataset"
    )
    parser.add_argument("--num_instances", type=int, default=2000)
    parser.add_argument("--num_vars", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/dvp-sat/dense_viability_data.pkl",
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    generator = SatGenerator(seed=int(args.seed))

    examples: List[dict] = []
    policy_counts: Dict[str, int] = defaultdict(int)
    depth_counts: Dict[int, int] = defaultdict(int)

    total_labels = 0
    total_conflicts = 0

    for inst_idx in range(int(args.num_instances)):
        instance = generator.generate_planted(
            num_vars=int(args.num_vars), alpha=float(args.alpha)
        )
        clauses = [[int(l) for l in clause] for clause in instance.clauses]

        scores, pos_scores, neg_scores = _var_scores(clauses, int(args.num_vars))

        roll = float(rng.random())
        if roll < 0.5:
            policy = "greedy"
        elif roll < 0.8:
            policy = "random"
        else:
            policy = "adversarial"
        policy_counts[str(policy)] += 1

        solver = Minisat22(bootstrap_with=[list(c) for c in clauses])
        partial: Dict[int, bool] = {}
        trail: List[int] = []

        for depth in range(int(args.max_depth)):
            unassigned = [v for v in range(int(args.num_vars)) if int(v) not in partial]
            if not unassigned:
                break

            ok, _ = solver.propagate(assumptions=list(trail))
            if not bool(ok):
                break

            labels = _dense_labels(solver, trail, int(args.num_vars))
            features = _extract_dense_features(clauses, trail, int(args.num_vars))

            valid_mask = labels >= 0
            valid_count = int(valid_mask.sum())
            conflict_count = int(np.sum(labels[valid_mask] == 0))
            total_labels += int(valid_count)
            total_conflicts += int(conflict_count)

            ex = {
                "clauses": clauses,
                "num_vars": int(args.num_vars),
                "trail": [int(x) for x in trail],
                "features": features.astype(np.float32, copy=False),
                "labels": labels.astype(np.int8, copy=False),
                "depth": int(depth),
                "instance_id": int(inst_idx),
            }
            examples.append(ex)
            depth_counts[int(depth)] += 1

            suggestions = _unit_suggestions(clauses, partial)
            if policy == "random":
                var, val = _select_random(unassigned, rng)
            elif policy == "adversarial":
                var, val = _select_adversarial(
                    unassigned, scores, pos_scores, neg_scores, suggestions, rng
                )
            else:
                var, val = _select_greedy(
                    unassigned, scores, pos_scores, neg_scores, suggestions, rng
                )

            lit = int(var + 1) if bool(val) else -(int(var) + 1)
            ok_lit, _ = solver.propagate(assumptions=list(trail) + [int(lit)])
            if not bool(ok_lit):
                break
            partial[int(var)] = bool(val)
            trail.append(int(lit))

        solver.delete()

        if (inst_idx + 1) % 100 == 0:
            logger.info(
                "generated %d/%d instances examples=%d labels=%d conflict_rate=%.3f",
                int(inst_idx + 1),
                int(args.num_instances),
                int(len(examples)),
                int(total_labels),
                float(total_conflicts) / max(float(total_labels), 1.0),
            )

    stats = {
        "total_examples": int(len(examples)),
        "total_literals_labeled": int(total_labels),
        "conflict_rate": float(total_conflicts) / max(float(total_labels), 1.0),
        "policy_counts": {k: int(v) for k, v in sorted(policy_counts.items())},
        "examples_by_depth": {int(k): int(v) for k, v in sorted(depth_counts.items())},
    }

    config = {
        "num_instances": int(args.num_instances),
        "num_vars": int(args.num_vars),
        "alpha": float(args.alpha),
        "max_depth": int(args.max_depth),
        "seed": int(args.seed),
        "policy_mix": {"greedy": 0.5, "random": 0.3, "adversarial": 0.2},
    }

    payload = {"examples": examples, "config": config, "stats": stats}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    logger.info(
        "saved %d examples to %s (literals=%d conflict_rate=%.3f)",
        int(len(examples)),
        str(output_path),
        int(total_labels),
        float(stats["conflict_rate"]),
    )


if __name__ == "__main__":
    main()
