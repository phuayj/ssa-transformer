"""Unit-propagation engine used for UP-CoT trace generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _lit_eval(lit: int, assignment: np.ndarray) -> int:
    """Evaluate literal under assignment.

    Returns:
        1 if literal is true, -1 if false, 0 if unassigned.
    """
    lit = int(lit)
    var_id = int(abs(lit) - 1)
    val = int(assignment[var_id])
    if val == 0:
        return 0
    if (lit > 0 and val == 1) or (lit < 0 and val == -1):
        return 1
    return -1


def compute_up_chain(
    clauses: List[Tuple[int, ...]],
    assignment: np.ndarray,
    var_id: int,
    polarity: int,
    num_vars: int,
    max_rounds: int = 20,
) -> Dict[str, Any]:
    """Run unit propagation round-by-round from a hypothetical branch decision.

    Args:
        clauses: DIMACS clauses with 1-indexed signed literals.
        assignment: Current assignment, shape [num_vars], values in {-1, 0, 1}.
        var_id: Variable being hypothetically assigned (0-indexed).
        polarity: +1 for True branch, -1 for False branch.
        num_vars: Number of variables.
        max_rounds: Maximum number of propagation rounds.

    Returns:
        Dict with keys:
            rounds: List[List[int]] of forced DIMACS literals per round.
            conflict: Whether a conflict was found.
            conflict_clause_id: Clause index when conflict was detected (if known).
            total_forced: Total number of forced literals in all rounds.
            num_rounds: Number of rounds completed (len(rounds)).
    """
    num_vars = int(num_vars)
    var_id = int(var_id)
    polarity = int(polarity)
    max_rounds = int(max_rounds)

    if polarity not in (-1, 1):
        raise ValueError(f"polarity must be ±1, got {polarity}")
    if var_id < 0 or var_id >= num_vars:
        raise ValueError(f"var_id out of range: {var_id} (num_vars={num_vars})")

    asgn = np.asarray(assignment, dtype=np.int8).copy()
    if asgn.shape[0] != num_vars:
        raise ValueError(
            f"assignment length {int(asgn.shape[0])} != num_vars {int(num_vars)}"
        )
    if int(asgn[var_id]) not in (0, polarity):
        return {
            "rounds": [],
            "conflict": True,
            "conflict_clause_id": None,
            "total_forced": 0,
            "num_rounds": 0,
        }

    asgn[var_id] = np.int8(polarity)

    rounds: List[List[int]] = []
    total_forced = 0

    for round_idx in range(max_rounds):
        forced_by_var: Dict[int, Tuple[int, int]] = {}
        conflict_clause_id: Optional[int] = None

        for cid, clause in enumerate(clauses):
            num_unassigned = 0
            last_unassigned_lit = 0
            clause_satisfied = False

            for lit_raw in clause:
                lit = int(lit_raw)
                ev = _lit_eval(lit, asgn)
                if ev == 1:
                    clause_satisfied = True
                    break
                if ev == 0:
                    num_unassigned += 1
                    last_unassigned_lit = int(lit)

            if clause_satisfied:
                continue
            if num_unassigned == 0:
                conflict_clause_id = int(cid)
                break
            if num_unassigned == 1:
                forced_lit = int(last_unassigned_lit)
                forced_var = int(abs(forced_lit) - 1)
                forced_pol = 1 if forced_lit > 0 else -1
                prev = forced_by_var.get(forced_var)
                if prev is None:
                    forced_by_var[forced_var] = (int(forced_pol), int(cid))
                elif int(prev[0]) != int(forced_pol):
                    conflict_clause_id = int(cid)
                    break

        if conflict_clause_id is not None:
            logger.debug(
                "compute_up_chain conflict: var=%d pol=%d round=%d clause=%d",
                int(var_id),
                int(polarity),
                int(round_idx),
                int(conflict_clause_id),
            )
            return {
                "rounds": rounds,
                "conflict": True,
                "conflict_clause_id": int(conflict_clause_id),
                "total_forced": int(total_forced),
                "num_rounds": int(len(rounds)),
            }

        if not forced_by_var:
            return {
                "rounds": rounds,
                "conflict": False,
                "conflict_clause_id": None,
                "total_forced": int(total_forced),
                "num_rounds": int(len(rounds)),
            }

        round_forced: List[int] = []
        for forced_var, (forced_pol, _cid) in forced_by_var.items():
            existing = int(asgn[forced_var])
            if existing != 0 and existing != int(forced_pol):
                return {
                    "rounds": rounds,
                    "conflict": True,
                    "conflict_clause_id": None,
                    "total_forced": int(total_forced),
                    "num_rounds": int(len(rounds)),
                }
            if existing == 0:
                asgn[forced_var] = np.int8(forced_pol)
                lit = (
                    int(forced_var + 1)
                    if int(forced_pol) == 1
                    else int(-(forced_var + 1))
                )
                round_forced.append(int(lit))

        round_forced = sorted(round_forced, key=lambda lit: (abs(int(lit)), int(lit)))
        if not round_forced:
            return {
                "rounds": rounds,
                "conflict": False,
                "conflict_clause_id": None,
                "total_forced": int(total_forced),
                "num_rounds": int(len(rounds)),
            }

        rounds.append([int(l) for l in round_forced])
        total_forced += int(len(round_forced))

    logger.warning(
        "compute_up_chain reached max_rounds=%d var=%d pol=%d total_forced=%d",
        int(max_rounds),
        int(var_id),
        int(polarity),
        int(total_forced),
    )
    return {
        "rounds": rounds,
        "conflict": False,
        "conflict_clause_id": None,
        "total_forced": int(total_forced),
        "num_rounds": int(len(rounds)),
    }


def compute_up_chain_with_reasons(
    clauses: List[Tuple[int, ...]],
    assignment: np.ndarray,
    var_id: int,
    polarity: int,
    num_vars: int,
    max_rounds: int = 20,
) -> Dict[str, Any]:
    """Like compute_up_chain but returns (clause_id, literal) pairs.

    Returns:
        Dict with keys:
            rounds: List[List[Tuple[int, int]]] — list of (clause_id, DIMACS_literal) pairs per round
            conflict: bool
            conflict_clause_id: Optional[int]
            total_forced: int
            num_rounds: int
    """
    num_vars = int(num_vars)
    var_id = int(var_id)
    polarity = int(polarity)
    max_rounds = int(max_rounds)

    if polarity not in (-1, 1):
        raise ValueError(f"polarity must be ±1, got {polarity}")
    if var_id < 0 or var_id >= num_vars:
        raise ValueError(f"var_id out of range: {var_id} (num_vars={num_vars})")

    asgn = np.asarray(assignment, dtype=np.int8).copy()
    if asgn.shape[0] != num_vars:
        raise ValueError(
            f"assignment length {int(asgn.shape[0])} != num_vars {int(num_vars)}"
        )
    if int(asgn[var_id]) not in (0, polarity):
        return {
            "rounds": [],
            "conflict": True,
            "conflict_clause_id": None,
            "total_forced": 0,
            "num_rounds": 0,
        }

    asgn[var_id] = np.int8(polarity)

    rounds: List[List[Tuple[int, int]]] = []
    total_forced = 0

    for round_idx in range(max_rounds):
        forced_by_var: Dict[int, Tuple[int, int]] = {}
        conflict_clause_id: Optional[int] = None

        for cid, clause in enumerate(clauses):
            num_unassigned = 0
            last_unassigned_lit = 0
            clause_satisfied = False

            for lit_raw in clause:
                lit = int(lit_raw)
                ev = _lit_eval(lit, asgn)
                if ev == 1:
                    clause_satisfied = True
                    break
                if ev == 0:
                    num_unassigned += 1
                    last_unassigned_lit = int(lit)

            if clause_satisfied:
                continue
            if num_unassigned == 0:
                conflict_clause_id = int(cid)
                break
            if num_unassigned == 1:
                forced_lit = int(last_unassigned_lit)
                forced_var = int(abs(forced_lit) - 1)
                forced_pol = 1 if forced_lit > 0 else -1
                prev = forced_by_var.get(forced_var)
                if prev is None:
                    forced_by_var[forced_var] = (int(forced_pol), int(cid))
                elif int(prev[0]) != int(forced_pol):
                    conflict_clause_id = int(cid)
                    break

        if conflict_clause_id is not None:
            logger.debug(
                "compute_up_chain_with_reasons conflict: var=%d pol=%d round=%d clause=%d",
                int(var_id),
                int(polarity),
                int(round_idx),
                int(conflict_clause_id),
            )
            return {
                "rounds": rounds,
                "conflict": True,
                "conflict_clause_id": int(conflict_clause_id),
                "total_forced": int(total_forced),
                "num_rounds": int(len(rounds)),
            }

        if not forced_by_var:
            return {
                "rounds": rounds,
                "conflict": False,
                "conflict_clause_id": None,
                "total_forced": int(total_forced),
                "num_rounds": int(len(rounds)),
            }

        round_forced: List[Tuple[int, int]] = []
        for forced_var, (forced_pol, cid) in forced_by_var.items():
            existing = int(asgn[forced_var])
            if existing != 0 and existing != int(forced_pol):
                return {
                    "rounds": rounds,
                    "conflict": True,
                    "conflict_clause_id": None,
                    "total_forced": int(total_forced),
                    "num_rounds": int(len(rounds)),
                }
            if existing == 0:
                asgn[forced_var] = np.int8(forced_pol)
                lit = (
                    int(forced_var + 1)
                    if int(forced_pol) == 1
                    else int(-(forced_var + 1))
                )
                round_forced.append((int(cid), int(lit)))

        round_forced = sorted(
            round_forced,
            key=lambda x: (int(x[0]), abs(int(x[1])), int(x[1])),
        )
        if not round_forced:
            return {
                "rounds": rounds,
                "conflict": False,
                "conflict_clause_id": None,
                "total_forced": int(total_forced),
                "num_rounds": int(len(rounds)),
            }

        rounds.append([(int(cid), int(lit)) for cid, lit in round_forced])
        total_forced += int(len(round_forced))

    logger.warning(
        "compute_up_chain_with_reasons reached max_rounds=%d var=%d pol=%d total_forced=%d",
        int(max_rounds),
        int(var_id),
        int(polarity),
        int(total_forced),
    )
    return {
        "rounds": rounds,
        "conflict": False,
        "conflict_clause_id": None,
        "total_forced": int(total_forced),
        "num_rounds": int(len(rounds)),
    }
