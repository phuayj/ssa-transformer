from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


class ConstraintType(IntEnum):
    """Constraint types across domains."""

    NEQ = 0  # x != y (Graph Coloring edges)
    ALLDIFF = 1  # all different (Sudoku row/col/box)
    CLAUSE = 2  # Boolean disjunction (SAT clauses)


class UnifiedActionType(IntEnum):
    """Universal action types."""

    ASSIGN = 0  # ASSIGN(var, value)
    BACKTRACK = 1
    DONE = 2


@dataclass(frozen=True)
class UnifiedAction:
    type: UnifiedActionType
    var: Optional[int] = None  # For ASSIGN
    value: Optional[int] = None  # For ASSIGN

    @classmethod
    def assign(cls, var: int, value: int) -> "UnifiedAction":
        return cls(UnifiedActionType.ASSIGN, var=int(var), value=int(value))

    @classmethod
    def backtrack(cls) -> "UnifiedAction":
        return cls(UnifiedActionType.BACKTRACK)

    @classmethod
    def done(cls) -> "UnifiedAction":
        return cls(UnifiedActionType.DONE)


@dataclass
class UnifiedObservation:
    """Unified observation for factor-graph CSP."""

    # Variable tensors
    var_domain_mask: np.ndarray  # [N, D_max] bool - valid values
    var_nogood_mask: np.ndarray  # [N, D_max] bool - branch-local nogoods
    var_assigned: np.ndarray  # [N] int, -1 = unassigned, else value index
    var_features: np.ndarray  # [N, F_v] float - features

    # Constraint tensors
    con_type: np.ndarray  # [M] int - ConstraintType
    con_scope: np.ndarray  # [M, R_max] int - var indices, -1 padded
    con_features: np.ndarray  # [M, F_c] float - features

    # Graph structure (COO format)
    edge_con_idx: np.ndarray  # [E] int - constraint index
    edge_var_idx: np.ndarray  # [E] int - variable index
    edge_features: np.ndarray  # [E, F_e] float - position, sign

    # Metadata
    num_vars: int
    num_constraints: int
    max_domain: int
    stack_depth: int
    propagation_pending: bool
    has_conflict: bool
    propagation_mode: int = 1  # 0=none, 1=forward_check (default: forward_check)
    domain_id: int = 0  # 0=CSP, 1=Coloring, 2=SAT (for debugging)


@dataclass
class UnifiedState:
    """Full state for unified environment."""

    obs: UnifiedObservation

    # Internal tracking (not exposed to model)
    assignment_stack: List[Tuple[int, int, np.ndarray]]  # (var, value, domain_snapshot)
    nogoods: dict
    step_count: int
    done: bool
    success: bool
