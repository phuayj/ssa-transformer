from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np

from .dsl import SatAction, SatActionType
from .env import DecisionFrame, SatState


def _val_bit(val: int) -> int:
    v = int(val)
    if v not in {-1, 1}:
        raise ValueError("val must be in {-1,+1}")
    return 1 if v == -1 else 2


def _open_decision_var(state: SatState) -> Optional[int]:
    if not state.decision_stack:
        return None
    top: DecisionFrame = state.decision_stack[-1]
    if int(state.assignment[int(top.decision_var)]) == 0:
        return int(top.decision_var)
    return None


def _effective_domain(state: SatState, var: int) -> Set[int]:
    v = int(var)
    if v < 0 or v >= int(state.num_vars):
        raise ValueError("var out of range")

    a = int(state.assignment[v])
    if a != 0:
        return {int(a)}

    open_var = _open_decision_var(state)
    if open_var is not None and int(open_var) == int(v):
        top = state.decision_stack[-1]
        dom: Set[int] = set()
        if (int(top.failed_mask) & _val_bit(-1)) == 0:
            dom.add(-1)
        if (int(top.failed_mask) & _val_bit(1)) == 0:
            dom.add(1)
        return dom

    return {-1, 1}


class SatVerifier:
    def is_valid(self, state: SatState, action: SatAction) -> Tuple[bool, str]:
        """Check action validity based on current state."""

        if int(state.num_vars) < 1:
            return False, "state.num_vars must be >= 1"
        if int(state.num_clauses) < 1:
            return False, "state.num_clauses must be >= 1"
        if state.assignment.shape != (int(state.num_vars),):
            return False, "assignment has wrong shape"
        if len(state.clauses) != int(state.num_clauses):
            return False, "clauses length mismatch"

        open_var = _open_decision_var(state)

        if action.type == SatActionType.SELECT_VAR:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if state.conflict_clause is not None:
                return False, "conflict: must BACKTRACK/DONE"
            if state.selected_var is not None:
                return False, "var already selected"
            if action.target is None:
                return False, "SELECT_VAR missing target"
            var = int(action.target)
            if var < 0 or var >= int(state.num_vars):
                return False, "var idx out of range"
            if int(state.assignment[var]) != 0:
                return False, "var already assigned"
            if open_var is not None and int(var) != int(open_var):
                return False, "must re-select open decision var"
            if len(_effective_domain(state, var)) == 0:
                return False, "var domain empty"
            return True, ""

        if action.type == SatActionType.ASSIGN_VALUE:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if state.conflict_clause is not None:
                return False, "conflict: must BACKTRACK/DONE"
            if state.selected_var is None:
                return False, "no var selected"
            if action.target is None:
                return False, "ASSIGN_VALUE missing target"
            t = int(action.target)
            if t not in {0, 1}:
                return False, "ASSIGN_VALUE target must be 0/1"
            var = int(state.selected_var)
            if var < 0 or var >= int(state.num_vars):
                return False, "selected_var out of range"
            if int(state.assignment[var]) != 0:
                return False, "var already assigned"
            if open_var is not None and int(var) != int(open_var):
                return False, "must assign open decision var"
            val = 1 if t == 1 else -1
            if val not in _effective_domain(state, var):
                return False, "value not in domain"
            return True, ""

        if action.type == SatActionType.PROPAGATE:
            if not state.propagation_pending:
                return False, "no pending assignment"
            return True, ""

        if action.type == SatActionType.BACKTRACK:
            if not state.decision_stack:
                return False, "stack empty"
            return True, ""

        if action.type == SatActionType.DONE:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if state.selected_var is not None:
                return False, "var still selected"
            return True, ""

        return False, f"Unknown action type: {action.type}"


if __name__ == "__main__":
    from .env import SatEnv, SatEnvStatus
    from .generator import SatGenerator
    from .oracle import SatOracle

    gen = SatGenerator(seed=0)
    inst = gen.generate_planted(num_vars=20, alpha=3.5)

    env = SatEnv(clauses=inst.clauses, num_vars=inst.num_vars, planted_solution=inst.planted_solution, mode="strict")
    oracle = SatOracle(env)
    verifier = SatVerifier()

    obs = env.reset()
    del obs

    steps = 0
    while True:
        st = env.get_state()
        act = oracle.get_action(st)
        ok, reason = verifier.is_valid(st, act)
        assert ok, reason

        res = env.step(act)
        steps += 1
        if res.done:
            break

    assert env.get_state().status == SatEnvStatus.SUCCESS
    print(f"verifier.py smoke test passed (steps={steps})")
