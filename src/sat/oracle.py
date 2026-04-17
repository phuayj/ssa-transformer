from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .dsl import SatAction
from .env import SatEnv, SatEnvStatus, SatState


class SatOracle:
    """Expert 3-SAT solver using DPLL + VSIDS-lite.

    Heuristic:
      - Mandatory PROPAGATE after each ASSIGN_VALUE.
      - BACKTRACK on conflict.
      - Variable selection: highest activity among unassigned (ties by smallest var idx).
      - Value selection: True first (unless pruned by branch-local nogood).

    Note: activity is maintained by the environment (bumped on conflicts, periodic decay).
    """

    def __init__(self, env: SatEnv):
        self.env = env

    def _select_var_vsids(self, state: SatState) -> int:
        unassigned = np.nonzero(state.assignment == 0)[0]
        if unassigned.size == 0:
            return 0

        act = state.activity[unassigned]
        best = int(unassigned[int(np.argmax(act))])

        # Tie-break by smallest index.
        best_act = float(state.activity[best])
        for v in unassigned.tolist():
            if float(state.activity[int(v)]) > best_act + 1e-12:
                best = int(v)
                best_act = float(state.activity[best])
            elif abs(float(state.activity[int(v)]) - best_act) <= 1e-12 and int(v) < best:
                best = int(v)
        return int(best)

    def get_action(self, state: SatState) -> SatAction:
        # Be robust to module re-execution (e.g., `python -m sat.env`) by
        # comparing by name rather than Enum identity.
        if state.status.name != "RUNNING":
            return SatAction.done()

        # 1) Mandatory macro propagation.
        if state.propagation_pending:
            return SatAction.propagate()

        # 2) Conflict handling.
        if state.conflict_clause is not None:
            if not state.decision_stack:
                # Root-level conflict => UNSAT.
                return SatAction.done()
            return SatAction.backtrack()

        # 3) SAT check.
        if self.env._all_satisfied(state):
            return SatAction.done()

        # 4) If we are at an open decision frame, we must retry it.
        open_var = self.env._open_decision_var(state)
        if open_var is not None:
            dom = self.env._effective_domain(state, int(open_var))
            if not dom:
                return SatAction.backtrack()

            if state.selected_var is None:
                return SatAction.select_var(int(open_var))

            # Should only ever be selecting the open var.
            if int(state.selected_var) != int(open_var):
                return SatAction.select_var(int(open_var))

            # Value choice for open var.
            if 1 in dom:
                return SatAction.assign_value(1)
            return SatAction.assign_value(0)

        # 5) Normal decision.
        if state.selected_var is None:
            var = self._select_var_vsids(state)
            return SatAction.select_var(int(var))

        var = int(state.selected_var)
        dom = self.env._effective_domain(state, var)
        if not dom:
            return SatAction.backtrack()

        if 1 in dom:
            return SatAction.assign_value(1)
        return SatAction.assign_value(0)

    def solve(self) -> List[Tuple[dict, SatAction]]:
        """Generate complete trace: list of (observation, action) pairs."""

        trace: List[Tuple[dict, SatAction]] = []

        obs = self.env.reset()

        while True:
            state = self.env.get_state()
            action = self.get_action(state)

            trace.append((obs, action))

            res = self.env.step(action)
            obs = res.observation

            if res.done:
                return trace


if __name__ == "__main__":
    # Smoke test: run oracle on a planted instance.
    from .generator import SatGenerator

    gen = SatGenerator(seed=0)
    inst = gen.generate_planted(num_vars=40, alpha=3.5)

    env = SatEnv(clauses=inst.clauses, num_vars=inst.num_vars, planted_solution=inst.planted_solution, mode="strict")
    oracle = SatOracle(env)

    trace = oracle.solve()
    st = env.get_state()

    assert st.status == SatEnvStatus.SUCCESS
    assert env._all_satisfied(st)

    print(f"oracle.py smoke test passed (steps={len(trace)})")
