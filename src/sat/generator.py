from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SatInstance:
    clauses: List[Tuple[int, int, int]]  # each clause: 3 literals ±(var+1)
    num_vars: int
    planted_solution: Optional[np.ndarray]  # (num_vars,) values in {-1,+1}
    is_satisfiable: Optional[bool]  # None if unknown until solved


class SatGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_planted(self, num_vars: int, alpha: float = 3.5) -> SatInstance:
        """Generate planted-solution 3-SAT (always SAT)."""

        if int(num_vars) < 3:
            raise ValueError("num_vars must be >= 3 for 3-SAT")
        if float(alpha) <= 0.0:
            raise ValueError("alpha must be > 0")

        n = int(num_vars)
        m = int(float(alpha) * float(n))
        m = max(1, m)

        planted = self.rng.choice(np.array([-1, 1], dtype=np.int64), size=n, replace=True)

        clauses: List[Tuple[int, int, int]] = []
        for _ in range(m):
            vars_ = self.rng.choice(n, size=3, replace=False)
            signs = self.rng.choice(np.array([-1, 1], dtype=np.int64), size=3, replace=True)

            lits = [(int(v) + 1) * int(s) for v, s in zip(vars_, signs)]

            # Reject if clause falsified by planted assignment.
            satisfied = any(int(planted[int(v)]) == int(s) for v, s in zip(vars_, signs))
            if not satisfied:
                flip_idx = int(self.rng.integers(0, 3))
                signs[flip_idx] *= -1
                lits = [(int(v) + 1) * int(s) for v, s in zip(vars_, signs)]

            clauses.append((int(lits[0]), int(lits[1]), int(lits[2])))

        return SatInstance(
            clauses=clauses,
            num_vars=n,
            planted_solution=np.array(planted, dtype=np.int64, copy=True),
            is_satisfiable=True,
        )

    def generate_random(self, num_vars: int, alpha: float = 5.0) -> SatInstance:
        """Generate random 3-SAT instance (may be SAT or UNSAT).

        Higher alpha (clause/var ratio) → more likely UNSAT.
        Phase transition at α ≈ 4.26.

        Args:
            num_vars: Number of Boolean variables
            alpha: Clause/variable ratio (default 5.0 for likely UNSAT)

        Returns:
            SatInstance with is_satisfiable=None (unknown until solved)
        """

        if int(num_vars) < 3:
            raise ValueError("num_vars must be >= 3 for 3-SAT")
        if float(alpha) <= 0.0:
            raise ValueError("alpha must be > 0")

        n = int(num_vars)
        m = int(float(alpha) * float(n))
        m = max(1, m)

        clauses: List[Tuple[int, int, int]] = []
        for _ in range(m):
            # Sample 3 distinct variables
            vars_ = self.rng.choice(n, size=3, replace=False)
            # Sample random signs
            signs = self.rng.choice([-1, 1], size=3)
            # Form literals: ±(var+1)
            lits = [(int(v) + 1) * int(s) for v, s in zip(vars_, signs)]
            clauses.append((int(lits[0]), int(lits[1]), int(lits[2])))

        return SatInstance(
            clauses=clauses,
            num_vars=n,
            planted_solution=None,  # Unknown
            is_satisfiable=None,  # Must be determined by solver
        )

    def generate(
        self,
        num_vars: int,
        alpha_sat: float = 3.5,
        alpha_unsat: float = 5.5,
        sat_ratio: float = 0.5,
    ) -> SatInstance:
        """Generate instance with controlled SAT/UNSAT distribution.

        Args:
            num_vars: Number of variables
            alpha_sat: Alpha for planted SAT instances
            alpha_unsat: Alpha for random (likely UNSAT) instances
            sat_ratio: Probability of generating a SAT instance

        Returns:
            SatInstance (SAT is guaranteed planted; UNSAT is random high-alpha)
        """

        r = float(sat_ratio)
        if r < 0.0 or r > 1.0:
            raise ValueError("sat_ratio must be in [0,1]")

        if float(self.rng.random()) < r:
            return self.generate_planted(num_vars, alpha=float(alpha_sat))
        return self.generate_random(num_vars, alpha=float(alpha_unsat))


def label_instance_with_oracle(instance: SatInstance, max_steps: int = 5000) -> SatInstance:
    """Run oracle on instance to determine SAT/UNSAT.

    Returns instance with is_satisfiable set based on oracle result.
    is_satisfiable=None indicates timeout/undetermined.
    """

    from .env import SatEnv, SatEnvStatus
    from .oracle import SatOracle

    env = SatEnv(
        clauses=instance.clauses,
        num_vars=int(instance.num_vars),
        planted_solution=instance.planted_solution,
        mode="strict",
        max_steps=int(max_steps),
    )
    oracle = SatOracle(env)
    oracle.solve()

    state = env.get_state()

    solution: Optional[np.ndarray] = None

    if state.status == SatEnvStatus.SUCCESS:
        is_sat: Optional[bool] = True
        solution = np.array(state.assignment, dtype=np.int64, copy=True)
        solution[solution == 0] = 1

    elif state.status == SatEnvStatus.FAILURE:
        term = getattr(state, "termination_reason", None)
        if term == "unsat" or ((not state.decision_stack) and state.conflict_clause is not None):
            is_sat = False
        else:
            # timeout / invalid / unknown
            is_sat = None

    else:
        # Should not happen for a completed rollout, but keep semantics explicit.
        is_sat = None

    return SatInstance(
        clauses=instance.clauses,
        num_vars=int(instance.num_vars),
        planted_solution=solution if is_sat is True else instance.planted_solution,
        is_satisfiable=is_sat,
    )


def _lit_value(assignment: np.ndarray, lit: int) -> int:
    """Return 1 if lit true, -1 if false, 0 if unassigned."""

    v = int(abs(int(lit)) - 1)
    sign = 1 if int(lit) > 0 else -1
    a = int(assignment[v])
    if a == 0:
        return 0
    return 1 if int(a) == int(sign) else -1


if __name__ == "__main__":
    gen = SatGenerator(seed=0)
    inst = gen.generate_planted(num_vars=20, alpha=3.0)

    assert inst.planted_solution is not None
    asn = inst.planted_solution

    # Planted solution should satisfy all clauses.
    for c in inst.clauses:
        vals = [_lit_value(asn, int(l)) for l in c]
        assert any(v == 1 for v in vals), f"Planted assignment falsifies clause {c}"

    print(f"generator.py smoke test passed (n={inst.num_vars} m={len(inst.clauses)})")
