from __future__ import annotations

import logging
import threading
from typing import Dict

try:
    from pysat.solvers import Glucose4  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - pysat is an optional runtime dep
    raise ImportError(
        "pysat is required for SolvabilityOracle. Install python-sat to use it."
    ) from exc

logger = logging.getLogger(__name__)


class SolvabilityOracle:
    """Check if a partial assignment extends to a SAT solution."""

    def __init__(
        self, clauses: list[tuple[int, ...]], time_limit_sec: float | None = 0.05
    ):
        """Initialize with CNF clauses. Each clause is a tuple of literals (1-indexed)."""
        self.solver = Glucose4(bootstrap_with=[list(c) for c in clauses])
        self.time_limit_sec = None if time_limit_sec is None else float(time_limit_sec)
        self.timeout_count = 0
        self.last_timed_out = False
        self._warned_no_limit = False

    def is_extendable(self, partial_assignment: Dict[int, bool]) -> bool:
        """Check if partial assignment can be extended to satisfy the formula.

        partial_assignment: {var_index (0-based) -> True/False}
        Returns True if SAT under these assumptions.
        """
        assumptions: list[int] = []
        for var_idx, val in partial_assignment.items():
            lit = (int(var_idx) + 1) if bool(val) else -(int(var_idx) + 1)
            assumptions.append(int(lit))

        self.last_timed_out = False

        if self.time_limit_sec is None or float(self.time_limit_sec) <= 0.0:
            return bool(self.solver.solve(assumptions=assumptions))

        if hasattr(self.solver, "solve_limited") and hasattr(self.solver, "interrupt"):
            timer = threading.Timer(float(self.time_limit_sec), self.solver.interrupt)
            timer.start()
            try:
                result = self.solver.solve_limited(
                    assumptions=assumptions, expect_interrupt=True
                )
            finally:
                timer.cancel()
                if hasattr(self.solver, "clear_interrupt"):
                    try:
                        self.solver.clear_interrupt()
                    except Exception:
                        pass

            if result is None:
                self.last_timed_out = True
                self.timeout_count += 1
                if int(self.timeout_count) <= 3 or int(self.timeout_count) % 50 == 0:
                    logger.warning(
                        "SolvabilityOracle timeout (assumptions=%d, timeout=%.3fs)",
                        int(len(assumptions)),
                        float(self.time_limit_sec),
                    )
                return True

            return bool(result)

        if not bool(self._warned_no_limit):
            logger.warning(
                "SolvabilityOracle solver lacks solve_limited/interrupt; running without time limit"
            )
            self._warned_no_limit = True
        return bool(self.solver.solve(assumptions=assumptions))

    def close(self) -> None:
        self.solver.delete()
