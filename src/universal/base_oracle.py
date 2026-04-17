"""Shared abstractions for oracle policies used in training.

Oracles provide expert, heuristic-driven actions that generate demonstration
trajectories. Those (observation, action) traces are consumed by the training
pipeline to supervise learned policies or to seed offline RL datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Generic, List, Tuple, TypeVar

# Type variables
E = TypeVar("E")  # Environment type
S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type

logger = logging.getLogger(__name__)


class BaseOracle(ABC, Generic[E, S, A]):
    """Abstract base class for constraint solving oracles.

    Oracles provide expert actions for training data generation.
    They implement domain-specific heuristics (MRV, DSATUR, VSIDS, etc.)
    to solve constraint problems optimally or near-optimally.

    In the training pipeline, :meth:`solve` should emit the full
    (observation, action) trace used to supervise downstream models.
    """

    def __init__(self, env: E):
        """Initialize oracle with environment."""
        self.env = env
        logger.debug("BaseOracle initialized env_type=%s", type(env).__name__)

    @abstractmethod
    def get_action(self, state: S) -> A:
        """Get expert action for current state.

        This should implement the domain-specific heuristic
        (e.g., MRV for CSP, DSATUR for graph coloring, VSIDS for SAT).
        """
        ...

    @abstractmethod
    def solve(self) -> List[Tuple[dict, A]]:
        """Solve the problem and return trace of (observation, action) pairs.

        This is used for training data generation.
        """
        ...

    def _backtrack_action(self, state: S) -> A:
        """Create a backtrack action.

        Default implementation raises NotImplementedError.
        Subclasses should override if backtracking is supported.
        """
        raise NotImplementedError("Backtracking not implemented for this oracle")
