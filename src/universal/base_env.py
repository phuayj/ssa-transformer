"""Shared abstractions for constraint-solving environments.

This module centralizes the common step result structure and abstract interface
used across CSP, graph coloring, SAT, and hybrid environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Generic, List, Set, Tuple, TypeVar

# Type variables for state and action types
S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Common step result structure."""

    observation: dict
    reward: float
    done: bool
    info: dict

    def __post_init__(self) -> None:
        logger.debug(
            "StepResult created reward=%s done=%s obs_keys=%s info_keys=%s",
            float(self.reward),
            bool(self.done),
            sorted(self.observation.keys()),
            sorted(self.info.keys()),
        )


class BaseConstraintEnv(ABC, Generic[S, A]):
    """Abstract base class for constraint solving environments.

    All domain-specific environments (CSP, GraphColoring, SAT, Hybrid)
    should implement this interface.
    """

    @abstractmethod
    def reset(self) -> dict:
        """Reset environment to initial state. Returns observation."""

    @abstractmethod
    def get_state(self) -> S:
        """Get current state."""

    @abstractmethod
    def step(self, action: A) -> StepResult:
        """Execute action and return (obs, reward, done, info)."""

    @abstractmethod
    def get_valid_actions(self) -> List[A]:
        """Get list of valid actions from current state."""

    @abstractmethod
    def get_observation(self, state: S) -> dict:
        """Convert state to observation dict."""

    # Protected methods that subclasses typically implement
    @abstractmethod
    def _is_valid(self, action: A, state: S) -> Tuple[bool, str]:
        """Check if action is valid. Returns (is_valid, error_message)."""

    @abstractmethod
    def _propagate(self, state: S) -> bool:
        """Propagate constraints. Returns True if no contradiction."""

    @abstractmethod
    def _has_contradiction(self, state: S) -> bool:
        """Check if current state has a contradiction."""

    @abstractmethod
    def _current_depth(self, state: S) -> int:
        """Get current search depth."""

    @abstractmethod
    def _effective_domain(self, state: S, var: int) -> Set[int]:
        """Get effective domain for variable after propagation."""
