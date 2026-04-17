"""Unified factor-graph interface for backtracking search domains."""

from .base_env import BaseConstraintEnv, StepResult
from .base_oracle import BaseOracle
from .types import (
    ConstraintType,
    UnifiedAction,
    UnifiedActionType,
    UnifiedObservation,
    UnifiedState,
)

# Core models
from .model import FactorGNN, create_model_inputs_from_obs, decode_action
from .constraint_transformer import ConstraintTransformer, ConstraintTransformerMinimal

# Domain wrappers (lazy - only import the ones for included domains)
from .wrapper import (
    GraphColoringUnifiedWrapper,
    SATUnifiedWrapper,
    UnifiedEnvWrapper,
)

__all__ = [
    "BaseOracle",
    "BaseConstraintEnv",
    "StepResult",
    "ConstraintType",
    "UnifiedActionType",
    "UnifiedAction",
    "UnifiedObservation",
    "UnifiedState",
    "UnifiedEnvWrapper",
    "GraphColoringUnifiedWrapper",
    "SATUnifiedWrapper",
    "FactorGNN",
    "create_model_inputs_from_obs",
    "decode_action",
    "ConstraintTransformer",
    "ConstraintTransformerMinimal",
]
