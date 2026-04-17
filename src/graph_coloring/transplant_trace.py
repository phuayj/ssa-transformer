"""Reusable graph-coloring history-transplant trace utilities."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_reference() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "evaluation" / "eval_gc_history_transplant.py"
    spec = importlib.util.spec_from_file_location("_gc_history_transplant_ref", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load GC transplant reference from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


_ref = _load_reference()

CanonicalStateKey = _ref.CanonicalStateKey
DecisionPoint = _ref.DecisionPoint
OracleTrace = _ref.OracleTrace
TokenMapper = _ref.TokenMapper

_canonical_state_key = _ref._canonical_state_key
_history_difference_tokens = _ref._history_difference_tokens
_load_checkpoint = _ref._load_checkpoint
_set_seed = _ref._set_seed
generate_oracle_trace_with_random_ties = _ref.generate_oracle_trace_with_random_ties

__all__ = [
    "CanonicalStateKey",
    "DecisionPoint",
    "OracleTrace",
    "TokenMapper",
    "_canonical_state_key",
    "_history_difference_tokens",
    "_load_checkpoint",
    "_set_seed",
    "generate_oracle_trace_with_random_ties",
]
