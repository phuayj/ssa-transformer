"""r^k mechanistic benchmark package."""

from .dataset import RkBenchmarkDataset, rk_collate_fn
from .generator import (
    PAD_TOKEN,
    RkBenchmarkConfig,
    extract_oracle_features_from_tokens,
    generate_example,
    generate_example_with_metadata,
)
from .models import RkOracleMLP, RkTransformer

__all__ = [
    "PAD_TOKEN",
    "RkBenchmarkConfig",
    "RkBenchmarkDataset",
    "RkOracleMLP",
    "RkTransformer",
    "extract_oracle_features_from_tokens",
    "generate_example",
    "generate_example_with_metadata",
    "rk_collate_fn",
]
