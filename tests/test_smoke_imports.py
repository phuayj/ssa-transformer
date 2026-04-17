"""Smoke tests for public-release top-level packages."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "package_name",
    [
        "universal",
        "graph_coloring",
        "sat",
        "blocks_world",
        "parsing",
        "rk_benchmark",
        "baselines",
    ],
)
def test_top_level_package_imports(package_name: str) -> None:
    module = importlib.import_module(package_name)

    assert module is not None
