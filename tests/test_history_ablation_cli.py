"""Smoke tests for history ablation CLI choices."""

from __future__ import annotations

import importlib


def _choices_for(parser, option: str) -> set[str]:
    for action in parser._actions:
        if option in action.option_strings:
            return set(action.choices or [])
    raise AssertionError(f"missing argparse option: {option}")


def test_history_ablation_parser_exposes_required_choices() -> None:
    module = importlib.import_module("scripts.training.train_history_ablation")
    parser = module._build_argparser()

    assert "local_block_only" in _choices_for(parser, "--mask_mode")
    assert "history_transplant" in _choices_for(parser, "--history_mode")
