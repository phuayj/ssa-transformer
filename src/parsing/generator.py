"""Random expression generator for the parsing domain."""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Sequence, Tuple

from parsing.grammar import IDENTIFIERS, NUMBERS

logger = logging.getLogger(__name__)


def _sample_atom(rng: random.Random) -> List[str]:
    if rng.random() < 0.5:
        return [str(rng.choice(IDENTIFIERS))]
    return [str(rng.choice(NUMBERS))]


def _normalized_expr_weights(
    p_call: float,
    p_index: float,
    p_tuple: float,
    p_neg: float,
) -> List[Tuple[str, float]]:
    raw = {
        "call": max(float(p_call), 0.0),
        "index": max(float(p_index), 0.0),
        "tuple": max(float(p_tuple), 0.0),
        "neg": max(float(p_neg), 0.0),
    }
    specified = sum(raw.values())
    if specified < 1.0:
        residual = 1.0 - specified
        raw["paren"] = residual / 2.0
        raw["atom"] = residual / 2.0
    else:
        # Preserve the user's relative preferences while reserving some mass for
        # terminating branches, which keeps generation stable at small depths.
        scale = 0.8 / max(specified, 1e-8)
        for key in list(raw.keys()):
            raw[key] *= scale
        raw["paren"] = 0.1
        raw["atom"] = 0.1

    total = sum(raw.values())
    return [(name, weight / total) for name, weight in raw.items() if weight > 0.0]


def _weighted_choice(rng: random.Random, weights: Sequence[Tuple[str, float]]) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for name, weight in weights:
        cumulative += float(weight)
        if threshold <= cumulative:
            return str(name)
    return str(weights[-1][0])


def _generate_expr_list(
    *,
    rng: random.Random,
    max_depth: int,
    min_items: int,
    max_items: int,
    p_call: float,
    p_index: float,
    p_tuple: float,
    p_neg: float,
) -> List[List[str]]:
    count = int(rng.randint(int(min_items), int(max_items)))
    items: List[List[str]] = []
    for _ in range(count):
        items.append(
            _generate_expr(
                rng=rng,
                max_depth=max_depth,
                p_call=p_call,
                p_index=p_index,
                p_tuple=p_tuple,
                p_neg=p_neg,
            )
        )
    return items


def _generate_expr(
    *,
    rng: random.Random,
    max_depth: int,
    p_call: float,
    p_index: float,
    p_tuple: float,
    p_neg: float,
) -> List[str]:
    if int(max_depth) <= 0:
        return _sample_atom(rng)

    kind = _weighted_choice(
        rng,
        _normalized_expr_weights(
            p_call=float(p_call),
            p_index=float(p_index),
            p_tuple=float(p_tuple),
            p_neg=float(p_neg),
        ),
    )

    if kind == "call":
        callee = str(rng.choice(IDENTIFIERS))
        args = _generate_expr_list(
            rng=rng,
            max_depth=int(max_depth) - 1,
            min_items=1,
            max_items=3,
            p_call=p_call,
            p_index=p_index,
            p_tuple=p_tuple,
            p_neg=p_neg,
        )
        tokens: List[str] = [callee, "("]
        for idx, item in enumerate(args):
            if idx > 0:
                tokens.append(",")
            tokens.extend(item)
        tokens.append(")")
        return tokens

    if kind == "index":
        target = str(rng.choice(IDENTIFIERS))
        inner = _generate_expr(
            rng=rng,
            max_depth=int(max_depth) - 1,
            p_call=p_call,
            p_index=p_index,
            p_tuple=p_tuple,
            p_neg=p_neg,
        )
        return [target, "["] + inner + ["]"]

    if kind == "tuple":
        items = _generate_expr_list(
            rng=rng,
            max_depth=int(max_depth) - 1,
            min_items=2,
            max_items=4,
            p_call=p_call,
            p_index=p_index,
            p_tuple=p_tuple,
            p_neg=p_neg,
        )
        tokens = ["("]
        for idx, item in enumerate(items):
            if idx > 0:
                tokens.append(",")
            tokens.extend(item)
        tokens.append(")")
        return tokens

    if kind == "paren":
        inner = _generate_expr(
            rng=rng,
            max_depth=int(max_depth) - 1,
            p_call=p_call,
            p_index=p_index,
            p_tuple=p_tuple,
            p_neg=p_neg,
        )
        return ["("] + inner + [")"]

    if kind == "neg":
        inner = _generate_expr(
            rng=rng,
            max_depth=int(max_depth) - 1,
            p_call=p_call,
            p_index=p_index,
            p_tuple=p_tuple,
            p_neg=p_neg,
        )
        return ["-"] + inner

    return _sample_atom(rng)


def generate_expression(
    max_depth: int = 4,
    p_call: float = 0.25,
    p_index: float = 0.2,
    p_tuple: float = 0.3,
    p_neg: float = 0.1,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Generate a random expression guaranteed to match the fixed grammar."""

    if int(max_depth) < 0:
        raise ValueError("max_depth must be >= 0")

    local_rng = rng if rng is not None else random.Random()
    tokens = _generate_expr(
        rng=local_rng,
        max_depth=int(max_depth),
        p_call=float(p_call),
        p_index=float(p_index),
        p_tuple=float(p_tuple),
        p_neg=float(p_neg),
    )
    logger.debug(
        "generated_expression len=%d max_depth=%d tokens=%s",
        int(len(tokens)),
        int(max_depth),
        " ".join(tokens),
    )
    return tokens
