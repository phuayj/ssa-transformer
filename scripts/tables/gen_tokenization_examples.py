#!/usr/bin/env python3
"""Generate concrete tokenization examples for the appendix.

Uses the actual SAT and graph-coloring tokenizers from src/sat and src/universal
to render small, fully worked traces. The output strings are pasted (or
\\input'ed via a generated .tex file) into the manuscript appendix so the
illustration stays in sync with the codebase.

Run:
    python scripts/tables/gen_tokenization_examples.py

Writes a single LaTeX fragment to
    output/tables/tab_tokenization_examples.tex
that ext_appendix.tex can \\input.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_module(rel_path: str, mod_name: str):
    """Import a module by file path, bypassing the package __init__.

    The package __init__ pulls in torch, which is not always available where
    this script is run.
    """
    spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {rel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wrap_for_latex(text: str, width: int = 78) -> str:
    """Wrap a long token-sequence string so LaTeX verbatim does not overflow."""
    out: List[str] = []
    line = ""
    for tok in text.split():
        if len(line) + len(tok) + 1 > width:
            out.append(line.rstrip())
            line = tok + " "
        else:
            line += tok + " "
    if line.strip():
        out.append(line.rstrip())
    return "\n".join(out)


def _split_blocks(text: str, block_starts: List[str]) -> List[List[str]]:
    """Split a token stream into blocks at the listed block-start markers.

    Returns a list of token lists. The first list contains everything before
    the first start marker (the problem prefix); subsequent lists each begin
    with one of the block-start markers.
    """
    tokens = text.split()
    blocks: List[List[str]] = [[]]
    for tok in tokens:
        if tok in block_starts and blocks[-1]:
            blocks.append([])
        blocks[-1].append(tok)
    return blocks


def _format_block_listing(text: str, block_starts: List[str], width: int = 76) -> str:
    """Format a token stream as a per-block listing with section comments.

    The first block is labeled as the problem prefix; subsequent blocks
    are labeled by their index $B_t$. The output is suitable for LaTeX
    verbatim.
    """
    blocks = _split_blocks(text, block_starts)
    # First block is the problem prefix; rest are decision blocks.
    rendered: List[str] = []
    rendered.append("# Problem prefix P")
    rendered.append(_wrap_for_latex(" ".join(blocks[0]), width=width))
    for idx, block in enumerate(blocks[1:], start=1):
        rendered.append("")
        rendered.append(f"# Decision block B{idx}")
        rendered.append(_wrap_for_latex(" ".join(block), width=width))
    return "\n".join(rendered)


def render_sat_solved() -> str:
    """A short SAT trace that solves a 3-variable, 4-clause instance."""
    sat_mod = _import_module("src/sat/interleaved_tokenizer.py", "sat_it_tk")
    tk = sat_mod.SATInterleavedTokenizer()

    # (v1 v ~v2 v v3) ^ (~v1 v v2 v v3) ^ (v1 v v2 v ~v3) ^ (~v1 v ~v2 v v3)
    clauses = [(1, -2, 3), (-1, 2, 3), (1, 2, -3), (-1, -2, 3)]
    num_vars = 3

    events = [
        {
            "type": "assign",
            "var": 0,
            "value": 1,
            "sorted_candidates": [0, 1, 2],
            "current_assignment": np.array([0, 0, 0]),
        },
        {
            "type": "assign",
            "var": 1,
            "value": 1,
            "sorted_candidates": [1, 2],
            "current_assignment": np.array([1, 0, 0]),
        },
        {
            "type": "assign",
            "var": 2,
            "value": 1,
            "sorted_candidates": [2],
            "current_assignment": np.array([1, 1, 0]),
        },
        {"type": "solved"},
    ]

    trace = tk.build_interleaved_trace(clauses, events, num_vars)
    return tk.decode_sequence(trace)


def render_sat_conflict() -> str:
    """A short SAT trace that exposes a conflict and emits a backtrack token."""
    sat_mod = _import_module("src/sat/interleaved_tokenizer.py", "sat_it_tk")
    tk = sat_mod.SATInterleavedTokenizer()

    # Six clauses over 3 vars; assigning v3=F and then v1=F makes C5 = (-v1 v v2 v -v3)
    # have one unassigned literal on a clause that becomes UNIT under propagation.
    clauses = [
        (1, 2, 3),
        (1, 2, -3),
        (1, -2, 3),
        (1, -2, -3),
        (-1, 2, 3),
        (-1, 2, -3),
    ]
    num_vars = 3
    events = [
        {
            "type": "assign",
            "var": 2,
            "value": -1,
            "sorted_candidates": [0, 1, 2],
            "current_assignment": np.array([0, 0, 0]),
        },
        {
            "type": "assign",
            "var": 0,
            "value": -1,
            "sorted_candidates": [0, 1],
            "current_assignment": np.array([0, 0, -1]),
        },
        {
            "type": "conflict",
            "clause_id": 5,
            "backjump_level": 0,
            "sorted_candidates": [0, 1],
            "current_assignment": np.array([-1, 0, -1]),
        },
    ]

    trace = tk.build_interleaved_trace(clauses, events, num_vars)
    return tk.decode_sequence(trace)


def render_gc_example() -> str:
    """A short graph-coloring trace using the sorted-slim format."""
    gc_mod = _import_module("src/universal/cdcl_tokenizer.py", "gc_cdcl_tk")
    tk = gc_mod.CDCLTokenizer()

    # 4-cycle, 4 colors.
    adj = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    num_nodes = 4
    num_colors = 4

    events = [
        {
            "type": "assign",
            "node": 0,
            "color": 1,
            "sorted_candidates": [0, 1, 2, 3],
            "domain_sizes": {0: 4, 1: 4, 2: 4, 3: 4},
            "domain": {1, 2, 3, 4},
        },
        {
            "type": "assign",
            "node": 1,
            "color": 2,
            "sorted_candidates": [1, 2, 3],
            "domain_sizes": {1: 3, 2: 4, 3: 3},
            "domain": {2, 3, 4},
        },
        {"type": "solved"},
    ]
    trace = tk.build_sorted_slim_trace(adj, events, num_nodes, num_colors)
    return tk.decode_sequence(trace)


def main() -> None:
    sat_solved = render_sat_solved()
    sat_conflict = render_sat_conflict()
    gc_solved = render_gc_example()

    logger.info("SAT (solved): %d tokens", len(sat_solved.split()))
    logger.info("SAT (conflict): %d tokens", len(sat_conflict.split()))
    logger.info("GC (solved): %d tokens", len(gc_solved.split()))

    # Sanity: the number of decision blocks must match the number of search events.
    # We verify this by counting STATE markers (which open each block).
    assert sat_solved.count("STATE") == 3, "expected 3 STATE blocks in SAT-solved"
    assert sat_conflict.count("STATE") == 3, "expected 3 STATE blocks in SAT-conflict"
    assert gc_solved.count("STATE") == 2, "expected 2 STATE blocks in GC-solved"

    out_path = REPO_ROOT / "output" / "tables" / "tab_tokenization_examples.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # SAT block boundaries open at STATE; GC also uses STATE in sorted-slim mode.
    sat_blocks = _format_block_listing(sat_solved, block_starts=["STATE"])
    sat_conflict_blocks = _format_block_listing(sat_conflict, block_starts=["STATE"])
    gc_blocks = _format_block_listing(gc_solved, block_starts=["STATE"])

    body: List[str] = []
    body.append("% Auto-generated tokenization examples.")
    body.append("% Do not edit by hand; rerun the generator to regenerate.")
    body.append("")
    body.append("\\paragraph{SAT, satisfying trace ($n=3$, $4$ clauses).}")
    body.append("\\begin{footnotesize}")
    body.append("\\begin{verbatim}")
    body.append(sat_blocks)
    body.append("\\end{verbatim}")
    body.append("\\end{footnotesize}")
    body.append("")
    body.append("\\paragraph{SAT, trace ending in a conflict ($n=3$, $6$ clauses).}")
    body.append("\\begin{footnotesize}")
    body.append("\\begin{verbatim}")
    body.append(sat_conflict_blocks)
    body.append("\\end{verbatim}")
    body.append("\\end{footnotesize}")
    body.append("")
    body.append("\\paragraph{Graph coloring, satisfying trace ($n=4$, $4$-cycle).}")
    body.append("\\begin{footnotesize}")
    body.append("\\begin{verbatim}")
    body.append(gc_blocks)
    body.append("\\end{verbatim}")
    body.append("\\end{footnotesize}")
    body.append("")

    out_path.write_text("\n".join(body))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
