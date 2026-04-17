#!/usr/bin/env python3
"""Generate the slot * mask factorial table for the MLJ paper.

Produces output/tables/tab_slot_mask_factorial.tex.

The factorial crosses the SSA mask against the causal mask with the slot
register count set to either 32 (default) or 0. Numbers are aggregated
from per-seed evaluation results.json files under REPO_ROOT/experiments.

Source experiment directories
-----------------------------
SAT n=50, budget 4096, enriched traces:
  Mask=selective_ssa, n_slots=32:
      sat-n50-enriched-selective_ssa-seed{42,123,456,789,2024}/eval_b4096
  Mask=full_causal, n_slots=32:
      sat-n50-enriched-full_causal-seed{42,123,456,789,2024}/eval_b4096
  Mask=selective_ssa, n_slots=0:
      slot-ablation-sat-n50-nslots0-selective_ssa-seed{42,123,456,789,2024}/eval_b4096
  Mask=full_causal, n_slots=0:
      slot-ablation-sat-n50-nslots0-full_causal-seed{42,123,456,789,2024}/eval_b4096

GC n=30, budget 2048:
  Mask=selective_ssa, n_slots=32:
      e3-eval-gc-selective-ssa-seed{42,123,456,789,2024}
  Mask=full_causal, n_slots=32:
      e3-eval-gc-full-causal-seed{42,123,456,789,2024}
  Mask=selective_ssa, n_slots=0:
      slot-ablation-gc-nslots0-selective_ssa-seed{42,123,456,789,2024}/eval_b2048
  Mask=full_causal, n_slots=0:
      slot-ablation-gc-nslots0-full_causal-seed{42,123,456,789,2024}/eval_b2048

Bolding policy
--------------
The two cells corresponding to "no slots" under SSA are bolded because the
table's narrative claim is that SSA without slots performs at least as well
as SSA with slots. We do not bold the largest cell column-wise to avoid
implying that the cells are directly comparable across columns.

Usage:
    python scripts/tables/gen_tab_slot_mask_factorial.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = REPO_ROOT / "experiments"

DEFAULT_OUTPUT = REPO_ROOT / "output" / "tables" / "tab_slot_mask_factorial.tex"

SEEDS = (42, 123, 456, 789, 2024)


@dataclass
class CellSource:
    domain: str  # "sat" or "gc"
    mask: str  # "ssa" or "causal"
    slots: int  # 0 or 32
    paths: list[Path]  # one per seed; results.json or its parent


def collect_cells() -> list[CellSource]:
    cells: list[CellSource] = []

    # --- SAT n=50, budget 4096 ---
    sat_n32_ssa = [
        EXPERIMENTS
        / f"sat-n50-enriched-selective_ssa-seed{seed}"
        / "eval_b4096"
        / "results.json"
        for seed in SEEDS
    ]
    sat_n32_causal = [
        EXPERIMENTS
        / f"sat-n50-enriched-full_causal-seed{seed}"
        / "eval_b4096"
        / "results.json"
        for seed in SEEDS
    ]
    sat_n0_ssa = [
        EXPERIMENTS
        / f"slot-ablation-sat-n50-nslots0-selective_ssa-seed{seed}"
        / "eval_b4096"
        / "results.json"
        for seed in SEEDS
    ]
    sat_n0_causal = [
        EXPERIMENTS
        / f"slot-ablation-sat-n50-nslots0-full_causal-seed{seed}"
        / "eval_b4096"
        / "results.json"
        for seed in SEEDS
    ]

    cells.append(CellSource("sat", "ssa", 32, sat_n32_ssa))
    cells.append(CellSource("sat", "causal", 32, sat_n32_causal))
    cells.append(CellSource("sat", "ssa", 0, sat_n0_ssa))
    cells.append(CellSource("sat", "causal", 0, sat_n0_causal))

    # --- GC n=30, budget 2048 ---
    gc_n32_ssa = [
        EXPERIMENTS / f"e3-eval-gc-selective-ssa-seed{seed}" / "results.json"
        for seed in SEEDS
    ]
    gc_n32_causal = [
        EXPERIMENTS / f"e3-eval-gc-full-causal-seed{seed}" / "results.json"
        for seed in SEEDS
    ]
    gc_n0_ssa = [
        EXPERIMENTS
        / f"slot-ablation-gc-nslots0-selective_ssa-seed{seed}"
        / "eval_b2048"
        / "results.json"
        for seed in SEEDS
    ]
    gc_n0_causal = [
        EXPERIMENTS
        / f"slot-ablation-gc-nslots0-full_causal-seed{seed}"
        / "eval_b2048"
        / "results.json"
        for seed in SEEDS
    ]

    cells.append(CellSource("gc", "ssa", 32, gc_n32_ssa))
    cells.append(CellSource("gc", "causal", 32, gc_n32_causal))
    cells.append(CellSource("gc", "ssa", 0, gc_n0_ssa))
    cells.append(CellSource("gc", "causal", 0, gc_n0_causal))

    return cells


def load_solve_rates(paths: Iterable[Path]) -> list[float]:
    rates: list[float] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing results file: {p}")
        with p.open() as f:
            data = json.load(f)
        results = data.get("results", [data])
        for r in results:
            if "solve_rate" in r:
                rates.append(float(r["solve_rate"]))
                break
        else:
            raise KeyError(f"No solve_rate in {p}")
    return rates


def fmt_cell(rates: list[float]) -> str:
    mean = statistics.mean(rates) * 100
    if len(rates) > 1:
        std = statistics.stdev(rates) * 100
    else:
        std = 0.0
    return f"${mean:.1f} \\pm {std:.1f}$"


def fmt_pp(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"${sign}{abs(value):.1f}$"


def render_table(cells: list[CellSource]) -> str:
    indexed: dict[tuple[str, str, int], list[float]] = {}
    for c in cells:
        indexed[(c.domain, c.mask, c.slots)] = load_solve_rates(c.paths)

    sat_ssa_32 = indexed[("sat", "ssa", 32)]
    sat_ssa_0 = indexed[("sat", "ssa", 0)]
    sat_caus_32 = indexed[("sat", "causal", 32)]
    sat_caus_0 = indexed[("sat", "causal", 0)]
    gc_ssa_32 = indexed[("gc", "ssa", 32)]
    gc_ssa_0 = indexed[("gc", "ssa", 0)]
    gc_caus_32 = indexed[("gc", "causal", 32)]
    gc_caus_0 = indexed[("gc", "causal", 0)]

    def m(rates: list[float]) -> float:
        return statistics.mean(rates) * 100

    sat_slot_ssa = m(sat_ssa_0) - m(sat_ssa_32)
    sat_slot_caus = m(sat_caus_0) - m(sat_caus_32)
    gc_slot_ssa = m(gc_ssa_0) - m(gc_ssa_32)
    gc_slot_caus = m(gc_caus_0) - m(gc_caus_32)
    sat_mask_32 = m(sat_ssa_32) - m(sat_caus_32)
    sat_mask_0 = m(sat_ssa_0) - m(sat_caus_0)
    gc_mask_32 = m(gc_ssa_32) - m(gc_caus_32)
    gc_mask_0 = m(gc_ssa_0) - m(gc_caus_0)

    bold_open = "\\boldsymbol{"
    bold_close = "}"

    def cell_bold(rates: list[float]) -> str:
        mean = statistics.mean(rates) * 100
        std = statistics.stdev(rates) * 100 if len(rates) > 1 else 0.0
        return f"$\\mathbf{{{mean:.1f} \\pm {std:.1f}}}$"

    lines: list[str] = []
    lines.append("% Generated by scripts/tables/gen_tab_slot_mask_factorial.py")
    lines.append("\\begin{tabular}{l c c c c}")
    lines.append("\\toprule")
    lines.append(
        " & \\multicolumn{2}{c}{\\textbf{SAT $n{=}50$}} "
        "& \\multicolumn{2}{c}{\\textbf{GC $n{=}30$}} \\\\"
    )
    lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    lines.append(
        "\\textbf{Mask} & \\textbf{$32$ slots} & \\textbf{$0$ slots} "
        "& \\textbf{$32$ slots} & \\textbf{$0$ slots} \\\\"
    )
    lines.append("\\midrule")
    lines.append(
        "SSA & "
        + fmt_cell(sat_ssa_32)
        + " & "
        + cell_bold(sat_ssa_0)
        + " & "
        + fmt_cell(gc_ssa_32)
        + " & "
        + cell_bold(gc_ssa_0)
        + " \\\\"
    )
    lines.append(
        "Causal & "
        + fmt_cell(sat_caus_32)
        + " & "
        + fmt_cell(sat_caus_0)
        + " & "
        + fmt_cell(gc_caus_32)
        + " & "
        + fmt_cell(gc_caus_0)
        + " \\\\"
    )
    lines.append("\\midrule")
    lines.append(
        "Slot effect (SSA) & \\multicolumn{2}{c}{"
        + fmt_pp(sat_slot_ssa)
        + " pp} & \\multicolumn{2}{c}{"
        + fmt_pp(gc_slot_ssa)
        + " pp} \\\\"
    )
    lines.append(
        "Slot effect (Causal) & \\multicolumn{2}{c}{"
        + fmt_pp(sat_slot_caus)
        + " pp} & \\multicolumn{2}{c}{"
        + fmt_pp(gc_slot_caus)
        + " pp} \\\\"
    )
    lines.append(
        "Mask effect & "
        + fmt_pp(sat_mask_32)
        + " pp & "
        + fmt_pp(sat_mask_0)
        + " pp & "
        + fmt_pp(gc_mask_32)
        + " pp & "
        + fmt_pp(gc_mask_0)
        + " pp \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output path for the LaTeX tabular (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the table to stdout instead of writing to disk.",
    )
    args = parser.parse_args()

    cells = collect_cells()
    content = render_table(cells)

    if args.stdout:
        print(content, end="")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
