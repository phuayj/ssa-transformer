#!/usr/bin/env python3
"""Generate the configuration legend (tab_experiment_map.tex).

This table is a reference legend that points each cited SSA configuration
to the experimental source where the value is reported. Configurations
share most settings (enriched trace, selective SSA mask, 32 slots, 5k
cumulative training, cumulative inference) but differ on a few axes that
matter when comparing solve rates: SAT problem scale (n=30 versus n=50),
number of training seeds, attention mask, slot count, and inference
protocol. Two settings are constant across every row -- the trace format
(Enriched) and the training corpus (Cum. 5k) -- and live in the caption
rather than as columns.

The script reads solve-rate values directly from the corresponding
experiment result JSONs where one exists, falling back to the canonical
value reported in the source table when the values are aggregated by
their own generation script (flagship, slot-mask factorial,
trace x mask factorial, multi-instance pooled).

Layout note: configurations are placed across columns and attributes down
rows. Earlier versions of this table laid configurations along rows with
ten columns of attributes; the resulting natural width forced an
aggressive \\resizebox shrink that made the body font too small to read.
The transposed layout has shorter cells in each column and accommodates a
larger body font.

Usage:
    python scripts/tables/gen_tab_experiment_map.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "tables" / "tab_experiment_map.tex"


@dataclass(frozen=True)
class Entry:
    label: str
    short_label: str  # multi-line column header for the transposed layout
    mask: str
    slots: str
    inference: str
    scale: str
    seeds: str
    source: str
    # Either a canonical value sourced from an aggregated table, or a list
    # of per-seed result.json paths to aggregate.
    value: Optional[str] = None
    seed_results: Optional[tuple[Path, ...]] = None


def _normalize_percent(raw: object) -> Optional[float]:
    if raw is None:
        return None
    if not isinstance(raw, (int, float, str)):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if 0.0 <= v <= 1.0:
        return 100.0 * v
    return v


def _solve_rate(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        direct = _normalize_percent(payload.get("solve_rate"))
        if direct is not None:
            return direct
        results = payload.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            sr = _normalize_percent(results[0].get("solve_rate"))
            if sr is not None:
                return sr
    return None


def _aggregate(seed_paths: tuple[Path, ...]) -> Optional[str]:
    values: list[float] = []
    for path in seed_paths:
        sr = _solve_rate(path)
        if sr is not None:
            values.append(sr)
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size >= 2:
        std = float(np.std(arr, ddof=1))
        return f"${mean:.1f} \\pm {std:.1f}$"
    return f"${mean:.1f}$"


# Per-seed result paths for entries we re-aggregate.
SAT_N50_SSA_DEFAULT = tuple(
    REPO_ROOT / "experiments" / f"sat-n50-enriched-selective_ssa-seed{seed}" / "eval_b4096" / "results.json"
    for seed in (42, 123, 456, 789, 2024)
)
SAT_N30_FACTORIAL_SELECTIVE = tuple(
    REPO_ROOT / "experiments" / f"factorial-eval-sat-enriched-selective-seed{seed}" / "results.json"
    for seed in (42, 123, 456, 789, 2024)
)
SAT_N30_FACTORIAL_BLANKET = tuple(
    REPO_ROOT / "experiments" / f"factorial-eval-sat-enriched-blanket-seed{seed}" / "results.json"
    for seed in (42, 123, 456, 789, 2024)
)
SAT_N30_CROSSMASK = (
    REPO_ROOT / "experiments" / "e3-eval-sat-selective-ssa-seed42" / "results.json",
)
SAT_N50_NO_SLOTS = tuple(
    REPO_ROOT / "experiments" / f"slot-ablation-sat-n50-nslots0-selective_ssa-seed{seed}" / "results.json"
    for seed in (42, 123, 456, 789, 2024)
)


# Canonical fallback values for entries that come from aggregated tables
# whose own generation script is the source of truth (flagship, slot-mask
# factorial, multi-instance pooled). Keeping a fallback avoids cross-script
# fragility when a result file moves; the script prefers per-seed
# aggregation when files are present.
ENTRIES: tuple[Entry, ...] = (
    Entry(
        label="Default",
        short_label="\\makecell{Default\\\\(Cum.)}",
        mask="Sel.",
        slots="$32$",
        inference="Cum.",
        scale="$n{=}50$",
        seeds="$5$",
        source="Tab.~\\ref{tab:protocol-bridge}A",
        value="$28.9 \\pm 1.9$",
        seed_results=SAT_N50_SSA_DEFAULT,
    ),
    Entry(
        label="Default state-rebuilt",
        short_label="\\makecell{Default\\\\(SR)}",
        mask="Sel.",
        slots="$32$",
        inference="SR",
        scale="$n{=}50$",
        seeds="$5$",
        source="Tab.~\\ref{tab:protocol-bridge}A",
        value="$93.4 \\pm 2.3$",
    ),
    Entry(
        label="\\quad + random variable",
        short_label="\\makecell{Default\\\\(SR+rand)}",
        mask="Sel.",
        slots="$32$",
        inference="SR+rand",
        scale="$n{=}50$",
        seeds="$5$",
        source="Tab.~\\ref{tab:protocol-bridge}A",
        value="$89.5 \\pm 1.1$",
    ),
    Entry(
        label="\\quad + dead-end oracle",
        short_label="\\makecell{Default\\\\(SR+oracle)}",
        mask="Sel.",
        slots="$32$",
        inference="SR+oracle",
        scale="$n{=}50$",
        seeds="$5$",
        source="Tab.~\\ref{tab:protocol-bridge}D",
        value="$100.0 \\pm 0.0$",
    ),
    Entry(
        label="No slots",
        short_label="\\makecell{No\\\\slots}",
        mask="Sel.",
        slots="$0$",
        inference="Cum.",
        scale="$n{=}50$",
        seeds="$5$",
        source="Tab.~\\ref{tab:slot-mask-factorial}",
        value="$34.1 \\pm 2.0$",
        seed_results=SAT_N50_NO_SLOTS,
    ),
    Entry(
        label="Blanket mask",
        short_label="\\makecell{Blanket\\\\mask}",
        mask="Blanket",
        slots="$32$",
        inference="Cum.",
        scale="$n{=}30$",
        seeds="$5$",
        source="Tab.~\\ref{tab:factorial}",
        value="$64.8 \\pm 2.0$",
        seed_results=SAT_N30_FACTORIAL_BLANKET,
    ),
    Entry(
        label="Selective mask (factorial)",
        short_label="\\makecell{Selective\\\\(factorial)}",
        mask="Sel.",
        slots="$32$",
        inference="Cum.",
        scale="$n{=}30$",
        seeds="$5$",
        source="Tab.~\\ref{tab:factorial}",
        value="$61.9 \\pm 3.1$",
        seed_results=SAT_N30_FACTORIAL_SELECTIVE,
    ),
    Entry(
        label="Multi-instance pooled",
        short_label="\\makecell{Multi-\\\\instance}",
        mask="Sel.",
        slots="$32$",
        inference="Cum.",
        scale="$n{=}50$",
        seeds="$5\\times 4$",
        source="Tab.~\\ref{tab:multi-instance-seed}",
        value="$35.9 \\pm 2.7$",
    ),
    Entry(
        label="Cross-mask matched",
        short_label="\\makecell{Cross-\\\\mask}",
        mask="Sel.",
        slots="$32$",
        inference="Cum.",
        scale="$n{=}30$",
        seeds="$1$",
        source="Tab.~\\ref{tab:crossmask}",
        value="$59.0$",
        seed_results=SAT_N30_CROSSMASK,
    ),
)


def _resolve_value(entry: Entry) -> str:
    if entry.seed_results is not None:
        recomputed = _aggregate(entry.seed_results)
        if recomputed is not None:
            return recomputed
        assert entry.value is not None, f"missing fallback value for {entry.label}"
        return entry.value
    assert entry.value is not None, f"missing canonical value for {entry.label}"
    return entry.value


def render_table() -> str:
    # Transposed layout: configurations across columns, attributes down rows.
    column_spec = "l " + " ".join(["c"] * len(ENTRIES))

    header_cells = ["\\textbf{Attribute}"] + [entry.short_label for entry in ENTRIES]

    attribute_rows: list[tuple[str, list[str]]] = [
        ("\\textbf{Mask}", [entry.mask for entry in ENTRIES]),
        ("\\textbf{Slots}", [entry.slots for entry in ENTRIES]),
        ("\\textbf{Inference}", [entry.inference for entry in ENTRIES]),
        ("\\textbf{Scale}", [entry.scale for entry in ENTRIES]),
        ("\\textbf{Seeds}", [entry.seeds for entry in ENTRIES]),
        ("\\textbf{Source}", [entry.source for entry in ENTRIES]),
        ("\\textbf{Solve (\\%)}", [_resolve_value(entry) for entry in ENTRIES]),
    ]

    lines = [
        "% Auto-generated configuration legend.",
        "% Do not edit by hand; rerun the generator to regenerate.",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]
    for attribute_label, cells in attribute_rows:
        row = " & ".join([attribute_label, *cells]) + " \\\\"
        lines.append(row)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    content = render_table()
    if args.stdout:
        print(content, end="")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
