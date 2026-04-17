#!/usr/bin/env python3
"""Generate LaTeX factorial table for SAT n=50 experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import ttest_ind


SEEDS = (42, 123, 456, 789, 2024)
TRACES = ("enriched", "minimal")
MASKS = ("full_causal", "selective_ssa")


def info(message: str) -> None:
    print(f"[info] {message}", file=sys.stderr)


def warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def _mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(values) >= 2 else 0.0
    return mean, std


def _load_metrics(path: Path) -> Optional[dict[str, float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warn(f"Missing result file: {path}")
        return None
    except json.JSONDecodeError as exc:
        warn(f"Invalid JSON at {path}: {exc}")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        warn(f"Failed reading {path}: {exc}")
        return None

    try:
        entry = payload["results"][0]
        solve_rate = float(entry["solve_rate"]) * 100.0
        mean_decisions = float(entry["mean_decisions"])
        timeout_rate = float(entry["timeout_rate"]) * 100.0
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        warn(f"Missing/invalid expected metrics in {path}: {exc}")
        return None

    return {
        "solve_rate": solve_rate,
        "mean_decisions": mean_decisions,
        "timeout_rate": timeout_rate,
    }


def _fmt_mean_std(
    mean: Optional[float], std: Optional[float], bold: bool = False
) -> str:
    if mean is None:
        return "$-$"
    body = f"{mean:.1f}"
    if std is not None:
        body = f"{body} \\pm {std:.1f}"
    if bold:
        return f"$\\mathbf{{{body}}}$"
    return f"${body}$"


def _fmt_signed(value: Optional[float]) -> str:
    if value is None:
        return "$-$"
    return f"${value:+.1f}$"


def _welch_p(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    test = ttest_ind(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        equal_var=False,
    )
    p_value = float(test.pvalue)
    if math.isnan(p_value):
        return None
    return p_value


def _fmt_delta_with_p(delta: Optional[float], p_value: Optional[float]) -> str:
    if delta is None:
        return "$-$"
    delta_txt = f"{delta:+.1f}"
    if p_value is None:
        return f"${delta_txt}$ ($p = \\mathrm{{NA}}$)"
    return f"${delta_txt}$ ($p = {p_value:.3f}$)"


def _safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory root (default: experiments)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/manuscript/parts/tables/tab_n50_factorial.tex"),
        help="Output LaTeX path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    experiments_dir = args.experiments_dir
    if not experiments_dir.is_absolute():
        experiments_dir = (repo_root / experiments_dir).resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()

    info(f"experiments_dir={experiments_dir}")
    info(f"output_path={output_path}")

    solve_per_seed: dict[str, dict[str, dict[int, float]]] = {
        trace: {mask: {} for mask in MASKS} for trace in TRACES
    }

    for trace in TRACES:
        for mask in MASKS:
            for seed in SEEDS:
                result_path = (
                    experiments_dir
                    / f"sat-n50-{trace}-{mask}-seed{seed}"
                    / "eval_b4096"
                    / "results.json"
                )
                metrics = _load_metrics(result_path)
                if metrics is None:
                    continue
                solve_per_seed[trace][mask][seed] = metrics["solve_rate"]
                info(
                    "Loaded "
                    f"trace={trace} mask={mask} seed={seed} "
                    f"solve={metrics['solve_rate']:.2f}% "
                    f"decisions={metrics['mean_decisions']:.2f} "
                    f"timeout={metrics['timeout_rate']:.2f}%"
                )

    stats_by_cell: dict[str, dict[str, dict[str, Optional[float]]]] = {
        trace: {mask: {"mean": None, "std": None} for mask in MASKS} for trace in TRACES
    }

    for trace in TRACES:
        for mask in MASKS:
            values = list(solve_per_seed[trace][mask].values())
            mean, std = _mean_std(values)
            stats_by_cell[trace][mask]["mean"] = mean
            stats_by_cell[trace][mask]["std"] = std
            info(
                "Summary "
                f"trace={trace} mask={mask} n={len(values)} "
                f"mean={mean if mean is not None else float('nan'):.3f} "
                f"std={std if std is not None else float('nan'):.3f}"
            )

    row_deltas: dict[str, Optional[float]] = {}
    row_p_values: dict[str, Optional[float]] = {}
    for trace in TRACES:
        causal = list(solve_per_seed[trace]["full_causal"].values())
        ssa = list(solve_per_seed[trace]["selective_ssa"].values())
        row_deltas[trace] = _safe_sub(
            stats_by_cell[trace]["selective_ssa"]["mean"],
            stats_by_cell[trace]["full_causal"]["mean"],
        )
        row_p_values[trace] = _welch_p(ssa, causal)
        info(
            f"SSA-vs-causal trace={trace} "
            f"delta={row_deltas[trace]} p={row_p_values[trace]} "
            f"n_ssa={len(ssa)} n_causal={len(causal)}"
        )

    trace_effect_causal = _safe_sub(
        stats_by_cell["enriched"]["full_causal"]["mean"],
        stats_by_cell["minimal"]["full_causal"]["mean"],
    )
    trace_effect_ssa = _safe_sub(
        stats_by_cell["enriched"]["selective_ssa"]["mean"],
        stats_by_cell["minimal"]["selective_ssa"]["mean"],
    )
    interaction = _safe_sub(trace_effect_ssa, trace_effect_causal)

    info(
        "Trace effects "
        f"causal={trace_effect_causal} ssa={trace_effect_ssa} interaction={interaction}"
    )

    def row_cells(trace: str) -> tuple[str, str]:
        causal_mean = stats_by_cell[trace]["full_causal"]["mean"]
        causal_std = stats_by_cell[trace]["full_causal"]["std"]
        ssa_mean = stats_by_cell[trace]["selective_ssa"]["mean"]
        ssa_std = stats_by_cell[trace]["selective_ssa"]["std"]
        causal_bold = False
        ssa_bold = False
        if causal_mean is not None and ssa_mean is not None:
            if ssa_mean > causal_mean:
                ssa_bold = True
            elif causal_mean > ssa_mean:
                causal_bold = True
        causal_cell = _fmt_mean_std(causal_mean, causal_std, bold=causal_bold)
        ssa_cell = _fmt_mean_std(ssa_mean, ssa_std, bold=ssa_bold)
        return causal_cell, ssa_cell

    enriched_causal, enriched_ssa = row_cells("enriched")
    minimal_causal, minimal_ssa = row_cells("minimal")

    lines = [
        r"\begin{tabular}{l cc c}",
        r"\toprule",
        r"& \textbf{Causal} & \textbf{SSA} & \textbf{$\Delta$ (SSA effect)} \\",
        r"\midrule",
        f"Enriched & {enriched_causal} & {enriched_ssa} & {_fmt_delta_with_p(row_deltas['enriched'], row_p_values['enriched'])} \\\\",
        f"Minimal & {minimal_causal} & {minimal_ssa} & {_fmt_delta_with_p(row_deltas['minimal'], row_p_values['minimal'])} \\\\",
        r"\midrule",
        "Trace effect "
        f"& {_fmt_signed(trace_effect_causal)} "
        f"& {_fmt_signed(trace_effect_ssa)} "
        f"& Interaction: {_fmt_signed(interaction)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    table = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")
    info(f"Wrote table to {output_path}")


if __name__ == "__main__":
    main()
