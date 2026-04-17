#!/usr/bin/env python3
"""Generate state isolation hierarchy LaTeX table for SAT n=50."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


SEEDS = (42, 123, 456, 789, 2024)


def info(message: str) -> None:
    print(f"[info] {message}", file=sys.stderr)


def warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    exp_pattern: str
    history: str
    local_state: str
    optional: bool = False


@dataclass
class ConditionStats:
    label: str
    history: str
    local_state: str
    solve_mean: Optional[float]
    solve_std: Optional[float]
    decisions_mean: Optional[float]
    timeout_mean: Optional[float]
    n: int


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        key="ssa",
        label="Enriched + SSA",
        exp_pattern="sat-n50-enriched-selective_ssa-seed{seed}",
        history="Blocked",
        local_state="Enriched",
    ),
    ConditionSpec(
        key="enriched_causal",
        label="Enriched + causal",
        exp_pattern="sat-n50-enriched-full_causal-seed{seed}",
        history="Full",
        local_state="Enriched",
    ),
    ConditionSpec(
        key="minimal_causal",
        label="Minimal + causal",
        exp_pattern="sat-n50-minimal-full_causal-seed{seed}",
        history="Full",
        local_state="Minimal",
    ),
    ConditionSpec(
        key="minimal_ssa",
        label="Minimal + SSA",
        exp_pattern="sat-n50-minimal-selective_ssa-seed{seed}",
        history="Blocked",
        local_state="Minimal",
    ),
    ConditionSpec(
        key="state_only",
        label="State-only",
        exp_pattern="sat-n50-state-only-full_causal-seed{seed}",
        history="None",
        local_state="Enriched",
    ),
    ConditionSpec(
        key="state_only_50k",
        label="State-only (50k)",
        exp_pattern="sat-n50-state-only-50k-seed{seed}",
        history="None",
        local_state="Enriched",
        optional=True,
    ),
    ConditionSpec(
        key="state_only_100k",
        label="State-only (100k)",
        exp_pattern="sat-n50-state-only-100k-seed{seed}",
        history="None",
        local_state="Enriched",
        optional=True,
    ),
)


def _mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(values) >= 2 else 0.0
    return mean, std


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def _fmt_solve(mean: Optional[float], std: Optional[float], bold: bool) -> str:
    if mean is None:
        return "$-$"
    base = f"{mean:.1f} \\pm {std:.1f}" if std is not None else f"{mean:.1f}"
    if bold:
        return f"$\\mathbf{{{base}}}$"
    return f"${base}$"


def _fmt_value(value: Optional[float]) -> str:
    if value is None:
        return "$-$"
    return f"${value:.1f}$"


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
        default=Path("paper/manuscript/parts/tables/tab_state_isolation.tex"),
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

    collected: list[ConditionStats] = []

    for spec in CONDITIONS:
        solve_values: list[float] = []
        decisions_values: list[float] = []
        timeout_values: list[float] = []

        for seed in SEEDS:
            result_path = (
                experiments_dir
                / spec.exp_pattern.format(seed=seed)
                / "eval_b4096"
                / "results.json"
            )
            metrics = _load_metrics(result_path)
            if metrics is None:
                continue

            solve_values.append(metrics["solve_rate"])
            decisions_values.append(metrics["mean_decisions"])
            timeout_values.append(metrics["timeout_rate"])
            info(
                "Loaded "
                f"condition={spec.key} seed={seed} "
                f"solve={metrics['solve_rate']:.2f}% "
                f"decisions={metrics['mean_decisions']:.2f} "
                f"timeout={metrics['timeout_rate']:.2f}%"
            )

        if not solve_values:
            message = f"No valid seeds for condition={spec.key}; skipping row"
            if spec.optional:
                info(message)
            else:
                warn(message)
            continue

        solve_mean, solve_std = _mean_std(solve_values)
        decisions_mean = _mean(decisions_values)
        timeout_mean = _mean(timeout_values)
        stats = ConditionStats(
            label=spec.label,
            history=spec.history,
            local_state=spec.local_state,
            solve_mean=solve_mean,
            solve_std=solve_std,
            decisions_mean=decisions_mean,
            timeout_mean=timeout_mean,
            n=len(solve_values),
        )
        collected.append(stats)
        info(
            "Summary "
            f"condition={spec.key} n={stats.n} "
            f"solve_mean={stats.solve_mean if stats.solve_mean is not None else float('nan'):.3f} "
            f"solve_std={stats.solve_std if stats.solve_std is not None else float('nan'):.3f} "
            f"decisions_mean={stats.decisions_mean if stats.decisions_mean is not None else float('nan'):.3f} "
            f"timeout_mean={stats.timeout_mean if stats.timeout_mean is not None else float('nan'):.3f}"
        )

    collected.sort(
        key=lambda row: row.solve_mean if row.solve_mean is not None else float("-inf"),
        reverse=True,
    )

    best_solve = max(
        (row.solve_mean for row in collected if row.solve_mean is not None),
        default=None,
    )

    lines = [
        r"\begin{tabular}{l ccccc}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{History} & \textbf{Local state} & \textbf{Solve (\%)} & \textbf{Decisions} & \textbf{Timeout (\%)} \\",
        r"\midrule",
    ]

    for row in collected:
        is_best = (
            best_solve is not None
            and row.solve_mean is not None
            and abs(row.solve_mean - best_solve) < 1e-9
        )
        lines.append(
            f"{row.label} & {row.history} & {row.local_state} "
            f"& {_fmt_solve(row.solve_mean, row.solve_std, is_best)} "
            f"& {_fmt_value(row.decisions_mean)} "
            f"& {_fmt_value(row.timeout_mean)} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    table = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")
    info(f"Wrote table to {output_path}")


if __name__ == "__main__":
    main()
