#!/usr/bin/env python3
"""Generate LaTeX tables for history-removal and n=75 scaling experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence


LOGGER = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "paper" / "manuscript" / "parts" / "tables"


@dataclass(frozen=True)
class HistoryCondition:
    key: str
    label: str
    mask: str
    cumulative: tuple[float, ...]
    context_clearing: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Stats:
    mean_pct: float
    std_pct: float


HISTORY_CONDITIONS: tuple[HistoryCondition, ...] = (
    HistoryCondition(
        key="full-ssa",
        label="SSA",
        mask="SSA",
        cumulative=(0.335, 0.315, 0.295),
        context_clearing=(0.925, 0.960, 0.935),
    ),
    HistoryCondition(
        key="full-causal",
        label="Full Causal",
        mask="causal",
        cumulative=(0.285, 0.265, 0.265),
        context_clearing=(0.040, 0.040, 0.045),
    ),
    HistoryCondition(
        key="block_dropout-p050",
        label=r"Block Dropout $p{=}0.5$",
        mask="causal",
        cumulative=(0.295, 0.235, 0.180),
        context_clearing=(0.040, 0.020, 0.020),
    ),
    HistoryCondition(
        key="block_dropout-p090",
        label=r"Block Dropout $p{=}0.9$",
        mask="causal",
        cumulative=(0.165, 0.250, 0.145),
    ),
    HistoryCondition(
        key="sliding_window-k3",
        label=r"Sliding Window $k{=}3$",
        mask="causal",
        cumulative=(0.060, 0.060, 0.070),
    ),
    HistoryCondition(
        key="null_history",
        label="Null History",
        mask="causal",
        cumulative=(0.060, 0.035, 0.030),
    ),
    HistoryCondition(
        key="prefix_only",
        label="Prefix Only",
        mask="causal",
        cumulative=(0.030, 0.045, 0.060),
    ),
)

N75_SCALING: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("SSA", (0.100, 0.120, 0.115, 0.090, 0.115, 0.125, 0.135, 0.115)),
    ("Causal", (0.100, 0.110, 0.015, 0.085, 0.110, 0.080, 0.005, 0.005)),
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def compute_stats(values: Sequence[float]) -> Stats:
    if not values:
        raise ValueError("Expected at least one value.")
    if len(values) == 1:
        return Stats(mean_pct=values[0] * 100.0, std_pct=0.0)
    return Stats(mean_pct=mean(values) * 100.0, std_pct=stdev(values) * 100.0)


def compute_delta_stats(
    cumulative: Sequence[float],
    context_clearing: Sequence[float],
) -> Stats:
    if len(cumulative) != len(context_clearing):
        raise ValueError("Cumulative and context-clearing results must be paired.")
    deltas_pct = [100.0 * (ctx - cum) for cum, ctx in zip(cumulative, context_clearing)]
    if len(deltas_pct) == 1:
        return Stats(mean_pct=deltas_pct[0], std_pct=0.0)
    return Stats(mean_pct=mean(deltas_pct), std_pct=stdev(deltas_pct))


def format_mean_std(stats: Stats, *, bold: bool = False, signed: bool = False) -> str:
    mean_text = f"{stats.mean_pct:+.1f}" if signed else f"{stats.mean_pct:.1f}"
    body = rf"{mean_text} \pm {stats.std_pct:.1f}"
    return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"


def format_scalar(value: float, *, bold: bool = False) -> str:
    body = f"{value:.1f}"
    return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"


def write_table(filename: str, content: str) -> None:
    output_path = TABLE_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    LOGGER.info("wrote table=%s", output_path)


def build_history_table() -> str:
    rows: list[tuple[str, Stats, Stats | None, Stats | None]] = []
    for condition in HISTORY_CONDITIONS:
        cumulative_stats = compute_stats(condition.cumulative)
        context_stats = (
            compute_stats(condition.context_clearing)
            if condition.context_clearing is not None
            else None
        )
        delta_stats = (
            compute_delta_stats(condition.cumulative, condition.context_clearing)
            if condition.context_clearing is not None
            else None
        )
        LOGGER.info(
            "history key=%s mask=%s cumulative=%.1f±%.1f context=%s delta=%s",
            condition.key,
            condition.mask,
            cumulative_stats.mean_pct,
            cumulative_stats.std_pct,
            (
                f"{context_stats.mean_pct:.1f}±{context_stats.std_pct:.1f}"
                if context_stats is not None
                else "---"
            ),
            (
                f"{delta_stats.mean_pct:+.1f}±{delta_stats.std_pct:.1f}"
                if delta_stats is not None
                else "---"
            ),
        )
        rows.append((condition.label, cumulative_stats, context_stats, delta_stats))

    best_cumulative = max(cumulative.mean_pct for _, cumulative, _, _ in rows)
    best_context = max(
        context.mean_pct for _, _, context, _ in rows if context is not None
    )
    best_delta = max(delta.mean_pct for _, _, _, delta in rows if delta is not None)

    lines = [
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"\textbf{Training Condition} & \textbf{Cumulative (\%)} & \textbf{Context-Clearing (\%)} & \textbf{$\Delta$ Transfer (pp)} \\",
        r"\midrule",
    ]

    for label, cumulative_stats, context_stats, delta_stats in rows:
        cumulative_text = format_mean_std(
            cumulative_stats,
            bold=cumulative_stats.mean_pct == best_cumulative,
        )
        context_text = (
            format_mean_std(
                context_stats,
                bold=context_stats.mean_pct == best_context,
            )
            if context_stats is not None
            else r"---"
        )
        delta_text = (
            format_mean_std(
                delta_stats,
                bold=delta_stats.mean_pct == best_delta,
                signed=True,
            )
            if delta_stats is not None
            else r"---"
        )
        lines.append(
            f"{label} & {cumulative_text} & {context_text} & {delta_text} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def build_n75_scaling_table() -> str:
    rows: list[tuple[str, Stats, float, float, int, int]] = []
    for label, values in N75_SCALING:
        stats = compute_stats(values)
        min_pct = min(values) * 100.0
        max_pct = max(values) * 100.0
        collapsed = sum(value < 0.05 for value in values)
        LOGGER.info(
            "n75 label=%s mean=%.1f±%.1f min=%.1f max=%.1f collapsed=%d/%d seeds=%s",
            label,
            stats.mean_pct,
            stats.std_pct,
            min_pct,
            max_pct,
            collapsed,
            len(values),
            ", ".join(f"{value * 100.0:.1f}" for value in values),
        )
        rows.append((label, stats, min_pct, max_pct, collapsed, len(values)))

    best_mean = max(stats.mean_pct for _, stats, _, _, _, _ in rows)
    best_min = max(min_pct for _, _, min_pct, _, _, _ in rows)
    best_max = max(max_pct for _, _, _, max_pct, _, _ in rows)
    best_collapsed = min(collapsed for _, _, _, _, collapsed, _ in rows)

    lines = [
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{Mean $\pm$ Std (\%)} & \textbf{Min (\%)} & \textbf{Max (\%)} & \textbf{Collapsed ($<5\%$)} \\",
        r"\midrule",
    ]

    for label, stats, min_pct, max_pct, collapsed, num_seeds in rows:
        collapsed_text = f"{collapsed}/{num_seeds}"
        if collapsed == best_collapsed:
            collapsed_text = rf"\textbf{{{collapsed_text}}}"
        mean_text = format_mean_std(stats, bold=stats.mean_pct == best_mean)
        min_text = format_scalar(min_pct, bold=min_pct == best_min)
        max_text = format_scalar(max_pct, bold=max_pct == best_max)
        lines.append(
            f"{label} & {mean_text} & {min_text} & {max_text} & {collapsed_text} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    history_table = build_history_table()
    scaling_table = build_n75_scaling_table()
    write_table("tab_history_removal.tex", history_table)
    write_table("tab_n75_scaling.tex", scaling_table)


if __name__ == "__main__":
    main()
