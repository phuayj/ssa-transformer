#!/usr/bin/env python3
"""Generate factorial LaTeX table for appendix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats


DOMAINS = ("gc", "sat")
TRACES = ("enriched", "stripped")
MASKS = ("blanket", "selective")
SEEDS = (42, 123, 456, 789, 2024)


def info(message: str) -> None:
    print(f"[info] {message}", file=sys.stderr)


def warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def _normalize_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= v <= 1.0:
        return 100.0 * v
    return v


def _extract_solve_rate_percent(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        direct = _normalize_percent(payload.get("solve_rate"))
        if direct is not None:
            return direct

        aggregate = payload.get("aggregate")
        if isinstance(aggregate, dict):
            agg = _normalize_percent(aggregate.get("solve_rate"))
            if agg is not None:
                return agg

        results = payload.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            nested = _normalize_percent(results[0].get("solve_rate"))
            if nested is not None:
                return nested

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        nested = _normalize_percent(payload[0].get("solve_rate"))
        if nested is not None:
            return nested

    return None


def _load_solve_rate_percent(path: Path) -> Optional[float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        warn(f"Failed to parse JSON: {path} ({exc})")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        warn(f"Failed to read {path}: {exc}")
        return None

    solve_rate = _extract_solve_rate_percent(payload)
    if solve_rate is None:
        warn(f"solve_rate missing in {path}")
    return solve_rate


def _first_existing_solve_rate(
    paths: list[Path],
) -> tuple[Optional[float], Optional[Path]]:
    for path in paths:
        if not path.exists():
            continue
        solve_rate = _load_solve_rate_percent(path)
        if solve_rate is not None:
            return solve_rate, path
    return None, None


def _mask_variants(domain: str, trace: str, mask: str) -> list[str]:
    if domain == "sat" and trace == "enriched":
        return [mask, f"{mask}_ssa"]
    return [f"{mask}_ssa", mask]


def _result_candidates(
    results_root: Path,
    domain: str,
    trace: str,
    mask: str,
    seed: int,
) -> list[Path]:
    candidates: list[Path] = []
    for variant in _mask_variants(domain, trace, mask):
        candidates.append(
            results_root
            / f"factorial-eval-{domain}-{trace}-{variant}-seed{seed}"
            / "results.json"
        )

    if domain == "sat" and trace == "enriched":
        candidates.append(
            results_root / f"e3-eval-sat-{mask}-ssa-seed{seed}" / "results.json"
        )

    return candidates


def _mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


def _signed(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f}"


def _fmt_mean_std_math(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "$-$"
    if std is None:
        return f"${mean:.1f}$"
    return f"${mean:.1f} \\pm {std:.1f}$"


def _fmt_interaction(mean: Optional[float], p_value: Optional[float]) -> str:
    if mean is None:
        return "$-$"
    if p_value is None:
        return f"${mean:+.1f}$ ($p = \\mathrm{{NA}}$)"
    return f"${mean:+.1f}$ ($p = {p_value:.3f}$)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments"),
        help="Results directory root (default: experiments)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = args.results_dir
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    results_root = results_root.resolve()
    info(f"results_root={results_root}")

    per_seed: dict[str, dict[str, dict[str, dict[int, float]]]] = {
        domain: {trace: {mask: {} for mask in MASKS} for trace in TRACES}
        for domain in DOMAINS
    }

    for domain in DOMAINS:
        for trace in TRACES:
            for mask in MASKS:
                for seed in SEEDS:
                    candidates = _result_candidates(
                        results_root, domain, trace, mask, seed
                    )
                    solve_rate, source = _first_existing_solve_rate(candidates)
                    if solve_rate is None:
                        warn(
                            "Missing result "
                            f"domain={domain} trace={trace} mask={mask} seed={seed}"
                        )
                        continue
                    per_seed[domain][trace][mask][seed] = solve_rate
                    info(
                        "Loaded result "
                        f"domain={domain} trace={trace} mask={mask} seed={seed} "
                        f"solve={solve_rate:.1f} path={source}"
                    )

    condition_stats: dict[str, dict[str, dict[str, dict[str, Optional[float]]]]] = {
        domain: {trace: {} for trace in TRACES} for domain in DOMAINS
    }
    deltas: dict[str, dict[str, Optional[float]]] = {
        domain: {trace: None for trace in TRACES} for domain in DOMAINS
    }
    interactions: dict[str, dict[str, Optional[float]]] = {
        domain: {"mean": None, "p_value": None} for domain in DOMAINS
    }

    for domain in DOMAINS:
        for trace in TRACES:
            for mask in MASKS:
                values = list(per_seed[domain][trace][mask].values())
                mean, std = _mean_std(values)
                condition_stats[domain][trace][mask] = {
                    "mean": mean,
                    "std": std,
                    "n": float(len(values)),
                }

            common = sorted(
                set(per_seed[domain][trace]["blanket"]).intersection(
                    per_seed[domain][trace]["selective"]
                )
            )
            diffs = [
                per_seed[domain][trace]["blanket"][seed]
                - per_seed[domain][trace]["selective"][seed]
                for seed in common
            ]
            delta_mean, delta_std = _mean_std(diffs)
            deltas[domain][trace] = delta_mean
            info(
                "Delta summary "
                f"domain={domain} trace={trace} n={len(common)} "
                f"delta_mean={_signed(delta_mean)} delta_std={delta_std if delta_std is not None else float('nan'):.3f}"
            )

        common_interaction = sorted(
            set(per_seed[domain]["enriched"]["blanket"])
            .intersection(per_seed[domain]["enriched"]["selective"])
            .intersection(per_seed[domain]["stripped"]["blanket"])
            .intersection(per_seed[domain]["stripped"]["selective"])
        )
        interaction_values = [
            (
                per_seed[domain]["enriched"]["blanket"][seed]
                - per_seed[domain]["enriched"]["selective"][seed]
            )
            - (
                per_seed[domain]["stripped"]["blanket"][seed]
                - per_seed[domain]["stripped"]["selective"][seed]
            )
            for seed in common_interaction
        ]
        interaction_mean, _ = _mean_std(interaction_values)

        if len(interaction_values) >= 2:
            test = stats.ttest_1samp(
                np.asarray(interaction_values, dtype=np.float64), popmean=0.0
            )
            p_value = float(test.pvalue)
            if math.isnan(p_value):
                p_value = None
        else:
            p_value = None

        interactions[domain] = {"mean": interaction_mean, "p_value": p_value}
        info(
            "Interaction summary "
            f"domain={domain} n={len(interaction_values)} "
            f"interaction_mean={_signed(interaction_mean)} p={p_value}"
        )

    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Trace} & \textbf{Mask} & \textbf{Solve \%} & \textbf{$\Delta$(B$-$S)} \\",
        r"\midrule",
        "\\multirow{4}{*}{GC} & \\multirow{2}{*}{Enriched} & Blanket & "
        f"{_fmt_mean_std_math(condition_stats['gc']['enriched']['blanket']['mean'], condition_stats['gc']['enriched']['blanket']['std'])} "
        f"& \\multirow{{2}}{{*}}{{${_signed(deltas['gc']['enriched'])}$}} \\\\",
        "& & Selective & "
        f"{_fmt_mean_std_math(condition_stats['gc']['enriched']['selective']['mean'], condition_stats['gc']['enriched']['selective']['std'])} & \\\\",
        "& \\multirow{2}{*}{Stripped} & Blanket & "
        f"{_fmt_mean_std_math(condition_stats['gc']['stripped']['blanket']['mean'], condition_stats['gc']['stripped']['blanket']['std'])} "
        f"& \\multirow{{2}}{{*}}{{${_signed(deltas['gc']['stripped'])}$}} \\\\",
        "& & Selective & "
        f"{_fmt_mean_std_math(condition_stats['gc']['stripped']['selective']['mean'], condition_stats['gc']['stripped']['selective']['std'])} & \\\\",
        r"\midrule",
        "\\multirow{4}{*}{SAT} & \\multirow{2}{*}{Enriched} & Blanket & "
        f"{_fmt_mean_std_math(condition_stats['sat']['enriched']['blanket']['mean'], condition_stats['sat']['enriched']['blanket']['std'])} "
        f"& \\multirow{{2}}{{*}}{{${_signed(deltas['sat']['enriched'])}$}} \\\\",
        "& & Selective & "
        f"{_fmt_mean_std_math(condition_stats['sat']['enriched']['selective']['mean'], condition_stats['sat']['enriched']['selective']['std'])} & \\\\",
        "& \\multirow{2}{*}{Stripped} & Blanket & "
        f"{_fmt_mean_std_math(condition_stats['sat']['stripped']['blanket']['mean'], condition_stats['sat']['stripped']['blanket']['std'])} "
        f"& \\multirow{{2}}{{*}}{{${_signed(deltas['sat']['stripped'])}$}} \\\\",
        "& & Selective & "
        f"{_fmt_mean_std_math(condition_stats['sat']['stripped']['selective']['mean'], condition_stats['sat']['stripped']['selective']['std'])} & \\\\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Interaction ($\Delta_\text{enriched} - \Delta_\text{stripped}$)}} \\",
        f"\\quad GC & & & {_fmt_interaction(interactions['gc']['mean'], interactions['gc']['p_value'])} \\\\",
        f"\\quad SAT & & & {_fmt_interaction(interactions['sat']['mean'], interactions['sat']['p_value'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    table = "\n".join(lines)

    output_path = (
        repo_root / "paper" / "manuscript" / "parts" / "tables" / "tab_factorial.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n", encoding="utf-8")
    info(f"Wrote LaTeX table: {output_path}")

    print(table)


if __name__ == "__main__":
    main()
