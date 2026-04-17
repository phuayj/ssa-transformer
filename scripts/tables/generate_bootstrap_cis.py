#!/usr/bin/env python3
"""Compute bootstrap CIs for E3 claims and generate appendix plots/tables."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


SEEDS: List[int] = [42, 123, 456, 789, 2024]
N_BOOTSTRAP_DEFAULT = 10_000

REPO_ROOT = Path(__file__).resolve().parents[2]
PICTURES_DIR = REPO_ROOT / "output" / "pictures"
TABLES_DIR = REPO_ROOT / "output" / "tables"

HARDCODED_SOLVE_RATES: Dict[str, Dict[str, List[float]]] = {
    "GC": {
        "Selective SSA": [47.5, 55.5, 52.0, 52.5, 49.0],
        "Blanket SSA": [47.0, 44.0, 53.5, 48.0, 47.0],
        "Full Causal": [36.5, 39.0, 34.0, 26.5, 35.5],
        "SWA-Prefix": [38.5, 44.5, 36.5, 44.0, 42.5],
    },
    "SAT": {
        "Selective SSA": [59.0, 67.0, 62.5, 60.0, 61.0],
        "Blanket SSA": [61.5, 67.0, 65.0, 65.5, 65.0],
        "Full Causal": [55.0, 50.0, 54.5, 53.0, 58.0],
        "SWA-Prefix": [9.5, 59.5, 53.5, 55.0, 57.0],
    },
}

RESULT_PATH_TEMPLATES: Dict[str, Dict[str, str]] = {
    "GC": {
        "Selective SSA": "experiments/e3v3-eval-selective-ssa-seed{seed}/results.json",
        "Blanket SSA": "experiments/e3v3-eval-blanket-ssa-seed{seed}/results.json",
        "Full Causal": "experiments/e3v3-eval-full-causal-seed{seed}/results.json",
        "SWA-Prefix": "experiments/e3v3-eval-swa-prefix-seed{seed}/results.json",
    },
    "SAT": {
        "Selective SSA": "experiments/e3v3-eval-sat-selective-ssa-seed{seed}/results.json",
        "Blanket SSA": "experiments/e3v3-eval-sat-blanket-ssa-seed{seed}/results.json",
        "Full Causal": "experiments/e3v3-eval-sat-full-causal-seed{seed}/results.json",
        "SWA-Prefix": "experiments/e3v3-eval-sat-swa-prefix-seed{seed}/results.json",
    },
}


@dataclass
class ComparisonResult:
    domain: str
    comparison: str
    mean_delta: float
    ci_low: float
    ci_high: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _extract_solve_rate_percent(result_path: Path) -> float:
    with result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or not payload.get("results"):
        raise ValueError(f"Malformed results payload at {result_path}")

    value = payload["results"][0].get("solve_rate")
    if value is None:
        value = payload["results"][0].get("success_rate")
    if value is None:
        raise KeyError(f"No solve_rate/success_rate in {result_path}")

    value = float(value)
    if value <= 1.0:
        value *= 100.0
    return value


def load_domain_values(domain: str) -> Dict[str, np.ndarray]:
    values: Dict[str, np.ndarray] = {}
    templates = RESULT_PATH_TEMPLATES[domain]

    for method_name, hardcoded in HARDCODED_SOLVE_RATES[domain].items():
        template = templates.get(method_name)
        loaded: List[float] = []
        all_found = template is not None

        if template is not None:
            for seed in SEEDS:
                candidate = REPO_ROOT / template.format(seed=seed)
                if not candidate.exists():
                    all_found = False
                    break
                loaded.append(_extract_solve_rate_percent(candidate))

        if all_found and len(loaded) == len(SEEDS):
            arr = np.array(loaded, dtype=float)
            logging.info(
                "[%s/%s] loaded from results files: %s", domain, method_name, arr
            )
            values[method_name] = arr
        else:
            arr = np.array(hardcoded, dtype=float)
            logging.info(
                "[%s/%s] using hardcoded fallback values: %s", domain, method_name, arr
            )
            values[method_name] = arr

    return values


def bootstrap_mean_ci(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> Tuple[float, float]:
    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    sample_means = values[idx].mean(axis=1)
    return float(np.percentile(sample_means, 2.5)), float(
        np.percentile(sample_means, 97.5)
    )


def compute_domain_comparisons(
    domain: str,
    domain_values: Dict[str, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> List[ComparisonResult]:
    selective = domain_values["Selective SSA"]
    blanket = domain_values["Blanket SSA"]
    causal = domain_values["Full Causal"]
    swa = domain_values["SWA-Prefix"]

    selective_mean = float(selective.mean())
    blanket_mean = float(blanket.mean())
    ssa_mean = max(selective_mean, blanket_mean)
    logging.info(
        "[%s] selective_mean=%.3f blanket_mean=%.3f ssa_mean(best)=%.3f",
        domain,
        selective_mean,
        blanket_mean,
        ssa_mean,
    )

    idx = rng.integers(0, len(selective), size=(n_bootstrap, len(selective)))
    selective_boot = selective[idx]
    blanket_boot = blanket[idx]
    causal_boot = causal[idx]
    swa_boot = swa[idx]

    ssa_boot_means = np.maximum(selective_boot.mean(axis=1), blanket_boot.mean(axis=1))
    ssa_minus_causal = ssa_boot_means - causal_boot.mean(axis=1)
    ssa_minus_swa = ssa_boot_means - swa_boot.mean(axis=1)
    sel_minus_blanket = selective_boot.mean(axis=1) - blanket_boot.mean(axis=1)

    comparisons: List[ComparisonResult] = [
        ComparisonResult(
            domain=domain,
            comparison="SSA - Causal",
            mean_delta=ssa_mean - float(causal.mean()),
            ci_low=float(np.percentile(ssa_minus_causal, 2.5)),
            ci_high=float(np.percentile(ssa_minus_causal, 97.5)),
        ),
        ComparisonResult(
            domain=domain,
            comparison="Selective - Blanket",
            mean_delta=selective_mean - blanket_mean,
            ci_low=float(np.percentile(sel_minus_blanket, 2.5)),
            ci_high=float(np.percentile(sel_minus_blanket, 97.5)),
        ),
        ComparisonResult(
            domain=domain,
            comparison="SSA - SWA-Prefix",
            mean_delta=ssa_mean - float(swa.mean()),
            ci_low=float(np.percentile(ssa_minus_swa, 2.5)),
            ci_high=float(np.percentile(ssa_minus_swa, 97.5)),
        ),
    ]

    for row in comparisons:
        logging.info(
            "[%s] %s mean_delta=%.3f ci=[%.3f, %.3f]",
            row.domain,
            row.comparison,
            row.mean_delta,
            row.ci_low,
            row.ci_high,
        )
    return comparisons


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def make_domain_plot(
    domain: str,
    domain_values: Dict[str, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
    out_path: Path,
) -> None:
    methods = ["Selective SSA", "Blanket SSA", "Full Causal", "SWA-Prefix"]
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]

    x = np.arange(1, len(methods) + 1)
    data = [domain_values[name] for name in methods]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    box = ax.boxplot(
        data,
        positions=x,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)

    for i, (name, color) in enumerate(zip(methods, colors), start=1):
        values = domain_values[name]
        jitter = rng.normal(0.0, 0.045, size=len(values))
        ax.scatter(
            np.full_like(values, i, dtype=float) + jitter,
            values,
            s=36,
            color=color,
            alpha=0.9,
            linewidths=0.5,
            edgecolors="white",
            zorder=3,
        )

        mean_val = float(values.mean())
        ci_low, ci_high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, rng=rng)
        ax.hlines(mean_val, i - 0.2, i + 0.2, colors=color, linewidth=2.4, zorder=4)
        ax.errorbar(
            i,
            mean_val,
            yerr=[[mean_val - ci_low], [ci_high - mean_val]],
            fmt="none",
            ecolor=color,
            elinewidth=1.6,
            capsize=3,
            zorder=4,
        )
        logging.info(
            "[%s/%s] per-seed=%s mean=%.3f ci=[%.3f, %.3f]",
            domain,
            name,
            values,
            mean_val,
            ci_low,
            ci_high,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Solve rate (%)")
    ax.set_title(f"{domain}: Per-seed solve rates")
    ax.grid(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot: %s", out_path)


def write_latex_table(results: List[ComparisonResult], out_path: Path) -> None:
    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "\\textbf{Domain} & \\textbf{Comparison} & \\textbf{Mean $\\Delta$} & \\textbf{95\\% Bootstrap CI} \\\\",
        "\\midrule",
    ]

    for row in results:
        lines.append(
            f"{row.domain} & {row.comparison.replace('-', '$-$')} & "
            f"${row.mean_delta:+.1f}$ & $[{row.ci_low:.1f}, {row.ci_high:.1f}]$ \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved LaTeX table: %s", out_path)


def print_summary(results: List[ComparisonResult]) -> None:
    print("Bootstrap CI summary (95%, paired seed bootstrap):")
    for row in results:
        excludes = "YES" if row.excludes_zero else "NO"
        print(
            f"- {row.domain:>3} | {row.comparison:<20} "
            f"mean_delta={row.mean_delta:+.2f} "
            f"CI=[{row.ci_low:+.2f}, {row.ci_high:+.2f}] "
            f"excludes_zero={excludes}"
        )

    significant = [f"{r.domain} {r.comparison}" for r in results if r.excludes_zero]
    print("\nComparisons with CI excluding zero:")
    if significant:
        for item in significant:
            print(f"  - {item}")
    else:
        print("  (none)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bootstrap CIs, per-seed plots, and LaTeX table for paper appendix."
    )
    parser.add_argument("--bootstrap-iters", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    configure_plot_style()

    rng = np.random.default_rng(args.random_seed)

    all_rows: List[ComparisonResult] = []
    for domain in ("GC", "SAT"):
        domain_values = load_domain_values(domain)
        rows = compute_domain_comparisons(
            domain,
            domain_values,
            n_bootstrap=args.bootstrap_iters,
            rng=rng,
        )
        all_rows.extend(rows)

        out_plot = PICTURES_DIR / f"fig_perseed_{domain.lower()}.pdf"
        make_domain_plot(
            domain,
            domain_values,
            n_bootstrap=args.bootstrap_iters,
            rng=rng,
            out_path=out_plot,
        )

    latex_path = TABLES_DIR / "tab_bootstrap_cis.tex"
    write_latex_table(all_rows, latex_path)
    print_summary(all_rows)


if __name__ == "__main__":
    main()
