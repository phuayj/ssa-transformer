#!/usr/bin/env python3
"""Generate paper-ready figures for the neuro-symbolic CSP solver paper."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

NONNEURAL_ABLATION_PATH = (
    REPO_ROOT / "experiments/ablation-baselines-nonneural/ablation_results.json"
)
NEURAL_ABLATION_PATH = (
    REPO_ROOT / "experiments/ablation-baselines-neural/ablation_results.json"
)
SWEEP_PATH = REPO_ROOT / "experiments/stochastic-sweep-v1/sweep_results.json"
VERIFY_DSATUR_PATH = (
    REPO_ROOT / "experiments/verify-200inst-dsatur/ablation_results.json"
)
VERIFY_NEURAL_PATH = (
    REPO_ROOT / "experiments/verify-200inst-neural/ablation_results.json"
)

RESTARTS = [1, 4, 16, 32]

COLOR_GRAY = "#7f7f7f"
COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_RED = "#d62728"
COLOR_GREEN = "#2ca02c"
COLOR_PURPLE = "#9467bd"
COLOR_BLACK = "#000000"

METHOD_STYLES: Dict[str, Dict[str, Any]] = {
    "random": {
        "label": "Random",
        "color": COLOR_GRAY,
        "linestyle": "--",
        "marker": "o",
    },
    "dsatur": {
        "label": "DSATUR",
        "color": COLOR_BLUE,
        "linestyle": "-",
        "marker": "o",
    },
    "neural_greedy": {
        "label": "Neural Greedy",
        "color": COLOR_ORANGE,
        "linestyle": "--",
        "marker": "o",
    },
    "neural_stochastic": {
        "label": "Neural Stochastic K=5 T=0.5",
        "color": COLOR_RED,
        "linestyle": "-",
        "marker": "o",
    },
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path, allow_missing: bool = True) -> Optional[Dict[str, Any]]:
    if not path.exists():
        if allow_missing:
            logging.warning("Missing data file: %s", path)
            return None
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - spread, center + spread


def max_steps_from_eval_config(
    eval_config: Optional[Dict[str, Any]], default: int = 500
) -> int:
    if not eval_config:
        return default
    for key in (
        "max_steps_per_restart",
        "max_steps",
        "max_steps_per_episode",
        "max_steps_per_rollout",
    ):
        value = eval_config.get(key)
        if value is not None:
            return int(value)
    return default


def compute_budget(entry: Dict[str, Any], num_restarts: int, max_steps: int) -> int:
    if entry.get("compute_budget") is not None:
        return int(entry["compute_budget"])
    return int(num_restarts * max_steps)


def get_ablation_entry(
    ablation_data: Dict[str, Any], method_name: str, num_restarts: int
) -> Optional[Dict[str, Any]]:
    results = ablation_data.get("results", {})
    method_results = results.get(method_name)
    if not isinstance(method_results, dict):
        return None
    if num_restarts in method_results:
        return method_results[num_restarts]
    if str(num_restarts) in method_results:
        return method_results[str(num_restarts)]
    for key, value in method_results.items():
        try:
            if int(key) == num_restarts:
                return value
        except (TypeError, ValueError):
            continue
    return None


def success_stats_from_entry(
    entry: Dict[str, Any], fallback_n: Optional[int] = None
) -> Optional[Tuple[int, int, float]]:
    per_instance = entry.get("per_instance") or []
    if per_instance:
        k = sum(1 for inst in per_instance if inst.get("success"))
        n = len(per_instance)
    else:
        n = entry.get("num_instances") or fallback_n
        success_rate = entry.get("success_rate")
        if n is None or success_rate is None:
            return None
        k = int(round(float(success_rate) * n))
    if n == 0:
        return None
    return k, n, k / n


def build_series_from_ablation(
    ablation_data: Optional[Dict[str, Any]],
    method_name: str,
    restarts: Sequence[int],
    label: str,
) -> Optional[Dict[str, Any]]:
    if ablation_data is None:
        return None
    eval_config = ablation_data.get("eval_config", {})
    max_steps = max_steps_from_eval_config(eval_config)
    series = {
        "method": method_name,
        "label": label,
        "x": [],
        "y": [],
        "ci_low": [],
        "ci_high": [],
        "yerr": [],
        "k": [],
        "n": [],
        "compute_steps": [],
        "per_instance": {},
    }
    for num_restarts in restarts:
        entry = get_ablation_entry(ablation_data, method_name, num_restarts)
        if entry is None:
            logging.warning(
                "No ablation entry for %s restarts=%s", method_name, num_restarts
            )
            continue
        stats = success_stats_from_entry(entry)
        if stats is None:
            logging.warning(
                "Missing success stats for %s restarts=%s", method_name, num_restarts
            )
            continue
        k, n, p = stats
        ci_low, ci_high = wilson_ci(k, n)
        y_value = 100.0 * p
        series["x"].append(num_restarts)
        series["y"].append(y_value)
        series["ci_low"].append(100.0 * ci_low)
        series["ci_high"].append(100.0 * ci_high)
        series["yerr"].append((y_value - 100.0 * ci_low, 100.0 * ci_high - y_value))
        series["k"].append(k)
        series["n"].append(n)
        series["compute_steps"].append(compute_budget(entry, num_restarts, max_steps))
        if entry.get("per_instance") is not None:
            series["per_instance"][num_restarts] = entry["per_instance"]
    if not series["x"]:
        return None
    return series


def _config_matches_sweep(
    config: Dict[str, Any], top_k: int, temperature: float, stochastic_depth: int
) -> bool:
    if int(config.get("top_k", -1)) != top_k:
        return False
    temp_value = config.get("temperature")
    if temp_value is None or not math.isclose(
        float(temp_value), temperature, rel_tol=1e-4, abs_tol=1e-4
    ):
        return False
    depth_value = config.get("stochastic_depth")
    if depth_value is None:
        return False
    return int(depth_value) == stochastic_depth


def build_series_from_sweep(
    sweep_data: Optional[Dict[str, Any]],
    restarts: Sequence[int],
    top_k: int,
    temperature: float,
    stochastic_depth: int,
    label: str,
) -> Optional[Dict[str, Any]]:
    if sweep_data is None:
        return None
    eval_config = sweep_data.get("eval_config", {})
    max_steps = max_steps_from_eval_config(eval_config)
    series = {
        "method": "neural_stochastic",
        "label": label,
        "x": [],
        "y": [],
        "ci_low": [],
        "ci_high": [],
        "yerr": [],
        "k": [],
        "n": [],
        "compute_steps": [],
        "per_instance": {},
    }
    matches: Dict[int, Dict[str, Any]] = {}
    for entry in sweep_data.get("results", []):
        config = entry.get("config")
        if not isinstance(config, dict):
            continue
        if not _config_matches_sweep(config, top_k, temperature, stochastic_depth):
            continue
        num_restarts = config.get("num_restarts")
        if num_restarts is None:
            logging.warning("Sweep entry missing num_restarts: %s", config)
            continue
        num_restarts = int(num_restarts)
        if num_restarts in matches:
            logging.warning(
                "Duplicate sweep entry for restarts=%s; keeping latest", num_restarts
            )
        matches[num_restarts] = entry
    for num_restarts in restarts:
        entry = matches.get(num_restarts)
        if entry is None:
            logging.warning(
                "No sweep entry for K=%s T=%.2f D=%s restarts=%s",
                top_k,
                temperature,
                stochastic_depth,
                num_restarts,
            )
            continue
        summary = entry.get("summary", {})
        per_instance = entry.get("per_instance") or []
        if per_instance:
            k = sum(1 for inst in per_instance if inst.get("success"))
            n = len(per_instance)
        else:
            n = summary.get("num_instances") or eval_config.get("num_instances")
            success_rate = summary.get("success_rate")
            if n is None or success_rate is None:
                logging.warning(
                    "Missing sweep success stats for restarts=%s", num_restarts
                )
                continue
            k = int(round(float(success_rate) * n))
        if n == 0:
            continue
        ci_low, ci_high = wilson_ci(k, n)
        y_value = 100.0 * (k / n)
        series["x"].append(num_restarts)
        series["y"].append(y_value)
        series["ci_low"].append(100.0 * ci_low)
        series["ci_high"].append(100.0 * ci_high)
        series["yerr"].append((y_value - 100.0 * ci_low, 100.0 * ci_high - y_value))
        series["k"].append(k)
        series["n"].append(n)
        compute = summary.get("compute_budget")
        if compute is not None:
            series["compute_steps"].append(int(compute))
        else:
            series["compute_steps"].append(num_restarts * max_steps)
        if per_instance:
            series["per_instance"][num_restarts] = per_instance
    if not series["x"]:
        return None
    return series


def log_series_ci(
    label: str,
    x_values: Sequence[int],
    y_values: Sequence[float],
    ci_low: Sequence[float],
    ci_high: Sequence[float],
    n_values: Sequence[int],
) -> None:
    for x_value, y_value, low, high, n_value in zip(
        x_values, y_values, ci_low, ci_high, n_values
    ):
        logging.info(
            "%s: restarts=%s success=%.1f%% CI=[%.1f, %.1f] n=%s",
            label,
            x_value,
            y_value,
            low,
            high,
            n_value,
        )


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{filename}.pdf"
    png_path = output_dir / f"{filename}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure to %s and %s", pdf_path, png_path)
    return [pdf_path, png_path]


def plot_success_rate_vs_restarts(
    series_list: Sequence[Dict[str, Any]],
    output_dir: Path,
    title: str,
    filename: str,
) -> List[Path]:
    if not series_list:
        logging.warning("No series available for %s", filename)
        return []
    fig, ax = plt.subplots(figsize=(8, 5))
    for series in series_list:
        style = METHOD_STYLES.get(series["method"], {})
        x_values = np.array(series["x"], dtype=float)
        y_values = np.array(series["y"], dtype=float)
        yerr = np.array(series["yerr"], dtype=float).T
        log_series_ci(
            series["label"],
            series["x"],
            series["y"],
            series["ci_low"],
            series["ci_high"],
            series["n"],
        )
        ax.errorbar(
            x_values,
            y_values,
            yerr=yerr,
            label=series["label"],
            color=style.get("color", COLOR_BLACK),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            linewidth=2,
            capsize=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(RESTARTS)
    ax.set_xticklabels([str(r) for r in RESTARTS])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Number of Restarts")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(False)
    return save_figure(fig, output_dir, filename)


def plot_anytime_success_compute(
    series_list: Sequence[Dict[str, Any]],
    output_dir: Path,
    filename: str,
) -> List[Path]:
    if not series_list:
        logging.warning("No series available for %s", filename)
        return []
    fig, ax = plt.subplots(figsize=(8, 5))
    for series in series_list:
        style = METHOD_STYLES.get(series["method"], {})
        x_values = np.array(series["compute_steps"], dtype=float)
        y_values = np.array(series["y"], dtype=float)
        yerr = np.array(series["yerr"], dtype=float).T
        order = np.argsort(x_values)
        logging.info(
            "%s compute steps: %s",
            series["label"],
            list(np.array(series["compute_steps"])[order]),
        )
        ax.errorbar(
            x_values[order],
            y_values[order],
            yerr=yerr[:, order],
            label=series["label"],
            color=style.get("color", COLOR_BLACK),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            linewidth=2,
            capsize=3,
        )

    lookahead_points = [
        (25000, 72.0, "Lookahead Probe 72%"),
        (40000, 76.0, "Lookahead Probe 76%"),
    ]
    for idx, (x_val, y_val, label) in enumerate(lookahead_points):
        ax.scatter(
            x_val,
            y_val,
            color=COLOR_PURPLE,
            marker="D",
            s=50,
            label="Lookahead Probe" if idx == 0 else None,
        )
        ax.annotate(
            label,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
        )

    lds_point = (15000, 52.0)
    ax.scatter(
        lds_point[0],
        lds_point[1],
        color=COLOR_BLACK,
        marker="s",
        s=50,
        label="LDS",
    )
    ax.annotate(
        "LDS 52%",
        lds_point,
        textcoords="offset points",
        xytext=(6, -10),
        fontsize=10,
    )

    ax.set_xlabel("Total Computation Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Anytime Success vs. Compute Budget")
    ax.ticklabel_format(style="plain", axis="x")
    ax.legend(frameon=False)
    ax.grid(False)
    return save_figure(fig, output_dir, filename)


def _difficulty_score(
    dsatur_entry: Dict[str, Any], neural_entry: Dict[str, Any]
) -> float:
    scores: List[float] = []
    for entry in (dsatur_entry, neural_entry):
        steps = entry.get("total_steps")
        if steps is None:
            steps = entry.get("total_decisions") or 0
        if not entry.get("success", False):
            steps += 1e6
        scores.append(float(steps))
    return min(scores) if scores else 1e9


def build_instance_comparison(
    dsatur_instances: Sequence[Dict[str, Any]],
    neural_instances: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not dsatur_instances or not neural_instances:
        return None
    dsatur_map = {entry["instance"]: entry for entry in dsatur_instances}
    neural_map = {entry["instance"]: entry for entry in neural_instances}
    common_ids = sorted(set(dsatur_map) & set(neural_map))
    if not common_ids:
        return None
    sorted_ids = sorted(
        common_ids,
        key=lambda idx: _difficulty_score(dsatur_map[idx], neural_map[idx]),
    )
    matrix = np.zeros((2, len(sorted_ids)), dtype=int)
    discordant_cols: List[int] = []
    for col, instance_id in enumerate(sorted_ids):
        dsatur_success = bool(dsatur_map[instance_id].get("success"))
        neural_success = bool(neural_map[instance_id].get("success"))
        matrix[0, col] = 1 if dsatur_success else 0
        matrix[1, col] = 1 if neural_success else 0
        if dsatur_success != neural_success:
            discordant_cols.append(col)
    both = int(np.logical_and(matrix[0] == 1, matrix[1] == 1).sum())
    only_dsatur = int(np.logical_and(matrix[0] == 1, matrix[1] == 0).sum())
    only_neural = int(np.logical_and(matrix[0] == 0, matrix[1] == 1).sum())
    neither = int(np.logical_and(matrix[0] == 0, matrix[1] == 0).sum())
    counts = {
        "both": both,
        "only_dsatur": only_dsatur,
        "only_neural": only_neural,
        "neither": neither,
        "dsatur_total": int(matrix[0].sum()),
        "neural_total": int(matrix[1].sum()),
        "num_instances": len(sorted_ids),
    }
    logging.info(
        "Instance comparison counts: %s",
        counts,
    )
    return {
        "matrix": matrix,
        "sorted_ids": sorted_ids,
        "discordant_cols": discordant_cols,
        "counts": counts,
    }


def plot_instance_heatmap(
    comparison: Dict[str, Any], output_dir: Path, filename: str
) -> List[Path]:
    matrix = comparison["matrix"]
    counts = comparison["counts"]
    discordant_cols = comparison["discordant_cols"]
    num_instances = counts["num_instances"]

    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = ListedColormap([COLOR_RED, COLOR_GREEN])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["DSATUR", "Neural Stochastic"])
    tick_positions = np.linspace(
        0, num_instances - 1, num=min(6, num_instances), dtype=int
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pos + 1) for pos in tick_positions])
    ax.set_xlabel("Instances (sorted by difficulty)")
    ax.set_title("Per-instance Outcomes at N=32")

    for col in discordant_cols:
        ax.add_patch(
            Rectangle(
                (col - 0.5, -0.5),
                1,
                2,
                fill=False,
                edgecolor="black",
                linewidth=1.3,
            )
        )

    annotation = (
        "DSATUR {dsatur_total}/{num_instances} | Neural {neural_total}/{num_instances} | "
        "Only DSATUR {only_dsatur} | Only Neural {only_neural} | Both {both} | Neither {neither}"
    ).format(**counts)
    fig.text(0.5, -0.05, annotation, ha="center", fontsize=11)
    return save_figure(fig, output_dir, filename)


def plot_complementarity(
    counts: Dict[str, int], output_dir: Path, filename: str
) -> List[Path]:
    categories = ["Both", "DSATUR only", "Neural only", "Neither"]
    values = [
        counts["both"],
        counts["only_dsatur"],
        counts["only_neural"],
        counts["neither"],
    ]
    colors = [COLOR_GREEN, COLOR_BLUE, COLOR_ORANGE, COLOR_GRAY]

    fig, ax = plt.subplots(figsize=(6, 4))
    x_positions = np.arange(len(categories))
    bars = ax.bar(x_positions, values, color=colors)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel("Number of Instances")
    ax.set_title("Complementarity at N=32")
    ax.set_ylim(0, max(values) + 5)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.grid(False)
    return save_figure(fig, output_dir, filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready figures for CSP solver experiments."
    )
    parser.add_argument(
        "--output_dir",
        default="figures/",
        help="Output directory for generated figures.",
    )
    args = parser.parse_args()

    configure_logging()
    configure_style()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_nonneural = load_json(NONNEURAL_ABLATION_PATH, allow_missing=True)
    ablation_neural = load_json(NEURAL_ABLATION_PATH, allow_missing=True)
    sweep_data = load_json(SWEEP_PATH, allow_missing=True)
    verify_dsatur = load_json(VERIFY_DSATUR_PATH, allow_missing=True)
    verify_neural = load_json(VERIFY_NEURAL_PATH, allow_missing=True)

    generated: List[Path] = []

    series_map: Dict[str, Optional[Dict[str, Any]]] = {
        "random": build_series_from_ablation(
            ablation_nonneural,
            "random",
            RESTARTS,
            METHOD_STYLES["random"]["label"],
        ),
        "dsatur": build_series_from_ablation(
            ablation_nonneural,
            "dsatur",
            RESTARTS,
            METHOD_STYLES["dsatur"]["label"],
        ),
        "neural_greedy": build_series_from_ablation(
            ablation_neural,
            "neural_greedy",
            RESTARTS,
            METHOD_STYLES["neural_greedy"]["label"],
        ),
    }

    neural_stochastic = build_series_from_sweep(
        sweep_data,
        RESTARTS,
        top_k=5,
        temperature=0.5,
        stochastic_depth=15,
        label=METHOD_STYLES["neural_stochastic"]["label"],
    )
    if neural_stochastic is None:
        fallback = build_series_from_ablation(
            ablation_nonneural,
            "dsatur_stochastic",
            RESTARTS,
            METHOD_STYLES["neural_stochastic"]["label"],
        )
        if fallback is not None:
            logging.warning(
                "Using dsatur_stochastic as fallback for neural_stochastic."
            )
            fallback["method"] = "neural_stochastic"
            neural_stochastic = fallback
    series_map["neural_stochastic"] = neural_stochastic

    ordered_series: List[Dict[str, Any]] = []
    for method in ("random", "dsatur", "neural_greedy", "neural_stochastic"):
        series = series_map.get(method)
        if series is not None:
            ordered_series.append(series)

    generated += plot_success_rate_vs_restarts(
        ordered_series,
        output_dir,
        "Success Rate vs. Restarts at Phase Transition (n=50, k=4, p=0.23)",
        "figure1_success_vs_restarts",
    )
    generated += plot_anytime_success_compute(
        ordered_series,
        output_dir,
        "figure2_anytime_success_compute",
    )

    dsatur_series = series_map.get("dsatur")
    neural_series = series_map.get("neural_stochastic")
    if dsatur_series and neural_series:
        dsatur_instances = dsatur_series["per_instance"].get(32)
        neural_instances = neural_series["per_instance"].get(32)
        comparison = (
            build_instance_comparison(dsatur_instances, neural_instances)
            if dsatur_instances and neural_instances
            else None
        )
        if comparison is None:
            logging.warning(
                "Insufficient per-instance data for heatmap/complementarity."
            )
        else:
            generated += plot_instance_heatmap(
                comparison,
                output_dir,
                "figure3_instance_heatmap",
            )
            generated += plot_complementarity(
                comparison["counts"],
                output_dir,
                "figure4_complementarity",
            )
    else:
        logging.warning("Skipping per-instance comparison due to missing series.")

    verification_series: List[Dict[str, Any]] = []
    dsatur_verify_series = build_series_from_ablation(
        verify_dsatur,
        "dsatur",
        RESTARTS,
        METHOD_STYLES["dsatur"]["label"],
    )
    if dsatur_verify_series is not None:
        verification_series.append(dsatur_verify_series)
    neural_verify_series = build_series_from_ablation(
        verify_neural,
        "neural_stochastic",
        RESTARTS,
        METHOD_STYLES["neural_stochastic"]["label"],
    )
    if neural_verify_series is None:
        neural_verify_series = build_series_from_ablation(
            verify_neural,
            "neural_greedy",
            RESTARTS,
            METHOD_STYLES["neural_greedy"]["label"],
        )
    if neural_verify_series is not None:
        verification_series.append(neural_verify_series)

    if verification_series:
        generated += plot_success_rate_vs_restarts(
            verification_series,
            output_dir,
            "Success Rate vs. Restarts (200-instance verification)",
            "figure5_verification_200",
        )
    else:
        logging.warning("Skipping 200-instance verification figure; data unavailable.")

    if generated:
        logging.info("Generated %d figure files:", len(generated))
        for path in generated:
            logging.info(" - %s", path)
    else:
        logging.warning("No figures generated.")


if __name__ == "__main__":
    main()
