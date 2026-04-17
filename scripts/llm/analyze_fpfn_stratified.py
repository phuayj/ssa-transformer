#!/usr/bin/env python3
"""Stratify FP/FN decomposition by correct-candidate count.

Uses oracle correctness labels to inject FP/FN noise, selects the top-scoring
noisy-correct candidate, and reports solve rates by candidate-saturation bins.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


NOISE_RATES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
SEEDS = list(range(10))

BIN_DEFINITIONS: List[Tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-12", 8, 12),
    ("13-16", 13, 16),
]

MODE_LABELS = {
    "fp_only": "FP-only",
    "fn_only": "FN-only",
    "symmetric": "Symmetric",
}

MODE_COLORS = {
    "fp_only": "#1f77b4",
    "fn_only": "#d62728",
    "symmetric": "#2ca02c",
}

MODE_MARKERS = {
    "fp_only": "o",
    "fn_only": "s",
    "symmetric": "^",
}


@dataclass(frozen=True)
class ProblemData:
    idx: int
    scores: np.ndarray
    labels: np.ndarray

    @property
    def num_correct(self) -> int:
        return int(np.sum(self.labels))


@dataclass(frozen=True)
class CandidateLine:
    idx: Optional[int]
    ground_truth: Any
    labels: List[int]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_eval(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    per_problem = data.get("per_problem", [])
    if not per_problem:
        raise ValueError(f"Missing per_problem in {path}")
    return per_problem


def _load_candidates_jsonl_by_line(
    path: Path, dataset_name: str
) -> List[CandidateLine]:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates file: {path}")
    entries: List[CandidateLine] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                raise ValueError(f"Empty line in {path} at line {line_num}")
            record = json.loads(line)
            candidates = record.get("candidates")
            if candidates is None:
                raise ValueError(f"Missing candidates list in {path} line {line_num}")
            labels = [int(c.get("correct", 0)) for c in candidates]
            ground_truth = record.get("ground_truth")
            if ground_truth is None:
                raise ValueError(f"Missing ground_truth in {path} line {line_num}")
            idx_value = record.get("idx")
            idx = int(idx_value) if idx_value is not None else None
            entries.append(
                CandidateLine(idx=idx, ground_truth=ground_truth, labels=labels)
            )
    if not entries:
        raise ValueError(f"No candidate lines found in {path}")
    logger.info(
        "%s: loaded %d candidate lines from %s", dataset_name, len(entries), path
    )
    return entries


def _labels_from_candidates_by_position(
    per_problem: Sequence[dict],
    candidates: Sequence[CandidateLine],
    dataset_name: str,
) -> List[List[int]]:
    if len(candidates) != len(per_problem):
        raise ValueError(
            f"{dataset_name} candidates count {len(candidates)} != eval count {len(per_problem)}"
        )
    labels_by_position: List[List[int]] = []
    for position, (eval_entry, cand_entry) in enumerate(zip(per_problem, candidates)):
        eval_gt = eval_entry.get("ground_truth")
        if eval_gt is None:
            raise ValueError(
                f"{dataset_name} eval missing ground_truth at position {position}"
            )
        cand_gt = cand_entry.ground_truth
        if eval_gt != cand_gt:
            eval_idx = eval_entry.get("idx")
            raise ValueError(
                f"{dataset_name} ground_truth mismatch at position {position}: "
                f"eval_idx={eval_idx} cand_idx={cand_entry.idx} "
                f"eval_gt={eval_gt!r} cand_gt={cand_gt!r}"
            )
        labels_by_position.append(cand_entry.labels)
    return labels_by_position


def _build_problem_data(
    per_problem: Sequence[dict], labels_by_position: Sequence[Sequence[int]]
) -> List[ProblemData]:
    if len(per_problem) != len(labels_by_position):
        raise ValueError(
            f"Label count {len(labels_by_position)} != eval count {len(per_problem)}"
        )
    problems: List[ProblemData] = []
    for position, entry in enumerate(per_problem):
        idx_value = entry.get("idx")
        if idx_value is None:
            raise ValueError(f"Missing idx at position {position}")
        idx = int(idx_value)
        labels = labels_by_position[position]
        scores = np.asarray(entry.get("all_scores", []), dtype=float)
        labels_array = np.asarray(labels, dtype=int)
        if scores.shape[0] != labels_array.shape[0]:
            raise ValueError(
                f"Length mismatch position={position} idx={idx}: scores={scores.shape[0]} labels={labels_array.shape[0]}"
            )
        if not np.isin(labels_array, [0, 1]).all():
            raise ValueError(f"Non-binary labels position={position} idx={idx}")
        problems.append(ProblemData(idx=idx, scores=scores, labels=labels_array))
    return problems


def _rate_key(rate: float) -> str:
    label = f"{float(rate):.2f}".rstrip("0").rstrip(".")
    return "0.0" if label == "0" else label


def _inject_noise(
    labels: np.ndarray, rate: float, mode: str, rng: np.random.Generator
) -> np.ndarray:
    noisy = labels.copy()
    if mode in {"fp_only", "symmetric"}:
        fp_mask = (labels == 0) & (rng.random(labels.shape[0]) < rate)
        noisy[fp_mask] = 1
    if mode in {"fn_only", "symmetric"}:
        fn_mask = (labels == 1) & (rng.random(labels.shape[0]) < rate)
        noisy[fn_mask] = 0
    return noisy


def _fpfn_decomposition(
    problems: Sequence[ProblemData],
    noise_rates: Sequence[float],
    seeds: Sequence[int],
    fallback_mode: str = "random",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for mode in ["fp_only", "fn_only", "symmetric"]:
        mode_results: Dict[str, Dict[str, Any]] = {}
        for rate in noise_rates:
            per_seed: List[float] = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                solved: List[float] = []
                for problem in problems:
                    noisy_labels = _inject_noise(problem.labels, rate, mode, rng)
                    if noisy_labels.sum() > 0:
                        masked_scores = np.where(
                            noisy_labels == 1, problem.scores, -np.inf
                        )
                        pick_idx = int(np.argmax(masked_scores))
                        solved.append(float(problem.labels[pick_idx]))
                    else:
                        if fallback_mode == "random":
                            pick_idx = int(rng.integers(0, problem.labels.shape[0]))
                            solved.append(float(problem.labels[pick_idx]))
                        else:
                            raise ValueError(
                                f"Unsupported fallback_mode: {fallback_mode}"
                            )
                per_seed.append(float(np.mean(solved)) if solved else float("nan"))
            rate_key = _rate_key(rate)
            mode_results[rate_key] = {
                "solve_rate_mean": float(np.nanmean(per_seed)) if per_seed else None,
                "solve_rate_std": float(np.nanstd(per_seed)) if per_seed else None,
                "per_seed": per_seed,
            }
        results[mode] = mode_results
    return results


def _assign_bins(
    problems: Sequence[ProblemData],
    bin_definitions: Sequence[Tuple[str, int, int]],
) -> Dict[str, List[ProblemData]]:
    bin_map: Dict[str, List[ProblemData]] = {name: [] for name, _, _ in bin_definitions}
    for problem in problems:
        assigned = False
        for name, low, high in bin_definitions:
            if low <= problem.num_correct <= high:
                bin_map[name].append(problem)
                assigned = True
                break
        if not assigned:
            raise ValueError(
                f"Problem idx={problem.idx} has num_correct={problem.num_correct} outside bin ranges"
            )
    return bin_map


def _plot_stratified_curves(
    results: Dict[str, dict],
    output_path: Path,
    bin_definitions: Sequence[Tuple[str, int, int]],
    noise_rates: Sequence[float],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for idx, (name, low, high) in enumerate(bin_definitions):
        ax = axes[idx]
        bin_data = results["bins"][name]
        metrics = bin_data["metrics"]
        for mode in ["fp_only", "fn_only", "symmetric"]:
            y_values = [
                metrics[mode][_rate_key(rate)]["solve_rate_mean"]
                for rate in noise_rates
            ]
            ax.plot(
                noise_rates,
                y_values,
                marker=MODE_MARKERS[mode],
                color=MODE_COLORS[mode],
                linewidth=1.6,
                label=MODE_LABELS[mode],
            )
        n = bin_data["problem_count"]
        ax.set_title(f"{name} correct (n={n})")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel("Solve rate")
        if idx >= 3:
            ax.set_xlabel("Noise rate ε")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_fp_summary(
    results: Dict[str, dict],
    output_path: Path,
    bin_definitions: Sequence[Tuple[str, int, int]],
    noise_rates: Sequence[float],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    color_map = plt.cm.get_cmap("tab10", len(bin_definitions))
    for idx, (name, low, high) in enumerate(bin_definitions):
        bin_data = results["bins"][name]
        metrics = bin_data["metrics"]["fp_only"]
        y_values = [metrics[_rate_key(rate)]["solve_rate_mean"] for rate in noise_rates]
        ax.plot(
            noise_rates,
            y_values,
            marker="o",
            linewidth=1.8,
            color=color_map(idx),
            label=f"{name} ({low}-{high})",
        )
    ax.set_xlabel("Noise rate ε")
    ax.set_ylabel("Solve rate (FP-only)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _print_summary_table(
    results: Dict[str, dict],
    bin_definitions: Sequence[Tuple[str, int, int]],
    noise_rates: Sequence[float],
) -> None:
    def fmt(value: Optional[float]) -> str:
        if value is None or np.isnan(value):
            return "nan"
        return f"{value:.3f}"

    print("FP/FN decomposition stratified by correct-candidate count")
    for name, low, high in bin_definitions:
        bin_data = results["bins"][name]
        metrics = bin_data["metrics"]
        print("")
        print(f"Bin {name} ({low}-{high} correct) | n={bin_data['problem_count']}")
        print("epsilon  fp_only  fn_only  symmetric")
        for rate in noise_rates:
            key = _rate_key(rate)
            fp_val = fmt(metrics["fp_only"][key]["solve_rate_mean"])
            fn_val = fmt(metrics["fn_only"][key]["solve_rate_mean"])
            sym_val = fmt(metrics["symmetric"][key]["solve_rate_mean"])
            print(f"{rate:>5.2f}   {fp_val:>7}  {fn_val:>7}  {sym_val:>9}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified FP/FN decomposition by correct-candidate count",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("/workspace/experiments/math-14b/eval_full_5000.json"),
        help="Path to eval JSON with per_problem/all_scores",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=Path("/workspace/experiments/math-14b/candidates_test_full_5000.jsonl"),
        help="Path to candidates JSONL with correctness labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/experiments/llm-precision-cliff"),
        help="Directory for output JSON/PDFs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_style()

    per_problem = _load_eval(args.eval_file)
    candidates = _load_candidates_jsonl_by_line(args.candidates_file, "math-14b")
    labels_by_position = _labels_from_candidates_by_position(
        per_problem, candidates, "math-14b"
    )
    problems = _build_problem_data(per_problem, labels_by_position)

    num_correct = np.asarray([p.num_correct for p in problems], dtype=int)
    num_candidates = np.asarray([p.scores.shape[0] for p in problems], dtype=int)
    logger.info("Loaded %d problems", len(problems))
    logger.info(
        "Candidate count stats: min=%d max=%d mean=%.2f",
        int(num_candidates.min()),
        int(num_candidates.max()),
        float(np.mean(num_candidates)),
    )
    logger.info(
        "Correct-count stats: min=%d max=%d mean=%.2f",
        int(num_correct.min()),
        int(num_correct.max()),
        float(np.mean(num_correct)),
    )
    correct_hist = {
        int(k): int(v) for k, v in zip(*np.unique(num_correct, return_counts=True))
    }
    logger.info("Correct-count distribution: %s", correct_hist)

    bin_map = _assign_bins(problems, BIN_DEFINITIONS)
    for name, low, high in BIN_DEFINITIONS:
        count = len(bin_map[name])
        logger.info("Bin %s (%d-%d): %d problems", name, low, high, count)

    results: Dict[str, dict] = {
        "config": {
            "eval_file": str(args.eval_file),
            "candidates_file": str(args.candidates_file),
            "noise_rates": [float(r) for r in NOISE_RATES],
            "seeds": [int(s) for s in SEEDS],
            "bins": [
                {"name": name, "min_correct": low, "max_correct": high}
                for name, low, high in BIN_DEFINITIONS
            ],
        },
        "bins": {},
    }

    for name, low, high in BIN_DEFINITIONS:
        problems_bin = bin_map[name]
        metrics = _fpfn_decomposition(problems_bin, NOISE_RATES, SEEDS)
        results["bins"][name] = {
            "problem_count": len(problems_bin),
            "correct_range": [low, high],
            "metrics": metrics,
        }
        baseline_key = _rate_key(0.0)
        for mode in ["fp_only", "fn_only", "symmetric"]:
            baseline = metrics[mode][baseline_key]["solve_rate_mean"]
            logger.info(
                "Bin %s baseline %s: %.3f",
                name,
                MODE_LABELS[mode],
                float(baseline) if baseline is not None else float("nan"),
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "fpfn_stratified_14b.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    logger.info("Saved JSON results to %s", json_path)

    stratified_pdf = output_dir / "fpfn_stratified_14b.pdf"
    _plot_stratified_curves(results, stratified_pdf, BIN_DEFINITIONS, NOISE_RATES)
    logger.info("Saved stratified plot to %s", stratified_pdf)

    summary_pdf = output_dir / "fpfn_stratified_summary.pdf"
    _plot_fp_summary(results, summary_pdf, BIN_DEFINITIONS, NOISE_RATES)
    logger.info("Saved FP-only summary plot to %s", summary_pdf)

    _print_summary_table(results, BIN_DEFINITIONS, NOISE_RATES)


if __name__ == "__main__":
    main()
