#!/usr/bin/env python3
"""Analyze precision cliff + FP/FN decomposition for LLM verifier.

Loads verifier scores + candidate correctness labels, runs threshold sweep,
injects FP/FN noise against oracle labels, and evaluates calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


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
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _safe_div(numer: float, denom: float) -> Optional[float]:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


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


def _find_candidates_file(search_dir: Path) -> Optional[Path]:
    matches = sorted(search_dir.glob("candidates_test*.jsonl"))
    if not matches:
        return None
    return matches[0]


def _reconstruct_labels_from_oracle(
    per_problem: Sequence[dict],
) -> List[List[int]]:
    labels_by_position: List[List[int]] = []
    for position, entry in enumerate(per_problem):
        idx_value = entry.get("idx")
        idx = int(idx_value) if idx_value is not None else position
        scores = np.asarray(entry.get("all_scores", []), dtype=float)
        k = int(scores.shape[0])
        labels = np.zeros(k, dtype=int)
        oracle = entry.get("oracle", 0)
        if isinstance(oracle, list):
            for cand_idx in oracle:
                cand_idx = int(cand_idx)
                if cand_idx < 0 or cand_idx >= k:
                    raise ValueError(
                        f"Oracle index out of range: idx={idx} cand_idx={cand_idx}"
                    )
                labels[cand_idx] = 1
        elif isinstance(oracle, (int, float)):
            count = int(oracle)
            if count < 0:
                raise ValueError(f"Negative oracle count for idx={idx}: {oracle}")
            if count > 0:
                if count > k:
                    raise ValueError(f"Oracle count {count} > K={k} for idx={idx}")
                top_indices = np.argsort(scores)[::-1][:count]
                labels[top_indices] = 1
        else:
            raise TypeError(f"Unsupported oracle type for idx={idx}: {type(oracle)}")
        labels_by_position.append(labels.tolist())
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
        idx = int(entry["idx"])
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


def _threshold_sweep(
    problems: Sequence[ProblemData],
    thresholds: Sequence[float],
    fallback_mode: str = "random_expected",
) -> Dict[str, List[Optional[float]]]:
    results: Dict[str, List[Optional[float]]] = {
        "thresholds": [],
        "candidate_precision": [],
        "candidate_recall": [],
        "solve_rate": [],
        "fp_rate": [],
        "fn_rate": [],
        "accept_rate": [],
    }
    total_problems = float(len(problems))

    for threshold in thresholds:
        accepted_correct = 0
        accepted_total = 0
        total_correct = 0
        total_incorrect = 0
        solve_sum = 0.0

        for problem in problems:
            scores = problem.scores
            labels = problem.labels
            accepted_mask = scores >= threshold

            if accepted_mask.any():
                masked_scores = np.where(accepted_mask, scores, -np.inf)
                pick_idx = int(np.argmax(masked_scores))
                solve_sum += float(labels[pick_idx])
            else:
                if fallback_mode == "best":
                    pick_idx = int(np.argmax(scores))
                    solve_sum += float(labels[pick_idx])
                elif fallback_mode == "random_expected":
                    solve_sum += float(np.mean(labels))
                else:
                    raise ValueError(f"Unsupported fallback_mode: {fallback_mode}")

            if problem.num_correct > 0:
                total_correct += problem.num_correct
                total_incorrect += int(labels.shape[0]) - problem.num_correct
                accepted_total += int(np.sum(accepted_mask))
                accepted_correct += int(np.sum(labels[accepted_mask]))

        accepted_incorrect = accepted_total - accepted_correct
        rejected_correct = total_correct - accepted_correct

        precision = _safe_div(accepted_correct, accepted_total)
        recall = _safe_div(accepted_correct, total_correct)
        fp_rate = _safe_div(accepted_incorrect, total_incorrect)
        fn_rate = _safe_div(rejected_correct, total_correct)
        accept_rate = _safe_div(accepted_total, total_correct + total_incorrect)

        results["thresholds"].append(float(threshold))
        results["candidate_precision"].append(precision)
        results["candidate_recall"].append(recall)
        results["solve_rate"].append(float(solve_sum) / total_problems)
        results["fp_rate"].append(fp_rate)
        results["fn_rate"].append(fn_rate)
        results["accept_rate"].append(accept_rate)

    return results


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
) -> Dict[str, dict]:
    results: Dict[str, dict] = {
        "config": {
            "noise_rates": [float(r) for r in noise_rates],
            "seeds": [int(s) for s in seeds],
            "fallback_mode": fallback_mode,
            "problem_count": int(len(problems)),
        }
    }

    for mode in ["fp_only", "fn_only", "symmetric"]:
        mode_results: Dict[str, dict] = {}
        for rate in noise_rates:
            per_seed: List[float] = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                solved = []
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
                        elif fallback_mode == "best":
                            pick_idx = int(np.argmax(problem.scores))
                            solved.append(float(problem.labels[pick_idx]))
                        elif fallback_mode == "random_expected":
                            solved.append(float(np.mean(problem.labels)))
                        else:
                            raise ValueError(
                                f"Unsupported fallback_mode: {fallback_mode}"
                            )
                per_seed.append(float(np.mean(solved)))

            rate_key = _rate_key(rate)
            mode_results[rate_key] = {
                "solve_rate_mean": float(np.mean(per_seed)),
                "solve_rate_std": float(np.std(per_seed)),
                "per_seed": per_seed,
            }
        results[mode] = mode_results
    return results


def _rate_key(rate: float) -> str:
    label = f"{float(rate):.2f}".rstrip("0").rstrip(".")
    return "0.0" if label == "0" else label


def _calibration_bins(
    problems: Sequence[ProblemData], num_bins: int = 10
) -> Dict[str, object]:
    scores = np.concatenate([p.scores for p in problems]).astype(float)
    labels = np.concatenate([p.labels for p in problems]).astype(int)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(scores, edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bins: List[dict] = []
    for i in range(num_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        correct = int(np.sum(labels[mask]))
        accuracy = _safe_div(correct, count)
        avg_score = float(np.mean(scores[mask])) if count > 0 else None
        bins.append(
            {
                "bin": int(i),
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "count": count,
                "correct": correct,
                "accuracy": accuracy,
                "avg_score": avg_score,
            }
        )
    return {"bins": bins, "edges": [float(e) for e in edges]}


def _summarize_thresholds(results: Dict[str, List[Optional[float]]]) -> dict:
    thresholds = np.asarray(results["thresholds"], dtype=float)
    solve_rates = np.asarray(results["solve_rate"], dtype=float)
    precision = np.asarray(
        [np.nan if p is None else float(p) for p in results["candidate_precision"]]
    )
    recall = np.asarray(
        [np.nan if r is None else float(r) for r in results["candidate_recall"]]
    )

    if solve_rates.size == 0:
        raise ValueError("Empty threshold sweep results")

    max_solve = float(np.nanmax(solve_rates))
    best_indices = np.where(solve_rates == max_solve)[0]
    best_idx = int(best_indices[0])

    diffs = np.diff(solve_rates)
    if diffs.size > 0:
        cliff_idx = int(np.argmin(diffs))
        cliff_threshold = float(thresholds[cliff_idx + 1])
        cliff_drop = float(diffs[cliff_idx])
    else:
        cliff_threshold = None
        cliff_drop = None

    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "max_solve_rate": max_solve,
        "precision_at_optimal": None
        if math.isnan(precision[best_idx])
        else float(precision[best_idx]),
        "recall_at_optimal": None
        if math.isnan(recall[best_idx])
        else float(recall[best_idx]),
        "precision_cliff_threshold": cliff_threshold,
        "precision_cliff_drop": cliff_drop,
    }


def _plot_precision_cliff(
    results: Dict[str, List[Optional[float]]],
    output_path: Path,
    title: str,
) -> None:
    thresholds = np.asarray(results["thresholds"], dtype=float)
    solve_rate = np.asarray(results["solve_rate"], dtype=float)
    precision = np.asarray(
        [np.nan if v is None else float(v) for v in results["candidate_precision"]]
    )
    recall = np.asarray(
        [np.nan if v is None else float(v) for v in results["candidate_recall"]]
    )

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(
        thresholds,
        solve_rate,
        color="C0",
        linewidth=2.0,
        label="Solve rate",
    )
    ax1.set_xlabel("Threshold τ")
    ax1.set_ylabel("Solve rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        thresholds,
        precision,
        color="C1",
        linestyle="--",
        linewidth=1.5,
        label="Candidate precision",
    )
    ax2.plot(
        thresholds,
        recall,
        color="C2",
        linestyle=":",
        linewidth=1.5,
        label="Candidate recall",
    )
    ax2.set_ylabel("Candidate precision / recall")
    ax2.set_ylim(0.0, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(handles=lines, labels=labels, loc="lower left", frameon=False)
    ax1.set_title(title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved precision cliff plot to %s", output_path)


def _plot_combined_precision(
    results_a: Dict[str, List[Optional[float]]],
    label_a: str,
    results_b: Dict[str, List[Optional[float]]],
    label_b: str,
    output_path: Path,
) -> None:
    thresholds_a = np.asarray(results_a["thresholds"], dtype=float)
    solve_a = np.asarray(results_a["solve_rate"], dtype=float)
    thresholds_b = np.asarray(results_b["thresholds"], dtype=float)
    solve_b = np.asarray(results_b["solve_rate"], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        thresholds_a,
        solve_a,
        color="C0",
        linewidth=2.0,
        label=label_a,
    )
    ax.plot(
        thresholds_b,
        solve_b,
        color="C3",
        linewidth=2.0,
        linestyle="--",
        label=label_b,
    )
    ax.set_xlabel("Threshold τ")
    ax.set_ylabel("Solve rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Precision cliff comparison")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined precision cliff plot to %s", output_path)


def _plot_fpfn_decomposition(
    results: Dict[str, dict],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, style in [
        ("fp_only", {"color": "C0", "marker": "o"}),
        ("fn_only", {"color": "C1", "marker": "s"}),
        ("symmetric", {"color": "C2", "marker": "^"}),
    ]:
        mode_data = results.get(mode, {})
        rates = sorted(float(k) for k in mode_data.keys())
        solve_rates = [float(mode_data[_rate_key(r)]["solve_rate_mean"]) for r in rates]
        ax.plot(
            rates,
            solve_rates,
            label=mode.replace("_", " "),
            linewidth=1.8,
            marker=style["marker"],
            color=style["color"],
        )
    ax.set_xlabel("Noise rate ε")
    ax.set_ylabel("Solve rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved FP/FN decomposition plot to %s", output_path)


def _plot_calibration(calibration: Dict[str, object], output_path: Path) -> None:
    bins = calibration["bins"]
    xs = []
    ys = []
    for entry in bins:
        if entry["count"] <= 0 or entry["accuracy"] is None:
            continue
        lower = float(entry["lower"])
        upper = float(entry["upper"])
        midpoint = (lower + upper) / 2.0
        xs.append(midpoint)
        ys.append(float(entry["accuracy"]))

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    ax.plot(xs, ys, marker="o", linewidth=1.5, color="C0")
    ax.set_xlabel("Verifier score (bin midpoint)")
    ax.set_ylabel("Fraction correct")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_title("Verifier calibration")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration plot to %s", output_path)


def _log_dataset_stats(name: str, problems: Sequence[ProblemData]) -> dict:
    num_problems = int(len(problems))
    num_candidates = int(sum(int(p.labels.shape[0]) for p in problems))
    oracle_count = int(sum(1 for p in problems if p.num_correct > 0))
    correct_total = int(sum(p.num_correct for p in problems))
    score_min = float(min(np.min(p.scores) for p in problems))
    score_max = float(max(np.max(p.scores) for p in problems))
    score_mean = float(np.mean([float(np.mean(p.scores)) for p in problems]))

    logger.info(
        "%s: problems=%d candidates=%d oracle_coverage=%.3f correct_candidates=%d",
        name,
        num_problems,
        num_candidates,
        float(oracle_count) / float(num_problems) if num_problems > 0 else 0.0,
        correct_total,
    )
    logger.info(
        "%s: score range [%.4f, %.4f] mean=%.4f",
        name,
        score_min,
        score_max,
        score_mean,
    )

    return {
        "num_problems": num_problems,
        "num_candidates": num_candidates,
        "oracle_coverage": float(oracle_count) / float(num_problems)
        if num_problems > 0
        else 0.0,
        "correct_candidates": correct_total,
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
    }


def _log_example_problem(name: str, problems: Sequence[ProblemData]) -> None:
    if not problems:
        return
    example = problems[0]
    top_idx = int(np.argmax(example.scores))
    logger.info(
        "%s example idx=%d top_idx=%d top_score=%.4f top_label=%d num_correct=%d",
        name,
        example.idx,
        top_idx,
        float(example.scores[top_idx]),
        int(example.labels[top_idx]),
        example.num_correct,
    )


def _log_threshold_snapshot(
    name: str, results: Dict[str, List[Optional[float]]], target: float
) -> None:
    thresholds = np.asarray(results["thresholds"], dtype=float)
    if thresholds.size == 0:
        return
    idx = int(np.argmin(np.abs(thresholds - target)))
    precision = results["candidate_precision"][idx]
    recall = results["candidate_recall"][idx]
    fp_rate = results["fp_rate"][idx]
    fn_rate = results["fn_rate"][idx]
    logger.info(
        "%s threshold %.2f snapshot: solve_rate=%.3f precision=%s recall=%s fp_rate=%s fn_rate=%s",
        name,
        float(thresholds[idx]),
        float(results["solve_rate"][idx]),
        f"{precision:.3f}" if precision is not None else "n/a",
        f"{recall:.3f}" if recall is not None else "n/a",
        f"{fp_rate:.3f}" if fp_rate is not None else "n/a",
        f"{fn_rate:.3f}" if fn_rate is not None else "n/a",
    )


def _print_summary_table(summaries: Dict[str, dict]) -> None:
    headers = [
        "model",
        "max_solve",
        "opt_thresh",
        "precision@opt",
        "recall@opt",
        "cliff_thresh",
        "cliff_drop",
    ]
    row_fmt = "{:<10} {:>9} {:>10} {:>13} {:>10} {:>12} {:>10}"
    print(row_fmt.format(*headers))
    print("-" * 78)
    for model_name, summary in summaries.items():
        if model_name == "config":
            continue
        row = [
            model_name,
            f"{summary['max_solve_rate']:.3f}",
            f"{summary['optimal_threshold']:.2f}",
            f"{summary['precision_at_optimal']:.3f}"
            if summary["precision_at_optimal"] is not None
            else "n/a",
            f"{summary['recall_at_optimal']:.3f}"
            if summary["recall_at_optimal"] is not None
            else "n/a",
            f"{summary['precision_cliff_threshold']:.2f}"
            if summary["precision_cliff_threshold"] is not None
            else "n/a",
            f"{summary['precision_cliff_drop']:.3f}"
            if summary["precision_cliff_drop"] is not None
            else "n/a",
        ]
        print(row_fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze precision cliff for LLM verifier"
    )
    parser.add_argument(
        "--eval-14b",
        type=str,
        default="/workspace/experiments/math-14b/eval_full_5000.json",
    )
    parser.add_argument(
        "--candidates-14b",
        type=str,
        default="/workspace/experiments/math-14b/candidates_test_full_5000.jsonl",
    )
    parser.add_argument(
        "--eval-qwen",
        type=str,
        default="/workspace/experiments/qwen-4b/eval_test.json",
    )
    parser.add_argument(
        "--candidates-qwen",
        type=str,
        default="/workspace/experiments/qwen-4b/candidates_test_full.jsonl",
        help="Qwen candidates JSONL path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/experiments/llm-precision-cliff",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--noise-seeds",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    _setup_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(0.0, 1.0 + args.threshold_step / 2.0, args.threshold_step)
    noise_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    seeds = list(range(args.noise_seeds))

    # Load 14B data
    eval_14b = _load_eval(Path(args.eval_14b))
    candidates_14b = _load_candidates_jsonl_by_line(
        Path(args.candidates_14b), "Ministral-14B"
    )
    labels_14b = _labels_from_candidates_by_position(
        eval_14b, candidates_14b, "Ministral-14B"
    )
    label_counts_14b = {
        "per_candidate": len(labels_14b),
        "reconstructed": 0,
    }
    logger.info(
        "Ministral-14B labels: per_candidate=%d reconstructed=%d",
        label_counts_14b["per_candidate"],
        label_counts_14b["reconstructed"],
    )
    problems_14b = _build_problem_data(eval_14b, labels_14b)
    stats_14b = _log_dataset_stats("Ministral-14B", problems_14b)
    _log_example_problem("Ministral-14B", problems_14b)

    # Load Qwen data
    eval_qwen = _load_eval(Path(args.eval_qwen))
    qwen_candidates_path = (
        Path(args.candidates_qwen)
        if args.candidates_qwen is not None
        else _find_candidates_file(Path(args.eval_qwen).parent)
    )
    label_counts_qwen = {"per_candidate": 0, "reconstructed": 0}
    if qwen_candidates_path and qwen_candidates_path.exists():
        candidates_qwen = _load_candidates_jsonl_by_line(
            qwen_candidates_path, "Qwen3-4B"
        )
        if len(candidates_qwen) != len(eval_qwen):
            logger.warning(
                "Qwen candidates count %d != eval count %d from %s; reconstructing labels from oracle",
                len(candidates_qwen),
                len(eval_qwen),
                qwen_candidates_path,
            )
            labels_qwen = _reconstruct_labels_from_oracle(eval_qwen)
            label_source_qwen = "oracle_reconstruction_count_mismatch"
            label_counts_qwen["reconstructed"] = len(labels_qwen)
        else:
            labels_qwen = _labels_from_candidates_by_position(
                eval_qwen, candidates_qwen, "Qwen3-4B"
            )
            label_source_qwen = str(qwen_candidates_path)
            label_counts_qwen["per_candidate"] = len(labels_qwen)
            logger.info("Qwen candidates loaded from %s", qwen_candidates_path)
    else:
        labels_qwen = _reconstruct_labels_from_oracle(eval_qwen)
        label_source_qwen = "oracle_reconstruction_missing_candidates"
        label_counts_qwen["reconstructed"] = len(labels_qwen)
        logger.warning(
            "Qwen candidates missing, reconstructing labels from oracle field"
        )
        logger.warning(
            "Oracle reconstruction assumes oracle=count of correct candidates and marks top-K by score"
        )
    logger.info(
        "Qwen3-4B labels: per_candidate=%d reconstructed=%d",
        label_counts_qwen["per_candidate"],
        label_counts_qwen["reconstructed"],
    )
    problems_qwen = _build_problem_data(eval_qwen, labels_qwen)
    stats_qwen = _log_dataset_stats("Qwen3-4B", problems_qwen)
    _log_example_problem("Qwen3-4B", problems_qwen)

    # Threshold sweep
    threshold_14b = _threshold_sweep(problems_14b, thresholds)
    threshold_qwen = _threshold_sweep(problems_qwen, thresholds)
    _log_threshold_snapshot("Ministral-14B", threshold_14b, 0.5)
    _log_threshold_snapshot("Qwen3-4B", threshold_qwen, 0.5)

    # FP/FN decomposition
    fpfn_14b = _fpfn_decomposition(problems_14b, noise_rates, seeds)
    fpfn_qwen = _fpfn_decomposition(problems_qwen, noise_rates, seeds)

    # Calibration (14B only)
    calibration_14b = _calibration_bins(problems_14b, num_bins=10)

    # Save JSON outputs
    (output_dir / "threshold_sweep_14b.json").write_text(
        json.dumps(threshold_14b, indent=2), encoding="utf-8"
    )
    (output_dir / "threshold_sweep_qwen4b.json").write_text(
        json.dumps(threshold_qwen, indent=2), encoding="utf-8"
    )
    (output_dir / "fpfn_decomposition_14b.json").write_text(
        json.dumps(fpfn_14b, indent=2), encoding="utf-8"
    )
    (output_dir / "fpfn_decomposition_qwen4b.json").write_text(
        json.dumps(fpfn_qwen, indent=2), encoding="utf-8"
    )
    (output_dir / "calibration_14b.json").write_text(
        json.dumps(calibration_14b, indent=2), encoding="utf-8"
    )

    # Summary JSON
    summary_14b = _summarize_thresholds(threshold_14b)
    summary_qwen = _summarize_thresholds(threshold_qwen)
    logger.info(
        "Ministral-14B optimal τ=%.2f max_solve=%.3f precision=%.3f recall=%.3f",
        summary_14b["optimal_threshold"],
        summary_14b["max_solve_rate"],
        summary_14b["precision_at_optimal"]
        if summary_14b["precision_at_optimal"] is not None
        else float("nan"),
        summary_14b["recall_at_optimal"]
        if summary_14b["recall_at_optimal"] is not None
        else float("nan"),
    )
    logger.info(
        "Qwen3-4B optimal τ=%.2f max_solve=%.3f precision=%.3f recall=%.3f",
        summary_qwen["optimal_threshold"],
        summary_qwen["max_solve_rate"],
        summary_qwen["precision_at_optimal"]
        if summary_qwen["precision_at_optimal"] is not None
        else float("nan"),
        summary_qwen["recall_at_optimal"]
        if summary_qwen["recall_at_optimal"] is not None
        else float("nan"),
    )
    summary = {
        "14b": {
            **summary_14b,
            **stats_14b,
            "labels_source": str(Path(args.candidates_14b)),
            "labels_per_candidate": label_counts_14b["per_candidate"],
            "labels_reconstructed": label_counts_14b["reconstructed"],
        },
        "qwen4b": {
            **summary_qwen,
            **stats_qwen,
            "labels_source": label_source_qwen,
            "labels_per_candidate": label_counts_qwen["per_candidate"],
            "labels_reconstructed": label_counts_qwen["reconstructed"],
        },
        "config": {
            "threshold_step": float(args.threshold_step),
            "noise_rates": noise_rates,
            "noise_seeds": seeds,
            "threshold_fallback_mode": "random_expected",
            "fpfn_fallback_mode": "random",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Plots
    _plot_precision_cliff(
        threshold_14b,
        output_dir / "precision_cliff_14b.pdf",
        title="Ministral-14B precision cliff",
    )
    _plot_precision_cliff(
        threshold_qwen,
        output_dir / "precision_cliff_qwen4b.pdf",
        title="Qwen3-4B precision cliff",
    )
    _plot_fpfn_decomposition(
        fpfn_14b,
        output_dir / "fpfn_decomposition_14b.pdf",
        title="Ministral-14B FP/FN decomposition",
    )
    _plot_fpfn_decomposition(
        fpfn_qwen,
        output_dir / "fpfn_decomposition_qwen4b.pdf",
        title="Qwen3-4B FP/FN decomposition",
    )
    _plot_calibration(calibration_14b, output_dir / "calibration_14b.pdf")
    _plot_combined_precision(
        threshold_14b,
        "Ministral-14B",
        threshold_qwen,
        "Qwen3-4B",
        output_dir / "combined_precision_cliff.pdf",
    )

    # Print summary table
    _print_summary_table(summary)


if __name__ == "__main__":
    main()
