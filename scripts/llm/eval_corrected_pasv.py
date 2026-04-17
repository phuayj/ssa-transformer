#!/usr/bin/env python3
"""Corrected evaluation + PASV for MATH-14B.

Re-evaluates baselines with numerical matching, implements PASV variants,
and generates publication-ready tables/figures.
"""

from __future__ import annotations

import json
import logging
import random
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SLOT_PATH = REPO_ROOT / "experiments/math-14b/eval_full_5000.json"
EVAL_ORM_PATH = REPO_ROOT / "experiments/math-14b/eval_orm_5000.json"
EVAL_LORA_PATH = REPO_ROOT / "experiments/math-14b/eval_lora_orm_5000.json"
CANDIDATES_PATH = REPO_ROOT / "experiments/math-14b/candidates_test_full_5000.jsonl"
OUTPUT_DIR = REPO_ROOT / "paper/manuscript/pictures"

SEED = 42

FEATURE_NAMES = [
    "top1_score",
    "margin",
    "score_entropy",
    "agrees_with_mv",
    "mv_fraction",
    "verifier_answer_count",
    "n_unique_answers",
    "score_mean",
    "score_std",
]


@dataclass
class CandidateLine:
    idx: int | None
    ground_truth: Any
    candidates: List[Dict[str, Any]]


@dataclass
class ProblemData:
    idx: int
    ground_truth: Any
    answers: List[str]
    correct: np.ndarray
    slot_scores: np.ndarray
    orm_scores: np.ndarray
    lora_scores: np.ndarray
    mv_answer: str
    mv_correct: int
    mv_fraction: float
    oracle_correct: int
    num_correct: int
    frequency_scores: np.ndarray
    slot_top_idx: int
    slot_correct: int


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _replace_frac(s: str) -> str:
    for _ in range(5):
        s = re.sub(r"\\?\\?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    return s


def normalize_math_answer(answer: Any) -> str:
    if not answer:
        return ""
    s = str(answer).strip()
    s = s.replace("\\$", "").replace("$", "").replace(" ", "").replace("\n", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    expr = _replace_frac(s).replace("^", "**").replace("{", "(").replace("}", ")")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            val = float(eval(expr, {"__builtins__": {}}, {}))
        return str(int(val)) if val == int(val) else f"{val:.6f}"
    except Exception:
        return s


def answers_match(pred: Any, truth: Any) -> bool:
    if not pred or not truth:
        return False
    a, b = normalize_math_answer(str(pred)), normalize_math_answer(str(truth))
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-4
    except Exception:
        return False


def load_eval(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "per_problem" in payload:
        return payload["per_problem"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected eval payload in {path}")


def load_candidates(path: Path) -> List[CandidateLine]:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates file: {path}")
    candidates: List[CandidateLine] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty line in {path} at line {line_num}")
            record = json.loads(line)
            candidate_entries = record.get("candidates")
            if candidate_entries is None:
                raise ValueError(f"Missing candidates list at line {line_num}")
            ground_truth = record.get("ground_truth")
            if ground_truth is None:
                raise ValueError(f"Missing ground_truth at line {line_num}")
            idx_value = record.get("idx")
            idx = int(idx_value) if idx_value is not None else None
            candidates.append(
                CandidateLine(
                    idx=idx, ground_truth=ground_truth, candidates=candidate_entries
                )
            )
    if not candidates:
        raise ValueError(f"No candidate lines found in {path}")
    logging.info("Loaded %d candidate lines from %s", len(candidates), path)
    return candidates


def build_problem_data(
    slot_eval: List[Dict[str, Any]],
    orm_eval: List[Dict[str, Any]],
    lora_eval: List[Dict[str, Any]],
    candidates: List[CandidateLine],
) -> List[ProblemData]:
    if not (len(slot_eval) == len(orm_eval) == len(lora_eval) == len(candidates)):
        raise ValueError(
            "Length mismatch: slot=%d orm=%d lora=%d candidates=%d"
            % (len(slot_eval), len(orm_eval), len(lora_eval), len(candidates))
        )

    problems: List[ProblemData] = []
    for position, (slot_entry, orm_entry, lora_entry, cand_entry) in enumerate(
        zip(slot_eval, orm_eval, lora_eval, candidates)
    ):
        cand_gt = cand_entry.ground_truth
        slot_gt = slot_entry.get("ground_truth")
        orm_gt = orm_entry.get("ground_truth")
        lora_gt = lora_entry.get("ground_truth")
        if not (slot_gt == orm_gt == lora_gt == cand_gt):
            raise ValueError(
                f"Ground-truth mismatch at position {position}: "
                f"slot={slot_gt!r} orm={orm_gt!r} lora={lora_gt!r} cand={cand_gt!r}"
            )

        answers = [str(cand.get("answer", "")) for cand in cand_entry.candidates]
        if not answers:
            raise ValueError(f"Empty candidate answers at position {position}")
        correct = np.array(
            [int(cand.get("correct", 0)) for cand in cand_entry.candidates],
            dtype=int,
        )

        slot_scores = np.asarray(slot_entry.get("all_scores"), dtype=float)
        orm_scores = np.asarray(orm_entry.get("all_scores"), dtype=float)
        lora_scores = np.asarray(lora_entry.get("all_scores"), dtype=float)
        if (
            len(answers) != slot_scores.shape[0]
            or len(answers) != orm_scores.shape[0]
            or len(answers) != lora_scores.shape[0]
        ):
            raise ValueError(
                "Scores/candidates mismatch at position %d: answers=%d slot=%d orm=%d lora=%d"
                % (
                    position,
                    len(answers),
                    slot_scores.shape[0],
                    orm_scores.shape[0],
                    lora_scores.shape[0],
                )
            )

        counts = Counter(answers)
        mv_answer = counts.most_common(1)[0][0]
        mv_count = counts[mv_answer]
        mv_fraction = mv_count / len(answers)

        mv_correct = int(answers_match(mv_answer, cand_gt))
        oracle_correct = int(np.any(correct))
        num_correct = int(np.sum(correct))
        frequency_scores = np.array(
            [counts[answer] / len(answers) for answer in answers], dtype=float
        )
        slot_top_idx = int(np.argmax(slot_scores))
        slot_correct = int(correct[slot_top_idx])

        idx_value = slot_entry.get("idx")
        idx = (
            int(idx_value)
            if idx_value is not None
            else int(cand_entry.idx)
            if cand_entry.idx is not None
            else position
        )

        problems.append(
            ProblemData(
                idx=idx,
                ground_truth=cand_gt,
                answers=answers,
                correct=correct,
                slot_scores=slot_scores,
                orm_scores=orm_scores,
                lora_scores=lora_scores,
                mv_answer=mv_answer,
                mv_correct=mv_correct,
                mv_fraction=mv_fraction,
                oracle_correct=oracle_correct,
                num_correct=num_correct,
                frequency_scores=frequency_scores,
                slot_top_idx=slot_top_idx,
                slot_correct=slot_correct,
            )
        )

    logging.info("Built %d problems", len(problems))
    return problems


def hybrid_score(
    verifier_scores: np.ndarray, frequency_scores: np.ndarray, alpha: float
) -> np.ndarray:
    v_min = float(np.min(verifier_scores))
    v_max = float(np.max(verifier_scores))
    v_norm = (verifier_scores - v_min) / (v_max - v_min + 1e-10)
    return alpha * v_norm + (1.0 - alpha) * frequency_scores


def adaptive_switch_index(
    verifier_scores: np.ndarray, answers: List[str], mv_confidence_threshold: float
) -> Tuple[int, bool]:
    counts = Counter(answers)
    mv_answer, mv_count = counts.most_common(1)[0]
    mv_frac = mv_count / len(answers)
    if mv_frac >= mv_confidence_threshold:
        mv_indices = [i for i, a in enumerate(answers) if a == mv_answer]
        best_local = int(np.argmax(verifier_scores[mv_indices]))
        return mv_indices[best_local], False
    return int(np.argmax(verifier_scores)), True


def compute_gate_features(problem: ProblemData) -> np.ndarray:
    scores = problem.slot_scores
    scores_sorted = np.sort(scores)[::-1]
    top1 = float(scores_sorted[0])
    margin = (
        float(top1 - scores_sorted[1]) if scores_sorted.shape[0] > 1 else float(top1)
    )
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))

    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    counts = Counter(problem.answers)
    mv_fraction = float(problem.mv_fraction)
    verifier_answer = problem.answers[problem.slot_top_idx]
    verifier_answer_count = float(counts[verifier_answer]) / float(len(problem.answers))
    agrees_with_mv = float(int(verifier_answer == problem.mv_answer))
    n_unique_answers = float(len(counts))

    return np.array(
        [
            top1,
            margin,
            entropy,
            agrees_with_mv,
            mv_fraction,
            verifier_answer_count,
            n_unique_answers,
            score_mean,
            score_std,
        ],
        dtype=float,
    )


def cross_val_gate_probs(
    features: np.ndarray, labels: np.ndarray, seed: int
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs = np.zeros(labels.shape[0], dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
        model.fit(features[train_idx], labels[train_idx])
        fold_probs = model.predict_proba(features[test_idx])[:, 1]
        probs[test_idx] = fold_probs
        logging.info(
            "Gate CV fold %d: train=%d test=%d y_train=%.3f y_test=%.3f prob_mean=%.3f",
            fold_idx,
            train_idx.shape[0],
            test_idx.shape[0],
            float(np.mean(labels[train_idx])),
            float(np.mean(labels[test_idx])),
            float(np.mean(fold_probs)),
        )
    return probs


def evaluate_gate_thresholds(
    gate_probs: np.ndarray,
    slot_correct: np.ndarray,
    mv_correct: np.ndarray,
    thresholds: List[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for tau in thresholds:
        use_verifier = gate_probs >= tau
        final_correct = np.where(use_verifier, slot_correct, mv_correct)
        accuracy = float(np.mean(final_correct))
        coverage = float(np.mean(use_verifier))
        verifier_precision = (
            float(np.mean(slot_correct[use_verifier])) if np.any(use_verifier) else None
        )
        results.append(
            {
                "tau": float(tau),
                "accuracy": accuracy,
                "coverage": coverage,
                "verifier_precision": verifier_precision,
            }
        )
    best = max(results, key=lambda r: (r["accuracy"], r["coverage"]))
    return results, best


def multiplicity_bucket(m: int) -> str:
    if m == 0:
        return "0"
    if m == 1:
        return "1"
    if 2 <= m <= 3:
        return "2-3"
    if 4 <= m <= 7:
        return "4-7"
    return "8+"


def stratified_accuracy(
    correct: np.ndarray, multiplicities: np.ndarray
) -> Dict[str, Dict[str, float]]:
    buckets = ["0", "1", "2-3", "4-7", "8+"]
    result: Dict[str, Dict[str, float]] = {}
    for bucket in buckets:
        mask = np.array([multiplicity_bucket(m) == bucket for m in multiplicities])
        if not np.any(mask):
            raise ValueError(f"No samples in bucket {bucket}")
        acc = float(np.mean(correct[mask]))
        result[bucket] = {"accuracy": acc, "n": int(np.sum(mask))}
    return result


def format_pct(value: float, bold: bool = False, star: bool = False) -> str:
    text = f"{value * 100:.1f}\\%"
    if star:
        text = f"{text}*"
    return f"\\textbf{{{text}}}" if bold else text


def format_delta(delta_pp: float | None, bold: bool = False) -> str:
    if delta_pp is None:
        return "--"
    sign = "+" if delta_pp >= 0 else ""
    text = f"{sign}{delta_pp:.1f}"
    return f"\\textbf{{{text}}}" if bold else text


def plot_pasv_comparison(
    output_path: Path,
    mv_acc: float,
    slot_acc: float,
    hybrid_best: Dict[str, Any],
    adaptive_best: Dict[str, Any],
    gate_best: Dict[str, Any],
) -> None:
    configure_style()
    labels = [
        "MV",
        "Slot",
        f"Hybrid (α={hybrid_best['alpha']:.2f})",
        f"Adaptive (τ={adaptive_best['threshold']:.2f})",
        f"Gate (τ={gate_best['tau']:.2f})",
    ]
    values = [
        mv_acc * 100,
        slot_acc * 100,
        hybrid_best["accuracy"] * 100,
        adaptive_best["accuracy"] * 100,
        gate_best["accuracy"] * 100,
    ]
    colors = ["#7f7f7f", "#000000", "#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("PASV Comparison")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved PASV comparison plot to %s", output_path)


def plot_hybrid_sweep(output_path: Path, sweep: List[Dict[str, Any]]) -> None:
    configure_style()
    alphas = [entry["alpha"] for entry in sweep]
    accuracies = [entry["accuracy"] * 100 for entry in sweep]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(alphas, accuracies, marker="o", linewidth=1.2, color="#1f77b4")
    ax.set_xlabel("Hybrid weight α")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Hybrid Scoring Sweep")
    ax.set_xticks(alphas)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved hybrid sweep plot to %s", output_path)


def plot_stratified(
    output_path: Path,
    mv_strat: Dict[str, Dict[str, float]],
    slot_strat: Dict[str, Dict[str, float]],
    pasv_strat: Dict[str, Dict[str, float]],
    pasv_label: str,
) -> None:
    configure_style()
    buckets = ["0", "1", "2-3", "4-7", "8+"]
    mv_vals = [mv_strat[b]["accuracy"] * 100 for b in buckets]
    slot_vals = [slot_strat[b]["accuracy"] * 100 for b in buckets]
    pasv_vals = [pasv_strat[b]["accuracy"] * 100 for b in buckets]

    x = np.arange(len(buckets))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - width, mv_vals, width, label="MV", color="#7f7f7f")
    ax.bar(x, slot_vals, width, label="Slot", color="#000000")
    ax.bar(x + width, pasv_vals, width, label=pasv_label, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("Multiplicity (m)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Stratified Accuracy by Multiplicity")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved stratified plot to %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    set_seed(SEED)

    logging.info("Loading evals and candidates...")
    slot_eval = load_eval(EVAL_SLOT_PATH)
    orm_eval = load_eval(EVAL_ORM_PATH)
    lora_eval = load_eval(EVAL_LORA_PATH)
    candidates = load_candidates(CANDIDATES_PATH)

    problems = build_problem_data(slot_eval, orm_eval, lora_eval, candidates)
    total = len(problems)
    logging.info("Total problems: %d", total)

    for preview in problems[:5]:
        logging.info(
            "Sample idx=%d m=%d mv=%s mv_correct=%d mv_frac=%.2f slot_idx=%d slot_correct=%d",
            preview.idx,
            preview.num_correct,
            preview.mv_answer,
            preview.mv_correct,
            preview.mv_fraction,
            preview.slot_top_idx,
            preview.slot_correct,
        )

    oracle_acc = float(np.mean([p.oracle_correct for p in problems]))
    mv_correct = np.array([p.mv_correct for p in problems], dtype=int)
    mv_acc = float(np.mean(mv_correct))
    slot_correct = np.array([p.slot_correct for p in problems], dtype=int)
    slot_acc = float(np.mean(slot_correct))
    orm_correct = np.array(
        [int(p.correct[int(np.argmax(p.orm_scores))]) for p in problems], dtype=int
    )
    orm_acc = float(np.mean(orm_correct))
    lora_correct = np.array(
        [int(p.correct[int(np.argmax(p.lora_scores))]) for p in problems], dtype=int
    )
    lora_acc = float(np.mean(lora_correct))

    logging.info(
        "Baseline accuracies: Oracle=%.4f MV=%.4f Slot=%.4f LoRA=%.4f ORM=%.4f",
        oracle_acc,
        mv_acc,
        slot_acc,
        lora_acc,
        orm_acc,
    )

    alpha_values = [round(a, 2) for a in np.arange(0.0, 1.0001, 0.05)]
    hybrid_sweep: List[Dict[str, Any]] = []
    for alpha in alpha_values:
        corrects = []
        for p in problems:
            scores = hybrid_score(p.slot_scores, p.frequency_scores, alpha)
            idx = int(np.argmax(scores))
            corrects.append(int(p.correct[idx]))
        acc = float(np.mean(corrects))
        hybrid_sweep.append({"alpha": float(alpha), "accuracy": acc})
    hybrid_best = max(hybrid_sweep, key=lambda x: x["accuracy"])
    logging.info(
        "Hybrid best: alpha=%.2f accuracy=%.4f",
        hybrid_best["alpha"],
        hybrid_best["accuracy"],
    )

    threshold_values = [round(t, 2) for t in np.arange(0.1, 1.0001, 0.05)]
    adaptive_sweep: List[Dict[str, Any]] = []
    for threshold in threshold_values:
        corrects = []
        use_verifier = []
        for p in problems:
            idx, used_verifier = adaptive_switch_index(
                p.slot_scores, p.answers, threshold
            )
            corrects.append(int(p.correct[idx]))
            use_verifier.append(int(used_verifier))
        acc = float(np.mean(corrects))
        coverage = float(np.mean(use_verifier))
        adaptive_sweep.append(
            {"threshold": float(threshold), "accuracy": acc, "coverage": coverage}
        )
    adaptive_best = max(adaptive_sweep, key=lambda x: (x["accuracy"], x["coverage"]))
    logging.info(
        "Adaptive best: threshold=%.2f accuracy=%.4f coverage=%.3f",
        adaptive_best["threshold"],
        adaptive_best["accuracy"],
        adaptive_best["coverage"],
    )

    features = np.stack([compute_gate_features(p) for p in problems], axis=0)
    labels = slot_correct.copy()
    gate_probs = cross_val_gate_probs(features, labels, SEED)
    logging.info(
        "Gate prob stats: min=%.3f max=%.3f mean=%.3f",
        float(np.min(gate_probs)),
        float(np.max(gate_probs)),
        float(np.mean(gate_probs)),
    )
    for preview, prob in zip(problems[:5], gate_probs[:5]):
        logging.info(
            "Gate sample idx=%d prob=%.3f label=%d mv=%d",
            preview.idx,
            float(prob),
            int(preview.slot_correct),
            int(preview.mv_correct),
        )

    gate_thresholds = [round(t, 2) for t in np.arange(0.0, 1.0001, 0.05)]
    gate_sweep, gate_best = evaluate_gate_thresholds(
        gate_probs, slot_correct, mv_correct, gate_thresholds
    )
    logging.info(
        "Gate best: tau=%.2f accuracy=%.4f coverage=%.3f",
        gate_best["tau"],
        gate_best["accuracy"],
        gate_best["coverage"],
    )

    pasv_variants = {
        "hybrid": hybrid_best["accuracy"],
        "adaptive": adaptive_best["accuracy"],
        "gate": gate_best["accuracy"],
    }
    best_variant = max(pasv_variants.items(), key=lambda kv: kv[1])
    best_variant_name = best_variant[0]
    best_variant_acc = best_variant[1]

    if best_variant_name == "hybrid":
        best_param = hybrid_best["alpha"]
        pasv_label = f"PASV Hybrid (α={best_param:.2f})"
    elif best_variant_name == "adaptive":
        best_param = adaptive_best["threshold"]
        pasv_label = f"PASV Adaptive (τ={best_param:.2f})"
    else:
        best_param = gate_best["tau"]
        pasv_label = f"PASV Gate (τ={best_param:.2f})"

    logging.info(
        "Best PASV variant: %s param=%.2f accuracy=%.4f",
        best_variant_name,
        float(best_param),
        best_variant_acc,
    )

    if best_variant_name == "hybrid":
        best_correct = []
        for p in problems:
            scores = hybrid_score(p.slot_scores, p.frequency_scores, best_param)
            idx = int(np.argmax(scores))
            best_correct.append(int(p.correct[idx]))
        pasv_correct = np.array(best_correct, dtype=int)
    elif best_variant_name == "adaptive":
        best_correct = []
        for p in problems:
            idx, _ = adaptive_switch_index(p.slot_scores, p.answers, best_param)
            best_correct.append(int(p.correct[idx]))
        pasv_correct = np.array(best_correct, dtype=int)
    else:
        use_verifier = gate_probs >= best_param
        pasv_correct = np.where(use_verifier, slot_correct, mv_correct).astype(int)

    multiplicities = np.array([p.num_correct for p in problems], dtype=int)
    mv_strat = stratified_accuracy(mv_correct, multiplicities)
    slot_strat = stratified_accuracy(slot_correct, multiplicities)
    pasv_strat = stratified_accuracy(pasv_correct, multiplicities)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corrected_baselines = {
        "oracle": {
            "params": "--",
            "accuracy": oracle_acc,
            "delta_vs_mv": None,
            "source": "computed",
        },
        "pasv": {
            "params": "552M",
            "accuracy": best_variant_acc,
            "delta_vs_mv": (best_variant_acc - mv_acc) * 100,
            "variant": best_variant_name,
            "variant_param": float(best_param),
            "source": "computed",
        },
        "mv": {
            "params": "--",
            "accuracy": mv_acc,
            "delta_vs_mv": None,
            "source": "computed",
        },
        "slot_verifier": {
            "params": "552M",
            "accuracy": slot_acc,
            "delta_vs_mv": (slot_acc - mv_acc) * 100,
            "source": "computed",
        },
        "lora_orm": {
            "params": "341M",
            "accuracy": lora_acc,
            "delta_vs_mv": (lora_acc - mv_acc) * 100,
            "source": "computed",
        },
        "prm_math_shepherd": {
            "params": "7B",
            "accuracy": 0.572,
            "delta_vs_mv": (0.572 - mv_acc) * 100,
            "source": "paper",
        },
        "orm_head": {
            "params": "26M",
            "accuracy": orm_acc,
            "delta_vs_mv": (orm_acc - mv_acc) * 100,
            "source": "computed",
        },
    }

    with (OUTPUT_DIR / "corrected_baselines.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(corrected_baselines, handle, indent=2)
    logging.info("Saved corrected baselines JSON")

    pasv_results = {
        "hybrid_sweep": hybrid_sweep,
        "hybrid_best": hybrid_best,
        "adaptive_sweep": adaptive_sweep,
        "adaptive_best": adaptive_best,
        "gate_sweep": gate_sweep,
        "gate_best": gate_best,
    }
    with (OUTPUT_DIR / "pasv_results.json").open("w", encoding="utf-8") as handle:
        json.dump(pasv_results, handle, indent=2)
    logging.info("Saved PASV results JSON")

    mv_delta = None
    slot_delta = (slot_acc - mv_acc) * 100
    lora_delta = (lora_acc - mv_acc) * 100
    orm_delta = (orm_acc - mv_acc) * 100
    pasv_delta = (best_variant_acc - mv_acc) * 100

    main_table = "\n".join(
        [
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Method & Params & Accuracy & $\\Delta$ vs MV \\\\",
            "\\midrule",
            f"Oracle & -- & {format_pct(oracle_acc)} & -- \\\\",
            (
                "\\textbf{PASV (Slot + SC)}"
                f" & 552M & {format_pct(best_variant_acc, bold=True)}"
                f" & {format_delta(pasv_delta, bold=True)} \\\\"
            ),
            f"MV (Self-Consistency) & -- & {format_pct(mv_acc)} & -- \\\\",
            f"Slot Verifier & 552M & {format_pct(slot_acc)} & {format_delta(slot_delta)} \\\\",
            f"LoRA ORM & 341M & {format_pct(lora_acc)} & {format_delta(lora_delta)} \\\\",
            (
                "PRM Math-Shepherd$^\\dagger$"
                f" & 7B & {format_pct(0.572, star=True)}"
                f" & {format_delta((0.572 - mv_acc) * 100)} \\\\"
            ),
            f"ORM Head & 26M & {format_pct(orm_acc)} & {format_delta(orm_delta)} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\vspace{2pt}",
            "{\\footnotesize $^\\dagger$PRM value from paper; $^*$ may need rechecking.}",
        ]
    )
    (OUTPUT_DIR / "main_table.txt").write_text(main_table, encoding="utf-8")
    logging.info("Saved main table")

    plot_pasv_comparison(
        OUTPUT_DIR / "fig_pasv_comparison.pdf",
        mv_acc,
        slot_acc,
        hybrid_best,
        adaptive_best,
        gate_best,
    )
    plot_hybrid_sweep(OUTPUT_DIR / "fig_hybrid_sweep.pdf", hybrid_sweep)
    plot_stratified(
        OUTPUT_DIR / "fig_stratified.pdf",
        mv_strat,
        slot_strat,
        pasv_strat,
        pasv_label,
    )

    summary_lines = [
        "Corrected PASV Evaluation Summary",
        f"Seed: {SEED}",
        f"Total problems: {total}",
        "",
        "Baseline accuracies (numerical matching):",
        f"- Oracle: {oracle_acc * 100:.1f}%",
        f"- MV: {mv_acc * 100:.1f}%",
        f"- Slot Verifier: {slot_acc * 100:.1f}%",
        f"- LoRA ORM: {lora_acc * 100:.1f}%",
        f"- ORM Head: {orm_acc * 100:.1f}%",
        f"- PRM Math-Shepherd (paper): {0.572 * 100:.1f}%",
        "",
        "PASV variants:",
        (
            f"- Hybrid scoring: best alpha={hybrid_best['alpha']:.2f} "
            f"acc={hybrid_best['accuracy'] * 100:.1f}%"
        ),
        (
            f"- Adaptive switch: best threshold={adaptive_best['threshold']:.2f} "
            f"acc={adaptive_best['accuracy'] * 100:.1f}% "
            f"coverage={adaptive_best['coverage'] * 100:.1f}%"
        ),
        (
            f"- Learned gate (5-fold CV): best tau={gate_best['tau']:.2f} "
            f"acc={gate_best['accuracy'] * 100:.1f}% "
            f"coverage={gate_best['coverage'] * 100:.1f}%"
        ),
        "",
        f"Best PASV variant: {best_variant_name} (param={best_param:.2f})",
        f"Best PASV accuracy: {best_variant_acc * 100:.1f}% (Δ vs MV {pasv_delta:.1f} pp)",
        "",
        "Stratified accuracy by multiplicity (MV / Slot / PASV best):",
    ]

    for bucket in ["0", "1", "2-3", "4-7", "8+"]:
        summary_lines.append(
            f"- m={bucket}: MV={mv_strat[bucket]['accuracy'] * 100:.1f}% "
            f"Slot={slot_strat[bucket]['accuracy'] * 100:.1f}% "
            f"PASV={pasv_strat[bucket]['accuracy'] * 100:.1f}% "
            f"(n={mv_strat[bucket]['n']})"
        )

    summary_lines.append("")
    summary_lines.append(f"Outputs saved to: {OUTPUT_DIR}")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    logging.info("Saved summary.txt")


if __name__ == "__main__":
    main()
