#!/usr/bin/env python3
"""Polarity-Aware Selective Verification (PASV) evaluation for MATH.

PASV trains a gate that predicts when the verifier is reliable. If the gate
abstains, we fall back to majority vote (MV). This script evaluates baselines
and PASV variants on MATH data.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_PATH = REPO_ROOT / "experiments/math-14b/eval_full_5000.json"
DEFAULT_CANDIDATES_PATH = (
    REPO_ROOT / "experiments/math-14b/candidates_test_full_5000.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/pasv"

QWEN_EVAL_PATH = REPO_ROOT / "experiments/qwen-4b/eval_test.json"
QWEN_CANDIDATES_PATH = REPO_ROOT / "experiments/qwen-4b/candidates_test_full.jsonl"
QWEN14B_EVAL_PATH = REPO_ROOT / "experiments/qwen-14b/eval_test.json"
QWEN14B_CANDIDATES_PATH = REPO_ROOT / "experiments/qwen-14b/candidates_test_full.jsonl"

FEATURE_NAMES = [
    "top1_score",
    "top2_score",
    "margin",
    "score_mean",
    "score_std",
    "score_entropy",
    "max_prob",
    "n_unique_answers",
    "n_candidates",
    "diversity_ratio",
    "agrees_with_mv",
    "mv_fraction",
    "verifier_answer_count",
    "top1_rank_by_count",
]


@dataclass
class DatasetBundle:
    name: str
    total_problems: int
    missing_candidates: int
    missing_scores: int
    missing_verifier: int
    features: np.ndarray
    labels: np.ndarray
    mv_correct: np.ndarray
    top1_scores: np.ndarray
    score_entropies: np.ndarray
    idxs: np.ndarray
    feature_dicts: List[Dict[str, float]]


@dataclass
class CandidateLine:
    idx: Optional[int]
    ground_truth: Any
    candidates: List[Dict[str, Any]]


def configure_style() -> None:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_eval_data(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict):
        problems = payload.get("per_problem")
        if problems is None:
            raise ValueError(f"Missing per_problem in {path}")
        return problems, payload.get("strategies", {})
    if isinstance(payload, list):
        return payload, {}
    raise ValueError(f"Unexpected eval payload structure in {path}")


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
                raise ValueError(
                    f"Missing candidates list in {path} line {line_num}"
                )
            ground_truth = record.get("ground_truth")
            if ground_truth is None:
                raise ValueError(f"Missing ground_truth in {path} line {line_num}")
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


def _get_field(entry: Dict[str, Any], names: Iterable[str]) -> Optional[Any]:
    for name in names:
        if name in entry:
            return entry[name]
    return None


def majority_vote_answer(candidate_answers: List[str]) -> str:
    if not candidate_answers:
        return ""
    counts = Counter(candidate_answers)
    return counts.most_common(1)[0][0]


def compute_mv_correct(mv_answer: str, candidate_entries: List[Dict[str, Any]]) -> int:
    for cand in candidate_entries:
        if cand.get("answer") == mv_answer and cand.get("correct"):
            return 1
    return 0


def compute_verifier_correct(
    verifier_best_idx: Optional[int], candidate_entries: List[Dict[str, Any]]
) -> Optional[int]:
    if verifier_best_idx is None:
        return None
    if 0 <= verifier_best_idx < len(candidate_entries):
        return int(bool(candidate_entries[verifier_best_idx].get("correct")))
    return None


def extract_features(
    scores: List[float], candidate_answers: List[str], mv_answer: str
) -> Dict[str, float]:
    scores_array = np.array(scores, dtype=float)
    if scores_array.size == 0:
        scores_array = np.zeros(1, dtype=float)

    sorted_scores = np.sort(scores_array)[::-1]
    features: Dict[str, float] = {}

    features["top1_score"] = float(sorted_scores[0])
    features["top2_score"] = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    if len(sorted_scores) > 1:
        features["margin"] = float(sorted_scores[0] - sorted_scores[1])
    else:
        features["margin"] = float(sorted_scores[0])
    features["score_mean"] = float(np.mean(scores_array))
    features["score_std"] = float(np.std(scores_array))

    exp_scores = np.exp(scores_array - np.max(scores_array))
    probs = exp_scores / np.sum(exp_scores)
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    features["score_entropy"] = entropy
    features["max_prob"] = float(np.max(probs))

    unique_answers = set(candidate_answers)
    features["n_unique_answers"] = float(len(unique_answers))
    features["n_candidates"] = float(len(scores_array))
    features["diversity_ratio"] = (
        float(len(unique_answers)) / float(len(scores_array))
        if len(scores_array)
        else 0.0
    )

    verifier_top_idx = int(np.argmax(scores_array))
    verifier_answer = candidate_answers[verifier_top_idx] if candidate_answers else ""
    features["agrees_with_mv"] = float(int(verifier_answer == mv_answer))

    answer_counts = Counter(candidate_answers)
    if answer_counts:
        mv_count = answer_counts.most_common(1)[0][1]
        features["mv_fraction"] = float(mv_count) / float(len(candidate_answers))
        features["verifier_answer_count"] = float(
            answer_counts.get(verifier_answer, 0)
        ) / float(len(candidate_answers))
    else:
        features["mv_fraction"] = 0.0
        features["verifier_answer_count"] = 0.0

    features["top1_rank_by_count"] = float(int(features["agrees_with_mv"] == 1.0))
    return features


def build_dataset(name: str, eval_path: Path, candidates_path: Path) -> DatasetBundle:
    problems, strategies = load_eval_data(eval_path)
    candidates = load_candidates(candidates_path)

    if len(candidates) != len(problems):
        raise ValueError(
            f"{name} candidates count {len(candidates)} != eval count {len(problems)}"
        )
    logging.info(
        "%s eval problems=%d candidates=%d", name, len(problems), len(candidates)
    )

    features: List[Dict[str, float]] = []
    labels: List[int] = []
    mv_corrects: List[int] = []
    top1_scores: List[float] = []
    score_entropies: List[float] = []
    idxs: List[int] = []

    total_problems = len(problems)
    missing_candidates = 0
    missing_scores = 0
    missing_verifier = 0
    for position, (entry, cand_entry) in enumerate(zip(problems, candidates)):
        eval_gt = entry.get("ground_truth")
        if eval_gt is None:
            raise ValueError(
                f"{name} eval missing ground_truth at position {position}"
            )
        cand_gt = cand_entry.ground_truth
        if eval_gt != cand_gt:
            eval_idx = entry.get("idx")
            raise ValueError(
                f"{name} ground_truth mismatch at position {position}: "
                f"eval_idx={eval_idx} cand_idx={cand_entry.idx} "
                f"eval_gt={eval_gt!r} cand_gt={cand_gt!r}"
            )

        candidate_entries = cand_entry.candidates
        idx_value = entry.get("idx")
        if idx_value is None:
            idx_value = cand_entry.idx if cand_entry.idx is not None else position
        idx = int(idx_value)

        scores = _get_field(entry, ["all_scores", "scores", "verifier_scores"])
        if scores is None:
            missing_scores += 1
            logging.warning("Missing scores for idx=%s", idx)
            continue

        candidate_answers = [str(cand.get("answer", "")) for cand in candidate_entries]
        if len(candidate_answers) != len(scores):
            min_len = min(len(candidate_answers), len(scores))
            logging.warning(
                "Mismatch candidate/scores length for idx=%s (answers=%d scores=%d)",
                idx,
                len(candidate_answers),
                len(scores),
            )
            candidate_answers = candidate_answers[:min_len]
            scores = scores[:min_len]

        mv_answer = _get_field(entry, ["mv_answer", "majority_vote_answer"]) or ""
        if not mv_answer:
            mv_answer = majority_vote_answer(candidate_answers)

        feats = extract_features(scores, candidate_answers, mv_answer)
        features.append(feats)
        top1_scores.append(feats["top1_score"])
        score_entropies.append(feats["score_entropy"])
        idxs.append(idx)

        verifier_correct = _get_field(
            entry, ["verifier_correct", "verifier_top1_correct", "top1_correct"]
        )
        if verifier_correct is None:
            verifier_best_idx = _get_field(
                entry, ["verifier_best_idx", "verifier_top_idx", "best_idx"]
            )
            if verifier_best_idx is None:
                verifier_best_idx = int(np.argmax(np.array(scores, dtype=float)))
            verifier_correct = compute_verifier_correct(
                int(verifier_best_idx), candidate_entries
            )
        if verifier_correct is None:
            missing_verifier += 1
            logging.warning("Missing verifier correctness for idx=%s", idx)
            continue

        mv_correct = _get_field(
            entry, ["mv_correct", "majority_vote_correct", "mv_is_correct"]
        )
        if mv_correct is None:
            mv_correct = compute_mv_correct(mv_answer, candidate_entries)
        labels.append(int(verifier_correct))
        mv_corrects.append(int(mv_correct))

    if missing_candidates or missing_scores or missing_verifier:
        logging.warning(
            "Skipped problems: missing_candidates=%d missing_scores=%d missing_verifier=%d",
            missing_candidates,
            missing_scores,
            missing_verifier,
        )

    feature_matrix = np.array(
        [[feat[name] for name in FEATURE_NAMES] for feat in features], dtype=float
    )
    labels_array = np.array(labels, dtype=int)
    mv_correct_array = np.array(mv_corrects, dtype=int)
    top1_scores_array = np.array(top1_scores, dtype=float)
    score_entropy_array = np.array(score_entropies, dtype=float)
    idxs_array = np.array(idxs, dtype=int)

    logging.info(
        "Loaded %s: %d examples (strategies=%s)",
        name,
        len(labels_array),
        list(strategies.keys()),
    )
    return DatasetBundle(
        name=name,
        total_problems=total_problems,
        missing_candidates=missing_candidates,
        missing_scores=missing_scores,
        missing_verifier=missing_verifier,
        features=feature_matrix,
        labels=labels_array,
        mv_correct=mv_correct_array,
        top1_scores=top1_scores_array,
        score_entropies=score_entropy_array,
        idxs=idxs_array,
        feature_dicts=features,
    )


def build_thresholds(values: np.ndarray) -> List[float]:
    unique_values = np.unique(values)
    if unique_values.size == 0:
        return [0.0]
    eps = 1e-12
    thresholds = [float(unique_values[0] - eps)]
    thresholds.extend(float(v) for v in unique_values.tolist())
    thresholds.append(float(unique_values[-1] + eps))
    return thresholds


def compute_metrics(
    use_verifier: np.ndarray, verifier_correct: np.ndarray, mv_correct: np.ndarray
) -> Tuple[float, float, Optional[float], int]:
    correct = np.where(use_verifier, verifier_correct, mv_correct)
    accuracy = float(np.mean(correct))
    coverage = float(np.mean(use_verifier))
    verifier_count = int(np.sum(use_verifier))
    if verifier_count == 0:
        return accuracy, coverage, None, verifier_count
    verifier_precision = float(np.mean(verifier_correct[use_verifier]))
    return accuracy, coverage, verifier_precision, verifier_count


def pareto_frontier(curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_curve = sorted(curve, key=lambda x: (x["coverage"], x["accuracy"]))
    frontier: List[Dict[str, Any]] = []
    best_accuracy = -1.0
    for entry in sorted_curve:
        if entry["accuracy"] >= best_accuracy - 1e-12:
            frontier.append(entry)
            best_accuracy = entry["accuracy"]
    return frontier


def evaluate_threshold_policy(
    verifier_correct: np.ndarray,
    mv_correct: np.ndarray,
    thresholds: List[float],
    use_verifier_fn,
    name: str,
) -> Dict[str, Any]:
    curve: List[Dict[str, Any]] = []
    for tau in thresholds:
        use_verifier = use_verifier_fn(tau)
        accuracy, coverage, verifier_precision, verifier_count = compute_metrics(
            use_verifier, verifier_correct, mv_correct
        )
        curve.append(
            {
                "threshold": float(tau),
                "accuracy": accuracy,
                "coverage": coverage,
                "verifier_precision": verifier_precision,
                "n_verifier": verifier_count,
            }
        )

    best = max(curve, key=lambda x: (x["accuracy"], x["coverage"]))
    frontier = pareto_frontier(curve)
    logging.info(
        "%s best accuracy=%.4f coverage=%.3f verifier_precision=%s",
        name,
        best["accuracy"],
        best["coverage"],
        "NA"
        if best["verifier_precision"] is None
        else f"{best['verifier_precision']:.3f}",
    )
    return {"curve": curve, "best": best, "pareto_frontier": frontier}


class GateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_gate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Tuple[StandardScaler, GateMLP]:
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GateMLP(input_dim=X_scaled.shape[1], hidden_dim=32, dropout=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            correct += int((preds == batch_y.long()).sum().item())
            total += int(batch_x.size(0))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(total, 1)
            acc = correct / max(total, 1)
            logging.info(
                "MLP epoch %d/%d loss=%.4f acc=%.3f",
                epoch + 1,
                epochs,
                avg_loss,
                acc,
            )

    return scaler, model


def predict_mlp_gate(
    scaler: StandardScaler, model: GateMLP, X: np.ndarray
) -> np.ndarray:
    model.eval()
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def summarize_feature_stats(feature_matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(FEATURE_NAMES):
        values = feature_matrix[:, idx]
        stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return stats


def plot_coverage_accuracy(
    output_path: Path,
    curves: Dict[str, Dict[str, Any]],
    baselines: Dict[str, Dict[str, Any]],
) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    palette = {
        "naive_threshold": "#1f77b4",
        "entropy_threshold": "#ff7f0e",
        "pasv_logreg": "#2ca02c",
        "pasv_mlp": "#9467bd",
        "pasv_gbt": "#d62728",
    }

    for key, data in curves.items():
        frontier = data["pareto_frontier"]
        xs = [pt["coverage"] for pt in frontier]
        ys = [pt["accuracy"] for pt in frontier]
        label = data["label"]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=palette[key],
            label=label,
        )

    ax.scatter(
        baselines["always_mv"]["coverage"],
        baselines["always_mv"]["accuracy"],
        marker="*",
        s=90,
        color="#7f7f7f",
        label="Always MV",
        zorder=5,
    )
    ax.scatter(
        baselines["always_verifier"]["coverage"],
        baselines["always_verifier"]["accuracy"],
        marker="*",
        s=90,
        color="#000000",
        label="Always Verifier",
        zorder=5,
    )

    ax.set_xlabel("Coverage (verifier usage)")
    ax.set_ylabel("Accuracy")
    ax.set_title("PASV Coverage-Accuracy Curves")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right", frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logging.info("Saved coverage-accuracy plot to %s", output_path)


def plot_feature_importance(
    output_path: Path,
    logreg_importance: Dict[str, float],
    gbt_importance: Dict[str, float],
) -> None:
    configure_style()
    ordered_features = sorted(
        FEATURE_NAMES,
        key=lambda name: (
            -(logreg_importance.get(name, 0.0) + gbt_importance.get(name, 0.0))
        ),
    )
    logreg_vals = [logreg_importance.get(name, 0.0) for name in ordered_features]
    gbt_vals = [gbt_importance.get(name, 0.0) for name in ordered_features]

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)

    axes[0].bar(range(len(ordered_features)), logreg_vals, color="#2ca02c")
    axes[0].set_ylabel("|Coef| (normalized)")
    axes[0].set_title("LogReg Feature Importance")

    axes[1].bar(range(len(ordered_features)), gbt_vals, color="#d62728")
    axes[1].set_ylabel("Importance (normalized)")
    axes[1].set_title("GBT Feature Importance")
    axes[1].set_xticks(range(len(ordered_features)))
    axes[1].set_xticklabels(ordered_features, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logging.info("Saved feature importance plot to %s", output_path)


def evaluate_dataset(
    dataset: DatasetBundle,
    output_dir: Path,
    *,
    seed: int,
    mlp_epochs: int,
    mlp_batch_size: int,
    mlp_lr: float,
    mlp_weight_decay: float,
) -> Dict[str, Any]:
    n = dataset.features.shape[0]
    calibration_size = 2500 if n >= 5000 else n // 2
    calibration_size = max(1, calibration_size)
    test_size = n - calibration_size

    logging.info(
        "%s coverage: total=%d used=%d missing_candidates=%d missing_scores=%d missing_verifier=%d",
        dataset.name,
        dataset.total_problems,
        n,
        dataset.missing_candidates,
        dataset.missing_scores,
        dataset.missing_verifier,
    )

    X_train = dataset.features[:calibration_size]
    y_train = dataset.labels[:calibration_size]
    mv_train = dataset.mv_correct[:calibration_size]
    X_test = dataset.features[calibration_size:]
    y_test = dataset.labels[calibration_size:]
    mv_test = dataset.mv_correct[calibration_size:]
    idx_test = dataset.idxs[calibration_size:]

    logging.info(
        "%s split: calibration=%d test=%d", dataset.name, calibration_size, test_size
    )
    verifier_accuracy = float(np.mean(y_test))
    mv_accuracy = float(np.mean(mv_test))
    logging.info(
        "%s baseline accuracies: verifier=%.4f mv=%.4f",
        dataset.name,
        verifier_accuracy,
        mv_accuracy,
    )

    both_correct = int(np.sum((y_test == 1) & (mv_test == 1)))
    verifier_only = int(np.sum((y_test == 1) & (mv_test == 0)))
    mv_only = int(np.sum((y_test == 0) & (mv_test == 1)))
    both_wrong = int(np.sum((y_test == 0) & (mv_test == 0)))
    logging.info(
        "%s verifier vs MV: both_correct=%d verifier_only=%d mv_only=%d both_wrong=%d",
        dataset.name,
        both_correct,
        verifier_only,
        mv_only,
        both_wrong,
    )

    top1_thresholds = build_thresholds(dataset.top1_scores[calibration_size:])
    entropy_thresholds = build_thresholds(dataset.score_entropies[calibration_size:])

    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
        ]
    )
    logreg.fit(X_train, y_train)
    logreg_probs = logreg.predict_proba(X_test)[:, 1]

    gbt = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
    gbt.fit(X_train, y_train)
    gbt_probs = gbt.predict_proba(X_test)[:, 1]

    mlp_scaler, mlp_model = train_mlp_gate(
        X_train,
        y_train,
        seed=seed,
        epochs=mlp_epochs,
        batch_size=mlp_batch_size,
        lr=mlp_lr,
        weight_decay=mlp_weight_decay,
    )
    mlp_probs = predict_mlp_gate(mlp_scaler, mlp_model, X_test)

    logging.info(
        "%s gate prob stats logreg: min=%.3f max=%.3f mean=%.3f",
        dataset.name,
        float(np.min(logreg_probs)),
        float(np.max(logreg_probs)),
        float(np.mean(logreg_probs)),
    )
    logging.info(
        "%s gate prob stats mlp: min=%.3f max=%.3f mean=%.3f",
        dataset.name,
        float(np.min(mlp_probs)),
        float(np.max(mlp_probs)),
        float(np.mean(mlp_probs)),
    )
    logging.info(
        "%s gate prob stats gbt: min=%.3f max=%.3f mean=%.3f",
        dataset.name,
        float(np.min(gbt_probs)),
        float(np.max(gbt_probs)),
        float(np.mean(gbt_probs)),
    )

    for preview_idx in range(min(5, len(idx_test))):
        logging.info(
            "%s sample idx=%d logreg=%.3f mlp=%.3f gbt=%.3f y=%d mv=%d",
            dataset.name,
            int(idx_test[preview_idx]),
            float(logreg_probs[preview_idx]),
            float(mlp_probs[preview_idx]),
            float(gbt_probs[preview_idx]),
            int(y_test[preview_idx]),
            int(mv_test[preview_idx]),
        )

    logreg_thresholds = build_thresholds(logreg_probs)
    mlp_thresholds = build_thresholds(mlp_probs)
    gbt_thresholds = build_thresholds(gbt_probs)

    naive_results = evaluate_threshold_policy(
        y_test,
        mv_test,
        top1_thresholds,
        lambda tau: dataset.top1_scores[calibration_size:] >= tau,
        "Top1 threshold",
    )
    entropy_results = evaluate_threshold_policy(
        y_test,
        mv_test,
        entropy_thresholds,
        lambda tau: dataset.score_entropies[calibration_size:] <= tau,
        "Entropy threshold",
    )
    pasv_logreg = evaluate_threshold_policy(
        y_test,
        mv_test,
        logreg_thresholds,
        lambda tau: logreg_probs >= tau,
        "PASV LogReg",
    )
    pasv_mlp = evaluate_threshold_policy(
        y_test,
        mv_test,
        mlp_thresholds,
        lambda tau: mlp_probs >= tau,
        "PASV MLP",
    )
    pasv_gbt = evaluate_threshold_policy(
        y_test,
        mv_test,
        gbt_thresholds,
        lambda tau: gbt_probs >= tau,
        "PASV GBT",
    )

    baselines = {
        "always_verifier": {
            "accuracy": verifier_accuracy,
            "coverage": 1.0,
            "verifier_precision": verifier_accuracy,
        },
        "always_mv": {
            "accuracy": mv_accuracy,
            "coverage": 0.0,
            "verifier_precision": None,
        },
    }

    logreg_coef = logreg.named_steps["clf"].coef_[0]
    logreg_importance = np.abs(logreg_coef)
    logreg_importance = logreg_importance / (np.sum(logreg_importance) + 1e-12)
    logreg_importance_map = {
        name: float(value) for name, value in zip(FEATURE_NAMES, logreg_importance)
    }
    gbt_importance = gbt.feature_importances_
    gbt_importance = gbt_importance / (np.sum(gbt_importance) + 1e-12)
    gbt_importance_map = {
        name: float(value) for name, value in zip(FEATURE_NAMES, gbt_importance)
    }

    curves_for_plot = {
        "naive_threshold": {"label": "Top1 Score", **naive_results},
        "entropy_threshold": {"label": "Entropy", **entropy_results},
        "pasv_logreg": {"label": "PASV LogReg", **pasv_logreg},
        "pasv_mlp": {"label": "PASV MLP", **pasv_mlp},
        "pasv_gbt": {"label": "PASV GBT", **pasv_gbt},
    }

    plot_coverage_accuracy(
        output_dir / "coverage_accuracy.pdf", curves_for_plot, baselines
    )
    plot_feature_importance(
        output_dir / "feature_importance.pdf",
        logreg_importance_map,
        gbt_importance_map,
    )

    feature_stats = summarize_feature_stats(dataset.features)

    results = {
        "dataset": {
            "name": dataset.name,
            "n_examples": n,
            "total_problems": dataset.total_problems,
            "missing_candidates": dataset.missing_candidates,
            "missing_scores": dataset.missing_scores,
            "missing_verifier": dataset.missing_verifier,
            "calibration_size": calibration_size,
            "test_size": test_size,
        },
        "baselines": baselines,
        "feature_stats": feature_stats,
        "methods": {
            "naive_threshold": naive_results,
            "entropy_threshold": entropy_results,
            "pasv_logreg": pasv_logreg,
            "pasv_mlp": pasv_mlp,
            "pasv_gbt": pasv_gbt,
        },
        "feature_importance": {
            "logreg": logreg_importance_map,
            "gbt": gbt_importance_map,
        },
    }

    summary_lines = [
        f"PASV summary for {dataset.name}",
        f"Total problems in eval: {dataset.total_problems}",
        f"Examples with candidates: {n}",
        (
            "Missing candidates: "
            f"{dataset.missing_candidates} (scores missing: {dataset.missing_scores}, "
            f"verifier missing: {dataset.missing_verifier})"
        ),
        f"Calibration size: {calibration_size}",
        f"Test size: {test_size}",
        f"Always verifier accuracy: {verifier_accuracy:.4f}",
        f"Always MV accuracy: {mv_accuracy:.4f}",
        "",
        "Best operating points (accuracy, coverage, verifier precision):",
    ]
    for method_key, label in [
        ("naive_threshold", "Top1 threshold"),
        ("entropy_threshold", "Entropy threshold"),
        ("pasv_logreg", "PASV LogReg"),
        ("pasv_mlp", "PASV MLP"),
        ("pasv_gbt", "PASV GBT"),
    ]:
        best = results["methods"][method_key]["best"]
        verifier_precision = best["verifier_precision"]
        summary_lines.append(
            f"- {label}: acc={best['accuracy']:.4f} cov={best['coverage']:.3f} "
            f"prec={'NA' if verifier_precision is None else f'{verifier_precision:.3f}'}"
        )

    summary_lines.append("")
    summary_lines.append("Top features (logreg):")
    top_logreg = sorted(
        logreg_importance_map.items(), key=lambda kv: kv[1], reverse=True
    )[:5]
    for name, value in top_logreg:
        summary_lines.append(f"- {name}: {value:.3f}")

    summary_lines.append("")
    summary_lines.append("Top features (gbt):")
    top_gbt = sorted(gbt_importance_map.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for name, value in top_gbt:
        summary_lines.append(f"- {name}: {value:.3f}")

    summary_text = "\n".join(summary_lines)
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    logging.info("Saved summary to %s", output_dir / "summary.txt")

    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logging.info("Saved results to %s", output_dir / "results.json")

    return results


def run_cross_validation(
    dataset: DatasetBundle,
    *,
    seed: int,
    mlp_epochs: int,
    mlp_batch_size: int,
    mlp_lr: float,
    mlp_weight_decay: float,
) -> Dict[str, Any]:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_metrics: Dict[str, List[Dict[str, float]]] = {
        "naive_threshold": [],
        "entropy_threshold": [],
        "pasv_logreg": [],
        "pasv_mlp": [],
        "pasv_gbt": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(dataset.features)):
        X_train = dataset.features[train_idx]
        y_train = dataset.labels[train_idx]
        mv_train = dataset.mv_correct[train_idx]
        X_test = dataset.features[test_idx]
        y_test = dataset.labels[test_idx]
        mv_test = dataset.mv_correct[test_idx]
        top1_scores = dataset.top1_scores[test_idx]
        score_entropies = dataset.score_entropies[test_idx]

        logreg = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
        logreg.fit(X_train, y_train)
        logreg_probs = logreg.predict_proba(X_test)[:, 1]

        gbt = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed
        )
        gbt.fit(X_train, y_train)
        gbt_probs = gbt.predict_proba(X_test)[:, 1]

        mlp_scaler, mlp_model = train_mlp_gate(
            X_train,
            y_train,
            seed=seed,
            epochs=mlp_epochs,
            batch_size=mlp_batch_size,
            lr=mlp_lr,
            weight_decay=mlp_weight_decay,
        )
        mlp_probs = predict_mlp_gate(mlp_scaler, mlp_model, X_test)

        naive = evaluate_threshold_policy(
            y_test,
            mv_test,
            build_thresholds(top1_scores),
            lambda tau: top1_scores >= tau,
            f"CV fold {fold_idx} top1",
        )
        entropy = evaluate_threshold_policy(
            y_test,
            mv_test,
            build_thresholds(score_entropies),
            lambda tau: score_entropies <= tau,
            f"CV fold {fold_idx} entropy",
        )
        pasv_logreg = evaluate_threshold_policy(
            y_test,
            mv_test,
            build_thresholds(logreg_probs),
            lambda tau: logreg_probs >= tau,
            f"CV fold {fold_idx} logreg",
        )
        pasv_mlp = evaluate_threshold_policy(
            y_test,
            mv_test,
            build_thresholds(mlp_probs),
            lambda tau: mlp_probs >= tau,
            f"CV fold {fold_idx} mlp",
        )
        pasv_gbt = evaluate_threshold_policy(
            y_test,
            mv_test,
            build_thresholds(gbt_probs),
            lambda tau: gbt_probs >= tau,
            f"CV fold {fold_idx} gbt",
        )

        cv_metrics["naive_threshold"].append(naive["best"])
        cv_metrics["entropy_threshold"].append(entropy["best"])
        cv_metrics["pasv_logreg"].append(pasv_logreg["best"])
        cv_metrics["pasv_mlp"].append(pasv_mlp["best"])
        cv_metrics["pasv_gbt"].append(pasv_gbt["best"])

        logging.info(
            "CV fold %d summary: logreg acc=%.4f mlp acc=%.4f gbt acc=%.4f",
            fold_idx,
            pasv_logreg["best"]["accuracy"],
            pasv_mlp["best"]["accuracy"],
            pasv_gbt["best"]["accuracy"],
        )

    cv_summary: Dict[str, Dict[str, float]] = {}
    for key, entries in cv_metrics.items():
        accs = np.array([entry["accuracy"] for entry in entries], dtype=float)
        covs = np.array([entry["coverage"] for entry in entries], dtype=float)
        cv_summary[key] = {
            "best_accuracy_mean": float(np.mean(accs)),
            "best_accuracy_std": float(np.std(accs)),
            "best_coverage_mean": float(np.mean(covs)),
            "best_coverage_std": float(np.std(covs)),
        }
        logging.info(
            "CV %s: acc=%.4f±%.4f cov=%.3f±%.3f",
            key,
            cv_summary[key]["best_accuracy_mean"],
            cv_summary[key]["best_accuracy_std"],
            cv_summary[key]["best_coverage_mean"],
            cv_summary[key]["best_coverage_std"],
        )
    return {"folds": cv_metrics, "summary": cv_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PASV gating for MATH")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Also run PASV on Qwen-4B if files exist",
    )
    parser.add_argument(
        "--include-qwen-14b",
        action="store_true",
        help="Also run PASV on Qwen-14B if files exist",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Eval path: %s", args.eval_path)
    logging.info("Candidates path: %s", args.candidates_path)
    logging.info("Output dir: %s", output_dir)

    dataset = build_dataset("math-14b", args.eval_path, args.candidates_path)
    results = evaluate_dataset(
        dataset,
        output_dir,
        seed=args.seed,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_lr=args.mlp_lr,
        mlp_weight_decay=args.mlp_weight_decay,
    )

    cv_results = run_cross_validation(
        dataset,
        seed=args.seed,
        mlp_epochs=max(10, args.mlp_epochs // 2),
        mlp_batch_size=args.mlp_batch_size,
        mlp_lr=args.mlp_lr,
        mlp_weight_decay=args.mlp_weight_decay,
    )
    results["cross_validation"] = cv_results
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logging.info("Updated results with cross-validation summary")

    if args.include_qwen:
        if QWEN_EVAL_PATH.exists() and QWEN_CANDIDATES_PATH.exists():
            qwen_output = output_dir / "qwen-4b"
            qwen_output.mkdir(parents=True, exist_ok=True)
            qwen_dataset = build_dataset(
                "qwen-4b", QWEN_EVAL_PATH, QWEN_CANDIDATES_PATH
            )
            evaluate_dataset(
                qwen_dataset,
                qwen_output,
                seed=args.seed,
                mlp_epochs=args.mlp_epochs,
                mlp_batch_size=args.mlp_batch_size,
                mlp_lr=args.mlp_lr,
                mlp_weight_decay=args.mlp_weight_decay,
            )
        else:
            logging.warning(
                "Qwen files not found (eval=%s candidates=%s)",
                QWEN_EVAL_PATH,
                QWEN_CANDIDATES_PATH,
            )

    if args.include_qwen_14b:
        if QWEN14B_EVAL_PATH.exists() and QWEN14B_CANDIDATES_PATH.exists():
            qwen14b_output = output_dir / "qwen-14b"
            qwen14b_output.mkdir(parents=True, exist_ok=True)
            qwen14b_dataset = build_dataset(
                "qwen-14b", QWEN14B_EVAL_PATH, QWEN14B_CANDIDATES_PATH
            )
            evaluate_dataset(
                qwen14b_dataset,
                qwen14b_output,
                seed=args.seed,
                mlp_epochs=args.mlp_epochs,
                mlp_batch_size=args.mlp_batch_size,
                mlp_lr=args.mlp_lr,
                mlp_weight_decay=args.mlp_weight_decay,
            )
        else:
            logging.warning(
                "Qwen-14B files not found (eval=%s candidates=%s)",
                QWEN14B_EVAL_PATH,
                QWEN14B_CANDIDATES_PATH,
            )


if __name__ == "__main__":
    main()
