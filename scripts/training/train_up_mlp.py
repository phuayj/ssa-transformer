#!/usr/bin/env python3
"""Train a tiny MLP on handcrafted UP-conflict features.

Usage:
    python scripts/train_up_mlp.py \
        --dataset experiments/sat-up-verifier/up_conflict_data.pkl \
        --output_dir experiments/sat-up-verifier/mlp_diagnostic
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sat.generator import SatGenerator

try:
    from pysat.solvers import Minisat22  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional runtime dep
    Minisat22 = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def extract_features(
    clauses: List[List[int]],
    assumptions: List[int],
    proposed_lit: int,
    num_vars: int = 20,
) -> List[float]:
    """Extract features from SAT state for UP-conflict prediction."""
    if not clauses:
        raise ValueError("clauses must be non-empty")

    assigned = set(abs(a) for a in assumptions)
    assignment = {abs(a): (a > 0) for a in assumptions}
    prop_var = abs(proposed_lit)
    prop_val = proposed_lit > 0

    features: List[float] = []

    # Global features
    features.append(len(assigned) / float(num_vars))  # fraction assigned
    features.append(float(len(clauses)))  # num clauses (normalized later)

    # Clause status features (assuming current assignment + proposed literal)
    full_assignment = dict(assignment)
    full_assignment[prop_var] = prop_val

    n_satisfied = 0
    n_unit = 0  # exactly 1 unset literal, no true literal
    n_empty = 0  # all literals false (contradiction!)
    n_unresolved = 0  # >1 unset literal, no true literal

    for clause in clauses:
        has_true = False
        n_unset = 0
        for lit in clause:
            var = abs(lit)
            if var in full_assignment:
                if (lit > 0) == full_assignment[var]:
                    has_true = True
                else:
                    pass
            else:
                n_unset += 1

        if has_true:
            n_satisfied += 1
        elif n_unset == 0:
            n_empty += 1  # contradiction
        elif n_unset == 1:
            n_unit += 1
        else:
            n_unresolved += 1

    clause_count = float(len(clauses))
    features.append(n_satisfied / clause_count)
    features.append(n_unit / clause_count)
    features.append(n_empty / clause_count)  # THIS is the key signal!
    features.append(n_unresolved / clause_count)

    # Proposed variable features
    pos_count = sum(1 for c in clauses for l in c if l == prop_var)
    neg_count = sum(1 for c in clauses for l in c if l == -prop_var)
    features.append(pos_count / clause_count)
    features.append(neg_count / clause_count)
    features.append(1.0 if prop_val else 0.0)

    return features  # 9 features


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[int(idx)], self.labels[int(idx)]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _compute_metrics(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    preds = probs >= float(threshold)
    labels_bool = labels.astype(bool)

    tp = int(np.sum(preds & labels_bool))
    fp = int(np.sum(preds & ~labels_bool))
    fn = int(np.sum(~preds & labels_bool))
    tn = int(np.sum(~preds & ~labels_bool))

    precision = float(tp) / max(float(tp + fp), 1.0)
    recall = float(tp) / max(float(tp + fn), 1.0)
    f1 = (
        2.0 * precision * recall / max(float(precision + recall), 1e-12)
        if (precision + recall) > 0.0
        else 0.0
    )
    accuracy = float(tp + tn) / max(float(tp + tn + fp + fn), 1.0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _calibrate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.9,
    steps: int = 200,
) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, int(steps) + 1)
    best_threshold: float | None = None
    best_metrics: Dict[str, float] | None = None

    for t in thresholds:
        metrics = _compute_metrics(probs, labels, float(t))
        metrics = dict(metrics)
        metrics["threshold"] = float(t)

        if metrics["precision"] >= float(target_precision):
            if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                best_threshold = float(t)
                best_metrics = metrics

    if best_metrics is None:
        for t in thresholds:
            metrics = _compute_metrics(probs, labels, float(t))
            metrics = dict(metrics)
            metrics["threshold"] = float(t)
            if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                best_threshold = float(t)
                best_metrics = metrics

    if best_metrics is None or best_threshold is None:
        raise RuntimeError("failed to calibrate threshold")
    return float(best_threshold), best_metrics


@torch.no_grad()
def _collect_probs(
    model: SimpleMLP, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: List[float] = []
    labels: List[float] = []
    for features, targets in loader:
        features = features.to(device)
        logits = model(features)
        logit = logits[:, 1] - logits[:, 0]
        batch_probs = torch.sigmoid(logit).cpu().numpy()
        probs.extend(float(x) for x in batch_probs.tolist())
        labels.extend(float(x) for x in targets.numpy().tolist())
    return np.array(probs, dtype=np.float32), np.array(labels, dtype=np.float32)


def _run_epoch(
    *,
    model: SimpleMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
    train: bool,
) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0
    total = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(train):
            logits = model(features)
            logit = logits[:, 1] - logits[:, 0]
            loss = criterion(logit, targets)
            if train:
                if optimizer is None:
                    raise ValueError("optimizer required for training")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = int(features.shape[0])
        total_loss += float(loss.item()) * float(batch_size)
        total += batch_size

    return float(total_loss) / max(float(total), 1.0)


def _clause_status(clause: Iterable[int], assignment: Dict[int, bool]) -> int:
    has_unassigned = False
    for lit in clause:
        var = abs(int(lit)) - 1
        if int(var) not in assignment:
            has_unassigned = True
            continue
        val = bool(assignment[int(var)])
        if (int(lit) > 0 and bool(val)) or (int(lit) < 0 and not bool(val)):
            return 1
    if bool(has_unassigned):
        return 0
    return -1


def _is_conflict(clauses: List[List[int]], assignment: Dict[int, bool]) -> bool:
    return any(int(_clause_status(clause, assignment)) == -1 for clause in clauses)


def _is_satisfied(clauses: List[List[int]], assignment: Dict[int, bool]) -> bool:
    return all(int(_clause_status(clause, assignment)) == 1 for clause in clauses)


def _var_scores(
    clauses: List[List[int]], num_vars: int
) -> Tuple[List[int], List[int], List[int]]:
    scores = [0 for _ in range(int(num_vars))]
    pos_scores = [0 for _ in range(int(num_vars))]
    neg_scores = [0 for _ in range(int(num_vars))]
    for clause in clauses:
        for lit in clause:
            var = abs(int(lit)) - 1
            scores[int(var)] += 1
            if int(lit) > 0:
                pos_scores[int(var)] += 1
            else:
                neg_scores[int(var)] += 1
    return scores, pos_scores, neg_scores


def _select_var(
    num_vars: int, assignment: Dict[int, bool], scores: List[int]
) -> int | None:
    unassigned = [int(v) for v in range(int(num_vars)) if int(v) not in assignment]
    if not unassigned:
        return None
    return int(max(unassigned, key=lambda v: (int(scores[int(v)]), -int(v))))


def _select_polarity(
    var: int, pos_scores: List[int], neg_scores: List[int], rng: random.Random
) -> bool:
    if int(pos_scores[int(var)]) > int(neg_scores[int(var)]):
        return True
    if int(neg_scores[int(var)]) > int(pos_scores[int(var)]):
        return False
    return bool(rng.choice([True, False]))


@torch.no_grad()
def _mlp_conflict_prob(
    *,
    model: SimpleMLP,
    clauses: List[List[int]],
    assumptions: List[int],
    proposed_lit: int,
    num_vars: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
) -> float:
    feat = np.array(
        extract_features(clauses, assumptions, int(proposed_lit), int(num_vars)),
        dtype=np.float32,
    )
    feat = (feat - feature_mean) / feature_std
    tensor = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
    logits = model(tensor)
    logit = logits[:, 1] - logits[:, 0]
    prob = torch.sigmoid(logit).item()
    return float(prob)


def _check_literal(
    *,
    mode: str,
    lit: int,
    clauses: List[List[int]],
    assumptions: List[int],
    solver: Minisat22 | None,
    model: SimpleMLP | None,
    num_vars: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    threshold: float,
) -> Tuple[bool, float]:
    if str(mode) == "greedy":
        return True, float("nan")
    if str(mode) == "oracle":
        if solver is None:
            raise RuntimeError("oracle requires a solver")
        ok, _implied = solver.propagate(assumptions=list(assumptions) + [int(lit)])
        return bool(ok), float("nan")
    if str(mode) == "mlp":
        if model is None:
            raise RuntimeError("mlp mode requires a model")
        prob = _mlp_conflict_prob(
            model=model,
            clauses=clauses,
            assumptions=assumptions,
            proposed_lit=int(lit),
            num_vars=int(num_vars),
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
        )
        return bool(float(prob) <= float(threshold)), float(prob)
    raise ValueError(f"unknown mode: {mode}")


@dataclass
class Decision:
    var: int
    tried: set[bool]


def _backtrack(
    *,
    decision_stack: List[Decision],
    assignment: Dict[int, bool],
    trail: List[int],
    mode: str,
    clauses: List[List[int]],
    solver: Minisat22 | None,
    model: SimpleMLP | None,
    num_vars: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    threshold: float,
) -> Tuple[bool, float]:
    while decision_stack:
        last = decision_stack.pop()
        var = int(last.var)
        if trail:
            trail.pop()
        if int(var) in assignment:
            assignment.pop(int(var))

        for alt_val in (True, False):
            if bool(alt_val) in last.tried:
                continue
            alt_lit = int(var + 1) if bool(alt_val) else -(int(var) + 1)
            allowed, prob = _check_literal(
                mode=mode,
                lit=int(alt_lit),
                clauses=clauses,
                assumptions=list(trail),
                solver=solver,
                model=model,
                num_vars=int(num_vars),
                feature_mean=feature_mean,
                feature_std=feature_std,
                device=device,
                threshold=float(threshold),
            )
            last.tried.add(bool(alt_val))
            if bool(allowed):
                assignment[int(var)] = bool(alt_val)
                trail.append(int(alt_lit))
                decision_stack.append(last)
                return True, float(prob)
    return False, float("nan")


def _eval_rollout(
    *,
    model: SimpleMLP | None,
    clauses: List[List[int]],
    num_vars: int,
    mode: str,
    threshold: float,
    max_steps: int,
    device: torch.device,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    rng: random.Random,
) -> Dict[str, float]:
    assignment: Dict[int, bool] = {}
    trail: List[int] = []
    decision_stack: List[Decision] = []
    scores, pos_scores, neg_scores = _var_scores(clauses, int(num_vars))
    solver: Minisat22 | None = None
    if str(mode) == "oracle":
        if Minisat22 is None:
            raise RuntimeError("pysat is required for oracle mode")
        solver = Minisat22(bootstrap_with=[list(c) for c in clauses])

    stats: Dict[str, float] = {
        "steps": 0,
        "backtracks": 0,
        "backtracked": 0,
        "solved": 0,
    }

    for step in range(int(max_steps)):
        if _is_satisfied(clauses, assignment):
            stats["solved"] = 1
            break

        if _is_conflict(clauses, assignment):
            ok, _prob = _backtrack(
                decision_stack=decision_stack,
                assignment=assignment,
                trail=trail,
                mode=mode,
                clauses=clauses,
                solver=solver,
                model=model,
                num_vars=int(num_vars),
                feature_mean=feature_mean,
                feature_std=feature_std,
                device=device,
                threshold=float(threshold),
            )
            stats["backtracks"] += 1
            stats["backtracked"] = 1
            if not bool(ok):
                break
            continue

        if int(len(assignment)) == int(num_vars):
            ok, _prob = _backtrack(
                decision_stack=decision_stack,
                assignment=assignment,
                trail=trail,
                mode=mode,
                clauses=clauses,
                solver=solver,
                model=model,
                num_vars=int(num_vars),
                feature_mean=feature_mean,
                feature_std=feature_std,
                device=device,
                threshold=float(threshold),
            )
            stats["backtracks"] += 1
            stats["backtracked"] = 1
            if not bool(ok):
                break
            continue

        var = _select_var(int(num_vars), assignment, scores)
        if var is None:
            break

        proposed_val = _select_polarity(int(var), pos_scores, neg_scores, rng)
        proposed_lit = int(var + 1) if bool(proposed_val) else -(int(var) + 1)

        allowed, prob = _check_literal(
            mode=mode,
            lit=int(proposed_lit),
            clauses=clauses,
            assumptions=list(trail),
            solver=solver,
            model=model,
            num_vars=int(num_vars),
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
            threshold=float(threshold),
        )

        tried: set[bool] = set()
        if not bool(allowed):
            tried.add(bool(proposed_val))
            alt_val = not bool(proposed_val)
            alt_lit = int(var + 1) if bool(alt_val) else -(int(var) + 1)
            allowed_alt, alt_prob = _check_literal(
                mode=mode,
                lit=int(alt_lit),
                clauses=clauses,
                assumptions=list(trail),
                solver=solver,
                model=model,
                num_vars=int(num_vars),
                feature_mean=feature_mean,
                feature_std=feature_std,
                device=device,
                threshold=float(threshold),
            )
            tried.add(bool(alt_val))
            if not bool(allowed_alt):
                ok, _ = _backtrack(
                    decision_stack=decision_stack,
                    assignment=assignment,
                    trail=trail,
                    mode=mode,
                    clauses=clauses,
                    solver=solver,
                    model=model,
                    num_vars=int(num_vars),
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    device=device,
                    threshold=float(threshold),
                )
                stats["backtracks"] += 1
                stats["backtracked"] = 1
                if not bool(ok):
                    break
                continue
            chosen_val = bool(alt_val)
            prob = float(alt_prob)
        else:
            chosen_val = bool(proposed_val)
            tried.add(bool(proposed_val))

        chosen_lit = int(var + 1) if bool(chosen_val) else -(int(var) + 1)
        assignment[int(var)] = bool(chosen_val)
        trail.append(int(chosen_lit))
        decision_stack.append(Decision(var=int(var), tried=tried))
        stats["steps"] += 1

        if int(step) < 3 and str(mode) == "mlp":
            logger.info(
                "mode=%s step=%d var=%d val=%d prob=%.3f",
                str(mode),
                int(step),
                int(var),
                1 if bool(chosen_val) else 0,
                float(prob),
            )

    if solver is not None:
        solver.delete()
    return stats


def _summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    total = float(len(results))
    solved = int(sum(int(r.get("solved", 0)) for r in results))
    solve_rate = float(solved) / max(float(total), 1.0)
    backtracks = float(sum(float(r.get("backtracks", 0)) for r in results))
    backtracks_per_solve = float(backtracks) / max(float(solved), 1.0)
    backtracked_instances = int(sum(int(r.get("backtracked", 0)) for r in results))
    recovered_instances = int(
        sum(
            1
            for r in results
            if int(r.get("backtracked", 0)) > 0 and int(r.get("solved", 0)) > 0
        )
    )
    recovery_rate = float(recovered_instances) / max(float(backtracked_instances), 1.0)
    solved_with_backtracks = int(
        sum(
            1
            for r in results
            if int(r.get("solved", 0)) > 0 and int(r.get("backtracks", 0)) > 0
        )
    )

    return {
        "solve_rate": float(solve_rate),
        "backtracks_per_solve": float(backtracks_per_solve),
        "recovery_rate": float(recovery_rate),
        "backtracked_instances": int(backtracked_instances),
        "recovered_instances": int(recovered_instances),
        "solved_with_backtracks": int(solved_with_backtracks),
        "n": int(total),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a simple MLP on handcrafted UP-conflict features"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="experiments/sat-up-verifier/up_conflict_data.pkl",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/sat-up-verifier/mlp_diagnostic",
    )
    parser.add_argument("--eval_instances", type=int, default=200)
    parser.add_argument("--num_vars", type=int, default=20)
    parser.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help="Positive-class weight for BCE (also fallback eval alpha)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    _set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    device = torch.device(str(args.device))

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    with dataset_path.open("rb") as f:
        payload = pickle.load(f)

    examples_raw = payload.get("examples", [])
    stats_raw = payload.get("stats", {})
    config_raw = payload.get("config", {})
    if not examples_raw:
        raise RuntimeError("dataset has no examples")

    logger.info(
        "loaded dataset examples=%d conflict_rate=%.3f",
        int(len(examples_raw)),
        float(stats_raw.get("conflict_rate", 0.0)),
    )

    features: List[List[float]] = []
    labels: List[int] = []

    for ex in examples_raw:
        clauses = ex["clauses"]
        assumptions = [int(l) for l in ex["assumptions"]]
        proposed_lit = int(ex["proposed_lit"])
        label = 1 if bool(ex["up_conflict"]) else 0

        feat = extract_features(
            clauses=clauses,
            assumptions=assumptions,
            proposed_lit=int(proposed_lit),
            num_vars=int(args.num_vars),
        )
        features.append(feat)
        labels.append(int(label))

    features_arr = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)

    logger.info(
        "feature_stats shape=%s pos_rate=%.3f",
        str(features_arr.shape),
        float(labels_arr.mean()),
    )

    indices = np.arange(int(len(labels_arr)))
    rng.shuffle(indices)
    val_size = max(1, int(0.1 * len(indices)))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_features_raw = features_arr[train_idx]
    val_features_raw = features_arr[val_idx]
    train_labels = labels_arr[train_idx]
    val_labels = labels_arr[val_idx]

    feature_mean = train_features_raw.mean(axis=0)
    feature_std = train_features_raw.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    train_features = (train_features_raw - feature_mean) / feature_std
    val_features = (val_features_raw - feature_mean) / feature_std

    logger.info(
        "feature_normalization mean=%s std=%s",
        np.array2string(feature_mean, precision=3),
        np.array2string(feature_std, precision=3),
    )

    train_loader = DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    val_loader = DataLoader(
        FeatureDataset(val_features, val_labels),
        batch_size=int(args.batch_size),
        shuffle=False,
    )

    model = SimpleMLP(input_dim=9, hidden_dim=64)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    pos_weight = torch.tensor(float(args.alpha), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": str(dataset_path),
        "num_vars": int(args.num_vars),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "device": str(args.device),
        "eval_instances": int(args.eval_instances),
        "dataset_alpha": float(config_raw.get("alpha", float(args.alpha))),
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    training_log: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            train=True,
        )

        val_probs, val_targets = _collect_probs(model, val_loader, device)
        val_metrics = _compute_metrics(val_probs, val_targets, threshold=0.5)

        sample_count = min(5, len(val_probs))
        if sample_count > 0:
            sample_probs = ", ".join(
                f"{float(val_probs[i]):.3f}/{int(val_targets[i])}"
                for i in range(sample_count)
            )
        else:
            sample_probs = ""

        logger.info(
            "epoch=%d loss=%.4f val_precision=%.3f val_recall=%.3f val_f1=%.3f samples=%s",
            int(epoch),
            float(train_loss),
            float(val_metrics["precision"]),
            float(val_metrics["recall"]),
            float(val_metrics["f1"]),
            sample_probs,
        )

        training_log.append(
            {
                "epoch": int(epoch),
                "loss": float(train_loss),
                **{k: float(v) for k, v in val_metrics.items()},
            }
        )

    val_probs, val_targets = _collect_probs(model, val_loader, device)
    threshold, cal_metrics = _calibrate_threshold(val_probs, val_targets)

    logger.info("=" * 80)
    logger.info(
        "CALIBRATED threshold=%.3f precision=%.3f recall=%.3f f1=%.3f",
        float(threshold),
        float(cal_metrics["precision"]),
        float(cal_metrics["recall"]),
        float(cal_metrics["f1"]),
    )
    logger.info("=" * 80)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": run_config,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "threshold": float(threshold),
        },
        output_dir / "mlp_checkpoint.pt",
    )

    with (output_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    with (output_dir / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(cal_metrics, f, indent=2)

    if float(cal_metrics["precision"]) >= 0.9:
        if Minisat22 is None:
            raise ImportError(
                "pysat is required for backtracking eval. Install python-sat to use it."
            )
        eval_alpha = float(config_raw.get("alpha", float(args.alpha)))
        generator = SatGenerator(seed=int(args.seed))
        eval_instances = [
            generator.generate_planted(
                num_vars=int(args.num_vars), alpha=float(eval_alpha)
            )
            for _ in range(int(args.eval_instances))
        ]

        modes = ["greedy", "mlp", "oracle"]
        eval_results: Dict[str, Dict[str, float]] = {}
        max_steps = 500

        for mode in modes:
            results: List[Dict[str, float]] = []
            for inst in eval_instances:
                stats = _eval_rollout(
                    model=model if str(mode) == "mlp" else None,
                    clauses=[list(c) for c in inst.clauses],
                    num_vars=int(args.num_vars),
                    mode=str(mode),
                    threshold=float(threshold),
                    max_steps=int(max_steps),
                    device=device,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    rng=rng,
                )
                results.append(stats)
            summary = _summarize(results)
            eval_results[str(mode)] = summary
            logger.info(
                "mode=%s solve_rate=%.3f backtracks_per_solve=%.2f recovery_rate=%.3f",
                str(mode),
                float(summary["solve_rate"]),
                float(summary["backtracks_per_solve"]),
                float(summary["recovery_rate"]),
            )

        with (output_dir / "eval_results.json").open("w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
    else:
        logger.info(
            "precision %.3f < 0.90, skipping backtracking eval",
            float(cal_metrics["precision"]),
        )


if __name__ == "__main__":
    main()
