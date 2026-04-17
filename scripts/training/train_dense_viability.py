#!/usr/bin/env python3
# pyright: reportIncompatibleMethodOverride=false
"""Train dense SAT literal viability predictors."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sat.dense_viability_net import DenseViabilityNet, SharedMLP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _make_dataset(
    examples: List[dict], mean: np.ndarray, std: np.ndarray
) -> TensorDataset:
    features = np.stack([ex["features"] for ex in examples], axis=0).astype(np.float32)
    labels = np.stack([ex["labels"] for ex in examples], axis=0).astype(np.int8)
    mask = (labels >= 0).astype(np.float32)
    features = (features - mean[None, None, None, :]) / std[None, None, None, :]
    return TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels.astype(np.float32)),
        torch.from_numpy(mask),
    )


def _compute_norm_stats(examples: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.stack([ex["features"] for ex in examples], axis=0).astype(np.float32)
    labels = np.stack([ex["labels"] for ex in examples], axis=0).astype(np.int8)
    valid = labels >= 0
    if not np.any(valid):
        raise RuntimeError("no valid labels in training set")
    valid_feats = feats[valid]
    mean = valid_feats.mean(axis=0)
    std = valid_feats.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred_viable = probs >= 0.5
    pred_dead = ~pred_viable

    valid = mask > 0.5
    y = labels[valid]
    p_dead = pred_dead[valid]
    p_viable = pred_viable[valid]

    y_dead = y < 0.5
    y_viable = y >= 0.5

    tp_dead = int(torch.sum(p_dead & y_dead).item())
    fp_dead = int(torch.sum(p_dead & y_viable).item())
    fn_dead = int(torch.sum((~p_dead) & y_dead).item())
    tn_dead = int(torch.sum((~p_dead) & y_viable).item())

    precision_dead = float(tp_dead) / max(float(tp_dead + fp_dead), 1.0)
    recall_dead = float(tp_dead) / max(float(tp_dead + fn_dead), 1.0)
    accuracy = float(tp_dead + tn_dead) / max(
        float(tp_dead + tn_dead + fp_dead + fn_dead), 1.0
    )

    both_valid = (mask[:, :, 0] > 0.5) & (mask[:, :, 1] > 0.5)
    both_pred_dead = pred_dead[:, :, 0] & pred_dead[:, :, 1] & both_valid
    at_least_one_true_dead = (
        (labels[:, :, 0] < 0.5) | (labels[:, :, 1] < 0.5)
    ) & both_valid
    inter_tp = int(torch.sum(both_pred_dead & at_least_one_true_dead).item())
    inter_total = int(torch.sum(both_pred_dead).item())
    intersectional_precision = float(inter_tp) / max(float(inter_total), 1.0)

    return {
        "accuracy": float(accuracy),
        "dead_precision": float(precision_dead),
        "dead_recall": float(recall_dead),
        "intersectional_precision": float(intersectional_precision),
        "intersectional_count": float(inter_total),
    }


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    sparse_supervision: bool,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(mode=train)

    total_loss = 0.0
    total_batches = 0
    total_supervised_literals = 0.0
    total_examples = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []

    for features, labels, mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        logits = model(features)
        valid = mask > 0.5

        if train and bool(sparse_supervision):
            flat_valid = valid.reshape(valid.shape[0], -1)
            has_valid = torch.any(flat_valid, dim=1)
            sparse_valid = torch.zeros_like(flat_valid)
            if bool(torch.any(has_valid)):
                row_idx = torch.nonzero(has_valid, as_tuple=False).squeeze(1)
                sampled_idx = torch.multinomial(
                    flat_valid[has_valid].float(), 1
                ).squeeze(1)
                sparse_valid[row_idx, sampled_idx] = True
            valid = sparse_valid.reshape_as(valid)

        n_supervised = int(torch.sum(valid).item())
        if n_supervised == 0:
            continue
        loss = criterion(logits[valid], labels[valid])

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        total_supervised_literals += float(n_supervised)
        total_examples += float(features.shape[0])
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_mask.append(mask.detach().cpu())

    if not all_logits:
        raise RuntimeError("no supervised literals found in epoch")

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    mask_cat = torch.cat(all_mask, dim=0)
    metrics = _metrics_from_logits(logits_cat, labels_cat, mask_cat)
    metrics["loss"] = float(total_loss) / max(float(total_batches), 1.0)
    metrics["avg_supervised_literals_per_example"] = float(
        total_supervised_literals
    ) / max(float(total_examples), 1.0)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dense viability predictor")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["dvp", "mlp"], default="dvp")
    parser.add_argument("--n_slots", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--pos_weight", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sparse training supervision: one random unassigned literal per example",
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(str(args.device))

    with Path(args.dataset).open("rb") as f:
        payload = pickle.load(f)

    examples: List[dict] = payload.get("examples", [])
    if not examples:
        raise RuntimeError("dataset has no examples")

    by_instance: Dict[int, List[dict]] = {}
    for ex in examples:
        inst_id = int(ex.get("instance_id", -1))
        if inst_id < 0:
            inst_id = hash(tuple(tuple(c) for c in ex["clauses"]))
        by_instance.setdefault(int(inst_id), []).append(ex)

    instance_ids = list(by_instance.keys())
    rng = random.Random(int(args.seed))
    rng.shuffle(instance_ids)
    split = max(1, int(0.8 * len(instance_ids)))
    train_ids = set(instance_ids[:split])
    val_ids = set(instance_ids[split:])
    if not val_ids:
        val_ids = {instance_ids[-1]}
        train_ids.discard(instance_ids[-1])

    train_examples = [ex for iid in train_ids for ex in by_instance[iid]]
    val_examples = [ex for iid in val_ids for ex in by_instance[iid]]

    mean, std = _compute_norm_stats(train_examples)

    train_ds = _make_dataset(train_examples, mean, std)
    val_ds = _make_dataset(val_examples, mean, std)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

    num_vars = int(train_examples[0]["num_vars"])
    feature_dim = int(train_examples[0]["features"].shape[-1])

    if str(args.model) == "dvp":
        model: nn.Module = DenseViabilityNet(
            num_vars=num_vars,
            feature_dim=feature_dim,
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            n_layers=int(args.n_layers),
            n_slots=int(args.n_slots),
            dropout=0.1,
        )
    else:
        model = SharedMLP(
            feature_dim=feature_dim, hidden_dim=int(args.d_model), n_layers=2
        )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(args.pos_weight), device=device)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": str(args.dataset),
        "model": str(args.model),
        "num_vars": int(num_vars),
        "feature_dim": int(feature_dim),
        "n_slots": int(args.n_slots),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "pos_weight": float(args.pos_weight),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(args.device),
        "train_instances": int(len(train_ids)),
        "val_instances": int(len(val_ids)),
        "train_examples": int(len(train_examples)),
        "val_examples": int(len(val_examples)),
        "sparse": bool(args.sparse),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    history: List[Dict[str, float]] = []
    best_val_ip = -1.0

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            sparse_supervision=bool(args.sparse),
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
                sparse_supervision=False,
            )

        record = {
            "epoch": float(epoch),
            **{f"train_{k}": float(v) for k, v in train_metrics.items()},
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        }
        history.append(record)

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f train_sup_lits=%.2f val_sup_lits=%.2f val_acc=%.3f dead_p=%.3f dead_r=%.3f inter_p=%.3f inter_n=%d",
            int(epoch),
            float(train_metrics["loss"]),
            float(val_metrics["loss"]),
            float(train_metrics["avg_supervised_literals_per_example"]),
            float(val_metrics["avg_supervised_literals_per_example"]),
            float(val_metrics["accuracy"]),
            float(val_metrics["dead_precision"]),
            float(val_metrics["dead_recall"]),
            float(val_metrics["intersectional_precision"]),
            int(val_metrics["intersectional_count"]),
        )

        if float(val_metrics["intersectional_precision"]) > float(best_val_ip):
            best_val_ip = float(val_metrics["intersectional_precision"])
            ckpt = {
                "model_state": model.state_dict(),
                "config": config,
                "feature_mean": mean,
                "feature_std": std,
                "best_val": val_metrics,
            }
            torch.save(ckpt, output_dir / "best_model.pt")
            logger.info(
                "new best model: epoch=%d val_intersectional_precision=%.4f",
                int(epoch),
                float(best_val_ip),
            )

    with (output_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "training done; best validation intersectional_precision=%.4f",
        float(best_val_ip),
    )


if __name__ == "__main__":
    main()
