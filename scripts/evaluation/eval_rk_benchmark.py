#!/usr/bin/env python3
"""Evaluate r^k benchmark models per cell and fit r^k curves."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rk_benchmark.generator import (
    BLOCK_END,
    BLOCK_START,
    CLS_TOKEN,
    PAD_TOKEN,
    RkBenchmarkConfig,
    generate_example_with_metadata,
)
from rk_benchmark.models import RkOracleMLP, RkTransformer, create_block_attention_mask

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _parse_csv_ints(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError(f"Expected non-empty CSV integer list, got {text!r}")
    return values


def _parse_bool(text: str) -> bool:
    t = str(text).strip().lower()
    if t in {"1", "true", "yes", "y"}:
        return True
    if t in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool value: {text}")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _corrupt_features(features: torch.Tensor, r: float) -> torch.Tensor:
    if r >= 1.0:
        return features
    if r <= 0.0:
        raise ValueError(f"feature corruption r must be > 0, got {r}")
    out = features.clone()
    valid = out >= 0.0
    flips = (torch.rand_like(out) < (1.0 - float(r))) & valid
    out[flips] = 1.0 - out[flips]
    return out


def _make_labeled_batch(
    num_examples: int,
    k: int,
    difficulty: int,
    num_records: int,
    key_len: int,
    alphabet_size: int,
    max_k: int,
    target_label: int,
    rng: random.Random,
    return_raw_tokens: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[List[List[int]]],
]:
    cfg = RkBenchmarkConfig(
        k=int(k),
        num_records=int(num_records),
        key_len=int(key_len),
        alphabet_size=int(alphabet_size),
        difficulty=int(difficulty),
        correlated=False,
        positive_rate=0.5,
        max_seq_len=2048,
    )

    all_tokens: List[torch.Tensor] = []
    all_labels: List[float] = []
    all_features: List[torch.Tensor] = []
    raw_tokens: List[List[int]] = []
    for _ in range(int(num_examples)):
        tokens, label, features = generate_example_with_metadata(
            config=cfg,
            rng=rng,
            target_label=int(target_label),
        )
        all_tokens.append(torch.tensor(tokens, dtype=torch.long))
        all_labels.append(float(label))
        if bool(return_raw_tokens):
            raw_tokens.append([int(t) for t in tokens])

        padded_feat = torch.full((int(max_k),), fill_value=-1.0, dtype=torch.float)
        padded_feat[: int(k)] = torch.tensor(features, dtype=torch.float)
        all_features.append(padded_feat)

    seq_len = max(int(t.shape[0]) for t in all_tokens)
    input_ids = torch.full(
        (int(num_examples), seq_len), fill_value=int(PAD_TOKEN), dtype=torch.long
    )
    padding_mask = torch.ones((int(num_examples), seq_len), dtype=torch.bool)
    for i, tokens in enumerate(all_tokens):
        n = int(tokens.shape[0])
        input_ids[i, :n] = tokens
        padding_mask[i, :n] = False

    labels = torch.tensor(all_labels, dtype=torch.float)
    oracle = torch.stack(all_features, dim=0)
    return (
        input_ids,
        padding_mask,
        labels,
        oracle,
        (raw_tokens if bool(return_raw_tokens) else None),
    )


def _pad_token_sequences(
    sequences: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Expected at least one sequence to pad")
    seq_len = max(len(seq) for seq in sequences)
    input_ids = torch.full(
        (len(sequences), seq_len), fill_value=int(PAD_TOKEN), dtype=torch.long
    )
    padding_mask = torch.ones((len(sequences), seq_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        n = len(seq)
        input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
        padding_mask[i, :n] = False
    return input_ids, padding_mask


def _split_into_blocks(tokens: List[int]) -> List[List[int]]:
    """Split a k-conjunction token sequence into standalone k=1 block sequences."""
    blocks: List[List[int]] = []
    current_block: List[int] = []
    in_block = False
    for token in tokens:
        if int(token) == int(BLOCK_START):
            in_block = True
            current_block = [int(BLOCK_START)]
        elif int(token) == int(BLOCK_END):
            if not in_block:
                raise ValueError("Encountered BLOCK_END outside of a block")
            current_block.append(int(BLOCK_END))
            blocks.append(current_block)
            in_block = False
            current_block = []
        elif in_block:
            current_block.append(int(token))

    if in_block:
        raise ValueError("Unterminated block while splitting tokens")

    return [[int(CLS_TOKEN)] + block for block in blocks]


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    model_type: str,
    input_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    oracle: torch.Tensor,
    feature_corruption_r: float,
    use_amp: bool = False,
    block_masked: bool = False,
) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
        if model_type == "transformer":
            block_mask = (
                create_block_attention_mask(input_ids) if bool(block_masked) else None
            )
            logits = model(
                input_ids=input_ids,
                padding_mask=padding_mask,
                block_mask=block_mask,
            ).squeeze(-1)
        else:
            oracle = _corrupt_features(oracle, float(feature_corruption_r))
            logits = model(oracle).squeeze(-1)
    probs = torch.sigmoid(logits)
    return (probs >= 0.5).float()


def _build_model(model_type: str, checkpoint: dict) -> torch.nn.Module:
    cfg = checkpoint.get("config", {})
    if model_type == "transformer":
        return RkTransformer(
            vocab_size=16,
            d_model=int(cfg.get("d_model", 256)),
            nhead=int(cfg.get("nhead", 8)),
            num_layers=int(cfg.get("num_layers", 6)),
            max_seq_len=2048,
        )
    return RkOracleMLP(
        max_k=int(cfg.get("max_k", 32)),
        hidden_dim=int(cfg.get("mlp_hidden_dim", 64)),
        num_layers=int(cfg.get("mlp_num_layers", 2)),
    )


def _plot_results(
    output_dir: Path,
    cells: List[dict],
    difficulties: Sequence[int],
    k_values: Sequence[int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable; skipping plots: %s", str(exc))
        return

    by_difficulty: Dict[int, Dict[int, dict]] = {int(d): {} for d in difficulties}
    for cell in cells:
        by_difficulty[int(cell["difficulty"])][int(cell["k"])] = cell

    # Positive accuracy vs k with r^k overlay.
    plt.figure(figsize=(8, 5))
    for d in difficulties:
        xs = [int(k) for k in k_values]
        ys = [float(by_difficulty[int(d)][int(k)]["positive_accuracy"]) for k in xs]
        pred = [float(by_difficulty[int(d)][int(k)]["r_k_predicted"]) for k in xs]
        plt.plot(xs, ys, marker="o", label=f"diff={int(d)} observed")
        plt.plot(xs, pred, linestyle="--", alpha=0.7, label=f"diff={int(d)} r^k")
    plt.xlabel("k")
    plt.ylabel("positive accuracy")
    plt.title("r^k benchmark: observed positive accuracy vs predicted r^k")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "positive_accuracy_vs_k.png", dpi=180)
    plt.close()

    # Log-linear plot.
    plt.figure(figsize=(8, 5))
    for d in difficulties:
        xs = [int(k) for k in k_values]
        ys = [
            max(1e-6, float(by_difficulty[int(d)][int(k)]["positive_accuracy"]))
            for k in xs
        ]
        pred = [
            max(1e-6, float(by_difficulty[int(d)][int(k)]["r_k_predicted"])) for k in xs
        ]
        plt.plot(xs, np.log(ys), marker="o", label=f"diff={int(d)} observed")
        plt.plot(
            xs, np.log(pred), linestyle="--", alpha=0.7, label=f"diff={int(d)} r^k"
        )
    plt.xlabel("k")
    plt.ylabel("log(positive accuracy)")
    plt.title("r^k benchmark: log-linear positive accuracy")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "log_positive_accuracy_vs_k.png", dpi=180)
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate r^k benchmark model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model_type", choices=["transformer", "oracle_mlp"], required=True
    )
    parser.add_argument("--eval_k_values", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--eval_difficulties", type=str, default="0,1,2,3,4")
    parser.add_argument("--eval_positive_only", type=str, default="true")
    parser.add_argument("--num_eval", type=int, default=5000)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_records", type=int, default=8)
    parser.add_argument("--key_len", type=int, default=4)
    parser.add_argument("--alphabet_size", type=int, default=8)
    parser.add_argument("--max_k", type=int, default=32)
    parser.add_argument("--feature_corruption_r", type=float, default=1.0)
    parser.add_argument("--block_masked", action="store_true")
    parser.add_argument("--factorized_eval", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output_dir", type=str, default="experiments/rk_benchmark/eval/"
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))
    eval_positive_only = _parse_bool(args.eval_positive_only)
    k_values = _parse_csv_ints(args.eval_k_values)
    difficulties = _parse_csv_ints(args.eval_difficulties)

    device = torch.device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    eval_batch_size = int(args.eval_batch_size)
    if eval_batch_size < 1:
        raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = _build_model(str(args.model_type), checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint=%s model_type=%s num_eval=%d eval_batch_size=%d amp=%s block_masked=%s",
        str(args.checkpoint),
        str(args.model_type),
        int(args.num_eval),
        int(eval_batch_size),
        str(use_amp),
        str(bool(args.block_masked)),
    )
    if eval_positive_only:
        logger.info(
            "eval_positive_only=true was set; full metrics are still computed for calibration"
        )

    factorized_eval_active = (
        bool(args.factorized_eval) and str(args.model_type) == "transformer"
    )
    if bool(args.factorized_eval) and not factorized_eval_active:
        logger.warning(
            "--factorized_eval requested but model_type=%s; factorized metrics require transformer and will be skipped",
            str(args.model_type),
        )

    rng = random.Random(int(args.seed) + 987)
    cells: List[dict] = []

    for k in k_values:
        for difficulty in difficulties:
            pos_correct = 0
            pos_total = 0
            factorized_raw_tokens: List[List[int]] = []
            factorized_oracle_features: List[List[float]] = []
            for start in range(0, int(args.num_eval), int(eval_batch_size)):
                batch_n = min(int(eval_batch_size), int(args.num_eval) - int(start))
                pos_input, pos_mask, pos_labels, pos_oracle, pos_raw_tokens = (
                    _make_labeled_batch(
                        num_examples=batch_n,
                        k=int(k),
                        difficulty=int(difficulty),
                        num_records=int(args.num_records),
                        key_len=int(args.key_len),
                        alphabet_size=int(args.alphabet_size),
                        max_k=int(args.max_k),
                        target_label=1,
                        rng=rng,
                        return_raw_tokens=bool(factorized_eval_active),
                    )
                )
                pos_preds = _predict(
                    model=model,
                    model_type=str(args.model_type),
                    input_ids=pos_input.to(device),
                    padding_mask=pos_mask.to(device),
                    oracle=pos_oracle.to(device),
                    feature_corruption_r=float(args.feature_corruption_r),
                    use_amp=use_amp,
                    block_masked=bool(args.block_masked),
                )
                pos_correct += int((pos_preds.cpu() == pos_labels).sum().item())
                pos_total += int(batch_n)

                if factorized_eval_active:
                    if pos_raw_tokens is None:
                        raise RuntimeError(
                            "Expected raw positive tokens for factorized evaluation"
                        )
                    factorized_raw_tokens.extend(pos_raw_tokens)
                    pos_oracle_k = pos_oracle[:, : int(k)].cpu().tolist()
                    factorized_oracle_features.extend(
                        [[float(v) for v in row] for row in pos_oracle_k]
                    )
            pos_acc = float(pos_correct) / max(int(pos_total), 1)

            neg_correct = 0
            neg_total = 0
            for start in range(0, int(args.num_eval), int(eval_batch_size)):
                batch_n = min(int(eval_batch_size), int(args.num_eval) - int(start))
                neg_input, neg_mask, neg_labels, neg_oracle, _ = _make_labeled_batch(
                    num_examples=batch_n,
                    k=int(k),
                    difficulty=int(difficulty),
                    num_records=int(args.num_records),
                    key_len=int(args.key_len),
                    alphabet_size=int(args.alphabet_size),
                    max_k=int(args.max_k),
                    target_label=0,
                    rng=rng,
                    return_raw_tokens=False,
                )
                neg_preds = _predict(
                    model=model,
                    model_type=str(args.model_type),
                    input_ids=neg_input.to(device),
                    padding_mask=neg_mask.to(device),
                    oracle=neg_oracle.to(device),
                    feature_corruption_r=float(args.feature_corruption_r),
                    use_amp=use_amp,
                    block_masked=bool(args.block_masked),
                )
                neg_correct += int((neg_preds.cpu() == neg_labels).sum().item())
                neg_total += int(batch_n)
            neg_acc = float(neg_correct) / max(int(neg_total), 1)
            overall_acc = 0.5 * (float(pos_acc) + float(neg_acc))

            cell = {
                "k": int(k),
                "difficulty": int(difficulty),
                "positive_accuracy": float(pos_acc),
                "negative_accuracy": float(neg_acc),
                "overall_accuracy": float(overall_acc),
            }

            if factorized_eval_active:
                if len(factorized_raw_tokens) != int(pos_total):
                    raise RuntimeError(
                        f"factorized token collection mismatch: got={len(factorized_raw_tokens)} expected={int(pos_total)}"
                    )
                if len(factorized_oracle_features) != int(pos_total):
                    raise RuntimeError(
                        f"factorized feature collection mismatch: got={len(factorized_oracle_features)} expected={int(pos_total)}"
                    )

                all_block_tokens: List[List[int]] = []
                all_block_labels: List[int] = []
                block_counts_per_example: List[int] = []
                for ex_idx, example_tokens in enumerate(factorized_raw_tokens):
                    split_blocks = _split_into_blocks(example_tokens)
                    if len(split_blocks) != int(k):
                        raise RuntimeError(
                            f"Expected {int(k)} blocks, got {len(split_blocks)} for example {ex_idx}"
                        )
                    block_counts_per_example.append(len(split_blocks))
                    block_values = factorized_oracle_features[ex_idx]
                    if len(block_values) != int(k):
                        raise RuntimeError(
                            f"Expected {int(k)} block labels, got {len(block_values)} for example {ex_idx}"
                        )
                    for blk_idx, block_tokens in enumerate(split_blocks):
                        all_block_tokens.append(block_tokens)
                        all_block_labels.append(int(block_values[blk_idx]))

                block_preds: List[int] = []
                block_eval_batch_size = int(eval_batch_size) * 4
                for start in range(0, len(all_block_tokens), block_eval_batch_size):
                    end = min(start + block_eval_batch_size, len(all_block_tokens))
                    batch_tokens = all_block_tokens[start:end]
                    block_input_ids, block_padding_mask = _pad_token_sequences(
                        batch_tokens
                    )
                    dummy_oracle = torch.zeros(
                        (len(batch_tokens), int(args.max_k)),
                        dtype=torch.float,
                        device=device,
                    )
                    batch_preds = _predict(
                        model=model,
                        model_type=str(args.model_type),
                        input_ids=block_input_ids.to(device),
                        padding_mask=block_padding_mask.to(device),
                        oracle=dummy_oracle,
                        feature_corruption_r=float(args.feature_corruption_r),
                        use_amp=use_amp,
                        block_masked=bool(args.block_masked),
                    )
                    block_preds.extend([int(x) for x in batch_preds.cpu().tolist()])

                if len(block_preds) != len(all_block_labels):
                    raise RuntimeError(
                        f"factorized predictions mismatch: preds={len(block_preds)} labels={len(all_block_labels)}"
                    )

                per_block_correct = [
                    int(pred == label)
                    for pred, label in zip(block_preds, all_block_labels)
                ]
                factorized_per_block_acc = float(sum(per_block_correct)) / max(
                    len(per_block_correct), 1
                )

                per_example_all_correct: List[int] = []
                cursor = 0
                for count in block_counts_per_example:
                    if cursor + count > len(per_block_correct):
                        raise RuntimeError("factorized per-example grouping overflow")
                    per_example_all_correct.append(
                        int(all(per_block_correct[cursor : cursor + count]))
                    )
                    cursor += count
                if cursor != len(per_block_correct):
                    raise RuntimeError("factorized per-example grouping underflow")
                factorized_conjunction_acc = float(sum(per_example_all_correct)) / max(
                    len(per_example_all_correct), 1
                )
                factorized_r_k_predicted = float(factorized_per_block_acc ** int(k))
                factorized_ratio = float(
                    factorized_conjunction_acc / max(factorized_r_k_predicted, 1e-8)
                )

                cell["full_model_positive_acc"] = float(pos_acc)
                cell["factorized_per_block_acc"] = float(factorized_per_block_acc)
                cell["factorized_conjunction_acc"] = float(factorized_conjunction_acc)
                cell["factorized_r_k_predicted"] = float(factorized_r_k_predicted)
                cell["factorized_ratio"] = float(factorized_ratio)

                logger.info(
                    "factorized cell k=%d difficulty=%d blocks=%d per_block_acc=%.4f conjunction_acc=%.4f r_k_pred=%.4f ratio=%.4f",
                    int(k),
                    int(difficulty),
                    len(all_block_tokens),
                    float(factorized_per_block_acc),
                    float(factorized_conjunction_acc),
                    float(factorized_r_k_predicted),
                    float(factorized_ratio),
                )

            logger.info(
                "cell k=%d difficulty=%d pos_acc=%.4f neg_acc=%s overall=%.4f pos_n=%d neg_n=%d",
                int(k),
                int(difficulty),
                float(pos_acc),
                "nan" if math.isnan(float(neg_acc)) else f"{float(neg_acc):.4f}",
                float(overall_acc),
                int(pos_total),
                int(neg_total),
            )
            cells.append(cell)

    # Calibrate r from k=1 per difficulty and compute r^k predictions.
    r_by_difficulty: Dict[int, float] = {}
    for d in difficulties:
        match = [
            c for c in cells if int(c["k"]) == 1 and int(c["difficulty"]) == int(d)
        ]
        if not match:
            raise RuntimeError(f"Missing k=1 calibration cell for difficulty={d}")
        r_by_difficulty[int(d)] = float(match[0]["positive_accuracy"])

    for c in cells:
        d = int(c["difficulty"])
        k = int(c["k"])
        r = float(r_by_difficulty[d])
        pred = float(r**k)
        c["r_from_k1"] = float(r)
        c["r_k_predicted"] = float(pred)
        c["ratio"] = float(c["positive_accuracy"] / max(pred, 1e-8))

    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "model_type": str(args.model_type),
            "eval_k_values": k_values,
            "eval_difficulties": difficulties,
            "eval_positive_only": bool(eval_positive_only),
            "num_eval": int(args.num_eval),
            "eval_batch_size": int(args.eval_batch_size),
            "num_records": int(args.num_records),
            "key_len": int(args.key_len),
            "alphabet_size": int(args.alphabet_size),
            "max_k": int(args.max_k),
            "feature_corruption_r": float(args.feature_corruption_r),
            "block_masked": bool(args.block_masked),
            "factorized_eval": bool(args.factorized_eval),
            "amp": bool(args.amp),
            "seed": int(args.seed),
            "device": str(args.device),
        },
        "r_by_difficulty": {str(k): float(v) for k, v in r_by_difficulty.items()},
        "cells": cells,
    }

    out_json = output_dir / "rk_eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("saved eval JSON to %s", str(out_json))

    _plot_results(
        output_dir=output_dir, cells=cells, difficulties=difficulties, k_values=k_values
    )
    logger.info("evaluation complete")


if __name__ == "__main__":
    main()
