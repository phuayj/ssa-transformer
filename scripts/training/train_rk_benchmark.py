#!/usr/bin/env python3
"""Train r^k benchmark models on mixed (k, difficulty) cells."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rk_benchmark.generator import (
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


def _make_batch(
    batch_size: int,
    k: int,
    difficulty: int,
    num_records: int,
    key_len: int,
    alphabet_size: int,
    rng: random.Random,
    positive_rate: float,
    max_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = RkBenchmarkConfig(
        k=int(k),
        num_records=int(num_records),
        key_len=int(key_len),
        alphabet_size=int(alphabet_size),
        difficulty=int(difficulty),
        correlated=False,
        positive_rate=float(positive_rate),
        max_seq_len=2048,
    )

    all_tokens: List[torch.Tensor] = []
    all_labels: List[float] = []
    all_features: List[torch.Tensor] = []
    for _ in range(int(batch_size)):
        tokens, label, features = generate_example_with_metadata(cfg, rng)
        all_tokens.append(torch.tensor(tokens, dtype=torch.long))
        all_labels.append(float(label))
        padded_feat = torch.full((int(max_k),), fill_value=-1.0, dtype=torch.float)
        padded_feat[: int(k)] = torch.tensor(features, dtype=torch.float)
        all_features.append(padded_feat)

    seq_len = max(int(t.shape[0]) for t in all_tokens)
    input_ids = torch.full(
        (int(batch_size), seq_len), fill_value=int(PAD_TOKEN), dtype=torch.long
    )
    padding_mask = torch.ones((int(batch_size), seq_len), dtype=torch.bool)
    for i, t in enumerate(all_tokens):
        n = int(t.shape[0])
        input_ids[i, :n] = t
        padding_mask[i, :n] = False

    labels = torch.tensor(all_labels, dtype=torch.float)
    oracle = torch.stack(all_features, dim=0)
    return input_ids, padding_mask, labels, oracle


@torch.no_grad()
def _quick_eval(
    model: nn.Module,
    model_type: str,
    device: torch.device,
    eval_k_values: Sequence[int],
    eval_difficulties: Sequence[int],
    num_eval_per_cell: int,
    num_records: int,
    key_len: int,
    alphabet_size: int,
    max_k: int,
    seed: int,
    feature_corruption_r: float,
    eval_batch_size: int,
    use_amp: bool,
    block_masked: bool,
) -> float:
    model.eval()
    rng = random.Random(int(seed))
    total_correct = 0
    total_count = 0

    for k in eval_k_values:
        for difficulty in eval_difficulties:
            for start in range(0, int(num_eval_per_cell), int(eval_batch_size)):
                end = min(start + int(eval_batch_size), int(num_eval_per_cell))
                batch_n = int(end - start)
                input_ids, padding_mask, labels, oracle = _make_batch(
                    batch_size=batch_n,
                    k=int(k),
                    difficulty=int(difficulty),
                    num_records=int(num_records),
                    key_len=int(key_len),
                    alphabet_size=int(alphabet_size),
                    rng=rng,
                    positive_rate=0.5,
                    max_k=int(max_k),
                )
                input_ids = input_ids.to(device)
                padding_mask = padding_mask.to(device)
                labels = labels.to(device)
                oracle = _corrupt_features(oracle.to(device), feature_corruption_r)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    if model_type == "transformer":
                        block_mask = (
                            create_block_attention_mask(input_ids)
                            if bool(block_masked)
                            else None
                        )
                        logits = model(
                            input_ids=input_ids,
                            padding_mask=padding_mask,
                            block_mask=block_mask,
                        ).squeeze(-1)
                    else:
                        logits = model(oracle).squeeze(-1)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                total_correct += int((preds == labels).sum().item())
                total_count += int(labels.numel())

    return float(total_correct) / max(total_count, 1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train r^k benchmark model")
    parser.add_argument(
        "--model_type", choices=["transformer", "oracle_mlp"], default="transformer"
    )
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_k", type=int, default=32)
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--mlp_num_layers", type=int, default=2)
    parser.add_argument("--train_k_values", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--train_difficulties", type=str, default="0,1,2,3,4")
    parser.add_argument("--num_records", type=int, default=8)
    parser.add_argument("--key_len", type=int, default=4)
    parser.add_argument("--alphabet_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--quick_eval_examples", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--feature_corruption_r", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--block_masked", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="experiments/rk_benchmark/")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    train_k_values = _parse_csv_ints(args.train_k_values)
    train_difficulties = _parse_csv_ints(args.train_difficulties)
    device = torch.device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    if gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_type == "transformer":
        model: nn.Module = RkTransformer(
            vocab_size=16,
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            num_layers=int(args.num_layers),
            max_seq_len=2048,
        )
    else:
        model = RkOracleMLP(
            max_k=int(args.max_k),
            hidden_dim=int(args.mlp_hidden_dim),
            num_layers=int(args.mlp_num_layers),
        )
    if bool(args.compile):
        model = torch.compile(model)
    model.to(device)

    logger.info(
        "training setup model_type=%s block_masked=%s amp=%s grad_accum=%d",
        str(args.model_type),
        str(bool(args.block_masked)),
        str(use_amp),
        int(gradient_accumulation_steps),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_eval_acc = -1.0
    best_path = Path(args.output_dir) / "best_model.pt"

    rng = random.Random(int(args.seed) + 1337)
    model.train()

    for step in range(1, int(args.num_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for accum_step in range(gradient_accumulation_steps):
            k = int(rng.choice(train_k_values))
            difficulty = int(rng.choice(train_difficulties))
            input_ids, padding_mask, labels, oracle = _make_batch(
                batch_size=int(args.batch_size),
                k=k,
                difficulty=difficulty,
                num_records=int(args.num_records),
                key_len=int(args.key_len),
                alphabet_size=int(args.alphabet_size),
                rng=rng,
                positive_rate=0.5,
                max_k=int(args.max_k),
            )

            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)
            oracle = _corrupt_features(
                oracle.to(device), float(args.feature_corruption_r)
            )

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                if args.model_type == "transformer":
                    block_mask = (
                        create_block_attention_mask(input_ids)
                        if bool(args.block_masked)
                        else None
                    )
                    logits = model(
                        input_ids=input_ids,
                        padding_mask=padding_mask,
                        block_mask=block_mask,
                    ).squeeze(-1)
                else:
                    logits = model(oracle).squeeze(-1)

                loss = loss_fn(logits, labels) / gradient_accumulation_steps

            scaler.scale(loss).backward()
            step_loss += float(loss.item())

        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                batch_acc = float((preds == labels).float().mean().item())
                logger.info(
                    "step=%d loss=%.6f batch_acc=%.4f k=%d difficulty=%d sample_pred=%s sample_target=%s",
                    int(step),
                    float(step_loss),
                    float(batch_acc),
                    int(k),
                    int(difficulty),
                    preds[:5].tolist(),
                    labels[:5].tolist(),
                )

        if step % int(args.save_every) == 0 and step > 0:
            latest_path = Path(args.output_dir) / "latest_model.pt"
            payload = {
                "model_type": str(args.model_type),
                "model_state_dict": model.state_dict(),
                "step": int(step),
                "config": vars(args),
            }
            torch.save(payload, latest_path)
            logger.info(
                "saved latest checkpoint to %s at step=%d", str(latest_path), int(step)
            )

        if step % int(args.eval_every) == 0 or step == int(args.num_steps):
            eval_acc = _quick_eval(
                model=model,
                model_type=str(args.model_type),
                device=device,
                eval_k_values=train_k_values,
                eval_difficulties=train_difficulties,
                num_eval_per_cell=int(args.quick_eval_examples),
                num_records=int(args.num_records),
                key_len=int(args.key_len),
                alphabet_size=int(args.alphabet_size),
                max_k=int(args.max_k),
                seed=int(args.seed) + int(step),
                feature_corruption_r=float(args.feature_corruption_r),
                eval_batch_size=int(args.eval_batch_size),
                use_amp=use_amp,
                block_masked=bool(args.block_masked),
            )
            logger.info(
                "quick_eval step=%d avg_accuracy=%.4f", int(step), float(eval_acc)
            )

            if eval_acc > best_eval_acc:
                best_eval_acc = float(eval_acc)
                payload = {
                    "model_type": str(args.model_type),
                    "model_state_dict": model.state_dict(),
                    "best_eval_accuracy": float(best_eval_acc),
                    "step": int(step),
                    "config": vars(args),
                }
                torch.save(payload, best_path)
                logger.info("saved new best model to %s", str(best_path))

    meta_path = Path(args.output_dir) / "train_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_eval_accuracy": float(best_eval_acc),
                "best_checkpoint": str(best_path),
                "args": vars(args),
            },
            f,
            indent=2,
        )
    logger.info("training complete. best_eval_accuracy=%.4f", float(best_eval_acc))


if __name__ == "__main__":
    main()
