#!/usr/bin/env python3
"""Train SSASlotDecoder on GC traces augmented with TRIED markers."""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from universal.cdcl_tokenizer import CDCLTokenizer
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def compute_block_ids(sequence: Sequence[int]) -> List[int]:
    """Compute SSA block IDs from a token sequence.

    block_id=0 for graph prefix (before first STATE/TRIED), then 1..K for decision
    blocks. TRIED starts a block, and STATE immediately following a TRIED block's
    END_TRIED remains in the same block.
    """
    tok = CDCLTokenizer()
    state_tok = int(tok.STATE)
    tried_tok = int(tok.TRIED)

    block_ids: List[int] = []
    current_block = 0
    in_tried_section = False

    for raw_token in sequence:
        token = int(raw_token)
        if current_block == 0:
            if token == tried_tok:
                current_block = 1
                in_tried_section = True
            elif token == state_tok:
                current_block = 1
                in_tried_section = False
        else:
            if token == tried_tok:
                current_block += 1
                in_tried_section = True
            elif token == state_tok and not in_tried_section:
                current_block += 1
            elif token == state_tok and in_tried_section:
                in_tried_section = False

        block_ids.append(int(current_block))

    return block_ids


class SSATriedDataset(Dataset[Tuple[List[int], List[bool], List[int]]]):
    def __init__(self, records: Sequence[Dict[str, Any]], max_seq_len: int):
        self.records = list(records)
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[bool], List[int]]:
        item = self.records[int(idx)]
        seq = [int(x) for x in item["sequence"]]
        lm = [bool(x) for x in item["loss_mask"]]
        if len(seq) != len(lm):
            raise ValueError("sequence/loss_mask mismatch")

        blk = compute_block_ids(seq)
        if len(blk) != len(seq):
            raise ValueError("sequence/block_ids mismatch")

        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
            lm = lm[: self.max_seq_len]
            blk = blk[: self.max_seq_len]
        return seq, lm, blk


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _collate(batch):
    bsz = len(batch)
    max_len = max(len(item[0]) for item in batch)
    input_ids = torch.full((bsz, max_len), int(CDCLTokenizer.PAD), dtype=torch.long)
    loss_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    block_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, (seq, lm, blk) in enumerate(batch):
        sl = len(seq)
        input_ids[i, :sl] = torch.tensor(seq, dtype=torch.long)
        loss_mask[i, :sl] = torch.tensor(lm, dtype=torch.bool)
        attention_mask[i, :sl] = 1
        block_ids[i, :sl] = torch.tensor(blk, dtype=torch.long)
    return input_ids, attention_mask, loss_mask, block_ids


def _compute_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    labels = input_ids[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    supervised = loss_mask[:, 1:] & (attention_mask[:, 1:] > 0)
    token_count = int(supervised.sum().item())
    if token_count == 0:
        raise RuntimeError("no supervised tokens in batch")
    labels = labels.masked_fill(~supervised, int(CDCLTokenizer.PAD))
    loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=int(CDCLTokenizer.PAD),
    )
    return loss, token_count, supervised, labels


def _run_epoch(
    *,
    loader: DataLoader,
    model: SSASlotDecoder,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    train: bool,
    device: torch.device,
    tokenizer: CDCLTokenizer,
    mode: str,
) -> Dict[str, float]:
    if train:
        model.train()
        if optimizer is None:
            raise ValueError("optimizer is required for training")
        opt = optimizer
    else:
        model.eval()
        opt = None

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_seq = 0
    total_blocks = 0.0
    logged = False

    for input_ids, attention_mask, loss_mask, block_ids in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device)
        block_ids = block_ids.to(device)

        with torch.set_grad_enabled(train):
            if str(mode) == "ssa":
                lm_logits, _verify_logits = model(
                    input_ids, attention_mask, block_ids=block_ids
                )
            else:
                lm_logits, _verify_logits = model(input_ids, attention_mask)

            lm_loss, token_count, supervised, labels = _compute_lm_loss(
                logits=lm_logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            if train:
                assert opt is not None
                opt.zero_grad()
                lm_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                if scheduler is not None:
                    scheduler.step()

        preds = lm_logits[:, :-1, :].argmax(dim=-1)
        total_correct += int(((preds == labels) & supervised).sum().item())
        total_tokens += int(token_count)
        batch_size = int(input_ids.size(0))
        total_seq += int(batch_size)
        total_loss += float(lm_loss.item()) * float(batch_size)

        if str(mode) == "ssa":
            for row_idx in range(batch_size):
                row_valid = attention_mask[row_idx] > 0
                if bool(row_valid.any().item()):
                    total_blocks += float(block_ids[row_idx, row_valid].max().item())

        if not logged:
            row = 0
            valid_pos = torch.nonzero(supervised[row], as_tuple=False).squeeze(-1)
            if valid_pos.numel() > 0:
                p = int(valid_pos[0].item())
                target_tok = int(labels[row, p].item())
                pred_tok = int(preds[row, p].item())
                logger.info(
                    "sample_token train=%s mode=%s pos=%d target=%s pred=%s",
                    str(train),
                    str(mode),
                    int(p),
                    tokenizer.decode_token(target_tok),
                    tokenizer.decode_token(pred_tok),
                )

            if str(mode) == "ssa":
                valid_block_ids = block_ids[row].masked_select(attention_mask[row] > 0)
                if valid_block_ids.numel() > 0:
                    logger.info(
                        "sample_blocks train=%s mode=%s graph_prefix_tokens=%d n_blocks=%d",
                        str(train),
                        str(mode),
                        int((valid_block_ids == 0).sum().item()),
                        int(valid_block_ids.max().item()),
                    )
            logged = True

    stats = {
        "loss": float(total_loss / max(float(total_seq), 1.0)),
        "token_acc": float(total_correct / max(float(total_tokens), 1.0)),
        "tokens": float(total_tokens),
        "sequences": float(total_seq),
    }
    if str(mode) == "ssa":
        stats["mean_blocks_per_seq"] = float(total_blocks / max(float(total_seq), 1.0))
    return stats


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, list):
        records = payload
    elif (
        isinstance(payload, dict) and "sequences" in payload and "loss_masks" in payload
    ):
        records = [
            {"sequence": seq, "loss_mask": lm}
            for seq, lm in zip(payload["sequences"], payload["loss_masks"])
        ]
    else:
        raise ValueError("unsupported data format; expected list[dict] or dict payload")
    if not records:
        raise ValueError("empty training data")
    return records


def _infer_model_init_config_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = ckpt["model_state_dict"]
    if "position_embedding.weight" not in state_dict:
        raise RuntimeError(
            f"checkpoint missing position_embedding.weight: {checkpoint_path}"
        )

    inferred: Dict[str, Any] = {
        "max_seq_len": int(state_dict["position_embedding.weight"].shape[0])
    }
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(ckpt_cfg, dict):
        for key in ("d_model", "n_layers", "n_heads", "n_slots", "dropout"):
            if key in ckpt_cfg:
                inferred[key] = ckpt_cfg[key]

    return inferred


def _load_with_vocab_expansion(
    model: SSASlotDecoder,
    checkpoint_path: Path,
    target_vocab_size: int,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = dict(ckpt["model_state_dict"])
    src_vocab = int(state_dict["token_embedding.weight"].shape[0])

    expanded = False
    if src_vocab != int(target_vocab_size):
        if src_vocab + 2 != int(target_vocab_size):
            raise RuntimeError(
                f"cannot expand vocab from {src_vocab} to {target_vocab_size}; expected +2"
            )

        d_model = int(state_dict["token_embedding.weight"].shape[1])
        expanded_embed = torch.empty(
            (target_vocab_size, d_model),
            dtype=state_dict["token_embedding.weight"].dtype,
        )
        torch.nn.init.normal_(expanded_embed, mean=0.0, std=0.02)
        expanded_embed[:src_vocab] = state_dict["token_embedding.weight"]
        state_dict["token_embedding.weight"] = expanded_embed
        state_dict["lm_head.weight"] = expanded_embed.clone()
        expanded = True

    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value

    if skipped:
        logger.warning(
            "Skipped %d checkpoint keys due to shape mismatch: %s",
            len(skipped),
            skipped,
        )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.warning("Missing keys after checkpoint load: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint load: %s", unexpected)

    return {
        "source_vocab": src_vocab,
        "expanded": expanded,
        "skipped_mismatch": len(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GC SSA Slot decoder with next-token loss"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["ssa", "causal"], default="ssa")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_slots", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CDCLTokenizer()
    vocab_size = int(tokenizer.TRIED_VOCAB_SIZE)
    device = torch.device(str(args.device))

    model_cfg: Dict[str, Any] = {
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "max_seq_len": int(args.max_seq_len),
        "n_slots": int(args.n_slots),
        "dropout": float(args.dropout),
    }
    if str(args.init_checkpoint).strip():
        checkpoint_path = Path(args.init_checkpoint)
        inferred_cfg = _infer_model_init_config_from_checkpoint(checkpoint_path)
        for key, inferred_value in inferred_cfg.items():
            if key in model_cfg and model_cfg[key] != inferred_value:
                logger.info(
                    "overriding model %s from %s to checkpoint value %s",
                    key,
                    model_cfg[key],
                    inferred_value,
                )
            model_cfg[key] = inferred_value

    records = _load_records(Path(args.data_path))
    random.Random(int(args.seed)).shuffle(records)
    split = int(round((1.0 - float(args.val_split)) * len(records)))
    split = max(1, min(split, len(records) - 1))
    train_records = records[:split]
    val_records = records[split:]

    train_ds = SSATriedDataset(train_records, max_seq_len=int(model_cfg["max_seq_len"]))
    val_ds = SSATriedDataset(val_records, max_seq_len=int(model_cfg["max_seq_len"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    model = SSASlotDecoder(
        vocab_size=int(vocab_size),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        n_slots=int(model_cfg["n_slots"]),
        dropout=float(model_cfg["dropout"]),
    )

    init_meta: Dict[str, Any] = {"used": False}
    if str(args.init_checkpoint).strip():
        init_meta = _load_with_vocab_expansion(
            model=model,
            checkpoint_path=Path(args.init_checkpoint),
            target_vocab_size=int(vocab_size),
        )
        init_meta["used"] = True
        logger.info("initialized_from=%s meta=%s", str(args.init_checkpoint), init_meta)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    total_steps = int(len(train_loader) * int(args.epochs))
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(float(step + 1) / max(float(warmup_steps), 1.0), 1.0)
            if step < warmup_steps
            else 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (step - warmup_steps)
                    / max(float(total_steps - warmup_steps), 1.0)
                )
            )
        ),
    )

    best_val = float("inf")
    history: List[Dict[str, float]] = []

    logger.info(
        "start_training mode=%s train=%d val=%d vocab_size=%d batch=%d epochs=%d",
        str(args.mode),
        int(len(train_ds)),
        int(len(val_ds)),
        int(vocab_size),
        int(args.batch_size),
        int(args.epochs),
    )

    for epoch in range(int(args.epochs)):
        train_stats = _run_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            device=device,
            tokenizer=tokenizer,
            mode=str(args.mode),
        )
        val_stats = _run_epoch(
            loader=val_loader,
            model=model,
            optimizer=None,
            scheduler=None,
            train=False,
            device=device,
            tokenizer=tokenizer,
            mode=str(args.mode),
        )

        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_stats["loss"]),
            "train_token_acc": float(train_stats["token_acc"]),
            "val_loss": float(val_stats["loss"]),
            "val_token_acc": float(val_stats["token_acc"]),
            "lr": float(lr_now),
        }
        if str(args.mode) == "ssa":
            row["train_mean_blocks_per_seq"] = float(
                train_stats.get("mean_blocks_per_seq", 0.0)
            )
            row["val_mean_blocks_per_seq"] = float(
                val_stats.get("mean_blocks_per_seq", 0.0)
            )
        history.append(row)

        logger.info(
            "epoch=%d/%d mode=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.2e",
            int(epoch + 1),
            int(args.epochs),
            str(args.mode),
            float(train_stats["loss"]),
            float(train_stats["token_acc"]),
            float(val_stats["loss"]),
            float(val_stats["token_acc"]),
            float(lr_now),
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": int(vocab_size),
                "d_model": int(model_cfg["d_model"]),
                "n_layers": int(model_cfg["n_layers"]),
                "n_heads": int(model_cfg["n_heads"]),
                "n_slots": int(model_cfg["n_slots"]),
                "max_seq_len": int(model_cfg["max_seq_len"]),
                "dropout": float(model_cfg["dropout"]),
                "mode": "gc_ssa",
                "attention_mode": str(args.mode),
            },
            "epoch": int(epoch + 1),
            "train_loss": float(train_stats["loss"]),
            "val_loss": float(val_stats["loss"]),
            "history": history,
            "init_meta": init_meta,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if float(val_stats["loss"]) < best_val:
            best_val = float(val_stats["loss"])
            torch.save(ckpt, output_dir / "best.pt")

    summary = {
        "mode": str(args.mode),
        "train_examples": int(len(train_ds)),
        "val_examples": int(len(val_ds)),
        "epochs": int(args.epochs),
        "best_val_loss": float(best_val),
        "history": history,
        "init_meta": init_meta,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "training_complete mode=%s best_val_loss=%.4f output_dir=%s",
        str(args.mode),
        best_val,
        output_dir,
    )


if __name__ == "__main__":
    main()
