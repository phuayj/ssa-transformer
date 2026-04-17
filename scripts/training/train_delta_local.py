#!/usr/bin/env python3
"""Train DeltaLocalSlotDecoder with LM + global verify + local verify losses."""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from universal.cdcl_tokenizer import CDCLTokenizer
from universal.slot_decoder import DeltaLocalSlotDecoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DeltaLocalDataset(Dataset):
    def __init__(
        self,
        sequences,
        loss_masks,
        verify_labels,
        neighbor_positions,
        neighbor_labels,
        assign_positions,
        max_seq_len,
        loss_weights=None,
    ):
        lengths = [
            len(sequences),
            len(loss_masks),
            len(verify_labels),
            len(neighbor_positions),
            len(neighbor_labels),
            len(assign_positions),
        ]
        if len(set(lengths)) != 1:
            raise ValueError(f"dataset fields length mismatch: {lengths}")
        if loss_weights is not None and len(loss_weights) != len(sequences):
            raise ValueError("loss_weights length mismatch")

        self.sequences = sequences
        self.loss_masks = loss_masks
        self.verify_labels = verify_labels
        self.neighbor_positions = neighbor_positions
        self.neighbor_labels = neighbor_labels
        self.assign_positions = assign_positions
        self.max_seq_len = int(max_seq_len)
        self.loss_weights = loss_weights

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):
        idx = int(idx)
        seq = list(self.sequences[idx])
        mask = list(self.loss_masks[idx])
        verify_label = int(self.verify_labels[idx])
        nb_pos = list(self.neighbor_positions[idx])
        nb_lab = list(self.neighbor_labels[idx])
        assign_pos = int(self.assign_positions[idx])
        weights = (
            list(self.loss_weights[idx]) if self.loss_weights is not None else None
        )

        if len(seq) != len(mask):
            raise ValueError("sequence/mask length mismatch")
        if len(nb_pos) != len(nb_lab):
            raise ValueError("neighbor_positions/labels length mismatch")

        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
            mask = mask[: self.max_seq_len]
            if weights is not None:
                weights = weights[: self.max_seq_len]

        if assign_pos >= len(seq):
            # Truncation can invalidate delta-local pointers from full-length traces.
            # Degrade to global-only verify path by using an in-range assign position
            # and empty neighbors (collate will emit zero neighbor mask/labels).
            assign_pos = max(0, len(seq) - 1)
            nb_pos = []
            nb_lab = []

        filtered_nb_pos: List[int] = []
        filtered_nb_lab: List[int] = []
        for p, y in zip(nb_pos, nb_lab):
            if int(p) < len(seq) and int(p) >= 0:
                filtered_nb_pos.append(int(p))
                filtered_nb_lab.append(int(y))

        if weights is not None:
            return (
                seq,
                mask,
                weights,
                verify_label,
                filtered_nb_pos,
                filtered_nb_lab,
                assign_pos,
            )
        return seq, mask, verify_label, filtered_nb_pos, filtered_nb_lab, assign_pos


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _collate_delta_batch(batch, max_neighbors: int):
    has_weights = len(batch[0]) == 7
    bsz = len(batch)
    max_len = max(len(item[0]) for item in batch)
    max_nb = min(
        int(max_neighbors),
        max(1, max(len(item[4 if has_weights else 3]) for item in batch)),
    )

    input_ids = torch.full((bsz, max_len), int(CDCLTokenizer.PAD), dtype=torch.long)
    loss_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    verify_labels = torch.zeros((bsz,), dtype=torch.long)
    assign_positions = torch.zeros((bsz,), dtype=torch.long)
    neighbor_positions = torch.zeros((bsz, max_nb), dtype=torch.long)
    neighbor_mask = torch.zeros((bsz, max_nb), dtype=torch.float32)
    neighbor_labels = torch.zeros((bsz, max_nb), dtype=torch.long)
    loss_weights = (
        torch.ones((bsz, max_len), dtype=torch.float32) if has_weights else None
    )

    for i, item in enumerate(batch):
        if has_weights:
            seq, lm, lw, gl, np_, nl, ap = item
        else:
            seq, lm, gl, np_, nl, ap = item
            lw = None

        sl = len(seq)
        input_ids[i, :sl] = torch.tensor(seq, dtype=torch.long)
        loss_mask[i, :sl] = torch.tensor(lm, dtype=torch.bool)
        attention_mask[i, :sl] = 1
        verify_labels[i] = int(gl)
        assign_positions[i] = int(ap)

        use_n = min(len(np_), max_nb)
        if use_n > 0:
            neighbor_positions[i, :use_n] = torch.tensor(np_[:use_n], dtype=torch.long)
            neighbor_labels[i, :use_n] = torch.tensor(nl[:use_n], dtype=torch.long)
            neighbor_mask[i, :use_n] = 1.0

        if has_weights and lw is not None:
            if loss_weights is None:
                raise RuntimeError("loss_weights tensor unexpectedly missing")
            loss_weights[i, :sl] = torch.tensor(lw, dtype=torch.float32)

    if has_weights:
        if loss_weights is None:
            raise RuntimeError("loss_weights tensor unexpectedly missing at return")
        return (
            input_ids,
            attention_mask,
            loss_mask,
            loss_weights,
            verify_labels,
            neighbor_positions,
            neighbor_mask,
            assign_positions,
            neighbor_labels,
        )

    return (
        input_ids,
        attention_mask,
        loss_mask,
        verify_labels,
        neighbor_positions,
        neighbor_mask,
        assign_positions,
        neighbor_labels,
    )


def _compute_lm_loss(logits, input_ids, attention_mask, loss_mask, loss_weights=None):
    labels = input_ids[:, 1:].clone()
    logits = logits[:, :-1, :]
    label_mask = loss_mask[:, 1:] & (attention_mask[:, 1:] > 0)
    token_count = int(label_mask.sum().item())
    if token_count == 0:
        raise RuntimeError("no masked tokens in batch")

    if loss_weights is not None:
        w = loss_weights[:, 1:]
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=int(CDCLTokenizer.PAD),
            reduction="none",
        ).reshape(labels.shape)
        loss = (per_tok * w * label_mask.float()).sum() / label_mask.float().sum()
    else:
        labels = labels.masked_fill(~label_mask, int(CDCLTokenizer.PAD))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=int(CDCLTokenizer.PAD),
        )
    return loss, token_count, label_mask, labels


def _focal_loss_binary_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    p = F.softmax(logits, dim=-1)
    p_t = p[torch.arange(p.size(0), device=p.device), targets]
    alpha_t = torch.where(
        targets == 1,
        torch.full_like(p_t, float(alpha)),
        torch.full_like(p_t, 1.0 - float(alpha)),
    )
    loss = alpha_t * ((1.0 - p_t).clamp_min(1e-8) ** float(gamma)) * ce
    return loss.mean()


def _run_epoch(
    *,
    loader,
    model,
    optimizer,
    scheduler,
    device,
    tokenizer,
    train,
    global_weight,
    local_weight,
    focal_alpha,
    focal_gamma,
):
    if train:
        model.train()
        if optimizer is None:
            raise ValueError("optimizer required when train=True")
    else:
        model.eval()

    total_lm = total_g = total_l = total_loss = 0.0
    total_tokens = total_seq = total_lm_correct = 0

    g_tp = g_fp = g_fn = g_tn = 0
    l_tp = l_fp = l_fn = l_tn = 0
    global_conflicts = 0
    local_count = 0
    logged = False

    for batch in loader:
        if len(batch) == 9:
            (
                input_ids,
                attention_mask,
                loss_mask,
                loss_weights,
                verify_labels,
                neighbor_positions,
                neighbor_mask,
                assign_positions,
                neighbor_labels,
            ) = batch
        else:
            (
                input_ids,
                attention_mask,
                loss_mask,
                verify_labels,
                neighbor_positions,
                neighbor_mask,
                assign_positions,
                neighbor_labels,
            ) = batch
            loss_weights = None

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device)
        verify_labels = verify_labels.to(device)
        neighbor_positions = neighbor_positions.to(device)
        neighbor_mask = neighbor_mask.to(device)
        assign_positions = assign_positions.to(device)
        neighbor_labels = neighbor_labels.to(device)
        if loss_weights is not None:
            loss_weights = loss_weights.to(device)

        bsz = int(input_ids.size(0))

        with torch.set_grad_enabled(train):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                verify_labels=verify_labels,
                neighbor_positions=neighbor_positions,
                neighbor_mask=neighbor_mask,
                assign_positions=assign_positions,
                neighbor_labels=neighbor_labels,
            )
            lm_logits, global_logits, neighbor_logits, _aux = out

            lm_loss, tok_count, label_mask, labels = _compute_lm_loss(
                lm_logits, input_ids, attention_mask, loss_mask, loss_weights
            )
            global_loss = F.cross_entropy(global_logits, verify_labels)

            valid_local = neighbor_mask.to(torch.bool)
            if valid_local.any():
                flat_logits = neighbor_logits.reshape(-1, 2)
                flat_targets = neighbor_labels.reshape(-1)
                flat_valid = valid_local.reshape(-1)
                v_logits = flat_logits[flat_valid]
                v_targets = flat_targets[flat_valid]
                if float(focal_gamma) > 0.0:
                    local_loss = _focal_loss_binary_logits(
                        v_logits,
                        v_targets,
                        alpha=float(focal_alpha),
                        gamma=float(focal_gamma),
                    )
                else:
                    local_loss = F.cross_entropy(v_logits, v_targets)
            else:
                local_loss = torch.zeros((), device=device, dtype=lm_logits.dtype)

            loss = (
                lm_loss
                + float(global_weight) * global_loss
                + float(local_weight) * local_loss
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        preds = lm_logits[:, :-1, :].argmax(dim=-1)
        total_lm_correct += int(((preds == labels) & label_mask).sum().item())
        total_tokens += int(tok_count)

        g_pred = global_logits.argmax(dim=-1)
        g_tp += int(((g_pred == 1) & (verify_labels == 1)).sum().item())
        g_fp += int(((g_pred == 1) & (verify_labels == 0)).sum().item())
        g_fn += int(((g_pred == 0) & (verify_labels == 1)).sum().item())
        g_tn += int(((g_pred == 0) & (verify_labels == 0)).sum().item())
        global_conflicts += int(verify_labels.sum().item())

        if valid_local.any():
            l_pred = neighbor_logits.argmax(dim=-1)
            l_true = neighbor_labels
            l_valid = valid_local
            l_tp += int(((l_pred == 1) & (l_true == 1) & l_valid).sum().item())
            l_fp += int(((l_pred == 1) & (l_true == 0) & l_valid).sum().item())
            l_fn += int(((l_pred == 0) & (l_true == 1) & l_valid).sum().item())
            l_tn += int(((l_pred == 0) & (l_true == 0) & l_valid).sum().item())
            local_count += int(l_valid.sum().item())

        total_lm += float(lm_loss.item()) * float(tok_count)
        total_g += float(global_loss.item()) * float(bsz)
        total_l += float(local_loss.item()) * float(bsz)
        total_loss += float(loss.item()) * float(bsz)
        total_seq += int(bsz)

        if not logged:
            pos = torch.nonzero(label_mask[0], as_tuple=False).flatten().tolist()[:8]
            if pos:

                def _decode(tok):
                    try:
                        return tokenizer.decode_token(int(tok))
                    except ValueError:
                        return f"UNK({int(tok)})"

                sample_targets = [_decode(labels[0, p].item()) for p in pos]
                sample_preds = [_decode(preds[0, p].item()) for p in pos]
                logger.info("sample_targets=%s", sample_targets)
                logger.info("sample_preds=%s", sample_preds)

            g_prob = F.softmax(global_logits[:8].float(), dim=-1)[:, 1]
            logger.info(
                "global_labels=%s", [int(v) for v in verify_labels[:8].tolist()]
            )
            logger.info("global_preds=%s", [int(v) for v in g_pred[:8].tolist()])
            logger.info("global_cf_probs=%s", [f"{p:.3f}" for p in g_prob.tolist()])

            if valid_local.any():
                idx = (
                    torch.nonzero(valid_local[0], as_tuple=False).flatten().tolist()[:8]
                )
                logger.info(
                    "local_labels_sample=%s",
                    [int(neighbor_labels[0, j].item()) for j in idx],
                )
                logger.info(
                    "local_preds_sample=%s",
                    [int(neighbor_logits[0, j].argmax().item()) for j in idx],
                )
            logged = True

    lm_loss_avg = total_lm / max(float(total_tokens), 1.0)
    g_loss_avg = total_g / max(float(total_seq), 1.0)
    l_loss_avg = total_l / max(float(total_seq), 1.0)
    total_avg = total_loss / max(float(total_seq), 1.0)

    return {
        "loss": float(total_avg),
        "lm_loss": float(lm_loss_avg),
        "global_loss": float(g_loss_avg),
        "local_loss": float(l_loss_avg),
        "lm_accuracy": float(total_lm_correct) / max(float(total_tokens), 1.0),
        "global_precision": float(g_tp) / max(float(g_tp + g_fp), 1.0),
        "global_recall": float(g_tp) / max(float(g_tp + g_fn), 1.0),
        "local_precision": float(l_tp) / max(float(l_tp + l_fp), 1.0),
        "local_recall": float(l_tp) / max(float(l_tp + l_fn), 1.0),
        "global_conflict_rate": float(global_conflicts) / max(float(total_seq), 1.0),
        "local_positive_rate": float(l_tp + l_fn) / max(float(local_count), 1.0),
        "tokens": float(total_tokens),
        "local_examples": int(local_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeltaLocalSlotDecoder")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/delta-local")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_slots", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_neighbors", type=int, default=30)
    parser.add_argument("--n_colors", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--loss-weights", action="store_true", default=False)
    parser.add_argument("--global_weight", type=float, default=1.0)
    parser.add_argument("--local_weight", type=float, default=1.0)
    parser.add_argument("--neighbor_focal_alpha", type=float, default=0.75)
    parser.add_argument("--neighbor_focal_gamma", type=float, default=0.0)
    args = parser.parse_args()

    _set_seed(int(args.seed))

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")

    with data_path.open("rb") as f:
        payload = pickle.load(f)

    sequences = payload.get("sequences")
    loss_masks = payload.get("loss_masks")
    verify_labels = payload.get("verify_labels")
    neighbor_positions = payload.get("neighbor_positions")
    neighbor_labels = payload.get("neighbor_labels")
    assign_positions = payload.get("assign_positions")

    required = [
        ("sequences", sequences),
        ("loss_masks", loss_masks),
        ("verify_labels", verify_labels),
        ("neighbor_positions", neighbor_positions),
        ("neighbor_labels", neighbor_labels),
        ("assign_positions", assign_positions),
    ]
    missing = [name for name, value in required if value is None]
    if missing:
        raise KeyError(f"dataset missing required fields: {missing}")

    tokenizer = CDCLTokenizer()
    data_config = (
        payload.get("config") if isinstance(payload.get("config"), dict) else {}
    )
    model_vocab_size = int(
        data_config.get("vocab_size", int(tokenizer.PROP_VOCAB_SIZE))
    )

    loss_weights = payload.get("loss_weights") if bool(args.loss_weights) else None
    if bool(args.loss_weights) and loss_weights is None:
        logger.warning("--loss-weights set but dataset has no loss_weights")

    seq_lengths = [len(seq) for seq in sequences]
    neighbor_counts = [len(x) for x in neighbor_positions]
    logger.info(
        "dataset examples=%d mean_len=%.1f max_len=%d mean_neighbors=%.2f global_conflict_rate=%.4f",
        int(len(sequences)),
        float(np.mean(seq_lengths)),
        int(np.max(seq_lengths)),
        float(np.mean(neighbor_counts)),
        float(sum(int(v) for v in verify_labels)) / max(float(len(verify_labels)), 1.0),
    )

    rng = random.Random(int(args.seed))
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    val_size = max(1, int(0.1 * float(len(indices))))
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]

    def _pick(arr, ids):
        return [arr[i] for i in ids]

    train_dataset = DeltaLocalDataset(
        sequences=_pick(sequences, tr_idx),
        loss_masks=_pick(loss_masks, tr_idx),
        verify_labels=_pick(verify_labels, tr_idx),
        neighbor_positions=_pick(neighbor_positions, tr_idx),
        neighbor_labels=_pick(neighbor_labels, tr_idx),
        assign_positions=_pick(assign_positions, tr_idx),
        max_seq_len=int(args.max_seq_len),
        loss_weights=_pick(loss_weights, tr_idx) if loss_weights is not None else None,
    )
    val_dataset = DeltaLocalDataset(
        sequences=_pick(sequences, val_idx),
        loss_masks=_pick(loss_masks, val_idx),
        verify_labels=_pick(verify_labels, val_idx),
        neighbor_positions=_pick(neighbor_positions, val_idx),
        neighbor_labels=_pick(neighbor_labels, val_idx),
        assign_positions=_pick(assign_positions, val_idx),
        max_seq_len=int(args.max_seq_len),
        loss_weights=_pick(loss_weights, val_idx) if loss_weights is not None else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda b: _collate_delta_batch(
            b, max_neighbors=int(args.max_neighbors)
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=lambda b: _collate_delta_batch(
            b, max_neighbors=int(args.max_neighbors)
        ),
    )

    device = torch.device(str(args.device))
    model = DeltaLocalSlotDecoder(
        vocab_size=int(model_vocab_size),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        n_slots=int(args.n_slots),
        max_seq_len=int(args.max_seq_len),
        dropout=float(args.dropout),
        n_colors=int(args.n_colors),
        max_neighbors=int(args.max_neighbors),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    total_steps = int(len(train_loader) * int(args.epochs))
    warmup_steps = int(total_steps * float(args.warmup_fraction))

    def _lr_lambda(step: int) -> float:
        if step < int(warmup_steps):
            return float(step) / max(int(warmup_steps), 1)
        progress = float(step - int(warmup_steps)) / max(
            int(total_steps - int(warmup_steps)), 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "data": str(data_path),
        "vocab_size": int(model_vocab_size),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "n_slots": int(args.n_slots),
        "max_seq_len": int(args.max_seq_len),
        "max_neighbors": int(args.max_neighbors),
        "n_colors": int(args.n_colors),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "global_weight": float(args.global_weight),
        "local_weight": float(args.local_weight),
        "neighbor_focal_alpha": float(args.neighbor_focal_alpha),
        "neighbor_focal_gamma": float(args.neighbor_focal_gamma),
        "loss_weights": bool(args.loss_weights),
        "warmup_fraction": float(args.warmup_fraction),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "device": str(args.device),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        tr = _run_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            tokenizer=tokenizer,
            train=True,
            global_weight=float(args.global_weight),
            local_weight=float(args.local_weight),
            focal_alpha=float(args.neighbor_focal_alpha),
            focal_gamma=float(args.neighbor_focal_gamma),
        )
        va = _run_epoch(
            loader=val_loader,
            model=model,
            optimizer=None,
            scheduler=None,
            device=device,
            tokenizer=tokenizer,
            train=False,
            global_weight=float(args.global_weight),
            local_weight=float(args.local_weight),
            focal_alpha=float(args.neighbor_focal_alpha),
            focal_gamma=float(args.neighbor_focal_gamma),
        )

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f train_lm=%.4f val_lm=%.4f train_g=%.4f val_g=%.4f train_l=%.4f val_l=%.4f",
            int(epoch),
            float(tr["loss"]),
            float(va["loss"]),
            float(tr["lm_loss"]),
            float(va["lm_loss"]),
            float(tr["global_loss"]),
            float(va["global_loss"]),
            float(tr["local_loss"]),
            float(va["local_loss"]),
        )
        logger.info(
            "epoch=%d train_g_prec=%.4f val_g_prec=%.4f train_g_rec=%.4f val_g_rec=%.4f train_l_prec=%.4f val_l_prec=%.4f train_l_rec=%.4f val_l_rec=%.4f",
            int(epoch),
            float(tr["global_precision"]),
            float(va["global_precision"]),
            float(tr["global_recall"]),
            float(va["global_recall"]),
            float(tr["local_precision"]),
            float(va["local_precision"]),
            float(tr["local_recall"]),
            float(va["local_recall"]),
        )

        if float(va["loss"]) < float(best_val):
            best_val = float(va["loss"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "epoch": int(epoch),
                    "val_loss": float(best_val),
                },
                output_dir / "best_model.pt",
            )
            logger.info("saved best_model.pt val_loss=%.4f", float(best_val))


if __name__ == "__main__":
    main()
