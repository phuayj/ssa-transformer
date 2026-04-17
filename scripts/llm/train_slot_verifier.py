#!/usr/bin/env python3
"""Train SlotLMWrapper verification head on GSM8K candidate data.

Loads candidate solutions labeled as correct/incorrect.
Trains ONLY slot modules + verification head (base LLM frozen).
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from universal.slot_wrapper import SlotLMWrapper
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class CandidateDataset(Dataset):
    """Dataset of (prompt + candidate solution, correct/incorrect) pairs."""

    def __init__(self, records: List[dict], tokenizer, max_length: int = 2048):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for idx, record in enumerate(records):
            question = record.get("question") or record.get("problem", "")
            for cand in record["candidates"]:
                text = f"Question: {question}\n\nSolution: {cand['solution']}"
                label = int(cand["correct"])
                self.examples.append((text, label, idx))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        text, label, problem_id = self.examples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": label,
            "problem_id": problem_id,
        }


def collate_fn(batch, pad_token_id: int = 0) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    problem_ids = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["label"])
        problem_ids.append(item["problem_id"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "problem_ids": problem_ids,
    }


def compute_ranking_loss(
    logits: torch.Tensor, labels: torch.Tensor, problem_ids: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise ranking loss: correct should score higher than incorrect within same problem."""
    loss = torch.tensor(0.0, device=logits.device)
    count = 0
    unique_pids = problem_ids.unique()
    for pid in unique_pids:
        mask = problem_ids == pid
        pid_logits = logits[mask]
        pid_labels = labels[mask]
        pos_logits = pid_logits[pid_labels == 1]
        neg_logits = pid_logits[pid_labels == 0]
        if len(pos_logits) > 0 and len(neg_logits) > 0:
            pos = pos_logits[
                torch.randint(pos_logits.size(0), (1,), device=logits.device)
            ]
            neg = neg_logits[
                torch.randint(neg_logits.size(0), (1,), device=logits.device)
            ]
            loss = loss + torch.log1p(torch.exp(-(pos - neg)))
            count += 1
    return loss / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Ministral-3-14B-Base-2512"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="JSONL from generate_gsm8k_candidates"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_slots", type=int, default=32)
    parser.add_argument("--slot_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument(
        "--ranking_loss_weight",
        type=float,
        default=0.0,
        help="Weight for within-problem pairwise ranking loss (0=disabled)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Loading data from %s", args.data)
    records = []
    with open(args.data) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d problems", len(records))

    n_correct = sum(c["correct"] for r in records for c in r["candidates"])
    n_total = sum(len(r["candidates"]) for r in records)
    logger.info(
        "Class balance: %d correct / %d total (%.1f%%)",
        n_correct,
        n_total,
        100 * n_correct / max(n_total, 1),
    )

    random.shuffle(records)
    val_size = max(1, int(len(records) * args.val_split))
    val_records = records[:val_size]
    train_records = records[val_size:]
    logger.info(
        "Split: %d train, %d val problems", len(train_records), len(val_records)
    )

    logger.info(
        "Loading model %s with %d slots (dim=%d)...",
        args.model,
        args.n_slots,
        args.slot_dim,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SlotLMWrapper.from_pretrained(
        args.model,
        n_slots=args.n_slots,
        slot_dim=args.slot_dim,
        slot_mode="normal",
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    logger.info(
        "Total params: %d, Trainable: %d (%.2f%%)",
        model.num_total_params(),
        model.num_trainable_params(),
        100 * model.num_trainable_params() / model.num_total_params(),
    )

    train_dataset = CandidateDataset(train_records, tokenizer, args.max_length)
    val_dataset = CandidateDataset(val_records, tokenizer, args.max_length)

    pad_id = tokenizer.pad_token_id or 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    logger.info(
        "Train examples: %d, Val examples: %d", len(train_dataset), len(val_dataset)
    )

    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=0.01,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.base_model.eval()

        train_loss = 0.0
        train_rank_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            verify_logits = output["verify_logits"]

            base_loss = F.cross_entropy(verify_logits, labels)
            if args.ranking_loss_weight > 0:
                problem_id_tensor = torch.tensor(batch["problem_ids"], device=device)
                rank_loss = compute_ranking_loss(
                    verify_logits[:, 1],
                    labels,
                    problem_id_tensor,
                )
                total_loss = base_loss + args.ranking_loss_weight * rank_loss
                train_rank_loss += float(rank_loss.item())
            else:
                total_loss = base_loss

            loss = total_loss / args.gradient_accumulation
            loss.backward()

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += total_loss.item()
            preds = verify_logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if (step + 1) % 50 == 0:
                if args.ranking_loss_weight > 0:
                    logger.info(
                        "epoch=%d step=%d/%d loss=%.4f rank_loss=%.4f acc=%.3f",
                        epoch,
                        step + 1,
                        len(train_loader),
                        train_loss / (step + 1),
                        train_rank_loss / (step + 1),
                        train_correct / max(train_total, 1),
                    )
                else:
                    logger.info(
                        "epoch=%d step=%d/%d loss=%.4f acc=%.3f",
                        epoch,
                        step + 1,
                        len(train_loader),
                        train_loss / (step + 1),
                        train_correct / max(train_total, 1),
                    )

        if train_total % args.gradient_accumulation != 0:
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_acc = train_correct / max(train_total, 1)
        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_train_rank_loss = train_rank_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_rank_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)
                verify_logits = output["verify_logits"]

                base_loss = F.cross_entropy(verify_logits, labels)
                if args.ranking_loss_weight > 0:
                    problem_id_tensor = torch.tensor(
                        batch["problem_ids"], device=device
                    )
                    rank_loss = compute_ranking_loss(
                        verify_logits[:, 1],
                        labels,
                        problem_id_tensor,
                    )
                    val_rank_loss += float(rank_loss.item())
                    total_loss = base_loss + args.ranking_loss_weight * rank_loss
                else:
                    total_loss = base_loss

                val_loss += total_loss.item()
                preds = verify_logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_val_rank_loss = val_rank_loss / max(len(val_loader), 1)

        if args.ranking_loss_weight > 0:
            logger.info(
                "epoch=%d train_loss=%.4f train_rank_loss=%.4f train_acc=%.3f val_loss=%.4f val_rank_loss=%.4f val_acc=%.3f",
                epoch,
                avg_train_loss,
                avg_train_rank_loss,
                train_acc,
                avg_val_loss,
                avg_val_rank_loss,
                val_acc,
            )
        else:
            logger.info(
                "epoch=%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
                epoch,
                avg_train_loss,
                train_acc,
                avg_val_loss,
                val_acc,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trainable_state = {
                "slot_embedding": model.slot_embedding.data.cpu(),
                "slot_modules": {
                    k: v.state_dict() for k, v in model.slot_modules.items()
                },
                "verify_head": model.verify_head.state_dict(),
                "config": {
                    "model_name": args.model,
                    "n_slots": args.n_slots,
                    "slot_dim": args.slot_dim,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                },
            }
            save_path = output_dir / "best_slot_weights.pt"
            torch.save(trainable_state, save_path)
            logger.info("Saved best model (val_acc=%.3f) to %s", val_acc, save_path)

    logger.info("Training complete. Best val_acc=%.3f", best_val_acc)


if __name__ == "__main__":
    main()
