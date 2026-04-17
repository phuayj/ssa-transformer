#!/usr/bin/env python3
"""Train SAT SSA models with contrastive history-invariance regularization."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.length_ood_common import compute_length_stats
from sat.interleaved_tokenizer import SATInterleavedTokenizer
from universal.ssa_decoder import SSASlotDecoder

from train_history_ablation import (
    BlockTruncatedDataset,
    POSITION_MODES,
    SSATriedDataset,
    _collate,
    _compute_lm_loss,
    _get_state_tried_tokens,
    _infer_model_init_config_from_checkpoint,
    _infer_vocab_size_from_records,
    _load_records,
    _load_with_vocab_expansion,
    _resolve_dataset_metadata,
    _resolve_effective_position_mode,
    _resolve_val_batch_size,
    _run_epoch as _run_lm_epoch,
    _safe_decode,
    _set_seed,
    build_history_transplant_example,
    prepare_history_transplant_donor_pool,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _is_action_token(tokens: torch.Tensor) -> torch.Tensor:
    return (
        (
            tokens >= int(SATInterleavedTokenizer.VAR_OFFSET)
        )
        & (tokens < int(SATInterleavedTokenizer.VOCAB_SIZE))
    ) | tokens.eq(int(SATInterleavedTokenizer.TRUE_VAL)) | tokens.eq(
        int(SATInterleavedTokenizer.FALSE_VAL)
    ) | tokens.eq(int(SATInterleavedTokenizer.BACKJUMP))


def _extract_current_action_positions(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    block_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids[:, 1:].clone()
    valid = attention_mask[:, 1:] > 0
    shifted_block_ids = block_ids[:, 1:]
    supervised_search = loss_mask & block_ids.gt(0)
    if not bool(supervised_search.any().item()):
        return torch.zeros_like(labels, dtype=torch.bool), labels
    current_block_ids = block_ids.masked_fill(~supervised_search, -1).max(dim=1).values
    action_mask = _is_action_token(labels)
    current_block_mask = shifted_block_ids.eq(current_block_ids[:, None])
    supervised = loss_mask[:, 1:] & valid & current_block_mask & action_mask
    return supervised, labels


def _zero_contrastive_loss_metrics(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    empty_targets = torch.empty(0, device=device, dtype=torch.long)
    loss = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    metrics = {
        "num_action_tokens": 0,
        "token_count": 0,
        "source_correct": 0,
        "pair_correct": 0,
        "ce_source": 0.0,
        "ce_pair": 0.0,
        "kl": 0.0,
        "loss_ce_source": 0.0,
        "loss_ce_pair": 0.0,
        "loss_kl": 0.0,
        "source_targets": empty_targets,
        "pair_targets": empty_targets,
    }
    return loss, metrics


def compute_symmetric_action_kl(
    source_action_logits: torch.Tensor,
    pair_action_logits: torch.Tensor,
) -> torch.Tensor:
    source_log_probs = F.log_softmax(source_action_logits, dim=-1)
    pair_log_probs = F.log_softmax(pair_action_logits, dim=-1)
    source_probs = source_log_probs.exp()
    pair_probs = pair_log_probs.exp()
    return 0.5 * (
        F.kl_div(source_log_probs, pair_probs, reduction="batchmean")
        + F.kl_div(pair_log_probs, source_probs, reduction="batchmean")
    )


def compute_contrastive_invariance_loss(
    *,
    source_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    source_input_ids: torch.Tensor,
    pair_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    pair_attention_mask: torch.Tensor,
    source_loss_mask: torch.Tensor,
    pair_loss_mask: torch.Tensor,
    source_block_ids: torch.Tensor,
    pair_block_ids: torch.Tensor,
    lambda_kl: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    source_supervised, source_labels = _extract_current_action_positions(
        input_ids=source_input_ids,
        attention_mask=source_attention_mask,
        loss_mask=source_loss_mask,
        block_ids=source_block_ids,
    )
    pair_supervised, pair_labels = _extract_current_action_positions(
        input_ids=pair_input_ids,
        attention_mask=pair_attention_mask,
        loss_mask=pair_loss_mask,
        block_ids=pair_block_ids,
    )

    source_action_logits = source_logits[:, :-1, :][source_supervised]
    pair_action_logits = pair_logits[:, :-1, :][pair_supervised]
    source_targets = source_labels[source_supervised]
    pair_targets = pair_labels[pair_supervised]

    source_token_count = int(source_targets.numel())
    pair_token_count = int(pair_targets.numel())
    if source_token_count == 0 or pair_token_count == 0:
        return _zero_contrastive_loss_metrics(
            device=source_logits.device,
            dtype=source_logits.dtype,
        )
    if source_token_count != pair_token_count:
        raise RuntimeError(
            "source/pair action token mismatch: "
            f"{source_token_count} vs {pair_token_count}"
        )
    if not torch.equal(source_targets, pair_targets):
        raise RuntimeError("source/pair action targets diverged; current block must match")

    loss_ce_source = F.cross_entropy(source_action_logits, source_targets)
    loss_ce_pair = F.cross_entropy(pair_action_logits, pair_targets)
    loss_kl = compute_symmetric_action_kl(source_action_logits, pair_action_logits)
    total_loss = loss_ce_source + loss_ce_pair + float(lambda_kl) * loss_kl

    source_preds = source_action_logits.argmax(dim=-1)
    pair_preds = pair_action_logits.argmax(dim=-1)
    metrics = {
        "num_action_tokens": int(source_token_count),
        "token_count": int(source_token_count),
        "source_correct": int((source_preds == source_targets).sum().item()),
        "pair_correct": int((pair_preds == pair_targets).sum().item()),
        "ce_source": float(loss_ce_source.detach().item()),
        "ce_pair": float(loss_ce_pair.detach().item()),
        "kl": float(loss_kl.detach().item()),
        "loss_ce_source": float(loss_ce_source.detach().item()),
        "loss_ce_pair": float(loss_ce_pair.detach().item()),
        "loss_kl": float(loss_kl.detach().item()),
        "source_targets": source_targets.detach(),
        "pair_targets": pair_targets.detach(),
    }
    return total_loss, metrics


class ContrastiveHistoryPairDataset(
    Dataset[
        Tuple[
            List[int],
            List[bool],
            List[int],
            List[int],
            List[bool],
            List[int],
        ]
    ]
):
    """Return (source, transplanted-pair) examples for contrastive training."""

    def __init__(
        self,
        base_dataset: Dataset[Tuple[List[int], List[bool], List[int]]],
        *,
        donor_pool: Sequence[Dict[str, Any]],
        seed: int,
    ):
        self.base_dataset = base_dataset
        self.max_seq_len, self.vocab_size = _resolve_dataset_metadata(base_dataset)
        self.donor_pool = prepare_history_transplant_donor_pool(
            donor_pool,
            max_seq_len=int(self.max_seq_len),
            vocab_size=int(self.vocab_size),
        )
        if len(self.donor_pool) == 0:
            raise ValueError("contrastive training requires a non-empty donor_pool")
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        List[int],
        List[bool],
        List[int],
        List[int],
        List[bool],
        List[int],
    ]:
        source_seq, source_lm, source_blk = self.base_dataset[int(idx)]
        pair_seq, pair_lm, pair_blk = build_history_transplant_example(
            source_seq,
            source_lm,
            source_blk,
            donor_pool=self.donor_pool,
            rng=self.rng,
            transplant_prob=1.0,
            partial_transplant=False,
            max_seq_len=int(self.max_seq_len),
        )
        return source_seq, source_lm, source_blk, pair_seq, pair_lm, pair_blk


def _collate_pairs(
    batch: Sequence[
        Tuple[
            List[int],
            List[bool],
            List[int],
            List[int],
            List[bool],
            List[int],
        ]
    ]
):
    source_batch = [(item[0], item[1], item[2]) for item in batch]
    pair_batch = [(item[3], item[4], item[5]) for item in batch]
    return (*_collate(source_batch), *_collate(pair_batch))


def _run_contrastive_epoch(
    *,
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    train: bool,
    device: torch.device,
    tokenizer: SATInterleavedTokenizer,
    mask_mode: str,
    position_mode: str,
    lambda_kl: float,
    grad_accum_steps: int,
) -> Dict[str, float]:
    if int(grad_accum_steps) <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if train:
        model.train()
        if optimizer is None:
            raise ValueError("optimizer is required for training")
        opt = optimizer
        opt.zero_grad(set_to_none=True)
    else:
        model.eval()
        opt = None

    total_loss = 0.0
    total_ce_source = 0.0
    total_ce_pair = 0.0
    total_kl = 0.0
    total_tokens = 0
    total_source_correct = 0
    total_pair_correct = 0
    total_seq = 0
    skipped_batches = 0
    optimizer_steps = 0
    logged = False
    window_has_grad = False
    total_batches = int(len(loader))

    for batch_idx, batch in enumerate(loader):
        (
            source_input_ids,
            source_attention_mask,
            source_loss_mask,
            source_block_ids,
            pair_input_ids,
            pair_attention_mask,
            pair_loss_mask,
            pair_block_ids,
        ) = batch
        source_input_ids = source_input_ids.to(device)
        source_attention_mask = source_attention_mask.to(device)
        source_loss_mask = source_loss_mask.to(device)
        source_block_ids = source_block_ids.to(device)
        pair_input_ids = pair_input_ids.to(device)
        pair_attention_mask = pair_attention_mask.to(device)
        pair_loss_mask = pair_loss_mask.to(device)
        pair_block_ids = pair_block_ids.to(device)

        with torch.set_grad_enabled(train):
            source_logits, _verify_logits = model(
                source_input_ids,
                source_attention_mask,
                block_ids=source_block_ids,
                mask_mode=mask_mode,
                position_mode=position_mode,
            )
            pair_logits, _verify_logits = model(
                pair_input_ids,
                pair_attention_mask,
                block_ids=pair_block_ids,
                mask_mode=mask_mode,
                position_mode=position_mode,
            )
            loss, loss_metrics = compute_contrastive_invariance_loss(
                source_logits=source_logits,
                pair_logits=pair_logits,
                source_input_ids=source_input_ids,
                pair_input_ids=pair_input_ids,
                source_attention_mask=source_attention_mask,
                pair_attention_mask=pair_attention_mask,
                source_loss_mask=source_loss_mask,
                pair_loss_mask=pair_loss_mask,
                source_block_ids=source_block_ids,
                pair_block_ids=pair_block_ids,
                lambda_kl=float(lambda_kl),
            )
            has_action_tokens = int(loss_metrics["num_action_tokens"]) > 0
            if not has_action_tokens:
                skipped_batches += 1
            if train:
                assert opt is not None
                if has_action_tokens:
                    (loss / float(grad_accum_steps)).backward()
                    window_has_grad = True
                should_step = ((batch_idx + 1) % int(grad_accum_steps) == 0) or (
                    batch_idx + 1 == total_batches
                )
                if should_step:
                    if window_has_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    optimizer_steps += 1
                    if scheduler is not None:
                        scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    window_has_grad = False

        batch_size = int(source_input_ids.size(0))
        total_seq += batch_size
        total_loss += float(loss.detach().item()) * float(batch_size)
        total_ce_source += float(loss_metrics["ce_source"]) * float(batch_size)
        total_ce_pair += float(loss_metrics["ce_pair"]) * float(batch_size)
        total_kl += float(loss_metrics["kl"]) * float(batch_size)
        total_tokens += int(loss_metrics["num_action_tokens"])
        total_source_correct += int(loss_metrics["source_correct"])
        total_pair_correct += int(loss_metrics["pair_correct"])

        if not logged and int(loss_metrics["num_action_tokens"]) > 0:
            target_tok = int(loss_metrics["source_targets"][0].item())
            logger.info(
                "sample_contrastive_token train=%s mask_mode=%s lambda_kl=%.3f target=%s source_seq=%d pair_seq=%d current_action_tokens=%d",
                str(train),
                str(mask_mode),
                float(lambda_kl),
                _safe_decode(tokenizer, target_tok),
                int(source_attention_mask[0].sum().item()),
                int(pair_attention_mask[0].sum().item()),
                int(loss_metrics["num_action_tokens"]),
            )
            logged = True

    skipped_fraction = float(skipped_batches / max(total_batches, 1))
    if skipped_batches > 0:
        log_fn = logger.error if skipped_fraction > 0.30 else logger.warning
        log_fn(
            "contrastive_epoch_skips train=%s skipped_batches=%d total_batches=%d skipped_fraction=%.3f",
            str(train),
            int(skipped_batches),
            int(total_batches),
            float(skipped_fraction),
        )

    return {
        "loss": float(total_loss / max(float(total_seq), 1.0)),
        "source_ce": float(total_ce_source / max(float(total_seq), 1.0)),
        "pair_ce": float(total_ce_pair / max(float(total_seq), 1.0)),
        "kl": float(total_kl / max(float(total_seq), 1.0)),
        "source_token_acc": float(total_source_correct / max(float(total_tokens), 1.0)),
        "pair_token_acc": float(total_pair_correct / max(float(total_tokens), 1.0)),
        "tokens": float(total_tokens),
        "sequences": float(total_seq),
        "skipped_batches": float(skipped_batches),
        "skipped_fraction": float(skipped_fraction),
        "optimizer_steps": float(optimizer_steps),
    }


def _maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SAT SSA decoder with contrastive invariance regularization"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="full_causal",
        choices=["full_causal", "selective_ssa"],
    )
    parser.add_argument(
        "--position_mode",
        type=str,
        default="auto",
        choices=list(POSITION_MODES),
        help=(
            "Forward-pass positional scheme. 'auto' preserves current behavior: "
            "use block-relative positions whenever block_ids are supplied."
        ),
    )
    parser.add_argument("--lambda_kl", type=float, default=1.0)
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "lstm"],
    )
    parser.add_argument("--max_seq_len", type=int, default=1500)
    parser.add_argument(
        "--max_train_blocks",
        type=int,
        default=0,
        help="If >0, truncate each training trace after the first K search blocks while keeping max_seq_len unchanged.",
    )
    parser.add_argument(
        "--match_token_budget",
        action="store_true",
        help="If set with --max_train_blocks>0, scale epochs to approximately match the full-training supervised token budget.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Accumulate gradients across this many micro-batches before each optimizer step.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help=(
            "Validation batch size. If omitted, defaults to --batch_size, except "
            "for --max_seq_len > 4096 where it auto-uses max(1, --batch_size // 4) "
            "to reduce validation OOM risk."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
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
    parser.add_argument(
        "--use_sdpa_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.nn.functional.scaled_dot_product_attention in the slot decoder when available.",
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_lstm_layers", type=int, default=4)
    parser.add_argument(
        "--block_mode",
        type=str,
        default="continuous",
        choices=["continuous", "block_reset"],
    )
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one paired train batch and one validation batch, then write smoke.json without training.",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    _set_seed(int(args.seed))

    if float(args.lambda_kl) < 0.0:
        raise ValueError("--lambda_kl must be non-negative")
    if int(args.max_train_blocks) < 0:
        raise ValueError("--max_train_blocks must be >= 0")
    if args.val_batch_size is not None and int(args.val_batch_size) <= 0:
        raise ValueError("--val_batch_size must be positive")
    if int(args.grad_accum_steps) <= 0:
        raise ValueError("--grad_accum_steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SATInterleavedTokenizer()
    records = _load_records(Path(args.data_path))
    base_vocab_size = _infer_vocab_size_from_records(records)
    _get_state_tried_tokens(base_vocab_size)
    effective_vocab_size = int(base_vocab_size)
    device = torch.device(str(args.device))
    resolved_position_mode = _resolve_effective_position_mode(
        position_mode=str(args.position_mode),
        has_block_ids=True,
    )

    model_cfg: Dict[str, Any] = {
        "model_type": str(args.model_type),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "max_seq_len": int(args.max_seq_len),
        "n_slots": int(args.n_slots),
        "dropout": float(args.dropout),
        "use_sdpa_attention": bool(args.use_sdpa_attention),
        "hidden_size": int(args.hidden_size),
        "n_lstm_layers": int(args.n_lstm_layers),
        "block_mode": str(args.block_mode),
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

    val_batch_size = _resolve_val_batch_size(
        args=args,
        max_seq_len=int(model_cfg["max_seq_len"]),
    )

    random.Random(int(args.seed)).shuffle(records)
    split = int(round((1.0 - float(args.val_split)) * len(records)))
    split = max(1, min(split, len(records) - 1))
    train_records = records[:split]
    val_records = records[split:]
    donor_pool = [dict(record) for record in train_records]
    random.Random(int(args.seed) + 17).shuffle(donor_pool)
    logger.info("contrastive_donor_pool size=%d", int(len(donor_pool)))

    all_length_stats = compute_length_stats(
        records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        max_train_blocks=int(args.max_train_blocks),
    )
    train_length_stats = compute_length_stats(
        train_records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        max_train_blocks=int(args.max_train_blocks),
    )
    val_length_stats = compute_length_stats(
        val_records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        max_train_blocks=0,
    )
    mean_full_supervised_tokens = float(
        train_length_stats["supervised_tokens_per_trace"]["mean"]
    )
    mean_truncated_supervised_tokens = float(mean_full_supervised_tokens)
    token_budget_multiplier = 1.0
    if int(args.max_train_blocks) > 0:
        trunc_stats = train_length_stats.get("truncation", {})
        mean_truncated_supervised_tokens = float(
            trunc_stats.get("supervised_tokens_per_trace", {}).get("mean", 0.0)
        )
        if mean_truncated_supervised_tokens <= 0.0:
            raise RuntimeError(
                "--max_train_blocks removed all supervised tokens; choose a larger K"
            )
        token_budget_multiplier = float(
            mean_full_supervised_tokens / max(mean_truncated_supervised_tokens, 1e-8)
        )

    effective_epochs = int(args.epochs)
    if bool(args.match_token_budget):
        if int(args.max_train_blocks) <= 0:
            logger.warning(
                "--match_token_budget requested without --max_train_blocks>0; keeping epochs=%d",
                int(args.epochs),
            )
        else:
            effective_epochs = max(
                1,
                int(round(float(args.epochs) * float(token_budget_multiplier))),
            )

    with (output_dir / "length_stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(args.data_path),
                "max_seq_len": int(model_cfg["max_seq_len"]),
                "max_train_blocks": int(args.max_train_blocks),
                "all_records": all_length_stats,
                "train_records": train_length_stats,
                "val_records": val_length_stats,
                "token_budget_matching": {
                    "match_token_budget": bool(args.match_token_budget),
                    "base_epochs": int(args.epochs),
                    "effective_epochs": int(effective_epochs),
                    "mean_full_supervised_tokens_per_trace": float(mean_full_supervised_tokens),
                    "mean_truncated_supervised_tokens_per_trace": float(
                        mean_truncated_supervised_tokens
                    ),
                    "supervised_token_multiplier": float(token_budget_multiplier),
                },
            },
            f,
            indent=2,
        )

    base_train_ds = SSATriedDataset(
        train_records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        vocab_size=int(base_vocab_size),
    )
    train_core_ds: Dataset[Tuple[List[int], List[bool], List[int]]] = base_train_ds
    if int(args.max_train_blocks) > 0:
        train_core_ds = BlockTruncatedDataset(
            base_train_ds,
            max_train_blocks=int(args.max_train_blocks),
        )
    train_ds = ContrastiveHistoryPairDataset(
        train_core_ds,
        donor_pool=donor_pool,
        seed=int(args.seed),
    )
    val_ds = SSATriedDataset(
        val_records,
        max_seq_len=int(model_cfg["max_seq_len"]),
        vocab_size=int(base_vocab_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=_collate_pairs,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(val_batch_size),
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    model: torch.nn.Module
    if str(model_cfg["model_type"]) == "lstm":
        from universal.lstm_decoder import LSTMDecoder

        model = LSTMDecoder(
            vocab_size=int(effective_vocab_size),
            d_model=int(model_cfg["d_model"]),
            hidden_size=int(model_cfg["hidden_size"]),
            n_lstm_layers=int(model_cfg["n_lstm_layers"]),
            max_seq_len=int(model_cfg["max_seq_len"]),
            n_slots=int(model_cfg["n_slots"]),
            dropout=float(model_cfg["dropout"]),
            block_mode=str(model_cfg["block_mode"]),
        )
    else:
        model = SSASlotDecoder(
            vocab_size=int(effective_vocab_size),
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            n_heads=int(model_cfg["n_heads"]),
            max_seq_len=int(model_cfg["max_seq_len"]),
            n_slots=int(model_cfg["n_slots"]),
            dropout=float(model_cfg["dropout"]),
            use_sdpa_attention=bool(model_cfg["use_sdpa_attention"]),
        )

    init_meta: Dict[str, Any] = {"used": False}
    if str(args.init_checkpoint).strip():
        init_meta = _load_with_vocab_expansion(
            model=model,
            checkpoint_path=Path(args.init_checkpoint),
            target_vocab_size=int(effective_vocab_size),
        )
        init_meta["used"] = True
        logger.info("initialized_from=%s meta=%s", str(args.init_checkpoint), init_meta)

    model = model.to(device)

    config = {
        "data_path": str(args.data_path),
        "output_dir": str(output_dir),
        "mask_mode": str(args.mask_mode),
        "training_position_mode": str(args.position_mode),
        "resolved_position_mode": str(resolved_position_mode),
        "lambda_kl": float(args.lambda_kl),
        "max_train_blocks": int(args.max_train_blocks),
        "match_token_budget": bool(args.match_token_budget),
        "effective_epochs": int(effective_epochs),
        "mean_full_supervised_tokens_per_trace": float(mean_full_supervised_tokens),
        "mean_truncated_supervised_tokens_per_trace": float(
            mean_truncated_supervised_tokens
        ),
        "supervised_token_multiplier": float(token_budget_multiplier),
        "base_vocab_size": int(base_vocab_size),
        "vocab_size": int(effective_vocab_size),
        "train_contrastive_invariance": True,
        **model_cfg,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "val_batch_size": int(val_batch_size),
        "requested_val_batch_size": (
            int(args.val_batch_size) if args.val_batch_size is not None else None
        ),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "val_split": float(args.val_split),
        "device": str(args.device),
        "seed": int(args.seed),
        "init_checkpoint": str(args.init_checkpoint),
        "donor_pool_size": int(len(donor_pool)),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_steps_per_epoch = int(
        math.ceil(float(len(train_loader)) / float(max(int(args.grad_accum_steps), 1)))
    )
    planned_total_steps = int(train_steps_per_epoch * int(effective_epochs))

    warmup_steps = int(planned_total_steps * float(args.warmup_ratio))

    if bool(args.smoke):
        train_batch = next(iter(train_loader))
        (
            source_input_ids,
            source_attention_mask,
            source_loss_mask,
            source_block_ids,
            pair_input_ids,
            pair_attention_mask,
            pair_loss_mask,
            pair_block_ids,
        ) = [tensor.to(device) for tensor in train_batch]
        val_input_ids, val_attention_mask, val_loss_mask, val_block_ids = next(
            iter(val_loader)
        )
        val_input_ids = val_input_ids.to(device)
        val_attention_mask = val_attention_mask.to(device)
        val_loss_mask = val_loss_mask.to(device)
        val_block_ids = val_block_ids.to(device)

        with torch.no_grad():
            source_logits, _verify_logits = model(
                source_input_ids,
                source_attention_mask,
                block_ids=source_block_ids,
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
            )
            pair_logits, _verify_logits = model(
                pair_input_ids,
                pair_attention_mask,
                block_ids=pair_block_ids,
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
            )
            train_loss, train_metrics = compute_contrastive_invariance_loss(
                source_logits=source_logits,
                pair_logits=pair_logits,
                source_input_ids=source_input_ids,
                pair_input_ids=pair_input_ids,
                source_attention_mask=source_attention_mask,
                pair_attention_mask=pair_attention_mask,
                source_loss_mask=source_loss_mask,
                pair_loss_mask=pair_loss_mask,
                source_block_ids=source_block_ids,
                pair_block_ids=pair_block_ids,
                lambda_kl=float(args.lambda_kl),
            )
            val_logits, _verify_logits = model(
                val_input_ids,
                val_attention_mask,
                block_ids=val_block_ids,
                mask_mode=str(args.mask_mode),
                position_mode=str(args.position_mode),
            )
            val_loss, val_token_count, val_supervised, val_labels = _compute_lm_loss(
                logits=val_logits,
                input_ids=val_input_ids,
                attention_mask=val_attention_mask,
                loss_mask=val_loss_mask,
            )
            val_preds = val_logits[:, :-1, :].argmax(dim=-1)
            val_correct = int(((val_preds == val_labels) & val_supervised).sum().item())

        smoke_payload = {
            "smoke": True,
            "mask_mode": str(args.mask_mode),
            "training_position_mode": str(args.position_mode),
            "resolved_position_mode": str(resolved_position_mode),
            "lambda_kl": float(args.lambda_kl),
            "batch_size": int(args.batch_size),
            "source_batch_shape": [int(x) for x in source_input_ids.shape],
            "pair_batch_shape": [int(x) for x in pair_input_ids.shape],
            "train_action_token_count": int(train_metrics["num_action_tokens"]),
            "train_source_token_acc": float(
                train_metrics["source_correct"]
                / max(int(train_metrics["num_action_tokens"]), 1)
            ),
            "train_pair_token_acc": float(
                train_metrics["pair_correct"]
                / max(int(train_metrics["num_action_tokens"]), 1)
            ),
            "train_loss": float(train_loss.item()),
            "train_source_ce": float(train_metrics["ce_source"]),
            "train_pair_ce": float(train_metrics["ce_pair"]),
            "train_kl": float(train_metrics["kl"]),
            "val_batch_shape": [int(x) for x in val_input_ids.shape],
            "val_token_count": int(val_token_count),
            "val_token_acc": float(val_correct / max(int(val_token_count), 1)),
            "val_loss": float(val_loss.item()),
            "planned_total_steps": int(planned_total_steps),
        }
        with (output_dir / "smoke.json").open("w", encoding="utf-8") as f:
            json.dump(smoke_payload, f, indent=2)
        logger.info(
            "smoke_complete output=%s source_batch_shape=%s pair_batch_shape=%s action_tokens=%d train_loss=%.4f train_kl=%.4f val_loss=%.4f",
            str(output_dir / "smoke.json"),
            str(tuple(int(x) for x in source_input_ids.shape)),
            str(tuple(int(x) for x in pair_input_ids.shape)),
            int(train_metrics["num_action_tokens"]),
            float(train_loss.item()),
            float(train_metrics["kl"]),
            float(val_loss.item()),
        )
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
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
                    / max(float(planned_total_steps - warmup_steps), 1.0)
                )
            )
        ),
    )

    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    logger.info(
        "start_training mask_mode=%s lambda_kl=%.3f requested_position_mode=%s resolved_position_mode=%s train=%d val=%d batch=%d grad_accum=%d use_sdpa=%s epochs=%d effective_epochs=%d donor_pool=%d planned_steps=%d",
        str(args.mask_mode),
        float(args.lambda_kl),
        str(args.position_mode),
        str(resolved_position_mode),
        int(len(train_ds)),
        int(len(val_ds)),
        int(args.batch_size),
        int(args.grad_accum_steps),
        str(bool(args.use_sdpa_attention)),
        int(args.epochs),
        int(effective_epochs),
        int(len(donor_pool)),
        int(planned_total_steps),
    )

    for epoch in range(int(effective_epochs)):
        train_stats = _run_contrastive_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            device=device,
            tokenizer=tokenizer,
            mask_mode=str(args.mask_mode),
            position_mode=str(args.position_mode),
            lambda_kl=float(args.lambda_kl),
            grad_accum_steps=int(args.grad_accum_steps),
        )
        _maybe_empty_cuda_cache()
        val_stats = _run_lm_epoch(
            loader=val_loader,
            model=model,
            optimizer=None,
            scheduler=None,
            train=False,
            device=device,
            tokenizer=tokenizer,
            mask_mode=str(args.mask_mode),
            position_mode=str(args.position_mode),
            history_mode="full",
            placeholder_token=None,
        )
        _maybe_empty_cuda_cache()
        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": float(epoch + 1),
            "effective_epochs": float(effective_epochs),
            "train_loss": float(train_stats["loss"]),
            "train_source_ce": float(train_stats["source_ce"]),
            "train_pair_ce": float(train_stats["pair_ce"]),
            "train_kl": float(train_stats["kl"]),
            "train_source_token_acc": float(train_stats["source_token_acc"]),
            "train_pair_token_acc": float(train_stats["pair_token_acc"]),
            "train_action_tokens": float(train_stats["tokens"]),
            "train_skipped_batches": float(train_stats["skipped_batches"]),
            "train_skipped_fraction": float(train_stats["skipped_fraction"]),
            "optimizer_steps": float(train_stats["optimizer_steps"]),
            "val_loss": float(val_stats["loss"]),
            "val_token_acc": float(val_stats["token_acc"]),
            "lr": float(lr_now),
            "mask_mode": str(args.mask_mode),
            "training_position_mode": str(args.position_mode),
            "resolved_position_mode": str(resolved_position_mode),
            "lambda_kl": float(args.lambda_kl),
        }
        history.append(row)

        logger.info(
            "epoch=%d/%d mask_mode=%s lambda_kl=%.3f train_loss=%.4f source_ce=%.4f pair_ce=%.4f kl=%.4f train_skipped=%.0f (%.3f) val_loss=%.4f val_acc=%.4f lr=%.2e",
            int(epoch + 1),
            int(args.epochs),
            str(args.mask_mode),
            float(args.lambda_kl),
            float(train_stats["loss"]),
            float(train_stats["source_ce"]),
            float(train_stats["pair_ce"]),
            float(train_stats["kl"]),
            float(train_stats["skipped_batches"]),
            float(train_stats["skipped_fraction"]),
            float(val_stats["loss"]),
            float(val_stats["token_acc"]),
            float(lr_now),
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": {
                **config,
                "attention_mode": "ssa",
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
        "mask_mode": str(args.mask_mode),
        "training_position_mode": str(args.position_mode),
        "resolved_position_mode": str(resolved_position_mode),
        "lambda_kl": float(args.lambda_kl),
        "max_train_blocks": int(args.max_train_blocks),
        "match_token_budget": bool(args.match_token_budget),
        "effective_epochs": int(effective_epochs),
        "planned_total_steps": int(planned_total_steps),
        "mean_full_supervised_tokens_per_trace": float(mean_full_supervised_tokens),
        "mean_truncated_supervised_tokens_per_trace": float(
            mean_truncated_supervised_tokens
        ),
        "supervised_token_multiplier": float(token_budget_multiplier),
        "base_vocab_size": int(base_vocab_size),
        "vocab_size": int(effective_vocab_size),
        "train_examples": int(len(train_ds)),
        "val_examples": int(len(val_ds)),
        "epochs": int(args.epochs),
        "best_val_loss": float(best_val),
        "config": config,
        "history": history,
        "init_meta": init_meta,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "training_complete mask_mode=%s lambda_kl=%.3f best_val_loss=%.4f output_dir=%s",
        str(args.mask_mode),
        float(args.lambda_kl),
        best_val,
        output_dir,
    )


if __name__ == "__main__":
    main()
