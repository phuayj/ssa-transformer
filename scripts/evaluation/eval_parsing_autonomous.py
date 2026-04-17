#!/usr/bin/env python3
"""Closed-loop autonomous evaluation for the backtracking parsing domain."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parsing.generator import generate_expression
from parsing.oracle_parser import ParserAction, PolicyParsingSimulator
from parsing.tokenizer import (
    ALT_BASE,
    BACKTRACK,
    VOCAB_SIZE,
    build_problem_prefix,
    compute_block_ids_for_vocab,
    serialize_state_block,
)
from universal.ssa_decoder import SSASlotDecoder


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


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _parse_str_list(raw: str) -> List[str]:
    return [str(x.strip()) for x in str(raw).split(",") if str(x).strip()]


def _extract_val_metrics(
    checkpoint: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    config = checkpoint.get("config", {})
    history = checkpoint.get("history")

    val_loss: Optional[float] = None
    val_acc: Optional[float] = None

    if checkpoint.get("val_loss") is not None:
        val_loss = float(checkpoint["val_loss"])
    elif config.get("val_loss") is not None:
        val_loss = float(config["val_loss"])

    if checkpoint.get("val_acc") is not None:
        val_acc = float(checkpoint["val_acc"])
    elif checkpoint.get("val_token_acc") is not None:
        val_acc = float(checkpoint["val_token_acc"])
    elif config.get("val_acc") is not None:
        val_acc = float(config["val_acc"])
    elif config.get("val_token_acc") is not None:
        val_acc = float(config["val_token_acc"])

    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            if val_loss is None and last.get("val_loss") is not None:
                val_loss = float(last["val_loss"])
            if val_acc is None:
                if last.get("val_acc") is not None:
                    val_acc = float(last["val_acc"])
                elif last.get("val_token_acc") is not None:
                    val_acc = float(last["val_token_acc"])
    return val_loss, val_acc


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    max_seq_len_fallback: int,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})
    vocab_size = int(
        config.get("vocab_size", state_dict["token_embedding.weight"].shape[0])
    )
    d_model = int(config.get("d_model", 256))
    n_layers = int(config.get("n_layers", 6))
    n_heads = int(config.get("n_heads", 8))
    n_slots = int(config.get("n_slots", 32))
    max_seq_len_model = int(config.get("max_seq_len", int(max_seq_len_fallback)))
    dropout = float(config.get("dropout", 0.1))
    mask_mode = str(config.get("mask_mode", "selective_ssa"))

    model = SSASlotDecoder(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        max_seq_len=int(max_seq_len_model),
        n_slots=int(n_slots),
        dropout=float(dropout),
    )
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(str(key))
        else:
            filtered[key] = value
    if skipped:
        logger.warning(
            "skipped %d checkpoint tensors due to shape mismatch", len(skipped)
        )
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()

    val_loss, val_acc = _extract_val_metrics(checkpoint)
    return model, {
        "checkpoint": str(checkpoint_path),
        "config": config,
        "mask_mode": str(mask_mode),
        "vocab_size": int(vocab_size),
        "max_seq_len_model": int(max_seq_len_model),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def _predict_next_token(
    model: torch.nn.Module,
    sequence: Sequence[int],
    allowed_tokens: Sequence[int],
    device: torch.device,
    mask_mode: str,
) -> int:
    if not allowed_tokens:
        raise ValueError("allowed_tokens must be non-empty")
    input_tensor = torch.tensor([list(sequence)], dtype=torch.long, device=device)
    block_ids = compute_block_ids_for_vocab(sequence)
    block_tensor = torch.tensor([block_ids], dtype=torch.long, device=device)
    lm_logits, verify_logits = model(
        input_tensor,
        block_ids=block_tensor,
        mask_mode=str(mask_mode),
    )
    _ = verify_logits
    next_logits = lm_logits[0, -1, :]
    mask = torch.full_like(next_logits, float("-inf"))
    for token in allowed_tokens:
        if 0 <= int(token) < int(next_logits.shape[0]):
            mask[int(token)] = 0.0
    pred = int(torch.argmax(next_logits + mask).item())
    if pred not in allowed_tokens:
        return int(allowed_tokens[0])
    return int(pred)


def _append_prompt(
    prefix_tokens: Sequence[int],
    state_prompt: Sequence[int],
    history_sequence: Sequence[int],
    history_mode: str,
) -> List[int]:
    if str(history_mode) == "state_only":
        return [int(x) for x in prefix_tokens] + [int(x) for x in state_prompt]
    return [int(x) for x in history_sequence] + [int(x) for x in state_prompt]


def _action_tokens_for_request(
    kind: str, available_alternatives: Sequence[int]
) -> List[int]:
    if str(kind) == "choice":
        return [int(ALT_BASE + int(alt)) for alt in available_alternatives]
    return [int(BACKTRACK)]


def _action_from_prediction(token_id: int) -> ParserAction:
    if int(token_id) == int(BACKTRACK):
        return ParserAction(kind="backtrack", alternative=None)
    if int(token_id) >= int(ALT_BASE):
        return ParserAction(kind="alt", alternative=int(token_id) - int(ALT_BASE))
    raise ValueError(f"prediction is not a parsing action token: {token_id}")


def _generate_instances(
    *,
    num_instances: int,
    max_input_len: int,
    max_depth: int,
    seed: int,
    p_call: float,
    p_index: float,
    p_tuple: float,
    p_neg: float,
) -> List[List[str]]:
    rng = random.Random(int(seed))
    instances: List[List[str]] = []
    while len(instances) < int(num_instances):
        tokens = generate_expression(
            max_depth=int(max_depth),
            p_call=float(p_call),
            p_index=float(p_index),
            p_tuple=float(p_tuple),
            p_neg=float(p_neg),
            rng=rng,
        )
        if len(tokens) <= int(max_input_len):
            instances.append(tokens)
    return instances


def _evaluate_single_instance(
    *,
    model: torch.nn.Module,
    tokens: Sequence[str],
    device: torch.device,
    mask_mode: str,
    budget: int,
    history_mode: str,
) -> Dict[str, Any]:
    simulator = PolicyParsingSimulator(tokens)
    prefix_tokens = build_problem_prefix(tokens)
    history_sequence: List[int] = list(prefix_tokens)
    actions: List[ParserAction] = []
    decisions = 0
    backtracks = 0
    steps = 0
    termination_reason = "budget"

    while steps < int(budget):
        state = simulator.simulate(actions)
        if state.status == "parsed":
            return {
                "parsed": True,
                "decisions": int(decisions),
                "backtracks": int(backtracks),
                "steps": int(steps),
                "termination_reason": "parsed",
                "final_action_count": int(len(actions)),
            }
        if state.status in {"failed", "invalid"}:
            return {
                "parsed": False,
                "decisions": int(decisions),
                "backtracks": int(backtracks),
                "steps": int(steps),
                "termination_reason": str(state.reason),
                "final_action_count": int(len(actions)),
            }
        if state.status != "need_action":
            return {
                "parsed": False,
                "decisions": int(decisions),
                "backtracks": int(backtracks),
                "steps": int(steps),
                "termination_reason": f"unexpected_status:{state.status}",
                "final_action_count": int(len(actions)),
            }

        state_prompt = serialize_state_block(
            tokens=tokens,
            cursor=int(state.cursor),
            stack=state.stack,
            action_token=None,
        )
        sequence = _append_prompt(
            prefix_tokens=prefix_tokens,
            state_prompt=state_prompt,
            history_sequence=history_sequence,
            history_mode=str(history_mode),
        )
        if len(sequence) >= int(budget):
            termination_reason = "budget"
            break

        allowed_tokens = _action_tokens_for_request(
            kind=str(state.reason),
            available_alternatives=state.available_alternatives,
        )
        pred = _predict_next_token(
            model=model,
            sequence=sequence,
            allowed_tokens=allowed_tokens,
            device=device,
            mask_mode=str(mask_mode),
        )
        action = _action_from_prediction(pred)
        actions.append(action)

        if str(action.kind) == "alt":
            decisions += 1
        else:
            backtracks += 1
        steps += 1

        if str(history_mode) == "state_only":
            history_sequence = list(prefix_tokens)
        history_sequence = (
            list(history_sequence) + [int(x) for x in state_prompt] + [int(pred)]
        )
        if len(history_sequence) >= int(budget):
            termination_reason = "budget"
            break

    return {
        "parsed": False,
        "decisions": int(decisions),
        "backtracks": int(backtracks),
        "steps": int(steps),
        "termination_reason": str(termination_reason),
        "final_action_count": int(len(actions)),
    }


def _evaluate_model(
    *,
    model: torch.nn.Module,
    mask_mode: str,
    history_mode: str,
    instances: Sequence[Sequence[str]],
    budget: int,
    device: torch.device,
) -> Dict[str, Any]:
    per_instance: List[Dict[str, Any]] = []
    for idx, tokens in enumerate(instances):
        stats = _evaluate_single_instance(
            model=model,
            tokens=tokens,
            device=device,
            mask_mode=str(mask_mode),
            budget=int(budget),
            history_mode=str(history_mode),
        )
        stats["expression"] = " ".join(str(tok) for tok in tokens)
        stats["input_len"] = int(len(tokens))
        per_instance.append(stats)

        if (idx + 1) % 25 == 0:
            successes = int(sum(int(item["parsed"]) for item in per_instance))
            timeout_count = int(
                sum(
                    1
                    for item in per_instance
                    if str(item["termination_reason"]) == "budget"
                )
            )
            logger.info(
                "processed=%d/%d parse_success=%.3f mean_decisions=%.2f mean_backtracks=%.2f timeout_rate=%.3f",
                int(idx + 1),
                int(len(instances)),
                float(_safe_div(successes, len(per_instance))),
                float(np.mean([item["decisions"] for item in per_instance]))
                if per_instance
                else 0.0,
                float(np.mean([item["backtracks"] for item in per_instance]))
                if per_instance
                else 0.0,
                float(_safe_div(timeout_count, len(per_instance))),
            )

    total = int(len(per_instance))
    parsed = int(sum(int(item["parsed"]) for item in per_instance))
    timeout_count = int(
        sum(1 for item in per_instance if str(item["termination_reason"]) == "budget")
    )
    return {
        "parse_success_rate": float(_safe_div(parsed, total)),
        "mean_decisions": float(np.mean([item["decisions"] for item in per_instance]))
        if per_instance
        else 0.0,
        "mean_backtracks": float(np.mean([item["backtracks"] for item in per_instance]))
        if per_instance
        else 0.0,
        "timeout_rate": float(_safe_div(timeout_count, total)),
        "per_instance": per_instance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate parsing checkpoints autonomously"
    )
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--max-input-len", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--history-mode",
        type=str,
        choices=("cumulative", "state_only"),
        default="cumulative",
    )
    parser.add_argument("--p-call", type=float, default=0.3)
    parser.add_argument("--p-index", type=float, default=0.2)
    parser.add_argument("--p-tuple", type=float, default=0.35)
    parser.add_argument("--p-neg", type=float, default=0.1)
    args = parser.parse_args()

    if int(args.num_instances) <= 0:
        raise ValueError("num-instances must be > 0")
    if int(args.max_input_len) <= 0:
        raise ValueError("max-input-len must be > 0")
    if int(args.max_depth) < 0:
        raise ValueError("max-depth must be >= 0")
    if int(args.budget) <= 0:
        raise ValueError("budget must be > 0")

    checkpoint_paths = _parse_str_list(args.checkpoints)
    if not checkpoint_paths:
        raise ValueError("--checkpoints must provide at least one path")
    labels = _parse_str_list(args.labels)
    if labels and len(labels) != len(checkpoint_paths):
        raise ValueError("--labels must match the number of checkpoints")

    _set_seed(int(args.seed))
    device = torch.device(str(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = _generate_instances(
        num_instances=int(args.num_instances),
        max_input_len=int(args.max_input_len),
        max_depth=int(args.max_depth),
        seed=int(args.seed),
        p_call=float(args.p_call),
        p_index=float(args.p_index),
        p_tuple=float(args.p_tuple),
        p_neg=float(args.p_neg),
    )
    logger.info(
        "generated_eval_instances count=%d mean_input_len=%.2f sample_expr=%s",
        int(len(instances)),
        float(np.mean([len(x) for x in instances])) if instances else 0.0,
        " ".join(instances[0]) if instances else "",
    )

    results: List[Dict[str, Any]] = []
    started = time.time()
    for idx, checkpoint in enumerate(checkpoint_paths):
        checkpoint_path = Path(checkpoint)
        model, meta = _load_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            max_seq_len_fallback=int(args.budget),
        )
        if int(meta["vocab_size"]) < int(VOCAB_SIZE):
            logger.warning(
                "checkpoint vocab_size=%d < parsing VOCAB_SIZE=%d; tokens >= %d will be out of range",
                int(meta["vocab_size"]),
                int(VOCAB_SIZE),
                int(meta["vocab_size"]),
            )
        run_label = labels[idx] if labels else checkpoint_path.stem
        summary = _evaluate_model(
            model=model,
            mask_mode=str(meta["mask_mode"]),
            history_mode=str(args.history_mode),
            instances=instances,
            budget=int(args.budget),
            device=device,
        )
        result_row = {
            "label": str(run_label),
            "checkpoint": str(checkpoint_path),
            "mask_mode": str(meta["mask_mode"]),
            "val_loss": meta["val_loss"],
            "val_acc": meta["val_acc"],
            "parse_success_rate": float(summary["parse_success_rate"]),
            "mean_decisions": float(summary["mean_decisions"]),
            "mean_backtracks": float(summary["mean_backtracks"]),
            "timeout_rate": float(summary["timeout_rate"]),
            "per_instance": summary["per_instance"],
        }
        results.append(result_row)
        logger.info(
            "checkpoint=%s label=%s parse_success=%.3f mean_decisions=%.2f mean_backtracks=%.2f timeout_rate=%.3f",
            str(checkpoint_path),
            str(run_label),
            float(result_row["parse_success_rate"]),
            float(result_row["mean_decisions"]),
            float(result_row["mean_backtracks"]),
            float(result_row["timeout_rate"]),
        )

    payload = {
        "task": "parsing_autonomous_eval",
        "num_instances": int(args.num_instances),
        "max_input_len": int(args.max_input_len),
        "max_depth": int(args.max_depth),
        "budget": int(args.budget),
        "history_mode": str(args.history_mode),
        "seed": int(args.seed),
        "elapsed_sec": float(time.time() - started),
        "results": results,
    }
    output_path = output_dir / "parsing_autonomous_eval.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("saved evaluation results path=%s", str(output_path))


if __name__ == "__main__":
    main()
