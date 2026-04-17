#!/usr/bin/env python3
"""Evaluate GC closed-loop performance with explicit/standard state formats."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graph_coloring.dsl import GraphColorAction
from graph_coloring.env import GraphColorEnv, GraphColorEnvStatus, GraphColorState
from graph_coloring.generator import GraphGenerator
from universal.cdcl_tokenizer import CDCLTokenizer
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PrefixKey = Tuple[Tuple[int, int], ...]


class TokenMapper:
    """Maps high-range special tokens from vocabulary size."""

    def __init__(self, vocab_size: int):
        if int(vocab_size) == 394:
            max_nodes = 30
        elif int(vocab_size) == 574:
            max_nodes = 75
        else:
            raise ValueError(f"Unsupported vocab_size={vocab_size}")
        max_colors = 4
        self.ASSIGN_OFFSET = int(240)
        self.MASK_OFFSET = int(240 + max_nodes * max_colors)
        self.STATE = int(self.MASK_OFFSET + 16)
        self.CF = int(self.MASK_OFFSET + 24)
        self.TRIED = int(self.MASK_OFFSET + 32)
        self.END_TRIED = int(self.MASK_OFFSET + 33)
        self.ASSIGN = int(14)
        self.DOMAIN = int(235)
        self.vocab_size = int(vocab_size)

    def assign_token(self, node: int, color: int) -> int:
        return int(self.ASSIGN_OFFSET + int(node) * 4 + int(color))

    def mask_token(self, domain: Set[int]) -> int:
        bitmask = 0
        for c in domain:
            bitmask |= 1 << int(c)
        return int(self.MASK_OFFSET + bitmask)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _append_tokens(
    sequence: List[int],
    block_ids: List[int],
    tokens: Iterable[int],
    max_seq_len: int,
    block_id: int,
) -> bool:
    chunk = [int(x) for x in tokens]
    if len(sequence) + len(chunk) > int(max_seq_len):
        return False
    sequence.extend(chunk)
    block_ids.extend([int(block_id)] * len(chunk))
    return True


def _is_solution(env: GraphColorEnv, state: GraphColorState) -> bool:
    if state.propagation_pending or state.selected_node is not None:
        return False
    if int(np.count_nonzero(state.assignment == 0)) != 0:
        return False
    return not env._has_contradiction(state)


def _unassigned_nodes(state: GraphColorState) -> List[int]:
    return [int(i) for i in range(state.num_nodes) if int(state.assignment[i]) == 0]


def _sorted_candidates(state: GraphColorState, degrees: np.ndarray) -> List[int]:
    unassigned = _unassigned_nodes(state)
    return sorted(
        unassigned,
        key=lambda nd: (
            int(len(state.domains[int(nd)])),
            -int(degrees[int(nd)]),
            int(nd),
        ),
    )


def _prefix_key_from_assignment(assignment: np.ndarray) -> PrefixKey:
    nz = np.nonzero(assignment)[0]
    return tuple(sorted((int(n), int(assignment[int(n)])) for n in nz))


def _apply_assignment(env: GraphColorEnv, node: int, color: int) -> Tuple[bool, str]:
    state = env.get_state()
    if state.propagation_pending:
        res = env.step(GraphColorAction.propagate())
        if res.done:
            return False, "terminated_during_propagate"

    if state.selected_node is not None and int(state.selected_node) != int(node):
        return False, "selected_node_mismatch"

    if state.selected_node is None:
        res = env.step(GraphColorAction.select_node(int(node)))
        if not bool(res.info.get("valid", True)):
            return False, f"invalid_select:{res.info.get('reason', 'unknown')}"
        if res.done:
            return False, "terminated_after_select"

    res = env.step(GraphColorAction.assign_color(int(color)))
    if not bool(res.info.get("valid", True)):
        return False, f"invalid_assign:{res.info.get('reason', 'unknown')}"
    if res.done:
        return False, "terminated_after_assign"

    res = env.step(GraphColorAction.propagate())
    if not bool(res.info.get("valid", True)):
        return False, f"invalid_propagate:{res.info.get('reason', 'unknown')}"
    if res.done:
        return False, "terminated_after_propagate"

    return True, "ok"


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

    model: torch.nn.Module = SSASlotDecoder(
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
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(k)
        else:
            filtered[k] = v
    if skipped:
        logger.warning("Skipped %d keys due to shape mismatch", len(skipped))

    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()

    val_loss, val_acc = _extract_val_metrics(checkpoint)
    return model, {
        "checkpoint": str(checkpoint_path),
        "config": config,
        "vocab_size": int(vocab_size),
        "max_seq_len_model": int(max_seq_len_model),
        "mask_mode": str(mask_mode),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def _generate_instances(
    num_instances: int,
    num_nodes: int,
    num_colors: int,
    edge_prob: float,
    seed: int,
) -> List[np.ndarray]:
    generator = GraphGenerator(
        num_nodes=int(num_nodes),
        num_colors=int(num_colors),
        edge_prob=float(edge_prob),
        seed=int(seed),
    )
    return [generator.generate_planted().adjacency for _ in range(int(num_instances))]


def build_explicit_state_tokens(
    state: GraphColorState,
    env: GraphColorEnv,
    depth: int,
    sorted_candidates: Sequence[int],
    token_mapper: TokenMapper,
) -> List[int]:
    """Build explicit state section: [STATE ASSIGN A... SEP DOMAIN N M... SEP]."""
    tokens: List[int] = [token_mapper.STATE, token_mapper.ASSIGN]

    assignments: List[Tuple[int, int]] = []
    for node, color, _ in state.assignment_stack:
        assignments.append((int(node), int(color)))
    assignments.sort(key=lambda x: x[0])

    for node, color in assignments:
        tokens.append(token_mapper.assign_token(int(node), int(color)))
    tokens.append(CDCLTokenizer.SEP)

    tokens.append(token_mapper.DOMAIN)
    for node in sorted_candidates:
        domain = set(
            int(c) for c in env._effective_domain(state, int(node), depth=depth)
        )
        tokens.append(CDCLTokenizer.node_token(int(node)))
        tokens.append(token_mapper.mask_token(domain))
    tokens.append(CDCLTokenizer.SEP)

    return tokens


def _build_standard_state_tokens(
    token_mapper: TokenMapper,
    tokenizer: CDCLTokenizer,
    sorted_candidates: Sequence[int],
) -> List[int]:
    tokens: List[int] = [token_mapper.STATE]
    tokens.extend(tokenizer.node_token(int(nd)) for nd in sorted_candidates)
    tokens.append(tokenizer.SEP)
    return tokens


def solve_instance(
    *,
    model: torch.nn.Module,
    tokenizer: CDCLTokenizer,
    token_mapper: TokenMapper,
    adjacency: np.ndarray,
    num_colors: int,
    degrees: np.ndarray,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    mask_mode: str,
    trace_format: str,
) -> Dict[str, Any]:
    env = GraphColorEnv(
        adjacency=adjacency,
        num_colors=int(num_colors),
        solution=None,
        mode="strict",
        max_steps=int(max_steps * 4 + 10),
        propagation_mode="forward_check",
    )
    env.reset()

    sequence: List[int] = tokenizer.build_graph_prefix(
        adjacency, int(adjacency.shape[0])
    )
    block_ids: List[int] = [0] * len(sequence)
    current_block = 0
    tried_at_prefix: Dict[PrefixKey, List[Tuple[int, int]]] = {}

    stats: Dict[str, Any] = {
        "solved": False,
        "steps": 0,
        "assignments": 0,
        "backtracks": 0,
        "total_decisions": 0,
        "revisited_state_decisions": 0,
        "repeat_errors": 0,
        "termination_reason": "max_steps",
        "final_seq_len": 0,
    }

    with torch.no_grad():
        for step in range(int(max_steps)):
            stats["steps"] = int(step + 1)
            state = env.get_state()

            if _is_solution(env, state):
                stats["solved"] = True
                stats["termination_reason"] = "solved"
                break
            if state.status != GraphColorEnvStatus.RUNNING:
                stats["termination_reason"] = "env_failure"
                break

            sorted_nodes = _sorted_candidates(state, degrees)
            if not sorted_nodes:
                stats["termination_reason"] = "no_candidates"
                break

            depth = int(len(state.assignment_stack) + 1)
            effective_domains = {
                int(nd): env._effective_domain(state, int(nd), depth=depth)
                for nd in sorted_nodes
            }
            min_domain = min(
                int(len(effective_domains[int(nd)])) for nd in sorted_nodes
            )
            prefix_key = _prefix_key_from_assignment(state.assignment)

            prior = tried_at_prefix.get(prefix_key, [])
            if prior:
                current_block += 1
                tried_tokens = [token_mapper.TRIED]
                for node_id, color_id in prior:
                    tried_tokens.append(tokenizer.node_token(int(node_id)))
                    tried_tokens.append(tokenizer.color_token(int(color_id)))
                tried_tokens.append(token_mapper.END_TRIED)
                if not _append_tokens(
                    sequence,
                    block_ids,
                    tried_tokens,
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "max_seq_len"
                    break

            current_block += 1
            if str(trace_format) == "explicit":
                state_tokens = build_explicit_state_tokens(
                    state=state,
                    env=env,
                    depth=int(depth),
                    sorted_candidates=sorted_nodes,
                    token_mapper=token_mapper,
                )
            elif str(trace_format) == "standard":
                state_tokens = _build_standard_state_tokens(
                    token_mapper=token_mapper,
                    tokenizer=tokenizer,
                    sorted_candidates=sorted_nodes,
                )
            else:
                raise ValueError(f"Unsupported trace_format={trace_format}")

            if not _append_tokens(
                sequence, block_ids, state_tokens, int(max_seq_len), int(current_block)
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            if min_domain == 0:
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [token_mapper.CF],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "max_seq_len"
                    break

                if state.assignment_stack:
                    failed_node, failed_color, _ = state.assignment_stack[-1]
                    parent_prefix = tuple(
                        sorted(
                            (int(n), int(c)) for n, c, _ in state.assignment_stack[:-1]
                        )
                    )
                    parent_tried = tried_at_prefix.setdefault(parent_prefix, [])
                    failed_pair = (int(failed_node), int(failed_color))
                    if failed_pair not in parent_tried:
                        parent_tried.append(failed_pair)
                    env.backjump_to(len(state.assignment_stack) - 1)
                    stats["backtracks"] += 1
                    continue

                stats["termination_reason"] = "unsolvable"
                break

            if not _append_tokens(
                sequence,
                block_ids,
                [tokenizer.OK],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            allowed_nodes = [
                int(nd)
                for nd in sorted_nodes
                if int(len(effective_domains[int(nd)])) > 0
            ]
            if not allowed_nodes:
                stats["termination_reason"] = "no_valid_nodes"
                break

            input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
            block_tensor = torch.tensor([block_ids], dtype=torch.long, device=device)
            lm_logits, _ = model(
                input_tensor,
                block_ids=block_tensor,
                mask_mode=str(mask_mode),
            )
            node_logits = lm_logits[0, -1, :]
            node_mask = torch.full_like(node_logits, float("-inf"))
            for nd in allowed_nodes:
                node_mask[tokenizer.node_token(int(nd))] = 0.0
            node_token = int(torch.argmax(node_logits + node_mask).item())
            selected_node = int(node_token - int(tokenizer.NODE_OFFSET))
            if selected_node not in allowed_nodes:
                selected_node = int(allowed_nodes[0])
                node_token = tokenizer.node_token(selected_node)

            if not _append_tokens(
                sequence, block_ids, [node_token], int(max_seq_len), int(current_block)
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            domain = {int(c) for c in effective_domains[int(selected_node)]}
            if not domain:
                stats["termination_reason"] = "empty_domain"
                break

            if not _append_tokens(
                sequence,
                block_ids,
                [token_mapper.mask_token(domain)],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
            block_tensor = torch.tensor([block_ids], dtype=torch.long, device=device)
            lm_logits2, _ = model(
                input_tensor,
                block_ids=block_tensor,
                mask_mode=str(mask_mode),
            )
            color_logits = lm_logits2[0, -1, :]
            color_mask = torch.full_like(color_logits, float("-inf"))
            for c in sorted(domain):
                color_mask[tokenizer.color_token(int(c))] = 0.0
            color_token = int(torch.argmax(color_logits + color_mask).item())
            selected_color = int(color_token - int(tokenizer.COLOR_OFFSET))
            if selected_color not in domain:
                selected_color = int(sorted(domain)[0])
                color_token = tokenizer.color_token(selected_color)

            prior_choices = tried_at_prefix.setdefault(prefix_key, [])
            if len(prior_choices) > 0:
                stats["revisited_state_decisions"] += 1
                if (int(selected_node), int(selected_color)) in prior_choices:
                    stats["repeat_errors"] += 1
            prior_choices.append((int(selected_node), int(selected_color)))
            stats["total_decisions"] += 1

            if not _append_tokens(
                sequence, block_ids, [color_token], int(max_seq_len), int(current_block)
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            ok, reason = _apply_assignment(env, selected_node, selected_color)
            if not ok:
                stats["termination_reason"] = f"apply_failed:{reason}"
                break

            if not _append_tokens(
                sequence,
                block_ids,
                [
                    tokenizer.OK,
                    tokenizer.node_token(selected_node),
                    tokenizer.color_token(selected_color),
                ],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            stats["assignments"] += 1

    stats["final_seq_len"] = int(len(sequence))
    return stats


def _parse_str_list(raw: str) -> List[str]:
    return [str(x.strip()) for x in str(raw).split(",") if str(x).strip()]


def _resolve_ckpt_inputs(
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str]]:
    checkpoints = (
        _parse_str_list(args.checkpoints) if str(args.checkpoints).strip() else []
    )
    if not checkpoints:
        if not str(args.checkpoint).strip():
            raise ValueError("Provide --checkpoint or --checkpoints")
        checkpoints = [str(args.checkpoint).strip()]

    labels = _parse_str_list(args.labels)
    if labels and len(labels) != len(checkpoints):
        raise ValueError("--labels must match number of checkpoints when provided")

    trace_formats = _parse_str_list(args.trace_formats)
    if not trace_formats:
        trace_formats = ["explicit"] * len(checkpoints)
    if len(trace_formats) != len(checkpoints):
        raise ValueError("--trace-formats must match number of checkpoints")

    norm_trace_formats: List[str] = []
    for fmt in trace_formats:
        item = str(fmt).strip().lower()
        if item not in {"explicit", "standard"}:
            raise ValueError(f"Unsupported trace format: {fmt}")
        norm_trace_formats.append(item)

    if not labels:
        labels = [f"{norm_trace_formats[i]}_{i}" for i in range(len(checkpoints))]

    return checkpoints, labels, norm_trace_formats


def _evaluate_model(
    *,
    model: torch.nn.Module,
    tokenizer: CDCLTokenizer,
    token_mapper: TokenMapper,
    mask_mode: str,
    trace_format: str,
    instances: Sequence[np.ndarray],
    num_colors: int,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, float]:
    per_instance: List[Dict[str, Any]] = []
    for idx, adjacency in enumerate(instances):
        degrees = np.sum(adjacency, axis=1).astype(np.int64)
        stats = solve_instance(
            model=model,
            tokenizer=tokenizer,
            token_mapper=token_mapper,
            adjacency=adjacency,
            num_colors=int(num_colors),
            degrees=degrees,
            max_steps=int(max_steps),
            max_seq_len=int(max_seq_len),
            device=device,
            mask_mode=mask_mode,
            trace_format=str(trace_format),
        )
        per_instance.append(stats)

        if (idx + 1) % 25 == 0:
            solved = int(sum(int(item["solved"]) for item in per_instance))
            revisited = int(
                sum(int(item["revisited_state_decisions"]) for item in per_instance)
            )
            repeats = int(sum(int(item["repeat_errors"]) for item in per_instance))
            mean_seq_len = float(
                np.mean([float(item["final_seq_len"]) for item in per_instance])
            )
            mean_tokens_per_decision = float(
                np.mean(
                    [
                        _safe_div(
                            float(item["final_seq_len"]),
                            float(item["total_decisions"]),
                        )
                        for item in per_instance
                    ]
                )
            )
            logger.info(
                "trace=%s mask_mode=%s processed=%d/%d solve_rate=%.3f repeat_rate=%.3f mean_backtracks=%.2f mean_seq_len=%.1f mean_tokens_per_decision=%.2f",
                str(trace_format),
                str(mask_mode),
                int(idx + 1),
                int(len(instances)),
                float(_safe_div(solved, len(per_instance))),
                float(_safe_div(repeats, revisited)),
                float(np.mean([float(x["backtracks"]) for x in per_instance])),
                mean_seq_len,
                mean_tokens_per_decision,
            )

    total = int(len(per_instance))
    solved = int(sum(int(item["solved"]) for item in per_instance))
    revisited = int(
        sum(int(item["revisited_state_decisions"]) for item in per_instance)
    )
    repeats = int(sum(int(item["repeat_errors"]) for item in per_instance))
    mean_backtracks = float(
        np.mean([float(item["backtracks"]) for item in per_instance]) if total else 0.0
    )
    mean_seq_len = float(
        np.mean([float(item["final_seq_len"]) for item in per_instance])
        if total
        else 0.0
    )
    mean_tokens_per_decision = float(
        np.mean(
            [
                _safe_div(float(item["final_seq_len"]), float(item["total_decisions"]))
                for item in per_instance
            ]
        )
        if total
        else 0.0
    )
    return {
        "solve_rate": float(_safe_div(solved, total)),
        "repeat_rate": float(_safe_div(repeats, revisited)),
        "mean_backtracks": mean_backtracks,
        "mean_seq_len": mean_seq_len,
        "mean_tokens_per_decision": mean_tokens_per_decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GC checkpoints with explicit/standard state traces"
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--trace-formats", type=str, default="explicit")
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=4)
    parser.add_argument("--edge-prob", type=float, default=0.35)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    checkpoint_paths, parsed_labels, parsed_trace_formats = _resolve_ckpt_inputs(args)

    _set_seed(int(args.seed))
    tokenizer = CDCLTokenizer()
    device = torch.device(str(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_instances = _generate_instances(
        num_instances=int(args.num_instances),
        num_nodes=int(args.num_nodes),
        num_colors=int(args.num_colors),
        edge_prob=float(args.edge_prob),
        seed=int(args.seed),
    )
    logger.info(
        "generated shared instances=%d n=%d colors=%d p=%.3f",
        int(len(shared_instances)),
        int(args.num_nodes),
        int(args.num_colors),
        float(args.edge_prob),
    )

    results: List[Dict[str, Any]] = []
    started_all = time.time()

    for idx, ckpt_raw in enumerate(checkpoint_paths):
        ckpt_path = Path(ckpt_raw)
        trace_format = str(parsed_trace_formats[idx])
        logger.info(
            "loading checkpoint=%s trace_format=%s",
            str(ckpt_path),
            str(trace_format),
        )
        model, meta = _load_checkpoint(
            checkpoint_path=ckpt_path,
            device=device,
            max_seq_len_fallback=int(args.max_seq_len),
        )

        mask_mode = str(meta["mask_mode"])
        label = str(parsed_labels[idx])
        token_mapper = TokenMapper(int(meta["vocab_size"]))
        effective_seq_len = int(
            min(int(args.budget), int(args.max_seq_len), int(meta["max_seq_len_model"]))
        )

        logger.info(
            "evaluating label=%s trace=%s mask_mode=%s vocab=%d budget=%d effective_seq_len=%d",
            label,
            str(trace_format),
            mask_mode,
            int(meta["vocab_size"]),
            int(args.budget),
            int(effective_seq_len),
        )

        run_started = time.time()
        aggregate = _evaluate_model(
            model=model,
            tokenizer=tokenizer,
            token_mapper=token_mapper,
            mask_mode=mask_mode,
            trace_format=str(trace_format),
            instances=shared_instances,
            num_colors=int(args.num_colors),
            max_steps=int(args.max_steps),
            max_seq_len=int(effective_seq_len),
            device=device,
        )

        row: Dict[str, Any] = {
            "checkpoint": str(ckpt_path),
            "label": str(label),
            "trace_format": str(trace_format),
            "mask_mode": str(mask_mode),
            "solve_rate": float(aggregate["solve_rate"]),
            "repeat_rate": float(aggregate["repeat_rate"]),
            "mean_backtracks": float(aggregate["mean_backtracks"]),
            "mean_seq_len": float(aggregate["mean_seq_len"]),
            "mean_tokens_per_decision": float(aggregate["mean_tokens_per_decision"]),
            "elapsed_sec": float(time.time() - run_started),
        }
        if meta.get("val_loss") is not None:
            row["val_loss"] = float(meta["val_loss"])
        if meta.get("val_acc") is not None:
            row["val_acc"] = float(meta["val_acc"])
        results.append(row)

        logger.info(
            "completed label=%s trace=%s solve_rate=%.3f repeat_rate=%.3f mean_backtracks=%.2f mean_seq_len=%.1f mean_tokens_per_decision=%.2f",
            label,
            str(trace_format),
            float(row["solve_rate"]),
            float(row["repeat_rate"]),
            float(row["mean_backtracks"]),
            float(row["mean_seq_len"]),
            float(row["mean_tokens_per_decision"]),
        )

    payload: Dict[str, Any] = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoints": checkpoint_paths,
            "labels": parsed_labels,
            "trace_formats": parsed_trace_formats,
            "num_instances": int(args.num_instances),
            "num_nodes": int(args.num_nodes),
            "num_colors": int(args.num_colors),
            "edge_prob": float(args.edge_prob),
            "max_steps": int(args.max_steps),
            "max_seq_len": int(args.max_seq_len),
            "budget": int(args.budget),
            "device": str(args.device),
            "seed": int(args.seed),
            "elapsed_sec": float(time.time() - started_all),
        },
        "results": results,
    }

    out_path = output_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote results to %s", str(out_path))


if __name__ == "__main__":
    main()
