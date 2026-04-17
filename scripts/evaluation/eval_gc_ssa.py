#!/usr/bin/env python3
"""Closed-loop eval: SSA vs causal on graph-coloring search."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.classical_solvers import dsatur_solve
from graph_coloring.dsl import GraphColorAction
from graph_coloring.env import GraphColorEnv, GraphColorEnvStatus, GraphColorState
from graph_coloring.generator import GraphGenerator
from universal.cdcl_tokenizer import CDCLTokenizer
from universal.slot_decoder import SlotCDCLDecoder


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PrefixKey = Tuple[Tuple[int, int], ...]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
        key=lambda nd: (len(state.domains[int(nd)]), -int(degrees[int(nd)])),
    )


def _prefix_key_from_assignment(assignment: np.ndarray) -> PrefixKey:
    nz = np.nonzero(assignment)[0]
    return tuple(sorted((int(n), int(assignment[int(n)])) for n in nz))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    num_colors: int,
    max_seq_len_fallback: int,
    max_neighbors_fallback: int,
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
    attention_mode = str(config.get("attention_mode", "causal"))

    if attention_mode == "ssa":
        from universal.ssa_decoder import SSASlotDecoder

        model: torch.nn.Module = SSASlotDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
        )
        model_kind = "SSASlotDecoder"
    else:
        _ = int(max_neighbors_fallback)
        _ = int(num_colors)
        model = SlotCDCLDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
        )
        model_kind = "SlotCDCLDecoder"

    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(k)
        else:
            filtered[k] = v

    if skipped:
        logger.warning("Skipped %d keys: %s", len(skipped), skipped)

    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()
    return model, {
        "kind": model_kind,
        "max_seq_len_model": int(max_seq_len_model),
        "config": config,
        "attention_mode": attention_mode,
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


def solve_instance(
    *,
    model: torch.nn.Module,
    tokenizer: CDCLTokenizer,
    adjacency: np.ndarray,
    num_nodes: int,
    num_colors: int,
    degrees: np.ndarray,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    use_tried_markers: bool,
    use_block_ids: bool,
    log_sample: bool,
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

    sequence: List[int] = tokenizer.build_graph_prefix(adjacency, num_nodes)
    block_ids: List[int] = [0] * len(sequence)
    current_block = 0
    tried_at_prefix: Dict[PrefixKey, List[Tuple[int, int]]] = {}

    stats: Dict[str, Any] = {
        "solved": False,
        "steps": 0,
        "assignments": 0,
        "backtracks": 0,
        "total_decisions": 0,
        "first_visit_decisions": 0,
        "revisited_state_decisions": 0,
        "repeat_errors": 0,
        "novel_at_revisit": 0,
        "termination_reason": "max_steps",
        "max_block_id": 0,
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

            state_preceded_by_tried = False
            if use_tried_markers:
                prior = tried_at_prefix.get(prefix_key, [])
                if prior:
                    current_block += 1
                    stats["max_block_id"] = int(
                        max(int(stats["max_block_id"]), current_block)
                    )
                    tried_tokens = [tokenizer.TRIED]
                    for node_id, color_id in prior:
                        tried_tokens.append(tokenizer.node_token(int(node_id)))
                        tried_tokens.append(tokenizer.color_token(int(color_id)))
                    tried_tokens.append(tokenizer.END_TRIED)
                    if not _append_tokens(
                        sequence,
                        block_ids,
                        tried_tokens,
                        int(max_seq_len),
                        int(current_block),
                    ):
                        stats["termination_reason"] = "max_seq_len"
                        break
                    state_preceded_by_tried = True

            if not state_preceded_by_tried:
                current_block += 1
                stats["max_block_id"] = int(
                    max(int(stats["max_block_id"]), current_block)
                )

            state_tokens = [tokenizer.STATE]
            state_tokens.extend(tokenizer.node_token(int(nd)) for nd in sorted_nodes)
            state_tokens.append(tokenizer.SEP)
            if not _append_tokens(
                sequence,
                block_ids,
                state_tokens,
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            if min_domain == 0:
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [tokenizer.CF],
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
            if use_block_ids:
                block_tensor = torch.tensor(
                    [block_ids], dtype=torch.long, device=device
                )
                lm_logits, _ = model(input_tensor, block_ids=block_tensor)
            else:
                lm_logits, _ = model(input_tensor)
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
                sequence,
                block_ids,
                [node_token],
                int(max_seq_len),
                int(current_block),
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
                [tokenizer.mask_token(domain)],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
            if use_block_ids:
                block_tensor = torch.tensor(
                    [block_ids], dtype=torch.long, device=device
                )
                lm_logits2, _ = model(input_tensor, block_ids=block_tensor)
            else:
                lm_logits2, _ = model(input_tensor)
            color_logits = lm_logits2[0, -1, :]
            color_mask = torch.full_like(color_logits, float("-inf"))
            for c in sorted(domain):
                color_mask[tokenizer.color_token(int(c))] = 0.0
            color_token = int(torch.argmax(color_logits + color_mask).item())
            selected_color = int(color_token - int(tokenizer.COLOR_OFFSET))
            if selected_color not in domain:
                selected_color = int(sorted(domain)[0])
                color_token = tokenizer.color_token(selected_color)

            chosen_pair = (int(selected_node), int(selected_color))
            prior_choices = tried_at_prefix.setdefault(prefix_key, [])
            if len(prior_choices) == 0:
                stats["first_visit_decisions"] += 1
            else:
                stats["revisited_state_decisions"] += 1
                if chosen_pair in prior_choices:
                    stats["repeat_errors"] += 1
                else:
                    stats["novel_at_revisit"] += 1
            prior_choices.append(chosen_pair)
            stats["total_decisions"] += 1

            if not _append_tokens(
                sequence,
                block_ids,
                [color_token],
                int(max_seq_len),
                int(current_block),
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
            if log_sample and step < 6:
                logger.info(
                    "sample assign step=%d node=%d color=%d blocks=%d revisits=%d repeats=%d use_blocks=%s",
                    int(step),
                    int(selected_node),
                    int(selected_color),
                    int(stats["max_block_id"]),
                    int(stats["revisited_state_decisions"]),
                    int(stats["repeat_errors"]),
                    str(use_block_ids),
                )

    stats["max_block_id"] = int(max(int(stats["max_block_id"]), current_block))
    stats["unique_prefixes"] = int(len(tried_at_prefix))
    stats["repeat_error_rate"] = float(
        _safe_div(stats["repeat_errors"], stats["revisited_state_decisions"])
    )
    stats["revisit_fraction"] = float(
        _safe_div(stats["revisited_state_decisions"], stats["total_decisions"])
    )
    return stats


def _aggregate(mode: str, per_instance: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = int(len(per_instance))
    solved = int(sum(int(item["solved"]) for item in per_instance))
    total_decisions = int(sum(int(item["total_decisions"]) for item in per_instance))
    revisited = int(
        sum(int(item["revisited_state_decisions"]) for item in per_instance)
    )
    repeats = int(sum(int(item["repeat_errors"]) for item in per_instance))
    return {
        "mode": str(mode),
        "aggregate": {
            "solve_rate": float(_safe_div(solved, total)),
            "mean_backtracks": float(
                np.mean([float(item["backtracks"]) for item in per_instance])
                if total
                else 0.0
            ),
            "mean_max_block_id": float(
                np.mean([float(item["max_block_id"]) for item in per_instance])
                if total
                else 0.0
            ),
            "total_decisions": int(total_decisions),
            "revisited_state_decisions": int(revisited),
            "repeat_errors": int(repeats),
            "repeat_error_rate": float(_safe_div(repeats, revisited)),
            "revisit_fraction": float(_safe_div(revisited, total_decisions)),
        },
        "per_instance": list(per_instance),
    }


def _bucket_name(backtracks: int) -> str:
    bt = int(backtracks)
    if bt == 0:
        return "0"
    if 1 <= bt <= 5:
        return "1-5"
    if 6 <= bt <= 10:
        return "6-10"
    if 11 <= bt <= 20:
        return "11-20"
    return "21+"


def _bucket_table(
    ssa_rows: Sequence[Dict[str, Any]],
    causal_rows: Sequence[Dict[str, Any]],
) -> List[Tuple[str, int, int, int, int]]:
    order = ["0", "1-5", "6-10", "11-20", "21+"]
    ssa_map = {k: {"total": 0, "solved": 0} for k in order}
    causal_map = {k: {"total": 0, "solved": 0} for k in order}

    for row in ssa_rows:
        b = _bucket_name(int(row["dsatur_backtracks"]))
        ssa_map[b]["total"] += 1
        ssa_map[b]["solved"] += int(row["solved"])

    for row in causal_rows:
        b = _bucket_name(int(row["dsatur_backtracks"]))
        causal_map[b]["total"] += 1
        causal_map[b]["solved"] += int(row["solved"])

    out: List[Tuple[str, int, int, int, int]] = []
    for key in order:
        out.append(
            (
                key,
                int(ssa_map[key]["solved"]),
                int(ssa_map[key]["total"]),
                int(causal_map[key]["solved"]),
                int(causal_map[key]["total"]),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop SSA vs causal evaluation on graph coloring"
    )
    parser.add_argument("--ssa_checkpoint", type=str, required=True)
    parser.add_argument("--causal_checkpoint", type=str, required=True)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--num_nodes", type=int, default=30)
    parser.add_argument("--num_colors", type=int, default=4)
    parser.add_argument("--edge_prob", type=float, default=0.35)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    tokenizer = CDCLTokenizer()
    device = torch.device(str(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ssa_model, ssa_meta = _load_checkpoint(
        checkpoint_path=Path(args.ssa_checkpoint),
        device=device,
        num_colors=int(args.num_colors),
        max_seq_len_fallback=int(args.max_seq_len),
        max_neighbors_fallback=int(args.num_nodes),
    )
    causal_model, causal_meta = _load_checkpoint(
        checkpoint_path=Path(args.causal_checkpoint),
        device=device,
        num_colors=int(args.num_colors),
        max_seq_len_fallback=int(args.max_seq_len),
        max_neighbors_fallback=int(args.num_nodes),
    )

    shared_instances = _generate_instances(
        num_instances=int(args.num_instances),
        num_nodes=int(args.num_nodes),
        num_colors=int(args.num_colors),
        edge_prob=float(args.edge_prob),
        seed=int(args.seed),
    )
    logger.info(
        "generated instances=%d n=%d colors=%d p=%.3f",
        int(len(shared_instances)),
        int(args.num_nodes),
        int(args.num_colors),
        float(args.edge_prob),
    )

    dsatur_stats: List[Dict[str, int]] = []
    dsatur_started = time.time()
    for idx, adjacency in enumerate(shared_instances):
        result = dsatur_solve(adjacency, int(args.num_colors), max_steps=100000)
        row = {
            "success": int(bool(result.get("success", False))),
            "steps": int(cast(Any, result.get("steps", 0))),
            "backtracks": int(cast(Any, result.get("backtracks", 0))),
        }
        dsatur_stats.append(row)
        if (idx + 1) % 25 == 0:
            logger.info(
                "dsatur processed=%d/%d mean_bt=%.2f success_rate=%.3f",
                int(idx + 1),
                int(len(shared_instances)),
                float(np.mean([float(x["backtracks"]) for x in dsatur_stats])),
                float(
                    _safe_div(
                        sum(int(x["success"]) for x in dsatur_stats),
                        len(dsatur_stats),
                    )
                ),
            )
    logger.info("dsatur pre-pass elapsed_sec=%.2f", float(time.time() - dsatur_started))

    runs = [
        ("ssa", ssa_model, True, ssa_meta, str(args.ssa_checkpoint)),
        ("causal", causal_model, False, causal_meta, str(args.causal_checkpoint)),
    ]

    summaries: Dict[str, Dict[str, Any]] = {}
    per_mode_instances: Dict[str, List[Dict[str, Any]]] = {}

    for mode_name, model, use_block_ids, meta, ckpt in runs:
        _set_seed(int(args.seed))
        started = time.time()
        if str(meta.get("attention_mode", "")).lower() == "ssa":
            max_len_eval = int(args.max_seq_len)
        else:
            max_len_eval = int(
                min(int(args.max_seq_len), int(meta["max_seq_len_model"]))
            )
        per_instance: List[Dict[str, Any]] = []

        logger.info(
            "starting mode=%s use_block_ids=%s attention_mode=%s",
            str(mode_name),
            str(use_block_ids),
            str(meta.get("attention_mode", "unknown")),
        )

        for idx, adjacency in enumerate(shared_instances):
            degrees = np.sum(adjacency, axis=1).astype(np.int64)
            stats = solve_instance(
                model=model,
                tokenizer=tokenizer,
                adjacency=adjacency,
                num_nodes=int(args.num_nodes),
                num_colors=int(args.num_colors),
                degrees=degrees,
                max_steps=int(args.max_steps),
                max_seq_len=int(max_len_eval),
                device=device,
                use_tried_markers=True,
                use_block_ids=bool(use_block_ids),
                log_sample=bool(idx < 2),
            )
            stats["instance_index"] = int(idx)
            stats["dsatur_backtracks"] = int(dsatur_stats[idx]["backtracks"])
            stats["dsatur_steps"] = int(dsatur_stats[idx]["steps"])
            stats["dsatur_success"] = bool(dsatur_stats[idx]["success"])
            per_instance.append(stats)

            if (idx + 1) % 10 == 0:
                pd = int(sum(int(x["total_decisions"]) for x in per_instance))
                pr = int(sum(int(x["revisited_state_decisions"]) for x in per_instance))
                pe = int(sum(int(x["repeat_errors"]) for x in per_instance))
                mb = float(np.mean([float(x["max_block_id"]) for x in per_instance]))
                logger.info(
                    "mode=%s processed=%d/%d solve_rate=%.3f mean_bt=%.2f mean_blocks=%.2f revisit=%.3f repeat=%.3f",
                    str(mode_name),
                    int(idx + 1),
                    int(len(shared_instances)),
                    float(
                        _safe_div(
                            sum(int(x["solved"]) for x in per_instance),
                            len(per_instance),
                        )
                    ),
                    float(np.mean([float(x["backtracks"]) for x in per_instance])),
                    float(mb),
                    float(_safe_div(pr, pd)),
                    float(_safe_div(pe, pr)),
                )

        payload = _aggregate(str(mode_name), per_instance)
        payload["config"] = {
            "mode": str(mode_name),
            "seed": int(args.seed),
            "num_instances": int(args.num_instances),
            "num_nodes": int(args.num_nodes),
            "num_colors": int(args.num_colors),
            "edge_prob": float(args.edge_prob),
            "max_steps": int(args.max_steps),
            "max_seq_len": int(max_len_eval),
            "device": str(args.device),
            "checkpoint": str(ckpt),
            "model_kind": str(meta["kind"]),
            "attention_mode": str(meta.get("attention_mode", "causal")),
            "use_block_ids": bool(use_block_ids),
            "use_tried_markers": True,
            "elapsed_sec": float(time.time() - started),
        }
        out_path = output_dir / f"{mode_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        summaries[mode_name] = payload["aggregate"]
        per_mode_instances[mode_name] = per_instance
        logger.info("mode=%s wrote=%s", mode_name, out_path)

    print()
    print("Mode      Solve  MeanBT  RepeatRate  RevisitFrac  MeanMaxBlock")
    for mode_name, *_rest in runs:
        agg = summaries[mode_name]
        print(
            f"{mode_name:<8} {float(agg['solve_rate']):>5.2f} "
            f"{float(agg['mean_backtracks']):>7.2f} "
            f"{float(agg['repeat_error_rate']):>10.2f} "
            f"{float(agg['revisit_fraction']):>11.2f} "
            f"{float(agg['mean_max_block_id']):>12.2f}"
        )

    print()
    print("DSATUR BT range    SSA Solve   Causal Solve")
    for bucket, ssa_solved, ssa_total, causal_solved, causal_total in _bucket_table(
        per_mode_instances["ssa"], per_mode_instances["causal"]
    ):
        print(
            f"{bucket:<16} {ssa_solved:>3d}/{ssa_total:<3d}"
            f"       {causal_solved:>3d}/{causal_total:<3d}"
        )


if __name__ == "__main__":
    main()
