#!/usr/bin/env python3
"""Evaluate E4 verification behavior for graph coloring checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graph_coloring.dsl import GraphColorAction
from graph_coloring.env import GraphColorEnv, GraphColorEnvStatus, GraphColorState
from graph_coloring.generator import GraphGenerator
from graph_coloring.oracle import GraphColorOracle
from universal.cdcl_tokenizer import CDCLTokenizer
from universal.slot_decoder import SlotCDCLDecoder
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PrefixKey = Tuple[Tuple[int, int], ...]
LOCAL_LEGAL = 239


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
    block_id: int,
    max_seq_len: int,
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


def _prefix_key_from_stack(stack: Sequence[Tuple[int, int, Any]]) -> PrefixKey:
    return tuple(sorted((int(n), int(c)) for n, c, _ in stack))


def _local_legal(
    adjacency: np.ndarray,
    assignment: np.ndarray,
    node: int,
    num_colors: int,
) -> List[int]:
    neighbor_colors = set()
    for nb in range(len(assignment)):
        if adjacency[int(node)][nb] and int(assignment[nb]) != 0:
            neighbor_colors.add(int(assignment[nb]))
    return sorted(c for c in range(1, int(num_colors) + 1) if c not in neighbor_colors)


def _decode_color_token(
    token_id: int, tokenizer: CDCLTokenizer, num_colors: int
) -> int | None:
    tid = int(token_id)
    lo = int(tokenizer.COLOR_OFFSET) + 1
    hi = int(tokenizer.COLOR_OFFSET) + int(num_colors)
    if lo <= tid <= hi:
        return int(tid - int(tokenizer.COLOR_OFFSET))
    return None


def _apply_assignment_allow_repeat(
    env: GraphColorEnv, node: int, color: int
) -> Tuple[bool, str]:
    """Apply assignment while clearing nogood blocks for this exact color if needed."""
    state = env.get_state()
    if state.propagation_pending:
        res = env.step(GraphColorAction.propagate())
        if res.done:
            return False, "terminated_during_pending_propagate"

    # Allow repeated tried-color applications by clearing current-depth nogood entry.
    depth = int(len(env._state.assignment_stack) + 1)  # type: ignore[union-attr]
    per_depth = env._state.nogoods.get(depth, {})  # type: ignore[union-attr]
    if int(node) in per_depth and int(color) in per_depth[int(node)]:
        per_depth[int(node)].discard(int(color))

    if state.selected_node is None:
        res = env.step(GraphColorAction.select_node(int(node)))
        if not bool(res.info.get("valid", True)):
            return False, f"invalid_select:{res.info.get('reason', 'unknown')}"
        if res.done:
            return False, "terminated_after_select"
    elif int(state.selected_node) != int(node):
        return False, "selected_node_mismatch"

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


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    max_seq_len_fallback: int,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = ckpt["model_state_dict"]
    config = ckpt.get("config", {})
    vocab_size = int(
        config.get("vocab_size", state_dict["token_embedding.weight"].shape[0])
    )
    d_model = int(config.get("d_model", 256))
    n_layers = int(config.get("n_layers", 6))
    n_heads = int(config.get("n_heads", 8))
    n_slots = int(config.get("n_slots", 32))
    max_seq_len_model = int(config.get("max_seq_len", int(max_seq_len_fallback)))
    dropout = float(config.get("dropout", 0.1))
    attention_mode = str(config.get("attention_mode", "causal")).lower()

    if attention_mode == "ssa":
        model: torch.nn.Module = SSASlotDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
        )
        kind = "SSASlotDecoder"
    else:
        model = SlotCDCLDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
        )
        kind = "SlotCDCLDecoder"

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

    return model, {
        "kind": kind,
        "attention_mode": attention_mode,
        "mask_mode": str(config.get("mask_mode", "full_causal")),
        "max_seq_len_model": int(max_seq_len_model),
        "checkpoint": str(checkpoint_path),
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


@torch.no_grad()
def solve_instance(
    *,
    model: torch.nn.Module,
    meta: Dict[str, Any],
    tokenizer: CDCLTokenizer,
    adjacency: np.ndarray,
    num_nodes: int,
    num_colors: int,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, Any]:
    env = GraphColorEnv(
        adjacency=adjacency,
        num_colors=int(num_colors),
        solution=None,
        mode="strict",
        max_steps=int(max_steps * 6 + 20),
        propagation_mode="forward_check",
    )
    oracle = GraphColorOracle(env)
    env.reset()

    sequence: List[int] = tokenizer.build_graph_prefix(adjacency, int(num_nodes))
    block_ids: List[int] = [0] * len(sequence)
    current_block = 0

    tried_for_state_node: Dict[Tuple[PrefixKey, int], List[int]] = {}
    repeat_flag_stack: List[bool] = []

    stats: Dict[str, Any] = {
        "solved": False,
        "steps": 0,
        "assignments": 0,
        "backtracks": 0,
        "repeat_errors": 0,
        "repeat_opportunities": 0,
        "exhausted_states": 0,
        "correct_cf_on_exhausted": 0,
        "non_exhausted_states": 0,
        "false_cf_on_non_exhausted": 0,
        "correct_color_on_non_exhausted": 0,
        "oracle_exact_match_on_non_exhausted": 0,
        "invalid_predictions": 0,
        "invalid_color_choices": 0,
        "backtrack_correct": 0,
        "backtrack_false_positive": 0,
        "repeat_induced_backtracks": 0,
        "termination_reason": "max_steps",
    }

    use_block_ids = str(meta.get("attention_mode", "causal")) == "ssa"
    mask_mode = str(meta.get("mask_mode", "full_causal"))

    for step in range(int(max_steps)):
        stats["steps"] = int(step + 1)
        state = env.get_state()

        if _is_solution(env, state):
            stats["solved"] = True
            stats["termination_reason"] = "solved"
            break
        has_contradiction = bool(env._has_contradiction(state))
        if state.status != GraphColorEnvStatus.RUNNING and not has_contradiction:
            stats["termination_reason"] = "env_failure"
            break

        depth = int(len(state.assignment_stack) + 1)
        if has_contradiction:
            unassigned = [
                int(i)
                for i in range(int(num_nodes))
                if int(state.assignment[int(i)]) == 0
            ]
            if not unassigned:
                stats["termination_reason"] = "no_unassigned_nodes"
                break
            contradicted = [
                int(nd)
                for nd in unassigned
                if len(env._effective_domain(state, int(nd), depth=depth)) == 0
            ]
            selected_node = int(contradicted[0] if contradicted else unassigned[0])
            logger.info(
                "contradiction_state step=%d depth=%d selected_node=%d unassigned=%d contradicted=%d",
                int(step),
                int(depth),
                int(selected_node),
                int(len(unassigned)),
                int(len(contradicted)),
            )
        else:
            selected_node = oracle._dsatur_select(state, depth=depth)
            if selected_node is None:
                stats["termination_reason"] = "no_selectable_node"
                break
        selected_node = int(selected_node)

        sorted_nodes = sorted(
            [
                int(i)
                for i in range(int(num_nodes))
                if int(state.assignment[int(i)]) == 0
            ],
            key=lambda nd: (
                len(env._effective_domain(state, int(nd), depth=depth)),
                -int(np.sum(adjacency[int(nd)])),
            ),
        )
        local_legal = _local_legal(
            adjacency, state.assignment, int(selected_node), int(num_colors)
        )

        prefix_key = _prefix_key_from_stack(state.assignment_stack)
        tried_key = (prefix_key, int(selected_node))
        tried_colors = list(tried_for_state_node.get(tried_key, []))
        tried_set = set(int(c) for c in tried_colors)

        available = [int(c) for c in local_legal if int(c) not in tried_set]
        exhausted = len(available) == 0
        oracle_color = int(min(available)) if not exhausted else -1
        local_legal_set = set(int(c) for c in local_legal)
        gt_token = (
            int(tokenizer.CF)
            if exhausted
            else int(tokenizer.color_token(int(oracle_color)))
        )

        if any(int(c) in tried_set for c in local_legal):
            stats["repeat_opportunities"] += 1
        if exhausted:
            stats["exhausted_states"] += 1
        else:
            stats["non_exhausted_states"] += 1

        next_block = int(current_block + 1)
        decision_prefix: List[int] = []
        if len(tried_colors) > 0:
            decision_prefix.append(int(tokenizer.TRIED))
            for c in tried_colors:
                decision_prefix.append(int(tokenizer.node_token(int(selected_node))))
                decision_prefix.append(int(tokenizer.color_token(int(c))))
            decision_prefix.append(int(tokenizer.END_TRIED))
        decision_prefix.append(int(tokenizer.STATE))
        decision_prefix.extend(
            int(tokenizer.node_token(int(nd))) for nd in sorted_nodes
        )
        decision_prefix.append(int(tokenizer.SEP))
        decision_prefix.append(int(LOCAL_LEGAL))
        decision_prefix.extend(int(tokenizer.color_token(int(c))) for c in local_legal)
        decision_prefix.append(int(tokenizer.SEP))
        decision_prefix.append(int(tokenizer.OK))

        if not _append_tokens(
            sequence,
            block_ids,
            decision_prefix,
            next_block,
            int(max_seq_len),
        ):
            stats["termination_reason"] = "budget_exceeded"
            break

        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        if use_block_ids:
            block_tensor = torch.tensor([block_ids], dtype=torch.long, device=device)
            lm_logits, _ = model(input_ids, block_ids=block_tensor, mask_mode=mask_mode)
        else:
            lm_logits, _ = model(input_ids)
        pred_token = int(torch.argmax(lm_logits[0, -1, :]).item())

        pred_color = _decode_color_token(pred_token, tokenizer, int(num_colors))
        predicted_cf = int(pred_token) == int(tokenizer.CF)

        chosen_token: int
        chosen_color: int | None
        chosen_is_repeat = False
        choose_backtrack = False

        if pred_color is not None:
            if int(pred_color) in local_legal_set:
                chosen_token = int(tokenizer.color_token(int(pred_color)))
                chosen_color = int(pred_color)
                chosen_is_repeat = int(pred_color) in tried_set
                if chosen_is_repeat:
                    stats["repeat_errors"] += 1
                elif (not exhausted) and int(pred_color) in set(available):
                    stats["correct_color_on_non_exhausted"] += 1

                if (not exhausted) and int(pred_color) == int(oracle_color):
                    stats["oracle_exact_match_on_non_exhausted"] += 1
            else:
                stats["invalid_color_choices"] += 1
                chosen_token = int(tokenizer.CF)
                chosen_color = None
                choose_backtrack = True
        elif predicted_cf:
            chosen_token = int(tokenizer.CF)
            chosen_color = None
            choose_backtrack = True
            if exhausted:
                stats["correct_cf_on_exhausted"] += 1
                stats["backtrack_correct"] += 1
            else:
                stats["false_cf_on_non_exhausted"] += 1
                stats["backtrack_false_positive"] += 1
        else:
            stats["invalid_predictions"] += 1
            chosen_token = int(tokenizer.CF)
            chosen_color = None
            choose_backtrack = True

        if not _append_tokens(
            sequence, block_ids, [chosen_token], next_block, int(max_seq_len)
        ):
            stats["termination_reason"] = "budget_exceeded"
            break

        if choose_backtrack:
            if not _append_tokens(
                sequence,
                block_ids,
                [int(tokenizer.OK), int(tokenizer.CF)],
                next_block,
                int(max_seq_len),
            ):
                stats["termination_reason"] = "budget_exceeded"
                break

            if not state.assignment_stack:
                stats["termination_reason"] = "unsat_root"
                current_block = int(next_block)
                break

            top_repeat = bool(repeat_flag_stack[-1]) if repeat_flag_stack else False
            if top_repeat:
                stats["repeat_induced_backtracks"] += 1
            if repeat_flag_stack:
                repeat_flag_stack.pop()

            failed_node, failed_color, _ = state.assignment_stack[-1]
            parent_prefix = _prefix_key_from_stack(state.assignment_stack[:-1])
            parent_key = (parent_prefix, int(failed_node))
            prior = tried_for_state_node.setdefault(parent_key, [])
            if int(failed_color) not in prior:
                prior.append(int(failed_color))
            env.backjump_to(len(state.assignment_stack) - 1)
            stats["backtracks"] += 1
            current_block = int(next_block)
            continue

        if chosen_color is None:
            stats["termination_reason"] = "internal_no_color"
            break

        ok, reason = _apply_assignment_allow_repeat(
            env, int(selected_node), int(chosen_color)
        )
        if not ok:
            stats["termination_reason"] = f"apply_failed:{reason}"
            break

        repeat_flag_stack.append(bool(chosen_is_repeat))
        if not _append_tokens(
            sequence,
            block_ids,
            [
                int(tokenizer.OK),
                int(tokenizer.node_token(int(selected_node))),
                int(tokenizer.color_token(int(chosen_color))),
            ],
            next_block,
            int(max_seq_len),
        ):
            stats["termination_reason"] = "budget_exceeded"
            break

        current_block = int(next_block)
        stats["assignments"] += 1

        if step < 6:
            logger.info(
                "sample step=%d node=%d local_legal=%s tried=%s pred=%d gt=%d chosen=%d repeat=%s exhausted=%s",
                int(step),
                int(selected_node),
                str(local_legal),
                str(tried_colors),
                int(pred_token),
                int(gt_token),
                int(chosen_token),
                str(bool(chosen_is_repeat)),
                str(bool(exhausted)),
            )

    stats["repeat_error_rate"] = float(
        _safe_div(stats["repeat_errors"], stats["repeat_opportunities"])
    )
    stats["backtrack_accuracy"] = float(
        _safe_div(stats["correct_cf_on_exhausted"], stats["exhausted_states"])
    )
    stats["backtrack_false_positive_rate"] = float(
        _safe_div(stats["false_cf_on_non_exhausted"], stats["non_exhausted_states"])
    )
    stats["color_accuracy"] = float(
        _safe_div(
            stats["correct_color_on_non_exhausted"], stats["non_exhausted_states"]
        )
    )
    stats["oracle_exact_color_accuracy"] = float(
        _safe_div(
            stats["oracle_exact_match_on_non_exhausted"], stats["non_exhausted_states"]
        )
    )
    stats["repeat_overhead"] = float(
        _safe_div(stats["repeat_induced_backtracks"], stats["backtracks"])
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E4 verification metrics")
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=4)
    parser.add_argument("--edge-prob", type=float, default=0.35)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="experiments/e4-eval/")
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device)
    tokenizer = CDCLTokenizer()

    checkpoints = [
        Path(x.strip()) for x in str(args.checkpoints).split(",") if x.strip()
    ]
    labels = [x.strip() for x in str(args.labels).split(",") if x.strip()]
    if len(checkpoints) != len(labels):
        raise ValueError("--checkpoints and --labels must have same count")
    if len(checkpoints) == 0:
        raise ValueError("No checkpoints provided")

    instances = _generate_instances(
        num_instances=int(args.num_instances),
        num_nodes=int(args.num_nodes),
        num_colors=int(args.num_colors),
        edge_prob=float(args.edge_prob),
        seed=int(args.seed),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for label, ckpt in zip(labels, checkpoints):
        t0 = time.time()
        model, meta = _load_checkpoint(
            checkpoint_path=ckpt,
            device=device,
            max_seq_len_fallback=int(args.budget),
        )

        per_instance: List[Dict[str, Any]] = []
        for idx, adjacency in enumerate(instances):
            stats = solve_instance(
                model=model,
                meta=meta,
                tokenizer=tokenizer,
                adjacency=adjacency,
                num_nodes=int(args.num_nodes),
                num_colors=int(args.num_colors),
                max_steps=int(args.max_steps),
                max_seq_len=int(args.budget),
                device=device,
            )
            per_instance.append(stats)
            if (idx + 1) % 25 == 0:
                logger.info(
                    "eval label=%s processed=%d/%d solve_rate=%.3f repeat_err=%.3f bt_acc=%.3f color_acc=%.3f",
                    str(label),
                    int(idx + 1),
                    int(len(instances)),
                    float(np.mean([1.0 if s["solved"] else 0.0 for s in per_instance])),
                    float(
                        np.mean([float(s["repeat_error_rate"]) for s in per_instance])
                    ),
                    float(
                        np.mean([float(s["backtrack_accuracy"]) for s in per_instance])
                    ),
                    float(np.mean([float(s["color_accuracy"]) for s in per_instance])),
                )

        aggregate = {
            "label": str(label),
            "checkpoint": str(ckpt),
            "model_kind": str(meta.get("kind", "unknown")),
            "attention_mode": str(meta.get("attention_mode", "unknown")),
            "mask_mode": str(meta.get("mask_mode", "unknown")),
            "num_instances": int(len(per_instance)),
            "solve_rate": float(
                np.mean([1.0 if s["solved"] else 0.0 for s in per_instance])
            ),
            "repeat_error_rate": float(
                _safe_div(
                    sum(float(s["repeat_errors"]) for s in per_instance),
                    sum(float(s["repeat_opportunities"]) for s in per_instance),
                )
            ),
            "backtrack_accuracy": float(
                _safe_div(
                    sum(float(s["correct_cf_on_exhausted"]) for s in per_instance),
                    sum(float(s["exhausted_states"]) for s in per_instance),
                )
            ),
            "backtrack_false_positive": float(
                _safe_div(
                    sum(float(s["false_cf_on_non_exhausted"]) for s in per_instance),
                    sum(float(s["non_exhausted_states"]) for s in per_instance),
                )
            ),
            "color_accuracy": float(
                _safe_div(
                    sum(
                        float(s["correct_color_on_non_exhausted"]) for s in per_instance
                    ),
                    sum(float(s["non_exhausted_states"]) for s in per_instance),
                )
            ),
            "oracle_exact_color_accuracy": float(
                _safe_div(
                    sum(
                        float(s["oracle_exact_match_on_non_exhausted"])
                        for s in per_instance
                    ),
                    sum(float(s["non_exhausted_states"]) for s in per_instance),
                )
            ),
            "repeat_overhead": float(
                _safe_div(
                    sum(float(s["repeat_induced_backtracks"]) for s in per_instance),
                    sum(float(s["backtracks"]) for s in per_instance),
                )
            ),
            "mean_backtracks": float(
                np.mean([float(s["backtracks"]) for s in per_instance])
            ),
            "mean_steps": float(np.mean([float(s["steps"]) for s in per_instance])),
            "elapsed_sec": float(time.time() - t0),
        }
        all_results.append({"aggregate": aggregate, "instances": per_instance})

        out_path = output_dir / f"{label}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results[-1], f, indent=2)
        logger.info(
            "done label=%s solve_rate=%.3f repeat_err=%.3f bt_acc=%.3f bt_fp=%.3f color_acc=%.3f repeat_overhead=%.3f elapsed=%.1fs",
            str(label),
            float(aggregate["solve_rate"]),
            float(aggregate["repeat_error_rate"]),
            float(aggregate["backtrack_accuracy"]),
            float(aggregate["backtrack_false_positive"]),
            float(aggregate["color_accuracy"]),
            float(aggregate["repeat_overhead"]),
            float(aggregate["elapsed_sec"]),
        )

    summary_rows = [x["aggregate"] for x in all_results]
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info("wrote summary=%s", str(summary_path))


if __name__ == "__main__":
    main()
