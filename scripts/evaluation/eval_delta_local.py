#!/usr/bin/env python3
"""Closed-loop eval for DeltaLocalSlotDecoder on graph coloring."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from graph_coloring.dsl import GraphColorAction
from graph_coloring.env import GraphColorEnv, GraphColorEnvStatus, GraphColorState
from graph_coloring.generator import GraphGenerator
from universal.cdcl_tokenizer import CDCLTokenizer
from universal.slot_decoder import DeltaLocalSlotDecoder


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _append_tokens(
    sequence: List[int], tokens: Iterable[int], max_seq_len: int
) -> bool:
    token_list = [int(t) for t in tokens]
    if int(len(sequence)) + int(len(token_list)) > int(max_seq_len):
        return False
    sequence.extend(token_list)
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


def _is_node_token(token: int, tokenizer: CDCLTokenizer) -> bool:
    tid = int(token)
    return int(tokenizer.NODE_OFFSET) <= tid < int(tokenizer.COLOR_OFFSET)


def _is_color_token(token: int, tokenizer: CDCLTokenizer, num_colors: int) -> bool:
    tid = int(token)
    lo = int(tokenizer.COLOR_OFFSET) + 1
    hi = int(tokenizer.COLOR_OFFSET) + int(num_colors)
    return lo <= tid <= hi


def _decode_state_node_token(
    token: int,
    tokenizer: CDCLTokenizer,
) -> Optional[int]:
    tid = int(token)
    if _is_node_token(tid, tokenizer):
        return int(tid - int(tokenizer.NODE_OFFSET))
    if tokenizer.is_assign_token(tid):
        node, _ = tokenizer.decode_assign_token(tid)
        return int(node)
    return None


def _build_delta_local_inputs(
    *,
    tokenizer: CDCLTokenizer,
    adjacency: np.ndarray,
    assignment: np.ndarray,
    sequence: Sequence[int],
    state_start: int,
    state_end: int,
    max_neighbors: int,
    last_assign: Optional[Tuple[int, int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Build neighbor_positions, neighbor_mask, assign_positions tensors."""
    nb_pos = torch.zeros((1, int(max_neighbors)), dtype=torch.long, device=device)
    nb_mask = torch.zeros((1, int(max_neighbors)), dtype=torch.float32, device=device)
    assign_pos = torch.zeros((1,), dtype=torch.long, device=device)

    if last_assign is None:
        return (
            nb_pos,
            nb_mask,
            assign_pos,
            {
                "assign_node": -1,
                "assign_color": -1,
                "num_neighbors": 0,
                "num_neighbors_in_state": 0,
            },
        )

    assign_node, assign_color, assign_anchor_pos = last_assign
    assign_pos[0] = int(assign_anchor_pos)

    node_to_pos: Dict[int, int] = {}
    for pos in range(int(state_start) + 1, int(state_end)):
        decoded_node = _decode_state_node_token(int(sequence[pos]), tokenizer)
        if decoded_node is not None and int(decoded_node) not in node_to_pos:
            node_to_pos[int(decoded_node)] = int(pos)

    neighbors = np.flatnonzero(adjacency[int(assign_node)]).astype(np.int64).tolist()
    uncolored_neighbors = [
        int(nb)
        for nb in neighbors
        if int(assignment[int(nb)]) == 0 and int(nb) != int(assign_node)
    ]

    kept = 0
    for nb in uncolored_neighbors:
        if int(nb) not in node_to_pos:
            continue
        if kept >= int(max_neighbors):
            break
        nb_pos[0, kept] = int(node_to_pos[int(nb)])
        nb_mask[0, kept] = 1.0
        kept += 1

    metadata = {
        "assign_node": int(assign_node),
        "assign_color": int(assign_color),
        "num_neighbors": int(len(uncolored_neighbors)),
        "num_neighbors_in_state": int(kept),
    }
    return nb_pos, nb_mask, assign_pos, metadata


def solve_instance(
    model: DeltaLocalSlotDecoder,
    tokenizer: CDCLTokenizer,
    adjacency: np.ndarray,
    num_nodes: int,
    num_colors: int,
    degrees: np.ndarray,
    max_steps: int,
    max_seq_len: int,
    max_neighbors: int,
    device: torch.device,
    cf_threshold: float = 0.5,
    no_safety_net: bool = False,
    log_sample: bool = False,
    oracle_policy: bool = False,
    oracle_verify: bool = False,
) -> Dict[str, Any]:
    """Solve one instance with closed-loop delta-local conflict checks."""
    env_max_steps = int(max_steps * 4 + 10)
    env = GraphColorEnv(
        adjacency=adjacency,
        num_colors=int(num_colors),
        solution=None,
        mode="strict",
        max_steps=env_max_steps,
        propagation_mode="forward_check",
    )
    env.reset()

    sequence: List[int] = tokenizer.build_graph_prefix(adjacency, num_nodes)

    stats: Dict[str, Any] = {
        "solved": False,
        "steps": 0,
        "assignments": 0,
        "backtracks": 0,
        "verify_correct_cf": 0,
        "verify_missed_cf": 0,
        "verify_false_cf": 0,
        "verify_correct_ok": 0,
        "forced_backtracks": 0,
        "termination_reason": "max_steps",
    }

    conflict_miss_limit = 999999 if no_safety_net else 3
    consecutive_misses = 0
    assignment_anchor_stack: List[Tuple[int, int, int]] = []

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

            unassigned = _unassigned_nodes(state)
            if not unassigned:
                stats["termination_reason"] = "no_unassigned_unsolved"
                break

            sorted_candidates = _sorted_candidates(state, degrees)
            if not sorted_candidates:
                stats["termination_reason"] = "no_candidates"
                break

            depth = int(len(state.assignment_stack) + 1)
            effective_domains = {
                int(nd): env._effective_domain(state, int(nd), depth)
                for nd in sorted_candidates
            }
            min_domain_size = min(
                int(len(effective_domains[int(nd)])) for nd in sorted_candidates
            )
            real_conflict = int(min_domain_size) == 0

            state_tokens = [tokenizer.STATE]
            for nd in sorted_candidates:
                state_tokens.append(tokenizer.node_token(int(nd)))
            state_tokens.append(tokenizer.SEP)

            state_start = int(len(sequence))
            if not _append_tokens(sequence, state_tokens, int(max_seq_len)):
                stats["termination_reason"] = "max_seq_len"
                break
            state_end = int(len(sequence) - 1)

            input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
            nb_pos, nb_mask, asgn_pos, delta_meta = _build_delta_local_inputs(
                sequence=sequence,
                tokenizer=tokenizer,
                adjacency=adjacency,
                assignment=state.assignment,
                state_start=state_start,
                state_end=state_end,
                max_neighbors=int(max_neighbors),
                last_assign=(
                    assignment_anchor_stack[-1] if assignment_anchor_stack else None
                ),
                device=device,
            )

            lm_logits: Optional[torch.Tensor] = None
            if not (oracle_policy and oracle_verify):
                if oracle_policy and not oracle_verify:
                    _, global_logits, _, _ = model(
                        input_tensor,
                        neighbor_positions=nb_pos,
                        neighbor_mask=nb_mask,
                        assign_positions=asgn_pos,
                    )
                elif oracle_verify and not oracle_policy:
                    lm_logits, _ = model(input_tensor)
                else:
                    lm_logits, global_logits, _, _ = model(
                        input_tensor,
                        neighbor_positions=nb_pos,
                        neighbor_mask=nb_mask,
                        assign_positions=asgn_pos,
                    )

            if oracle_verify:
                cf_prob = 1.0 if real_conflict else 0.0
                model_says_cf = bool(real_conflict)
            else:
                cf_prob = float(torch.softmax(global_logits[0], dim=-1)[1].item())
                model_says_cf = cf_prob > float(cf_threshold)

            if log_sample and step < 6:
                logger.info(
                    "sample step=%d candidates=%d min_domain=%d cf_prob=%.3f real_conflict=%s verdict=%s assign=(n%d,c%d,pos=%d) active_nb=%d/%d",
                    int(step),
                    int(len(sorted_candidates)),
                    int(min_domain_size),
                    float(cf_prob),
                    str(bool(real_conflict)),
                    "CF" if model_says_cf else "OK",
                    int(delta_meta["assign_node"]),
                    int(delta_meta["assign_color"]),
                    int(asgn_pos[0].item()),
                    int(delta_meta["num_neighbors_in_state"]),
                    int(delta_meta["num_neighbors"]),
                )

            if real_conflict and model_says_cf:
                stats["verify_correct_cf"] += 1
                consecutive_misses = 0
            elif real_conflict and not model_says_cf:
                stats["verify_missed_cf"] += 1
                consecutive_misses += 1
            elif not real_conflict and model_says_cf:
                stats["verify_false_cf"] += 1
                consecutive_misses = 0
            else:
                stats["verify_correct_ok"] += 1
                consecutive_misses = 0

            should_backtrack = bool(model_says_cf)
            if real_conflict and consecutive_misses >= int(conflict_miss_limit):
                should_backtrack = True
                stats["forced_backtracks"] += 1
                consecutive_misses = 0

            if should_backtrack:
                if not _append_tokens(sequence, [tokenizer.CF], int(max_seq_len)):
                    stats["termination_reason"] = "max_seq_len"
                    break
                if state.assignment_stack:
                    target_depth = int(len(state.assignment_stack) - 1)
                    info = env.backjump_to(int(target_depth))
                    assignment_anchor_stack = assignment_anchor_stack[
                        : int(target_depth)
                    ]
                    stats["backtracks"] += 1
                    if log_sample and step < 6:
                        logger.info(
                            "sample backtrack step=%d target_depth=%d popped=%d final_depth=%d",
                            int(step),
                            int(target_depth),
                            int(info["num_popped"]),
                            int(info["final_stack_depth"]),
                        )
                    continue
                stats["termination_reason"] = "unsolvable"
                break

            if not _append_tokens(sequence, [tokenizer.OK], int(max_seq_len)):
                stats["termination_reason"] = "max_seq_len"
                break

            if real_conflict:
                continue

            allowed_nodes = [
                int(nd)
                for nd in sorted_candidates
                if int(len(effective_domains[int(nd)])) > 0
            ]
            if not allowed_nodes:
                stats["termination_reason"] = "no_valid_nodes"
                break

            if oracle_policy:
                selected_node = int(allowed_nodes[0])
                node_token = tokenizer.node_token(int(selected_node))
            else:
                if lm_logits is None:
                    raise RuntimeError("lm_logits missing for learned policy")
                next_logits = lm_logits[0, -1, :]
                mask = torch.full_like(next_logits, float("-inf"))
                for nd in allowed_nodes:
                    mask[tokenizer.node_token(int(nd))] = 0.0
                constrained = next_logits + mask
                node_token = int(torch.argmax(constrained).item())
                selected_node = int(node_token - int(tokenizer.NODE_OFFSET))
                if int(selected_node) not in allowed_nodes:
                    selected_node = int(allowed_nodes[0])
                    node_token = tokenizer.node_token(int(selected_node))

            if not _append_tokens(sequence, [int(node_token)], int(max_seq_len)):
                stats["termination_reason"] = "max_seq_len"
                break

            domain = effective_domains[int(selected_node)]
            domain_set = {int(c) for c in domain}
            if not domain_set:
                stats["termination_reason"] = "empty_domain"
                break

            if oracle_policy:
                selected_color = int(min(domain_set))
                color_token = tokenizer.color_token(int(selected_color))
            else:
                if not _append_tokens(
                    sequence, [tokenizer.mask_token(domain_set)], int(max_seq_len)
                ):
                    stats["termination_reason"] = "max_seq_len"
                    break

                input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
                lm_logits2, _ = model(input_tensor)
                color_logits = lm_logits2[0, -1, :]

                color_mask = torch.full_like(color_logits, float("-inf"))
                for c in domain_set:
                    color_mask[tokenizer.color_token(int(c))] = 0.0
                constrained_color = color_logits + color_mask
                color_token = int(torch.argmax(constrained_color).item())
                selected_color = int(color_token - int(tokenizer.COLOR_OFFSET))
                if int(selected_color) not in domain_set:
                    selected_color = int(sorted(domain_set)[0])
                    color_token = tokenizer.color_token(int(selected_color))

            if not _append_tokens(sequence, [int(color_token)], int(max_seq_len)):
                stats["termination_reason"] = "max_seq_len"
                break

            ok, reason = _apply_assignment(env, int(selected_node), int(selected_color))
            if not ok:
                stats["termination_reason"] = f"apply_failed:{reason}"
                break

            if not _append_tokens(
                sequence,
                [
                    tokenizer.OK,
                    tokenizer.node_token(int(selected_node)),
                    tokenizer.color_token(int(selected_color)),
                ],
                int(max_seq_len),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            anchor_pos = int(len(sequence) - 3)
            assignment_anchor_stack = [
                entry
                for entry in assignment_anchor_stack
                if int(entry[0]) != int(selected_node)
            ]
            assignment_anchor_stack.append(
                (int(selected_node), int(selected_color), int(anchor_pos))
            )

            stats["assignments"] += 1
            if log_sample and step < 6:
                logger.info(
                    "sample assign step=%d node=%d color=%d domain=%s",
                    int(step),
                    int(selected_node),
                    int(selected_color),
                    str(sorted(domain_set)),
                )

    return stats


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _compute_metrics(all_stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = int(len(all_stats))
    solved = [s for s in all_stats if bool(s["solved"])]

    verify_correct_cf = int(sum(int(s["verify_correct_cf"]) for s in all_stats))
    verify_missed_cf = int(sum(int(s["verify_missed_cf"]) for s in all_stats))
    verify_false_cf = int(sum(int(s["verify_false_cf"]) for s in all_stats))
    verify_correct_ok = int(sum(int(s["verify_correct_ok"]) for s in all_stats))

    return {
        "solve_rate": float(_safe_div(len(solved), total)),
        "precision": float(
            _safe_div(verify_correct_cf, verify_correct_cf + verify_false_cf)
        ),
        "recall": float(
            _safe_div(verify_correct_cf, verify_correct_cf + verify_missed_cf)
        ),
        "false_cf": int(verify_false_cf),
        "correct_cf": int(verify_correct_cf),
        "missed_cf": int(verify_missed_cf),
        "correct_ok": int(verify_correct_ok),
        "mean_backtracks": float(
            np.mean([float(s["backtracks"]) for s in all_stats]) if total else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DeltaLocalSlotDecoder closed-loop on graph coloring"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--num_nodes", type=int, default=30)
    parser.add_argument("--num_colors", type=int, default=4)
    parser.add_argument("--edge_prob", type=float, default=0.35)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--cf_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-safety-net", action="store_true", default=False)
    parser.add_argument("--oracle_policy", action="store_true", default=False)
    parser.add_argument("--oracle_verify", action="store_true", default=False)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    tokenizer = CDCLTokenizer()
    device = torch.device(str(args.device))

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("checkpoint missing model_state_dict")

    config = checkpoint.get("config", {})
    vocab_size = int(config.get("vocab_size", 392))
    d_model = int(config.get("d_model", 256))
    n_layers = int(config.get("n_layers", 6))
    n_heads = int(config.get("n_heads", 8))
    n_slots = int(config.get("n_slots", 32))
    max_seq_len_model = int(config.get("max_seq_len", int(args.max_seq_len)))
    dropout = float(config.get("dropout", 0.1))
    n_colors_model = int(config.get("n_colors", int(args.num_colors)))
    max_neighbors = int(config.get("max_neighbors", int(args.num_nodes)))

    model = DeltaLocalSlotDecoder(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        n_slots=int(n_slots),
        max_seq_len=int(max_seq_len_model),
        dropout=float(dropout),
        n_colors=int(n_colors_model),
        max_neighbors=int(max_neighbors),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    eval_max_seq_len = int(min(int(args.max_seq_len), int(max_seq_len_model)))
    logger.info(
        "model_vocab=%d d_model=%d n_layers=%d n_heads=%d n_slots=%d max_seq_len=%d eval_max_seq_len=%d max_neighbors=%d",
        int(vocab_size),
        int(d_model),
        int(n_layers),
        int(n_heads),
        int(n_slots),
        int(max_seq_len_model),
        int(eval_max_seq_len),
        int(max_neighbors),
    )
    logger.info(
        "num_instances=%d cf_threshold=%.3f no_safety_net=%s",
        int(args.num_instances),
        float(args.cf_threshold),
        str(bool(args.no_safety_net)),
    )
    logger.info(
        "mode oracle_policy=%s oracle_verify=%s",
        str(bool(args.oracle_policy)),
        str(bool(args.oracle_verify)),
    )

    generator = GraphGenerator(
        num_nodes=int(args.num_nodes),
        num_colors=int(args.num_colors),
        edge_prob=float(args.edge_prob),
        seed=int(args.seed),
    )

    all_stats: List[Dict[str, Any]] = []
    start_time = time.time()

    for idx in range(int(args.num_instances)):
        instance = generator.generate_planted()
        degrees = np.sum(instance.adjacency, axis=1).astype(np.int64)
        stats = solve_instance(
            model=model,
            tokenizer=tokenizer,
            adjacency=instance.adjacency,
            num_nodes=int(args.num_nodes),
            num_colors=int(args.num_colors),
            degrees=degrees,
            max_steps=int(args.max_steps),
            max_seq_len=int(eval_max_seq_len),
            max_neighbors=int(max_neighbors),
            device=device,
            cf_threshold=float(args.cf_threshold),
            no_safety_net=bool(args.no_safety_net),
            log_sample=bool(idx < 2),
            oracle_policy=bool(args.oracle_policy),
            oracle_verify=bool(args.oracle_verify),
        )
        all_stats.append(stats)

        if idx < 2:
            logger.info(
                "instance=%d solved=%s steps=%d assignments=%d backtracks=%d cf_correct=%d cf_missed=%d cf_false=%d reason=%s",
                int(idx),
                str(bool(stats["solved"])),
                int(stats["steps"]),
                int(stats["assignments"]),
                int(stats["backtracks"]),
                int(stats["verify_correct_cf"]),
                int(stats["verify_missed_cf"]),
                int(stats["verify_false_cf"]),
                str(stats["termination_reason"]),
            )

        if logger.isEnabledFor(logging.INFO) and (idx + 1) % 10 == 0:
            solved_so_far = sum(int(s["solved"]) for s in all_stats)
            logger.info(
                "processed %d/%d solve_rate=%.3f mean_backtracks=%.2f",
                int(idx + 1),
                int(args.num_instances),
                float(_safe_div(solved_so_far, len(all_stats))),
                float(np.mean([float(s["backtracks"]) for s in all_stats])),
            )

    elapsed = float(time.time() - start_time)
    metrics = _compute_metrics(all_stats)
    metrics["config"] = {
        "checkpoint": str(checkpoint_path),
        "num_instances": int(args.num_instances),
        "num_nodes": int(args.num_nodes),
        "num_colors": int(args.num_colors),
        "edge_prob": float(args.edge_prob),
        "max_steps": int(args.max_steps),
        "max_seq_len": int(eval_max_seq_len),
        "cf_threshold": float(args.cf_threshold),
        "seed": int(args.seed),
        "device": str(args.device),
        "max_neighbors": int(max_neighbors),
        "no_safety_net": bool(args.no_safety_net),
        "oracle_policy": bool(args.oracle_policy),
        "oracle_verify": bool(args.oracle_verify),
    }
    metrics["oracle_policy"] = bool(args.oracle_policy)
    metrics["oracle_verify"] = bool(args.oracle_verify)
    metrics["elapsed_sec"] = float(elapsed)

    logger.info(
        "results solve_rate=%.3f precision=%.3f recall=%.3f false_cf=%d correct_cf=%d missed_cf=%d correct_ok=%d mean_backtracks=%.2f",
        float(metrics["solve_rate"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        int(metrics["false_cf"]),
        int(metrics["correct_cf"]),
        int(metrics["missed_cf"]),
        int(metrics["correct_ok"]),
        float(metrics["mean_backtracks"]),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("wrote results to %s", str(output_path))


if __name__ == "__main__":
    main()
