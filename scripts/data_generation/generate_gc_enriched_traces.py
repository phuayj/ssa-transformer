#!/usr/bin/env python3
"""Generate graph-coloring traces with LOCAL_LEGAL sections (maskless-compatible)."""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graph_coloring.dsl import GraphColorAction
from graph_coloring.env import GraphColorEnv, GraphColorEnvStatus, GraphColorState
from graph_coloring.generator import GraphGenerator
from graph_coloring.oracle import GraphColorOracle
from universal.cdcl_tokenizer import CDCLTokenizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PrefixKey = Tuple[Tuple[int, int], ...]
LOCAL_LEGAL = 239
TRIED_VOCAB_SIZE = 574


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _append_with_block(
    tokens: List[int],
    block_ids: List[int],
    chunk: List[int],
    max_seq_len: int,
    block_id: int,
) -> bool:
    if len(tokens) + len(chunk) > int(max_seq_len):
        return False
    tokens.extend(int(x) for x in chunk)
    block_ids.extend([int(block_id)] * int(len(chunk)))
    return True


def _prefix_key_from_stack(stack: List[Tuple[int, int, List[set[int]]]]) -> PrefixKey:
    return tuple(sorted((int(n), int(c)) for n, c, _ in stack))


def _is_solution(env: GraphColorEnv, state: GraphColorState) -> bool:
    if state.propagation_pending or state.selected_node is not None:
        return False
    if int(np.count_nonzero(state.assignment == 0)) != 0:
        return False
    return not env._has_contradiction(state)


def _sorted_candidates(state: GraphColorState, degrees: np.ndarray) -> List[int]:
    unassigned = [
        int(i) for i in range(state.num_nodes) if int(state.assignment[i]) == 0
    ]
    return sorted(
        unassigned,
        key=lambda nd: (len(state.domains[int(nd)]), -int(degrees[int(nd)])),
    )


def _local_legal(
    adjacency: np.ndarray, assignment: np.ndarray, node: int, num_colors: int
) -> List[int]:
    """Colors not used by any assigned neighbor of `node`."""
    neighbor_colors = set()
    for nb in range(len(adjacency)):
        if adjacency[node][nb] and assignment[nb] != 0:
            neighbor_colors.add(int(assignment[nb]))
    return sorted(c for c in range(1, num_colors + 1) if c not in neighbor_colors)


def _apply_assignment(env: GraphColorEnv, node: int, color: int) -> None:
    state = env.get_state()
    if state.propagation_pending:
        res = env.step(GraphColorAction.propagate())
        if res.done:
            raise RuntimeError("env terminated during pending propagate")

    if state.selected_node is None:
        res = env.step(GraphColorAction.select_node(int(node)))
        if not bool(res.info.get("valid", True)):
            raise RuntimeError(f"invalid select: {res.info}")
    elif int(state.selected_node) != int(node):
        raise RuntimeError("selected node mismatch")

    res = env.step(GraphColorAction.assign_color(int(color)))
    if not bool(res.info.get("valid", True)):
        raise RuntimeError(f"invalid assign: {res.info}")

    res = env.step(GraphColorAction.propagate())
    if not bool(res.info.get("valid", True)):
        raise RuntimeError(f"invalid propagate: {res.info}")


def _build_single_trace_core(
    *,
    adjacency: np.ndarray,
    num_nodes: int,
    num_colors: int,
    p_error: float,
    max_steps: int,
    max_seq_len: int,
    rng: random.Random,
    tokenizer: CDCLTokenizer,
    emit_local_legal: bool,
) -> Dict[str, Any]:
    env = GraphColorEnv(
        adjacency=adjacency,
        num_colors=int(num_colors),
        solution=None,
        mode="strict",
        max_steps=int(max_steps * 4 + 10),
        propagation_mode="forward_check",
    )
    oracle = GraphColorOracle(env)
    env.reset()
    degrees = np.sum(adjacency, axis=1).astype(np.int64)

    tokens: List[int] = tokenizer.build_graph_prefix(adjacency, int(num_nodes))
    block_ids: List[int] = [0] * len(tokens)
    current_block = 0
    tried_at_prefix: Dict[PrefixKey, List[Tuple[int, int]]] = {}

    backtracks = 0
    decisions = 0
    forced_errors = 0
    decision_pairs: List[Tuple[int, int]] = []
    local_legal_sizes: List[int] = []

    for _step in range(int(max_steps)):
        state = env.get_state()

        if _is_solution(env, state):
            _append_with_block(
                tokens,
                block_ids,
                [tokenizer.SOLVED],
                int(max_seq_len),
                int(current_block),
            )
            break
        if state.status != GraphColorEnvStatus.RUNNING:
            _append_with_block(
                tokens,
                block_ids,
                [tokenizer.FAILED],
                int(max_seq_len),
                int(current_block),
            )
            break

        # Structural-domain traces intentionally ignore current-depth nogoods.
        if env._state is None:
            raise RuntimeError("environment state is None")
        depth = int(len(env._state.assignment_stack) + 1)
        env._state.nogoods.pop(int(depth), None)
        state = env.get_state()

        current_block += 1
        prefix_key = _prefix_key_from_stack(state.assignment_stack)
        prior = tried_at_prefix.get(prefix_key, [])
        if prior:
            tried_tokens = [tokenizer.TRIED]
            for node_id, color_id in prior:
                tried_tokens.append(tokenizer.node_token(int(node_id)))
                tried_tokens.append(tokenizer.color_token(int(color_id)))
            tried_tokens.append(tokenizer.END_TRIED)
            if not _append_with_block(
                tokens,
                block_ids,
                tried_tokens,
                int(max_seq_len),
                int(current_block),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": tokens,
                    "block_ids": block_ids,
                    "loss_mask": [],
                    "backtracks": backtracks,
                    "decisions": decisions,
                    "forced_errors": forced_errors,
                    "decision_pairs": decision_pairs,
                    "local_legal_sizes": local_legal_sizes,
                }

        sorted_nodes = _sorted_candidates(state, degrees)
        if not sorted_nodes:
            _append_with_block(
                tokens,
                block_ids,
                [tokenizer.FAILED],
                int(max_seq_len),
                int(current_block),
            )
            break

        structural_domains = {
            int(nd): set(int(c) for c in state.domains[int(nd)]) for nd in sorted_nodes
        }
        min_domain_size = min(
            int(len(structural_domains[int(nd)])) for nd in sorted_nodes
        )
        if min_domain_size == 0:
            if not _append_with_block(
                tokens,
                block_ids,
                [tokenizer.CF],
                int(max_seq_len),
                int(current_block),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": tokens,
                    "block_ids": block_ids,
                    "loss_mask": [],
                    "backtracks": backtracks,
                    "decisions": decisions,
                    "forced_errors": forced_errors,
                    "decision_pairs": decision_pairs,
                    "local_legal_sizes": local_legal_sizes,
                }
            if not state.assignment_stack:
                _append_with_block(
                    tokens,
                    block_ids,
                    [tokenizer.FAILED],
                    int(max_seq_len),
                    int(current_block),
                )
                break

            failed_node, failed_color, _ = state.assignment_stack[-1]
            parent_prefix = tuple(
                sorted((int(n), int(c)) for n, c, _ in state.assignment_stack[:-1])
            )
            parent_tried = tried_at_prefix.setdefault(parent_prefix, [])
            pair = (int(failed_node), int(failed_color))
            if pair not in parent_tried:
                parent_tried.append(pair)

            env.backjump_to(len(state.assignment_stack) - 1)
            backtracks += 1
            continue

        state_tokens = [tokenizer.STATE]
        state_tokens.extend(tokenizer.node_token(int(nd)) for nd in sorted_nodes)
        state_tokens.append(tokenizer.SEP)
        if not _append_with_block(
            tokens,
            block_ids,
            state_tokens,
            int(max_seq_len),
            int(current_block),
        ):
            return {
                "ok": False,
                "reason": "max_seq_len",
                "sequence": tokens,
                "block_ids": block_ids,
                "loss_mask": [],
                "backtracks": backtracks,
                "decisions": decisions,
                "forced_errors": forced_errors,
                "decision_pairs": decision_pairs,
                "local_legal_sizes": local_legal_sizes,
            }

        depth = int(len(state.assignment_stack) + 1)
        selected_node = oracle._dsatur_select(state, depth=depth)
        if selected_node is None:
            if not _append_with_block(
                tokens,
                block_ids,
                [tokenizer.CF],
                int(max_seq_len),
                int(current_block),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": tokens,
                    "block_ids": block_ids,
                    "loss_mask": [],
                    "backtracks": backtracks,
                    "decisions": decisions,
                    "forced_errors": forced_errors,
                    "decision_pairs": decision_pairs,
                    "local_legal_sizes": local_legal_sizes,
                }
            if not state.assignment_stack:
                _append_with_block(
                    tokens,
                    block_ids,
                    [tokenizer.FAILED],
                    int(max_seq_len),
                    int(current_block),
                )
                break
            env.backjump_to(len(state.assignment_stack) - 1)
            backtracks += 1
            continue

        selected_node = int(selected_node)
        structural_domain = set(int(c) for c in state.domains[int(selected_node)])

        if emit_local_legal:
            legal_colors = _local_legal(
                adjacency=adjacency,
                assignment=state.assignment,
                node=selected_node,
                num_colors=int(num_colors),
            )
            local_legal_tokens = [LOCAL_LEGAL]
            local_legal_tokens.extend(
                tokenizer.color_token(int(c)) for c in legal_colors
            )
            local_legal_tokens.append(tokenizer.SEP)
            if not _append_with_block(
                tokens,
                block_ids,
                local_legal_tokens,
                int(max_seq_len),
                int(current_block),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": tokens,
                    "block_ids": block_ids,
                    "loss_mask": [],
                    "backtracks": backtracks,
                    "decisions": decisions,
                    "forced_errors": forced_errors,
                    "decision_pairs": decision_pairs,
                    "local_legal_sizes": local_legal_sizes,
                }
            local_legal_sizes.append(len(legal_colors))

        if not _append_with_block(
            tokens,
            block_ids,
            [tokenizer.OK],
            int(max_seq_len),
            int(current_block),
        ):
            return {
                "ok": False,
                "reason": "max_seq_len",
                "sequence": tokens,
                "block_ids": block_ids,
                "loss_mask": [],
                "backtracks": backtracks,
                "decisions": decisions,
                "forced_errors": forced_errors,
                "decision_pairs": decision_pairs,
                "local_legal_sizes": local_legal_sizes,
            }

        if not _append_with_block(
            tokens,
            block_ids,
            [tokenizer.node_token(selected_node)],
            int(max_seq_len),
            int(current_block),
        ):
            return {
                "ok": False,
                "reason": "max_seq_len",
                "sequence": tokens,
                "block_ids": block_ids,
                "loss_mask": [],
                "backtracks": backtracks,
                "decisions": decisions,
                "forced_errors": forced_errors,
                "decision_pairs": decision_pairs,
                "local_legal_sizes": local_legal_sizes,
            }

        if not structural_domain:
            if not _append_with_block(
                tokens,
                block_ids,
                [tokenizer.CF],
                int(max_seq_len),
                int(current_block),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": tokens,
                    "block_ids": block_ids,
                    "loss_mask": [],
                    "backtracks": backtracks,
                    "decisions": decisions,
                    "forced_errors": forced_errors,
                    "decision_pairs": decision_pairs,
                    "local_legal_sizes": local_legal_sizes,
                }
            continue

        tried_colors_here = set(
            int(color)
            for node, color in tried_at_prefix.get(prefix_key, [])
            if int(node) == int(selected_node)
        )
        available_colors = structural_domain - tried_colors_here
        if not available_colors:
            if not _append_with_block(
                tokens,
                block_ids,
                [tokenizer.CF],
                int(max_seq_len),
                int(current_block),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": tokens,
                    "block_ids": block_ids,
                    "loss_mask": [],
                    "backtracks": backtracks,
                    "decisions": decisions,
                    "forced_errors": forced_errors,
                    "decision_pairs": decision_pairs,
                    "local_legal_sizes": local_legal_sizes,
                }
            if not state.assignment_stack:
                _append_with_block(
                    tokens,
                    block_ids,
                    [tokenizer.FAILED],
                    int(max_seq_len),
                    int(current_block),
                )
                break

            failed_node, failed_color, _ = state.assignment_stack[-1]
            parent_prefix = tuple(
                sorted((int(n), int(c)) for n, c, _ in state.assignment_stack[:-1])
            )
            parent_tried = tried_at_prefix.setdefault(parent_prefix, [])
            pair = (int(failed_node), int(failed_color))
            if pair not in parent_tried:
                parent_tried.append(pair)

            env.backjump_to(len(state.assignment_stack) - 1)
            backtracks += 1
            continue

        oracle_color = int(min(available_colors))
        alternatives = sorted(
            int(c) for c in available_colors if int(c) != int(oracle_color)
        )
        use_forced_error = bool(alternatives) and (rng.random() < float(p_error))
        if use_forced_error:
            chosen_color = int(rng.choice(alternatives))
            forced_errors += 1
        else:
            chosen_color = int(oracle_color)

        if not _append_with_block(
            tokens,
            block_ids,
            [tokenizer.color_token(chosen_color)],
            int(max_seq_len),
            int(current_block),
        ):
            return {
                "ok": False,
                "reason": "max_seq_len",
                "sequence": tokens,
                "block_ids": block_ids,
                "loss_mask": [],
                "backtracks": backtracks,
                "decisions": decisions,
                "forced_errors": forced_errors,
                "decision_pairs": decision_pairs,
                "local_legal_sizes": local_legal_sizes,
            }

        _apply_assignment(env, selected_node, chosen_color)
        decisions += 1
        decision_pairs.append((selected_node, chosen_color))

        if not _append_with_block(
            tokens,
            block_ids,
            [
                tokenizer.OK,
                tokenizer.node_token(selected_node),
                tokenizer.color_token(chosen_color),
            ],
            int(max_seq_len),
            int(current_block),
        ):
            return {
                "ok": False,
                "reason": "max_seq_len",
                "sequence": tokens,
                "block_ids": block_ids,
                "loss_mask": [],
                "backtracks": backtracks,
                "decisions": decisions,
                "forced_errors": forced_errors,
                "decision_pairs": decision_pairs,
                "local_legal_sizes": local_legal_sizes,
            }

    if not tokens or tokens[-1] != int(tokenizer.EOS):
        _append_with_block(
            tokens,
            block_ids,
            [tokenizer.EOS],
            int(max_seq_len),
            int(current_block),
        )

    if len(tokens) != len(block_ids):
        raise RuntimeError("sequence/block_ids length mismatch")

    loss_mask = [True] * len(tokens)
    if loss_mask:
        loss_mask[0] = False
        loss_mask[-1] = False

    return {
        "ok": True,
        "reason": "ok",
        "sequence": tokens,
        "block_ids": block_ids,
        "loss_mask": loss_mask,
        "backtracks": backtracks,
        "decisions": decisions,
        "forced_errors": forced_errors,
        "decision_pairs": decision_pairs,
        "local_legal_sizes": local_legal_sizes,
        "num_blocks": int(max(block_ids) + 1) if block_ids else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GC enriched traces with LOCAL_LEGAL sections"
    )
    parser.add_argument("--num-graphs", type=int, default=1000)
    parser.add_argument("--num-traces-per-graph", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=4)
    parser.add_argument("--edge-prob", type=float, default=0.35)
    parser.add_argument("--p-error", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/gc-enriched-traces/traces.pkl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    tokenizer = CDCLTokenizer()
    generator = GraphGenerator(
        num_nodes=int(args.num_nodes),
        num_colors=int(args.num_colors),
        edge_prob=float(args.edge_prob),
        seed=int(args.seed),
    )

    examples: List[Dict[str, object]] = []
    trace_lengths: List[int] = []
    block_counts: List[int] = []
    tokens_per_block: List[float] = []
    backtracks: List[int] = []
    decisions: List[int] = []
    forced_errors: List[int] = []
    local_legal_sizes_all: List[int] = []
    filtered = 0
    kept = 0
    total_requested = int(args.num_graphs) * int(args.num_traces_per_graph)

    for graph_idx in range(int(args.num_graphs)):
        adjacency = generator.generate_planted().adjacency
        for trace_idx in range(int(args.num_traces_per_graph)):
            sample = _build_single_trace_core(
                adjacency=adjacency,
                num_nodes=int(args.num_nodes),
                num_colors=int(args.num_colors),
                p_error=float(args.p_error),
                max_steps=int(args.max_steps),
                max_seq_len=int(args.max_seq_len),
                rng=rng,
                tokenizer=tokenizer,
                emit_local_legal=True,
            )

            if not bool(sample["ok"]):
                filtered += 1
                continue

            seq = list(sample["sequence"])
            blk = list(sample["block_ids"])
            lm = list(sample["loss_mask"])
            if len(seq) != len(lm):
                raise RuntimeError("sequence/loss_mask length mismatch")
            if len(seq) != len(blk):
                raise RuntimeError("sequence/block_ids length mismatch")
            if len(seq) > int(args.max_seq_len):
                filtered += 1
                continue
            if any(int(tok) < 0 or int(tok) >= int(TRIED_VOCAB_SIZE) for tok in seq):
                raise RuntimeError(
                    "token out of vocabulary bounds for TRIED_VOCAB_SIZE=574"
                )

            examples.append({"sequence": seq, "block_ids": blk, "loss_mask": lm})
            trace_lengths.append(len(seq))
            block_count = int(max(blk) + 1) if blk else 0
            block_counts.append(block_count)
            if block_count > 0:
                tokens_per_block.append(float(len(seq)) / float(block_count))
            backtracks.append(int(sample["backtracks"]))
            decisions.append(int(sample["decisions"]))
            forced_errors.append(int(sample["forced_errors"]))
            local_legal_sizes_all.extend(int(x) for x in sample["local_legal_sizes"])
            kept += 1

            if kept % 100 == 0:
                logger.info(
                    "processed=%d/%d kept=%d filtered=%d mean_len=%.1f max_len=%d mean_blocks=%.2f mean_tokens_per_block=%.2f mean_local_legal=%.2f",
                    int(kept + filtered),
                    int(total_requested),
                    int(kept),
                    int(filtered),
                    float(np.mean(trace_lengths)) if trace_lengths else 0.0,
                    int(np.max(trace_lengths)) if trace_lengths else 0,
                    float(np.mean(block_counts)) if block_counts else 0.0,
                    float(np.mean(tokens_per_block)) if tokens_per_block else 0.0,
                    float(np.mean(local_legal_sizes_all))
                    if local_legal_sizes_all
                    else 0.0,
                )

            logger.info(
                "graph=%d trace_in_graph=%d len=%d blocks=%d decisions=%d backtracks=%d forced_errors=%d mean_local_legal_so_far=%.3f",
                int(graph_idx + 1),
                int(trace_idx + 1),
                int(len(seq)),
                int(block_count),
                int(sample["decisions"]),
                int(sample["backtracks"]),
                int(sample["forced_errors"]),
                float(np.mean(local_legal_sizes_all)) if local_legal_sizes_all else 0.0,
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(examples, f)

    counts = {k: 0 for k in range(int(args.num_colors) + 1)}
    for size in local_legal_sizes_all:
        if 0 <= int(size) <= int(args.num_colors):
            counts[int(size)] += 1

    logger.info(
        "done traces=%d filtered=%d mean_len=%.1f max_len=%d mean_blocks=%.2f mean_tokens_per_block=%.2f mean_backtracks=%.2f mean_decisions=%.2f mean_forced_errors=%.2f mean_local_legal=%.3f local_legal_hist=%s vocab_size_check=%d",
        int(len(examples)),
        int(filtered),
        float(np.mean(trace_lengths)) if trace_lengths else 0.0,
        int(np.max(trace_lengths)) if trace_lengths else 0,
        float(np.mean(block_counts)) if block_counts else 0.0,
        float(np.mean(tokens_per_block)) if tokens_per_block else 0.0,
        float(np.mean(backtracks)) if backtracks else 0.0,
        float(np.mean(decisions)) if decisions else 0.0,
        float(np.mean(forced_errors)) if forced_errors else 0.0,
        float(np.mean(local_legal_sizes_all)) if local_legal_sizes_all else 0.0,
        counts,
        int(TRIED_VOCAB_SIZE),
    )


if __name__ == "__main__":
    main()
