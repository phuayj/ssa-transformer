#!/usr/bin/env python3
"""Generate graph-coloring traces with structural domains and TRIED markers."""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _append(tokens: List[int], chunk: Sequence[int], max_seq_len: int) -> bool:
    if len(tokens) + len(chunk) > int(max_seq_len):
        return False
    tokens.extend(int(x) for x in chunk)
    return True


def _append_with_block(
    tokens: List[int],
    block_ids: List[int],
    chunk: Sequence[int],
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


def _build_single_trace(
    *,
    adjacency: np.ndarray,
    num_nodes: int,
    num_colors: int,
    p_error: float,
    max_steps: int,
    max_seq_len: int,
    rng: random.Random,
    tokenizer: CDCLTokenizer,
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
            }

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
            }

        depth = int(len(state.assignment_stack) + 1)
        selected_node = oracle._dsatur_select(state, depth=depth)
        if selected_node is None:
            _append_with_block(
                tokens,
                block_ids,
                [tokenizer.CF],
                int(max_seq_len),
                int(current_block),
            )
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
            }

        structural_domain = set(int(c) for c in state.domains[int(selected_node)])
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
                }
            continue

        if not _append_with_block(
            tokens,
            block_ids,
            [tokenizer.mask_token(structural_domain)],
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
            }

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
            }

        _apply_assignment(env, selected_node, chosen_color)
        decisions += 1

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

    # Standard causal next-token objective over full sequence.
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GC traces with structural domains and TRIED markers"
    )
    parser.add_argument("--num_instances", type=int, default=5000)
    parser.add_argument("--num_nodes", type=int, default=30)
    parser.add_argument("--num_colors", type=int, default=4)
    parser.add_argument("--edge_prob", type=float, default=0.35)
    parser.add_argument("--p_error", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/gc-structural-traces/traces.pkl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    backtracks: List[int] = []
    decisions: List[int] = []
    forced_errors: List[int] = []
    filtered = 0

    for idx in range(int(args.num_instances)):
        adjacency = generator.generate_planted().adjacency
        sample: Dict[str, Any] = _build_single_trace(
            adjacency=adjacency,
            num_nodes=int(args.num_nodes),
            num_colors=int(args.num_colors),
            p_error=float(args.p_error),
            max_steps=int(args.max_steps),
            max_seq_len=int(args.max_seq_len),
            rng=rng,
            tokenizer=tokenizer,
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

        examples.append({"sequence": seq, "block_ids": blk, "loss_mask": lm})
        trace_lengths.append(len(seq))
        backtracks.append(int(sample["backtracks"]))
        decisions.append(int(sample["decisions"]))
        forced_errors.append(int(sample["forced_errors"]))

        if (idx + 1) % 100 == 0:
            logger.info(
                "processed=%d/%d kept=%d filtered=%d mean_len=%.1f mean_backtracks=%.2f mean_decisions=%.2f mean_forced_errors=%.2f",
                int(idx + 1),
                int(args.num_instances),
                int(len(examples)),
                int(filtered),
                float(np.mean(trace_lengths)) if trace_lengths else 0.0,
                float(np.mean(backtracks)) if backtracks else 0.0,
                float(np.mean(decisions)) if decisions else 0.0,
                float(np.mean(forced_errors)) if forced_errors else 0.0,
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(examples, f)

    logger.info(
        "done examples=%d filtered=%d mean_len=%.1f max_len=%d mean_backtracks=%.2f mean_decisions=%.2f mean_forced_errors=%.2f target_backtracks_range=2-5",
        int(len(examples)),
        int(filtered),
        float(np.mean(trace_lengths)) if trace_lengths else 0.0,
        int(np.max(trace_lengths)) if trace_lengths else 0,
        float(np.mean(backtracks)) if backtracks else 0.0,
        float(np.mean(decisions)) if decisions else 0.0,
        float(np.mean(forced_errors)) if forced_errors else 0.0,
    )


if __name__ == "__main__":
    main()
