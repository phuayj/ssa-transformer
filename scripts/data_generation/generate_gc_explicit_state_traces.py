#!/usr/bin/env python3
"""Generate graph-coloring traces with explicit ASSIGN/DOMAIN state encoding."""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from graph_coloring.dsl import GraphColorAction
from graph_coloring.env import GraphColorEnv, GraphColorEnvStatus, GraphColorState
from graph_coloring.generator import GraphGenerator
from graph_coloring.oracle import GraphColorOracle
from universal.cdcl_tokenizer import CDCLTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PrefixKey = Tuple[Tuple[int, int], ...]


class TokenMapper:
    """Vocabulary-size aware mapping for high-range trace tokens."""

    def __init__(self, vocab_size: int):
        self.vocab_size = int(vocab_size)
        if self.vocab_size == 394:
            self.max_nodes = 30
        elif self.vocab_size == 574:
            self.max_nodes = 75
        else:
            raise ValueError(
                f"Unsupported vocab_size={vocab_size}; expected 394 or 574"
            )

        self.max_colors = 4
        self.ASSIGN = int(CDCLTokenizer.ASSIGN)
        self.DOMAIN = int(CDCLTokenizer.DOMAIN)
        self.ASSIGN_OFFSET = 240
        self.MASK_OFFSET = int(self.ASSIGN_OFFSET + self.max_nodes * self.max_colors)
        self.STATE = int(self.MASK_OFFSET + 16)
        self.CF = int(self.MASK_OFFSET + 24)
        self.TRIED = int(self.MASK_OFFSET + 32)
        self.END_TRIED = int(self.MASK_OFFSET + 33)

    def assign_token(self, node: int, color: int) -> int:
        node_i = int(node)
        color_i = int(color)
        if not 0 <= node_i < self.max_nodes:
            raise ValueError(f"assign node out of range: {node_i}")
        if not 1 <= color_i <= self.max_colors:
            raise ValueError(f"assign color out of range: {color_i}")
        return int(self.ASSIGN_OFFSET + node_i * self.max_colors + (color_i - 1))

    def decode_assign_token(self, token: int) -> Tuple[int, int]:
        idx = int(token) - self.ASSIGN_OFFSET
        if idx < 0 or idx >= self.max_nodes * self.max_colors:
            raise ValueError(f"assign token out of range: {token}")
        node = int(idx // self.max_colors)
        color = int(idx % self.max_colors + 1)
        return node, color

    def _domain_bitmask(self, domain: Set[int]) -> int:
        mask = 0
        for color in range(1, self.max_colors + 1):
            if color in domain:
                mask |= 1 << (self.max_colors - color)
        return int(mask)

    def mask_token(self, domain: Set[int]) -> int:
        for c in domain:
            c_i = int(c)
            if not 1 <= c_i <= self.max_colors:
                raise ValueError(f"domain color out of range: {c_i}")
        return int(self.MASK_OFFSET + self._domain_bitmask(set(int(c) for c in domain)))

    def decode_mask_token(self, token: int) -> Set[int]:
        token_i = int(token)
        if token_i < self.MASK_OFFSET or token_i >= self.MASK_OFFSET + 16:
            raise ValueError(f"mask token out of range: {token_i}")
        mask = int(token_i - self.MASK_OFFSET)
        domain: Set[int] = set()
        for color in range(1, self.max_colors + 1):
            if mask & (1 << (self.max_colors - color)):
                domain.add(int(color))
        return domain


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _append_chunk(
    sequence: List[int],
    loss_mask: List[bool],
    chunk: Sequence[int],
    *,
    max_seq_len: int,
    true_positions: Set[int] | None = None,
    vocab_size: int,
) -> bool:
    chunk_list = [int(x) for x in chunk]
    if len(sequence) + len(chunk_list) > int(max_seq_len):
        return False

    for tok in chunk_list:
        if tok < 0 or tok >= int(vocab_size):
            raise AssertionError(f"token out of range [0,{vocab_size}): {tok}")

    true_positions = true_positions or set()
    sequence.extend(chunk_list)
    for i in range(len(chunk_list)):
        loss_mask.append(bool(i in true_positions))
    return True


def _is_solution(env: GraphColorEnv, state: GraphColorState) -> bool:
    if state.propagation_pending or state.selected_node is not None:
        return False
    if int(np.count_nonzero(state.assignment == 0)) != 0:
        return False
    return not env._has_contradiction(state)


def _prefix_key(state: GraphColorState) -> PrefixKey:
    return tuple(sorted((int(n), int(c)) for n, c, _ in state.assignment_stack))


def _apply_assignment(env: GraphColorEnv, node: int, color: int) -> bool:
    state = env.get_state()
    if state.propagation_pending:
        res = env.step(GraphColorAction.propagate())
        if res.done:
            return False

    if state.selected_node is None:
        res = env.step(GraphColorAction.select_node(int(node)))
        if not bool(res.info.get("valid", True)):
            return False
    elif int(state.selected_node) != int(node):
        return False

    res = env.step(GraphColorAction.assign_color(int(color)))
    if not bool(res.info.get("valid", True)):
        return False

    res = env.step(GraphColorAction.propagate())
    if not bool(res.info.get("valid", True)):
        return False
    return True


def _saturation(state: GraphColorState, node: int) -> int:
    neigh = np.nonzero(state.adjacency[int(node)])[0]
    colors = {
        int(state.assignment[int(j)])
        for j in neigh
        if int(state.assignment[int(j)]) != 0
    }
    return int(len(colors))


def _domain_order(
    env: GraphColorEnv,
    state: GraphColorState,
    *,
    depth: int,
) -> List[Tuple[int, Set[int], int, int]]:
    entries: List[Tuple[int, Set[int], int, int]] = []
    for node in range(state.num_nodes):
        if int(state.assignment[node]) != 0:
            continue
        dom = set(
            int(c) for c in env._effective_domain(state, int(node), depth=int(depth))
        )
        sat = _saturation(state, int(node))
        entries.append((int(node), dom, int(len(dom)), int(sat)))
    entries.sort(key=lambda x: (int(x[2]), -int(x[3]), int(x[0])))
    return entries


def _build_single_trace(
    *,
    adjacency: np.ndarray,
    num_nodes: int,
    num_colors: int,
    p_error: float,
    max_seq_len: int,
    rng: random.Random,
    token_mapper: TokenMapper,
    tokenizer: CDCLTokenizer,
) -> Dict[str, Any]:
    env = GraphColorEnv(
        adjacency=adjacency,
        num_colors=int(num_colors),
        solution=None,
        mode="strict",
        max_steps=int(num_nodes * num_nodes * 8),
        propagation_mode="forward_check",
    )
    oracle = GraphColorOracle(env)
    env.reset()

    sequence: List[int] = tokenizer.build_graph_prefix(adjacency, int(num_nodes))
    loss_mask: List[bool] = [False] * len(sequence)
    tried_at_prefix: Dict[PrefixKey, List[Tuple[int, int]]] = {}

    decisions = 0
    backtracks = 0
    forced_errors = 0
    max_steps = int(num_nodes * num_nodes * 8)

    for _ in range(max_steps):
        state = env.get_state()

        if _is_solution(env, state):
            _append_chunk(
                sequence,
                loss_mask,
                [tokenizer.SOLVED],
                max_seq_len=int(max_seq_len),
                vocab_size=int(token_mapper.vocab_size),
            )
            break
        if state.status != GraphColorEnvStatus.RUNNING:
            _append_chunk(
                sequence,
                loss_mask,
                [tokenizer.FAILED],
                max_seq_len=int(max_seq_len),
                vocab_size=int(token_mapper.vocab_size),
            )
            break

        prefix = _prefix_key(state)
        prior = tried_at_prefix.get(prefix, [])
        if prior:
            tried_tokens = [int(token_mapper.TRIED)]
            for n, c in prior:
                tried_tokens.extend(
                    [tokenizer.node_token(int(n)), tokenizer.color_token(int(c))]
                )
            tried_tokens.append(int(token_mapper.END_TRIED))
            if not _append_chunk(
                sequence,
                loss_mask,
                tried_tokens,
                max_seq_len=int(max_seq_len),
                vocab_size=int(token_mapper.vocab_size),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": sequence,
                    "loss_mask": loss_mask,
                    "decisions": decisions,
                    "backtracks": backtracks,
                    "forced_errors": forced_errors,
                }

        depth = int(len(state.assignment_stack) + 1)
        domain_entries = _domain_order(env, state, depth=depth)
        domain_nodes = [int(n) for n, _dom, _sz, _sat in domain_entries]

        if not domain_nodes:
            _append_chunk(
                sequence,
                loss_mask,
                [tokenizer.FAILED],
                max_seq_len=int(max_seq_len),
                vocab_size=int(token_mapper.vocab_size),
            )
            break

        min_domain_size = min(int(sz) for _n, _dom, sz, _sat in domain_entries)
        if min_domain_size == 0:
            if not _append_chunk(
                sequence,
                loss_mask,
                [int(token_mapper.CF)],
                max_seq_len=int(max_seq_len),
                true_positions={0},
                vocab_size=int(token_mapper.vocab_size),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": sequence,
                    "loss_mask": loss_mask,
                    "decisions": decisions,
                    "backtracks": backtracks,
                    "forced_errors": forced_errors,
                }

            if not state.assignment_stack:
                _append_chunk(
                    sequence,
                    loss_mask,
                    [tokenizer.FAILED],
                    max_seq_len=int(max_seq_len),
                    vocab_size=int(token_mapper.vocab_size),
                )
                break

            failed_node, failed_color, _ = state.assignment_stack[-1]
            parent_prefix = tuple(
                sorted((int(n), int(c)) for n, c, _ in state.assignment_stack[:-1])
            )
            parent = tried_at_prefix.setdefault(parent_prefix, [])
            pair = (int(failed_node), int(failed_color))
            if pair not in parent:
                parent.append(pair)
            env.backjump_to(len(state.assignment_stack) - 1)
            backtracks += 1
            continue

        selected_by_oracle = oracle._dsatur_select(state, depth=depth)
        assert selected_by_oracle is not None, (
            "oracle returned None despite positive-domain candidates"
        )
        assert int(selected_by_oracle) == int(domain_nodes[0]), (
            f"DOMAIN order mismatch: oracle={selected_by_oracle}, first={domain_nodes[0]}"
        )

        assign_pairs = sorted(
            (int(n), int(c))
            for n, c in enumerate(state.assignment.tolist())
            if int(c) != 0
        )
        state_tokens: List[int] = [int(token_mapper.STATE), int(token_mapper.ASSIGN)]
        for node_id, color_id in assign_pairs:
            tok = int(token_mapper.assign_token(int(node_id), int(color_id)))
            dec_node, dec_color = token_mapper.decode_assign_token(tok)
            assert dec_node == int(node_id) and dec_color == int(color_id), (
                f"assign roundtrip mismatch: ({node_id},{color_id}) -> {tok} -> ({dec_node},{dec_color})"
            )
            state_tokens.append(tok)
        state_tokens.append(int(tokenizer.SEP))

        state_tokens.append(int(token_mapper.DOMAIN))
        for node_id, domain, _dom_size, _sat in domain_entries:
            mask_tok = int(token_mapper.mask_token(domain))
            decoded_domain = token_mapper.decode_mask_token(mask_tok)
            assert decoded_domain == set(domain), (
                f"mask roundtrip mismatch: node={node_id}, domain={sorted(domain)}, decoded={sorted(decoded_domain)}"
            )
            state_tokens.extend([tokenizer.node_token(int(node_id)), mask_tok])
        state_tokens.append(int(tokenizer.SEP))

        if not _append_chunk(
            sequence,
            loss_mask,
            state_tokens,
            max_seq_len=int(max_seq_len),
            vocab_size=int(token_mapper.vocab_size),
        ):
            return {
                "ok": False,
                "reason": "max_seq_len",
                "sequence": sequence,
                "loss_mask": loss_mask,
                "decisions": decisions,
                "backtracks": backtracks,
                "forced_errors": forced_errors,
            }

        selected_node = int(selected_by_oracle)
        selected_domain = set(
            int(c) for c in env._effective_domain(state, selected_node, depth=depth)
        )
        if not selected_domain:
            raise AssertionError("selected oracle node has empty domain")

        oracle_color = int(min(selected_domain))
        alternatives = sorted(
            int(c) for c in selected_domain if int(c) != int(oracle_color)
        )
        use_forced_error = bool(alternatives) and (rng.random() < float(p_error))
        if use_forced_error:
            selected_color = int(rng.choice(alternatives))
            forced_errors += 1
        else:
            selected_color = int(oracle_color)

        selected_mask_token = int(token_mapper.mask_token(selected_domain))
        decoded_selected_domain = token_mapper.decode_mask_token(selected_mask_token)
        assert decoded_selected_domain == selected_domain, (
            "selected domain mask decode mismatch"
        )

        if not _append_chunk(
            sequence,
            loss_mask,
            [
                tokenizer.node_token(selected_node),
                selected_mask_token,
                tokenizer.color_token(selected_color),
            ],
            max_seq_len=int(max_seq_len),
            true_positions={0, 2},
            vocab_size=int(token_mapper.vocab_size),
        ):
            return {
                "ok": False,
                "reason": "max_seq_len",
                "sequence": sequence,
                "loss_mask": loss_mask,
                "decisions": decisions,
                "backtracks": backtracks,
                "forced_errors": forced_errors,
            }

        if not _apply_assignment(env, selected_node, selected_color):
            if not _append_chunk(
                sequence,
                loss_mask,
                [int(token_mapper.CF)],
                max_seq_len=int(max_seq_len),
                true_positions={0},
                vocab_size=int(token_mapper.vocab_size),
            ):
                return {
                    "ok": False,
                    "reason": "max_seq_len",
                    "sequence": sequence,
                    "loss_mask": loss_mask,
                    "decisions": decisions,
                    "backtracks": backtracks,
                    "forced_errors": forced_errors,
                }
            break

        decisions += 1
        if not _append_chunk(
            sequence,
            loss_mask,
            [
                tokenizer.OK,
                tokenizer.node_token(selected_node),
                tokenizer.color_token(selected_color),
            ],
            max_seq_len=int(max_seq_len),
            true_positions={0, 1, 2},
            vocab_size=int(token_mapper.vocab_size),
        ):
            return {
                "ok": False,
                "reason": "max_seq_len",
                "sequence": sequence,
                "loss_mask": loss_mask,
                "decisions": decisions,
                "backtracks": backtracks,
                "forced_errors": forced_errors,
            }

    if not sequence or sequence[-1] != int(tokenizer.EOS):
        _append_chunk(
            sequence,
            loss_mask,
            [tokenizer.EOS],
            max_seq_len=int(max_seq_len),
            vocab_size=int(token_mapper.vocab_size),
        )

    if len(sequence) != len(loss_mask):
        raise RuntimeError("sequence/loss_mask length mismatch")
    if any(
        int(tok) < 0 or int(tok) >= int(token_mapper.vocab_size) for tok in sequence
    ):
        raise RuntimeError("token out of vocabulary bounds")

    return {
        "ok": True,
        "reason": "ok",
        "sequence": sequence,
        "loss_mask": loss_mask,
        "decisions": decisions,
        "backtracks": backtracks,
        "forced_errors": forced_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GC traces with explicit ASSIGN+DOMAIN state sections",
    )
    parser.add_argument("--num-graphs", type=int, default=3000)
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.3)
    parser.add_argument("--p-error", type=float, default=0.2)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=str,
        default="experiments/gc-explicit-state-traces/traces.pkl",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=394,
        choices=[394, 574],
        help="394 for n<=30 models, 574 for n<=75 models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    rng = random.Random(int(args.seed))

    token_mapper = TokenMapper(vocab_size=int(args.vocab_size))
    if int(args.num_nodes) > int(token_mapper.max_nodes):
        raise ValueError(
            f"num_nodes={args.num_nodes} exceeds mapper max_nodes={token_mapper.max_nodes} for vocab_size={args.vocab_size}"
        )
    if int(args.num_colors) > 4:
        raise ValueError("num_colors must be <= 4 for this tokenizer")

    tokenizer = CDCLTokenizer()
    generator = GraphGenerator(
        num_nodes=int(args.num_nodes),
        num_colors=int(args.num_colors),
        edge_prob=float(args.edge_prob),
        seed=int(args.seed),
    )

    records: List[Dict[str, Any]] = []
    lengths: List[int] = []
    decisions: List[int] = []
    backtracks: List[int] = []
    forced_errors: List[int] = []
    filtered = 0

    for i in range(int(args.num_graphs)):
        instance = generator.generate_planted()
        sample = _build_single_trace(
            adjacency=instance.adjacency,
            num_nodes=int(args.num_nodes),
            num_colors=int(args.num_colors),
            p_error=float(args.p_error),
            max_seq_len=int(args.max_seq_len),
            rng=rng,
            token_mapper=token_mapper,
            tokenizer=tokenizer,
        )

        if not bool(sample["ok"]):
            filtered += 1
            continue

        seq = [int(x) for x in sample["sequence"]]
        lm = [bool(x) for x in sample["loss_mask"]]
        if len(seq) != len(lm):
            raise RuntimeError("sequence/loss_mask mismatch")
        if len(seq) > int(args.max_seq_len):
            filtered += 1
            continue

        records.append({"sequence": seq, "loss_mask": lm})
        lengths.append(len(seq))
        decisions.append(int(sample["decisions"]))
        backtracks.append(int(sample["backtracks"]))
        forced_errors.append(int(sample["forced_errors"]))

        logger.info(
            "trace=%d len=%d decisions=%d backtracks=%d forced_errors=%d",
            int(i + 1),
            int(len(seq)),
            int(sample["decisions"]),
            int(sample["backtracks"]),
            int(sample["forced_errors"]),
        )

        if (i + 1) % 100 == 0:
            logger.info(
                "processed=%d/%d kept=%d filtered=%d mean_len=%.1f mean_decisions=%.2f mean_backtracks=%.2f mean_forced_errors=%.2f",
                int(i + 1),
                int(args.num_graphs),
                int(len(records)),
                int(filtered),
                float(np.mean(lengths)) if lengths else 0.0,
                float(np.mean(decisions)) if decisions else 0.0,
                float(np.mean(backtracks)) if backtracks else 0.0,
                float(np.mean(forced_errors)) if forced_errors else 0.0,
            )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(records, f)

    logger.info(
        "done kept=%d filtered=%d mean_len=%.1f max_len=%d mean_decisions=%.2f mean_backtracks=%.2f mean_forced_errors=%.2f",
        int(len(records)),
        int(filtered),
        float(np.mean(lengths)) if lengths else 0.0,
        int(np.max(lengths)) if lengths else 0,
        float(np.mean(decisions)) if decisions else 0.0,
        float(np.mean(backtracks)) if backtracks else 0.0,
        float(np.mean(forced_errors)) if forced_errors else 0.0,
    )


if __name__ == "__main__":
    main()
