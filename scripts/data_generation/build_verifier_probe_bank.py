#!/usr/bin/env python3
"""Build model-free verifier-only state-equivalence probe banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LAST_DEPTH_QS: List[float] = [0.0, 0.0, 0.0]
LAST_SIZE_QS: List[float] = [0.0, 0.0, 0.0]


def _quartiles(values: Sequence[int]) -> List[float]:
    if not values:
        return [0.0, 0.0, 0.0]
    return [float(x) for x in np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])]


def _quartile(value: int, thresholds: Sequence[float]) -> int:
    return int(sum(float(value) > float(t) for t in thresholds) + 1)


def _state_start(tokens: Sequence[int], block_ids: Sequence[int], sep_pos: int, state_tok: int) -> int:
    j = int(sep_pos)
    while j >= 0 and int(block_ids[j]) == int(block_ids[sep_pos]):
        if int(tokens[j]) == int(state_tok):
            return int(j)
        j -= 1
    raise RuntimeError("could not find STATE token before SEP")


def _make_history(trace: Any, sep_pos: int, state_start: int, prefix_len: int) -> Dict[str, Any]:
    return {
        "trace_idx": int(trace._trace_idx),
        "prefix_tokens": [int(x) for x in trace.tokens[: int(sep_pos) + 1]],
        "prefix_block_ids": [int(x) for x in trace.block_ids[: int(sep_pos) + 1]],
        "history_token_length": int(state_start) - int(prefix_len),
        "current_block_start": int(state_start),
        "current_block_sep_position": int(sep_pos),
        "decision_block_id": int(trace.block_ids[int(sep_pos)]),
    }


def _state_block_tokens(history: Dict[str, Any]) -> Tuple[int, ...]:
    tokens = [int(x) for x in history["prefix_tokens"]]
    start = int(history["current_block_start"])
    sep = int(history["current_block_sep_position"])
    if not (0 <= start <= sep < len(tokens)):
        raise RuntimeError(
            f"invalid state block bounds start={start} sep={sep} len={len(tokens)}"
        )
    return tuple(tokens[start : sep + 1])


def _select_histories(matches: Sequence[Dict[str, Any]], max_histories: int) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen_keys = set()
    for item in sorted(
        matches,
        key=lambda m: (
            int(m["history"]["trace_idx"]),
            int(m["history"]["history_token_length"]),
            int(m["history"]["current_block_start"]),
            int(m["history"]["current_block_sep_position"]),
        ),
    ):
        hist = item["history"]
        key = (
            int(hist["trace_idx"]),
            int(hist["current_block_start"]),
            int(hist["current_block_sep_position"]),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(item)

    selected: List[Dict[str, Any]] = []
    used_lengths = set()
    for item in unique:
        hist_len = int(item["history"].get("history_token_length", 0))
        if hist_len in used_lengths:
            continue
        selected.append(item)
        used_lengths.add(hist_len)
        if len(selected) >= int(max_histories):
            return selected

    for item in unique:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= int(max_histories):
            break
    return selected


def _sat_allowed_from_state_tokens(tokens: Sequence[int], num_vars: int) -> List[Dict[str, int]]:
    from sat.interleaved_tokenizer import SATInterleavedTokenizer

    tok = SATInterleavedTokenizer()
    out: List[Dict[str, int]] = []
    i = 1
    while i + 1 < len(tokens) and int(tokens[i]) != int(tok.SEP):
        var_token = int(tokens[i])
        dom_token = int(tokens[i + 1])
        var = int(var_token - tok.VAR_OFFSET)
        if 0 <= var < int(num_vars):
            vals = []
            if dom_token == int(tok.UNASSIGNED):
                vals = [-1, 1]
            elif dom_token == int(tok.TRUE_VAL):
                vals = [1]
            elif dom_token == int(tok.FALSE_VAL):
                vals = [-1]
            out.extend({"var": int(var), "value": int(v)} for v in vals)
        i += 2
    return out


def _scan_sat_conflicts(trace: Any, num_vars: int, prefix_len: int) -> List[Dict[str, Any]]:
    from sat.interleaved_tokenizer import SATInterleavedTokenizer

    tok = SATInterleavedTokenizer()
    points: List[Dict[str, Any]] = []
    for i in range(1, len(trace.tokens)):
        if int(trace.tokens[i]) == int(tok.CONFLICT) and int(trace.tokens[i - 1]) == int(tok.SEP):
            sep = int(i - 1)
            start = _state_start(trace.tokens, trace.block_ids, sep, int(tok.STATE))
            state_tokens = [int(x) for x in trace.tokens[start : sep + 1]]
            points.append(
                {
                    "signature": repr(("conflict", tuple(state_tokens))),
                    "label": "exposed_conflict",
                    "depth": int(trace.block_ids[sep]),
                    "size": sum(1 for x in state_tokens if int(tok.VAR_OFFSET) <= int(x) < int(tok.VOCAB_SIZE)),
                    "allowed": _sat_allowed_from_state_tokens(state_tokens, int(num_vars)),
                    "history": _make_history(trace, sep, start, int(prefix_len)),
                }
            )
    return points


def _collect_sat(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from sat.generator import SatGenerator
    from sat.interleaved_tokenizer import SATInterleavedTokenizer
    from sat.transplant_trace import generate_oracle_trace_with_random_ties as sat_trace

    generator = SatGenerator(seed=int(args.seed))
    by_state: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for instance_idx in range(int(args.n_instances)):
        inst = generator.generate_planted(num_vars=int(args.num_vars), alpha=float(args.alpha))
        clauses = [tuple(int(x) for x in clause) for clause in inst.clauses]
        planted = None if inst.planted_solution is None else np.asarray(inst.planted_solution, dtype=np.int64)
        for trace_idx in range(int(args.n_traces_per_instance)):
            tie_seed = int(args.seed) + instance_idx * 10007 + trace_idx * 97
            tr = sat_trace(clauses=clauses, num_vars=int(args.num_vars), planted_solution=planted, max_seq_len=int(args.max_seq_len), max_steps=int(args.num_vars) * 16 + 32, tie_seed=tie_seed, state_sort=str(args.state_sort))
            tr._trace_idx = int(trace_idx)
            for dp in tr.decision_points:
                start = _state_start(tr.tokens, tr.block_ids, int(dp.position), SATInterleavedTokenizer.STATE)
                by_state[(instance_idx, repr(dp.canonical_state))].append({
                    "signature": repr(dp.canonical_state), "label": "viable", "depth": int(dp.decision_level),
                    "size": int(sum(1 for v in dp.assignment if int(v) == 0)),
                    "allowed": [{"var": int(v), "value": int(val)} for v, val in dp.action_candidates],
                    "history": _make_history(tr, int(dp.position), int(start), int(tr.clause_prefix_len)),
                })
            for point in _scan_sat_conflicts(tr, int(args.num_vars), int(tr.clause_prefix_len)):
                by_state[(instance_idx, point["signature"])].append(point)
        if (instance_idx + 1) % 10 == 0:
            logger.info("sat instances=%d states_indexed=%d", instance_idx + 1, len(by_state))
    return _materialize(by_state, args)


def _gc_allowed(dp: Any, num_colors: int) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for node, dom in enumerate(dp.domains):
        if int(dp.assignment[node]) != 0:
            continue
        vals = [int(c) for c in dom if 1 <= int(c) <= int(num_colors)]
        out.extend({"var": int(node), "value": int(c)} for c in vals)
    return out


def _scan_gc_conflicts(trace: Any, token_mapper: Any, prefix_len: int, num_colors: int) -> List[Dict[str, Any]]:
    from universal.cdcl_tokenizer import CDCLTokenizer

    tok = CDCLTokenizer()
    points: List[Dict[str, Any]] = []
    seen_seps: set[int] = set()
    for i, token in enumerate(trace.tokens):
        if int(token) != int(token_mapper.CF):
            continue

        cf_block_id = int(trace.block_ids[int(i)])

        start = int(i)
        while start >= 0 and int(trace.block_ids[start]) == cf_block_id:
            if int(trace.tokens[start]) == int(token_mapper.STATE):
                break
            start -= 1
        if start < 0 or int(trace.block_ids[start]) != cf_block_id:
            logger.debug("skip gc CF at pos=%d block=%d: no STATE in block", int(i), cf_block_id)
            continue

        sep = start + 1
        while sep < len(trace.tokens) and int(trace.block_ids[sep]) == cf_block_id:
            if int(trace.tokens[sep]) == int(tok.SEP):
                break
            sep += 1
        if sep >= len(trace.tokens) or int(trace.block_ids[sep]) != cf_block_id:
            logger.debug("skip gc CF at pos=%d block=%d: no STATE SEP", int(i), cf_block_id)
            continue
        if int(sep) >= int(i):
            logger.debug("skip gc CF at pos=%d block=%d: CF not after STATE SEP", int(i), cf_block_id)
            continue
        if int(sep) in seen_seps:
            continue
        seen_seps.add(int(sep))

        state_tokens = tuple(int(x) for x in trace.tokens[int(start) : int(sep) + 1])
        nodes = [
            int(x) - int(tok.NODE_OFFSET)
            for x in state_tokens
            if int(tok.NODE_OFFSET) <= int(x) < int(tok.NODE_OFFSET + tok.MAX_NODES)
        ]
        allowed = [
            {"var": int(n), "value": int(c)}
            for n in nodes
            for c in range(1, int(num_colors) + 1)
        ]
        points.append(
            {
                "signature": repr(("conflict_state", state_tokens)),
                "label": "exposed_conflict",
                "depth": int(trace.block_ids[int(sep)]),
                "size": len(nodes),
                "allowed": allowed,
                "history": _make_history(trace, int(sep), int(start), int(prefix_len)),
            }
        )
    return points


def _collect_gc(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from graph_coloring.generator import GraphGenerator
    from graph_coloring.transplant_trace import TokenMapper as GCTokenMapper
    from graph_coloring.transplant_trace import generate_oracle_trace_with_random_ties as gc_trace

    num_nodes = int(args.num_nodes)
    vocab_size = int(args.vocab_size) if int(args.vocab_size) > 0 else (394 if num_nodes <= 30 else 574)
    logger.info("gc token mapper vocab_size=%d num_nodes=%d", int(vocab_size), int(num_nodes))
    token_mapper = GCTokenMapper(vocab_size=vocab_size)
    generator = GraphGenerator(num_nodes=num_nodes, num_colors=int(args.num_colors), edge_prob=float(args.edge_prob), seed=int(args.seed))
    by_state: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for instance_idx in range(int(args.n_instances)):
        adjacency = generator.generate_planted().adjacency
        for trace_idx in range(int(args.n_traces_per_instance)):
            tie_seed = int(args.seed) + instance_idx * 10007 + trace_idx * 97
            tr = gc_trace(adjacency=adjacency, num_colors=int(args.num_colors), max_seq_len=int(args.max_seq_len), max_steps=int(args.num_nodes) * int(args.num_nodes) * 2, token_mapper=token_mapper, tie_seed=tie_seed)
            tr._trace_idx = int(trace_idx)
            for dp in tr.decision_points:
                start = _state_start(tr.tokens, tr.block_ids, int(dp.position), int(token_mapper.STATE))
                by_state[(instance_idx, repr(dp.canonical_state))].append({"signature": repr(dp.canonical_state), "label": "viable", "depth": int(dp.decision_level), "size": int(sum(1 for v in dp.assignment if int(v) == 0)), "allowed": _gc_allowed(dp, int(args.num_colors)), "history": _make_history(tr, int(dp.position), int(start), int(tr.graph_prefix_len))})
            for point in _scan_gc_conflicts(tr, token_mapper, int(tr.graph_prefix_len), int(args.num_colors)):
                by_state[(instance_idx, point["signature"])].append(point)
        if (instance_idx + 1) % 10 == 0:
            logger.info("gc instances=%d states_indexed=%d", instance_idx + 1, len(by_state))
    return _materialize(by_state, args)


def _materialize(by_state: Dict[Tuple[int, str], List[Dict[str, Any]]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    global LAST_DEPTH_QS, LAST_SIZE_QS
    probes = []
    dropped_state_mismatch = 0
    same_history_length = 0
    for (instance_idx, sig), matches in sorted(by_state.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        selected = _select_histories(matches, int(args.max_histories_per_state))
        if len(selected) < int(args.min_histories_per_state):
            continue

        history_lengths = [int(m["history"].get("history_token_length", 0)) for m in selected]
        if len(set(history_lengths)) < 2:
            same_history_length += 1

        state_blocks = [_state_block_tokens(m["history"]) for m in selected]
        if any(block != state_blocks[0] for block in state_blocks[1:]):
            dropped_state_mismatch += 1
            if dropped_state_mismatch <= 5:
                logger.warning(
                    "dropping probe instance=%d signature=%s: state-block bytes differ across histories",
                    int(instance_idx),
                    str(sig)[:160],
                )
            continue

        first = selected[0]
        for hid, item in enumerate(selected):
            item["history"]["history_id"] = int(hid)
        probe_id = hashlib.sha1(f"{args.domain}:{instance_idx}:{sig}".encode()).hexdigest()
        probes.append({"probe_id": probe_id, "instance_idx": int(instance_idx), "label": first["label"], "depth": int(first["depth"]), "depth_quartile": 1, "size": int(first["size"]), "size_quartile": 1, "canonical_state_signature": str(sig), "allowed_continue_actions": first["allowed"], "histories": [m["history"] for m in selected]})
    if dropped_state_mismatch:
        logger.warning("dropped %d probes with non-identical state-block bytes", int(dropped_state_mismatch))
    if same_history_length:
        logger.warning("kept %d probes whose histories have equal history_token_length", int(same_history_length))
    dqs, sqs = _quartiles([p["depth"] for p in probes]), _quartiles([p["size"] for p in probes])
    for p in probes:
        p["depth_quartile"] = _quartile(int(p["depth"]), dqs)
        p["size_quartile"] = _quartile(int(p["size"]), sqs)
    LAST_DEPTH_QS, LAST_SIZE_QS = dqs, sqs
    return probes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=["sat", "gc"], required=True)
    ap.add_argument("--num_vars", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--num_nodes", type=int, default=30)
    ap.add_argument("--num_colors", type=int, default=4)
    ap.add_argument("--edge_prob", type=float, default=0.23)
    ap.add_argument("--vocab_size", type=int, default=0)
    ap.add_argument("--n_instances", type=int, default=100)
    ap.add_argument("--n_traces_per_instance", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--state_sort", choices=["lexical", "vsids"], default="lexical")
    ap.add_argument("--min_histories_per_state", type=int, default=2)
    ap.add_argument("--max_histories_per_state", type=int, default=4)
    ap.add_argument("--output_path", type=str, required=True)
    args = ap.parse_args()
    random.seed(int(args.seed)); np.random.seed(int(args.seed))
    probes = _collect_sat(args) if args.domain == "sat" else _collect_gc(args)
    labels = Counter(p["label"] for p in probes)
    n_pairs = sum(len(p["histories"]) * (len(p["histories"]) - 1) // 2 for p in probes)
    stats = {"n_instances": int(args.n_instances), "n_canonical_states": len(probes), "n_states_with_pairs": len(probes), "n_pairs": int(n_pairs), "label_distribution": {"viable": int(labels.get("viable", 0)), "exposed_conflict": int(labels.get("exposed_conflict", 0))}, "depth_quartile_thresholds": LAST_DEPTH_QS, "size_quartile_thresholds": LAST_SIZE_QS}
    if stats["label_distribution"]["exposed_conflict"] < 200:
        logger.warning("exposed-conflict probes below 200 (%d); consider increasing --n_instances", stats["label_distribution"]["exposed_conflict"])
    payload = {"config": vars(args), "stats": stats, "probes": probes}
    out = Path(args.output_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s probes=%d pairs=%d labels=%s", out, len(probes), n_pairs, dict(labels))


if __name__ == "__main__":
    main()
