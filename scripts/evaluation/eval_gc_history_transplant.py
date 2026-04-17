#!/usr/bin/env python3
"""History-transplant behavioral test for graph coloring SSA vs causal models."""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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


class TokenMapper:
    """Maps high-range token IDs based on model vocabulary size."""

    def __init__(self, vocab_size: int):
        vocab_size = int(vocab_size)
        if vocab_size == 394:
            max_nodes = 30
        elif vocab_size == 574:
            max_nodes = 75
        else:
            raise ValueError(
                f"Unsupported vocab_size={vocab_size}, expected 394 or 574"
            )

        max_colors = 4
        self.vocab_size = int(vocab_size)
        self.MASK_OFFSET = int(240 + max_nodes * max_colors)
        self.STATE = int(self.MASK_OFFSET + 16)
        self.CF = int(self.MASK_OFFSET + 24)
        self.TRIED = int(self.MASK_OFFSET + 32)
        self.END_TRIED = int(self.MASK_OFFSET + 33)

    def mask_token(self, domain: set[int]) -> int:
        bitmask = 0
        for c in domain:
            bitmask |= 1 << int(c)
        return int(self.MASK_OFFSET + bitmask)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


PrefixKey = Tuple[Tuple[int, int], ...]
CanonicalStateKey = Tuple[
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, Tuple[int, ...]], ...],
    bool,
    int,
    Tuple[Tuple[int, int], ...],
]


@dataclass
class DecisionPoint:
    position: int
    block_id: int
    block_start: int
    assignment: List[int]
    domains: List[List[int]]
    canonical_state: CanonicalStateKey
    conflict_status: bool
    decision_level: int
    tried_alternatives: List[Tuple[int, int]]


@dataclass
class OracleTrace:
    tokens: List[int]
    block_ids: List[int]
    decision_points: List[DecisionPoint]
    graph_prefix_len: int


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _ensure_checkpoint_exists(checkpoint_path: Path, arg_name: str) -> None:
    if checkpoint_path.exists():
        return
    raise FileNotFoundError(
        f"{arg_name} not found: {checkpoint_path}. "
        "Pass an explicit checkpoint path that exists on disk."
    )


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


def _prefix_key_from_assignment(assignment: np.ndarray) -> PrefixKey:
    nz = np.nonzero(assignment)[0]
    return tuple(sorted((int(node), int(assignment[int(node)])) for node in nz))


def _sorted_candidates(state: GraphColorState, degrees: np.ndarray) -> List[int]:
    unassigned = [
        int(i) for i in range(state.num_nodes) if int(state.assignment[int(i)]) == 0
    ]
    return sorted(
        unassigned,
        key=lambda nd: (len(state.domains[int(nd)]), -int(degrees[int(nd)])),
    )


def _is_solution(env: GraphColorEnv, state: GraphColorState) -> bool:
    if state.propagation_pending or state.selected_node is not None:
        return False
    if int(np.count_nonzero(state.assignment == 0)) != 0:
        return False
    return not env._has_contradiction(state)


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


def _canonical_state_key(
    env: GraphColorEnv,
    state: GraphColorState,
    tried_alternatives: Sequence[Tuple[int, int]],
) -> CanonicalStateKey:
    assignment_key = _prefix_key_from_assignment(state.assignment)
    domains_key = tuple(
        (
            int(node_id),
            tuple(sorted(int(color_id) for color_id in state.domains[int(node_id)])),
        )
        for node_id in range(state.num_nodes)
        if int(state.assignment[int(node_id)]) == 0
    )
    return (
        assignment_key,
        domains_key,
        bool(env._has_contradiction(state)),
        int(len(state.assignment_stack)),
        tuple(
            sorted(
                (int(node_id), int(color_id))
                for node_id, color_id in tried_alternatives
            )
        ),
    )


def _dsatur_select_random_tie(
    env: GraphColorEnv,
    state: GraphColorState,
    depth: int,
    rng: random.Random,
) -> Optional[int]:
    assigned = state.assignment
    adj = state.adjacency

    best_nodes: List[int] = []
    best_dom_size = 10**9
    best_saturation = -1

    for node in range(state.num_nodes):
        if int(assigned[int(node)]) != 0:
            continue

        dom = env._effective_domain(state, int(node), depth=int(depth))
        dom_size = int(len(dom))
        if dom_size == 0:
            continue

        neighbors = np.nonzero(adj[int(node)])[0]
        sat_colors = {
            int(assigned[int(nb)]) for nb in neighbors if int(assigned[int(nb)]) != 0
        }
        saturation = int(len(sat_colors))

        if dom_size < best_dom_size or (
            dom_size == best_dom_size and saturation > best_saturation
        ):
            best_nodes = [int(node)]
            best_dom_size = int(dom_size)
            best_saturation = int(saturation)
        elif dom_size == best_dom_size and saturation == best_saturation:
            best_nodes.append(int(node))

    if not best_nodes:
        return None
    rng.shuffle(best_nodes)
    return int(best_nodes[0])


def generate_oracle_trace_with_random_ties(
    *,
    adjacency: np.ndarray,
    num_colors: int,
    max_seq_len: int,
    max_steps: int,
    token_mapper: TokenMapper,
    tie_seed: int,
) -> OracleTrace:
    """Run oracle-like solver and emit tokenized trace with randomized DSATUR tie-breaks."""
    num_nodes = int(adjacency.shape[0])
    tokenizer = CDCLTokenizer()
    degrees = np.sum(adjacency, axis=1).astype(np.int64)
    rng = random.Random(int(tie_seed))

    env = GraphColorEnv(
        adjacency=adjacency,
        num_colors=int(num_colors),
        solution=None,
        mode="strict",
        max_steps=int(max_steps * 4 + 10),
        propagation_mode="forward_check",
    )
    oracle = GraphColorOracle(env)
    _ = oracle
    env.reset()

    tokens: List[int] = tokenizer.build_graph_prefix(adjacency, num_nodes)
    graph_prefix_len = int(len(tokens))
    block_ids: List[int] = [0] * len(tokens)
    current_block = 0

    tried_at_prefix: Dict[PrefixKey, List[Tuple[int, int]]] = {}
    decision_points: List[DecisionPoint] = []

    for _step in range(int(max_steps)):
        state = env.get_state()
        if _is_solution(env, state):
            break
        if state.status != GraphColorEnvStatus.RUNNING:
            break

        sorted_nodes = _sorted_candidates(state, degrees)
        if not sorted_nodes:
            break

        depth = int(len(state.assignment_stack) + 1)
        effective_domains = {
            int(nd): env._effective_domain(state, int(nd), depth=depth)
            for nd in sorted_nodes
        }
        min_domain = min(int(len(effective_domains[int(nd)])) for nd in sorted_nodes)

        prefix_key = _prefix_key_from_assignment(state.assignment)
        prior = tried_at_prefix.get(prefix_key, [])
        block_start_index: Optional[int] = None
        state_preceded_by_tried = False

        if prior:
            current_block += 1
            tried_tokens = [int(token_mapper.TRIED)]
            for node_id, color_id in prior:
                tried_tokens.append(tokenizer.node_token(int(node_id)))
                tried_tokens.append(tokenizer.color_token(int(color_id)))
            tried_tokens.append(int(token_mapper.END_TRIED))
            block_start_index = len(tokens)
            if not _append_tokens(
                tokens,
                block_ids,
                tried_tokens,
                block_id=current_block,
                max_seq_len=max_seq_len,
            ):
                break
            state_preceded_by_tried = True

        if not state_preceded_by_tried:
            current_block += 1
            block_start_index = len(tokens)

        state_tokens = [int(token_mapper.STATE)]
        state_tokens.extend(tokenizer.node_token(int(nd)) for nd in sorted_nodes)
        state_tokens.append(tokenizer.SEP)
        if not _append_tokens(
            tokens,
            block_ids,
            state_tokens,
            block_id=current_block,
            max_seq_len=max_seq_len,
        ):
            break

        if min_domain == 0:
            if not _append_tokens(
                tokens,
                block_ids,
                [int(token_mapper.CF)],
                block_id=current_block,
                max_seq_len=max_seq_len,
            ):
                break

            if state.assignment_stack:
                failed_node, failed_color, _ = state.assignment_stack[-1]
                parent_prefix = tuple(
                    sorted((int(n), int(c)) for n, c, _ in state.assignment_stack[:-1])
                )
                parent_tried = tried_at_prefix.setdefault(parent_prefix, [])
                failed_pair = (int(failed_node), int(failed_color))
                if failed_pair not in parent_tried:
                    parent_tried.append(failed_pair)
                env.backjump_to(len(state.assignment_stack) - 1)
                continue
            break

        if not _append_tokens(
            tokens,
            block_ids,
            [tokenizer.OK],
            block_id=current_block,
            max_seq_len=max_seq_len,
        ):
            break

        decision_pos = int(len(tokens) - 1)
        decision_points.append(
            DecisionPoint(
                position=int(decision_pos),
                block_id=int(current_block),
                block_start=int(block_start_index)
                if block_start_index is not None
                else 0,
                assignment=[int(x) for x in state.assignment.tolist()],
                domains=[[int(c) for c in sorted(dom)] for dom in state.domains],
                canonical_state=_canonical_state_key(
                    env=env,
                    state=state,
                    tried_alternatives=prior,
                ),
                conflict_status=bool(env._has_contradiction(state)),
                decision_level=int(len(state.assignment_stack)),
                tried_alternatives=[
                    (int(node_id), int(color_id)) for node_id, color_id in prior
                ],
            )
        )

        selected_node = _dsatur_select_random_tie(
            env=env,
            state=state,
            depth=int(depth),
            rng=rng,
        )
        if selected_node is None:
            break

        selected_domain = set(
            int(c)
            for c in env._effective_domain(state, int(selected_node), depth=depth)
        )
        if not selected_domain:
            break
        selected_color = int(min(selected_domain))

        if not _append_tokens(
            tokens,
            block_ids,
            [
                tokenizer.node_token(int(selected_node)),
                token_mapper.mask_token(selected_domain),
                tokenizer.color_token(int(selected_color)),
            ],
            block_id=current_block,
            max_seq_len=max_seq_len,
        ):
            break

        ok, _reason = _apply_assignment(env, int(selected_node), int(selected_color))
        if not ok:
            break

        if not _append_tokens(
            tokens,
            block_ids,
            [
                tokenizer.OK,
                tokenizer.node_token(int(selected_node)),
                tokenizer.color_token(int(selected_color)),
            ],
            block_id=current_block,
            max_seq_len=max_seq_len,
        ):
            break

    if len(tokens) != len(block_ids):
        raise RuntimeError("trace token/block length mismatch")

    return OracleTrace(
        tokens=tokens,
        block_ids=block_ids,
        decision_points=decision_points,
        graph_prefix_len=int(graph_prefix_len),
    )


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any], bool]:
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
    max_seq_len_model = int(config.get("max_seq_len", 2048))
    dropout = float(config.get("dropout", 0.1))
    attention_mode = str(config.get("attention_mode", "causal"))

    if attention_mode == "ssa":
        from universal.ssa_decoder import SSASlotDecoder

        model: torch.nn.Module = SSASlotDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len_model,
            n_slots=n_slots,
            dropout=dropout,
        )
        is_ssa = True
        kind = "SSASlotDecoder"
    else:
        model = SlotCDCLDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len_model,
            n_slots=n_slots,
            dropout=dropout,
        )
        is_ssa = False
        kind = "SlotCDCLDecoder"

    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state_dict.items():
        if key in model_state and value.shape != model_state[key].shape:
            skipped.append(str(key))
            continue
        filtered[key] = value

    if skipped:
        logger.warning(
            "Skipping %d mismatched keys from %s", len(skipped), checkpoint_path
        )

    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()

    meta = {
        "kind": kind,
        "attention_mode": attention_mode,
        "config": config,
        "max_seq_len_model": int(max_seq_len_model),
        "n_layers": int(n_layers),
        "n_slots": int(n_slots),
        "vocab_size": int(vocab_size),
    }
    return model, meta, is_ssa


@torch.no_grad()
def _forward_logits(
    model: torch.nn.Module,
    input_ids: Sequence[int],
    block_ids: Optional[Sequence[int]],
    device: torch.device,
    is_ssa: bool,
) -> torch.Tensor:
    input_tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    if is_ssa:
        if block_ids is None:
            raise ValueError("SSA forward requires block_ids")
        block_tensor = torch.tensor([list(block_ids)], dtype=torch.long, device=device)
        lm_logits, _verify_logits = model(input_tensor, block_ids=block_tensor)
    else:
        lm_logits, _verify_logits = model(input_tensor)
    return lm_logits[0, -1, :]


def _argmax_node(
    logits: torch.Tensor,
    tokenizer: CDCLTokenizer,
    allowed_nodes: Sequence[int],
) -> int:
    node_mask = torch.full_like(logits, float("-inf"))
    for nd in allowed_nodes:
        node_mask[tokenizer.node_token(int(nd))] = 0.0
    node_token = int(torch.argmax(logits + node_mask).item())
    selected_node = int(node_token - int(tokenizer.NODE_OFFSET))
    if selected_node not in allowed_nodes:
        return int(allowed_nodes[0])
    return int(selected_node)


def _argmax_color(
    logits: torch.Tensor,
    tokenizer: CDCLTokenizer,
    allowed_colors: Sequence[int],
) -> int:
    color_mask = torch.full_like(logits, float("-inf"))
    for color in allowed_colors:
        color_mask[tokenizer.color_token(int(color))] = 0.0
    color_token = int(torch.argmax(logits + color_mask).item())
    selected_color = int(color_token - int(tokenizer.COLOR_OFFSET))
    if selected_color not in allowed_colors:
        return int(sorted(allowed_colors)[0])
    return int(selected_color)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-12
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    kl = torch.sum(p * (torch.log(p) - torch.log(q)))
    return float(kl.item())


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1, eps=1e-8)
    return float(sim.item())


def _history_difference_tokens(
    history_a: Sequence[int], history_b: Sequence[int]
) -> int:
    min_len = min(int(len(history_a)), int(len(history_b)))
    diffs = sum(
        1
        for idx in range(int(min_len))
        if int(history_a[int(idx)]) != int(history_b[int(idx)])
    )
    return int(diffs + abs(int(len(history_a)) - int(len(history_b))))


def _predict_action_and_distribution(
    *,
    model: torch.nn.Module,
    is_ssa: bool,
    prefix_tokens: Sequence[int],
    prefix_block_ids: Sequence[int],
    decision_point: DecisionPoint,
    tokenizer: CDCLTokenizer,
    token_mapper: TokenMapper,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, Any]:
    node_logits = _forward_logits(
        model=model,
        input_ids=prefix_tokens,
        block_ids=prefix_block_ids if is_ssa else None,
        device=device,
        is_ssa=is_ssa,
    )
    node_probs = torch.softmax(node_logits, dim=-1)

    allowed_nodes = [
        int(node_id)
        for node_id in range(len(decision_point.assignment))
        if int(decision_point.assignment[int(node_id)]) == 0
    ]
    if not allowed_nodes:
        raise RuntimeError("no allowed node candidates at decision point")
    selected_node = _argmax_node(node_logits, tokenizer, allowed_nodes)

    selected_domain = sorted(int(c) for c in decision_point.domains[int(selected_node)])
    if not selected_domain:
        raise RuntimeError("selected node has empty domain at decision point")

    color_prefix_tokens = list(prefix_tokens)
    color_prefix_block_ids = list(prefix_block_ids)
    color_prefix_tokens.append(tokenizer.node_token(int(selected_node)))
    color_prefix_tokens.append(token_mapper.mask_token(set(selected_domain)))
    color_prefix_block_ids.append(int(decision_point.block_id))
    color_prefix_block_ids.append(int(decision_point.block_id))

    if len(color_prefix_tokens) > int(max_seq_len):
        raise RuntimeError("color probe exceeds max_seq_len")

    color_logits = _forward_logits(
        model=model,
        input_ids=color_prefix_tokens,
        block_ids=color_prefix_block_ids if is_ssa else None,
        device=device,
        is_ssa=is_ssa,
    )
    color_probs = torch.softmax(color_logits, dim=-1)
    selected_color = _argmax_color(color_logits, tokenizer, selected_domain)

    return {
        "action": (int(selected_node), int(selected_color)),
        "node_logits": node_logits.detach(),
        "node_probs": node_probs.detach(),
        "color_logits": color_logits.detach(),
        "color_probs": color_probs.detach(),
    }


def _compare_prefix_behavior(
    *,
    model: torch.nn.Module,
    is_ssa: bool,
    prefix_a_tokens: Sequence[int],
    prefix_a_blocks: Sequence[int],
    dp_a: DecisionPoint,
    prefix_b_tokens: Sequence[int],
    prefix_b_blocks: Sequence[int],
    dp_b: DecisionPoint,
    tokenizer: CDCLTokenizer,
    token_mapper: TokenMapper,
    max_seq_len: int,
    device: torch.device,
    store_full_distributions: bool,
) -> Dict[str, Any]:
    probe_a = _predict_action_and_distribution(
        model=model,
        is_ssa=is_ssa,
        prefix_tokens=prefix_a_tokens,
        prefix_block_ids=prefix_a_blocks,
        decision_point=dp_a,
        tokenizer=tokenizer,
        token_mapper=token_mapper,
        max_seq_len=max_seq_len,
        device=device,
    )
    probe_b = _predict_action_and_distribution(
        model=model,
        is_ssa=is_ssa,
        prefix_tokens=prefix_b_tokens,
        prefix_block_ids=prefix_b_blocks,
        decision_point=dp_b,
        tokenizer=tokenizer,
        token_mapper=token_mapper,
        max_seq_len=max_seq_len,
        device=device,
    )

    p_a = probe_a["node_probs"]
    p_b = probe_b["node_probs"]
    kl_ab = _kl_divergence(p_a, p_b)
    kl_ba = _kl_divergence(p_b, p_a)
    kl_sym = float(0.5 * (kl_ab + kl_ba))
    cos = _cosine_similarity(probe_a["node_logits"], probe_b["node_logits"])

    record: Dict[str, Any] = {
        "action_a": {
            "node": int(probe_a["action"][0]),
            "color": int(probe_a["action"][1]),
        },
        "action_b": {
            "node": int(probe_b["action"][0]),
            "color": int(probe_b["action"][1]),
        },
        "action_agreement": bool(
            int(probe_a["action"][0]) == int(probe_b["action"][0])
            and int(probe_a["action"][1]) == int(probe_b["action"][1])
        ),
        "kl_ab": float(kl_ab),
        "kl_ba": float(kl_ba),
        "kl_symmetric": float(kl_sym),
        "cosine_sim": float(cos),
        "argmax_node_token_a": int(torch.argmax(probe_a["node_logits"]).item()),
        "argmax_node_token_b": int(torch.argmax(probe_b["node_logits"]).item()),
    }

    if store_full_distributions:
        record["node_distribution_a"] = [
            float(x) for x in probe_a["node_probs"].detach().cpu().tolist()
        ]
        record["node_distribution_b"] = [
            float(x) for x in probe_b["node_probs"].detach().cpu().tolist()
        ]

    return record


def _aggregate_behavior(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {
            "action_agreement": 0.0,
            "mean_kl_divergence": 0.0,
            "std_kl_divergence": 0.0,
            "mean_cosine_sim": 0.0,
            "std_cosine_sim": 0.0,
        }

    agreement = [1.0 if bool(e["action_agreement"]) else 0.0 for e in entries]
    kls = [float(e["kl_symmetric"]) for e in entries]
    cosines = [float(e["cosine_sim"]) for e in entries]
    return {
        "action_agreement": float(_safe_mean(agreement)),
        "mean_kl_divergence": float(_safe_mean(kls)),
        "std_kl_divergence": float(_safe_std(kls)),
        "mean_cosine_sim": float(_safe_mean(cosines)),
        "std_cosine_sim": float(_safe_std(cosines)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="History transplant behavioral test for GC models"
    )
    parser.add_argument(
        "--ssa_checkpoint",
        type=str,
        default="experiments/gc-ssa-v2-pretrained/best.pt",
    )
    parser.add_argument(
        "--causal_checkpoint",
        type=str,
        default="experiments/gc-causal-v2-pretrained/best.pt",
    )
    parser.add_argument(
        "--n_graphs", "--n_instances", dest="n_graphs", type=int, default=100
    )
    parser.add_argument(
        "--n_traces_per_graph",
        "--n_traces_per_instance",
        dest="n_traces_per_graph",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--n_nodes", "--num_nodes", dest="n_nodes", type=int, default=30
    )
    parser.add_argument(
        "--n_colors", "--num_colors", dest="n_colors", type=int, default=4
    )
    parser.add_argument("--edge_prob", type=float, default=0.3)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="experiments/history-transplant"
    )
    parser.add_argument(
        "--store_full_distributions",
        action="store_true",
        help="If set, stores full node softmax vectors per pair/prefix",
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_checkpoint_exists(Path(args.ssa_checkpoint), "--ssa_checkpoint")
    _ensure_checkpoint_exists(Path(args.causal_checkpoint), "--causal_checkpoint")

    ssa_model, ssa_meta, ssa_is_ssa = _load_checkpoint(
        Path(args.ssa_checkpoint), device
    )
    causal_model, causal_meta, causal_is_ssa = _load_checkpoint(
        Path(args.causal_checkpoint), device
    )

    if not ssa_is_ssa:
        raise RuntimeError("SSA checkpoint does not declare attention_mode='ssa'")
    if causal_is_ssa:
        raise RuntimeError(
            "Causal checkpoint unexpectedly declares attention_mode='ssa'"
        )

    ssa_vocab_size = int(ssa_meta["vocab_size"])
    causal_vocab_size = int(causal_meta["vocab_size"])
    if ssa_vocab_size != causal_vocab_size:
        raise RuntimeError(
            "SSA and causal checkpoints use different vocab sizes: "
            f"ssa={ssa_vocab_size} causal={causal_vocab_size}"
        )

    token_mapper = TokenMapper(vocab_size=ssa_vocab_size)
    tokenizer = CDCLTokenizer()

    logger.info(
        "token_mapper vocab_size=%d mask_offset=%d state=%d cf=%d tried=%d end_tried=%d",
        int(token_mapper.vocab_size),
        int(token_mapper.MASK_OFFSET),
        int(token_mapper.STATE),
        int(token_mapper.CF),
        int(token_mapper.TRIED),
        int(token_mapper.END_TRIED),
    )

    effective_max_seq_len = int(
        min(
            int(args.max_seq_len),
            int(ssa_meta["max_seq_len_model"]),
            int(causal_meta["max_seq_len_model"]),
        )
    )
    max_steps = int(args.n_nodes * args.n_nodes * 2)

    generator = GraphGenerator(
        num_nodes=int(args.n_nodes),
        num_colors=int(args.n_colors),
        edge_prob=float(args.edge_prob),
        seed=int(args.seed),
    )

    ssa_entries: List[Dict[str, Any]] = []
    causal_entries: List[Dict[str, Any]] = []

    total_candidate_matches = 0
    skipped_identical_history = 0
    skipped_too_long = 0
    skipped_bad_probe = 0
    graphs_with_pairs = 0
    pair_id = 0

    for graph_idx in range(int(args.n_graphs)):
        adjacency = generator.generate_planted().adjacency

        traces: List[OracleTrace] = []
        for trace_idx in range(int(args.n_traces_per_graph)):
            tie_seed = int(args.seed) + int(graph_idx) * 10_007 + int(trace_idx) * 97
            trace = generate_oracle_trace_with_random_ties(
                adjacency=adjacency,
                num_colors=int(args.n_colors),
                max_seq_len=effective_max_seq_len,
                max_steps=max_steps,
                token_mapper=token_mapper,
                tie_seed=int(tie_seed),
            )
            traces.append(trace)

        state_index: Dict[CanonicalStateKey, List[Tuple[int, int]]] = {}
        for trace_idx, trace in enumerate(traces):
            for dp_idx, dp in enumerate(trace.decision_points):
                state_index.setdefault(dp.canonical_state, []).append(
                    (int(trace_idx), int(dp_idx))
                )

        local_pairs = 0
        for _state_key, matches in state_index.items():
            if len(matches) < 2:
                continue

            for i in range(len(matches) - 1):
                trace_a_idx, dp_a_idx = matches[i]
                for j in range(i + 1, len(matches)):
                    trace_b_idx, dp_b_idx = matches[j]
                    if int(trace_a_idx) == int(trace_b_idx):
                        continue
                    total_candidate_matches += 1

                    trace_a = traces[int(trace_a_idx)]
                    trace_b = traces[int(trace_b_idx)]
                    dp_a = trace_a.decision_points[int(dp_a_idx)]
                    dp_b = trace_b.decision_points[int(dp_b_idx)]

                    if int(trace_a.tokens[int(dp_a.position)]) != int(tokenizer.OK):
                        skipped_bad_probe += 1
                        continue
                    if int(trace_b.tokens[int(dp_b.position)]) != int(tokenizer.OK):
                        skipped_bad_probe += 1
                        continue

                    prefix_a_tokens = [
                        int(x) for x in trace_a.tokens[: int(dp_a.position) + 1]
                    ]
                    prefix_a_blocks = [
                        int(x) for x in trace_a.block_ids[: int(dp_a.position) + 1]
                    ]
                    prefix_b_tokens = [
                        int(x) for x in trace_b.tokens[: int(dp_b.position) + 1]
                    ]
                    prefix_b_blocks = [
                        int(x) for x in trace_b.block_ids[: int(dp_b.position) + 1]
                    ]

                    if len(prefix_a_tokens) > int(effective_max_seq_len) or len(
                        prefix_b_tokens
                    ) > int(effective_max_seq_len):
                        skipped_too_long += 1
                        continue

                    history_a = prefix_a_tokens[
                        int(trace_a.graph_prefix_len) : int(dp_a.block_start)
                    ]
                    history_b = prefix_b_tokens[
                        int(trace_b.graph_prefix_len) : int(dp_b.block_start)
                    ]
                    if history_a == history_b:
                        skipped_identical_history += 1
                        continue

                    history_diff_tokens = _history_difference_tokens(
                        history_a, history_b
                    )
                    if int(history_diff_tokens) <= 0:
                        skipped_identical_history += 1
                        continue

                    try:
                        ssa_behavior = _compare_prefix_behavior(
                            model=ssa_model,
                            is_ssa=True,
                            prefix_a_tokens=prefix_a_tokens,
                            prefix_a_blocks=prefix_a_blocks,
                            dp_a=dp_a,
                            prefix_b_tokens=prefix_b_tokens,
                            prefix_b_blocks=prefix_b_blocks,
                            dp_b=dp_b,
                            tokenizer=tokenizer,
                            token_mapper=token_mapper,
                            max_seq_len=effective_max_seq_len,
                            device=device,
                            store_full_distributions=bool(
                                args.store_full_distributions
                            ),
                        )
                        causal_behavior = _compare_prefix_behavior(
                            model=causal_model,
                            is_ssa=False,
                            prefix_a_tokens=prefix_a_tokens,
                            prefix_a_blocks=prefix_a_blocks,
                            dp_a=dp_a,
                            prefix_b_tokens=prefix_b_tokens,
                            prefix_b_blocks=prefix_b_blocks,
                            dp_b=dp_b,
                            tokenizer=tokenizer,
                            token_mapper=token_mapper,
                            max_seq_len=effective_max_seq_len,
                            device=device,
                            store_full_distributions=bool(
                                args.store_full_distributions
                            ),
                        )
                    except RuntimeError as exc:
                        skipped_bad_probe += 1
                        logger.debug("skip pair runtime_error=%s", str(exc))
                        continue

                    shared = {
                        "pair_id": int(pair_id),
                        "graph_idx": int(graph_idx),
                        "trace_a_idx": int(trace_a_idx),
                        "trace_b_idx": int(trace_b_idx),
                        "decision_a_idx": int(dp_a_idx),
                        "decision_b_idx": int(dp_b_idx),
                        "prefix_a_len": int(len(prefix_a_tokens)),
                        "prefix_b_len": int(len(prefix_b_tokens)),
                        "history_tokens_a": int(len(history_a)),
                        "history_tokens_b": int(len(history_b)),
                        "history_tokens_differ": int(history_diff_tokens),
                        "decision_level": int(dp_a.decision_level),
                        "conflict_status": bool(dp_a.conflict_status),
                        "tried_alternatives": [
                            {"node": int(node_id), "color": int(color_id)}
                            for node_id, color_id in dp_a.tried_alternatives
                        ],
                    }
                    ssa_entries.append({**shared, **ssa_behavior})
                    causal_entries.append({**shared, **causal_behavior})

                    pair_id += 1
                    local_pairs += 1

        if local_pairs > 0:
            graphs_with_pairs += 1

        if (graph_idx + 1) % 10 == 0:
            logger.info(
                "processed_graphs=%d/%d pairs=%d candidates=%d skipped_identical=%d skipped_too_long=%d skipped_bad_probe=%d",
                int(graph_idx + 1),
                int(args.n_graphs),
                int(len(ssa_entries)),
                int(total_candidate_matches),
                int(skipped_identical_history),
                int(skipped_too_long),
                int(skipped_bad_probe),
            )
            if ssa_entries:
                last_ssa = ssa_entries[-1]
                last_causal = causal_entries[-1]
                logger.info(
                    "sample_pair id=%d hist_diff=%d ssa(agree=%s kl=%.4f cos=%.4f) causal(agree=%s kl=%.4f cos=%.4f)",
                    int(last_ssa["pair_id"]),
                    int(last_ssa["history_tokens_differ"]),
                    str(bool(last_ssa["action_agreement"])),
                    float(last_ssa["kl_symmetric"]),
                    float(last_ssa["cosine_sim"]),
                    str(bool(last_causal["action_agreement"])),
                    float(last_causal["kl_symmetric"]),
                    float(last_causal["cosine_sim"]),
                )

    ssa_summary = _aggregate_behavior(ssa_entries)
    causal_summary = _aggregate_behavior(causal_entries)

    payload = {
        "config": {
            "ssa_checkpoint": str(args.ssa_checkpoint),
            "causal_checkpoint": str(args.causal_checkpoint),
            "n_graphs": int(args.n_graphs),
            "n_instances": int(args.n_graphs),
            "n_traces_per_graph": int(args.n_traces_per_graph),
            "n_traces_per_instance": int(args.n_traces_per_graph),
            "n_nodes": int(args.n_nodes),
            "num_nodes": int(args.n_nodes),
            "n_colors": int(args.n_colors),
            "num_colors": int(args.n_colors),
            "edge_prob": float(args.edge_prob),
            "max_seq_len": int(args.max_seq_len),
            "effective_max_seq_len": int(effective_max_seq_len),
            "device": str(args.device),
            "seed": int(args.seed),
            "output_dir": str(output_dir),
            "max_steps": int(max_steps),
            "store_full_distributions": bool(args.store_full_distributions),
        },
        "n_pairs": int(len(ssa_entries)),
        "pair_generation": {
            "graphs_with_pairs": int(graphs_with_pairs),
            "candidate_matches": int(total_candidate_matches),
            "n_candidate_matches": int(total_candidate_matches),
            "skipped_identical_history": int(skipped_identical_history),
            "skipped_too_long": int(skipped_too_long),
            "skipped_bad_probe": int(skipped_bad_probe),
            "retained_ratio": float(
                _safe_div(len(ssa_entries), total_candidate_matches)
            ),
        },
        "ssa": {
            **ssa_summary,
            "per_pair": ssa_entries,
        },
        "causal": {
            **causal_summary,
            "per_pair": causal_entries,
        },
    }

    output_path = output_dir / "results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== History Transplant Behavioral Test ===")
    print(f"pairs: {int(len(ssa_entries))}")
    print(
        "| model  | action_agreement | mean_kl_divergence | mean_cosine_sim | std_kl | std_cos |"
    )
    print(
        "|--------|------------------|--------------------|-----------------|--------|---------|"
    )
    print(
        "| SSA    | "
        f"{float(ssa_summary['action_agreement']):.4f}            | "
        f"{float(ssa_summary['mean_kl_divergence']):.6f}           | "
        f"{float(ssa_summary['mean_cosine_sim']):.4f}          | "
        f"{float(ssa_summary['std_kl_divergence']):.6f} | "
        f"{float(ssa_summary['std_cosine_sim']):.4f}  |"
    )
    print(
        "| Causal | "
        f"{float(causal_summary['action_agreement']):.4f}            | "
        f"{float(causal_summary['mean_kl_divergence']):.6f}           | "
        f"{float(causal_summary['mean_cosine_sim']):.4f}          | "
        f"{float(causal_summary['std_kl_divergence']):.6f} | "
        f"{float(causal_summary['std_cosine_sim']):.4f}  |"
    )
    print(f"results_json={str(output_path)}")


if __name__ == "__main__":
    main()
