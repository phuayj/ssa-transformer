#!/usr/bin/env python3
"""Evaluate SAT E4 verification behavior (polarity-or-CF)."""

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

from sat.dsl import SatAction
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator
from sat.interleaved_tokenizer import SATInterleavedTokenizer
from universal.slot_decoder import SlotCDCLDecoder
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Same explicit mapping as trace generator.
TOK_CLAUSE = 4
TOK_STATE = 11
TOK_TRIED = 17
TOK_END_TRIED = 20
TOK_CURRENT = 19
TOK_OK = 8
TOK_CF = 6
TOK_SEP = 3
TOK_EOS = 2
TOK_BOS = 1
TOK_TRUE = 15
TOK_FALSE = 16
VAR_OFFSET = 30
NEG_VAR_OFFSET = 130

PrefixKey = Tuple[Tuple[int, int], ...]


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


def _var_token(var_idx: int) -> int:
    return int(VAR_OFFSET + int(var_idx) + 1)


def _lit_token(lit: int) -> int:
    v = int(abs(int(lit)))
    return int(VAR_OFFSET + v) if int(lit) > 0 else int(NEG_VAR_OFFSET + v)


def _polarity_token(val: int) -> int:
    return int(TOK_TRUE) if int(val) == 1 else int(TOK_FALSE)


def _decode_polarity_token(token: int) -> int | None:
    if int(token) == int(TOK_TRUE):
        return 1
    if int(token) == int(TOK_FALSE):
        return -1
    return None


def _build_clause_prefix(clauses: Sequence[Tuple[int, int, int]]) -> List[int]:
    out: List[int] = [int(TOK_BOS)]
    for clause in clauses:
        out.append(int(TOK_CLAUSE))
        for lit in clause:
            out.append(int(_lit_token(int(lit))))
        out.append(int(TOK_SEP))
    return out


def _is_clause_satisfied(clause: Sequence[int], assignment: np.ndarray) -> bool:
    for lit in clause:
        v = int(abs(int(lit)) - 1)
        a = int(assignment[v])
        if a == 0:
            continue
        if (int(lit) > 0 and a == 1) or (int(lit) < 0 and a == -1):
            return True
    return False


def _phase_score(
    var_idx: int, val: int, state: SatState, clauses: Sequence[Tuple[int, int, int]]
) -> int:
    score = 0
    for clause in clauses:
        if _is_clause_satisfied(clause, state.assignment):
            continue
        for lit in clause:
            v = int(abs(int(lit)) - 1)
            if int(v) != int(var_idx):
                continue
            if (int(lit) > 0 and int(val) == 1) or (int(lit) < 0 and int(val) == -1):
                score += 1
                break
    return int(score)


def _oracle_polarity(
    var_idx: int, state: SatState, clauses: Sequence[Tuple[int, int, int]]
) -> int:
    t_score = _phase_score(int(var_idx), 1, state, clauses)
    f_score = _phase_score(int(var_idx), -1, state, clauses)
    return 1 if int(t_score) >= int(f_score) else -1


def _select_var_oracle(
    state: SatState, clauses: Sequence[Tuple[int, int, int]]
) -> int | None:
    if state.conflict_clause is not None:
        return None

    if state.decision_stack:
        top = state.decision_stack[-1]
        tv = int(top.decision_var)
        if int(state.assignment[tv]) == 0:
            return int(tv)

    unassigned = [
        v for v in range(int(state.num_vars)) if int(state.assignment[int(v)]) == 0
    ]
    if not unassigned:
        return None

    counts = np.zeros((int(state.num_vars),), dtype=np.int64)
    for clause in clauses:
        if _is_clause_satisfied(clause, state.assignment):
            continue
        for lit in clause:
            var = int(abs(int(lit)) - 1)
            if int(state.assignment[var]) == 0:
                counts[var] += 1

    ranked = sorted(unassigned, key=lambda v: (-int(counts[v]), int(v)))
    return int(ranked[0])


def _prefix_key_from_stack(
    state: SatState, current_var: int | None = None
) -> PrefixKey:
    key: List[Tuple[int, int]] = []
    for i, frame in enumerate(state.decision_stack):
        is_last = i == len(state.decision_stack) - 1
        if is_last and current_var is not None:
            if (
                int(frame.decision_var) == int(current_var)
                and int(state.assignment[int(current_var)]) == 0
            ):
                continue
        v = int(frame.decision_var)
        if int(state.assignment[v]) == 0:
            continue
        key.append((v, int(frame.chosen_val)))
    return tuple(key)


def _decision_prefix_tokens(
    *,
    include_tried: bool,
    current_var: int,
    tried_vals: Sequence[int],
    sorted_vars: Sequence[int],
) -> List[int]:
    out: List[int] = []
    if include_tried and len(tried_vals) > 0:
        out.append(int(TOK_TRIED))
        for val in tried_vals:
            out.append(int(_var_token(int(current_var))))
            out.append(int(_polarity_token(int(val))))
        out.append(int(TOK_END_TRIED))
    out.append(int(TOK_STATE))
    out.extend(int(_var_token(int(v))) for v in sorted_vars)
    out.append(int(TOK_SEP))
    out.append(int(TOK_CURRENT))
    out.append(int(_var_token(int(current_var))))
    out.append(int(TOK_OK))
    return out


def _clear_failed_mask_for_repeat(env: SatEnv, var_idx: int, val: int) -> None:
    state = env._state  # type: ignore[attr-defined]
    if state is None or not state.decision_stack:
        return
    top = state.decision_stack[-1]
    if int(top.decision_var) != int(var_idx):
        return
    if int(state.assignment[int(var_idx)]) != 0:
        return
    bit = 2 if int(val) == 1 else 1
    if int(top.failed_mask) & int(bit):
        top.failed_mask &= ~int(bit)


def _apply_assignment_allow_repeat(
    env: SatEnv, var_idx: int, val: int
) -> Tuple[bool, str]:
    st = env.get_state()
    if st.propagation_pending:
        res_prop0 = env.step(SatAction.propagate())
        if not bool(res_prop0.info.get("valid", True)):
            return (
                False,
                f"invalid_pending_propagate:{res_prop0.info.get('reason', 'unknown')}",
            )

    st = env.get_state()
    if st.selected_var is None:
        res_sel = env.step(SatAction.select_var(int(var_idx)))
        if not bool(res_sel.info.get("valid", True)):
            return False, f"invalid_select:{res_sel.info.get('reason', 'unknown')}"
    elif int(st.selected_var) != int(var_idx):
        return False, "selected_var_mismatch"

    _clear_failed_mask_for_repeat(env, int(var_idx), int(val))

    val_tok = 1 if int(val) == 1 else 0
    res_asn = env.step(SatAction.assign_value(int(val_tok)))
    if not bool(res_asn.info.get("valid", True)):
        return False, f"invalid_assign:{res_asn.info.get('reason', 'unknown')}"

    res_prop = env.step(SatAction.propagate())
    if not bool(res_prop.info.get("valid", True)):
        return False, f"invalid_propagate:{res_prop.info.get('reason', 'unknown')}"
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
    n_vars: int,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    generator = SatGenerator(seed=int(seed))
    rows: List[Dict[str, Any]] = []
    for _ in range(int(num_instances)):
        inst = generator.generate_planted(num_vars=int(n_vars), alpha=float(alpha))
        rows.append(
            {
                "clauses": [(int(c[0]), int(c[1]), int(c[2])) for c in inst.clauses],
                "num_vars": int(inst.num_vars),
                "planted_solution": None
                if inst.planted_solution is None
                else np.array(inst.planted_solution, dtype=np.int64, copy=True),
            }
        )
    return rows


@torch.no_grad()
def solve_instance(
    *,
    model: torch.nn.Module,
    meta: Dict[str, Any],
    tokenizer: SATInterleavedTokenizer,
    clauses: List[Tuple[int, int, int]],
    num_vars: int,
    planted_solution: np.ndarray | None,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, Any]:
    _ = tokenizer
    env = SatEnv(
        clauses=clauses,
        num_vars=int(num_vars),
        planted_solution=None
        if planted_solution is None
        else np.array(planted_solution, dtype=np.int64, copy=True),
        mode="strict",
        max_steps=int(max_steps * 8 + 30),
    )
    env.reset()

    sequence: List[int] = _build_clause_prefix(clauses)
    block_ids: List[int] = [0] * len(sequence)
    current_block = 0

    tried_for_state_var: Dict[Tuple[PrefixKey, int], List[int]] = {}
    repeat_flag_stack: List[bool] = []
    conflict_retry_count = 0

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
        "correct_polarity_on_non_exhausted": 0,
        "oracle_exact_match_on_non_exhausted": 0,
        "invalid_predictions": 0,
        "invalid_polarity_choices": 0,
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

        if state.status != SatEnvStatus.RUNNING:
            stats["solved"] = bool(state.status == SatEnvStatus.SUCCESS)
            stats["termination_reason"] = str(state.termination_reason or "env_done")
            break

        if state.propagation_pending:
            res_prop = env.step(SatAction.propagate())
            if not bool(res_prop.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_propagate:{res_prop.info.get('reason', 'unknown')}"
                )
                break
            state = env.get_state()

        if env._all_satisfied(state) and state.conflict_clause is None:
            done_res = env.step(SatAction.done())
            stats["solved"] = bool(
                done_res.done and env.get_state().status == SatEnvStatus.SUCCESS
            )
            stats["termination_reason"] = "solved"
            break

        sorted_vars = [
            int(v) for v in range(int(num_vars)) if int(state.assignment[int(v)]) == 0
        ]

        if state.conflict_clause is not None:
            if not state.decision_stack:
                done_res = env.step(SatAction.done())
                stats["termination_reason"] = (
                    "unsat_root"
                    if bool(done_res.done)
                    else "failed_done_after_root_conflict"
                )
                break

            current_var = int(state.decision_stack[-1].decision_var)
            prefix_key = _prefix_key_from_stack(state, current_var=int(current_var))
            tried_vals = list(
                tried_for_state_var.get((prefix_key, int(current_var)), [])
            )

            next_block = int(current_block + 1)
            decision_prefix = _decision_prefix_tokens(
                include_tried=True,
                current_var=int(current_var),
                tried_vals=tried_vals,
                sorted_vars=sorted_vars,
            )
            if not _append_tokens(
                sequence, block_ids, decision_prefix, next_block, int(max_seq_len)
            ):
                stats["termination_reason"] = "budget_exceeded"
                break

            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            if use_block_ids:
                block_tensor = torch.tensor(
                    [block_ids], dtype=torch.long, device=device
                )
                lm_logits, _ = model(
                    input_ids, block_ids=block_tensor, mask_mode=mask_mode
                )
            else:
                lm_logits, _ = model(input_ids)
            pred_token = int(torch.argmax(lm_logits[0, -1, :]).item())

            if int(pred_token) == int(TOK_CF):
                stats["correct_cf_on_exhausted"] += 1
                stats["backtrack_correct"] += 1
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(TOK_CF), int(TOK_OK), int(TOK_CF)],
                    next_block,
                    int(max_seq_len),
                ):
                    stats["termination_reason"] = "budget_exceeded"
                    break

                top = state.decision_stack[-1]
                prior = tried_for_state_var.setdefault(
                    (prefix_key, int(current_var)), []
                )
                if int(top.chosen_val) not in prior:
                    prior.append(int(top.chosen_val))

                top_repeat = bool(repeat_flag_stack[-1]) if repeat_flag_stack else False
                if top_repeat:
                    stats["repeat_induced_backtracks"] += 1
                if repeat_flag_stack:
                    repeat_flag_stack.pop()

                bt_res = env.step(SatAction.backtrack())
                if not bool(bt_res.info.get("valid", True)):
                    stats["termination_reason"] = (
                        f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                    )
                    break
                stats["backtracks"] += 1
                conflict_retry_count = 0
                current_block = int(next_block)
                continue

            stats["invalid_predictions"] += 1
            conflict_retry_count += 1
            if not _append_tokens(
                sequence,
                block_ids,
                [int(pred_token), int(TOK_OK), int(pred_token)],
                next_block,
                int(max_seq_len),
            ):
                stats["termination_reason"] = "budget_exceeded"
                break
            current_block = int(next_block)
            if int(conflict_retry_count) >= 3:
                stats["termination_reason"] = "conflict_stuck_non_cf"
                break
            continue

        current_var = _select_var_oracle(state, clauses)
        if current_var is None:
            stats["termination_reason"] = "no_selectable_var"
            break

        prefix_key = _prefix_key_from_stack(state, current_var=int(current_var))
        tried_key = (prefix_key, int(current_var))
        tried_vals = list(tried_for_state_var.get(tried_key, []))
        tried_set = set(int(v) for v in tried_vals)

        oracle_val = _oracle_polarity(int(current_var), state, clauses)
        opposite = -1 if int(oracle_val) == 1 else 1
        available = [
            v for v in [int(oracle_val), int(opposite)] if int(v) not in tried_set
        ]
        exhausted = len(available) == 0

        if any(v in tried_set for v in [1, -1]):
            stats["repeat_opportunities"] += 1
        if exhausted:
            stats["exhausted_states"] += 1
        else:
            stats["non_exhausted_states"] += 1

        next_block = int(current_block + 1)
        decision_prefix = _decision_prefix_tokens(
            include_tried=True,
            current_var=int(current_var),
            tried_vals=tried_vals,
            sorted_vars=sorted_vars,
        )
        if not _append_tokens(
            sequence, block_ids, decision_prefix, next_block, int(max_seq_len)
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

        pred_val = _decode_polarity_token(pred_token)
        pred_cf = int(pred_token) == int(TOK_CF)

        chosen_is_repeat = False
        if pred_cf:
            if exhausted:
                stats["correct_cf_on_exhausted"] += 1
                stats["backtrack_correct"] += 1
            else:
                stats["false_cf_on_non_exhausted"] += 1
                stats["backtrack_false_positive"] += 1

            if not _append_tokens(
                sequence,
                block_ids,
                [int(TOK_CF), int(TOK_OK), int(TOK_CF)],
                next_block,
                int(max_seq_len),
            ):
                stats["termination_reason"] = "budget_exceeded"
                break

            if not state.decision_stack:
                stats["termination_reason"] = "unsat_root"
                current_block = int(next_block)
                break

            top = state.decision_stack[-1]
            prior = tried_for_state_var.setdefault(tried_key, [])
            if int(top.chosen_val) not in prior:
                prior.append(int(top.chosen_val))

            top_repeat = bool(repeat_flag_stack[-1]) if repeat_flag_stack else False
            if top_repeat:
                stats["repeat_induced_backtracks"] += 1
            if repeat_flag_stack:
                repeat_flag_stack.pop()

            bt_res = env.step(SatAction.backtrack())
            if not bool(bt_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                )
                break
            stats["backtracks"] += 1
            current_block = int(next_block)
            conflict_retry_count = 0
            continue

        if pred_val is None:
            stats["invalid_predictions"] += 1
            stats["termination_reason"] = "invalid_prediction_token"
            break

        chosen_is_repeat = int(pred_val) in tried_set
        if chosen_is_repeat:
            stats["repeat_errors"] += 1

        if not exhausted and int(pred_val) == int(oracle_val):
            stats["oracle_exact_match_on_non_exhausted"] += 1
        if (
            not exhausted
            and int(pred_val) in [1, -1]
            and int(pred_val) not in tried_set
        ):
            stats["correct_polarity_on_non_exhausted"] += 1
        if exhausted:
            stats["invalid_polarity_choices"] += 1

        ok, reason = _apply_assignment_allow_repeat(
            env, int(current_var), int(pred_val)
        )
        if not ok:
            stats["termination_reason"] = f"apply_failed:{reason}"
            break

        if not _append_tokens(
            sequence,
            block_ids,
            [
                int(_polarity_token(int(pred_val))),
                int(TOK_OK),
                int(_var_token(int(current_var))),
                int(_polarity_token(int(pred_val))),
            ],
            next_block,
            int(max_seq_len),
        ):
            stats["termination_reason"] = "budget_exceeded"
            break

        repeat_flag_stack.append(bool(chosen_is_repeat))
        current_block = int(next_block)
        stats["assignments"] += 1
        conflict_retry_count = 0

        if step < 6:
            logger.info(
                "sample step=%d var=%d tried=%s pred=%d exhausted=%s oracle_val=%d",
                int(step),
                int(current_var),
                str(tried_vals),
                int(pred_token),
                str(bool(exhausted)),
                int(oracle_val),
            )

    stats["repeat_rate"] = float(
        _safe_div(stats["repeat_errors"], stats["repeat_opportunities"])
    )
    stats["bt_acc"] = float(
        _safe_div(stats["correct_cf_on_exhausted"], stats["exhausted_states"])
    )
    stats["bt_fp"] = float(
        _safe_div(stats["false_cf_on_non_exhausted"], stats["non_exhausted_states"])
    )
    stats["polarity_accuracy"] = float(
        _safe_div(
            stats["correct_polarity_on_non_exhausted"], stats["non_exhausted_states"]
        )
    )
    stats["oracle_exact_polarity_accuracy"] = float(
        _safe_div(
            stats["oracle_exact_match_on_non_exhausted"], stats["non_exhausted_states"]
        )
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAT E4 verification metrics")
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--n-vars", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="experiments/sat-e4-eval/")
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device)
    tokenizer = SATInterleavedTokenizer()

    checkpoints = [
        Path(x.strip()) for x in str(args.checkpoints).split(",") if x.strip()
    ]
    labels = [x.strip() for x in str(args.labels).split(",") if x.strip()]
    if len(checkpoints) != len(labels):
        raise ValueError("--checkpoints and --labels must have same count")
    if not checkpoints:
        raise ValueError("No checkpoints provided")

    instances = _generate_instances(
        num_instances=int(args.num_instances),
        n_vars=int(args.n_vars),
        alpha=float(args.alpha),
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
        for idx, row in enumerate(instances):
            stats = solve_instance(
                model=model,
                meta=meta,
                tokenizer=tokenizer,
                clauses=[(int(c[0]), int(c[1]), int(c[2])) for c in row["clauses"]],
                num_vars=int(row["num_vars"]),
                planted_solution=None
                if row.get("planted_solution") is None
                else np.array(row["planted_solution"], dtype=np.int64, copy=True),
                max_steps=int(args.max_steps),
                max_seq_len=int(args.budget),
                device=device,
            )
            per_instance.append(stats)

            if (idx + 1) % 25 == 0:
                logger.info(
                    "eval label=%s processed=%d/%d solve_rate=%.3f repeat=%.3f bt_acc=%.3f bt_fp=%.3f pol_acc=%.3f",
                    str(label),
                    int(idx + 1),
                    int(len(instances)),
                    float(np.mean([1.0 if s["solved"] else 0.0 for s in per_instance])),
                    float(np.mean([float(s["repeat_rate"]) for s in per_instance])),
                    float(np.mean([float(s["bt_acc"]) for s in per_instance])),
                    float(np.mean([float(s["bt_fp"]) for s in per_instance])),
                    float(
                        np.mean([float(s["polarity_accuracy"]) for s in per_instance])
                    ),
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
            "repeat_rate": float(
                _safe_div(
                    sum(float(s["repeat_errors"]) for s in per_instance),
                    sum(float(s["repeat_opportunities"]) for s in per_instance),
                )
            ),
            "bt_acc": float(
                _safe_div(
                    sum(float(s["correct_cf_on_exhausted"]) for s in per_instance),
                    sum(float(s["exhausted_states"]) for s in per_instance),
                )
            ),
            "bt_fp": float(
                _safe_div(
                    sum(float(s["false_cf_on_non_exhausted"]) for s in per_instance),
                    sum(float(s["non_exhausted_states"]) for s in per_instance),
                )
            ),
            "polarity_accuracy": float(
                _safe_div(
                    sum(
                        float(s["correct_polarity_on_non_exhausted"])
                        for s in per_instance
                    ),
                    sum(float(s["non_exhausted_states"]) for s in per_instance),
                )
            ),
            "oracle_exact_polarity_accuracy": float(
                _safe_div(
                    sum(
                        float(s["oracle_exact_match_on_non_exhausted"])
                        for s in per_instance
                    ),
                    sum(float(s["non_exhausted_states"]) for s in per_instance),
                )
            ),
            "mean_backtracks": float(
                np.mean([float(s["backtracks"]) for s in per_instance])
            ),
            "mean_steps": float(np.mean([float(s["steps"]) for s in per_instance])),
            "elapsed_sec": float(time.time() - t0),
        }

        payload = {"aggregate": aggregate, "instances": per_instance}
        all_results.append(payload)

        out_path = output_dir / f"{label}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "done label=%s solve_rate=%.3f repeat=%.3f bt_acc=%.3f bt_fp=%.3f pol_acc=%.3f elapsed=%.1fs",
            str(label),
            float(aggregate["solve_rate"]),
            float(aggregate["repeat_rate"]),
            float(aggregate["bt_acc"]),
            float(aggregate["bt_fp"]),
            float(aggregate["polarity_accuracy"]),
            float(aggregate["elapsed_sec"]),
        )

    summary_rows = [x["aggregate"] for x in all_results]
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info("wrote summary=%s", str(summary_path))


if __name__ == "__main__":
    main()
