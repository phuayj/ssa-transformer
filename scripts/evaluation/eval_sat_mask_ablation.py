#!/usr/bin/env python3
"""Evaluate SAT closed-loop performance across mask-mode SSA checkpoints."""

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
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.dsl import SatAction, SatActionType
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator
from sat.interleaved_tokenizer import SATInterleavedTokenizer
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _parse_str_list(raw: str) -> List[str]:
    return [str(x.strip()) for x in str(raw).split(",") if str(x).strip()]


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
    num_vars: int,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    generator = SatGenerator(seed=int(seed))
    rows: List[Dict[str, Any]] = []
    for _ in range(int(num_instances)):
        inst = generator.generate_planted(num_vars=int(num_vars), alpha=float(alpha))
        rows.append(
            {
                "clauses": [tuple(int(x) for x in clause) for clause in inst.clauses],
                "num_vars": int(inst.num_vars),
                "planted_solution": None
                if inst.planted_solution is None
                else np.array(inst.planted_solution, dtype=np.int64, copy=True),
            }
        )
    return rows


def _variable_occurrence_counts(
    clauses: Sequence[Tuple[int, ...]],
    num_vars: int,
) -> np.ndarray:
    counts = np.zeros((int(num_vars),), dtype=np.int64)
    for clause in clauses:
        for lit in clause:
            var = int(abs(int(lit)) - 1)
            if 0 <= var < int(num_vars):
                counts[var] += 1
    return counts


def _as_three_lit_clauses(
    clauses: Sequence[Tuple[int, ...]],
) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for clause in clauses:
        if len(clause) != 3:
            raise ValueError(f"Expected 3-literal clause, got len={len(clause)}")
        out.append((int(clause[0]), int(clause[1]), int(clause[2])))
    return out


def _sorted_unassigned_vars(
    env: SatEnv,
    state: SatState,
    occurrence: np.ndarray,
) -> List[int]:
    _ = env
    unassigned = [
        int(v) for v in range(int(state.num_vars)) if int(state.assignment[int(v)]) == 0
    ]
    return sorted(
        unassigned,
        key=lambda v: (
            -float(state.activity[int(v)]),
            -int(occurrence[int(v)]),
            int(v),
        ),
    )


def _domain_token_for_var(
    env: SatEnv,
    state: SatState,
    var_id: int,
    tokenizer: SATInterleavedTokenizer,
) -> Optional[int]:
    """Return domain token for var: UNASSIGNED, TRUE_VAL, FALSE_VAL, or None."""
    domain = env._effective_domain(state, int(var_id))
    if domain is None:
        domain_set: set[int] = set()
    else:
        domain_set = {int(v) for v in domain}

    if len(domain_set) == 0:
        return None
    if domain_set == {-1, 1} or domain_set == {0, 1}:
        return int(tokenizer.UNASSIGNED)
    if 1 in domain_set and -1 not in domain_set:
        return int(tokenizer.TRUE_VAL)
    if -1 in domain_set and 1 not in domain_set:
        return int(tokenizer.FALSE_VAL)
    return int(tokenizer.UNASSIGNED)


def _predict_next_token(
    model: torch.nn.Module,
    sequence: List[int],
    block_ids: List[int],
    allowed_tokens: Sequence[int],
    device: torch.device,
    mask_mode: str,
) -> int:
    if not allowed_tokens:
        raise ValueError("allowed_tokens must be non-empty")

    input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
    block_tensor = torch.tensor([block_ids], dtype=torch.long, device=device)
    lm_logits, _ = model(
        input_tensor,
        block_ids=block_tensor,
        mask_mode=str(mask_mode),
    )
    next_logits = lm_logits[0, -1, :]
    mask = torch.full_like(next_logits, float("-inf"))
    for tok in allowed_tokens:
        if 0 <= int(tok) < int(next_logits.shape[0]):
            mask[int(tok)] = 0.0
    pred = int(torch.argmax(next_logits + mask).item())
    if pred not in allowed_tokens:
        return int(allowed_tokens[0])
    return int(pred)


def _backtrack_after_conflict(
    *,
    env: SatEnv,
    tokenizer: SATInterleavedTokenizer,
    sequence: List[int],
    block_ids: List[int],
    max_seq_len: int,
    current_block: int,
    stats: Dict[str, Any],
) -> bool:
    """Backtrack at least once, then keep popping exhausted/conflicting frames.

    Returns True if caller should terminate the solve loop, else False.
    """
    bt_res = env.step(SatAction.backtrack())
    if not bool(bt_res.info.get("valid", True)):
        stats["termination_reason"] = (
            f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
        )
        return True

    stats["backtracks"] += 1
    if not _append_tokens(
        sequence,
        block_ids,
        [int(tokenizer.BACKJUMP)],
        int(max_seq_len),
        int(current_block),
    ):
        stats["termination_reason"] = "budget"
        return True

    while True:
        post_bt_state = env.get_state()
        if post_bt_state.status != SatEnvStatus.RUNNING:
            stats["solved"] = bool(post_bt_state.status == SatEnvStatus.SUCCESS)
            stats["termination_reason"] = str(
                post_bt_state.termination_reason or "env_done"
            )
            return True

        if post_bt_state.conflict_clause is not None:
            if not post_bt_state.decision_stack:
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_root_conflict"
                else:
                    stats["termination_reason"] = "unsat_root_conflict"
                return True

            bt_res = env.step(SatAction.backtrack())
            if not bool(bt_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                )
                return True

            stats["backtracks"] += 1
            if not _append_tokens(
                sequence,
                block_ids,
                [int(tokenizer.BACKJUMP)],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "budget"
                return True
            continue

        valid_actions = env.get_valid_actions()
        has_assign = any(a.type == SatActionType.ASSIGN_VALUE for a in valid_actions)
        has_select = any(a.type == SatActionType.SELECT_VAR for a in valid_actions)

        if has_assign or has_select:
            return False

        if not post_bt_state.decision_stack:
            done_res = env.step(SatAction.done())
            if not bool(done_res.done):
                stats["termination_reason"] = "failed_done_after_exhausted"
            else:
                stats["termination_reason"] = "unsat_exhausted"
            return True

        bt_res = env.step(SatAction.backtrack())
        if not bool(bt_res.info.get("valid", True)):
            stats["termination_reason"] = (
                f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
            )
            return True

        stats["backtracks"] += 1
        if not _append_tokens(
            sequence,
            block_ids,
            [int(tokenizer.BACKJUMP)],
            int(max_seq_len),
            int(current_block),
        ):
            stats["termination_reason"] = "budget"
            return True


def solve_instance(
    *,
    model: torch.nn.Module,
    tokenizer: SATInterleavedTokenizer,
    clauses: List[Tuple[int, ...]],
    num_vars: int,
    planted_solution: Optional[np.ndarray],
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    mask_mode: str,
) -> Dict[str, Any]:
    clauses_3 = _as_three_lit_clauses(clauses)
    env = SatEnv(
        clauses=clauses_3,
        num_vars=int(num_vars),
        planted_solution=None
        if planted_solution is None
        else np.array(planted_solution, dtype=np.int64, copy=True),
        mode="strict",
        max_steps=int(max_steps * 8 + 20),
    )
    env.reset()

    sequence: List[int] = tokenizer.build_clause_prefix(
        clauses_3,
        int(num_vars),
    )
    block_ids: List[int] = [0] * len(sequence)
    current_block = 0

    occurrence = _variable_occurrence_counts(clauses, int(num_vars))

    stats: Dict[str, Any] = {
        "solved": False,
        "steps": 0,
        "decisions": 0,
        "conflicts": 0,
        "backtracks": 0,
        "termination_reason": "max_steps",
    }

    with torch.no_grad():
        for step in range(int(max_steps)):
            stats["steps"] = int(step + 1)
            state = env.get_state()

            if state.status != SatEnvStatus.RUNNING:
                stats["solved"] = bool(state.status == SatEnvStatus.SUCCESS)
                stats["termination_reason"] = str(
                    state.termination_reason or "env_done"
                )
                break

            if bool(state.propagation_pending):
                prop_res = env.step(SatAction.propagate())
                if not bool(prop_res.info.get("valid", True)):
                    stats["termination_reason"] = (
                        f"invalid_propagate:{prop_res.info.get('reason', 'unknown')}"
                    )
                    break
                state = env.get_state()

            if state.status != SatEnvStatus.RUNNING:
                stats["solved"] = bool(state.status == SatEnvStatus.SUCCESS)
                stats["termination_reason"] = str(
                    state.termination_reason or "env_done"
                )
                break

            if state.conflict_clause is not None:
                stats["conflicts"] += 1
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(tokenizer.CONFLICT)],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break

                if state.decision_stack:
                    stop_now = _backtrack_after_conflict(
                        env=env,
                        tokenizer=tokenizer,
                        sequence=sequence,
                        block_ids=block_ids,
                        max_seq_len=int(max_seq_len),
                        current_block=int(current_block),
                        stats=stats,
                    )
                    if stop_now:
                        break
                    continue

                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_root_conflict"
                    break
                stats["termination_reason"] = "unsat_root_conflict"
                break

            if env._all_satisfied(state):
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(tokenizer.SOLVED)],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_solved"
                    break
                stats["solved"] = True
                stats["termination_reason"] = "solved"
                break

            sorted_candidates = _sorted_unassigned_vars(env, state, occurrence)
            if not sorted_candidates:
                stats["termination_reason"] = "no_unassigned_candidates"
                break

            current_block += 1
            state_tokens = [int(tokenizer.STATE)]
            for var_id in sorted_candidates:
                state_tokens.append(int(tokenizer.var_token(int(var_id))))
                domain_tok = _domain_token_for_var(env, state, int(var_id), tokenizer)
                if domain_tok is not None:
                    state_tokens.append(int(domain_tok))
                else:
                    state_tokens.pop()
            state_tokens.append(int(tokenizer.SEP))
            if not _append_tokens(
                sequence,
                block_ids,
                state_tokens,
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "budget"
                break

            valid_actions = env.get_valid_actions()
            has_assign = any(
                a.type == SatActionType.ASSIGN_VALUE for a in valid_actions
            )
            has_select = any(a.type == SatActionType.SELECT_VAR for a in valid_actions)
            if not has_assign and not has_select:
                stats["termination_reason"] = "no_valid_frontier_actions"
                break

            selected_var: Optional[int] = None
            valid_values: List[int] = []

            if has_assign:
                selected_var_raw = env.get_state().selected_var
                if selected_var_raw is None:
                    stats["termination_reason"] = "assign_available_but_no_selected_var"
                    break
                selected_var = int(selected_var_raw)
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(tokenizer.var_token(int(selected_var)))],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break
                valid_values = [
                    int(a.target)
                    for a in valid_actions
                    if a.type == SatActionType.ASSIGN_VALUE and a.target is not None
                ]
            else:
                selectable_vars = [
                    int(a.target)
                    for a in valid_actions
                    if a.type == SatActionType.SELECT_VAR and a.target is not None
                ]
                if not selectable_vars:
                    stats["termination_reason"] = "no_valid_select_actions"
                    break

                allowed_var_tokens = [
                    int(tokenizer.var_token(int(var_id))) for var_id in selectable_vars
                ]
                pred_var_token = _predict_next_token(
                    model=model,
                    sequence=sequence,
                    block_ids=block_ids,
                    allowed_tokens=allowed_var_tokens,
                    device=device,
                    mask_mode=str(mask_mode),
                )

                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(pred_var_token)],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break

                selected_var = int(pred_var_token) - int(tokenizer.VAR_OFFSET)
                if selected_var not in selectable_vars:
                    selected_var = int(selectable_vars[0])

                select_res = env.step(SatAction.select_var(int(selected_var)))
                if not bool(select_res.info.get("valid", True)):
                    stats["termination_reason"] = (
                        f"invalid_select:{select_res.info.get('reason', 'unknown')}"
                    )
                    break

                valid_actions = env.get_valid_actions()
                valid_values = []
                for a in valid_actions:
                    if a.type == SatActionType.ASSIGN_VALUE and a.target is not None:
                        valid_values.append(int(a.target))

            if selected_var is None:
                stats["termination_reason"] = "selected_var_missing"
                break

            if not valid_values:
                stats["termination_reason"] = "no_valid_assign_values"
                break

            allowed_val_tokens: List[int] = []
            for val in valid_values:
                if int(val) == 1:
                    allowed_val_tokens.append(int(tokenizer.TRUE_VAL))
                else:
                    allowed_val_tokens.append(int(tokenizer.FALSE_VAL))

            if not allowed_val_tokens:
                stats["termination_reason"] = "no_valid_value_tokens"
                break

            pred_val_token = _predict_next_token(
                model=model,
                sequence=sequence,
                block_ids=block_ids,
                allowed_tokens=allowed_val_tokens,
                device=device,
                mask_mode=str(mask_mode),
            )

            if not _append_tokens(
                sequence,
                block_ids,
                [int(pred_val_token)],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "budget"
                break

            val_target = 1 if int(pred_val_token) == int(tokenizer.TRUE_VAL) else 0
            assign_res = env.step(SatAction.assign_value(int(val_target)))
            if not bool(assign_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_assign:{assign_res.info.get('reason', 'unknown')}"
                )
                break

            stats["decisions"] += 1

            prop_res = env.step(SatAction.propagate())
            if not bool(prop_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_propagate:{prop_res.info.get('reason', 'unknown')}"
                )
                break

            post_state = env.get_state()
            if post_state.conflict_clause is not None:
                stats["conflicts"] += 1
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(tokenizer.CONFLICT)],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break
                if post_state.decision_stack:
                    stop_now = _backtrack_after_conflict(
                        env=env,
                        tokenizer=tokenizer,
                        sequence=sequence,
                        block_ids=block_ids,
                        max_seq_len=int(max_seq_len),
                        current_block=int(current_block),
                        stats=stats,
                    )
                    if stop_now:
                        break
                    continue

                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_root_conflict"
                    break
                stats["termination_reason"] = "unsat_root_conflict"
                break

            if env._all_satisfied(post_state):
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(tokenizer.SOLVED)],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_solved"
                    break
                stats["solved"] = True
                stats["termination_reason"] = "solved"
                break

            if not _append_tokens(
                sequence,
                block_ids,
                [
                    int(tokenizer.OK),
                    int(tokenizer.var_token(int(selected_var))),
                    int(pred_val_token),
                ],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "budget"
                break

    return stats


def _evaluate_model(
    *,
    model: torch.nn.Module,
    tokenizer: SATInterleavedTokenizer,
    mask_mode: str,
    instances: Sequence[Dict[str, Any]],
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, float]:
    per_instance: List[Dict[str, Any]] = []

    for idx, row in enumerate(instances):
        stats = solve_instance(
            model=model,
            tokenizer=tokenizer,
            clauses=[tuple(int(x) for x in c) for c in row["clauses"]],
            num_vars=int(row["num_vars"]),
            planted_solution=None
            if row.get("planted_solution") is None
            else np.array(row["planted_solution"], dtype=np.int64, copy=True),
            max_steps=int(max_steps),
            max_seq_len=int(max_seq_len),
            device=device,
            mask_mode=str(mask_mode),
        )
        per_instance.append(stats)

        if (idx + 1) % 25 == 0:
            solved = int(sum(int(item["solved"]) for item in per_instance))
            decisions = [float(item["decisions"]) for item in per_instance]
            conflicts = [float(item["conflicts"]) for item in per_instance]
            backtracks = [float(item["backtracks"]) for item in per_instance]
            timeouts = int(
                sum(
                    1
                    for item in per_instance
                    if str(item["termination_reason"])
                    in {"max_steps", "budget", "timeout"}
                )
            )
            logger.info(
                "mask_mode=%s processed=%d/%d solve_rate=%.3f mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f timeout_rate=%.3f",
                str(mask_mode),
                int(idx + 1),
                int(len(instances)),
                float(_safe_div(solved, len(per_instance))),
                float(np.mean(decisions)) if decisions else 0.0,
                float(np.mean(conflicts)) if conflicts else 0.0,
                float(np.mean(backtracks)) if backtracks else 0.0,
                float(_safe_div(timeouts, len(per_instance))),
            )

    total = int(len(per_instance))
    solved = int(sum(int(item["solved"]) for item in per_instance))
    timeout_count = int(
        sum(
            1
            for item in per_instance
            if str(item["termination_reason"]) in {"max_steps", "budget", "timeout"}
        )
    )
    return {
        "solve_rate": float(_safe_div(solved, total)),
        "mean_decisions": float(
            np.mean([float(item["decisions"]) for item in per_instance])
            if total
            else 0.0
        ),
        "mean_conflicts": float(
            np.mean([float(item["conflicts"]) for item in per_instance])
            if total
            else 0.0
        ),
        "mean_backtracks": float(
            np.mean([float(item["backtracks"]) for item in per_instance])
            if total
            else 0.0
        ),
        "timeout_rate": float(_safe_div(timeout_count, total)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SAT mask-mode ablations from SSA checkpoints"
    )
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--num-vars", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=3.5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    checkpoint_paths = _parse_str_list(args.checkpoints)
    if not checkpoint_paths:
        raise ValueError("--checkpoints must provide at least one path")

    parsed_labels = _parse_str_list(args.labels)
    if parsed_labels and len(parsed_labels) != len(checkpoint_paths):
        raise ValueError("--labels must match number of checkpoints when provided")

    _set_seed(int(args.seed))
    tokenizer = SATInterleavedTokenizer()
    device = torch.device(str(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_instances = _generate_instances(
        num_instances=int(args.num_instances),
        num_vars=int(args.num_vars),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )
    logger.info(
        "generated shared SAT instances=%d num_vars=%d alpha=%.3f",
        int(len(shared_instances)),
        int(args.num_vars),
        float(args.alpha),
    )

    results: List[Dict[str, Any]] = []
    started_all = time.time()

    for idx, ckpt_raw in enumerate(checkpoint_paths):
        ckpt_path = Path(ckpt_raw)
        logger.info("loading checkpoint=%s", str(ckpt_path))
        model, meta = _load_checkpoint(
            checkpoint_path=ckpt_path,
            device=device,
            max_seq_len_fallback=int(args.budget),
        )

        if int(meta["vocab_size"]) != int(tokenizer.VOCAB_SIZE):
            logger.warning(
                "checkpoint vocab_size=%d differs from SATInterleavedTokenizer.VOCAB_SIZE=%d",
                int(meta["vocab_size"]),
                int(tokenizer.VOCAB_SIZE),
            )

        mask_mode = str(meta["mask_mode"])
        label = str(parsed_labels[idx]) if parsed_labels else str(mask_mode)
        effective_seq_len = int(min(int(args.budget), int(meta["max_seq_len_model"])))

        logger.info(
            "evaluating label=%s mask_mode=%s vocab=%d budget=%d effective_seq_len=%d",
            label,
            mask_mode,
            int(meta["vocab_size"]),
            int(args.budget),
            int(effective_seq_len),
        )

        run_started = time.time()
        aggregate = _evaluate_model(
            model=model,
            tokenizer=tokenizer,
            mask_mode=mask_mode,
            instances=shared_instances,
            max_steps=int(args.max_steps),
            max_seq_len=int(effective_seq_len),
            device=device,
        )

        row: Dict[str, Any] = {
            "checkpoint": str(ckpt_path),
            "mask_mode": str(mask_mode),
            "label": str(label),
            "solve_rate": float(aggregate["solve_rate"]),
            "mean_decisions": float(aggregate["mean_decisions"]),
            "mean_conflicts": float(aggregate["mean_conflicts"]),
            "mean_backtracks": float(aggregate["mean_backtracks"]),
            "timeout_rate": float(aggregate["timeout_rate"]),
            "elapsed_sec": float(time.time() - run_started),
        }
        if meta.get("val_loss") is not None:
            row["val_loss"] = float(meta["val_loss"])
        if meta.get("val_acc") is not None:
            row["val_acc"] = float(meta["val_acc"])
        results.append(row)

        logger.info(
            "completed label=%s solve_rate=%.3f mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f timeout_rate=%.3f val_loss=%s val_acc=%s",
            label,
            float(row["solve_rate"]),
            float(row["mean_decisions"]),
            float(row["mean_conflicts"]),
            float(row["mean_backtracks"]),
            float(row["timeout_rate"]),
            "n/a" if "val_loss" not in row else f"{float(row['val_loss']):.4f}",
            "n/a" if "val_acc" not in row else f"{float(row['val_acc']):.4f}",
        )

    payload: Dict[str, Any] = {
        "config": {
            "checkpoints": checkpoint_paths,
            "labels": parsed_labels,
            "num_instances": int(args.num_instances),
            "num_vars": int(args.num_vars),
            "alpha": float(args.alpha),
            "max_steps": int(args.max_steps),
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
