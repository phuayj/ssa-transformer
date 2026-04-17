#!/usr/bin/env python3
"""SAT autonomous evaluation with oracle branch verification decomposition.

Conditions:
1) no_oracle       : baseline autonomous loop
2) perfect_oracle  : proactive branch extendability verification
3) noisy_<rate>    : perfect oracle + false-positive noise on extendable branches
"""

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
from sat.solvability_oracle import SolvabilityOracle
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MASKED_DOMAIN = 21


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


def _parse_noise_rates(raw: str) -> List[float]:
    vals: List[float] = []
    for token in _parse_str_list(raw):
        rate = float(token)
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"noise rate must be in [0,1], got {rate}")
        vals.append(float(rate))
    return vals


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
    model_type = str(config.get("model_type", "transformer"))
    vocab_size = int(
        config.get("vocab_size", state_dict["token_embedding.weight"].shape[0])
    )
    d_model = int(config.get("d_model", 256))
    n_layers = int(config.get("n_layers", 6))
    n_heads = int(config.get("n_heads", 8))
    n_slots = int(config.get("n_slots", 32))
    hidden_size = int(config.get("hidden_size", d_model))
    n_lstm_layers = int(config.get("n_lstm_layers", n_layers))
    block_mode = str(config.get("block_mode", "continuous"))
    max_seq_len_model = int(config.get("max_seq_len", int(max_seq_len_fallback)))
    dropout = float(config.get("dropout", 0.1))
    mask_mode = str(config.get("mask_mode", "selective_ssa"))

    model: torch.nn.Module
    if model_type == "lstm":
        from universal.lstm_decoder import LSTMDecoder

        model = LSTMDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            hidden_size=int(hidden_size),
            n_lstm_layers=int(n_lstm_layers),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
            block_mode=str(block_mode),
        )
    else:
        model = SSASlotDecoder(
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
        "model_type": str(model_type),
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
    trace_format: str = "enriched",
) -> Optional[int]:
    """Return domain token for var based on trace format."""
    domain = env._effective_domain(state, int(var_id))
    domain_set = {int(v) for v in domain}

    if len(domain_set) == 0:
        return None

    if str(trace_format) == "stripped":
        return int(MASKED_DOMAIN)

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
    lm_logits, verify_logits = model(
        input_tensor,
        block_ids=block_tensor,
        mask_mode=str(mask_mode),
    )
    _ = verify_logits

    next_logits = lm_logits[0, -1, :]
    mask = torch.full_like(next_logits, float("-inf"))
    for tok in allowed_tokens:
        if 0 <= int(tok) < int(next_logits.shape[0]):
            mask[int(tok)] = 0.0
    pred = int(torch.argmax(next_logits + mask).item())
    if pred not in allowed_tokens:
        return int(allowed_tokens[0])
    return int(pred)


def _trim_tried_levels(
    tried_values_by_level: Dict[int, set[int]], max_level: int
) -> None:
    stale = [int(k) for k in tried_values_by_level if int(k) > int(max_level)]
    for key in stale:
        tried_values_by_level.pop(int(key), None)


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
    trace_format: str,
    history_mode: str,
    oracle: Optional[SolvabilityOracle] = None,
    noise_rate: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    clauses_3 = _as_three_lit_clauses(clauses)
    clauses_for_prefix: List[Tuple[int, ...]] = [
        (int(a), int(b), int(c)) for (a, b, c) in clauses_3
    ]
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

    local_rng = rng if rng is not None else random.Random(0)

    prefix_tokens: List[int] = tokenizer.build_clause_prefix(
        clauses_for_prefix,
        int(num_vars),
    )
    sequence: List[int] = list(prefix_tokens)
    block_ids: List[int] = [0] * len(prefix_tokens)
    current_block = 0

    occurrence = _variable_occurrence_counts(clauses, int(num_vars))
    tried_values_by_level: Dict[int, set[int]] = {}

    stats: Dict[str, Any] = {
        "solved": False,
        "steps": 0,
        "decisions": 0,
        "conflicts": 0,
        "backtracks": 0,
        "post_backtrack_decisions": 0,
        "repeat_errors": 0,
        "oracle_backtracks": 0,
        "oracle_queries": 0,
        "termination_reason": "max_steps",
    }

    def _has_frontier_actions(actions: Sequence[Any]) -> Tuple[bool, bool]:
        has_assign_local = any(a.type == SatActionType.ASSIGN_VALUE for a in actions)
        has_select_local = any(a.type == SatActionType.SELECT_VAR for a in actions)
        return bool(has_assign_local), bool(has_select_local)

    def _cascade_exhausted_backtracks(block_id: int) -> bool:
        """Backtrack mechanically while current decision frame has no valid frontier actions."""
        while True:
            cascade_state = env.get_state()
            _trim_tried_levels(
                tried_values_by_level,
                int(len(cascade_state.decision_stack)),
            )

            if cascade_state.status != SatEnvStatus.RUNNING:
                return True

            cascade_actions = env.get_valid_actions()
            has_assign_local, has_select_local = _has_frontier_actions(cascade_actions)
            if has_assign_local or has_select_local:
                return True

            if not cascade_state.decision_stack:
                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_root_conflict"
                else:
                    stats["termination_reason"] = "unsat"
                return False

            if not _append_tokens(
                sequence,
                block_ids,
                [int(tokenizer.BACKJUMP)],
                int(max_seq_len),
                int(block_id),
            ):
                stats["termination_reason"] = "budget"
                return False

            bt_res = env.step(SatAction.backtrack())
            if not bool(bt_res.info.get("valid", True)):
                stats["termination_reason"] = (
                    f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                )
                return False

            stats["backtracks"] += 1
            logger.debug(
                "cascaded exhausted-frame backtrack depth=%d total_backtracks=%d",
                int(len(cascade_state.decision_stack)),
                int(stats["backtracks"]),
            )

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
                    [int(tokenizer.CONFLICT), int(tokenizer.BACKJUMP)],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break

                if not state.decision_stack:
                    done_res = env.step(SatAction.done())
                    if not bool(done_res.done):
                        stats["termination_reason"] = "failed_done_after_root_conflict"
                    else:
                        stats["termination_reason"] = "unsat"
                    break

                bt_res = env.step(SatAction.backtrack())
                if not bool(bt_res.info.get("valid", True)):
                    stats["termination_reason"] = (
                        f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                    )
                    break
                stats["backtracks"] += 1

                if not _cascade_exhausted_backtracks(int(current_block)):
                    break
                continue

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

            if str(history_mode) == "state_only":
                sequence = list(prefix_tokens)
                block_ids = [0] * len(prefix_tokens)
                current_block = 0

            current_block += 1
            state_tokens = [int(tokenizer.STATE)]
            for var_id in sorted_candidates:
                state_tokens.append(int(tokenizer.var_token(int(var_id))))
                domain_tok = _domain_token_for_var(
                    env,
                    state,
                    int(var_id),
                    tokenizer,
                    trace_format=str(trace_format),
                )
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
            has_assign, has_select = _has_frontier_actions(valid_actions)
            if not has_assign and not has_select:
                stats["termination_reason"] = "no_valid_frontier_actions"
                break

            selected_var: Optional[int] = None
            decision_level: Optional[int] = None

            if has_assign:
                selected_var_raw = state.selected_var
                if selected_var_raw is None:
                    stats["termination_reason"] = "assign_available_but_no_selected_var"
                    break
                selected_var = int(selected_var_raw)
                decision_level = int(len(state.decision_stack))

                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(tokenizer.var_token(int(selected_var)))],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break
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

                decision_level = int(len(state.decision_stack) + 1)

            if selected_var is None or decision_level is None:
                stats["termination_reason"] = "selected_var_missing"
                break

            allowed_val_tokens = [int(tokenizer.TRUE_VAL), int(tokenizer.FALSE_VAL)]
            pred_val_token = _predict_next_token(
                model=model,
                sequence=sequence,
                block_ids=block_ids,
                allowed_tokens=allowed_val_tokens,
                device=device,
                mask_mode=str(mask_mode),
            )

            chosen_value = 1 if int(pred_val_token) == int(tokenizer.TRUE_VAL) else 0
            prior_tried = tried_values_by_level.get(int(decision_level), set())
            if len(prior_tried) > 0:
                stats["post_backtrack_decisions"] += 1
                if int(chosen_value) in prior_tried:
                    stats["repeat_errors"] += 1

            if not _append_tokens(
                sequence,
                block_ids,
                [int(pred_val_token)],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "budget"
                break

            assign_res = env.step(SatAction.assign_value(int(chosen_value)))
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

            # Oracle check: after successful assign+propagate and before conflict handling.
            if oracle is not None and post_state.conflict_clause is None:
                stats["oracle_queries"] += 1
                partial: Dict[int, bool] = {}
                for var_idx in range(int(num_vars)):
                    val = int(post_state.assignment[int(var_idx)])
                    if val != 0:
                        partial[int(var_idx)] = bool(val == 1)

                extendable = bool(oracle.is_extendable(partial))
                if float(noise_rate) > 0.0 and extendable:
                    if float(local_rng.random()) < float(noise_rate):
                        extendable = False

                if not extendable:
                    stats["oracle_backtracks"] += 1
                    tried_values_by_level.setdefault(int(decision_level), set()).add(
                        int(chosen_value)
                    )

                    if not _append_tokens(
                        sequence,
                        block_ids,
                        [int(tokenizer.CONFLICT), int(tokenizer.BACKJUMP)],
                        int(max_seq_len),
                        int(current_block),
                    ):
                        stats["termination_reason"] = "budget"
                        break

                    if post_state.decision_stack:
                        bt_res = env.step(SatAction.backtrack())
                        if not bool(bt_res.info.get("valid", True)):
                            stats["termination_reason"] = (
                                f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                            )
                            break
                        stats["backtracks"] += 1
                        if not _cascade_exhausted_backtracks(int(current_block)):
                            break
                        continue

                    done_res = env.step(SatAction.done())
                    if not bool(done_res.done):
                        stats["termination_reason"] = "failed_done_after_root_conflict"
                        break
                    stats["termination_reason"] = "unsat"
                    break

            if post_state.conflict_clause is not None:
                post_level = int(len(post_state.decision_stack))
                tried_values_by_level.setdefault(post_level, set()).add(
                    int(chosen_value)
                )

                stats["conflicts"] += 1
                if not _append_tokens(
                    sequence,
                    block_ids,
                    [int(tokenizer.CONFLICT), int(tokenizer.BACKJUMP)],
                    int(max_seq_len),
                    int(current_block),
                ):
                    stats["termination_reason"] = "budget"
                    break

                if post_state.decision_stack:
                    bt_res = env.step(SatAction.backtrack())
                    if not bool(bt_res.info.get("valid", True)):
                        stats["termination_reason"] = (
                            f"invalid_backtrack:{bt_res.info.get('reason', 'unknown')}"
                        )
                        break
                    stats["backtracks"] += 1

                    if not _cascade_exhausted_backtracks(int(current_block)):
                        break
                    continue

                done_res = env.step(SatAction.done())
                if not bool(done_res.done):
                    stats["termination_reason"] = "failed_done_after_root_conflict"
                    break
                stats["termination_reason"] = "unsat"
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
    trace_format: str,
    history_mode: str,
    instances: Sequence[Dict[str, Any]],
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    oracle_condition: str,
    oracle_timeout: float,
    noise_rate: float,
    eval_seed: int,
) -> Dict[str, float]:
    per_instance: List[Dict[str, Any]] = []

    for idx, row in enumerate(instances):
        oracle: Optional[SolvabilityOracle] = None
        condition_uses_oracle = str(oracle_condition) != "no_oracle"
        rng = random.Random(int(eval_seed) * 1_000_003 + int(idx) * 9_973)
        instance_clauses: List[Tuple[int, ...]] = [
            tuple(int(x) for x in c) for c in row["clauses"]
        ]
        try:
            if condition_uses_oracle:
                oracle = SolvabilityOracle(
                    clauses=instance_clauses,
                    time_limit_sec=float(oracle_timeout),
                )

            stats = solve_instance(
                model=model,
                tokenizer=tokenizer,
                clauses=instance_clauses,
                num_vars=int(row["num_vars"]),
                planted_solution=None
                if row.get("planted_solution") is None
                else np.array(row["planted_solution"], dtype=np.int64, copy=True),
                max_steps=int(max_steps),
                max_seq_len=int(max_seq_len),
                device=device,
                mask_mode=str(mask_mode),
                trace_format=str(trace_format),
                history_mode=str(history_mode),
                oracle=oracle,
                noise_rate=float(noise_rate),
                rng=rng,
            )
        finally:
            if oracle is not None:
                oracle.close()

        per_instance.append(stats)

        if (idx + 1) % 25 == 0:
            solved = int(sum(int(item["solved"]) for item in per_instance))
            decisions = [float(item["decisions"]) for item in per_instance]
            conflicts = [float(item["conflicts"]) for item in per_instance]
            backtracks = [float(item["backtracks"]) for item in per_instance]
            oracle_backtracks = int(
                sum(int(item["oracle_backtracks"]) for item in per_instance)
            )
            oracle_queries = int(
                sum(int(item["oracle_queries"]) for item in per_instance)
            )
            repeats = int(sum(int(item["repeat_errors"]) for item in per_instance))
            revisits = int(
                sum(int(item["post_backtrack_decisions"]) for item in per_instance)
            )
            timeouts = int(
                sum(
                    1
                    for item in per_instance
                    if str(item["termination_reason"])
                    in {"max_steps", "budget", "timeout"}
                )
            )
            logger.info(
                "mask_mode=%s oracle=%s noise=%.3f processed=%d/%d solve_rate=%.3f repeat_rate=%.3f mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f oracle_backtracks=%d oracle_query_rate=%.3f timeout_rate=%.3f",
                str(mask_mode),
                str(oracle_condition),
                float(noise_rate),
                int(idx + 1),
                int(len(instances)),
                float(_safe_div(solved, len(per_instance))),
                float(_safe_div(repeats, revisits)),
                float(np.mean(decisions)) if decisions else 0.0,
                float(np.mean(conflicts)) if conflicts else 0.0,
                float(np.mean(backtracks)) if backtracks else 0.0,
                int(oracle_backtracks),
                float(_safe_div(oracle_queries, len(per_instance))),
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
    repeat_errors = int(sum(int(item["repeat_errors"]) for item in per_instance))
    post_bt_decisions = int(
        sum(int(item["post_backtrack_decisions"]) for item in per_instance)
    )
    oracle_backtracks = int(
        sum(int(item["oracle_backtracks"]) for item in per_instance)
    )
    oracle_queries = int(sum(int(item["oracle_queries"]) for item in per_instance))

    return {
        "solve_rate": float(_safe_div(solved, total)),
        "repeat_rate": float(_safe_div(repeat_errors, post_bt_decisions)),
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
        "mean_repeats_per_instance": float(
            np.mean([float(item["repeat_errors"]) for item in per_instance])
            if total
            else 0.0
        ),
        "oracle_backtracks": int(oracle_backtracks),
        "oracle_queries": int(oracle_queries),
        "mean_oracle_backtracks": float(_safe_div(oracle_backtracks, total)),
        "mean_oracle_queries": float(_safe_div(oracle_queries, total)),
    }


def _oracle_conditions(noise_rates: Sequence[float]) -> List[Tuple[str, bool, float]]:
    out: List[Tuple[str, bool, float]] = [
        ("no_oracle", False, 0.0),
        ("perfect_oracle", True, 0.0),
    ]
    for rate in noise_rates:
        out.append((f"noisy_{float(rate):g}", True, float(rate)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SAT checkpoints with oracle decomposition"
    )
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--labels", type=str, default="")
    parser.add_argument(
        "--num-instances",
        "--num_instances",
        dest="num_instances",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num-vars", "--num_vars", dest="num_vars", type=int, default=50
    )
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument(
        "--max-steps", "--max_steps", dest="max_steps", type=int, default=500
    )
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=str, required=True
    )
    parser.add_argument(
        "--mask-modes",
        "--mask_modes",
        dest="mask_modes",
        type=str,
        default="selective_ssa,full_causal",
        help="Comma-separated mask modes to evaluate",
    )
    parser.add_argument(
        "--trace-format",
        "--trace_format",
        dest="trace_format",
        type=str,
        choices=("enriched", "stripped"),
        default="enriched",
    )
    parser.add_argument(
        "--history-mode",
        "--history_mode",
        dest="history_mode",
        type=str,
        choices=("append", "state_only", "cumulative"),
        default="append",
    )
    parser.add_argument(
        "--oracle-timeout",
        "--oracle_timeout",
        dest="oracle_timeout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--noise-rates",
        "--noise_rates",
        dest="noise_rates",
        type=str,
        default="0.0,0.1,0.2",
        help="Comma-separated false-positive rates for noisy oracle conditions",
    )
    args = parser.parse_args()

    valid_mask_modes = {"selective_ssa", "blanket_ssa", "full_causal", "swa_prefix"}
    checkpoint_paths = _parse_str_list(args.checkpoints)
    if not checkpoint_paths:
        raise ValueError("--checkpoints must provide at least one path")

    parsed_labels = _parse_str_list(args.labels)
    if parsed_labels and len(parsed_labels) != len(checkpoint_paths):
        raise ValueError("--labels must match number of checkpoints when provided")

    mask_modes = _parse_str_list(args.mask_modes)
    if not mask_modes:
        raise ValueError("--mask_modes must provide at least one mask mode")
    invalid_modes = [m for m in mask_modes if m not in valid_mask_modes]
    if invalid_modes:
        raise ValueError(
            f"invalid mask mode(s): {invalid_modes}; allowed={sorted(valid_mask_modes)}"
        )

    noise_rates = _parse_noise_rates(args.noise_rates)
    history_mode = (
        "state_only" if str(args.history_mode) == "state_only" else "cumulative"
    )

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
        "generated SAT instances=%d num_vars=%d alpha=%.3f",
        int(len(shared_instances)),
        int(args.num_vars),
        float(args.alpha),
    )

    conditions = _oracle_conditions(noise_rates)
    results: List[Dict[str, Any]] = []
    started_all = time.time()

    for ckpt_idx, ckpt_raw in enumerate(checkpoint_paths):
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

        label = (
            str(parsed_labels[ckpt_idx]) if parsed_labels else str(meta["mask_mode"])
        )
        effective_seq_len = int(min(int(args.budget), int(meta["max_seq_len_model"])))

        for mask_mode in mask_modes:
            for cond_idx, (cond_name, use_oracle, noise_rate) in enumerate(conditions):
                logger.info(
                    "evaluating checkpoint=%s label=%s mask_mode=%s oracle_condition=%s noise_rate=%.3f trace_format=%s history_mode=%s budget=%d effective_seq_len=%d",
                    str(ckpt_path),
                    str(label),
                    str(mask_mode),
                    str(cond_name),
                    float(noise_rate),
                    str(args.trace_format),
                    str(args.history_mode),
                    int(args.budget),
                    int(effective_seq_len),
                )

                run_started = time.time()
                aggregate = _evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    mask_mode=str(mask_mode),
                    trace_format=str(args.trace_format),
                    history_mode=str(history_mode),
                    instances=shared_instances,
                    max_steps=int(args.max_steps),
                    max_seq_len=int(effective_seq_len),
                    device=device,
                    oracle_condition=str(cond_name if use_oracle else "no_oracle"),
                    oracle_timeout=float(args.oracle_timeout),
                    noise_rate=float(noise_rate if use_oracle else 0.0),
                    eval_seed=int(args.seed) + int(ckpt_idx) * 10000 + int(cond_idx),
                )

                row: Dict[str, Any] = {
                    "checkpoint": str(ckpt_path),
                    "label": str(label),
                    "mask_mode": str(mask_mode),
                    "trace_format": str(args.trace_format),
                    "history_mode": str(args.history_mode),
                    "oracle_condition": str(cond_name),
                    "oracle_timeout": float(args.oracle_timeout if use_oracle else 0.0),
                    "noise_rate": float(noise_rate if use_oracle else 0.0),
                    "solve_rate": float(aggregate["solve_rate"]),
                    "repeat_rate": float(aggregate["repeat_rate"]),
                    "mean_decisions": float(aggregate["mean_decisions"]),
                    "mean_conflicts": float(aggregate["mean_conflicts"]),
                    "mean_backtracks": float(aggregate["mean_backtracks"]),
                    "timeout_rate": float(aggregate["timeout_rate"]),
                    "mean_repeats_per_instance": float(
                        aggregate["mean_repeats_per_instance"]
                    ),
                    "oracle_backtracks": int(aggregate["oracle_backtracks"]),
                    "oracle_queries": int(aggregate["oracle_queries"]),
                    "mean_oracle_backtracks": float(
                        aggregate["mean_oracle_backtracks"]
                    ),
                    "mean_oracle_queries": float(aggregate["mean_oracle_queries"]),
                    "elapsed_sec": float(time.time() - run_started),
                }
                if meta.get("val_loss") is not None:
                    row["val_loss"] = float(meta["val_loss"])
                if meta.get("val_acc") is not None:
                    row["val_acc"] = float(meta["val_acc"])
                results.append(row)

                logger.info(
                    "completed checkpoint=%s mask_mode=%s oracle=%s solve_rate=%.3f mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f oracle_backtracks=%d oracle_queries=%d timeout_rate=%.3f",
                    str(ckpt_path),
                    str(mask_mode),
                    str(cond_name),
                    float(row["solve_rate"]),
                    float(row["mean_decisions"]),
                    float(row["mean_conflicts"]),
                    float(row["mean_backtracks"]),
                    int(row["oracle_backtracks"]),
                    int(row["oracle_queries"]),
                    float(row["timeout_rate"]),
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
            "mask_modes": [str(m) for m in mask_modes],
            "trace_format": str(args.trace_format),
            "history_mode": str(args.history_mode),
            "oracle_timeout": float(args.oracle_timeout),
            "noise_rates": [float(x) for x in noise_rates],
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
