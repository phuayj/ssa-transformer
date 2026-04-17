#!/usr/bin/env python3
"""On-policy DAgger training for SAT with solvability verifier.

Usage:
    python scripts/train_sat_onpolicy.py \
        --num_vars 20 \
        --alpha 4.0 \
        --num_instances 500 \
        --num_eval 200 \
        --dagger_rounds 5 \
        --oracle_rate_start 0.8 \
        --oracle_rate_end 0.1 \
        --epochs_per_round 3 \
        --device cuda:0 \
        --seed 42 \
        --output_dir experiments/sat-onpolicy
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sat.dsl import SatAction
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator
from sat.interleaved_tokenizer import SATInterleavedTokenizer
from sat.solvability_oracle import SolvabilityOracle
from universal.slot_decoder import SlotCDCLDecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MacroAction:
    kind: str  # "assign" | "backtrack"
    var: Optional[int] = None
    val: Optional[int] = None  # -1 or +1

    def __str__(self) -> str:
        if self.kind == "assign":
            return f"ASSIGN(v={self.var}, val={self.val})"
        return "BACKTRACK"


class DaggerDataset(Dataset):
    def __init__(
        self,
        sequences: List[List[int]],
        loss_masks: List[List[bool]],
        solvability_labels: List[List[int]],
        solvability_masks: List[List[bool]],
        max_seq_len: int,
    ):
        if (
            len(sequences) != len(loss_masks)
            or len(sequences) != len(solvability_labels)
            or len(sequences) != len(solvability_masks)
        ):
            raise ValueError("sequence/mask/label lengths must match")
        self.sequences = sequences
        self.loss_masks = loss_masks
        self.solvability_labels = solvability_labels
        self.solvability_masks = solvability_masks
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return int(len(self.sequences))

    def __getitem__(self, idx: int):
        seq = list(self.sequences[int(idx)])
        loss_mask = list(self.loss_masks[int(idx)])
        solv_labels = list(self.solvability_labels[int(idx)])
        solv_mask = list(self.solvability_masks[int(idx)])
        if not (len(seq) == len(loss_mask) == len(solv_labels) == len(solv_mask)):
            raise ValueError("sequence/mask length mismatch")
        if len(seq) > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len; collection should filter")
        return seq, loss_mask, solv_labels, solv_mask


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sorted_candidates_vsids(state: SatState) -> List[int]:
    unassigned = [
        int(i) for i in range(int(state.num_vars)) if int(state.assignment[int(i)]) == 0
    ]
    return sorted(
        unassigned,
        key=lambda v: (-float(state.activity[int(v)]), int(v)),
    )


def _select_var_vsids(state: SatState) -> Optional[int]:
    unassigned = np.nonzero(state.assignment == 0)[0]
    if unassigned.size == 0:
        return None
    act = state.activity[unassigned]
    best = int(unassigned[int(np.argmax(act))])
    best_act = float(state.activity[int(best)])
    for v in unassigned.tolist():
        if float(state.activity[int(v)]) > best_act + 1e-12:
            best = int(v)
            best_act = float(state.activity[int(best)])
        elif abs(float(state.activity[int(v)]) - best_act) <= 1e-12 and int(v) < best:
            best = int(v)
    return int(best)


def _state_signature(state: SatState) -> Tuple[int, ...]:
    return tuple(int(x) for x in state.assignment.tolist())


def _partial_assignment_from_state(state: SatState) -> Dict[int, bool]:
    partial: Dict[int, bool] = {}
    for idx, val in enumerate(state.assignment.tolist()):
        if int(val) == 0:
            continue
        partial[int(idx)] = bool(int(val) == 1)
    return partial


def _build_state_tokens(
    tokenizer: SATInterleavedTokenizer, sorted_candidates: List[int]
) -> List[int]:
    tokens = [int(tokenizer.STATE)]
    for v in sorted_candidates:
        tokens.append(int(tokenizer.var_token(int(v))))
        tokens.append(int(tokenizer.UNASSIGNED))
    tokens.append(int(tokenizer.SEP))
    return tokens


def _oracle_action(state: SatState, env: SatEnv, extendable: bool) -> MacroAction:
    if not bool(extendable):
        return MacroAction("backtrack")

    open_var = env._open_decision_var(state)
    if open_var is not None:
        var = int(open_var)
    else:
        var = _select_var_vsids(state)
        if var is None:
            return MacroAction("backtrack")

    dom = env._effective_domain(state, int(var))
    if not dom:
        return MacroAction("backtrack")

    val = 1 if 1 in dom else -1
    return MacroAction("assign", var=int(var), val=int(val))


def _macro_action_tokens(
    action: MacroAction, tokenizer: SATInterleavedTokenizer
) -> List[int]:
    # BACKTRACK is encoded with BACKJUMP to reuse the existing vocab.
    if action.kind == "backtrack":
        return [int(tokenizer.BACKJUMP)]
    if action.var is None or action.val is None:
        raise ValueError("assign action missing var/val")
    value_token = int(
        tokenizer.TRUE_VAL if int(action.val) == 1 else tokenizer.FALSE_VAL
    )
    return [int(tokenizer.var_token(int(action.var))), int(value_token)]


def _apply_macro_action(env: SatEnv, state: SatState, action: MacroAction):
    if action.kind == "backtrack":
        return env.step(SatAction.backtrack())

    if action.var is None or action.val is None:
        raise ValueError("assign action missing var/val")

    target_var = int(action.var)
    if state.selected_var is None or int(state.selected_var) != int(target_var):
        res = env.step(SatAction.select_var(int(target_var)))
        if res.done or not bool(res.info.get("valid", True)):
            return res
        state = env.get_state()

    assign_token = 1 if int(action.val) == 1 else 0
    return env.step(SatAction.assign_value(int(assign_token)))


@torch.no_grad()
def _predict_model_action(
    *,
    model: SlotCDCLDecoder,
    tokenizer: SATInterleavedTokenizer,
    sequence: List[int],
    state: SatState,
    env: SatEnv,
    device: torch.device,
    allow_backtrack: bool,
) -> Tuple[Optional[MacroAction], Dict[str, float]]:
    input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
    lm_logits, verify_logits = model(input_tensor)
    next_logits = lm_logits[0, -1, :]

    open_var = env._open_decision_var(state)
    if open_var is not None:
        allowed_vars = [int(open_var)]
    else:
        allowed_vars = _sorted_candidates_vsids(state)

    allowed_tokens = {int(tokenizer.var_token(int(v))) for v in allowed_vars}
    if allow_backtrack:
        allowed_tokens.add(int(tokenizer.BACKJUMP))

    if not allowed_tokens:
        return None, {
            "solvability_prob": float(
                torch.softmax(verify_logits[0, -1], dim=-1)[1].item()
            )
        }

    mask = torch.full_like(next_logits, float("-inf"))
    for tok in allowed_tokens:
        mask[int(tok)] = 0.0
    next_logits = next_logits + mask

    next_token = int(torch.argmax(next_logits).item())
    solvability_prob = float(torch.softmax(verify_logits[0, -1], dim=-1)[1].item())

    if int(next_token) == int(tokenizer.BACKJUMP):
        return MacroAction("backtrack"), {"solvability_prob": solvability_prob}

    if int(next_token) not in allowed_tokens:
        return None, {"solvability_prob": solvability_prob}

    var = int(next_token) - int(tokenizer.VAR_OFFSET)
    dom = env._effective_domain(state, int(var))
    if not dom:
        return None, {"solvability_prob": solvability_prob}

    seq2 = [*sequence, int(next_token)]
    input_tensor2 = torch.tensor([seq2], dtype=torch.long, device=device)
    lm_logits2, _verify_logits2 = model(input_tensor2)
    val_logits = lm_logits2[0, -1, :]

    val_mask = torch.full_like(val_logits, float("-inf"))
    if 1 in dom:
        val_mask[int(tokenizer.TRUE_VAL)] = 0.0
    if -1 in dom:
        val_mask[int(tokenizer.FALSE_VAL)] = 0.0
    val_logits = val_logits + val_mask
    val_token = int(torch.argmax(val_logits).item())

    if int(val_token) == int(tokenizer.TRUE_VAL):
        val = 1
    elif int(val_token) == int(tokenizer.FALSE_VAL):
        val = -1
    else:
        val = 1 if 1 in dom else -1

    return (
        MacroAction("assign", var=int(var), val=int(val)),
        {"solvability_prob": solvability_prob},
    )


def _build_sample(
    *,
    prefix_tokens: List[int],
    action: MacroAction,
    extendable: bool,
    timed_out: bool,
    tokenizer: SATInterleavedTokenizer,
) -> Tuple[List[int], List[bool], List[int], List[bool]]:
    action_tokens = _macro_action_tokens(action, tokenizer)
    sequence = list(prefix_tokens) + list(action_tokens)
    loss_mask = [False] * len(prefix_tokens) + [True] * len(action_tokens)
    solv_labels = [0 for _ in range(len(sequence))]
    solv_mask = [False for _ in range(len(sequence))]
    sep_idx = len(prefix_tokens) - 1
    if int(sep_idx) < 0 or int(sep_idx) >= int(len(sequence)):
        raise ValueError("SEP index out of bounds")
    if not bool(timed_out):
        solv_mask[int(sep_idx)] = True
        solv_labels[int(sep_idx)] = 1 if bool(extendable) else 0
    return sequence, loss_mask, solv_labels, solv_mask


def _collate_batch(
    batch: List[Tuple[List[int], List[bool], List[int], List[bool]]],
    pad_token: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(len(batch))
    max_len = max(int(len(item[0])) for item in batch)
    input_ids = torch.full((batch_size, max_len), int(pad_token), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    solvability_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    solvability_labels = torch.zeros((batch_size, max_len), dtype=torch.long)

    for idx, (seq, lm_mask, solv_labels, solv_mask) in enumerate(batch):
        seq_len = int(len(seq))
        input_ids[idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[idx, :seq_len] = 1
        loss_mask[idx, :seq_len] = torch.tensor(lm_mask, dtype=torch.bool)
        solvability_mask[idx, :seq_len] = torch.tensor(solv_mask, dtype=torch.bool)
        solvability_labels[idx, :seq_len] = torch.tensor(solv_labels, dtype=torch.long)

    return input_ids, attention_mask, loss_mask, solvability_labels, solvability_mask


def _compute_lm_loss(
    lm_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    pad_token: int,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    labels = input_ids[:, 1:].clone()
    logits = lm_logits[:, :-1, :]
    label_mask = loss_mask[:, 1:] & (attention_mask[:, 1:] > 0)
    token_count = int(label_mask.sum().item())
    if token_count == 0:
        raise RuntimeError("no masked tokens in batch")
    labels = labels.masked_fill(~label_mask, int(pad_token))
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=int(pad_token),
    )
    return loss, token_count, label_mask, labels


def _compute_solvability_loss(
    verify_logits: torch.Tensor,
    solvability_labels: torch.Tensor,
    solvability_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    if int(solvability_mask.sum().item()) == 0:
        return torch.tensor(0.0, device=verify_logits.device), 0, 0
    masked_logits = verify_logits[solvability_mask]
    masked_labels = solvability_labels[solvability_mask]
    loss = F.cross_entropy(masked_logits, masked_labels)
    preds = masked_logits.argmax(dim=-1)
    correct = int((preds == masked_labels).sum().item())
    return loss, int(masked_labels.numel()), int(correct)


def _safe_decode_tokens(
    tokenizer: SATInterleavedTokenizer, tokens: Iterable[int]
) -> List[str]:
    out = []
    for t in tokens:
        try:
            out.append(tokenizer.decode_token(int(t)))
        except ValueError:
            out.append(f"UNK({int(t)})")
    return out


def _run_epoch(
    *,
    loader: DataLoader,
    model: SlotCDCLDecoder,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    tokenizer: SATInterleavedTokenizer,
    train: bool,
    solvability_weight: float,
) -> Dict[str, float]:
    if train:
        model.train()
        if optimizer is None:
            raise ValueError("optimizer required for training")
        assert optimizer is not None
    else:
        model.eval()

    total_lm_loss = total_solv_loss = total_loss = 0.0
    total_tokens = total_sequences = 0
    total_lm_correct = 0
    total_solv_positions = 0
    total_solv_correct = 0
    logged_sample = False

    for input_ids, attention_mask, loss_mask, solv_labels, solv_mask in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device)
        solv_labels = solv_labels.to(device)
        solv_mask = solv_mask.to(device)

        with torch.set_grad_enabled(train):
            lm_logits, verify_logits = model(input_ids, attention_mask)
            lm_loss, token_count, label_mask, labels = _compute_lm_loss(
                lm_logits,
                input_ids,
                attention_mask,
                loss_mask,
                pad_token=int(tokenizer.PAD),
            )
            solv_loss, solv_positions, solv_correct = _compute_solvability_loss(
                verify_logits, solv_labels, solv_mask
            )
            loss = lm_loss + float(solvability_weight) * solv_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        preds = lm_logits[:, :-1, :].argmax(dim=-1)
        total_lm_correct += int(((preds == labels) & label_mask).sum().item())
        total_tokens += int(token_count)
        total_lm_loss += float(lm_loss.item()) * float(token_count)

        if int(solv_positions) > 0:
            total_solv_loss += float(solv_loss.item()) * float(solv_positions)
            total_solv_positions += int(solv_positions)
            total_solv_correct += int(solv_correct)

        total_loss += float(loss.item()) * float(input_ids.size(0))
        total_sequences += int(input_ids.size(0))

        if not logged_sample:
            positions = (
                torch.nonzero(label_mask[0], as_tuple=False).flatten().tolist()[:6]
            )
            if positions:
                targets = [labels[0, pos].item() for pos in positions]
                preds_txt = [preds[0, pos].item() for pos in positions]
                logger.info(
                    "sample_action_targets=%s",
                    _safe_decode_tokens(tokenizer, targets),
                )
                logger.info(
                    "sample_action_preds=%s",
                    _safe_decode_tokens(tokenizer, preds_txt),
                )

            if int(solv_mask[0].sum().item()) > 0:
                idxs = torch.nonzero(solv_mask[0], as_tuple=False).flatten().tolist()
                idxs = idxs[:4]
                if idxs:
                    logits = verify_logits[0, idxs]
                    probs = torch.softmax(logits.float(), dim=-1)[:, 1]
                    labels_txt = [int(solv_labels[0, i].item()) for i in idxs]
                    probs_txt = [float(p.item()) for p in probs]
                    logger.info("solvability_labels=%s", labels_txt)
                    logger.info("solvability_probs=%s", probs_txt)
            logged_sample = True

    lm_loss_avg = total_lm_loss / max(float(total_tokens), 1.0)
    lm_acc = float(total_lm_correct) / max(float(total_tokens), 1.0)
    solv_loss_avg = total_solv_loss / max(float(total_solv_positions), 1.0)
    solv_acc = float(total_solv_correct) / max(float(total_solv_positions), 1.0)
    avg_loss = total_loss / max(float(total_sequences), 1.0)
    return {
        "loss": float(avg_loss),
        "lm_loss": float(lm_loss_avg),
        "lm_acc": float(lm_acc),
        "solv_loss": float(solv_loss_avg),
        "solv_acc": float(solv_acc),
        "tokens": float(total_tokens),
        "solv_positions": float(total_solv_positions),
    }


def _linear_beta(round_idx: int, rounds: int, start: float, end: float) -> float:
    if int(rounds) <= 1:
        return float(start)
    t = float(round_idx) / float(max(int(rounds) - 1, 1))
    return float(start) + (float(end) - float(start)) * float(t)


def _collect_rollout(
    *,
    model: Optional[SlotCDCLDecoder],
    tokenizer: SATInterleavedTokenizer,
    clauses: List[Tuple[int, int, int]],
    num_vars: int,
    planted_solution: Optional[np.ndarray],
    beta: float,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    time_limit_sec: float,
    rng: random.Random,
    log_sample: bool,
) -> Tuple[List[Tuple[List[int], List[bool], List[int], List[bool]]], Dict[str, int]]:
    env = SatEnv(
        clauses=clauses,
        num_vars=int(num_vars),
        planted_solution=planted_solution,
        mode="soft",
        max_steps=int(max_steps) * 5 + 10,
    )
    env.reset()
    oracle = SolvabilityOracle(list(clauses), time_limit_sec=float(time_limit_sec))

    sequence = tokenizer.build_clause_prefix(clauses, int(num_vars))

    stats: Dict[str, int] = {
        "steps": 0,
        "assignments": 0,
        "backtracks": 0,
        "invalid_actions": 0,
        "oracle_actions": 0,
        "model_actions": 0,
        "model_fallbacks": 0,
        "solvability_timeouts": 0,
        "states_visited": 0,
    }
    states_seen = set()
    samples: List[Tuple[List[int], List[bool], List[int], List[bool]]] = []

    model_was_training = False
    if model is not None:
        model_was_training = bool(model.training)
        model.eval()

    for step in range(int(max_steps)):
        state = env.get_state()

        if state.status != SatEnvStatus.RUNNING:
            break

        while bool(state.propagation_pending):
            res = env.step(SatAction.propagate())
            if res.done:
                state = env.get_state()
                break
            state = env.get_state()

        if state.status != SatEnvStatus.RUNNING:
            break

        if env._all_satisfied(state) and state.conflict_clause is None:
            env.step(SatAction.done())
            break

        if state.conflict_clause is not None and not state.decision_stack:
            env.step(SatAction.done())
            break

        sorted_candidates = _sorted_candidates_vsids(state)
        if not sorted_candidates:
            break

        state_tokens = _build_state_tokens(tokenizer, sorted_candidates)
        prefix_tokens = list(sequence) + list(state_tokens)
        if int(len(prefix_tokens)) + 2 > int(max_seq_len):
            if log_sample:
                logger.info(
                    "rollout terminate: max_seq_len reached at step=%d seq_len=%d",
                    int(step),
                    int(len(prefix_tokens)),
                )
            break

        states_seen.add(_state_signature(state))

        partial = _partial_assignment_from_state(state)
        extendable = oracle.is_extendable(partial)
        timed_out = bool(oracle.last_timed_out)
        if timed_out:
            stats["solvability_timeouts"] += 1

        oracle_action = _oracle_action(state, env, extendable)
        if oracle_action.kind == "backtrack" and not state.decision_stack:
            env.step(SatAction.done())
            break
        sample = _build_sample(
            prefix_tokens=prefix_tokens,
            action=oracle_action,
            extendable=extendable,
            timed_out=timed_out,
            tokenizer=tokenizer,
        )
        if int(len(sample[0])) > int(max_seq_len):
            break
        samples.append(sample)

        use_oracle = model is None or float(rng.random()) < float(beta)
        if use_oracle:
            chosen_action = oracle_action
            stats["oracle_actions"] += 1
            action_info = {"solvability_prob": float("nan")}
        else:
            action, action_info = _predict_model_action(
                model=model,
                tokenizer=tokenizer,
                sequence=prefix_tokens,
                state=state,
                env=env,
                device=device,
                allow_backtrack=bool(state.decision_stack),
            )
            stats["model_actions"] += 1
            if action is None:
                stats["model_fallbacks"] += 1
                chosen_action = oracle_action
            else:
                chosen_action = action

        action_tokens = _macro_action_tokens(chosen_action, tokenizer)
        if int(len(prefix_tokens)) + int(len(action_tokens)) > int(max_seq_len):
            break

        sequence = list(prefix_tokens) + list(action_tokens)

        res = _apply_macro_action(env, state, chosen_action)
        stats["steps"] += 1
        if chosen_action.kind == "assign":
            stats["assignments"] += 1
        else:
            stats["backtracks"] += 1

        if not bool(res.info.get("valid", True)):
            stats["invalid_actions"] += 1
            break

        if log_sample and step < 4:
            logger.info(
                "rollout step=%d extendable=%s action=%s sol_prob=%.3f candidates=%d",
                int(step),
                str(bool(extendable)),
                str(chosen_action),
                float(action_info.get("solvability_prob", float("nan"))),
                int(len(sorted_candidates)),
            )

        if res.done:
            break

    if model is not None and model_was_training:
        model.train()

    oracle.close()
    stats["states_visited"] = int(len(states_seen))
    return samples, stats


@torch.no_grad()
def _eval_rollout(
    *,
    model: SlotCDCLDecoder,
    tokenizer: SATInterleavedTokenizer,
    clauses: List[Tuple[int, int, int]],
    num_vars: int,
    planted_solution: Optional[np.ndarray],
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    time_limit_sec: float,
) -> Tuple[Dict[str, float], set[Tuple[int, ...]]]:
    env = SatEnv(
        clauses=clauses,
        num_vars=int(num_vars),
        planted_solution=planted_solution,
        mode="soft",
        max_steps=int(max_steps) * 5 + 10,
    )
    env.reset()
    oracle = SolvabilityOracle(list(clauses), time_limit_sec=float(time_limit_sec))

    sequence = tokenizer.build_clause_prefix(clauses, int(num_vars))
    stats: Dict[str, float] = {
        "steps": 0,
        "assignments": 0,
        "backtracks": 0,
        "invalid_actions": 0,
        "wrong_branches": 0,
        "solvability_correct": 0,
        "solvability_total": 0,
        "solvability_timeouts": 0,
        "solved": 0,
    }
    states_seen: set[Tuple[int, ...]] = set()

    for step in range(int(max_steps)):
        state = env.get_state()
        if state.status != SatEnvStatus.RUNNING:
            break

        while bool(state.propagation_pending):
            res = env.step(SatAction.propagate())
            if res.done:
                state = env.get_state()
                break
            state = env.get_state()

        if state.status != SatEnvStatus.RUNNING:
            break

        if env._all_satisfied(state) and state.conflict_clause is None:
            env.step(SatAction.done())
            stats["solved"] = 1
            break

        if state.conflict_clause is not None and not state.decision_stack:
            env.step(SatAction.done())
            break

        sorted_candidates = _sorted_candidates_vsids(state)
        if not sorted_candidates:
            break

        state_tokens = _build_state_tokens(tokenizer, sorted_candidates)
        prefix_tokens = list(sequence) + list(state_tokens)
        if int(len(prefix_tokens)) + 2 > int(max_seq_len):
            break

        states_seen.add(_state_signature(state))

        partial = _partial_assignment_from_state(state)
        extendable = oracle.is_extendable(partial)
        timed_out = bool(oracle.last_timed_out)
        if timed_out:
            stats["solvability_timeouts"] += 1

        oracle_action = _oracle_action(state, env, extendable)
        if oracle_action.kind == "backtrack" and not state.decision_stack:
            env.step(SatAction.done())
            break

        action, action_info = _predict_model_action(
            model=model,
            tokenizer=tokenizer,
            sequence=prefix_tokens,
            state=state,
            env=env,
            device=device,
            allow_backtrack=bool(state.decision_stack),
        )
        if action is None:
            stats["invalid_actions"] += 1
            action = oracle_action
        if oracle_action.kind == "backtrack" and not state.decision_stack:
            env.step(SatAction.done())
            break

        if str(action) != str(oracle_action):
            stats["wrong_branches"] += 1

        if not timed_out:
            pred_extendable = bool(action_info.get("solvability_prob", 0.0) >= 0.5)
            stats["solvability_total"] += 1
            stats["solvability_correct"] += int(pred_extendable == bool(extendable))

        action_tokens = _macro_action_tokens(action, tokenizer)
        if int(len(prefix_tokens)) + int(len(action_tokens)) > int(max_seq_len):
            break

        sequence = list(prefix_tokens) + list(action_tokens)

        res = _apply_macro_action(env, state, action)
        stats["steps"] += 1
        if action.kind == "assign":
            stats["assignments"] += 1
        else:
            stats["backtracks"] += 1

        if not bool(res.info.get("valid", True)):
            stats["invalid_actions"] += 1
            break

        if res.done:
            state = env.get_state()
            stats["solved"] = int(state.status == SatEnvStatus.SUCCESS)
            break

    oracle.close()
    return stats, states_seen


def _oracle_state_set(
    *,
    clauses: List[Tuple[int, int, int]],
    num_vars: int,
    planted_solution: Optional[np.ndarray],
    max_steps: int,
    time_limit_sec: float,
) -> set[Tuple[int, ...]]:
    env = SatEnv(
        clauses=clauses,
        num_vars=int(num_vars),
        planted_solution=planted_solution,
        mode="soft",
        max_steps=int(max_steps) * 5 + 10,
    )
    env.reset()
    oracle = SolvabilityOracle(list(clauses), time_limit_sec=float(time_limit_sec))
    states_seen: set[Tuple[int, ...]] = set()

    for _ in range(int(max_steps)):
        state = env.get_state()
        if state.status != SatEnvStatus.RUNNING:
            break

        while bool(state.propagation_pending):
            res = env.step(SatAction.propagate())
            if res.done:
                state = env.get_state()
                break
            state = env.get_state()

        if state.status != SatEnvStatus.RUNNING:
            break

        if env._all_satisfied(state) and state.conflict_clause is None:
            env.step(SatAction.done())
            break

        if state.conflict_clause is not None and not state.decision_stack:
            env.step(SatAction.done())
            break

        sorted_candidates = _sorted_candidates_vsids(state)
        if not sorted_candidates:
            break

        states_seen.add(_state_signature(state))

        partial = _partial_assignment_from_state(state)
        extendable = oracle.is_extendable(partial)
        oracle_action = _oracle_action(state, env, extendable)
        if oracle_action.kind == "backtrack" and not state.decision_stack:
            env.step(SatAction.done())
            break

        res = _apply_macro_action(env, state, oracle_action)
        if res.done:
            break

    oracle.close()
    return states_seen


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return float(len(a & b)) / float(len(a | b))


def _evaluate_model(
    *,
    model: SlotCDCLDecoder,
    tokenizer: SATInterleavedTokenizer,
    num_instances: int,
    num_vars: int,
    alpha: float,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    seed: int,
    time_limit_sec: float,
) -> Dict[str, float]:
    was_training = bool(model.training)
    model.eval()
    generator = SatGenerator(seed=int(seed))
    results: List[Dict[str, float]] = []
    overlaps: List[float] = []
    wrong_instances = 0
    recovered_instances = 0
    solved_instances = 0
    backtracks_solved: List[int] = []

    for idx in range(int(num_instances)):
        instance = generator.generate_planted(
            num_vars=int(num_vars), alpha=float(alpha)
        )

        stats, model_states = _eval_rollout(
            model=model,
            tokenizer=tokenizer,
            clauses=instance.clauses,
            num_vars=int(instance.num_vars),
            planted_solution=instance.planted_solution,
            max_steps=int(max_steps),
            max_seq_len=int(max_seq_len),
            device=device,
            time_limit_sec=float(time_limit_sec),
        )

        oracle_states = _oracle_state_set(
            clauses=instance.clauses,
            num_vars=int(instance.num_vars),
            planted_solution=instance.planted_solution,
            max_steps=int(max_steps),
            time_limit_sec=float(time_limit_sec),
        )
        overlap = _jaccard(model_states, oracle_states)
        overlaps.append(float(overlap))

        if int(stats.get("wrong_branches", 0)) > 0:
            wrong_instances += 1
            if int(stats.get("solved", 0)) == 1:
                recovered_instances += 1

        if int(stats.get("solved", 0)) == 1:
            solved_instances += 1
            backtracks_solved.append(int(stats.get("backtracks", 0)))

        results.append(stats)

        if idx < 3:
            logger.info(
                "eval sample idx=%d solved=%s steps=%d backtracks=%d wrong_branches=%d overlap=%.3f",
                int(idx),
                str(bool(stats.get("solved", 0))),
                int(stats.get("steps", 0)),
                int(stats.get("backtracks", 0)),
                int(stats.get("wrong_branches", 0)),
                float(overlap),
            )

    total = float(len(results))
    solve_rate = float(solved_instances) / max(total, 1.0)
    avg_backtracks = float(np.mean([r.get("backtracks", 0) for r in results]))
    backtracks_per_solve = (
        float(np.mean(backtracks_solved)) if backtracks_solved else 0.0
    )
    solv_total = float(sum(r.get("solvability_total", 0) for r in results))
    solv_correct = float(sum(r.get("solvability_correct", 0) for r in results))
    solv_timeouts = int(sum(r.get("solvability_timeouts", 0) for r in results))
    solv_acc = float(solv_correct) / max(float(solv_total), 1.0)
    recovery_rate = (
        float(recovered_instances) / max(float(wrong_instances), 1.0)
        if wrong_instances > 0
        else 0.0
    )

    summary = {
        "solve_rate": float(solve_rate),
        "avg_backtracks": float(avg_backtracks),
        "backtracks_per_solve": float(backtracks_per_solve),
        "recovery_rate": float(recovery_rate),
        "wrong_instances": int(wrong_instances),
        "recovered_instances": int(recovered_instances),
        "state_overlap": float(np.mean(overlaps) if overlaps else 0.0),
        "solvability_accuracy": float(solv_acc),
        "solvability_total": int(solv_total),
        "solvability_timeouts": int(solv_timeouts),
        "n": int(total),
    }
    if was_training:
        model.train()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="On-policy DAgger training for SAT with solvability oracle"
    )
    parser.add_argument("--num_vars", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--num_instances", type=int, default=500)
    parser.add_argument("--num_eval", type=int, default=200)
    parser.add_argument("--dagger_rounds", type=int, default=5)
    parser.add_argument("--oracle_rate_start", type=float, default=0.8)
    parser.add_argument("--oracle_rate_end", type=float, default=0.1)
    parser.add_argument("--epochs_per_round", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments/sat-onpolicy")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_slots", type=int, default=32)
    parser.add_argument("--solvability_weight", type=float, default=2.0)
    parser.add_argument("--solvability_timeout", type=float, default=0.05)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(str(args.device))
    tokenizer = SATInterleavedTokenizer()

    if int(args.num_vars) > int(tokenizer.MAX_VARS):
        raise ValueError("num_vars exceeds tokenizer.MAX_VARS")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model: SlotCDCLDecoder
    config: Dict[str, float | int | str]
    if args.checkpoint:
        ckpt = torch.load(Path(args.checkpoint), map_location="cpu")
        config = ckpt.get("config", {})
        model = SlotCDCLDecoder(
            vocab_size=int(config.get("vocab_size", tokenizer.VOCAB_SIZE)),
            d_model=int(config.get("d_model", args.d_model)),
            n_layers=int(config.get("n_layers", args.n_layers)),
            n_heads=int(config.get("n_heads", args.n_heads)),
            max_seq_len=int(config.get("max_seq_len", args.max_seq_len)),
            n_slots=int(config.get("n_slots", args.n_slots)),
            dropout=float(config.get("dropout", args.dropout)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint: %s", str(args.checkpoint))
    else:
        config = {}
        model = SlotCDCLDecoder(
            vocab_size=int(tokenizer.VOCAB_SIZE),
            d_model=int(args.d_model),
            n_layers=int(args.n_layers),
            n_heads=int(args.n_heads),
            max_seq_len=int(args.max_seq_len),
            n_slots=int(args.n_slots),
            dropout=float(args.dropout),
        )

    model = model.to(device)
    max_seq_len = min(int(args.max_seq_len), int(model.max_seq_len))
    if int(max_seq_len) != int(args.max_seq_len):
        logger.info(
            "Clipping max_seq_len to model limit: requested=%d model=%d",
            int(args.max_seq_len),
            int(model.max_seq_len),
        )

    run_config = {
        "num_vars": int(args.num_vars),
        "alpha": float(args.alpha),
        "num_instances": int(args.num_instances),
        "num_eval": int(args.num_eval),
        "dagger_rounds": int(args.dagger_rounds),
        "oracle_rate_start": float(args.oracle_rate_start),
        "oracle_rate_end": float(args.oracle_rate_end),
        "epochs_per_round": int(args.epochs_per_round),
        "max_steps": int(args.max_steps),
        "max_seq_len": int(max_seq_len),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "warmup_fraction": float(args.warmup_fraction),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "n_slots": int(args.n_slots),
        "solvability_weight": float(args.solvability_weight),
        "solvability_timeout": float(args.solvability_timeout),
        "seed": int(args.seed),
        "device": str(args.device),
        "vocab_size": int(tokenizer.VOCAB_SIZE),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    buffer_sequences: List[List[int]] = []
    buffer_loss_masks: List[List[bool]] = []
    buffer_solv_labels: List[List[int]] = []
    buffer_solv_masks: List[List[bool]] = []

    metrics_by_round: List[Dict[str, float]] = []

    for round_idx in range(int(args.dagger_rounds)):
        beta = _linear_beta(
            round_idx,
            int(args.dagger_rounds),
            args.oracle_rate_start,
            args.oracle_rate_end,
        )
        logger.info(
            "DAgger round %d/%d beta=%.3f buffer=%d",
            int(round_idx + 1),
            int(args.dagger_rounds),
            float(beta),
            int(len(buffer_sequences)),
        )

        rng = random.Random(int(args.seed) + int(round_idx))
        generator = SatGenerator(seed=int(args.seed) + int(round_idx))

        round_samples = 0
        round_steps = 0
        round_backtracks = 0
        round_assignments = 0
        round_timeouts = 0

        start_time = time.time()
        for idx in range(int(args.num_instances)):
            instance = generator.generate_planted(
                num_vars=int(args.num_vars), alpha=float(args.alpha)
            )
            rollout_model = model if (int(round_idx) > 0 or args.checkpoint) else None
            samples, stats = _collect_rollout(
                model=rollout_model,
                tokenizer=tokenizer,
                clauses=instance.clauses,
                num_vars=int(instance.num_vars),
                planted_solution=instance.planted_solution,
                beta=float(beta),
                max_steps=int(args.max_steps),
                max_seq_len=int(max_seq_len),
                device=device,
                time_limit_sec=float(args.solvability_timeout),
                rng=rng,
                log_sample=bool(idx < 2),
            )

            for seq, loss_mask, solv_labels, solv_mask in samples:
                buffer_sequences.append(seq)
                buffer_loss_masks.append(loss_mask)
                buffer_solv_labels.append(solv_labels)
                buffer_solv_masks.append(solv_mask)

            round_samples += int(len(samples))
            round_steps += int(stats.get("steps", 0))
            round_backtracks += int(stats.get("backtracks", 0))
            round_assignments += int(stats.get("assignments", 0))
            round_timeouts += int(stats.get("solvability_timeouts", 0))

            if int(idx) < 2:
                logger.info(
                    "round %d sample idx=%d steps=%d assignments=%d backtracks=%d samples=%d",
                    int(round_idx + 1),
                    int(idx),
                    int(stats.get("steps", 0)),
                    int(stats.get("assignments", 0)),
                    int(stats.get("backtracks", 0)),
                    int(len(samples)),
                )

        elapsed = time.time() - start_time
        logger.info(
            "round %d collection done samples=%d steps=%d backtracks=%d assignments=%d timeouts=%d elapsed=%.1fs",
            int(round_idx + 1),
            int(round_samples),
            int(round_steps),
            int(round_backtracks),
            int(round_assignments),
            int(round_timeouts),
            float(elapsed),
        )

        dataset = DaggerDataset(
            buffer_sequences,
            buffer_loss_masks,
            buffer_solv_labels,
            buffer_solv_masks,
            max_seq_len=int(max_seq_len),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=lambda batch: _collate_batch(batch, int(tokenizer.PAD)),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        total_steps = int(len(loader) * int(args.epochs_per_round))
        warmup_steps = int(total_steps * float(args.warmup_fraction))

        def _lr_lambda(step: int) -> float:
            if step < int(warmup_steps):
                return float(step) / max(int(warmup_steps), 1)
            progress = float(step - int(warmup_steps)) / max(
                int(total_steps - int(warmup_steps)), 1
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

        for epoch in range(1, int(args.epochs_per_round) + 1):
            train_metrics = _run_epoch(
                loader=loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                tokenizer=tokenizer,
                train=True,
                solvability_weight=float(args.solvability_weight),
            )
            logger.info(
                "round %d epoch %d/%d train loss=%.4f lm=%.4f lm_acc=%.3f solv=%.4f solv_acc=%.3f",
                int(round_idx + 1),
                int(epoch),
                int(args.epochs_per_round),
                float(train_metrics["loss"]),
                float(train_metrics["lm_loss"]),
                float(train_metrics["lm_acc"]),
                float(train_metrics["solv_loss"]),
                float(train_metrics["solv_acc"]),
            )

        eval_metrics = _evaluate_model(
            model=model,
            tokenizer=tokenizer,
            num_instances=int(args.num_eval),
            num_vars=int(args.num_vars),
            alpha=float(args.alpha),
            max_steps=int(args.max_steps),
            max_seq_len=int(max_seq_len),
            device=device,
            seed=int(args.seed) + 10_000 + int(round_idx),
            time_limit_sec=float(args.solvability_timeout),
        )
        logger.info(
            "round %d eval solve_rate=%.3f backtracks_per_solve=%.2f overlap=%.3f solv_acc=%.3f recovery_rate=%.3f",
            int(round_idx + 1),
            float(eval_metrics["solve_rate"]),
            float(eval_metrics["backtracks_per_solve"]),
            float(eval_metrics["state_overlap"]),
            float(eval_metrics["solvability_accuracy"]),
            float(eval_metrics["recovery_rate"]),
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": int(tokenizer.VOCAB_SIZE),
                "d_model": int(args.d_model),
                "n_layers": int(args.n_layers),
                "n_heads": int(args.n_heads),
                "max_seq_len": int(max_seq_len),
                "n_slots": int(args.n_slots),
                "dropout": float(args.dropout),
                "round": int(round_idx + 1),
            },
            "eval_metrics": eval_metrics,
        }
        ckpt_path = output_dir / f"checkpoint_round_{int(round_idx + 1)}.pt"
        torch.save(checkpoint, ckpt_path)
        logger.info("Saved checkpoint: %s", str(ckpt_path))

        metrics_by_round.append(
            {
                "round": int(round_idx + 1),
                "beta": float(beta),
                "samples": int(round_samples),
                "steps": int(round_steps),
                "backtracks": int(round_backtracks),
                "assignments": int(round_assignments),
                "timeouts": int(round_timeouts),
                **eval_metrics,
            }
        )

        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"rounds": metrics_by_round}, f, indent=2)


if __name__ == "__main__":
    main()
