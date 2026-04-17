#!/usr/bin/env python3
"""Train UP-conflict verifier head and evaluate SAT backtracking.

Usage:
    python scripts/train_up_verifier.py \
        --dataset data/up_conflict.pkl \
        --num_vars 20 \
        --n_slots 32 \
        --epochs 30 \
        --batch_size 16 \
        --lr 1e-3 \
        --focal_gamma 2.0 \
        --fp_weight 5.0 \
        --device cuda:0 \
        --seed 42 \
        --output_dir experiments/up-verifier \
        --eval_instances 200
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

from sat.generator import SatGenerator
from sat.interleaved_tokenizer import SATInterleavedTokenizer
from universal.slot_decoder import SlotCDCLDecoder

try:
    from pysat.solvers import Minisat22  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - optional runtime dep
    raise ImportError(
        "pysat is required for UP verifier evaluation. Install python-sat to use it."
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Example:
    tokens: List[int]
    label: int
    label_idx: int


class VerifierDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return int(len(self.examples))

    def __getitem__(self, idx: int) -> Tuple[List[int], int, int]:
        ex = self.examples[int(idx)]
        return list(ex.tokens), int(ex.label), int(ex.label_idx)


@dataclass
class Decision:
    var: int
    tried: set[bool]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _var_scores(
    clauses: List[List[int]], num_vars: int
) -> Tuple[List[int], List[int], List[int]]:
    scores = [0 for _ in range(int(num_vars))]
    pos_scores = [0 for _ in range(int(num_vars))]
    neg_scores = [0 for _ in range(int(num_vars))]
    for clause in clauses:
        for lit in clause:
            var = abs(int(lit)) - 1
            scores[int(var)] += 1
            if int(lit) > 0:
                pos_scores[int(var)] += 1
            else:
                neg_scores[int(var)] += 1
    return scores, pos_scores, neg_scores


def _sorted_candidates(
    num_vars: int, assignment: Dict[int, bool], scores: List[int]
) -> List[int]:
    unassigned = [int(v) for v in range(int(num_vars)) if int(v) not in assignment]
    return sorted(unassigned, key=lambda v: (-int(scores[int(v)]), int(v)))


def _build_state_tokens(
    tokenizer: SATInterleavedTokenizer, sorted_candidates: List[int]
) -> List[int]:
    tokens = [int(tokenizer.STATE)]
    for v in sorted_candidates:
        tokens.append(int(tokenizer.var_token(int(v))))
        tokens.append(int(tokenizer.UNASSIGNED))
    tokens.append(int(tokenizer.SEP))
    return tokens


def _build_sequence(
    tokenizer: SATInterleavedTokenizer,
    clauses: List[List[int]],
    assumptions: List[int],
    proposed_lit: int,
    num_vars: int,
    scores: List[int],
    prefix_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]],
) -> Tuple[List[int], int]:
    clause_key = tuple(tuple(int(l) for l in clause) for clause in clauses)
    if clause_key not in prefix_cache:
        prefix_cache[clause_key] = tokenizer.build_clause_prefix(
            [tuple(int(l) for l in clause) for clause in clauses], int(num_vars)
        )
    sequence = list(prefix_cache[clause_key])

    assignment: Dict[int, bool] = {}
    for lit in assumptions:
        var = abs(int(lit)) - 1
        val = bool(int(lit) > 0)
        if int(var) in assignment:
            raise ValueError("assumption variable repeated in sequence")
        sorted_candidates = _sorted_candidates(int(num_vars), assignment, scores)
        sequence.extend(_build_state_tokens(tokenizer, sorted_candidates))
        sequence.append(int(tokenizer.var_token(int(var))))
        sequence.append(
            int(tokenizer.TRUE_VAL) if bool(val) else int(tokenizer.FALSE_VAL)
        )
        assignment[int(var)] = bool(val)

    proposed_var = abs(int(proposed_lit)) - 1
    proposed_val = bool(int(proposed_lit) > 0)
    if int(proposed_var) in assignment:
        raise ValueError("proposed literal already assigned")
    sorted_candidates = _sorted_candidates(int(num_vars), assignment, scores)
    sequence.extend(_build_state_tokens(tokenizer, sorted_candidates))
    sequence.append(int(tokenizer.var_token(int(proposed_var))))
    sequence.append(
        int(tokenizer.TRUE_VAL) if bool(proposed_val) else int(tokenizer.FALSE_VAL)
    )
    label_idx = int(len(sequence) - 1)
    return sequence, int(label_idx)


def _collate_batch(
    batch: List[Tuple[List[int], int, int]], pad_token: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(len(seq)) for seq, _, _ in batch)
    batch_size = int(len(batch))
    input_ids = torch.full((batch_size, max_len), int(pad_token), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    label_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.zeros((batch_size, max_len), dtype=torch.long)

    for idx, (seq, label, label_idx) in enumerate(batch):
        seq_len = int(len(seq))
        input_ids[idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[idx, :seq_len] = 1
        label_mask[idx, int(label_idx)] = True
        labels[idx, int(label_idx)] = int(label)

    return input_ids, attention_mask, labels, label_mask


def _asymmetric_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    fp_weight: float,
) -> torch.Tensor:
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    if int(masked_labels.numel()) == 0:
        return torch.tensor(0.0, device=logits.device)

    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = torch.exp(log_probs)
    idx = torch.arange(masked_labels.size(0), device=logits.device)
    pt = probs[idx, masked_labels]
    ce = -log_probs[idx, masked_labels]
    alpha = torch.where(
        masked_labels == 0,
        torch.full_like(ce, float(fp_weight)),
        torch.ones_like(ce),
    )
    loss = alpha * torch.pow(1.0 - pt, float(gamma)) * ce
    return loss.mean()


def _compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    if int(masked_labels.numel()) == 0:
        return {
            "positions": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }
    probs = torch.softmax(masked_logits.float(), dim=-1)[:, 1]
    preds = probs >= float(threshold)
    labels_conflict = masked_labels == 1

    tp = int((preds & labels_conflict).sum().item())
    fp = int((preds & ~labels_conflict).sum().item())
    fn = int((~preds & labels_conflict).sum().item())
    tn = int((~preds & ~labels_conflict).sum().item())

    precision = float(tp) / max(float(tp + fp), 1.0)
    recall = float(tp) / max(float(tp + fn), 1.0)
    f1 = (
        2.0 * precision * recall / max(float(precision + recall), 1e-12)
        if (precision + recall) > 0.0
        else 0.0
    )
    accuracy = float(tp + tn) / max(float(tp + tn + fp + fn), 1.0)

    return {
        "positions": float(masked_labels.numel()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _run_epoch(
    *,
    loader: DataLoader,
    model: SlotCDCLDecoder,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    gamma: float,
    fp_weight: float,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
        if optimizer is None:
            raise ValueError("optimizer required for training")
    else:
        model.eval()

    total_loss = 0.0
    total_positions = 0
    total_tp = total_fp = total_fn = total_tn = 0
    logged_sample = False

    for input_ids, attention_mask, labels, label_mask in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        label_mask = label_mask.to(device)

        with torch.set_grad_enabled(train):
            _, verify_logits = model(input_ids, attention_mask)
            loss = _asymmetric_focal_loss(
                verify_logits, labels, label_mask, gamma=gamma, fp_weight=fp_weight
            )

            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        metrics = _compute_metrics(verify_logits, labels, label_mask, threshold=0.5)
        positions = int(metrics["positions"])
        total_positions += int(positions)
        total_loss += float(loss.item()) * float(positions)
        total_tp += int(metrics["tp"])
        total_fp += int(metrics["fp"])
        total_fn += int(metrics["fn"])
        total_tn += int(metrics["tn"])

        if not logged_sample and int(positions) > 0:
            probs = torch.softmax(verify_logits[label_mask].float(), dim=-1)[:, 1]
            labels_txt = labels[label_mask].tolist()[:8]
            probs_txt = [float(p) for p in probs.tolist()[:8]]
            logger.info("sample_conflict_labels=%s", labels_txt)
            logger.info("sample_conflict_probs=%s", probs_txt)
            logged_sample = True

    precision = float(total_tp) / max(float(total_tp + total_fp), 1.0)
    recall = float(total_tp) / max(float(total_tp + total_fn), 1.0)
    f1 = (
        2.0 * precision * recall / max(float(precision + recall), 1e-12)
        if (precision + recall) > 0.0
        else 0.0
    )
    accuracy = float(total_tp + total_tn) / max(float(total_positions), 1.0)
    avg_loss = float(total_loss) / max(float(total_positions), 1.0)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positions": float(total_positions),
    }


def _collect_probs(
    loader: DataLoader, model: SlotCDCLDecoder, device: torch.device
) -> Tuple[List[float], List[int]]:
    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels, label_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            label_mask = label_mask.to(device)
            _, verify_logits = model(input_ids, attention_mask)
            masked_logits = verify_logits[label_mask]
            masked_labels = batch_labels[label_mask]
            if int(masked_labels.numel()) == 0:
                continue
            batch_probs = torch.softmax(masked_logits.float(), dim=-1)[:, 1]
            probs.extend([float(p) for p in batch_probs.tolist()])
            labels.extend([int(x) for x in masked_labels.tolist()])
    return probs, labels


def _clause_status(clause: Iterable[int], assignment: Dict[int, bool]) -> int:
    has_unassigned = False
    for lit in clause:
        var = abs(int(lit)) - 1
        if int(var) not in assignment:
            has_unassigned = True
            continue
        val = bool(assignment[int(var)])
        if (int(lit) > 0 and bool(val)) or (int(lit) < 0 and not bool(val)):
            return 1
    if bool(has_unassigned):
        return 0
    return -1


def _is_conflict(clauses: List[List[int]], assignment: Dict[int, bool]) -> bool:
    return any(int(_clause_status(clause, assignment)) == -1 for clause in clauses)


def _is_satisfied(clauses: List[List[int]], assignment: Dict[int, bool]) -> bool:
    return all(int(_clause_status(clause, assignment)) == 1 for clause in clauses)


def _select_var(
    num_vars: int, assignment: Dict[int, bool], scores: List[int]
) -> int | None:
    unassigned = [int(v) for v in range(int(num_vars)) if int(v) not in assignment]
    if not unassigned:
        return None
    return int(max(unassigned, key=lambda v: (int(scores[int(v)]), -int(v))))


def _select_polarity(
    var: int, pos_scores: List[int], neg_scores: List[int], rng: random.Random
) -> bool:
    if int(pos_scores[int(var)]) > int(neg_scores[int(var)]):
        return True
    if int(neg_scores[int(var)]) > int(pos_scores[int(var)]):
        return False
    return bool(rng.choice([True, False]))


@torch.no_grad()
def _predict_conflict_prob(
    *,
    model: SlotCDCLDecoder,
    tokenizer: SATInterleavedTokenizer,
    clauses: List[List[int]],
    assumptions: List[int],
    proposed_lit: int,
    num_vars: int,
    scores: List[int],
    prefix_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]],
    device: torch.device,
    max_seq_len: int,
) -> float:
    sequence, label_idx = _build_sequence(
        tokenizer,
        clauses,
        assumptions,
        int(proposed_lit),
        int(num_vars),
        scores,
        prefix_cache,
    )
    if int(len(sequence)) > int(max_seq_len):
        raise RuntimeError(
            f"sequence length {len(sequence)} exceeds max_seq_len {max_seq_len}"
        )
    input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
    _, verify_logits = model(input_tensor)
    logits = verify_logits[0, int(label_idx)]
    prob = torch.softmax(logits.float(), dim=-1)[1].item()
    return float(prob)


def _check_literal(
    *,
    mode: str,
    lit: int,
    clauses: List[List[int]],
    assumptions: List[int],
    solver: Minisat22 | None,
    model: SlotCDCLDecoder | None,
    tokenizer: SATInterleavedTokenizer,
    num_vars: int,
    scores: List[int],
    prefix_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]],
    device: torch.device,
    threshold: float,
    max_seq_len: int,
) -> Tuple[bool, float]:
    if str(mode) == "greedy":
        return True, float("nan")
    if str(mode) == "oracle_up":
        if solver is None:
            raise RuntimeError("oracle_up requires a solver")
        ok, _implied = solver.propagate(assumptions=list(assumptions) + [int(lit)])
        return bool(ok), float("nan")
    if str(mode) == "oracle_full":
        if solver is None:
            raise RuntimeError("oracle_full requires a solver")
        ok = solver.solve(assumptions=list(assumptions) + [int(lit)])
        return bool(ok), float("nan")
    if str(mode) == "up_verifier":
        if model is None:
            raise RuntimeError("up_verifier requires a model")
        prob = _predict_conflict_prob(
            model=model,
            tokenizer=tokenizer,
            clauses=clauses,
            assumptions=assumptions,
            proposed_lit=int(lit),
            num_vars=int(num_vars),
            scores=scores,
            prefix_cache=prefix_cache,
            device=device,
            max_seq_len=int(max_seq_len),
        )
        return bool(float(prob) <= float(threshold)), float(prob)
    raise ValueError(f"unknown mode: {mode}")


def _backtrack(
    *,
    decision_stack: List[Decision],
    assignment: Dict[int, bool],
    trail: List[int],
    mode: str,
    clauses: List[List[int]],
    solver: Minisat22 | None,
    model: SlotCDCLDecoder | None,
    tokenizer: SATInterleavedTokenizer,
    num_vars: int,
    scores: List[int],
    prefix_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]],
    device: torch.device,
    threshold: float,
    max_seq_len: int,
) -> Tuple[bool, float]:
    while decision_stack:
        last = decision_stack.pop()
        var = int(last.var)
        if trail:
            trail.pop()
        if int(var) in assignment:
            assignment.pop(int(var))

        for alt_val in (True, False):
            if bool(alt_val) in last.tried:
                continue
            alt_lit = int(var + 1) if bool(alt_val) else -(int(var) + 1)
            allowed, prob = _check_literal(
                mode=mode,
                lit=int(alt_lit),
                clauses=clauses,
                assumptions=list(trail),
                solver=solver,
                model=model,
                tokenizer=tokenizer,
                num_vars=int(num_vars),
                scores=scores,
                prefix_cache=prefix_cache,
                device=device,
                threshold=float(threshold),
                max_seq_len=int(max_seq_len),
            )
            last.tried.add(bool(alt_val))
            if bool(allowed):
                assignment[int(var)] = bool(alt_val)
                trail.append(int(alt_lit))
                decision_stack.append(last)
                return True, float(prob)
    return False, float("nan")


def _eval_rollout(
    *,
    model: SlotCDCLDecoder | None,
    tokenizer: SATInterleavedTokenizer,
    clauses: List[List[int]],
    num_vars: int,
    mode: str,
    threshold: float,
    max_steps: int,
    device: torch.device,
    rng: random.Random,
    max_seq_len: int,
) -> Dict[str, float]:
    assignment: Dict[int, bool] = {}
    trail: List[int] = []
    decision_stack: List[Decision] = []
    scores, pos_scores, neg_scores = _var_scores(clauses, int(num_vars))
    prefix_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]] = {}
    solver = (
        Minisat22(bootstrap_with=[list(c) for c in clauses])
        if str(mode) in ("oracle_up", "oracle_full")
        else None
    )

    stats: Dict[str, float] = {
        "steps": 0,
        "backtracks": 0,
        "backtracked": 0,
        "solved": 0,
    }

    for step in range(int(max_steps)):
        if _is_satisfied(clauses, assignment):
            stats["solved"] = 1
            break

        if _is_conflict(clauses, assignment):
            ok, _prob = _backtrack(
                decision_stack=decision_stack,
                assignment=assignment,
                trail=trail,
                mode=mode,
                clauses=clauses,
                solver=solver,
                model=model,
                tokenizer=tokenizer,
                num_vars=int(num_vars),
                scores=scores,
                prefix_cache=prefix_cache,
                device=device,
                threshold=float(threshold),
                max_seq_len=int(max_seq_len),
            )
            stats["backtracks"] += 1
            stats["backtracked"] = 1
            if not bool(ok):
                break
            continue

        if int(len(assignment)) == int(num_vars):
            ok, _prob = _backtrack(
                decision_stack=decision_stack,
                assignment=assignment,
                trail=trail,
                mode=mode,
                clauses=clauses,
                solver=solver,
                model=model,
                tokenizer=tokenizer,
                num_vars=int(num_vars),
                scores=scores,
                prefix_cache=prefix_cache,
                device=device,
                threshold=float(threshold),
                max_seq_len=int(max_seq_len),
            )
            stats["backtracks"] += 1
            stats["backtracked"] = 1
            if not bool(ok):
                break
            continue

        var = _select_var(int(num_vars), assignment, scores)
        if var is None:
            break

        proposed_val = _select_polarity(int(var), pos_scores, neg_scores, rng)
        proposed_lit = int(var + 1) if bool(proposed_val) else -(int(var) + 1)

        allowed, prob = _check_literal(
            mode=mode,
            lit=int(proposed_lit),
            clauses=clauses,
            assumptions=list(trail),
            solver=solver,
            model=model,
            tokenizer=tokenizer,
            num_vars=int(num_vars),
            scores=scores,
            prefix_cache=prefix_cache,
            device=device,
            threshold=float(threshold),
            max_seq_len=int(max_seq_len),
        )

        tried: set[bool] = set()
        if not bool(allowed):
            tried.add(bool(proposed_val))
            alt_val = not bool(proposed_val)
            alt_lit = int(var + 1) if bool(alt_val) else -(int(var) + 1)
            allowed_alt, alt_prob = _check_literal(
                mode=mode,
                lit=int(alt_lit),
                clauses=clauses,
                assumptions=list(trail),
                solver=solver,
                model=model,
                tokenizer=tokenizer,
                num_vars=int(num_vars),
                scores=scores,
                prefix_cache=prefix_cache,
                device=device,
                threshold=float(threshold),
                max_seq_len=int(max_seq_len),
            )
            tried.add(bool(alt_val))
            if not bool(allowed_alt):
                ok, _ = _backtrack(
                    decision_stack=decision_stack,
                    assignment=assignment,
                    trail=trail,
                    mode=mode,
                    clauses=clauses,
                    solver=solver,
                    model=model,
                    tokenizer=tokenizer,
                    num_vars=int(num_vars),
                    scores=scores,
                    prefix_cache=prefix_cache,
                    device=device,
                    threshold=float(threshold),
                    max_seq_len=int(max_seq_len),
                )
                stats["backtracks"] += 1
                stats["backtracked"] = 1
                if not bool(ok):
                    break
                continue
            chosen_val = bool(alt_val)
            prob = float(alt_prob)
        else:
            chosen_val = bool(proposed_val)
            tried.add(bool(proposed_val))

        chosen_lit = int(var + 1) if bool(chosen_val) else -(int(var) + 1)
        assignment[int(var)] = bool(chosen_val)
        trail.append(int(chosen_lit))
        decision_stack.append(Decision(var=int(var), tried=tried))
        stats["steps"] += 1

        if int(step) < 3:
            logger.info(
                "mode=%s step=%d var=%d val=%d prob=%.3f",
                str(mode),
                int(step),
                int(var),
                1 if bool(chosen_val) else 0,
                float(prob),
            )

    if solver is not None:
        solver.delete()
    return stats


def _summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    total = float(len(results))
    solved = int(sum(int(r.get("solved", 0)) for r in results))
    solve_rate = float(solved) / max(float(total), 1.0)
    backtracks = float(sum(float(r.get("backtracks", 0)) for r in results))
    backtracks_per_solve = float(backtracks) / max(float(solved), 1.0)
    backtracked_instances = int(sum(int(r.get("backtracked", 0)) for r in results))
    recovered_instances = int(
        sum(
            1
            for r in results
            if int(r.get("backtracked", 0)) > 0 and int(r.get("solved", 0)) > 0
        )
    )
    recovery_rate = float(recovered_instances) / max(float(backtracked_instances), 1.0)
    solved_with_backtracks = int(
        sum(
            1
            for r in results
            if int(r.get("solved", 0)) > 0 and int(r.get("backtracks", 0)) > 0
        )
    )

    return {
        "solve_rate": float(solve_rate),
        "backtracks_per_solve": float(backtracks_per_solve),
        "recovery_rate": float(recovery_rate),
        "backtracked_instances": int(backtracked_instances),
        "recovered_instances": int(recovered_instances),
        "solved_with_backtracks": int(solved_with_backtracks),
        "n": int(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train UP-conflict verifier and evaluate backtracking"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_vars", type=int, default=20)
    parser.add_argument("--n_slots", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--fp_weight", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_instances", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    tokenizer = SATInterleavedTokenizer()
    device = torch.device(str(args.device))

    if int(args.num_vars) > int(tokenizer.MAX_VARS):
        raise ValueError("num_vars exceeds tokenizer.MAX_VARS")

    dataset_path = Path(args.dataset)
    with dataset_path.open("rb") as f:
        payload = pickle.load(f)

    examples_raw = payload.get("examples", [])
    stats_raw = payload.get("stats", {})
    config_raw = payload.get("config", {})

    logger.info(
        "loaded dataset examples=%d conflict_rate=%.3f",
        int(len(examples_raw)),
        float(stats_raw.get("conflict_rate", 0.0)),
    )

    prefix_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]] = {}
    score_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]] = {}
    examples: List[Example] = []
    dropped = 0

    for ex in examples_raw:
        clauses = ex["clauses"]
        assumptions = [int(l) for l in ex["assumptions"]]
        proposed_lit = int(ex["proposed_lit"])
        label = 1 if bool(ex["up_conflict"]) else 0

        clause_key = tuple(tuple(int(l) for l in clause) for clause in clauses)
        if clause_key not in score_cache:
            scores, _pos, _neg = _var_scores(clauses, int(args.num_vars))
            score_cache[clause_key] = scores
        scores = score_cache[clause_key]

        try:
            sequence, label_idx = _build_sequence(
                tokenizer,
                clauses,
                assumptions,
                proposed_lit,
                int(args.num_vars),
                scores,
                prefix_cache,
            )
        except ValueError:
            dropped += 1
            continue

        if int(len(sequence)) > int(args.max_seq_len):
            dropped += 1
            continue
        examples.append(
            Example(tokens=sequence, label=int(label), label_idx=int(label_idx))
        )

    if dropped > 0:
        logger.warning("dropped %d examples (sequence too long/invalid)", int(dropped))

    rng.shuffle(examples)
    val_size = max(int(0.1 * len(examples)), 1)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    logger.info(
        "examples=%d train=%d val=%d",
        int(len(examples)),
        int(len(train_examples)),
        int(val_size),
    )

    train_loader = DataLoader(
        VerifierDataset(train_examples),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda b: _collate_batch(b, int(tokenizer.PAD)),
    )
    val_loader = DataLoader(
        VerifierDataset(val_examples),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=lambda b: _collate_batch(b, int(tokenizer.PAD)),
    )

    model = SlotCDCLDecoder(
        vocab_size=int(tokenizer.VOCAB_SIZE),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        max_seq_len=int(args.max_seq_len),
        n_slots=int(args.n_slots),
        dropout=float(args.dropout),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": str(dataset_path),
        "num_vars": int(args.num_vars),
        "n_slots": int(args.n_slots),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "focal_gamma": float(args.focal_gamma),
        "fp_weight": float(args.fp_weight),
        "seed": int(args.seed),
        "device": str(args.device),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "max_seq_len": int(args.max_seq_len),
        "dropout": float(args.dropout),
        "eval_instances": int(args.eval_instances),
        "alpha": float(args.alpha) if args.alpha is not None else None,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    best_precision = -1.0
    best_epoch = -1
    training_log: List[Dict[str, float]] = []

    for epoch in range(int(args.epochs)):
        train_metrics = _run_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            gamma=float(args.focal_gamma),
            fp_weight=float(args.fp_weight),
            train=True,
        )
        val_metrics = _run_epoch(
            loader=val_loader,
            model=model,
            optimizer=None,
            device=device,
            gamma=float(args.focal_gamma),
            fp_weight=float(args.fp_weight),
            train=False,
        )

        record = {
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_f1": float(train_metrics["f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
        }
        training_log.append(record)

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_precision=%.3f val_recall=%.3f val_f1=%.3f",
            int(epoch),
            float(record["train_loss"]),
            float(record["val_loss"]),
            float(record["val_precision"]),
            float(record["val_recall"]),
            float(record["val_f1"]),
        )

        if float(record["val_precision"]) > float(best_precision):
            best_precision = float(record["val_precision"])
            best_epoch = int(epoch)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": int(tokenizer.VOCAB_SIZE),
                    "d_model": int(args.d_model),
                    "n_layers": int(args.n_layers),
                    "n_heads": int(args.n_heads),
                    "max_seq_len": int(args.max_seq_len),
                    "n_slots": int(args.n_slots),
                    "dropout": float(args.dropout),
                    "dataset": str(dataset_path),
                    "num_vars": int(args.num_vars),
                },
                "epoch": int(epoch),
                "val_precision": float(best_precision),
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            logger.info(
                "saved best_model.pt (epoch=%d precision=%.3f)",
                int(epoch),
                float(best_precision),
            )

    with (output_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    if best_epoch < 0:
        raise RuntimeError("no valid checkpoint saved")

    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    probs, labels = _collect_probs(val_loader, model, device)
    thresholds = [round(x * 0.01, 2) for x in range(1, 100)]
    curve: List[Dict[str, float]] = []

    for tau in thresholds:
        tp = fp = fn = 0
        for p, y in zip(probs, labels):
            pred = float(p) > float(tau)
            if bool(pred) and int(y) == 1:
                tp += 1
            elif bool(pred) and int(y) == 0:
                fp += 1
            elif (not bool(pred)) and int(y) == 1:
                fn += 1
        precision = float(tp) / max(float(tp + fp), 1.0)
        recall = float(tp) / max(float(tp + fn), 1.0)
        curve.append(
            {
                "tau": float(tau),
                "precision": float(precision),
                "recall": float(recall),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }
        )

    candidates = [c for c in curve if c["precision"] >= 0.9]
    if not candidates:
        best_tau = max(curve, key=lambda c: float(c["precision"]))
        meets_precision = False
    else:
        best_tau = max(
            candidates, key=lambda c: (float(c["recall"]), float(c["precision"]))
        )
        meets_precision = True

    calibration = {
        "selected_tau": float(best_tau["tau"]),
        "curve": curve,
        "criteria": "precision>=0.9",
        "meets_precision": bool(meets_precision),
        "best_epoch": int(best_epoch),
        "best_val_precision": float(best_precision),
    }

    with (output_dir / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    logger.info(
        "calibrated tau=%.2f precision=%.3f recall=%.3f meets_precision=%s",
        float(best_tau["tau"]),
        float(best_tau["precision"]),
        float(best_tau["recall"]),
        str(bool(meets_precision)),
    )

    model.eval()

    eval_alpha = (
        float(args.alpha)
        if args.alpha is not None
        else float(config_raw.get("alpha", 4.0))
    )
    generator = SatGenerator(seed=int(args.seed) + 10_000)
    instances = [
        generator.generate_planted(num_vars=int(args.num_vars), alpha=float(eval_alpha))
        for _ in range(int(args.eval_instances))
    ]

    eval_max_seq_len = min(int(args.max_seq_len), int(model.max_seq_len))
    if int(eval_max_seq_len) != int(args.max_seq_len):
        logger.info(
            "Clipping max_seq_len to model limit: requested=%d model=%d",
            int(args.max_seq_len),
            int(model.max_seq_len),
        )

    results: Dict[str, List[Dict[str, float]]] = {}
    for mode in ("greedy", "up_verifier", "oracle_up", "oracle_full"):
        mode_results: List[Dict[str, float]] = []
        for idx, inst in enumerate(instances):
            stats = _eval_rollout(
                model=model if mode == "up_verifier" else None,
                tokenizer=tokenizer,
                clauses=[list(c) for c in inst.clauses],
                num_vars=int(inst.num_vars),
                mode=str(mode),
                threshold=float(best_tau["tau"]),
                max_steps=int(args.max_steps),
                device=device,
                rng=rng,
                max_seq_len=int(eval_max_seq_len),
            )
            stats["instance_id"] = int(idx)
            mode_results.append(stats)
            if idx < 3:
                logger.info(
                    "mode=%s idx=%d solved=%s backtracks=%d",
                    str(mode),
                    int(idx),
                    str(bool(stats.get("solved", 0))),
                    int(stats.get("backtracks", 0)),
                )
        results[str(mode)] = mode_results

    greedy_summary = _summarize(results["greedy"])
    verifier_summary = _summarize(results["up_verifier"])
    oracle_up_summary = _summarize(results["oracle_up"])
    oracle_full_summary = _summarize(results["oracle_full"])

    logger.info(
        "greedy solve_rate=%.3f backtracks_per_solve=%.2f",
        float(greedy_summary["solve_rate"]),
        float(greedy_summary["backtracks_per_solve"]),
    )
    logger.info(
        "up_verifier solve_rate=%.3f backtracks_per_solve=%.2f recovery_rate=%.3f",
        float(verifier_summary["solve_rate"]),
        float(verifier_summary["backtracks_per_solve"]),
        float(verifier_summary["recovery_rate"]),
    )
    logger.info(
        "oracle_up solve_rate=%.3f backtracks_per_solve=%.2f",
        float(oracle_up_summary["solve_rate"]),
        float(oracle_up_summary["backtracks_per_solve"]),
    )
    logger.info(
        "oracle_full solve_rate=%.3f backtracks_per_solve=%.2f",
        float(oracle_full_summary["solve_rate"]),
        float(oracle_full_summary["backtracks_per_solve"]),
    )

    acceptance = (
        float(verifier_summary["solve_rate"]) > float(greedy_summary["solve_rate"])
        and int(verifier_summary["solved_with_backtracks"]) > 0
    )
    if acceptance:
        logger.info(
            "✅ up_verifier > greedy and backtracking triggered (solves_with_backtracks=%d)",
            int(verifier_summary["solved_with_backtracks"]),
        )
    else:
        logger.info("❌ up_verifier did not exceed greedy or no backtracking benefit")

    eval_output = {
        "dataset": str(dataset_path),
        "threshold": float(best_tau["tau"]),
        "precision_at_tau": float(best_tau["precision"]),
        "recall_at_tau": float(best_tau["recall"]),
        "meets_precision": bool(meets_precision),
        "greedy": greedy_summary,
        "up_verifier": verifier_summary,
        "oracle_up": oracle_up_summary,
        "oracle_full": oracle_full_summary,
        "acceptance": bool(acceptance),
    }

    with (output_dir / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)


if __name__ == "__main__":
    main()
