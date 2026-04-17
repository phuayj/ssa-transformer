#!/usr/bin/env python3
"""Evaluate SAT policy with controlled oracle corruption noise."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

try:
    from pysat.solvers import Minisat22  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pysat is required for corruption evaluation (install python-sat)."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sat.dense_viability_net import DenseViabilityNet, SharedMLP
from sat.dsl import SatAction
from sat.env import SatEnv, SatEnvStatus, SatState
from sat.generator import SatGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


SWEEP_RATES: Tuple[float, ...] = (
    0.0,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _extract_dense_features(
    clauses: List[List[int]], trail: List[int], num_vars: int
) -> np.ndarray:
    n_vars = int(num_vars)
    clause_arr = np.asarray(clauses, dtype=np.int64)
    n_clauses = int(clause_arr.shape[0])

    assigned_vals = np.zeros((n_vars,), dtype=np.int8)
    for lit in trail:
        assigned_vals[abs(int(lit)) - 1] = 1 if int(lit) > 0 else -1

    clause_vars = np.abs(clause_arr) - 1
    clause_signs = np.where(clause_arr > 0, 1, -1).astype(np.int8)

    base_var_vals = assigned_vals[clause_vars]
    base_assigned = base_var_vals != 0
    base_lit_vals = np.where(
        base_assigned, np.where(base_var_vals == clause_signs, 1, -1), 0
    ).astype(np.int8)

    cand_var = np.repeat(np.arange(n_vars, dtype=np.int64), 2)[:, None, None]
    cand_val = np.tile(np.array([-1, 1], dtype=np.int8), n_vars)[:, None, None]

    clause_vars_b = clause_vars[None, :, :]
    clause_signs_b = clause_signs[None, :, :]
    base_assigned_b = base_assigned[None, :, :]
    base_lit_vals_b = base_lit_vals[None, :, :]

    assigned_by_candidate = (~base_assigned_b) & (clause_vars_b == cand_var)
    cand_lit_vals = np.where(clause_signs_b == cand_val, 1, -1).astype(np.int8)
    lit_vals = np.where(
        base_assigned_b,
        base_lit_vals_b,
        np.where(assigned_by_candidate, cand_lit_vals, 0),
    )

    has_true = np.any(lit_vals == 1, axis=2)
    n_unset = np.sum(lit_vals == 0, axis=2)

    n_satisfied = np.sum(has_true, axis=1).astype(np.float32)
    n_empty = np.sum((~has_true) & (n_unset == 0), axis=1).astype(np.float32)
    n_unit = np.sum((~has_true) & (n_unset == 1), axis=1).astype(np.float32)
    n_unresolved = np.sum((~has_true) & (n_unset > 1), axis=1).astype(np.float32)

    pos_occ = np.zeros((n_vars,), dtype=np.float32)
    neg_occ = np.zeros((n_vars,), dtype=np.float32)
    flat_lits = clause_arr.reshape(-1)
    flat_vars = np.abs(flat_lits) - 1
    np.add.at(pos_occ, flat_vars[flat_lits > 0], 1.0)
    np.add.at(neg_occ, flat_vars[flat_lits < 0], 1.0)

    frac_assigned = float(np.count_nonzero(assigned_vals)) / float(max(n_vars, 1))
    clause_count = float(max(n_clauses, 1))

    feats = np.zeros((2 * n_vars, 9), dtype=np.float32)
    feats[:, 0] = float(frac_assigned)
    feats[:, 1] = float(n_clauses)
    feats[:, 2] = n_satisfied / clause_count
    feats[:, 3] = n_unit / clause_count
    feats[:, 4] = n_empty / clause_count
    feats[:, 5] = n_unresolved / clause_count

    cand_var_flat = np.repeat(np.arange(n_vars, dtype=np.int64), 2)
    feats[:, 6] = pos_occ[cand_var_flat] / clause_count
    feats[:, 7] = neg_occ[cand_var_flat] / clause_count
    feats[:, 8] = np.tile(np.array([0.0, 1.0], dtype=np.float32), n_vars)

    return feats.reshape(n_vars, 2, 9)


def _infer_dense(
    model: nn.Module,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = (features - mean[None, None, :]) / std[None, None, :]
    t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)[0]
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs.astype(np.float32)


def _apply_assignment(env: SatEnv, state: SatState, var: int, val_sign: int) -> bool:
    res = env.step(SatAction.select_var(int(var)))
    if not bool(res.info.get("valid", True)):
        return False
    assign_token = 1 if int(val_sign) == 1 else 0
    res = env.step(SatAction.assign_value(int(assign_token)))
    return bool(res.info.get("valid", True))


def _is_clause_currently_unsatisfied(clause: List[int], assignment: np.ndarray) -> bool:
    for lit in clause:
        var_idx = abs(int(lit)) - 1
        var_val = int(assignment[var_idx])
        if (int(lit) > 0 and var_val == 1) or (int(lit) < 0 and var_val == -1):
            return False
    return True


def _select_vsids_variable(
    clauses: List[List[int]], state: SatState, unassigned: List[int]
) -> int:
    assignment = np.asarray(state.assignment, dtype=np.int8)
    var_counts = np.zeros((len(unassigned),), dtype=np.int32)
    var_to_local = {int(v): int(i) for i, v in enumerate(unassigned)}

    for clause in clauses:
        if not _is_clause_currently_unsatisfied(clause, assignment):
            continue
        for lit in clause:
            var0 = abs(int(lit)) - 1
            local_idx = var_to_local.get(int(var0))
            if local_idx is not None:
                var_counts[int(local_idx)] += 1

    best_key: Optional[Tuple[int, float, int]] = None
    best_var = int(unassigned[0])
    for local_idx, var in enumerate(unassigned):
        key = (
            int(var_counts[int(local_idx)]),
            float(state.activity[int(var)]),
            -int(var),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_var = int(var)
    return int(best_var)


def _select_vsids_polarity(clauses: List[List[int]], state: SatState, var: int) -> int:
    assignment = np.asarray(state.assignment, dtype=np.int8)
    true_gain = 0
    false_gain = 0
    pos_lit = int(var) + 1
    neg_lit = -int(var + 1)

    for clause in clauses:
        if not _is_clause_currently_unsatisfied(clause, assignment):
            continue
        if int(pos_lit) in clause:
            true_gain += 1
        if int(neg_lit) in clause:
            false_gain += 1

    return 1 if int(true_gain) >= int(false_gain) else -1


def check_satisfiable_with_trail(clauses: List[List[int]], trail: List[int]) -> bool:
    with Minisat22() as solver:
        for clause in clauses:
            solver.add_clause([int(l) for l in clause])
        for lit in trail:
            solver.add_clause([int(lit)])
        return bool(solver.solve())


def corrupted_oracle_check(
    clauses: List[List[int]],
    trail: List[int],
    rng: random.Random,
    fp_rate: float,
    fn_rate: float,
) -> Tuple[bool, bool, bool, bool]:
    """Return corrupted SAT answer + corruption accounting flags.

    Returns:
        decision_is_sat, true_is_sat, did_fp_corrupt, did_fn_corrupt
    """
    true_is_sat = check_satisfiable_with_trail(clauses, trail)
    decision_is_sat = bool(true_is_sat)
    did_fp_corrupt = False
    did_fn_corrupt = False

    if bool(true_is_sat) and float(fp_rate) > 0.0:
        if float(rng.random()) < float(fp_rate):
            decision_is_sat = False
            did_fp_corrupt = True
    elif (not bool(true_is_sat)) and float(fn_rate) > 0.0:
        if float(rng.random()) < float(fn_rate):
            decision_is_sat = True
            did_fn_corrupt = True

    return (
        decision_is_sat,
        bool(true_is_sat),
        bool(did_fp_corrupt),
        bool(did_fn_corrupt),
    )


def _run_solver(
    *,
    clauses: List[List[int]],
    num_vars: int,
    max_steps: int,
    max_backtracks: int,
    policy_rng: random.Random,
    corruption_rng: random.Random,
    policy: str,
    model: Optional[nn.Module],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    device: torch.device,
    fp_rate: float,
    fn_rate: float,
) -> Dict[str, float]:
    env = SatEnv(
        clauses=[(int(c[0]), int(c[1]), int(c[2])) for c in clauses],
        num_vars=int(num_vars),
        mode="soft",
        max_steps=int(max_steps) * 6 + 10,
    )
    env.reset()

    stats: Dict[str, float] = {
        "solved": 0,
        "steps": 0,
        "assignments": 0,
        "backtracks": 0,
        "proactive_backtracks": 0,
        "backtracked": 0,
        "oracle_checks": 0,
        "oracle_unsat": 0,
        "fp_opportunities": 0,
        "fn_opportunities": 0,
        "actual_fp_count": 0,
        "actual_fn_count": 0,
        "total_flagged_dead": 0,
        "total_true_dead": 0,
        "true_dead_flagged": 0,
    }

    for step in range(int(max_steps)):
        state = env.get_state()

        while bool(state.propagation_pending):
            env.step(SatAction.propagate())
            state = env.get_state()

        if state.status != SatEnvStatus.RUNNING:
            break

        if env._all_satisfied(state) and state.conflict_clause is None:
            env.step(SatAction.done())
            state = env.get_state()
            stats["solved"] = int(state.status == SatEnvStatus.SUCCESS)
            break

        if state.conflict_clause is not None:
            if not state.decision_stack or int(stats["backtracks"]) >= int(
                max_backtracks
            ):
                env.step(SatAction.done())
                break
            env.step(SatAction.backtrack())
            stats["backtracks"] += 1
            stats["backtracked"] = 1
            continue

        unassigned = [
            v for v in range(int(num_vars)) if int(state.assignment[int(v)]) == 0
        ]
        if not unassigned:
            env.step(SatAction.done())
            break

        trail = [int(x) for x in state.trail]
        use_model_policy = str(policy) == "model"
        if bool(use_model_policy):
            if model is None or mean is None or std is None:
                raise ValueError("model policy requires model checkpoint artifacts")
            features = _extract_dense_features(clauses, trail, int(num_vars))
            viability = _infer_dense(model, features, mean, std, device)
        else:
            viability = None

        open_var = env._open_decision_var(state)
        if open_var is not None:
            selected_var = int(open_var)
        else:
            if bool(use_model_policy):
                best_key: Tuple[float, float, int, int] | None = None
                best_choice: Tuple[int, int] | None = None
                assert viability is not None
                for v in unassigned:
                    for pol_idx, val_sign in enumerate([-1, 1]):
                        score = float(viability[int(v), int(pol_idx)])
                        key = (
                            float(score),
                            float(state.activity[int(v)]),
                            -int(v),
                            int(pol_idx),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_choice = (int(v), int(val_sign))
                if best_choice is None:
                    selected_var = int(unassigned[0])
                    selected_val = int(policy_rng.choice([-1, 1]))
                else:
                    selected_var, selected_val = best_choice
            else:
                selected_var = _select_vsids_variable(
                    clauses=clauses,
                    state=state,
                    unassigned=unassigned,
                )

        if bool(use_model_policy):
            assert viability is not None
            var_scores = viability[int(selected_var)]
            best_pol_idx = 1 if float(var_scores[1]) >= float(var_scores[0]) else 0
            preferred_val = 1 if int(best_pol_idx) == 1 else -1
        else:
            preferred_val = _select_vsids_polarity(
                clauses=clauses,
                state=state,
                var=int(selected_var),
            )

        if open_var is not None:
            dom = env._effective_domain(state, int(selected_var))
            if 1 in dom and -1 in dom:
                selected_val = int(preferred_val)
            elif 1 in dom:
                selected_val = 1
            elif -1 in dom:
                selected_val = -1
            else:
                if state.decision_stack and int(stats["backtracks"]) < int(
                    max_backtracks
                ):
                    env.step(SatAction.backtrack())
                    stats["backtracks"] += 1
                    stats["backtracked"] = 1
                    continue
                env.step(SatAction.done())
                break
        else:
            selected_val = int(preferred_val)

        ok = _apply_assignment(env, state, int(selected_var), int(selected_val))
        if not bool(ok):
            if state.decision_stack and int(stats["backtracks"]) < int(max_backtracks):
                env.step(SatAction.backtrack())
                stats["backtracks"] += 1
                stats["backtracked"] = 1
                continue
            break

        stats["steps"] += 1
        stats["assignments"] += 1

        look_state = env.get_state()
        while bool(look_state.propagation_pending):
            env.step(SatAction.propagate())
            look_state = env.get_state()

        if (
            look_state.status == SatEnvStatus.RUNNING
            and look_state.conflict_clause is None
        ):
            look_trail = [int(x) for x in look_state.trail]
            stats["oracle_checks"] += 1
            decision_is_sat, true_is_sat, did_fp, did_fn = corrupted_oracle_check(
                clauses=clauses,
                trail=look_trail,
                rng=corruption_rng,
                fp_rate=float(fp_rate),
                fn_rate=float(fn_rate),
            )

            if bool(true_is_sat):
                stats["fp_opportunities"] += 1
            else:
                stats["fn_opportunities"] += 1
                stats["total_true_dead"] += 1

            if bool(did_fp):
                stats["actual_fp_count"] += 1
            if bool(did_fn):
                stats["actual_fn_count"] += 1

            if not bool(decision_is_sat):
                stats["oracle_unsat"] += 1
                stats["total_flagged_dead"] += 1
                if not bool(true_is_sat):
                    stats["true_dead_flagged"] += 1

                if look_state.decision_stack and int(stats["backtracks"]) < int(
                    max_backtracks
                ):
                    env.step(SatAction.backtrack())
                    stats["backtracks"] += 1
                    stats["proactive_backtracks"] += 1
                    stats["backtracked"] = 1
                    if int(step) < 5:
                        logger.info(
                            "corrupted_oracle_backtrack step=%d var=%d val=%d trail=%d true_sat=%s did_fp=%s did_fn=%s",
                            int(step),
                            int(selected_var),
                            int(selected_val),
                            int(len(look_trail)),
                            str(bool(true_is_sat)),
                            str(bool(did_fp)),
                            str(bool(did_fn)),
                        )
                    continue

        if int(step) < 3:
            logger.info(
                "step=%d var=%d val=%d trail=%d oracle_checks=%d fp=%.3f fn=%.3f",
                int(step),
                int(selected_var),
                int(selected_val),
                int(len(env.get_state().trail)),
                int(stats["oracle_checks"]),
                float(fp_rate),
                float(fn_rate),
            )

    return stats


def _summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    n = float(len(results))
    solved = int(sum(int(r.get("solved", 0)) for r in results))
    solve_rate = float(solved) / max(float(n), 1.0)

    backtracks = float(sum(float(r.get("backtracks", 0)) for r in results))
    proactive = float(sum(float(r.get("proactive_backtracks", 0)) for r in results))
    backtracks_per_solve = backtracks / max(float(solved), 1.0)

    total_flagged_dead = float(
        sum(float(r.get("total_flagged_dead", 0)) for r in results)
    )
    true_dead_flagged = float(
        sum(float(r.get("true_dead_flagged", 0)) for r in results)
    )
    total_true_dead = float(sum(float(r.get("total_true_dead", 0)) for r in results))

    actual_precision = true_dead_flagged / max(total_flagged_dead, 1.0)
    actual_recall = true_dead_flagged / max(total_true_dead, 1.0)

    fp_opportunities = float(sum(float(r.get("fp_opportunities", 0)) for r in results))
    fn_opportunities = float(sum(float(r.get("fn_opportunities", 0)) for r in results))
    actual_fp_count = float(sum(float(r.get("actual_fp_count", 0)) for r in results))
    actual_fn_count = float(sum(float(r.get("actual_fn_count", 0)) for r in results))

    actual_fp_rate = actual_fp_count / max(fp_opportunities, 1.0)
    actual_fn_rate = actual_fn_count / max(fn_opportunities, 1.0)

    return {
        "n": int(n),
        "solve_rate": float(solve_rate),
        "backtracks_per_solve": float(backtracks_per_solve),
        "proactive_backtracks": float(proactive),
        "oracle_checks": float(sum(float(r.get("oracle_checks", 0)) for r in results)),
        "oracle_unsat": float(sum(float(r.get("oracle_unsat", 0)) for r in results)),
        "actual_fp_count": float(actual_fp_count),
        "actual_fn_count": float(actual_fn_count),
        "fp_opportunities": float(fp_opportunities),
        "fn_opportunities": float(fn_opportunities),
        "actual_fp_rate": float(actual_fp_rate),
        "actual_fn_rate": float(actual_fn_rate),
        "actual_precision": float(actual_precision),
        "actual_recall": float(actual_recall),
        "total_flagged_dead": float(total_flagged_dead),
        "total_true_dead": float(total_true_dead),
        "true_dead_flagged": float(true_dead_flagged),
    }


def _build_configs(
    mode: str, fp_rate: float, fn_rate: float
) -> Tuple[str, List[Tuple[float, float]]]:
    if str(mode) == "single":
        return "single", [(float(fp_rate), float(fn_rate))]
    if str(mode) == "sweep_fp":
        return "fp", [(float(r), 0.0) for r in SWEEP_RATES]
    if str(mode) == "sweep_fn":
        return "fn", [(0.0, float(r)) for r in SWEEP_RATES]
    if str(mode) == "sweep_both":
        return "both", [(float(r), float(r)) for r in SWEEP_RATES]
    raise ValueError(f"unknown mode: {mode}")


def _safe_tag(rate: float) -> str:
    return f"{float(rate):.3f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAT oracle corruption curve")
    parser.add_argument(
        "--policy_checkpoint",
        type=str,
        default="experiments/dvp-sat/n50_mlp_baseline/best_model.pt",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["model", "vsids"],
        default="model",
        help="Branching policy source: trained model or VSIDS-like heuristic",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Compatibility alias for policy checkpoint",
    )
    parser.add_argument("--model_type", choices=["dvp", "mlp"], default="mlp")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--num_vars", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--max_backtracks", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/dvp-sat/corruption_curve",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "sweep_fp", "sweep_fn", "sweep_both"],
        default="single",
    )
    parser.add_argument("--fp_rate", type=float, default=0.0)
    parser.add_argument("--fn_rate", type=float, default=0.0)
    args = parser.parse_args()

    effective_policy_checkpoint = (
        str(args.checkpoint).strip()
        if str(args.checkpoint).strip()
        else str(args.policy_checkpoint)
    )

    _set_seed(int(args.seed))

    req_device = torch.device(str(args.device))
    if req_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("requested cuda device is unavailable; falling back to cpu")
        device = torch.device("cpu")
    else:
        device = req_device

    model: Optional[nn.Module] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    model_type = "vsids"
    effective_checkpoint_for_output = str(effective_policy_checkpoint)

    if str(args.policy) == "model":
        ckpt = torch.load(
            Path(effective_policy_checkpoint), map_location="cpu", weights_only=False
        )
        cfg = ckpt.get("config", {})
        mean = np.asarray(ckpt.get("feature_mean"), dtype=np.float32)
        std = np.asarray(ckpt.get("feature_std"), dtype=np.float32)

        feature_dim = int(cfg.get("feature_dim", 9))
        num_vars = int(cfg.get("num_vars", args.num_vars))
        if int(num_vars) != int(args.num_vars):
            raise ValueError(
                f"checkpoint num_vars={num_vars} but args num_vars={args.num_vars}"
            )

        model_type = str(cfg.get("model", "mlp"))
        if model_type == "dvp":
            model = DenseViabilityNet(
                num_vars=int(num_vars),
                feature_dim=int(feature_dim),
                d_model=int(cfg.get("d_model", 64)),
                n_heads=int(cfg.get("n_heads", 4)),
                n_layers=int(cfg.get("n_layers", 3)),
                n_slots=int(cfg.get("n_slots", 0)),
                dropout=0.1,
            )
        else:
            model = SharedMLP(
                feature_dim=int(feature_dim),
                hidden_dim=int(cfg.get("d_model", 64)),
                n_layers=2,
            )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        logger.info(
            "loaded policy checkpoint=%s model=%s num_vars=%d device=%s",
            str(effective_policy_checkpoint),
            str(model_type),
            int(args.num_vars),
            str(device),
        )
    else:
        effective_checkpoint_for_output = ""
        logger.info(
            "using VSIDS policy heuristic num_vars=%d device=%s",
            int(args.num_vars),
            str(device),
        )

    generator = SatGenerator(seed=int(args.seed))
    instances = [
        generator.generate_planted(num_vars=int(args.num_vars), alpha=float(args.alpha))
        for _ in range(int(args.num_instances))
    ]
    logger.info(
        "generated shared instance set count=%d seed=%d",
        int(args.num_instances),
        int(args.seed),
    )

    sweep_type, configs = _build_configs(
        mode=str(args.mode), fp_rate=float(args.fp_rate), fn_rate=float(args.fn_rate)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_results: List[Dict[str, float]] = []
    for cfg_idx, (fp_rate, fn_rate) in enumerate(configs):
        logger.info(
            "eval config idx=%d/%d fp_rate=%.3f fn_rate=%.3f",
            int(cfg_idx + 1),
            int(len(configs)),
            float(fp_rate),
            float(fn_rate),
        )

        policy_rng = random.Random(int(args.seed))
        corruption_rng = random.Random(int(args.seed) + 1000)

        per_instance: List[Dict[str, object]] = []
        run_results: List[Dict[str, float]] = []
        for idx, inst in enumerate(instances):
            stats = _run_solver(
                clauses=[list(c) for c in inst.clauses],
                num_vars=int(args.num_vars),
                max_steps=int(args.max_steps),
                max_backtracks=int(args.max_backtracks),
                policy_rng=policy_rng,
                corruption_rng=corruption_rng,
                policy=str(args.policy),
                model=model,
                mean=mean,
                std=std,
                device=device,
                fp_rate=float(fp_rate),
                fn_rate=float(fn_rate),
            )
            run_results.append(stats)
            per_instance.append(
                {
                    "instance_id": int(idx),
                    "fp_rate": float(fp_rate),
                    "fn_rate": float(fn_rate),
                    "stats": {k: float(v) for k, v in stats.items()},
                }
            )
            if idx < 3:
                logger.info(
                    "instance=%d solved=%s backtracks=%.0f proactive=%.0f fp_count=%.0f fn_count=%.0f",
                    int(idx),
                    str(bool(stats["solved"])),
                    float(stats["backtracks"]),
                    float(stats["proactive_backtracks"]),
                    float(stats["actual_fp_count"]),
                    float(stats["actual_fn_count"]),
                )

        summary = _summarize(run_results)
        summary["fp_rate"] = float(fp_rate)
        summary["fn_rate"] = float(fn_rate)
        sweep_results.append(summary)

        logger.info(
            "config_done fp=%.3f fn=%.3f solve_rate=%.3f precision=%.3f recall=%.3f actual_fp=%.3f actual_fn=%.3f",
            float(fp_rate),
            float(fn_rate),
            float(summary["solve_rate"]),
            float(summary["actual_precision"]),
            float(summary["actual_recall"]),
            float(summary["actual_fp_rate"]),
            float(summary["actual_fn_rate"]),
        )

        per_config_output = {
            "policy_checkpoint": str(effective_checkpoint_for_output),
            "model_type": str(model_type),
            "mode": str(args.mode),
            "num_instances": int(args.num_instances),
            "num_vars": int(args.num_vars),
            "alpha": float(args.alpha),
            "max_backtracks": int(args.max_backtracks),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "corruption_seed": int(args.seed) + 1000,
            "fp_rate": float(fp_rate),
            "fn_rate": float(fn_rate),
            "summary": summary,
            "per_instance": per_instance,
        }

        per_cfg_path = (
            out_dir / f"results_fp{_safe_tag(fp_rate)}_fn{_safe_tag(fn_rate)}.json"
        )
        with per_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(per_config_output, f, indent=2)

    if str(args.mode) == "single":
        single_output = {
            "sweep_type": "single",
            "results": sweep_results,
        }
        out_path = out_dir / "results.json"
    else:
        single_output = {
            "sweep_type": str(sweep_type),
            "results": sweep_results,
        }
        out_path = out_dir / "sweep_summary.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(single_output, f, indent=2)

    logger.info("saved summary to %s", str(out_path))

    logger.info("=" * 112)
    logger.info(
        "%-8s %-8s %-10s %-14s %-14s %-12s %-12s %-12s %-12s",
        "fp_rate",
        "fn_rate",
        "solve_rate",
        "bt_per_solve",
        "proactive_bt",
        "precision",
        "recall",
        "act_fp_rate",
        "act_fn_rate",
    )
    for row in sweep_results:
        logger.info(
            "%-8.3f %-8.3f %-10.3f %-14.3f %-14.1f %-12.3f %-12.3f %-12.3f %-12.3f",
            float(row["fp_rate"]),
            float(row["fn_rate"]),
            float(row["solve_rate"]),
            float(row["backtracks_per_solve"]),
            float(row["proactive_backtracks"]),
            float(row["actual_precision"]),
            float(row["actual_recall"]),
            float(row["actual_fp_rate"]),
            float(row["actual_fn_rate"]),
        )
    logger.info("=" * 112)


if __name__ == "__main__":
    main()
