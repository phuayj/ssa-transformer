#!/usr/bin/env python3
"""Closed-loop evaluation of SAT CDCL with learned/decoder backjumping."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sat.cdcl_solver import CDCLSolver, compute_1uip
from sat.cdcl_tokenizer import SATCDCLTokenizer
from sat.generator import SatGenerator
from universal.cdcl_decoder import CDCLDecoder


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


def _level_from_token(
    token_id: Optional[int], tokenizer: SATCDCLTokenizer
) -> Optional[int]:
    if token_id is None:
        return None
    token_id = int(token_id)
    if token_id < int(tokenizer.LEVEL_OFFSET):
        return None
    level = int(token_id) - int(tokenizer.LEVEL_OFFSET)
    if level < 0 or level >= int(tokenizer.MAX_LEVELS):
        return None
    return int(level)


def _extract_target_level(
    tokens: List[int], tokenizer: SATCDCLTokenizer
) -> Optional[int]:
    try:
        idx = int(tokens.index(int(tokenizer.TARGET_SEC)))
    except ValueError:
        return None
    if int(idx) + 1 >= int(len(tokens)):
        return None
    return _level_from_token(int(tokens[int(idx) + 1]), tokenizer)


def load_decoder_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[CDCLDecoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config")
    if not config:
        raise RuntimeError("Checkpoint missing config")
    model = CDCLDecoder(
        vocab_size=int(config.get("vocab_size", SATCDCLTokenizer.VOCAB_SIZE)),
        d_model=int(config["d_model"]),
        n_layers=int(config["n_layers"]),
        n_heads=int(config["n_heads"]),
        max_seq_len=int(config.get("max_seq_len", 1024)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def _serialize_state_tokens(
    state: dict, tokenizer: SATCDCLTokenizer, mode: str
) -> Optional[List[int]]:
    mode = str(mode)
    if mode not in {"scratchpad", "flat"}:
        raise ValueError(f"Unknown decoder mode: {mode}")

    clauses = state["clauses"]
    if int(len(clauses)) > int(tokenizer.MAX_CLAUSES):
        return None

    tokens: List[int] = [int(tokenizer.CLAUSE_SEC)]

    for cid, clause in enumerate(clauses):
        try:
            tokens.append(int(tokenizer.clause_token(int(cid))))
        except ValueError:
            return None
        tokens.append(int(tokenizer.COLON))
        for lit in clause:
            try:
                tokens.append(int(tokenizer.lit_token(int(lit))))
            except ValueError:
                return None
        tokens.append(int(tokenizer.SEP))

    tokens.append(int(tokenizer.TRAIL_SEC))
    trail = state["trail"]
    trail_levels = state["trail_levels"]
    var_reason = state["var_reason"]

    if len(trail) != len(trail_levels):
        return None

    for idx, lit in enumerate(trail):
        try:
            tokens.append(int(tokenizer.lit_token(int(lit))))
        except ValueError:
            return None
        tokens.append(int(tokenizer.AT_KW))
        level = int(trail_levels[int(idx)])
        if int(level) < 0 or int(level) >= int(tokenizer.MAX_LEVELS):
            return None
        tokens.append(int(tokenizer.level_token(int(level))))

        var_id = int(abs(int(lit)) - 1)
        reason = var_reason[int(var_id)]
        if reason is None:
            tokens.append(int(tokenizer.DEC_KW))
        else:
            try:
                tokens.append(int(tokenizer.REAS_KW))
                tokens.append(int(tokenizer.clause_token(int(reason))))
            except ValueError:
                return None
        tokens.append(int(tokenizer.SEP))

    try:
        conflict_id = int(state["conflict_clause_id"])
        tokens.append(int(tokenizer.CONFLICT_SEC))
        tokens.append(int(tokenizer.clause_token(int(conflict_id))))
        tokens.append(int(tokenizer.SEP))
    except ValueError:
        return None

    if mode == "flat":
        tokens.append(int(tokenizer.TARGET_SEC))
    else:
        tokens.append(int(tokenizer.THINK_SEC))

    return tokens


@torch.no_grad()
def predict_backjump_level(
    state: dict,
    tokenizer: SATCDCLTokenizer,
    model: CDCLDecoder,
    mode: str,
    device: torch.device,
    max_seq_len: int,
) -> Optional[int]:
    tokens = _serialize_state_tokens(state, tokenizer, mode)
    if tokens is None:
        return None

    prefix = [int(tokenizer.BOS), *tokens]
    if mode == "scratchpad" and int(prefix[-1]) != int(tokenizer.THINK_SEC):
        prefix.append(int(tokenizer.THINK_SEC))
    if mode == "flat" and int(prefix[-1]) != int(tokenizer.TARGET_SEC):
        prefix.append(int(tokenizer.TARGET_SEC))

    if len(prefix) > int(max_seq_len):
        logger.warning(
            "decoder prefix length=%d exceeds max_seq_len=%d",
            int(len(prefix)),
            int(max_seq_len),
        )
        return None

    max_new = 1 if mode == "flat" else int(max_seq_len) - int(len(prefix))
    if int(max_new) <= 0:
        return None

    prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(
        prefix_tensor,
        max_new_tokens=int(max_new),
        temperature=0.0,
        stop_token=int(tokenizer.EOS),
    )
    gen_tokens = generated[0].tolist()
    return _extract_target_level(gen_tokens, tokenizer)


def make_decoder_analyzer(
    *,
    model: CDCLDecoder,
    tokenizer: SATCDCLTokenizer,
    device: torch.device,
    max_seq_len: int,
    mode: str,
    metrics: dict,
    log_examples: bool,
    log_prefix: str,
) -> Callable[[dict], Tuple[List[int], int, int]]:
    mode = str(mode)
    if mode not in {"scratchpad", "flat"}:
        raise ValueError(f"Unknown decoder mode: {mode}")

    log_limit = 3

    def analyzer(state: dict) -> Tuple[List[int], int, int]:
        learned_clause, oracle_bj, asserting_lit = compute_1uip(
            clauses=state["clauses"],
            trail=state["trail"],
            trail_levels=state["trail_levels"],
            var_reason=state["var_reason"],
            conflict_clause_id=state["conflict_clause_id"],
            num_vars=state["num_vars"],
            level=state["level"],
        )

        predicted = predict_backjump_level(
            state=state,
            tokenizer=tokenizer,
            model=model,
            mode=mode,
            device=device,
            max_seq_len=int(max_seq_len),
        )

        if predicted is None:
            metrics["decoder_fallbacks"] += 1
            if log_examples and int(metrics["fallback_logged"]) < int(log_limit):
                logger.info(
                    "%s decoder_fallback oracle_bj=%d", log_prefix, int(oracle_bj)
                )
                metrics["fallback_logged"] += 1
            return list(learned_clause), int(oracle_bj), int(asserting_lit)

        level = int(state["level"])
        clipped = int(max(0, min(int(predicted), int(level) - 1)))
        metrics["prediction_total"] += 1
        dist = int(abs(int(clipped) - int(oracle_bj)))
        metrics["prediction_exact"] += int(dist == 0)
        metrics["prediction_near"] += int(dist <= 1)
        metrics["prediction_distance_sum"] += float(dist)

        if log_examples and int(metrics["prediction_logged"]) < int(log_limit):
            logger.info(
                "%s decoder_backjump dl=%d oracle=%d pred=%d",
                log_prefix,
                int(level),
                int(oracle_bj),
                int(clipped),
            )
            metrics["prediction_logged"] += 1

        return list(learned_clause), int(clipped), int(asserting_lit)

    return analyzer


def evaluate_instance(
    *,
    clauses: List[Tuple[int, ...]],
    num_vars: int,
    mode: str,
    max_conflicts: int,
    tokenizer: SATCDCLTokenizer,
    decoder_model: Optional[CDCLDecoder] = None,
    decoder_config: Optional[dict] = None,
    device: torch.device = torch.device("cpu"),
    log_examples: bool = False,
    log_prefix: str = "",
) -> dict:
    decoder_metrics = {
        "decoder_fallbacks": 0,
        "prediction_total": 0,
        "prediction_exact": 0,
        "prediction_near": 0,
        "prediction_distance_sum": 0.0,
        "prediction_logged": 0,
        "fallback_logged": 0,
    }

    if mode == "chronological":
        solver = CDCLSolver(
            clauses=clauses,
            num_vars=int(num_vars),
            max_conflicts=int(max_conflicts),
            chronological=True,
        )
    elif mode == "oracle":
        solver = CDCLSolver(
            clauses=clauses,
            num_vars=int(num_vars),
            max_conflicts=int(max_conflicts),
        )
    elif mode in {"decoder_scratchpad", "decoder_flat"}:
        if decoder_model is None or decoder_config is None:
            raise RuntimeError("decoder mode requires a decoder model")
        decoder_mode = "scratchpad" if mode == "decoder_scratchpad" else "flat"
        analyzer = make_decoder_analyzer(
            model=decoder_model,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=int(decoder_config.get("max_seq_len", 1024)),
            mode=str(decoder_mode),
            metrics=decoder_metrics,
            log_examples=log_examples,
            log_prefix=log_prefix,
        )
        solver = CDCLSolver(
            clauses=clauses,
            num_vars=int(num_vars),
            max_conflicts=int(max_conflicts),
            conflict_analyzer=analyzer,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    start_time = time.perf_counter()
    result = solver.solve()
    wall_time = float(time.perf_counter() - start_time)

    payload = {
        **result,
        "wall_time": float(wall_time),
        "solved": str(result.get("status")) == "sat",
        "decoder_fallbacks": int(decoder_metrics["decoder_fallbacks"]),
        "prediction_total": int(decoder_metrics["prediction_total"]),
        "prediction_exact": int(decoder_metrics["prediction_exact"]),
        "prediction_near": int(decoder_metrics["prediction_near"]),
        "prediction_distance_sum": float(decoder_metrics["prediction_distance_sum"]),
    }
    return payload


def summarize_results(per_instance: List[dict], mode: str) -> dict:
    decisions = [int(item.get("decisions", 0)) for item in per_instance]
    conflicts = [int(item.get("conflicts", 0)) for item in per_instance]
    backtracks = [int(item.get("backtracks", 0)) for item in per_instance]
    learned = [int(item.get("learned_clauses", 0)) for item in per_instance]
    props = [int(item.get("propagations", 0)) for item in per_instance]
    wall_times = [float(item.get("wall_time", 0.0)) for item in per_instance]

    solve_rate = float(
        sum(1 for item in per_instance if item.get("solved"))
        / max(len(per_instance), 1)
    )

    summary = {
        "mean_decisions": float(np.mean(decisions)) if decisions else 0.0,
        "mean_conflicts": float(np.mean(conflicts)) if conflicts else 0.0,
        "mean_backtracks": float(np.mean(backtracks)) if backtracks else 0.0,
        "mean_learned": float(np.mean(learned)) if learned else 0.0,
        "mean_propagations": float(np.mean(props)) if props else 0.0,
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else 0.0,
        "solved": solve_rate,
    }

    if mode in {"decoder_scratchpad", "decoder_flat"}:
        pred_total = sum(int(item.get("prediction_total", 0)) for item in per_instance)
        pred_exact = sum(int(item.get("prediction_exact", 0)) for item in per_instance)
        pred_near = sum(int(item.get("prediction_near", 0)) for item in per_instance)
        pred_dist_sum = sum(
            float(item.get("prediction_distance_sum", 0.0)) for item in per_instance
        )
        total_conflicts = sum(int(item.get("conflicts", 0)) for item in per_instance)

        summary.update(
            {
                "oracle_exact_match": float(pred_exact / max(pred_total, 1)),
                "oracle_near_match": float(pred_near / max(pred_total, 1)),
                "oracle_mean_distance": float(pred_dist_sum / max(pred_total, 1)),
                "prediction_coverage": float(pred_total / max(total_conflicts, 1)),
            }
        )

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop evaluation of SAT CDCL backjumping strategies"
    )
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--num_vars", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratchpad_model", type=str, default=None)
    parser.add_argument("--flat_model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_conflicts", type=int, default=50000)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/sat-cdcl-eval/cdcl_closedloop.json",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    device = torch.device(str(args.device))
    tokenizer = SATCDCLTokenizer()

    generator = SatGenerator(seed=int(args.seed))
    instances = [
        generator.generate_planted(num_vars=int(args.num_vars), alpha=float(args.alpha))
        for _ in range(int(args.num_instances))
    ]

    modes = ["chronological", "oracle"]

    decoder_models: Dict[str, CDCLDecoder] = {}
    decoder_configs: Dict[str, dict] = {}

    if args.scratchpad_model:
        scratchpad_path = Path(args.scratchpad_model)
        if not scratchpad_path.exists():
            raise FileNotFoundError(f"Missing scratchpad model: {scratchpad_path}")
        model, config = load_decoder_model(scratchpad_path, device)
        decoder_models["decoder_scratchpad"] = model
        decoder_configs["decoder_scratchpad"] = config
        modes.append("decoder_scratchpad")
        logger.info("loaded scratchpad decoder from %s", str(scratchpad_path))

    if args.flat_model:
        flat_path = Path(args.flat_model)
        if not flat_path.exists():
            raise FileNotFoundError(f"Missing flat model: {flat_path}")
        model, config = load_decoder_model(flat_path, device)
        decoder_models["decoder_flat"] = model
        decoder_configs["decoder_flat"] = config
        modes.append("decoder_flat")
        logger.info("loaded flat decoder from %s", str(flat_path))

    logger.info(
        "eval config modes=%s num_instances=%d num_vars=%d alpha=%.3f max_conflicts=%d",
        ",".join(modes),
        int(args.num_instances),
        int(args.num_vars),
        float(args.alpha),
        int(args.max_conflicts),
    )

    results: Dict[str, dict] = {}

    for mode in modes:
        per_instance: List[dict] = []
        decoder_model = decoder_models.get(mode)
        decoder_config = decoder_configs.get(mode)

        for instance_id, instance in enumerate(instances):
            log_examples = int(instance_id) == 0
            log_prefix = f"[{mode} #{instance_id}]"
            result = evaluate_instance(
                clauses=instance.clauses,
                num_vars=int(instance.num_vars),
                mode=str(mode),
                max_conflicts=int(args.max_conflicts),
                tokenizer=tokenizer,
                decoder_model=decoder_model,
                decoder_config=decoder_config,
                device=device,
                log_examples=log_examples,
                log_prefix=log_prefix,
            )
            result["instance_id"] = int(instance_id)
            result["is_satisfiable"] = bool(instance.is_satisfiable)
            per_instance.append(result)

            logger.info(
                "mode=%s instance=%d status=%s decisions=%d conflicts=%d backtracks=%d learned=%d",
                str(mode),
                int(instance_id),
                str(result.get("status")),
                int(result.get("decisions", 0)),
                int(result.get("conflicts", 0)),
                int(result.get("backtracks", 0)),
                int(result.get("learned_clauses", 0)),
            )

        results[mode] = summarize_results(per_instance, mode)
        logger.info(
            "mode=%s mean_decisions=%.2f mean_conflicts=%.2f mean_backtracks=%.2f solved=%.2f",
            str(mode),
            float(results[mode].get("mean_decisions", 0.0)),
            float(results[mode].get("mean_conflicts", 0.0)),
            float(results[mode].get("mean_backtracks", 0.0)),
            float(results[mode].get("solved", 0.0)),
        )

        if mode in {"decoder_scratchpad", "decoder_flat"}:
            logger.info(
                "mode=%s oracle_exact=%.3f oracle_near=%.3f oracle_mean_dist=%.3f coverage=%.3f",
                str(mode),
                float(results[mode].get("oracle_exact_match", 0.0)),
                float(results[mode].get("oracle_near_match", 0.0)),
                float(results[mode].get("oracle_mean_distance", 0.0)),
                float(results[mode].get("prediction_coverage", 0.0)),
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        **results,
        "metadata": {
            "num_instances": int(args.num_instances),
            "num_vars": int(args.num_vars),
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "max_conflicts": int(args.max_conflicts),
            "device": str(args.device),
            "modes": list(modes),
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("saved results to %s", str(output_path))


if __name__ == "__main__":
    main()
