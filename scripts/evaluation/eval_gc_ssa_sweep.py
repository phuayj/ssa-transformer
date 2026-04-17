#!/usr/bin/env python3
"""Run SSA vs causal eval over GC config/budget sweep."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_gc_ssa.py"

CONFIGS: List[Tuple[str, int, int, float]] = [
    # (name, num_nodes, num_colors, edge_prob)
    ("n30_k4_p035", 30, 4, 0.35),  # in-distribution (baseline)
    ("n30_k3_p035", 30, 3, 0.35),  # fewer colors (harder)
    ("n30_k5_p035", 30, 5, 0.35),  # more colors (easier)
    ("n20_k4_p035", 20, 4, 0.35),  # smaller graphs
    ("n40_k4_p035", 40, 4, 0.35),  # larger graphs (OOD)
    ("n30_k4_p025", 30, 4, 0.25),  # sparser
    ("n30_k4_p045", 30, 4, 0.45),  # denser (harder)
]

BUDGETS = [2048, 4096, 8192]
MAX_STEPS = 2000


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_repeat_rate(aggregate: Dict[str, Any]) -> float:
    if "repeat_rate" in aggregate:
        return float(aggregate["repeat_rate"])
    if "repeat_error_rate" in aggregate:
        return float(aggregate["repeat_error_rate"])
    raise KeyError("missing repeat_rate/repeat_error_rate in aggregate")


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _run_single(
    args: argparse.Namespace, config: Tuple[str, int, int, float], budget: int
) -> Dict[str, Any]:
    config_name, num_nodes, num_colors, edge_prob = config
    out_dir = Path(args.output_dir) / f"{config_name}_budget{int(budget)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--ssa_checkpoint",
        str(args.ssa_checkpoint),
        "--causal_checkpoint",
        str(args.causal_checkpoint),
        "--num_instances",
        str(int(args.num_instances)),
        "--num_nodes",
        str(int(num_nodes)),
        "--num_colors",
        str(int(num_colors)),
        "--edge_prob",
        str(float(edge_prob)),
        "--max_steps",
        str(int(MAX_STEPS)),
        "--max_seq_len",
        str(int(budget)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--output_dir",
        str(out_dir),
    ]

    logger.info(
        "running config=%s n=%d k=%d p=%.3f budget=%d max_steps=%d",
        config_name,
        int(num_nodes),
        int(num_colors),
        float(edge_prob),
        int(budget),
        int(MAX_STEPS),
    )

    row: Dict[str, Any] = {
        "config": str(config_name),
        "num_nodes": int(num_nodes),
        "num_colors": int(num_colors),
        "edge_prob": float(edge_prob),
        "budget": int(budget),
        "max_steps": int(MAX_STEPS),
        "run_dir": str(out_dir),
        "success": False,
    }

    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "failed config=%s budget=%d returncode=%d",
            config_name,
            int(budget),
            int(exc.returncode),
        )
        row["error"] = f"subprocess_failed_returncode_{int(exc.returncode)}"
        return row

    ssa_path = out_dir / "ssa.json"
    causal_path = out_dir / "causal.json"
    if not ssa_path.exists() or not causal_path.exists():
        logger.error(
            "missing outputs config=%s budget=%d ssa=%s causal=%s",
            config_name,
            int(budget),
            str(ssa_path.exists()),
            str(causal_path.exists()),
        )
        row["error"] = "missing_output_json"
        return row

    ssa_payload = _read_json(ssa_path)
    causal_payload = _read_json(causal_path)
    ssa_agg = dict(ssa_payload.get("aggregate", {}))
    causal_agg = dict(causal_payload.get("aggregate", {}))

    row.update(
        {
            "success": True,
            "ssa_solve": float(ssa_agg["solve_rate"]),
            "causal_solve": float(causal_agg["solve_rate"]),
            "ssa_repeat": float(_extract_repeat_rate(ssa_agg)),
            "causal_repeat": float(_extract_repeat_rate(causal_agg)),
            "ssa_mean_backtracks": float(ssa_agg["mean_backtracks"]),
            "causal_mean_backtracks": float(causal_agg["mean_backtracks"]),
        }
    )
    return row


def _print_table(rows: List[Dict[str, Any]]) -> None:
    print(
        "Config          Budget   SSA_Solve  Causal_Solve  "
        "SSA_Repeat  Causal_Repeat  SSA_MeanBT  Causal_MeanBT"
    )
    for row in rows:
        if not bool(row.get("success", False)):
            print(f"{row['config']:<15} {int(row['budget']):<8} FAIL")
            continue
        print(
            f"{row['config']:<15} {int(row['budget']):<8} "
            f"{_format_value(row.get('ssa_solve')):<10} "
            f"{_format_value(row.get('causal_solve')):<13} "
            f"{_format_value(row.get('ssa_repeat')):<11} "
            f"{_format_value(row.get('causal_repeat')):<14} "
            f"{_format_value(row.get('ssa_mean_backtracks')):<10} "
            f"{_format_value(row.get('causal_mean_backtracks')):<10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SSA vs causal eval over GC config/budget sweep"
    )
    parser.add_argument("--ssa_checkpoint", type=str, required=True)
    parser.add_argument("--causal_checkpoint", type=str, required=True)
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(CONFIGS) * len(BUDGETS)
    rows: List[Dict[str, Any]] = []
    run_idx = 0

    for config in CONFIGS:
        for budget in BUDGETS:
            run_idx += 1
            logger.info("progress %d/%d", int(run_idx), int(total_runs))
            rows.append(_run_single(args, config, int(budget)))

    summary = {
        "configs": [
            {
                "name": name,
                "num_nodes": int(n),
                "num_colors": int(k),
                "edge_prob": float(p),
            }
            for name, n, k, p in CONFIGS
        ],
        "budgets": [int(x) for x in BUDGETS],
        "max_steps": int(MAX_STEPS),
        "num_instances": int(args.num_instances),
        "device": str(args.device),
        "seed": int(args.seed),
        "rows": rows,
    }

    summary_path = output_dir / "ssa_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote summary json: %s", str(summary_path))

    print()
    _print_table(rows)


if __name__ == "__main__":
    main()
