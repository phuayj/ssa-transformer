#!/usr/bin/env python3
"""Generate LaTeX tabular files for the SSA experiments section."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "paper" / "manuscript" / "parts" / "tables"
ROW_END = r"\\"


def load_json(relative_path: str) -> dict[str, Any]:
    path = REPO / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_table(filename: str, content: str) -> None:
    outpath = TABLE_DIR / filename
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(content + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", outpath)


def fmt2(value: float) -> str:
    return f"{value:.2f}"


def fmt1(value: float) -> str:
    return f"{value:.1f}"


def aggregate_metrics(relative_path: str) -> dict[str, float]:
    payload = load_json(relative_path)
    aggregate = payload["aggregate"]
    return {
        "solve_rate": float(aggregate["solve_rate"]),
        "repeat_rate": float(aggregate["repeat_error_rate"]),
        "mean_backtracks": float(aggregate["mean_backtracks"]),
    }


def generate_budget_sweep() -> None:
    pairs = [
        ("SSA", 2048, "experiments/gc-ssa-v2-eval/ssa.json"),
        ("SSA", 4096, "experiments/gc-ssa-v2-eval-extended/ssa.json"),
        ("SSA", 8192, "experiments/gc-ssa-v2-eval-8192/ssa.json"),
        ("Causal", 2048, "experiments/gc-ssa-v2-eval/causal.json"),
        ("Causal", 4096, "experiments/gc-ssa-v2-eval-extended/causal.json"),
        ("Causal", 8192, "experiments/gc-ssa-v2-eval-8192/causal.json"),
    ]

    rows: list[str] = []
    for mode, budget, path in pairs:
        m = aggregate_metrics(path)
        LOGGER.info(
            "budget_sweep mode=%s budget=%d solve=%.3f repeat=%.3f mean_bt=%.3f",
            mode,
            budget,
            m["solve_rate"],
            m["repeat_rate"],
            m["mean_backtracks"],
        )
        rows.append(
            f"{mode} & {budget} & {fmt2(m['solve_rate'])} & {fmt2(m['repeat_rate'])} & {fmt1(m['mean_backtracks'])} {ROW_END}"
        )

    lines = [
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"\textbf{Mode} & \textbf{Budget} & \textbf{Solve Rate} & \textbf{Repeat Rate} & \textbf{Mean BT} \\",
        r"\midrule",
        rows[0],
        rows[1],
        rows[2],
        r"\midrule",
        rows[3],
        rows[4],
        rows[5],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_table("tab_ssa_budget_sweep.tex", "\n".join(lines))


def generate_init_robustness() -> None:
    entries = [
        ("SSA", "Random", "experiments/gc-ssa-eval/ssa.json"),
        ("SSA", "Pretrained", "experiments/gc-ssa-v2-eval/ssa.json"),
        ("Causal", "Random", "experiments/gc-ssa-eval/causal.json"),
        ("Causal", "Pretrained", "experiments/gc-ssa-v2-eval/causal.json"),
    ]

    rows: list[str] = []
    for attention, init, path in entries:
        m = aggregate_metrics(path)
        LOGGER.info(
            "init_robustness attention=%s init=%s solve=%.3f repeat=%.3f",
            attention,
            init,
            m["solve_rate"],
            m["repeat_rate"],
        )
        rows.append(
            f"{attention} & {init} & {fmt2(m['solve_rate'])} & {fmt2(m['repeat_rate'])} {ROW_END}"
        )

    lines = [
        r"\begin{tabular}{ll cc}",
        r"\toprule",
        r"\textbf{Attention} & \textbf{Init} & \textbf{Solve Rate} & \textbf{Repeat Rate} \\",
        r"\midrule",
        rows[0],
        rows[1],
        r"\midrule",
        rows[2],
        rows[3],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_table("tab_ssa_init.tex", "\n".join(lines))


def generate_ablation() -> None:
    summary = load_json("experiments/gc-ablation/ablation_summary.json")
    rows_by_mode = {
        row["mode"]: row for row in summary["rows"] if int(row["budget"]) == 2048
    }
    order = [
        ("selective_ssa", "Selective SSA", "Exact", "Blocked"),
        ("blanket_ssa", "Blanket SSA", "Blocked", "Blocked"),
        ("full_causal", "Full causal", "Exact", "Exact"),
        ("reverse_selective", "Reverse selective", "Blocked", "Exact"),
        ("random_matched", "Random matched", "Random", "Random"),
    ]

    lines = [
        r"\begin{tabular}{lcc ccc}",
        r"\toprule",
        r"\textbf{Mode} & \textbf{Problem} & \textbf{Trajectory} & \textbf{Solve Rate} & \textbf{Repeat Rate} & \textbf{Mean BT} \\",
        r"\midrule",
    ]

    for mode_key, label, problem_lbl, traj_lbl in order:
        if mode_key not in rows_by_mode:
            raise KeyError(f"Missing budget-2048 ablation row for mode={mode_key}")
        row = rows_by_mode[mode_key]
        LOGGER.info(
            "ablation mode=%s solve=%.3f repeat=%.3f mean_bt=%.3f",
            mode_key,
            float(row["solve_rate"]),
            float(row["repeat_rate"]),
            float(row["mean_backtracks"]),
        )
        lines.append(
            f"{label} & {problem_lbl} & {traj_lbl} & {fmt2(float(row['solve_rate']))} & {fmt2(float(row['repeat_rate']))} & {fmt1(float(row['mean_backtracks']))} {ROW_END}"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_ssa_ablation.tex", "\n".join(lines))


def generate_bw() -> None:
    entries = [
        ("SSA", 2048, "experiments/bw-ssa-eval-2048/ssa.json"),
        ("SSA", 4096, "experiments/bw-ssa-eval-4096/ssa.json"),
        ("Causal", 2048, "experiments/bw-ssa-eval-2048/causal.json"),
        ("Causal", 4096, "experiments/bw-ssa-eval-4096/causal.json"),
    ]

    rows: list[str] = []
    for mode, budget, path in entries:
        m = aggregate_metrics(path)
        LOGGER.info(
            "bw mode=%s budget=%d solve=%.3f repeat=%.3f mean_bt=%.3f",
            mode,
            budget,
            m["solve_rate"],
            m["repeat_rate"],
            m["mean_backtracks"],
        )
        rows.append(
            f"{mode} & {budget} & {fmt2(m['solve_rate'])} & {fmt2(m['repeat_rate'])} & {fmt1(m['mean_backtracks'])} {ROW_END}"
        )

    lines = [
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"\textbf{Mode} & \textbf{Budget} & \textbf{Solve Rate} & \textbf{Repeat Rate} & \textbf{Mean BT} \\",
        r"\midrule",
        rows[0],
        rows[1],
        r"\midrule",
        rows[2],
        rows[3],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_table("tab_ssa_bw.tex", "\n".join(lines))


def _config_label(row: dict[str, Any]) -> str:
    n = int(row["num_nodes"])
    k = int(row["num_colors"])
    p = float(row["edge_prob"])
    return f"$n{{=}}{n}, k{{=}}{k}, p{{=}}{p:.2f}$"


def generate_gc_sweep() -> None:
    payload = load_json("experiments/gc-ssa-sweep/ssa_sweep_summary.json")
    rows = payload["rows"]
    num_instances = int(payload.get("num_instances", 100))

    by_config_budget: dict[str, dict[int, dict[str, Any]]] = {}
    config_order: list[str] = []
    for row in rows:
        cfg = str(row["config"])
        if cfg not in by_config_budget:
            by_config_budget[cfg] = {}
            config_order.append(cfg)
        by_config_budget[cfg][int(row["budget"])] = row

    lines = [
        r"\begin{tabular}{lc cc cc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Budget 2048}} & \multicolumn{2}{c}{\textbf{Budget 8192}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"\textbf{Config} & \textbf{$N$} & \textbf{SSA} & \textbf{Causal} & \textbf{SSA} & \textbf{Causal} \\",
        r"\midrule",
    ]

    included = 0
    for cfg in config_order:
        cfg_rows = by_config_budget[cfg]
        if 2048 not in cfg_rows or 8192 not in cfg_rows:
            LOGGER.warning("gc_sweep skipping config=%s missing 2048/8192 rows", cfg)
            continue
        r2048 = cfg_rows[2048]
        r8192 = cfg_rows[8192]
        if not r2048.get("success", False) or not r8192.get("success", False):
            LOGGER.warning("gc_sweep skipping config=%s unsuccessful runs", cfg)
            continue

        label = _config_label(r2048)
        lines.append(
            f"{label} & {num_instances} & {fmt2(float(r2048['ssa_solve']))} & {fmt2(float(r2048['causal_solve']))} & {fmt2(float(r8192['ssa_solve']))} & {fmt2(float(r8192['causal_solve']))} {ROW_END}"
        )
        included += 1
        LOGGER.info(
            "gc_sweep config=%s solve2048(ssa=%.3f causal=%.3f) solve8192(ssa=%.3f causal=%.3f)",
            cfg,
            float(r2048["ssa_solve"]),
            float(r2048["causal_solve"]),
            float(r8192["ssa_solve"]),
            float(r8192["causal_solve"]),
        )

    if included == 0:
        raise RuntimeError("No successful GC sweep rows for budgets 2048 and 8192")

    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_ssa_gc_sweep.tex", "\n".join(lines))


def generate_proofwriter() -> None:
    payload = load_json(
        "experiments/proofwriter-ssa-baseline/proofwriter_ssa_eval_seed42.json"
    )
    summary = payload["summary"]
    modes = [
        ("direct", "Direct"),
        ("cot", "CoT"),
        ("cot_truncated", "CoT truncated"),
        ("recurrent_state", "Recurrent state"),
        ("oracle_state", "Oracle state"),
    ]

    lines = [
        r"\begin{tabular}{l cccccc c c}",
        r"\toprule",
        r"\textbf{Mode} & \textbf{D0} & \textbf{D1} & \textbf{D2} & \textbf{D3} & \textbf{D4} & \textbf{D5} & \textbf{Overall} & \textbf{Degrad.} \\",
        r"\midrule",
    ]

    for mode_key, label in modes:
        mode_stats = summary[mode_key]
        by_depth = mode_stats["by_depth"]
        dvals = [float(by_depth[str(depth)]["accuracy"]) / 100.0 for depth in range(6)]
        overall = float(mode_stats["overall"]["accuracy"]) / 100.0
        degrad = dvals[0] - dvals[5]
        lines.append(
            f"{label} & {fmt2(dvals[0])} & {fmt2(dvals[1])} & {fmt2(dvals[2])} & {fmt2(dvals[3])} & {fmt2(dvals[4])} & {fmt2(dvals[5])} & {fmt2(overall)} & {fmt2(degrad)} {ROW_END}"
        )
        LOGGER.info(
            "proofwriter mode=%s d0=%.3f d5=%.3f overall=%.3f degrad=%.3f",
            mode_key,
            dvals[0],
            dvals[5],
            overall,
            degrad,
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_ssa_proofwriter.tex", "\n".join(lines))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    generate_budget_sweep()
    generate_init_robustness()
    generate_ablation()
    generate_bw()
    generate_gc_sweep()
    generate_proofwriter()
    LOGGER.info("all SSA tables generated")


if __name__ == "__main__":
    main()
