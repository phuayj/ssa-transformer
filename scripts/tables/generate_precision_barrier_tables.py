#!/usr/bin/env python3
"""Generate LaTeX tables for the Precision Barrier section.

Uses multi-seed aggregate data (v2) where available, falling back to
single-seed experiment JSONs for configurations without multi-seed runs.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "paper" / "manuscript" / "parts" / "tables"
MULTISEED_AGG = REPO_ROOT / "experiments/dvp-sat/multiseed_section6_v2/aggregate.json"
UNIFIED_TAU_ROOT = REPO_ROOT / "experiments/dvp-sat/unified_tau_005"
ROW_END = r"\\"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> str:
    return f"{value * 100.0:.1f}\\%"


def pct_ci(mean: float, std: float) -> str:
    """Format as '$mean \\pm std\\%$' for multi-seed entries."""
    return f"${mean * 100.0:.1f} \\pm {std * 100.0:.1f}\\%$"


def fixed3(value: float) -> str:
    return f"{value:.3f}"


def with_commas(value: float, decimals: int = 0) -> str:
    text = f"{value:,.{decimals}f}" if decimals > 0 else f"{int(round(value)):,}"
    return text.replace(",", "{,}")


def maybe_bold(text: str, cond: bool) -> str:
    return f"\\textbf{{{text}}}" if cond else text


def row_cells(cells: list[str], bold: bool = False) -> str:
    rendered = [maybe_bold(c, bold) for c in cells] if bold else cells
    return " & ".join(rendered)


def write_table(filename: str, content: str) -> Path:
    out = TABLES_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", out)
    return out


def _extract_mlp_metrics(payload: dict[str, Any]) -> dict[str, float]:
    row = payload["results"]["mlp"]
    return {
        "solve_rate": float(row["solve_rate"]),
        "proactive_backtracks": float(row["proactive_backtracks"]),
        "backtracks_per_solve": float(row["backtracks_per_solve"]),
    }


def load_multiseed() -> dict[str, Any]:
    """Load the multi-seed aggregate (v2) if available."""
    if MULTISEED_AGG.exists():
        LOGGER.info("loaded multi-seed aggregate from %s", MULTISEED_AGG)
        return load_json(MULTISEED_AGG)
    LOGGER.warning("multi-seed aggregate not found; using single-seed only")
    return {}


def load_unified_tau() -> dict[int, dict[str, float]]:
    """Load unified-tau (0.05) iteration data for rounds 0..3 across 5 seeds."""
    rounds: dict[int, dict[str, float]] = {}
    seeds = [42, 123, 456, 789, 1337]
    for rnd in [0, 1, 2, 3]:
        solve_vals: list[float] = []
        pro_bts_vals: list[float] = []
        for seed in seeds:
            path = UNIFIED_TAU_ROOT / f"round_{rnd}" / f"seed_{seed}" / "results.json"
            row = _extract_mlp_metrics(load_json(path))
            solve_vals.append(row["solve_rate"])
            pro_bts_vals.append(row["proactive_backtracks"])

        rounds[rnd] = {
            "mean": statistics.mean(solve_vals),
            "std": statistics.pstdev(solve_vals),
            "pro_bts_mean": statistics.mean(pro_bts_vals),
            "pro_bts_std": statistics.pstdev(pro_bts_vals),
        }
        LOGGER.info(
            "unified_tau round=%d solve=%.1f±%.1f%% pro_bts=%.0f±%.0f",
            rnd,
            rounds[rnd]["mean"] * 100.0,
            rounds[rnd]["std"] * 100.0,
            rounds[rnd]["pro_bts_mean"],
            rounds[rnd]["pro_bts_std"],
        )
    return rounds


# ---------------------------------------------------------------------------
# Table: Corruption Curve
# ---------------------------------------------------------------------------


def build_corruption_curve_table(ms: dict[str, Any]) -> str:
    fp = load_json(
        REPO_ROOT / "experiments/dvp-sat/corruption_curve/sweep_summary.json"
    )
    fn = load_json(
        REPO_ROOT / "experiments/dvp-sat/corruption_curve_fn/sweep_summary.json"
    )
    fp_rows = {float(r["fp_rate"]): r for r in fp["results"]}
    fn_rows = {float(r["fn_rate"]): r for r in fn["results"]}
    rates = sorted(set(fp_rows) & set(fn_rows))

    # Multi-seed keys: corruption_fp_0.1, corruption_fp_0.2, corruption_fp_0.3
    ms_fp = {}
    for key, val in ms.items():
        if key.startswith("corruption_fp_"):
            rate_str = key.replace("corruption_fp_", "")
            ms_fp[float(rate_str)] = val

    fp_best_val = -1.0
    for rate in rates:
        ms_entry = ms_fp.get(rate)
        val = ms_entry["mean"] if ms_entry else float(fp_rows[rate]["solve_rate"])
        if val > fp_best_val:
            fp_best_val = val
    fn_best = max(float(r["solve_rate"]) for r in fn_rows.values())
    fp_marked = False
    fn_marked = False

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Synthetic corruption of oracle look-ahead decisions ($n{=}50$, 5~seeds $\\times$ 200 instances where available, single-seed otherwise). Left: false-positive injection collapses solve rate. Right: false-negative injection has no effect.}",
        "\\label{tab:corruption_curve}",
        "\\footnotesize",
        "\\begin{tabular}{cccccc}",
        "  \\toprule",
        f"  \\textbf{{FP Rate}} & \\textbf{{Solve (FP)}} & \\textbf{{Precision}} & \\textbf{{FN Rate}} & \\textbf{{Solve (FN)}} & \\textbf{{Recall}} {ROW_END}",
        "  \\midrule",
    ]

    for rate in rates:
        fp_row = fp_rows[rate]
        fn_row = fn_rows[rate]
        ms_entry = ms_fp.get(rate)

        if ms_entry and rate > 0:
            fp_solve = pct_ci(ms_entry["mean"], ms_entry["std"])
        else:
            fp_solve = pct(float(fp_row["solve_rate"]))
        fn_solve = pct(float(fn_row["solve_rate"]))

        solve_val = ms_entry["mean"] if ms_entry else float(fp_row["solve_rate"])
        fp_is_best = (not fp_marked) and (solve_val == fp_best_val)
        fn_is_best = (not fn_marked) and (float(fn_row["solve_rate"]) == fn_best)
        fp_marked = fp_marked or fp_is_best
        fn_marked = fn_marked or fn_is_best

        cells = [
            f"{pct(rate)}",
            maybe_bold(fp_solve, fp_is_best),
            fixed3(float(fp_row["actual_precision"])),
            f"{pct(rate)}",
            maybe_bold(fn_solve, fn_is_best),
            fixed3(float(fn_row["actual_recall"])),
        ]
        lines.append(f"  {row_cells(cells, bold=(fp_is_best or fn_is_best))} {ROW_END}")

    lines += ["  \\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table: Threshold Curve (single-seed; sweep without multi-seed data)
# ---------------------------------------------------------------------------


def build_threshold_curve_table() -> str:
    summary = load_json(
        REPO_ROOT / "experiments/dvp-sat/sweep_p2_threshold/summary_p2_threshold.json"
    )
    t01 = load_json(
        REPO_ROOT / "experiments/dvp-sat/eval_onpolicy_thresh/t0.1/results.json"
    )
    t02 = load_json(
        REPO_ROOT / "experiments/dvp-sat/eval_onpolicy_thresh/t0.2/results.json"
    )

    entries = {
        float(e["threshold"]): {
            "solve_rate": float(e["solve_rate"]),
            "proactive_backtracks": float(e["proactive_backtracks"]),
            "backtracks_per_solve": float(e["backtracks_per_solve"]),
        }
        for e in summary["entries"]
    }
    entries[0.10] = _extract_mlp_metrics(t01)
    entries[0.20] = _extract_mlp_metrics(t02)

    thresholds = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]
    best_solve = max(entries[t]["solve_rate"] for t in thresholds if t in entries)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Threshold operating curve for on-policy round-1 verifier with look-ahead ($n{=}50$, budget 500, single seed). Peak at $\\tau{=}0.05$.}",
        "\\label{tab:threshold_curve}",
        "\\footnotesize",
        "\\begin{tabular}{cccc}",
        "  \\toprule",
        f"  \\textbf{{Threshold}} & \\textbf{{Solve Rate}} & \\textbf{{Proactive BTs}} & \\textbf{{BTs/Solve}} {ROW_END}",
        "  \\midrule",
    ]

    for t in thresholds:
        if t not in entries:
            LOGGER.warning("missing threshold point t=%.2f", t)
            lines.append(f"  {t:.2f} & --- & --- & --- {ROW_END}")
            continue
        row = entries[t]
        cells = [
            f"{t:.2f}",
            pct(row["solve_rate"]),
            with_commas(float(row["proactive_backtracks"])),
            with_commas(float(row["backtracks_per_solve"]), decimals=1),
        ]
        lines.append(
            f"  {row_cells(cells, bold=row['solve_rate'] == best_solve)} {ROW_END}"
        )

    lines += ["  \\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table: Oracle Upper Bound (deterministic; single seed is sufficient)
# ---------------------------------------------------------------------------


def build_oracle_upperbound_table(ms: dict[str, Any]) -> str:
    payload = load_json(REPO_ROOT / "experiments/dvp-sat/oracle_lookahead/results.json")
    reactive = payload["results"]["reactive_only"]
    oracle = payload["results"]["oracle_lookahead"]

    # Use multi-seed data if available
    ms_reactive = ms.get("reactive_policy")
    ms_oracle = ms.get("oracle")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Oracle upper bound (5~seeds $\\times$ 200 instances). Perfect look-ahead verification achieves 100\\% solve rate.}",
        "\\label{tab:oracle_upperbound}",
        "\\footnotesize",
        "\\begin{tabular}{lccc}",
        "  \\toprule",
        f"  \\textbf{{Configuration}} & \\textbf{{Solve Rate}} & \\textbf{{BTs/Solve}} & \\textbf{{Proactive BTs}} {ROW_END}",
        "  \\midrule",
    ]

    for label, row, ms_entry in [
        ("Reactive only", reactive, ms_reactive),
        ("Oracle look-ahead", oracle, ms_oracle),
    ]:
        if ms_entry:
            solve_txt = pct_ci(ms_entry["mean"], ms_entry["std"])
        else:
            solve_txt = pct(float(row["solve_rate"]))
        cells = [
            label,
            solve_txt,
            with_commas(float(row["backtracks_per_solve"]), decimals=1),
            with_commas(float(row["proactive_backtracks"])),
        ]
        is_best = float(row["solve_rate"]) >= 0.99
        lines.append(f"  {row_cells(cells, bold=is_best)} {ROW_END}")

    lines += ["  \\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table: Iteration Curve (unified-tau multi-seed for all rounds)
# ---------------------------------------------------------------------------


def build_iteration_curve_table(unified_tau: dict[int, dict[str, float]]) -> str:
    summary = load_json(
        REPO_ROOT / "experiments/dvp-sat/iteration_curve/iteration_summary.json"
    )
    rounds = summary["rounds"]

    # Determine best solve rate from unified multi-seed means
    best = max(unified_tau[int(r["round"])]["mean"] for r in rounds)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{On-policy mining iteration curve ($n{=}50$, $\\tau{=}0.05$, budget 500, 5~seeds $\\times$ 200 instances). All rounds use the same threshold. Round 2 peaks; round 3 collapses from distributional drift.}",
        "\\label{tab:iteration_curve}",
        "\\footnotesize",
        "\\begin{tabular}{ccccc}",
        "  \\toprule",
        f"  \\textbf{{Round}} & \\textbf{{Training Data}} & \\textbf{{Hard Neg \\%}} & \\textbf{{Solve Rate}} & \\textbf{{Proactive BTs}} {ROW_END}",
        "  \\midrule",
    ]

    for r in rounds:
        rnd = int(r["round"])
        train = float(r["num_train_examples"])
        hard = float(r["num_hard_neg"])
        hard_pct = 0.0 if train == 0 else hard / train
        ms_entry = unified_tau[rnd]
        solve_txt = pct_ci(ms_entry["mean"], ms_entry["std"])
        pro_bts = f"${with_commas(ms_entry['pro_bts_mean'])} \\pm {with_commas(ms_entry['pro_bts_std'])}$"
        solve_val = ms_entry["mean"]
        cells = [
            f"{rnd}",
            with_commas(train),
            pct(hard_pct),
            solve_txt,
            pro_bts,
        ]
        lines.append(f"  {row_cells(cells, bold=(solve_val == best))} {ROW_END}")

    lines += ["  \\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table: Off-Policy Ablation (multi-seed for 630K, R1, R2)
# ---------------------------------------------------------------------------


def build_offpolicy_ablation_table(
    ms: dict[str, Any], unified_tau: dict[int, dict[str, float]]
) -> str:
    summary = load_json(
        REPO_ROOT / "experiments/dvp-sat/iteration_curve/iteration_summary.json"
    )
    large_eval = load_json(
        REPO_ROOT / "experiments/dvp-sat/offpolicy_large_eval/results.json"
    )
    large_cfg = load_json(
        REPO_ROOT / "experiments/dvp-sat/offpolicy_large_verifier_pw1/config.json"
    )

    rounds = {int(r["round"]): r for r in summary["rounds"]}
    r0, r1, r2 = rounds[0], rounds[1], rounds[2]
    lg = _extract_mlp_metrics(large_eval)

    # Multi-seed entries
    ms_offpolicy = ms.get("offpolicy_t01")
    ms_r1 = unified_tau[1]
    ms_r2 = unified_tau[2]

    rows = [
        (
            "Off-policy (original 24K)",
            float(r0["num_train_examples"]),
            "Off-policy",
            float(r0["eval_solve_rate"]),
            float(r0["eval_proactive_bts"]),
            None,
        ),
        (
            "Off-policy (large 630K)",
            float(large_cfg["train_examples"]) + float(large_cfg["val_examples"]),
            "Off-policy",
            float(lg["solve_rate"]),
            float(lg["proactive_backtracks"]),
            ms_offpolicy,
        ),
        (
            "On-policy round 1 (179K)",
            float(r1["num_train_examples"]),
            "On-policy",
            float(r1["eval_solve_rate"]),
            float(r1["eval_proactive_bts"]),
            ms_r1,
        ),
        (
            "On-policy round 2 (408K)",
            float(r2["num_train_examples"]),
            "On-policy",
            float(r2["eval_solve_rate"]),
            float(r2["eval_proactive_bts"]),
            ms_r2,
        ),
    ]

    # Best by multi-seed mean where available
    best = max((ms_e["mean"] if ms_e else solve) for _, _, _, solve, _, ms_e in rows)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Off-policy scaling ablation ($n{=}50$). Off-policy rows use $\\tau{=}0.1$; on-policy rows use $\\tau{=}0.05$ (matching Table~\\ref{tab:iteration_curve}). Multi-seed entries report 5-seed mean $\\pm$ std. 630K off-policy examples produce zero proactive backtracks.}",
        "\\label{tab:offpolicy_ablation}",
        "\\footnotesize",
        "\\begin{tabular}{lcccc}",
        "  \\toprule",
        f"  \\textbf{{Training Data}} & \\textbf{{Size}} & \\textbf{{Source}} & \\textbf{{Solve Rate}} & \\textbf{{Proactive BTs}} {ROW_END}",
        "  \\midrule",
    ]

    for label, size, source, solve, proactive, ms_entry in rows:
        if ms_entry:
            solve_txt = pct_ci(ms_entry["mean"], ms_entry["std"])
            pro_txt = with_commas(ms_entry.get("pro_bts_mean", proactive))
        else:
            solve_txt = pct(solve)
            pro_txt = with_commas(proactive)
        solve_val = ms_entry["mean"] if ms_entry else solve
        cells = [label, with_commas(size), source, solve_txt, pro_txt]
        lines.append(f"  {row_cells(cells, bold=(solve_val == best))} {ROW_END}")

    lines += ["  \\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table: Budget Curves (multi-seed for 200, 500, 1000, 2000)
# ---------------------------------------------------------------------------


def build_budget_curves_table(ms: dict[str, Any]) -> str:
    reactive = load_json(
        REPO_ROOT / "experiments/dvp-sat/sweep_p1_budget/summary_p1_reactive.json"
    )
    proactive = load_json(
        REPO_ROOT / "experiments/dvp-sat/sweep_p1_budget/summary_p1_proactive.json"
    )
    r_map = {int(e["max_backtracks"]): e for e in reactive["entries"]}
    p_map = {int(e["max_backtracks"]): e for e in proactive["entries"]}
    budgets = [50, 100, 200, 500, 1000, 2000]

    # Build multi-seed lookup
    ms_reactive = {}
    ms_proactive = {}
    for key, val in ms.items():
        if key.startswith("reactive_b"):
            b = int(key.replace("reactive_b", ""))
            ms_reactive[b] = val
        elif key.startswith("proactive_b"):
            b = int(key.replace("proactive_b", ""))
            ms_proactive[b] = val

    # Find first overtake budget using multi-seed means
    overtake = None
    for b in budgets:
        ms_r = ms_reactive.get(b)
        ms_p = ms_proactive.get(b)
        r_val = ms_r["mean"] if ms_r else float(r_map[b]["solve_rate"])
        p_val = ms_p["mean"] if ms_p else float(p_map[b]["solve_rate"])
        if p_val > r_val:
            overtake = b
            break

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Budget efficiency curves ($n{=}50$, 5~seeds $\\times$ 200 instances where available). Proactive overtakes reactive at budget ${\\sim}500$.}",
        "\\label{tab:budget_curves}",
        "\\footnotesize",
        "\\begin{tabular}{cccc}",
        "  \\toprule",
        f"  \\textbf{{Budget}} & \\textbf{{Reactive}} & \\textbf{{Proactive}} & \\textbf{{$\\Delta$}} {ROW_END}",
        "  \\midrule",
    ]

    for b in budgets:
        ms_r = ms_reactive.get(b)
        ms_p = ms_proactive.get(b)

        if ms_r:
            r_txt = pct_ci(ms_r["mean"], ms_r["std"])
            r_val = ms_r["mean"]
        else:
            r_val = float(r_map[b]["solve_rate"])
            r_txt = pct(r_val)

        if ms_p:
            p_txt = pct_ci(ms_p["mean"], ms_p["std"])
            p_val = ms_p["mean"]
        else:
            p_val = float(p_map[b]["solve_rate"])
            p_txt = pct(p_val)

        p_txt = maybe_bold(p_txt, overtake is not None and b == overtake)
        delta = (p_val - r_val) * 100.0
        lines.append(f"  {b} & {r_txt} & {p_txt} & {delta:+.1f}\\% {ROW_END}")

    lines += ["  \\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ms = load_multiseed()
    unified_tau = load_unified_tau()

    generated = [
        write_table("tab_corruption_curve.tex", build_corruption_curve_table(ms)),
        write_table("tab_threshold_curve.tex", build_threshold_curve_table()),
        write_table("tab_oracle_upperbound.tex", build_oracle_upperbound_table(ms)),
        write_table(
            "tab_iteration_curve.tex", build_iteration_curve_table(unified_tau)
        ),
        write_table(
            "tab_offpolicy_ablation.tex",
            build_offpolicy_ablation_table(ms, unified_tau),
        ),
        write_table("tab_budget_curves.tex", build_budget_curves_table(ms)),
    ]
    print("Generated Precision Barrier tables:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
