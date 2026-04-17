#!/usr/bin/env python3
"""Generate revision statistical tables for the paper.

Usage:
    python scripts/generate_paper_tables.py [--latex-dir paper/tables] [--bootstrap-n 10000]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


LOGGER = logging.getLogger("generate_paper_tables")
SEED_REGEX = re.compile(r"^(?P<prefix>.+)-seed(?P<seed>\d+)$")
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 2024)


@dataclass
class SeedResult:
    seed: int
    solve_rate: float  # fraction in [0,1]
    path: Path
    per_instance_solved: np.ndarray | None


@dataclass
class ConditionData:
    name: str
    entries: dict[int, SeedResult]

    def sorted_seed_results(self) -> list[SeedResult]:
        return [self.entries[k] for k in sorted(self.entries)]

    def solve_rates(self) -> np.ndarray:
        return np.asarray(
            [r.solve_rate for r in self.sorted_seed_results()], dtype=float
        )

    def n(self) -> int:
        return len(self.entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--latex-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def _normalize_solve_rate(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v > 1.0:
        v /= 100.0
    if v < 0.0 or v > 1.0:
        return None
    return v


def _extract_solve_rate(payload: dict[str, Any]) -> float | None:
    direct = _normalize_solve_rate(payload.get("solve_rate"))
    if direct is not None:
        return direct

    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict):
        agg = _normalize_solve_rate(aggregate.get("solve_rate"))
        if agg is not None:
            return agg

    results = payload.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        nested = _normalize_solve_rate(results[0].get("solve_rate"))
        if nested is not None:
            return nested
    return None


def _extract_per_instance_solved(payload: dict[str, Any]) -> np.ndarray | None:
    candidates: list[Any] = []
    direct = payload.get("per_instance")
    if isinstance(direct, list):
        candidates.append(direct)

    results = payload.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        nested = results[0].get("per_instance")
        if isinstance(nested, list):
            candidates.append(nested)

    for per_instance in candidates:
        solved: list[float] = []
        for row in per_instance:
            if not isinstance(row, dict):
                return None
            flag = row.get("solved")
            if isinstance(flag, bool):
                solved.append(1.0 if flag else 0.0)
            elif isinstance(flag, (int, float)):
                solved.append(1.0 if float(flag) > 0 else 0.0)
            else:
                return None
        return np.asarray(solved, dtype=float)
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Failed to read %s (%s)", path, exc)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Malformed JSON object at %s", path)
        return None
    return payload


def _find_seed_run_ancestor(
    path: Path, experiments_root: Path
) -> tuple[str, int] | None:
    for parent in path.parents:
        if parent == experiments_root.parent:
            break
        match = SEED_REGEX.match(parent.name)
        if match:
            return match.group("prefix"), int(match.group("seed"))
    return None


def discover_results(experiments_root: Path) -> dict[str, dict[int, list[Path]]]:
    index: dict[str, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    files = list(experiments_root.rglob("results.json"))
    LOGGER.info(
        "Discovered %d results.json files under %s", len(files), experiments_root
    )
    for file_path in files:
        parsed = _find_seed_run_ancestor(file_path, experiments_root)
        if parsed is None:
            continue
        prefix, seed = parsed
        index[prefix][seed].append(file_path)
    LOGGER.info("Indexed %d experiment prefixes", len(index))
    return index


def _path_priority(path: Path, domain: str) -> int:
    text = str(path)
    if domain == "sat":
        prefs = ["/eval_b4096/results.json", "/results.json"]
    else:
        prefs = ["/eval_b2048/results.json", "/results.json"]
    for idx, suffix in enumerate(prefs):
        if text.endswith(suffix):
            return idx
    return len(prefs)


def choose_best_candidate(candidates: list[Path], domain: str) -> Path:
    return sorted(candidates, key=lambda p: (_path_priority(p, domain), len(str(p))))[0]


def load_condition(
    *,
    name: str,
    prefixes: list[str],
    index: dict[str, dict[int, list[Path]]],
    domain: str,
) -> ConditionData:
    entries: dict[int, SeedResult] = {}
    candidate_count = 0
    for prefix in prefixes:
        for seed, paths in index.get(prefix, {}).items():
            candidate_count += len(paths)
            best_path = choose_best_candidate(paths, domain=domain)
            existing = entries.get(seed)
            if existing is not None:
                # Keep current unless new path has better domain-specific priority.
                if _path_priority(best_path, domain) >= _path_priority(
                    existing.path, domain
                ):
                    continue
            payload = _read_json(best_path)
            if payload is None:
                continue
            solve_rate = _extract_solve_rate(payload)
            if solve_rate is None:
                LOGGER.warning("solve_rate missing for %s", best_path)
                continue
            entries[seed] = SeedResult(
                seed=seed,
                solve_rate=solve_rate,
                path=best_path,
                per_instance_solved=_extract_per_instance_solved(payload),
            )

    LOGGER.info(
        "Condition %-24s | prefixes=%d | candidate_files=%d | loaded_seeds=%s | meansolve=%.3f",
        name,
        len(prefixes),
        candidate_count,
        sorted(entries),
        float(np.mean([e.solve_rate for e in entries.values()]))
        if entries
        else float("nan"),
    )
    if not entries:
        LOGGER.warning("No data found for condition '%s' (prefixes=%s)", name, prefixes)
    return ConditionData(name=name, entries=entries)


def pct(v: float) -> str:
    return f"{100.0 * v:.1f}%"


def pct_num(v: float) -> str:
    return f"{100.0 * v:.1f}"


def bootstrap_mean_ci(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])

    try:
        res = stats.bootstrap(
            data=(arr,),
            statistic=np.mean,
            n_resamples=n_bootstrap,
            confidence_level=0.95,
            method="BCa",
            random_state=rng,
        )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning(
            "BCa bootstrap unavailable/failing (%s); using percentile bootstrap", exc
        )
        idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
        sample_means = arr[idx].mean(axis=1)
        return float(np.percentile(sample_means, 2.5)), float(
            np.percentile(sample_means, 97.5)
        )


def paired_bootstrap_diff_ci(
    diffs: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    sample_means = arr[idx].mean(axis=1)
    return (
        float(np.percentile(sample_means, 2.5)),
        float(np.percentile(sample_means, 97.5)),
        float(np.mean(sample_means > 0.0)),
    )


def paired_ttest_and_d(a: ConditionData, b: ConditionData) -> dict[str, Any]:
    common = sorted(set(a.entries).intersection(b.entries))
    if len(common) < 2:
        return {
            "n": len(common),
            "mean_diff": float("nan"),
            "p_value": float("nan"),
            "t_stat": float("nan"),
            "cohen_d": float("nan"),
        }
    av = np.asarray([a.entries[s].solve_rate for s in common], dtype=float)
    bv = np.asarray([b.entries[s].solve_rate for s in common], dtype=float)
    diffs = av - bv
    test = stats.ttest_rel(av, bv)
    std = float(np.std(diffs, ddof=1))
    d = float(np.mean(diffs) / std) if std > 0 else float("nan")
    return {
        "n": len(common),
        "mean_diff": float(np.mean(diffs)),
        "p_value": float(test.pvalue),
        "t_stat": float(test.statistic),
        "cohen_d": d,
    }


def paired_instance_bootstrap(
    a: ConditionData,
    b: ConditionData,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    common = sorted(set(a.entries).intersection(b.entries))
    diffs: list[np.ndarray] = []
    used_seeds: list[int] = []
    for seed in common:
        arr_a = a.entries[seed].per_instance_solved
        arr_b = b.entries[seed].per_instance_solved
        if arr_a is None or arr_b is None:
            continue
        if arr_a.shape[0] != arr_b.shape[0]:
            LOGGER.warning(
                "Per-instance length mismatch for seed=%d (%d vs %d), truncating",
                seed,
                arr_a.shape[0],
                arr_b.shape[0],
            )
        n = min(arr_a.shape[0], arr_b.shape[0])
        diffs.append(arr_a[:n] - arr_b[:n])
        used_seeds.append(seed)

    if not diffs:
        return {
            "seeds": [],
            "n_instances": 0,
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_proxy": float("nan"),
        }

    concat = np.concatenate(diffs)
    ci_low, ci_high, p_proxy = paired_bootstrap_diff_ci(
        concat, n_bootstrap=n_bootstrap, rng=rng
    )
    return {
        "seeds": used_seeds,
        "n_instances": int(concat.shape[0]),
        "mean_diff": float(np.mean(concat)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_proxy": p_proxy,
    }


def format_paired_instance_result(result: dict[str, Any]) -> str:
    if int(result.get("n_instances", 0)) <= 0:
        return "unavailable (per-instance records missing)"
    return (
        f"Δ={pct(float(result['mean_diff']))} "
        f"[{pct_num(float(result['ci_low']))}, {pct_num(float(result['ci_high']))}], "
        f"p_proxy={float(result['p_proxy']):.4f}, "
        f"n_instances={int(result['n_instances'])}"
    )


def format_mean_ci(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> str:
    mean = float(np.mean(values))
    low, high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, rng=rng)
    return f"{pct(mean)} [{pct_num(low)}, {pct_num(high)}]"


def format_seed_values(condition: ConditionData) -> str:
    if not condition.entries:
        return "-"
    return ", ".join(
        f"{seed}:{100 * condition.entries[seed].solve_rate:.1f}"
        for seed in sorted(condition.entries)
    )


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_tex(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")
    LOGGER.info("Wrote LaTeX table: %s", path)


def build_table1(
    conditions: list[ConditionData],
    latex_dir: Path,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> None:
    print("\n" + "=" * 100)
    print("TABLE 1: Main Architecture Comparison (SAT n=50)")
    print("=" * 100)
    print(f"{'Condition':<24} {'Solve rate [95% CI]':<28} {'n':>3}  Seeds (solve%)")
    rows_tex: list[str] = []

    for cond in conditions:
        rates = cond.solve_rates()
        if rates.size == 0:
            summary = "missing"
        else:
            summary = format_mean_ci(rates, n_bootstrap=n_bootstrap, rng=rng)
        seeds_text = format_seed_values(cond)
        print(f"{cond.name:<24} {summary:<28} {cond.n():>3}  {seeds_text}")
        rows_tex.append(
            f"{latex_escape(cond.name)} & {summary} & {cond.n()} & {latex_escape(seeds_text)} \\\\"
        )

    tex = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Main architecture comparison on SAT $n{=}50$.}",
            "\\label{tab:main_arch_sat_n50}",
            "\\small",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Condition & Solve rate [95\\% CI] & n & Seed values (\\%) \\\\",
            "\\midrule",
            *rows_tex,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    write_tex(latex_dir / "table1_main_architecture_sat_n50.tex", tex)


def _factorial_seed_vector(*conditions: ConditionData) -> tuple[list[int], np.ndarray]:
    common = sorted(set.intersection(*(set(c.entries) for c in conditions)))
    if not common:
        return [], np.asarray([], dtype=float)
    data = np.stack(
        [[c.entries[s].solve_rate for s in common] for c in conditions], axis=0
    )
    return common, data


def build_2x2_ablation_table(
    *,
    title: str,
    tex_name: str,
    latex_label: str,
    latex_dir: Path,
    ssa0: ConditionData,
    ssa32: ConditionData,
    causal0: ConditionData,
    causal32: ConditionData,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for cond in (ssa0, ssa32, causal0, causal32):
        rates = cond.solve_rates()
        summary = (
            "missing"
            if rates.size == 0
            else format_mean_ci(rates, n_bootstrap=n_bootstrap, rng=rng)
        )
        print(
            f"{cond.name:<24} {summary:<28} n={cond.n()} seeds=[{format_seed_values(cond)}]"
        )

    # seed-level factorial effects (requires all four present for each seed)
    common, matrix = _factorial_seed_vector(ssa0, ssa32, causal0, causal32)
    if matrix.size == 0:
        LOGGER.warning("Cannot compute 2x2 effects for %s (no common seeds)", title)
        effect_text = "Effects unavailable (no common seeds)."
        effects_rows = []
    else:
        # matrix order: [ssa0, ssa32, causal0, causal32]
        slot_ssa = matrix[1] - matrix[0]
        slot_causal = matrix[3] - matrix[2]
        mask_0 = matrix[2] - matrix[0]
        mask_32 = matrix[3] - matrix[1]
        slot_main = 0.5 * (slot_ssa + slot_causal)
        mask_main = 0.5 * (mask_0 + mask_32)
        interaction = slot_causal - slot_ssa

        slot_ssa_test = paired_ttest_and_d(ssa32, ssa0)
        slot_causal_test = paired_ttest_and_d(causal32, causal0)
        per_inst_mask0 = paired_instance_bootstrap(
            ssa0, causal0, n_bootstrap=n_bootstrap, rng=rng
        )
        per_inst_mask32 = paired_instance_bootstrap(
            ssa32, causal32, n_bootstrap=n_bootstrap, rng=rng
        )

        LOGGER.info(
            "%s effects | common_seeds=%s | slot_main=%.4f | mask_main=%.4f | interaction=%.4f",
            title,
            common,
            float(np.mean(slot_main)),
            float(np.mean(mask_main)),
            float(np.mean(interaction)),
        )

        effect_lines = []
        effects_rows = []
        for name, vec in (
            ("Slot effect (SSA)", slot_ssa),
            ("Slot effect (Causal)", slot_causal),
            ("Mask effect (n_slots=0)", mask_0),
            ("Mask effect (n_slots=32)", mask_32),
            ("Main slot effect", slot_main),
            ("Main mask effect", mask_main),
            ("Interaction", interaction),
        ):
            lo, hi = bootstrap_mean_ci(vec, n_bootstrap=n_bootstrap, rng=rng)
            effect_lines.append(
                f"- {name}: {pct(float(np.mean(vec)))} [{pct_num(lo)}, {pct_num(hi)}]"
            )
            effects_rows.append(
                f"{latex_escape(name)} & {pct(float(np.mean(vec)))} [{pct_num(lo)}, {pct_num(hi)}] \\\\"
            )
        effect_text = "\n".join(effect_lines)

        print("\nEffects:")
        print(effect_text)
        print(
            f"Paired t-test slot effect (SSA): n={slot_ssa_test['n']}, "
            f"p={slot_ssa_test['p_value']:.4g}, Cohen's d={slot_ssa_test['cohen_d']:.3f}"
        )
        print(
            f"Paired t-test slot effect (Causal): n={slot_causal_test['n']}, "
            f"p={slot_causal_test['p_value']:.4g}, Cohen's d={slot_causal_test['cohen_d']:.3f}"
        )
        print(
            "Paired-instance bootstrap SSA-Causal (n_slots=0): "
            f"{format_paired_instance_result(per_inst_mask0)}"
        )
        print(
            "Paired-instance bootstrap SSA-Causal (n_slots=32): "
            f"{format_paired_instance_result(per_inst_mask32)}"
        )

    cell_rows = []
    for cond in (ssa0, ssa32, causal0, causal32):
        rates = cond.solve_rates()
        summary = (
            "missing"
            if rates.size == 0
            else format_mean_ci(rates, n_bootstrap=n_bootstrap, rng=rng)
        )
        cell_rows.append(f"{latex_escape(cond.name)} & {summary} & {cond.n()} \\\\")

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(title)}.}}",
        f"\\label{{{latex_label}}}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Condition & Solve rate [95\\% CI] & n \\\\",
        "\\midrule",
        *cell_rows,
        "\\midrule",
        "\\multicolumn{3}{l}{\\textbf{Estimated effects}} \\\\",
        *effects_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_tex(latex_dir / tex_name, "\n".join(tex_lines))


def build_table4_trace_mask_sat(
    enriched_ssa: ConditionData,
    enriched_causal: ConditionData,
    minimal_ssa: ConditionData,
    minimal_causal: ConditionData,
    latex_dir: Path,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> None:
    build_2x2_ablation_table(
        title="TABLE 4: Trace Format × Mask (Factorial, SAT n=50)",
        tex_name="table4_trace_mask_sat_n50.tex",
        latex_label="tab:trace_mask_sat_n50",
        latex_dir=latex_dir,
        ssa0=minimal_ssa,
        ssa32=enriched_ssa,
        causal0=minimal_causal,
        causal32=enriched_causal,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )


def load_dagger_rounds(experiments_root: Path, seed: int) -> dict[int, float]:
    metrics_path = (
        experiments_root / f"dagger-sat-n50-causal-seed{seed}" / "metrics.json"
    )
    payload = _read_json(metrics_path)
    if payload is None:
        LOGGER.warning("Missing DAgger metrics for seed=%d", seed)
        return {}
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        return {}

    out: dict[int, float] = {}
    for row in rounds:
        if not isinstance(row, dict):
            continue
        rid = row.get("round")
        rate = _normalize_solve_rate(row.get("solve_rate"))
        if isinstance(rid, int) and rate is not None:
            out[rid] = rate
    LOGGER.info("Loaded DAgger seed=%d rounds=%s", seed, sorted(out))
    return out


def build_table5_dagger(
    experiments_root: Path,
    latex_dir: Path,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> None:
    print("\n" + "=" * 100)
    print("TABLE 5: DAgger Results")
    print("=" * 100)

    seed42 = load_dagger_rounds(experiments_root, 42)
    seed123 = load_dagger_rounds(experiments_root, 123)
    all_rounds = sorted(set(seed42).union(seed123))
    print(f"{'Round':<8} {'Seed42':>8} {'Seed123':>8} {'Mean [95% CI]':>22}")

    rows_tex = []
    for rid in all_rounds:
        vals = [v for v in [seed42.get(rid), seed123.get(rid)] if v is not None]
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            summary = "missing"
        else:
            summary = format_mean_ci(arr, n_bootstrap=n_bootstrap, rng=rng)
        s42 = "-" if rid not in seed42 else pct(seed42[rid])
        s123 = "-" if rid not in seed123 else pct(seed123[rid])
        print(f"{rid:<8} {s42:>8} {s123:>8} {summary:>22}")
        rows_tex.append(f"{rid} & {s42} & {s123} & {summary} \\\\")

    tex = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{DAgger SAT n=50 round-by-round solve rates.}",
            "\\label{tab:dagger_sat_n50}",
            "\\small",
            "\\begin{tabular}{rccc}",
            "\\toprule",
            "Round & Seed 42 & Seed 123 & Mean [95\\% CI] \\\\",
            "\\midrule",
            *rows_tex,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    write_tex(latex_dir / "table5_dagger_results.tex", tex)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    experiments_root = args.experiments_dir
    if not experiments_root.is_absolute():
        experiments_root = repo_root / experiments_root
    latex_dir = args.latex_dir
    if not latex_dir.is_absolute():
        latex_dir = repo_root / latex_dir
    latex_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    index = discover_results(experiments_root)

    sat_cond = {
        "SSA n_slots=0": load_condition(
            name="SSA n_slots=0",
            prefixes=["slot-ablation-sat-n50-nslots0-selective_ssa"],
            index=index,
            domain="sat",
        ),
        "SSA n_slots=32": load_condition(
            name="SSA n_slots=32",
            prefixes=["sat-n50-enriched-selective_ssa"],
            index=index,
            domain="sat",
        ),
        "Causal n_slots=32": load_condition(
            name="Causal n_slots=32",
            prefixes=["sat-n50-enriched-full_causal"],
            index=index,
            domain="sat",
        ),
        "Causal n_slots=0": load_condition(
            name="Causal n_slots=0",
            prefixes=["slot-ablation-sat-n50-nslots0-full_causal"],
            index=index,
            domain="sat",
        ),
        "LSTM continuous": load_condition(
            name="LSTM continuous",
            prefixes=["lstm-sat-n50-continuous"],
            index=index,
            domain="sat",
        ),
        "LSTM block_reset": load_condition(
            name="LSTM block_reset",
            prefixes=["lstm-sat-n50-block_reset"],
            index=index,
            domain="sat",
        ),
        "Minimal + SSA": load_condition(
            name="Minimal + SSA",
            prefixes=["sat-n50-minimal-selective_ssa"],
            index=index,
            domain="sat",
        ),
        "Minimal + Causal": load_condition(
            name="Minimal + Causal",
            prefixes=["sat-n50-minimal-full_causal"],
            index=index,
            domain="sat",
        ),
        "State-only (100k)": load_condition(
            name="State-only (100k)",
            prefixes=["sat-n50-state-only-100k", "sat-n50-stateonly-100k"],
            index=index,
            domain="sat",
        ),
        "State-only (50k)": load_condition(
            name="State-only (50k)",
            prefixes=["sat-n50-state-only-50k", "sat-n50-stateonly-50k"],
            index=index,
            domain="sat",
        ),
    }

    build_table1(
        [
            sat_cond["SSA n_slots=0"],
            sat_cond["SSA n_slots=32"],
            sat_cond["Causal n_slots=32"],
            sat_cond["Causal n_slots=0"],
            sat_cond["LSTM continuous"],
            sat_cond["LSTM block_reset"],
            sat_cond["Minimal + SSA"],
            sat_cond["Minimal + Causal"],
            sat_cond["State-only (100k)"],
            sat_cond["State-only (50k)"],
        ],
        latex_dir=latex_dir,
        n_bootstrap=args.bootstrap_n,
        rng=rng,
    )

    build_2x2_ablation_table(
        title="TABLE 2: Slot Ablation 2×2 (SAT n=50)",
        tex_name="table2_slot_ablation_sat_n50.tex",
        latex_label="tab:slot_ablation_sat_n50",
        latex_dir=latex_dir,
        ssa0=sat_cond["SSA n_slots=0"],
        ssa32=sat_cond["SSA n_slots=32"],
        causal0=sat_cond["Causal n_slots=0"],
        causal32=sat_cond["Causal n_slots=32"],
        n_bootstrap=args.bootstrap_n,
        rng=rng,
    )

    gc_cond = {
        "GC SSA n_slots=0": load_condition(
            name="GC SSA n_slots=0",
            prefixes=["slot-ablation-gc-nslots0-selective_ssa"],
            index=index,
            domain="gc",
        ),
        "GC SSA n_slots=32": load_condition(
            name="GC SSA n_slots=32",
            prefixes=[
                "factorial-eval-gc-enriched-selective_ssa",
                "factorial-gc-enriched-selective_ssa",
            ],
            index=index,
            domain="gc",
        ),
        "GC Causal n_slots=0": load_condition(
            name="GC Causal n_slots=0",
            prefixes=["slot-ablation-gc-nslots0-full_causal"],
            index=index,
            domain="gc",
        ),
        "GC Causal n_slots=32": load_condition(
            name="GC Causal n_slots=32",
            prefixes=["e3-eval-gc-full-causal"],
            index=index,
            domain="gc",
        ),
    }

    build_2x2_ablation_table(
        title="TABLE 3: Slot Ablation 2×2 (GC n=30)",
        tex_name="table3_slot_ablation_gc_n30.tex",
        latex_label="tab:slot_ablation_gc_n30",
        latex_dir=latex_dir,
        ssa0=gc_cond["GC SSA n_slots=0"],
        ssa32=gc_cond["GC SSA n_slots=32"],
        causal0=gc_cond["GC Causal n_slots=0"],
        causal32=gc_cond["GC Causal n_slots=32"],
        n_bootstrap=args.bootstrap_n,
        rng=rng,
    )

    build_table4_trace_mask_sat(
        enriched_ssa=sat_cond["SSA n_slots=32"],
        enriched_causal=sat_cond["Causal n_slots=32"],
        minimal_ssa=sat_cond["Minimal + SSA"],
        minimal_causal=sat_cond["Minimal + Causal"],
        latex_dir=latex_dir,
        n_bootstrap=args.bootstrap_n,
        rng=rng,
    )

    build_table5_dagger(
        experiments_root=experiments_root,
        latex_dir=latex_dir,
        n_bootstrap=args.bootstrap_n,
        rng=rng,
    )

    LOGGER.info("Completed paper table generation. Outputs saved to %s", latex_dir)


if __name__ == "__main__":
    main()
