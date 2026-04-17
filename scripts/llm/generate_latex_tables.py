#!/usr/bin/env python3
"""Generate LaTeX tables from MATH evaluation outputs.

This script reads evaluation JSONs plus candidate metadata and produces
publication-ready LaTeX tables for the paper:
  - main_results.tex
  - per_level.tex
  - per_category.tex
  - ablation.tex (optional if ablation files exist)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LOGGER = logging.getLogger(__name__)

RANDOM_KEYS = {"random"}
MV_KEYS = {"majority_vote"}
ORACLE_KEYS = {"oracle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from MATH eval results."
    )
    parser.add_argument(
        "--eval_dir",
        default="experiments/math-14b",
        help="Directory containing eval JSONs",
    )
    parser.add_argument(
        "--candidates",
        default="experiments/math-14b/candidates_test_full_5000.jsonl",
        help="Path to candidates JSONL",
    )
    parser.add_argument(
        "--output_dir",
        default="paper/tables",
        help="Directory to write .tex files",
    )
    parser.add_argument(
        "--main_eval",
        default="eval_full_5000.json",
        help="Filename of main eval JSON",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}") from exc
    return records


def problem_prefix(problem: str, n_chars: int = 200) -> str:
    return problem[:n_chars]


def format_percent(value: float) -> str:
    return f"{value:.1f}"


def format_delta(value: float) -> str:
    return f"{value:+.1f}"


def bold_if(text: str, condition: bool) -> str:
    return f"\\textbf{{{text}}}" if condition else text


def pretty_category(raw: str) -> str:
    formatted = raw.replace("_", " ").title()
    return formatted.replace(" And ", " \\& ")


def select_key(strategies: Dict[str, dict], keys: Iterable[str], label: str) -> str:
    matches = [k for k in strategies.keys() if k in keys]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} key in strategies, found {matches}"
        )
    return matches[0]


def select_primary_key(strategies: Dict[str, dict]) -> str:
    excluded = RANDOM_KEYS | MV_KEYS | ORACLE_KEYS
    candidates = [k for k in strategies.keys() if k not in excluded]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one primary strategy key (non-random/MV/oracle), "
            f"found {candidates}"
        )
    return candidates[0]


def strategy_accuracy(strategy: dict) -> float:
    if "accuracy" not in strategy:
        raise KeyError("Strategy entry missing accuracy field")
    return strategy["accuracy"] * 100.0


def latex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def load_math_levels(cache_dir: str | None) -> Dict[str, str]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'datasets'. Install datasets to load MATH."
        ) from exc
    configs = get_dataset_config_names("EleutherAI/hendrycks_math", cache_dir=cache_dir)
    LOGGER.info("MATH dataset configs: %s", configs)
    level_by_prefix: Dict[str, str] = {}
    total = 0
    for config in configs:
        ds = load_dataset(
            "EleutherAI/hendrycks_math", config, split="test", cache_dir=cache_dir
        )
        LOGGER.info("Loaded %d test problems from %s", len(ds), config)
        for item in ds:
            total += 1
            prefix = problem_prefix(item["problem"])
            level = item["level"]
            if prefix in level_by_prefix and level_by_prefix[prefix] != level:
                LOGGER.warning(
                    "Conflicting levels for prefix (keeping first): %s vs %s",
                    level_by_prefix[prefix], level,
                )
                continue
            level_by_prefix[prefix] = level
    LOGGER.info(
        "Loaded %d MATH test problems (%d unique prefixes)",
        total,
        len(level_by_prefix),
    )
    return level_by_prefix


def build_idx_metadata(
    candidates: List[dict], level_by_prefix: Dict[str, str]
) -> Dict[int, dict]:
    idx_meta: Dict[int, dict] = {}
    missing_levels: List[int] = []
    for i, item in enumerate(candidates):
        idx = i
        prefix = problem_prefix(item["problem"])
        level = level_by_prefix.get(prefix)
        if level is None:
            missing_levels.append(idx)
            continue
        idx_meta[idx] = {
            "category": item["category"],
            "level": level,
            "problem": item["problem"],
        }
    if missing_levels:
        raise ValueError(
            f"Missing MATH levels for {len(missing_levels)} problems. "
            f"First 10 missing idx: {missing_levels[:10]}"
        )
    return idx_meta


def compute_per_group_stats(
    per_problem: List[dict], idx_meta: Dict[int, dict]
) -> Tuple[dict, dict, dict]:
    per_level = {}
    per_category = {}
    totals = {"n": 0, "mv": 0, "verifier": 0, "oracle": 0}
    missing_meta: List[int] = []

    for i, row in enumerate(per_problem):
        idx = i
        meta = idx_meta.get(idx)
        if meta is None:
            missing_meta.append(idx)
            continue
        mv_correct = int(row["mv_correct"])
        verifier_correct = int(row["verifier_correct"])
        if "oracle" in row:
            oracle_correct = int(row["oracle"])
        elif "oracle_correct" in row:
            oracle_correct = int(row["oracle_correct"])
        else:
            raise KeyError(f"Missing oracle correctness for idx={idx}")

        level = meta["level"]
        category = meta["category"]

        if level not in per_level:
            per_level[level] = {"n": 0, "mv": 0, "verifier": 0, "oracle": 0}
        if category not in per_category:
            per_category[category] = {"n": 0, "mv": 0, "verifier": 0, "oracle": 0}

        for bucket in (per_level[level], per_category[category], totals):
            bucket["n"] += 1
            bucket["mv"] += mv_correct
            bucket["verifier"] += verifier_correct
            bucket["oracle"] += oracle_correct

    if missing_meta:
        raise ValueError(
            f"Missing candidate metadata for {len(missing_meta)} problems. "
            f"First 10 missing idx: {missing_meta[:10]}"
        )

    return per_level, per_category, totals


def accuracy_from_bucket(bucket: dict, key: str) -> float:
    if bucket["n"] == 0:
        return 0.0
    return bucket[key] / bucket["n"] * 100.0


def compute_gap(verifier: float, mv: float, oracle: float) -> float:
    if oracle == mv:
        return 0.0
    return (verifier - mv) / (oracle - mv) * 100.0


def format_table_main(strategies: Dict[str, dict], num_candidates: int) -> str:
    random_key = select_key(strategies, RANDOM_KEYS, "random")
    mv_key = select_key(strategies, MV_KEYS, "majority_vote")
    oracle_key = select_key(strategies, ORACLE_KEYS, "oracle")
    primary_key = select_primary_key(strategies)

    random_acc = strategy_accuracy(strategies[random_key])
    mv_acc = strategy_accuracy(strategies[mv_key])
    verifier_acc = strategy_accuracy(strategies[primary_key])
    oracle_acc = strategy_accuracy(strategies[oracle_key])

    best_non_oracle = max(
        [
            ("Random", random_acc),
            ("Majority Vote", mv_acc),
            ("Slot Verifier", verifier_acc),
        ],
        key=lambda x: x[1],
    )[0]

    k_str = str(num_candidates)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{MATH benchmark results with Ministral-3-14B-Base. "
        f"The slot verifier selects the highest-scored candidate from $K{{=}}{k_str}$ "
        "samples. All methods use the same candidate pool.}"
    )
    lines.append("\\label{tab:main-results}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Strategy} & \\textbf{Accuracy (\\%)} & \\textbf{$\\Delta$ vs MV} \\\\"
    )
    lines.append("\\midrule")

    rows = [
        ("Random", random_acc, random_acc - mv_acc),
        ("Majority Vote", mv_acc, None),
        ("Slot Verifier", verifier_acc, verifier_acc - mv_acc),
        ("Oracle (upper bound)", oracle_acc, oracle_acc - mv_acc),
    ]

    for name, acc, delta in rows:
        is_best = name == best_non_oracle
        if name == "Oracle (upper bound)":
            lines.append("\\midrule")
        acc_str = format_percent(acc)
        delta_str = "---" if delta is None else format_delta(delta)
        name_str = bold_if(name, is_best)
        acc_str = bold_if(acc_str, is_best)
        delta_str = bold_if(delta_str, is_best)
        lines.append(f"{name_str} & {acc_str} & {delta_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def format_table_per_level(per_level: dict, totals: dict) -> Tuple[str, str]:
    level_rows = []
    for level_name, bucket in per_level.items():
        mv_acc = accuracy_from_bucket(bucket, "mv")
        verifier_acc = accuracy_from_bucket(bucket, "verifier")
        oracle_acc = accuracy_from_bucket(bucket, "oracle")
        delta = verifier_acc - mv_acc
        gap = compute_gap(verifier_acc, mv_acc, oracle_acc)
        level_rows.append(
            {
                "level": level_name,
                "n": bucket["n"],
                "mv": mv_acc,
                "verifier": verifier_acc,
                "oracle": oracle_acc,
                "delta": delta,
                "gap": gap,
            }
        )

    level_rows.sort(key=lambda r: int(r["level"].split()[-1]))
    best_row = max(level_rows, key=lambda r: r["verifier"])
    best_level = best_row["level"].replace(" ", "~")
    improvements = sum(1 for r in level_rows if r["delta"] >= 0)
    total_levels = len(level_rows)
    if improvements == total_levels:
        improvement_clause = "The verifier improves over majority vote at every level"
    else:
        improvement_clause = (
            f"The verifier improves over majority vote in {improvements}/{total_levels} "
            "levels"
        )

    caption = (
        "Accuracy by MATH difficulty level. "
        f"{improvement_clause}, with the largest absolute gain at {best_level}."
    )

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:per-level}")
    lines.append("\\begin{tabular}{lrcccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Level} & \\textbf{$N$} & \\textbf{MV} & \\textbf{Verifier} & \\textbf{$\\Delta$} & \\textbf{Gap\\%} \\\\"
    )
    lines.append("\\midrule")

    for row in level_rows:
        is_best = row is best_row
        cells = [
            row["level"],
            str(row["n"]),
            format_percent(row["mv"]),
            format_percent(row["verifier"]),
            format_delta(row["delta"]),
            format_percent(row["gap"]),
        ]
        cells = [bold_if(cell, is_best) for cell in cells]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    all_mv = accuracy_from_bucket(totals, "mv")
    all_verifier = accuracy_from_bucket(totals, "verifier")
    all_oracle = accuracy_from_bucket(totals, "oracle")
    all_delta = all_verifier - all_mv
    all_gap = compute_gap(all_verifier, all_mv, all_oracle)
    all_cells = [
        "All",
        str(totals["n"]),
        format_percent(all_mv),
        format_percent(all_verifier),
        format_delta(all_delta),
        format_percent(all_gap),
    ]
    lines.append(" & ".join(all_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines), best_level


def format_table_per_category(per_category: dict) -> str:
    rows = []
    for category, bucket in per_category.items():
        mv_acc = accuracy_from_bucket(bucket, "mv")
        verifier_acc = accuracy_from_bucket(bucket, "verifier")
        oracle_acc = accuracy_from_bucket(bucket, "oracle")
        delta = verifier_acc - mv_acc
        rows.append(
            {
                "category": category,
                "n": bucket["n"],
                "mv": mv_acc,
                "verifier": verifier_acc,
                "oracle": oracle_acc,
                "delta": delta,
            }
        )

    rows.sort(key=lambda r: r["delta"], reverse=True)
    best_row = max(rows, key=lambda r: r["verifier"])
    improvements = sum(1 for r in rows if r["delta"] >= 0)
    total_categories = len(rows)
    if improvements == total_categories:
        improvement_clause = "The verifier improves in all categories"
    else:
        improvement_clause = (
            f"The verifier improves in {improvements}/{total_categories} categories"
        )

    top_two = rows[:2]
    if top_two:
        top_strings = [
            f"{pretty_category(r['category'])} ({format_delta(r['delta'])})"
            for r in top_two
        ]
        top_clause = " and ".join(top_strings)
        gain_clause = f"with particularly large gains in {top_clause}"
    else:
        gain_clause = ""

    caption = "Accuracy by MATH category. " + improvement_clause
    if gain_clause:
        caption = f"{caption}, {gain_clause}."
    else:
        caption = f"{caption}."

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:per-category}")
    lines.append("\\begin{tabular}{lrcccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Category} & \\textbf{$N$} & \\textbf{MV} & \\textbf{Verifier} & \\textbf{Oracle} & \\textbf{$\\Delta$} \\\\"
    )
    lines.append("\\midrule")

    for row in rows:
        is_best = row is best_row
        cells = [
            pretty_category(row["category"]),
            str(row["n"]),
            format_percent(row["mv"]),
            format_percent(row["verifier"]),
            format_percent(row["oracle"]),
            format_delta(row["delta"]),
        ]
        cells = [bold_if(cell, is_best) for cell in cells]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def collect_ablation_rows(
    eval_dir: Path,
    main_eval: Path,
    ablation_specs: List[Tuple[str, str, str]],
) -> Tuple[List[dict], float, int]:
    rows = []
    mv_accs = []
    totals = []

    def load_eval_metrics(path: Path) -> Tuple[float, float, int, str]:
        data = load_json(path)
        strategies = data["strategies"]
        mv_key = select_key(strategies, MV_KEYS, "majority_vote")
        mv_acc = strategy_accuracy(strategies[mv_key])
        total = strategies[mv_key].get("total")
        primary_key = select_primary_key(strategies)
        method_acc = strategy_accuracy(strategies[primary_key])
        return mv_acc, method_acc, total, primary_key

    for filename, label, params in ablation_specs:
        path = eval_dir / filename
        if not path.exists():
            continue
        mv_acc, method_acc, total, primary_key = load_eval_metrics(path)
        mv_accs.append(mv_acc)
        if total is not None:
            totals.append(total)
        rows.append(
            {
                "label": label,
                "params": params,
                "accuracy": method_acc,
                "primary_key": primary_key,
            }
        )

    mv_main, full_acc, total_main, primary_key = load_eval_metrics(main_eval)
    mv_accs.append(mv_main)
    if total_main is not None:
        totals.append(total_main)
    rows.append(
        {
            "label": "Slot Verifier (full)",
            "params": "552M",
            "accuracy": full_acc,
            "primary_key": primary_key,
            "full": True,
        }
    )

    if not rows:
        raise ValueError("No ablation rows collected.")

    # Use MV from the first ablation file (all ablation evals used the same test set)
    baseline_mv = mv_accs[0] if mv_accs else 0.0

    total = totals[0] if totals else 0
    return rows, baseline_mv, total


def format_table_ablation(
    eval_dir: Path,
    main_eval: Path,
    ablation_specs: List[Tuple[str, str, str]],
) -> str | None:
    if not any((eval_dir / spec[0]).exists() for spec in ablation_specs):
        return None

    rows, mv_acc, total = collect_ablation_rows(eval_dir, main_eval, ablation_specs)

    for row in rows:
        row["delta"] = row["accuracy"] - mv_acc

    best_row = max(rows, key=lambda r: r["accuracy"])
    full_row = next(r for r in rows if r.get("full"))

    drop_candidates = [r for r in rows if r is not full_row]
    worst_drop = min(
        drop_candidates, key=lambda r: r["accuracy"] - full_row["accuracy"]
    )
    worst_delta = worst_drop["accuracy"] - full_row["accuracy"]
    lora_row = next((r for r in rows if "LoRA ORM" in r["label"]), None)
    lora_clause = ""
    if lora_row is not None:
        lora_delta = lora_row["accuracy"] - full_row["accuracy"]
        lora_clause = (
            f" while the capacity-matched LoRA ORM trails full slots by "
            f"{format_delta(lora_delta)}"
        )

    caption = (
        f"Ablation study on {latex_int(total)} MATH test problems. "
        f"The largest drop comes from {worst_drop['label']} "
        f"({format_delta(worst_delta)}){lora_clause}."
    )

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Model} & \\textbf{Params} & \\textbf{Accuracy (\\%)} & \\textbf{$\\Delta$ vs MV} \\\\"
    )
    lines.append("\\midrule")
    lines.append(f"Majority Vote & --- & {format_percent(mv_acc)} & --- \\\\")
    lines.append("\\midrule")

    for row in rows:
        is_best = row is best_row
        label = bold_if(row["label"], is_best)
        params = bold_if(row["params"], is_best)
        acc = bold_if(format_percent(row["accuracy"]), is_best)
        delta = bold_if(format_delta(row["delta"]), is_best)
        lines.append(f"{label} & {params} & {acc} & {delta} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    eval_dir = Path(args.eval_dir)
    main_eval_path = eval_dir / args.main_eval
    candidates_path = Path(args.candidates)
    output_dir = Path(args.output_dir)

    LOGGER.info("Loading main eval from %s", main_eval_path)
    eval_data = load_json(main_eval_path)
    strategies = eval_data["strategies"]
    per_problem = eval_data["per_problem"]
    LOGGER.info("Loaded %d per-problem entries", len(per_problem))

    LOGGER.info("Loading candidates from %s", candidates_path)
    candidates = load_jsonl(candidates_path)
    LOGGER.info("Loaded %d candidate problems", len(candidates))

    if not candidates:
        raise ValueError("Candidates file is empty; cannot infer K.")
    if "candidates" not in candidates[0]:
        raise KeyError("Candidates JSONL missing 'candidates' field.")
    candidate_lengths = {len(item["candidates"]) for item in candidates if "candidates" in item}
    if len(candidate_lengths) != 1:
        raise ValueError(f"Inconsistent candidate counts found: {sorted(candidate_lengths)}")
    num_candidates = candidate_lengths.pop()
    LOGGER.info("Detected %d candidates per problem", num_candidates)

    hf_home = os.environ.get("HF_HOME", None)
    level_by_prefix = load_math_levels(hf_home)
    idx_meta = build_idx_metadata(candidates, level_by_prefix)
    per_level, per_category, totals = compute_per_group_stats(per_problem, idx_meta)

    if per_problem:
        sample = per_problem[:3]
        for i, row in enumerate(sample):
            meta = idx_meta[i]
            if "oracle" in row:
                oracle_flag = row["oracle"]
            elif "oracle_correct" in row:
                oracle_flag = row["oracle_correct"]
            else:
                raise KeyError(f"Missing oracle correctness for idx={i}")
            LOGGER.info(
                "Sample idx=%s category=%s level=%s mv=%s verifier=%s oracle=%s",
                i,
                meta["category"],
                meta["level"],
                row["mv_correct"],
                row["verifier_correct"],
                oracle_flag,
            )

    random_key = select_key(strategies, RANDOM_KEYS, "random")
    mv_key = select_key(strategies, MV_KEYS, "majority_vote")
    oracle_key = select_key(strategies, ORACLE_KEYS, "oracle")
    primary_key = select_primary_key(strategies)
    LOGGER.info(
        "Strategy accuracies (%%): random=%.1f mv=%.1f primary(%s)=%.1f oracle=%.1f",
        strategy_accuracy(strategies[random_key]),
        strategy_accuracy(strategies[mv_key]),
        primary_key,
        strategy_accuracy(strategies[primary_key]),
        strategy_accuracy(strategies[oracle_key]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    main_table = format_table_main(strategies, num_candidates)
    per_level_table, best_level = format_table_per_level(per_level, totals)
    per_category_table = format_table_per_category(per_category)

    tables = {
        "main_results.tex": main_table,
        "per_level.tex": per_level_table,
        "per_category.tex": per_category_table,
    }

    ablation_specs = [
        ("eval_orm_baseline.json", "ORM (small head)", "26M"),
        ("eval_reset_per_layer.json", "Slot (reset per layer)", "552M"),
        ("eval_lora_orm.json", "LoRA ORM (capacity-matched)", "341M"),
        ("eval_shuffled_slots.json", "Slot (shuffled identities)", "552M"),
        ("eval_no_write.json", "Slot (no write-back)", "552M"),
    ]
    ablation_table = format_table_ablation(eval_dir, main_eval_path, ablation_specs)
    if ablation_table is not None:
        tables["ablation.tex"] = ablation_table

    for filename, content in tables.items():
        output_path = output_dir / filename
        with output_path.open("w") as f:
            f.write(content + "\n")
        print(content)
        print()
        LOGGER.info("Wrote %s", output_path)

    LOGGER.info("Best per-level verifier row: %s", best_level)
    LOGGER.info("Tables generated successfully.")


if __name__ == "__main__":
    main()
