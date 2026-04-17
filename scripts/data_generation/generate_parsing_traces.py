#!/usr/bin/env python3
"""Generate oracle backtracking parsing traces for SSA training."""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parsing.generator import generate_expression
from parsing.oracle_parser import oracle_parse
from parsing.tokenizer import ParseTrace, serialize_trace


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _trace_to_record(trace: ParseTrace) -> Dict[str, Any]:
    return {
        "sequence": [int(x) for x in trace.sequence],
        "loss_mask": [bool(x) for x in trace.loss_mask],
        "block_ids": [int(x) for x in trace.block_ids],
        "label": str(trace.label),
        "meta": dict(trace.meta),
    }


def _worker_fn(
    worker_args: Tuple[int, int, int, int, float, float, float, float],
) -> Dict[str, Any]:
    (
        worker_index,
        global_seed,
        max_depth,
        max_input_len,
        p_call,
        p_index,
        p_tuple,
        p_neg,
    ) = worker_args

    rng = random.Random(int(global_seed) + int(worker_index))
    tokens = generate_expression(
        max_depth=int(max_depth),
        p_call=float(p_call),
        p_index=float(p_index),
        p_tuple=float(p_tuple),
        p_neg=float(p_neg),
        rng=rng,
    )
    if len(tokens) > int(max_input_len):
        return {
            "ok": False,
            "reason": "input_too_long",
            "input_len": int(len(tokens)),
            "tokens": tokens,
        }

    result = oracle_parse(tokens)
    if not bool(result.success):
        return {
            "ok": False,
            "reason": "parse_failed",
            "input_len": int(len(tokens)),
            "tokens": tokens,
        }

    trace = serialize_trace(tokens, result)
    record = _trace_to_record(trace)
    record["meta"]["expression"] = " ".join(tokens)
    return {
        "ok": True,
        "record": record,
        "tokens": tokens,
        "input_len": int(len(tokens)),
        "seq_len": int(len(trace.sequence)),
        "n_choices": int(trace.meta["n_choices"]),
        "n_backtracks": int(trace.meta["n_backtracks"]),
        "max_depth": int(trace.meta["max_depth"]),
    }


def _generate_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    _set_seed(int(args.seed))
    target = int(args.num_traces)
    bt_target = int(math.ceil(float(target) * float(args.min_backtrack_fraction)))
    workers = max(1, int(args.workers))

    with_backtrack: List[Dict[str, Any]] = []
    without_backtrack: List[Dict[str, Any]] = []

    attempts = 0
    dropped_input_len = 0
    dropped_seq_len = 0
    dropped_parse = 0
    worker_index = 0

    input_lengths: List[int] = []
    seq_lengths: List[int] = []
    choice_counts: List[int] = []
    backtrack_counts: List[int] = []
    max_depths: List[int] = []
    sample_exprs: List[str] = []

    while (
        len(with_backtrack) < bt_target
        or (len(with_backtrack) + len(without_backtrack)) < target
    ):
        remaining = max(
            target - (len(with_backtrack) + len(without_backtrack)),
            bt_target - len(with_backtrack),
            workers,
        )
        batch_size = max(int(workers * 4), int(remaining))
        worker_args: List[Tuple[int, int, int, int, float, float, float, float]] = []
        for _ in range(batch_size):
            worker_args.append(
                (
                    int(worker_index),
                    int(args.seed),
                    int(args.max_depth),
                    int(args.max_input_len),
                    float(args.p_call),
                    float(args.p_index),
                    float(args.p_tuple),
                    float(args.p_neg),
                )
            )
            worker_index += 1

        with Pool(processes=int(workers)) as pool:
            for item in pool.imap_unordered(_worker_fn, worker_args):
                attempts += 1
                if not bool(item.get("ok", False)):
                    reason = str(item.get("reason", "unknown"))
                    if reason == "input_too_long":
                        dropped_input_len += 1
                    else:
                        dropped_parse += 1
                    continue

                record = dict(item["record"])
                if len(record["sequence"]) > int(args.max_seq_len):
                    dropped_seq_len += 1
                    continue

                n_backtracks = int(item["n_backtracks"])
                if n_backtracks > 0:
                    with_backtrack.append(record)
                elif len(with_backtrack) + len(without_backtrack) < target:
                    without_backtrack.append(record)
                else:
                    continue

                input_lengths.append(int(item["input_len"]))
                seq_lengths.append(int(item["seq_len"]))
                choice_counts.append(int(item["n_choices"]))
                backtrack_counts.append(int(item["n_backtracks"]))
                max_depths.append(int(item["max_depth"]))
                if len(sample_exprs) < 3:
                    sample_exprs.append(str(record["meta"]["expression"]))
                    logger.info(
                        "sample_trace idx=%d expr=%s input_len=%d seq_len=%d choices=%d backtracks=%d depth=%d",
                        int(len(sample_exprs) - 1),
                        str(record["meta"]["expression"]),
                        int(item["input_len"]),
                        int(item["seq_len"]),
                        int(item["n_choices"]),
                        int(item["n_backtracks"]),
                        int(item["max_depth"]),
                    )

                kept = int(len(with_backtrack) + len(without_backtrack))
                if kept % 100 == 0:
                    logger.info(
                        "progress kept=%d/%d attempts=%d bt_kept=%d bt_rate=%.3f dropped_input_len=%d dropped_seq_len=%d dropped_parse=%d mean_input_len=%.2f mean_seq_len=%.2f mean_choices=%.2f mean_backtracks=%.2f mean_depth=%.2f",
                        int(kept),
                        int(target),
                        int(attempts),
                        int(len(with_backtrack)),
                        float(len(with_backtrack) / max(kept, 1)),
                        int(dropped_input_len),
                        int(dropped_seq_len),
                        int(dropped_parse),
                        float(np.mean(input_lengths)) if input_lengths else 0.0,
                        float(np.mean(seq_lengths)) if seq_lengths else 0.0,
                        float(np.mean(choice_counts)) if choice_counts else 0.0,
                        float(np.mean(backtrack_counts)) if backtrack_counts else 0.0,
                        float(np.mean(max_depths)) if max_depths else 0.0,
                    )

                if (
                    len(with_backtrack) >= bt_target
                    and (len(with_backtrack) + len(without_backtrack)) >= target
                ):
                    break
            else:
                continue
            break

    final_records: List[Dict[str, Any]] = []
    final_records.extend(with_backtrack)
    remaining_plain = max(target - len(final_records), 0)
    final_records.extend(without_backtrack[:remaining_plain])
    final_records = final_records[:target]

    realized_bt = sum(int(rec["meta"]["n_backtracks"] > 0) for rec in final_records)
    logger.info(
        "summary total=%d attempts=%d backtracking_fraction=%.3f dropped_input_len=%d dropped_seq_len=%d dropped_parse=%d mean_input_len=%.2f mean_seq_len=%.2f mean_choices=%.2f mean_backtracks=%.2f mean_depth=%.2f",
        int(len(final_records)),
        int(attempts),
        float(realized_bt / max(len(final_records), 1)),
        int(dropped_input_len),
        int(dropped_seq_len),
        int(dropped_parse),
        float(np.mean(input_lengths)) if input_lengths else 0.0,
        float(np.mean(seq_lengths)) if seq_lengths else 0.0,
        float(np.mean(choice_counts)) if choice_counts else 0.0,
        float(np.mean(backtrack_counts)) if backtrack_counts else 0.0,
        float(np.mean(max_depths)) if max_depths else 0.0,
    )
    return final_records


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate parsing search traces")
    parser.add_argument("--num-traces", "--num_traces", type=int, default=5000)
    parser.add_argument("--max-depth", "--max_depth", type=int, default=4)
    parser.add_argument("--max-input-len", "--max_input_len", type=int, default=30)
    parser.add_argument("--max-seq-len", "--max_seq_len", type=int, default=2048)
    parser.add_argument("--output-dir", "--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--p-call", "--p_call", type=float, default=0.3)
    parser.add_argument("--p-index", "--p_index", type=float, default=0.2)
    parser.add_argument("--p-tuple", "--p_tuple", type=float, default=0.35)
    parser.add_argument("--p-neg", "--p_neg", type=float, default=0.1)
    parser.add_argument("--min-backtrack-fraction", type=float, default=0.3)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if int(args.num_traces) <= 0:
        raise ValueError("num-traces must be > 0")
    if int(args.max_depth) < 0:
        raise ValueError("max-depth must be >= 0")
    if int(args.max_input_len) <= 0:
        raise ValueError("max-input-len must be > 0")
    if int(args.max_seq_len) <= 0:
        raise ValueError("max-seq-len must be > 0")
    if not (0.0 <= float(args.min_backtrack_fraction) <= 1.0):
        raise ValueError("min-backtrack-fraction must be in [0, 1]")

    records = _generate_dataset(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.pkl"
    metadata_path = output_dir / "run_metadata.json"

    with traces_path.open("wb") as handle:
        pickle.dump(records, handle)

    run_metadata = {
        "num_records": int(len(records)),
        "max_depth": int(args.max_depth),
        "max_input_len": int(args.max_input_len),
        "max_seq_len": int(args.max_seq_len),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "p_call": float(args.p_call),
        "p_index": float(args.p_index),
        "p_tuple": float(args.p_tuple),
        "p_neg": float(args.p_neg),
        "min_backtrack_fraction": float(args.min_backtrack_fraction),
        "realized_backtrack_fraction": float(
            sum(int(rec["meta"]["n_backtracks"] > 0) for rec in records)
            / max(len(records), 1)
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    logger.info(
        "saved traces path=%s count=%d metadata_path=%s",
        str(traces_path),
        int(len(records)),
        str(metadata_path),
    )


if __name__ == "__main__":
    main()
