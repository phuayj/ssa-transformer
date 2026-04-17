#!/usr/bin/env python3
"""Convert enriched SAT traces into state-only training records."""

from __future__ import annotations

import argparse
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.interleaved_tokenizer import SATInterleavedTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _validate_trace(trace: Dict[str, Any], idx: int) -> None:
    required = ("sequence", "loss_mask", "block_ids", "label")
    for key in required:
        if key not in trace:
            raise KeyError(f"trace {idx} missing key: {key}")
    seq = trace["sequence"]
    loss = trace["loss_mask"]
    blocks = trace["block_ids"]
    if (
        not isinstance(seq, list)
        or not isinstance(loss, list)
        or not isinstance(blocks, list)
    ):
        raise TypeError(f"trace {idx} sequence/loss_mask/block_ids must be lists")
    if len(seq) != len(loss) or len(seq) != len(blocks):
        raise ValueError(
            f"trace {idx} malformed lengths: sequence={len(seq)} loss_mask={len(loss)} block_ids={len(blocks)}"
        )


def build_state_only_records(
    traces: List[Dict[str, Any]],
    max_records: Optional[int],
) -> List[Dict[str, Any]]:
    tokenizer = SATInterleavedTokenizer()
    out: List[Dict[str, Any]] = []

    for trace_idx, trace in enumerate(traces):
        _validate_trace(trace, trace_idx)

        sequence = [int(tok) for tok in trace["sequence"]]
        loss_mask = [bool(x) for x in trace["loss_mask"]]
        block_ids = [int(x) for x in trace["block_ids"]]

        prefix_tokens = [tok for tok, bid in zip(sequence, block_ids) if int(bid) == 0]
        if any(int(tok) == int(tokenizer.STATE) for tok in prefix_tokens):
            logger.warning(
                "trace %d has STATE token in block_id==0 prefix", int(trace_idx)
            )

        block_values = sorted({int(bid) for bid in block_ids if int(bid) > 0})
        for source_block_id in block_values:
            block_tokens = [
                int(tok)
                for tok, bid in zip(sequence, block_ids)
                if int(bid) == int(source_block_id)
            ]
            block_loss = [
                bool(m)
                for m, bid in zip(loss_mask, block_ids)
                if int(bid) == int(source_block_id)
            ]

            if not any(block_loss):
                continue

            source_meta = trace.get("meta")
            meta: Dict[str, Any] = {}
            if isinstance(source_meta, dict):
                meta.update(source_meta)
            elif source_meta is not None:
                meta["source_meta"] = source_meta

            meta["source_trace_index"] = int(trace_idx)
            meta["source_block_id"] = int(source_block_id)
            meta["history_mode"] = "state_only"

            record: Dict[str, Any] = {
                "sequence": list(prefix_tokens) + list(block_tokens),
                "loss_mask": [False] * len(prefix_tokens) + list(block_loss),
                "block_ids": [0] * len(prefix_tokens) + [1] * len(block_tokens),
                "label": str(trace["label"]),
                "meta": meta,
            }
            out.append(record)

            if max_records is not None and len(out) >= int(max_records):
                return out

    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SAT state-only dataset from enriched traces"
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_records", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"input pickle not found: {input_path}")

    with input_path.open("rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, list):
        raise TypeError(f"expected list of traces, got {type(raw)}")

    traces: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"trace {i} is not a dict: {type(item)}")
        traces.append(item)

    output_records = build_state_only_records(
        traces=traces,
        max_records=None if args.max_records is None else int(args.max_records),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(output_records, f)

    lengths = [len(rec["sequence"]) for rec in output_records]
    supervised_counts = [
        int(sum(bool(x) for x in rec["loss_mask"])) for rec in output_records
    ]
    supervised_dist = Counter(supervised_counts)

    logger.info("input_traces=%d", int(len(traces)))
    logger.info("output_records=%d", int(len(output_records)))
    logger.info(
        "sequence_len mean=%.2f max=%d",
        float(sum(lengths) / len(lengths)) if lengths else 0.0,
        int(max(lengths)) if lengths else 0,
    )
    logger.info(
        "supervised_tokens_per_block count_3=%d count_4=%d other=%s",
        int(supervised_dist.get(3, 0)),
        int(supervised_dist.get(4, 0)),
        {
            int(k): int(v)
            for k, v in sorted(supervised_dist.items())
            if int(k) not in (3, 4)
        },
    )
    logger.info("wrote state-only dataset to %s", str(output_path))


if __name__ == "__main__":
    main()
