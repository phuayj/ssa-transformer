#!/usr/bin/env python3
"""Post-process SAT enriched traces by masking STATE-domain annotations."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TypedDict, cast


class TraceRecord(TypedDict):
    sequence: List[int]
    loss_mask: List[bool]
    block_ids: List[int]
    label: str
    meta: Dict[str, Any]


# Keep in sync with src/sat/interleaved_tokenizer.py
TOK_STATE = 6
TOK_SEP = 3
TOK_TRUE = 15
TOK_FALSE = 16
TOK_UNASSIGNED = 17
TOK_MASKED_DOMAIN = 21
TOK_VAR_OFFSET = 630


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _mask_state_domains(
    sequence: Sequence[int],
) -> Tuple[List[int], int, int, int, int]:
    """Mask STATE section domain tokens in (var, domain) pairs only."""
    state_token = int(TOK_STATE)
    sep_token = int(TOK_SEP)
    var_offset = int(TOK_VAR_OFFSET)
    true_token = int(TOK_TRUE)
    false_token = int(TOK_FALSE)
    unassigned_token = int(TOK_UNASSIGNED)
    masked_token = int(TOK_MASKED_DOMAIN)
    domain_tokens = {true_token, false_token, unassigned_token}

    out = [int(tok) for tok in sequence]
    replaced = 0
    t_count = 0
    f_count = 0
    u_count = 0

    i = 0
    while i < len(out):
        if int(out[i]) != state_token:
            i += 1
            continue

        i += 1
        while i < len(out) and int(out[i]) != sep_token:
            if (
                i + 1 < len(out)
                and int(out[i]) >= var_offset
                and int(out[i + 1]) in domain_tokens
            ):
                old_domain = int(out[i + 1])
                if old_domain == true_token:
                    t_count += 1
                elif old_domain == false_token:
                    f_count += 1
                elif old_domain == unassigned_token:
                    u_count += 1

                out[i + 1] = masked_token
                replaced += 1
                i += 2
                continue
            i += 1

        if i < len(out) and int(out[i]) == sep_token:
            i += 1

    return out, int(replaced), int(t_count), int(f_count), int(u_count)


def _verify_output(
    input_traces: Sequence[TraceRecord],
    output_traces: Sequence[TraceRecord],
) -> None:
    if len(input_traces) != len(output_traces):
        raise ValueError(
            f"trace count mismatch: in={len(input_traces)} out={len(output_traces)}"
        )

    state_token = int(TOK_STATE)
    sep_token = int(TOK_SEP)
    domain_tokens = {
        int(TOK_TRUE),
        int(TOK_FALSE),
        int(TOK_UNASSIGNED),
    }

    for idx, (src, dst) in enumerate(zip(input_traces, output_traces)):
        src_seq = list(src["sequence"])
        dst_seq = list(dst["sequence"])
        if len(src_seq) != len(dst_seq):
            raise ValueError(
                f"trace {idx}: sequence length changed in post-process "
                f"({len(src_seq)} -> {len(dst_seq)})"
            )
        if list(src["loss_mask"]) != list(dst["loss_mask"]):
            raise ValueError(f"trace {idx}: loss_mask changed")
        if list(src["block_ids"]) != list(dst["block_ids"]):
            raise ValueError(f"trace {idx}: block_ids changed")

        i = 0
        while i < len(dst_seq):
            tok = int(dst_seq[i])
            if tok != state_token:
                i += 1
                continue
            i += 1
            while i < len(dst_seq) and int(dst_seq[i]) != sep_token:
                if int(dst_seq[i]) in domain_tokens:
                    raise ValueError(
                        f"trace {idx}: found raw T/F/U token {int(dst_seq[i])} inside STATE section"
                    )
                i += 1
            if i < len(dst_seq) and int(dst_seq[i]) == sep_token:
                i += 1


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate stripped SAT traces by masking STATE domains"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="experiments/sat-deep-enriched-traces/traces.pkl",
        help="Input pickle with enriched SAT traces",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/sat-stripped-traces/traces.pkl",
        help="Output pickle path for stripped traces",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"input traces not found: {input_path}")

    with input_path.open("rb") as f:
        traces_raw = pickle.load(f)

    if not isinstance(traces_raw, list):
        raise TypeError(f"expected list of trace dicts, got {type(traces_raw)}")

    traces: List[TraceRecord] = []
    for idx, trace in enumerate(traces_raw):
        if not isinstance(trace, dict):
            raise TypeError(f"trace {idx} is not a dict: {type(trace)}")
        for key in ("sequence", "loss_mask", "block_ids", "label", "meta"):
            if key not in trace:
                raise KeyError(f"trace {idx} missing key: {key}")
        traces.append(cast(TraceRecord, trace))

    output_traces: List[TraceRecord] = []
    total_replaced = 0
    total_t = 0
    total_f = 0
    total_u = 0

    for idx, trace in enumerate(traces):
        sequence = trace["sequence"]
        loss_mask = trace["loss_mask"]
        block_ids = trace["block_ids"]
        if len(sequence) != len(loss_mask) or len(sequence) != len(block_ids):
            raise ValueError(
                f"trace {idx} malformed lengths: "
                f"sequence={len(sequence)} loss_mask={len(loss_mask)} block_ids={len(block_ids)}"
            )

        stripped_seq, replaced, t_count, f_count, u_count = _mask_state_domains(
            sequence
        )
        total_replaced += int(replaced)
        total_t += int(t_count)
        total_f += int(f_count)
        total_u += int(u_count)

        out_trace: TraceRecord = dict(trace)
        out_trace["sequence"] = stripped_seq
        output_traces.append(out_trace)

    _verify_output(traces, output_traces)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(output_traces, f)

    logger.info(
        "stripped_traces_done traces=%d replaced_domains=%d t=%d f=%d u=%d input=%s output=%s",
        int(len(traces)),
        int(total_replaced),
        int(total_t),
        int(total_f),
        int(total_u),
        str(input_path),
        str(output_path),
    )


if __name__ == "__main__":
    main()
