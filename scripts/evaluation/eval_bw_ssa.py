#!/usr/bin/env python3
"""Closed-loop eval: SSA vs causal on Blocks World search."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blocks_world.env import BlocksWorldEnv, BlocksWorldState, dfs_solve
from blocks_world.tokenizer import BlocksWorldTokenizer
from universal.slot_decoder import SlotCDCLDecoder


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

Move = Tuple[int, int, int]
CanonState = Tuple[Tuple[int, ...], ...]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _append_tokens(
    sequence: List[int],
    block_ids: List[int],
    tokens: Iterable[int],
    max_seq_len: int,
    block_id: int,
) -> bool:
    chunk = [int(x) for x in tokens]
    if len(sequence) + len(chunk) > int(max_seq_len):
        return False
    sequence.extend(chunk)
    block_ids.extend([int(block_id)] * len(chunk))
    return True


def _model_last_logits(
    model: torch.nn.Module,
    sequence: Sequence[int],
    block_ids: Sequence[int],
    use_block_ids: bool,
    device: torch.device,
) -> torch.Tensor:
    input_tensor = torch.tensor([list(sequence)], dtype=torch.long, device=device)
    with torch.no_grad():
        if bool(use_block_ids):
            block_tensor = torch.tensor(
                [list(block_ids)], dtype=torch.long, device=device
            )
            lm_logits, _ = model(input_tensor, block_ids=block_tensor)
        else:
            lm_logits, _ = model(input_tensor)
    return lm_logits[0, -1, :]


def _masked_argmax(logits: torch.Tensor, valid_tokens: Sequence[int]) -> int:
    masked = torch.full_like(logits, float("-inf"))
    for tok in valid_tokens:
        masked[int(tok)] = logits[int(tok)]
    return int(torch.argmax(masked).item())


def _decode_block_id(token_id: int, tok: BlocksWorldTokenizer) -> int:
    return int(token_id - int(tok.BLOCK_OFFSET))


def _decode_stack_id(token_id: int, tok: BlocksWorldTokenizer) -> int:
    return int(token_id - int(tok.STACK_OFFSET))


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    max_seq_len_fallback: int,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})
    vocab_size = int(
        config.get("vocab_size", state_dict["token_embedding.weight"].shape[0])
    )
    d_model = int(config.get("d_model", 128))
    n_layers = int(config.get("n_layers", 4))
    n_heads = int(config.get("n_heads", 4))
    n_slots = int(config.get("n_slots", 16))
    max_seq_len_model = int(config.get("max_seq_len", int(max_seq_len_fallback)))
    dropout = float(config.get("dropout", 0.1))
    attention_mode = str(config.get("attention_mode", "causal"))

    if attention_mode == "ssa":
        from universal.ssa_decoder import SSASlotDecoder

        model: torch.nn.Module = SSASlotDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
        )
        model_kind = "SSASlotDecoder"
    else:
        model = SlotCDCLDecoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len_model),
            n_slots=int(n_slots),
            dropout=float(dropout),
        )
        model_kind = "SlotCDCLDecoder"

    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state_dict.items():
        if key in model_state and value.shape != model_state[key].shape:
            skipped.append(key)
            continue
        filtered[key] = value
    if skipped:
        logger.warning("Skipped %d keys: %s", len(skipped), skipped)

    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()
    return model, {
        "kind": model_kind,
        "max_seq_len_model": int(max_seq_len_model),
        "attention_mode": attention_mode,
        "config": config,
    }


def _generate_instances(
    *,
    num_instances: int,
    num_blocks: int,
    num_stacks: int,
    min_scramble: int,
    max_scramble: int,
    seed: int,
) -> List[Tuple[BlocksWorldState, BlocksWorldState]]:
    rng = random.Random(int(seed))
    env = BlocksWorldEnv(num_stacks=int(num_stacks), num_blocks=int(num_blocks))
    out: List[Tuple[BlocksWorldState, BlocksWorldState]] = []
    for _ in range(int(num_instances)):
        start, goal = env.generate_instance(rng, int(min_scramble), int(max_scramble))
        out.append((start, goal))
    return out


def solve_instance(
    *,
    model: torch.nn.Module,
    tokenizer: BlocksWorldTokenizer,
    start: BlocksWorldState,
    goal: BlocksWorldState,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    use_block_ids: bool,
    log_sample: bool,
) -> Dict[str, Any]:
    sequence: List[int] = tokenizer.build_prefix(goal)
    block_ids: List[int] = [0] * len(sequence)
    current_block = 0

    current_state = start
    move_stack: List[Tuple[Move, BlocksWorldState]] = []
    tried_actions: Dict[CanonState, List[Move]] = {}
    state_visit_counts: Dict[CanonState, int] = {}

    stats: Dict[str, Any] = {
        "solved": False,
        "steps": 0,
        "backtracks": 0,
        "total_decisions": 0,
        "revisited_state_decisions": 0,
        "repeat_errors": 0,
        "revisit_states": 0,
        "illegal_moves": 0,
        "termination_reason": "max_steps",
        "max_block_id": 0,
    }

    for step in range(int(max_steps)):
        stats["steps"] = int(step + 1)

        if current_state.matches(goal):
            stats["solved"] = True
            stats["termination_reason"] = "solved"
            break

        canonical = current_state.canonical()
        seen = int(state_visit_counts.get(canonical, 0))
        state_visit_counts[canonical] = int(seen + 1)
        if seen > 0:
            stats["revisit_states"] += 1
            stats["revisited_state_decisions"] += 1

        prior = tried_actions.get(canonical, [])
        state_preceded_by_tried = False
        if prior:
            current_block += 1
            stats["max_block_id"] = max(int(stats["max_block_id"]), current_block)
            tried_tokens = tokenizer.encode_tried(list(prior))
            if not _append_tokens(
                sequence,
                block_ids,
                tried_tokens,
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break
            state_preceded_by_tried = True

        if not state_preceded_by_tried:
            current_block += 1
            stats["max_block_id"] = max(int(stats["max_block_id"]), current_block)

        state_tokens = tokenizer.encode_state(current_state)
        if not _append_tokens(
            sequence,
            block_ids,
            state_tokens,
            int(max_seq_len),
            int(current_block),
        ):
            stats["termination_reason"] = "max_seq_len"
            break

        action_logits = _model_last_logits(
            model=model,
            sequence=sequence,
            block_ids=block_ids,
            use_block_ids=bool(use_block_ids),
            device=device,
        )
        action_token = _masked_argmax(
            action_logits,
            [tokenizer.MOVE, tokenizer.BACKTRACK, tokenizer.SOLVED],
        )
        stats["total_decisions"] += 1

        if action_token == int(tokenizer.MOVE):
            legal_moves = [tuple(m) for m in current_state.legal_moves()]
            if not legal_moves:
                stats["termination_reason"] = "no_legal_moves"
                break

            if not _append_tokens(
                sequence,
                block_ids,
                [tokenizer.MOVE],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            valid_blocks = sorted({int(mv[0]) for mv in legal_moves})
            block_logits = _model_last_logits(
                model=model,
                sequence=sequence,
                block_ids=block_ids,
                use_block_ids=bool(use_block_ids),
                device=device,
            )
            block_token = _masked_argmax(
                block_logits,
                [tokenizer.block_token(int(block)) for block in valid_blocks],
            )
            chosen_block = _decode_block_id(int(block_token), tokenizer)
            if chosen_block not in valid_blocks:
                chosen_block = int(valid_blocks[0])
                block_token = int(tokenizer.block_token(chosen_block))

            if not _append_tokens(
                sequence,
                block_ids,
                [int(block_token)],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            valid_from = sorted(
                {int(mv[1]) for mv in legal_moves if int(mv[0]) == int(chosen_block)}
            )
            from_logits = _model_last_logits(
                model=model,
                sequence=sequence,
                block_ids=block_ids,
                use_block_ids=bool(use_block_ids),
                device=device,
            )
            from_token = _masked_argmax(
                from_logits,
                [tokenizer.stack_token(int(st)) for st in valid_from],
            )
            chosen_from = _decode_stack_id(int(from_token), tokenizer)
            if chosen_from not in valid_from:
                chosen_from = int(valid_from[0])
                from_token = int(tokenizer.stack_token(chosen_from))

            if not _append_tokens(
                sequence,
                block_ids,
                [int(from_token)],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            valid_to = sorted(
                {
                    int(mv[2])
                    for mv in legal_moves
                    if int(mv[0]) == int(chosen_block)
                    and int(mv[1]) == int(chosen_from)
                }
            )
            to_logits = _model_last_logits(
                model=model,
                sequence=sequence,
                block_ids=block_ids,
                use_block_ids=bool(use_block_ids),
                device=device,
            )
            to_token = _masked_argmax(
                to_logits,
                [tokenizer.stack_token(int(st)) for st in valid_to],
            )
            chosen_to = _decode_stack_id(int(to_token), tokenizer)
            if chosen_to not in valid_to:
                chosen_to = int(valid_to[0])
                to_token = int(tokenizer.stack_token(chosen_to))

            if not _append_tokens(
                sequence,
                block_ids,
                [int(to_token)],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            chosen_move: Move = (int(chosen_block), int(chosen_from), int(chosen_to))
            prior_choices = tried_actions.setdefault(canonical, [])
            if chosen_move in prior_choices:
                stats["repeat_errors"] += 1
            prior_choices.append(chosen_move)

            if chosen_move not in legal_moves:
                stats["illegal_moves"] += 1
                stats["termination_reason"] = "illegal_move"
                break

            previous_state = current_state
            current_state = current_state.apply_move(*chosen_move)
            move_stack.append((chosen_move, previous_state))

            if log_sample and step < 8:
                logger.info(
                    "sample_move step=%d move=%s depth=%d blocks=%d revisits=%d repeats=%d",
                    int(step),
                    str(chosen_move),
                    int(len(move_stack)),
                    int(stats["max_block_id"]),
                    int(stats["revisited_state_decisions"]),
                    int(stats["repeat_errors"]),
                )

        elif action_token == int(tokenizer.BACKTRACK):
            if not _append_tokens(
                sequence,
                block_ids,
                [tokenizer.BACKTRACK],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            stats["backtracks"] += 1
            if not move_stack:
                stats["termination_reason"] = "backtrack_on_empty_stack"
                break

            _last_move, parent_state = move_stack.pop()
            current_state = parent_state

            if log_sample and step < 8:
                logger.info(
                    "sample_backtrack step=%d depth=%d blocks=%d",
                    int(step),
                    int(len(move_stack)),
                    int(stats["max_block_id"]),
                )

        elif action_token == int(tokenizer.SOLVED):
            if not _append_tokens(
                sequence,
                block_ids,
                [tokenizer.SOLVED],
                int(max_seq_len),
                int(current_block),
            ):
                stats["termination_reason"] = "max_seq_len"
                break

            if current_state.matches(goal):
                stats["solved"] = True
                stats["termination_reason"] = "solved"
            else:
                stats["termination_reason"] = "premature_solved"
            break

        else:
            stats["termination_reason"] = "invalid_action_token"
            break

    stats["max_block_id"] = max(int(stats["max_block_id"]), int(current_block))
    stats["repeat_error_rate"] = float(
        _safe_div(stats["repeat_errors"], stats["revisited_state_decisions"])
    )
    stats["revisit_fraction"] = float(
        _safe_div(stats["revisited_state_decisions"], stats["total_decisions"])
    )
    return stats


def _aggregate(mode: str, per_instance: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = int(len(per_instance))
    solved = int(sum(int(item["solved"]) for item in per_instance))
    total_decisions = int(sum(int(item["total_decisions"]) for item in per_instance))
    revisited = int(
        sum(int(item["revisited_state_decisions"]) for item in per_instance)
    )
    repeats = int(sum(int(item["repeat_errors"]) for item in per_instance))
    return {
        "mode": str(mode),
        "aggregate": {
            "solve_rate": float(_safe_div(solved, total)),
            "mean_backtracks": float(
                np.mean([float(item["backtracks"]) for item in per_instance])
                if total
                else 0.0
            ),
            "mean_max_block_id": float(
                np.mean([float(item["max_block_id"]) for item in per_instance])
                if total
                else 0.0
            ),
            "total_decisions": int(total_decisions),
            "revisited_state_decisions": int(revisited),
            "repeat_errors": int(repeats),
            "repeat_error_rate": float(_safe_div(repeats, revisited)),
            "revisit_fraction": float(_safe_div(revisited, total_decisions)),
        },
        "per_instance": list(per_instance),
    }


def _bucket_name(backtracks: int) -> str:
    bt = int(backtracks)
    if bt == 0:
        return "0"
    if 1 <= bt <= 5:
        return "1-5"
    if 6 <= bt <= 10:
        return "6-10"
    if 11 <= bt <= 20:
        return "11-20"
    return "21+"


def _bucket_table(
    ssa_rows: Sequence[Dict[str, Any]],
    causal_rows: Sequence[Dict[str, Any]],
) -> List[Tuple[str, int, int, int, int]]:
    order = ["0", "1-5", "6-10", "11-20", "21+"]
    ssa_map = {k: {"total": 0, "solved": 0} for k in order}
    causal_map = {k: {"total": 0, "solved": 0} for k in order}

    for row in ssa_rows:
        b = _bucket_name(int(row["dfs_backtracks"]))
        ssa_map[b]["total"] += 1
        ssa_map[b]["solved"] += int(row["solved"])

    for row in causal_rows:
        b = _bucket_name(int(row["dfs_backtracks"]))
        causal_map[b]["total"] += 1
        causal_map[b]["solved"] += int(row["solved"])

    out: List[Tuple[str, int, int, int, int]] = []
    for key in order:
        out.append(
            (
                key,
                int(ssa_map[key]["solved"]),
                int(ssa_map[key]["total"]),
                int(causal_map[key]["solved"]),
                int(causal_map[key]["total"]),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop SSA vs causal evaluation on Blocks World"
    )
    parser.add_argument("--ssa_checkpoint", type=str, required=True)
    parser.add_argument("--causal_checkpoint", type=str, required=True)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--num_stacks", type=int, default=3)
    parser.add_argument("--min_scramble", type=int, default=3)
    parser.add_argument("--max_scramble", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    tokenizer = BlocksWorldTokenizer(
        num_blocks=int(args.num_blocks), num_stacks=int(args.num_stacks)
    )
    device = torch.device(str(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ssa_model, ssa_meta = _load_checkpoint(
        checkpoint_path=Path(args.ssa_checkpoint),
        device=device,
        max_seq_len_fallback=int(args.max_seq_len),
    )
    causal_model, causal_meta = _load_checkpoint(
        checkpoint_path=Path(args.causal_checkpoint),
        device=device,
        max_seq_len_fallback=int(args.max_seq_len),
    )

    shared_instances = _generate_instances(
        num_instances=int(args.num_instances),
        num_blocks=int(args.num_blocks),
        num_stacks=int(args.num_stacks),
        min_scramble=int(args.min_scramble),
        max_scramble=int(args.max_scramble),
        seed=int(args.seed),
    )
    logger.info(
        "generated instances=%d blocks=%d stacks=%d scramble=[%d,%d]",
        int(len(shared_instances)),
        int(args.num_blocks),
        int(args.num_stacks),
        int(args.min_scramble),
        int(args.max_scramble),
    )

    dfs_stats: List[Dict[str, int]] = []
    dfs_started = time.time()
    for idx, (start, goal) in enumerate(shared_instances):
        result = dfs_solve(start, goal, max_steps=int(args.max_steps))
        row = {
            "success": int(bool(result.get("solved", False))),
            "steps": int(result.get("steps", 0)),
            "backtracks": int(result.get("backtracks", 0)),
        }
        dfs_stats.append(row)
        if (idx + 1) % 25 == 0:
            logger.info(
                "dfs processed=%d/%d mean_bt=%.2f success_rate=%.3f",
                int(idx + 1),
                int(len(shared_instances)),
                float(np.mean([float(x["backtracks"]) for x in dfs_stats])),
                float(
                    _safe_div(sum(int(x["success"]) for x in dfs_stats), len(dfs_stats))
                ),
            )
    logger.info("dfs pre-pass elapsed_sec=%.2f", float(time.time() - dfs_started))

    runs = [
        ("ssa", ssa_model, True, ssa_meta, str(args.ssa_checkpoint)),
        ("causal", causal_model, False, causal_meta, str(args.causal_checkpoint)),
    ]

    summaries: Dict[str, Dict[str, Any]] = {}
    per_mode_instances: Dict[str, List[Dict[str, Any]]] = {}

    for mode_name, model, use_block_ids, meta, ckpt in runs:
        _set_seed(int(args.seed))
        started = time.time()
        if str(meta.get("attention_mode", "")).lower() == "ssa":
            max_len_eval = int(args.max_seq_len)
        else:
            max_len_eval = int(
                min(int(args.max_seq_len), int(meta["max_seq_len_model"]))
            )

        logger.info(
            "starting mode=%s use_block_ids=%s attention_mode=%s",
            str(mode_name),
            str(use_block_ids),
            str(meta.get("attention_mode", "unknown")),
        )

        per_instance: List[Dict[str, Any]] = []
        for idx, (start, goal) in enumerate(shared_instances):
            stats = solve_instance(
                model=model,
                tokenizer=tokenizer,
                start=start,
                goal=goal,
                max_steps=int(args.max_steps),
                max_seq_len=int(max_len_eval),
                device=device,
                use_block_ids=bool(use_block_ids),
                log_sample=bool(idx < 2),
            )
            stats["instance_index"] = int(idx)
            stats["dfs_backtracks"] = int(dfs_stats[idx]["backtracks"])
            stats["dfs_steps"] = int(dfs_stats[idx]["steps"])
            stats["dfs_success"] = bool(dfs_stats[idx]["success"])
            per_instance.append(stats)

            if (idx + 1) % 10 == 0:
                pd = int(sum(int(x["total_decisions"]) for x in per_instance))
                pr = int(sum(int(x["revisited_state_decisions"]) for x in per_instance))
                pe = int(sum(int(x["repeat_errors"]) for x in per_instance))
                mb = float(np.mean([float(x["max_block_id"]) for x in per_instance]))
                logger.info(
                    "mode=%s processed=%d/%d solve_rate=%.3f mean_bt=%.2f mean_blocks=%.2f revisit=%.3f repeat=%.3f",
                    str(mode_name),
                    int(idx + 1),
                    int(len(shared_instances)),
                    float(
                        _safe_div(
                            sum(int(x["solved"]) for x in per_instance),
                            len(per_instance),
                        )
                    ),
                    float(np.mean([float(x["backtracks"]) for x in per_instance])),
                    float(mb),
                    float(_safe_div(pr, pd)),
                    float(_safe_div(pe, pr)),
                )

        payload = _aggregate(str(mode_name), per_instance)
        payload["config"] = {
            "mode": str(mode_name),
            "seed": int(args.seed),
            "num_instances": int(args.num_instances),
            "num_blocks": int(args.num_blocks),
            "num_stacks": int(args.num_stacks),
            "min_scramble": int(args.min_scramble),
            "max_scramble": int(args.max_scramble),
            "max_steps": int(args.max_steps),
            "max_seq_len": int(max_len_eval),
            "device": str(args.device),
            "checkpoint": str(ckpt),
            "model_kind": str(meta["kind"]),
            "attention_mode": str(meta.get("attention_mode", "causal")),
            "use_block_ids": bool(use_block_ids),
            "elapsed_sec": float(time.time() - started),
        }
        out_path = output_dir / f"{mode_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        summaries[mode_name] = payload["aggregate"]
        per_mode_instances[mode_name] = per_instance
        logger.info("mode=%s wrote=%s", mode_name, out_path)

    print()
    print("Mode      Solve  MeanBT  RepeatRate  RevisitFrac  MeanMaxBlock")
    for mode_name, *_rest in runs:
        agg = summaries[mode_name]
        print(
            f"{mode_name:<8} {float(agg['solve_rate']):>5.2f} "
            f"{float(agg['mean_backtracks']):>7.2f} "
            f"{float(agg['repeat_error_rate']):>10.2f} "
            f"{float(agg['revisit_fraction']):>11.2f} "
            f"{float(agg['mean_max_block_id']):>12.2f}"
        )

    print()
    print("DFS BT range      SSA Solve   Causal Solve")
    for bucket, ssa_solved, ssa_total, causal_solved, causal_total in _bucket_table(
        per_mode_instances["ssa"], per_mode_instances["causal"]
    ):
        print(
            f"{bucket:<16} {ssa_solved:>3d}/{ssa_total:<3d}"
            f"       {causal_solved:>3d}/{causal_total:<3d}"
        )


if __name__ == "__main__":
    main()
