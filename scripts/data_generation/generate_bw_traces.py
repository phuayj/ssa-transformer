"""Generate Blocks World DFS traces for training."""

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from blocks_world.env import BlocksWorldEnv, dfs_solve
from blocks_world.tokenizer import BlocksWorldTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_instances", type=int, default=5000)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--num_stacks", type=int, default=3)
    parser.add_argument("--min_scramble", type=int, default=3)
    parser.add_argument("--max_scramble", type=int, default=10)
    parser.add_argument("--max_solve_steps", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = BlocksWorldEnv(num_stacks=args.num_stacks, num_blocks=args.num_blocks)
    tok = BlocksWorldTokenizer(num_blocks=args.num_blocks, num_stacks=args.num_stacks)

    records = []
    solved_count = 0
    bt_total = 0

    for i in range(args.num_instances):
        start, goal = env.generate_instance(rng, args.min_scramble, args.max_scramble)
        result = dfs_solve(start, goal, max_steps=args.max_solve_steps)

        if not result["solved"]:
            continue  # skip unsolved instances for training

        seq, loss_mask = tok.trace_to_sequence(goal, result["trace"])

        if len(seq) > args.max_seq_len:
            continue  # skip too-long sequences

        records.append(
            {
                "sequence": seq,
                "loss_mask": loss_mask,
                "num_blocks": args.num_blocks,
                "num_stacks": args.num_stacks,
                "steps": result["steps"],
                "backtracks": result["backtracks"],
            }
        )
        solved_count += 1
        bt_total += result["backtracks"]

        if (i + 1) % 500 == 0:
            logger.info(
                f"processed={i + 1}/{args.num_instances} solved={solved_count} "
                f"mean_bt={bt_total / max(solved_count, 1):.1f} "
                f"mean_seq_len={sum(len(r['sequence']) for r in records) / max(len(records), 1):.0f}"
            )

    logger.info(
        f"Final: {solved_count} solved traces, mean_bt={bt_total / max(solved_count, 1):.1f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(records, f)
    logger.info(f"Saved {len(records)} traces to {output_path}")


if __name__ == "__main__":
    main()
