#!/usr/bin/env python3
"""Generate candidate solutions for MATH benchmark and label by answer correctness.

For each MATH problem:
1. Format as instruction prompt
2. Sample K solutions with temperature
3. Extract final boxed answer from each
4. Label correct (1) or incorrect (0) by matching ground truth

Output: JSON lines file with columns:
  - problem: str
  - category: str
  - ground_truth: str
  - candidates: List[{solution: str, answer: str, correct: int}]
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \boxed{...} in MATH format."""
    # Find the last \boxed{...}
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Fallback: try #### pattern (GSM8K style)
    match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None


def _extract_braced(text: str, start_idx: int) -> Optional[Tuple[str, int]]:
    """Extract { ... } content starting at start_idx, returning (content, next_idx)."""
    if start_idx >= len(text) or text[start_idx] != "{":
        return None
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx + 1 : i], i + 1
    return None


def _replace_frac(text: str) -> str:
    """Replace LaTeX \frac{a}{b} with (a)/(b) for numeric eval."""
    if "\\frac" not in text:
        return text

    out = []
    i = 0
    while i < len(text):
        if text.startswith("\\frac", i):
            i += len("\\frac")
            while i < len(text) and text[i].isspace():
                i += 1
            numerator = _extract_braced(text, i)
            if not numerator:
                out.append("/")
                continue
            num_text, i = numerator
            denominator = _extract_braced(text, i)
            if not denominator:
                out.append(f"({num_text})/")
                continue
            den_text, i = denominator
            num_text = _replace_frac(num_text)
            den_text = _replace_frac(den_text)
            out.append(f"({num_text})/({den_text})")
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def normalize_math_answer(answer: str) -> str:
    """Normalize a MATH answer for comparison."""
    if answer is None:
        return ""

    s = answer.strip()
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace(" ", "").replace("\n", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)

    # Try numeric comparison
    expr = _replace_frac(s)
    expr = expr.replace("^", "**")
    expr = expr.replace("{", "(").replace("}", ")")
    try:
        val = float(eval(expr, {"__builtins__": {}}, {}))
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}"
    except Exception:
        return s


def answers_match(pred: str, truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if not pred or not truth:
        return False
    np = normalize_math_answer(pred)
    nt = normalize_math_answer(truth)
    if np == nt:
        return True
    try:
        return abs(float(np) - float(nt)) < 1e-4
    except Exception:
        return False


def format_prompt(problem: str) -> str:
    """Format MATH problem as few-shot prompt for a base model."""
    example_1_q = "What is the value of $\\frac{3}{5} + \\frac{2}{5}$?"
    example_1_a = (
        "We add the fractions with common denominators: "
        "$\\frac{3}{5} + \\frac{2}{5} = \\frac{3+2}{5} = \\frac{5}{5} = 1$.\n\n"
        "The answer is $\\boxed{1}$."
    )
    example_2_q = "Solve for $x$: $2x + 5 = 13$."
    example_2_a = (
        "Subtract 5 from both sides: $2x = 8$. Divide by 2: $x = 4$.\n\n"
        "The answer is $\\boxed{4}$."
    )
    return (
        "Solve each math problem step by step. Put your final answer in \\boxed{}.\n\n"
        f"Problem: {example_1_q}\n\n"
        f"Solution: {example_1_a}\n\n"
        f"Problem: {example_2_q}\n\n"
        f"Solution: {example_2_a}\n\n"
        f"Problem: {problem}\n\n"
        "Solution:"
    )


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    k: int = 16,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: str = "cuda:0",
    gen_batch_size: int = 2,
) -> List[str]:
    """Generate K candidate solutions."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    candidates = []
    batch_size = min(k, gen_batch_size)

    for i in range(0, k, batch_size):
        current_batch = min(batch_size, k - i)
        batch_input = input_ids.expand(current_batch, -1)
        batch_mask = attention_mask.expand(current_batch, -1)

        with torch.no_grad():
            outputs = model.generate(
                batch_input,
                attention_mask=batch_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j in range(current_batch):
            gen_tokens = outputs[j][input_ids.shape[1] :]
            solution = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            candidates.append(solution)

    return candidates


def truncate_at_continuation(solution: str) -> str:
    """Truncate generated text at the first sign of a new problem.

    Base models continue generating after answering, producing
    hallucinated follow-up problems. We stop at the first continuation marker.
    """
    import re
    # Common continuation patterns from base models
    markers = [
        "\n\nProblem:",
        "\nProblem:",
        "\n\nQuestion:",
        "\nQuestion:",
        "\n\nSolve each",
        "\n\nSolve the following",
    ]
    earliest = len(solution)
    for marker in markers:
        idx = solution.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return solution[:earliest].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Ministral-3-14B-Base-2512"
    )
    parser.add_argument("--k", type=int, default=16, help="Candidates per problem")
    parser.add_argument(
        "--max_problems", type=int, default=None, help="Limit number of problems"
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Generation batch size (higher = faster but more VRAM)",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Comma-separated categories or 'all'",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_index", type=int, default=0, help="Start index into dataset (for parallel generation)")
    parser.add_argument("--end_index", type=int, default=None, help="End index (exclusive) into dataset")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    logger.info("Generation batch size: %d", args.batch_size)

    all_categories = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    if args.categories == "all":
        categories = all_categories
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    unknown = [c for c in categories if c not in all_categories]
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}. Supported: {all_categories}")

    logger.info("Loading MATH %s split, categories: %s", args.split, categories)
    all_problems = []
    for cat in categories:
        ds = load_dataset("EleutherAI/hendrycks_math", cat, split=args.split)
        logger.info("Loaded %d problems from %s", len(ds), cat)
        for item in ds:
            all_problems.append(
                {
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "category": cat,
                    "ground_truth": extract_boxed_answer(item["solution"]),
                }
            )

    # Apply range selection for parallel generation
    start = args.start_index
    end = args.end_index if args.end_index is not None else len(all_problems)
    end = min(end, len(all_problems))
    all_problems = all_problems[start:end]
    if args.max_problems:
        all_problems = all_problems[:args.max_problems]
    logger.info("Total problems: %d", len(all_problems))

    logger.info("Loading %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    logger.info("Detected model type: %s", getattr(config, "model_type", "unknown"))
    if getattr(config, "model_type", "") == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model = Mistral3ForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        )
    model.eval()
    logger.info("Model loaded on %s", args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_correct = 0
    total_candidates = 0
    per_category = {}

    with open(output_path, "w") as f:
        for idx, prob in enumerate(all_problems):
            prompt = format_prompt(prob["problem"])
            candidates = generate_candidates(
                model,
                tokenizer,
                prompt,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
                gen_batch_size=args.batch_size,
            )

            gt = prob["ground_truth"] or ""
            labeled = []
            for sol in candidates:
                sol = truncate_at_continuation(sol)
                pred = extract_boxed_answer(sol)
                correct = int(answers_match(pred, gt)) if pred and gt else 0
                labeled.append(
                    {
                        "solution": sol,
                        "answer": normalize_math_answer(pred) if pred else "",
                        "correct": correct,
                    }
                )
                total_correct += correct
                total_candidates += 1

            cat = prob["category"]
            if cat not in per_category:
                per_category[cat] = {"correct": 0, "total": 0}
            cat_correct = sum(c["correct"] for c in labeled)
            per_category[cat]["correct"] += cat_correct
            per_category[cat]["total"] += len(labeled)

            record = {
                "idx": idx,
                "problem": prob["problem"],
                "category": cat,
                "ground_truth": gt,
                "candidates": labeled,
            }
            f.write(json.dumps(record) + "\n")

            if (idx + 1) % 10 == 0:
                acc = total_correct / max(total_candidates, 1)
                sample = labeled[0] if labeled else {"answer": "", "correct": 0}
                logger.info(
                    "processed %d/%d accuracy=%.3f sample_pred=%s gt=%s sample_correct=%d",
                    idx + 1,
                    len(all_problems),
                    acc,
                    sample["answer"],
                    gt,
                    sample["correct"],
                )

    overall_acc = total_correct / max(total_candidates, 1)
    logger.info(
        "DONE: %d problems, %d candidates, accuracy=%.3f, saved to %s",
        len(all_problems),
        total_candidates,
        overall_acc,
        str(output_path),
    )
    for cat, data in per_category.items():
        logger.info(
            "  %s: %.3f (%d/%d)",
            cat,
            data["correct"] / max(data["total"], 1),
            data["correct"],
            data["total"],
        )


if __name__ == "__main__":
    main()
