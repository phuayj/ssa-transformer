#!/usr/bin/env python3
"""Generate cross-mask LaTeX table for appendix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


FALLBACK_MATCHED_PCT: dict[tuple[str, str], float] = {
    ("sat", "blanket"): 61.5,
    ("sat", "selective"): 59.0,
    ("gc", "blanket"): 47.0,
    ("gc", "selective"): 47.5,
}


def info(message: str) -> None:
    print(f"[info] {message}", file=sys.stderr)


def warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def _normalize_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= v <= 1.0:
        return 100.0 * v
    return v


def _extract_solve_rate_percent(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        direct = _normalize_percent(payload.get("solve_rate"))
        if direct is not None:
            return direct

        results = payload.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            nested = _normalize_percent(results[0].get("solve_rate"))
            if nested is not None:
                return nested

        aggregate = payload.get("aggregate")
        if isinstance(aggregate, dict):
            agg = _normalize_percent(aggregate.get("solve_rate"))
            if agg is not None:
                return agg

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        nested = _normalize_percent(payload[0].get("solve_rate"))
        if nested is not None:
            return nested

    return None


def _load_solve_rate_percent(path: Path) -> Optional[float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        warn(f"Failed to parse JSON: {path} ({exc})")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        warn(f"Failed to read {path}: {exc}")
        return None

    solve_rate = _extract_solve_rate_percent(payload)
    if solve_rate is None:
        warn(f"solve_rate missing in {path}")
    return solve_rate


def _first_solve_rate(paths: list[Path]) -> tuple[Optional[float], Optional[Path]]:
    for path in paths:
        if not path.exists():
            continue
        solve_rate = _load_solve_rate_percent(path)
        if solve_rate is not None:
            return solve_rate, path
    return None, None


def _matched_candidates(results_root: Path, domain: str, mask: str) -> list[Path]:
    # Prefer E3 matched evals, then E3v3, matching user guidance.
    return [
        results_root / f"e3-eval-{domain}-{mask}-ssa-seed42" / "results.json",
        results_root / f"e3-eval-{domain}-{mask}_ssa-seed42" / "results.json",
        results_root / f"e3v3-eval-{domain}-{mask}-ssa-seed42" / "results.json",
        results_root / f"e3v3-eval-{domain}-{mask}_ssa-seed42" / "results.json",
    ]


def _crossmask_path(
    results_root: Path, domain: str, train_mask: str, eval_mask: str
) -> Path:
    return (
        results_root
        / f"crossmask-{domain}-{train_mask}-w-{eval_mask}-m"
        / "results.json"
    )


def _fmt_pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def _build_table(
    sat_matched_blanket: Optional[float],
    sat_matched_selective: Optional[float],
    sat_cross_b_to_s: Optional[float],
    sat_cross_s_to_b: Optional[float],
    gc_matched_blanket: Optional[float],
    gc_matched_selective: Optional[float],
    gc_cross_b_to_s: Optional[float],
    gc_cross_s_to_b: Optional[float],
) -> str:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Condition} & \textbf{Train Mask} & \textbf{Eval Mask} & \textbf{Solve \%} \\",
        r"\midrule",
        f"\\multirow{{4}}{{*}}{{SAT}} & Matched & Blanket & Blanket & {_fmt_pct(sat_matched_blanket)} \\\\",
        f"& Matched & Selective & Selective & {_fmt_pct(sat_matched_selective)} \\\\",
        f"& Cross & Blanket & Selective & {_fmt_pct(sat_cross_b_to_s)} \\\\",
        f"& Cross & Selective & Blanket & {_fmt_pct(sat_cross_s_to_b)} \\\\",
        r"\midrule",
        f"\\multirow{{4}}{{*}}{{GC}} & Matched & Blanket & Blanket & {_fmt_pct(gc_matched_blanket)} \\\\",
        f"& Matched & Selective & Selective & {_fmt_pct(gc_matched_selective)} \\\\",
        f"& Cross & Blanket & Selective & {_fmt_pct(gc_cross_b_to_s)} \\\\",
        f"& Cross & Selective & Blanket & {_fmt_pct(gc_cross_s_to_b)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments"),
        help="Results directory root (default: experiments)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = args.results_dir
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    results_root = results_root.resolve()

    info(f"results_root={results_root}")

    matched: dict[tuple[str, str], Optional[float]] = {}
    for domain in ("sat", "gc"):
        for mask in ("blanket", "selective"):
            candidates = _matched_candidates(results_root, domain, mask)
            solve_rate, source = _first_solve_rate(candidates)
            if solve_rate is None:
                fallback = FALLBACK_MATCHED_PCT[(domain, mask)]
                matched[(domain, mask)] = fallback
                warn(
                    "Matched result missing; using fallback "
                    f"domain={domain} mask={mask} solve={fallback:.1f}"
                )
            else:
                matched[(domain, mask)] = solve_rate
                info(
                    "Matched source "
                    f"domain={domain} mask={mask} solve={solve_rate:.1f} path={source}"
                )

    sat_cross_b_to_s = _load_solve_rate_percent(
        _crossmask_path(results_root, "sat", "blanket", "selective")
    )
    sat_cross_s_to_b = _load_solve_rate_percent(
        _crossmask_path(results_root, "sat", "selective", "blanket")
    )
    gc_cross_b_to_s = _load_solve_rate_percent(
        _crossmask_path(results_root, "gc", "blanket", "selective")
    )
    gc_cross_s_to_b = _load_solve_rate_percent(
        _crossmask_path(results_root, "gc", "selective", "blanket")
    )

    info(
        "Crossmask solve rates "
        f"SAT(B->S)={_fmt_pct(sat_cross_b_to_s)} "
        f"SAT(S->B)={_fmt_pct(sat_cross_s_to_b)} "
        f"GC(B->S)={_fmt_pct(gc_cross_b_to_s)} "
        f"GC(S->B)={_fmt_pct(gc_cross_s_to_b)}"
    )

    table = _build_table(
        sat_matched_blanket=matched[("sat", "blanket")],
        sat_matched_selective=matched[("sat", "selective")],
        sat_cross_b_to_s=sat_cross_b_to_s,
        sat_cross_s_to_b=sat_cross_s_to_b,
        gc_matched_blanket=matched[("gc", "blanket")],
        gc_matched_selective=matched[("gc", "selective")],
        gc_cross_b_to_s=gc_cross_b_to_s,
        gc_cross_s_to_b=gc_cross_s_to_b,
    )

    output_path = (
        repo_root / "paper" / "manuscript" / "parts" / "tables" / "tab_crossmask.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n", encoding="utf-8")
    info(f"Wrote LaTeX table: {output_path}")

    print(table)


if __name__ == "__main__":
    main()
