#!/usr/bin/env python3
"""Generate the flagship SAT n=50 results tables for the MLJ paper.

Produces two LaTeX tabulars:
  - paper/manuscript/parts/tables/tab_flagship.tex (main body, panels A and B)
  - paper/manuscript/parts/tables/tab_verifier_swap.tex (appendix, panels C and D)

Panels A and B carry the spine of the empirical evaluation (state-rebuilt
transfer and history-reduction baselines). Panels C and D characterize
verification methods and oracle decomposition; they live in the appendix
because they extend the spine rather than carrying it.

Bolding policy
--------------
None. Every numeric cell is rendered in plain math mode. Attempts to bold
either "the best value" or "the SSA headline" ran into two incompatible
conventions:
  - Best-value bolding would highlight MLP (state features) at 99.2% and
    Oracle at 100.0% in panels they are listed as references or upper
    bounds rather than as comparable methods.
  - Method-highlight bolding would mark SSA's result even when its value
    was not the largest in the column, which breaks the usual reading
    convention.
We therefore let the prose in the paper body and the caption carry the
interpretation and keep the tables purely informational.

Values are sourced from the canonical SAT n=50 results (5 seeds x 200
planted instances, test seed 42, budget 4096) and must match the auxiliary
tables tab_sat_n50_master and tab_verifier_comparison.

Usage:
    python scripts/tables/gen_tab_flagship.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_FLAGSHIP = REPO_ROOT / "output" / "tables" / "tab_flagship.tex"
DEFAULT_OUTPUT_VERIFIER_SWAP = REPO_ROOT / "output" / "tables" / "tab_verifier_swap.tex"


def wrap(value: str) -> str:
    """Wrap a math-mode value in dollar signs."""
    return f"${value}$"


# ---- Panel A: evaluation protocol comparison ----
# Columns: label, train, cumulative, state-rebuilt, SR+rand.var.
PANEL_A = [
    ("Causal", "5k cum.", "16.5 \\pm 11.2", "4.8 \\pm 1.2", "4.0 \\pm 0.0"),
    ("SSA", "5k cum.", "28.9 \\pm 1.9", "93.4 \\pm 2.3", "89.5 \\pm 1.1"),
    ("Causal (state-only)", "5k state", None, "69.3 \\pm 18.1", "44.9 \\pm 8.9"),
    (
        "Causal (state-only)$^{\\dagger}$",
        "100k state",
        None,
        "93.0 \\pm 2.0",
        "86.0 \\pm 6.9^{\\dagger}",
    ),
    ("MLP (state features)", "100k state", None, "99.2 \\pm 0.8", "100.0 \\pm 0.0"),
]


# ---- Panel B: history-reduction and architecture baselines ----
# Columns: label, cumulative, state-rebuilt, delta-pp
PANEL_B = [
    ("SSA", "28.9 \\pm 1.9", "93.4 \\pm 2.3", "+64.5"),
    ("Causal", "16.5 \\pm 11.2", "4.8 \\pm 1.2", "-11.7"),
    ("Block Dropout $p{=}0.5$", "24.8 \\pm 4.4", "3.8 \\pm 1.9", "-21.0"),
    ("Sliding Window $k{=}3$", "5.3 \\pm 1.5", "3.2 \\pm 0.4", "-2.1"),
    ("Null History", "4.2 \\pm 1.3", "4.2 \\pm 0.4", "+0.1"),
    ("LSTM", "6.8 \\pm 3.6", "3.1 \\pm 0.9", "-3.7"),
]


# ---- Panel C: verification methods (fixed SSA branching policy) ----
# Columns: label, supervision, solve-%, gap-closed
# Note: the MLP and DVP dead-end classifiers fire zero proactive interventions
# when plugged into the SSA search loop (mlexp note 7648), so their solve
# rate is identical to the reactive baseline. That failure is described in
# Section 7.5 of the text; we omit the row here to keep the panel focused on
# verifiers that actually affect solve rate.
PANEL_C = [
    ("Propagation only (no learned verifier)", "---", "32.2 \\pm 2.8", "0.0\\%"),
    ("Bounded CDCL sidecar (50)", "symbolic", "79.1 \\pm 3.0", "69.2\\%"),
    ("Bounded CDCL sidecar (100)", "symbolic", "80.1 \\pm 3.0", "70.6\\%"),
    ("SSA learned backtrack token", "traces", "93.4 \\pm 2.3", "90.3\\%"),
    ("Perfect dead-end oracle", "perfect", "100.0 \\pm 0.0", "100.0\\%"),
]


# ---- Panel D: oracle decomposition ----
# Columns: label, no-oracle, perfect-oracle, lift-pp
PANEL_D = [
    ("SSA", "28.9 \\pm 1.9", "100.0 \\pm 0.0", "+71.1"),
    ("Causal", "16.5 \\pm 11.2", "46.3 \\pm 38.0", "+29.8"),
]


def fmt_math(value: str | None) -> str:
    if value is None:
        return "---"
    return wrap(value)


def panel_a_rows() -> list[str]:
    lines: list[str] = []
    for label, train, cumulative, sr, sr_rand in PANEL_A:
        cells = [fmt_math(cumulative), fmt_math(sr), fmt_math(sr_rand)]
        lines.append(
            f"{label} & {train} & {cells[0]} & {cells[1]} & {cells[2]} \\\\"
        )
    return lines


def panel_b_rows() -> list[str]:
    lines: list[str] = []
    for label, cumulative, sr, delta in PANEL_B:
        cells = [fmt_math(cumulative), fmt_math(sr), fmt_math(delta)]
        lines.append(
            f"{label} & & {cells[0]} & {cells[1]} & {cells[2]} \\\\"
        )
    return lines


def panel_c_rows() -> list[str]:
    """Panel C: all numeric cells wrapped in math mode for consistency."""
    lines: list[str] = []
    for label, supervision, solve, gap in PANEL_C:
        formatted_solve = fmt_math(solve)
        formatted_gap = fmt_math(gap)
        lines.append(
            f"{label} & {supervision} & \\multicolumn{{2}}{{c}}{{{formatted_solve}}} & {formatted_gap} \\\\"
        )
    return lines


def panel_d_rows() -> list[str]:
    lines: list[str] = []
    for label, no_oracle, perfect, lift in PANEL_D:
        cells = [fmt_math(no_oracle), fmt_math(perfect), fmt_math(lift)]
        lines.append(
            f"{label} & & {cells[0]} & {cells[1]} & {cells[2]} \\\\"
        )
    return lines


def render_flagship_table() -> str:
    """Render panels A and B for the main-text flagship table."""
    lines: list[str] = []
    lines.append("% Generated by scripts/tables/gen_tab_flagship.py")
    lines.append("% Bolding policy: none. See the script docstring for the rationale.")
    lines.append("\\begin{tabular}{l c ccc}")
    lines.append("\\toprule")

    # Panel A header
    lines.append(
        "\\multicolumn{5}{l}{\\textbf{A.} \\textit{Evaluation protocol comparison}} \\\\"
    )
    lines.append("\\midrule")
    lines.append(
        "& \\textbf{Train} & \\textbf{Cumulative} & \\textbf{State-rebuilt} "
        "& \\textbf{SR + rand.\\ var.} \\\\"
    )
    lines.append("\\cmidrule(lr){3-5}")
    lines.extend(panel_a_rows())
    lines.append("\\midrule")

    # Panel B header
    lines.append(
        "\\multicolumn{5}{l}{\\textbf{B.} \\textit{History-reduction and architecture baselines} "
        "(all use 5k cumulative traces)} \\\\"
    )
    lines.append("\\midrule")
    lines.append(
        "& & \\textbf{Cumulative} & \\textbf{State-rebuilt} & \\textbf{$\\Delta$ (pp)} \\\\"
    )
    lines.append("\\cmidrule(lr){3-5}")
    lines.extend(panel_b_rows())
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def render_verifier_swap_table() -> str:
    """Render panels C and D for the appendix verifier-swap table."""
    lines: list[str] = []
    lines.append("% Generated by scripts/tables/gen_tab_flagship.py")
    lines.append("% Bolding policy: none. See the script docstring for the rationale.")
    lines.append("\\begin{tabular}{l c ccc}")
    lines.append("\\toprule")

    # Panel C header
    lines.append(
        "\\multicolumn{5}{l}{\\textbf{A.} \\textit{Verification methods} (fixed SSA branching policy)} \\\\"
    )
    lines.append("\\midrule")
    lines.append(
        "\\textbf{Verifier} & \\textbf{Supervision} & \\multicolumn{2}{c}{\\textbf{Solve \\%}} "
        "& \\textbf{Gap closed} \\\\"
    )
    lines.append("\\cmidrule(lr){1-5}")
    lines.extend(panel_c_rows())
    lines.append("\\midrule")

    # Panel D header
    lines.append(
        "\\multicolumn{5}{l}{\\textbf{B.} \\textit{Oracle decomposition}} \\\\"
    )
    lines.append("\\midrule")
    lines.append(
        "& & \\textbf{No oracle} & \\textbf{+ Perfect oracle} & \\textbf{Lift (pp)} \\\\"
    )
    lines.append("\\cmidrule(lr){3-5}")
    lines.extend(panel_d_rows())
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-flagship",
        default=str(DEFAULT_OUTPUT_FLAGSHIP),
        help=f"Output path for the main flagship table (default: {DEFAULT_OUTPUT_FLAGSHIP}).",
    )
    parser.add_argument(
        "--output-verifier-swap",
        default=str(DEFAULT_OUTPUT_VERIFIER_SWAP),
        help=f"Output path for the appendix verifier-swap table (default: {DEFAULT_OUTPUT_VERIFIER_SWAP}).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print both tables to stdout instead of writing to disk.",
    )
    args = parser.parse_args()

    flagship_content = render_flagship_table()
    verifier_swap_content = render_verifier_swap_table()

    if args.stdout:
        print("% --- tab_flagship.tex ---")
        print(flagship_content, end="")
        print("\n% --- tab_verifier_swap.tex ---")
        print(verifier_swap_content, end="")
        return

    flagship_path = Path(args.output_flagship)
    flagship_path.parent.mkdir(parents=True, exist_ok=True)
    flagship_path.write_text(flagship_content)
    print(f"Wrote {flagship_path}")

    verifier_swap_path = Path(args.output_verifier_swap)
    verifier_swap_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_swap_path.write_text(verifier_swap_content)
    print(f"Wrote {verifier_swap_path}")


if __name__ == "__main__":
    main()
