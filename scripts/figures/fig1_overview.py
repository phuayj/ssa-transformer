#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

COLOR_PROBLEM = "#555555"
COLOR_HISTORY = "#d9d9d9"
COLOR_CURRENT_FILL = "#d1e5f0"
COLOR_CURRENT_EDGE = "#2166ac"
COLOR_CAUSAL = "#d73027"
COLOR_SSA = "#2166ac"
COLOR_TEXT = "#222222"
COLOR_TREE_LINK = "#9a9a9a"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    setup_style()

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=300)
    ax.set_xlim(0.3, 10.2)
    ax.set_ylim(-0.35, 6.0)
    ax.axis("off")

    # ── Sequence-block layout ──
    y_center = 1.1
    h = 0.8
    y_bottom = y_center - h / 2  # 0.7
    y_top = y_center + h / 2  # 1.5

    # Problem block
    x_prob = 0.55
    w_prob = 1.2
    cx_prob = x_prob + w_prob / 2

    # History blocks (wider to fit cumulative state labels)
    w_b = 1.45
    x_b1 = x_prob + w_prob
    x_b2 = x_b1 + w_b
    x_b3 = x_b2 + w_b
    x_b4 = x_b3 + w_b

    cx_b1 = x_b1 + w_b / 2
    cx_b2 = x_b2 + w_b / 2
    cx_b3 = x_b3 + w_b / 2
    cx_b4 = x_b4 + w_b / 2

    # Current block
    x_curr = x_b4 + w_b
    w_curr = 1.55
    cx_curr = x_curr + w_curr / 2

    # ? box
    gap = 0.15
    x_q = x_curr + w_curr + gap
    w_q = 0.55

    # ── 1. Draw blocks ──

    # Problem
    ax.add_patch(
        patches.Rectangle(
            (x_prob, y_bottom),
            w_prob,
            h,
            facecolor=COLOR_PROBLEM,
            edgecolor=COLOR_PROBLEM,
            zorder=3,
        )
    )
    # CNF formula: (v1 ∨ v2) ∧ (¬v1 ∨ v2) ∧ (¬v1 ∨ ¬v2)
    # Unique solution: v1=F, v2=T  (matches s_t in current block)
    cnf_lines = [
        r"$(v_1 \vee v_2)$",
        r"$(\bar{v}_1 \vee v_2)$",
        r"$(\bar{v}_1 \vee \bar{v}_2)$",
    ]
    line_spacing = 0.22
    y_start = y_center + line_spacing
    for i, line in enumerate(cnf_lines):
        ax.text(
            cx_prob,
            y_start - i * line_spacing,
            line,
            color="white",
            ha="center",
            va="center",
            zorder=4,
            fontsize=7.5,
        )

    # History blocks with cumulative states
    # Each tuple: (x, state_label, show_dead_end)
    # Cumulative states following the search tree:
    #   Step 1: assign v1=T                → {v1=T}
    #   Step 2: assign v2=T under v1=T     → {v1=T, v2=T}  dead end
    #   Step 3: assign v2=F under v1=T     → {v1=T, v2=F}  dead end
    #   Step 4: backtrack, assign v1=F     → {v1=F}
    history_blocks = [
        (x_b1, r"$v_1\!=\!T$", False),
        (x_b2, r"$v_1\!=\!T,\; v_2\!=\!T$", True),
        (x_b3, r"$v_1\!=\!T,\; v_2\!=\!F$", True),
        (x_b4, r"$v_1\!=\!F$", False),
    ]
    for x, label, show_dead_end in history_blocks:
        ax.add_patch(
            patches.Rectangle(
                (x, y_bottom),
                w_b,
                h,
                facecolor=COLOR_HISTORY,
                edgecolor="#b7b7b7",
                linewidth=1.2,
                zorder=3,
            )
        )
        if show_dead_end:
            # State on upper line, × on lower line
            ax.text(
                x + w_b / 2,
                y_center + 0.12,
                label,
                color=COLOR_TEXT,
                ha="center",
                va="center",
                zorder=4,
                fontsize=8,
            )
            ax.text(
                x + w_b / 2,
                y_center - 0.18,
                r"$\times$",
                color=COLOR_CAUSAL,
                weight="bold",
                ha="center",
                va="center",
                zorder=4,
                fontsize=12,
            )
        else:
            ax.text(
                x + w_b / 2,
                y_center,
                label,
                color=COLOR_TEXT,
                ha="center",
                va="center",
                zorder=4,
                fontsize=8.5,
            )

    # Current Block
    ax.add_patch(
        patches.Rectangle(
            (x_curr, y_bottom),
            w_curr,
            h,
            facecolor=COLOR_CURRENT_FILL,
            edgecolor=COLOR_CURRENT_EDGE,
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.text(
        cx_curr,
        y_center + 0.05,
        r"$v_1\!=\!F,\; v_2\!=\!T$",
        color=COLOR_TEXT,
        ha="center",
        va="center",
        zorder=4,
        fontsize=9,
    )
    ax.text(
        cx_curr,
        y_center - 0.22,
        "$s_t$",
        color=COLOR_CURRENT_EDGE,
        ha="center",
        va="center",
        zorder=4,
        fontsize=10,
    )

    # ? Box
    ax.add_patch(
        patches.Rectangle(
            (x_q, y_bottom),
            w_q,
            h,
            facecolor="white",
            edgecolor=COLOR_TEXT,
            linewidth=1.0,
            zorder=3,
        )
    )
    ax.text(
        x_q + w_q / 2,
        y_center,
        "?",
        color=COLOR_TEXT,
        weight="bold",
        ha="center",
        va="center",
        zorder=4,
        fontsize=16,
    )

    # ── 2. Deployment cut & wash ──
    wash_w = w_b * 4
    ax.add_patch(
        patches.Rectangle(
            (x_b1, y_bottom), wash_w, h, facecolor="#e0e0e0", alpha=0.6, zorder=5
        )
    )
    # ── 3. Causal attention arrows (per-block, above) ──
    # Causal attention: ? attends to every preceding block
    cx_q = x_q + w_q / 2
    causal_targets = [cx_prob, cx_b1, cx_b2, cx_b3, cx_b4, cx_curr]
    causal_rads = [0.22, 0.18, 0.14, 0.10, 0.06, 0.20]
    for target_cx, rad in zip(causal_targets, causal_rads):
        arc = FancyArrowPatch(
            (cx_q, y_top + 0.05),
            (target_cx, y_top + 0.05),
            connectionstyle=f"arc3,rad={rad}",
            color=COLOR_CAUSAL,
            linestyle="--",
            linewidth=1.2,
            arrowstyle="-|>",
            mutation_scale=10,
            zorder=2,
        )
        ax.add_patch(arc)

    ax.text(
        (cx_q + cx_prob) / 2,
        y_top + 1.05,
        "Causal",
        color=COLOR_CAUSAL,
        weight="bold",
        ha="center",
        va="bottom",
        zorder=8,
        fontsize=11,
    )

    # ── 4. SSA attention arrows (below) ──
    # SSA: ? attends only to current block and problem (both originate from ?)
    ssa_targets = [(cx_curr, -0.20), (cx_prob, -0.14)]
    for target_cx, rad in ssa_targets:
        arc = FancyArrowPatch(
            (cx_q, y_bottom - 0.05),
            (target_cx, y_bottom - 0.05),
            connectionstyle=f"arc3,rad={rad}",
            color=COLOR_SSA,
            linestyle="-",
            linewidth=1.8,
            arrowstyle="-|>",
            mutation_scale=12,
            zorder=2,
        )
        ax.add_patch(arc)
    ax.text(
        cx_prob - 0.15,
        y_bottom - 0.22,
        "SSA",
        color=COLOR_SSA,
        weight="bold",
        ha="right",
        va="center",
        zorder=8,
        fontsize=11,
    )

    # SSA Blocked markers on history blocks
    for x in [x_b1, x_b2, x_b3, x_b4]:
        center_x = x + w_b / 2
        center_y = y_bottom - 0.22
        ring = patches.Circle(
            (center_x, center_y),
            radius=0.07,
            facecolor="white",
            edgecolor=COLOR_SSA,
            linewidth=1.2,
            alpha=0.9,
            zorder=8,
        )
        ax.add_patch(ring)
        ax.plot(
            [center_x - 0.05, center_x + 0.05],
            [center_y - 0.05, center_y + 0.05],
            color=COLOR_SSA,
            linewidth=1.2,
            alpha=0.9,
            zorder=8,
        )

    # ── 5. Punchline badges ──
    badge_c_y = y_top + 0.2
    ax.add_patch(
        patches.Circle(
            (cx_q, badge_c_y),
            radius=0.2,
            facecolor="white",
            edgecolor=COLOR_CAUSAL,
            linewidth=1.5,
            zorder=8,
        )
    )
    ax.text(
        cx_q,
        badge_c_y - 0.01,
        r"$\times$",
        color=COLOR_CAUSAL,
        weight="bold",
        ha="center",
        va="center",
        fontsize=13,
        zorder=9,
    )

    badge_s_y = y_bottom - 0.2
    ax.add_patch(
        patches.Circle(
            (cx_q, badge_s_y),
            radius=0.2,
            facecolor="white",
            edgecolor=COLOR_SSA,
            linewidth=1.5,
            zorder=8,
        )
    )
    ax.text(
        cx_q,
        badge_s_y + 0.02,
        "✓",
        color=COLOR_SSA,
        fontfamily="DejaVu Sans",
        weight="bold",
        ha="center",
        va="center",
        fontsize=12,
        zorder=9,
    )

    # ── 6. Tree topology ──
    # Search trace shown above the sequence:
    #   root -> v1=T -> {v2=T, v2=F} -> dead-end
    #   root -> v1=F -> v2=T (current state)
    pos_root = (cx_prob, 5.6)
    pos_v1T = (cx_b1, 4.65)
    pos_v1F = (cx_b4, 4.65)
    pos_v2T_dead = (cx_b2, 3.7)
    pos_v2F_dead = (cx_b3, 3.7)
    pos_curr = (cx_curr, 3.7)

    LOGGER.info(
        "Tree search trace nodes aligned to blocks: root=%s problem_x=%.3f "
        "failed_branch=(%s -> [%s, %s]) active_branch=(%s -> %s)",
        pos_root,
        cx_prob,
        pos_v1T,
        pos_v2T_dead,
        pos_v2F_dead,
        pos_v1F,
        pos_curr,
    )

    # Dotted connections from tree nodes down to sequence blocks
    connections = [
        (pos_root, (cx_prob, y_top)),
        (pos_v1T, (cx_b1, y_top)),
        (pos_v2T_dead, (cx_b2, y_top)),
        (pos_v2F_dead, (cx_b3, y_top)),
        (pos_v1F, (cx_b4, y_top)),
        (pos_curr, (cx_curr, y_top)),
    ]
    for p1, p2 in connections:
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=COLOR_TREE_LINK,
            linestyle=":",
            linewidth=1.0,
            alpha=0.4,
            zorder=1,
        )

    # Tree edges
    edges = [
        (pos_root, pos_v1T, "#888888", 0.4),
        (pos_root, pos_v1F, "#555555", 1.0),
        (pos_v1T, pos_v2T_dead, "#888888", 0.4),
        (pos_v1T, pos_v2F_dead, "#888888", 0.4),
        (pos_v1F, pos_curr, "#555555", 1.0),
    ]
    for p1, p2, color, alpha in edges:
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=1.5,
            alpha=alpha,
            zorder=9,
        )

    # Tree nodes
    for pos, fc, ec, alpha in [
        (pos_root, COLOR_PROBLEM, COLOR_PROBLEM, 1.0),
        (pos_v1T, "#e6e6e6", "#cccccc", 0.6),
        (pos_v1F, "#d9d9d9", "#999999", 1.0),
        (pos_v2T_dead, "#e6e6e6", "#cccccc", 0.6),
        (pos_v2F_dead, "#e6e6e6", "#cccccc", 0.6),
        (pos_curr, COLOR_CURRENT_FILL, COLOR_CURRENT_EDGE, 1.0),
    ]:
        circle = patches.Circle(
            pos,
            radius=0.15,
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
            alpha=alpha,
            zorder=10,
        )
        ax.add_patch(circle)

    # Dead-end markers in tree
    for pos in [pos_v2T_dead, pos_v2F_dead]:
        ax.text(
            pos[0],
            pos[1] - 0.48,
            r"$\times$",
            color=COLOR_CAUSAL,
            fontsize=16,
            ha="center",
            va="center",
            weight="bold",
            alpha=0.6,
            zorder=10,
        )

    # Tree node labels
    ax.text(
        pos_root[0],
        pos_root[1] + 0.22,
        "root",
        fontsize=10,
        ha="center",
        va="bottom",
        weight="bold",
        color=COLOR_PROBLEM,
        zorder=11,
    )
    ax.text(
        pos_v1T[0] - 0.22,
        pos_v1T[1],
        r"$v_1\!=\!T$",
        fontsize=10,
        ha="right",
        va="center",
        alpha=0.6,
        zorder=11,
    )
    ax.text(
        pos_v1F[0] + 0.22,
        pos_v1F[1],
        r"$v_1\!=\!F$",
        fontsize=10,
        ha="left",
        va="center",
        alpha=1.0,
        zorder=11,
    )
    ax.text(
        pos_v2T_dead[0] - 0.22,
        pos_v2T_dead[1],
        r"$v_2\!=\!T$",
        fontsize=10,
        ha="right",
        va="center",
        alpha=0.6,
        zorder=11,
    )
    ax.text(
        pos_v2F_dead[0] + 0.22,
        pos_v2F_dead[1],
        r"$v_2\!=\!F$",
        fontsize=10,
        ha="left",
        va="center",
        alpha=0.6,
        zorder=11,
    )
    ax.text(
        pos_curr[0] + 0.22,
        pos_curr[1],
        r"$v_2\!=\!T$",
        fontsize=10,
        ha="left",
        va="center",
        alpha=1.0,
        zorder=11,
    )
    ax.text(
        pos_curr[0] + 0.22,
        pos_curr[1] - 0.24,
        r"$s_t$",
        fontsize=10,
        ha="left",
        va="center",
        color=COLOR_CURRENT_EDGE,
        zorder=11,
    )

    # ── Output ──
    out_dir = REPO_ROOT / "output" / "pictures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig1_overview.pdf"
    png_path = out_dir / "fig1_overview.png"

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    LOGGER.info("Saved %s", pdf_path)
    LOGGER.info("Saved %s", png_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
