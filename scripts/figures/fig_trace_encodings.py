#!/usr/bin/env python3
"""Trace encodings figure for tree traversal.

Layout: two rows.
- Row 1: Tree diagram | (a) BFS trace | (b) DFS baseline
- Row 2: (c) DFS, localized (full width, three internal columns)

The two-row layout fits a typical journal column width without shrinking
the text. Subfigure (c) gets the full row so its longer trace can be laid
out across three internal columns at the same font size as (a) and (b).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parents[2]

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

# Color mapping for each token type. Numbers and "NONE" are black.
C_MAP = {
    "START": "purple",
    "FINISH": "purple",
    "BFS": "red",
    "DFS": "red",
    "BACKTRACK": "red",
    "POSITION": "gray",
    "OPTIONS": "#e07ab6",  # pink, slightly darker for readability
    "VISITED": "green",
    "COMMAND": "orange",
    "GOTO": "#1aa3a3",  # teal, slightly darker than cyan
}


def token_color(word):
    return C_MAP.get(word, "black")


def is_keyword(word):
    return word in C_MAP


# ============================================================================
# Trace token strings
# ============================================================================

BFS_TRACE = [
    "START BFS",
    "POSITION 0",
    "POSITION 1",
    "POSITION 2",
    "POSITION 5",
    "POSITION 3",
    "POSITION 4",
    "FINISH",
]

DFS_BASE_TRACE = [
    "START DFS",
    "POSITION 0",
    "POSITION 1",
    "POSITION 0",
    "POSITION 2",
    "POSITION 3",
    "POSITION 2",
    "POSITION 4",
    "POSITION 2",
    "POSITION 0",
    "POSITION 5",
    "POSITION 0",
    "FINISH",
]

DFS_LOC_TRACE = [
    "START DFS",
    "POSITION 0",
    "OPTIONS 1 2 5",
    "VISITED NONE",
    "COMMAND GOTO 1",
    "POSITION 1",
    "OPTIONS NONE",
    "COMMAND BACKTRACK",
    "POSITION 0",
    "OPTIONS 1 2 5",
    "VISITED 1",
    "COMMAND GOTO 2",
    "POSITION 2",
    "OPTIONS 3 4",
    "VISITED NONE",
    "COMMAND GOTO 3",
    "POSITION 3",
    "OPTIONS NONE",
    "COMMAND BACKTRACK",
    "POSITION 2",
    "OPTIONS 3 4",
    "VISITED 3",
    "COMMAND GOTO 4",
    "POSITION 4",
    "OPTIONS NONE",
    "COMMAND BACKTRACK",
    "POSITION 2",
    "OPTIONS 3 4",
    "VISITED 3 4",
    "COMMAND BACKTRACK",
    "POSITION 0",
    "OPTIONS 1 2 5",
    "VISITED 1 2",
    "COMMAND GOTO 5",
    "POSITION 5",
    "OPTIONS NONE",
    "COMMAND BACKTRACK",
    "POSITION 0",
    "OPTIONS 1 2 5",
    "VISITED 1 2 5",
    "FINISH",
]


# ============================================================================
# Drawing helpers
# ============================================================================


def draw_trace_lines(ax, lines, n_cols, title, font_size=10):
    """Draw a list of trace lines into `n_cols` columns within `ax`.

    Each line is rendered as a colored keyword followed by black operands.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 1.06, title, ha="center", va="bottom",
        fontsize=font_size + 1, weight="bold", transform=ax.transAxes,
    )

    # Compute lines-per-column from the requested n_cols.
    rows_per_col = (len(lines) + n_cols - 1) // n_cols

    # Layout parameters in axes-fraction coordinates.
    col_left_margin = 0.01
    col_gap = 0.07
    col_width = (1.0 - 2 * col_left_margin - (n_cols - 1) * col_gap) / n_cols
    line_height = 0.96 / rows_per_col
    y_top = 0.98

    for i, line in enumerate(lines):
        col_idx = i // rows_per_col
        row_idx = i % rows_per_col
        x_left = col_left_margin + col_idx * (col_width + col_gap)
        y = y_top - row_idx * line_height

        tokens = line.split()
        if not tokens:
            continue

        # First token: always a colored keyword in our traces.
        first = tokens[0]
        rest = tokens[1:]
        ax.text(
            x_left, y, first, color=token_color(first),
            weight="bold", fontsize=font_size, ha="left", va="top",
            transform=ax.transAxes, family="monospace",
        )

        # If the second token is also a keyword (e.g. COMMAND GOTO),
        # render it with its own keyword color.
        if rest and is_keyword(rest[0]):
            second = rest[0]
            operands = rest[1:]
            ax.text(
                x_left + col_width * 0.38, y, second,
                color=token_color(second), weight="bold",
                fontsize=font_size, ha="left", va="top",
                transform=ax.transAxes, family="monospace",
            )
            if operands:
                ax.text(
                    x_left + col_width - 0.005, y, " ".join(operands),
                    color="black", fontsize=font_size, ha="right",
                    va="top", transform=ax.transAxes, family="monospace",
                )
        else:
            if rest:
                ax.text(
                    x_left + col_width - 0.005, y, " ".join(rest),
                    color="black", fontsize=font_size, ha="right",
                    va="top", transform=ax.transAxes, family="monospace",
                )


def draw_tree(ax, title="Tree structure"):
    """Draw a small reference tree (rooted at 0; children 1, 2, 5; 2 has 3, 4)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 1.06, title, ha="center", va="bottom",
        fontsize=11, weight="bold", transform=ax.transAxes,
    )

    nodes = {
        "0": (0.5, 0.85),
        "1": (0.18, 0.55),
        "2": (0.5, 0.55),
        "5": (0.82, 0.55),
        "3": (0.35, 0.20),
        "4": (0.65, 0.20),
    }
    edges = [("0", "1"), ("0", "2"), ("0", "5"), ("2", "3"), ("2", "4")]
    radius = 0.085

    for u, v in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        ax.plot(
            [x1 + ux * radius, x2 - ux * radius],
            [y1 + uy * radius, y2 - uy * radius],
            color="black", lw=1.4, zorder=2,
        )
    for node, (x, y) in nodes.items():
        c = patches.Circle(
            (x, y), radius, facecolor="white", edgecolor="black",
            lw=1.4, zorder=3,
        )
        ax.add_patch(c)
        ax.text(
            x, y, node, ha="center", va="center",
            fontsize=11, weight="bold", zorder=4,
        )


# ============================================================================
# Figure layout: two rows
# Row 1: Tree | (a) BFS | (b) DFS-baseline (each ~1/3 of width)
# Row 2: (c) DFS-localized in 3 internal columns (full width)
# ============================================================================

fig = plt.figure(figsize=(7.0, 6.0))
gs = fig.add_gridspec(
    2, 3,
    width_ratios=[0.30, 0.30, 0.40],
    height_ratios=[1.4, 1.6],  # row 1 taller so 13-line DFS baseline is readable
    wspace=0.05,
    hspace=0.18,
)

ax_tree = fig.add_subplot(gs[0, 0])
ax_bfs = fig.add_subplot(gs[0, 1])
ax_dfs_base = fig.add_subplot(gs[0, 2])
ax_dfs_loc = fig.add_subplot(gs[1, :])  # spans all 3 columns

draw_tree(ax_tree)
draw_trace_lines(ax_bfs, BFS_TRACE, n_cols=1, title="(a) BFS trace", font_size=10)
draw_trace_lines(ax_dfs_base, DFS_BASE_TRACE, n_cols=1, title="(b) DFS, baseline", font_size=10)
draw_trace_lines(ax_dfs_loc, DFS_LOC_TRACE, n_cols=3, title="(c) DFS, localized", font_size=9.5)

plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02)

# ============================================================================
# Output
# ============================================================================
out_dir = REPO_ROOT / "output" / "pictures"
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(
        out_dir / f"fig_trace_encodings.{ext}",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=300,
    )
plt.close(fig)
print("Saved fig_trace_encodings")
