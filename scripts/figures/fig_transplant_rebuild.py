#!/usr/bin/env python3
"""History transplant + state-rebuilt inference figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

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

# Colors
C_HIST = "#d9d9d9"
C_HIST2 = "#bdbdbd"
C_STATE = "#d1e5f0"
C_STATE_EDGE = "#2166ac"
C_PROB = "#555555"
C_CAUSAL = "#d73027"
C_SSA = "#2166ac"

fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(7.5, 4.0), gridspec_kw={"height_ratios": [1.2, 1]}
)
plt.subplots_adjust(hspace=0.1)


def draw_block(
    ax, x, y, w, h, text, bg, ec="black", tc="black", lw=1, fs=11, hatch=None, text_bg=None, cross=False
):
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        facecolor=bg,
        edgecolor=ec,
        linewidth=lw,
        hatch=hatch,
    )
    ax.add_patch(rect)
    text_kwargs = dict(ha="center", va="center", color=tc, fontsize=fs)
    if text_bg is not None:
        text_kwargs["bbox"] = dict(facecolor=text_bg, edgecolor='none', pad=1.5)
    
    if cross:
        ax.text(x + w / 2 - 0.1, y + h / 2, text, **text_kwargs)
        ax.text(x + w - 0.2, y + h / 2, "$\\times$", ha="center", va="center", color=C_CAUSAL, fontsize=fs+3, weight="bold")
    else:
        ax.text(x + w / 2, y + h / 2, text, **text_kwargs)


def draw_blocked(ax, cx, cy, color):
    """Draw ⊘ marker."""
    r = 0.09
    ax.add_patch(patches.Circle((cx, cy), r, fill=False, ec=color, lw=1.8, zorder=5))
    ax.plot(
        [cx - r * 0.7, cx + r * 0.7],
        [cy - r * 0.7, cy + r * 0.7],
        color=color,
        lw=1.8,
        zorder=5,
    )


# ═══════════════════════════════════════════════════════════════
# Panel (a): History Transplant Test
# ═══════════════════════════════════════════════════════════════
ax_a.set_xlim(-0.1, 10.8)
ax_a.set_ylim(-0.35, 2.35)
ax_a.axis("off")

ax_a.text(0.0, 2.2, "(a)", fontsize=13, weight="bold", va="center")
ax_a.text(
    0.60,
    2.2,
    "History transplant: same state, different history",
    fontsize=12,
    va="center",
)

bh = 0.7
gap = 0.06
y1 = 1.2  # top row
y2 = 0.1  # bottom row
lbl_xa = 0.0
blk_xa = 0.8  # indent blocks to make room for H₁/H₂ labels

# ── Row 1: Trace with history H₁ ──
ax_a.text(
    lbl_xa,
    y1 + bh / 2,
    "$H_1$:",
    ha="left",
    va="center",
    fontsize=11,
    color="#666666",
    weight="bold",
)
x = blk_xa
draw_block(ax_a, x, y1, 1.1, bh, "Problem", C_PROB, tc="white", fs=10)
x += 1.1 + gap
h1_x = x
draw_block(ax_a, x, y1, 1.1, bh, "$v_1\\!=\\!T$", C_HIST, fs=11)
x += 1.1 + gap
draw_block(ax_a, x, y1, 1.1, bh, "$v_2\\!=\\!T$", C_HIST, fs=11, cross=True)
x += 1.1 + gap
s_x = x
draw_block(
    ax_a,
    x,
    y1,
    1.6,
    bh,
    "$v_1\\!=\\!F,\\; v_2\\!=\\!T$",
    C_STATE,
    ec=C_STATE_EDGE,
    lw=1.5,
    fs=10,
)
x += 1.6
ax_a.annotate(
    "",
    xy=(x + 0.3, y1 + bh / 2),
    xytext=(x + 0.05, y1 + bh / 2),
    arrowprops=dict(arrowstyle="-|>", lw=1.3),
)
pred_x = x + 0.35
draw_block(ax_a, pred_x, y1 + 0.12, 0.55, bh - 0.24, "$P_1$", "white", fs=12)

# ── Row 2: Trace with history H₂ ──
ax_a.text(
    lbl_xa,
    y2 + bh / 2,
    "$H_2$:",
    ha="left",
    va="center",
    fontsize=11,
    color="#666666",
    weight="bold",
)
x = blk_xa
draw_block(ax_a, x, y2, 1.1, bh, "Problem", C_PROB, tc="white", fs=10)
x += 1.1 + gap
draw_block(ax_a, x, y2, 1.1, bh, "$v_2\\!=\\!F$", C_HIST2, fs=11)
x += 1.1 + gap
draw_block(
    ax_a, x, y2, 1.1, bh, "$v_1\\!=\\!T$", C_HIST2, fs=11, cross=True
)
x += 1.1 + gap
draw_block(
    ax_a,
    x,
    y2,
    1.6,
    bh,
    "$v_1\\!=\\!F,\\; v_2\\!=\\!T$",
    C_STATE,
    ec=C_STATE_EDGE,
    lw=1.5,
    fs=10,
)
x += 1.6
ax_a.annotate(
    "",
    xy=(x + 0.3, y2 + bh / 2),
    xytext=(x + 0.05, y2 + bh / 2),
    arrowprops=dict(arrowstyle="-|>", lw=1.3),
)
draw_block(ax_a, pred_x, y2 + 0.12, 0.55, bh - 0.24, "$P_2$", "white", fs=12)

# SSA blocked markers under history blocks
for row_y in [y1, y2]:
    for hx in [h1_x + 1.1 / 2, h1_x + 1.1 + gap + 1.1 / 2]:
        draw_blocked(ax_a, hx, row_y - 0.16, C_SSA)

# Results bracket + text
bx = pred_x + 0.55 + 0.1
ax_a.plot(
    [bx, bx + 0.06, bx + 0.06, bx],
    [y2 + bh / 2, y2 + bh / 2, y1 + bh / 2, y1 + bh / 2],
    color="#444444",
    lw=1,
)

mid = (y1 + y2 + bh) / 2
ax_a.text(
    bx + 0.15,
    mid + 0.28,
    "SSA: agreement 100.0%",
    ha="left",
    va="center",
    color=C_SSA,
    fontsize=11.5,
    weight="bold",
)
ax_a.text(
    bx + 0.15,
    mid - 0.28,
    "Causal: agreement 71.4%",
    ha="left",
    va="center",
    color=C_CAUSAL,
    fontsize=11.5,
    weight="bold",
)


# ═══════════════════════════════════════════════════════════════
# Panel (b): State-Rebuilt Inference
# ═══════════════════════════════════════════════════════════════
ax_b.set_xlim(-0.1, 10.8)
ax_b.set_ylim(-0.1, 2.15)
ax_b.axis("off")

ax_b.text(0.0, 2.0, "(b)", fontsize=13, weight="bold", va="center")
ax_b.text(
    0.60,
    2.0,
    "State-rebuilt inference: discard history at deployment",
    fontsize=12,
    va="center",
)

y_train = 1.0
y_deploy = 0.0

# ── Training row ──
lbl_x = 0.0
blk_x0 = 2.2  # indent blocks to make room for "Deployment:" label
ax_b.text(
    lbl_x,
    y_train + bh / 2,
    "Training:",
    fontsize=11,
    weight="bold",
    va="center",
    ha="left",
    color="#444444",
)
x = blk_x0
draw_block(ax_b, x, y_train, 1.1, bh, "Problem", C_PROB, tc="white", fs=10)
x += 1.1 + gap
for label in ["$h_1$", "$h_2$", "$h_3$"]:
    draw_block(ax_b, x, y_train, 0.7, bh, label, C_HIST, fs=11)
    x += 0.7 + gap
draw_block(
    ax_b,
    x,
    y_train,
    1.6,
    bh,
    "$v_1\\!=\\!F,\\; v_2\\!=\\!T$",
    C_STATE,
    ec=C_STATE_EDGE,
    lw=1.5,
    fs=10,
)

# ── Deployment row ──
ax_b.text(
    lbl_x,
    y_deploy + bh / 2,
    "Deployment:",
    fontsize=11,
    weight="bold",
    va="center",
    ha="left",
    color="#444444",
)
x = blk_x0
draw_block(ax_b, x, y_deploy, 1.1, bh, "Problem", C_PROB, tc="white", fs=10)
x += 1.1 + gap
# Dashed gap where history was
gap_start = x
gap_end = gap_start + 0.7 * 3 + gap * 2
ax_b.plot(
    [gap_start, gap_end],
    [y_deploy + bh / 2, y_deploy + bh / 2],
    color="#aaaaaa",
    ls="--",
    lw=1.5,
)
ax_b.text(
    (gap_start + gap_end) / 2,
    y_deploy + bh / 2 + 0.12,
    "discarded",
    ha="center",
    va="bottom",
    fontsize=10,
    color="#999999",
    style="italic",
)
x = gap_end + gap
draw_block(
    ax_b,
    x,
    y_deploy,
    1.6,
    bh,
    "$v_1\\!=\\!F,\\; v_2\\!=\\!T$",
    C_STATE,
    ec=C_STATE_EDGE,
    lw=1.5,
    fs=10,
)
x += 1.6

# Arrow + results
ax_b.annotate(
    "",
    xy=(x + 0.35, y_deploy + bh / 2),
    xytext=(x + 0.05, y_deploy + bh / 2),
    arrowprops=dict(arrowstyle="-|>", lw=1.3),
)
rx = x + 0.45
ax_b.text(
    rx,
    y_deploy + bh / 2 + 0.18,
    "SSA: $\\checkmark$",
    ha="left",
    va="center",
    color=C_SSA,
    fontsize=12,
    weight="bold",
)
ax_b.text(
    rx,
    y_deploy + bh / 2 - 0.18,
    "Causal: $\\times$",
    ha="left",
    va="center",
    color=C_CAUSAL,
    fontsize=12,
    weight="bold",
)


# Output
out_dir = REPO_ROOT / "output" / "pictures"
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(
        out_dir / f"fig_transplant_rebuild.{ext}",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=300,
    )
plt.close(fig)
print("Saved fig_transplant_rebuild")
