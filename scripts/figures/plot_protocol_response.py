import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pathlib import Path

# Set design requirements
matplotlib.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.titlesize": 10,
        "legend.fontsize": 8,
    }
)

# Colors
color_ssa = "#007f86"  # Teal
color_causal = "#d63a3a"  # Red

# Data
# SAT n=50
ssa_sat50 = np.array(
    [
        [31.5, 97.0, 90.0],
        [27.0, 93.5, 90.0],
        [29.5, 92.0, 90.0],
        [27.0, 91.0, 90.0],
        [29.5, 93.5, 87.5],
    ]
)
causal_sat50 = np.array(
    [
        [26.5, 4.5, 4.0],
        [5.0, 5.0, 4.0],
        [26.0, 6.0, 4.0],
        [4.0, 3.0, 4.0],
        [21.0, 5.5, 4.0],
    ]
)

# SAT n=75
ssa_sat75 = np.array([12.0, 10.0, 11.5, 9.0, 11.5, 11.5, 12.5, 13.5])
causal_sat75 = np.array([11.0, 10.0, 1.5, 0.5, 0.5, 8.0, 8.5, 11.0])

# Parsing
ssa_parsing = np.array(
    [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]
)
causal_parsing = np.array(
    [[99.7, 100.0], [99.7, 96.0], [99.7, 21.0], [99.7, 21.0], [99.0, 59.7]]
)


def plot_errorbars(ax, data, x_positions, color, label, offset=0):
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0, ddof=1)

    x_pos = np.array(x_positions) + offset

    ax.errorbar(
        x_pos,
        mean_data,
        yerr=std_data,
        fmt="-o",
        color=color,
        capsize=3,
        linewidth=2,
        markersize=7,
        label=label,
        zorder=5,
    )


def plot_strip(ax, data, x_position, color):
    num_seeds = len(data)
    # Use linspace for clean spread, and shuffle to avoid ordering correlation
    np.random.seed(42)
    jitter = np.linspace(-0.1, 0.1, num_seeds)
    np.random.shuffle(jitter)

    x_jittered = x_position + jitter
    ax.scatter(x_jittered, data, s=15, color=color, alpha=0.4, zorder=3)

    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    ax.errorbar(
        x_position,
        mean_data,
        yerr=std_data,
        fmt="o",
        color=color,
        capsize=3,
        linewidth=2,
        markersize=7,
        zorder=5,
    )


fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)

# Common styling
for ax in axes:
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Add collapse band
    ax.axhspan(-5, 5, color="gray", alpha=0.15, zorder=0)

# Y-axis label only on the left
axes[0].set_ylabel("Solve rate (%)")
axes[0].text(
    0.5,
    2.5,
    "Collapsed",
    color="dimgray",
    fontsize=7,
    ha="center",
    va="center",
    zorder=1,
)

# Panel 1: SAT n=50
x_sat50 = [0, 1, 2]
plot_errorbars(axes[0], ssa_sat50, x_sat50, color_ssa, "SSA", offset=-0.05)
plot_errorbars(axes[0], causal_sat50, x_sat50, color_causal, "Causal", offset=0.05)
axes[0].set_xticks(x_sat50)
axes[0].set_xticklabels(["Cumulative", "State-rebuilt", "Random var."])
axes[0].set_title("SAT $n=50$")

# Panel 2: SAT n=75
plot_strip(axes[1], ssa_sat75, 0, color_ssa)
plot_strip(axes[1], causal_sat75, 1, color_causal)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["SSA", "Causal"])
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_title("SAT $n=75$")
axes[1].set_xlabel("Cumulative")

# Panel 3: Parsing
x_parsing = np.array([0, 1])
bar_width = 0.3
offset = bar_width / 2 + 0.025

ssa_mean = np.mean(ssa_parsing, axis=0)
ssa_std = np.std(ssa_parsing, axis=0, ddof=1)
causal_mean = np.mean(causal_parsing, axis=0)
causal_std = np.std(causal_parsing, axis=0, ddof=1)

axes[2].bar(
    x_parsing - offset,
    ssa_mean,
    width=bar_width,
    color=color_ssa,
    edgecolor="#00595e",
    linewidth=1.2,
    yerr=ssa_std,
    capsize=3,
    error_kw={"elinewidth": 1.5, "capthick": 1.5, "zorder": 5},
    zorder=3,
)
axes[2].bar(
    x_parsing + offset,
    causal_mean,
    width=bar_width,
    color=color_causal,
    edgecolor="#962828",
    linewidth=1.2,
    yerr=causal_std,
    capsize=3,
    error_kw={"elinewidth": 1.5, "capthick": 1.5, "zorder": 5},
    zorder=3,
)

axes[2].set_xticks(x_parsing)
axes[2].set_xticklabels(["Cumulative", "State-rebuilt"])
axes[2].set_xlim(-0.5, 1.5)
axes[2].set_title("Parsing")

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    frameon=False,
)

plt.tight_layout()
# Adjust top to make room for legend
plt.subplots_adjust(top=0.85)

# Save figure
output_path = Path("protocol_response.pdf")
plt.savefig(
    output_path,
    bbox_inches="tight",
    dpi=300,
)
print(f"Plot saved to {output_path}")
