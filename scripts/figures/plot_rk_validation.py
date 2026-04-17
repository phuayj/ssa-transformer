import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.framealpha": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot r^k validation figures")
    parser.add_argument(
        "--transformer_json",
        type=str,
        default="experiments/rk_benchmark/eval_transformer_10k_factorized/rk_eval_results.json",
    )
    parser.add_argument(
        "--oracle_json",
        type=str,
        default="experiments/rk_benchmark/eval_oracle_mlp/rk_eval_results.json",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/rk_benchmark/figures/"
    )
    return parser.parse_args()


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Data file not found: {filepath}")
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def build_cell_dict(data):
    if not data or "cells" not in data:
        return {}
    cells_dict = {}
    for cell in data["cells"]:
        k = cell["k"]
        d = cell["difficulty"]
        if d not in cells_dict:
            cells_dict[d] = {}
        cells_dict[d][k] = cell
    return cells_dict


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trans_data = load_data(args.transformer_json)
    oracle_data = load_data(args.oracle_json)

    trans_cells = build_cell_dict(trans_data)
    oracle_cells = build_cell_dict(oracle_data)

    if not trans_cells:
        print("Error: Transformer data missing or invalid.")
        return

    blue = "#1f77b4"
    red = "#d62728"
    green = "#2ca02c"
    grey = "#7f7f7f"
    light_grey = "#b0b0b0"

    # ------------------
    # Figure 1: rk_validation_main.pdf
    # ------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)

    # Left Panel: diff=4
    if 4 in trans_cells:
        k_values = sorted(trans_cells[4].keys())
        trans_full_acc = [
            trans_cells[4][k].get(
                "full_model_positive_acc", trans_cells[4][k].get("positive_accuracy", 0)
            )
            for k in k_values
        ]
        trans_fact_acc = [
            trans_cells[4][k].get("factorized_conjunction_acc", 0) for k in k_values
        ]
        trans_rk_pred = [
            trans_cells[4][k].get(
                "factorized_r_k_predicted", trans_cells[4][k].get("r_k_predicted", 0)
            )
            for k in k_values
        ]

        ax1.plot(
            k_values,
            trans_full_acc,
            marker="o",
            color=blue,
            linestyle="-",
            label="Transformer (full attention)",
        )
        ax1.plot(
            k_values,
            trans_fact_acc,
            marker="^",
            color=red,
            linestyle="-",
            label="Transformer (factorized)",
        )
        ax1.plot(
            k_values,
            trans_rk_pred,
            marker="",
            color=grey,
            linestyle="--",
            label="$r^k$ prediction",
        )

        if oracle_cells and 4 in oracle_cells:
            oracle_k = sorted(oracle_cells[4].keys())
            oracle_acc = [oracle_cells[4][k]["positive_accuracy"] for k in oracle_k]
            ax1.plot(
                oracle_k,
                oracle_acc,
                marker="s",
                color=green,
                linestyle="-",
                label="Oracle MLP",
            )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(k_values)
    ax1.set_xticklabels([str(k) for k in k_values])
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Positive accuracy")
    ax1.set_title("Positive accuracy vs k")
    ax1.legend(loc="upper right")
    ax1.text(
        -0.15,
        1.05,
        "(a)",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Right Panel: factorized ratio
    colors_diff = {0: blue, 2: "#ff7f0e", 4: green}
    markers_diff = {0: "o", 2: "^", 4: "s"}

    ax2.axhspan(0.9, 1.1, color="lightgrey", alpha=0.5, zorder=0)
    ax2.axhline(1.0, color="black", linestyle="--", label="perfect r^k", zorder=1)

    for diff in [0, 2, 4]:
        if diff in trans_cells:
            k_vals = sorted(trans_cells[diff].keys())
            valid_k = []
            ratios = []
            for k in k_vals:
                cell = trans_cells[diff][k]
                fact_acc = cell.get("factorized_conjunction_acc", 0)
                if fact_acc > 0:
                    valid_k.append(k)
                    ratio = cell.get("factorized_ratio", cell.get("ratio", 1.0))
                    ratios.append(ratio)
            if valid_k:
                # Add label conditionally
                label = f"Difficulty {diff}" if diff in [0, 2, 4] else None
                ax2.plot(
                    valid_k,
                    ratios,
                    marker=markers_diff[diff],
                    color=colors_diff[diff],
                    linestyle="-",
                    label=label,
                    zorder=2,
                )

    ax2.set_xscale("log", base=2)
    ax2.set_xticks(k_values)
    ax2.set_xticklabels([str(k) for k in k_values])
    ax2.set_ylim(0.5, 1.5)
    ax2.set_xlabel("k")
    ax2.set_ylabel("Ratio (observed/predicted)")
    ax2.set_title("Factorized ratio vs k")
    ax2.legend(loc="upper right")
    ax2.text(
        -0.15,
        1.05,
        "(b)",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    fig1.savefig(os.path.join(args.output_dir, "rk_validation_main.pdf"))
    plt.close(fig1)

    # ------------------
    # Figure 2: rk_validation_log_linear.pdf
    # ------------------
    fig2, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)

    def get_log_arrays(diff):
        k_vals = sorted(trans_cells[diff].keys())

        valid_fact_k = [
            k
            for k in k_vals
            if trans_cells[diff][k].get("factorized_conjunction_acc", 0) > 0
        ]
        fact_acc = [
            np.log10(trans_cells[diff][k]["factorized_conjunction_acc"])
            for k in valid_fact_k
        ]

        valid_pred_k = [
            k
            for k in k_vals
            if trans_cells[diff][k].get(
                "factorized_r_k_predicted", trans_cells[diff][k].get("r_k_predicted", 0)
            )
            > 0
        ]
        pred_acc = [
            np.log10(
                trans_cells[diff][k].get(
                    "factorized_r_k_predicted",
                    trans_cells[diff][k].get("r_k_predicted", 0),
                )
            )
            for k in valid_pred_k
        ]

        valid_full_k = [
            k
            for k in k_vals
            if trans_cells[diff][k].get(
                "full_model_positive_acc",
                trans_cells[diff][k].get("positive_accuracy", 0),
            )
            > 0
        ]
        full_acc = [
            np.log10(
                trans_cells[diff][k].get(
                    "full_model_positive_acc",
                    trans_cells[diff][k].get("positive_accuracy", 0),
                )
            )
            for k in valid_full_k
        ]

        return valid_fact_k, fact_acc, valid_pred_k, pred_acc, valid_full_k, full_acc

    if 4 in trans_cells:
        vk_fact, fact, vk_pred, pred, vk_full, full = get_log_arrays(4)
        if vk_fact:
            ax.plot(
                vk_fact,
                fact,
                marker="^",
                color=red,
                linestyle="-",
                label="Fact. (diff=4)",
            )
        if vk_pred:
            ax.plot(
                vk_pred,
                pred,
                marker="",
                color=grey,
                linestyle="--",
                label="$r^k$ (diff=4)",
            )
        if vk_full:
            ax.plot(
                vk_full,
                full,
                marker="^",
                color=red,
                linestyle=":",
                markerfacecolor="none",
                label="Full (diff=4)",
            )

    if 0 in trans_cells:
        vk_fact, fact, vk_pred, pred, vk_full, full = get_log_arrays(0)
        if vk_fact:
            ax.plot(
                vk_fact,
                fact,
                marker="o",
                color=blue,
                linestyle="-",
                label="Fact. (diff=0)",
            )
        if vk_pred:
            ax.plot(
                vk_pred,
                pred,
                marker="",
                color=light_grey,
                linestyle="--",
                label="$r^k$ (diff=0)",
            )
        if vk_full:
            ax.plot(
                vk_full,
                full,
                marker="o",
                color=blue,
                linestyle=":",
                markerfacecolor="none",
                label="Full (diff=0)",
            )

    ax.set_xlabel("k")
    ax.set_ylabel("log$_{10}$(Positive accuracy)")
    ax.set_title("Log-linear diagnostic")
    ax.legend(loc="lower left", framealpha=0.9)
    fig2.savefig(os.path.join(args.output_dir, "rk_validation_log_linear.pdf"))
    plt.close(fig2)


if __name__ == "__main__":
    main()
