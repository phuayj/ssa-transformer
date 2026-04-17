#!/usr/bin/env python3
"""Create publication-quality comparison plots for r^k benchmark eval results."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _parse_csv_ints(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError(f"Expected non-empty CSV integer list, got {text!r}")
    return values


def _parse_csv_paths(text: Optional[str]) -> List[Path]:
    if text is None:
        return []
    paths = [Path(x.strip()) for x in str(text).split(",") if x.strip()]
    return paths


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_eval_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "cells" not in payload:
        raise ValueError(f"Missing 'cells' in eval JSON: {path}")
    logger.info("loaded %s with %d cells", str(path), len(payload.get("cells", [])))
    return payload


def _index_cells(payload: dict) -> Dict[Tuple[int, int], dict]:
    indexed: Dict[Tuple[int, int], dict] = {}
    for cell in payload.get("cells", []):
        key = (int(cell["k"]), int(cell["difficulty"]))
        indexed[key] = cell
    return indexed


def _difficulty_r(payload: dict, difficulty: int) -> Optional[float]:
    r_map = payload.get("r_by_difficulty", {})
    if str(int(difficulty)) in r_map:
        return float(r_map[str(int(difficulty))])
    return None


def _predicted_value(cell: dict, r_value: Optional[float], k: int) -> Optional[float]:
    if "r_k_predicted" in cell:
        return float(cell["r_k_predicted"])
    if r_value is None:
        return None
    return float(r_value ** int(k))


def _make_subplots(n_panels: int) -> Tuple[plt.Figure, np.ndarray]:
    if n_panels <= 3:
        nrows, ncols = 1, n_panels
    else:
        nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.8 * nrows),
        sharey=True,
        constrained_layout=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr[n_panels:]:
        ax.set_visible(False)
    return fig, axes_arr


def _collect_k_values(
    difficulties: Sequence[int],
    transformer_idx: Optional[Dict[Tuple[int, int], dict]],
    oracle_idx: Optional[Dict[Tuple[int, int], dict]],
    corrupted_idxs: Sequence[Dict[Tuple[int, int], dict]],
) -> List[int]:
    ks = set()
    for d in difficulties:
        for idx in [transformer_idx, oracle_idx, *corrupted_idxs]:
            if idx is None:
                continue
            ks.update(int(k) for (k, diff) in idx.keys() if int(diff) == int(d))
    if not ks:
        return [1, 2, 4, 8, 16, 32]
    return sorted(ks)


def _plot_accuracy_panels(
    output_path: Path,
    difficulties: Sequence[int],
    ks: Sequence[int],
    transformer_payload: Optional[dict],
    oracle_payload: Optional[dict],
    corrupted_payloads: Sequence[dict],
    log_y: bool,
) -> None:
    transformer_idx = _index_cells(transformer_payload) if transformer_payload else None
    oracle_idx = _index_cells(oracle_payload) if oracle_payload else None
    corrupted_idxs = [_index_cells(p) for p in corrupted_payloads]

    fig, axes = _make_subplots(len(difficulties))
    for panel_i, difficulty in enumerate(difficulties):
        ax = axes[panel_i]
        xs = [int(k) for k in ks]

        title_r = None
        for payload in [transformer_payload, oracle_payload, *corrupted_payloads]:
            if payload is None:
                continue
            maybe_r = _difficulty_r(payload, int(difficulty))
            if maybe_r is not None:
                title_r = maybe_r
                break
        if title_r is None:
            ax.set_title(f"difficulty = {int(difficulty)}")
        else:
            ax.set_title(f"difficulty = {int(difficulty)} (r={title_r:.3f})")

        if transformer_idx is not None:
            assert transformer_payload is not None
            y_obs = []
            y_pred = []
            for k in xs:
                cell = transformer_idx.get((int(k), int(difficulty)))
                if cell is None:
                    y_obs.append(np.nan)
                    y_pred.append(np.nan)
                    continue
                y_obs.append(float(cell["positive_accuracy"]))
                y_pred.append(
                    _predicted_value(
                        cell,
                        _difficulty_r(transformer_payload, int(difficulty)),
                        int(k),
                    )
                )
            y_obs_arr = np.asarray(y_obs, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            if log_y:
                y_obs_arr = np.log(np.clip(y_obs_arr, 1e-8, 1.0))
                y_pred_arr = np.log(np.clip(y_pred_arr, 1e-8, 1.0))
            ax.plot(
                xs,
                y_obs_arr,
                color="#1f77b4",
                marker="o",
                linestyle="-",
                label="Transformer",
            )
            ax.plot(
                xs,
                y_pred_arr,
                color="#7f7f7f",
                linestyle="--",
                linewidth=1.5,
                label=r"$r^k$",
            )

        if oracle_idx is not None:
            y_oracle = []
            for k in xs:
                cell = oracle_idx.get((int(k), int(difficulty)))
                y_oracle.append(
                    np.nan if cell is None else float(cell["positive_accuracy"])
                )
            y_oracle_arr = np.asarray(y_oracle, dtype=float)
            if log_y:
                y_oracle_arr = np.log(np.clip(y_oracle_arr, 1e-8, 1.0))
            ax.plot(
                xs,
                y_oracle_arr,
                color="#2ca02c",
                marker="s",
                linestyle="-",
                label="Oracle MLP",
            )

        for c_idx, corr_idx in enumerate(corrupted_idxs):
            y_corr = []
            for k in xs:
                cell = corr_idx.get((int(k), int(difficulty)))
                y_corr.append(
                    np.nan if cell is None else float(cell["positive_accuracy"])
                )
            y_corr_arr = np.asarray(y_corr, dtype=float)
            if log_y:
                y_corr_arr = np.log(np.clip(y_corr_arr, 1e-8, 1.0))
            label = "Corrupted MLP" if c_idx == 0 else "Corrupted MLP (alt)"
            ax.plot(
                xs,
                y_corr_arr,
                color="#ff7f0e",
                marker="^",
                linestyle="-",
                linewidth=1.2,
                alpha=max(0.35, 0.95 - 0.2 * c_idx),
                label=label,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("k")
        if log_y:
            ax.grid(True, alpha=0.25)
        else:
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)

    ylabel = "log(positive accuracy)" if log_y else "Positive accuracy"
    fig.supylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False
        )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("saved figure %s", str(output_path))


def _plot_ratio(
    output_path: Path,
    difficulties: Sequence[int],
    ks: Sequence[int],
    transformer_payload: Optional[dict],
    oracle_payload: Optional[dict],
    corrupted_payloads: Sequence[dict],
) -> None:
    primary = transformer_payload
    if primary is None:
        primary = oracle_payload
    if primary is None and corrupted_payloads:
        primary = corrupted_payloads[0]
    if primary is None:
        logger.warning(
            "No input data available for ratio plot; skipping %s", str(output_path)
        )
        return

    indexed = _index_cells(primary)
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)

    for d in difficulties:
        ratios = []
        valid_x = []
        r_value = _difficulty_r(primary, int(d))
        for k in ks:
            cell = indexed.get((int(k), int(d)))
            if cell is None:
                continue
            pred = _predicted_value(cell, r_value, int(k))
            if pred is None:
                continue
            obs = float(cell["positive_accuracy"])
            valid_x.append(int(k))
            ratios.append(float(obs) / max(float(pred), 1e-8))
        if valid_x:
            ax.plot(
                valid_x, ratios, marker="o", linestyle="-", label=f"difficulty={int(d)}"
            )

    ax.axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(k) for k in ks])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("k")
    ax.set_ylabel("ratio = observed / predicted")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("saved figure %s", str(output_path))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot r^k benchmark model comparisons")
    parser.add_argument("--transformer_json", type=str, default=None)
    parser.add_argument("--oracle_json", type=str, default=None)
    parser.add_argument("--corrupted_jsons", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/rk_benchmark/figures/",
    )
    parser.add_argument("--difficulties", type=str, default="0,1,2,3,4")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    _setup_style()

    difficulties = _parse_csv_ints(args.difficulties)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformer_payload = None
    oracle_payload = None
    corrupted_payloads: List[dict] = []

    if args.transformer_json:
        transformer_payload = _load_eval_json(Path(args.transformer_json))
    if args.oracle_json:
        oracle_payload = _load_eval_json(Path(args.oracle_json))
    for path in _parse_csv_paths(args.corrupted_jsons):
        corrupted_payloads.append(_load_eval_json(path))

    if (
        transformer_payload is None
        and oracle_payload is None
        and not corrupted_payloads
    ):
        logger.warning("No input JSONs provided; nothing to plot.")
        return

    transformer_idx = _index_cells(transformer_payload) if transformer_payload else None
    oracle_idx = _index_cells(oracle_payload) if oracle_payload else None
    corrupted_idxs = [_index_cells(p) for p in corrupted_payloads]
    ks = _collect_k_values(difficulties, transformer_idx, oracle_idx, corrupted_idxs)
    logger.info("plotting figures for difficulties=%s k_values=%s", difficulties, ks)

    _plot_accuracy_panels(
        output_path=output_dir / "rk_positive_accuracy.pdf",
        difficulties=difficulties,
        ks=ks,
        transformer_payload=transformer_payload,
        oracle_payload=oracle_payload,
        corrupted_payloads=corrupted_payloads,
        log_y=False,
    )
    _plot_accuracy_panels(
        output_path=output_dir / "rk_log_linear.pdf",
        difficulties=difficulties,
        ks=ks,
        transformer_payload=transformer_payload,
        oracle_payload=oracle_payload,
        corrupted_payloads=corrupted_payloads,
        log_y=True,
    )
    _plot_ratio(
        output_path=output_dir / "rk_ratio.pdf",
        difficulties=difficulties,
        ks=ks,
        transformer_payload=transformer_payload,
        oracle_payload=oracle_payload,
        corrupted_payloads=corrupted_payloads,
    )
    logger.info("plotting complete")


if __name__ == "__main__":
    main()
