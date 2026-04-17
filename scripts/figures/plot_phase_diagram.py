#!/usr/bin/env python3
"""Plot phase-diagram heatmaps from SAT/GC sweep JSON outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_axes(payload: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    policy = sorted({float(row["policy_noise"]) for row in payload["grid"]})
    fp_values = sorted(
        {
            float(row["fp_rate"])
            for row in payload["grid"]
            if bool(row.get("verification_enabled")) and row.get("fp_rate") is not None
        }
    )
    labels = ["no_verification"] + [f"fp={v:.2f}" for v in fp_values]
    return policy, labels


def _build_matrix(payload: Dict[str, Any]) -> Tuple[np.ndarray, List[float], List[str]]:
    policy_values, verifier_labels = _collect_axes(payload)
    matrix = np.full(
        (len(verifier_labels), len(policy_values)), np.nan, dtype=np.float64
    )

    row_idx = {name: i for i, name in enumerate(verifier_labels)}
    col_idx = {float(v): i for i, v in enumerate(policy_values)}

    for row in payload["grid"]:
        p = float(row["policy_noise"])
        if bool(row.get("verification_enabled")):
            verifier_key = f"fp={float(row['fp_rate']):.2f}"
        else:
            verifier_key = "no_verification"
        matrix[row_idx[verifier_key], col_idx[p]] = float(
            row["aggregate"]["solve_rate_mean"]
        )

    return matrix, policy_values, verifier_labels


def _plot_single(payload: Dict[str, Any], title: str, output_path: Path) -> None:
    matrix, policy_values, verifier_labels = _build_matrix(payload)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Policy noise ε_p")
    ax.set_ylabel("Verifier setting")
    ax.set_xticks(np.arange(len(policy_values)))
    ax.set_xticklabels([f"{x:.2f}" for x in policy_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(verifier_labels)))
    ax.set_yticklabels(verifier_labels)

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if not np.isnan(matrix[r, c]):
                ax.text(
                    c,
                    r,
                    f"{matrix[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Solve rate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SAT/GC phase-diagram heatmaps")
    parser.add_argument(
        "--sat_json",
        type=str,
        default="experiments/phase_diagram/sat/phase_diagram_sat.json",
        required=False,
    )
    parser.add_argument(
        "--gc_json",
        type=str,
        default="experiments/phase_diagram/gc/phase_diagram_gc.json",
        required=False,
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/phase_diagram/figures"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sat_json_path = Path(args.sat_json)
    if sat_json_path.exists():
        sat_payload = _load_json(sat_json_path)
        _plot_single(
            sat_payload, "SAT Applicability Frontier", out_dir / "phase_diagram_sat.png"
        )
    else:
        logging.warning("SAT JSON not found, skipping SAT plot: %s", sat_json_path)

    gc_json_path = Path(args.gc_json)
    if gc_json_path.exists():
        gc_payload = _load_json(gc_json_path)
        _plot_single(
            gc_payload, "GC Applicability Frontier", out_dir / "phase_diagram_gc.png"
        )
    else:
        logging.warning("GC JSON not found, skipping GC plot: %s", gc_json_path)


if __name__ == "__main__":
    main()
