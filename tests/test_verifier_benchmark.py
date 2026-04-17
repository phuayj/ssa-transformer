from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_verifier_benchmark_sat_smoke(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from sat.interleaved_tokenizer import SATInterleavedTokenizer
    from universal.ssa_decoder import SSASlotDecoder

    bank_path = tmp_path / "sat_n10_bank.json"
    ckpt_path = tmp_path / "tiny_sat_checkpoint.pt"
    out_path = tmp_path / "eval.json"

    model = SSASlotDecoder(
        vocab_size=SATInterleavedTokenizer.VOCAB_SIZE,
        d_model=16,
        n_layers=1,
        n_heads=2,
        max_seq_len=768,
        n_slots=2,
        dropout=0.0,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": SATInterleavedTokenizer.VOCAB_SIZE,
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 2,
                "n_slots": 2,
                "max_seq_len": 768,
                "dropout": 0.0,
                "mask_mode": "full_causal",
            },
        },
        ckpt_path,
    )

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "data_generation" / "build_verifier_probe_bank.py"),
            "--domain",
            "sat",
            "--num_vars",
            "10",
            "--alpha",
            "3.5",
            "--n_instances",
            "2",
            "--n_traces_per_instance",
            "2",
            "--max_seq_len",
            "768",
            "--seed",
            "7",
            "--min_histories_per_state",
            "2",
            "--max_histories_per_state",
            "2",
            "--output_path",
            str(bank_path),
        ],
        cwd=repo,
        check=True,
    )
    bank = json.loads(bank_path.read_text())
    assert "config" in bank and "stats" in bank and "probes" in bank

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "evaluation" / "eval_verifier_benchmark.py"),
            "--probe_bank",
            str(bank_path),
            "--checkpoint",
            str(ckpt_path),
            "--architecture",
            "causal",
            "--protocol",
            "cumulative",
            "--device",
            "cpu",
            "--bootstrap_iters",
            "5",
            "--bootstrap_seed",
            "11",
            "--output_path",
            str(out_path),
        ],
        cwd=repo,
        check=True,
    )
    payload = json.loads(out_path.read_text())
    assert set(payload) >= {"config", "panel_a", "panel_b", "calibration"}
    assert set(payload["panel_a"]) >= {
        "alpha_v_overall",
        "alpha_v_by_depth_quartile",
        "alpha_v_by_size_quartile",
        "beta_overall",
        "auroc",
        "auprc",
    }
    assert set(payload["panel_b"]) >= {"argmax_agreement", "mean_symmetric_kl"}
    assert set(payload["calibration"]) >= {"ece_15bins", "brier", "reliability_curve"}
