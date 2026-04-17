#!/usr/bin/env python3
"""Evaluate verifier-only state-equivalence probe banks."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(np.asarray(xs, dtype=float))) if xs else 0.0


def _ci(xs: Sequence[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    arr = np.asarray(xs, dtype=float)
    return {"mean": float(np.mean(arr)), "ci_lo": float(np.quantile(arr, 0.025)), "ci_hi": float(np.quantile(arr, 0.975))}


def _binary_kl(p: float, q: float) -> float:
    eps = 1e-12
    p = min(max(float(p), eps), 1.0 - eps); q = min(max(float(q), eps), 1.0 - eps)
    return float((1 - p) * np.log((1 - p) / (1 - q)) + p * np.log(p / q))


def _roc_auc(y: Sequence[int], s: Sequence[float]) -> float:
    pos = [float(v) for yy, v in zip(y, s) if int(yy) == 1]
    neg = [float(v) for yy, v in zip(y, s) if int(yy) == 0]
    if not pos or not neg:
        return 0.0
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return float(wins / (len(pos) * len(neg)))


def _average_precision(y: Sequence[int], s: Sequence[float]) -> float:
    pairs = sorted(zip(s, y), key=lambda t: -float(t[0]))
    total_pos = sum(int(yy) == 1 for yy in y)
    if total_pos == 0:
        return 0.0
    hit = 0; acc = 0.0
    for rank, (_score, yy) in enumerate(pairs, start=1):
        if int(yy) == 1:
            hit += 1; acc += hit / rank
    return float(acc / total_pos)


def _ece_reliability(y: Sequence[int], s: Sequence[float], n_bins: int = 15) -> Tuple[float, List[Dict[str, Any]]]:
    if not y:
        return 0.0, []
    order = np.argsort(np.asarray(s, dtype=float))
    bins = np.array_split(order, min(int(n_bins), len(order)))
    curve: List[Dict[str, Any]] = []
    ece = 0.0
    for idx in bins:
        preds = np.asarray([s[int(i)] for i in idx], dtype=float)
        acts = np.asarray([y[int(i)] for i in idx], dtype=float)
        if len(preds) == 0:
            continue
        pp, pa = float(np.mean(preds)), float(np.mean(acts))
        ece += float(len(preds) / len(y)) * abs(pp - pa)
        curve.append({"bin_lo": float(np.min(preds)), "bin_hi": float(np.max(preds)), "p_predicted": pp, "p_actual": pa, "n": int(len(preds))})
    return float(ece), curve


def _mask_mode_for_arch(arch: str, checkpoint_mode: str) -> str:
    arch = str(arch)
    if arch in {"ssa", "contrastive_ssa"}:
        return "selective_ssa"
    if arch == "current_block_only":
        return "local_block_only"
    if arch in {"causal", "contrastive", "contrastive_causal", "block_dropout", "sliding_window", "null_history", "history_transplant", "block_dropout_p050"}:
        return "full_causal"
    if arch == "lstm":
        return str(checkpoint_mode or "full_causal")
    return str(checkpoint_mode or "full_causal")


def _forward(model: torch.nn.Module, tokens: Sequence[int], blocks: Sequence[int], device: torch.device, *, domain: str, is_gc_ssa: bool, mask_mode: str, vocab_size: int) -> torch.Tensor:
    import torch

    x = torch.tensor([list(map(int, tokens))], dtype=torch.long, device=device)
    b = torch.tensor([list(map(int, blocks))], dtype=torch.long, device=device)
    with torch.no_grad():
        if str(domain) == "gc" and not bool(is_gc_ssa):
            lm, _ = model(x)
        else:
            lm, _ = model(x, block_ids=b, mask_mode=str(mask_mode))
    return lm[0, -1, : int(vocab_size)]


def _rebuilt(history: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    tokens = [int(x) for x in history["prefix_tokens"]]
    blocks = [int(x) for x in history["prefix_block_ids"]]
    start = int(history["current_block_start"])
    hist_len = int(history["history_token_length"])
    prefix_end = int(start - hist_len)
    new_tokens = tokens[:prefix_end] + tokens[start:]
    new_blocks = blocks[:prefix_end] + [1] * (len(tokens) - start)
    return new_tokens, new_blocks


def _continue_tokens(domain: str, allowed: Sequence[Dict[str, int]], vocab_size: int) -> List[int]:
    if domain == "sat":
        from sat.interleaved_tokenizer import SATInterleavedTokenizer

        tok = SATInterleavedTokenizer()
        return sorted({int(tok.var_token(int(a["var"]))) for a in allowed if int(tok.var_token(int(a["var"]))) < int(vocab_size)})
    from universal.cdcl_tokenizer import CDCLTokenizer

    tok = CDCLTokenizer()
    return sorted({int(tok.node_token(int(a["var"]))) for a in allowed if int(tok.node_token(int(a["var"]))) < int(vocab_size)})


def _cf_token(domain: str, config: Dict[str, Any], vocab_size: int) -> int:
    if domain == "sat":
        from sat.interleaved_tokenizer import SATInterleavedTokenizer

        return int(SATInterleavedTokenizer.CONFLICT)
    from graph_coloring.transplant_trace import TokenMapper

    return int(TokenMapper(vocab_size=int(vocab_size)).CF)


def _p_bt(logits: torch.Tensor, cf: int, cont: Sequence[int]) -> float:
    import torch

    ids = [int(cf)] + [int(t) for t in cont if int(t) != int(cf)]
    picked = logits[torch.tensor(ids, dtype=torch.long, device=logits.device)]
    probs = torch.softmax(picked.float(), dim=0)
    return float(probs[0].item())


def _compute_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    y = [int(r["y"]) for r in rows]; s = [float(r["p_bt_mean"]) for r in rows]
    pred = [float(v) > 0.5 for v in s]
    alpha = _mean([1.0 if pred[i] else 0.0 for i, yy in enumerate(y) if yy == 0])
    beta = _mean([1.0 if not pred[i] else 0.0 for i, yy in enumerate(y) if yy == 1])
    agr, kls = [], []
    for r in rows:
        vals = [float(v) for v in r["p_bt_per_history"]]
        for a, b in combinations(vals, 2):
            agr.append(1.0 if (a > 0.5) == (b > 0.5) else 0.0)
            kls.append(0.5 * (_binary_kl(a, b) + _binary_kl(b, a)))
    return {"alpha_v": float(alpha), "beta": float(beta), "auroc": _roc_auc(y, s), "auprc": _average_precision(y, s), "argmax_agreement": _mean(agr), "mean_symmetric_kl": _mean(kls)}


def _bootstrap(rows: Sequence[Dict[str, Any]], iters: int, seed: int) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {k: _ci([]) for k in ["alpha_v", "beta", "auroc", "auprc", "argmax_agreement", "mean_symmetric_kl"]}
    rng = random.Random(int(seed)); n = len(rows)
    samples: Dict[str, List[float]] = defaultdict_list()
    for _ in range(int(iters)):
        sub = [rows[rng.randrange(n)] for _ in range(n)]
        m = _compute_metrics(sub)
        for k, v in m.items():
            samples[k].append(float(v))
    return {k: _ci(v) for k, v in samples.items()}


def defaultdict_list() -> Dict[str, List[float]]:
    from collections import defaultdict
    return defaultdict(list)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe_bank", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--architecture", choices=["ssa", "causal", "lstm", "factor_gnn", "current_block_only", "contrastive", "contrastive_ssa", "contrastive_causal", "block_dropout", "block_dropout_p050", "sliding_window", "null_history", "history_transplant"], required=True)
    ap.add_argument("--protocol", choices=["cumulative", "state_rebuilt"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--bootstrap_seed", type=int, default=42)
    ap.add_argument("--store_per_probe", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_path); out.parent.mkdir(parents=True, exist_ok=True)
    if args.architecture == "factor_gnn":
        out.write_text(json.dumps({"skipped": True, "reason": "factor GNN consumes (state, action) pairs natively, requires custom adapter", "config": vars(args)}, indent=2), encoding="utf-8")
        return

    bank = json.loads(Path(args.probe_bank).read_text(encoding="utf-8"))
    domain = str(bank.get("config", {}).get("domain", "sat"))
    import torch

    device = torch.device(str(args.device))
    if domain == "sat":
        from sat.transplant_trace import _load_checkpoint as _load_sat_checkpoint

        model, meta = _load_sat_checkpoint(Path(args.checkpoint), device, int(bank.get("config", {}).get("max_seq_len", 4096)))
        is_gc_ssa = False
    else:
        from graph_coloring.transplant_trace import _load_checkpoint as _load_gc_checkpoint

        model, meta, is_gc_ssa = _load_gc_checkpoint(Path(args.checkpoint), device)
    vocab_size = int(meta.get("vocab_size", 0))
    checkpoint_mode = str(meta.get("mask_mode", meta.get("attention_mode", "")))
    mask_mode = _mask_mode_for_arch(str(args.architecture), checkpoint_mode)
    if args.architecture in {"ssa", "contrastive_ssa"} and checkpoint_mode not in {"selective_ssa", "ssa"}:
        logger.warning("architecture=%s but checkpoint mode=%s", args.architecture, checkpoint_mode)
    if args.architecture in {"causal", "contrastive", "contrastive_causal"} and checkpoint_mode not in {"full_causal", "causal"}:
        logger.warning("architecture=%s but checkpoint mode=%s", args.architecture, checkpoint_mode)
    cf = _cf_token(domain, bank.get("config", {}), int(vocab_size))
    rows: List[Dict[str, Any]] = []
    for pi, probe in enumerate(bank.get("probes", [])):
        cont = _continue_tokens(domain, probe.get("allowed_continue_actions", []), int(vocab_size))
        vals: List[float] = []
        for hist in probe.get("histories", []):
            tokens, blocks = (hist["prefix_tokens"], hist["prefix_block_ids"]) if args.protocol == "cumulative" else _rebuilt(hist)
            logits = _forward(model, tokens, blocks, device, domain=domain, is_gc_ssa=bool(is_gc_ssa), mask_mode=str(mask_mode), vocab_size=int(vocab_size))
            vals.append(_p_bt(logits, int(cf), cont))
        if pi < 3:
            logger.info("sample probe=%s y=%s p_bt=%s", probe.get("probe_id"), probe.get("label"), [round(v, 4) for v in vals])
        rows.append({"probe_id": probe.get("probe_id"), "label": probe.get("label"), "y": 1 if probe.get("label") == "exposed_conflict" else 0, "depth_quartile": int(probe.get("depth_quartile", 1)), "size_quartile": int(probe.get("size_quartile", 1)), "p_bt_mean": _mean(vals), "p_bt_per_history": vals})

    if args.architecture in {"ssa", "contrastive_ssa"}:
        bad = sum(1 for r in rows for a, b in combinations(r["p_bt_per_history"], 2) if 0.5 * (_binary_kl(a, b) + _binary_kl(b, a)) > 1e-3)
        if bad:
            logger.warning("SSA invariance sanity check found %d within-probe history pairs with KL > 1e-3", bad)
    metrics = _compute_metrics(rows); boot = _bootstrap(rows, int(args.bootstrap_iters), int(args.bootstrap_seed))
    y = [int(r["y"]) for r in rows]; s = [float(r["p_bt_mean"]) for r in rows]
    ece, curve = _ece_reliability(y, s, 15)
    brier = _mean([(float(si) - int(yi)) ** 2 for yi, si in zip(y, s)])
    def pack(name: str) -> Dict[str, float]:
        return {"mean": float(metrics[name]), "ci_lo": float(boot.get(name, {}).get("ci_lo", metrics[name])), "ci_hi": float(boot.get(name, {}).get("ci_hi", metrics[name]))}
    def byq(name: str, key: str) -> List[float]:
        outv = []
        for q in range(1, 5):
            sub = [r for r in rows if int(r.get(key, 1)) == q]
            outv.append(float(_compute_metrics(sub)[name]) if sub else 0.0)
        return outv
    payload: Dict[str, Any] = {"config": {**vars(args), "checkpoint_config": meta.get("config", {}), "domain": domain, "mask_mode_used": mask_mode}, "panel_a": {"alpha_v_overall": pack("alpha_v"), "alpha_v_by_depth_quartile": byq("alpha_v", "depth_quartile"), "alpha_v_by_size_quartile": byq("alpha_v", "size_quartile"), "beta_overall": pack("beta"), "auroc": pack("auroc"), "auprc": pack("auprc")}, "panel_b": {"argmax_agreement": pack("argmax_agreement"), "mean_symmetric_kl": pack("mean_symmetric_kl")}, "calibration": {"ece_15bins": float(ece), "brier": float(brier), "reliability_curve": curve}}
    if args.store_per_probe:
        payload["per_probe"] = [{"probe_id": r["probe_id"], "label": r["label"], "p_bt_mean": r["p_bt_mean"], "p_bt_per_history": r["p_bt_per_history"]} for r in rows]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s probes=%d alpha=%.4f beta=%.4f auroc=%.4f", out, len(rows), metrics["alpha_v"], metrics["beta"], metrics["auroc"])


if __name__ == "__main__":
    main()
