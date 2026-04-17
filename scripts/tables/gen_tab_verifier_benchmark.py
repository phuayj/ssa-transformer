#!/usr/bin/env python3
"""Aggregate verifier-only benchmark JSONs into main-text and appendix tables."""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Mapping, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

EXPECTED_SEEDS = 5
MISSING = "---"

ARCH_DISPLAY: Dict[str, str] = {
    "ssa": "SSA (selective)",
    "causal": "Causal",
    "current_block_only": "Current-block-only",
    "block_dropout": r"Block dropout $p=0.5$",
    "sliding_window": r"Sliding window $k=3$",
    "null_history": "Null history",
    "lstm": "LSTM",
    "contrastive": "Contrastive (causal)",
    "contrastive_ssa": "Contrastive (SSA)",
    "factor_gnn": r"Factor GNN $\dagger$",
    "history_transplant": "Hist. transplant",
    "state_only_100k": "Causal (state-only)",
}

ROW_ORDER: Tuple[str, ...] = (
    "ssa",
    "causal",
    "current_block_only",
    "block_dropout",
    "lstm",
    "contrastive",
)

DOMAIN_DISPLAY: Dict[str, str] = {
    "sat50": r"SAT $n=50$",
    "sat": r"SAT $n=50$",
    "gc30": r"GC $n=30$",
    "gc": r"GC $n=30$",
}

PROTOCOL_DISPLAY: Dict[str, str] = {
    "cumulative": "cumulative",
    "state_rebuilt": "state-rebuilt",
}

PROTOCOL_ALIASES: Dict[str, str] = {
    "cum": "cumulative",
    "cumulative": "cumulative",
    "sr": "state_rebuilt",
    "state_rebuilt": "state_rebuilt",
    "state-rebuilt": "state_rebuilt",
    "state_rebuild": "state_rebuilt",
    "state-rebuild": "state_rebuilt",
}

PANEL_A_METRICS: Tuple[Tuple[str, str], ...] = (
    ("alpha_v_overall", r"$\alpha_v\downarrow$"),
    ("beta_overall", r"$\beta\downarrow$"),
    ("auroc", r"AUROC$\uparrow$"),
)

PANEL_B_METRICS: Tuple[Tuple[str, str], ...] = (
    ("argmax_agreement", r"Argmax$\uparrow$"),
    ("mean_symmetric_kl", r"Sym KL$\downarrow$"),
)

APPENDIX_METRICS: Tuple[Tuple[str, str], ...] = (
    ("alpha_v_overall", r"$\alpha_v \pm s$"),
    ("beta_overall", r"$\beta \pm s$"),
    ("auroc", r"AUROC $\pm s$"),
    ("auprc", r"AUPRC $\pm s$"),
    ("ece_15bins", "ECE"),
    ("brier", "Brier"),
)


@dataclass(frozen=True)
class EvalRecord:
    path: Path
    arch: str
    domain: str
    seed: int
    protocol: str
    payload: Mapping[str, Any]
    canonical_filename: bool


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    n: int
    seeds: Tuple[int, ...]
    values: Tuple[float, ...]


def _normalize_arch(raw: str) -> str:
    arch = raw.strip().lower().replace("-", "_")
    if arch == "selective_ssa":
        return "ssa"
    if arch == "full_causal":
        return "causal"
    return arch


def _normalize_domain(raw: str) -> str:
    domain = raw.strip().lower().replace("-", "_")
    if domain == "sat":
        return "sat50"
    if domain == "gc":
        return "gc30"
    return domain


def _normalize_protocol(raw: str) -> str:
    protocol = raw.strip().lower().replace("-", "_")
    return PROTOCOL_ALIASES.get(protocol, protocol)


def _extract_seed_from_text(text: str) -> int | None:
    match = re.search(r"seed(\d+)", text)
    return int(match.group(1)) if match else None


def _parse_filename(path: Path) -> tuple[str | None, str | None, int | None, str | None, bool]:
    stem = path.stem
    match = re.match(r"(?P<arch>.+)_(?P<domain>sat50|gc30|sat|gc)_seed(?P<seed>\d+)_(?P<protocol>.+)$", stem)
    if not match:
        return None, None, None, None, False
    protocol_raw = match.group("protocol")
    protocol = _normalize_protocol(protocol_raw)
    return (
        _normalize_arch(match.group("arch")),
        _normalize_domain(match.group("domain")),
        int(match.group("seed")),
        protocol,
        protocol_raw in {"cumulative", "state_rebuilt"},
    )


def _record_from_path(path: Path) -> EvalRecord | None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("skipped"):
        LOGGER.info("Skipping %s because skipped=true", path.name)
        return None

    file_arch, file_domain, file_seed, file_protocol, canonical_filename = _parse_filename(path)
    cfg = payload.get("config", {})
    arch = file_arch or _normalize_arch(str(cfg.get("architecture", "")))
    domain = file_domain or _normalize_domain(str(cfg.get("domain", "")))
    seed = file_seed or _extract_seed_from_text(str(cfg.get("checkpoint", ""))) or _extract_seed_from_text(path.stem)
    protocol = file_protocol or _normalize_protocol(str(cfg.get("protocol", "")))

    if not arch or not domain or seed is None or not protocol:
        raise ValueError(f"Could not infer arch/domain/seed/protocol from {path}")

    return EvalRecord(
        path=path,
        arch=arch,
        domain=domain,
        seed=seed,
        protocol=protocol,
        payload=payload,
        canonical_filename=canonical_filename,
    )


def load_records(eval_dir: Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for path in sorted(eval_dir.glob("*.json")):
        if path.name == "aggregated.json":
            continue
        try:
            record = _record_from_path(path)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Skipping invalid JSON %s: %s", path, exc)
            continue
        if record is not None:
            records.append(record)
            LOGGER.info(
                "Loaded %s: arch=%s domain=%s seed=%d protocol=%s",
                path.name,
                record.arch,
                record.domain,
                record.seed,
                record.protocol,
            )
    return records


def deduplicate_records(records: Sequence[EvalRecord]) -> list[EvalRecord]:
    by_seed: dict[tuple[str, str, str, int], EvalRecord] = {}
    for record in records:
        key = (record.arch, record.domain, record.protocol, record.seed)
        previous = by_seed.get(key)
        if previous is None:
            by_seed[key] = record
            continue
        keep_new = record.canonical_filename and not previous.canonical_filename
        chosen = record if keep_new else previous
        dropped = previous if keep_new else record
        by_seed[key] = chosen
        LOGGER.warning(
            "Duplicate seed for %s/%s/%s seed%d; keeping %s and dropping %s",
            record.arch,
            record.domain,
            record.protocol,
            record.seed,
            chosen.path.name,
            dropped.path.name,
        )
    return sorted(by_seed.values(), key=lambda r: (r.arch, r.domain, r.protocol, r.seed, r.path.name))


def grouped_records(records: Sequence[EvalRecord]) -> dict[tuple[str, str, str], list[EvalRecord]]:
    groups: DefaultDict[tuple[str, str, str], list[EvalRecord]] = defaultdict(list)
    for record in records:
        groups[(record.arch, record.domain, record.protocol)].append(record)
    for key, group in sorted(groups.items()):
        seeds = sorted(record.seed for record in group)
        if len(seeds) < EXPECTED_SEEDS:
            LOGGER.warning("Only %d/%d seeds for %s: %s", len(seeds), EXPECTED_SEEDS, key, seeds)
    return dict(groups)


def _metric_from_payload(payload: Mapping[str, Any], metric: str) -> float | None:
    if metric in {"ece_15bins", "brier"}:
        value = payload.get("calibration", {}).get(metric)
    elif metric in {"argmax_agreement", "mean_symmetric_kl"}:
        value = payload.get("panel_b", {}).get(metric)
    else:
        value = payload.get("panel_a", {}).get(metric)
    if isinstance(value, Mapping):
        value = value.get("mean")
    if value is None:
        return None
    return float(value)


def summarize_values(records: Sequence[EvalRecord], metric: str) -> MetricSummary | None:
    values_by_seed: list[tuple[int, float]] = []
    for record in sorted(records, key=lambda r: r.seed):
        value = _metric_from_payload(record.payload, metric)
        if value is not None:
            values_by_seed.append((record.seed, value))
    if not values_by_seed:
        return None
    values = tuple(value for _seed, value in values_by_seed)
    seeds = tuple(seed for seed, _value in values_by_seed)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return MetricSummary(mean=statistics.mean(values), std=std, n=len(values), seeds=seeds, values=values)


def aggregate(groups: Mapping[tuple[str, str, str], Sequence[EvalRecord]]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    summary: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    all_metrics = [
        "alpha_v_overall",
        "beta_overall",
        "auroc",
        "auprc",
        "argmax_agreement",
        "mean_symmetric_kl",
        "ece_15bins",
        "brier",
    ]
    for (arch, domain, protocol), records in sorted(groups.items()):
        metrics: dict[str, Any] = {}
        for metric in all_metrics:
            metric_summary = summarize_values(records, metric)
            if metric_summary is None:
                continue
            metrics[metric] = {
                "mean": metric_summary.mean,
                "std": metric_summary.std,
                "n": metric_summary.n,
                "seeds": list(metric_summary.seeds),
                "values": list(metric_summary.values),
            }
        summary.setdefault(arch, {}).setdefault(domain, {})[protocol] = {
            "n_seeds": len({record.seed for record in records}),
            "seeds": sorted({record.seed for record in records}),
            "files": [str(record.path) for record in sorted(records, key=lambda r: r.seed)],
            "metrics": metrics,
        }
    return summary


def _ordered_arches(records: Sequence[EvalRecord]) -> list[str]:
    present = {record.arch for record in records}
    ordered = [arch for arch in ROW_ORDER if arch in present or arch in ROW_ORDER]
    extras = sorted(present.difference(ordered))
    if "factor_gnn" not in ordered and "factor_gnn" in present:
        extras = [arch for arch in extras if arch != "factor_gnn"] + ["factor_gnn"]
    return ordered + extras


def _display_arch(arch: str) -> str:
    return ARCH_DISPLAY.get(arch, arch.replace("_", " ").title())


def _display_domain(domain: str) -> str:
    return DOMAIN_DISPLAY.get(domain, domain.upper())


def _display_protocol(protocol: str) -> str:
    return PROTOCOL_DISPLAY.get(protocol, protocol.replace("_", "-"))


def _format_summary(summary: MetricSummary | None, *, pm: bool = True, signed: bool = False) -> str:
    if summary is None:
        return MISSING
    if signed and not pm:
        return f"{summary.mean:+.3f}"
    if pm:
        return rf"${summary.mean:.3f} \pm {summary.std:.3f}$"
    return f"{summary.mean:.3f}"


def _summary_obj(
    aggregated: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    arch: str,
    domain: str,
    protocol: str,
    metric: str,
) -> MetricSummary | None:
    metric_data = aggregated.get(arch, {}).get(domain, {}).get(protocol, {}).get("metrics", {}).get(metric)
    if not metric_data:
        return None
    return MetricSummary(
        mean=float(metric_data["mean"]),
        std=float(metric_data["std"]),
        n=int(metric_data["n"]),
        seeds=tuple(int(seed) for seed in metric_data["seeds"]),
        values=tuple(float(value) for value in metric_data["values"]),
    )


def _delta_auroc(
    aggregated: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    arch: str,
    domain: str,
) -> MetricSummary | None:
    cumulative = _summary_obj(aggregated, arch, domain, "cumulative", "auroc")
    rebuilt = _summary_obj(aggregated, arch, domain, "state_rebuilt", "auroc")
    if cumulative is None or rebuilt is None:
        return None
    cum_by_seed = dict(zip(cumulative.seeds, cumulative.values))
    rebuilt_by_seed = dict(zip(rebuilt.seeds, rebuilt.values))
    paired_seeds = tuple(sorted(set(cum_by_seed).intersection(rebuilt_by_seed)))
    if not paired_seeds:
        return None
    values = tuple(cum_by_seed[seed] - rebuilt_by_seed[seed] for seed in paired_seeds)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return MetricSummary(mean=statistics.mean(values), std=std, n=len(values), seeds=paired_seeds, values=values)


def render_panel_a(aggregated: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]], arches: Sequence[str]) -> str:
    lines = [
        r"\begin{tabular}{lcccccccccccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{SAT $n=50$ (cumulative)} & \multicolumn{3}{c}{SAT $n=50$ (state-rebuilt)} & SAT $\Delta$ & \multicolumn{3}{c}{GC $n=30$ (cumulative)} & \multicolumn{3}{c}{GC $n=30$ (state-rebuilt)} & GC $\Delta$ \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
        r"Architecture & $\alpha_v\downarrow$ & $\beta\downarrow$ & AUROC$\uparrow$ & $\alpha_v\downarrow$ & $\beta\downarrow$ & AUROC$\uparrow$ & $\Delta$AUROC & $\alpha_v\downarrow$ & $\beta\downarrow$ & AUROC$\uparrow$ & $\alpha_v\downarrow$ & $\beta\downarrow$ & AUROC$\uparrow$ & $\Delta$AUROC \\",
        r"\midrule",
    ]
    for arch in arches:
        row = [_display_arch(arch)]
        for domain in ("sat50", "gc30"):
            for protocol in ("cumulative", "state_rebuilt"):
                for metric, _label in PANEL_A_METRICS:
                    row.append(_format_summary(_summary_obj(aggregated, arch, domain, protocol, metric)))
            row.append(_format_summary(_delta_auroc(aggregated, arch, domain), pm=False, signed=True))
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_panel_b(aggregated: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]], arches: Sequence[str]) -> str:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{SAT $n=50$} & \multicolumn{2}{c}{GC $n=30$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Architecture & Argmax$\uparrow$ & Sym KL$\downarrow$ & Argmax$\uparrow$ & Sym KL$\downarrow$ \\",
        r"\midrule",
    ]
    for arch in arches:
        row = [_display_arch(arch)]
        for domain in ("sat50", "gc30"):
            for metric, _label in PANEL_B_METRICS:
                row.append(_format_summary(_summary_obj(aggregated, arch, domain, "cumulative", metric)))
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_calibration(aggregated: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]], arches: Sequence[str]) -> str:
    lines = [
        r"\begin{tabular}{lllcccccc}",
        r"\toprule",
        "Architecture & Domain & Protocol & " + " & ".join(label for _metric, label in APPENDIX_METRICS) + r" \\",
        r"\midrule",
    ]
    domains = ["sat50", "gc30"]
    protocols = ["cumulative", "state_rebuilt"]
    for arch in arches:
        for domain in domains:
            for protocol in protocols:
                condition = aggregated.get(arch, {}).get(domain, {}).get(protocol)
                if condition is None:
                    continue
                row = [_display_arch(arch), _display_domain(domain), _display_protocol(protocol)]
                for metric, _label in APPENDIX_METRICS:
                    row.append(_format_summary(_summary_obj(aggregated, arch, domain, protocol, metric)))
                lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def missing_report(aggregated: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]], arches: Sequence[str]) -> dict[str, Any]:
    expected: list[dict[str, Any]] = []
    for arch in arches:
        for domain in ("sat50", "gc30"):
            for protocol in ("cumulative", "state_rebuilt"):
                condition = aggregated.get(arch, {}).get(domain, {}).get(protocol)
                seeds = condition.get("seeds", []) if condition else []
                if len(seeds) < EXPECTED_SEEDS:
                    expected.append(
                        {
                            "arch": arch,
                            "domain": domain,
                            "protocol": protocol,
                            "found_seeds": seeds,
                            "missing_seed_count": EXPECTED_SEEDS - len(seeds),
                        }
                    )
    found = []
    for arch, domains in sorted(aggregated.items()):
        for domain, protocols in sorted(domains.items()):
            for protocol, condition in sorted(protocols.items()):
                found.append(
                    {
                        "arch": arch,
                        "display_arch": _display_arch(arch),
                        "domain": domain,
                        "protocol": protocol,
                        "seeds": condition.get("seeds", []),
                    }
                )
    return {"found_conditions": found, "missing_or_incomplete_conditions": expected}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    LOGGER.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", required=True, type=Path)
    parser.add_argument("--output_panel_a", required=True, type=Path)
    parser.add_argument("--output_panel_b", required=True, type=Path)
    parser.add_argument("--output_summary", required=True, type=Path)
    parser.add_argument(
        "--output_calibration",
        type=Path,
        default=None,
        help="Appendix calibration/full-metrics table path. Defaults to tab_verifier_calibration.tex next to panel tables.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    records = deduplicate_records(load_records(args.eval_dir))
    groups = grouped_records(records)
    aggregated = aggregate(groups)
    arches = _ordered_arches(records)

    output_calibration = args.output_calibration or args.output_panel_b.parent / "tab_verifier_calibration.tex"
    write_text(args.output_panel_a, render_panel_a(aggregated, arches))
    write_text(args.output_panel_b, render_panel_b(aggregated, arches))
    write_text(output_calibration, render_calibration(aggregated, arches))

    report = missing_report(aggregated, arches)
    summary_payload = {
        "eval_dir": str(args.eval_dir),
        "expected_seeds_per_condition": EXPECTED_SEEDS,
        "aggregated": aggregated,
        "report": report,
    }
    write_text(args.output_summary, json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")

    print(f"Loaded {len(records)} unique eval JSONs from {args.eval_dir}")
    print("Found conditions:")
    for item in report["found_conditions"]:
        print(f"  - {item['arch']}/{item['domain']}/{item['protocol']}: seeds={item['seeds']}")
    print("Missing or incomplete conditions:")
    for item in report["missing_or_incomplete_conditions"]:
        print(
            f"  - {item['arch']}/{item['domain']}/{item['protocol']}: "
            f"found={item['found_seeds']} missing_count={item['missing_seed_count']}"
        )


if __name__ == "__main__":
    main()
