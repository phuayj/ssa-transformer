#!/usr/bin/env python3
"""P3/P4 SAT history-irrelevance diagnostic on enriched traces.

P3: fixed state-feature logistic regressions with and without lightweight history.
P4: linear probes on frozen SSA/causal checkpoints plus same-state cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
except ImportError:
    SklearnLogisticRegression = None

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sat.interleaved_tokenizer import SATInterleavedTokenizer
from scripts.train_gc_mask_ablation import compute_block_ids_for_vocab
from universal.ssa_decoder import SSASlotDecoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


STATE_TOKEN = int(SATInterleavedTokenizer.STATE)
SEP_TOKEN = int(SATInterleavedTokenizer.SEP)
CONFLICT_TOKEN = int(SATInterleavedTokenizer.CONFLICT)
BACKJUMP_TOKEN = int(SATInterleavedTokenizer.BACKJUMP)
SOLVED_TOKEN = int(SATInterleavedTokenizer.SOLVED)
FAILED_TOKEN = int(SATInterleavedTokenizer.FAILED)
VAR_OFFSET = int(SATInterleavedTokenizer.VAR_OFFSET)
MAX_VARS = 50
UNASSIGNED_TOKEN = int(SATInterleavedTokenizer.UNASSIGNED)
TRUE_TOKEN = int(SATInterleavedTokenizer.TRUE_VAL)
FALSE_TOKEN = int(SATInterleavedTokenizer.FALSE_VAL)
NEWLY_TRUE_TOKEN = int(SATInterleavedTokenizer.NEWLY_TRUE)
NEWLY_FALSE_TOKEN = int(SATInterleavedTokenizer.NEWLY_FALSE)

FORWARD_ACTION = "forward"
CONFLICT_ACTION = "conflict"
SOLVED_ACTION = "solved"
FAILED_ACTION = "failed"
OTHER_ACTION = "other"
ACTION_TYPES: Tuple[str, ...] = (
    FORWARD_ACTION,
    CONFLICT_ACTION,
    SOLVED_ACTION,
    FAILED_ACTION,
    OTHER_ACTION,
)


@dataclass(frozen=True)
class DecisionSample:
    trace_index: int
    decision_index: int
    state_start: int
    sep_pos: int
    block_id: int
    prefix_len: int
    state_features: np.ndarray
    history_features: np.ndarray
    oracle_label: int
    state_hash: Tuple[float, ...]
    action_type: str


@dataclass(frozen=True)
class ModelMeta:
    config: Dict[str, Any]
    mask_mode: str
    max_seq_len: int
    vocab_size: int
    n_slots: int
    d_model: int


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1.0)


def _var_from_token(token_id: int) -> int:
    tok = int(token_id)
    if tok < VAR_OFFSET:
        raise ValueError(f"token is not a variable token: {tok}")
    return int(tok - VAR_OFFSET)


def _is_var_token(token_id: int) -> bool:
    tok = int(token_id)
    return int(VAR_OFFSET) <= tok < int(VAR_OFFSET + SATInterleavedTokenizer.MAX_VARS)


def _load_traces(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, list):
        raise TypeError(f"expected list in traces file, got {type(raw)}")
    traces: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"trace {idx} is not a dict: {type(item)}")
        traces.append({str(k): v for k, v in item.items()})
    return traces


def load_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[SSASlotDecoder, ModelMeta]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"checkpoint missing model_state_dict: {checkpoint_path}")

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})
    vocab_size = int(config.get("vocab_size", 2030))
    d_model = int(config.get("d_model", 256))
    n_heads = int(config.get("n_heads", 8))
    n_layers = int(config.get("n_layers", 6))
    n_slots = int(config.get("n_slots", 32))
    max_seq_len = int(config.get("max_seq_len", 4096))
    dropout = float(config.get("dropout", 0.1))
    mask_mode = str(config.get("mask_mode", "selective_ssa"))

    model = SSASlotDecoder(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        n_slots=int(n_slots),
        max_seq_len=int(max_seq_len),
        dropout=float(dropout),
        cbv_enabled=bool(config.get("cbv_enabled", False)),
        n_branch_slots=int(config.get("n_branch_slots", 12)),
        n_verifier_slots=int(config.get("n_verifier_slots", 8)),
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    meta = ModelMeta(
        config=dict(config),
        mask_mode=str(mask_mode),
        max_seq_len=int(max_seq_len),
        vocab_size=int(vocab_size),
        n_slots=int(n_slots),
        d_model=int(d_model),
    )
    return model, meta


def build_ssa_mask(block_ids: Sequence[int], n_slots: int) -> torch.Tensor:
    """Build SSA block-diagonal attention mask for diagnostics/debugging."""
    seq_len = len(block_ids)
    mask = torch.zeros(seq_len + int(n_slots), seq_len + int(n_slots), dtype=torch.bool)
    mask[: int(n_slots), : int(n_slots)] = True
    block_ids_t = torch.tensor(list(block_ids), dtype=torch.long)
    mask[int(n_slots) :, : int(n_slots)] = True
    prefix_visible = block_ids_t.eq(0)
    mask[: int(n_slots), int(n_slots) :] = prefix_visible.unsqueeze(0).expand(
        int(n_slots), -1
    )
    for i in range(seq_len):
        bi = int(block_ids_t[i].item())
        if bi == 0:
            visible = block_ids_t[: i + 1].eq(0)
        else:
            visible = block_ids_t[: i + 1].eq(0) | block_ids_t[: i + 1].eq(bi)
        mask[int(n_slots) + i, int(n_slots) : int(n_slots) + i + 1] = visible
    return mask


def _action_type_from_next_token(next_token: Optional[int]) -> str:
    if next_token is None:
        return OTHER_ACTION
    tok = int(next_token)
    if tok == CONFLICT_TOKEN:
        return CONFLICT_ACTION
    if _is_var_token(tok):
        return FORWARD_ACTION
    if tok == SOLVED_TOKEN:
        return SOLVED_ACTION
    if tok == FAILED_TOKEN:
        return FAILED_ACTION
    return OTHER_ACTION


def _parse_state_features(
    sequence: Sequence[int],
    state_start: int,
    sep_pos: int,
    block_id: int,
    num_vars: int = MAX_VARS,
) -> np.ndarray:
    feature_dim = int(num_vars) * 4 + 3
    features = np.zeros((feature_dim,), dtype=np.float32)
    state_tokens = [int(tok) for tok in sequence[int(state_start) + 1 : int(sep_pos)]]
    if len(state_tokens) % 2 != 0:
        raise ValueError("STATE block must contain var/status pairs")

    visible_vars = 0
    for idx in range(0, len(state_tokens), 2):
        var_id = _var_from_token(int(state_tokens[idx]))
        if var_id < 0 or var_id >= int(num_vars):
            raise ValueError(f"variable out of range in state block: {var_id}")
        status_tok = int(state_tokens[idx + 1])
        base = int(var_id) * 4
        if status_tok == UNASSIGNED_TOKEN:
            features[base + 0] = 1.0
        elif status_tok in (TRUE_TOKEN, NEWLY_TRUE_TOKEN):
            features[base + 1] = 1.0
        elif status_tok in (FALSE_TOKEN, NEWLY_FALSE_TOKEN):
            features[base + 2] = 1.0
        else:
            raise ValueError(
                f"unexpected SAT status token in enriched trace: {status_tok}"
            )
        features[base + 3] = 1.0
        visible_vars += 1

    global_offset = int(num_vars) * 4
    features[global_offset + 0] = float(1.0 - (float(visible_vars) / float(num_vars)))
    features[global_offset + 1] = float(visible_vars)
    features[global_offset + 2] = float(block_id) / float(num_vars)
    return features


def _state_hash_from_features(state_features: np.ndarray) -> Tuple[float, ...]:
    semantic = np.array(state_features, dtype=np.float32, copy=True)
    semantic[-1] = 0.0
    rounded = np.round(semantic.astype(np.float64), 3)
    return tuple(float(x) for x in rounded.tolist())


def _history_features(
    previous_backtracks: int,
    previous_decisions: int,
    previous_conflicts: int,
    recent_actions: Iterable[str],
) -> np.ndarray:
    counts = Counter(str(action) for action in recent_actions)
    features = np.array(
        [
            float(previous_backtracks),
            float(previous_decisions),
            float(previous_conflicts),
            float(counts.get(FORWARD_ACTION, 0)),
            float(counts.get(CONFLICT_ACTION, 0)),
            float(counts.get(SOLVED_ACTION, 0)),
            float(counts.get(FAILED_ACTION, 0)),
            float(counts.get(OTHER_ACTION, 0)),
        ],
        dtype=np.float32,
    )
    return features


def extract_decision_samples(
    traces: Sequence[Dict[str, Any]],
    num_vars: int = MAX_VARS,
) -> List[List[DecisionSample]]:
    per_trace: List[List[DecisionSample]] = []
    total_decisions = 0
    total_forward = 0
    total_backtrack = 0

    for trace_index, trace in enumerate(traces):
        sequence = [int(tok) for tok in trace["sequence"]]
        block_ids = [int(x) for x in trace["block_ids"]]
        if len(sequence) != len(block_ids):
            raise ValueError(f"trace {trace_index} sequence/block_ids length mismatch")

        trace_samples: List[DecisionSample] = []
        previous_decisions = 0
        previous_conflicts = 0
        previous_backtracks = 0
        recent_actions: Deque[str] = deque(maxlen=5)
        decision_index = 0
        pos = 0

        while pos < len(sequence):
            token = int(sequence[pos])
            if token != STATE_TOKEN:
                pos += 1
                continue

            sep_pos = pos + 1
            while sep_pos < len(sequence) and int(sequence[sep_pos]) != SEP_TOKEN:
                sep_pos += 1
            if sep_pos >= len(sequence):
                raise ValueError(f"trace {trace_index} STATE at {pos} missing SEP")

            next_token = (
                int(sequence[sep_pos + 1]) if sep_pos + 1 < len(sequence) else None
            )
            action_type = _action_type_from_next_token(next_token)
            if action_type not in (FORWARD_ACTION, CONFLICT_ACTION):
                pos = sep_pos + 1
                continue

            block_id = int(block_ids[sep_pos])
            state_features = _parse_state_features(
                sequence=sequence,
                state_start=int(pos),
                sep_pos=int(sep_pos),
                block_id=int(block_id),
                num_vars=int(num_vars),
            )
            history_features = _history_features(
                previous_backtracks=int(previous_backtracks),
                previous_decisions=int(previous_decisions),
                previous_conflicts=int(previous_conflicts),
                recent_actions=recent_actions,
            )
            oracle_label = 1 if action_type == CONFLICT_ACTION else 0
            trace_samples.append(
                DecisionSample(
                    trace_index=int(trace_index),
                    decision_index=int(decision_index),
                    state_start=int(pos),
                    sep_pos=int(sep_pos),
                    block_id=int(block_id),
                    prefix_len=int(sep_pos + 1),
                    state_features=state_features,
                    history_features=history_features,
                    oracle_label=int(oracle_label),
                    state_hash=_state_hash_from_features(state_features),
                    action_type=str(action_type),
                )
            )

            total_decisions += 1
            total_backtrack += int(oracle_label == 1)
            total_forward += int(oracle_label == 0)

            previous_decisions += 1
            if action_type == CONFLICT_ACTION:
                previous_conflicts += 1
                previous_backtracks += 1
            recent_actions.append(str(action_type))
            decision_index += 1
            pos = sep_pos + 1

        per_trace.append(trace_samples)

    logger.info(
        "parsed_decision_points traces=%d total=%d forward=%d backtrack=%d backtrack_rate=%.4f",
        int(len(per_trace)),
        int(total_decisions),
        int(total_forward),
        int(total_backtrack),
        _safe_div(float(total_backtrack), float(total_decisions)),
    )
    return per_trace


def _split_trace_indices(num_traces: int, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(int(num_traces)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    split = max(1, int(math.floor(0.8 * float(len(indices)))))
    split = min(split, max(len(indices) - 1, 1)) if len(indices) > 1 else len(indices)
    train_indices = indices[:split]
    test_indices = indices[split:]
    if not test_indices and train_indices:
        test_indices = [train_indices.pop()]
    return train_indices, test_indices


def _flatten_selected(
    per_trace_samples: Sequence[Sequence[DecisionSample]],
    trace_indices: Sequence[int],
) -> List[DecisionSample]:
    flat: List[DecisionSample] = []
    for trace_idx in trace_indices:
        flat.extend(list(per_trace_samples[int(trace_idx)]))
    return flat


def _fit_logreg_binary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> float:
    unique = np.unique(y_train)
    if unique.shape[0] < 2:
        majority = int(unique[0])
        logger.warning(
            "training labels contain one class=%s; using constant predictor fallback",
            str(unique.tolist()),
        )
        preds = np.full_like(y_test, fill_value=int(majority))
        return float(np.mean(preds == y_test))
    if SklearnLogisticRegression is not None:
        clf = SklearnLogisticRegression(max_iter=2000, random_state=int(seed))
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        return float(np.mean(preds == y_test))

    logger.warning("sklearn not available; using torch logistic-regression fallback")
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    torch.manual_seed(int(seed))
    model = torch.nn.Linear(int(x_train.shape[1]), 1)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=200,
        line_search_fn="strong_wolfe",
    )

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = model(x_train_t).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train_t)
        loss.backward()
        return loss

    optimizer.step(_closure)
    with torch.no_grad():
        preds = (torch.sigmoid(model(x_test_t).squeeze(-1)) >= 0.5).to(
            dtype=torch.int64
        )
    return float((preds.cpu().numpy() == y_test).mean())


def run_p3(
    train_samples: Sequence[DecisionSample],
    test_samples: Sequence[DecisionSample],
    seed: int,
) -> Dict[str, Any]:
    x_train_state = np.stack(
        [sample.state_features for sample in train_samples], axis=0
    )
    x_test_state = np.stack([sample.state_features for sample in test_samples], axis=0)
    x_train_hist = np.stack(
        [
            np.concatenate([sample.state_features, sample.history_features])
            for sample in train_samples
        ],
        axis=0,
    )
    x_test_hist = np.stack(
        [
            np.concatenate([sample.state_features, sample.history_features])
            for sample in test_samples
        ],
        axis=0,
    )
    y_train = np.asarray(
        [int(sample.oracle_label) for sample in train_samples], dtype=np.int64
    )
    y_test = np.asarray(
        [int(sample.oracle_label) for sample in test_samples], dtype=np.int64
    )

    acc_state = _fit_logreg_binary(
        x_train=x_train_state,
        y_train=y_train,
        x_test=x_test_state,
        y_test=y_test,
        seed=int(seed),
    )
    acc_state_history = _fit_logreg_binary(
        x_train=x_train_hist,
        y_train=y_train,
        x_test=x_test_hist,
        y_test=y_test,
        seed=int(seed),
    )
    delta = float(acc_state_history - acc_state)
    logger.info(
        "p3_history_irrelevance n_train=%d n_test=%d acc_T=%.4f acc_TH=%.4f delta=%.4f",
        int(len(train_samples)),
        int(len(test_samples)),
        float(acc_state),
        float(acc_state_history),
        float(delta),
    )
    return {
        "acc_state_only": float(acc_state),
        "acc_state_plus_history": float(acc_state_history),
        "delta": float(delta),
        "n_train": int(len(train_samples)),
        "n_test": int(len(test_samples)),
    }


def _subsample_samples(
    samples: Sequence[DecisionSample],
    max_samples: int,
    seed: int,
) -> List[DecisionSample]:
    chosen = list(samples)
    if len(chosen) <= int(max_samples):
        return chosen
    rng = random.Random(int(seed))
    rng.shuffle(chosen)
    return chosen[: int(max_samples)]


def _extract_hidden_vector(
    model: SSASlotDecoder,
    input_tokens: Sequence[int],
    block_ids: Sequence[int],
    mask_mode: str,
    device: torch.device,
) -> np.ndarray:
    last_block_output: Dict[str, torch.Tensor] = {}

    def _hook(
        _module: torch.nn.Module,
        _inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        last_block_output["value"] = output.detach()

    handle = model.blocks[-1].register_forward_hook(_hook)
    input_tensor = torch.tensor([list(input_tokens)], dtype=torch.long, device=device)
    block_tensor = torch.tensor([list(block_ids)], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            model(
                input_tensor,
                block_ids=block_tensor,
                mask_mode=str(mask_mode),
            )
    finally:
        handle.remove()

    if "value" not in last_block_output:
        raise RuntimeError(
            "missing hidden-state hook output from final transformer block"
        )

    hidden = last_block_output["value"]
    seq_hidden = hidden[:, int(model.n_slots) :, :]
    sep_index = int(len(input_tokens) - 1)
    vector = seq_hidden[0, sep_index, :].detach().to(dtype=torch.float32).cpu().numpy()
    return vector


def _build_prefix_from_trace(
    trace: Dict[str, Any], sample: DecisionSample
) -> Tuple[List[int], List[int]]:
    sequence = [int(tok) for tok in trace["sequence"][: int(sample.prefix_len)]]
    block_ids = compute_block_ids_for_vocab(
        sequence, vocab_size=int(SATInterleavedTokenizer.VOCAB_SIZE)
    )
    if len(sequence) != len(block_ids):
        raise RuntimeError("prefix sequence/block_ids length mismatch")
    if int(sequence[-1]) != SEP_TOKEN:
        raise RuntimeError("prefix must end at STATE SEP token")
    return sequence, block_ids


def extract_probe_dataset(
    *,
    model: SSASlotDecoder,
    model_name: str,
    mask_mode: str,
    traces: Sequence[Dict[str, Any]],
    samples: Sequence[DecisionSample],
    device: torch.device,
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skipped_budget = 0

    for idx, sample in enumerate(samples):
        prefix_tokens, prefix_block_ids = _build_prefix_from_trace(
            trace=traces[int(sample.trace_index)],
            sample=sample,
        )
        if len(prefix_tokens) > int(max_seq_len):
            skipped_budget += 1
            continue

        hidden_vector = _extract_hidden_vector(
            model=model,
            input_tokens=prefix_tokens,
            block_ids=prefix_block_ids,
            mask_mode=str(mask_mode),
            device=device,
        )
        rows.append(
            {
                "hidden_vector": hidden_vector,
                "oracle_label": int(sample.oracle_label),
                "state_hash": sample.state_hash,
                "trace_index": int(sample.trace_index),
                "decision_index": int(sample.decision_index),
                "block_id": int(sample.block_id),
                "prefix_len": int(sample.prefix_len),
            }
        )

        if (idx + 1) % 250 == 0 or idx == len(samples) - 1:
            logger.info(
                "%s_hidden_extract processed=%d/%d kept=%d skipped_budget=%d last_prefix_len=%d",
                str(model_name),
                int(idx + 1),
                int(len(samples)),
                int(len(rows)),
                int(skipped_budget),
                int(len(prefix_tokens)),
            )

    if rows:
        norms = [
            float(np.linalg.norm(row["hidden_vector"]))
            for row in rows[: min(256, len(rows))]
        ]
        logger.info(
            "%s_hidden_extract_done kept=%d skipped_budget=%d sample_hidden_norm_mean=%.4f",
            str(model_name),
            int(len(rows)),
            int(skipped_budget),
            float(np.mean(norms)) if norms else 0.0,
        )
    else:
        logger.warning("%s_hidden_extract produced no usable rows", str(model_name))
    return rows


def _probe_accuracy(
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    seed: int,
) -> float:
    x_train = np.stack(
        [np.asarray(row["hidden_vector"], dtype=np.float32) for row in train_rows],
        axis=0,
    )
    y_train = np.asarray(
        [int(row["oracle_label"]) for row in train_rows], dtype=np.int64
    )
    x_test = np.stack(
        [np.asarray(row["hidden_vector"], dtype=np.float32) for row in test_rows],
        axis=0,
    )
    y_test = np.asarray([int(row["oracle_label"]) for row in test_rows], dtype=np.int64)
    return _fit_logreg_binary(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        seed=int(seed),
    )


def _pairwise_same_state_cosine(rows: Sequence[Dict[str, Any]]) -> Tuple[float, int]:
    grouped: Dict[Tuple[float, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["state_hash"]].append(row)

    cosines: List[float] = []
    pair_count = 0
    for group_rows in grouped.values():
        if len(group_rows) < 2:
            continue
        for i in range(len(group_rows) - 1):
            for j in range(i + 1, len(group_rows)):
                a = group_rows[i]
                b = group_rows[j]
                different_history = (
                    int(a["trace_index"]) != int(b["trace_index"])
                    or int(a["block_id"]) != int(b["block_id"])
                    or int(a["decision_index"]) != int(b["decision_index"])
                )
                if not different_history:
                    continue
                vec_a = torch.tensor(a["hidden_vector"], dtype=torch.float32)
                vec_b = torch.tensor(b["hidden_vector"], dtype=torch.float32)
                cos = float(
                    F.cosine_similarity(
                        vec_a.unsqueeze(0),
                        vec_b.unsqueeze(0),
                        dim=-1,
                        eps=1e-8,
                    ).item()
                )
                cosines.append(float(cos))
                pair_count += 1
    if not cosines:
        return 0.0, 0
    return float(sum(cosines) / len(cosines)), int(pair_count)


def run_p4(
    *,
    traces: Sequence[Dict[str, Any]],
    train_samples: Sequence[DecisionSample],
    test_samples: Sequence[DecisionSample],
    ssa_checkpoint: Path,
    causal_checkpoint: Path,
    device: torch.device,
    max_probe_samples: int,
    seed: int,
) -> Dict[str, Any]:
    ssa_model, ssa_meta = load_model(ssa_checkpoint, device)
    causal_model, causal_meta = load_model(causal_checkpoint, device)

    if int(ssa_meta.vocab_size) != int(causal_meta.vocab_size):
        raise RuntimeError(
            "SSA and causal checkpoints use different vocab sizes: "
            f"ssa={ssa_meta.vocab_size} causal={causal_meta.vocab_size}"
        )

    max_prefix_len = min(int(ssa_meta.max_seq_len), int(causal_meta.max_seq_len))
    logger.info(
        "loaded_probe_models ssa_mask=%s causal_mask=%s shared_max_seq_len=%d ssa_d_model=%d causal_d_model=%d",
        str(ssa_meta.mask_mode),
        str(causal_meta.mask_mode),
        int(max_prefix_len),
        int(ssa_meta.d_model),
        int(causal_meta.d_model),
    )

    probe_train_samples = _subsample_samples(
        samples=train_samples,
        max_samples=int(max_probe_samples),
        seed=int(seed),
    )
    probe_test_samples = _subsample_samples(
        samples=test_samples,
        max_samples=int(max_probe_samples),
        seed=int(seed) + 1,
    )
    logger.info(
        "probe_sample_budget train=%d/%d test=%d/%d",
        int(len(probe_train_samples)),
        int(len(train_samples)),
        int(len(probe_test_samples)),
        int(len(test_samples)),
    )

    ssa_train_rows = extract_probe_dataset(
        model=ssa_model,
        model_name="ssa_train",
        mask_mode=str(ssa_meta.mask_mode),
        traces=traces,
        samples=probe_train_samples,
        device=device,
        max_seq_len=int(max_prefix_len),
    )
    ssa_test_rows = extract_probe_dataset(
        model=ssa_model,
        model_name="ssa_test",
        mask_mode=str(ssa_meta.mask_mode),
        traces=traces,
        samples=probe_test_samples,
        device=device,
        max_seq_len=int(max_prefix_len),
    )
    causal_train_rows = extract_probe_dataset(
        model=causal_model,
        model_name="causal_train",
        mask_mode=str(causal_meta.mask_mode),
        traces=traces,
        samples=probe_train_samples,
        device=device,
        max_seq_len=int(max_prefix_len),
    )
    causal_test_rows = extract_probe_dataset(
        model=causal_model,
        model_name="causal_test",
        mask_mode=str(causal_meta.mask_mode),
        traces=traces,
        samples=probe_test_samples,
        device=device,
        max_seq_len=int(max_prefix_len),
    )

    common_train = min(len(ssa_train_rows), len(causal_train_rows))
    common_test = min(len(ssa_test_rows), len(causal_test_rows))
    if common_train == 0 or common_test == 0:
        raise RuntimeError("probe extraction produced no common train/test rows")

    ssa_train_rows = ssa_train_rows[:common_train]
    causal_train_rows = causal_train_rows[:common_train]
    ssa_test_rows = ssa_test_rows[:common_test]
    causal_test_rows = causal_test_rows[:common_test]

    ssa_acc = _probe_accuracy(ssa_train_rows, ssa_test_rows, seed=int(seed))
    causal_acc = _probe_accuracy(causal_train_rows, causal_test_rows, seed=int(seed))
    ssa_cos, ssa_pairs = _pairwise_same_state_cosine(ssa_test_rows)
    causal_cos, causal_pairs = _pairwise_same_state_cosine(causal_test_rows)

    logger.info(
        "p4_linear_probes n_train=%d n_test=%d ssa_acc=%.4f causal_acc=%.4f ssa_same_state_cos=%.4f causal_same_state_cos=%.4f",
        int(common_train),
        int(common_test),
        float(ssa_acc),
        float(causal_acc),
        float(ssa_cos),
        float(causal_cos),
    )

    return {
        "ssa_probe_accuracy": float(ssa_acc),
        "causal_probe_accuracy": float(causal_acc),
        "ssa_mean_cosine_same_state": float(ssa_cos),
        "causal_mean_cosine_same_state": float(causal_cos),
        "n_same_state_pairs_ssa": int(ssa_pairs),
        "n_same_state_pairs_causal": int(causal_pairs),
        "n_probe_train": int(common_train),
        "n_probe_test": int(common_test),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAT history irrelevance and frozen-model linear probes"
    )
    parser.add_argument(
        "--traces",
        type=str,
        default="experiments/sat-n50-enriched-traces/traces.pkl",
    )
    parser.add_argument(
        "--ssa-checkpoint",
        type=str,
        default="experiments/sat-n50-enriched-selective_ssa-seed42/best.pt",
    )
    parser.add_argument(
        "--causal-checkpoint",
        type=str,
        default="experiments/sat-n50-enriched-full_causal-seed42/best.pt",
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-traces", type=int, default=500)
    parser.add_argument("--max-probe-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))

    traces_path = Path(args.traces)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))

    traces = _load_traces(traces_path)
    if int(args.max_traces) > 0:
        traces = traces[: int(args.max_traces)]
    if len(traces) < 2:
        raise ValueError("need at least two traces for train/test split")

    logger.info(
        "loaded_traces path=%s selected=%d seed=%d device=%s",
        str(traces_path),
        int(len(traces)),
        int(args.seed),
        str(device),
    )

    per_trace_samples = extract_decision_samples(traces)
    train_trace_indices, test_trace_indices = _split_trace_indices(
        num_traces=int(len(per_trace_samples)),
        seed=int(args.seed),
    )
    train_samples = _flatten_selected(per_trace_samples, train_trace_indices)
    test_samples = _flatten_selected(per_trace_samples, test_trace_indices)

    if not train_samples or not test_samples:
        raise RuntimeError("empty train/test decision sample split")

    logger.info(
        "trace_split train_traces=%d test_traces=%d train_decisions=%d test_decisions=%d",
        int(len(train_trace_indices)),
        int(len(test_trace_indices)),
        int(len(train_samples)),
        int(len(test_samples)),
    )

    p3_results = run_p3(
        train_samples=train_samples,
        test_samples=test_samples,
        seed=int(args.seed),
    )
    p4_results = run_p4(
        traces=traces,
        train_samples=train_samples,
        test_samples=test_samples,
        ssa_checkpoint=Path(args.ssa_checkpoint),
        causal_checkpoint=Path(args.causal_checkpoint),
        device=device,
        max_probe_samples=int(args.max_probe_samples),
        seed=int(args.seed),
    )

    payload = {
        "p3_history_irrelevance": p3_results,
        "p4_linear_probes": p4_results,
    }

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("wrote_results path=%s", str(results_path))


if __name__ == "__main__":
    main()
