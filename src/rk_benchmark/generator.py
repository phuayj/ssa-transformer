"""Data generation for the r^k mechanistic benchmark."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Token constants.
VALUE_TRUE = 8
VALUE_FALSE = 9
BLOCK_START = 10
BLOCK_END = 11
RECORD_SEP = 12
QUERY_MARK = 13
CLS_TOKEN = 14
PAD_TOKEN = 15


@dataclass
class RkBenchmarkConfig:
    k: int = 4
    num_records: int = 8
    key_len: int = 4
    alphabet_size: int = 8
    difficulty: int = 2
    correlated: bool = False
    positive_rate: float = 0.5
    max_seq_len: int = 2048


def _sample_key(key_len: int, alphabet_size: int, rng: random.Random) -> List[int]:
    return [rng.randrange(alphabet_size) for _ in range(int(key_len))]


def _value_to_token(value_bit: int) -> int:
    return VALUE_TRUE if int(value_bit) == 1 else VALUE_FALSE


def _sample_effective_difficulty(config: RkBenchmarkConfig, rng: random.Random) -> int:
    d = int(config.difficulty)
    if d < 0 or d > 4:
        raise ValueError(f"difficulty must be in [0, 4], got {d}")
    if config.correlated:
        return d
    candidates = sorted({max(0, d - 1), d, min(4, d + 1)})
    return int(rng.choice(candidates))


def _make_distractor_key(
    query_key: Sequence[int],
    difficulty: int,
    rng: random.Random,
    alphabet_size: int,
) -> List[int]:
    """Generate a distractor key with controlled similarity to query."""
    key = list(query_key)
    if difficulty == 0:
        for i in range(len(key)):
            choices = [x for x in range(alphabet_size) if x != query_key[i]]
            key[i] = int(rng.choice(choices))
    elif difficulty == 1:
        num_flips = rng.randint(2, len(key))
        positions = rng.sample(range(len(key)), num_flips)
        for i in positions:
            choices = [x for x in range(alphabet_size) if x != query_key[i]]
            key[i] = int(rng.choice(choices))
    elif difficulty == 2:
        if rng.random() < 0.5:
            pos = rng.randint(0, len(key) - 1)
            choices = [x for x in range(alphabet_size) if x != query_key[pos]]
            key[pos] = int(rng.choice(choices))
        else:
            num_flips = rng.randint(min(3, len(key)), len(key))
            positions = rng.sample(range(len(key)), num_flips)
            for i in positions:
                choices = [x for x in range(alphabet_size) if x != query_key[i]]
                key[i] = int(rng.choice(choices))
    elif difficulty == 3:
        pos = rng.randint(0, len(key) - 1)
        choices = [x for x in range(alphabet_size) if x != query_key[pos]]
        key[pos] = int(rng.choice(choices))
    elif difficulty == 4:
        pos = rng.randint(0, len(key) - 1)
        choices = [x for x in range(alphabet_size) if x != query_key[pos]]
        key[pos] = int(rng.choice(choices))
    else:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    return key


def _make_block(
    config: RkBenchmarkConfig,
    rng: random.Random,
    block_answer: int,
    block_difficulty: int,
) -> Tuple[List[int], int]:
    query_key = _sample_key(config.key_len, config.alphabet_size, rng)
    records: List[Tuple[List[int], int]] = [(list(query_key), int(block_answer))]
    used_keys = {tuple(query_key)}

    for _ in range(int(config.num_records) - 1):
        for _attempt in range(256):
            key = _make_distractor_key(
                query_key=query_key,
                difficulty=int(block_difficulty),
                rng=rng,
                alphabet_size=int(config.alphabet_size),
            )
            key_tuple = tuple(key)
            if key_tuple in used_keys:
                continue

            if int(block_difficulty) == 4:
                value_bit = int(block_answer)
            else:
                value_bit = int(rng.randint(0, 1))

            records.append((key, value_bit))
            used_keys.add(key_tuple)
            break
        else:
            # Fail fast if the space is too constrained.
            raise RuntimeError(
                "Unable to sample unique distractor keys; "
                "increase alphabet_size/key_len or reduce num_records"
            )

    rng.shuffle(records)

    tokens: List[int] = [BLOCK_START]
    for key, value_bit in records:
        tokens.extend(int(x) for x in key)
        tokens.append(_value_to_token(value_bit))
        tokens.append(RECORD_SEP)
    tokens.append(QUERY_MARK)
    tokens.extend(int(x) for x in query_key)
    tokens.append(BLOCK_END)
    return tokens, int(block_answer)


def generate_example_with_metadata(
    config: RkBenchmarkConfig,
    rng: random.Random,
    target_label: Optional[int] = None,
) -> Tuple[List[int], int, List[int]]:
    """Generate one full example and return (tokens, label, oracle_features)."""
    if int(config.k) <= 0:
        raise ValueError(f"k must be positive, got {config.k}")
    if int(config.num_records) < 2:
        raise ValueError(f"num_records must be >=2, got {config.num_records}")
    if int(config.key_len) <= 0:
        raise ValueError(f"key_len must be positive, got {config.key_len}")
    if int(config.alphabet_size) < 2:
        raise ValueError(f"alphabet_size must be >=2, got {config.alphabet_size}")

    if target_label is None:
        label = 1 if rng.random() < float(config.positive_rate) else 0
    else:
        label = int(target_label)
        if label not in {0, 1}:
            raise ValueError(f"target_label must be 0 or 1, got {target_label}")

    oracle_features: List[int]
    if label == 1:
        oracle_features = [1] * int(config.k)
    else:
        oracle_features = [1] * int(config.k)
        zero_index = int(rng.randrange(int(config.k)))
        oracle_features[zero_index] = 0
        for i in range(int(config.k)):
            if i == zero_index:
                continue
            if rng.random() < 0.35:
                oracle_features[i] = 0

    if config.correlated:
        shared_difficulty = _sample_effective_difficulty(config, rng)
        per_block_difficulty = [shared_difficulty] * int(config.k)
    else:
        per_block_difficulty = [
            _sample_effective_difficulty(config, rng) for _ in range(int(config.k))
        ]

    tokens: List[int] = [CLS_TOKEN]
    block_answers: List[int] = []
    for block_idx in range(int(config.k)):
        block_tokens, block_answer = _make_block(
            config=config,
            rng=rng,
            block_answer=int(oracle_features[block_idx]),
            block_difficulty=int(per_block_difficulty[block_idx]),
        )
        tokens.extend(block_tokens)
        block_answers.append(int(block_answer))

    computed_label = int(all(int(x) == 1 for x in block_answers))
    if computed_label != int(label):
        raise RuntimeError(
            f"Label mismatch: requested={label} derived={computed_label}; "
            "generator invariants violated"
        )

    if len(tokens) > int(config.max_seq_len):
        raise ValueError(
            f"Generated sequence length {len(tokens)} exceeds max_seq_len "
            f"{config.max_seq_len}"
        )

    return tokens, int(label), [int(x) for x in oracle_features]


def generate_example(
    config: RkBenchmarkConfig, rng: random.Random
) -> Tuple[List[int], int]:
    """Returns (token_ids, label)."""
    tokens, label, _oracle = generate_example_with_metadata(config=config, rng=rng)
    return tokens, label


def extract_oracle_features_from_tokens(
    tokens: Sequence[int], k: int, key_len: int
) -> List[int]:
    """Extract true per-block answers by exact key-match lookup from tokens."""
    idx = 0
    n = len(tokens)
    if n == 0 or int(tokens[0]) != CLS_TOKEN:
        raise ValueError("Malformed example: missing CLS token at position 0")
    idx += 1

    features: List[int] = []
    for _ in range(int(k)):
        if idx >= n or int(tokens[idx]) != BLOCK_START:
            raise ValueError(f"Malformed example: expected BLOCK_START at index {idx}")
        idx += 1

        records: dict[Tuple[int, ...], int] = {}
        while idx < n and int(tokens[idx]) != QUERY_MARK:
            if idx + int(key_len) >= n:
                raise ValueError("Malformed block: truncated record")
            key = tuple(int(x) for x in tokens[idx : idx + int(key_len)])
            idx += int(key_len)

            if idx >= n:
                raise ValueError("Malformed block: missing value token")
            value_tok = int(tokens[idx])
            idx += 1
            if value_tok not in {VALUE_TRUE, VALUE_FALSE}:
                raise ValueError(f"Unexpected value token: {value_tok}")
            if key in records:
                raise ValueError("Malformed block: duplicate record key")
            records[key] = 1 if value_tok == VALUE_TRUE else 0

            if idx >= n or int(tokens[idx]) != RECORD_SEP:
                raise ValueError("Malformed block: missing RECORD_SEP")
            idx += 1

        if idx >= n or int(tokens[idx]) != QUERY_MARK:
            raise ValueError("Malformed block: missing QUERY_MARK")
        idx += 1

        if idx + int(key_len) > n:
            raise ValueError("Malformed block: truncated query key")
        query = tuple(int(x) for x in tokens[idx : idx + int(key_len)])
        idx += int(key_len)

        if idx >= n or int(tokens[idx]) != BLOCK_END:
            raise ValueError("Malformed block: missing BLOCK_END")
        idx += 1

        if query not in records:
            raise ValueError("Malformed block: query key missing from records")
        features.append(int(records[query]))

    return features
