"""Tokenizer and serializer for parsing search traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Union

from parsing.grammar import IDENTIFIERS, NUMBERS, SERIALIZABLE_SYMBOLS
from parsing.oracle_parser import (
    BacktrackEvent,
    ChoiceEvent,
    OracleParseResult,
    TraceStep,
)

# Special tokens
PAD = 0
BOS = 1
EOS = 2
SEP = 3
STATE = 4
BACKTRACK = 5
CURSOR = 6
COLON = 7

# Input terminals
IDENT_BASE = 10
NUM_BASE = 36
LPAREN = 46
RPAREN = 47
LBRACKET = 48
RBRACKET = 49
COMMA = 50
MINUS = 51

# Grammar symbols
SYM_BASE = 60

# Actions
ALT_BASE = 100
VOCAB_SIZE = 120

SYMBOL_TO_ID: Dict[str, int] = {
    symbol: int(SYM_BASE + idx) for idx, symbol in enumerate(SERIALIZABLE_SYMBOLS)
}


@dataclass
class ParseTrace:
    sequence: List[int]
    loss_mask: List[bool]
    block_ids: List[int]
    label: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def encode_input_token(token: str) -> int:
    token_str = str(token)
    if token_str in IDENTIFIERS:
        return int(IDENT_BASE + IDENTIFIERS.index(token_str))
    if token_str in NUMBERS:
        return int(NUM_BASE + NUMBERS.index(token_str))
    if token_str == "(":
        return int(LPAREN)
    if token_str == ")":
        return int(RPAREN)
    if token_str == "[":
        return int(LBRACKET)
    if token_str == "]":
        return int(RBRACKET)
    if token_str == ",":
        return int(COMMA)
    if token_str == "-":
        return int(MINUS)
    raise ValueError(f"unknown input token: {token_str!r}")


def encode_stack_symbol(symbol: str) -> int:
    if str(symbol) not in SYMBOL_TO_ID:
        raise ValueError(f"symbol not serializable: {symbol!r}")
    return int(SYMBOL_TO_ID[str(symbol)])


def decode_token(token_id: int) -> str:
    token_id = int(token_id)
    if token_id == PAD:
        return "PAD"
    if token_id == BOS:
        return "BOS"
    if token_id == EOS:
        return "EOS"
    if token_id == SEP:
        return "SEP"
    if token_id == STATE:
        return "STATE"
    if token_id == BACKTRACK:
        return "BACKTRACK"
    if token_id == CURSOR:
        return "CURSOR"
    if token_id == COLON:
        return "COLON"
    if IDENT_BASE <= token_id < IDENT_BASE + 26:
        return IDENTIFIERS[token_id - IDENT_BASE]
    if NUM_BASE <= token_id < NUM_BASE + 10:
        return NUMBERS[token_id - NUM_BASE]
    if token_id == LPAREN:
        return "("
    if token_id == RPAREN:
        return ")"
    if token_id == LBRACKET:
        return "["
    if token_id == RBRACKET:
        return "]"
    if token_id == COMMA:
        return ","
    if token_id == MINUS:
        return "-"
    if SYM_BASE <= token_id < ALT_BASE:
        reverse = {value: key for key, value in SYMBOL_TO_ID.items()}
        return reverse[token_id]
    if ALT_BASE <= token_id < VOCAB_SIZE:
        return f"ALT_{token_id - ALT_BASE}"
    raise ValueError(f"unknown token id: {token_id}")


def encode_input_with_cursor(tokens: Sequence[str], cursor: int) -> List[int]:
    encoded: List[int] = []
    inserted = False
    for idx, token in enumerate(tokens):
        if idx == int(cursor):
            encoded.append(int(CURSOR))
            inserted = True
        encoded.append(int(encode_input_token(str(token))))
    if not inserted:
        encoded.append(int(CURSOR))
    return encoded


def serialize_state_block(
    tokens: Sequence[str],
    cursor: int,
    stack: Sequence[str],
    action_token: int | None = None,
) -> List[int]:
    block: List[int] = [int(STATE)]
    block.extend(encode_input_with_cursor(tokens, int(cursor)))
    block.append(int(COLON))
    for symbol in stack:
        block.append(int(encode_stack_symbol(str(symbol))))
    block.append(int(SEP))
    if action_token is not None:
        block.append(int(action_token))
    return block


def build_problem_prefix(tokens: Sequence[str]) -> List[int]:
    encoded = [int(BOS)]
    encoded.extend(int(encode_input_token(str(tok))) for tok in tokens)
    encoded.append(int(SEP))
    return encoded


def compute_block_ids_for_vocab(sequence: Sequence[int]) -> List[int]:
    block_ids: List[int] = []
    current_block = 0
    for token in sequence:
        if int(token) == int(STATE):
            current_block += 1
        block_ids.append(int(current_block))
    return block_ids


def _action_token_for_event(event: Union[ChoiceEvent, BacktrackEvent]) -> int:
    if isinstance(event, ChoiceEvent):
        return int(ALT_BASE + int(event.alternative))
    if isinstance(event, BacktrackEvent):
        return int(BACKTRACK)
    raise TypeError(f"unsupported event type: {type(event)!r}")


def _validate_serialized_lengths(
    sequence: Sequence[int], loss_mask: Sequence[bool], block_ids: Sequence[int]
) -> None:
    if len(sequence) != len(loss_mask):
        raise ValueError("sequence/loss_mask length mismatch")
    if len(sequence) != len(block_ids):
        raise ValueError("sequence/block_ids length mismatch")


def serialize_trace(tokens: Sequence[str], result: OracleParseResult) -> ParseTrace:
    sequence = build_problem_prefix(tokens)
    loss_mask = [False] * len(sequence)
    block_ids = [0] * len(sequence)

    for block_idx, step in enumerate(result.steps, start=1):
        action_token = _action_token_for_event(step.event)
        block = serialize_state_block(
            tokens=tokens,
            cursor=int(step.cursor),
            stack=step.stack,
            action_token=int(action_token),
        )
        sequence.extend(block)
        loss_mask.extend([False] * (len(block) - 1) + [True])
        block_ids.extend([int(block_idx)] * len(block))

    sequence.append(int(EOS))
    loss_mask.append(False)
    block_ids.append(block_ids[-1] if block_ids else 0)
    computed = compute_block_ids_for_vocab(sequence)
    if computed != block_ids:
        raise RuntimeError("explicit block ids do not match computed block ids")
    _validate_serialized_lengths(sequence, loss_mask, block_ids)

    label = "parsed" if bool(result.success) else "failed"
    n_choices = sum(isinstance(step.event, ChoiceEvent) for step in result.steps)
    n_backtracks = sum(isinstance(step.event, BacktrackEvent) for step in result.steps)
    return ParseTrace(
        sequence=[int(tok) for tok in sequence],
        loss_mask=[bool(flag) for flag in loss_mask],
        block_ids=[int(x) for x in block_ids],
        label=str(label),
        meta={
            "input_len": int(len(tokens)),
            "n_choices": int(n_choices),
            "n_backtracks": int(n_backtracks),
            "max_depth": int(result.max_stack_depth),
        },
    )


__all__ = [
    "ALT_BASE",
    "BACKTRACK",
    "BOS",
    "COLON",
    "COMMA",
    "CURSOR",
    "EOS",
    "IDENT_BASE",
    "LBRACKET",
    "LPAREN",
    "MINUS",
    "NUM_BASE",
    "PAD",
    "ParseTrace",
    "RBRACKET",
    "RPAREN",
    "SEP",
    "STATE",
    "SYM_BASE",
    "SYMBOL_TO_ID",
    "VOCAB_SIZE",
    "build_problem_prefix",
    "compute_block_ids_for_vocab",
    "decode_token",
    "encode_input_token",
    "encode_stack_symbol",
    "serialize_state_block",
    "serialize_trace",
]
