"""Fixed ambiguous PEG grammar for expression parsing experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence

IDENTIFIERS: List[str] = [chr(ord("a") + i) for i in range(26)]
NUMBERS: List[str] = [str(i) for i in range(10)]
PUNCTUATION: List[str] = ["(", ")", "[", "]", ",", "-"]

TERMINAL_SYMBOLS = {"IDENT", "NUMBER", "(", ")", "[", "]", ",", "-"}

# Internal grammar representation. The public language matches the task prompt,
# while arglist/exprlist repetition is lowered into recursive tail rules.
GRAMMAR: Dict[str, List[List[str]]] = {
    "expr": [["call"], ["index"], ["tuple"], ["paren"], ["neg"], ["atom"]],
    "call": [["IDENT", "(", "arglist", ")"]],
    "index": [["IDENT", "[", "expr", "]"]],
    "tuple": [["(", "expr", ",", "exprlist", ")"]],
    "paren": [["(", "expr", ")"]],
    "neg": [["-", "expr"]],
    "atom": [["IDENT"], ["NUMBER"]],
    "arglist": [["expr", "arglist_tail"]],
    "arglist_tail": [[",", "expr", "arglist_tail"], []],
    "exprlist": [["expr", "exprlist_tail"]],
    "exprlist_tail": [[",", "expr", "exprlist_tail"], []],
}

START_RULE = "expr"
CHOICE_RULES = ("expr", "atom")

# Symbol order used by the tokenizer for parse-stack serialization.
SERIALIZABLE_SYMBOLS: List[str] = [
    "expr",
    "call",
    "index",
    "tuple",
    "paren",
    "neg",
    "atom",
    "arglist",
    "exprlist",
    "IDENT",
    "NUMBER",
    "(",
    ")",
    "[",
    "]",
    ",",
    "-",
]


def is_identifier_token(token: str) -> bool:
    return str(token) in IDENTIFIERS


def is_number_token(token: str) -> bool:
    return str(token) in NUMBERS


def matches_terminal(symbol: str, token: str) -> bool:
    if symbol == "IDENT":
        return is_identifier_token(token)
    if symbol == "NUMBER":
        return is_number_token(token)
    return str(symbol) == str(token)


def is_nonterminal(symbol: str) -> bool:
    return str(symbol) in GRAMMAR


def validate_token_sequence(tokens: Sequence[str]) -> None:
    for token in tokens:
        token_str = str(token)
        if token_str in PUNCTUATION:
            continue
        if is_identifier_token(token_str) or is_number_token(token_str):
            continue
        raise ValueError(f"unsupported terminal token: {token_str!r}")
