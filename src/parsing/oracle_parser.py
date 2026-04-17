"""Oracle PEG parser and action-conditioned parsing simulator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from parsing.grammar import (
    GRAMMAR,
    START_RULE,
    is_identifier_token,
    is_number_token,
    validate_token_sequence,
)

logger = logging.getLogger(__name__)


@dataclass
class ChoiceEvent:
    """A nondeterministic choice point."""

    rule: str
    cursor: int
    stack: List[str]
    alternative: int
    succeeded: bool


@dataclass
class BacktrackEvent:
    """Backtrack from a failed alternative."""

    rule: str
    cursor: int
    failed_alternative: int


@dataclass
class FailureState:
    cursor: int
    stack: List[str]


@dataclass
class TraceStep:
    event: Union[ChoiceEvent, BacktrackEvent]
    cursor: int
    stack: List[str]


@dataclass
class OracleParseResult:
    success: bool
    events: List[Union[ChoiceEvent, BacktrackEvent]]
    steps: List[TraceStep]
    final_cursor: int
    max_stack_depth: int


@dataclass(frozen=True)
class ParserAction:
    kind: str
    alternative: Optional[int] = None


@dataclass
class DecisionRequest:
    kind: str
    cursor: int
    stack: List[str]
    rule: Optional[str]
    available_alternatives: List[int] = field(default_factory=list)
    action_index: int = 0


@dataclass
class SimulationResult:
    status: str
    cursor: int
    stack: List[str]
    action_index: int
    rule: Optional[str] = None
    available_alternatives: List[int] = field(default_factory=list)
    reason: str = ""
    success: bool = False


@dataclass
class _Attempt:
    success: bool
    cursor: int
    failure: Optional[FailureState] = None


@dataclass
class _SimSuccess:
    cursor: int
    action_index: int


@dataclass
class _SimFail:
    failure: FailureState
    action_index: int


@dataclass
class _SimInvalid:
    reason: str
    action_index: int
    cursor: int
    stack: List[str]


_SimOutcome = Union[_SimSuccess, _SimFail, _SimInvalid, DecisionRequest]


def _compute_first_info(
    alt: Sequence[str],
    cache: Dict[str, Tuple[Set[str], bool]],
    visiting: Set[str],
) -> Tuple[Set[str], bool]:
    if len(alt) == 0:
        return set(), True

    first_tokens: Set[str] = set()
    for symbol in alt:
        symbol_str = str(symbol)
        if symbol_str not in GRAMMAR:
            first_tokens.add(symbol_str)
            return first_tokens, False

        if symbol_str in cache:
            nested_first, nested_nullable = cache[symbol_str]
        else:
            if symbol_str in visiting:
                raise ValueError(
                    f"left recursion unsupported while computing FIRST({symbol_str})"
                )
            nested_first = set()
            nested_nullable = False
            for nested_alt in GRAMMAR[symbol_str]:
                alt_first, alt_nullable = _compute_first_info(
                    nested_alt,
                    cache,
                    visiting | {symbol_str},
                )
                nested_first.update(alt_first)
                nested_nullable = bool(nested_nullable or alt_nullable)
            cache[symbol_str] = (set(nested_first), bool(nested_nullable))

        first_tokens.update(nested_first)
        if not nested_nullable:
            return first_tokens, False

    return first_tokens, True


def compute_first_tokens(alt: Sequence[str]) -> Set[str]:
    """Return terminal symbols that can start a match for an alternative."""

    first_tokens, _ = _compute_first_info(alt, cache={}, visiting=set())
    return set(first_tokens)


def _token_matches_first(token: Optional[str], first_tokens: Collection[str]) -> bool:
    if token is None:
        return False
    return any(
        _matches_terminal(str(first_token), token) for first_token in first_tokens
    )


_EXPR_ALT_SYMBOLS: List[List[str]] = [
    ["call"],
    ["index"],
    ["tuple"],
    ["paren"],
    ["neg"],
    ["atom"],
]
_EXPR_ALT_FIRST = [compute_first_tokens(alt) for alt in _EXPR_ALT_SYMBOLS]

_ATOM_ALT_SYMBOLS: List[List[str]] = [["IDENT"], ["NUMBER"]]
_ATOM_ALT_FIRST = [compute_first_tokens(alt) for alt in _ATOM_ALT_SYMBOLS]


def _matches_terminal(symbol: str, token: Optional[str]) -> bool:
    if token is None:
        return False
    if symbol == "IDENT":
        return is_identifier_token(token)
    if symbol == "NUMBER":
        return is_number_token(token)
    return str(symbol) == str(token)


class OracleBacktrackingParser:
    """Recursive-descent PEG parser with ordered alternatives and tracing."""

    def __init__(self, tokens: Sequence[str]):
        self.tokens = [str(tok) for tok in tokens]
        validate_token_sequence(self.tokens)
        self.events: List[Union[ChoiceEvent, BacktrackEvent]] = []
        self.steps: List[TraceStep] = []
        self.max_stack_depth = 0

    def parse(self) -> List[Union[ChoiceEvent, BacktrackEvent]]:
        return self.parse_detailed().events

    def parse_detailed(self) -> OracleParseResult:
        attempt = self._parse_expr(0, [])
        success = bool(attempt.success and int(attempt.cursor) == len(self.tokens))
        final_cursor = int(attempt.cursor)
        if attempt.success and final_cursor != len(self.tokens):
            success = False
        logger.debug(
            "oracle_parse success=%s input_len=%d events=%d max_stack_depth=%d",
            bool(success),
            int(len(self.tokens)),
            int(len(self.events)),
            int(self.max_stack_depth),
        )
        return OracleParseResult(
            success=bool(success),
            events=list(self.events),
            steps=list(self.steps),
            final_cursor=int(final_cursor),
            max_stack_depth=int(self.max_stack_depth),
        )

    def _observe_stack(self, stack: Sequence[str]) -> None:
        self.max_stack_depth = max(int(self.max_stack_depth), int(len(stack)))

    def _current_token(self, cursor: int) -> Optional[str]:
        if int(cursor) >= len(self.tokens):
            return None
        return self.tokens[int(cursor)]

    def _viable_alternatives(
        self, cursor: int, first_sets: Sequence[Collection[str]]
    ) -> List[int]:
        token = self._current_token(cursor)
        return [
            int(idx)
            for idx, first_tokens in enumerate(first_sets)
            if _token_matches_first(token, first_tokens)
        ]

    def _match(self, expected: str, cursor: int, stack: Sequence[str]) -> _Attempt:
        stack_copy = [str(sym) for sym in stack]
        self._observe_stack(stack_copy)
        token = self.tokens[int(cursor)] if int(cursor) < len(self.tokens) else None
        if _matches_terminal(str(expected), token):
            return _Attempt(success=True, cursor=int(cursor) + 1)
        return _Attempt(
            success=False,
            cursor=int(cursor),
            failure=FailureState(cursor=int(cursor), stack=stack_copy),
        )

    def _record_choice(
        self,
        rule: str,
        cursor: int,
        stack: Sequence[str],
        alternative: int,
    ) -> ChoiceEvent:
        stack_copy = [str(sym) for sym in stack]
        self._observe_stack(stack_copy)
        event = ChoiceEvent(
            rule=str(rule),
            cursor=int(cursor),
            stack=stack_copy,
            alternative=int(alternative),
            succeeded=False,
        )
        self.events.append(event)
        self.steps.append(TraceStep(event=event, cursor=int(cursor), stack=stack_copy))
        return event

    def _record_backtrack(
        self,
        rule: str,
        cursor: int,
        failed_alternative: int,
        failure: Optional[FailureState],
    ) -> None:
        event = BacktrackEvent(
            rule=str(rule),
            cursor=int(cursor),
            failed_alternative=int(failed_alternative),
        )
        snapshot = (
            FailureState(cursor=int(cursor), stack=[str(rule)])
            if failure is None
            else failure
        )
        self._observe_stack(snapshot.stack)
        self.events.append(event)
        self.steps.append(
            TraceStep(
                event=event,
                cursor=int(snapshot.cursor),
                stack=[str(sym) for sym in snapshot.stack],
            )
        )

    def _parse_choice(
        self,
        *,
        rule: str,
        cursor: int,
        stack: List[str],
        alternatives: List[Callable[[int, List[str]], _Attempt]],
        first_sets: Sequence[Collection[str]],
    ) -> _Attempt:
        rule_stack = [str(rule)] + list(stack)
        viable = self._viable_alternatives(int(cursor), first_sets)
        if len(viable) == 0:
            return _Attempt(
                success=False,
                cursor=int(cursor),
                failure=FailureState(cursor=int(cursor), stack=rule_stack),
            )
        if len(viable) == 1:
            return alternatives[int(viable[0])](int(cursor), list(stack))

        last_failure: Optional[FailureState] = None
        for alt_idx in viable:
            event = self._record_choice(
                str(rule), int(cursor), rule_stack, int(alt_idx)
            )
            attempt = alternatives[int(alt_idx)](int(cursor), list(stack))
            event.succeeded = bool(attempt.success)
            if attempt.success:
                return attempt
            last_failure = (
                attempt.failure if attempt.failure is not None else last_failure
            )
            self._record_backtrack(
                str(rule), int(cursor), int(alt_idx), attempt.failure
            )
        return _Attempt(
            success=False,
            cursor=int(cursor),
            failure=(
                last_failure
                if last_failure is not None
                else FailureState(cursor=int(cursor), stack=rule_stack)
            ),
        )

    def _parse_expr(self, cursor: int, stack: List[str]) -> _Attempt:
        return self._parse_choice(
            rule="expr",
            cursor=int(cursor),
            stack=list(stack),
            alternatives=[
                self._parse_call,
                self._parse_index,
                self._parse_tuple,
                self._parse_paren,
                self._parse_neg,
                self._parse_atom,
            ],
            first_sets=_EXPR_ALT_FIRST,
        )

    def _parse_atom(self, cursor: int, stack: List[str]) -> _Attempt:
        return self._parse_choice(
            rule="atom",
            cursor=int(cursor),
            stack=list(stack),
            alternatives=[self._parse_atom_ident, self._parse_atom_number],
            first_sets=_ATOM_ALT_FIRST,
        )

    def _parse_atom_ident(self, cursor: int, stack: List[str]) -> _Attempt:
        return self._match("IDENT", int(cursor), ["IDENT"] + list(stack))

    def _parse_atom_number(self, cursor: int, stack: List[str]) -> _Attempt:
        return self._match("NUMBER", int(cursor), ["NUMBER"] + list(stack))

    def _parse_call(self, cursor: int, stack: List[str]) -> _Attempt:
        attempt = self._match(
            "IDENT", cursor, ["IDENT", "(", "arglist", ")"] + list(stack)
        )
        if not attempt.success:
            return attempt
        attempt = self._match("(", attempt.cursor, ["(", "arglist", ")"] + list(stack))
        if not attempt.success:
            return attempt
        attempt = self._parse_arglist(attempt.cursor, [")"] + list(stack))
        if not attempt.success:
            return attempt
        return self._match(")", attempt.cursor, [")"] + list(stack))

    def _parse_index(self, cursor: int, stack: List[str]) -> _Attempt:
        attempt = self._match(
            "IDENT", cursor, ["IDENT", "[", "expr", "]"] + list(stack)
        )
        if not attempt.success:
            return attempt
        attempt = self._match("[", attempt.cursor, ["[", "expr", "]"] + list(stack))
        if not attempt.success:
            return attempt
        attempt = self._parse_expr(attempt.cursor, ["]"] + list(stack))
        if not attempt.success:
            return attempt
        return self._match("]", attempt.cursor, ["]"] + list(stack))

    def _parse_tuple(self, cursor: int, stack: List[str]) -> _Attempt:
        attempt = self._match(
            "(", cursor, ["(", "expr", ",", "exprlist", ")"] + list(stack)
        )
        if not attempt.success:
            return attempt
        attempt = self._parse_expr(attempt.cursor, [",", "exprlist", ")"] + list(stack))
        if not attempt.success:
            return attempt
        attempt = self._match(",", attempt.cursor, [",", "exprlist", ")"] + list(stack))
        if not attempt.success:
            return attempt
        attempt = self._parse_exprlist(attempt.cursor, [")"] + list(stack))
        if not attempt.success:
            return attempt
        return self._match(")", attempt.cursor, [")"] + list(stack))

    def _parse_paren(self, cursor: int, stack: List[str]) -> _Attempt:
        attempt = self._match("(", cursor, ["(", "expr", ")"] + list(stack))
        if not attempt.success:
            return attempt
        attempt = self._parse_expr(attempt.cursor, [")"] + list(stack))
        if not attempt.success:
            return attempt
        return self._match(")", attempt.cursor, [")"] + list(stack))

    def _parse_neg(self, cursor: int, stack: List[str]) -> _Attempt:
        attempt = self._match("-", cursor, ["-", "expr"] + list(stack))
        if not attempt.success:
            return attempt
        return self._parse_expr(attempt.cursor, list(stack))

    def _parse_arglist(self, cursor: int, stack: List[str]) -> _Attempt:
        current_stack = ["arglist"] + list(stack)
        attempt = self._parse_expr(cursor, current_stack)
        if not attempt.success:
            return attempt
        cur = int(attempt.cursor)
        while cur < len(self.tokens) and self.tokens[cur] == ",":
            comma_attempt = self._match(
                ",", cur, [",", "expr", "arglist"] + list(stack)
            )
            if not comma_attempt.success:
                return comma_attempt
            attempt = self._parse_expr(comma_attempt.cursor, current_stack)
            if not attempt.success:
                return attempt
            cur = int(attempt.cursor)
        return _Attempt(success=True, cursor=int(cur))

    def _parse_exprlist(self, cursor: int, stack: List[str]) -> _Attempt:
        current_stack = ["exprlist"] + list(stack)
        attempt = self._parse_expr(cursor, current_stack)
        if not attempt.success:
            return attempt
        cur = int(attempt.cursor)
        while cur < len(self.tokens) and self.tokens[cur] == ",":
            comma_attempt = self._match(
                ",", cur, [",", "expr", "exprlist"] + list(stack)
            )
            if not comma_attempt.success:
                return comma_attempt
            attempt = self._parse_expr(comma_attempt.cursor, current_stack)
            if not attempt.success:
                return attempt
            cur = int(attempt.cursor)
        return _Attempt(success=True, cursor=int(cur))


class PolicyParsingSimulator:
    """Replay the parser under an externally supplied action history."""

    def __init__(self, tokens: Sequence[str]):
        self.tokens = [str(tok) for tok in tokens]
        validate_token_sequence(self.tokens)

    def simulate(self, actions: Sequence[ParserAction]) -> SimulationResult:
        attempt = self._parse_expr(0, [], list(actions), 0)
        if isinstance(attempt, DecisionRequest):
            return SimulationResult(
                status="need_action",
                cursor=int(attempt.cursor),
                stack=[str(sym) for sym in attempt.stack],
                action_index=int(attempt.action_index),
                rule=attempt.rule,
                available_alternatives=[int(x) for x in attempt.available_alternatives],
                reason=str(attempt.kind),
            )
        if isinstance(attempt, _SimInvalid):
            return SimulationResult(
                status="invalid",
                cursor=int(attempt.cursor),
                stack=[str(sym) for sym in attempt.stack],
                action_index=int(attempt.action_index),
                reason=str(attempt.reason),
            )
        if isinstance(attempt, _SimFail):
            return SimulationResult(
                status="failed",
                cursor=int(attempt.failure.cursor),
                stack=[str(sym) for sym in attempt.failure.stack],
                action_index=int(attempt.action_index),
                reason="parse_failed",
                success=False,
            )
        success = bool(int(attempt.cursor) == len(self.tokens))
        if not success:
            return SimulationResult(
                status="failed",
                cursor=int(attempt.cursor),
                stack=[],
                action_index=int(attempt.action_index),
                reason="input_remaining",
                success=False,
            )
        return SimulationResult(
            status="parsed",
            cursor=int(attempt.cursor),
            stack=[],
            action_index=int(attempt.action_index),
            reason="parsed",
            success=True,
        )

    def _current_action(
        self, actions: Sequence[ParserAction], action_index: int
    ) -> Optional[ParserAction]:
        if int(action_index) >= len(actions):
            return None
        return actions[int(action_index)]

    def _current_token(self, cursor: int) -> Optional[str]:
        if int(cursor) >= len(self.tokens):
            return None
        return self.tokens[int(cursor)]

    def _viable_alternatives(
        self, cursor: int, first_sets: Sequence[Collection[str]]
    ) -> List[int]:
        token = self._current_token(cursor)
        return [
            int(idx)
            for idx, first_tokens in enumerate(first_sets)
            if _token_matches_first(token, first_tokens)
        ]

    def _match(
        self,
        expected: str,
        cursor: int,
        stack: Sequence[str],
        action_index: int,
    ) -> _SimOutcome:
        token = self.tokens[int(cursor)] if int(cursor) < len(self.tokens) else None
        if _matches_terminal(str(expected), token):
            return _SimSuccess(cursor=int(cursor) + 1, action_index=int(action_index))
        return _SimFail(
            failure=FailureState(cursor=int(cursor), stack=[str(sym) for sym in stack]),
            action_index=int(action_index),
        )

    def _parse_choice(
        self,
        *,
        rule: str,
        cursor: int,
        stack: List[str],
        alternatives: List[
            Callable[[int, List[str], Sequence[ParserAction], int], _SimOutcome]
        ],
        first_sets: Sequence[Collection[str]],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        rule_stack = [str(rule)] + list(stack)
        available = self._viable_alternatives(int(cursor), first_sets)
        if len(available) == 0:
            return _SimFail(
                failure=FailureState(cursor=int(cursor), stack=rule_stack),
                action_index=int(action_index),
            )
        if len(available) == 1:
            return alternatives[int(available[0])](
                int(cursor),
                list(stack),
                actions,
                int(action_index),
            )

        while True:
            if not available:
                return _SimFail(
                    failure=FailureState(cursor=int(cursor), stack=rule_stack),
                    action_index=int(action_index),
                )
            action = self._current_action(actions, action_index)
            if action is None:
                return DecisionRequest(
                    kind="choice",
                    cursor=int(cursor),
                    stack=rule_stack,
                    rule=str(rule),
                    available_alternatives=[int(x) for x in available],
                    action_index=int(action_index),
                )
            if str(action.kind) != "alt":
                return _SimInvalid(
                    reason=f"expected alt action at rule={rule}, got {action.kind}",
                    action_index=int(action_index),
                    cursor=int(cursor),
                    stack=rule_stack,
                )
            alt = -1 if action.alternative is None else int(action.alternative)
            if alt not in available:
                return _SimInvalid(
                    reason=f"alternative {alt} unavailable for rule={rule}",
                    action_index=int(action_index),
                    cursor=int(cursor),
                    stack=rule_stack,
                )
            next_result = alternatives[int(alt)](
                int(cursor),
                list(stack),
                actions,
                int(action_index) + 1,
            )
            if isinstance(next_result, (_SimSuccess, DecisionRequest, _SimInvalid)):
                return next_result
            failure = next_result.failure
            bt_action = self._current_action(actions, next_result.action_index)
            if bt_action is None:
                return DecisionRequest(
                    kind="backtrack",
                    cursor=int(failure.cursor),
                    stack=[str(sym) for sym in failure.stack],
                    rule=str(rule),
                    available_alternatives=[],
                    action_index=int(next_result.action_index),
                )
            if str(bt_action.kind) != "backtrack":
                return _SimInvalid(
                    reason=f"expected backtrack after failed alternative for rule={rule}",
                    action_index=int(next_result.action_index),
                    cursor=int(failure.cursor),
                    stack=[str(sym) for sym in failure.stack],
                )
            action_index = int(next_result.action_index) + 1
            available.remove(int(alt))

    def _parse_expr(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        return self._parse_choice(
            rule="expr",
            cursor=int(cursor),
            stack=list(stack),
            alternatives=[
                self._parse_call,
                self._parse_index,
                self._parse_tuple,
                self._parse_paren,
                self._parse_neg,
                self._parse_atom,
            ],
            first_sets=_EXPR_ALT_FIRST,
            actions=actions,
            action_index=int(action_index),
        )

    def _parse_atom(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        return self._parse_choice(
            rule="atom",
            cursor=int(cursor),
            stack=list(stack),
            alternatives=[self._parse_atom_ident, self._parse_atom_number],
            first_sets=_ATOM_ALT_FIRST,
            actions=actions,
            action_index=int(action_index),
        )

    def _parse_atom_ident(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        _ = actions
        return self._match("IDENT", cursor, ["IDENT"] + list(stack), action_index)

    def _parse_atom_number(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        _ = actions
        return self._match("NUMBER", cursor, ["NUMBER"] + list(stack), action_index)

    def _parse_call(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        attempt = self._match(
            "IDENT", cursor, ["IDENT", "(", "arglist", ")"] + list(stack), action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        attempt = self._match(
            "(",
            attempt.cursor,
            ["(", "arglist", ")"] + list(stack),
            attempt.action_index,
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        attempt = self._parse_arglist(
            attempt.cursor, [")"] + list(stack), actions, attempt.action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        return self._match(
            ")", attempt.cursor, [")"] + list(stack), attempt.action_index
        )

    def _parse_index(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        attempt = self._match(
            "IDENT", cursor, ["IDENT", "[", "expr", "]"] + list(stack), action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        attempt = self._match(
            "[", attempt.cursor, ["[", "expr", "]"] + list(stack), attempt.action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        attempt = self._parse_expr(
            attempt.cursor, ["]"] + list(stack), actions, attempt.action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        return self._match(
            "]", attempt.cursor, ["]"] + list(stack), attempt.action_index
        )

    def _parse_tuple(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        attempt = self._match(
            "(", cursor, ["(", "expr", ",", "exprlist", ")"] + list(stack), action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        attempt = self._parse_expr(
            attempt.cursor,
            [",", "exprlist", ")"] + list(stack),
            actions,
            attempt.action_index,
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        attempt = self._match(
            ",",
            attempt.cursor,
            [",", "exprlist", ")"] + list(stack),
            attempt.action_index,
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        attempt = self._parse_exprlist(
            attempt.cursor, [")"] + list(stack), actions, attempt.action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        return self._match(
            ")", attempt.cursor, [")"] + list(stack), attempt.action_index
        )

    def _parse_paren(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        attempt = self._match(
            "(", cursor, ["(", "expr", ")"] + list(stack), action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        attempt = self._parse_expr(
            attempt.cursor, [")"] + list(stack), actions, attempt.action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        return self._match(
            ")", attempt.cursor, [")"] + list(stack), attempt.action_index
        )

    def _parse_neg(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        attempt = self._match("-", cursor, ["-", "expr"] + list(stack), action_index)
        if not isinstance(attempt, _SimSuccess):
            return attempt
        return self._parse_expr(
            attempt.cursor, list(stack), actions, attempt.action_index
        )

    def _parse_arglist(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        attempt = self._parse_expr(
            cursor, ["arglist"] + list(stack), actions, action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        cur = int(attempt.cursor)
        act_idx = int(attempt.action_index)
        while cur < len(self.tokens) and self.tokens[cur] == ",":
            comma_attempt = self._match(
                ",", cur, [",", "expr", "arglist"] + list(stack), act_idx
            )
            if not isinstance(comma_attempt, _SimSuccess):
                return comma_attempt
            attempt = self._parse_expr(
                comma_attempt.cursor,
                ["arglist"] + list(stack),
                actions,
                comma_attempt.action_index,
            )
            if not isinstance(attempt, _SimSuccess):
                return attempt
            cur = int(attempt.cursor)
            act_idx = int(attempt.action_index)
        return _SimSuccess(cursor=int(cur), action_index=int(act_idx))

    def _parse_exprlist(
        self,
        cursor: int,
        stack: List[str],
        actions: Sequence[ParserAction],
        action_index: int,
    ) -> _SimOutcome:
        attempt = self._parse_expr(
            cursor, ["exprlist"] + list(stack), actions, action_index
        )
        if not isinstance(attempt, _SimSuccess):
            return attempt
        cur = int(attempt.cursor)
        act_idx = int(attempt.action_index)
        while cur < len(self.tokens) and self.tokens[cur] == ",":
            comma_attempt = self._match(
                ",", cur, [",", "expr", "exprlist"] + list(stack), act_idx
            )
            if not isinstance(comma_attempt, _SimSuccess):
                return comma_attempt
            attempt = self._parse_expr(
                comma_attempt.cursor,
                ["exprlist"] + list(stack),
                actions,
                comma_attempt.action_index,
            )
            if not isinstance(attempt, _SimSuccess):
                return attempt
            cur = int(attempt.cursor)
            act_idx = int(attempt.action_index)
        return _SimSuccess(cursor=int(cur), action_index=int(act_idx))


def oracle_parse(tokens: Sequence[str]) -> OracleParseResult:
    return OracleBacktrackingParser(tokens).parse_detailed()


def simulate_with_actions(
    tokens: Sequence[str], actions: Sequence[ParserAction]
) -> SimulationResult:
    return PolicyParsingSimulator(tokens).simulate(actions)


__all__ = [
    "BacktrackEvent",
    "ChoiceEvent",
    "compute_first_tokens",
    "DecisionRequest",
    "OracleBacktrackingParser",
    "OracleParseResult",
    "ParserAction",
    "PolicyParsingSimulator",
    "SimulationResult",
    "oracle_parse",
    "simulate_with_actions",
]
