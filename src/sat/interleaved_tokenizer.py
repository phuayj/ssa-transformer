"""Tokenizer for interleaved AR SAT solving with PROP scratchpad."""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SATInterleavedTokenizer:
    """Tokenizer for interleaved AR SAT solving with PROP scratchpad."""

    # Special tokens (same IDs as graph coloring for consistency)
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3

    # Section markers
    CLAUSE_START = 4
    SEARCH_START = 5
    STATE = 6
    PROP = 7
    ENDPROP = 8

    # Operators
    COLON = 9

    # Action/status tokens
    OK = 10
    SOLVED = 11
    FAILED = 12
    BACKJUMP = 13
    CONFLICT = 14

    # SAT-specific tokens
    TRUE_VAL = 15
    FALSE_VAL = 16
    UNASSIGNED = 17
    UNIT = 18
    SAT_OK = 19
    FREE = 20
    MASKED_DOMAIN = 21
    NEWLY_TRUE = 22
    NEWLY_FALSE = 23
    BIN = 24

    # UP-CoT tokens
    UP_T = 25
    END_UP_T = 26
    UP_F = 27
    END_UP_F = 28
    ROUND = 29

    # Structured token offsets
    POS_LIT_OFFSET = 30
    NEG_LIT_OFFSET = POS_LIT_OFFSET + 200
    CLAUSE_OFFSET = NEG_LIT_OFFSET + 200
    LEVEL_OFFSET = CLAUSE_OFFSET + 1200
    VAR_OFFSET = LEVEL_OFFSET + 200

    MAX_VARS = 200
    MAX_CLAUSES = 1200
    MAX_LEVELS = 200
    VOCAB_SIZE = VAR_OFFSET + MAX_VARS

    _SPECIAL_TOKEN_STRINGS: Dict[int, str] = {
        PAD: "[PAD]",
        BOS: "[BOS]",
        EOS: "[EOS]",
        SEP: "SEP",
        CLAUSE_START: "[CLAUSES]",
        SEARCH_START: "[SEARCH]",
        STATE: "STATE",
        PROP: "[PROP]",
        ENDPROP: "[/PROP]",
        COLON: ":",
        OK: "OK",
        SOLVED: "SOLVED",
        FAILED: "FAILED",
        BACKJUMP: "BJ",
        CONFLICT: "CONFLICT",
        TRUE_VAL: "T",
        FALSE_VAL: "F",
        UNASSIGNED: "U",
        UNIT: "UNIT",
        SAT_OK: "SAT_OK",
        FREE: "FREE",
        MASKED_DOMAIN: "?",
        NEWLY_TRUE: "NT",
        NEWLY_FALSE: "NF",
        BIN: "BIN",
        UP_T: "[UP_T]",
        END_UP_T: "[/UP_T]",
        UP_F: "[UP_F]",
        END_UP_F: "[/UP_F]",
        ROUND: "[R]",
    }

    @staticmethod
    def _lit_var(lit: int) -> int:
        lit = int(lit)
        if lit == 0:
            raise ValueError("literal cannot be 0")
        return int(abs(lit) - 1)

    @staticmethod
    def _lit_sign(lit: int) -> int:
        return 1 if int(lit) > 0 else -1

    @staticmethod
    def pos_lit_token(var_id: int) -> int:
        """Token for positive literal +v{var_id}. var_id is 0-based."""
        var_id = int(var_id)
        if var_id < 0 or var_id >= SATInterleavedTokenizer.MAX_VARS:
            raise ValueError(f"var_id out of range: {var_id}")
        return SATInterleavedTokenizer.POS_LIT_OFFSET + var_id

    @staticmethod
    def neg_lit_token(var_id: int) -> int:
        """Token for negative literal -v{var_id}."""
        var_id = int(var_id)
        if var_id < 0 or var_id >= SATInterleavedTokenizer.MAX_VARS:
            raise ValueError(f"var_id out of range: {var_id}")
        return SATInterleavedTokenizer.NEG_LIT_OFFSET + var_id

    @staticmethod
    def lit_token(lit: int) -> int:
        """Token for a DIMACS literal (±(var+1))."""
        lit = int(lit)
        if lit == 0:
            raise ValueError("literal cannot be 0")
        if lit > 0:
            return SATInterleavedTokenizer.POS_LIT_OFFSET + (lit - 1)
        return SATInterleavedTokenizer.NEG_LIT_OFFSET + (abs(lit) - 1)

    @staticmethod
    def clause_token(clause_id: int) -> int:
        clause_id = int(clause_id)
        if clause_id < 0 or clause_id >= SATInterleavedTokenizer.MAX_CLAUSES:
            raise ValueError(f"clause_id out of range: {clause_id}")
        return SATInterleavedTokenizer.CLAUSE_OFFSET + clause_id

    @staticmethod
    def level_token(level: int) -> int:
        level = int(level)
        if level < 0 or level >= SATInterleavedTokenizer.MAX_LEVELS:
            raise ValueError(f"level out of range: {level}")
        return SATInterleavedTokenizer.LEVEL_OFFSET + level

    @staticmethod
    def var_token(var_id: int) -> int:
        """Token for unsigned variable reference (for STATE section)."""
        var_id = int(var_id)
        if var_id < 0 or var_id >= SATInterleavedTokenizer.MAX_VARS:
            raise ValueError(f"var_id out of range: {var_id}")
        return SATInterleavedTokenizer.VAR_OFFSET + var_id

    def decode_token(self, token_id: int) -> str:
        """Human-readable string for a token."""
        token_id = int(token_id)
        if token_id in self._SPECIAL_TOKEN_STRINGS:
            return self._SPECIAL_TOKEN_STRINGS[token_id]
        if self.POS_LIT_OFFSET <= token_id < self.NEG_LIT_OFFSET:
            var_id = token_id - self.POS_LIT_OFFSET
            return f"+v{int(var_id)}"
        if self.NEG_LIT_OFFSET <= token_id < self.CLAUSE_OFFSET:
            var_id = token_id - self.NEG_LIT_OFFSET
            return f"-v{int(var_id)}"
        if self.CLAUSE_OFFSET <= token_id < self.LEVEL_OFFSET:
            clause_id = token_id - self.CLAUSE_OFFSET
            return f"C{int(clause_id)}"
        if self.LEVEL_OFFSET <= token_id < self.VAR_OFFSET:
            level = token_id - self.LEVEL_OFFSET
            return f"L{int(level)}"
        if self.VAR_OFFSET <= token_id < self.VOCAB_SIZE:
            var_id = token_id - self.VAR_OFFSET
            return f"v{int(var_id)}"
        raise ValueError(f"Unknown token id: {token_id}")

    def decode_sequence(self, token_ids: List[int]) -> str:
        return " ".join(self.decode_token(token_id) for token_id in token_ids)

    def _validate_clauses(self, clauses: List[Tuple[int, ...]], num_vars: int) -> None:
        num_vars = int(num_vars)
        if num_vars <= 0 or num_vars > self.MAX_VARS:
            raise ValueError(f"num_vars out of range: {num_vars}")
        if len(clauses) == 0:
            raise ValueError("clauses must be non-empty")
        if len(clauses) > self.MAX_CLAUSES:
            raise ValueError(f"too many clauses: {len(clauses)}")
        for cid, clause in enumerate(clauses):
            if len(clause) == 0:
                raise ValueError(f"clause {cid} must be non-empty")
            for lit in clause:
                var_id = self._lit_var(int(lit))
                if var_id < 0 or var_id >= num_vars:
                    raise ValueError(
                        f"clause {cid} literal out of range: {lit} (num_vars={num_vars})"
                    )

    def _value_token(self, value: int) -> int:
        v = int(value)
        if v == 1:
            return int(self.TRUE_VAL)
        if v == -1:
            return int(self.FALSE_VAL)
        raise ValueError(f"assignment value must be -1/1, got {value}")

    def _verdict_token(self, verdict: str) -> int:
        if verdict == "ok":
            return int(self.SAT_OK)
        if verdict == "unit":
            return int(self.UNIT)
        if verdict == "conflict":
            return int(self.CONFLICT)
        raise ValueError(f"unknown verdict: {verdict}")

    def build_clause_prefix(
        self, clauses: List[Tuple[int, ...]], num_vars: int
    ) -> List[int]:
        tokens: List[int] = [int(self.BOS), int(self.CLAUSE_START)]
        self._validate_clauses(clauses, num_vars)
        for cid, clause in enumerate(clauses):
            tokens.append(self.clause_token(cid))
            tokens.append(int(self.COLON))
            for lit in clause:
                tokens.append(self.lit_token(lit))
            tokens.append(int(self.SEP))
        tokens.append(int(self.SEARCH_START))
        logger.debug(
            "build_clause_prefix: num_vars=%d clauses=%d tokens_len=%d",
            int(num_vars),
            int(len(clauses)),
            int(len(tokens)),
        )
        return tokens

    def build_prop_evidence(
        self,
        clause_id: int,
        clauses: List[Tuple[int, ...]],
        assignment: np.ndarray,
    ) -> Tuple[List[int], str]:
        """Build PROP evidence tokens for checking a clause."""
        clause_id = int(clause_id)
        if clause_id < 0 or clause_id >= len(clauses):
            raise ValueError(f"clause_id out of range: {clause_id}")
        if clause_id >= self.MAX_CLAUSES:
            raise ValueError(f"clause_id exceeds max: {clause_id}")

        clause = clauses[clause_id]
        asgn = np.asarray(assignment)

        tokens: List[int] = []

        # Phase 1: Copy clause literal list
        tokens.append(self.clause_token(clause_id))
        tokens.append(int(self.COLON))
        for lit in clause:
            tokens.append(self.lit_token(lit))
        tokens.append(int(self.SEP))

        # Phase 2: Copy each literal's current value
        num_true = 0
        num_false = 0
        num_unassigned = 0

        for lit in clause:
            tokens.append(self.lit_token(lit))
            var = self._lit_var(int(lit))
            if var < 0 or var >= int(asgn.shape[0]):
                raise ValueError(f"assignment missing var {var} for lit {lit}")
            val = int(asgn[int(var)])
            if val == 0:
                tokens.append(int(self.UNASSIGNED))
                num_unassigned += 1
            elif val in (-1, 1):
                lit_sign = self._lit_sign(int(lit))
                if val == lit_sign:
                    tokens.append(int(self.TRUE_VAL))
                    num_true += 1
                else:
                    tokens.append(int(self.FALSE_VAL))
                    num_false += 1
            else:
                raise ValueError(f"invalid assignment value {val} for var {var}")
        tokens.append(int(self.SEP))

        # Phase 3: Verdict
        if num_true > 0:
            tokens.append(int(self.SAT_OK))
            verdict = "ok"
        elif num_unassigned == 0:
            tokens.append(int(self.CONFLICT))
            verdict = "conflict"
        elif num_unassigned == 1:
            tokens.append(int(self.UNIT))
            verdict = "unit"
        else:
            tokens.append(int(self.SAT_OK))
            verdict = "ok"

        logger.debug(
            "build_prop_evidence: clause_id=%d lits=%d true=%d false=%d unassigned=%d verdict=%s tokens_len=%d",
            clause_id,
            int(len(clause)),
            int(num_true),
            int(num_false),
            int(num_unassigned),
            verdict,
            int(len(tokens)),
        )

        return tokens, verdict

    def _find_most_constrained_clause(
        self,
        clauses: List[Tuple[int, ...]],
        assignment: np.ndarray,
    ) -> int:
        """Find clause ID of most constrained unsatisfied clause."""
        if len(clauses) == 0:
            raise ValueError("clauses must be non-empty")
        asgn = np.asarray(assignment)

        best_id = 0
        best_unassigned = len(clauses[0]) + 1
        unsatisfied = 0

        for cid, clause in enumerate(clauses):
            num_true = 0
            num_unassigned = 0
            for lit in clause:
                var = self._lit_var(int(lit))
                if var < 0 or var >= int(asgn.shape[0]):
                    raise ValueError(f"assignment missing var {var} for lit {lit}")
                val = int(asgn[int(var)])
                if val == 0:
                    num_unassigned += 1
                else:
                    lit_sign = self._lit_sign(int(lit))
                    if val == lit_sign:
                        num_true += 1

            if num_true > 0:
                continue

            unsatisfied += 1
            if num_unassigned < best_unassigned:
                best_unassigned = num_unassigned
                best_id = cid

        logger.debug(
            "_find_most_constrained_clause: clauses=%d unsatisfied=%d best_id=%d best_unassigned=%d",
            int(len(clauses)),
            int(unsatisfied),
            int(best_id),
            int(best_unassigned),
        )

        return int(best_id)

    def build_interleaved_trace(
        self,
        clauses: List[Tuple[int, ...]],
        events: List[dict],
        num_vars: int,
    ) -> List[int]:
        """Build full interleaved trace with PROP scratchpad (View A)."""
        num_vars = int(num_vars)
        tokens = self.build_clause_prefix(clauses, num_vars)

        assign_steps = 0
        conflict_steps = 0
        state_candidates = 0
        logger.debug(
            "build_interleaved_trace: num_vars=%d clauses=%d events=%d",
            int(num_vars),
            int(len(clauses)),
            int(len(events)),
        )

        for event in events:
            event_type = str(event.get("type"))

            if event_type in ("assign", "conflict"):
                sorted_candidates = event.get("sorted_candidates")
                if sorted_candidates is None:
                    raise ValueError(f"{event_type} event missing sorted_candidates")
                state_candidates += int(len(sorted_candidates))
                tokens.append(int(self.STATE))
                for var_id in sorted_candidates:
                    tokens.append(self.var_token(int(var_id)))
                    tokens.append(int(self.UNASSIGNED))
                tokens.append(int(self.SEP))

                asgn_raw = event.get("current_assignment")
                if asgn_raw is None:
                    raise ValueError(f"{event_type} event missing current_assignment")
                asgn = np.asarray(asgn_raw)
                if int(asgn.shape[0]) != int(num_vars):
                    raise ValueError(
                        f"current_assignment length {int(asgn.shape[0])} != num_vars {num_vars}"
                    )
                checked = self._find_most_constrained_clause(clauses, asgn)
                prop_inner, _verdict = self.build_prop_evidence(checked, clauses, asgn)
                tokens.append(int(self.PROP))
                tokens.extend(prop_inner)
                tokens.append(int(self.ENDPROP))

                if event_type == "assign":
                    assign_steps += 1
                    var_raw = event.get("var")
                    if var_raw is None:
                        raise ValueError("assign event missing var")
                    var_id = int(var_raw)
                    if var_id < 0 or var_id >= num_vars:
                        raise ValueError(f"assign var out of range: {var_id}")
                    value = event.get("value")
                    if value is None:
                        raise ValueError("assign event missing value")
                    value_token = self._value_token(int(value))
                    tokens.append(self.var_token(var_id))
                    tokens.append(int(value_token))
                    tokens.append(int(self.OK))
                    # Shadow assignment anchor for easy lookup
                    tokens.append(self.var_token(var_id))
                    tokens.append(int(value_token))
                else:
                    conflict_steps += 1
                    clause_raw = event.get("clause_id")
                    if clause_raw is None:
                        raise ValueError("conflict event missing clause_id")
                    clause_id = int(clause_raw)
                    if clause_id < 0 or clause_id >= len(clauses):
                        raise ValueError(
                            f"conflict clause_id out of range: {clause_id}"
                        )
                    backjump_level = event.get("backjump_level")
                    if backjump_level is None:
                        raise ValueError("conflict event missing backjump_level")
                    tokens.append(int(self.CONFLICT))
                    tokens.append(self.clause_token(clause_id))
                    tokens.append(int(self.BACKJUMP))
                    tokens.append(self.level_token(int(backjump_level)))
                continue

            if event_type == "solved":
                tokens.append(int(self.SOLVED))
                continue
            if event_type == "failed":
                tokens.append(int(self.FAILED))
                continue

            raise ValueError(f"Unknown event type: {event_type}")

        tokens.append(int(self.EOS))
        logger.debug(
            "build_interleaved_trace done: tokens_len=%d assign_steps=%d conflict_steps=%d state_candidates=%d",
            int(len(tokens)),
            int(assign_steps),
            int(conflict_steps),
            int(state_candidates),
        )
        return tokens

    def build_verdict_only_trace(
        self,
        clauses: List[Tuple[int, ...]],
        events: List[dict],
        num_vars: int,
    ) -> List[int]:
        """Build interleaved trace with clause verdict only (View B)."""
        num_vars = int(num_vars)
        tokens = self.build_clause_prefix(clauses, num_vars)

        assign_steps = 0
        conflict_steps = 0
        state_candidates = 0
        logger.debug(
            "build_verdict_only_trace: num_vars=%d clauses=%d events=%d",
            int(num_vars),
            int(len(clauses)),
            int(len(events)),
        )

        for event in events:
            event_type = str(event.get("type"))

            if event_type in ("assign", "conflict"):
                sorted_candidates = event.get("sorted_candidates")
                if sorted_candidates is None:
                    raise ValueError(f"{event_type} event missing sorted_candidates")
                state_candidates += int(len(sorted_candidates))
                tokens.append(int(self.STATE))
                for var_id in sorted_candidates:
                    tokens.append(self.var_token(int(var_id)))
                    tokens.append(int(self.UNASSIGNED))
                tokens.append(int(self.SEP))

                asgn_raw = event.get("current_assignment")
                if asgn_raw is None:
                    raise ValueError(f"{event_type} event missing current_assignment")
                asgn = np.asarray(asgn_raw)
                if int(asgn.shape[0]) != int(num_vars):
                    raise ValueError(
                        f"current_assignment length {int(asgn.shape[0])} != num_vars {num_vars}"
                    )
                checked = self._find_most_constrained_clause(clauses, asgn)
                _prop_inner, verdict = self.build_prop_evidence(checked, clauses, asgn)
                tokens.append(self._verdict_token(verdict))

                if event_type == "assign":
                    assign_steps += 1
                    var_raw = event.get("var")
                    if var_raw is None:
                        raise ValueError("assign event missing var")
                    var_id = int(var_raw)
                    if var_id < 0 or var_id >= num_vars:
                        raise ValueError(f"assign var out of range: {var_id}")
                    value = event.get("value")
                    if value is None:
                        raise ValueError("assign event missing value")
                    value_token = self._value_token(int(value))
                    tokens.append(self.var_token(var_id))
                    tokens.append(int(value_token))
                    tokens.append(int(self.OK))
                    tokens.append(self.var_token(var_id))
                    tokens.append(int(value_token))
                else:
                    conflict_steps += 1
                    clause_raw = event.get("clause_id")
                    if clause_raw is None:
                        raise ValueError("conflict event missing clause_id")
                    clause_id = int(clause_raw)
                    if clause_id < 0 or clause_id >= len(clauses):
                        raise ValueError(
                            f"conflict clause_id out of range: {clause_id}"
                        )
                    backjump_level = event.get("backjump_level")
                    if backjump_level is None:
                        raise ValueError("conflict event missing backjump_level")
                    tokens.append(int(self.CONFLICT))
                    tokens.append(self.clause_token(clause_id))
                    tokens.append(int(self.BACKJUMP))
                    tokens.append(self.level_token(int(backjump_level)))
                continue

            if event_type == "solved":
                tokens.append(int(self.SOLVED))
                continue
            if event_type == "failed":
                tokens.append(int(self.FAILED))
                continue

            raise ValueError(f"Unknown event type: {event_type}")

        tokens.append(int(self.EOS))
        logger.debug(
            "build_verdict_only_trace done: tokens_len=%d assign_steps=%d conflict_steps=%d state_candidates=%d",
            int(len(tokens)),
            int(assign_steps),
            int(conflict_steps),
            int(state_candidates),
        )
        return tokens

    def build_loss_mask(self, token_ids: List[int]) -> List[bool]:
        """Loss mask for View A (PROP) or View B (verdict-only) traces."""
        mask = [False] * int(len(token_ids))
        if not token_ids:
            return mask

        search_idx = None
        for idx, tok in enumerate(token_ids):
            if int(tok) == int(self.SEARCH_START):
                search_idx = int(idx)
                break
        if search_idx is None:
            return mask

        i = int(search_idx) + 1
        while i < len(token_ids):
            tok = int(token_ids[i])
            if tok in (int(self.SOLVED), int(self.FAILED)):
                mask[i] = True
                i += 1
                continue
            if tok == int(self.EOS):
                i += 1
                continue

            if tok == int(self.STATE):
                i += 1
                while i < len(token_ids) and int(token_ids[i]) != int(self.SEP):
                    i += 1
                i += 1
                if i >= len(token_ids):
                    break

                if int(token_ids[i]) == int(self.PROP):
                    mask[i] = False
                    i += 1
                    while i < len(token_ids) and int(token_ids[i]) != int(self.ENDPROP):
                        mask[i] = True
                        i += 1
                    if i < len(token_ids):
                        mask[i] = False
                        i += 1
                elif int(token_ids[i]) in (
                    int(self.CONFLICT),
                    int(self.UNIT),
                    int(self.SAT_OK),
                ):
                    mask[i] = True
                    i += 1
                else:
                    i += 1

                if i >= len(token_ids):
                    break

                action_tok = int(token_ids[i])
                if self.VAR_OFFSET <= action_tok < self.VAR_OFFSET + self.MAX_VARS:
                    mask[i] = True
                    i += 1
                    if i >= len(token_ids):
                        break
                    if int(token_ids[i]) not in (
                        int(self.TRUE_VAL),
                        int(self.FALSE_VAL),
                    ):
                        raise ValueError("assign action missing TRUE/FALSE token")
                    mask[i] = True
                    i += 1
                    if i < len(token_ids) and int(token_ids[i]) == int(self.OK):
                        mask[i] = False
                        i += 1
                    if i < len(token_ids):
                        shadow_tok = int(token_ids[i])
                        if (
                            self.VAR_OFFSET
                            <= shadow_tok
                            < self.VAR_OFFSET + self.MAX_VARS
                        ):
                            mask[i] = False
                            i += 1
                            if i < len(token_ids) and int(token_ids[i]) in (
                                int(self.TRUE_VAL),
                                int(self.FALSE_VAL),
                            ):
                                mask[i] = False
                                i += 1
                    continue

                if action_tok == int(self.CONFLICT):
                    mask[i] = True
                    i += 1
                    if i < len(token_ids):
                        mask[i] = True
                        i += 1
                    if i < len(token_ids):
                        if int(token_ids[i]) != int(self.BACKJUMP):
                            raise ValueError("conflict action missing BACKJUMP")
                        mask[i] = True
                        i += 1
                    if i < len(token_ids):
                        mask[i] = True
                        i += 1
                    continue

                i += 1
                continue

            i += 1

        if token_ids and int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if token_ids and int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        true_count = int(sum(1 for value in mask if value))
        logger.debug(
            "build_loss_mask: tokens_len=%d true_tokens=%d",
            int(len(token_ids)),
            int(true_count),
        )
        return mask


def serialize_annotated_state(
    clauses: List[Tuple[int, ...]],
    assignment: np.ndarray,
    num_vars: int,
    tokenizer: SATInterleavedTokenizer,
    include_conflict: bool = True,
    include_unit: bool = True,
    include_binary: bool = True,
) -> Tuple[List[int], bool]:
    """Serialize state with conflict/unit/binary clause annotations.

    Args:
        clauses: DIMACS clauses with 1-indexed signed literals.
        assignment: Array of shape [num_vars] with values {-1, 0, 1}.
        num_vars: Number of variables.
        tokenizer: SATInterleavedTokenizer instance used for token mapping.
        include_conflict: Append CONFLICT token when immediate conflict exists.
        include_unit: Append UNIT section with currently unit literals.
        include_binary: Append BIN section with unresolved binary clauses.

    Returns:
        (tokens, is_conflict): Serialized STATE block and conflict flag.
    """
    num_vars = int(num_vars)
    asgn = np.asarray(assignment)
    if int(asgn.shape[0]) != num_vars:
        raise ValueError(
            f"assignment length {int(asgn.shape[0])} != num_vars {num_vars}"
        )

    tokens: List[int] = [int(tokenizer.STATE)]
    is_conflict = False
    unit_literals: Set[int] = set()
    binary_pairs: Set[Tuple[int, int]] = set()

    for clause in clauses:
        unassigned_lits: List[int] = []
        clause_satisfied = False

        for lit_raw in clause:
            lit = int(lit_raw)
            var_id = abs(lit) - 1
            if var_id < 0 or var_id >= num_vars:
                raise ValueError(
                    f"literal out of range for num_vars={num_vars}: {lit}"
                )

            val = int(asgn[var_id])
            if val not in (-1, 0, 1):
                raise ValueError(f"invalid assignment value {val} for var {var_id}")

            if val == 0:
                unassigned_lits.append(lit)
                continue

            if (lit > 0 and val == 1) or (lit < 0 and val == -1):
                clause_satisfied = True
                break

        if clause_satisfied:
            continue

        n_unassigned = int(len(unassigned_lits))
        if n_unassigned == 0:
            is_conflict = True
        elif n_unassigned == 1 and include_unit:
            unit_literals.add(int(unassigned_lits[0]))
        elif n_unassigned == 2 and include_binary:
            l0 = int(unassigned_lits[0])
            l1 = int(unassigned_lits[1])
            pair = tuple(sorted((l0, l1), key=lambda lit: (abs(lit), lit)))
            binary_pairs.add((int(pair[0]), int(pair[1])))

    # Existing STATE format: list unassigned variables as (v_i, U) pairs.
    for var_id in range(num_vars):
        if int(asgn[var_id]) != 0:
            continue
        tokens.append(int(tokenizer.var_token(var_id)))
        tokens.append(int(tokenizer.UNASSIGNED))

    if is_conflict and include_conflict:
        tokens.append(int(tokenizer.CONFLICT))

    if include_unit and unit_literals:
        tokens.append(int(tokenizer.UNIT))
        for lit in sorted(unit_literals, key=lambda x: (abs(x), x)):
            tokens.append(int(tokenizer.lit_token(int(lit))))

    if include_binary and binary_pairs:
        tokens.append(int(tokenizer.BIN))
        for lit0, lit1 in sorted(binary_pairs, key=lambda p: (abs(p[0]), p[0], abs(p[1]), p[1])):
            tokens.append(int(tokenizer.lit_token(int(lit0))))
            tokens.append(int(tokenizer.lit_token(int(lit1))))

    tokens.append(int(tokenizer.SEP))

    logger.debug(
        "serialize_annotated_state: clauses=%d num_vars=%d unassigned_vars=%d conflict=%s units=%d binaries=%d tokens_len=%d",
        int(len(clauses)),
        int(num_vars),
        int(np.count_nonzero(asgn == 0)),
        bool(is_conflict),
        int(len(unit_literals)),
        int(len(binary_pairs)),
        int(len(tokens)),
    )

    return tokens, bool(is_conflict)
