"""Tokenizer for SAT CDCL conflict analysis sequences.

Serializes SAT conflict states and 1-UIP resolution scratchpads into tokens.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SATCDCLTokenizer:
    """Tokenizer for serialized SAT conflict states and scratchpads."""

    # Special tokens
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3

    # Section markers
    CLAUSE_SEC = 4
    TRAIL_SEC = 5
    CONFLICT_SEC = 6
    THINK_SEC = 7
    TARGET_SEC = 8

    # Keywords
    COLON = 9
    ARROW = 10
    DEC_KW = 11
    REAS_KW = 12
    PIVOT_KW = 13
    UIP_KW = 14
    LEARN_KW = 15
    BJ_KW = 16
    UNSAT_KW = 17
    AT_KW = 18
    STEP_KW = 19
    HASH_KW = 20

    # Structured token offsets
    POS_LIT_OFFSET = 30
    NEG_LIT_OFFSET = 130
    CLAUSE_OFFSET = 230
    LEVEL_OFFSET = 530
    INDEX_OFFSET = 630

    MAX_VARS = 100
    MAX_CLAUSES = 300
    MAX_LEVELS = 100
    MAX_TRAIL = 150
    VOCAB_SIZE = 780

    _SPECIAL_TOKEN_STRINGS: Dict[int, str] = {
        PAD: "[PAD]",
        BOS: "[BOS]",
        EOS: "[EOS]",
        SEP: "SEP",
        CLAUSE_SEC: "[CLAUSES]",
        TRAIL_SEC: "[TRAIL]",
        CONFLICT_SEC: "[CONFLICT]",
        THINK_SEC: "[THINK]",
        TARGET_SEC: "[TARGET]",
        COLON: ":",
        ARROW: "->",
        DEC_KW: "DEC",
        REAS_KW: "REAS",
        PIVOT_KW: "P",
        UIP_KW: "UIP",
        LEARN_KW: "LEARN",
        BJ_KW: "BJ",
        UNSAT_KW: "UNSAT",
        AT_KW: "@",
        STEP_KW: "R",
        HASH_KW: "#",
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

    @classmethod
    def pos_lit_token(cls, var_id: int) -> int:
        """Token for positive literal of variable var_id (0-based)."""
        var_id = int(var_id)
        if var_id < 0 or var_id >= cls.MAX_VARS:
            raise ValueError(f"var_id out of range: {var_id}")
        return cls.POS_LIT_OFFSET + var_id

    @classmethod
    def neg_lit_token(cls, var_id: int) -> int:
        """Token for negative literal of variable var_id (0-based)."""
        var_id = int(var_id)
        if var_id < 0 or var_id >= cls.MAX_VARS:
            raise ValueError(f"var_id out of range: {var_id}")
        return cls.NEG_LIT_OFFSET + var_id

    @classmethod
    def lit_token(cls, lit: int) -> int:
        """Encode a literal ±(var+1) into a token."""
        var_id = cls._lit_var(lit)
        return cls.pos_lit_token(var_id) if int(lit) > 0 else cls.neg_lit_token(var_id)

    @classmethod
    def clause_token(cls, clause_id: int) -> int:
        clause_id = int(clause_id)
        if clause_id < 0 or clause_id >= cls.MAX_CLAUSES:
            raise ValueError(f"clause_id out of range: {clause_id}")
        return cls.CLAUSE_OFFSET + clause_id

    @classmethod
    def level_token(cls, level: int) -> int:
        level = int(level)
        if level < 0 or level >= cls.MAX_LEVELS:
            raise ValueError(f"level out of range: {level}")
        return cls.LEVEL_OFFSET + level

    @classmethod
    def index_token(cls, idx: int) -> int:
        idx = int(idx)
        if idx < 0 or idx >= cls.MAX_TRAIL:
            raise ValueError(f"index out of range: {idx}")
        return cls.INDEX_OFFSET + idx

    def decode_token(self, token_id: int) -> str:
        """Convert token ID to a human-readable string."""
        token_id = int(token_id)
        if token_id in self._SPECIAL_TOKEN_STRINGS:
            return self._SPECIAL_TOKEN_STRINGS[token_id]
        if self.POS_LIT_OFFSET <= token_id < self.NEG_LIT_OFFSET:
            var_id = token_id - self.POS_LIT_OFFSET
            return f"v{int(var_id) + 1}"
        if self.NEG_LIT_OFFSET <= token_id < self.CLAUSE_OFFSET:
            var_id = token_id - self.NEG_LIT_OFFSET
            return f"~v{int(var_id) + 1}"
        if self.CLAUSE_OFFSET <= token_id < self.LEVEL_OFFSET:
            clause_id = token_id - self.CLAUSE_OFFSET
            return f"C{int(clause_id)}"
        if self.LEVEL_OFFSET <= token_id < self.INDEX_OFFSET:
            level = token_id - self.LEVEL_OFFSET
            return f"LV{int(level)}"
        if self.INDEX_OFFSET <= token_id < self.VOCAB_SIZE:
            idx = token_id - self.INDEX_OFFSET
            return f"I{int(idx)}"
        raise ValueError(f"Unknown token id: {token_id}")

    def decode_sequence(self, token_ids: List[int]) -> str:
        """Convert token sequence to a human-readable string for debugging."""
        return " ".join(self.decode_token(token_id) for token_id in token_ids)

    def _validate_inputs(
        self,
        clauses,
        trail: List[int],
        trail_levels: List[int],
        var_reason: List[Optional[int]],
        conflict_clause_id: int,
        num_vars: int,
    ) -> None:
        num_vars = int(num_vars)
        if num_vars <= 0 or num_vars > self.MAX_VARS:
            raise ValueError(f"num_vars out of range: {num_vars}")

        if len(clauses) == 0:
            raise ValueError("clauses must be non-empty")
        if len(clauses) > self.MAX_CLAUSES:
            raise ValueError(f"too many clauses: {len(clauses)}")

        for cid, clause in enumerate(clauses):
            if len(clause) != 3:
                raise ValueError(f"clause {cid} must have length 3")
            for lit in clause:
                var_id = self._lit_var(int(lit))
                if var_id < 0 or var_id >= num_vars:
                    raise ValueError(
                        f"clause {cid} literal out of range: {lit} (num_vars={num_vars})"
                    )

        if len(trail) != len(trail_levels):
            raise ValueError("trail and trail_levels length mismatch")
        if len(trail) > self.MAX_TRAIL:
            raise ValueError(f"trail too long: {len(trail)}")

        seen_vars = set()
        for idx, lit in enumerate(trail):
            var_id = self._lit_var(int(lit))
            if var_id < 0 or var_id >= num_vars:
                raise ValueError(f"trail literal out of range: {lit}")
            if var_id in seen_vars:
                raise ValueError(f"duplicate assignment for var {var_id} in trail")
            seen_vars.add(int(var_id))
            lvl = int(trail_levels[int(idx)])
            if lvl < 0 or lvl >= self.MAX_LEVELS:
                raise ValueError(f"trail level out of range: {lvl}")

        if len(var_reason) != num_vars:
            raise ValueError("var_reason length mismatch")
        for var, reason in enumerate(var_reason):
            if reason is None:
                continue
            rid = int(reason)
            if rid < 0 or rid >= len(clauses):
                raise ValueError(f"var_reason out of range: var={var} reason={rid}")

        cid = int(conflict_clause_id)
        if cid < 0 or cid >= len(clauses):
            raise ValueError(f"conflict_clause_id out of range: {cid}")

    def _validate_1uip_inputs(
        self,
        clauses,
        trail: List[int],
        trail_levels: List[int],
        var_reason: List[Optional[int]],
        conflict_clause_id: int,
        num_vars: int,
    ) -> None:
        num_vars = int(num_vars)
        if num_vars <= 0:
            raise ValueError(f"num_vars out of range: {num_vars}")

        if len(clauses) == 0:
            raise ValueError("clauses must be non-empty")

        cid = int(conflict_clause_id)
        if cid < 0 or cid >= len(clauses):
            raise ValueError(f"conflict_clause_id out of range: {cid}")

        if len(trail) != len(trail_levels):
            raise ValueError("trail and trail_levels length mismatch")

        if len(var_reason) != num_vars:
            raise ValueError("var_reason length mismatch")

    @staticmethod
    def _canonicalize_lits(lits: List[int]) -> List[int]:
        seen: Dict[int, int] = {}
        cleaned: List[int] = []
        for lit in lits:
            lit = int(lit)
            var = int(abs(lit) - 1)
            sign = 1 if lit > 0 else -1
            if var in seen:
                if int(seen[var]) != int(sign):
                    raise ValueError(
                        "tautological clause encountered during resolution"
                    )
                continue
            seen[var] = int(sign)
            cleaned.append(int(lit))
        cleaned.sort(key=lambda x: (abs(int(x)), int(x)))
        return cleaned

    def serialize_state(
        self,
        clauses,
        trail: List[int],
        trail_levels: List[int],
        var_reason: List[Optional[int]],
        conflict_clause_id: int,
        num_vars: int,
        mode: str = "flat",
    ) -> List[int]:
        """Serialize a conflict state into a token sequence (without BOS/EOS)."""
        mode = str(mode)
        if mode not in {"flat", "scratchpad"}:
            raise ValueError(f"Unknown mode: {mode}")

        self._validate_inputs(
            clauses,
            trail,
            trail_levels,
            var_reason,
            conflict_clause_id,
            num_vars,
        )

        tokens: List[int] = [self.CLAUSE_SEC]

        for cid, clause in enumerate(clauses):
            tokens.append(self.clause_token(int(cid)))
            tokens.append(self.COLON)
            for lit in clause:
                tokens.append(self.lit_token(int(lit)))
            tokens.append(self.SEP)

        tokens.append(self.TRAIL_SEC)
        for idx, lit in enumerate(trail):
            tokens.append(self.lit_token(int(lit)))
            tokens.append(self.AT_KW)
            tokens.append(self.level_token(int(trail_levels[int(idx)])))
            var_id = self._lit_var(int(lit))
            reason = var_reason[int(var_id)]
            if reason is None:
                tokens.append(self.DEC_KW)
            else:
                tokens.append(self.REAS_KW)
                tokens.append(self.clause_token(int(reason)))
            tokens.append(self.SEP)

        tokens.append(self.CONFLICT_SEC)
        tokens.append(self.clause_token(int(conflict_clause_id)))
        tokens.append(self.SEP)

        if mode == "flat":
            tokens.append(self.TARGET_SEC)
        else:
            tokens.append(self.THINK_SEC)

        return tokens

    def compute_1uip(
        self,
        clauses,
        trail: List[int],
        trail_levels: List[int],
        var_reason: List[Optional[int]],
        conflict_clause_id: int,
        num_vars: int,
    ) -> dict:
        """Compute 1-UIP resolution trace for a conflict."""
        self._validate_1uip_inputs(
            clauses,
            trail,
            trail_levels,
            var_reason,
            conflict_clause_id,
            num_vars,
        )

        logger.debug(
            "compute_1uip inputs: num_vars=%d num_clauses=%d conflict_clause_id=%d trail_len=%d trail_levels_len=%d var_reason_len=%d",
            int(num_vars),
            int(len(clauses)),
            int(conflict_clause_id),
            int(len(trail)),
            int(len(trail_levels)),
            int(len(var_reason)),
        )

        if not trail_levels:
            raise ValueError("empty trail for conflict")

        dcur = int(max(trail_levels))
        if dcur == 0:
            raise ValueError("conflict at decision level 0")

        var_level = [-1 for _ in range(int(num_vars))]
        trail_pos: Dict[int, int] = {}
        for idx, lit in enumerate(trail):
            var_id = self._lit_var(int(lit))
            var_level[int(var_id)] = int(trail_levels[int(idx)])
            trail_pos[int(var_id)] = int(idx)

        conflict_clause_raw = [int(l) for l in clauses[int(conflict_clause_id)]]
        conflict_clause = self._canonicalize_lits(list(conflict_clause_raw))
        current_clause = list(conflict_clause)

        def _count_current_level(lits: List[int]) -> Tuple[int, Optional[int]]:
            count = 0
            last_lit: Optional[int] = None
            for lit in lits:
                lvl = var_level[self._lit_var(int(lit))]
                if int(lvl) == int(dcur):
                    count += 1
                    last_lit = int(lit)
            return count, last_lit

        resolution_steps: List[dict] = []

        count, _ = _count_current_level(current_clause)
        while int(count) > 1:
            current_level_lits = [
                lit
                for lit in current_clause
                if int(var_level[self._lit_var(int(lit))]) == int(dcur)
            ]
            if not current_level_lits:
                raise RuntimeError("no literals at current decision level")

            pivot = max(
                current_level_lits,
                key=lambda lit: trail_pos[self._lit_var(int(lit))],
            )

            var_id = self._lit_var(int(pivot))
            reason_id = var_reason[int(var_id)]
            if reason_id is None:
                raise RuntimeError(
                    f"missing reason clause for pivot var={var_id} at level {dcur}"
                )

            reason_clause = [int(l) for l in clauses[int(reason_id)]]
            assigned_lit = int(-pivot)
            if assigned_lit not in reason_clause:
                raise RuntimeError(
                    f"reason clause {reason_id} missing assigned lit {assigned_lit}"
                )

            new_clause = [lit for lit in current_clause if int(lit) != int(pivot)]
            for lit in reason_clause:
                if int(lit) == int(assigned_lit):
                    continue
                new_clause.append(int(lit))

            resolved = self._canonicalize_lits(new_clause)
            resolution_steps.append(
                {
                    "pivot_lit": int(pivot),
                    "reason_clause_id": int(reason_id),
                    "reason_clause": list(reason_clause),
                    "resolvent": list(resolved),
                }
            )

            current_clause = list(resolved)
            count, _ = _count_current_level(current_clause)

        if int(count) != 1:
            raise RuntimeError("failed to reach 1-UIP clause")

        asserting_lits = [
            lit
            for lit in current_clause
            if int(var_level[self._lit_var(int(lit))]) == int(dcur)
        ]
        if len(asserting_lits) != 1:
            raise RuntimeError("asserting literal not unique")

        asserting_literal = int(asserting_lits[0])

        backjump_level = 0
        for lit in current_clause:
            lvl = var_level[self._lit_var(int(lit))]
            if int(lvl) != int(dcur):
                backjump_level = max(int(backjump_level), int(lvl))

        logger.debug(
            "compute_1uip result: decision_level=%d backjump_level=%d learned_clause_len=%d resolution_steps=%d asserting_literal=%d",
            int(dcur),
            int(backjump_level),
            int(len(current_clause)),
            int(len(resolution_steps)),
            int(asserting_literal),
        )

        return {
            "learned_clause": list(current_clause),
            "backjump_level": int(backjump_level),
            "resolution_steps": resolution_steps,
            "num_resolution_steps": int(len(resolution_steps)),
            "asserting_literal": int(asserting_literal),
            "decision_level": int(dcur),
            "conflict_clause": list(conflict_clause_raw),
        }

    def generate_scratchpad(
        self,
        clauses,
        trail: List[int],
        trail_levels: List[int],
        var_reason: List[Optional[int]],
        conflict_clause_id: int,
        num_vars: int,
    ) -> Tuple[List[int], int]:
        """Generate the 1-UIP scratchpad tokens and the target backjump level."""
        trace = self.compute_1uip(
            clauses=clauses,
            trail=trail,
            trail_levels=trail_levels,
            var_reason=var_reason,
            conflict_clause_id=conflict_clause_id,
            num_vars=num_vars,
        )

        tokens: List[int] = []

        tokens.append(self.DEC_KW)
        tokens.append(self.level_token(int(trace["decision_level"])))
        tokens.append(self.SEP)

        tokens.append(self.CONFLICT_SEC)
        tokens.append(self.clause_token(int(conflict_clause_id)))
        tokens.append(self.COLON)
        for lit in trace["conflict_clause"]:
            tokens.append(self.lit_token(int(lit)))
        tokens.append(self.SEP)

        for step in trace["resolution_steps"]:
            tokens.append(self.PIVOT_KW)
            tokens.append(self.lit_token(int(step["pivot_lit"])))
            tokens.append(self.REAS_KW)
            tokens.append(self.clause_token(int(step["reason_clause_id"])))
            tokens.append(self.COLON)
            for lit in step["reason_clause"]:
                tokens.append(self.lit_token(int(lit)))
            tokens.append(self.ARROW)
            for lit in step["resolvent"]:
                tokens.append(self.lit_token(int(lit)))
            tokens.append(self.SEP)

        tokens.append(self.UIP_KW)
        tokens.append(self.lit_token(int(trace["asserting_literal"])))
        tokens.append(self.SEP)

        tokens.append(self.LEARN_KW)
        tokens.append(self.COLON)
        for lit in trace["learned_clause"]:
            tokens.append(self.lit_token(int(lit)))
        tokens.append(self.SEP)

        tokens.append(self.BJ_KW)
        tokens.append(self.level_token(int(trace["backjump_level"])))
        tokens.append(self.SEP)

        return tokens, int(trace["backjump_level"])

    def build_training_sequence(
        self,
        clauses,
        trail: List[int],
        trail_levels: List[int],
        var_reason: List[Optional[int]],
        conflict_clause_id: int,
        num_vars: int,
        mode: str = "scratchpad",
    ) -> Tuple[List[int], int]:
        """Build the full training sequence and target backjump level."""
        mode = str(mode)
        if mode not in {"flat", "scratchpad"}:
            raise ValueError(f"Unknown mode: {mode}")

        state_tokens = self.serialize_state(
            clauses,
            trail,
            trail_levels,
            var_reason,
            conflict_clause_id,
            num_vars,
            mode=mode,
        )

        if mode == "flat":
            trace = self.compute_1uip(
                clauses=clauses,
                trail=trail,
                trail_levels=trail_levels,
                var_reason=var_reason,
                conflict_clause_id=conflict_clause_id,
                num_vars=num_vars,
            )
            target_level = int(trace["backjump_level"])
            sequence = [
                self.BOS,
                *state_tokens,
                self.level_token(target_level),
                self.EOS,
            ]
            return sequence, int(target_level)

        scratchpad_tokens, target_level = self.generate_scratchpad(
            clauses,
            trail,
            trail_levels,
            var_reason,
            conflict_clause_id,
            num_vars,
        )

        sequence = [
            self.BOS,
            *state_tokens,
            *scratchpad_tokens,
            self.TARGET_SEC,
            self.level_token(int(target_level)),
            self.EOS,
        ]

        return sequence, int(target_level)

    def serialize_state_augmented(
        self,
        clauses,
        trail: List[int],
        trail_levels: List[int],
        var_reason: List[Optional[int]],
        conflict_clause_id: int,
        num_vars: int,
        mode: str = "scratchpad",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[int], int]:
        """Build a training sequence with a random variable permutation."""
        if rng is None:
            rng = np.random.default_rng()

        num_vars = int(num_vars)
        perm = np.array(rng.permutation(num_vars), dtype=np.int64)

        def _remap_lit(lit: int) -> int:
            lit = int(lit)
            var_id = self._lit_var(lit)
            new_var = int(perm[int(var_id)])
            sign = 1 if lit > 0 else -1
            return int(sign * (new_var + 1))

        permuted_clauses = [
            tuple(_remap_lit(int(lit)) for lit in clause) for clause in clauses
        ]
        permuted_trail = [_remap_lit(int(lit)) for lit in trail]

        permuted_var_reason: List[Optional[int]] = [None for _ in range(num_vars)]
        for old_var in range(num_vars):
            new_var = int(perm[int(old_var)])
            reason = var_reason[int(old_var)]
            permuted_var_reason[int(new_var)] = None if reason is None else int(reason)

        return self.build_training_sequence(
            permuted_clauses,
            permuted_trail,
            list(trail_levels),
            permuted_var_reason,
            int(conflict_clause_id),
            num_vars,
            mode=mode,
        )
