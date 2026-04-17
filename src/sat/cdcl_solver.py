from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _lit_var(lit: int) -> int:
    return int(abs(int(lit)) - 1)


def _lit_sign(lit: int) -> int:
    return 1 if int(lit) > 0 else -1


def _lit_value(assignment: List[int], lit: int) -> int:
    v = int(assignment[_lit_var(int(lit))])
    if v == 0:
        return 0
    return 1 if int(v) == int(_lit_sign(int(lit))) else -1


def _val_bit(val: int) -> int:
    v = int(val)
    if v not in {-1, 1}:
        raise ValueError("val must be in {-1,+1}")
    return 1 if v == -1 else 2


def _dedup_clause(lits: List[int]) -> List[int]:
    seen = set()
    cleaned: List[int] = []
    for lit in lits:
        lit = int(lit)
        if lit in seen:
            continue
        seen.add(int(lit))
        cleaned.append(int(lit))
    return cleaned


ConflictAnalyzer = Callable[[dict], Optional[Tuple[List[int], int, int]]]


@dataclass
class Decision:
    var: int
    level: int
    tried_mask: int


def compute_1uip(
    *,
    clauses: List[List[int]],
    trail: List[int],
    trail_levels: List[int],
    var_reason: List[Optional[int]],
    conflict_clause_id: int,
    num_vars: int,
    level: Optional[int] = None,
) -> Tuple[List[int], int, int]:
    if int(num_vars) <= 0:
        raise ValueError("num_vars must be positive")
    if not clauses:
        raise ValueError("clauses must be non-empty")
    if len(trail) != len(trail_levels):
        raise ValueError("trail and trail_levels length mismatch")
    if len(var_reason) != int(num_vars):
        raise ValueError("var_reason length mismatch")

    cid = int(conflict_clause_id)
    if cid < 0 or cid >= len(clauses):
        raise ValueError(f"conflict_clause_id out of range: {cid}")

    if level is None:
        if not trail_levels:
            raise ValueError("empty trail for conflict analysis")
        level = int(max(trail_levels))
    level = int(level)
    if level <= 0:
        raise ValueError("conflict at decision level 0")

    var_level = [-1 for _ in range(int(num_vars))]
    trail_pos: Dict[int, int] = {}
    for idx, lit in enumerate(trail):
        v = _lit_var(int(lit))
        var_level[int(v)] = int(trail_levels[int(idx)])
        trail_pos[int(v)] = int(idx)

    current_clause = _dedup_clause([int(l) for l in clauses[int(cid)]])

    def _count_current_level(lits: List[int]) -> Tuple[int, Optional[int]]:
        count = 0
        last_lit: Optional[int] = None
        for lit in lits:
            if int(var_level[_lit_var(int(lit))]) == int(level):
                count += 1
                last_lit = int(lit)
        return count, last_lit

    count, _ = _count_current_level(current_clause)
    while int(count) > 1:
        pivot = None
        pivot_pos = -1
        for lit in current_clause:
            v = _lit_var(int(lit))
            if int(var_level[int(v)]) != int(level):
                continue
            pos = int(trail_pos.get(int(v), -1))
            if int(pos) > int(pivot_pos):
                pivot = int(lit)
                pivot_pos = int(pos)

        if pivot is None:
            raise RuntimeError("no pivot literal found during conflict analysis")

        var = _lit_var(int(pivot))
        reason_id = var_reason[int(var)]
        if reason_id is None:
            raise RuntimeError(
                f"missing reason clause for pivot var={var} at level {level}"
            )

        reason_clause = clauses[int(reason_id)]
        assigned_lit = int(-pivot)
        if int(assigned_lit) not in {int(l) for l in reason_clause}:
            raise RuntimeError(
                f"reason clause {reason_id} missing assigned lit {assigned_lit}"
            )

        new_clause = [lit for lit in current_clause if int(lit) != int(pivot)]
        for lit in reason_clause:
            if int(lit) == int(assigned_lit):
                continue
            if int(lit) not in {int(x) for x in new_clause}:
                new_clause.append(int(lit))

        current_clause = _dedup_clause(new_clause)
        count, _ = _count_current_level(current_clause)

    if int(count) != 1:
        raise RuntimeError("failed to reach 1-UIP clause")

    asserting_lits = [
        lit
        for lit in current_clause
        if int(var_level[_lit_var(int(lit))]) == int(level)
    ]
    if len(asserting_lits) != 1:
        raise RuntimeError("asserting literal not unique")

    asserting_literal = int(asserting_lits[0])
    backjump_level = 0
    for lit in current_clause:
        lit_level = int(var_level[_lit_var(int(lit))])
        if int(lit_level) != int(level):
            backjump_level = max(int(backjump_level), int(lit_level))

    return list(current_clause), int(backjump_level), int(asserting_literal)


def compute_1uip_from_state(state: dict) -> Tuple[List[int], int, int]:
    return compute_1uip(
        clauses=state["clauses"],
        trail=state["trail"],
        trail_levels=state["trail_levels"],
        var_reason=state["var_reason"],
        conflict_clause_id=state["conflict_clause_id"],
        num_vars=state["num_vars"],
        level=state.get("level"),
    )


class CDCLSolver:
    """Simple CDCL solver for evaluation purposes."""

    def __init__(
        self,
        clauses: List[Tuple[int, ...]],
        num_vars: int,
        max_conflicts: int = 50000,
        conflict_analyzer: Optional[ConflictAnalyzer] = None,
        chronological: bool = False,
    ) -> None:
        if int(num_vars) <= 0:
            raise ValueError("num_vars must be >= 1")
        if int(max_conflicts) <= 0:
            raise ValueError("max_conflicts must be >= 1")
        if not clauses:
            raise ValueError("clauses must be non-empty")

        self.num_vars = int(num_vars)
        self.max_conflicts = int(max_conflicts)
        self.conflict_analyzer = conflict_analyzer
        self.chronological = bool(chronological)

        self.clauses: List[List[int]] = []
        for c in clauses:
            lits = [int(l) for l in c]
            if not lits:
                raise ValueError("clause cannot be empty")
            for lit in lits:
                if int(lit) == 0:
                    raise ValueError("literal cannot be 0")
                v = abs(int(lit))
                if v < 1 or v > int(self.num_vars):
                    raise ValueError(
                        f"literal var out of range: lit={lit} (expected abs in [1,{self.num_vars}])"
                    )
            self.clauses.append(list(lits))

        self.num_original_clauses = int(len(self.clauses))

        self.assignment = [0 for _ in range(int(self.num_vars))]
        self.num_assigned = 0
        self.trail: List[Tuple[int, int, Optional[int]]] = []
        self.trail_pos: Dict[int, int] = {}
        self.level = 0
        self.var_level = [0 for _ in range(int(self.num_vars))]
        self.var_reason: List[Optional[int]] = [None for _ in range(int(self.num_vars))]

        self.watches: List[Tuple[int, int]] = []
        self.watch_list: Dict[int, List[int]] = defaultdict(list)
        self._init_watches()

        self.propagation_queue: List[int] = []

        self.activity = [0.0 for _ in range(int(self.num_vars))]
        self.activity_decay = 0.95

        self.decision_stack: List[Decision] = []

        self.stats = {
            "decisions": 0,
            "conflicts": 0,
            "backtracks": 0,
            "propagations": 0,
            "learned_clauses": 0,
            "max_level": 0,
        }

    def _init_watches(self) -> None:
        for cid, clause in enumerate(self.clauses):
            if not clause:
                raise ValueError("clause cannot be empty")
            if len(clause) == 1:
                w1 = w2 = 0
            else:
                w1, w2 = 0, 1
            self.watches.append((int(w1), int(w2)))
            l1 = int(clause[int(w1)])
            l2 = int(clause[int(w2)])
            self.watch_list[int(l1)].append(int(cid))
            if int(l2) != int(l1):
                self.watch_list[int(l2)].append(int(cid))

    def _assign(self, lit: int, level: int, reason: Optional[int]) -> bool:
        var = _lit_var(int(lit))
        val = _lit_sign(int(lit))
        current = int(self.assignment[int(var)])
        if int(current) != 0:
            return int(current) == int(val)
        self.assignment[int(var)] = int(val)
        self.var_level[int(var)] = int(level)
        self.var_reason[int(var)] = None if reason is None else int(reason)
        self.trail_pos[int(var)] = int(len(self.trail))
        self.trail.append(
            (int(lit), int(level), None if reason is None else int(reason))
        )
        self.num_assigned += 1
        self.propagation_queue.append(int(-lit))
        return True

    def _propagate(self) -> Optional[int]:
        while self.propagation_queue:
            lit = int(self.propagation_queue.pop())
            watch_clauses = self.watch_list[int(lit)]
            idx = 0
            while idx < len(watch_clauses):
                cid = int(watch_clauses[int(idx)])
                clause = self.clauses[int(cid)]
                w1, w2 = self.watches[int(cid)]
                if int(clause[int(w1)]) == int(lit):
                    false_idx = int(w1)
                    other_idx = int(w2)
                else:
                    false_idx = int(w2)
                    other_idx = int(w1)

                other_lit = int(clause[int(other_idx)])
                other_val = int(_lit_value(self.assignment, int(other_lit)))
                if int(other_val) == 1:
                    idx += 1
                    continue

                moved = False
                for new_idx, cand in enumerate(clause):
                    if int(new_idx) in {int(false_idx), int(other_idx)}:
                        continue
                    if int(_lit_value(self.assignment, int(cand))) != -1:
                        if int(false_idx) == int(w1):
                            self.watches[int(cid)] = (int(new_idx), int(other_idx))
                        else:
                            self.watches[int(cid)] = (int(other_idx), int(new_idx))
                        watch_clauses[int(idx)] = watch_clauses[-1]
                        watch_clauses.pop()
                        self.watch_list[int(cand)].append(int(cid))
                        moved = True
                        break

                if moved:
                    continue

                if int(other_val) == -1:
                    return int(cid)

                if int(other_val) == 0:
                    var = _lit_var(int(other_lit))
                    current = int(self.assignment[int(var)])
                    if int(current) == 0:
                        if not self._assign(int(other_lit), int(self.level), int(cid)):
                            return int(cid)
                        self.stats["propagations"] += 1
                    elif int(current) != int(_lit_sign(int(other_lit))):
                        return int(cid)

                idx += 1
        return None

    def _bump_activity(self, var: int) -> None:
        self.activity[int(var)] += 1.0

    def _decay_activity(self) -> None:
        for v in range(int(self.num_vars)):
            self.activity[int(v)] *= float(self.activity_decay)

    def _bump_activity_clause(self, clause: List[int]) -> None:
        seen = set()
        for lit in clause:
            var = _lit_var(int(lit))
            if int(var) in seen:
                continue
            seen.add(int(var))
            self._bump_activity(int(var))

    def _analyze_conflict(self, conflict_cid: int) -> Tuple[List[int], int, int]:
        state = self._build_conflict_state(int(conflict_cid))
        learned_clause, backjump_level, asserting_lit = compute_1uip_from_state(state)
        self._bump_activity_clause(list(learned_clause))
        self._decay_activity()
        logger.debug(
            "conflict cid=%d level=%d learned_len=%d backjump=%d",
            int(conflict_cid),
            int(self.level),
            int(len(learned_clause)),
            int(backjump_level),
        )
        return list(learned_clause), int(backjump_level), int(asserting_lit)

    def _add_learned_clause(self, lits: List[int]) -> int:
        clause = _dedup_clause(list(lits))
        if not clause:
            raise RuntimeError("learned clause is empty")
        cid = int(len(self.clauses))
        self.clauses.append(list(clause))
        if len(clause) == 1:
            w1 = w2 = 0
        else:
            w1, w2 = 0, 1
        self.watches.append((int(w1), int(w2)))
        l1 = int(clause[int(w1)])
        l2 = int(clause[int(w2)])
        self.watch_list[int(l1)].append(int(cid))
        if int(l2) != int(l1):
            self.watch_list[int(l2)].append(int(cid))
        return int(cid)

    def _backjump(self, target_level: int) -> None:
        target_level = int(target_level)
        if int(target_level) < 0:
            raise ValueError("target_level must be >= 0")
        while self.trail and int(self.trail[-1][1]) > int(target_level):
            lit, lvl, _ = self.trail.pop()
            var = _lit_var(int(lit))
            if int(self.assignment[int(var)]) != 0:
                self.assignment[int(var)] = 0
                self.num_assigned -= 1
            self.var_level[int(var)] = 0
            self.var_reason[int(var)] = None
            self.trail_pos.pop(int(var), None)
        while self.decision_stack and int(self.decision_stack[-1].level) > int(
            target_level
        ):
            self.decision_stack.pop()
        self.propagation_queue = []
        self.stats["backtracks"] += 1

    def _clause_unit_literal(self, clause: List[int]) -> Optional[int]:
        unit_lit: Optional[int] = None
        for lit in clause:
            val = _lit_value(self.assignment, int(lit))
            if int(val) == 1:
                return None
            if int(val) == 0:
                if unit_lit is not None:
                    return None
                unit_lit = int(lit)
        return unit_lit

    def _pick_var(self) -> Optional[int]:
        best_var: Optional[int] = None
        best_act = -1.0
        for v in range(int(self.num_vars)):
            if int(self.assignment[int(v)]) != 0:
                continue
            act = float(self.activity[int(v)])
            if act > best_act + 1e-12:
                best_var = int(v)
                best_act = float(act)
            elif (
                abs(act - best_act) <= 1e-12
                and best_var is not None
                and int(v) < int(best_var)
            ):
                best_var = int(v)
        return best_var

    def _build_conflict_state(self, conflict_cid: int) -> dict:
        return {
            "clauses": [list(c) for c in self.clauses],
            "trail": [int(lit) for lit, _, _ in self.trail],
            "trail_levels": [int(level) for _, level, _ in self.trail],
            "var_reason": [None if r is None else int(r) for r in self.var_reason],
            "conflict_clause_id": int(conflict_cid),
            "num_vars": int(self.num_vars),
            "level": int(self.level),
        }

    def _resolve_conflict(self, conflict_cid: int) -> Tuple[List[int], int, int]:
        if self.conflict_analyzer is not None:
            state = self._build_conflict_state(int(conflict_cid))
            result = self.conflict_analyzer(state)
            if result is not None:
                learned_clause, backjump_level, asserting_lit = result
                learned_clause = list(learned_clause)
                backjump_level = int(backjump_level)
                asserting_lit = int(asserting_lit)
                if backjump_level < 0 or backjump_level >= int(self.level):
                    raise ValueError(
                        f"invalid backjump_level={backjump_level} at level={self.level}"
                    )
                if not learned_clause:
                    raise ValueError("learned clause must be non-empty")
                self._bump_activity_clause(list(learned_clause))
                self._decay_activity()
                return learned_clause, backjump_level, asserting_lit
        return self._analyze_conflict(int(conflict_cid))

    def _chronological_backtrack(self) -> bool:
        while self.decision_stack:
            last = self.decision_stack.pop()
            target_level = int(last.level) - 1
            self._backjump(int(target_level))
            self.level = int(target_level)

            if int(last.tried_mask) == 3:
                continue

            if int(last.tried_mask) == 2:
                next_val = -1
            else:
                next_val = 1

            self.level += 1
            self.stats["max_level"] = max(int(self.stats["max_level"]), int(self.level))
            lit = int((int(last.var) + 1) * int(next_val))
            self.stats["decisions"] += 1
            if not self._assign(int(lit), int(self.level), None):
                continue
            new_mask = int(last.tried_mask) | int(_val_bit(int(next_val)))
            self.decision_stack.append(
                Decision(
                    var=int(last.var), level=int(self.level), tried_mask=int(new_mask)
                )
            )
            return True
        return False

    def solve(self) -> dict:
        """Run CDCL solver. Returns stats dict."""

        for cid, clause in enumerate(self.clauses):
            if len(clause) == 1:
                lit = int(clause[0])
                var = _lit_var(int(lit))
                current = int(self.assignment[int(var)])
                if int(current) == 0:
                    if not self._assign(int(lit), 0, int(cid)):
                        self.stats["conflicts"] += 1
                        return self._finalize("unsat")
                    self.stats["propagations"] += 1
                elif int(current) != int(_lit_sign(int(lit))):
                    self.stats["conflicts"] += 1
                    return self._finalize("unsat")

        conflict = self._propagate()
        if conflict is not None:
            self.stats["conflicts"] += 1
            return self._finalize("unsat")

        while True:
            if int(self.stats["conflicts"]) >= int(self.max_conflicts):
                return self._finalize("unknown")

            if int(self.num_assigned) == int(self.num_vars):
                return self._finalize("sat")

            var = self._pick_var()
            if var is None:
                return self._finalize("sat")

            self.level += 1
            self.stats["max_level"] = max(int(self.stats["max_level"]), int(self.level))
            decision_val = 1
            decision_lit = int((int(var) + 1) * int(decision_val))
            self.stats["decisions"] += 1
            if not self._assign(int(decision_lit), int(self.level), None):
                raise RuntimeError("decision assignment inconsistent")
            self.decision_stack.append(
                Decision(
                    var=int(var),
                    level=int(self.level),
                    tried_mask=int(_val_bit(int(decision_val))),
                )
            )

            while True:
                conflict = self._propagate()
                if conflict is None:
                    break

                self.stats["conflicts"] += 1

                if int(self.level) == 0:
                    return self._finalize("unsat")

                if self.chronological:
                    self._bump_activity_clause(self.clauses[int(conflict)])
                    self._decay_activity()
                    if not self._chronological_backtrack():
                        return self._finalize("unsat")
                    continue

                learned_clause, backjump_level, _asserting_lit = self._resolve_conflict(
                    int(conflict)
                )
                learned_cid = self._add_learned_clause(list(learned_clause))
                self.stats["learned_clauses"] += 1
                self._backjump(int(backjump_level))
                self.level = int(backjump_level)

                unit_lit = self._clause_unit_literal(self.clauses[int(learned_cid)])
                if unit_lit is not None:
                    if not self._assign(
                        int(unit_lit), int(self.level), int(learned_cid)
                    ):
                        raise RuntimeError("asserting assignment inconsistent")
                    self.stats["propagations"] += 1

        return self._finalize("unknown")

    def _finalize(self, status: str) -> dict:
        payload = {
            "status": str(status),
            "assignment": [int(x) for x in self.assignment],
            "level": int(self.level),
            "num_clauses": int(len(self.clauses)),
        }
        payload.update({k: int(v) for k, v in self.stats.items()})
        logger.info(
            "cdcl status=%s decisions=%d conflicts=%d backtracks=%d learned=%d propagations=%d",
            str(status),
            int(self.stats["decisions"]),
            int(self.stats["conflicts"]),
            int(self.stats["backtracks"]),
            int(self.stats["learned_clauses"]),
            int(self.stats["propagations"]),
        )
        return payload
