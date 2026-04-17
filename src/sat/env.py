from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np

from .dsl import SatAction, SatActionType

logger = logging.getLogger(__name__)


class SatEnvStatus(Enum):
    RUNNING = auto()
    SUCCESS = auto()  # SAT found
    FAILURE = auto()  # UNSAT / invalid / step limit


@dataclass
class DecisionFrame:
    decision_var: int
    chosen_val: int  # +1 (True) or -1 (False)
    trail_start: int
    tried_mask: int  # bits: which values tried at this node
    failed_mask: int  # bits: which values proven inconsistent (branch-local nogood)


@dataclass
class SatState:
    num_vars: int
    num_clauses: int
    clauses: List[Tuple[int, ...]]

    # Assignment: -1=False, 0=Unassigned, +1=True
    assignment: np.ndarray  # (num_vars,)

    # Optional heuristic state (VSIDS-lite activity)
    activity: np.ndarray  # (num_vars,) float

    # UI selection (like graph_coloring.selected_node)
    selected_var: Optional[int]

    # Trail for backtracking
    trail: List[int]  # chronological list of assigned literals
    trail_levels: List[int]  # decision level for each trail entry
    var_reason: List[Optional[int]]  # clause that implied var (None if decision)

    # Decision stack with branch-local nogoods
    decision_stack: List[DecisionFrame]

    # Watched literals (2 per clause)
    watch_pos: List[Tuple[int, int]]  # watch_pos[c] = (i, j) positions in clause
    watch_list: Dict[int, List[int]]  # lit -> [clause_ids watching this lit]

    # Propagation
    propagation_queue: List[int]  # falsified watched literals to process
    propagation_pending: bool
    conflict_clause: Optional[
        int
    ]  # clause id if in conflict state; may be <0 sentinel at level 0

    # Terminal
    status: SatEnvStatus
    step_count: int
    termination_reason: Optional[str] = (
        None  # "sat" | "unsat" | "timeout" | "invalid" | ...
    )


@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


def _lit_var(lit: int) -> int:
    return int(abs(int(lit)) - 1)


def _lit_sign(lit: int) -> int:
    return 1 if int(lit) > 0 else -1


def _lit_value(assignment: np.ndarray, lit: int) -> int:
    """Return 1 if lit true, -1 if false, 0 if unassigned."""

    v = _lit_var(lit)
    a = int(assignment[v])
    if a == 0:
        return 0
    s = _lit_sign(lit)
    return 1 if int(a) == int(s) else -1


def _val_bit(val: int) -> int:
    """Bit encoding for {-1,+1} values."""

    v = int(val)
    if v not in {-1, 1}:
        raise ValueError("val must be in {-1,+1}")
    return 1 if v == -1 else 2


def _num_assigned(assignment: np.ndarray) -> int:
    return int(np.count_nonzero(assignment))


def _copy_watch_list(w: Dict[int, List[int]]) -> Dict[int, List[int]]:
    return {int(lit): [int(x) for x in lst] for lit, lst in w.items()}


def _copy_decision_stack(stack: List[DecisionFrame]) -> List[DecisionFrame]:
    return [
        DecisionFrame(
            decision_var=int(f.decision_var),
            chosen_val=int(f.chosen_val),
            trail_start=int(f.trail_start),
            tried_mask=int(f.tried_mask),
            failed_mask=int(f.failed_mask),
        )
        for f in stack
    ]


class SatEnv:
    """SAT environment with explicit propagation and branch-local nogoods.

    Workflow (like graph_coloring):
      1) SELECT_VAR(i)
      2) ASSIGN_VALUE(v)
      3) PROPAGATE
      4) (repeat or BACKTRACK)
      5) DONE (declares SUCCESS if SAT; otherwise FAILURE)

    Literals are encoded as ±(var_idx+1).
    """

    def __init__(
        self,
        clauses: List[Tuple[int, ...]],
        num_vars: int,
        planted_solution: Optional[np.ndarray] = None,
        mode: str = "soft",  # "soft" or "strict"
        max_steps: int = 1000,
        activity_bins: int = 16,
        activity_clip: float = 10.0,
    ):
        if mode not in {"strict", "soft"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be >= 1")
        if int(num_vars) < 1:
            raise ValueError("num_vars must be >= 1")
        if int(activity_bins) < 2:
            raise ValueError("activity_bins must be >= 2")
        if float(activity_clip) <= 0.0:
            raise ValueError("activity_clip must be > 0")

        self.num_vars = int(num_vars)

        cls: List[Tuple[int, ...]] = []
        for c in clauses:
            if len(c) == 0:
                raise ValueError("Each clause must be non-empty")
            lits = tuple(int(x) for x in c)
            for lit in lits:
                if int(lit) == 0:
                    raise ValueError("Literal cannot be 0")
                v = abs(int(lit))
                if v < 1 or v > self.num_vars:
                    raise ValueError(
                        f"Literal var out of range: lit={lit} (expected abs in [1,{self.num_vars}])"
                    )
            cls.append(tuple(int(x) for x in lits))

        if not cls:
            raise ValueError("clauses must be non-empty")

        self.clauses = cls
        self.num_clauses = int(len(cls))

        self.planted_solution: Optional[np.ndarray]
        if planted_solution is not None:
            sol = np.array(planted_solution, dtype=np.int64, copy=True)
            if sol.shape != (self.num_vars,):
                raise ValueError(f"planted_solution must have shape ({self.num_vars},)")
            if not np.all(np.isin(sol, [-1, 1])):
                raise ValueError("planted_solution must be in {-1,+1}")
            # Verify solution satisfies all clauses.
            for cid, cl in enumerate(self.clauses):
                if not any(_lit_value(sol, int(l)) == 1 for l in cl):
                    raise ValueError(f"planted_solution falsifies clause {cid}: {cl}")
            self.planted_solution = sol
        else:
            self.planted_solution = None

        self.mode = mode
        self.max_steps = int(max_steps)

        # Rewards (kept consistent with other envs).
        self.goal_reward = 1.0
        self.step_penalty = -0.01
        self.invalid_penalty = -1.0

        self.activity_bins = int(activity_bins)
        self.activity_clip = float(activity_clip)

        # Constant observation payload.
        self._clauses_obs = [[int(lit) for lit in clause] for clause in self.clauses]

        self._state: Optional[SatState] = None

    def reset(self) -> dict:
        # Init watched literals: watch positions (0,1) per clause (or (0,0) for unit clauses).
        watch_pos: List[Tuple[int, int]] = []
        watch_list: Dict[int, List[int]] = {}

        for cid, cl in enumerate(self.clauses):
            if len(cl) == 1:
                watch_pos.append((0, 0))
                l0 = int(cl[0])
                watch_list.setdefault(int(l0), []).append(int(cid))
            else:
                watch_pos.append((0, 1))
                l0, l1 = int(cl[0]), int(cl[1])
                watch_list.setdefault(int(l0), []).append(int(cid))
                watch_list.setdefault(int(l1), []).append(int(cid))

        self._state = SatState(
            num_vars=int(self.num_vars),
            num_clauses=int(self.num_clauses),
            clauses=list(self.clauses),
            assignment=np.zeros((self.num_vars,), dtype=np.int64),
            activity=np.zeros((self.num_vars,), dtype=np.float32),
            selected_var=None,
            trail=[],
            trail_levels=[],
            var_reason=[None for _ in range(self.num_vars)],
            decision_stack=[],
            watch_pos=watch_pos,
            watch_list=watch_list,
            propagation_queue=[],
            propagation_pending=False,
            conflict_clause=None,
            status=SatEnvStatus.RUNNING,
            step_count=0,
        )

        return self.get_observation(self._state)

    def _require_running(self) -> SatState:
        if self._state is None:
            raise RuntimeError("Environment not reset")
        if self._state.status != SatEnvStatus.RUNNING:
            raise RuntimeError(f"Environment already terminated: {self._state.status}")
        return self._state

    def get_state(self) -> SatState:
        if self._state is None:
            raise RuntimeError("Environment not reset")
        s = self._state
        return SatState(
            num_vars=int(s.num_vars),
            num_clauses=int(s.num_clauses),
            clauses=list(s.clauses),
            assignment=np.array(s.assignment, copy=True),
            activity=np.array(s.activity, copy=True),
            selected_var=None if s.selected_var is None else int(s.selected_var),
            trail=[int(x) for x in s.trail],
            trail_levels=[int(x) for x in s.trail_levels],
            var_reason=[None if r is None else int(r) for r in s.var_reason],
            decision_stack=_copy_decision_stack(s.decision_stack),
            watch_pos=[(int(a), int(b)) for (a, b) in s.watch_pos],
            watch_list=_copy_watch_list(s.watch_list),
            propagation_queue=[int(x) for x in s.propagation_queue],
            propagation_pending=bool(s.propagation_pending),
            conflict_clause=None
            if s.conflict_clause is None
            else int(s.conflict_clause),
            status=s.status,
            step_count=int(s.step_count),
            termination_reason=None
            if s.termination_reason is None
            else str(s.termination_reason),
        )

    def _open_decision_var(self, state: SatState) -> Optional[int]:
        if not state.decision_stack:
            return None
        top = state.decision_stack[-1]
        if int(state.assignment[int(top.decision_var)]) == 0:
            return int(top.decision_var)
        return None

    def _effective_domain(self, state: SatState, var: int) -> set[int]:
        v = int(var)
        if v < 0 or v >= state.num_vars:
            raise ValueError("var out of range")

        a = int(state.assignment[v])
        if a != 0:
            return {int(a)}

        open_var = self._open_decision_var(state)
        if open_var is not None and int(open_var) == int(v):
            top = state.decision_stack[-1]
            dom: set[int] = set()
            if (int(top.failed_mask) & _val_bit(-1)) == 0:
                dom.add(-1)
            if (int(top.failed_mask) & _val_bit(1)) == 0:
                dom.add(1)
            return dom

        return {-1, 1}

    def _all_satisfied(self, state: SatState) -> bool:
        for cl in state.clauses:
            if not any(_lit_value(state.assignment, int(l)) == 1 for l in cl):
                return False
        return True

    def _is_valid(self, action: SatAction, state: SatState) -> tuple[bool, str]:
        open_var = self._open_decision_var(state)

        if action.type == SatActionType.SELECT_VAR:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if state.conflict_clause is not None:
                return False, "conflict: must BACKTRACK/DONE"
            if state.selected_var is not None:
                return False, "var already selected"
            if action.target is None:
                return False, "SELECT_VAR missing target"
            var = int(action.target)
            if var < 0 or var >= state.num_vars:
                return False, "var idx out of range"
            if int(state.assignment[var]) != 0:
                return False, "var already assigned"
            if open_var is not None and int(var) != int(open_var):
                return False, "must re-select open decision var"
            if len(self._effective_domain(state, var)) == 0:
                return False, "var domain empty"
            return True, ""

        if action.type == SatActionType.ASSIGN_VALUE:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if state.conflict_clause is not None:
                return False, "conflict: must BACKTRACK/DONE"
            if state.selected_var is None:
                return False, "no var selected"
            if action.target is None:
                return False, "ASSIGN_VALUE missing target"
            t = int(action.target)
            if t not in {0, 1}:
                return False, "ASSIGN_VALUE target must be 0/1"
            var = int(state.selected_var)
            if var < 0 or var >= state.num_vars:
                return False, "selected_var out of range"
            if int(state.assignment[var]) != 0:
                return False, "var already assigned"
            if open_var is not None and int(var) != int(open_var):
                return False, "must assign open decision var"
            val = 1 if t == 1 else -1
            if val not in self._effective_domain(state, var):
                return False, "value not in domain"
            return True, ""

        if action.type == SatActionType.PROPAGATE:
            if not state.propagation_pending:
                return False, "no pending assignment"
            return True, ""

        if action.type == SatActionType.BACKTRACK:
            if not state.decision_stack:
                return False, "stack empty"
            return True, ""

        if action.type == SatActionType.DONE:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if state.selected_var is not None:
                return False, "var still selected"
            return True, ""

        return False, f"Unknown action type: {action.type}"

    def _move_watch(
        self, state: SatState, clause_id: int, old_lit: int, new_lit: int
    ) -> None:
        old = int(old_lit)
        new = int(new_lit)
        cid = int(clause_id)

        lst = state.watch_list.get(old)
        if lst is not None:
            try:
                lst.remove(cid)
            except ValueError:
                pass
            if len(lst) == 0:
                del state.watch_list[old]

        state.watch_list.setdefault(new, []).append(cid)

    def _assign_literal(
        self,
        state: SatState,
        lit: int,
        *,
        decision_level: int,
        reason_clause: Optional[int],
    ) -> bool:
        """Assign literal to True.

        Returns True iff a new assignment was made.
        On conflict, sets state.conflict_clause.
        """

        l = int(lit)
        var = _lit_var(l)
        val = _lit_sign(l)

        cur = int(state.assignment[var])
        if cur == int(val):
            return False
        if cur == int(-val):
            # Contradiction: variable already assigned opposite.
            if state.conflict_clause is None:
                state.conflict_clause = (
                    int(reason_clause) if reason_clause is not None else -3
                )
            state.propagation_queue.clear()
            state.propagation_pending = False
            return False

        state.assignment[var] = int(val)
        state.var_reason[var] = None if reason_clause is None else int(reason_clause)
        state.trail.append(int(l))
        state.trail_levels.append(int(decision_level))

        # Enqueue falsified literal (the negation of assigned literal).
        state.propagation_queue.append(int(-l))
        return True

    def _undo_to(self, state: SatState, trail_start: int) -> None:
        tgt = int(trail_start)
        if tgt < 0 or tgt > len(state.trail):
            raise ValueError("trail_start out of range")
        while len(state.trail) > tgt:
            lit = int(state.trail.pop())
            _lvl = int(state.trail_levels.pop())
            v = _lit_var(lit)
            state.assignment[v] = 0
            state.var_reason[v] = None

    def _maybe_decay_activity(self, state: SatState) -> None:
        # Very small VSIDS-lite: periodic decay.
        if int(state.step_count) > 0 and int(state.step_count) % 100 == 0:
            state.activity *= np.float32(0.95)

    def _bump_activity_on_conflict(self, state: SatState, conflict_clause: int) -> None:
        cid = int(conflict_clause)
        if cid < 0 or cid >= state.num_clauses:
            return
        cl = state.clauses[cid]
        for lit in cl:
            v = _lit_var(int(lit))
            state.activity[v] += np.float32(1.0)

    def _propagate(self, state: SatState) -> None:
        """Unit propagation to fixpoint using watched literals."""

        if state.conflict_clause is not None:
            return

        decision_level = int(len(state.decision_stack))

        while state.propagation_queue:
            fals_lit = int(state.propagation_queue.pop())
            watching = list(state.watch_list.get(fals_lit, []))

            for cid in watching:
                if state.conflict_clause is not None:
                    state.propagation_queue.clear()
                    break

                cl = state.clauses[int(cid)]
                w0, w1 = state.watch_pos[int(cid)]

                lit0 = int(cl[int(w0)])
                lit1 = int(cl[int(w1)])

                if int(lit0) == int(fals_lit):
                    fals_pos = int(w0)
                    other_pos = int(w1)
                elif int(lit1) == int(fals_lit):
                    fals_pos = int(w1)
                    other_pos = int(w0)
                else:
                    # Clause moved watch already.
                    continue

                other_lit = int(cl[int(other_pos)])
                other_val = _lit_value(state.assignment, other_lit)

                if int(other_val) == 1:
                    # Clause already satisfied by other watch.
                    continue

                # Try to find a new literal to watch.
                new_pos: Optional[int] = None
                for k in range(len(cl)):
                    if int(k) == int(fals_pos) or int(k) == int(other_pos):
                        continue
                    cand_lit = int(cl[int(k)])
                    cand_val = _lit_value(state.assignment, cand_lit)
                    if int(cand_val) != -1:
                        new_pos = int(k)
                        break

                if new_pos is not None:
                    new_lit = int(cl[int(new_pos)])

                    # Update watch positions.
                    if int(fals_pos) == int(w0):
                        state.watch_pos[int(cid)] = (int(new_pos), int(other_pos))
                    else:
                        state.watch_pos[int(cid)] = (int(other_pos), int(new_pos))

                    self._move_watch(state, int(cid), int(fals_lit), int(new_lit))
                    continue

                # No new watch found => unit or conflict.
                if int(other_val) == -1:
                    state.conflict_clause = int(cid)
                    state.propagation_queue.clear()
                    break

                if int(other_val) == 0:
                    # Unit clause => propagate other_lit.
                    self._assign_literal(
                        state,
                        int(other_lit),
                        decision_level=decision_level,
                        reason_clause=int(cid),
                    )
                    if state.conflict_clause is not None:
                        state.propagation_queue.clear()
                        break

        # Update VSIDS-lite on conflict.
        if state.conflict_clause is not None:
            self._bump_activity_on_conflict(state, int(state.conflict_clause))
            self._maybe_decay_activity(state)

    def _backtrack(self, state: SatState) -> None:
        """Backtrack one decision level and record branch-local nogood."""

        if not state.decision_stack:
            raise RuntimeError("Cannot backtrack: decision stack empty")

        state.selected_var = None
        state.propagation_pending = False
        state.propagation_queue.clear()

        top = state.decision_stack[-1]
        top_var = int(top.decision_var)

        # Case A: top decision is currently assigned => undo within this decision frame.
        if int(state.assignment[top_var]) != 0:
            bit = _val_bit(int(top.chosen_val))
            top.failed_mask |= int(bit)
            top.tried_mask |= int(bit)

            self._undo_to(state, int(top.trail_start))
            state.conflict_clause = None
            return

        # Case B: top frame is already open (var unassigned) => pop and backtrack parent.
        # Ensure trail is at frame start.
        if len(state.trail) != int(top.trail_start):
            self._undo_to(state, int(top.trail_start))

        state.decision_stack.pop()

        if not state.decision_stack:
            # Exhausted search: represent UNSAT as a root-level conflict sentinel.
            state.conflict_clause = -2
            return

        parent = state.decision_stack[-1]
        pv = int(parent.decision_var)
        if int(state.assignment[pv]) == 0:
            raise RuntimeError(
                "internal error: parent decision var unassigned during backtrack"
            )

        bit = _val_bit(int(parent.chosen_val))
        parent.failed_mask |= int(bit)
        parent.tried_mask |= int(bit)

        self._undo_to(state, int(parent.trail_start))
        state.conflict_clause = None

    def backjump_to(self, target_level: int) -> dict:
        """Backjump to target_level: undo all decisions above target_level.

        After backjump, the decision frame at target_level remains but its
        variable is unassigned with the current value marked as failed.
        The oracle will then retry with the other value or backtrack further.

        Args:
            target_level: 0-indexed decision level to jump to (0 = first decision)

        Returns:
            Info dict with num_popped, final_level, target_var.
        """
        state = self._require_running()
        current_level = int(len(state.decision_stack))
        target_level = int(target_level)

        if target_level < 0 or target_level >= current_level:
            raise ValueError(
                f"target_level {target_level} out of range [0, {current_level})"
            )

        target_frame = state.decision_stack[int(target_level)]

        self._undo_to(state, int(target_frame.trail_start))

        num_popped = int(current_level) - int(target_level) - 1
        state.decision_stack = state.decision_stack[: int(target_level) + 1]

        bit = _val_bit(int(target_frame.chosen_val))
        target_frame.failed_mask |= int(bit)
        target_frame.tried_mask |= int(bit)

        state.conflict_clause = None
        state.propagation_pending = False
        state.propagation_queue.clear()
        state.selected_var = None

        return {
            "num_popped": int(num_popped),
            "final_level": int(len(state.decision_stack)),
            "target_var": int(target_frame.decision_var),
        }

    def cdcl_backjump(
        self,
        backjump_level: int,
        learned_clause: List[int],
        asserting_literal: int,
    ) -> dict:
        """CDCL backjump with clause learning and asserting propagation."""
        state = self._require_running()
        current_level = int(len(state.decision_stack))
        backjump_level = int(backjump_level)

        if backjump_level < 0 or backjump_level >= current_level:
            raise ValueError(
                f"backjump_level {backjump_level} out of range [0, {current_level})"
            )

        if not learned_clause:
            raise ValueError("learned_clause must be non-empty")

        new_clause = tuple(int(l) for l in learned_clause)
        new_cid = int(state.num_clauses)

        state.clauses.append(new_clause)
        state.num_clauses = int(new_cid) + 1

        if len(new_clause) == 1:
            w0 = 0
            w1 = 0
        else:
            w0 = 0
            w1 = 1

        state.watch_pos.append((int(w0), int(w1)))

        lit0 = int(new_clause[int(w0)])
        state.watch_list.setdefault(lit0, []).append(int(new_cid))
        if int(w1) != int(w0):
            lit1 = int(new_clause[int(w1)])
            state.watch_list.setdefault(lit1, []).append(int(new_cid))

        if int(backjump_level) + 1 < int(current_level):
            undo_start = int(state.decision_stack[int(backjump_level) + 1].trail_start)
        else:
            undo_start = int(state.decision_stack[int(current_level) - 1].trail_start)

        self._undo_to(state, int(undo_start))

        num_popped = int(current_level) - int(backjump_level) - 1
        state.decision_stack = state.decision_stack[: int(backjump_level) + 1]

        state.conflict_clause = None
        state.propagation_pending = False
        state.propagation_queue.clear()
        state.selected_var = None

        decision_level = int(len(state.decision_stack))
        assigned = self._assign_literal(
            state,
            int(asserting_literal),
            decision_level=int(decision_level),
            reason_clause=int(new_cid),
        )

        state.propagation_pending = bool(state.conflict_clause is None)

        logger.debug(
            "cdcl_backjump current=%d backjump=%d popped=%d learned_cid=%d clause_len=%d asserting_lit=%d assigned=%s",
            int(current_level),
            int(backjump_level),
            int(num_popped),
            int(new_cid),
            int(len(new_clause)),
            int(asserting_literal),
            bool(assigned),
        )

        return {
            "num_popped": int(num_popped),
            "final_level": int(len(state.decision_stack)),
            "learned_clause_id": int(new_cid),
            "backjump_level": int(backjump_level),
            "asserting_literal": int(asserting_literal),
        }

    def step(self, action: SatAction) -> StepResult:
        state = self._require_running()
        state.step_count += 1

        valid, reason = self._is_valid(action, state)

        info = {
            "valid": bool(valid),
            "reason": str(reason),
            "action": action.to_token(),
            "step_count": int(state.step_count),
        }

        reward = float(self.step_penalty)

        if not valid:
            reward = float(self.invalid_penalty)
            if self.mode == "strict":
                state.status = SatEnvStatus.FAILURE
                state.termination_reason = "invalid"
        else:
            if action.type == SatActionType.SELECT_VAR:
                state.selected_var = int(action.target)  # type: ignore[arg-type]

            elif action.type == SatActionType.ASSIGN_VALUE:
                var = int(state.selected_var)  # type: ignore[arg-type]
                t = int(action.target)  # type: ignore[arg-type]
                val = 1 if int(t) == 1 else -1

                open_var = self._open_decision_var(state)

                if open_var is not None:
                    # Re-assign within the open decision frame.
                    top = state.decision_stack[-1]
                    if int(top.decision_var) != int(var):
                        raise RuntimeError("internal error: open decision var mismatch")
                    if len(state.trail) != int(top.trail_start):
                        raise RuntimeError(
                            "internal error: trail not at open frame start"
                        )

                    top.chosen_val = int(val)
                    top.tried_mask |= int(_val_bit(int(val)))

                    lit = int((var + 1) * int(val))
                    self._assign_literal(
                        state,
                        lit,
                        decision_level=int(len(state.decision_stack)),
                        reason_clause=None,
                    )

                else:
                    # New decision.
                    frame = DecisionFrame(
                        decision_var=int(var),
                        chosen_val=int(val),
                        trail_start=int(len(state.trail)),
                        tried_mask=int(_val_bit(int(val))),
                        failed_mask=0,
                    )
                    state.decision_stack.append(frame)

                    lit = int((var + 1) * int(val))
                    self._assign_literal(
                        state,
                        lit,
                        decision_level=int(len(state.decision_stack)),
                        reason_clause=None,
                    )

                state.selected_var = None
                state.propagation_pending = True

            elif action.type == SatActionType.PROPAGATE:
                self._propagate(state)
                state.propagation_pending = False

            elif action.type == SatActionType.BACKTRACK:
                self._backtrack(state)

            elif action.type == SatActionType.DONE:
                if self._all_satisfied(state) and state.conflict_clause is None:
                    state.status = SatEnvStatus.SUCCESS
                    state.termination_reason = "sat"
                    reward = float(self.goal_reward)
                else:
                    state.status = SatEnvStatus.FAILURE
                    if state.conflict_clause is not None and (not state.decision_stack):
                        state.termination_reason = "unsat"
                    else:
                        state.termination_reason = "done_failure"
                    reward = 0.0

        if state.status == SatEnvStatus.RUNNING and int(state.step_count) >= int(
            self.max_steps
        ):
            state.status = SatEnvStatus.FAILURE
            state.termination_reason = "timeout"
            info["reason"] = "step_limit"

        info.update(
            {
                "status": state.status.name,
                "termination_reason": None
                if state.termination_reason is None
                else str(state.termination_reason),
                "selected_var": -1
                if state.selected_var is None
                else int(state.selected_var),
                "propagation_pending": bool(state.propagation_pending),
                "conflict_clause": -1
                if state.conflict_clause is None
                else int(state.conflict_clause),
                "num_assigned": _num_assigned(state.assignment),
                "stack_depth": int(len(state.decision_stack)),
                "trail_len": int(len(state.trail)),
                "queue_len": int(len(state.propagation_queue)),
                "solved": int(
                    self._all_satisfied(state) and state.conflict_clause is None
                ),
            }
        )

        obs = self.get_observation(state)
        done = state.status != SatEnvStatus.RUNNING
        return StepResult(
            observation=obs, reward=float(reward), done=bool(done), info=info
        )

    def get_valid_actions(self) -> List[SatAction]:
        if self._state is None:
            raise RuntimeError("Environment not reset")
        if self._state.status != SatEnvStatus.RUNNING:
            return []

        state = self._state
        actions: List[SatAction] = []

        # SELECT_VAR candidates.
        if (
            state.selected_var is None
            and (not state.propagation_pending)
            and state.conflict_clause is None
        ):
            open_var = self._open_decision_var(state)
            if open_var is not None:
                a = SatAction.select_var(int(open_var))
                ok, _ = self._is_valid(a, state)
                if ok:
                    actions.append(a)
            else:
                for v in range(state.num_vars):
                    a = SatAction.select_var(int(v))
                    ok, _ = self._is_valid(a, state)
                    if ok:
                        actions.append(a)

        # ASSIGN_VALUE candidates.
        if state.selected_var is not None:
            for t in [0, 1]:
                a = SatAction.assign_value(int(t))
                ok, _ = self._is_valid(a, state)
                if ok:
                    actions.append(a)

        # Operator actions.
        for a in [SatAction.propagate(), SatAction.backtrack(), SatAction.done()]:
            ok, _ = self._is_valid(a, state)
            if ok:
                actions.append(a)

        return actions

    def get_observation(self, state: SatState) -> dict:
        selected = -1 if state.selected_var is None else int(state.selected_var)
        conflict_flag = int(state.conflict_clause is not None)

        global_features = [
            int(selected),
            int(_num_assigned(state.assignment)),
            int(conflict_flag),
            int(bool(state.propagation_pending)),
            int(len(state.decision_stack)),
        ]

        open_var = self._open_decision_var(state)
        top_failed_mask = 0
        if open_var is not None:
            top_failed_mask = int(state.decision_stack[-1].failed_mask)

        vars_feat: List[List[int]] = []
        var_domain_mask: List[List[bool]] = []

        for v in range(state.num_vars):
            a = int(state.assignment[v])
            if a == 0:
                assigned_idx = 0
            elif a == -1:
                assigned_idx = 1
            else:
                assigned_idx = 2

            is_sel = int(v == selected)

            if a == 0:
                if open_var is not None and int(open_var) == int(v):
                    allow_false = (top_failed_mask & _val_bit(-1)) == 0
                    allow_true = (top_failed_mask & _val_bit(1)) == 0
                else:
                    allow_false = True
                    allow_true = True
            else:
                allow_false = a == -1
                allow_true = a == 1

            dom_size = int(bool(allow_false)) + int(bool(allow_true))

            act = float(state.activity[v])
            act_clip = max(0.0, min(float(self.activity_clip), max(0.0, act)))
            if float(self.activity_clip) <= 0.0:
                act_bin = 0
            else:
                act_bin = int(
                    round(
                        act_clip
                        / float(self.activity_clip)
                        * float(self.activity_bins - 1)
                    )
                )
                act_bin = max(0, min(int(self.activity_bins - 1), int(act_bin)))

            vars_feat.append(
                [int(v), int(assigned_idx), int(is_sel), int(dom_size), int(act_bin)]
            )
            var_domain_mask.append([bool(allow_false), bool(allow_true)])

        clause_feat: List[List[int]] = []
        for cid, cl in enumerate(state.clauses):
            vals = [_lit_value(state.assignment, int(l)) for l in cl]
            sat = int(any(int(x) == 1 for x in vals))
            num_true = int(sum(1 for x in vals if int(x) == 1))
            num_unassigned = int(sum(1 for x in vals if int(x) == 0))
            is_conf = int(state.conflict_clause == cid)
            clause_feat.append(
                [int(cid), int(sat), int(num_unassigned), int(num_true), int(is_conf)]
            )

        obs = {
            "meta": [int(state.num_vars), int(state.num_clauses)],
            "global": global_features,
            "vars": vars_feat,
            "var_domain_mask": var_domain_mask,
            "clauses": self._clauses_obs,
            "clause_features": clause_feat,
            "propagation_pending": bool(state.propagation_pending),
            "conflict_clause": -1
            if state.conflict_clause is None
            else int(state.conflict_clause),
        }
        return obs


if __name__ == "__main__":
    # Smoke test: solve a planted instance with the oracle.
    from .generator import SatGenerator
    from .oracle import SatOracle

    gen = SatGenerator(seed=0)
    inst = gen.generate_planted(num_vars=30, alpha=3.5)

    env = SatEnv(
        clauses=inst.clauses,
        num_vars=inst.num_vars,
        planted_solution=inst.planted_solution,
        mode="strict",
    )
    oracle = SatOracle(env)

    trace = oracle.solve()
    st = env.get_state()

    assert st.status == SatEnvStatus.SUCCESS
    assert env._all_satisfied(st)

    print(
        f"env.py smoke test passed (steps={len(trace)} stack_depth={len(st.decision_stack)})"
    )
