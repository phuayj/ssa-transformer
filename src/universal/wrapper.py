from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Optional, Tuple

import numpy as np

from .types import ConstraintType, UnifiedAction, UnifiedActionType, UnifiedObservation

try:
    from hybrid.dsl import HybridAction
    from hybrid.env import HybridEnv

    _HAS_HYBRID = True
except ImportError:
    _HAS_HYBRID = False

logger = logging.getLogger(__name__)


class UnifiedEnvWrapper(ABC):
    """Abstract wrapper converting a domain-specific env to a unified interface."""

    env: Any
    max_domain: int

    def reset(self) -> UnifiedObservation:
        """Reset and return initial observation."""
        self.env.reset()
        return self._state_to_obs()

    @abstractmethod
    def step(
        self, action: UnifiedAction
    ) -> Tuple[UnifiedObservation, float, bool, dict]:
        """Execute action, return (obs, reward, done, info)."""

    @abstractmethod
    def get_valid_actions(self) -> list[UnifiedAction]:
        """Get list of valid actions in current state."""

    @property
    @abstractmethod
    def domain_id(self) -> int:
        """Return domain identifier (0=CSP, 1=Coloring, 2=SAT)."""

    @abstractmethod
    def _state_to_obs(self) -> UnifiedObservation:
        """Convert current state to unified observation."""
        ...

    @abstractmethod
    def _build_constraint_graph(self) -> None:
        """Build constraint graph structure for this domain."""
        ...

    def _build_var_features(
        self,
        var_assigned: np.ndarray,
        var_domain_mask: np.ndarray,
        selected_var: Optional[int],
        num_vars: int,
    ) -> np.ndarray:
        """Build variable features array [is_assigned, domain_size_norm, is_selected]."""
        var_features = np.zeros((num_vars, 3), dtype=np.float32)
        for i in range(num_vars):
            var_features[i, 0] = float(var_assigned[i] != -1)
            var_features[i, 1] = float(var_domain_mask[i].sum()) / float(
                self.max_domain
            )
            var_features[i, 2] = (
                float(selected_var == i) if selected_var is not None else 0.0
            )
        return var_features

    def _build_edge_features(
        self,
        edge_pos: np.ndarray,
        edge_sign: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build edge features array with position and optional sign."""
        num_edges = len(edge_pos)
        edge_features = np.zeros((num_edges, 2), dtype=np.float32)
        edge_features[:, 0] = edge_pos / 3.0
        if edge_sign is not None:
            edge_features[:, 1] = edge_sign
        return edge_features


class CSPUnifiedWrapper(UnifiedEnvWrapper):
    """Wrapper for Sudoku-like CSP environment."""

    def __init__(
        self, env, max_domain: int = 9, propagation_mode: str = "forward_check"
    ):
        """Args:
        env: CSPEnv / SudokuEnv from src/csp/env.py
        max_domain: padding domain size (>=grid_size)
        """

        self.env = env
        self.max_domain = int(max_domain)
        grid_size = int(getattr(self.env, "config").grid_size)
        if self.max_domain < grid_size:
            raise ValueError(
                f"max_domain must be >= grid_size ({grid_size}); got {self.max_domain}"
            )
        self.grid_size = grid_size
        self.num_cells = int(grid_size * grid_size)

        propagation_mode = str(propagation_mode)
        if propagation_mode not in {"none", "forward_check"}:
            raise ValueError(
                "propagation_mode must be 'none' or 'forward_check' "
                f"(got {propagation_mode!r})"
            )
        self.propagation_mode_id = 0 if propagation_mode == "none" else 1

        self._build_constraint_graph()

    def _build_constraint_graph(self) -> None:
        """Build factor graph from CSP constraints."""

        use_box = bool(getattr(self.env, "use_box_constraints", False))
        config = getattr(self.env, "config")

        # Decompose ALLDIFF (row/col/box) into NEQ binary constraints.
        pairs: set[tuple[int, int]] = set()

        # Row constraints.
        for r in range(self.grid_size):
            cells = [r * self.grid_size + c for c in range(self.grid_size)]
            for i in range(self.grid_size):
                for j in range(i + 1, self.grid_size):
                    a, b = int(cells[i]), int(cells[j])
                    pairs.add((a, b) if a < b else (b, a))

        # Column constraints.
        for c in range(self.grid_size):
            cells = [r * self.grid_size + c for r in range(self.grid_size)]
            for i in range(self.grid_size):
                for j in range(i + 1, self.grid_size):
                    a, b = int(cells[i]), int(cells[j])
                    pairs.add((a, b) if a < b else (b, a))

        # Optional box constraints.
        if use_box:
            num_box_rows = int(self.grid_size // config.box_height)
            num_box_cols = int(self.grid_size // config.box_width)
            for br in range(num_box_rows):
                for bc in range(num_box_cols):
                    cells: list[int] = []
                    for dr in range(config.box_height):
                        for dc in range(config.box_width):
                            cells.append(
                                (br * config.box_height + dr) * self.grid_size
                                + (bc * config.box_width + dc)
                            )
                    for i in range(len(cells)):
                        for j in range(i + 1, len(cells)):
                            a, b = int(cells[i]), int(cells[j])
                            pairs.add((a, b) if a < b else (b, a))

        self.constraints = sorted(pairs)
        self.num_constraints = int(len(self.constraints))

        edge_con: list[int] = []
        edge_var: list[int] = []
        edge_pos: list[int] = []

        for c_idx, (v1, v2) in enumerate(self.constraints):
            edge_con.extend([int(c_idx), int(c_idx)])
            edge_var.extend([int(v1), int(v2)])
            edge_pos.extend([0, 1])

        self.edge_con_idx = np.asarray(edge_con, dtype=np.int64)
        self.edge_var_idx = np.asarray(edge_var, dtype=np.int64)
        self.edge_pos = np.asarray(edge_pos, dtype=np.int64)

    def _current_depth(self, state) -> int:
        # Environment stores nogoods keyed by the 1-indexed depth of the *next* assignment.
        return int(len(state.assignment_stack) + 1)

    def _effective_domain(self, state, cell_idx: int) -> set[int]:
        idx = int(cell_idx)
        r, c = idx // self.grid_size, idx % self.grid_size
        if int(state.grid[r, c]) != 0:
            return set(int(v) for v in state.domains[idx])

        dom = set(int(v) for v in state.domains[idx])
        banned = state.nogoods.get(self._current_depth(state), {}).get(idx)
        if banned:
            dom.difference_update(int(v) for v in banned)
        return dom

    def _has_contradiction(self, state) -> bool:
        for idx in range(self.num_cells):
            r, c = idx // self.grid_size, idx % self.grid_size
            if int(state.grid[r, c]) != 0:
                continue
            if len(self._effective_domain(state, idx)) == 0:
                return True
        return False

    def _state_to_obs(self) -> UnifiedObservation:
        state = self.env.get_state()
        n_vars = int(self.num_cells)

        # Variable assignment [-1 or 0..grid_size-1]
        var_assigned = np.full((n_vars,), -1, dtype=np.int64)
        for i in range(n_vars):
            r, c = i // self.grid_size, i % self.grid_size
            val = int(state.grid[r, c])
            if val != 0:
                var_assigned[i] = int(val - 1)

        # Branch-local nogoods (at current depth).
        var_nogood_mask = np.zeros((n_vars, self.max_domain), dtype=bool)
        depth = self._current_depth(state)
        per_depth = state.nogoods.get(int(depth), {})
        for cell_idx, failed_vals in per_depth.items():
            for v in failed_vals:
                vv = int(v)
                if 1 <= vv <= self.max_domain:
                    var_nogood_mask[int(cell_idx), int(vv - 1)] = True

        # Variable domain mask [num_cells, max_domain] (effective domain; nogoods removed).
        var_domain_mask = np.zeros((n_vars, self.max_domain), dtype=bool)
        for i in range(n_vars):
            for v in self._effective_domain(state, i):
                vv = int(v)
                if 1 <= vv <= self.max_domain:
                    var_domain_mask[i, int(vv - 1)] = True

        # Variable features: [is_assigned, domain_size_norm, is_selected]
        selected_cell = (
            int(state.selected_cell) if state.selected_cell is not None else None
        )
        var_features = self._build_var_features(
            var_assigned=var_assigned,
            var_domain_mask=var_domain_mask,
            selected_var=selected_cell,
            num_vars=n_vars,
        )

        # Constraint tensors
        con_type = np.full(
            (self.num_constraints,), int(ConstraintType.NEQ), dtype=np.int64
        )

        con_scope_padded = np.full((self.num_constraints, 4), -1, dtype=np.int64)
        if self.constraints:
            con_scope_padded[:, :2] = np.asarray(self.constraints, dtype=np.int64)

        # Constraint features: [arity_norm, is_satisfied]
        con_features = np.zeros((self.num_constraints, 2), dtype=np.float32)
        con_features[:, 0] = np.float32(2.0 / 4.0)  # arity=2 normalized by R_max=4

        if self.max_domain > 0:
            avg_domain = float(var_domain_mask.sum()) / float(n_vars * self.max_domain)
            logger.debug(
                "CSPUnifiedWrapper obs grid_size=%s avg_domain=%.3f",
                self.grid_size,
                avg_domain,
            )

        for c_idx, (v1, v2) in enumerate(self.constraints):
            a1, a2 = int(var_assigned[int(v1)]), int(var_assigned[int(v2)])
            if a1 == -1 or a2 == -1:
                con_features[int(c_idx), 1] = np.float32(0.5)  # unknown
            else:
                con_features[int(c_idx), 1] = np.float32(float(a1 != a2))

        # Edge features: [position_norm, sign (unused for NEQ)]
        edge_features = self._build_edge_features(self.edge_pos)

        return UnifiedObservation(
            var_domain_mask=var_domain_mask,
            var_nogood_mask=var_nogood_mask,
            var_assigned=var_assigned,
            var_features=var_features,
            con_type=con_type,
            con_scope=con_scope_padded,
            con_features=con_features,
            edge_con_idx=self.edge_con_idx,
            edge_var_idx=self.edge_var_idx,
            edge_features=edge_features,
            num_vars=int(n_vars),
            num_constraints=int(self.num_constraints),
            max_domain=int(self.max_domain),
            stack_depth=int(len(state.assignment_stack)),
            propagation_pending=bool(state.propagation_pending),
            has_conflict=bool(self._has_contradiction(state)),
            propagation_mode=int(self.propagation_mode_id),
            domain_id=0,
        )

    def step(
        self, action: UnifiedAction
    ) -> Tuple[UnifiedObservation, float, bool, dict]:
        from csp.dsl import CSPAction

        state = self.env.get_state()

        if action.type == UnifiedActionType.ASSIGN:
            if action.var is None or action.value is None:
                raise ValueError("ASSIGN requires var and value")

            var = int(action.var)
            val0 = int(action.value)

            if state.selected_cell is not None and int(state.selected_cell) != int(var):
                # Wrapper invariant violated: we only support assigning the currently selected cell.
                return (
                    self._state_to_obs(),
                    0.0,
                    False,
                    {"valid": False, "reason": "cell already selected"},
                )

            if state.selected_cell is None:
                sel_res = self.env.step(CSPAction.select_cell(int(var)))
                if (not bool(sel_res.info.get("valid", True))) or bool(sel_res.done):
                    return (
                        self._state_to_obs(),
                        float(sel_res.reward),
                        bool(sel_res.done),
                        dict(sel_res.info),
                    )

            res = self.env.step(CSPAction.assign_value(int(val0 + 1)))
            if (not bool(res.info.get("valid", True))) or bool(res.done):
                return (
                    self._state_to_obs(),
                    float(res.reward),
                    bool(res.done),
                    dict(res.info),
                )

            new_state = self.env.get_state()
            if bool(new_state.propagation_pending):
                res = self.env.step(CSPAction.propagate())

        elif action.type == UnifiedActionType.BACKTRACK:
            res = self.env.step(CSPAction.backtrack())

        elif action.type == UnifiedActionType.DONE:
            res = self.env.step(CSPAction.done())

        else:
            raise ValueError(f"Unknown action type: {action.type}")

        return self._state_to_obs(), float(res.reward), bool(res.done), dict(res.info)

    def get_valid_actions(self) -> list[UnifiedAction]:
        state = self.env.get_state()

        # If propagation is pending, the underlying env requires PROPAGATE; the unified interface
        # does not expose it, so expose only BACKTRACK (if possible) and DONE.
        if bool(state.propagation_pending):
            actions: list[UnifiedAction] = []
            if state.assignment_stack:
                actions.append(UnifiedAction.backtrack())
            return actions

        # Contradiction: CSP requires BACKTRACK.
        if self._has_contradiction(state):
            if state.assignment_stack:
                return [UnifiedAction.backtrack()]
            return [UnifiedAction.done()]

        depth = self._current_depth(state)
        per_depth = state.nogoods.get(int(depth), {})

        actions: list[UnifiedAction] = []

        # If a cell is already selected, only allow assigning it.
        candidate_vars: list[int]
        if state.selected_cell is None:
            candidate_vars = [i for i in range(self.num_cells)]
        else:
            candidate_vars = [int(state.selected_cell)]

        for var_idx in candidate_vars:
            r, c = int(var_idx) // self.grid_size, int(var_idx) % self.grid_size
            if int(state.grid[r, c]) != 0:
                continue

            banned = per_depth.get(int(var_idx), set())

            for v in sorted(int(x) for x in state.domains[int(var_idx)]):
                if int(v) in banned:
                    continue
                actions.append(UnifiedAction.assign(int(var_idx), int(v - 1)))

        if state.assignment_stack:
            actions.append(UnifiedAction.backtrack())

        # Only allow DONE when all cells are assigned (prevents premature DONE)
        all_assigned = all(
            int(state.grid[i // self.grid_size, i % self.grid_size]) != 0
            for i in range(self.num_cells)
        )
        if state.selected_cell is None and all_assigned:
            actions.append(UnifiedAction.done())

        return actions

    @property
    def domain_id(self) -> int:
        return 0


class GraphColoringUnifiedWrapper(UnifiedEnvWrapper):
    """Wrapper for Graph k-Coloring environment."""

    def __init__(
        self, env, max_domain: int = 10, propagation_mode: str = "forward_check"
    ):
        self.env = env
        self.max_domain = int(max_domain)
        if self.max_domain < 1:
            raise ValueError("max_domain must be >= 1")
        propagation_mode = str(propagation_mode)
        if propagation_mode not in {"none", "forward_check"}:
            raise ValueError(
                f"propagation_mode must be 'none' or 'forward_check'; got {propagation_mode!r}"
            )
        self.propagation_mode_id = 0 if propagation_mode == "none" else 1
        self._build_constraint_graph()

    def _build_constraint_graph(self) -> None:
        adj = np.asarray(self.env.adjacency, dtype=bool)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("env.adjacency must be square")

        n = int(adj.shape[0])
        self.num_vars = int(n)

        constraints: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if bool(adj[i, j]):
                    constraints.append((int(i), int(j)))

        self.constraints = constraints
        self.num_constraints = int(len(constraints))

        edge_con: list[int] = []
        edge_var: list[int] = []
        edge_pos: list[int] = []

        for c_idx, (v1, v2) in enumerate(constraints):
            edge_con.extend([int(c_idx), int(c_idx)])
            edge_var.extend([int(v1), int(v2)])
            edge_pos.extend([0, 1])

        self.edge_con_idx = np.asarray(edge_con, dtype=np.int64)
        self.edge_var_idx = np.asarray(edge_var, dtype=np.int64)
        self.edge_pos = np.asarray(edge_pos, dtype=np.int64)

    def _current_depth(self, state) -> int:
        return int(len(state.assignment_stack) + 1)

    def _effective_domain(self, state, node: int) -> set[int]:
        n = int(node)
        if int(state.assignment[n]) != 0:
            return set(int(v) for v in state.domains[n])

        dom = set(int(v) for v in state.domains[n])
        banned = state.nogoods.get(self._current_depth(state), {}).get(n)
        if banned:
            dom.difference_update(int(v) for v in banned)
        return dom

    def _has_contradiction(self, state) -> bool:
        # Empty effective domain.
        for i in range(int(state.num_nodes)):
            if int(state.assignment[i]) != 0:
                continue
            if len(self._effective_domain(state, i)) == 0:
                return True
        # Coloring conflict.
        for i in range(int(state.num_nodes)):
            c = int(state.assignment[i])
            if c == 0:
                continue
            if bool(np.any(state.adjacency[i] & (state.assignment == c))):
                return True
        return False

    def _state_to_obs(self) -> UnifiedObservation:
        state = self.env.get_state()
        n = int(state.num_nodes)

        # Variable assignment
        var_assigned = np.full((n,), -1, dtype=np.int64)
        for i in range(n):
            if int(state.assignment[i]) != 0:
                var_assigned[i] = int(state.assignment[i] - 1)

        # Branch-local nogoods
        var_nogood_mask = np.zeros((n, self.max_domain), dtype=bool)
        depth = self._current_depth(state)
        per_depth = state.nogoods.get(int(depth), {})
        for node, failed_colors in per_depth.items():
            for c in failed_colors:
                cc = int(c)
                if 1 <= cc <= self.max_domain:
                    var_nogood_mask[int(node), int(cc - 1)] = True

        # Variable domain mask (effective domain)
        var_domain_mask = np.zeros((n, self.max_domain), dtype=bool)
        for i in range(n):
            for c in self._effective_domain(state, i):
                cc = int(c)
                if 1 <= cc <= self.max_domain:
                    var_domain_mask[i, int(cc - 1)] = True

        # Variable features
        selected_node = (
            int(state.selected_node) if state.selected_node is not None else None
        )
        var_features = self._build_var_features(
            var_assigned=var_assigned,
            var_domain_mask=var_domain_mask,
            selected_var=selected_node,
            num_vars=n,
        )

        # Constraint tensors
        con_type = np.full(
            (self.num_constraints,), int(ConstraintType.NEQ), dtype=np.int64
        )

        con_scope_padded = np.full((self.num_constraints, 4), -1, dtype=np.int64)
        if self.constraints:
            con_scope_padded[:, :2] = np.asarray(self.constraints, dtype=np.int64)

        con_features = np.zeros((self.num_constraints, 2), dtype=np.float32)
        con_features[:, 0] = np.float32(2.0 / 4.0)

        for c_idx, (v1, v2) in enumerate(self.constraints):
            a1, a2 = int(var_assigned[int(v1)]), int(var_assigned[int(v2)])
            if a1 == -1 or a2 == -1:
                con_features[int(c_idx), 1] = np.float32(0.5)
            else:
                con_features[int(c_idx), 1] = np.float32(float(a1 != a2))

        # Edge features
        edge_features = self._build_edge_features(self.edge_pos)

        return UnifiedObservation(
            var_domain_mask=var_domain_mask,
            var_nogood_mask=var_nogood_mask,
            var_assigned=var_assigned,
            var_features=var_features,
            con_type=con_type,
            con_scope=con_scope_padded,
            con_features=con_features,
            edge_con_idx=self.edge_con_idx,
            edge_var_idx=self.edge_var_idx,
            edge_features=edge_features,
            num_vars=int(n),
            num_constraints=int(self.num_constraints),
            max_domain=int(self.max_domain),
            stack_depth=int(len(state.assignment_stack)),
            propagation_pending=bool(state.propagation_pending),
            has_conflict=bool(self._has_contradiction(state)),
            propagation_mode=int(self.propagation_mode_id),
            domain_id=1,
        )

    def step(
        self, action: UnifiedAction
    ) -> Tuple[UnifiedObservation, float, bool, dict]:
        from graph_coloring.dsl import GraphColorAction

        state = self.env.get_state()

        if action.type == UnifiedActionType.ASSIGN:
            if action.var is None or action.value is None:
                raise ValueError("ASSIGN requires var and value")

            var = int(action.var)
            val0 = int(action.value)

            if state.selected_node is not None and int(state.selected_node) != int(var):
                return (
                    self._state_to_obs(),
                    0.0,
                    False,
                    {"valid": False, "reason": "node already selected"},
                )

            if state.selected_node is None:
                sel_res = self.env.step(GraphColorAction.select_node(int(var)))
                if (not bool(sel_res.info.get("valid", True))) or bool(sel_res.done):
                    return (
                        self._state_to_obs(),
                        float(sel_res.reward),
                        bool(sel_res.done),
                        dict(sel_res.info),
                    )

            res = self.env.step(GraphColorAction.assign_color(int(val0 + 1)))
            if (not bool(res.info.get("valid", True))) or bool(res.done):
                return (
                    self._state_to_obs(),
                    float(res.reward),
                    bool(res.done),
                    dict(res.info),
                )

            new_state = self.env.get_state()
            if bool(new_state.propagation_pending):
                res = self.env.step(GraphColorAction.propagate())

        elif action.type == UnifiedActionType.BACKTRACK:
            res = self.env.step(GraphColorAction.backtrack())

        elif action.type == UnifiedActionType.DONE:
            res = self.env.step(GraphColorAction.done())

        else:
            raise ValueError(f"Unknown action type: {action.type}")

        return self._state_to_obs(), float(res.reward), bool(res.done), dict(res.info)

    def get_valid_actions(self) -> list[UnifiedAction]:
        state = self.env.get_state()

        if bool(state.propagation_pending):
            actions: list[UnifiedAction] = []
            if state.assignment_stack:
                actions.append(UnifiedAction.backtrack())
            return actions

        if self._has_contradiction(state):
            if state.assignment_stack:
                return [UnifiedAction.backtrack()]
            return [UnifiedAction.done()]

        depth = self._current_depth(state)
        per_depth = state.nogoods.get(int(depth), {})

        actions: list[UnifiedAction] = []

        candidate_vars: list[int]
        if state.selected_node is None:
            candidate_vars = [i for i in range(int(state.num_nodes))]
        else:
            candidate_vars = [int(state.selected_node)]

        for node in candidate_vars:
            if int(state.assignment[int(node)]) != 0:
                continue

            banned = per_depth.get(int(node), set())
            for c in sorted(int(x) for x in state.domains[int(node)]):
                if int(c) in banned:
                    continue
                actions.append(UnifiedAction.assign(int(node), int(c - 1)))

        if state.assignment_stack:
            actions.append(UnifiedAction.backtrack())

        # Only allow DONE when all nodes are assigned (prevents premature DONE)
        all_assigned = all(
            int(state.assignment[i]) != 0 for i in range(int(state.num_nodes))
        )
        if state.selected_node is None and all_assigned:
            actions.append(UnifiedAction.done())

        return actions

    @property
    def domain_id(self) -> int:
        return 1


class SATUnifiedWrapper(UnifiedEnvWrapper):
    """Wrapper for 3-SAT environment."""

    def __init__(
        self, env, max_domain: int = 2, propagation_mode: str = "forward_check"
    ):
        self.env = env
        self.max_domain = int(max_domain)
        if self.max_domain < 2:
            raise ValueError("max_domain must be >= 2 for SAT")
        propagation_mode = str(propagation_mode)
        if propagation_mode != "forward_check":
            raise ValueError(
                "SATUnifiedWrapper supports only watched-literals propagation"
            )
        self.propagation_mode_id = 1
        self._build_constraint_graph()

    def _build_constraint_graph(self) -> None:
        self.num_vars = int(self.env.num_vars)
        self.clauses = list(self.env.clauses)
        self.num_constraints = int(len(self.clauses))

        edge_con: list[int] = []
        edge_var: list[int] = []
        edge_pos: list[int] = []
        edge_sign: list[float] = []

        for c_idx, clause in enumerate(self.clauses):
            for pos, lit in enumerate(clause):
                var = abs(int(lit)) - 1
                sign = 1.0 if int(lit) > 0 else -1.0
                edge_con.append(int(c_idx))
                edge_var.append(int(var))
                edge_pos.append(int(pos))
                edge_sign.append(float(sign))

        self.edge_con_idx = np.asarray(edge_con, dtype=np.int64)
        self.edge_var_idx = np.asarray(edge_var, dtype=np.int64)
        self.edge_pos = np.asarray(edge_pos, dtype=np.int64)
        self.edge_sign = np.asarray(edge_sign, dtype=np.float32)

    def _open_decision_var(self, state) -> Optional[int]:
        if not state.decision_stack:
            return None
        top = state.decision_stack[-1]
        v = int(top.decision_var)
        if int(state.assignment[v]) == 0:
            return v
        return None

    def _lit_value(self, assignment: np.ndarray, lit: int) -> int:
        """Return 1 if lit true, -1 if false, 0 if unassigned."""

        v = abs(int(lit)) - 1
        sign = 1 if int(lit) > 0 else -1
        a = int(assignment[int(v)])
        if a == 0:
            return 0
        return 1 if int(a) == int(sign) else -1

    def _state_to_obs(self) -> UnifiedObservation:
        state = self.env.get_state()
        n = int(state.num_vars)

        open_var = self._open_decision_var(state)
        top_failed_mask = (
            int(state.decision_stack[-1].failed_mask) if open_var is not None else 0
        )

        # Variable domain mask [n, max_domain]
        var_domain_mask = np.zeros((n, self.max_domain), dtype=bool)
        for i in range(n):
            a = int(state.assignment[i])
            if a == 0:
                allow_false = True
                allow_true = True
                if open_var is not None and int(open_var) == int(i):
                    allow_false = (top_failed_mask & 1) == 0
                    allow_true = (top_failed_mask & 2) == 0
            else:
                allow_false = a == -1
                allow_true = a == 1

            var_domain_mask[i, 0] = bool(allow_false)
            var_domain_mask[i, 1] = bool(allow_true)

        # Variable assignment: -1=unassigned, 0=False, 1=True
        var_assigned = np.full((n,), -1, dtype=np.int64)
        for i in range(n):
            a = int(state.assignment[i])
            if a == -1:
                var_assigned[i] = 0
            elif a == 1:
                var_assigned[i] = 1

        # Branch-local nogoods from open decision frame.
        var_nogood_mask = np.zeros((n, self.max_domain), dtype=bool)
        if open_var is not None:
            if (top_failed_mask & 1) != 0:
                var_nogood_mask[int(open_var), 0] = True
            if (top_failed_mask & 2) != 0:
                var_nogood_mask[int(open_var), 1] = True

        # Variable features
        selected_var = (
            int(state.selected_var) if state.selected_var is not None else None
        )
        var_features = self._build_var_features(
            var_assigned=var_assigned,
            var_domain_mask=var_domain_mask,
            selected_var=selected_var,
            num_vars=n,
        )

        # Constraint tensors
        con_type = np.full(
            (self.num_constraints,), int(ConstraintType.CLAUSE), dtype=np.int64
        )

        con_scope_padded = np.full((self.num_constraints, 4), -1, dtype=np.int64)
        for c_idx, clause in enumerate(self.clauses):
            for pos, lit in enumerate(clause):
                con_scope_padded[int(c_idx), int(pos)] = abs(int(lit)) - 1

        # Constraint features
        con_features = np.zeros((self.num_constraints, 2), dtype=np.float32)
        con_features[:, 0] = np.float32(3.0 / 4.0)

        for c_idx, clause in enumerate(self.clauses):
            vals = [self._lit_value(state.assignment, int(l)) for l in clause]
            if any(int(v) == 1 for v in vals):
                con_features[int(c_idx), 1] = np.float32(1.0)
            elif all(int(v) != 0 for v in vals):
                con_features[int(c_idx), 1] = np.float32(0.0)
            else:
                con_features[int(c_idx), 1] = np.float32(0.5)

        # Edge features: [position_norm, sign]
        edge_sign = (self.edge_sign + np.float32(1.0)) / np.float32(2.0)
        edge_features = self._build_edge_features(self.edge_pos, edge_sign=edge_sign)

        return UnifiedObservation(
            var_domain_mask=var_domain_mask,
            var_nogood_mask=var_nogood_mask,
            var_assigned=var_assigned,
            var_features=var_features,
            con_type=con_type,
            con_scope=con_scope_padded,
            con_features=con_features,
            edge_con_idx=self.edge_con_idx,
            edge_var_idx=self.edge_var_idx,
            edge_features=edge_features,
            num_vars=int(n),
            num_constraints=int(self.num_constraints),
            max_domain=int(self.max_domain),
            stack_depth=int(len(state.decision_stack)),
            propagation_pending=bool(state.propagation_pending),
            has_conflict=bool(state.conflict_clause is not None),
            propagation_mode=int(self.propagation_mode_id),
            domain_id=2,
        )

    def step(
        self, action: UnifiedAction
    ) -> Tuple[UnifiedObservation, float, bool, dict]:
        from sat.dsl import SatAction

        state = self.env.get_state()

        if action.type == UnifiedActionType.ASSIGN:
            if action.var is None or action.value is None:
                raise ValueError("ASSIGN requires var and value")

            var = int(action.var)
            val = int(action.value)

            if state.selected_var is not None and int(state.selected_var) != int(var):
                return (
                    self._state_to_obs(),
                    0.0,
                    False,
                    {"valid": False, "reason": "var already selected"},
                )

            if state.selected_var is None:
                sel_res = self.env.step(SatAction.select_var(int(var)))
                if (not bool(sel_res.info.get("valid", True))) or bool(sel_res.done):
                    return (
                        self._state_to_obs(),
                        float(sel_res.reward),
                        bool(sel_res.done),
                        dict(sel_res.info),
                    )

            res = self.env.step(SatAction.assign_value(int(val)))
            if (not bool(res.info.get("valid", True))) or bool(res.done):
                return (
                    self._state_to_obs(),
                    float(res.reward),
                    bool(res.done),
                    dict(res.info),
                )

            new_state = self.env.get_state()
            if bool(new_state.propagation_pending):
                res = self.env.step(SatAction.propagate())

        elif action.type == UnifiedActionType.BACKTRACK:
            res = self.env.step(SatAction.backtrack())

        elif action.type == UnifiedActionType.DONE:
            res = self.env.step(SatAction.done())

        else:
            raise ValueError(f"Unknown action type: {action.type}")

        return self._state_to_obs(), float(res.reward), bool(res.done), dict(res.info)

    def get_valid_actions(self) -> list[UnifiedAction]:
        state = self.env.get_state()

        # Root-level UNSAT sentinel: only DONE.
        if state.conflict_clause is not None and not state.decision_stack:
            if state.selected_var is None and (not bool(state.propagation_pending)):
                return [UnifiedAction.done()]
            return []

        if bool(state.propagation_pending):
            actions: list[UnifiedAction] = []
            if state.decision_stack:
                actions.append(UnifiedAction.backtrack())
            return actions

        actions: list[UnifiedAction] = []

        open_var = self._open_decision_var(state)
        top_failed_mask = (
            int(state.decision_stack[-1].failed_mask) if open_var is not None else 0
        )

        # Candidate vars: if already selected, only it; else if open decision var exists, only it; else any unassigned.
        candidate_vars: list[int]
        if state.selected_var is not None:
            candidate_vars = [int(state.selected_var)]
        elif open_var is not None:
            candidate_vars = [int(open_var)]
        else:
            candidate_vars = [
                i for i in range(int(state.num_vars)) if int(state.assignment[i]) == 0
            ]

        # Assign actions (only if not in conflict).
        if state.conflict_clause is None:
            for v in candidate_vars:
                if int(state.assignment[int(v)]) != 0:
                    continue

                allow_false = True
                allow_true = True
                if open_var is not None and int(open_var) == int(v):
                    allow_false = (top_failed_mask & 1) == 0
                    allow_true = (top_failed_mask & 2) == 0

                if allow_false:
                    actions.append(UnifiedAction.assign(int(v), 0))
                if allow_true:
                    actions.append(UnifiedAction.assign(int(v), 1))

        if state.decision_stack:
            actions.append(UnifiedAction.backtrack())

        # Only allow DONE when all variables are assigned (prevents premature DONE)
        all_assigned = all(
            int(state.assignment[i]) != 0 for i in range(int(state.num_vars))
        )
        if state.selected_var is None and all_assigned:
            actions.append(UnifiedAction.done())

        return actions

    @property
    def domain_id(self) -> int:
        return 2


class HybridUnifiedWrapper(UnifiedEnvWrapper):
    """Wrapper for Hybrid CSP environment."""

    def __init__(self, env: HybridEnv, max_domain: int = 10):
        if not _HAS_HYBRID:
            raise ImportError(
                "hybrid module not available; install hybrid domain to use HybridUnifiedWrapper"
            )
        self.env = env
        self.max_domain = int(max_domain)
        self.n_color = int(env.n_color)
        self.n_bool = int(env.n_bool)
        self.n_total = int(self.n_color + self.n_bool)
        self.k = int(env.k)

        min_domain = max(2, int(self.k))
        if int(self.max_domain) < int(min_domain):
            raise ValueError("max_domain must be >= max(2, k)")

        self._build_constraint_graph()
        logger.debug(
            "HybridUnifiedWrapper init n_color=%s n_bool=%s k=%s max_domain=%s "
            "num_constraints=%s num_neq=%s num_clauses=%s",
            self.n_color,
            self.n_bool,
            self.k,
            self.max_domain,
            self.num_constraints,
            self.num_neq,
            len(self.clauses),
        )

    def _build_constraint_graph(self) -> None:
        self.neq_constraints: list[tuple[int, int]] = []
        for i, j in self.env.color_edges:
            self.neq_constraints.append((int(i), int(j)))
        for b_idx, c_idx in self.env.bridge_edges:
            self.neq_constraints.append((int(self.n_color + int(b_idx)), int(c_idx)))

        self.clauses = list(self.env.clauses)
        self.num_constraints = int(len(self.neq_constraints) + len(self.clauses))
        self.num_neq = int(len(self.neq_constraints))

        self._con_type = np.zeros((self.num_constraints,), dtype=np.int64)
        self._con_scope = np.full((self.num_constraints, 4), -1, dtype=np.int64)

        for idx, (v1, v2) in enumerate(self.neq_constraints):
            self._con_type[int(idx)] = int(ConstraintType.NEQ)
            self._con_scope[int(idx), 0] = int(v1)
            self._con_scope[int(idx), 1] = int(v2)

        offset = int(self.num_neq)
        for cid, clause in enumerate(self.clauses):
            con_id = int(offset + cid)
            self._con_type[int(con_id)] = int(ConstraintType.CLAUSE)
            for pos, lit in enumerate(clause):
                var = int(self.n_color + (abs(int(lit)) - 1))
                self._con_scope[int(con_id), int(pos)] = int(var)

        edge_con: list[int] = []
        edge_var: list[int] = []
        edge_pos: list[int] = []
        edge_sign: list[float] = []
        edge_is_clause: list[bool] = []

        for c_idx, (v1, v2) in enumerate(self.neq_constraints):
            edge_con.extend([int(c_idx), int(c_idx)])
            edge_var.extend([int(v1), int(v2)])
            edge_pos.extend([0, 1])
            edge_sign.extend([0.0, 0.0])
            edge_is_clause.extend([False, False])

        for cid, clause in enumerate(self.clauses):
            con_id = int(offset + cid)
            for pos, lit in enumerate(clause):
                var = int(self.n_color + (abs(int(lit)) - 1))
                sign = 1.0 if int(lit) > 0 else -1.0
                edge_con.append(int(con_id))
                edge_var.append(int(var))
                edge_pos.append(int(pos))
                edge_sign.append(float(sign))
                edge_is_clause.append(True)

        self.edge_con_idx = np.asarray(edge_con, dtype=np.int64)
        self.edge_var_idx = np.asarray(edge_var, dtype=np.int64)
        self.edge_pos = np.asarray(edge_pos, dtype=np.int64)
        self.edge_sign = np.asarray(edge_sign, dtype=np.float32)
        self.edge_is_clause = np.asarray(edge_is_clause, dtype=bool)

    def _current_depth(self, state) -> int:
        return int(len(state.assignment_stack) + 1)

    def _effective_domain(self, state, var: int, *, depth: int) -> set[int]:
        v = int(var)
        dom = set(int(x) for x in state.domains[int(v)])
        if int(state.assignment[int(v)]) != 0:
            return dom
        banned = state.nogoods.get(int(depth), {}).get(int(v))
        if banned:
            dom.difference_update(int(x) for x in banned)
        return dom

    def _neq_value(self, var_idx: int, assignment_val: int) -> int:
        if int(var_idx) < int(self.n_color):
            return int(assignment_val)
        if int(assignment_val) == -1:
            return 1
        if int(assignment_val) == 1:
            return 2
        return 0

    def _lit_value(self, assignment: np.ndarray, lit: int) -> int:
        var = abs(int(lit)) - 1
        gvar = int(self.n_color + int(var))
        a = int(assignment[int(gvar)])
        if int(a) == 0:
            return 0
        sign = 1 if int(lit) > 0 else -1
        return 1 if int(a) == int(sign) else -1

    def _state_to_obs(self) -> UnifiedObservation:
        state = self.env.get_state()
        n = int(self.n_total)
        depth = self._current_depth(state)

        var_assigned = np.full((n,), -1, dtype=np.int64)
        for i in range(n):
            val = int(state.assignment[int(i)])
            if int(val) == 0:
                continue
            if int(i) < int(self.n_color):
                var_assigned[int(i)] = int(val - 1)
            else:
                var_assigned[int(i)] = 0 if int(val) == -1 else 1

        var_nogood_mask = np.zeros((n, self.max_domain), dtype=bool)
        per_depth = state.nogoods.get(int(depth), {})
        for var, failed_vals in per_depth.items():
            v = int(var)
            if int(v) < int(self.n_color):
                for c in failed_vals:
                    cc = int(c)
                    if 1 <= cc <= self.max_domain:
                        var_nogood_mask[int(v), int(cc - 1)] = True
            else:
                for b in failed_vals:
                    bb = int(b)
                    if int(bb) == -1:
                        var_nogood_mask[int(v), 0] = True
                    elif int(bb) == 1:
                        var_nogood_mask[int(v), 1] = True

        var_domain_mask = np.zeros((n, self.max_domain), dtype=bool)
        for i in range(n):
            dom = self._effective_domain(state, int(i), depth=int(depth))
            if int(i) < int(self.n_color):
                for c in dom:
                    cc = int(c)
                    if 1 <= cc <= self.max_domain:
                        var_domain_mask[int(i), int(cc - 1)] = True
            else:
                for b in dom:
                    bb = int(b)
                    if int(bb) == -1:
                        var_domain_mask[int(i), 0] = True
                    elif int(bb) == 1:
                        var_domain_mask[int(i), 1] = True

        selected_var = (
            int(state.selected_var) if state.selected_var is not None else None
        )
        var_features = self._build_var_features(
            var_assigned=var_assigned,
            var_domain_mask=var_domain_mask,
            selected_var=selected_var,
            num_vars=n,
        )

        con_features = np.zeros((self.num_constraints, 2), dtype=np.float32)
        if int(self.num_constraints) > 0:
            con_features[: int(self.num_neq), 0] = np.float32(2.0 / 4.0)
            con_features[int(self.num_neq) :, 0] = np.float32(3.0 / 4.0)

        for c_idx, (v1, v2) in enumerate(self.neq_constraints):
            a1 = int(state.assignment[int(v1)])
            a2 = int(state.assignment[int(v2)])
            if int(a1) == 0 or int(a2) == 0:
                con_features[int(c_idx), 1] = np.float32(0.5)
            else:
                v1_val = self._neq_value(int(v1), int(a1))
                v2_val = self._neq_value(int(v2), int(a2))
                con_features[int(c_idx), 1] = np.float32(
                    float(int(v1_val) != int(v2_val))
                )

        offset = int(self.num_neq)
        for cid, clause in enumerate(self.clauses):
            con_idx = int(offset + cid)
            vals = [self._lit_value(state.assignment, int(l)) for l in clause]
            if any(int(v) == 1 for v in vals):
                con_features[int(con_idx), 1] = np.float32(1.0)
            elif all(int(v) == -1 for v in vals):
                con_features[int(con_idx), 1] = np.float32(0.0)
            else:
                con_features[int(con_idx), 1] = np.float32(0.5)

        edge_sign = np.zeros((int(self.edge_con_idx.shape[0]),), dtype=np.float32)
        if np.any(self.edge_is_clause):
            idx = np.nonzero(self.edge_is_clause)[0]
            edge_sign[idx] = (self.edge_sign[idx] + np.float32(1.0)) / np.float32(2.0)
        edge_features = self._build_edge_features(self.edge_pos, edge_sign=edge_sign)

        return UnifiedObservation(
            var_domain_mask=var_domain_mask,
            var_nogood_mask=var_nogood_mask,
            var_assigned=var_assigned,
            var_features=var_features,
            con_type=np.array(self._con_type, copy=True),
            con_scope=np.array(self._con_scope, copy=True),
            con_features=con_features,
            edge_con_idx=self.edge_con_idx,
            edge_var_idx=self.edge_var_idx,
            edge_features=edge_features,
            num_vars=int(n),
            num_constraints=int(self.num_constraints),
            max_domain=int(self.max_domain),
            stack_depth=int(len(state.assignment_stack)),
            propagation_pending=bool(state.propagation_pending),
            has_conflict=bool(self.env._has_contradiction(state)),
            propagation_mode=1,
            domain_id=3,
        )

    def step(
        self, action: UnifiedAction
    ) -> Tuple[UnifiedObservation, float, bool, dict]:
        state = self.env.get_state()

        if action.type == UnifiedActionType.ASSIGN:
            if action.var is None or action.value is None:
                raise ValueError("ASSIGN requires var and value")

            var = int(action.var)
            val = int(action.value)

            if state.selected_var is not None and int(state.selected_var) != int(var):
                return (
                    self._state_to_obs(),
                    0.0,
                    False,
                    {"valid": False, "reason": "var already selected"},
                )

            if state.selected_var is None:
                sel_res = self.env.step(HybridAction.select_var(int(var)))
                if (not bool(sel_res.info.get("valid", True))) or bool(sel_res.done):
                    return (
                        self._state_to_obs(),
                        float(sel_res.reward),
                        bool(sel_res.done),
                        dict(sel_res.info),
                    )

            if int(var) < int(self.n_color):
                env_val = int(val + 1)
            else:
                if int(val) not in {0, 1}:
                    raise ValueError("Boolean assignment must be 0/1")
                env_val = int(val)

            res = self.env.step(HybridAction.assign_value(int(env_val)))
            if (not bool(res.info.get("valid", True))) or bool(res.done):
                return (
                    self._state_to_obs(),
                    float(res.reward),
                    bool(res.done),
                    dict(res.info),
                )

            new_state = self.env.get_state()
            if bool(new_state.propagation_pending):
                res = self.env.step(HybridAction.propagate())

        elif action.type == UnifiedActionType.BACKTRACK:
            res = self.env.step(HybridAction.backtrack())

        elif action.type == UnifiedActionType.DONE:
            res = self.env.step(HybridAction.done())

        else:
            raise ValueError(f"Unknown action type: {action.type}")

        return self._state_to_obs(), float(res.reward), bool(res.done), dict(res.info)

    def get_valid_actions(self) -> list[UnifiedAction]:
        state = self.env.get_state()

        if bool(state.propagation_pending):
            actions: list[UnifiedAction] = []
            if state.assignment_stack:
                actions.append(UnifiedAction.backtrack())
            return actions

        if self.env._has_contradiction(state):
            if state.assignment_stack:
                return [UnifiedAction.backtrack()]
            return [UnifiedAction.done()]

        depth = self._current_depth(state)
        per_depth = state.nogoods.get(int(depth), {})

        actions: list[UnifiedAction] = []

        if state.selected_var is None:
            candidate_vars = [i for i in range(int(self.n_total))]
        else:
            candidate_vars = [int(state.selected_var)]

        for var_idx in candidate_vars:
            if int(state.assignment[int(var_idx)]) != 0:
                continue
            banned = per_depth.get(int(var_idx), set())

            for v in sorted(int(x) for x in state.domains[int(var_idx)]):
                if int(v) in banned:
                    continue
                if int(var_idx) < int(self.n_color):
                    actions.append(UnifiedAction.assign(int(var_idx), int(v - 1)))
                else:
                    if int(v) == -1:
                        actions.append(UnifiedAction.assign(int(var_idx), 0))
                    elif int(v) == 1:
                        actions.append(UnifiedAction.assign(int(var_idx), 1))

        if state.assignment_stack:
            actions.append(UnifiedAction.backtrack())

        if state.selected_var is None:
            actions.append(UnifiedAction.done())

        return actions

    @property
    def domain_id(self) -> int:
        return 3
