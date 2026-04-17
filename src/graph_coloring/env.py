from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .dsl import GraphColorAction, GraphColorActionType


class GraphColorEnvStatus(Enum):
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class GraphColorState:
    num_nodes: int
    num_colors: int  # k
    adjacency: np.ndarray  # (n, n) bool adjacency matrix
    assignment: np.ndarray  # (n,) int, 0 = unassigned, 1..k = colors
    domains: List[Set[int]]  # domains[i] = valid colors for node i
    selected_node: Optional[int]
    assignment_stack: List[
        Tuple[int, int, List[Set[int]]]
    ]  # (node, color, domain_snapshot)
    nogoods: Dict[int, Dict[int, Set[int]]]  # depth -> node -> refuted colors
    propagation_pending: bool
    step_count: int
    status: GraphColorEnvStatus


@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


def _copy_domains(domains: List[Set[int]]) -> List[Set[int]]:
    return [set(d) for d in domains]


def _copy_stack(
    stack: List[Tuple[int, int, List[Set[int]]]],
) -> List[Tuple[int, int, List[Set[int]]]]:
    return [
        (int(node), int(color), _copy_domains(snapshot))
        for (node, color, snapshot) in stack
    ]


def _copy_nogoods(
    nogoods: Dict[int, Dict[int, Set[int]]],
) -> Dict[int, Dict[int, Set[int]]]:
    return {
        int(depth): {
            int(node): set(int(c) for c in colors) for node, colors in per_node.items()
        }
        for depth, per_node in nogoods.items()
    }


def _num_assigned(assignment: np.ndarray) -> int:
    return int(np.count_nonzero(assignment))


def _num_empty_domains(state: GraphColorState) -> int:
    n = 0
    for i, dom in enumerate(state.domains):
        if int(state.assignment[i]) != 0:
            continue
        if len(dom) == 0:
            n += 1
    return int(n)


class GraphColorEnv:
    """Graph k-coloring CSP environment with explicit propagation and nogoods.

    Workflow:
      1) SELECT_NODE(i)
      2) ASSIGN_COLOR(c)
      3) PROPAGATE
      4) (repeat or BACKTRACK)
      5) DONE (declares SUCCESS if valid complete coloring, else FAILURE)
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        num_colors: int,
        solution: Optional[np.ndarray] = None,  # For verification
        mode: str = "soft",  # "soft" or "strict"
        max_steps: int = 500,
        propagation_mode: str = "forward_check",  # "none" or "forward_check"
    ):
        if mode not in {"strict", "soft"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be >= 1")
        if int(num_colors) < 1:
            raise ValueError("num_colors must be >= 1")
        if propagation_mode not in {"none", "forward_check"}:
            raise ValueError(
                f"propagation_mode must be 'none' or 'forward_check'; got {propagation_mode!r}"
            )

        adj = np.array(adjacency, copy=True)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(
                f"adjacency must be square (n,n); got shape={tuple(adj.shape)}"
            )
        if adj.shape[0] < 1:
            raise ValueError("adjacency must have at least 1 node")

        adj_bool = adj.astype(bool, copy=False)
        if not np.array_equal(adj_bool, adj_bool.T):
            raise ValueError("adjacency must be symmetric")
        if np.any(np.diag(adj_bool)):
            raise ValueError(
                "adjacency must have no self-loops (diagonal must be False)"
            )

        self.adjacency = np.array(adj_bool, dtype=bool, copy=True)
        self.num_nodes = int(self.adjacency.shape[0])
        self.num_colors = int(num_colors)

        self.solution: Optional[np.ndarray]
        if solution is not None:
            sol = np.array(solution, dtype=np.int64, copy=True)
            if sol.shape != (self.num_nodes,):
                raise ValueError(
                    f"solution must have shape ({self.num_nodes},); got {tuple(sol.shape)}"
                )
            if np.any(sol < 1) or np.any(sol > self.num_colors):
                raise ValueError("solution colors must be in [1, num_colors]")
            if not self._is_valid_coloring(sol):
                raise ValueError("Provided solution is not a valid coloring")
            self.solution = sol
        else:
            self.solution = None

        self.mode = mode
        self.max_steps = int(max_steps)
        self.propagation_mode = str(propagation_mode)

        # Rewards (kept consistent with other envs).
        self.goal_reward = 1.0
        self.step_penalty = -0.01
        self.invalid_penalty = -1.0

        # Constant observation payload for graph structure.
        self._adjacency_obs = self.adjacency.astype(np.int8).tolist()
        self._degrees = self.adjacency.sum(axis=1).astype(np.int64)

        self._state: Optional[GraphColorState] = None

    def reset(self) -> dict:
        domains: List[Set[int]] = [
            set(range(1, self.num_colors + 1)) for _ in range(self.num_nodes)
        ]

        self._state = GraphColorState(
            num_nodes=int(self.num_nodes),
            num_colors=int(self.num_colors),
            adjacency=np.array(self.adjacency, copy=True),
            assignment=np.zeros((self.num_nodes,), dtype=np.int64),
            domains=domains,
            selected_node=None,
            assignment_stack=[],
            nogoods={},
            propagation_pending=False,
            step_count=0,
            status=GraphColorEnvStatus.RUNNING,
        )

        return self.get_observation(self._state)

    def _require_running(self) -> GraphColorState:
        if self._state is None:
            raise RuntimeError("Environment not reset")
        if self._state.status != GraphColorEnvStatus.RUNNING:
            raise RuntimeError(f"Environment already terminated: {self._state.status}")
        return self._state

    def get_state(self) -> GraphColorState:
        if self._state is None:
            raise RuntimeError("Environment not reset")
        s = self._state
        return GraphColorState(
            num_nodes=int(s.num_nodes),
            num_colors=int(s.num_colors),
            adjacency=np.array(s.adjacency, copy=True),
            assignment=np.array(s.assignment, copy=True),
            domains=_copy_domains(s.domains),
            selected_node=None if s.selected_node is None else int(s.selected_node),
            assignment_stack=_copy_stack(s.assignment_stack),
            nogoods=_copy_nogoods(s.nogoods),
            propagation_pending=bool(s.propagation_pending),
            step_count=int(s.step_count),
            status=s.status,
        )

    def _is_valid_coloring(self, assignment: np.ndarray) -> bool:
        if assignment.shape != (self.num_nodes,):
            return False
        if np.any(assignment < 0) or np.any(assignment > self.num_colors):
            return False

        # Check adjacent nodes do not share a (non-zero) color.
        a = assignment.reshape(self.num_nodes, 1)
        same = (a == a.T) & (a != 0)
        conflict = bool(np.any(self.adjacency & same))
        return not conflict

    def _is_complete_solution(self, state: GraphColorState) -> bool:
        if int(np.count_nonzero(state.assignment == 0)) != 0:
            return False
        if np.any(state.assignment < 1) or np.any(state.assignment > state.num_colors):
            return False
        return self._is_valid_coloring(state.assignment)

    def _is_valid(
        self, action: GraphColorAction, state: GraphColorState
    ) -> tuple[bool, str]:
        if action.type == GraphColorActionType.SELECT_NODE:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if self._has_contradiction(state):
                return False, "contradiction: must BACKTRACK"
            if state.selected_node is not None:
                return False, "node already selected"
            if action.target is None:
                return False, "SELECT_NODE missing target"
            node = int(action.target)
            if node < 0 or node >= state.num_nodes:
                return False, "node idx out of range"
            if int(state.assignment[node]) != 0:
                return False, "node already assigned"
            if len(self._effective_domain(state, node)) == 0:
                return False, "node domain empty"
            return True, ""

        if action.type == GraphColorActionType.ASSIGN_COLOR:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if self._has_contradiction(state):
                return False, "contradiction: must BACKTRACK"
            if state.selected_node is None:
                return False, "no node selected"
            if action.target is None:
                return False, "ASSIGN_COLOR missing color"
            color = int(action.target)
            if color < 1 or color > state.num_colors:
                return False, "color out of range"
            node = int(state.selected_node)
            if int(state.assignment[node]) != 0:
                return False, "node already assigned"
            if color not in self._effective_domain(state, node):
                return False, "color not in domain"
            return True, ""

        if action.type == GraphColorActionType.PROPAGATE:
            if not state.propagation_pending:
                return False, "no pending assignment"
            return True, ""

        if action.type == GraphColorActionType.BACKTRACK:
            if not state.assignment_stack:
                return False, "stack empty"
            return True, ""

        if action.type == GraphColorActionType.DONE:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if state.selected_node is not None:
                return False, "node still selected"
            # DONE is always allowed to terminate (success if solved; otherwise failure).
            return True, ""

        return False, f"Unknown action type: {action.type}"

    def step(self, action: GraphColorAction) -> StepResult:
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
                state.status = GraphColorEnvStatus.FAILURE
        else:
            if action.type == GraphColorActionType.SELECT_NODE:
                state.selected_node = int(action.target)  # type: ignore[arg-type]

            elif action.type == GraphColorActionType.ASSIGN_COLOR:
                node = int(state.selected_node)  # type: ignore[arg-type]
                color = int(action.target)  # type: ignore[arg-type]

                snapshot = _copy_domains(state.domains)
                state.assignment_stack.append((int(node), int(color), snapshot))

                state.assignment[node] = int(color)

                state.selected_node = None
                state.propagation_pending = True

            elif action.type == GraphColorActionType.PROPAGATE:
                changed = self._propagate(state)
                state.propagation_pending = False

                info["domains_changed"] = bool(changed)

                if self._has_contradiction(state) and not state.assignment_stack:
                    state.status = GraphColorEnvStatus.FAILURE

            elif action.type == GraphColorActionType.BACKTRACK:
                depth = int(len(state.assignment_stack))
                node, color, _snapshot = state.assignment_stack[-1]

                per_depth = state.nogoods.setdefault(depth, {})
                per_node = per_depth.setdefault(int(node), set())
                per_node.add(int(color))

                node, _color, snapshot = state.assignment_stack.pop()
                state.domains = _copy_domains(snapshot)
                state.assignment[int(node)] = 0

                # Clear nogoods for deeper decision levels (branch-local).
                for d in list(state.nogoods.keys()):
                    if int(d) > depth:
                        del state.nogoods[d]

                state.selected_node = None
                state.propagation_pending = False

            elif action.type == GraphColorActionType.DONE:
                if self._is_complete_solution(state):
                    state.status = GraphColorEnvStatus.SUCCESS
                    reward = float(self.goal_reward)
                else:
                    state.status = GraphColorEnvStatus.FAILURE
                    reward = 0.0

        if state.status == GraphColorEnvStatus.RUNNING and int(state.step_count) >= int(
            self.max_steps
        ):
            state.status = GraphColorEnvStatus.FAILURE
            info["reason"] = "step_limit"

        info.update(
            {
                "status": state.status.name,
                "selected_node": -1
                if state.selected_node is None
                else int(state.selected_node),
                "propagation_pending": bool(state.propagation_pending),
                "num_assigned": _num_assigned(state.assignment),
                "num_empty_domains": _num_empty_domains(state),
                "stack_depth": int(len(state.assignment_stack)),
                "solved": int(self._is_complete_solution(state)),
            }
        )

        obs = self.get_observation(state)
        done = state.status != GraphColorEnvStatus.RUNNING
        return StepResult(
            observation=obs, reward=float(reward), done=bool(done), info=info
        )

    def backjump_to(self, target_depth: int) -> dict:
        """Backjump: unwind the stack to target_depth (0-indexed).

        Pops entries from assignment_stack until len(stack) == target_depth.
        For the target entry:
          - Record nogood (current color for node at current depth)
          - Restore domains from the snapshot at target_depth
          - Clear assignment for the node
          - Clear nogoods for deeper levels

        After all pops, restore domains from the snapshot saved at position target_depth.

        Returns info dict with: num_popped, final_stack_depth
        """
        state = self._require_running()
        stack_size = int(len(state.assignment_stack))
        target_depth = int(target_depth)

        if target_depth < 0 or target_depth >= stack_size:
            raise ValueError(
                f"target_depth {target_depth} out of range [0, {stack_size})"
            )

        target_node, target_color, target_snapshot = state.assignment_stack[target_depth]
        target_depth_key = int(target_depth + 1)

        per_depth = state.nogoods.setdefault(target_depth_key, {})
        per_node = per_depth.setdefault(int(target_node), set())
        per_node.add(int(target_color))

        for i in range(stack_size - 1, target_depth - 1, -1):
            node, _color, _snapshot = state.assignment_stack[i]
            state.assignment[int(node)] = 0

        state.domains = _copy_domains(target_snapshot)
        state.assignment_stack = state.assignment_stack[:target_depth]

        for d in list(state.nogoods.keys()):
            if int(d) > int(target_depth_key):
                del state.nogoods[int(d)]

        state.selected_node = None
        state.propagation_pending = False

        state.step_count += 1
        if state.status == GraphColorEnvStatus.RUNNING and int(state.step_count) >= int(
            self.max_steps
        ):
            state.status = GraphColorEnvStatus.FAILURE

        return {
            "num_popped": int(stack_size - target_depth),
            "final_stack_depth": int(len(state.assignment_stack)),
            "target_node": int(target_node),
            "target_color": int(target_color),
        }

    def get_valid_actions(self) -> List[GraphColorAction]:
        if self._state is None:
            raise RuntimeError("Environment not reset")
        if self._state.status != GraphColorEnvStatus.RUNNING:
            return []

        state = self._state
        actions: List[GraphColorAction] = []

        # SELECT_NODE candidates.
        for node in range(state.num_nodes):
            a = GraphColorAction.select_node(int(node))
            ok, _ = self._is_valid(a, state)
            if ok:
                actions.append(a)

        # ASSIGN_COLOR candidates.
        if state.selected_node is not None:
            node = int(state.selected_node)
            for c in sorted(self._effective_domain(state, node)):
                a = GraphColorAction.assign_color(int(c))
                ok, _ = self._is_valid(a, state)
                if ok:
                    actions.append(a)

        # Operator actions.
        for a in [
            GraphColorAction.propagate(),
            GraphColorAction.backtrack(),
            GraphColorAction.done(),
        ]:
            ok, _ = self._is_valid(a, state)
            if ok:
                actions.append(a)

        return actions

    def _propagate(self, state: GraphColorState) -> bool:
        """Forward-checking propagation.

        For each assigned node, remove its color from neighbors' domains.
        Returns True if any domain changed.
        """

        changed = False

        # Force assigned nodes to singleton domains.
        for i in range(state.num_nodes):
            c = int(state.assignment[i])
            if c == 0:
                continue
            if state.domains[i] != {c}:
                state.domains[i] = {c}
                changed = True

        # Skip neighbor propagation if mode is "none"
        if self.propagation_mode == "none":
            return bool(changed)

        # Remove assigned colors from unassigned neighbors.
        for i in range(state.num_nodes):
            c = int(state.assignment[i])
            if c == 0:
                continue

            neigh = np.nonzero(state.adjacency[i])[0]
            for j in neigh:
                if int(state.assignment[int(j)]) != 0:
                    continue
                if c in state.domains[int(j)]:
                    state.domains[int(j)].discard(c)
                    changed = True

        return bool(changed)

    def _has_contradiction(self, state: GraphColorState) -> bool:
        # Empty domain contradiction.
        if _num_empty_domains(state) > 0:
            return True

        # Direct coloring conflict contradiction (should not happen if domains are respected).
        for i in range(state.num_nodes):
            c = int(state.assignment[i])
            if c == 0:
                continue
            if bool(np.any(state.adjacency[i] & (state.assignment == c))):
                return True

        return False

    def _current_depth(self, state: GraphColorState) -> int:
        return int(len(state.assignment_stack) + 1)

    def _effective_domain(
        self,
        state: GraphColorState,
        node: int,
        depth: Optional[int] = None,
    ) -> Set[int]:
        """Domain for `node` excluding per-depth nogoods."""

        node_i = int(node)
        if node_i < 0 or node_i >= state.num_nodes:
            raise ValueError(f"node out of range: {node_i}")

        if depth is None:
            depth = self._current_depth(state)

        dom = set(state.domains[node_i])
        if int(state.assignment[node_i]) != 0:
            return dom

        banned = state.nogoods.get(int(depth), {}).get(int(node_i))
        if banned:
            dom.difference_update(banned)
        return dom

    def get_observation(self, state: GraphColorState) -> dict:
        selected = -1 if state.selected_node is None else int(state.selected_node)

        nodes: List[List[int]] = []
        for i in range(state.num_nodes):
            v = int(state.assignment[i])
            is_sel = int(i == selected)
            deg = int(self._degrees[i])
            nodes.append([int(i), int(deg), int(v), int(is_sel)])

        depth = self._current_depth(state)
        domains_repr: List[List[int]] = []
        for i in range(state.num_nodes):
            if int(state.assignment[i]) == 0:
                dom = self._effective_domain(state, i, depth=depth)
            else:
                dom = set(state.domains[i])
            domains_repr.append(sorted(int(x) for x in dom))

        obs = {
            "meta": [int(state.num_nodes), int(state.num_colors)],
            "global": [
                int(selected),
                int(_num_assigned(state.assignment)),
                int(_num_empty_domains(state)),
            ],
            "nodes": nodes,
            "domains": domains_repr,
            "adjacency": self._adjacency_obs,
            "propagation_pending": bool(state.propagation_pending),
        }
        return obs


def compute_conflict_witness(state: GraphColorState) -> dict:
    """Compute conflict witness when a contradiction exists.

    For each unassigned node u with empty domain, identify which assigned
    neighbors block each color. The witness set is the set of blocking nodes.
    The backjump target is the shallowest (earliest-pushed) witness in the stack.
    The deepest witness is also reported for CBJ-correct targeting.

    Returns dict with:
        - contradiction_nodes: list of nodes with empty domains
        - witnesses: dict mapping contradiction_node -> list of (blocker_node, blocked_color)
        - witness_nodes: set of all blocker nodes (union across all contradiction nodes)
        - witness_stack_indices: list of stack indices for witness nodes
        - backjump_target: stack index of shallowest witness (0-indexed), or -1 if no witness found
        - backjump_depth: number of stack entries to pop (depth - backjump_target)
        - deepest_witness_target: stack index of deepest witness (0-indexed), or -1 if no witness found
        - deepest_witness_depth: number of stack entries to pop (depth - deepest_witness_target)
    """

    contradiction_nodes: List[int] = []
    witnesses: Dict[int, List[Tuple[int, int]]] = {}
    witness_nodes: Set[int] = set()

    assignment = state.assignment
    num_colors = int(state.num_colors)
    adjacency = state.adjacency

    for u in range(state.num_nodes):
        if int(assignment[u]) != 0:
            continue
        if len(state.domains[u]) != 0:
            continue
        contradiction_nodes.append(int(u))
        per_node_witnesses: List[Tuple[int, int]] = []
        neighbors = np.nonzero(adjacency[u])[0]
        for color in range(1, num_colors + 1):
            blockers = [
                int(v)
                for v in neighbors
                if int(assignment[int(v)]) == int(color)
            ]
            for blocker in blockers:
                per_node_witnesses.append((int(blocker), int(color)))
                witness_nodes.add(int(blocker))
        witnesses[int(u)] = per_node_witnesses

    stack_index: Dict[int, int] = {}
    for idx, (node, _color, _snapshot) in enumerate(state.assignment_stack):
        node_id = int(node)
        if node_id not in stack_index:
            stack_index[node_id] = int(idx)

    witness_stack_indices = sorted(
        int(stack_index[node])
        for node in witness_nodes
        if int(node) in stack_index
    )

    if witness_stack_indices:
        shallowest_target = int(min(witness_stack_indices))
        deepest_target = int(max(witness_stack_indices))
        shallowest_depth = int(len(state.assignment_stack) - shallowest_target)
        deepest_depth = int(len(state.assignment_stack) - deepest_target)
    else:
        shallowest_target = -1
        deepest_target = -1
        shallowest_depth = 0
        deepest_depth = 0

    return {
        "contradiction_nodes": contradiction_nodes,
        "witnesses": witnesses,
        "witness_nodes": witness_nodes,
        "witness_stack_indices": witness_stack_indices,
        "backjump_target": shallowest_target,
        "backjump_depth": shallowest_depth,
        "deepest_witness_target": deepest_target,
        "deepest_witness_depth": deepest_depth,
    }
