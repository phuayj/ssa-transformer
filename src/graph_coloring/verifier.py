from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np

from .dsl import GraphColorAction, GraphColorActionType
from .env import GraphColorState


def _current_depth(state: GraphColorState) -> int:
    return int(len(state.assignment_stack) + 1)


def _effective_domain(state: GraphColorState, node: int, *, depth: int) -> Set[int]:
    node_i = int(node)
    if node_i < 0 or node_i >= state.num_nodes:
        raise ValueError("node out of range")

    dom = set(state.domains[node_i])
    if int(state.assignment[node_i]) != 0:
        return dom

    banned = state.nogoods.get(int(depth), {}).get(int(node_i))
    if banned:
        dom.difference_update(banned)
    return dom


def _has_contradiction(state: GraphColorState) -> bool:
    # Empty domain on unassigned node.
    for i, dom in enumerate(state.domains):
        if int(state.assignment[i]) != 0:
            continue
        if len(dom) == 0:
            return True

    # Direct conflict (adjacent assigned same color).
    for i in range(state.num_nodes):
        c = int(state.assignment[i])
        if c == 0:
            continue
        if bool(np.any(state.adjacency[i] & (state.assignment == c))):
            return True

    return False


class GraphColorVerifier:
    def is_valid(self, state: GraphColorState, action: GraphColorAction) -> Tuple[bool, str]:
        """Check action validity based on current state."""

        if int(state.num_nodes) < 1:
            return False, "state.num_nodes must be >= 1"
        if int(state.num_colors) < 1:
            return False, "state.num_colors must be >= 1"
        if state.adjacency.shape != (state.num_nodes, state.num_nodes):
            return False, "adjacency has wrong shape"
        if state.assignment.shape != (state.num_nodes,):
            return False, "assignment has wrong shape"
        if len(state.domains) != state.num_nodes:
            return False, "domains length mismatch"

        depth = _current_depth(state)
        has_contra = _has_contradiction(state)

        if action.type == GraphColorActionType.SELECT_NODE:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if has_contra:
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
            if len(_effective_domain(state, node, depth=depth)) == 0:
                return False, "node domain empty"
            return True, ""

        if action.type == GraphColorActionType.ASSIGN_COLOR:
            if state.propagation_pending:
                return False, "PROPAGATE required"
            if has_contra:
                return False, "contradiction: must BACKTRACK"
            if state.selected_node is None:
                return False, "no node selected"
            if action.target is None:
                return False, "ASSIGN_COLOR missing target"
            color = int(action.target)
            if color < 1 or color > state.num_colors:
                return False, "color out of range"
            node = int(state.selected_node)
            if int(state.assignment[node]) != 0:
                return False, "node already assigned"
            if color not in _effective_domain(state, node, depth=depth):
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
            return True, ""

        return False, f"Unknown action type: {action.type}"

    def is_valid_coloring(self, adjacency: np.ndarray, assignment: np.ndarray) -> bool:
        """Check if assignment is a valid k-coloring (no adjacent same colors).

        Note: treats 0 as "unassigned" and ignores those entries.
        """

        adj = np.array(adjacency, copy=False)
        asn = np.array(assignment, copy=False)

        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            return False
        n = int(adj.shape[0])
        if asn.shape != (n,):
            return False

        adj_bool = adj.astype(bool, copy=False)

        # Check symmetry and no diagonal self-loops (best-effort).
        if not np.array_equal(adj_bool, adj_bool.T):
            return False
        if np.any(np.diag(adj_bool)):
            return False

        a = asn.reshape(n, 1)
        same = (a == a.T) & (a != 0)
        conflict = bool(np.any(adj_bool & same))
        return not conflict
