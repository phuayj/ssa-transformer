from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .dsl import GraphColorAction
from .env import GraphColorEnv, GraphColorEnvStatus, GraphColorState


class GraphColorOracle:
    """Expert graph k-coloring solver using DFS + forward checking.

    Heuristic:
      - Mandatory PROPAGATE after each assignment.
      - Backtrack on contradiction.
      - Variable selection: MRV over *effective* domain, tie-break by DSATUR saturation.
      - Value selection: smallest color in effective domain.
    """

    def __init__(self, env: GraphColorEnv):
        self.env = env
        self._retry: Optional[Tuple[int, int]] = None  # (depth, node) for retry after backtrack

    def _backtrack_action(self, state: GraphColorState) -> GraphColorAction:
        if not state.assignment_stack:
            # Nothing left to undo.
            return GraphColorAction.done()

        node, _color, _snapshot = state.assignment_stack[-1]
        depth = int(len(state.assignment_stack))  # 1-indexed assignment depth
        self._retry = (int(depth), int(node))
        return GraphColorAction.backtrack()

    def get_action(self, state: GraphColorState) -> GraphColorAction:
        # 1. If propagation pending -> PROPAGATE
        if state.status != GraphColorEnvStatus.RUNNING:
            return GraphColorAction.done()

        if state.propagation_pending:
            return GraphColorAction.propagate()

        # 2. If contradiction -> BACKTRACK
        if self.env._has_contradiction(state):
            return self._backtrack_action(state)

        # 3. If all assigned -> DONE
        if int(np.count_nonzero(state.assignment == 0)) == 0:
            return GraphColorAction.done()

        next_depth = int(len(state.assignment_stack) + 1)

        # 4. Handle retry after backtrack
        if self._retry is not None:
            retry_depth, retry_node = self._retry

            if int(next_depth) != int(retry_depth):
                # State moved (e.g., multiple backtracks). Drop retry.
                self._retry = None
            else:
                # Check candidates BEFORE selecting.
                candidates = sorted(self.env._effective_domain(state, retry_node, depth=retry_depth))
                if not candidates:
                    self._retry = None
                    return self._backtrack_action(state)

                if state.selected_node is None:
                    return GraphColorAction.select_node(int(retry_node))

                if int(state.selected_node) != int(retry_node):
                    # Force re-select.
                    return GraphColorAction.select_node(int(retry_node))

                # Now assign a new color.
                self._retry = None
                return GraphColorAction.assign_color(int(candidates[0]))

        # 5. Select node using DSATUR (min effective domain, tie-break by max saturation)
        if state.selected_node is None:
            node = self._dsatur_select(state, depth=next_depth)
            if node is None:
                return self._backtrack_action(state)
            return GraphColorAction.select_node(int(node))

        # 6. Assign smallest color in effective domain
        node = int(state.selected_node)
        candidates = sorted(self.env._effective_domain(state, node, depth=next_depth))
        if not candidates:
            return self._backtrack_action(state)
        return GraphColorAction.assign_color(int(candidates[0]))

    def _dsatur_select(self, state: GraphColorState, depth: int) -> Optional[int]:
        """Select an unassigned node.

        Criteria:
          1) Minimum effective domain size (MRV)
          2) Break ties by maximum saturation (distinct colors among colored neighbors)
          3) Break further ties by node index
        """

        best_node: Optional[int] = None
        best_dom_size = 10**9
        best_saturation = -1

        assigned = state.assignment
        adj = state.adjacency

        for node in range(state.num_nodes):
            if int(assigned[node]) != 0:
                continue

            dom = self.env._effective_domain(state, node, depth=int(depth))
            dom_size = int(len(dom))
            if dom_size == 0:
                continue

            neigh = np.nonzero(adj[node])[0]
            sat_colors = {int(assigned[int(j)]) for j in neigh if int(assigned[int(j)]) != 0}
            saturation = int(len(sat_colors))

            if (
                dom_size < best_dom_size
                or (dom_size == best_dom_size and saturation > best_saturation)
                or (
                    dom_size == best_dom_size
                    and saturation == best_saturation
                    and (best_node is None or node < best_node)
                )
            ):
                best_node = int(node)
                best_dom_size = int(dom_size)
                best_saturation = int(saturation)

        return best_node

    def solve(self) -> List[Tuple[dict, GraphColorAction]]:
        """Generate complete trace: list of (observation, action) pairs."""

        self._retry = None

        trace: List[Tuple[dict, GraphColorAction]] = []

        obs = self.env.reset()

        while True:
            state = self.env.get_state()
            action = self.get_action(state)

            trace.append((obs, action))

            res = self.env.step(action)
            obs = res.observation

            if res.done:
                return trace
