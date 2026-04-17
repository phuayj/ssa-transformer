from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GraphInstance:
    adjacency: np.ndarray  # (n, n) bool
    num_colors: int
    solution: Optional[np.ndarray]  # If known
    is_colorable: bool


class GraphGenerator:
    def __init__(
        self,
        num_nodes: int = 30,
        num_colors: int = 3,
        edge_prob: float = 0.3,
        seed: Optional[int] = None,
    ):
        if int(num_nodes) < 1:
            raise ValueError("num_nodes must be >= 1")
        if int(num_colors) < 1:
            raise ValueError("num_colors must be >= 1")
        if float(edge_prob) < 0.0 or float(edge_prob) > 1.0:
            raise ValueError("edge_prob must be in [0.0, 1.0]")

        self.num_nodes = int(num_nodes)
        self.num_colors = int(num_colors)
        self.edge_prob = float(edge_prob)
        self.rng = np.random.default_rng(seed)

    def _make_symmetric_no_diag(self, adj: np.ndarray) -> np.ndarray:
        if adj.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("internal error: adjacency has wrong shape")
        a = np.triu(adj.astype(bool, copy=False), 1)
        a = a | a.T
        np.fill_diagonal(a, False)
        return a.astype(bool, copy=False)

    def generate_erdos_renyi(self) -> GraphInstance:
        """Generate G(n,p) random graph and try to find a k-coloring."""

        u = self.rng.random((self.num_nodes, self.num_nodes))
        adj = self._make_symmetric_no_diag(u < self.edge_prob)

        # Try to find a coloring using the oracle.
        from .env import GraphColorEnv, GraphColorEnvStatus
        from .oracle import GraphColorOracle

        env = GraphColorEnv(adj, num_colors=self.num_colors, solution=None, mode="strict", max_steps=5000)
        oracle = GraphColorOracle(env)
        _trace = oracle.solve()

        st = env.get_state()
        if st.status == GraphColorEnvStatus.SUCCESS:
            sol = np.array(st.assignment, dtype=np.int64, copy=True)
            return GraphInstance(adjacency=adj, num_colors=int(self.num_colors), solution=sol, is_colorable=True)

        return GraphInstance(adjacency=adj, num_colors=int(self.num_colors), solution=None, is_colorable=False)

    def generate_planted(self) -> GraphInstance:
        """Generate graph with planted k-coloring (guaranteed colorable).

        1) Sample a random color for each node
        2) Add edges only between differently-colored nodes
        3) Control edge density via edge_prob
        """

        sol = self.rng.integers(1, self.num_colors + 1, size=(self.num_nodes,), dtype=np.int64)

        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if int(sol[i]) == int(sol[j]):
                    continue
                if float(self.rng.random()) < float(self.edge_prob):
                    adj[i, j] = True
                    adj[j, i] = True

        np.fill_diagonal(adj, False)

        return GraphInstance(
            adjacency=adj,
            num_colors=int(self.num_colors),
            solution=np.array(sol, dtype=np.int64, copy=True),
            is_colorable=True,
        )

    def generate(
        self,
        num_nodes: Optional[int] = None,
        num_colors: Optional[int] = None,
        edge_prob: Optional[float] = None,
        planted_ratio: float = 0.5,
    ) -> GraphInstance:
        """Mix of planted (colorable) and random (may be uncolorable).

        Args:
            num_nodes: Optional override for this generator instance.
            num_colors: Optional override for this generator instance.
            edge_prob: Optional override for this generator instance.
            planted_ratio: Probability of generating a planted (guaranteed-colorable) graph.
        """

        if num_nodes is not None:
            if int(num_nodes) < 1:
                raise ValueError("num_nodes must be >= 1")
            self.num_nodes = int(num_nodes)

        if num_colors is not None:
            if int(num_colors) < 1:
                raise ValueError("num_colors must be >= 1")
            self.num_colors = int(num_colors)

        if edge_prob is not None:
            if float(edge_prob) < 0.0 or float(edge_prob) > 1.0:
                raise ValueError("edge_prob must be in [0.0, 1.0]")
            self.edge_prob = float(edge_prob)

        if float(planted_ratio) < 0.0 or float(planted_ratio) > 1.0:
            raise ValueError("planted_ratio must be in [0.0, 1.0]")

        if float(self.rng.random()) < float(planted_ratio):
            return self.generate_planted()
        return self.generate_erdos_renyi()
