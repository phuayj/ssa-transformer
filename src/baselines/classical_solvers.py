from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from csp.sudoku import (
        SudokuConfig,
        get_box_constraints,
        get_col_constraints,
        get_row_constraints,
        is_valid_assignment,
    )

    _HAS_CSP = True
except ImportError:
    _HAS_CSP = False

logger = logging.getLogger(__name__)


@dataclass
class DSATURStats:
    steps: int = 0
    backtracks: int = 0


def _edges_to_adjacency(edges: List[Tuple[int, int]]) -> np.ndarray:
    if not edges:
        raise ValueError("edge list must be non-empty")
    max_node = max(max(int(u), int(v)) for u, v in edges)
    n = int(max_node) + 1
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        if u == v:
            raise ValueError("self-loops are not allowed in edge list")
        adj[int(u), int(v)] = True
        adj[int(v), int(u)] = True
    np.fill_diagonal(adj, False)
    return adj


def _normalize_graph(graph: object) -> np.ndarray:
    if isinstance(graph, np.ndarray):
        arr = np.array(graph, copy=True)
        if arr.ndim != 2:
            raise ValueError("graph adjacency must be 2D")
        if arr.shape[0] == arr.shape[1]:
            adj = arr.astype(bool, copy=False)
            if np.any(np.diag(adj)):
                raise ValueError("adjacency must have no self-loops")
            return adj
        if arr.shape[1] == 2:
            edges = [(int(u), int(v)) for u, v in arr.tolist()]
            return _edges_to_adjacency(edges)
        raise ValueError("graph adjacency must be square or an edge list")

    if isinstance(graph, (list, tuple)):
        if not graph:
            raise ValueError("graph must be non-empty")

        if not all(isinstance(item, (list, tuple, set, np.ndarray)) for item in graph):
            raise ValueError("graph list entries must be iterables")

        n = len(graph)
        neighbors_list: List[Set[int]] = []
        adjacency_ok = True

        for i, neighbors in enumerate(graph):
            try:
                neighbor_iter = (
                    neighbors.tolist()
                    if isinstance(neighbors, np.ndarray)
                    else list(neighbors)
                )
            except TypeError:
                adjacency_ok = False
                break

            neigh_set: Set[int] = set()
            for j in neighbor_iter:
                j_int = int(j)
                if j_int < 0 or j_int >= n or j_int == i:
                    adjacency_ok = False
                    break
                neigh_set.add(j_int)
            if not adjacency_ok:
                break
            neighbors_list.append(neigh_set)

        if adjacency_ok:
            adj = np.zeros((n, n), dtype=bool)
            for i, neigh in enumerate(neighbors_list):
                for j in neigh:
                    adj[int(i), int(j)] = True
            np.fill_diagonal(adj, False)
            return np.array(adj | adj.T, dtype=bool)

        edges: List[Tuple[int, int]] = []
        for item in graph:
            pair = np.asarray(item, dtype=np.int64).reshape(-1)
            if pair.shape[0] != 2:
                raise ValueError("graph list entries must be length-2 edge pairs")
            edges.append((int(pair[0]), int(pair[1])))
        return _edges_to_adjacency(edges)

    raise ValueError("graph must be an adjacency matrix, list, or edge list")


def _dsatur_select(
    adjacency: np.ndarray,
    assignment: np.ndarray,
    unassigned: Set[int],
) -> Optional[int]:
    best_node: Optional[int] = None
    best_saturation = -1
    best_degree = -1

    for node in unassigned:
        neighbors = np.nonzero(adjacency[node])[0]
        sat_colors = {int(assignment[n]) for n in neighbors if int(assignment[n]) != 0}
        saturation = int(len(sat_colors))
        degree = int(adjacency[node].sum())

        if (
            saturation > best_saturation
            or (saturation == best_saturation and degree > best_degree)
            or (
                saturation == best_saturation
                and degree == best_degree
                and (best_node is None or node < best_node)
            )
        ):
            best_node = int(node)
            best_saturation = int(saturation)
            best_degree = int(degree)

    return best_node


def dsatur_solve(
    graph: object,
    num_colors: int,
    max_steps: int = 1000,
) -> Dict[str, object]:
    """Solve graph coloring using DSATUR heuristic with backtracking.

    Args:
        graph: Adjacency list or adjacency matrix
        num_colors: Number of colors (k)
        max_steps: Maximum recursive calls before timeout

    Returns:
        dict with keys: success, coloring, steps, backtracks
    """

    if int(num_colors) < 1:
        raise ValueError("num_colors must be >= 1")
    if int(max_steps) < 1:
        raise ValueError("max_steps must be >= 1")

    adjacency = _normalize_graph(graph)
    n = int(adjacency.shape[0])

    assignment = np.zeros((n,), dtype=np.int64)
    unassigned: Set[int] = set(range(n))
    stats = DSATURStats()

    def _available_colors(node: int) -> List[int]:
        neighbors = np.nonzero(adjacency[node])[0]
        used = {int(assignment[n]) for n in neighbors if int(assignment[n]) != 0}
        return [c for c in range(1, int(num_colors) + 1) if c not in used]

    def _search() -> bool:
        if stats.steps >= int(max_steps):
            return False
        if not unassigned:
            return True

        stats.steps += 1
        node = _dsatur_select(adjacency, assignment, unassigned)
        if node is None:
            return False

        candidates = _available_colors(node)
        if not candidates:
            return False

        unassigned.remove(node)
        for color in candidates:
            assignment[node] = int(color)
            if _search():
                return True
            assignment[node] = 0
            stats.backtracks += 1

        unassigned.add(node)
        return False

    success = _search()

    logger.info(
        "DSATUR finished success=%s steps=%d backtracks=%d max_steps=%d",
        bool(success),
        int(stats.steps),
        int(stats.backtracks),
        int(max_steps),
    )

    coloring = assignment.tolist() if bool(success) else assignment.tolist()
    return {
        "success": bool(success),
        "coloring": coloring,
        "steps": int(stats.steps),
        "backtracks": int(stats.backtracks),
    }


@dataclass
class MRVStats:
    steps: int = 0
    backtracks: int = 0


def _sudoku_peers(config: SudokuConfig, use_box_constraints: bool) -> List[List[int]]:
    peers: List[List[int]] = []
    for idx in range(config.num_cells):
        cell_peers = set(get_row_constraints(idx, config)) | set(
            get_col_constraints(idx, config)
        )
        if use_box_constraints:
            cell_peers |= set(get_box_constraints(idx, config))
        cell_peers.discard(idx)
        peers.append(sorted(int(x) for x in cell_peers))
    return peers


def _initial_domains(
    grid: np.ndarray,
    config: SudokuConfig,
    *,
    use_box_constraints: bool,
    peers: List[List[int]],
) -> Optional[List[Set[int]]]:
    domains: List[Set[int]] = []
    for idx in range(config.num_cells):
        r, c = divmod(idx, config.grid_size)
        val = int(grid[r, c])
        if val != 0:
            domains.append({val})
        else:
            domains.append(set(config.values))

    for idx in range(config.num_cells):
        r, c = divmod(idx, config.grid_size)
        v = int(grid[r, c])
        if v == 0:
            continue
        for peer in peers[idx]:
            pr, pc = divmod(peer, config.grid_size)
            if int(grid[pr, pc]) == v:
                return None
            domains[peer].discard(v)
            if len(domains[peer]) == 0 and int(grid[pr, pc]) == 0:
                return None

    return domains


def _select_mrv(domains: List[Set[int]], grid: np.ndarray) -> Optional[int]:
    best_idx: Optional[int] = None
    best_size = 10**9

    for idx, dom in enumerate(domains):
        r, c = divmod(idx, grid.shape[0])
        if int(grid[r, c]) != 0:
            continue
        size = int(len(dom))
        if size == 0:
            return None
        if size < best_size:
            best_idx = int(idx)
            best_size = int(size)
            if best_size == 1:
                break
    return best_idx


def _order_lcv(
    idx: int,
    domains: List[Set[int]],
    grid: np.ndarray,
    peers: List[List[int]],
) -> List[int]:
    scores: List[Tuple[int, int]] = []
    for val in sorted(domains[idx]):
        eliminated = 0
        for peer in peers[idx]:
            pr, pc = divmod(peer, grid.shape[0])
            if int(grid[pr, pc]) != 0:
                continue
            if val in domains[peer]:
                eliminated += 1
        scores.append((eliminated, int(val)))
    scores.sort()
    return [val for _elim, val in scores]


def mrv_lcv_solve(
    puzzle: np.ndarray,
    max_steps: int = 1000,
) -> Dict[str, object]:
    """Solve Sudoku using MRV + LCV + Forward-Check.

    Args:
        puzzle: 2D numpy array (0 = empty)
        max_steps: Maximum recursive calls before timeout

    Returns:
        dict with keys: success, solution, steps, backtracks
    """

    if not _HAS_CSP:
        raise ImportError(
            "csp module not available; install csp domain to use mrv_lcv_solve"
        )

    if int(max_steps) < 1:
        raise ValueError("max_steps must be >= 1")

    grid = np.array(puzzle, dtype=np.int64, copy=True)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError("puzzle must be square")

    config = SudokuConfig.standard(int(grid.shape[0]))
    use_box_constraints = False
    peers = _sudoku_peers(config, use_box_constraints=use_box_constraints)

    domains = _initial_domains(
        grid,
        config,
        use_box_constraints=use_box_constraints,
        peers=peers,
    )
    if domains is None:
        return {
            "success": False,
            "solution": grid,
            "steps": 0,
            "backtracks": 0,
        }

    stats = MRVStats()

    def _forward_check(
        idx: int,
        val: int,
        domains_in: List[Set[int]],
    ) -> Optional[List[Set[int]]]:
        new_domains = [set(d) for d in domains_in]
        new_domains[idx] = {int(val)}
        for peer in peers[idx]:
            pr, pc = divmod(peer, grid.shape[0])
            if int(grid[pr, pc]) != 0:
                continue
            if val in new_domains[peer]:
                new_domains[peer].discard(int(val))
                if len(new_domains[peer]) == 0:
                    return None
        return new_domains

    def _search(domains_state: List[Set[int]]) -> bool:
        if stats.steps >= int(max_steps):
            return False

        idx = _select_mrv(domains_state, grid)
        if idx is None:
            return bool(np.all(grid != 0))

        r, c = divmod(idx, grid.shape[0])
        if int(grid[r, c]) != 0:
            return _search(domains_state)

        stats.steps += 1
        values = _order_lcv(idx, domains_state, grid, peers)
        for val in values:
            if not is_valid_assignment(
                grid,
                r,
                c,
                int(val),
                config,
                use_box_constraints=use_box_constraints,
            ):
                continue

            grid[r, c] = int(val)
            next_domains = _forward_check(idx, int(val), domains_state)
            if next_domains is not None and _search(next_domains):
                return True
            grid[r, c] = 0
            stats.backtracks += 1

        return False

    success = _search(domains)

    logger.info(
        "MRV+LCV finished success=%s steps=%d backtracks=%d max_steps=%d",
        bool(success),
        int(stats.steps),
        int(stats.backtracks),
        int(max_steps),
    )

    return {
        "success": bool(success),
        "solution": np.array(grid, copy=True),
        "steps": int(stats.steps),
        "backtracks": int(stats.backtracks),
    }
