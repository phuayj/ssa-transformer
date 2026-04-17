"""Multi-domain dataset for universal backtracking training."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .types import UnifiedAction, UnifiedActionType, UnifiedObservation
from .wrapper import CSPUnifiedWrapper, GraphColoringUnifiedWrapper, SATUnifiedWrapper

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Configuration for a single domain."""

    name: str
    weight: float  # Sampling weight (will be normalized)
    num_instances: int
    params: dict


@dataclass
class MultiDomainDataConfig:
    """Configuration for multi-domain dataset."""

    # CSP (Sudoku)
    csp_num_instances: int = 1000
    csp_grid_size: int = 4  # 4 for 4x4, 9 for 9x9
    csp_num_clues: int = 4  # For 4x4; will need to scale for 9x9
    propagation_mode: str = "forward_check"  # "none" or "forward_check"
    csp_propagation_mix: bool = False
    csp_propagation_modes: Tuple[str, ...] = ("forward_check",)

    # Graph Coloring
    gc_num_instances: int = 1000
    gc_num_nodes: int = 20
    gc_num_colors: int = 4
    gc_edge_prob: float = 0.3
    gc_planted_ratio: float = 0.5
    # Graph Coloring - degree diversity
    gc_degree_diverse: bool = False  # If True, sample varying avg degrees
    gc_min_avg_degree: float = 3.0  # Minimum average degree
    gc_max_avg_degree: float = 20.0  # Maximum average degree
    # Graph Coloring - size diversity
    gc_size_diverse: bool = False  # If True, sample varying (n, p)
    gc_min_nodes: int = 15  # Minimum nodes when size_diverse=True
    gc_max_nodes: int = 45  # Maximum nodes when size_diverse=True
    gc_min_edge_prob: float = 0.15  # Minimum p when size_diverse=True
    gc_max_edge_prob: float = 0.45  # Maximum p when size_diverse=True

    # SAT
    sat_num_instances: int = 1000
    sat_num_vars: int = 20
    sat_alpha: float = 3.5
    sat_ratio: float = 0.5  # SAT vs UNSAT (via planted vs random)

    # Hybrid
    hybrid_num_instances: int = 500
    hybrid_n_color: int = 15
    hybrid_n_bool: int = 15
    hybrid_num_colors: int = 4
    hybrid_lambda: float = 0.3
    hybrid_d_col: float = 3.0
    hybrid_alpha: float = 3.0
    hybrid_beta: float = 1.5

    # General
    max_steps_per_instance: int = 500
    seed: int = 42

    # Model capacity (for collate padding)
    max_vars: int = 100
    max_constraints: int = 700  # Increased for 9x9 Sudoku (648)
    max_edges: int = 3000  # Increased for larger graphs
    max_domain: int = 10

    # Domain weights (balanced by decision steps)
    csp_weight: float = 1.0
    gc_weight: float = 1.0
    sat_weight: float = 1.0
    hybrid_weight: float = 1.0


class MultiDomainStepDataset(Dataset):
    """Dataset of (observation, action) decision steps across multiple domains.

    This dataset runs domain-specific oracles to generate expert traces, converts
    them into the unified action space, and stores individual decision steps for
    imitation learning.

    Domain mixing is balanced by *decision steps* (not instances) according to
    the configured domain weights.
    """

    def __init__(
        self,
        config: MultiDomainDataConfig,
        split: str = "train",
        domains: Optional[List[str]] = None,  # None = all domains
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val'; got {split!r}")

        self.config = config
        self.split = split
        self.domains = domains or ["gc", "sat"]

        unknown = [d for d in self.domains if d not in {"csp", "gc", "sat", "hybrid"}]
        if unknown:
            raise ValueError(f"Unknown domains: {unknown}")

        seed = int(config.seed)
        if split == "val":
            seed += 1_000_000

        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.steps: List[Tuple[UnifiedObservation, UnifiedAction, int]] = []
        self._generate_all_steps()

    def _num_instances_for_split(self, n: int) -> int:
        n = int(n)
        if self.split == "val":
            # Keep val smaller, but never larger than train.
            return min(n, max(n // 10, 50))
        return n

    def _generate_all_steps(self) -> None:
        per_domain_steps: dict[
            str, List[Tuple[UnifiedObservation, UnifiedAction, int]]
        ] = {}

        if "csp" in self.domains:
            per_domain_steps["csp"] = self._generate_csp_steps()

        if "gc" in self.domains:
            per_domain_steps["gc"] = self._generate_gc_steps()

        if "sat" in self.domains:
            per_domain_steps["sat"] = self._generate_sat_steps()

        if "hybrid" in self.domains:
            per_domain_steps["hybrid"] = self._generate_hybrid_steps()

        # Drop empty domains (keep dataset usable, but warn).
        per_domain_steps = {k: v for k, v in per_domain_steps.items() if len(v) > 0}
        if not per_domain_steps:
            self.steps = []
            return

        # Balance by decision steps using weights.
        weights = {
            "csp": float(self.config.csp_weight),
            "gc": float(self.config.gc_weight),
            "sat": float(self.config.sat_weight),
            "hybrid": float(self.config.hybrid_weight),
        }

        active_domains = sorted(per_domain_steps.keys())
        for d in active_domains:
            if weights[d] <= 0.0:
                raise ValueError(
                    f"Domain weight for {d!r} must be > 0; got {weights[d]}"
                )

        # Find largest scaling factor k such that weight[d]*k <= available_steps[d] for all d.
        k = min(float(len(per_domain_steps[d])) / weights[d] for d in active_domains)

        mixed: list[Tuple[UnifiedObservation, UnifiedAction, int]] = []
        per_domain_target: dict[str, int] = {}

        for d in active_domains:
            target = int(weights[d] * k)
            target = max(target, 0)
            per_domain_target[d] = target

            dom_steps = per_domain_steps[d]
            if target >= len(dom_steps):
                chosen = list(dom_steps)
            else:
                chosen = self.rng.sample(dom_steps, k=target)
            mixed.extend(chosen)

        self.rng.shuffle(mixed)
        self.steps = mixed

        logger.info(
            "MultiDomainStepDataset(%s) generated steps: %s",
            self.split,
            {
                d: {"available": len(per_domain_steps[d]), "used": per_domain_target[d]}
                for d in active_domains
            },
        )

    def _generate_csp_steps(
        self,
    ) -> List[Tuple[UnifiedObservation, UnifiedAction, int]]:
        from csp.env import SudokuEnv
        from csp.oracle import OracleCSP
        from csp.sudoku import SudokuGenerator

        from csp.dsl import CSPActionType

        num_instances = self._num_instances_for_split(self.config.csp_num_instances)

        gen = SudokuGenerator(
            grid_size=int(self.config.csp_grid_size),
            num_clues=int(self.config.csp_num_clues),
        )
        gen.use_box_constraints = False

        steps: List[Tuple[UnifiedObservation, UnifiedAction, int]] = []

        prop_mode_counts = {m: 0 for m in self.config.csp_propagation_modes}

        accepted = 0
        for _ in range(int(num_instances)):
            puzzle, solution = gen.generate()

            if (
                self.config.csp_propagation_mix
                and len(self.config.csp_propagation_modes) > 1
            ):
                prop_mode = self.rng.choice(self.config.csp_propagation_modes)
            else:
                prop_mode = self.config.propagation_mode
            prop_mode = str(prop_mode)
            prop_mode_counts[prop_mode] = prop_mode_counts.get(prop_mode, 0) + 1

            env = SudokuEnv(
                puzzle=puzzle,
                solution=solution,
                config=gen.config,
                mode="strict",
                max_steps=int(self.config.max_steps_per_instance),
                use_box_constraints=bool(gen.use_box_constraints),
                propagation_mode=prop_mode,
            )
            wrapper = CSPUnifiedWrapper(
                env,
                max_domain=int(self.config.max_domain),
                propagation_mode=str(prop_mode),
            )

            oracle = OracleCSP(env)
            trace = oracle.solve()
            if not trace:
                continue

            # Skip truncated rollouts (step limit / invalid): require terminal DONE.
            if trace[-1][1].type != CSPActionType.DONE:
                continue

            unified_actions: list[UnifiedAction] = []
            selected: Optional[int] = None

            for _native_obs, a in trace:
                if a.type == CSPActionType.SELECT_CELL:
                    if a.target is None:
                        continue
                    selected = int(a.target)

                elif a.type == CSPActionType.ASSIGN_VALUE:
                    if selected is None or a.target is None:
                        continue
                    unified_actions.append(
                        UnifiedAction.assign(int(selected), int(a.target) - 1)
                    )
                    selected = None

                elif a.type == CSPActionType.PROPAGATE:
                    continue

                elif a.type == CSPActionType.BACKTRACK:
                    unified_actions.append(UnifiedAction.backtrack())
                    selected = None

                elif a.type == CSPActionType.DONE:
                    unified_actions.append(UnifiedAction.done())
                    selected = None

            if not unified_actions:
                continue

            obs = wrapper.reset()
            for ua in unified_actions:
                steps.append((obs, ua, 0))

                obs, _reward, done, info = wrapper.step(ua)
                if not bool(info.get("valid", True)):
                    raise RuntimeError(f"Invalid unified CSP action {ua}: info={info}")

                if done:
                    break

            accepted += 1

        logger.info(
            "CSP steps: grid_size=%d num_clues=%d instances=%d accepted=%d steps=%d propagation_modes=%s",
            int(self.config.csp_grid_size),
            int(self.config.csp_num_clues),
            int(num_instances),
            int(accepted),
            int(len(steps)),
            dict(prop_mode_counts),
        )
        return steps

    def _generate_gc_steps(self) -> List[Tuple[UnifiedObservation, UnifiedAction, int]]:
        from graph_coloring.env import GraphColorEnv
        from graph_coloring.generator import GraphGenerator
        from graph_coloring.oracle import GraphColorOracle

        from graph_coloring.dsl import GraphColorActionType

        num_instances = self._num_instances_for_split(self.config.gc_num_instances)

        gen = GraphGenerator(seed=int(self.np_rng.integers(0, 2**31 - 1)))

        steps: List[Tuple[UnifiedObservation, UnifiedAction, int]] = []

        accepted = 0
        fixed_num_nodes = int(self.config.gc_num_nodes)
        degree_diverse = bool(self.config.gc_degree_diverse)
        size_diverse = bool(self.config.gc_size_diverse)
        min_avg_degree = float(self.config.gc_min_avg_degree)
        max_avg_degree = float(self.config.gc_max_avg_degree)
        min_nodes = int(self.config.gc_min_nodes)
        max_nodes = int(self.config.gc_max_nodes)
        min_edge_prob = float(self.config.gc_min_edge_prob)
        max_edge_prob = float(self.config.gc_max_edge_prob)
        planted_ratio = float(self.config.gc_planted_ratio)
        use_degree_diverse = bool(degree_diverse and not size_diverse)
        degree_targets: list[float] = []
        edge_probs: list[float] = []
        node_counts: list[int] = []
        avg_degrees: list[float] = []
        clipped = 0

        if planted_ratio < 0.0 or planted_ratio > 1.0:
            raise ValueError("gc_planted_ratio must be in [0.0, 1.0]")

        if size_diverse and degree_diverse:
            logger.warning(
                "gc_size_diverse=True overrides gc_degree_diverse; ignoring degree-based sampling."
            )

        if size_diverse:
            if min_nodes < 1 or max_nodes < 1:
                raise ValueError("gc_min_nodes and gc_max_nodes must be >= 1")
            if min_nodes > max_nodes:
                raise ValueError("gc_min_nodes must be <= gc_max_nodes")
            if min_edge_prob < 0.0 or max_edge_prob < 0.0:
                raise ValueError("gc_min_edge_prob and gc_max_edge_prob must be >= 0")
            if min_edge_prob > max_edge_prob:
                raise ValueError("gc_min_edge_prob must be <= gc_max_edge_prob")
            if min_edge_prob > 1.0 or max_edge_prob > 1.0:
                raise ValueError("gc_min_edge_prob and gc_max_edge_prob must be <= 1")

        if use_degree_diverse:
            if fixed_num_nodes <= 1:
                raise ValueError(
                    "gc_num_nodes must be >= 2 when gc_degree_diverse is True"
                )
            if min_avg_degree < 0.0 or max_avg_degree < 0.0:
                raise ValueError("gc_min_avg_degree and gc_max_avg_degree must be >= 0")
            if min_avg_degree > max_avg_degree:
                raise ValueError("gc_min_avg_degree must be <= gc_max_avg_degree")

        size_samples: list[tuple[int, float]] = []

        for instance_idx in range(int(num_instances)):
            if size_diverse:
                instance_rng = np.random.default_rng(self.seed + instance_idx)
                num_nodes = int(instance_rng.integers(min_nodes, max_nodes + 1))
                edge_prob = float(instance_rng.uniform(min_edge_prob, max_edge_prob))
                node_counts.append(num_nodes)
                edge_probs.append(edge_prob)
                size_samples.append((num_nodes, edge_prob))
                logger.debug(
                    "GC size sample idx=%d num_nodes=%d edge_prob=%.3f",
                    int(instance_idx),
                    int(num_nodes),
                    float(edge_prob),
                )
            elif use_degree_diverse:
                num_nodes = fixed_num_nodes
                # Sample target average degree per instance.
                target_degree = float(
                    self.np_rng.uniform(min_avg_degree, max_avg_degree)
                )
                edge_prob = target_degree / float(num_nodes - 1)
                clipped_edge_prob = float(np.clip(edge_prob, 0.05, 0.95))
                if clipped_edge_prob != edge_prob:
                    clipped += 1
                edge_prob = clipped_edge_prob
                degree_targets.append(target_degree)
                edge_probs.append(edge_prob)
            else:
                num_nodes = fixed_num_nodes
                edge_prob = float(self.config.gc_edge_prob)

            instance = gen.generate(
                num_nodes=num_nodes,
                num_colors=int(self.config.gc_num_colors),
                edge_prob=edge_prob,
                planted_ratio=planted_ratio,
            )
            avg_degree = (
                float(instance.adjacency.sum() / float(num_nodes))
                if num_nodes > 0
                else 0.0
            )
            avg_degrees.append(avg_degree)

            env = GraphColorEnv(
                adjacency=instance.adjacency,
                num_colors=int(instance.num_colors),
                solution=instance.solution,
                mode="strict",
                max_steps=int(self.config.max_steps_per_instance),
            )
            wrapper = GraphColoringUnifiedWrapper(
                env,
                max_domain=int(self.config.max_domain),
                propagation_mode="forward_check",
            )

            oracle = GraphColorOracle(env)
            trace = oracle.solve()
            if not trace:
                continue

            if trace[-1][1].type != GraphColorActionType.DONE:
                continue

            unified_actions: list[UnifiedAction] = []
            selected: Optional[int] = None

            for _native_obs, a in trace:
                if a.type == GraphColorActionType.SELECT_NODE:
                    if a.target is None:
                        continue
                    selected = int(a.target)

                elif a.type == GraphColorActionType.ASSIGN_COLOR:
                    if selected is None or a.target is None:
                        continue
                    unified_actions.append(
                        UnifiedAction.assign(int(selected), int(a.target) - 1)
                    )
                    selected = None

                elif a.type == GraphColorActionType.PROPAGATE:
                    continue

                elif a.type == GraphColorActionType.BACKTRACK:
                    unified_actions.append(UnifiedAction.backtrack())
                    selected = None

                elif a.type == GraphColorActionType.DONE:
                    unified_actions.append(UnifiedAction.done())
                    selected = None

            if not unified_actions:
                continue

            obs = wrapper.reset()
            for ua in unified_actions:
                steps.append((obs, ua, 1))

                obs, _reward, done, info = wrapper.step(ua)
                if not bool(info.get("valid", True)):
                    raise RuntimeError(f"Invalid unified GC action {ua}: info={info}")

                if done:
                    break

            accepted += 1

        if avg_degrees:
            avg_min = float(np.min(avg_degrees))
            avg_max = float(np.max(avg_degrees))
            avg_mean = float(np.mean(avg_degrees))
        else:
            avg_min = 0.0
            avg_max = 0.0
            avg_mean = 0.0

        if size_diverse:
            if node_counts:
                node_min = int(np.min(node_counts))
                node_max = int(np.max(node_counts))
                node_mean = float(np.mean(node_counts))
            else:
                node_min = 0
                node_max = 0
                node_mean = 0.0

            if edge_probs:
                edge_min = float(np.min(edge_probs))
                edge_max = float(np.max(edge_probs))
                edge_mean = float(np.mean(edge_probs))
            else:
                edge_min = 0.0
                edge_max = 0.0
                edge_mean = 0.0

            sample_preview = [
                (int(n), float(round(p, 3)))
                for n, p in size_samples[: min(5, len(size_samples))]
            ]

            logger.info(
                "GC size distribution: size_diverse=%s nodes[min=%d max=%d mean=%.2f] "
                "edge_prob[min=%.3f max=%.3f mean=%.3f] actual_avg_degree[min=%.2f max=%.2f mean=%.2f] "
                "sampled_pairs=%s",
                size_diverse,
                node_min,
                node_max,
                node_mean,
                edge_min,
                edge_max,
                edge_mean,
                avg_min,
                avg_max,
                avg_mean,
                sample_preview,
            )
        elif use_degree_diverse:
            if degree_targets:
                target_min = float(np.min(degree_targets))
                target_max = float(np.max(degree_targets))
                target_mean = float(np.mean(degree_targets))
            else:
                target_min = 0.0
                target_max = 0.0
                target_mean = 0.0

            if edge_probs:
                edge_min = float(np.min(edge_probs))
                edge_max = float(np.max(edge_probs))
                edge_mean = float(np.mean(edge_probs))
            else:
                edge_min = 0.0
                edge_max = 0.0
                edge_mean = 0.0

            logger.info(
                "GC degree distribution: degree_diverse=%s target_degree[min=%.2f max=%.2f mean=%.2f] "
                "edge_prob[min=%.3f max=%.3f mean=%.3f] actual_avg_degree[min=%.2f max=%.2f mean=%.2f] "
                "clipped=%d",
                use_degree_diverse,
                target_min,
                target_max,
                target_mean,
                edge_min,
                edge_max,
                edge_mean,
                avg_min,
                avg_max,
                avg_mean,
                int(clipped),
            )
        else:
            logger.info(
                "GC degree distribution: degree_diverse=%s edge_prob=%.3f num_nodes=%d "
                "actual_avg_degree[min=%.2f max=%.2f mean=%.2f]",
                use_degree_diverse,
                float(self.config.gc_edge_prob),
                int(fixed_num_nodes),
                avg_min,
                avg_max,
                avg_mean,
            )

        logger.info(
            "GC steps: instances=%d accepted=%d steps=%d planted_ratio=%.2f",
            int(num_instances),
            int(accepted),
            int(len(steps)),
            float(planted_ratio),
        )
        return steps

    def _generate_sat_steps(
        self,
    ) -> List[Tuple[UnifiedObservation, UnifiedAction, int]]:
        from sat.env import SatEnv
        from sat.generator import SatGenerator
        from sat.oracle import SatOracle

        from sat.dsl import SatActionType

        num_instances = self._num_instances_for_split(self.config.sat_num_instances)

        gen = SatGenerator(seed=int(self.np_rng.integers(0, 2**31 - 1)))

        steps: List[Tuple[UnifiedObservation, UnifiedAction, int]] = []

        accepted = 0
        for _ in range(int(num_instances)):
            instance = gen.generate(
                num_vars=int(self.config.sat_num_vars),
                alpha_sat=float(self.config.sat_alpha),
                alpha_unsat=float(self.config.sat_alpha) + 2.0,
                sat_ratio=float(self.config.sat_ratio),
            )

            env = SatEnv(
                clauses=instance.clauses,
                num_vars=int(instance.num_vars),
                planted_solution=instance.planted_solution,
                mode="strict",
                max_steps=int(self.config.max_steps_per_instance),
            )
            wrapper = SATUnifiedWrapper(
                env,
                max_domain=int(self.config.max_domain),
                propagation_mode="forward_check",
            )

            oracle = SatOracle(env)
            trace = oracle.solve()
            if not trace:
                continue

            if trace[-1][1].type != SatActionType.DONE:
                continue

            unified_actions: list[UnifiedAction] = []
            selected: Optional[int] = None

            for _native_obs, a in trace:
                if a.type == SatActionType.SELECT_VAR:
                    if a.target is None:
                        continue
                    selected = int(a.target)

                elif a.type == SatActionType.ASSIGN_VALUE:
                    if selected is None or a.target is None:
                        continue
                    unified_actions.append(
                        UnifiedAction.assign(int(selected), int(a.target))
                    )
                    selected = None

                elif a.type == SatActionType.PROPAGATE:
                    continue

                elif a.type == SatActionType.BACKTRACK:
                    unified_actions.append(UnifiedAction.backtrack())
                    selected = None

                elif a.type == SatActionType.DONE:
                    unified_actions.append(UnifiedAction.done())
                    selected = None

            if not unified_actions:
                continue

            obs = wrapper.reset()
            for ua in unified_actions:
                steps.append((obs, ua, 2))

                obs, _reward, done, info = wrapper.step(ua)
                if not bool(info.get("valid", True)):
                    raise RuntimeError(f"Invalid unified SAT action {ua}: info={info}")

                if done:
                    break

            accepted += 1

        logger.info(
            "SAT steps: instances=%d accepted=%d steps=%d",
            int(num_instances),
            int(accepted),
            int(len(steps)),
        )
        return steps

    def _generate_hybrid_steps(
        self,
    ) -> List[Tuple[UnifiedObservation, UnifiedAction, int]]:
        from hybrid import HybridActionType, HybridEnv, HybridGenerator, HybridOracle
        from universal.wrapper import HybridUnifiedWrapper

        num_instances = self._num_instances_for_split(self.config.hybrid_num_instances)

        gen = HybridGenerator(seed=int(self.np_rng.integers(0, 2**31 - 1)))

        steps: List[Tuple[UnifiedObservation, UnifiedAction, int]] = []

        accepted = 0
        for _ in range(int(num_instances)):
            instance = gen.generate(
                n_color=int(self.config.hybrid_n_color),
                n_bool=int(self.config.hybrid_n_bool),
                k=int(self.config.hybrid_num_colors),
                lam=float(self.config.hybrid_lambda),
                d_col=float(self.config.hybrid_d_col),
                alpha=float(self.config.hybrid_alpha),
                beta=float(self.config.hybrid_beta),
            )

            env = HybridEnv(
                instance=instance,
                mode="strict",
                max_steps=int(self.config.max_steps_per_instance),
            )
            wrapper = HybridUnifiedWrapper(env, max_domain=int(self.config.max_domain))

            oracle = HybridOracle(env)
            trace = oracle.solve()
            if not trace:
                continue

            if trace[-1][1].type != HybridActionType.DONE:
                continue

            unified_actions: list[UnifiedAction] = []
            selected: Optional[int] = None

            for _native_obs, a in trace:
                if a.type == HybridActionType.SELECT_VAR:
                    if a.target is None:
                        continue
                    selected = int(a.target)

                elif a.type == HybridActionType.ASSIGN_VALUE:
                    if selected is None or a.target is None:
                        continue
                    if int(selected) < int(instance.n_color):
                        unified_value = int(a.target) - 1
                    else:
                        unified_value = int(a.target)
                    unified_actions.append(
                        UnifiedAction.assign(int(selected), int(unified_value))
                    )
                    selected = None

                elif a.type == HybridActionType.PROPAGATE:
                    continue

                elif a.type == HybridActionType.BACKTRACK:
                    unified_actions.append(UnifiedAction.backtrack())
                    selected = None

                elif a.type == HybridActionType.DONE:
                    unified_actions.append(UnifiedAction.done())
                    selected = None

            if not unified_actions:
                continue

            obs = wrapper.reset()
            for ua in unified_actions:
                steps.append((obs, ua, 3))

                obs, _reward, done, info = wrapper.step(ua)
                if not bool(info.get("valid", True)):
                    raise RuntimeError(
                        f"Invalid unified Hybrid action {ua}: info={info}"
                    )

                if done:
                    break

            accepted += 1

        logger.info(
            "Hybrid steps: instances=%d accepted=%d steps=%d n_color=%d n_bool=%d k=%d "
            "lambda=%.2f d_col=%.2f alpha=%.2f beta=%.2f",
            int(num_instances),
            int(accepted),
            int(len(steps)),
            int(self.config.hybrid_n_color),
            int(self.config.hybrid_n_bool),
            int(self.config.hybrid_num_colors),
            float(self.config.hybrid_lambda),
            float(self.config.hybrid_d_col),
            float(self.config.hybrid_alpha),
            float(self.config.hybrid_beta),
        )
        return steps

    def __len__(self) -> int:
        return int(len(self.steps))

    def _get_single_item(self, idx: int) -> Dict[str, Any]:
        obs, action, domain_id = self.steps[int(idx)]

        if action.type == UnifiedActionType.ASSIGN:
            action_type = 0
            action_var = int(action.var) if action.var is not None else 0
            action_value = int(action.value) if action.value is not None else 0
        elif action.type == UnifiedActionType.BACKTRACK:
            action_type = 1
            action_var = 0
            action_value = 0
        else:
            action_type = 2
            action_var = 0
            action_value = 0

        return {
            "var_features": obs.var_features.astype(np.float32, copy=False),
            "var_domain_mask": obs.var_domain_mask,
            "var_nogood_mask": obs.var_nogood_mask,
            "var_assigned": obs.var_assigned,
            "con_type": obs.con_type,
            "con_features": obs.con_features.astype(np.float32, copy=False),
            "edge_con_idx": obs.edge_con_idx,
            "edge_var_idx": obs.edge_var_idx,
            "edge_features": obs.edge_features.astype(np.float32, copy=False),
            "num_vars": int(obs.num_vars),
            "num_constraints": int(obs.num_constraints),
            "stack_depth": int(obs.stack_depth),
            "propagation_pending": bool(obs.propagation_pending),
            "has_conflict": bool(obs.has_conflict),
            "propagation_mode": int(getattr(obs, "propagation_mode", 1)),
            "action_type": int(action_type),
            "action_var": int(action_var),
            "action_value": int(action_value),
            "domain_id": int(domain_id),
        }

    def __getitem__(self, idx):  # type: ignore[override]
        return self._get_single_item(int(idx))

    def __getitems__(self, indices):
        return [self._get_single_item(int(i)) for i in indices]


def collate_multi_domain(
    batch: List[Dict[str, Any]],
    max_vars: int = 100,
    max_constraints: int = 500,
    max_domain: int = 10,
    max_edges: int = 2000,
) -> Dict[str, torch.Tensor]:
    """Collate function that pads unified observations to fixed sizes."""

    if not batch:
        raise ValueError("Empty batch")

    B = len(batch)

    v_feat_dim = int(batch[0]["var_features"].shape[1])
    c_feat_dim = int(batch[0]["con_features"].shape[1])
    e_feat_dim = int(batch[0]["edge_features"].shape[1])

    var_features = np.zeros((B, int(max_vars), v_feat_dim), dtype=np.float32)
    var_domain_mask = np.zeros((B, int(max_vars), int(max_domain)), dtype=bool)
    var_nogood_mask = np.zeros((B, int(max_vars), int(max_domain)), dtype=bool)
    var_assigned = np.full((B, int(max_vars)), -1, dtype=np.int64)

    con_type = np.zeros((B, int(max_constraints)), dtype=np.int64)
    con_features = np.zeros((B, int(max_constraints), c_feat_dim), dtype=np.float32)

    edge_con_idx = np.zeros((B, int(max_edges)), dtype=np.int64)
    edge_var_idx = np.zeros((B, int(max_edges)), dtype=np.int64)
    edge_features = np.zeros((B, int(max_edges), e_feat_dim), dtype=np.float32)

    var_mask = np.zeros((B, int(max_vars)), dtype=bool)
    con_mask = np.zeros((B, int(max_constraints)), dtype=bool)
    edge_mask = np.zeros((B, int(max_edges)), dtype=bool)

    # (stack_depth_norm, propagation_pending, has_conflict, propagation_mode)
    global_features = np.zeros((B, 4), dtype=np.float32)

    action_type = np.zeros(B, dtype=np.int64)
    action_var = np.zeros(B, dtype=np.int64)
    action_value = np.zeros(B, dtype=np.int64)
    domain_id = np.zeros(B, dtype=np.int64)

    for i, item in enumerate(batch):
        N = int(item["num_vars"])
        M = int(item["num_constraints"])
        E = int(len(item["edge_con_idx"]))
        D = int(item["var_domain_mask"].shape[1])

        if N > int(max_vars):
            raise ValueError(f"num_vars={N} exceeds max_vars={max_vars}")
        if M > int(max_constraints):
            raise ValueError(
                f"num_constraints={M} exceeds max_constraints={max_constraints}"
            )
        if E > int(max_edges):
            raise ValueError(f"num_edges={E} exceeds max_edges={max_edges}")

        d_fill = min(D, int(max_domain))

        var_features[i, :N] = item["var_features"]
        var_domain_mask[i, :N, :d_fill] = item["var_domain_mask"][:, :d_fill]
        var_nogood_mask[i, :N, :d_fill] = item["var_nogood_mask"][:, :d_fill]
        var_assigned[i, :N] = item["var_assigned"]

        con_type[i, :M] = item["con_type"]
        con_features[i, :M] = item["con_features"]

        edge_con_idx[i, :E] = item["edge_con_idx"]
        edge_var_idx[i, :E] = item["edge_var_idx"]
        edge_features[i, :E] = item["edge_features"]

        var_mask[i, :N] = True
        con_mask[i, :M] = True
        edge_mask[i, :E] = True

        global_features[i, 0] = float(item["stack_depth"]) / 50.0
        global_features[i, 1] = float(item["propagation_pending"])
        global_features[i, 2] = float(item["has_conflict"])
        global_features[i, 3] = float(item.get("propagation_mode", 1))

        action_type[i] = int(item["action_type"])
        action_var[i] = int(item["action_var"])
        action_value[i] = int(item["action_value"])
        domain_id[i] = int(item["domain_id"])

    return {
        "var_features": torch.from_numpy(var_features),
        "var_domain_mask": torch.from_numpy(var_domain_mask),
        "var_nogood_mask": torch.from_numpy(var_nogood_mask),
        "var_assigned": torch.from_numpy(var_assigned),
        "con_type": torch.from_numpy(con_type),
        "con_features": torch.from_numpy(con_features),
        "edge_con_idx": torch.from_numpy(edge_con_idx),
        "edge_var_idx": torch.from_numpy(edge_var_idx),
        "edge_features": torch.from_numpy(edge_features),
        "var_mask": torch.from_numpy(var_mask),
        "con_mask": torch.from_numpy(con_mask),
        "edge_mask": torch.from_numpy(edge_mask),
        "global_features": torch.from_numpy(global_features),
        "action_type": torch.from_numpy(action_type),
        "action_var": torch.from_numpy(action_var),
        "action_value": torch.from_numpy(action_value),
        "domain_id": torch.from_numpy(domain_id),
    }


def create_multi_domain_dataloader(
    config: MultiDomainDataConfig,
    split: str = "train",
    batch_size: int = 64,
    domains: Optional[List[str]] = None,
    **kwargs: Any,
) -> DataLoader:
    """Create DataLoader for multi-domain training."""

    from functools import partial

    dataset = MultiDomainStepDataset(config=config, split=split, domains=domains)

    collate_fn = partial(
        collate_multi_domain,
        max_vars=int(config.max_vars),
        max_constraints=int(config.max_constraints),
        max_domain=int(config.max_domain),
        max_edges=int(config.max_edges),
    )

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        **kwargs,
    )
