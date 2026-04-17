from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .dsl import GraphColorActionType
from .env import GraphColorEnv, GraphColorState
from .generator import GraphGenerator, GraphInstance
from .oracle import GraphColorOracle
from .verifier import GraphColorVerifier


@dataclass
class GraphColorDataConfig:
    num_train: int = 5000
    num_val: int = 500
    num_test: int = 0

    num_nodes: int = 30
    num_colors: int = 3
    edge_prob: float = 0.3
    planted_ratio: float = 0.5

    max_steps: int = 500
    seed: int = 42


def _split_seed(seed: int, split: str) -> int:
    if split == "train":
        return int(seed) + 0
    if split == "val":
        return int(seed) + 1_000_000
    if split == "test":
        return int(seed) + 2_000_000
    raise ValueError(f"Unknown split: {split!r}")


def _num_graphs_for_split(config: GraphColorDataConfig, split: str) -> int:
    if split == "train":
        return int(config.num_train)
    if split == "val":
        return int(config.num_val)
    if split == "test":
        return int(config.num_test)
    raise ValueError(f"Unknown split: {split!r}")


class GraphColorStepDataset(Dataset):
    """Step dataset for Graph k-Coloring.

    Each item is one (observation, action) step from an oracle trajectory.

    Returned dict:
      global_features: (5,) long
          [selected_node (-1 if none), num_assigned, num_empty_domains,
           propagation_pending (0/1), stack_depth]

      node_features: (n, 4) long
          For each node: [node_idx, degree, assigned_color (0 if unassigned), is_selected]

      adjacency: (n, n) bool

      domain_values: (n, k) long
      domain_mask: (n, k) bool

      action_type: scalar long in [0,4]
      action_target: scalar long
          SELECT_NODE: node index
          ASSIGN_COLOR: color-1 (0..k-1)
          else: 0

      action_valid: scalar long (0/1)
    """

    def __init__(self, config: GraphColorDataConfig, split: str = "train"):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split!r}")
        if int(config.num_nodes) < 1:
            raise ValueError("num_nodes must be >= 1")
        if int(config.num_colors) < 1:
            raise ValueError("num_colors must be >= 1")
        if float(config.edge_prob) < 0.0 or float(config.edge_prob) > 1.0:
            raise ValueError("edge_prob must be in [0.0, 1.0]")
        if float(config.planted_ratio) < 0.0 or float(config.planted_ratio) > 1.0:
            raise ValueError("planted_ratio must be in [0.0, 1.0]")
        if int(config.max_steps) < 1:
            raise ValueError("max_steps must be >= 1")

        self.config = config
        self.split = split

        self._verifier = GraphColorVerifier()

        self._examples = self._generate_examples()
        if not self._examples:
            raise RuntimeError("Generated empty GraphColorStepDataset")

    def _extract_example(self, obs: dict, state: GraphColorState, action) -> dict:
        n = int(state.num_nodes)
        k = int(state.num_colors)

        # --- Global features ---
        selected_node = int(obs["global"][0])
        num_assigned = int(obs["global"][1])
        num_empty_domains = int(obs["global"][2])
        propagation_pending = int(bool(state.propagation_pending))
        stack_depth = int(len(state.assignment_stack))

        global_features = [
            selected_node,
            num_assigned,
            num_empty_domains,
            propagation_pending,
            stack_depth,
        ]

        # --- Node features ---
        nodes = obs["nodes"]
        if len(nodes) != n:
            raise RuntimeError(f"Expected {n} nodes; got {len(nodes)}")
        node_features = [[int(x) for x in row] for row in nodes]

        # --- Adjacency ---
        adj = obs["adjacency"]
        if len(adj) != n or any(len(row) != n for row in adj):
            raise RuntimeError("adjacency has wrong shape in observation")

        # --- Domains ---
        domains = obs["domains"]
        if len(domains) != n:
            raise RuntimeError(f"Expected {n} domains; got {len(domains)}")

        domain_values: List[List[int]] = []
        domain_mask: List[List[bool]] = []
        for d in domains:
            vals = sorted(int(v) for v in d)
            padded = vals[:k] + [0] * max(0, k - len(vals))
            mask = [True] * min(k, len(vals)) + [False] * max(0, k - len(vals))
            domain_values.append([int(x) for x in padded])
            domain_mask.append([bool(x) for x in mask])

        # --- Targets ---
        if action.type == GraphColorActionType.SELECT_NODE:
            if action.target is None:
                raise RuntimeError("SELECT_NODE missing target")
            action_type = int(GraphColorActionType.SELECT_NODE)
            action_target = int(action.target)
        elif action.type == GraphColorActionType.ASSIGN_COLOR:
            if action.target is None:
                raise RuntimeError("ASSIGN_COLOR missing target")
            action_type = int(GraphColorActionType.ASSIGN_COLOR)
            action_target = int(action.target) - 1
        elif action.type == GraphColorActionType.PROPAGATE:
            action_type = int(GraphColorActionType.PROPAGATE)
            action_target = 0
        elif action.type == GraphColorActionType.BACKTRACK:
            action_type = int(GraphColorActionType.BACKTRACK)
            action_target = 0
        elif action.type == GraphColorActionType.DONE:
            action_type = int(GraphColorActionType.DONE)
            action_target = 0
        else:
            raise RuntimeError(f"Unknown action type: {action.type}")

        ok, reason = self._verifier.is_valid(state, action)
        if not ok:
            raise RuntimeError(
                "Oracle produced invalid action. "
                f"action={action.to_token()} reason={reason!r} "
                f"selected_node={selected_node} propagation_pending={propagation_pending} "
                f"stack_depth={stack_depth} num_assigned={num_assigned} num_empty_domains={num_empty_domains}"
            )

        return {
            "global_features": global_features,
            "node_features": node_features,
            "adjacency": [[bool(x) for x in row] for row in adj],
            "domain_values": domain_values,
            "domain_mask": domain_mask,
            "action_type": int(action_type),
            "action_target": int(action_target),
            "action_valid": int(ok),
        }

    def _generate_instance_examples(self, instance: GraphInstance) -> List[dict]:
        env = GraphColorEnv(
            adjacency=instance.adjacency,
            num_colors=int(instance.num_colors),
            solution=instance.solution,
            mode="strict",
            max_steps=int(self.config.max_steps),
        )
        oracle = GraphColorOracle(env)

        examples: List[dict] = []

        obs = env.reset()
        while True:
            state = env.get_state()
            action = oracle.get_action(state)

            examples.append(self._extract_example(obs, state, action))

            res = env.step(action)
            obs = res.observation

            if res.done:
                break

        return examples

    def _generate_examples(self) -> Dict[str, torch.Tensor]:
        seed = _split_seed(self.config.seed, self.split)
        n_graphs = _num_graphs_for_split(self.config, self.split)

        gen = GraphGenerator(
            num_nodes=int(self.config.num_nodes),
            num_colors=int(self.config.num_colors),
            edge_prob=float(self.config.edge_prob),
            seed=int(seed),
        )

        all_examples: List[dict] = []

        for _ in range(int(n_graphs)):
            instance = gen.generate(planted_ratio=float(self.config.planted_ratio))
            all_examples.extend(self._generate_instance_examples(instance))

        if not all_examples:
            return {}

        return {
            "global_features": torch.tensor(
                [ex["global_features"] for ex in all_examples], dtype=torch.long
            ),
            "node_features": torch.tensor([ex["node_features"] for ex in all_examples], dtype=torch.long),
            "adjacency": torch.tensor([ex["adjacency"] for ex in all_examples], dtype=torch.bool),
            "domain_values": torch.tensor([ex["domain_values"] for ex in all_examples], dtype=torch.long),
            "domain_mask": torch.tensor([ex["domain_mask"] for ex in all_examples], dtype=torch.bool),
            "action_type": torch.tensor([ex["action_type"] for ex in all_examples], dtype=torch.long),
            "action_target": torch.tensor([ex["action_target"] for ex in all_examples], dtype=torch.long),
            "action_valid": torch.tensor([ex["action_valid"] for ex in all_examples], dtype=torch.long),
        }

    def __len__(self) -> int:
        return int(self._examples["action_type"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "global_features": self._examples["global_features"][idx],
            "node_features": self._examples["node_features"][idx],
            "adjacency": self._examples["adjacency"][idx],
            "domain_values": self._examples["domain_values"][idx],
            "domain_mask": self._examples["domain_mask"][idx],
            "action_type": self._examples["action_type"][idx],
            "action_target": self._examples["action_target"][idx],
            "action_valid": self._examples["action_valid"][idx],
        }


if __name__ == "__main__":
    cfg = GraphColorDataConfig(num_train=2, num_val=1, num_test=1, num_nodes=12, num_colors=3, seed=0)
    ds = GraphColorStepDataset(cfg, split="train")

    ex = ds[0]
    n = int(cfg.num_nodes)
    k = int(cfg.num_colors)

    assert tuple(ex["global_features"].shape) == (5,)
    assert tuple(ex["node_features"].shape) == (n, 4)
    assert tuple(ex["adjacency"].shape) == (n, n)
    assert tuple(ex["domain_values"].shape) == (n, k)
    assert tuple(ex["domain_mask"].shape) == (n, k)
    assert ex["action_type"].ndim == 0
    assert ex["action_target"].ndim == 0
    assert ex["action_valid"].ndim == 0

    print(f"dataset.py smoke test passed (n_steps={len(ds)})")
