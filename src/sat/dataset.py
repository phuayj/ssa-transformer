from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .dsl import SatActionType
from .env import SatEnv, SatEnvStatus, SatState
from .generator import SatGenerator, SatInstance
from .oracle import SatOracle
from .verifier import SatVerifier

logger = logging.getLogger(__name__)


@dataclass
class SatDataConfig:
    num_train: int = 3000
    num_val: int = 300
    num_test: int = 0

    num_vars: int = 30
    alpha_sat: float = 3.5  # Alpha for planted SAT
    alpha_unsat: float = 5.5  # Alpha for random UNSAT-ish
    sat_ratio: float = 0.5  # Ratio of SAT instances

    max_steps: int = 1000
    seed: int = 42

    activity_bins: int = 16
    activity_clip: float = 10.0


def _split_seed(seed: int, split: str) -> int:
    if split == "train":
        return int(seed) + 0
    if split == "val":
        return int(seed) + 1_000_000
    if split == "test":
        return int(seed) + 2_000_000
    raise ValueError(f"Unknown split: {split!r}")


def _num_instances_for_split(config: SatDataConfig, split: str) -> int:
    if split == "train":
        return int(config.num_train)
    if split == "val":
        return int(config.num_val)
    if split == "test":
        return int(config.num_test)
    raise ValueError(f"Unknown split: {split!r}")


class SatStepDataset(Dataset):
    """Step dataset for 3-SAT.

    Each item is one (observation, action) step from an oracle trajectory.

    Note: clauses/clause_features are padded to `max_clauses` so a batch can
    be stacked. Use `clause_mask` to ignore padding.

    Returned dict:
      global_features: (5,) long
          [selected_var (-1 if none), num_assigned, conflict (0/1), propagation_pending (0/1), stack_depth]

      var_features: (n, 5) long
          For each var: [var_idx, assigned_value_idx (0=unassigned,1=false,2=true), is_selected, domain_size, activity_bin]

      var_domain_mask: (n, 2) bool
          For each var: [allow_false, allow_true]

      clauses: (max_clauses, 3) long
          Literal encoding ±(var+1)

      clause_features: (max_clauses, 5) long
          For each clause: [clause_idx, satisfied, num_unassigned, num_true, is_conflict]

      clause_mask: (max_clauses,) bool
          True for real clauses, False for padding

      action_type: scalar long in [0,4]
      action_target: scalar long
          SELECT_VAR: var index
          ASSIGN_VALUE: 0/1 (0=False,1=True)
          else: 0

      action_valid: scalar long (0/1)
    """

    def __init__(self, config: SatDataConfig, split: str = "train"):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split!r}")
        if int(config.num_vars) < 3:
            raise ValueError("num_vars must be >= 3 for 3-SAT")
        if float(config.alpha_sat) <= 0.0:
            raise ValueError("alpha_sat must be > 0")
        if float(config.alpha_unsat) <= 0.0:
            raise ValueError("alpha_unsat must be > 0")
        if float(config.sat_ratio) < 0.0 or float(config.sat_ratio) > 1.0:
            raise ValueError("sat_ratio must be in [0,1]")
        if int(config.max_steps) < 1:
            raise ValueError("max_steps must be >= 1")
        if int(config.activity_bins) < 2:
            raise ValueError("activity_bins must be >= 2")
        if float(config.activity_clip) <= 0.0:
            raise ValueError("activity_clip must be > 0")

        self.config = config
        self.split = split

        self.max_clauses = max(
            1,
            int(float(config.alpha_sat) * float(config.num_vars)),
            int(float(config.alpha_unsat) * float(config.num_vars)),
        )

        self._verifier = SatVerifier()
        self._examples = self._generate_examples()

        if not self._examples:
            raise RuntimeError("Generated empty SatStepDataset")

    def _extract_example(self, obs: dict, state: SatState, action) -> dict:
        n = int(state.num_vars)
        m = int(state.num_clauses)

        global_features = [int(x) for x in obs["global"]]
        if len(global_features) != 5:
            raise RuntimeError(f"Expected 5 global features; got {len(global_features)}")

        var_features = obs["vars"]
        if len(var_features) != n:
            raise RuntimeError(f"Expected {n} vars; got {len(var_features)}")

        var_domain_mask = obs["var_domain_mask"]
        if len(var_domain_mask) != n or any(len(row) != 2 for row in var_domain_mask):
            raise RuntimeError("var_domain_mask has wrong shape")

        clauses = obs["clauses"]
        if len(clauses) != m or any(len(row) != 3 for row in clauses):
            raise RuntimeError("clauses has wrong shape")

        clause_features = obs["clause_features"]
        if len(clause_features) != m:
            raise RuntimeError(f"Expected {m} clause_features; got {len(clause_features)}")

        if int(m) > int(self.max_clauses):
            raise RuntimeError(f"Instance has m={m} clauses but max_clauses={self.max_clauses}")

        clauses_rows = [[int(x) for x in row] for row in clauses]
        clause_feat_rows = [[int(x) for x in row] for row in clause_features]

        # Pad to max_clauses so tensors stack.
        clause_mask = [True for _ in range(m)] + [False for _ in range(int(self.max_clauses) - int(m))]
        if m < int(self.max_clauses):
            pad_clause = clauses_rows[-1] if clauses_rows else [1, 2, 3]
            pad_feat = [0, 1, 3, 0, 0]
            for _ in range(int(self.max_clauses) - int(m)):
                clauses_rows.append([int(pad_clause[0]), int(pad_clause[1]), int(pad_clause[2])])
                clause_feat_rows.append([int(x) for x in pad_feat])

        # --- Targets ---
        if action.type == SatActionType.SELECT_VAR:
            if action.target is None:
                raise RuntimeError("SELECT_VAR missing target")
            action_type = int(SatActionType.SELECT_VAR)
            action_target = int(action.target)
        elif action.type == SatActionType.ASSIGN_VALUE:
            if action.target is None:
                raise RuntimeError("ASSIGN_VALUE missing target")
            action_type = int(SatActionType.ASSIGN_VALUE)
            action_target = int(action.target)
        elif action.type == SatActionType.PROPAGATE:
            action_type = int(SatActionType.PROPAGATE)
            action_target = 0
        elif action.type == SatActionType.BACKTRACK:
            action_type = int(SatActionType.BACKTRACK)
            action_target = 0
        elif action.type == SatActionType.DONE:
            action_type = int(SatActionType.DONE)
            action_target = 0
        else:
            raise RuntimeError(f"Unknown action type: {action.type}")

        ok, reason = self._verifier.is_valid(state, action)
        if not ok:
            raise RuntimeError(
                "Oracle produced invalid action. "
                f"action={action.to_token()} reason={reason!r} "
                f"global={global_features}"
            )

        return {
            "global_features": global_features,
            "var_features": [[int(x) for x in row] for row in var_features],
            "var_domain_mask": [[bool(x) for x in row] for row in var_domain_mask],
            "clauses": clauses_rows,
            "clause_features": clause_feat_rows,
            "clause_mask": [bool(x) for x in clause_mask],
            "action_type": int(action_type),
            "action_target": int(action_target),
            "action_valid": int(ok),
        }

    def _generate_instance_examples(self, instance: SatInstance) -> tuple[List[dict], SatState]:
        env = SatEnv(
            clauses=instance.clauses,
            num_vars=int(instance.num_vars),
            planted_solution=instance.planted_solution,
            mode="strict",
            max_steps=int(self.config.max_steps),
            activity_bins=int(self.config.activity_bins),
            activity_clip=float(self.config.activity_clip),
        )
        oracle = SatOracle(env)

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

        return examples, env.get_state()

    def _generate_examples(self) -> Dict[str, torch.Tensor]:
        seed = _split_seed(self.config.seed, self.split)
        n_instances = _num_instances_for_split(self.config, self.split)

        gen = SatGenerator(seed=int(seed))

        all_examples: List[dict] = []

        attempts = 0
        accepted = 0
        sat_count = 0
        unsat_count = 0
        skipped = 0
        total_steps = 0

        # Target instance-level mix (rounded to nearest int).
        target_sat = int(round(float(self.config.sat_ratio) * float(n_instances)))
        target_sat = max(0, min(int(n_instances), int(target_sat)))
        target_unsat = int(n_instances) - int(target_sat)

        max_attempts = max(100, int(n_instances) * 50)

        while accepted < int(n_instances):
            attempts += 1
            if attempts > int(max_attempts):
                raise RuntimeError(
                    "Failed to generate enough SAT/UNSAT instances within max_attempts. "
                    f"split={self.split!r} requested={n_instances} accepted={accepted} "
                    f"target_sat={target_sat} target_unsat={target_unsat} "
                    f"attempts={attempts} sat={sat_count} unsat={unsat_count} skipped={skipped}"
                )

            # Force generation towards remaining quota.
            if sat_count >= int(target_sat):
                sat_ratio = 0.0
            elif unsat_count >= int(target_unsat):
                sat_ratio = 1.0
            else:
                sat_ratio = float(self.config.sat_ratio)

            inst = gen.generate(
                num_vars=int(self.config.num_vars),
                alpha_sat=float(self.config.alpha_sat),
                alpha_unsat=float(self.config.alpha_unsat),
                sat_ratio=float(sat_ratio),
            )

            examples, final_state = self._generate_instance_examples(inst)

            term = final_state.termination_reason
            is_sat = final_state.status == SatEnvStatus.SUCCESS or term == "sat"
            is_unsat = term == "unsat"

            if inst.planted_solution is not None:
                # Expected SAT.
                if not is_sat:
                    skipped += 1
                    continue
                if sat_count >= int(target_sat):
                    skipped += 1
                    continue
                sat_count += 1
            else:
                # Expected UNSAT.
                if not is_unsat:
                    skipped += 1
                    continue
                if unsat_count >= int(target_unsat):
                    skipped += 1
                    continue
                unsat_count += 1

            accepted += 1
            total_steps += int(len(examples))
            all_examples.extend(examples)

        if int(n_instances) > 0:
            logger.info(
                "SatStepDataset(split=%s): instances=%d attempts=%d sat=%d/%d unsat=%d/%d skipped=%d avg_steps=%.1f max_clauses=%d",
                self.split,
                accepted,
                attempts,
                sat_count,
                target_sat,
                unsat_count,
                target_unsat,
                skipped,
                float(total_steps) / float(max(1, accepted)),
                int(self.max_clauses),
            )

        if not all_examples:
            return {}

        return {
            "global_features": torch.tensor([ex["global_features"] for ex in all_examples], dtype=torch.long),
            "var_features": torch.tensor([ex["var_features"] for ex in all_examples], dtype=torch.long),
            "var_domain_mask": torch.tensor([ex["var_domain_mask"] for ex in all_examples], dtype=torch.bool),
            "clauses": torch.tensor([ex["clauses"] for ex in all_examples], dtype=torch.long),
            "clause_features": torch.tensor([ex["clause_features"] for ex in all_examples], dtype=torch.long),
            "clause_mask": torch.tensor([ex["clause_mask"] for ex in all_examples], dtype=torch.bool),
            "action_type": torch.tensor([ex["action_type"] for ex in all_examples], dtype=torch.long),
            "action_target": torch.tensor([ex["action_target"] for ex in all_examples], dtype=torch.long),
            "action_valid": torch.tensor([ex["action_valid"] for ex in all_examples], dtype=torch.long),
        }

    def __len__(self) -> int:
        return int(self._examples["action_type"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "global_features": self._examples["global_features"][idx],
            "var_features": self._examples["var_features"][idx],
            "var_domain_mask": self._examples["var_domain_mask"][idx],
            "clauses": self._examples["clauses"][idx],
            "clause_features": self._examples["clause_features"][idx],
            "clause_mask": self._examples["clause_mask"][idx],
            "action_type": self._examples["action_type"][idx],
            "action_target": self._examples["action_target"][idx],
            "action_valid": self._examples["action_valid"][idx],
        }


if __name__ == "__main__":
    cfg = SatDataConfig(
        num_train=2,
        num_val=1,
        num_test=1,
        num_vars=20,
        alpha_sat=3.0,
        alpha_unsat=5.0,
        sat_ratio=1.0,
        seed=0,
        max_steps=2000,
    )
    ds = SatStepDataset(cfg, split="train")

    ex = ds[0]
    n = int(cfg.num_vars)
    m = max(int(float(cfg.alpha_sat) * float(cfg.num_vars)), int(float(cfg.alpha_unsat) * float(cfg.num_vars)))

    assert tuple(ex["global_features"].shape) == (5,)
    assert tuple(ex["var_features"].shape) == (n, 5)
    assert tuple(ex["var_domain_mask"].shape) == (n, 2)
    assert tuple(ex["clauses"].shape) == (m, 3)
    assert tuple(ex["clause_features"].shape) == (m, 5)
    assert tuple(ex["clause_mask"].shape) == (m,)
    assert ex["action_type"].ndim == 0
    assert ex["action_target"].ndim == 0
    assert ex["action_valid"].ndim == 0

    print(f"dataset.py smoke test passed (n_steps={len(ds)})")
