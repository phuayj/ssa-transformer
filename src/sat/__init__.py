"""3-SAT environment for neuro-symbolic backtracking learning."""

from .dataset import SatDataConfig, SatStepDataset
from .dense_viability_net import DenseViabilityNet, SharedMLP
from .dsl import SatAction, SatActionType
from .env import DecisionFrame, SatEnv, SatEnvStatus, SatState, StepResult
from .generator import SatGenerator, SatInstance
from .model import SatModel, SatModelConfig
from .oracle import SatOracle
from .verifier import SatVerifier

__all__ = [
    "SatAction",
    "SatActionType",
    "SatEnv",
    "SatEnvStatus",
    "SatState",
    "DecisionFrame",
    "StepResult",
    "SatOracle",
    "SatVerifier",
    "SatGenerator",
    "SatInstance",
    "SatDataConfig",
    "SatStepDataset",
    "DenseViabilityNet",
    "SharedMLP",
    "SatModel",
    "SatModelConfig",
]


if __name__ == "__main__":
    gen = SatGenerator(seed=42)
    inst = gen.generate_planted(num_vars=30, alpha=3.5)

    env = SatEnv(
        clauses=inst.clauses,
        num_vars=inst.num_vars,
        planted_solution=inst.planted_solution,
        mode="strict",
    )
    oracle = SatOracle(env)

    trace = oracle.solve()
    state = env.get_state()

    assert state.status == SatEnvStatus.SUCCESS
    print(f"Solved in {len(trace)} steps")
