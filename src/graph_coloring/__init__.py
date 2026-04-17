"""Graph k-Coloring CSP environment."""

from .dataset import GraphColorDataConfig, GraphColorStepDataset
from .dsl import GraphColorAction, GraphColorActionType
from .env import GraphColorEnv, GraphColorEnvStatus, GraphColorState, StepResult
from .generator import GraphGenerator, GraphInstance
from .oracle import GraphColorOracle
from .verifier import GraphColorVerifier

__all__ = [
    "GraphColorAction",
    "GraphColorActionType",
    "GraphColorEnv",
    "GraphColorEnvStatus",
    "GraphColorState",
    "StepResult",
    "GraphColorOracle",
    "GraphColorVerifier",
    "GraphGenerator",
    "GraphInstance",
    "GraphColorDataConfig",
    "GraphColorStepDataset",
]


if __name__ == "__main__":
    gen = GraphGenerator(num_nodes=20, num_colors=3, seed=42)
    instance = gen.generate_planted()

    env = GraphColorEnv(instance.adjacency, instance.num_colors, instance.solution)
    oracle = GraphColorOracle(env)

    trace = oracle.solve()
    state = env.get_state()

    assert state.status == GraphColorEnvStatus.SUCCESS
    print(f"Solved in {len(trace)} steps")
