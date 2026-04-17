"""Test unified CSP interface."""

import numpy as np


def test_csp_wrapper_basic():
    from csp.env import CSPEnv
    from universal.types import UnifiedAction
    from universal.wrapper import CSPUnifiedWrapper

    puzzle = np.zeros((4, 4), dtype=np.int64)
    puzzle[0, 0] = 1
    puzzle[0, 1] = 2

    env = CSPEnv(puzzle, mode="soft", propagation_mode="forward_check")
    wrapper = CSPUnifiedWrapper(env, max_domain=4, propagation_mode="forward_check")

    obs = wrapper.reset()

    assert obs.num_vars == 16
    assert obs.var_domain_mask.shape == (16, 4)
    assert obs.var_assigned[0] == 0  # Cell (0,0) = 1, 0-indexed
    assert obs.var_assigned[1] == 1  # Cell (0,1) = 2, 0-indexed
    assert obs.var_assigned[2] == -1  # Unassigned

    # Get valid actions
    actions = wrapper.get_valid_actions()
    assert len(actions) > 0

    # Try an assignment
    assign_actions = [a for a in actions if a.type.value == 0]
    if assign_actions:
        obs2, reward, done, info = wrapper.step(assign_actions[0])
        del reward, done, info
        assert obs2.num_vars == 16


def test_graph_coloring_wrapper_basic():
    from graph_coloring.env import GraphColorEnv
    from graph_coloring.generator import GraphGenerator
    from universal.wrapper import GraphColoringUnifiedWrapper

    gen = GraphGenerator(seed=42)
    instance = gen.generate(num_nodes=10, num_colors=3, edge_prob=0.3)

    env = GraphColorEnv(
        adjacency=instance.adjacency,
        num_colors=instance.num_colors,
        mode="soft",
    )
    wrapper = GraphColoringUnifiedWrapper(
        env, max_domain=4, propagation_mode="forward_check"
    )

    obs = wrapper.reset()

    assert obs.num_vars == 10
    assert obs.var_domain_mask.shape == (10, 4)
    assert obs.domain_id == 1

    actions = wrapper.get_valid_actions()
    assert len(actions) > 0


def test_sat_wrapper_basic():
    from sat.env import SatEnv
    from sat.generator import SatGenerator
    from universal.wrapper import SATUnifiedWrapper

    gen = SatGenerator(seed=42)
    instance = gen.generate_planted(num_vars=10, alpha=3.0)

    env = SatEnv(
        clauses=instance.clauses,
        num_vars=instance.num_vars,
        planted_solution=instance.planted_solution,
        mode="soft",
    )
    wrapper = SATUnifiedWrapper(env, max_domain=2, propagation_mode="forward_check")

    obs = wrapper.reset()

    assert obs.num_vars == 10
    assert obs.var_domain_mask.shape == (10, 2)
    assert obs.domain_id == 2
    assert obs.max_domain == 2

    actions = wrapper.get_valid_actions()
    assert len(actions) > 0


def test_unified_obs_shapes():
    """Verify all wrappers produce consistent observation shapes."""

    from csp.env import CSPEnv
    from graph_coloring.env import GraphColorEnv
    from graph_coloring.generator import GraphGenerator
    from sat.env import SatEnv
    from sat.generator import SatGenerator
    from universal.wrapper import (
        CSPUnifiedWrapper,
        GraphColoringUnifiedWrapper,
        SATUnifiedWrapper,
    )

    max_domain = 9

    # CSP
    csp_env = CSPEnv(
        np.zeros((4, 4), dtype=np.int64),
        mode="soft",
        propagation_mode="forward_check",
    )
    csp_wrapper = CSPUnifiedWrapper(
        csp_env, max_domain=max_domain, propagation_mode="forward_check"
    )
    csp_obs = csp_wrapper.reset()

    # Coloring
    gc_gen = GraphGenerator(seed=42)
    gc_inst = gc_gen.generate(num_nodes=10, num_colors=4, edge_prob=0.3)
    gc_env = GraphColorEnv(
        adjacency=gc_inst.adjacency, num_colors=gc_inst.num_colors, mode="soft"
    )
    gc_wrapper = GraphColoringUnifiedWrapper(
        gc_env, max_domain=max_domain, propagation_mode="forward_check"
    )
    gc_obs = gc_wrapper.reset()

    # SAT
    sat_gen = SatGenerator(seed=42)
    sat_inst = sat_gen.generate_planted(num_vars=10, alpha=3.0)
    sat_env = SatEnv(
        clauses=sat_inst.clauses,
        num_vars=sat_inst.num_vars,
        planted_solution=sat_inst.planted_solution,
        mode="soft",
    )
    sat_wrapper = SATUnifiedWrapper(
        sat_env, max_domain=max_domain, propagation_mode="forward_check"
    )
    sat_obs = sat_wrapper.reset()

    # Check domain_mask shape consistent with max_domain
    assert csp_obs.var_domain_mask.shape[1] == max_domain
    assert gc_obs.var_domain_mask.shape[1] == max_domain
    assert sat_obs.var_domain_mask.shape[1] == max_domain

    # Check nogood_mask shape
    assert csp_obs.var_nogood_mask.shape[1] == max_domain
    assert gc_obs.var_nogood_mask.shape[1] == max_domain
    assert sat_obs.var_nogood_mask.shape[1] == max_domain


def test_factor_gnn_forward():
    """Test FactorGNN forward pass."""

    import torch

    from csp.env import CSPEnv
    from universal.model import FactorGNN, create_model_inputs_from_obs
    from universal.wrapper import CSPUnifiedWrapper

    # Create a simple CSP instance
    puzzle = np.zeros((4, 4), dtype=np.int64)
    puzzle[0, 0] = 1

    env = CSPEnv(puzzle, mode="soft", propagation_mode="forward_check")
    wrapper = CSPUnifiedWrapper(env, max_domain=9, propagation_mode="forward_check")
    obs = wrapper.reset()

    # Create model
    model = FactorGNN(
        max_vars=100,
        max_constraints=500,
        max_domain=10,
        d_model=64,
        num_layers=2,
    )

    # Create inputs
    device = torch.device("cpu")
    inputs = create_model_inputs_from_obs(
        obs, device, max_vars=100, max_constraints=500, max_domain=10
    )

    # Forward pass
    assign_logits, bt_logit, done_logit, _attention_weights = model(**inputs)

    assert assign_logits.shape == (1, 100, 10)
    assert bt_logit.shape == (1, 1)
    assert done_logit.shape == (1, 1)

    # Check that invalid assignments have -inf.
    # Variables beyond N should have -inf.
    assert torch.isinf(assign_logits[0, 20, 0])  # Beyond N=16

    print("FactorGNN forward pass test passed!")


def test_decode_action():
    """Test action decoding."""

    import torch

    from universal.model import decode_action
    from universal.types import UnifiedActionType

    # Test ASSIGN action
    assign_logits = torch.full((1, 10, 5), -float("inf"))
    assign_logits[0, 3, 2] = 10.0  # Best: var=3, value=2
    bt_logit = torch.tensor([[5.0]])
    done_logit = torch.tensor([[2.0]])
    var_mask = torch.ones(1, 10, dtype=torch.bool)

    action = decode_action(assign_logits, bt_logit, done_logit, var_mask)
    assert action.type == UnifiedActionType.ASSIGN
    assert action.var == 3
    assert action.value == 2

    # Test BACKTRACK action
    assign_logits = torch.full((1, 10, 5), -float("inf"))
    bt_logit = torch.tensor([[10.0]])
    done_logit = torch.tensor([[2.0]])

    action = decode_action(assign_logits, bt_logit, done_logit, var_mask)
    assert action.type == UnifiedActionType.BACKTRACK

    print("decode_action test passed!")


def test_multi_domain_dataset():
    """Test multi-domain dataset generation."""

    from universal.dataset import (
        MultiDomainDataConfig,
        MultiDomainStepDataset,
        collate_multi_domain,
    )

    config = MultiDomainDataConfig(
        csp_num_instances=10,
        gc_num_instances=10,
        sat_num_instances=10,
        max_domain=10,
        seed=42,
    )

    dataset = MultiDomainStepDataset(
        config, split="train", domains=["csp", "gc", "sat"]
    )

    print(f"Dataset size: {len(dataset)} steps")
    assert len(dataset) > 0, "Dataset should have steps"

    # Test single item
    item = dataset[0]
    assert "var_features" in item
    assert "action_type" in item
    assert item["action_type"] in [0, 1, 2]
    assert "propagation_mode" in item
    assert item["propagation_mode"] in [0, 1]

    # Test collation
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    collated = collate_multi_domain(
        batch, max_vars=100, max_constraints=500, max_domain=10
    )

    assert collated["var_features"].shape[0] == len(batch)
    assert collated["var_features"].shape[1] == 100  # max_vars
    assert collated["global_features"].shape[1] == 4

    print(f"Test multi_domain_dataset passed! ({len(dataset)} steps)")


def test_multi_domain_single_domain():
    """Test dataset with single domain (for zero-shot experiments)."""

    from universal.dataset import MultiDomainDataConfig, MultiDomainStepDataset

    config = MultiDomainDataConfig(
        csp_num_instances=10,
        gc_num_instances=10,
        sat_num_instances=10,
        seed=42,
    )

    # Only CSP
    csp_only = MultiDomainStepDataset(config, split="train", domains=["csp"])
    assert len(csp_only) > 0

    # Check all steps are CSP
    for i in range(min(10, len(csp_only))):
        assert csp_only[i]["domain_id"] == 0

    print(f"Single domain test passed! (CSP: {len(csp_only)} steps)")


if __name__ == "__main__":
    test_csp_wrapper_basic()
    test_graph_coloring_wrapper_basic()
    test_sat_wrapper_basic()
    test_unified_obs_shapes()
    test_factor_gnn_forward()
    test_decode_action()
    test_multi_domain_dataset()
    test_multi_domain_single_domain()
    print("All tests passed!")
