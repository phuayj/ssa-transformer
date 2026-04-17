from __future__ import annotations

import logging

import torch

from backtrack.eval import Evaluator
from backtrack.model import ActionPredictor, TracePredictor
from backtrack.dataset import DataConfig, StepDataset, TrajectoryDataset
from backtrack.tokenizer import Tokenizer
from backtrack.tree import Tree, TreeNode
from backtrack.training import ClosedLoopTrainer, OpenLoopTrainer, TrainConfig


def _make_tiny_goal_tree() -> Tree:
    # 0 -> [1, 2]
    # 1 -> [3]
    # 2 -> []
    # 3 -> [] (goal)
    n0 = TreeNode(id=0)
    n1 = TreeNode(id=1, parent=n0)
    n2 = TreeNode(id=2, parent=n0)
    n3 = TreeNode(id=3, parent=n1, is_goal=True)
    n0.children = [n1, n2]
    n1.children = [n3]

    nodes = [n0, n1, n2, n3]
    return Tree(root=n0, nodes=nodes, goal_node=n3)


def test_tokenizer_roundtrip() -> None:
    tok = Tokenizer(max_node_id=10, max_degree=8)

    for a in ["GOTO_0", "GOTO_7", "BACKTRACK", "DONE", "FAIL"]:
        tid = tok.encode_action(a)
        assert tok.decode_action(tid) == a

    obs = {"node_id": 3, "degree": 2, "goal": 0, "visited_children": [0, 1]}
    obs_ids = tok.encode_observation(obs)
    assert all(isinstance(x, int) for x in obs_ids)

    step = tok.encode_step(obs, "GOTO_0")
    assert step[0] == tok.bos_id
    assert step[-2] == tok.sep_id
    assert tok.decode_action(step[-1]) == "GOTO_0"


def test_step_dataset_generation() -> None:
    cfg = DataConfig(
        min_nodes=4,
        max_nodes=6,
        max_depth=3,
        max_branching=3,
        num_train=5,
        num_val=2,
        num_test=2,
        seed=0,
    )

    ds = StepDataset(cfg, split="train", observation_mode="E1")
    assert len(ds) > 0

    item = ds[0]
    assert set(item.keys()) == {"obs_tokens", "action_id"}
    assert item["obs_tokens"].dtype == torch.long
    assert item["action_id"].dtype == torch.long

    tok = ds.tokenizer
    assert int(item["obs_tokens"][0].item()) == tok.bos_id

    # Last non-pad token should be <sep>.
    non_pad = (item["obs_tokens"] != tok.pad_id).nonzero(as_tuple=False)
    assert non_pad.numel() > 0
    last_idx = int(non_pad[-1].item())
    assert int(item["obs_tokens"][last_idx].item()) == tok.sep_id

    assert tok.is_action_token_id(int(item["action_id"].item()))


def test_trajectory_dataset_generation() -> None:
    cfg = DataConfig(
        min_nodes=4,
        max_nodes=6,
        max_depth=3,
        max_branching=3,
        num_train=3,
        num_val=2,
        num_test=2,
        seed=123,
    )
    tok = Tokenizer(max_node_id=cfg.max_nodes, max_degree=8)
    ds = TrajectoryDataset(cfg, split="train", observation_mode="E1", tokenizer=tok, max_seq_len=64)

    ex = ds[0]
    assert set(ex.keys()) == {"input_ids", "labels", "attention_mask"}
    assert ex["input_ids"].shape == (64,)
    assert ex["labels"].shape == (64,)
    assert ex["attention_mask"].shape == (64,)
    assert ex["input_ids"].dtype == torch.long
    assert ex["labels"].dtype == torch.long
    assert ex["attention_mask"].dtype == torch.long

    assert int(ex["input_ids"][0].item()) == tok.bos_id

    pad_mask = ex["attention_mask"] == 0
    if int(pad_mask.sum().item()) > 0:
        assert torch.all(ex["labels"][pad_mask] == -100)


def test_action_predictor_forward() -> None:
    tok = Tokenizer(max_node_id=20, max_degree=8)
    model = ActionPredictor(
        vocab_size=tok.vocab_size,
        num_actions=11,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        max_seq_len=64,
    )

    obs_tokens = torch.randint(0, tok.vocab_size, (2, 12), dtype=torch.long)
    attn = torch.ones((2, 12), dtype=torch.long)

    logits = model(obs_tokens, attention_mask=attn)
    assert logits.shape == (2, 11)


def test_trace_predictor_forward() -> None:
    tok = Tokenizer(max_node_id=20, max_degree=8)
    model = TracePredictor(
        vocab_size=tok.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        max_seq_len=64,
    )

    input_ids = torch.randint(0, tok.vocab_size, (2, 16), dtype=torch.long)
    attn = torch.ones((2, 16), dtype=torch.long)

    logits = model(input_ids, attention_mask=attn)
    assert logits.shape == (2, 16, tok.vocab_size)


def test_closed_loop_trainer_single_step() -> None:
    logging.basicConfig(level=logging.INFO)

    cfg = DataConfig(
        min_nodes=4,
        max_nodes=6,
        max_depth=3,
        max_branching=3,
        num_train=4,
        num_val=2,
        num_test=2,
        seed=7,
    )
    tok = Tokenizer(max_node_id=cfg.max_nodes, max_degree=8)

    train_ds = StepDataset(cfg, split="train", observation_mode="E1", tokenizer=tok)
    val_ds = StepDataset(cfg, split="val", observation_mode="E1", tokenizer=tok)

    seq_len = int(train_ds[0]["obs_tokens"].shape[0])
    model = ActionPredictor(
        vocab_size=tok.vocab_size,
        num_actions=11,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_seq_len=seq_len,
        dropout=0.0,
    )

    tcfg = TrainConfig(
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        warmup_steps=0,
        log_interval=0,
    )

    trainer = ClosedLoopTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=tcfg,
        tokenizer=tok,
        device="cpu",
    )

    metrics = trainer.train_epoch()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0


def test_open_loop_trainer_single_step() -> None:
    cfg = DataConfig(
        min_nodes=4,
        max_nodes=6,
        max_depth=3,
        max_branching=3,
        num_train=3,
        num_val=2,
        num_test=2,
        seed=9,
    )
    tok = Tokenizer(max_node_id=cfg.max_nodes, max_degree=8)

    train_ds = TrajectoryDataset(cfg, split="train", observation_mode="E1", tokenizer=tok, max_seq_len=64)
    val_ds = TrajectoryDataset(cfg, split="val", observation_mode="E1", tokenizer=tok, max_seq_len=64)

    model = TracePredictor(
        vocab_size=tok.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        max_seq_len=64,
        dropout=0.0,
    )

    tcfg = TrainConfig(
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        warmup_steps=0,
        log_interval=0,
    )

    trainer = OpenLoopTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=tcfg,
        tokenizer=tok,
        device="cpu",
    )

    metrics = trainer.train_epoch()
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    assert metrics["loss"] > 0.0


def test_evaluator_closed_loop() -> None:
    tree = _make_tiny_goal_tree()
    tok = Tokenizer(max_node_id=10, max_degree=8)
    model = ActionPredictor(
        vocab_size=tok.vocab_size,
        num_actions=11,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_seq_len=64,
        dropout=0.0,
    )

    evaluator = Evaluator(
        tokenizer=tok,
        observation_mode="E1",
        env_mode="strict",
        max_steps=50,
        device="cpu",
    )

    metrics = evaluator.evaluate_closed_loop(model, [tree])
    assert set(metrics.keys()) == {
        "success_rate",
        "valid_rate",
        "mean_steps",
        "mean_backtracks",
        "mean_efficiency",
    }
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert 0.0 <= metrics["valid_rate"] <= 1.0
    assert metrics["mean_steps"] >= 0.0
