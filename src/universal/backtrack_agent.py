"""Backtracking agent for 3-way comparison experiments."""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .decoder_only import PointerHead


logger = logging.getLogger(__name__)


class BacktrackTokenType(IntEnum):
    BOS = 0
    GLOBAL = 1
    VAR = 2
    SEP = 3
    ACT_ASSIGN = 4
    CONFLICT = 5
    ACT_BACKTRACK = 6
    STATE = 7
    DECIDE = 8
    RESTART = 9


def preprocess_sample(
    data: dict,
    mode: str,
    max_nodes: int,
    max_colors: int,
    max_steps: int,
) -> dict:
    mode = str(mode)
    max_nodes = int(max_nodes)
    max_colors = int(max_colors)
    max_steps = int(max_steps)
    device = torch.device("cpu")

    adjacency = data["adjacency"]
    if torch.is_tensor(adjacency):
        adjacency = adjacency.detach().cpu().numpy()
    adj = np.array(adjacency, copy=False).astype(bool)
    n_vars = int(adj.shape[0])
    if n_vars > max_nodes:
        raise ValueError(f"n_vars={n_vars} exceeds max_nodes={max_nodes}")

    num_colors = int(data["num_colors"])
    if num_colors > max_colors:
        raise ValueError(f"num_colors={num_colors} exceeds max_colors={max_colors}")

    current_assignment = data["current_assignment"]
    if torch.is_tensor(current_assignment):
        current_assignment = current_assignment.detach().cpu().numpy()
    assign = np.array(current_assignment, copy=False).astype(np.int64)

    event_trace = list(data.get("event_trace", []))
    nogoods = data.get("nogoods", {})

    if mode == "history" and len(event_trace) > max_steps:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "backtrack_agent.trace capped sample=%d trace=%d cap=%d",
                int(data.get("sample_idx", -1)),
                int(len(event_trace)),
                int(max_steps),
            )
        event_trace = event_trace[-max_steps:]

    trace_len = int(len(event_trace)) if mode == "history" else 0
    struct_len = 2 + n_vars + 1
    total_len = struct_len + trace_len + 1 if mode == "history" else struct_len + 2

    token_types = torch.zeros(total_len, dtype=torch.long, device=device)
    token_types[0] = BacktrackTokenType.BOS
    token_types[1] = BacktrackTokenType.GLOBAL
    token_types[2 : 2 + n_vars] = BacktrackTokenType.VAR
    token_types[2 + n_vars] = BacktrackTokenType.SEP

    binding_events: List[Tuple[int, int]] = []
    restart_count = 0
    if mode == "history":
        for t, event in enumerate(event_trace):
            event_type = str(event["type"])
            pos = struct_len + t
            if event_type == "assign":
                token_types[pos] = BacktrackTokenType.ACT_ASSIGN
                binding_events.append((pos, int(event["var"])))
            elif event_type == "backtrack":
                token_types[pos] = BacktrackTokenType.ACT_BACKTRACK
                binding_events.append((pos, int(event["var"])))
            elif event_type == "conflict":
                token_types[pos] = BacktrackTokenType.CONFLICT
            elif event_type == "restart":
                token_types[pos] = BacktrackTokenType.RESTART
                restart_count += 1
            else:
                raise ValueError(f"Unknown event type: {event_type!r}")
        if logger.isEnabledFor(logging.DEBUG) and restart_count > 0:
            logger.debug(
                "backtrack_agent.restart_events sample=%d count=%d trace_len=%d",
                int(data.get("sample_idx", -1)),
                int(restart_count),
                int(trace_len),
            )
        token_types[struct_len + trace_len] = BacktrackTokenType.DECIDE
    else:
        token_types[struct_len] = BacktrackTokenType.STATE
        token_types[struct_len + 1] = BacktrackTokenType.DECIDE

    pos_ids = torch.zeros(total_len, dtype=torch.long, device=device)
    pos_ids[0] = 0
    pos_ids[1] = 0
    pos_ids[2 : 2 + n_vars] = 1
    pos_ids[2 + n_vars] = 2
    if mode == "history":
        for t in range(trace_len + 1):
            pos_ids[struct_len + t] = 3 + t
    else:
        pos_ids[struct_len] = 3
        pos_ids[struct_len + 1] = 4
    max_position_id = 3 + max_steps if mode == "history" else 4
    pos_ids = pos_ids.clamp(0, max_position_id)

    base_feature_dim = 4
    if mode == "stateless_nogoods":
        var_feature_dim = base_feature_dim + 2 * max_colors
    else:
        var_feature_dim = base_feature_dim + max_colors

    features = np.zeros((n_vars, var_feature_dim), dtype=np.float32)
    degrees = adj.sum(axis=1).astype(np.float32)
    denom = max(float(n_vars), 1.0)
    color_denom = max(float(num_colors), 1.0)
    max_color = min(int(num_colors), max_colors)

    for var_id in range(n_vars):
        assigned = int(assign[var_id]) != 0
        features[var_id, 0] = 1.0 if assigned else 0.0

        neighbors = np.where(adj[var_id])[0]
        neighbor_colors = {
            int(assign[j]) for j in neighbors if int(assign[j]) != 0
        }
        if not assigned:
            available = set(range(1, int(num_colors) + 1)) - neighbor_colors
            features[var_id, 1] = float(len(available)) / color_denom
        else:
            available = set()
        features[var_id, 2] = float(len(neighbor_colors)) / color_denom
        features[var_id, 3] = degrees[var_id] / denom

        if max_color > 0:
            for color in available:
                color_id = int(color)
                if 1 <= color_id <= max_colors:
                    features[var_id, 4 + (color_id - 1)] = 1.0

    if mode == "stateless_nogoods":
        depth = int(np.count_nonzero(assign)) + 1
        depth_nogoods = nogoods.get(depth, {})
        for var_id in range(n_vars):
            nogood_colors = depth_nogoods.get(var_id, set())
            for color in nogood_colors:
                color_id = int(color)
                if 1 <= color_id <= max_colors:
                    features[var_id, 4 + max_colors + (color_id - 1)] = 1.0

    var_features = torch.tensor(features, dtype=torch.float32, device=device)

    event_feats = None
    if mode == "history":
        if trace_len > 0:
            event_feats_list = []
            for t, event in enumerate(event_trace):
                event_type = str(event["type"])
                step_val = float(event.get("step", t))
                step_norm = step_val / max(float(max_steps), 1.0)
                if event_type in {"assign", "backtrack"}:
                    var_id = int(event["var"])
                    color = int(event["color"])
                    if var_id < 0 or var_id >= n_vars:
                        raise ValueError(
                            f"Invalid var_id {var_id} for n_vars={n_vars}"
                        )
                    if color < 1 or color > num_colors:
                        raise ValueError(
                            f"Invalid color {color} for num_colors={num_colors}"
                        )
                    event_feats_list.append(
                        [
                            float(var_id) / max(float(n_vars), 1.0),
                            float(color) / max(float(num_colors), 1.0),
                            step_norm,
                        ]
                    )
                elif event_type == "conflict":
                    event_feats_list.append([0.0, 0.0, step_norm])
                elif event_type == "restart":
                    event_feats_list.append([0.0, 0.0, step_norm])
                else:
                    raise ValueError(f"Unknown event type: {event_type!r}")

            event_feats = torch.tensor(
                event_feats_list, dtype=torch.float32, device=device
            )
        else:
            event_feats = torch.zeros((0, 3), dtype=torch.float32, device=device)

    attention_mask = torch.full((total_len, total_len), float("-inf"), device=device)

    # STRUCT region: graph-sparse across BOS/GLOBAL/VAR/SEP
    attention_mask[:, 0] = 0.0
    attention_mask[:struct_len, 1] = 0.0
    attention_mask[1, :struct_len] = 0.0

    if n_vars > 0:
        var_start = 2
        var_positions = torch.arange(var_start, var_start + n_vars, device=device)
        attention_mask[var_positions, var_positions] = 0.0

        adj_t = torch.as_tensor(adj, dtype=torch.bool, device=device)
        if adj_t.numel() > 0:
            row_idx, col_idx = torch.nonzero(adj_t, as_tuple=True)
            if row_idx.numel() > 0:
                attention_mask[var_start + row_idx, var_start + col_idx] = 0.0

    sep_pos = 2 + n_vars
    attention_mask[sep_pos, :struct_len] = 0.0

    if mode == "history":
        for t in range(trace_len):
            pos = struct_len + t
            attention_mask[pos, :struct_len] = 0.0
            attention_mask[pos, struct_len : pos + 1] = 0.0
        decide_pos = struct_len + trace_len
        attention_mask[decide_pos, : decide_pos + 1] = 0.0
    else:
        state_pos = struct_len
        decide_pos = struct_len + 1
        attention_mask[state_pos, : state_pos + 1] = 0.0
        attention_mask[decide_pos, : decide_pos + 1] = 0.0

    valid_mask = torch.zeros(n_vars, dtype=torch.bool, device=device)
    for i in range(n_vars):
        if int(assign[i]) == 0:
            valid_mask[i] = True

    assigned_vars: List[Tuple[int, int]] = []
    if mode != "history":
        assigned_vars = [
            (int(i), int(assign[i])) for i in range(n_vars) if int(assign[i]) != 0
        ]

    return {
        "token_types": token_types,
        "pos_ids": pos_ids,
        "var_features": var_features,
        "event_feats": event_feats,
        "attention_mask": attention_mask,
        "binding_events": binding_events,
        "valid_mask": valid_mask,
        "assigned_vars": assigned_vars,
        "n_vars": n_vars,
        "struct_len": struct_len,
        "trace_len": trace_len,
        "total_len": total_len,
        "num_colors": num_colors,
        "selected_var": data.get("selected_var"),
        "adj": adj,
        "assign": assign,
    }


class BacktrackAgent(nn.Module):
    """Transformer agent for backtracking trace prediction."""

    def __init__(
        self,
        mode: str,
        max_nodes: int = 50,
        max_colors: int = 4,
        max_steps: int = 200,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if mode not in {"history", "stateless_nogoods", "stateless"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if int(max_nodes) <= 0:
            raise ValueError("max_nodes must be positive")
        if int(max_colors) <= 0:
            raise ValueError("max_colors must be positive")
        if int(d_model) % int(n_heads) != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.mode = str(mode)
        self.max_nodes = int(max_nodes)
        self.max_colors = int(max_colors)
        self.max_steps = int(max_steps)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)

        base_feature_dim = 4
        if self.mode == "stateless_nogoods":
            self.var_feature_dim = base_feature_dim + 2 * self.max_colors
        else:
            self.var_feature_dim = base_feature_dim + self.max_colors

        if self.mode == "history":
            self.max_position_id = 3 + self.max_steps
            self.max_seq_len = self.max_nodes + self.max_steps + 4
        else:
            self.max_position_id = 4
            self.max_seq_len = self.max_nodes + 5

        self.token_type_embed = nn.Embedding(len(BacktrackTokenType), self.d_model)
        self.position_embed = nn.Embedding(self.max_position_id + 1, self.d_model)

        self.var_proj = nn.Linear(self.var_feature_dim, self.d_model)
        self.event_proj = nn.Linear(3, self.d_model)
        self.var_one_hot_embed = nn.Embedding(self.max_nodes, self.d_model)
        self.color_one_hot_embed = nn.Embedding(self.max_colors, self.d_model)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_heads,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model)

        self.pointer_head = PointerHead(self.d_model)
        self.color_head = nn.Linear(self.d_model, self.max_colors)

        if self.mode == "history":
            self.binding_gate = nn.Linear(self.d_model, 1)
            self.binding_proj = nn.Linear(self.d_model, self.d_model)
            nn.init.normal_(self.binding_gate.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.binding_gate.bias, -2.0)
            nn.init.normal_(self.binding_proj.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.binding_proj.bias)
        else:
            self.binding_gate = None
            self.binding_proj = None

    def _compute_current_domain(
        self,
        adjacency: np.ndarray,
        assignment: np.ndarray,
        num_colors: int,
        var_id: int,
    ) -> set:
        if int(assignment[var_id]) != 0:
            return set()
        neighbors = np.where(adjacency[var_id])[0]
        used = {int(assignment[j]) for j in neighbors if int(assignment[j]) != 0}
        return set(range(1, int(num_colors) + 1)) - used

    def forward(self, batch_tensors: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_tensors: dict of pre-padded tensors from the collate_fn
        Returns:
            var_logits [B, max_nodes], color_logits [B, max_colors]
        """
        device = next(self.parameters()).device

        token_types = batch_tensors["token_types"].to(device)
        pos_ids = batch_tensors["pos_ids"].to(device)
        var_features = batch_tensors["var_features"].to(device)
        attn_mask = batch_tensors["attn_mask"].to(device)
        event_feats = batch_tensors.get("event_feats")
        if event_feats is not None:
            event_feats = event_feats.to(device)
        meta_list = batch_tensors["meta"]
        seq_lens = batch_tensors.get("seq_lens")
        n_vars_list = batch_tensors.get("n_vars_list")

        batch_size = token_types.shape[0]

        embeds = self.token_type_embed(token_types) + self.position_embed(pos_ids)

        max_nv = (
            int(n_vars_list.max().item())
            if n_vars_list is not None
            else max(int(meta["n_vars"]) for meta in meta_list)
        )
        if max_nv > 0:
            var_proj_out = self.var_proj(var_features[:, :max_nv])
            for b in range(batch_size):
                nv = int(meta_list[b]["n_vars"])
                if nv > 0:
                    embeds[b, 2 : 2 + nv] = embeds[b, 2 : 2 + nv] + var_proj_out[
                        b, :nv
                    ]

        if logger.isEnabledFor(logging.DEBUG) and batch_size > 0:
            n_vars = int(meta_list[0]["n_vars"])
            if n_vars > 0:
                var_feats = batch_tensors["var_features"][0, :n_vars]
                assigned_ratio = float(var_feats[:, 0].mean().item())
                domain_mean = float(var_feats[:, 1].mean().item())
                saturation_mean = float(var_feats[:, 2].mean().item())
                degree_mean = float(var_feats[:, 3].mean().item())
            else:
                assigned_ratio = 0.0
                domain_mean = 0.0
                saturation_mean = 0.0
                degree_mean = 0.0
            adj = meta_list[0]["adj"]
            edge_denom = max(float(n_vars * n_vars), 1.0)
            adj_density = float(adj.sum()) / edge_denom
            logger.debug(
                "backtrack_agent.var_features n_vars=%d assigned_ratio=%.3f domain_mean=%.3f saturation_mean=%.3f degree_mean=%.3f adj_density=%.3f",
                int(n_vars),
                assigned_ratio,
                domain_mean,
                saturation_mean,
                degree_mean,
                adj_density,
            )

        if self.mode == "history" and event_feats is not None:
            for b in range(batch_size):
                trace_len = int(meta_list[b]["trace_len"])
                struct_len = int(meta_list[b]["struct_len"])
                if trace_len > 0:
                    proj = self.event_proj(event_feats[b, :trace_len])
                    embeds[b, struct_len : struct_len + trace_len] = (
                        embeds[b, struct_len : struct_len + trace_len] + proj
                    )

        if self.mode != "history":
            for b in range(batch_size):
                assigned_vars = meta_list[b].get("assigned_vars", [])
                state_pos = int(meta_list[b]["struct_len"])
                if assigned_vars:
                    state_embed = embeds[b, state_pos].clone()
                    for var_id, color in assigned_vars:
                        if var_id < 0 or var_id >= self.max_nodes:
                            raise ValueError(
                                f"Invalid var_id {var_id} for max_nodes={self.max_nodes}"
                            )
                        if color < 1 or color > self.max_colors:
                            raise ValueError(
                                f"Invalid color {color} for max_colors={self.max_colors}"
                            )
                        state_embed = (
                            state_embed
                            + self.var_one_hot_embed(
                                torch.tensor(var_id, device=device)
                            )
                            + self.color_one_hot_embed(
                                torch.tensor(color - 1, device=device)
                            )
                        )
                    state_embed = state_embed / float(len(assigned_vars))
                    embeds[b, state_pos] = state_embed

        attn_mask_expanded = attn_mask.repeat_interleave(self.n_heads, dim=0)

        if logger.isEnabledFor(logging.DEBUG) and seq_lens is not None:
            min_len = int(seq_lens.min().item())
            max_len = int(seq_lens.max().item())
            mean_len = float(seq_lens.float().mean().item())
            logger.debug(
                "backtrack_agent.batch size=%d min_len=%d max_len=%d mean_len=%.2f",
                int(batch_size),
                min_len,
                max_len,
                mean_len,
            )

        x = embeds
        for layer in self.layers:
            x = layer(x, src_mask=attn_mask_expanded)
        x = self.final_norm(x)

        var_logits_list: List[torch.Tensor] = []
        color_logits_list: List[torch.Tensor] = []

        for b in range(batch_size):
            meta = meta_list[b]
            n_vars = int(meta["n_vars"])
            struct_len = int(meta["struct_len"])
            trace_len = int(meta["trace_len"])
            binding_events = list(meta.get("binding_events", []))
            assign = meta["assign"]
            adj = meta["adj"]
            num_colors = int(meta["num_colors"])
            selected_var = meta.get("selected_var")
            valid_mask = meta["valid_mask"].to(device)

            sample_x = x[b : b + 1]

            if self.mode == "history" and trace_len > 0:
                if self.binding_gate is None or self.binding_proj is None:
                    raise RuntimeError("Binding modules missing in history mode")
                binding_delta = torch.zeros_like(sample_x)
                for act_pos, var_id in binding_events:
                    var_pos = 2 + int(var_id)
                    gate = torch.sigmoid(
                        self.binding_gate(sample_x[0, act_pos])
                    ).squeeze(-1)
                    binding_delta[0, var_pos] = binding_delta[0, var_pos] + gate * (
                        self.binding_proj(sample_x[0, act_pos])
                    )
                sample_x = sample_x + binding_delta
                if logger.isEnabledFor(logging.DEBUG) and b == 0:
                    logger.debug(
                        "backtrack_agent.binding_delta trace=%d delta_norm=%.6f",
                        int(trace_len),
                        float(binding_delta.detach().norm().item()),
                    )

            decide_pos = (
                struct_len + trace_len if self.mode == "history" else struct_len + 1
            )
            query_hidden = sample_x[0, decide_pos]
            var_hidden = sample_x[0, 2 : 2 + n_vars]

            if int(valid_mask.sum().item()) == 0:
                raise RuntimeError("No valid variables to select")

            var_logits = self.pointer_head(
                query_hidden.unsqueeze(0),
                var_hidden.unsqueeze(0),
                valid_mask.unsqueeze(0),
            )

            padded_var_logits = torch.full(
                (1, self.max_nodes), float("-inf"), device=device
            )
            padded_var_logits[0, :n_vars] = var_logits[0]

            color_logits = self.color_head(query_hidden)
            if selected_var is None:
                selected_var = int(torch.argmax(var_logits[0]).item())
            selected_var = int(selected_var)
            if selected_var < 0 or selected_var >= n_vars:
                raise ValueError(
                    f"selected_var={selected_var} out of range for n_vars={n_vars}"
                )
            domain = self._compute_current_domain(adj, assign, num_colors, selected_var)
            if not domain:
                raise RuntimeError(
                    f"Empty domain for var={selected_var} with num_colors={num_colors}"
                )
            valid_color_mask = torch.zeros(
                self.max_colors, dtype=torch.bool, device=device
            )
            for c in domain:
                if 1 <= int(c) <= self.max_colors:
                    valid_color_mask[int(c) - 1] = True
            color_logits = color_logits.masked_fill(~valid_color_mask, float("-inf"))

            if logger.isEnabledFor(logging.DEBUG) and b == 0:
                logger.debug(
                    "backtrack_agent.sample vars=%d trace=%d valid_vars=%d domain=%d",
                    int(n_vars),
                    int(trace_len),
                    int(valid_mask.sum().item()),
                    int(len(domain)),
                )

            var_logits_list.append(padded_var_logits)
            color_logits_list.append(color_logits.unsqueeze(0))

        return torch.cat(var_logits_list, dim=0), torch.cat(color_logits_list, dim=0)
