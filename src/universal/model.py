"""Factor-GNN model for universal backtracking policy."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageLayer(nn.Module):
    """Single message passing layer."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)

        # NOTE: num_heads is kept for future attention-style messages.
        self.num_heads = int(num_heads)

        # Combine node state + edge features -> message.
        self.message_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.d_model),
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, h_node: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """Compute messages.

        Args:
            h_node: [B, E, H] node hidden state at each edge endpoint.
            e_feat: [B, E, H] embedded edge features.

        Returns:
            messages: [B, E, H]
        """

        msg_input = torch.cat([h_node, e_feat], dim=-1)
        msg = self.message_mlp(msg_input)
        return self.norm(msg + h_node)


class FactorGNN(nn.Module):
    """Factor Graph Neural Network for unified CSP solving.

    Architecture:
    - Variable and constraint nodes with separate embeddings
    - Bipartite message passing between variables and constraints
    - Output heads for assignment, backtrack, and done actions
    """

    def __init__(
        self,
        max_vars: int = 100,
        max_constraints: int = 500,
        max_domain: int = 10,
        num_constraint_types: int = 3,  # NEQ, ALLDIFF, CLAUSE
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        var_feature_dim: int = 3,
        con_feature_dim: int = 2,
        edge_feature_dim: int = 2,
        global_feature_dim: int = 4,
    ):
        super().__init__()

        self.max_vars = int(max_vars)
        self.max_constraints = int(max_constraints)
        self.max_domain = int(max_domain)
        self.num_constraint_types = int(num_constraint_types)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.global_feature_dim = int(global_feature_dim)

        # Variable input embedding.
        # Input:
        #   - var_features: [N, var_feature_dim]
        #   - domain_mask:  [N, max_domain]
        #   - nogood_mask:  [N, max_domain]
        #   - assigned:     [N, max_domain] one-hot (all zeros if unassigned)
        var_input_dim = int(var_feature_dim) + (self.max_domain * 3)
        self.var_embed = nn.Sequential(
            nn.Linear(var_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Constraint input embedding.
        # Input:
        #   - con_features: [M, con_feature_dim]
        #   - con_type:     [M, num_constraint_types] one-hot
        con_input_dim = int(con_feature_dim) + self.num_constraint_types
        self.con_embed = nn.Sequential(
            nn.Linear(con_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Edge feature embedding.
        self.edge_embed = nn.Linear(int(edge_feature_dim), self.d_model)

        # Message passing layers.
        self.var_to_con_layers = nn.ModuleList(
            [
                MessageLayer(self.d_model, num_heads, dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.con_to_var_layers = nn.ModuleList(
            [
                MessageLayer(self.d_model, num_heads, dropout)
                for _ in range(self.num_layers)
            ]
        )

        # Node update GRUs.
        self.var_gru = nn.GRUCell(self.d_model, self.d_model)
        self.con_gru = nn.GRUCell(self.d_model, self.d_model)

        # Value embedding for assignment head.
        self.value_embed = nn.Embedding(self.max_domain, self.d_model)

        # Assignment head: score for each (var, value) pair.
        self.assign_head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )

        # Global pooling for backtrack/done.
        self.global_pool = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
        )

        self.backtrack_head = nn.Linear(self.d_model, 1)
        self.done_head = nn.Linear(self.d_model, 1)

        # Global features: (stack_depth_norm, propagation_pending, has_conflict, propagation_mode)
        self.global_embed = nn.Linear(self.global_feature_dim, self.d_model)

    def forward(
        self,
        var_features: torch.Tensor,  # [B, N, F_v]
        var_domain_mask: torch.Tensor,  # [B, N, D]
        var_nogood_mask: torch.Tensor,  # [B, N, D]
        var_assigned: torch.Tensor,  # [B, N] in {-1, 0..D-1}
        con_type: torch.Tensor,  # [B, M]
        con_features: torch.Tensor,  # [B, M, F_c]
        edge_con_idx: torch.Tensor,  # [B, E]
        edge_var_idx: torch.Tensor,  # [B, E]
        edge_features: torch.Tensor,  # [B, E, F_e]
        var_mask: torch.Tensor,  # [B, N] bool
        con_mask: torch.Tensor,  # [B, M] bool
        edge_mask: torch.Tensor,  # [B, E] bool
        global_features: torch.Tensor,  # [B, G] includes propagation descriptor
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """Forward pass.

        Returns:
            assign_logits: [B, N, D] logits for ASSIGN(var, value)
            backtrack_logit: [B, 1]
            done_logit: [B, 1]
            attention_weights: [] when requested (FactorGNN has no attention layers),
                otherwise None
        """

        del con_mask  # Currently unused (padding handled by masks on edges/pooling).

        B, N, D = var_domain_mask.shape
        M = con_type.shape[1]
        E = edge_con_idx.shape[1]

        if D > self.max_domain:
            raise ValueError(
                f"Input domain size D={D} exceeds model max_domain={self.max_domain}"
            )

        device = var_features.device
        dtype = var_features.dtype

        # Encode variable assignment as one-hot over domain values.
        # Unassigned (-1) -> all zeros.
        assigned_mask = (var_assigned >= 0).to(dtype=dtype).unsqueeze(-1)  # [B, N, 1]
        var_assigned_idx = var_assigned.clamp(min=0, max=D - 1)
        var_assigned_onehot = F.one_hot(var_assigned_idx, num_classes=D).to(dtype=dtype)
        var_assigned_onehot = var_assigned_onehot * assigned_mask  # [B, N, D]

        # Variable input.
        var_input = torch.cat(
            [
                var_features,
                var_domain_mask.to(dtype=dtype),
                var_nogood_mask.to(dtype=dtype),
                var_assigned_onehot,
            ],
            dim=-1,
        )
        h_v = self.var_embed(var_input)  # [B, N, H]

        # Constraint input.
        con_type_oh = F.one_hot(
            con_type.clamp(min=0, max=self.num_constraint_types - 1),
            num_classes=self.num_constraint_types,
        ).to(dtype=dtype)
        con_input = torch.cat([con_features, con_type_oh], dim=-1)
        h_c = self.con_embed(con_input)  # [B, M, H]

        # Edge features.
        e_feat = self.edge_embed(edge_features.to(dtype=dtype))  # [B, E, H]
        edge_mask_f = edge_mask.to(dtype=dtype).unsqueeze(-1)  # [B, E, 1]

        # Message passing.
        for layer_idx in range(self.num_layers):
            # Variable -> Constraint.
            edge_var_idx_exp = edge_var_idx.unsqueeze(-1).expand(
                -1, -1, self.d_model
            )  # [B, E, H]
            h_v_at_edge = torch.gather(h_v, 1, edge_var_idx_exp)  # [B, E, H]
            m_v2c = self.var_to_con_layers[layer_idx](h_v_at_edge, e_feat) * edge_mask_f

            edge_con_idx_exp = edge_con_idx.unsqueeze(-1).expand(
                -1, -1, self.d_model
            )  # [B, E, H]
            agg_v2c = torch.zeros((B, M, self.d_model), device=device, dtype=dtype)
            agg_v2c = agg_v2c.scatter_add(1, edge_con_idx_exp, m_v2c)

            h_c = self.con_gru(
                agg_v2c.reshape(B * M, self.d_model), h_c.reshape(B * M, self.d_model)
            ).reshape(B, M, self.d_model)

            # Constraint -> Variable.
            h_c_at_edge = torch.gather(h_c, 1, edge_con_idx_exp)  # [B, E, H]
            m_c2v = self.con_to_var_layers[layer_idx](h_c_at_edge, e_feat) * edge_mask_f

            agg_c2v = torch.zeros((B, N, self.d_model), device=device, dtype=dtype)
            agg_c2v = agg_c2v.scatter_add(1, edge_var_idx_exp, m_c2v)

            h_v = self.var_gru(
                agg_c2v.reshape(B * N, self.d_model), h_v.reshape(B * N, self.d_model)
            ).reshape(B, N, self.d_model)

        # Assignment logits: score each (var, value).
        value_emb = self.value_embed.weight[:D]  # [D, H]
        value_emb = (
            value_emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        )  # [B, N, D, H]
        h_v_exp = h_v.unsqueeze(2).expand(-1, -1, D, -1)  # [B, N, D, H]
        assign_input = torch.cat([h_v_exp, value_emb], dim=-1)  # [B, N, D, 2H]
        assign_logits = self.assign_head(assign_input).squeeze(-1)  # [B, N, D]

        # Mask invalid assignments.
        valid_assign = var_domain_mask & ~var_nogood_mask
        valid_assign = valid_assign & (var_assigned == -1).unsqueeze(-1)
        valid_assign = valid_assign & var_mask.unsqueeze(-1)
        assign_logits = assign_logits.masked_fill(~valid_assign, float("-inf"))

        # Global pooling for backtrack/done.
        global_features = global_features.to(dtype=dtype)
        if int(global_features.shape[1]) != int(self.global_feature_dim):
            raise ValueError(
                "global_features must have shape (batch, "
                f"{self.global_feature_dim}); got {tuple(global_features.shape)}"
            )
        global_emb = self.global_embed(global_features)
        var_mask_f = var_mask.to(dtype=dtype).unsqueeze(-1)
        h_v_mean = (h_v * var_mask_f).sum(dim=1) / var_mask_f.sum(dim=1).clamp(min=1.0)
        global_state = self.global_pool(h_v_mean + global_emb)

        backtrack_logit = self.backtrack_head(global_state)
        done_logit = self.done_head(global_state)
        attention_weights: list[torch.Tensor] | None = [] if return_attention else None
        return assign_logits, backtrack_logit, done_logit, attention_weights


def create_model_inputs_from_obs(
    obs,  # UnifiedObservation
    device: torch.device,
    max_vars: int = 100,
    max_constraints: int = 500,
    max_domain: int = 10,
    max_edges: int = 2000,
) -> dict[str, torch.Tensor]:
    """Convert UnifiedObservation to batched model inputs (batch size 1)."""

    import numpy as np

    N = int(obs.num_vars)
    M = int(obs.num_constraints)
    D_obs = int(obs.max_domain)
    E = int(len(obs.edge_con_idx))

    if N > int(max_vars):
        raise ValueError(f"obs.num_vars={N} exceeds max_vars={max_vars}")
    if M > int(max_constraints):
        raise ValueError(
            f"obs.num_constraints={M} exceeds max_constraints={max_constraints}"
        )
    if E > int(max_edges):
        raise ValueError(f"num edges E={E} exceeds max_edges={max_edges}")

    d_fill = min(D_obs, int(max_domain))

    # Pad to max sizes.
    var_features = np.zeros(
        (1, int(max_vars), int(obs.var_features.shape[1])), dtype=np.float32
    )
    var_features[0, :N] = obs.var_features

    var_domain_mask = np.zeros((1, int(max_vars), int(max_domain)), dtype=np.bool_)
    var_domain_mask[0, :N, :d_fill] = obs.var_domain_mask[:, :d_fill]

    var_nogood_mask = np.zeros((1, int(max_vars), int(max_domain)), dtype=np.bool_)
    var_nogood_mask[0, :N, :d_fill] = obs.var_nogood_mask[:, :d_fill]

    var_assigned = np.full((1, int(max_vars)), -1, dtype=np.int64)
    var_assigned[0, :N] = obs.var_assigned

    con_type = np.zeros((1, int(max_constraints)), dtype=np.int64)
    con_type[0, :M] = obs.con_type

    con_features = np.zeros(
        (1, int(max_constraints), int(obs.con_features.shape[1])), dtype=np.float32
    )
    con_features[0, :M] = obs.con_features

    edge_con_idx = np.zeros((1, int(max_edges)), dtype=np.int64)
    edge_var_idx = np.zeros((1, int(max_edges)), dtype=np.int64)
    edge_features = np.zeros(
        (1, int(max_edges), int(obs.edge_features.shape[1])), dtype=np.float32
    )
    edge_con_idx[0, :E] = obs.edge_con_idx
    edge_var_idx[0, :E] = obs.edge_var_idx
    edge_features[0, :E] = obs.edge_features

    var_mask = np.zeros((1, int(max_vars)), dtype=np.bool_)
    var_mask[0, :N] = True

    con_mask = np.zeros((1, int(max_constraints)), dtype=np.bool_)
    con_mask[0, :M] = True

    edge_mask = np.zeros((1, int(max_edges)), dtype=np.bool_)
    edge_mask[0, :E] = True

    global_features = np.asarray(
        [
            [
                float(obs.stack_depth) / 50.0,
                float(obs.propagation_pending),
                float(obs.has_conflict),
                float(getattr(obs, "propagation_mode", 1)),
            ]
        ],
        dtype=np.float32,
    )

    return {
        "var_features": torch.from_numpy(var_features).to(device),
        "var_domain_mask": torch.from_numpy(var_domain_mask).to(device),
        "var_nogood_mask": torch.from_numpy(var_nogood_mask).to(device),
        "var_assigned": torch.from_numpy(var_assigned).to(device),
        "con_type": torch.from_numpy(con_type).to(device),
        "con_features": torch.from_numpy(con_features).to(device),
        "edge_con_idx": torch.from_numpy(edge_con_idx).to(device),
        "edge_var_idx": torch.from_numpy(edge_var_idx).to(device),
        "edge_features": torch.from_numpy(edge_features).to(device),
        "var_mask": torch.from_numpy(var_mask).to(device),
        "con_mask": torch.from_numpy(con_mask).to(device),
        "edge_mask": torch.from_numpy(edge_mask).to(device),
        "global_features": torch.from_numpy(global_features).to(device),
    }


def decode_action(
    assign_logits: torch.Tensor,  # [B, N, D]
    backtrack_logit: torch.Tensor,  # [B, 1]
    done_logit: torch.Tensor,  # [B, 1]
    var_mask: torch.Tensor,  # [B, N]
    can_backtrack: bool = True,
    can_done: bool = True,
):
    """Decode action from model outputs."""

    del var_mask  # Currently unused (assignment logits already masked).

    from .types import UnifiedAction

    B = int(assign_logits.shape[0])
    if B != 1:
        raise ValueError("decode_action currently supports batch size 1")

    # Best assignment among (var, value).
    assign_flat = assign_logits.view(B, -1)
    best_assign_score, best_assign_idx = assign_flat.max(dim=-1)  # [B]

    bt_score = (
        backtrack_logit.squeeze(-1)
        if can_backtrack
        else torch.full_like(best_assign_score, float("-inf"))
    )
    done_score = (
        done_logit.squeeze(-1)
        if can_done
        else torch.full_like(best_assign_score, float("-inf"))
    )

    all_scores = torch.stack(
        [best_assign_score, bt_score, done_score], dim=-1
    )  # [B, 3]
    action_type = int(all_scores.argmax(dim=-1).item())

    if action_type == 0:
        N = int(assign_logits.shape[1])
        D = int(assign_logits.shape[2])
        flat_idx = int(best_assign_idx.item())
        var_idx = int(flat_idx // D)
        value_idx = int(flat_idx % D)
        if not (0 <= var_idx < N and 0 <= value_idx < D):
            raise RuntimeError("Decoded invalid ASSIGN indices")
        return UnifiedAction.assign(var_idx, value_idx)

    if action_type == 1:
        return UnifiedAction.backtrack()

    return UnifiedAction.done()
