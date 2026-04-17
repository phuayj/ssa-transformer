"""Autoregressive CSP agent for graph coloring assignments."""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .decoder_only import PointerHead, TransformerBlock


logger = logging.getLogger(__name__)


class AgentTokenType(IntEnum):
    BOS = 0
    GLOBAL = 1
    VAR = 2
    CON = 3
    SEP = 4
    ACT_ASSIGN = 5
    STATE = 6
    DECIDE = 7


class AutoregressiveCSPAgent(nn.Module):
    """Minimal autoregressive agent for graph coloring assignment sequences."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_nodes: int = 100,
        max_colors: int = 10,
        max_edges: int = 500,
        max_steps: int = 200,
        var_feature_dim: int = 9,
        mode: str = "history",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in {"history", "stateless"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if int(var_feature_dim) not in (5, 9):
            raise ValueError(
                f"Unsupported var_feature_dim={var_feature_dim}; expected 5 or 9"
            )

        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_nodes = int(max_nodes)
        self.max_colors = int(max_colors)
        self.max_edges = int(max_edges)
        self.max_steps = int(max_steps)
        self.var_feature_dim = int(var_feature_dim)
        self.mode = str(mode)

        # STRUCT: BOS + GLOBAL + max_nodes + max_edges + SEP
        self.max_struct_len = 3 + self.max_nodes + self.max_edges
        if self.mode == "history":
            self.max_seq_len = self.max_struct_len + self.max_steps + 1
            self.max_position_id = 4 + self.max_steps
        else:
            self.max_seq_len = self.max_struct_len + 2
            self.max_position_id = 5

        # Embeddings
        self.token_type_embed = nn.Embedding(len(AgentTokenType), self.d_model)
        self.position_embed = nn.Embedding(self.max_position_id + 1, self.d_model)

        # Feature projections
        self.var_proj = nn.Linear(self.var_feature_dim, self.d_model)
        self.con_proj = nn.Linear(3, self.d_model)
        self.action_proj = nn.Linear(3, self.d_model)

        # Stateless bag-of-assignments embeddings
        self.var_one_hot_embed = nn.Embedding(self.max_nodes, self.d_model)
        self.color_one_hot_embed = nn.Embedding(self.max_colors, self.d_model)

        # Transformer
        self.layers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, self.n_heads, dropout)
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model)

        # Heads
        self.pointer_head = PointerHead(self.d_model)
        self.color_head = nn.Linear(self.d_model, self.max_colors)

        # Grounding parameters (history mode)
        self.binding_gate = nn.Linear(self.d_model, 1)
        self.binding_proj = nn.Linear(self.d_model, self.d_model)
        nn.init.normal_(self.binding_gate.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.binding_gate.bias, -2.0)
        nn.init.normal_(self.binding_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.binding_proj.bias)

    def _compute_var_features(
        self,
        adj: np.ndarray,
        assign: np.ndarray,
        domains: List[set],
        num_colors: int,
        nogoods: dict,
        n_vars: int,
    ) -> np.ndarray:
        """Compute VAR features (copied from DecoderOnlyCSPBatched)."""
        feature_dim = self.var_feature_dim
        features = np.zeros((n_vars, feature_dim), dtype=np.float32)
        degrees = adj.sum(axis=1)
        max_degree = max(float(degrees.max()), 1.0)
        color_norm = max(num_colors, 1)
        assign_arr = np.asarray(assign)

        for i in range(n_vars):
            features[i, 0] = float(assign_arr[i] != 0)
            dom_size = len(domains[i]) if assign_arr[i] == 0 else 1
            features[i, 1] = dom_size / color_norm
            neighbors = np.where(adj[i])[0]
            neighbor_colors = set(
                assign_arr[j] for j in neighbors if assign_arr[j] != 0
            )
            features[i, 2] = len(neighbor_colors) / color_norm
            features[i, 3] = degrees[i] / max_degree
            bt_count = sum(len(nogoods.get(d, {}).get(i, set())) for d in nogoods)
            features[i, 4] = min(bt_count / 10.0, 1.0)

            if feature_dim == 9:
                unassigned_neighbors = neighbors[assign_arr[neighbors] == 0]
                unassigned_count = int(unassigned_neighbors.size)
                features[i, 5] = unassigned_count / max_degree
                if unassigned_count == 0:
                    features[i, 6] = 0.0
                    features[i, 7] = 1.0
                    features[i, 8] = 0.0
                else:
                    neighbor_domains = np.array(
                        [len(domains[j]) for j in unassigned_neighbors],
                        dtype=np.float32,
                    )
                    features[i, 6] = float(neighbor_domains.mean()) / color_norm
                    features[i, 7] = float(neighbor_domains.min()) / color_norm
                    features[i, 8] = float(np.mean(neighbor_domains <= 2))

        return features

    def _compute_con_features(
        self,
        adj: np.ndarray,
        assign: np.ndarray,
        domains: List[set],
        edges: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Compute CON features (copied from DecoderOnlyCSPBatched)."""
        m = len(edges)
        features = np.zeros((m, 3), dtype=np.float32)
        for idx, (i, j) in enumerate(edges):
            assigned_count = (assign[i] != 0) + (assign[j] != 0)
            features[idx, 0] = assigned_count / 2.0
            if assigned_count == 2:
                features[idx, 1] = float(assign[i] != assign[j])
            else:
                features[idx, 1] = 1.0
            dom_i = len(domains[i]) if assign[i] == 0 else 1
            dom_j = len(domains[j]) if assign[j] == 0 else 1
            features[idx, 2] = 1.0 / max(dom_i * dom_j, 1)
        return features

    def _build_attention_mask(
        self,
        n_vars: int,
        n_cons: int,
        edges: List[Tuple[int, int]],
        trace_len: int,
        struct_len: int,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for a single sample."""
        mask = torch.full((total_len, total_len), float("-inf"), device=device)

        BOS_POS = 0
        GLOBAL_POS = 1
        VAR_START = 2
        VAR_END = 2 + n_vars
        CON_START = VAR_END
        CON_END = CON_START + n_cons
        SEP_POS = CON_END

        # Everyone attends to BOS
        mask[:, BOS_POS] = 0.0

        # GLOBAL attends to all STRUCT, all STRUCT attends to GLOBAL
        mask[GLOBAL_POS, :struct_len] = 0.0
        mask[:struct_len, GLOBAL_POS] = 0.0

        # Self-attention
        for i in range(total_len):
            mask[i, i] = 0.0

        # VAR-CON connectivity
        for con_idx, (i, j) in enumerate(edges):
            var_i_pos = VAR_START + i
            var_j_pos = VAR_START + j
            con_pos = CON_START + con_idx
            mask[var_i_pos, con_pos] = 0.0
            mask[var_j_pos, con_pos] = 0.0
            mask[con_pos, var_i_pos] = 0.0
            mask[con_pos, var_j_pos] = 0.0

        # SEP attends to all STRUCT
        mask[SEP_POS, :struct_len] = 0.0

        if self.mode == "history":
            # TRACE: causal + attend to all STRUCT
            for t in range(trace_len + 1):  # +1 for DECIDE
                pos = struct_len + t
                mask[pos, :struct_len] = 0.0
                mask[pos, struct_len : pos + 1] = 0.0
        else:
            state_pos = struct_len
            decide_pos = struct_len + 1
            mask[state_pos, : struct_len + 1] = 0.0
            mask[decide_pos, : struct_len + 1] = 0.0

        return mask

    def _compute_current_domains(
        self,
        adjacency: np.ndarray,
        assignment: np.ndarray,
        num_colors: int,
        base_domains: List[set] | None = None,
    ) -> List[set]:
        """Compute current domains from assignment and adjacency."""
        n_vars = int(adjacency.shape[0])
        if base_domains is None:
            base_domains = [set(range(1, int(num_colors) + 1)) for _ in range(n_vars)]

        domains: List[set] = []
        for i in range(n_vars):
            if int(assignment[i]) != 0:
                domains.append({int(assignment[i])})
                continue
            neighbors = np.where(adjacency[i])[0]
            used = {int(assignment[j]) for j in neighbors if int(assignment[j]) != 0}
            dom = set(base_domains[i]) - used
            domains.append(dom)
        return domains

    def forward(self, batch_data: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_data: List of dicts with keys:
                - adjacency: np.ndarray (n, n)
                - num_colors: int
                - domains: List[Set[int]]  # initial domains
                - action_history: List[Tuple[int, int]]
                - current_assignment: np.ndarray (n,)
                - selected_var: Optional[int]  # optional override for color masking

        Returns:
            var_logits: [B, max_nodes]
            color_logits: [B, max_colors]
        """
        device = next(self.parameters()).device
        var_logits_list: List[torch.Tensor] = []
        color_logits_list: List[torch.Tensor] = []
        embeds_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        meta_list: List[dict] = []
        total_lens: List[int] = []

        for b, data in enumerate(batch_data):
            adjacency = data["adjacency"]
            if torch.is_tensor(adjacency):
                adjacency = adjacency.detach().cpu().numpy()
            adj = np.array(adjacency, copy=False).astype(bool)
            num_colors = int(data["num_colors"])
            action_history = list(data.get("action_history", []))
            current_assignment = data["current_assignment"]
            if torch.is_tensor(current_assignment):
                current_assignment = current_assignment.detach().cpu().numpy()
            assign = np.array(current_assignment, copy=False).astype(np.int64)

            n_vars = int(adj.shape[0])
            if n_vars > self.max_nodes:
                raise ValueError(f"n_vars={n_vars} exceeds max_nodes={self.max_nodes}")

            # Build edges (upper triangle)
            edges: List[Tuple[int, int]] = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if bool(adj[i, j]):
                        edges.append((int(i), int(j)))
            if len(edges) > self.max_edges:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "ar_agent.edges capped sample=%d edges=%d cap=%d",
                        b,
                        int(len(edges)),
                        int(self.max_edges),
                    )
                edges = edges[: self.max_edges]
            n_cons = int(len(edges))

            # Truncate action history only for history mode
            if self.mode == "history" and len(action_history) > self.max_steps:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "ar_agent.trace capped sample=%d trace=%d cap=%d",
                        b,
                        int(len(action_history)),
                        int(self.max_steps),
                    )
                action_history = action_history[-self.max_steps :]

            trace_len = int(len(action_history)) if self.mode == "history" else 0

            # STRUCT: [BOS, GLOBAL, VAR*n, CON*m, SEP]
            struct_len = 2 + n_vars + n_cons + 1
            total_len = (
                struct_len + trace_len + 1 if self.mode == "history" else struct_len + 2
            )

            # Token types
            token_types = torch.zeros(total_len, dtype=torch.long, device=device)
            token_types[0] = AgentTokenType.BOS
            token_types[1] = AgentTokenType.GLOBAL
            token_types[2 : 2 + n_vars] = AgentTokenType.VAR
            token_types[2 + n_vars : 2 + n_vars + n_cons] = AgentTokenType.CON
            token_types[2 + n_vars + n_cons] = AgentTokenType.SEP

            if self.mode == "history":
                for t in range(trace_len):
                    token_types[struct_len + t] = AgentTokenType.ACT_ASSIGN
                token_types[struct_len + trace_len] = AgentTokenType.DECIDE
            else:
                token_types[struct_len] = AgentTokenType.STATE
                token_types[struct_len + 1] = AgentTokenType.DECIDE

            # Position IDs (collapsed for STRUCT)
            pos_ids = torch.zeros(total_len, dtype=torch.long, device=device)
            pos_ids[0] = 0
            pos_ids[1] = 0
            pos_ids[2 : 2 + n_vars] = 1
            pos_ids[2 + n_vars : 2 + n_vars + n_cons] = 2
            pos_ids[2 + n_vars + n_cons] = 3
            if self.mode == "history":
                for t in range(trace_len + 1):
                    pos_ids[struct_len + t] = 4 + t
            else:
                pos_ids[struct_len] = 4
                pos_ids[struct_len + 1] = 5
            pos_ids = pos_ids.clamp(0, self.max_position_id)

            embeds = self.token_type_embed(token_types)
            embeds = embeds + self.position_embed(pos_ids)

            # Initial state features only (no leakage from current assignment)
            init_assign = np.zeros((n_vars,), dtype=np.int64)
            init_domains = [set(range(1, num_colors + 1)) for _ in range(n_vars)]
            var_feats = self._compute_var_features(
                adj, init_assign, init_domains, num_colors, {}, n_vars
            )
            var_feats_t = torch.tensor(var_feats, dtype=torch.float32, device=device)
            embeds[2 : 2 + n_vars] = embeds[2 : 2 + n_vars] + self.var_proj(var_feats_t)

            if n_cons > 0:
                con_feats = self._compute_con_features(
                    adj, init_assign, init_domains, edges
                )
                con_feats_t = torch.tensor(
                    con_feats, dtype=torch.float32, device=device
                )
                embeds[2 + n_vars : 2 + n_vars + n_cons] = embeds[
                    2 + n_vars : 2 + n_vars + n_cons
                ] + self.con_proj(con_feats_t)

            if self.mode == "history" and trace_len > 0:
                action_feats = []
                for t, (var_id, color) in enumerate(action_history):
                    if int(var_id) < 0 or int(var_id) >= n_vars:
                        raise ValueError(f"Invalid var_id {var_id} for n_vars={n_vars}")
                    if int(color) < 1 or int(color) > num_colors:
                        raise ValueError(
                            f"Invalid color {color} for num_colors={num_colors}"
                        )
                    action_feats.append(
                        [
                            float(var_id) / max(float(n_vars), 1.0),
                            float(color) / max(float(num_colors), 1.0),
                            float(t + 1) / max(float(self.max_steps), 1.0),
                        ]
                    )
                action_feats_t = torch.tensor(
                    action_feats, dtype=torch.float32, device=device
                )
                action_embeds = self.action_proj(action_feats_t)
                embeds[struct_len : struct_len + trace_len] = (
                    embeds[struct_len : struct_len + trace_len] + action_embeds
                )

            if self.mode == "stateless":
                assigned_vars = list(action_history)
                state_pos = struct_len
                state_embed = embeds[state_pos]
                if len(assigned_vars) > 0:
                    for var_id, color in assigned_vars:
                        if int(var_id) < 0 or int(var_id) >= self.max_nodes:
                            raise ValueError(
                                f"Invalid var_id {var_id} for max_nodes={self.max_nodes}"
                            )
                        if int(color) < 1 or int(color) > self.max_colors:
                            raise ValueError(
                                f"Invalid color {color} for max_colors={self.max_colors}"
                            )
                        state_embed = (
                            state_embed
                            + self.var_one_hot_embed(
                                torch.tensor(int(var_id), device=device)
                            )
                            + self.color_one_hot_embed(
                                torch.tensor(int(color) - 1, device=device)
                            )
                        )
                    state_embed = state_embed / float(len(assigned_vars))
                embeds[state_pos] = state_embed

            attention_mask = self._build_attention_mask(
                n_vars,
                n_cons,
                edges,
                trace_len,
                struct_len,
                total_len,
                device,
            )

            embeds_list.append(embeds)
            mask_list.append(attention_mask)
            total_lens.append(total_len)
            meta_list.append(
                {
                    "n_vars": n_vars,
                    "n_cons": n_cons,
                    "struct_len": struct_len,
                    "trace_len": trace_len,
                    "action_history": action_history,
                    "assign": assign,
                    "adj": adj,
                    "num_colors": num_colors,
                    "selected_var": data.get("selected_var"),
                    "domains": data.get("domains"),
                }
            )

        batch_size = len(batch_data)
        max_total_len = max(total_lens)
        embed_batch = torch.zeros(
            (batch_size, max_total_len, self.d_model), device=device
        )
        mask_batch = torch.full(
            (batch_size, 1, max_total_len, max_total_len),
            float("-inf"),
            device=device,
        )
        for i in range(batch_size):
            seq_len = total_lens[i]
            embed_batch[i, :seq_len] = embeds_list[i]
            mask_batch[i, 0, :seq_len, :seq_len] = mask_list[i]
            if seq_len < max_total_len:
                pad_idx = torch.arange(seq_len, max_total_len, device=device)
                mask_batch[i, 0, pad_idx, pad_idx] = 0.0

        if logger.isEnabledFor(logging.DEBUG):
            min_len = int(min(total_lens))
            max_len = int(max_total_len)
            mean_len = float(sum(total_lens)) / float(batch_size)
            logger.debug(
                "ar_agent.batch size=%d min_len=%d max_len=%d mean_len=%.2f",
                int(batch_size),
                min_len,
                max_len,
                mean_len,
            )

        x = embed_batch
        for layer in self.layers:
            x = layer(x, mask_batch)
        x = self.final_norm(x)

        for b, meta in enumerate(meta_list):
            n_vars = meta["n_vars"]
            n_cons = meta["n_cons"]
            struct_len = meta["struct_len"]
            trace_len = meta["trace_len"]
            action_history = meta["action_history"]
            assign = meta["assign"]
            adj = meta["adj"]
            num_colors = meta["num_colors"]
            selected_var = meta["selected_var"]
            base_domains = meta["domains"]

            sample_x = x[b : b + 1]

            if self.mode == "history" and trace_len > 0:
                binding_delta = torch.zeros_like(sample_x)
                for act_idx, (var_id, _color) in enumerate(action_history):
                    act_pos = struct_len + act_idx
                    var_pos = 2 + int(var_id)
                    gate = torch.sigmoid(
                        self.binding_gate(sample_x[0, act_pos])
                    ).squeeze(-1)
                    binding_delta[0, var_pos] = binding_delta[0, var_pos] + gate * self.binding_proj(
                        sample_x[0, act_pos]
                    )
                sample_x = sample_x + binding_delta
                if logger.isEnabledFor(logging.DEBUG) and b == 0:
                    logger.debug(
                        "ar_agent.binding_delta trace=%d delta_norm=%.6f",
                        int(trace_len),
                        float(binding_delta.detach().norm().item()),
                    )

            decide_pos = (
                struct_len + trace_len if self.mode == "history" else struct_len + 1
            )
            query_hidden = sample_x[0, decide_pos]
            var_hidden = sample_x[0, 2 : 2 + n_vars]

            # Valid variable mask: unassigned only
            valid_mask = torch.zeros(n_vars, dtype=torch.bool, device=device)
            for i in range(n_vars):
                if int(assign[i]) == 0:
                    valid_mask[i] = True

            var_logits = self.pointer_head(
                query_hidden.unsqueeze(0),
                var_hidden.unsqueeze(0),
                valid_mask.unsqueeze(0),
            )

            padded_var_logits = torch.full(
                (1, self.max_nodes), float("-inf"), device=device
            )
            padded_var_logits[0, :n_vars] = var_logits[0]

            # Color logits with masking based on selected variable's domain
            color_logits = self.color_head(query_hidden)
            if selected_var is None:
                selected_var = int(torch.argmax(var_logits[0]).item())
            selected_var = int(selected_var)
            current_domains = self._compute_current_domains(
                adj,
                assign,
                num_colors,
                base_domains=base_domains,
            )
            if selected_var < 0 or selected_var >= len(current_domains):
                raise ValueError(
                    f"selected_var={selected_var} out of range for n_vars={n_vars}"
                )
            domain = current_domains[selected_var]
            valid_color_mask = torch.zeros(
                self.max_colors, dtype=torch.bool, device=device
            )
            for c in domain:
                if 1 <= int(c) <= self.max_colors:
                    valid_color_mask[int(c) - 1] = True
            color_logits = color_logits.masked_fill(~valid_color_mask, float("-inf"))

            if logger.isEnabledFor(logging.DEBUG) and b == 0:
                logger.debug(
                    "ar_agent.sample vars=%d cons=%d trace=%d valid_vars=%d domain=%d",
                    int(n_vars),
                    int(n_cons),
                    int(trace_len),
                    int(valid_mask.sum().item()),
                    int(len(domain)),
                )

            var_logits_list.append(padded_var_logits)
            color_logits_list.append(color_logits.unsqueeze(0))

        return torch.cat(var_logits_list, dim=0), torch.cat(color_logits_list, dim=0)
