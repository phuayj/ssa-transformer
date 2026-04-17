"""Decoder-only transformer for constraint satisfaction variable selection.

This module implements a decoder-only architecture with:
- Bidirectional attention in STRUCT region (graph structure)
- Causal attention in TRACE region (action history)
- Pointer head for variable selection

Designed for permutation equivariance via collapsed position IDs.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


logger = logging.getLogger(__name__)


class TokenType(IntEnum):
    """Token types for the decoder-only CSP model."""

    BOS = 0
    GLOBAL = 1
    VAR = 2
    CON = 3
    SEP_STRUCT = 4
    ACT_ASSIGN = 5
    ACT_BACKTRACK = 6
    QUERY_SELECT = 7


@dataclass
class PackedState:
    """Packed state for model input."""

    input_embeds: torch.Tensor  # [B, L, d_model]
    attention_mask: torch.Tensor  # [B, 1, L, L] additive mask
    position_ids: torch.Tensor  # [B, L]
    var_positions: torch.Tensor  # [B, N] indices of VAR tokens in sequence
    valid_var_mask: torch.Tensor  # [B, N] bool mask for selectable vars
    struct_len: int
    trace_len: int


class AttentionMaskBuilder:
    """Build hybrid attention masks for decoder-only CSP model."""

    @staticmethod
    def build_struct_mask(
        num_vars: int,
        num_cons: int,
        edge_list: List[Tuple[int, int]],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Build bidirectional graph-sparse attention mask for STRUCT region.

        Token order: [BOS, GLOBAL, VAR_0, ..., VAR_{n-1}, CON_0, ..., CON_{m-1}, SEP_STRUCT]

        Args:
            num_vars: Number of variable tokens
            num_cons: Number of constraint tokens
            edge_list: List of (var_pos, con_pos) tuples for graph connectivity
            device: Torch device

        Returns:
            [S, S] additive mask (0 = allowed, -inf = blocked)
        """
        # S = 1(BOS) + 1(GLOBAL) + num_vars + num_cons + 1(SEP)
        S = 2 + num_vars + num_cons + 1

        # Start with all blocked
        mask = torch.full((S, S), float("-inf"), device=device)

        # Token position indices
        BOS_POS = 0
        GLOBAL_POS = 1
        VAR_START = 2
        VAR_END = 2 + num_vars
        CON_START = VAR_END
        CON_END = CON_START + num_cons
        SEP_POS = CON_END

        # BOS: attends to nothing, everyone attends to BOS
        mask[:, BOS_POS] = 0.0  # All can attend to BOS

        # GLOBAL: attends to all STRUCT, all STRUCT attends to GLOBAL
        mask[GLOBAL_POS, :S] = 0.0  # GLOBAL attends to all
        mask[:S, GLOBAL_POS] = 0.0  # All attend to GLOBAL

        # Self-attention for all tokens
        for i in range(S):
            mask[i, i] = 0.0

        # VAR-CON connectivity from edge_list
        for var_pos, con_pos in edge_list:
            # VAR attends to connected CON
            mask[var_pos, con_pos] = 0.0
            # CON attends to connected VAR
            mask[con_pos, var_pos] = 0.0

        # SEP attends to all STRUCT
        mask[SEP_POS, :S] = 0.0

        return mask

    @staticmethod
    def build_full_mask(
        struct_mask: torch.Tensor,
        trace_len: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Build full attention mask: STRUCT (bidirectional) + TRACE (causal) + QUERY.

        Args:
            struct_mask: [S, S] mask for STRUCT region
            trace_len: Number of TRACE tokens (actions)
            device: Torch device

        Returns:
            [L, L] additive mask where L = S + trace_len + 1 (query)
        """
        S = struct_mask.shape[0]
        L = S + trace_len + 1  # +1 for QUERY_SELECT

        mask = torch.full((L, L), float("-inf"), device=device)

        # Copy STRUCT mask
        mask[:S, :S] = struct_mask

        # TRACE region: causal within itself, full attention to STRUCT
        trace_start = S
        query_pos = S + trace_len

        for t in range(trace_len):
            pos = trace_start + t
            # Attend to all STRUCT
            mask[pos, :S] = 0.0
            # Attend to self and previous TRACE tokens (causal)
            mask[pos, trace_start : pos + 1] = 0.0

        # QUERY_SELECT: attends to everything before it
        mask[query_pos, : query_pos + 1] = 0.0

        return mask


class PositionIdBuilder:
    """Build collapsed position IDs for permutation equivariance."""

    @staticmethod
    def build(
        num_vars: int,
        num_cons: int,
        trace_len: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Build position IDs with collapsed positions for STRUCT.

        Position assignment:
        - BOS: 0
        - GLOBAL: 0
        - All VAR tokens: 1 (same for equivariance!)
        - All CON tokens: 2 (same for equivariance!)
        - SEP_STRUCT: 3
        - TRACE tokens: 4, 5, 6, ...
        - QUERY_SELECT: 4 + trace_len

        Returns:
            [L] position IDs
        """
        positions = []

        # BOS and GLOBAL
        positions.extend([0, 0])

        # All VARs get position 1
        positions.extend([1] * num_vars)

        # All CONs get position 2
        positions.extend([2] * num_cons)

        # SEP gets position 3
        positions.append(3)

        # TRACE tokens get increasing positions starting at 4
        for t in range(trace_len):
            positions.append(4 + t)

        # QUERY_SELECT
        positions.append(4 + trace_len)

        return torch.tensor(positions, dtype=torch.long, device=device)


class PointerHead(nn.Module):
    """Pointer network head for variable selection."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = d_model**-0.5

    def forward(
        self,
        query_hidden: torch.Tensor,  # [B, d_model]
        var_hidden: torch.Tensor,  # [B, N, d_model]
        valid_var_mask: torch.Tensor,  # [B, N] bool
    ) -> torch.Tensor:
        """Compute pointer logits over VAR tokens."""
        q = self.query_proj(query_hidden)  # [B, d_model]
        k = self.key_proj(var_hidden)  # [B, N, d_model]

        # Scaled dot-product: [B, N]
        logits = torch.einsum("bd,bnd->bn", q, k) * self.scale

        # Mask invalid variables
        logits = logits.masked_fill(~valid_var_mask, float("-inf"))

        return logits


class TransformerBlock(nn.Module):
    """Single transformer block with custom attention mask support."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, d_model]
        attention_mask: torch.Tensor,  # [B, 1, L, L] additive mask
    ) -> torch.Tensor:
        B, L, _ = x.shape

        # Pre-norm
        normed = self.norm1(x)

        # Multi-head attention
        q = self.q_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with mask
        # attention_mask: [B, 1, L, L] -> broadcast to [B, n_heads, L, L]
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.o_proj(attn_out)

        # Residual
        x = x + self.dropout(attn_out)

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))

        return x


class GraphToSequence:
    """Convert Graph Coloring state to token sequence."""

    def __init__(self, d_model: int, max_colors: int = 10):
        self.d_model = d_model
        self.max_colors = max_colors

        # Feature dimensions
        self.var_feat_dim = (
            5  # is_assigned, domain_size, saturation, degree, backtrack_count
        )
        self.con_feat_dim = 3  # endpoint_assigned, is_satisfied, tension
        self.action_feat_dim = 3  # type, var_idx_norm, color_norm

    def compute_var_features(
        self,
        adjacency: np.ndarray,
        assignment: np.ndarray,
        domains: List[set],
        num_colors: int,
        nogoods: dict,
    ) -> np.ndarray:
        """Compute features for each variable."""
        n = len(assignment)
        features = np.zeros((n, self.var_feat_dim), dtype=np.float32)

        degrees = adjacency.sum(axis=1)
        max_degree = max(degrees.max(), 1)

        for i in range(n):
            # is_assigned
            features[i, 0] = float(assignment[i] != 0)

            # domain_size_norm
            dom_size = len(domains[i]) if assignment[i] == 0 else 1
            features[i, 1] = dom_size / max(num_colors, 1)

            # saturation: distinct colors in neighbors
            neighbors = np.where(adjacency[i])[0]
            neighbor_colors = set(
                assignment[j] for j in neighbors if assignment[j] != 0
            )
            features[i, 2] = len(neighbor_colors) / max(num_colors, 1)

            # degree_norm
            features[i, 3] = degrees[i] / max_degree

            # backtrack_count (from nogoods)
            bt_count = sum(len(nogoods.get(d, {}).get(i, set())) for d in nogoods)
            features[i, 4] = min(bt_count / 10.0, 1.0)  # Normalize

        return features

    def compute_con_features(
        self,
        adjacency: np.ndarray,
        assignment: np.ndarray,
        domains: List[set],
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Compute features for each constraint (edge).

        Returns:
            features: [M, con_feat_dim] array
            edge_info: List of (var_i, var_j, con_idx, _) for each edge
        """
        n = adjacency.shape[0]
        edges = []

        # Get unique edges (upper triangle)
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j]:
                    edges.append((i, j))

        m = len(edges)
        features = np.zeros((m, self.con_feat_dim), dtype=np.float32)
        edge_info = []

        for idx, (i, j) in enumerate(edges):
            # endpoint_assigned_count
            assigned_count = (assignment[i] != 0) + (assignment[j] != 0)
            features[idx, 0] = assigned_count / 2.0

            # is_satisfied (if both assigned, colors differ)
            if assigned_count == 2:
                features[idx, 1] = float(assignment[i] != assignment[j])
            else:
                features[idx, 1] = 1.0  # Not violated yet

            # tension: inverse of domain product
            dom_i = len(domains[i]) if assignment[i] == 0 else 1
            dom_j = len(domains[j]) if assignment[j] == 0 else 1
            features[idx, 2] = 1.0 / max(dom_i * dom_j, 1)

            edge_info.append((i, j, idx, 0))

        return features, edge_info

    def build_edge_list(
        self,
        edge_info: List[Tuple[int, int, int, int]],
        num_vars: int,
    ) -> List[Tuple[int, int]]:
        """
        Build edge list mapping VAR positions to CON positions.

        Token order: [BOS, GLOBAL, VAR_0, ..., VAR_{n-1}, CON_0, ..., CON_{m-1}, SEP]
        """
        VAR_START = 2
        CON_START = 2 + num_vars

        edge_list = []
        for var_i, var_j, con_idx, _ in edge_info:
            var_i_pos = VAR_START + var_i
            var_j_pos = VAR_START + var_j
            con_pos = CON_START + con_idx

            edge_list.append((var_i_pos, con_pos))
            edge_list.append((var_j_pos, con_pos))

        return edge_list


class DecoderOnlyCSP(nn.Module):
    """Decoder-only model for CSP variable selection."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_nodes: int = 100,
        max_colors: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_nodes = max_nodes
        self.max_colors = max_colors

        # Token type embedding
        self.token_type_embed = nn.Embedding(len(TokenType), d_model)

        # Position embedding (for RoPE-style, but we use learned for simplicity)
        self.max_positions = 2048  # Increased from 512
        self.position_embed = nn.Embedding(self.max_positions, d_model)

        # Feature projections
        self.packer = GraphToSequence(d_model, max_colors)
        self.var_proj = nn.Linear(self.packer.var_feat_dim, d_model)
        self.con_proj = nn.Linear(self.packer.con_feat_dim, d_model)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Pointer head
        self.pointer_head = PointerHead(d_model)

    def forward(
        self,
        adjacency: torch.Tensor,  # [B, N, N]
        assignment: torch.Tensor,  # [B, N]
        domains: List[List[set]],  # B x N sets
        num_colors: int,
        nogoods: List[dict],  # B dicts
        action_history: Optional[List[List]] = None,  # B x T actions
    ) -> torch.Tensor:
        """
        Forward pass returning pointer logits.

        Returns: [B, N] logits for variable selection
        """
        B = adjacency.shape[0]
        N = adjacency.shape[1]
        device = adjacency.device

        # Process each batch item (can be optimized later)
        all_logits = []

        for b in range(B):
            adj_np = adjacency[b].cpu().numpy()
            assign_np = assignment[b].cpu().numpy()
            doms = domains[b]
            ngs = nogoods[b] if nogoods else {}
            actions = action_history[b] if action_history else []

            # Limit trace length to avoid exceeding max positions
            max_trace_len = 500
            if len(actions) > max_trace_len:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "decoder_only.trace cap batch=%d item=%d trace=%d cap=%d",
                        B,
                        b,
                        len(actions),
                        max_trace_len,
                    )
                actions = actions[-max_trace_len:]

            # Compute features
            var_feats = self.packer.compute_var_features(
                adj_np, assign_np, doms, num_colors, ngs
            )
            con_feats, edge_info = self.packer.compute_con_features(
                adj_np, assign_np, doms
            )

            num_vars = N
            num_cons = len(edge_info)
            trace_len = len(actions)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "decoder_only.input batch=%d item=%d vars=%d cons=%d trace=%d",
                    B,
                    b,
                    num_vars,
                    num_cons,
                    trace_len,
                )

            # Build edge list for attention mask
            edge_list = self.packer.build_edge_list(edge_info, num_vars)

            # Build attention mask
            struct_mask = AttentionMaskBuilder.build_struct_mask(
                num_vars, num_cons, edge_list, device
            )
            full_mask = AttentionMaskBuilder.build_full_mask(
                struct_mask, trace_len, device
            )

            # Build position IDs
            position_ids = PositionIdBuilder.build(
                num_vars, num_cons, trace_len, device
            )

            max_position_id = int(position_ids.max().item())
            if max_position_id >= self.max_positions:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "decoder_only.position clamp batch=%d item=%d max_position=%d max_positions=%d",
                        B,
                        b,
                        max_position_id,
                        self.max_positions,
                    )

            # Clamp position IDs to valid range
            position_ids = position_ids.clamp(0, self.max_positions - 1)

            # Build input embeddings
            struct_len = 2 + num_vars + num_cons + 1
            total_len = struct_len + trace_len + 1

            embeds = torch.zeros(total_len, self.d_model, device=device)

            # Token type embeddings
            token_types = torch.zeros(total_len, dtype=torch.long, device=device)
            token_types[0] = TokenType.BOS
            token_types[1] = TokenType.GLOBAL
            token_types[2 : 2 + num_vars] = TokenType.VAR
            token_types[2 + num_vars : 2 + num_vars + num_cons] = TokenType.CON
            token_types[2 + num_vars + num_cons] = TokenType.SEP_STRUCT

            # TRACE tokens
            for t, action in enumerate(actions):
                pos = struct_len + t
                if action[0] == "assign":
                    token_types[pos] = TokenType.ACT_ASSIGN
                else:
                    token_types[pos] = TokenType.ACT_BACKTRACK

            token_types[-1] = TokenType.QUERY_SELECT

            embeds = self.token_type_embed(token_types)

            # Add position embeddings
            embeds = embeds + self.position_embed(position_ids)

            # Add feature embeddings for VAR tokens
            var_feats_t = torch.tensor(var_feats, dtype=torch.float32, device=device)
            var_embeds = self.var_proj(var_feats_t)
            embeds[2 : 2 + num_vars] = embeds[2 : 2 + num_vars] + var_embeds

            # Add feature embeddings for CON tokens
            if num_cons > 0:
                con_feats_t = torch.tensor(
                    con_feats, dtype=torch.float32, device=device
                )
                con_embeds = self.con_proj(con_feats_t)
                embeds[2 + num_vars : 2 + num_vars + num_cons] = (
                    embeds[2 + num_vars : 2 + num_vars + num_cons] + con_embeds
                )

            # Add batch dimension
            embeds = embeds.unsqueeze(0)  # [1, L, d]
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

            # Forward through transformer
            x = embeds
            for layer in self.layers:
                x = layer(x, full_mask)
            x = self.final_norm(x)

            # Get query hidden state and VAR hidden states
            query_hidden = x[0, -1]  # [d]
            var_hidden = x[0, 2 : 2 + num_vars]  # [N, d]

            # Compute valid variable mask
            valid_mask = torch.zeros(num_vars, dtype=torch.bool, device=device)
            for i in range(num_vars):
                if assign_np[i] == 0 and len(doms[i]) > 0:
                    valid_mask[i] = True

            # Pointer logits
            logits = self.pointer_head(
                query_hidden.unsqueeze(0),
                var_hidden.unsqueeze(0),
                valid_mask.unsqueeze(0),
            )

            if logger.isEnabledFor(logging.DEBUG):
                finite_logits = logits[0][torch.isfinite(logits[0])]
                if finite_logits.numel() > 0:
                    logger.debug(
                        "decoder_only.pointer stats min=%.4f max=%.4f mean=%.4f valid_vars=%d",
                        finite_logits.min().item(),
                        finite_logits.max().item(),
                        finite_logits.mean().item(),
                        int(valid_mask.sum().item()),
                    )

            all_logits.append(logits)

        return torch.cat(all_logits, dim=0)  # [B, N]


class DecoderOnlyCSPBatched(nn.Module):
    """Optimized decoder-only model with true batched processing."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_nodes: int = 100,
        max_colors: int = 10,
        max_edges: int = 500,  # Max constraint tokens
        max_trace_len: int = 200,  # Reduced for efficiency
        no_trace: bool = False,
        dropout: float = 0.1,
        var_feature_dim: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_trace_len = max_trace_len
        self.no_trace = bool(no_trace)
        self.var_feature_dim = int(var_feature_dim)

        # Pre-compute max sequence length
        # STRUCT: BOS + GLOBAL + max_nodes + max_edges + SEP = 2 + max_nodes + max_edges + 1
        # TRACE: max_trace_len
        # QUERY: 1
        self.max_struct_len = 3 + max_nodes + max_edges
        self.max_seq_len = self.max_struct_len + max_trace_len + 1

        # Embeddings
        self.token_type_embed = nn.Embedding(len(TokenType), d_model)
        self.position_embed = nn.Embedding(self.max_seq_len, d_model)

        # Feature projections
        self.var_proj = nn.Linear(self.var_feature_dim, d_model)  # var features
        self.con_proj = nn.Linear(3, d_model)  # con features
        self.action_proj = nn.Linear(3, d_model)  # action features (type, var_norm, color_norm)

        # Transformer
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Pointer head
        self.pointer_head = PointerHead(d_model)

        # For logging
        self._trace_cap_count = 0

    def _build_batched_inputs(
        self,
        batch_data: List[dict],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build batched inputs with padding.

        Returns:
            embeds: [B, L, d_model] - padded embeddings
            attention_mask: [B, 1, L, L] - attention mask
            position_ids: [B, L] - position IDs
            var_positions: [B, max_nodes] - positions of VAR tokens (-1 for padding)
            valid_var_mask: [B, max_nodes] - mask for valid/selectable variables
        """
        B = len(batch_data)
        L = self.max_seq_len

        # Initialize tensors
        embeds = torch.zeros(B, L, self.d_model, device=device)
        attention_mask = torch.full((B, 1, L, L), float("-inf"), device=device)
        position_ids = torch.zeros(B, L, dtype=torch.long, device=device)
        var_positions = torch.full((B, self.max_nodes), -1, dtype=torch.long, device=device)
        valid_var_mask = torch.zeros(B, self.max_nodes, dtype=torch.bool, device=device)

        for b, data in enumerate(batch_data):
            # Extract data
            adj = (
                data["adjacency"].numpy()
                if torch.is_tensor(data["adjacency"])
                else data["adjacency"]
            )
            assign = (
                data["assignment"].numpy()
                if torch.is_tensor(data["assignment"])
                else data["assignment"]
            )
            domains = data.get("effective_domains")
            if domains is None:
                domains = data["domains"]
            nogoods = data["nogoods"]
            actions = data["action_history"]
            num_colors = data["num_colors"]

            n_vars = len(assign)

            # Compute edges (constraints)
            edges = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adj[i, j]:
                        edges.append((i, j))
            n_cons = min(len(edges), self.max_edges)
            edges = edges[:n_cons]

            # Cap trace length (or set to 0 if no_trace mode)
            action_len = len(actions)
            if self.no_trace:
                trace_len = 0
            else:
                trace_len = min(action_len, self.max_trace_len)
                if action_len > self.max_trace_len:
                    actions = actions[-self.max_trace_len :]
                    self._trace_cap_count += 1

            # Compute sequence positions
            # STRUCT: [BOS, GLOBAL, VAR*n, CON*m, SEP]
            struct_len = 2 + n_vars + n_cons + 1
            total_len = struct_len + trace_len + 1  # +1 for QUERY

            # Build token types
            token_types = torch.zeros(total_len, dtype=torch.long, device=device)
            token_types[0] = TokenType.BOS
            token_types[1] = TokenType.GLOBAL
            token_types[2 : 2 + n_vars] = TokenType.VAR
            token_types[2 + n_vars : 2 + n_vars + n_cons] = TokenType.CON
            token_types[2 + n_vars + n_cons] = TokenType.SEP_STRUCT
            for t in range(trace_len):
                pos = struct_len + t
                if actions[t][0] == "assign":
                    token_types[pos] = TokenType.ACT_ASSIGN
                else:
                    token_types[pos] = TokenType.ACT_BACKTRACK
            token_types[struct_len + trace_len] = TokenType.QUERY_SELECT

            # Position IDs (collapsed for STRUCT)
            pos_ids = torch.zeros(total_len, dtype=torch.long, device=device)
            pos_ids[0] = 0  # BOS
            pos_ids[1] = 0  # GLOBAL
            pos_ids[2 : 2 + n_vars] = 1  # All VAR
            pos_ids[2 + n_vars : 2 + n_vars + n_cons] = 2  # All CON
            pos_ids[2 + n_vars + n_cons] = 3  # SEP
            for t in range(trace_len + 1):  # +1 for QUERY
                pos_ids[struct_len + t] = 4 + t
            pos_ids_clamped = pos_ids.clamp(0, self.position_embed.num_embeddings - 1)
            position_ids[b, :total_len] = pos_ids_clamped

            # Build sample embedding non-inplace
            sample_embed = self.token_type_embed(token_types)
            sample_embed = sample_embed + self.position_embed(pos_ids_clamped)

            # VAR features
            var_feats = self._compute_var_features(
                adj, assign, domains, num_colors, nogoods, n_vars
            )
            var_feats_t = torch.tensor(var_feats, dtype=torch.float32, device=device)
            var_embed_addition = torch.zeros_like(sample_embed)
            var_embed_addition[2 : 2 + n_vars] = self.var_proj(var_feats_t)
            sample_embed = sample_embed + var_embed_addition

            # CON features
            if n_cons > 0:
                con_feats = self._compute_con_features(adj, assign, domains, edges)
                con_feats_t = torch.tensor(con_feats, dtype=torch.float32, device=device)
                con_embed_addition = torch.zeros_like(sample_embed)
                con_embed_addition[
                    2 + n_vars : 2 + n_vars + n_cons
                ] = self.con_proj(con_feats_t)
                sample_embed = sample_embed + con_embed_addition

            # ACTION features for trace tokens
            if trace_len > 0:
                action_feats_list = []
                assign_count = 0
                for t in range(trace_len):
                    action_type, var_idx, color = actions[t]
                    is_assign = 1.0 if action_type == "assign" else 0.0
                    if action_type == "assign":
                        assign_count += 1
                    action_feats_list.append(
                        [
                            is_assign,
                            var_idx / n_vars if var_idx is not None else 0.0,
                            color / num_colors if color is not None else 0.0,
                        ]
                    )
                action_feats_t = torch.tensor(
                    action_feats_list, dtype=torch.float32, device=device
                )
                action_embed = self.action_proj(action_feats_t)
                action_embed_addition = torch.zeros_like(sample_embed)
                action_embed_addition[struct_len : struct_len + trace_len] = action_embed
                sample_embed = sample_embed + action_embed_addition

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "decoder_only.action_feats sample=%d trace=%d assign=%d backtrack=%d",
                        b,
                        trace_len,
                        assign_count,
                        trace_len - assign_count,
                    )

            # Assign to batched tensor
            embeds[b, :total_len] = sample_embed

            if logger.isEnabledFor(logging.DEBUG):
                pos_ids_max = min(self.position_embed.num_embeddings - 1, 4 + trace_len)
                logger.debug(
                    "decoder_only.batched_input sample=%d total_len=%d vars=%d cons=%d trace=%d pos_ids_max=%d no_trace=%s action_len=%d",
                    b,
                    total_len,
                    n_vars,
                    n_cons,
                    trace_len,
                    pos_ids_max,
                    self.no_trace,
                    action_len,
                )

            # Build attention mask for this sample
            mask = self._build_attention_mask(
                n_vars, n_cons, edges, trace_len, struct_len, total_len, device
            )
            attention_mask[b, 0, :total_len, :total_len] = mask

            # Record VAR positions
            for i in range(n_vars):
                var_positions[b, i] = 2 + i
                # Valid if unassigned and has non-empty effective domain
                if assign[i] == 0 and len(domains[i]) > 0:
                    valid_var_mask[b, i] = True

        return embeds, attention_mask, position_ids, var_positions, valid_var_mask

    def _compute_var_features(self, adj, assign, domains, num_colors, nogoods, n_vars):
        """Compute VAR features."""
        if self.var_feature_dim not in (5, 9):
            raise ValueError(
                f"Unsupported var_feature_dim={self.var_feature_dim}; expected 5 or 9"
            )
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
            neighbor_colors = set(assign_arr[j] for j in neighbors if assign_arr[j] != 0)
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

        if feature_dim == 9 and logger.isEnabledFor(logging.DEBUG) and n_vars > 0:
            logger.debug(
                "decoder_only.var_features residual_degree=%.3f mean_neighbor_domain=%.3f "
                "min_neighbor_domain=%.3f tightness=%.3f n_vars=%d",
                float(features[:, 5].mean()),
                float(features[:, 6].mean()),
                float(features[:, 7].mean()),
                float(features[:, 8].mean()),
                int(n_vars),
            )

        return features

    def _compute_con_features(self, adj, assign, domains, edges):
        """Compute CON features."""
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
        self, n_vars, n_cons, edges, trace_len, struct_len, total_len, device
    ):
        """Build attention mask for a single sample."""
        mask = torch.full((total_len, total_len), float("-inf"), device=device)

        # STRUCT positions
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
        for var_idx, (i, j) in enumerate(edges):
            var_i_pos = VAR_START + i
            var_j_pos = VAR_START + j
            con_pos = CON_START + var_idx
            mask[var_i_pos, con_pos] = 0.0
            mask[var_j_pos, con_pos] = 0.0
            mask[con_pos, var_i_pos] = 0.0
            mask[con_pos, var_j_pos] = 0.0

        # SEP attends to all STRUCT
        mask[SEP_POS, :struct_len] = 0.0

        # TRACE: causal + attend to all STRUCT
        for t in range(trace_len + 1):  # +1 for QUERY
            pos = struct_len + t
            mask[pos, :struct_len] = 0.0  # Attend to STRUCT
            mask[pos, struct_len : pos + 1] = 0.0  # Causal within TRACE

        return mask

    def forward(
        self,
        batch_data: List[dict],
    ) -> torch.Tensor:
        """
        Batched forward pass.

        Args:
            batch_data: List of dicts with keys:
                - adjacency, assignment, domains, nogoods, action_history, num_colors

        Returns:
            logits: [B, max_nodes] pointer logits
        """
        device = next(self.parameters()).device
        B = len(batch_data)

        # Build batched inputs
        (
            embeds,
            attention_mask,
            position_ids,
            var_positions,
            valid_var_mask,
        ) = self._build_batched_inputs(batch_data, device)

        # Forward through transformer
        x = embeds
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)

        # Get QUERY hidden states (last non-padded position per sample)
        query_hidden = []
        var_hidden = []

        for b in range(B):
            data = batch_data[b]
            adj = data["adjacency"]
            n_vars = adj.shape[0] if torch.is_tensor(adj) else len(adj)
            actions = data["action_history"]
            action_len = len(actions)

            # Compute edges
            adj_np = adj.numpy() if torch.is_tensor(adj) else adj
            n_cons = sum(
                1 for i in range(n_vars) for j in range(i + 1, n_vars) if adj_np[i, j]
            )
            n_cons = min(n_cons, self.max_edges)

            if self.no_trace:
                trace_len = 0
            else:
                trace_len = min(action_len, self.max_trace_len)
            struct_len = 2 + n_vars + n_cons + 1
            query_pos = struct_len + trace_len

            query_hidden.append(x[b, query_pos])
            var_hidden.append(x[b, 2 : 2 + n_vars])

        # Pad var_hidden to max_nodes
        var_hidden_padded = torch.zeros(B, self.max_nodes, self.d_model, device=device)
        for b in range(B):
            n = var_hidden[b].shape[0]
            var_hidden_padded[b, :n] = var_hidden[b]

        query_hidden = torch.stack(query_hidden)  # [B, d_model]

        # Pointer logits
        logits = self.pointer_head(query_hidden, var_hidden_padded, valid_var_mask)

        return logits

    def log_stats(self):
        """Log accumulated stats and reset counters."""
        if self._trace_cap_count > 0:
            logger.info(
                "Trace capping: %d samples capped to %d",
                self._trace_cap_count,
                self.max_trace_len,
            )
            self._trace_cap_count = 0
