"""Constraint Transformer model using standard Transformer layers."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _IncompatibleKeys

logger = logging.getLogger(__name__)


class EdgeBiasModule(nn.Module):
    """Compute attention bias from edge features (position, polarity, direction)."""

    def __init__(self, num_heads: int, d_edge: int = 16):
        super().__init__()
        self.num_heads = int(num_heads)
        self.d_edge = int(d_edge)

        self.emb_pos = nn.Embedding(4, self.d_edge)
        self.emb_pol = nn.Embedding(2, self.d_edge)
        self.emb_dir = nn.Embedding(2, self.d_edge)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_edge * 3, self.d_edge * 2),
            nn.ReLU(),
            nn.Linear(self.d_edge * 2, self.num_heads),
        )

    def _edge_embeddings(
        self, edge_features: torch.Tensor, direction: int
    ) -> torch.Tensor:
        pos_idx = edge_features[..., 0].to(torch.long)
        pol_idx = edge_features[..., 1].to(torch.long)
        dir_idx = torch.full_like(pos_idx, int(direction), dtype=torch.long)
        pos_emb = self.emb_pos(pos_idx)
        pol_emb = self.emb_pol(pol_idx)
        dir_emb = self.emb_dir(dir_idx)
        return torch.cat([pos_emb, pol_emb, dir_emb], dim=-1)

    def forward_direction(
        self, edge_features: torch.Tensor, direction: int
    ) -> torch.Tensor:
        """Return per-edge bias for a fixed direction."""
        edge_emb = self._edge_embeddings(edge_features, direction)
        return self.mlp(edge_emb)

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_var_idx: torch.Tensor,
        edge_con_idx: torch.Tensor,
        edge_mask: torch.Tensor,
        num_vars: int,
        num_cons: int,
        num_heads: int | None = None,
        *,
        include_global_token: bool = True,
    ) -> torch.Tensor:
        """Return a dense [B, H, S, S] attention bias tensor."""
        if num_heads is not None and int(num_heads) != self.num_heads:
            raise ValueError(
                f"num_heads mismatch: expected {self.num_heads}, got {int(num_heads)}"
            )
        if edge_features.shape[-1] < 2:
            raise ValueError(
                "edge_features must contain at least position and polarity columns"
            )

        B = int(edge_features.shape[0])
        E = int(edge_features.shape[1])
        N = int(num_vars)
        M = int(num_cons)
        S = N + M + (1 if include_global_token else 0)
        bias_dtype = self.emb_pos.weight.dtype
        bias = torch.zeros(
            (int(B), int(self.num_heads), int(S), int(S)),
            device=edge_features.device,
            dtype=bias_dtype,
        )

        if E == 0 or N == 0 or M == 0:
            return bias

        edge_var_idx = edge_var_idx.to(torch.long)
        edge_con_idx = edge_con_idx.to(torch.long)
        edge_mask = edge_mask.bool()

        valid_edges = (
            edge_mask
            & (edge_var_idx >= 0)
            & (edge_con_idx >= 0)
            & (edge_var_idx < N)
            & (edge_con_idx < M)
        )
        safe_var = torch.where(
            valid_edges, edge_var_idx, torch.zeros_like(edge_var_idx)
        )
        safe_con = torch.where(
            valid_edges, edge_con_idx, torch.zeros_like(edge_con_idx)
        )
        var_pos = safe_var
        con_pos = N + safe_con

        bias_v2c = self.forward_direction(edge_features, direction=0)
        bias_c2v = self.forward_direction(edge_features, direction=1)

        valid_scale = valid_edges.unsqueeze(-1).to(dtype=bias_v2c.dtype)
        bias_v2c = bias_v2c * valid_scale
        bias_c2v = bias_c2v * valid_scale

        for h in range(self.num_heads):
            flat = bias[:, h].view(B, S * S)
            flat.scatter_add_(1, (var_pos * S + con_pos).view(B, E), bias_v2c[:, :, h])
            flat.scatter_add_(1, (con_pos * S + var_pos).view(B, E), bias_c2v[:, :, h])

        return bias


class LiteralTokenEmbedding(nn.Module):
    """Embed edge/literal tokens with polarity, position, and endpoint info."""

    def __init__(self, d_model: int, max_clause_size: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)

        self.pol_emb = nn.Embedding(2, self.d_model)
        self.slot_emb = nn.Embedding(int(max_clause_size), self.d_model)

        self.var_proj = nn.Linear(self.d_model, self.d_model)
        self.con_proj = nn.Linear(self.d_model, self.d_model)

        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        edge_features: torch.Tensor,  # [B, E, 2]
        var_hidden: torch.Tensor,  # [B, N, d_model]
        con_hidden: torch.Tensor,  # [B, M, d_model]
        edge_var_idx: torch.Tensor,  # [B, E]
        edge_con_idx: torch.Tensor,  # [B, E]
        edge_mask: torch.Tensor,  # [B, E]
    ) -> torch.Tensor:
        if edge_features.shape[-1] < 2:
            raise ValueError(
                "edge_features must have at least 2 columns (position, polarity)"
            )

        position = edge_features[..., 0].to(torch.long)
        polarity = edge_features[..., 1].to(torch.long)
        position = position.clamp(0, self.slot_emb.num_embeddings - 1)
        polarity = polarity.clamp(0, 1)

        lit_emb = self.pol_emb(polarity) + self.slot_emb(position)

        var_idx_expanded = (
            edge_var_idx.to(torch.long).unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        con_idx_expanded = (
            edge_con_idx.to(torch.long).unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        var_states = torch.gather(
            var_hidden, 1, var_idx_expanded.clamp(0, var_hidden.size(1) - 1)
        )
        con_states = torch.gather(
            con_hidden, 1, con_idx_expanded.clamp(0, con_hidden.size(1) - 1)
        )

        lit_emb = lit_emb + self.var_proj(var_states) + self.con_proj(con_states)
        lit_tokens = self.out_proj(lit_emb)

        lit_tokens = lit_tokens * edge_mask.unsqueeze(-1).to(dtype=lit_tokens.dtype)
        return lit_tokens


class GraphTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with optional edge-conditioned attention bias."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply a pre-norm encoder layer.

        Returns:
            src: Updated hidden states [B, S, d_model]
            attn_weights: Attention weights [B, H, S, S] when requested, else None
        """
        x = self.norm1(src)
        x, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        if not return_attention:
            attn_weights = None
        src = src + self.dropout(x)

        x = self.norm2(src)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout(x)

        return src, attn_weights


class ConstraintTransformer(nn.Module):
    """Transformer for constraint satisfaction with structure-aware attention.

    Unlike Factor-GNN which uses custom message passing, this uses standard
    nn.TransformerEncoderLayer with attention masks that encode the constraint graph.

    Token sequence: [VAR_1, ..., VAR_N, CON_1, ..., CON_M, LIT_1, ..., LIT_E, GLOBAL]
    when use_global_token=True, otherwise the global token is omitted.
    Literal tokens are included only when edge features are non-zero (SAT).

    Attention mask encodes:
    - Variables can attend to: themselves, their literals, global token
    - Constraints can attend to: themselves, their literals, global token
    - Literals can attend to: themselves, their endpoints, global token
    - Global can attend to: everything
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
        use_structure_mask: bool = True,
        use_token_type_embed: bool = True,
        use_nogood_mask: bool = True,
        use_global_token: bool = True,
        use_value_embed: bool = True,
    ):
        super().__init__()

        self.max_vars = int(max_vars)
        self.max_constraints = int(max_constraints)
        self.max_domain = int(max_domain)
        self.num_constraint_types = int(num_constraint_types)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.global_feature_dim = int(global_feature_dim)
        self.use_structure_mask = bool(use_structure_mask)
        self.use_token_type_embed = bool(use_token_type_embed)
        self.use_nogood_mask = bool(use_nogood_mask)
        self.use_global_token = bool(use_global_token)
        self.use_value_embed = bool(use_value_embed)

        # Variable input embedding (same as Factor-GNN).
        var_input_dim = int(var_feature_dim) + (self.max_domain * 3)
        self.var_embed = nn.Sequential(
            nn.Linear(var_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Constraint input embedding (same as Factor-GNN).
        con_input_dim = int(con_feature_dim) + self.num_constraint_types
        self.con_embed = nn.Sequential(
            nn.Linear(con_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Global token embedding.
        self.global_embed = nn.Linear(self.global_feature_dim, self.d_model)

        # Token type embeddings (optional via use_token_type_embed).
        self.token_type_embed = nn.Embedding(
            4, self.d_model
        )  # 0=var, 1=con, 2=global, 3=literal

        # Literal token embedding (SAT literals as tokens).
        self.literal_embed = LiteralTokenEmbedding(
            d_model=self.d_model, max_clause_size=4
        )

        # Edge-conditioned attention bias (legacy path without literals).
        self.edge_bias = EdgeBiasModule(num_heads=self.num_heads)
        self.edge_gate = nn.Sequential(
            nn.Linear(self.global_feature_dim, self.num_heads),
            nn.Sigmoid(),
        )

        # Graph-aware Transformer encoder layers.
        self.layers = nn.ModuleList(
            [
                GraphTransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.num_heads,
                    dim_feedforward=self.d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(self.num_layers)
            ]
        )

        # Value embedding for assignment head (same as Factor-GNN).
        self.value_embed = nn.Embedding(self.max_domain, self.d_model)

        # Assignment head: score for each (var, value) pair.
        self.assign_head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )

        # Backtrack and done heads (operate on global token).
        self.backtrack_head = nn.Linear(self.d_model, 1)
        self.done_head = nn.Linear(self.d_model, 1)

    def _build_attention_mask(
        self,
        batch_size: int,
        num_vars: int,
        num_cons: int,
        edge_var_idx: torch.Tensor,  # [B, E]
        edge_con_idx: torch.Tensor,  # [B, E]
        edge_mask: torch.Tensor,  # [B, E]
        var_mask: torch.Tensor,  # [B, N]
        con_mask: torch.Tensor,  # [B, M]
        device: torch.device,
        *,
        include_global_token: bool = True,
    ) -> torch.Tensor:
        """Build attention mask encoding graph structure.

        Returns:
            attn_mask: [B, S, S] where S = N + M + (1 if include_global_token)
                        True means MASKED (cannot attend), False means can attend
        """

        N = int(num_vars)
        M = int(num_cons)
        S = N + M + (1 if include_global_token else 0)

        var_mask = var_mask.bool()
        con_mask = con_mask.bool()
        invalid_var = ~var_mask
        invalid_con = ~con_mask

        if not self.use_structure_mask:
            attn_mask = torch.zeros(
                (int(batch_size), int(S), int(S)), dtype=torch.bool, device=device
            )

            if N > 0 and invalid_var.any():
                attn_mask[:, :, :N] |= invalid_var[:, None, :]
                attn_mask[:, :N, :] |= invalid_var[:, :, None]

            if M > 0 and invalid_con.any():
                con_slice = slice(N, N + M)
                attn_mask[:, :, con_slice] |= invalid_con[:, None, :]
                attn_mask[:, con_slice, :] |= invalid_con[:, :, None]

            row_all_masked = attn_mask.all(dim=-1)
            if row_all_masked.any():
                batch_idx, token_idx = row_all_masked.nonzero(as_tuple=True)
                attn_mask[batch_idx, token_idx, token_idx] = False

            if logger.isEnabledFor(logging.DEBUG):
                num_padded_vars = int(invalid_var.sum().item())
                num_padded_cons = int(invalid_con.sum().item())
                num_repaired_rows = int(row_all_masked.sum().item())
                logger.debug(
                    "Vanilla attention mask stats: padded_vars=%d padded_cons=%d repaired_rows=%d",
                    num_padded_vars,
                    num_padded_cons,
                    num_repaired_rows,
                )

            return attn_mask

        # Start with all masked (True = cannot attend). We'll unmask allowed connections.
        attn_mask = torch.ones(
            (int(batch_size), int(S), int(S)), dtype=torch.bool, device=device
        )

        # 1. Variables can attend to themselves (diagonal).
        if N > 0:
            var_indices = torch.arange(N, device=device)
            attn_mask[:, var_indices, var_indices] = False

        # 2. Constraints can attend to themselves (diagonal).
        if M > 0:
            con_indices = torch.arange(M, device=device) + N
            attn_mask[:, con_indices, con_indices] = False

        # 3. Global token can attend to everything, everything can attend to global.
        if include_global_token:
            global_idx = S - 1
            attn_mask[:, global_idx, :] = False  # global attends to all
            attn_mask[:, :, global_idx] = False  # all attend to global

        # 4. Variables <-> Constraints based on edges.
        # edge_var_idx[b, e] and edge_con_idx[b, e] define which var-con pairs are connected.
        edge_mask = edge_mask.bool()
        edge_var_idx = edge_var_idx.to(torch.long)
        edge_con_idx = edge_con_idx.to(torch.long)
        E = int(edge_var_idx.shape[1])
        valid_edges = edge_mask

        if E > 0:
            valid_edges = valid_edges & (edge_var_idx < N) & (edge_con_idx < M)
            if valid_edges.any():
                batch_indices = (
                    torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, E)
                )
                b_idx = batch_indices[valid_edges]
                v_idx = edge_var_idx[valid_edges]
                c_idx = edge_con_idx[valid_edges]

                # Variable can attend to this constraint.
                attn_mask[b_idx, v_idx, N + c_idx] = False
                # Constraint can attend to this variable.
                attn_mask[b_idx, N + c_idx, v_idx] = False

        # 5. Apply var_mask and con_mask (padding).
        # If a variable is padded, nothing should attend to it.
        var_mask = var_mask.bool()
        con_mask = con_mask.bool()
        invalid_var = ~var_mask
        invalid_con = ~con_mask

        if N > 0 and invalid_var.any():
            attn_mask[:, :, :N] |= invalid_var[:, None, :]
            attn_mask[:, :N, :] |= invalid_var[:, :, None]

        if M > 0 and invalid_con.any():
            con_slice = slice(N, N + M)
            attn_mask[:, :, con_slice] |= invalid_con[:, None, :]
            attn_mask[:, con_slice, :] |= invalid_con[:, :, None]

        # 6. Safety check: ensure no row is fully masked (avoid NaNs).
        row_all_masked = attn_mask.all(dim=-1)
        if row_all_masked.any():
            batch_idx, token_idx = row_all_masked.nonzero(as_tuple=True)
            attn_mask[batch_idx, token_idx, token_idx] = False

        if logger.isEnabledFor(logging.DEBUG):
            num_valid_edges = int(valid_edges.sum().item())
            num_padded_vars = int(invalid_var.sum().item())
            num_padded_cons = int(invalid_con.sum().item())
            num_repaired_rows = int(row_all_masked.sum().item())
            logger.debug(
                "Attention mask stats: valid_edges=%d padded_vars=%d padded_cons=%d repaired_rows=%d",
                num_valid_edges,
                num_padded_vars,
                num_padded_cons,
                num_repaired_rows,
            )

        return attn_mask

    def _build_attention_mask_with_literals(
        self,
        batch_size: int,
        num_vars: int,
        num_cons: int,
        num_lits: int,
        edge_var_idx: torch.Tensor,  # [B, E]
        edge_con_idx: torch.Tensor,  # [B, E]
        edge_mask: torch.Tensor,  # [B, E]
        var_mask: torch.Tensor,  # [B, N]
        con_mask: torch.Tensor,  # [B, M]
        device: torch.device,
        dtype: torch.dtype,
        *,
        include_global_token: bool = True,
    ) -> torch.Tensor:
        """Build attention mask with literal tokens.

        Returns:
            attn_mask: [B, S, S] float tensor where S = N + M + E +
                        (1 if include_global_token else 0)
                        0.0 for allowed, -inf for blocked
        """
        N = int(num_vars)
        M = int(num_cons)
        E = int(num_lits)
        S = N + M + E + (1 if include_global_token else 0)
        mask_value = torch.finfo(dtype).min

        attn_mask = torch.full(
            (batch_size, S, S), mask_value, device=device, dtype=dtype
        )

        diag = torch.arange(S, device=device)
        attn_mask[:, diag, diag] = 0.0

        if include_global_token:
            attn_mask[:, -1, :] = 0.0
            attn_mask[:, :, -1] = 0.0

        edge_mask = edge_mask.bool()
        edge_var_idx = edge_var_idx.to(torch.long)
        edge_con_idx = edge_con_idx.to(torch.long)
        valid_edges = edge_mask

        if E > 0:
            valid_edges = valid_edges & (edge_var_idx >= 0) & (edge_con_idx >= 0)
            if N > 0:
                valid_edges = valid_edges & (edge_var_idx < N)
            else:
                valid_edges = valid_edges & torch.zeros_like(
                    edge_var_idx, dtype=torch.bool
                )
            if M > 0:
                valid_edges = valid_edges & (edge_con_idx < M)
            else:
                valid_edges = valid_edges & torch.zeros_like(
                    edge_con_idx, dtype=torch.bool
                )

            if valid_edges.any():
                batch_indices = (
                    torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, E)
                )
                lit_positions = (
                    torch.arange(E, device=device).unsqueeze(0).expand(batch_size, -1)
                )
                lit_positions = lit_positions + (N + M)

                b_idx = batch_indices[valid_edges]
                l_idx = lit_positions[valid_edges]

                if N > 0:
                    v_idx = edge_var_idx[valid_edges]
                    attn_mask[b_idx, v_idx, l_idx] = 0.0
                    attn_mask[b_idx, l_idx, v_idx] = 0.0

                if M > 0:
                    c_idx = N + edge_con_idx[valid_edges]
                    attn_mask[b_idx, c_idx, l_idx] = 0.0
                    attn_mask[b_idx, l_idx, c_idx] = 0.0

        var_mask = var_mask.bool()
        con_mask = con_mask.bool()
        invalid_var = ~var_mask
        invalid_con = ~con_mask
        invalid_lit = ~edge_mask

        if N > 0 and invalid_var.any():
            attn_mask[:, :, :N] = attn_mask[:, :, :N].masked_fill(
                invalid_var[:, None, :], mask_value
            )
            attn_mask[:, :N, :] = attn_mask[:, :N, :].masked_fill(
                invalid_var[:, :, None], mask_value
            )

        if M > 0 and invalid_con.any():
            con_slice = slice(N, N + M)
            attn_mask[:, :, con_slice] = attn_mask[:, :, con_slice].masked_fill(
                invalid_con[:, None, :], mask_value
            )
            attn_mask[:, con_slice, :] = attn_mask[:, con_slice, :].masked_fill(
                invalid_con[:, :, None], mask_value
            )

        if E > 0 and invalid_lit.any():
            lit_slice = slice(N + M, N + M + E)
            attn_mask[:, :, lit_slice] = attn_mask[:, :, lit_slice].masked_fill(
                invalid_lit[:, None, :], mask_value
            )
            attn_mask[:, lit_slice, :] = attn_mask[:, lit_slice, :].masked_fill(
                invalid_lit[:, :, None], mask_value
            )

        row_all_masked = attn_mask.eq(mask_value).all(dim=-1)
        if row_all_masked.any():
            batch_idx, token_idx = row_all_masked.nonzero(as_tuple=True)
            attn_mask[batch_idx, token_idx, token_idx] = 0.0

        if logger.isEnabledFor(logging.DEBUG):
            num_valid_edges = int(valid_edges.sum().item()) if E > 0 else 0
            num_padded_vars = int(invalid_var.sum().item()) if N > 0 else 0
            num_padded_cons = int(invalid_con.sum().item()) if M > 0 else 0
            num_padded_lits = int(invalid_lit.sum().item()) if E > 0 else 0
            num_repaired_rows = int(row_all_masked.sum().item())
            mask_min = float(attn_mask.min().item())
            mask_max = float(attn_mask.max().item())
            logger.debug(
                "Literal attention mask stats: S=%d valid_edges=%d padded_vars=%d "
                "padded_cons=%d padded_lits=%d repaired_rows=%d mask[min=%.4f max=%.4f]",
                S,
                num_valid_edges,
                num_padded_vars,
                num_padded_cons,
                num_padded_lits,
                num_repaired_rows,
                mask_min,
                mask_max,
            )

        return attn_mask

    def _build_vanilla_attention_mask_with_literals(
        self,
        batch_size: int,
        num_vars: int,
        num_cons: int,
        num_lits: int,
        edge_mask: torch.Tensor,
        var_mask: torch.Tensor,
        con_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        *,
        include_global_token: bool = True,
    ) -> torch.Tensor:
        """Build vanilla attention mask for literal token path.

        Returns:
            attn_mask: [B * H, S, S] float tensor
        """
        N = int(num_vars)
        M = int(num_cons)
        E = int(num_lits)
        S = N + M + E + (1 if include_global_token else 0)
        mask_value = torch.finfo(dtype).min

        attn_mask = torch.zeros(
            (int(batch_size), int(S), int(S)), dtype=dtype, device=device
        )

        var_mask = var_mask.bool()
        con_mask = con_mask.bool()
        invalid_var = ~var_mask
        invalid_con = ~con_mask
        invalid_lit = ~edge_mask.bool()

        if N > 0 and invalid_var.any():
            attn_mask[:, :, :N] = attn_mask[:, :, :N].masked_fill(
                invalid_var[:, None, :], mask_value
            )
            attn_mask[:, :N, :] = attn_mask[:, :N, :].masked_fill(
                invalid_var[:, :, None], mask_value
            )

        if M > 0 and invalid_con.any():
            con_slice = slice(N, N + M)
            attn_mask[:, :, con_slice] = attn_mask[:, :, con_slice].masked_fill(
                invalid_con[:, None, :], mask_value
            )
            attn_mask[:, con_slice, :] = attn_mask[:, con_slice, :].masked_fill(
                invalid_con[:, :, None], mask_value
            )

        if E > 0 and invalid_lit.any():
            lit_slice = slice(N + M, N + M + E)
            attn_mask[:, :, lit_slice] = attn_mask[:, :, lit_slice].masked_fill(
                invalid_lit[:, None, :], mask_value
            )
            attn_mask[:, lit_slice, :] = attn_mask[:, lit_slice, :].masked_fill(
                invalid_lit[:, :, None], mask_value
            )

        row_all_masked = attn_mask.eq(mask_value).all(dim=-1)
        if row_all_masked.any():
            batch_idx, token_idx = row_all_masked.nonzero(as_tuple=True)
            attn_mask[batch_idx, token_idx, token_idx] = 0.0

        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).clone()
        attn_mask = attn_mask.reshape(batch_size * self.num_heads, S, S)

        if logger.isEnabledFor(logging.DEBUG):
            num_padded_vars = int(invalid_var.sum().item()) if N > 0 else 0
            num_padded_cons = int(invalid_con.sum().item()) if M > 0 else 0
            num_padded_lits = int(invalid_lit.sum().item()) if E > 0 else 0
            num_repaired_rows = int(row_all_masked.sum().item())
            mask_min = float(attn_mask.min().item())
            mask_max = float(attn_mask.max().item())
            logger.debug(
                "Vanilla literal mask stats: S=%d padded_vars=%d padded_cons=%d "
                "padded_lits=%d repaired_rows=%d mask[min=%.4f max=%.4f]",
                S,
                num_padded_vars,
                num_padded_cons,
                num_padded_lits,
                num_repaired_rows,
                mask_min,
                mask_max,
            )

        return attn_mask

    def _build_attention_mask_with_edge_bias(
        self,
        batch_size: int,
        num_vars: int,
        num_cons: int,
        edge_var_idx: torch.Tensor,
        edge_con_idx: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        var_mask: torch.Tensor,
        con_mask: torch.Tensor,
        global_features: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        *,
        include_global_token: bool = True,
    ) -> torch.Tensor:
        """Build attention mask with conditional edge bias.

        Returns:
            attn_mask: [B * H, S, S] float tensor
        """
        if edge_features.dim() != 3:
            raise ValueError(
                "edge_features must have shape [B, E, F]; "
                f"got {tuple(edge_features.shape)}"
            )
        if edge_features.shape[-1] < 2:
            raise ValueError(
                "edge_features must have at least 2 columns (position, polarity)"
            )

        N = int(num_vars)
        M = int(num_cons)
        S = N + M + (1 if include_global_token else 0)

        base_mask = self._build_attention_mask(
            batch_size,
            N,
            M,
            edge_var_idx,
            edge_con_idx,
            edge_mask,
            var_mask,
            con_mask,
            device,
            include_global_token=include_global_token,
        )

        mask_value = torch.finfo(dtype).min
        attn_mask = torch.zeros(
            (int(batch_size), int(S), int(S)), dtype=dtype, device=device
        )
        attn_mask = attn_mask.masked_fill(base_mask, mask_value)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).clone()

        if not self.use_structure_mask:
            attn_mask = attn_mask.reshape(batch_size * self.num_heads, S, S)
            if logger.isEnabledFor(logging.DEBUG):
                mask_min = float(attn_mask.min().item())
                mask_max = float(attn_mask.max().item())
                logger.debug(
                    "Vanilla attention mask with edge bias: S=%d mask[min=%.4f max=%.4f]",
                    S,
                    mask_min,
                    mask_max,
                )
            return attn_mask

        need_edge_bias = edge_features.abs().sum(dim=(1, 2)) > 0
        num_edge_bias = int(need_edge_bias.sum().item())

        if num_edge_bias > 0:
            idx = need_edge_bias.nonzero(as_tuple=True)[0]
            edge_bias = self.edge_bias(
                edge_features[idx],
                edge_var_idx[idx],
                edge_con_idx[idx],
                edge_mask[idx],
                N,
                M,
                self.num_heads,
                include_global_token=include_global_token,
            )
            gate_input = global_features[idx].to(dtype=edge_bias.dtype)
            gate = self.edge_gate(gate_input)
            edge_bias = edge_bias * gate[:, :, None, None]
            edge_bias = edge_bias.to(dtype=dtype)

            edge_bias_full = torch.index_add(
                torch.zeros_like(attn_mask),
                0,
                idx,
                edge_bias,
            )
            attn_mask = attn_mask + edge_bias_full

            if logger.isEnabledFor(logging.DEBUG):
                bias_min = float(edge_bias.min().item())
                bias_max = float(edge_bias.max().item())
                gate_min = float(gate.min().item())
                gate_max = float(gate.max().item())
                logger.debug(
                    "Edge bias applied: batch=%d active=%d "
                    "bias[min=%.4f max=%.4f] gate[min=%.4f max=%.4f]",
                    batch_size,
                    num_edge_bias,
                    bias_min,
                    bias_max,
                    gate_min,
                    gate_max,
                )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Edge bias skipped: batch=%d active=%d",
                batch_size,
                num_edge_bias,
            )

        attn_mask = attn_mask.reshape(batch_size * self.num_heads, S, S)

        if logger.isEnabledFor(logging.DEBUG):
            mask_min = float(attn_mask.min().item())
            mask_max = float(attn_mask.max().item())
            logger.debug(
                "Attention mask with edge bias: S=%d mask[min=%.4f max=%.4f]",
                S,
                mask_min,
                mask_max,
            )

        return attn_mask

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
        edge_features: torch.Tensor,  # [B, E, F_e] (position, polarity for SAT bias)
        var_mask: torch.Tensor,  # [B, N] bool
        con_mask: torch.Tensor,  # [B, M] bool
        edge_mask: torch.Tensor,  # [B, E] bool
        global_features: torch.Tensor,  # [B, G]
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """Forward pass.

        Returns:
            assign_logits: [B, N, D] logits for ASSIGN(var, value)
            backtrack_logit: [B, 1]
            done_logit: [B, 1]
            attention_weights: list of [B, H, S, S] for each layer when requested,
                otherwise None
        """

        B, N, D = var_domain_mask.shape
        M = con_type.shape[1]

        if edge_features.shape[-1] < 2:
            raise ValueError(
                "edge_features must have at least 2 columns (position, polarity); "
                f"got {tuple(edge_features.shape)}"
            )

        if D > self.max_domain:
            raise ValueError(
                f"Input domain size D={D} exceeds model max_domain={self.max_domain}"
            )

        if int(global_features.shape[1]) != int(self.global_feature_dim):
            raise ValueError(
                "global_features must have shape (batch, "
                f"{self.global_feature_dim}); got {tuple(global_features.shape)}"
            )

        device = var_features.device
        dtype = var_features.dtype

        if not self.use_nogood_mask:
            var_nogood_mask = torch.zeros_like(var_nogood_mask)

        # === 1. Build token embeddings ===

        # Variable tokens.
        assigned_mask = (var_assigned >= 0).to(dtype=dtype).unsqueeze(-1)
        var_assigned_idx = var_assigned.clamp(min=0, max=D - 1)
        var_assigned_onehot = F.one_hot(var_assigned_idx, num_classes=D).to(dtype=dtype)
        var_assigned_onehot = var_assigned_onehot * assigned_mask

        var_input = torch.cat(
            [
                var_features,
                var_domain_mask.to(dtype=dtype),
                var_nogood_mask.to(dtype=dtype),
                var_assigned_onehot,
            ],
            dim=-1,
        )
        var_tokens = self.var_embed(var_input)  # [B, N, d_model]

        # Constraint tokens.
        con_type_oh = F.one_hot(
            con_type.clamp(min=0, max=self.num_constraint_types - 1),
            num_classes=self.num_constraint_types,
        ).to(dtype=dtype)
        con_input = torch.cat([con_features, con_type_oh], dim=-1)
        con_tokens = self.con_embed(con_input)  # [B, M, d_model]

        # Global token.
        global_token = None
        if self.use_global_token:
            global_token = self.global_embed(
                global_features.to(dtype=dtype)
            )  # [B, d_model]
            global_token = global_token.unsqueeze(1)  # [B, 1, d_model]

        # === 2. Add token type embeddings ===
        if self.use_token_type_embed:
            var_type = torch.zeros((int(B), int(N)), dtype=torch.long, device=device)
            con_type_id = torch.ones((int(B), int(M)), dtype=torch.long, device=device)

            var_tokens = var_tokens + self.token_type_embed(var_type)
            con_tokens = con_tokens + self.token_type_embed(con_type_id)

            if self.use_global_token and global_token is not None:
                global_type = torch.full(
                    (int(B), 1), 2, dtype=torch.long, device=device
                )
                global_token = global_token + self.token_type_embed(global_type)

        # === 3. Concatenate tokens and build attention mask ===
        use_literal_tokens = bool(edge_features.abs().sum().item() > 0)
        E = int(edge_features.shape[1])

        if use_literal_tokens:
            lit_tokens = self.literal_embed(
                edge_features,
                var_tokens,
                con_tokens,
                edge_var_idx,
                edge_con_idx,
                edge_mask,
            )
            if self.use_token_type_embed:
                lit_type = torch.full((B, E), 3, dtype=torch.long, device=device)
                lit_tokens = lit_tokens + self.token_type_embed(lit_type)

            tokens_parts = [var_tokens, con_tokens, lit_tokens]
            if self.use_global_token:
                if global_token is None:
                    raise ValueError("Global token requested but not initialized")
                tokens_parts.append(global_token)

            tokens = torch.cat(tokens_parts, dim=1)
            if self.use_structure_mask:
                attn_mask = self._build_attention_mask_with_literals(
                    B,
                    N,
                    M,
                    E,
                    edge_var_idx,
                    edge_con_idx,
                    edge_mask,
                    var_mask,
                    con_mask,
                    device,
                    dtype,
                    include_global_token=self.use_global_token,
                )
                S = tokens.size(1)
                attn_mask_expanded = (
                    attn_mask.unsqueeze(1)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(B * self.num_heads, S, S)
                )
            else:
                attn_mask_expanded = self._build_vanilla_attention_mask_with_literals(
                    B,
                    N,
                    M,
                    E,
                    edge_mask,
                    var_mask,
                    con_mask,
                    device,
                    dtype,
                    include_global_token=self.use_global_token,
                )
        else:
            # Sequence: [VAR_1, ..., VAR_N, CON_1, ..., CON_M, GLOBAL?]
            tokens_parts = [var_tokens, con_tokens]
            if self.use_global_token:
                if global_token is None:
                    raise ValueError("Global token requested but not initialized")
                tokens_parts.append(global_token)
            tokens = torch.cat(tokens_parts, dim=1)
            attn_mask_expanded = self._build_attention_mask_with_edge_bias(
                B,
                N,
                M,
                edge_var_idx,
                edge_con_idx,
                edge_features,
                edge_mask,
                var_mask,
                con_mask,
                global_features,
                device,
                dtype,
                include_global_token=self.use_global_token,
            )

        if logger.isEnabledFor(logging.DEBUG):
            token_count = int(tokens.size(1))
            logger.debug(
                "Tokenization path: use_literals=%s tokens=%d vars=%d cons=%d lits=%d "
                "global_token=%s token_type_embed=%s",
                use_literal_tokens,
                token_count,
                N,
                M,
                E,
                self.use_global_token,
                self.use_token_type_embed,
            )

        # === 4. Apply Transformer layers ===
        attention_weights: list[torch.Tensor] | None = [] if return_attention else None
        for layer in self.layers:
            tokens, layer_attn = layer(
                tokens,
                attn_mask=attn_mask_expanded,
                return_attention=return_attention,
            )
            if return_attention:
                if layer_attn is None:
                    raise RuntimeError(
                        "Expected attention weights when return_attention=True"
                    )
                attention_weights.append(layer_attn)

        # === 5. Extract outputs ===
        var_hidden = tokens[:, :N, :]  # [B, N, d_model]
        if N > 0:
            valid_var_counts = var_mask.to(dtype=var_hidden.dtype).sum(dim=1)
        else:
            valid_var_counts = torch.zeros(
                (B,), device=var_hidden.device, dtype=var_hidden.dtype
            )

        if self.use_global_token:
            global_hidden = tokens[:, -1, :]  # [B, d_model]
            global_source = "global"
        else:
            if N > 0:
                var_weights = var_mask.to(dtype=var_hidden.dtype).unsqueeze(-1)
                denom = valid_var_counts.clamp(min=1.0).unsqueeze(-1)
                global_hidden = (var_hidden * var_weights).sum(dim=1) / denom
            else:
                global_hidden = torch.zeros(
                    (B, self.d_model), device=var_hidden.device, dtype=var_hidden.dtype
                )
            global_source = "var_mean"

        # === 6. Assignment logits ===
        if self.use_value_embed:
            value_emb = self.value_embed.weight[:D]  # [D, d_model]
        else:
            if self.d_model < D:
                raise ValueError(
                    "use_value_embed=False requires d_model >= domain size; "
                    f"got d_model={self.d_model} D={D}"
                )
            value_emb = torch.eye(D, device=var_hidden.device, dtype=var_hidden.dtype)
            if self.d_model > D:
                value_emb = F.pad(value_emb, (0, self.d_model - D))
        value_emb = value_emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        var_hidden_exp = var_hidden.unsqueeze(2).expand(-1, -1, D, -1)
        assign_input = torch.cat([var_hidden_exp, value_emb], dim=-1)
        assign_logits = self.assign_head(assign_input).squeeze(-1)

        # Mask invalid assignments.
        valid_assign = var_domain_mask & ~var_nogood_mask
        valid_assign = valid_assign & (var_assigned == -1).unsqueeze(-1)
        valid_assign = valid_assign & var_mask.unsqueeze(-1)
        assign_logits = assign_logits.masked_fill(~valid_assign, float("-inf"))

        # === 7. Backtrack and done logits ===
        backtrack_logit = self.backtrack_head(global_hidden)
        done_logit = self.done_head(global_hidden)

        if logger.isEnabledFor(logging.DEBUG):
            valid_count = int(valid_assign.sum().item())
            valid_var_mean = float(valid_var_counts.mean().item())
            assign_min = float(assign_logits.detach().min().item())
            assign_max = float(assign_logits.detach().max().item())
            backtrack_mean = float(backtrack_logit.detach().mean().item())
            done_mean = float(done_logit.detach().mean().item())
            attention_layers = (
                0 if attention_weights is None else len(attention_weights)
            )
            logger.debug(
                "ConstraintTransformer forward metrics: B=%d N=%d M=%d D=%d "
                "valid_assign=%d valid_vars_avg=%.2f assign_logits[min=%.4f max=%.4f] "
                "backtrack_mean=%.4f done_mean=%.4f global_source=%s "
                "use_global_token=%s use_value_embed=%s use_token_type_embed=%s "
                "use_nogood_mask=%s return_attention=%s attention_layers=%d",
                B,
                N,
                M,
                D,
                valid_count,
                valid_var_mean,
                assign_min,
                assign_max,
                backtrack_mean,
                done_mean,
                global_source,
                self.use_global_token,
                self.use_value_embed,
                self.use_token_type_embed,
                self.use_nogood_mask,
                return_attention,
                attention_layers,
            )

        return assign_logits, backtrack_logit, done_logit, attention_weights


class ConstraintTransformerMinimal(ConstraintTransformer):
    """Minimal ConstraintTransformer that removes unused parameters."""

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
        use_structure_mask: bool = True,
        use_token_type_embed: bool = True,
        use_nogood_mask: bool = True,
        use_global_token: bool = True,
        use_value_embed: bool = True,
    ):
        super().__init__(
            max_vars=max_vars,
            max_constraints=max_constraints,
            max_domain=max_domain,
            num_constraint_types=num_constraint_types,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            var_feature_dim=var_feature_dim,
            con_feature_dim=con_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            use_structure_mask=use_structure_mask,
            use_token_type_embed=use_token_type_embed,
            use_nogood_mask=use_nogood_mask,
            use_global_token=use_global_token,
            use_value_embed=use_value_embed,
        )

        if self.use_nogood_mask:
            var_input_dim = int(var_feature_dim) + (self.max_domain * 3)
        else:
            var_input_dim = int(var_feature_dim) + (self.max_domain * 2)

        self.var_embed = nn.Sequential(
            nn.Linear(var_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        if self.use_value_embed:
            assign_head_in = self.d_model * 2
        else:
            self.value_embed = None
            self.assign_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 1),
            )
            assign_head_in = self.d_model

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ConstraintTransformerMinimal init: var_input_dim=%d assign_head_in=%d "
                "use_value_embed=%s use_nogood_mask=%s",
                var_input_dim,
                assign_head_in,
                self.use_value_embed,
                self.use_nogood_mask,
            )

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        if strict:
            return super().load_state_dict(state_dict, strict=True, assign=assign)

        model_state = self.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in state_dict.items():
            target = model_state.get(key)
            if target is None:
                skipped.append(key)
                continue
            if tuple(target.shape) != tuple(value.shape):
                skipped.append(key)
                continue
            filtered[key] = value

        incompatible = super().load_state_dict(filtered, strict=False, assign=assign)

        if skipped:
            logger.warning(
                "ConstraintTransformerMinimal skipped %d incompatible keys: %s",
                len(skipped),
                skipped,
            )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "ConstraintTransformerMinimal load_state_dict missing_keys=%s "
                "unexpected_keys=%s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        return incompatible

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
        edge_features: torch.Tensor,  # [B, E, F_e] (position, polarity for SAT bias)
        var_mask: torch.Tensor,  # [B, N] bool
        con_mask: torch.Tensor,  # [B, M] bool
        edge_mask: torch.Tensor,  # [B, E] bool
        global_features: torch.Tensor,  # [B, G]
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """Forward pass.

        Returns:
            assign_logits: [B, N, D] logits for ASSIGN(var, value)
            backtrack_logit: [B, 1]
            done_logit: [B, 1]
            attention_weights: list of [B, H, S, S] for each layer when requested,
                otherwise None
        """

        B, N, D = var_domain_mask.shape
        M = con_type.shape[1]

        if edge_features.shape[-1] < 2:
            raise ValueError(
                "edge_features must have at least 2 columns (position, polarity); "
                f"got {tuple(edge_features.shape)}"
            )

        if D > self.max_domain:
            raise ValueError(
                f"Input domain size D={D} exceeds model max_domain={self.max_domain}"
            )

        if int(global_features.shape[1]) != int(self.global_feature_dim):
            raise ValueError(
                "global_features must have shape (batch, "
                f"{self.global_feature_dim}); got {tuple(global_features.shape)}"
            )

        device = var_features.device
        dtype = var_features.dtype

        if not self.use_nogood_mask:
            var_nogood_mask = torch.zeros_like(var_nogood_mask)

        # === 1. Build token embeddings ===

        # Variable tokens.
        assigned_mask = (var_assigned >= 0).to(dtype=dtype).unsqueeze(-1)
        var_assigned_idx = var_assigned.clamp(min=0, max=D - 1)
        var_assigned_onehot = F.one_hot(var_assigned_idx, num_classes=D).to(dtype=dtype)
        var_assigned_onehot = var_assigned_onehot * assigned_mask

        if self.use_nogood_mask:
            var_input = torch.cat(
                [
                    var_features,
                    var_domain_mask.to(dtype=dtype),
                    var_nogood_mask.to(dtype=dtype),
                    var_assigned_onehot,
                ],
                dim=-1,
            )
        else:
            var_input = torch.cat(
                [
                    var_features,
                    var_domain_mask.to(dtype=dtype),
                    var_assigned_onehot,
                ],
                dim=-1,
            )

        var_tokens = self.var_embed(var_input)  # [B, N, d_model]

        # Constraint tokens.
        con_type_oh = F.one_hot(
            con_type.clamp(min=0, max=self.num_constraint_types - 1),
            num_classes=self.num_constraint_types,
        ).to(dtype=dtype)
        con_input = torch.cat([con_features, con_type_oh], dim=-1)
        con_tokens = self.con_embed(con_input)  # [B, M, d_model]

        # Global token.
        global_token = None
        if self.use_global_token:
            global_token = self.global_embed(
                global_features.to(dtype=dtype)
            )  # [B, d_model]
            global_token = global_token.unsqueeze(1)  # [B, 1, d_model]

        # === 2. Add token type embeddings ===
        if self.use_token_type_embed:
            var_type = torch.zeros((int(B), int(N)), dtype=torch.long, device=device)
            con_type_id = torch.ones((int(B), int(M)), dtype=torch.long, device=device)

            var_tokens = var_tokens + self.token_type_embed(var_type)
            con_tokens = con_tokens + self.token_type_embed(con_type_id)

            if self.use_global_token and global_token is not None:
                global_type = torch.full(
                    (int(B), 1), 2, dtype=torch.long, device=device
                )
                global_token = global_token + self.token_type_embed(global_type)

        # === 3. Concatenate tokens and build attention mask ===
        use_literal_tokens = bool(edge_features.abs().sum().item() > 0)
        E = int(edge_features.shape[1])

        if use_literal_tokens:
            lit_tokens = self.literal_embed(
                edge_features,
                var_tokens,
                con_tokens,
                edge_var_idx,
                edge_con_idx,
                edge_mask,
            )
            if self.use_token_type_embed:
                lit_type = torch.full((B, E), 3, dtype=torch.long, device=device)
                lit_tokens = lit_tokens + self.token_type_embed(lit_type)

            tokens_parts = [var_tokens, con_tokens, lit_tokens]
            if self.use_global_token:
                if global_token is None:
                    raise ValueError("Global token requested but not initialized")
                tokens_parts.append(global_token)

            tokens = torch.cat(tokens_parts, dim=1)
            if self.use_structure_mask:
                attn_mask = self._build_attention_mask_with_literals(
                    B,
                    N,
                    M,
                    E,
                    edge_var_idx,
                    edge_con_idx,
                    edge_mask,
                    var_mask,
                    con_mask,
                    device,
                    dtype,
                    include_global_token=self.use_global_token,
                )
                S = tokens.size(1)
                attn_mask_expanded = (
                    attn_mask.unsqueeze(1)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(B * self.num_heads, S, S)
                )
            else:
                attn_mask_expanded = self._build_vanilla_attention_mask_with_literals(
                    B,
                    N,
                    M,
                    E,
                    edge_mask,
                    var_mask,
                    con_mask,
                    device,
                    dtype,
                    include_global_token=self.use_global_token,
                )
        else:
            # Sequence: [VAR_1, ..., VAR_N, CON_1, ..., CON_M, GLOBAL?]
            tokens_parts = [var_tokens, con_tokens]
            if self.use_global_token:
                if global_token is None:
                    raise ValueError("Global token requested but not initialized")
                tokens_parts.append(global_token)
            tokens = torch.cat(tokens_parts, dim=1)
            attn_mask_expanded = self._build_attention_mask_with_edge_bias(
                B,
                N,
                M,
                edge_var_idx,
                edge_con_idx,
                edge_features,
                edge_mask,
                var_mask,
                con_mask,
                global_features,
                device,
                dtype,
                include_global_token=self.use_global_token,
            )

        if logger.isEnabledFor(logging.DEBUG):
            token_count = int(tokens.size(1))
            logger.debug(
                "Tokenization path: use_literals=%s tokens=%d vars=%d cons=%d lits=%d "
                "global_token=%s token_type_embed=%s",
                use_literal_tokens,
                token_count,
                N,
                M,
                E,
                self.use_global_token,
                self.use_token_type_embed,
            )

        # === 4. Apply Transformer layers ===
        attention_weights: list[torch.Tensor] | None = [] if return_attention else None
        for layer in self.layers:
            tokens, layer_attn = layer(
                tokens,
                attn_mask=attn_mask_expanded,
                return_attention=return_attention,
            )
            if return_attention:
                if layer_attn is None:
                    raise RuntimeError(
                        "Expected attention weights when return_attention=True"
                    )
                attention_weights.append(layer_attn)

        # === 5. Extract outputs ===
        var_hidden = tokens[:, :N, :]  # [B, N, d_model]
        if N > 0:
            valid_var_counts = var_mask.to(dtype=var_hidden.dtype).sum(dim=1)
        else:
            valid_var_counts = torch.zeros(
                (B,), device=var_hidden.device, dtype=var_hidden.dtype
            )

        if self.use_global_token:
            global_hidden = tokens[:, -1, :]  # [B, d_model]
            global_source = "global"
        else:
            if N > 0:
                var_weights = var_mask.to(dtype=var_hidden.dtype).unsqueeze(-1)
                denom = valid_var_counts.clamp(min=1.0).unsqueeze(-1)
                global_hidden = (var_hidden * var_weights).sum(dim=1) / denom
            else:
                global_hidden = torch.zeros(
                    (B, self.d_model), device=var_hidden.device, dtype=var_hidden.dtype
                )
            global_source = "var_mean"

        # === 6. Assignment logits ===
        var_hidden_exp = var_hidden.unsqueeze(2).expand(-1, -1, D, -1)
        if self.use_value_embed:
            value_emb = self.value_embed.weight[:D]  # [D, d_model]
            value_emb = value_emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            assign_input = torch.cat([var_hidden_exp, value_emb], dim=-1)
            assign_logits = self.assign_head(assign_input).squeeze(-1)
        else:
            assign_logits = self.assign_head(var_hidden_exp).squeeze(-1)

        # Mask invalid assignments.
        valid_assign = var_domain_mask & ~var_nogood_mask
        valid_assign = valid_assign & (var_assigned == -1).unsqueeze(-1)
        valid_assign = valid_assign & var_mask.unsqueeze(-1)
        assign_logits = assign_logits.masked_fill(~valid_assign, float("-inf"))

        # === 7. Backtrack and done logits ===
        backtrack_logit = self.backtrack_head(global_hidden)
        done_logit = self.done_head(global_hidden)

        if logger.isEnabledFor(logging.DEBUG):
            valid_count = int(valid_assign.sum().item())
            valid_var_mean = float(valid_var_counts.mean().item())
            assign_min = float(assign_logits.detach().min().item())
            assign_max = float(assign_logits.detach().max().item())
            backtrack_mean = float(backtrack_logit.detach().mean().item())
            done_mean = float(done_logit.detach().mean().item())
            attention_layers = (
                0 if attention_weights is None else len(attention_weights)
            )
            logger.debug(
                "ConstraintTransformer forward metrics: B=%d N=%d M=%d D=%d "
                "valid_assign=%d valid_vars_avg=%.2f assign_logits[min=%.4f max=%.4f] "
                "backtrack_mean=%.4f done_mean=%.4f global_source=%s "
                "use_global_token=%s use_value_embed=%s use_token_type_embed=%s "
                "use_nogood_mask=%s return_attention=%s attention_layers=%d",
                B,
                N,
                M,
                D,
                valid_count,
                valid_var_mean,
                assign_min,
                assign_max,
                backtrack_mean,
                done_mean,
                global_source,
                self.use_global_token,
                self.use_value_embed,
                self.use_token_type_embed,
                self.use_nogood_mask,
                return_attention,
                attention_layers,
            )

        return assign_logits, backtrack_logit, done_logit, attention_weights
