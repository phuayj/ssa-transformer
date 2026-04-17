"""Delta-local verification head for slot decoder."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class DeltaLocalVerifyHead(nn.Module):
    """Delta-local verification: only checks neighbors of last-assigned node."""

    def __init__(self, d_model: int, n_colors: int = 4, max_neighbors: int = 30):
        super().__init__()
        self.d_model = int(d_model)
        self.n_colors = int(n_colors)
        self.max_neighbors = int(max_neighbors)

        self.neighbor_proj = nn.Linear(self.d_model, self.d_model)
        self.assign_proj = nn.Linear(self.d_model, self.d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=4,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 2),
        )

    def forward(
        self,
        seq_hidden: Tensor,
        neighbor_positions: Tensor,
        neighbor_mask: Tensor,
        assign_hidden: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            seq_hidden: [B, T, D] - full sequence hidden states
            neighbor_positions: [B, max_neighbors] - positions of neighbor tokens in the STATE block
            neighbor_mask: [B, max_neighbors] - 1 for valid neighbors, 0 for padding
            assign_hidden: [B, D] - hidden state at the assignment position
        Returns:
            global_logits: [B, 2] - aggregated OK/CF logits
            neighbor_logits: [B, max_neighbors, 2] - per-neighbor OK/CF logits
        """
        if int(seq_hidden.dim()) != 3:
            raise ValueError("seq_hidden must be [B, T, D]")

        batch_size, seq_len, hidden_dim = seq_hidden.shape
        if int(hidden_dim) != int(self.d_model):
            raise ValueError(
                f"hidden dim mismatch: got {hidden_dim}, expected {self.d_model}"
            )

        if int(neighbor_positions.dim()) != 2:
            raise ValueError("neighbor_positions must be [B, M]")
        if int(neighbor_mask.dim()) != 2:
            raise ValueError("neighbor_mask must be [B, M]")
        if int(assign_hidden.dim()) != 2:
            raise ValueError("assign_hidden must be [B, D]")

        if int(neighbor_positions.size(0)) != int(batch_size):
            raise ValueError("neighbor_positions batch mismatch")
        if int(neighbor_mask.size(0)) != int(batch_size):
            raise ValueError("neighbor_mask batch mismatch")

        neighbor_positions = neighbor_positions.to(seq_hidden.device, dtype=torch.long)
        neighbor_mask = neighbor_mask.to(seq_hidden.device)
        assign_hidden = assign_hidden.to(seq_hidden.device)

        valid_mask = (neighbor_positions >= 0) & neighbor_mask.to(torch.bool)
        safe_positions = neighbor_positions.clamp(min=0, max=max(seq_len - 1, 0))

        gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
        neighbor_hiddens = torch.gather(seq_hidden, dim=1, index=gather_index)

        neighbor_hiddens = self.neighbor_proj(neighbor_hiddens)
        assign_query = self.assign_proj(assign_hidden).unsqueeze(1)

        attn_context = torch.zeros(
            (batch_size, 1, hidden_dim),
            device=seq_hidden.device,
            dtype=seq_hidden.dtype,
        )
        has_any_neighbor = valid_mask.any(dim=1)
        if has_any_neighbor.any():
            active_idx = torch.nonzero(has_any_neighbor, as_tuple=False).flatten()
            active_q = assign_query[active_idx]
            active_kv = neighbor_hiddens[active_idx]
            active_kpm = ~valid_mask[active_idx]
            active_ctx, _ = self.cross_attn(
                query=active_q,
                key=active_kv,
                value=active_kv,
                key_padding_mask=active_kpm,
            )
            attn_context[active_idx] = active_ctx

        fused_neighbor = neighbor_hiddens + attn_context.expand_as(neighbor_hiddens)
        neighbor_logits = self.classifier(fused_neighbor)

        eps = 1e-8
        neighbor_probs_cf = F.softmax(neighbor_logits.float(), dim=-1)[..., 1]
        neighbor_probs_cf = neighbor_probs_cf * valid_mask.float()
        p_no_conflict = torch.prod(1.0 - neighbor_probs_cf, dim=1)
        p_conflict = 1.0 - p_no_conflict
        p_ok = 1.0 - p_conflict

        global_logits = torch.stack(
            [
                torch.log(p_ok.clamp_min(eps)),
                torch.log(p_conflict.clamp_min(eps)),
            ],
            dim=-1,
        ).to(seq_hidden.dtype)

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                active_count = int(valid_mask.sum().item())
                mean_local_cf = (
                    float(neighbor_probs_cf[valid_mask].mean().item())
                    if active_count > 0
                    else 0.0
                )
                logger.debug(
                    "DeltaLocalVerifyHead: batch=%d seq_len=%d active_neighbors=%d mean_local_cf=%.6f mean_global_cf=%.6f",
                    int(batch_size),
                    int(seq_len),
                    active_count,
                    mean_local_cf,
                    float(p_conflict.mean().item()),
                )

        return global_logits, neighbor_logits
