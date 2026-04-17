from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GraphColorModelConfig:
    max_nodes: int = 50
    num_colors: int = 3
    d_model: int = 128
    num_gnn_layers: int = 3
    num_transformer_layers: int = 2
    nhead: int = 4
    dropout: float = 0.1


class _GraphSAGELayer(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_neigh = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, neigh: torch.Tensor) -> torch.Tensor:
        x = self.lin_self(h) + self.lin_neigh(neigh)
        x = self.act(x)
        x = self.dropout(x)
        return self.norm(h + x)


class GraphColorModel(nn.Module):
    """Equivariant-ish model for graph k-coloring.

    MVP design:
      - GraphSAGE-style message passing on adjacency matrix.
      - Transformer over [global tokens | node tokens] (no padding mask in MVP).
      - Heads:
          - action_type_logits: (B, 5)
          - node_logits: (B, N)
          - color_logits: (B, K)
          - validity_logits: (B, 1)

    Notes on color equivariance:
      - This model does NOT use per-color ID embeddings in its color head.
      - Color logits are produced from per-color *state-derived* features (counts),
        then masked by the selected node's domain.

    Expected inputs match GraphColorStepDataset.
    """

    GLOBAL_FEATURES_LEN = 5
    NODE_FEATURES_LEN = 4

    def __init__(
        self,
        max_nodes: int = 50,
        num_colors: int = 3,
        d_model: int = 128,
        num_gnn_layers: int = 3,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if int(max_nodes) < 1:
            raise ValueError("max_nodes must be >= 1")
        if int(num_colors) < 1:
            raise ValueError("num_colors must be >= 1")
        if int(d_model) < 1:
            raise ValueError("d_model must be >= 1")
        if int(num_gnn_layers) < 0:
            raise ValueError("num_gnn_layers must be >= 0")
        if int(num_transformer_layers) < 1:
            raise ValueError("num_transformer_layers must be >= 1")
        if int(nhead) < 1:
            raise ValueError("nhead must be >= 1")

        self.max_nodes = int(max_nodes)
        self.num_colors = int(num_colors)
        self.d_model = int(d_model)

        # --- Node embeddings (discrete features from GraphColorStepDataset) ---
        self.node_idx_embedding = nn.Embedding(self.max_nodes, self.d_model)
        self.degree_embedding = nn.Embedding(self.max_nodes, self.d_model)
        self.is_selected_embedding = nn.Embedding(2, self.d_model)
        self.is_assigned_embedding = nn.Embedding(2, self.d_model)
        self.domain_size_embedding = nn.Embedding(self.num_colors + 1, self.d_model)
        self.colored_neighbors_embedding = nn.Embedding(self.max_nodes + 1, self.d_model)

        # --- Global embeddings (same 5-tuple pattern as CSP) ---
        self.selected_node_embedding = nn.Embedding(self.max_nodes + 1, self.d_model)  # -1..N-1 -> 0..N
        self.num_assigned_embedding = nn.Embedding(self.max_nodes + 1, self.d_model)
        self.num_empty_embedding = nn.Embedding(self.max_nodes + 1, self.d_model)
        self.propagation_pending_embedding = nn.Embedding(2, self.d_model)
        self.stack_depth_embedding = nn.Embedding(self.max_nodes + 1, self.d_model)

        self.global_pos_encoding = nn.Embedding(self.GLOBAL_FEATURES_LEN, self.d_model)

        self.dropout = nn.Dropout(float(dropout))

        # --- Graph message passing ---
        self.gnn_layers = nn.ModuleList([
            _GraphSAGELayer(self.d_model, float(dropout)) for _ in range(int(num_gnn_layers))
        ])

        # --- Transformer encoder over [global tokens | node tokens] ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(4 * self.d_model),
            dropout=float(dropout),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_transformer_layers))

        # --- Color "set" encoder (token-per-color, no positional encoding) ---
        self.color_mlp = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.d_model),
        )
        color_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(4 * self.d_model),
            dropout=float(dropout),
            batch_first=True,
        )
        self.color_transformer = nn.TransformerEncoder(color_enc_layer, num_layers=1)

        # --- Output heads ---
        self.action_type_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 5),
        )

        # Pointer scoring: MLP([g; x; g*x]) -> scalar.
        self.node_pointer = nn.Sequential(
            nn.Linear(3 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

        self.color_pointer = nn.Sequential(
            nn.Linear(3 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

        self.validity_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,  # (B, N, 4)
        adjacency: torch.Tensor,  # (B, N, N)
        domain_values: torch.Tensor,  # (B, N, K)
        domain_mask: torch.Tensor,  # (B, N, K)
        global_features: torch.Tensor,  # (B, 5)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if global_features.ndim != 2 or global_features.shape[1] != self.GLOBAL_FEATURES_LEN:
            raise ValueError(
                f"global_features must have shape (batch, {self.GLOBAL_FEATURES_LEN}); got {tuple(global_features.shape)}"
            )
        if node_features.ndim != 3 or node_features.shape[0] != global_features.shape[0]:
            raise ValueError(
                "node_features must have shape (batch, N, 4) with same batch as global_features; "
                f"got node_features={tuple(node_features.shape)} global_features={tuple(global_features.shape)}"
            )
        if node_features.shape[2] != self.NODE_FEATURES_LEN:
            raise ValueError(
                f"node_features must have last dim {self.NODE_FEATURES_LEN}; got {tuple(node_features.shape)}"
            )

        bsz = int(global_features.shape[0])
        n = int(node_features.shape[1])

        if int(n) < 1:
            raise ValueError("N must be >= 1")
        if int(n) > int(self.max_nodes):
            raise ValueError(f"N={n} exceeds max_nodes={self.max_nodes}")

        if adjacency.shape != (bsz, n, n):
            raise ValueError(f"adjacency must have shape (batch, N, N); got {tuple(adjacency.shape)}")
        if domain_values.shape[:2] != (bsz, n):
            raise ValueError(
                f"domain_values must have shape (batch, N, K); got {tuple(domain_values.shape)}"
            )
        if domain_mask.shape[:2] != (bsz, n):
            raise ValueError(
                f"domain_mask must have shape (batch, N, K); got {tuple(domain_mask.shape)}"
            )

        k = int(domain_values.shape[2])
        if k != int(self.num_colors):
            raise ValueError(f"Expected K=num_colors={self.num_colors}; got K={k}")
        if domain_mask.shape[2] != k:
            raise ValueError("domain_mask K mismatch")

        device = global_features.device

        # Ensure dtypes.
        global_features = global_features.long()
        node_features = node_features.long()
        adjacency_bool = adjacency.bool()
        domain_values = domain_values.long()
        domain_mask_bool = domain_mask.bool()

        # --- Parse global features ---
        selected_node = global_features[:, 0]
        num_assigned = global_features[:, 1]
        num_empty = global_features[:, 2]
        propagation_pending = global_features[:, 3]
        stack_depth = global_features[:, 4]

        if (selected_node < -1).any() or (selected_node > n - 1).any():
            bad = selected_node[(selected_node < -1) | (selected_node > n - 1)][:5].tolist()
            raise ValueError(f"selected_node out of range [-1, {n - 1}]: {bad}")
        if (num_assigned < 0).any() or (num_assigned > n).any():
            bad = num_assigned[(num_assigned < 0) | (num_assigned > n)][:5].tolist()
            raise ValueError(f"num_assigned out of range [0, {n}]: {bad}")
        if (num_empty < 0).any() or (num_empty > n).any():
            bad = num_empty[(num_empty < 0) | (num_empty > n)][:5].tolist()
            raise ValueError(f"num_empty_domains out of range [0, {n}]: {bad}")
        if ((propagation_pending != 0) & (propagation_pending != 1)).any():
            bad = propagation_pending[((propagation_pending != 0) & (propagation_pending != 1))][:5].tolist()
            raise ValueError(f"propagation_pending must be 0/1; got {bad}")
        if (stack_depth < 0).any() or (stack_depth > n).any():
            bad = stack_depth[(stack_depth < 0) | (stack_depth > n)][:5].tolist()
            raise ValueError(f"stack_depth out of range [0, {n}]: {bad}")

        # --- Parse node features ---
        node_idx = node_features[:, :, 0]
        degree_obs = node_features[:, :, 1]
        assigned_color = node_features[:, :, 2]
        is_selected = node_features[:, :, 3]

        if (node_idx < 0).any() or (node_idx > n - 1).any():
            bad = node_idx[(node_idx < 0) | (node_idx > n - 1)][:5].tolist()
            raise ValueError(f"node_idx out of range [0, {n - 1}]: {bad}")
        if (degree_obs < 0).any() or (degree_obs > n - 1).any():
            bad = degree_obs[(degree_obs < 0) | (degree_obs > n - 1)][:5].tolist()
            raise ValueError(f"degree out of range [0, {n - 1}]: {bad}")
        if (assigned_color < 0).any() or (assigned_color > k).any():
            bad = assigned_color[(assigned_color < 0) | (assigned_color > k)][:5].tolist()
            raise ValueError(f"assigned_color out of range [0, {k}]: {bad}")
        if ((is_selected != 0) & (is_selected != 1)).any():
            bad = is_selected[((is_selected != 0) & (is_selected != 1))][:5].tolist()
            raise ValueError(f"is_selected must be 0/1; got {bad}")

        # Use adjacency to compute degree for message passing (more robust than trusting obs).
        degree = adjacency_bool.sum(dim=-1).clamp(min=0, max=self.max_nodes - 1).long()

        is_assigned = assigned_color != 0
        domain_size = domain_mask_bool.sum(dim=2).clamp(min=0, max=k).long()

        colored_neighbors = (
            adjacency_bool & is_assigned.unsqueeze(1).expand(-1, n, -1)
        ).sum(dim=2)
        colored_neighbors = colored_neighbors.clamp(min=0, max=self.max_nodes).long()

        # --- Node token embeddings ---
        node_x = (
            self.node_idx_embedding(node_idx)
            + self.degree_embedding(degree)
            + self.is_selected_embedding(is_selected)
            + self.is_assigned_embedding(is_assigned.long())
            + self.domain_size_embedding(domain_size)
            + self.colored_neighbors_embedding(colored_neighbors)
        )

        # --- Graph message passing ---
        if len(self.gnn_layers) > 0:
            adj_f = adjacency_bool.to(dtype=node_x.dtype)
            for layer in self.gnn_layers:
                neigh_sum = torch.bmm(adj_f, node_x)
                denom = adj_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
                neigh_mean = neigh_sum / denom
                node_x = layer(node_x, neigh_mean)

        # --- Global token embeddings ---
        selected_node_idx = selected_node + 1  # -1..N-1 -> 0..N

        g0 = self.selected_node_embedding(selected_node_idx)
        g1 = self.num_assigned_embedding(num_assigned)
        g2 = self.num_empty_embedding(num_empty)
        g3 = self.propagation_pending_embedding(propagation_pending)
        g4 = self.stack_depth_embedding(stack_depth)
        global_x = torch.stack([g0, g1, g2, g3, g4], dim=1)

        pos = torch.arange(self.GLOBAL_FEATURES_LEN, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        global_x = global_x + self.global_pos_encoding(pos)

        # --- Transformer ---
        scale = float(self.d_model**0.5)
        x = torch.cat([global_x, node_x], dim=1) * scale
        x = self.dropout(x)

        h = self.transformer(x)
        global_repr = h[:, : self.GLOBAL_FEATURES_LEN, :].mean(dim=1)
        node_repr = h[:, self.GLOBAL_FEATURES_LEN :, :]

        # --- Heads ---
        action_type_logits = self.action_type_head(global_repr)

        g = global_repr.unsqueeze(1).expand(-1, n, -1)
        node_feat = torch.cat([g, node_repr, g * node_repr], dim=-1)
        node_logits = self.node_pointer(node_feat).squeeze(-1)

        # Mask nodes that cannot be selected: assigned or empty domain.
        node_selectable = (~is_assigned) & (domain_size > 0)
        node_logits = node_logits.masked_fill(~node_selectable, -1e9)

        # --- Color head ---
        colors = torch.arange(1, k + 1, device=device, dtype=torch.long)

        assigned_onehot = assigned_color.unsqueeze(-1) == colors.view(1, 1, k)

        # dom_color[b, i, c] = whether color (c+1) appears in node i's domain.
        dom_color = (
            (domain_values.unsqueeze(-1) == colors.view(1, 1, 1, k))
            & domain_mask_bool.unsqueeze(-1)
        ).any(dim=2)

        unassigned = ~is_assigned

        count_assigned = assigned_onehot.sum(dim=1).to(torch.float32)  # (B, K)
        count_available = (dom_color & unassigned.unsqueeze(-1)).sum(dim=1).to(torch.float32)  # (B, K)

        # Normalize by N to keep feature magnitudes stable across graph sizes.
        denom_n = max(1.0, float(n))
        color_feats = torch.stack([count_assigned / denom_n, count_available / denom_n], dim=-1)  # (B, K, 2)

        color_x = self.color_mlp(color_feats)
        color_repr = self.color_transformer(color_x)

        sel_ok = (selected_node >= 0) & (selected_node < n)
        safe_sel = torch.where(sel_ok, selected_node, torch.zeros_like(selected_node))
        sel_repr = node_repr[torch.arange(bsz, device=device), safe_sel]

        s = sel_repr.unsqueeze(1).expand(-1, k, -1)
        color_ptr_feat = torch.cat([s, color_repr, s * color_repr], dim=-1)
        color_logits = self.color_pointer(color_ptr_feat).squeeze(-1)

        dom_color_sel = dom_color[torch.arange(bsz, device=device), safe_sel]
        in_domain = dom_color_sel & sel_ok.unsqueeze(1)
        color_logits = color_logits.masked_fill(~in_domain, -1e9)

        validity_logits = self.validity_head(global_repr)

        return action_type_logits, node_logits, color_logits, validity_logits


if __name__ == "__main__":
    # Smoke test: instantiate and run a forward pass.
    torch.manual_seed(0)

    bsz = 2
    n = 10
    k = 3

    model = GraphColorModel(max_nodes=50, num_colors=k, d_model=64, num_gnn_layers=2, num_transformer_layers=2)

    # Fake graph batch (no padding)
    adj = torch.zeros((bsz, n, n), dtype=torch.bool)
    # make a simple chain graph
    for i in range(n - 1):
        adj[:, i, i + 1] = True
        adj[:, i + 1, i] = True

    # node_features: [idx, degree, assigned_color, is_selected]
    node_features = torch.zeros((bsz, n, 4), dtype=torch.long)
    node_features[:, :, 0] = torch.arange(n).view(1, n)
    node_features[:, :, 1] = adj.sum(dim=-1).long()
    node_features[:, 0, 2] = 1  # assign node 0 color 1
    node_features[:, 0, 3] = 0

    # domains: full domains for all nodes
    domain_values = torch.zeros((bsz, n, k), dtype=torch.long)
    domain_values[:] = torch.arange(1, k + 1).view(1, 1, k)
    domain_mask = torch.ones((bsz, n, k), dtype=torch.bool)

    global_features = torch.tensor(
        [
            [-1, 1, 0, 0, 0],
            [2, 1, 0, 0, 1],
        ],
        dtype=torch.long,
    )

    out = model(
        node_features=node_features,
        adjacency=adj,
        domain_values=domain_values,
        domain_mask=domain_mask,
        global_features=global_features,
    )

    at, node_logits, col_logits, valid = out
    assert at.shape == (bsz, 5)
    assert node_logits.shape == (bsz, n)
    assert col_logits.shape == (bsz, k)
    assert valid.shape == (bsz, 1)

    print("OK - action_type_logits.shape=", tuple(at.shape))
    print("OK - node_logits.shape=", tuple(node_logits.shape))
    print("OK - color_logits.shape=", tuple(col_logits.shape))
    print("OK - validity_logits.shape=", tuple(valid.shape))
