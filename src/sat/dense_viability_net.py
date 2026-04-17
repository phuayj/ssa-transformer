from __future__ import annotations

import torch
from torch import nn


class DenseViabilityNet(nn.Module):
    """Dense viability predictor for SAT literals.

    Input: per-literal features [B, N, 2, F]
    Output: per-literal viability logits [B, N, 2]
    """

    def __init__(
        self,
        num_vars: int = 20,
        feature_dim: int = 9,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        n_slots: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if int(num_vars) <= 0:
            raise ValueError("num_vars must be positive")
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if int(d_model) <= 0:
            raise ValueError("d_model must be positive")
        if int(n_heads) <= 0:
            raise ValueError("n_heads must be positive")
        if int(n_layers) <= 0:
            raise ValueError("n_layers must be positive")
        if int(n_slots) < 0:
            raise ValueError("n_slots must be >= 0")

        self.num_vars = int(num_vars)
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)

        self.feature_encoder = nn.Linear(self.feature_dim, self.d_model)
        self.var_embedding = nn.Embedding(self.num_vars, self.d_model)
        self.polarity_embedding = nn.Embedding(2, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(float(dropout))

        if self.n_slots > 0:
            self.slot_embeddings = nn.Parameter(
                torch.randn(self.n_slots, self.d_model) * 0.02
            )
        else:
            self.register_parameter("slot_embeddings", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=4 * self.d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(n_layers),
            norm=nn.LayerNorm(self.d_model),
        )
        self.output_head = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,N,2,F], got shape={tuple(x.shape)}")
        bsz, n_vars, n_polarities, feat_dim = x.shape
        if int(n_polarities) != 2:
            raise ValueError(f"expected polarity axis=2, got {n_polarities}")
        if int(feat_dim) != int(self.feature_dim):
            raise ValueError(
                f"feature dim mismatch: expected {self.feature_dim}, got {feat_dim}"
            )
        if int(n_vars) != int(self.num_vars):
            raise ValueError(
                f"num vars mismatch: expected {self.num_vars}, got {n_vars}"
            )

        tok = self.feature_encoder(x)

        var_ids = torch.arange(n_vars, device=x.device).view(1, n_vars, 1)
        pol_ids = torch.arange(2, device=x.device).view(1, 1, 2)
        tok = (
            tok
            + self.var_embedding(var_ids).expand(bsz, -1, 2, -1)
            + self.polarity_embedding(pol_ids).expand(bsz, n_vars, -1, -1)
        )
        tok = self.dropout(self.input_norm(tok))

        tok = tok.reshape(bsz, n_vars * 2, self.d_model)

        if self.n_slots > 0:
            slots = self.slot_embeddings.unsqueeze(0).expand(bsz, -1, -1)
            tok = torch.cat([slots, tok], dim=1)

        tok = self.encoder(tok)

        if self.n_slots > 0:
            tok = tok[:, self.n_slots :, :]

        logits = self.output_head(tok).squeeze(-1)
        return logits.reshape(bsz, n_vars, 2)


class SharedMLP(nn.Module):
    """Per-literal MLP baseline (no cross-literal attention)."""

    def __init__(self, feature_dim: int = 9, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        if int(n_layers) < 1:
            raise ValueError("n_layers must be >= 1")

        layers: list[nn.Module] = []
        in_dim = int(feature_dim)
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,N,2,F], got shape={tuple(x.shape)}")
        out = self.mlp(x).squeeze(-1)
        return out
