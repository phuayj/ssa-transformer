"""Model definitions for the r^k benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn

from rk_benchmark.generator import BLOCK_END, BLOCK_START


def create_block_attention_mask(
    input_ids: torch.Tensor,
    block_start_token: int = BLOCK_START,
    block_end_token: int = BLOCK_END,
) -> torch.Tensor:
    """Create boolean attention mask for within-block + CLS/global attention.

    Returns a mask with shape (batch, seq_len, seq_len) where True means blocked.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, seq), got {input_ids.shape}")

    batch_size, seq_len = input_ids.shape
    block_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=input_ids.device)

    for b in range(batch_size):
        current_block = 0
        in_block = False
        for i in range(seq_len):
            token = int(input_ids[b, i].item())
            if token == int(block_start_token):
                current_block += 1
                in_block = True

            if in_block:
                block_ids[b, i] = int(current_block)
            else:
                block_ids[b, i] = 0

            if token == int(block_end_token):
                in_block = False

    bi = block_ids.unsqueeze(2)  # (batch, seq, 1)
    bj = block_ids.unsqueeze(1)  # (batch, 1, seq)

    same_block = (bi == bj) & (bi > 0)
    is_global_i = bi == 0
    is_global_j = bj == 0
    can_attend = same_block | is_global_i | is_global_j
    return ~can_attend


class RkTransformer(nn.Module):
    """Encoder-only transformer for k-conjunction classification."""

    def __init__(
        self,
        vocab_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.nhead = int(nhead)
        self.token_embedding = nn.Embedding(int(vocab_size), int(d_model))
        self.position_embedding = nn.Embedding(int(max_seq_len), int(d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
        )
        self.norm = nn.LayerNorm(int(d_model))
        self.classifier = nn.Linear(int(d_model), 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if int(seq_len) > int(self.max_seq_len):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        attn_mask = None
        if block_mask is not None:
            if block_mask.shape != (batch_size, seq_len, seq_len):
                raise ValueError(
                    "block_mask must have shape "
                    f"(batch, seq, seq)=({batch_size}, {seq_len}, {seq_len}), got {tuple(block_mask.shape)}"
                )
            if block_mask.dtype != torch.bool:
                raise ValueError(f"block_mask must be torch.bool, got {block_mask.dtype}")
            attn_mask = torch.zeros_like(block_mask, dtype=hidden.dtype)
            attn_mask.masked_fill_(block_mask, float("-inf"))
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            attn_mask = attn_mask.reshape(batch_size * self.nhead, seq_len, seq_len)

        encoded = self.encoder(hidden, mask=attn_mask, src_key_padding_mask=padding_mask)
        cls_state = self.norm(encoded[:, 0, :])
        logits = self.classifier(cls_state)
        return logits


class RkOracleMLP(nn.Module):
    """MLP that receives pre-extracted per-block features (bypasses retrieval)."""

    def __init__(self, max_k: int = 32, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >=1, got {num_layers}")
        self.max_k = int(max_k)

        layers: list[nn.Module] = []
        in_dim = int(max_k)
        for _ in range(int(num_layers) - 1):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, oracle_features: torch.Tensor) -> torch.Tensor:
        if oracle_features.ndim != 2:
            raise ValueError(
                f"oracle_features must have shape (batch, max_k), got {oracle_features.shape}"
            )
        if int(oracle_features.shape[1]) != int(self.max_k):
            raise ValueError(
                f"Expected feature width {self.max_k}, got {oracle_features.shape[1]}"
            )
        x = oracle_features.float()
        return self.net(x)
