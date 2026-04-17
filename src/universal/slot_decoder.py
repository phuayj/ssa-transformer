"""Slot-augmented CDCL decoder with register tokens and verification head."""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cdcl_decoder import FeedForward, TransformerBlock
from .delta_local_verifier import DeltaLocalVerifyHead

logger = logging.getLogger(__name__)


class SlotAwareSelfAttention(nn.Module):
    """Self-attention that accepts a pre-built 4D attention mask."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float,
        use_sdpa_attention: bool = True,
    ):
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.use_sdpa_attention = bool(use_sdpa_attention)
        self.qkv = nn.Linear(int(d_model), int(3 * d_model))
        self.out_proj = nn.Linear(int(d_model), int(d_model))
        self.attn_drop = nn.Dropout(float(dropout))
        self.resid_drop = nn.Dropout(float(dropout))

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn_dropout = self.attn_drop.p if self.training else 0.0

        if attention_mask is not None and int(attention_mask.dim()) != 4:
            raise ValueError("attention_mask must be 4D [B, 1, T, T]")

        if self.use_sdpa_attention:
            sdpa_mask = (
                attention_mask.to(device=query.device, dtype=torch.bool)
                if attention_mask is not None
                else None
            )
            attn_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=sdpa_mask,
                dropout_p=float(attn_dropout),
            )
        else:
            attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                mask_value = torch.finfo(attn_scores.dtype).min
                attn_scores = attn_scores.masked_fill(
                    ~attention_mask.to(device=query.device, dtype=torch.bool),
                    mask_value,
                )
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_drop(attn_probs)
            attn_out = attn_probs @ value

        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        )
        attn_out = self.out_proj(attn_out)
        attn_out = self.resid_drop(attn_out)
        return attn_out


class SlotTransformerBlock(TransformerBlock):
    """Transformer block using slot-aware attention masks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float,
        resid_scale: float,
        use_sdpa_attention: bool = True,
    ):
        super().__init__(
            d_model=int(d_model),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len),
            dropout=float(dropout),
            resid_scale=float(resid_scale),
        )
        self.attn = SlotAwareSelfAttention(
            d_model=int(d_model),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len),
            dropout=float(dropout),
            use_sdpa_attention=bool(use_sdpa_attention),
        )
        self.mlp = FeedForward(d_model=int(d_model), dropout=float(dropout))


class SlotCDCLDecoder(nn.Module):
    """CDCLDecoder augmented with persistent register/slot tokens.

    Adds R learnable slot tokens prepended to every input. Slots can attend
    to all positions (bidirectional among slots + full sequence). Sequence
    tokens attend to all slots + causally to preceding sequence tokens.

    A verification head reads from sequence positions to predict conflict/ok.

    Args:
        vocab_size: Token vocabulary size
        d_model: Hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        max_seq_len: Maximum SEQUENCE length (excluding slots)
        n_slots: Number of register/slot tokens
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 392,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 4096,
        n_slots: int = 32,
        dropout: float = 0.1,
        use_sdpa_attention: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)
        self.n_slots = int(n_slots)
        self.use_sdpa_attention = bool(use_sdpa_attention)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.embed_drop = nn.Dropout(float(dropout))

        self.slot_embedding = nn.Parameter(
            torch.randn(1, self.n_slots, self.d_model) * 0.02
        )

        resid_scale = 1.0 / math.sqrt(float(self.n_layers))
        total_len = self.n_slots + self.max_seq_len
        self.blocks = nn.ModuleList(
            [
                SlotTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    max_seq_len=total_len,
                    dropout=float(dropout),
                    resid_scale=resid_scale,
                    use_sdpa_attention=bool(self.use_sdpa_attention),
                )
                for _ in range(self.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.verify_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 2),
        )

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the attention mask for [slots | sequence].

        The mask allows:
        - Slots (positions 0..R-1): attend to ALL positions (slots + full sequence)
        - Sequence (positions R..R+T-1): attend to ALL slots + CAUSALLY to sequence

        Returns: [1, 1, R+T, R+T] boolean mask (True = attend, False = mask out)
        """
        slots = self.n_slots
        total = int(slots) + int(seq_len)

        mask = torch.zeros(total, total, dtype=torch.bool, device=device)

        mask[:slots, :] = True
        mask[slots:, :slots] = True
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        mask[slots:, slots:] = causal

        if padding_mask is not None:
            slot_mask = torch.ones(batch_size, slots, dtype=torch.bool, device=device)
            full_padding = torch.cat([slot_mask, padding_mask.to(torch.bool)], dim=1)
            key_mask = full_padding[:, None, None, :]
            return mask[None, None, :, :] & key_mask

        return mask[None, None, :, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [B, T] token IDs (sequence only, no slots)
            attention_mask: [B, T] optional padding mask (1=attend, 0=ignore)

        Returns:
            lm_logits: [B, T, vocab_size] next-token prediction logits
            verify_logits: [B, T, 2] per-position conflict verification logits ([ok, cf])
        """
        batch_size, seq_len = input_ids.shape
        if int(seq_len) > int(self.max_seq_len):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seq_embed = self.token_embedding(input_ids) + self.position_embedding(positions)

        slot_embed = self.slot_embedding.expand(batch_size, -1, -1)
        x = torch.cat([slot_embed, seq_embed], dim=1)
        x = self.embed_drop(x)

        attn_mask = self._build_attention_mask(
            batch_size, seq_len, device, attention_mask
        )

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x)

        seq_out = x[:, self.n_slots :, :]

        lm_logits = self.lm_head(seq_out)

        verify_logits = self.verify_head(seq_out)

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                cf_probs = F.softmax(verify_logits, dim=-1)[..., 1]
                if attention_mask is not None:
                    valid_mask = attention_mask.to(torch.bool)
                    if valid_mask.any():
                        cf_prob = cf_probs.masked_select(valid_mask).mean().item()
                    else:
                        cf_prob = cf_probs.mean().item()
                else:
                    cf_prob = cf_probs.mean().item()
                logger.debug(
                    "SlotCDCLDecoder forward: batch=%d seq_len=%d lm_mean=%.6f cf_prob=%.6f",
                    batch_size,
                    seq_len,
                    lm_logits.mean().item(),
                    cf_prob,
                )

        return lm_logits, verify_logits

    def forward_lm_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning only LM logits (compatible with CDCLDecoder interface)."""
        lm_logits, _ = self.forward(input_ids, attention_mask)
        return lm_logits


class DeltaLocalSlotDecoder(SlotCDCLDecoder):
    """SlotCDCLDecoder with delta-local verification head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_slots: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        n_colors: int = 4,
        max_neighbors: int = 30,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            n_slots=n_slots,
            dropout=dropout,
        )
        self.delta_verify = DeltaLocalVerifyHead(
            d_model=int(d_model),
            n_colors=int(n_colors),
            max_neighbors=int(max_neighbors),
        )

    def _forward_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if int(seq_len) > int(self.max_seq_len):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seq_embed = self.token_embedding(input_ids) + self.position_embedding(positions)

        slot_embed = self.slot_embedding.expand(batch_size, -1, -1)
        x = torch.cat([slot_embed, seq_embed], dim=1)
        x = self.embed_drop(x)

        attn_mask = self._build_attention_mask(
            batch_size, seq_len, device, attention_mask
        )
        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        seq_out = x[:, self.n_slots :, :]
        lm_logits = self.lm_head(seq_out)
        return lm_logits, seq_out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        verify_labels: Optional[torch.Tensor] = None,
        neighbor_positions: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        assign_positions: Optional[torch.Tensor] = None,
        neighbor_labels: Optional[torch.Tensor] = None,
    ):
        del loss_mask, targets

        lm_logits, seq_out = self._forward_backbone(input_ids, attention_mask)

        if neighbor_positions is None or assign_positions is None:
            verify_logits = self.verify_head(seq_out)
            return lm_logits, verify_logits

        batch_size = int(input_ids.size(0))
        seq_len = int(input_ids.size(1))
        assign_positions = assign_positions.to(device=input_ids.device, dtype=torch.long)
        assign_positions = assign_positions.clamp(min=0, max=max(seq_len - 1, 0))
        assign_hidden = seq_out[
            torch.arange(batch_size, device=input_ids.device), assign_positions
        ]

        if neighbor_mask is None:
            neighbor_mask = (neighbor_positions >= 0).to(torch.float32)
        global_logits, neighbor_logits = self.delta_verify(
            seq_out,
            neighbor_positions.to(device=input_ids.device, dtype=torch.long),
            neighbor_mask.to(device=input_ids.device),
            assign_hidden,
        )

        aux_losses = {}
        if verify_labels is not None:
            aux_losses["global_verify_loss"] = F.cross_entropy(
                global_logits, verify_labels.to(input_ids.device, dtype=torch.long)
            )
        if neighbor_labels is not None:
            valid_neighbors = neighbor_mask.to(input_ids.device).to(torch.bool)
            if valid_neighbors.any():
                neighbor_targets = neighbor_labels.to(input_ids.device, dtype=torch.long)
                flat_logits = neighbor_logits.reshape(-1, 2)
                flat_targets = neighbor_targets.reshape(-1)
                flat_valid = valid_neighbors.reshape(-1)
                aux_losses["local_verify_loss"] = F.cross_entropy(
                    flat_logits[flat_valid], flat_targets[flat_valid]
                )
            else:
                aux_losses["local_verify_loss"] = torch.zeros(
                    (), device=input_ids.device, dtype=seq_out.dtype
                )

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                global_probs = F.softmax(global_logits, dim=-1)
                local_cf = F.softmax(neighbor_logits, dim=-1)[..., 1]
                if neighbor_mask is not None:
                    local_active = local_cf[neighbor_mask.to(torch.bool)]
                    mean_local_cf = (
                        float(local_active.mean().item()) if local_active.numel() > 0 else 0.0
                    )
                else:
                    mean_local_cf = float(local_cf.mean().item())
                logger.debug(
                    "DeltaLocalSlotDecoder forward: batch=%d seq_len=%d lm_mean=%.6f global_cf_prob=%.6f local_cf_mean=%.6f",
                    batch_size,
                    seq_len,
                    float(lm_logits.mean().item()),
                    float(global_probs[:, 1].mean().item()),
                    mean_local_cf,
                )

        return lm_logits, global_logits, neighbor_logits, aux_losses
