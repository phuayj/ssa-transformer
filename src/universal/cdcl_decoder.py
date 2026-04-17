"""Small GPT-style decoder-only transformer for CDCL conflict analysis.

Learns to predict conflict resolution (backjump targets) from serialized
graph states, optionally with chain-of-thought scratchpad supervision.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal self-attention with GPT-style masking."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.qkv = nn.Linear(int(d_model), int(3 * d_model))
        self.out_proj = nn.Linear(int(d_model), int(d_model))
        self.attn_drop = nn.Dropout(float(dropout))
        self.resid_drop = nn.Dropout(float(dropout))
        self.max_seq_len = int(max_seq_len)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            full_mask = causal[None, None, :, :] & key_mask
        else:
            full_mask = causal[None, None, :, :]

        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~full_mask, mask_value)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        attn_out = attn_probs @ value
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        )
        attn_out = self.out_proj(attn_out)
        attn_out = self.resid_drop(attn_out)
        return attn_out


class FeedForward(nn.Module):
    """Transformer MLP block with GELU activation."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc_in = nn.Linear(int(d_model), int(4 * d_model))
        self.fc_out = nn.Linear(int(4 * d_model), int(d_model))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc_out(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with causal self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float,
        resid_scale: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(int(d_model))
        self.ln2 = nn.LayerNorm(int(d_model))
        self.attn = CausalSelfAttention(
            d_model=int(d_model),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len),
            dropout=float(dropout),
        )
        self.mlp = FeedForward(d_model=int(d_model), dropout=float(dropout))
        self.resid_scale = float(resid_scale)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.resid_scale * self.attn(self.ln1(x), attention_mask)
        x = x + self.resid_scale * self.mlp(self.ln2(x))
        return x


class CDCLDecoder(nn.Module):
    """GPT-2 style decoder-only transformer.

    Args:
        vocab_size: Size of token vocabulary (default 233 from CDCLTokenizer)
        d_model: Hidden dimension (default 256)
        n_layers: Number of transformer layers (default 6)
        n_heads: Number of attention heads (default 8)
        max_seq_len: Maximum sequence length (default 1024)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        vocab_size: int = 233,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.embed_drop = nn.Dropout(float(dropout))

        resid_scale = 1.0 / math.sqrt(float(self.n_layers))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    max_seq_len=self.max_seq_len,
                    dropout=float(dropout),
                    resid_scale=resid_scale,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [B, T] token IDs
            attention_mask: [B, T] optional padding mask (1=attend, 0=ignore)

        Returns:
            logits: [B, T, vocab_size] next-token prediction logits
        """
        batch_size, seq_len = input_ids.shape
        if int(seq_len) > int(self.max_seq_len):
            raise ValueError("input sequence length exceeds max_seq_len")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_drop(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        stop_token: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation (greedy or with temperature).

        Args:
            input_ids: [B, T] prefix token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: 0.0 for greedy, >0 for sampling
            stop_token: Token ID to stop at (default EOS=2)

        Returns:
            [B, T+generated] full sequence including generated tokens
        """
        if int(max_new_tokens) <= 0:
            return input_ids

        max_allowed = int(self.max_seq_len) - int(input_ids.size(1))
        if max_allowed <= 0:
            return input_ids
        max_steps = min(int(max_new_tokens), int(max_allowed))

        was_training = self.training
        self.eval()

        generated = input_ids
        finished = torch.zeros(
            generated.size(0), dtype=torch.bool, device=generated.device
        )

        for _ in range(max_steps):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :]
            if float(temperature) > 0.0:
                probs = F.softmax(next_logits / float(temperature), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            if stop_token is not None:
                if bool(finished.any()):
                    stop_tokens = torch.full_like(next_token, int(stop_token))
                    next_token = torch.where(finished[:, None], stop_tokens, next_token)
                generated = torch.cat([generated, next_token], dim=1)
                finished = finished | (next_token.squeeze(-1) == int(stop_token))
                if bool(finished.all()):
                    break
            else:
                generated = torch.cat([generated, next_token], dim=1)

        if was_training:
            self.train()

        return generated
