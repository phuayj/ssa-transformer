"""LSTM baseline decoder with Slot/SSA-compatible interface."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LSTMDecoder(nn.Module):
    """Recurrent baseline decoder compatible with SSASlotDecoder API."""

    def __init__(
        self,
        vocab_size: int = 392,
        d_model: int = 256,
        hidden_size: int = 256,
        n_lstm_layers: int = 4,
        max_seq_len: int = 4096,
        n_slots: int = 0,
        dropout: float = 0.1,
        block_mode: str = "continuous",
    ):
        super().__init__()
        if str(block_mode) not in {"continuous", "block_reset"}:
            raise ValueError(
                "block_mode must be one of ['continuous', 'block_reset'], "
                f"got '{block_mode}'"
            )

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.n_lstm_layers = int(n_lstm_layers)
        self.max_seq_len = int(max_seq_len)
        self.n_slots = int(n_slots)
        self.dropout = float(dropout)
        self.block_mode = str(block_mode)

        # Compatibility attributes expected by existing checkpoint/loading code.
        self.n_layers = int(n_lstm_layers)
        self.n_heads = 1

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.embed_drop = nn.Dropout(float(dropout))

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.n_lstm_layers,
            batch_first=True,
            dropout=float(dropout),
        )
        self.post_lstm_ln = nn.LayerNorm(self.hidden_size)
        self.hidden_to_model = (
            nn.Linear(self.hidden_size, self.d_model)
            if int(self.hidden_size) != int(self.d_model)
            else nn.Identity()
        )

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

    def _run_lstm_continuous(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out

    def _run_lstm_block_reset(
        self, x: torch.Tensor, block_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(
            self.n_lstm_layers,
            batch_size,
            self.hidden_size,
            dtype=x.dtype,
            device=device,
        )
        c = torch.zeros(
            self.n_lstm_layers,
            batch_size,
            self.hidden_size,
            dtype=x.dtype,
            device=device,
        )

        outputs = []
        for t in range(int(seq_len)):
            if t > 0:
                prev_ids = block_ids[:, t - 1]
                curr_ids = block_ids[:, t]
                boundary = curr_ids.ne(prev_ids) & curr_ids.gt(0)
                if bool(boundary.any().item()):
                    reset_mask = boundary[None, :, None].to(dtype=x.dtype)
                    keep_mask = 1.0 - reset_mask
                    h = h * keep_mask
                    c = c * keep_mask

            step_out, (h, c) = self.lstm(x[:, t : t + 1, :], (h, c))
            outputs.append(step_out)

        return torch.cat(outputs, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        mask_mode: str = "selective_ssa",
        position_mode: str = "auto",
        window_size: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (lm_logits [B, T, vocab], verify_logits [B, T, 2])."""
        del mask_mode, position_mode, window_size

        batch_size, seq_len = input_ids.shape
        if int(seq_len) > int(self.max_seq_len):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        positions = positions.clamp(max=int(self.max_seq_len) - 1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_drop(x)

        if self.block_mode == "block_reset" and block_ids is not None:
            if tuple(block_ids.shape) != (int(batch_size), int(seq_len)):
                raise ValueError(
                    f"block_ids shape {tuple(block_ids.shape)} must match {(batch_size, seq_len)}"
                )
            block_ids = block_ids.to(device=device, dtype=torch.long)
            lstm_out = self._run_lstm_block_reset(x=x, block_ids=block_ids)
        else:
            lstm_out = self._run_lstm_continuous(x)

        hidden = self.post_lstm_ln(lstm_out)
        seq_out = self.hidden_to_model(hidden)

        lm_logits = self.lm_head(seq_out)
        verify_logits = self.verify_head(seq_out)

        if attention_mask is not None:
            if tuple(attention_mask.shape) != (int(batch_size), int(seq_len)):
                raise ValueError(
                    "attention_mask shape "
                    f"{tuple(attention_mask.shape)} must match {(batch_size, seq_len)}"
                )
            valid = attention_mask.to(device=device, dtype=torch.bool)
            lm_logits = lm_logits.masked_fill(~valid[:, :, None], 0.0)
            verify_logits = verify_logits.masked_fill(~valid[:, :, None], 0.0)

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                cf_probs = F.softmax(verify_logits, dim=-1)[..., 1]
                if attention_mask is not None:
                    valid_mask = attention_mask.to(device=device, dtype=torch.bool)
                    if bool(valid_mask.any().item()):
                        cf_prob = float(
                            cf_probs.masked_select(valid_mask).mean().item()
                        )
                    else:
                        cf_prob = float(cf_probs.mean().item())
                else:
                    cf_prob = float(cf_probs.mean().item())

                if block_ids is not None:
                    block_ids_f = block_ids.to(device=device, dtype=torch.long)
                    transitions = int(
                        (
                            block_ids_f[:, 1:].ne(block_ids_f[:, :-1])
                            & block_ids_f[:, 1:].gt(0)
                        )
                        .sum()
                        .item()
                    )
                else:
                    transitions = 0

                logger.debug(
                    "LSTMDecoder forward: batch=%d seq_len=%d block_mode=%s hidden_size=%d layers=%d lm_mean=%.6f cf_prob=%.6f reset_transitions=%d",
                    int(batch_size),
                    int(seq_len),
                    str(self.block_mode),
                    int(self.hidden_size),
                    int(self.n_lstm_layers),
                    float(lm_logits.mean().item()),
                    float(cf_prob),
                    int(transitions),
                )

        return lm_logits, verify_logits

    def forward_lm_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        mask_mode: str = "selective_ssa",
        position_mode: str = "auto",
        window_size: int = 256,
    ) -> torch.Tensor:
        """Forward pass returning only LM logits."""
        lm_logits, _ = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            block_ids=block_ids,
            mask_mode=mask_mode,
            position_mode=position_mode,
            window_size=window_size,
        )
        return lm_logits
