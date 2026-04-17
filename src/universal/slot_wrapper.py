"""Side-slot memory wrapper for HuggingFace causal LMs.

Adds learnable register slots + verification head to any pretrained
causal language model. Base model weights are frozen. Only slot modules
and verification head are trainable.

Design principles:
- No positional disruption (slots don't shift RoPE)
- Gate-initialized-near-zero (model starts as exact base)
- Upper-half injection (lower layers handle syntax, upper handle semantics)
- Model-agnostic (works with any HF CausalLM)
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class SlotCrossAttention(nn.Module):
    """Cross-attention between slots and token hidden states.

    Can work in two modes:
    - read: slots attend to tokens (slots are Q, tokens are KV)
    - write: tokens attend to slots (tokens are Q, slots are KV)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        qk_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.qk_dim = int(qk_dim) if qk_dim is not None else int(d_model)
        self.n_heads = int(n_heads)
        if self.qk_dim % self.n_heads != 0:
            raise ValueError("qk_dim must be divisible by n_heads")
        self.head_dim = int(self.qk_dim // self.n_heads)

        self.q_proj = nn.Linear(self.d_model, self.qk_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.qk_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.qk_dim, bias=False)
        self.o_proj = nn.Linear(self.qk_dim, self.d_model, bias=False)
        self.dropout = nn.Dropout(float(dropout))

        # Initialize small so initial impact is near-zero
        nn.init.normal_(self.q_proj.weight, std=0.01)
        nn.init.normal_(self.k_proj.weight, std=0.01)
        nn.init.normal_(self.v_proj.weight, std=0.01)
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        query: torch.Tensor,  # [B, Nq, D]
        key_value: torch.Tensor,  # [B, Nkv, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, Nkv] bool
    ) -> torch.Tensor:
        B, Nq, _ = query.shape
        Nkv = key_value.shape[1]

        q = self.q_proj(query).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(key_value)
            .view(B, Nkv, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(key_value)
            .view(B, Nkv, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask: [B, Nkv] -> [B, 1, 1, Nkv]
            mask = attention_mask[:, None, None, :].bool()
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Nq, self.qk_dim)
        return self.o_proj(out)


class SlotModule(nn.Module):
    """Per-layer slot processing module.

    At each hooked layer:
    1. Read: slots attend to token hidden states
    2. Self-attn: slots attend to each other
    3. Write: tokens attend to slots (gated)
    """

    def __init__(
        self, d_model: int, n_heads: int = 8, slot_dim: int = 256, dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.slot_dim = int(slot_dim)

        # Read: slots <- tokens
        self.read_ln = nn.LayerNorm(self.d_model)
        self.read_attn = SlotCrossAttention(
            self.d_model, n_heads, qk_dim=self.slot_dim, dropout=dropout
        )

        # Self-attn: slots <- slots
        self.self_ln = nn.LayerNorm(self.d_model)
        self.self_attn = SlotCrossAttention(
            self.d_model, n_heads, qk_dim=self.slot_dim, dropout=dropout
        )

        # Write: tokens <- slots (gated)
        self.write_ln = nn.LayerNorm(self.d_model)
        self.write_attn = SlotCrossAttention(
            self.d_model, n_heads, qk_dim=self.slot_dim, dropout=dropout
        )

        # Gate logit for write path (initialized negative -> near-zero gate)
        self.write_gate = nn.Parameter(torch.tensor([-10.0]))

        # Slot FFN
        ffn_dim = self.slot_dim * 4
        self.slot_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, self.d_model),
        )
        # Initialize FFN output near zero
        nn.init.zeros_(self.slot_ffn[-1].weight)
        nn.init.zeros_(self.slot_ffn[-1].bias)

    def forward(
        self,
        token_hidden: torch.Tensor,  # [B, T, D]
        slot_hidden: torch.Tensor,  # [B, R, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] for tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process slots and optionally modify token hidden states.

        Returns (updated_token_hidden, updated_slot_hidden).
        """
        # 1. Read: slots attend to tokens
        slot_hidden = slot_hidden + self.read_attn(
            self.read_ln(slot_hidden), token_hidden, attention_mask
        )

        # 2. Self-attention among slots
        slot_hidden = slot_hidden + self.self_attn(
            self.self_ln(slot_hidden), slot_hidden
        )

        # 3. Slot FFN
        slot_hidden = slot_hidden + self.slot_ffn(slot_hidden)

        # 4. Write: tokens attend to slots (gated)
        gate = torch.sigmoid(self.write_gate)
        token_update = self.write_attn(self.write_ln(token_hidden), slot_hidden)
        token_hidden = token_hidden + gate * token_update

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                gate_value = float(gate.item())
                slot_norm = float(slot_hidden.norm(dim=-1).mean().item())
                update_norm = float(token_update.norm(dim=-1).mean().item())
                logger.debug(
                    "SlotModule forward: gate=%.6f slot_norm=%.6f update_norm=%.6f",
                    gate_value,
                    slot_norm,
                    update_norm,
                )

        return token_hidden, slot_hidden


class SlotLMWrapper(nn.Module):
    """Wraps a HuggingFace CausalLM with side-slot memory + verification head.

    Base model weights are FROZEN. Only slot modules and verification head
    are trainable.

    Args:
        base_model: Pretrained HuggingFace CausalLM
        n_slots: Number of register slots
        n_slot_heads: Number of attention heads in slot modules
        slot_dim: Bottleneck dimension for slot attention/FFN
        start_layer: First layer to inject slots (default: num_layers // 2)
        dropout: Dropout rate for slot modules
        slot_mode: Slot persistence variant (normal, no_write, reset_per_layer, shuffled_slots)
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        n_slots: int = 32,
        n_slot_heads: int = 8,
        slot_dim: int = 256,
        start_layer: Optional[int] = None,
        dropout: float = 0.0,
        slot_mode: str = "normal",
    ):
        super().__init__()
        self.base_model = base_model
        self.n_slots = int(n_slots)
        self.slot_dim = int(slot_dim)
        self.slot_mode = str(slot_mode)
        allowed_modes = ["normal", "no_write", "reset_per_layer", "shuffled_slots"]
        if self.slot_mode not in allowed_modes:
            raise ValueError(
                f"Invalid slot_mode '{self.slot_mode}'. Must be one of {allowed_modes}."
            )
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get model config
        config = base_model.config
        # For multimodal models, get the text config
        if hasattr(config, "get_text_config"):
            text_config = config.get_text_config()
        else:
            text_config = config
        d_model = int(text_config.hidden_size)
        num_layers = int(text_config.num_hidden_layers)
        self.d_model = d_model
        self.num_layers = num_layers
        base_param = next(base_model.parameters())
        slot_dtype = base_param.dtype
        self.slot_dtype = slot_dtype

        # Determine which layers get slot modules
        if start_layer is None:
            start_layer = num_layers // 2
        self.start_layer = int(start_layer)
        self.slot_layers = list(range(self.start_layer, self.num_layers))

        # Learnable slot embeddings
        self.slot_embedding = nn.Parameter(
            torch.randn(1, self.n_slots, d_model, dtype=slot_dtype) * 0.02
        )

        # Per-layer slot modules (only for upper layers)
        self.slot_modules = nn.ModuleDict(
            {
                str(layer_idx): SlotModule(
                    d_model, n_slot_heads, slot_dim=self.slot_dim, dropout=dropout
                )
                for layer_idx in self.slot_layers
            }
        )

        # Verification head (reads pooled slots -> binary classification)
        self.verify_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        # Move slot modules to same device as base model
        device = base_param.device
        dtype = base_param.dtype
        self.slot_embedding = nn.Parameter(
            self.slot_embedding.to(device=device, dtype=dtype)
        )
        for module in self.slot_modules.values():
            module.to(device=device, dtype=dtype)
        self.verify_head.to(device=device, dtype=dtype)
        self.slot_dtype = dtype

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "SlotLMWrapper init: device=%s dtype=%s slot_dim=%d slot_mode=%s trainable_params=%d",
                device,
                dtype,
                self.slot_dim,
                self.slot_mode,
                self.num_trainable_params(),
            )

        # Register hooks to intercept layer outputs
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._slot_hidden: Optional[torch.Tensor] = None
        self._current_attention_mask: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on transformer layers to inject slot processing."""
        layers = self._get_transformer_layers()

        for layer_idx in self.slot_layers:
            hook = layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def _get_transformer_layers(self) -> nn.ModuleList:
        """Get the transformer layer list from the base model."""
        # Try common attribute paths
        # Mistral3 multimodal: layers are under model.language_model
        if hasattr(self.base_model, "model"):
            inner = self.base_model.model
            if hasattr(inner, "language_model") and hasattr(
                inner.language_model, "layers"
            ):
                return inner.language_model.layers
        if hasattr(self.base_model, "model") and hasattr(
            self.base_model.model, "layers"
        ):
            return self.base_model.model.layers
        if hasattr(self.base_model, "transformer") and hasattr(
            self.base_model.transformer, "h"
        ):
            return self.base_model.transformer.h
        if hasattr(self.base_model, "gpt_neox") and hasattr(
            self.base_model.gpt_neox, "layers"
        ):
            return self.base_model.gpt_neox.layers
        raise ValueError(f"Cannot find transformer layers in {type(self.base_model)}")

    def _make_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""

        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if self._slot_hidden is None:
                raise RuntimeError("Slot state was not initialized before forward.")

            if self.slot_mode == "reset_per_layer":
                batch_size = int(hidden_states.shape[0])
                self._slot_hidden = self.slot_embedding.expand(batch_size, -1, -1).clone()
                self._slot_hidden = self._slot_hidden.to(hidden_states.device)

            # Process slots
            slot_module = self.slot_modules[str(layer_idx)]
            hidden_states_new, self._slot_hidden = slot_module(
                hidden_states, self._slot_hidden, self._current_attention_mask
            )

            if self.slot_mode == "shuffled_slots":
                perm = torch.randperm(self.n_slots, device=self._slot_hidden.device)
                self._slot_hidden = self._slot_hidden[:, perm, :]

            if self.slot_mode == "no_write":
                if isinstance(output, tuple):
                    return output
                return hidden_states

            # Return modified output
            if isinstance(output, tuple):
                return (hidden_states_new,) + output[1:]
            return hidden_states_new

        return hook

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass with slot processing.

        Args:
            input_ids: [B, T] token IDs
            attention_mask: [B, T] attention mask (1=attend)
            labels: [B, T] optional labels for LM loss

        Returns:
            dict with keys: lm_logits, verify_logits, lm_loss (if labels provided)
        """
        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])
        device = input_ids.device

        # Initialize slot hidden states
        self._slot_hidden = self.slot_embedding.expand(batch_size, -1, -1).clone()
        self._slot_hidden = self._slot_hidden.to(device)
        self._current_attention_mask = attention_mask

        # Run base model (hooks will intercept and process slots)
        base_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Get LM logits from base model output
        if hasattr(base_output, "logits"):
            lm_logits = base_output.logits
        else:
            lm_logits = base_output[0]

        # Verification head: pool slot representations
        slot_pooled = self._slot_hidden.mean(dim=1)  # [B, D]
        verify_logits = self.verify_head(slot_pooled)  # [B, 2]

        # Compute LM loss if labels provided
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                cf_prob = F.softmax(verify_logits, dim=-1)[:, 1].mean().item()
                slot_norm = float(self._slot_hidden.norm(dim=-1).mean().item())
                gate_values = [
                    float(torch.sigmoid(module.write_gate).item())
                    for module in self.slot_modules.values()
                ]
                logger.debug(
                    "SlotLMWrapper forward: batch=%d seq_len=%d lm_mean=%.6f cf_prob=%.6f slot_norm=%.6f slot_mode=%s gates=%s",
                    batch_size,
                    seq_len,
                    float(lm_logits.mean().item()),
                    float(cf_prob),
                    float(slot_norm),
                    self.slot_mode,
                    gate_values,
                )
                if lm_loss is not None:
                    logger.debug("SlotLMWrapper lm_loss=%.6f", float(lm_loss.item()))

        # Clean up hook state
        self._slot_hidden = self._slot_hidden.detach()
        self._current_attention_mask = None

        return {
            "lm_logits": lm_logits,
            "verify_logits": verify_logits,
            "lm_loss": lm_loss,
        }

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return only trainable parameters (slots + verification head)."""
        params: list[nn.Parameter] = []
        params.append(self.slot_embedding)
        for module in self.slot_modules.values():
            params.extend(module.parameters())
        params.extend(self.verify_head.parameters())
        return params

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(int(p.numel()) for p in self.get_trainable_params())

    def num_total_params(self) -> int:
        """Count all parameters including frozen base."""
        return sum(int(p.numel()) for p in self.parameters())

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        n_slots: int = 32,
        n_slot_heads: int = 8,
        slot_dim: int = 256,
        start_layer: Optional[int] = None,
        slot_mode: str = "normal",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
        **model_kwargs,
    ) -> "SlotLMWrapper":
        """Load pretrained model and wrap with slots.

        Example:
            model = SlotLMWrapper.from_pretrained(
                "mistralai/Ministral-8B-Instruct-2410",
                n_slots=32, torch_dtype=torch.bfloat16,
            )
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if getattr(config, "model_type", "") == "mistral3":
            from transformers import Mistral3ForConditionalGeneration

            base_model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                **model_kwargs,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                **model_kwargs,
            )
        wrapper = cls(
            base_model=base_model,
            n_slots=n_slots,
            n_slot_heads=n_slot_heads,
            slot_dim=slot_dim,
            start_layer=start_layer,
            slot_mode=slot_mode,
        )
        # Ensure all trainable modules are on the right device and dtype
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        wrapper.slot_embedding = nn.Parameter(
            wrapper.slot_embedding.to(device=device, dtype=dtype)
        )
        for mod in wrapper.slot_modules.values():
            mod.to(device=device, dtype=dtype)
        wrapper.verify_head.to(device=device, dtype=dtype)
        return wrapper
