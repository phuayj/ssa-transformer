"""Sufficient-State Attention decoder built on top of SlotCDCLDecoder."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .slot_decoder import SlotCDCLDecoder

logger = logging.getLogger(__name__)


class SSASlotDecoder(SlotCDCLDecoder):
    """SlotCDCLDecoder with Sufficient-State Attention (SSA).

    Restricts attention so each decision block only sees slots + graph prefix +
    its own (causal) tokens. Cross-block information must flow via slots.
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
        cbv_enabled: bool = False,
        n_branch_slots: int = 12,
        n_verifier_slots: int = 8,
    ):
        super().__init__(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len),
            n_slots=int(n_slots),
            dropout=float(dropout),
            use_sdpa_attention=bool(use_sdpa_attention),
        )
        self.cbv_enabled = bool(cbv_enabled)
        self.n_branch_slots = int(n_branch_slots)
        self.n_verifier_slots = int(n_verifier_slots)

        if self.cbv_enabled:
            if int(self.n_slots) != (2 * self.n_branch_slots + self.n_verifier_slots):
                raise ValueError(
                    "When cbv_enabled=True, n_slots must equal "
                    "2 * n_branch_slots + n_verifier_slots"
                )
            if self.n_verifier_slots <= 0:
                raise ValueError("n_verifier_slots must be positive when cbv_enabled=True")

            self.polarity_embedding = nn.Embedding(2, int(self.d_model))
            # CBV branch-specific readout heads.
            # Keep parent-class verify_head untouched for non-CBV paths.
            self.verify_head_t = nn.Sequential(
                nn.Linear(2 * int(self.d_model), int(self.d_model)),
                nn.GELU(),
                nn.Linear(int(self.d_model), 1),
            )
            self.verify_head_f = nn.Sequential(
                nn.Linear(2 * int(self.d_model), int(self.d_model)),
                nn.GELU(),
                nn.Linear(int(self.d_model), 1),
            )
            self._init_weights(self.polarity_embedding)
            self.verify_head_t.apply(self._init_weights)
            self.verify_head_f.apply(self._init_weights)

    def _build_slot_to_slot_mask(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build [B, n_slots, n_slots] slot visibility mask."""
        slots = int(self.n_slots)
        if not self.cbv_enabled:
            return torch.ones(batch_size, slots, slots, dtype=torch.bool, device=device)

        slot_mask = torch.zeros(slots, slots, dtype=torch.bool, device=device)
        bt = int(self.n_branch_slots)
        bf_start = bt
        bf_end = 2 * bt
        v_start = bf_end

        # Branch-T sees Branch-T + Verifier
        slot_mask[:bt, :bt] = True
        slot_mask[:bt, v_start:] = True

        # Branch-F sees Branch-F + Verifier
        slot_mask[bf_start:bf_end, bf_start:bf_end] = True
        slot_mask[bf_start:bf_end, v_start:] = True

        # Verifier sees all slots
        slot_mask[v_start:, :] = True
        return slot_mask.unsqueeze(0).expand(int(batch_size), -1, -1)

    def _apply_cbv_polarity(self, slot_states: torch.Tensor) -> torch.Tensor:
        """Apply CBV polarity bias to branch slots only."""
        if not self.cbv_enabled:
            return slot_states

        slot_states = slot_states.clone()
        bt = int(self.n_branch_slots)
        bf_start = bt
        bf_end = 2 * bt

        true_bias = self.polarity_embedding(
            torch.zeros(1, dtype=torch.long, device=slot_states.device)
        ).view(1, 1, -1)
        false_bias = self.polarity_embedding(
            torch.ones(1, dtype=torch.long, device=slot_states.device)
        ).view(1, 1, -1)

        slot_states[:, :bt, :] = slot_states[:, :bt, :] + true_bias
        slot_states[:, bf_start:bf_end, :] = slot_states[:, bf_start:bf_end, :] + false_bias
        return slot_states

    def get_verify_logits(self, slot_hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract CBV verification logits from final-layer branch/verifier slot states.

        Args:
            slot_hidden_states: [batch, n_slots, d_model]

        Returns:
            [batch, 2] logits for (is_extendable_T, is_extendable_F)
        """
        if not self.cbv_enabled:
            raise RuntimeError("CBV not enabled")

        bt = int(self.n_branch_slots)

        t_slots = slot_hidden_states[:, :bt, :]
        f_slots = slot_hidden_states[:, bt : 2 * bt, :]
        v_slots = slot_hidden_states[:, 2 * bt :, :]

        t_pool = t_slots.mean(dim=1)
        f_pool = f_slots.mean(dim=1)
        v_pool = v_slots.mean(dim=1)

        z_t = torch.cat([t_pool, v_pool], dim=-1)
        z_f = torch.cat([f_pool, v_pool], dim=-1)

        logit_t = self.verify_head_t(z_t)
        logit_f = self.verify_head_f(z_f)
        return torch.cat([logit_t, logit_f], dim=-1)

    def _build_ssa_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        block_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build SSA mask for [slots | sequence].

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            device: Tensor device.
            block_ids: [B, T] integer block IDs (0=graph prefix, >0 decision blocks).
            padding_mask: [B, T] optional key padding mask.

        Returns:
            [B, 1, R+T, R+T] bool attention mask.
        """
        if tuple(block_ids.shape) != (int(batch_size), int(seq_len)):
            raise ValueError(
                f"block_ids shape {tuple(block_ids.shape)} must match {(batch_size, seq_len)}"
            )

        slots = int(self.n_slots)
        total = slots + int(seq_len)
        block_ids = block_ids.to(device=device, dtype=torch.long)

        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)

        # slots -> slots (dense by default, CBV-isolated when enabled)
        mask[:, :slots, :slots] = self._build_slot_to_slot_mask(
            batch_size=batch_size,
            device=device,
        )

        # slot visibility to graph prefix only (no decision-block history channel)
        graph_prefix = block_ids.eq(0)  # [B, T]
        mask[:, :slots, slots:] = graph_prefix[:, None, :]

        # sequence -> slots always visible
        mask[:, slots:, :slots] = True

        # sequence -> sequence
        # - graph-prefix queries: causal inside graph-prefix only
        # - block queries: full graph-prefix + causal inside same block
        query_block = block_ids[:, :, None]  # [B, T, 1]
        key_block = block_ids[:, None, :]  # [B, 1, T]
        same_block = query_block.eq(key_block)

        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        causal = causal[None, :, :]  # [1, T, T]
        same_block_causal = same_block & causal

        query_is_graph = query_block.eq(0)
        key_is_graph = key_block.eq(0)

        seq_to_seq = torch.where(
            query_is_graph,
            same_block_causal,
            key_is_graph | same_block_causal,
        )
        mask[:, slots:, slots:] = seq_to_seq

        if padding_mask is not None:
            if tuple(padding_mask.shape) != (int(batch_size), int(seq_len)):
                raise ValueError(
                    "padding_mask shape "
                    f"{tuple(padding_mask.shape)} must match {(batch_size, seq_len)}"
                )
            slot_mask = torch.ones(batch_size, slots, dtype=torch.bool, device=device)
            full_padding = torch.cat(
                [slot_mask, padding_mask.to(device=device, dtype=torch.bool)], dim=1
            )
            mask = mask & full_padding[:, None, :]

        return mask[:, None, :, :]

    def _build_swa_prefix_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        block_ids: torch.Tensor,
        window_size: int = 256,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build SWA+prefix mask for [slots | sequence].

        Slots use standard dense visibility. Sequence queries can always read:
        - any graph-prefix key token (block_id == 0)
        - sequence keys within the last `window_size` positions (causal window)
        """
        if tuple(block_ids.shape) != (int(batch_size), int(seq_len)):
            raise ValueError(
                f"block_ids shape {tuple(block_ids.shape)} must match {(batch_size, seq_len)}"
            )

        slots = int(self.n_slots)
        total = slots + int(seq_len)
        block_ids = block_ids.to(device=device, dtype=torch.long)
        window = max(1, int(window_size))

        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)

        # slots -> sequence all (same as full causal baseline)
        mask[:, :slots, slots:] = True
        # slots -> slots (dense by default, CBV-isolated when enabled)
        mask[:, :slots, :slots] = self._build_slot_to_slot_mask(
            batch_size=batch_size,
            device=device,
        )
        # sequence -> slots always visible
        mask[:, slots:, :slots] = True

        # sequence -> sequence: (prefix always visible) OR (causal window)
        is_prefix = block_ids.eq(0)  # [B, T]
        prefix_visible = is_prefix[:, None, :].expand(-1, seq_len, -1)  # [B, T, T]

        positions = torch.arange(seq_len, device=device)
        dist = positions[:, None] - positions[None, :]  # dist[i, j] = i - j
        in_window = (dist >= 0) & (dist < window)  # [T, T]
        in_window = in_window[None, :, :].expand(batch_size, -1, -1)

        seq_visible = prefix_visible | in_window
        mask[:, slots:, slots:] = seq_visible

        if padding_mask is not None:
            if tuple(padding_mask.shape) != (int(batch_size), int(seq_len)):
                raise ValueError(
                    "padding_mask shape "
                    f"{tuple(padding_mask.shape)} must match {(batch_size, seq_len)}"
                )
            slot_mask = torch.ones(batch_size, slots, dtype=torch.bool, device=device)
            full_padding = torch.cat(
                [slot_mask, padding_mask.to(device=device, dtype=torch.bool)], dim=1
            )
            mask = mask & full_padding[:, None, :]

        return mask[:, None, :, :]

    def _build_ablation_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        block_ids: torch.Tensor,
        mask_mode: str,
        padding_mask: Optional[torch.Tensor] = None,
        rng_seed: int = 0,
        window_size: int = 256,
    ) -> torch.Tensor:
        """Build ablation masks for SSA variants on [slots | sequence]."""
        valid_modes = {
            "full_causal",
            "selective_ssa",
            "blanket_ssa",
            "local_block_only",
            "reverse_selective",
            "random_matched",
            "swa_prefix",
        }
        if str(mask_mode) not in valid_modes:
            raise ValueError(
                f"unknown mask_mode='{mask_mode}', expected one of {sorted(valid_modes)}"
            )

        if str(mask_mode) == "full_causal":
            return super()._build_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                padding_mask=padding_mask,
            )

        if str(mask_mode) == "selective_ssa":
            return self._build_ssa_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                block_ids=block_ids,
                padding_mask=padding_mask,
            )

        if str(mask_mode) == "swa_prefix":
            return self._build_swa_prefix_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                block_ids=block_ids,
                window_size=int(window_size),
                padding_mask=padding_mask,
            )

        if tuple(block_ids.shape) != (int(batch_size), int(seq_len)):
            raise ValueError(
                f"block_ids shape {tuple(block_ids.shape)} must match {(batch_size, seq_len)}"
            )

        slots = int(self.n_slots)
        total = slots + int(seq_len)
        block_ids = block_ids.to(device=device, dtype=torch.long)

        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)

        # slots -> slots always dense (or CBV-isolated)
        mask[:, :slots, :slots] = self._build_slot_to_slot_mask(
            batch_size=batch_size,
            device=device,
        )
        # sequence -> slots always visible
        mask[:, slots:, :slots] = True

        query_block = block_ids[:, :, None]  # [B, T, 1]
        key_block = block_ids[:, None, :]  # [B, 1, T]
        same_block = query_block.eq(key_block)
        graph_prefix = block_ids.eq(0)

        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )[None, :, :]

        if str(mask_mode) == "blanket_ssa":
            # Block-isolated blanket SSA:
            # - slots cannot aggregate search trajectories (only graph-prefix channel)
            # - seq->seq is causal within the same block only (graph included as block 0)
            mask[:, :slots, slots:] = graph_prefix[:, None, :]
            mask[:, slots:, slots:] = same_block & causal
        elif str(mask_mode) == "local_block_only":
            # LBM hierarchy: LBM < PBM (prefix+block, no slots) < SSA (prefix+block+slots).
            # Local-block-only keeps slot-to-slot + slot-to-prefix connectivity and prefix LM
            # training intact, but decision-block tokens are isolated to causal attention within
            # their own block only: no prefix, no slots, no cross-block history.
            mask[:, :slots, slots:] = graph_prefix[:, None, :]
            query_is_graph = query_block.eq(0)
            mask[:, slots:, :slots] = query_is_graph.expand(-1, -1, slots)
            mask[:, slots:, slots:] = same_block & causal
        elif str(mask_mode) == "reverse_selective":
            # Reverse selective:
            # - slots read graph prefix only (same as selective SSA)
            # - graph-prefix queries: causal inside graph prefix only
            # - search queries: causal over prior search tokens, blocked from graph prefix
            key_is_graph = key_block.eq(0)
            query_is_graph = query_block.eq(0)
            query_is_search = query_block.gt(0)
            key_is_search = key_block.gt(0)
            mask[:, :slots, slots:] = graph_prefix[:, None, :]
            graph_causal = key_is_graph & causal
            search_causal = query_is_search & key_is_search & causal
            seq_to_seq = torch.where(query_is_graph, graph_causal, search_causal)
            mask[:, slots:, slots:] = seq_to_seq
        else:
            # random_matched:
            # Match per-row visible count from selective SSA while respecting base causality.
            selective = self._build_ssa_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                block_ids=block_ids,
                padding_mask=padding_mask,
            )[:, 0, :, :]

            causal_base = super()._build_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                padding_mask=padding_mask,
            )[:, 0, :, :]

            row_visible = selective.sum(dim=-1)
            sampled = torch.zeros_like(selective)

            generator: Optional[torch.Generator] = None
            if int(rng_seed) != 0:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(rng_seed))

            for b in range(int(batch_size)):
                for r in range(int(total)):
                    candidates = torch.nonzero(causal_base[b, r], as_tuple=False).squeeze(-1)
                    keep = int(row_visible[b, r].item())
                    if keep <= 0 or int(candidates.numel()) == 0:
                        continue
                    keep = min(keep, int(candidates.numel()))
                    if keep == int(candidates.numel()):
                        sampled[b, r, candidates] = True
                        continue
                    perm = torch.randperm(
                        int(candidates.numel()), device=device, generator=generator
                    )
                    picked = candidates[perm[:keep]]
                    sampled[b, r, picked] = True

            return sampled[:, None, :, :]

        if padding_mask is not None:
            if tuple(padding_mask.shape) != (int(batch_size), int(seq_len)):
                raise ValueError(
                    "padding_mask shape "
                    f"{tuple(padding_mask.shape)} must match {(batch_size, seq_len)}"
                )
            slot_mask = torch.ones(batch_size, slots, dtype=torch.bool, device=device)
            full_padding = torch.cat(
                [slot_mask, padding_mask.to(device=device, dtype=torch.bool)], dim=1
            )
            mask = mask & full_padding[:, None, :]

        return mask[:, None, :, :]

    def _build_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        padding_mask: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        mask_mode: str = "selective_ssa",
        window_size: int = 256,
    ) -> torch.Tensor:
        """Build standard causal, selective SSA, SWA-prefix, or ablation attention mask."""
        if block_ids is None:
            mask = super()._build_attention_mask(
                batch_size, seq_len, device, padding_mask
            )
            if not self.cbv_enabled:
                return mask
            if int(mask.size(0)) == 1 and int(batch_size) > 1:
                mask = mask.expand(int(batch_size), -1, -1, -1)
            mask = mask.clone()
            mask[:, 0, : self.n_slots, : self.n_slots] = self._build_slot_to_slot_mask(
                batch_size=batch_size,
                device=device,
            )
            return mask
        if str(mask_mode) == "selective_ssa":
            return self._build_ssa_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                block_ids=block_ids,
                padding_mask=padding_mask,
            )
        if str(mask_mode) == "swa_prefix":
            return self._build_swa_prefix_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                block_ids=block_ids,
                window_size=int(window_size),
                padding_mask=padding_mask,
            )
        return self._build_ablation_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            block_ids=block_ids,
            mask_mode=str(mask_mode),
            padding_mask=padding_mask,
            window_size=int(window_size),
        )

    def _compute_block_relative_positions(
        self, block_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute block-relative position IDs [B, T].

        Graph prefix (block_id==0): positions 0..P-1.
        Any decision block (block_id>0): positions P, P+1, ... within that block.
        """
        if block_ids.dim() != 2:
            raise ValueError("block_ids must be rank-2 [B, T]")

        block_ids = block_ids.to(dtype=torch.long)
        batch_size, seq_len = block_ids.shape
        device = block_ids.device

        graph_mask = block_ids.eq(0)
        graph_prefix_len = graph_mask.sum(dim=1, keepdim=True)  # [B, 1]

        # 0-based rank within the same block using block-equality + causal triangle.
        query_block = block_ids[:, :, None]  # [B, T, 1]
        key_block = block_ids[:, None, :]  # [B, 1, T]
        same_block = query_block.eq(key_block)
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )[None, :, :]
        within_block_rank = (same_block & causal).sum(dim=-1) - 1  # [B, T]

        positions = torch.where(
            graph_mask,
            within_block_rank,
            graph_prefix_len + within_block_rank,
        )
        positions = positions.clamp(min=0, max=int(self.max_seq_len) - 1)
        return positions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        mask_mode: str = "selective_ssa",
        position_mode: str = "auto",
        window_size: int = 256,
        return_slot_states: bool = False,
        return_hidden_states: bool = False,
    ):
        """Forward pass with independently selectable mask/position modes.

        CBV behavior is active only when ``cbv_enabled=True``.
        """
        position_mode = str(position_mode)
        valid_position_modes = {"auto", "standard", "block_relative"}
        if position_mode not in valid_position_modes:
            raise ValueError(
                f"unknown position_mode='{position_mode}', expected one of {sorted(valid_position_modes)}"
            )

        batch_size, seq_len = input_ids.shape
        if int(seq_len) > int(self.max_seq_len):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        if block_ids is not None and tuple(block_ids.shape) != (int(batch_size), int(seq_len)):
            raise ValueError(
                f"block_ids shape {tuple(block_ids.shape)} must match {(batch_size, seq_len)}"
            )

        device = input_ids.device
        if block_ids is not None:
            block_ids = block_ids.to(device=device, dtype=torch.long)

        if block_ids is None:
            if position_mode == "block_relative":
                raise ValueError("position_mode='block_relative' requires block_ids")
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            positions = positions.clamp(max=int(self.max_seq_len) - 1)
        else:
            if position_mode == "standard":
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                positions = positions.clamp(max=int(self.max_seq_len) - 1)
            else:
                positions = self._compute_block_relative_positions(block_ids).to(device=device)

        seq_embed = self.token_embedding(input_ids) + self.position_embedding(positions)

        slot_embed = self.slot_embedding.expand(batch_size, -1, -1)
        slot_embed = self._apply_cbv_polarity(slot_embed)
        x = torch.cat([slot_embed, seq_embed], dim=1)
        x = self.embed_drop(x)

        attn_mask = self._build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            padding_mask=attention_mask,
            block_ids=block_ids,
            mask_mode=str(mask_mode),
            window_size=int(window_size),
        )

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        slot_out = x[:, : self.n_slots, :]
        seq_out = x[:, self.n_slots :, :]

        lm_logits = self.lm_head(seq_out)
        verify_logits = (
            self.get_verify_logits(slot_out)
            if self.cbv_enabled
            else self.verify_head(seq_out)
        )

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                cf_probs = F.softmax(verify_logits, dim=-1)
                if int(verify_logits.dim()) == 3:
                    cf_values = cf_probs[..., 1]
                    if attention_mask is not None:
                        valid_mask = attention_mask.to(device=device, dtype=torch.bool)
                        cf_prob = (
                            cf_values.masked_select(valid_mask).mean().item()
                            if valid_mask.any()
                            else cf_values.mean().item()
                        )
                    else:
                        cf_prob = cf_values.mean().item()
                else:
                    # CBV returns [B, 2] where [:,0]=extendable_T and [:,1]=extendable_F
                    mean_extend_t = float(cf_probs[:, 0].mean().item())
                    mean_extend_f = float(cf_probs[:, 1].mean().item())
                    cf_prob = mean_extend_f
                    logger.debug(
                        "SSASlotDecoder CBV verify stats: batch=%d seq_len=%d mean_extendable_t=%.6f mean_extendable_f=%.6f slot_mean=%.6f",
                        batch_size,
                        seq_len,
                        mean_extend_t,
                        mean_extend_f,
                        float(slot_out.mean().item()),
                    )

                if block_ids is not None:
                    graph_tokens = block_ids.eq(0).sum(dim=1).to(torch.float32)
                    mean_graph_tokens = float(graph_tokens.mean().item())
                    n_blocks = (block_ids.max(dim=1).values + 1).to(torch.float32)
                    mean_blocks = float(n_blocks.mean().item())
                else:
                    mean_graph_tokens = float(seq_len)
                    mean_blocks = 1.0
                logger.debug(
                    "SSASlotDecoder forward: batch=%d seq_len=%d cbv_enabled=%s mask_mode=%s position_mode=%s window_size=%d blocks_mean=%.2f graph_tokens_mean=%.2f lm_mean=%.6f verify_stat=%.6f",
                    batch_size,
                    seq_len,
                    str(self.cbv_enabled),
                    str(mask_mode),
                    str(position_mode),
                    int(window_size),
                    mean_blocks,
                    mean_graph_tokens,
                    float(lm_logits.mean().item()),
                    float(cf_prob),
                )

        if return_slot_states:
            if return_hidden_states:
                return lm_logits, verify_logits, slot_out, seq_out
            return lm_logits, verify_logits, slot_out
        if return_hidden_states:
            return lm_logits, verify_logits, seq_out
        return lm_logits, verify_logits
