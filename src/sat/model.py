from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SatModelConfig:
    max_vars: int = 50
    max_clauses: int = 200
    activity_bins: int = 16

    d_model: int = 128
    num_gnn_layers: int = 3
    num_transformer_layers: int = 2
    nhead: int = 4
    dropout: float = 0.1


class _UpdateLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_msg = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        x = self.lin_self(h) + self.lin_msg(msg)
        x = self.act(x)
        x = self.dropout(x)
        return self.norm(h + x)


class SatModel(nn.Module):
    """GNN + Transformer model on the SAT factor graph.

    Expected inputs match SatStepDataset.

    Inputs:
      global_features: (B, 5) long
      var_features: (B, N, 5) long
      var_domain_mask: (B, N, 2) bool
      clauses: (B, C, 3) long   (literal encoding ±(var+1))
      clause_features: (B, C, 5) long

    Outputs:
      action_type_logits: (B, 5)
      var_logits: (B, N)
      value_logits: (B, 2)
      validity_logits: (B, 1)
    """

    GLOBAL_FEATURES_LEN = 5
    VAR_FEATURES_LEN = 5
    CLAUSE_FEATURES_LEN = 5

    def __init__(
        self,
        max_vars: int = 50,
        max_clauses: int = 200,
        activity_bins: int = 16,
        d_model: int = 128,
        num_gnn_layers: int = 3,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if int(max_vars) < 1:
            raise ValueError("max_vars must be >= 1")
        if int(max_clauses) < 1:
            raise ValueError("max_clauses must be >= 1")
        if int(activity_bins) < 2:
            raise ValueError("activity_bins must be >= 2")
        if int(d_model) < 1:
            raise ValueError("d_model must be >= 1")
        if int(num_gnn_layers) < 0:
            raise ValueError("num_gnn_layers must be >= 0")
        if int(num_transformer_layers) < 1:
            raise ValueError("num_transformer_layers must be >= 1")
        if int(nhead) < 1:
            raise ValueError("nhead must be >= 1")

        self.max_vars = int(max_vars)
        self.max_clauses = int(max_clauses)
        self.activity_bins = int(activity_bins)
        self.d_model = int(d_model)

        # --- Variable embeddings ---
        self.var_idx_embedding = nn.Embedding(self.max_vars, self.d_model)
        self.var_value_embedding = nn.Embedding(3, self.d_model)  # 0=unassigned,1=false,2=true
        self.var_is_selected_embedding = nn.Embedding(2, self.d_model)
        self.var_domain_size_embedding = nn.Embedding(3, self.d_model)  # 0..2
        self.var_activity_embedding = nn.Embedding(self.activity_bins, self.d_model)

        # --- Clause embeddings ---
        self.clause_idx_embedding = nn.Embedding(self.max_clauses, self.d_model)
        self.clause_satisfied_embedding = nn.Embedding(2, self.d_model)
        self.clause_num_unassigned_embedding = nn.Embedding(4, self.d_model)
        self.clause_num_true_embedding = nn.Embedding(4, self.d_model)
        self.clause_is_conflict_embedding = nn.Embedding(2, self.d_model)

        # --- Global embeddings ---
        self.selected_var_embedding = nn.Embedding(self.max_vars + 1, self.d_model)  # -1..N-1 -> 0..N
        self.num_assigned_embedding = nn.Embedding(self.max_vars + 1, self.d_model)
        self.conflict_embedding = nn.Embedding(2, self.d_model)
        self.propagation_pending_embedding = nn.Embedding(2, self.d_model)
        self.stack_depth_embedding = nn.Embedding(self.max_vars + 1, self.d_model)
        self.global_pos_encoding = nn.Embedding(self.GLOBAL_FEATURES_LEN, self.d_model)

        # --- Factor graph message passing ---
        self.literal_sign_embedding = nn.Embedding(2, self.d_model)  # 0=neg, 1=pos

        self.var_to_clause = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.d_model),
        )
        self.clause_to_var = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.d_model),
        )

        self.var_update_layers = nn.ModuleList([
            _UpdateLayer(self.d_model, float(dropout)) for _ in range(int(num_gnn_layers))
        ])
        self.clause_update_layers = nn.ModuleList([
            _UpdateLayer(self.d_model, float(dropout)) for _ in range(int(num_gnn_layers))
        ])

        self.dropout = nn.Dropout(float(dropout))

        # --- Transformer over [global | var | clause] tokens ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(4 * self.d_model),
            dropout=float(dropout),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(num_transformer_layers))

        # --- Heads ---
        self.action_type_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 5),
        )

        self.var_pointer = nn.Sequential(
            nn.Linear(3 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(3 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 2),
        )

        self.validity_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

    def forward(
        self,
        *,
        global_features: torch.Tensor,
        var_features: torch.Tensor,
        var_domain_mask: torch.Tensor,
        clauses: torch.Tensor,
        clause_features: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if global_features.ndim != 2 or global_features.shape[1] != self.GLOBAL_FEATURES_LEN:
            raise ValueError(
                f"global_features must have shape (batch, {self.GLOBAL_FEATURES_LEN}); got {tuple(global_features.shape)}"
            )
        if var_features.ndim != 3 or var_features.shape[0] != global_features.shape[0]:
            raise ValueError("var_features batch mismatch")
        if var_features.shape[2] != self.VAR_FEATURES_LEN:
            raise ValueError(
                f"var_features must have last dim {self.VAR_FEATURES_LEN}; got {tuple(var_features.shape)}"
            )
        if var_domain_mask.ndim != 3 or var_domain_mask.shape[:2] != var_features.shape[:2] or var_domain_mask.shape[2] != 2:
            raise ValueError("var_domain_mask must have shape (batch, N, 2)")
        if clauses.ndim != 3 or clauses.shape[0] != global_features.shape[0] or clauses.shape[2] != 3:
            raise ValueError("clauses must have shape (batch, C, 3)")
        if clause_features.ndim != 3 or clause_features.shape[0] != global_features.shape[0]:
            raise ValueError("clause_features batch mismatch")
        if clause_features.shape[2] != self.CLAUSE_FEATURES_LEN:
            raise ValueError(
                f"clause_features must have last dim {self.CLAUSE_FEATURES_LEN}; got {tuple(clause_features.shape)}"
            )

        bsz = int(global_features.shape[0])
        n = int(var_features.shape[1])
        c = int(clauses.shape[1])

        if n < 1:
            raise ValueError("N must be >= 1")
        if c < 1:
            raise ValueError("C must be >= 1")
        if n > int(self.max_vars):
            raise ValueError(f"N={n} exceeds max_vars={self.max_vars}")
        if c > int(self.max_clauses):
            raise ValueError(f"C={c} exceeds max_clauses={self.max_clauses}")
        if clause_features.shape[1] != c:
            raise ValueError("clause_features C mismatch")

        device = global_features.device

        # Ensure dtypes.
        global_features = global_features.long()
        var_features = var_features.long()
        clause_features = clause_features.long()
        clauses = clauses.long()
        var_domain_mask_bool = var_domain_mask.bool()

        if clause_mask is None:
            clause_mask_bool = torch.ones((bsz, c), device=device, dtype=torch.bool)
        else:
            if clause_mask.ndim != 2 or tuple(clause_mask.shape) != (bsz, c):
                raise ValueError(
                    f"clause_mask must have shape (batch, C) == ({bsz}, {c}); got {tuple(clause_mask.shape)}"
                )
            clause_mask_bool = clause_mask.bool()

        # --- Parse global features ---
        selected_var = global_features[:, 0]
        num_assigned = global_features[:, 1]
        conflict = global_features[:, 2]
        prop_pending = global_features[:, 3]
        stack_depth = global_features[:, 4]

        if (selected_var < -1).any() or (selected_var > n - 1).any():
            bad = selected_var[(selected_var < -1) | (selected_var > n - 1)][:5].tolist()
            raise ValueError(f"selected_var out of range [-1, {n - 1}]: {bad}")
        if (num_assigned < 0).any() or (num_assigned > n).any():
            bad = num_assigned[(num_assigned < 0) | (num_assigned > n)][:5].tolist()
            raise ValueError(f"num_assigned out of range [0, {n}]: {bad}")
        if ((conflict != 0) & (conflict != 1)).any():
            bad = conflict[((conflict != 0) & (conflict != 1))][:5].tolist()
            raise ValueError(f"conflict must be 0/1; got {bad}")
        if ((prop_pending != 0) & (prop_pending != 1)).any():
            bad = prop_pending[((prop_pending != 0) & (prop_pending != 1))][:5].tolist()
            raise ValueError(f"prop_pending must be 0/1; got {bad}")
        if (stack_depth < 0).any() or (stack_depth > n).any():
            bad = stack_depth[(stack_depth < 0) | (stack_depth > n)][:5].tolist()
            raise ValueError(f"stack_depth out of range [0, {n}]: {bad}")

        # --- Parse var features ---
        var_idx = var_features[:, :, 0]
        var_val_idx = var_features[:, :, 1]
        var_is_sel = var_features[:, :, 2]
        var_dom_size = var_features[:, :, 3]
        var_act_bin = var_features[:, :, 4]

        if (var_idx < 0).any() or (var_idx > n - 1).any():
            bad = var_idx[(var_idx < 0) | (var_idx > n - 1)][:5].tolist()
            raise ValueError(f"var_idx out of range [0, {n - 1}]: {bad}")
        if (var_val_idx < 0).any() or (var_val_idx > 2).any():
            bad = var_val_idx[(var_val_idx < 0) | (var_val_idx > 2)][:5].tolist()
            raise ValueError(f"var_val_idx out of range [0,2]: {bad}")
        if ((var_is_sel != 0) & (var_is_sel != 1)).any():
            bad = var_is_sel[((var_is_sel != 0) & (var_is_sel != 1))][:5].tolist()
            raise ValueError(f"var_is_sel must be 0/1; got {bad}")
        if (var_dom_size < 0).any() or (var_dom_size > 2).any():
            bad = var_dom_size[(var_dom_size < 0) | (var_dom_size > 2)][:5].tolist()
            raise ValueError(f"var_dom_size out of range [0,2]: {bad}")
        if (var_act_bin < 0).any() or (var_act_bin > self.activity_bins - 1).any():
            bad = var_act_bin[(var_act_bin < 0) | (var_act_bin > self.activity_bins - 1)][:5].tolist()
            raise ValueError(f"var_act_bin out of range [0,{self.activity_bins - 1}]: {bad}")

        # --- Parse clause features ---
        clause_idx = clause_features[:, :, 0]
        clause_sat = clause_features[:, :, 1]
        clause_num_unassigned = clause_features[:, :, 2]
        clause_num_true = clause_features[:, :, 3]
        clause_is_conf = clause_features[:, :, 4]

        if (clause_idx < 0).any() or (clause_idx > c - 1).any():
            bad = clause_idx[(clause_idx < 0) | (clause_idx > c - 1)][:5].tolist()
            raise ValueError(f"clause_idx out of range [0, {c - 1}]: {bad}")
        if ((clause_sat != 0) & (clause_sat != 1)).any():
            bad = clause_sat[((clause_sat != 0) & (clause_sat != 1))][:5].tolist()
            raise ValueError(f"clause_sat must be 0/1; got {bad}")
        if (clause_num_unassigned < 0).any() or (clause_num_unassigned > 3).any():
            bad = clause_num_unassigned[(clause_num_unassigned < 0) | (clause_num_unassigned > 3)][:5].tolist()
            raise ValueError(f"clause_num_unassigned out of range [0,3]: {bad}")
        if (clause_num_true < 0).any() or (clause_num_true > 3).any():
            bad = clause_num_true[(clause_num_true < 0) | (clause_num_true > 3)][:5].tolist()
            raise ValueError(f"clause_num_true out of range [0,3]: {bad}")
        if ((clause_is_conf != 0) & (clause_is_conf != 1)).any():
            bad = clause_is_conf[((clause_is_conf != 0) & (clause_is_conf != 1))][:5].tolist()
            raise ValueError(f"clause_is_conf must be 0/1; got {bad}")

        # --- Literal->edge indices ---
        lit_abs = clauses.abs()
        if (lit_abs < 1).any() or (lit_abs > n).any():
            bad = clauses[(lit_abs < 1) | (lit_abs > n)][:5].tolist()
            raise ValueError(f"clauses contain out-of-range literal (abs must be in [1,{n}]): {bad}")

        var_idx_edge = (lit_abs - 1).view(bsz, c * 3)  # (B, E)
        sign_edge = (clauses > 0).long().view(bsz, c * 3)  # (B, E)

        # clause_id per edge (0..C-1 repeated 3 times)
        clause_id = (
            torch.arange(c, device=device, dtype=torch.long)
            .view(1, c, 1)
            .expand(bsz, c, 3)
            .reshape(bsz, c * 3)
        )

        # --- Embeddings ---
        var_x = (
            self.var_idx_embedding(var_idx)
            + self.var_value_embedding(var_val_idx)
            + self.var_is_selected_embedding(var_is_sel)
            + self.var_domain_size_embedding(var_dom_size)
            + self.var_activity_embedding(var_act_bin)
        )

        clause_x = (
            self.clause_idx_embedding(clause_idx)
            + self.clause_satisfied_embedding(clause_sat)
            + self.clause_num_unassigned_embedding(clause_num_unassigned)
            + self.clause_num_true_embedding(clause_num_true)
            + self.clause_is_conflict_embedding(clause_is_conf)
        )

        selected_var_idx = selected_var + 1  # -1..N-1 -> 0..N

        g0 = self.selected_var_embedding(selected_var_idx)
        g1 = self.num_assigned_embedding(num_assigned)
        g2 = self.conflict_embedding(conflict)
        g3 = self.propagation_pending_embedding(prop_pending)
        g4 = self.stack_depth_embedding(stack_depth)

        global_x = torch.stack([g0, g1, g2, g3, g4], dim=1)
        pos = torch.arange(self.GLOBAL_FEATURES_LEN, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        global_x = global_x + self.global_pos_encoding(pos)

        # --- Factor graph message passing ---
        if len(self.var_update_layers) != len(self.clause_update_layers):
            raise RuntimeError("internal error: gnn layer list length mismatch")

        if len(self.var_update_layers) > 0:
            batch_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)

            # Edge mask (B, E): ignore padded clauses.
            edge_mask = clause_mask_bool.view(bsz, c, 1).expand(bsz, c, 3).reshape(bsz, c * 3)
            edge_mask_f = edge_mask.unsqueeze(-1).to(dtype=var_x.dtype)

            for v_layer, c_layer in zip(self.var_update_layers, self.clause_update_layers):
                # Var -> Clause
                var_edge_h = var_x[batch_idx, var_idx_edge]  # (B, E, d)
                sign_h = self.literal_sign_embedding(sign_edge)  # (B, E, d)
                msg_vc = self.var_to_clause(torch.cat([var_edge_h, sign_h], dim=-1))  # (B, E, d)
                msg_vc = msg_vc * edge_mask_f
                msg_c = msg_vc.view(bsz, c, 3, self.d_model).sum(dim=2)  # (B, C, d)
                clause_x = c_layer(clause_x, msg_c)

                # Clause -> Var
                clause_edge_h = clause_x[batch_idx, clause_id]  # (B, E, d)
                msg_cv = self.clause_to_var(torch.cat([clause_edge_h, sign_h], dim=-1))  # (B, E, d)
                msg_cv = msg_cv * edge_mask_f

                agg = torch.zeros((bsz, n, self.d_model), device=device, dtype=msg_cv.dtype)
                idx = var_idx_edge.unsqueeze(-1).expand(-1, -1, self.d_model)
                agg = agg.scatter_add(1, idx, msg_cv)

                cnt = torch.zeros((bsz, n, 1), device=device, dtype=msg_cv.dtype)
                cnt = cnt.scatter_add(
                    1,
                    var_idx_edge.unsqueeze(-1),
                    edge_mask_f,
                )
                agg_mean = agg / cnt.clamp(min=1.0)

                var_x = v_layer(var_x, agg_mean)

        # --- Transformer ---
        scale = float(self.d_model**0.5)
        x = torch.cat([global_x, var_x, clause_x], dim=1) * scale
        x = self.dropout(x)

        src_key_padding_mask: Optional[torch.Tensor] = None
        if clause_mask is not None:
            prefix = torch.zeros((bsz, self.GLOBAL_FEATURES_LEN + n), device=device, dtype=torch.bool)
            src_key_padding_mask = torch.cat([prefix, ~clause_mask_bool], dim=1)

        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        global_repr = h[:, : self.GLOBAL_FEATURES_LEN, :].mean(dim=1)
        var_repr = h[:, self.GLOBAL_FEATURES_LEN : self.GLOBAL_FEATURES_LEN + n, :]

        # --- Heads ---
        action_type_logits = self.action_type_head(global_repr)

        g = global_repr.unsqueeze(1).expand(-1, n, -1)
        ptr_feat = torch.cat([g, var_repr, g * var_repr], dim=-1)
        var_logits = self.var_pointer(ptr_feat).squeeze(-1)

        is_assigned = var_val_idx != 0
        has_domain = var_domain_mask_bool.any(dim=2)
        selectable = (~is_assigned) & has_domain
        var_logits = var_logits.masked_fill(~selectable, -1e9)

        sel_ok = (selected_var >= 0) & (selected_var < n)
        safe_sel = torch.where(sel_ok, selected_var, torch.zeros_like(selected_var))
        sel_repr = var_repr[torch.arange(bsz, device=device), safe_sel]

        val_feat = torch.cat([global_repr, sel_repr, global_repr * sel_repr], dim=-1)
        value_logits = self.value_head(val_feat)

        sel_mask = var_domain_mask_bool[torch.arange(bsz, device=device), safe_sel]
        allowed = sel_mask & sel_ok.unsqueeze(1)
        value_logits = value_logits.masked_fill(~allowed, -1e9)

        validity_logits = self.validity_head(global_repr)

        return action_type_logits, var_logits, value_logits, validity_logits


if __name__ == "__main__":
    # Smoke test: instantiate and run a forward pass.
    torch.manual_seed(0)

    bsz = 2
    n = 20
    c = 60

    model = SatModel(max_vars=50, max_clauses=200, d_model=64, num_gnn_layers=2, num_transformer_layers=2)

    global_features = torch.tensor(
        [
            [-1, 0, 0, 0, 0],
            [3, 5, 0, 0, 2],
        ],
        dtype=torch.long,
    )

    # var_features: [var_idx, assigned_value_idx, is_selected, domain_size, activity_bin]
    var_features = torch.zeros((bsz, n, 5), dtype=torch.long)
    var_features[:, :, 0] = torch.arange(n).view(1, n)
    var_features[:, :, 3] = 2

    var_domain_mask = torch.ones((bsz, n, 2), dtype=torch.bool)

    # Random 3-SAT clauses
    vars_idx = torch.randint(0, n, (bsz, c, 3), dtype=torch.long)
    # Force distinct vars per clause (best-effort for smoke test)
    for b in range(bsz):
        for i in range(c):
            a = vars_idx[b, i].tolist()
            if len(set(a)) < 3:
                vars_idx[b, i, 1] = (vars_idx[b, i, 0] + 1) % n
                vars_idx[b, i, 2] = (vars_idx[b, i, 0] + 2) % n

    signs = torch.randint(0, 2, (bsz, c, 3), dtype=torch.long) * 2 - 1  # {-1,+1}
    clauses = (vars_idx + 1) * signs

    clause_features = torch.zeros((bsz, c, 5), dtype=torch.long)
    clause_features[:, :, 0] = torch.arange(c).view(1, c)

    out = model(
        global_features=global_features,
        var_features=var_features,
        var_domain_mask=var_domain_mask,
        clauses=clauses,
        clause_features=clause_features,
    )

    at, var_logits, val_logits, valid = out
    assert at.shape == (bsz, 5)
    assert var_logits.shape == (bsz, n)
    assert val_logits.shape == (bsz, 2)
    assert valid.shape == (bsz, 1)

    print("OK - action_type_logits.shape=", tuple(at.shape))
    print("OK - var_logits.shape=", tuple(var_logits.shape))
    print("OK - value_logits.shape=", tuple(val_logits.shape))
    print("OK - validity_logits.shape=", tuple(valid.shape))
