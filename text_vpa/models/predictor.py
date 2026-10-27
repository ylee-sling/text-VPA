# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
predictor.py -- pass 3 of PR-JEPA: the JEPA latent predictor g_phi (Sec. 3.1, Eqs. 3 and 8).

    Z_target = s_hat_y = g_phi(s_x, u, Delta) in R^{M x d}                                   (Eq. 3)

g_phi is a shared transformer trunk followed by n_ens lightweight heads, in the manner of deep
ensembles (Sec. 3.1, "Epistemic uncertainty"). The plan is the head mean, the uncertainty their
RMS spread:

    Z_target  = (1/n_ens) sum_e s_hat^(e)                                                    (Eq. 8)
    sigma_bar = ( (1/(n_ens M d)) sum_e || s_hat^(e) - Z_target ||_F^2 )^(1/2)             (Eq. 8)

Inputs:
    s_x    context latents from f_theta (pass 1)                         [B, M_c, d]
    u      mode from pi_phi^h (pass 2): integer ids, or a (straight-     [B] int64, or [B, |U|] float
           through) one-hot / soft vector, contracted with ModeEmbed
    Delta  learnable slot queries, a parameter in R^{M x d}; ``slot_ids``  [B, M'] int64 (optional)
           selects and orders a subset of them per sample (AST positions)

Trunk: slot query m of sample b is Delta_m + ModeEmbed(u_b); L pre-norm blocks of [self-attention
over the slots, cross-attention to the layer-normed s_x, GELU MLP]; a final LayerNorm.
Heads: n_ens independently initialised linear maps d -> d (``head_hidden_dim`` turns them into
two-layer GELU MLPs). All heads run as one batched product (``EnsembleLinear``), so the whole
ensemble is a single network evaluation: pass 3 of the K + 3 chain (Algorithm 1, line 3).

Variable-size plans and prompts use boolean masks (True = real slot / token). With masks, the M of
Eq. 8 becomes the number of real slots M_b of sample b, padded slots are excluded from every key
set, and padded outputs are exact zeros.

Decisions the paper leaves open (CLAUDE.md, "Decisions"): ModeEmbed belongs to g_phi (the bridge's
e(u) is a separate table); the mean is computed relative to head 0, so identical heads give
sigma_bar = 0 exactly (as in VPA); sigma_bar is not divided by a latent scale (VPA divided by
sqrt(d) gamma*), since Eq. 8 is a plain RMS and Sec. 4.5 calibrates it on held-out rollouts.
sigma_bar is a diagnostic, not a training signal: its square root has an unbounded derivative
where all heads agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from text_vpa.models.layers import EnsembleLinear, TransformerBlock, embed_modes

if TYPE_CHECKING:
    from text_vpa.configs.schema import PlannerConfig

__all__ = ["JEPAPredictor", "PredictorOutput", "ensemble_mean", "ensemble_uncertainty"]


class PredictorOutput(NamedTuple):
    """Output of ``JEPAPredictor.forward``."""

    z_target: Tensor          # [B, M, d]         plan Z_target (ensemble mean), zeros at padded slots
    sigma_bar: Tensor         # [B]               RMS ensemble spread (Eq. 8), >= 0
    head_predictions: Tensor  # [n_ens, B, M, d]  per-head plans s_hat^(e), zeros at padded slots
    slot_mask: Tensor         # [B, M] bool       True at real slots


def ensemble_mean(head_predictions: Tensor) -> Tensor:
    """Z_target = (1/n_ens) sum_e s_hat^(e)   (Eq. 8).

    Evaluated as s_hat^(0) + (1/n_ens) sum_e (s_hat^(e) - s_hat^(0)): the same mean in exact
    arithmetic, but bit-exact when all heads agree, so sigma_bar is then exactly 0.

    Shapes: ``[n_ens, *, d]`` -> ``[*, d]``.
    """
    if head_predictions.dim() < 2:
        raise ValueError(f"expected [n_ens, ..., d], got {tuple(head_predictions.shape)}")
    return head_predictions[0] + (head_predictions - head_predictions[0:1]).mean(dim=0)


def ensemble_uncertainty(head_predictions: Tensor, z_target: Tensor, slot_mask: Tensor | None = None) -> Tensor:
    """sigma_bar_b = ( sum_e sum_{m real} ||s_hat^(e)_bm - Z_bm||_2^2 / (n_ens M_b d) )^(1/2)   (Eq. 8).

    Without a mask, M_b = M for every sample and this is Eq. 8 verbatim.

    Args:
        head_predictions: ``[n_ens, B, M, d]``.
        z_target: ``[B, M, d]``, the ensemble mean.
        slot_mask: ``[B, M]`` bool (True = real slot) or None.

    Returns:
        ``[B]``, non-negative.
    """
    if head_predictions.dim() != 4 or head_predictions.shape[1:] != z_target.shape:
        raise ValueError(f"expected head_predictions [n_ens, B, M, d] and z_target [B, M, d], got "
                         f"{tuple(head_predictions.shape)} and {tuple(z_target.shape)}")
    n_ens, b, m, d = head_predictions.shape
    if m == 0:
        raise ValueError("Eq. 8 is undefined for M = 0 slots")
    sq = (head_predictions - z_target.unsqueeze(0)).pow(2).sum(dim=(0, 3))      # [B, M]
    if slot_mask is None:
        count = torch.full((b,), float(m), dtype=sq.dtype, device=sq.device)  # M_b = M
    else:
        if slot_mask.dtype != torch.bool or slot_mask.shape != (b, m):
            raise ValueError(f"expected a bool slot_mask of shape [{b}, {m}], got {tuple(slot_mask.shape)}")
        if not bool(slot_mask.any(dim=1).all()):
            raise ValueError("Eq. 8 is undefined for a sample without real slots (M_b = 0)")
        sq = sq.masked_fill(~slot_mask, 0.0)
        count = slot_mask.sum(dim=1).to(sq.dtype)                            # M_b
    return torch.sqrt(sq.sum(dim=1) / (n_ens * d * count))


class JEPAPredictor(nn.Module):
    """g_phi(s_x, u, Delta): shared trunk plus n_ens ensemble heads (Eqs. 3, 8).

    Args:
        latent_dim: d (width of s_x, s_y and Z_target).
        num_modes: |U|.
        num_slots: M, rows of the learnable slot-query matrix Delta.
        n_ens: ensemble heads (>= 2).
        num_layers, num_heads, mlp_ratio, dropout: trunk hyperparameters.
        head_hidden_dim: None for linear heads, else the hidden width of MLP heads.

    Shapes:
        ``forward_heads(s_x [B, M_c, d], u [B] or [B, |U|], slot_ids [B, M'] or None,
        context_mask [B, M_c] or None, slot_mask [B, M'] or None)`` -> ``[n_ens, B, M', d]``,
        where M' = M when ``slot_ids`` is None.
        ``forward(...)`` -> ``PredictorOutput``.
    """

    def __init__(
        self,
        latent_dim: int,
        num_modes: int,
        num_slots: int,
        n_ens: int = 5,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        head_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_ens < 2:
            raise ValueError("n_ens must be >= 2: sigma_bar (Eq. 8) needs an ensemble spread")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_modes < 1 or num_slots < 1:
            raise ValueError("num_modes and num_slots must be >= 1")
        self.latent_dim: int = latent_dim
        self.num_modes: int = num_modes
        self.num_slots: int = num_slots
        self.n_ens: int = n_ens
        d = latent_dim
        self.context_norm = nn.LayerNorm(d)
        self.slot_queries = nn.Parameter(torch.empty(num_slots, d))                 # Delta in R^{M x d}
        self.mode_embedding = nn.Embedding(num_modes, d)                           # g_phi's own table
        self.blocks = nn.ModuleList(
            TransformerBlock(d, num_heads, mlp_ratio, dropout, cross_attention=True) for _ in range(num_layers)
        )
        self.trunk_norm = nn.LayerNorm(d)
        if head_hidden_dim is None:
            self.heads: nn.Module = EnsembleLinear(n_ens, d, d)
        else:
            self.heads = nn.Sequential(
                EnsembleLinear(n_ens, d, head_hidden_dim), nn.GELU(), EnsembleLinear(n_ens, head_hidden_dim, d)
            )
        nn.init.normal_(self.slot_queries, std=0.02)
        nn.init.normal_(self.mode_embedding.weight, std=0.02)

    @classmethod
    def from_config(cls, cfg: PlannerConfig) -> JEPAPredictor:
        p = cfg.predictor
        return cls(cfg.latent_dim, cfg.num_modes, cfg.num_slots, p.n_ens, p.num_layers, p.num_heads,
                   p.mlp_ratio, p.head_hidden_dim, p.dropout)

    def _slot_tokens(self, batch: int, slot_ids: Tensor | None) -> Tensor:
        """Delta rows for every sample: ``[B, M', d]``."""
        if slot_ids is None:
            return self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        if slot_ids.dim() != 2 or slot_ids.shape[0] != batch or slot_ids.dtype.is_floating_point \
                or slot_ids.dtype == torch.bool or slot_ids.is_complex():
            raise ValueError(f"expected integer slot_ids of shape [{batch}, M'], got "
                             f"{tuple(slot_ids.shape)} {slot_ids.dtype}")
        if slot_ids.shape[1] == 0:
            raise ValueError("slot_ids selects no slot (M' = 0): Eq. 8 is undefined")
        if bool((slot_ids < 0).any()) or bool((slot_ids >= self.num_slots).any()):
            raise ValueError(f"slot_ids must lie in [0, {self.num_slots})")
        return self.slot_queries[slot_ids]

    def forward_heads(
        self,
        s_x: Tensor,
        u: Tensor,
        slot_ids: Tensor | None = None,
        context_mask: Tensor | None = None,
        slot_mask: Tensor | None = None,
    ) -> Tensor:
        """Per-head plans s_hat^(e) ``[n_ens, B, M', d]``, exact zeros at padded slots."""
        if s_x.dim() != 3 or s_x.shape[-1] != self.latent_dim:
            raise ValueError(f"expected s_x of shape [B, M_c, {self.latent_dim}], got {tuple(s_x.shape)}")
        if s_x.shape[1] == 0:
            raise ValueError("s_x has no context tokens (M_c = 0): cross-attention needs a non-empty key set")
        b = s_x.shape[0]
        mode = embed_modes(self.mode_embedding, u)                                  # [B, d]
        if mode.shape[0] != b:
            raise ValueError(f"u has batch size {mode.shape[0]}, s_x has {b}")
        x = self._slot_tokens(b, slot_ids) + mode.unsqueeze(1)                      # [B, M', d]
        m = x.shape[1]
        for name, mask, shape in (("context_mask", context_mask, tuple(s_x.shape[:2])), ("slot_mask", slot_mask, (b, m))):
            if mask is None:
                continue
            if mask.dtype != torch.bool or tuple(mask.shape) != shape:
                raise ValueError(f"expected bool {name} of shape {list(shape)}, got {tuple(mask.shape)} {mask.dtype}")
            if not bool(mask.any(dim=1).all()):
                raise ValueError(f"every row of {name} needs at least one True entry")
        memory = self.context_norm(s_x)                                             # [B, M_c, d]
        for block in self.blocks:
            x = block(x, slot_mask, memory, context_mask)
        h = self.trunk_norm(x).reshape(1, b * m, self.latent_dim)                   # [1, B*M', d]
        heads = self.heads(h.expand(self.n_ens, -1, -1)).reshape(self.n_ens, b, m, self.latent_dim)
        if slot_mask is not None:
            heads = heads.masked_fill(~slot_mask[None, :, :, None], 0.0)
        return heads

    def forward(
        self,
        s_x: Tensor,
        u: Tensor,
        slot_ids: Tensor | None = None,
        context_mask: Tensor | None = None,
        slot_mask: Tensor | None = None,
    ) -> PredictorOutput:
        """Pass 3, Eqs. 3 and 8: (Z_target, sigma_bar) = g_phi(s_x, u, Delta)."""
        heads = self.forward_heads(s_x, u, slot_ids, context_mask, slot_mask)
        z_target = ensemble_mean(heads)                                             # Eq. 8
        sigma_bar = ensemble_uncertainty(heads, z_target, slot_mask)                # Eq. 8
        mask = slot_mask if slot_mask is not None else torch.ones(
            heads.shape[1:3], dtype=torch.bool, device=heads.device
        )
        return PredictorOutput(z_target, sigma_bar, heads, mask)
