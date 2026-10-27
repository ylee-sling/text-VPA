# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
fs_dfm.py -- the primary realizer: conditional FS-DFM with the exact few-step jump kernel (Sec. 3.3).

    probability path   p_t(w^i | w_0^i, w_1^i) = (1 - kappa_t) delta_{w_0^i} + kappa_t delta_{w_1^i}   (Eq. 11)
    denoiser           p^psi_{1|t}(j | w_t, t, h, Z) = softmax(F_psi(w_t, t, h, Z))
    training loss      L_real = E[ - sum_i log p^psi_{1|t}(w_1^i | w_t, t, h, sg(Z)) ]                  (Eq. 12)
    sampling kernel    exact jump mass pi_{t,h} = (kappa_{t+h} - kappa_t)/(1 - kappa_t), tilted and
                       clipped by the constraint energies (Eqs. 15-17; ``kernels.py``; Algorithm 1)

The source p_0 is uniform over V (as in the released FS-DFM checkpoints) or the [MASK] token.
F_psi's cross-attention reads the bridge's cached keys and values; it never recomputes them.

Training (Sec. 5.1): the plan comes from the frozen planner (``prepare`` accepts only a
``PredictedPlan``), the step budget K is drawn from the trained budgets and conditions the network
through h = 1/K, and plan dropout replaces Z~ by the null token with probability 0.1. Distillation from
a 1,024-step teacher (FS-DFM) is not part of this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from text_vpa.models.bridge import BridgeKVCache, CrossAttentionBridge
from text_vpa.models.plan import PredictedPlan, require_predicted_plan
from text_vpa.models.realizers.backbone import RealizerBackbone
from text_vpa.models.realizers.paths import KappaSchedule, MixtureSample, mixture_path_sample

if TYPE_CHECKING:
    from text_vpa.configs.schema import TextVPAConfig

__all__ = ["ConditionalFSDFMRealizer", "MixtureSample", "mixture_path_sample"]  # the last two re-exported from paths


class ConditionalFSDFMRealizer(nn.Module):
    """F_psi for the CTMC realizer: token states -> logits of p^psi_{1|t}.

    Args:
        vocab_size: S.
        bridge: the cross-attention bridge (owned by the realizer; L_r and d_r must match).
        num_heads, mlp_ratio, dropout, max_len: backbone hyperparameters.
        schedule: kappa schedule name ("linear", "sqrt", "cosine").
        source: "uniform" or "mask".
        mask_id: id of [MASK] (required for the mask source).

    Shapes:
        ``forward(w_t [B, N] int64, t [B], h [B], cache)`` -> logits ``[B, N, S]``
    """

    kind = "fs_dfm"

    def __init__(self, vocab_size: int, bridge: CrossAttentionBridge, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, max_len: int = 1024, schedule: str = "linear", source: str = "uniform",
                 mask_id: int | None = None) -> None:
        super().__init__()
        if source not in ("uniform", "mask"):
            raise ValueError("source must be 'uniform' or 'mask'")
        if source == "mask" and mask_id is None:
            raise ValueError("the mask source needs mask_id")
        self.vocab_size = vocab_size
        self.bridge = bridge
        self.schedule = KappaSchedule(schedule)
        self.source = source
        self.mask_id = mask_id
        self.token_embedding = nn.Embedding(vocab_size, bridge.hidden_dim)
        self.backbone = RealizerBackbone(vocab_size, bridge.hidden_dim, bridge.num_layers, num_heads, mlp_ratio,
                                         dropout, max_len)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    @classmethod
    def from_config(cls, cfg: TextVPAConfig, mask_id: int | None = None) -> ConditionalFSDFMRealizer:
        r = cfg.realizer
        return cls(r.vocab_size, CrossAttentionBridge.from_config(cfg), r.num_heads, r.mlp_ratio, r.dropout,
                   r.max_output_len, r.schedule, r.source, mask_id)

    @property
    def max_len(self) -> int:
        return self.backbone.max_len

    def prepare(self, plan: PredictedPlan, drop: Tensor | None = None, with_null: bool = False,
                generator: torch.Generator | None = None) -> BridgeKVCache:
        """Bridge K/V of Z~ = [Z_target; s_x; e(u)]/gamma, computed once per generation (fused into pass 4)."""
        plan = require_predicted_plan(plan)
        return self.bridge(plan.z_target, plan.s_x, plan.mode_input, None, plan.context_mask,
                           drop=drop, with_null=with_null, generator=generator)

    def forward(self, w_t: Tensor, t: Tensor, h: Tensor, cache: BridgeKVCache) -> Tensor:
        if w_t.dim() != 2 or w_t.dtype.is_floating_point:
            raise ValueError(f"expected integer w_t of shape [B, N], got {tuple(w_t.shape)} {w_t.dtype}")
        return self.backbone(self.token_embedding(w_t), t, h, self.bridge, cache)

    def sample_source(self, batch: int, length: int, device: torch.device, generator: torch.Generator | None = None,
                      allowed: Tensor | None = None) -> Tensor:
        """w_0 ~ p_0: [MASK], or uniform over V -- over ``allowed`` [B, S] tokens when given (Prop. 4(d))."""
        if self.source == "mask":
            return torch.full((batch, length), self.mask_id, dtype=torch.long, device=device)  # type: ignore[arg-type]
        if allowed is None:
            return torch.randint(self.vocab_size, (batch, length), generator=generator, device=device)
        weights = allowed.to(device=device, dtype=torch.float32)
        if bool((weights.sum(dim=1) == 0).any()):
            raise ValueError("a sample has no admissible token to start from")
        draws = torch.multinomial(weights, length, replacement=True, generator=generator)
        return draws

    def loss(self, logits: Tensor, w_1: Tensor, reduction: str = "mean") -> Tensor:
        """Eq. 12: -sum_i log p(w_1^i | ...); ``"mean"`` divides by B N (a constant rescaling), ``"sum"``
        divides by B only (Eq. 12 per sample)."""
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), w_1.reshape(-1), reduction="none")
        nll = nll.view(w_1.shape)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum(dim=1).mean()
        raise ValueError("reduction must be 'mean' or 'sum'")
