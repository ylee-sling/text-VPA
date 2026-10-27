# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
logit_kl.py -- the alternative realizer: Logit-KL flow matching along KL geodesics (Sec. 3.4).

State: logits l_t in R^{B x N x S}. Smoothed one-hot targets x_1^eps = (1 - eps) e_k + (eps/S) 1 have
logits l_1 = a e_k + b 1 (Eq. 18). From l_0 ~ N(0, s^2 I):

    l_t = (1 - t) l_0 + t l_1,   x_t = softmax(l_t)                                         (Eq. 19)
    v*(l, t) = (a p(. | l_t = l) + b 1 - l)/(1 - t)                                         (Lemma 2, Eq. 20)

so the network is trained with cross-entropy to output p_psi(k | l_t, t, Z), and sampling integrates
v* with Euler steps on t_k = k/K (Eq. 21; b 1 is dropped because softmax ignores constant shifts):

    l_{k+1} = l_k + (1/K) (a p_psi(. | l_k, t_k, Z) - l_k)/(1 - t_k),   l_K = a p_psi at the last step.

Tokens are decoded as argmax_j l_K(j); energies enter by tilting the posterior, p~ ∝ p exp(-E/T).
The network reads x_t = softmax(l_t) through a soft token embedding x_t E. Memory: the state is
N x S numbers per sample (about 0.3 GB in bf16 for N = 1024 and S = 152,064; Sec. 3.4).
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
from text_vpa.models.realizers.paths import logit_euler_step, logit_path, smoothed_logit_coefficients

if TYPE_CHECKING:
    from text_vpa.configs.schema import TextVPAConfig

__all__ = ["LogitKLRealizer"]


class LogitKLRealizer(nn.Module):
    """F_psi for the logit-space ODE.

    Args:
        vocab_size: S.
        bridge: the cross-attention bridge (owned by the realizer).
        num_heads, mlp_ratio, dropout, max_len: backbone hyperparameters.
        smoothing: eps of the smoothed targets (Eq. 18).
        source_std: s, the std of l_0.

    Shapes:
        ``forward(l_t [B, N, S], t [B], h [B], cache)`` -> logits of p_psi ``[B, N, S]``
    """

    kind = "logit_kl"

    def __init__(self, vocab_size: int, bridge: CrossAttentionBridge, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, max_len: int = 1024, smoothing: float = 0.05, source_std: float = 1.0) -> None:
        super().__init__()
        if not source_std > 0.0:
            raise ValueError("source_std must be positive")
        self.vocab_size = vocab_size
        self.bridge = bridge
        self.smoothing = float(smoothing)
        self.source_std = float(source_std)
        self.a, self.b = smoothed_logit_coefficients(smoothing, vocab_size)
        self.token_embedding = nn.Parameter(torch.empty(vocab_size, bridge.hidden_dim))
        self.backbone = RealizerBackbone(vocab_size, bridge.hidden_dim, bridge.num_layers, num_heads, mlp_ratio,
                                         dropout, max_len)
        nn.init.normal_(self.token_embedding, std=0.02)

    @classmethod
    def from_config(cls, cfg: TextVPAConfig) -> LogitKLRealizer:
        r = cfg.realizer
        return cls(r.vocab_size, CrossAttentionBridge.from_config(cfg), r.num_heads, r.mlp_ratio, r.dropout,
                   r.max_output_len, r.logit_smoothing, r.logit_source_std)

    @property
    def max_len(self) -> int:
        return self.backbone.max_len

    def prepare(self, plan: PredictedPlan, drop: Tensor | None = None, with_null: bool = False,
                generator: torch.Generator | None = None) -> BridgeKVCache:
        """Bridge K/V, computed once per generation (fused into pass 4)."""
        plan = require_predicted_plan(plan)
        return self.bridge(plan.z_target, plan.s_x, plan.mode_input, None, plan.context_mask,
                           drop=drop, with_null=with_null, generator=generator)

    def forward(self, l_t: Tensor, t: Tensor, h: Tensor, cache: BridgeKVCache) -> Tensor:
        if l_t.dim() != 3 or l_t.shape[-1] != self.vocab_size:
            raise ValueError(f"expected l_t of shape [B, N, {self.vocab_size}], got {tuple(l_t.shape)}")
        x_t = torch.softmax(l_t.float(), dim=-1).to(self.token_embedding.dtype)   # point on the simplex (Eq. 19)
        return self.backbone(x_t @ self.token_embedding, t, h, self.bridge, cache)

    # -----------------------------------------------------------------------------------------
    def target_logits(self, w_1: Tensor) -> Tensor:
        """l_1 = a e_k + b 1 (Eq. 18): ``[B, N]`` -> ``[B, N, S]`` float32."""
        return self.a * F.one_hot(w_1, self.vocab_size).float() + self.b

    def sample_source(self, batch: int, length: int, device: torch.device,
                      generator: torch.Generator | None = None) -> Tensor:
        """l_0 ~ N(0, s^2 I): ``[B, N, S]``."""
        return self.source_std * torch.randn(batch, length, self.vocab_size, generator=generator, device=device)

    def training_state(self, w_1: Tensor, step_budgets: tuple[int, ...] | list[int], time_sampling: str = "grid",
                       generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """(l_t, t, h) for the cross-entropy objective of Lemma 2, with l_t on the path of Eq. 19."""
        b, n = w_1.shape
        budgets = torch.tensor(list(step_budgets), dtype=torch.float32)
        k_steps = budgets[torch.randint(len(budgets), (b,), generator=generator)]
        if time_sampling == "grid":
            t = torch.floor(torch.rand(b, generator=generator) * k_steps) / k_steps
        elif time_sampling == "uniform":
            t = torch.rand(b, generator=generator)
        else:
            raise ValueError("time_sampling must be 'grid' or 'uniform'")
        l0 = self.sample_source(b, n, torch.device("cpu"), generator).to(w_1.device)
        t = t.to(w_1.device)
        l_t = logit_path(l0, self.target_logits(w_1), t[:, None, None])
        return l_t, t, (1.0 / k_steps).to(w_1.device)

    def euler_step(self, l_k: Tensor, posterior: Tensor, t: float, h: float) -> Tensor:
        """Eq. 21; returns a p exactly at the last step."""
        return logit_euler_step(l_k, posterior, t, h, self.a)

    def loss(self, logits: Tensor, w_1: Tensor, reduction: str = "mean") -> Tensor:
        """Cross-entropy -log p_psi(w_1 | l_t, t, Z) (Lemma 2: its optimum gives v*)."""
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), w_1.reshape(-1), reduction="none")
        nll = nll.view(w_1.shape)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum(dim=1).mean()
        raise ValueError("reduction must be 'mean' or 'sum'")
