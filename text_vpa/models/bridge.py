# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
bridge.py -- the cross-attention bridge from the plan to the realizer (Sec. 3.2, Eq. 10).

Conditioning tokens, M~ = M + M_c + 1 rows of width d:

    Z~ = [Z_target ; s_x ; e(u)] / gamma                                                     (Sec. 3.2)

Realizer block l attends to them with (Eq. 10, multi-head, heads concatenated and projected by W_O^l):

    CA_l(H, Z~) = softmax( (H W_Q^l)(Z~ W_K^l)^T / sqrt(d_h) ) (Z~ W_V^l)                    (Eq. 10)

Z~ does not change during sampling, so K_l = Z~ W_K^l and V_l = Z~ W_V^l are computed once per
generation by ``precompute_kv`` and reused by all K realizer passes through ``cross_attention``.
The cache holds 2 L_r M~ d_r numbers per sample, independent of the output length N
(48 MiB for M~ = 256, L_r = 24, d_r = 2048 in bf16; Sec. 3.2).

Spectral normalisation (Miyato et al., 2018): every W_K^l and W_V^l is reparametrised as
W / sigma_max(W) with ``torch.nn.utils.parametrizations.spectral_norm`` (sigma_max by power
iteration: 15 iterations at registration, ``power_iterations`` per forward in training mode, none
in eval mode). The projections have no bias, so Z~ -> K_l and Z~ -> V_l are linear maps of operator
norm 1, and scaling by the constant 1/gamma instead of a layer norm keeps the bridge Lipschitz in
the plan (Sec. 3.2; assumption A1).

Plan dropout: in training, each sample's Z~ is replaced with probability ``plan_dropout`` (0.1) by a
learned null token -- the sample's conditioning becomes that single token, the other rows masked.
``null_conditioning`` builds the same unconditional input for predictor-free guidance,
(1 + omega) u(. | Z) - omega u(. | null) (Sec. 3.3).

Depth accounting: the projections depend only on the plan; they are launched together with the
first realizer pass ("fused into pass 4") and add no sequential evaluation to K + 3.

Decisions the paper leaves open: e(u) is the bridge's own embedding table (g_phi has another);
the null token replaces Z~ after the 1/gamma scaling; W_Q^l, W_O^l live here with W_K^l, W_V^l so
that Eq. 10 is implemented in one place; no source-type embedding is added to Z~ (the paper's Z~
has none).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm

from text_vpa.models.layers import embed_modes, padding_attn_mask

if TYPE_CHECKING:
    from text_vpa.configs.schema import TextVPAConfig

__all__ = ["BridgeConditioning", "BridgeKVCache", "CrossAttentionBridge"]


class BridgeConditioning(NamedTuple):
    """Conditioning tokens for one generation."""

    tokens: Tensor   # [B, M~, d]  Z~ (or the null token in row 0)
    mask: Tensor     # [B, M~] bool, True at real conditioning tokens
    dropped: Tensor  # [B] bool, True where Z~ was replaced by the null token


class BridgeKVCache(NamedTuple):
    """Per-layer keys and values, computed once per generation and held fixed for all K passes."""

    keys: tuple[Tensor, ...]    # L_r x [B, H, M~, d_h]   K_l = Z~ W_K^l
    values: tuple[Tensor, ...]  # L_r x [B, H, M~, d_h]   V_l = Z~ W_V^l
    mask: Tensor                # [B, M~] bool

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def nbytes(self) -> int:
        """Bytes held by keys and values: 2 L_r B M~ d_r times the element size; no dependence on N."""
        return sum(t.numel() * t.element_size() for t in (*self.keys, *self.values))


class CrossAttentionBridge(nn.Module):
    """Plan-to-realizer bridge with spectrally normalised, cached keys and values.

    Args:
        latent_dim: d, width of Z_target, s_x and e(u).
        hidden_dim: d_r, realizer width (= total key/value width d_k).
        num_layers: L_r, realizer blocks with cross-attention.
        num_heads: attention heads; must divide ``hidden_dim``.
        num_modes: |U|, for the embedding e(u).
        gamma: conditioning scale (``VICRegConfig.gamma``).
        plan_dropout: probability of replacing Z~ by the null token during training.
        spectral_norm: apply spectral normalisation to W_K^l and W_V^l.
        power_iterations: power iterations per training forward of each normalised weight.

    Shapes:
        ``build_conditioning(z_target [B, M, d], s_x [B, M_c, d], u [B] or [B, |U|], ...)``
        -> ``BridgeConditioning(tokens [B, M~, d], mask [B, M~], dropped [B])``
        ``precompute_kv(conditioning)`` -> ``BridgeKVCache`` (L_r x [B, H, M~, d_h] each)
        ``cross_attention(layer, hidden [B, N, d_r], cache)`` -> ``[B, N, d_r]``
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_modes: int,
        gamma: float = 1.0,
        plan_dropout: float = 0.1,
        spectral_norm: bool = True,
        power_iterations: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if not gamma > 0.0:
            raise ValueError("gamma must be positive")
        if not 0.0 <= plan_dropout < 1.0:
            raise ValueError("plan_dropout must be in [0, 1)")
        if num_layers < 1 or power_iterations < 1:
            raise ValueError("num_layers and power_iterations must be >= 1")
        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.gamma: float = float(gamma)
        self.plan_dropout: float = float(plan_dropout)
        self.spectral_norm: bool = spectral_norm

        self.mode_embedding = nn.Embedding(num_modes, latent_dim)                 # e(u)
        self.null_token = nn.Parameter(torch.empty(latent_dim))                    # replaces Z~ (plan dropout)

        def conditioning_projection() -> nn.Linear:
            layer = nn.Linear(latent_dim, hidden_dim, bias=False)
            return _spectral_norm(layer, n_power_iterations=power_iterations) if spectral_norm else layer

        self.key_proj = nn.ModuleList(conditioning_projection() for _ in range(num_layers))    # W_K^l
        self.value_proj = nn.ModuleList(conditioning_projection() for _ in range(num_layers))  # W_V^l
        self.query_proj = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers))
        self.out_proj = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers))
        nn.init.normal_(self.mode_embedding.weight, std=0.02)
        nn.init.normal_(self.null_token, std=0.02)

    @classmethod
    def from_config(cls, cfg: TextVPAConfig) -> CrossAttentionBridge:
        return cls(
            latent_dim=cfg.planner.latent_dim, hidden_dim=cfg.realizer.hidden_dim, num_layers=cfg.realizer.num_layers,
            num_heads=cfg.realizer.num_heads, num_modes=cfg.planner.num_modes, gamma=cfg.vicreg.gamma,
            plan_dropout=cfg.bridge.plan_dropout, spectral_norm=cfg.bridge.spectral_norm,
            power_iterations=cfg.bridge.power_iterations,
        )

    @staticmethod
    def kv_cache_bytes(num_layers: int, cond_tokens: int, hidden_dim: int, bytes_per_element: int,
                       batch_size: int = 1) -> int:
        """2 L_r M~ d_k b per sample (Sec. 3.2), times the batch size."""
        return 2 * num_layers * cond_tokens * hidden_dim * bytes_per_element * batch_size

    # -----------------------------------------------------------------------------------------
    def build_conditioning(
        self,
        z_target: Tensor,
        s_x: Tensor,
        u: Tensor,
        slot_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
        drop: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> BridgeConditioning:
        """Z~ = [Z_target; s_x; e(u)] / gamma, with plan dropout in training.

        Args:
            z_target: ``[B, M, d]`` plan from g_phi (never f_theta_bar(y): Sec. 5.1, Stage 2).
            s_x: ``[B, M_c, d]`` context latents.
            u: ``[B]`` int64 modes or ``[B, |U|]`` (straight-through) one-hot vectors.
            slot_mask, context_mask: ``[B, M]`` / ``[B, M_c]`` bool, True at real rows.
            drop: ``[B]`` bool, overrides the random plan-dropout draw (None: Bernoulli(p) per
                sample in training mode, no dropout in eval mode).
            generator: random generator for the dropout draw (on the tensors' device).
        """
        if z_target.dim() != 3 or s_x.dim() != 3 or z_target.shape[-1] != self.latent_dim \
                or s_x.shape[-1] != self.latent_dim or z_target.shape[0] != s_x.shape[0]:
            raise ValueError(f"expected z_target [B, M, {self.latent_dim}] and s_x [B, M_c, {self.latent_dim}], got "
                             f"{tuple(z_target.shape)} and {tuple(s_x.shape)}")
        b, m, _ = z_target.shape
        m_c = s_x.shape[1]
        mode = embed_modes(self.mode_embedding, u)                                    # [B, d]
        if mode.shape[0] != b:
            raise ValueError(f"u has batch size {mode.shape[0]}, the plan has {b}")
        tokens = torch.cat([z_target, s_x, mode.unsqueeze(1)], dim=1) / self.gamma   # [B, M~, d]
        ones = z_target.new_ones((b, 1), dtype=torch.bool)
        slot_mask = slot_mask if slot_mask is not None else ones.expand(b, m)
        context_mask = context_mask if context_mask is not None else ones.expand(b, m_c)
        if slot_mask.shape != (b, m) or context_mask.shape != (b, m_c):
            raise ValueError("slot_mask / context_mask shapes do not match z_target / s_x")
        mask = torch.cat([slot_mask, context_mask, ones], dim=1)                     # [B, M~]
        if drop is None:
            if self.training and self.plan_dropout > 0.0:
                draw = torch.rand(b, generator=generator, device=tokens.device)
                drop = draw < self.plan_dropout
            else:
                drop = torch.zeros(b, dtype=torch.bool, device=tokens.device)
        elif drop.shape != (b,) or drop.dtype != torch.bool:
            raise ValueError(f"expected a bool drop mask of shape [{b}]")
        if bool(drop.any()):
            null_tokens = torch.zeros_like(tokens)
            null_tokens[:, 0] = self.null_token.to(tokens.dtype)
            null_mask = torch.zeros_like(mask)
            null_mask[:, 0] = True
            tokens = torch.where(drop[:, None, None], null_tokens, tokens)
            mask = torch.where(drop[:, None], null_mask, mask)
        return BridgeConditioning(tokens, mask, drop)

    def null_conditioning(self, batch_size: int, device: torch.device | None = None,
                          dtype: torch.dtype | None = None) -> BridgeConditioning:
        """Unconditional input (the learned null token) for predictor-free guidance: ``[B, 1, d]``."""
        token = self.null_token.to(device=device or self.null_token.device, dtype=dtype or self.null_token.dtype)
        tokens = token.expand(batch_size, 1, self.latent_dim)
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        return BridgeConditioning(tokens, mask, torch.ones(batch_size, dtype=torch.bool, device=tokens.device))

    def precompute_kv(self, conditioning: BridgeConditioning) -> BridgeKVCache:
        """K_l = Z~ W_K^l and V_l = Z~ W_V^l for every realizer block l, once per generation."""
        tokens = conditioning.tokens
        if tokens.dim() != 3 or tokens.shape[-1] != self.latent_dim:
            raise ValueError(f"expected conditioning tokens [B, M~, {self.latent_dim}], got {tuple(tokens.shape)}")
        if not bool(conditioning.mask.any(dim=1).all()):
            raise ValueError("every sample needs at least one real conditioning token")
        keys = tuple(rearrange(proj(tokens), "b m (h e) -> b h m e", h=self.num_heads) for proj in self.key_proj)
        values = tuple(rearrange(proj(tokens), "b m (h e) -> b h m e", h=self.num_heads) for proj in self.value_proj)
        return BridgeKVCache(keys, values, conditioning.mask)

    def cross_attention(self, layer: int, hidden: Tensor, cache: BridgeKVCache) -> Tensor:
        """Eq. 10 for realizer block ``layer``: queries from ``hidden [B, N, d_r]``, cached K_l, V_l."""
        if not 0 <= layer < self.num_layers or cache.num_layers != self.num_layers:
            raise ValueError(f"layer must be in [0, {self.num_layers}) and the cache must hold {self.num_layers} layers")
        if hidden.dim() != 3 or hidden.shape[-1] != self.hidden_dim or hidden.shape[0] != cache.mask.shape[0]:
            raise ValueError(f"expected hidden [B, N, {self.hidden_dim}] with the cache's batch size, got "
                             f"{tuple(hidden.shape)}")
        q = rearrange(self.query_proj[layer](hidden), "b n (h e) -> b h n e", h=self.num_heads)
        out = F.scaled_dot_product_attention(q, cache.keys[layer], cache.values[layer],
                                             attn_mask=padding_attn_mask(cache.mask))
        return self.out_proj[layer](rearrange(out, "b h n e -> b n (h e)"))

    def forward(
        self,
        z_target: Tensor,
        s_x: Tensor,
        u: Tensor,
        slot_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
        drop: Tensor | None = None,
        with_null: bool = False,
        generator: torch.Generator | None = None,
    ) -> BridgeKVCache:
        """``build_conditioning`` followed by ``precompute_kv`` (the part of pass 4 run once).

        With ``with_null`` the cache holds 2B rows, [conditional; null], so predictor-free guidance
        evaluates both in one realizer pass and the sequential depth stays K + 3.
        """
        cond = self.build_conditioning(z_target, s_x, u, slot_mask, context_mask, drop=drop, generator=generator)
        if with_null:
            b = z_target.shape[0]
            null = self.build_conditioning(z_target, s_x, u, slot_mask, context_mask,
                                           drop=torch.ones(b, dtype=torch.bool, device=z_target.device))
            cond = BridgeConditioning(torch.cat([cond.tokens, null.tokens]), torch.cat([cond.mask, null.mask]),
                                      torch.cat([cond.dropped, null.dropped]))
        return self.precompute_kv(cond)
