# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
backbone.py -- the realizer network F_psi shared by the FS-DFM and Logit-KL realizers (Sec. 3.2).

A bidirectional transformer over the N output positions with L_r blocks. Block l applies (Sec. 3.2)

    x <- x + g1 * SelfAttn( modulate(LN(x), shift1, scale1) )
    x <- x + g2 * CA_l( modulate(LN(x), shift2, scale2), Z~ )        cached K_l, V_l (Eq. 10)
    x <- x + g3 * MLP( modulate(LN(x), shift3, scale3) )

with adaptive layer-norm modulation by embeddings of the time t and the step size h (FS-DFM's
step-budget conditioning), modulate(x, s, c) = x (1 + c) + s. The cross-attention CA_l is the bridge's
``cross_attention`` with keys and values computed once per generation (``BridgeKVCache``); this network
never recomputes them.

AdaLN design (choices the paper leaves open): "adaLN-single" -- one projection SiLU(c) -> 9 d_r shared by
all blocks plus a learned per-block table [9, d_r] -- because per-block projections would add
9 d_r^2 L_r parameters (about 0.9B at d_r = 2048, L_r = 24). The projection and the gate rows of the
tables start at zero ("adaLN-Zero"), so every block is the identity at initialisation. LayerNorms in
the blocks have no affine parameters; the modulation supplies them. c = MLP_t(sin(1000 t)) +
MLP_h(sin(1000 h)) with sinusoidal features of width ``time_dim``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from text_vpa.models.bridge import BridgeKVCache, CrossAttentionBridge
from text_vpa.models.layers import FeedForward, MultiHeadAttention

__all__ = ["RealizerBackbone", "RealizerBlock", "timestep_embedding"]


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10_000.0, scale: float = 1000.0) -> Tensor:
    """Sinusoidal features of ``scale * t``: ``[B]`` -> ``[B, dim]`` (cos half, then sin half)."""
    if t.dim() != 1:
        raise ValueError(f"expected t of shape [B], got {tuple(t.shape)}")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (scale * t.float())[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


class RealizerBlock(nn.Module):
    """One AdaLN block. Shapes: ``x [B, N, d_r]``, ``modulation [B, 9, d_r]`` -> ``[B, N, d_r]``."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)
        # Per-block (shift, scale, gate) x (self-attn, cross-attn, mlp); zero: identity at initialisation.
        self.table = nn.Parameter(torch.zeros(9, dim))

    def forward(self, x: Tensor, modulation: Tensor, layer: int, bridge: CrossAttentionBridge,
                cache: BridgeKVCache) -> Tensor:
        m = (modulation + self.table).unsqueeze(2)                     # [B, 9, 1, d_r]
        sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = m.unbind(dim=1)    # each [B, 1, d_r]
        h = _modulate(self.norm_self(x), sh1, sc1)
        x = x + g1 * self.drop(self.self_attn(h, h))
        h = _modulate(self.norm_cross(x), sh2, sc2)
        x = x + g2 * self.drop(bridge.cross_attention(layer, h, cache))
        h = _modulate(self.norm_mlp(x), sh3, sc3)
        return x + g3 * self.drop(self.mlp(h))


class RealizerBackbone(nn.Module):
    """F_psi without its input embedding: ``[B, N, d_r]`` embeddings -> ``[B, N, S]`` logits.

    Args:
        vocab_size: S.
        hidden_dim: d_r (must equal the bridge's ``hidden_dim``).
        num_layers: L_r (must equal the bridge's ``num_layers``).
        num_heads, mlp_ratio, dropout: transformer hyperparameters.
        max_len: maximum N (learned absolute positions).
        time_dim: width of the sinusoidal t and h features.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, mlp_ratio: float,
                 dropout: float, max_len: int, time_dim: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.time_dim = time_dim
        self.position = nn.Embedding(max_len, hidden_dim)
        self.t_embed = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.h_embed = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.block_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 9 * hidden_dim))
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.blocks = nn.ModuleList(RealizerBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, vocab_size)
        # Recompute each block in the backward pass instead of storing its activations (training only).
        self.activation_checkpointing = False
        nn.init.normal_(self.position.weight, std=0.02)
        for proj in (self.block_modulation[-1], self.final_modulation[-1]):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x: Tensor, t: Tensor, h: Tensor, bridge: CrossAttentionBridge, cache: BridgeKVCache) -> Tensor:
        """Args:
            x: ``[B, N, d_r]`` input embeddings of the state.
            t: ``[B]`` flow time in [0, 1).
            h: ``[B]`` step size 1/K.
            bridge: owner of W_Q^l, W_O^l and the attention of Eq. 10.
            cache: keys and values computed once by ``bridge.precompute_kv``.
        """
        b, n, d = x.shape
        if d != self.hidden_dim or n > self.max_len:
            raise ValueError(f"expected [B, N <= {self.max_len}, {self.hidden_dim}], got {tuple(x.shape)}")
        if t.shape != (b,) or h.shape != (b,):
            raise ValueError(f"t and h must have shape [{b}], got {tuple(t.shape)} and {tuple(h.shape)}")
        if cache.num_layers != self.num_layers or cache.mask.shape[0] != b:
            raise ValueError("the bridge cache does not match this realizer (layers or batch size)")
        dtype = self.position.weight.dtype
        c = (self.t_embed(timestep_embedding(t, self.time_dim).to(dtype))
             + self.h_embed(timestep_embedding(h, self.time_dim).to(dtype)))
        modulation = self.block_modulation(c).view(b, 9, d)
        x = x + self.position(torch.arange(n, device=x.device))
        recompute = self.activation_checkpointing and self.training and torch.is_grad_enabled()
        for layer, block in enumerate(self.blocks):
            if recompute:
                x = checkpoint(block, x, modulation, layer, bridge, cache, use_reentrant=False)
            else:
                x = block(x, modulation, layer, bridge, cache)
        shift, scale = self.final_modulation(c).view(b, 2, 1, d).unbind(dim=1)
        return self.head(_modulate(self.final_norm(x), shift, scale))
