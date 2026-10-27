# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
layers.py -- building blocks shared by the planner networks (encoders, selector, predictor).

    * ``MultiHeadAttention``: softmax((X W_Q)(Y W_K)^T / sqrt(d_h)) (Y W_V), heads concatenated and
      projected by W_O, with key padding; runs on ``F.scaled_dot_product_attention`` (CUDA, ROCm,
      MPS and CPU kernels).
    * ``TransformerBlock``: pre-norm block with self-attention, optional cross-attention to a memory,
      and a GELU MLP.
    * ``EnsembleLinear``: n_ens independent affine maps in one batched product (shared with VPA).
    * ``masked_mean``, ``embed_modes``: masked pooling, and mode embedding for integer modes or
      (straight-through) one-hot / soft mode vectors.

Masks are boolean with True at real positions. Every row of a key mask must contain at least one
True entry; attention over an empty key set is undefined (softmax of an all -inf row).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

__all__ = [
    "EnsembleLinear",
    "FeedForward",
    "MultiHeadAttention",
    "TransformerBlock",
    "embed_modes",
    "masked_mean",
    "padding_attn_mask",
]


def padding_attn_mask(key_mask: Tensor | None) -> Tensor | None:
    """``[B, L_k]`` bool key mask -> ``[B, 1, 1, L_k]`` boolean SDPA mask (True = attend)."""
    if key_mask is None:
        return None
    if key_mask.dtype != torch.bool or key_mask.dim() != 2:
        raise ValueError(f"expected a bool key mask of shape [B, L_k], got {tuple(key_mask.shape)} {key_mask.dtype}")
    return key_mask[:, None, None, :]


class MultiHeadAttention(nn.Module):
    """Multi-head (cross-)attention with key padding.

    Args:
        dim: query and output width.
        num_heads: heads; must divide ``dim``.
        kv_dim: width of the memory the keys and values are computed from (default ``dim``).
        dropout: attention dropout (training only).
        bias: whether the four projections carry biases.

    Shapes:
        ``query [B, L_q, dim]``, ``memory [B, L_k, kv_dim]``, ``key_mask [B, L_k]`` bool or None
        -> ``[B, L_q, dim]``
    """

    def __init__(self, dim: int, num_heads: int, kv_dim: int | None = None, dropout: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        kv_dim = dim if kv_dim is None else kv_dim
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(kv_dim, dim, bias=bias)
        self.v_proj = nn.Linear(kv_dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, query: Tensor, memory: Tensor, key_mask: Tensor | None = None) -> Tensor:
        q = rearrange(self.q_proj(query), "b l (h e) -> b h l e", h=self.num_heads)
        k = rearrange(self.k_proj(memory), "b l (h e) -> b h l e", h=self.num_heads)
        v = rearrange(self.v_proj(memory), "b l (h e) -> b h l e", h=self.num_heads)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=padding_attn_mask(key_mask), dropout_p=self.dropout if self.training else 0.0
        )                                                                   # [B, H, L_q, d_h]
        return self.out_proj(rearrange(out, "b h l e -> b l (h e)"))


class FeedForward(nn.Module):
    """GELU MLP ``dim -> hidden -> dim``. Shapes: ``[..., dim]`` -> ``[..., dim]``."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, int(round(dim * mlp_ratio)))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block.

        x <- x + SelfAttn(LN(x))
        x <- x + CrossAttn(LN(x), memory)        (only if ``cross_attention``)
        x <- x + MLP(LN(x))

    Shapes:
        ``x [B, L, dim]``, ``mask [B, L]`` bool or None, ``memory [B, L_m, memory_dim]``,
        ``memory_mask [B, L_m]`` bool or None -> ``[B, L, dim]``
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 cross_attention: bool = False, memory_dim: int | None = None) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.has_cross_attention: bool = cross_attention
        if cross_attention:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = MultiHeadAttention(dim, num_heads, kv_dim=memory_dim, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None, memory: Tensor | None = None,
                memory_mask: Tensor | None = None) -> Tensor:
        h = self.norm_self(x)
        x = x + self.drop(self.self_attn(h, h, mask))
        if self.has_cross_attention:
            if memory is None:
                raise ValueError("this block has cross-attention and needs a memory")
            x = x + self.drop(self.cross_attn(self.norm_cross(x), memory, memory_mask))
        elif memory is not None:
            raise ValueError("this block has no cross-attention; memory must be None")
        return x + self.drop(self.mlp(self.norm_mlp(x)))


class EnsembleLinear(nn.Module):
    """n_ens independent affine maps evaluated in one batched product: y_e = x_e W_e + b_e.

    Each member is initialised independently with the ``nn.Linear`` scheme U(+-1/sqrt(in)), giving
    the independent initialisations a deep ensemble relies on.

    Shapes:
        ``x [n_ens, R, in_features]`` -> ``[n_ens, R, out_features]``
    """

    def __init__(self, num_members: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.num_members: int = num_members
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight = nn.Parameter(torch.empty(num_members, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_members, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.baddbmm(self.bias, x, self.weight)

    def extra_repr(self) -> str:
        return f"num_members={self.num_members}, in_features={self.in_features}, out_features={self.out_features}"


def masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    """Mean over the sequence axis of the real positions: ``[B, L, d]``, ``[B, L]`` -> ``[B, d]``."""
    if mask is None:
        return x.mean(dim=1)
    if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
        raise ValueError(f"expected a bool mask of shape {tuple(x.shape[:2])}, got {tuple(mask.shape)} {mask.dtype}")
    # Exclude padded rows with where(), not a product: they may hold non-finite values (0 * inf = NaN).
    kept = torch.where(mask.unsqueeze(-1), x, torch.zeros((), dtype=x.dtype, device=x.device))
    count = mask.sum(dim=1, keepdim=True).to(x.dtype)                         # [B, 1]
    return kept.sum(dim=1) / count.clamp_min(1.0)


def embed_modes(table: nn.Embedding, u: Tensor) -> Tensor:
    """Mode embedding for integer modes ``[B]`` or one-hot / soft mode vectors ``[B, |U|]``.

    A float ``u`` is contracted with the table, ``u @ E``, so a straight-through Gumbel-softmax
    sample (``ModeSelector``) propagates gradients into the selector. For an exact one-hot vector
    the result equals ``table(argmax u)``.

    Returns:
        ``[B, d]``.
    """
    if u.dtype.is_floating_point:
        if u.dim() != 2 or u.shape[1] != table.num_embeddings:
            raise ValueError(f"expected a mode distribution of shape [B, {table.num_embeddings}], got {tuple(u.shape)}")
        return u.to(table.weight.dtype) @ table.weight
    if u.dim() != 1 or u.dtype == torch.bool or u.is_complex():
        raise ValueError(f"expected integer modes of shape [B], got {tuple(u.shape)} {u.dtype}")
    if u.numel() and (bool((u < 0).any()) or bool((u >= table.num_embeddings).any())):
        raise ValueError(f"modes must lie in [0, {table.num_embeddings})")
    return table(u)
