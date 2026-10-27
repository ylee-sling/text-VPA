# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
selector.py -- pass 2 of PR-JEPA: the discrete mode selector pi_phi^h(u | s_x) (Sec. 3.1; Algorithm 1, line 2).

    u ~ pi_phi^h(. | s_x),   u in U = {0, ..., |U| - 1}

pi_phi^h is a non-autoregressive classifier: a masked mean of the real context latents s_x, a
LayerNorm, and a GELU MLP to |U| logits. One forward pass per generation.

Why the mode is sampled: by Prop. 1(a) a deterministic plan Z = g(c) leaves the total correlation
TC(W | c) -- the error of one factorised parallel step -- unchanged; sampling a discrete mode removes
its between-mode part, TC(W | c, U) = 0 when each mode's target factorises (Prop. 1(b)). Strategies:

    "sample"  u ~ softmax(logits / T)                      Algorithm 1 (T = 1); the default
    "greedy"  u = argmax logits                            deterministic decoding
    "gumbel"  u = argmax(logits + g), g ~ Gumbel(0, 1)^|U| (Gumbel-max: an exact draw from
              softmax(logits)); the returned one-hot is the straight-through Gumbel-softmax
              y = y_hard - sg(y_soft) + y_soft with y_soft = softmax((logits + g) / T), so gradients
              reach the logits when the one-hot feeds ``embed_modes`` (exploration in joint training)

Training uses ``ModeCrossEntropyLoss`` (text_vpa.engine.losses) on ``forward``'s logits against the
modes extracted from the reference (Sec. 5.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from text_vpa.models.layers import masked_mean

if TYPE_CHECKING:
    from text_vpa.configs.schema import PlannerConfig

__all__ = ["ModeSelection", "ModeSelector", "choose_mode"]

STRATEGIES: tuple[str, ...] = ("sample", "greedy", "gumbel")


class ModeSelection(NamedTuple):
    """A selected mode per batch element."""

    mode: Tensor      # [B] int64
    one_hot: Tensor   # [B, |U|] one-hot of ``mode``; straight-through (differentiable) for "gumbel"
    logits: Tensor    # [B, |U|]
    log_prob: Tensor  # [B] log pi_phi^h(mode | s_x), at temperature 1


def choose_mode(
    logits: Tensor,
    strategy: str = "sample",
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> ModeSelection:
    """Select modes from logits ``[B, |U|]`` (no network evaluation).

    ``generator`` must live on the logits' device (``torch.Generator(device=...)``).
    """
    if logits.dim() != 2:
        raise ValueError(f"expected logits of shape [B, |U|], got {tuple(logits.shape)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if not temperature > 0.0:
        raise ValueError("temperature must be positive")
    num_modes = logits.shape[1]
    if strategy == "greedy":
        mode = logits.argmax(dim=1)
        one_hot = F.one_hot(mode, num_modes).to(logits.dtype)
    elif strategy == "sample":
        probs = torch.softmax(logits.detach().float() / temperature, dim=1)
        mode = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        one_hot = F.one_hot(mode, num_modes).to(logits.dtype)
    else:  # gumbel
        # Noise and perturbed logits in at least float32: half-precision noise biases the draw.
        work = _at_least_float32(logits)
        uniform = torch.rand(work.shape, generator=generator, device=work.device, dtype=work.dtype)
        tiny = torch.finfo(work.dtype).tiny
        gumbel = -torch.log(-torch.log(uniform.clamp_min(tiny)))                # Gumbel(0, 1)
        perturbed = work + gumbel
        mode = perturbed.argmax(dim=1)                                          # Gumbel-max: independent of T
        y_soft = torch.softmax(perturbed / temperature, dim=1)
        y_hard = F.one_hot(mode, num_modes).to(work.dtype)
        one_hot = (y_hard - y_soft.detach() + y_soft).to(logits.dtype)          # straight-through
    log_prob = torch.log_softmax(_at_least_float32(logits), dim=1).gather(1, mode.unsqueeze(1)).squeeze(1)
    return ModeSelection(mode, one_hot, logits, log_prob.to(logits.dtype))


def _at_least_float32(x: Tensor) -> Tensor:
    """Upcast float16 / bfloat16 to float32; leave float32 and float64 unchanged."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class ModeSelector(nn.Module):
    """pi_phi^h: pooled s_x -> logits over the |U| design modes.

    Args:
        latent_dim: d, width of s_x.
        num_modes: |U|.
        hidden_dim: width of the MLP's hidden layers.
        num_layers: number of hidden layers (>= 1).
        dropout: dropout after each hidden layer.

    Shapes:
        ``forward(s_x [B, M_c, d], context_mask [B, M_c] bool or None)`` -> ``logits [B, |U|]``
        ``select(...)`` -> ``ModeSelection``
    """

    def __init__(self, latent_dim: int, num_modes: int, hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if num_modes < 2:
            raise ValueError("num_modes must be >= 2")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.latent_dim: int = latent_dim
        self.num_modes: int = num_modes
        self.norm = nn.LayerNorm(latent_dim)
        layers: list[nn.Module] = []
        width = latent_dim
        for _ in range(num_layers):
            layers += [nn.Linear(width, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            width = hidden_dim
        layers.append(nn.Linear(width, num_modes))
        self.mlp = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, cfg: PlannerConfig) -> ModeSelector:
        return cls(cfg.latent_dim, cfg.num_modes, cfg.selector.hidden_dim, cfg.selector.num_layers, cfg.selector.dropout)

    def forward(self, s_x: Tensor, context_mask: Tensor | None = None) -> Tensor:
        """Logits of pi_phi^h(. | s_x)."""
        if s_x.dim() != 3 or s_x.shape[-1] != self.latent_dim:
            raise ValueError(f"expected s_x of shape [B, M_c, {self.latent_dim}], got {tuple(s_x.shape)}")
        if context_mask is not None:
            if context_mask.dtype != torch.bool or context_mask.shape != s_x.shape[:2]:
                raise ValueError(f"expected a bool context_mask of shape {tuple(s_x.shape[:2])}")
            if not bool(context_mask.any(dim=1).all()):
                raise ValueError("every row of context_mask needs at least one real token")
        return self.mlp(self.norm(masked_mean(s_x, context_mask)))

    def select(
        self,
        s_x: Tensor,
        context_mask: Tensor | None = None,
        strategy: str = "sample",
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> ModeSelection:
        """Pass 2: one evaluation of pi_phi^h, then ``choose_mode``."""
        return choose_mode(self(s_x, context_mask), strategy, temperature, generator)
