# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
bridge.py -- the plan's way into the Stage 0 realizer: gated cross-attention in every block (docs/stage1.md).

Conditioning tokens, M~ = M (+ 1 with a mode) rows of width d:

    Z~ = [Z ; e(u)] / gamma                           (PR-JEPA Sec. 3.2 without the s_x rows: this realizer
                                                       already attends to the prompt's cached keys and values)

Block l of the realizer, on canvas positions only and after its self-attention, becomes

    h <- h + tanh(a_l) CA_l(norm_l(h), Z~),            a_l = 0 at initialisation

    CA_l(H, Z~) = softmax( Q_l (Z~ W_K^l)^T / sqrt(d_h) ) (Z~ W_V^l) W_O^l                      (Eq. 10)
    Q_l = H W_Q^l + P W_P^l                             P: the fixed code of each canvas position

so before any training the realizer with a plan is the realizer without one, exactly (tested: same hidden
states, same sampled tokens). W_K^l and W_V^l have no bias and are spectrally normalised (operator norm 1), and
1/gamma is a constant, so the conditioning is Lipschitz in the plan as in Sec. 3.2. The attention width is
``BridgeSpec.width`` (256, 8 heads), not the realizer's 1,536.

The position term is a choice the paper does not need: its realizer has learned positions, Gemma 4 has rotary
attention only, and a slot plan is useful to a canvas position only if the position can find the slot it lies
in. Canvas index i predicts canvas token i (the AR shift of ``stage0.backbone``), so P_i is the code of i, the
index that a slot's (start, length) refers to.

Keys and values depend on the plan alone: ``project`` computes them for all blocks once per generation
(``PlanCache``), every one of the K passes reuses them, and ``sampler.realize`` audits that (nothing projected
during the passes; every block of every pass attends). The projections belong to the first realizer pass and
add no sequential evaluation.

Plan dropout: in training each sample's Z~ is replaced with probability ``plan_dropout`` by one learned null
token (the other rows masked); ``null_conditioning`` is the same input for a whole batch: the "null plan".

Where the blocks are entered. ``attach`` wraps two frozen norms of every Gemma 4 block: ``input_layernorm`` (to
read the block's input x) and ``post_attention_layernorm`` (whose output y the block adds to x). With a plan the
second returns y + tanh(a_l) CA_l(norm_l(x + y), Z~), so the block's own addition yields the update above on the
residual stream. Both wrappers are inert unless a canvas pass runs with a plan, so the prompt cache and Stage 0
are untouched. The plan reaches them as an argument of ``Stage0Realizer._block`` (activation checkpointing calls
a block again in the backward pass).

Shapes: B batch, C = N + 1 canvas positions, d_r realizer width, w attention width, H heads, d_h = w / H.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm

from text_vpa.stage0.backbone import AdapterState, Stage0Realizer
from text_vpa.stage1.slots import position_code

__all__ = ["BridgeSpec", "Conditioning", "PlanBridge", "PlanCache"]


@dataclass(frozen=True)
class BridgeSpec:
    """Hyperparameters of the bridge (stored in every checkpoint that holds one)."""

    latent_dim: int = 256          # d: width of the plan's rows
    width: int = 256               # w: attention width, all heads
    num_heads: int = 8
    position_dim: int = 64         # width of the canvas-position code
    num_modes: int = 0             # |U|; 0: no e(u) row (the structure oracle carries no mode)
    gamma: float = 1.0
    plan_dropout: float = 0.1
    spectral_norm: bool = True
    query_position: bool = True
    power_iterations: int = 1

    def __post_init__(self) -> None:
        if self.width % self.num_heads or self.position_dim % 2:
            raise ValueError("width must be divisible by num_heads and position_dim must be even")
        if not self.gamma > 0 or not 0.0 <= self.plan_dropout < 1.0:
            raise ValueError("gamma must be positive and plan_dropout in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BridgeSpec:
        return cls(**data)


class Conditioning(NamedTuple):
    """The conditioning tokens of a batch."""

    tokens: Tensor     # [B, M~, d]  Z~, or the null token in row 0
    mask: Tensor       # [B, M~] bool, True at real rows
    dropped: Tensor    # [B] bool, True where Z~ was replaced by the null token


class PlanCache:
    """Keys and values of a plan for every block, computed once per generation.

    keys[l], values[l]: ``[B, H, M~, d_h]``; mask: ``[B, M~]`` bool. ``bridge`` is the module whose queries,
    gates and output maps the blocks use.
    """

    def __init__(self, bridge: PlanBridge, keys: tuple[Tensor, ...], values: tuple[Tensor, ...], mask: Tensor
                 ) -> None:
        self.bridge, self.keys, self.values, self.mask = bridge, keys, values, mask

    @property
    def batch_size(self) -> int:
        return int(self.mask.shape[0])

    def expand(self, rows: int) -> PlanCache:
        """The cache of one plan repeated ``rows`` times (views, no copy)."""
        if self.batch_size != 1:
            raise ValueError("expand needs the cache of one plan")
        return PlanCache(self.bridge, tuple(k.expand(rows, -1, -1, -1) for k in self.keys),
                         tuple(v.expand(rows, -1, -1, -1) for v in self.values), self.mask.expand(rows, -1))

    def audit(self) -> tuple[int, int]:
        """(projections, block cross-attentions) the bridge has run so far."""
        return self.bridge.passes["projection"], self.bridge.passes["cross"]

    def check_audit(self, before: tuple[int, int], canvas_passes: int) -> None:
        """Raise unless, since ``before``, nothing was projected and every block of ``canvas_passes`` passes
        attended to this cache: keys and values were computed once and reused (CLAUDE.md rule 3)."""
        projections = self.bridge.passes["projection"] - before[0]
        crossed = self.bridge.passes["cross"] - before[1]
        expected = self.bridge.num_layers * canvas_passes
        if projections != 0 or crossed != expected:
            raise RuntimeError(f"depth audit: {projections} plan projections during the realizer passes (must be "
                               f"0) and {crossed} block cross-attentions for {canvas_passes} passes (must be "
                               f"{expected})")


class _ResidualTap(nn.Module):
    """A block's first norm; with a plan it also records the block's input for ``_PlanInjection``."""

    def __init__(self, inner: nn.Module, state: AdapterState) -> None:
        super().__init__()
        self.inner = inner
        self._state = [state]                                    # in a list: not a submodule

    def forward(self, x: Tensor) -> Tensor:
        state = self._state[0]
        if state.active and state.plan is not None:
            state.residual = x
        return self.inner(x)


class _PlanInjection(nn.Module):
    """A block's norm after self-attention; with a plan its output also carries the gated cross-attention."""

    def __init__(self, inner: nn.Module, state: AdapterState, layer_idx: int) -> None:
        super().__init__()
        self.inner = inner
        self._state = [state]
        self.layer_idx = layer_idx

    def forward(self, x: Tensor) -> Tensor:
        y = self.inner(x)
        state = self._state[0]
        plan = state.plan
        if not state.active or plan is None:
            return y
        if state.residual is None:
            raise RuntimeError("the bridge did not see this block's input (was the realizer rebuilt after attach?)")
        hidden = state.residual + y                              # the residual stream after self-attention
        state.residual = None
        return y + plan.bridge.cross_attention(self.layer_idx, hidden, plan).to(y.dtype)


class PlanBridge(nn.Module):
    """Gated cross-attention from every realizer block to the plan.

    Args:
        spec: ``BridgeSpec``.
        hidden_size: d_r, the realizer's width.
        num_layers: L, the realizer's blocks.

    Shapes:
        ``conditioning(z [B, M, d], mask [B, M], mode [B] | None)`` -> ``Conditioning``
        ``project(conditioning)`` -> ``PlanCache`` (L x ``[B, H, M~, d_h]`` keys and values)
        ``cross_attention(layer, hidden [B, C, d_r], cache)`` -> ``[B, C, d_r]`` (already gated)
    """

    def __init__(self, spec: BridgeSpec, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.spec, self.hidden_size, self.num_layers = spec, hidden_size, num_layers
        d, w = spec.latent_dim, spec.width

        def plan_projection() -> nn.Module:
            layer = nn.Linear(d, w, bias=False)
            return _spectral_norm(layer, n_power_iterations=spec.power_iterations) if spec.spectral_norm else layer

        self.norm = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.query_proj = nn.ModuleList(nn.Linear(hidden_size, w, bias=False) for _ in range(num_layers))   # W_Q
        self.key_proj = nn.ModuleList(plan_projection() for _ in range(num_layers))                         # W_K
        self.value_proj = nn.ModuleList(plan_projection() for _ in range(num_layers))                       # W_V
        self.out_proj = nn.ModuleList(nn.Linear(w, hidden_size, bias=False) for _ in range(num_layers))     # W_O
        self.position_proj = nn.ModuleList(
            nn.Linear(spec.position_dim, w, bias=False) for _ in range(num_layers if spec.query_position else 0))
        self.gate = nn.Parameter(torch.zeros(num_layers))                                                   # a_l
        self.null_token = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.null_token, std=0.02)
        self.mode_embedding: nn.Embedding | None = None
        if spec.num_modes > 0:
            self.mode_embedding = nn.Embedding(spec.num_modes, d)                                           # e(u)
            nn.init.normal_(self.mode_embedding.weight, std=0.02)
        self.passes: dict[str, int] = {"projection": 0, "cross": 0}

    # -- installation --------------------------------------------------------------------------
    def attach(self, realizer: Stage0Realizer) -> None:
        """Enter every block of ``realizer`` (see the module docstring). The realizer's own parameters, its
        checkpoint format and its behaviour without a plan do not change."""
        if realizer.num_layers != self.num_layers or realizer.hidden_size != self.hidden_size:
            raise ValueError(f"the bridge was built for {self.num_layers} blocks of width {self.hidden_size}, the "
                             f"realizer has {realizer.num_layers} of width {realizer.hidden_size}")
        for index, layer in enumerate(realizer.text.layers):
            if isinstance(layer.post_attention_layernorm, _PlanInjection):
                raise RuntimeError("a bridge is already attached to this realizer")
            layer.input_layernorm = _ResidualTap(layer.input_layernorm, realizer.state)
            layer.post_attention_layernorm = _PlanInjection(layer.post_attention_layernorm, realizer.state, index)

    def gates(self) -> Tensor:
        """tanh(a_l), ``[L]``."""
        return torch.tanh(self.gate)

    # -- conditioning --------------------------------------------------------------------------
    def conditioning(self, z: Tensor, mask: Tensor, mode: Tensor | None = None, drop: Tensor | None = None,
                     generator: torch.Generator | None = None) -> Conditioning:
        """Z~ = [Z ; e(u)] / gamma, with plan dropout in training.

        Args:
            z: ``[B, M, d]`` slot latents.
            mask: ``[B, M]`` bool, True at real slots.
            mode: ``[B]`` int64 design modes; required iff the bridge has a mode embedding.
            drop: ``[B]`` bool, overrides the dropout draw (None: Bernoulli(plan_dropout) per sample in
                training mode, nothing in eval mode).
            generator: CPU generator of the dropout draw.
        """
        if z.dim() != 3 or z.shape[-1] != self.spec.latent_dim or mask.shape != z.shape[:2]:
            raise ValueError(f"expected z [B, M, {self.spec.latent_dim}] and mask [B, M], got {tuple(z.shape)} "
                             f"and {tuple(mask.shape)}")
        if (mode is None) != (self.mode_embedding is None):
            raise ValueError("a mode is required exactly when the bridge has a mode embedding (num_modes > 0)")
        b = z.shape[0]
        tokens, mask = z, mask.bool()
        if mode is not None:
            row = self.mode_embedding(mode.to(z.device))[:, None, :].to(z.dtype)                 # [B, 1, d]
            tokens = torch.cat([tokens, row], dim=1)
            mask = torch.cat([mask, torch.ones(b, 1, dtype=torch.bool, device=mask.device)], dim=1)
        tokens = tokens / self.spec.gamma
        if drop is None:
            if self.training and self.spec.plan_dropout > 0.0:
                drop = torch.rand(b, generator=generator) < self.spec.plan_dropout
            else:
                drop = torch.zeros(b, dtype=torch.bool)
        elif drop.shape != (b,) or drop.dtype != torch.bool:
            raise ValueError(f"expected a bool drop mask of shape [{b}]")
        drop = drop.to(tokens.device)
        if bool(drop.any()):
            null_tokens = torch.zeros_like(tokens)
            null_tokens[:, 0] = self.null_token.to(tokens.dtype)
            null_mask = torch.zeros_like(mask)
            null_mask[:, 0] = True
            tokens = torch.where(drop[:, None, None], null_tokens, tokens)
            mask = torch.where(drop[:, None], null_mask, mask)
        return Conditioning(tokens, mask, drop)

    def null_conditioning(self, batch_size: int) -> Conditioning:
        """The null plan: the learned null token as the only row, ``[B, 1, d]``."""
        tokens = self.null_token[None, None, :].expand(batch_size, 1, -1)
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        return Conditioning(tokens, mask, torch.ones(batch_size, dtype=torch.bool, device=tokens.device))

    def project(self, conditioning: Conditioning) -> PlanCache:
        """K_l = Z~ W_K^l and V_l = Z~ W_V^l for every block, once per generation (float32: the power
        iteration of the spectral norm is not run in bfloat16)."""
        tokens, mask = conditioning.tokens, conditioning.mask
        if tokens.dim() != 3 or tokens.shape[-1] != self.spec.latent_dim:
            raise ValueError(f"expected conditioning tokens [B, M~, {self.spec.latent_dim}]")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sample needs at least one real conditioning row")
        b, m, _ = tokens.shape
        heads = self.spec.num_heads
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            tokens = tokens.float()
            keys = tuple(proj(tokens).view(b, m, heads, -1).transpose(1, 2) for proj in self.key_proj)
            values = tuple(proj(tokens).view(b, m, heads, -1).transpose(1, 2) for proj in self.value_proj)
        self.passes["projection"] += 1
        return PlanCache(self, keys, values, mask)

    # -- the block's cross-attention -----------------------------------------------------------
    def cross_attention(self, layer: int, hidden: Tensor, cache: PlanCache) -> Tensor:
        """tanh(a_l) CA_l(norm_l(hidden), Z~) for block ``layer``: ``[B, C, d_r]`` -> ``[B, C, d_r]``."""
        if cache.bridge is not self or len(cache.keys) != self.num_layers:
            raise ValueError("this cache was projected by another bridge")
        if hidden.dim() != 3 or hidden.shape[-1] != self.hidden_size or hidden.shape[0] != cache.batch_size:
            raise ValueError(f"expected hidden [B, C, {self.hidden_size}] with the cache's batch size, got "
                             f"{tuple(hidden.shape)} for {cache.batch_size} plans")
        b, c, _ = hidden.shape
        heads = self.spec.num_heads
        # The backbone may run in bfloat16 without autocast (a CPU run of the released checkpoint): the bridge's
        # own parameters are float32, as autocast would make this norm on a GPU.
        hidden = hidden.to(self.gate.dtype)
        query = self.query_proj[layer](self.norm[layer](hidden))                                 # [B, C, w]
        if self.spec.query_position:
            code = position_code(torch.arange(c, device=hidden.device), self.spec.position_dim)  # [C, p]
            query = query + self.position_proj[layer](code.to(query.dtype))[None]
        query = query.view(b, c, heads, -1).transpose(1, 2)                                      # [B, H, C, d_h]
        keys, values = cache.keys[layer].to(query.dtype), cache.values[layer].to(query.dtype)
        out = F.scaled_dot_product_attention(query, keys, values, attn_mask=cache.mask[:, None, None, :])
        out = self.out_proj[layer](out.transpose(1, 2).reshape(b, c, -1))                        # [B, C, d_r]
        self.passes["cross"] += 1
        return torch.tanh(self.gate[layer]).to(out.dtype) * out
