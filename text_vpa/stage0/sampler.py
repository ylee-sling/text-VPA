# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sampler.py -- K parallel realizer steps on the mask source (Text-VPA Algorithm 1 without planner or energies).

Starting from w_0 = [MASK]^N, step k = 0 .. K-1 evaluates the realizer once at t_k = k/K, h = 1/K and applies
the exact few-step jump kernel (Eq. 15; ``models/realizers/kernels.py`` with no energy tilt): every position
still at [MASK] jumps with probability

    pi_{t,h} = (kappa_{t+h} - kappa_t) / (1 - kappa_t)

to a token drawn from the posterior p_{1|t}, and keeps [MASK] otherwise. [MASK] is not a possible value of w_1
(its posterior is set to zero), so a jump always leaves the source and revealed positions never change: with
the mask source the kernel's jump mass is exactly pi. The last step has pi = 1 and reveals everything left.

Because only jumping positions need a token, the S = 262,144 logits are computed for those positions alone
(``order="random"``, the kernel above; tested against ``kernels.kernel_distribution``).

``order="confidence"`` is a secondary sampler, not the paper's kernel: step k reveals the same expected number
of positions, round(pi * masked), but takes the ones whose drawn token has the highest posterior probability
(at temperature 1), as in MaskGIT.

Token draw (both orders, and the autoregressive baseline through ``generate``): temperature, then the
``top_k`` most probable tokens, then the smallest nucleus of mass >= ``top_p`` among them. temperature 0 is the
argmax.

Depth: one prefill of the prompt by the frozen backbone and K canvas passes, K + 1 sequential evaluations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from text_vpa.models.realizers.paths import KappaSchedule, uniform_time_grid
from text_vpa.stage0.backbone import PrefixCache, Stage0Realizer

__all__ = ["RealizeResult", "SamplerConfig", "draw_tokens", "realize", "reveal_counts"]


@dataclass(frozen=True)
class SamplerConfig:
    num_steps: int = 8
    schedule: str = "cosine"
    order: str = "random"          # "random": the exact jump kernel; "confidence": most confident first
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 1024
    chunk: int = 512               # positions whose logits are held at once

    def __post_init__(self) -> None:
        if self.num_steps < 1 or self.order not in ("random", "confidence"):
            raise ValueError("need num_steps >= 1 and order in {'random', 'confidence'}")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k < 1:
            raise ValueError("need temperature >= 0, 0 < top_p <= 1 and top_k >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RealizeResult:
    tokens: Tensor                 # [B, N] int64, no [MASK]
    revealed: list[int]            # positions revealed at each step (summed over the batch)
    canvas_passes: int             # K
    sequential_depth: int          # K + 1 (prefill + canvas passes)


def draw_tokens(logits: Tensor, temperature: float, top_p: float, top_k: int, forbidden: Sequence[int] = (),
                generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """One token per row, and the posterior probability of that token (temperature 1, before top-k and top-p;
    the confidence order ranks positions by it, so it must not be flattened to 1 by a low temperature).

    Shapes: logits ``[M, S]`` -> tokens ``[M]`` int64, probability ``[M]`` float32.
    """
    logits = logits.float()
    if forbidden:
        logits = logits.index_fill(-1, torch.as_tensor(list(forbidden), device=logits.device), float("-inf"))
    log_norm = torch.logsumexp(logits, dim=-1)
    if temperature == 0.0:
        tokens = logits.argmax(dim=-1)
    else:
        top, index = torch.topk(logits / temperature, min(top_k, logits.shape[-1]), dim=-1)  # sorted, descending
        probs = torch.softmax(top, dim=-1)
        before = probs.cumsum(dim=-1) - probs                    # mass of the more probable tokens
        probs = torch.where(before < top_p, probs, torch.zeros_like(probs))
        choice = torch.multinomial(probs / probs.sum(dim=-1, keepdim=True), 1, generator=generator)
        tokens = index.gather(-1, choice)[:, 0]
    return tokens, torch.exp(logits.gather(-1, tokens[:, None])[:, 0] - log_norm)


def reveal_counts(masked: Tensor, pi: float, last: bool) -> Tensor:
    """Positions the confidence order reveals per row: round(pi * masked), at least 1 while any is masked, and
    all of them at the last step. ``masked`` is ``[B]`` int64."""
    if last:
        return masked.clone()
    count = torch.round(masked.double() * pi).long().clamp(min=1)
    return torch.minimum(count, masked)


def _draw_in_chunks(realizer: Stage0Realizer, hidden: Tensor, cfg: SamplerConfig, forbidden: Sequence[int],
                    generator: torch.Generator | None) -> tuple[Tensor, Tensor]:
    tokens, probs = [], []
    for start in range(0, hidden.shape[0], cfg.chunk):
        tok, p = draw_tokens(realizer.logits(hidden[start:start + cfg.chunk]), cfg.temperature, cfg.top_p,
                             cfg.top_k, forbidden, generator)
        tokens.append(tok)
        probs.append(p)
    if not tokens:
        empty = torch.zeros(0, device=hidden.device)
        return empty.long(), empty.float()
    return torch.cat(tokens), torch.cat(probs)


@torch.no_grad()
def realize(realizer: Stage0Realizer, cache: PrefixCache, anchor: Tensor, canvas_len: int, cfg: SamplerConfig,
            forbidden: Sequence[int] = (), generator: torch.Generator | None = None,
            mode: str = "bidirectional") -> RealizeResult:
    """K parallel steps from the all-[MASK] canvas.

    Args:
        cache: the prompt prefix's keys and values, one row per sample.
        anchor: ``[B]`` the last prompt token of each sample (clamped at canvas index 0).
        canvas_len: N.
        forbidden: token ids that are never drawn in addition to [MASK] (e.g. <pad>, <bos>).
        generator: on the realizer's device.
    """
    mask_id = realizer.spec.mask_id
    device = anchor.device
    b = anchor.shape[0]
    banned = sorted({mask_id, *forbidden})
    schedule = KappaSchedule(cfg.schedule)
    grid = uniform_time_grid(cfg.num_steps)
    tokens = torch.full((b, canvas_len), mask_id, dtype=torch.long, device=device)
    front = torch.zeros(b, 1, dtype=torch.bool, device=device)
    passes_before = realizer.passes["canvas"]
    revealed: list[int] = []
    for k, (t, h) in enumerate(grid):
        last = k == len(grid) - 1
        masked = tokens == mask_id
        hidden = realizer.canvas_hidden(
            cache, torch.cat([anchor[:, None], tokens], dim=1), torch.cat([front, masked], dim=1),
            torch.full((b,), t, device=device), torch.full((b,), h, device=device), mode=mode,
        )[:, :canvas_len]                                           # index j predicts canvas token j + 1
        pi = 1.0 if last else float(schedule.exact_jump_mass(t, h))
        if cfg.order == "random":
            coin = torch.rand(masked.shape, generator=generator, device=device)
            jump = masked if last else masked & (coin < pi)
            drawn, _ = _draw_in_chunks(realizer, hidden[jump], cfg, banned, generator)
            tokens[jump] = drawn
        else:
            drawn, prob = _draw_in_chunks(realizer, hidden[masked], cfg, banned, generator)
            score = torch.full(masked.shape, -1.0, device=device)
            score[masked] = prob
            proposal = tokens.clone()
            proposal[masked] = drawn
            count = reveal_counts(masked.sum(dim=1), pi, last)
            rank = score.argsort(dim=1, descending=True).argsort(dim=1)      # 0 = most confident
            jump = masked & (rank < count[:, None])
            tokens = torch.where(jump, proposal, tokens)
        revealed.append(int(jump.sum()))
    if bool((tokens == mask_id).any()):
        raise RuntimeError("positions are still [MASK] after the last step")
    canvas_passes = realizer.passes["canvas"] - passes_before
    if canvas_passes != cfg.num_steps:
        raise RuntimeError(f"depth audit: {canvas_passes} canvas passes for K = {cfg.num_steps}")
    return RealizeResult(tokens, revealed, canvas_passes, canvas_passes + 1)
