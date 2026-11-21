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

``order="left_to_right"`` (a diagnostic, semi-autoregressive) reveals the same number of positions as the
confidence order, but always the leftmost masked ones. With the cosine schedule and N = 256 that is blocks of
5, 15, 24, 32, ... tokens at K = 8; one token per step for the first 42 positions, then 2 to 3, at K = 128;
and exactly one token per step for all 256 positions at K = 256: the adapted model decoding autoregressively. With ``stop_id`` it stops once every row has revealed that token (its
<eos>): revealed positions never change and everything after the first <eos> is discarded, so the output is the
same as without the stop; only passes that cannot change it are skipped. ``row_done`` (checked every
``check_every`` steps) can end a row earlier, e.g. once the truncation rule has found its stop in the revealed
prefix, which likewise fixes the program.

``adapters=False`` with ``mode="causal"`` and one token per step is the frozen backbone's own autoregressive
decoding, computed through this code path: the check that the realizer's implementation reproduces the
baseline on the real model.

The (t, h) the network is given. ``time_input="schedule"`` (default): t_k = k/K and h = 1/K, as in Algorithm 1.
The realizer is trained only at t = k/K, h = 1/K for its step budgets (K in {2, 4, 8}: t a multiple of 1/8, h
in {1/2, 1/4, 1/8}), and enters them as sinusoidal features of 1000 t and 1000 h, so K > 8 gives it values it
never saw (h = 1/256 -> 3.9 against 125 to 500). ``time_input="matched"`` gives it a pair from training instead:
h = 1/``time_grid`` and, per row, the grid time t = j/``time_grid`` whose kappa_t is closest to the fraction of
the canvas already revealed (with the mask source a training state at t has a fraction kappa_t revealed). The
reveal schedule itself is unchanged.

Token draw (both orders, and the autoregressive baseline through ``generate``): temperature, then the
``top_k`` most probable tokens, then the smallest nucleus of mass >= ``top_p`` among them. temperature 0 is the
argmax.

Depth: one prefill of the prompt by the frozen backbone and K canvas passes, K + 1 sequential evaluations
(fewer for a left-to-right row that has stopped).

With a plan (the planner stage, ``text_vpa.stage1.bridge``): ``plan`` holds the keys and values of the plan for
every block, projected once before the first step, and each canvas pass attends to them. The depth audit then
also checks that nothing was projected again during the K passes and that every block of every pass used the
cache; the projections are part of the first canvas pass and add no sequential evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from text_vpa.models.realizers.paths import KappaSchedule, uniform_time_grid
from text_vpa.stage0.backbone import PrefixCache, Stage0Realizer

__all__ = ["ORDERS", "TIME_INPUTS", "RealizeResult", "SamplerConfig", "draw_tokens", "network_time", "realize",
           "reveal_counts"]

ORDERS: tuple[str, ...] = ("random", "confidence", "left_to_right")
TIME_INPUTS: tuple[str, ...] = ("schedule", "matched")


@dataclass(frozen=True)
class SamplerConfig:
    num_steps: int = 8
    schedule: str = "cosine"
    order: str = "random"          # "random": the exact jump kernel; "confidence"; "left_to_right"
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 1024
    chunk: int = 512               # positions whose logits are held at once
    time_input: str = "schedule"   # "schedule": t = k/K, h = 1/K; "matched": a (t, h) pair seen in training
    time_grid: int = 8             # the finest trained step budget (matched: t on its grid, h = 1/time_grid)

    def __post_init__(self) -> None:
        if self.num_steps < 1 or self.order not in ORDERS:
            raise ValueError(f"need num_steps >= 1 and order in {ORDERS}")
        if self.time_input not in TIME_INPUTS or self.time_grid < 1:
            raise ValueError(f"need time_input in {TIME_INPUTS} and time_grid >= 1")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k < 1:
            raise ValueError("need temperature >= 0, 0 < top_p <= 1 and top_k >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RealizeResult:
    tokens: Tensor                 # [B, N] int64, no [MASK]
    revealed: list[int]            # positions revealed at each executed step (summed over the batch)
    canvas_passes: int             # canvas passes executed for the batch: K, or fewer after a left-to-right stop
    sequential_depth: int          # canvas_passes + 1 (prefill + canvas passes)
    row_passes: Tensor             # [B] int64, canvas passes each row needed (K unless it stopped)


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


def network_time(cfg: SamplerConfig, schedule: KappaSchedule, t: float, h: float, masked: Tensor
                 ) -> tuple[Tensor, Tensor]:
    """The (t, h) given to the network at a step, ``[B]`` each (see the module docstring).

    ``masked`` is ``[B, N]`` bool, the canvas before the step's reveal.
    """
    b, device = masked.shape[0], masked.device
    if cfg.time_input == "schedule":
        return torch.full((b,), t, device=device), torch.full((b,), h, device=device)
    grid = torch.arange(cfg.time_grid, device=device, dtype=torch.float32) / cfg.time_grid     # t seen in training
    revealed = 1.0 - masked.float().mean(dim=1)                                                # [B]
    nearest = (schedule.kappa(grid)[None, :] - revealed[:, None]).abs().argmin(dim=1)
    return grid[nearest], torch.full((b,), 1.0 / cfg.time_grid, device=device)


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
            mode: str = "bidirectional", stop_id: int | None = None, adapters: bool = True,
            row_done: Callable[[Tensor], Tensor] | None = None, check_every: int = 8,
            plan: Any = None) -> RealizeResult:
    """K parallel steps from the all-[MASK] canvas.

    Args:
        cache: the prompt prefix's keys and values, one row per sample.
        anchor: ``[B]`` the last prompt token of each sample (clamped at canvas index 0).
        canvas_len: N.
        forbidden: token ids that are never drawn in addition to [MASK] (e.g. <pad>, <bos>).
        generator: on the realizer's device.
        stop_id: left-to-right order only: stop once every row has revealed this token; the remaining positions
            are filled with it.
        adapters: False runs the frozen backbone on the canvas (LoRA and modulation off).
        row_done: left-to-right order only, needs ``stop_id``: ``[B, N]`` tokens (a revealed prefix, then [MASK])
            -> ``[B]`` bool, True once a row's output is final; checked every ``check_every`` steps.
        plan: a ``PlanCache`` with one row per sample (the planner stage); None is Stage 0.
    """
    if row_done is not None and stop_id is None:
        raise ValueError("row_done needs stop_id (it fills the rest of a finished row)")
    mask_id = realizer.spec.mask_id
    device = anchor.device
    b = anchor.shape[0]
    banned = sorted({mask_id, *forbidden})
    schedule = KappaSchedule(cfg.schedule)
    grid = uniform_time_grid(cfg.num_steps)
    tokens = torch.full((b, canvas_len), mask_id, dtype=torch.long, device=device)
    front = torch.zeros(b, 1, dtype=torch.bool, device=device)
    passes_before = realizer.passes["canvas"]
    with_plan = {} if plan is None else {"plan": plan}
    plan_before = None if plan is None else plan.audit()
    revealed: list[int] = []
    row_passes = torch.full((b,), cfg.num_steps, dtype=torch.long, device=device)
    stopped = torch.zeros(b, dtype=torch.bool, device=device)
    executed = 0
    for k, (t, h) in enumerate(grid):
        last = k == len(grid) - 1
        masked = tokens == mask_id
        t_in, h_in = network_time(cfg, schedule, t, h, masked)
        hidden = realizer.canvas_hidden(
            cache, torch.cat([anchor[:, None], tokens], dim=1), torch.cat([front, masked], dim=1), t_in, h_in,
            mode=mode, adapters=adapters, **with_plan,
        )[:, :canvas_len]                                           # index j predicts canvas token j + 1
        pi = 1.0 if last else float(schedule.exact_jump_mass(t, h))
        if cfg.order == "random":
            coin = torch.rand(masked.shape, generator=generator, device=device)
            jump = masked if last else masked & (coin < pi)
            drawn, _ = _draw_in_chunks(realizer, hidden[jump], cfg, banned, generator)
            tokens[jump] = drawn
        elif cfg.order == "left_to_right":
            count = reveal_counts(masked.sum(dim=1), pi, last)
            jump = masked & (masked.long().cumsum(dim=1) <= count[:, None])      # the leftmost `count` masked
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
        executed += 1
        if cfg.order == "left_to_right" and stop_id is not None and not last:
            # Revealed positions are a prefix, so a revealed stop token ends the row's output.
            now = (tokens == stop_id).any(dim=1)
            if row_done is not None and (k + 1) % check_every == 0:
                now = now | row_done(tokens).to(device)
            now = now & ~stopped
            row_passes[now] = k + 1
            stopped |= now
            if bool(stopped.all()):
                tokens = torch.where(tokens == mask_id, torch.full_like(tokens, stop_id), tokens)
                break
    if bool((tokens == mask_id).any()):
        raise RuntimeError("positions are still [MASK] after the last step")
    canvas_passes = realizer.passes["canvas"] - passes_before
    if canvas_passes != executed or (stop_id is None and canvas_passes != cfg.num_steps):
        raise RuntimeError(f"depth audit: {canvas_passes} canvas passes for K = {cfg.num_steps}")
    if plan is not None:
        plan.check_audit(plan_before, canvas_passes)
    return RealizeResult(tokens, revealed, canvas_passes, canvas_passes + 1, row_passes)
