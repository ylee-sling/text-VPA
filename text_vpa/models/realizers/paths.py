# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
paths.py -- probability paths and step kernels of the two realizers (PR-JEPA Secs. 3.3-3.4).

Conditional FS-DFM (Sec. 3.3). Each position follows the mixture path (Eq. 11)

    p_t(w^i | w_0^i, w_1^i) = (1 - kappa_t) delta_{w_0^i} + kappa_t delta_{w_1^i},   kappa_0 = 0, kappa_1 = 1,

with schedules  linear kappa_t = t,  sqrt kappa_t = t^(1/2),  cosine kappa_t = 1 - cos(pi t / 2).
Holding the posterior fixed over [t, t + h], the exact few-step jump mass is (Eq. 15)

    pi_{t,h} = (kappa_{t+h} - kappa_t) / (1 - kappa_t)  in [0, 1],

and Lemma 1 states that with the oracle posterior the kernel maps p_t to p_{t+h} exactly:
kappa_t + (1 - kappa_t) pi_{t,h} = kappa_{t+h}. The Euler jump mass h kappa_dot_t / (1 - kappa_t) is
a probability only when it is <= 1, which the sqrt schedule violates near t = 0 ("Why not Euler").
On the uniform grid t_k = k/K the linear schedule gives pi_{t_k,1/K} = 1/(K - k), whose sum over
the K steps is the harmonic number H_K (Prop. 5).

Logit-KL (Sec. 3.4). Smoothed one-hot targets have logits l_1 = a e_k + b 1 with (Eq. 18)

    a = log((1 - eps + eps/S) / (eps/S)),   b = log(eps/S);

the path is l_t = (1 - t) l_0 + t l_1 (Eq. 19), with conditional velocity
l_1 - l_0 = (l_1 - l_t)/(1 - t), and the marginal velocity is affine in the posterior (Lemma 2, Eq. 20)

    v*(l, t) = (a p(. | l_t = l) + b 1 - l) / (1 - t).

The Euler sampler (Eq. 21) drops b 1 (softmax ignores constant shifts); at the last grid step
t_{K-1} = 1 - 1/K it returns l_K = a p exactly, so the singularity at t = 1 is never evaluated.

These functions are shared by the realizers and by ``tests/test_theory_verification.py``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, overload

import torch
from torch import Tensor

__all__ = [
    "KappaSchedule",
    "MixtureSample",
    "conditional_logit_velocity",
    "logit_euler_step",
    "logit_path",
    "mixture_path_sample",
    "posterior_logit_velocity",
    "smoothed_logit_coefficients",
    "smoothed_target_logits",
    "uniform_time_grid",
]

SCHEDULES: tuple[str, ...] = ("linear", "sqrt", "cosine")


def uniform_time_grid(num_steps: int) -> list[tuple[float, float]]:
    """``[(t_k, h)]`` for t_k = k/K, h = 1/K, k = 0..K-1 (Algorithm 1, line 8)."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    h = 1.0 / num_steps
    return [(k * h, h) for k in range(num_steps)]


class KappaSchedule:
    """Mixture-path schedule kappa_t on [0, 1] with kappa_0 = 0, kappa_1 = 1, kappa_dot >= 0 (Eq. 11).

    Methods accept a Python float or a tensor and return the same kind.
    """

    def __init__(self, kind: str = "linear") -> None:
        if kind not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {kind!r}")
        self.kind: str = kind

    def __repr__(self) -> str:
        return f"KappaSchedule({self.kind!r})"

    @overload
    def kappa(self, t: float) -> float: ...
    @overload
    def kappa(self, t: Tensor) -> Tensor: ...

    def kappa(self, t: float | Tensor) -> float | Tensor:
        """kappa_t."""
        if isinstance(t, Tensor):
            if self.kind == "linear":
                return t.clone()
            if self.kind == "sqrt":
                return t.clamp_min(0.0).sqrt()
            return 1.0 - torch.cos(math.pi * t / 2.0)
        if self.kind == "linear":
            return float(t)
        if self.kind == "sqrt":
            return math.sqrt(max(t, 0.0))
        return 1.0 - math.cos(math.pi * t / 2.0)

    @overload
    def kappa_dot(self, t: float) -> float: ...
    @overload
    def kappa_dot(self, t: Tensor) -> Tensor: ...

    def kappa_dot(self, t: float | Tensor) -> float | Tensor:
        """d kappa_t / dt (``inf`` at t = 0 for the sqrt schedule)."""
        if isinstance(t, Tensor):
            if self.kind == "linear":
                return torch.ones_like(t)
            if self.kind == "sqrt":
                return 0.5 / t.sqrt()
            return (math.pi / 2.0) * torch.sin(math.pi * t / 2.0)
        if self.kind == "linear":
            return 1.0
        if self.kind == "sqrt":
            return math.inf if t <= 0.0 else 0.5 / math.sqrt(t)
        return (math.pi / 2.0) * math.sin(math.pi * t / 2.0)

    @overload
    def exact_jump_mass(self, t: float, h: float) -> float: ...
    @overload
    def exact_jump_mass(self, t: Tensor, h: float | Tensor) -> Tensor: ...

    def exact_jump_mass(self, t: float | Tensor, h: float | Tensor) -> float | Tensor:
        """pi_{t,h} = (kappa_{t+h} - kappa_t) / (1 - kappa_t)  (Eq. 15); t + h is clipped to 1.

        Requires kappa_t < 1, i.e. t < 1: the sampler never evaluates t = 1.
        """
        if isinstance(t, Tensor):
            end = (t + h).clamp_max(1.0)
            k_t = self.kappa(t)
            if bool((t >= 1.0).any()) or bool((k_t >= 1.0).any()):
                raise ValueError("pi_{t,h} is undefined where kappa_t = 1 (t = 1)")
            return (self.kappa(end) - k_t) / (1.0 - k_t)
        k_t = self.kappa(t)
        if t >= 1.0 or k_t >= 1.0:
            raise ValueError("pi_{t,h} is undefined where kappa_t = 1 (t = 1)")
        return (self.kappa(min(t + float(h), 1.0)) - k_t) / (1.0 - k_t)

    @overload
    def euler_jump_mass(self, t: float, h: float) -> float: ...
    @overload
    def euler_jump_mass(self, t: Tensor, h: float | Tensor) -> Tensor: ...

    def euler_jump_mass(self, t: float | Tensor, h: float | Tensor) -> float | Tensor:
        """h kappa_dot_t / (1 - kappa_t): the Euler kernel's jump mass, a probability only if <= 1."""
        if isinstance(t, Tensor):
            return h * self.kappa_dot(t) / (1.0 - self.kappa(t))
        return float(h) * self.kappa_dot(t) / (1.0 - self.kappa(t))

    def step_masses(self, num_steps: int) -> list[float]:
        """pi_{t_k, 1/K} on the uniform grid, k = 0..K-1 (one per realizer pass)."""
        return [self.exact_jump_mass(t, h) for t, h in uniform_time_grid(num_steps)]


# ---------------------------------------------------------------------------------------------
# Mixture-path training states (Eqs. 11-12)
# ---------------------------------------------------------------------------------------------
class MixtureSample(NamedTuple):
    """A training state on the mixture path."""

    w_t: Tensor     # [B, N] int64
    t: Tensor       # [B] float32 flow time
    h: Tensor       # [B] float32 step size 1/K
    w_0: Tensor     # [B, N] int64 source tokens


def mixture_path_sample(
    w_1: Tensor,
    schedule: KappaSchedule,
    vocab_size: int,
    step_budgets: tuple[int, ...] | list[int],
    source: str = "uniform",
    mask_id: int | None = None,
    time_sampling: str = "grid",
    generator: torch.Generator | None = None,
) -> MixtureSample:
    """Draw (w_t, t, h) for Eq. 12: w_t^i = w_1^i with probability kappa_t, else w_0^i (Eq. 11).

    ``time_sampling="grid"`` draws K from ``step_budgets`` and t = k/K, k ~ U{0..K-1}: exactly the
    states Algorithm 1 evaluates at. ``"uniform"`` draws t ~ U[0, 1) (h still from the budgets).
    """
    if w_1.dim() != 2:
        raise ValueError(f"expected w_1 of shape [B, N], got {tuple(w_1.shape)}")
    b, n = w_1.shape
    device = w_1.device
    budgets = torch.tensor(list(step_budgets), dtype=torch.float32)
    k_steps = budgets[torch.randint(len(budgets), (b,), generator=generator)]
    if time_sampling == "grid":
        k = torch.floor(torch.rand(b, generator=generator) * k_steps)
        t = k / k_steps
    elif time_sampling == "uniform":
        t = torch.rand(b, generator=generator)
    else:
        raise ValueError("time_sampling must be 'grid' or 'uniform'")
    h = 1.0 / k_steps
    if source == "uniform":
        w_0 = torch.randint(vocab_size, (b, n), generator=generator)
    elif source == "mask":
        if mask_id is None:
            raise ValueError("the mask source needs mask_id")
        w_0 = torch.full((b, n), mask_id, dtype=torch.long)
    else:
        raise ValueError("source must be 'uniform' or 'mask'")
    kappa = schedule.kappa(t)                                                  # [B]
    arrived = torch.rand(b, n, generator=generator) < kappa[:, None]
    w_t = torch.where(arrived, w_1.cpu(), w_0)
    return MixtureSample(w_t.to(device), t.to(device), h.to(device), w_0.to(device))


# ---------------------------------------------------------------------------------------------
# Logit-KL path (Sec. 3.4)
# ---------------------------------------------------------------------------------------------
def smoothed_logit_coefficients(smoothing: float, vocab_size: int) -> tuple[float, float]:
    """(a, b) of Eq. 18 for x_1^eps = (1 - eps) e_k + (eps/S) 1."""
    if not 0.0 < smoothing < 1.0 or vocab_size < 2:
        raise ValueError("need 0 < smoothing < 1 and vocab_size >= 2")
    floor = smoothing / vocab_size
    return math.log((1.0 - smoothing + floor) / floor), math.log(floor)


def smoothed_target_logits(tokens: Tensor, vocab_size: int, smoothing: float) -> Tensor:
    """l_1 = a e_k + b 1 for token ids ``[...]`` -> ``[..., S]`` (float32)."""
    a, b = smoothed_logit_coefficients(smoothing, vocab_size)
    one_hot = torch.nn.functional.one_hot(tokens.long(), vocab_size).to(torch.float32)
    return a * one_hot + b


def logit_path(l0: Tensor, l1: Tensor, t: float | Tensor) -> Tensor:
    """l_t = (1 - t) l_0 + t l_1  (Eq. 19). A tensor ``t`` must broadcast against ``[..., S]``."""
    return (1.0 - t) * l0 + t * l1


def conditional_logit_velocity(l1: Tensor, lt: Tensor, t: float | Tensor) -> Tensor:
    """l_1 - l_0 written through the state: (l_1 - l_t) / (1 - t), for t < 1."""
    return (l1 - lt) / (1.0 - t)


def posterior_logit_velocity(posterior: Tensor, lt: Tensor, t: float | Tensor, a: float, b: float = 0.0) -> Tensor:
    """v*(l, t) = (a p(. | l_t) + b 1 - l_t) / (1 - t)  (Lemma 2, Eq. 20).

    ``posterior`` is p(w_1 = k | l_t) over the last axis ``[..., S]``. With b = 0 this is the
    velocity the sampler integrates (the constant b 1 does not change softmax(l)).
    """
    return (a * posterior + b - lt) / (1.0 - t)


def logit_euler_step(lt: Tensor, posterior: Tensor, t: float, h: float, a: float) -> Tensor:
    """l_{k+1} = l_k + h (a p - l_k) / (1 - t_k)  (Eq. 21); at t = 1 - h it returns a p exactly."""
    if not 0.0 <= t < 1.0 or not 0.0 < h <= 1.0 - t + 1e-12:
        raise ValueError("need 0 <= t < 1 and 0 < h <= 1 - t")
    if abs(t + h - 1.0) < 1e-12:
        return a * posterior                     # l_K = a p: the t = 1 singularity is never evaluated
    return lt + h * posterior_logit_velocity(posterior, lt, t, a)
