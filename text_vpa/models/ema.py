# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
ema.py -- exponential-moving-average update of the target encoder f_theta_bar (Sec. 3.1, Eq. 4).

    theta_bar <- m_k theta_bar + (1 - m_k) theta

Momentum schedules, with k the update count, T_train the total number of updates and
frac = min(k / T_train, 1):

    linear (Eq. 4, default):  m_k = m0 + (m_end - m0) frac
    cosine (BYOL):            m_k = m_end - (m_end - m0) (cos(pi frac) + 1) / 2

With m_end = 1 these are the paper's m_k = m0 + (1 - m0) k / T_train and the cosine form
m_k = 1 - (1 - m0)(cos(pi k / T_train) + 1) / 2. Both start at m0 = 0.996 and reach 1 at k = T_train,
after which the target is frozen.

Parameters are paired by name, so ``online`` (f_theta) and ``target`` (f_theta_bar) must have the
same parameter names, shapes and dtypes; ``TargetEncoder.from_online`` guarantees this. Buffers are
copied rather than averaged, as in the VPA reference implementation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from text_vpa.configs.schema import VICRegConfig

__all__ = ["EMATargetEncoder", "ema_momentum"]


def ema_momentum(step: int, total_steps: int, momentum_start: float = 0.996, momentum_end: float = 1.0,
                 schedule: str = "linear") -> float:
    """m_k of Eq. 4 (``linear``) or its cosine counterpart, clamped to the schedule's end point."""
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    frac = min(max(step, 0), total_steps) / total_steps
    if schedule == "linear":
        return momentum_start + (momentum_end - momentum_start) * frac
    if schedule == "cosine":
        return momentum_end - (momentum_end - momentum_start) * (math.cos(math.pi * frac) + 1.0) / 2.0
    raise ValueError(f"schedule must be 'linear' or 'cosine', got {schedule!r}")


class EMATargetEncoder:
    """EMA manager for theta_bar <- m_k theta_bar + (1 - m_k) theta.

    Args:
        online: f_theta (trained by the optimizer).
        target: f_theta_bar (frozen; same parameter names as ``online``).
        total_steps: T_train, the number of updates over which m_k rises from m0 to m_end.
        momentum_start: m0 (0.996).
        momentum_end: m_end (1.0).
        schedule: ``"linear"`` (Eq. 4) or ``"cosine"``.

    Call ``update()`` once after every optimizer step; it uses m_k for the current update count k
    and then increments k. ``state_dict`` / ``load_state_dict`` persist k for resumption.
    """

    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        total_steps: int,
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
        schedule: str = "linear",
    ) -> None:
        if not 0.0 <= momentum_start <= momentum_end <= 1.0:
            raise ValueError("momentum must satisfy 0 <= m0 <= m_end <= 1")
        ema_momentum(0, total_steps, momentum_start, momentum_end, schedule)  # validates the rest
        if online is target:
            raise ValueError("online and target must be different modules")
        online_params = dict(online.named_parameters())
        target_params = dict(target.named_parameters())
        if online_params.keys() != target_params.keys():
            missing = sorted(online_params.keys() ^ target_params.keys())[:5]
            raise ValueError(f"online and target parameter names differ, e.g. {missing}")
        for name, t in target_params.items():
            o = online_params[name]
            if t.shape != o.shape or t.dtype != o.dtype or t.device != o.device:
                raise ValueError(f"parameter {name!r} differs in shape, dtype or device between online and target")
            if t.requires_grad:
                raise ValueError(f"target parameter {name!r} requires grad; f_theta_bar must be frozen")
        online_buffers = dict(online.named_buffers())
        target_buffers = dict(target.named_buffers())
        if online_buffers.keys() != target_buffers.keys():
            raise ValueError("online and target buffer names differ")
        for name, t in target_buffers.items():
            o = online_buffers[name]
            if t.shape != o.shape or t.dtype != o.dtype or t.device != o.device:
                # copy_ would broadcast a smaller online buffer silently.
                raise ValueError(f"buffer {name!r} differs in shape, dtype or device between online and target")
        names = sorted(target_params)
        self._params: list[tuple[torch.Tensor, torch.Tensor]] = [(target_params[n], online_params[n]) for n in names]
        self._buffers: list[tuple[torch.Tensor, torch.Tensor]] = [
            (target_buffers[n], online_buffers[n]) for n in sorted(target_buffers)
        ]
        self.total_steps: int = total_steps
        self.momentum_start: float = momentum_start
        self.momentum_end: float = momentum_end
        self.schedule: str = schedule
        self.step: int = 0

    @classmethod
    def from_config(cls, online: nn.Module, target: nn.Module, cfg: VICRegConfig, total_steps: int) -> EMATargetEncoder:
        """Use ``VICRegConfig.ema_*``; ``total_steps`` is Stage 1's ``max_steps``."""
        return cls(online, target, total_steps, cfg.ema_momentum_start, cfg.ema_momentum_end, cfg.ema_schedule)

    def momentum(self, step: int | None = None) -> float:
        """m_k for update count ``step`` (default: the next update)."""
        k = self.step if step is None else step
        return ema_momentum(k, self.total_steps, self.momentum_start, self.momentum_end, self.schedule)

    @torch.no_grad()
    def update(self) -> float:
        """Apply theta_bar <- m_k theta_bar + (1 - m_k) theta, increment k, and return the m_k used."""
        m = self.momentum()
        if m < 1.0:
            for target, online in self._params:
                target.lerp_(online, 1.0 - m)          # theta_bar + (1 - m)(theta - theta_bar)
        for target, online in self._buffers:
            target.copy_(online)
        self.step += 1
        return m

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "momentum_start": self.momentum_start,
            "momentum_end": self.momentum_end,
            "schedule": self.schedule,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("total_steps", "momentum_start", "momentum_end", "schedule"):
            if state[key] != getattr(self, key):
                raise ValueError(f"EMA {key} mismatch: checkpoint {state[key]!r} vs current {getattr(self, key)!r}")
        self.step = int(state["step"])
