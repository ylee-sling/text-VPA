# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
kernels.py -- the energy-tilted, clipped few-step jump kernel (PR-JEPA Sec. 3.3, Eqs. 15-17; Prop. 4).

For position i with current token a, posterior P = p^psi_{1|t}(. | w_t, Z) and energies E^i:

    rho~(j) = min( P(j) exp(-E(j)/T), c_max ),   j != a                                   (Eq. 17)
    K~(j)   = pi_{t,h} rho~(j) / max(1, sum_{j'!=a} rho~(j'))                            (Prop. 4)
    K~(a)   = 1 - sum_{j != a} K~(j)

with the exact jump mass pi_{t,h} = (kappa_{t+h} - kappa_t)/(1 - kappa_t) (Eq. 15). Rejected jump mass
becomes probability of staying. Sampling (Algorithm 1, lines 11-12): with probability
pi sum_j rho~(j) / max(1, sum_j rho~(j)) jump to j ~ rho~ / sum rho~, otherwise keep a.

The safety normalisation max(1, .) is inactive whenever P is a probability vector and the multiplier is
at most 1 (Prop. 4(a)); it is needed for guided posteriors (1 + w) P_c - w P_u, whose lower clip at 0 is
applied here, and for the repair-aware multiplier exp(-[E(j) - E(a)]/T), which can exceed one (Prop. 4(c)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["JumpKernel", "kernel_distribution", "repair_aware_multiplier", "sample_jump", "tilted_jump_kernel"]


@dataclass(frozen=True)
class JumpKernel:
    """Kernel of one parallel step for every position."""

    rho: Tensor        # [..., S] rho~ with rho~(a) = 0 at the current token
    mass: Tensor       # [...] sum_{j != a} rho~(j)
    jump_prob: Tensor  # [...] pi * mass / max(1, mass)

    @property
    def stay_prob(self) -> Tensor:
        return 1.0 - self.jump_prob


def repair_aware_multiplier(energies: Tensor, current: Tensor, temperature: float) -> Tensor:
    """exp(-[E(j) - E(a)]/T) (Sec. 3.3), with exp(+inf) for admissible j when E(a) = inf and 0 when both
    are infinite. Shapes: energies ``[..., S]``, current ``[...]`` -> ``[..., S]``."""
    e_cur = energies.gather(-1, current.unsqueeze(-1))
    diff = energies - e_cur
    diff = torch.where(torch.isinf(energies), torch.full_like(diff, float("inf")), diff)
    return torch.exp(-diff / temperature)


def tilted_jump_kernel(
    posterior: Tensor,
    current: Tensor,
    pi: float | Tensor,
    multiplier: Tensor | None = None,
    c_max: float = 1.0,
) -> JumpKernel:
    """Eq. 17 with Prop. 4's safety normalisation.

    Args:
        posterior: ``[..., S]`` P (or a guided combination; negative entries are clipped to 0).
        current: ``[...]`` int64 current tokens a.
        pi: exact jump mass pi_{t,h} (scalar or broadcastable to ``[...]``).
        multiplier: ``[..., S]`` exp(-E/T) (None: no tilt); may contain +inf (repair-aware variant).
        c_max: upper clip, in (0, 1].
    """
    if not 0.0 < c_max <= 1.0:
        raise ValueError("c_max must lie in (0, 1] (Prop. 4(a))")
    p = posterior.clamp_min(0.0)
    if multiplier is not None:
        tilted = p * multiplier
        rho = torch.where(p > 0, tilted, torch.zeros_like(tilted))    # 0 * inf -> 0
        rho = torch.nan_to_num(rho, nan=0.0)
    else:
        rho = p
    rho = rho.clamp_max(c_max)
    rho = rho.scatter(-1, current.unsqueeze(-1), 0.0)                    # j != a
    mass = rho.sum(dim=-1)
    pi_t = torch.as_tensor(pi, dtype=mass.dtype, device=mass.device)
    jump_prob = pi_t * mass / mass.clamp_min(1.0)
    return JumpKernel(rho, mass, jump_prob)


def kernel_distribution(kernel: JumpKernel, current: Tensor) -> Tensor:
    """The full K~ over V, ``[..., S]``: rows sum to one (Prop. 4(a))."""
    scale = torch.where(kernel.mass > 0, kernel.jump_prob / kernel.mass.clamp_min(1e-30), torch.zeros_like(kernel.mass))
    dist = kernel.rho * scale.unsqueeze(-1)
    return dist.scatter_add(-1, current.unsqueeze(-1), kernel.stay_prob.unsqueeze(-1))


def sample_jump(kernel: JumpKernel, current: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Algorithm 1, line 12: jump with probability ``jump_prob`` to j ~ rho~ / sum rho~, else stay.

    Uses the Gumbel-max trick on log rho~ (zero entries are never drawn); ``generator`` must live on the
    tensors' device.
    """
    u = torch.rand(kernel.jump_prob.shape, generator=generator, device=current.device, dtype=torch.float32)
    jump = (u < kernel.jump_prob.float()) & (kernel.mass > 0)
    g = torch.rand(kernel.rho.shape, generator=generator, device=current.device, dtype=torch.float32)
    gumbel = -torch.log(-torch.log(g.clamp_min(torch.finfo(torch.float32).tiny)))
    proposal = (torch.log(kernel.rho.float()) + gumbel).argmax(dim=-1)
    return torch.where(jump, proposal, current)
