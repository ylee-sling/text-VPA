# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
losses.py -- the planner objective L_plan and the anti-collapse diagnostics (PR-JEPA Secs. 3.1, 4.2, 5.1).

    L_plan = lambda_invar (1/(BM)) sum_b || s_hat_y^(b) - sg(s_y^(b)) ||_F^2
           + lambda_v [ v(Z_x) + v(Z_hat) ] + lambda_c [ c(Z_x) + c(Z_hat) ] + lambda_u L_CE(pi_phi^h)   (Eq. 5)

    v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(z^j) + eps)),
           Var(z^j) = (1/(n-1)) sum_i (z_i^j - zbar^j)^2                                     (Eq. 6)
    C(Z) = (1/(n-1)) sum_i (z_i - zbar)(z_i - zbar)^T,   c(Z) = (1/d) sum_{j != k} C(Z)_jk^2  (Eq. 7)

Z_x and Z_hat are the online context latents s_x and the predicted plan Z_target of the batch,
stacked into n rows of width d (n = BM for full plans; with masks, the real rows only). The
regularisers act only on branches that receive gradients; on the stop-gradient target they would
have no effect (Sec. 3.1). The prediction term is "normalised" by the number of slots BM: it sums
the squared error over the d features of a slot and averages over slots, exactly as in Eq. 5.

Diagnostics (Sec. 4.2, Sec. 5.1):
    srank(C) = (tr C)^2 / ||C||_F^2 = (sum_j C_jj)^2 / (sum_j C_jj^2 + d c(Z)) <= rank(C)   (Prop. 2)
    Theorem 1: if v(Z) = 0 and c(Z) < (gamma^2 - eps)^2 / (d(d - 1)), then
               lambda_min(C) >= gamma^2 - eps - sqrt(d(d - 1) c(Z)) > 0.
    Prop. 3:   for an active hinge, ||grad_{z^j}(gamma - sqrt(Var(z^j) + eps))|| = sqrt(Var / ((n-1)(Var + eps))).
Stage 1 aborts when srank(C) < d/2 (``StableRankMonitor``, ``calculate_stable_rank(min_fraction=0.5)``).

Numerics: every function is dtype-generic and runs on any device in float32; eigenvalues for logging
are computed in float64 on the CPU (MPS has no float64) from a covariance accumulated there.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from text_vpa.configs.schema import VICRegConfig

__all__ = [
    "CovarianceDiagnostics",
    "LossOutput",
    "ModeCrossEntropyLoss",
    "PlanLoss",
    "RepresentationCollapseError",
    "StableRankMonitor",
    "StreamingCovariance",
    "VICRegLoss",
    "calculate_stable_rank",
    "covariance_diagnostics",
    "covariance_penalty",
    "masked_rows",
    "sample_covariance",
    "stable_rank",
    "theorem1_lambda_min_bound",
    "variance_hinge",
]


class RepresentationCollapseError(RuntimeError):
    """Raised when the stable rank of a latent covariance falls below the abort threshold (Sec. 5.1)."""


# ---------------------------------------------------------------------------------------------
# Eqs. 6 and 7
# ---------------------------------------------------------------------------------------------
def _check_rows(z: Tensor) -> tuple[int, int]:
    if z.dim() < 2:
        raise ValueError(f"expected rows of shape [..., n, d], got {tuple(z.shape)}")
    n, d = z.shape[-2], z.shape[-1]
    if n < 2:
        raise ValueError("the variance and covariance estimators need n >= 2 rows")
    return n, d


def sample_covariance(z: Tensor) -> Tensor:
    """C(Z) = (1/(n-1)) sum_i (z_i - zbar)(z_i - zbar)^T  (Eq. 7).  ``[..., n, d]`` -> ``[..., d, d]``."""
    n, _ = _check_rows(z)
    centred = z - z.mean(dim=-2, keepdim=True)
    return centred.transpose(-1, -2) @ centred / (n - 1)


def variance_hinge(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(z^j) + eps)), Bessel-corrected Var  (Eq. 6).

    ``[..., n, d]`` -> ``[...]``.
    """
    _check_rows(z)
    std = torch.sqrt(z.var(dim=-2, correction=1) + eps)                        # [..., d]
    return F.relu(gamma - std).mean(dim=-1)


def covariance_penalty(z: Tensor) -> Tensor:
    """c(Z) = (1/d) sum_{j != k} C(Z)_jk^2  (Eq. 7).  ``[..., n, d]`` -> ``[...]``."""
    cov = sample_covariance(z)
    d = cov.shape[-1]
    off_diagonal = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    return off_diagonal.pow(2).sum(dim=(-2, -1)) / d


def stable_rank(cov: Tensor) -> Tensor:
    """srank(C) = (tr C)^2 / ||C||_F^2 for covariances ``[..., d, d]`` -> ``[...]`` (Prop. 2).

    A zero covariance (a batch collapsed to a point) has srank 0 by convention.
    """
    if cov.dim() < 2 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError(f"expected square matrices [..., d, d], got {tuple(cov.shape)}")
    trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
    frobenius_sq = cov.pow(2).sum(dim=(-2, -1))
    safe = torch.where(frobenius_sq > 0, frobenius_sq, torch.ones_like(frobenius_sq))
    return torch.where(frobenius_sq > 0, trace.pow(2) / safe, torch.zeros_like(trace))


def calculate_stable_rank(z: Tensor, min_fraction: float | None = None) -> Tensor:
    """srank(C(Z)) of the rows ``z [..., n, d]``; optionally assert the anti-collapse threshold.

    Args:
        z: latent rows (e.g. ``masked_rows(plan, slot_mask)``).
        min_fraction: if given, raise ``RepresentationCollapseError`` when srank < min_fraction * d
            (Sec. 5.1 uses 1/2). Because rank(C) <= n - 1, the check needs n - 1 >= min_fraction * d
            rows to be meaningful; fewer rows raise ``ValueError`` instead of a false alarm.

    Returns:
        ``[...]`` stable ranks, detached.
    """
    with torch.no_grad():
        n, d = _check_rows(z)
        srank = stable_rank(sample_covariance(z.float() if z.dtype in (torch.float16, torch.bfloat16) else z))
        if min_fraction is not None:
            if not 0.0 < min_fraction <= 1.0:
                raise ValueError("min_fraction must be in (0, 1]")
            threshold = min_fraction * d
            if n - 1 < threshold:
                raise ValueError(f"{n} rows cannot certify srank >= {threshold:g}: rank(C) <= n - 1; use more rows "
                                 "(e.g. StableRankMonitor, which accumulates over steps)")
            low = srank < threshold
            if bool(low.any()):
                raise RepresentationCollapseError(
                    f"stable rank {float(srank.min()):.3f} < {threshold:g} = {min_fraction:g} d (d = {d}): "
                    "the representation is collapsing (Sec. 5.1)"
                )
    return srank


def theorem1_lambda_min_bound(cov: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """gamma^2 - eps - sqrt(d(d - 1) c(Z)), the Theorem 1 lower bound on lambda_min(C).

    It is a valid bound only when v(Z) = 0 (every Var(z^j) + eps >= gamma^2); it certifies a positive
    definite covariance when it is also positive. ``[..., d, d]`` -> ``[...]``.
    """
    d = cov.shape[-1]
    off_diagonal = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    c = off_diagonal.pow(2).sum(dim=(-2, -1)) / d
    return gamma ** 2 - eps - torch.sqrt(d * (d - 1) * c)


def masked_rows(x: Tensor, mask: Tensor | None) -> Tensor:
    """Stack the real rows of ``x [B, L, d]`` into ``[n, d]`` (n = number of True entries of ``mask``)."""
    if x.dim() != 3:
        raise ValueError(f"expected [B, L, d], got {tuple(x.shape)}")
    if mask is None:
        return x.reshape(-1, x.shape[-1])
    if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
        raise ValueError(f"expected a bool mask of shape {tuple(x.shape[:2])}, got {tuple(mask.shape)} {mask.dtype}")
    return x[mask]


# ---------------------------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LossOutput:
    """A scalar loss with grad, plus detached terms for logging."""

    total: Tensor
    terms: dict[str, Tensor]


class VICRegLoss(nn.Module):
    """The JEPA part of Eq. 5: lambda_invar * prediction + lambda_v [v(Z_x) + v(Z_hat)] + lambda_c [c(Z_x) + c(Z_hat)].

    Args:
        lambda_invar: weight of the prediction (invariance) term (Eq. 5 uses 1; see VICRegConfig).
        lambda_v: weight of the variance hinge; > 0 (Theorem 1).
        lambda_c: weight of the covariance penalty; >= 0.
        gamma: target standard deviation, > 0.
        eps: 0 < eps < gamma^2, and well below it (Prop. 3).
        head_reduction: the prediction term is applied to every ensemble head and then averaged
            (``"mean"``, which reduces to Eq. 5 for one head) or summed (``"sum"``).
    """

    def __init__(self, lambda_invar: float = 25.0, lambda_v: float = 25.0, lambda_c: float = 1.0, gamma: float = 1.0,
                 eps: float = 1e-4, head_reduction: str = "mean", normalization: str = "per_dim") -> None:
        super().__init__()
        if not lambda_invar > 0.0:
            raise ValueError("lambda_invar must be positive")
        if not lambda_v > 0.0:
            raise ValueError("lambda_v must be positive (Theorem 1 needs the variance hinge)")
        if lambda_c < 0.0:
            raise ValueError("lambda_c must be non-negative")
        if not gamma > 0.0:
            raise ValueError("gamma must be positive")
        if not 0.0 < eps < gamma ** 2:
            raise ValueError("eps must satisfy 0 < eps < gamma^2")
        if head_reduction not in ("mean", "sum"):
            raise ValueError("head_reduction must be 'mean' or 'sum'")
        if normalization not in ("per_dim", "frobenius"):
            raise ValueError("normalization must be 'per_dim' or 'frobenius'")
        self.normalization = normalization
        self.lambda_invar = float(lambda_invar)
        self.lambda_v = float(lambda_v)
        self.lambda_c = float(lambda_c)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.head_reduction = head_reduction

    @classmethod
    def from_config(cls, cfg: VICRegConfig) -> VICRegLoss:
        return cls(cfg.lambda_invar, cfg.lambda_v, cfg.lambda_c, cfg.gamma, cfg.eps, cfg.head_reduction,
                   cfg.invariance_normalization)

    def invariance(self, predictions: Tensor, target: Tensor, slot_mask: Tensor | None = None) -> Tensor:
        """Prediction term, per head, then reduced over heads.

        ``frobenius``: (1/(BM)) sum_b ||s_hat_y^(b) - sg(s_y^(b))||_F^2, Eq. 5 verbatim (squared error summed
        over the d features of a slot). ``per_dim`` (default): the same divided by d, i.e. the mean squared
        error per element ("normalised MSE"), which is VICReg's invariance term; with it the published
        25 : 25 : 1 weights keep VICReg's balance at every d. (Eq. 5's unit weight on the Frobenius form
        equals lambda_invar = d with ``per_dim``.)

        Args:
            predictions: ``[n_ens, B, M, d]`` per-head plans (or ``[B, M, d]`` for one predictor).
            target: ``[B, M, d]`` target-encoder latents s_y; detached here (sg).
            slot_mask: ``[B, M]`` bool; BM becomes the number of real slots.
        """
        heads = predictions.unsqueeze(0) if predictions.dim() == 3 else predictions
        if heads.dim() != 4 or heads.shape[1:] != target.shape:
            raise ValueError(f"shape mismatch: predictions {tuple(predictions.shape)} vs target {tuple(target.shape)}")
        sq = (heads - target.detach().unsqueeze(0)).pow(2).sum(dim=-1)              # [n_ens, B, M]
        if self.normalization == "per_dim":
            sq = sq / target.shape[-1]
        if slot_mask is None:
            per_head = sq.mean(dim=(1, 2))
        else:
            if slot_mask.shape != target.shape[:2] or slot_mask.dtype != torch.bool:
                raise ValueError("slot_mask must be a bool tensor of shape [B, M]")
            if not bool(slot_mask.any()):
                raise ValueError("slot_mask selects no slot: the prediction term of Eq. 5 is undefined")
            # masked_fill, not a product: padded slots may hold non-finite values (0 * inf = NaN).
            per_head = sq.masked_fill(~slot_mask, 0.0).sum(dim=(1, 2)) / slot_mask.sum()
        return per_head.mean() if self.head_reduction == "mean" else per_head.sum()

    def variance(self, rows: Tensor) -> Tensor:
        """v(Z) of Eq. 6 on rows ``[n, d]``."""
        return variance_hinge(rows, self.gamma, self.eps)

    def covariance(self, rows: Tensor) -> Tensor:
        """c(Z) of Eq. 7 on rows ``[n, d]``."""
        return covariance_penalty(rows)

    def forward(
        self,
        *,
        predictions: Tensor,
        target: Tensor,
        context: Tensor,
        plan: Tensor,
        context_mask: Tensor | None = None,
        slot_mask: Tensor | None = None,
    ) -> LossOutput:
        """Args:
            predictions: ``[n_ens, B, M, d]`` per-head plans s_hat^(e) (or ``[B, M, d]``).
            target: ``[B, M, d]`` s_y = f_theta_bar(y) (stop-gradient applied here).
            context: ``[B, M_c, d]`` online context latents s_x (Z_x).
            plan: ``[B, M, d]`` Z_target, the ensemble mean (Z_hat).
            context_mask, slot_mask: bool masks of the real context tokens and slots.
        """
        if plan.shape != target.shape:
            raise ValueError(f"plan {tuple(plan.shape)} and target {tuple(target.shape)} must have the same shape")
        inv = self.invariance(predictions, target, slot_mask)
        z_x = masked_rows(context, context_mask)
        z_hat = masked_rows(plan, slot_mask)
        v_x, v_hat = self.variance(z_x), self.variance(z_hat)
        c_x, c_hat = self.covariance(z_x), self.covariance(z_hat)
        total = self.lambda_invar * inv + self.lambda_v * (v_x + v_hat) + self.lambda_c * (c_x + c_hat)
        terms = {"invariance": inv, "variance_context": v_x, "variance_plan": v_hat,
                 "covariance_context": c_x, "covariance_plan": c_hat}
        return LossOutput(total, {k: t.detach() for k, t in terms.items()})


class ModeCrossEntropyLoss(nn.Module):
    """L_CE(pi_phi^h): cross-entropy of the selector's logits against reference modes (Eq. 5, Sec. 5.1).

    Args:
        label_smoothing: passed to ``F.cross_entropy`` (default 0).
        ignore_index: target value of samples without a reference mode (excluded from the mean).

    Shapes: ``logits [B, |U|]``, ``targets [B]`` int64 -> scalar.
    """

    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = -100) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Mean cross-entropy over the labelled samples; 0 (with a graph) when no sample is labelled."""
        if logits.dim() != 2 or targets.shape != logits.shape[:1] or targets.dtype.is_floating_point \
                or targets.dtype == torch.bool:
            raise ValueError(f"expected logits [B, |U|] and integer targets [B], got {tuple(logits.shape)} and "
                             f"{tuple(targets.shape)} {targets.dtype}")
        if not bool((targets != self.ignore_index).any()):
            # F.cross_entropy returns 0/0 = NaN here; a batch without reference modes contributes nothing.
            return logits.sum() * 0.0
        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)


class PlanLoss(nn.Module):
    """L_plan of Eq. 5: ``VICRegLoss`` + lambda_u * ``ModeCrossEntropyLoss``."""

    def __init__(self, vicreg: VICRegLoss, lambda_u: float = 1.0, mode_loss: ModeCrossEntropyLoss | None = None) -> None:
        super().__init__()
        if lambda_u < 0.0:
            raise ValueError("lambda_u must be non-negative")
        self.vicreg = vicreg
        self.mode_loss = mode_loss if mode_loss is not None else ModeCrossEntropyLoss()
        self.lambda_u = float(lambda_u)

    @classmethod
    def from_config(cls, cfg: VICRegConfig) -> PlanLoss:
        return cls(VICRegLoss.from_config(cfg), cfg.lambda_u)

    def forward(
        self,
        *,
        predictions: Tensor,
        target: Tensor,
        context: Tensor,
        plan: Tensor,
        mode_logits: Tensor,
        mode_targets: Tensor,
        context_mask: Tensor | None = None,
        slot_mask: Tensor | None = None,
    ) -> LossOutput:
        """Eq. 5. Arguments as in ``VICRegLoss.forward`` plus the selector's ``mode_logits [B, |U|]``
        and the reference ``mode_targets [B]``."""
        jepa = self.vicreg(predictions=predictions, target=target, context=context, plan=plan,
                           context_mask=context_mask, slot_mask=slot_mask)
        ce = self.mode_loss(mode_logits, mode_targets)
        total = jepa.total + self.lambda_u * ce
        return LossOutput(total, {**jepa.terms, "mode_ce": ce.detach(), "total": total.detach()})


# ---------------------------------------------------------------------------------------------
# Collapse diagnostics and the Stage-1 guard
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CovarianceDiagnostics:
    """Summary of a latent covariance C (all values from the same C)."""

    num_rows: int
    dim: int
    variance_hinge: float           # v(Z), Eq. 6
    covariance_penalty: float       # c(Z), Eq. 7
    stable_rank: float              # (tr C)^2 / ||C||_F^2
    lambda_min: float | None        # smallest eigenvalue (if eigenvalues were computed)
    certified_lambda_min: float | None  # Theorem 1 bound, only when its premise holds
    eigenvalues: Tensor | None      # ascending, float64, CPU

    @property
    def srank_fraction(self) -> float:
        return self.stable_rank / self.dim

    def as_log_dict(self, prefix: str = "") -> dict[str, float]:
        out = {f"{prefix}variance_hinge": self.variance_hinge, f"{prefix}covariance_penalty": self.covariance_penalty,
               f"{prefix}stable_rank": self.stable_rank, f"{prefix}srank_fraction": self.srank_fraction}
        if self.lambda_min is not None:
            out[f"{prefix}lambda_min"] = self.lambda_min
        if self.certified_lambda_min is not None:
            out[f"{prefix}certified_lambda_min"] = self.certified_lambda_min
        return out


def covariance_diagnostics(cov: Tensor, num_rows: int, gamma: float = 1.0, eps: float = 1e-4,
                           compute_eigenvalues: bool = True) -> CovarianceDiagnostics:
    """Diagnostics of a covariance ``[d, d]`` estimated from ``num_rows`` rows (computed in float64 on CPU).

    v(Z) and c(Z) depend on Z only through C, so they are evaluated from C directly.
    """
    c64 = cov.detach().to(device="cpu", dtype=torch.float64)
    d = c64.shape[-1]
    diag = torch.diagonal(c64)
    v = float(F.relu(gamma - torch.sqrt(diag + eps)).mean())
    off = c64 - torch.diag(diag)
    c = float(off.pow(2).sum() / d)
    srank = float(stable_rank(c64))
    eigenvalues = torch.linalg.eigvalsh(c64) if compute_eigenvalues else None
    bound = gamma ** 2 - eps - math.sqrt(d * (d - 1) * c)
    certified = bound if (v == 0.0 and bound > 0.0) else None
    return CovarianceDiagnostics(
        num_rows=num_rows, dim=d, variance_hinge=v, covariance_penalty=c, stable_rank=srank,
        lambda_min=float(eigenvalues[0]) if eigenvalues is not None else None,
        certified_lambda_min=certified, eigenvalues=eigenvalues,
    )


class StreamingCovariance:
    """Running mean and covariance over many batches of rows (Chan et al. pairwise update, float64 on CPU).

    Each batch's mean and centred cross-product are computed on the rows' device, then merged on
    the CPU in float64, so the estimate is that of all rows stacked, without storing them.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = torch.zeros(self.dim, dtype=torch.float64)
        self.m2 = torch.zeros(self.dim, self.dim, dtype=torch.float64)

    @torch.no_grad()
    def update(self, rows: Tensor) -> None:
        """Add rows ``[n, d]``."""
        if rows.dim() != 2 or rows.shape[1] != self.dim:
            raise ValueError(f"expected rows of shape [n, {self.dim}], got {tuple(rows.shape)}")
        n = rows.shape[0]
        if n == 0:
            return
        work = rows.float() if rows.dtype in (torch.float16, torch.bfloat16) else rows
        batch_mean = work.mean(dim=0)
        centred = work - batch_mean
        batch_m2 = (centred.transpose(0, 1) @ centred).to(device="cpu", dtype=torch.float64)
        batch_mean = batch_mean.to(device="cpu", dtype=torch.float64)
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n / total)
        self.m2 = self.m2 + batch_m2 + torch.outer(delta, delta) * (self.count * n / total)
        self.count = total

    def covariance(self) -> Tensor:
        """Unbiased covariance of all rows seen, ``[d, d]`` float64 CPU."""
        if self.count < 2:
            raise ValueError("need at least 2 rows")
        return self.m2 / (self.count - 1)


class StableRankMonitor:
    """Stage-1 collapse guard: abort when srank(C) < abort_fraction * d for any watched stream (Sec. 5.1).

    Rows are accumulated over a window (the paper checks once per epoch) so that n >> d; a single
    batch with n <= d rows has rank(C) <= n - 1 and cannot certify srank >= d/2.

    Usage::

        monitor = StableRankMonitor(dim=d)                    # streams "context" and "plan"
        monitor.update("context", masked_rows(s_x, context_mask))
        monitor.update("plan", masked_rows(z_target, slot_mask))
        if step % window == 0:
            report = monitor.check()                          # raises RepresentationCollapseError
    """

    def __init__(self, dim: int, abort_fraction: float = 0.5, gamma: float = 1.0, eps: float = 1e-4,
                 streams: Iterable[str] = ("context", "plan")) -> None:
        if not 0.0 < abort_fraction <= 1.0:
            raise ValueError("abort_fraction must be in (0, 1]")
        self.dim = dim
        self.abort_fraction = abort_fraction
        self.gamma = gamma
        self.eps = eps
        self._streams = {name: StreamingCovariance(dim) for name in streams}

    @classmethod
    def from_config(cls, dim: int, cfg: VICRegConfig) -> StableRankMonitor:
        return cls(dim, cfg.srank_abort_fraction, cfg.gamma, cfg.eps)

    @property
    def threshold(self) -> float:
        return self.abort_fraction * self.dim

    def update(self, stream: str, rows: Tensor) -> None:
        self._streams[stream].update(rows)

    def reset(self) -> None:
        """Start a new window."""
        for acc in self._streams.values():
            acc.reset()

    def below_threshold(self, report: dict[str, CovarianceDiagnostics]) -> dict[str, float]:
        """Streams of ``report`` whose stable rank is below ``abort_fraction * d``."""
        return {name: d.stable_rank for name, d in report.items() if d.stable_rank < self.threshold}

    @property
    def min_rows(self) -> int:
        """Rows accumulated in the emptiest stream of the current window."""
        return min(acc.count for acc in self._streams.values())

    def summarize(self, compute_eigenvalues: bool = True) -> dict[str, CovarianceDiagnostics]:
        out = {}
        for name, acc in self._streams.items():
            if acc.count <= self.dim:
                raise ValueError(f"stream {name!r} has {acc.count} rows; need more than d = {self.dim} "
                                 "(rank(C) <= n - 1)")
            out[name] = covariance_diagnostics(acc.covariance(), acc.count, self.gamma, self.eps, compute_eigenvalues)
        return out

    def check(self, compute_eigenvalues: bool = True, reset: bool = True) -> dict[str, CovarianceDiagnostics]:
        """Summarise every stream, then raise ``RepresentationCollapseError`` if any srank < threshold."""
        report = self.summarize(compute_eigenvalues)
        if reset:
            for acc in self._streams.values():
                acc.reset()
        failed = {name: diag.stable_rank for name, diag in report.items() if diag.stable_rank < self.threshold}
        if failed:
            details = ", ".join(f"{name}: {value:.2f}" for name, value in failed.items())
            raise RepresentationCollapseError(
                f"stable rank below {self.threshold:g} = {self.abort_fraction:g} d (d = {self.dim}) -- {details}"
            )
        return report
