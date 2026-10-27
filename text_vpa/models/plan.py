# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
plan.py -- ``PredictedPlan``, the only conditioning the realizer accepts (PR-JEPA Sec. 5.1, Stage 2).

The realizer is trained and run on the plan the planner *predicts from the context*,
Z_target = g_phi(s_x, u, Delta), never on the target encoder's view f_theta_bar(y): conditioning on
the target view leaks the answer into the plan and opens a train-test gap. ``PredictedPlan`` is
produced only by ``TextVPAPlanner.plan``; the realizer's ``prepare`` rejects every other type --
in particular ``TargetOutput``, the target encoder's output -- so the leak cannot happen by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

__all__ = ["PredictedPlan", "require_predicted_plan"]


@dataclass(frozen=True)
class PredictedPlan:
    """Outputs of planner passes 1-3 for one batch.

    Attributes:
        s_x: ``[B, M_c, d]`` context latents (pass 1).
        context_mask: ``[B, M_c]`` bool, True at real context tokens.
        mode: ``[B]`` int64 selected design modes u (pass 2).
        mode_input: what g_phi and the bridge consumed: ``[B]`` int64, or ``[B, |U|]`` for a
            straight-through Gumbel one-hot.
        mode_logits: ``[B, |U|]`` selector logits, or None if the selector was not run.
        z_target: ``[B, M, d]`` the plan, ensemble mean of g_phi's heads (pass 3, Eq. 8).
        sigma_bar: ``[B]`` ensemble spread (Eq. 8).
        head_predictions: ``[n_ens, B, M, d]``.
    """

    s_x: Tensor
    context_mask: Tensor
    mode: Tensor
    mode_input: Tensor
    mode_logits: Tensor | None
    z_target: Tensor
    sigma_bar: Tensor
    head_predictions: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.z_target.shape[0])

    def detach(self) -> PredictedPlan:
        return PredictedPlan(
            self.s_x.detach(), self.context_mask, self.mode, self.mode_input.detach(),
            None if self.mode_logits is None else self.mode_logits.detach(),
            self.z_target.detach(), self.sigma_bar.detach(), self.head_predictions.detach(),
        )


def require_predicted_plan(plan: object) -> PredictedPlan:
    """Anti-leakage guard: accept only planner outputs as realizer conditioning."""
    if not isinstance(plan, PredictedPlan):
        raise TypeError(
            f"the realizer conditions only on a PredictedPlan from TextVPAPlanner.plan (Sec. 5.1), got "
            f"{type(plan).__name__}; target-encoder latents f_theta_bar(y) must never reach the realizer"
        )
    return plan
