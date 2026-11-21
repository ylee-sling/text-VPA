# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
plan.py -- ``OraclePlan``: a plan read from the reference program. A diagnostic's conditioning, never the method's.

The anti-leakage rule (PR-JEPA Sec. 5.1, CLAUDE.md rule 4) lets the realizer of the method condition on a
``PredictedPlan`` only. The oracles of docs/stage1.md break that rule on purpose: they ask what a realizer could
do if it were told the reference's structure. So that an oracle can never be mistaken for the method,

* its plan has its own type, which ``models.plan.require_predicted_plan`` (the guard of the Stage 2 trainer)
  rejects like any other;
* the oracle trainer and evaluation accept nothing else (``require_oracle_plan``);
* every checkpoint, summary and report of an oracle run carries ``ORACLE_LABEL``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

__all__ = ["ORACLE_LABEL", "OraclePlan", "require_oracle_plan"]

ORACLE_LABEL = "oracle: not the method"


@dataclass(frozen=True)
class OraclePlan:
    """Slot latents computed from the reference program.

    Attributes:
        z: ``[B, M, d]`` one latent per line slot.
        mask: ``[B, M]`` bool, True at the reference's real slots.
        source: what of the reference went in: ``"structure"`` (kinds, depths, layout; no backbone state).
    """

    z: Tensor
    mask: Tensor
    source: str

    @property
    def batch_size(self) -> int:
        return int(self.z.shape[0])


def require_oracle_plan(plan: object) -> OraclePlan:
    """The oracle diagnostics run on an ``OraclePlan`` and on nothing else."""
    if not isinstance(plan, OraclePlan):
        raise TypeError(f"an oracle diagnostic conditions on an OraclePlan, got {type(plan).__name__}; the "
                        "method's plans go through the Stage 2 trainer")
    return plan
