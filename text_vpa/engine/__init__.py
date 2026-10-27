# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Training (PR-JEPA Secs. 3.1, 4.2, 5.1).

    losses.py          L_plan (Eq. 5): VICRegLoss (prediction term, v(Z) Eq. 6, c(Z) Eq. 7),
                       ModeCrossEntropyLoss, PlanLoss; stable rank (Prop. 2), Theorem 1 bound,
                       and the Stage-1 collapse guard StableRankMonitor
    trainer_stage1.py  Stage 1: planner pretraining, EMA target encoder, stable-rank rejection
    trainer_stage2.py  Stage 2: frozen planner, realizer + readout (anti-leakage invariant);
                       Stage 3: joint fine-tuning
    runtime.py         DDP context, bf16 autocast, AdamW groups, warm-up/cosine schedule,
                       atomic checkpoints, metric logging (JSONL, optional Weights & Biases)
"""

from text_vpa.engine.losses import (
    CovarianceDiagnostics,
    LossOutput,
    ModeCrossEntropyLoss,
    PlanLoss,
    RepresentationCollapseError,
    StableRankMonitor,
    StreamingCovariance,
    VICRegLoss,
    calculate_stable_rank,
    covariance_diagnostics,
    covariance_penalty,
    masked_rows,
    sample_covariance,
    stable_rank,
    theorem1_lambda_min_bound,
    variance_hinge,
)

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
