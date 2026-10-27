# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Parallel discrete realizers F_psi (PR-JEPA Secs. 3.3-3.4).

    backbone.py  bidirectional AdaLN transformer reading the bridge's cached K/V (Sec. 3.2)
    fs_dfm.py    conditional FS-DFM realizer: token states, Eq. 12 loss (primary)
    logit_kl.py  Logit-KL realizer: logit states, Euler steps along KL geodesics (Eqs. 18-21)
    kernels.py   energy-tilted, clipped exact jump kernel (Eqs. 15-17, Prop. 4)
    paths.py     kappa schedules, exact jump mass, mixture-path training states, Logit-KL path
"""

from text_vpa.models.realizers.paths import (
    KappaSchedule,
    conditional_logit_velocity,
    logit_euler_step,
    logit_path,
    posterior_logit_velocity,
    smoothed_logit_coefficients,
    smoothed_target_logits,
    uniform_time_grid,
)

__all__ = [
    "KappaSchedule",
    "conditional_logit_velocity",
    "logit_euler_step",
    "logit_path",
    "posterior_logit_velocity",
    "smoothed_logit_coefficients",
    "smoothed_target_logits",
    "uniform_time_grid",
]
