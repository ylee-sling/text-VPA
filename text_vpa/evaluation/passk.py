# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
passk.py -- the unbiased Pass@k estimator (Chen et al. 2021, "Evaluating large language models trained on
code", ref. [34] of PR-JEPA): with n samples of which c pass,

    pass@k = E_tasks[ 1 - C(n - c, k) / C(n, k) ],

computed as 1 - prod_{i = n-c+1}^{n} (1 - k / i) for numerical stability (1 when n - c < k).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["pass_at_k", "summarize_pass_at_k"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k for one task with ``n`` samples and ``c`` correct ones."""
    if not 0 <= c <= n or k < 1:
        raise ValueError(f"need 0 <= c <= n and k >= 1, got n={n}, c={c}, k={k}")
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples per task, got {n}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1, dtype=np.float64)))


def summarize_pass_at_k(counts: Sequence[tuple[int, int]], ks: Sequence[int]) -> dict[str, float]:
    """Mean pass@k over tasks given (n, c) per task, for every k <= min n."""
    if not counts:
        raise ValueError("no task")
    n_min = min(n for n, _ in counts)
    out = {}
    for k in sorted(set(ks)):
        if k <= n_min:
            out[f"pass@{k}"] = float(np.mean([pass_at_k(n, c, k) for n, c in counts]))
    return out
