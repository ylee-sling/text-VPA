# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Shared test configuration.

Rules (CLAUDE.md): references are computed independently from the paper's formulas, in float64 on
the CPU, never with the function under test; seeds are fixed; float32 module outputs are compared
with rtol=1e-5, atol=1e-6 unless a stated numerical reason requires more.
"""

from __future__ import annotations

import pytest
import torch

RTOL, ATOL = 1e-5, 1e-6


def accelerator() -> torch.device | None:
    """The first available accelerator (CUDA/ROCm, then MPS), or None."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)


@pytest.fixture
def tiny_config():
    from text_vpa.configs import load_config

    return load_config(["planner=tiny", "data=tiny", "realizer=tiny"])
