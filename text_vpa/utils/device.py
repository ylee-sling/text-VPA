# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Backend selection for CUDA, ROCm, Apple MPS and CPU.

ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` API, so ``"cuda"`` covers both.
MPS has no float64: keep every parameter, buffer and forward-path tensor in float32 or lower;
float64 appears only in CPU-side diagnostics and test references.
"""

from __future__ import annotations

import torch

__all__ = ["resolve_device"]


def resolve_device(name: str = "auto") -> torch.device:
    """Return the requested device; ``"auto"`` picks cuda (incl. ROCm) > mps > cpu.

    Raises:
        RuntimeError: an explicitly requested backend is not available.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but torch.cuda.is_available() is False")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested but torch.backends.mps.is_available() is False")
    return device
