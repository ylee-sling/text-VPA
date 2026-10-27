# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Utilities.

    device.py        backend selection (CUDA / ROCm, MPS, CPU)
    verification.py  the numerical checks of Sec. 4.6 / Appendix B (python -m text_vpa.utils.verification)
    profiler.py      roofline model of Sec. 5.4 / Table 3, measured latency, simulated AR baseline
"""

from text_vpa.utils.device import resolve_device

__all__ = ["resolve_device"]
