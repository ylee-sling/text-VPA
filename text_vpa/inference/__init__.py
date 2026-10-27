# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Inference (PR-JEPA Algorithm 1, Secs. 3.3 and 4.3).

    sampler.py       PRJEPASampler: K + 3 sequential passes, audited at runtime; Prop. 4(d) check
    energy_terms.py  constraint energies (plan-mediated and draft-dependent), lexer, repair pass
    loading.py       rebuild model, tokenizer and sampler from a checkpoint
"""
