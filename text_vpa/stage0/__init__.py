# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Stage 0 of the Text-VPA protocol: a pretrained autoregressive backbone (Gemma 4 E2B) adapted into the few-step
parallel realizer WITHOUT a plan, compared with the same backbone's own autoregressive decoding (hypothesis H2).

    backbone.py     Gemma 4 text decoder as a realizer: prompt prefill by the frozen backbone (one KV cache),
                    canvas-only LoRA, bidirectional canvas with two-sided local windows, mask embedding,
                    zero-initialised (t, h) modulation, the AR shift with a clamped anchor token
    data.py         prompt formats (complete / write), corpus stream, canvases, training states
    sampler.py      K parallel steps with the exact jump kernel on the mask source
    autoregressive.py  the baseline: the released model's own ``generate``
    postprocess.py  one truncation rule for both decoders
    train.py        the trainer (CLI)
    evaluate.py     Pass@k of either decoder in the repository's harness (CLI)
    report.py       the H2 verdict with a paired bootstrap interval (CLI)

Decisions the paper leaves open are recorded in ``docs/stage0.md``.
"""
