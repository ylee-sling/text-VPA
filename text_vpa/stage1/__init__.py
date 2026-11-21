# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Stage 1 of the Text-VPA protocol: a planner on the Stage 0 realizer (design and gates: ``docs/stage1.md``).

Built so far, step 1 of the design (the structure oracle, a diagnostic that is never the method):

    slots.py        line slots of a canvas program (kind, depth, start, length), the fixed codes of positions and
                    layouts, line skeletons and their codebook
    bridge.py       gated cross-attention from every realizer block to a plan; keys and values cached once
    plan.py         ``OraclePlan`` and the label every oracle result carries
    oracle.py       the structure oracle's plan, the normalised reference at evaluation, plan adherence
    data.py         Stage 0's examples with their line slots
    train.py        the oracle trainer: Stage 0's loop and recipe, started from a Stage 0 checkpoint (CLI)
    evaluate.py     Pass@1 with the oracle, the null and a shuffled plan (CLI)
    report.py       the table, the gate after step 1 and the check that the plan reached the realizer (CLI)
    preflight.py    token offsets, line counts, indentation and skeleton coverage, before any training (CLI)

Not built: the planner (f_theta's head, the mode selector on skeletons, g_phi, L_plan), the realizer on
predicted plans, the latent oracle.
"""
