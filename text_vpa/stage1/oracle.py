# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
oracle.py -- the structure oracle of docs/stage1.md, step 1. A diagnostic: it is never the method.

The realizer is told the reference program's true line structure and nothing else of it:

    Z_m = E_kind[kind_m] + E_depth[depth_m] + W code(start_m, length_m)        m = 1 .. M (line slots)

with ``slots.layout_code`` the fixed code of the slot's first canvas token and its number of tokens. No token of
the reference and no backbone state of it enters: two programs with the same line structure get the same plan
(tested). E_kind, E_depth and W are trained together with the bridge. The plan has no mode row: a line skeleton
is a function of the kinds and depths that the slots already hold.

In training the reference is the example's own canvas. At evaluation it is the benchmark's canonical solution,
*normalised* first (``normalize_reference``: ``ast.unparse`` of its syntax tree). MBPP's solutions are indented
with tabs or with one, two or three spaces in two thirds of the problems (60 of the 90 validation problems) and
often carry trailing blanks; read as they are, their depths (indentation // 4) and token layout would describe a
formatting the realizer was not trained to write, and the oracle would understate what structure can give. The
normalised program is the same program (it must still pass its tests; the preflight checks that).

Three plans can be evaluated with one trained model, all of them labelled as oracle rows:

    oracle     the structure of the problem's own reference
    null       the bridge's null token (what the same weights do without a plan)
    shuffled   the structure of another problem's reference (is it this structure, or any structure?)

``kind_match`` measures whether a sample followed its plan: the share of lines whose kind is the planned one. It
is what tells "the structure did not help" from "the structure did not reach the realizer" (docs/stage1.md,
"Reading the oracle").
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from text_vpa.evaluation.tasks import EvalTask
from text_vpa.stage0.backbone import Stage0Realizer
from text_vpa.stage0.evaluate import realizer_samples
from text_vpa.stage0.sampler import SamplerConfig
from text_vpa.stage1.bridge import PlanBridge, PlanCache
from text_vpa.stage1.plan import ORACLE_LABEL, OraclePlan, require_oracle_plan
from text_vpa.stage1.slots import (
    KINDS,
    MAX_DEPTH,
    LineSlots,
    SlotBatch,
    collate_slots,
    encode_with_starts,
    layout_code,
    line_slots,
    program_lines,
)

__all__ = ["PLAN_VARIANTS", "StructureOracle", "TaskPlans", "kind_match", "normalize_reference", "oracle_samples"]

PLAN_VARIANTS: tuple[str, ...] = ("oracle", "null", "shuffled")


class StructureOracle(nn.Module):
    """Line slots of the reference -> ``OraclePlan`` (see the module docstring).

    Shapes: ``SlotBatch`` of ``[B, M]`` tensors -> z ``[B, M, d]``, mask ``[B, M]``.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kind = nn.Embedding(len(KINDS), latent_dim)             # E_kind
        self.depth = nn.Embedding(MAX_DEPTH + 1, latent_dim)         # E_depth
        self.layout = nn.Linear(latent_dim, latent_dim)              # W on the fixed code of (start, length)

    def forward(self, slots: SlotBatch) -> OraclePlan:
        code = layout_code(slots.start, slots.length, self.latent_dim)                 # [B, M, d], no parameters
        z = self.kind(slots.kind) + self.depth(slots.depth) + self.layout(code).float()
        return OraclePlan(z * slots.mask[..., None], slots.mask, "structure")


# ---------------------------------------------------------------------------------------------
# The reference at evaluation
# ---------------------------------------------------------------------------------------------
def normalize_reference(code: str) -> tuple[str, bool]:
    """A canonical solution in canonical formatting: ``ast.unparse`` of its tree (four-space indentation, no
    comments, no trailing blanks), ending in a newline. Returns (text, True), or the solution with its line
    ends cleaned and False if it does not parse."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.unparse(ast.parse(code)) + "\n", True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        lines = [line.rstrip() for line in code.replace("\r\n", "\n").split("\n")]
        return "\n".join(lines).rstrip("\n") + "\n", False


def kind_match(sample: str, planned: Sequence[int]) -> float:
    """Share of lines on which ``sample`` has the planned kind: lines are compared by index, and a missing or
    extra line counts as a miss (the denominator is the longer of the two)."""
    kinds = [kind for kind, _ in program_lines(sample)]
    longest = max(len(kinds), len(planned))
    if longest == 0:
        return 1.0
    return sum(1 for a, b in zip(kinds, planned, strict=False) if a == b) / longest


class TaskPlans:
    """The plans of a list of evaluation problems, as the ``plan_for`` of ``stage0.evaluate.realizer_samples``.

    For each problem the normalised reference is tokenised once and its line slots are kept; a call returns the
    one-row ``PlanCache`` of the chosen variant.
    """

    def __init__(self, bridge: PlanBridge, oracle: StructureOracle, tokenizer: Any, references: Sequence[str],
                 canvas_len: int, max_slots: int, variant: str, device: torch.device) -> None:
        if variant not in PLAN_VARIANTS:
            raise ValueError(f"plan must be one of {PLAN_VARIANTS}")
        if variant == "shuffled" and len(references) < 2:
            raise ValueError("the shuffled plan needs at least two problems")
        self.bridge, self.oracle, self.variant, self.device = bridge, oracle, variant, device
        self.max_slots = max_slots
        normalized = [normalize_reference(text) for text in references]
        self.texts = [text for text, _ in normalized]
        self.unparsed = sum(1 for _, ok in normalized if not ok)
        ids, starts = encode_with_starts(tokenizer, self.texts)
        budget = canvas_len - 1                                    # the canvas also holds one <eos>
        self.truncated = sum(1 for row in ids if len(row) > budget)
        self.slots: list[LineSlots] = [line_slots(text, row, max_slots, max_tokens=budget)
                                       for text, row in zip(self.texts, starts, strict=True)]
        self.kinds: list[list[int]] = [[kind for kind, _ in program_lines(text)] for text in self.texts]

    def source(self, index: int) -> int:
        """The problem whose reference gives problem ``index`` its plan."""
        return (index + 1) % len(self.slots) if self.variant == "shuffled" else index

    @torch.no_grad()
    def __call__(self, index: int, task: EvalTask | None = None) -> PlanCache:
        if self.variant == "null":
            return self.bridge.project(self.bridge.null_conditioning(1))
        slots = collate_slots([self.slots[self.source(index)]], self.max_slots).to(self.device)
        plan = require_oracle_plan(self.oracle(slots))
        return self.bridge.project(self.bridge.conditioning(plan.z, plan.mask))


def oracle_samples(realizer: Stage0Realizer, bridge: PlanBridge, oracle: StructureOracle, tokenizer: Any,
                   tasks: Sequence[EvalTask], benchmark: str, cfg: SamplerConfig, canvas_len: int,
                   num_samples: int, batch_size: int, seed: int, device: torch.device, variant: str = "oracle",
                   max_slots: int = 32, progress: bool = False) -> tuple[list[list[str]], dict, list[list[dict]]]:
    """``realizer_samples`` with the plan of each problem's reference (or the null or a shuffled plan).

    Returns the texts, the run's information (with ``ORACLE_LABEL``) and, per sample, how it followed the plan
    it was given: ``kind_match`` and whether it has the planned number of lines.
    """
    if benchmark != "mbpp":
        raise ValueError("the oracle reads a whole reference program: implemented for MBPP (the write format) only")
    if bridge.training or oracle.training:
        raise RuntimeError("evaluate the oracle in eval mode (plan dropout and the power iteration must be off)")
    plans = TaskPlans(bridge, oracle, tokenizer, [task.reference for task in tasks], canvas_len, max_slots, variant,
                      device)
    texts, info = realizer_samples(realizer, tokenizer, tasks, benchmark, cfg, canvas_len, num_samples, batch_size,
                                   seed, device, progress=progress, plan_for=plans)
    followed: list[list[dict]] = []
    matches, same_lines, total = 0.0, 0, 0
    for index, group in enumerate(texts):
        planned = plans.kinds[plans.source(index)]
        rows = []
        for text in group:
            lines = len(program_lines(text))
            row = {"kind_match": round(kind_match(text, planned), 4), "lines": lines, "plan_lines": len(planned)}
            matches, same_lines, total = matches + row["kind_match"], same_lines + (lines == len(planned)), total + 1
            rows.append(row)
        followed.append(rows)
    info = {**info, "stage": "stage1-oracle", "plan": variant, "label": ORACLE_LABEL, "max_slots": max_slots,
            "kind_match": round(matches / max(total, 1), 4), "line_count_match": round(same_lines / max(total, 1), 4),
            "references_not_parsed": plans.unparsed, "references_truncated": plans.truncated}
    return texts, info, followed
