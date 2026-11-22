# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
oracle.py -- the two oracles of docs/stage1.md: structure (step 1) and latent (step 2). Diagnostics: they are
never the method.

**Structure oracle.** The realizer is told the reference program's true line structure and nothing else of it:

    Z_m = E_kind[kind_m] + E_depth[depth_m] + W code(start_m, length_m)        m = 1 .. M (line slots)

with ``slots.layout_code`` the fixed code of the slot's first canvas token and its number of tokens. No token of
the reference and no backbone state of it enters: two programs with the same line structure get the same plan
(tested). E_kind, E_depth and W are trained together with the bridge. The plan has no mode row: a line skeleton
is a function of the kinds and depths that the slots already hold.

**Latent oracle.** The realizer is told what the reference's lines *say*, as the frozen backbone reads them:

    Z_m = [len_m > 0] W_s LN( mean_{k in line m} H_k ) + W code(start_m, length_m)

with H_k the frozen backbone's hidden state after token k of the reference, read causally after its prompt
through the first ``latent_layers`` blocks (15 on Gemma 4 E2B: the KV-storing blocks, the paper's l_p = 15 and
what the realizer's prefill computes for the prompt). LN is a LayerNorm with its own scale and shift (the
residual stream's scale varies by orders of magnitude between dimensions; the bridge's keys are Lipschitz in
Z, so Z's scale must be controlled). A line without a token of its own has no state and gets its layout code
only. This is the most a slot plan of M rows of width d can carry about the reference: if it does not make
K = 8 work, no slot plan will (the design's step 2).

In training the reference is the example's own canvas. At evaluation it is the benchmark's canonical solution,
*normalised* first (``normalize_reference``: ``ast.unparse`` of its syntax tree). MBPP's solutions are indented
with tabs or with one, two or three spaces in two thirds of the problems (60 of the 90 validation problems) and
often carry trailing blanks; read as they are, their depths (indentation // 4) and token layout would describe a
formatting the realizer was not trained to write, and the oracle would understate what structure can give. The
normalised program is the same program (it must still pass its tests; the preflight checks that).

Three plans can be evaluated with one trained model, all of them labelled as oracle rows:

    oracle     the plan of the problem's own reference
    null       the bridge's null token (what the same weights do without a plan)
    shuffled   the plan of another problem's reference (for the latent oracle, read after this problem's prompt)

How a sample followed its plan: ``kind_match``, the share of lines whose kind is the planned one, and
``reproduced``, whether the sample is the plan's reference program (the same syntax tree). The first tells "the
structure did not help" from "the structure did not reach the realizer"; the second says how much of the
content a latent plan delivers.
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from text_vpa.evaluation.tasks import EvalTask
from text_vpa.stage0.backbone import PrefixCache, Stage0Realizer
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

__all__ = ["ORACLE_KINDS", "PLAN_VARIANTS", "LatentOracle", "StructureOracle", "TaskPlans", "kind_match",
           "line_means", "normalize_reference", "oracle_samples", "reference_canvas", "reproduced"]

PLAN_VARIANTS: tuple[str, ...] = ("oracle", "null", "shuffled")
ORACLE_KINDS: dict[str, str] = {"oracle-structure": "structure", "oracle-latent": "latent"}


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


def line_means(states: Tensor, slots: SlotBatch) -> Tensor:
    """Mean state of each slot's tokens: ``[B, N + 1, D]`` (index k + 1 after canvas token k, as
    ``Stage0Realizer.frozen_states``), ``SlotBatch`` -> ``[B, M, D]`` float32; zero for a slot without tokens.

    Slot m holds canvas tokens start_m .. start_m + length_m - 1, i.e. state indices start_m + 1 .. start_m +
    length_m."""
    b, c, _ = states.shape
    index = torch.arange(c - 1, device=states.device)                                       # canvas token k
    start, length = slots.start[..., None], slots.length[..., None]                         # [B, M, 1]
    member = (index >= start) & (index < start + length) & slots.mask[..., None]           # [B, M, N]
    weights = member.float() / length.clamp_min(1).float()
    return torch.bmm(weights, states[:, 1:].float())


class LatentOracle(nn.Module):
    """Frozen backbone states of the reference and its line slots -> ``OraclePlan`` (module docstring).

    Shapes: states ``[B, N + 1, D]``, ``SlotBatch`` of ``[B, M]`` tensors -> z ``[B, M, d]``, mask ``[B, M]``.
    """

    def __init__(self, hidden_size: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.norm = nn.LayerNorm(hidden_size)                       # LN
        self.content = nn.Linear(hidden_size, latent_dim)           # W_s
        self.layout = nn.Linear(latent_dim, latent_dim)             # W on the fixed code of (start, length)

    def forward(self, states: Tensor, slots: SlotBatch) -> OraclePlan:
        if states.dim() != 3 or states.shape[0] != slots.kind.shape[0]:
            raise ValueError("expected states [B, N + 1, D] with the slots' batch size")
        pooled = line_means(states, slots)                                                  # [B, M, D]
        has_tokens = (slots.length > 0) & slots.mask
        content = self.content(self.norm(pooled)) * has_tokens[..., None]
        code = layout_code(slots.start, slots.length, self.latent_dim)
        z = content + self.layout(code).float()
        return OraclePlan(z * slots.mask[..., None], slots.mask, "latent")


def reference_canvas(anchor: Tensor, program: Sequence[Sequence[int]], canvas_len: int, eos_id: int) -> Tensor:
    """[anchor, program tokens, <eos> ...] as the realizer's canvas, ``[B, N + 1]`` (training's layout)."""
    rows = torch.full((len(program), canvas_len + 1), eos_id, dtype=torch.long)
    rows[:, 0] = anchor.cpu()
    for b, ids in enumerate(program):
        cut = list(ids)[: canvas_len - 1]
        rows[b, 1:1 + len(cut)] = torch.tensor(cut, dtype=torch.long)
    return rows.to(anchor.device)


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


def reproduced(sample: str, reference: str) -> bool:
    """True if ``sample`` parses to the same program as the (normalised) ``reference``: equal ``ast.unparse``."""
    text, ok = normalize_reference(sample)
    return ok and text == reference


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
    one-row ``PlanCache`` of the chosen variant. The latent oracle reads the reference with the frozen backbone
    after the prompt that ``realizer_samples`` has just cached for the problem (``realizer`` is then required).
    """

    def __init__(self, bridge: PlanBridge, oracle: StructureOracle | LatentOracle, tokenizer: Any,
                 references: Sequence[str], canvas_len: int, max_slots: int, variant: str, device: torch.device,
                 realizer: Stage0Realizer | None = None, latent_layers: int = 15) -> None:
        if variant not in PLAN_VARIANTS:
            raise ValueError(f"plan must be one of {PLAN_VARIANTS}")
        if variant == "shuffled" and len(references) < 2:
            raise ValueError("the shuffled plan needs at least two problems")
        if isinstance(oracle, LatentOracle) and realizer is None:
            raise ValueError("the latent oracle reads the reference with the realizer's frozen backbone")
        self.bridge, self.oracle, self.variant, self.device = bridge, oracle, variant, device
        self.realizer, self.latent_layers, self.canvas_len = realizer, latent_layers, canvas_len
        self.eos_id = tokenizer.eos_id
        self.max_slots = max_slots
        normalized = [normalize_reference(text) for text in references]
        self.texts = [text for text, _ in normalized]
        self.unparsed = sum(1 for _, ok in normalized if not ok)
        ids, starts = encode_with_starts(tokenizer, self.texts)
        self.ids = ids
        budget = canvas_len - 1                                    # the canvas also holds one <eos>
        self.truncated = sum(1 for row in ids if len(row) > budget)
        self.slots: list[LineSlots] = [line_slots(text, row, max_slots, max_tokens=budget)
                                       for text, row in zip(self.texts, starts, strict=True)]
        self.kinds: list[list[int]] = [[kind for kind, _ in program_lines(text)] for text in self.texts]

    def source(self, index: int) -> int:
        """The problem whose reference gives problem ``index`` its plan."""
        return (index + 1) % len(self.slots) if self.variant == "shuffled" else index

    @torch.no_grad()
    def __call__(self, index: int, task: EvalTask | None = None, cache: PrefixCache | None = None,
                 anchor: int | None = None) -> PlanCache:
        if self.variant == "null":
            return self.bridge.project(self.bridge.null_conditioning(1))
        source = self.source(index)
        slots = collate_slots([self.slots[source]], self.max_slots).to(self.device)
        if isinstance(self.oracle, LatentOracle):
            if cache is None or anchor is None:
                raise ValueError("the latent oracle needs the problem's prompt cache and anchor")
            canvas = reference_canvas(torch.tensor([anchor], device=self.device), [self.ids[source]],
                                      self.canvas_len, self.eos_id)
            states = self.realizer.frozen_states(cache, canvas, self.latent_layers)
            plan = require_oracle_plan(self.oracle(states, slots))
        else:
            plan = require_oracle_plan(self.oracle(slots))
        return self.bridge.project(self.bridge.conditioning(plan.z, plan.mask))


def oracle_samples(realizer: Stage0Realizer, bridge: PlanBridge, oracle: StructureOracle | LatentOracle,
                   tokenizer: Any, tasks: Sequence[EvalTask], benchmark: str, cfg: SamplerConfig, canvas_len: int,
                   num_samples: int, batch_size: int, seed: int, device: torch.device, variant: str = "oracle",
                   max_slots: int = 32, progress: bool = False, latent_layers: int = 15
                   ) -> tuple[list[list[str]], dict, list[list[dict]]]:
    """``realizer_samples`` with the plan of each problem's reference (or the null or a shuffled plan).

    Returns the texts, the run's information (with ``ORACLE_LABEL``) and, per sample, how it followed the plan
    it was given: ``kind_match``, whether it has the planned number of lines, and whether it reproduced the
    plan's reference program.
    """
    if benchmark != "mbpp":
        raise ValueError("the oracle reads a whole reference program: implemented for MBPP (the write format) only")
    if bridge.training or oracle.training:
        raise RuntimeError("evaluate the oracle in eval mode (plan dropout and the power iteration must be off)")
    plans = TaskPlans(bridge, oracle, tokenizer, [task.reference for task in tasks], canvas_len, max_slots, variant,
                      device, realizer=realizer, latent_layers=latent_layers)
    texts, info = realizer_samples(realizer, tokenizer, tasks, benchmark, cfg, canvas_len, num_samples, batch_size,
                                   seed, device, progress=progress, plan_for=plans)
    followed: list[list[dict]] = []
    matches, same_lines, copies, total = 0.0, 0, 0, 0
    for index, group in enumerate(texts):
        planned = plans.kinds[plans.source(index)]
        reference = plans.texts[plans.source(index)]
        rows = []
        for text in group:
            lines = len(program_lines(text))
            row = {"kind_match": round(kind_match(text, planned), 4), "lines": lines, "plan_lines": len(planned),
                   "reproduced": reproduced(text, reference)}
            matches, same_lines, total = matches + row["kind_match"], same_lines + (lines == len(planned)), total + 1
            copies += row["reproduced"]
            rows.append(row)
        followed.append(rows)
    kind = "latent" if isinstance(oracle, LatentOracle) else "structure"
    info = {**info, "stage": "stage1-oracle", "oracle": kind, "plan": variant, "label": ORACLE_LABEL,
            "max_slots": max_slots, "kind_match": round(matches / max(total, 1), 4),
            "line_count_match": round(same_lines / max(total, 1), 4), "reproduced": round(copies / max(total, 1), 4),
            "references_not_parsed": plans.unparsed, "references_truncated": plans.truncated}
    return texts, info, followed
