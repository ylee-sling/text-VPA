# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
data.py -- Stage 0's training examples with the line slots of their canvas program.

Nothing about the examples changes: the same corpus stream, the same two prompt formats, the same tokens, the
same examples dropped for the same reasons and the same order (tested against ``stage0.data``), so a stream
state saved by a Stage 0 run continues here. Each example additionally carries ``LineSlots`` (``slots.py``),
computed from the canvas text and the tokenizer's character offsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from text_vpa.stage0.data import (
    Batch,
    CorpusStream,
    Example,
    Stage0Tokenizer,
    _is_unicode,
    collate,
    load_validation,
)
from text_vpa.stage1.slots import LineSlots, SlotBatch, collate_slots, encode_with_starts, line_slots

__all__ = ["PlanBatch", "PlanCorpusStream", "PlanExample", "collate_plan", "load_plan_validation",
           "tokenize_plan_examples"]


@dataclass(frozen=True)
class PlanExample(Example):
    slots: LineSlots               # of the canvas program (without its <eos>)


@dataclass
class PlanBatch(Batch):
    slots: SlotBatch               # [B, M] each


def tokenize_plan_examples(pairs: Sequence[tuple[str, str, str]], tokenizer: Stage0Tokenizer, canvas_len: int,
                           max_prompt_tokens: int, max_slots: int = 32
                           ) -> tuple[list[PlanExample], dict[str, int]]:
    """``stage0.data.tokenize_examples`` with the line slots of each kept canvas (same examples, same drops)."""
    dropped = {"canvas_too_long": 0, "prompt_too_long": 0, "special_token": 0, "invalid_unicode": 0}
    valid = [pair for pair in pairs if _is_unicode(pair[0]) and _is_unicode(pair[1])]
    dropped["invalid_unicode"] = len(pairs) - len(valid)
    prompts = tokenizer.encode_batch([p for p, _, _ in valid])
    canvases, starts = encode_with_starts(tokenizer, [c for _, c, _ in valid])
    out: list[PlanExample] = []
    special = {tokenizer.mask_id, tokenizer.eos_id, tokenizer.bos_id}
    for (_, text, kind), prompt, canvas, offsets in zip(valid, prompts, canvases, starts, strict=True):
        if len(canvas) + 1 > canvas_len:
            dropped["canvas_too_long"] += 1
        elif len(prompt) + 1 > max_prompt_tokens or not prompt:
            dropped["prompt_too_long"] += 1
        elif special.intersection(canvas):
            dropped["special_token"] += 1
        else:
            out.append(PlanExample([tokenizer.bos_id, *prompt], [*canvas, tokenizer.eos_id], kind,
                                   line_slots(text, offsets, max_slots)))
    return out, dropped


def collate_plan(examples: Sequence[PlanExample], canvas_len: int, eos_id: int, max_slots: int) -> PlanBatch:
    base = collate(examples, canvas_len, eos_id)
    return PlanBatch(base.prefix_rows, base.anchor, base.target, base.length, base.kinds,
                     collate_slots([example.slots for example in examples], max_slots))


class PlanCorpusStream(CorpusStream):
    """``CorpusStream`` whose examples carry line slots; its ``state()`` is interchangeable with Stage 0's."""

    def __init__(self, *args: Any, max_slots: int = 32, **kwargs: Any) -> None:
        self.max_slots = max_slots
        super().__init__(*args, **kwargs)

    def tokenize(self, pairs: Sequence[tuple[str, str, str]]) -> tuple[list[Example], dict[str, int]]:
        return tokenize_plan_examples(pairs, self.tokenizer, self.canvas_len, self.max_prompt_tokens,
                                      self.max_slots)


def load_plan_validation(corpus_dir: str | Path, tokenizer: Stage0Tokenizer, canvas_len: int,
                         max_prompt_tokens: int, limit: int, max_slots: int = 32) -> list[PlanExample]:
    """Stage 0's held-out examples (``load_validation``), with line slots."""
    return load_validation(corpus_dir, tokenizer, canvas_len, max_prompt_tokens, limit,
                           tokenize=partial(tokenize_plan_examples, max_slots=max_slots))
