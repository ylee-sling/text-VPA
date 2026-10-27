# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
dataset.py -- training data for PR-JEPA: records -> tokenised examples -> bucketed batches (Sec. 5.1, Table 1).

Each example carries
    c        context tokens (specification, docstring and signature)
    w_1      target tokens (the program), realised on a canvas padded with [PAD]
    y        the structural view: slots from ``TreeSitterASTExtractor`` serialised as
             [SLOT_<KIND>] slot-text [SLOT_<KIND>] slot-text ... with the slot of every token in
             ``slot_index`` (the target encoder pools per slot); [NULL_PLAN] if the program has no slot
    u        the design mode extracted from the reference (``ModeExtractor``, Sec. 5.1)
    readout  slot kinds, slot texts, identifier-token inventory and length bucket (plan readout targets)

``MixtureCorruption`` draws the FS-DFM training state (Eq. 11): t on the step-budget grid (or uniform),
w_0 ~ Uniform(V) or [MASK], and w_t^i ~ (1 - kappa_t) delta_{w_0^i} + kappa_t delta_{w_1^i}.
``LengthBucketSampler`` groups examples by canvas length to limit padding and shards batches across
data-parallel ranks.

Loaders read local files (no network): HumanEval and MBPP JSONL (optionally gzipped) and a generic
``{"context": ..., "target": ...}`` JSONL format.
"""

from __future__ import annotations

import gzip
import itertools
import json
import logging
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from text_vpa.data.ast_parser import ModeExtractor, SlotKind, TreeSitterASTExtractor
from text_vpa.data.synthetic import CodeRecord
from text_vpa.data.tokenization import TextVPATokenizer
from text_vpa.models.realizers.paths import KappaSchedule, mixture_path_sample

__all__ = [
    "Batch",
    "Example",
    "LengthBucketSampler",
    "MixtureCorruption",
    "PRJEPACollator",
    "PRJEPADataset",
    "build_dataloader",
    "epoch_batches",
    "load_humaneval",
    "load_jsonl",
    "load_mbpp",
]

logger = logging.getLogger(__name__)

TEXT_KINDS = (int(SlotKind.FUNCTION), int(SlotKind.IMPORT), int(SlotKind.CLASS))


# =============================================================================================
# Loaders
# =============================================================================================
def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
    return rows


def load_humaneval(path: str | Path) -> list[CodeRecord]:
    """HumanEval JSONL: context = prompt (signature + docstring), target = prompt + canonical_solution."""
    out = []
    for row in _read_jsonl(path):
        try:
            out.append(CodeRecord(row["prompt"], row["prompt"] + row["canonical_solution"], "python", row["task_id"]))
        except KeyError as exc:
            raise ValueError(f"{path}: HumanEval record without field {exc}") from exc
    return out


def load_mbpp(path: str | Path) -> list[CodeRecord]:
    """MBPP JSONL: context = text + the first test assertion (fixes the function name), target = code."""
    out = []
    for row in _read_jsonl(path):
        try:
            tests = row.get("test_list") or []
            context = row["text"] + ("\n" + tests[0] if tests else "")
            out.append(CodeRecord(context, row["code"].replace("\r\n", "\n"), "python", str(row.get("task_id"))))
        except KeyError as exc:
            raise ValueError(f"{path}: MBPP record without field {exc}") from exc
    return out


def load_jsonl(path: str | Path) -> list[CodeRecord]:
    """Generic JSONL: {"context": str, "target": str, "language"?: str, "task_id"?: str}."""
    out = []
    for row in _read_jsonl(path):
        try:
            out.append(CodeRecord(row["context"], row["target"], row.get("language", "python"), row.get("task_id")))
        except KeyError as exc:
            raise ValueError(f"{path}: record without field {exc}") from exc
    return out


# =============================================================================================
# Examples
# =============================================================================================
@dataclass(frozen=True)
class Example:
    context: list[int]
    target: list[int]
    structure: list[int]
    slot_index: list[int]
    mode: int
    slot_kinds: list[int]          # [M]
    slot_texts: list[list[int]]    # [M][L_text]
    inventory: list[int]           # identifier-bearing token ids of the target
    num_slots: int                 # real slots (<= M)
    task_id: str | None = None


class PRJEPADataset(Dataset):
    """Tokenised PR-JEPA examples.

    Args:
        records: ``CodeRecord`` pairs.
        tokenizer: shared tokenizer (context, structure and target).
        num_slots: M, slots of the structural view.
        max_context_tokens, max_output_tokens, max_structure_tokens: limits; targets that do not fit the
            canvas with one closing [PAD] are skipped, contexts and structures are truncated.
        max_text_tokens: L_text of the readout's slot texts.
        mode_scheme: ``ModeExtractor`` scheme.
        skip_unparseable: drop programs the extractor flags as erroneous.
        word_bearing: optional bool table [S] marking identifier-bearing tokens (inventory targets);
            computed from the tokenizer when omitted.
    """

    def __init__(self, records: Sequence[CodeRecord], tokenizer: TextVPATokenizer, num_slots: int,
                 max_context_tokens: int, max_output_tokens: int, max_structure_tokens: int,
                 max_text_tokens: int = 48, mode_scheme: str = "control_library", skip_unparseable: bool = True,
                 word_bearing: Sequence[bool] | None = None) -> None:
        self.tokenizer = tokenizer
        self.num_slots = num_slots
        self.max_output_tokens = max_output_tokens
        self.modes = ModeExtractor(mode_scheme)
        if word_bearing is None:
            word_bearing = [any(ch.isalpha() or ch == "_" for ch in s) and not tokenizer.is_special(i)
                            for i, s in enumerate(tokenizer.token_strings)]
        self.word_bearing = list(word_bearing)
        extractors: dict[str, TreeSitterASTExtractor] = {}
        self.examples: list[Example] = []
        self.skipped: dict[str, int] = {"too_long": 0, "unparseable": 0}
        for record in records:
            if record.language not in extractors:
                extractors[record.language] = TreeSitterASTExtractor(record.language)
            example = self._build(record, extractors[record.language], max_context_tokens, max_structure_tokens,
                                  max_text_tokens, skip_unparseable)
            if example is not None:
                self.examples.append(example)
        if any(self.skipped.values()):
            logger.info("skipped records: %s", self.skipped)

    def _build(self, record: CodeRecord, extractor: TreeSitterASTExtractor, max_context: int, max_structure: int,
               max_text: int, skip_unparseable: bool) -> Example | None:
        tok = self.tokenizer
        target = tok.encode(record.target)
        if len(target) + 1 > self.max_output_tokens:          # + the closing [PAD]
            self.skipped["too_long"] += 1
            return None
        try:
            structure = extractor.parse(record.target)
        except ValueError:                                     # e.g. source larger than the extractor's limit
            self.skipped["unparseable"] += 1
            return None
        if skip_unparseable and structure.has_errors:
            self.skipped["unparseable"] += 1
            return None
        slots = structure.slot_records(self.num_slots)
        y: list[int] = []
        slot_index: list[int] = []
        kinds: list[int] = []
        texts: list[list[int]] = []
        real = 0
        for m, slot in enumerate(slots):
            kinds.append(int(slot.kind))
            if slot.kind == SlotKind.EMPTY:
                texts.append([tok.pad_id] * max_text)
                continue
            real += 1
            ids = tok.encode(slot.text)
            piece = [tok.slot_token_id(int(slot.kind))] + ids
            room = max_structure - len(y)
            if room > 0:
                piece = piece[:room]
                y.extend(piece)
                slot_index.extend([m] * len(piece))
            texts.append(tok.pad_to(ids, max_text) if int(slot.kind) in TEXT_KINDS else [tok.pad_id] * max_text)
        if not y:
            y, slot_index = [tok.null_plan_id], [-1]
        inventory = sorted({i for i in target if self.word_bearing[i]})
        return Example(
            context=tok.encode(record.context)[:max_context] or [tok.pad_id],
            target=target, structure=y, slot_index=slot_index, mode=self.modes(structure),
            slot_kinds=kinds, slot_texts=texts, inventory=inventory, num_slots=real, task_id=record.task_id,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    def canvas_lengths(self, bucket_size: int) -> list[int]:
        """Canvas length of every example: target + one [PAD], rounded up to the bucket size."""
        return [_round_up(len(e.target) + 1, bucket_size, self.max_output_tokens) for e in self.examples]


def _round_up(n: int, multiple: int, cap: int) -> int:
    return min(cap, multiple * math.ceil(n / multiple))


# =============================================================================================
# Batches
# =============================================================================================
@dataclass(frozen=True)
class Batch:
    context: Tensor            # [B, M_c] int64
    context_mask: Tensor       # [B, M_c] bool
    target: Tensor             # [B, N] int64 w_1 on a [PAD] canvas
    target_lengths: Tensor     # [B] int64 (without the closing [PAD])
    structure: Tensor          # [B, T] int64 y
    structure_mask: Tensor     # [B, T] bool
    slot_index: Tensor         # [B, T] int64 (-1: no slot)
    modes: Tensor              # [B] int64
    slot_kinds: Tensor         # [B, M] int64
    slot_texts: Tensor         # [B, M, L_text] int64
    slot_text_mask: Tensor     # [B, M] bool
    inventory: Tensor          # [B, S] float32 multi-hot
    w_t: Tensor | None = None  # [B, N] corrupted canvas (FS-DFM training state)
    t: Tensor | None = None    # [B]
    h: Tensor | None = None    # [B]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> Batch:
        moved = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            moved[name] = value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value
        return Batch(**moved)

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])


class MixtureCorruption:
    """FS-DFM training state on the mixture path (Eq. 11), drawn in the collator.

    Args:
        vocab_size: S.
        step_budgets: trained budgets K (h = 1/K).
        schedule: kappa schedule.
        source: "uniform" or "mask".
        mask_id: [MASK] id for the mask source.
        time_sampling: "grid" (t = k/K) or "uniform".

    Each call draws from ``torch``'s global generator unless one is passed (DataLoader workers are seeded
    by PyTorch).
    """

    def __init__(self, vocab_size: int, step_budgets: Sequence[int], schedule: str = "linear",
                 source: str = "uniform", mask_id: int | None = None, time_sampling: str = "grid") -> None:
        self.vocab_size = vocab_size
        self.step_budgets = tuple(step_budgets)
        self.schedule = KappaSchedule(schedule)
        self.source = source
        self.mask_id = mask_id
        self.time_sampling = time_sampling

    def __call__(self, w_1: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        s = mixture_path_sample(w_1, self.schedule, self.vocab_size, self.step_budgets, self.source, self.mask_id,
                                self.time_sampling, generator)
        return s.w_t, s.t, s.h


class PRJEPACollator:
    """Pads a list of ``Example`` into a ``Batch``.

    The canvas length N is the longest target + 1 (the closing [PAD]), rounded up to ``bucket_size``
    and capped at ``max_output_tokens``. With ``corruption`` the FS-DFM state (w_t, t, h) is added.
    """

    def __init__(self, tokenizer: TextVPATokenizer, num_slots: int, max_text_tokens: int, bucket_size: int,
                 max_output_tokens: int, corruption: MixtureCorruption | None = None) -> None:
        self.pad_id = tokenizer.pad_id
        self.vocab_size = tokenizer.vocab_size
        self.num_slots = num_slots
        self.max_text_tokens = max_text_tokens
        self.bucket_size = bucket_size
        self.max_output_tokens = max_output_tokens
        self.corruption = corruption

    def _pad(self, rows: list[list[int]], length: int, fill: int) -> Tensor:
        out = torch.full((len(rows), length), fill, dtype=torch.long)
        for i, row in enumerate(rows):
            out[i, : len(row)] = torch.tensor(row[:length], dtype=torch.long)
        return out

    def __call__(self, examples: Sequence[Example]) -> Batch:
        if not examples:
            raise ValueError("empty batch")
        b = len(examples)
        m_c = max(len(e.context) for e in examples)
        n = _round_up(max(len(e.target) for e in examples) + 1, self.bucket_size, self.max_output_tokens)
        t_len = max(len(e.structure) for e in examples)
        context = self._pad([e.context for e in examples], m_c, self.pad_id)
        context_mask = torch.zeros(b, m_c, dtype=torch.bool)
        structure_mask = torch.zeros(b, t_len, dtype=torch.bool)
        inventory = torch.zeros(b, self.vocab_size, dtype=torch.float32)
        for i, e in enumerate(examples):
            context_mask[i, : len(e.context)] = True
            structure_mask[i, : len(e.structure)] = True
            if e.inventory:
                inventory[i, e.inventory] = 1.0
        slot_texts = torch.tensor([e.slot_texts for e in examples], dtype=torch.long)
        kinds = torch.tensor([e.slot_kinds for e in examples], dtype=torch.long)
        if kinds.shape != (b, self.num_slots) or slot_texts.shape != (b, self.num_slots, self.max_text_tokens):
            raise ValueError("examples were built with a different num_slots or max_text_tokens")
        text_mask = torch.zeros(b, self.num_slots, dtype=torch.bool)
        for kind in TEXT_KINDS:
            text_mask |= kinds == kind
        batch = Batch(
            context=context, context_mask=context_mask,
            target=self._pad([e.target for e in examples], n, self.pad_id),
            target_lengths=torch.tensor([len(e.target) for e in examples], dtype=torch.long),
            structure=self._pad([e.structure for e in examples], t_len, self.pad_id),
            structure_mask=structure_mask,
            slot_index=self._pad([e.slot_index for e in examples], t_len, -1),
            modes=torch.tensor([e.mode for e in examples], dtype=torch.long),
            slot_kinds=kinds, slot_texts=slot_texts, slot_text_mask=text_mask, inventory=inventory,
        )
        if self.corruption is not None:
            w_t, t, h = self.corruption(batch.target)
            batch = replace(batch, w_t=w_t, t=t, h=h)
        return batch


# =============================================================================================
# Sampling
# =============================================================================================
class LengthBucketSampler(Sampler[list[int]]):
    """Batches of examples with equal canvas length, shuffled per epoch and sharded across ranks.

    Every rank receives the same number of batches (the remainder is dropped), as data-parallel training
    requires.
    """

    def __init__(self, canvas_lengths: Sequence[int], batch_size: int, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = False, rank: int = 0, world_size: int = 1) -> None:
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid batch_size / rank / world_size")
        self.lengths = list(canvas_lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self._skip = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def skip_next(self, num_batches: int) -> None:
        """Skip the first ``num_batches`` batches of the next iteration only (exact resumption mid-epoch)."""
        if num_batches < 0:
            raise ValueError("num_batches must be non-negative")
        self._skip = num_batches

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets: dict[int, list[int]] = {}
        for index, length in enumerate(self.lengths):
            buckets.setdefault(length, []).append(index)
        batches = []
        for length in sorted(buckets):
            indices = buckets[length]
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        usable = len(batches) - len(batches) % self.world_size
        return batches[:usable][self.rank::self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()[self._skip:]
        self._skip = 0
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches())


def build_dataloader(dataset: PRJEPADataset, collator: PRJEPACollator, batch_size: int, bucket_size: int,
                     shuffle: bool = True, seed: int = 0, num_workers: int = 0, rank: int = 0, world_size: int = 1,
                     drop_last: bool = False) -> DataLoader:
    """DataLoader with length-bucketed, rank-sharded batches."""
    sampler = LengthBucketSampler(dataset.canvas_lengths(bucket_size), batch_size, shuffle, seed, drop_last,
                                  rank, world_size)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collator, num_workers=num_workers,
                      persistent_workers=num_workers > 0)


def epoch_batches(loader: object, epoch: int, skip: int) -> Iterator:
    """Batches of one epoch of ``loader``, starting after ``skip`` already-consumed batches.

    Uses ``LengthBucketSampler.set_epoch`` / ``skip_next`` when available (no collation of skipped batches),
    otherwise collates and discards them.
    """
    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)  # type: ignore[union-attr]
    if skip and hasattr(sampler, "skip_next"):
        sampler.skip_next(skip)  # type: ignore[union-attr]
        return iter(loader)  # type: ignore[call-overload]
    return itertools.islice(iter(loader), skip, None)  # type: ignore[call-overload]
