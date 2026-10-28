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
import hashlib
import itertools
import json
import logging
import math
import multiprocessing
import pickle
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
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
class Example:
    """One tokenised example, stored compactly: int32 arrays, and slot texts only for the slots that carry one.

    Python lists of token ids cost about 36 bytes per token; at M = 256 slots x L_text = 48 a dense slot-text
    table alone is ~100 KB per example, i.e. tens of GB for a real corpus. Here an example of a few hundred
    tokens takes a few KB. The list-valued properties (``context``, ``target``, ``structure``, ``slot_index``,
    ``slot_kinds``, ``slot_texts`` [M][L_text], ``inventory``) materialise the full layout on demand; the
    collator reads the arrays.
    """

    __slots__ = ("context_ids", "target_ids", "structure_ids", "slot_index_ids", "mode", "kinds", "text_slots",
                 "text_ids", "text_len", "pad_id", "inventory_ids", "num_slots", "task_id")

    def __init__(self, context: Sequence[int], target: Sequence[int], structure: Sequence[int],
                 slot_index: Sequence[int], mode: int, slot_kinds: Sequence[int],
                 text_rows: dict[int, Sequence[int]], text_len: int, pad_id: int, inventory: Sequence[int],
                 num_slots: int, task_id: str | None = None) -> None:
        self.context_ids = np.asarray(context, dtype=np.int32)
        self.target_ids = np.asarray(target, dtype=np.int32)
        self.structure_ids = np.asarray(structure, dtype=np.int32)
        self.slot_index_ids = np.asarray(slot_index, dtype=np.int32)
        if self.slot_index_ids.shape != self.structure_ids.shape:
            raise ValueError("structure and slot_index must have the same length")
        self.mode = int(mode)
        self.kinds = np.asarray(slot_kinds, dtype=np.int8)
        slots = sorted(text_rows)
        self.text_slots = np.asarray(slots, dtype=np.int32)
        self.text_ids = tuple(np.asarray(list(text_rows[m])[:text_len], dtype=np.int32) for m in slots)
        self.text_len = int(text_len)
        self.pad_id = int(pad_id)
        self.inventory_ids = np.asarray(inventory, dtype=np.int32)
        self.num_slots = int(num_slots)
        self.task_id = task_id

    def __getstate__(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            object.__setattr__(self, k, v)

    @property
    def context(self) -> list[int]:
        return self.context_ids.tolist()

    @property
    def target(self) -> list[int]:
        return self.target_ids.tolist()

    @property
    def structure(self) -> list[int]:
        return self.structure_ids.tolist()

    @property
    def slot_index(self) -> list[int]:
        return self.slot_index_ids.tolist()

    @property
    def slot_kinds(self) -> list[int]:
        return self.kinds.tolist()

    @property
    def inventory(self) -> list[int]:
        return self.inventory_ids.tolist()

    @property
    def slot_texts(self) -> list[list[int]]:
        """[M][L_text]: each text slot's ids right-padded with [PAD]; every other slot all [PAD]."""
        dense = [[self.pad_id] * self.text_len for _ in range(len(self.kinds))]
        for m, ids in zip(self.text_slots.tolist(), self.text_ids, strict=True):
            dense[m][: len(ids)] = ids.tolist()
        return dense

    @property
    def nbytes(self) -> int:
        arrays = (self.context_ids, self.target_ids, self.structure_ids, self.slot_index_ids, self.kinds,
                  self.text_slots, self.inventory_ids, *self.text_ids)
        return sum(a.nbytes for a in arrays)


class _ExampleBuilder:
    """Records -> ``Example``s (picklable: it is sent once to each worker of a parallel build)."""

    def __init__(self, tokenizer: TextVPATokenizer, num_slots: int, max_context: int, max_output: int,
                 max_structure: int, max_text: int, mode_scheme: str, skip_unparseable: bool,
                 word_bearing: Sequence[bool]) -> None:
        self.tokenizer = tokenizer
        self.num_slots = num_slots
        self.max_context = max_context
        self.max_output = max_output
        self.max_structure = max_structure
        self.max_text = max_text
        self.mode_scheme = mode_scheme
        self.skip_unparseable = skip_unparseable
        self.word_bearing = np.asarray(word_bearing, dtype=bool)
        self._extractors: dict[str, TreeSitterASTExtractor] = {}
        self._modes: ModeExtractor | None = None

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_extractors"], state["_modes"] = {}, None        # tree-sitter parsers are rebuilt per process
        return state

    def build_many(self, records: Sequence[CodeRecord]) -> tuple[list[Example], dict[str, int]]:
        skipped = {"too_long": 0, "unparseable": 0}
        out = []
        for record in records:
            example = self.build(record, skipped)
            if example is not None:
                out.append(example)
        return out, skipped

    def build(self, record: CodeRecord, skipped: dict[str, int]) -> Example | None:
        tok = self.tokenizer
        if self._modes is None:
            self._modes = ModeExtractor(self.mode_scheme)
        if record.language not in self._extractors:
            self._extractors[record.language] = TreeSitterASTExtractor(record.language)
        extractor = self._extractors[record.language]
        target = tok.encode(record.target)
        if len(target) + 1 > self.max_output:                  # + the closing [PAD]
            skipped["too_long"] += 1
            return None
        try:
            structure = extractor.parse(record.target)
        except ValueError:                                     # e.g. source larger than the extractor's limit
            skipped["unparseable"] += 1
            return None
        if self.skip_unparseable and structure.has_errors:
            skipped["unparseable"] += 1
            return None
        slots = structure.slot_records(self.num_slots)
        y: list[int] = []
        slot_index: list[int] = []
        kinds: list[int] = []
        texts: dict[int, list[int]] = {}
        real = 0
        for m, slot in enumerate(slots):
            kinds.append(int(slot.kind))
            if slot.kind == SlotKind.EMPTY:
                continue
            real += 1
            ids = tok.encode(slot.text)
            piece = [tok.slot_token_id(int(slot.kind))] + ids
            room = self.max_structure - len(y)
            if room > 0:
                piece = piece[:room]
                y.extend(piece)
                slot_index.extend([m] * len(piece))
            if int(slot.kind) in TEXT_KINDS:
                texts[m] = ids[: self.max_text]
        if not y:
            y, slot_index = [tok.null_plan_id], [-1]
        inventory = sorted({i for i in target if self.word_bearing[i]})
        return Example(
            context=tok.encode(record.context)[: self.max_context] or [tok.pad_id],
            target=target, structure=y, slot_index=slot_index, mode=self._modes(structure),
            slot_kinds=kinds, text_rows=texts, text_len=self.max_text, pad_id=tok.pad_id, inventory=inventory,
            num_slots=real, task_id=record.task_id,
        )


_WORKER: dict[str, _ExampleBuilder] = {}         # the builder of this worker process


def _init_worker(builder: _ExampleBuilder) -> None:
    _WORKER["builder"] = builder


def _build_chunk(records: Sequence[CodeRecord]) -> tuple[list[Example], dict[str, int]]:
    return _WORKER["builder"].build_many(records)


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
        num_proc: worker processes for the build (the examples and their order do not depend on it).
    """

    def __init__(self, records: Sequence[CodeRecord], tokenizer: TextVPATokenizer, num_slots: int,
                 max_context_tokens: int, max_output_tokens: int, max_structure_tokens: int,
                 max_text_tokens: int = 48, mode_scheme: str = "control_library", skip_unparseable: bool = True,
                 word_bearing: Sequence[bool] | None = None, num_proc: int = 1) -> None:
        self.tokenizer = tokenizer
        self.num_slots = num_slots
        self.max_output_tokens = max_output_tokens
        self.modes = ModeExtractor(mode_scheme)
        if word_bearing is None:
            word_bearing = [any(ch.isalpha() or ch == "_" for ch in s) and not tokenizer.is_special(i)
                            for i, s in enumerate(tokenizer.token_strings)]
        self.word_bearing = list(word_bearing)
        builder = _ExampleBuilder(tokenizer, num_slots, max_context_tokens, max_output_tokens, max_structure_tokens,
                                  max_text_tokens, mode_scheme, skip_unparseable, self.word_bearing)
        self.examples: list[Example] = []
        self.skipped: dict[str, int] = {"too_long": 0, "unparseable": 0}
        records = list(records)
        if num_proc > 1 and len(records) >= 2 * num_proc:
            size = max(1, min(2000, math.ceil(len(records) / (4 * num_proc))))
            chunks = [records[i:i + size] for i in range(0, len(records), size)]
            context = multiprocessing.get_context("spawn")   # fork is unsafe in a multi-threaded (torch) process
            with context.Pool(num_proc, initializer=_init_worker, initargs=(builder,)) as pool:
                parts = pool.imap(_build_chunk, chunks)              # ordered: same examples as num_proc = 1
                for examples, skipped in parts:
                    self.examples.extend(examples)
                    for key, value in skipped.items():
                        self.skipped[key] += value
        else:
            self.examples, self.skipped = builder.build_many(records)
        if any(self.skipped.values()):
            logger.info("skipped records: %s", self.skipped)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    def canvas_lengths(self, bucket_size: int) -> list[int]:
        """Canvas length of every example: target + one [PAD], rounded up to the bucket size."""
        return [_round_up(len(e.target_ids) + 1, bucket_size, self.max_output_tokens) for e in self.examples]

    @property
    def nbytes(self) -> int:
        """Bytes held by the examples' arrays."""
        return sum(e.nbytes for e in self.examples)

    # -----------------------------------------------------------------------------------------
    @staticmethod
    def cache_key(records: Sequence[CodeRecord], tokenizer: TextVPATokenizer, **settings: object) -> str:
        """SHA-256 of the records, the tokenizer spec and the build settings."""
        digest = hashlib.sha256()
        digest.update(json.dumps({"tokenizer": tokenizer.spec(), "settings": settings, "format": 2},
                                 sort_keys=True, default=str).encode())
        for r in records:
            for field in (r.context, r.target, r.language, r.task_id or ""):
                digest.update(field.encode("utf-8", "surrogatepass"))
                digest.update(b"\x00")
        return digest.hexdigest()

    @classmethod
    def cached(cls, cache_dir: str | Path | None, records: Sequence[CodeRecord], tokenizer: TextVPATokenizer,
               num_slots: int, max_context_tokens: int, max_output_tokens: int, max_structure_tokens: int,
               max_text_tokens: int = 48, mode_scheme: str = "control_library", num_proc: int = 1) -> PRJEPADataset:
        """Build the dataset, or load it from ``cache_dir`` when an identical build was saved there."""
        settings = dict(num_slots=num_slots, max_context=max_context_tokens, max_output=max_output_tokens,
                        max_structure=max_structure_tokens, max_text=max_text_tokens, mode_scheme=mode_scheme)
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / f"dataset-{cls.cache_key(records, tokenizer, **settings)[:32]}.pkl"
            if path.exists():
                with open(path, "rb") as handle:
                    state = pickle.load(handle)
                dataset = cls.__new__(cls)
                dataset.tokenizer = tokenizer
                dataset.modes = ModeExtractor(mode_scheme)
                dataset.__dict__.update(state)
                logger.info("loaded %d cached examples from %s", len(dataset), path)
                return dataset
        dataset = cls(records, tokenizer, num_slots, max_context_tokens, max_output_tokens, max_structure_tokens,
                      max_text_tokens, mode_scheme, num_proc=num_proc)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as handle:
                pickle.dump({k: getattr(dataset, k) for k in ("num_slots", "max_output_tokens", "word_bearing",
                                                               "examples", "skipped")},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            logger.info("cached %d examples (%.1f MB of arrays) in %s", len(dataset), dataset.nbytes / 2**20, path)
        return dataset


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

    @staticmethod
    def _pad(rows: Sequence[np.ndarray], length: int, fill: int) -> Tensor:
        out = torch.full((len(rows), length), fill, dtype=torch.long)
        for i, row in enumerate(rows):
            row = row[:length]
            out[i, : len(row)] = torch.from_numpy(row.astype(np.int64))
        return out

    def __call__(self, examples: Sequence[Example]) -> Batch:
        if not examples:
            raise ValueError("empty batch")
        b = len(examples)
        m_c = max(len(e.context_ids) for e in examples)
        n = _round_up(max(len(e.target_ids) for e in examples) + 1, self.bucket_size, self.max_output_tokens)
        t_len = max(len(e.structure_ids) for e in examples)
        context_mask = torch.zeros(b, m_c, dtype=torch.bool)
        structure_mask = torch.zeros(b, t_len, dtype=torch.bool)
        inventory = torch.zeros(b, self.vocab_size, dtype=torch.float32)
        kinds = torch.zeros(b, self.num_slots, dtype=torch.long)
        slot_texts = torch.full((b, self.num_slots, self.max_text_tokens), self.pad_id, dtype=torch.long)
        for i, e in enumerate(examples):
            if len(e.kinds) != self.num_slots or e.text_len != self.max_text_tokens:
                raise ValueError("examples were built with a different num_slots or max_text_tokens")
            context_mask[i, : len(e.context_ids)] = True
            structure_mask[i, : len(e.structure_ids)] = True
            if len(e.inventory_ids):
                inventory[i, torch.from_numpy(e.inventory_ids.astype(np.int64))] = 1.0
            kinds[i] = torch.from_numpy(e.kinds.astype(np.int64))
            for m, ids in zip(e.text_slots.tolist(), e.text_ids, strict=True):
                slot_texts[i, m, : len(ids)] = torch.from_numpy(ids.astype(np.int64))
        text_mask = torch.zeros(b, self.num_slots, dtype=torch.bool)
        for kind in TEXT_KINDS:
            text_mask |= kinds == kind
        batch = Batch(
            context=self._pad([e.context_ids for e in examples], m_c, self.pad_id), context_mask=context_mask,
            target=self._pad([e.target_ids for e in examples], n, self.pad_id),
            target_lengths=torch.tensor([len(e.target_ids) for e in examples], dtype=torch.long),
            structure=self._pad([e.structure_ids for e in examples], t_len, self.pad_id),
            structure_mask=structure_mask,
            slot_index=self._pad([e.slot_index_ids for e in examples], t_len, -1),
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
