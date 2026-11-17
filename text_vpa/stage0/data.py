# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
data.py -- prompts, canvases and training states of Stage 0.

Two prompt formats, one per benchmark, used identically by the realizer, by its training data and by the
autoregressive baseline:

    complete   the prompt is a function's imports + signature + docstring (HumanEval's prompt; the corpus's
               ``context``); the canvas is the rest of the function (the body).
    write      the prompt is a description inside a module docstring followed by a blank line,
                   '\"\"\"\\n' + description + '\\n\"\"\"\\n\\n'
               (for MBPP the description is the task text and its first test); the canvas is the whole program.
               The blank line matters: with a single newline after the docstring (the BigCode harness's format)
               Gemma 4 E2B writes function bodies without indentation, greedy and sampled alike (measured on
               the L4, 17 Nov 2026); real modules put a blank line there.

Training examples come from the repository's corpus (``text_vpa.data.corpus``: {"context", "target", ...} rows,
``target`` starting with ``context``). Every row yields a ``complete`` example; a documented top-level function
yields, with probability ``write_fraction``, a ``write`` example instead: description = its docstring followed by
the line ``assert name(parameters) == ...`` (MBPP prompts name the function in an assert; the corpus has no
tests), canvas = imports + the function without its docstring.

A canvas is N token positions: the program's tokens, one <eos>, then <eos> up to N. Examples whose program does
not fit are dropped, never truncated (a cut-off program would teach unterminated code).

Training states (Text-VPA Eq. 11 with the mask source): K is drawn from the trained budgets, t = k/K on the grid
the sampler evaluates, and each canvas position holds its token with probability kappa_t, else [MASK].
"""

from __future__ import annotations

import ast
import glob
import gzip
import hashlib
import inspect
import json
import random
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from text_vpa.models.realizers.paths import KappaSchedule, mixture_path_sample

__all__ = [
    "Batch",
    "CorpusStream",
    "Example",
    "HFTokenizer",
    "Stage0Tokenizer",
    "TrainingState",
    "collate",
    "complete_example",
    "load_validation",
    "mbpp_prompt",
    "tokenize_examples",
    "training_state",
    "write_example",
    "write_prompt",
]


class Stage0Tokenizer(Protocol):
    bos_id: int
    eos_id: int
    pad_id: int
    mask_id: int
    vocab_size: int

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


class HFTokenizer:
    """A Hugging Face tokenizer behind ``Stage0Tokenizer`` (no special tokens are added by ``encode_batch``)."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.bos_id, self.eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.vocab_size = len(tokenizer)
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            candidate = tokenizer.convert_tokens_to_ids("<mask>")
            mask_id = candidate if candidate is not None and candidate != tokenizer.unk_token_id else None
        if mask_id is None or self.bos_id is None or self.eos_id is None:
            raise ValueError("the tokenizer needs <bos>, <eos> and a <mask> token (Gemma's vocabulary has all three)")
        self.mask_id = int(mask_id)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        if not texts:
            return []
        return self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)


# ---------------------------------------------------------------------------------------------
# Prompt formats
# ---------------------------------------------------------------------------------------------
def write_prompt(description: str) -> str:
    return '"""\n' + description.strip("\n") + '\n"""\n\n'


def mbpp_prompt(context: str) -> str:
    """``EvalTask.context`` of an MBPP task (text + first test) in the ``write`` format."""
    return write_prompt(context)


def complete_example(row: dict) -> tuple[str, str] | None:
    """(prompt, canvas text) of the ``complete`` format, or None if the row is not prefix-shaped."""
    context, target = row["context"], row["target"]
    if not context or not target.startswith(context) or len(target) <= len(context):
        return None
    return context, target[len(context):]


def write_example(row: dict) -> tuple[str, str] | None:
    """(prompt, canvas text) of the ``write`` format, or None when the row has no usable docstring.

    The function is the last top-level definition of ``target`` (imports come first). Methods are skipped
    (their ``self`` has no caller in a description), as are one-line definitions.
    """
    if not row.get("documented", True) or row.get("method", False):
        return None
    target = row["target"]
    try:
        with warnings.catch_warnings():                          # e.g. "\\d" in a plain string of corpus code
            warnings.simplefilter("ignore")
            module = ast.parse(target)
    except (SyntaxError, ValueError, RecursionError):
        return None
    functions = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return None
    func = functions[-1]
    doc = ast.get_docstring(func, clean=False)
    first = func.body[0]
    if doc is None or len(func.body) < 2 or first.lineno <= func.lineno or first.end_lineno is None:
        return None
    if func.body[1].lineno <= first.end_lineno:                    # code on the docstring's last line
        return None
    lines = target.splitlines(keepends=True)
    code = "".join(lines[: first.lineno - 1] + lines[first.end_lineno:])
    args = func.args
    names = [a.arg for a in args.posonlyargs + args.args]
    names += ["*" + args.vararg.arg] if args.vararg else []
    names += [a.arg for a in args.kwonlyargs]
    names += ["**" + args.kwarg.arg] if args.kwarg else []
    description = inspect.cleandoc(doc) + f"\nassert {func.name}({', '.join(names)}) == ..."
    return write_prompt(description), code


# ---------------------------------------------------------------------------------------------
# Examples and batches
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Example:
    prompt_ids: list[int]          # <bos> + prompt; its last token is the canvas anchor
    canvas_ids: list[int]          # program tokens + one <eos> (at most N)
    kind: str                      # "complete" | "write"


def tokenize_examples(pairs: Sequence[tuple[str, str, str]], tokenizer: Stage0Tokenizer, canvas_len: int,
                      max_prompt_tokens: int) -> tuple[list[Example], dict[str, int]]:
    """(prompt, canvas text, kind) triples -> examples; counts what was dropped and why."""
    prompts = tokenizer.encode_batch([p for p, _, _ in pairs])
    canvases = tokenizer.encode_batch([c for _, c, _ in pairs])
    out: list[Example] = []
    dropped = {"canvas_too_long": 0, "prompt_too_long": 0, "special_token": 0}
    special = {tokenizer.mask_id, tokenizer.eos_id, tokenizer.bos_id}
    for (_, _, kind), prompt, canvas in zip(pairs, prompts, canvases, strict=True):
        if len(canvas) + 1 > canvas_len:
            dropped["canvas_too_long"] += 1
        elif len(prompt) + 1 > max_prompt_tokens or not prompt:
            dropped["prompt_too_long"] += 1
        elif special.intersection(canvas):
            dropped["special_token"] += 1
        else:
            out.append(Example([tokenizer.bos_id, *prompt], [*canvas, tokenizer.eos_id], kind))
    return out, dropped


@dataclass
class Batch:
    prefix_rows: list[list[int]]   # prompts without their last token (>= 1 token each: <bos>)
    anchor: Tensor                 # [B] the last prompt token
    target: Tensor                 # [B, N] canvas tokens, <eos> after the program
    length: Tensor                 # [B] program tokens + 1 (the first <eos>)
    kinds: list[str]


def collate(examples: Sequence[Example], canvas_len: int, eos_id: int) -> Batch:
    target = torch.full((len(examples), canvas_len), eos_id, dtype=torch.long)
    for b, ex in enumerate(examples):
        if len(ex.canvas_ids) > canvas_len or len(ex.prompt_ids) < 2:
            raise ValueError("an example does not fit the canvas or has an empty prompt")
        target[b, : len(ex.canvas_ids)] = torch.tensor(ex.canvas_ids, dtype=torch.long)
    return Batch(
        prefix_rows=[ex.prompt_ids[:-1] for ex in examples],
        anchor=torch.tensor([ex.prompt_ids[-1] for ex in examples], dtype=torch.long),
        target=target,
        length=torch.tensor([len(ex.canvas_ids) for ex in examples], dtype=torch.long),
        kinds=[ex.kind for ex in examples],
    )


@dataclass
class TrainingState:
    canvas_ids: Tensor             # [B, N + 1] anchor, then w_t
    is_mask: Tensor                # [B, N + 1] bool (False at the anchor)
    t: Tensor                      # [B]
    h: Tensor                      # [B]
    weights: Tensor                # [B, N] loss weight of each canvas token (0 where it is not masked)


def training_state(batch: Batch, schedule: KappaSchedule, step_budgets: Sequence[int], mask_id: int,
                   vocab_size: int, pad_weight: float, generator: torch.Generator | None = None,
                   budget: int | None = None, step_index: int | None = None) -> TrainingState:
    """A state on the mask-source mixture path and its loss weights (on the CPU).

    Loss weights: 1 on masked program tokens and on the first <eos>, ``pad_weight`` on the masked <eos> filling
    the canvas after it, 0 on revealed positions (the mask source keeps them clean, so they carry no loss).
    ``budget`` and ``step_index`` pin K and k (evaluation); by default they are drawn (``mixture_path_sample``).
    """
    target = batch.target
    b, n = target.shape
    if budget is not None:
        k = step_index if step_index is not None else 0
        if not 0 <= k < budget:
            raise ValueError("need 0 <= step_index < budget")
        t = torch.full((b,), k / budget)
        h = torch.full((b,), 1.0 / budget)
        arrived = torch.rand(b, n, generator=generator) < schedule.kappa(t)[:, None]
        w_t = torch.where(arrived, target, torch.full_like(target, mask_id))
    else:
        sample = mixture_path_sample(target, schedule, vocab_size, list(step_budgets), source="mask",
                                     mask_id=mask_id, time_sampling="grid", generator=generator)
        w_t, t, h = sample.w_t, sample.t, sample.h
    masked = w_t == mask_id
    positions = torch.arange(n)[None, :]
    weights = torch.where(positions < batch.length[:, None], 1.0, pad_weight) * masked
    front = torch.zeros(b, 1, dtype=torch.bool)
    return TrainingState(torch.cat([batch.anchor[:, None], w_t], dim=1), torch.cat([front, masked], dim=1),
                         t.float(), h.float(), weights.float())


# ---------------------------------------------------------------------------------------------
# The corpus stream
# ---------------------------------------------------------------------------------------------
def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "big")


class CorpusStream:
    """Examples from the corpus's ``train-*.jsonl.gz`` shards, in a deterministic shuffled order, resumable.

    Each epoch visits the shards in a seeded permutation and reads ``chunk_rows`` rows at a time; a chunk's
    examples are built and shuffled with a seed derived from (seed, epoch, shard, row), so ``state()`` --
    (epoch, shard index, row index of the chunk, examples already taken from it) -- restores the stream exactly.
    """

    def __init__(self, corpus_dir: str | Path, tokenizer: Stage0Tokenizer, canvas_len: int, max_prompt_tokens: int,
                 write_fraction: float = 0.5, seed: int = 0, chunk_rows: int = 2048,
                 state: dict[str, int] | None = None) -> None:
        self.shards = sorted(glob.glob(str(Path(corpus_dir) / "train-*.jsonl.gz")))
        if not self.shards:
            raise FileNotFoundError(f"no train-*.jsonl.gz under {corpus_dir}")
        self.tokenizer, self.canvas_len, self.max_prompt_tokens = tokenizer, canvas_len, max_prompt_tokens
        self.write_fraction, self.seed, self.chunk_rows = write_fraction, seed, chunk_rows
        self.dropped: dict[str, int] = {"canvas_too_long": 0, "prompt_too_long": 0, "special_token": 0,
                                        "not_prefix": 0}
        self.rows_read = 0
        self._start = dict(state) if state else {"epoch": 0, "shard": 0, "row": 0, "taken": 0}
        self._chunk_state = dict(self._start)
        self._taken = self._start["taken"]
        self._iter = self._examples()

    def state(self) -> dict[str, int]:
        return {**self._chunk_state, "taken": self._taken}

    def _order(self, epoch: int) -> list[int]:
        order = list(range(len(self.shards)))
        random.Random(_seed(self.seed, "shards", epoch)).shuffle(order)
        return order

    def _chunks(self) -> Iterator[tuple[dict[str, int], list[dict]]]:
        epoch, shard, row = self._start["epoch"], self._start["shard"], self._start["row"]
        while True:
            order = self._order(epoch)
            while shard < len(order):
                chunk: list[dict] = []
                chunk_row = row
                with gzip.open(self.shards[order[shard]], "rt", encoding="utf-8") as handle:
                    for index, line in enumerate(handle):
                        if index < row:
                            continue
                        chunk.append(json.loads(line))
                        if len(chunk) == self.chunk_rows:
                            yield {"epoch": epoch, "shard": shard, "row": chunk_row}, chunk
                            chunk, chunk_row = [], index + 1
                if chunk:
                    yield {"epoch": epoch, "shard": shard, "row": chunk_row}, chunk
                shard, row = shard + 1, 0
            epoch, shard, row = epoch + 1, 0, 0

    def build(self, rows: Sequence[dict], rng: random.Random) -> list[Example]:
        """Rows -> examples (the write/complete choice and the order are drawn from ``rng``)."""
        pairs: list[tuple[str, str, str]] = []
        for row in rows:
            made = write_example(row) if rng.random() < self.write_fraction else None
            kind = "write"
            if made is None:
                made, kind = complete_example(row), "complete"
            if made is None:
                self.dropped["not_prefix"] += 1
                continue
            pairs.append((made[0], made[1], kind))
        examples, dropped = tokenize_examples(pairs, self.tokenizer, self.canvas_len, self.max_prompt_tokens)
        for key, value in dropped.items():
            self.dropped[key] += value
        rng.shuffle(examples)
        return examples

    def _examples(self) -> Iterator[Example]:
        skip = self._start["taken"]
        for position, rows in self._chunks():
            self.rows_read += len(rows)
            examples = self.build(rows, random.Random(_seed(self.seed, "chunk", *position.values())))
            self._chunk_state, self._taken = position, 0
            for example in examples:
                self._taken += 1
                if skip:
                    skip -= 1
                    continue
                yield example
            skip = 0

    def take(self, count: int) -> list[Example]:
        return [next(self._iter) for _ in range(count)]


def load_validation(corpus_dir: str | Path, tokenizer: Stage0Tokenizer, canvas_len: int, max_prompt_tokens: int,
                    limit: int, seed: int = 0) -> list[Example]:
    """The first ``limit`` usable examples of ``val.jsonl.gz``, alternating the two formats where possible."""
    path = Path(corpus_dir) / "val.jsonl.gz"
    pairs: list[tuple[str, str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            made = write_example(row) if index % 2 else None
            kind = "write"
            if made is None:
                made, kind = complete_example(row), "complete"
            if made is not None:
                pairs.append((made[0], made[1], kind))
            if len(pairs) >= 3 * limit:
                break
    examples, _ = tokenize_examples(pairs, tokenizer, canvas_len, max_prompt_tokens)
    return examples[:limit]
