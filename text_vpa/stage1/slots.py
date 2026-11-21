# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
slots.py -- line slots: the structure of a canvas program that a plan carries (docs/stage1.md, "Components").

Slot m is physical line m of the program on the canvas (the text between two newlines; the newline belongs to
the line it ends). Lines from the M-th on share the last slot. Per slot:

    kind      a lexical class of the line, read from its first token (``KINDS``, 27 classes): a statement keyword
              (def, return, if, for, ...), or one of blank / comment / decorator / string / close / assign / call /
              other. It is a function of the line's own characters only, so it is defined for lines of generated
              or broken programs and for continuation lines too (``line_kind``).
    depth     min(7, indentation columns // 4), a tab counting as 4 columns; 0 for a blank line.
    start     index of the line's first canvas token = the number of tokens of all earlier lines.
    length    number of canvas tokens of the line. A token belongs to the line of its first character
              (``token_starts``: character offsets from the tokenizer). A line can have no token of its own
              (a blank line whose newline was merged into the previous line's token): length 0.

The <eos> after the program and the <eos> filling the canvas belong to no slot.

The mode of the planner stage (step 3 of the design) is the program's *line skeleton*: the (kind, depth) of its
non-blank lines, cut at 16 lines (``skeleton``); ``Codebook`` holds the most frequent skeletons and "other".

``position_code`` / ``layout_code`` are the fixed sinusoidal codes (no parameters) of a canvas position and of a
slot's (start, length): what lets a canvas position find the slot it lies in.
"""

from __future__ import annotations

import bisect
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "KINDS",
    "MAX_DEPTH",
    "Codebook",
    "LineSlots",
    "SlotBatch",
    "collate_slots",
    "encode_with_starts",
    "indentation_columns",
    "layout_code",
    "line_depth",
    "line_kind",
    "line_slots",
    "position_code",
    "program_lines",
    "skeleton",
    "split_lines",
    "starts_by_decoding",
]

KINDS: tuple[str, ...] = (
    "blank", "comment", "decorator", "def", "class", "return", "if", "elif", "else", "for", "while", "try",
    "except", "finally", "with", "import", "raise", "assert", "pass", "break", "continue", "yield",
    "string", "close", "assign", "call", "other",
)
_KIND = {name: index for index, name in enumerate(KINDS)}
# Statement keywords that open a line. "from" is an import; the rest of Python's keywords (lambda, del, global,
# nonlocal, await, not, ...) fall through to assign / call / other.
_KEYWORDS: dict[str, str] = {
    "def": "def", "class": "class", "return": "return", "if": "if", "elif": "elif", "else": "else", "for": "for",
    "while": "while", "try": "try", "except": "except", "finally": "finally", "with": "with", "import": "import",
    "from": "import", "raise": "raise", "assert": "assert", "pass": "pass", "break": "break",
    "continue": "continue", "yield": "yield",
}
MAX_DEPTH: int = 7
TAB_COLUMNS: int = 4
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_STRING = re.compile(r"(?:[rR]|[uU]|[bB]|[fF]|[bB][rR]|[rR][bB]|[fF][rR]|[rR][fF])?['\"]")
_CALL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\s*\.\s*[A-Za-z_][A-Za-z_0-9]*)*\s*\(")


# ---------------------------------------------------------------------------------------------
# Lines: kind and depth
# ---------------------------------------------------------------------------------------------
def split_lines(text: str) -> list[tuple[int, str]]:
    """(character offset, line without its newline) of every physical line. A text that ends in a newline has
    no extra empty last line; the empty text has no line."""
    lines: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        lines.append((offset, line))
        offset += len(line) + 1
    if lines and lines[-1][1] == "" and (text.endswith("\n") or text == ""):
        lines.pop()
    return lines


def indentation_columns(line: str) -> int:
    """Leading whitespace in columns: a space is 1, a tab is ``TAB_COLUMNS`` (4)."""
    columns = 0
    for char in line:
        if char == " ":
            columns += 1
        elif char == "\t":
            columns += TAB_COLUMNS
        else:
            break
    return columns


def line_depth(line: str) -> int:
    """min(7, indentation columns // 4); 0 for a blank line."""
    if not line.strip():
        return 0
    return min(MAX_DEPTH, indentation_columns(line) // 4)


def _has_assignment(stripped: str) -> bool:
    """True if the line holds an assignment operator (=, +=, ..., not ==, !=, <=, >=, :=) outside brackets,
    strings and comments. Brackets are counted within the line only."""
    depth, quote, i, n = 0, "", 0, len(stripped)
    while i < n:
        char = stripped[i]
        if quote:
            if char == "\\":
                i += 2
            elif stripped.startswith(quote, i):
                i += len(quote)
                quote = ""
            else:
                i += 1
            continue
        if char == "#":
            return False
        if char in "\"'":
            quote = stripped[i:i + 3] if stripped[i:i + 3] in ('"""', "'''") else char
            i += len(quote)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "=" and depth == 0:
            if stripped[i + 1:i + 2] == "=":                       # ==
                i += 2
                continue
            before = stripped[i - 1:i]
            if before in ("<", ">"):
                if stripped[i - 2:i - 1] == before:               # <<= and >>=; <= and >= compare
                    return True
            elif before not in ("!", ":"):                        # != compares, := is an expression
                return True
        i += 1
    return False


def line_kind(line: str) -> int:
    """Index into ``KINDS`` of a physical line, from its first token (see the module docstring)."""
    stripped = line.strip()
    if not stripped:
        return _KIND["blank"]
    first = stripped[0]
    if first == "#":
        return _KIND["comment"]
    if first == "@":
        return _KIND["decorator"]
    if first in ")]}":
        return _KIND["close"]
    if _STRING.match(stripped):
        return _KIND["string"]
    word = _WORD.match(stripped)
    if word is not None:
        name = word.group()
        if name == "async":                                      # async def / async for / async with
            following = _WORD.match(stripped[word.end():].lstrip())
            if following is not None and following.group() in _KEYWORDS:
                name = following.group()
        if name in _KEYWORDS:
            return _KIND[_KEYWORDS[name]]
    if _has_assignment(stripped):
        return _KIND["assign"]
    if _CALL.match(stripped):
        return _KIND["call"]
    return _KIND["other"]


def program_lines(text: str) -> list[tuple[int, int]]:
    """(kind, depth) of every physical line of ``text``."""
    return [(line_kind(line), line_depth(line)) for _, line in split_lines(text)]


# ---------------------------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LineSlots:
    """The line slots of one canvas program (at most M)."""

    kind: tuple[int, ...]
    depth: tuple[int, ...]
    start: tuple[int, ...]        # first canvas token of the slot (0-based)
    length: tuple[int, ...]       # canvas tokens of the slot
    lines: int                    # physical lines of the program (before lines beyond M were merged)

    @property
    def count(self) -> int:
        return len(self.kind)


def line_slots(text: str, token_starts: Sequence[int], max_slots: int = 32, max_tokens: int | None = None
               ) -> LineSlots:
    """Line slots of a program from its text and the character offset of each of its tokens.

    Args:
        text: the program on the canvas.
        token_starts: for each canvas token of the program (no <eos>), the offset of its first character in
            ``text``; non-decreasing.
        max_slots: M; lines from the M-th on are merged into slot M - 1 (its kind and depth are line M - 1's).
        max_tokens: if the program has more tokens (a reference longer than the canvas, at evaluation), only
            the first ``max_tokens`` count and the slots that would start after them are dropped.

    The empty program has one slot (blank, no token), so a plan always has a row.
    """
    if max_slots < 1:
        raise ValueError("max_slots must be >= 1")
    starts = list(token_starts)
    if any(b < a for a, b in zip(starts, starts[1:], strict=False)) or (starts and not 0 <= starts[0]) \
            or (starts and starts[-1] > len(text)):
        raise ValueError("token_starts must be non-decreasing offsets into the text")
    truncated = max_tokens is not None and len(starts) > max_tokens
    if truncated:
        starts = starts[:max(0, max_tokens)]
    lines = split_lines(text)
    if not lines:
        lines = [(0, "")]
    offsets = [offset for offset, _ in lines]
    lengths = [0] * len(lines)
    for start in starts:
        lengths[bisect.bisect_right(offsets, start) - 1] += 1
    kinds = [line_kind(line) for _, line in lines]
    depths = [line_depth(line) for _, line in lines]
    total_lines = len(lines)
    if total_lines > max_slots:
        lengths = [*lengths[:max_slots - 1], sum(lengths[max_slots - 1:])]
        kinds, depths = kinds[:max_slots], depths[:max_slots]
    first, running = [], 0
    for length in lengths:
        first.append(running)
        running += length
    if truncated:                                                # drop the slots past the cut (keep one)
        keep = max(1, sum(1 for s in first if s < len(starts)))
        kinds, depths, first, lengths = kinds[:keep], depths[:keep], first[:keep], lengths[:keep]
    return LineSlots(tuple(kinds), tuple(depths), tuple(first), tuple(lengths), total_lines)


@dataclass
class SlotBatch:
    """Line slots of a batch, padded to M. All ``[B, M]``; ``mask`` is True at real slots."""

    kind: Tensor
    depth: Tensor
    start: Tensor
    length: Tensor
    mask: Tensor

    def to(self, device: torch.device | str) -> SlotBatch:
        return SlotBatch(self.kind.to(device), self.depth.to(device), self.start.to(device),
                         self.length.to(device), self.mask.to(device))

    def roll(self, shift: int = 1) -> SlotBatch:
        """Every row gets another row's slots (the "shuffled plan" control)."""
        return SlotBatch(*(getattr(self, name).roll(shift, dims=0)
                           for name in ("kind", "depth", "start", "length", "mask")))


def collate_slots(rows: Sequence[LineSlots], max_slots: int) -> SlotBatch:
    b = len(rows)
    kind = torch.zeros(b, max_slots, dtype=torch.long)
    depth, start, length = torch.zeros_like(kind), torch.zeros_like(kind), torch.zeros_like(kind)
    mask = torch.zeros(b, max_slots, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = row.count
        if not 1 <= n <= max_slots:
            raise ValueError(f"a program has {n} slots; expected 1 .. {max_slots}")
        kind[i, :n] = torch.tensor(row.kind, dtype=torch.long)
        depth[i, :n] = torch.tensor(row.depth, dtype=torch.long)
        start[i, :n] = torch.tensor(row.start, dtype=torch.long)
        length[i, :n] = torch.tensor(row.length, dtype=torch.long)
        mask[i, :n] = True
    return SlotBatch(kind, depth, start, length, mask)


# ---------------------------------------------------------------------------------------------
# Token offsets
# ---------------------------------------------------------------------------------------------
def starts_by_decoding(tokenizer: Any, ids: Sequence[int]) -> list[int]:
    """Character offset of each token as the length of the decoded prefix before it. Slow (one decode per
    token) and independent of the tokenizer's own offsets: the preflight uses it to check them."""
    return [len(tokenizer.decode(list(ids[:i]))) for i in range(len(ids))]


def _in_order(starts: Sequence[int], limit: int) -> list[int]:
    """Offsets clipped to [0, limit] and made non-decreasing. It changes nothing for a tokenizer whose tokens
    follow the text; it keeps one odd token from stopping a training run (the preflight measures how often the
    offsets and an independent computation disagree)."""
    out, running = [], 0
    for start in starts:
        running = max(running, min(max(int(start), 0), limit))
        out.append(running)
    return out


def encode_with_starts(tokenizer: Any, texts: Sequence[str]) -> tuple[list[list[int]], list[list[int]]]:
    """Token ids (as ``tokenizer.encode_batch``: no special tokens) and each token's character offset.

    A Hugging Face fast tokenizer behind ``stage0.data.HFTokenizer`` gives the offsets itself; any other
    tokenizer goes through ``starts_by_decoding``.
    """
    if not texts:
        return [], []
    fast = getattr(tokenizer, "tokenizer", None)
    if fast is not None and getattr(fast, "is_fast", False):
        encoded = fast(list(texts), add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        starts = [[span[0] for span in row] for row in encoded["offset_mapping"]]
    else:
        ids = tokenizer.encode_batch(texts)
        starts = [starts_by_decoding(tokenizer, row) for row in ids]
    return ids, [_in_order(row, len(text)) for row, text in zip(starts, texts, strict=True)]


# ---------------------------------------------------------------------------------------------
# Fixed codes of positions and layouts
# ---------------------------------------------------------------------------------------------
def position_code(x: Tensor, dim: int, max_period: float = 1024.0) -> Tensor:
    """Sinusoidal code of token positions: ``[...]`` -> ``[..., dim]`` float32, no parameters.

    [sin(w_k x)]_k then [cos(w_k x)]_k for k = 0 .. dim/2 - 1, with wavelengths spaced geometrically from 2
    tokens (w_0 = pi: the parity of x) to ``max_period`` tokens: w_k = pi (2 / max_period)^(k / (dim/2 - 1)).
    """
    if dim < 2 or dim % 2:
        raise ValueError("dim must be even and >= 2")
    half = dim // 2
    exponent = torch.arange(half, device=x.device, dtype=torch.float32) / max(half - 1, 1)
    omega = math.pi * torch.pow(torch.tensor(2.0 / max_period, device=x.device), exponent)
    angle = x.to(torch.float32)[..., None] * omega
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def layout_code(start: Tensor, length: Tensor, dim: int) -> Tensor:
    """Fixed code of a slot's layout: ``[...]``, ``[...]`` -> ``[..., dim]``.

    The codes of start (3/8 of the width), length (1/4) and end = start + length (3/8), concatenated. The end
    is a function of the other two, written out because a linear map of sinusoids of start and length cannot
    form it, and a position lies in a slot when start <= position < end.
    """
    if dim % 16:
        raise ValueError("dim must be a multiple of 16")
    edge, middle = 3 * dim // 8, dim // 4
    return torch.cat([position_code(start, edge), position_code(length, middle),
                      position_code(start + length, edge)], dim=-1)


# ---------------------------------------------------------------------------------------------
# Skeletons and the mode codebook (step 3 of the design; measured by the preflight of step 1)
# ---------------------------------------------------------------------------------------------
def skeleton(lines: Iterable[tuple[int, int]], max_lines: int = 16, with_depth: bool = True) -> tuple:
    """The (kind, depth) of a program's non-blank lines, cut at ``max_lines``; kinds alone without depth."""
    blank = _KIND["blank"]
    kept = [(kind, depth) for kind, depth in lines if kind != blank][:max_lines]
    return tuple(kept) if with_depth else tuple(kind for kind, _ in kept)


class Codebook:
    """Mode ids of line skeletons: 0 is "other", 1 .. size - 1 the most frequent skeletons of a sample."""

    def __init__(self, entries: Sequence[tuple], with_depth: bool = True, max_lines: int = 16) -> None:
        self.entries = [tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in entry) for entry in entries]
        self.with_depth, self.max_lines = with_depth, max_lines
        self._index = {entry: i + 1 for i, entry in enumerate(self.entries)}
        if len(self._index) != len(self.entries):
            raise ValueError("duplicate skeletons in a codebook")

    def __len__(self) -> int:
        return len(self.entries) + 1

    @classmethod
    def build(cls, counts: Counter, size: int = 1024, with_depth: bool = True, max_lines: int = 16) -> Codebook:
        """The ``size - 1`` most frequent skeletons (ties: the smaller skeleton first, so the result does not
        depend on the order of the sample)."""
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return cls([entry for entry, _ in ranked[:size - 1]], with_depth, max_lines)

    def mode(self, lines: Iterable[tuple[int, int]]) -> int:
        return self._index.get(skeleton(lines, self.max_lines, self.with_depth), 0)

    def coverage(self, programs: Iterable[Iterable[tuple[int, int]]]) -> float:
        """Share of programs whose skeleton is in the codebook (not "other")."""
        modes = [self.mode(lines) for lines in programs]
        return sum(1 for m in modes if m) / len(modes) if modes else 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"with_depth": self.with_depth, "max_lines": self.max_lines,
                                          "kinds": list(KINDS), "entries": self.entries}))

    @classmethod
    def load(cls, path: str | Path) -> Codebook:
        data = json.loads(Path(path).read_text())
        if data["kinds"] != list(KINDS):
            raise ValueError("the codebook was built with another list of line kinds")
        return cls(data["entries"], data["with_depth"], data["max_lines"])
