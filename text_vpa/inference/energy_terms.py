# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
energy_terms.py -- constraint energies E^i(j | w, Z) >= 0 for the energy-tilted kernel (PR-JEPA Secs. 3.3, 4.3).

The sampler tilts the realizer posterior by exp(-E/T) and clips it (Eq. 17). E = 0 marks an admissible
token, E = inf a forbidden one (hard), a finite E a penalty (soft). Two classes of terms matter for
Proposition 4(d):

  plan-mediated   E^i(j | Z) depends on the token j and the plan only. If every position of w_t is
                  admissible, every position of w_{t+h} is admissible under the fully parallel update.
  draft-dependent E^i(j | w^{-i}, Z) reads other positions of the draft (here: the prefix w_{<i}). Its hard
                  guarantee needs sequential updates or a final repair pass (``ConstraintEnergyEvaluator.repair``).

Terms implemented (Table 1, general-purpose code; Sec. 3.3, "Energy terms for the other domains"):

  SpecialTokenEnergy        plan-mediated, hard. Control tokens ([MASK], [NULL_PLAN], slot delimiters)
                            never appear in an output. ([PAD] ends the output and is allowed.)
  PlanVocabularyEnergy      plan-mediated, hard by default. "Only identifiers the plan declares may be
                            referenced", at token level: a token containing identifier characters is
                            admissible only if the plan licenses it -- the plan's token inventory plus the
                            language's keywords and builtins. Tokens without identifier characters
                            (operators, brackets, digits, whitespace) are always admissible.
  GrammarEnergy             draft-dependent, hard: E_gram = inf for tokens that no completion of the prefix
                            can parse -- a closing bracket that does not match the innermost open bracket
                            outside strings and comments, a token whose own brackets mismatch, or [PAD]
                            (end of output) inside an open bracket or string. Over-approximation: it never
                            rejects a token some completion could parse; tokens containing quote, comment or
                            escape characters are never rejected.
  ArityEnergy               draft-dependent, soft by default: inside a call of a planned function, a token
                            that would give more arguments than the planned signature allows, or close the
                            call with fewer than it requires; inside a planned ``def``, the parameter count
                            must equal the planned one. Calls with * / ** unpacking are not checked.
  TypeSignatureEnergy       draft-dependent, soft by default (Python): inside a parameter annotation of a
                            planned ``def`` and after its parameter list, tokens must continue the planned
                            annotation text ("x: int", "-> List[int]"); whitespace is ignored.

Prefix-dependent terms are causal (they read only w_{<i}), so ``repair`` -- a left-to-right pass that
replaces each inadmissible token by the best admissible one under the last posterior -- makes the final
output satisfy every hard term without any extra network evaluation.

Energies are returned compactly (``EnergyResult``): a position-independent plan part [B, S] and
deduplicated draft rows [U, S] indexed by position [B, N], because a dense [B, N, S] tensor of a
152k-token vocabulary would not fit in memory for long outputs.
"""

from __future__ import annotations

import bisect
import keyword
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from text_vpa.data.ast_parser import CPP_DEFAULT_BUILTINS, PYTHON_DEFAULT_BUILTINS, FunctionSignature

if TYPE_CHECKING:
    from text_vpa.data.ast_parser import ProgramStructure
    from text_vpa.data.tokenization import TextVPATokenizer

__all__ = [
    "AdmissibilityViolation",
    "ArityEnergy",
    "ConstraintEnergyEvaluator",
    "EnergyResult",
    "EnergyTerm",
    "GrammarEnergy",
    "ParallelUpdateReport",
    "PlanConstraints",
    "PlanVocabularyEnergy",
    "PrefixLexer",
    "SpecialTokenEnergy",
    "TypeSignatureEnergy",
    "VocabularyAnalysis",
    "base_licensed_words",
]

OPENERS = "([{"
CLOSERS = ")]}"
MATCH = dict(zip(CLOSERS, OPENERS, strict=True))
_WS = re.compile(r"\s+")

CPP_KEYWORDS = frozenset(
    "alignas alignof and asm auto bool break case catch char class const constexpr continue decltype default "
    "delete do double else enum explicit extern false float for friend goto if inline int long mutable namespace "
    "new noexcept not nullptr operator or private protected public register return short signed sizeof static "
    "static_assert struct switch template this throw true try typedef typeid typename union unsigned using "
    "virtual void volatile while".split()
)


def _norm(text: str) -> str:
    """Remove all whitespace (annotation texts are compared whitespace-insensitively)."""
    return _WS.sub("", text)


def base_licensed_words(language: str) -> frozenset[str]:
    """Words every plan licenses: keywords and builtins of the language."""
    if language == "python":
        return frozenset(keyword.kwlist) | frozenset(keyword.softkwlist) | PYTHON_DEFAULT_BUILTINS
    if language == "cpp":
        return CPP_KEYWORDS | CPP_DEFAULT_BUILTINS
    raise ValueError(f"unsupported language {language!r}")


# =============================================================================================
# Plan constraints (decoded from Z_target or taken from a reference)
# =============================================================================================
@dataclass(frozen=True)
class PlanConstraints:
    """Symbolic content of one sample's plan, the input of every energy term.

    Attributes:
        licensed_tokens: ids of identifier-bearing tokens the plan licenses (None: no vocabulary constraint).
        signatures: planned function signatures by name (arity and annotations): one signature, or a tuple
            when several definitions share a name (methods of different classes, property getter/setter).
        declared_identifiers: names the plan declares (informational; licensing works on tokens).
        source: ``"reference"`` (oracle plan from the target) or ``"readout"`` (decoded from Z_target).
        classes: planned class names (``ClassName.method(obj, ...)`` passes self explicitly).
    """

    licensed_tokens: frozenset[int] | None = None
    signatures: Mapping[str, FunctionSignature | tuple[FunctionSignature, ...]] = field(default_factory=dict)
    declared_identifiers: frozenset[str] = frozenset()
    source: str = "readout"
    classes: frozenset[str] = frozenset()

    @classmethod
    def from_reference(cls, target_text: str, tokenizer: TextVPATokenizer, structure: ProgramStructure,
                       analysis: VocabularyAnalysis) -> PlanConstraints:
        """Oracle plan of a reference program: every identifier-bearing token of its tokenization is
        licensed, so the reference itself is admissible under ``PlanVocabularyEnergy``."""
        ids = {int(i) for i in tokenizer.encode(target_text) if analysis.word_bearing[int(i)]}
        signatures: dict[str, tuple[FunctionSignature, ...]] = {}
        for f in structure.functions:
            signatures[f.name] = signatures.get(f.name, ()) + (f,)
        return cls(
            licensed_tokens=frozenset(ids),
            signatures=signatures,
            declared_identifiers=frozenset(structure.declared_identifiers),
            source="reference",
            classes=frozenset(structure.classes),
        )


# =============================================================================================
# Vocabulary analysis
# =============================================================================================
def _bracket_summary(text: str) -> tuple[str, bool]:
    """(unmatched closers in order, internal mismatch) of a token's brackets on an empty local stack."""
    local: list[str] = []
    closers: list[str] = []
    for ch in text:
        if ch in OPENERS:
            local.append(ch)
        elif ch in CLOSERS:
            if local:
                if local[-1] != MATCH[ch]:
                    return "".join(closers), True
                local.pop()
            else:
                closers.append(ch)
    return "".join(closers), False


def _arity_summary(text: str) -> tuple[bool, int, int, bool, bool, bool]:
    """Effect of a token on the innermost call frame when processed at that frame's depth.

    Returns (first_content, n_commas, mid_content, last_content, closes, star):
      first_content  non-space content before the first depth-0 comma (or close / end)
      n_commas       depth-0 commas before the close (or end)
      mid_content    number of content-bearing segments strictly between depth-0 commas
      last_content   content after the last depth-0 comma (meaningful if n_commas > 0)
      closes         a depth-0 ')' closes the frame inside this token
      star           a depth-0 '*' (unpacking or bare separator): the frame cannot be checked
    """
    depth = 0
    segments: list[bool] = [False]
    closes = False
    star = False
    for ch in text:
        if depth == 0:
            if ch == ")":
                closes = True
                break
            if ch in "]}":
                break
            if ch == ",":
                segments.append(False)
                continue
            if ch == "*":
                star = True
            if not ch.isspace():
                segments[-1] = True
        if ch in OPENERS:
            depth += 1
        elif ch in CLOSERS:
            depth -= 1
    n_commas = len(segments) - 1
    mid = sum(segments[1:-1]) if n_commas >= 2 else 0
    last = segments[-1] if n_commas >= 1 else False
    return segments[0], n_commas, mid, last, closes, star


class VocabularyAnalysis:
    """Per-token tables derived once from a tokenizer's surface strings.

    Attributes (NumPy arrays of length S unless noted):
        word_bearing    token contains an identifier character (a letter or '_')
        opaque          token contains a character that can change the lexical mode (quote, comment,
                        escape); the grammar, arity and type terms never reject it
        special         special token
        closer_class    index into ``closer_suffixes`` of the token's unmatched-closer sequence (0: none)
        closer_suffixes (list) stack suffix each closer class requires, e.g. ")]" -> "[(" (top last)
        mismatch        the token's own brackets mismatch (e.g. "(]")
        arity fields    see ``_arity_summary``
    """

    def __init__(self, tokenizer: TextVPATokenizer, language: str = "python") -> None:
        if language not in ("python", "cpp"):
            raise ValueError(f"unsupported language {language!r}")
        self.language = language
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id
        self.mask_id = tokenizer.mask_id
        self.strings: tuple[str, ...] = tokenizer.token_strings
        s = self.vocab_size
        self.special = np.zeros(s, dtype=bool)
        self.special[list(tokenizer.special_ids)] = True
        opaque_chars = "'\"#\\" if language == "python" else "'\"\\/"
        self.word_bearing = np.zeros(s, dtype=bool)
        self.opaque = np.zeros(s, dtype=bool)
        self.mismatch = np.zeros(s, dtype=bool)
        self.closer_class = np.zeros(s, dtype=np.int64)
        self.first_content = np.zeros(s, dtype=bool)
        self.n_commas = np.zeros(s, dtype=np.int64)
        self.mid_content = np.zeros(s, dtype=np.int64)
        self.last_content = np.zeros(s, dtype=bool)
        self.closes = np.zeros(s, dtype=bool)
        self.star = np.zeros(s, dtype=bool)
        classes: dict[str, int] = {"": 0}
        self.closer_suffixes: list[str] = [""]
        normalized: list[tuple[str, int]] = []
        for i, text in enumerate(self.strings):
            if self.special[i]:
                continue
            self.word_bearing[i] = any(ch.isalpha() or ch == "_" for ch in text)
            self.opaque[i] = any(ch in opaque_chars for ch in text)
            closers, bad = _bracket_summary(text)
            self.mismatch[i] = bad
            if closers not in classes:
                classes[closers] = len(classes)
                self.closer_suffixes.append("".join(MATCH[c] for c in reversed(closers)))
            self.closer_class[i] = classes[closers]
            (self.first_content[i], self.n_commas[i], self.mid_content[i], self.last_content[i],
             self.closes[i], self.star[i]) = _arity_summary(text)
            normalized.append((_norm(text), i))
        normalized.sort()
        self._norm_keys = [k for k, _ in normalized]
        self._norm_ids = [i for _, i in normalized]
        self._by_norm: dict[str, list[int]] = {}
        for k, i in normalized:
            self._by_norm.setdefault(k, []).append(i)
        self.always_licensed = ~self.word_bearing & ~self.special   # no identifier characters

    def ids_with_norm_prefix(self, prefix: str) -> list[int]:
        """Ids whose whitespace-free text starts with ``prefix``."""
        lo = bisect.bisect_left(self._norm_keys, prefix)
        out = []
        for k in range(lo, len(self._norm_keys)):
            if not self._norm_keys[k].startswith(prefix):
                break
            out.append(self._norm_ids[k])
        return out

    def ids_with_norm(self, text: str) -> list[int]:
        return self._by_norm.get(text, [])

    def licensed_ids_for_words(self, tokenizer: TextVPATokenizer, words: Iterable[str]) -> set[int]:
        """Identifier-bearing ids of each word tokenised alone, after a space, and after '.' / '('."""
        out: set[int] = set()
        for word in words:
            for variant in (word, " " + word, "." + word, "(" + word):
                out.update(int(i) for i in tokenizer.encode(variant) if self.word_bearing[int(i)])
        return out


# =============================================================================================
# Prefix lexer (draft-dependent terms)
# =============================================================================================
CODE, STRING, COMMENT, ENDED = 0, 1, 2, 3
_STRING_PREFIXES = frozenset({"", "r", "u", "f", "b", "br", "rb", "fr", "rf"})
_TRIPLES = ("'" * 3, '"' * 3)


@dataclass(frozen=True)
class FrameState:
    """Arity state of a planned call / def frame at the top of the bracket stack."""

    min_args: int
    max_args: int          # -1: unbounded
    segments: int          # completed content-bearing argument segments
    content: bool          # current segment has content


@dataclass(frozen=True)
class AnnotationState:
    """Planned annotation still to be typed: ``required`` (whitespace-free), already typed ``typed``."""

    required: str
    typed: str
    terminators: str


@dataclass(frozen=True)
class PositionState:
    """Lexer state before a position (what the draft-dependent energies of that position read)."""

    mode: int
    stack: str
    frame: FrameState | None
    annotation: AnnotationState | None

    def key(self) -> tuple:
        return (self.mode, self.stack, self.frame, self.annotation)


@dataclass
class _Frame:
    opener: str
    kind: int = 0                       # 0 plain, 1 planned call, 2 planned def
    min_args: int = 0
    max_args: int = -1
    segments: int = 0
    content: bool = False
    disabled: bool = False
    signature: FunctionSignature | None = None
    segment_text: str = ""
    annotation: str | None = None       # required (normalised) annotation while one is being typed
    annotation_text: str = ""


@dataclass
class _Str:
    """An open string literal. f-strings switch to expression mode inside replacement fields (PEP 701)."""

    quote: str
    triple: bool
    fstring: bool
    expr_depth: int = 0                 # > 0: inside a replacement field {...}
    brace_pending: bool = False         # saw '{' in the literal part; '{{' is a literal brace
    close_run: int = 0                  # consecutive closing quotes of a triple-quoted string
    escape: bool = False


def _param_segments(text: str) -> int:
    """Depth-0, comma-separated, non-empty segments of the parameter list in a signature text,
    skipping string literals (defaults such as sep="," or b="(")."""
    start = text.find("(")
    if start < 0:
        return 0
    depth, segments, i, n = 0, [False], start + 1, len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"":
            quote = text[i:i + 3] if text[i:i + 3] in _TRIPLES else ch
            j = i + len(quote)
            while j < n and not text.startswith(quote, j):
                j += 2 if text[j] == "\\" else 1
            segments[-1] = True
            i = j + len(quote)
            continue
        if depth == 0 and ch == ")":
            break
        if ch in OPENERS:
            depth += 1
        elif ch in CLOSERS:
            depth -= 1
        if depth == 0 and ch == ",":
            segments.append(False)
        elif not ch.isspace():
            segments[-1] = True
        i += 1
    return sum(segments)


def _call_bounds(sig: FunctionSignature, drop_self: bool) -> tuple[int, int | None]:
    """(min, max) arguments of a call, positional and keyword together: keyword-only parameters count, and
    any *args / **kwargs makes the maximum unbounded. ``drop_self`` removes a leading self / cls."""
    positional = [p for p in sig.parameters if p.kind == "positional"]
    keyword_only = [p for p in sig.parameters if p.kind == "keyword_only"]
    if drop_self and positional and positional[0].name in ("self", "cls"):
        positional = positional[1:]
    low = sum(not p.has_default for p in positional) + sum(not p.has_default for p in keyword_only)
    variadic = any(p.kind in ("var_positional", "var_keyword") for p in sig.parameters)
    return low, None if variadic else len(positional) + len(keyword_only)


def _as_tuple(value: FunctionSignature | Sequence[FunctionSignature]) -> tuple[FunctionSignature, ...]:
    return (value,) if isinstance(value, FunctionSignature) else tuple(value)


class PrefixLexer:
    """Incremental lexer over a draft: bracket stack, strings (incl. PEP 701 f-strings) and comments,
    planned call and def frames.

    Feed tokens with ``feed``; ``state()`` is the state before the next token. Unmatched or mismatched
    closers in a noisy draft are ignored (the prefix is then unparseable anyway, and any rule is a valid
    over-approximation), so constraints stay informative while the draft is still mostly noise.

    Arity frames: a plain call ``name(`` is checked against the planned non-method functions of that name,
    an attribute call ``obj.name(`` against the planned methods of that name (minus self/cls unless the
    receiver is a planned class, which passes self explicitly), a ``def name(`` against the planned
    parameter counts. With several planned definitions of one name the frame accepts the union of their
    bounds. Frames containing ``lambda`` or a generator ``for`` are not checked (their commas are not
    argument separators), nor calls with * / ** unpacking.
    """

    def __init__(self, analysis: VocabularyAnalysis, plan: PlanConstraints | None = None) -> None:
        self.a = analysis
        self.python = analysis.language == "python"
        signatures = plan.signatures if plan is not None else {}
        self.signatures: dict[str, tuple[FunctionSignature, ...]] = {k: _as_tuple(v) for k, v in signatures.items()}
        self.classes = plan.classes if plan is not None else frozenset()
        self.mode = CODE                    # CODE / COMMENT / ENDED; open strings live on self.strings
        self.stack: list[_Frame] = []
        self.strings: list[_Str] = []
        self.opening: dict | None = None    # an opening quote run not yet resolved (Python triple quotes)
        self.block_comment = False
        self.comment_chars = 0
        self.prev_char = ""
        self.word = ""
        self.word_after_dot = False
        self.last_word = ""
        self.word_before = ""
        self.receiver: str | None = None    # name before the '.' of the last dotted word
        self.dot_receiver = ""
        self.expr_word = ""                 # identifier characters just before a quote inside f-string fields
        self.ret: dict | None = None        # return annotation still expected after a planned def's ')'
        self.errors = 0

    # -----------------------------------------------------------------------------------------
    def state(self) -> PositionState:
        if self.mode == ENDED:
            return PositionState(ENDED, "", None, None)
        stack = "".join(f.opener for f in self.stack)
        if self.mode == COMMENT:
            return PositionState(COMMENT, stack, None, None)
        if self.strings or (self.opening is not None and self.opening["count"] == 1):
            return PositionState(STRING, stack, None, None)
        frame_state = None
        if self.stack:
            top = self.stack[-1]
            if top.kind and not top.disabled:
                frame_state = FrameState(top.min_args, top.max_args, top.segments, top.content)
        annotation = None
        for fr in reversed(self.stack):
            if fr.kind == 2:
                if fr.annotation is not None:
                    annotation = AnnotationState(fr.annotation, _norm(fr.annotation_text), ",)=")
                break
        if annotation is None and self.ret is not None:
            annotation = AnnotationState(self.ret["required"], _norm(self.ret["typed"]), ":")
        return PositionState(CODE, stack, frame_state, annotation)

    def feed(self, token: int) -> None:
        if self.mode == ENDED:
            return
        if token == self.a.pad_id:
            self.mode = ENDED
            return
        if self.a.special[token]:
            return                          # noise / source tokens carry no characters
        for ch in self.a.strings[token]:
            self._char(ch)

    # -----------------------------------------------------------------------------------------
    def _char(self, ch: str) -> None:
        if self.mode == COMMENT:
            self.comment_chars += 1
            if self.block_comment:
                if self.prev_char == "*" and ch == "/" and self.comment_chars > 1:
                    self.mode = CODE
            elif ch == "\n":
                self.mode = CODE
        elif self.opening is not None:
            self._resolve_opening(ch)
        elif self.strings:
            self._in_string(ch)
        else:
            self._code_char(ch)
        self.prev_char = ch

    def _start_string(self, quote: str, prefix: str) -> None:
        """A quote opens a string; Python resolves single vs triple quotes over the next characters."""
        fstring = self.python and "f" in prefix.lower() and prefix.lower() in _STRING_PREFIXES
        if self.python:
            self.opening = {"quote": quote, "count": 1, "fstring": fstring}
        else:
            self.strings.append(_Str(quote, False, False))

    def _resolve_opening(self, ch: str) -> None:
        o = self.opening
        if o is None:  # pragma: no cover - guarded by the caller
            return
        if ch == o["quote"] and o["count"] < 3:
            o["count"] += 1
            self._string_text(ch)
            if o["count"] == 3:
                self.strings.append(_Str(o["quote"], True, o["fstring"]))
                self.opening = None
            return
        self.opening = None
        if o["count"] == 2:                 # "" -- an empty string; the character belongs to the enclosing context
            if self.strings:
                self._in_string(ch)
            else:
                self._code_char(ch)
            return
        self.strings.append(_Str(o["quote"], False, o["fstring"]))
        self._in_string(ch)

    def _in_string(self, ch: str) -> None:
        top = self.strings[-1]
        if top.expr_depth > 0:              # inside an f-string replacement field: expression characters
            if ch in "'\"":
                prefix, self.expr_word = self.expr_word, ""
                self._start_string(ch, prefix if len(prefix) <= 2 else "")
            elif ch == "{":
                top.expr_depth += 1
                self.expr_word = ""
            elif ch == "}":
                top.expr_depth -= 1
                self.expr_word = ""
            elif ch.isalnum() or ch == "_":
                self.expr_word += ch
            else:
                self.expr_word = ""
            return
        self._string_text(ch)
        if top.escape:
            top.escape = False
            return
        if top.fstring and top.brace_pending:
            top.brace_pending = False
            if ch == "{":
                return                       # '{{': a literal brace
            top.expr_depth = 1               # a replacement field starts; ch is its first character
            self.expr_word = ""
            self._in_string(ch)
            return
        if ch == "\\":
            top.escape = True
            return
        if top.fstring and ch == "{":
            top.brace_pending = True
            return
        if top.triple:
            if ch == top.quote:
                top.close_run += 1
                if top.close_run == 3:
                    self.strings.pop()
            else:
                top.close_run = 0
            return
        if ch == top.quote:
            self.strings.pop()
        elif ch == "\n":                     # unterminated single-line string: recover
            self.strings.pop()
            self.errors += 1

    def _string_text(self, ch: str) -> None:
        """Characters of a string literal also belong to an annotation being typed (e.g. 'Node')."""
        if len(self.strings) > 1 or (self.strings and self.strings[-1].expr_depth > 0):
            return                           # nested inside an f-string field: not annotation text
        for fr in reversed(self.stack):
            if fr.kind == 2:
                if fr.annotation is not None:
                    fr.annotation_text += ch
                break
        self._track_return(ch)

    # -----------------------------------------------------------------------------------------
    def _code_char(self, ch: str) -> None:
        self._track_return(ch)
        if self.python and ch == "#":
            self._end_word()
            self.mode, self.block_comment, self.comment_chars = COMMENT, False, 0
            return
        if not self.python and ch in "/*" and self.prev_char == "/":
            self.mode, self.block_comment, self.comment_chars = COMMENT, ch == "*", 0
            return
        if ch in "'\"":
            prefix = self.word if len(self.word) <= 2 else ""
            self._end_word()
            self._annotate(ch)
            self._content(ch)
            self._start_string(ch, prefix)
            return
        if ch.isalnum() or ch == "_":
            if not self.word:
                self.word_after_dot = self.prev_char == "."
            self.word += ch
            self._annotate(ch)
            self._content(ch)
            return
        self._end_word()
        if ch in OPENERS:
            self._annotate(ch)
            self._content(ch)
            self._open(ch)
        elif ch in CLOSERS:
            self._close(ch)
        elif ch == ",":
            self._comma()
        else:
            self._other(ch)

    def _end_word(self) -> None:
        if not self.word:
            return
        self.receiver = self.dot_receiver if self.word_after_dot else None
        self.word_before = "." if self.word_after_dot else self.last_word
        self.last_word, self.word = self.word, ""
        if self.python and self.last_word in ("lambda", "for") and self.stack and self.stack[-1].kind:
            self.stack[-1].disabled = True   # lambda parameters / generator targets: commas are not arguments

    def _content(self, ch: str) -> None:
        """Mark argument content in the innermost frame."""
        if self.stack and not ch.isspace():
            top = self.stack[-1]
            if top.kind:
                if ch == "*" and not top.content and top.kind == 1:
                    top.disabled = True      # *args / **kwargs unpacking: argument count unknown
                top.content = True

    def _annotate(self, ch: str) -> None:
        """Append to the parameter text / active annotation of the nearest enclosing planned def."""
        for fr in reversed(self.stack):
            if fr.kind == 2:
                if fr is self.stack[-1]:
                    fr.segment_text += ch
                if fr.annotation is not None:
                    fr.annotation_text += ch
                return

    def _callee(self) -> str | None:
        if not self.last_word or self.prev_char not in _CALLEE_PREV or self.last_word[0].isdigit():
            return None
        reserved = keyword.kwlist if self.python else CPP_KEYWORDS
        return None if self.last_word in reserved else self.last_word

    def _open(self, ch: str) -> None:
        frame = _Frame(ch)
        name = self._callee() if ch == "(" else None
        candidates = self.signatures.get(name, ()) if name is not None else ()
        if candidates:
            if self.python and self.word_before == "def":
                counts = [_param_segments(sig.text) for sig in candidates]
                # Several definitions of one name: accept any of their counts, enforce no annotation.
                frame = _Frame(ch, kind=2, min_args=min(counts), max_args=max(counts),
                               signature=candidates[0] if len(candidates) == 1 else None)
            elif self.word_before != "class":
                attribute = self.word_before == "."
                matching = [sig for sig in candidates if sig.is_method == attribute]
                constructor = not attribute and name in self.classes
                if matching and not constructor:
                    drop_self = attribute and self.receiver not in self.classes
                    bounds = [_call_bounds(sig, drop_self) for sig in matching]
                    low = min(b[0] for b in bounds)
                    highs = [b[1] for b in bounds]
                    high = -1 if any(h is None for h in highs) else max(h for h in highs if h is not None)
                    frame = _Frame(ch, kind=1, min_args=low, max_args=high)
        self.last_word = ""
        self.stack.append(frame)

    def _close(self, ch: str) -> None:
        if not self.stack or self.stack[-1].opener != MATCH[ch]:
            self.errors += 1                 # unmatched / mismatched closer in a noisy draft: ignore it
            return
        if self.stack[-1].kind != 2:
            self._annotate(ch)               # e.g. the ']' of "List[int]" inside an annotation
        frame = self.stack.pop()
        if frame.kind == 2 and frame.signature is not None and frame.signature.return_annotation:
            self.ret = {"required": "->" + _norm(frame.signature.return_annotation), "typed": "",
                        "depth": len(self.stack)}
        self._content(ch)
        self.last_word = ""

    def _comma(self) -> None:
        top = self.stack[-1] if self.stack else None
        if top is not None and top.kind:
            if top.content:
                top.segments += 1
            top.content = False
            if top.kind == 2:
                top.annotation, top.annotation_text, top.segment_text = None, "", ""
        else:
            self._annotate(",")               # a comma nested inside an annotation, e.g. Dict[str, int]
        self.last_word = ""

    def _other(self, ch: str) -> None:
        top = self.stack[-1] if self.stack else None
        if top is not None and top.kind == 2 and self.python:
            if ch == ":" and top.annotation is None:
                name = _norm(top.segment_text).lstrip("*")
                param = None
                if top.signature is not None:
                    param = next((p for p in top.signature.parameters if p.name == name), None)
                if param is not None and param.annotation:
                    top.annotation, top.annotation_text = _norm(param.annotation), ""
                top.content = True
                return
            if ch == "=" and top.annotation is not None:
                top.annotation = None         # a default value follows the annotation
        if ch == ".":
            self.dot_receiver = self.last_word
        self._annotate(ch)
        self._content(ch)
        if not ch.isspace():
            self.last_word = ""

    def _track_return(self, ch: str) -> None:
        """Follow the text after a planned def's ')' until its ':' (whitespace-insensitive)."""
        if self.ret is None:
            return
        if (ch == ":" and not self.strings and len(self.stack) == self.ret["depth"]
                and self.ret["required"] == _norm(self.ret["typed"])):
            self.ret = None
            return
        typed = _norm(self.ret["typed"] + ch)
        if not self.ret["required"].startswith(typed):
            self.ret = None                   # the draft left the planned annotation: nothing sound to enforce
            return
        self.ret["typed"] += ch


_CALLEE_PREV = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ \t")


# =============================================================================================
# Energy terms
# =============================================================================================
class EnergyTerm:
    """Base class. ``weight`` = inf makes the term hard."""

    name: str = "energy"
    plan_mediated: bool = True

    def __init__(self, weight: float = math.inf) -> None:
        if not weight > 0.0:
            raise ValueError("weight must be positive (use math.inf for a hard term)")
        self.weight = float(weight)

    @property
    def hard(self) -> bool:
        return math.isinf(self.weight)

    def plan_energy(self, analysis: VocabularyAnalysis, plan: PlanConstraints) -> np.ndarray | None:
        """Position-independent energies [S] (plan-mediated terms)."""
        return None

    def row_energy(self, analysis: VocabularyAnalysis, state: PositionState) -> np.ndarray | None:
        """Energies [S] at a position with lexer ``state`` (draft-dependent terms); None = all zero."""
        return None


class SpecialTokenEnergy(EnergyTerm):
    name = "special_tokens"
    plan_mediated = True

    def plan_energy(self, analysis: VocabularyAnalysis, plan: PlanConstraints) -> np.ndarray:
        e = np.zeros(analysis.vocab_size, dtype=np.float64)
        e[analysis.special] = self.weight
        e[analysis.pad_id] = 0.0
        return e


class PlanVocabularyEnergy(EnergyTerm):
    """Token-level identifier licensing (Table 1: "only identifiers the plan declares may be referenced")."""

    name = "plan_vocabulary"
    plan_mediated = True

    def __init__(self, base_licensed: Iterable[int] = (), weight: float = math.inf) -> None:
        super().__init__(weight)
        self.base_licensed = frozenset(int(i) for i in base_licensed)

    def plan_energy(self, analysis: VocabularyAnalysis, plan: PlanConstraints) -> np.ndarray | None:
        if plan.licensed_tokens is None:
            return None
        licensed = np.zeros(analysis.vocab_size, dtype=bool)
        ids = list(plan.licensed_tokens | self.base_licensed)
        if ids:
            licensed[ids] = True
        e = np.zeros(analysis.vocab_size, dtype=np.float64)
        e[analysis.word_bearing & ~licensed] = self.weight
        return e


class GrammarEnergy(EnergyTerm):
    name = "grammar"
    plan_mediated = False

    def row_energy(self, analysis: VocabularyAnalysis, state: PositionState) -> np.ndarray | None:
        if state.mode == ENDED:
            return None
        e = np.zeros(analysis.vocab_size, dtype=np.float64)
        if state.mode == CODE:
            valid_class = np.array([state.stack.endswith(sfx) for sfx in analysis.closer_suffixes], dtype=bool)
            bad = (~valid_class[analysis.closer_class] | analysis.mismatch) & ~analysis.opaque & ~analysis.special
            e[bad] = self.weight
        # The output ends at [PAD]: impossible inside an open bracket or an unterminated string.
        if state.stack or state.mode == STRING:
            e[analysis.pad_id] = self.weight
        return e if e.any() else None


class ArityEnergy(EnergyTerm):
    name = "arity"
    plan_mediated = False

    def __init__(self, weight: float = 5.0) -> None:
        super().__init__(weight)

    def row_energy(self, analysis: VocabularyAnalysis, state: PositionState) -> np.ndarray | None:
        f = state.frame
        if f is None or state.mode != CODE:
            return None
        a = analysis
        cur = np.where(a.n_commas > 0, a.last_content, f.content | a.first_content)
        done = f.segments + np.where(a.n_commas > 0, (f.content | a.first_content).astype(np.int64) + a.mid_content, 0)
        total = done + cur.astype(np.int64)
        base = f.segments + int(f.content)           # arguments already present before this token
        # Penalise only tokens that add an argument beyond the maximum (an already-violating noisy prefix
        # must not penalise every token alike), and closing tokens below the minimum.
        too_many = ((total > f.max_args) & (total > base)) if f.max_args >= 0 else np.zeros_like(cur)
        too_few = a.closes & (total < f.min_args)
        bad = (too_many | too_few) & ~a.opaque & ~a.star & ~a.special
        if not bad.any():
            return None
        e = np.zeros(a.vocab_size, dtype=np.float64)
        e[bad] = self.weight
        return e


class TypeSignatureEnergy(EnergyTerm):
    name = "type_signature"
    plan_mediated = False

    def __init__(self, weight: float = 5.0) -> None:
        super().__init__(weight)

    def row_energy(self, analysis: VocabularyAnalysis, state: PositionState) -> np.ndarray | None:
        ann = state.annotation
        if ann is None or state.mode != CODE or analysis.language != "python":
            return None
        if not ann.required.startswith(ann.typed):
            return None                  # the draft already deviates: nothing sound to enforce
        rest = ann.required[len(ann.typed):]
        ok: set[int] = set()
        for k in range(len(rest) + 1):           # tokens continuing the annotation (incl. whitespace-only)
            ok.update(analysis.ids_with_norm(rest[:k]))
        for i in analysis.ids_with_norm_prefix(rest):  # tokens completing it and then terminating it
            s = _norm(analysis.strings[i])
            if len(s) > len(rest) and s[len(rest)] in ann.terminators:
                ok.add(i)
        e = np.full(analysis.vocab_size, self.weight, dtype=np.float64)
        e[list(ok)] = 0.0
        e[analysis.opaque | analysis.special] = 0.0
        return e


# =============================================================================================
# Evaluator
# =============================================================================================
@dataclass(frozen=True)
class EnergyResult:
    """E^i(j) = plan[b, j] + rows[row_index[b, i], j]; row 0 is all zeros."""

    plan: Tensor        # [B, S] float32, may contain inf
    rows: Tensor        # [U, S] float32
    row_index: Tensor   # [B, N] int64

    def dense(self) -> Tensor:
        """[B, N, S] (small problems / tests only)."""
        return self.plan[:, None, :] + self.rows[self.row_index]

    def multiplier(self, temperature: float) -> Tensor:
        """exp(-E/T) as [B, N, S] (0 where E = inf)."""
        return torch.exp(-self.dense() / temperature)


class AdmissibilityViolation(AssertionError):
    """Proposition 4(d) failed: an admissible position became inadmissible under a plan-mediated hard term."""


@dataclass(frozen=True)
class ParallelUpdateReport:
    admissible_before: Tensor   # [B, N] bool
    admissible_after: Tensor    # [B, N] bool
    premise: Tensor             # [B] bool: every position admissible before
    conclusion: Tensor          # [B] bool: every position admissible after


class ConstraintEnergyEvaluator:
    """Evaluates all energy terms for a batch of drafts and plans.

    Args:
        tokenizer: the shared tokenizer (its token strings drive every term).
        terms: energy terms; ``default_terms`` builds the standard set.
        language: ``"python"`` or ``"cpp"`` (lexing rules, keywords, builtins).
    """

    def __init__(self, tokenizer: TextVPATokenizer, terms: Sequence[EnergyTerm] | None = None,
                 language: str = "python") -> None:
        self.tokenizer = tokenizer
        self.analysis = VocabularyAnalysis(tokenizer, language)
        self.language = language
        self.terms: list[EnergyTerm] = list(terms) if terms is not None else self.default_terms()

    def default_terms(self, identifier_weight: float = math.inf, grammar_weight: float = math.inf,
                      arity_weight: float = 5.0, type_weight: float = 5.0) -> list[EnergyTerm]:
        base = self.analysis.licensed_ids_for_words(self.tokenizer, base_licensed_words(self.language))
        return [
            SpecialTokenEnergy(),
            PlanVocabularyEnergy(base, weight=identifier_weight),
            GrammarEnergy(weight=grammar_weight),
            ArityEnergy(weight=arity_weight),
            TypeSignatureEnergy(weight=type_weight),
        ]

    @property
    def plan_terms(self) -> list[EnergyTerm]:
        return [t for t in self.terms if t.plan_mediated]

    @property
    def draft_terms(self) -> list[EnergyTerm]:
        return [t for t in self.terms if not t.plan_mediated]

    # -----------------------------------------------------------------------------------------
    def plan_energies(self, plans: Sequence[PlanConstraints], hard_only: bool = False) -> np.ndarray:
        """Position-independent energies [B, S] of the plan-mediated terms."""
        out = np.zeros((len(plans), self.analysis.vocab_size), dtype=np.float64)
        for b, plan in enumerate(plans):
            for term in self.plan_terms:
                if hard_only and not term.hard:
                    continue
                e = term.plan_energy(self.analysis, plan)
                if e is not None:
                    out[b] += e
        return out

    def plan_admissible(self, plans: Sequence[PlanConstraints]) -> Tensor:
        """A(Z): tokens with zero hard plan-mediated energy, [B, S] bool (position-independent)."""
        return torch.from_numpy(np.isfinite(self.plan_energies(plans, hard_only=True)))

    def lexer_states(self, tokens: Sequence[int], plan: PlanConstraints) -> list[PositionState]:
        lexer = PrefixLexer(self.analysis, plan)
        states = []
        for token in tokens:
            states.append(lexer.state())
            lexer.feed(int(token))
        return states

    def __call__(self, drafts: Tensor, plans: Sequence[PlanConstraints]) -> EnergyResult:
        """Energies for drafts [B, N] (int64) under ``plans`` (one per sample)."""
        if drafts.dim() != 2 or len(plans) != drafts.shape[0]:
            raise ValueError(f"expected drafts [B, N] and B plans, got {tuple(drafts.shape)} and {len(plans)}")
        plan_part = self.plan_energies(plans)
        rows: list[np.ndarray] = [np.zeros(self.analysis.vocab_size)]
        row_index = np.zeros(tuple(drafts.shape), dtype=np.int64)
        if self.draft_terms:
            cache: dict[tuple, int] = {}
            draft_list = drafts.detach().cpu().tolist()
            for b, plan in enumerate(plans):
                for i, state in enumerate(self.lexer_states(draft_list[b], plan)):
                    key = state.key()
                    if key not in cache:
                        acc = None
                        for term in self.draft_terms:
                            e = term.row_energy(self.analysis, state)
                            if e is not None:
                                acc = e if acc is None else acc + e
                        if acc is None:
                            cache[key] = 0
                        else:
                            cache[key] = len(rows)
                            rows.append(acc)
                    row_index[b, i] = cache[key]
        return EnergyResult(
            plan=torch.from_numpy(plan_part).float(),
            rows=torch.from_numpy(np.stack(rows)).float(),
            row_index=torch.from_numpy(row_index),
        )

    # -----------------------------------------------------------------------------------------
    def positions_admissible(self, tokens: Tensor, plans: Sequence[PlanConstraints],
                             source_token: int | None = None, allowed: Tensor | None = None) -> Tensor:
        """[B, N] bool: token admissible under the hard plan-mediated terms (source tokens count as admissible).

        ``allowed`` is ``plan_admissible(plans)``, if already computed.
        """
        if allowed is None:
            allowed = self.plan_admissible(plans)
        ok = allowed.to(tokens.device).gather(1, tokens)                            # [B, N]
        if source_token is not None:
            ok = ok | (tokens == source_token)
        return ok

    def check_parallel_update(self, before: Tensor, after: Tensor, plans: Sequence[PlanConstraints],
                              source_token: int | None = None, strict: bool = True,
                              allowed: Tensor | None = None) -> ParallelUpdateReport:
        """Proposition 4(d), per position: an admissible position stays admissible after the parallel update.

        (The statement holds per position, which implies the paper's "every position" form.)
        """
        if allowed is None:
            allowed = self.plan_admissible(plans)
        adm_before = self.positions_admissible(before, plans, source_token, allowed)
        adm_after = self.positions_admissible(after, plans, source_token, allowed)
        broken = adm_before & ~adm_after
        report = ParallelUpdateReport(adm_before, adm_after, adm_before.all(dim=1), adm_after.all(dim=1))
        if strict and bool(broken.any()):
            b, i = (int(x) for x in broken.nonzero()[0])
            raise AdmissibilityViolation(
                f"Prop. 4(d) violated at sample {b}, position {i}: token {int(before[b, i])} -> {int(after[b, i])}"
            )
        return report

    def hard_violations(self, tokens: Tensor, plans: Sequence[PlanConstraints]) -> Tensor:
        """[B, N] bool: output positions (up to and including the first [PAD]) whose token has infinite
        energy given its own prefix, under all hard terms."""
        tok = tokens.detach().cpu()
        energies = self(tok, plans)
        e_tok = energies.plan.gather(1, tok) + energies.rows[energies.row_index, tok]   # [B, N], no dense tensor
        in_output = (tok == self.analysis.pad_id).long().cumsum(dim=1) - (tok == self.analysis.pad_id).long() == 0
        return torch.isinf(e_tok) & in_output

    def repair(self, tokens: Tensor, scores: Tensor, plans: Sequence[PlanConstraints],
               source_token: int | None = None, unrealized: Tensor | None = None,
               temperature: float = 1.0) -> tuple[Tensor, int]:
        """Network-free left-to-right repair (the final repair pass of Sec. 3.3).

        Positions up to the output's first [PAD] are visited in order with the energies of every term given the
        REPAIRED prefix, E^i(j | w_{<i}, Z). A position is replaced when its token has infinite energy, when it
        holds the source token (a leftover [MASK]), or when it never left its source token (``unrealized``: a
        stall, Prop. 4(b) -- rejected jump mass is probability of staying, so a stalled position still holds
        noise from p_0). The replacement is argmax_j scores^i(j) e^{-E^i(j)/T} over the tokens of finite
        energy: the tilt of Eq. 17 applied sequentially, with exact instead of stale draft-dependent energies.
        Returns (tokens, number of positions with a hard violation and no admissible token).

        Args:
            tokens: [B, N] int64 final draft.
            scores: [B, N, S] non-negative scores ranking replacements: the last posterior p_{1|t} (untilted;
                the tilt is recomputed here on the repaired prefix).
            plans: one ``PlanConstraints`` per sample.
            source_token: a leftover source token (e.g. [MASK]) is always replaced.
            unrealized: [B, N] bool, positions that never jumped during the K steps (uniform source).
            temperature: T of the tilt that ranks replacements.
        """
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        out = tokens.detach().cpu().clone()
        sc = scores.detach().double().cpu()
        expected = (*out.shape, self.analysis.vocab_size)
        if tuple(sc.shape) != expected:
            raise ValueError(f"expected scores of shape {expected}, got {tuple(sc.shape)}")
        stalled = None if unrealized is None else unrealized.detach().cpu().bool()
        if stalled is not None and stalled.shape != out.shape:
            raise ValueError(f"expected unrealized of shape {tuple(out.shape)}, got {tuple(stalled.shape)}")
        plan_part = torch.from_numpy(self.plan_energies(plans))                       # [B, S] float64, all terms
        unrepaired = 0
        for b, plan in enumerate(plans):
            lexer = PrefixLexer(self.analysis, plan)
            rows: dict[tuple, Tensor] = {}
            for i in range(out.shape[1]):
                state = lexer.state()
                key = state.key()
                if key not in rows:
                    energy = plan_part[b].clone()
                    for term in self.draft_terms:
                        e = term.row_energy(self.analysis, state)
                        if e is not None:
                            energy += torch.from_numpy(e)
                    rows[key] = energy
                energy = rows[key]                                                    # E^i(. | repaired prefix)
                forbidden = torch.isinf(energy)
                if source_token is not None:
                    forbidden[source_token] = True   # a leftover source token is always replaced
                token = int(out[b, i])
                if forbidden[token] or (stalled is not None and bool(stalled[b, i])):
                    if bool((~forbidden).any()):
                        ranked = (sc[b, i].clamp_min(0.0) * torch.exp(-energy / temperature)).masked_fill(
                            forbidden, -math.inf)
                        token = int(ranked.argmax())
                        out[b, i] = token
                    elif forbidden[token]:
                        unrepaired += 1
                lexer.feed(token)
                if lexer.mode == ENDED:
                    break
        return out.to(tokens.device), unrepaired
