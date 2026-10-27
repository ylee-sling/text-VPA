# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Independent tests of the constraint energies (``text_vpa.inference.energy_terms``).

Ground truth: PR-JEPA Sec. 3.3 ("Energy barrier and safety clipping", "Energy terms for HLS-C / for the other
domains"), Table 1, Proposition 4(d) and the module, class and method docstrings:

  * E = 0 marks an admissible token, E = inf a forbidden one (hard), a finite E a penalty (soft).
  * Grammar: E_gram = inf for tokens that no completion of the draft can parse; "over-approximation is safe:
    the check only needs never to reject a valid token". Hence every token of a valid program has zero
    grammar energy given its true prefix.
  * Plan-mediated terms depend on (j, Z) only and survive the fully parallel update (Prop. 4(d));
    draft-dependent terms read the prefix w_{<i} only and need sequential updates or a repair pass.
  * The oracle plan of a reference program is consistent with that program, so (Table 1: "only identifiers
    the plan declares", "argument counts match planned signatures") every token of its own tokenization has
    zero energy under every term.

Expected values are written by hand or computed by small oracles in this file from those statements; the
function under test is never used to build its own reference. ``ListTokenizer`` is an explicit-vocabulary
tokenizer, so tokens such as "(]", ", x" or "a, b)" (which a GPT-2-style BPE never forms, because its
pre-tokenizer splits punctuation from letters) can be analysed by hand.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import keyword
import math
import string
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import torch

from text_vpa.data.ast_parser import TreeSitterASTExtractor
from text_vpa.data.tokenization import SPECIAL_TOKENS, ByteTokenizer, TextVPATokenizer, train_bpe_tokenizer
from text_vpa.inference.energy_terms import (
    AdmissibilityViolation,
    ArityEnergy,
    ConstraintEnergyEvaluator,
    EnergyResult,
    EnergyTerm,
    GrammarEnergy,
    PlanConstraints,
    PlanVocabularyEnergy,
    SpecialTokenEnergy,
    TypeSignatureEnergy,
    VocabularyAnalysis,
    base_licensed_words,
)
from text_vpa.models.realizers.kernels import (
    kernel_distribution,
    repair_aware_multiplier,
    sample_jump,
    tilted_jump_kernel,
)

INF = math.inf
OPENERS, CLOSERS = "([{", ")]}"
PAIR = {")": "(", "]": "[", "}": "{"}


# =============================================================================================
# Tokenizers and fixtures
# =============================================================================================
class ListTokenizer(TextVPATokenizer):
    """Explicit vocabulary (tests only): ``pieces`` followed by the special tokens; greedy longest-match encode."""

    def __init__(self, pieces: Sequence[str]) -> None:
        if len(set(pieces)) != len(pieces) or "" in pieces:
            raise ValueError("pieces must be distinct and non-empty")
        self.pieces = tuple(pieces)
        self.index = {p: i for i, p in enumerate(self.pieces)}
        self.longest = max(map(len, self.pieces))

    @property
    def vocab_size(self) -> int:
        return len(self.pieces) + len(SPECIAL_TOKENS)

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for n in range(min(self.longest, len(text) - i), 0, -1):
                j = self.index.get(text[i:i + n])
                if j is not None:
                    out.append(j)
                    i += n
                    break
            else:
                raise ValueError(f"no piece for {text[i]!r}")
        return out

    def _decode_ids(self, ids: Sequence[int]) -> str:
        return "".join(self.pieces[int(i)] for i in ids)

    def _token_string(self, token_id: int) -> str:
        return self.pieces[token_id]

    def special_token_id(self, token: str) -> int:
        try:
            return len(self.pieces) + SPECIAL_TOKENS.index(token)
        except ValueError as exc:
            raise KeyError(token) from exc

    def spec(self) -> dict[str, Any]:
        return {"type": "list", "pieces": list(self.pieces)}


MULTI_PIECES = [
    # brackets
    "])", ")]", "(]", "[)", "()", ")(", "]))", ")]}", "]]", "],",
    # opaque (quote / comment / escape) tokens that contain brackets
    "')'", "#)", "]#", "\\)",
    # arity
    ", x", "),", "a, b)", "x, y, z", "a, , b", "g(a, b)", "*args", "**kw", "(x", "x)", "z)",
    # words
    "def", "class", "self", "foo", "bar", "return", "lambda", "None",
    # annotations
    "int", " int", "int)", "int,", "int=", "int]", "int):", "List", "Dict", "str", "float", "teger",
    ", int]", ",int", " ->", "->", "-> int", "-> int:", "-> str", "):",
    # whitespace runs
    "    ", "\n    ",
]
CHARS = [chr(c) for c in range(32, 127)] + ["\n", "\t"]

BPE_SNIPPETS = [
    "def f(x: int, y: int) -> int:\n    return g(x, y)\n",
    "def g(a, b=1):\n    return a[b])\n",
    "xs = [f(x) for x in range(10)]\nys = xs[0](x)\n",
    "def h(x: List[int]) -> Dict[str, int]:\n    return {str(x[0]): int(x[1])}\n",
    "print(f(1, 2), g(3), h([1, 2]))\n",
    "a = b[c[d]]\nz = q(x, y), r(x, y)\n",
]


@pytest.fixture(scope="module")
def list_tok() -> ListTokenizer:
    return ListTokenizer(CHARS + MULTI_PIECES)


@pytest.fixture(scope="module")
def byte_tok() -> ByteTokenizer:
    return ByteTokenizer()


@pytest.fixture(scope="module")
def bpe_tok() -> TextVPATokenizer:
    tok = train_bpe_tokenizer(BPE_SNIPPETS * 4, vocab_size=400)
    for piece in ("])", "]]", "),", " int", "->", "):"):
        assert piece in tok.token_strings, f"the small BPE is expected to contain {piece!r}"
    return tok


@pytest.fixture(scope="module")
def extractor() -> TreeSitterASTExtractor:
    return TreeSitterASTExtractor("python")


# =============================================================================================
# Helpers and oracles (written from the docstrings, independent of the implementation)
# =============================================================================================
def tid(tok: TextVPATokenizer, piece: str) -> int:
    """Id of a special token name or of the token whose surface string is exactly ``piece``."""
    if piece in SPECIAL_TOKENS:
        return tok.special_token_id(piece)
    return tok.token_strings.index(piece)


def ids_of(tok: TextVPATokenizer, pieces: Sequence[str]) -> list[int]:
    return [tid(tok, p) for p in pieces]


def energy_row(ev: ConstraintEnergyEvaluator, pieces: Sequence[str], plan: PlanConstraints | None = None) -> np.ndarray:
    """E(.) [S] at the position right after the prefix ``pieces`` (one token each)."""
    ids = ids_of(ev.tokenizer, pieces)
    draft = torch.tensor([ids + [ev.tokenizer.pad_id]])
    result = ev(draft, [plan if plan is not None else PlanConstraints()])
    return result.dense()[0, len(ids)].double().numpy()


def assert_energies(row: np.ndarray, tok: TextVPATokenizer, expected: dict[str, float]) -> None:
    got = {piece: float(row[tid(tok, piece)]) for piece in expected}
    assert got == expected


def word_bearing(text: str) -> bool:
    """An identifier character: a letter or '_'."""
    return any(ch.isalpha() or ch == "_" for ch in text)


def opaque(text: str, language: str = "python") -> bool:
    """A character that can change the lexical mode: quote, comment or escape."""
    return any(ch in ("'\"#\\" if language == "python" else "'\"\\/") for ch in text)


def no_ws(text: str) -> str:
    return "".join(text.split())


def closer_oracle(text: str) -> tuple[str | None, bool]:
    """(stack suffix the token's unmatched closers require, bottom -> top; own-bracket mismatch)."""
    local: list[str] = []
    needed: list[str] = []
    for ch in text:
        if ch in OPENERS:
            local.append(ch)
        elif ch in CLOSERS:
            if not local:
                needed.append(PAIR[ch])                 # closes the innermost open bracket of the prefix
            elif local.pop() != PAIR[ch]:
                return None, True
    return "".join(reversed(needed)), False


def arity_oracle(text: str) -> tuple[bool, int, int, bool, bool, bool]:
    """(first_content, n_commas, mid_content, last_content, closes, star) of ``_arity_summary``'s docstring: the
    token is read at the frame's depth up to a depth-0 ')' (closing the frame) or ']' / '}'; depth-0 commas
    split that part into argument segments; content = a non-space character."""
    depth, cut, closes = 0, len(text), False
    for k, ch in enumerate(text):
        if depth == 0 and ch in CLOSERS:
            cut, closes = k, ch == ")"
            break
        depth += (ch in OPENERS) - (ch in CLOSERS)
    segments, depth, star = [""], 0, False
    for ch in text[:cut]:
        if depth == 0 and ch == ",":
            segments.append("")
            continue
        star = star or (depth == 0 and ch == "*")
        segments[-1] += ch
        depth += (ch in OPENERS) - (ch in CLOSERS)
    content = [seg.strip() != "" for seg in segments]
    n = len(segments) - 1
    return content[0], n, (sum(content[1:-1]) if n >= 2 else 0), (content[-1] if n >= 1 else False), closes, star


def analysis_arity(va: VocabularyAnalysis, i: int) -> tuple[bool, int, int, bool, bool, bool]:
    return (bool(va.first_content[i]), int(va.n_commas[i]), int(va.mid_content[i]), bool(va.last_content[i]),
            bool(va.closes[i]), bool(va.star[i]))


def type_oracle_row(tok: TextVPATokenizer, rest: str, terminators: str, weight: float = 5.0) -> np.ndarray:
    """TypeSignatureEnergy's documented rule for the still-missing annotation text ``rest``: a token is free if
    its whitespace-free text continues the annotation (a prefix of ``rest``) or completes it and continues with
    a terminator; quote / comment / escape tokens and special tokens are exempt."""
    row = np.full(tok.vocab_size, weight)
    for j, text in enumerate(tok.token_strings):
        if j in tok.special_ids or opaque(text):
            row[j] = 0.0
            continue
        n = no_ws(text)
        if rest.startswith(n) or (n.startswith(rest) and len(n) > len(rest) and n[len(rest)] in terminators):
            row[j] = 0.0
    return row


def first_pad(tokens: Sequence[int], pad_id: int) -> int:
    return next((i for i, t in enumerate(tokens) if t == pad_id), len(tokens))


# =============================================================================================
# Programs
# =============================================================================================
PROGRAM_COUNTER = '''import re
from typing import Dict, List, Optional


class Counter:
    """Counts words; brackets in strings: ([{ and }]) are ignored."""

    def __init__(self, words: List[str], limit: int = 10) -> None:
        self.words = words  # keep (a reference
        self.limit = limit

    def top(self, k: int = 3) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for i, w in enumerate(self.words[: self.limit]):
            out.setdefault(w, []).append(i)
        return dict(sorted(out.items(), key=lambda kv: -len(kv[1]))[:k])


def fact(n: int) -> int:
    # recursion: fact(n - 1) * n  )]}
    return 1 if n <= 1 else n * fact(n - 1)


def combine(*args, **kwargs):
    return list(args) + sorted(kwargs)


def scale(xs: List[float], factor: float = 2.0, offset: Optional[float] = None) -> List[float]:
    base = 0.0 if offset is None else offset
    return [base + factor * x for x in xs]


def main() -> int:
    c = Counter(re.findall(r"\\w+", "a b a (c) [d] {e}"), limit=5)
    table = c.top(2)
    total = fact(5) + len(combine(1, 2, *[3], key="]"))
    vals = scale([1.0, 2.0], 3.0)
    print(f"{total} {table!r} {vals[0]:.2f}", '#not a comment', """multi
line ) ] } string""")
    return total
'''

PROGRAM_PAIRS = '''from typing import Callable, Dict, List, Optional, Tuple


def parse_pairs(text: str, sep: str = ";") -> List[Tuple[str, int]]:
    """Parse 'a=1, b=2' into pairs; ignores '(', '[' and '{' in values."""
    out: List[Tuple[str, int]] = []
    for chunk in text.split(sep):
        if "=" not in chunk:
            continue  # skip a malformed ) chunk
        key, value = chunk.split("=", 1)
        out.append((key.strip(), int(value)))
    return out


def group(pairs: List[Tuple[str, int]], key: Optional[Callable[[str], str]] = None) -> Dict[str, List[int]]:
    table: Dict[str, List[int]] = {}
    for name, val in pairs:
        k = key(name) if key else name
        table.setdefault(k, []).append(val)
    return table


def depth(tree: Dict[str, List[int]], level: int = 0) -> int:
    if not tree:
        return level
    first = next(iter(tree))
    return depth({k: v for k, v in tree.items() if k != first}, level + 1)


class Stack:
    def __init__(self, *items: int, **meta: str) -> None:
        self.items = list(items)
        self.meta = dict(meta)

    def push(self, item: int) -> None:
        self.items.append(item)

    def pop_many(self, n: int = 1) -> List[int]:
        return [self.items.pop() for _ in range(min(n, len(self.items)))]

    def total(self) -> int:
        return sum(self.items) + self.size()

    def size(self) -> int:
        return len(self.items)


def main() -> None:
    pairs = parse_pairs("a=1, b=2, a=3")
    print(group(pairs, key=lambda s: s.upper()), depth(group(pairs)))
    s = Stack(1, 2, 3, mode="lifo")
    s.push(4)
    print(s.pop_many(2), s.total(), f"{s.size()}", 'it\\'s (done]')
'''

PROGRAM_TREE = '''import math


def memo(fn):
    cache = {}

    def wrapper(n):
        if n not in cache:
            cache[n] = fn(n)
        return cache[n]

    return wrapper


@memo
def catalan(n: int) -> int:
    if n <= 1:
        return 1
    return sum(catalan(i) * catalan(n - 1 - i) for i in range(n))


class Tree:
    def __init__(self, value: float, children: list = None) -> None:
        self.value = value
        self.children = children or []

    def add(self, value: float) -> None:
        self.children.append(Tree(value))

    def height(self) -> int:
        return 1 + max((c.height() for c in self.children), default=0)

    def norm(self, p: float = 2.0) -> float:
        own = abs(self.value) ** p
        return math.pow(own + sum(c.norm(p) ** p for c in self.children), 1.0 / p)


def build(depth: int, width: int = 2, scale: float = 1.0) -> Tree:
    root = Tree(scale)
    if depth > 0:
        for k in range(width):
            root.children.append(build(depth - 1, width, scale / (k + 2)))
    return root


if __name__ == "__main__":
    t = build(3)
    t.add(0.5)
    print(catalan(5), t.height(), round(t.norm(), 3), t.norm(p=1.0))
'''

REFERENCE_PROGRAMS = {"counter": PROGRAM_COUNTER, "pairs": PROGRAM_PAIRS, "tree": PROGRAM_TREE}

# Lexically tricky but valid programs (Python 3.13): brackets in strings / comments / docstrings, escapes,
# empty strings next to brackets, raw / byte strings, triple quotes, f-strings with other-quote strings,
# line continuations, decorators, lambdas, comprehensions, walrus, match, type parameters, async.
TRICKY_PROGRAMS = [
    'def f(a, b):\n    """Doc with ) ] } and ( [ {."""\n    s = "([{" + \')]}\'  # ) ] }\n'
    '    return (a, [b, {"k": s}])\n',
    'x = ("", \'\', "\\"(", \'\\\'[\', r"\\d)", b"\\x00]")\ny = [""] + [\'\'] + [""""""]\n',
    "s = '''it's ( \"quoted\" ]'''\nt = \"\"\"a\"\"b\"\"\" + \"\"\n",
    'w = 3\nv = {"k": [1, 2]}\nmsg = f"{v[\'k\'][0]:>{w}} {(lambda q: q)(1)!r} {{literal}} {\')\'}"\n',
    'total = (1 +  # ) comment with \' and "\n         2) \\\n    + 3\nitems = [\n    1,  # ]\n    2,\n]\n',
    "import functools\n\n\n@functools.lru_cache(maxsize=None)\ndef fib(n):\n    return n if n < 2 else "
    "fib(n - 1) + fib(n - 2)\n\n\nsq = {k: [v * v for v in range(k)] for k in range(3) if (m := k) >= 0}\n"
    "odd = sorted({x for x in range(9)}, key=lambda z: -z)[1:3]\n",
    'def g(x):\n    match x:\n        case [a, *rest]:\n            return a\n        case {"k": (b, c)}:\n'
    "            return b\n        case _:\n            return x\n\n\nasync def h(xs):\n"
    "    return [await y for y in xs]\n",
    "#!/usr/bin/env python\n# only comments ( [ {\n",
]
if sys.version_info >= (3, 12):
    TRICKY_PROGRAMS.append("def first[T](xs: list[T]) -> T:\n    return xs[0]\n\n\ntype Pair[K] = tuple[K, K]\n")

STDLIB_MODULES = ["textwrap", "json.encoder", "shlex", "fnmatch", "string", "tokenize"]


# =============================================================================================
# VocabularyAnalysis
# =============================================================================================
def test_vocabulary_analysis_tables_of_the_byte_tokenizer(byte_tok: ByteTokenizer) -> None:
    """Single-byte tokens: identifier characters are ASCII letters and '_' (bytes >= 128 decode to U+FFFD);
    the Python opaque characters are quote, '#' and backslash; C++ swaps '#' for '/'."""
    va = VocabularyAnalysis(byte_tok)
    specials = set(byte_tok.special_ids)
    assert set(np.flatnonzero(va.special)) == specials
    assert set(np.flatnonzero(va.word_bearing)) == {ord(c) for c in string.ascii_letters + "_"}
    assert set(np.flatnonzero(va.opaque)) == {ord(c) for c in "'\"#\\"}
    assert set(np.flatnonzero(VocabularyAnalysis(byte_tok, "cpp").opaque)) == {ord(c) for c in "'\"\\/"}
    np.testing.assert_array_equal(va.always_licensed, ~va.word_bearing & ~va.special)
    assert not va.mismatch.any()
    for b in range(256):
        ch = chr(b) if b < 128 else "\ufffd"
        assert va.closer_suffixes[va.closer_class[b]] == {")": "(", "]": "[", "}": "{"}.get(ch, ""), b
        if ch in "]}":
            continue                                    # a depth-0 ']' / '}' ends the frame's part (no arity)
        got = analysis_arity(va, b)
        assert got[:3] + got[4:] == (not ch.isspace() and ch not in ",)", int(ch == ","), 0, ch == ")", ch == "*")
        assert ch != "," or got[3] is False             # nothing after the comma
    for s in specials:
        assert not va.word_bearing[s] and not va.always_licensed[s]


CLOSER_CASES = {                                        # token: (required stack suffix bottom -> top, mismatch)
    ")": ("(", False), "]": ("[", False), "}": ("{", False),
    "])": ("([", False),                                # closes '[' (innermost), then '('
    ")]": ("[(", False),
    "]))": ("(([", False),
    ")]}": ("{[(", False),
    "x)": ("(", False), ")(": ("(", False), "()": ("", False), "g(a, b)": ("", False), "x": ("", False),
    "(]": (None, True), "[)": (None, True),
}
ARITY_CASES = {      # token: (first_content, n_commas, mid_content, last_content, closes, star)
    ",": (False, 1, 0, False, False, False),
    ", x": (False, 1, 0, True, False, False),
    "),": (False, 0, 0, False, True, False),
    "a, b)": (True, 1, 0, True, True, False),
    "x, y, z": (True, 2, 1, True, False, False),
    "a, , b": (True, 2, 0, True, False, False),
    "g(a, b)": (True, 0, 0, False, False, False),
    "(x": (True, 0, 0, False, False, False),
    "x)": (True, 0, 0, False, True, False),
    ")": (False, 0, 0, False, True, False),
    "*args": (True, 0, 0, False, False, True),
    "**kw": (True, 0, 0, False, False, True),
    " ": (False, 0, 0, False, False, False),
    "    ": (False, 0, 0, False, False, False),
}


def test_vocabulary_analysis_hand_computed_multichar_tokens(list_tok: ListTokenizer) -> None:
    va = VocabularyAnalysis(list_tok)
    for piece, (suffix, mismatch) in CLOSER_CASES.items():
        i = tid(list_tok, piece)
        assert bool(va.mismatch[i]) == mismatch, piece
        if not mismatch:
            assert va.closer_suffixes[va.closer_class[i]] == suffix, piece
    for piece, expected in ARITY_CASES.items():
        got = analysis_arity(va, tid(list_tok, piece))
        if expected[1] == 0:                            # last_content is only meaningful after a comma
            got, expected = got[:3] + got[4:], expected[:3] + expected[4:]
        assert got == expected, piece
    for piece, wb, op in [("foo", True, False), ("_", True, False), ("int)", True, False), ("1", False, False),
                          ("')'", False, True), ("#)", False, True), ("\\)", False, True), ("->", False, False),
                          ("]#", False, True), ("None", True, False)]:
        i = tid(list_tok, piece)
        assert bool(va.word_bearing[i]) == wb and bool(va.opaque[i]) == op, piece


def test_vocabulary_analysis_of_a_small_bpe_matches_the_oracles(bpe_tok: TextVPATokenizer) -> None:
    """Hand checks on multi-character BPE tokens, then every token against the docstring oracles."""
    va = VocabularyAnalysis(bpe_tok)
    i = tid(bpe_tok, "])")
    assert va.closer_suffixes[va.closer_class[i]] == "([" and not va.mismatch[i]
    assert not va.closes[i]                             # its depth-0 ']' ends the call frame's part first
    i = tid(bpe_tok, "]]")
    assert va.closer_suffixes[va.closer_class[i]] == "[["
    i = tid(bpe_tok, "),")
    assert analysis_arity(va, i)[4] and va.closer_suffixes[va.closer_class[i]] == "("
    i = tid(bpe_tok, " int")
    assert va.word_bearing[i] and not va.opaque[i] and analysis_arity(va, i)[:2] == (True, 0)
    i = tid(bpe_tok, "->")
    assert not va.word_bearing[i] and va.always_licensed[i]
    for j, text in enumerate(bpe_tok.token_strings):
        if j in bpe_tok.special_ids:
            assert va.special[j]
            continue
        assert bool(va.word_bearing[j]) == word_bearing(text), text
        assert bool(va.opaque[j]) == opaque(text), text
        suffix, mismatch = closer_oracle(text)
        assert bool(va.mismatch[j]) == mismatch, text
        if not mismatch:
            assert va.closer_suffixes[va.closer_class[j]] == suffix, text
        got, want = analysis_arity(va, j), arity_oracle(text)
        if want[1] == 0:
            got, want = got[:3] + got[4:], want[:3] + want[4:]
        assert got == want, text


def test_whitespace_free_lookups_and_word_licensing(bpe_tok: TextVPATokenizer) -> None:
    va = VocabularyAnalysis(bpe_tok)
    strings = bpe_tok.token_strings
    regular = [j for j in range(bpe_tok.vocab_size) if j not in bpe_tok.special_ids]
    for prefix in ("", "int", "->", ")", "re", "Li", "zzz"):
        want = {j for j in regular if no_ws(strings[j]).startswith(prefix)}
        assert set(va.ids_with_norm_prefix(prefix)) == want, prefix
        assert set(va.ids_with_norm(prefix)) == {j for j in regular if no_ws(strings[j]) == prefix}, prefix
    words = ["print", "range", "int"]
    want = set()
    for word in words:
        for variant in (word, " " + word, "." + word, "(" + word):
            want |= {j for j in bpe_tok.encode(variant) if word_bearing(strings[j])}
    assert va.licensed_ids_for_words(bpe_tok, words) == want


def test_unsupported_language_is_rejected(byte_tok: ByteTokenizer) -> None:
    with pytest.raises(ValueError):
        VocabularyAnalysis(byte_tok, "java")
    with pytest.raises(ValueError):
        base_licensed_words("java")


def test_base_licensed_words_are_keywords_and_builtins() -> None:
    python = base_licensed_words("python")
    assert set(keyword.kwlist) <= python and set(keyword.softkwlist) <= python
    assert {"print", "len", "range", "isinstance", "ValueError", "__name__"} <= python
    assert "foo" not in python and "self" not in python
    assert {"int", "return", "for", "printf", "sizeof"} <= base_licensed_words("cpp")


# =============================================================================================
# Plan-mediated terms
# =============================================================================================
def test_special_token_energy(bpe_tok: TextVPATokenizer) -> None:
    """inf on [MASK], [NULL_PLAN] and the slot delimiters; 0 on [PAD] (end of output) and ordinary tokens."""
    va = VocabularyAnalysis(bpe_tok)
    for weight in (INF, 3.0):
        e = SpecialTokenEnergy(weight).plan_energy(va, PlanConstraints())
        for name in SPECIAL_TOKENS:
            assert e[bpe_tok.special_token_id(name)] == (0.0 if name == "[PAD]" else weight), name
        regular = [j for j in range(bpe_tok.vocab_size) if j not in bpe_tok.special_ids]
        assert (e[regular] == 0.0).all()
    assert SpecialTokenEnergy.plan_mediated


def test_plan_vocabulary_energy(list_tok: ListTokenizer) -> None:
    """Only identifier-bearing tokens outside licensed U base get the weight; tokens without identifier
    characters are always free; licensed_tokens=None imposes no constraint."""
    va = VocabularyAnalysis(list_tok)
    licensed = {tid(list_tok, "foo"), tid(list_tok, "x"), tid(list_tok, "int)")}
    base = {tid(list_tok, "def"), tid(list_tok, "return")}
    plan = PlanConstraints(licensed_tokens=frozenset(licensed))
    for weight in (INF, 2.5):
        e = PlanVocabularyEnergy(base, weight=weight).plan_energy(va, plan)
        for j, text in enumerate(list_tok.token_strings):
            bad = j not in list_tok.special_ids and word_bearing(text) and j not in licensed | base
            assert e[j] == (weight if bad else 0.0), text
    row = PlanVocabularyEnergy(base).plan_energy(va, plan)
    assert_energies(row, list_tok, {"foo": 0.0, "bar": INF, "def": 0.0, "class": INF, "y": INF, "(": 0.0,
                                    "->": 0.0, "1": 0.0, " ": 0.0, "')'": 0.0, "int)": 0.0, "int": INF})
    assert PlanVocabularyEnergy(base).plan_energy(va, PlanConstraints()) is None
    assert PlanVocabularyEnergy.plan_mediated


def test_default_terms_compose_the_documented_set(list_tok: ListTokenizer) -> None:
    ev = ConstraintEnergyEvaluator(list_tok)
    kinds = [type(t) for t in ev.terms]
    assert kinds == [SpecialTokenEnergy, PlanVocabularyEnergy, GrammarEnergy, ArityEnergy, TypeSignatureEnergy]
    assert [t.hard for t in ev.terms] == [True, True, True, False, False]
    assert [t.weight for t in ev.terms[3:]] == [5.0, 5.0]
    assert [type(t) for t in ev.plan_terms] == [SpecialTokenEnergy, PlanVocabularyEnergy]
    assert [type(t) for t in ev.draft_terms] == [GrammarEnergy, ArityEnergy, TypeSignatureEnergy]
    terms = ev.default_terms(identifier_weight=4.0, grammar_weight=7.0, arity_weight=2.0, type_weight=3.0)
    assert [t.weight for t in terms] == [INF, 4.0, 7.0, 2.0, 3.0]
    # keywords and builtins are licensed under any plan, other words only through the plan
    row = energy_row(ev, [], PlanConstraints(licensed_tokens=frozenset()))
    assert_energies(row, list_tok, {"def": 0.0, "return": 0.0, "class": 0.0, "lambda": 0.0, "int": 0.0,
                                    "str": 0.0, "float": 0.0, "None": 0.0, "foo": INF, "bar": INF, "self": INF,
                                    "List": INF, "teger": INF})


def test_from_reference_fields(bpe_tok: TextVPATokenizer, extractor: TreeSitterASTExtractor) -> None:
    structure = extractor.parse(PROGRAM_PAIRS)
    va = VocabularyAnalysis(bpe_tok)
    plan = PlanConstraints.from_reference(PROGRAM_PAIRS, bpe_tok, structure, va)
    strings = bpe_tok.token_strings
    assert plan.licensed_tokens == {j for j in bpe_tok.encode(PROGRAM_PAIRS) if word_bearing(strings[j])}
    expected: dict[str, tuple] = {}
    for f in structure.functions:                       # one tuple per name: repeated names are all kept
        expected[f.name] = expected.get(f.name, ()) + (f,)
    assert dict(plan.signatures) == expected
    assert plan.classes == frozenset(structure.classes)
    assert plan.declared_identifiers == frozenset(structure.declared_identifiers)
    assert plan.source == "reference"


@pytest.mark.parametrize("tokenizer", ["byte_tok", "bpe_tok"])
@pytest.mark.parametrize("name", sorted(REFERENCE_PROGRAMS))
def test_reference_program_has_zero_energy_under_its_oracle_plan(
    tokenizer: str, name: str, request: pytest.FixtureRequest, extractor: TreeSitterASTExtractor
) -> None:
    """Every token of a reference program (strings with brackets, comments, nested calls, nested annotations,
    defaults, *args / **kwargs, methods, recursion) and the [PAD] after it have energy 0 under ALL default
    terms of its own oracle plan, and no hard violation."""
    tok = request.getfixturevalue(tokenizer)
    src = REFERENCE_PROGRAMS[name]
    ast.parse(src)
    structure = extractor.parse(src)
    assert not structure.has_errors
    ev = ConstraintEnergyEvaluator(tok)
    plan = PlanConstraints.from_reference(src, tok, structure, ev.analysis)
    ids = tok.encode(src) + [tok.pad_id]
    energies = ev(torch.tensor([ids]), [plan]).dense()[0]
    e_tok = energies[torch.arange(len(ids)), torch.tensor(ids)]
    bad = [(i, tok.decode(ids[max(0, i - 8):i]), tok.decode([ids[i]], stop_at_pad=False, skip_special=False),
            float(e_tok[i])) for i in torch.nonzero(e_tok).flatten().tolist()]
    assert bad == []
    assert not ev.hard_violations(torch.tensor([ids]), [plan]).any()


SOFT_FALSE_POSITIVES = [
    pytest.param(
        "def f(a, *, key=None):\n    return a\n\n\nx = f(1, key=2)\n",
        id="keyword_only_argument"),
    pytest.param(
        'def split_csv(line, sep=","):\n    return line.split(sep)\n',
        id="comma_in_string_default"),
    pytest.param(
        'def f(a, b="(", c=1):\n    return a\n',
        id="bracket_in_string_default"),
    pytest.param(
        "class Node:\n    def link(self, other: 'Node') -> 'Node':\n        return other\n",
        id="string_annotation"),
    pytest.param(
        "class A:\n    @property\n    def x(self):\n        return 1\n\n    @x.setter\n    def x(self, value):\n"
        "        pass\n",
        id="duplicate_function_name"),
    pytest.param(
        "class A:\n    def __init__(self, x):\n        self.x = x\n\n\nclass B:\n    def __init__(self, x, y):\n"
        "        self.y = y\n",
        id="two_classes_with_init"),
    pytest.param(
        "def append(items, item):\n    return items + [item]\n\n\nxs = []\nxs.append(1)\n",
        id="attribute_call_name_collision"),
    pytest.param(
        "def apply(fn):\n    return fn(1, 2)\n\n\ny = apply(lambda a, b: a + b)\n",
        id="multi_parameter_lambda"),
    pytest.param(
        "def total(xs):\n    return sum(xs)\n\n\nz = total(a + b for a, b in [(1, 2)])\n",
        id="generator_tuple_target"),
    pytest.param(
        "class A:\n    def m(self, x):\n        return x\n\n\nv = A.m(A(), 3)\n",
        id="unbound_method_call"),
]


@pytest.mark.parametrize("src", SOFT_FALSE_POSITIVES)
def test_reference_program_constructs_with_soft_false_positives(
    src: str, byte_tok: ByteTokenizer, extractor: TreeSitterASTExtractor
) -> None:
    """Valid programs that once drew soft penalties under their own oracle plan (regression guards):
    keyword-only arguments, string defaults, string annotations, repeated names, attribute calls,
    lambda / generator commas and unbound method calls must all have zero energy."""
    ast.parse(src)
    structure = extractor.parse(src)
    assert not structure.has_errors
    ev = ConstraintEnergyEvaluator(byte_tok)
    plan = PlanConstraints.from_reference(src, byte_tok, structure, ev.analysis)
    ids = byte_tok.encode(src) + [byte_tok.pad_id]
    energies = ev(torch.tensor([ids]), [plan]).dense()[0]
    e_tok = energies[torch.arange(len(ids)), torch.tensor(ids)]
    assert torch.nonzero(e_tok).flatten().tolist() == []


# =============================================================================================
# GrammarEnergy
# =============================================================================================
GRAMMAR_CASES = [
    # (prefix pieces, expected energies of some next tokens)
    (["f", "(", "a", "["], {")": INF, "]": 0.0, "])": 0.0, ")]": INF, "}": INF, "(]": INF, "[)": INF,
                            "()": 0.0, ")(": INF, "]))": INF, "x": 0.0, "[PAD]": INF}),
    (["f", "(", "a", "[", "x", "]"], {")": 0.0, "]": INF, ")]": INF, "),": 0.0, "[PAD]": INF}),
    (["x", " ", "=", " ", "1"], {")": INF, "]": INF, "}": INF, "()": 0.0, ")(": INF, "(]": INF, "x": 0.0,
                                 "[PAD]": 0.0}),
    # inside a string (right after the opening quote, and later): nothing but [PAD] is rejected
    (["x", "=", '"'], {")": 0.0, "]": 0.0, "(]": 0.0, "]))": 0.0, "[PAD]": INF}),
    (["f", "(", '"', "(", "["], {")": 0.0, "]": 0.0, "}": 0.0, "(]": 0.0, "[PAD]": INF}),
    (['"', '"', '"', "a", "\n", ")"], {")": 0.0, "}": 0.0, "[PAD]": INF}),                 # triple-quoted
    (['"', '"', '"', "a", '"', '"', '"'], {")": INF, "]": INF, "[PAD]": 0.0}),             # ... closed
    (['"', "\\", '"', "("], {")": 0.0, "[PAD]": INF}),                                     # escaped quote
    (['"', "\\", '"', '"'], {")": INF, "[PAD]": 0.0}),                                     # ... closed
    (["f", "(", '"', '"'], {")": 0.0, "]": INF, "[PAD]": INF}),                            # empty string
    # comments: nothing rejected; [PAD] only because of an open bracket
    (["x", " ", "#", " ", "("], {")": 0.0, "]": 0.0, "(]": 0.0, "[PAD]": 0.0}),
    (["f", "(", "#", " ", ")"], {")": 0.0, "]": 0.0, "}": 0.0, "[PAD]": INF}),
    (["f", "(", "#", ")", "\n"], {")": 0.0, "]": INF, "[PAD]": INF}),                       # comment ended
    # after the first [PAD] nothing is constrained
    (["f", "(", "[", "[PAD]"], {")": 0.0, "]": 0.0, "}": 0.0, "(]": 0.0, "[PAD]": 0.0}),
]


@pytest.mark.parametrize("prefix, expected", GRAMMAR_CASES, ids=["".join(c[0]) for c in GRAMMAR_CASES])
def test_grammar_energy_with_explicit_prefixes(prefix: list[str], expected: dict[str, float],
                                               list_tok: ListTokenizer) -> None:
    ev = ConstraintEnergyEvaluator(list_tok, terms=[GrammarEnergy()])
    row = energy_row(ev, prefix)
    assert_energies(row, list_tok, expected)
    assert set(np.unique(row)) <= {0.0, INF}


@pytest.mark.parametrize("prefix", [["f", "(", "a", "["], ["x"], ["f", "(", "#", ")", "\n"], [")", "]"]])
def test_grammar_never_rejects_quote_comment_or_escape_tokens(prefix: list[str], list_tok: ListTokenizer) -> None:
    ev = ConstraintEnergyEvaluator(list_tok, terms=[GrammarEnergy()])
    row = energy_row(ev, prefix)
    opaque_ids = [j for j, t in enumerate(list_tok.token_strings) if j not in list_tok.special_ids and opaque(t)]
    assert len(opaque_ids) >= 8
    assert (row[opaque_ids] == 0.0).all()


def test_grammar_energy_weight_is_the_rejection_value(list_tok: ListTokenizer) -> None:
    ev = ConstraintEnergyEvaluator(list_tok, terms=[GrammarEnergy(weight=7.0)])
    assert_energies(energy_row(ev, ["f", "(", "a", "["]), list_tok, {")": 7.0, "]": 0.0, "[PAD]": 7.0})
    assert not GrammarEnergy.plan_mediated and GrammarEnergy().hard and not GrammarEnergy(7.0).hard


def grammar_energies_of_program(tok: TextVPATokenizer, src: str) -> list[tuple[int, str, str]]:
    """(position, preceding text, token) of every token of ``src`` + [PAD] with non-zero grammar energy."""
    ev = ConstraintEnergyEvaluator(tok, terms=[GrammarEnergy()])
    ids = tok.encode(src) + [tok.pad_id]
    result = ev(torch.tensor([ids]), [PlanConstraints()])
    e_tok = result.rows[result.row_index[0], torch.tensor(ids)]
    return [(i, tok.decode(ids[max(0, i - 12):i]), tok.decode([ids[i]], stop_at_pad=False, skip_special=False))
            for i in torch.nonzero(e_tok).flatten().tolist()]


@pytest.mark.parametrize("tokenizer", ["byte_tok", "bpe_tok"])
def test_grammar_never_rejects_a_token_of_a_valid_program(tokenizer: str, request: pytest.FixtureRequest) -> None:
    """Soundness of the over-approximation: for valid programs (tricky hand-written ones and standard-library
    modules) every token, and the [PAD] that ends the program, has zero grammar energy given its true prefix."""
    tok = request.getfixturevalue(tokenizer)
    sources = list(TRICKY_PROGRAMS) + list(REFERENCE_PROGRAMS.values())
    sources += [inspect.getsource(importlib.import_module(m)) for m in STDLIB_MODULES]
    for src in sources:
        ast.parse(src)
        assert grammar_energies_of_program(tok, src) == [], src[:60]


@pytest.mark.parametrize("src", [
    pytest.param('def first(line):\n    return f"{line.split(")")[0]}"\n', id="closer_in_nested_string"),
    pytest.param('print(f"{"#"}")\nvalue = 1\n', id="hash_in_nested_string"),
])
@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 701 f-string syntax needs Python >= 3.12")
def test_grammar_accepts_pep701_nested_same_quote_fstrings(src: str, byte_tok: ByteTokenizer) -> None:
    ast.parse(src)
    assert grammar_energies_of_program(byte_tok, src) == []


# =============================================================================================
# ArityEnergy
# =============================================================================================
ARITY_SRC = "def f(a, b=1):\n    return a\n\n\nclass K:\n    def m(self, x, y=1):\n        return x\n"


@pytest.fixture(scope="module")
def arity_plan(extractor: TreeSitterASTExtractor) -> PlanConstraints:
    structure = extractor.parse(ARITY_SRC)
    sig = {f.name: f for f in structure.functions}
    assert (sig["f"].min_arity, sig["f"].max_arity) == (1, 2)                     # premises of the cases below
    assert (sig["m"].min_arity, sig["m"].max_arity, sig["m"].call_arity_bounds()) == (2, 3, (1, 2))
    return PlanConstraints(signatures=sig)


W = 5.0
ARITY_CASES_BY_PREFIX = [
    # planned call f(a, b=1): 1 <= arguments <= 2
    (["f", "("], {")": W, "),": W, "x": 0.0, "x)": 0.0, "a, b)": 0.0, "x, y, z": W, " ": 0.0, "*args": 0.0}),
    (["f", " ", "("], {")": W, "x)": 0.0}),
    (["f", "(", "x", ","], {"y": 0.0, ")": 0.0, "a, b)": W}),
    (["f", "(", "x", ",", " ", "y", ","], {"z": W, "z)": W, "[": W, ")": 0.0, " ": 0.0, "\n": 0.0, "*args": 0.0,
                                          "'": 0.0, "#": 0.0}),
    (["f", "(", "x", ",", " ", "y"], {",": 0.0, ", x": W, ")": 0.0, "),": 0.0, "z": 0.0}),
    (["f", "(", "x", ",", " ", "y", ",", " ", "z"], {",": 0.0, ", x": W, ")": 0.0}),     # already violating
    (["f", "(", "x", ",", " ", "[", "a", ",", "b", ","], {"z": 0.0}),                    # nested plain bracket
    (["f", "(", "x", " ", "*", " ", "y", ",", " ", "z", ","], {"w": W}),                 # '*' is a product here
    (["f", "(", '"'], {")": 0.0, "x, y, z": 0.0}),                                       # inside a string
    # unpacking: the frame cannot be checked
    (["f", "(", "*args", ",", " "], {"z": 0.0, ")": 0.0, "x, y, z": 0.0}),
    (["f", "(", "x", ",", " ", "*", "y", ",", " "], {"z": 0.0, "x, y, z": 0.0}),
    (["f", "(", "**kw"], {")": 0.0, ", x": 0.0}),
    # method m(self, x, y=1): obj.m(...) has 1..2 arguments, a plain m(...) 2..3
    (["self", ".", "m", "("], {")": W, "x)": 0.0, "a, b)": 0.0, "x, y, z": W}),
    (["self", ".", "m", "(", "x"], {")": 0.0}),
    # a plain call m(...) does not refer to the method m (no receiver): not checked against it
    (["m", "(", "x"], {")": 0.0, ", x": 0.0}),
    (["m", "(", "x", ",", " ", "y", ",", " ", "z", ","], {"w": 0.0, ")": 0.0}),
    # unplanned functions and class definitions are not checked
    (["g", "("], {")": 0.0}),
    (["g", "(", "x", ",", "y", ",", "z", ","], {"w": 0.0}),
    (["class", " ", "f", "("], {")": 0.0, "x, y, z": 0.0}),
    # the planned def f(a, b=1) must have exactly 2 parameters
    (["def", " ", "f", "("], {")": W, "x)": W, "a, b)": 0.0}),
    (["def", " ", "f", "(", "a", ",", " ", "b", ","], {"c": W, ")": 0.0}),
    (["def", " ", "f", "(", "a", ",", " ", "b", "=", "1"], {")": 0.0, "):": 0.0, ",": 0.0, ", x": W}),
    # after the call is closed nothing is checked
    (["f", "(", "x", ")"], {",": 0.0, "x, y, z": 0.0}),
]


@pytest.mark.parametrize("prefix, expected", ARITY_CASES_BY_PREFIX,
                         ids=["".join(c[0]) for c in ARITY_CASES_BY_PREFIX])
def test_arity_energy_with_explicit_prefixes(prefix: list[str], expected: dict[str, float],
                                             list_tok: ListTokenizer, arity_plan: PlanConstraints) -> None:
    ev = ConstraintEnergyEvaluator(list_tok, terms=[ArityEnergy()])
    row = energy_row(ev, prefix, arity_plan)
    assert_energies(row, list_tok, expected)
    assert set(np.unique(row)) <= {0.0, W}


def test_arity_energy_weight(list_tok: ListTokenizer, arity_plan: PlanConstraints) -> None:
    """Soft by default (5.0); the penalty equals the weight exactly; weight inf makes it hard."""
    assert ArityEnergy().weight == 5.0 and not ArityEnergy().hard and not ArityEnergy.plan_mediated
    for weight in (2.5, INF):
        ev = ConstraintEnergyEvaluator(list_tok, terms=[ArityEnergy(weight)])
        assert_energies(energy_row(ev, ["f", "("], arity_plan), list_tok, {")": weight, "x": 0.0})
    assert ArityEnergy(INF).hard


# =============================================================================================
# TypeSignatureEnergy
# =============================================================================================
TYPE_SRC = "def f(x: int, y: List[int], z: Dict[str, int] = None) -> int:\n    return x\n"
DEF_F = ["def", " ", "f", "("]
X_INT = ["x", ":", " ", "int"]
Y_LIST = ["y", ":", " ", "List", "[", "int", "]"]
Z_DICT = ["z", ":", " ", "Dict", "[", "str", ",", " ", "int", "]"]
SEP = [",", " "]
PARAMS = DEF_F + X_INT + SEP + Y_LIST + SEP + Z_DICT + [" ", "=", " ", "None", ")"]


@pytest.fixture(scope="module")
def type_plan(extractor: TreeSitterASTExtractor) -> PlanConstraints:
    structure = extractor.parse(TYPE_SRC)
    (sig,) = structure.functions
    assert [(p.name, p.annotation) for p in sig.parameters] == [("x", "int"), ("y", "List[int]"),
                                                                ("z", "Dict[str, int]")]
    assert sig.return_annotation == "int"
    return PlanConstraints(signatures={sig.name: sig})


TYPE_CASES = [
    # (prefix, still-missing whitespace-free annotation text or None if nothing is enforced, terminators)
    (DEF_F + ["x", ":", " "], "int", ",)="),
    (DEF_F + ["x", ":", " ", "i", "n"], "t", ",)="),
    (DEF_F + ["x", ":", " ", "i", " ", "n"], "t", ",)="),                               # whitespace-insensitive
    (DEF_F + X_INT, "", ",)="),
    (DEF_F + X_INT + SEP + ["y", ":", " ", "List", "[", " "], "int]", ",)="),
    (DEF_F + X_INT + SEP + ["y", ":", " ", "List", "[", "int"], "]", ",)="),            # nested bracket tracked
    (DEF_F + X_INT + SEP + Y_LIST + SEP + ["z", ":", " ", "Dict", "[", "str"], ",int]", ",)="),
    (DEF_F + X_INT + SEP + Y_LIST + SEP + ["z", ":", " ", "Dict", "[", "str", ",", " "], "int]", ",)="),
    (DEF_F + X_INT + SEP + Y_LIST + SEP + Z_DICT, "", ",)="),
    (DEF_F + X_INT + SEP + Y_LIST + SEP + Z_DICT + [" ", "=", " "], None, ""),         # default value follows
    (PARAMS, "->int", ":"),                                                             # return annotation
    (PARAMS + [" ->", " "], "int", ":"),
    (PARAMS + [" ->", " ", "int"], "", ":"),
    (PARAMS + [" ->", " ", "int", ":"], None, ""),
    # once the draft deviates from the planned annotation nothing is enforced for it
    (DEF_F + ["x", ":", " ", "str"], None, ""),
    (DEF_F + ["x", ":", " ", "str", ",", " ", "y", ":", " "], "List[int]", ",)="),     # next parameter: again
    (PARAMS + [" ->", " ", "str"], None, ""),
    (PARAMS + [":"], None, ""),
    # unplanned def / parameter
    (["def", " ", "g", "(", "x", ":", " "], None, ""),
    (DEF_F + ["w", ":", " "], None, ""),
]


@pytest.mark.parametrize("prefix, rest, terminators", TYPE_CASES, ids=["".join(c[0]) for c in TYPE_CASES])
def test_type_signature_energy_rows_follow_the_planned_annotation(
    prefix: list[str], rest: str | None, terminators: str, list_tok: ListTokenizer, type_plan: PlanConstraints
) -> None:
    ev = ConstraintEnergyEvaluator(list_tok, terms=[TypeSignatureEnergy()])
    row = energy_row(ev, prefix, type_plan)
    expected = np.zeros(list_tok.vocab_size) if rest is None else type_oracle_row(list_tok, rest, terminators)
    mismatched = [list_tok.token_strings[j] for j in np.flatnonzero(row != expected)]
    assert mismatched == []


def test_type_signature_energy_hand_values(list_tok: ListTokenizer, type_plan: PlanConstraints) -> None:
    ev = ConstraintEnergyEvaluator(list_tok, terms=[TypeSignatureEnergy()])
    assert_energies(energy_row(ev, DEF_F + ["x", ":", " "], type_plan), list_tok, {
        "int": 0.0, " int": 0.0, "i": 0.0, " ": 0.0, "\n    ": 0.0, "int)": 0.0, "int,": 0.0, "int=": 0.0,
        "int):": 0.0, "int]": W, "str": W, "float": W, "teger": W, ")": W, ",": W, "List": W,
        "'": 0.0, "#": 0.0, "[MASK]": 0.0})
    assert_energies(energy_row(ev, DEF_F + ["x", ":", " ", "i", "n"], type_plan), list_tok,
                    {"t": 0.0, "teger": W, "int": W})
    assert_energies(energy_row(ev, DEF_F + X_INT + SEP + ["y", ":", " ", "List", "[", "int"], type_plan),
                    list_tok, {"]": 0.0, "])": 0.0, "],": 0.0, "]]": W, ")": W, "int": W})
    assert_energies(energy_row(ev, DEF_F + X_INT + SEP + Y_LIST + SEP + ["z", ":", " ", "Dict", "[", "str"],
                               type_plan), list_tok, {",": 0.0, ", int]": 0.0, ",int": 0.0, "]": W, "int": W})
    assert_energies(energy_row(ev, PARAMS, type_plan), list_tok, {
        ":": W, "):": W, " ->": 0.0, "->": 0.0, "-> int": 0.0, "-> int:": 0.0, "-> str": W, " ": 0.0})
    assert_energies(energy_row(ev, PARAMS + [" ->", " ", "int"], type_plan), list_tok, {":": 0.0, ",": W, " ": 0.0})
    assert TypeSignatureEnergy().weight == 5.0 and not TypeSignatureEnergy().hard
    assert not TypeSignatureEnergy.plan_mediated


# =============================================================================================
# Causality, EnergyResult and the evaluator
# =============================================================================================
def test_draft_dependent_energies_read_only_the_prefix(bpe_tok: TextVPATokenizer,
                                                       extractor: TreeSitterASTExtractor) -> None:
    """E^i depends on w_{<i} only: changing tokens at positions >= i leaves the energies of positions <= i."""
    ev = ConstraintEnergyEvaluator(bpe_tok)
    plan = PlanConstraints.from_reference(PROGRAM_COUNTER, bpe_tok, extractor.parse(PROGRAM_COUNTER), ev.analysis)
    ids = bpe_tok.encode(PROGRAM_COUNTER)[:120]
    base = ev(torch.tensor([ids]), [plan]).dense()[0]
    gen = torch.Generator().manual_seed(3)
    for cut in (0, 7, 31, 64, 119):
        tail = torch.randint(0, bpe_tok.vocab_size, (len(ids) - cut,), generator=gen).tolist()
        other = ev(torch.tensor([ids[:cut] + tail]), [plan]).dense()[0]
        assert torch.equal(base[: cut + 1], other[: cut + 1]), cut


def test_energy_result_dense_and_multiplier_by_hand() -> None:
    plan = torch.tensor([[0.0, INF, 2.0, 0.0], [1.0, 0.0, 0.0, INF]])
    rows = torch.tensor([[0.0, 0.0, 0.0, 0.0], [INF, 0.0, 5.0, 0.0], [0.0, 3.0, 0.0, INF]])
    row_index = torch.tensor([[0, 1, 2], [2, 0, 1]])
    result = EnergyResult(plan, rows, row_index)
    dense = result.dense()
    assert dense.shape == (2, 3, 4)
    for b in range(2):
        for i in range(3):
            for j in range(4):
                e = float(plan[b, j]) + float(rows[int(row_index[b, i]), j])
                assert float(dense[b, i, j]) == e
                for temperature in (0.5, 4.0):
                    m = float(result.multiplier(temperature)[b, i, j])
                    assert m == (0.0 if math.isinf(e) else pytest.approx(math.exp(-e / temperature), rel=1e-6))


def test_evaluator_output_layout_and_row_deduplication(bpe_tok: TextVPATokenizer,
                                                       extractor: TreeSitterASTExtractor) -> None:
    """plan [B, S], rows [U, S] with an all-zero row 0, row_index [B, N]; identical lexer states share a row and
    every other row belongs to exactly one state."""
    ev = ConstraintEnergyEvaluator(bpe_tok)
    programs = [PROGRAM_PAIRS, PROGRAM_TREE, PROGRAM_PAIRS]
    plans = [PlanConstraints.from_reference(p, bpe_tok, extractor.parse(p), ev.analysis) for p in programs]
    n = 150
    drafts = torch.tensor([bpe_tok.encode(p)[:n] for p in programs])
    drafts[2, 40:] = drafts[2, :n - 40].clone()                                       # repeated states
    result = ev(drafts, plans)
    s = bpe_tok.vocab_size
    assert result.plan.shape == (3, s) and result.plan.dtype == torch.float32
    assert result.rows.shape[1] == s and result.rows.dtype == torch.float32
    assert result.row_index.shape == (3, n) and result.row_index.dtype == torch.int64
    assert (result.rows[0] == 0).all()
    row_of_key: dict[tuple, int] = {}
    for b in range(3):
        for i, state in enumerate(ev.lexer_states(drafts[b].tolist(), plans[b])):
            r = int(result.row_index[b, i])
            assert row_of_key.setdefault(state.key(), r) == r
    used = [r for r in row_of_key.values() if r != 0]
    assert len(used) == len(set(used)) == result.rows.shape[0] - 1


def test_evaluator_rejects_mismatched_inputs(list_tok: ListTokenizer) -> None:
    ev = ConstraintEnergyEvaluator(list_tok)
    with pytest.raises(ValueError):
        ev(torch.zeros(3, dtype=torch.long), [PlanConstraints()])
    with pytest.raises(ValueError):
        ev(torch.zeros(2, 3, dtype=torch.long), [PlanConstraints()])


@pytest.mark.parametrize("weight", [0.0, -1.0, math.nan])
def test_energy_term_weight_must_be_positive(weight: float) -> None:
    for term in (SpecialTokenEnergy, GrammarEnergy, ArityEnergy, TypeSignatureEnergy):
        with pytest.raises(ValueError):
            term(weight)
    with pytest.raises(ValueError):
        PlanVocabularyEnergy((), weight=weight)


def noisy_drafts(tok: TextVPATokenizer, sources: Sequence[str], n: int, gen: torch.Generator,
                 p_noise: float = 0.3) -> torch.Tensor:
    """Reference tokens with a fraction replaced by noise -- half uniform over V, half drawn from risky tokens
    (closers, specials, identifier tokens absent from the source) -- plus one early [PAD] and two [MASK]s."""
    clean = torch.tensor([tok.encode(src)[:n] for src in sources])
    drafts = clean.clone()
    strings = tok.token_strings
    for b, src in enumerate(sources):
        own = set(tok.encode(src))
        risky = torch.tensor([j for j, t in enumerate(strings) if j in tok.special_ids or any(c in CLOSERS for c in t)
                              or (word_bearing(t) and j not in own)])
        uniform = torch.randint(0, tok.vocab_size, (n,), generator=gen)
        picked = risky[torch.randint(0, len(risky), (n,), generator=gen)]
        noise = torch.where(torch.rand(n, generator=gen) < 0.5, uniform, picked)
        drafts[b] = torch.where(torch.rand(n, generator=gen) < p_noise, noise, clean[b])
    drafts[0, n // 2] = tok.pad_id
    drafts[1, 3] = drafts[1, 9] = tok.mask_id
    return drafts


def test_hard_violations_agree_with_a_per_position_recomputation(bpe_tok: TextVPATokenizer,
                                                                 extractor: TreeSitterASTExtractor) -> None:
    """Dense energies and hard violations recomputed here from lexer_states, term.row_energy and plan_energy:
    E^i(j) = sum_plan-terms E(j | Z) + sum_draft-terms E(j | state_i); a violation is an infinite energy of the
    position's own token, at positions up to and including the first [PAD]."""
    ev = ConstraintEnergyEvaluator(bpe_tok)
    sources = [PROGRAM_COUNTER, PROGRAM_PAIRS, PROGRAM_TREE, PROGRAM_COUNTER]
    plans = [PlanConstraints.from_reference(p, bpe_tok, extractor.parse(p), ev.analysis) for p in sources]
    gen = torch.Generator().manual_seed(17)
    drafts = noisy_drafts(bpe_tok, sources, 64, gen)
    dense = ev(drafts, plans).dense()
    violations = ev.hard_violations(drafts, plans)
    for b, plan in enumerate(plans):
        tokens = drafts[b].tolist()
        plan_e = np.zeros(bpe_tok.vocab_size)
        for term in ev.terms:
            if term.plan_mediated and (e := term.plan_energy(ev.analysis, plan)) is not None:
                plan_e = plan_e + e
        end = first_pad(tokens, bpe_tok.pad_id)
        for i, state in enumerate(ev.lexer_states(tokens, plan)):
            e_i = plan_e.copy()
            for term in ev.terms:
                if not term.plan_mediated and (r := term.row_energy(ev.analysis, state)) is not None:
                    e_i = e_i + r
            np.testing.assert_array_equal(dense[b, i].double().numpy(), e_i.astype(np.float32).astype(np.float64))
            assert bool(violations[b, i]) == (math.isinf(e_i[tokens[i]]) and i <= end), (b, i)
    assert violations.any() and not violations[0, 64 // 2 + 1:].any()


# =============================================================================================
# Proposition 4(d): plan admissibility and the parallel-update check
# =============================================================================================
@pytest.fixture(scope="module")
def plan_ev(list_tok: ListTokenizer) -> tuple[ConstraintEnergyEvaluator, PlanConstraints]:
    terms = [SpecialTokenEnergy(), PlanVocabularyEnergy([tid(list_tok, "def")]), GrammarEnergy(), ArityEnergy()]
    plan = PlanConstraints(licensed_tokens=frozenset({tid(list_tok, "foo"), tid(list_tok, "x")}))
    return ConstraintEnergyEvaluator(list_tok, terms=terms), plan


def test_plan_admissible_set(plan_ev: tuple[ConstraintEnergyEvaluator, PlanConstraints],
                             list_tok: ListTokenizer) -> None:
    """A(Z): tokens with zero HARD plan-mediated energy, independent of the position and of draft terms."""
    ev, plan = plan_ev
    allowed = ev.plan_admissible([plan, PlanConstraints()])
    assert allowed.shape == (2, list_tok.vocab_size) and allowed.dtype == torch.bool
    licensed = {tid(list_tok, "foo"), tid(list_tok, "x"), tid(list_tok, "def")}
    for j, text in enumerate(list_tok.token_strings):
        if j in list_tok.special_ids:                                           # [PAD] ends the output: allowed
            assert bool(allowed[0, j]) == bool(allowed[1, j]) == (j == list_tok.pad_id), text
            continue
        assert bool(allowed[0, j]) == (not word_bearing(text) or j in licensed), text
        assert bool(allowed[1, j]), text                                        # licensed_tokens=None
    soft = ConstraintEnergyEvaluator(list_tok, terms=[SpecialTokenEnergy(), PlanVocabularyEnergy((), weight=5.0)])
    assert bool(soft.plan_admissible([plan])[0, tid(list_tok, "bar")])          # soft terms do not count


def test_check_parallel_update_detects_an_admissible_position_turning_inadmissible(
    plan_ev: tuple[ConstraintEnergyEvaluator, PlanConstraints], list_tok: ListTokenizer
) -> None:
    ev, plan = plan_ev
    before = torch.tensor([ids_of(list_tok, ["x", "(", "foo", ")", "def"])])
    after = torch.tensor([ids_of(list_tok, ["x", "(", "bar", ")", "def"])])
    with pytest.raises(AdmissibilityViolation, match="sample 0, position 2"):
        ev.check_parallel_update(before, after, [plan])
    report = ev.check_parallel_update(before, after, [plan], strict=False)
    assert report.admissible_before.tolist() == [[True] * 5]
    assert report.admissible_after.tolist() == [[True, True, False, True, True]]
    assert report.premise.tolist() == [True] and report.conclusion.tolist() == [False]
    ok = ev.check_parallel_update(before, torch.tensor([ids_of(list_tok, ["foo", ")", "x", "(", "-"])]), [plan])
    assert ok.premise.tolist() == [True] and ok.conclusion.tolist() == [True]


def test_check_parallel_update_counts_the_source_token_as_admissible(
    plan_ev: tuple[ConstraintEnergyEvaluator, PlanConstraints], list_tok: ListTokenizer
) -> None:
    ev, plan = plan_ev
    mask = list_tok.mask_id
    before = torch.tensor([[mask, tid(list_tok, "x"), tid(list_tok, "y")]])
    assert ev.positions_admissible(before, [plan]).tolist() == [[False, True, False]]
    assert ev.positions_admissible(before, [plan], source_token=mask).tolist() == [[True, True, False]]
    same = ev.check_parallel_update(before, before.clone(), [plan], source_token=mask)
    assert same.premise.tolist() == [False]                                           # 'y' is inadmissible
    worse = before.clone()
    worse[0, 2] = tid(list_tok, "bar")                                                # inadmissible -> inadmissible
    ev.check_parallel_update(before, worse, [plan], source_token=mask)
    broken = before.clone()
    broken[0, 0] = tid(list_tok, "bar")                                               # source -> inadmissible
    with pytest.raises(AdmissibilityViolation):
        ev.check_parallel_update(before, broken, [plan], source_token=mask)
    allowed = ev.plan_admissible([plan])
    assert torch.equal(ev.positions_admissible(before, [plan], mask, allowed),
                       ev.positions_admissible(before, [plan], mask))


def test_plan_mediated_hard_terms_survive_parallel_updates_of_the_sampler_kernel(
    bpe_tok: TextVPATokenizer, extractor: TreeSitterASTExtractor
) -> None:
    """Prop. 4(d) end to end: start from a plan-admissible draft, tilt random posteriors by the evaluator's full
    energies (plain and repair-aware), draw the parallel update with the library kernel: no position of the
    draft ever leaves A(Z) (``check_parallel_update`` would raise)."""
    ev = ConstraintEnergyEvaluator(bpe_tok)
    sources = [PROGRAM_COUNTER, PROGRAM_TREE]
    plans = [PlanConstraints.from_reference(p, bpe_tok, extractor.parse(p), ev.analysis) for p in sources]
    allowed = ev.plan_admissible(plans)
    gen = torch.Generator().manual_seed(23)
    w = torch.multinomial(allowed.double(), 40, replacement=True, generator=gen)
    for step in range(4):
        post = torch.softmax(3.0 * torch.randn(2, 40, bpe_tok.vocab_size, generator=gen), dim=-1)
        energies = ev(w, plans).dense()
        for mult in (torch.exp(-energies / 0.5), repair_aware_multiplier(energies, w, 0.5)):
            kern = tilted_jump_kernel(post, w, 1.0 if step == 3 else 0.6, mult, 1.0)
            w_next = sample_jump(kern, w, generator=gen)
            report = ev.check_parallel_update(w, w_next, plans)
            assert report.premise.all() and report.conclusion.all()
        w = w_next


def test_draft_dependent_grammar_term_is_not_safe_under_parallel_updates(list_tok: ListTokenizer) -> None:
    """Prop. 4(d), failure case with the library's grammar term: in "( x x" each of positions 1 and 2 may close
    the bracket given its own prefix, so the parallel update reaches "( ) )" (an inadmissible ')') with positive
    probability, while a sequential update (position 2 re-evaluated after position 1 moved) cannot."""
    ev = ConstraintEnergyEvaluator(list_tok, terms=[GrammarEnergy()])
    close, x = tid(list_tok, ")"), tid(list_tok, "x")
    w = torch.tensor([ids_of(list_tok, ["(", "x", "x"])])
    energies = ev(w, [PlanConstraints()]).dense()
    assert energies[0, 1, close] == 0 and energies[0, 2, close] == 0
    post = torch.zeros(1, 3, list_tok.vocab_size, dtype=torch.float64)
    post[0, :, close], post[0, :, x] = 0.6, 0.4
    k = kernel_distribution(tilted_jump_kernel(post, w, 1.0, torch.exp(-energies.double()), 1.0), w)[0]
    assert k[1, close].item() == pytest.approx(0.6) and k[2, close].item() == pytest.approx(0.6)
    assert k[1, close].item() * k[2, close].item() > 0.0                               # P("( ) )") > 0
    both = torch.tensor([ids_of(list_tok, ["(", ")", ")"])])
    assert ev.hard_violations(both, [PlanConstraints()]).tolist() == [[False, False, True]]
    after_first = torch.tensor([ids_of(list_tok, ["(", ")", "x"])])
    assert math.isinf(ev(after_first, [PlanConstraints()]).dense()[0, 2, close])      # sequential: blocked


# =============================================================================================
# Repair
# =============================================================================================
TINY = ["x", "y", "foo", "(", ")", "[", "]", ",", " "]


@pytest.fixture(scope="module")
def tiny_tok() -> ListTokenizer:
    return ListTokenizer(TINY)


def test_repair_hand_ranked_example(tiny_tok: ListTokenizer) -> None:
    """Left to right: keep admissible tokens; replace each inadmissible one by the highest-scoring token that is
    admissible given the REPAIRED prefix; stop at the first [PAD]."""
    t = {p: tid(tiny_tok, p) for p in TINY + ["[PAD]", "[MASK]", "[NULL_PLAN]"]}
    ev = ConstraintEnergyEvaluator(tiny_tok, terms=[SpecialTokenEnergy(), PlanVocabularyEnergy(()), GrammarEnergy()])
    plan = PlanConstraints(licensed_tokens=frozenset({t["x"], t["y"]}))
    draft = torch.tensor([[t["("], t["foo"], t["]"], t["[MASK]"], t[")"], t["x"], t["[PAD]"], t[")"]]])
    scores = torch.zeros(1, 8, tiny_tok.vocab_size)
    ranks = {
        0: {")": 0.9, "(": 0.1},                                          # "(" is admissible: kept
        1: {"foo": 0.9, "]": 0.8, "[NULL_PLAN]": 0.7, "y": 0.6, "x": 0.5},  # foo: unlicensed; ']' after '('
        2: {"]": 0.9, "[PAD]": 0.8, ")": 0.7},                            # [PAD] inside an open bracket
        3: {"[MASK]": 0.95, "]": 0.9, ")": 0.85, "foo": 0.8, "(": 0.7},   # empty stack: no closer
        4: {"x": 0.9},
        5: {"y": 0.9},
        6: {"x": 0.9},
        7: {"x": 0.9},                                                    # after the output: untouched
    }
    for i, r in ranks.items():
        for piece, v in r.items():
            scores[0, i, t[piece]] = v
    assert ev.hard_violations(draft, [plan]).tolist() == [[False, True, True, True, False, False, False, False]]
    out, unrepaired = ev.repair(draft, scores, [plan])
    expected = [t["("], t["y"], t[")"], t["("], t[")"], t["x"], t["[PAD]"], t[")"]]
    assert out.tolist() == [expected] and unrepaired == 0
    assert not ev.hard_violations(out, [plan]).any()
    assert draft.tolist()[0][1] == t["foo"]                                           # input not modified


def test_repair_replaces_leftover_source_tokens(tiny_tok: ListTokenizer) -> None:
    """With a grammar-only evaluator [MASK] is not forbidden by any term; as the source token it is replaced."""
    ev = ConstraintEnergyEvaluator(tiny_tok, terms=[GrammarEnergy()])
    x, y, mask = tid(tiny_tok, "x"), tid(tiny_tok, "y"), tiny_tok.mask_id
    draft = torch.tensor([[x, mask, y]])
    scores = torch.zeros(1, 3, tiny_tok.vocab_size)
    scores[0, 1, y], scores[0, 1, mask] = 0.9, 0.1
    out, unrepaired = ev.repair(draft, scores, [PlanConstraints()], source_token=mask)
    assert out.tolist() == [[x, y, y]] and unrepaired == 0
    kept, _ = ev.repair(draft, scores, [PlanConstraints()])
    assert kept.tolist() == [[x, mask, y]]


def test_repair_replaces_a_top_scoring_source_token(tiny_tok: ListTokenizer) -> None:
    ev = ConstraintEnergyEvaluator(tiny_tok, terms=[GrammarEnergy()])
    x, y, mask = tid(tiny_tok, "x"), tid(tiny_tok, "y"), tiny_tok.mask_id
    scores = torch.zeros(1, 3, tiny_tok.vocab_size)
    scores[0, 1, mask], scores[0, 1, y] = 0.9, 0.5
    out, _ = ev.repair(torch.tensor([[x, mask, y]]), scores, [PlanConstraints()], source_token=mask)
    assert out.tolist() == [[x, y, y]]


class ForbidEverything(EnergyTerm):
    """A plan-mediated hard term without any admissible token (to exercise the unrepaired count)."""

    name = "forbid_everything"
    plan_mediated = True

    def plan_energy(self, analysis: VocabularyAnalysis, plan: PlanConstraints) -> np.ndarray:
        return np.full(analysis.vocab_size, self.weight)


def test_repair_counts_positions_without_an_admissible_token(tiny_tok: ListTokenizer) -> None:
    ev = ConstraintEnergyEvaluator(tiny_tok, terms=[ForbidEverything()])
    t = {p: tid(tiny_tok, p) for p in ["x", "(", "y", "[PAD]"]}
    draft = torch.tensor([[t["x"], t["("], t["y"], t["[PAD]"], t["x"]],
                          [t["x"], t["("], t["y"], t["x"], t["x"]]])
    out, unrepaired = ev.repair(draft, torch.rand(2, 5, tiny_tok.vocab_size), [PlanConstraints()] * 2)
    assert torch.equal(out, draft)
    assert unrepaired == 4 + 5                                                        # up to the first [PAD]


def test_repair_of_noisy_drafts(bpe_tok: TextVPATokenizer, extractor: TreeSitterASTExtractor) -> None:
    """Default terms, oracle plans, noisy drafts with [MASK] / [PAD]: the output has no hard violation; a token
    is kept iff it is admissible given the repaired prefix (and is not the source token); every replacement
    maximises score(j) e^{-E(j)/T} over the admissible tokens, E (soft terms included) given the repaired
    prefix and T = 1; nothing after the output's first [PAD] changes."""
    ev = ConstraintEnergyEvaluator(bpe_tok)
    sources = [PROGRAM_PAIRS, PROGRAM_COUNTER, PROGRAM_TREE]
    plans = [PlanConstraints.from_reference(p, bpe_tok, extractor.parse(p), ev.analysis) for p in sources]
    gen = torch.Generator().manual_seed(29)
    drafts = noisy_drafts(bpe_tok, sources, 40, gen, p_noise=0.35)
    scores = torch.rand(*drafts.shape, bpe_tok.vocab_size, generator=gen)
    mask = bpe_tok.mask_id
    out, unrepaired = ev.repair(drafts, scores, plans, source_token=mask)
    assert unrepaired == 0
    assert not ev.hard_violations(out, plans).any()
    assert ev.hard_violations(drafts, plans).any()                                  # there was work to do
    changed = 0
    for b, plan in enumerate(plans):
        repaired, original = out[b].tolist(), drafts[b].tolist()
        end = first_pad(repaired, bpe_tok.pad_id)
        for i in range(end + 1, len(repaired)):
            assert repaired[i] == original[i]
        for i in range(min(end + 1, len(repaired))):
            prefix = torch.tensor([repaired[:i] + [bpe_tok.pad_id]])
            row = ev(prefix, [plan]).dense()[0, i].double()
            admissible = torch.isfinite(row)
            admissible[mask] = False
            if bool(admissible[original[i]]):
                assert repaired[i] == original[i], (b, i)
            else:
                changed += 1
                assert bool(admissible[repaired[i]]), (b, i)
                best = max(float(scores[b, i, j]) * math.exp(-float(row[j]))
                           for j in range(bpe_tok.vocab_size) if admissible[j])
                chosen = float(scores[b, i, repaired[i]]) * math.exp(-float(row[repaired[i]]))
                assert math.isclose(chosen, best, rel_tol=1e-12), (b, i)
    assert changed > 5


class _SoftInsideBracket(EnergyTerm):
    """Test-local draft-dependent soft term: E(j) = weight for the listed tokens inside any open bracket."""

    name = "soft_inside_bracket"
    plan_mediated = False

    def __init__(self, tokens: Sequence[int], weight: float) -> None:
        super().__init__(weight)
        self.tokens = list(tokens)

    def row_energy(self, analysis: VocabularyAnalysis, state: Any) -> np.ndarray | None:
        if not state.stack:
            return None
        e = np.zeros(analysis.vocab_size, dtype=np.float64)
        e[self.tokens] = self.weight
        return e


def test_repair_refills_positions_that_never_left_their_source(tiny_tok: ListTokenizer) -> None:
    """An unrealized position (a stall: it still holds its w_0 token) is refilled like a leftover [MASK], even
    when its token is admissible; a realized admissible token is kept whatever the scores; positions after the
    output's first [PAD] are untouched."""
    t = {p: tid(tiny_tok, p) for p in TINY + ["[PAD]"]}
    ev = ConstraintEnergyEvaluator(tiny_tok, terms=[SpecialTokenEnergy(), PlanVocabularyEnergy(()), GrammarEnergy()])
    plan = PlanConstraints(licensed_tokens=frozenset({t["x"], t["y"]}))
    draft = torch.tensor([[t["x"], t["y"], t["("], t["x"], t[")"], t["[PAD]"], t["y"]]])
    unrealized = torch.tensor([[False, True, False, True, False, False, True]])
    scores = torch.zeros(1, 7, tiny_tok.vocab_size)
    ranks = {
        0: {"y": 0.9, "x": 0.1},                       # realized and admissible: kept
        1: {"foo": 0.9, "x": 0.5, "y": 0.2},           # unrealized: foo is unlicensed, so x
        3: {"x": 0.9, "y": 0.8},                       # unrealized, and its own token is the best: x again
        6: {"x": 0.9},                                 # after the output: untouched
    }
    for i, r in ranks.items():
        for piece, v in r.items():
            scores[0, i, t[piece]] = v
    out, unrepaired = ev.repair(draft, scores, [plan], unrealized=unrealized)
    assert out.tolist() == [[t["x"], t["x"], t["("], t["x"], t[")"], t["[PAD]"], t["y"]]] and unrepaired == 0
    kept, _ = ev.repair(draft, scores, [plan])                                     # no stalls reported
    assert torch.equal(kept, draft)


def test_repair_ranks_replacements_with_energies_of_the_repaired_prefix(tiny_tok: ListTokenizer) -> None:
    """Replacement = argmax_j score(j) e^{-E(j)/T} with E given the REPAIRED prefix. Draft ") x": ')' closes
    nothing and becomes '(' (score 0.9); position 1 is unrealized with scores y 0.6, x 0.5. Inside the repaired
    '(' the soft term costs y 1.0: 0.6 e^{-1} = 0.221 < 0.5, so x; with T = 100, 0.6 e^{-0.01} = 0.594 > 0.5, so
    y. (Given the unrepaired prefix ')' there is no open bracket and y would win at any T.)"""
    t = {p: tid(tiny_tok, p) for p in TINY}
    ev = ConstraintEnergyEvaluator(tiny_tok, terms=[GrammarEnergy(), _SoftInsideBracket([t["y"]], 1.0)])
    draft = torch.tensor([[t[")"], t["x"]]])
    unrealized = torch.tensor([[False, True]])
    scores = torch.zeros(1, 2, tiny_tok.vocab_size)
    scores[0, 0, t["("]], scores[0, 0, t["x"]] = 0.9, 0.2
    scores[0, 1, t["y"]], scores[0, 1, t["x"]] = 0.6, 0.5
    out, _ = ev.repair(draft, scores, [PlanConstraints()], unrealized=unrealized)
    assert out.tolist() == [[t["("], t["x"]]]
    hot, _ = ev.repair(draft, scores, [PlanConstraints()], unrealized=unrealized, temperature=100.0)
    assert hot.tolist() == [[t["("], t["y"]]]


def test_repair_rejects_malformed_inputs(tiny_tok: ListTokenizer) -> None:
    ev = ConstraintEnergyEvaluator(tiny_tok, terms=[GrammarEnergy()])
    draft = torch.zeros(1, 3, dtype=torch.long)
    scores = torch.zeros(1, 3, tiny_tok.vocab_size)
    with pytest.raises(ValueError):
        ev.repair(draft, scores[:, :2], [PlanConstraints()])
    with pytest.raises(ValueError):
        ev.repair(draft, scores, [PlanConstraints()], unrealized=torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        ev.repair(draft, scores, [PlanConstraints()], temperature=0.0)


def test_repair_keeps_an_admissible_reference(bpe_tok: TextVPATokenizer, extractor: TreeSitterASTExtractor) -> None:
    ev = ConstraintEnergyEvaluator(bpe_tok)
    plan = PlanConstraints.from_reference(PROGRAM_TREE, bpe_tok, extractor.parse(PROGRAM_TREE), ev.analysis)
    ids = torch.tensor([bpe_tok.encode(PROGRAM_TREE) + [bpe_tok.pad_id] * 3])
    out, unrepaired = ev.repair(ids, torch.rand(*ids.shape, bpe_tok.vocab_size), [plan], source_token=bpe_tok.mask_id)
    assert torch.equal(out, ids) and unrepaired == 0
