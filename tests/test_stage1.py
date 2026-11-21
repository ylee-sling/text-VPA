# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 1, step 1 (the structure oracle): line slots against explicit loops and Python's own tokenizer, the
bridge against a hand-written Gemma 4 block and a float64 version of Eq. 10, the closed gate against the Stage 0
realizer, the oracle's plan, data, trainer, evaluation and report. All on a tiny random Gemma 4."""

from __future__ import annotations

import io
import json
import keyword
import math
import random
import tokenize
from collections import Counter, UserDict

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForCausalLM"):                    # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_stage0 import (  # noqa: E402
    MASK,
    VOCAB,
    WINDOW,
    CharTokenizer,
    _loop_masks,
    _tasks,
    _write_corpus,
    randomize_adapters,
    tiny_backbone,
    tiny_realizer,
)

from text_vpa.evaluation.tasks import EvalTask  # noqa: E402
from text_vpa.models.encoders import TargetOutput  # noqa: E402
from text_vpa.models.plan import require_predicted_plan  # noqa: E402
from text_vpa.stage0 import data as s0data  # noqa: E402
from text_vpa.stage0.backbone import RealizerSpec, Stage0Realizer, left_pad  # noqa: E402
from text_vpa.stage0.evaluate import realizer_samples  # noqa: E402
from text_vpa.stage0.sampler import SamplerConfig, realize  # noqa: E402
from text_vpa.stage0.train import Stage0Config, Stage0Trainer  # noqa: E402
from text_vpa.stage1.bridge import BridgeSpec, PlanBridge  # noqa: E402
from text_vpa.stage1.data import (  # noqa: E402
    PlanCorpusStream,
    collate_plan,
    load_plan_validation,
    tokenize_plan_examples,
)
from text_vpa.stage1.oracle import (  # noqa: E402
    StructureOracle,
    TaskPlans,
    kind_match,
    normalize_reference,
    oracle_samples,
)
from text_vpa.stage1.plan import ORACLE_LABEL, OraclePlan, require_oracle_plan  # noqa: E402
from text_vpa.stage1.slots import (  # noqa: E402
    KINDS,
    Codebook,
    LineSlots,
    collate_slots,
    encode_with_starts,
    layout_code,
    line_depth,
    line_kind,
    line_slots,
    position_code,
    program_lines,
    skeleton,
    split_lines,
    starts_by_decoding,
)
from text_vpa.stage1.train import OracleTrainer, Stage1Config, build_plan_modules  # noqa: E402

PROGRAM = (
    "import re\n"
    "\n"
    "def count(words, pattern=None):\n"
    '    """Count."""\n'
    "    total = 0\n"
    "    for word in words:\n"
    "        if pattern and not re.match(pattern, word):\n"
    "            continue\n"
    "        total += 1  # one more\n"
    "    print(total)\n"
    "    return total\n"
)


def kind(name: str) -> int:
    return KINDS.index(name)


# =============================================================================================
# Lines: kinds and depths
# =============================================================================================
def test_split_lines_and_depth():
    assert split_lines("a\n  b\n") == [(0, "a"), (2, "  b")]             # no empty line after the last newline
    assert split_lines("a\n\nb") == [(0, "a"), (2, ""), (3, "b")] and split_lines("") == []
    assert split_lines("\n") == [(0, "")] and split_lines("a\r\nb\n") == [(0, "a\r"), (3, "b")]
    assert [line_depth(line) for line in ("x", "    x", "        x", "\tx", "\t\tx", "  x", "      x", "   ", "",
                                          " " * 40 + "x")] == [0, 1, 2, 1, 2, 0, 1, 0, 0, 7]


def test_line_kinds_by_hand():
    cases = {
        "": "blank", "   \t": "blank", "# note": "comment", "    # x = 1": "comment", "@staticmethod": "decorator",
        "def f(a=1):": "def", "async def f():": "def", "class A(B):": "class", "return x == 1": "return",
        "return": "return", "if x:": "if", "if(x):": "if", "elif y == 2:": "elif", "else:": "else",
        "else: x = 1": "else", "for i in range(n):": "for", "async for x in y:": "for", "while True:": "while",
        "try:": "try", "except ValueError as e:": "except", "finally:": "finally", "with open(p) as f:": "with",
        "async with a as b:": "with", "import os": "import", "from a import b": "import", "raise ValueError(x)": "raise",
        "assert x == 1, 'no'": "assert", "pass": "pass", "break": "break", "continue": "continue",
        "yield x": "yield", '"""Doc."""': "string", "'a'": "string", 'f"{x}"': "string", "rb'x'": "string",
        ")": "close", "]": "close", "})": "close", "x = 1": "assign", "x += 1": "assign", "x: int = 1": "assign",
        "a, b = b, a": "assign", "self.x[i] = f(y=2)": "assign", "x >>= 2": "assign", "x <<= 2": "assign",
        "x //= 2": "assign", "(a, b) = c": "assign", "foo(x).bar = 1": "assign", "print(x)": "call",
        "self.items.append(x)": "call", "foo (x)": "call", "f(a=1, b=2)": "call", "x == 1": "other", "x <= y": "other",
        "x >= y": "other", "x != y": "other", "x[0](y)": "other", "lambda: 0": "other", "del x": "other",
        "global x": "other", "await f()": "other", "(n := 10)": "other", "1 + 2": "other", "-x": "other",
        "x": "other", "x  # y = 2": "other", "print('a = b')": "call", "s = 'it''s # no comment'": "assign",
        "import_module(x)": "call", "format = 3": "assign", "returned = 1": "assign", "iffy(x)": "call",
        "x = (": "assign", "key=value,": "assign", "name,": "other", 'r = "\\"="': "assign", 'print("\\"=")': "call",
    }
    for line, want in cases.items():
        assert KINDS[line_kind(line)] == want, line
        assert KINDS[line_kind("    " + line + "  ")] == want, line          # indentation and trailing blanks
    assert len(KINDS) == 27 and len(set(KINDS)) == 27


_ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", ">>=", "<<=", "&=", "^=", "|=", "@="}
_KEYWORD_KINDS = {"def": "def", "class": "class", "return": "return", "if": "if", "elif": "elif", "else": "else",
                  "for": "for", "while": "while", "try": "try", "except": "except", "finally": "finally",
                  "with": "with", "import": "import", "from": "import", "raise": "raise", "assert": "assert",
                  "pass": "pass", "break": "break", "continue": "continue", "yield": "yield"}
_SKIP = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}


def _kinds_by_tokenize(source: str) -> dict[int, str]:
    """The definition of ``line_kind`` computed with Python's own tokenizer instead of the module's scanner:
    row (0-based) -> kind, for every row on which a token starts (rows inside a multi-line string have none)."""
    rows: dict[int, list[tokenize.TokenInfo]] = {}
    inside_string = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _SKIP:
            continue
        rows.setdefault(tok.start[0] - 1, []).append(tok)
        for row in range(tok.start[0], tok.end[0]):                    # rows a multi-line token continues on
            inside_string.add(row)
    middle = {getattr(tokenize, name) for name in ("FSTRING_MIDDLE", "FSTRING_END") if hasattr(tokenize, name)}
    string_start = {tokenize.STRING} | {getattr(tokenize, n) for n in ("FSTRING_START",) if hasattr(tokenize, n)}
    out: dict[int, str] = {}
    for row, tokens in rows.items():
        first = tokens[0]
        if row in inside_string or first.type in middle:
            continue
        if first.type == tokenize.COMMENT:
            out[row] = "comment"
        elif first.string == "@":
            out[row] = "decorator"
        elif first.string in (")", "]", "}"):
            out[row] = "close"
        elif first.type in string_start:
            out[row] = "string"
        else:
            name = first.string if first.type == tokenize.NAME else ""
            if name == "async" and len(tokens) > 1 and tokens[1].string in _KEYWORD_KINDS:
                name = tokens[1].string
            if name in _KEYWORD_KINDS:
                out[row] = _KEYWORD_KINDS[name]
                continue
            depth, assigned = 0, False
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    break
                if tok.type == tokenize.OP and tok.string in "([{":
                    depth += 1
                elif tok.type == tokenize.OP and tok.string in ")]}":
                    depth = max(0, depth - 1)
                elif tok.type == tokenize.OP and tok.string in _ASSIGN_OPS and depth == 0:
                    assigned = True
                    break
            if assigned:
                out[row] = "assign"
                continue
            i = 0
            call = first.type == tokenize.NAME
            while call and i + 2 < len(tokens) and tokens[i + 1].string == "." and tokens[i + 2].type == tokenize.NAME:
                i += 2
            call = call and i + 1 < len(tokens) and tokens[i + 1].string == "("
            out[row] = "call" if call else "other"
    return out


def test_line_kinds_agree_with_pythons_tokenizer():
    """Every line of real modules on which a token starts gets the kind that Python's tokenizer implies; blank
    lines are blank. (Rows inside multi-line strings carry no token and are not compared.)"""
    import bisect as bisect_module
    import heapq
    import inspect
    import textwrap

    compared = Counter()
    for module in (textwrap, heapq, bisect_module, keyword, json.decoder, json.encoder, random):
        source = inspect.getsource(module)
        want = _kinds_by_tokenize(source)
        lines = [line for _, line in split_lines(source)]
        for row, line in enumerate(lines):
            if not line.strip():
                assert KINDS[line_kind(line)] == "blank"
            elif row in want:
                assert KINDS[line_kind(line)] == want[row], (module.__name__, row + 1, line)
                compared[want[row]] += 1
    assert sum(compared.values()) > 1500 and len(compared) >= 20, compared     # most kinds occur


# =============================================================================================
# Slots
# =============================================================================================
def _slots_by_characters(text: str, starts: list[int], max_slots: int) -> tuple[list[int], list[int]]:
    """Reference: walk the characters, number the lines, and count the tokens that begin on each."""
    line_of_char, line = [], 0
    for char in text:
        line_of_char.append(line)
        if char == "\n":
            line += 1
    lines = line if text.endswith("\n") or not text else line + 1
    counts = [0] * max(lines, 1)
    for start in starts:
        index = line_of_char[start] if start < len(text) else lines - 1
        counts[min(index, len(counts) - 1)] += 1
    if len(counts) > max_slots:
        counts = counts[:max_slots - 1] + [sum(counts[max_slots - 1:])]
    first = [sum(counts[:m]) for m in range(len(counts))]
    return first, counts


def test_line_slots_against_a_loop_over_characters():
    rng = random.Random(0)
    texts = [PROGRAM, PROGRAM.rstrip("\n"), "x = 1\n\n\n\ny = 2\n", "pass", "\n\nx\n", "a\r\nb\r\n",
             "".join(f"v{i} = {i}\n" for i in range(50))]
    for text in texts:
        for trial in range(20):
            # Random token boundaries: every token starts at some character; a few tokens span newlines.
            cuts = sorted(rng.sample(range(1, len(text)), k=min(len(text) - 1, rng.randint(0, 40)))) if len(text) > 1 else []
            starts = [0, *cuts]
            for max_slots in (4, 32):
                got = line_slots(text, starts, max_slots)
                first, counts = _slots_by_characters(text, starts, max_slots)
                assert list(got.start) == first and list(got.length) == counts, (text[:20], trial, max_slots)
                assert sum(got.length) == len(starts) and got.count == len(counts) <= max_slots
                lines = [line for _, line in split_lines(text)]
                assert got.lines == len(lines)
                assert list(got.kind) == [line_kind(line) for line in lines][:max_slots]      # the slot's own line
                assert list(got.depth) == [line_depth(line) for line in lines][:max_slots]

    # Characters as tokens: a slot's length is its line's length with the newline.
    tok = CharTokenizer()
    ids, starts = encode_with_starts(tok, [PROGRAM])
    slots = line_slots(PROGRAM, starts[0])
    assert list(slots.length) == [len(line) + 1 for _, line in split_lines(PROGRAM)] and len(ids[0]) == len(PROGRAM)
    assert [KINDS[k] for k in slots.kind] == ["import", "blank", "def", "string", "assign", "for", "if", "continue",
                                              "assign", "call", "return"]
    assert list(slots.depth) == [0, 0, 0, 1, 1, 1, 2, 3, 2, 1, 1]

    # A blank line whose newline was merged into the previous line's token has no token of its own.
    merged = line_slots("a\n\nb\n", [0, 1, 3, 4])                       # tokens "a", "\n\n", "b", "\n"
    assert list(merged.length) == [2, 0, 2] and list(merged.start) == [0, 2, 2]
    assert [KINDS[k] for k in merged.kind] == ["other", "blank", "other"]
    empty = line_slots("", [])
    assert empty.count == 1 and empty.length == (0,) and KINDS[empty.kind[0]] == "blank"
    with pytest.raises(ValueError):
        line_slots("ab\n", [1, 0])
    with pytest.raises(ValueError):
        line_slots("ab\n", [0, 9])

    # A reference longer than the canvas (evaluation): tokens past the budget do not count, later slots go.
    long = "".join(f"v{i} = {i}\n" for i in range(10))                  # 7 characters per line
    full = line_slots(long, list(range(len(long))), 32)
    cut = line_slots(long, list(range(len(long))), 32, max_tokens=17)
    assert cut.count == 3 and list(cut.length) == [7, 7, 3] and cut.kind == full.kind[:3] and cut.lines == 10
    assert line_slots(long, list(range(len(long))), 32, max_tokens=len(long)) == full       # it fits: unchanged


def test_collate_slots_and_roll():
    a = line_slots(PROGRAM, list(range(len(PROGRAM))), 16)
    b = line_slots("x = 1\n", list(range(6)), 16)
    batch = collate_slots([a, b], 16)
    assert batch.mask.sum(dim=1).tolist() == [11, 1] and batch.kind.shape == (2, 16)
    assert batch.length[0, :11].tolist() == list(a.length) and batch.start[1].tolist() == [0] * 16
    assert not bool(batch.length[~batch.mask].any())
    rolled = batch.roll(1)
    assert torch.equal(rolled.kind[0], batch.kind[1]) and torch.equal(rolled.mask[1], batch.mask[0])
    with pytest.raises(ValueError):
        collate_slots([a], 8)                                           # 11 slots do not fit M = 8


def test_position_and_layout_codes_against_float64_loops():
    x = torch.tensor([0, 1, 2, 7, 255, 256])
    got = position_code(x, 16, max_period=1024.0)
    half = 8
    # Tolerance: the code is float32 (no float64 in forward paths), and a float32 angle of pi * 256 = 804 is
    # known to 6e-5 (its unit in the last place), so sines and cosines are compared to 1e-4.
    for row, value in enumerate(x.tolist()):
        for k in range(half):
            omega = math.pi * (2.0 / 1024.0) ** (k / (half - 1))
            assert math.isclose(float(got[row, k]), math.sin(omega * value), abs_tol=1e-4)
            assert math.isclose(float(got[row, half + k]), math.cos(omega * value), abs_tol=1e-4)
    # The shortest wavelength is 2 tokens (parity), the longest max_period: 256 positions get distinct codes.
    assert torch.allclose(position_code(torch.arange(4), 16)[:, half], torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=1e-5)
    codes = position_code(torch.arange(257), 64)
    assert float(torch.cdist(codes, codes).fill_diagonal_(9.0).min()) > 0.5
    start, length = torch.tensor([[3, 40]]), torch.tensor([[5, 0]])
    layout = layout_code(start, length, 64)
    assert layout.shape == (1, 2, 64)
    torch.testing.assert_close(layout[0, 0], torch.cat([position_code(torch.tensor(3), 24),
                                                        position_code(torch.tensor(5), 16),
                                                        position_code(torch.tensor(8), 24)]))
    torch.testing.assert_close(layout[0, 1, :24], layout[0, 1, 40:])     # length 0: end = start
    with pytest.raises(ValueError):
        layout_code(start, length, 60)
    with pytest.raises(ValueError):
        position_code(x, 7)


def test_skeleton_and_codebook(tmp_path):
    lines = program_lines(PROGRAM)
    full = skeleton(lines)
    assert full == tuple((kind(k), d) for k, d in [("import", 0), ("def", 0), ("string", 1), ("assign", 1), ("for", 1),
                                                   ("if", 2), ("continue", 3), ("assign", 2), ("call", 1),
                                                   ("return", 1)])       # the blank line is not part of it
    assert skeleton(lines, max_lines=3) == full[:3]
    assert skeleton(lines, with_depth=False) == tuple(k for k, _ in full)
    programs = {"a": "x = 1\nreturn x\n", "b": "for i in y:\n    f(i)\n", "c": "x = 2\n\nreturn x\n",
                "d": "if x:\n    pass\n", "e": "for j in z:\n        g(j)\n"}
    counts = Counter(skeleton(program_lines(text)) for text in programs.values())
    assert counts.most_common(1)[0][1] == 2                              # a and c share a skeleton
    book = Codebook.build(counts, size=3)                                # "other" + the two most frequent
    assert len(book) == 3 and book.mode(program_lines(programs["a"])) == book.mode(program_lines(programs["c"])) == 1
    tied = sorted(s for s, n in counts.items() if n == 1)                # ties: the smaller skeleton first
    assert book.entries[1] == tied[0] and book.mode(program_lines("while x:\n    pass\n")) == 0
    covered = [text for text in programs.values() if book.mode(program_lines(text))]
    assert math.isclose(book.coverage(program_lines(t) for t in programs.values()), len(covered) / 5)
    flat = Codebook.build(Counter(skeleton(program_lines(t), with_depth=False) for t in programs.values()), 3,
                          with_depth=False)
    assert flat.mode(program_lines(programs["b"])) == flat.mode(program_lines(programs["e"])) == 1   # depth dropped
    book.save(tmp_path / "book.json")
    again = Codebook.load(tmp_path / "book.json")
    assert again.entries == book.entries and again.with_depth and again.max_lines == 16
    assert [again.mode(program_lines(t)) for t in programs.values()] == \
        [book.mode(program_lines(t)) for t in programs.values()]


def test_token_starts_from_a_fast_tokenizer_equal_decoded_prefix_lengths(tmp_path):
    from test_stage0_cli import write_checkpoint

    write_checkpoint(tmp_path / "model")
    tokenizer = s0data.HFTokenizer(transformers.AutoTokenizer.from_pretrained(tmp_path / "model"))
    texts = [PROGRAM, "x = 1\n\n\ny = 2\n", "    return a + b\n"]
    ids, starts = encode_with_starts(tokenizer, texts)
    assert ids == tokenizer.encode_batch(texts)                          # the tokens are Stage 0's
    for row, offsets, text in zip(ids, starts, texts, strict=True):
        assert offsets == starts_by_decoding(tokenizer, row) == list(range(len(text)))
    plain = CharTokenizer()                                              # no offsets of its own: decoded prefixes
    assert encode_with_starts(plain, texts)[1] == starts and encode_with_starts(plain, []) == ([], [])

    class Odd(CharTokenizer):                                            # offsets out of order or past the text
        def decode(self, ids):
            return "x" * {0: 0, 1: 3, 2: 2, 3: 9}[len(ids)]

    assert encode_with_starts(Odd(), ["abcd"])[1] == [[0, 3, 3, 4]]      # clipped, non-decreasing: slots exist
    assert line_slots("abcd", encode_with_starts(Odd(), ["abcd"])[1][0]).length == (4,)


# =============================================================================================
# The bridge
# =============================================================================================
D, WIDTH, HEADS = 16, 16, 4


def bridged(seed: int = 0, num_modes: int = 0, open_gates: bool = False, **spec):
    realizer, reference = tiny_realizer(seed)
    torch.manual_seed(seed + 100)
    bridge = PlanBridge(BridgeSpec(latent_dim=D, width=WIDTH, num_heads=HEADS, position_dim=8, num_modes=num_modes,
                                   **spec), realizer.hidden_size, realizer.num_layers)
    bridge.attach(realizer)
    bridge.eval()
    if open_gates:
        with torch.no_grad():
            bridge.gate.copy_(torch.linspace(-0.9, 1.1, realizer.num_layers))
    return realizer, bridge, reference


def some_plan(batch: int, slots: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, slots, D, generator=generator)
    mask = torch.ones(batch, slots, dtype=torch.bool)
    mask[0, slots - 2:] = False
    return z, mask


def test_a_closed_gate_is_the_stage0_realizer_exactly():
    """a_l = 0: hidden states with a plan equal those without one bit for bit, the sampler draws the same tokens
    in every order, and the prompt cache does not change."""
    realizer, bridge, _ = bridged()
    randomize_adapters(realizer)                                       # a trained Stage 0 realizer
    assert float(bridge.gates().detach().abs().max()) == 0.0
    ids = torch.randint(5, VOCAB, (2, 30))
    canvas = ids[:, 9:]
    is_mask = torch.zeros(2, 21, dtype=torch.bool)
    is_mask[:, 4:15] = True
    t, h = torch.tensor([0.25, 0.5]), torch.tensor([0.125, 0.5])
    with torch.no_grad():
        cache = realizer.prefill(*left_pad([r.tolist() for r in ids[:, :9]], 0))
        plan = bridge.project(bridge.conditioning(*some_plan(2)))
        without = realizer.canvas_hidden(cache, canvas, is_mask, t, h)
        with_plan = realizer.canvas_hidden(cache, canvas, is_mask, t, h, plan=plan)
        torch.testing.assert_close(with_plan, without, rtol=0, atol=0)
        again = realizer.prefill(*left_pad([r.tolist() for r in ids[:, :9]], 0))
        for layer, keys in cache.keys.items():
            torch.testing.assert_close(again.keys[layer], keys, rtol=0, atol=0)
        with pytest.raises(ValueError, match="adapters"):
            realizer.canvas_hidden(cache, canvas, is_mask, t, h, adapters=False, plan=plan)
        one = realizer.prefill(*left_pad([[2, 30, 31, 32]], 0))
        single = bridge.project(bridge.conditioning(*[x[1:] for x in some_plan(2)]))
        for order in ("random", "confidence", "left_to_right"):
            cfg = SamplerConfig(num_steps=6, schedule="cosine", order=order, temperature=0.8)
            base = realize(realizer, one.expand(4), torch.full((4,), 50), 24, cfg, forbidden=[0, 2],
                           generator=torch.Generator().manual_seed(3))
            planned = realize(realizer, one.expand(4), torch.full((4,), 50), 24, cfg, forbidden=[0, 2],
                              generator=torch.Generator().manual_seed(3), plan=single.expand(4))
            assert torch.equal(planned.tokens, base.tokens) and planned.revealed == base.revealed, order
    # Attaching adds nothing to the realizer's own checkpoint and cannot be done twice.
    assert all(not name.startswith("bridge") for name in realizer.trainable_state())
    with pytest.raises(RuntimeError, match="already attached"):
        bridge.attach(realizer)


def _reference_cross_attention(bridge: PlanBridge, layer: int, hidden: torch.Tensor, tokens: torch.Tensor,
                               mask: torch.Tensor) -> np.ndarray:
    """tanh(a_l) CA_l(norm_l(hidden), Z~) in float64 with explicit loops over samples, heads and positions
    (Eq. 10 with the position term): hidden [B, C, d_r], tokens [B, M~, d] (already Z~), mask [B, M~]."""
    spec = bridge.spec
    dh = spec.width // spec.num_heads
    w_q = bridge.query_proj[layer].weight.detach().double().numpy()
    w_k = bridge.key_proj[layer].weight.detach().double().numpy()        # the normalised weight, as a given
    w_v = bridge.value_proj[layer].weight.detach().double().numpy()
    w_o = bridge.out_proj[layer].weight.detach().double().numpy()
    w_p = bridge.position_proj[layer].weight.detach().double().numpy() if spec.query_position else None
    norm = bridge.norm[layer]
    gain, bias = norm.weight.detach().double().numpy(), norm.bias.detach().double().numpy()
    gate = math.tanh(float(bridge.gate[layer].detach()))
    h64, z64 = hidden.detach().double().numpy(), tokens.detach().double().numpy()
    b, c, _ = h64.shape
    out = np.zeros_like(h64)
    half = spec.position_dim // 2
    for i in range(b):
        keys, values = z64[i] @ w_k.T, z64[i] @ w_v.T                     # [M~, w]
        for j in range(c):
            x = h64[i, j]
            x = (x - x.mean()) / math.sqrt(x.var() + norm.eps) * gain + bias
            query = w_q @ x
            if w_p is not None:
                omega = [math.pi * (2.0 / 1024.0) ** (k / (half - 1)) for k in range(half)]
                code = np.array([math.sin(o * j) for o in omega] + [math.cos(o * j) for o in omega])
                query = query + w_p @ code
            mixed = np.zeros(spec.width)
            for head in range(spec.num_heads):
                sl = slice(head * dh, (head + 1) * dh)
                scores = np.array([query[sl] @ keys[m, sl] / math.sqrt(dh) if mask[i, m] else -np.inf
                                   for m in range(keys.shape[0])])
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                mixed[sl] = weights @ values[:, sl]
            out[i, j] = gate * (w_o @ mixed)
    return out


@pytest.mark.parametrize("query_position", [True, False])
def test_cross_attention_is_eq_10(query_position):
    realizer, bridge, _ = bridged(open_gates=True, num_modes=6, gamma=1.7, query_position=query_position)
    z, mask = some_plan(3, slots=6)
    mode = torch.tensor([0, 5, 2])
    cond = bridge.conditioning(z, mask, mode=mode, drop=torch.tensor([False, False, True]))
    # Z~ = [Z ; e(u)] / gamma for kept samples; the null token alone for the dropped one.
    want = torch.cat([z, bridge.mode_embedding.weight[mode][:, None]], dim=1) / 1.7
    torch.testing.assert_close(cond.tokens[:2], want[:2])
    assert cond.mask[0].tolist() == [True] * 4 + [False] * 2 + [True] and cond.mask[1].all()
    assert cond.mask[2].tolist() == [True] + [False] * 6
    torch.testing.assert_close(cond.tokens[2, 0], bridge.null_token.detach())
    null = bridge.null_conditioning(2)
    assert null.tokens.shape == (2, 1, D) and bool(null.dropped.all())
    cache = bridge.project(cond)
    hidden = torch.randn(3, 9, realizer.hidden_size)
    for layer in (0, 3, 6):
        with torch.no_grad():
            got = bridge.cross_attention(layer, hidden, cache)
        want = _reference_cross_attention(bridge, layer, hidden, cond.tokens, cond.mask)
        np.testing.assert_allclose(got.double().numpy(), want, rtol=1e-4, atol=1e-5)
    # The dropped sample depends on nothing but the null token.
    other = bridge.project(bridge.conditioning(torch.randn_like(z), mask, mode=mode,
                                               drop=torch.tensor([False, False, True])))
    with torch.no_grad():
        torch.testing.assert_close(bridge.cross_attention(3, hidden, other)[2],
                                   bridge.cross_attention(3, hidden, cache)[2])
    with pytest.raises(ValueError, match="mode"):
        bridge.conditioning(z, mask)                                    # this bridge has a mode row
    with pytest.raises(ValueError, match="at least one"):
        bridge.project(bridge.conditioning(z, torch.zeros_like(mask), mode=mode)._replace(
            mask=torch.zeros(3, 7, dtype=torch.bool)))


def _block_by_hand(reference, bridge: PlanBridge | None, ids: torch.Tensor, prefix_len: int,
                   tokens: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor:
    """Gemma 4's forward pass over [prompt; canvas] written out block by block from the layer's submodules, with
    the bridge's term added to the canvas rows after self-attention. Returns the logits ``[1, T, S]``."""
    text = reference.model
    canvas_len = ids.shape[1] - prefix_len
    masks = _loop_masks(prefix_len, canvas_len, WINDOW, "bidirectional")
    positions = torch.arange(ids.shape[1])[None]
    hidden = text.embed_tokens(ids)
    per_layer = text.project_per_layer_inputs(hidden, text.get_per_layer_inputs(ids, None))
    rotary = {k: text.rotary_emb(hidden, positions, k) for k in set(text.config.layer_types)}
    shared: UserDict = UserDict()
    for index, layer in enumerate(text.layers):
        layer_type = text.config.layer_types[index]
        residual = hidden
        attended, _ = layer.self_attn(hidden_states=layer.input_layernorm(hidden),
                                      position_embeddings=rotary[layer_type], attention_mask=masks[layer_type],
                                      shared_kv_states=shared, position_ids=positions, past_key_values=None)
        hidden = residual + layer.post_attention_layernorm(attended)
        if bridge is not None:                                          # h <- h + tanh(a_l) CA_l(norm(h), Z~)
            term = _reference_cross_attention(bridge, index, hidden[:, prefix_len:], tokens, mask)
            hidden = torch.cat([hidden[:, :prefix_len], hidden[:, prefix_len:] + torch.from_numpy(term).float()], 1)
        residual = hidden
        hidden = residual + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(hidden)))
        gated = layer.act_fn(layer.per_layer_input_gate(hidden)) * per_layer[:, :, index, :]
        hidden = hidden + layer.post_per_layer_input_norm(layer.per_layer_projection(gated))
        hidden = hidden * layer.layer_scalar
    logits = reference.lm_head(text.norm(hidden))
    return torch.tanh(logits / 30.0) * 30.0


def test_the_bridge_enters_each_block_after_self_attention_on_the_canvas_only():
    """With open gates the bridged realizer equals a block written out by hand: the update lands on the residual
    stream after self-attention, on canvas rows only, in every block."""
    realizer, bridge, reference = bridged(open_gates=True)             # LoRA and modulation at their initial values
    prefix_len, canvas_len = 10, 14
    ids = torch.randint(5, VOCAB, (1, prefix_len + canvas_len))
    z, mask = some_plan(1, slots=4, seed=2)
    mask = torch.tensor([[True, True, False, True]])
    cond = bridge.conditioning(z, mask)
    with torch.no_grad():
        plain = _block_by_hand(reference, None, ids, prefix_len, None, None)
        library = reference(ids, attention_mask=_loop_masks(prefix_len, canvas_len, WINDOW, "bidirectional")).logits
        torch.testing.assert_close(plain, library, rtol=1e-4, atol=1e-5)      # the hand-written block is Gemma 4's
        want = _block_by_hand(reference, bridge, ids, prefix_len, cond.tokens, cond.mask)
        cache = realizer.prefill(*left_pad([ids[0, :prefix_len].tolist()], 0))
        canvas = ids[:, prefix_len:]
        hidden = realizer.canvas_hidden(cache, canvas, torch.zeros_like(canvas, dtype=torch.bool), torch.zeros(1),
                                        torch.ones(1), plan=bridge.project(cond))
        got = realizer.logits(hidden)
    torch.testing.assert_close(got, want[:, prefix_len:], rtol=2e-4, atol=2e-5)
    assert float((want[:, prefix_len:] - plain[:, prefix_len:]).abs().max()) > 1e-2     # the plan does something
    torch.testing.assert_close(want[:, :prefix_len], plain[:, :prefix_len], rtol=1e-5, atol=1e-6)   # not to the prompt


def test_keys_and_values_are_projected_once_and_the_audit_enforces_it():
    realizer, bridge, _ = bridged(open_gates=True)
    randomize_adapters(realizer)
    z, mask = some_plan(1)
    cond = bridge.conditioning(z, mask)
    cache = bridge.project(cond)
    again = bridge.project(cond)                                       # eval mode: no power iteration, same result
    for layer in range(bridge.num_layers):
        torch.testing.assert_close(again.keys[layer], cache.keys[layer], rtol=0, atol=0)
        torch.testing.assert_close(again.values[layer], cache.values[layer], rtol=0, atol=0)
        want = (cond.tokens @ bridge.key_proj[layer].weight.T).view(1, 5, HEADS, -1).transpose(1, 2)
        torch.testing.assert_close(cache.keys[layer], want, rtol=1e-5, atol=1e-6)
    assert cache.keys[0].shape == (1, HEADS, 5, WIDTH // HEADS) and cache.expand(3).values[2].shape[0] == 3
    with pytest.raises(ValueError, match="one plan"):
        cache.expand(3).expand(2)

    cfg = SamplerConfig(num_steps=5, schedule="cosine", temperature=0.7)
    with torch.no_grad():
        prompt = realizer.prefill(*left_pad([[2, 30, 31]], 0))
    before = dict(bridge.passes)
    result = realize(realizer, prompt.expand(3), torch.full((3,), 50), 20, cfg, forbidden=[0, 2],
                     generator=torch.Generator().manual_seed(0), plan=cache.expand(3))
    assert bridge.passes["projection"] == before["projection"]         # nothing projected during the K passes
    assert bridge.passes["cross"] - before["cross"] == bridge.num_layers * 5 and result.sequential_depth == 6
    # A plan that changes the samples (the gates are open), under the same seed.
    base = realize(realizer, prompt.expand(3), torch.full((3,), 50), 20, cfg, forbidden=[0, 2],
                   generator=torch.Generator().manual_seed(0))
    assert not torch.equal(base.tokens, result.tokens)

    # A realizer that projects the plan again inside the loop is caught, and so is a block that skips the cache.
    original = realizer.canvas_hidden

    def reprojecting(*args, **kwargs):
        bridge.project(cond)
        return original(*args, **kwargs)

    realizer.canvas_hidden = reprojecting                               # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="depth audit"):
        realize(realizer, prompt.expand(3), torch.full((3,), 50), 20, cfg, forbidden=[0, 2], plan=cache.expand(3))

    def forgetting(*args, **kwargs):
        kwargs.pop("plan")
        return original(*args, **kwargs)

    realizer.canvas_hidden = forgetting                                 # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="depth audit"):
        realize(realizer, prompt.expand(3), torch.full((3,), 50), 20, cfg, forbidden=[0, 2], plan=cache.expand(3))
    realizer.canvas_hidden = original                                   # type: ignore[method-assign]
    other = PlanBridge(bridge.spec, realizer.hidden_size, realizer.num_layers)
    with pytest.raises(ValueError, match="another bridge"):
        other.cross_attention(0, torch.zeros(1, 4, realizer.hidden_size), cache)


def test_plan_projections_have_operator_norm_one_and_dropout_follows_its_rate():
    torch.manual_seed(0)
    bridge = PlanBridge(BridgeSpec(latent_dim=32, width=32, num_heads=4, position_dim=8), 64, 3)
    bridge.train()
    z, mask = torch.randn(4, 6, 32), torch.ones(4, 6, dtype=torch.bool)
    for _ in range(60):                                                # the power iteration advances in training
        bridge.project(bridge.conditioning(z, mask, drop=torch.zeros(4, dtype=torch.bool)))
    bridge.eval()
    for proj in (*bridge.key_proj, *bridge.value_proj):
        sigma = float(np.linalg.svd(proj.weight.detach().double().numpy(), compute_uv=False)[0])
        assert abs(sigma - 1.0) < 2e-3                                 # Lipschitz in the plan (Sec. 3.2)
    assert not bool(bridge.conditioning(z, mask).dropped.any())        # eval: no dropout
    bridge.train()
    generator = torch.Generator().manual_seed(1)
    big, big_mask = torch.randn(4000, 3, 32), torch.ones(4000, 3, dtype=torch.bool)
    cond = bridge.conditioning(big, big_mask, generator=generator)
    rate = float(cond.dropped.float().mean())
    assert abs(rate - 0.1) < 5 * math.sqrt(0.1 * 0.9 / 4000)
    assert bool((cond.mask[cond.dropped].sum(dim=1) == 1).all()) and bool(cond.mask[~cond.dropped].all())
    torch.testing.assert_close(cond.tokens[~cond.dropped], big[~cond.dropped])
    second = bridge.conditioning(big, big_mask, generator=torch.Generator().manual_seed(1))
    assert torch.equal(second.dropped, cond.dropped)                   # the draw is the generator's
    with pytest.raises(ValueError):
        BridgeSpec(width=30, num_heads=4)
    with pytest.raises(ValueError):
        BridgeSpec(plan_dropout=1.0)
    assert BridgeSpec.from_dict(bridge.spec.to_dict()) == bridge.spec


def test_gradients_reach_the_gate_first_and_checkpointing_changes_nothing():
    realizer, bridge, _ = bridged()
    randomize_adapters(realizer)
    oracle = StructureOracle(D)
    slots = collate_slots([line_slots(PROGRAM[:40], list(range(40)), 8), line_slots("x = 1\n", list(range(6)), 8)], 8)
    ids = torch.randint(5, VOCAB, (2, 18))
    canvas = ids[:, 7:]
    is_mask = torch.zeros(2, 11, dtype=torch.bool)
    is_mask[:, 3:9] = True
    t, h = torch.tensor([0.0, 0.5]), torch.tensor([0.25, 0.5])
    cache = realizer.prefill(*left_pad([r.tolist() for r in ids[:, :7]], 0))
    targets = torch.randint(5, VOCAB, (2, 11))

    def gradients(checkpointing: bool) -> dict[str, torch.Tensor]:
        for module in (realizer, bridge, oracle):
            module.zero_grad()
        plan = oracle(slots)
        hidden = realizer.canvas_hidden(cache, canvas, is_mask, t, h, checkpointing=checkpointing,
                                        plan=bridge.project(bridge.conditioning(plan.z, plan.mask)))
        loss, _ = realizer.weighted_nll(hidden[is_mask], targets[is_mask], torch.ones(int(is_mask.sum())), chunk=5)
        loss.backward()
        named = [*(("bridge." + n, p) for n, p in bridge.named_parameters()),
                 *(("oracle." + n, p) for n, p in oracle.named_parameters()),
                 *((n, p) for n, p in realizer.named_parameters() if p.requires_grad)]
        return {n: (torch.zeros_like(p) if p.grad is None else p.grad.clone()) for n, p in named}

    closed = gradients(False)
    # Closed gates: the loss moves the gates and nothing else of the plan's path (tanh(0) = 0 stops the rest).
    assert float(closed["bridge.gate"].abs().min()) > 0
    assert all(float(g.abs().max()) == 0 for n, g in closed.items()
               if n.startswith(("bridge.", "oracle.")) and n != "bridge.gate")
    with torch.no_grad():
        bridge.gate.fill_(0.3)
    opened = gradients(False)
    silent = [n for n, g in opened.items() if n.startswith(("bridge.", "oracle.")) and float(g.abs().max()) == 0]
    assert silent == ["bridge.null_token"], silent                      # no sample was dropped in this batch
    replayed = gradients(True)
    assert set(replayed) == set(opened)
    for name, grad in opened.items():
        torch.testing.assert_close(replayed[name], grad, rtol=2e-4, atol=1e-6, msg=name)


# =============================================================================================
# Plan types and the oracle's plan
# =============================================================================================
def test_an_oracle_plan_is_rejected_by_the_methods_guard():
    plan = OraclePlan(torch.zeros(1, 2, D), torch.ones(1, 2, dtype=torch.bool), "structure")
    assert require_oracle_plan(plan) is plan and plan.batch_size == 1
    with pytest.raises(TypeError, match="PredictedPlan"):
        require_predicted_plan(plan)                                   # what the Stage 2 trainer calls
    with pytest.raises(TypeError, match="PredictedPlan"):
        require_predicted_plan(TargetOutput(torch.zeros(1, 2, D), torch.ones(1, 2, dtype=torch.bool)))
    with pytest.raises(TypeError, match="OraclePlan"):
        require_oracle_plan(TargetOutput(torch.zeros(1, 2, D), torch.ones(1, 2, dtype=torch.bool)))
    with pytest.raises(TypeError, match="OraclePlan"):
        require_oracle_plan(torch.zeros(1, 2, D))


def test_structure_oracle_reads_structure_and_nothing_else():
    torch.manual_seed(0)
    oracle = StructureOracle(32)
    tok = CharTokenizer()
    first = "def f(a):\n    return a\n"
    second = "def g(b):\n    return b\n"                                # other tokens, the same line structure
    third = "def g(b):\n    b = b * 2\n"
    _, starts = encode_with_starts(tok, [first, second, third])
    rows = [line_slots(text, s, 6) for text, s in zip((first, second, third), starts, strict=True)]
    assert rows[0] == rows[1] != rows[2]
    batch = collate_slots(rows, 6)
    with torch.no_grad():
        plan = oracle(batch)
    assert plan.source == "structure" and plan.z.shape == (3, 6, 32) and plan.mask.sum(dim=1).tolist() == [2, 2, 2]
    torch.testing.assert_close(plan.z[0], plan.z[1], rtol=0, atol=0)
    assert float((plan.z[0] - plan.z[2]).abs().max()) > 1e-3 and not bool(plan.z[:, 2:].any())
    # Z_m = E_kind[kind_m] + E_depth[depth_m] + W code(start_m, length_m), written out.
    code = layout_code(torch.tensor(10), torch.tensor(13), 32).double()
    want = (oracle.kind.weight[kind("return")].double() + oracle.depth.weight[1].double()
            + oracle.layout.weight.double() @ code + oracle.layout.bias.double())
    torch.testing.assert_close(plan.z[0, 1].double(), want, rtol=1e-5, atol=1e-6)


def test_references_are_normalised_before_their_structure_is_read():
    tabs = "def f(a):\r\n\tif(a>1): \r\n\t\treturn (a) \r\n\treturn 0"
    two = "def f(a):\n  if a > 1:\n    return a\n  return 0\n"
    text, ok = normalize_reference(tabs)
    assert ok and text == "def f(a):\n    if a > 1:\n        return a\n    return 0\n" == normalize_reference(two)[0]
    assert [d for _, d in program_lines(text)] == [0, 1, 2, 1]
    assert [d for _, d in program_lines(two)] == [0, 0, 1, 0]            # read raw, the depths would be wrong
    broken, ok = normalize_reference("def f(:\r\n  pass  \r\n")
    assert not ok and broken == "def f(:\n  pass\n"
    exec(compile(text, "<normalised>", "exec"), {})                     # still a program
    assert kind_match(text, [k for k, _ in program_lines(text)]) == 1.0
    assert kind_match("def f(a):\n    return a\n", [k for k, _ in program_lines(text)]) == 0.25
    assert kind_match(text + "print(1)\n", [k for k, _ in program_lines(text)]) == 0.8 and kind_match("", []) == 1.0


# =============================================================================================
# Data
# =============================================================================================
def test_plan_examples_are_stage0s_examples_with_slots(tmp_path):
    _write_corpus(tmp_path)
    tok = CharTokenizer()
    pairs = [("def f(x):\n", "    y = x\n    return y\n", "complete"), ("p", "y" * 80, "complete"),
             ("q" * 200, "z", "complete"), ("def g():\n", json.loads('"    return \\"\\ud83d\\"\\n"'), "complete")]
    plain, dropped = s0data.tokenize_examples(pairs, tok, 40, 16)
    planned, dropped_plan = tokenize_plan_examples(pairs, tok, 40, 16, max_slots=8)
    assert dropped == dropped_plan and len(plain) == len(planned) == 1
    assert planned[0].prompt_ids == plain[0].prompt_ids and planned[0].canvas_ids == plain[0].canvas_ids
    assert planned[0].slots == LineSlots((kind("assign"), kind("return")), (1, 1), (0, 10), (10, 13), 2)
    batch = collate_plan(planned, 40, tok.eos_id, 8)
    base = s0data.collate(plain, 40, tok.eos_id)
    assert torch.equal(batch.target, base.target) and batch.prefix_rows == base.prefix_rows
    assert batch.slots.mask[0].tolist() == [True, True] + [False] * 6 and batch.slots.length[0, :2].tolist() == [10, 13]
    assert int(batch.slots.length.sum()) == int(batch.length[0]) - 1      # the <eos> belongs to no slot

    def streams(state=None):
        return (s0data.CorpusStream(tmp_path, tok, 96, 128, write_fraction=0.5, seed=3, chunk_rows=7, state=state),
                PlanCorpusStream(tmp_path, tok, 96, 128, write_fraction=0.5, seed=3, chunk_rows=7, state=state,
                                 max_slots=8))

    old, new = streams()
    a, b = old.take(70), new.take(70)
    assert [(e.prompt_ids, e.canvas_ids, e.kind) for e in a] == [(e.prompt_ids, e.canvas_ids, e.kind) for e in b]
    assert old.state() == new.state() and old.dropped == new.dropped
    # A Stage 0 stream's position continues in the planner stage's stream (a new oracle run does this).
    _, resumed = streams(json.loads(json.dumps(old.state())))
    assert [e.canvas_ids for e in resumed.take(30)] == [e.canvas_ids for e in old.take(30)]
    for example in b:
        assert sum(example.slots.length) == len(example.canvas_ids) - 1 and example.slots.count <= 8
    validation = load_plan_validation(tmp_path, tok, 96, 128, limit=6, max_slots=8)
    assert [e.canvas_ids for e in validation] == [e.canvas_ids for e in s0data.load_validation(tmp_path, tok, 96, 128, 6)]


# =============================================================================================
# The oracle trainer
# =============================================================================================
def _structured_corpus(root, functions: int = 24, shards: int = 2) -> None:
    """Functions whose bodies have 1 to 4 lines of different kinds: the line structure tells a lot about them."""
    import gzip

    root.mkdir(parents=True, exist_ok=True)
    bodies = ["    return a\n", "    b = a\n    return b\n", "    if a:\n        return a\n    return 0\n",
              "    for b in a:\n        print(b)\n    pass\n    return a\n"]
    rows = []
    for i in range(functions):
        context = f"def f{i % 7}(a):\n"
        rows.append({"context": context, "target": context + bodies[i % 4], "documented": False, "method": False})
    for s in range(shards):
        with gzip.open(root / f"train-{s:05d}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for row in rows[s::shards]:
                handle.write(json.dumps(row) + "\n")
    with gzip.open(root / "val.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows[:8]:
            handle.write(json.dumps(row) + "\n")


def _stage0_checkpoint(tmp_path, steps: int = 6):
    """A short Stage 0 run on the structured corpus: the checkpoint an oracle run starts from."""
    model = tiny_backbone()
    realizer = Stage0Realizer.from_pretrained_model(model, RealizerSpec(mask_id=MASK, lora_rank=8, lora_alpha=8.0,
                                                                        time_dim=32))
    tok = CharTokenizer()
    cfg = Stage0Config(canvas_len=64, max_prompt_tokens=32, lora_rank=8, lora_alpha=8.0, lr=1e-2, warmup_steps=2,
                       steps=steps, schedule_steps=steps, micro_batch=4, grad_accum=1, log_every=1000,
                       save_every=1000, eval_every=1000, val_examples=8, write_fraction=0.0, prefix_fraction=0.5)
    stream = s0data.CorpusStream(tmp_path / "corpus", tok, cfg.canvas_len, cfg.max_prompt_tokens, 0.0, cfg.seed,
                                 chunk_rows=8)
    validation = s0data.load_validation(tmp_path / "corpus", tok, cfg.canvas_len, cfg.max_prompt_tokens, 8)
    trainer = Stage0Trainer(realizer, tok, cfg, stream, validation, tmp_path / "stage0", torch.device("cpu"))
    trainer.set_training(True)
    for _ in range(steps):
        trainer.train_step()
    trainer.save()
    return trainer


def _oracle_trainer(tmp_path, run: str, steps: int, init: dict, **overrides):
    tok = CharTokenizer()
    known = {f for f in Stage1Config.__dataclass_fields__}
    from text_vpa.stage1.train import RECIPE

    recipe = {k: v for k, v in init["config"].items() if k in RECIPE and k in known}
    cfg = Stage1Config(**{**recipe, "steps": steps, "schedule_steps": steps, "eval_every": 1000, "latent_dim": 16,
                          "bridge_width": 16, "bridge_heads": 4, "position_dim": 8, "max_slots": 8, "lr": 1e-2,
                          "init": "stage0/last.pt", **overrides})
    realizer = Stage0Realizer.from_pretrained_model(tiny_backbone(), RealizerSpec.from_dict(init["spec"]))
    realizer.load_trainable_state(init["trainable"])
    bridge, oracle = build_plan_modules(cfg, realizer)
    stream = PlanCorpusStream(tmp_path / "corpus", tok, cfg.canvas_len, cfg.max_prompt_tokens, cfg.write_fraction,
                              cfg.seed, chunk_rows=8, state=init["stream"], max_slots=cfg.max_slots)
    validation = load_plan_validation(tmp_path / "corpus", tok, cfg.canvas_len, cfg.max_prompt_tokens, 8, cfg.max_slots)
    return OracleTrainer(realizer, bridge, oracle, tok, cfg, stream, validation, tmp_path / run, torch.device("cpu"),
                         init={"path": "stage0/last.pt", "stage": "stage0", "step": int(init["step"])})


def test_oracle_training_starts_at_the_stage0_checkpoint_uses_the_plan_and_resumes_exactly(tmp_path):
    _structured_corpus(tmp_path / "corpus")
    stage0 = _stage0_checkpoint(tmp_path)
    init = torch.load(tmp_path / "stage0" / "last.pt", weights_only=True)
    trainer = _oracle_trainer(tmp_path, "a", 80, init)
    cfg = trainer.cfg
    assert cfg.prefix_fraction == 0.5 and cfg.canvas_len == 64 and cfg.init == "stage0/last.pt"     # the recipe
    assert trainer.stream.state() == stage0.stream.state()             # the corpus continues where Stage 0 stopped
    report = trainer.fit_memory()
    assert report["micro_batch"] == 4

    # Step 0: gates closed, so every loss is the Stage 0 checkpoint's own, with any plan.
    want = stage0.validate()
    before = trainer.validate()
    for key in ("val_loss", "val_loss_k0", "val_loss_k6", "val_loss_prefix", "val_loss_next",
                "val_loss_next_backbone"):
        assert math.isclose(before[key], want[key], rel_tol=1e-6), key
    assert math.isclose(before["val_loss_null"], want["val_loss"], rel_tol=1e-6)
    assert math.isclose(before["val_loss_shuffled_k0"], want["val_loss_k0"], rel_tol=1e-6)
    assert before["plan_gain_k0"] == 0.0 and before["plan_gain"] == 0.0

    backbone = {n: p.clone() for n, p in trainer.realizer.named_parameters() if not p.requires_grad}
    trainer.set_training(True)
    for _ in range(15):
        trainer.train_step()
    trainer.save()
    for _ in range(65):
        last = trainer.train_step()
    after = trainer.validate()
    assert math.isfinite(last["loss"]) and 0 < last["gate_mean"] <= last["gate_max"] < 1
    # The structure is used: with everything masked, the oracle's plan beats no plan and another function's plan
    # (nats per masked token; the tiny model's logits are flat, so all of its losses move slowly).
    assert after["plan_gain_k0"] > 0.02 and after["val_loss_k0"] < after["val_loss_shuffled_k0"] - 0.02
    assert after["val_loss_k0"] < before["val_loss_k0"] - 0.05
    assert after["val_loss_next_backbone"] == before["val_loss_next_backbone"]
    for name, p in trainer.realizer.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, backbone[name]), name                 # the released weights never change

    payload = torch.load(tmp_path / "a" / "last.pt", weights_only=True)
    assert payload["stage"] == "stage1-oracle" and payload["label"] == ORACLE_LABEL and payload["step"] == 15
    assert payload["init"] == {"path": "stage0/last.pt", "stage": "stage0", "step": 6}
    assert any("_u" in key for key in payload["bridge"])                # the spectral norm's vectors are saved
    resumed = _oracle_trainer(tmp_path, "b", 80, init)
    resumed.stream = PlanCorpusStream(tmp_path / "corpus", CharTokenizer(), 64, 32, 0.0, 0, chunk_rows=8,
                                      state=payload["stream"], max_slots=8)
    resumed.restore(payload)
    assert resumed.step == 15
    resumed.set_training(True)
    for _ in range(65):
        resumed.train_step()
    pairs = [(trainer.realizer.trainable_state(), resumed.realizer.trainable_state()),
             (trainer.bridge.state_dict(), resumed.bridge.state_dict()),
             (trainer.oracle.state_dict(), resumed.oracle.state_dict())]
    for first, second in pairs:
        assert set(first) == set(second)
        for name in first:
            torch.testing.assert_close(second[name], first[name], rtol=1e-5, atol=1e-7, msg=name)
    with pytest.raises(ValueError, match="plan"):
        build_plan_modules(Stage1Config(plan="oracle-latent"), tiny_realizer()[0])


# =============================================================================================
# Evaluation and report
# =============================================================================================
def _mbpp_tasks() -> list[EvalTask]:
    tasks = []
    for i, body in enumerate(("    return a + b", "  c = a * b\n  return (c) ", "\tif a > b:\n\t\treturn a\n\treturn b")):
        code = f"def op{i}(a, b):\r\n{body}"
        tasks.append(EvalTask(f"MBPP/{511 + i}", f"Operation {i}.\nassert op{i}(1, 2) == 0", "",
                              f"\n\nassert op{i}(1, 2) is not None\n", code.replace("\r\n", "\n")))
    return tasks


def test_oracle_samples_follow_stage0s_sampling_and_carry_the_label():
    realizer, bridge, _ = bridged()
    randomize_adapters(realizer)
    torch.manual_seed(0)
    oracle = StructureOracle(D).eval()
    tok, tasks = CharTokenizer(), _mbpp_tasks()
    cfg = SamplerConfig(num_steps=4, schedule="cosine", temperature=0.9)
    common = dict(canvas_len=64, num_samples=3, batch_size=2, seed=0, device=torch.device("cpu"))
    base, base_info = realizer_samples(realizer, tok, tasks, "mbpp", cfg, **common)
    texts, info, followed = oracle_samples(realizer, bridge, oracle, tok, tasks, "mbpp", cfg, max_slots=8, **common)
    assert texts == base                                               # closed gates: Stage 0's samples exactly
    assert info["label"] == ORACLE_LABEL and info["stage"] == "stage1-oracle" and info["plan"] == "oracle"
    assert info["sequential_depth"] == base_info["sequential_depth"] == 5
    assert info["references_not_parsed"] == 0 and info["references_truncated"] == 0
    assert len(followed) == 3 and all(len(group) == 3 for group in followed)
    assert [group[0]["plan_lines"] for group in followed] == [2, 3, 4]   # of the normalised references
    flat = [row["kind_match"] for group in followed for row in group]
    assert math.isclose(info["kind_match"], sum(flat) / 9, abs_tol=1e-4)

    plans = TaskPlans(bridge, oracle, tok, [t.reference for t in tasks], 64, 8, "oracle", torch.device("cpu"))
    assert plans.texts[1] == "def op1(a, b):\n    c = a * b\n    return c\n"
    assert [KINDS[k] for k in plans.slots[2].kind] == ["def", "if", "return", "return"]
    assert list(plans.slots[2].depth) == [0, 1, 2, 1]                   # tabs in the reference: normalised
    assert plans.truncated == 0 and plans.slots[2].length == (15, 14, 17, 13) and len(plans.texts[2]) == 59
    short = TaskPlans(bridge, oracle, tok, [t.reference for t in tasks], 40, 8, "oracle", torch.device("cpu"))
    assert short.truncated == 2 and short.slots[0] == plans.slots[0]     # references longer than the canvas: cut
    assert short.slots[2].length == (15, 14, 10) and short.slots[2].lines == 4 and short.kinds[2] == plans.kinds[2]
    shuffled = TaskPlans(bridge, oracle, tok, [t.reference for t in tasks], 64, 8, "shuffled", torch.device("cpu"))
    assert [shuffled.source(i) for i in range(3)] == [1, 2, 0]
    with torch.no_grad():
        bridge.gate.fill_(0.7)
    opened, _, _ = oracle_samples(realizer, bridge, oracle, tok, tasks, "mbpp", cfg, max_slots=8, **common)
    nulled, null_info, _ = oracle_samples(realizer, bridge, oracle, tok, tasks, "mbpp", cfg, max_slots=8,
                                          variant="null", **common)
    assert opened != base and nulled != opened and null_info["plan"] == "null" and null_info["label"] == ORACLE_LABEL
    with pytest.raises(ValueError, match="MBPP"):
        oracle_samples(realizer, bridge, oracle, tok, [_tasks()[0]], "humaneval", cfg, **common)
    bridge.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        oracle_samples(realizer, bridge, oracle, tok, tasks, "mbpp", cfg, **common)


def _write_row(root, tag: str, solved: int, **summary) -> None:
    folder = root / "mbpp" / tag
    folder.mkdir(parents=True)
    with open(folder / "samples.jsonl", "w", encoding="utf-8") as handle:
        for task in range(10):
            for j in range(5):
                handle.write(json.dumps({"task_id": f"MBPP/{task}", "sample": j, "passed": task < solved}) + "\n")
    (folder / "summary.json").write_text(json.dumps({"pass@1": solved / 10, "problems": 10, "samples_per_problem": 5,
                                                     "parse_rate": 0.9, "mean_sequential_depth": 9.0, **summary}))


def test_step1_gate_and_the_report_refuse_to_drop_the_label(tmp_path):
    from text_vpa.stage1 import report
    from text_vpa.stage1.report import bootstrap_difference, delivery_verdict, gate_reading, step1_gate

    assert step1_gate(0.6, 0.52, 0.7).startswith("clearly at least 0.5") and "build the planner" in step1_gate(0.5, 0.3, 0.7)
    assert step1_gate(0.5, 0.3, 0.7).startswith("at least 0.5")
    assert step1_gate(0.3, 0.15, 0.45).startswith("clearly between 0.1 and 0.5")
    assert step1_gate(0.3, 0.05, 0.45).startswith("between 0.1 and 0.5") and "step 2" in step1_gate(0.3, 0.05, 0.45)
    assert step1_gate(0.05, 0.0, 0.09).startswith("clearly under 0.1") and step1_gate(0.09, 0.0, 0.2).startswith("under 0.1")
    assert step1_gate(float("nan"), 0, 0).startswith("undecided")
    assert "reaches" in delivery_verdict(0.93) and "NOT shown" in delivery_verdict(0.6)
    assert "not evaluated" in delivery_verdict(None)
    # A band under 0.5 is read only once the delivery check has passed; at least 0.5 is read regardless.
    assert gate_reading(0.05, 0.0, 0.09, 0.14).startswith("NOT READ") and gate_reading(0.3, 0.1, 0.4, None).startswith("NOT READ")
    assert gate_reading(0.05, 0.0, 0.09, 0.95) == step1_gate(0.05, 0.0, 0.09)
    assert gate_reading(0.6, 0.52, 0.7, 0.2) == step1_gate(0.6, 0.52, 0.7)
    rng = np.random.default_rng(0)
    a, b = (rng.random(300) < 0.5).astype(float), (rng.random(300) < 0.3).astype(float)
    diff, low, high = bootstrap_difference(a, b)
    assert math.isclose(diff, a.mean() - b.mean()) and low < diff < high and 0.05 < low and high < 0.35

    label = {"stage": "stage1-oracle", "label": ORACLE_LABEL, "step": 40}
    sweep, reference = tmp_path / "sweep", tmp_path / "stage0-sweep"
    _write_row(sweep, "oracle-K8", 3, plan="oracle", kind_match=0.7, line_count_match=0.5, **label)
    _write_row(sweep, "null-K8", 1, plan="null", kind_match=0.3, line_count_match=0.1, **label)
    _write_row(sweep, "oracle-K256-left_to_right", 5, plan="oracle", kind_match=0.95, line_count_match=0.9, **label)
    _write_row(reference, "ar", 6)
    _write_row(reference, "realizer-K8", 0)
    (tmp_path / "run").mkdir()
    with open(tmp_path / "run" / "metrics.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 20, "train/gate_mean": 0.12}) + "\n")
        handle.write(json.dumps({"step": 20, "eval/probe_pass@1": 0.1, "eval/plan_gain_k0": 0.4,
                                 "eval/probe_kind_match": 0.6}) + "\n")
    assert report.main(["--eval", str(sweep), "--run", str(tmp_path / "run"), "--reference", str(reference)]) == 0
    result = json.loads((sweep / "report.json").read_text())
    assert math.isclose(result["gate"]["rho"], 0.5) and result["gate"]["verdict"].startswith("at least 0.5")
    assert result["gate"]["band"] == result["gate"]["verdict"] and result["gate"]["problems"] == 10
    assert result["label"] == ORACLE_LABEL and result["delivery"]["kind_match"] == 0.95
    row = next(r for r in result["comparison"] if r["k"] == 8 and r["order"] == "exact kernel")
    assert row["oracle"] == 0.3 and row["null"] == 0.1 and row["no_plan"] == 0.0
    assert math.isclose(row["oracle_minus_null"][0], 0.2)
    assert result["curve"] == [{"step": 20, "probe_pass@1": 0.1, "probe_kind_match": 0.6, "plan_gain_k0": 0.4,
                                "plan_gain": None, "val_loss_k0": None, "val_loss_null_k0": None, "gate_mean": 0.12}]
    text = (sweep / "report.md").read_text()
    assert text.splitlines()[0].endswith(f"({ORACLE_LABEL})") and text.count(ORACLE_LABEL) >= 4
    assert "rho = 0.500" in text and "the structure reaches the realizer" in text and "10 problems x 5 samples" in text

    # The smoke run's situation: the oracle solves nothing at K = 8 and the plan has not reached the realizer.
    smoke = tmp_path / "smoke"
    _write_row(smoke, "oracle-K8", 0, plan="oracle", kind_match=0.26, line_count_match=0.1, **label)
    _write_row(smoke, "oracle-K256-left_to_right", 4, plan="oracle", kind_match=0.14, line_count_match=0.0,
               order="left_to_right", num_steps=256, canvas_len=256, **label)
    assert report.main(["--eval", str(smoke), "--reference", str(reference)]) == 0
    read = json.loads((smoke / "report.json").read_text())["gate"]
    assert read["band"].startswith("clearly under 0.1") and read["verdict"].startswith("NOT READ")
    assert "run step 2" not in (smoke / "report.md").read_text().split("## Did the structure")[0]

    # Without the baseline on the same problems the gate is not evaluated, and says why.
    assert report.main(["--eval", str(sweep), "--out", str(tmp_path / "alone")]) == 0
    assert json.loads((tmp_path / "alone" / "report.json").read_text())["gate"]["reason"].startswith("no baseline")
    # An oracle row that lost its label is refused, and so is a report text without it.
    summary = sweep / "mbpp" / "null-K8" / "summary.json"
    stripped = {k: v for k, v in json.loads(summary.read_text()).items() if k not in ("label", "stage")}
    summary.write_text(json.dumps(stripped))
    with pytest.raises(ValueError, match="label"):
        report.main(["--eval", str(sweep), "--reference", str(reference)])
    with pytest.raises(ValueError, match="label"):
        report._write(tmp_path / "x", text.replace(ORACLE_LABEL, "a result"), result)
    assert not (tmp_path / "x" / "report.md").exists()
