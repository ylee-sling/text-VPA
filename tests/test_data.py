# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Independent tests of the data pipeline (``text_vpa.data``): tokenizers, local loaders, the serialised
PR-JEPA example (structural view y, slot index, mode u and readout targets; Table 1, Sec. 5.1), the
bucketed collator, the FS-DFM mixture-path corruption (Eq. 11) and the length-bucketed, rank-sharded
sampler.

Every expected value is written by hand from the module docstrings and the paper (the byte tokenizer is
plain UTF-8, so the serialised views can be spelled out byte by byte); no reference is built with the
function under test. Monte Carlo checks use 5 standard errors.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re

import pytest
import torch

from text_vpa.data.ast_parser import ModeExtractor, TreeSitterASTExtractor
from text_vpa.data.dataset import (
    LengthBucketSampler,
    MixtureCorruption,
    PRJEPACollator,
    PRJEPADataset,
    build_dataloader,
    load_humaneval,
    load_jsonl,
    load_mbpp,
)
from text_vpa.data.synthetic import CodeRecord, synthetic_corpus
from text_vpa.data.tokenization import ByteTokenizer, tokenizer_from_spec, train_bpe_tokenizer

# ---------------------------------------------------------------------------------------------
# Hand-written layout (tokenization.py docstring): 256 UTF-8 bytes, then the specials in this order.
# ---------------------------------------------------------------------------------------------
SPECIAL_NAMES = ("[PAD]", "[MASK]", "[NULL_PLAN]", "[SLOT_IMPORT]", "[SLOT_FUNCTION]", "[SLOT_CLASS]",
                 "[SLOT_VARIABLE]", "[SLOT_BLOCK]", "[SLOT_DIRECTIVE]")
SLOT_KIND_NAME = {1: "IMPORT", 2: "FUNCTION", 3: "CLASS", 4: "VARIABLE", 5: "BLOCK", 6: "DIRECTIVE"}
BYTE_ID = {name: 256 + i for i, name in enumerate(SPECIAL_NAMES)}
S = 256 + len(SPECIAL_NAMES)
PAD, MASK, NULL_PLAN = BYTE_ID["[PAD]"], BYTE_ID["[MASK]"], BYTE_ID["[NULL_PLAN]"]
SLOT_ID = {kind: BYTE_ID[f"[SLOT_{name}]"] for kind, name in SLOT_KIND_NAME.items()}
TEXT_KINDS = (1, 2, 3)          # readout slot texts only for IMPORT, FUNCTION, CLASS

# Programs with their structural slots as the ast_parser docstring defines them (kind, normalised text), the
# "control" mode c and the library bucket b (u = 5 c + b in the "control_library" scheme).
P_IMPORT = "import math\n\n\ndef area(r: float) -> float:\n    return math.pi * r ** 2\n"
P_LOOP = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
P_EMPTY = "print(1)\n"
P_CLASS = ("from collections import Counter\nimport re\n\n\nclass Box(object):\n    size: int = 3\n\n"
           "    def grow(self, k):\n        while k > 0:\n            k -= 1\n        return [v for v in range(k)]\n")
P_BROKEN = "def broken(:\n    return\n"

CASES = {
    # name: (context, program, slots, control mode c, library bucket b)
    "import": ("Write area(r) for a circle.", P_IMPORT,
               [(1, "import math"), (2, "def area(r: float) -> float")], 1, 1),          # straight line, numeric
    "loop": ("Sum a list with a for loop.", P_LOOP,
             [(2, "def total(xs)"), (4, "s"), (5, "for x in xs"), (4, "x")], 2, 0),        # for loop, no import
    "empty": ("Print one.", P_EMPTY, [], 1, 0),                                            # no slot at all
    "class": ("A box that grows.", P_CLASS,
              [(1, "from collections import Counter"), (1, "import re"), (3, "class Box(object)"),
               (4, "size: int"), (2, "def grow(self, k)"), (5, "while k > 0")], 3, 2),     # while; containers < text
}


def ascii_word_bytes(text: str) -> list[int]:
    """Identifier-bearing byte tokens: ASCII letters and underscore (a lone byte >= 128 is not a character)."""
    return sorted({b for b in text.encode() if (65 <= b <= 90) or (97 <= b <= 122) or b == 95})


def expected_example(name: str, num_slots: int, max_text: int, max_context: int, max_structure: int) -> dict:
    """The serialised example, by hand: y = [SLOT_<KIND>] slot-text ... (first M slots), truncated."""
    context, program, slots, _, _ = CASES[name]
    kept = slots[:num_slots]
    y: list[int] = []
    index: list[int] = []
    kinds: list[int] = []
    texts: list[list[int]] = []
    for m in range(num_slots):
        if m < len(kept):
            kind, text = kept[m]
            piece = [SLOT_ID[kind]] + list(text.encode())
            y += piece
            index += [m] * len(piece)
            kinds.append(kind)
            raw = list(text.encode())
            texts.append((raw + [PAD] * max_text)[:max_text] if kind in TEXT_KINDS else [PAD] * max_text)
        else:
            kinds.append(0)
            texts.append([PAD] * max_text)
    if y:
        y, index = y[:max_structure], index[:max_structure]
    else:
        y, index = [NULL_PLAN], [-1]
    return {
        "context": list(context.encode())[:max_context], "target": list(program.encode()), "structure": y,
        "slot_index": index, "slot_kinds": kinds, "slot_texts": texts, "inventory": ascii_word_bytes(program),
        "num_slots": len(kept),
    }


def make_dataset(names, num_slots=6, max_text=16, max_context=64, max_output=512, max_structure=512,
                 scheme="control_library", **kwargs) -> PRJEPADataset:
    records = [CodeRecord(CASES[n][0], CASES[n][1], "python", f"case/{n}") for n in names]
    return PRJEPADataset(records, ByteTokenizer(), num_slots, max_context, max_output, max_structure, max_text,
                         scheme, **kwargs)


def round_up(n: int, multiple: int, cap: int) -> int:
    return min(cap, -(-n // multiple) * multiple)


@pytest.fixture(scope="module")
def bpe():
    records = synthetic_corpus(64, seed=0)
    return train_bpe_tokenizer([r.target for r in records] + [r.context for r in records], vocab_size=400)


# =============================================================================================
# Tokenizers
# =============================================================================================
def test_byte_tokenizer_layout() -> None:
    tok = ByteTokenizer()
    assert tok.vocab_size == S == 265
    for name, token_id in BYTE_ID.items():
        assert tok.special_token_id(name) == token_id
    assert (tok.pad_id, tok.mask_id, tok.null_plan_id) == (PAD, MASK, NULL_PLAN)
    for kind, token_id in SLOT_ID.items():
        assert tok.slot_token_id(kind) == token_id
    for bad in (0, 7):
        with pytest.raises(ValueError):
            tok.slot_token_id(bad)
    with pytest.raises(KeyError):
        tok.special_token_id("[CLS]")


def test_byte_round_trip_arbitrary_unicode() -> None:
    tok = ByteTokenizer()
    texts = ["", "def f(x):\n\treturn x  # ünïcöde ✓\n", "日本語 \U0001f40d é́",
             "\x00\x7f\r\n", "a" * 300]
    rng = random.Random(0)
    ranges = [(0x20, 0x7E), (0xA0, 0x24F), (0x370, 0x3FF), (0x4E00, 0x9FFF), (0x1F300, 0x1F6FF), (0x10000, 0x10FFFF)]
    for _ in range(60):
        chars = []
        for _ in range(rng.randint(1, 40)):
            lo, hi = rng.choice(ranges)
            chars.append(chr(rng.randint(lo, hi)))
        texts.append("".join(chars))
    for text in texts:
        ids = tok.encode(text)
        assert ids == list(text.encode("utf-8"))
        assert all(0 <= i < 256 for i in ids)                       # never a special token
        assert tok.decode(ids) == text


def test_bpe_round_trip_on_training_like_code(bpe) -> None:
    held_out = synthetic_corpus(24, seed=7)                         # same templates, other names / constants
    texts = [r.target for r in held_out] + [r.context for r in held_out]
    texts.append("def f(values):\n\treturn [v ** 2 for v in values]  # ünicode ✓\n")
    total_tokens = total_bytes = 0
    for text in texts:
        ids = bpe.encode(text)
        assert all(0 <= i < bpe.vocab_size for i in ids)
        assert not any(bpe.is_special(i) for i in ids)
        assert bpe.decode(ids) == text
        total_tokens += len(ids)
        total_bytes += len(text.encode())
    assert total_tokens < 0.6 * total_bytes                       # it is a trained BPE, not bytes


@pytest.mark.parametrize("kind", ["byte", "bpe"])
def test_specials_distinct_and_inside_vocabulary(kind, bpe) -> None:
    tok = ByteTokenizer() if kind == "byte" else bpe
    ids = [tok.special_token_id(name) for name in SPECIAL_NAMES]
    assert len(set(ids)) == len(SPECIAL_NAMES)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.special_ids == frozenset(ids)
    assert (tok.pad_id, tok.mask_id, tok.null_plan_id) == tuple(ids[:3])
    for kind_value, kind_name in SLOT_KIND_NAME.items():
        assert tok.slot_token_id(kind_value) == tok.special_token_id(f"[SLOT_{kind_name}]")
    assert len(tok.token_strings) == tok.vocab_size
    for name, i in zip(SPECIAL_NAMES, ids, strict=True):
        assert tok.is_special(i) and tok.token_strings[i] == name
    assert sum(tok.is_special(i) for i in range(tok.vocab_size)) == len(SPECIAL_NAMES)


@pytest.mark.parametrize("kind", ["byte", "bpe"])
def test_spec_rebuilds_identical_tokenizer(kind, bpe) -> None:
    tok = ByteTokenizer() if kind == "byte" else bpe
    spec = json.loads(json.dumps(tok.spec()))                     # stored in checkpoints as plain JSON
    rebuilt = tokenizer_from_spec(spec)
    assert rebuilt.vocab_size == tok.vocab_size
    assert [rebuilt.special_token_id(n) for n in SPECIAL_NAMES] == [tok.special_token_id(n) for n in SPECIAL_NAMES]
    for record in synthetic_corpus(10, seed=3):
        for text in (record.target, record.context):
            assert rebuilt.encode(text) == tok.encode(text)
    assert rebuilt.token_strings == tok.token_strings
    with pytest.raises(ValueError):
        tokenizer_from_spec({"type": "sentencepiece"})


@pytest.mark.parametrize("kind", [
    "byte",
    pytest.param("bpe"),
])
def test_literal_special_names_are_ordinary_text(kind, bpe) -> None:
    """Code may contain the text '[PAD]' (e.g. BERT vocabularies); it must stay text, not become [PAD]."""
    tok = ByteTokenizer() if kind == "byte" else bpe
    text = "specials = ['[PAD]', '[MASK]', '[SLOT_CLASS]']\n"
    ids = tok.encode(text)
    assert not any(tok.is_special(i) for i in ids)
    assert tok.decode(ids) == text


def test_decode_stops_at_first_pad_and_skips_specials() -> None:
    tok = ByteTokenizer()
    ids = list(b"abc") + [PAD] + list(b"def") + [PAD]
    assert tok.decode(ids) == "abc"
    assert tok.decode(torch.tensor(ids)) == "abc"                  # any iterable of ints
    assert tok.decode(ids, stop_at_pad=False) == "abcdef"
    assert tok.decode(ids, stop_at_pad=False, skip_special=False) == "abc[PAD]def[PAD]"
    ids = list(b"ab") + [MASK] + list(b"c") + [SLOT_ID[2], NULL_PLAN]
    assert tok.decode(ids) == "abc"
    assert tok.decode(ids, skip_special=False) == "ab[MASK]c[SLOT_FUNCTION][NULL_PLAN]"
    assert tok.decode([PAD] + list(b"xyz")) == ""


def test_decode_bpe_stops_at_pad(bpe) -> None:
    head, tail = "def area(r):\n    return r * r\n", "junk after the end"
    ids = bpe.encode(head) + [bpe.pad_id] + bpe.encode(tail)
    assert bpe.decode(ids) == head
    assert bpe.decode(ids, stop_at_pad=False) == head + tail
    assert bpe.decode(ids, stop_at_pad=False, skip_special=False) == head + "[PAD]" + tail


def test_token_strings_and_pad_to() -> None:
    tok = ByteTokenizer()
    strings = tok.token_strings
    assert len(strings) == S
    assert strings[ord("a")] == "a" and strings[ord("_")] == "_" and strings[ord("\n")] == "\n"
    assert strings[0xC3] == "�"                               # a lone UTF-8 lead byte is not a character
    assert strings[PAD] == "[PAD]" and strings[SLOT_ID[6]] == "[SLOT_DIRECTIVE]"
    assert tok.pad_to([1, 2, 3], 5) == [1, 2, 3, PAD, PAD]
    assert tok.pad_to([1, 2, 3], 2) == [1, 2]
    assert tok.pad_to([], 1) == [PAD]


# =============================================================================================
# Loaders
# =============================================================================================
def write_jsonl(path, rows, gz: bool = False) -> None:
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write((row if isinstance(row, str) else json.dumps(row)) + "\n")


HUMANEVAL = [
    {"task_id": "HumanEval/0", "prompt": 'def add(a: int, b: int) -> int:\n    """Sum."""\n',
     "canonical_solution": "    return a + b\n", "test": "def check(c):\n    assert c(1, 2) == 3\n", "entry_point": "add"},
    {"task_id": "HumanEval/1", "prompt": "def neg(x):\n", "canonical_solution": "    return -x\n",
     "test": "", "entry_point": "neg"},
]
MBPP = [
    {"task_id": 11, "text": "Write a function to double a number.", "code": "def double(x):\n    return 2 * x\n",
     "test_list": ["assert double(2) == 4", "assert double(0) == 0"]},
    {"task_id": 12, "text": "Write a function returning one.", "code": "def one():\n    return 1\n", "test_list": []},
]


def test_load_humaneval_fields(tmp_path) -> None:
    path = tmp_path / "he.jsonl"
    write_jsonl(path, HUMANEVAL)
    records = load_humaneval(path)
    assert len(records) == 2
    for record, row in zip(records, HUMANEVAL, strict=True):
        assert record.context == row["prompt"]                                  # signature + docstring
        assert record.target == row["prompt"] + row["canonical_solution"]       # the complete program
        assert record.task_id == row["task_id"] and record.language == "python"


def test_load_mbpp_context_includes_first_test(tmp_path) -> None:
    path = tmp_path / "mbpp.jsonl"
    write_jsonl(path, MBPP)
    first, second = load_mbpp(path)
    assert first.context.startswith(MBPP[0]["text"])
    assert MBPP[0]["test_list"][0] in first.context                    # fixes the function name
    assert MBPP[0]["test_list"][1] not in first.context                # only the first assertion
    assert first.target == MBPP[0]["code"] and first.task_id == "11" and first.language == "python"
    assert second.context == MBPP[1]["text"]                            # no test, no suffix
    assert second.target == MBPP[1]["code"]


def test_load_jsonl_generic_and_gzip(tmp_path) -> None:
    rows = [{"context": "Sum two ints.", "target": "def f(a, b):\n    return a + b\n"},
            {"context": "HLS add.", "target": "void k(int a) {}\n", "language": "cpp", "task_id": "hls/3"}]
    plain, packed = tmp_path / "data.jsonl", tmp_path / "data.jsonl.gz"
    write_jsonl(plain, rows)
    write_jsonl(packed, rows, gz=True)
    expected = [CodeRecord(rows[0]["context"], rows[0]["target"], "python", None),
                CodeRecord(rows[1]["context"], rows[1]["target"], "cpp", "hls/3")]
    assert load_jsonl(plain) == expected
    assert load_jsonl(packed) == expected
    he = tmp_path / "he.jsonl.gz"
    write_jsonl(he, HUMANEVAL, gz=True)
    assert [r.target for r in load_humaneval(he)] == [r["prompt"] + r["canonical_solution"] for r in HUMANEVAL]


@pytest.mark.parametrize("loader", [load_jsonl, load_humaneval, load_mbpp])
def test_loader_reports_line_number_of_malformed_json(tmp_path, loader) -> None:
    path = tmp_path / "bad.jsonl"
    good = {"context": "c", "target": "t", "task_id": "x", "prompt": "p", "canonical_solution": "s", "text": "t",
            "code": "c"}
    write_jsonl(path, [good, "", '{"context": "c", "target": '])          # physical line 3 is broken
    with pytest.raises(ValueError) as info:
        loader(path)
    message = str(info.value)
    assert path.name in message
    assert re.search(r"(:|line )3\b", message), message


def test_loader_missing_field_raises_value_error(tmp_path) -> None:
    path = tmp_path / "missing.jsonl"
    write_jsonl(path, [{"task_id": "HumanEval/9", "prompt": "def f():\n"}])
    with pytest.raises(ValueError):
        load_humaneval(path)
    write_jsonl(path, [{"context": "only a context"}])
    with pytest.raises(ValueError):
        load_jsonl(path)


# =============================================================================================
# Examples: structural view, slot index, mode, readout targets
# =============================================================================================
def test_extractor_slots_used_by_the_hand_expectations() -> None:
    """Precondition: the AST extractor yields the slots written above (ast_parser docstring)."""
    extractor = TreeSitterASTExtractor("python")
    for _, program, slots, _, _ in CASES.values():
        structure = extractor.parse(program)
        assert not structure.has_errors
        assert [(int(s.kind), s.text) for s in structure.slots] == slots


def test_example_serialisation_by_hand() -> None:
    names = ["import", "loop", "empty", "class"]
    dataset = make_dataset(names, num_slots=6, max_text=16, max_context=12)
    assert len(dataset) == 4 and dataset.skipped == {"too_long": 0, "unparseable": 0}
    for name, example in zip(names, dataset.examples, strict=True):
        want = expected_example(name, num_slots=6, max_text=16, max_context=12, max_structure=512)
        for key, value in want.items():
            assert getattr(example, key) == value, (name, key)
        assert len(example.slot_index) == len(example.structure)
        assert example.task_id == f"case/{name}"
    # spot checks of the hand layout
    imp = dataset.examples[0]
    assert imp.structure[0] == BYTE_ID["[SLOT_IMPORT]"] and imp.structure[12] == BYTE_ID["[SLOT_FUNCTION]"]
    assert imp.slot_index == [0] * 12 + [1] * 28
    empty = dataset.examples[2]
    assert empty.structure == [NULL_PLAN] and empty.slot_index == [-1] and empty.num_slots == 0
    assert empty.slot_kinds == [0] * 6 and empty.slot_texts == [[PAD] * 16] * 6


@pytest.mark.parametrize("scheme", ["control", "control_library"])
def test_mode_follows_the_mode_scheme_table(scheme) -> None:
    names = list(CASES)
    dataset = make_dataset(names, scheme=scheme)
    expected = [CASES[n][3] if scheme == "control" else 5 * CASES[n][3] + CASES[n][4] for n in names]
    assert [e.mode for e in dataset.examples] == expected                # [1, 2, 1, 3] or [6, 10, 5, 17]
    extractor, modes = TreeSitterASTExtractor("python"), ModeExtractor(scheme)
    assert [modes(extractor.parse(CASES[n][1])) for n in names] == expected


def test_structure_truncation_and_slot_limit() -> None:
    dataset = make_dataset(["import"], max_structure=20)
    example, want = dataset.examples[0], expected_example("import", 6, 16, 64, 20)
    # 12 tokens of the IMPORT piece, then [SLOT_FUNCTION] + 7 bytes of the signature: 20 in all
    assert example.structure == want["structure"] == [SLOT_ID[1]] + list(b"import math") + [SLOT_ID[2]] + list(b"def are")
    assert example.slot_index == [0] * 12 + [1] * 8
    assert example.slot_kinds == [1, 2, 0, 0, 0, 0]                  # readout targets are not truncated
    assert example.num_slots == 2
    dataset = make_dataset(["class"], num_slots=3)                    # M = 3 < 6 slots: the first three
    example, want = dataset.examples[0], expected_example("class", 3, 16, 64, 512)
    assert example.slot_kinds == [1, 1, 3] and example.num_slots == 3
    assert example.structure == want["structure"] and example.slot_index == want["slot_index"]
    assert max(example.slot_index) == 2


def test_skipping_rules_and_counters() -> None:
    limit = len(P_IMPORT.encode()) + 1                                # P_IMPORT fits with its closing [PAD]
    assert len(P_LOOP.encode()) + 1 <= limit and len(P_EMPTY.encode()) + 1 <= limit
    assert len(P_CLASS.encode()) + 1 > limit
    records = [CodeRecord(CASES[n][0], CASES[n][1]) for n in ("import", "loop", "class", "empty")]
    records.append(CodeRecord("broken", P_BROKEN))
    dataset = PRJEPADataset(records, ByteTokenizer(), 6, 64, limit, 512, 16, "control_library")
    assert dataset.skipped == {"too_long": 1, "unparseable": 1}
    assert [e.target for e in dataset.examples] == [list(p.encode()) for p in (P_IMPORT, P_LOOP, P_EMPTY)]
    tight = PRJEPADataset(records[:1], ByteTokenizer(), 6, 64, limit - 1, 512, 16, "control_library")
    assert len(tight) == 0 and tight.skipped["too_long"] == 1         # len + 1 > max_output_tokens
    kept = PRJEPADataset(records[4:], ByteTokenizer(), 6, 64, 512, 512, 16, "control_library",
                         skip_unparseable=False)
    assert len(kept) == 1 and kept.examples[0].mode == 0             # mode "unparseable" / "no_import"
    assert kept.skipped == {"too_long": 0, "unparseable": 0}


# =============================================================================================
# Collator
# =============================================================================================
def test_collator_shapes_padding_and_masks() -> None:
    names = ["import", "loop", "empty", "class"]
    m, length, bucket, cap = 6, 16, 32, 512
    dataset = make_dataset(names, num_slots=m, max_text=length)
    batch = PRJEPACollator(ByteTokenizer(), m, length, bucket, cap)(dataset.examples)
    wants = [expected_example(n, m, length, 64, 512) for n in names]
    b = len(names)
    lengths = [len(w["target"]) for w in wants]
    n = round_up(max(lengths) + 1, bucket, cap)
    m_c = max(len(w["context"]) for w in wants)
    t_len = max(len(w["structure"]) for w in wants)
    assert batch.batch_size == b
    assert batch.target.shape == (b, n) and batch.target.dtype == torch.int64
    assert batch.context.shape == batch.context_mask.shape == (b, m_c)
    assert batch.structure.shape == batch.structure_mask.shape == batch.slot_index.shape == (b, t_len)
    assert batch.slot_kinds.shape == batch.slot_text_mask.shape == (b, m)
    assert batch.slot_texts.shape == (b, m, length)
    assert batch.inventory.shape == (b, S) and batch.inventory.dtype == torch.float32
    assert batch.context_mask.dtype == batch.structure_mask.dtype == batch.slot_text_mask.dtype == torch.bool
    assert batch.w_t is None and batch.t is None and batch.h is None
    assert batch.target_lengths.tolist() == lengths
    assert batch.modes.tolist() == [5 * CASES[x][3] + CASES[x][4] for x in names]
    for i, w in enumerate(wants):
        k, s = len(w["context"]), len(w["structure"])
        assert batch.target[i].tolist() == w["target"] + [PAD] * (n - lengths[i])
        assert batch.context[i].tolist() == w["context"] + [PAD] * (m_c - k)
        assert batch.context_mask[i].tolist() == [True] * k + [False] * (m_c - k)
        assert batch.structure[i].tolist() == w["structure"] + [PAD] * (t_len - s)
        assert batch.structure_mask[i].tolist() == [True] * s + [False] * (t_len - s)
        assert batch.slot_index[i].tolist() == w["slot_index"] + [-1] * (t_len - s)
        assert batch.slot_kinds[i].tolist() == w["slot_kinds"]
        assert batch.slot_texts[i].tolist() == w["slot_texts"]
        assert batch.slot_text_mask[i].tolist() == [kind in TEXT_KINDS for kind in w["slot_kinds"]]
        assert torch.nonzero(batch.inventory[i]).flatten().tolist() == w["inventory"]
        assert set(batch.inventory[i].unique().tolist()) <= {0.0, 1.0}
    assert batch.slot_text_mask[3].tolist() == [True, True, True, False, True, False]      # IMPORT IMPORT CLASS ...


def test_collator_canvas_rounding_and_cap() -> None:
    tok = ByteTokenizer()
    length = len(P_IMPORT.encode())                                   # 71 bytes -> 72 with the closing [PAD]
    dataset = make_dataset(["import"], max_output=100)
    for bucket, cap, expected in [(16, 100, 80), (8, 100, 72), (1, 100, length + 1), (64, 100, 100), (32, 72, 72)]:
        assert expected == round_up(length + 1, bucket, cap)
        batch = PRJEPACollator(tok, 6, 16, bucket, cap)(dataset.examples)
        assert batch.target.shape[1] == expected, (bucket, cap)
        assert batch.target[0, length:].eq(PAD).all()                 # at least the closing [PAD]
    assert dataset.canvas_lengths(16) == [80] and dataset.canvas_lengths(64) == [100]


def test_collator_corruption_and_errors() -> None:
    tok = ByteTokenizer()
    dataset = make_dataset(["import", "loop"])
    mixture = MixtureCorruption(S, (2, 4, 8), "linear", "mask", MASK)
    batch = PRJEPACollator(tok, 6, 16, 32, 512, mixture)(dataset.examples)
    assert batch.w_t.shape == batch.target.shape and batch.t.shape == batch.h.shape == (2,)
    assert torch.all((batch.w_t == batch.target) | (batch.w_t == MASK))
    with pytest.raises(ValueError):
        PRJEPACollator(tok, 6, 16, 32, 512)([])
    with pytest.raises(ValueError):                                   # built with another M
        PRJEPACollator(tok, 5, 16, 32, 512)(dataset.examples)
    moved = batch.to("cpu")
    assert torch.equal(moved.target, batch.target) and moved.batch_size == 2


# =============================================================================================
# Mixture-path corruption (Eq. 11)
# =============================================================================================
def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_mixture_corruption_times_on_the_step_budget_grid() -> None:
    budgets, rows = (2, 4, 8), 6000
    w_1 = torch.randint(0, 256, (rows, 4), generator=gen(1))
    _, t, h = MixtureCorruption(S, budgets, "linear", "mask", MASK, "grid")(w_1, generator=gen(2))
    assert t.shape == h.shape == (rows,)
    k_budget = torch.round(1.0 / h.double())
    assert torch.allclose(1.0 / h.double(), k_budget, atol=1e-6)
    assert set(k_budget.long().tolist()) <= set(budgets)
    k = t.double() * k_budget                                          # t = k / K exactly
    assert torch.allclose(k, torch.round(k), atol=1e-5)
    assert ((torch.round(k) >= 0) & (torch.round(k) <= k_budget - 1)).all()
    for budget in budgets:                                             # K ~ U(budgets), k ~ U{0..K-1}
        for j in range(budget):
            p = 1.0 / (len(budgets) * budget)
            freq = ((k_budget == budget) & (torch.round(k) == j)).double().mean().item()
            assert abs(freq - p) <= 5 * math.sqrt(p * (1 - p) / rows), (budget, j)


@pytest.mark.parametrize("schedule, kappa", [("linear", lambda t: t),
                                              ("cosine", lambda t: 1.0 - math.cos(math.pi * t / 2.0))])
def test_mixture_corruption_mask_source_arrival_frequency(schedule, kappa) -> None:
    rows, n = 4000, 48
    w_1 = torch.randint(0, 256, (rows, n), generator=gen(3))           # never [MASK]
    w_t, t, _ = MixtureCorruption(S, (2, 4, 8), schedule, "mask", MASK)(w_1, generator=gen(4))
    arrived = w_t == w_1
    assert torch.all(arrived ^ (w_t == MASK))                          # w_1 where arrived, source [MASK] elsewhere
    values = torch.unique(t).tolist()
    assert len(values) == 8                                            # k/K for K in {2, 4, 8}: 0, 1/8, ..., 7/8
    for value in values:
        rows_t = t == value
        positions = int(rows_t.sum()) * n
        freq = arrived[rows_t].double().mean().item()
        k = kappa(value)
        assert abs(freq - k) <= 5 * math.sqrt(k * (1 - k) / positions) + 1e-12, (value, freq, k)


def test_mixture_corruption_uniform_source() -> None:
    rows, n = 4000, 48
    w_1 = torch.randint(0, 256, (rows, n), generator=gen(5))
    w_t, t, _ = MixtureCorruption(S, (2, 4, 8), "linear", "uniform")(w_1, generator=gen(6))
    assert w_t.min() >= 0 and w_t.max() < S
    for value in torch.unique(t).tolist():
        rows_t = t == value
        positions = int(rows_t.sum()) * n
        p = value + (1 - value) / S                                    # a uniform draw can coincide with w_1
        freq = (w_t[rows_t] == w_1[rows_t]).double().mean().item()
        assert abs(freq - p) <= 5 * math.sqrt(p * (1 - p) / positions), value
    source = w_t[t == 0].flatten()                                     # kappa_0 = 0: pure source, Uniform(V)
    counts = torch.bincount(source, minlength=S).double()
    expected = source.numel() / S
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert abs(chi2 - (S - 1)) <= 5 * math.sqrt(2 * (S - 1))


def test_mixture_corruption_uniform_time_sampling_and_determinism() -> None:
    rows = 20000
    corruption = MixtureCorruption(S, (2, 4, 8), "linear", "mask", MASK, "uniform")
    w_1 = torch.zeros(rows, 2, dtype=torch.long)
    _, t, h = corruption(w_1, generator=gen(7))
    assert ((t >= 0) & (t < 1)).all()
    assert abs(t.double().mean().item() - 0.5) <= 5 * math.sqrt(1 / 12 / rows)
    assert (t * 8 == torch.round(t * 8)).double().mean() < 0.01        # not the k/K grid
    for budget in (2, 4, 8):
        freq = (h == 1.0 / budget).double().mean().item()
        assert abs(freq - 1 / 3) <= 5 * math.sqrt(2 / 9 / rows)
    first = MixtureCorruption(S, (2, 4, 8), "linear", "uniform")(w_1[:50], generator=gen(8))
    second = MixtureCorruption(S, (2, 4, 8), "linear", "uniform")(w_1[:50], generator=gen(8))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


# =============================================================================================
# Length-bucketed, rank-sharded sampler
# =============================================================================================
LENGTHS = [16, 32, 16, 48, 32, 16, 16, 32, 48, 16, 64, 32, 16]          # 6 x 16, 4 x 32, 2 x 48, 1 x 64


def test_bucket_sampler_one_canvas_length_per_batch_and_full_coverage() -> None:
    sampler = LengthBucketSampler(LENGTHS, batch_size=2, shuffle=True, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 3 + 2 + 1 + 1               # ceil(count / 2) per length
    for batch in batches:
        assert 1 <= len(batch) <= 2
        assert len({LENGTHS[i] for i in batch}) == 1
    assert sorted(i for batch in batches for i in batch) == list(range(len(LENGTHS)))
    dropped = list(LengthBucketSampler(LENGTHS, batch_size=2, seed=0, drop_last=True))
    assert len(dropped) == 3 + 2 + 1 + 0 and all(len(b) == 2 for b in dropped)
    ordered = list(LengthBucketSampler(LENGTHS, batch_size=2, shuffle=False))
    assert sorted(ordered) == sorted([[0, 2], [5, 6], [9, 12], [1, 4], [7, 11], [3, 8], [10]])   # no shuffling
    assert ordered == list(LengthBucketSampler(LENGTHS, batch_size=2, shuffle=False, seed=99))
    for bad in ({"batch_size": 0}, {"world_size": 0}, {"rank": 2, "world_size": 2}):
        with pytest.raises(ValueError):
            LengthBucketSampler(LENGTHS, **{"batch_size": 2, **bad})


def test_bucket_sampler_set_epoch_reshuffles_deterministically() -> None:
    sampler = LengthBucketSampler(LENGTHS, batch_size=2, seed=11)
    sampler.set_epoch(3)
    first, again = list(sampler), list(sampler)
    assert first == again
    twin = LengthBucketSampler(LENGTHS, batch_size=2, seed=11)
    twin.set_epoch(3)
    assert list(twin) == first
    sampler.set_epoch(4)
    other = list(sampler)
    assert other != first
    assert sorted(i for b in other for i in b) == list(range(len(LENGTHS)))


@pytest.mark.parametrize("world_size", [2, 3])
def test_bucket_sampler_ranks_get_disjoint_batches_and_equal_counts(world_size) -> None:
    lengths = LENGTHS * 3
    single = LengthBucketSampler(lengths, batch_size=2, seed=5)
    single.set_epoch(1)
    global_batches = [tuple(b) for b in single]
    shards = []
    for rank in range(world_size):
        sampler = LengthBucketSampler(lengths, batch_size=2, seed=5, rank=rank, world_size=world_size)
        sampler.set_epoch(1)
        shards.append([tuple(b) for b in sampler])
        assert len(sampler) == len(shards[-1])
    counts = {len(s) for s in shards}
    assert counts == {len(global_batches) // world_size}               # equal counts, remainder dropped
    seen = [i for shard in shards for b in shard for i in b]
    assert len(seen) == len(set(seen))                                  # disjoint across ranks
    assert {b for shard in shards for b in shard} <= set(global_batches)
    for shard in shards:
        for batch in shard:
            assert len({lengths[i] for i in batch}) == 1


def test_build_dataloader_batches_have_the_members_canvas_length() -> None:
    records = synthetic_corpus(24, seed=2)
    dataset = PRJEPADataset(records, ByteTokenizer(), 16, 128, 256, 512, 48, "control_library")
    assert len(dataset) == 24
    loader = build_dataloader(dataset, PRJEPACollator(ByteTokenizer(), 16, 48, 32, 256), batch_size=3,
                              bucket_size=32, shuffle=True, seed=0)
    total = 0
    for batch in loader:
        width = batch.target.shape[1]
        assert {round_up(int(n) + 1, 32, 256) for n in batch.target_lengths} == {width}
        total += batch.batch_size
    assert total == len(dataset)
