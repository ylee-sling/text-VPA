# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Tests of the training-corpus builder (``text_vpa.data.corpus``), against hand-written expectations:

  * a function is kept iff it is top level, undecorated, documented, has a real body and reads only its
    parameters, its locals, builtins and module-level imports; the imports it reads are kept, grouped
  * context = imports + signature + docstring (HumanEval prompt); target = imports + whole function
  * on request also methods of top-level non-test classes (``Class.method``, moved to column 0, tabs kept)
    and undocumented functions of >= 2 statements (context = imports + signature + leading comments)
  * duplicates up to formatting, comments and docstring are dropped; the first occurrence is kept
  * any function sharing a 13-token n-gram with an evaluation problem is dropped
  * only permissively licensed files are used; the output does not depend on the number of workers
  * the output is the JSONL format ``train.py`` loads, and builds a PRJEPADataset
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pytest

from text_vpa.data import corpus as corpus_module
from text_vpa.data.corpus import (
    PERMISSIVE_LICENSES,
    CorpusLimits,
    Decontaminator,
    build_corpus,
    download_hf_parquet,
    download_hf_shards,
    evaluation_texts,
    extract_functions,
    is_permissive,
    iter_sources,
)
from text_vpa.data.dataset import PRJEPADataset, load_jsonl
from text_vpa.data.tokenization import ByteTokenizer

FILE = '''"""A module."""
import math
import os.path
import numpy as np
from typing import List, Optional
from collections import Counter as C
from .local import helper

LIMIT = 10


def area(r: float) -> float:
    """Area of a circle of radius r."""
    return math.pi * r ** 2


def count_words(words: List[str]) -> Optional[str]:
    """Most common word, or None for an empty list."""
    # a comment
    if not words:
        return None
    return C(words).most_common(1)[0][0]


def uses_global(x):
    """Reads a module-level constant, so it is not self-contained."""
    return x + LIMIT


def uses_relative(x):
    """Calls a helper imported relatively (not resolvable)."""
    return helper(x)


def no_doc(x):
    return x


def stub(x):
    """Not implemented yet, only a stub."""
    raise NotImplementedError


@property
def decorated(x):
    """Decorated functions depend on their decorator."""
    return x


def test_area():
    """A test, not a specification of a function."""
    assert area(1) > 3


def writes_global(x):
    """Rebinds a module global."""
    global LIMIT
    LIMIT = x
    return x


def joined(a: str, b: str) -> str:
    """Join two path pieces with os.path and scale them with numpy, recursively if empty."""
    if not a:
        return joined(b, b)
    return os.path.join(a, b) + str(np.zeros(1))


class Box:
    def method(self):
        """Methods are not top-level functions."""
        return 1
'''

AREA_CONTEXT = 'import math\n\n\ndef area(r: float) -> float:\n    """Area of a circle of radius r."""\n'
AREA_TARGET = AREA_CONTEXT + "    return math.pi * r ** 2\n"


def test_extraction_keeps_self_contained_documented_functions() -> None:
    pairs, reasons = extract_functions(FILE)
    assert [p.name for p in pairs] == ["area", "count_words", "joined"]
    area, words, joined = pairs
    assert area.context == AREA_CONTEXT and area.target == AREA_TARGET
    assert words.context.startswith("from collections import Counter as C\nfrom typing import List, Optional\n\n\n"
                                    "def count_words(words: List[str]) -> Optional[str]:\n")
    assert "    # a comment\n" in words.target                    # the body is copied verbatim
    assert joined.context.startswith("import numpy as np\nimport os.path\n\n\ndef joined(")
    assert reasons == {"external_names": 2, "no_docstring": 1, "trivial_body": 1, "decorated": 1,
                       "test_function": 1, "global_state": 1}
    for p in pairs:                                            # the context is a prefix of the target
        assert p.target.startswith(p.context)
        compile(p.target, "<target>", "exec")


@pytest.mark.parametrize("source,reason", [
    ("def f(:\n    pass\n", "file_unparseable"),
    ("# Auto-generated file, do not edit\ndef f(x):\n    '''Doc string of twenty+ chars.'''\n    return x\n",
     "file_generated"),
    ("def f(x):\n    '''Short.'''\n    return x\n", "docstring_length"),
    ("def f(x):\n    '''A docstring long enough to keep.'''; return x\n", "inline_body"),
    ("def f(x):\n    '''A docstring long enough to keep.'''\n    ...\n", "trivial_body"),
    ("def f(x):\n    '''A docstring long enough to keep.'''\n" + "    x += 1\n" * 90 + "    return x\n",
     "too_many_lines"),
])
def test_extraction_drop_reasons(source: str, reason: str) -> None:
    pairs, reasons = extract_functions(source)
    assert pairs == [] and reasons == {reason: 1}


def test_extraction_test_name_filter_is_not_a_prefix_match() -> None:
    src = "def testable(x):\n    '''A docstring long enough to keep.'''\n    return x\n"
    assert [p.name for p in extract_functions(src)[0]] == ["testable"]
    for name in ("test", "test_x", "testFoo", "_test_y", "test2"):
        assert extract_functions(src.replace("testable", name))[1] == {"test_function": 1}


def test_locals_comprehensions_lambdas_and_exceptions_count_as_defined() -> None:
    src = '''def f(xs, *args, key=None, **kw):
    """Every name read here is bound inside the function."""
    total = 0
    for i, x in enumerate(xs):
        total += x
    ys = [y * 2 for y in xs if y]
    g = lambda z: z + total
    try:
        import json as js
        js.dumps(ys)
    except ValueError as err:
        raise RuntimeError(str(err)) from err
    def inner(q):
        return q
    match total:
        case [first, *rest]:
            return first, rest
        case {"k": v, **others}:
            return v, others
    return inner(g(total)), args, key, kw
'''
    pairs, reasons = extract_functions(src)
    assert [p.name for p in pairs] == ["f"] and not reasons


def test_dedup_key_ignores_formatting_comments_and_docstring() -> None:
    a = "def f(x):\n    '''First docstring, long enough.'''\n    return x+1\n"
    b = "def f(x):\n    '''Another docstring entirely.'''\n    # comment\n    return (x + 1)\n"
    c = "def f(x):\n    '''First docstring, long enough.'''\n    return x+2\n"
    ka, kb, kc = (extract_functions(s)[0][0].key for s in (a, b, c))
    assert ka == kb != kc


METHODS = '''import math


class Circle:
    """A circle."""

    def __init__(self, r):
        self.r = r

    def area(self) -> float:
        """Area of the circle, pi r squared."""
        return math.pi * self.r ** 2

    @property
    def diameter(self):
        """Twice the radius of the circle."""
        return 2 * self.r

    def describe(self):
        """A multi-line description of the circle."""
        text = """circle
    of radius"""
        return text + str(self.r)

    class Inner:
        def deep(self):
            """Nested classes are not visited."""
            return 0


class TestCircle:
    def check(self):
        """Methods of test classes are skipped."""
        return Circle(1).area()


class CircleTests:
    def check(self):
        """Methods of test classes are skipped too."""
        return 1


class Testimony:
    def said(self, words):
        """Not a test class, despite its name."""
        return words
'''


def test_methods_are_extracted_at_column_zero_only_when_asked() -> None:
    assert extract_functions(METHODS)[0] == []
    limits = CorpusLimits(include_methods=True)
    pairs, reasons = extract_functions(METHODS, limits)
    assert [(p.name, p.method, p.documented) for p in pairs] == [
        ("Circle.area", True, True), ("Circle.describe", True, True), ("Testimony.said", True, True)]
    area, describe, _ = pairs
    assert area.context == 'import math\n\n\ndef area(self) -> float:\n    """Area of the circle, pi r squared."""\n'
    assert area.target == area.context + "    return math.pi * self.r ** 2\n"
    assert describe.target == ('def describe(self):\n    """A multi-line description of the circle."""\n'
                               '    text = """circle\nof radius"""\n    return text + str(self.r)\n')
    assert reasons == {"no_docstring": 1, "decorated": 1}             # __init__, diameter
    for p in pairs:
        assert p.target.startswith(p.context)
        compile(p.target, "<target>", "exec")
    tabs = "class T:\n\tdef run(self, n):\n\t\t'''Count down from n to zero.'''\n\t\twhile n:\n\t\t\tn -= 1\n\t\treturn n\n"
    (run,), _ = extract_functions(tabs, limits)
    assert run.target == "def run(self, n):\n\t'''Count down from n to zero.'''\n\twhile n:\n\t\tn -= 1\n\treturn n\n"
    assert run.context == "def run(self, n):\n\t'''Count down from n to zero.'''\n"


UNDOCUMENTED = '''import os


def join_all(parts):
    # join the parts with the separator
    out = ""
    for p in parts:
        out = os.path.join(out, p)
    return out


def ident(x):
    return x


def guard(x):
    if x:
        return 1


def one_line(x): y = x; return y


def split_sig(a,
              b):
    total = a + b
    return total
'''


def test_undocumented_functions_get_a_signature_context_only_when_asked() -> None:
    assert extract_functions(UNDOCUMENTED)[1] == {"no_docstring": 5}
    pairs, reasons = extract_functions(UNDOCUMENTED, CorpusLimits(require_docstring=False))
    assert [(p.name, p.documented, p.method) for p in pairs] == [
        ("join_all", False, False), ("guard", False, False), ("split_sig", False, False)]
    join_all, guard, split_sig = pairs
    assert join_all.context == "import os\n\n\ndef join_all(parts):\n    # join the parts with the separator\n"
    assert join_all.target == join_all.context + ('    out = ""\n    for p in parts:\n'
                                                  "        out = os.path.join(out, p)\n    return out\n")
    assert guard.context == "def guard(x):\n" and guard.target == "def guard(x):\n    if x:\n        return 1\n"
    assert split_sig.context == "def split_sig(a,\n              b):\n"
    assert split_sig.target == split_sig.context + "    total = a + b\n    return total\n"
    assert reasons == {"trivial_body": 1, "inline_body": 1}             # ident (1 statement), one_line
    # documented functions keep their docstring context; the defaults' other filters still apply
    pairs, reasons = extract_functions(FILE, CorpusLimits(require_docstring=False, include_methods=True))
    assert [p.name for p in pairs] == ["area", "count_words", "joined", "Box.method"]
    assert reasons["trivial_body"] == 2 and "no_docstring" not in reasons    # stub; no_doc has 1 statement


def test_dedup_key_of_an_undocumented_function_covers_its_whole_body() -> None:
    limits = CorpusLimits(require_docstring=False)
    a = "def f(x):\n    y = x + 1\n    return y\n"
    b = "def f(x):\n    '''Increment x by one and return it.'''\n    y = x + 1\n    return y\n"
    c = "def f(x):\n    y = x + 2\n    return y\n"
    ka, kb, kc = (extract_functions(s, limits)[0][0].key for s in (a, b, c))
    assert ka == kb != kc
    method = "class A:\n    def f(x):\n        y = x + 1\n        return y\n"
    assert extract_functions(method, CorpusLimits(require_docstring=False, include_methods=True))[0][0].key == ka


def test_build_cli_widens_the_corpus_and_counts_what_it_kept(tmp_path) -> None:
    write_jsonl(tmp_path / "in.jsonl.gz", [{"content": FILE, "license": "mit", "path": "file.py"},
                                           {"content": METHODS, "license": "mit", "path": "methods.py"},
                                           {"content": UNDOCUMENTED, "license": "mit", "path": "undoc.py"}])
    common = ["build", "--source", f"jsonl:{tmp_path}/in.jsonl.gz", "--no-decontamination", "--workers", "1",
              "--val-fraction", "0"]
    assert corpus_module.main([*common, "--out", str(tmp_path / "strict")]) == 0
    assert corpus_module.main([*common, "--out", str(tmp_path / "wide"), "--include-methods",
                               "--allow-undocumented"]) == 0
    strict = json.loads((tmp_path / "strict" / "stats.json").read_text())
    wide = json.loads((tmp_path / "wide" / "stats.json").read_text())
    assert (strict["train"], strict["documented"], strict["methods"]) == (3, 3, 0)
    # FILE: area, count_words, joined, Box.method; METHODS: Circle.area, Circle.describe, Testimony.said
    # (Circle.__init__ has 1 statement); UNDOCUMENTED: join_all, guard, split_sig
    assert (wide["train"], wide["documented"], wide["methods"]) == (10, 7, 4)
    assert wide["limits"]["include_methods"] and not wide["limits"]["require_docstring"]
    rows = read_rows(tmp_path / "wide" / "train-00000.jsonl.gz")
    assert {row["task_id"] for row in rows if row["method"]} == {
        "file.py::Box.method", "methods.py::Circle.area", "methods.py::Circle.describe", "methods.py::Testimony.said"}
    assert sorted(row["task_id"] for row in rows if not row["documented"]) == [
        "undoc.py::guard", "undoc.py::join_all", "undoc.py::split_sig"]


def test_decontaminator_matches_shared_13_word_grams_only() -> None:
    reference = ('def close_pair(numbers: List[float], threshold: float) -> bool:\n'
                 '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
                 '    given threshold."""\n')
    decon = Decontaminator([reference], n=13)
    reformatted = ("def   close_pair( numbers:List[float],threshold : float )->bool:\n"
                   "  '''Check if in given list of numbers,\n are any two numbers closer'''")
    assert decon.contaminated(reformatted)                    # formatting does not hide an overlap
    same_signature = "def close_pair(numbers: List[float], threshold: float) -> bool:\n    return False\n"
    assert not decon.contaminated(same_signature)              # a shared signature alone is not contamination
    assert not Decontaminator([reference], n=40).contaminated(reformatted)   # fewer than 40 shared words
    with pytest.raises(ValueError):
        Decontaminator([], n=0)


def test_license_rules() -> None:
    assert is_permissive("mit") and is_permissive("apache-2.0,bsd-3-clause")
    assert not is_permissive("mit,gpl-3.0") and not is_permissive(None) and not is_permissive("")
    assert "gpl-3.0" not in PERMISSIVE_LICENSES and "agpl-3.0" not in PERMISSIVE_LICENSES


# ---------------------------------------------------------------------------------------------
# Sources and build
# ---------------------------------------------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def snippet(i: int) -> str:
    return (f"import math\n\n\ndef f{i}(x: float) -> float:\n    '''Scale x by the constant number {i}.'''\n"
            f"    return math.sqrt(x) * {i} + {i * i}\n")


ROWS = ([{"content": snippet(i), "license": "mit", "path": f"repo/a{i}.py"} for i in range(30)]
        + [{"content": snippet(100), "license": "gpl-3.0", "path": "repo/gpl.py"},
           {"content": snippet(101), "path": "repo/nolicense.py"},
           {"content": snippet(3).replace("'''Scale", "'''Multiply"), "license": "mit", "path": "repo/dup.py"},
           {"content": FILE, "licenses": ["mit"], "path": "repo/file.py"}])


def test_sources_from_jsonl_parquet_and_directories(tmp_path) -> None:
    write_jsonl(tmp_path / "part-0.jsonl.gz", ROWS[:10])
    write_jsonl(tmp_path / "part-1.jsonl.gz", ROWS[10:])
    rows = list(iter_sources(f"jsonl:{tmp_path}/part-*.jsonl.gz"))
    assert [r.origin for r in rows] == [r["path"] for r in ROWS]
    assert rows[-1].license == "mit" and rows[31].license is None and rows[30].license == "gpl-3.0"
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"code": [r["content"] for r in ROWS], "license": [r.get("license") for r in ROWS]})
    pq.write_table(table, tmp_path / "x.parquet")
    parquet = list(iter_sources(f"parquet:{tmp_path}/*.parquet", batch_rows=7))
    assert [r.content for r in parquet] == [r["content"] for r in ROWS]
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "m.py").write_text(snippet(7))
    (tmp_path / "src" / "notes.txt").write_text("not python")
    found = list(iter_sources(f"dir:{tmp_path / 'src'}", assume_license="MIT"))
    assert len(found) == 1 and found[0].license == "mit"
    with pytest.raises(FileNotFoundError):
        list(iter_sources(f"jsonl:{tmp_path}/missing-*.jsonl"))
    with pytest.raises(ValueError):
        list(iter_sources("ftp:somewhere"))


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_build_filters_dedups_decontaminates_and_splits(tmp_path) -> None:
    write_jsonl(tmp_path / "in.jsonl.gz", ROWS)
    contaminating = snippet(5)                                     # pretend f5 is an evaluation problem
    stats = build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "out", [contaminating],
                         val_fraction=0.2, shard_size=10)
    assert stats.files_seen == len(ROWS) and stats.files_license_rejected == 2 and stats.files_used == 32
    assert stats.contaminated == 1 and stats.duplicates == 1               # f5; dup.py repeats f3
    assert stats.train + stats.val == 30 - 1 + 3                           # 29 snippets + area/count/joined
    train = [row for p in sorted((tmp_path / "out").glob("train-*.jsonl.gz")) for row in read_rows(p)]
    val = read_rows(tmp_path / "out" / "val.jsonl.gz")
    assert len(train) == stats.train and len(val) == stats.val and stats.val > 0
    assert all(len(read_rows(p)) <= 10 for p in (tmp_path / "out").glob("train-*.jsonl.gz"))
    names = [row["task_id"].split("::")[1] for row in train + val]
    assert "f5" not in names and names.count("f3") == 1 and "f100" not in names and "f101" not in names
    assert {row["license"] for row in train + val} == {"mit"}
    summary = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert summary["decontaminated"] and summary["dropped"]["external_names"] == 2
    # the split depends only on the function, not on the run
    again = build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "again", [contaminating],
                         val_fraction=0.2, shard_size=10)
    assert read_rows(tmp_path / "again" / "val.jsonl.gz") == val and again == stats


def test_unknown_licenses_only_when_allowed(tmp_path) -> None:
    write_jsonl(tmp_path / "in.jsonl.gz", ROWS)
    stats = build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "out", allow_unknown_license=True,
                         val_fraction=0.0)
    names = [row["task_id"].split("::")[1] for row in read_rows(tmp_path / "out" / "train-00000.jsonl.gz")]
    assert "f101" in names and "f100" not in names and stats.val == 0


def test_build_output_does_not_depend_on_the_number_of_workers(tmp_path) -> None:
    write_jsonl(tmp_path / "in.jsonl.gz", ROWS)
    a = build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "w1", val_fraction=0.2, workers=1, chunk_files=3)
    b = build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "w2", val_fraction=0.2, workers=2, chunk_files=3)
    assert a == b
    for name in ("train-00000.jsonl.gz", "val.jsonl.gz"):
        assert read_rows(tmp_path / "w1" / name) == read_rows(tmp_path / "w2" / name)


def test_max_functions_stops_early(tmp_path) -> None:
    write_jsonl(tmp_path / "in.jsonl.gz", ROWS)
    stats = build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "out", max_functions=5, val_fraction=0.0)
    assert stats.train == 5


def test_output_loads_with_train_py_and_builds_examples(tmp_path) -> None:
    write_jsonl(tmp_path / "in.jsonl.gz", ROWS)
    build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "out", val_fraction=0.0)
    records = load_jsonl(tmp_path / "out" / "train-00000.jsonl.gz")
    assert records and all(r.language == "python" for r in records)
    dataset = PRJEPADataset(records, ByteTokenizer(), 8, 512, 1024, 1024, 32, "control_library")
    assert len(dataset) == len(records) and dataset.skipped == {"too_long": 0, "unparseable": 0}


def test_evaluation_texts_cover_prompts_solutions_and_descriptions(tmp_path) -> None:
    he = tmp_path / "he.jsonl"
    he.write_text(json.dumps({"task_id": "HumanEval/0", "prompt": "def f(x):\n", "canonical_solution": "    return x\n",
                              "test": "", "entry_point": "f"}) + "\n")
    mb = tmp_path / "mbpp.jsonl"
    mb.write_text(json.dumps({"task_id": 1, "text": "Write a function.", "code": "def g(): pass",
                              "test_list": ["assert g() is None"]}) + "\n")
    texts = evaluation_texts(he, mb)
    assert texts == ["def f(x):\n", "def f(x):\n    return x\n", "Write a function.", "def g(): pass",
                     "assert g() is None"]


def test_cli_refuses_a_build_without_evaluation_sets(tmp_path) -> None:
    write_jsonl(tmp_path / "in.jsonl.gz", ROWS)
    with pytest.raises(SystemExit):
        corpus_module.main(["build", "--source", f"jsonl:{tmp_path}/in.jsonl.gz", "--out", str(tmp_path / "o")])
    assert corpus_module.main(["build", "--source", f"jsonl:{tmp_path}/in.jsonl.gz", "--out", str(tmp_path / "o"),
                               "--no-decontamination", "--workers", "1"]) == 0


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_lists_and_fetches_parquet_files_resumably(tmp_path, monkeypatch) -> None:
    requested: list[tuple[str, dict]] = []
    listing = ["https://example.org/0.parquet", "https://example.org/1.parquet", "https://example.org/2.parquet"]

    def fake_urlopen(request, timeout=0):
        requested.append((request.full_url, dict(request.header_items())))
        if request.full_url.endswith("/parquet/Python-all/train"):
            return _Response(json.dumps(listing).encode())
        return _Response(b"PAR1" + request.full_url.encode())

    monkeypatch.setattr(corpus_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("HF_TOKEN", "secret")
    paths = download_hf_parquet("org/ds", "Python-all", tmp_path, max_files=2)
    assert [p.name for p in paths] == ["org__ds-Python-all-train-00000.parquet", "org__ds-Python-all-train-00001.parquet"]
    assert paths[1].read_bytes() == b"PAR1https://example.org/1.parquet"
    assert requested[0][0] == "https://huggingface.co/api/datasets/org/ds/parquet/Python-all/train"
    assert all(h.get("Authorization") == "Bearer secret" for _, h in requested)
    requested.clear()
    download_hf_parquet("org/ds", "Python-all", tmp_path, max_files=2)          # nothing to fetch again
    assert [url for url, _ in requested] == [requested[0][0]]
    assert not list(tmp_path.glob("*.part"))


def test_download_retries_transient_errors_and_truncated_files(tmp_path, monkeypatch) -> None:
    """A dropped connection and a short body (fewer bytes than Content-Length) are retried; the file is kept
    only when complete."""
    import urllib.error

    calls = {"listing": 0, "file": 0}
    sleeps: list[float] = []

    class Sized(_Response):
        def __init__(self, body: bytes, declared: int) -> None:
            super().__init__(body)
            self.headers = {"Content-Length": str(declared)}

    def flaky_urlopen(request, timeout=0):
        if request.full_url.endswith("/parquet/cfg/train"):
            calls["listing"] += 1
            if calls["listing"] == 1:
                raise urllib.error.URLError("connection reset")
            return _Response(json.dumps(["https://example.org/0.parquet"]).encode())
        calls["file"] += 1
        if calls["file"] == 1:
            return Sized(b"PAR1-trunc", 20)                         # 10 of 20 bytes, then the stream ends
        return Sized(b"PAR1-complete-file!!", 20)

    monkeypatch.setattr(corpus_module.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(corpus_module.time, "sleep", sleeps.append)
    paths = download_hf_parquet("org/ds", "cfg", tmp_path, attempts=3, base_delay=2.0)
    assert paths[0].read_bytes() == b"PAR1-complete-file!!"
    assert calls == {"listing": 2, "file": 2} and sleeps == [2.0, 2.0]          # one retry each
    assert not list(tmp_path.glob("*.part"))


def test_download_gives_up_after_the_attempts_and_does_not_retry_404(tmp_path, monkeypatch) -> None:
    import urllib.error

    attempts = {"n": 0}

    def always_down(request, timeout=0):
        attempts["n"] += 1
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(corpus_module.urllib.request, "urlopen", always_down)
    monkeypatch.setattr(corpus_module.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        download_hf_parquet("org/ds", "cfg", tmp_path, attempts=3)
    assert attempts["n"] == 3

    def not_found(request, timeout=0):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    attempts["n"] = 0
    monkeypatch.setattr(corpus_module.urllib.request, "urlopen", not_found)
    with pytest.raises(ValueError, match="no Parquet files"):
        download_hf_parquet("org/ds", "cfg", tmp_path, attempts=3)
    assert attempts["n"] == 2                     # the listing, then the export's tree; a 404 is never retried

    def forbidden(request, timeout=0):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    attempts["n"] = 0
    monkeypatch.setattr(corpus_module.urllib.request, "urlopen", forbidden)
    with pytest.raises(urllib.error.HTTPError):   # e.g. a gated dataset without HF_TOKEN: no fallback
        download_hf_parquet("org/ds", "cfg", tmp_path, attempts=3)
    assert attempts["n"] == 1


class _Paged(_Response):
    def __init__(self, body: bytes, link: str | None = None) -> None:
        super().__init__(body)
        self.headers = {"Link": link} if link else {}


@pytest.mark.parametrize("listing", ["404", "empty", "error-object"])
@pytest.mark.parametrize("layout", ["partial/train", "partial-train", "train"])
def test_download_falls_back_to_the_parquet_export_tree(tmp_path, monkeypatch, listing, layout) -> None:
    """When the /parquet listing has no files, the split's files come from the refs/convert/parquet tree, in
    shard order across pages, whichever folder layout the export uses; other splits and files are ignored."""
    import urllib.error

    tree = "https://huggingface.co/api/datasets/org/ds/tree/refs%2Fconvert%2Fparquet/Py-all?recursive=true"
    page2 = tree + "&cursor=abc"
    first = [{"type": "directory", "path": f"Py-all/{layout}"},
             {"type": "file", "path": f"Py-all/{layout}/0001.parquet", "size": 3},
             {"type": "file", "path": "Py-all/partial-test/0000.parquet", "size": 3},
             {"type": "file", "path": f"Py-all/{layout}/README.md", "size": 3}]
    second = [{"type": "file", "path": f"Py-all/{layout}/0000.parquet", "size": 3}, "not-an-entry"]
    requested: list[str] = []

    def fake_urlopen(request, timeout=0):
        url = request.full_url
        requested.append(url)
        if url.endswith("/parquet/Py-all/train"):
            if listing == "404":
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return _Response(b"[]" if listing == "empty" else b'{"error": "not ready"}')
        if url == tree:
            return _Paged(json.dumps(first).encode(), f'<{page2}>; rel="next"')
        if url == page2:
            return _Paged(json.dumps(second).encode())
        return _Response(b"PAR1" + url.encode())

    monkeypatch.setattr(corpus_module.urllib.request, "urlopen", fake_urlopen)
    paths = download_hf_parquet("org/ds", "Py-all", tmp_path)
    resolve = "https://huggingface.co/datasets/org/ds/resolve/refs%2Fconvert%2Fparquet/Py-all/"
    expected = [resolve + f"{layout}/0000.parquet", resolve + f"{layout}/0001.parquet"]
    assert requested[1:3] == [tree, page2]
    assert requested[3:] == expected
    assert [p.read_bytes() for p in paths] == [b"PAR1" + url.encode() for url in expected]
    assert corpus_module.parquet_tree_urls("org/ds", "Py-all") == expected


def test_parquet_tree_urls_of_an_unknown_dataset_is_empty(monkeypatch) -> None:
    import urllib.error

    def not_found(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(corpus_module.urllib.request, "urlopen", not_found)
    assert corpus_module.parquet_tree_urls("org/missing", "cfg") == []


# ---------------------------------------------------------------------------------------------
# Raw shards (download-shards), against a local Hub + CDN
# ---------------------------------------------------------------------------------------------
def _shard_table(index: int):
    pa = pytest.importorskip("pyarrow")
    rows = [  # (language, license, path) -> kept iff Python and permissive
        ("Python", "mit", f"pkg/a{index}.py"), ("Java", "mit", f"src/A{index}.java"),
        ("Python", "gpl-3.0", f"pkg/gpl{index}.py"), ("Python", "apache-2.0", f"pkg/b{index}.py"),
        ("Python", None, f"pkg/none{index}.py")]
    return pa.table({"code": [snippet(10 * index + i) for i in range(len(rows))],
                     "repo_name": [f"owner/repo{index}"] * len(rows), "path": [r[2] for r in rows],
                     "language": [r[0] for r in rows], "license": [r[1] for r in rows],
                     "size": [100] * len(rows)})


class _Hub:
    """A Hub (127.0.0.1) whose resolve URLs redirect to a CDN on another host (localhost), with faults:
    shard 1's first download is truncated, shard 2's first download is not Parquet, ``broken`` always fails."""

    def __init__(self, tmp_path: Path, broken: frozenset[int] = frozenset()) -> None:
        import http.server
        import threading

        pq = pytest.importorskip("pyarrow.parquet")
        self.files: dict[str, bytes] = {}
        for i in range(3):
            pq.write_table(_shard_table(i), tmp_path / f"shard{i}.parquet")
            self.files[f"data/train-{i:05d}-of-00003.parquet"] = (tmp_path / f"shard{i}.parquet").read_bytes()
        self.requests: list[tuple[str, str, str | None]] = []          # (host, path, Authorization)
        self.served: dict[str, int] = {}
        hub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                host = self.headers.get("Host", "")
                hub.requests.append((host.split(":")[0], self.path, self.headers.get("Authorization")))
                port = self.server.server_address[1]
                if self.path.startswith("/api/datasets/org/ds/tree/main/data"):
                    names = sorted(hub.files)
                    page = names[:2] if "cursor" not in self.path else names[2:]
                    body = json.dumps(([{"type": "directory", "path": "data"}] if "cursor" not in self.path else [])
                                      + [{"type": "file", "path": n, "size": len(hub.files[n]),
                                          "lfs": {"size": len(hub.files[n])}} for n in page]
                                      + [{"type": "file", "path": "data/README.md", "size": 3}]).encode()
                    self.send_response(200)
                    if "cursor" not in self.path:
                        self.send_header("Link", f'<http://127.0.0.1:{port}{self.path}&cursor=2>; rel="next"')
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/datasets/org/ds/resolve/main/"):
                    self.send_response(302)
                    self.send_header("Location", f"http://localhost:{port}/cdn/{self.path.split('/main/', 1)[1]}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path.startswith("/cdn/"):
                    name = self.path[len("/cdn/"):]
                    index = int(name.split("train-")[1][:5])
                    count = hub.served[name] = hub.served.get(name, 0) + 1
                    body = hub.files[name]
                    if index in broken:
                        self.send_error(500)
                        return
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if index == 1 and count == 1:
                        self.wfile.write(body[: len(body) // 2])             # the connection drops mid-file
                    elif index == 2 and count == 1:
                        self.wfile.write(b"X" * len(body))                   # complete, but not Parquet
                    else:
                        self.wfile.write(body)
                else:
                    self.send_error(404)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def no_proxy(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


def test_download_shards_filters_rows_resumes_and_keeps_the_token_off_the_cdn(tmp_path, monkeypatch,
                                                                                 no_proxy) -> None:
    hub = _Hub(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "secret")
    try:
        out = tmp_path / "shards"
        summary = download_hf_shards("org/ds", out, endpoint=hub.endpoint, workers=2, attempts=3, base_delay=0.0)
        assert summary["completed"] == 3 and summary["rows"] == 15 and summary["kept"] == 6 and not summary["failed"]
        assert sorted(p.name for p in out.glob("*.parquet")) == [f"train-{i:05d}-of-00003.parquet" for i in range(3)]
        assert not list((out / "raw").iterdir())                          # raw shards deleted, no .part left
        sources = list(iter_sources(f"parquet:{out}/*.parquet"))
        assert [(s.origin, s.license) for s in sources] == [
            (f"owner/repo{i}/pkg/{name}{i}.py", lic) for i in range(3) for name, lic in (("a", "mit"),
                                                                                       ("b", "apache-2.0"))]
        assert [s.content for s in sources] == [snippet(10 * i + j) for i in range(3) for j in (0, 3)]
        assert hub.served == {f"data/train-{i:05d}-of-00003.parquet": n for i, n in ((0, 1), (1, 2), (2, 2))}
        hub_auth = {auth for host, path, auth in hub.requests if host == "127.0.0.1"}
        cdn_auth = {auth for host, path, auth in hub.requests if host == "localhost"}
        assert hub_auth == {"Bearer secret"} and cdn_auth == {None}       # the token never reaches the CDN
        hub.requests.clear()
        again = download_hf_shards("org/ds", out, endpoint=hub.endpoint, workers=2, base_delay=0.0)
        assert again["kept"] == 6 and all(path.startswith("/api/") for _, path, _ in hub.requests)
        record = json.loads((out / "train-00001-of-00003.json").read_text())
        assert record["rows"] == 5 and record["kept"] == 2 and record["source"] == "data/train-00001-of-00003.parquet"
    finally:
        hub.close()


def test_download_shards_selection_and_failures(tmp_path, no_proxy) -> None:
    hub = _Hub(tmp_path, broken=frozenset({1}))
    try:
        out = tmp_path / "shards"
        with pytest.raises(RuntimeError, match="1 of 3 shard"):
            download_hf_shards("org/ds", out, endpoint=hub.endpoint, workers=3, attempts=2, base_delay=0.0)
        summary = json.loads((out / "shards.json").read_text())
        assert summary["completed"] == 2 and list(summary["failed"]) == ["data/train-00001-of-00003.parquet"]
        assert sorted(p.name for p in out.glob("*.parquet")) == ["train-00000-of-00003.parquet",
                                                                 "train-00002-of-00003.parquet"]
        assert hub.served["data/train-00001-of-00003.parquet"] == 2               # attempts = 2
        only = download_hf_shards("org/ds", tmp_path / "one", endpoint=hub.endpoint, shards="2",
                                  licenses=None, languages=None, base_delay=0.0)
        assert only["completed"] == 1 and only["kept"] == 5                 # no filter: every row
    finally:
        hub.close()
    assert corpus_module.parse_shard_selection("0-2,5,8-", 10) == [0, 1, 2, 5, 8, 9]
    assert corpus_module.parse_shard_selection(None, 3) == [0, 1, 2]
    for bad in ("3-1", "10", "-11"):
        with pytest.raises(ValueError):
            corpus_module.parse_shard_selection(bad, 10)


def test_filter_parquet_rows_selects_by_extension_without_a_language_column(tmp_path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    table = _shard_table(0).drop(["language"])
    pq.write_table(table, tmp_path / "in.parquet")
    assert corpus_module.filter_parquet_rows(tmp_path / "in.parquet", tmp_path / "out.parquet") == (5, 2)
    kept = pq.read_table(tmp_path / "out.parquet")
    assert kept.column_names == ["code", "license", "path", "repo_name"]
    assert kept.column("path").to_pylist() == ["pkg/a0.py", "pkg/b0.py"]
    pq.write_table(table.drop(["license"]), tmp_path / "nolicense.parquet")
    with pytest.raises(corpus_module.SchemaError):
        corpus_module.filter_parquet_rows(tmp_path / "nolicense.parquet", tmp_path / "x.parquet")


@pytest.mark.parametrize("cap,documented,undocumented", [(5, 5, 0), (8, 7, 1), (20, 7, 3)])
def test_prefer_documented_fills_the_cap_with_undocumented_functions_last(tmp_path, cap, documented,
                                                                           undocumented) -> None:
    write_jsonl(tmp_path / "in.jsonl.gz", [{"content": UNDOCUMENTED, "license": "mit", "path": "undoc.py"},
                                           {"content": FILE, "license": "mit", "path": "file.py"},
                                           {"content": METHODS, "license": "mit", "path": "methods.py"}])
    limits = CorpusLimits(include_methods=True, require_docstring=False)
    plain = build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "plain", limits=limits, max_functions=8,
                         val_fraction=0.0)
    assert (plain.train, plain.documented) == (8, 5)               # file order: the 3 undocumented come first
    stats = build_corpus([f"jsonl:{tmp_path}/in.jsonl.gz"], tmp_path / "out", limits=limits, max_functions=cap,
                         val_fraction=0.0, prefer_documented=True)
    assert (stats.train, stats.documented) == (documented + undocumented, documented)
    rows = read_rows(tmp_path / "out" / "train-00000.jsonl.gz")
    assert [row["documented"] for row in rows] == [True] * documented + [False] * undocumented
    assert [row["task_id"] for row in rows if not row["documented"]] == [
        "undoc.py::join_all", "undoc.py::guard", "undoc.py::split_sig"][:undocumented]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["stats.json", "train-00000.jsonl.gz"]
