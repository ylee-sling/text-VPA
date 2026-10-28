# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Tests of the training-corpus builder (``text_vpa.data.corpus``), against hand-written expectations:

  * a function is kept iff it is top level, undecorated, documented, has a real body and reads only its
    parameters, its locals, builtins and module-level imports; the imports it reads are kept, grouped
  * context = imports + signature + docstring (HumanEval prompt); target = imports + whole function
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
    Decontaminator,
    build_corpus,
    download_hf_parquet,
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
    with pytest.raises(urllib.error.HTTPError):
        download_hf_parquet("org/ds", "cfg", tmp_path, attempts=3)
    assert attempts["n"] == 1
