# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of ``text_vpa.data.ast_parser``: the structural view y and the reference mode u (PR-JEPA Table 1,
Sec. 5.1).

Specification as checked here (module docstring of ``ast_parser``):

    slots         IMPORT / FUNCTION / CLASS / VARIABLE / BLOCK / DIRECTIVE records in source order, texts
                  whitespace-collapsed, without comments or bodies; depth = enclosing construct bodies
    VARIABLE      one slot per name per scope (module / function / class / lambda; C++ blocks) at its first
                  binding; parameters, imports, def / class names and global / nonlocal names are bindings
                  without a VARIABLE slot; comprehension variables have their own scope and no slot
    inventories   declared / referenced / attribute / call names, deduplicated in source order
    modes         "control" (6) and "control_library" (6 x 5 = 30, u = 5 c + b)

Every expected value below was derived by hand from the fixed input programs; byte offsets are computed from
the inputs with ``str.index`` / ``str.encode``, never with the code under test.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from text_vpa.data.ast_parser import (
    CONTROL_LABELS,
    EMPTY_SLOT,
    LIBRARY_LABELS,
    MODE_SCHEMES,
    ModeExtractor,
    Parameter,
    ProgramStructure,
    SlotKind,
    TreeSitterASTExtractor,
    library_bucket,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT, FUNCTION, CLASS, VARIABLE, BLOCK, DIRECTIVE = (
    SlotKind.IMPORT,
    SlotKind.FUNCTION,
    SlotKind.CLASS,
    SlotKind.VARIABLE,
    SlotKind.BLOCK,
    SlotKind.DIRECTIVE,
)


@pytest.fixture(scope="module")
def py() -> TreeSitterASTExtractor:
    return TreeSitterASTExtractor("python")


@pytest.fixture(scope="module")
def cpp() -> TreeSitterASTExtractor:
    return TreeSitterASTExtractor("cpp")


def kinds_texts_depths(structure: ProgramStructure) -> list[tuple[SlotKind, str, int]]:
    return [(slot.kind, slot.text, slot.depth) for slot in structure.slots]


def span_of(structure: ProgramStructure, text: str) -> tuple[int, int]:
    (slot,) = [s for s in structure.slots if s.text == text]
    return slot.start_byte, slot.end_byte


# ----------------------------------------------------------------------------------------------------------------
# Python: imports
# ----------------------------------------------------------------------------------------------------------------

IMPORTS = (
    "import numpy as np\n"
    "import os.path, sys as system\n"
    "from math import sqrt, pi\n"
    "from . import sibling\n"
    "from ..pkg.mod import (a as b,\n"
    "                       c)\n"
    "from x import *\n"
)


def test_python_imports(py: TreeSitterASTExtractor) -> None:
    s = py.parse(IMPORTS)
    assert not s.has_errors
    assert [(t.kind, t.text, t.name, t.depth) for t in s.slots] == [
        (IMPORT, "import numpy as np", "numpy", 0),
        (IMPORT, "import os.path, sys as system", "os.path", 0),
        (IMPORT, "from math import sqrt, pi", "math", 0),
        (IMPORT, "from . import sibling", ".", 0),
        (IMPORT, "from ..pkg.mod import a as b, c", "..pkg.mod", 0),  # parentheses and line break dropped
        (IMPORT, "from x import *", "x", 0),
    ]
    assert s.imports == ("numpy", "os.path", "sys", "math", ".", "..pkg.mod", "x")
    # bound names: aliases, the top-level package of a dotted import, from-imported names; never the path
    assert s.declared_identifiers == ("np", "os", "system", "sqrt", "pi", "sibling", "b", "c")
    assert s.referenced_identifiers == ()
    assert (s.slots[0].start_byte, s.slots[0].end_byte) == (0, 18)
    assert (s.slots[1].start_byte, s.slots[1].end_byte) == (19, 48)
    assert (s.slots[4].start_byte, s.slots[4].end_byte) == (IMPORTS.index("from .."), IMPORTS.index("c)") + 2)


def test_python_local_import_binds_in_function_scope(py: TreeSitterASTExtractor) -> None:
    s = py.parse("def f():\n    import json\n    json = json.dumps(1)\n    return json\n")
    # the import binds json in f's scope, so the later assignment is not a first binding (no VARIABLE slot)
    assert kinds_texts_depths(s) == [(FUNCTION, "def f()", 0), (IMPORT, "import json", 1)]
    assert s.imports == ("json",)
    assert s.attribute_names == ("dumps",)


# ----------------------------------------------------------------------------------------------------------------
# Python: signatures and parameters
# ----------------------------------------------------------------------------------------------------------------

SIGNATURES = (
    "from typing import Dict, List, Optional\n"
    "\n"
    "\n"
    "async def fetch(a, b: int = 3, *args, c, d: Optional[Dict[str, int]] = None, **kwargs) -> List[int]:\n"
    "    return [a]\n"
    "\n"
    "\n"
    "def g(a, /, b, *, c=1):\n"
    "    pass\n"
    "\n"
    "\n"
    "def h(x, y=2, z=3):\n"
    "    pass\n"
    "\n"
    "\n"
    "def k(*, key):\n"
    "    pass\n"
    "\n"
    "\n"
    "def t(*items: int, **options: str) -> None:\n"
    "    pass\n"
)


def test_python_parameter_kinds_and_arity(py: TreeSitterASTExtractor) -> None:
    s = py.parse(SIGNATURES)
    assert not s.has_errors
    fetch, g, h, k, t = s.functions
    assert fetch.text == (
        "async def fetch(a, b: int = 3, *args, c, d: Optional[Dict[str, int]] = None, **kwargs) -> List[int]"
    )
    assert fetch.parameters == (
        Parameter("a", None, False, "positional"),
        Parameter("b", "int", True, "positional"),
        Parameter("args", None, False, "var_positional"),
        Parameter("c", None, False, "keyword_only"),
        Parameter("d", "Optional[Dict[str, int]]", True, "keyword_only"),
        Parameter("kwargs", None, False, "var_keyword"),
    )
    assert (fetch.name, fetch.return_annotation, fetch.is_async, fetch.is_method, fetch.has_body) == (
        "fetch",
        "List[int]",
        True,
        False,
        True,
    )
    # min: positional without default (a); max: None because of *args / **kwargs
    assert (fetch.min_arity, fetch.max_arity, fetch.call_arity_bounds()) == (1, None, (1, None))

    # "/" and the bare "*" stay in the text but are not parameters
    assert g.text == "def g(a, /, b, *, c=1)"
    assert [(p.name, p.kind, p.has_default) for p in g.parameters] == [
        ("a", "positional", False),
        ("b", "positional", False),
        ("c", "keyword_only", True),
    ]
    assert (g.min_arity, g.max_arity) == (2, 2)
    assert (h.min_arity, h.max_arity) == (1, 3)
    assert (k.min_arity, k.max_arity) == (0, 0)  # a keyword-only parameter is not positional arity
    assert t.parameters == (
        Parameter("items", "int", False, "var_positional"),
        Parameter("options", "str", False, "var_keyword"),
    )
    assert (t.return_annotation, t.min_arity, t.max_arity) == ("None", 0, None)

    # the FUNCTION slot is the signature region, from "async" to the end of the return annotation
    start = SIGNATURES.index("async def")
    end = SIGNATURES.index("-> List[int]") + len("-> List[int]")
    assert span_of(s, fetch.text) == (start, end)
    assert [slot.text for slot in s.slots if slot.kind == FUNCTION] == [f.text for f in s.functions]


def test_python_multiline_signature_is_canonical(py: TreeSitterASTExtractor) -> None:
    source = (
        "def h(\n"
        "    x: int,  # the x\n"
        "    y: List[int] = [],\n"
        ") -> None:\n"
        "    # body comment\n"
        "    pass\n"
    )
    (h,) = py.parse(source).functions
    # comment removed, parameters re-joined with ", ", trailing comma and line breaks gone
    assert h.text == "def h(x: int, y: List[int] = []) -> None"
    assert h.parameters == (Parameter("x", "int", False, "positional"), Parameter("y", "List[int]", True, "positional"))


def test_parameter_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        Parameter("x", None, False, "positional_only")


# ----------------------------------------------------------------------------------------------------------------
# Python: classes, methods and nesting
# ----------------------------------------------------------------------------------------------------------------

METHODS = (
    "class Stack(Base, metaclass=Meta):\n"
    "    limit: int = 10\n"
    "\n"
    "    def __init__(self, items=None):\n"
    "        self.items = items or []\n"
    "\n"
    "    def push(self, x, *rest):\n"
    "        self.items.append(x)\n"
    "\n"
    "    @staticmethod\n"
    "    def make(n):\n"
    "        return Stack()\n"
    "\n"
    "    @classmethod\n"
    "    def build(cls, a, b=0):\n"
    "        return cls()\n"
    "\n"
    "\n"
    "def free(self, y):\n"
    "    return y\n"
)


def test_python_class_methods_and_call_arity(py: TreeSitterASTExtractor) -> None:
    s = py.parse(METHODS)
    assert [(t.kind, t.text, t.name, t.depth) for t in s.slots] == [
        (CLASS, "class Stack(Base, metaclass=Meta)", "Stack", 0),
        (VARIABLE, "limit: int", "limit", 1),
        (FUNCTION, "def __init__(self, items=None)", "__init__", 1),
        (FUNCTION, "def push(self, x, *rest)", "push", 1),
        (FUNCTION, "def make(n)", "make", 1),
        (FUNCTION, "def build(cls, a, b=0)", "build", 1),
        (FUNCTION, "def free(self, y)", "free", 0),
    ]
    assert s.classes == ("Stack",)
    by_name = {f.name: f for f in s.functions}
    assert {name: f.is_method for name, f in by_name.items()} == {
        "__init__": True,
        "push": True,
        "make": True,
        "build": True,
        "free": False,
    }
    # obj.m(...) supplies self / cls: both bounds drop by one; a staticmethod and a plain function keep theirs
    assert (by_name["__init__"].min_arity, by_name["__init__"].max_arity) == (1, 2)
    assert by_name["__init__"].call_arity_bounds() == (0, 1)
    assert by_name["push"].call_arity_bounds() == (1, None)
    assert by_name["make"].call_arity_bounds() == (1, 1)
    assert by_name["build"].call_arity_bounds() == (1, 2)
    assert by_name["free"].call_arity_bounds() == (2, 2)
    # "self.items = ..." assigns an attribute: no VARIABLE slot, "items" is an attribute name
    assert s.attribute_names == ("items", "append")
    assert s.calls == ("append", "Stack", "cls")
    assert "metaclass" not in s.referenced_identifiers  # a keyword-argument name is not a reference
    assert span_of(s, "limit: int") == (METHODS.index("limit"), METHODS.index("limit: int") + len("limit: int"))


def test_python_nested_function_is_not_a_method(py: TreeSitterASTExtractor) -> None:
    source = (
        "class A:\n"
        "    def method(self):\n"
        "        def helper(x):\n"
        "            return x\n"
        "        return helper(1)\n"
    )
    s = py.parse(source)
    assert kinds_texts_depths(s) == [
        (CLASS, "class A", 0),
        (FUNCTION, "def method(self)", 1),
        (FUNCTION, "def helper(x)", 2),
    ]
    assert [(f.name, f.is_method) for f in s.functions] == [("method", True), ("helper", False)]
    assert s.features.num_functions == 2


# ----------------------------------------------------------------------------------------------------------------
# Python: variable slots, blocks, depth and inventories
# ----------------------------------------------------------------------------------------------------------------

BLOCKS = (
    "def outer(n):\n"
    "    total = 0\n"
    "    for i, (j, k) in enumerate(pairs):\n"
    "        total += i\n"
    "        if i > j:\n"
    "            total = total - 1\n"
    "        elif i < 0:\n"
    "            pass\n"
    "        else:\n"
    "            k = 2\n"
    "    while total < n:\n"
    "        total += 1\n"
    "    def inner(m):\n"
    "        nonlocal total\n"
    "        total = m\n"
    "        return m\n"
    "    with open(path) as fh, lock:\n"
    "        data = fh.read()\n"
    "    try:\n"
    "        x: float = 1.0\n"
    "    except ValueError as err:\n"
    "        raise\n"
    "    finally:\n"
    "        x = 2.0\n"
    "    return inner(total)\n"
)


def test_python_block_headers_slot_order_and_depth(py: TreeSitterASTExtractor) -> None:
    s = py.parse(BLOCKS)
    assert not s.has_errors
    assert kinds_texts_depths(s) == [
        (FUNCTION, "def outer(n)", 0),
        (VARIABLE, "total", 1),
        (BLOCK, "for i, (j, k) in enumerate(pairs)", 1),
        (VARIABLE, "i", 1),  # for targets sit at the depth of their for header
        (VARIABLE, "j", 1),
        (VARIABLE, "k", 1),
        (BLOCK, "if i > j", 2),  # "total = total - 1": total is already bound in outer -> no slot
        (BLOCK, "elif i < 0", 2),  # elif / else at the depth of their if
        (BLOCK, "else", 2),  # "k = 2": already bound
        (BLOCK, "while total < n", 1),
        (FUNCTION, "def inner(m)", 1),  # nonlocal total binds total in inner's scope: "total = m" has no slot
        (BLOCK, "with open(path) as fh, lock", 1),
        (VARIABLE, "fh", 1),
        (VARIABLE, "data", 2),
        (BLOCK, "try", 1),
        (VARIABLE, "x: float", 2),
        (BLOCK, "except ValueError as err", 1),
        (VARIABLE, "err", 1),
        (BLOCK, "finally", 1),  # "x = 2.0": already bound in outer
    ]
    starts = [slot.start_byte for slot in s.slots]
    assert starts == sorted(starts)
    for_start = BLOCKS.index("for i,")
    assert span_of(s, "for i, (j, k) in enumerate(pairs)") == (
        for_start,
        BLOCKS.index("enumerate(pairs)") + len("enumerate(pairs)"),
    )
    assert span_of(s, "x: float") == (BLOCKS.index("x: float"), BLOCKS.index("x: float") + len("x: float"))
    assert span_of(s, "try") == (BLOCKS.index("try"), BLOCKS.index("try") + 3)

    assert s.declared_identifiers == ("outer", "n", "total", "i", "j", "k", "inner", "m", "fh", "data", "x", "err")
    assert s.referenced_identifiers == (
        "enumerate", "pairs", "total", "i", "j", "n", "m", "open", "path", "lock", "fh", "float", "ValueError",
        "inner",
    )  # fmt: skip
    assert s.attribute_names == ("read",)
    assert s.calls == ("enumerate", "open", "read", "inner")
    assert s.undefined_names() == ("pairs", "path", "lock")
    f = s.features
    assert (f.has_for, f.has_while, f.has_try, f.has_recursion, f.has_comprehension) == (True, True, True, False, False)
    assert (f.num_functions, f.max_loop_depth, f.num_top_level_statements) == (2, 1, 1)


def test_python_variable_slots_first_binding_per_scope(py: TreeSitterASTExtractor) -> None:
    source = (
        "count = 0\n"
        "count = 1\n"
        "a, *rest = items\n"
        "class C:\n"
        "    count: int = 5\n"
        "    def m(self, count):\n"
        "        count = 2\n"
        "        other = count\n"
        "        return other\n"
        "squares = [v * v for v in rest]\n"
        "v = 3\n"
        "if (n := len(rest)) > 1:\n"
        "    pass\n"
    )
    s = py.parse(source)
    assert [(t.kind, t.text, t.name, t.depth) for t in s.slots] == [
        (VARIABLE, "count", "count", 0),  # the second "count = 1" is not a first binding
        (VARIABLE, "a", "a", 0),
        (VARIABLE, "rest", "rest", 0),  # starred unpacking target
        (CLASS, "class C", "C", 0),
        (VARIABLE, "count: int", "count", 1),  # class scope is a new scope
        (FUNCTION, "def m(self, count)", "m", 1),  # parameter count: "count = 2" is no first binding
        (VARIABLE, "other", "other", 2),
        (VARIABLE, "squares", "squares", 0),  # the comprehension variable v has its own scope, no slot
        (VARIABLE, "v", "v", 0),
        (BLOCK, "if (n := len(rest)) > 1", None, 0),
        (VARIABLE, "n", "n", 0),  # a walrus binds in the enclosing scope
    ]
    # the slot of v is the module-level "v = 3", not the comprehension variable before it
    assert span_of(s, "v") == (source.index("v = 3"), source.index("v = 3") + 1)
    assert span_of(s, "count") == (0, 5)
    assert s.declared_identifiers == ("count", "a", "rest", "C", "m", "self", "other", "squares", "v", "n")
    assert s.features.has_comprehension


def test_python_inventories_skip_strings_comments_keywords(py: TreeSitterASTExtractor) -> None:
    source = (
        "import numpy as np\n"
        "# a comment mentions ghost\n"
        "def run(data, *, scale=1.0):\n"
        "    '''a docstring mentions phantom'''\n"
        "    out = np.linalg.norm(data, axis=0) * scale\n"
        "    label = f\"{out!r} for {data.shape}\"\n"
        "    return print(label, sep='')\n"
    )
    s = py.parse(source)
    assert s.declared_identifiers == ("np", "run", "data", "scale", "out", "label")
    # f-string replacement fields are code; comments, docstrings and keyword-argument names are not
    assert s.referenced_identifiers == ("np", "data", "scale", "out", "print", "label")
    assert s.attribute_names == ("linalg", "norm", "shape")
    assert s.calls == ("norm", "print")
    norm, printed = s.call_sites
    assert (norm.name, norm.num_positional, norm.keywords, norm.receiver) == ("norm", 1, ("axis",), "np.linalg")
    assert (printed.name, printed.num_positional, printed.keywords, printed.receiver) == ("print", 1, ("sep",), None)
    assert s.undefined_names() == ()


def test_python_call_sites_count_star_arguments(py: TreeSitterASTExtractor) -> None:
    (call,) = py.parse("f(1, 2, *xs, key=3, **kw)\n").call_sites
    assert (call.name, call.num_positional, call.keywords, call.has_star_args, call.has_star_kwargs) == (
        "f",
        2,
        ("key",),
        True,
        True,
    )
    (generator_call,) = py.parse("total = sum(x for x in range(3))\n").call_sites[:1]
    assert (generator_call.name, generator_call.num_positional) == ("sum", 1)


def test_undefined_names_reports_typos_not_builtins(py: TreeSitterASTExtractor) -> None:
    s = py.parse("def area(radius):\n    return 3.14 * radus ** 2\n\nprint(len(area(2)))\n")
    assert s.referenced_identifiers == ("radus", "print", "len", "area")
    assert s.undefined_names() == ("radus",)
    assert s.undefined_names(builtins=frozenset()) == ("radus", "print", "len")
    assert py.parse("if __name__ == '__main__':\n    print(__file__)\n").undefined_names() == ()


# ----------------------------------------------------------------------------------------------------------------
# Python: recursion and comprehensions
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n", True),
        ("class T:\n    def walk(self, n):\n        return self.walk(n - 1)\n", True),
        ("class T:\n    @classmethod\n    def make(cls, n):\n        return cls.make(n)\n", True),
        ("def f(n):\n    g = lambda k: f(k)\n    return g(n)\n", True),  # a lambda inside f calls f by name
        ("def f(x):\n    return g(x)\n", False),
        ("class A:\n    def run(self):\n        return other.run()\n", False),  # another object's method
        ("class A:\n    def run(self):\n        return run()\n", False),  # a global run, not the method
        ("def outer():\n    def inner():\n        return outer()\n    return inner\n", False),  # indirect
    ],
)
def test_python_recursion_detection(py: TreeSitterASTExtractor, source: str, expected: bool) -> None:
    assert py.parse(source).features.has_recursion is expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x = 1\n", 0),
        ("for i in a:\n    for j in b:\n        while c:\n            pass\n", 3),
        ("for i in a:\n    pass\nfor j in b:\n    pass\n", 1),  # siblings, not nested
        ("for i in a:\n    def f():\n        for j in b:\n            pass\n", 1),  # a body restarts the count
        ("while a:\n    pass\nelse:\n    for j in b:\n        pass\n", 1),  # a loop's else is outside it
    ],
)
def test_python_max_loop_depth(py: TreeSitterASTExtractor, source: str, expected: int) -> None:
    assert py.parse(source).features.max_loop_depth == expected


@pytest.mark.parametrize(
    ("source", "expected", "variables"),
    [
        ("ys = [x * x for x in xs]\n", True, ["ys"]),
        ("ys = {x for x in xs}\n", True, ["ys"]),
        ("ys = {x: 1 for x in xs}\n", True, ["ys"]),
        ("total = sum(x for x in xs)\n", True, ["total"]),
        ("for x in xs:\n    print(x)\n", False, ["x"]),
    ],
)
def test_python_comprehension_detection(
    py: TreeSitterASTExtractor, source: str, expected: bool, variables: list[str]
) -> None:
    s = py.parse(source)
    assert s.features.has_comprehension is expected
    assert "x" in s.declared_identifiers  # comprehension variables are declared ...
    assert [t.text for t in s.slots if t.kind == VARIABLE] == variables  # ... but get no VARIABLE slot


def test_comments_and_line_continuations_inside_headers_are_removed(
    py: TreeSitterASTExtractor, cpp: TreeSitterASTExtractor
) -> None:
    python_source = (
        "if (ready and  # both flags\n"
        "        steady):\n"
        "    pass\n"
        "while ready and \\\n"
        "        steady:\n"
        "    break\n"
        "def f(x: Dict[str,  # the key\n"
        "              int]) -> None:\n"
        "    pass\n"
    )
    s = py.parse(python_source)
    assert [t.text for t in s.slots] == [
        "if (ready and steady)",
        "while ready and steady",
        "def f(x: Dict[str, int]) -> None",  # the comment sits inside the annotation's span
    ]
    assert s.functions[0].parameters[0].annotation == "Dict[str, int]"
    c = cpp.parse("void g(int n /* count */) {\n    for (int i = 0; /* bound */ i < n; ++i) { }\n}\n")
    assert [t.text for t in c.slots] == ["void g(int n)", "for (int i = 0; i < n; ++i)", "int i"]


# ----------------------------------------------------------------------------------------------------------------
# slot_records, byte offsets, determinism
# ----------------------------------------------------------------------------------------------------------------


def test_slot_records_truncates_and_pads(py: TreeSitterASTExtractor) -> None:
    s = py.parse("import os\nx = 1\ny = 2\n")
    assert [t.text for t in s.slots] == ["import os", "x", "y"]
    assert s.slot_records(2) == s.slots[:2]
    padded = s.slot_records(5)
    assert len(padded) == 5 and padded[:3] == s.slots
    for empty in padded[3:]:
        assert empty == EMPTY_SLOT
        assert (empty.kind, empty.text, empty.name, empty.start_byte, empty.end_byte, empty.depth) == (
            SlotKind.EMPTY,
            "",
            None,
            -1,
            -1,
            0,
        )
    assert s.slot_records(0) == ()
    with pytest.raises(ValueError):
        s.slot_records(-1)


def test_byte_offsets_refer_to_utf8(py: TreeSitterASTExtractor) -> None:
    source = "# h\u00e9llo w\u00f6rld\nnombre_\u00f1 = '\u00e9'\n"
    (slot,) = py.parse(source).slots
    start = len("# h\u00e9llo w\u00f6rld\n".encode())  # 16 bytes for 14 characters
    assert slot.text == "nombre_\u00f1"
    assert (slot.start_byte, slot.end_byte) == (start, start + len("nombre_\u00f1".encode()))


def test_parse_is_deterministic_and_accepts_bytes(py: TreeSitterASTExtractor) -> None:
    assert py.parse(BLOCKS) == py.parse(BLOCKS)
    assert py.parse(BLOCKS.encode()) == py.parse(BLOCKS)


# ----------------------------------------------------------------------------------------------------------------
# Malformed Python never raises
# ----------------------------------------------------------------------------------------------------------------


def test_unclosed_paren_keeps_valid_parts(py: TreeSitterASTExtractor) -> None:
    source = "import os\n\ndef good(a):\n    return a\n\nx = foo(1, 2\n\ndef later(b):\n    return b\n"
    s = py.parse(source)
    assert s.has_errors
    assert s.imports == ("os",)
    assert [slot.text for slot in s.slots][:2] == ["import os", "def good(a)"]
    assert s.functions[0].name == "good"
    assert "def" not in s.referenced_identifiers  # keywords from error recovery are dropped
    assert ModeExtractor("control")(s) == CONTROL_LABELS.index("unparseable")


def test_stray_indentation_is_an_error(py: TreeSitterASTExtractor) -> None:
    source = "import os\ndef f(a):\n    return a\n        y = 2\nclass K:\n    pass\n"
    s = py.parse(source)
    # tree-sitter recovers silently; CPython rejects the unexpected indent
    assert s.has_errors
    assert any("IndentationError" in w for w in s.warnings)
    assert [t.text for t in s.slots] == ["import os", "def f(a)", "y", "class K"]
    lenient = TreeSitterASTExtractor("python", validate_syntax=False).parse(source)
    assert not lenient.has_errors and lenient.slots == s.slots
    assert py.parse("def f():\n        x = 1\n    y = 2\n").has_errors  # dedent to no outer level


@pytest.mark.parametrize(
    "source",
    [
        bytes(range(256)).decode("latin-1"),
        bytes(range(256)) * 3,  # invalid UTF-8
        b"\xff\xfe\x00\x00import os\n",
        "x = 1\n\udcff\udc80 = 2\n",  # lone surrogates cannot be encoded strictly
        "x = 1\x00\ny = 2\n",
        "(" * 500,
        ")]}" * 100 + "def",
    ],
)
def test_binary_garbage_never_raises(py: TreeSitterASTExtractor, source: str | bytes) -> None:
    s = py.parse(source)
    assert s.has_errors
    assert ModeExtractor()(s) // len(LIBRARY_LABELS) == CONTROL_LABELS.index("unparseable")
    assert all(slot.text and "\n" not in slot.text for slot in s.slots)


@pytest.mark.parametrize("opener, closer", [("(", ")"), ("[", "]")])
def test_deep_nesting_never_raises(py: TreeSitterASTExtractor, opener: str, closer: str) -> None:
    source = "x = " + opener * 500 + "1" + closer * 500 + "\n"
    s = py.parse(source)
    # tree-sitter accepts 500 levels; CPython's tokenizer stops at 200 ("too many nested parentheses")
    assert s.has_errors and any("nested parentheses" in w for w in s.warnings)
    assert [t.text for t in s.slots] == ["x"]
    assert not TreeSitterASTExtractor("python", validate_syntax=False).parse(source).has_errors
    deeper = py.parse("y = " + opener * 5000 + closer * 5000 + "\n")  # far deeper than Python's recursion limit
    assert deeper.has_errors and [t.text for t in deeper.slots] == ["y"]


def test_node_budget_stops_traversal(py: TreeSitterASTExtractor) -> None:
    source = "".join(f"v{i} = {i}\n" for i in range(50))
    full = py.parse(source)
    assert len(full.slots) == 50 and not full.has_errors
    partial = TreeSitterASTExtractor("python", max_nodes=40).parse(source)
    assert partial.has_errors
    assert any("max_nodes=40" in w for w in partial.warnings)
    assert 0 < len(partial.slots) < 50
    assert partial.slots == full.slots[: len(partial.slots)]


def test_syntax_warnings_do_not_leak_or_raise(py: TreeSitterASTExtractor) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # as under ``python -W error``
        s = py.parse("x = '\\d'\n")  # an invalid escape: a SyntaxWarning, not an error
    assert not s.has_errors and [t.text for t in s.slots] == ["x"]


# ----------------------------------------------------------------------------------------------------------------
# C++ / HLS-C
# ----------------------------------------------------------------------------------------------------------------

HLS_KERNEL = (
    "#include <ap_int.h>\n"
    '#include "kernel.h"\n'
    "\n"
    "struct Pixel { ap_uint<8> r; ap_uint<8> g; };\n"
    "\n"
    "int add(int a, int b = 1);\n"
    "\n"
    "void kernel(ap_uint<8> x[16], int n, const float *p, int &out) {\n"
    "#pragma HLS INTERFACE m_axi port=x\n"
    "#pragma HLS PIPELINE II=1  // pipelined\n"
    "    int acc = 0, tmp;\n"
    "    for (int i = 0; i < n; ++i) {\n"
    "#pragma HLS UNROLL factor=4\n"
    "        acc += x[i];\n"
    "    }\n"
    "    while (acc > 255) { acc -= 256; }\n"
    "    if (acc > n) { out = acc; } else if (acc < 0) { out = 0; } else { out = add(acc, n); }\n"
    "}\n"
    "\n"
    "int fact(int k) { return k <= 1 ? 1 : k * fact(k - 1); }\n"
)


def test_cpp_hls_kernel_slots(cpp: TreeSitterASTExtractor) -> None:
    s = cpp.parse(HLS_KERNEL)
    assert not s.has_errors
    assert [(t.kind, t.text, t.name, t.depth) for t in s.slots] == [
        (IMPORT, "#include <ap_int.h>", "ap_int.h", 0),
        (IMPORT, '#include "kernel.h"', "kernel.h", 0),
        (CLASS, "struct Pixel", "Pixel", 0),
        (VARIABLE, "ap_uint<8> r", "r", 1),
        (VARIABLE, "ap_uint<8> g", "g", 1),
        (FUNCTION, "int add(int a, int b = 1)", "add", 0),  # a prototype
        (FUNCTION, "void kernel(ap_uint<8> x[16], int n, const float *p, int &out)", "kernel", 0),
        (DIRECTIVE, "#pragma HLS INTERFACE m_axi port=x", None, 1),
        (DIRECTIVE, "#pragma HLS PIPELINE II=1", None, 1),  # trailing comment removed
        (VARIABLE, "int acc", "acc", 1),
        (VARIABLE, "int tmp", "tmp", 1),
        (BLOCK, "for (int i = 0; i < n; ++i)", None, 1),
        (VARIABLE, "int i", "i", 1),
        (DIRECTIVE, "#pragma HLS UNROLL factor=4", None, 2),
        (BLOCK, "while (acc > 255)", None, 1),
        (BLOCK, "if (acc > n)", None, 1),
        (BLOCK, "else if (acc < 0)", None, 1),  # one header, at the depth of its if
        (BLOCK, "else", None, 1),
        (FUNCTION, "int fact(int k)", "fact", 0),
    ]
    add, kernel, fact = s.functions
    assert add.parameters == (Parameter("a", "int", False, "positional"), Parameter("b", "int", True, "positional"))
    assert (add.has_body, add.min_arity, add.max_arity, add.return_annotation) == (False, 1, 2, "int")
    assert kernel.parameters == (
        Parameter("x", "ap_uint<8>[16]", False, "positional"),
        Parameter("n", "int", False, "positional"),
        Parameter("p", "const float *", False, "positional"),
        Parameter("out", "int &", False, "positional"),
    )
    assert (kernel.return_annotation, kernel.has_body, kernel.is_method) == ("void", True, False)
    assert fact.return_annotation == "int"
    assert span_of(s, kernel.text) == (HLS_KERNEL.index("void kernel"), HLS_KERNEL.index(") {\n#pragma") + 1)
    assert s.imports == ("ap_int.h", "kernel.h")
    assert s.classes == ("Pixel",)
    assert s.declared_identifiers == (
        "Pixel", "r", "g", "add", "a", "b", "kernel", "x", "n", "p", "out", "acc", "tmp", "i", "fact", "k",
    )  # fmt: skip
    assert s.referenced_identifiers == ("i", "n", "acc", "x", "out", "add", "k", "fact")
    assert s.calls == ("add", "fact")
    assert s.undefined_names() == ()
    f = s.features
    assert (f.has_for, f.has_while, f.has_recursion, f.has_try, f.has_comprehension) == (True, True, True, False, False)
    assert (f.num_functions, f.max_loop_depth) == (2, 1)  # the prototype is not a definition
    assert ModeExtractor("control")(s) == CONTROL_LABELS.index("recursion")
    assert ModeExtractor()(s) == 5 * CONTROL_LABELS.index("recursion") + LIBRARY_LABELS.index("other")


def test_cpp_class_methods_typedef_define(cpp: TreeSitterASTExtractor) -> None:
    source = (
        "class Derived : public Base {\n"
        "public:\n"
        "    int value;\n"
        "    int get(int k) const { return this->get(k - 1) + value; }\n"
        "    Derived(int v) : value(v) {}\n"
        "};\n"
        "typedef ap_uint<12> data_t;\n"
        "#define N 16  // array size\n"
        "using word_t = ap_int<16>;\n"
    )
    s = cpp.parse(source)
    assert [(t.kind, t.text, t.name, t.depth) for t in s.slots] == [
        (CLASS, "class Derived : public Base", "Derived", 0),
        (VARIABLE, "int value", "value", 1),
        (FUNCTION, "int get(int k) const", "get", 1),
        (FUNCTION, "Derived(int v)", "Derived", 1),
        (VARIABLE, "typedef ap_uint<12> data_t", "data_t", 0),
        (VARIABLE, "#define N 16", "N", 0),
        (VARIABLE, "using word_t = ap_int<16>", "word_t", 0),
    ]
    get, ctor = s.functions
    assert (get.is_method, get.return_annotation, ctor.is_method, ctor.return_annotation) == (True, "int", True, None)
    assert s.features.has_recursion  # this->get(...) inside get
    assert s.attribute_names == ("get",)
    assert s.call_sites[0].receiver == "this"


def test_cpp_loops_switch_try(cpp: TreeSitterASTExtractor) -> None:
    source = (
        "void f(int n, std::vector<int> &v) {\n"
        "    do { n--; } while (n > 0);\n"
        "    for (const auto &e : v) { n += e; }\n"
        "    switch (n) {\n"
        "    case 1: n = 0; break;\n"
        "    default: break;\n"
        "    }\n"
        "    try { g(n); } catch (const std::exception &err) { }\n"
        "}\n"
    )
    s = cpp.parse(source)
    assert kinds_texts_depths(s) == [
        (FUNCTION, "void f(int n, std::vector<int> &v)", 0),
        (BLOCK, "do while (n > 0)", 1),
        (BLOCK, "for (const auto &e : v)", 1),
        (VARIABLE, "const auto &e", 1),
        (BLOCK, "switch (n)", 1),
        (BLOCK, "case 1", 2),
        (BLOCK, "default", 2),
        (BLOCK, "try", 1),
        (BLOCK, "catch (const std::exception &err)", 1),
    ]
    (f,) = s.functions
    assert f.parameters[1] == Parameter("v", "std::vector<int> &", False, "positional")
    assert (s.features.has_for, s.features.has_while, s.features.has_try) == (True, True, True)
    assert ModeExtractor("control")(s) == CONTROL_LABELS.index("while_loop")  # do-while counts as while


def test_cpp_block_scopes_and_redeclaration(cpp: TreeSitterASTExtractor) -> None:
    source = (
        "extern int g;\n"
        "int g = 1;\n"
        "void f() {\n"
        "    int i = 0;\n"
        "    for (int i = 0; i < 3; ++i) { }\n"
        "    { int t = i; }\n"
        "    { int t = 2; }\n"
        "}\n"
    )
    s = cpp.parse(source)
    # a redeclaration in the same scope is no new slot; the for header and each block are new scopes
    assert kinds_texts_depths(s) == [
        (VARIABLE, "extern int g", 0),
        (FUNCTION, "void f()", 0),
        (VARIABLE, "int i", 1),
        (BLOCK, "for (int i = 0; i < 3; ++i)", 1),
        (VARIABLE, "int i", 1),
        (VARIABLE, "int t", 1),
        (VARIABLE, "int t", 1),
    ]
    assert [t.start_byte for t in s.slots if t.text == "int t"] == [
        source.index("int t = i"),
        source.index("int t = 2"),
    ]


def test_cpp_prototypes_variadic_void_unnamed(cpp: TreeSitterASTExtractor) -> None:
    s = cpp.parse("int f(int a, int b = 3, ...);\nint g(void);\nint h(int);\n")
    f, g, h = s.functions
    assert f.text == "int f(int a, int b = 3, ...)"
    assert f.parameters[-1] == Parameter("...", None, False, "var_positional")
    assert (f.min_arity, f.max_arity) == (1, None)
    assert (g.text, g.parameters, g.max_arity) == ("int g(void)", (), 0)
    assert h.parameters == (Parameter("", "int", False, "positional"),)
    assert not any(fn.has_body for fn in s.functions)
    assert (s.features.num_functions, s.features.num_top_level_statements) == (0, 3)
    assert ModeExtractor("control")(s) == CONTROL_LABELS.index("straight_line")


def test_cpp_multiline_pragma_and_recovery(cpp: TreeSitterASTExtractor) -> None:
    pragma = cpp.parse("void k() {\n#pragma HLS ARRAY_PARTITION variable=buf \\\n    cyclic factor=4\n}\n")
    assert [t.text for t in pragma.slots if t.kind == DIRECTIVE] == [
        "#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=4"
    ]
    broken = cpp.parse("#include <x.h>\nint f(int a { return a; }\nint g(int b) { return b; }\n")
    assert broken.has_errors
    assert broken.imports == ("x.h",) and [fn.name for fn in broken.functions] == ["f", "g"]
    garbage = cpp.parse(bytes(range(256)) * 3)
    assert garbage.has_errors
    deep = cpp.parse("int x = " + "(" * 3000 + "1" + ")" * 3000 + ";\n")
    assert not deep.has_errors and [t.text for t in deep.slots] == ["int x"]


# ----------------------------------------------------------------------------------------------------------------
# Modes (Sec. 5.1)
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ("", "unparseable"),  # empty source: no function and no statement
        ("# just a comment\n", "unparseable"),
        ("def broken(:\n", "unparseable"),
        ("x = 1\nprint(x)\n", "straight_line"),
        ("def f():\n    pass\n", "straight_line"),
        ("for i in range(3):\n    print(i)\n", "for_loop"),
        ("for i in range(3):\n    pass\nwhile True:\n    break\n", "while_loop"),  # while > for
        ("def f(n):\n    while n:\n        n -= 1\n    return f(n)\n", "recursion"),  # recursion > while
        ("squares = [i * i for i in range(3)]\n", "comprehension"),
        ("for i in range(3):\n    pass\nys = {i for i in range(3)}\n", "for_loop"),  # for > comprehension
    ],
)
def test_control_modes_and_priority(py: TreeSitterASTExtractor, source: str, label: str) -> None:
    extractor = ModeExtractor("control")
    mode = extractor(py.parse(source))
    assert mode == CONTROL_LABELS.index(label)
    assert extractor.describe(mode) == label


@pytest.mark.parametrize(
    ("source", "bucket"),
    [
        ("x = 1\n", "no_import"),
        ("from __future__ import annotations\nx = 1\n", "no_import"),  # a compiler directive, not a library
        ("import numpy.linalg\nx = 1\n", "numeric"),  # the top-level module decides
        ("from collections import deque\nx = 1\n", "containers"),
        ("import re\nx = 1\n", "text"),
        ("import os\nx = 1\n", "other"),
        ("from .utils import helper\nx = 1\n", "other"),
        ("import os\nimport re\nfrom itertools import chain\nx = 1\n", "containers"),  # smallest positive wins
        ("import string\nimport statistics\nx = 1\n", "numeric"),
    ],
)
def test_library_buckets(py: TreeSitterASTExtractor, source: str, bucket: str) -> None:
    extractor = ModeExtractor("control_library")
    mode = extractor(py.parse(source))
    assert mode == 5 * CONTROL_LABELS.index("straight_line") + LIBRARY_LABELS.index(bucket)
    assert extractor.describe(mode) == f"straight_line/{bucket}"


def test_control_library_combines_both(py: TreeSitterASTExtractor) -> None:
    s = py.parse("import math\nfor i in range(3):\n    print(math.sqrt(i))\n")
    extractor = ModeExtractor()
    assert extractor(s) == 2 * 5 + 1 == 11
    assert extractor.describe(11) == "for_loop/numeric"


def test_mode_schemes_and_describe() -> None:
    assert MODE_SCHEMES["control"].size == ModeExtractor("control").num_modes == 6
    assert MODE_SCHEMES["control_library"].size == ModeExtractor("control_library").num_modes == 30
    assert MODE_SCHEMES["control"].labels == (
        "unparseable", "straight_line", "for_loop", "while_loop", "recursion", "comprehension",
    )  # fmt: skip
    extractor = ModeExtractor()
    assert extractor.describe(0) == "unparseable/no_import"
    assert extractor.describe(29) == "comprehension/other"
    for bad in (30, -1, True, 2.0, "3"):
        with pytest.raises(ValueError):
            extractor.describe(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ModeExtractor("ast_clusters")


def test_cpp_library_buckets() -> None:
    assert library_bucket(["ap_int.h", "hls_stream.h"], "cpp") == LIBRARY_LABELS.index("other")
    assert library_bucket(["vector", "ap_int.h"], "cpp") == LIBRARY_LABELS.index("containers")
    assert library_bucket(["math.h", "vector"], "cpp") == LIBRARY_LABELS.index("numeric")
    assert library_bucket(["cstring"], "cpp") == LIBRARY_LABELS.index("text")
    assert library_bucket([], "cpp") == LIBRARY_LABELS.index("no_import")


# ----------------------------------------------------------------------------------------------------------------
# Construction, limits, pickling, lazy grammars
# ----------------------------------------------------------------------------------------------------------------


def test_constructor_and_parse_validation() -> None:
    with pytest.raises(ValueError):
        TreeSitterASTExtractor("java")
    with pytest.raises(ValueError):
        TreeSitterASTExtractor("python", max_nodes=0)
    small = TreeSitterASTExtractor("python", max_source_bytes=10)
    assert small.parse("x = 12345") is not None  # 9 bytes: within the limit
    with pytest.raises(ValueError):
        small.parse("x = 1234567890\n")  # 15 bytes
    with pytest.raises(TypeError):
        small.parse(123)  # type: ignore[arg-type]


def test_extractor_pickles_for_dataloader_workers() -> None:
    extractor = TreeSitterASTExtractor("cpp", max_nodes=1234)
    clone = pickle.loads(pickle.dumps(extractor))
    assert (clone.language, clone.max_nodes) == ("cpp", 1234)
    assert [t.text for t in clone.parse("int x;\n").slots] == ["int x"]
    structure = clone.parse(HLS_KERNEL)
    assert pickle.loads(pickle.dumps(structure)) == structure


def test_missing_grammar_raises_import_error_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tree_sitter_cpp", None)  # makes ``import tree_sitter_cpp`` fail
    with pytest.raises(ImportError, match="tree-sitter-cpp"):
        TreeSitterASTExtractor("cpp")
    assert [t.text for t in TreeSitterASTExtractor("python").parse("x = 1\n").slots] == ["x"]


def test_module_import_does_not_need_tree_sitter() -> None:
    code = (
        "import sys\n"
        "sys.modules['tree_sitter'] = None\n"
        "import text_vpa.data.ast_parser as m\n"
        "try:\n"
        "    m.TreeSitterASTExtractor('python')\n"
        "except ImportError as exc:\n"
        "    print('ImportError', 'tree-sitter' in str(exc))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ImportError True"
