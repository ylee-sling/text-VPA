# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
ast_parser.py -- the structural view y and the reference mode u of a program (PR-JEPA Table 1; Sec. 5.1).

Table 1, general-purpose code: the plan slots are "AST regions from a parser such as tree-sitter: imports,
signatures, class skeletons, declared identifiers with types"; the mode u is the "algorithm or data-structure
choice; libraries used"; the plan-mediated constraints are "only identifiers the plan declares may be
referenced; argument counts match planned signatures; imports match planned dependencies". Table 1, HLS-C:
"interface, array declarations with directives, loop nests with directives". Sec. 5.1, Stage 1: "modes u are
extracted from the reference: ... imported libraries and clusters of AST shapes for general-purpose code".

    TreeSitterASTExtractor(language).parse(source) -> ProgramStructure
        .slots                    the structural view y: StructuralSlot records in source order
        .imports, .functions, .declared_identifiers, .referenced_identifiers, .calls, .call_sites
                                  the inventories that the constraint energies of Table 1 check
    ModeExtractor(scheme)(structure) -> u in {0, ..., |U| - 1}    the reference mode of L_CE in Eq. 5

Slot texts
----------
Comments and line continuations are removed, every whitespace run becomes one space (inside string literals
too, so a slot text is always a single line) and no body is ever included.

    kind       Python                                          C++ / HLS-C
    IMPORT     "import numpy as np", "from math import sqrt, pi"    "#include <ap_int.h>"
    FUNCTION   "def f(a, b: int = 3) -> int", "async def g()"      "void kernel(ap_uint<8> x[16], int n)"
    CLASS      "class Name(Base1, Base2)", "class Name"            "struct Name", "class Name : public Base"
    VARIABLE   "x: int", "x"                                       "int acc", "static ap_uint<8> buf[16]",
                                                                    "typedef ap_uint<12> data_t", "#define N 16"
    BLOCK      "for i in range(n)", "elif x < 0", "else", "try",    "for (int i = 0; i < n; ++i)", "while (i < n)",
               "except ValueError as e", "with open(p) as f",       "if (a > b)", "else if (a < b)", "else",
               "match cmd", "case [x, *rest]"                       "do while (i > 0)", "switch (n)", "case 1",
                                                                    "try", "catch (const E &e)"
    DIRECTIVE  --                                                  "#pragma HLS PIPELINE II=1"

Canonical forms: a parameter list (or a Python base-class list) is re-joined with ", ", so trailing commas
and line breaks of multi-line signatures disappear; a Python from-import list drops its parentheses. A header
text runs up to, but excludes, the colon (Python) or the body (C++). ``StructuralSlot.name`` is the function,
class or variable name, the (first) module path of an import, and None for blocks and directives.
[start_byte, end_byte) is the source region of the slot (header only, never the body) in UTF-8 bytes.

Slot rules
----------
* VARIABLE: one slot per name per scope at its FIRST binding. Python scopes are module / function / class /
  lambda; bindings are assignment, augmented and annotated targets (with tuple unpacking and starred targets),
  for targets, with-as, except-as, walrus targets, match captures and ``type X = ...``. Names bound by
  parameters, imports, def, class, global and nonlocal count as bindings of their scope but get no VARIABLE
  slot (they live in their own slots, or in another scope). Comprehension variables live in their own scope
  and get no slot. C++ scopes are blocks, so every declarator of a declaration is a slot (redeclaration in one
  scope is ill-formed); parameters are not, typedefs and macros are.
* depth = number of construct bodies (function, class, lambda, and every BLOCK construct) that enclose the
  slot. Headers sit at the depth of their construct, so ``elif`` / ``else`` / ``except`` / ``finally`` and a
  C++ ``else if`` sit at the depth of their ``if`` / ``try``, as their indentation suggests, and a for target
  sits at the depth of its ``for``.
* FUNCTION slots are definitions plus C++ prototypes (``FunctionSignature.has_body`` is False for those).
  ``is_method`` holds when the innermost enclosing scope is a class body. CLASS slots need a body (a C++
  ``enum`` with a body counts; a forward declaration does not).

Identifier inventories
----------------------
``declared_identifiers``: every name bound anywhere (see the field). ``referenced_identifiers``: identifiers in
load context, including f-string replacement fields and augmented targets; keyword-argument names, attribute
names, import paths and binding occurrences are excluded. For C++, type names, namespace qualifiers and the
names inside qualified identifiers are excluded (they resolve through headers and namespaces), and identifiers
in ``#if`` / ``#ifdef`` conditions are ignored. Both inventories are deduplicated in source order, and reserved
keywords that only appear through error recovery are dropped. ``undefined_names`` does not expand
``from m import *`` or C/C++ headers.

Modes (Sec. 5.1)
----------------
``"control"`` (6 modes; the first rule that holds, in this order):

    0 unparseable    has_errors, or no function and no top-level statement (an empty source or comments only)
    4 recursion      ControlFlowFeatures.has_recursion
    3 while_loop     has_while (a C++ do-while counts)
    2 for_loop       has_for (comprehension clauses do not count)
    5 comprehension  has_comprehension
    1 straight_line  otherwise

``"control_library"`` (30 modes): u = 5 c + b with c the control mode and b the library bucket of the imports:

    0 no_import   no import ("from __future__" is a compiler directive and is ignored)
    1 numeric     math, cmath, numpy, statistics, fractions, decimal, random
    2 containers  collections, itertools, functools, heapq, bisect, operator, copy
    3 text        re, string, textwrap, unicodedata
    4 other       any other import (relative imports included)

The top-level module of a dotted path decides ("numpy.linalg" -> numeric); with several imports the smallest
positive bucket wins (numeric > containers > text > other). C++ includes are keyed by the header stem without
extension and directories ("math.h" -> math): numeric also takes complex, numeric, valarray, hls_math;
containers takes the STL container / algorithm headers (vector, map, algorithm, ...); text takes cstring,
cctype, regex, sstream, string_view, charconv. HLS type headers (ap_int.h, ap_fixed.h, hls_stream.h) are
"other".

Robustness
----------
* Grammars are imported lazily: importing this module never needs tree-sitter; constructing an extractor for
  a language whose grammar package is missing raises ImportError naming the package.
* The traversal is an iterative DFS with an explicit stack, bounded by ``max_nodes`` (when exceeded the
  traversal stops and ``has_errors`` is set). Per-node extraction runs under ``try`` / ``except`` of
  UnicodeDecodeError, IndexError, AttributeError, ValueError, TypeError and KeyError: a failing node is
  recorded in ``warnings``, sets ``has_errors`` and is traversed generically, so the rest of the file is kept.
* Malformed source never raises; it yields ``has_errors=True`` with whatever slots tree-sitter's error
  recovery left intact. Node text is decoded from the UTF-8 bytes with ``errors="replace"``; ``str`` input is
  encoded with ``surrogatepass`` so lone surrogates cannot raise. A source larger than ``max_source_bytes``
  raises ValueError (a caller error), as does an unsupported language.
* ``has_errors`` is True when tree-sitter reports an ERROR or MISSING node, when the node budget is exceeded,
  when a node's extraction fails, and, for Python with ``validate_syntax=True`` (the default), when CPython's
  own parser (``ast.parse``) rejects the source. tree-sitter's error recovery silently accepts some programs
  CPython rejects: an unexpected indent, an inconsistent dedent, more than 200 nested brackets, more than 100
  indentation levels.
* The extractor is picklable (the parser is rebuilt lazily), so it can live inside a DataLoader dataset under
  the ``spawn`` start method used on macOS.
"""

from __future__ import annotations

import ast
import bisect
import builtins as _builtins_module
import enum
import importlib
import keyword
import operator
import re
import warnings as _warnings_module
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from tree_sitter import Node, Parser

__all__ = [
    "CONTROL_LABELS",
    "CPP_DEFAULT_BUILTINS",
    "EMPTY_SLOT",
    "LIBRARY_LABELS",
    "MODE_SCHEMES",
    "PARAMETER_KINDS",
    "PYTHON_DEFAULT_BUILTINS",
    "CallSite",
    "ControlFlowFeatures",
    "FunctionSignature",
    "ModeExtractor",
    "ModeScheme",
    "Parameter",
    "ProgramStructure",
    "SlotKind",
    "StructuralSlot",
    "TreeSitterASTExtractor",
    "control_mode",
    "library_bucket",
]


# --------------------------------------------------------------------------------------------------------------
# Public records
# --------------------------------------------------------------------------------------------------------------


class SlotKind(enum.IntEnum):
    """Kind of a structural slot of the plan view y (Table 1)."""

    EMPTY = 0  # padding slot
    IMPORT = 1  # import / #include
    FUNCTION = 2  # function signature
    CLASS = 3  # class / struct skeleton
    VARIABLE = 4  # declared identifier (first binding per scope), with type if annotated
    BLOCK = 5  # block boundary: for / while / if / elif / with / try headers
    DIRECTIVE = 6  # C/C++ #pragma (HLS directives)


PARAMETER_KINDS: tuple[str, ...] = ("positional", "keyword_only", "var_positional", "var_keyword")


@dataclass(frozen=True)
class Parameter:
    """One parameter of a signature.

    ``annotation`` is the Python annotation, or the C++ parameter type including pointer, reference and array
    declarators (``"ap_uint<8>[16]"``, ``"const float *"``, ``"int &"``). ``name`` is ``""`` for an unnamed
    C++ parameter and ``"..."`` for a C variadic ``...``.
    """

    name: str
    annotation: str | None
    has_default: bool
    kind: str  # "positional" | "keyword_only" | "var_positional" (*args) | "var_keyword" (**kwargs)

    def __post_init__(self) -> None:
        if self.kind not in PARAMETER_KINDS:
            raise ValueError(f"Parameter.kind must be one of {PARAMETER_KINDS}, got {self.kind!r}")


@dataclass(frozen=True)
class FunctionSignature:
    """A planned signature (Table 1: "argument counts match planned signatures")."""

    name: str
    parameters: tuple[Parameter, ...]
    return_annotation: str | None
    is_method: bool  # defined directly inside a class body
    text: str  # normalized signature text, == the FUNCTION slot text
    is_async: bool = False
    has_body: bool = True  # False for a C++ prototype

    @property
    def min_arity(self) -> int:
        """Positional parameters without defaults (``*args``, ``**kwargs`` and keyword-only excluded)."""
        return _min_arity(self.parameters)

    @property
    def max_arity(self) -> int | None:
        """Number of positional parameters; None if ``*args`` or ``**kwargs`` is present."""
        return _max_arity(self.parameters)

    def call_arity_bounds(self) -> tuple[int, int | None]:
        """Bounds on the positional argument count of a call ``obj.name(...)``.

        For a method whose first parameter is ``self`` or ``cls`` the bound receiver supplies it, so both
        bounds drop by one; otherwise they are ``(min_arity, max_arity)``.
        """
        params = self.parameters
        if self.is_method and params and params[0].name in ("self", "cls") and params[0].kind == "positional":
            params = params[1:]
        return _min_arity(params), _max_arity(params)


def _min_arity(parameters: Sequence[Parameter]) -> int:
    return sum(1 for p in parameters if p.kind == "positional" and not p.has_default)


def _max_arity(parameters: Sequence[Parameter]) -> int | None:
    if any(p.kind in ("var_positional", "var_keyword") for p in parameters):
        return None
    return sum(1 for p in parameters if p.kind == "positional")


@dataclass(frozen=True)
class StructuralSlot:
    """One slot of the structural view y: a normalized header text and its source region."""

    kind: SlotKind
    text: str  # normalized text (whitespace runs collapsed to one space, no comments, no body)
    name: str | None  # function/class/variable name; module path for imports; None for blocks/directives
    start_byte: int
    end_byte: int
    depth: int  # number of enclosing function/class/block constructs


EMPTY_SLOT = StructuralSlot(SlotKind.EMPTY, "", None, -1, -1, 0)


@dataclass(frozen=True)
class CallSite:
    """One call, for the argument-count energy of Table 1 (compare with ``FunctionSignature.call_arity_bounds``).

    ``receiver`` is None for a plain call ``f(...)`` (including C++ ``ns::f(...)``); for ``obj.m(...)`` it is
    the receiver's text when that is a name or a dotted chain of names (``"self"``, ``"np.linalg"``,
    ``"this"``) and ``""`` for any other receiver expression.
    """

    name: str
    num_positional: int
    keywords: tuple[str, ...]
    has_star_args: bool
    has_star_kwargs: bool
    receiver: str | None
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class ControlFlowFeatures:
    """AST-shape features behind the "control" mode scheme (Sec. 5.1)."""

    has_for: bool
    has_while: bool
    has_recursion: bool  # some function calls itself by name (directly, incl. self.name(...) for methods)
    has_comprehension: bool  # list/set/dict comprehension or generator expression
    has_try: bool
    num_functions: int  # function definitions (def / async def; C++ bodies); lambdas and prototypes excluded
    max_loop_depth: int  # loop nesting; a function or class body restarts the count
    num_top_level_statements: int = 0  # direct children of the module / translation unit, comments excluded


@dataclass(frozen=True)
class ProgramStructure:
    """Everything the plan of one program fixes (Table 1): slots, inventories and AST-shape features."""

    language: str
    slots: tuple[StructuralSlot, ...]  # ALL slots, in source order (sorted by start_byte)
    imports: tuple[str, ...]  # imported module paths / names, in order, deduplicated
    functions: tuple[FunctionSignature, ...]  # in source order
    classes: tuple[str, ...]  # one name per class definition, in source order
    # every name bound anywhere: function / class names, parameters, assignment / augmented / annotated targets
    # (tuple unpacking included), for targets, comprehension variables, import names / aliases, with-as,
    # except-as, lambda parameters, global / nonlocal, walrus targets, match captures; first-occurrence order
    declared_identifiers: tuple[str, ...]
    # every identifier used in code (not inside strings / comments); attribute names excluded
    referenced_identifiers: tuple[str, ...]
    attribute_names: tuple[str, ...]  # names after '.' (or '->') in attribute access
    calls: tuple[str, ...]  # callee names: plain f(...) -> "f"; obj.m(...) -> "m"
    features: ControlFlowFeatures
    # tree-sitter ERROR / MISSING nodes, node budget exceeded, a node's extraction failed, or (Python,
    # validate_syntax) CPython's parser rejects the source; ``warnings`` says which
    has_errors: bool
    call_sites: tuple[CallSite, ...] = ()
    warnings: tuple[str, ...] = ()

    def slot_records(self, num_slots: int) -> tuple[StructuralSlot, ...]:
        """The first ``num_slots`` slots, padded with ``EMPTY_SLOT`` to exactly ``num_slots`` records."""
        if num_slots < 0:
            raise ValueError(f"num_slots must be >= 0, got {num_slots}")
        records = self.slots[:num_slots]
        return records + (EMPTY_SLOT,) * (num_slots - len(records))

    def undefined_names(self, builtins: frozenset[str] | None = None) -> tuple[str, ...]:
        """Referenced names that are neither declared nor builtins, in source order.

        ``builtins`` defaults to ``PYTHON_DEFAULT_BUILTINS`` for Python and ``CPP_DEFAULT_BUILTINS`` otherwise.
        """
        if builtins is None:
            builtins = PYTHON_DEFAULT_BUILTINS if self.language == "python" else CPP_DEFAULT_BUILTINS
        declared = set(self.declared_identifiers)
        return tuple(n for n in self.referenced_identifiers if n not in declared and n not in builtins)


# dir(builtins) + keyword.softkwlist as specified, plus the implicit module / class-body names (__file__, ...)
# that CPython binds without a statement.
PYTHON_DEFAULT_BUILTINS: frozenset[str] = (
    frozenset(dir(_builtins_module))
    | frozenset(keyword.softkwlist)
    | frozenset({"__file__", "__builtins__", "__annotations__", "__cached__", "__path__"})
    | frozenset({"__class__", "__module__", "__qualname__"})
)

# Headers are not expanded, so C++ has no exact builtin set. This default holds predefined names and the C
# library functions HLS-C kernels and testbenches use most (stdio, stdlib, string.h, math.h, iostream).
CPP_DEFAULT_BUILTINS: frozenset[str] = frozenset(
    {
        "__func__", "__FILE__", "__LINE__", "__DATE__", "__TIME__", "__cplusplus", "__SYNTHESIS__",
        "NULL", "EOF", "errno", "stdin", "stdout", "stderr", "assert", "offsetof", "size_t",
        "printf", "fprintf", "sprintf", "snprintf", "scanf", "fscanf", "sscanf", "puts", "putchar", "getchar",
        "fopen", "fclose", "fread", "fwrite", "fgets", "fputs", "perror",
        "malloc", "calloc", "realloc", "free", "exit", "abort", "abs", "labs", "atoi", "atof", "rand", "srand",
        "qsort", "memcpy", "memset", "memmove", "memcmp", "strlen", "strcmp", "strncmp", "strcpy", "strncpy",
        "strcat", "sqrt", "sqrtf", "pow", "powf", "exp", "expf", "log", "logf", "log2", "log10", "sin", "sinf",
        "cos", "cosf", "tan", "tanf", "tanh", "asin", "acos", "atan", "atan2", "sinh", "cosh", "fabs", "fabsf",
        "floor", "floorf", "ceil", "ceilf", "round", "trunc", "fmod", "fmin", "fmax", "hypot",
        "std", "cout", "cin", "cerr", "endl", "min", "max", "swap",
    }
)  # fmt: skip

_PYTHON_KEYWORDS: frozenset[str] = frozenset(keyword.kwlist)
_CPP_KEYWORDS: frozenset[str] = frozenset(
    """alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t
    class compl concept const consteval constexpr constinit const_cast continue co_await co_return co_yield
    decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto
    if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected
    public register reinterpret_cast requires return short signed sizeof static static_assert static_cast
    struct switch template this thread_local throw true try typedef typeid typename union unsigned using virtual
    void volatile wchar_t while xor xor_eq""".split()
)


# --------------------------------------------------------------------------------------------------------------
# Mode schemes (Sec. 5.1: modes extracted from the reference)
# --------------------------------------------------------------------------------------------------------------

CONTROL_LABELS: tuple[str, ...] = (
    "unparseable",
    "straight_line",
    "for_loop",
    "while_loop",
    "recursion",
    "comprehension",
)
LIBRARY_LABELS: tuple[str, ...] = ("no_import", "numeric", "containers", "text", "other")

_UNPARSEABLE, _STRAIGHT_LINE, _FOR_LOOP, _WHILE_LOOP, _RECURSION, _COMPREHENSION = range(6)
_NO_IMPORT, _NUMERIC, _CONTAINERS, _TEXT, _OTHER = range(5)

_PYTHON_LIBRARY_BUCKETS: dict[str, int] = {
    **dict.fromkeys(("math", "cmath", "numpy", "statistics", "fractions", "decimal", "random"), _NUMERIC),
    **dict.fromkeys(("collections", "itertools", "functools", "heapq", "bisect", "operator", "copy"), _CONTAINERS),
    **dict.fromkeys(("re", "string", "textwrap", "unicodedata"), _TEXT),
}
_CPP_LIBRARY_BUCKETS: dict[str, int] = {
    **dict.fromkeys(("cmath", "math", "complex", "numeric", "valarray", "random", "hls_math"), _NUMERIC),
    **dict.fromkeys(
        (
            "vector", "array", "deque", "list", "forward_list", "map", "set", "unordered_map", "unordered_set",
            "queue", "stack", "algorithm", "iterator", "functional", "utility", "tuple", "bitset", "span",
        ),
        _CONTAINERS,
    ),  # fmt: skip
    **dict.fromkeys(("string", "cstring", "regex", "sstream", "cctype", "ctype", "string_view", "charconv"), _TEXT),
}
_CPP_HEADER_SUFFIXES: tuple[str, ...] = (".h", ".hh", ".hpp", ".hxx", ".h++")


@dataclass(frozen=True)
class ModeScheme:
    """A discrete mode set U: ``labels[u]`` names mode u."""

    name: str
    labels: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.labels)


MODE_SCHEMES: dict[str, ModeScheme] = {
    "control": ModeScheme("control", CONTROL_LABELS),
    "control_library": ModeScheme(
        "control_library", tuple(f"{c}/{b}" for c in CONTROL_LABELS for b in LIBRARY_LABELS)
    ),
}


def control_mode(structure: ProgramStructure) -> int:
    """The "control" mode of a program (module docstring, priority order)."""
    f = structure.features
    if structure.has_errors or (f.num_functions == 0 and f.num_top_level_statements == 0):
        return _UNPARSEABLE
    if f.has_recursion:
        return _RECURSION
    if f.has_while:
        return _WHILE_LOOP
    if f.has_for:
        return _FOR_LOOP
    if f.has_comprehension:
        return _COMPREHENSION
    return _STRAIGHT_LINE


def library_bucket(imports: Iterable[str], language: str = "python") -> int:
    """Library bucket in {0, ..., 4} of a list of imported modules (module docstring)."""
    table = _CPP_LIBRARY_BUCKETS if language == "cpp" else _PYTHON_LIBRARY_BUCKETS
    best = _NO_IMPORT
    for module in imports:
        key = _library_key(module, language)
        if key is None:
            continue
        bucket = table.get(key, _OTHER)
        if best == _NO_IMPORT or bucket < best:
            best = bucket
    return best


def _library_key(module: str, language: str) -> str | None:
    """The table key of one import: the top-level module (Python) or the header stem (C++)."""
    module = module.strip()
    if not module:
        return None
    if language == "cpp":
        stem = module.replace("\\", "/").rsplit("/", 1)[-1]
        for suffix in _CPP_HEADER_SUFFIXES:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem
    if module == "__future__":
        return None
    if module.startswith("."):
        return ""  # a relative import is local code: "other"
    return module.split(".", 1)[0]


class ModeExtractor:
    """Maps a ``ProgramStructure`` to its reference mode u (the target of the selector's L_CE, Eq. 5)."""

    def __init__(self, scheme: str = "control_library") -> None:
        if scheme not in MODE_SCHEMES:
            raise ValueError(f"unknown mode scheme {scheme!r}; choose one of {sorted(MODE_SCHEMES)}")
        self.scheme: ModeScheme = MODE_SCHEMES[scheme]

    @property
    def num_modes(self) -> int:
        return self.scheme.size

    def __call__(self, structure: ProgramStructure) -> int:
        control = control_mode(structure)
        if self.scheme.name == "control":
            return control
        return control * len(LIBRARY_LABELS) + library_bucket(structure.imports, structure.language)

    def describe(self, mode: int) -> str:
        """The label of ``mode`` (any integer type, e.g. a numpy or 0-d torch integer); ValueError if invalid."""
        try:
            index = operator.index(mode)
        except TypeError:
            index = -1
        if isinstance(mode, bool) or not 0 <= index < self.num_modes:
            raise ValueError(f"mode must be an integer in [0, {self.num_modes}), got {mode!r}")
        return self.scheme.labels[index]

    def __repr__(self) -> str:
        return f"ModeExtractor(scheme={self.scheme.name!r}, num_modes={self.num_modes})"


# --------------------------------------------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------------------------------------------

_GRAMMAR_PACKAGES: dict[str, str] = {"python": "tree_sitter_python", "cpp": "tree_sitter_cpp"}

# Exceptions a single node's extraction may raise on malformed input; anything else is a bug and propagates.
_NODE_ERRORS: tuple[type[Exception], ...] = (
    UnicodeDecodeError,
    IndexError,
    AttributeError,
    ValueError,
    TypeError,
    KeyError,
)


def _load_parser(language: str) -> Parser:
    """A tree-sitter >= 0.22 parser for ``language``; ImportError names the missing package."""
    package = _GRAMMAR_PACKAGES[language]
    try:
        tree_sitter = importlib.import_module("tree_sitter")
    except ImportError as exc:
        raise ImportError(
            "TreeSitterASTExtractor needs the 'tree-sitter' package (>= 0.22): pip install tree-sitter"
        ) from exc
    try:
        grammar = importlib.import_module(package)
    except ImportError as exc:
        raise ImportError(
            f"language {language!r} needs the grammar package {package.replace('_', '-')!r}: "
            f"pip install {package.replace('_', '-')}"
        ) from exc
    try:
        return tree_sitter.Parser(tree_sitter.Language(grammar.language()))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ImportError(f"incompatible tree-sitter / {package} versions (need tree-sitter >= 0.22): {exc}") from exc


class TreeSitterASTExtractor:
    """Parses one source file into a ``ProgramStructure`` (module docstring).

    ``validate_syntax`` (Python only) additionally runs CPython's parser when tree-sitter finds no error. An
    extractor owns one tree-sitter parser, which is not thread-safe: use one extractor per thread (DataLoader
    worker processes each get their own copy).
    """

    SUPPORTED_LANGUAGES = ("python", "cpp")

    def __init__(
        self,
        language: str = "python",
        max_source_bytes: int = 2_000_000,
        max_nodes: int = 500_000,
        *,
        validate_syntax: bool = True,
    ) -> None:
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language {language!r}; supported: {self.SUPPORTED_LANGUAGES}")
        if max_source_bytes <= 0 or max_nodes <= 0:
            raise ValueError(f"max_source_bytes and max_nodes must be positive, got {max_source_bytes}, {max_nodes}")
        self.language = language
        self.max_source_bytes = int(max_source_bytes)
        self.max_nodes = int(max_nodes)
        self.validate_syntax = bool(validate_syntax)
        self._parser: Parser | None = _load_parser(language)  # fails fast when the grammar is missing

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_parser"] = None  # tree-sitter parsers do not pickle; rebuilt on first use
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"TreeSitterASTExtractor(language={self.language!r}, max_source_bytes={self.max_source_bytes}, "
            f"max_nodes={self.max_nodes}, validate_syntax={self.validate_syntax})"
        )

    def parse(self, source: str | bytes) -> ProgramStructure:
        """Extract the structure of ``source`` (str, or UTF-8 bytes). Never raises on malformed source."""
        if isinstance(source, str):
            data = source.encode("utf-8", errors="surrogatepass")
        elif isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        else:
            raise TypeError(f"source must be str or bytes, got {type(source).__name__}")
        if len(data) > self.max_source_bytes:
            raise ValueError(f"source is {len(data)} bytes, more than max_source_bytes={self.max_source_bytes}")
        if self._parser is None:
            self._parser = _load_parser(self.language)
        walker = _Walker(data, self.language, self.max_nodes)
        try:
            tree = self._parser.parse(data)
        except (ValueError, RuntimeError, TypeError) as exc:
            walker.warnings.append(f"tree-sitter failed to parse: {type(exc).__name__}: {exc}")
            return walker.build(has_errors=True, num_top_level_statements=0)
        root = tree.root_node
        walker.run(root)
        has_errors = bool(root.has_error) or walker.budget_exceeded or walker.failed
        if self.language == "python" and self.validate_syntax and not has_errors:
            message = _python_syntax_error(source if isinstance(source, str) else data)
            if message is not None:
                walker.warnings.append(f"CPython rejects the source: {message}")
                has_errors = True
        top_level = sum(1 for child in root.children if child.is_named and child.type not in walker.extra_types)
        return walker.build(has_errors=has_errors, num_top_level_statements=top_level)


def _python_syntax_error(source: str | bytes) -> str | None:
    """CPython's verdict on ``source``: None if ``ast.parse`` accepts it, else a one-line message."""
    try:
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore")  # SyntaxWarning (e.g. invalid escapes) must not raise
            ast.parse(source)
    except SyntaxError as exc:
        return f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})"
    except (ValueError, RecursionError, MemoryError, OverflowError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


# --------------------------------------------------------------------------------------------------------------
# Traversal internals
# --------------------------------------------------------------------------------------------------------------

# Identifier modes: how an ``identifier`` reached in this context is recorded.
_LOAD = 0  # a reference
_BIND = 1  # a binding that gets a VARIABLE slot at the first binding in its scope (Python)
_AUGMENT = 2  # augmented target: a reference and a binding
_DECLARE = 3  # a declaration without a slot: parameters, comprehension variables
_PATTERN = 4  # inside a Python match pattern: bare captures bind
_SKIP = 5  # ignored: import paths, qualified names, preprocessor conditions

# Python nodes through which a binding mode reaches the identifiers below (unpacking targets, parameter lists).
_TARGET_CONTAINERS: frozenset[str] = frozenset(
    {
        "pattern_list", "tuple_pattern", "list_pattern", "tuple", "list", "parenthesized_expression",
        "expression_list", "list_splat_pattern", "list_splat", "dictionary_splat_pattern", "as_pattern_target",
        "parenthesized_list_splat", "parameters", "lambda_parameters",
    }
)  # fmt: skip
_COMPREHENSIONS: frozenset[str] = frozenset(
    {"list_comprehension", "set_comprehension", "dictionary_comprehension", "generator_expression"}
)
_BODY_FIELDS: frozenset[str] = frozenset({"body", "consequence"})
_CPP_NAME_TYPES: frozenset[str] = frozenset(
    {
        "identifier", "field_identifier", "type_identifier", "qualified_identifier", "destructor_name",
        "operator_name", "template_function", "template_method",
    }
)  # fmt: skip
_CPP_CLASS_TYPES: frozenset[str] = frozenset(
    {"struct_specifier", "class_specifier", "union_specifier", "enum_specifier"}
)
_CPP_PARAMETER_TYPES: frozenset[str] = frozenset(
    {"parameter_declaration", "optional_parameter_declaration", "variadic_parameter_declaration"}
)
_MAX_CHAIN = 256  # bound on declarator / qualified-name chains (each step descends one tree level)
_RECEIVER_MAX_CHARS = 80
_LINE_SPLICE = re.compile(r"\\\r?\n")


class _Span(NamedTuple):
    """A source region of a text template; ``preproc`` regions get C comments stripped lexically."""

    start: int
    end: int
    preproc: bool = False


_Piece = str | _Span


def _span(node: Node) -> _Span:
    return _Span(node.start_byte, node.end_byte)


def _joined(pieces: Iterable[_Piece], separator: str) -> list[_Piece]:
    out: list[_Piece] = []
    for piece in pieces:
        if out:
            out.append(separator)
        out.append(piece)
    return out


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _strip_c_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments outside string / character literals (preprocessor arguments)."""
    out: list[str] = []
    i, n, quote = 0, len(text), ""
    while i < n:
        ch = text[i]
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
        elif ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
        elif text.startswith("//", i):
            break  # a preprocessor argument is one logical line
        elif text.startswith("/*", i):
            close = text.find("*/", i + 2)
            i = n if close < 0 else close + 2
            out.append(" ")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


class _Scope:
    """A binding scope: names bound so far (first-binding rule of VARIABLE slots)."""

    __slots__ = ("bound", "kind")

    def __init__(self, kind: str, bound: Iterable[str] = ()) -> None:
        self.kind = kind  # "module" | "function" | "class" | "lambda" | "block" | "namespace"
        self.bound: set[str] = set(bound)


class _FunctionContext(NamedTuple):
    """The innermost enclosing named function, for recursion detection."""

    name: str
    receivers: frozenset[str]  # receivers r for which r.name(...) is a self-call
    plain_calls: bool  # whether a plain name(...) is a self-call (False for Python methods)


class _Context(NamedTuple):
    depth: int
    scope: _Scope
    mode: int
    function: _FunctionContext | None
    loop_depth: int
    class_name: str | None


class _PendingSlot(NamedTuple):
    kind: SlotKind
    template: tuple[_Piece, ...]
    name: str | None
    start: int
    end: int
    depth: int


class _PendingParameter(NamedTuple):
    name: str
    annotation: tuple[_Piece, ...] | None
    suffix: tuple[_Piece, ...] | None  # C++ declarator suffix ("*", "&", "[16]") appended to the type
    has_default: bool
    kind: str


class _PendingFunction(NamedTuple):
    slot: int  # index into _Walker.slots
    name: str
    parameters: tuple[_PendingParameter, ...]
    return_annotation: tuple[_Piece, ...] | None
    is_method: bool
    is_async: bool
    has_body: bool


def _with_mode(ctx: _Context, mode: int) -> _Context:
    return ctx if ctx.mode == mode else ctx._replace(mode=mode)


if TYPE_CHECKING:
    _Pushes = list[tuple[Node, _Context]]  # children to traverse with their contexts, in source order
    _Handler = Callable[[Node, _Context], _Pushes]


class _Walker:
    """One traversal of one tree (phase 1: collect), then ``build`` (phase 2: render texts, sort, dedupe)."""

    def __init__(self, data: bytes, language: str, max_nodes: int) -> None:
        self.data = data
        self.language = language
        self.python = language == "python"
        self.max_nodes = max_nodes
        self.extra_types: frozenset[str] = (
            frozenset({"comment", "line_continuation"}) if self.python else frozenset({"comment"})
        )
        self.keywords = _PYTHON_KEYWORDS if self.python else _CPP_KEYWORDS
        self.slots: list[_PendingSlot] = []
        self.functions: list[_PendingFunction] = []
        self.classes: list[tuple[int, str]] = []
        self.imports: list[tuple[int, str]] = []
        self.declared: list[tuple[int, str]] = []
        self.referenced: list[tuple[int, str]] = []
        self.attributes: list[tuple[int, str]] = []
        self.calls: list[tuple[int, str]] = []
        self.call_sites: list[CallSite] = []
        self.extras: list[tuple[int, int]] = []
        self.extra_starts: list[int] = []
        self.warnings: list[str] = []
        self.failed = False
        self.budget_exceeded = False
        self.has_for = self.has_while = self.has_recursion = self.has_comprehension = self.has_try = False
        self.num_functions = 0
        self.max_loop_depth = 0
        self.handlers: dict[str, _Handler] = (
            self._python_handlers() if self.python else self._cpp_handlers()
        )

    # -- phase 1: traversal ------------------------------------------------------------------------------------

    def run(self, root: Node) -> None:
        """Iterative pre-order DFS (explicit stack, children pushed in reverse so they pop in source order)."""
        stack: list[tuple[Node, _Context]] = [(root, _Context(0, _Scope("module"), _LOAD, None, 0, None))]
        visited = 0
        handlers, extra_types = self.handlers, self.extra_types
        while stack:
            node, ctx = stack.pop()
            visited += 1
            if visited > self.max_nodes:
                self.budget_exceeded = True
                self.warnings.append(f"node budget exceeded (max_nodes={self.max_nodes}); traversal stopped")
                break
            node_type = node.type
            if node_type in extra_types:
                self.extras.append((node.start_byte, node.end_byte))
                continue
            handler = self._generic if ctx.mode == _SKIP else handlers.get(node_type, self._generic)
            try:
                pushes = handler(node, ctx)
            except _NODE_ERRORS as exc:
                self.failed = True
                self.warnings.append(f"{node_type} at byte {node.start_byte}: {type(exc).__name__}: {exc}")
                pushes = self._generic(node, ctx)
            stack.extend(reversed(pushes))

    def _generic(self, node: Node, ctx: _Context) -> _Pushes:
        if ctx.mode in (_BIND, _AUGMENT, _DECLARE) and node.type not in _TARGET_CONTAINERS:
            ctx = ctx._replace(mode=_LOAD)
        return [(child, ctx) for child in node.children if child.is_named]

    def _skip(self, node: Node, ctx: _Context) -> _Pushes:
        skip = _with_mode(ctx, _SKIP)
        return [(child, skip) for child in node.children if child.is_named]

    # -- shared helpers ----------------------------------------------------------------------------------------

    def _raw(self, node: Node | None) -> str:
        if node is None:
            return ""
        return self.data[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _name(self, node: Node | None) -> str:
        """Identifier-like text with whitespace removed (dotted import paths may contain spaces)."""
        return "".join(self._raw(node).split())

    def _fields(self, node: Node) -> list[tuple[Node, str | None]]:
        return [(child, node.field_name_for_child(i)) for i, child in enumerate(node.children) if child.is_named]

    def _first_named(self, node: Node, types: Iterable[str] | None = None) -> Node | None:
        wanted = None if types is None else frozenset(types)
        for child in node.children:
            if child.is_named and child.type not in self.extra_types and (wanted is None or child.type in wanted):
                return child
        return None

    def _slot(
        self, kind: SlotKind, template: Sequence[_Piece], name: str | None, start: int, end: int, depth: int
    ) -> int:
        self.slots.append(_PendingSlot(kind, tuple(template), name, start, max(start, end), depth))
        return len(self.slots) - 1

    def _declare(self, node: Node, name: str) -> None:
        if name and name not in self.keywords:
            self.declared.append((node.start_byte, name))

    def _declare_in_scope(self, node: Node, name: str, ctx: _Context) -> None:
        """A binding that has its own slot kind (def, class, import, global): bound, but no VARIABLE slot."""
        self._declare(node, name)
        if name:
            ctx.scope.bound.add(name)

    def _bind(self, node: Node, name: str, ctx: _Context, annotation: Node | None = None) -> None:
        """A Python binding: declared, plus a VARIABLE slot if it is the first binding of ``name`` in its scope."""
        if not name or name in self.keywords:
            return
        self.declared.append((node.start_byte, name))
        if name in ctx.scope.bound:
            return
        ctx.scope.bound.add(name)
        if annotation is None:
            self._slot(SlotKind.VARIABLE, (_span(node),), name, node.start_byte, node.end_byte, ctx.depth)
        else:
            template = (_span(node), ": ", _span(annotation))
            self._slot(SlotKind.VARIABLE, template, name, node.start_byte, annotation.end_byte, ctx.depth)

    def _header_end(self, node: Node) -> int:
        """End byte of a construct's header: up to the colon (Python, C++ case) or the body (C++)."""
        stop_at_colon = self.python or node.type == "case_statement"
        end = node.start_byte
        for index, child in enumerate(node.children):
            child_type = child.type
            if stop_at_colon and child_type == ":":
                break
            if node.field_name_for_child(index) in _BODY_FIELDS or (self.python and child_type == "block"):
                break
            if child_type not in self.extra_types:
                end = child.end_byte
        return end

    def _header_slot(self, node: Node, ctx: _Context, prefix: str = "", start: int | None = None) -> None:
        end = self._header_end(node)
        template: list[_Piece] = [prefix] if prefix else []
        template.append(_Span(node.start_byte, end))
        self._slot(SlotKind.BLOCK, template, None, node.start_byte if start is None else start, end, ctx.depth)

    def _enter_loop(self, ctx: _Context) -> int:
        loop_depth = ctx.loop_depth + 1
        self.max_loop_depth = max(self.max_loop_depth, loop_depth)
        return loop_depth

    def _receiver_text(self, node: Node | None) -> str:
        if node is None:
            return ""
        if node.type not in ("identifier", "attribute", "field_expression", "this", "dotted_name"):
            return ""
        text = _collapse(self._raw(node))
        return text if len(text) <= _RECEIVER_MAX_CHARS else ""

    def _record_call(
        self,
        call: Node,
        callee: Node,
        name: str,
        receiver: str | None,
        arguments: tuple[int, tuple[str, ...], bool, bool],
        ctx: _Context,
    ) -> None:
        if not name or name in self.keywords:
            return
        self.calls.append((callee.start_byte, name))
        positional, keywords, star, double_star = arguments
        self.call_sites.append(
            CallSite(name, positional, keywords, star, double_star, receiver, call.start_byte, call.end_byte)
        )
        function = ctx.function
        if function is not None and name == function.name:
            if (receiver is None and function.plain_calls) or (receiver is not None and receiver in function.receivers):
                self.has_recursion = True

    def _identifier(self, node: Node, ctx: _Context) -> _Pushes:
        mode = ctx.mode
        if mode == _SKIP or node.is_missing:
            return []
        name = self._raw(node)  # keywords from error recovery are dropped by _bind and by build()
        if not name:
            return []
        if mode == _LOAD:
            self.referenced.append((node.start_byte, name))
        elif mode == _DECLARE:
            self.declared.append((node.start_byte, name))
        elif mode == _PATTERN:
            if name != "_":
                self._bind(node, name, ctx)
        else:
            if mode == _AUGMENT:
                self.referenced.append((node.start_byte, name))
            self._bind(node, name, ctx)
        return []

    # -- Python ------------------------------------------------------------------------------------------------

    def _python_handlers(self) -> dict[str, _Handler]:
        handlers: dict[str, _Handler] = {
            "identifier": self._identifier,
            "import_statement": self._py_import,
            "import_from_statement": self._py_import_from,
            "future_import_statement": self._py_import_from,
            "function_definition": self._py_function,
            "class_definition": self._py_class,
            "lambda": self._py_lambda,
            "assignment": self._py_assignment,
            "augmented_assignment": self._py_augmented_assignment,
            "named_expression": self._py_named_expression,
            "for_statement": self._py_for,
            "while_statement": self._py_while,
            "try_statement": self._py_try,
            "for_in_clause": self._py_for_in_clause,
            "call": self._py_call,
            "attribute": self._py_attribute,
            "keyword_argument": self._py_keyword_argument,
            "global_statement": self._py_global,
            "nonlocal_statement": self._py_global,
            "type_alias_statement": self._py_type_alias,
            "as_pattern": self._py_as_pattern,
            "dotted_name": self._py_dotted_name,
            "class_pattern": self._py_class_pattern,
            "keyword_pattern": self._py_keyword_pattern,
            "dict_pattern": self._py_dict_pattern,
        }
        for block in (
            "if_statement",
            "elif_clause",
            "else_clause",
            "except_clause",
            "except_group_clause",
            "finally_clause",
            "with_statement",
            "match_statement",
            "case_clause",
        ):
            handlers[block] = self._py_block
        for parameter in ("typed_parameter", "default_parameter", "typed_default_parameter"):
            handlers[parameter] = self._py_parameter
        for comprehension in _COMPREHENSIONS:
            handlers[comprehension] = self._py_comprehension
        return handlers

    def _py_import(self, node: Node, ctx: _Context) -> _Pushes:
        names: list[_Piece] = []
        first_module: str | None = None
        for child, field in self._fields(node):
            if field != "name":
                continue
            if child.type == "aliased_import":
                module = self._name(child.child_by_field_name("name"))
                bound = child.child_by_field_name("alias")
                bound_name = self._raw(bound)
            else:
                module = self._name(child)
                bound = self._first_named(child, ("identifier",))
                bound_name = self._raw(bound)
            if not module:
                continue
            self.imports.append((child.start_byte, module))
            if bound is not None:
                self._declare_in_scope(bound, bound_name, ctx)
            names.append(_span(child))
            first_module = first_module or module
        if names:
            start, end = self._extent(node)
            self._slot(SlotKind.IMPORT, ["import ", *_joined(names, ", ")], first_module, start, end, ctx.depth)
        return self._skip(node, ctx)

    def _py_import_from(self, node: Node, ctx: _Context) -> _Pushes:
        module_node = node.child_by_field_name("module_name")
        if node.type == "future_import_statement":
            module = "__future__"
        else:
            module = self._name(module_node)
        module_start = node.start_byte if module_node is None else module_node.start_byte
        names: list[_Piece] = []
        for child, field in self._fields(node):
            if child.type == "wildcard_import":
                names.append("*")
            elif field == "name":
                bound = child.child_by_field_name("alias") if child.type == "aliased_import" else child
                self._declare_in_scope(bound, self._name(bound), ctx)
                names.append(_span(child))
        if module:
            self.imports.append((module_start, module))
            template = [f"from {module} import ", *_joined(names, ", ")]
            self._slot(SlotKind.IMPORT, template, module, *self._extent(node), ctx.depth)
        return self._skip(node, ctx)

    def _extent(self, node: Node, named_only: bool = False) -> tuple[int, int]:
        """[start, end) of a statement without trailing extras (and, if ``named_only``, without a ';')."""
        end = node.start_byte
        for child in node.children:
            if child.type not in self.extra_types and (child.is_named or not named_only):
                end = child.end_byte
        return node.start_byte, end

    def _py_parameters(self, params: Node) -> tuple[tuple[_PendingParameter, ...], list[_Piece], list[str]]:
        """Parameters of a ``parameters`` / ``lambda_parameters`` node, their text spans and their names."""
        parameters: list[_PendingParameter] = []
        spans: list[_Piece] = []
        names: list[str] = []
        keyword_only = False
        for child in params.children:
            child_type = child.type
            if not child.is_named or child_type in self.extra_types:
                continue
            spans.append(_span(child))
            if child_type == "keyword_separator":
                keyword_only = True
                continue
            if child_type == "positional_separator":
                continue
            annotation: tuple[_Piece, ...] | None = None
            has_default = False
            target: Node | None = child
            if child_type in ("typed_parameter", "default_parameter", "typed_default_parameter"):
                type_node = child.child_by_field_name("type")
                annotation = None if type_node is None else (_span(type_node),)
                has_default = child.child_by_field_name("value") is not None
                target = child.child_by_field_name("name") or self._first_named(
                    child, ("identifier", "list_splat_pattern", "dictionary_splat_pattern")
                )
            if target is None:
                continue
            if target.type == "list_splat_pattern":
                kind, keyword_only = "var_positional", True
                target = self._first_named(target, ("identifier",))
            elif target.type == "dictionary_splat_pattern":
                kind = "var_keyword"
                target = self._first_named(target, ("identifier",))
            elif target.type in ("identifier", "tuple_pattern"):
                kind = "keyword_only" if keyword_only else "positional"
            else:
                continue  # ERROR and other recovery nodes: kept in the text, not a parameter
            name = _collapse(self._raw(target))
            if not name:
                continue
            parameters.append(_PendingParameter(name, annotation, None, has_default, kind))
            names.append(name)
        return tuple(parameters), spans, names

    def _py_type_parameters(self, node: Node, ctx: _Context) -> _Pushes:
        """PEP 695 ``[T, U: int, *Ts, **P]``: declare the parameter names, load bounds and defaults."""
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child in node.children:
            if not child.is_named:
                continue
            inner = self._first_named(child) if child.type == "type" else None
            if inner is not None and inner.type == "identifier":
                self._declare(inner, self._raw(inner))
            elif inner is not None and inner.type in ("constrained_type", "splat_type"):
                parts = [c for c in inner.children if c.is_named and c.type not in self.extra_types]
                head = parts[0] if parts else None
                if head is not None and head.type == "type":
                    head = self._first_named(head)
                if head is not None and head.type == "identifier":
                    self._declare(head, self._raw(head))
                    pushes.extend((part, load) for part in parts[1:])
                else:
                    pushes.append((child, load))
            else:
                pushes.append((child, load))
        return pushes

    def _py_function(self, node: Node, ctx: _Context) -> _Pushes:
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        if name_node is None or params_node is None:
            return self._generic(node, ctx)
        name = self._raw(name_node)
        self._declare_in_scope(name_node, name, ctx)
        is_async = bool(node.children) and node.children[0].type == "async"
        type_params = node.child_by_field_name("type_parameters")
        return_type = node.child_by_field_name("return_type")
        parameters, param_spans, param_names = self._py_parameters(params_node)
        template: list[_Piece] = ["async def " if is_async else "def ", _span(name_node)]
        if type_params is not None:
            template.append(_span(type_params))
        template += ["(", *_joined(param_spans, ", "), ")"]
        end = params_node.end_byte
        if return_type is not None:
            template += [" -> ", _span(return_type)]
            end = return_type.end_byte
        is_method = ctx.scope.kind == "class"
        slot = self._slot(SlotKind.FUNCTION, template, name, node.start_byte, end, ctx.depth)
        returns = None if return_type is None else (_span(return_type),)
        self.functions.append(_PendingFunction(slot, name, parameters, returns, is_method, is_async, True))
        self.num_functions += 1
        if is_method:
            receivers = frozenset({"self", "cls"} | ({ctx.class_name} if ctx.class_name else set()))
            function = _FunctionContext(name, receivers, plain_calls=False)
        else:
            function = _FunctionContext(name, frozenset(), plain_calls=True)
        body = _Context(ctx.depth + 1, _Scope("function", param_names), _LOAD, function, 0, None)
        load, declare = _with_mode(ctx, _LOAD), _with_mode(ctx, _DECLARE)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "name":
                continue
            if field == "body":
                pushes.append((child, body))
            elif field == "parameters":
                pushes.append((child, declare))
            elif field == "type_parameters":
                pushes.extend(self._py_type_parameters(child, ctx))
            else:
                pushes.append((child, load))
        return pushes

    def _py_class(self, node: Node, ctx: _Context) -> _Pushes:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return self._generic(node, ctx)
        name = self._raw(name_node)
        self._declare_in_scope(name_node, name, ctx)
        type_params = node.child_by_field_name("type_parameters")
        superclasses = node.child_by_field_name("superclasses")
        template: list[_Piece] = ["class ", _span(name_node)]
        end = name_node.end_byte
        if type_params is not None:
            template.append(_span(type_params))
            end = type_params.end_byte
        if superclasses is not None:
            bases = [_span(c) for c in superclasses.children if c.is_named and c.type not in self.extra_types]
            if bases:
                template += ["(", *_joined(bases, ", "), ")"]
            end = superclasses.end_byte
        self._slot(SlotKind.CLASS, template, name, node.start_byte, end, ctx.depth)
        self.classes.append((node.start_byte, name))
        body = _Context(ctx.depth + 1, _Scope("class"), _LOAD, None, 0, name)
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "name":
                continue
            if field == "body":
                pushes.append((child, body))
            elif field == "type_parameters":
                pushes.extend(self._py_type_parameters(child, ctx))
            else:
                pushes.append((child, load))
        return pushes

    def _py_lambda(self, node: Node, ctx: _Context) -> _Pushes:
        params = node.child_by_field_name("parameters")
        names = [] if params is None else self._py_parameters(params)[2]
        body = ctx._replace(depth=ctx.depth + 1, scope=_Scope("lambda", names), mode=_LOAD)
        load, declare = _with_mode(ctx, _LOAD), _with_mode(ctx, _DECLARE)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "parameters":
                pushes.append((child, declare))
            elif field == "body":
                pushes.append((child, body))
            else:
                pushes.append((child, load))
        return pushes

    def _py_parameter(self, node: Node, ctx: _Context) -> _Pushes:
        load, declare = _with_mode(ctx, _LOAD), _with_mode(ctx, _DECLARE)
        return [(child, load if field in ("type", "value") else declare) for child, field in self._fields(node)]

    def _py_assignment(self, node: Node, ctx: _Context) -> _Pushes:
        type_node = node.child_by_field_name("type")
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "left":
                if type_node is not None and child.type == "identifier":
                    self._bind(child, self._raw(child), ctx, annotation=type_node)
                else:
                    pushes.append((child, _with_mode(ctx, _BIND)))
            else:
                pushes.append((child, load))
        return pushes

    def _py_augmented_assignment(self, node: Node, ctx: _Context) -> _Pushes:
        load, augment = _with_mode(ctx, _LOAD), _with_mode(ctx, _AUGMENT)
        return [(child, augment if field == "left" else load) for child, field in self._fields(node)]

    def _py_named_expression(self, node: Node, ctx: _Context) -> _Pushes:
        load, bind = _with_mode(ctx, _LOAD), _with_mode(ctx, _BIND)
        return [(child, bind if field == "name" else load) for child, field in self._fields(node)]

    def _py_block(self, node: Node, ctx: _Context, loop: bool = False) -> _Pushes:
        """A BLOCK construct: header slot at ``ctx.depth``, body one level deeper, clauses at the same depth."""
        self._header_slot(node, ctx)
        loop_depth = self._enter_loop(ctx) if loop else ctx.loop_depth
        body = ctx._replace(depth=ctx.depth + 1, mode=_LOAD, loop_depth=loop_depth)
        load, bind, pattern = _with_mode(ctx, _LOAD), _with_mode(ctx, _BIND), _with_mode(ctx, _PATTERN)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field in _BODY_FIELDS or child.type == "block":
                pushes.append((child, body))
            elif field in ("left", "alias"):
                pushes.append((child, bind))
            elif child.type == "case_pattern":
                pushes.append((child, pattern))
            else:
                pushes.append((child, load))  # conditions, subjects, guards, alternatives and clauses
        return pushes

    def _py_for(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_for = True
        return self._py_block(node, ctx, loop=True)

    def _py_while(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_while = True
        return self._py_block(node, ctx, loop=True)

    def _py_try(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_try = True
        return self._py_block(node, ctx)

    def _py_comprehension(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_comprehension = True
        return self._generic(node, ctx)

    def _py_for_in_clause(self, node: Node, ctx: _Context) -> _Pushes:
        load, declare = _with_mode(ctx, _LOAD), _with_mode(ctx, _DECLARE)
        return [(child, declare if field == "left" else load) for child, field in self._fields(node)]

    def _py_arguments(self, arguments: Node | None) -> tuple[int, tuple[str, ...], bool, bool]:
        if arguments is None:
            return 0, (), False, False
        if arguments.type == "generator_expression":
            return 1, (), False, False
        positional, keywords, star, double_star = 0, [], False, False
        for child in arguments.children:
            child_type = child.type
            if not child.is_named or child_type in self.extra_types:
                continue
            if child_type == "keyword_argument":
                keywords.append(self._raw(child.child_by_field_name("name")))
            elif child_type in ("list_splat", "parenthesized_list_splat"):
                star = True
            elif child_type == "dictionary_splat":
                double_star = True
            else:
                positional += 1
        return positional, tuple(keywords), star, double_star

    def _py_call(self, node: Node, ctx: _Context) -> _Pushes:
        callee = node.child_by_field_name("function")
        if callee is not None:
            name, receiver = "", None
            if callee.type == "identifier":
                name = self._raw(callee)
            elif callee.type == "attribute":
                name = self._raw(callee.child_by_field_name("attribute"))
                receiver = self._receiver_text(callee.child_by_field_name("object"))
            if name:
                arguments = self._py_arguments(node.child_by_field_name("arguments"))
                self._record_call(node, callee, name, receiver, arguments, ctx)
        return self._generic(node, _with_mode(ctx, _LOAD))

    def _py_attribute(self, node: Node, ctx: _Context) -> _Pushes:
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "attribute":
                name = self._raw(child)
                if name and not child.is_missing:
                    self.attributes.append((child.start_byte, name))
            else:
                pushes.append((child, load))
        return pushes

    def _py_keyword_argument(self, node: Node, ctx: _Context) -> _Pushes:
        load = _with_mode(ctx, _LOAD)
        return [(child, load) for child, field in self._fields(node) if field != "name"]

    def _py_global(self, node: Node, ctx: _Context) -> _Pushes:
        for child in node.children:
            if child.type == "identifier":
                self._declare_in_scope(child, self._raw(child), ctx)
        return self._skip(node, ctx)

    def _py_type_alias(self, node: Node, ctx: _Context) -> _Pushes:
        left = node.child_by_field_name("left")
        head = None if left is None else self._first_named(left)
        for _ in range(_MAX_CHAIN):
            if head is None or head.type == "identifier":
                break
            head = self._first_named(head)  # ``type A[T] = ...``: generic_type -> identifier
        if head is not None and head.type == "identifier":
            name = self._raw(head)
            self._declare(head, name)
            if name and name not in ctx.scope.bound:
                ctx.scope.bound.add(name)
                start, end = self._extent(node)
                self._slot(SlotKind.VARIABLE, (_Span(start, end),), name, start, end, ctx.depth)
        load, skip = _with_mode(ctx, _LOAD), _with_mode(ctx, _SKIP)
        return [(child, skip if field == "left" else load) for child, field in self._fields(node)]

    def _py_as_pattern(self, node: Node, ctx: _Context) -> _Pushes:
        """``expr as target`` (with, except) and ``pattern as name`` (case): the target always binds."""
        fields = self._fields(node)
        alias = next((child for child, field in fields if field == "alias"), None)
        if alias is None:
            named = [child for child, _ in fields if child.type not in self.extra_types]
            if len(named) >= 2 and named[-1].type == "identifier":
                alias = named[-1]  # ``case P() as name``: the alias has no field name
        inner = _with_mode(ctx, _PATTERN if ctx.mode == _PATTERN else _LOAD)
        bind = _with_mode(ctx, _BIND)
        return [(child, bind if alias is not None and child == alias else inner) for child, _ in fields]

    def _py_dotted_name(self, node: Node, ctx: _Context) -> _Pushes:
        parts = [c for c in node.children if c.type == "identifier" and not c.is_missing]
        if ctx.mode == _PATTERN and len(parts) == 1:
            name = self._raw(parts[0])
            if name != "_":
                self._bind(parts[0], name, ctx)  # a capture pattern
        elif parts:
            head = self._raw(parts[0])
            if head and head not in self.keywords:
                self.referenced.append((parts[0].start_byte, head))  # a value pattern / dotted reference
            for part in parts[1:]:
                self.attributes.append((part.start_byte, self._raw(part)))
        skip = _with_mode(ctx, _SKIP)
        return [(child, skip) for child in node.children if child.is_named and child.type != "identifier"]

    def _py_class_pattern(self, node: Node, ctx: _Context) -> _Pushes:
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        class_seen = False
        for child in node.children:
            if not child.is_named:
                continue
            if not class_seen and child.type == "dotted_name":
                class_seen = True
                pushes.append((child, load))  # the class is a reference, not a capture
            else:
                pushes.append((child, ctx))
        return pushes

    def _py_keyword_pattern(self, node: Node, ctx: _Context) -> _Pushes:
        children = [c for c in node.children if c.is_named]
        if children and children[0].type == "identifier":
            children = children[1:]  # the keyword names an attribute, it binds nothing
        return [(child, ctx) for child in children]

    def _py_dict_pattern(self, node: Node, ctx: _Context) -> _Pushes:
        load = _with_mode(ctx, _LOAD)
        return [(child, load if field == "key" else ctx) for child, field in self._fields(node)]

    # -- C++ ---------------------------------------------------------------------------------------------------

    def _cpp_handlers(self) -> dict[str, _Handler]:
        handlers: dict[str, _Handler] = {
            "identifier": self._identifier,
            "preproc_include": self._cpp_include,
            "preproc_def": self._cpp_define,
            "preproc_function_def": self._cpp_define,
            "preproc_call": self._cpp_preproc_call,
            "function_definition": self._cpp_function,
            "declaration": self._cpp_declaration,
            "field_declaration": self._cpp_declaration,
            "type_definition": self._cpp_declaration,
            "alias_declaration": self._cpp_alias,
            "type_parameter_declaration": self._cpp_type_parameter,
            "optional_type_parameter_declaration": self._cpp_type_parameter,
            "variadic_type_parameter_declaration": self._cpp_type_parameter,
            "enumerator": self._cpp_enumerator,
            "namespace_definition": self._cpp_namespace,
            "compound_statement": self._cpp_compound,
            "for_statement": self._cpp_for,
            "for_range_loop": self._cpp_for_range,
            "while_statement": self._cpp_while,
            "do_statement": self._cpp_do,
            "if_statement": self._cpp_if,
            "else_clause": self._cpp_else,
            "switch_statement": self._cpp_switch,
            "case_statement": self._cpp_case,
            "try_statement": self._cpp_try,
            "catch_clause": self._cpp_catch,
            "call_expression": self._cpp_call,
            "field_expression": self._cpp_field_expression,
            "lambda_expression": self._cpp_lambda,
            "qualified_identifier": self._skip,
            "using_declaration": self._skip,
            "attribute_declaration": self._skip,
        }
        for conditional in ("preproc_ifdef", "preproc_if", "preproc_elif", "preproc_elifdef"):
            handlers[conditional] = self._cpp_preproc_conditional
        for parameter in _CPP_PARAMETER_TYPES:
            handlers[parameter] = self._cpp_parameter
        for class_type in _CPP_CLASS_TYPES:
            handlers[class_type] = self._cpp_class
        return handlers

    def _cpp_name(self, node: Node | None) -> str:
        """The unqualified name: ``Foo::bar`` -> ``bar``, ``max<int>`` -> ``max``, ``~Foo`` stays."""
        for _ in range(_MAX_CHAIN):
            if node is None or node.type not in ("qualified_identifier", "template_function", "template_type",
                                                 "template_method"):  # fmt: skip
                break
            inner = node.child_by_field_name("name")
            if inner is None:
                break
            node = inner
        return _collapse(self._raw(node))

    def _cpp_unwrap(self, declarator: Node) -> tuple[list[Node], Node, Node | None, list[tuple[Node, int]]]:
        """Follow a declarator chain to its name(s).

        Returns ``(names, core, function, side)``: the declared name nodes, the declarator without its
        initializer, the function_declarator when the chain declares a function (not a function pointer), and
        the nodes to traverse with their modes (array sizes and initializers load, parameter lists declare).
        """
        side: list[tuple[Node, int]] = []
        names: list[Node] = []
        core = declarator
        function: Node | None = None
        node: Node | None = declarator
        for _ in range(_MAX_CHAIN):
            if node is None:
                break
            node_type = node.type
            if node_type in _CPP_NAME_TYPES:
                names.append(node)
                break
            if node_type == "structured_binding_declarator":
                names.extend(c for c in node.children if c.type == "identifier")
                break
            chain = node.child_by_field_name("declarator")
            if chain is None:  # reference / parenthesized declarators: the inner declarator has no field name
                candidates = [c for c in node.children if c.is_named and c.type not in self.extra_types]
                candidates = [c for c in candidates if c.type not in ("type_qualifier", "attribute_declaration")]
                chain = candidates[-1] if candidates else None
            if node_type == "function_declarator" and function is None and chain is not None:
                if chain.type in _CPP_NAME_TYPES:
                    function = node
            for child, field in self._fields(node):
                if chain is not None and child == chain:
                    continue
                mode = _DECLARE if field == "parameters" else _LOAD
                side.append((child, mode))
            if node_type == "init_declarator" and chain is not None:
                core = chain
            node = chain
        return names, core, function, side

    def _cpp_type_piece(self, node: Node) -> _Span:
        """A type specifier for slot texts: a struct / class / enum with a body contributes its header only."""
        if node.type in _CPP_CLASS_TYPES:
            body = node.child_by_field_name("body")
            if body is not None:
                return _Span(node.start_byte, self._header_end(node))
        return _span(node)

    def _cpp_parameters(
        self, function_declarator: Node
    ) -> tuple[tuple[_PendingParameter, ...], list[_Piece], list[str]]:
        params = function_declarator.child_by_field_name("parameters")
        if params is None:
            return (), [], []
        parameters: list[_PendingParameter] = []
        spans: list[_Piece] = []
        names: list[str] = []
        named = [c for c in params.children if c.is_named and c.type not in self.extra_types]
        for child in params.children:
            child_type = child.type
            if child_type == "...":
                spans.append(_span(child))
                parameters.append(_PendingParameter("...", None, None, False, "var_positional"))
                continue
            if not child.is_named or child_type in self.extra_types:
                continue
            spans.append(_span(child))
            if child_type not in _CPP_PARAMETER_TYPES:
                continue
            declarator = child.child_by_field_name("declarator")
            if (
                len(named) == 1
                and declarator is None
                and child_type == "parameter_declaration"
                and _collapse(self._raw(child)) == "void"
            ):
                continue  # ``f(void)`` declares no parameter
            type_pieces = [
                self._cpp_type_piece(c)
                for c, f in self._fields(child)
                if f not in ("declarator", "default_value") and c.type not in self.extra_types
            ]
            name, suffix = "", None
            if declarator is not None:
                found, core, _, _ = self._cpp_unwrap(declarator)
                if found:
                    name_node = found[0]
                    name = self._cpp_name(name_node)
                    suffix = (_Span(core.start_byte, name_node.start_byte), _Span(name_node.end_byte, core.end_byte))
                else:
                    suffix = (_span(core),)
            kind = "var_positional" if child_type == "variadic_parameter_declaration" else "positional"
            has_default = child_type == "optional_parameter_declaration"
            annotation = tuple(_joined(type_pieces, " ")) or None
            parameters.append(_PendingParameter(name, annotation, suffix, has_default, kind))
            if name:
                names.append(name)
        return tuple(parameters), spans, names

    def _cpp_signature(
        self, prefix: Sequence[Node], declarator: Node, function_declarator: Node
    ) -> tuple[list[_Piece], tuple[_PendingParameter, ...], list[str]]:
        parameters, param_spans, names = self._cpp_parameters(function_declarator)
        template: list[_Piece] = []
        for node in prefix:
            template += [self._cpp_type_piece(node), " "]
        params = function_declarator.child_by_field_name("parameters")
        if params is None:
            template.append(_span(declarator))
        else:
            template += [_Span(declarator.start_byte, params.start_byte), "(", *_joined(param_spans, ", "), ")"]
            template.append(_Span(params.end_byte, declarator.end_byte))
        return template, parameters, names

    def _cpp_return_type(
        self, owner: Node, declarator: Node, function_declarator: Node
    ) -> tuple[_Piece, ...] | None:
        type_node = owner.child_by_field_name("type")
        if type_node is None:
            return None  # constructors, destructors, conversion operators
        pieces = [
            self._cpp_type_piece(c)
            for c, f in self._fields(owner)
            if (f == "type" or c.type == "type_qualifier") and c.end_byte <= declarator.start_byte
        ]
        template: list[_Piece] = _joined(pieces, " ")
        if function_declarator.start_byte > declarator.start_byte:  # ``int *f()``: the pointer is the return's
            template += [" ", _Span(declarator.start_byte, function_declarator.start_byte)]
        return tuple(template)

    def _cpp_prefix(self, fields: Sequence[tuple[Node, str | None]]) -> list[Node]:
        """Specifiers and type before the first declarator (``static``, ``const``, ``ap_uint<8>``)."""
        prefix: list[Node] = []
        for child, field in fields:
            if field == "declarator":
                break
            if child.type not in self.extra_types:
                prefix.append(child)
        return prefix

    def _cpp_function(self, node: Node, ctx: _Context) -> _Pushes:
        declarator = node.child_by_field_name("declarator")
        if declarator is None:
            return self._generic(node, ctx)
        names, _, function_declarator, side = self._cpp_unwrap(declarator)
        if function_declarator is None or not names:
            return self._generic(node, ctx)
        name = self._cpp_name(names[0])
        self._declare_in_scope(names[0], name, ctx)
        fields = self._fields(node)
        prefix = self._cpp_prefix(fields)
        template, parameters, param_names = self._cpp_signature(prefix, declarator, function_declarator)
        is_method = ctx.scope.kind == "class"
        slot = self._slot(SlotKind.FUNCTION, template, name, node.start_byte, declarator.end_byte, ctx.depth)
        returns = self._cpp_return_type(node, declarator, function_declarator)
        self.functions.append(_PendingFunction(slot, name, parameters, returns, is_method, False, True))
        self.num_functions += 1
        function = _FunctionContext(name, frozenset({"this"}), plain_calls=True)
        body = _Context(ctx.depth + 1, _Scope("function", param_names), _LOAD, function, 0, None)
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in fields:
            if field == "declarator":
                pushes.extend((n, _with_mode(ctx, mode)) for n, mode in side)
            elif field == "body":
                pushes.append((child, body))
            else:
                pushes.append((child, load))
        return pushes

    def _cpp_declaration(self, node: Node, ctx: _Context) -> _Pushes:
        """``declaration`` / ``field_declaration`` / ``type_definition``: one slot per declarator."""
        fields = self._fields(node)
        prefix = self._cpp_prefix(fields)
        prefix_template: list[_Piece] = ["typedef "] if node.type == "type_definition" else []
        prefix_template += _joined([self._cpp_type_piece(n) for n in prefix], " ")
        bitfield = next((c for c, _ in fields if c.type == "bitfield_clause"), None)
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        first = True
        for child, field in fields:
            if field != "declarator":
                pushes.append((child, load))
                continue
            names, core, function_declarator, side = self._cpp_unwrap(child)
            pushes.extend((n, _with_mode(ctx, mode)) for n, mode in side)
            start = node.start_byte if first else child.start_byte
            first = False
            if function_declarator is not None and names and node.type != "type_definition":
                name = self._cpp_name(names[0])
                self._declare_in_scope(names[0], name, ctx)
                template, parameters, _ = self._cpp_signature(prefix, child, function_declarator)
                slot = self._slot(SlotKind.FUNCTION, template, name, start, child.end_byte, ctx.depth)
                returns = self._cpp_return_type(node, child, function_declarator)
                is_method = ctx.scope.kind == "class"
                self.functions.append(_PendingFunction(slot, name, parameters, returns, is_method, False, False))
                continue
            for name_node in names:
                name = self._cpp_name(name_node)
                if not name or name in self.keywords:
                    continue
                self.declared.append((name_node.start_byte, name))
                if name in ctx.scope.bound:
                    continue
                ctx.scope.bound.add(name)
                template = [*prefix_template, " ", _span(core)]
                end = core.end_byte
                if bitfield is not None and bitfield.start_byte >= core.end_byte:
                    template += [" ", _span(bitfield)]
                    end = bitfield.end_byte
                self._slot(SlotKind.VARIABLE, template, name, start, end, ctx.depth)
        return pushes

    def _cpp_alias(self, node: Node, ctx: _Context) -> _Pushes:
        """``using word_t = ap_int<16>;`` -> VARIABLE "using word_t = ap_int<16>"."""
        name_node = node.child_by_field_name("name")
        name = self._cpp_name(name_node)
        if name_node is not None and name:
            self.declared.append((name_node.start_byte, name))
            if name not in ctx.scope.bound:
                ctx.scope.bound.add(name)
                end = self._extent(node, named_only=True)[1]  # without the ';'
                self._slot(SlotKind.VARIABLE, (_Span(node.start_byte, end),), name, node.start_byte, end, ctx.depth)
        load = _with_mode(ctx, _LOAD)
        return [(child, load) for child, field in self._fields(node) if field != "name"]

    def _cpp_parameter(self, node: Node, ctx: _Context) -> _Pushes:
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "declarator":
                names, _, _, side = self._cpp_unwrap(child)
                for name_node in names:
                    self._declare(name_node, self._cpp_name(name_node))
                pushes.extend((n, _with_mode(ctx, mode)) for n, mode in side)
            else:
                pushes.append((child, load))
        return pushes

    def _cpp_type_parameter(self, node: Node, ctx: _Context) -> _Pushes:
        name_node = node.child_by_field_name("name") or self._first_named(node, ("type_identifier",))
        if name_node is not None:
            self._declare(name_node, self._raw(name_node))
        return self._skip(node, ctx)

    def _cpp_enumerator(self, node: Node, ctx: _Context) -> _Pushes:
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "name":
                self._declare_in_scope(child, self._raw(child), ctx)
            else:
                pushes.append((child, load))
        return pushes

    def _cpp_class(self, node: Node, ctx: _Context) -> _Pushes:
        body_node = node.child_by_field_name("body")
        if body_node is None:
            return self._skip(node, ctx)  # a type reference or a forward declaration
        name_node = node.child_by_field_name("name")
        name = self._cpp_name(name_node) if name_node is not None else ""
        if name_node is not None and name:
            self._declare_in_scope(name_node, name, ctx)
            self.classes.append((node.start_byte, name))
        end = self._header_end(node)
        self._slot(SlotKind.CLASS, (_Span(node.start_byte, end),), name or None, node.start_byte, end, ctx.depth)
        body = _Context(ctx.depth + 1, _Scope("class"), _LOAD, None, 0, name or None)
        skip, load = _with_mode(ctx, _SKIP), _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "body":
                pushes.append((child, body))
            elif field in ("name", "base") or child.type == "base_class_clause":
                pushes.append((child, skip))
            else:
                pushes.append((child, load))
        return pushes

    def _cpp_namespace(self, node: Node, ctx: _Context) -> _Pushes:
        inner = ctx._replace(scope=_Scope("namespace"), mode=_LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "name":
                self._declare(child, self._raw(child))
            else:
                pushes.append((child, inner if field == "body" else _with_mode(ctx, _LOAD)))
        return pushes

    def _cpp_compound(self, node: Node, ctx: _Context) -> _Pushes:
        block = ctx._replace(scope=_Scope("block"), mode=_LOAD)
        return [(child, block) for child in node.children if child.is_named]

    def _cpp_include(self, node: Node, ctx: _Context) -> _Pushes:
        path = node.child_by_field_name("path")
        if path is not None:
            text = _collapse(self._raw(path))
            module = text[1:-1] if len(text) >= 2 and text[0] + text[-1] in ('<>', '""') else text
            if module:
                self.imports.append((path.start_byte, module))
                template = ("#include ", _span(path))
                self._slot(SlotKind.IMPORT, template, module, node.start_byte, path.end_byte, ctx.depth)
        return self._skip(node, ctx)

    def _cpp_define(self, node: Node, ctx: _Context) -> _Pushes:
        name_node = node.child_by_field_name("name")
        name = self._raw(name_node)
        if name_node is not None and name:
            self.declared.append((name_node.start_byte, name))
            if name not in ctx.scope.bound:
                ctx.scope.bound.add(name)
                template: list[_Piece] = ["#define ", _span(name_node)]
                end = name_node.end_byte
                params = node.child_by_field_name("parameters")
                value = node.child_by_field_name("value")
                if params is not None:  # function-like macro: the signature, not the body
                    template.append(_span(params))
                    end = params.end_byte
                elif value is not None:
                    template += [" ", _Span(value.start_byte, value.end_byte, preproc=True)]
                    end = value.end_byte
                self._slot(SlotKind.VARIABLE, template, name, node.start_byte, end, ctx.depth)
        return self._skip(node, ctx)

    def _cpp_preproc_call(self, node: Node, ctx: _Context) -> _Pushes:
        directive = node.child_by_field_name("directive")
        if directive is not None and self._name(directive) == "#pragma":
            argument = node.child_by_field_name("argument")
            if argument is None:
                self._slot(SlotKind.DIRECTIVE, ("#pragma",), None, node.start_byte, directive.end_byte, ctx.depth)
            else:
                template = ("#pragma ", _Span(argument.start_byte, argument.end_byte, preproc=True))
                self._slot(SlotKind.DIRECTIVE, template, None, node.start_byte, argument.end_byte, ctx.depth)
        return self._skip(node, ctx)

    def _cpp_preproc_conditional(self, node: Node, ctx: _Context) -> _Pushes:
        skip = _with_mode(ctx, _SKIP)
        return [(child, skip if field in ("name", "condition") else ctx) for child, field in self._fields(node)]

    def _cpp_statement(
        self, node: Node, ctx: _Context, loop: bool, prefix: str = "", start: int | None = None
    ) -> _Pushes:
        """A BLOCK statement: header slot at ``ctx.depth``; its own scope; body one level deeper."""
        self._header_slot(node, ctx, prefix=prefix, start=start)
        statement = ctx._replace(scope=_Scope("block"), mode=_LOAD)
        loop_depth = self._enter_loop(ctx) if loop else ctx.loop_depth
        body = statement._replace(depth=ctx.depth + 1, loop_depth=loop_depth)
        outer = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field in _BODY_FIELDS:
                pushes.append((child, body))
            elif field == "alternative":
                pushes.append((child, outer))
            else:
                pushes.append((child, statement))
        return pushes

    def _cpp_for(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_for = True
        return self._cpp_statement(node, ctx, loop=True)

    def _cpp_while(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_while = True
        return self._cpp_statement(node, ctx, loop=True)

    def _cpp_switch(self, node: Node, ctx: _Context) -> _Pushes:
        return self._cpp_statement(node, ctx, loop=False)

    def _cpp_try(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_try = True
        return self._cpp_statement(node, ctx, loop=False)

    def _cpp_catch(self, node: Node, ctx: _Context) -> _Pushes:
        return self._cpp_statement(node, ctx, loop=False)

    def _cpp_if(self, node: Node, ctx: _Context, else_start: int | None = None) -> _Pushes:
        prefix = "" if else_start is None else "else "
        return self._cpp_statement(node, ctx, loop=False, prefix=prefix, start=else_start)

    def _cpp_else(self, node: Node, ctx: _Context) -> _Pushes:
        statement = self._first_named(node)
        pushes: _Pushes = [(c, ctx) for c in node.children if c.is_named and c.type in self.extra_types]
        if statement is not None and statement.type == "if_statement":
            return pushes + self._cpp_if(statement, ctx, else_start=node.start_byte)  # ``else if``: one header
        keyword_end = node.children[0].end_byte if node.children else node.start_byte
        template = (_Span(node.start_byte, keyword_end),)
        self._slot(SlotKind.BLOCK, template, None, node.start_byte, keyword_end, ctx.depth)
        if statement is not None:
            pushes.append((statement, ctx._replace(depth=ctx.depth + 1, mode=_LOAD)))
        return pushes

    def _cpp_do(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_while = True
        condition = node.child_by_field_name("condition")
        keyword_end = node.children[0].end_byte if node.children else node.start_byte
        template: tuple[_Piece, ...] = ("do",) if condition is None else ("do while ", _span(condition))
        self._slot(SlotKind.BLOCK, template, None, node.start_byte, keyword_end, ctx.depth)
        body = ctx._replace(depth=ctx.depth + 1, mode=_LOAD, scope=_Scope("block"), loop_depth=self._enter_loop(ctx))
        load = _with_mode(ctx, _LOAD)
        return [(child, body if field == "body" else load) for child, field in self._fields(node)]

    def _cpp_case(self, node: Node, ctx: _Context) -> _Pushes:
        self._header_slot(node, ctx)
        body = ctx._replace(depth=ctx.depth + 1, mode=_LOAD)
        load = _with_mode(ctx, _LOAD)
        return [(child, load if field == "value" else body) for child, field in self._fields(node)]

    def _cpp_for_range(self, node: Node, ctx: _Context) -> _Pushes:
        self.has_for = True
        self._header_slot(node, ctx)
        statement = ctx._replace(scope=_Scope("block"), mode=_LOAD)
        body = statement._replace(depth=ctx.depth + 1, loop_depth=self._enter_loop(ctx))
        fields = self._fields(node)
        prefix = [self._cpp_type_piece(n) for n in self._cpp_prefix(fields)]
        pushes: _Pushes = []
        for child, field in fields:
            if field == "body":
                pushes.append((child, body))
            elif field == "declarator":
                names, core, _, side = self._cpp_unwrap(child)
                pushes.extend((n, _with_mode(statement, mode)) for n, mode in side)
                for name_node in names:
                    name = self._cpp_name(name_node)
                    if not name or name in self.keywords:
                        continue
                    self.declared.append((name_node.start_byte, name))
                    if name not in statement.scope.bound:
                        statement.scope.bound.add(name)
                        start = prefix[0].start if prefix else core.start_byte
                        template = [*_joined(prefix, " "), " ", _span(core)]
                        self._slot(SlotKind.VARIABLE, template, name, start, core.end_byte, ctx.depth)
            else:
                pushes.append((child, statement))
        return pushes

    def _cpp_arguments(self, arguments: Node | None) -> tuple[int, tuple[str, ...], bool, bool]:
        if arguments is None:
            return 0, (), False, False
        count = sum(1 for c in arguments.children if c.is_named and c.type not in self.extra_types)
        return count, (), False, False

    def _cpp_call(self, node: Node, ctx: _Context) -> _Pushes:
        callee = node.child_by_field_name("function")
        if callee is not None:
            name, receiver = "", None
            if callee.type in ("identifier", "qualified_identifier", "template_function"):
                name = self._cpp_name(callee)
            elif callee.type == "field_expression":
                name = self._cpp_name(callee.child_by_field_name("field"))
                receiver = self._receiver_text(callee.child_by_field_name("argument"))
            if name:
                arguments = self._cpp_arguments(node.child_by_field_name("arguments"))
                self._record_call(node, callee, name, receiver, arguments, ctx)
        return self._generic(node, _with_mode(ctx, _LOAD))

    def _cpp_field_expression(self, node: Node, ctx: _Context) -> _Pushes:
        load = _with_mode(ctx, _LOAD)
        pushes: _Pushes = []
        for child, field in self._fields(node):
            if field == "field":
                name = self._cpp_name(child)
                if name:
                    self.attributes.append((child.start_byte, name))
            else:
                pushes.append((child, load))
        return pushes

    def _cpp_lambda(self, node: Node, ctx: _Context) -> _Pushes:
        body = ctx._replace(depth=ctx.depth + 1, scope=_Scope("lambda"), mode=_LOAD)
        load = _with_mode(ctx, _LOAD)
        return [(child, body if field == "body" else load) for child, field in self._fields(node)]

    # -- phase 2: texts and records ----------------------------------------------------------------------------

    def _slice(self, span: _Span) -> str:
        start, end = max(0, span.start), min(len(self.data), span.end)
        if end <= start:
            return ""
        if span.preproc:
            text = _LINE_SPLICE.sub("", self.data[start:end].decode("utf-8", errors="replace"))
            return _strip_c_comments(text)
        pieces: list[bytes] = []
        position = start
        index = bisect.bisect_left(self.extra_starts, start)
        while index < len(self.extras) and self.extras[index][0] < end:
            extra_start, extra_end = self.extras[index]
            if extra_start >= position:
                pieces += [self.data[position:extra_start], b" "]
                position = min(extra_end, end)
            index += 1
        pieces.append(self.data[position:end])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def _render(self, template: Iterable[_Piece]) -> str:
        text = "".join(piece if isinstance(piece, str) else self._slice(piece) for piece in template)
        if not self.python:
            text = _LINE_SPLICE.sub("", text)
        return _collapse(text)

    def _parameter(self, pending: _PendingParameter) -> Parameter:
        annotation = self._render(pending.annotation) if pending.annotation else ""
        if pending.suffix is not None:
            suffix = self._render(pending.suffix)
            if suffix:
                glue = "" if (not annotation or suffix.startswith(("[", "..."))) else " "
                annotation = f"{annotation}{glue}{suffix}"
        return Parameter(pending.name, annotation or None, pending.has_default, pending.kind)

    def build(self, has_errors: bool, num_top_level_statements: int) -> ProgramStructure:
        self.extras.sort()
        self.extra_starts = [start for start, _ in self.extras]
        texts: list[str] = []
        for pending in self.slots:
            try:
                texts.append(self._render(pending.template))
            except _NODE_ERRORS as exc:
                self.failed = has_errors = True
                self.warnings.append(f"slot at byte {pending.start}: {type(exc).__name__}: {exc}")
                texts.append("")
        order = sorted(range(len(self.slots)), key=lambda i: (self.slots[i].start, self.slots[i].end, i))
        slots = tuple(
            StructuralSlot(s.kind, texts[i], s.name, s.start, s.end, s.depth)
            for i in order
            if texts[i]
            for s in (self.slots[i],)
        )
        functions: list[tuple[int, FunctionSignature]] = []
        for pending in self.functions:
            text = texts[pending.slot]
            if not text:
                continue
            try:
                parameters = tuple(self._parameter(p) for p in pending.parameters)
                returns = self._render(pending.return_annotation) if pending.return_annotation else ""
            except _NODE_ERRORS as exc:
                has_errors = True
                self.warnings.append(f"signature of {pending.name!r}: {type(exc).__name__}: {exc}")
                continue
            signature = FunctionSignature(
                pending.name, parameters, returns or None, pending.is_method, text, pending.is_async, pending.has_body
            )
            functions.append((self.slots[pending.slot].start, signature))
        functions.sort(key=lambda item: item[0])
        features = ControlFlowFeatures(
            has_for=self.has_for,
            has_while=self.has_while,
            has_recursion=self.has_recursion,
            has_comprehension=self.has_comprehension,
            has_try=self.has_try,
            num_functions=self.num_functions,
            max_loop_depth=self.max_loop_depth,
            num_top_level_statements=num_top_level_statements,
        )
        return ProgramStructure(
            language=self.language,
            slots=slots,
            imports=_unique(self.imports),
            functions=tuple(signature for _, signature in functions),
            classes=tuple(name for _, name in sorted(self.classes, key=lambda item: item[0])),
            declared_identifiers=_unique(self.declared, self.keywords),
            referenced_identifiers=_unique(self.referenced, self.keywords),
            attribute_names=_unique(self.attributes, self.keywords),
            calls=_unique(self.calls, self.keywords),
            features=features,
            has_errors=has_errors,
            call_sites=tuple(sorted(self.call_sites, key=lambda c: (c.start_byte, c.end_byte))),
            warnings=tuple(self.warnings),
        )


def _unique(items: list[tuple[int, str]], drop: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """Names deduplicated in source order (first occurrence), empty names and ``drop`` removed."""
    seen: set[str] = set()
    out: list[str] = []
    for _, name in sorted(items, key=lambda item: item[0]):
        if name and name not in seen and name not in drop:
            seen.add(name)
            out.append(name)
    return tuple(out)
