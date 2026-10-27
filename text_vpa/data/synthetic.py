# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
synthetic.py -- a small generator of specification -> Python program pairs with known design modes.

For tests, smoke runs and debugging only; it is not a benchmark. Each record pairs a natural-language
specification with a signature (the context c) and a short, valid Python function (the target w_1).
Templates cover every control-flow class of the ``control`` mode scheme (straight line, for loop,
while loop, recursion, comprehension) and several library buckets (math, collections, re), so the
mode selector sees genuinely multimodal data. Names and constants are randomised; every program is
checked with ``ast.parse``.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass

__all__ = ["CodeRecord", "synthetic_corpus"]


@dataclass(frozen=True)
class CodeRecord:
    """One training / evaluation pair."""

    context: str
    target: str
    language: str = "python"
    task_id: str | None = None


_FUNCS = ["solve", "compute", "process", "transform", "evaluate", "combine", "measure", "reduce_values", "score"]
_LISTS = ["xs", "values", "items", "numbers", "data"]
_VARS = ["x", "v", "item", "n", "value"]


def _for_sum(r: random.Random, f: str) -> tuple[str, str]:
    xs, x = r.choice(_LISTS), r.choice(_VARS)
    k = r.randint(2, 9)
    ctx = f"Write {f}({xs}) that returns the sum of {k} times each element of a list of integers using a for loop."
    code = (f"from typing import List\n\n\ndef {f}({xs}: List[int]) -> int:\n    total = 0\n"
            f"    for {x} in {xs}:\n        total += {k} * {x}\n    return total\n")
    return ctx, code


def _while_gcd(r: random.Random, f: str) -> tuple[str, str]:
    a, b = r.sample(["a", "b", "p", "q", "m"], 2)
    ctx = f"Write {f}({a}, {b}) returning the greatest common divisor of two integers with a while loop."
    code = f"def {f}({a}: int, {b}: int) -> int:\n    while {b}:\n        {a}, {b} = {b}, {a} % {b}\n    return {a}\n"
    return ctx, code


def _recursion(r: random.Random, f: str) -> tuple[str, str]:
    n = r.choice(["n", "k", "m"])
    base = r.randint(1, 3)
    ctx = f"Write a recursive function {f}({n}) that returns the factorial of {n}, with {f}({n}) = 1 for {n} < {base + 1}."
    code = f"def {f}({n}: int) -> int:\n    if {n} <= {base}:\n        return 1\n    return {n} * {f}({n} - 1)\n"
    return ctx, code


def _comprehension(r: random.Random, f: str) -> tuple[str, str]:
    xs, x = r.choice(_LISTS), r.choice(_VARS)
    k = r.randint(2, 5)
    ctx = f"Write {f}({xs}) returning the even elements of a list multiplied by {k}, using a list comprehension."
    code = (f"from typing import List\n\n\ndef {f}({xs}: List[int]) -> List[int]:\n"
            f"    return [{x} * {k} for {x} in {xs} if {x} % 2 == 0]\n")
    return ctx, code


def _straight(r: random.Random, f: str) -> tuple[str, str]:
    a, b, c = r.sample(["a", "b", "c", "x", "y", "z"], 3)
    ctx = f"Write {f}({a}, {b}, {c}) that returns {a} * {b} + {c}."
    code = f"def {f}({a}: int, {b}: int, {c}: int) -> int:\n    return {a} * {b} + {c}\n"
    return ctx, code


def _math(r: random.Random, f: str) -> tuple[str, str]:
    rad = r.choice(["r", "radius", "rad"])
    ctx = f"Write {f}({rad}) returning the area of a circle of radius {rad}; import math."
    code = f"import math\n\n\ndef {f}({rad}: float) -> float:\n    return math.pi * {rad} ** 2\n"
    return ctx, code


def _counter(r: random.Random, f: str) -> tuple[str, str]:
    words = r.choice(["words", "tokens", "names"])
    ctx = f"Write {f}({words}) returning the most common string of a list, using collections.Counter."
    code = (f"from collections import Counter\nfrom typing import List\n\n\ndef {f}({words}: List[str]) -> str:\n"
            f"    return Counter({words}).most_common(1)[0][0]\n")
    return ctx, code


def _regex(r: random.Random, f: str) -> tuple[str, str]:
    text = r.choice(["text", "line", "s"])
    ctx = f"Write {f}({text}) returning every run of digits in a string as a list of strings, using re."
    code = (f"import re\nfrom typing import List\n\n\ndef {f}({text}: str) -> List[str]:\n"
            f"    return re.findall(r\"\\d+\", {text})\n")
    return ctx, code


TEMPLATES = (_for_sum, _while_gcd, _recursion, _comprehension, _straight, _math, _counter, _regex)


def synthetic_corpus(num_records: int, seed: int = 0, templates: tuple = TEMPLATES) -> list[CodeRecord]:
    """``num_records`` records cycling through ``templates`` with randomised names and constants."""
    rng = random.Random(seed)
    out = []
    for i in range(num_records):
        template = templates[i % len(templates)]
        name = rng.choice(_FUNCS)
        context, code = template(rng, name)
        ast.parse(code)                      # every target must be valid Python
        out.append(CodeRecord(context=context, target=code, task_id=f"synthetic/{i}"))
    return out
