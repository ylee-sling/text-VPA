# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
analyze.py -- how the samples of every evaluated configuration failed.

    python -m text_vpa.stage0.analyze --eval runs/stage0/eval

Reads ``EVAL/<benchmark>/<tag>/samples.jsonl`` (written by ``evaluate.py``) and sorts every sample into one
outcome, in this order:

    passed      the program exits with status 0
    empty       the decoder produced no text
    syntax      the program does not parse (SyntaxError, IndentationError, TabError)
    wrong       it parses and runs, and a test assertion fails
    runtime     it parses and raises another exception (NameError, TypeError, ...)
    timeout     it parses and exceeds the time limit

The six fractions sum to 1 for each configuration. Also reported: the most frequent syntax-error messages, and
adjacent repeated identifiers per 100 words of output ("x x", "return return"), a symptom of positions
decided in parallel from the same context.

Writes ``EVAL/analysis.md`` and ``EVAL/analysis.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = ["OUTCOMES", "classify", "error_line", "repeated_identifiers", "summarize_samples", "tag_order"]

OUTCOMES: tuple[str, ...] = ("passed", "empty", "syntax", "wrong", "runtime", "timeout")
SYNTAX_ERRORS: tuple[str, ...] = ("SyntaxError", "IndentationError", "TabError")
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]")
_LOCATION = re.compile(r"\s*(\(.*\))?\s*(on line \d+)?\s*$")


def error_line(detail: str) -> str:
    """The last non-empty line of a traceback (``"IndentationError: expected an indented block ..."``)."""
    lines = [line.strip() for line in (detail or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _exception_name(line: str) -> str:
    head = line.split(":", 1)[0].strip()
    return head.rsplit(".", 1)[-1] if head and " " not in head else ""


def classify(row: dict[str, Any]) -> str:
    """One of ``OUTCOMES`` for a row of ``samples.jsonl``."""
    if row.get("passed"):
        return "passed"
    if not str(row.get("raw", "")).strip():
        return "empty"
    if not row.get("parses", True):
        return "syntax"
    if row.get("status") == "timeout":
        return "timeout"
    name = _exception_name(error_line(row.get("detail", "")))
    if name in SYNTAX_ERRORS:
        return "syntax"
    return "wrong" if name == "AssertionError" else "runtime"


def syntax_message(row: dict[str, Any]) -> str:
    """The syntax error without its location, e.g. ``"IndentationError: expected an indented block after
    function definition"`` or ``"SyntaxError: '(' was never closed"``."""
    line = error_line(row.get("detail", ""))
    if not line:
        return "(no message)"
    return _LOCATION.sub("", line) or line


def repeated_identifiers(text: str) -> tuple[int, int]:
    """(adjacent repeated identifiers, words) in ``text``."""
    words = _WORD.findall(text)
    repeats = sum(1 for a, b in zip(words, words[1:], strict=False) if a == b and a.isidentifier())
    return repeats, len(words)


def summarize_samples(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    messages: Counter[str] = Counter()
    repeats = words = 0
    total = 0
    for row in rows:
        total += 1
        outcome = classify(row)
        outcomes[outcome] += 1
        if outcome == "syntax":
            messages[syntax_message(row)] += 1
        r, w = repeated_identifiers(str(row.get("raw", "")))
        repeats, words = repeats + r, words + w
    fractions = {k: outcomes[k] / total if total else 0.0 for k in OUTCOMES}
    return {"samples": total, **fractions, "repeats_per_100_words": 100 * repeats / words if words else 0.0,
            "top_syntax_errors": messages.most_common(3)}


def tag_order(tag: str) -> tuple[int, int, str]:
    """ar first, then the realizer by K, the exact kernel before the confidence order; reference last."""
    if tag.startswith("ar"):
        return (0, 0, tag)
    match = re.match(r"realizer-K(\d+)(.*)", tag)
    if match:
        return (1, int(match.group(1)), match.group(2))
    return (2, 0, tag)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 0: how the samples failed", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="default: the --eval directory")
    args = p.parse_args(argv)
    out = args.out or args.eval

    result: dict[str, dict[str, Any]] = {}
    for bench_dir in sorted(d for d in args.eval.iterdir() if d.is_dir()):
        tags = sorted((d.name for d in bench_dir.iterdir() if (d / "samples.jsonl").exists()), key=tag_order)
        for tag in tags:
            with open(bench_dir / tag / "samples.jsonl", encoding="utf-8") as handle:
                result.setdefault(bench_dir.name, {})[tag] = summarize_samples(map(json.loads, handle))
    if not result:
        raise SystemExit(f"no samples.jsonl under {args.eval}/<benchmark>/<tag>/")

    lines = ["# Stage 0: how the samples failed", "",
             "Fractions of all samples; each row sums to 100%. Repeats: adjacent repeated identifiers per 100 words.",
             ""]
    for bench, tags in result.items():
        lines += [f"## {bench}", "", "| decoder | samples | " + " | ".join(OUTCOMES) + " | repeats | top syntax errors |",
                  "|---|---|" + "---|" * len(OUTCOMES) + "---|---|"]
        for tag, s in tags.items():
            errors = "; ".join(f"{m} ({n})" for m, n in s["top_syntax_errors"]) or "-"
            lines.append(f"| {tag} | {s['samples']} | " + " | ".join(f"{100 * s[k]:.1f}%" for k in OUTCOMES)
                         + f" | {s['repeats_per_100_words']:.2f} | {errors} |")
        lines.append("")
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.md").write_text(text)
    (out / "analysis.json").write_text(json.dumps(result, indent=2))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
