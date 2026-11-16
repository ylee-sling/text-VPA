# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
postprocess.py -- from generated text to the program the harness runs; one rule for both decoders.

A language model that continues a prompt does not stop where the function ends, so every completion-style
evaluation truncates. The same truncation is applied to the realizer's canvas and to the autoregressive
baseline's continuation:

    complete (HumanEval)   keep lines up to the first non-blank line that starts in column 0 (the body is
                           indented; this covers Codex's stop sequences "\\ndef", "\\nclass", "\\nif", "\\nprint",
                           "\\n#"). The program is prompt + kept lines.
    write (MBPP)           keep lines up to the first line that starts with one of ``WRITE_STOPS`` (the BigCode
                           harness's MBPP stop sequences, without "\\nclass": helper classes are legitimate).

``chat`` (the instruction-tuned reference row only): the first fenced code block of the reply, else the reply.
"""

from __future__ import annotations

import ast
import re

from text_vpa.evaluation.tasks import EvalTask

__all__ = ["WRITE_STOPS", "build_sample", "extract_fenced", "is_complete", "parses", "truncate_complete",
           "truncate_write"]

WRITE_STOPS: tuple[str, ...] = ("assert ", "print(", "if __name__", '"""', "'''", "```", "<")
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def _cut(text: str, kind: str) -> tuple[str, bool]:
    kept: list[str] = []
    for line in text.split("\n"):
        stop = (bool(line) and not line[0].isspace()) if kind == "complete" else line.startswith(WRITE_STOPS)
        if stop:
            return "\n".join(kept), True
        kept.append(line)
    return "\n".join(kept), False


def truncate_complete(text: str) -> str:
    return _cut(text, "complete")[0].rstrip() + "\n"


def truncate_write(text: str) -> str:
    return _cut(text, "write")[0].rstrip() + "\n"


def is_complete(text: str, kind: str) -> bool:
    """True once the truncation rule has found its stop: generating further cannot change the program."""
    return _cut(text, kind)[1]


def extract_fenced(reply: str) -> str:
    match = _FENCE.search(reply)
    return (match.group(1) if match else reply).rstrip() + "\n"


def build_sample(task: EvalTask, benchmark: str, text: str, chat: bool = False) -> str:
    """The ``sample`` of ``EvalTask.program`` (a whole program) from a decoder's text."""
    if chat:
        return extract_fenced(text)
    if benchmark == "humaneval":
        return task.context + truncate_complete(text)
    return truncate_write(text)


def parses(program: str) -> bool:
    try:
        ast.parse(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return True
