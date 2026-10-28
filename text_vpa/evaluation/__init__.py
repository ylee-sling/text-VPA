# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Functional-correctness evaluation (PR-JEPA Sec. 5.2): Pass@k on HumanEval and MBPP."""

from text_vpa.evaluation.execution import ExecutionResult, run_program, run_programs
from text_vpa.evaluation.passk import pass_at_k, summarize_pass_at_k
from text_vpa.evaluation.tasks import EvalTask, load_humaneval_tasks, load_mbpp_tasks

__all__ = [
    "EvalTask",
    "ExecutionResult",
    "load_humaneval_tasks",
    "load_mbpp_tasks",
    "pass_at_k",
    "run_program",
    "run_programs",
    "summarize_pass_at_k",
]
