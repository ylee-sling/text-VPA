# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Tests of the Pass@k evaluation (``text_vpa.evaluation`` and ``evaluate.py``).

  * pass@k = E[1 - C(n - c, k) / C(n, k)] (Chen et al. 2021), checked against exact binomials and against a
    brute-force average over all k-subsets of the n samples
  * a program passes iff it exits with status 0 within the timeout; crashes, failed asserts, hangs and memory
    blow-ups fail without affecting the caller
  * HumanEval runs prompt + sample + test + check(entry_point); MBPP runs setup + sample + asserts; the
    canonical solutions pass (harness check)
  * evaluate.py writes one line per sample and a summary with the K + 3 depth
"""

from __future__ import annotations

import itertools
import json
import math
import time

import pytest

import evaluate as evaluate_cli
import train as train_cli
from text_vpa.evaluation import (
    load_humaneval_tasks,
    load_mbpp_tasks,
    pass_at_k,
    run_program,
    run_programs,
    summarize_pass_at_k,
)


# =============================================================================================
# pass@k
# =============================================================================================
@pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
def test_pass_at_k_equals_the_binomial_formula(n: int) -> None:
    for c in range(n + 1):
        for k in range(1, n + 1):
            expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
            assert pass_at_k(n, c, k) == pytest.approx(expected, rel=1e-12, abs=1e-12), (n, c, k)


def test_pass_at_k_is_the_mean_over_all_k_subsets() -> None:
    outcomes = [True, False, False, True, False, False, False]                # n = 7, c = 2
    for k in range(1, 8):
        subsets = list(itertools.combinations(outcomes, k))
        brute = sum(any(s) for s in subsets) / len(subsets)
        assert pass_at_k(7, 2, k) == pytest.approx(brute, rel=1e-12)


def test_pass_at_k_edge_cases_and_summary() -> None:
    assert pass_at_k(10, 0, 1) == 0.0 and pass_at_k(10, 10, 5) == 1.0 and pass_at_k(10, 1, 1) == pytest.approx(0.1)
    for bad in ((5, 6, 1), (5, -1, 1), (5, 2, 0), (5, 2, 6)):
        with pytest.raises(ValueError):
            pass_at_k(*bad)
    summary = summarize_pass_at_k([(10, 1), (10, 0), (10, 10)], [1, 10, 100])
    assert summary == pytest.approx({"pass@1": (0.1 + 0 + 1) / 3, "pass@10": (1 + 0 + 1) / 3})
    with pytest.raises(ValueError):
        summarize_pass_at_k([], [1])


# =============================================================================================
# Execution
# =============================================================================================
@pytest.mark.parametrize("program,status", [
    ("assert sorted([3, 1, 2]) == [1, 2, 3]\n", "passed"),
    ("import sys\nsys.exit(0)\n", "passed"),
    ("assert 1 + 1 == 3\n", "failed"),
    ("raise ValueError('boom')\n", "failed"),
    ("import sys\nsys.exit(3)\n", "failed"),
    ("def f(:\n", "failed"),
    ("x = bytearray(64 << 30)\n", "failed"),                     # address-space limit, not the host's memory
])
def test_run_program_status(program: str, status: str) -> None:
    result = run_program(program, timeout=10)
    assert result.status == status and result.passed is (status == "passed")
    if status == "failed":
        assert result.detail


def test_run_program_kills_a_hang_at_the_timeout() -> None:
    start = time.monotonic()
    result = run_program("while True:\n    pass\n", timeout=1.5)
    assert result.status == "timeout" and not result.passed
    assert time.monotonic() - start < 6


def test_run_program_isolation_environment_and_cwd(tmp_path) -> None:
    probe = ("import os, sys\n"
             "assert 'SECRET_TOKEN' not in os.environ\n"
             "assert os.path.basename(os.getcwd()).startswith('text-vpa-exec-')\n"
             "assert sys.flags.isolated == 1\n")
    import os

    os.environ["SECRET_TOKEN"] = "x"
    try:
        assert run_program(probe).passed
    finally:
        del os.environ["SECRET_TOKEN"]


def test_run_programs_keeps_the_input_order() -> None:
    programs = [f"assert {i} % 2 == 0\n" for i in range(9)]
    results = run_programs(programs, timeout=10, workers=4)
    assert [r.passed for r in results] == [i % 2 == 0 for i in range(9)]


# =============================================================================================
# Tasks
# =============================================================================================
HUMANEVAL = [
    {"task_id": "HumanEval/0", "entry_point": "add",
     "prompt": "from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    \"\"\"Sum of xs.\n    >>> add([1, 2])\n"
               "    3\n    \"\"\"\n",
     "canonical_solution": "    return sum(xs)\n",
     "test": "\n\nMETADATA = {}\n\n\ndef check(candidate):\n    assert candidate([1, 2]) == 3\n"
             "    assert candidate([]) == 0\n"},
    {"task_id": "HumanEval/1", "entry_point": "twice",
     "prompt": "def helper(x: int) -> int:\n    return 2 * x\n\n\ndef twice(x: int) -> int:\n"
               "    \"\"\"Twice x, using helper.\"\"\"\n",
     "canonical_solution": "    return helper(x)\n",
     "test": "def check(candidate):\n    assert candidate(4) == 8\n"},
]
MBPP = [
    {"task_id": 5, "text": "Write a function to square a number.", "code": "def square(x):\r\n  return x * x",
     "test_list": ["assert square(3) == 9", "assert square(0) == 0"], "test_setup_code": ""},
    {"task_id": 11, "text": "Write a function to reverse a list.", "code": "def rev(xs):\n    return xs[::-1]",
     "test_list": ["assert rev([1, 2]) == [2, 1]"], "test_setup_code": "import math"},
    {"task_id": 700, "text": "Write a function returning one.", "code": "def one():\n    return 1",
     "test_list": ["assert one() == 1"], "test_setup_code": ""},
]


@pytest.fixture
def bench(tmp_path):
    he = tmp_path / "HumanEval.jsonl"
    he.write_text("".join(json.dumps(r) + "\n" for r in HUMANEVAL))
    mb = tmp_path / "mbpp.jsonl"
    mb.write_text("".join(json.dumps(r) + "\n" for r in MBPP))
    return he, mb


def test_humaneval_programs_and_contexts(bench) -> None:
    tasks = load_humaneval_tasks(bench[0])
    assert [t.task_id for t in tasks] == ["HumanEval/0", "HumanEval/1"]
    first = tasks[0]
    assert first.context == HUMANEVAL[0]["prompt"]                        # = load_humaneval's context
    assert first.reference == HUMANEVAL[0]["prompt"] + HUMANEVAL[0]["canonical_solution"]
    sample = "from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    return sum(xs)\n\n"
    assert first.program(sample) == (HUMANEVAL[0]["prompt"].rstrip() + "\n\n\n" + sample.rstrip() + "\n"
                                     + "\n\n" + HUMANEVAL[0]["test"].strip() + "\n\n\ncheck(add)\n")
    assert all(run_program(t.program(t.reference)).passed for t in tasks)
    # the sample replaces the prompt's docstring-only definition; helpers of the prompt stay available
    assert run_program(tasks[1].program("def twice(x: int) -> int:\n    return helper(x)\n")).passed
    assert not run_program(tasks[0].program("def add(xs):\n    return 0\n")).passed
    assert not run_program(tasks[0].program("")).passed                  # the prompt alone returns None


@pytest.mark.parametrize("split,ids", [("test", ["MBPP/11"]), ("prompt", ["MBPP/5"]), ("train", ["MBPP/700"]),
                                       ("all", ["MBPP/5", "MBPP/11", "MBPP/700"])])
def test_mbpp_splits_programs_and_contexts(bench, split, ids) -> None:
    tasks = load_mbpp_tasks(bench[1], split)
    assert [t.task_id for t in tasks] == ids
    for t in tasks:
        row = next(r for r in MBPP if f"MBPP/{r['task_id']}" == t.task_id)
        assert t.context == row["text"] + "\n" + row["test_list"][0]      # = load_mbpp's context
        assert "\r" not in t.reference and run_program(t.program(t.reference)).passed
    if split == "test":
        assert tasks[0].prefix == "import math\n\n" and tasks[0].suffix == "\n\nassert rev([1, 2]) == [2, 1]\n"
    with pytest.raises(ValueError):
        load_mbpp_tasks(bench[1], "dev")


# =============================================================================================
# evaluate.py
# =============================================================================================
def test_evaluate_cli_reference_mode(bench, tmp_path) -> None:
    assert evaluate_cli.main(["--benchmark", "humaneval", "--data", str(bench[0]), "--reference",
                              "--out", str(tmp_path / "ref")]) == 0
    summary = json.loads((tmp_path / "ref" / "summary.json").read_text())
    assert summary["pass@1"] == 1.0 and summary["problems"] == 2 and summary["solved_problems"] == 2
    with pytest.raises(SystemExit):
        evaluate_cli.main(["--benchmark", "mbpp", "--data", str(bench[1]), "--out", str(tmp_path / "x")])


TINY = ["planner=tiny", "realizer=tiny", "data=tiny", "data.tokenizer=bpe", "data.bpe_vocab_size=400",
        "training.batch_size=4", "training.planner.warmup_steps=1", "training.realizer.warmup_steps=1",
        "training.save_every=100000"]


def test_evaluate_cli_with_a_trained_checkpoint(bench, tmp_path) -> None:
    out = tmp_path / "runs"
    assert train_cli.main(["--stage", "1", "--data", "synthetic:48", "--output-dir", str(out / "s1"),
                           "--device", "cpu", "--max-steps", "3", *TINY]) == 0
    assert train_cli.main(["--stage", "2", "--data", "synthetic:48", "--output-dir", str(out / "s2"),
                           "--device", "cpu", "--max-steps", "2", "--planner-checkpoint",
                           str(out / "s1" / "stage1_final.pt"), "realizer=tiny", "training.batch_size=4"]) == 0
    rc = evaluate_cli.main(["--benchmark", "mbpp", "--data", str(bench[1]), "--mbpp-split", "all",
                            "--checkpoint", str(out / "s2" / "best_stage2.pt"), "--num-samples", "3",
                            "--batch-size", "2", "--k", "1", "2", "10", "--k-steps", "2", "--device", "cpu",
                            "--out", str(tmp_path / "eval")])
    assert rc == 0
    summary = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert summary["problems"] == 3 and summary["samples_per_problem"] == 3
    assert set(summary) >= {"pass@1", "pass@2"} and "pass@10" not in summary            # k <= n only
    assert summary["k_steps"] == 2 and summary["sequential_depth"] == 2 + 3
    lines = [json.loads(x) for x in (tmp_path / "eval" / "samples.jsonl").read_text().splitlines()]
    assert len(lines) == 9 and {x["task_id"] for x in lines} == {"MBPP/5", "MBPP/11", "MBPP/700"}
    assert all(x["status"] in ("passed", "failed", "timeout") for x in lines)


# =============================================================================================
# Diagnostics after a Pass@k run (text_vpa.evaluation.diagnose)
# =============================================================================================
def test_diagnose_helpers_find_functions_and_score_programs() -> None:
    import difflib

    from text_vpa.evaluation.diagnose import Score, defined_functions, parses, requested_function

    prompt = 'def helper(x):\n    return x\n\n\nasync def target(a: int) -> int:\n    """Doc."""\n'
    assert defined_functions(prompt) == ["helper", "target"] and requested_function(prompt) == "target"
    assert requested_function("x = 1\n") is None
    assert parses("def f():\n    return 1\n")
    assert not parses("def all_with_list(003: Generator[0: = list =]) -> float:\n    pass\n")    # the L4 sample
    assert not parses("a = 1\x00")
    reference, broken = "def target(a):\n    return a\n", "def other(:\n"
    score = Score()
    score.add(reference, reference, "target", stalls=4.0, length_error=2.0)
    score.add(broken, reference, "target", stalls=0.0)
    assert (score.n, score.syntax, score.named, score.exact) == (2, 1, 1, 1)
    assert score.fraction("stalls") == 2.0 and score.fraction("length_error") == 1.0
    expected = (1.0 + difflib.SequenceMatcher(None, reference, broken, autojunk=False).ratio()) / 2
    assert score.fraction("similarity") == pytest.approx(expected, rel=1e-12)


def test_diagnose_reading_names_the_first_suspect(capsys) -> None:
    from text_vpa.evaluation.diagnose import Score, reading

    def score(syntax: float, similarity: float) -> Score:
        return Score(n=10, syntax=round(10 * syntax), similarity=10 * similarity)

    train = "training contexts (seen in Stage 2)"
    cases = [({train: score(0.9, 0.8)}, "memorized the corpus"),
             ({train: score(0.1, 0.1), train + " / reference length": score(0.2, 0.5)}, "length head"),
             ({train: score(0.1, 0.1), train + " / no energies": score(0.6, 0.1)}, "energies"),
             ({train: score(0.0, 0.05)}, "does not match training")]
    for verdict, expected in cases:
        reading(verdict)
        assert expected in capsys.readouterr().out


def test_diagnose_cli_on_a_trained_checkpoint(bench, tmp_path, capsys) -> None:
    import gzip

    from text_vpa.data.synthetic import synthetic_corpus
    from text_vpa.evaluation import diagnose

    run, data = tmp_path / "runs", tmp_path / "data"
    assert train_cli.main(["--stage", "1", "--data", "synthetic:48", "--output-dir", str(run / "stage1"),
                           "--device", "cpu", "--max-steps", "3", *TINY]) == 0
    assert train_cli.main(["--stage", "2", "--data", "synthetic:48", "--eval-data", "synthetic:8",
                           "--eval-every", "1", "--output-dir", str(run / "stage2"), "--device", "cpu",
                           "--max-steps", "2", "--planner-checkpoint", str(run / "stage1" / "stage1_final.pt"),
                           "realizer=tiny", "training.batch_size=4"]) == 0
    (data / "eval").mkdir(parents=True)
    with gzip.open(data / "eval" / "HumanEval.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in HUMANEVAL)
    assert evaluate_cli.main(["--benchmark", "humaneval", "--data", str(data / "eval" / "HumanEval.jsonl.gz"),
                              "--checkpoint", str(run / "stage2" / "best_stage2.pt"), "--num-samples", "2",
                              "--k", "1", "--k-steps", "2", "--device", "cpu",
                              "--out", str(run / "eval" / "humaneval")]) == 0
    summary = json.loads((run / "eval" / "humaneval" / "summary.json").read_text())
    assert summary["license_context"] is False and summary["inventory_threshold"] == 0.5     # the defaults
    assert evaluate_cli.main(["--benchmark", "humaneval", "--data", str(data / "eval" / "HumanEval.jsonl.gz"),
                              "--checkpoint", str(run / "stage2" / "best_stage2.pt"), "--num-samples", "1",
                              "--k", "1", "--k-steps", "2", "--device", "cpu", "--limit", "1",
                              "--license-context", "--inventory-threshold", "0.3",
                              "--out", str(tmp_path / "eval-licensed")]) == 0
    licensed = json.loads((tmp_path / "eval-licensed" / "summary.json").read_text())
    assert licensed["license_context"] is True and licensed["inventory_threshold"] == 0.3
    with pytest.raises(SystemExit):
        evaluate_cli.main(["--benchmark", "humaneval", "--data", str(data / "eval" / "HumanEval.jsonl.gz"),
                           "--checkpoint", str(run / "stage2" / "best_stage2.pt"), "--inventory-threshold", "1.5",
                           "--device", "cpu", "--out", str(tmp_path / "bad")])
    (data / "corpus").mkdir()
    records = synthetic_corpus(12, seed=3)
    for name, part in (("train-00000.jsonl.gz", records[:8]), ("val.jsonl.gz", records[8:])):
        with gzip.open(data / "corpus" / name, "wt", encoding="utf-8") as handle:
            handle.writelines(json.dumps({"context": r.context, "target": r.target}) + "\n" for r in part)
    capsys.readouterr()
    assert diagnose.main(["--run", str(run), "--data", str(data), "--examples", "3", "--k-steps", "2",
                          "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    best = load_checkpoint_step(run / "stage2" / "best_stage2.pt")
    assert f"best_stage2.pt holds step {best}" in out and "minimum" in out
    assert "4 completions:" in out and "the prompt asks for add()" in out
    assert out.count("as evaluated") >= 3 and "validation contexts (unseen): 3 examples, K = 2" in out
    assert out.count("context licensed") >= 2 and "plan-vocabulary energy: the readout licenses" in out
    assert "== Reading" in out


def load_checkpoint_step(path) -> int:
    from text_vpa.engine.runtime import load_checkpoint

    return int(load_checkpoint(path)["step"])
