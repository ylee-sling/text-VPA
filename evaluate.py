# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
evaluate.py -- functional correctness (Pass@k) of a trained text-VPA checkpoint on HumanEval or MBPP.

    python evaluate.py --checkpoint runs/s2/best_stage2.pt --benchmark humaneval \\
        --data data/eval/HumanEval.jsonl.gz --num-samples 10 --k 1 10 --out runs/eval/humaneval
    python evaluate.py --benchmark mbpp --data data/eval/mbpp.jsonl --reference --out runs/eval/mbpp-reference

For every problem, n samples are drawn with Algorithm 1 (K + 3 sequential evaluations each) from the same
context the training loaders build; each sample is run against the problem's tests in a separate,
resource-limited process (``text_vpa.evaluation.execution``: isolation, not a security sandbox -- run it in a
disposable VM). pass@k is the unbiased estimator of Chen et al. (2021). ``--reference`` runs the canonical
solutions instead of a model (harness check: HumanEval 164/164; MBPP test split 499/500, task 367's tests use
an undefined class).

Writes ``samples.jsonl`` (one line per sample: task, text, status) and ``summary.json`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from text_vpa.configs.schema import RealizerConfig, SamplerConfig
from text_vpa.evaluation import (
    load_humaneval_tasks,
    load_mbpp_tasks,
    run_programs,
    summarize_pass_at_k,
)
from text_vpa.evaluation.tasks import MBPP_SPLITS, EvalTask
from text_vpa.inference.loading import build_sampler, load_model
from text_vpa.utils.device import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pass@k of a text-VPA checkpoint", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", choices=("humaneval", "mbpp"), required=True)
    p.add_argument("--data", type=Path, required=True, help="HumanEval.jsonl(.gz) or mbpp.jsonl")
    p.add_argument("--mbpp-split", choices=sorted(MBPP_SPLITS), default="test")
    p.add_argument("--checkpoint", type=Path, help="Stage-2 or Stage-3 checkpoint")
    p.add_argument("--reference", action="store_true", help="run the canonical solutions (harness check)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=10, help="n samples per problem")
    p.add_argument("--k", type=int, nargs="+", default=[1, 10, 100], help="pass@k values (k <= n)")
    p.add_argument("--k-steps", type=int, default=None, help="realizer steps K (default: the checkpoint's)")
    p.add_argument("--batch-size", type=int, default=16, help="samples generated per sampler call")
    p.add_argument("--mode-strategy", choices=("sample", "greedy", "gumbel"), default=None)
    p.add_argument("--temperature", type=float, default=None, help="energy temperature T")
    p.add_argument("--no-energy", action="store_true", help="disable the constraint energies and the repair")
    p.add_argument("--limit", type=int, default=None, help="evaluate the first N problems only")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds per test program")
    p.add_argument("--exec-workers", type=int, default=4, help="test programs run in parallel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def load_tasks(args: argparse.Namespace) -> list[EvalTask]:
    tasks = (load_humaneval_tasks(args.data) if args.benchmark == "humaneval"
             else load_mbpp_tasks(args.data, args.mbpp_split))
    return tasks[: args.limit] if args.limit else tasks


def generate(args: argparse.Namespace, tasks: list[EvalTask]) -> tuple[list[list[str]], dict]:
    device = resolve_device(args.device)
    loaded = load_model(args.checkpoint, device)
    cfg = loaded.cfg
    rcfg = cfg.realizer
    if args.temperature is not None:
        rcfg = RealizerConfig.model_validate({**rcfg.model_dump(), "temperature": args.temperature})
    sampler = build_sampler(loaded, use_energy=not args.no_energy)
    sampler.rcfg = rcfg
    if args.mode_strategy:
        sampler.scfg = SamplerConfig.model_validate({**cfg.sampler.model_dump(), "mode_strategy": args.mode_strategy})
    k_steps = args.k_steps or rcfg.num_steps
    tok = loaded.tokenizer
    samples: list[list[str]] = []
    seconds = 0.0
    depth = None
    stats = {"hard_violations": 0, "unrepaired": 0, "unrealized": 0}
    for t_index, task in enumerate(tasks):
        ids = tok.encode(task.context)[: cfg.data.max_context_tokens] or [tok.pad_id]
        texts: list[str] = []
        generator = torch.Generator(device=device).manual_seed(args.seed * 100_003 + t_index)
        while len(texts) < args.num_samples:
            rows = min(args.batch_size, args.num_samples - len(texts))
            context = torch.tensor([ids] * rows, device=device)
            start = time.perf_counter()
            result = sampler.sample(context, num_steps=k_steps, generator=generator)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds += time.perf_counter() - start
            depth = result.depth
            texts += result.texts
            stats["hard_violations"] += result.hard_violations
            stats["unrepaired"] += result.unrepaired
            stats["unrealized"] += result.unrealized
        samples.append(texts)
        if (t_index + 1) % 20 == 0:
            print(f"  generated {t_index + 1}/{len(tasks)} problems", file=sys.stderr)
    info = {"checkpoint": str(args.checkpoint), "stage": loaded.stage, "step": loaded.step,
            "realizer": rcfg.kind, "k_steps": k_steps, "sequential_depth": depth, "device": str(device),
            "energy": not args.no_energy, "generation_seconds": round(seconds, 3),
            "seconds_per_problem": round(seconds / max(len(tasks), 1), 4), **stats}
    return samples, info


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reference == (args.checkpoint is not None):
        raise SystemExit("give exactly one of --checkpoint and --reference")
    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("no problem to evaluate")
    if args.reference:
        samples, info = [[t.reference] for t in tasks], {"reference": True}
    else:
        samples, info = generate(args, tasks)
    programs = [task.program(s) for task, group in zip(tasks, samples, strict=True) for s in group]
    start = time.perf_counter()
    results = run_programs(programs, args.timeout, args.exec_workers)
    info["execution_seconds"] = round(time.perf_counter() - start, 3)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = []
    index = 0
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as handle:
        for task, group in zip(tasks, samples, strict=True):
            passed = 0
            for j, text in enumerate(group):
                r = results[index]
                index += 1
                passed += r.passed
                handle.write(json.dumps({"task_id": task.task_id, "sample": j, "passed": r.passed,
                                         "status": r.status, "detail": r.detail, "completion": text}) + "\n")
            counts.append((len(group), passed))
    n = min(c[0] for c in counts)
    summary = {"benchmark": args.benchmark, "split": args.mbpp_split if args.benchmark == "mbpp" else "test",
               "problems": len(tasks), "samples_per_problem": n,
               **summarize_pass_at_k(counts, [k for k in args.k if k <= n] or [1]),
               "solved_problems": sum(1 for _, c in counts if c > 0), **info}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
