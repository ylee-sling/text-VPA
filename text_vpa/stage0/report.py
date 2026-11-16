# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
report.py -- the Stage-0 result: Pass@k of every configuration and the verdict on H2.

    python -m text_vpa.stage0.report --eval runs/stage0/eval --run runs/stage0

H2 (Text-VPA Sec. 6): at K = 8 the realizer reaches at least half of the autoregressive backbone's Pass@1.
For each benchmark the report gives the ratio

    rho = Pass@1(realizer, K = 8, exact jump kernel) / Pass@1(autoregressive)

and a 95% interval from a paired bootstrap over problems (both decoders are scored on the same resampled
problems; 10,000 resamples of the per-problem pass rates c_i / n). The decision rule, fixed before the run:

    holds            rho >= 0.5 on both benchmarks
    fails            rho <  0.5 on both benchmarks
    mixed            otherwise
    "clearly"        is added when every interval lies on the same side of 0.5

The learning curve (the trainer's Pass@1 probe on MBPP's validation split) is reported next to the verdict,
because a failure of a realizer that was still improving when training stopped is a statement about the
training budget, not about the method.

Reads ``EVAL/<benchmark>/<tag>/{summary.json, samples.jsonl}``; writes ``report.md`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["bootstrap_ratio", "h2_verdict", "learning_curve", "per_task_rates"]

THRESHOLD = 0.5
PRIMARY = "realizer-K8"
BASELINE = "ar"


def per_task_rates(samples_path: Path) -> dict[str, float]:
    """task_id -> fraction of its samples that passed."""
    passed: dict[str, list[bool]] = {}
    with open(samples_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            passed.setdefault(row["task_id"], []).append(bool(row["passed"]))
    return {task: sum(flags) / len(flags) for task, flags in passed.items()}


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, resamples: int = 10_000, seed: int = 0
                    ) -> tuple[float, float, float]:
    """mean(numerator) / mean(denominator) and its 95% paired-bootstrap interval (NaN where undefined)."""
    if numerator.shape != denominator.shape or numerator.ndim != 1 or numerator.size == 0:
        raise ValueError("need two equally long vectors of per-problem pass rates")
    base = denominator.mean()
    point = float(numerator.mean() / base) if base > 0 else float("nan")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, numerator.size, size=(resamples, numerator.size))
    num, den = numerator[index].mean(axis=1), denominator[index].mean(axis=1)
    ratios = num[den > 0] / den[den > 0]
    if ratios.size == 0:
        return point, float("nan"), float("nan")
    low, high = np.percentile(ratios, [2.5, 97.5])
    return point, float(low), float(high)


def h2_verdict(ratios: dict[str, tuple[float, float, float]]) -> str:
    """The decision rule of the module docstring on {benchmark: (rho, low, high)}."""
    if not ratios or any(np.isnan(r[0]) for r in ratios.values()):
        return "undecided (a benchmark is missing or its baseline solves nothing)"
    above = [r[0] >= THRESHOLD for r in ratios.values()]
    if all(above):
        clear = all(r[1] >= THRESHOLD for r in ratios.values())
        return ("clearly " if clear else "") + "holds"
    if not any(above):
        clear = all(r[2] < THRESHOLD for r in ratios.values())
        return ("clearly " if clear else "") + "fails"
    return "mixed"


def learning_curve(run_dir: Path) -> list[tuple[int, float]]:
    """(step, probe Pass@1) from the trainer's metrics."""
    path = run_dir / "metrics.jsonl"
    points: dict[int, float] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if "eval/probe_pass@1" in row:
                    points[int(row["step"])] = float(row["eval/probe_pass@1"])
    return sorted(points.items())


def curve_note(curve: list[tuple[int, float]]) -> str:
    if len(curve) < 4:
        return "too few probe points to judge convergence"
    values = [v for _, v in curve]
    recent, earlier = float(np.mean(values[-2:])), float(np.mean(values[-4:-2]))
    best = max(values)
    if best == 0:
        return "the probe never solved a problem"
    if recent > earlier + 0.1 * best:
        return ("still rising when training stopped (last two probes above the two before by more than 10% of "
                "the best): the result is budget-limited")
    return "flat over the last four probes: more steps of this recipe are unlikely to change the result"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 0 report and H2 verdict", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--run", type=Path, default=None, help="training run (learning curve)")
    p.add_argument("--out", type=Path, default=None, help="default: the --eval directory")
    args = p.parse_args(argv)
    out = args.out or args.eval

    table: list[dict[str, Any]] = []
    ratios: dict[str, tuple[float, float, float]] = {}
    for bench_dir in sorted(d for d in args.eval.iterdir() if d.is_dir()):
        summaries = {}
        for tag_dir in sorted(d for d in bench_dir.iterdir() if (d / "summary.json").exists()):
            summaries[tag_dir.name] = json.loads((tag_dir / "summary.json").read_text())
            s = summaries[tag_dir.name]
            table.append({"benchmark": bench_dir.name, "tag": tag_dir.name, "pass@1": s.get("pass@1"),
                          "pass@10": s.get("pass@10"), "parse_rate": s.get("parse_rate"),
                          "depth": s.get("sequential_depth", s.get("mean_sequential_depth")),
                          "mean_tokens": s.get("mean_tokens"), "problems": s.get("problems"),
                          "samples": s.get("samples_per_problem")})
        if PRIMARY in summaries and BASELINE in summaries:
            a = per_task_rates(bench_dir / PRIMARY / "samples.jsonl")
            b = per_task_rates(bench_dir / BASELINE / "samples.jsonl")
            if set(a) != set(b):
                raise SystemExit(f"{bench_dir.name}: the two decoders were scored on different problems")
            tasks = sorted(a)
            ratios[bench_dir.name] = bootstrap_ratio(np.array([a[t] for t in tasks]), np.array([b[t] for t in tasks]))

    verdict = h2_verdict(ratios)
    curve = learning_curve(args.run) if args.run else []
    lines = ["# Text-VPA Stage 0: plan-free Gemma 4 E2B realizer against its own autoregressive decoding", "",
             "| benchmark | decoder | Pass@1 | Pass@10 | parses | sequential depth | tokens |",
             "|---|---|---|---|---|---|---|"]

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{100 * value:.1f}%"

    for row in table:
        lines.append(f"| {row['benchmark']} | {row['tag']} | {pct(row['pass@1'])} | {pct(row['pass@10'])} | "
                     f"{pct(row['parse_rate'])} | {row['depth']} | {row['mean_tokens']} |")
    lines += ["", f"## H2: {verdict}", "",
              f"Ratio of Pass@1, `{PRIMARY}` over `{BASELINE}`; threshold {THRESHOLD}; 95% paired bootstrap over problems.",
              ""]
    for bench, (rho, low, high) in ratios.items():
        lines.append(f"- {bench}: rho = {rho:.3f} [{low:.3f}, {high:.3f}]")
    if not ratios:
        lines.append(f"- no benchmark has both `{PRIMARY}` and `{BASELINE}`")
    if curve:
        lines += ["", "## Learning curve (Pass@1 on MBPP validation, K = 8)", "",
                  ", ".join(f"{step}: {100 * value:.1f}%" for step, value in curve), "", curve_note(curve)]
    text = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    (out / "report.json").write_text(json.dumps({
        "verdict": verdict, "threshold": THRESHOLD, "primary": PRIMARY, "baseline": BASELINE,
        "ratios": {b: {"rho": r[0], "low": r[1], "high": r[2]} for b, r in ratios.items()},
        "table": table, "learning_curve": curve, "curve_note": curve_note(curve) if curve else None}, indent=2))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
