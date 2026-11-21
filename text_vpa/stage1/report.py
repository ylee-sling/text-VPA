# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
report.py -- the result of the structure oracle (docs/stage1.md, step 1) and the gate that follows it.

    python -m text_vpa.stage1.report --eval runs/stage1-oracle/sweep --run runs/stage1-oracle \\
        --reference runs/stage0-a/sweep

Everything here is a diagnostic on MBPP's validation split, and the report says so in its title and on every
row: an oracle run cannot write a report without ``ORACLE_LABEL`` (a summary without it is refused).

The gate after step 1, fixed on 21 Nov 2026 before anything was built. With

    rho = Pass@1(oracle plan, exact jump kernel, K = 8) / Pass@1(baseline: the released model's own generate)

and a 95% paired bootstrap over problems (``stage0.report.bootstrap_ratio``):

    rho >= 0.5          structure is enough in principle: build the planner (steps 3 and 4)
    0.1 <= rho < 0.5    H2 is out of reach for a structural plan with this realizer; steps 3 and 4 still measure
                        what the plan buys at K = 32 and K = 128, and step 2 says whether richer plans are worth
                        designing
    rho < 0.1           run step 2 (the latent oracle)

"clearly" is added when the whole interval lies in the same band.

Reading the oracle (added on 21 Nov 2026 while building it, before any run). A low rho says "structure is not
enough" only if the structure reached the realizer. That is checked where parallel decisions play no part: at
one token per step (``oracle-K256-left_to_right``) the samples must follow their plans, i.e. the mean share of
lines with the planned kind (``kind_match``) must be at least 0.9. Below that, the two lower bands of the gate
are not read as a statement about structure: the bridge or its training has to be looked at first, and the report
prints "not read" in place of the band (``gate_reading``). The band rho >= 0.5 needs no such check.

The baseline is ``REFERENCE/mbpp/ar`` (the Stage 0 sweep of the checkpoint the oracle started from) or, without
``--reference``, ``EVAL/mbpp/ar``; it must have been scored on the same problems. The reference's
``realizer-K*`` rows are the realizer without a plan and without the oracle run's extra training steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from text_vpa.stage0.report import bootstrap_ratio, per_task_rates
from text_vpa.stage1.oracle import PLAN_VARIANTS
from text_vpa.stage1.plan import ORACLE_LABEL

__all__ = ["BUILD", "CLOSED", "DELIVERY", "GATE_ROW", "bootstrap_difference", "delivery_row", "delivery_verdict",
           "gate_reading", "is_oracle_row", "render", "step1_gate"]

STAGE = "stage1-oracle"
BASELINE = "ar"
GATE_ROW = "oracle-K8"                                   # exact jump kernel, K = 8
DELIVERY_ROW = "oracle-K256-left_to_right"               # one token per step on the 256-token canvas
BUILD, CLOSED = 0.5, 0.1                                 # the gate's two thresholds
DELIVERY = 0.9                                           # kind_match at one token per step
K_ROWS: tuple[tuple[int, str], ...] = ((8, "random"), (32, "random"), (128, "random"), (8, "confidence"),
                                       (32, "confidence"), (128, "confidence"), (256, "left_to_right"))


def is_oracle_row(tag: str, summary: dict[str, Any]) -> bool:
    return summary.get("stage") == STAGE or tag.split("-K")[0] in PLAN_VARIANTS


def step1_gate(rho: float, low: float, high: float) -> str:
    """The gate of the module docstring on (rho, interval)."""
    if math.isnan(rho):
        return "undecided (the baseline solves nothing)"
    if rho >= BUILD:
        return ("clearly " if low >= BUILD else "") + "at least 0.5: structure is enough in principle; build the " \
            "planner (steps 3 and 4)"
    if rho >= CLOSED:
        clear = low >= CLOSED and high < BUILD
        return ("clearly " if clear else "") + "between 0.1 and 0.5: H2 is out of reach for a structural plan " \
            "with this realizer; steps 3 and 4 measure what the plan buys at K = 32 and K = 128, step 2 says " \
            "whether richer plans are worth designing"
    return ("clearly " if high < CLOSED else "") + "under 0.1: run step 2 (the latent oracle)"


def gate_reading(rho: float, low: float, high: float, kind_match: float | None) -> str:
    """The gate's band as it may be read: a band under 0.5 only once the delivery check has passed."""
    band = step1_gate(rho, low, high)
    if math.isnan(rho) or rho >= BUILD or (kind_match is not None and kind_match >= DELIVERY):
        return band
    return ("NOT READ: the delivery check below has not passed, so a ratio under 0.5 says nothing yet about "
            "structure (docs/stage1.md, 'Reading the oracle')")


def delivery_row(rows: dict[str, dict[str, Any]], plan: str = "oracle") -> str:
    """The tag of ``plan``'s one-token-per-step row: left to right with as many steps as canvas positions."""
    for tag, summary in rows.items():
        if (summary.get("plan") == plan and summary.get("order") == "left_to_right"
                and summary.get("num_steps") is not None and summary.get("num_steps") == summary.get("canvas_len")):
            return tag
    return DELIVERY_ROW.replace("oracle", plan, 1)


def delivery_verdict(kind_match: float | None) -> str:
    if kind_match is None:
        return ("not evaluated (the oracle's one-token-per-step row is missing): the gate's two lower bands "
                "cannot be read yet")
    if kind_match >= DELIVERY:
        return f"the structure reaches the realizer (kind match {kind_match:.3f} >= {DELIVERY})"
    return (f"NOT shown (kind match {kind_match:.3f} < {DELIVERY}): a ratio under 0.5 is not a statement about "
            "structure until the bridge and its training have been looked at")


def bootstrap_difference(a: np.ndarray, b: np.ndarray, resamples: int = 10_000, seed: int = 0
                         ) -> tuple[float, float, float]:
    """mean(a) - mean(b) and its 95% paired-bootstrap interval over problems."""
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("need two equally long vectors of per-problem pass rates")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, a.size, size=(resamples, a.size))
    diffs = a[index].mean(axis=1) - b[index].mean(axis=1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(low), float(high)


def _summaries(bench_dir: Path) -> dict[str, dict[str, Any]]:
    if not bench_dir.is_dir():
        return {}
    return {d.name: json.loads((d / "summary.json").read_text())
            for d in sorted(bench_dir.iterdir()) if (d / "summary.json").exists()}


def _paired(dir_a: Path, tag_a: str, dir_b: Path, tag_b: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-problem pass rates of two configurations, or None if they were scored on different problems."""
    a = per_task_rates(dir_a / tag_a / "samples.jsonl")
    b = per_task_rates(dir_b / tag_b / "samples.jsonl")
    if set(a) != set(b):
        return None
    tasks = sorted(a)
    return np.array([a[t] for t in tasks]), np.array([b[t] for t in tasks])


def _curve(run_dir: Path | None) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl" if run_dir else None
    points: dict[int, dict[str, Any]] = {}
    gate: dict[int, float] = {}
    if path is not None and path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if "eval/plan_gain_k0" in row or "eval/probe_pass@1" in row:
                    points[int(row["step"])] = {
                        "step": int(row["step"]), "probe_pass@1": row.get("eval/probe_pass@1"),
                        "probe_kind_match": row.get("eval/probe_kind_match"),
                        "plan_gain_k0": row.get("eval/plan_gain_k0"), "plan_gain": row.get("eval/plan_gain"),
                        "val_loss_k0": row.get("eval/val_loss_k0"), "val_loss_null_k0": row.get("eval/val_loss_null_k0")}
                if "train/gate_mean" in row:
                    gate[int(row["step"])] = float(row["train/gate_mean"])
    out = [points[step] for step in sorted(points)]
    for point in out:
        earlier = [s for s in gate if s <= point["step"]]
        point["gate_mean"] = gate[max(earlier)] if earlier else 0.0
    return out


def render(result: dict[str, Any]) -> str:
    """The report's text from its data (``main`` builds ``result``)."""
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{100 * value:.1f}%"

    def num(value: float | None, digits: int = 3) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [f"# Text-VPA Stage 1, step 1: the structure oracle ({ORACLE_LABEL})", "",
             "A diagnostic on MBPP's validation split: the realizer is told the line structure of each problem's "
             "reference program. No number below is a result of the method.", "",
             "| row | plan | Pass@1 | parses | mean depth | kind match | same line count |", "|---|---|---|---|---|---|---|"]
    for row in result["rows"]:
        lines.append(f"| {row['tag']} | {row['plan']} ({ORACLE_LABEL}) | {pct(row['pass@1'])} | "
                     f"{pct(row['parse_rate'])} | {row['depth']} | {num(row['kind_match'])} | "
                     f"{pct(row['line_count_match'])} |")
    comparison = result["comparison"]
    if comparison:
        lines += ["", "## By number of steps", "",
                  "Pass@1. \"no plan\": the realizer the oracle started from, in its own sweep. \"null\" and "
                  "\"shuffled\": the oracle run's weights with the null plan and with another problem's plan.", "",
                  "| K | order | no plan (reference run) | oracle | null | shuffled | oracle - null [95%] |",
                  "|---|---|---|---|---|---|---|"]
        for row in comparison:
            diff = row.get("oracle_minus_null")
            cell = "-" if diff is None else f"{100 * diff[0]:+.1f} [{100 * diff[1]:+.1f}, {100 * diff[2]:+.1f}]"
            lines.append(f"| {row['k']} | {row['order']} | {pct(row.get('no_plan'))} | {pct(row.get('oracle'))} | "
                         f"{pct(row.get('null'))} | {pct(row.get('shuffled'))} | {cell} |")
    gate = result["gate"]
    lines += ["", "## Gate after step 1", "",
              f"rho = Pass@1(`{GATE_ROW}`) / Pass@1(baseline `{BASELINE}`), 95% paired bootstrap over problems; "
              f"thresholds {BUILD} and {CLOSED} (docs/stage1.md).", ""]
    if gate.get("rho") is None:
        lines.append(f"- not evaluated: {gate['reason']}")
    else:
        lines.append(f"- {gate['problems']} problems x {gate['samples']} samples; baseline "
                     f"{pct(gate['baseline'])}, oracle at K = 8 {pct(gate['oracle'])}: rho = "
                     f"{gate['rho']:.3f} [{gate['low']:.3f}, {gate['high']:.3f}]: {gate['verdict']}")
    lines += ["", "## Did the structure reach the realizer?", "",
              f"At one token per step (`{result['delivery']['row']}`) the samples must have the planned kind on at "
              f"least {DELIVERY:.0%} of their lines.", "", f"- {result['delivery']['verdict']}"]
    if result["delivery"].get("null_kind_match") is not None:
        lines.append(f"- for comparison, the same weights without a plan: kind match "
                     f"{result['delivery']['null_kind_match']:.3f}")
    curve = result["curve"]
    if curve:
        lines += ["", "## While it trained", "",
                  "Probe: Pass@1 on MBPP validation at K = 8 with the oracle plan. Plan gain: null - oracle loss at "
                  "the all-masked state, nats per masked token. Gate: mean |tanh(a_l)| over the blocks.", "",
                  "| step | probe Pass@1 | probe kind match | plan gain (all masked) | mean gate |", "|---|---|---|---|---|"]
        for point in curve:
            lines.append(f"| {point['step']} | {pct(point['probe_pass@1'])} | {num(point['probe_kind_match'])} | "
                         f"{num(point['plan_gain_k0'])} | {num(point['gate_mean'])} |")
    notes = result.get("notes") or []
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def _write(out: Path, text: str, result: dict[str, Any]) -> None:
    """Write the report; refuse a text that lost the oracle label (title and every row)."""
    if ORACLE_LABEL not in text.split("\n", 1)[0] or text.count(ORACLE_LABEL) < 1 + len(result["rows"]):
        raise ValueError(f"an oracle report must carry the label {ORACLE_LABEL!r} in its title and on every row")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    (out / "report.json").write_text(json.dumps({**result, "label": ORACLE_LABEL}, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 1, step 1: the structure oracle's report and gate",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True, help="the oracle run's evaluation directory")
    p.add_argument("--run", type=Path, default=None, help="the oracle run (metrics for the training curve)")
    p.add_argument("--reference", type=Path, default=None,
                   help="a Stage 0 sweep directory: its mbpp/ar is the baseline, its realizer rows are 'no plan'")
    p.add_argument("--out", type=Path, default=None, help="default: the --eval directory")
    args = p.parse_args(argv)
    out = args.out or args.eval
    bench = args.eval / "mbpp"
    own = _summaries(bench)
    oracle_rows = {tag: s for tag, s in own.items() if is_oracle_row(tag, s)}
    if not oracle_rows:
        raise SystemExit(f"no oracle row under {bench}")
    unlabelled = [tag for tag, s in oracle_rows.items() if s.get("label") != ORACLE_LABEL]
    if unlabelled:
        raise ValueError(f"oracle rows without the label {ORACLE_LABEL!r}: {unlabelled}; an oracle run cannot be "
                         "reported without it")
    reference_bench = args.reference / "mbpp" if args.reference else None
    reference = _summaries(reference_bench) if reference_bench else {}
    notes: list[str] = []

    def order_key(tag: str) -> tuple[int, int, str]:
        plan, _, rest = tag.partition("-K")
        steps = int(rest.split("-")[0]) if rest.split("-")[0].isdigit() else 0
        return (PLAN_VARIANTS.index(plan) if plan in PLAN_VARIANTS else 99, steps, rest)

    rows = [{"tag": tag, "plan": s.get("plan"), "pass@1": s.get("pass@1"), "parse_rate": s.get("parse_rate"),
             "depth": s.get("mean_sequential_depth", s.get("sequential_depth")),
             "kind_match": s.get("kind_match"), "line_count_match": s.get("line_count_match"),
             "problems": s.get("problems"), "samples": s.get("samples_per_problem"), "step": s.get("step")}
            for tag, s in sorted(oracle_rows.items(), key=lambda item: order_key(item[0]))]

    comparison = []
    for k, order in K_ROWS:
        suffix = f"K{k}" + ("" if order == "random" else f"-{order}")
        cells: dict[str, Any] = {"k": k, "order": {"random": "exact kernel"}.get(order, order.replace("_", " "))}
        for plan in PLAN_VARIANTS:
            if f"{plan}-{suffix}" in oracle_rows:
                cells[plan] = oracle_rows[f"{plan}-{suffix}"].get("pass@1")
        if f"realizer-{suffix}" in reference:
            cells["no_plan"] = reference[f"realizer-{suffix}"].get("pass@1")
        if "oracle" in cells and "null" in cells:
            pair = _paired(bench, f"oracle-{suffix}", bench, f"null-{suffix}")
            if pair is not None:
                cells["oracle_minus_null"] = bootstrap_difference(*pair)
        if len(cells) > 2:
            comparison.append(cells)

    gate: dict[str, Any] = {"rho": None}
    baseline_dir = reference_bench if BASELINE in reference else (bench if BASELINE in own else None)
    if GATE_ROW not in oracle_rows:
        gate["reason"] = f"`{GATE_ROW}` has not been evaluated"
    elif baseline_dir is None:
        gate["reason"] = f"no baseline `{BASELINE}` under --reference or --eval"
    else:
        pair = _paired(bench, GATE_ROW, baseline_dir, BASELINE)
        if pair is None:
            gate["reason"] = (f"the baseline in {baseline_dir} was scored on other problems than `{GATE_ROW}` "
                              "(evaluate the baseline on this run's problems)")
        else:
            rho, low, high = bootstrap_ratio(*pair)
            gate = {"rho": rho, "low": low, "high": high, "band": step1_gate(rho, low, high),
                    "problems": oracle_rows[GATE_ROW].get("problems"),
                    "samples": oracle_rows[GATE_ROW].get("samples_per_problem"),
                    "oracle": float(pair[0].mean()), "baseline": float(pair[1].mean()),
                    "baseline_dir": str(baseline_dir / BASELINE)}
    one_per_step = delivery_row(oracle_rows)
    delivered = oracle_rows.get(one_per_step, {}).get("kind_match")
    if gate.get("rho") is not None:
        gate["verdict"] = gate_reading(gate["rho"], gate["low"], gate["high"], delivered)
    delivery = {"row": one_per_step, "kind_match": delivered, "threshold": DELIVERY,
                "verdict": delivery_verdict(delivered),
                "null_kind_match": oracle_rows.get(delivery_row(oracle_rows, "null"), {}).get("kind_match")}
    for tag, s in oracle_rows.items():
        if s.get("references_not_parsed") or s.get("references_truncated"):
            notes.append(f"`{tag}`: {s.get('references_not_parsed', 0)} references did not parse (their plan was "
                         f"read from the raw text) and {s.get('references_truncated', 0)} did not fit the canvas "
                         "(their plan was cut)")
            break
    steps = sorted({s.get("step") for s in oracle_rows.values()})
    if len(steps) > 1:
        notes.append(f"the rows come from different checkpoint steps: {steps}")
    result = {"rows": rows, "comparison": comparison, "gate": gate, "delivery": delivery, "curve": _curve(args.run),
              "notes": notes, "thresholds": {"build": BUILD, "closed": CLOSED, "delivery": DELIVERY}}
    text = render(result)
    _write(out, text, result)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
