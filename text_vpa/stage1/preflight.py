# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
preflight.py -- what the planner stage assumes about tokens, lines and skeletons, measured before any training.
No weights are loaded. Exits 1 on anything that would make the structure oracle meaningless.

    python -m text_vpa.stage1.preflight --model data2/models/gemma-4-E2B --corpus data2/corpus \\
        --eval-dir data2/eval --init runs/stage0-a/last.pt --out runs/stage1-oracle/preflight.json

1. The checkpoint the run starts from is a Stage 0 checkpoint with a data position.
2. Token offsets. Line slots rest on the character offset of every token. The tokenizer's own offsets are
   compared with an independent computation (the length of the decoded prefix before each token) on corpus
   canvases and on MBPP's normalised references: the two must give the same slots. FAILS under 99.5%.
3. Lines. How many programs fit M line slots (corpus canvases of both formats, MBPP references), and how their
   lines are indented: depth is indentation // 4, so the share of programs indented in multiples of four says
   how often a slot's depth is the line's nesting level.
4. MBPP's validation references, as the oracle reads them: how many normalise (``ast.unparse``), fit the canvas,
   and still pass their tests. FAILS if a reference passes as written and fails once normalised.
5. Line skeletons (the modes of step 3; nothing of step 1 depends on them). A codebook of the most frequent
   skeletons of a corpus sample, and the share of held-out corpus functions and of MBPP validation references
   whose skeleton it holds, with and without depth. The design drops depth from the skeleton if coverage with it
   is under 60%; that is decided on the held-out corpus, never on a benchmark.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from text_vpa.stage0.data import HFTokenizer, complete_example, write_example
from text_vpa.stage0.evaluate import load_tasks, score
from text_vpa.stage1.oracle import normalize_reference
from text_vpa.stage1.slots import (
    Codebook,
    encode_with_starts,
    indentation_columns,
    line_slots,
    program_lines,
    skeleton,
    split_lines,
    starts_by_decoding,
)

__all__ = ["canvases", "indented_in_fours", "offsets_agree"]

DEPTH_RULE = 0.60                    # docs/stage1.md: under this coverage, the skeleton drops depth


def _rows(paths: list[str], limit: int) -> Iterator[dict]:
    """Up to ``limit`` rows, the same number from the head of every file."""
    if not paths or limit <= 0:
        return
    quota = max(1, -(-limit // len(paths)))
    taken = 0
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= quota or taken >= limit:
                    break
                taken += 1
                yield json.loads(line)


def canvases(rows: Iterator[dict], write_fraction: float = 0.5, seed: int = 0) -> Iterator[tuple[str, str]]:
    """(format, canvas text) as the corpus stream builds them (``stage0.data.CorpusStream.build``)."""
    rng = random.Random(seed)
    for row in rows:
        made = write_example(row) if rng.random() < write_fraction else None
        kind = "write"
        if made is None:
            made, kind = complete_example(row), "complete"
        if made is not None:
            yield kind, made[1]


def indented_in_fours(text: str) -> bool:
    """True if no line is indented with a tab and every non-blank line starts at a multiple of four columns."""
    for _, line in split_lines(text):
        if not line.strip():
            continue
        lead = line[: len(line) - len(line.lstrip(" \t"))]
        if "\t" in lead or indentation_columns(line) % 4:
            return False
    return True


def offsets_agree(tokenizer: Any, texts: list[str], max_slots: int) -> tuple[float, float]:
    """(share of programs whose line slots are the same from the tokenizer's offsets and from decoded prefixes,
    share whose tokens decode back to the text)."""
    if not texts:
        return 1.0, 1.0
    ids, starts = encode_with_starts(tokenizer, texts)
    same = round_trip = 0
    for text, row, offsets in zip(texts, ids, starts, strict=True):
        round_trip += tokenizer.decode(row) == text
        try:
            same += line_slots(text, offsets, max_slots) == line_slots(text, starts_by_decoding(tokenizer, row),
                                                                       max_slots)
        except ValueError:                                           # decoded prefixes longer than the text
            pass
    return same / len(texts), round_trip / len(texts)


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    a = np.array(values)
    return {"mean": round(float(a.mean()), 1), "p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)),
            "p99": int(np.percentile(a, 99)), "max": int(a.max())}


def _share(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 1 preflight", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="local directory (or Hub id) of the backbone: its tokenizer")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--init", type=Path, default=None, help="the Stage 0 checkpoint the oracle run starts from")
    p.add_argument("--canvas-len", type=int, default=256)
    p.add_argument("--max-slots", type=int, default=32)
    p.add_argument("--corpus-sample", type=int, default=2000, help="corpus rows for the line and token statistics")
    p.add_argument("--offset-sample", type=int, default=300, help="programs for the token-offset check")
    p.add_argument("--codebook-sample", type=int, default=200_000, help="corpus rows the codebook is built from")
    p.add_argument("--codebook-size", type=int, default=1024)
    p.add_argument("--skeleton-lines", type=int, default=16)
    p.add_argument("--exec-workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=None, help="write the findings as JSON")
    args = p.parse_args(argv)
    from transformers import AutoTokenizer

    problems: list[str] = []
    report: dict[str, Any] = {"model": args.model, "canvas_len": args.canvas_len, "max_slots": args.max_slots}

    # ---- 1. the checkpoint to start from ----------------------------------------------------------
    if args.init is not None:
        from text_vpa.engine.runtime import load_checkpoint

        if not args.init.exists():
            problems.append(f"{args.init} does not exist: the oracle run starts from a Stage 0 checkpoint")
        else:
            init = load_checkpoint(args.init)
            config = init.get("config", {})
            report["init"] = {"path": str(args.init), "stage": init.get("stage"), "step": init.get("step"),
                              "canvas_len": config.get("canvas_len"), "prefix_fraction": config.get("prefix_fraction"),
                              "stream": init.get("stream")}
            print("init:", json.dumps(report["init"]))
            if init.get("stage") != "stage0" or "stream" not in init or "trainable" not in init:
                problems.append(f"{args.init} is not a Stage 0 checkpoint with a data position")
            elif config.get("canvas_len") != args.canvas_len:
                problems.append(f"the checkpoint's canvas has {config.get('canvas_len')} tokens, not {args.canvas_len}")

    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(args.model))
    shards = sorted(glob.glob(str(args.corpus / "train-*.jsonl.gz")))
    if not shards or not (args.corpus / "val.jsonl.gz").exists():
        problems.append(f"{args.corpus} has no train-*.jsonl.gz shards or no val.jsonl.gz")
    sample = list(canvases(_rows(shards, args.corpus_sample)))
    mbpp = load_tasks("mbpp", args.eval_dir, "validation")
    normalized = [normalize_reference(task.reference) for task in mbpp]
    references = [text for text, _ in normalized]

    # ---- 2. token offsets -------------------------------------------------------------------------
    step = max(1, len(sample) // max(args.offset_sample, 1))
    checked = [text for _, text in sample[::step]][: args.offset_sample]
    corpus_same, corpus_round = offsets_agree(tokenizer, checked, args.max_slots)
    mbpp_same, mbpp_round = offsets_agree(tokenizer, references, args.max_slots)
    report["offsets"] = {"corpus_programs": len(checked), "corpus_same_slots": round(corpus_same, 4),
                         "corpus_round_trip": round(corpus_round, 4), "mbpp_programs": len(references),
                         "mbpp_same_slots": round(mbpp_same, 4), "mbpp_round_trip": round(mbpp_round, 4)}
    print("token offsets:", json.dumps(report["offsets"]))
    advice = "; do not train: the slots would be wrong (the remedy is to take offsets from decoded prefixes)"
    if checked and corpus_same < 0.995:
        problems.append(f"the tokenizer's offsets and decoded prefixes give the same line slots for only "
                        f"{corpus_same:.1%} of the corpus canvases" + advice)
    if references and mbpp_same < 0.995:
        problems.append(f"the tokenizer's offsets and decoded prefixes give the same line slots for only "
                        f"{mbpp_same:.1%} of MBPP's normalised references" + advice)

    # ---- 3. lines ---------------------------------------------------------------------------------
    report["lines"] = {}
    for kind in ("write", "complete"):
        texts = [text for k, text in sample if k == kind]
        ids, starts = encode_with_starts(tokenizer, texts)
        fitting = [(text, s) for text, row, s in zip(texts, ids, starts, strict=True) if len(row) + 1 <= args.canvas_len]
        slots = [line_slots(text, s, args.max_slots) for text, s in fitting]
        per_line = [length for row in slots for length in row.length]
        report["lines"][f"corpus_{kind}"] = {
            "programs": len(texts), "fit_canvas": _share([len(row) + 1 <= args.canvas_len for row in ids]),
            "within_slots": _share([row.lines <= args.max_slots for row in slots]),
            "lines": _quantiles([row.lines for row in slots]), "tokens_per_line": _quantiles(per_line),
            "lines_without_a_token": _share([length == 0 for length in per_line]),
            "indented_in_fours": _share([indented_in_fours(text) for text, _ in fitting])}
        print(f"lines, corpus {kind}:", json.dumps(report["lines"][f"corpus_{kind}"]))
    if not sample:
        problems.append("the corpus sample yields no example")

    # ---- 4. MBPP's validation references as the oracle reads them --------------------------------
    ids, starts = encode_with_starts(tokenizer, references)
    slots = [line_slots(text, s, args.max_slots, max_tokens=args.canvas_len - 1)
             for text, s in zip(references, starts, strict=True)]
    raw_summary, raw_rows = score(mbpp, "mbpp", [[task.reference] for task in mbpp], [1], workers=args.exec_workers)
    norm_summary, norm_rows = score(mbpp, "mbpp", [[text] for text in references], [1], workers=args.exec_workers)
    broken = [task.task_id for task, a, b in zip(mbpp, raw_rows, norm_rows, strict=True)
              if a["passed"] and not b["passed"]]
    report["mbpp_validation"] = {
        "problems": len(mbpp), "normalised": sum(ok for _, ok in normalized),
        "indented_in_fours_as_written": _share([indented_in_fours(task.reference) for task in mbpp]),
        "indented_in_fours_normalised": _share([indented_in_fours(text) for text in references]),
        "fit_canvas": _share([len(row) + 1 <= args.canvas_len for row in ids]),
        "within_slots": _share([row.lines <= args.max_slots for row in slots]),
        "lines": _quantiles([row.lines for row in slots]),
        "tokens": _quantiles([len(row) for row in ids]),
        "pass_as_written": raw_summary["solved_problems"], "pass_normalised": norm_summary["solved_problems"],
        "broken_by_normalising": broken}
    print("MBPP validation references:", json.dumps(report["mbpp_validation"]))
    if broken:
        problems.append(f"{len(broken)} MBPP references pass as written and fail once normalised: {broken[:5]}")
    if mbpp and report["mbpp_validation"]["normalised"] < 0.95 * len(mbpp):
        problems.append("fewer than 95% of MBPP's validation references parse")

    # ---- 5. line skeletons (step 3) ---------------------------------------------------------------
    held_out = list(canvases(_rows([str(args.corpus / "val.jsonl.gz")], 10**9))) if shards else []
    report["skeletons"] = {"sample_rows": args.codebook_sample, "size": args.codebook_size,
                           "skeleton_lines": args.skeleton_lines}
    both: dict[bool, Counter] = {True: Counter(), False: Counter()}
    built = 0
    for _, text in canvases(_rows(shards, args.codebook_sample)):       # one pass over the sample
        lines = program_lines(text)
        for with_depth, counter in both.items():
            counter[skeleton(lines, args.skeleton_lines, with_depth)] += 1
        built += 1
    for with_depth, counts in both.items():
        book = Codebook.build(counts, args.codebook_size, with_depth, args.skeleton_lines)
        covered = sum(counts[entry] for entry in book.entries)
        entry = {
            "programs": built, "distinct": len(counts), "in_sample": round(covered / max(built, 1), 4),
            "held_out": round(book.coverage(program_lines(text) for _, text in held_out), 4),
            "held_out_write": round(book.coverage(program_lines(t) for k, t in held_out if k == "write"), 4),
            "held_out_complete": round(book.coverage(program_lines(t) for k, t in held_out if k == "complete"), 4),
            "mbpp_validation": round(book.coverage(program_lines(text) for text in references), 4)}
        report["skeletons"]["with_depth" if with_depth else "without_depth"] = entry
        print(f"skeletons {'with' if with_depth else 'without'} depth:", json.dumps(entry))
        if with_depth and args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            book.save(args.out.with_name("skeleton_codebook.json"))
    report["skeletons"]["held_out_programs"] = len(held_out)
    with_depth_coverage = report["skeletons"]["with_depth"]["held_out"]
    report["skeletons"]["rule"] = (
        f"held-out coverage with depth is {with_depth_coverage:.1%}: "
        + ("depth stays in the skeleton" if with_depth_coverage >= DEPTH_RULE else
           f"under {DEPTH_RULE:.0%}, so the design drops depth from the skeleton (step 3)"))
    print("skeleton rule:", report["skeletons"]["rule"])

    report["problems"] = problems
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    for line in problems:
        print("PROBLEM  " + line, file=sys.stderr)
    print("stage 1 preflight: " + ("FAILED" if problems else "ok"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
