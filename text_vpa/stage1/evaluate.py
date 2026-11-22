# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
evaluate.py -- Pass@k of an oracle run (structure or latent) in Stage 0's harness. A diagnostic: every row is an
oracle row.

    python -m text_vpa.stage1.evaluate --checkpoint runs/stage1-oracle/last.pt --model data2/models/gemma-4-E2B \\
        --eval-dir data2/eval --out runs/stage1-oracle/sweep --plans oracle --k-steps 8 32 128 --orders random

Prompts, canvas length, token draw, truncation, tests and seeds are Stage 0's (``stage0.evaluate``), so a row
differs from the Stage 0 row of the same K and order by the trained weights and the plan only. The plan
(``--plans``):

    oracle     the plan of the problem's normalised reference (``oracle.py``: its line structure, or its lines'
               frozen backbone states read after the problem's prompt)
    null       the bridge's null token: the same weights without a plan
    shuffled   the plan of the next problem's reference

MBPP only (the oracle reads a whole reference program), and by default its validation split: nothing of this
stage is selected on a test set.

Results go to ``OUT/mbpp/<tag>/{samples.jsonl, summary.json}`` with tags ``oracle-K8``, ``oracle-K8-confidence``,
``oracle-K256-left_to_right``, ``null-K8``, ``shuffled-K8``. Each summary carries the oracle label, and each
sample how far it followed the plan it was given (``kind_match``, ``lines``, ``plan_lines``). A configuration
whose summary exists for the same checkpoint step and problems is skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from text_vpa.engine.runtime import load_checkpoint
from text_vpa.evaluation.tasks import MBPP_SPLITS
from text_vpa.stage0.backbone import load_gemma4
from text_vpa.stage0.data import HFTokenizer
from text_vpa.stage0.evaluate import load_tasks, score, write_results
from text_vpa.stage0.sampler import ORDERS, TIME_INPUTS, SamplerConfig
from text_vpa.stage1.oracle import PLAN_VARIANTS, oracle_samples
from text_vpa.stage1.plan import ORACLE_LABEL
from text_vpa.stage1.train import load_oracle_run
from text_vpa.utils.device import resolve_device

__all__ = ["config_tag"]


def config_tag(plan: str, k_steps: int, order: str, time_input: str = "schedule") -> str:
    return (f"{plan}-K{k_steps}" + ("" if order == "random" else f"-{order}")
            + ("-matched_time" if time_input == "matched" else ""))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1, step 1: Pass@k of the structure oracle (a diagnostic)",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="a stage1-oracle checkpoint")
    p.add_argument("--model", default=None, help="Hub id or local directory of the backbone (default: the run's)")
    p.add_argument("--revision", default=None)
    p.add_argument("--eval-dir", type=Path, required=True, help="directory with mbpp.jsonl")
    p.add_argument("--mbpp-split", choices=sorted(MBPP_SPLITS), default="validation")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plans", nargs="+", choices=PLAN_VARIANTS, default=["oracle"])
    p.add_argument("--k-steps", type=int, nargs="+", default=[8, 32, 128], help="realizer steps K")
    p.add_argument("--orders", nargs="+", choices=ORDERS, default=["random"],
                   help="random: the exact jump kernel (the gate); confidence; left_to_right")
    p.add_argument("--time-input", choices=TIME_INPUTS, default="schedule")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--k", type=int, nargs="+", default=[1])
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--exec-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint)
    model_id = args.model or payload["config"]["model"]
    model, hf_tokenizer = load_gemma4(model_id, revision=args.revision, token=os.environ.get("HF_TOKEN") or None)
    tokenizer = HFTokenizer(hf_tokenizer)
    try:
        realizer, bridge, oracle, cfg = load_oracle_run(payload, model, tokenizer)
    except ValueError as exc:
        raise SystemExit(f"{args.checkpoint}: {exc}") from exc
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)                    # the realizer is text-only
    realizer.to(device).eval()
    bridge.to(device).eval()
    oracle.to(device).eval()
    tasks = load_tasks("mbpp", args.eval_dir, args.mbpp_split, args.limit)
    if not tasks:
        raise SystemExit(f"no MBPP problem under {args.eval_dir}")
    common = {"model": model_id, "revision": args.revision, "seed": args.seed, "device": str(device),
              "checkpoint": str(args.checkpoint), "step": payload["step"], "schedule": cfg.schedule,
              "split": args.mbpp_split, "init": payload.get("init")}

    for plan in args.plans:
        for order in args.orders:
            for k_steps in args.k_steps:
                tag = config_tag(plan, k_steps, order, args.time_input)
                out = args.out / "mbpp" / tag
                if (out / "summary.json").exists() and not args.force:
                    previous = json.loads((out / "summary.json").read_text())
                    if previous.get("step") == common["step"] and previous.get("problems") == len(tasks):
                        print(f"{out}: done, skipping (--force to redo)", file=sys.stderr)
                        continue
                    print(f"{out}: from another checkpoint step or problem set, evaluating again", file=sys.stderr)
                print(f"== mbpp / {tag} ({ORACLE_LABEL}): {len(tasks)} problems x {args.num_samples} samples",
                      file=sys.stderr, flush=True)
                sampler = SamplerConfig(k_steps, cfg.schedule, order, args.temperature, args.top_p, args.top_k,
                                        time_input=args.time_input, time_grid=max(cfg.step_budgets))
                texts, info, followed = oracle_samples(
                    realizer, bridge, oracle, tokenizer, tasks, "mbpp", sampler, cfg.canvas_len, args.num_samples,
                    args.batch_size, args.seed, device, variant=plan, max_slots=cfg.max_slots, progress=True,
                    latent_layers=cfg.latent_layers)
                summary, rows = score(tasks, "mbpp", texts, args.k, False, args.timeout, args.exec_workers)
                flat = [row for group in followed for row in group]
                rows = [{**row, **extra} for row, extra in zip(rows, flat, strict=True)]
                summary = {**summary, "tag": tag, **info, **common}
                write_results(out, summary, rows)
                print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
