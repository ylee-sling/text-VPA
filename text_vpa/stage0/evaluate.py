# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
evaluate.py -- Pass@k of the Stage-0 realizer or of the autoregressive baseline, in the repository's harness.

    python -m text_vpa.stage0.evaluate --decoder ar --model google/gemma-4-E2B \\
        --eval-dir data2/eval --out runs/stage0/eval
    python -m text_vpa.stage0.evaluate --decoder realizer --model google/gemma-4-E2B \\
        --checkpoint runs/stage0/last.pt --k-steps 2 4 8 --eval-dir data2/eval --out runs/stage0/eval

Both decoders get the same prompts (``data``: HumanEval ``complete``, MBPP ``write``), the same token budget (the
canvas length N), the same token draw (temperature, top-k, top-p), the same truncation (``postprocess``) and the
same tests (``text_vpa.evaluation``: each program in its own resource-limited process -- isolation, not a
security sandbox). pass@k is the unbiased estimator over ``--num-samples`` samples per problem.

Results go to ``OUT/<benchmark>/<tag>/{samples.jsonl, summary.json}`` with tag ``ar``, ``realizer-K8`` or
``realizer-K8-confidence`` (``--tag-suffix`` appends to it). A configuration whose summary exists for the same
checkpoint step and problems is skipped unless ``--force``. ``--chat`` evaluates an instruction-tuned checkpoint through its chat template (a reference
row; it is not the H2 baseline).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from text_vpa.evaluation import load_humaneval_tasks, load_mbpp_tasks, run_programs, summarize_pass_at_k
from text_vpa.evaluation.tasks import MBPP_SPLITS, EvalTask
from text_vpa.stage0.autoregressive import cut_at_eos, generate_autoregressive
from text_vpa.stage0.backbone import RealizerSpec, Stage0Realizer, left_pad, load_gemma4
from text_vpa.stage0.data import HFTokenizer, Stage0Tokenizer, mbpp_prompt
from text_vpa.stage0.postprocess import build_sample, is_complete, parses
from text_vpa.stage0.sampler import ORDERS, TIME_INPUTS, SamplerConfig, realize
from text_vpa.utils.device import resolve_device

__all__ = ["load_tasks", "prompt_text", "realizer_samples", "score", "write_results"]

CHAT_MAX_NEW_TOKENS = 768          # a chat reply repeats the signature and docstring inside a code fence
CHAT_INSTRUCTIONS = {
    "humaneval": "Complete the following Python function. Reply with the whole function in one ```python code "
                 "block and nothing else.\n\n",
    "mbpp": "Write a Python function for the following task. Reply with the code in one ```python code block and "
            "nothing else.\n\n",
}


def load_tasks(benchmark: str, eval_dir: str | Path, mbpp_split: str = "test", limit: int | None = None
               ) -> list[EvalTask]:
    eval_dir = Path(eval_dir)
    tasks = (load_humaneval_tasks(eval_dir / "HumanEval.jsonl.gz") if benchmark == "humaneval"
             else load_mbpp_tasks(eval_dir / "mbpp.jsonl", mbpp_split))
    return tasks[:limit] if limit else tasks


def prompt_text(task: EvalTask, benchmark: str) -> str:
    return task.context if benchmark == "humaneval" else mbpp_prompt(task.context)


def autocast(device: torch.device) -> Any:
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def realizer_samples(realizer: Stage0Realizer, tokenizer: Stage0Tokenizer, tasks: Sequence[EvalTask],
                     benchmark: str, cfg: SamplerConfig, canvas_len: int, num_samples: int, batch_size: int,
                     seed: int, device: torch.device, progress: bool = False, mode: str = "bidirectional",
                     adapters: bool = True, plan_for: Callable[[int, EvalTask], Any] | None = None
                     ) -> tuple[list[list[str]], dict]:
    """``num_samples`` canvases per task, decoded up to their first <eos>.

    ``mode`` and ``adapters``: the canvas masks and whether LoRA and modulation act (``causal`` without adapters
    is the backbone's own decoding through this code path). A left-to-right row stops at its <eos>, or once the
    truncation rule has found its stop in the revealed prefix; neither changes the program.

    ``plan_for`` (the planner stage): (task index, task) -> a one-row ``PlanCache``, built once per task and
    shared by all of its samples; prompts, seeds and the token draw are the same with and without it.
    """
    texts: list[list[str]] = []
    forbidden = [tokenizer.pad_id, tokenizer.bos_id]
    stop_id = tokenizer.eos_id if cfg.order == "left_to_right" else None
    kind = "complete" if benchmark == "humaneval" else "write"

    def row_done(tokens: torch.Tensor) -> torch.Tensor:
        done = []
        for row in tokens.tolist():
            prefix = cut_at_eos(row, [tokenizer.eos_id, realizer.spec.mask_id])
            done.append(is_complete(tokenizer.decode(prefix), kind))
        return torch.tensor(done, dtype=torch.bool)

    seconds, tokens_out, depth, row_passes = 0.0, 0, 0, 0
    revealed = [0] * cfg.num_steps
    for index, task in enumerate(tasks):
        ids = [tokenizer.bos_id, *tokenizer.encode_batch([prompt_text(task, benchmark)])[0]]
        generator = torch.Generator(device=device).manual_seed(seed * 100_003 + index)
        start = time.perf_counter()
        with autocast(device):
            cache = realizer.prefill(*left_pad([ids[:-1]], tokenizer.pad_id, device))
            plan = plan_for(index, task) if plan_for is not None else None
            rows: list[str] = []
            while len(rows) < num_samples:
                n = min(batch_size, num_samples - len(rows))
                anchor = torch.full((n,), ids[-1], dtype=torch.long, device=device)
                result = realize(realizer, cache.expand(n), anchor, canvas_len, cfg, forbidden, generator,
                                 mode=mode, stop_id=stop_id, adapters=adapters,
                                 row_done=row_done if stop_id is not None else None,
                                 **({} if plan is None else {"plan": plan.expand(n)}))
                depth = max(depth, result.sequential_depth)
                row_passes += int(result.row_passes.sum())
                for step, count in enumerate(result.revealed):
                    revealed[step] += count
                for row in result.tokens.tolist():
                    kept = cut_at_eos(row, [tokenizer.eos_id])
                    tokens_out += len(kept)
                    rows.append(tokenizer.decode(kept))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds += time.perf_counter() - start
        texts.append(rows)
        if progress and (index + 1) % 20 == 0:
            print(f"  generated {index + 1}/{len(tasks)} problems", file=sys.stderr, flush=True)
    total = max(len(tasks) * num_samples, 1)
    info = {"decoder": "realizer", **cfg.to_dict(), "canvas_mode": mode, "adapters": adapters,
            "canvas_len": canvas_len, "sequential_depth": depth,
            "mean_sequential_depth": round(1 + row_passes / total, 2),
            "generation_seconds": round(seconds, 3), "mean_tokens": round(tokens_out / total, 2),
            "mean_revealed_per_step": [round(r / total, 2) for r in revealed]}
    return texts, info


def autoregressive_samples(model: Any, hf_tokenizer: Any, tasks: Sequence[EvalTask], benchmark: str,
                           args: argparse.Namespace, max_new_tokens: int) -> tuple[list[list[str]], dict]:
    kind = "complete" if benchmark == "humaneval" else "write"
    texts: list[list[str]] = []
    seconds, tokens_out = 0.0, 0
    for index, task in enumerate(tasks):
        if args.chat:
            messages = [{"role": "user", "content": CHAT_INSTRUCTIONS[benchmark] + task.context}]
            rendered = hf_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            ids = hf_tokenizer(rendered, add_special_tokens=False)["input_ids"]
            done = None
        else:
            ids = [hf_tokenizer.bos_token_id,
                   *hf_tokenizer(prompt_text(task, benchmark), add_special_tokens=False)["input_ids"]]

            def done(text: str, kind: str = kind) -> bool:
                return is_complete(text, kind)
        rows: list[str] = []
        start = time.perf_counter()
        while len(rows) < args.num_samples:
            n = min(args.batch_size, args.num_samples - len(rows))
            out, lengths = generate_autoregressive(
                model, hf_tokenizer, ids, n, max_new_tokens, args.temperature, args.top_p, args.top_k,
                seed=args.seed * 100_003 + index * 101 + len(rows), done=done)
            rows += out
            tokens_out += sum(lengths)
        seconds += time.perf_counter() - start
        texts.append(rows)
        if (index + 1) % 20 == 0:
            print(f"  generated {index + 1}/{len(tasks)} problems", file=sys.stderr, flush=True)
    total = max(len(tasks) * args.num_samples, 1)
    info = {"decoder": "ar-chat" if args.chat else "ar", "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "max_new_tokens": max_new_tokens, "generation_seconds": round(seconds, 3),
            "mean_tokens": round(tokens_out / total, 2),
            "mean_sequential_depth": round(1 + tokens_out / total, 2)}
    return texts, info


def score(tasks: Sequence[EvalTask], benchmark: str, texts: Sequence[Sequence[str]], ks: Sequence[int],
          chat: bool = False, timeout: float = 10.0, workers: int = 4) -> tuple[dict, list[dict]]:
    """Run every sample against its task's tests. Returns the summary and one row per sample."""
    samples = [[build_sample(task, benchmark, text, chat) for text in group]
               for task, group in zip(tasks, texts, strict=True)]
    programs = [task.program(s) for task, group in zip(tasks, samples, strict=True) for s in group]
    start = time.perf_counter()
    results = run_programs(programs, timeout, workers)
    rows, counts, parsed, index = [], [], 0, 0
    for task, group, raw in zip(tasks, samples, texts, strict=True):
        passed = 0
        for j, sample in enumerate(group):
            r = results[index]
            ok = parses(programs[index])
            index += 1
            passed += r.passed
            parsed += ok
            rows.append({"task_id": task.task_id, "sample": j, "passed": r.passed, "status": r.status,
                         "parses": ok, "detail": r.detail, "completion": sample, "raw": raw[j]})
        counts.append((len(group), passed))
    n = min(c[0] for c in counts)
    summary = {"benchmark": benchmark, "problems": len(tasks), "samples_per_problem": n,
               **summarize_pass_at_k(counts, [k for k in ks if k <= n] or [1]),
               "solved_problems": sum(1 for _, c in counts if c > 0),
               "parse_rate": round(parsed / max(len(programs), 1), 4),
               "execution_seconds": round(time.perf_counter() - start, 3)}
    return summary, rows


def write_results(out: Path, summary: dict, rows: Sequence[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    os.replace(tmp, out / "summary.json")                         # the summary marks the configuration as done


def reference_text(task: EvalTask, benchmark: str) -> str:
    """The canonical solution as the text a decoder would have to produce after the prompt."""
    return task.reference[len(task.context):] if benchmark == "humaneval" else task.reference


def reference_check(args: argparse.Namespace) -> int:
    """Canonical solutions through ``postprocess``: HumanEval 164/164, MBPP test 499/500 (task 367's tests use
    an undefined class), as with ``evaluate.py --reference``."""
    for benchmark in args.benchmarks:
        tasks = load_tasks(benchmark, args.eval_dir, args.mbpp_split, args.limit)
        summary, rows = score(tasks, benchmark, [[reference_text(t, benchmark)] for t in tasks], [1],
                              timeout=args.timeout, workers=args.exec_workers)
        summary = {**summary, "tag": "reference", "split": args.mbpp_split if benchmark == "mbpp" else "test"}
        write_results(args.out / benchmark / "reference", summary, rows)
        print(json.dumps(summary, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 0: Pass@k of the realizer or the autoregressive baseline",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decoder", choices=("ar", "realizer", "reference"), required=True,
                   help="reference: the canonical solutions through the same truncation (a harness check)")
    p.add_argument("--model", default="google/gemma-4-E2B", help="Hub id or local directory of the backbone")
    p.add_argument("--revision", default=None)
    p.add_argument("--checkpoint", type=Path, help="Stage-0 checkpoint (realizer)")
    p.add_argument("--benchmarks", nargs="+", choices=("humaneval", "mbpp"), default=["humaneval", "mbpp"])
    p.add_argument("--eval-dir", type=Path, required=True, help="directory with HumanEval.jsonl.gz and mbpp.jsonl")
    p.add_argument("--mbpp-split", choices=sorted(MBPP_SPLITS), default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k-steps", type=int, nargs="+", default=[2, 4, 8], help="realizer steps K")
    p.add_argument("--orders", nargs="+", choices=ORDERS, default=["random"],
                   help="random: the exact jump kernel (H2); confidence; left_to_right (diagnostic)")
    p.add_argument("--schedule", default=None, help="kappa schedule (default: the checkpoint's)")
    p.add_argument("--canvas-len", type=int, default=None, help="N (default: the checkpoint's; ar: 256)")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--k", type=int, nargs="+", default=[1, 10])
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--chat", action="store_true", help="ar only: prompt through the chat template")
    p.add_argument("--canvas-mode", choices=("bidirectional", "causal"), default="bidirectional",
                   help="realizer only: the canvas's attention masks")
    p.add_argument("--time-input", choices=TIME_INPUTS, default="schedule",
                   help="realizer only: the (t, h) given to the network; matched: a pair seen in training")
    p.add_argument("--no-adapters", action="store_true",
                   help="realizer only: LoRA and modulation off; with --canvas-mode causal and left_to_right at "
                        "K = N this is the backbone's own decoding through the realizer's code path")
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--exec-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.decoder == "realizer" and args.checkpoint is None:
        raise SystemExit("--decoder realizer needs --checkpoint")
    if args.chat and args.decoder != "ar":
        raise SystemExit("--chat applies to --decoder ar")
    if args.decoder == "reference":
        return reference_check(args)
    device = resolve_device(args.device)
    token = os.environ.get("HF_TOKEN") or None
    model, hf_tokenizer = load_gemma4(args.model, revision=args.revision, token=token)
    common = {"model": args.model, "revision": args.revision, "seed": args.seed, "device": str(device)}

    configs: list[tuple[str, SamplerConfig | None]] = []
    canvas_len = args.canvas_len or 256
    if args.decoder == "ar":
        model.to(device)
        configs.append(("ar-chat" if args.chat else "ar", None))
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if ckpt.get("stage", "stage0") != "stage0":
            # A planner-stage checkpoint also holds a bridge and, for the oracles, a label its results must carry.
            raise SystemExit(f"{args.checkpoint} is a {ckpt['stage']!r} checkpoint: evaluate it with "
                             "text_vpa.stage1.evaluate")
        tokenizer = HFTokenizer(hf_tokenizer)
        spec = RealizerSpec.from_dict(ckpt["spec"])
        if spec.mask_id != tokenizer.mask_id:
            raise SystemExit("the checkpoint's [MASK] id is not this tokenizer's")
        realizer = Stage0Realizer.from_pretrained_model(model, spec)
        realizer.load_trainable_state(ckpt["trainable"])
        for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            if hasattr(model.model, tower):
                setattr(model.model, tower, None)               # the realizer is text-only
        realizer.to(device).eval()
        canvas_len = args.canvas_len or ckpt["config"]["canvas_len"]
        schedule = args.schedule or ckpt["config"]["schedule"]
        common.update(checkpoint=str(args.checkpoint), step=ckpt["step"], schedule=schedule)
        for order in args.orders:
            for k_steps in args.k_steps:
                tag = (f"realizer-K{k_steps}" + ("" if order == "random" else f"-{order}")
                       + ("-causal" if args.canvas_mode == "causal" else "")
                       + ("-no_adapters" if args.no_adapters else "")
                       + ("-matched_time" if args.time_input == "matched" else ""))
                configs.append((tag, SamplerConfig(k_steps, schedule, order, args.temperature, args.top_p,
                                                   args.top_k, time_input=args.time_input,
                                                   time_grid=max(ckpt["config"]["step_budgets"]))))

    for benchmark in args.benchmarks:
        tasks = load_tasks(benchmark, args.eval_dir, args.mbpp_split, args.limit)
        if not tasks:
            raise SystemExit(f"no {benchmark} problem under {args.eval_dir}")
        for tag, cfg in configs:
            out = args.out / benchmark / (tag + args.tag_suffix)
            if (out / "summary.json").exists() and not args.force:
                previous = json.loads((out / "summary.json").read_text())
                if previous.get("step") == common.get("step") and previous.get("problems") == len(tasks):
                    print(f"{out}: done, skipping (--force to redo)", file=sys.stderr)
                    continue
                print(f"{out}: from another checkpoint step or problem set, evaluating again", file=sys.stderr)
            print(f"== {benchmark} / {tag}{args.tag_suffix}: {len(tasks)} problems x {args.num_samples} samples",
                  file=sys.stderr, flush=True)
            if cfg is None:
                texts, info = autoregressive_samples(model, hf_tokenizer, tasks, benchmark, args,
                                                     CHAT_MAX_NEW_TOKENS if args.chat else canvas_len)
            else:
                texts, info = realizer_samples(realizer, tokenizer, tasks, benchmark, cfg, canvas_len,
                                               args.num_samples, args.batch_size, args.seed, device, progress=True,
                                               mode=args.canvas_mode, adapters=not args.no_adapters)
            summary, rows = score(tasks, benchmark, texts, args.k, args.chat, args.timeout, args.exec_workers)
            summary = {**summary, "split": args.mbpp_split if benchmark == "mbpp" else "test", "tag": tag,
                       **info, **common}
            write_results(out, summary, rows)
            print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
