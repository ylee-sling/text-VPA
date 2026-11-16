# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
preflight.py -- check, without loading any weights, that the downloaded checkpoint, the tokenizer and the data
are what Stage 0 assumes. Exits 1 on anything that would make the run meaningless.

    python -m text_vpa.stage0.preflight --model data2/models/gemma-4-E2B --corpus data2/corpus --eval-dir data2/eval

1. Architecture: the facts the paper states about Gemma 4 E2B (Sec. 3) against the checkpoint's ``config.json``.
2. Tokenizer: <bos>, <eos> and <mask> exist; the references survive encode -> decode unchanged; how many
   HumanEval solutions and MBPP programs fit a canvas of N tokens (with the <eos>), and how long the prompts are.
3. Corpus: shards and the validation file exist; on a sample of rows, how many examples each format yields and
   how many are dropped for length.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from text_vpa.stage0.data import HFTokenizer, complete_example, mbpp_prompt, tokenize_examples, write_example
from text_vpa.stage0.evaluate import load_tasks, reference_text

__all__ = ["EXPECTED_E2B", "architecture_facts", "compare_facts"]

# Text-VPA Sec. 3 / the model card and config.json of google/gemma-4-E2B (read 2026-11-17).
EXPECTED_E2B: dict[str, Any] = {
    "blocks": 35, "width": 1536, "local_window": 512, "local_blocks": 28, "global_blocks": 7,
    "local_blocks_per_global": 4, "last_block": "full_attention", "kv_shared_blocks": 20, "vocabulary": 262_144,
    "tied_embeddings": True, "query_heads": 8, "kv_heads": 1, "local_head_dim": 256, "global_head_dim": 512,
    "per_layer_embedding_dim": 256, "final_logit_softcap": 30.0,
}


def architecture_facts(text_config: Any) -> dict[str, Any]:
    """The same facts read from a ``Gemma4TextConfig``."""
    kinds = list(text_config.layer_types)
    global_at = [i for i, k in enumerate(kinds) if k == "full_attention"]
    gaps = {b - a - 1 for a, b in zip([-1, *global_at[:-1]], global_at, strict=True)}
    per_layer = text_config.per_layer_config
    local_at = kinds.index("sliding_attention") if "sliding_attention" in kinds else 0
    return {
        "blocks": text_config.num_hidden_layers, "width": text_config.hidden_size,
        "local_window": text_config.sliding_window, "local_blocks": kinds.count("sliding_attention"),
        "global_blocks": len(global_at), "local_blocks_per_global": gaps.pop() if len(gaps) == 1 else sorted(gaps),
        "last_block": kinds[-1], "kv_shared_blocks": text_config.num_kv_shared_layers,
        "vocabulary": text_config.vocab_size, "tied_embeddings": bool(text_config.tie_word_embeddings),
        "query_heads": text_config.num_attention_heads, "kv_heads": per_layer[local_at].num_key_value_heads,
        "local_head_dim": per_layer[local_at].head_dim,
        "global_head_dim": per_layer[global_at[0]].head_dim if global_at else None,
        "per_layer_embedding_dim": text_config.hidden_size_per_layer_input,
        "final_logit_softcap": text_config.final_logit_softcapping,
    }


def compare_facts(found: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Human-readable differences (empty when everything matches)."""
    return [f"{key}: expected {want!r}, the checkpoint has {found.get(key)!r}"
            for key, want in expected.items() if found.get(key) != want]


def _quantiles(values: list[int]) -> dict[str, float]:
    a = np.array(values)
    return {"mean": round(float(a.mean()), 1), "p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)),
            "p99": int(np.percentile(a, 99)), "max": int(a.max())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 0 preflight", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="local directory (or Hub id) of the backbone")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--canvas-len", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--min-coverage", type=float, default=0.95, help="benchmark references that must fit the canvas")
    p.add_argument("--corpus-sample", type=int, default=2000)
    p.add_argument("--allow-mismatch", action="store_true", help="report architecture differences without failing")
    p.add_argument("--out", type=Path, default=None, help="write the findings as JSON")
    args = p.parse_args(argv)
    from transformers import AutoConfig, AutoTokenizer

    problems: list[str] = []
    report: dict[str, Any] = {"model": args.model}

    config = AutoConfig.from_pretrained(args.model)
    text_config = getattr(config, "text_config", config)
    facts = architecture_facts(text_config)
    differences = compare_facts(facts, EXPECTED_E2B)
    report["architecture"], report["architecture_differences"] = facts, differences
    print("architecture:", json.dumps(facts))
    for line in differences:
        print("  DIFFERS  " + line)
    if differences and not args.allow_mismatch:
        problems.append("the checkpoint is not the Gemma 4 E2B the paper describes (see DIFFERS above)")

    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(args.model))
    report["tokenizer"] = {"bos": tokenizer.bos_id, "eos": tokenizer.eos_id, "pad": tokenizer.pad_id,
                           "mask": tokenizer.mask_id, "size": tokenizer.vocab_size}
    print("tokenizer:", json.dumps(report["tokenizer"]))
    if tokenizer.vocab_size > text_config.vocab_size:
        problems.append("the tokenizer has more tokens than the model's vocabulary")

    report["benchmarks"] = {}
    for benchmark in ("humaneval", "mbpp"):
        tasks = load_tasks(benchmark, args.eval_dir)
        references = [reference_text(t, benchmark) for t in tasks]
        encoded = tokenizer.encode_batch(references)
        canvas = [len(ids) + 1 for ids in encoded]
        # Generated text reaches the harness through decode: indentation must survive encode -> decode.
        round_trip = float(np.mean([tokenizer.decode(ids) == text for ids, text in zip(encoded, references,
                                                                                         strict=True)]))
        prompts = [len(ids) + 1 for ids in tokenizer.encode_batch(
            [t.context if benchmark == "humaneval" else mbpp_prompt(t.context) for t in tasks])]
        coverage = float(np.mean([n <= args.canvas_len for n in canvas]))
        report["benchmarks"][benchmark] = {"problems": len(tasks), "canvas_tokens": _quantiles(canvas),
                                           "prompt_tokens": _quantiles(prompts), "fit_canvas": round(coverage, 4),
                                           "round_trip": round(round_trip, 4)}
        print(f"{benchmark}: {json.dumps(report['benchmarks'][benchmark])}")
        if round_trip < 0.98:
            problems.append(f"only {round_trip:.1%} of the {benchmark} references survive tokenizer encode -> decode")
        if coverage < args.min_coverage:
            problems.append(f"only {coverage:.1%} of the {benchmark} references fit N = {args.canvas_len}")

    shards = sorted(glob.glob(str(args.corpus / "train-*.jsonl.gz")))
    if not shards or not (args.corpus / "val.jsonl.gz").exists():
        problems.append(f"{args.corpus} has no train-*.jsonl.gz shards or no val.jsonl.gz")
    else:
        rows = []
        with gzip.open(shards[0], "rt", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
                if len(rows) >= args.corpus_sample:
                    break
        rng = random.Random(0)
        pairs = []
        for row in rows:
            made = write_example(row) if rng.random() < 0.5 else None
            kind = "write"
            if made is None:
                made, kind = complete_example(row), "complete"
            if made is not None:
                pairs.append((made[0], made[1], kind))
        examples, dropped = tokenize_examples(pairs, tokenizer, args.canvas_len, args.max_prompt_tokens)
        kept = len(examples) / max(len(rows), 1)
        stats = {}
        if (args.corpus / "stats.json").exists():
            raw = json.loads((args.corpus / "stats.json").read_text())
            stats = {k: raw[k] for k in ("train", "val", "documented", "methods") if k in raw}
        report["corpus"] = {"shards": len(shards), **stats, "sampled_rows": len(rows), "kept": round(kept, 4),
                            "write": sum(e.kind == "write" for e in examples),
                            "complete": sum(e.kind == "complete" for e in examples), "dropped": dropped,
                            "canvas_tokens": _quantiles([len(e.canvas_ids) for e in examples]) if examples else {}}
        print("corpus:", json.dumps(report["corpus"]))
        if kept < 0.5:
            problems.append(f"only {kept:.1%} of the sampled corpus rows yield an example")

    report["problems"] = problems
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    for line in problems:
        print("PROBLEM  " + line, file=sys.stderr)
    print("preflight: " + ("FAILED" if problems else "ok"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
