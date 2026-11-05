# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Diagnose a Stage-2 checkpoint after a Pass@k run.

    python -m text_vpa.evaluation.diagnose                      # run in the repository root
    python -m text_vpa.evaluation.diagnose --skip-generation    # files only (no model, no GPU)

1. Checkpoint selection: every Stage-2 validation loss and the step stored in ``best_stage2.pt``.
2. HumanEval completions (``evaluate.py``'s samples.jsonl): a few of them with their errors, how many parse on
   their own, and how many define the function the prompt asks for.
3. Generation from corpus contexts with ``best_stage2.pt``: training contexts (seen in Stage 2) and validation
   contexts (unseen), each as evaluated (energies on, canvas length from the readout), without the energies,
   with the reference canvas length, with the context's words licensed, and under Stage 2's training
   conditions (no energies, the reference length and the reference mode, as in teacher-forced training). A model that reproduces its training targets but not unseen ones does
   not generalize (too little data); one that fails on its training contexts too points at the sampling path
   (the canvas length, the energies, or a train/inference mismatch).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import glob
import json
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from text_vpa.configs.schema import SamplerConfig
from text_vpa.data.ast_parser import ModeExtractor, TreeSitterASTExtractor
from text_vpa.data.dataset import load_jsonl
from text_vpa.data.synthetic import CodeRecord
from text_vpa.evaluation.tasks import load_humaneval_tasks
from text_vpa.inference.energy_terms import PlanVocabularyEnergy, license_context_words
from text_vpa.inference.loading import build_sampler, load_model
from text_vpa.utils.device import resolve_device

__all__ = ["Score", "defined_functions", "main", "parses", "requested_function"]

_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(", re.MULTILINE)


def defined_functions(code: str) -> list[str]:
    """Names of every ``def`` in ``code``, in order."""
    return _DEF.findall(code)


def requested_function(context: str) -> str | None:
    """The function a prompt asks for: its last ``def`` (HumanEval prompts can define helpers first)."""
    names = defined_functions(context)
    return names[-1] if names else None


def parses(code: str) -> bool:
    """Whether ``code`` is syntactically valid Python on its own."""
    with warnings.catch_warnings():              # generated code triggers SyntaxWarnings (escapes, literals)
        warnings.simplefilter("ignore")
        try:
            ast.parse(code)
        except (SyntaxError, ValueError):        # ValueError: null bytes
            return False
    return True


@dataclass
class Score:
    """Running statistics of generated programs against their references."""

    n: int = 0
    syntax: int = 0
    named: int = 0
    exact: int = 0
    similarity: float = 0.0
    stalls: float = 0.0
    length_error: float = 0.0

    def add(self, generated: str, reference: str, requested: str | None, stalls: float,
            length_error: float = 0.0) -> None:
        self.n += 1
        self.syntax += parses(generated)
        self.named += requested is not None and requested in defined_functions(generated)
        self.exact += generated.strip() == reference.strip()
        self.similarity += difflib.SequenceMatcher(None, reference, generated, autojunk=False).ratio()
        self.stalls += stalls
        self.length_error += length_error

    def fraction(self, field: str) -> float:
        return getattr(self, field) / max(self.n, 1)

    def line(self) -> str:
        return (f"parses {self.fraction('syntax'):4.0%}  defines the function {self.fraction('named'):4.0%}  "
                f"exact {self.fraction('exact'):4.0%}  similarity {self.fraction('similarity'):.2f}  "
                f"stalls/sample {self.fraction('stalls'):5.1f}  |N - N_ref| {self.fraction('length_error'):5.1f}")


# ---------------------------------------------------------------------------------------------
def checkpoint_report(run: Path) -> None:
    print("== 1. Checkpoint selection (Stage 2)")
    metrics = run / "stage2" / "metrics.jsonl"
    evals = []
    if metrics.exists():
        for line in metrics.open(encoding="utf-8"):
            row = json.loads(line)
            if "stage2/eval_loss" in row:
                evals.append((int(row["step"]), float(row["stage2/eval_loss"])))
    if evals:
        best_step, best_loss = min(evals, key=lambda e: e[1])
        print("  validation loss:", ", ".join(f"{s}: {v:.3f}{'*' if s == best_step else ''}" for s, v in evals))
        print(f"  minimum {best_loss:.3f} at step {best_step}")
    else:
        print("  no validation losses in", metrics)
    best = run / "stage2" / "best_stage2.pt"
    if best.exists():
        state = torch.load(best, map_location="cpu", weights_only=True, mmap=True)
        print(f"  best_stage2.pt holds step {state.get('step')} (best_eval_loss {state.get('best_eval_loss')})")


def samples_report(run: Path, data: Path, show: int) -> None:
    print("\n== 2. HumanEval completions (evaluate.py)")
    path, source = run / "eval" / "humaneval" / "samples.jsonl", data / "eval" / "HumanEval.jsonl.gz"
    if not path.exists() or not source.exists():
        print("  no samples.jsonl or HumanEval file")
        return
    tasks = {t.task_id: t for t in load_humaneval_tasks(source)}
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    parsed = named = 0
    for row in rows:
        requested = requested_function(tasks[row["task_id"]].context) if row["task_id"] in tasks else None
        parsed += parses(row["completion"])
        named += requested is not None and requested in defined_functions(row["completion"])
    total = max(len(rows), 1)
    print(f"  {len(rows)} completions: {parsed / total:.0%} parse on their own, "
          f"{named / total:.0%} define the function the prompt asks for")
    for row in [r for r in rows if r.get("sample") == 0][:show]:
        task = tasks.get(row["task_id"])
        requested = requested_function(task.context) if task else None
        print(f"\n  --- {row['task_id']}, sample 0 ({row['status']}); the prompt asks for {requested}()")
        lines = row["completion"].rstrip().splitlines()
        for i, text in enumerate(lines[:20], 1):
            print(f"  {i:3d} | {text}")
        if len(lines) > 20:
            print(f"      ... {len(lines) - 20} more lines")
        if row.get("detail"):
            print("  error:", row["detail"].strip().splitlines()[-1])


def generation_report(run: Path, data: Path, examples: int, k_steps: int | None, device_name: str,
                      seed: int) -> None:
    print("\n== 3. Generation from corpus contexts (best_stage2.pt)")
    device = resolve_device(device_name)
    loaded = load_model(run / "stage2" / "best_stage2.pt", device)
    tok, cfg = loaded.tokenizer, loaded.cfg
    samplers = {"as evaluated": build_sampler(loaded, use_energy=True),
                "no energies": build_sampler(loaded, use_energy=False),
                "context licensed": build_sampler(loaded, use_energy=True)}
    samplers["context licensed"].scfg = SamplerConfig.model_validate(
        {**cfg.sampler.model_dump(), "license_context": True})
    extractor, modes = TreeSitterASTExtractor(cfg.data.language), ModeExtractor(cfg.data.mode_scheme)
    energy = samplers["as evaluated"].energy
    analysis = energy.analysis if energy is not None else None
    base = frozenset().union(*(t.base_licensed for t in energy.terms if isinstance(t, PlanVocabularyEnergy))) \
        if energy is not None else frozenset()
    k = k_steps or cfg.realizer.num_steps
    train_files = sorted(glob.glob(str(data / "corpus" / "train-*.jsonl.gz")))
    val_file = data / "corpus" / "val.jsonl.gz"
    splits: dict[str, list[CodeRecord]] = {
        "training contexts (seen in Stage 2)": load_jsonl(train_files[0]) if train_files else [],
        "validation contexts (unseen)": load_jsonl(val_file) if val_file.exists() else [],
    }
    rng = random.Random(seed)
    verdict: dict[str, Score] = {}
    example: tuple[CodeRecord, str] | None = None
    for split, records in splits.items():
        if not records:
            print(f"  {split}: none found")
            continue
        picked = rng.sample(records, min(examples, len(records)))
        scores = {name: Score() for name in ("as evaluated", "no energies", "reference length", "context licensed",
                                             "training conditions")}
        licensed_size = covered = covered_with_context = 0.0
        for i, record in enumerate(picked):
            ids = tok.encode(record.context)[: cfg.data.max_context_tokens] or [tok.pad_id]
            context = torch.tensor([ids], device=device)
            target_len = len(tok.encode(record.target))
            bucket = cfg.data.bucket_size
            reference_n = min(-(-(target_len + 1) // bucket) * bucket, cfg.realizer.max_output_len)
            reference_mode = _reference_mode(record.target, extractor, modes, device)
            for name, score in scores.items():
                sampler = samplers[{"reference length": "as evaluated", "training conditions": "no energies"}
                                   .get(name, name)]
                generator = torch.Generator(device=device).manual_seed(seed * 100_003 + i)
                train_like = name == "training conditions"
                result = sampler.sample(context, num_steps=k, generator=generator,
                                        length=reference_n if name in ("reference length", "training conditions")
                                        else None, mode=reference_mode if train_like else None)
                text = result.texts[0]
                score.add(text, record.target, requested_function(record.context), float(result.unrealized),
                          abs(int(result.tokens.shape[1]) - reference_n))
                if example is None and name == "as evaluated":
                    example = (record, text)
                plan = result.plans[0] if result.plans else None
                if name == "as evaluated" and analysis is not None and plan is not None \
                        and plan.licensed_tokens is not None:
                    # How much of the reference's own vocabulary the plan-vocabulary energy lets through.
                    reference_words = {int(t) for t in tok.encode(record.target) if analysis.word_bearing[int(t)]}
                    licensed = plan.licensed_tokens | base
                    with_context = license_context_words([plan], [record.context], tok, analysis)[0]
                    licensed_size += len(plan.licensed_tokens)
                    covered += len(reference_words & licensed) / max(len(reference_words), 1)
                    assert with_context.licensed_tokens is not None
                    covered_with_context += (len(reference_words & (with_context.licensed_tokens | base))
                                             / max(len(reference_words), 1))
        print(f"  {split}: {len(picked)} examples, K = {k}")
        for name, score in scores.items():
            print(f"    {name:<19} {score.line()}")
        n = max(len(picked), 1)
        if analysis is not None:
            print(f"    plan-vocabulary energy: the readout licenses {licensed_size / n:.0f} tokens per plan (plus "
                  f"{len(base)} keyword/builtin tokens); they cover {covered / n:.0%} of the reference's word "
                  f"tokens, {covered_with_context / n:.0%} with the context's words licensed")
        verdict[split] = scores["as evaluated"]
        verdict[split + " / reference length"] = scores["reference length"]
        verdict[split + " / no energies"] = scores["no energies"]
        verdict[split + " / context licensed"] = scores["context licensed"]
        verdict[split + " / training conditions"] = scores["training conditions"]
    if example is not None:
        record, text = example
        print("\n  one training example, as evaluated -- reference (left) | generated (right):")
        ref_lines, gen_lines = record.target.splitlines(), text.splitlines()
        for i in range(min(max(len(ref_lines), len(gen_lines)), 16)):
            left = ref_lines[i] if i < len(ref_lines) else ""
            right = gen_lines[i] if i < len(gen_lines) else ""
            print(f"  {left[:58]:<58} | {right[:70]}")
    reading(verdict)


def _reference_mode(target: str, extractor: TreeSitterASTExtractor, modes: ModeExtractor,
                    device: torch.device) -> torch.Tensor | None:
    """The mode u Stage 2 teacher-forces for ``target`` ([1] int64), or None if the target does not parse."""
    try:
        return torch.tensor([modes(extractor.parse(target))], device=device)
    except (ValueError, RuntimeError):
        return None


def _reproduces(score: Score | None) -> bool:
    return score is not None and score.fraction("syntax") >= 0.5 and score.fraction("similarity") >= 0.6


def reading(verdict: dict[str, Score]) -> None:
    """A few lines of interpretation; the thresholds are stated in the output."""
    seen, unseen = "training contexts (seen in Stage 2)", "validation contexts (unseen)"
    train = verdict.get(seen)
    if train is None:
        return
    print("\n== Reading (thresholds: parses >= 50% and similarity >= 0.6 count as reproducing)")
    cond = verdict.get(seen + " / training conditions")
    ref = verdict.get(seen + " / reference length")
    off = verdict.get(seen + " / no energies")
    ctx = verdict.get(seen + " / context licensed")

    def better(alt: Score | None) -> bool:
        return alt is not None and (alt.fraction("syntax") >= train.fraction("syntax") + 0.2
                                    or alt.fraction("similarity") >= train.fraction("similarity") + 0.2
                                    or alt.fraction("named") >= train.fraction("named") + 0.2)

    if cond is not None:
        if _reproduces(cond):
            print("  Under Stage 2's training conditions (no energies, reference length and mode) it reproduces")
            print("  programs it trained on: the realizer and the sampler work at this scale.")
            if not _reproduces(verdict.get(unseen + " / training conditions")):
                print("  It does not on unseen contexts: it does not generalize. The remedy is more training data")
                print("  (and a model sized to it), not decoding settings.")
        else:
            print("  Even under Stage 2's training conditions (no energies, reference length and mode) it does not")
            print("  reproduce programs it trained on: the realizer is under-trained for these programs (or the")
            print("  recipe cannot fit them). Check that it can memorize a few hundred programs")
            print("  (scripts/gcp/overfit_check.sh) before collecting more data.")
        if not _reproduces(train) and (better(off) or better(ctx)):
            print("  Separately, the plan-vocabulary energy makes generation worse (compare 'as evaluated' with 'no")
            print("  energies' and 'context licensed'): evaluate with --no-energy or --license-context.")
        return
    if _reproduces(train):
        print("  It reproduces its training targets but not unseen programs: it memorized the corpus and does")
        print("  not generalize. The remedy is more training data (and a model sized to it), not decoding settings.")
    elif ref is not None and ref.fraction("similarity") >= train.fraction("similarity") + 0.2:
        print("  It fails on its own training contexts, but much less with the reference canvas length: the")
        print("  readout's length head is the first suspect.")
    elif better(off) or better(ctx):
        which = "with the context's words licensed" if better(ctx) else "without the energies"
        print(f"  It fails on its own training contexts as evaluated, but does much better {which}: the")
        print("  plan-vocabulary energy (the readout's inventory) forbids the tokens it needs. Evaluate with")
        print("  --license-context (or --no-energy) before concluding anything about the realizer.")
    else:
        print("  It fails even on contexts it trained on, with any length and with or without the energies:")
        print("  the sampling path does not match training (or Stage 2 never learned to realize); investigate")
        print("  before collecting more data.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/l4"), help="run directory (stage2/, eval/)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="data directory (corpus/, eval/)")
    parser.add_argument("--examples", type=int, default=16, help="corpus examples per split")
    parser.add_argument("--show", type=int, default=3, help="HumanEval completions to print")
    parser.add_argument("--k-steps", type=int, default=None, help="K (default: the checkpoint's)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-generation", action="store_true", help="parts 1 and 2 only")
    args = parser.parse_args(argv)
    checkpoint_report(args.run)
    samples_report(args.run, args.data, args.show)
    if not args.skip_generation:
        generation_report(args.run, args.data, args.examples, args.k_steps, args.device, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
