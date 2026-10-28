# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
train.py -- entry point of the staged training protocol (PR-JEPA Sec. 5.1).

  Stage 1  planner pretraining with L_plan (EMA target encoder, VICReg, stable-rank guard)
      python train.py --stage 1 --data synthetic:4000 --output-dir runs/stage1 planner=tiny realizer=tiny data=tiny
  Stage 2  realizer + readout on the frozen planner's predicted plan (never on f_theta_bar(y))
      python train.py --stage 2 --planner-checkpoint runs/stage1/stage1_final.pt --data synthetic:4000 \\
             --output-dir runs/stage2 realizer=tiny
  Stage 3  optional joint fine-tuning (planner at 0.1 x lr, VICReg on)
      python train.py --stage 3 --init-checkpoint runs/stage2/best_stage2.pt \\
             --planner-checkpoint runs/stage1/stage1_final.pt --data synthetic:4000 --output-dir runs/stage3

Data specs: ``synthetic:N``, ``humaneval:PATH``, ``mbpp:PATH``, ``jsonl:PATH`` (several allowed; PATH may be a
glob such as ``data/corpus/train-*.jsonl.gz``, quoted so that the shell does not expand it).
Trailing ``key=value`` arguments are Hydra overrides of ``text_vpa/configs/conf`` (``planner=tiny``,
``training.batch_size=32``, ``training.precision=bf16``, ...). In Stage 2 the planner, data and VICReg
sections and the tokenizer come from the Stage-1 checkpoint; in Stage 3 every model section (planner,
bridge, realizer, readout, data, VICReg) comes from the Stage-2 checkpoint. Overrides of those sections are
ignored (with a warning); training, energy and sampler overrides apply. ``--max-steps N`` sets the
stage's ``max_steps`` -- the EMA horizon T_train and the learning-rate schedule follow it -- and caps its
warm-up at N. Resuming (``--resume``) continues the data stream exactly with ``data.num_workers=0``.

Distributed data parallel: ``torchrun --nproc_per_node=8 train.py --stage 1 ...``.

Large corpora: ``--build-workers N`` tokenises and parses in N processes, and ``--dataset-cache DIR`` stores the
built examples so later stages and resumed runs load them instead of rebuilding. Memory-limited GPUs:
``training.<stage>.grad_accum_steps=k`` and ``training.activation_checkpointing=true`` (see
``text_vpa/configs/conf/*/l4.yaml``).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import torch

from text_vpa.configs import TextVPAConfig, compose_raw
from text_vpa.data.dataset import (
    MixtureCorruption,
    PRJEPACollator,
    PRJEPADataset,
    build_dataloader,
    load_humaneval,
    load_jsonl,
    load_mbpp,
)
from text_vpa.data.synthetic import CodeRecord, synthetic_corpus
from text_vpa.data.tokenization import (
    ByteTokenizer,
    HFTokenizer,
    TextVPATokenizer,
    tokenizer_from_spec,
    train_bpe_tokenizer,
)
from text_vpa.engine.runtime import DistributedContext, MetricLogger, load_checkpoint, set_seed
from text_vpa.engine.trainer_stage1 import Stage1Trainer
from text_vpa.engine.trainer_stage2 import Stage2Trainer, Stage3Trainer
from text_vpa.models.text_vpa_model import TextVPAModel, TextVPAPlanner

logger = logging.getLogger("text_vpa.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged PR-JEPA training (Sec. 5.1)",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--data", nargs="+", required=True, help="training data specs")
    p.add_argument("--eval-data", nargs="*", default=[], help="evaluation data specs (selects best_stage{2,3}.pt)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--planner-checkpoint", type=Path, help="Stage-1 checkpoint (Stages 2-3)")
    p.add_argument("--init-checkpoint", type=Path, help="Stage-2 checkpoint to fine-tune (Stage 3)")
    p.add_argument("--resume", type=Path, help="resume this stage from its own checkpoint")
    p.add_argument("--max-steps", type=int, help="set the stage's max_steps (EMA horizon and LR schedule follow)")
    p.add_argument("--eval-every", type=int, help="evaluation interval in steps (Stages 2-3)")
    p.add_argument("--device", default=None, help="auto | cpu | cuda | mps (default: training.device)")
    p.add_argument("--allow-rejected", action="store_true", help="accept a Stage-1 checkpoint rejected for collapse")
    p.add_argument("--dataset-cache", type=Path, default=None,
                   help="directory for built datasets (reused by later stages and resumed runs)")
    p.add_argument("--build-workers", type=int, default=1, help="processes that build the dataset")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. planner=tiny training.batch_size=32")
    return p.parse_args(argv)


def load_records(specs: list[str], seed: int) -> list[CodeRecord]:
    records: list[CodeRecord] = []
    for spec in specs:
        kind, _, value = spec.partition(":")
        if kind == "synthetic":
            records += synthetic_corpus(int(value), seed=seed)
        elif kind == "humaneval":
            records += load_humaneval(value)
        elif kind == "mbpp":
            records += load_mbpp(value)
        elif kind == "jsonl":
            paths = sorted(glob.glob(value)) if glob.has_magic(value) else [value]
            if not paths:
                raise SystemExit(f"no file matches {value!r}")
            for path in paths:
                records += load_jsonl(path)
        else:
            raise SystemExit(f"unknown data spec {spec!r} (synthetic:N, humaneval:PATH, mbpp:PATH, jsonl:PATH)")
    if not records:
        raise SystemExit("no training records")
    return records


def make_tokenizer(cfg: TextVPAConfig, records: list[CodeRecord]) -> TextVPATokenizer:
    if cfg.data.tokenizer == "byte":
        return ByteTokenizer()
    if cfg.data.tokenizer == "hf":
        if not cfg.data.tokenizer_name:
            raise SystemExit("data.tokenizer=hf needs data.tokenizer_name")
        return HFTokenizer.from_pretrained(cfg.data.tokenizer_name)
    texts = [r.target for r in records] + [r.context for r in records]
    return train_bpe_tokenizer(texts, vocab_size=cfg.data.bpe_vocab_size)


STAGE_KEYS = {1: "planner", 2: "realizer", 3: "joint"}
STAGE1_SECTIONS = ("planner", "data", "vicreg")
STAGE2_SECTIONS = ("planner", "data", "vicreg", "bridge", "realizer", "readout")


def load_stage_checkpoint(path: Path, stage: int, allow_rejected: bool = False) -> dict:
    state = load_checkpoint(path)
    if state.get("stage") != stage:
        raise SystemExit(f"{path} is not a Stage-{stage} checkpoint")
    if state.get("rejected") and not allow_rejected:
        raise SystemExit(f"{path} was rejected in Stage {stage} ({state.get('rejection_reason')}); "
                         "pass --allow-rejected to use it anyway")
    return state


def resolve_config(raw: dict, stage: int, max_steps: int | None, source: dict | None = None,
                   sections: tuple[str, ...] = (), overrides: list[str] | None = None) -> TextVPAConfig:
    """Replace ``sections`` of the composed (unvalidated) config with the checkpoint's, apply --max-steps,
    then validate the combination once."""
    if source is not None:
        touched = {o.lstrip("+~").split("=")[0].split(".")[0] for o in overrides or ()}
        for section in sections:
            if section in touched:
                logger.warning("override of %r ignored: that section comes from the checkpoint", section)
            raw[section] = source["config"][section]
    if max_steps is not None:
        stage_cfg = raw["training"][STAGE_KEYS[stage]]
        stage_cfg["max_steps"] = max_steps
        stage_cfg["warmup_steps"] = min(stage_cfg["warmup_steps"], max_steps)
    return TextVPAConfig.model_validate(raw)


def build_loader(cfg: TextVPAConfig, records: list[CodeRecord], tokenizer: TextVPATokenizer, ctx: DistributedContext,
                 corruption: bool, shuffle: bool, drop_last: bool, cache_dir: Path | None = None,
                 build_workers: int = 1):
    dataset = PRJEPADataset.cached(cache_dir, records, tokenizer, cfg.planner.num_slots, cfg.data.max_context_tokens,
                                   cfg.data.max_output_tokens, cfg.data.max_structure_tokens,
                                   cfg.readout.max_text_tokens, cfg.data.mode_scheme,
                                   num_proc=max(1, min(build_workers, os.cpu_count() or 1)))
    if len(dataset) == 0:
        raise SystemExit(f"no usable record (skipped: {dataset.skipped})")
    mixture = None
    if corruption and cfg.realizer.kind == "fs_dfm":
        mixture = MixtureCorruption(tokenizer.vocab_size, cfg.realizer.step_budgets, cfg.realizer.schedule,
                                    cfg.realizer.source, tokenizer.mask_id, cfg.data.time_sampling)
    collator = PRJEPACollator(tokenizer, cfg.planner.num_slots, cfg.readout.max_text_tokens, cfg.data.bucket_size,
                              cfg.data.max_output_tokens, mixture)
    loader = build_dataloader(dataset, collator, cfg.training.batch_size, cfg.data.bucket_size, shuffle,
                              cfg.training.seed, cfg.data.num_workers, ctx.rank, ctx.world_size, drop_last)
    if len(loader) == 0:
        raise SystemExit(f"{len(dataset)} examples form no full batch of {cfg.training.batch_size} per rank; "
                         "lower training.batch_size or data.bucket granularity")
    return loader


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    raw = compose_raw(args.overrides)
    stage1_state = stage2_state = None
    if args.stage >= 2:
        if args.planner_checkpoint is None:
            raise SystemExit(f"--stage {args.stage} needs --planner-checkpoint (the Stage-1 checkpoint)")
        stage1_state = load_stage_checkpoint(args.planner_checkpoint, 1, args.allow_rejected)
    if args.stage == 3:
        if args.init_checkpoint is None:
            raise SystemExit("--stage 3 needs --init-checkpoint (a Stage-2 checkpoint)")
        stage2_state = load_stage_checkpoint(args.init_checkpoint, 2)
    if args.stage == 1:
        cfg = resolve_config(raw, 1, args.max_steps)
    elif args.stage == 2:
        cfg = resolve_config(raw, 2, args.max_steps, stage1_state, STAGE1_SECTIONS, args.overrides)
    else:
        cfg = resolve_config(raw, 3, args.max_steps, stage2_state, STAGE2_SECTIONS, args.overrides)
    ctx = DistributedContext.initialize(args.device or cfg.training.device)
    set_seed(cfg.training.seed)                               # identical initialisation on every rank
    records = load_records(args.data, cfg.training.seed)
    eval_records = load_records(args.eval_data, cfg.training.seed + 1) if args.eval_data else None
    out_dir: Path = args.output_dir
    try:
        cache = {"cache_dir": args.dataset_cache, "build_workers": args.build_workers}
        if args.stage == 1:
            if args.resume:                    # the same vocabulary as the interrupted run, not a retrained one
                tokenizer = tokenizer_from_spec(load_checkpoint(args.resume)["tokenizer"])
            else:
                tokenizer = make_tokenizer(cfg, records)
            cfg = cfg.with_vocab_size(tokenizer.vocab_size)
            planner = TextVPAPlanner.from_config(cfg.planner)
            loader = build_loader(cfg, records, tokenizer, ctx, corruption=False, shuffle=True, drop_last=True,
                                  **cache)
            metric_logger = MetricLogger(out_dir, ctx.is_main, args.wandb_project, args.run_name,
                                         cfg.model_dump(mode="json"))
            trainer = Stage1Trainer(cfg, planner, tokenizer, loader, out_dir, ctx, metric_logger)
            if args.resume:
                trainer.resume(args.resume)
            set_seed(cfg.training.seed + ctx.rank)
            result = trainer.train()
            if ctx.is_main:
                status = f"REJECTED ({result.reason})" if result.rejected else "ok"
                print(f"stage 1 finished after {result.steps} steps: {status}; checkpoint {result.checkpoint}")
            return 2 if result.rejected else 0

        assert stage1_state is not None
        tokenizer = tokenizer_from_spec(stage1_state["tokenizer"])
        cfg = cfg.with_vocab_size(tokenizer.vocab_size)
        model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)
        model.planner.load_state_dict(stage1_state["planner"])
        loader = build_loader(cfg, records, tokenizer, ctx, corruption=True, shuffle=True, drop_last=True, **cache)
        eval_loader = (build_loader(cfg, eval_records, tokenizer, ctx, corruption=True, shuffle=False, drop_last=False,
                                    **cache)
                       if eval_records else None)
        metric_logger = MetricLogger(out_dir, ctx.is_main, args.wandb_project, args.run_name,
                                     cfg.model_dump(mode="json"))
        if args.stage == 2:
            trainer: Stage2Trainer = Stage2Trainer(cfg, model, tokenizer, loader, out_dir, ctx, eval_loader,
                                                   metric_logger, str(args.planner_checkpoint))
        else:
            init = stage2_state
            assert init is not None
            if tokenizer_from_spec(init["tokenizer"]).vocab_size != tokenizer.vocab_size:
                raise SystemExit("the Stage-1 and Stage-2 checkpoints use different tokenizers")
            model.planner.load_state_dict(init["planner"])
            model.realizer.load_state_dict(init["realizer"])
            if model.readout is not None and "readout" in init:
                model.readout.load_state_dict(init["readout"])
            trainer = Stage3Trainer(cfg, model, tokenizer, loader, out_dir, ctx, eval_loader, metric_logger,
                                    str(args.init_checkpoint), target_state=stage1_state["target_encoder"])
        if args.resume:
            trainer.resume(args.resume)
        set_seed(cfg.training.seed + ctx.rank)
        result2 = trainer.train(eval_every=args.eval_every)
        if ctx.is_main:
            print(f"stage {args.stage} finished after {result2.steps} steps; checkpoint {result2.checkpoint}; "
                  f"best {result2.best_checkpoint}")
        return 0
    finally:
        ctx.shutdown()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    sys.exit(main())
