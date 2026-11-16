# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
train.py -- Stage 0: adapt the frozen backbone into the plan-free realizer.

    python -m text_vpa.stage0.train --corpus data2/corpus --eval-dir data2/eval --run runs/stage0

Objective (Text-VPA Eq. 12 on the mask-source path, AR-shifted): for a canvas w_1 of N tokens, K drawn from the
trained budgets, t = k/K and w_t (``data.training_state``),

    L = sum_j m_j omega_j ( -log p_psi(w_1^j | prompt, w_t, t, h) ) / sum_j m_j omega_j,

with m_j = 1 where w_t^j = [MASK], omega_j = 1 on program tokens and the first <eos> and ``pad_weight`` on the
<eos> filling the rest of the canvas, and p_psi(w^j | .) read from the output at canvas index j - 1.

One step = ``grad_accum`` micro-batches of ``micro_batch`` examples. Per micro-batch: the frozen backbone
prefills the prompts without gradients (the prompt cache does not depend on any trained parameter), then one
realizer pass over the canvas with gradients. The released weights never change; a checkpoint holds only the
LoRA factors, the mask embedding and the modulation (plus optimizer, data position and random state).

Before the first step the trainer runs a worst-case micro-batch (longest prompt, everything masked): if it does
not fit, activation checkpointing is switched on, and if that does not fit either the micro-batch is halved and
the accumulation doubled. The outcome is stored in the run directory and reused on resume.

Every ``eval_every`` steps: masked-token loss on held-out corpus functions at four grid states of K = 8, and a
probe -- Pass@1 on MBPP's *validation* split (90 problems, never the test split) at K = 8 -- which is the
learning curve ``report.py`` uses to say whether training had converged. The final checkpoint is the one
evaluated; nothing is selected on a test set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from text_vpa.engine.runtime import MetricLogger, load_checkpoint, save_checkpoint
from text_vpa.models.realizers.paths import KappaSchedule
from text_vpa.stage0.backbone import RealizerSpec, Stage0Realizer, left_pad, load_gemma4
from text_vpa.stage0.data import (
    Batch,
    CorpusStream,
    Example,
    HFTokenizer,
    Stage0Tokenizer,
    collate,
    load_validation,
    training_state,
)
from text_vpa.stage0.evaluate import autocast, load_tasks, realizer_samples, score
from text_vpa.stage0.sampler import SamplerConfig
from text_vpa.utils.device import resolve_device

__all__ = ["Stage0Config", "Stage0Trainer", "learning_rate"]


@dataclass
class Stage0Config:
    """Everything that defines a Stage-0 run (written to ``RUN/config.json`` and into every checkpoint)."""

    model: str = "google/gemma-4-E2B"
    revision: str | None = None
    canvas_len: int = 256                  # N
    max_prompt_tokens: int = 512
    write_fraction: float = 0.5
    schedule: str = "cosine"
    step_budgets: list[int] = field(default_factory=lambda: [2, 4, 8])
    pad_weight: float = 0.1
    lora_rank: int = 64
    lora_alpha: float = 64.0
    modulation: bool = True
    lr: float = 2e-4
    min_lr_fraction: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    steps: int = 25_000                    # optimizer steps of this run
    schedule_steps: int = 25_000           # horizon of the cosine decay; later steps stay at the floor
    micro_batch: int = 8
    grad_accum: int = 4
    grad_checkpoint: bool = False
    seed: int = 0
    log_every: int = 20
    save_every: int = 500
    eval_every: int = 2_500
    val_examples: int = 256
    probe_samples: int = 5                 # samples per MBPP-validation problem in the probe
    probe_steps: int = 8                   # K of the probe

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stage0Config:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def learning_rate(step: int, cfg: Stage0Config) -> float:
    """Linear warm-up, cosine decay to ``min_lr_fraction * lr`` at ``schedule_steps``, constant afterwards."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.schedule_steps - cfg.warmup_steps))
    return cfg.lr * (cfg.min_lr_fraction + (1 - cfg.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))


class Stage0Trainer:
    """The training loop. ``realizer`` is on ``device``; ``stream`` yields examples; ``validation`` is fixed."""

    def __init__(self, realizer: Stage0Realizer, tokenizer: Stage0Tokenizer, cfg: Stage0Config,
                 stream: CorpusStream, validation: Sequence[Example], run_dir: str | Path, device: torch.device,
                 probe_tasks: Sequence[Any] = (), logger: MetricLogger | None = None) -> None:
        self.realizer, self.tokenizer, self.cfg, self.stream = realizer, tokenizer, cfg, stream
        self.validation, self.run_dir, self.device = list(validation), Path(run_dir), device
        self.probe_tasks = list(probe_tasks)
        self.schedule = KappaSchedule(cfg.schedule)
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.optimizer = torch.optim.AdamW(realizer.trainable_parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                                           weight_decay=cfg.weight_decay)
        self.step = 0
        self.logger = logger or MetricLogger(self.run_dir)
        self.examples_seen = 0

    # -- one micro-batch -----------------------------------------------------------------------
    def loss_on(self, batch: Batch, budget: int | None = None, step_index: int | None = None,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        """The objective on one batch and its statistics (weighted sums, for pooling)."""
        cfg, dev = self.cfg, self.device
        state = training_state(batch, self.schedule, cfg.step_budgets, self.tokenizer.mask_id,
                               self.tokenizer.vocab_size, cfg.pad_weight, generator or self.generator, budget,
                               step_index)
        with autocast(dev):
            cache = self.realizer.prefill(*left_pad(batch.prefix_rows, self.tokenizer.pad_id, dev))
            hidden = self.realizer.canvas_hidden(
                cache, state.canvas_ids.to(dev), state.is_mask.to(dev), state.t.to(dev), state.h.to(dev),
                checkpointing=cfg.grad_checkpoint and torch.is_grad_enabled())
            predicted = hidden[:, : cfg.canvas_len]                  # index j predicts canvas token j + 1
            weights = state.weights.to(dev)
            chosen = weights > 0
            total_weight = weights.sum().clamp_min(1e-8)
            nll, hits = self.realizer.weighted_nll(predicted[chosen], batch.target.to(dev)[chosen], weights[chosen])
        program = (state.weights == 1.0).sum()
        stats = {"nll_sum": float(nll.detach()), "weight_sum": float(total_weight), "hits": float(hits),
                 "masked": float(chosen.sum()), "program_masked": float(program)}
        return nll / total_weight, stats

    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        lr = learning_rate(self.step, cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        pooled = {"nll_sum": 0.0, "weight_sum": 0.0, "hits": 0.0, "masked": 0.0}
        for _ in range(cfg.grad_accum):
            batch = collate(self.stream.take(cfg.micro_batch), cfg.canvas_len, self.tokenizer.eos_id)
            loss, stats = self.loss_on(batch)
            (loss / cfg.grad_accum).backward()
            for key in pooled:
                pooled[key] += stats[key]
            self.examples_seen += cfg.micro_batch
        grad_norm = torch.nn.utils.clip_grad_norm_(self.realizer.trainable_parameters(), cfg.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {self.step}")
        self.optimizer.step()
        self.step += 1
        return {"loss": pooled["nll_sum"] / max(pooled["weight_sum"], 1e-8),
                "accuracy": pooled["hits"] / max(pooled["masked"], 1.0), "lr": lr, "grad_norm": float(grad_norm)}

    # -- evaluation ----------------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Masked-token loss and accuracy on the held-out functions at k = 0, 2, 4, 6 of K = 8 (fixed draws)."""
        cfg = self.cfg
        self.realizer.eval()
        out: dict[str, float] = {}
        sums = {"nll_sum": 0.0, "weight_sum": 0.0, "hits": 0.0, "masked": 0.0}
        for k in (0, 2, 4, 6):
            at_k = {"nll_sum": 0.0, "weight_sum": 0.0, "hits": 0.0, "masked": 0.0}
            generator = torch.Generator().manual_seed(1_000 + k)
            for start in range(0, len(self.validation), cfg.micro_batch):
                batch = collate(self.validation[start:start + cfg.micro_batch], cfg.canvas_len,
                                self.tokenizer.eos_id)
                _, stats = self.loss_on(batch, budget=8, step_index=k, generator=generator)
                for key in at_k:
                    at_k[key] += stats[key]
            out[f"val_loss_k{k}"] = at_k["nll_sum"] / max(at_k["weight_sum"], 1e-8)
            out[f"val_accuracy_k{k}"] = at_k["hits"] / max(at_k["masked"], 1.0)
            for key in sums:
                sums[key] += at_k[key]
        out["val_loss"] = sums["nll_sum"] / max(sums["weight_sum"], 1e-8)
        out["val_accuracy"] = sums["hits"] / max(sums["masked"], 1.0)
        self.realizer.train()
        return out

    @torch.no_grad()
    def probe(self) -> dict[str, float]:
        """Pass@1 on MBPP's validation split at K = ``probe_steps`` (the learning curve; not a test set)."""
        if not self.probe_tasks:
            return {}
        cfg = self.cfg
        self.realizer.eval()
        sampler = SamplerConfig(num_steps=cfg.probe_steps, schedule=cfg.schedule)
        texts, _ = realizer_samples(self.realizer, self.tokenizer, self.probe_tasks, "mbpp", sampler, cfg.canvas_len,
                                    cfg.probe_samples, cfg.probe_samples, seed=7, device=self.device)
        summary, _ = score(self.probe_tasks, "mbpp", texts, [1])
        self.realizer.train()
        return {"probe_pass@1": summary["pass@1"], "probe_parse_rate": summary["parse_rate"]}

    # -- memory --------------------------------------------------------------------------------
    def worst_case_batch(self, micro_batch: int) -> Batch:
        cfg, tok = self.cfg, self.tokenizer
        filler = tok.eos_id + 10 if tok.eos_id + 10 < tok.vocab_size else tok.eos_id
        example = Example([tok.bos_id] + [filler] * (cfg.max_prompt_tokens - 1), [filler] * (cfg.canvas_len - 1)
                          + [tok.eos_id], "complete")
        return collate([example] * micro_batch, cfg.canvas_len, tok.eos_id)

    def fit_memory(self) -> dict[str, Any]:
        """Find a micro-batch setting whose worst case fits the device; updates ``cfg`` in place."""
        cfg = self.cfg
        report: dict[str, Any] = {"tried": []}
        while True:
            try:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(self.device)
                start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self.loss_on(self.worst_case_batch(cfg.micro_batch), budget=8, step_index=0,
                                       generator=torch.Generator().manual_seed(0))
                loss.backward()
                self.optimizer.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0.0
                total = (torch.cuda.get_device_properties(self.device).total_memory / 2**30
                         if self.device.type == "cuda" else float("inf"))
                if peak > 0.92 * total:
                    raise torch.OutOfMemoryError(f"worst case peaks at {peak:.1f} of {total:.1f} GiB")
                report.update(micro_batch=cfg.micro_batch, grad_accum=cfg.grad_accum,
                              grad_checkpoint=cfg.grad_checkpoint, peak_gib=round(peak, 2),
                              seconds=round(time.perf_counter() - start, 2))
                return report
            except torch.OutOfMemoryError as exc:
                report["tried"].append({"micro_batch": cfg.micro_batch, "grad_checkpoint": cfg.grad_checkpoint,
                                        "error": str(exc)[:120]})
                self.optimizer.zero_grad(set_to_none=True)
                if not cfg.grad_checkpoint:
                    cfg.grad_checkpoint = True
                elif cfg.micro_batch > 1:
                    cfg.micro_batch //= 2
                    cfg.grad_accum *= 2
                else:
                    raise

    # -- persistence ---------------------------------------------------------------------------
    def save(self, name: str = "last.pt") -> Path:
        return save_checkpoint(self.run_dir / name, {
            "stage": "stage0", "step": self.step, "examples_seen": self.examples_seen,
            "spec": {**self.realizer.spec.to_dict(), "lora_targets": list(self.realizer.spec.lora_targets)},
            "config": self.cfg.to_dict(), "trainable": self.realizer.trainable_state(),
            "optimizer": self.optimizer.state_dict(), "stream": self.stream.state(),
            "generator": self.generator.get_state(),
        })

    def restore(self, payload: dict[str, Any]) -> None:
        self.realizer.load_trainable_state(payload["trainable"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.generator.set_state(payload["generator"].to(device="cpu", dtype=torch.uint8))
        self.step, self.examples_seen = int(payload["step"]), int(payload.get("examples_seen", 0))

    # -- the loop ------------------------------------------------------------------------------
    def evaluate(self) -> dict[str, float]:
        metrics = {**self.validate(), **self.probe()}
        self.logger.log(self.step, metrics, prefix="eval/")
        return metrics

    def fit(self) -> None:
        cfg = self.cfg
        self.realizer.train()
        if self.step == 0:
            self.evaluate()                                          # the untrained realizer: the curve's origin
        window: list[dict[str, float]] = []
        clock, clock_step = time.perf_counter(), self.step
        while self.step < cfg.steps:
            window.append(self.train_step())
            if self.step % cfg.log_every == 0 or self.step == cfg.steps:
                elapsed = time.perf_counter() - clock
                rate = (self.step - clock_step) / max(elapsed, 1e-9)
                mean = {k: sum(m[k] for m in window) / len(window) for k in window[0]}
                canvas_tokens = cfg.micro_batch * cfg.grad_accum * cfg.canvas_len
                self.logger.log(self.step, {**mean, "steps_per_s": rate, "canvas_tokens_per_s": rate * canvas_tokens,
                                            "eta_h": (cfg.steps - self.step) / max(rate, 1e-9) / 3600,
                                            "examples": self.examples_seen}, prefix="train/")
                window, clock, clock_step = [], time.perf_counter(), self.step
            if self.step % cfg.save_every == 0 or self.step == cfg.steps:
                self.save()
            if self.step % cfg.eval_every == 0 or self.step == cfg.steps:
                self.evaluate()
                shutil.copyfile(self.run_dir / "last.pt", self.run_dir / f"step-{self.step:06d}.pt")
                clock, clock_step = time.perf_counter(), self.step
        self.save()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 0: train the plan-free Gemma 4 realizer", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True, help="corpus directory (train-*.jsonl.gz, val.jsonl.gz)")
    p.add_argument("--eval-dir", type=Path, default=None, help="directory with mbpp.jsonl for the probe")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-memory-check", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override Stage0Config fields of a new run, e.g. steps=25000 lora_rank=64")
    p.add_argument("--steps", type=int, default=None,
                   help="stop at this step (a resumed run keeps its cosine horizon: extra steps run at the floor)")
    return p.parse_args(argv)


def _overrides(cfg: Stage0Config, pairs: Sequence[str]) -> Stage0Config:
    data = cfg.to_dict()
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if key not in data:
            raise SystemExit(f"unknown Stage0Config field {key!r}")
        try:
            data[key] = json.loads(raw)
        except json.JSONDecodeError:
            data[key] = raw
    return Stage0Config.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    args.run.mkdir(parents=True, exist_ok=True)
    last = args.run / "last.pt"
    payload = load_checkpoint(last) if last.exists() else None
    if payload is not None:
        cfg = Stage0Config.from_dict(payload["config"])
        if args.set:
            print("resuming: --set is ignored (the run's configuration is in its checkpoint)", file=sys.stderr)
    else:
        cfg = _overrides(Stage0Config(), args.set)
        if not any(pair.startswith("schedule_steps=") for pair in args.set):
            cfg.schedule_steps = cfg.steps
    if args.steps is not None:
        cfg.steps = args.steps

    model, hf_tokenizer = load_gemma4(cfg.model, revision=cfg.revision, token=os.environ.get("HF_TOKEN") or None)
    tokenizer = HFTokenizer(hf_tokenizer)
    spec = RealizerSpec(mask_id=tokenizer.mask_id, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                        modulation=cfg.modulation)
    torch.manual_seed(cfg.seed)
    realizer = Stage0Realizer.from_pretrained_model(model, spec)
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)                    # text only: frees the encoders' memory
    realizer.to(device)
    stream = CorpusStream(args.corpus, tokenizer, cfg.canvas_len, cfg.max_prompt_tokens, cfg.write_fraction,
                          cfg.seed, state=payload["stream"] if payload else None)
    validation = load_validation(args.corpus, tokenizer, cfg.canvas_len, cfg.max_prompt_tokens, cfg.val_examples)
    probe_tasks = load_tasks("mbpp", args.eval_dir, "validation") if args.eval_dir else []
    trainer = Stage0Trainer(realizer, tokenizer, cfg, stream, validation, args.run, device, probe_tasks)
    if payload is not None:
        trainer.restore(payload)
        print(f"resumed {last} at step {trainer.step}", file=sys.stderr)
    elif not args.no_memory_check:
        report = trainer.fit_memory()
        (args.run / "memory.json").write_text(json.dumps(report, indent=2))
        print(f"memory check: {json.dumps(report)}", file=sys.stderr)
    trainable = sum(p.numel() for p in realizer.trainable_parameters())
    (args.run / "config.json").write_text(json.dumps({**cfg.to_dict(), "trainable_parameters": trainable,
                                                      "mask_id": tokenizer.mask_id}, indent=2))
    print(f"Stage 0: {trainable / 1e6:.1f}M trainable parameters, steps {trainer.step} -> {cfg.steps}, "
          f"{cfg.micro_batch} x {cfg.grad_accum} examples per step", file=sys.stderr)
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
