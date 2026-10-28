# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
memory_probe.py -- will a configuration fit the GPU, and how long will it train? Run before a long job.

    python -m text_vpa.utils.memory_probe planner=base realizer=l4 data=code training=l4

Builds the model of the given Hydra overrides with the vocabulary ``data.bpe_vocab_size``, then runs real
optimizer steps of Stage 1 and Stage 2 on synthetic worst-case batches: every context, target and structural
view at its maximum length, every slot a text slot, the configured micro-batch size and accumulation. It
reports the parameters, the peak device memory of each stage (CUDA: ``max_memory_allocated``), the time per
optimizer step and the projected wall time of ``max_steps``. Real batches are bucketed by length, so they are
usually shorter (and faster) than these.

Exit status 1 when a stage's peak exceeds ``--budget`` of the device memory (default 0.9).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from text_vpa.configs import TextVPAConfig, load_config
from text_vpa.data.dataset import Batch, MixtureCorruption
from text_vpa.engine.runtime import DistributedContext, MetricLogger
from text_vpa.engine.trainer_stage1 import Stage1Trainer
from text_vpa.engine.trainer_stage2 import Stage2Trainer
from text_vpa.models.text_vpa_model import TextVPAModel, TextVPAPlanner
from text_vpa.utils.device import resolve_device

__all__ = ["ProbeResult", "probe", "worst_case_batch"]


@dataclass
class ProbeResult:
    stage: int
    parameters: int
    trainable: int
    peak_bytes: int | None
    device_bytes: int | None
    seconds_per_step: float
    tokens_per_step: int
    projected_hours: float

    def as_dict(self) -> dict:
        gib = 2**30
        return {
            "stage": self.stage, "parameters_M": round(self.parameters / 1e6, 1),
            "trainable_M": round(self.trainable / 1e6, 1),
            "peak_GiB": None if self.peak_bytes is None else round(self.peak_bytes / gib, 2),
            "device_GiB": None if self.device_bytes is None else round(self.device_bytes / gib, 2),
            "seconds_per_step": round(self.seconds_per_step, 3), "target_tokens_per_step": self.tokens_per_step,
            "projected_hours_for_max_steps": round(self.projected_hours, 1),
        }


def _tokenizer_stub(vocab_size: int) -> SimpleNamespace:
    return SimpleNamespace(pad_id=0, mask_id=1, vocab_size=vocab_size, spec=lambda: {"type": "probe"})


def worst_case_batch(cfg: TextVPAConfig, batch_size: int, generator: torch.Generator) -> Batch:
    """Every sequence at its maximum length. Slot texts are signatures and imports, i.e. lines of the target,
    so together they hold at most ~N tokens: N / L_text full-length text slots per example."""
    v = cfg.realizer.vocab_size
    m_c, n, t_len = cfg.data.max_context_tokens, cfg.data.max_output_tokens, cfg.data.max_structure_tokens
    m, l_text = cfg.planner.num_slots, cfg.readout.max_text_tokens
    b = batch_size

    def ids(*shape: int) -> torch.Tensor:
        return torch.randint(2, v, shape, generator=generator)

    target = ids(b, n)
    target[:, -1] = 0                                                   # the closing [PAD]
    slot_index = (torch.arange(t_len) * m // t_len).repeat(b, 1)       # tokens spread over all M slots
    kinds = torch.full((m,), 4)                                         # non-text slots (e.g. statements)
    kinds[: max(1, min(m, n // l_text))] = 2                            # FUNCTION slots with full texts
    batch = Batch(
        context=ids(b, m_c), context_mask=torch.ones(b, m_c, dtype=torch.bool), target=target,
        target_lengths=torch.full((b,), n - 1), structure=ids(b, t_len),
        structure_mask=torch.ones(b, t_len, dtype=torch.bool), slot_index=slot_index,
        modes=torch.randint(0, cfg.planner.num_modes, (b,), generator=generator),
        slot_kinds=kinds.repeat(b, 1), slot_texts=ids(b, m, l_text), slot_text_mask=(kinds == 2).repeat(b, 1),
        inventory=(torch.rand(b, v, generator=generator) < 0.01).float(),
    )
    mixture = MixtureCorruption(v, cfg.realizer.step_budgets, cfg.realizer.schedule, cfg.realizer.source, 1,
                                cfg.data.time_sampling)
    w_t, t, h = mixture(target, generator)
    return Batch(**{**batch.__dict__, "w_t": w_t, "t": t, "h": h})


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(step, device: torch.device, warmup: int, steps: int) -> tuple[float, int | None]:
    for _ in range(warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)
    seconds = (time.perf_counter() - start) / steps
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return seconds, peak


def probe(cfg: TextVPAConfig, device: torch.device, steps: int = 3, warmup: int = 1,
          stages: tuple[int, ...] = (1, 2)) -> list[ProbeResult]:
    """Run ``warmup`` + ``steps`` optimizer steps of each stage on worst-case batches."""
    out = []
    gen = torch.Generator().manual_seed(0)
    tok = _tokenizer_stub(cfg.realizer.vocab_size)
    ctx = DistributedContext(device=device)
    logger = MetricLogger(".", enabled=False)
    total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
    tcfg = cfg.training
    if 1 in stages:
        accum = tcfg.planner.grad_accum_steps
        planner = TextVPAPlanner.from_config(cfg.planner)
        trainer = Stage1Trainer(cfg, planner, tok, [], "/nonexistent", ctx, logger)        # type: ignore[arg-type]
        micro = [worst_case_batch(cfg, tcfg.batch_size, gen) for _ in range(accum)]
        seconds, peak = _measure(lambda: trainer.train_step(micro), device, warmup, steps)
        params = sum(p.numel() for p in planner.parameters())
        tokens = tcfg.batch_size * accum * cfg.data.max_structure_tokens
        out.append(ProbeResult(1, params, params, peak, total, seconds, tokens,
                               seconds * tcfg.planner.max_steps / 3600))
        del trainer, planner, micro
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if 2 in stages:
        accum = tcfg.realizer.grad_accum_steps
        model = TextVPAModel.from_config(cfg, mask_id=1)
        trainer2 = Stage2Trainer(cfg, model, tok, [], "/nonexistent", ctx, None, logger)  # type: ignore[arg-type]
        micro = [worst_case_batch(cfg, tcfg.batch_size, gen) for _ in range(accum)]
        seconds, peak = _measure(lambda: trainer2.train_step(micro), device, warmup, steps)
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tokens = tcfg.batch_size * accum * cfg.data.max_output_tokens
        out.append(ProbeResult(2, params, trainable, peak, total, seconds, tokens,
                               seconds * tcfg.realizer.max_steps / 3600))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2], choices=(1, 2))
    p.add_argument("--budget", type=float, default=0.9, help="maximum fraction of device memory")
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. planner=base realizer=l4 training=l4")
    args = p.parse_args(argv)
    cfg = load_config(args.overrides)
    cfg = cfg.with_vocab_size(cfg.data.bpe_vocab_size if cfg.data.tokenizer == "bpe" else cfg.realizer.vocab_size)
    device = resolve_device(args.device)
    torch.manual_seed(0)
    results = probe(cfg, device, args.steps, args.warmup, tuple(args.stages))
    report = {"device": str(device), "vocab_size": cfg.realizer.vocab_size, "micro_batch": cfg.training.batch_size,
              "precision": cfg.training.precision, "activation_checkpointing": cfg.training.activation_checkpointing,
              "stages": [r.as_dict() for r in results]}
    print(json.dumps(report, indent=2))
    over = [r for r in results if r.peak_bytes and r.device_bytes and r.peak_bytes > args.budget * r.device_bytes]
    if over:
        print(f"over budget: stage(s) {[r.stage for r in over]} exceed {args.budget:.0%} of device memory; lower "
              "training.batch_size and raise training.<stage>.grad_accum_steps", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
