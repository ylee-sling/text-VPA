# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
train.py -- the oracles of docs/stage1.md: the Stage 0 realizer plus the bridge, trained on a plan read from the
reference. ``plan=oracle-structure`` (step 1): the reference's true line structure; ``plan=oracle-latent``
(step 2): the frozen backbone's states of the reference's lines (``oracle.py``). Diagnostics. They are never the
method, and everything they write says so.

    python -m text_vpa.stage1.train --init runs/stage0-a/last.pt --corpus data2/corpus --eval-dir data2/eval \\
        --run runs/stage1-oracle                                     # step 1
    python -m text_vpa.stage1.train --init runs/stage0-a/last.pt --corpus data2/corpus --eval-dir data2/eval \\
        --run runs/stage1-latent --set plan=oracle-latent            # step 2

``train_adapters=false`` keeps the realizer's own trained parts (LoRA, mask embedding, modulation) at the init
checkpoint's values: only the bridge and the oracle train, so the run's "no plan" reference is the init
checkpoint itself and not a realizer trained further.

What is the same as Stage 0, so that the plan is the only change: the loop (``Stage0Trainer``), the training
states (mask-source path, the init run's ``prefix_fraction``), the loss, AdamW and its schedule (warm-up, cosine
to the floor at ``steps``), the data format, and every recipe value of the checkpoint the run starts from
(``RECIPE``). The corpus stream continues where that run stopped, so the oracle sees functions the realizer has
not been trained on.

What is new: the gated cross-attention bridge in every block (``bridge.py``, about 32M parameters on Gemma 4
E2B), the oracle's embeddings (``oracle.py``), and each example's line slots as the plan. The realizer's
adapters keep training. The gates start at zero, so step 0 is the Stage 0 checkpoint exactly (tested), and the
first evaluation reproduces its numbers.

Every ``eval_every`` steps, on the held-out functions: Stage 0's masked-token losses with the oracle plan, and
the same with the null plan and with another function's plan (``val_loss_null*``, ``val_loss_shuffled*``).
``plan_gain_k0`` = null - oracle at the all-masked state, in nats per masked token: what the structure tells the
realizer before any token is known. The probe is Pass@1 on MBPP's validation split at K = 8 with the oracle
plan of each problem's normalised reference, and the share of lines on which the samples have the planned kind.

A checkpoint holds the realizer's adapters (as in Stage 0), the bridge, the oracle's embeddings, the optimizer,
the data position and the random states, with ``"stage": "stage1-oracle"`` and the oracle label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from text_vpa.engine.runtime import MetricLogger, load_checkpoint
from text_vpa.stage0.backbone import RealizerSpec, Stage0Realizer, load_gemma4
from text_vpa.stage0.data import Batch, HFTokenizer, Stage0Tokenizer, TrainingState
from text_vpa.stage0.evaluate import load_tasks, score
from text_vpa.stage0.sampler import SamplerConfig
from text_vpa.stage0.train import Stage0Config, Stage0Trainer, _overrides
from text_vpa.stage1.bridge import BridgeSpec, PlanBridge, PlanCache
from text_vpa.stage1.data import PlanCorpusStream, PlanExample, collate_plan, load_plan_validation
from text_vpa.stage1.oracle import ORACLE_KINDS, PLAN_VARIANTS, LatentOracle, StructureOracle, oracle_samples
from text_vpa.stage1.plan import ORACLE_LABEL, require_oracle_plan
from text_vpa.stage1.slots import KINDS, LineSlots
from text_vpa.utils.device import resolve_device

__all__ = ["RECIPE", "OracleTrainer", "Stage1Config", "build_plan_modules", "load_oracle_run"]

STAGE = "stage1-oracle"
# What a run takes over from the Stage 0 checkpoint it starts from: the recipe. Its length (steps, the cosine
# horizon), the evaluation cadence and the memory setting are this run's own.
RECIPE: tuple[str, ...] = (
    "model", "revision", "canvas_len", "max_prompt_tokens", "write_fraction", "schedule", "step_budgets",
    "prefix_fraction", "pad_weight", "lora_rank", "lora_alpha", "modulation", "lr", "min_lr_fraction",
    "warmup_steps", "weight_decay", "grad_clip", "micro_batch", "grad_accum", "seed", "log_every", "save_every",
    "val_examples", "probe_samples", "probe_steps",
)


@dataclass
class Stage1Config(Stage0Config):
    """An oracle run: Stage 0's fields (recipe values come from the init checkpoint) and the plan's."""

    steps: int = 5_000
    schedule_steps: int = 5_000
    eval_every: int = 1_000
    init: str = ""                         # the Stage 0 checkpoint the run starts from
    plan: str = "oracle-structure"
    max_slots: int = 32                    # M
    latent_dim: int = 256                  # d
    bridge_width: int = 256
    bridge_heads: int = 8
    position_dim: int = 64
    plan_dropout: float = 0.1
    gamma: float = 1.0
    spectral_norm: bool = True
    query_position: bool = True
    latent_layers: int = 15                # the latent oracle reads the reference through blocks 0 .. 14
    train_adapters: bool = True            # False: the realizer's own parts stay at the init checkpoint

    def bridge_spec(self) -> BridgeSpec:
        return BridgeSpec(latent_dim=self.latent_dim, width=self.bridge_width, num_heads=self.bridge_heads,
                          position_dim=self.position_dim, num_modes=0, gamma=self.gamma,
                          plan_dropout=self.plan_dropout, spectral_norm=self.spectral_norm,
                          query_position=self.query_position)


def build_plan_modules(cfg: Stage1Config, realizer: Stage0Realizer
                       ) -> tuple[PlanBridge, StructureOracle | LatentOracle]:
    """The bridge (attached to ``realizer``) and the oracle's modules, initialised from ``cfg.seed``."""
    if cfg.plan not in ORACLE_KINDS:
        raise ValueError(f"unknown plan {cfg.plan!r}: this trainer builds {sorted(ORACLE_KINDS)}")
    if cfg.plan == "oracle-latent" and not 1 <= cfg.latent_layers <= realizer.num_kv_layers:
        raise ValueError(f"latent_layers must be in 1 .. {realizer.num_kv_layers} (the KV-storing blocks)")
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(cfg.seed)
    bridge = PlanBridge(cfg.bridge_spec(), realizer.hidden_size, realizer.num_layers)
    oracle: StructureOracle | LatentOracle = (LatentOracle(realizer.hidden_size, cfg.latent_dim)
                                              if cfg.plan == "oracle-latent" else StructureOracle(cfg.latent_dim))
    torch.random.set_rng_state(generator_state)
    bridge.attach(realizer)
    return bridge, oracle


class OracleTrainer(Stage0Trainer):
    """``Stage0Trainer`` with an oracle's plan on every canvas pass."""

    stage = STAGE

    def __init__(self, realizer: Stage0Realizer, bridge: PlanBridge, oracle: StructureOracle | LatentOracle,
                 tokenizer: Stage0Tokenizer, cfg: Stage1Config, stream: PlanCorpusStream,
                 validation: Sequence[PlanExample], run_dir: str | Path, device: torch.device,
                 probe_tasks: Sequence[Any] = (), logger: MetricLogger | None = None,
                 init: dict[str, Any] | None = None) -> None:
        self.bridge, self.oracle = bridge, oracle
        self.variant = "oracle"                                  # which plan ``canvas_pass`` gives the realizer
        self.dropout_generator = torch.Generator().manual_seed(cfg.seed + 1)
        self.init = dict(init or {})
        super().__init__(realizer, tokenizer, cfg, stream, validation, run_dir, device, probe_tasks, logger)

    # -- Stage 0's hooks -----------------------------------------------------------------------
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        own = self.realizer.trainable_parameters() if self.cfg.train_adapters else []
        return [*own, *self.bridge.parameters(), *self.oracle.parameters()]

    def set_training(self, mode: bool) -> None:
        self.realizer.train(mode)
        self.bridge.train(mode)
        self.oracle.train(mode)

    def collate(self, examples: Sequence[PlanExample]) -> Batch:
        return collate_plan(examples, self.cfg.canvas_len, self.tokenizer.eos_id, self.cfg.max_slots)

    def plan_cache(self, batch: Batch, cache: Any) -> PlanCache:
        """Keys and values of the batch's plans: the oracle's (with plan dropout in training), the null plan, or
        each row's neighbour's plan (``self.variant``). The latent oracle reads each row's clean program with the
        frozen backbone after the row's own prompt (``cache``) before the rows are rolled."""
        if self.variant not in PLAN_VARIANTS:
            raise ValueError(f"plan must be one of {PLAN_VARIANTS}")
        slots = batch.slots.to(self.device)
        if self.variant == "null":
            b = slots.kind.shape[0]
            return self.bridge.project(self.bridge.null_conditioning(b))
        if isinstance(self.oracle, LatentOracle):
            canvas = torch.cat([batch.anchor[:, None], batch.target], dim=1).to(self.device)
            states = self.realizer.frozen_states(cache, canvas, self.cfg.latent_layers)
            if self.variant == "shuffled":
                slots, states = slots.roll(1), states.roll(1, dims=0)
            plan = require_oracle_plan(self.oracle(states, slots))
        else:
            if self.variant == "shuffled":
                slots = slots.roll(1)
            plan = require_oracle_plan(self.oracle(slots))
        drop = None
        return self.bridge.project(self.bridge.conditioning(plan.z, plan.mask, drop=drop,
                                                            generator=self.dropout_generator))

    def canvas_pass(self, batch: Batch, state: TrainingState, cache: Any, mode: str, adapters: bool
                    ) -> torch.Tensor:
        dev = self.device
        plan = self.plan_cache(batch, cache) if adapters else None    # adapters off: the frozen backbone
        return self.realizer.canvas_hidden(
            cache, state.canvas_ids.to(dev), state.is_mask.to(dev), state.t.to(dev), state.h.to(dev),
            mode=mode, adapters=adapters, checkpointing=self.cfg.grad_checkpoint and torch.is_grad_enabled(),
            plan=plan)

    def extra_state(self) -> dict[str, Any]:
        return {"plan": self.cfg.plan, "label": ORACLE_LABEL, "bridge_spec": self.bridge.spec.to_dict(),
                "bridge": {k: v.detach().cpu() for k, v in self.bridge.state_dict().items()},
                "oracle_encoder": {k: v.detach().cpu() for k, v in self.oracle.state_dict().items()},
                "dropout_generator": self.dropout_generator.get_state(), "kinds": list(KINDS), "init": self.init}

    def load_extra_state(self, payload: dict[str, Any]) -> None:
        if payload.get("kinds") != list(KINDS):
            raise ValueError("the checkpoint was trained with another list of line kinds")
        self.bridge.load_state_dict(payload["bridge"])
        self.oracle.load_state_dict(payload["oracle_encoder"])
        self.dropout_generator.set_state(payload["dropout_generator"].to(device="cpu", dtype=torch.uint8))
        self.init = dict(payload.get("init") or {})

    # -- one step ------------------------------------------------------------------------------
    def train_step(self) -> dict[str, float]:
        # train_adapters=False: the realizer's own parts take no gradient during the step (they stay trainable
        # parameters of the realizer otherwise, so its checkpoint format does not change).
        frozen = [] if self.cfg.train_adapters else self.realizer.trainable_parameters()
        for p in frozen:
            p.requires_grad_(False)
        try:
            out = super().train_step()
        finally:
            for p in frozen:
                p.requires_grad_(True)
        gates = self.bridge.gates().detach().abs()
        return {**out, "gate_mean": float(gates.mean()), "gate_max": float(gates.max())}

    # -- evaluation ----------------------------------------------------------------------------
    def _grid_loss(self, variant: str) -> dict[str, float]:
        """Stage 0's masked-token loss at k = 0, 2, 4, 6 of K = 8 (the same fixed draws) under ``variant``."""
        cfg = self.cfg
        out: dict[str, float] = {}
        sums = {"nll_sum": 0.0, "weight_sum": 0.0}
        self.variant = variant
        try:
            for k in (0, 2, 4, 6):
                at_k = {"nll_sum": 0.0, "weight_sum": 0.0}
                generator = torch.Generator().manual_seed(1_000 + k)
                for start in range(0, len(self.validation), cfg.micro_batch):
                    batch = self.collate(self.validation[start:start + cfg.micro_batch])
                    _, stats = self.loss_on(batch, budget=8, step_index=k, generator=generator)
                    for key in at_k:
                        at_k[key] += stats[key]
                out[f"val_loss_{variant}_k{k}"] = at_k["nll_sum"] / max(at_k["weight_sum"], 1e-8)
                for key in sums:
                    sums[key] += at_k[key]
        finally:
            self.variant = "oracle"
        out[f"val_loss_{variant}"] = sums["nll_sum"] / max(sums["weight_sum"], 1e-8)
        return out

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Stage 0's validation with the oracle plan, then the grid losses with the null and a shuffled plan.
        ``plan_gain*`` = null - oracle, in nats per masked token."""
        out = super().validate()
        self.set_training(False)
        out.update(self._grid_loss("null"))
        if len(self.validation) > 1:
            out.update(self._grid_loss("shuffled"))
        out["plan_gain_k0"] = out["val_loss_null_k0"] - out["val_loss_k0"]
        out["plan_gain"] = out["val_loss_null"] - out["val_loss"]
        self.set_training(True)
        return out

    @torch.no_grad()
    def probe(self) -> dict[str, float]:
        """Pass@1 on MBPP's validation split at K = ``probe_steps`` with each problem's oracle plan (the gate's
        quantity, on the validation split), and how far the samples follow their plans."""
        if not self.probe_tasks:
            return {}
        cfg = self.cfg
        self.set_training(False)
        sampler = SamplerConfig(num_steps=cfg.probe_steps, schedule=cfg.schedule)
        texts, info, _ = oracle_samples(self.realizer, self.bridge, self.oracle, self.tokenizer, self.probe_tasks,
                                        "mbpp", sampler, cfg.canvas_len, cfg.probe_samples, cfg.probe_samples,
                                        seed=7, device=self.device, max_slots=cfg.max_slots,
                                        latent_layers=cfg.latent_layers)
        summary, _ = score(self.probe_tasks, "mbpp", texts, [1])
        self.set_training(True)
        return {"probe_pass@1": summary["pass@1"], "probe_parse_rate": summary["parse_rate"],
                "probe_kind_match": info["kind_match"], "probe_reproduced": info["reproduced"]}

    # -- memory --------------------------------------------------------------------------------
    def worst_case_batch(self, micro_batch: int) -> Batch:
        """Stage 0's worst case (longest prompt, full canvas) with all M slots in use."""
        cfg, tok = self.cfg, self.tokenizer
        filler = tok.eos_id + 10 if tok.eos_id + 10 < tok.vocab_size else tok.eos_id
        tokens, m = cfg.canvas_len - 1, cfg.max_slots
        lengths = [tokens // m] * (m - 1) + [tokens - (tokens // m) * (m - 1)]
        starts = [sum(lengths[:i]) for i in range(m)]
        slots = LineSlots((KINDS.index("other"),) * m, (1,) * m, tuple(starts), tuple(lengths), m)
        example = PlanExample([tok.bos_id] + [filler] * (cfg.max_prompt_tokens - 1), [filler] * tokens + [tok.eos_id],
                              "complete", slots)
        return self.collate([example] * micro_batch)

    def fit_memory(self) -> dict[str, Any]:
        state = self.dropout_generator.get_state()               # the check must not move the run's random draws
        try:
            return super().fit_memory()
        finally:
            self.dropout_generator.set_state(state)


def load_oracle_run(payload: dict[str, Any], model: Any, tokenizer: Stage0Tokenizer
                    ) -> tuple[Stage0Realizer, PlanBridge, StructureOracle | LatentOracle, Stage1Config]:
    """Rebuild the realizer, the bridge and the oracle of a ``stage1-oracle`` checkpoint on a loaded Gemma 4
    (on the CPU, eval mode)."""
    if payload.get("stage") != STAGE or payload.get("label") != ORACLE_LABEL:
        raise ValueError("not an oracle checkpoint (stage or label is missing)")
    if payload.get("kinds") != list(KINDS):
        raise ValueError("the checkpoint was trained with another list of line kinds")
    cfg = Stage1Config.from_dict(payload["config"])
    spec = RealizerSpec.from_dict(payload["spec"])
    if spec.mask_id != tokenizer.mask_id:
        raise ValueError("the checkpoint's [MASK] id is not this tokenizer's")
    realizer = Stage0Realizer.from_pretrained_model(model, spec)
    realizer.load_trainable_state(payload["trainable"])
    bridge, oracle = build_plan_modules(cfg, realizer)
    bridge.load_state_dict(payload["bridge"])
    oracle.load_state_dict(payload["oracle_encoder"])
    realizer.eval()
    bridge.eval()
    oracle.eval()
    return realizer, bridge, oracle, cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1, steps 1 and 2: train an oracle (a diagnostic)",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", type=Path, default=None, help="the Stage 0 checkpoint to start from (a new run)")
    p.add_argument("--corpus", type=Path, required=True, help="corpus directory (train-*.jsonl.gz, val.jsonl.gz)")
    p.add_argument("--eval-dir", type=Path, default=None, help="directory with mbpp.jsonl for the probe")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-memory-check", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override Stage1Config fields of a new run, e.g. steps=5000 plan_dropout=0.1")
    p.add_argument("--steps", type=int, default=None,
                   help="stop at this step (a resumed run keeps its cosine horizon: extra steps run at the floor)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    args.run.mkdir(parents=True, exist_ok=True)
    last = args.run / "last.pt"
    payload = load_checkpoint(last) if last.exists() else None
    if payload is not None:
        if payload.get("stage") != STAGE:
            raise SystemExit(f"{last} is a {payload.get('stage')!r} checkpoint, not an oracle run")
        cfg = Stage1Config.from_dict(payload["config"])
        start = payload
        if args.set or args.init:
            print("resuming: --init and --set are ignored (the run's configuration is in its checkpoint)",
                  file=sys.stderr)
    else:
        if args.init is None:
            raise SystemExit("a new run needs --init, the Stage 0 checkpoint it starts from")
        start = load_checkpoint(args.init)
        if start.get("stage") != "stage0":
            raise SystemExit(f"--init must be a Stage 0 checkpoint; {args.init} is {start.get('stage')!r}")
        known = {f.name for f in fields(Stage1Config)}
        recipe = {k: v for k, v in start["config"].items() if k in RECIPE and k in known}
        cfg = _overrides(Stage1Config(**recipe, init=str(args.init)), args.set)
        if not any(pair.startswith("schedule_steps=") for pair in args.set):
            cfg.schedule_steps = cfg.steps
    if args.steps is not None:
        cfg.steps = args.steps

    model, hf_tokenizer = load_gemma4(cfg.model, revision=cfg.revision, token=os.environ.get("HF_TOKEN") or None)
    tokenizer = HFTokenizer(hf_tokenizer)
    spec = RealizerSpec.from_dict(start["spec"])
    if spec.mask_id != tokenizer.mask_id:
        raise SystemExit("the checkpoint's [MASK] id is not this tokenizer's")
    torch.manual_seed(cfg.seed)
    realizer = Stage0Realizer.from_pretrained_model(model, spec)
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)                    # text only: frees the encoders' memory
    realizer.load_trainable_state(start["trainable"])            # a new run: the Stage 0 adapters
    bridge, oracle = build_plan_modules(cfg, realizer)
    realizer.to(device)
    bridge.to(device)
    oracle.to(device)
    # A new run continues the corpus where the Stage 0 run stopped: functions its adapters have not seen.
    stream = PlanCorpusStream(args.corpus, tokenizer, cfg.canvas_len, cfg.max_prompt_tokens, cfg.write_fraction,
                              cfg.seed, state=start["stream"], max_slots=cfg.max_slots)
    validation = load_plan_validation(args.corpus, tokenizer, cfg.canvas_len, cfg.max_prompt_tokens,
                                      cfg.val_examples, cfg.max_slots)
    probe_tasks = load_tasks("mbpp", args.eval_dir, "validation") if args.eval_dir else []
    init = payload["init"] if payload is not None else {"path": str(args.init), "stage": start["stage"],
                                                        "step": int(start["step"])}
    trainer = OracleTrainer(realizer, bridge, oracle, tokenizer, cfg, stream, validation, args.run, device,
                            probe_tasks, init=init)
    if payload is not None:
        trainer.restore(payload)
        print(f"resumed {last} at step {trainer.step}", file=sys.stderr)
    elif not args.no_memory_check:
        report = trainer.fit_memory()
        (args.run / "memory.json").write_text(json.dumps(report, indent=2))
        print(f"memory check: {json.dumps(report)}", file=sys.stderr)
    counts = {"realizer_adapters": sum(p.numel() for p in realizer.trainable_parameters()) if cfg.train_adapters else 0,
              "bridge": sum(p.numel() for p in bridge.parameters()),
              "oracle": sum(p.numel() for p in oracle.parameters())}
    kind = ORACLE_KINDS[cfg.plan]
    what = {"structure": "the line structure of the reference program (docs/stage1.md, step 1)",
            "latent": "the frozen backbone's states of the reference program's lines (docs/stage1.md, step 2)"}[kind]
    (args.run / "config.json").write_text(json.dumps(
        {**cfg.to_dict(), "stage": STAGE, "label": ORACLE_LABEL, "init_checkpoint": init,
         "trainable_parameters": counts, "mask_id": tokenizer.mask_id}, indent=2))
    (args.run / "ORACLE.md").write_text(
        f"# {ORACLE_LABEL}\n\nThis run conditions the realizer on {what}.\nIt is a diagnostic. No number of it is a "
        "result of the method.\n")
    print(f"Stage 1 {kind} oracle ({ORACLE_LABEL}): "
          + ", ".join(f"{name} {count / 1e6:.1f}M" for name, count in counts.items())
          + f" trainable parameters{'' if cfg.train_adapters else ' (realizer adapters frozen)'}, steps "
          f"{trainer.step} -> {cfg.steps}, {cfg.micro_batch} x {cfg.grad_accum} examples per step, from "
          f"{init['path']} (step {init['step']})", file=sys.stderr)
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
