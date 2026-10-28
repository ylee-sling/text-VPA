# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
trainer_stage2.py -- Stage 2, realizer training, and the optional Stage 3, joint fine-tuning (Sec. 5.1).

Stage 2. The planner (f_theta, pi_phi^h, g_phi) is frozen. Each batch is planned from its context alone,
Z_target = g_phi(s_x, u, Delta), and the realizer F_psi (with its bridge) learns L_real (Eq. 12) conditioned
on that predicted plan, with step-budget conditioning h = 1/K and plan dropout 0.1 inside the bridge. The
plan readout learns, on the same predicted plan, to decode the symbolic content the energies need.

Anti-leakage invariant ("never condition on f_theta_bar(y)"), enforced rather than assumed:
  * no target encoder exists in Stage 2, and the structural view y of a batch is never passed to a network
    (only labels derived from it reach the readout's loss);
  * the realizer accepts only a ``PredictedPlan`` (``require_predicted_plan``);
  * planner parameters do not require grad, are absent from the optimizer, and the planner stays in eval
    mode; a weight fingerprint is re-checked every ``fingerprint_every`` steps.

The mode u fed to g_phi is the reference mode (teacher forcing, ``training.stage2_mode_source=reference``)
or a draw from pi_phi^h (``sampled``); in both cases the plan itself is predicted from the context.

Stage 3 (``Stage3Trainer``). The planner is unfrozen at ``joint.planner_lr_scale`` (0.1) times the learning
rate, L_plan with its VICReg penalties stays on (the EMA target encoder is restored from Stage 1), and
L_real's gradients reach the planner through Z_target. The realizer still conditions on the predicted plan.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from text_vpa.configs.schema import TextVPAConfig
from text_vpa.data.dataset import Batch, epoch_batches
from text_vpa.data.tokenization import TextVPATokenizer
from text_vpa.engine.losses import PlanLoss, StableRankMonitor, masked_rows
from text_vpa.engine.runtime import (
    DistributedContext,
    MetricLogger,
    autocast,
    build_optimizer,
    grouped,
    load_checkpoint,
    maybe_no_sync,
    rng_state,
    save_checkpoint,
    set_activation_checkpointing,
    set_rng_state,
    warmup_cosine,
)
from text_vpa.models.ema import EMATargetEncoder
from text_vpa.models.encoders import TargetEncoder
from text_vpa.models.plan import PredictedPlan
from text_vpa.models.readout import ReadoutTargets
from text_vpa.models.realizers.fs_dfm import ConditionalFSDFMRealizer
from text_vpa.models.realizers.paths import mixture_path_sample
from text_vpa.models.text_vpa_model import TextVPAModel

__all__ = ["JointObjective", "LeakageError", "RealizerObjective", "Stage2Result", "Stage2Trainer", "Stage3Trainer"]


class LeakageError(RuntimeError):
    """The Stage-2 anti-leakage invariant was violated."""


class RealizerObjective(nn.Module):
    """L_real (Eq. 12) + readout loss for one batch, given a predicted plan."""

    def __init__(self, model: TextVPAModel, cfg: TextVPAConfig, pad_id: int) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.pad_id = pad_id

    def forward(self, batch: Batch, plan: PredictedPlan) -> tuple[Tensor, dict[str, Tensor]]:
        realizer = self.model.realizer
        rcfg, tcfg, ocfg = self.cfg.realizer, self.cfg.training, self.cfg.readout
        with autocast(batch.target.device, tcfg.precision):
            cache = realizer.prepare(plan)                          # plan dropout inside (training mode)
            if isinstance(realizer, ConditionalFSDFMRealizer):
                if batch.w_t is None:
                    s = mixture_path_sample(batch.target, realizer.schedule, realizer.vocab_size, rcfg.step_budgets,
                                            realizer.source, realizer.mask_id, self.cfg.data.time_sampling)
                    w_t, t, h = s.w_t, s.t, s.h
                else:
                    w_t, t, h = batch.w_t, batch.t, batch.h
                logits = realizer(w_t, t, h, cache)
            else:
                l_t, t, h = realizer.training_state(batch.target, rcfg.step_budgets, self.cfg.data.time_sampling)
                logits = realizer(l_t, t, h, cache)
        loss_real = realizer.loss(logits.float(), batch.target, tcfg.realizer_loss_reduction)
        terms: dict[str, Tensor] = {"loss_real": loss_real.detach()}
        total = loss_real
        if self.model.readout is not None and tcfg.readout_weight > 0:
            readout = self.model.readout
            out = readout(plan.z_target.float())
            targets = ReadoutTargets(batch.slot_kinds, batch.slot_texts, batch.slot_text_mask, batch.inventory,
                                     readout.length_bucket(batch.target_lengths + 1))
            loss_ro, ro_terms = readout.loss(out, targets, self.pad_id, (ocfg.kind_weight, ocfg.text_weight,
                                                                         ocfg.inventory_weight, ocfg.length_weight))
            total = total + tcfg.readout_weight * loss_ro
            terms.update(ro_terms)
        return total, terms


@dataclass
class Stage2Result:
    steps: int
    checkpoint: Path | None
    best_checkpoint: Path | None
    best_eval_loss: float | None


class Stage2Trainer:
    """Stage 2 training loop (planner frozen).

    Args:
        cfg: full config (``training.realizer`` holds the stage's optimiser settings).
        model: planner (already trained in Stage 1), realizer and readout.
        tokenizer: stored in checkpoints.
        train_loader / eval_loader: iterables of ``Batch`` (``eval_loader`` selects ``best_stage2.pt``).
        output_dir: checkpoints and metrics.
        ctx: distributed context.
        source_checkpoint: path of the Stage-1 checkpoint (recorded for provenance).
    """

    stage = 2
    fingerprint_every = 100

    def __init__(self, cfg: TextVPAConfig, model: TextVPAModel, tokenizer: TextVPATokenizer,
                 train_loader: Iterable[Batch], output_dir: str | Path, ctx: DistributedContext | None = None,
                 eval_loader: Iterable[Batch] | None = None, metric_logger: MetricLogger | None = None,
                 source_checkpoint: str | None = None) -> None:
        self.cfg = cfg
        self.ctx = ctx or DistributedContext(device=torch.device("cpu"))
        self.device = self.ctx.device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.loader = train_loader
        self.eval_loader = eval_loader
        self.output_dir = Path(output_dir)
        self.source_checkpoint = source_checkpoint
        self.logger = metric_logger or MetricLogger(self.output_dir, enabled=self.ctx.is_main)
        self.step = 0
        self.epoch = 0                      # data position, so a resumed run continues the same stream
        self.batch_in_epoch = 0
        self.best_eval: float | None = None
        self._configure()

    # -----------------------------------------------------------------------------------------
    def _trainable_parts(self) -> list[tuple[str, nn.Module, float]]:
        lr = self.stage_cfg.learning_rate
        parts: list[tuple[str, nn.Module, float]] = [("realizer", self.model.realizer, lr)]
        if self.model.readout is not None:
            parts.append(("readout", self.model.readout, lr))
        return parts

    @property
    def stage_cfg(self):  # noqa: ANN201 - StageConfig
        return self.cfg.training.realizer

    def _configure(self) -> None:
        planner = self.model.planner
        planner.requires_grad_(False)
        planner.eval()
        set_activation_checkpointing(self.model.realizer, self.cfg.training.activation_checkpointing)
        self.optimizer = build_optimizer(self._trainable_parts(), self.stage_cfg.weight_decay)
        self.scheduler = warmup_cosine(self.optimizer, self.stage_cfg.warmup_steps, self.stage_cfg.max_steps)
        self.objective: nn.Module = RealizerObjective(self.model, self.cfg, self.tokenizer.pad_id)
        if self.ctx.enabled:
            self.objective = DistributedDataParallel(
                self.objective, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=True)
        self.assert_planner_frozen()
        self._fingerprint = self.planner_fingerprint()

    def planner_fingerprint(self) -> str:
        """SHA-256 over every planner tensor's bytes: any change (sign flips and permutations included) shows."""
        digest = hashlib.sha256()
        for name, tensor in self.model.planner.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def assert_planner_frozen(self) -> None:
        """The Stage-2 invariant: the planner cannot change and never sees the target."""
        planner = self.model.planner
        if any(p.requires_grad for p in planner.parameters()):
            raise LeakageError("planner parameters require grad in Stage 2")
        planner_ids = {id(p) for p in planner.parameters()}
        if any(id(p) in planner_ids for g in self.optimizer.param_groups for p in g["params"]):
            raise LeakageError("planner parameters are in the Stage-2 optimizer")
        if planner.training:
            raise LeakageError("the planner must stay in eval mode in Stage 2")

    def plan(self, batch: Batch) -> PredictedPlan:
        """Z_target = g_phi(s_x, u, Delta) from the context only (frozen planner, no gradient)."""
        mode = batch.modes if self.cfg.training.stage2_mode_source == "reference" else None
        with torch.no_grad(), autocast(self.device, self.cfg.training.precision):
            plan = self.model.planner.plan(batch.context, batch.context_mask, mode=mode, strategy="sample")
        return plan.detach()

    def train_step(self, batch: Batch | Sequence[Batch]) -> dict[str, float]:
        """One optimizer step over one batch or a group of micro-batches (loss scaled by 1 / #micro-batches;
        DDP all-reduces once, on the last micro-batch)."""
        self.model.realizer.train()
        if self.model.readout is not None:
            self.model.readout.train()
        self.model.planner.eval()
        micro = [batch] if isinstance(batch, Batch) else list(batch)
        if not micro:
            raise ValueError("train_step needs at least one batch")
        self.optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        loss_sum = 0.0
        for i, mb in enumerate(micro):
            mb = mb.to(self.device)
            plan = self.plan(mb)
            with maybe_no_sync(self.objective, skip_sync=i < len(micro) - 1):
                loss, terms = self.objective(mb, plan)
                (loss / len(micro)).backward()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            loss_sum += float(loss.detach())
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.training.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        if self.step % self.fingerprint_every == 0:
            self.verify_planner_unchanged()
        n = len(micro)
        out = {k: v / n for k, v in sums.items()}
        out.update(loss=loss_sum / n, grad_norm=float(grad_norm), lr=self.optimizer.param_groups[0]["lr"])
        return out

    def verify_planner_unchanged(self) -> None:
        if self.planner_fingerprint() != self._fingerprint:
            raise LeakageError("planner weights changed during Stage 2")
        self.assert_planner_frozen()

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Batch], seed: int = 1234) -> float:
        """Mean training objective on ``loader`` with fixed corruption noise and no plan dropout."""
        was = torch.get_rng_state()
        torch.manual_seed(seed)
        self.model.eval()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            loss, _ = self.objective(batch, self.plan(batch))
            total += float(loss) * batch.batch_size
            count += batch.batch_size
        torch.set_rng_state(was)
        return self.ctx.mean(total / max(count, 1))

    def train(self, max_steps: int | None = None, eval_every: int | None = None) -> Stage2Result:
        limit = self.stage_cfg.max_steps if max_steps is None else max_steps
        tcfg = self.cfg.training
        eval_every = eval_every or tcfg.save_every
        best_path = None
        start = time.time()
        prefix = f"stage{self.stage}/"
        last_eval = -1
        while self.step < limit:
            consumed = 0
            batches = epoch_batches(self.loader, self.epoch, self.batch_in_epoch)
            for group in grouped(batches, self.stage_cfg.grad_accum_steps):
                consumed += len(group)
                metrics = self.train_step(group)
                self.batch_in_epoch += len(group)
                if self.step % tcfg.log_every == 0 or self.step == 1:
                    metrics["steps_per_s"] = self.step / max(time.time() - start, 1e-9)
                    self.logger.log(self.step, metrics, prefix=prefix)
                if self.eval_loader is not None and self.step % eval_every == 0:
                    best_path = self._eval_and_keep_best(prefix) or best_path
                    last_eval = self.step
                if self.ctx.is_main and self.step % tcfg.save_every == 0:
                    self.save(f"stage{self.stage}_last.pt")
                if self.step >= limit:
                    break
            else:
                if consumed == 0 and self.batch_in_epoch == 0:
                    raise RuntimeError("the training loader yields no batch")
                self.epoch, self.batch_in_epoch = self.epoch + 1, 0
        self.verify_planner_unchanged()
        if self.eval_loader is not None and last_eval != self.step:
            best_path = self._eval_and_keep_best(prefix) or best_path
        path = self.save(f"stage{self.stage}_final.pt")
        if best_path is None and path is not None:
            best_path = self.save(f"best_stage{self.stage}.pt")      # no evaluation data: best = final
        return Stage2Result(self.step, path, best_path, self.best_eval)

    def _eval_and_keep_best(self, prefix: str) -> Path | None:
        value = self.evaluate(self.eval_loader)  # type: ignore[arg-type]
        self.logger.log(self.step, {"eval_loss": value}, prefix=prefix)
        if self.best_eval is None or value < self.best_eval:
            self.best_eval = value
            return self.save(f"best_stage{self.stage}.pt")
        return None

    # -----------------------------------------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        out = {
            "stage": self.stage, "step": self.step, "config": self.cfg.model_dump(mode="json"),
            "tokenizer": self.tokenizer.spec(), "planner": self.model.planner.state_dict(),
            "realizer": self.model.realizer.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "rng": rng_state(), "rejected": False,
            "source_checkpoint": self.source_checkpoint, "best_eval_loss": self.best_eval,
            "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
        }
        if self.model.readout is not None:
            out["readout"] = self.model.readout.state_dict()
        return out

    def save(self, name: str) -> Path | None:
        if not self.ctx.is_main:
            return None
        return save_checkpoint(self.output_dir / name, self.payload())

    def resume(self, path: str | Path) -> None:
        state = load_checkpoint(path, map_location=self.device)
        if state.get("stage") != self.stage:
            raise ValueError(f"{path} is not a Stage-{self.stage} checkpoint")
        self.model.planner.load_state_dict(state["planner"])
        self.model.realizer.load_state_dict(state["realizer"])
        if self.model.readout is not None and "readout" in state:
            self.model.readout.load_state_dict(state["readout"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        set_rng_state(state["rng"])
        self.step = int(state["step"])
        self.epoch = int(state.get("epoch", 0))
        self.batch_in_epoch = int(state.get("batch_in_epoch", 0))
        self.best_eval = state.get("best_eval_loss")
        if self.stage == 2:
            self._fingerprint = self.planner_fingerprint()


class JointObjective(nn.Module):
    """Stage 3 forward: plan from the context (with gradients), L_real + readout, and L_plan against the EMA
    target encoder (whose output is only a loss target, never realizer conditioning)."""

    def __init__(self, model: TextVPAModel, cfg: TextVPAConfig, pad_id: int, target_encoder: TargetEncoder) -> None:
        super().__init__()
        self.realizer_objective = RealizerObjective(model, cfg, pad_id)
        self.plan_loss = PlanLoss.from_config(cfg.vicreg)
        self.cfg = cfg
        # A plain attribute, not a submodule: the target encoder is frozen and outside DDP's parameters.
        self.__dict__["target_encoder"] = target_encoder

    def forward(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor], PredictedPlan, Tensor]:
        model = self.realizer_objective.model
        mode = batch.modes if self.cfg.training.stage2_mode_source == "reference" else None
        with autocast(batch.context.device, self.cfg.training.precision):
            plan = model.planner.plan(batch.context, batch.context_mask, mode=mode, strategy="sample")
        loss_real, terms = self.realizer_objective(batch, plan)
        target = self.target_encoder(batch.structure, batch.slot_index, self.cfg.planner.num_slots,
                                     batch.structure_mask)
        jepa = self.plan_loss(
            predictions=plan.head_predictions.float(), target=target.s_y, context=plan.s_x.float(),
            plan=plan.z_target.float(), mode_logits=plan.mode_logits.float(),  # type: ignore[union-attr]
            mode_targets=batch.modes, context_mask=batch.context_mask, slot_mask=target.slot_mask,
        )
        terms = {**terms, **{f"plan_{k}": v for k, v in jepa.terms.items()}}
        return loss_real + jepa.total, terms, plan.detach(), target.slot_mask


class Stage3Trainer(Stage2Trainer):
    """Stage 3: planner unfrozen at ``joint.planner_lr_scale`` x lr; L_real + readout + L_plan (VICReg on).

    Args (in addition to ``Stage2Trainer``):
        target_state: the Stage-1 target encoder's state dict (EMA weights), required for L_plan.

    The EMA schedule restarts at m0 over the Stage-3 steps (Eq. 4 with T_train = joint.max_steps).
    """

    stage = 3

    def __init__(self, *args: Any, target_state: dict[str, Tensor] | None = None, **kwargs: Any) -> None:
        self._target_state = target_state
        super().__init__(*args, **kwargs)

    @property
    def stage_cfg(self):  # noqa: ANN201 - JointStageConfig
        return self.cfg.training.joint

    def _configure(self) -> None:
        planner = self.model.planner
        planner.requires_grad_(True)
        set_activation_checkpointing(self.model.realizer, self.cfg.training.activation_checkpointing)
        lr = self.stage_cfg.learning_rate
        parts = self._trainable_parts() + [("planner", planner, lr * self.stage_cfg.planner_lr_scale)]
        self.optimizer = build_optimizer(parts, self.stage_cfg.weight_decay)
        self.scheduler = warmup_cosine(self.optimizer, self.stage_cfg.warmup_steps, self.stage_cfg.max_steps)
        self.target_encoder = TargetEncoder.from_online(planner.context_encoder)
        if self._target_state is not None:
            self.target_encoder.load_state_dict(self._target_state)
        self.ema = EMATargetEncoder.from_config(planner.context_encoder, self.target_encoder, self.cfg.vicreg,
                                                total_steps=self.stage_cfg.max_steps)
        self.monitor = StableRankMonitor.from_config(self.cfg.planner.latent_dim, self.cfg.vicreg)
        self.joint: nn.Module = JointObjective(self.model, self.cfg, self.tokenizer.pad_id, self.target_encoder)
        self.objective = self.joint.realizer_objective      # used by evaluate()
        if self.ctx.enabled:
            self.joint = DistributedDataParallel(
                self.joint, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=True)

    def assert_planner_frozen(self) -> None:  # the planner trains in Stage 3
        return None

    def verify_planner_unchanged(self) -> None:
        return None

    def plan(self, batch: Batch) -> PredictedPlan:
        """Predicted plan from the context only (evaluation path; no gradient needed there)."""
        mode = batch.modes if self.cfg.training.stage2_mode_source == "reference" else None
        with torch.no_grad(), autocast(self.device, self.cfg.training.precision):
            return self.model.planner.plan(batch.context, batch.context_mask, mode=mode, strategy="sample").detach()

    def train_step(self, batch: Batch | Sequence[Batch]) -> dict[str, float]:
        self.model.train()
        micro = [batch] if isinstance(batch, Batch) else list(batch)
        if not micro:
            raise ValueError("train_step needs at least one batch")
        self.optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        loss_sum = 0.0
        for i, mb in enumerate(micro):
            mb = mb.to(self.device)
            with maybe_no_sync(self.joint, skip_sync=i < len(micro) - 1):
                loss, terms, plan, slot_mask = self.joint(mb)
                (loss / len(micro)).backward()
            self.monitor.update("context", masked_rows(plan.s_x.float(), mb.context_mask))
            self.monitor.update("plan", masked_rows(plan.z_target.float(), slot_mask))
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            loss_sum += float(loss.detach())
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.training.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        momentum = self.ema.update()
        self.step += 1
        n = len(micro)
        out = {k: v / n for k, v in sums.items()}
        out.update(loss=loss_sum / n, grad_norm=float(grad_norm), ema_momentum=momentum,
                   lr=self.optimizer.param_groups[0]["lr"])
        return out

    def payload(self) -> dict[str, Any]:
        out = super().payload()
        out.update(target_encoder=self.target_encoder.state_dict(), ema=self.ema.state_dict())
        return out

    def resume(self, path: str | Path) -> None:
        super().resume(path)
        state = load_checkpoint(path, map_location=self.device)
        self.target_encoder.load_state_dict(state["target_encoder"])
        self.ema.load_state_dict(state["ema"])
