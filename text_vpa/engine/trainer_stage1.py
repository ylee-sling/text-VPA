# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
trainer_stage1.py -- Stage 1, planner pretraining (PR-JEPA Sec. 5.1).

Trains f_theta, pi_phi^h and g_phi with L_plan (Eq. 5):

    L_plan = lambda_invar (1/BM) sum ||s_hat_y - sg(s_y)||_F^2 + lambda_v [v(Z_x) + v(Z_hat)]
           + lambda_c [c(Z_x) + c(Z_hat)] + lambda_u L_CE(pi_phi^h)

with s_y = f_theta_bar(y) from the EMA target encoder (no gradient; theta_bar <- m_k theta_bar + (1 - m_k)
theta after every optimizer step, m_k from m0 = 0.996 to 1, Eq. 4) and g_phi fed the reference mode u.
The predictor runs on all M slots, as at inference; the prediction term and the plan's VICReg rows use the
slots the reference actually has.

Collapse guard: the covariances of the context latents and of the plan rows are accumulated over a window
of ``training.collapse_check_every`` steps; v(Z), c(Z), the stable rank srank(C) = (tr C)^2 / ||C||_F^2 and
the eigenvalue range are logged. If either stable rank falls below ``vicreg.srank_abort_fraction`` d
(d/2 by default), training stops and the checkpoint is written with ``rejected = True``; Stage 2 refuses
rejected checkpoints. Under DDP every rank checks its own shard and the decision is shared.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from text_vpa.configs.schema import TextVPAConfig
from text_vpa.data.dataset import Batch, epoch_batches
from text_vpa.data.tokenization import TextVPATokenizer
from text_vpa.engine.losses import (
    LossOutput,
    PlanLoss,
    StableRankMonitor,
    masked_rows,
)
from text_vpa.engine.runtime import (
    DistributedContext,
    MetricLogger,
    autocast,
    build_optimizer,
    load_checkpoint,
    rng_state,
    save_checkpoint,
    set_rng_state,
    warmup_cosine,
)
from text_vpa.models.ema import EMATargetEncoder
from text_vpa.models.encoders import TargetEncoder
from text_vpa.models.plan import PredictedPlan
from text_vpa.models.text_vpa_model import TextVPAPlanner

__all__ = ["PlannerObjective", "Stage1Result", "Stage1Trainer"]

logger = logging.getLogger(__name__)


class PlannerObjective(nn.Module):
    """One Stage-1 forward: planner passes 1-3 with the reference mode, then L_plan (in float32)."""

    def __init__(self, planner: TextVPAPlanner, loss: PlanLoss, precision: str = "fp32") -> None:
        super().__init__()
        self.planner = planner
        self.loss = loss
        self.precision = precision

    def forward(self, batch: Batch, s_y: Tensor, target_slot_mask: Tensor) -> tuple[Tensor, dict[str, Tensor],
                                                                                  PredictedPlan]:
        with autocast(batch.context.device, self.precision):
            plan = self.planner.plan(batch.context, batch.context_mask, mode=batch.modes, run_selector=True)
        out: LossOutput = self.loss(
            predictions=plan.head_predictions.float(), target=s_y.float(), context=plan.s_x.float(),
            plan=plan.z_target.float(), mode_logits=plan.mode_logits.float(),  # type: ignore[union-attr]
            mode_targets=batch.modes, context_mask=batch.context_mask, slot_mask=target_slot_mask,
        )
        return out.total, out.terms, plan


@dataclass
class Stage1Result:
    steps: int
    rejected: bool
    reason: str | None
    checkpoint: Path | None
    collapse_report: dict[str, float] = field(default_factory=dict)


class Stage1Trainer:
    """Stage 1 training loop.

    Args:
        cfg: full config (``training.planner`` holds the stage's optimiser settings).
        planner: the planner to train (its ``context_encoder`` seeds the target encoder).
        tokenizer: stored in checkpoints.
        train_loader: iterable of ``Batch``.
        output_dir: checkpoints and metrics.
        ctx: distributed context (single process by default).
        metric_logger: optional ``MetricLogger`` (created in ``output_dir`` otherwise).
    """

    stage = 1

    def __init__(self, cfg: TextVPAConfig, planner: TextVPAPlanner, tokenizer: TextVPATokenizer,
                 train_loader: Iterable[Batch], output_dir: str | Path, ctx: DistributedContext | None = None,
                 metric_logger: MetricLogger | None = None) -> None:
        self.cfg = cfg
        self.ctx = ctx or DistributedContext(device=torch.device("cpu"))
        self.device = self.ctx.device
        self.tokenizer = tokenizer
        self.loader = train_loader
        self.output_dir = Path(output_dir)
        self.planner = planner.to(self.device)
        self.target_encoder = TargetEncoder.from_online(self.planner.context_encoder)
        self.objective: nn.Module = PlannerObjective(self.planner, PlanLoss.from_config(cfg.vicreg),
                                                     cfg.training.precision)
        if self.ctx.enabled:
            self.objective = DistributedDataParallel(
                self.objective, device_ids=[self.device.index] if self.device.type == "cuda" else None)
        stage = cfg.training.planner
        self.max_steps = stage.max_steps
        self.optimizer = build_optimizer([("planner", self.planner, stage.learning_rate)], stage.weight_decay)
        self.scheduler = warmup_cosine(self.optimizer, stage.warmup_steps, stage.max_steps)
        self.ema = EMATargetEncoder.from_config(self.planner.context_encoder, self.target_encoder, cfg.vicreg,
                                                total_steps=stage.max_steps)
        self.monitor = StableRankMonitor.from_config(cfg.planner.latent_dim, cfg.vicreg)
        self.logger = metric_logger or MetricLogger(self.output_dir, enabled=self.ctx.is_main)
        self.step = 0
        self.epoch = 0                      # data position, so a resumed run continues the same stream
        self.batch_in_epoch = 0
        self.last_report: dict[str, float] = {}

    # -----------------------------------------------------------------------------------------
    def train_step(self, batch: Batch) -> dict[str, float]:
        self.planner.train()
        batch = batch.to(self.device)
        target = self.target_encoder(batch.structure, batch.slot_index, self.cfg.planner.num_slots,
                                     batch.structure_mask)                          # sg(f_theta_bar(y))
        if int(target.slot_mask.sum()) < 2:
            raise ValueError("a Stage-1 batch needs at least two real slots (VICReg statistics)")
        loss, terms, plan = self.objective(batch, target.s_y, target.slot_mask)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.planner.parameters(), self.cfg.training.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        momentum = self.ema.update()                                                # Eq. 4
        self.monitor.update("context", masked_rows(plan.s_x.detach().float(), batch.context_mask))
        self.monitor.update("plan", masked_rows(plan.z_target.detach().float(), target.slot_mask))
        self.step += 1
        out = {k: float(v) for k, v in terms.items()}
        out.update(loss=float(loss.detach()), grad_norm=float(grad_norm), ema_momentum=momentum,
                   lr=self.optimizer.param_groups[0]["lr"], sigma_bar=float(plan.sigma_bar.detach().mean()))
        return out

    def check_collapse(self) -> tuple[bool, str | None]:
        """Window check (Sec. 5.1): log v, c, srank and the eigenvalue range of both streams; abort (after the
        grace period) if a stable rank is below ``srank_abort_fraction`` d. Returns (collapsed on any rank, reason)."""
        collapsed, reason = False, None
        if self.monitor.min_rows > self.cfg.planner.latent_dim:
            report = self.monitor.summarize(compute_eigenvalues=True)
            self.monitor.reset()
            self.last_report = {}
            for name, diag in report.items():
                self.last_report.update(diag.as_log_dict(prefix=f"{name}/"))
                if diag.eigenvalues is not None:
                    self.last_report[f"{name}/lambda_max"] = float(diag.eigenvalues[-1])
            self.logger.log(self.step, self.last_report, prefix="collapse/")
            low = self.monitor.below_threshold(report)
            if low:
                message = (f"stable rank below {self.monitor.threshold:g} = {self.monitor.abort_fraction:g} d "
                           f"(d = {self.monitor.dim}): " + ", ".join(f"{k} {v:.2f}" for k, v in low.items()))
                if self.step >= self.cfg.training.collapse_grace_steps:
                    collapsed, reason = True, message
                else:
                    logger.info("step %d, inside the grace period (no abort): %s", self.step, message)
        else:
            logger.info("stable-rank window has %d rows <= d; extending it", self.monitor.min_rows)
        if self.ctx.any_true(collapsed):
            return True, reason or "representation collapse on another rank"
        return False, None

    def train(self, max_steps: int | None = None) -> Stage1Result:
        """Run until ``max_steps`` (default ``training.planner.max_steps``) or collapse."""
        limit = self.max_steps if max_steps is None else max_steps
        tcfg = self.cfg.training
        start = time.time()
        while self.step < limit:
            consumed = 0
            for batch in epoch_batches(self.loader, self.epoch, self.batch_in_epoch):
                consumed += 1
                metrics = self.train_step(batch)
                self.batch_in_epoch += 1
                if self.step % tcfg.log_every == 0 or self.step == 1:
                    metrics["steps_per_s"] = self.step / max(time.time() - start, 1e-9)
                    self.logger.log(self.step, metrics, prefix="stage1/")
                if self.step % tcfg.collapse_check_every == 0:
                    collapsed, reason = self.check_collapse()
                    if collapsed:
                        path = self.save("stage1_rejected.pt", rejected=True, reason=reason)
                        self.logger.log(self.step, {"rejected": 1.0}, prefix="stage1/")
                        return Stage1Result(self.step, True, reason, path, self.last_report)
                if self.ctx.is_main and self.step % tcfg.save_every == 0:
                    self.save("stage1_last.pt")
                if self.step >= limit:
                    break
            else:
                if consumed == 0 and self.batch_in_epoch == 0:
                    raise RuntimeError("the training loader yields no batch")
                self.epoch, self.batch_in_epoch = self.epoch + 1, 0
        if self.step % tcfg.collapse_check_every != 0:          # the window was not just checked
            collapsed, reason = self.check_collapse()
            if collapsed:
                path = self.save("stage1_rejected.pt", rejected=True, reason=reason)
                return Stage1Result(self.step, True, reason, path, self.last_report)
        path = self.save("stage1_final.pt")
        return Stage1Result(self.step, False, None, path, self.last_report)

    # -----------------------------------------------------------------------------------------
    def payload(self, rejected: bool = False, reason: str | None = None) -> dict[str, Any]:
        return {
            "stage": 1, "step": self.step, "config": self.cfg.model_dump(mode="json"),
            "tokenizer": self.tokenizer.spec(), "planner": self.planner.state_dict(),
            "target_encoder": self.target_encoder.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "ema": self.ema.state_dict(), "rng": rng_state(),
            "metrics": dict(self.last_report), "rejected": rejected, "rejection_reason": reason,
            "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
        }

    def save(self, name: str, rejected: bool = False, reason: str | None = None) -> Path | None:
        if not self.ctx.is_main:
            return None
        return save_checkpoint(self.output_dir / name, self.payload(rejected, reason))

    def resume(self, path: str | Path) -> None:
        state = load_checkpoint(path, map_location=self.device)
        if state.get("stage") != 1:
            raise ValueError(f"{path} is not a Stage-1 checkpoint")
        self.planner.load_state_dict(state["planner"])
        self.target_encoder.load_state_dict(state["target_encoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.ema.load_state_dict(state["ema"])
        set_rng_state(state["rng"])
        self.step = int(state["step"])
        self.epoch = int(state.get("epoch", 0))
        self.batch_in_epoch = int(state.get("batch_in_epoch", 0))
