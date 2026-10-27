# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
text_vpa_model.py -- the networks of one PR-JEPA generation, assembled (Fig. 1, Algorithm 1).

    TextVPAPlanner   passes 1-3: f_theta (context encoder), pi_phi^h (mode selector), g_phi (predictor)
    TextVPAModel     planner + realizer F_psi (FS-DFM or Logit-KL, owning the bridge) + plan readout

Sequential depth of a generation: f_theta -> pi_phi^h -> g_phi -> F_psi x K = K + 3 network evaluations.
The bridge projections and the plan readout read only Z_target; they run once, between pass 3 and the
first realizer pass, and are fused into pass 4 (Sec. 3.2). ``text_vpa.inference.sampler`` audits this
order at runtime with forward hooks, so every network is always called as ``module(...)``.

The target encoder f_theta_bar is not part of the model: it exists only in Stage 1 / Stage 3 training
(``text_vpa.engine``), and the realizer accepts only a ``PredictedPlan``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from text_vpa.models.encoders import ContextEncoder
from text_vpa.models.plan import PredictedPlan
from text_vpa.models.predictor import JEPAPredictor
from text_vpa.models.readout import PlanReadout
from text_vpa.models.realizers.fs_dfm import ConditionalFSDFMRealizer
from text_vpa.models.realizers.logit_kl import LogitKLRealizer
from text_vpa.models.selector import ModeSelector, choose_mode

if TYPE_CHECKING:
    from text_vpa.configs.schema import PlannerConfig, TextVPAConfig

__all__ = ["Realizer", "TextVPAModel", "TextVPAPlanner"]

Realizer = ConditionalFSDFMRealizer | LogitKLRealizer


class TextVPAPlanner(nn.Module):
    """Latent planner: f_theta, pi_phi^h and g_phi (Sec. 3.1)."""

    def __init__(self, context_encoder: ContextEncoder, selector: ModeSelector, predictor: JEPAPredictor) -> None:
        super().__init__()
        if not (context_encoder.latent_dim == selector.latent_dim == predictor.latent_dim):
            raise ValueError("f_theta, pi_phi^h and g_phi must share the latent width d")
        if selector.num_modes != predictor.num_modes:
            raise ValueError("pi_phi^h and g_phi must share the mode vocabulary U")
        self.context_encoder = context_encoder
        self.selector = selector
        self.predictor = predictor

    @classmethod
    def from_config(cls, cfg: PlannerConfig) -> TextVPAPlanner:
        return cls(ContextEncoder.from_config(cfg), ModeSelector.from_config(cfg), JEPAPredictor.from_config(cfg))

    @property
    def num_modes(self) -> int:
        return self.selector.num_modes

    def plan(
        self,
        context: Tensor,
        context_mask: Tensor | None = None,
        mode: Tensor | None = None,
        strategy: str = "sample",
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        run_selector: bool = True,
    ) -> PredictedPlan:
        """Passes 1-3 (Algorithm 1, lines 1-4).

        Args:
            context: ``[B, M_c]`` context token ids c.
            context_mask: ``[B, M_c]`` bool, True at real tokens (None: all real).
            mode: ``[B]`` int64 to force the mode u (teacher forcing in Stage 2), else u ~ pi_phi^h.
            strategy, temperature, generator: mode selection (``choose_mode``).
            run_selector: evaluate pi_phi^h even when ``mode`` is given (keeps the logits and the K + 3 chain).
        """
        if context_mask is None:
            context_mask = torch.ones(context.shape, dtype=torch.bool, device=context.device)
        s_x = self.context_encoder(context, context_mask)                              # pass 1
        logits = self.selector(s_x, context_mask) if (run_selector or mode is None) else None  # pass 2
        if mode is None:
            selection = choose_mode(logits, strategy, temperature, generator)          # type: ignore[arg-type]
            mode = selection.mode
            mode_input = selection.one_hot if strategy == "gumbel" else selection.mode
        else:
            if mode.shape != (context.shape[0],):
                raise ValueError(f"mode must have shape [{context.shape[0]}], got {tuple(mode.shape)}")
            mode_input = mode
        out = self.predictor(s_x, mode_input, context_mask=context_mask)              # pass 3
        return PredictedPlan(s_x, context_mask, mode, mode_input, logits, out.z_target, out.sigma_bar,
                             out.head_predictions)


class TextVPAModel(nn.Module):
    """Planner + realizer (+ bridge, inside the realizer) + plan readout."""

    def __init__(self, planner: TextVPAPlanner, realizer: nn.Module, readout: PlanReadout | None = None) -> None:
        super().__init__()
        if not isinstance(realizer, ConditionalFSDFMRealizer | LogitKLRealizer):
            raise TypeError("realizer must be a ConditionalFSDFMRealizer or a LogitKLRealizer")
        if realizer.bridge.latent_dim != planner.predictor.latent_dim:
            raise ValueError("the bridge's latent width must equal the planner's d")
        self.planner = planner
        self.realizer = realizer
        self.readout = readout

    @classmethod
    def from_config(cls, cfg: TextVPAConfig, mask_id: int | None = None) -> TextVPAModel:
        """Build every network from the config (vocabulary sizes must already match the tokenizer)."""
        if cfg.planner.vocab_size != cfg.realizer.vocab_size:
            raise ValueError("planner.vocab_size and realizer.vocab_size must both equal the tokenizer's size")
        planner = TextVPAPlanner.from_config(cfg.planner)
        if cfg.realizer.kind == "fs_dfm":
            realizer: nn.Module = ConditionalFSDFMRealizer.from_config(cfg, mask_id=mask_id)
        else:
            realizer = LogitKLRealizer.from_config(cfg)
        readout = None
        if cfg.readout.enabled:
            buckets = math.ceil(cfg.realizer.max_output_len / cfg.readout.length_bucket_size)
            readout = PlanReadout(cfg.planner.latent_dim, cfg.realizer.vocab_size, cfg.readout.hidden_dim,
                                  cfg.readout.max_text_tokens, buckets, cfg.readout.length_bucket_size)
        return cls(planner, realizer, readout)

    @property
    def realizer_kind(self) -> str:
        return self.realizer.kind  # type: ignore[return-value]

    def planner_parameters(self) -> list[nn.Parameter]:
        return list(self.planner.parameters())

    def realizer_parameters(self) -> list[nn.Parameter]:
        """F_psi, the bridge and the readout: what Stage 2 trains."""
        params = list(self.realizer.parameters())
        if self.readout is not None:
            params += list(self.readout.parameters())
        return params
