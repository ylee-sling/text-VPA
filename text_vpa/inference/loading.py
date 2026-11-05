# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
loading.py -- rebuild a trained text-VPA model, its tokenizer and a ready sampler from a checkpoint.

Checkpoints are self-contained: they store the validated config, the tokenizer spec (a trained BPE embeds
its JSON) and every network's weights, so ``generate.py`` needs nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from text_vpa.configs.schema import TextVPAConfig
from text_vpa.data.ast_parser import TreeSitterASTExtractor
from text_vpa.data.tokenization import TextVPATokenizer, tokenizer_from_spec
from text_vpa.engine.runtime import load_checkpoint
from text_vpa.inference.energy_terms import ConstraintEnergyEvaluator
from text_vpa.inference.sampler import PRJEPASampler
from text_vpa.models.text_vpa_model import TextVPAModel

__all__ = ["LoadedModel", "build_energy", "build_sampler", "load_model", "override_section"]


@dataclass
class LoadedModel:
    cfg: TextVPAConfig
    tokenizer: TextVPATokenizer
    model: TextVPAModel
    stage: int
    step: int
    calibration: dict | None = None      # the inventory threshold calibrated at the checkpoint's evaluation

    def inventory_threshold(self) -> tuple[float, str]:
        """The readout's inventory threshold and where it comes from: the checkpoint's calibration (when
        ``readout.inventory_recall`` is set and the checkpoint holds one) or ``readout.inventory_threshold``."""
        cal = self.calibration or {}
        if self.cfg.readout.inventory_recall is not None and cal.get("inventory_threshold"):
            return float(cal["inventory_threshold"]), f"calibrated for recall {cal.get('target_recall')}"
        return self.cfg.readout.inventory_threshold, "config"


def load_model(path: str | Path, device: torch.device | str = "cpu") -> LoadedModel:
    """Load a Stage-2 or Stage-3 checkpoint (a Stage-1 checkpoint has no realizer)."""
    state = load_checkpoint(path, map_location="cpu")
    if state.get("rejected"):
        raise ValueError(f"{path} was rejected during training ({state.get('rejection_reason')})")
    if "realizer" not in state:
        raise ValueError(f"{path} holds no realizer (stage {state.get('stage')}); train Stage 2 first")
    cfg = TextVPAConfig.model_validate(state["config"])
    tokenizer = tokenizer_from_spec(state["tokenizer"])
    if tokenizer.vocab_size != cfg.realizer.vocab_size:
        raise ValueError("the checkpoint's tokenizer does not match its config")
    model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)
    model.planner.load_state_dict(state["planner"])
    model.realizer.load_state_dict(state["realizer"])
    if model.readout is not None:
        if "readout" not in state:
            raise ValueError("the config enables the readout but the checkpoint has none")
        model.readout.load_state_dict(state["readout"])
    model.to(device).eval()
    return LoadedModel(cfg, tokenizer, model, int(state["stage"]), int(state["step"]), state.get("calibration"))


def override_section(cfg: TextVPAConfig, section: str, **updates: object) -> TextVPAConfig:
    """``cfg`` with the fields ``updates`` (those not None) of one section replaced, validated again."""
    updates = {k: v for k, v in updates.items() if v is not None}
    if not updates:
        return cfg
    raw = cfg.model_dump()
    raw[section] = {**raw[section], **updates}
    return TextVPAConfig.model_validate(raw)


def build_energy(cfg: TextVPAConfig, tokenizer: TextVPATokenizer) -> ConstraintEnergyEvaluator | None:
    """The configured constraint energies (``EnergyConfig``), or None when disabled."""
    if not cfg.energy.enabled:
        return None
    evaluator = ConstraintEnergyEvaluator(tokenizer, terms=[], language=cfg.data.language)
    e = cfg.energy
    evaluator.terms = evaluator.default_terms(e.identifier_weight, e.grammar_weight, e.arity_weight, e.type_weight)
    if not all(t.weight > 0 and (math.isinf(t.weight) or math.isfinite(t.weight)) for t in evaluator.terms):
        raise ValueError("invalid energy weights")
    return evaluator


def build_sampler(loaded: LoadedModel, energy: ConstraintEnergyEvaluator | None = None,
                  use_energy: bool = True) -> PRJEPASampler:
    """``PRJEPASampler`` with the checkpoint's realizer and sampler settings."""
    cfg = loaded.cfg
    if use_energy and energy is None:
        energy = build_energy(cfg, loaded.tokenizer)
    plan_source = "readout" if cfg.energy.plan_source == "readout" else "none"
    return PRJEPASampler(loaded.model, loaded.tokenizer, cfg.realizer, cfg.sampler, energy if use_energy else None,
                         TreeSitterASTExtractor(cfg.data.language), loaded.inventory_threshold()[0], plan_source)
