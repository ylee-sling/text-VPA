# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
schema.py -- strongly typed configuration schema for text-VPA (PR-JEPA, Secs. 3 and 5.1).

Every section is an immutable Pydantic model with ``extra="forbid"``: an unknown key, a value of the
wrong type, or a violated constraint raises ``pydantic.ValidationError`` when the config is built.
The constraints are the ones the paper imposes:

    * realizer step budgets 1 <= K <= 8 and the inference K among the trained budgets (Algorithm 1;
      a generation costs K + 3 sequential network evaluations),
    * 0 < eps < gamma^2 for the variance hinge (Eq. 6; Prop. 3 wants eps << gamma^2),
    * lambda_v > 0 (Theorem 1 needs the variance hinge), lambda_c, lambda_u >= 0,
    * c_max in (0, 1] and T > 0 for the energy-tilted kernel (Eq. 17, Prop. 4(a)),
    * n_ens >= 2, so the ensemble spread sigma-bar of Eq. 8 is defined,
    * EMA momentum 0 <= m0 <= m_end <= 1 (Eq. 4).

Values the paper does not fix (all widths, |U|, S, learning rates, lambda_u) are placeholders.
The sections are composed from YAML by Hydra in ``text_vpa.configs.loader``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MAX_REALIZER_STEPS",
    "PLANNER_PASSES",
    "BridgeConfig",
    "DataConfig",
    "EncoderConfig",
    "EnergyConfig",
    "JointStageConfig",
    "PlannerConfig",
    "PredictorConfig",
    "ReadoutConfig",
    "RealizerConfig",
    "SamplerConfig",
    "SelectorConfig",
    "StageConfig",
    "TextVPAConfig",
    "TrainingConfig",
    "VICRegConfig",
]

#: Upper bound on the realizer step budget K (few-step regime, Sec. 1.3).
MAX_REALIZER_STEPS: int = 8
#: Sequential planner evaluations per generation: f_theta, pi_phi^h, g_phi (Algorithm 1, lines 1-3).
PLANNER_PASSES: int = 3


class _Section(BaseModel):
    """Base for all sections: immutable, unknown keys rejected, defaults validated."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


def _divisible(width: int, heads: int, what: str) -> None:
    if width % heads != 0:
        raise ValueError(f"{what}: width {width} must be divisible by num_heads {heads}")


# ---------------------------------------------------------------------------------------------
# Planner: f_theta / f_theta_bar, pi_phi^h, g_phi (Sec. 3.1)
# ---------------------------------------------------------------------------------------------
class EncoderConfig(_Section):
    """Context encoder f_theta (pass 1) and its EMA copy f_theta_bar (training only)."""

    dim: int = Field(512, ge=1, description="Internal transformer width of f_theta (context encoder dim).")
    num_layers: int = Field(6, ge=1)
    num_heads: int = Field(8, ge=1)
    mlp_ratio: float = Field(4.0, gt=0.0)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_heads(self) -> EncoderConfig:
        _divisible(self.dim, self.num_heads, "planner.encoder")
        return self


class SelectorConfig(_Section):
    """Mode selector pi_phi^h(u | s_x) (pass 2): masked mean pool of s_x, then an MLP."""

    hidden_dim: int = Field(512, ge=1)
    num_layers: int = Field(2, ge=1)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)


class PredictorConfig(_Section):
    """JEPA predictor g_phi (pass 3): shared trunk plus n_ens heads (Eqs. 3, 8).

    ``head_hidden_dim = None`` gives linear heads; an integer gives two-layer GELU MLP heads.
    """

    num_layers: int = Field(4, ge=1)
    num_heads: int = Field(8, ge=1)
    mlp_ratio: float = Field(4.0, gt=0.0)
    n_ens: int = Field(5, ge=2, description="Ensemble heads; sigma-bar (Eq. 8) needs at least two.")
    head_hidden_dim: int | None = Field(None, ge=1)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)


class PlannerConfig(_Section):
    """Latent planner (passes 1-3). ``latent_dim`` is d, the width of s_x, s_y and Z_target."""

    vocab_size: int = Field(32768, ge=2, description="Tokenizer of the context c and structural view y.")
    latent_dim: int = Field(256, ge=1, description="d")
    num_slots: int = Field(256, ge=1, description="M, number of learnable slot queries Delta.")
    num_modes: int = Field(64, ge=2, description="|U|; one mode cannot resolve multimodality (Prop. 1(b)).")
    max_context_len: int = Field(1024, ge=1, description="Upper bound on M_c.")
    max_structure_len: int = Field(2048, ge=1, description="Upper bound on the token length of y.")
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)

    @model_validator(mode="after")
    def _check_widths(self) -> PlannerConfig:
        _divisible(self.latent_dim, self.predictor.num_heads, "planner.predictor")
        return self

    @property
    def max_positions(self) -> int:
        """Position table size shared by f_theta and f_theta_bar (they are the same network)."""
        return max(self.max_context_len, self.max_structure_len)


# ---------------------------------------------------------------------------------------------
# Bridge (Sec. 3.2)
# ---------------------------------------------------------------------------------------------
class BridgeConfig(_Section):
    """Cross-attention bridge. Its conditioning scale 1/gamma is ``VICRegConfig.gamma``."""

    spectral_norm: bool = Field(True, description="Spectral normalization of W_K and W_V.")
    power_iterations: int = Field(1, ge=1, description="Power iterations per training forward.")
    plan_dropout: float = Field(0.1, ge=0.0, lt=1.0, description="P(replace Z~ by the learned null token).")


# ---------------------------------------------------------------------------------------------
# Realizer (Secs. 3.3-3.4, Algorithm 1)
# ---------------------------------------------------------------------------------------------
class RealizerConfig(_Section):
    """Parallel discrete realizer F_psi.

    ``step_budgets`` are the budgets K the realizer is trained for (FS-DFM conditions on the step
    size h = 1/K); ``num_steps`` is the K used at inference and must be one of them. The ``sqrt``
    schedule is valid only with the exact jump kernel of Eq. 15, not with Euler steps.
    """

    kind: Literal["fs_dfm", "logit_kl"] = "fs_dfm"
    vocab_size: int = Field(32768, ge=2, description="S = |V|, output vocabulary.")
    hidden_dim: int = Field(2048, ge=1, description="d_r (= d_k of the bridge keys and values).")
    num_layers: int = Field(24, ge=1, description="L_r")
    num_heads: int = Field(16, ge=1)
    mlp_ratio: float = Field(4.0, gt=0.0)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    max_output_len: int = Field(1024, ge=1, description="Upper bound on N.")
    step_budgets: tuple[int, ...] = (2, 4, 8)
    num_steps: int = Field(8, description="K at inference.")
    schedule: Literal["linear", "sqrt", "cosine"] = "linear"
    source: Literal["uniform", "mask"] = "uniform"
    temperature: float = Field(1.0, gt=0.0, description="T of the energy tilt (Eq. 17).")
    c_max: float = Field(1.0, gt=0.0, le=1.0, description="Safety clip (Eq. 17, Prop. 4(a)).")
    guidance_scale: float = Field(0.0, ge=0.0, description="Predictor-free guidance omega; 0 disables it.")
    logit_smoothing: float = Field(0.05, gt=0.0, lt=1.0, description="epsilon of Eq. 18 (Logit-KL).")
    logit_source_std: float = Field(1.0, gt=0.0, description="s, std of l0 (Logit-KL).")

    @model_validator(mode="after")
    def _check_budgets(self) -> RealizerConfig:
        budgets = self.step_budgets
        if not budgets:
            raise ValueError("realizer.step_budgets must not be empty")
        if any(not 1 <= k <= MAX_REALIZER_STEPS for k in budgets):
            raise ValueError(f"every step budget K must satisfy 1 <= K <= {MAX_REALIZER_STEPS}, got {list(budgets)}")
        if list(budgets) != sorted(set(budgets)):
            raise ValueError(f"realizer.step_budgets must be strictly increasing, got {list(budgets)}")
        if self.num_steps not in budgets:
            raise ValueError(f"realizer.num_steps={self.num_steps} must be one of the trained budgets {list(budgets)}")
        _divisible(self.hidden_dim, self.num_heads, "realizer")
        return self

    @property
    def sequential_depth(self) -> int:
        """K + 3: planner passes 1-3 plus K realizer passes (Algorithm 1)."""
        return self.num_steps + PLANNER_PASSES


# ---------------------------------------------------------------------------------------------
# Planner objective L_plan (Eq. 5) and the target-encoder EMA (Eq. 4)
# ---------------------------------------------------------------------------------------------
class VICRegConfig(_Section):
    """Weights and constants of L_plan (Eq. 5), the Stage-1 collapse guard, and the EMA (Eq. 4).

    ``lambda_invar`` multiplies the prediction term. The paper states VICReg's published weights 25 : 25 : 1
    but writes the prediction term of Eq. 5 as a squared Frobenius norm per slot, i.e. summed over the d
    features, where VICReg averages its invariance MSE over them. Applying 25 to the summed form makes the
    prediction term d times stronger relative to the variance hinge than in VICReg (256x at d = 256), and the
    plan's variance then collapses towards the slowly moving EMA targets (observed in Stage-1 runs). The
    default ``invariance_normalization = per_dim`` therefore averages over d, which reproduces VICReg's
    balance; ``frobenius`` with ``lambda_invar = 1`` is Eq. 5 verbatim. ``lambda_u`` has no value in the paper.
    """

    lambda_invar: float = Field(25.0, gt=0.0)
    lambda_v: float = Field(25.0, gt=0.0)
    lambda_c: float = Field(1.0, ge=0.0)
    lambda_u: float = Field(1.0, ge=0.0)
    gamma: float = Field(1.0, gt=0.0, description="Target standard deviation (margin) of the hinge.")
    eps: float = Field(1.0e-4, gt=0.0)
    head_reduction: Literal["mean", "sum"] = "mean"
    invariance_normalization: Literal["per_dim", "frobenius"] = Field(
        "per_dim", description="per_dim: prediction error averaged over the d features (VICReg's MSE; keeps the "
                               "25 : 25 : 1 balance at every d). frobenius: summed over d, Eq. 5 verbatim.")
    srank_abort_fraction: float = Field(0.5, gt=0.0, le=1.0, description="Stage 1 aborts if srank(C) < this * d.")
    ema_momentum_start: float = Field(0.996, ge=0.0, le=1.0, description="m0 of Eq. 4")
    ema_momentum_end: float = Field(1.0, ge=0.0, le=1.0)
    ema_schedule: Literal["linear", "cosine"] = "linear"

    @model_validator(mode="after")
    def _check_constants(self) -> VICRegConfig:
        if not self.eps < self.gamma ** 2:
            raise ValueError("vicreg.eps must satisfy 0 < eps < gamma^2 (Eq. 6; Prop. 3 wants eps << gamma^2)")
        if not self.ema_momentum_start <= self.ema_momentum_end:
            raise ValueError("EMA momentum must satisfy m0 <= m_end (Eq. 4)")
        return self


# ---------------------------------------------------------------------------------------------
# Training (Sec. 5.1)
# ---------------------------------------------------------------------------------------------
class StageConfig(_Section):
    """Optimizer settings of one training stage."""

    learning_rate: float = Field(gt=0.0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0.0)
    max_steps: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_warmup(self) -> StageConfig:
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must not exceed max_steps")
        return self


class JointStageConfig(StageConfig):
    """Stage 3: the planner trains at ``planner_lr_scale`` times the stage learning rate."""

    planner_lr_scale: float = Field(0.1, gt=0.0, le=1.0)


def _planner_stage() -> StageConfig:
    return StageConfig(learning_rate=3.0e-4, warmup_steps=2_000, weight_decay=0.05, max_steps=100_000)


def _realizer_stage() -> StageConfig:
    return StageConfig(learning_rate=3.0e-4, warmup_steps=5_000, weight_decay=0.1, max_steps=300_000)


def _joint_stage() -> JointStageConfig:
    return JointStageConfig(learning_rate=1.0e-4, warmup_steps=1_000, weight_decay=0.05, max_steps=50_000,
                            planner_lr_scale=0.1)


class TrainingConfig(_Section):
    """Staged training protocol (Sec. 5.1).

    ``planner``: Stage 1, L_plan, target encoder EMA-only. ``realizer``: Stage 2, planner frozen and
    the realizer conditioned on the predicted plan only, never on f_theta_bar(y). ``joint``: optional
    Stage 3 with the planner unfrozen at ``joint.planner_lr_scale`` times the learning rate and the
    VICReg penalties kept on. ``collapse_check_every`` is the window, in optimizer steps, over which
    the stable-rank guard accumulates its covariance (the paper checks once per epoch).
    """

    stage: Literal["planner", "realizer", "joint"] = "planner"
    batch_size: int = Field(64, ge=2, description="B; VICReg statistics need n >= 2 rows.")
    planner: StageConfig = Field(default_factory=_planner_stage)
    realizer: StageConfig = Field(default_factory=_realizer_stage)
    joint: JointStageConfig = Field(default_factory=_joint_stage)
    grad_clip_norm: float = Field(1.0, gt=0.0)
    collapse_check_every: int = Field(1_000, ge=1)
    collapse_grace_steps: int = Field(
        1_000, ge=0, description="Stable rank is logged but cannot abort before this step (the paper checks after "
                                 "each epoch, i.e. during training, not at random initialisation).")
    seed: int = 0
    device: str = Field("auto", description="auto | cpu | cuda[:i] (also ROCm) | mps")
    precision: Literal["fp32", "bf16"] = "fp32"
    stage2_mode_source: Literal["reference", "sampled"] = Field(
        "reference", description="Mode u fed to g_phi in Stage 2: the reference mode (teacher forcing) or a draw "
                                 "from pi_phi^h. The plan itself is always predicted from the context.")
    readout_weight: float = Field(1.0, ge=0.0, description="Weight of the plan-readout loss in Stages 2-3.")
    realizer_loss_reduction: Literal["mean", "sum"] = "mean"
    log_every: int = Field(10, ge=1)
    save_every: int = Field(1_000, ge=1)

    @property
    def active_stage(self) -> StageConfig:
        return {"planner": self.planner, "realizer": self.realizer, "joint": self.joint}[self.stage]


# ---------------------------------------------------------------------------------------------
# Data, plan readout, energies, sampler
# ---------------------------------------------------------------------------------------------
MODE_SCHEME_SIZES: dict[str, int] = {"control": 6, "control_library": 30}


class DataConfig(_Section):
    """Tokenizer, language and length limits of the data pipeline (Sec. 5.1, Table 1)."""

    language: Literal["python", "cpp"] = "python"
    tokenizer: Literal["byte", "bpe", "hf"] = "bpe"
    tokenizer_name: str | None = Field(None, description="Hugging Face name or path when tokenizer = hf.")
    bpe_vocab_size: int = Field(4096, ge=265, description="Size of a BPE trained on the training corpus.")
    mode_scheme: Literal["control", "control_library"] = "control_library"
    max_context_tokens: int = Field(512, ge=1)
    max_output_tokens: int = Field(512, ge=1)
    max_structure_tokens: int = Field(1024, ge=1)
    bucket_size: int = Field(32, ge=1, description="Canvas lengths are rounded up to a multiple of this.")
    time_sampling: Literal["grid", "uniform"] = "grid"
    num_workers: int = Field(0, ge=0)

    @property
    def num_modes(self) -> int:
        return MODE_SCHEME_SIZES[self.mode_scheme]


class ReadoutConfig(_Section):
    """Plan readout (slot decoder) on Z_target; trained in Stages 2-3 on the predicted plan."""

    enabled: bool = True
    hidden_dim: int = Field(256, ge=1)
    max_text_tokens: int = Field(48, ge=1)
    length_bucket_size: int = Field(32, ge=1)
    inventory_threshold: float = Field(0.5, gt=0.0, lt=1.0)
    kind_weight: float = Field(1.0, ge=0.0)
    text_weight: float = Field(1.0, ge=0.0)
    inventory_weight: float = Field(1.0, ge=0.0)
    length_weight: float = Field(0.1, ge=0.0)


class EnergyConfig(_Section):
    """Constraint energies of the tilted kernel (Sec. 3.3). A weight of .inf makes a term hard."""

    enabled: bool = True
    plan_source: Literal["readout", "reference", "none"] = Field(
        "readout", description="Symbolic plan for the energies: decoded from Z_target, oracle from the reference "
                               "(evaluation only), or none.")
    identifier_weight: float = Field(math.inf, gt=0.0)
    grammar_weight: float = Field(math.inf, gt=0.0)
    arity_weight: float = Field(5.0, gt=0.0)
    type_weight: float = Field(5.0, gt=0.0)


class SamplerConfig(_Section):
    """Algorithm 1 options. K, T, c_max, the schedule and guidance live in ``RealizerConfig``."""

    mode_strategy: Literal["sample", "greedy", "gumbel"] = "sample"
    mode_temperature: float = Field(1.0, gt=0.0)
    restrict_source: bool = Field(True, description="Draw w_0 from the plan-admissible tokens (Prop. 4(d) premise).")
    final_repair: bool = Field(True, description="Network-free left-to-right repair of hard violations and of "
                                                 "positions that never left their source token (stalls).")
    check_admissibility: bool = Field(True, description="Runtime assertion of Prop. 4(d) at every step.")
    audit_depth: bool = Field(True, description="Runtime assertion that a generation costs exactly K + 3 passes.")
    repair_aware: bool = False
    default_length: int | None = Field(None, ge=1, description="Canvas length N; None: the readout's length head.")


# ---------------------------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------------------------
class TextVPAConfig(_Section):
    """Root configuration."""

    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    realizer: RealizerConfig = Field(default_factory=RealizerConfig)
    vicreg: VICRegConfig = Field(default_factory=VICRegConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    readout: ReadoutConfig = Field(default_factory=ReadoutConfig)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)

    @model_validator(mode="after")
    def _check_cross_section(self) -> TextVPAConfig:
        if self.data.max_context_tokens > self.planner.max_context_len:
            raise ValueError("data.max_context_tokens must not exceed planner.max_context_len")
        if self.data.max_structure_tokens > self.planner.max_structure_len:
            raise ValueError("data.max_structure_tokens must not exceed planner.max_structure_len")
        if self.data.max_output_tokens > self.realizer.max_output_len:
            raise ValueError("data.max_output_tokens must not exceed realizer.max_output_len")
        if self.planner.num_modes < self.data.num_modes:
            raise ValueError(f"planner.num_modes={self.planner.num_modes} is smaller than the "
                             f"{self.data.mode_scheme!r} mode scheme ({self.data.num_modes} modes)")
        if self.sampler.default_length is not None and self.sampler.default_length > self.realizer.max_output_len:
            raise ValueError("sampler.default_length must not exceed realizer.max_output_len")
        return self

    def with_vocab_size(self, vocab_size: int) -> TextVPAConfig:
        """Copy with planner and realizer vocabularies set to the tokenizer's size S (validated)."""
        data = self.model_dump()
        data["planner"]["vocab_size"] = vocab_size
        data["realizer"]["vocab_size"] = vocab_size
        return TextVPAConfig.model_validate(data)

    def to_yaml(self) -> str:
        """Serialise, e.g. to store next to a checkpoint. ``from_yaml`` reads it back."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TextVPAConfig:
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))
