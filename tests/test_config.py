# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of the Hydra-composed Pydantic configuration (``text_vpa.configs``).

The expected values come from the paper and the project specification, not from the schema:

    * VICReg weights 25 : 25 : 1 with gamma = 1 and eps = 1e-4 (Sec. 3.1, after Eq. 7); eps < gamma^2 (Prop. 3)
    * EMA momentum m0 = 0.996 rising to 1 (Eq. 4, Sec. 5.1)
    * K <= 8 realizer steps, trained budgets {2, 4, 8}, sequential depth K + 3 (Algorithm 1, Sec. 3.3)
    * c_max in (0, 1] and T > 0 for the energy-tilted kernel (Algorithm 1 line 11, Prop. 4(a))
    * n_ens heads with spread sigma-bar (Eq. 8; default 5), L_r = 24 and d_k = 2048 (Sec. 3.2)
    * Stage 1 aborts below srank d/2, plan dropout 0.1, Stage 3 at one tenth of the learning rate (Sec. 5.1)
"""

from __future__ import annotations

import pydantic
import pytest
import yaml
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from text_vpa.configs import (
    MAX_REALIZER_STEPS,
    PLANNER_PASSES,
    JointStageConfig,
    PredictorConfig,
    RealizerConfig,
    StageConfig,
    TextVPAConfig,
    TrainingConfig,
    VICRegConfig,
    load_config,
    validate_config,
)


def raises_validation_error(overrides: list[str]) -> pydantic.ValidationError:
    with pytest.raises(pydantic.ValidationError) as info:
        load_config(overrides)
    return info.value


def has_error_at(error: pydantic.ValidationError, loc: tuple, error_type: str | None = None) -> bool:
    """True if some error is located at ``loc`` (or below it) and, if given, has type ``error_type``."""
    return any(tuple(e["loc"][:len(loc)]) == loc and (error_type is None or e["type"] == error_type)
               for e in error.errors())


# ---------------------------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------------------------
def test_yaml_defaults_equal_schema_defaults():
    """The packaged YAML groups and the Pydantic field defaults describe the same configuration."""
    assert load_config() == TextVPAConfig()


@pytest.mark.parametrize("source", ["yaml", "schema"])
def test_documented_defaults(source):
    cfg = load_config() if source == "yaml" else TextVPAConfig()
    planner, realizer, vicreg, training = cfg.planner, cfg.realizer, cfg.vicreg, cfg.training
    assert (planner.latent_dim, planner.num_slots, planner.predictor.n_ens) == (256, 256, 5)
    # Sec. 3.1: VICReg's 25 : 25 : 1 for invariance, variance and covariance, gamma = 1, eps = 1e-4.
    assert (vicreg.lambda_invar, vicreg.lambda_v, vicreg.lambda_c) == (25.0, 25.0, 1.0)
    assert (vicreg.gamma, vicreg.eps) == (1.0, 1e-4)
    # Eq. 4: m0 = 0.996, rising to 1.
    assert (vicreg.ema_momentum_start, vicreg.ema_momentum_end) == (0.996, 1.0)
    # Sec. 5.1: stop when srank < d/2.
    assert vicreg.srank_abort_fraction == 0.5
    # Sec. 3.2: L_r = 24 and d_k = 2048.
    assert (realizer.kind, realizer.hidden_dim, realizer.num_layers) == ("fs_dfm", 2048, 24)
    assert realizer.step_budgets == (2, 4, 8)
    assert (realizer.c_max, realizer.temperature) == (1.0, 1.0)
    # Sec. 3.2 and 5.1: spectral normalisation of W_K, W_V; plan dropout 0.1.
    assert cfg.bridge.spectral_norm is True and cfg.bridge.plan_dropout == 0.1
    assert training.batch_size == 64 and training.stage == "planner"


def test_paper_constants():
    """Algorithm 1: three planner passes; Sec. 1.3: K <= 8."""
    assert PLANNER_PASSES == 3
    assert MAX_REALIZER_STEPS == 8


def test_tiny_groups_shrink_widths_but_keep_the_paper_constants(tiny_config):
    assert (tiny_config.planner.latent_dim, tiny_config.realizer.hidden_dim) == (32, 64)
    assert tiny_config.realizer.sequential_depth == tiny_config.realizer.num_steps + 3 == 7
    assert tiny_config.vicreg == VICRegConfig()


# ---------------------------------------------------------------------------------------------
# Composition and the K + 3 depth
# ---------------------------------------------------------------------------------------------
def test_group_and_value_overrides_compose():
    """Swapping groups and overriding a value: kind logit_kl, K = 4, sequential depth K + 3 = 7."""
    cfg = load_config(["planner=tiny", "data=tiny", "realizer=logit_kl", "realizer.num_steps=4"])
    assert cfg.realizer.kind == "logit_kl"
    assert cfg.realizer.num_steps == 4
    assert cfg.realizer.sequential_depth == 7
    assert cfg.planner.latent_dim == 32                    # from planner/tiny.yaml
    assert cfg.vicreg == VICRegConfig()                     # untouched groups keep their defaults


@pytest.mark.parametrize("realizer", ["fs_dfm", "logit_kl"])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_sequential_depth_is_k_plus_3(realizer, k):
    """Algorithm 1: passes 1-3 of the planner, then K realizer passes."""
    cfg = load_config([f"realizer={realizer}", f"realizer.num_steps={k}"])
    assert cfg.realizer.sequential_depth == k + 3


def test_override_of_a_misspelled_key_fails():
    """Hydra refuses to override a key that does not exist (loader docstring: typos are not silently ignored)."""
    with pytest.raises(ConfigCompositionException):
        load_config(["realizer.num_step=4"])


def test_validate_config_accepts_plain_and_hydra_trees():
    assert validate_config({}) == TextVPAConfig()
    raw = OmegaConf.create({"realizer": {"num_steps": 4}, "vicreg": {"lambda_u": 0.5, "lambda_c": "${vicreg.lambda_u}"}})
    cfg = validate_config(raw)
    assert cfg.realizer.num_steps == 4
    assert cfg.vicreg.lambda_c == 0.5                       # interpolation resolved before validation


def test_validate_config_rejects_unknown_keys_in_plain_trees():
    with pytest.raises(pydantic.ValidationError) as info:
        validate_config({"realizer": {"num_step": 4}})
    assert has_error_at(info.value, ("realizer", "num_step"), "extra_forbidden")


# ---------------------------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------------------------
CONSTRAINT_VIOLATIONS = [
    # Algorithm 1 and Sec. 1.3: K is one of the trained budgets, 1 <= K <= 8, budgets strictly increasing.
    pytest.param(["realizer.num_steps=3"], ("realizer",), id="num_steps-not-a-trained-budget"),
    pytest.param(["realizer.step_budgets=[2,4,8,16]"], ("realizer",), id="budget-above-8"),
    pytest.param(["realizer.step_budgets=[0,2,4,8]"], ("realizer",), id="budget-below-1"),
    pytest.param(["realizer.step_budgets=[2,8,4]"], ("realizer",), id="budgets-not-increasing"),
    pytest.param(["realizer.step_budgets=[2,4,4,8]"], ("realizer",), id="budgets-repeated"),
    pytest.param(["realizer.step_budgets=[]"], ("realizer",), id="budgets-empty"),
    pytest.param(["realizer=logit_kl", "realizer.num_steps=5"], ("realizer",), id="logit_kl-num_steps-not-a-budget"),
    # Eq. 6 and Prop. 3: 0 < eps < gamma^2.
    pytest.param(["vicreg.eps=1.0"], ("vicreg",), id="eps-equal-to-gamma^2"),
    pytest.param(["vicreg.eps=2.0"], ("vicreg",), id="eps-above-gamma^2"),
    pytest.param(["vicreg.gamma=0.5", "vicreg.eps=0.3"], ("vicreg",), id="gamma^2<eps<gamma"),
    pytest.param(["vicreg.eps=0"], ("vicreg", "eps"), id="eps-zero"),
    # Prop. 4(a): c_max in (0, 1], T > 0.
    pytest.param(["realizer.c_max=1.5"], ("realizer", "c_max"), id="c_max-above-1"),
    pytest.param(["realizer.c_max=0"], ("realizer", "c_max"), id="c_max-zero"),
    pytest.param(["realizer.c_max=-0.5"], ("realizer", "c_max"), id="c_max-negative"),
    pytest.param(["realizer.temperature=0"], ("realizer", "temperature"), id="temperature-zero"),
    pytest.param(["realizer.temperature=-1"], ("realizer", "temperature"), id="temperature-negative"),
    # Eq. 8: the ensemble spread needs at least two heads.
    pytest.param(["planner.predictor.n_ens=1"], ("planner", "predictor", "n_ens"), id="n_ens-1"),
    # Eq. 4: 0 <= m0 <= m_end <= 1.
    pytest.param(["vicreg.ema_momentum_start=0.999", "vicreg.ema_momentum_end=0.99"], ("vicreg",),
                 id="ema-start-above-end"),
    pytest.param(["vicreg.ema_momentum_end=1.01"], ("vicreg", "ema_momentum_end"), id="ema-end-above-1"),
    # Attention widths divisible by the number of heads.
    pytest.param(["realizer.hidden_dim=2050"], ("realizer",), id="realizer-width-not-divisible-by-heads"),
    pytest.param(["planner.encoder.dim=510"], ("planner", "encoder"), id="encoder-width-not-divisible-by-heads"),
    pytest.param(["planner.latent_dim=250"], ("planner",), id="latent-dim-not-divisible-by-predictor-heads"),
    # Optimizer schedules.
    pytest.param(["training.planner.warmup_steps=200000"], ("training", "planner"), id="planner-warmup-above-max"),
    pytest.param(["training.realizer.warmup_steps=300001"], ("training", "realizer"), id="realizer-warmup-above-max"),
    pytest.param(["training.joint.warmup_steps=60000"], ("training", "joint"), id="joint-warmup-above-max"),
    pytest.param(["training.joint.planner_lr_scale=0"], ("training", "joint", "planner_lr_scale"),
                 id="planner-lr-scale-zero"),
    pytest.param(["training.joint.planner_lr_scale=1.5"], ("training", "joint", "planner_lr_scale"),
                 id="planner-lr-scale-above-1"),
    # Weights of Eq. 5 (lambda_v > 0 for Theorem 1), the n >= 2 rows of Eqs. 6-7, and the Sec. 5.1 guard.
    pytest.param(["vicreg.lambda_v=0"], ("vicreg", "lambda_v"), id="lambda_v-zero"),
    pytest.param(["vicreg.lambda_c=-0.1"], ("vicreg", "lambda_c"), id="lambda_c-negative"),
    pytest.param(["vicreg.lambda_u=-1"], ("vicreg", "lambda_u"), id="lambda_u-negative"),
    pytest.param(["training.batch_size=1"], ("training", "batch_size"), id="batch-size-1"),
    pytest.param(["vicreg.srank_abort_fraction=0"], ("vicreg", "srank_abort_fraction"), id="abort-fraction-zero"),
    pytest.param(["bridge.plan_dropout=1.0"], ("bridge", "plan_dropout"), id="plan-dropout-1"),
]


@pytest.mark.parametrize(("overrides", "loc"), CONSTRAINT_VIOLATIONS)
def test_violated_constraint_raises_validation_error(overrides, loc):
    """Each case breaks one rule (the empty budget list also leaves K outside the budgets); the error is reported
    at the field or section that holds the rule."""
    assert has_error_at(raises_validation_error(overrides), loc)


@pytest.mark.parametrize("overrides", [
    pytest.param(["realizer.step_budgets=[1,8]"], id="K=1-and-K=8"),
    pytest.param(["realizer.step_budgets=[1]", "realizer.num_steps=1"], id="single-budget-K=1"),
    pytest.param(["realizer.c_max=1.0", "realizer.temperature=1e-3"], id="c_max=1-small-T"),
    pytest.param(["vicreg.gamma=0.5", "vicreg.eps=0.2499"], id="eps-just-below-gamma^2"),
    pytest.param(["vicreg.ema_momentum_start=1.0"], id="ema-start=end"),
    pytest.param(["planner.predictor.n_ens=2"], id="n_ens=2"),
    pytest.param(["training.planner.warmup_steps=100000"], id="warmup=max_steps"),
    pytest.param(["vicreg.lambda_c=0"], id="lambda_c=0-ablation-A6"),
    pytest.param(["training.batch_size=2"], id="batch-size-2"),
])
def test_boundary_values_are_accepted(overrides):
    """The constraints are no tighter than the paper's: closed ends stay admissible."""
    load_config(overrides)


@pytest.mark.parametrize(("override", "loc"), [
    ("+planner.foo=1", ("planner", "foo")),
    ("+foo=1", ("foo",)),
    ("+vicreg.lambda_x=1", ("vicreg", "lambda_x")),
    ("+training.planner.momentum=0.9", ("training", "planner", "momentum")),
])
def test_unknown_key_raises_validation_error(override, loc):
    assert has_error_at(raises_validation_error([override]), loc, "extra_forbidden")


@pytest.mark.parametrize(("override", "loc"), [
    ("training.batch_size=abc", ("training", "batch_size")),
    ("realizer.num_steps=2.5", ("realizer", "num_steps")),
    ("realizer.temperature=hot", ("realizer", "temperature")),
    ("bridge.spectral_norm=maybe", ("bridge", "spectral_norm")),
    ("realizer.kind=autoregressive", ("realizer", "kind")),
    ("training.stage=pretrain", ("training", "stage")),
])
def test_wrong_type_raises_validation_error(override, loc):
    assert has_error_at(raises_validation_error([override]), loc)


@pytest.mark.parametrize("build", [
    pytest.param(lambda: RealizerConfig(num_steps=3), id="realizer-num_steps"),
    pytest.param(lambda: RealizerConfig(step_budgets=(2, 4, 8, 16)), id="realizer-budget-16"),
    pytest.param(lambda: VICRegConfig(eps=1.0), id="vicreg-eps"),
    pytest.param(lambda: PredictorConfig(n_ens=1), id="predictor-n_ens"),
    pytest.param(lambda: StageConfig(learning_rate=1e-4, warmup_steps=11, weight_decay=0.0, max_steps=10),
                 id="stage-warmup"),
    pytest.param(lambda: TextVPAConfig.model_validate({"realizer": {"c_max": 2.0}}), id="root-c_max"),
])
def test_constraints_hold_without_hydra(build):
    """The constraints live in the schema, so direct construction is checked too."""
    with pytest.raises(pydantic.ValidationError):
        build()


# ---------------------------------------------------------------------------------------------
# Immutability and serialisation
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("path", "field", "value"), [
    (("realizer",), "num_steps", 2),
    (("planner", "predictor"), "n_ens", 3),
    (("vicreg",), "eps", 0.5),
    (("training", "joint"), "planner_lr_scale", 0.5),
    ((), "vicreg", VICRegConfig()),
])
def test_configs_are_frozen(path, field, value):
    cfg = load_config()
    section = cfg
    for name in path:
        section = getattr(section, name)
    before = getattr(section, field)
    with pytest.raises(pydantic.ValidationError):
        setattr(section, field, value)
    assert getattr(section, field) == before


@pytest.mark.parametrize("overrides", [
    pytest.param([], id="defaults"),
    pytest.param(["planner=tiny", "data=tiny", "realizer=logit_kl", "realizer.step_budgets=[1,4]", "realizer.num_steps=4",
                  "planner.predictor.head_hidden_dim=64", "vicreg.head_reduction=sum", "vicreg.eps=1e-6",
                  "training.stage=joint", "training.device=cuda:1"], id="customised"),
])
def test_yaml_round_trip(tmp_path, overrides):
    cfg = load_config(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(cfg.to_yaml(), encoding="utf-8")
    restored = TextVPAConfig.from_yaml(path)
    assert restored == cfg
    assert isinstance(restored.realizer.step_budgets, tuple)


def test_saved_config_recomposes_with_hydra_overrides(tmp_path):
    """A config stored next to a checkpoint is itself a primary Hydra config that accepts overrides."""
    cfg = load_config(["planner=tiny", "data=tiny", "realizer=logit_kl", "realizer.num_steps=4"])
    (tmp_path / "run.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    recomposed = load_config(["realizer.num_steps=2"], config_name="run", directory=tmp_path)
    expected = cfg.model_dump()
    expected["realizer"]["num_steps"] = 2
    assert recomposed == TextVPAConfig.model_validate(expected)
    assert recomposed.realizer.sequential_depth == 5


def test_from_yaml_validates(tmp_path):
    data = yaml.safe_load(load_config().to_yaml())
    data["realizer"]["num_steps"] = 3
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        TextVPAConfig.from_yaml(path)


# ---------------------------------------------------------------------------------------------
# Training stages (Sec. 5.1)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("stage", ["planner", "realizer", "joint"])
def test_active_stage_follows_training_stage(stage):
    training = load_config([f"training.stage={stage}"]).training
    assert training.stage == stage
    assert training.active_stage is getattr(training, stage)


def test_joint_stage_trains_the_planner_at_one_tenth_of_the_learning_rate():
    """Sec. 5.1, Stage 3: the planner is unfrozen at one tenth of the learning rate."""
    assert load_config().training.joint.planner_lr_scale == 0.1
    assert TrainingConfig().joint.planner_lr_scale == 0.1
    joint = JointStageConfig(learning_rate=1e-4, warmup_steps=0, weight_decay=0.0, max_steps=10)
    assert joint.planner_lr_scale == 0.1
    assert isinstance(load_config(["training.stage=joint"]).training.active_stage, JointStageConfig)
