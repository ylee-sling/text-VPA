# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of ``text_vpa.models.ema`` against PR-JEPA Sec. 3.1, Eq. 4 (EMA target encoder).

Specification as checked here, with k the update count and T = T_train:

    theta_bar <- m_k theta_bar + (1 - m_k) theta                      (Eq. 4; f_theta_bar gets no gradient)
    linear (Eq. 4):  m_k = m0 + (m_end - m0) min(k, T) / T             paper: m_end = 1, m0 = 0.996
    cosine:          m_k = m_end - (m_end - m0) (cos(pi min(k, T) / T) + 1) / 2
                         = 1 - (1 - m0) (cos(pi k / T) + 1) / 2         for m_end = 1

``update()`` uses m_k at the current count k, then increments k, and returns m_k. Parameters are paired
by name; buffers are copied, not averaged. References are written by hand from these formulas (float64
for tensors, Python floats for the schedule) and never call the code under test.
"""

from __future__ import annotations

import io
import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from tests.conftest import ATOL, RTOL
from text_vpa.configs.schema import VICRegConfig
from text_vpa.models.ema import EMATargetEncoder, ema_momentum
from text_vpa.models.encoders import ContextEncoder, TargetEncoder

M0 = 0.996  # Eq. 4
ENDPOINTS = [(0.996, 1.0), (0.9, 0.99), (0.0, 1.0), (0.5, 0.5)]
SQRT_HALF = math.sqrt(2.0) / 2.0  # cos(pi / 4)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def linear_momentum(k: int, total: int, m0: float, m_end: float) -> float:
    return m0 + (m_end - m0) * min(k, total) / total


def cosine_momentum(k: int, total: int, m0: float, m_end: float) -> float:
    return m_end - (m_end - m0) * (math.cos(math.pi * min(k, total) / total) + 1.0) / 2.0


def frozen(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def make_pair() -> tuple[ContextEncoder, TargetEncoder]:
    """f_theta and f_theta_bar = from_online(f_theta), then one simulated optimizer step on f_theta."""
    online = ContextEncoder(vocab_size=23, latent_dim=6, dim=8, num_layers=1, num_heads=2, mlp_ratio=2.0,
                            max_positions=10)
    target = TargetEncoder.from_online(online)
    perturb(online)
    return online, target


def perturb(module: nn.Module, scale: float = 0.1) -> None:
    """Stand-in for an optimizer step: theta <- theta + scale * N(0, 1)."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn_like(parameter))


def snapshot(module: nn.Module) -> dict[str, Tensor]:
    """float64 copies of every parameter, by name."""
    return {name: p.detach().to(torch.float64).clone() for name, p in module.named_parameters()}


def ema_reference(theta_bar: dict[str, Tensor], theta: dict[str, Tensor], m: float) -> dict[str, Tensor]:
    """theta_bar_new = m theta_bar + (1 - m) theta, per name, in float64."""
    return {name: m * theta_bar[name] + (1.0 - m) * theta[name] for name in theta_bar}


def assert_parameters_close(module: nn.Module, expected: dict[str, Tensor]) -> None:
    actual = dict(module.named_parameters())
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name].detach().double(), value, rtol=RTOL, atol=ATOL, msg=name)


class WithBuffers(nn.Module):
    """A linear map, a registered buffer of its own, and a BatchNorm (float running statistics, int counter)."""

    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.register_buffer("scale", torch.ones(width))
        self.norm = nn.BatchNorm1d(width)


class NamedBuffer(nn.Module):
    def __init__(self, buffer_name: str = "stat", size: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer(buffer_name, torch.arange(1.0, size + 1.0))


# ---------------------------------------------------------------------------------------------
# Momentum schedules
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize(("m0", "m_end"), ENDPOINTS)
def test_schedule_starts_at_m0_and_ends_at_m_end(schedule: str, m0: float, m_end: float) -> None:
    """m_0 = m0 and m_T = m_end for both schedules."""
    total = 1000
    assert ema_momentum(0, total, m0, m_end, schedule) == pytest.approx(m0, rel=0.0, abs=1e-15)
    assert ema_momentum(total, total, m0, m_end, schedule) == pytest.approx(m_end, rel=0.0, abs=1e-15)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize(("m0", "m_end"), ENDPOINTS)
def test_schedule_stays_at_m_end_after_T(schedule: str, m0: float, m_end: float) -> None:
    """For k > T the momentum is clamped at m_end (the target stops moving once m_end = 1)."""
    total = 40
    for k in (total + 1, 2 * total, 100 * total):
        assert ema_momentum(k, total, m0, m_end, schedule) == pytest.approx(m_end, rel=0.0, abs=1e-15)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("m0", [0.996, 0.99, 0.9])
def test_schedule_reaches_exactly_one(schedule: str, m0: float) -> None:
    """With m_end = 1 the final momentum is exactly 1.0, so the target is truly frozen after T updates."""
    assert ema_momentum(7, 7, m0, 1.0, schedule) == 1.0
    assert ema_momentum(8, 7, m0, 1.0, schedule) == 1.0


def test_linear_schedule_matches_eq4() -> None:
    """Eq. 4: m_k = m0 + (1 - m0) k / T_train with m0 = 0.996, at intermediate k."""
    total = 1000
    for k in (1, 10, 250, 500, 750, 999):
        expected = M0 + (1.0 - M0) * k / total
        assert ema_momentum(k, total, M0, 1.0, "linear") == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert ema_momentum(500, total, M0, 1.0, "linear") == pytest.approx(0.998, rel=0.0, abs=1e-15)


def test_linear_schedule_with_general_end_point() -> None:
    """m_k = m0 + (m_end - m0) k / T for m_end < 1."""
    total = 12
    for k in range(total + 1):
        expected = linear_momentum(k, total, 0.9, 0.99)
        assert ema_momentum(k, total, 0.9, 0.99, "linear") == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert ema_momentum(3, total, 0.9, 0.99, "linear") == pytest.approx(0.9225, rel=0.0, abs=1e-15)


def test_cosine_schedule_matches_closed_form_with_unit_end() -> None:
    """With m_end = 1: m_k = 1 - (1 - m0) (cos(pi k / T) + 1) / 2 at every k."""
    total = 50
    for k in range(total + 1):
        expected = 1.0 - (1.0 - M0) * (math.cos(math.pi * k / total) + 1.0) / 2.0
        assert ema_momentum(k, total, M0, 1.0, "cosine") == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_cosine_schedule_matches_closed_form_with_general_end() -> None:
    """m_k = m_end - (m_end - m0) (cos(pi k / T) + 1) / 2 at every k."""
    total = 30
    for k in range(total + 1):
        expected = cosine_momentum(k, total, 0.9, 0.99)
        assert ema_momentum(k, total, 0.9, 0.99, "cosine") == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_cosine_schedule_spot_values() -> None:
    """At T = 8: k = 2, 4, 6 give 1 - 0.002 (1 + sqrt(2)/2), 0.998 and 1 - 0.002 (1 - sqrt(2)/2)."""
    assert ema_momentum(2, 8, M0, 1.0, "cosine") == pytest.approx(1.0 - 0.002 * (1.0 + SQRT_HALF), abs=1e-15)
    assert ema_momentum(4, 8, M0, 1.0, "cosine") == pytest.approx(0.998, abs=1e-15)
    assert ema_momentum(6, 8, M0, 1.0, "cosine") == pytest.approx(1.0 - 0.002 * (1.0 - SQRT_HALF), abs=1e-15)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize(("m0", "m_end"), ENDPOINTS)
def test_schedule_is_monotone_non_decreasing(schedule: str, m0: float, m_end: float) -> None:
    """m_{k+1} >= m_k for every k, through and beyond T."""
    total = 64
    values = [ema_momentum(k, total, m0, m_end, schedule) for k in range(total + 5)]
    assert all(later >= earlier for earlier, later in zip(values[:-1], values[1:], strict=True))
    if m_end > m0:
        assert values[total] > values[0]


@pytest.mark.parametrize(("total_steps", "schedule"), [(0, "linear"), (-3, "cosine"), (10, "exponential"),
                                                       (10, "Linear")])
def test_ema_momentum_rejects_invalid_arguments(total_steps: int, schedule: str) -> None:
    """total_steps < 1 and unknown schedules raise ValueError."""
    with pytest.raises(ValueError):
        ema_momentum(1, total_steps, M0, 1.0, schedule)


# ---------------------------------------------------------------------------------------------
# The update theta_bar <- m theta_bar + (1 - m) theta
# ---------------------------------------------------------------------------------------------
def test_update_matches_hand_computed_ema_for_every_parameter() -> None:
    """One update with Eq. 4's defaults equals 0.996 theta_bar + 0.004 theta for every named parameter."""
    online, target = make_pair()
    theta_bar, theta = snapshot(target), snapshot(online)
    EMATargetEncoder(online, target, total_steps=100).update()
    expected = ema_reference(theta_bar, theta, M0)
    assert_parameters_close(target, expected)
    moved = max(float((expected[n] - theta_bar[n]).abs().max()) for n in theta_bar)
    assert moved > 1e-4  # the update is large enough for the comparison to tell it from a no-op


def test_successive_updates_follow_the_schedule() -> None:
    """Five updates with m0 = 0.7, T = 3 (m = 0.7, 0.8, 0.9, 1, 1) and theta changing in between match a
    float64 recursion of Eq. 4."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=3, momentum_start=0.7, momentum_end=1.0)
    expected = snapshot(target)
    for m in (0.7, 0.8, 0.9, 1.0, 1.0):
        expected = ema_reference(expected, snapshot(online), m)
        ema.update()
        assert_parameters_close(target, expected)
        perturb(online)


def test_update_returns_momentum_of_current_count_and_increments_step() -> None:
    """update() returns m_k for k = 0, 1, ... (m0 = 0.6, T = 4: 0.6, 0.7, 0.8, 0.9, 1.0, 1.0)."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=4, momentum_start=0.6, momentum_end=1.0)
    assert ema.step == 0
    for k, expected in enumerate([0.6, 0.7, 0.8, 0.9, 1.0, 1.0]):
        assert ema.update() == pytest.approx(expected, rel=0.0, abs=1e-15)
        assert ema.step == k + 1


def test_momentum_reports_the_next_update_or_a_given_step() -> None:
    """momentum() is m_k at the current count; momentum(j) is m_j (cosine schedule, m0 = 0.996, T = 8)."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=8, schedule="cosine")
    assert ema.momentum() == pytest.approx(M0, abs=1e-15)
    ema.update()
    ema.update()
    assert ema.momentum() == pytest.approx(1.0 - 0.002 * (1.0 + SQRT_HALF), abs=1e-15)
    assert ema.momentum(6) == pytest.approx(1.0 - 0.002 * (1.0 - SQRT_HALF), abs=1e-15)
    assert ema.step == 2


def test_cosine_update_uses_cosine_momentum() -> None:
    """An update at k = 4 of T = 8 under the cosine schedule averages with m = 0.998."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=8, schedule="cosine")
    ema.step = 4
    theta_bar, theta = snapshot(target), snapshot(online)
    assert ema.update() == pytest.approx(0.998, abs=1e-15)
    assert_parameters_close(target, ema_reference(theta_bar, theta, 0.998))


def test_update_leaves_online_parameters_unchanged() -> None:
    """The EMA writes only theta_bar; theta is read, never modified."""
    online, target = make_pair()
    before = {name: p.clone() for name, p in online.named_parameters()}
    EMATargetEncoder(online, target, total_steps=10, momentum_start=0.5).update()
    for name, p in online.named_parameters():
        assert torch.equal(p, before[name]), name


def test_unit_momentum_leaves_target_bit_identical() -> None:
    """m = 1 (m0 = m_end = 1) leaves every target parameter bit-identical, however far theta has moved."""
    online, target = make_pair()
    with torch.no_grad():
        target.head.bias.zero_()  # a zero entry exposes even a 1e-16 leak of theta into theta_bar
    before = {name: p.clone() for name, p in target.named_parameters()}
    ema = EMATargetEncoder(online, target, total_steps=5, momentum_start=1.0, momentum_end=1.0)
    for _ in range(3):
        assert ema.update() == 1.0
        perturb(online)
    for name, p in target.named_parameters():
        assert torch.equal(p, before[name]), name


def test_target_is_frozen_once_the_schedule_reaches_one() -> None:
    """Eq. 4 with T = 2: updates at k = 0, 1 move theta_bar; from k = 2 on (m_k = 1) it no longer changes."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=2)
    ema.update()
    ema.update()
    perturb(online)
    with torch.no_grad():
        target.head.bias.zero_()
    before = {name: p.clone() for name, p in target.named_parameters()}
    for _ in range(3):
        ema.update()
        perturb(online)
    for name, p in target.named_parameters():
        assert torch.equal(p, before[name]), name


def test_target_stays_frozen_and_graph_free_after_updates() -> None:
    """After updates f_theta_bar's parameters are still frozen leaves without autograd history."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=10)
    with torch.enable_grad():
        online_latents = online(torch.randint(0, 23, (2, 4)))
        ema.update()
    assert online_latents.requires_grad  # theta was part of a live graph during the update
    target.assert_frozen()
    assert all(p.grad_fn is None and p.is_leaf and not p.requires_grad for p in target.parameters())


def test_buffers_are_copied_not_averaged() -> None:
    """A registered buffer, BatchNorm running statistics and its integer counter are copied, not averaged."""
    online = WithBuffers()
    target = frozen(WithBuffers())
    target.load_state_dict(online.state_dict())
    perturb(online)
    online.scale.copy_(torch.tensor([0.5, 2.0, -3.0]))
    online.norm.running_mean.copy_(torch.tensor([3.0, -1.0, 0.5]))
    online.norm.running_var.copy_(torch.tensor([2.0, 0.25, 4.0]))
    online.norm.num_batches_tracked.fill_(7)
    theta_bar, theta = snapshot(target), snapshot(online)
    EMATargetEncoder(online, target, total_steps=10, momentum_start=0.9).update()
    for name, buffer in online.named_buffers():
        assert torch.equal(dict(target.named_buffers())[name], buffer), name
    assert_parameters_close(target, ema_reference(theta_bar, theta, 0.9))


def test_parameters_are_paired_by_name_not_registration_order() -> None:
    """Target submodules registered in another order are still averaged with their namesakes."""
    online = nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 2)})
    target = frozen(nn.ModuleDict({"b": nn.Linear(2, 2), "a": nn.Linear(2, 2)}))
    theta_bar, theta = snapshot(target), snapshot(online)
    EMATargetEncoder(online, target, total_steps=10, momentum_start=0.75).update()
    assert_parameters_close(target, ema_reference(theta_bar, theta, 0.75))


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
def test_rejects_mismatched_parameter_names() -> None:
    """Parameters are paired by name, so differing names raise ValueError."""
    online = nn.ModuleDict({"a": nn.Linear(2, 2)})
    with pytest.raises(ValueError):
        EMATargetEncoder(online, frozen(nn.ModuleDict({"b": nn.Linear(2, 2)})), total_steps=10)


def test_rejects_mismatched_parameter_shapes() -> None:
    """A namesake of another shape cannot be averaged."""
    online = nn.ModuleDict({"a": nn.Linear(2, 2)})
    with pytest.raises(ValueError):
        EMATargetEncoder(online, frozen(nn.ModuleDict({"a": nn.Linear(2, 3)})), total_steps=10)


def test_rejects_mismatched_parameter_dtypes() -> None:
    """A namesake of another dtype is rejected."""
    online = nn.ModuleDict({"a": nn.Linear(2, 2)})
    with pytest.raises(ValueError):
        EMATargetEncoder(online, frozen(nn.ModuleDict({"a": nn.Linear(2, 2)}).double()), total_steps=10)


def test_rejects_target_that_requires_grad() -> None:
    """A target with even one trainable parameter is rejected: f_theta_bar receives no gradient."""
    online = nn.ModuleDict({"a": nn.Linear(2, 2)})
    target = frozen(nn.ModuleDict({"a": nn.Linear(2, 2)}))
    target["a"].bias.requires_grad_(True)
    with pytest.raises(ValueError):
        EMATargetEncoder(online, target, total_steps=10)


def test_rejects_online_module_used_as_its_own_target() -> None:
    """theta_bar must be a separate copy of theta."""
    module = frozen(nn.ModuleDict({"a": nn.Linear(2, 2)}))
    with pytest.raises(ValueError):
        EMATargetEncoder(module, module, total_steps=10)


def test_rejects_invalid_schedule() -> None:
    """Only the linear (Eq. 4) and cosine schedules exist."""
    online, target = make_pair()
    with pytest.raises(ValueError):
        EMATargetEncoder(online, target, total_steps=10, schedule="exponential")


@pytest.mark.parametrize("total_steps", [0, -1])
def test_rejects_non_positive_total_steps(total_steps: int) -> None:
    """T_train must be at least one update."""
    online, target = make_pair()
    with pytest.raises(ValueError):
        EMATargetEncoder(online, target, total_steps=total_steps)


@pytest.mark.parametrize(("m0", "m_end"), [(0.999, 0.996), (-0.1, 1.0), (0.996, 1.1)])
def test_rejects_momentum_outside_ordered_unit_interval(m0: float, m_end: float) -> None:
    """Eq. 4 needs 0 <= m0 <= m_end <= 1."""
    online, target = make_pair()
    with pytest.raises(ValueError):
        EMATargetEncoder(online, target, total_steps=10, momentum_start=m0, momentum_end=m_end)


def test_rejects_mismatched_buffer_names() -> None:
    """Buffers are paired by name for copying, so differing names raise ValueError."""
    with pytest.raises(ValueError):
        EMATargetEncoder(NamedBuffer("stat"), frozen(NamedBuffer("other")), total_steps=10)


def test_rejects_mismatched_buffer_shapes() -> None:
    """A buffer cannot be copied into a buffer of another shape; this is a construction-time ValueError."""
    with pytest.raises(ValueError):
        EMATargetEncoder(NamedBuffer(size=1), frozen(NamedBuffer(size=3)), total_steps=10)


# ---------------------------------------------------------------------------------------------
# Resumption and configuration
# ---------------------------------------------------------------------------------------------
def test_state_dict_round_trip_restores_step() -> None:
    """A fresh EMA loaded from state_dict resumes at the saved count and applies m_k for that count."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=10, momentum_start=0.5)
    for _ in range(3):
        ema.update()
    resumed = EMATargetEncoder(online, target, total_steps=10, momentum_start=0.5)
    resumed.load_state_dict(ema.state_dict())
    assert resumed.step == 3
    theta_bar, theta = snapshot(target), snapshot(online)
    assert resumed.update() == pytest.approx(0.65, abs=1e-15)  # 0.5 + 0.5 * 3 / 10
    assert_parameters_close(target, ema_reference(theta_bar, theta, 0.65))


def test_state_dict_survives_torch_save_and_weights_only_load() -> None:
    """The EMA state is plain data: it round-trips through torch.save / torch.load(weights_only=True)."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=10, schedule="cosine")
    for _ in range(4):
        ema.update()
    buffer = io.BytesIO()
    torch.save(ema.state_dict(), buffer)
    buffer.seek(0)
    resumed = EMATargetEncoder(online, target, total_steps=10, schedule="cosine")
    resumed.load_state_dict(torch.load(buffer, weights_only=True))
    assert resumed.step == 4
    assert resumed.momentum() == pytest.approx(cosine_momentum(4, 10, M0, 1.0), abs=1e-15)


@pytest.mark.parametrize(("key", "value"), [("total_steps", 11), ("momentum_start", 0.99),
                                            ("momentum_end", 0.999), ("schedule", "cosine")])
def test_load_state_dict_rejects_mismatched_config(key: str, value: object) -> None:
    """A checkpoint from a different schedule raises ValueError and leaves the count untouched."""
    online, target = make_pair()
    ema = EMATargetEncoder(online, target, total_steps=10)
    ema.update()
    state = ema.state_dict()
    state[key] = value
    resumed = EMATargetEncoder(online, target, total_steps=10)
    with pytest.raises(ValueError):
        resumed.load_state_dict(state)
    assert resumed.step == 0


def test_from_config_reads_vicreg_ema_fields() -> None:
    """from_config takes m0, m_end and the schedule from VICRegConfig and T from the caller."""
    cfg = VICRegConfig(ema_momentum_start=0.99, ema_momentum_end=0.999, ema_schedule="cosine")
    online, target = make_pair()
    ema = EMATargetEncoder.from_config(online, target, cfg, total_steps=10)
    assert (ema.momentum_start, ema.momentum_end, ema.schedule, ema.total_steps) == (0.99, 0.999, "cosine", 10)
    assert ema.momentum(5) == pytest.approx(0.999 - 0.009 * 0.5, abs=1e-15)  # cos(pi / 2) = 0
    assert ema.momentum(10) == pytest.approx(0.999, abs=1e-15)


def test_from_config_with_default_vicreg_config_is_eq4(tiny_config) -> None:
    """The packaged defaults give Eq. 4: linear from m0 = 0.996 to 1."""
    online, target = make_pair()
    ema = EMATargetEncoder.from_config(online, target, tiny_config.vicreg, total_steps=1000)
    assert (ema.momentum_start, ema.momentum_end, ema.schedule) == (0.996, 1.0, "linear")
    assert ema.momentum(250) == pytest.approx(0.996 + 0.004 * 0.25, abs=1e-15)
