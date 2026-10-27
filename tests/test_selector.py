# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of ``text_vpa.models.selector`` against PR-JEPA Sec. 3.1, Algorithm 1 (line 2) and Prop. 1.

Specification as checked here:

    pi_phi^h(u | s_x) = softmax(logits),  logits = MLP(LN(pool(s_x)))
        pool: mean of s_x over the REAL context tokens (padded tokens never matter)
        MLP:  [Linear -> GELU] x num_layers, then Linear to |U| logits
    choose_mode strategies:
        "sample"  u ~ softmax(logits / T)                  (Algorithm 1, line 2, with T = 1)
        "greedy"  u = argmax logits
        "gumbel"  u = argmax(logits + g), g ~ Gumbel(0, 1)  (Gumbel-max: u ~ softmax(logits) for every T)
                  one_hot = y_hard - sg(y_soft) + y_soft,  y_soft = softmax((logits + g) / T):
                  forward value one-hot at u, gradient that of y_soft
    log_prob = log softmax(logits)[u]   (temperature 1)
    select() is pass 2 of the K + 3 chain: exactly one evaluation of pi_phi^h.

Sampling tests use fixed ``torch.Generator`` seeds and n independent draws; every empirical frequency
must lie within 5 standard errors sqrt(p (1 - p) / n) of its probability. References are float64
computations written from the formulas above; they never call the code under test.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from tests.conftest import ATOL, RTOL, accelerator
from text_vpa.models.selector import ModeSelection, ModeSelector, choose_mode

LATENT, NUM_MODES, HIDDEN = 12, 4, 10
BASE_LOGITS = torch.tensor([1.2, 0.1, -0.7, 0.5])  # softmax ~ (0.505, 0.168, 0.076, 0.251)
NUM_DRAWS = 20_000
MAX_STANDARD_ERRORS = 5.0
STRATEGIES = ("sample", "greedy", "gumbel")


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
def make_selector(num_layers: int = 2) -> ModeSelector:
    """A small pi_phi^h in eval mode, with a non-trivial LayerNorm gain and shift."""
    selector = ModeSelector(LATENT, NUM_MODES, hidden_dim=HIDDEN, num_layers=num_layers)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        selector.norm.weight.copy_(1.0 + 0.1 * torch.randn(LATENT, generator=generator))
        selector.norm.bias.copy_(0.1 * torch.randn(LATENT, generator=generator))
    return selector.eval()


def make_context() -> tuple[Tensor, Tensor]:
    """s_x for three contexts of 7 tokens: no padding, trailing padding, scattered padding."""
    s_x = torch.randn(3, 7, LATENT, generator=torch.Generator().manual_seed(1))
    mask = torch.tensor([
        [True] * 7,
        [True] * 4 + [False] * 3,
        [False, True, False, True, True, False, False],
    ])
    return s_x, mask


def random_logits(rows: int, seed: int = 3, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.randn(rows, NUM_MODES, generator=torch.Generator().manual_seed(seed)).to(dtype)


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------------------------------------
# float64 references written from the specification
# ---------------------------------------------------------------------------------------------
def f64(t: Tensor) -> Tensor:
    return t.detach().to(device="cpu", dtype=torch.float64)


def ref_layer_norm(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + norm.eps) * f64(norm.weight) + f64(norm.bias)


def ref_linear(x: Tensor, linear: nn.Linear) -> Tensor:
    return x @ f64(linear.weight).T + f64(linear.bias)


def ref_gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def ref_logits(selector: ModeSelector, s_x: Tensor, mask: Tensor | None) -> Tensor:
    """Loop-pooled mean over real tokens -> LayerNorm -> [Linear -> GELU]* -> Linear, in float64."""
    x = f64(s_x)
    pooled = []
    for b in range(x.shape[0]):
        real = [t for t in range(x.shape[1]) if mask is None or bool(mask[b, t])]
        pooled.append(sum(x[b, t] for t in real) / len(real))
    h = ref_layer_norm(torch.stack(pooled), selector.norm)
    linears = [m for m in selector.mlp if isinstance(m, nn.Linear)]
    for linear in linears[:-1]:
        h = ref_gelu(ref_linear(h, linear))
    return ref_linear(h, linears[-1])


def ref_softmax(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """softmax(logits / T) of one row, in float64."""
    x = f64(logits) / temperature
    e = torch.exp(x - x.max())
    return e / e.sum()


def ref_log_prob(logits: Tensor, modes: Tensor) -> Tensor:
    """log softmax(logits)[mode] per row: l_u - (max + log sum exp(l - max)), in float64."""
    out = []
    for row, mode in zip(f64(logits), modes.tolist(), strict=True):
        top = row.max()
        out.append(row[mode] - (top + torch.log(torch.exp(row - top).sum())))
    return torch.stack(out)


def ref_one_hot(modes: Tensor, num_modes: int = NUM_MODES) -> Tensor:
    out = torch.zeros(modes.numel(), num_modes, dtype=torch.float64)
    for b, mode in enumerate(modes.tolist()):
        out[b, mode] = 1.0
    return out


def assert_frequencies_match(modes: Tensor, probs: Tensor) -> None:
    """Every empirical frequency within MAX_STANDARD_ERRORS * sqrt(p (1 - p) / n) of p."""
    n = modes.numel()
    freq = torch.bincount(modes.cpu(), minlength=probs.numel()).double() / n
    z = (freq - probs) / torch.sqrt(probs * (1.0 - probs) / n)
    assert bool((z.abs() <= MAX_STANDARD_ERRORS).all()), (
        f"frequencies {freq.tolist()} vs probabilities {probs.tolist()}: z = {z.tolist()}")


def assert_distinguishable(p: Tensor, q: Tensor, n: int = NUM_DRAWS) -> None:
    """p and q differ by far more than the frequency tolerance, so matching one excludes the other."""
    largest_se = float(torch.sqrt(torch.maximum(p * (1 - p), q * (1 - q)) / n).max())
    assert float((p - q).abs().max()) > 4 * MAX_STANDARD_ERRORS * largest_se


# ---------------------------------------------------------------------------------------------
# The network pi_phi^h
# ---------------------------------------------------------------------------------------------
def test_logits_shape_and_dtype() -> None:
    """forward maps s_x [B, M_c, d] to logits [B, |U|]."""
    s_x, mask = make_context()
    logits = make_selector()(s_x, mask)
    assert logits.shape == (3, NUM_MODES)
    assert logits.dtype == torch.float32


def test_padded_context_rows_do_not_change_logits() -> None:
    """Replacing the padded rows of s_x by arbitrary large values leaves the logits bit-identical."""
    selector = make_selector()
    s_x, mask = make_context()
    other = s_x.clone()
    other[~mask] = 1e3 * torch.randn(int((~mask).sum()), LATENT)
    assert torch.equal(selector(other, mask), selector(s_x, mask))


def test_non_finite_padded_rows_do_not_change_logits() -> None:
    """Padded rows are excluded from the pool, so even non-finite values there cannot reach the logits."""
    selector = make_selector()
    s_x, mask = make_context()
    other = s_x.clone()
    other[1, 5] = math.inf
    other[2, 0] = math.nan
    torch.testing.assert_close(selector(other, mask), selector(s_x, mask), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_float64_forward_matches_reference(num_layers: int) -> None:
    """pi_phi^h run in float64 equals the hand-written pool -> LayerNorm -> GELU MLP reference."""
    selector = make_selector(num_layers)
    s_x, mask = make_context()
    actual = copy.deepcopy(selector).double()(s_x.double(), mask)
    torch.testing.assert_close(actual, ref_logits(selector, s_x, mask), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_float32_forward_matches_reference(num_layers: int) -> None:
    """The float32 forward agrees with the float64 reference within the float32 tolerance."""
    selector = make_selector(num_layers)
    s_x, mask = make_context()
    torch.testing.assert_close(selector(s_x, mask).double(), ref_logits(selector, s_x, mask), rtol=RTOL, atol=ATOL)


def test_missing_mask_pools_over_every_token() -> None:
    """context_mask=None averages all M_c tokens, as an all-True mask does."""
    selector = make_selector()
    s_x, _ = make_context()
    torch.testing.assert_close(selector(s_x).double(), ref_logits(selector, s_x, None), rtol=RTOL, atol=ATOL)
    assert torch.equal(selector(s_x), selector(s_x, torch.ones(3, 7, dtype=torch.bool)))


def test_from_config_uses_planner_config_widths(tiny_config) -> None:
    """from_config takes d, |U| and the MLP's hidden width and depth from PlannerConfig."""
    planner = tiny_config.planner
    selector = ModeSelector.from_config(planner)
    assert selector.norm.normalized_shape == (planner.latent_dim,)
    linears = [m for m in selector.mlp if isinstance(m, nn.Linear)]
    hidden = planner.selector.hidden_dim
    expected = [(planner.latent_dim, hidden)] + [(hidden, hidden)] * (planner.selector.num_layers - 1)
    expected.append((hidden, planner.num_modes))
    assert [(m.in_features, m.out_features) for m in linears] == expected
    assert sum(isinstance(m, nn.GELU) for m in selector.mlp) == planner.selector.num_layers
    s_x = torch.randn(2, 5, planner.latent_dim)
    assert selector(s_x).shape == (2, planner.num_modes)


def test_tiny_config_selector_matches_reference(tiny_config) -> None:
    """The tiny-config pi_phi^h in float64 equals the reference."""
    selector = ModeSelector.from_config(tiny_config.planner).eval()
    s_x = torch.randn(4, 6, tiny_config.planner.latent_dim)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[1, 2:] = False
    mask[3, 0] = False
    actual = copy.deepcopy(selector).double()(s_x.double(), mask)
    torch.testing.assert_close(actual, ref_logits(selector, s_x, mask), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_select_evaluates_the_network_exactly_once(strategy: str) -> None:
    """Pass 2 is one network evaluation; the selection is made from that evaluation's logits."""
    selector = make_selector()
    s_x, mask = make_context()
    outputs: list[Tensor] = []
    mlp_calls: list[int] = []
    handles = [
        selector.register_forward_hook(lambda module, args, output: outputs.append(output)),
        selector.mlp.register_forward_hook(lambda *_: mlp_calls.append(1)),
    ]
    selection = selector.select(s_x, mask, strategy, 1.0, generator())
    for handle in handles:
        handle.remove()
    assert len(outputs) == 1 and len(mlp_calls) == 1
    assert torch.equal(selection.logits, outputs[0])


def test_selection_fields_have_documented_shapes_and_types() -> None:
    """ModeSelection(mode [B] int64, one_hot [B, |U|], logits [B, |U|], log_prob [B])."""
    selector = make_selector()
    s_x, mask = make_context()
    selection = selector.select(s_x, mask, generator=generator())
    assert isinstance(selection, ModeSelection)
    assert selection.mode.shape == (3,) and selection.mode.dtype == torch.int64
    assert selection.one_hot.shape == (3, NUM_MODES) and selection.one_hot.dtype == torch.float32
    assert selection.logits.shape == (3, NUM_MODES)
    assert selection.log_prob.shape == (3,)
    assert bool(((selection.mode >= 0) & (selection.mode < NUM_MODES)).all())


# ---------------------------------------------------------------------------------------------
# "greedy"
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [1.0, 0.3, 4.0])
def test_greedy_is_argmax_at_every_temperature(temperature: float) -> None:
    """greedy returns the index of the largest logit (found by a Python loop), whatever T is."""
    logits = random_logits(64)
    expected = torch.tensor([max(range(NUM_MODES), key=lambda j, row=row: float(row[j])) for row in logits])
    selection = choose_mode(logits, "greedy", temperature)
    assert torch.equal(selection.mode, expected)
    assert torch.equal(selection.one_hot.double(), ref_one_hot(expected))


# ---------------------------------------------------------------------------------------------
# "sample": u ~ softmax(logits / T)
# ---------------------------------------------------------------------------------------------
def test_sample_is_reproducible_with_a_fixed_generator() -> None:
    """The same generator seed gives the same modes; a different seed gives different ones."""
    logits = BASE_LOGITS.repeat(256, 1)
    first = choose_mode(logits, "sample", 1.0, generator(11)).mode
    assert torch.equal(choose_mode(logits, "sample", 1.0, generator(11)).mode, first)
    assert not torch.equal(choose_mode(logits, "sample", 1.0, generator(12)).mode, first)


def test_select_sample_is_reproducible_with_a_fixed_generator() -> None:
    """ModeSelector.select passes the generator through to the draw."""
    selector = make_selector()
    s_x, mask = make_context()
    first = selector.select(s_x.repeat(50, 1, 1), mask.repeat(50, 1), "sample", 1.0, generator(4)).mode
    again = selector.select(s_x.repeat(50, 1, 1), mask.repeat(50, 1), "sample", 1.0, generator(4)).mode
    assert torch.equal(again, first)


def test_sample_frequencies_match_softmax() -> None:
    """20000 draws at T = 1 follow softmax(logits) (Algorithm 1, line 2)."""
    modes = choose_mode(BASE_LOGITS.repeat(NUM_DRAWS, 1), "sample", 1.0, generator(0)).mode
    assert_frequencies_match(modes, ref_softmax(BASE_LOGITS))


@pytest.mark.parametrize("temperature", [0.5, 2.0])
def test_sample_frequencies_follow_temperature(temperature: float) -> None:
    """At temperature T the draws follow softmax(logits / T), which is clearly not softmax(logits)."""
    expected = ref_softmax(BASE_LOGITS, temperature)
    assert_distinguishable(expected, ref_softmax(BASE_LOGITS))
    modes = choose_mode(BASE_LOGITS.repeat(NUM_DRAWS, 1), "sample", temperature, generator(1)).mode
    assert_frequencies_match(modes, expected)


def test_sample_draws_each_row_from_its_own_logits() -> None:
    """Rows with different logits are sampled from their own distributions (interleaved in one batch)."""
    second = torch.tensor([-1.0, 0.8, 0.3, -0.2])
    logits = torch.stack([BASE_LOGITS, second]).repeat(NUM_DRAWS, 1)
    modes = choose_mode(logits, "sample", 1.0, generator(2)).mode
    assert_frequencies_match(modes[0::2], ref_softmax(BASE_LOGITS))
    assert_frequencies_match(modes[1::2], ref_softmax(second))


# ---------------------------------------------------------------------------------------------
# One-hot outputs and gradient paths
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["sample", "greedy"])
def test_sample_and_greedy_one_hot_is_exact_and_has_no_gradient_path(strategy: str) -> None:
    """For sample and greedy the one-hot is exact and detached, even when the logits require grad."""
    logits = random_logits(32).requires_grad_(True)
    selection = choose_mode(logits, strategy, 1.0, generator())
    assert torch.equal(selection.one_hot.double(), ref_one_hot(selection.mode))
    assert not selection.one_hot.requires_grad and selection.one_hot.grad_fn is None


@pytest.mark.parametrize("temperature", [0.1, 1.0, 5.0])
def test_gumbel_one_hot_forward_value_is_one_hot_at_mode(temperature: float) -> None:
    """The straight-through forward value equals the one-hot of the chosen mode within 1e-6."""
    selection = choose_mode(random_logits(256), "gumbel", temperature, generator(6))
    torch.testing.assert_close(selection.one_hot.double(), ref_one_hot(selection.mode), rtol=0.0, atol=1e-6)


def test_gumbel_one_hot_carries_gradient_to_the_logits() -> None:
    """A loss on the Gumbel one-hot gives every row of the logits a non-zero gradient."""
    logits = random_logits(16).requires_grad_(True)
    selection = choose_mode(logits, "gumbel", 1.0, generator(8))
    assert selection.one_hot.requires_grad
    (selection.one_hot * torch.randn(16, NUM_MODES)).sum().backward()
    assert logits.grad is not None
    assert bool((logits.grad.abs().sum(dim=1) > 0).all())


def test_gumbel_one_hot_carries_gradient_into_the_selector() -> None:
    """Through select(), a loss on the Gumbel one-hot reaches every selector parameter."""
    selector = make_selector()
    s_x, mask = make_context()
    selection = selector.select(s_x, mask, "gumbel", 1.0, generator(9))
    (selection.one_hot * torch.randn(3, NUM_MODES)).sum().backward()
    for name, parameter in selector.named_parameters():
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0, name


@pytest.mark.parametrize("temperature", [0.7, 2.0])
def test_gumbel_gradient_is_the_tempered_softmax_jacobian(temperature: float) -> None:
    """d one_hot / d logits = (diag(y) - y y^T) / T with y a probability vector whose argmax is the mode.

    y = softmax((logits + g) / T) is not observable, so it is recovered from the off-diagonal entries
    (T J_jk = -y_j y_k, hence y_j^2 = T J_jk T J_jl / (-T J_kl) for distinct j, k, l); the test then checks
    that y sums to one (this fails for a wrong temperature factor), peaks at the mode, and reproduces the
    whole Jacobian. float64 logits keep the recovery exact to ~1e-15.
    """
    rows = 64
    logits = random_logits(rows, dtype=torch.float64).requires_grad_(True)
    selection = choose_mode(logits, "gumbel", temperature, generator(10))
    jacobian = torch.stack([
        torch.autograd.grad(selection.one_hot[:, j].sum(), logits, retain_graph=True)[0] for j in range(NUM_MODES)
    ], dim=1)                                                                   # [B, j, k] = d one_hot_j / d l_k
    scaled = temperature * jacobian
    y = torch.empty(rows, NUM_MODES, dtype=torch.float64)
    for j in range(NUM_MODES):
        k1, k2 = [i for i in range(NUM_MODES) if i != j][:2]
        y[:, j] = torch.sqrt(scaled[:, j, k1] * scaled[:, j, k2] / -scaled[:, k1, k2])
    torch.testing.assert_close(y.sum(dim=1), torch.ones(rows, dtype=torch.float64), rtol=1e-10, atol=1e-10)
    assert torch.equal(y.argmax(dim=1), selection.mode)
    expected = torch.diag_embed(y) - y[:, :, None] * y[:, None, :]
    torch.testing.assert_close(scaled, expected, rtol=1e-9, atol=1e-14)


# ---------------------------------------------------------------------------------------------
# "gumbel": Gumbel-max draws from softmax(logits) at every temperature
# ---------------------------------------------------------------------------------------------
def test_gumbel_is_reproducible_with_a_fixed_generator() -> None:
    """The same generator seed gives the same Gumbel draws; a different seed gives different ones."""
    logits = random_logits(256)
    first = choose_mode(logits, "gumbel", 1.0, generator(13)).mode
    assert torch.equal(choose_mode(logits, "gumbel", 1.0, generator(13)).mode, first)
    assert not torch.equal(choose_mode(logits, "gumbel", 1.0, generator(14)).mode, first)


@pytest.mark.parametrize("temperature", [0.2, 1.0, 5.0])
def test_gumbel_frequencies_match_softmax_at_every_temperature(temperature: float) -> None:
    """Gumbel-max: 20000 draws follow softmax(logits), not softmax(logits / T), whatever T is."""
    expected = ref_softmax(BASE_LOGITS)
    if temperature != 1.0:
        assert_distinguishable(expected, ref_softmax(BASE_LOGITS, temperature))
    modes = choose_mode(BASE_LOGITS.repeat(NUM_DRAWS, 1), "gumbel", temperature, generator(3)).mode
    assert_frequencies_match(modes, expected)


# Regression guard: the hard sample is argmax(logits + g), computed in >= float32, so T cannot change it
# (a tempered softmax in the logits' dtype rounded near-equal entries into ties).
@pytest.mark.parametrize(("dtype", "temperature"), [
    pytest.param(torch.float32, 0.1, id="float32-T0.1"),
    pytest.param(torch.float32, 0.5, id="float32-T0.5"),
    pytest.param(torch.float32, 2.0, id="float32-T2"),
    pytest.param(torch.float32, 10.0, id="float32-T10"),
    pytest.param(torch.float64, 1e6, id="float64-T1e6"),
    pytest.param(torch.float32, 1e6, id="float32-T1e6"),
    pytest.param(torch.bfloat16, 4.0, id="bfloat16-T4"),
])
def test_gumbel_mode_does_not_depend_on_temperature(dtype: torch.dtype, temperature: float) -> None:
    """For the same noise the hard sample argmax(logits + g) is the same at every temperature T."""
    logits = random_logits(NUM_DRAWS, seed=1, dtype=dtype)
    at_unit_temperature = choose_mode(logits, "gumbel", 1.0, generator(5)).mode
    assert torch.equal(choose_mode(logits, "gumbel", temperature, generator(5)).mode, at_unit_temperature)


@pytest.mark.parametrize("strategy", [
    "sample",
    "gumbel",
])
def test_bfloat16_logits_are_sampled_from_their_softmax(strategy: str) -> None:
    """bf16 logits must still be sampled from softmax(logits); 1e6 draws resolve a bias of 0.25 points."""
    logits = BASE_LOGITS.to(torch.bfloat16)
    modes = choose_mode(logits.repeat(1_000_000, 1), strategy, 1.0, generator(0)).mode
    assert_frequencies_match(modes, ref_softmax(logits))


# ---------------------------------------------------------------------------------------------
# log_prob
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("temperature", [1.0, 2.5])
def test_log_prob_is_log_softmax_of_logits_at_the_mode(strategy: str, temperature: float) -> None:
    """log_prob = log softmax(logits)[mode] at temperature 1, whatever temperature was used to choose."""
    logits = random_logits(32)
    selection = choose_mode(logits, strategy, temperature, generator(15))
    torch.testing.assert_close(selection.log_prob.double(), ref_log_prob(logits, selection.mode),
                               rtol=RTOL, atol=ATOL)


def test_log_prob_gradient_is_the_score_function() -> None:
    """d log pi(u) / d logits = e_u - softmax(logits) (the REINFORCE score), checked in float64."""
    logits = random_logits(8, dtype=torch.float64).requires_grad_(True)
    selection = choose_mode(logits, "sample", 1.0, generator(16))
    selection.log_prob.sum().backward()
    expected = ref_one_hot(selection.mode) - torch.stack([ref_softmax(row) for row in logits])
    torch.testing.assert_close(logits.grad, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["argmax", "Sample", ""])
def test_choose_mode_rejects_unknown_strategy(strategy: str) -> None:
    """Only "sample", "greedy" and "gumbel" are accepted (case-sensitive)."""
    with pytest.raises(ValueError):
        choose_mode(random_logits(2), strategy)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_choose_mode_rejects_non_positive_temperature(strategy: str, temperature: float) -> None:
    """T must be a positive number for every strategy (NaN included)."""
    with pytest.raises(ValueError):
        choose_mode(random_logits(2), strategy, temperature)


@pytest.mark.parametrize("shape", [(NUM_MODES,), (2, 3, NUM_MODES)])
def test_choose_mode_rejects_logits_that_are_not_two_dimensional(shape: tuple[int, ...]) -> None:
    """Logits must be [B, |U|]."""
    with pytest.raises(ValueError):
        choose_mode(torch.zeros(shape))


def _bad_selector_inputs(case: str) -> tuple[Tensor, Tensor | None]:
    s_x, mask = make_context()
    return {
        "2-D s_x": (s_x[0], None),
        "wrong latent width": (torch.randn(3, 7, LATENT + 1), None),
        "integer mask": (s_x, mask.long()),
        "mask shape mismatch": (s_x, mask[:, :6]),
        "all-padding row": (s_x, mask & torch.tensor([[True], [False], [True]])),
    }[case]


@pytest.mark.parametrize("case", ["2-D s_x", "wrong latent width", "integer mask", "mask shape mismatch",
                                  "all-padding row"])
def test_selector_rejects_invalid_inputs(case: str) -> None:
    """Malformed s_x or context masks raise ValueError."""
    s_x, mask = _bad_selector_inputs(case)
    with pytest.raises(ValueError):
        make_selector()(s_x, mask)


@pytest.mark.parametrize(("num_modes", "num_layers"), [(1, 2), (0, 2), (4, 0)])
def test_selector_rejects_degenerate_architecture(num_modes: int, num_layers: int) -> None:
    """|U| >= 2 (one mode cannot resolve multimodality, Prop. 1(b)) and at least one hidden layer."""
    with pytest.raises(ValueError):
        ModeSelector(LATENT, num_modes, hidden_dim=HIDDEN, num_layers=num_layers)


# ---------------------------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------------------------
def test_selector_on_accelerator_matches_cpu() -> None:
    """On CUDA/ROCm/MPS the logits agree with the CPU run and every strategy returns device tensors.

    Tolerance: accelerator matmul kernels use other reduction orders (and may use TF32), so rtol=1e-3, atol=1e-4.
    """
    device = accelerator()
    if device is None:
        pytest.skip("no CUDA/ROCm/MPS device available")
    selector = make_selector()
    s_x, mask = make_context()
    expected = selector(s_x, mask)
    moved = copy.deepcopy(selector).to(device)
    torch.testing.assert_close(moved(s_x.to(device), mask.to(device)).cpu(), expected, rtol=1e-3, atol=1e-4)
    for strategy in STRATEGIES:
        device_generator = torch.Generator(device=device).manual_seed(0)
        selection = moved.select(s_x.to(device), mask.to(device), strategy, 1.0, device_generator)
        assert selection.mode.device.type == device.type
        assert torch.equal(selection.one_hot.round().cpu().double(), ref_one_hot(selection.mode.cpu()))
