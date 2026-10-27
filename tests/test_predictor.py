# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of the JEPA latent predictor g_phi (PR-JEPA Sec. 3.1, Eqs. 3 and 8).

    Z_target  = g_phi(s_x, u, Delta)                                                (Eq. 3)
    Z_target  = (1/n_ens) sum_e s^(e)                                               (Eq. 8)
    sigma_bar = ( (1/(n_ens M d)) sum_e || s^(e) - Z_target ||_F^2 )^(1/2)          (Eq. 8)

With a slot mask, M is the number of real slots M_b of sample b and padded slots are exact zeros.

Every reference is computed independently of the code under test, in float64 on the CPU, from the
module's parameters: explicit loops for Eq. 8, and a hand-written pre-norm transformer (manual layer
norm, softmax attention over the real keys only, erf-GELU) for the trunk documented in predictor.py.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tests.conftest import ATOL, RTOL, accelerator
from text_vpa.configs import load_config
from text_vpa.models.layers import EnsembleLinear
from text_vpa.models.predictor import JEPAPredictor, PredictorOutput, ensemble_mean, ensemble_uncertainty

D = 8            # latent width d
NUM_MODES = 3    # |U|
NUM_SLOTS = 6    # rows of the slot-query matrix Delta
N_ENS = 3        # ensemble heads
NUM_HEADS = 2    # attention heads of the trunk (d_h = 4)

# Per-sample slot selection (order and repeats allowed), and masks with padding in every sample.
SLOT_IDS = torch.tensor([[5, 0, 3, 1], [2, 2, 4, 0], [1, 3, 5, 5]])
SLOT_MASK = torch.tensor([[True, True, True, False], [True, False, True, True], [True, True, True, True]])
CONTEXT_MASK = torch.tensor([[True, True, False, True, True], [True, True, True, True, False],
                             [False, True, True, True, True]])

# float32 forward of a 1-2 layer trunk against the float64 reference: rounding (2^-24 ~ 6e-8 per
# operation) accumulates over ~15 matrix products and reductions and is amplified by the 1/std of the
# layer norms (eight in a two-layer trunk; slot activations start with std ~0.03). The worst deviation
# seen over 10 seeds was 6e-7, so atol = 5e-6 keeps a ~8x margin while any modelling error (O(1e-2) or
# larger at these widths) is still caught.
F32_FORWARD_ATOL = 5e-6
# float64 module against the float64 reference: only reduction-order differences (observed < 1e-15).
F64_TOL = 1e-10
# Standard LayerNorm epsilon (nn.LayerNorm's default; the trunk uses plain LayerNorms). Fixed here rather
# than read from the module; slot activations have variance ~1e-3, so the reference is sensitive to it.
LN_EPS = 1e-5


# ------------------------------------------------------------------------------------------------
# Helpers and independent float64 references
# ------------------------------------------------------------------------------------------------
def _predictor(num_layers: int = 2, head_hidden_dim: int | None = None, n_ens: int = N_ENS) -> JEPAPredictor:
    return JEPAPredictor(latent_dim=D, num_modes=NUM_MODES, num_slots=NUM_SLOTS, n_ens=n_ens,
                         num_layers=num_layers, num_heads=NUM_HEADS, mlp_ratio=2.0,
                         head_hidden_dim=head_hidden_dim).eval()


def _inputs(batch: int = 3, context_len: int = 5) -> tuple[Tensor, Tensor]:
    """s_x [B, M_c, d] and integer modes u [B]."""
    return torch.randn(batch, context_len, D), torch.tensor([2, 0, 1])[:batch]


def _masked_kwargs(padded: bool) -> dict[str, Tensor]:
    return {"slot_ids": SLOT_IDS, "context_mask": CONTEXT_MASK, "slot_mask": SLOT_MASK} if padded else {}


def _f64(t: Tensor) -> Tensor:
    return t.detach().to(torch.float64)


def _linear(x: Tensor, layer: nn.Linear) -> Tensor:
    y = x @ _f64(layer.weight).T
    return y if layer.bias is None else y + _f64(layer.bias)


def _layer_norm(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    """(x - mean) / sqrt(var + eps) * weight + bias over the last axis, with the biased variance."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + LN_EPS) * _f64(norm.weight) + _f64(norm.bias)


def _gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _attention(q: Tensor, k: Tensor, v: Tensor, key_mask: Tensor | None, num_heads: int) -> Tensor:
    """Per head h: softmax(q_h k_h^T / sqrt(d_h)) v_h over the real keys only; heads concatenated."""
    batch, n_q, width = q.shape
    d_h = width // num_heads
    out = torch.zeros(batch, n_q, width, dtype=torch.float64)
    for b in range(batch):
        keys = torch.arange(k.shape[1]) if key_mask is None else torch.nonzero(key_mask[b]).flatten()
        for h in range(num_heads):
            cols = slice(h * d_h, (h + 1) * d_h)
            logits = q[b, :, cols] @ k[b, keys, cols].T / math.sqrt(d_h)
            weights = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
            out[b, :, cols] = (weights / weights.sum(dim=-1, keepdim=True)) @ v[b, keys, cols]
    return out


def _multi_head(attn: nn.Module, query: Tensor, memory: Tensor, key_mask: Tensor | None, num_heads: int) -> Tensor:
    q, k, v = _linear(query, attn.q_proj), _linear(memory, attn.k_proj), _linear(memory, attn.v_proj)
    return _linear(_attention(q, k, v, key_mask, num_heads), attn.out_proj)


def _ensemble_heads(heads: nn.Module, trunk: Tensor, n_ens: int, mlp: bool) -> Tensor:
    """s^(e) = t W_e + b_e (linear heads) or GELU(t W1_e + b1_e) W2_e + b2_e (MLP heads): [n_ens, B, M, d]."""
    outs = []
    for e in range(n_ens):
        if not mlp:
            outs.append(trunk @ _f64(heads.weight[e]) + _f64(heads.bias[e]))
        else:
            first, last = heads[0], heads[-1]
            hidden = _gelu(trunk @ _f64(first.weight[e]) + _f64(first.bias[e]))
            outs.append(hidden @ _f64(last.weight[e]) + _f64(last.bias[e]))
    return torch.stack(outs)


def _eq8_loop(heads: Tensor, slot_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Eq. 8 with explicit loops in float64.

    Z_bm = (1/n_ens) sum_e s^(e)_bm;  sigma_b = sqrt( sum_e sum_{m real} sum_j (s^(e)_bmj - Z_bmj)^2 / (n_ens M_b d) ).
    """
    heads = _f64(heads)
    n_ens, batch, slots, d = heads.shape
    z = torch.zeros(batch, slots, d, dtype=torch.float64)
    for e in range(n_ens):
        z += heads[e]
    z /= n_ens
    sigma = torch.zeros(batch, dtype=torch.float64)
    for b in range(batch):
        total, real_slots = 0.0, 0
        for m in range(slots):
            if slot_mask is not None and not bool(slot_mask[b, m]):
                continue
            real_slots += 1
            for e in range(n_ens):
                for j in range(d):
                    total += float(heads[e, b, m, j] - z[b, m, j]) ** 2
        sigma[b] = math.sqrt(total / (n_ens * real_slots * d))
    return z, sigma


def _reference_forward(p: JEPAPredictor, s_x: Tensor, u: Tensor, slot_ids: Tensor | None = None,
                       context_mask: Tensor | None = None, slot_mask: Tensor | None = None,
                       num_heads: int = NUM_HEADS, n_ens: int = N_ENS,
                       head_hidden_dim: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """g_phi(s_x, u, Delta) in float64 from p's parameters: (Z_target, sigma_bar, head predictions).

    num_heads, n_ens and head_hidden_dim are the values the test constructed p with, not read back from p.
    """
    batch = s_x.shape[0]
    delta, table = _f64(p.slot_queries), _f64(p.mode_embedding.weight)
    mode = u.to(torch.float64) @ table if u.dtype.is_floating_point else table[u]        # ModeEmbed(u)
    queries = delta.expand(batch, -1, -1) if slot_ids is None else delta[slot_ids]       # Delta rows
    x = queries + mode[:, None, :]
    memory = _layer_norm(_f64(s_x), p.context_norm)
    for block in p.blocks:
        h = _layer_norm(x, block.norm_self)
        x = x + _multi_head(block.self_attn, h, h, slot_mask, num_heads)
        x = x + _multi_head(block.cross_attn, _layer_norm(x, block.norm_cross), memory, context_mask, num_heads)
        x = x + _linear(_gelu(_linear(_layer_norm(x, block.norm_mlp), block.mlp.net[0])), block.mlp.net[-1])
    heads = _ensemble_heads(p.heads, _layer_norm(x, p.trunk_norm), n_ens, mlp=head_hidden_dim is not None)
    if slot_mask is not None:
        heads = heads * slot_mask[None, :, :, None].to(torch.float64)                   # padded slots: 0
    z_target, sigma_bar = _eq8_loop(heads, slot_mask)
    return z_target, sigma_bar, heads


def _assert_outputs_equal(a: PredictorOutput, b: PredictorOutput) -> None:
    for name in ("z_target", "sigma_bar", "head_predictions"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0.0, atol=0.0, msg=name)


# ------------------------------------------------------------------------------------------------
# Eq. 8: ensemble mean and uncertainty
# ------------------------------------------------------------------------------------------------
def test_ensemble_mean_matches_eq8_loop() -> None:
    """Eq. 8: Z_target = (1/n_ens) sum_e s^(e), against an explicit loop in float64."""
    heads = torch.randn(5, 3, 4, 6)
    z_ref, _ = _eq8_loop(heads)
    torch.testing.assert_close(ensemble_mean(heads).double(), z_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(ensemble_mean(heads.double()), z_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("masked", [False, True], ids=["no_mask", "slot_mask"])
def test_ensemble_uncertainty_matches_eq8_loop(masked: bool) -> None:
    """Eq. 8: sigma_bar_b = (sum_e ||s^(e) - Z||_F^2 / (n_ens M_b d))^(1/2); padded entries (random here) are ignored."""
    heads = torch.randn(5, 3, 4, 6, dtype=torch.float64)
    slot_mask = torch.tensor([[True, True, False, True], [True, False, False, False],
                              [True, True, True, True]]) if masked else None
    z_ref, sigma_ref = _eq8_loop(heads, slot_mask)
    torch.testing.assert_close(ensemble_uncertainty(heads, z_ref, slot_mask), sigma_ref, rtol=1e-12, atol=1e-12)
    sigma32 = ensemble_uncertainty(heads.float(), z_ref.float(), slot_mask)
    torch.testing.assert_close(sigma32.double(), sigma_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("padded", [False, True], ids=["no_mask", "slot_mask"])
def test_forward_plan_and_uncertainty_are_eq8_of_its_heads(padded: bool) -> None:
    """Eq. 8 inside forward: (z_target, sigma_bar) equal the loop reference applied to head_predictions."""
    p = _predictor()
    s_x, u = _inputs()
    out = p(s_x, u, **_masked_kwargs(padded))
    z_ref, sigma_ref = _eq8_loop(out.head_predictions, SLOT_MASK if padded else None)
    torch.testing.assert_close(out.z_target.double(), z_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(out.sigma_bar.double(), sigma_ref, rtol=RTOL, atol=ATOL)


def test_ensemble_mean_of_identical_heads_is_exact() -> None:
    """Eq. 8, identical heads: Z_target equals the common head bit for bit, so sigma_bar is exactly 0."""
    head = torch.randn(3, 4, 6)
    heads = head.expand(5, -1, -1, -1).clone()
    assert not torch.equal(heads.sum(dim=0) / 5, head), "precondition: the plain float32 mean is inexact for this data"
    z = ensemble_mean(heads)
    assert torch.equal(z, head)
    assert torch.equal(ensemble_uncertainty(heads, z), torch.zeros(3))


@pytest.mark.parametrize("head_hidden_dim", [None, 12], ids=["linear_heads", "mlp_heads"])
@pytest.mark.parametrize("padded", [False, True], ids=["no_mask", "slot_mask"])
def test_identical_heads_give_exactly_zero_uncertainty(head_hidden_dim: int | None, padded: bool) -> None:
    """Eq. 8: with head 1's weights copied into every member, sigma_bar = 0 exactly and Z_target = s^(1)."""
    p = _predictor(head_hidden_dim=head_hidden_dim)
    layers = [p.heads] if head_hidden_dim is None else [p.heads[0], p.heads[-1]]
    with torch.no_grad():
        for layer in layers:
            for param in (layer.weight, layer.bias):
                param.copy_(param[1:2].clone().expand_as(param))
    s_x, u = _inputs()
    out = p(s_x, u, **_masked_kwargs(padded))
    assert torch.equal(out.sigma_bar, torch.zeros(3))
    assert torch.equal(out.z_target, out.head_predictions[1])


@pytest.mark.parametrize("padded", [False, True], ids=["no_mask", "slot_mask"])
def test_eq8_gradients_match_closed_form(padded: bool) -> None:
    """Eq. 8 derivatives: dZ/ds^(e) = 1/n_ens and dsigma_b/ds^(e)_bm = (s^(e)_bm - Z_bm) / (n_ens M_b d sigma_b)
    at real slots, 0 at padded ones (the path through Z vanishes because sum_e (s^(e) - Z) = 0)."""
    n_ens, d = 4, 6
    heads = torch.randn(n_ens, 3, 5, d, dtype=torch.float64, requires_grad=True)
    slot_mask = torch.tensor([[True, False, True, True, False], [True, True, True, True, True],
                              [False, False, True, False, False]]) if padded else None
    probe = torch.randn(3, 5, d, dtype=torch.float64)
    (grad_mean,) = torch.autograd.grad((ensemble_mean(heads) * probe).sum(), heads)
    torch.testing.assert_close(grad_mean, (probe / n_ens).expand_as(heads), rtol=1e-12, atol=1e-12)

    (grad_sigma,) = torch.autograd.grad(ensemble_uncertainty(heads, ensemble_mean(heads), slot_mask).sum(), heads)
    z_ref, sigma_ref = _eq8_loop(heads, slot_mask)
    real = torch.ones(3, 5, dtype=torch.bool) if slot_mask is None else slot_mask
    count = real.sum(dim=1).to(torch.float64)
    expected = (_f64(heads) - z_ref) / (n_ens * d * count * sigma_ref)[:, None, None]
    expected = expected * real[None, :, :, None].to(torch.float64)
    torch.testing.assert_close(grad_sigma, expected, rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------------------------------------
# Full forward against a hand-written float64 reference
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("head_hidden_dim", [None, 12], ids=["linear_heads", "mlp_heads"])
@pytest.mark.parametrize("padded", [False, True], ids=["all_slots", "slot_ids_and_masks"])
def test_forward_matches_float64_reference(num_layers: int, head_hidden_dim: int | None, padded: bool) -> None:
    """Eqs. 3 and 8: z_target, head_predictions and sigma_bar of g_phi against the hand-written reference."""
    p = _predictor(num_layers=num_layers, head_hidden_dim=head_hidden_dim)
    s_x, u = _inputs()
    kwargs = _masked_kwargs(padded)
    z_ref, sigma_ref, heads_ref = _reference_forward(p, s_x, u, **kwargs, head_hidden_dim=head_hidden_dim)

    out64 = copy.deepcopy(p).double()(s_x.double(), u, **kwargs)
    torch.testing.assert_close(out64.head_predictions, heads_ref, rtol=F64_TOL, atol=F64_TOL)
    torch.testing.assert_close(out64.z_target, z_ref, rtol=F64_TOL, atol=F64_TOL)
    torch.testing.assert_close(out64.sigma_bar, sigma_ref, rtol=F64_TOL, atol=F64_TOL)

    out32 = p(s_x, u, **kwargs)
    torch.testing.assert_close(out32.head_predictions.double(), heads_ref, rtol=RTOL, atol=F32_FORWARD_ATOL)
    torch.testing.assert_close(out32.z_target.double(), z_ref, rtol=RTOL, atol=F32_FORWARD_ATOL)
    torch.testing.assert_close(out32.sigma_bar.double(), sigma_ref, rtol=RTOL, atol=F32_FORWARD_ATOL)


def test_soft_mode_vector_is_contracted_with_mode_embedding() -> None:
    """A soft u in the simplex enters as sum_k u_k ModeEmbed_k (the straight-through relaxation of pass 2)."""
    p = copy.deepcopy(_predictor()).double()
    s_x, _ = _inputs()
    u_soft = torch.softmax(torch.randn(3, NUM_MODES, dtype=torch.float64), dim=-1)
    z_ref, sigma_ref, heads_ref = _reference_forward(p, s_x, u_soft, SLOT_IDS, CONTEXT_MASK, SLOT_MASK)
    out = p(s_x.double(), u_soft, SLOT_IDS, CONTEXT_MASK, SLOT_MASK)
    torch.testing.assert_close(out.head_predictions, heads_ref, rtol=F64_TOL, atol=F64_TOL)
    torch.testing.assert_close(out.z_target, z_ref, rtol=F64_TOL, atol=F64_TOL)
    torch.testing.assert_close(out.sigma_bar, sigma_ref, rtol=F64_TOL, atol=F64_TOL)


def test_one_hot_float_mode_equals_integer_mode() -> None:
    """A one-hot float u selects exactly ModeEmbed(u), so every output equals the integer-mode output."""
    p = _predictor()
    s_x, u = _inputs()
    by_index = p(s_x, u, SLOT_IDS, CONTEXT_MASK, SLOT_MASK)
    by_one_hot = p(s_x, F.one_hot(u, NUM_MODES).float(), SLOT_IDS, CONTEXT_MASK, SLOT_MASK)
    _assert_outputs_equal(by_one_hot, by_index)


# ------------------------------------------------------------------------------------------------
# Slot queries Delta and slot_ids
# ------------------------------------------------------------------------------------------------
def test_slot_ids_none_uses_every_query_in_order() -> None:
    """slot_ids=None means Delta itself: all num_slots queries in order, i.e. slot_ids = arange(num_slots)."""
    p = _predictor()
    s_x, u = _inputs()
    default = p(s_x, u)
    explicit = p(s_x, u, slot_ids=torch.arange(NUM_SLOTS).expand(3, -1))
    assert default.z_target.shape == (3, NUM_SLOTS, D)
    assert default.head_predictions.shape == (N_ENS, 3, NUM_SLOTS, D)
    _assert_outputs_equal(default, explicit)
    assert torch.equal(default.slot_mask, torch.ones(3, NUM_SLOTS, dtype=torch.bool))


def test_slot_ids_reorder_the_plan_rows() -> None:
    """Slots have no position other than their query Delta_m, so permuting slot_ids permutes the plan rows."""
    p = _predictor()
    s_x, u = _inputs()
    perm = torch.randperm(NUM_SLOTS)
    default = p(s_x, u)
    permuted = p(s_x, u, slot_ids=perm.expand(3, -1))
    torch.testing.assert_close(permuted.head_predictions, default.head_predictions[:, :, perm], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(permuted.sigma_bar, default.sigma_bar, rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------------------------------------
# Padding
# ------------------------------------------------------------------------------------------------
def test_padded_slots_are_exact_zeros() -> None:
    """Padded slots of z_target and of every head are exact zeros; the returned mask is the given one."""
    p = _predictor()
    s_x, u = _inputs()
    out = p(s_x, u, SLOT_IDS, CONTEXT_MASK, SLOT_MASK)
    assert torch.equal(out.slot_mask, SLOT_MASK)
    assert torch.equal(out.head_predictions[:, ~SLOT_MASK], torch.zeros(N_ENS, int((~SLOT_MASK).sum()), D))
    assert torch.equal(out.z_target[~SLOT_MASK], torch.zeros(int((~SLOT_MASK).sum()), D))
    assert bool((out.z_target[SLOT_MASK].abs().sum(dim=-1) > 0).all())


def test_outputs_ignore_padded_context_rows() -> None:
    """Padded rows of s_x are excluded from the cross-attention keys: replacing them changes nothing."""
    p = _predictor()
    s_x, u = _inputs()
    noisy = s_x.clone()
    noisy[~CONTEXT_MASK] = 1e3 * torch.randn(int((~CONTEXT_MASK).sum()), D)
    _assert_outputs_equal(p(noisy, u, SLOT_IDS, CONTEXT_MASK, SLOT_MASK), p(s_x, u, SLOT_IDS, CONTEXT_MASK, SLOT_MASK))


def test_outputs_ignore_padded_slot_ids() -> None:
    """Padded slots are excluded from the self-attention keys: their slot ids do not affect real slots."""
    p = _predictor()
    s_x, u = _inputs()
    other_ids = SLOT_IDS.clone()
    other_ids[~SLOT_MASK] = torch.tensor([4, 1])
    assert not torch.equal(other_ids, SLOT_IDS)
    _assert_outputs_equal(p(s_x, u, other_ids, CONTEXT_MASK, SLOT_MASK), p(s_x, u, SLOT_IDS, CONTEXT_MASK, SLOT_MASK))


def test_outputs_ignore_queries_used_only_by_padded_slots() -> None:
    """Changing a Delta row that only padded slots use leaves every output unchanged."""
    p = _predictor()
    s_x, u = _inputs(batch=2)
    slot_ids = torch.tensor([[0, 1, 2, 5], [3, 4, 5, 4]])          # row 5 appears only at padded slots
    slot_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    before = p(s_x, u, slot_ids, slot_mask=slot_mask)
    with torch.no_grad():
        p.slot_queries[5] = 10.0 * torch.randn(D)
    _assert_outputs_equal(p(s_x, u, slot_ids, slot_mask=slot_mask), before)


# ------------------------------------------------------------------------------------------------
# Gradients and ensemble heads
# ------------------------------------------------------------------------------------------------
def test_gradients_reach_trunk_and_every_head() -> None:
    """A loss on (Z_target, sigma_bar) back-propagates into Delta, ModeEmbed, the trunk and each of the n_ens heads."""
    p = _predictor()
    s_x, _ = _inputs()
    u = torch.tensor([0, 2, 0])                                     # mode 1 unused
    slot_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 2, 2]])  # Delta rows 4 and 5 unused
    out = p(s_x, u, slot_ids)
    ((out.z_target * torch.randn_like(out.z_target)).sum() + out.sigma_bar.sum()).backward()

    for name, param in p.named_parameters():
        assert param.grad is not None and bool(torch.isfinite(param.grad).all()), name
        # Softmax is shift-invariant per query, so key biases get no gradient in exact arithmetic.
        if not name.startswith(("slot_queries", "mode_embedding", "heads")) and not name.endswith("k_proj.bias"):
            assert param.grad.abs().sum() > 0, name
    for e in range(N_ENS):
        assert p.heads.weight.grad[e].abs().sum() > 0 and p.heads.bias.grad[e].abs().sum() > 0, e
    used = torch.tensor([True, True, True, True, False, False])
    assert bool((p.slot_queries.grad[used].abs().sum(dim=1) > 0).all())
    assert torch.equal(p.slot_queries.grad[~used], torch.zeros(2, D))
    assert bool((p.mode_embedding.weight.grad[[0, 2]].abs().sum(dim=1) > 0).all())
    assert torch.equal(p.mode_embedding.weight.grad[1], torch.zeros(D))


@pytest.mark.parametrize("head_hidden_dim", [None, 24], ids=["linear_heads", "mlp_heads"])
def test_heads_are_independently_initialised(head_hidden_dim: int | None) -> None:
    """Deep-ensemble heads (Sec. 3.1) start from independent draws: pairwise different, uncorrelated weights.

    For independent zero-mean entries the sample correlation of n pairs has standard deviation ~1/sqrt(n);
    |r| < 5/sqrt(n) also rejects near-copies such as W_e = W_0 + small noise.
    """
    n_ens = 5
    p = JEPAPredictor(latent_dim=32, num_modes=4, num_slots=16, n_ens=n_ens, num_layers=1, num_heads=4,
                      head_hidden_dim=head_hidden_dim).eval()
    layers = [p.heads] if head_hidden_dim is None else [p.heads[0], p.heads[-1]]
    for layer in layers:
        weights, biases = layer.weight.detach().reshape(n_ens, -1), layer.bias.detach().reshape(n_ens, -1)
        bound = 5.0 / math.sqrt(weights.shape[1])
        for e in range(n_ens):
            for f in range(e + 1, n_ens):
                assert not torch.allclose(weights[e], weights[f]) and not torch.allclose(biases[e], biases[f])
                r = torch.corrcoef(torch.stack([weights[e], weights[f]]))[0, 1]
                assert abs(float(r)) < bound, (e, f, float(r))
    out = p(torch.randn(2, 7, 32), torch.tensor([0, 3]))
    assert bool((out.sigma_bar > 0).all())


def _trunk_features(p: JEPAPredictor, s_x: Tensor, u: Tensor) -> tuple[PredictorOutput, Tensor]:
    """Forward pass plus the trunk output LN(x) [B, M, d] that feeds the heads (captured by a hook)."""
    captured: list[Tensor] = []
    handle = p.trunk_norm.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    try:
        out = p(s_x, u)
    finally:
        handle.remove()
    return out, _f64(captured[0])


def test_default_heads_are_linear_maps_of_the_trunk_output() -> None:
    """Default heads: n_ens independent linear maps d -> d, s^(e) = t W_e + b_e on the shared trunk output t."""
    p = _predictor()
    assert isinstance(p.heads, EnsembleLinear)
    assert p.heads.weight.shape == (N_ENS, D, D)
    s_x, u = _inputs()
    out, trunk = _trunk_features(p, s_x, u)
    for e in range(N_ENS):
        expected = trunk @ _f64(p.heads.weight[e]) + _f64(p.heads.bias[e])          # W_e in R^{d x d}
        torch.testing.assert_close(out.head_predictions[e].double(), expected, rtol=RTOL, atol=ATOL)


def test_head_hidden_dim_gives_two_layer_gelu_heads() -> None:
    """head_hidden_dim = h: s^(e) = GELU(t W1_e + b1_e) W2_e + b2_e with W1_e in R^{d x h}, W2_e in R^{h x d}."""
    p = _predictor(head_hidden_dim=12)
    assert isinstance(p.heads, nn.Sequential) and len(p.heads) == 3
    first, activation, last = p.heads
    assert isinstance(first, EnsembleLinear) and isinstance(last, EnsembleLinear) and isinstance(activation, nn.GELU)
    assert first.weight.shape == (N_ENS, D, 12) and last.weight.shape == (N_ENS, 12, D)
    s_x, u = _inputs()
    out, trunk = _trunk_features(p, s_x, u)
    for e in range(N_ENS):
        expected = _gelu(trunk @ _f64(first.weight[e]) + _f64(first.bias[e])) @ _f64(last.weight[e]) + _f64(last.bias[e])
        torch.testing.assert_close(out.head_predictions[e].double(), expected, rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("overrides", [
    pytest.param({"n_ens": 1}, id="n_ens_1"),
    pytest.param({"num_layers": 0}, id="no_layers"),
    pytest.param({"num_modes": 0}, id="no_modes"),
    pytest.param({"num_slots": 0}, id="no_slots"),
    pytest.param({"num_heads": 3}, id="heads_not_dividing_d"),
])
def test_invalid_hyperparameters_raise_value_error(overrides: dict[str, int]) -> None:
    """n_ens >= 2 (Eq. 8 needs a spread), at least one layer, mode and slot, and heads dividing d."""
    kwargs = {"latent_dim": D, "num_modes": NUM_MODES, "num_slots": NUM_SLOTS, "n_ens": N_ENS, "num_layers": 1,
              "num_heads": NUM_HEADS}
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        JEPAPredictor(**kwargs)


_FORWARD_ERRORS = {
    "s_x_width": lambda s_x, u: {"s_x": torch.randn(3, 5, D + 1)},
    "s_x_2d": lambda s_x, u: {"s_x": torch.randn(3, D)},
    "slot_ids_negative": lambda s_x, u: {"slot_ids": torch.tensor([[0, 1], [2, -1], [3, 4]])},
    "slot_ids_too_large": lambda s_x, u: {"slot_ids": torch.tensor([[0, 1], [2, NUM_SLOTS], [3, 4]])},
    "slot_ids_float": lambda s_x, u: {"slot_ids": torch.zeros(3, 2)},
    "slot_ids_batch": lambda s_x, u: {"slot_ids": torch.zeros(2, 2, dtype=torch.long)},
    "slot_ids_1d": lambda s_x, u: {"slot_ids": torch.zeros(3, dtype=torch.long)},
    "context_mask_all_false_row": lambda s_x, u: {"context_mask": torch.tensor([[True] * 5, [False] * 5, [True] * 5])},
    "slot_mask_all_false_row": lambda s_x, u: {"slot_mask": torch.tensor([[True] * 6, [True] * 6, [False] * 6])},
    "context_mask_int": lambda s_x, u: {"context_mask": torch.ones(3, 5, dtype=torch.long)},
    "context_mask_float": lambda s_x, u: {"context_mask": torch.ones(3, 5)},
    "slot_mask_uint8": lambda s_x, u: {"slot_mask": torch.ones(3, NUM_SLOTS, dtype=torch.uint8)},
    "context_mask_shape": lambda s_x, u: {"context_mask": torch.ones(3, 4, dtype=torch.bool)},
    "slot_mask_shape": lambda s_x, u: {"slot_mask": torch.ones(3, NUM_SLOTS + 1, dtype=torch.bool)},
    "u_batch": lambda s_x, u: {"u": u[:2]},
    "u_soft_width": lambda s_x, u: {"u": torch.ones(3, NUM_MODES + 1)},
    "u_int_2d": lambda s_x, u: {"u": u[:, None]},
}


@pytest.mark.parametrize("case", list(_FORWARD_ERRORS))
def test_invalid_forward_inputs_raise_value_error(case: str) -> None:
    """Wrong s_x width or rank, out-of-range or malformed slot_ids, non-bool or empty masks, bad u: ValueError."""
    p = _predictor()
    s_x, u = _inputs()
    kwargs = {"s_x": s_x, "u": u}
    kwargs.update(_FORWARD_ERRORS[case](s_x, u))
    with pytest.raises(ValueError):
        p(**kwargs)


def test_invalid_eq8_inputs_raise_value_error() -> None:
    """Eq. 8 helpers reject inputs without an ensemble axis or with mismatched shapes."""
    with pytest.raises(ValueError):
        ensemble_mean(torch.randn(4))
    with pytest.raises(ValueError):
        ensemble_uncertainty(torch.randn(3, 2, 4, 5), torch.randn(2, 4, 6))
    with pytest.raises(ValueError):
        ensemble_uncertainty(torch.randn(2, 4, 5), torch.randn(2, 4, 5))


def test_ensemble_uncertainty_rejects_a_sample_without_real_slots() -> None:
    """Eq. 8 with M_b = 0 has no value; the helper should refuse it like forward() does."""
    heads = torch.randn(3, 2, 4, 5)
    slot_mask = torch.tensor([[True, True, False, True], [False, False, False, False]])
    with pytest.raises(ValueError):
        ensemble_uncertainty(heads, ensemble_mean(heads), slot_mask)


@pytest.mark.parametrize(("dims", "slot_ids"), [
    pytest.param((D, NUM_SLOTS, 3), torch.ones(3, 4, dtype=torch.bool), id="index_error"),
    # B == num_slots and M' == d: boolean indexing succeeds and silently returns M' = 1 slot.
    pytest.param((4, 2, 2), torch.tensor([[True, True, False, False], [False, False, True, True]]), id="silent"),
])
def test_bool_slot_ids_raise_value_error(dims: tuple[int, int, int], slot_ids: Tensor) -> None:
    """slot_ids index Delta (Eq. 3) and must be integers; a boolean tensor must raise ValueError."""
    d, num_slots, batch = dims
    p = JEPAPredictor(latent_dim=d, num_modes=NUM_MODES, num_slots=num_slots, n_ens=2, num_layers=1,
                      num_heads=2).eval()
    with pytest.raises(ValueError):
        p(torch.randn(batch, 3, d), torch.zeros(batch, dtype=torch.long), slot_ids=slot_ids)


def test_empty_slot_set_raises_value_error() -> None:
    """A plan with M' = 0 slots has no Eq. 8 uncertainty; it must be rejected, not return NaN."""
    p = _predictor()
    s_x, u = _inputs()
    with pytest.raises(ValueError):
        p(s_x, u, slot_ids=torch.zeros(3, 0, dtype=torch.long))


def test_empty_context_raises_value_error() -> None:
    """Cross-attention to s_x (Eq. 3) needs at least one context token; M_c = 0 must be rejected."""
    p = _predictor()
    _, u = _inputs()
    with pytest.raises(ValueError):
        p(torch.randn(3, 0, D), u)


# ------------------------------------------------------------------------------------------------
# Configuration and devices
# ------------------------------------------------------------------------------------------------
def test_from_config_builds_the_tiny_predictor(tiny_config) -> None:
    """from_config maps planner.latent_dim, num_slots, num_modes and predictor.* onto g_phi."""
    planner = tiny_config.planner
    p = JEPAPredictor.from_config(planner).eval()
    d = planner.latent_dim
    assert (p.latent_dim, p.num_modes, p.num_slots, p.n_ens) == (d, planner.num_modes, planner.num_slots,
                                                                 planner.predictor.n_ens)
    assert p.slot_queries.shape == (planner.num_slots, d)
    assert p.mode_embedding.weight.shape == (planner.num_modes, d)
    assert len(p.blocks) == planner.predictor.num_layers
    assert p.blocks[0].self_attn.num_heads == planner.predictor.num_heads
    assert p.blocks[0].mlp.net[0].out_features == round(d * planner.predictor.mlp_ratio)
    assert isinstance(p.heads, EnsembleLinear) and p.heads.weight.shape == (planner.predictor.n_ens, d, d)
    out = p(torch.randn(2, 7, d), torch.tensor([0, planner.num_modes - 1]))
    assert out.z_target.shape == (2, planner.num_slots, d)
    assert out.head_predictions.shape == (planner.predictor.n_ens, 2, planner.num_slots, d)
    assert out.sigma_bar.shape == (2,)


def test_from_config_passes_predictor_settings() -> None:
    """n_ens, num_layers and head_hidden_dim overrides reach the module."""
    cfg = load_config(["planner=tiny", "data=tiny", "realizer=tiny", "planner.predictor.n_ens=3",
                       "planner.predictor.head_hidden_dim=24", "planner.predictor.num_layers=1"])
    p = JEPAPredictor.from_config(cfg.planner)
    assert p.n_ens == 3 and len(p.blocks) == 1
    assert isinstance(p.heads, nn.Sequential)
    assert p.heads[0].weight.shape == (3, cfg.planner.latent_dim, 24)
    assert p.heads[-1].weight.shape == (3, 24, cfg.planner.latent_dim)


@pytest.mark.skipif(accelerator() is None, reason="no CUDA/ROCm/MPS device")
def test_forward_on_accelerator_matches_cpu_reference() -> None:
    """Same plan on the accelerator. Tolerance 1e-4: device kernels reduce in other orders (and may use TF32)."""
    device = accelerator()
    p = _predictor()
    s_x, u = _inputs()
    z_ref, sigma_ref, heads_ref = _reference_forward(p, s_x, u, SLOT_IDS, CONTEXT_MASK, SLOT_MASK)
    on_device = copy.deepcopy(p).to(device)
    out = on_device(s_x.to(device), u.to(device), SLOT_IDS.to(device), CONTEXT_MASK.to(device), SLOT_MASK.to(device))
    assert out.z_target.device.type == device.type
    torch.testing.assert_close(out.head_predictions.cpu().double(), heads_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out.z_target.cpu().double(), z_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out.sigma_bar.cpu().double(), sigma_ref, rtol=1e-4, atol=1e-4)
