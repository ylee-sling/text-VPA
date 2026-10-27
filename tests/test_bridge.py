# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of the cross-attention bridge (PR-JEPA Sec. 3.2, Eq. 10).

    Z~ = [Z_target ; s_x ; e(u)] / gamma,   M~ = M + M_c + 1 conditioning tokens         (Sec. 3.2)
    CA_l(H, Z~) = softmax( (H W_Q^l)(Z~ W_K^l)^T / sqrt(d_k) ) Z~ W_V^l                   (Eq. 10)

implemented multi-head (per head d_h = d_r / H, heads concatenated, then W_O^l). K_l = Z~ W_K^l and
V_l = Z~ W_V^l are computed once per generation and reused at all K steps; they cost 2 L_r M~ d_k b
bytes, independent of N (48 MiB for M~ = 256, L_r = 24, d_k = 2048 and bf16). W_K^l and W_V^l are
spectrally normalised; in training the conditioning is replaced by a learned null token with
probability 0.1.

References are computed independently in float64 on the CPU from the module's effective weights
(``module.weight``, which the spectral-norm parametrization computes): explicit per-head softmax over
the real keys, with K and V recomputed from Z~ at every call.
"""

from __future__ import annotations

import copy
import math
from collections import Counter

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import parametrize

from tests.conftest import ATOL, RTOL, accelerator
from text_vpa.configs import load_config
from text_vpa.models.bridge import BridgeConditioning, BridgeKVCache, CrossAttentionBridge

D = 8              # latent width d of Z_target, s_x and e(u)
D_R = 16           # realizer width d_r (= d_k)
L_R = 3            # realizer layers with cross-attention
NUM_HEADS = 4      # d_h = 4
NUM_MODES = 3      # |U|
GAMMA = 2.0        # a power of two, so the 1/gamma scaling is exact in floating point
B, M, M_C, N = 2, 3, 4, 6
M_TILDE = M + M_C + 1
K_STEPS = 4        # realizer passes per generation

SLOT_MASK = torch.tensor([[True, False, True], [True, True, True]])
CONTEXT_MASK = torch.tensor([[True, True, True, False], [False, True, True, True]])


# ------------------------------------------------------------------------------------------------
# Helpers and independent float64 references
# ------------------------------------------------------------------------------------------------
def _bridge(**overrides: object) -> CrossAttentionBridge:
    kwargs: dict[str, object] = {"latent_dim": D, "hidden_dim": D_R, "num_layers": L_R, "num_heads": NUM_HEADS,
                                 "num_modes": NUM_MODES, "gamma": GAMMA}
    kwargs.update(overrides)
    return CrossAttentionBridge(**kwargs).eval()


def _plan_inputs(batch: int = B) -> tuple[Tensor, Tensor, Tensor]:
    """Z_target [B, M, d], s_x [B, M_c, d] and integer modes u [B]."""
    return torch.randn(batch, M, D), torch.randn(batch, M_C, D), torch.tensor([1, 2, 0, 1])[:batch]


def _f64(t: Tensor) -> Tensor:
    return t.detach().to(torch.float64)


def _projections(bridge: CrossAttentionBridge) -> list[nn.Module]:
    return [*bridge.key_proj, *bridge.value_proj]


def _conditioning_reference(bridge: CrossAttentionBridge, z_target: Tensor, s_x: Tensor, u: Tensor,
                            slot_mask: Tensor | None = None, context_mask: Tensor | None = None,
                            gamma: float = GAMMA) -> tuple[Tensor, Tensor]:
    """Z~ = [Z_target; s_x; e(u)] / gamma in float64 (differentiable in z_target and s_x), and its key mask.

    gamma is the value the test constructed the bridge with, not read back from the module.
    """
    batch = z_target.shape[0]
    table = _f64(bridge.mode_embedding.weight)
    mode = u.to(torch.float64) @ table if u.dtype.is_floating_point else table[u]
    tokens = torch.cat([z_target.to(torch.float64), s_x.to(torch.float64), mode[:, None, :]], dim=1) / gamma
    ones = torch.ones(batch, 1, dtype=torch.bool)
    slot_mask = ones.expand(batch, z_target.shape[1]) if slot_mask is None else slot_mask
    context_mask = ones.expand(batch, s_x.shape[1]) if context_mask is None else context_mask
    return tokens, torch.cat([slot_mask, context_mask, ones], dim=1)


def _softmax_rows(logits: Tensor) -> Tensor:
    weights = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    return weights / weights.sum(dim=-1, keepdim=True)


def _eq10_reference(bridge: CrossAttentionBridge, layer: int, hidden: Tensor, tokens: Tensor,
                    key_mask: Tensor, num_heads: int = NUM_HEADS) -> Tensor:
    """Eq. 10 per head over the real keys, heads concatenated, then W_O; K and V recomputed from Z~.

    Paper convention (right multiplication): Q = H W_Q with W_Q = query_proj.weight^T, and so on.
    num_heads is the value the test constructed the bridge with, not read back from the module.
    """
    w_q = _f64(bridge.query_proj[layer].weight).T
    w_k = _f64(bridge.key_proj[layer].weight).T          # effective, spectrally normalised weight
    w_v = _f64(bridge.value_proj[layer].weight).T
    w_o = _f64(bridge.out_proj[layer].weight).T
    q, k, v = hidden.to(torch.float64) @ w_q, tokens @ w_k, tokens @ w_v
    d_h = hidden.shape[-1] // num_heads
    samples = []
    for b in range(q.shape[0]):
        heads = []
        for h in range(num_heads):
            cols = slice(h * d_h, (h + 1) * d_h)
            logits = q[b, :, cols] @ k[b, :, cols].T / math.sqrt(d_h)
            logits = logits.masked_fill(~key_mask[b][None, :], float("-inf"))
            heads.append(_softmax_rows(logits) @ v[b, :, cols])
        samples.append(torch.cat(heads, dim=-1))
    return torch.stack(samples) @ w_o


def _exact_conditioning(bridge: CrossAttentionBridge, z_target: Tensor, s_x: Tensor, u: Tensor) -> Tensor:
    """[Z_target; s_x; E[u]] / GAMMA in the input dtype: bit-exact, since dividing by 2 is exact in floating point."""
    return torch.cat([z_target, s_x, bridge.mode_embedding.weight.detach()[u][:, None, :]], dim=1) / GAMMA


def _null_output_reference(bridge: CrossAttentionBridge, layer: int, null_token: Tensor) -> Tensor:
    """Attention over the single null key has weight 1 in every head: CA = (null W_V) W_O for every query."""
    return (null_token @ _f64(bridge.value_proj[layer].weight).T) @ _f64(bridge.out_proj[layer].weight).T


# ------------------------------------------------------------------------------------------------
# Conditioning tokens Z~ (Sec. 3.2)
# ------------------------------------------------------------------------------------------------
def test_conditioning_concatenates_plan_context_and_mode_scaled_by_inverse_gamma() -> None:
    """Z~ = [Z_target; s_x; e(u)] / gamma along the token axis; gamma = 2 makes the scaling exact."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    cond = bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK)
    assert cond.tokens.shape == (B, M_TILDE, D)
    assert torch.equal(cond.tokens[:, :M], z / GAMMA)
    assert torch.equal(cond.tokens[:, M:M + M_C], s / GAMMA)
    assert torch.equal(cond.tokens[:, M + M_C], bridge.mode_embedding(u).detach() / GAMMA)    # e(u) / gamma
    assert torch.equal(cond.mask, torch.cat([SLOT_MASK, CONTEXT_MASK, torch.ones(B, 1, dtype=torch.bool)], dim=1))
    assert torch.equal(cond.dropped, torch.zeros(B, dtype=torch.bool))


@pytest.mark.parametrize("gamma", [0.3, 3.0])
def test_conditioning_scale_matches_float64_reference(gamma: float) -> None:
    """The constant 1/gamma (not a normalisation) for gamma that is not a power of two."""
    bridge = _bridge(gamma=gamma)
    z, s, u = _plan_inputs()
    tokens, mask = _conditioning_reference(bridge, z, s, u, SLOT_MASK, CONTEXT_MASK, gamma=gamma)
    cond = bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK)
    torch.testing.assert_close(cond.tokens.double(), tokens, rtol=RTOL, atol=ATOL)
    assert torch.equal(cond.mask, mask)


def test_one_hot_mode_vector_gives_the_same_conditioning() -> None:
    """A (straight-through) one-hot u contracted with e's table equals e(u) for the integer mode."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    by_index = bridge.build_conditioning(z, s, u)
    by_one_hot = bridge.build_conditioning(z, s, F.one_hot(u, NUM_MODES).float())
    assert torch.equal(by_one_hot.tokens, by_index.tokens)
    assert torch.equal(by_one_hot.mask, by_index.mask)


def test_conditioning_without_masks_marks_every_token_real() -> None:
    """Without masks all M~ = M + M_c + 1 conditioning tokens are keys."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    cond = bridge.build_conditioning(z, s, u)
    assert cond.mask.dtype == torch.bool and torch.equal(cond.mask, torch.ones(B, M_TILDE, dtype=torch.bool))


# ------------------------------------------------------------------------------------------------
# Eq. 10
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_heads", [1, NUM_HEADS])
def test_cross_attention_matches_eq10_reference_with_masked_keys(num_heads: int) -> None:
    """Eq. 10 for every layer l, with padded plan slots and context rows excluded from the keys."""
    bridge = _bridge(num_heads=num_heads)
    z, s, u = _plan_inputs()
    cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK))
    tokens, mask = _conditioning_reference(bridge, z, s, u, SLOT_MASK, CONTEXT_MASK)
    hidden = torch.randn(B, N, D_R)
    for layer in range(L_R):
        out = bridge.cross_attention(layer, hidden, cache)
        assert out.shape == (B, N, D_R)
        torch.testing.assert_close(out.double(), _eq10_reference(bridge, layer, hidden, tokens, mask, num_heads),
                                   rtol=RTOL, atol=ATOL)


def test_single_head_cross_attention_is_eq10_verbatim() -> None:
    """One head (d_h = d_k = d_r) and W_O = I: the output is softmax((H W_Q)(Z~ W_K)^T / sqrt(d_k)) Z~ W_V."""
    bridge = _bridge(num_heads=1)
    with torch.no_grad():
        for proj in bridge.out_proj:
            proj.weight.copy_(torch.eye(D_R))
    z, s, u = _plan_inputs()
    cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u))
    z_tilde = torch.cat([_f64(z), _f64(s), _f64(bridge.mode_embedding.weight)[u][:, None]], dim=1) / GAMMA
    hidden = torch.randn(B, N, D_R)
    for layer in range(L_R):
        w_q, w_k, w_v = (_f64(p[layer].weight).T for p in (bridge.query_proj, bridge.key_proj, bridge.value_proj))
        logits = (hidden.double() @ w_q) @ (z_tilde @ w_k).transpose(1, 2) / math.sqrt(D_R)
        expected = _softmax_rows(logits) @ (z_tilde @ w_v)
        torch.testing.assert_close(bridge.cross_attention(layer, hidden, cache).double(), expected,
                                   rtol=RTOL, atol=ATOL)


def test_masked_conditioning_rows_do_not_affect_the_output() -> None:
    """Padded slots of Z_target and padded rows of s_x are masked keys: their content is irrelevant."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    hidden = torch.randn(B, N, D_R)
    z_noisy, s_noisy = z.clone(), s.clone()
    z_noisy[~SLOT_MASK] = 1e3 * torch.randn(int((~SLOT_MASK).sum()), D)
    s_noisy[~CONTEXT_MASK] = 1e3 * torch.randn(int((~CONTEXT_MASK).sum()), D)
    clean = bridge.precompute_kv(bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK))
    noisy = bridge.precompute_kv(bridge.build_conditioning(z_noisy, s_noisy, u, SLOT_MASK, CONTEXT_MASK))
    for layer in range(L_R):
        assert torch.equal(bridge.cross_attention(layer, hidden, noisy), bridge.cross_attention(layer, hidden, clean))


def test_forward_is_conditioning_then_key_value_projection() -> None:
    """forward = precompute_kv(build_conditioning(...)), the part of pass 4 that runs once."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    cache = bridge(z, s, u, SLOT_MASK, CONTEXT_MASK)
    expected = bridge.precompute_kv(bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK))
    assert cache.num_layers == L_R
    for got, want in zip((*cache.keys, *cache.values, cache.mask), (*expected.keys, *expected.values, expected.mask),
                         strict=True):
        assert torch.equal(got, want)


# ------------------------------------------------------------------------------------------------
# Spectral normalisation of W_K^l and W_V^l
# ------------------------------------------------------------------------------------------------
def test_conditioning_projections_are_bias_free() -> None:
    """Eq. 10 has no bias: Z~ -> Z~ W_K^l and Z~ -> Z~ W_V^l are linear (so their Lipschitz constant is ||W||_2)."""
    bridge = _bridge()
    for proj in (*bridge.key_proj, *bridge.value_proj, *bridge.query_proj, *bridge.out_proj):
        assert proj.bias is None
    for proj in _projections(bridge):
        assert parametrize.is_parametrized(proj, "weight")


def test_spectral_norm_divides_by_a_power_iteration_lower_estimate_of_sigma_max() -> None:
    """As constructed: W_eff = W / s_hat with one scalar s_hat = u^T W v <= sigma_max(W) (unit u, v), so
    sigma_max(W_eff) >= 1; power iteration approaches sigma_max from below (tolerance: float32 rounding)."""
    bridge = _bridge()
    for proj in _projections(bridge):
        raw, effective = _f64(proj.parametrizations.weight.original), _f64(proj.weight)
        s_hat = float((raw * effective).sum() / (effective * effective).sum())
        torch.testing.assert_close(effective * s_hat, raw, rtol=1e-6, atol=1e-7)
        assert 0.0 < s_hat <= float(torch.linalg.matrix_norm(raw, ord=2)) * (1 + 1e-6)
        assert float(torch.linalg.matrix_norm(effective, ord=2)) >= 1 - 1e-6


# Raw spectrum with gap s2/s1 = 1/2 (min(d_r, d) = 8 singular values).
SPECTRUM = torch.tensor([2.0, 1.0, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float64)


def _prescribe_spectrum(bridge: CrossAttentionBridge) -> None:
    """Set every raw W_K^l, W_V^l to U diag(SPECTRUM) V^T with random orthonormal U [d_r, d], V [d, d]."""
    for proj in _projections(bridge):
        left, _ = torch.linalg.qr(torch.randn(D_R, D, dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
        with torch.no_grad():
            proj.parametrizations.weight.original.copy_((left * SPECTRUM) @ right.T)


def _run_training_forwards(bridge: CrossAttentionBridge, conditioning: BridgeConditioning, forwards: int) -> None:
    """Training-mode precompute_kv calls (each runs ``power_iterations`` power-iteration steps), then eval mode."""
    bridge.train()
    for _ in range(forwards):
        bridge.precompute_kv(conditioning)
    bridge.eval()


@pytest.mark.parametrize(("power_iterations", "forwards"), [(1, 30), (30, 1)])
def test_spectral_norm_converges_to_unit_operator_norm(power_iterations: int, forwards: int) -> None:
    """sigma_max(W_eff) = 1 and W_eff = W / sigma_max(W) after power iteration in training forwards.

    Raw weights get a prescribed spectrum with s2/s1 = 1/2. Each step v <- W^T W v / ||.|| shrinks the
    angle theta to the top right singular vector, tan(theta) by (s2/s1)^2 = 1/4, and the estimate
    s_hat = ||W^T W v|| / ||W v|| lies in [s1 cos(theta), s1]. After 30 steps 1 - cos(theta) <= tan^2(theta)/2
    ~ 2^-121 tan^2(theta_0), far below float32 rounding, which sets the tolerance (rtol 1e-5).
    """
    bridge = _bridge(power_iterations=power_iterations)
    _prescribe_spectrum(bridge)
    z, s, u = _plan_inputs()
    _run_training_forwards(bridge, bridge.build_conditioning(z, s, u), forwards)
    for proj in _projections(bridge):
        effective = _f64(proj.weight)
        torch.testing.assert_close(torch.linalg.svdvals(effective), SPECTRUM / SPECTRUM[0], rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(effective * SPECTRUM[0], _f64(proj.parametrizations.weight.original),
                                   rtol=RTOL, atol=ATOL)


def test_keys_and_values_are_one_over_gamma_lipschitz_in_the_plan() -> None:
    """Sec. 3.2: with sigma_max(W_K^l) = sigma_max(W_V^l) = 1 and the constant 1/gamma,
    ||K_l(Z) - K_l(Z')||_F <= ||Z - Z'||_F / gamma (same for V_l), with equality for a plan perturbation along
    the top right singular vector of W^l: the bound is tight and no normalisation of Z~ rescales the plan.

    Tolerance 1e-5: K is float32, so the difference of two O(1) caches carries ~1e-7 relative rounding.
    """
    bridge = _bridge()
    _prescribe_spectrum(bridge)
    z, s, u = _plan_inputs()
    _run_training_forwards(bridge, bridge.build_conditioning(z, s, u), forwards=30)

    @torch.no_grad()
    def caches(plan: Tensor) -> list[Tensor]:
        cache = bridge.precompute_kv(bridge.build_conditioning(plan, s, u))
        return [t.double() for t in (*cache.keys, *cache.values)]

    base = caches(z)

    def gains(dz: Tensor) -> list[float]:
        return [float((moved - b).norm() / (dz.double().norm() / GAMMA))
                for moved, b in zip(caches(z + dz), base, strict=True)]

    for _ in range(10):
        assert max(gains(torch.randn(B, M, D))) <= 1 + 1e-5
    for index, proj in enumerate(_projections(bridge)):          # keys of layers 0..L_r-1, then values
        top_right = torch.linalg.svd(_f64(proj.weight)).Vh[0]
        dz = torch.zeros(B, M, D)
        dz[1, 2] = top_right.float()
        assert gains(dz)[index] == pytest.approx(1.0, rel=1e-5)


def test_without_spectral_norm_the_raw_weights_are_used() -> None:
    """spectral_norm=False: plain bias-free linear maps, K_l = Z~ W_K^l with the raw W_K^l (whose norm is not 1)."""
    bridge = _bridge(spectral_norm=False)
    for proj in _projections(bridge):
        assert not parametrize.is_parametrized(proj)
        assert isinstance(proj.weight, nn.Parameter) and proj.bias is None
    assert any(abs(float(torch.linalg.matrix_norm(_f64(p.weight), ord=2)) - 1) > 0.05 for p in _projections(bridge))
    z, s, u = _plan_inputs()
    weights_before = [p.weight.detach().clone() for p in _projections(bridge)]
    bridge.train()                                              # no power iteration either
    cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u, drop=torch.zeros(B, dtype=torch.bool)))
    tokens, _ = _conditioning_reference(bridge, z, s, u)
    d_h = D_R // NUM_HEADS
    for layer in range(L_R):
        for got, proj in ((cache.keys[layer], bridge.key_proj[layer]), (cache.values[layer], bridge.value_proj[layer])):
            expected = (tokens @ _f64(proj.weight).T).reshape(B, M_TILDE, NUM_HEADS, d_h).transpose(1, 2)
            torch.testing.assert_close(got.double(), expected, rtol=RTOL, atol=ATOL)
    for before, proj in zip(weights_before, _projections(bridge), strict=True):
        assert torch.equal(before, proj.weight)


# ------------------------------------------------------------------------------------------------
# Keys and values computed once per generation
# ------------------------------------------------------------------------------------------------
def test_keys_and_values_are_projected_once_per_generation() -> None:
    """precompute_kv evaluates each W_K^l and W_V^l exactly once (L_r each); K cross-attention passes add none."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    conditioning = bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK)
    hiddens = [torch.randn(B, N, D_R) for _ in range(K_STEPS)]
    counts: Counter[str] = Counter()

    def count(key: str):
        return lambda module, args, output: counts.update([key])

    handles = [proj.register_forward_hook(count(name)) for name in ("key_proj", "value_proj", "query_proj", "out_proj")
               for proj in getattr(bridge, name)]
    # Also count evaluations of the spectrally normalised weights themselves (any read of W_K^l, W_V^l).
    handles += [proj.parametrizations.weight[0].register_forward_hook(count(name + ".weight"))
                for name in ("key_proj", "value_proj") for proj in getattr(bridge, name)]
    try:
        cache = bridge.precompute_kv(conditioning)
        assert counts == Counter({"key_proj": L_R, "value_proj": L_R, "key_proj.weight": L_R, "value_proj.weight": L_R})
        for hidden in hiddens:                                   # realizer passes 4 .. K + 3
            for layer in range(L_R):
                bridge.cross_attention(layer, hidden, cache)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == Counter({"key_proj": L_R, "value_proj": L_R, "key_proj.weight": L_R, "value_proj.weight": L_R,
                              "query_proj": K_STEPS * L_R, "out_proj": K_STEPS * L_R})


def test_reusing_the_cache_equals_recomputing_keys_and_values() -> None:
    """At every one of K steps (new realizer hidden states), attention with the cached K_l, V_l equals attention
    that recomputes them from Z~: bit-identical to a freshly built cache, and equal to the float64 Eq. 10."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    conditioning = bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK)
    cache = bridge.precompute_kv(conditioning)
    tokens, mask = _conditioning_reference(bridge, z, s, u, SLOT_MASK, CONTEXT_MASK)
    for _ in range(K_STEPS):
        hidden = torch.randn(B, N, D_R)
        for layer in range(L_R):
            cached = bridge.cross_attention(layer, hidden, cache)
            assert torch.equal(cached, bridge.cross_attention(layer, hidden, bridge.precompute_kv(conditioning)))
            torch.testing.assert_close(cached.double(), _eq10_reference(bridge, layer, hidden, tokens, mask),
                                       rtol=RTOL, atol=ATOL)


def test_cross_attention_reads_keys_and_values_only_from_the_cache() -> None:
    """The cache is held fixed: changing W_K^l, W_V^l after precompute_kv leaves cross_attention unchanged."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    conditioning = bridge.build_conditioning(z, s, u)
    cache = bridge.precompute_kv(conditioning)
    hidden = torch.randn(B, N, D_R)
    before = [bridge.cross_attention(layer, hidden, cache) for layer in range(L_R)]
    with torch.no_grad():
        for proj in _projections(bridge):
            proj.parametrizations.weight.original.mul_(-3.0).add_(1.0)
    for layer in range(L_R):
        assert torch.equal(bridge.cross_attention(layer, hidden, cache), before[layer])
    fresh = bridge.precompute_kv(conditioning)                  # only a new precompute sees the new weights
    assert not torch.equal(bridge.cross_attention(0, hidden, fresh), before[0])


def test_cache_does_not_depend_on_output_length() -> None:
    """Sec. 3.2: K_l, V_l have M~ rows whatever N; the same cache serves N = 5 and N = 50 unchanged."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK))
    snapshot = [t.clone() for t in (*cache.keys, *cache.values)]
    nbytes = cache.nbytes()
    for t in (*cache.keys, *cache.values):
        assert t.shape == (B, NUM_HEADS, M_TILDE, D_R // NUM_HEADS)
    assert cache.mask.shape == (B, M_TILDE)
    short_hidden = torch.randn(B, 5, D_R)
    long_hidden = torch.cat([short_hidden, torch.randn(B, 45, D_R)], dim=1)
    short = bridge.cross_attention(1, short_hidden, cache)
    long = bridge.cross_attention(1, long_hidden, cache)
    assert short.shape == (B, 5, D_R) and long.shape == (B, 50, D_R)
    torch.testing.assert_close(long[:, :5], short, rtol=RTOL, atol=ATOL)       # each query attends on its own
    assert cache.nbytes() == nbytes
    for before, after in zip(snapshot, (*cache.keys, *cache.values), strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cache_bytes_are_2_L_r_Mtilde_d_k_b_per_sample(dtype: torch.dtype) -> None:
    """Sec. 3.2: the cache holds 2 L_r M~ d_k b bytes per sample (b = 4 for float32, 2 for bf16)."""
    batch = 3
    bridge = _bridge().to(dtype)
    z, s, u = _plan_inputs(batch)
    cache = bridge.precompute_kv(bridge.build_conditioning(z.to(dtype), s.to(dtype), u))
    b = torch.finfo(dtype).bits // 8
    expected = 2 * L_R * M_TILDE * D_R * b * batch
    assert cache.keys[0].dtype == dtype
    assert cache.nbytes() == expected
    assert CrossAttentionBridge.kv_cache_bytes(L_R, M_TILDE, D_R, b) * batch == expected
    assert CrossAttentionBridge.kv_cache_bytes(L_R, M_TILDE, D_R, b, batch_size=batch) == expected


def test_kv_cache_bytes_reproduces_the_paper_example() -> None:
    """Sec. 3.2: M~ = 256, L_r = 24, d_k = 2048 and bf16 give exactly 48 MiB."""
    assert CrossAttentionBridge.kv_cache_bytes(24, 256, 2048, 2) == 48 * 2**20


# ------------------------------------------------------------------------------------------------
# Plan dropout and the null token
# ------------------------------------------------------------------------------------------------
def test_drop_override_replaces_only_the_chosen_samples() -> None:
    """Dropped samples become [null token; masked rest]; the other samples are bit-identical to no dropout."""
    bridge = _bridge()
    batch = 4
    z, s, u = _plan_inputs(batch)
    slot_mask = torch.tensor([[True, False, True], [True, True, True], [False, True, True], [True, True, False]])
    context_mask = torch.ones(batch, M_C, dtype=torch.bool)
    context_mask[1, 0] = False
    drop = torch.tensor([True, False, True, False])
    full_mask = torch.cat([slot_mask, context_mask, torch.ones(batch, 1, dtype=torch.bool)], dim=1)
    mixed = bridge.build_conditioning(z, s, u, slot_mask, context_mask, drop=drop)
    assert torch.equal(mixed.dropped, drop)
    assert torch.equal(mixed.tokens[~drop], _exact_conditioning(bridge, z, s, u)[~drop])
    assert torch.equal(mixed.mask[~drop], full_mask[~drop])
    null_mask = torch.tensor([True] + [False] * (M_TILDE - 1))
    for b in torch.nonzero(drop).flatten().tolist():
        assert torch.equal(mixed.tokens[b, 0], bridge.null_token.detach())    # replaces Z~ itself (not / gamma)
        assert torch.equal(mixed.mask[b], null_mask)


def test_training_drop_rate_matches_plan_dropout() -> None:
    """In training each sample is dropped independently with p = 0.1 (Sec. 3.2).

    Over n = 20000 draws the rate lies within 5 standard errors, 5 sqrt(p (1 - p) / n) = 0.0106, of p,
    and both members of a disjoint pair are dropped at rate p^2 within 5 sqrt(p^2 (1 - p^2) / (n / 2))
    (a two-sided 5-sigma miss has probability ~6e-7; the generator is fixed, so the test is deterministic).
    """
    bridge = _bridge().train()
    p, n = 0.1, 20_000
    assert bridge.plan_dropout == p
    zeros = torch.zeros(n, 1, D)
    dropped = bridge.build_conditioning(zeros, zeros, torch.zeros(n, dtype=torch.long),
                                        generator=torch.Generator().manual_seed(1234)).dropped
    assert dropped.shape == (n,) and dropped.dtype == torch.bool
    assert abs(dropped.double().mean().item() - p) < 5 * math.sqrt(p * (1 - p) / n)
    both = (dropped[0::2] & dropped[1::2]).double().mean().item()
    assert abs(both - p * p) < 5 * math.sqrt(p * p * (1 - p * p) / (n // 2))


def test_random_plan_dropout_uses_the_null_token_replacement() -> None:
    """In training, each sample flagged as dropped carries [null token; masked rest] and every other sample
    carries its own Z~ unchanged."""
    bridge = _bridge().train()
    n = 200
    z, s = torch.randn(n, M, D), torch.randn(n, M_C, D)
    u = torch.randint(0, NUM_MODES, (n,))
    cond = bridge.build_conditioning(z, s, u, generator=torch.Generator().manual_seed(5))
    dropped = cond.dropped
    assert 0 < int(dropped.sum()) < n
    assert torch.equal(cond.tokens[~dropped], _exact_conditioning(bridge, z, s, u)[~dropped])
    assert bool(cond.mask[~dropped].all())
    k = int(dropped.sum())
    assert torch.equal(cond.tokens[dropped, 0], bridge.null_token.detach().expand(k, D))
    assert torch.equal(cond.mask[dropped], torch.tensor([True] + [False] * (M_TILDE - 1)).expand(k, -1))


def test_dropout_draw_is_reproducible_with_a_generator() -> None:
    """The plan-dropout draw is a function of the generator state."""
    bridge = _bridge().train()
    z, s, u = torch.zeros(500, 1, D), torch.zeros(500, 1, D), torch.zeros(500, dtype=torch.long)

    def draw(seed: int) -> Tensor:
        return bridge.build_conditioning(z, s, u, generator=torch.Generator().manual_seed(seed)).dropped

    assert torch.equal(draw(7), draw(7))
    assert not torch.equal(draw(7), draw(8))


@pytest.mark.parametrize("setting", ["eval_mode", "train_mode_p0"])
def test_no_plan_dropout_in_eval_mode_or_with_zero_probability(setting: str) -> None:
    """Eval mode never drops (even with p = 0.5); training with p = 0 never drops either."""
    bridge = _bridge(plan_dropout=0.5) if setting == "eval_mode" else _bridge(plan_dropout=0.0).train()
    n = 2000
    z, s = torch.randn(n, 1, D), torch.randn(n, 1, D)
    u = torch.randint(0, NUM_MODES, (n,))
    cond = bridge.build_conditioning(z, s, u, generator=torch.Generator().manual_seed(0))
    assert not bool(cond.dropped.any())
    assert bool(cond.mask.all())
    assert torch.equal(cond.tokens, _exact_conditioning(bridge, z, s, u))


def test_null_conditioning_is_the_learned_token_alone() -> None:
    """Unconditional input for predictor-free guidance: one real token, the null token, per sample."""
    bridge = _bridge()
    cond = bridge.null_conditioning(3)
    assert isinstance(cond, BridgeConditioning)
    assert cond.tokens.shape == (3, 1, D)
    assert torch.equal(cond.tokens, bridge.null_token.detach().expand(3, 1, D))
    assert torch.equal(cond.mask, torch.ones(3, 1, dtype=torch.bool))
    assert torch.equal(cond.dropped, torch.ones(3, dtype=torch.bool))
    as_double = bridge.null_conditioning(2, dtype=torch.float64)
    assert as_double.tokens.dtype == torch.float64
    assert torch.equal(as_double.tokens, _f64(bridge.null_token).expand(2, 1, D))


def test_dropped_sample_attends_only_to_the_null_token() -> None:
    """A dropped sample's output ignores its plan, context and mode: CA = (null W_V) W_O for every query,
    which is also what null_conditioning gives (the u(. | null) of predictor-free guidance)."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    drop = torch.tensor([True, False])
    hidden = torch.randn(B, N, D_R)
    z_other, s_other, u_other = z.clone(), s.clone(), u.clone()
    z_other[0], s_other[0], u_other[0] = 10 * torch.randn(M, D), torch.randn(M_C, D), (u[0] + 1) % NUM_MODES
    cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK, drop=drop))
    cache_other = bridge.precompute_kv(bridge.build_conditioning(z_other, s_other, u_other, SLOT_MASK,
                                                                 CONTEXT_MASK, drop=drop))
    null_cache = bridge.precompute_kv(bridge.null_conditioning(B))
    for layer in range(L_R):
        out = bridge.cross_attention(layer, hidden, cache)
        assert torch.equal(bridge.cross_attention(layer, hidden, cache_other), out)
        expected = _null_output_reference(bridge, layer, _f64(bridge.null_token)).expand(N, D_R)
        torch.testing.assert_close(out[0].double(), expected, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(bridge.cross_attention(layer, hidden, null_cache)[0], out[0], rtol=RTOL, atol=ATOL)
    with torch.no_grad():
        bridge.null_token.add_(1.0)
    changed = bridge.precompute_kv(bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK, drop=drop))
    assert not torch.allclose(bridge.cross_attention(0, hidden, changed)[0], bridge.cross_attention(0, hidden, cache)[0])


# ------------------------------------------------------------------------------------------------
# Gradients (Stage 3 back-propagates through the cache; the null token is learned)
# ------------------------------------------------------------------------------------------------
def test_gradients_flow_to_plan_and_context_through_the_cache() -> None:
    """d loss / d Z_target and d loss / d s_x through K_l, V_l equal float64 autograd through Eq. 10;
    padded rows get exactly zero gradient; W_Q, W_K, W_V, W_O and e(u) all receive gradients."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    z.requires_grad_(True)
    s.requires_grad_(True)
    hidden = torch.randn(B, N, D_R)
    probes = [torch.randn(B, N, D_R) for _ in range(L_R)]
    cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK))
    sum((bridge.cross_attention(layer, hidden, cache) * probes[layer]).sum() for layer in range(L_R)).backward()

    z64, s64 = _f64(z).requires_grad_(True), _f64(s).requires_grad_(True)
    tokens, mask = _conditioning_reference(bridge, z64, s64, u, SLOT_MASK, CONTEXT_MASK)
    reference = sum((_eq10_reference(bridge, layer, hidden, tokens, mask) * probes[layer].double()).sum()
                    for layer in range(L_R))
    grad_z, grad_s = torch.autograd.grad(reference, (z64, s64))
    torch.testing.assert_close(z.grad.double(), grad_z, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s.grad.double(), grad_s, rtol=RTOL, atol=ATOL)
    assert torch.equal(z.grad[~SLOT_MASK], torch.zeros(int((~SLOT_MASK).sum()), D))
    assert torch.equal(s.grad[~CONTEXT_MASK], torch.zeros(int((~CONTEXT_MASK).sum()), D))
    assert bool((z.grad[SLOT_MASK].abs().sum(dim=-1) > 0).all())

    for layer in range(L_R):
        for proj in (bridge.key_proj[layer], bridge.value_proj[layer]):
            assert proj.parametrizations.weight.original.grad.abs().sum() > 0
        for proj in (bridge.query_proj[layer], bridge.out_proj[layer]):
            assert proj.weight.grad.abs().sum() > 0
    table_grad = bridge.mode_embedding.weight.grad
    assert bool((table_grad[u].abs().sum(dim=-1) > 0).all()) and torch.equal(table_grad[0], torch.zeros(D))
    assert bridge.null_token.grad is None or not bool(bridge.null_token.grad.any())


def test_dropped_sample_sends_gradient_to_the_null_token_only() -> None:
    """With sample 0 dropped, its loss reaches the null token (d/d null of sum_n probe_n (null W_V) W_O) and
    nothing reaches its plan or context; sample 1 still sends gradient to its plan."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    z.requires_grad_(True)
    s.requires_grad_(True)
    hidden = torch.randn(B, N, D_R)
    probes = [torch.randn(B, N, D_R) for _ in range(L_R)]
    drop = torch.tensor([True, False])
    cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u, SLOT_MASK, CONTEXT_MASK, drop=drop))
    sum((bridge.cross_attention(layer, hidden, cache) * probes[layer]).sum() for layer in range(L_R)).backward()

    assert torch.equal(z.grad[0], torch.zeros(M, D)) and torch.equal(s.grad[0], torch.zeros(M_C, D))
    assert z.grad[1].abs().sum() > 0
    null64 = _f64(bridge.null_token).requires_grad_(True)
    reference = sum((_null_output_reference(bridge, layer, null64) * probes[layer][0].double()).sum()
                    for layer in range(L_R))
    (expected,) = torch.autograd.grad(reference, null64)
    assert bridge.null_token.grad is not None
    torch.testing.assert_close(bridge.null_token.grad.double(), expected, rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("overrides", [
    pytest.param({"hidden_dim": 18}, id="heads_not_dividing_d_r"),
    pytest.param({"gamma": 0.0}, id="gamma_zero"),
    pytest.param({"gamma": -1.0}, id="gamma_negative"),
    pytest.param({"gamma": float("nan")}, id="gamma_nan"),
    pytest.param({"plan_dropout": 1.0}, id="dropout_one"),
    pytest.param({"plan_dropout": -0.1}, id="dropout_negative"),
    pytest.param({"num_layers": 0}, id="no_layers"),
    pytest.param({"power_iterations": 0}, id="no_power_iterations"),
])
def test_invalid_hyperparameters_raise_value_error(overrides: dict[str, object]) -> None:
    """Heads must divide d_r, gamma > 0 (Z~ / gamma), 0 <= p < 1, and L_r, power iterations >= 1."""
    with pytest.raises(ValueError):
        _bridge(**overrides)


_CONDITIONING_ERRORS = {
    "z_target_2d": {"z_target": torch.zeros(B, D)},
    "z_target_width": {"z_target": torch.zeros(B, M, D + 1)},
    "s_x_width": {"s_x": torch.zeros(B, M_C, D + 1)},
    "batch_mismatch": {"s_x": torch.zeros(B + 1, M_C, D)},
    "u_batch": {"u": torch.tensor([0, 1, 2])},
    "u_int_2d": {"u": torch.tensor([[0], [1]])},
    "u_soft_width": {"u": torch.ones(B, NUM_MODES + 1)},
    "slot_mask_shape": {"slot_mask": torch.ones(B, M + 1, dtype=torch.bool)},
    "context_mask_shape": {"context_mask": torch.ones(B, M_C - 1, dtype=torch.bool)},
    "drop_shape": {"drop": torch.tensor([True])},
    "drop_dtype": {"drop": torch.tensor([1, 0])},
}


@pytest.mark.parametrize("case", list(_CONDITIONING_ERRORS))
def test_invalid_conditioning_inputs_raise_value_error(case: str) -> None:
    """Z~ needs Z_target [B, M, d], s_x [B, M_c, d], u [B] or [B, |U|], matching masks and a bool drop."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    kwargs: dict[str, Tensor] = {"z_target": z, "s_x": s, "u": u}
    kwargs.update(_CONDITIONING_ERRORS[case])
    with pytest.raises(ValueError):
        bridge.build_conditioning(**kwargs)


def test_invalid_cache_inputs_raise_value_error() -> None:
    """precompute_kv needs tokens [B, M~, d] and at least one real key per sample (Eq. 10 softmax)."""
    bridge = _bridge()
    ones, none_dropped = torch.ones(B, 5, dtype=torch.bool), torch.zeros(B, dtype=torch.bool)
    for conditioning in (BridgeConditioning(torch.randn(B, 5, D + 1), ones, none_dropped),     # wrong width
                         BridgeConditioning(torch.randn(B, D), ones, none_dropped),            # no token axis
                         BridgeConditioning(torch.randn(B, 5, D), torch.tensor([[True] * 5, [False] * 5]),
                                            none_dropped)):                                     # no real token
        with pytest.raises(ValueError):
            bridge.precompute_kv(conditioning)


def test_invalid_cross_attention_inputs_raise_value_error() -> None:
    """Layer index in [0, L_r), hidden [B, N, d_r] with the cache's batch, and a cache of L_r layers."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u))
    hidden = torch.randn(B, N, D_R)
    short_cache = BridgeKVCache(cache.keys[:1], cache.values[:1], cache.mask)
    for layer, h, c in ((L_R, hidden, cache), (-1, hidden, cache), (0, torch.randn(B, N, D_R + 4), cache),
                        (0, torch.randn(B + 1, N, D_R), cache), (0, torch.randn(N, D_R), cache), (0, hidden, short_cache)):
        with pytest.raises(ValueError):
            bridge.cross_attention(layer, h, c)


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint8, torch.float32])
@pytest.mark.parametrize("which", ["slot_mask", "context_mask"])
def test_non_bool_masks_never_reach_the_attention(dtype: torch.dtype, which: str) -> None:
    """Key masks are boolean (True = real); a 0/1 tensor of another dtype is rejected with ValueError
    before any attention output is produced."""
    bridge = _bridge()
    z, s, u = _plan_inputs()
    mask = torch.ones(B, M if which == "slot_mask" else M_C, dtype=dtype)
    with pytest.raises(ValueError):
        cache = bridge.precompute_kv(bridge.build_conditioning(z, s, u, **{which: mask}))
        bridge.cross_attention(0, torch.randn(B, N, D_R), cache)


# ------------------------------------------------------------------------------------------------
# Configuration and devices
# ------------------------------------------------------------------------------------------------
def test_from_config_uses_planner_realizer_and_vicreg_widths(tiny_config) -> None:
    """d from the planner, d_r, L_r and heads from the realizer, gamma from VICReg, p from the bridge."""
    bridge = CrossAttentionBridge.from_config(tiny_config).eval()
    d, d_r = tiny_config.planner.latent_dim, tiny_config.realizer.hidden_dim
    heads, layers = tiny_config.realizer.num_heads, tiny_config.realizer.num_layers
    assert (bridge.latent_dim, bridge.hidden_dim, bridge.num_layers, bridge.num_heads) == (d, d_r, layers, heads)
    assert bridge.gamma == tiny_config.vicreg.gamma
    assert bridge.plan_dropout == tiny_config.bridge.plan_dropout
    for name in ("key_proj", "value_proj", "query_proj", "out_proj"):
        assert len(getattr(bridge, name)) == layers
    for proj in _projections(bridge):
        assert proj.weight.shape == (d_r, d)
        assert parametrize.is_parametrized(proj, "weight") == tiny_config.bridge.spectral_norm
    for proj in (*bridge.query_proj, *bridge.out_proj):
        assert proj.weight.shape == (d_r, d_r)
    assert bridge.mode_embedding.weight.shape == (tiny_config.planner.num_modes, d)
    assert bridge.null_token.shape == (d,)
    cache = bridge(torch.randn(2, 5, d), torch.randn(2, 7, d), torch.tensor([0, tiny_config.planner.num_modes - 1]))
    assert cache.num_layers == layers
    assert cache.keys[0].shape == cache.values[-1].shape == (2, heads, 5 + 7 + 1, d_r // heads)
    assert bridge.cross_attention(layers - 1, torch.randn(2, 9, d_r), cache).shape == (2, 9, d_r)


def test_from_config_passes_bridge_settings() -> None:
    """vicreg.gamma, bridge.spectral_norm, plan_dropout and power_iterations reach the module."""
    cfg = load_config(["planner=tiny", "data=tiny", "realizer=tiny", "vicreg.gamma=2.0", "bridge.spectral_norm=false",
                       "bridge.plan_dropout=0.25"])
    bridge = CrossAttentionBridge.from_config(cfg).eval()
    assert bridge.gamma == 2.0 and bridge.plan_dropout == 0.25
    assert not any(parametrize.is_parametrized(proj) for proj in _projections(bridge))
    d = cfg.planner.latent_dim
    z, s = torch.randn(2, 3, d), torch.randn(2, 4, d)
    cond = bridge.build_conditioning(z, s, torch.tensor([0, 1]))
    assert torch.equal(cond.tokens[:, :3], z / 2.0) and torch.equal(cond.tokens[:, 3:7], s / 2.0)   # 1/gamma
    cfg = load_config(["planner=tiny", "data=tiny", "realizer=tiny", "bridge.power_iterations=3"])
    bridge = CrossAttentionBridge.from_config(cfg)
    assert all(proj.parametrizations.weight[0].n_power_iterations == 3 for proj in _projections(bridge))


@pytest.mark.skipif(accelerator() is None, reason="no CUDA/ROCm/MPS device")
def test_cross_attention_on_accelerator_matches_cpu_reference() -> None:
    """Eq. 10 on the accelerator. Tolerance 1e-4: device kernels reduce in other orders (and may use TF32)."""
    device = accelerator()
    bridge = _bridge()
    z, s, u = _plan_inputs()
    tokens, mask = _conditioning_reference(bridge, z, s, u, SLOT_MASK, CONTEXT_MASK)
    hidden = torch.randn(B, N, D_R)
    expected = [_eq10_reference(bridge, layer, hidden, tokens, mask) for layer in range(L_R)]
    on_device = copy.deepcopy(bridge).to(device)
    cache = on_device.precompute_kv(on_device.build_conditioning(z.to(device), s.to(device), u.to(device),
                                                                 SLOT_MASK.to(device), CONTEXT_MASK.to(device)))
    for layer in range(L_R):
        out = on_device.cross_attention(layer, hidden.to(device), cache)
        assert out.device.type == device.type
        torch.testing.assert_close(out.cpu().double(), expected[layer], rtol=1e-4, atol=1e-4)
