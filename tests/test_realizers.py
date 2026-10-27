# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of the realizers F_psi, the plan readout and the planner assembly (PR-JEPA Secs. 3.2-3.4, 5.1).

    realizer block     x <- x + g1 SA(mod(LN x)); x <- x + g2 CA_l(mod(LN x), Z~); x <- x + g3 MLP(mod(LN x))
                       AdaLN modulation by embeddings of t and h; adaLN-Zero gates (Sec. 3.2)
    cross-attention    CA_l(H, Z~) = softmax((H W_Q)(Z~ W_K)^T / sqrt(d_k)) Z~ W_V, K/V computed once  (Eq. 10)
    mixture path       p_t(w | w_0, w_1) = (1 - kappa_t) delta_{w_0} + kappa_t delta_{w_1}            (Eq. 11)
    FS-DFM loss        L_real = E[ -sum_i log p_{1|t}(w_1^i | w_t, t, h, sg(Z)) ]                       (Eq. 12)
    exact jump mass    pi_{t,h} = (kappa_{t+h} - kappa_t)/(1 - kappa_t)                                  (Eq. 15)
    Logit-KL           l_1 = a e_k + b 1, a = log((1 - eps + eps/S)/(eps/S)), b = log(eps/S)             (Eq. 18)
                       l_t = (1 - t) l_0 + t l_1, l_0 ~ N(0, s^2 I)                                      (Eq. 19)
                       l_{k+1} = l_k + (1/K)(a p - l_k)/(1 - t_k), l_K = a p at t_{K-1} = 1 - 1/K        (Eq. 21)
    Stage 2            the realizer conditions only on the predicted plan, never on f_theta_bar(y)     (Sec. 5.1)

References are written from these formulas in float64 on the CPU and never call the function under test.
Monte Carlo checks use 5 standard errors. Most comparisons run the realizer in float32 against float64
references, at tolerances justified where they are set; test_realizer_runs_in_float64 checks float64 support.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import parametrize

from tests.conftest import ATOL, RTOL
from text_vpa.configs.schema import TextVPAConfig
from text_vpa.data.ast_parser import SlotKind, TreeSitterASTExtractor
from text_vpa.data.tokenization import ByteTokenizer
from text_vpa.inference.energy_terms import VocabularyAnalysis
from text_vpa.models.bridge import BridgeKVCache
from text_vpa.models.encoders import TargetEncoder, TargetOutput
from text_vpa.models.plan import PredictedPlan, require_predicted_plan
from text_vpa.models.readout import PlanReadout, ReadoutOutput, ReadoutTargets
from text_vpa.models.realizers.backbone import RealizerBackbone, timestep_embedding
from text_vpa.models.realizers.fs_dfm import ConditionalFSDFMRealizer
from text_vpa.models.realizers.logit_kl import LogitKLRealizer
from text_vpa.models.realizers.paths import KappaSchedule, mixture_path_sample
from text_vpa.models.text_vpa_model import TextVPAModel, TextVPAPlanner

TOK = ByteTokenizer()
S = TOK.vocab_size               # 256 bytes + 9 specials
PAD, MASK = TOK.pad_id, TOK.mask_id
B, N, M_C = 2, 12, 10            # batch, canvas length, context length
EPS32 = 2.0 ** -24               # float32 unit roundoff
NUM_KINDS = len(SlotKind)


# ------------------------------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------------------------------
def _config(base: TextVPAConfig, **sections: dict) -> TextVPAConfig:
    """``base`` with S = |V| of the byte tokenizer and some section fields overridden (validated)."""
    data = base.with_vocab_size(S).model_dump()
    for name, values in sections.items():
        data[name].update(values)
    return TextVPAConfig.model_validate(data)


def _model(cfg: TextVPAConfig) -> TextVPAModel:
    return TextVPAModel.from_config(cfg, mask_id=MASK).eval()


def _context(seed: int, batch: int = B) -> Tensor:
    return torch.randint(0, 256, (batch, M_C), generator=torch.Generator().manual_seed(seed))


def _plan(model: TextVPAModel, seed: int, batch: int = B) -> PredictedPlan:
    """Planner passes 1-3 with forced modes (distinct seeds give distinct plans)."""
    modes = torch.randint(0, model.planner.num_modes, (batch,), generator=torch.Generator().manual_seed(seed + 99))
    with torch.no_grad():
        return model.planner.plan(_context(seed, batch), mode=modes)


def _open_gates(backbone: RealizerBackbone, *, tables: bool = True, projection: bool = False, std: float = 0.5,
                seed: int = 7) -> None:
    """Move the adaLN-Zero parameters away from zero: per-block tables and/or the shared t/h projections."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if tables:
            for block in backbone.blocks:
                block.table.copy_(std * torch.randn(block.table.shape, generator=g))
        if projection:
            for proj in (backbone.block_modulation[-1], backbone.final_modulation[-1]):
                proj.weight.copy_(std * torch.randn(proj.weight.shape, generator=g))
                proj.bias.copy_(std * torch.randn(proj.bias.shape, generator=g))


def _raw_weight(proj: nn.Module) -> Tensor:
    """The trainable weight behind a (possibly spectrally normalised) projection."""
    return proj.parametrizations.weight.original if parametrize.is_parametrized(proj) else proj.weight


def _times(t: float, h: float, batch: int = B) -> tuple[Tensor, Tensor]:
    return torch.full((batch,), t), torch.full((batch,), h)


def _nonzero_grad(p: Tensor) -> bool:
    return p.grad is not None and bool(p.grad.abs().sum() > 0)


# ------------------------------------------------------------------------------------------------
# Independent float64 references
# ------------------------------------------------------------------------------------------------
def _sinusoid64(t: Tensor, dim: int, max_period: float = 10_000.0, scale: float = 1000.0) -> Tensor:
    """[cos(scale t f_k)]_k ++ [sin(scale t f_k)]_k (++ 0 for odd dim), f_k = max_period^(-k/half)."""
    half = dim // 2
    freqs = max_period ** (-torch.arange(half, dtype=torch.float64) / half)
    args = scale * t.double()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(t.shape[0], 1, dtype=torch.float64)], dim=1)
    return emb


def _w(p: Tensor) -> Tensor:
    return p.detach().double()


def _linear64(layer: nn.Linear, x: Tensor) -> Tensor:
    y = x @ _w(layer.weight).T
    return y if layer.bias is None else y + _w(layer.bias)


def _ln64(x: Tensor, eps: float = 1e-5) -> Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)


def _silu64(x: Tensor) -> Tensor:
    return x / (1.0 + torch.exp(-x))


def _gelu64(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _softmax64(x: Tensor) -> Tensor:
    z = torch.exp(x - x.amax(dim=-1, keepdim=True))
    return z / z.sum(dim=-1, keepdim=True)


def _log_softmax64(x: Tensor) -> Tensor:
    m = x.amax(dim=-1, keepdim=True)
    return x - m - torch.log(torch.exp(x - m).sum(dim=-1, keepdim=True))


def _attention64(q: Tensor, k: Tensor, v: Tensor, heads: int) -> Tensor:
    """Multi-head softmax(q k^T / sqrt(d_h)) v, heads concatenated: [B, Lq, D], [B, Lk, D] -> [B, Lq, D]."""
    b, lq, d = q.shape
    dh = d // heads
    q, k, v = (x.reshape(b, -1, heads, dh).transpose(1, 2) for x in (q, k, v))
    weights = _softmax64(q @ k.transpose(-1, -2) / math.sqrt(dh))
    return (weights @ v).transpose(1, 2).reshape(b, lq, d)


def _realizer_reference64(realizer: ConditionalFSDFMRealizer, w_t: Tensor, t: Tensor, h: Tensor,
                          plan: PredictedPlan) -> Tensor:
    """Logits of F_psi written out from Sec. 3.2 and Eq. 10, with the conditioning recomputed from the plan."""
    bb, bridge = realizer.backbone, realizer.bridge
    b, n = w_t.shape
    d = bb.hidden_dim
    e_u = _w(bridge.mode_embedding.weight)[plan.mode_input]                          # integer modes
    z_tilde = torch.cat([_w(plan.z_target), _w(plan.s_x), e_u[:, None, :]], dim=1) / bridge.gamma

    def mlp(seq: nn.Sequential, x: Tensor) -> Tensor:                                  # Linear -> SiLU -> Linear
        return _linear64(seq[2], _silu64(_linear64(seq[0], x)))

    c = mlp(bb.t_embed, _sinusoid64(t, bb.time_dim)) + mlp(bb.h_embed, _sinusoid64(h, bb.time_dim))
    modulation = _linear64(bb.block_modulation[1], _silu64(c)).reshape(b, 9, d)
    x = _w(realizer.token_embedding.weight)[w_t] + _w(bb.position.weight)[:n]
    for layer, block in enumerate(bb.blocks):
        m = modulation + _w(block.table)
        sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = (m[:, i, None, :] for i in range(9))
        hid = _ln64(x) * (1.0 + sc1) + sh1
        sa = block.self_attn
        attn = _attention64(_linear64(sa.q_proj, hid), _linear64(sa.k_proj, hid), _linear64(sa.v_proj, hid),
                            sa.num_heads)
        x = x + g1 * _linear64(sa.out_proj, attn)
        hid = _ln64(x) * (1.0 + sc2) + sh2
        keys = z_tilde @ _w(bridge.key_proj[layer].weight).T                            # Z~ W_K^l (Eq. 10)
        values = z_tilde @ _w(bridge.value_proj[layer].weight).T                        # Z~ W_V^l
        ca = _attention64(hid @ _w(bridge.query_proj[layer].weight).T, keys, values, bridge.num_heads)
        x = x + g2 * (ca @ _w(bridge.out_proj[layer].weight).T)
        hid = _ln64(x) * (1.0 + sc3) + sh3
        net = block.mlp.net
        x = x + g3 * _linear64(net[3], _gelu64(_linear64(net[0], hid)))
    shift, scale = _linear64(bb.final_modulation[1], _silu64(c)).reshape(b, 2, 1, d).unbind(dim=1)
    return _linear64(bb.head, _ln64(x) * (1.0 + scale) + shift)


def _smoothed_coefficients64(eps: float, vocab: int) -> tuple[float, float]:
    """(a, b) of Eq. 18."""
    return math.log((1.0 - eps + eps / vocab) / (eps / vocab)), math.log(eps / vocab)


# ================================================================================================
# Realizer backbone: AdaLN, timestep features, cross-attention through the cache
# ================================================================================================
@pytest.mark.parametrize("dim", [256, 7])
def test_timestep_embedding_matches_the_sinusoid_formula(dim: int) -> None:
    t = torch.tensor([0.0, 1 / 8, 0.3, 5 / 8, 0.875, 1.0])
    out = timestep_embedding(t, dim)
    expected = _sinusoid64(t, dim)
    assert out.shape == (t.shape[0], dim)
    # float32 phase rounding: the argument scale * t * f_k (<= 1000 here) carries an absolute error of a few
    # units of 1000 * 2^-24 (rounded frequency and product); cos/sin are 1-Lipschitz.
    tol = 4 * 1000.0 * EPS32 + 1e-6
    torch.testing.assert_close(out.double(), expected, rtol=0.0, atol=tol)
    if dim % 2:
        assert bool((out[:, -1] == 0).all())
    # other period and scale arguments
    out2 = timestep_embedding(t, dim, max_period=100.0, scale=10.0)
    torch.testing.assert_close(out2.double(), _sinusoid64(t, dim, 100.0, 10.0), rtol=0.0, atol=4 * 10 * EPS32 + 1e-6)
    with pytest.raises(ValueError):
        timestep_embedding(t[:, None], dim)


def test_every_block_is_the_identity_at_initialisation(tiny_config) -> None:
    """adaLN-Zero: blocks pass x = E[w] + P through unchanged; logits = head(LN(x)), independent of plan, t, h."""
    model = _model(_config(tiny_config))
    realizer = model.realizer
    w_t = torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(1))
    seen: list[Tensor] = []
    handle = realizer.backbone.final_norm.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach()))
    try:
        with torch.no_grad():
            cache_a = realizer.prepare(_plan(model, 0))
            cache_b = realizer.prepare(_plan(model, 1))
            logits = realizer(w_t, *_times(0.0, 0.5), cache_a)
            other_plan = realizer(w_t, *_times(0.0, 0.5), cache_b)
            other_time = realizer(w_t, *_times(0.625, 0.125), cache_a)
    finally:
        handle.remove()
    embeddings = realizer.token_embedding.weight[w_t] + realizer.backbone.position.weight[:N]
    assert torch.equal(seen[0], embeddings)                         # every block was the identity, exactly
    assert torch.equal(logits, other_plan) and torch.equal(logits, other_time)
    expected = _linear64(realizer.backbone.head, _ln64(_w(embeddings)))
    torch.testing.assert_close(logits.double(), expected, rtol=RTOL, atol=ATOL)


def test_at_initialisation_only_the_gates_receive_gradient_from_the_blocks(tiny_config) -> None:
    """With zero gates the bridge gets exactly zero gradient; the gate rows (2, 5, 8) are what start training."""
    model = _model(_config(tiny_config))
    realizer = model.realizer
    w_t = torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(2))
    logits = realizer(w_t, *_times(0.25, 0.25), realizer.prepare(_plan(model, 0)))
    realizer.loss(logits, torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(3))).backward()
    for p in realizer.bridge.parameters():
        assert p.grad is None or bool((p.grad == 0).all())
    for block in realizer.backbone.blocks:
        gate_rows, other_rows = block.table.grad[[2, 5, 8]], block.table.grad[[0, 1, 3, 4, 6, 7]]
        assert bool((gate_rows.abs().sum(dim=1) > 0).all())
        assert bool((other_rows == 0).all())


def test_opening_the_gates_makes_the_logits_depend_on_plan_t_and_h(tiny_config) -> None:
    model = _model(_config(tiny_config))
    realizer = model.realizer
    w_t = torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        cache_a, cache_b = realizer.prepare(_plan(model, 0)), realizer.prepare(_plan(model, 1))
        _open_gates(realizer.backbone, tables=True, projection=False)
        base = realizer(w_t, *_times(0.25, 0.25), cache_a)
        # Tables alone: the gates are constants, so the plan matters but t and h still do not reach the blocks.
        assert (base - realizer(w_t, *_times(0.25, 0.25), cache_b)).abs().max() > 1e-3
        assert torch.equal(base, realizer(w_t, *_times(0.75, 0.125), cache_a))
        _open_gates(realizer.backbone, tables=True, projection=True)
        base = realizer(w_t, *_times(0.25, 0.25), cache_a)
        assert (base - realizer(w_t, *_times(0.5, 0.25), cache_a)).abs().max() > 1e-3     # depends on t
        assert (base - realizer(w_t, *_times(0.25, 0.125), cache_a)).abs().max() > 1e-3   # depends on h


def test_realizer_logits_match_a_float64_reference_of_the_adaln_block(tiny_config) -> None:
    model = _model(_config(tiny_config))
    realizer = model.realizer
    _open_gates(realizer.backbone, tables=True, projection=True)
    plan = _plan(model, 5)
    w_t = torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(5))
    t, h = torch.tensor([0.25, 0.625]), torch.tensor([0.5, 0.125])
    with torch.no_grad():
        logits = realizer(w_t, t, h, realizer.prepare(plan))
    expected = _realizer_reference64(realizer, w_t, t, h, plan)
    # The network runs in float32: the sinusoid phases (arguments up to 625 here) carry absolute errors of
    # ~625 * 2^-24 ~ 4e-5 that propagate through the t/h MLPs into all 9 L_r + 2 modulation vectors, plus
    # float32 rounding in 3 blocks (observed ~1e-5 on logits of size ~3); 5e-5 bounds this and is far below
    # the O(1e-1) effect of any misplaced gate, shift or scale.
    torch.testing.assert_close(logits.double(), expected, rtol=RTOL, atol=5e-5)


def test_forward_reuses_cached_keys_and_values_across_the_k_passes(tiny_config) -> None:
    """Eq. 10: K_l, V_l are computed once (in prepare) and never again by the K realizer passes."""
    model = _model(_config(tiny_config))
    realizer, bridge = model.realizer, model.realizer.bridge
    _open_gates(realizer.backbone, tables=True, projection=True)
    num_layers, k_steps = bridge.num_layers, 4
    calls: Counter[str] = Counter()
    handles = [proj.register_forward_hook(lambda *_a, name=name: calls.update([name]))
               for name, group in (("key", bridge.key_proj), ("value", bridge.value_proj), ("query", bridge.query_proj))
               for proj in group]
    plan = _plan(model, 0)
    w_t = torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(6))
    try:
        with torch.no_grad():
            cache = realizer.prepare(plan)
            assert calls == {"key": num_layers, "value": num_layers}
            reused = [realizer(w_t, *_times(k / k_steps, 1 / k_steps), cache) for k in range(k_steps)]
            assert calls == {"key": num_layers, "value": num_layers, "query": k_steps * num_layers}
            fresh = [realizer(w_t, *_times(k / k_steps, 1 / k_steps), realizer.prepare(plan)) for k in range(k_steps)]
    finally:
        for handle in handles:
            handle.remove()
    for a, b in zip(reused, fresh, strict=True):
        assert torch.equal(a, b)
    assert (reused[0] - reused[-1]).abs().max() > 1e-3          # the passes really differ (t enters)


def test_logits_read_the_plan_only_through_the_cache(tiny_config) -> None:
    model = _model(_config(tiny_config))
    realizer, bridge = model.realizer, model.realizer.bridge
    _open_gates(realizer.backbone, tables=True, projection=True)
    w_t = torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(8))
    times = _times(0.5, 0.25)
    with torch.no_grad():
        plan_a = _plan(model, 0)
        cache_a = realizer.prepare(plan_a)
        logits_a = realizer(w_t, *times, cache_a)
        assert (logits_a - realizer(w_t, *times, realizer.prepare(_plan(model, 1)))).abs().max() > 1e-3
        # Change W_K^l and W_V^l after the cache was built: a cached generation does not see the change ...
        g = torch.Generator().manual_seed(9)
        for proj in (*bridge.key_proj, *bridge.value_proj):
            raw = _raw_weight(proj)
            raw.add_(0.5 * torch.randn(raw.shape, generator=g))
        assert torch.equal(realizer(w_t, *times, cache_a), logits_a)
        # ... while recomputing the cache from the same plan does.
        assert (realizer(w_t, *times, realizer.prepare(plan_a)) - logits_a).abs().max() > 1e-3
        # The keys and values are read from the cache object itself: zeroing its values changes the logits.
        zeroed = BridgeKVCache(cache_a.keys, tuple(torch.zeros_like(v) for v in cache_a.values), cache_a.mask)
        assert (realizer(w_t, *times, zeroed) - logits_a).abs().max() > 1e-3


# ================================================================================================
# Conditional FS-DFM realizer
# ================================================================================================
@pytest.mark.parametrize("kind", ["fs_dfm", "logit_kl"])
def test_prepare_accepts_only_a_predicted_plan(tiny_config, kind: str) -> None:
    """Sec. 5.1: conditioning on f_theta_bar(y) leaks the target; prepare must refuse it (TypeError)."""
    model = _model(_config(tiny_config, realizer={"kind": kind}))
    plan = _plan(model, 0)
    num_slots = model.planner.predictor.num_slots
    target_encoder = TargetEncoder.from_online(model.planner.context_encoder)
    tokens = torch.randint(0, 256, (B, 14), generator=torch.Generator().manual_seed(1))
    target = target_encoder(tokens, torch.arange(14).repeat(B, 1) % num_slots, num_slots)
    assert isinstance(target, TargetOutput) and target.s_y.shape == plan.z_target.shape
    lookalike = SimpleNamespace(**{f.name: getattr(plan, f.name) for f in dataclasses.fields(plan)})
    for bad in (target, target.s_y, plan.z_target, lookalike, (plan.z_target, plan.s_x, plan.mode)):
        with pytest.raises(TypeError):
            model.realizer.prepare(bad)
        with pytest.raises(TypeError):
            require_predicted_plan(bad)
    assert require_predicted_plan(plan) is plan
    with torch.no_grad():
        cache = model.realizer.prepare(plan)
    assert cache.num_layers == model.realizer.bridge.num_layers and cache.mask.shape == (B, num_slots + M_C + 1)


def test_with_null_stacks_the_conditional_and_null_caches(tiny_config) -> None:
    model = _model(_config(tiny_config))
    realizer, bridge = model.realizer, model.realizer.bridge
    heads = bridge.num_heads
    plan = _plan(model, 0)
    with torch.no_grad():
        cond = realizer.prepare(plan)
        both = realizer.prepare(plan, with_null=True)
    m_tilde = cond.mask.shape[1]
    assert both.mask.shape == (2 * B, m_tilde)
    assert torch.equal(both.mask[:B], cond.mask)
    null_mask = torch.zeros(B, m_tilde, dtype=torch.bool)
    null_mask[:, 0] = True                                         # the learned null token alone
    assert torch.equal(both.mask[B:], null_mask)
    null64 = _w(bridge.null_token)
    for layer in range(bridge.num_layers):
        torch.testing.assert_close(both.keys[layer][:B], cond.keys[layer], rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(both.values[layer][:B], cond.values[layer], rtol=RTOL, atol=ATOL)
        for got, proj in ((both.keys[layer], bridge.key_proj[layer]), (both.values[layer], bridge.value_proj[layer])):
            ref = (null64 @ _w(proj.weight).T).reshape(heads, -1)  # [H, d_h]
            torch.testing.assert_close(got[B:, :, 0, :].double(), ref.expand(B, -1, -1), rtol=RTOL, atol=ATOL)
    # Semantics: the null half behaves exactly like the bridge's null conditioning.
    _open_gates(realizer.backbone, tables=True, projection=True)
    w_t = torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(2))
    t, h = _times(0.25, 0.25)
    with torch.no_grad():
        stacked = realizer(torch.cat([w_t, w_t]), torch.cat([t, t]), torch.cat([h, h]), both)
        null_only = realizer(w_t, t, h, bridge.precompute_kv(bridge.null_conditioning(B)))
        conditional = realizer(w_t, t, h, cond)
    # Equal in exact arithmetic (a masked softmax over M~ keys vs one key; batch 2B vs B), but the float32
    # reductions differ: ulp-level differences of O(1) activations, amplified by 3 modulated blocks, reach ~1e-6.
    torch.testing.assert_close(stacked[B:], null_only, rtol=RTOL, atol=1e-5)
    torch.testing.assert_close(stacked[:B], conditional, rtol=RTOL, atol=1e-5)
    assert (stacked[:B] - stacked[B:]).abs().max() > 1e-3


def _assert_uniform_counts(counts: Tensor, total: int, support: int) -> None:
    """Each count within 5 binomial standard errors of total / support."""
    p = 1.0 / support
    se = math.sqrt(total * p * (1 - p))
    assert float((counts.double() - total * p).abs().max()) <= 5 * se


def test_sample_source_with_allowed_draws_uniformly_from_the_allowed_tokens(tiny_config) -> None:
    realizer = _model(_config(tiny_config)).realizer
    allowed = torch.zeros(3, S, dtype=torch.bool)
    allowed[0, [5, 40, 41, 200, PAD]] = True
    allowed[1, torch.arange(0, S, 7)] = True
    allowed[2] = True
    allowed[2, 100:200] = False
    n = 40_000
    draws = realizer.sample_source(3, n, torch.device("cpu"), torch.Generator().manual_seed(0), allowed=allowed)
    assert draws.shape == (3, n) and draws.dtype == torch.long
    for row in range(3):
        assert bool(allowed[row][draws[row]].all())               # every draw is allowed
        ids = allowed[row].nonzero().flatten()
        _assert_uniform_counts(torch.bincount(draws[row], minlength=S)[ids], n, ids.numel())
    with pytest.raises(ValueError):
        realizer.sample_source(3, 4, torch.device("cpu"), allowed=torch.zeros(3, S, dtype=torch.bool))


def test_sample_source_without_allowed_is_uniform_over_v_and_mask_source_is_all_mask(tiny_config) -> None:
    realizer = _model(_config(tiny_config)).realizer
    n = 200_000
    draws = realizer.sample_source(1, n, torch.device("cpu"), torch.Generator().manual_seed(1))
    assert int(draws.min()) >= 0 and int(draws.max()) < S
    _assert_uniform_counts(torch.bincount(draws.flatten(), minlength=S), n, S)
    masked = _model(_config(tiny_config, realizer={"source": "mask"})).realizer
    allowed = torch.ones(2, S, dtype=torch.bool)
    out = masked.sample_source(2, 9, torch.device("cpu"), allowed=allowed)
    assert torch.equal(out, torch.full((2, 9), MASK))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_fs_dfm_loss_is_the_eq12_cross_entropy(tiny_config, reduction: str) -> None:
    realizer = _model(_config(tiny_config)).realizer
    g = torch.Generator().manual_seed(3)
    logits = 3.0 * torch.randn(B, N, S, generator=g)
    w_1 = torch.randint(0, S, (B, N), generator=g)
    nll = -_log_softmax64(logits.double()).gather(-1, w_1[..., None]).squeeze(-1)     # [B, N]
    expected = nll.sum() / (B * N) if reduction == "mean" else nll.sum() / B          # docstring: mean / per-sample
    got = realizer.loss(logits, w_1, reduction)
    torch.testing.assert_close(got.double(), expected, rtol=RTOL, atol=ATOL)
    with pytest.raises(ValueError):
        realizer.loss(logits, w_1, "none")


def test_stage2_gradients_reach_realizer_and_bridge_but_not_the_planner(tiny_config) -> None:
    model = _model(_config(tiny_config))
    realizer, bridge = model.realizer, model.realizer.bridge
    _open_gates(realizer.backbone, tables=True, projection=True)
    modes = torch.tensor([1, 4])
    plan = model.planner.plan(_context(0), mode=modes)              # with autograd, as a joint stage would
    assert plan.z_target.requires_grad and plan.s_x.requires_grad
    frozen = plan.detach()
    for value in (frozen.s_x, frozen.z_target, frozen.sigma_bar, frozen.head_predictions, frozen.mode_logits):
        assert not value.requires_grad
    g = torch.Generator().manual_seed(4)
    w_1 = torch.randint(0, S, (B, N), generator=g)
    sample = mixture_path_sample(w_1, realizer.schedule, S, (2, 4, 8), generator=g)
    realizer.loss(realizer(sample.w_t, sample.t, sample.h, realizer.prepare(frozen)), w_1).backward()
    assert all(p.grad is None for p in model.planner.parameters())
    for p in (realizer.token_embedding.weight, realizer.backbone.head.weight, realizer.backbone.position.weight,
              realizer.backbone.block_modulation[-1].weight, bridge.mode_embedding.weight):
        assert _nonzero_grad(p)
    for block in realizer.backbone.blocks:
        assert _nonzero_grad(block.table)
    for layer in range(bridge.num_layers):
        for proj in (bridge.query_proj[layer], bridge.out_proj[layer]):
            assert _nonzero_grad(proj.weight)
        for proj in (bridge.key_proj[layer], bridge.value_proj[layer]):
            assert _nonzero_grad(_raw_weight(proj))                 # through the spectral normalisation
    # Control: without the detach (Stage 3) the same loss reaches the planner through Z_target.
    model.zero_grad(set_to_none=True)
    realizer.loss(realizer(sample.w_t, sample.t, sample.h, realizer.prepare(plan)), w_1).backward()
    assert _nonzero_grad(model.planner.predictor.heads.weight)


# ================================================================================================
# Mixture path (Eq. 11) and kappa schedules
# ================================================================================================
KAPPA = {
    "linear": lambda t: t,
    "sqrt": math.sqrt,
    "cosine": lambda t: 1.0 - math.cos(math.pi * t / 2.0),
}


@pytest.mark.parametrize("kind", ["linear", "sqrt", "cosine"])
def test_kappa_schedule_closed_forms_and_exact_jump_mass(kind: str) -> None:
    sched, ref = KappaSchedule(kind), KAPPA[kind]
    grid = [0.0, 0.05, 0.25, 0.5, 0.8, 1.0]
    for t in grid:
        assert sched.kappa(t) == pytest.approx(ref(t), abs=1e-15)
    ts = torch.tensor(grid, dtype=torch.float64)
    torch.testing.assert_close(sched.kappa(ts), torch.tensor([ref(t) for t in grid], dtype=torch.float64))
    assert sched.kappa(0.0) == 0.0 and sched.kappa(1.0) == pytest.approx(1.0, abs=1e-15)
    for k_steps in (1, 2, 4, 8):
        for k in range(k_steps):
            t, h = k / k_steps, 1 / k_steps
            expected = (ref(t + h) - ref(t)) / (1.0 - ref(t))                       # Eq. 15
            assert sched.exact_jump_mass(t, h) == pytest.approx(expected, abs=1e-12)
            assert 0.0 <= sched.exact_jump_mass(t, h) <= 1.0 + 1e-12


@pytest.mark.parametrize("kind", [
    "linear",
    "sqrt",
    pytest.param("cosine"),
])
def test_exact_jump_mass_rejects_t_equal_to_one(kind: str) -> None:
    with pytest.raises(ValueError):
        KappaSchedule(kind).exact_jump_mass(1.0, 0.25)


def _five_se_proportion(hits: int, total: int, p: float) -> bool:
    return abs(hits / total - p) <= 5 * math.sqrt(p * (1 - p) / total)


@pytest.mark.parametrize("budgets", [(2, 4, 8), (1, 3, 5, 6)])
def test_grid_time_sampling_draws_k_over_K_with_K_from_the_budgets(budgets: tuple[int, ...]) -> None:
    rows = 24_000
    w_1 = torch.zeros(rows, 1, dtype=torch.long)
    s = mixture_path_sample(w_1, KappaSchedule("linear"), S, budgets, generator=torch.Generator().manual_seed(0))
    assert s.t.shape == s.h.shape == (rows,)
    k_steps = torch.round(1.0 / s.h.double())
    torch.testing.assert_close(s.h.double(), 1.0 / k_steps, rtol=0.0, atol=1e-7)   # h = 1/K
    assert set(k_steps.long().unique().tolist()) <= set(budgets)
    k = s.t.double() * k_steps
    torch.testing.assert_close(k, torch.round(k), rtol=0.0, atol=1e-5)              # t = k/K
    k = torch.round(k).long()
    assert bool((k >= 0).all()) and bool((k < k_steps.long()).all())                # never t = 1
    for big_k in budgets:
        rows_k = k_steps.long() == big_k
        assert _five_se_proportion(int(rows_k.sum()), rows, 1 / len(budgets))     # K ~ U(budgets)
        counts = torch.bincount(k[rows_k], minlength=big_k)
        for c in counts.tolist():                                                     # k | K ~ U{0..K-1}
            assert _five_se_proportion(c, int(rows_k.sum()), 1 / big_k)


def test_uniform_time_sampling_draws_t_uniformly_with_h_from_the_budgets() -> None:
    rows = 20_000
    s = mixture_path_sample(torch.zeros(rows, 1, dtype=torch.long), KappaSchedule("linear"), S, (2, 4, 8),
                            time_sampling="uniform", generator=torch.Generator().manual_seed(1))
    t = s.t.double()
    assert bool((t >= 0).all()) and bool((t < 1).all())
    assert abs(float(t.mean()) - 0.5) <= 5 * math.sqrt(1 / 12 / rows)
    assert abs(float((t < 0.25).double().mean()) - 0.25) <= 5 * math.sqrt(0.25 * 0.75 / rows)
    assert set(torch.round(1 / s.h).long().unique().tolist()) == {2, 4, 8}
    with pytest.raises(ValueError):
        mixture_path_sample(torch.zeros(2, 1, dtype=torch.long), KappaSchedule("linear"), S, (2,), time_sampling="x")


@pytest.mark.parametrize("kind", ["linear", "cosine", "sqrt"])
def test_mask_source_arrivals_follow_kappa_t_independently_per_position(kind: str) -> None:
    rows, positions = 4096, 64
    g = torch.Generator().manual_seed(2)
    w_1 = torch.randint(0, 256, (rows, positions), generator=g)                       # never [MASK]
    s = mixture_path_sample(w_1, KappaSchedule(kind), S, (2, 4, 8), source="mask", mask_id=MASK, generator=g)
    assert torch.equal(s.w_0, torch.full_like(w_1, MASK))
    arrived = s.w_t == w_1
    assert torch.equal(s.w_t, torch.where(arrived, w_1, torch.full_like(w_1, MASK)))   # [MASK] where not arrived
    for t in s.t.unique().tolist():
        sel = s.t == t
        kappa = KAPPA[kind](t)
        total = int(sel.sum()) * positions
        hits = int(arrived[sel].sum())
        if kappa == 0.0:
            assert hits == 0
        else:
            assert _five_se_proportion(hits, total, kappa), (t, hits / total, kappa)
    # Arrivals are independent across the positions of a row: per-row counts are Binomial(N, kappa).
    sel = s.t == 0.5
    p = KAPPA[kind](0.5)
    counts = arrived[sel].double().sum(dim=1)
    n_rows = counts.numel()
    var, mu4 = positions * p * (1 - p), positions * p * (1 - p) * (1 + 3 * (positions - 2) * p * (1 - p))
    se_var = math.sqrt((mu4 - var ** 2 * (n_rows - 3) / (n_rows - 1)) / n_rows)
    assert abs(float(counts.var()) - var) <= 5 * se_var
    with pytest.raises(ValueError):
        mixture_path_sample(w_1[:2], KappaSchedule(kind), S, (2,), source="mask")


def test_uniform_source_is_uniform_and_states_mix_source_and_target() -> None:
    rows, positions = 2048, 64
    g = torch.Generator().manual_seed(3)
    w_1 = torch.randint(0, S, (rows, positions), generator=g)
    s = mixture_path_sample(w_1, KappaSchedule("cosine"), S, (2, 4, 8), source="uniform", generator=g)
    assert int(s.w_0.min()) >= 0 and int(s.w_0.max()) < S
    _assert_uniform_counts(torch.bincount(s.w_0.flatten(), minlength=S), rows * positions, S)
    assert bool(((s.w_t == s.w_0) | (s.w_t == w_1)).all())                          # Eq. 11 support
    differ = s.w_0 != w_1
    for t in s.t.unique().tolist():
        sel = (s.t == t)[:, None] & differ
        kappa = KAPPA["cosine"](t)
        hits, total = int((s.w_t == w_1)[sel].sum()), int(sel.sum())
        if kappa == 0.0:
            assert hits == 0
        else:
            assert _five_se_proportion(hits, total, kappa)


# ================================================================================================
# Logit-KL realizer
# ================================================================================================
@pytest.mark.parametrize("eps", [0.05, 0.2])
def test_target_logits_are_eq18_and_softmax_to_the_smoothed_one_hot(tiny_config, eps: float) -> None:
    realizer = _model(_config(tiny_config, realizer={"kind": "logit_kl", "logit_smoothing": eps})).realizer
    a, b = _smoothed_coefficients64(eps, S)
    assert realizer.a == pytest.approx(a, rel=1e-12) and realizer.b == pytest.approx(b, rel=1e-12)
    w_1 = torch.randint(0, S, (B, N), generator=torch.Generator().manual_seed(0))
    one_hot = F.one_hot(w_1, S).double()
    l1 = realizer.target_logits(w_1)
    assert l1.shape == (B, N, S)
    torch.testing.assert_close(l1.double(), a * one_hot + b, rtol=RTOL, atol=ATOL)
    smoothed = (1.0 - eps) * one_hot + eps / S                                    # x_1^eps
    torch.testing.assert_close(_softmax64(l1.double()), smoothed, rtol=RTOL, atol=ATOL)


def test_training_state_lies_on_the_eq19_path_from_a_gaussian_source(tiny_config) -> None:
    s_std = 2.5
    realizer = _model(_config(tiny_config, realizer={"kind": "logit_kl", "logit_source_std": s_std})).realizer
    rows, positions = 512, 16
    g = torch.Generator().manual_seed(1)
    w_1 = torch.randint(0, S, (rows, positions), generator=g)
    l_t, t, h = realizer.training_state(w_1, (2, 4, 8), generator=g)
    assert l_t.shape == (rows, positions, S)
    k_steps = torch.round(1 / h.double())
    assert set(k_steps.long().unique().tolist()) <= {2, 4, 8}
    k = t.double() * k_steps
    assert torch.equal(k, torch.round(k)) and bool((k < k_steps).all())          # t = k/K on the grid
    a, b = _smoothed_coefficients64(realizer.smoothing, S)
    l1 = a * F.one_hot(w_1, S).double() + b
    t64 = t.double()[:, None, None]
    l0 = (l_t.double() - t64 * l1) / (1.0 - t64)                                 # invert Eq. 19
    on_target = F.one_hot(w_1, S).bool()
    for x in (l0[on_target], l0[~on_target]):
        n = x.numel()
        assert abs(float(x.mean())) <= 5 * s_std / math.sqrt(n)
        assert abs(float(x.var()) - s_std ** 2) <= 5 * s_std ** 2 * math.sqrt(2 / (n - 1))
        p_in = math.erf(1 / math.sqrt(2))                                          # P(|x| <= s)
        assert _five_se_proportion(int((x.abs() <= s_std).sum()), n, p_in)
    # Independent across positions and rows: empirical correlations of neighbouring draws vanish (5 SE).
    for u, v in ((l0[:, 0], l0[:, 1]), (l0[0::2], l0[1::2])):
        u, v = u.flatten(), v.flatten()
        corr = float(((u - u.mean()) * (v - v.mean())).mean() / (u.std() * v.std()))
        assert abs(corr) <= 5 / math.sqrt(u.numel())


@pytest.mark.parametrize("num_steps", [1, 2, 4, 8])
def test_euler_step_is_eq21_and_returns_a_p_at_the_last_step(tiny_config, num_steps: int) -> None:
    realizer = _model(_config(tiny_config, realizer={"kind": "logit_kl"})).realizer
    a, _ = _smoothed_coefficients64(realizer.smoothing, S)
    g = torch.Generator().manual_seed(num_steps)
    posterior = torch.softmax(2.0 * torch.randn(B, N, S, generator=g), dim=-1)
    l_k = torch.randn(B, N, S, generator=g)
    for k in range(num_steps):
        t, h = k / num_steps, 1 / num_steps
        out = realizer.euler_step(l_k, posterior, t, h)
        if k == num_steps - 1:                                                      # t_{K-1} = 1 - 1/K
            torch.testing.assert_close(out.double(), a * posterior.double(), rtol=RTOL, atol=ATOL)
            assert torch.equal(out, realizer.euler_step(l_k + 7.0, posterior, t, h))   # l_K does not see l_{K-1}
        else:
            expected = l_k.double() + h * (a * posterior.double() - l_k.double()) / (1.0 - t)
            torch.testing.assert_close(out.double(), expected, rtol=RTOL, atol=ATOL)
        l_k = out
    with pytest.raises(ValueError):
        realizer.euler_step(l_k, posterior, 0.75, 0.5)                            # would step past t = 1


def test_logit_kl_network_reads_softmax_of_the_state_through_the_embedding(tiny_config) -> None:
    model = _model(_config(tiny_config, realizer={"kind": "logit_kl"}))
    realizer = model.realizer
    seen: list[Tensor] = []
    handle = realizer.backbone.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach()))
    l_t = 3.0 * torch.randn(B, N, S, generator=torch.Generator().manual_seed(0))
    try:
        with torch.no_grad():
            logits = realizer(l_t, *_times(0.25, 0.25), realizer.prepare(_plan(model, 0)))
    finally:
        handle.remove()
    assert logits.shape == (B, N, S)
    expected = _softmax64(l_t.double()) @ _w(realizer.token_embedding)          # x_t E with x_t = softmax(l_t)
    torch.testing.assert_close(seen[0].double(), expected, rtol=RTOL, atol=ATOL)
    with pytest.raises(ValueError):
        realizer(l_t[..., :-1], *_times(0.25, 0.25), realizer.prepare(_plan(model, 0)))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_logit_kl_loss_is_the_cross_entropy_of_lemma_2(tiny_config, reduction: str) -> None:
    realizer = _model(_config(tiny_config, realizer={"kind": "logit_kl"})).realizer
    g = torch.Generator().manual_seed(5)
    logits = 2.0 * torch.randn(B, N, S, generator=g)
    w_1 = torch.randint(0, S, (B, N), generator=g)
    nll = -_log_softmax64(logits.double()).gather(-1, w_1[..., None]).squeeze(-1)
    expected = nll.mean() if reduction == "mean" else nll.sum(dim=1).mean()
    torch.testing.assert_close(realizer.loss(logits, w_1, reduction).double(), expected, rtol=RTOL, atol=ATOL)


# ================================================================================================
# Plan readout (slot decoder)
# ================================================================================================
LATENT, HIDDEN, L_TEXT, BUCKETS, BUCKET = 16, 32, 24, 4, 16


def _readout() -> PlanReadout:
    return PlanReadout(LATENT, S, HIDDEN, L_TEXT, BUCKETS, BUCKET)


def test_readout_shapes_and_per_slot_locality() -> None:
    readout = _readout()
    z = torch.randn(B, 5, LATENT, generator=torch.Generator().manual_seed(0))
    out = readout(z)
    assert out.kind_logits.shape == (B, 5, NUM_KINDS) and out.slot_features.shape == (B, 5, HIDDEN)
    assert out.inventory_logits.shape == (B, S) and out.length_logits.shape == (B, BUCKETS)
    assert readout.text_logits(out.slot_features[0]).shape == (5, L_TEXT, S)
    # Kind and slot features are per slot: perturbing slot 2 leaves the other slots unchanged.
    z2 = z.clone()
    z2[:, 2] += 1.0
    out2 = readout(z2)
    keep = [0, 1, 3, 4]
    assert torch.equal(out.kind_logits[:, keep], out2.kind_logits[:, keep])
    assert torch.equal(out.slot_features[:, keep], out2.slot_features[:, keep])
    # The text decoder is parallel over slots: one slot's text logits ignore the others.
    feats = out.slot_features[0]
    both = readout.text_logits(feats)
    torch.testing.assert_close(readout.text_logits(feats[1:2]), both[1:2], rtol=RTOL, atol=ATOL)
    with pytest.raises(ValueError):
        readout(torch.randn(B, 5, LATENT + 1))


def _targets(seed: int, with_text: bool = True) -> tuple[ReadoutTargets, list[list[int]]]:
    """Two samples with four slots; text slots have left-aligned texts of lengths 5, L (full), 0 and 3."""
    g = torch.Generator().manual_seed(seed)
    F_, V_, E_, I_, C_ = (int(k) for k in (SlotKind.FUNCTION, SlotKind.VARIABLE, SlotKind.EMPTY, SlotKind.IMPORT,
                                           SlotKind.CLASS))
    kinds = torch.tensor([[F_, V_, E_, F_], [I_, E_, V_, C_]]) if with_text else torch.tensor([[V_, V_, E_, E_],
                                                                                             [E_, E_, V_, V_]])
    text_mask = torch.isin(kinds, torch.tensor([F_, I_, C_]))
    texts = torch.randint(0, 256, (2, 4, L_TEXT), generator=g)                 # non-text slots: junk, ignored
    lengths = {(0, 0): 5, (0, 3): L_TEXT, (1, 0): 0, (1, 3): 3}
    for (b, m), n_tok in lengths.items():
        texts[b, m, n_tok:] = PAD
    inventory = (torch.rand(2, S, generator=g) < 0.1).float()
    bucket = torch.tensor([1, 3])
    return ReadoutTargets(kinds, texts, text_mask, inventory, bucket), [[b, m, n] for (b, m), n in lengths.items()]


def _readout_loss_reference64(readout: PlanReadout, out: ReadoutOutput, targets: ReadoutTargets,
                              slots: list[list[int]], weights: tuple[float, ...]) -> dict[str, float]:
    kind = -_log_softmax64(_w(out.kind_logits)).gather(-1, targets.kinds[..., None]).mean()
    text_nll, count = 0.0, 0
    for b, m, n_tok in slots:
        if not bool(targets.text_mask[b, m]):
            continue
        logp = _log_softmax64(_w(readout.text_logits(out.slot_features[b, m][None]))[0])     # [L, S]
        for pos in range(min(n_tok + 1, L_TEXT)):              # the tokens, then the first [PAD] only
            text_nll -= float(logp[pos, targets.texts[b, m, pos]])
            count += 1
    text = text_nll / count if count else 0.0
    x, y = _w(out.inventory_logits), targets.inventory.double()
    inventory = (torch.clamp(x, min=0) - x * y + torch.log1p(torch.exp(-x.abs()))).mean()  # BCE with logits
    length = -_log_softmax64(_w(out.length_logits)).gather(-1, targets.length_bucket[:, None]).mean()
    terms = {"readout_kind": float(kind), "readout_text": text, "readout_inventory": float(inventory),
             "readout_length": float(length)}
    terms["total"] = sum(w * terms[k] for w, k in zip(weights, ("readout_kind", "readout_text", "readout_inventory",
                                                                 "readout_length"), strict=True))
    return terms


@pytest.mark.parametrize("weights", [(1.0, 1.0, 1.0, 0.1), (0.5, 2.0, 0.3, 1.0)])
def test_readout_loss_components_match_hand_computation(weights: tuple[float, ...]) -> None:
    readout = _readout()
    z = torch.randn(2, 4, LATENT, generator=torch.Generator().manual_seed(1))
    out = readout(z)
    targets, slots = _targets(2)
    total, terms = readout.loss(out, targets, PAD, weights)
    expected = _readout_loss_reference64(readout, out, targets, slots, weights)
    for key in ("readout_kind", "readout_text", "readout_inventory", "readout_length"):
        assert float(terms[key]) == pytest.approx(expected[key], rel=1e-5, abs=1e-6), key
    assert float(total.detach()) == pytest.approx(expected["total"], rel=1e-5, abs=1e-6)


def test_readout_loss_without_text_slots_has_a_zero_text_term() -> None:
    readout = _readout()
    out = readout(torch.randn(2, 4, LATENT, generator=torch.Generator().manual_seed(3)))
    targets, slots = _targets(4, with_text=False)
    total, terms = readout.loss(out, targets, PAD)
    expected = _readout_loss_reference64(readout, out, targets, slots, (1.0, 1.0, 1.0, 0.1))
    assert float(terms["readout_text"]) == 0.0
    assert float(total.detach()) == pytest.approx(expected["total"], rel=1e-5, abs=1e-6)


def test_length_bucket_k_covers_lengths_in_k_bs_to_k_plus_1_bs() -> None:
    readout = _readout()
    lengths = torch.tensor([-3, 0, 1, 15, 16, 17, 32, 33, 48, 49, 64, 65, 1000])
    expected = [min(BUCKETS - 1, math.ceil(max(int(n), 1) / BUCKET) - 1) for n in lengths]
    assert readout.length_bucket(lengths).tolist() == expected


def _spelling_readout(text: str) -> PlanReadout:
    """A readout whose text decoder spells ``text`` (then [PAD]s) for a zero slot feature.

    Position l gets the input 10 e_l; LayerNorm, an identity layer and GELU keep coordinate l large
    (~5.6) and the others slightly negative (~-0.08); the last layer maps coordinate l to the l-th token.
    """
    ids = TOK.encode(text)
    ids = ids + [PAD] * (L_TEXT - len(ids))
    readout = _readout()
    norm, first, _, last = readout.text_out
    with torch.no_grad():
        readout.text_positions.copy_(10.0 * torch.eye(L_TEXT, HIDDEN))
        norm.weight.fill_(1.0)
        norm.bias.zero_()
        first.weight.copy_(torch.eye(HIDDEN))
        first.bias.zero_()
        last.weight.zero_()
        last.bias.zero_()
        for pos, token in enumerate(ids):
            last.weight[token, pos] = 1.0
    return readout


def _hand_output(slot_kinds: list[list[int]], inventory: Tensor, length_argmax: list[int]) -> ReadoutOutput:
    kinds = torch.tensor(slot_kinds)
    kind_logits = -5.0 * torch.ones(*kinds.shape, NUM_KINDS)
    kind_logits.scatter_(-1, kinds[..., None], 5.0)
    length_logits = torch.zeros(len(length_argmax), BUCKETS)
    length_logits[torch.arange(len(length_argmax)), torch.tensor(length_argmax)] = 3.0
    return ReadoutOutput(kind_logits, inventory, length_logits, torch.zeros(*kinds.shape, HIDDEN))


def _word_bearing(token: int) -> bool:
    """Identifier character (letter or '_') in the token's surface text (byte tokens only here)."""
    return token < 256 and any(c.isalpha() or c == "_" for c in bytes([token]).decode("utf-8", "replace"))


@pytest.mark.parametrize("threshold", [0.5, 0.75])
def test_decode_parses_signatures_and_licenses_identifier_tokens(threshold: float) -> None:
    readout = _spelling_readout("def f(x: int) -> int")
    assert TOK.decode(readout.text_logits(torch.zeros(1, HIDDEN)).argmax(-1)[0].tolist()) == "def f(x: int) -> int"
    fn, var, empty = int(SlotKind.FUNCTION), int(SlotKind.VARIABLE), int(SlotKind.EMPTY)
    inventory = torch.full((2, S), -4.0)
    for ch in "fx_(1":
        inventory[0, ord(ch)] = 4.0                     # sigmoid 0.98
    inventory[0, ord("q")] = 0.8                        # sigmoid 0.69: licensed at 0.5, not at 0.75
    inventory[0, MASK] = 4.0                            # special token: never licensed (not identifier-bearing)
    out = _hand_output([[var, fn, empty], [var, empty, empty]], inventory, [2, 0])
    plans, lengths = readout.decode(out, TOK, TreeSitterASTExtractor("python"), VocabularyAnalysis(TOK, "python"),
                                    threshold)
    assert lengths.tolist() == [(2 + 1) * BUCKET, (0 + 1) * BUCKET]          # upper end of the argmax bucket
    probs = torch.sigmoid(inventory)
    for b, plan in enumerate(plans):
        expected = {i for i in range(S) if float(probs[b, i]) > threshold and _word_bearing(i)}
        assert plan.licensed_tokens == frozenset(expected)
        assert plan.source == "readout"
    assert plans[0].licensed_tokens == frozenset(map(ord, "fx_q" if threshold == 0.5 else "fx_"))
    sig = plans[0].signatures["f"]
    assert set(plans[0].signatures) == {"f"} and plans[0].declared_identifiers == frozenset({"f"})
    assert sig.name == "f" and sig.return_annotation == "int"
    assert [(p.name, p.annotation, p.has_default, p.kind) for p in sig.parameters] == [("x", "int", False,
                                                                                         "positional")]
    assert plans[1].signatures == {} and plans[1].declared_identifiers == frozenset()


@pytest.mark.parametrize("text", ["def f(x: int -> int", "x = 1", "f(x)"])
def test_decode_drops_slot_texts_that_do_not_parse_as_one_signature(text: str) -> None:
    readout = _spelling_readout(text)
    out = _hand_output([[int(SlotKind.FUNCTION)]], torch.full((1, S), -4.0), [1])
    plans, _ = readout.decode(out, TOK, TreeSitterASTExtractor("python"), VocabularyAnalysis(TOK, "python"))
    assert plans[0].signatures == {} and plans[0].declared_identifiers == frozenset()


# ================================================================================================
# Planner and model assembly
# ================================================================================================
def test_plan_with_a_forced_mode_equals_the_three_networks_run_by_hand(tiny_config) -> None:
    planner = TextVPAPlanner.from_config(_config(tiny_config).planner).eval()
    context = _context(0)
    mask = torch.ones(B, M_C, dtype=torch.bool)
    mask[1, 7:] = False
    modes = torch.tensor([3, planner.num_modes - 1])                              # an interior and the last mode
    with torch.no_grad():
        plan = planner.plan(context, mask, mode=modes)
        s_x = planner.context_encoder(context, mask)                               # pass 1
        logits = planner.selector(s_x, mask)                                       # pass 2
        out = planner.predictor(s_x, modes, context_mask=mask)                     # pass 3
        skipped = planner.plan(context, mask, mode=modes, run_selector=False)
    assert torch.equal(plan.s_x, s_x) and torch.equal(plan.context_mask, mask)
    assert torch.equal(plan.mode_logits, logits)
    assert torch.equal(plan.mode, modes) and torch.equal(plan.mode_input, modes)
    assert torch.equal(plan.z_target, out.z_target) and torch.equal(plan.sigma_bar, out.sigma_bar)
    assert torch.equal(plan.head_predictions, out.head_predictions)
    assert skipped.mode_logits is None and torch.equal(skipped.z_target, out.z_target)
    with pytest.raises(ValueError):
        planner.plan(context, mask, mode=torch.tensor([1]))


def test_gumbel_strategy_feeds_a_straight_through_one_hot_to_g_phi(tiny_config) -> None:
    planner = TextVPAPlanner.from_config(_config(tiny_config).planner).eval()
    plan = planner.plan(_context(1), strategy="gumbel", generator=torch.Generator().manual_seed(0))
    u = plan.mode_input
    assert u.dtype.is_floating_point and u.shape == (B, planner.num_modes)
    torch.testing.assert_close(u, F.one_hot(plan.mode, planner.num_modes).to(u.dtype), rtol=0.0, atol=1e-6)
    assert torch.equal(u.argmax(dim=1), plan.mode)
    with torch.no_grad():
        assert torch.equal(planner.predictor(plan.s_x, u).z_target, plan.z_target)       # g_phi consumed u
        torch.testing.assert_close(planner.predictor(plan.s_x, plan.mode).z_target, plan.z_target, rtol=RTOL,
                                   atol=1e-5)
    plan.z_target.square().sum().backward()                                               # straight-through
    assert any(_nonzero_grad(p) for p in planner.selector.parameters())


@pytest.mark.parametrize("kind", ["fs_dfm", "logit_kl"])
def test_from_config_builds_the_configured_realizer_and_readout(tiny_config, kind: str) -> None:
    cfg = _config(tiny_config, realizer={"kind": kind, "schedule": "cosine", "source": "mask", "logit_smoothing": 0.1,
                                         "logit_source_std": 2.0}, readout={"length_bucket_size": 24})
    model = TextVPAModel.from_config(cfg, mask_id=MASK)
    realizer, r = model.realizer, cfg.realizer
    assert type(realizer) is {"fs_dfm": ConditionalFSDFMRealizer, "logit_kl": LogitKLRealizer}[kind]
    assert model.realizer_kind == kind and realizer.vocab_size == S and realizer.max_len == r.max_output_len
    bridge = realizer.bridge
    assert (bridge.latent_dim, bridge.hidden_dim, bridge.num_layers, bridge.num_heads) == (
        cfg.planner.latent_dim, r.hidden_dim, r.num_layers, r.num_heads)
    assert len(realizer.backbone.blocks) == r.num_layers and realizer.backbone.head.out_features == S
    if kind == "fs_dfm":
        assert realizer.schedule.kind == "cosine" and realizer.source == "mask" and realizer.mask_id == MASK
    else:
        a, b = _smoothed_coefficients64(0.1, S)
        assert realizer.a == pytest.approx(a) and realizer.b == pytest.approx(b) and realizer.source_std == 2.0
    readout = model.readout
    assert isinstance(readout, PlanReadout)
    assert (readout.latent_dim, readout.vocab_size, readout.max_text_tokens, readout.length_bucket_size) == (
        cfg.planner.latent_dim, S, cfg.readout.max_text_tokens, 24)
    assert readout.num_length_buckets == math.ceil(r.max_output_len / 24) == 3
    assert model.planner.context_encoder.vocab_size == S
    trained = {id(p) for p in model.realizer_parameters()}
    assert all(id(p) in trained for p in (*realizer.parameters(), *readout.parameters()))
    assert not any(id(p) in trained for p in model.planner.parameters())
    assert TextVPAModel.from_config(_config(tiny_config, readout={"enabled": False})).readout is None


def test_from_config_rejects_mismatched_vocabulary_sizes(tiny_config) -> None:
    data = tiny_config.model_dump()
    data["planner"]["vocab_size"], data["realizer"]["vocab_size"] = S, S + 1
    with pytest.raises(ValueError):
        TextVPAModel.from_config(TextVPAConfig.model_validate(data), mask_id=MASK)


@pytest.mark.parametrize("kind", ["fs_dfm", "logit_kl"])
def test_realizer_runs_in_float64(tiny_config, kind: str) -> None:
    model = _model(_config(tiny_config, realizer={"kind": kind})).double()
    plan = _plan(model, 0)
    state = (torch.randint(0, S, (B, N)) if kind == "fs_dfm"
             else torch.randn(B, N, S, dtype=torch.float64))
    t, h = torch.full((B,), 0.25, dtype=torch.float64), torch.full((B,), 0.25, dtype=torch.float64)
    with torch.no_grad():
        logits = model.realizer(state, t, h, model.realizer.prepare(plan))
    assert logits.dtype == torch.float64
