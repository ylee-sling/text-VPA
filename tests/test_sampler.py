# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of Algorithm 1 (PR-JEPA Sec. 3.3) as implemented by ``PRJEPASampler``.

    1-3   s_x <- f_theta(c); u ~ pi_phi^h(. | s_x); Z <- g_phi(s_x, u, Delta)          passes 1-3
    5     (K_l, V_l) <- bridge projections of [Z; s_x; e(u)]/gamma                     once, fused into pass 4
    6     w <- N tokens drawn from p_0 (uniform or [MASK])
    7-12  for k = 0..K-1: t = k/K, h = 1/K, pi = (kappa_{t+h} - kappa_t)/(1 - kappa_t);
          P <- p_{1|t}(. | w, t, h, Z) (one pass); rho^i(j) = min(P_i(j) e^{-E/T}, c_max), j != w^i;
          with probability pi sum_j rho^i(j) jump to j ~ rho^i / sum rho^i                passes 4..K+3
    depth K + 3, independent of N; with the linear schedule the last step has pi = 1 (every position is
    redrawn); Lemma 1: with the oracle posterior the kernel maps p_t to p_{t+h}; Prop. 4(d): plan-mediated
    hard constraints survive fully parallel updates, draft-dependent ones need a final repair.
    Logit-KL (Sec. 3.4): l_{k+1} = l_k + (1/K)(a p - l_k)/(1 - t_k), tokens = argmax l_K, l_K = a p.

References are written from these statements (float64 where numeric) and never call the code under test;
admissibility and bracket validity are re-checked with independent rules below.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import Tensor

import text_vpa.inference.sampler as sampler_module
from tests.conftest import RTOL
from text_vpa.configs.schema import SamplerConfig, TextVPAConfig
from text_vpa.data.ast_parser import TreeSitterASTExtractor
from text_vpa.data.tokenization import ByteTokenizer
from text_vpa.inference.energy_terms import (
    AdmissibilityViolation,
    ConstraintEnergyEvaluator,
    EnergyTerm,
    GrammarEnergy,
    PlanConstraints,
    PlanVocabularyEnergy,
    SpecialTokenEnergy,
    VocabularyAnalysis,
)
from text_vpa.inference.sampler import DepthAudit, DepthAuditError, PRJEPASampler
from text_vpa.models.text_vpa_model import TextVPAModel

TOK = ByteTokenizer()
S = TOK.vocab_size
PAD, MASK = TOK.pad_id, TOK.mask_id
B, M_C, N = 2, 12, 32
PLANNER = ["f_theta", "pi_phi", "g_phi"]
REFERENCE = "def f(x):\n    return (x + 1) * [x][0]\n"        # licenses the letters d e f n r t u x
UNLICENSED = ord("q")
STRING_CHARS = "'\"#\\"
KAPPA: dict[str, Callable[[float], float]] = {
    "linear": lambda t: t,
    "sqrt": math.sqrt,
    "cosine": lambda t: 1.0 - math.cos(math.pi * t / 2.0),
}


# ------------------------------------------------------------------------------------------------
# Builders and recorders
# ------------------------------------------------------------------------------------------------
def _config(base: TextVPAConfig, **sections: dict) -> TextVPAConfig:
    data = base.with_vocab_size(S).model_dump()
    for name, values in sections.items():
        data[name].update(values)
    return TextVPAConfig.model_validate(data)


def _model(cfg: TextVPAConfig, open_gates: bool = False) -> TextVPAModel:
    """Tiny model; ``open_gates`` moves the adaLN-Zero parameters off zero so the plan, t and h matter."""
    model = TextVPAModel.from_config(cfg, mask_id=MASK).eval()
    if open_gates:
        g = torch.Generator().manual_seed(7)
        backbone = model.realizer.backbone
        with torch.no_grad():
            for block in backbone.blocks:
                block.table.copy_(0.5 * torch.randn(block.table.shape, generator=g))
            for proj in (backbone.block_modulation[-1], backbone.final_modulation[-1]):
                proj.weight.copy_(0.5 * torch.randn(proj.weight.shape, generator=g))
    return model


def _sampler(model: TextVPAModel, cfg: TextVPAConfig, energy: ConstraintEnergyEvaluator | None = None,
             **options: object) -> PRJEPASampler:
    return PRJEPASampler(model, TOK, cfg.realizer, SamplerConfig(**options), energy)


def _context(seed: int, batch: int = B) -> Tensor:
    return torch.randint(0, 256, (batch, M_C), generator=torch.Generator().manual_seed(seed))


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _record_realizer(model: TextVPAModel) -> tuple[list[tuple], list[Tensor], list]:
    """Inputs (state, t, h, cache) and output logits of every F_psi pass."""
    inputs: list[tuple] = []
    outputs: list[Tensor] = []
    handles = [
        model.realizer.register_forward_pre_hook(
            lambda _m, args: inputs.append(tuple(a.detach().clone() if isinstance(a, Tensor) else a for a in args))),
        model.realizer.register_forward_hook(lambda _m, _a, out: outputs.append(out.detach().clone())),
    ]
    return inputs, outputs, handles


def _record_calls(model: TextVPAModel) -> tuple[list[str], list]:
    """Independent call log of the generation networks (own forward hooks)."""
    names = {"f_theta": model.planner.context_encoder, "pi_phi": model.planner.selector,
             "g_phi": model.planner.predictor, "bridge": model.realizer.bridge, "F_psi": model.realizer}
    if model.readout is not None:
        names["readout"] = model.readout
    log: list[str] = []
    handles = [m.register_forward_hook(lambda *_a, n=n: log.append(n)) for n, m in names.items()]
    return log, handles


def _oracle(model: TextVPAModel, target: Tensor) -> None:
    """F_psi returns a delta posterior on ``target`` [B, N] (logit 0 there, -inf elsewhere); hooks still run."""
    def forward(state: Tensor, t: Tensor, h: Tensor, cache: object) -> Tensor:
        rows = state.shape[0]
        logits = torch.full((rows, target.shape[1], S), -math.inf)
        return logits.scatter(-1, target.repeat(rows // target.shape[0], 1)[..., None], 0.0)

    model.realizer.forward = forward


def _fixed_posterior(model: TextVPAModel, probs: dict[int, float]) -> None:
    """F_psi returns the same posterior ``probs`` (token -> probability) at every position."""
    def forward(state: Tensor, t: Tensor, h: Tensor, cache: object) -> Tensor:
        logits = torch.full((state.shape[0], state.shape[1], S), -math.inf)
        for token, p in probs.items():
            logits[..., token] = math.log(p)
        return logits

    model.realizer.forward = forward


class _SoftPenalty(EnergyTerm):
    """Test-local plan-mediated soft term: E(j) = weight for the listed tokens, 0 elsewhere."""

    name = "soft_penalty"
    plan_mediated = True

    def __init__(self, tokens: list[int], weight: float) -> None:
        super().__init__(weight)
        self.tokens = list(tokens)

    def plan_energy(self, analysis: VocabularyAnalysis, plan: PlanConstraints) -> np.ndarray:
        e = np.zeros(analysis.vocab_size, dtype=np.float64)
        e[self.tokens] = self.weight
        return e


def _softmax64(x: Tensor) -> Tensor:
    z = torch.exp(x - x.amax(dim=-1, keepdim=True))
    return z / z.sum(dim=-1, keepdim=True)


def _five_se_proportion(hits: int, total: int, p: float) -> bool:
    return abs(hits / total - p) <= 5 * math.sqrt(p * (1 - p) / total)


# ------------------------------------------------------------------------------------------------
# Energies and independent admissibility rules
# ------------------------------------------------------------------------------------------------
class _NoStringsOrComments(EnergyTerm):
    """Test-local plan-mediated hard term: forbids quote, comment and escape characters, so drafts stay in plain
    code and the independent bracket scanner below is exact."""

    name = "no_strings_or_comments"
    plan_mediated = True

    def plan_energy(self, analysis: VocabularyAnalysis, plan: PlanConstraints) -> np.ndarray:
        e = np.zeros(analysis.vocab_size, dtype=np.float64)
        for i, text in enumerate(analysis.strings):
            if not analysis.special[i] and any(c in STRING_CHARS for c in text):
                e[i] = self.weight
        return e


def _evaluator(grammar: bool, forbid_strings: bool) -> ConstraintEnergyEvaluator:
    """Hard terms only. The plan licenses identifier tokens with no keyword/builtin base, so with byte tokens
    the vocabulary term is informative (most letters are forbidden)."""
    terms: list[EnergyTerm] = [SpecialTokenEnergy(), PlanVocabularyEnergy(())]
    if forbid_strings:
        terms.append(_NoStringsOrComments())
    if grammar:
        terms.append(GrammarEnergy())
    return ConstraintEnergyEvaluator(TOK, terms=terms)


def _oracle_plan(energy: ConstraintEnergyEvaluator) -> PlanConstraints:
    structure = TreeSitterASTExtractor("python").parse(REFERENCE)
    return PlanConstraints.from_reference(REFERENCE, TOK, structure, energy.analysis)


def _word_bearing(token: int) -> bool:
    return token < 256 and any(c.isalpha() or c == "_" for c in bytes([token]).decode("utf-8", "replace"))


def _plan_admissible(forbid_strings: bool) -> Tensor:
    """A(Z) for the reference plan, from the rules: [PAD] but no other control token; identifier-bearing
    bytes only if they occur in the reference; optionally no quote / comment / escape characters."""
    licensed = {b for b in REFERENCE.encode() if _word_bearing(b)}
    ok = torch.zeros(S, dtype=torch.bool)
    ok[PAD] = True
    for i in range(256):
        if _word_bearing(i) and i not in licensed:
            continue
        if forbid_strings and chr(i) in STRING_CHARS:
            continue
        ok[i] = True
    return ok


def _output(row: list[int]) -> list[int]:
    """The output ends at the first [PAD]."""
    return row[: row.index(PAD)] if PAD in row else row


def _bracket_errors(row: list[int]) -> int:
    """Closers that do not match the innermost open bracket, plus an end ([PAD]) inside an open bracket."""
    pairs = {ord(")"): ord("("), ord("]"): ord("["), ord("}"): ord("{")}
    stack: list[int] = []
    errors = 0
    for token in row:
        if token == PAD:
            return errors + int(bool(stack))
        if token in pairs.values():
            stack.append(token)
        elif token in pairs:
            if stack and stack[-1] == pairs[token]:
                stack.pop()
            else:
                errors += 1
    return errors


# ================================================================================================
# Depth audit: exactly K + 3 sequential network evaluations
# ================================================================================================
@pytest.mark.parametrize("kind", ["fs_dfm", "logit_kl"])
@pytest.mark.parametrize("num_steps", [1, 2, 4, 8])
def test_depth_is_k_plus_3_with_bridge_and_readout_fused_before_the_first_pass(tiny_config, kind: str,
                                                                               num_steps: int) -> None:
    cfg = _config(tiny_config, realizer={"kind": kind})
    model = _model(cfg)
    sampler = _sampler(model, cfg)
    log, handles = _record_calls(model)
    try:
        # Canvas from the readout's length head: the readout runs, fused with the bridge into pass 4.
        result = sampler.sample(_context(0), num_steps=num_steps, generator=_gen(0))
        fused_log = list(log)
        log.clear()
        explicit = sampler.sample(_context(0), length=16, num_steps=num_steps, generator=_gen(0))
    finally:
        for handle in handles:
            handle.remove()
    assert result.depth == explicit.depth == num_steps + 3 and result.num_steps == num_steps
    for events in (result.events, fused_log):
        assert events[:3] == PLANNER
        assert sorted(events[3:-num_steps]) == ["bridge", "readout"]                # once each, fused
        assert events[-num_steps:] == ["F_psi"] * num_steps
    assert explicit.events == log == PLANNER + ["bridge"] + ["F_psi"] * num_steps    # no readout needed


@pytest.mark.parametrize("kind", ["fs_dfm", "logit_kl"])
def test_guidance_batches_conditional_and_null_rows_into_one_realizer_pass(tiny_config, monkeypatch,
                                                                           kind: str) -> None:
    omega = 1.5
    cfg = _config(tiny_config, realizer={"kind": kind, "guidance_scale": omega})
    model = _model(cfg, open_gates=True)
    inputs, outputs, handles = _record_realizer(model)
    posteriors: list[Tensor] = []
    if kind == "fs_dfm":
        original_kernel = sampler_module.tilted_jump_kernel

        def kernel(posterior: Tensor, current: Tensor, pi: float, multiplier=None, c_max: float = 1.0):  # noqa: ANN001
            posteriors.append(posterior.clone())                   # the guided posterior of line 11
            return original_kernel(posterior, current, pi, multiplier, c_max)

        monkeypatch.setattr(sampler_module, "tilted_jump_kernel", kernel)
    else:
        original_step = model.realizer.euler_step

        def step(l_k: Tensor, posterior: Tensor, t: float, h: float) -> Tensor:
            posteriors.append(posterior.clone())                   # the posterior entering Eq. 21
            return original_step(l_k, posterior, t, h)

        monkeypatch.setattr(model.realizer, "euler_step", step)
    try:
        result = _sampler(model, cfg).sample(_context(0), length=N, num_steps=4, generator=_gen(0))
    finally:
        for handle in handles:
            handle.remove()
    assert result.depth == 7 and result.events.count("F_psi") == 4 and result.events.count("bridge") == 1
    assert len(inputs) == len(posteriors) == 4
    for (state, t, h, cache), logits, posterior in zip(inputs, outputs, posteriors, strict=True):
        assert state.shape[0] == 2 * B and cache.mask.shape[0] == 2 * B
        assert torch.equal(state[:B], state[B:]) and torch.equal(t[:B], t[B:]) and torch.equal(h[:B], h[B:])
        assert bool(cache.mask[B:, 0].all()) and not bool(cache.mask[B:, 1:].any())   # null rows: null token only
        guided = (1 + omega) * _softmax64(logits[:B].double()) - omega * _softmax64(logits[B:].double())
        if kind == "logit_kl":                                   # clipped at 0 and renormalised (no energies)
            guided = guided.clamp_min(0.0)
            guided = guided / guided.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(posterior.double(), guided, rtol=RTOL, atol=1e-6)
    assert (outputs[0][:B] - outputs[0][B:]).abs().max() > 1e-3                      # the two halves differ


GOOD = PLANNER + ["readout", "bridge"] + ["F_psi"] * 3


@pytest.mark.parametrize("events", [
    pytest.param(GOOD + ["F_psi"], id="extra-F_psi"),
    pytest.param(GOOD[:-1], id="missing-F_psi"),
    pytest.param(["f_theta", "g_phi", "readout", "bridge"] + ["F_psi"] * 3, id="missing-selector"),
    pytest.param(["pi_phi", "f_theta", "g_phi", "bridge"] + ["F_psi"] * 3, id="planner-out-of-order"),
    pytest.param(["f_theta", "pi_phi", "pi_phi", "g_phi", "bridge"] + ["F_psi"] * 3, id="selector-twice"),
    pytest.param(PLANNER + ["bridge", "bridge"] + ["F_psi"] * 3, id="bridge-twice"),
    pytest.param(PLANNER + ["F_psi", "bridge", "F_psi", "F_psi"], id="bridge-after-first-F_psi"),
    pytest.param(PLANNER + ["F_psi"] * 3, id="bridge-missing"),
    pytest.param(["f_theta", "pi_phi", "bridge", "g_phi"] + ["F_psi"] * 3, id="bridge-before-g_phi"),
    pytest.param(PLANNER + ["readout", "readout", "bridge"] + ["F_psi"] * 3, id="readout-twice"),
    pytest.param(PLANNER + ["bridge", "F_psi", "readout", "F_psi", "F_psi"], id="readout-after-first-F_psi"),
])
def test_depth_audit_rejects_wrong_chains(tiny_config, events: list[str]) -> None:
    audit = DepthAudit(_model(_config(tiny_config)))
    audit.events = list(events)
    with pytest.raises(DepthAuditError):
        audit.verify(3)


def test_depth_audit_accepts_the_algorithm_1_chain_and_removes_its_hooks(tiny_config) -> None:
    model = _model(_config(tiny_config))
    audit = DepthAudit(model)
    for events in (GOOD, PLANNER + ["bridge", "readout"] + ["F_psi"] * 3, PLANNER + ["bridge"] + ["F_psi"] * 3):
        audit.events = list(events)
        audit.verify(3)
        assert audit.depth == 6
    with audit:
        with torch.no_grad():
            model.planner.plan(_context(0), mode=torch.tensor([0, 1]))
    assert audit.events == PLANNER
    with torch.no_grad():
        model.planner.plan(_context(0), mode=torch.tensor([0, 1]))                  # hooks are gone
    assert audit.events == PLANNER


def test_sample_raises_when_the_runtime_chain_is_not_k_plus_3(tiny_config, monkeypatch) -> None:
    cfg = _config(tiny_config)
    model = _model(cfg)
    original = PRJEPASampler._fs_dfm

    def one_pass_too_many(self, cache, n, k_steps, *args):  # noqa: ANN001, ANN002, ANN202
        tokens, posterior, unrealized = original(self, cache, n, k_steps, *args)
        rows = tokens.shape[0]
        self.model.realizer(tokens, torch.zeros(rows), torch.full((rows,), 1.0 / k_steps), cache)
        return tokens, posterior, unrealized

    monkeypatch.setattr(PRJEPASampler, "_fs_dfm", one_pass_too_many)
    with pytest.raises(DepthAuditError):
        _sampler(model, cfg).sample(_context(0), length=N, num_steps=2, generator=_gen(0))
    unchecked = _sampler(model, cfg, audit_depth=False).sample(_context(0), length=N, num_steps=2, generator=_gen(0))
    assert unchecked.depth == 6


# ================================================================================================
# Algorithm 1 semantics (FS-DFM)
# ================================================================================================
@pytest.mark.parametrize("schedule", ["linear", "cosine", "sqrt"])
@pytest.mark.parametrize("num_steps", [1, 2, 4, 8])
def test_each_pass_uses_t_k_over_K_h_one_over_K_and_the_exact_jump_mass(tiny_config, monkeypatch, schedule: str,
                                                                         num_steps: int) -> None:
    cfg = _config(tiny_config, realizer={"schedule": schedule})
    model = _model(cfg, open_gates=True)
    inputs, outputs, handles = _record_realizer(model)
    masses: list[float] = []
    posteriors: list[Tensor] = []
    currents: list[Tensor] = []
    draws: list[Tensor] = []
    original_kernel, original_jump = sampler_module.tilted_jump_kernel, sampler_module.sample_jump

    def kernel(posterior: Tensor, current: Tensor, pi: float, multiplier=None, c_max: float = 1.0):  # noqa: ANN001
        masses.append(float(pi))
        posteriors.append(posterior.clone())
        currents.append(current.clone())
        return original_kernel(posterior, current, pi, multiplier, c_max)

    def jump(k, current: Tensor, generator: torch.Generator | None = None) -> Tensor:  # noqa: ANN001
        out = original_jump(k, current, generator)
        draws.append(out.clone())
        return out

    monkeypatch.setattr(sampler_module, "tilted_jump_kernel", kernel)
    monkeypatch.setattr(sampler_module, "sample_jump", jump)
    try:
        result = _sampler(model, cfg).sample(_context(0), length=N, num_steps=num_steps, generator=_gen(1))
    finally:
        for handle in handles:
            handle.remove()
    kappa = KAPPA[schedule]
    assert len(inputs) == len(masses) == len(draws) == num_steps
    for k, (w, t, h, _cache) in enumerate(inputs):
        t_k = k / num_steps
        torch.testing.assert_close(t, torch.full((B,), t_k), rtol=0.0, atol=1e-7)         # line 8: t = k/K
        torch.testing.assert_close(h, torch.full((B,), 1 / num_steps), rtol=0.0, atol=1e-7)   # h = 1/K
        expected = (kappa(t_k + 1 / num_steps) - kappa(t_k)) / (1.0 - kappa(t_k))          # Eq. 15
        assert masses[k] == pytest.approx(expected, abs=1e-12)
        torch.testing.assert_close(posteriors[k].double(), _softmax64(outputs[k].double()), rtol=RTOL, atol=1e-7)
        assert torch.equal(currents[k], w)                    # the kernel moves the state the network saw
        if k > 0:
            assert torch.equal(w, draws[k - 1])               # ... which is the previous step's draw
    assert torch.equal(result.tokens, draws[-1])              # no energies: the last draw is returned
    if schedule == "linear":
        assert masses == pytest.approx([1 / (num_steps - k) for k in range(num_steps)], abs=1e-12)
        assert masses[-1] == pytest.approx(1.0, abs=1e-12)    # the last step redraws every position


@pytest.mark.parametrize("source", ["uniform", "mask"])
@pytest.mark.parametrize(("schedule", "num_steps"),
                         [("linear", k) for k in range(1, 9)] + [("cosine", k) for k in (1, 2, 4, 8)])
def test_oracle_posterior_reaches_the_target_exactly_after_k_steps(tiny_config, source: str, schedule: str,
                                                                   num_steps: int) -> None:
    """Lemma 1 with the oracle posterior delta_{w_1}: pi_{t_{K-1}, 1/K} = 1, so w_K = w_1 exactly."""
    cfg = _config(tiny_config, realizer={"source": source, "schedule": schedule})
    model = _model(cfg)
    target = torch.randint(0, 256, (B, N), generator=_gen(num_steps))
    _oracle(model, target)
    result = _sampler(model, cfg).sample(_context(0), length=N, num_steps=num_steps, generator=_gen(2))
    assert torch.equal(result.tokens, target)
    assert result.depth == num_steps + 3


def test_oracle_marginals_follow_kappa_after_every_step(tiny_config, monkeypatch) -> None:
    """Lemma 1 by Monte Carlo (mask source, cosine schedule): P(w_{t_{k+1}}^i = w_1^i) = kappa_{t_{k+1}}."""
    batch, length, num_steps = 32, 64, 4
    cfg = _config(tiny_config, realizer={"source": "mask", "schedule": "cosine"})
    model = _model(cfg)
    target = torch.randint(0, 256, (batch, length), generator=_gen(3))
    _oracle(model, target)
    states: list[Tensor] = []
    original_jump = sampler_module.sample_jump

    def jump(k, current: Tensor, generator: torch.Generator | None = None) -> Tensor:  # noqa: ANN001
        out = original_jump(k, current, generator)
        states.append(out.clone())
        return out

    monkeypatch.setattr(sampler_module, "sample_jump", jump)
    _sampler(model, cfg).sample(_context(0, batch), length=length, num_steps=num_steps, generator=_gen(4))
    previous = torch.full_like(target, MASK)
    for k, w in enumerate(states):
        arrived = w == target
        assert bool((arrived | (w == MASK)).all())                 # jumps land only on the target
        assert bool(arrived[previous == target].all())            # an arrived position stays (Lemma 1 proof)
        kappa = KAPPA["cosine"]((k + 1) / num_steps)
        hits = int(arrived.sum())
        if k == num_steps - 1:
            assert hits == batch * length                         # kappa_1 = 1
        else:
            assert _five_se_proportion(hits, batch * length, kappa), (k, hits / (batch * length), kappa)
        previous = w


TOKEN_A, TOKEN_B = ord("x"), ord("n")


@pytest.mark.parametrize(("c_max", "temperature", "penalty"),
                         [(1.0, 1.0, 0.0), (0.5, 1.0, 0.0), (1.0, 0.5, 2.0), (0.6, 2.0, 1.0)])
def test_one_step_kernel_follows_eq17_tilt_clip_and_staying_mass(tiny_config, c_max: float, temperature: float,
                                                                 penalty: float) -> None:
    """K = 1 (pi = 1), mask source, posterior P = 0.75 delta_A + 0.25 delta_B and E(B) = penalty:
    rho(A) = min(0.75, c_max), rho(B) = min(0.25 e^{-penalty/T}, c_max); the position moves to A or B with
    these probabilities and keeps [MASK] with the rejected mass 1 - rho(A) - rho(B) (lines 11-12, Eq. 17)."""
    batch, length = 128, 64
    cfg = _config(tiny_config, realizer={"source": "mask", "c_max": c_max, "temperature": temperature})
    model = _model(cfg)
    _fixed_posterior(model, {TOKEN_A: 0.75, TOKEN_B: 0.25})
    energy = ConstraintEnergyEvaluator(TOK, terms=[_SoftPenalty([TOKEN_B], penalty)]) if penalty else None
    result = _sampler(model, cfg, energy, final_repair=False).sample(_context(0, batch), length=length, num_steps=1,
                                                                     generator=_gen(13))
    rho_a = min(0.75, c_max)
    rho_b = min(0.25 * math.exp(-penalty / temperature), c_max)
    total = batch * length
    for token, p in ((TOKEN_A, rho_a), (TOKEN_B, rho_b), (MASK, 1.0 - rho_a - rho_b)):
        hits = int((result.tokens == token).sum())
        if p == 0.0:
            assert hits == 0
        else:
            assert _five_se_proportion(hits, total, p), (token, hits / total, p)
    assert int(((result.tokens == TOKEN_A) | (result.tokens == TOKEN_B) | (result.tokens == MASK)).sum()) == total


def test_clipped_oracle_arrivals_compound_over_the_k_steps(tiny_config, monkeypatch) -> None:
    """With c_max < 1 the oracle jump mass is pi_k c_max, so P(arrived after step k) = 1 - prod_j (1 - c_max pi_j)
    with pi_j = 1/(K - j) (linear schedule)."""
    batch, length, num_steps, c_max = 64, 64, 4, 0.5
    cfg = _config(tiny_config, realizer={"source": "mask", "c_max": c_max})
    model = _model(cfg)
    target = torch.randint(0, 256, (batch, length), generator=_gen(14))
    _oracle(model, target)
    states: list[Tensor] = []
    original_jump = sampler_module.sample_jump

    def jump(k, current: Tensor, generator: torch.Generator | None = None) -> Tensor:  # noqa: ANN001
        out = original_jump(k, current, generator)
        states.append(out.clone())
        return out

    monkeypatch.setattr(sampler_module, "sample_jump", jump)
    _sampler(model, cfg).sample(_context(0, batch), length=length, num_steps=num_steps, generator=_gen(15))
    stay = 1.0
    for k, w in enumerate(states):
        stay *= 1.0 - c_max / (num_steps - k)
        assert _five_se_proportion(int((w == target).sum()), batch * length, 1.0 - stay), k


# ================================================================================================
# Energies: Prop. 4(d), restricted source, final repair, mask source
# ================================================================================================
@pytest.mark.parametrize("kind", ["fs_dfm", "logit_kl"])
def test_oracle_plan_outputs_are_admissible_and_bracket_valid(tiny_config, kind: str) -> None:
    cfg = _config(tiny_config, realizer={"kind": kind})
    model = _model(cfg, open_gates=True)
    energy = _evaluator(grammar=True, forbid_strings=True)
    plan = _oracle_plan(energy)
    licensed = {b for b in REFERENCE.encode() if _word_bearing(b)}
    assert plan.licensed_tokens == frozenset(licensed) and UNLICENSED not in licensed
    result = _sampler(model, cfg, energy).sample(_context(0), length=48, num_steps=4, plans=[plan] * B,
                                                 generator=_gen(5))
    assert result.hard_violations == 0 and result.unrepaired == 0
    allowed = _plan_admissible(forbid_strings=True)
    for row in result.tokens.tolist():
        assert all(bool(allowed[token]) for token in _output(row))
        assert _bracket_errors(row) == 0
    assert result.depth == 7


@pytest.mark.parametrize(("guidance", "repair_aware"), [(0.0, False), (1.5, False), (0.0, True)])
def test_plan_mediated_admissibility_survives_every_parallel_update(tiny_config, monkeypatch, guidance: float,
                                                                   repair_aware: bool) -> None:
    """Prop. 4(d): with an admissible w_0 and only plan-mediated hard terms, every intermediate state is
    admissible at every position -- without any repair."""
    cfg = _config(tiny_config, realizer={"guidance_scale": guidance})
    model = _model(cfg, open_gates=True)
    energy = _evaluator(grammar=False, forbid_strings=False)
    inputs, _outputs, handles = _record_realizer(model)
    states: list[Tensor] = []
    original_jump = sampler_module.sample_jump

    def jump(k, current: Tensor, generator: torch.Generator | None = None) -> Tensor:  # noqa: ANN001
        out = original_jump(k, current, generator)
        states.append(out.clone())
        return out

    monkeypatch.setattr(sampler_module, "sample_jump", jump)
    try:
        result = _sampler(model, cfg, energy, final_repair=False, repair_aware=repair_aware).sample(
            _context(0), length=48, num_steps=8, plans=[_oracle_plan(energy)] * B, generator=_gen(6))
    finally:
        for handle in handles:
            handle.remove()
    allowed = _plan_admissible(forbid_strings=False)
    w_0 = inputs[0][0][:B]
    for w in [w_0, *states]:
        assert bool(allowed[w].all())
    assert torch.equal(result.tokens, states[-1]) and result.hard_violations == 0
    assert len({tuple(w.flatten().tolist()) for w in states}) > 1                # the draft did move


def test_runtime_prop4d_check_catches_an_inadmissible_parallel_move(tiny_config, monkeypatch) -> None:
    cfg = _config(tiny_config)
    model = _model(cfg)
    energy = _evaluator(grammar=False, forbid_strings=False)
    plans = [_oracle_plan(energy)] * B
    original_jump = sampler_module.sample_jump

    def corrupting(k, current: Tensor, generator: torch.Generator | None = None) -> Tensor:  # noqa: ANN001
        out = original_jump(k, current, generator).clone()
        out[0, 0] = UNLICENSED                                  # admissible -> unlicensed identifier letter
        return out

    monkeypatch.setattr(sampler_module, "sample_jump", corrupting)
    with pytest.raises(AdmissibilityViolation):
        _sampler(model, cfg, energy).sample(_context(0), length=24, num_steps=2, plans=plans, generator=_gen(7))
    # Control: the same move passes when the check is off, and the final repair then removes it.
    result = _sampler(model, cfg, energy, check_admissibility=False).sample(_context(0), length=24, num_steps=2,
                                                                            plans=plans, generator=_gen(7))
    assert int(result.tokens[0, 0]) != UNLICENSED and result.hard_violations == 0


@pytest.mark.parametrize("restrict", [True, False])
def test_restricted_source_draws_w0_from_the_plan_admissible_tokens(tiny_config, restrict: bool) -> None:
    cfg = _config(tiny_config)
    model = _model(cfg)
    energy = _evaluator(grammar=False, forbid_strings=False)
    inputs, _outputs, handles = _record_realizer(model)
    try:
        _sampler(model, cfg, energy, restrict_source=restrict).sample(
            _context(0), length=48, num_steps=2, plans=[_oracle_plan(energy)] * B, generator=_gen(8))
    finally:
        for handle in handles:
            handle.remove()
    allowed = _plan_admissible(forbid_strings=False)
    w_0 = inputs[0][0]
    # ~45 of the 265 tokens are inadmissible: 96 unrestricted draws all admissible has probability ~2e-8.
    assert bool(allowed[w_0].all()) is restrict


def test_final_repair_removes_draft_dependent_violations(tiny_config, monkeypatch) -> None:
    cfg = _config(tiny_config)
    model = _model(cfg, open_gates=True)
    energy = _evaluator(grammar=True, forbid_strings=True)
    plans = [_oracle_plan(energy)] * B
    num_steps = 4
    calls = {"n": 0}
    original_jump = sampler_module.sample_jump

    def last_step_closer(k, current: Tensor, generator: torch.Generator | None = None) -> Tensor:  # noqa: ANN001
        out = original_jump(k, current, generator)
        calls["n"] += 1
        if calls["n"] % num_steps == 0:                         # the last parallel update of a generation
            out = out.clone()
            out[0, 0] = ord(")")                                # plan-admissible, but closes nothing
        return out

    monkeypatch.setattr(sampler_module, "sample_jump", last_step_closer)
    raw = _sampler(model, cfg, energy, final_repair=False).sample(_context(0), length=48, num_steps=num_steps,
                                                                  plans=plans, generator=_gen(9))
    assert int(raw.tokens[0, 0]) == ord(")")
    assert raw.hard_violations >= 1 and _bracket_errors(raw.tokens[0].tolist()) >= 1
    fixed = _sampler(model, cfg, energy, final_repair=True).sample(_context(0), length=48, num_steps=num_steps,
                                                                   plans=plans, generator=_gen(9))
    assert fixed.hard_violations == 0 and fixed.unrepaired == 0 and int(fixed.tokens[0, 0]) != ord(")")
    allowed = _plan_admissible(forbid_strings=True)
    for row in fixed.tokens.tolist():
        assert _bracket_errors(row) == 0
        assert all(bool(allowed[token]) for token in _output(row))
    assert fixed.events == raw.events and fixed.depth == num_steps + 3              # the repair is network-free


def test_mask_source_leaves_no_mask_in_the_output_after_repair(tiny_config) -> None:
    cfg = _config(tiny_config, realizer={"source": "mask"})
    model = _model(cfg)
    energy = _evaluator(grammar=True, forbid_strings=True)
    plans = [_oracle_plan(energy)] * B
    target = torch.tensor([ord(c) for c in "x + 1 "] * 6)[:N].repeat(B, 1)
    stuck = [(0, 0), (0, 13), (1, 5), (1, N - 1)]
    for b, i in stuck:
        target[b, i] = MASK                                     # posterior pinned on [MASK]: never leaves it
    _oracle(model, target)
    raw = _sampler(model, cfg, energy, final_repair=False).sample(_context(0), length=N, num_steps=4,
                                                                  plans=plans, generator=_gen(10))
    assert int((raw.tokens == MASK).sum()) == len(stuck) and raw.hard_violations == len(stuck)
    fixed = _sampler(model, cfg, energy, final_repair=True).sample(_context(0), length=N, num_steps=4,
                                                                   plans=plans, generator=_gen(10))
    for row in fixed.tokens.tolist():
        assert MASK not in _output(row)
    assert not bool((fixed.tokens == MASK).any()) and fixed.hard_violations == 0
    keep = target != MASK
    assert torch.equal(fixed.tokens[keep], target[keep])        # only the leftovers were replaced


def test_stale_draft_energy_stall_is_refilled_by_the_repair(tiny_config, monkeypatch) -> None:
    """Draft-dependent hard terms under a parallel update (Sec. 3.3; Prop. 4(b): rejected jump mass is
    probability of staying). K = 1, so the only update (pi = 1) is tilted with the energies of w_0. With
    w_0 = "[" + "1"s and the oracle posterior on "(x)" + [PAD]s, ')' and [PAD] are forbidden after the stale
    prefix "[1...", so those positions keep their uniform-source token -- admissible noise that no hard check
    flags. The repair refills them with the energies of the repaired prefix "(x": ')' then [PAD]."""
    cfg = _config(tiny_config)
    model = _model(cfg)
    energy = _evaluator(grammar=True, forbid_strings=True)
    plans = [_oracle_plan(energy)] * B
    target = torch.full((B, N), PAD)
    target[:, :3] = torch.tensor([ord("("), ord("x"), ord(")")])
    _oracle(model, target)
    w_0 = torch.full((B, N), ord("1"))
    w_0[:, 0] = ord("[")
    monkeypatch.setattr(model.realizer, "sample_source", lambda *args, **kwargs: w_0.clone())
    raw = _sampler(model, cfg, energy, final_repair=False).sample(_context(0), length=N, num_steps=1, plans=plans,
                                                                  generator=_gen(12))
    expected = w_0.clone()
    expected[:, :2] = target[:, :2]                             # '(' and 'x' were legal after the stale prefix
    assert torch.equal(raw.tokens, expected)
    assert raw.unrealized == B * (N - 2) and raw.hard_violations == 0
    fixed = _sampler(model, cfg, energy).sample(_context(0), length=N, num_steps=1, plans=plans, generator=_gen(12))
    assert fixed.texts == ["(x)"] * B and torch.equal(fixed.tokens[:, :4], target[:, :4])
    assert fixed.hard_violations == 0 and fixed.unrepaired == 0 and fixed.unrealized == raw.unrealized
    assert fixed.events == raw.events and fixed.depth == 1 + 3                       # the repair is network-free


def test_clip_stalls_keep_source_tokens_that_the_repair_refills(tiny_config) -> None:
    """With c_max < 1 the oracle's jump mass is capped (rho~ = c_max), so a position never jumps with
    probability prod_k (1 - c_max pi_k) > 0 -- even the last step (pi = 1) keeps it with probability 1 - c_max.
    Those positions still hold w_0 after Algorithm 1; every other position holds the target. The repair refills
    exactly the stalled ones with the posterior's token."""
    cfg = _config(tiny_config, realizer={"c_max": 0.5})
    model = _model(cfg)
    energy = _evaluator(grammar=False, forbid_strings=False)
    plans = [_oracle_plan(energy)] * B
    target = torch.tensor([ord(c) for c in "x + 1 "] * 6)[:N].repeat(B, 1)
    _oracle(model, target)
    inputs, _outputs, handles = _record_realizer(model)
    try:
        raw = _sampler(model, cfg, energy, final_repair=False).sample(_context(0), length=N, num_steps=4,
                                                                      plans=plans, generator=_gen(13))
    finally:
        for handle in handles:
            handle.remove()
    w_0 = inputs[0][0]
    stalled = raw.tokens == w_0                     # a jump always moves to the target, which differs from w_0
    # E[#stalled] = B N (1 - 1/8)(1 - 1/6)(1 - 1/4)(1 - 1/2) = 64 * 0.273 = 17.5
    assert raw.unrealized == int(stalled.sum()) >= 5
    assert torch.equal(raw.tokens[~stalled], target[~stalled])
    fixed = _sampler(model, cfg, energy).sample(_context(0), length=N, num_steps=4, plans=plans, generator=_gen(13))
    assert torch.equal(fixed.tokens, target) and fixed.unrealized == raw.unrealized


def test_readout_plans_drive_the_energies_unless_plan_source_is_none(tiny_config) -> None:
    cfg = _config(tiny_config)
    model = _model(cfg, open_gates=True)
    energy = _evaluator(grammar=False, forbid_strings=False)
    extractor = TreeSitterASTExtractor("python")
    decoded = PRJEPASampler(model, TOK, cfg.realizer, SamplerConfig(), energy, extractor).sample(
        _context(0), length=N, num_steps=2, generator=_gen(16))
    assert decoded.plans is not None and all(p.source == "readout" for p in decoded.plans)
    assert decoded.events.count("readout") == 1 and decoded.hard_violations == 0
    unlicensed = 0
    for row, plan in zip(decoded.tokens.tolist(), decoded.plans, strict=True):
        assert plan.licensed_tokens is not None
        unlicensed += sum(1 for i in range(256) if _word_bearing(i) and i not in plan.licensed_tokens)
        for token in _output(row):
            assert token < 256                                   # no control token in the output
            assert not _word_bearing(token) or token in plan.licensed_tokens
    assert unlicensed > 0                                        # the decoded plans do constrain the letters
    plain = PRJEPASampler(model, TOK, cfg.realizer, SamplerConfig(), energy, extractor, plan_source="none").sample(
        _context(0), length=N, num_steps=2, generator=_gen(16))
    assert plain.plans is not None and all(p.source == "none" and p.licensed_tokens is None for p in plain.plans)
    assert "readout" not in plain.events
    with pytest.raises(ValueError):
        PRJEPASampler(model, TOK, cfg.realizer, SamplerConfig(), energy, extractor, plan_source="reference")


def test_mode_selection_greedy_forced_and_gumbel(tiny_config) -> None:
    cfg = _config(tiny_config)
    model = _model(cfg, open_gates=True)
    context = _context(0)
    with torch.no_grad():
        logits = model.planner.selector(model.planner.context_encoder(context))
    greedy = _sampler(model, cfg, mode_strategy="greedy").sample(context, length=N, num_steps=2, generator=_gen(0))
    assert torch.equal(greedy.modes, logits.argmax(dim=1))
    forced = torch.tensor([5, 2])
    result = _sampler(model, cfg).sample(context, length=N, num_steps=2, mode=forced, generator=_gen(0))
    assert torch.equal(result.modes, forced)
    assert result.events[:3] == PLANNER and result.depth == 5          # pi_phi^h still runs (K + 3 chain)
    gumbel = _sampler(model, cfg, mode_strategy="gumbel").sample(context, length=N, num_steps=2, generator=_gen(0))
    assert gumbel.depth == 5 and bool(((gumbel.modes >= 0) & (gumbel.modes < model.planner.num_modes)).all())


@pytest.mark.parametrize("kind", ["fs_dfm", "logit_kl"])
def test_generator_seed_alone_determines_the_sample(tiny_config, kind: str) -> None:
    cfg = _config(tiny_config, realizer={"kind": kind})
    model = _model(cfg, open_gates=True)
    sampler = _sampler(model, cfg)
    results = []
    for global_seed, seed in ((1, 11), (2, 11), (3, 12)):
        torch.manual_seed(global_seed)                          # the global RNG must not matter
        results.append(sampler.sample(_context(0), length=N, num_steps=4, generator=_gen(seed)))
    assert torch.equal(results[0].tokens, results[1].tokens) and torch.equal(results[0].modes, results[1].modes)
    assert not torch.equal(results[0].tokens, results[2].tokens)
    for row, text in zip(results[0].tokens.tolist(), results[0].texts, strict=True):   # text ends at the first [PAD]
        assert text == bytes(t for t in _output(row) if t < 256).decode("utf-8", "replace")


# ================================================================================================
# Canvas length and argument checks
# ================================================================================================
def _force_length_logits(model: TextVPAModel, logits: Tensor) -> object:
    return model.readout.register_forward_hook(lambda _m, _a, out: out._replace(length_logits=logits))


def test_canvas_length_resolution_order(tiny_config) -> None:
    cfg = _config(tiny_config)
    model = _model(cfg)
    bucket = model.readout.length_bucket_size
    sampler = _sampler(model, cfg)
    context = _context(0)

    def length_of(s: PRJEPASampler, **kwargs: object) -> int:
        return int(s.sample(context, num_steps=2, generator=_gen(0), **kwargs).tokens.shape[1])

    assert length_of(sampler, length=20) == 20                                       # explicit
    assert length_of(_sampler(model, cfg, default_length=24)) == 24                 # SamplerConfig.default_length
    assert length_of(_sampler(model, cfg, default_length=24), length=20) == 20
    for logits, expected in (([[3.0, 0.0], [3.0, 0.0]], bucket), ([[3.0, 0.0], [0.0, 3.0]], 2 * bucket)):
        handle = _force_length_logits(model, torch.tensor(logits))
        try:
            assert length_of(sampler) == expected                                     # (argmax + 1) B_s, max over B
        finally:
            handle.remove()
    cfg_plain = _config(tiny_config, readout={"enabled": False})
    assert length_of(_sampler(_model(cfg_plain), cfg_plain)) == cfg.realizer.max_output_len


def test_invalid_lengths_steps_vocabularies_and_plans_raise(tiny_config) -> None:
    cfg = _config(tiny_config)
    model = _model(cfg)
    sampler = _sampler(model, cfg)
    for length in (0, cfg.realizer.max_output_len + 1):
        with pytest.raises(ValueError):
            sampler.sample(_context(0), length=length, num_steps=2)
    for num_steps in (0, 9):
        with pytest.raises(ValueError):
            sampler.sample(_context(0), length=N, num_steps=num_steps)
    wide = _config(tiny_config).model_dump()
    wide["planner"]["vocab_size"] = wide["realizer"]["vocab_size"] = S + 1
    wide_cfg = TextVPAConfig.model_validate(wide)
    with pytest.raises(ValueError):
        PRJEPASampler(TextVPAModel.from_config(wide_cfg, mask_id=MASK), TOK, wide_cfg.realizer)
    energy = _evaluator(grammar=False, forbid_strings=False)
    with pytest.raises(ValueError):
        _sampler(model, cfg, energy).sample(_context(0), length=N, num_steps=2, plans=[_oracle_plan(energy)])


def test_readout_length_past_the_canvas_is_clamped_to_the_realizer_maximum(tiny_config) -> None:
    cfg = _config(tiny_config, readout={"length_bucket_size": 24})
    model = _model(cfg)
    assert model.readout.num_length_buckets == 3
    handle = _force_length_logits(model, torch.tensor([[0.0, 0.0, 3.0]] * B))
    try:
        result = _sampler(model, cfg).sample(_context(0), num_steps=2, generator=_gen(0))
    finally:
        handle.remove()
    assert result.tokens.shape == (B, cfg.realizer.max_output_len)


# ================================================================================================
# Logit-KL sampler (Sec. 3.4)
# ================================================================================================
def test_logit_kl_sampler_integrates_eq21_and_decodes_the_scaled_posterior(tiny_config) -> None:
    s_std, num_steps = 1.5, 4
    cfg = _config(tiny_config, realizer={"kind": "logit_kl", "logit_source_std": s_std})
    model = _model(cfg, open_gates=True)
    eps = cfg.realizer.logit_smoothing
    a = math.log((1 - eps + eps / S) / (eps / S))                                   # Eq. 18
    inputs, outputs, handles = _record_realizer(model)
    try:
        result = _sampler(model, cfg).sample(_context(0), length=N, num_steps=num_steps, generator=_gen(11))
    finally:
        for handle in handles:
            handle.remove()
    l_0 = inputs[0][0].double()
    n = l_0.numel()
    assert abs(float(l_0.mean())) <= 5 * s_std / math.sqrt(n)                       # l_0 ~ N(0, s^2 I)
    assert abs(float(l_0.var()) - s_std ** 2) <= 5 * s_std ** 2 * math.sqrt(2 / (n - 1))
    for k in range(num_steps):
        torch.testing.assert_close(inputs[k][1], torch.full((B,), k / num_steps), rtol=0.0, atol=1e-7)
        torch.testing.assert_close(inputs[k][2], torch.full((B,), 1 / num_steps), rtol=0.0, atol=1e-7)
    for k in range(num_steps - 1):
        l_k, t_k = inputs[k][0].double(), k / num_steps
        p = _softmax64(outputs[k].double())
        expected = l_k + (1 / num_steps) * (a * p - l_k) / (1 - t_k)               # Eq. 21
        # float32 state of magnitude <= a ~ 8.5 (ulp ~ 1e-6) and float32 softmax
        torch.testing.assert_close(inputs[k + 1][0].double(), expected, rtol=RTOL, atol=1e-5)
    final = a * _softmax64(outputs[-1].double())                                      # l_K = a p
    assert torch.equal(result.tokens, final.argmax(dim=-1))
    assert result.depth == num_steps + 3


def test_logit_kl_energies_tilt_the_posterior_entering_eq21(tiny_config, monkeypatch) -> None:
    """p~ ∝ p e^{-E/T} (Sec. 3.4) with a soft plan-mediated energy on a few tokens."""
    temperature, penalty, penalised = 0.7, 1.3, [ord(c) for c in "abc ("]
    cfg = _config(tiny_config, realizer={"kind": "logit_kl", "temperature": temperature})
    model = _model(cfg, open_gates=True)
    energy = ConstraintEnergyEvaluator(TOK, terms=[_SoftPenalty(penalised, penalty)])
    inputs, outputs, handles = _record_realizer(model)
    posteriors: list[Tensor] = []
    original_step = model.realizer.euler_step

    def step(l_k: Tensor, posterior: Tensor, t: float, h: float) -> Tensor:
        posteriors.append(posterior.clone())
        return original_step(l_k, posterior, t, h)

    monkeypatch.setattr(model.realizer, "euler_step", step)
    try:
        _sampler(model, cfg, energy, final_repair=False).sample(_context(0), length=N, num_steps=4, generator=_gen(17))
    finally:
        for handle in handles:
            handle.remove()
    weight = torch.ones(S, dtype=torch.float64)
    weight[penalised] = math.exp(-penalty / temperature)
    for logits, posterior in zip(outputs, posteriors, strict=True):
        tilted = _softmax64(logits.double()) * weight
        torch.testing.assert_close(posterior.double(), tilted / tilted.sum(dim=-1, keepdim=True), rtol=RTOL,
                                   atol=1e-7)


def test_logit_kl_tilt_keeps_plan_mediated_constraints_without_repair(tiny_config) -> None:
    """p~ ∝ p e^{-E/T} is zero on inadmissible tokens, so argmax(a p~) is admissible at every position."""
    cfg = _config(tiny_config, realizer={"kind": "logit_kl"})
    model = _model(cfg, open_gates=True)
    energy = _evaluator(grammar=False, forbid_strings=True)
    result = _sampler(model, cfg, energy, final_repair=False).sample(
        _context(0), length=48, num_steps=4, plans=[_oracle_plan(energy)] * B, generator=_gen(12))
    assert bool(_plan_admissible(forbid_strings=True)[result.tokens].all())
    assert result.hard_violations == 0
