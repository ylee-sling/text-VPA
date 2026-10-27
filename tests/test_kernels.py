# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Independent tests of the energy-tilted few-step jump kernel (``text_vpa.models.realizers.kernels``).

Ground truth (PR-JEPA, Sec. 3.3, Algorithm 1 and Sec. 4.3):

    pi_{t,h} = (kappa_{t+h} - kappa_t) / (1 - kappa_t)                                        (Eq. 15)
    K(j)     = (1 - pi) delta_a(j) + pi P(j)                                                 (Eq. 15)
    rho~(j)  = min(P(j) exp(-E(j)/T), c_max),   j != a                                       (Eq. 17)
    K~(j)    = pi rho~(j) / max(1, sum_{j' != a} rho~(j')),  j != a;   K~(a) = 1 - sum     (Prop. 4)
    Algorithm 1, lines 11-12: with probability pi sum_j rho~(j) draw j ~ rho~ / sum rho~ and jump.

Every reference below is an explicit float64 loop over these formulas, with the two conventions the paper
implies: the lower clip at 0 of guided posteriors (1 + w) P_c - w P_u (Eq. 16) and 0 * inf = 0 (a token the
posterior gives no mass is never proposed, whatever its repair-aware multiplier).
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from tests.conftest import ATOL, RTOL
from text_vpa.models.realizers.kernels import (
    kernel_distribution,
    repair_aware_multiplier,
    sample_jump,
    tilted_jump_kernel,
)

INF = math.inf


# =============================================================================================
# Float64 references written from Eqs. 15-17 and Prop. 4
# =============================================================================================
def ref_row(post: list[float], a: int, pi: float, mult: list[float] | None = None,
            c_max: float = 1.0) -> tuple[list[float], float, float, list[float]]:
    """(rho~, sum rho~, jump probability, K~) of one position, as Python floats."""
    s = len(post)
    rho = [0.0] * s
    for j in range(s):
        if j == a:
            continue                                    # rho~ is defined for j != a only
        p = post[j] if post[j] > 0.0 else 0.0           # lower clip at 0 (Eq. 16)
        m = 1.0 if mult is None else mult[j]
        tilted = 0.0 if p == 0.0 else p * m             # 0 * inf = 0
        rho[j] = tilted if tilted < c_max else c_max    # upper clip (Eq. 17)
    mass = math.fsum(rho)
    norm = mass if mass > 1.0 else 1.0                  # max(1, sum rho~) (Prop. 4)
    k = [pi * r / norm for r in rho]
    jump = math.fsum(k)
    k[a] = 1.0 - jump                                   # rejected jump mass becomes staying probability
    return rho, mass, jump, k


def ref_batch(post: torch.Tensor, cur: torch.Tensor, pi: float | torch.Tensor, mult: torch.Tensor | None = None,
              c_max: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``ref_row`` over every leading index; returns float64 (rho [..., S], mass [...], jump [...], K [..., S])."""
    s = post.shape[-1]
    lead = tuple(post.shape[:-1])
    p2 = post.double().reshape(-1, s).tolist()
    a2 = cur.reshape(-1).tolist()
    pi2 = torch.as_tensor(pi, dtype=torch.float64).expand(lead).reshape(-1).tolist()
    m2 = None if mult is None else mult.double().reshape(-1, s).tolist()
    rho, mass, jump, dist = [], [], [], []
    for r in range(len(p2)):
        rr, mm, jj, kk = ref_row(p2[r], a2[r], pi2[r], None if m2 is None else m2[r], c_max)
        rho.append(rr)
        mass.append(mm)
        jump.append(jj)
        dist.append(kk)
    f64 = torch.float64
    return (torch.tensor(rho, dtype=f64).reshape(*lead, s), torch.tensor(mass, dtype=f64).reshape(lead),
            torch.tensor(jump, dtype=f64).reshape(lead), torch.tensor(dist, dtype=f64).reshape(*lead, s))


def ref_repair_multiplier(energy: list[float], a: int, temperature: float) -> list[float]:
    """exp(-[E(j) - E(a)]/T) (Sec. 3.3) with the limits: E(j) = inf -> 0; E(j) finite, E(a) = inf -> +inf."""
    out = []
    for e_j in energy:
        if math.isinf(e_j):
            out.append(0.0)
        elif math.isinf(energy[a]):
            out.append(INF)
        else:
            out.append(math.exp(-(e_j - energy[a]) / temperature))
    return out


# =============================================================================================
# Random cases
# =============================================================================================
def random_posteriors(gen: torch.Generator, lead: tuple[int, ...], s: int, zero_frac: float = 0.15,
                      dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Probability vectors with some exact zeros."""
    p = torch.softmax(2.0 * torch.randn(*lead, s, generator=gen, dtype=torch.float64), dim=-1)
    zero = torch.rand(*lead, s, generator=gen, dtype=torch.float64) < zero_frac
    zero[..., 0] = False
    p = p.masked_fill(zero, 0.0)
    return (p / p.sum(-1, keepdim=True)).to(dtype)


def random_energies(gen: torch.Generator, lead: tuple[int, ...], s: int, inf_frac: float = 0.25,
                    zero_frac: float = 0.25) -> torch.Tensor:
    """E >= 0 in float64: admissible (0), soft (finite) and hard (inf) entries."""
    e = 4.0 * torch.rand(*lead, s, generator=gen, dtype=torch.float64)
    e = e.masked_fill(torch.rand(*lead, s, generator=gen, dtype=torch.float64) < zero_frac, 0.0)
    return e.masked_fill(torch.rand(*lead, s, generator=gen, dtype=torch.float64) < inf_frac, INF)


def guided_posteriors(gen: torch.Generator, lead: tuple[int, ...], s: int, w: float) -> torch.Tensor:
    """(1 + w) P_c - w P_u: sums to one, has negative entries and entries above one."""
    p_c = random_posteriors(gen, lead, s, zero_frac=0.0)
    p_u = random_posteriors(gen, lead, s, zero_frac=0.0)
    return (1.0 + w) * p_c - w * p_u


def assert_kernel_matches(kern, dist: torch.Tensor, ref: tuple, rtol: float = RTOL, atol: float = ATOL) -> None:
    rho, mass, jump, k = ref
    torch.testing.assert_close(kern.rho.double(), rho, rtol=rtol, atol=atol)
    torch.testing.assert_close(kern.mass.double(), mass, rtol=rtol, atol=atol)
    torch.testing.assert_close(kern.jump_prob.double(), jump, rtol=rtol, atol=atol)
    torch.testing.assert_close(kern.stay_prob.double(), 1.0 - jump, rtol=rtol, atol=atol)
    torch.testing.assert_close(dist.double(), k, rtol=rtol, atol=atol)


# =============================================================================================
# Eq. 17 and the safety normalisation, value by value
# =============================================================================================
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_eq17_tilted_kernel_matches_hand_loop(dtype: torch.dtype) -> None:
    """rho~(j) = min(P(j) e^{-E(j)/T}, c_max), rho~(a) = 0, jump = pi sum rho~ / max(1, sum rho~), K~ over V,
    for random posteriors, energies with inf entries, T, c_max and per-position pi (incl. 0 and 1)."""
    gen = torch.Generator().manual_seed(11)
    b, n, s = 3, 7, 12
    for temperature, c_max in itertools.product((0.05, 1.0, 20.0), (0.02, 0.3, 1.0)):
        post = random_posteriors(gen, (b, n), s, dtype=dtype)
        mult = torch.exp(-random_energies(gen, (b, n), s) / temperature).to(dtype)
        cur = torch.randint(0, s, (b, n), generator=gen)
        pi = torch.rand(b, n, generator=gen, dtype=torch.float64).to(dtype)
        pi[0, 0], pi[0, 1] = 0.0, 1.0
        kern = tilted_jump_kernel(post, cur, pi, mult, c_max)
        assert_kernel_matches(kern, kernel_distribution(kern, cur), ref_batch(post, cur, pi, mult, c_max))
        assert kern.rho.gather(-1, cur.unsqueeze(-1)).eq(0).all()             # rho~(a) = 0


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_safety_normalisation_values_with_repair_aware_and_guided_inputs(dtype: torch.dtype) -> None:
    """Prop. 4's K~(j) = pi rho~(j) / max(1, sum rho~) when sum rho~ > 1: repair-aware multipliers (> 1, +inf)
    and guided posteriors (negative entries, entries above one)."""
    gen = torch.Generator().manual_seed(12)
    n, s, temperature = 40, 9, 0.7
    active = 0
    for c_max in (0.15, 0.6, 1.0):
        energy = random_energies(gen, (n,), s, inf_frac=0.3)
        cur = torch.randint(0, s, (n,), generator=gen)
        energy[: n // 2].scatter_(-1, cur[: n // 2, None], INF)                # half the current tokens inadmissible
        mult = torch.tensor([ref_repair_multiplier(energy[r].tolist(), int(cur[r]), temperature) for r in range(n)],
                            dtype=torch.float64)
        for post64 in (random_posteriors(gen, (n,), s), guided_posteriors(gen, (n,), s, w=2.0)):
            post, m = post64.to(dtype), mult.to(dtype)
            kern = tilted_jump_kernel(post, cur, 0.8, m, c_max)
            ref = ref_batch(post, cur, 0.8, m, c_max)
            assert_kernel_matches(kern, kernel_distribution(kern, cur), ref)
            active += int((ref[1] > 1.0).sum())
        post = guided_posteriors(gen, (n,), s, w=3.0).to(dtype)                 # guidance without any tilt
        kern = tilted_jump_kernel(post, cur, 0.5, None, c_max)
        ref = ref_batch(post, cur, 0.5, None, c_max)
        assert_kernel_matches(kern, kernel_distribution(kern, cur), ref)
        active += int((ref[1] > 1.0).sum())
    assert active > 20, "the cases must exercise the normalisation max(1, sum rho~) > 1"


def test_scalar_pi_and_no_multiplier_are_accepted() -> None:
    """``pi`` may be a Python float and ``multiplier`` None (no tilt): plain Eq. 17 with E = 0."""
    gen = torch.Generator().manual_seed(13)
    post = random_posteriors(gen, (5, 4), 10)
    cur = torch.randint(0, 10, (5, 4), generator=gen)
    kern = tilted_jump_kernel(post, cur, 0.375, None, 0.2)
    assert_kernel_matches(kern, kernel_distribution(kern, cur), ref_batch(post, cur, 0.375, None, 0.2))


@pytest.mark.parametrize("c_max", [0.0, -0.5, 1.0 + 1e-9, 2.0, math.nan])
def test_c_max_outside_unit_interval_is_rejected(c_max: float) -> None:
    """Prop. 4 holds for c_max in (0, 1]; the docstring requires that range."""
    post = torch.full((2, 4), 0.25, dtype=torch.float64)
    with pytest.raises(ValueError):
        tilted_jump_kernel(post, torch.zeros(2, dtype=torch.long), 0.5, None, c_max)


# =============================================================================================
# Proposition 4 (a)-(c)
# =============================================================================================
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_prop4a_rows_are_probability_vectors_and_normalisation_inactive(dtype: torch.dtype) -> None:
    """(a) For every T > 0, E >= 0 (incl. inf) and c_max in (0, 1]: K~ is a probability distribution on V and
    sum rho~ <= sum P <= 1, so max(1, sum rho~) = 1 (jump probability = pi sum rho~)."""
    gen = torch.Generator().manual_seed(21)
    rows, s = 48, 16
    tol = 1e-12 if dtype == torch.float64 else 1e-6
    for temperature, c_max in itertools.product((1e-3, 0.1, 1.0, 10.0, 1e3), (1e-6, 1e-3, 0.1, 0.5, 1.0)):
        post = random_posteriors(gen, (rows,), s, dtype=dtype)
        mult = torch.exp(-random_energies(gen, (rows,), s) / temperature).to(dtype)
        cur = torch.randint(0, s, (rows,), generator=gen)
        pi = torch.rand(rows, generator=gen, dtype=torch.float64).to(dtype)
        kern = tilted_jump_kernel(post, cur, pi, mult, c_max)
        dist = kernel_distribution(kern, cur).double()
        assert (dist >= 0).all()
        assert (dist.sum(-1) - 1.0).abs().max() <= 1e-6
        assert (kern.mass.double() <= 1.0 + tol).all()
        torch.testing.assert_close(kern.jump_prob.double(), pi.double() * kern.mass.double(), rtol=RTOL, atol=ATOL)


def test_prop4b_staying_probability_is_at_least_one_minus_pi() -> None:
    """(b) K~(a) >= 1 - pi, for the plain tilt and (by the normalisation) for repair-aware / guided inputs."""
    gen = torch.Generator().manual_seed(22)
    rows, s, temperature = 256, 10, 0.5
    cur = torch.randint(0, s, (rows,), generator=gen)
    energy = random_energies(gen, (rows,), s, inf_frac=0.3)
    energy[::3].scatter_(-1, cur[::3, None], INF)
    plain = torch.exp(-energy / temperature)
    repair = torch.tensor([ref_repair_multiplier(energy[r].tolist(), int(cur[r]), temperature) for r in range(rows)],
                          dtype=torch.float64)
    pi = torch.rand(rows, generator=gen, dtype=torch.float64)
    pi[:8] = 1.0
    cases = [
        (random_posteriors(gen, (rows,), s), plain),
        (random_posteriors(gen, (rows,), s), repair),
        (guided_posteriors(gen, (rows,), s, w=1.5), plain),
        (guided_posteriors(gen, (rows,), s, w=4.0), repair),
    ]
    for post, mult in cases:
        for c_max in (0.05, 0.5, 1.0):
            kern = tilted_jump_kernel(post, cur, pi, mult, c_max)
            stay = kernel_distribution(kern, cur).gather(-1, cur[:, None]).squeeze(-1)
            assert (kern.stay_prob >= 1.0 - pi - 1e-12).all()
            assert (stay >= 1.0 - pi - 1e-12).all()


def test_prop4c_repair_aware_multiplier_matches_hand_loop() -> None:
    """exp(-[E(j) - E(a)]/T): above one when E(a) > E(j), +inf for finite E(j) when E(a) = inf, 0 for E(j) = inf."""
    gen = torch.Generator().manual_seed(23)
    rows, s = 30, 8
    energy = random_energies(gen, (rows,), s, inf_frac=0.3)
    cur = torch.randint(0, s, (rows,), generator=gen)
    energy[:10].scatter_(-1, cur[:10, None], INF)
    energy[10:20].scatter_(-1, cur[10:20, None], 3.0)                            # soft current token: multipliers > 1
    for temperature in (0.25, 1.0, 5.0):
        got = repair_aware_multiplier(energy, cur, temperature)
        want = torch.tensor([ref_repair_multiplier(energy[r].tolist(), int(cur[r]), temperature)
                             for r in range(rows)], dtype=torch.float64)
        assert torch.equal(torch.isinf(got), torch.isinf(want))
        assert torch.equal(got == 0, want == 0)
        finite = torch.isfinite(want)
        torch.testing.assert_close(got[finite], want[finite], rtol=RTOL, atol=ATOL)
        assert (got > 1.0).any() and torch.isinf(got).any()


def test_prop4c_rows_stay_probability_vectors_with_repair_aware_and_guided_inputs() -> None:
    """(c) Multipliers above one (or +inf) and guided posteriors: K~ is still a distribution, thanks to the
    safety normalisation; where it is active the jump probability is exactly pi."""
    gen = torch.Generator().manual_seed(24)
    rows, s = 200, 12
    for temperature, c_max, w in itertools.product((0.1, 1.0), (0.05, 0.4, 1.0), (0.5, 3.0)):
        energy = random_energies(gen, (rows,), s, inf_frac=0.3)
        cur = torch.randint(0, s, (rows,), generator=gen)
        energy[::2].scatter_(-1, cur[::2, None], INF)
        mult = repair_aware_multiplier(energy, cur, temperature)
        for post in (random_posteriors(gen, (rows,), s), guided_posteriors(gen, (rows,), s, w)):
            pi = torch.rand(rows, generator=gen, dtype=torch.float64)
            kern = tilted_jump_kernel(post, cur, pi, mult, c_max)
            dist = kernel_distribution(kern, cur)
            assert torch.isfinite(dist).all() and (dist >= 0).all()
            assert (dist.sum(-1) - 1.0).abs().max() <= 1e-12
            over = kern.mass > 1.0
            torch.testing.assert_close(kern.jump_prob[over], pi[over], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("c_max", [1.0, 0.8, 0.4])
def test_prop4c_hand_example_with_inadmissible_current_token(c_max: float) -> None:
    """P = (0.5, 0.3, 0.2), a = 0 with E(a) = inf and E(1) = E(2) = 0: the repair-aware multiplier is +inf on
    tokens 1 and 2, so rho~ = (0, c, c) and K~(1) = K~(2) = pi c / max(1, 2c)."""
    pi = 0.9
    post = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    cur = torch.tensor([0])
    mult = repair_aware_multiplier(torch.tensor([[INF, 0.0, 0.0]], dtype=torch.float64), cur, 1.0)
    assert mult[0, 1] == INF and mult[0, 2] == INF
    kern = tilted_jump_kernel(post, cur, pi, mult, c_max)
    each = pi * c_max / max(1.0, 2.0 * c_max)
    expected = torch.tensor([[1.0 - 2.0 * each, each, each]], dtype=torch.float64)
    torch.testing.assert_close(kernel_distribution(kern, cur), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(kern.rho, torch.tensor([[0.0, c_max, c_max]], dtype=torch.float64))


# =============================================================================================
# Eq. 15 and Lemma 1
# =============================================================================================
def test_untilted_kernel_with_unit_clip_is_eq15() -> None:
    """No tilt and c_max = 1: K = (1 - pi) delta_a + pi P, i.e. pi P(j) for j != a and 1 - pi + pi P(a)."""
    gen = torch.Generator().manual_seed(31)
    rows, s = 64, 9
    post = random_posteriors(gen, (rows,), s)
    cur = torch.randint(0, s, (rows,), generator=gen)
    pi = torch.rand(rows, generator=gen, dtype=torch.float64)
    expected = torch.zeros(rows, s, dtype=torch.float64)
    for r in range(rows):
        for j in range(s):
            expected[r, j] = (1.0 - pi[r]) * (1.0 if j == int(cur[r]) else 0.0) + pi[r] * post[r, j]
    kern = tilted_jump_kernel(post, cur, pi)
    torch.testing.assert_close(kernel_distribution(kern, cur), expected, rtol=1e-12, atol=1e-12)
    stay = torch.tensor([1.0 - pi[r] + pi[r] * post[r, int(cur[r])] for r in range(rows)], dtype=torch.float64)
    torch.testing.assert_close(kern.stay_prob, stay, rtol=1e-12, atol=1e-12)


SCHEDULES = {
    "linear": lambda t: t,
    "quadratic": lambda t: t * t,
    "cosine": lambda t: 1.0 - math.cos(math.pi * t / 2.0),
}


@pytest.mark.parametrize("schedule", sorted(SCHEDULES))
@pytest.mark.parametrize("num_steps", [1, 2, 4, 8])
def test_lemma1_oracle_posterior_maps_the_mixture_path_exactly(schedule: str, num_steps: int) -> None:
    """With the oracle posterior delta_{w1}: a position at w0 != w1 arrives with probability pi, a position at
    w1 stays, so p_t = (1 - kappa_t) delta_{w0} + kappa_t delta_{w1} is mapped to p_{t+h} for every step."""
    kappa = SCHEDULES[schedule]
    s, w0, w1 = 6, 1, 4
    oracle = torch.zeros(1, s, dtype=torch.float64)
    oracle[0, w1] = 1.0
    for k in range(num_steps):
        t, h = k / num_steps, 1.0 / num_steps
        pi = (kappa(t + h) - kappa(t)) / (1.0 - kappa(t))
        from_w0 = kernel_distribution(tilted_jump_kernel(oracle, torch.tensor([w0]), pi), torch.tensor([w0]))[0]
        from_w1 = kernel_distribution(tilted_jump_kernel(oracle, torch.tensor([w1]), pi), torch.tensor([w1]))[0]
        assert from_w0[w1].item() == pytest.approx(pi, abs=1e-12)                 # arrival probability pi
        assert from_w0[w0].item() == pytest.approx(1.0 - pi, abs=1e-12)
        assert from_w1[w1].item() == pytest.approx(1.0, abs=1e-12)                 # already at w1: stays
        marginal = (1.0 - kappa(t)) * from_w0 + kappa(t) * from_w1
        expected = torch.zeros(s, dtype=torch.float64)
        expected[w0], expected[w1] = 1.0 - kappa(t + h), kappa(t + h)
        torch.testing.assert_close(marginal, expected, rtol=0.0, atol=1e-12)


# =============================================================================================
# Sampling (Algorithm 1, lines 11-12)
# =============================================================================================
def sampling_cases() -> list[tuple[torch.Tensor, int, float, torch.Tensor | None, float]]:
    """(P, a, pi, multiplier, c_max) rows with distinct regimes, S = 7."""
    p = torch.tensor([0.05, 0.30, 0.0, 0.25, 0.10, 0.20, 0.10], dtype=torch.float64)
    e = torch.tensor([0.0, 0.0, 0.0, INF, 1.0, 0.0, INF], dtype=torch.float64)
    guided = 2.0 * p - torch.tensor([0.30, 0.05, 0.10, 0.05, 0.30, 0.05, 0.15], dtype=torch.float64)
    cur = 4
    soft_cur = e.clone()
    soft_cur[cur] = INF                                                          # inadmissible current token
    return [
        (p, 0, 0.6, None, 1.0),                                                  # Eq. 15
        (p, 1, 0.9, torch.exp(-e / 0.5), 1.0),                                   # hard + soft tilt
        (p, cur, 0.8, repair_aware_multiplier(soft_cur[None], torch.tensor([cur]), 1.0)[0], 0.7),  # sum rho > 1
        (guided, 5, 1.0, None, 1.0),                                             # guided, sum rho > 1, pi = 1
        (p, 3, 1.0, None, 0.05),                                                 # heavy clipping
    ]


def test_sample_jump_frequencies_match_the_kernel() -> None:
    """Empirical transition frequencies of ``sample_jump`` equal K~ within 5 standard errors (200k draws per
    case); tokens with K~ = 0 (rho~ = 0) are never reached."""
    draws = 200_000
    gen = torch.Generator().manual_seed(41)
    for post, a, pi, mult, c_max in sampling_cases():
        s = post.shape[0]
        _, _, _, k = ref_row(post.tolist(), a, pi, None if mult is None else mult.tolist(), c_max)
        cur = torch.full((draws // 100, 100), a, dtype=torch.long)               # batched [B, N] layout
        kern = tilted_jump_kernel(post.expand(*cur.shape, s), cur, pi,
                                  None if mult is None else mult.expand(*cur.shape, s), c_max)
        out = sample_jump(kern, cur, generator=gen)
        assert out.shape == cur.shape and out.dtype == torch.long
        freq = torch.bincount(out.reshape(-1), minlength=s).double() / draws
        for j in range(s):
            se = math.sqrt(k[j] * (1.0 - k[j]) / draws)
            if k[j] == 0.0:
                assert freq[j] == 0.0, f"token {j} has K~ = 0 but was drawn"
            else:
                assert abs(freq[j].item() - k[j]) <= 5.0 * se + 1e-12, (j, freq[j].item(), k[j], se)


def test_sample_jump_never_proposes_tokens_with_zero_rho() -> None:
    """Random posteriors and energies with inf entries: every move lands on a token with rho~ > 0."""
    gen = torch.Generator().manual_seed(42)
    rows, s = 4000, 11
    post = random_posteriors(gen, (rows,), s, zero_frac=0.3)
    energy = random_energies(gen, (rows,), s, inf_frac=0.4)
    cur = torch.randint(0, s, (rows,), generator=gen)
    rho, _, _, _ = ref_batch(post, cur, 1.0, torch.exp(-energy), 1.0)
    for _ in range(5):
        kern = tilted_jump_kernel(post, cur, 1.0, torch.exp(-energy), 1.0)
        out = sample_jump(kern, cur, generator=gen)
        moved = (out != cur).nonzero().squeeze(-1)
        assert moved.numel() > 0
        assert (rho[moved, out[moved]] > 0).all()


def test_sample_jump_does_not_move_without_jump_mass() -> None:
    """pi = 0, an all-inadmissible alternative set (sum rho~ = 0) or P = delta_a: every position stays."""
    gen = torch.Generator().manual_seed(43)
    rows, s = 500, 8
    post = random_posteriors(gen, (rows,), s)
    cur = torch.randint(0, s, (rows,), generator=gen)
    kern = tilted_jump_kernel(post, cur, 0.0)
    assert torch.equal(sample_jump(kern, cur, generator=gen), cur)
    blocked = torch.full((rows, s), INF, dtype=torch.float64)
    blocked.scatter_(-1, cur[:, None], 0.0)
    kern = tilted_jump_kernel(post, cur, 1.0, torch.exp(-blocked), 1.0)
    assert (kern.jump_prob == 0).all()
    assert torch.equal(sample_jump(kern, cur, generator=gen), cur)
    delta = torch.zeros(rows, s, dtype=torch.float64).scatter_(-1, cur[:, None], 1.0)
    assert torch.equal(sample_jump(tilted_jump_kernel(delta, cur, 1.0), cur, generator=gen), cur)


def test_sample_jump_with_unit_pi_and_oracle_posterior_always_arrives() -> None:
    """pi = 1 (last step of the linear schedule), c_max = 1 and P = delta_{w1}: every position lands on w1."""
    gen = torch.Generator().manual_seed(44)
    rows, s = 300, 6
    target = torch.randint(0, s, (rows,), generator=gen)
    cur = torch.randint(0, s, (rows,), generator=gen)
    oracle = torch.zeros(rows, s, dtype=torch.float64).scatter_(-1, target[:, None], 1.0)
    assert torch.equal(sample_jump(tilted_jump_kernel(oracle, cur, 1.0), cur, generator=gen), target)


def test_sample_jump_is_reproducible_with_a_generator() -> None:
    gen = torch.Generator().manual_seed(45)
    post = random_posteriors(gen, (64, 32), 10)
    cur = torch.randint(0, 10, (64, 32), generator=gen)
    kern = tilted_jump_kernel(post, cur, 0.7)
    first = sample_jump(kern, cur, generator=torch.Generator().manual_seed(7))
    second = sample_jump(kern, cur, generator=torch.Generator().manual_seed(7))
    other = sample_jump(kern, cur, generator=torch.Generator().manual_seed(8))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


# =============================================================================================
# Proposition 4(d): plan-mediated vs draft-dependent hard constraints under the parallel update
# =============================================================================================
def two_bit_kernels(state: tuple[int, int], post: torch.Tensor, pi: float, energy_fn) -> torch.Tensor:
    """Per-position K~ [2, 2] at ``state``, position i tilted by E^i(j | state) = energy_fn(i, j, state)."""
    cur = torch.tensor(state)
    energy = torch.tensor([[energy_fn(i, j, state) for j in (0, 1)] for i in (0, 1)], dtype=torch.float64)
    return kernel_distribution(tilted_jump_kernel(post, cur, pi, torch.exp(-energy)), cur)


def test_prop4d_draft_dependent_constraint_fails_under_the_parallel_update() -> None:
    """The paper's counterexample: two bits, constraint w1 + w2 <= 1, state (0, 0). Each position may move to 1
    given the OTHER position's current value, so the simultaneous update reaches the inadmissible (1, 1) with
    probability K~^1(1) K~^2(1) = pi^2 P1(1) P2(1) > 0; a sequential update never does."""
    pi = 0.5
    post = torch.tensor([[0.3, 0.7], [0.4, 0.6]], dtype=torch.float64)

    def draft_dependent(i: int, j: int, w: tuple[int, int]) -> float:
        return 0.0 if j + w[1 - i] <= 1 else INF

    k = two_bit_kernels((0, 0), post, pi, draft_dependent)
    assert k[0, 1].item() == pytest.approx(pi * 0.7, abs=1e-12)
    assert k[1, 1].item() == pytest.approx(pi * 0.6, abs=1e-12)
    joint = {(x, y): k[0, x].item() * k[1, y].item() for x in (0, 1) for y in (0, 1)}   # independent given w_t
    assert sum(joint.values()) == pytest.approx(1.0, abs=1e-12)
    assert joint[(1, 1)] == pytest.approx(pi * pi * 0.7 * 0.6, abs=1e-12) and joint[(1, 1)] > 0.0

    sequential = 0.0                                       # position 1 moves first, position 2 sees the new w1
    for x in (0, 1):
        k2 = two_bit_kernels((x, 0), post, pi, draft_dependent)
        sequential += k[0, x].item() * k2[1, 1].item() * (x == 1)
    assert sequential == 0.0

    gen = torch.Generator().manual_seed(51)                # the library sampler reaches (1, 1) at that rate
    draws = 100_000
    cur = torch.zeros(draws, 2, dtype=torch.long)
    energy = torch.tensor([[draft_dependent(i, j, (0, 0)) for j in (0, 1)] for i in (0, 1)], dtype=torch.float64)
    kern = tilted_jump_kernel(post.expand(draws, 2, 2), cur, pi, torch.exp(-energy).expand(draws, 2, 2))
    out = sample_jump(kern, cur, generator=gen)
    rate = (out.sum(-1) == 2).double().mean().item()
    se = math.sqrt(joint[(1, 1)] * (1 - joint[(1, 1)]) / draws)
    assert abs(rate - joint[(1, 1)]) <= 5 * se


def test_prop4d_plan_mediated_constraint_survives_the_parallel_update() -> None:
    """The same constraint routed through the plan: admissible sets A1 = {0, 1}, A2 = {0} that do not depend on
    the other position. From every admissible state the parallel update reaches an inadmissible state with
    probability exactly 0, and every reachable state satisfies w1 + w2 <= 1."""
    pi = 1.0
    post = torch.tensor([[0.3, 0.7], [0.4, 0.6]], dtype=torch.float64)
    admissible = ({0, 1}, {0})

    def plan_mediated(i: int, j: int, w: tuple[int, int]) -> float:
        return 0.0 if j in admissible[i] else INF

    for state in [(x, y) for x in admissible[0] for y in admissible[1]]:
        k = two_bit_kernels(state, post, pi, plan_mediated)
        bad = sum(k[0, x].item() * k[1, y].item() for x in (0, 1) for y in (0, 1)
                  if x not in admissible[0] or y not in admissible[1])
        assert bad == 0.0
        reachable = [(x, y) for x in (0, 1) for y in (0, 1) if k[0, x] * k[1, y] > 0]
        assert all(x + y <= 1 for x, y in reachable)


def test_prop4d_random_plan_mediated_sets_are_closed_under_parallel_updates() -> None:
    """Random position-wise admissible sets A_i(Z) (hard energies E^i(j | Z) in {0, inf}), random posteriors,
    T, c_max and pi, both the plain and the repair-aware multiplier: K~ puts zero mass outside A_i whenever the
    current token is in A_i, and repeated parallel draws never leave the admissible product set."""
    gen = torch.Generator().manual_seed(52)
    b, n, s = 8, 24, 13
    allowed = torch.rand(b, n, s, generator=gen) < 0.5
    allowed[..., 0] = True                                                       # non-empty sets
    energy = torch.where(allowed, 0.0, INF).double()                             # plan-mediated: fixed over steps
    w = torch.multinomial(allowed.reshape(-1, s).double(), 1, generator=gen).reshape(b, n)
    for step in range(6):
        post = random_posteriors(gen, (b, n), s, zero_frac=0.0)
        temperature, c_max = (0.3, 1.0) if step % 2 else (2.0, 0.25)
        for mult in (torch.exp(-energy / temperature), repair_aware_multiplier(energy, w, temperature)):
            kern = tilted_jump_kernel(post, w, 0.9, mult, c_max)
            dist = kernel_distribution(kern, w)
            assert (dist[~allowed] == 0).all()
            torch.testing.assert_close(dist.sum(-1), torch.ones(b, n, dtype=torch.float64), rtol=0, atol=1e-12)
        w = sample_jump(kern, w, generator=gen)
        assert allowed.gather(-1, w[..., None]).all()
