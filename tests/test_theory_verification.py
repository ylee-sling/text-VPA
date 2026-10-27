# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Numerical verification of the PR-JEPA theory (Sec. 4.6, Table 2; Appendix B).

Two layers per check:
  * the Appendix B reference (``text_vpa.utils.verification``, NumPy only) must reproduce Table 2;
  * the PyTorch library code must agree with that reference and with independent closed forms.

    Check 1  Prop. 3    variance-hinge gradient norm sqrt(Var / ((n-1)(Var + eps)))
    Check 2  Thm. 1     Gershgorin anti-collapse bound;  Prop. 2  stable rank <= rank
    Check 3  Lemma 1    exact jump mass pi_{t,h} in [0, 1] for linear / sqrt / cosine, K in {2, 4, 8}
    Check 4  Lemma 2    Logit-KL velocity identity and posterior parameterisation
    extra    Thm. 2 (Euler sensitivity), Prop. 5 (harmonic sum), Thm. 4 (risk-calibrated gate)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from text_vpa.engine.losses import (
    calculate_stable_rank,
    covariance_diagnostics,
    covariance_penalty,
    stable_rank,
    theorem1_lambda_min_bound,
    variance_hinge,
)
from text_vpa.models.realizers.paths import (
    KappaSchedule,
    conditional_logit_velocity,
    logit_euler_step,
    logit_path,
    posterior_logit_velocity,
    smoothed_logit_coefficients,
    smoothed_target_logits,
    uniform_time_grid,
)
from text_vpa.utils import verification as ref

GAMMA, EPS = 1.0, 1e-4


@pytest.fixture(scope="module")
def anticollapse() -> ref.AntiCollapseCheck:
    return ref.check_anticollapse_bound(keep=64)


def exact_sample(cov: np.ndarray, n: int, seed: int) -> torch.Tensor:
    """Rows Z [n, d] whose unbiased sample covariance is exactly ``cov`` (float64).

    Z = sqrt(n - 1) Q L^T with Q [n, d] orthonormal and orthogonal to the all-ones vector, L L^T = cov.
    """
    d = cov.shape[0]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, d))
    a -= a.mean(axis=0, keepdims=True)                       # columns orthogonal to 1
    q, _ = np.linalg.qr(a)
    q -= q.mean(axis=0, keepdims=True)                       # QR keeps the span; re-centre numerically
    q, _ = np.linalg.qr(q)
    z = math.sqrt(n - 1) * q @ np.linalg.cholesky(cov).T
    return torch.from_numpy(z)


# =============================================================================================
# Check 1: Proposition 3
# =============================================================================================
def test_check1_reference_reproduces_table2() -> None:
    """Appendix B (1): the gradient norm matches the closed form to 4 significant digits at every Var."""
    checks = ref.check_variance_hinge_gradient()
    assert [c.variance for c in checks] == [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]
    for c in checks:
        assert f"{c.grad_norm:.4e}" == f"{c.formula:.4e}"
        assert c.ratio == pytest.approx(c.ratio_formula, rel=1e-12)


@pytest.mark.parametrize("variance", [1e-1, 1e-2, 1e-3, 1e-4, 1e-6])
def test_check1_library_hinge_gradient_matches_eq26(variance: float) -> None:
    """Autograd of the library's v(Z) on one active dimension equals Eq. 26 times the 1/d of v(Z)."""
    n, d = 256, 3
    gen = torch.Generator().manual_seed(7)
    z = torch.randn(n, d, generator=gen, dtype=torch.float64)
    z = (z - z.mean(0)) / z.std(0, unbiased=True) * torch.tensor([math.sqrt(variance), 2.0, 3.0], dtype=torch.float64)
    z.requires_grad_(True)
    variance_hinge(z, GAMMA, EPS).backward()
    var = float(z[:, 0].detach().var(unbiased=True))
    eq26 = math.sqrt(var / ((n - 1) * (var + EPS)))
    assert float(z.grad[:, 0].norm()) == pytest.approx(eq26 / d, rel=1e-10)
    # Dimensions 1 and 2 have std >= gamma: their hinges are inactive and contribute no gradient.
    assert torch.count_nonzero(z.grad[:, 1:]) == 0


def test_check1_gradient_does_not_vanish_above_eps() -> None:
    """Sec. 4.2: the gradient keeps 99.5% of its asymptotic size at Var = 1e-2 and 71% at Var = 1e-4."""
    def kept(var: float) -> float:
        return math.sqrt(var / (var + EPS))           # Eq. 26 divided by its limit 1/sqrt(n - 1)
    assert kept(1e-2) == pytest.approx(0.995, abs=5e-4)
    assert kept(1e-4) == pytest.approx(0.7071, abs=1e-4)


def test_check1_variance_hinge_ratio_matches_formula() -> None:
    """The std hinge's gradient is 1/(2 sqrt(Var + eps)) times that of a hinge on the variance itself."""
    n = 128
    z = torch.randn(n, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(3)) * 0.05
    z_std = z.clone().requires_grad_(True)
    variance_hinge(z_std, GAMMA, EPS).backward()
    z_var = z.clone().requires_grad_(True)
    torch.relu(GAMMA - z_var.var(dim=0, unbiased=True)).sum().backward()
    var = float(z.var(unbiased=True))
    assert float(z_std.grad.norm() / z_var.grad.norm()) == pytest.approx(1 / (2 * math.sqrt(var + EPS)), rel=1e-10)


# =============================================================================================
# Check 2: Theorem 1 and Proposition 2
# =============================================================================================
def test_check2_reference_reproduces_table2(anticollapse: ref.AntiCollapseCheck) -> None:
    """Appendix B (2): 3,207 of 20,000 draws meet the premise; smallest slack 0.0039; srank <= rank always."""
    assert anticollapse.num_draws == 20_000
    assert anticollapse.premise_count == 3207
    assert anticollapse.min_slack >= 0.0
    # The script prints f"{min:.5f}" = 0.00385; Table 2's "0.0039" rounds that printed value once more.
    assert f"{anticollapse.min_slack:.5f}" == "0.00385"
    assert anticollapse.rank_ge_srank


def test_check2_library_agrees_on_premise_draws(anticollapse: ref.AntiCollapseCheck) -> None:
    """On rows with exactly the drawn covariance, v, c, srank and the Theorem 1 bound match NumPy."""
    d = 32
    for i, cov in enumerate(anticollapse.premise_matrices):
        z = exact_sample(cov, n=4 * d, seed=i)
        cov_t = torch.from_numpy(cov)
        off = cov - np.diag(np.diag(cov))
        c_ref = (off ** 2).sum() / d
        assert float(variance_hinge(z, GAMMA, EPS)) == 0.0                          # premise v(Z) = 0
        assert float(covariance_penalty(z)) == pytest.approx(c_ref, rel=1e-9)
        bound_ref = GAMMA ** 2 - EPS - math.sqrt(d * (d - 1) * c_ref)
        assert float(theorem1_lambda_min_bound(cov_t, GAMMA, EPS)) == pytest.approx(bound_ref, rel=1e-12)
        assert np.linalg.eigvalsh(cov).min() >= bound_ref                            # Theorem 1 (Eq. 24)
        srank_ref = np.trace(cov) ** 2 / np.linalg.norm(cov, "fro") ** 2
        assert float(calculate_stable_rank(z, min_fraction=0.5)) == pytest.approx(srank_ref, rel=1e-9)
        diag = covariance_diagnostics(cov_t, num_rows=4 * d, gamma=GAMMA, eps=EPS)
        assert diag.certified_lambda_min == pytest.approx(bound_ref, rel=1e-12)
        assert diag.lambda_min >= diag.certified_lambda_min


def test_check2_prop2_identity_and_bound_on_random_covariances() -> None:
    """Eq. 25: srank = (sum C_jj)^2 / (sum C_jj^2 + d c(Z)) <= rank(C), including rank-deficient C."""
    rng = np.random.default_rng(11)
    for trial in range(200):
        d = int(rng.integers(2, 12))
        n = int(rng.integers(2, 3 * d))                    # n <= d gives rank-deficient covariances
        z = torch.from_numpy(rng.standard_normal((n, d)) @ rng.standard_normal((d, d)))
        cov = np.cov(z.numpy(), rowvar=False, ddof=1).reshape(d, d)
        diag = np.diag(cov)
        c = ((cov - np.diag(diag)) ** 2).sum() / d
        identity = diag.sum() ** 2 / ((diag ** 2).sum() + d * c)
        lib = float(calculate_stable_rank(z))
        assert lib == pytest.approx(identity, rel=1e-9), trial
        assert lib <= np.linalg.matrix_rank(cov, tol=1e-9) + 1e-9, trial


@pytest.mark.parametrize("s2", [0.5, 1.0, 3.0])
def test_check2_prop2_equal_variance_formula(s2: float) -> None:
    """Prop. 2: if every variance equals s^2, srank = d / (1 + c(Z)/s^4) exactly.

    (Table 2 lists one exact case with value 23.0764; Appendix B does not give that instance, so the
    identity is checked on constructed covariances instead.)
    """
    d = 24
    rng = np.random.default_rng(5)
    a = rng.standard_normal((d, d))
    off = 0.02 * s2 * (a + a.T)
    np.fill_diagonal(off, 0.0)
    cov = s2 * np.eye(d) + off
    c = (off ** 2).sum() / d
    assert float(stable_rank(torch.from_numpy(cov))) == pytest.approx(d / (1 + c / s2 ** 2), rel=1e-12)


def test_check2_c_over_s4_of_one_tenth_guarantees_091d() -> None:
    """Sec. 4.2: c(Z)/s^4 = 0.1 guarantees rank >= d / 1.1 > 0.909 d."""
    assert 1 / (1 + 0.1) == pytest.approx(0.909, abs=1e-3)


def test_check2_collapsed_batches() -> None:
    """A point (all rows equal) has v = gamma - sqrt(eps), c = 0, srank 0; a line has srank exactly 1."""
    point = torch.ones(10, 6, dtype=torch.float64)
    assert float(variance_hinge(point, GAMMA, EPS)) == pytest.approx(GAMMA - math.sqrt(EPS), rel=1e-12)
    assert float(covariance_penalty(point)) == 0.0
    assert float(calculate_stable_rank(point)) == 0.0
    line = torch.linspace(-1, 1, 10, dtype=torch.float64)[:, None] * torch.arange(1, 7, dtype=torch.float64)
    assert float(calculate_stable_rank(line)) == pytest.approx(1.0, rel=1e-12)


# =============================================================================================
# Check 3: Lemma 1 (exact few-step kernel) and Proposition 5
# =============================================================================================
def test_check3_reference_reproduces_table2() -> None:
    """Appendix B (4): the exact mass is always valid; the Euler mass is far above 1 for sqrt."""
    checks = {(c.schedule, c.num_steps): c for c in ref.check_jump_kernels()}
    assert len(checks) == 9
    assert all(c.exact_valid for c in checks.values())
    for k in (2, 4, 8):
        assert checks[("sqrt", k)].max_euler_mass > 1e4
        assert checks[("linear", k)].max_euler_mass == pytest.approx(1.0)        # equality at the last step
        assert checks[("cosine", k)].max_euler_mass <= 1.0


@pytest.mark.parametrize("schedule", ["linear", "sqrt", "cosine"])
@pytest.mark.parametrize("num_steps", [2, 4, 8])
def test_check3_library_exact_mass_in_unit_interval(schedule: str, num_steps: int) -> None:
    """Library pi_{t_k,1/K} (Eq. 15) equals the reference on the grid and lies in [0, 1]."""
    lib = KappaSchedule(schedule)
    reference = {(c.schedule, c.num_steps): c for c in ref.check_jump_kernels()}[(schedule, num_steps)]
    masses = lib.step_masses(num_steps)
    assert len(masses) == num_steps
    for k, (m_lib, m_ref) in enumerate(zip(masses, reference.exact_masses, strict=True)):
        assert 0.0 <= m_lib <= 1.0
        if not (schedule == "sqrt" and k == 0):           # the reference evaluates sqrt at t = 1e-12 instead of 0
            assert m_lib == pytest.approx(m_ref, rel=1e-12, abs=1e-15)
    assert masses[-1] == pytest.approx(1.0)               # pi = 1 at the last step: kappa_1 = 1


@pytest.mark.parametrize("schedule", ["linear", "sqrt", "cosine"])
def test_check3_lemma1_exactness_identity(schedule: str) -> None:
    """Lemma 1: kappa_t + (1 - kappa_t) pi_{t,h} = kappa_{t+h} for every grid point and step size."""
    lib = KappaSchedule(schedule)
    for t in np.linspace(0.0, 0.95, 20):
        for h in (0.05, 0.25, 1.0 - t):
            arrived = lib.kappa(float(t)) + (1 - lib.kappa(float(t))) * lib.exact_jump_mass(float(t), float(h))
            assert arrived == pytest.approx(lib.kappa(min(float(t + h), 1.0)), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("schedule", ["linear", "cosine", "sqrt"])
def test_check3_lemma1_monte_carlo_marginals(schedule: str) -> None:
    """Lemma 1 by simulation: with the oracle posterior the arrived fraction after step k equals kappa_{t_{k+1}}."""
    lib = KappaSchedule(schedule)
    num_chains, num_steps = 200_000, 8
    gen = torch.Generator().manual_seed(0)
    arrived = torch.zeros(num_chains, dtype=torch.bool)       # all chains start at w_0 != w_1
    for k, (t, h) in enumerate(uniform_time_grid(num_steps)):
        pi = lib.exact_jump_mass(t, h)
        # Oracle posterior delta_{w_1}: a chain that has not arrived jumps (onto w_1) with probability pi.
        jump = torch.rand(num_chains, generator=gen) < pi
        arrived |= jump
        expected = lib.kappa(min(t + h, 1.0))
        se = math.sqrt(max(expected * (1 - expected), 1e-12) / num_chains)
        assert abs(arrived.double().mean().item() - expected) <= 5 * se + 1e-12, (k, expected)
    assert bool(arrived.all())                                # kappa_1 = 1: every chain has arrived


def test_check3_euler_kernel_invalid_for_sqrt_valid_for_linear() -> None:
    """'Why not Euler': h kappa_dot / (1 - kappa) > 1 near t = 0 for sqrt; <= 1 on the linear grid."""
    sqrt_schedule, linear = KappaSchedule("sqrt"), KappaSchedule("linear")
    assert sqrt_schedule.euler_jump_mass(1e-6, 0.125) > 1.0
    assert math.isinf(sqrt_schedule.euler_jump_mass(0.0, 0.125))
    for t, h in uniform_time_grid(8):
        assert linear.euler_jump_mass(t, h) <= 1.0 + 1e-12


@pytest.mark.parametrize("num_steps", [1, 2, 4, 8, 16])
def test_prop5_harmonic_sum(num_steps: int) -> None:
    """Prop. 5: sum_k pi_{t_k,1/K} = H_K for the linear schedule (H_8 = 2.72)."""
    lib_sum = sum(KappaSchedule("linear").step_masses(num_steps))
    harmonic = sum(1 / j for j in range(1, num_steps + 1))
    assert lib_sum == pytest.approx(harmonic, rel=1e-12)
    assert ref.harmonic_jump_sums()[num_steps] == pytest.approx(harmonic, rel=1e-12)
    if num_steps == 8:
        assert round(harmonic, 2) == 2.72


def test_jump_mass_tensor_and_float_paths_agree() -> None:
    """The tensor overload of Eq. 15 equals the float overload."""
    for name in ("linear", "sqrt", "cosine"):
        lib = KappaSchedule(name)
        t = torch.tensor([0.0, 0.125, 0.5, 0.875], dtype=torch.float64)
        got = lib.exact_jump_mass(t, 0.125)
        want = torch.tensor([lib.exact_jump_mass(float(x), 0.125) for x in t], dtype=torch.float64)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-15)


# =============================================================================================
# Check 4: Lemma 2 (Logit-KL)
# =============================================================================================
def test_check4_reference_reproduces_table2() -> None:
    """Appendix B (7), with the script's random stream: error 8.9e-16."""
    _, rng = ref.check_euler_sensitivity()
    assert f"{ref.check_logit_kl_identity(rng):.1e}" == "8.9e-16"


def test_check4_library_identity() -> None:
    """l_1 - l_0 = (l_1 - l_t)/(1 - t) with the library's path and smoothed targets (Eqs. 18-19)."""
    vocab, smoothing = 7, 0.05
    tokens = torch.tensor([3, 0, 6])
    l1 = smoothed_target_logits(tokens, vocab, smoothing).double()
    l0 = torch.randn(3, vocab, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    for t in (0.0, 0.3, 0.6, 0.95):
        lt = logit_path(l0, l1, t)
        torch.testing.assert_close(conditional_logit_velocity(l1, lt, t), l1 - l0, rtol=0, atol=1e-13)


def test_check4_smoothed_target_coefficients() -> None:
    """Eq. 18: softmax(a e_k + b 1) is exactly the smoothed one-hot (1 - eps) e_k + (eps/S) 1."""
    vocab, smoothing = 11, 0.05
    a, b = smoothed_logit_coefficients(smoothing, vocab)
    assert a == pytest.approx(math.log((1 - smoothing + smoothing / vocab) / (smoothing / vocab)), rel=1e-15)
    assert b == pytest.approx(math.log(smoothing / vocab), rel=1e-15)
    l1 = a * torch.nn.functional.one_hot(torch.tensor(4), vocab).double() + b
    x1 = torch.softmax(l1, dim=0)
    want = torch.full((vocab,), smoothing / vocab, dtype=torch.float64)
    want[4] += 1 - smoothing
    torch.testing.assert_close(x1, want, rtol=1e-12, atol=1e-15)


def test_check4_lemma2_posterior_velocity_equals_conditional_expectation() -> None:
    """Lemma 2 (Eq. 20): E[l_1 - l_0 | l_t = l] = (a p(.|l) + b 1 - l)/(1 - t) for a discrete prior over k.

    With l_0 ~ N(0, s^2 I) and w_1 ~ prior, l_t | w_1 = k ~ N(t l_1(k), (1-t)^2 s^2 I), so the posterior
    p(k | l_t = l) and the conditional expectation are available in closed form and are computed here
    by explicit enumeration over k, independently of the library.
    """
    vocab, smoothing, s = 5, 0.1, 1.3
    a, b = smoothed_logit_coefficients(smoothing, vocab)
    prior = np.array([0.1, 0.4, 0.2, 0.25, 0.05])
    rng = np.random.default_rng(4)
    targets = [b * np.ones(vocab) + a * np.eye(vocab)[k] for k in range(vocab)]
    for t in (0.1, 0.5, 0.9):
        for _ in range(5):
            k_true = rng.choice(vocab, p=prior)
            l = (1 - t) * s * rng.standard_normal(vocab) + t * targets[k_true]
            log_w = np.log(prior) - np.array([((l - t * l1) ** 2).sum() for l1 in targets]) / (2 * ((1 - t) * s) ** 2)
            post = np.exp(log_w - log_w.max())
            post /= post.sum()
            # E[l_1 - l_0 | l_t = l] = sum_k p(k|l) (l_1(k) - (l - t l_1(k))/(1 - t))
            expectation = sum(post[k] * (targets[k] - (l - t * targets[k]) / (1 - t)) for k in range(vocab))
            lib = posterior_logit_velocity(torch.from_numpy(post), torch.from_numpy(l), t, a, b)
            np.testing.assert_allclose(lib.numpy(), expectation, rtol=1e-10, atol=1e-10)


def test_check4_last_euler_step_returns_scaled_posterior() -> None:
    """Eq. 21: at t_{K-1} = 1 - 1/K the update returns l_K = a p exactly (t = 1 is never evaluated)."""
    a = 4.0
    p = torch.softmax(torch.randn(3, 6, dtype=torch.float64), dim=-1)
    lk = torch.randn(3, 6, dtype=torch.float64)
    for num_steps in (1, 2, 4, 8):
        t, h = uniform_time_grid(num_steps)[-1]
        out = logit_euler_step(lk, p, t, h, a)
        torch.testing.assert_close(out, a * p, rtol=0, atol=0)
    t, h = uniform_time_grid(4)[1]
    torch.testing.assert_close(logit_euler_step(lk, p, t, h, a), lk + h * (a * p - lk) / (1 - t), rtol=1e-14, atol=0)


# =============================================================================================
# Theorem 2 and Theorem 4
# =============================================================================================
def test_theorem2_euler_sensitivity_reference() -> None:
    """Appendix B (3): largest empirical / bound ratio 0.83 over 2,500 cases; factors 1.00 ... 1.64."""
    ratio, _ = ref.check_euler_sensitivity()
    assert ratio <= 1.0
    assert round(ratio, 2) == 0.83
    factors = [round(ref.euler_sensitivity_factor(1.0, k), 2) for k in (1, 2, 4, 8, 16)]
    assert factors == [1.0, 1.25, 1.44, 1.57, 1.64]
    assert all(ref.euler_sensitivity_factor(1.0, k) <= math.e - 1 for k in (1, 2, 4, 8, 16, 1024))


def test_theorem4_risk_gate_reference() -> None:
    """Appendix B (6): at beta_min the maximal risk equals alpha (0.05, 0.05, 0.10); beta >= 13.2 for H_max = 16."""
    gates = ref.check_risk_gate()
    assert [round(g.max_risk, 4) for g in gates] == [0.05, 0.05, 0.10]
    for g in gates:
        assert g.max_risk <= g.alpha + 1e-9
    assert round(gates[0].beta_min, 1) == 13.2
