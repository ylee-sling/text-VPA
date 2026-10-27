# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
verification.py -- the numerical checks of PR-JEPA Sec. 4.6 (Table 2), ported from Appendix B.

Each function reproduces one check of the paper's reproducibility script ``pr_jepa_checks.py``
with the same NumPy random streams, so the printed numbers match Table 2:

    (1) Prop. 3    variance-hinge gradient norm vs sqrt(Var / ((n-1)(Var + eps))), n = 256
    (2) Thm. 1     lambda_min(C) >= gamma^2 - eps - sqrt(d(d-1) c(Z)) when v(Z) = 0  (3,207 of 20,000 draws;
        Prop. 2    smallest slack 0.0039) and rank(C) >= (tr C)^2 / ||C||_F^2
    (3) Thm. 2     K Euler steps amplify a plan error by at most (L_z/L_x)[(1 + L_x/K)^K - 1] (max ratio 0.83)
    (4) Lemma 1    exact jump mass pi_{t,h} in [0, 1]; Euler mass h kappa_dot / (1 - kappa) > 1 for sqrt
    (5) Prop. 5    sum_k pi_{t_k, 1/K} = H_K for the linear schedule
    (6) Thm. 4     risk-calibrated gate: max risk = alpha at the minimal beta
    (7) Lemma 2    l_1 - l_0 = (l_1 - l_t)/(1 - t) to machine precision (8.9e-16)

The functions depend only on NumPy; they are references, independent of the PyTorch library code
(``tests/test_theory_verification.py`` cross-checks the library against them). Run::

    python -m text_vpa.utils.verification
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AntiCollapseCheck",
    "GateCheck",
    "HingeGradientCheck",
    "JumpKernelCheck",
    "check_anticollapse_bound",
    "check_euler_sensitivity",
    "check_jump_kernels",
    "check_logit_kl_identity",
    "check_risk_gate",
    "check_variance_hinge_gradient",
    "euler_sensitivity_factor",
    "harmonic_jump_sums",
    "main",
    "run_all",
]

KAPPA: dict[str, tuple[Callable[[float], float], Callable[[float], float]]] = {
    "linear": (lambda t: t, lambda t: 1.0),
    "sqrt": (lambda t: t ** 0.5, lambda t: 0.5 * t ** -0.5),
    "cosine": (lambda t: 1 - math.cos(math.pi * t / 2), lambda t: math.pi / 2 * math.sin(math.pi * t / 2)),
}


# ---------------------------------------------------------------------------------------------
# (1) Proposition 3
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HingeGradientCheck:
    variance: float          # requested Var
    grad_norm: float         # ||grad of gamma - sqrt(Var + eps)||, analytic gradient on the sample
    formula: float           # sqrt(Var / ((n - 1)(Var + eps)))   (Eq. 26)
    ratio: float             # ||grad std-hinge|| / ||grad variance-hinge||
    ratio_formula: float     # 1 / (2 sqrt(Var + eps))


def check_variance_hinge_gradient(
    n: int = 256, eps: float = 1e-4, variances: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6), seed: int = 0
) -> list[HingeGradientCheck]:
    """Appendix B check (1): one dimension with n samples standardised to each variance."""
    rng = np.random.default_rng(seed)
    out = []
    for var in variances:
        z = rng.standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1) * math.sqrt(var)
        v = z.var(ddof=1)
        g_std = -(z - z.mean()) / ((n - 1) * math.sqrt(v + eps))
        g_var = -2 * (z - z.mean()) / (n - 1)
        out.append(HingeGradientCheck(
            variance=var,
            grad_norm=float(np.linalg.norm(g_std)),
            formula=math.sqrt(v / ((n - 1) * (v + eps))),
            ratio=float(np.linalg.norm(g_std) / np.linalg.norm(g_var)),
            ratio_formula=1 / (2 * math.sqrt(v + eps)),
        ))
    return out


# ---------------------------------------------------------------------------------------------
# (2) Theorem 1 and Proposition 2
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AntiCollapseCheck:
    num_draws: int
    num_psd: int
    premise_count: int        # draws with v(Z) = 0 and a positive bound
    min_slack: float          # min over premise draws of lambda_min - bound (must be >= 0)
    rank_ge_srank: bool       # rank(C) >= (tr C)^2 / ||C||_F^2 for every PSD draw
    premise_matrices: tuple[np.ndarray, ...]  # the covariances that met the premise (for cross-checks)


def check_anticollapse_bound(
    d: int = 32, trials: int = 20_000, gamma: float = 1.0, eps: float = 1e-4, seed: int = 1, keep: int = 0
) -> AntiCollapseCheck:
    """Appendix B check (2). ``keep`` premise matrices are returned for library cross-checks."""
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    kept: list[np.ndarray] = []
    srank_ok = True
    num_psd = 0
    for _ in range(trials):
        diag = rng.uniform(1.0, 2.0, d)
        s = rng.standard_normal((d, d))
        s = (s + s.T) / 2
        np.fill_diagonal(s, 0)
        s *= rng.uniform(0, 0.05)
        c_mat = np.diag(diag) + s
        lam = np.linalg.eigvalsh(c_mat)
        if lam.min() < 0:
            continue
        num_psd += 1
        off = c_mat - np.diag(np.diag(c_mat))
        c = (off ** 2).sum() / d
        v = np.mean(np.maximum(0, gamma - np.sqrt(np.diag(c_mat) + eps)))
        if np.linalg.matrix_rank(c_mat) < (np.trace(c_mat) ** 2) / (np.linalg.norm(c_mat, "fro") ** 2) - 1e-9:
            srank_ok = False
        bound = gamma ** 2 - eps - math.sqrt(d * (d - 1) * c)
        if v == 0 and bound > 0:
            gaps.append(float(lam.min() - bound))
            if len(kept) < keep:
                kept.append(c_mat)
    return AntiCollapseCheck(trials, num_psd, len(gaps), min(gaps) if gaps else math.inf, srank_ok, tuple(kept))


# ---------------------------------------------------------------------------------------------
# (3) Theorem 2
# ---------------------------------------------------------------------------------------------
def euler_sensitivity_factor(lx: float, num_steps: int) -> float:
    """(1 + L_x/K)^K - 1, the factor multiplying L_z / L_x in Theorem 2 (Eq. 27)."""
    return (1 + lx / num_steps) ** num_steps - 1


def check_euler_sensitivity(
    trials: int = 500, steps: Sequence[int] = (1, 2, 4, 8, 16), seed: int = 0, rng: np.random.Generator | None = None
) -> tuple[float, np.random.Generator]:
    """Appendix B check (3): the largest empirical / bound ratio over trials x len(steps) cases (<= 1).

    Returns the ratio and the generator, whose state check (7) continues from, as in the script.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    dx, dz = 16, 8
    max_ratio = 0.0
    for _ in range(trials):
        a = rng.standard_normal((dx, dx))
        a *= rng.uniform(0.2, 3.0) / np.linalg.norm(a, 2)
        b = rng.standard_normal((dx, dz))
        b *= rng.uniform(0.2, 2.0) / np.linalg.norm(b, 2)
        lx, lz = np.linalg.norm(a, 2), np.linalg.norm(b, 2)
        rng.standard_normal((dx, dx))  # the script draws an unused matrix W here; keep the stream aligned

        def velocity(x: np.ndarray, z: np.ndarray, a: np.ndarray = a, b: np.ndarray = b) -> np.ndarray:
            return np.tanh(a @ x + b @ z)

        for k in steps:
            x0 = rng.standard_normal(dx)
            z = rng.standard_normal(dz)
            dzv = 1e-3 * rng.standard_normal(dz)
            xa, xb = x0.copy(), x0.copy()
            for _ in range(k):
                xa = xa + velocity(xa, z) / k
                xb = xb + velocity(xb, z + dzv) / k
            emp = np.linalg.norm(xa - xb) / np.linalg.norm(dzv)
            bound = (lz / lx) * euler_sensitivity_factor(lx, k)
            max_ratio = max(max_ratio, float(emp / bound))
    return max_ratio, rng


# ---------------------------------------------------------------------------------------------
# (4) Lemma 1 and (5) Proposition 5
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class JumpKernelCheck:
    schedule: str
    num_steps: int
    max_euler_mass: float     # max_k h kappa_dot(t_k) / (1 - kappa(t_k)); a probability only if <= 1
    exact_valid: bool         # every exact pi_{t_k, h} lies in [0, 1]
    exact_masses: tuple[float, ...]


def check_jump_kernels(steps: Sequence[int] = (2, 4, 8)) -> list[JumpKernelCheck]:
    """Appendix B check (4) for the linear, sqrt and cosine schedules."""
    out = []
    for name, (kap, dkap) in KAPPA.items():
        for k_steps in steps:
            h = 1 / k_steps
            worst = 0.0
            ok = True
            masses = []
            for k in range(k_steps):
                t = k * h
                if t == 0 and name == "sqrt":
                    t = 1e-12
                euler = h * dkap(max(t, 1e-12)) / (1 - kap(t))
                exact = (kap(min(t + h, 1)) - kap(t)) / (1 - kap(t))
                worst = max(worst, euler)
                ok &= 0 <= exact <= 1 + 1e-12
                masses.append(exact)
            out.append(JumpKernelCheck(name, k_steps, worst, ok, tuple(masses)))
    return out


def harmonic_jump_sums(steps: Sequence[int] = (1, 2, 4, 8, 16)) -> dict[int, float]:
    """Appendix B check (5): sum_k pi_{t_k,1/K} = sum_k 1/(K - k) = H_K (linear schedule)."""
    return {k_steps: sum(1 / (k_steps - k) for k in range(k_steps)) for k_steps in steps}


# ---------------------------------------------------------------------------------------------
# (6) Theorem 4
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GateCheck:
    h_max: int
    tau: float
    alpha: float
    beta_min: float
    max_risk: float


def check_risk_gate(
    settings: Sequence[tuple[int, float, float]] = ((16, 1.0, 0.05), (32, 1.0, 0.05), (64, 1.0, 0.1)),
) -> list[GateCheck]:
    """Appendix B check (6): beta = 2 sqrt(H_max) / (e tau sqrt(alpha)) gives max risk = alpha (Eq. 32)."""
    out = []
    sig = np.linspace(1e-4, 5, 200_000)
    for h_max, tau, alpha in settings:
        beta = 2 * math.sqrt(h_max) / (math.e * tau * math.sqrt(alpha))
        risk = h_max * np.exp(-beta * sig) * sig ** 2 / tau ** 2
        out.append(GateCheck(h_max, tau, alpha, beta, float(risk.max())))
    return out


# ---------------------------------------------------------------------------------------------
# (7) Lemma 2
# ---------------------------------------------------------------------------------------------
def check_logit_kl_identity(rng: np.random.Generator, vocab: int = 7, smoothing: float = 0.05, t: float = 0.6,
                            token: int = 3) -> float:
    """Appendix B check (7): max |(l_1 - l_0) - (l_1 - l_t)/(1 - t)| for a smoothed one-hot target."""
    a = math.log((1 - smoothing + smoothing / vocab) / (smoothing / vocab))
    b = math.log(smoothing / vocab)
    rng.dirichlet(np.ones(vocab))  # the script draws an unused posterior p here; keep the stream aligned
    l0 = rng.standard_normal(vocab)
    l1 = b * np.ones(vocab)
    l1[token] += a
    lt = (1 - t) * l0 + t * l1
    return float(np.abs((l1 - l0) - (l1 - lt) / (1 - t)).max())


# ---------------------------------------------------------------------------------------------
def run_all() -> dict[str, object]:
    """All checks with the script's seeds and stream order."""
    hinge = check_variance_hinge_gradient()
    collapse = check_anticollapse_bound()
    euler_ratio, rng = check_euler_sensitivity()
    return {
        "hinge": hinge,
        "anticollapse": collapse,
        "euler_max_ratio": euler_ratio,
        "jump_kernels": check_jump_kernels(),
        "harmonic": harmonic_jump_sums(),
        "gate": check_risk_gate(),
        "logit_kl_error": check_logit_kl_identity(rng),
    }


def main() -> None:
    r = run_all()
    print("=== (1) Prop. 3: variance-hinge gradient norm (n = 256, eps = 1e-4) ===")
    for c in r["hinge"]:  # type: ignore[union-attr]
        print(f"Var={c.variance:.0e} ||g||={c.grad_norm:.4e} (formula {c.formula:.4e}) "
              f"ratio={c.ratio:.2f} (1/(2 sqrt(Var+eps))={c.ratio_formula:.2f})")
    a = r["anticollapse"]
    print("\n=== (2) Thm. 1 / Prop. 2 ===")
    print(f"premise met in {a.premise_count} of {a.num_draws} draws; min(lambda_min - bound) = "  # type: ignore[union-attr]
          f"{a.min_slack:.5f} (>= 0); rank >= srank: {a.rank_ge_srank}")  # type: ignore[union-attr]
    print(f"\n=== (3) Thm. 2: max empirical/bound = {r['euler_max_ratio']:.3f} (<= 1) ===")
    for k in (1, 2, 4, 8, 16):
        print(f" K={k}: (1 + 1/K)^K - 1 = {euler_sensitivity_factor(1.0, k):.3f}")
    print("\n=== (4) Lemma 1: exact vs Euler jump mass ===")
    for c in r["jump_kernels"]:  # type: ignore[union-attr]
        print(f" {c.schedule:7s} K={c.num_steps}: max Euler mass {c.max_euler_mass:9.3f}   exact in [0,1]: {c.exact_valid}")
    print("\n=== (5) Prop. 5: harmonic sums ===")
    for k, s in r["harmonic"].items():  # type: ignore[union-attr]
        print(f" K={k}: H_K = {s:.3f}")
    print("\n=== (6) Thm. 4: risk-calibrated gate ===")
    for g in r["gate"]:  # type: ignore[union-attr]
        print(f" H_max={g.h_max} tau={g.tau} alpha={g.alpha}: beta_min={g.beta_min:.3f} max risk={g.max_risk:.4f}")
    print(f"\n=== (7) Lemma 2: max error = {r['logit_kl_error']:.1e} ===")


if __name__ == "__main__":
    main()
