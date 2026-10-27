# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Equation-level tests of ``text_vpa.engine.losses`` against PR-JEPA Secs. 3.1, 4.2 and 5.1.

Every reference below is computed independently from the paper's formulas with explicit Python loops
in float64 (the ``ref_*`` helpers), never with the function under test:

    Eq. 5   L_plan = (1/(BM)) sum_b ||s_hat_y^(b) - sg(s_y^(b))||_F^2
                   + lambda_v [v(Z_x) + v(Z_hat)] + lambda_c [c(Z_x) + c(Z_hat)] + lambda_u L_CE
    Eq. 6   v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(z^j) + eps)),  Var(z^j) = (1/(n-1)) sum_i (z_i^j - zbar^j)^2
    Eq. 7   C(Z) = (1/(n-1)) sum_i (z_i - zbar)(z_i - zbar)^T,  c(Z) = (1/d) sum_{j != k} C(Z)_jk^2
    Eq. 24  Theorem 1: v(Z) = 0 and c(Z) < (gamma^2 - eps)^2 / (d(d-1))
            => lambda_min(C) >= gamma^2 - eps - sqrt(d(d-1) c(Z)) > 0
    Eq. 25  Prop. 2: rank(C) >= (tr C)^2 / ||C||_F^2 = (sum_j C_jj)^2 / (sum_j C_jj^2 + d c(Z))
    Eq. 26  Prop. 3: ||grad_{z^j} (gamma - sqrt(Var(z^j) + eps))|| = sqrt(Var / ((n-1)(Var + eps)))

The library weights the prediction term by ``lambda_invar`` (Eq. 5 is lambda_invar = 1) and, for an
ensemble, applies it to every head before reducing over the heads.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.conftest import ATOL, RTOL
from text_vpa.configs.schema import VICRegConfig
from text_vpa.engine.losses import (
    ModeCrossEntropyLoss,
    PlanLoss,
    RepresentationCollapseError,
    StableRankMonitor,
    StreamingCovariance,
    VICRegLoss,
    calculate_stable_rank,
    covariance_diagnostics,
    covariance_penalty,
    masked_rows,
    sample_covariance,
    stable_rank,
    theorem1_lambda_min_bound,
    variance_hinge,
)

F64 = torch.float64
F32 = torch.float32
#: float32 module output vs float64 reference.
LOOSE = {"rtol": RTOL, "atol": ATOL}
#: float64 inputs: library and loop reference differ only in the rounding order of O(n d^2) flops.
TIGHT = {"rtol": 1e-12, "atol": 1e-12}
TOL = {F32: LOOSE, F64: TIGHT}
DTYPES = pytest.mark.parametrize("dtype", [F32, F64], ids=["float32", "float64"])

#: Distinct weights, so a swapped or dropped weight changes the total.
WEIGHTS = {"lambda_invar": 2.0, "lambda_v": 3.0, "lambda_c": 0.5, "gamma": 1.2, "eps": 1e-3, "normalization": "frobenius"}
#: Real context tokens (B = 2, M_c = 4) and real slots (B = 2, M = 5); padding need not be a suffix.
CONTEXT_MASK = torch.tensor([[True, True, True, False], [True, True, False, False]])
SLOT_MASK = torch.tensor([[True, True, True, True, True], [True, False, True, False, False]])

Rows = list[list[float]]


# ---------------------------------------------------------------------------------------------
# Independent float64 references (explicit loops over Python floats)
# ---------------------------------------------------------------------------------------------
def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def py(x: torch.Tensor) -> list:
    """Exact values of ``x`` as nested Python floats (float32 -> float64 is exact)."""
    return x.detach().to(device="cpu", dtype=F64).tolist()


def assert_matches(actual, expected, tol: dict[str, float] = LOOSE) -> None:
    """Compare a library value (tensor of any float dtype, or float) with a float64 reference."""
    def as64(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=F64)
        return torch.as_tensor(value, dtype=F64)

    torch.testing.assert_close(as64(actual), as64(expected), **tol)


def ref_mean(rows: Rows) -> list[float]:
    n, d = len(rows), len(rows[0])
    return [sum(row[j] for row in rows) / n for j in range(d)]


def ref_covariance(rows: Rows) -> Rows:
    """C(Z) = (1/(n-1)) sum_i (z_i - zbar)(z_i - zbar)^T  (Eq. 7)."""
    n, d = len(rows), len(rows[0])
    mean = ref_mean(rows)
    return [[sum((row[j] - mean[j]) * (row[k] - mean[k]) for row in rows) / (n - 1) for k in range(d)]
            for j in range(d)]


def ref_column_variance(rows: Rows, j: int) -> float:
    """Var(z^j) = (1/(n-1)) sum_i (z_i^j - zbar^j)^2  (Eq. 6)."""
    n = len(rows)
    mean = sum(row[j] for row in rows) / n
    return sum((row[j] - mean) ** 2 for row in rows) / (n - 1)


def ref_variance_hinge(rows: Rows, gamma: float, eps: float) -> float:
    """v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(z^j) + eps))  (Eq. 6)."""
    d = len(rows[0])
    return sum(max(0.0, gamma - math.sqrt(ref_column_variance(rows, j) + eps)) for j in range(d)) / d


def ref_off_diagonal_mass(cov: Rows) -> float:
    """c = (1/d) sum_{j != k} C_jk^2 of a given covariance  (Eq. 7)."""
    d = len(cov)
    return sum(cov[j][k] ** 2 for j in range(d) for k in range(d) if j != k) / d


def ref_covariance_penalty(rows: Rows) -> float:
    return ref_off_diagonal_mass(ref_covariance(rows))


def ref_stable_rank(cov: Rows) -> float:
    """(tr C)^2 / ||C||_F^2 (Prop. 2); 0 for the zero matrix."""
    d = len(cov)
    trace = sum(cov[j][j] for j in range(d))
    frobenius_sq = sum(cov[j][k] ** 2 for j in range(d) for k in range(d))
    return trace ** 2 / frobenius_sq if frobenius_sq > 0 else 0.0


def ref_theorem1_bound(cov: Rows, gamma: float, eps: float) -> float:
    """gamma^2 - eps - sqrt(d(d-1) c)  (Eq. 24)."""
    d = len(cov)
    return gamma ** 2 - eps - math.sqrt(d * (d - 1) * ref_off_diagonal_mass(cov))


def ref_real_rows(x: torch.Tensor, mask: torch.Tensor | None) -> Rows:
    """Rows of ``x [B, L, d]`` at the real positions, stacked (Z_x or Z_hat of Eq. 5)."""
    keep = None if mask is None else mask.tolist()
    return [row for b, rows_b in enumerate(py(x)) for m, row in enumerate(rows_b) if keep is None or keep[b][m]]


def ref_invariance(heads: torch.Tensor, target: torch.Tensor, slot_mask: torch.Tensor | None,
                   head_reduction: str) -> float:
    """Eq. 5's prediction term for each head, (1/#slots) sum_{real (b, m)} sum_j (s_hat - s_y)^2, then the mean
    or the sum over the heads."""
    keep = None if slot_mask is None else slot_mask.tolist()
    target_values = py(target)
    per_head = []
    for head in py(heads):
        total, slots = 0.0, 0
        for b, (pred_b, target_b) in enumerate(zip(head, target_values, strict=True)):
            for m, (pred, tgt) in enumerate(zip(pred_b, target_b, strict=True)):
                if keep is None or keep[b][m]:
                    total += sum((p - t) ** 2 for p, t in zip(pred, tgt, strict=True))
                    slots += 1
        per_head.append(total / slots)
    return sum(per_head) / len(per_head) if head_reduction == "mean" else sum(per_head)


def ref_vicreg(batch: dict[str, torch.Tensor], context_mask: torch.Tensor | None, slot_mask: torch.Tensor | None, *,
               lambda_invar: float, lambda_v: float, lambda_c: float, gamma: float, eps: float,
               head_reduction: str = "mean", normalization: str = "frobenius") -> tuple[float, dict[str, float]]:
    """The JEPA part of Eq. 5 with the prediction term weighted by lambda_invar.

    ``frobenius`` is Eq. 5 verbatim; ``per_dim`` divides the prediction term by d (VICReg's per-element MSE).
    """
    z_x = ref_real_rows(batch["context"], context_mask)
    z_hat = ref_real_rows(batch["plan"], slot_mask)
    invariance = ref_invariance(batch["predictions"], batch["target"], slot_mask, head_reduction)
    if normalization == "per_dim":
        invariance /= batch["target"].shape[-1]
    terms = {
        "invariance": invariance,
        "variance_context": ref_variance_hinge(z_x, gamma, eps),
        "variance_plan": ref_variance_hinge(z_hat, gamma, eps),
        "covariance_context": ref_covariance_penalty(z_x),
        "covariance_plan": ref_covariance_penalty(z_hat),
    }
    total = (lambda_invar * terms["invariance"]
             + lambda_v * (terms["variance_context"] + terms["variance_plan"])
             + lambda_c * (terms["covariance_context"] + terms["covariance_plan"]))
    return total, terms


def ref_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100,
                      label_smoothing: float = 0.0) -> float:
    """Mean over samples with a reference mode of -log softmax(l)_u, with optional uniform label smoothing."""
    per_sample = []
    for row, label in zip(py(logits), targets.tolist(), strict=True):
        if label == ignore_index:
            continue
        top = max(row)
        log_normaliser = top + math.log(sum(math.exp(value - top) for value in row))
        nll = log_normaliser - row[label]
        uniform = sum(log_normaliser - value for value in row) / len(row)
        per_sample.append((1.0 - label_smoothing) * nll + label_smoothing * uniform)
    return sum(per_sample) / len(per_sample)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
def correlated_rows(n: int, d: int, seed: int, dtype: torch.dtype = F32) -> torch.Tensor:
    """Rows with correlated columns of different spreads and a non-zero mean."""
    g = gen(seed)
    x = torch.randn(n, d, generator=g, dtype=F64)
    mixing = torch.eye(d, dtype=F64) + 0.4 * torch.randn(d, d, generator=g, dtype=F64)
    spread = torch.linspace(0.2, 2.0, d, dtype=F64)
    shift = torch.randn(d, generator=g, dtype=F64)
    return ((x @ mixing) * spread + shift).to(dtype)


def subspace_rows(n: int, d: int, rank: int, seed: int) -> torch.Tensor:
    """Rows confined to an affine subspace of dimension ``rank``."""
    g = gen(seed)
    return torch.randn(n, rank, generator=g) @ torch.randn(rank, d, generator=g) + torch.randn(d, generator=g)


def plan_batch(seed: int, *, n_ens: int = 3, batch: int = 2, slots: int = 5, context_len: int = 4, d: int = 4,
               dtype: torch.dtype = F32) -> dict[str, torch.Tensor]:
    """VICRegLoss inputs: per-head predictions, target s_y, context s_x and plan Z_target (the head mean, Eq. 8)."""
    g = gen(seed)
    spread = torch.linspace(0.3, 2.5, d, dtype=F64)
    target = torch.randn(batch, slots, d, generator=g, dtype=F64) * spread
    predictions = target + 0.3 * torch.randn(n_ens, batch, slots, d, generator=g, dtype=F64)
    mixing = torch.eye(d, dtype=F64) + 0.5 * torch.randn(d, d, generator=g, dtype=F64)
    context = (torch.randn(batch, context_len, d, generator=g, dtype=F64) @ mixing) * spread
    return {"predictions": predictions.to(dtype), "target": target.to(dtype), "context": context.to(dtype),
            "plan": predictions.mean(0).to(dtype)}


def overwrite_padding(x: torch.Tensor, mask: torch.Tensor, seed: int) -> torch.Tensor:
    """Replace the padded rows of ``x [..., B, L, d]`` by large random values."""
    garbage = 100.0 * torch.randn(x.shape, generator=gen(seed), dtype=x.dtype)
    return torch.where(mask[..., None], x, garbage)


# ---------------------------------------------------------------------------------------------
# Eqs. 6 and 7
# ---------------------------------------------------------------------------------------------
@DTYPES
def test_sample_covariance_matches_eq7(dtype):
    """C(Z) = (1/(n-1)) sum_i (z_i - zbar)(z_i - zbar)^T (Eq. 7, Bessel-corrected)."""
    z = correlated_rows(11, 5, seed=1, dtype=dtype)
    cov = sample_covariance(z)
    assert cov.shape == (5, 5) and cov.dtype == dtype
    assert_matches(cov, ref_covariance(py(z)), TOL[dtype])


@DTYPES
@pytest.mark.parametrize(("gamma", "eps"), [(1.0, 1e-4), (1.3, 0.05)])
def test_variance_hinge_matches_eq6(dtype, gamma, eps):
    """v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(z^j) + eps)) with Var over n - 1 (Eq. 6)."""
    z = correlated_rows(12, 6, seed=2, dtype=dtype)
    rows = py(z)
    hinges = [gamma - math.sqrt(ref_column_variance(rows, j) + eps) for j in range(6)]
    assert min(hinges) < 0.0 < max(hinges), "the data must contain active and inactive hinges"
    assert_matches(variance_hinge(z, gamma=gamma, eps=eps), ref_variance_hinge(rows, gamma, eps), TOL[dtype])


@DTYPES
def test_covariance_penalty_matches_eq7(dtype):
    """c(Z) = (1/d) sum_{j != k} C(Z)_jk^2: off-diagonal mass only, divided by d (Eq. 7)."""
    z = correlated_rows(12, 6, seed=3, dtype=dtype)
    assert_matches(covariance_penalty(z), ref_covariance_penalty(py(z)), TOL[dtype])


def test_eq6_eq7_and_stable_rank_treat_leading_dims_as_separate_batches():
    """Inputs [..., n, d]: every slice is its own batch Z (Eqs. 6, 7 and Prop. 2 per slice)."""
    z = torch.stack([torch.stack([correlated_rows(7, 4, seed=10 * a + b) for b in range(3)]) for a in range(2)])
    v, c, cov = variance_hinge(z), covariance_penalty(z), sample_covariance(z)
    srank = stable_rank(cov)
    assert v.shape == c.shape == srank.shape == (2, 3) and cov.shape == (2, 3, 4, 4)
    for a in range(2):
        for b in range(3):
            rows = py(z[a, b])
            assert_matches(v[a, b], ref_variance_hinge(rows, 1.0, 1e-4))
            assert_matches(c[a, b], ref_covariance_penalty(rows))
            assert_matches(cov[a, b], ref_covariance(rows))
            assert_matches(srank[a, b], ref_stable_rank(ref_covariance(rows)))


@pytest.mark.parametrize("fn", [sample_covariance, variance_hinge, covariance_penalty, calculate_stable_rank],
                         ids=lambda fn: fn.__name__)
@pytest.mark.parametrize("shape", [(1, 4), (3, 1, 4), (4,)], ids=["n=1", "batched-n=1", "no-row-axis"])
def test_estimators_reject_inputs_without_two_rows(fn, shape):
    """Var and C divide by n - 1 (Eqs. 6, 7): fewer than two rows raise ValueError."""
    with pytest.raises(ValueError):
        fn(torch.randn(shape))


@DTYPES
@pytest.mark.parametrize(("gamma", "eps"), [(1.0, 1e-4), (1.5, 0.04)])
def test_batch_collapsed_to_a_point(dtype, gamma, eps):
    """Identical rows: Var = 0, so v = gamma - sqrt(eps) (Eq. 6); C = 0, so c = 0 (Eq. 7); srank 0 by convention."""
    # Dyadic entries and n = 8 make the mean exact, so C is exactly zero in both dtypes.
    z = torch.tensor([0.5, -1.25, 3.0, 2.0], dtype=dtype).expand(8, 4).clone()
    assert_matches(variance_hinge(z, gamma=gamma, eps=eps), gamma - math.sqrt(eps), TOL[dtype])
    assert torch.count_nonzero(sample_covariance(z)) == 0
    assert covariance_penalty(z).item() == 0.0
    assert stable_rank(sample_covariance(z)).item() == 0.0
    assert calculate_stable_rank(z).item() == 0.0


@DTYPES
def test_batch_on_a_line_has_stable_rank_one(dtype):
    """Rows z_i = p + a_i u: C = Var(a) u u^T has rank one and (tr C)^2 / ||C||_F^2 = 1 (Prop. 2)."""
    # Dyadic a_i, p and u: the rows lie exactly on the line in both dtypes.
    a = torch.tensor([-1.5, -1.0, -0.5, -0.25, 0.25, 0.75, 1.25, 2.0], dtype=F64)[:, None]
    p = torch.tensor([1.0, -2.0, 0.5, 3.0, 0.0], dtype=F64)
    u = torch.tensor([0.25, -1.25, 2.0, 0.75, -0.5], dtype=F64)
    z = (p + a * u).to(dtype)
    assert int(torch.linalg.matrix_rank(torch.tensor(ref_covariance(py(z)), dtype=F64))) == 1
    assert_matches(calculate_stable_rank(z), 1.0, TOL[dtype])


@DTYPES
@pytest.mark.parametrize(("n", "d"), [(12, 5), (6, 6), (4, 6)], ids=["n>d", "n=d", "n<d"])
def test_prop2_stable_rank_identity_and_rank_bound(dtype, n, d):
    """srank(C) = (sum_j C_jj)^2 / (sum_j C_jj^2 + d c(Z)) and 1 <= srank(C) <= rank(C) (Eq. 25)."""
    z = correlated_rows(n, d, seed=5 + n, dtype=dtype)
    cov = ref_covariance(py(z))
    diagonal = [cov[j][j] for j in range(d)]
    eq25 = sum(diagonal) ** 2 / (sum(x * x for x in diagonal) + d * ref_off_diagonal_mass(cov))
    rank = int(torch.linalg.matrix_rank(torch.tensor(cov, dtype=F64)))
    assert rank == min(n - 1, d)
    srank = stable_rank(sample_covariance(z))
    assert_matches(srank, eq25, TOL[dtype])
    assert_matches(calculate_stable_rank(z), eq25, TOL[dtype])
    assert 1.0 <= float(srank) <= rank * (1.0 + RTOL)


def test_prop2_equal_variances_give_d_over_one_plus_c_over_s4():
    """If every Var(z^j) = s^2, srank(C) = d / (1 + c(Z)/s^4) (Prop. 2, second statement, attained)."""
    s2, d = 2.25, 5
    x = correlated_rows(15, d, seed=6, dtype=F64)
    z = (x - x.mean(0)) / x.std(0, correction=1) * math.sqrt(s2)
    rows = py(z)
    assert all(abs(ref_column_variance(rows, j) - s2) < 1e-12 for j in range(d))
    assert_matches(calculate_stable_rank(z), d / (1.0 + ref_covariance_penalty(rows) / s2 ** 2), TIGHT)


@pytest.mark.parametrize("shape", [(3, 4), (4,)])
def test_stable_rank_rejects_non_square_input(shape):
    with pytest.raises(ValueError):
        stable_rank(torch.ones(shape))


def test_prop3_variance_hinge_gradient_matches_eq26():
    """Active hinge: d/dz_i^j [gamma - sqrt(Var + eps)] = -(z_i^j - zbar^j) / ((n-1) sqrt(Var + eps)), of norm
    sqrt(Var / ((n-1)(Var + eps))) (Eq. 26); v(Z) adds the factor 1/d; an inactive hinge has zero gradient."""
    n, gamma, eps = 9, 1.0, 1e-4
    variances = [1e-2, 0.25, 4.0]                      # active, active, inactive for gamma = 1
    d = len(variances)
    x = torch.randn(n, d, generator=gen(7), dtype=F64)
    z = ((x - x.mean(0)) / x.std(0, correction=1) * torch.tensor(variances, dtype=F64).sqrt()).requires_grad_(True)
    variance_hinge(z, gamma=gamma, eps=eps).backward()
    rows = py(z)
    for j in range(d):
        var = ref_column_variance(rows, j)
        mean = sum(row[j] for row in rows) / n
        active = gamma - math.sqrt(var + eps) > 0.0
        assert active == (variances[j] < 1.0)
        expected = [-(row[j] - mean) / ((n - 1) * math.sqrt(var + eps)) / d if active else 0.0 for row in rows]
        expected_norm = math.sqrt(var / ((n - 1) * (var + eps))) / d if active else 0.0
        assert_matches(z.grad[:, j], expected, TIGHT)
        assert_matches(z.grad[:, j].norm(), expected_norm, TIGHT)


# ---------------------------------------------------------------------------------------------
# calculate_stable_rank and the d/2 guard (Sec. 5.1)
# ---------------------------------------------------------------------------------------------
def test_calculate_stable_rank_returns_detached_srank():
    z = correlated_rows(10, 4, seed=8, dtype=F64).requires_grad_(True)
    srank = calculate_stable_rank(z)
    assert not srank.requires_grad
    assert_matches(srank, ref_stable_rank(ref_covariance(py(z))), TIGHT)


@pytest.mark.parametrize("case", ["point", "line", "plane", "one-slice-of-a-batch"])
def test_calculate_stable_rank_raises_collapse_below_min_fraction(case):
    """srank(C) < min_fraction * d raises RepresentationCollapseError (Sec. 5.1: d/2); here d = 8, threshold 4."""
    n, d = 40, 8
    z = {
        "point": torch.ones(n, d),
        "line": subspace_rows(n, d, 1, seed=9),
        "plane": subspace_rows(n, d, 2, seed=9),
        "one-slice-of-a-batch": torch.stack([torch.randn(n, d, generator=gen(9)), subspace_rows(n, d, 2, seed=9)]),
    }[case]
    sranks = [ref_stable_rank(ref_covariance(py(rows))) for rows in z.reshape(-1, n, d)]
    assert min(sranks) < 0.5 * d
    with pytest.raises(RepresentationCollapseError):
        calculate_stable_rank(z, min_fraction=0.5)


@pytest.mark.parametrize(("d", "min_fraction"), [(6, 0.5), (6, 1.0)])
def test_calculate_stable_rank_needs_enough_rows_to_certify(d, min_fraction):
    """rank(C) <= n - 1, so with n - 1 < min_fraction * d the check is not meaningful: ValueError, not a collapse
    alarm. One more row makes it meaningful, and so few random rows then fail it (srank < rank = n - 1)."""
    n_short = math.ceil(min_fraction * d)
    with pytest.raises(ValueError):
        calculate_stable_rank(torch.randn(n_short, d, generator=gen(10)), min_fraction=min_fraction)
    with pytest.raises(RepresentationCollapseError):
        calculate_stable_rank(torch.randn(n_short + 1, d, generator=gen(10)), min_fraction=min_fraction)


def test_calculate_stable_rank_passes_healthy_isotropic_rows():
    """Isotropic Gaussian rows with n >> d: srank close to d, so the d/2 guard passes and returns srank."""
    d = 6
    z = torch.randn(600, d, generator=gen(11))
    srank = calculate_stable_rank(z, min_fraction=0.5)
    assert_matches(srank, ref_stable_rank(ref_covariance(py(z))))
    assert 0.9 * d < float(srank) <= d


@pytest.mark.parametrize("min_fraction", [0.0, -0.5, 1.5])
def test_calculate_stable_rank_rejects_min_fraction_outside_unit_interval(min_fraction):
    with pytest.raises(ValueError):
        calculate_stable_rank(torch.randn(50, 4, generator=gen(12)), min_fraction=min_fraction)


# ---------------------------------------------------------------------------------------------
# Theorem 1
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("gamma", "eps"), [(1.0, 1e-4), (1.5, 0.01)])
def test_theorem1_bound_matches_eq24(gamma, eps):
    """theorem1_lambda_min_bound(C) = gamma^2 - eps - sqrt(d(d-1) c) (Eq. 24), for stacked covariances."""
    near_diagonal = [[1.2, 0.02, 0.0, 0.01, 0.0], [0.02, 1.5, 0.0, 0.0, 0.05], [0.0, 0.0, 1.1, -0.03, 0.0],
                     [0.01, 0.0, -0.03, 2.0, 0.0], [0.0, 0.05, 0.0, 0.0, 1.7]]
    correlated = ref_covariance(py(correlated_rows(20, 5, seed=13, dtype=F64)))
    covs = torch.tensor([near_diagonal, correlated], dtype=F64)
    bounds = theorem1_lambda_min_bound(covs, gamma=gamma, eps=eps)
    assert bounds.shape == (2,)
    assert_matches(bounds, [ref_theorem1_bound(near_diagonal, gamma, eps), ref_theorem1_bound(correlated, gamma, eps)],
                   TIGHT)


def test_theorem1_holds_for_rows_meeting_its_premise():
    """Rows with v(Z) = 0 and c(Z) < (gamma^2 - eps)^2 / (d(d-1)): lambda_min(C) >= the Eq. 24 bound > 0."""
    n = 50
    target = torch.tensor([[1.2, 0.02, 0.0, 0.01], [0.02, 1.5, 0.0, 0.0], [0.0, 0.0, 1.1, -0.03],
                           [0.01, 0.0, -0.03, 2.0]], dtype=F64)
    # Rows whose sample covariance is exactly `target`: whiten Gaussian rows, then colour them with chol(target).
    x = torch.randn(n, 4, generator=gen(14), dtype=F64)
    x = x - x.mean(0)
    white = x @ torch.linalg.inv(torch.linalg.cholesky(x.T @ x / (n - 1))).T
    z = white @ torch.linalg.cholesky(target).T + 5.0
    cov = ref_covariance(py(z))
    expected = ref_theorem1_bound(cov, 1.0, 1e-4)
    assert ref_variance_hinge(py(z), 1.0, 1e-4) == 0.0 and expected > 0.0          # the premise of Theorem 1
    bound = theorem1_lambda_min_bound(sample_covariance(z))
    assert_matches(bound, expected, TIGHT)
    assert float(torch.linalg.eigvalsh(torch.tensor(cov, dtype=F64))[0]) >= float(bound)


def test_theorem1_bound_never_exceeds_lambda_min_on_random_covariances():
    """Appendix B check (2) at d = 6: symmetric C with diagonal in [1, 2] (so v = 0 for gamma = 1) and random
    off-diagonal mass. The library bound equals gamma^2 - eps - sqrt(d(d-1) c) evaluated naively, and whenever that
    bound is positive (the premise) it lies below lambda_min(C) (Eq. 24)."""
    trials, d, gamma, eps = 2000, 6, 1.0, 1e-4
    g = gen(15)
    diagonal = 1.0 + torch.rand(trials, d, generator=g, dtype=F64)
    noise = torch.randn(trials, d, d, generator=g, dtype=F64)
    noise = (noise + noise.mT) / 2
    noise = noise * (1.0 - torch.eye(d, dtype=F64))                                  # zero diagonal
    covs = torch.diag_embed(diagonal) + noise * (0.4 * torch.rand(trials, 1, 1, generator=g, dtype=F64))
    c = (covs * (1.0 - torch.eye(d, dtype=F64))).pow(2).sum(dim=(-2, -1)) / d       # Eq. 7 from C
    expected = gamma ** 2 - eps - torch.sqrt(d * (d - 1) * c)
    premise = expected > 0
    assert int(premise.sum()) >= 100 and int((~premise).sum()) >= 100
    bounds = theorem1_lambda_min_bound(covs, gamma=gamma, eps=eps)
    assert_matches(bounds, expected, TIGHT)
    lambda_min = torch.linalg.eigvalsh(covs)[:, 0]
    assert torch.all(lambda_min[premise] >= bounds[premise])


# ---------------------------------------------------------------------------------------------
# masked_rows
# ---------------------------------------------------------------------------------------------
def test_masked_rows_stacks_real_rows_in_row_major_order():
    x = torch.arange(12, dtype=F32).reshape(2, 3, 2)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    assert masked_rows(x, mask).tolist() == [x[b, m].tolist() for b in range(2) for m in range(3) if mask[b, m]]
    assert masked_rows(x, None).tolist() == [x[b, m].tolist() for b in range(2) for m in range(3)]


@pytest.mark.parametrize(("x_shape", "mask"), [
    ((2, 3, 2), torch.ones(2, 3, dtype=torch.int64)),
    ((2, 3, 2), torch.ones(3, 2, dtype=torch.bool)),
    ((6, 2), None),
], ids=["integer-mask", "wrong-mask-shape", "not-3d"])
def test_masked_rows_rejects_malformed_inputs(x_shape, mask):
    with pytest.raises(ValueError):
        masked_rows(torch.zeros(x_shape), mask)


# ---------------------------------------------------------------------------------------------
# VICRegLoss: the prediction (invariance) term of Eq. 5
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("head_reduction", ["mean", "sum"])
def test_invariance_is_eq5_per_head_then_reduced_over_heads(head_reduction):
    """Per head: squared error summed over d, averaged over the B*M slots (Eq. 5); then mean or sum over heads."""
    batch = plan_batch(20)
    value = VICRegLoss(normalization="frobenius", head_reduction=head_reduction).invariance(batch["predictions"], batch["target"])
    assert_matches(value, ref_invariance(batch["predictions"], batch["target"], None, head_reduction))


@pytest.mark.parametrize("head_reduction", ["mean", "sum"])
def test_invariance_with_slot_mask_averages_over_real_slots(head_reduction):
    """With a slot mask, BM in Eq. 5 is the number of real slots and padded slots contribute nothing."""
    batch = plan_batch(21)
    value = VICRegLoss(normalization="frobenius", head_reduction=head_reduction).invariance(batch["predictions"], batch["target"], SLOT_MASK)
    assert_matches(value, ref_invariance(batch["predictions"], batch["target"], SLOT_MASK, head_reduction))


@pytest.mark.parametrize("head_reduction", ["mean", "sum"])
def test_invariance_of_a_single_predictor_is_the_one_head_case(head_reduction):
    """A [B, M, d] prediction is one head: identical to [1, B, M, d] and to Eq. 5's prediction term."""
    batch = plan_batch(22)
    loss = VICRegLoss(normalization="frobenius", head_reduction=head_reduction)
    value = loss.invariance(batch["plan"], batch["target"])
    assert torch.equal(value, loss.invariance(batch["plan"].unsqueeze(0), batch["target"]))
    assert_matches(value, ref_invariance(batch["plan"].unsqueeze(0), batch["target"], None, "mean"))


def test_head_mean_of_invariance_is_eq5_on_the_plan_plus_d_sigma_bar_squared():
    """Bias-variance identity tying the per-head reduction to Eqs. 5 and 8:
    (1/n_ens) sum_e (1/(BM)) sum_b ||s_hat^(e) - s_y||_F^2 = (1/(BM)) sum_b ||Z_target - s_y||_F^2 + d sigma_bar^2,
    with Z_target the head mean and sigma_bar^2 = (1/(n_ens B M d)) sum_{e,b} ||s_hat^(e) - Z_target||_F^2."""
    batch = plan_batch(23, dtype=F64)
    heads, target = py(batch["predictions"]), py(batch["target"])
    n_ens, b_size, m_size, d = batch["predictions"].shape
    cells = [(b, m, j) for b in range(b_size) for m in range(m_size) for j in range(d)]
    plan = {cell: sum(heads[e][cell[0]][cell[1]][cell[2]] for e in range(n_ens)) / n_ens for cell in cells}
    eq5_on_plan = sum((plan[(b, m, j)] - target[b][m][j]) ** 2 for b, m, j in cells) / (b_size * m_size)
    sigma_bar_sq = sum((heads[e][b][m][j] - plan[(b, m, j)]) ** 2
                       for e in range(n_ens) for b, m, j in cells) / (n_ens * b_size * m_size * d)
    value = VICRegLoss(normalization="frobenius", head_reduction="mean").invariance(batch["predictions"], batch["target"])
    assert_matches(value, eq5_on_plan + d * sigma_bar_sq, TIGHT)


@pytest.mark.parametrize("head_reduction", ["mean", "sum"])
def test_invariance_stops_the_gradient_to_the_target(head_reduction):
    """sg(s_y) in Eq. 5: the target gets no gradient; head e gets 2 (s_hat^(e) - s_y) / (BM), divided by n_ens for
    the head mean."""
    batch = plan_batch(24, dtype=F64)
    predictions = batch["predictions"].clone().requires_grad_(True)
    target = batch["target"].clone().requires_grad_(True)
    VICRegLoss(normalization="frobenius", head_reduction=head_reduction).invariance(predictions, target).backward()
    assert target.grad is None
    n_ens, b_size, m_size, _ = predictions.shape
    scale = 2.0 / (b_size * m_size) / (n_ens if head_reduction == "mean" else 1)
    assert_matches(predictions.grad, scale * (batch["predictions"] - batch["target"]), TIGHT)


# ---------------------------------------------------------------------------------------------
# VICRegLoss.forward: Eq. 5 without the mode term
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("head_reduction", ["mean", "sum"])
def test_vicreg_total_matches_hand_computed_eq5(head_reduction):
    """total = lambda_invar inv + lambda_v [v(Z_x) + v(Z_hat)] + lambda_c [c(Z_x) + c(Z_hat)], with Z_x the stacked
    context rows and Z_hat the stacked plan rows (Eqs. 5-7)."""
    batch = plan_batch(25)
    out = VICRegLoss(**WEIGHTS, head_reduction=head_reduction)(**batch)
    expected_total, expected_terms = ref_vicreg(batch, None, None, head_reduction=head_reduction, **WEIGHTS)
    assert all(value > 0.0 for value in expected_terms.values()), "every term must be active"
    assert_matches(out.total, expected_total)
    for name, value in expected_terms.items():
        assert_matches(out.terms[name], value)


def test_vicreg_with_masks_uses_only_real_rows():
    """With masks, Z_x and Z_hat stack only the real context tokens and slots, and BM counts the real slots."""
    batch = plan_batch(26)
    out = VICRegLoss(**WEIGHTS)(**batch, context_mask=CONTEXT_MASK, slot_mask=SLOT_MASK)
    expected_total, expected_terms = ref_vicreg(batch, CONTEXT_MASK, SLOT_MASK, **WEIGHTS)
    assert_matches(out.total, expected_total)
    for name, value in expected_terms.items():
        assert_matches(out.terms[name], value)


def test_vicreg_padding_content_does_not_change_the_loss():
    """Rows where context_mask or slot_mask is False are not part of Eq. 5: overwriting them changes nothing."""
    batch = plan_batch(27)
    loss = VICRegLoss(**WEIGHTS)
    base = loss(**batch, context_mask=CONTEXT_MASK, slot_mask=SLOT_MASK)
    padded = {
        "predictions": overwrite_padding(batch["predictions"], SLOT_MASK, seed=1),
        "target": overwrite_padding(batch["target"], SLOT_MASK, seed=2),
        "plan": overwrite_padding(batch["plan"], SLOT_MASK, seed=3),
        "context": overwrite_padding(batch["context"], CONTEXT_MASK, seed=4),
    }
    assert all(not torch.equal(padded[name], batch[name]) for name in padded)
    out = loss(**padded, context_mask=CONTEXT_MASK, slot_mask=SLOT_MASK)
    assert torch.equal(out.total, base.total)
    for name, value in base.terms.items():
        assert torch.equal(out.terms[name], value)


def test_vicreg_non_finite_padding_does_not_change_the_loss():
    """Padded slots lie outside the sum of Eq. 5 whatever they hold, NaN included (cf. the finite case above)."""
    batch = plan_batch(27)
    loss = VICRegLoss(**WEIGHTS)
    base = loss(**batch, context_mask=CONTEXT_MASK, slot_mask=SLOT_MASK)
    target = batch["target"].masked_fill(~SLOT_MASK[..., None], float("nan"))
    out = loss(**{**batch, "target": target}, context_mask=CONTEXT_MASK, slot_mask=SLOT_MASK)
    assert torch.equal(out.total, base.total)


def test_vicreg_gradients_reach_real_rows_only_and_never_the_target():
    """Padded positions get exactly zero gradient, real ones a non-zero one; sg(s_y) leaves the target without one."""
    batch = {name: value.clone().requires_grad_(True) for name, value in plan_batch(28).items()}
    VICRegLoss(**WEIGHTS)(**batch, context_mask=CONTEXT_MASK, slot_mask=SLOT_MASK).total.backward()
    assert batch["target"].grad is None
    for name, mask in (("predictions", SLOT_MASK), ("plan", SLOT_MASK), ("context", CONTEXT_MASK)):
        grad = batch[name].grad
        real = mask[..., None].expand_as(grad)
        assert torch.count_nonzero(grad[~real]) == 0
        assert torch.count_nonzero(grad[real]) > 0


def test_vicreg_terms_are_detached():
    batch = {name: value.clone().requires_grad_(True) for name, value in plan_batch(29).items()}
    out = VICRegLoss(normalization="frobenius")(**batch)
    assert out.total.requires_grad
    assert set(out.terms) == {"invariance", "variance_context", "variance_plan", "covariance_context",
                              "covariance_plan"}
    assert all(not term.requires_grad and term.grad_fn is None for term in out.terms.values())


@pytest.mark.parametrize("kwargs", [
    pytest.param({"lambda_v": 0.0}, id="lambda_v=0"),
    pytest.param({"lambda_v": -1.0}, id="lambda_v<0"),
    pytest.param({"lambda_c": -0.1}, id="lambda_c<0"),
    pytest.param({"lambda_invar": 0.0}, id="lambda_invar=0"),
    pytest.param({"gamma": 0.0}, id="gamma=0"),
    pytest.param({"eps": 0.0}, id="eps=0"),
    pytest.param({"gamma": 1.0, "eps": 1.0}, id="eps=gamma^2"),
    pytest.param({"gamma": 0.5, "eps": 0.4}, id="gamma^2<eps<gamma"),
    pytest.param({"head_reduction": "max"}, id="unknown-head-reduction"),
])
def test_vicreg_rejects_invalid_hyperparameters(kwargs):
    """lambda_v > 0 (Theorem 1), lambda_c >= 0, 0 < eps < gamma^2 (Eq. 6, Prop. 3), head_reduction in {mean, sum}."""
    with pytest.raises(ValueError):
        VICRegLoss(**kwargs)


@pytest.mark.parametrize("kwargs", [{"lambda_c": 0.0}, {"gamma": 0.5, "eps": 0.24}], ids=["lambda_c=0", "eps<gamma^2"])
def test_vicreg_accepts_boundary_hyperparameters(kwargs):
    """lambda_c = 0 is ablation A6 (Table 8); any eps strictly below gamma^2 is admissible."""
    loss = VICRegLoss(**kwargs)
    batch = plan_batch(30)
    assert torch.isfinite(loss(**batch).total)


@pytest.mark.parametrize("change", ["plan-shape", "predictions-shape", "integer-slot-mask"])
def test_vicreg_rejects_inconsistent_shapes(change):
    batch = plan_batch(31)
    slot_mask = None
    if change == "plan-shape":
        batch["plan"] = batch["plan"][:, :-1]
    elif change == "predictions-shape":
        batch["predictions"] = batch["predictions"][..., :-1, :]
    else:
        slot_mask = SLOT_MASK.long()
    with pytest.raises(ValueError):
        VICRegLoss(normalization="frobenius")(**batch, slot_mask=slot_mask)


# ---------------------------------------------------------------------------------------------
# L_CE and L_plan
# ---------------------------------------------------------------------------------------------
def test_mode_cross_entropy_is_mean_negative_log_softmax():
    """L_CE = mean_b [log sum_k exp(l_bk) - l_{b, u_b}] (the selector term of Eq. 5)."""
    logits = 2.0 * torch.randn(6, 5, generator=gen(32))
    modes = torch.tensor([0, 4, 2, 2, 1, 3])
    assert_matches(ModeCrossEntropyLoss()(logits, modes), ref_cross_entropy(logits, modes))


@pytest.mark.parametrize("ignore_index", [-100, -1])
def test_mode_cross_entropy_excludes_samples_without_a_reference_mode(ignore_index):
    """Samples whose target is ignore_index leave both the sum and the count of the mean."""
    logits = 2.0 * torch.randn(6, 5, generator=gen(33))
    modes = torch.tensor([0, ignore_index, 2, ignore_index, 1, 3])
    value = ModeCrossEntropyLoss(ignore_index=ignore_index)(logits, modes)
    assert_matches(value, ref_cross_entropy(logits, modes, ignore_index=ignore_index))


def test_mode_cross_entropy_label_smoothing():
    """Smoothed target (1 - a) e_u + a/|U|: loss = (1 - a)(-log p_u) + (a/|U|) sum_k (-log p_k), mean over samples."""
    logits = 2.0 * torch.randn(5, 4, generator=gen(34))
    modes = torch.tensor([1, -100, 3, 0, 2])
    value = ModeCrossEntropyLoss(label_smoothing=0.1)(logits, modes)
    assert_matches(value, ref_cross_entropy(logits, modes, label_smoothing=0.1))


@pytest.mark.parametrize(("logits", "modes"), [
    (torch.zeros(3, 4), torch.zeros(3)),
    (torch.zeros(3, 4, 2), torch.zeros(3, dtype=torch.int64)),
    (torch.zeros(3, 4), torch.zeros(2, dtype=torch.int64)),
], ids=["float-targets", "3d-logits", "batch-mismatch"])
def test_mode_cross_entropy_rejects_malformed_inputs(logits, modes):
    with pytest.raises(ValueError):
        ModeCrossEntropyLoss()(logits, modes)


@pytest.mark.parametrize("label_smoothing", [-0.1, 1.0])
def test_mode_cross_entropy_rejects_label_smoothing_outside_unit_interval(label_smoothing):
    with pytest.raises(ValueError):
        ModeCrossEntropyLoss(label_smoothing=label_smoothing)


def test_plan_loss_is_vicreg_plus_lambda_u_times_cross_entropy():
    """L_plan = (JEPA part of Eq. 5) + lambda_u L_CE, every part recomputed by hand; logged terms are detached."""
    batch = {name: value.clone().requires_grad_(True) for name, value in plan_batch(35).items()}
    logits = torch.randn(2, 4, generator=gen(36)).requires_grad_(True)
    modes = torch.tensor([3, 1])
    lambda_u = 0.7
    out = PlanLoss(VICRegLoss(**WEIGHTS), lambda_u=lambda_u)(
        **batch, mode_logits=logits, mode_targets=modes, context_mask=CONTEXT_MASK, slot_mask=SLOT_MASK)
    jepa_total, jepa_terms = ref_vicreg(batch, CONTEXT_MASK, SLOT_MASK, **WEIGHTS)
    ce = ref_cross_entropy(logits, modes)
    assert_matches(out.total, jepa_total + lambda_u * ce)
    assert_matches(out.terms["mode_ce"], ce)
    assert_matches(out.terms["total"], jepa_total + lambda_u * ce)
    for name, value in jepa_terms.items():
        assert_matches(out.terms[name], value)
    assert out.total.requires_grad
    assert all(not term.requires_grad for term in out.terms.values())


def test_plan_loss_from_config_uses_every_weight():
    """PlanLoss.from_config wires lambda_invar, lambda_v, lambda_c, lambda_u, gamma, eps and head_reduction."""
    cfg = VICRegConfig(lambda_invar=2.0, lambda_v=3.0, lambda_c=0.5, lambda_u=0.25, gamma=1.2, eps=1e-3,
                       head_reduction="sum")
    batch = plan_batch(37)
    logits = torch.randn(2, 4, generator=gen(38))
    modes = torch.tensor([0, 2])
    out = PlanLoss.from_config(cfg)(**batch, mode_logits=logits, mode_targets=modes)
    # VICRegConfig's default invariance_normalization is per_dim (VICReg's per-element MSE).
    jepa_total, _ = ref_vicreg(batch, None, None, head_reduction="sum", **{**WEIGHTS, "normalization": "per_dim"})
    assert_matches(out.total, jepa_total + 0.25 * ref_cross_entropy(logits, modes))


@pytest.mark.parametrize("head_reduction", ["mean", "sum"])
def test_per_dim_invariance_is_frobenius_divided_by_d(head_reduction):
    """normalization="per_dim" (the default) is Eq. 5's prediction term divided by d: VICReg's per-element MSE."""
    batch = plan_batch(91)
    d = batch["target"].shape[-1]
    frob = VICRegLoss(normalization="frobenius", head_reduction=head_reduction).invariance(
        batch["predictions"], batch["target"], SLOT_MASK)
    per_dim = VICRegLoss(head_reduction=head_reduction).invariance(batch["predictions"], batch["target"], SLOT_MASK)
    assert_matches(per_dim, ref_invariance(batch["predictions"], batch["target"], SLOT_MASK, head_reduction) / d)
    torch.testing.assert_close(per_dim * d, frob, rtol=1e-6, atol=1e-7)


def test_invalid_invariance_normalization_raises():
    with pytest.raises(ValueError):
        VICRegLoss(normalization="sum_over_slots")


def test_plan_loss_rejects_negative_lambda_u():
    with pytest.raises(ValueError):
        PlanLoss(VICRegLoss(), lambda_u=-0.1)


# ---------------------------------------------------------------------------------------------
# covariance_diagnostics
# ---------------------------------------------------------------------------------------------
def test_covariance_diagnostics_match_float64_reference():
    """v (Eq. 6), c (Eq. 7) and srank (Eq. 25) from C; eigenvalues ascending, float64, CPU; lambda_min is the
    smallest eigenvalue of the float64 reference covariance."""
    gamma, eps, d, n = 1.2, 1e-3, 5, 30
    x = torch.randn(n, d, generator=gen(39), dtype=F64)
    z = (x @ (torch.eye(d, dtype=F64) + 0.3 * torch.randn(d, d, generator=gen(40), dtype=F64))).to(F32)
    cov = ref_covariance(py(z))
    diag = covariance_diagnostics(torch.tensor(cov, dtype=F32), num_rows=n, gamma=gamma, eps=eps)
    reference_eigenvalues = torch.linalg.eigvalsh(torch.tensor(cov, dtype=F64))
    assert (diag.num_rows, diag.dim) == (n, d)
    assert diag.eigenvalues.dtype == F64 and diag.eigenvalues.device.type == "cpu"
    assert diag.eigenvalues.shape == (d,) and torch.all(diag.eigenvalues[1:] >= diag.eigenvalues[:-1])
    # float32 rounding of C perturbs each eigenvalue by at most ||delta C||_2 ~ 1e-7 here.
    assert_matches(diag.eigenvalues, reference_eigenvalues)
    assert_matches(diag.lambda_min, reference_eigenvalues[0])
    rows_hinge = sum(max(0.0, gamma - math.sqrt(cov[j][j] + eps)) for j in range(d)) / d
    assert_matches(diag.variance_hinge, rows_hinge)
    assert_matches(diag.variance_hinge, ref_variance_hinge(py(z), gamma, eps))
    assert_matches(diag.covariance_penalty, ref_off_diagonal_mass(cov))
    assert_matches(diag.stable_rank, ref_stable_rank(cov))
    assert_matches(diag.srank_fraction, ref_stable_rank(cov) / d)


_PREMISE_HOLDS = [[1.2, 0.02, 0.0, 0.01], [0.02, 1.5, 0.0, 0.0], [0.0, 0.0, 1.1, -0.03], [0.01, 0.0, -0.03, 2.0]]
_HINGE_ACTIVE = [[1.2, 0.02, 0.0, 0.01], [0.02, 0.5, 0.0, 0.0], [0.0, 0.0, 1.1, -0.03], [0.01, 0.0, -0.03, 2.0]]
_BOUND_NEGATIVE = [[1.5 if j == k else 0.6 for k in range(4)] for j in range(4)]   # 0.9 I + 0.6 J, PSD


@pytest.mark.parametrize(("cov", "certified"), [
    pytest.param(_PREMISE_HOLDS, True, id="v=0-and-bound>0"),
    pytest.param(_HINGE_ACTIVE, False, id="v>0"),
    pytest.param(_BOUND_NEGATIVE, False, id="bound<=0"),
])
def test_covariance_diagnostics_certify_only_under_theorem1_premise(cov, certified):
    """certified_lambda_min is the Eq. 24 bound when v(Z) = 0 and the bound is positive, otherwise None; when set,
    it does not exceed lambda_min (Theorem 1)."""
    gamma, eps = 1.0, 1e-4
    hinge = sum(max(0.0, gamma - math.sqrt(cov[j][j] + eps)) for j in range(len(cov))) / len(cov)
    bound = ref_theorem1_bound(cov, gamma, eps)
    assert (hinge == 0.0 and bound > 0.0) == certified
    diag = covariance_diagnostics(torch.tensor(cov, dtype=F64), num_rows=100, gamma=gamma, eps=eps)
    if certified:
        assert_matches(diag.certified_lambda_min, bound, TIGHT)
        assert diag.certified_lambda_min <= diag.lambda_min
    else:
        assert diag.certified_lambda_min is None


def test_covariance_diagnostics_can_skip_eigenvalues():
    diag = covariance_diagnostics(torch.tensor(_PREMISE_HOLDS, dtype=F64), num_rows=100, compute_eigenvalues=False)
    assert diag.eigenvalues is None and diag.lambda_min is None
    assert_matches(diag.certified_lambda_min, ref_theorem1_bound(_PREMISE_HOLDS, 1.0, 1e-4), TIGHT)


# ---------------------------------------------------------------------------------------------
# StreamingCovariance
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("dtype", "offset", "tol"), [
    pytest.param(F64, 0.0, TIGHT, id="float64"),
    # A common offset of 1e4 costs ~4 of float64's 16 digits in any centring; an update that accumulates raw
    # second moments would lose ~8 and fail this tolerance.
    pytest.param(F64, 1e4, {"rtol": 1e-9, "atol": 1e-9}, id="float64-offset-1e4"),
    pytest.param(F32, 0.0, LOOSE, id="float32"),
])
def test_streaming_covariance_of_uneven_batches_equals_all_rows_at_once(dtype, offset, tol):
    """Merging batches of 3, 17, 1 and 40 rows gives the mean and C(Z) (Eq. 7) of all 61 rows stacked."""
    d = 5
    rows = (correlated_rows(61, d, seed=41, dtype=F64) + offset).to(dtype)
    acc = StreamingCovariance(d)
    for chunk in rows.split([3, 17, 1, 40]):
        acc.update(chunk)
    assert acc.count == 61
    reference = py(rows)
    assert_matches(acc.mean, ref_mean(reference), tol)
    cov = acc.covariance()
    assert cov.dtype == F64 and cov.device.type == "cpu"
    assert_matches(cov, ref_covariance(reference), tol)


def test_streaming_covariance_ignores_empty_batches():
    rows = correlated_rows(10, 3, seed=42, dtype=F64)
    acc = StreamingCovariance(3)
    acc.update(rows[:4])
    acc.update(rows[:0])
    acc.update(rows[4:])
    assert acc.count == 10
    assert_matches(acc.covariance(), ref_covariance(py(rows)), TIGHT)


def test_streaming_covariance_needs_two_rows():
    acc = StreamingCovariance(3)
    with pytest.raises(ValueError):
        acc.covariance()
    acc.update(torch.randn(1, 3, generator=gen(43)))
    assert acc.count == 1
    with pytest.raises(ValueError):
        acc.covariance()


@pytest.mark.parametrize("shape", [(4, 2), (4,), (2, 4, 3)])
def test_streaming_covariance_rejects_rows_of_the_wrong_shape(shape):
    with pytest.raises(ValueError):
        StreamingCovariance(3).update(torch.zeros(shape))


def test_streaming_covariance_reset_forgets_all_rows():
    acc = StreamingCovariance(3)
    acc.update(torch.randn(20, 3, generator=gen(44)))
    acc.reset()
    assert acc.count == 0
    with pytest.raises(ValueError):
        acc.covariance()
    fresh = correlated_rows(8, 3, seed=45, dtype=F64)
    acc.update(fresh)
    assert_matches(acc.covariance(), ref_covariance(py(fresh)), TIGHT)


# ---------------------------------------------------------------------------------------------
# StableRankMonitor: the Stage-1 guard (Sec. 5.1)
# ---------------------------------------------------------------------------------------------
def test_stable_rank_monitor_passes_healthy_streams():
    """Isotropic rows accumulated over a window: srank close to d, above d/2; the diagnostics are those of all the
    window's rows stacked."""
    d = 8
    monitor = StableRankMonitor(dim=d)
    streams = {"context": torch.randn(384, d, generator=gen(46)),
               "plan": 1.5 * torch.randn(384, d, generator=gen(47)) + 2.0}
    for name, rows in streams.items():
        for chunk in rows.split(64):
            monitor.update(name, chunk)
    report = monitor.check()
    assert set(report) == {"context", "plan"}
    for name, rows in streams.items():
        cov = ref_covariance(py(rows))
        diag = report[name]
        assert (diag.num_rows, diag.dim) == (384, d)
        assert_matches(diag.stable_rank, ref_stable_rank(cov))
        assert_matches(diag.lambda_min, torch.linalg.eigvalsh(torch.tensor(cov, dtype=F64))[0])
        assert 0.9 * d < diag.stable_rank <= d


def test_stable_rank_monitor_aborts_on_a_collapsed_stream():
    """Plan rows in a fixed 2-dim subspace of d = 16: srank <= 2 < d/2 = 8 raises RepresentationCollapseError."""
    d = 16
    g = gen(48)
    basis = torch.randn(2, d, generator=g)
    monitor = StableRankMonitor(dim=d)
    for _ in range(6):
        monitor.update("context", torch.randn(40, d, generator=g))
        monitor.update("plan", torch.randn(40, 2, generator=g) @ basis)
    with pytest.raises(RepresentationCollapseError, match="plan"):
        monitor.check()


def test_stable_rank_monitor_needs_more_than_d_rows():
    """With n <= d rows per stream, rank(C) <= n - 1 < d: ValueError; d + 1 rows are enough to summarise."""
    d = 6
    rows = torch.randn(d + 1, d, generator=gen(49))
    monitor = StableRankMonitor(dim=d)
    for name in ("context", "plan"):
        monitor.update(name, rows[:d])
    with pytest.raises(ValueError):
        monitor.summarize()
    with pytest.raises(ValueError):
        monitor.check()
    for name in ("context", "plan"):
        monitor.update(name, rows[d:])
    assert all(diag.num_rows == d + 1 for diag in monitor.summarize().values())


def test_stable_rank_monitor_reset_clears_the_window():
    d = 4
    monitor = StableRankMonitor(dim=d)
    first = torch.randn(200, d, generator=gen(50))
    for name in ("context", "plan"):
        monitor.update(name, first)
    kept = monitor.check(reset=False)
    again = monitor.check(reset=True)
    assert again["plan"].num_rows == kept["plan"].num_rows == 200
    assert again["plan"].stable_rank == kept["plan"].stable_rank
    with pytest.raises(ValueError):
        monitor.summarize()
    second = torch.randn(50, d, generator=gen(51))
    for name in ("context", "plan"):
        monitor.update(name, second)
    report = monitor.check()
    assert report["plan"].num_rows == 50
    assert_matches(report["plan"].stable_rank, ref_stable_rank(ref_covariance(py(second))))


def test_stable_rank_monitor_threshold_from_config():
    monitor = StableRankMonitor.from_config(16, VICRegConfig(srank_abort_fraction=0.25))
    assert monitor.threshold == 4.0
    assert StableRankMonitor(dim=16).threshold == 8.0                  # default d/2 (Sec. 5.1)


@pytest.mark.parametrize("abort_fraction", [0.0, 1.5])
def test_stable_rank_monitor_rejects_abort_fraction_outside_unit_interval(abort_fraction):
    with pytest.raises(ValueError):
        StableRankMonitor(dim=4, abort_fraction=abort_fraction)
