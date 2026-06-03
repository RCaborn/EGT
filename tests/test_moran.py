"""Unit tests for egt.moran (transition algebra, reproducibility, validation).

The canonical Moran results (neutral 1/N, constant-selection closed form, drift
sign vs replicator, the 1/3 law) are exercised as verification gates in
tests/test_verification.py; these are the narrower source-level checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from egt.moran import (
    estimate_fixation_probability,
    fitnesses,
    fixation_probability,
    fixation_probability_constant_selection,
    simulate_fixation,
)


def test_fitnesses_exclude_self_interaction() -> None:
    # At i=1 strategy 0 meets only strategy 1 (no other 0), so f0 = M01.
    N = 10
    M = np.array([[5.0, 2.0], [3.0, 7.0]])
    F0, F1 = fitnesses(N, M, w=1.0)  # w=1 => fitness == payoff
    # k=0 corresponds to i=1.
    assert F0[0] == pytest.approx(2.0)                 # f0(1) = M01
    # f1(1) = [M10*1 + M11*(N-2)] / (N-1)
    assert F1[0] == pytest.approx((3.0 * 1 + 7.0 * (N - 2)) / (N - 1))
    # At i=N-1 strategy 1 meets only strategy 0, so f1 = M10.
    assert F1[-1] == pytest.approx(3.0)                # f1(N-1) = M10


def test_neutral_w_zero_gives_equal_fitness() -> None:
    F0, F1 = fitnesses(20, np.array([[3.0, 1.0], [4.0, 2.0]]), w=0.0)
    np.testing.assert_allclose(F0, 1.0)
    np.testing.assert_allclose(F1, 1.0)


def test_fixation_probability_endpoints() -> None:
    # rho_i is a probability and monotone increasing in the starting count.
    N = 15
    M = np.array([[2.0, 2.0], [1.0, 1.0]])  # constant selection, advantageous
    rhos = [fixation_probability(N, M, 1.0, i0=i) for i in range(1, N)]
    assert all(0.0 < r < 1.0 for r in rhos)
    assert all(b > a for a, b in zip(rhos, rhos[1:]))  # strictly increasing


def test_constant_selection_helper_neutral_limit() -> None:
    assert fixation_probability_constant_selection(1.0, 25) == pytest.approx(1.0 / 25)


def test_estimate_is_reproducible_with_seed() -> None:
    M = np.array([[2.0, 2.0], [1.0, 1.0]])
    out_a = estimate_fixation_probability(30, M, 1.0, n_runs=5000, seed=7)
    out_b = estimate_fixation_probability(30, M, 1.0, n_runs=5000, seed=7)
    out_c = estimate_fixation_probability(30, M, 1.0, n_runs=5000, seed=8)
    assert out_a == out_b          # identical seed -> identical result
    assert out_a[0] != out_c[0]    # different seed -> (almost surely) different


def test_simulate_fixation_accepts_generator_and_int_seed() -> None:
    M = np.array([[2.0, 2.0], [1.0, 1.0]])
    r1 = simulate_fixation(20, M, 1.0, rng=np.random.default_rng(42))
    r2 = simulate_fixation(20, M, 1.0, rng=42)
    assert r1 == r2 and isinstance(r1, bool)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(N=1, M=np.eye(2), w=0.0), "N must be"),
        (dict(N=10, M=np.ones((3, 3)), w=0.0), "2x2"),
        (dict(N=10, M=np.eye(2), w=1.5), r"w must be"),
    ],
)
def test_fitnesses_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        fitnesses(**kwargs)


def test_nonpositive_fitness_rejected() -> None:
    # w=1 with a negative payoff drives effective fitness <= 0 somewhere.
    with pytest.raises(ValueError, match="Non-positive"):
        fitnesses(10, np.array([[-5.0, -5.0], [1.0, 1.0]]), w=1.0)


def test_fixation_probability_rejects_bad_i0() -> None:
    with pytest.raises(ValueError, match="i0"):
        fixation_probability(10, np.eye(2), 0.0, i0=0)
