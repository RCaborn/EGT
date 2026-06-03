"""Unit tests for egt.stochastic (log-score Euler-Maruyama).

The canonical correctness gates (exact Gaussian logit for the neutral game,
zero-noise deterministic limit, simplex invariance under noise) live in
tests/test_verification.py; these are the narrower source-level checks:
drift convergence order, reproducibility, the small-noise mean, and validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from egt.games import hawk_dove
from egt.replicator import simulate as ode_simulate
from egt.stochastic import simulate_replicator_sde


def test_drift_convergence_is_first_order() -> None:
    # With sigma=0 the scheme is explicit Euler on the log scores; error vs the
    # high-accuracy solve_ivp solution should halve as dt halves (order 1).
    A = hawk_dove(2.0, 5.0)
    x0 = np.array([0.8, 0.2])
    T = 10.0
    ref = ode_simulate(A, x0, (0.0, T)).y[0, -1]
    errs = []
    for dt in (0.04, 0.02, 0.01, 0.005):
        f = simulate_replicator_sde(A, x0, 0.0, (0.0, T), dt, seed=0, n_paths=1).final[0, 0]
        errs.append(abs(f - ref))
    for a, b in zip(errs, errs[1:]):
        assert 1.7 < a / b < 2.3, f"convergence ratio {a / b:.2f} not ~2"


def test_reproducible_with_seed() -> None:
    A = hawk_dove(2.0, 5.0)
    x0 = np.array([0.5, 0.5])
    kw = dict(sigma=0.3, t_span=(0.0, 5.0), dt=0.01, n_paths=10)
    a = simulate_replicator_sde(A, x0, seed=42, **kw).final
    b = simulate_replicator_sde(A, x0, seed=42, **kw).final
    c = simulate_replicator_sde(A, x0, seed=43, **kw).final
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generator_and_int_seed_agree() -> None:
    A = hawk_dove(2.0, 5.0)
    x0 = np.array([0.5, 0.5])
    kw = dict(sigma=0.2, t_span=(0.0, 3.0), dt=0.01, n_paths=5)
    via_int = simulate_replicator_sde(A, x0, seed=11, **kw).final
    via_gen = simulate_replicator_sde(A, x0, seed=np.random.default_rng(11), **kw).final
    assert np.array_equal(via_int, via_gen)


def test_small_noise_mean_tracks_deterministic() -> None:
    # For small sigma the ensemble mean should follow the deterministic path.
    A = hawk_dove(2.0, 5.0)
    x0 = np.array([0.85, 0.15])
    T, dt = 15.0, 0.005
    res = simulate_replicator_sde(A, x0, 0.05, (0.0, T), dt, seed=3, n_paths=3000)
    mean_hawk = res.X.mean(axis=0)[0]
    ode = ode_simulate(A, x0, (0.0, T), t_eval=res.t).y[0]
    assert np.max(np.abs(mean_hawk - ode)) < 0.02


def test_full_path_shapes_and_grid() -> None:
    A = hawk_dove(2.0, 5.0)
    res = simulate_replicator_sde(A, np.array([0.5, 0.5]), 0.1, (0.0, 1.0), 0.01,
                                  seed=0, n_paths=4)
    assert res.X.shape == (4, 2, 101)
    assert res.t.shape == (101,)
    assert res.t[0] == 0.0 and abs(res.t[-1] - 1.0) < 1e-12
    np.testing.assert_allclose(res.X[:, :, -1], res.final)


def test_keep_full_false_stores_only_final() -> None:
    A = hawk_dove(2.0, 5.0)
    res = simulate_replicator_sde(A, np.array([0.5, 0.5]), 0.1, (0.0, 1.0), 0.01,
                                  seed=0, n_paths=4, keep_full=False)
    assert res.X is None
    assert res.final.shape == (4, 2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(A=np.ones((2, 3)), x0=[0.5, 0.5], sigma=0.1, t_span=(0, 1), dt=0.1), "square"),
        (dict(A=np.eye(3), x0=[0.5, 0.5], sigma=0.1, t_span=(0, 1), dt=0.1), "shape"),
        (dict(A=np.eye(2), x0=[0.0, 1.0], sigma=0.1, t_span=(0, 1), dt=0.1), "strictly positive"),
        (dict(A=np.eye(2), x0=[0.5, 0.5], sigma=-0.1, t_span=(0, 1), dt=0.1), "non-negative"),
        (dict(A=np.eye(2), x0=[0.5, 0.5], sigma=0.1, t_span=(0, 1), dt=-0.1), "dt must be"),
        (dict(A=np.eye(2), x0=[0.5, 0.5], sigma=0.1, t_span=(0, 0), dt=0.1), "steps"),
    ],
)
def test_input_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        simulate_replicator_sde(seed=0, **kwargs)
