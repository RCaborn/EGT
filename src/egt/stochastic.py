"""Stochastic replicator dynamics with aggregate shocks (Fudenberg-Harris).

Model
-----
Each strategy ``i`` has an unnormalised "score" ``Y_i`` (think: population
biomass) driven by a geometric SDE whose growth rate is its expected payoff plus
an aggregate Brownian shock that hits every individual of type ``i`` at once
(Fudenberg & Harris 1992, *J. Econ. Theory* 57:420; Imhof 2005, *Ann. Appl.
Probab.* 15:1019):

    dY_i = Y_i [ (A x)_i dt + sigma_i dW_i ],     x_i = Y_i / sum_j Y_j ,

with independent standard Brownian motions ``W_i`` and per-strategy noise
intensities ``sigma_i >= 0``. ``A[i, j]`` is the payoff to ``i`` against ``j``
(same convention as :mod:`egt.games`). The induced frequency process is the
*stochastic replicator equation*; in Ito form

    dx_i = x_i [ (A x)_i - x.Ax - sigma_i^2 x_i + sum_j sigma_j^2 x_j^2 ] dt
           + x_i [ sigma_i dW_i - sum_j x_j sigma_j dW_j ].

Setting all ``sigma_i = 0`` recovers the deterministic replicator equation.

Method
------
Euler-Maruyama applied in **log-score coordinates**. By Ito's lemma the log
scores satisfy a constant-diffusion SDE

    d log Y_i = ( (A x)_i - sigma_i^2 / 2 ) dt + sigma_i dW_i ,

which is integrated as

    z_i <- z_i + ( (A x)_i - sigma_i^2 / 2 ) dt + sigma_i sqrt(dt) * xi_i ,
    x   <- softmax(z) ,                              xi_i ~ N(0, 1) i.i.d.

This is mathematically equivalent to the stochastic replicator SDE above (the
common ``-x.Ax`` drift cancels under the softmax normalisation), but it is
**positivity preserving** (``Y_i = e^{z_i} > 0`` always) and keeps
``sum_i x_i = 1`` to roundoff for every step size, sidestepping the negativity
that plain Euler-Maruyama on the frequency SDE would suffer. The multiplicative
noise has constant coefficients in log-space, so it is integrated *exactly*;
discretisation error enters only through the drift (first order in ``dt``).

Reproducibility: every run takes an explicit ``seed`` (or ``Generator``); there
is no implicit global RNG state (CLAUDE.md: Reproducibility). All tolerances /
step sizes are explicit.

Confidence: the derivation (including the ``-sigma_i^2/2`` Ito correction) is
standard but was re-derived here; it is checked against an exact distribution in
``tests/test_verification.py`` (neutral game -> Gaussian logit) and against the
deterministic solver in the zero-noise limit before being used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

RngLike = Union[np.random.Generator, int, None]


def _as_generator(rng: RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise softmax: x_i = e^{z_i} / sum_j e^{z_j}, computed stably."""
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / np.sum(e, axis=-1, keepdims=True)


@dataclass(frozen=True)
class SdeResult:
    """Output of :func:`simulate_replicator_sde`.

    Attributes
    ----------
    t:
        Time grid, shape ``(n_steps + 1,)``.
    X:
        Frequency paths, shape ``(n_paths, n, n_steps + 1)`` if the full path was
        stored, else ``None``.
    final:
        Final frequencies, shape ``(n_paths, n)``.
    sigma:
        Noise intensities used, shape ``(n,)``.
    dt:
        Time step.
    seed:
        Seed (or ``None``) passed by the caller.
    n_paths:
        Number of independent realisations.
    """

    t: NDArray[np.float64]
    X: Optional[NDArray[np.float64]]
    final: NDArray[np.float64]
    sigma: NDArray[np.float64]
    dt: float
    seed: RngLike
    n_paths: int


def simulate_replicator_sde(
    A: ArrayLike,
    x0: ArrayLike,
    sigma: ArrayLike,
    t_span: tuple[float, float],
    dt: float,
    *,
    seed: RngLike = None,
    n_paths: int = 1,
    keep_full: bool = True,
) -> SdeResult:
    """Integrate the stochastic replicator (aggregate shocks) by log-score EM.

    Parameters
    ----------
    A:
        Payoff matrix, shape ``(n, n)``.
    x0:
        Initial frequencies, shape ``(n,)``. Must be **strictly positive** (the
        log-score scheme cannot represent an absent strategy); normalised to the
        simplex before integration.
    sigma:
        Noise intensity, scalar or shape ``(n,)`` (broadcast). ``sigma = 0`` is
        the deterministic limit.
    t_span:
        ``(t0, tf)`` interval; the number of steps is ``round((tf - t0) / dt)``.
    dt:
        Time step (> 0). Explicit by design.
    seed:
        Seed or :class:`numpy.random.Generator` for reproducibility.
    n_paths:
        Number of independent realisations to simulate in parallel.
    keep_full:
        If ``True`` store the whole path (memory ``O(n_paths * n * n_steps)``);
        if ``False`` keep only the final state.

    Returns
    -------
    SdeResult
    """
    A = np.asarray(A, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape {A.shape}.")
    n = A.shape[0]
    if x0.shape != (n,):
        raise ValueError(f"x0 must have shape ({n},); got {x0.shape}.")
    if np.any(x0 <= 0.0):
        raise ValueError("x0 must be strictly positive for the log-score scheme.")
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n,)).copy()
    if np.any(sigma < 0.0):
        raise ValueError("sigma must be non-negative.")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1; got {n_paths}.")

    t0, tf = t_span
    n_steps = int(round((tf - t0) / dt))
    if n_steps < 1:
        raise ValueError(f"t_span/dt yields {n_steps} steps; need >= 1.")

    generator = _as_generator(seed)
    sqrt_dt = np.sqrt(dt)
    half_s2 = 0.5 * sigma**2  # Ito correction, shape (n,)

    x0 = x0 / x0.sum()
    z = np.broadcast_to(np.log(x0), (n_paths, n)).copy()

    X: Optional[NDArray[np.float64]] = None
    if keep_full:
        X = np.empty((n_paths, n, n_steps + 1), dtype=np.float64)
        X[:, :, 0] = _softmax(z)

    for step in range(n_steps):
        x = _softmax(z)                      # (n_paths, n)
        fitness = x @ A.T                    # (A x)_i per path, (n_paths, n)
        drift = fitness - half_s2            # broadcast over paths
        xi = generator.standard_normal((n_paths, n))
        z = z + drift * dt + sigma * sqrt_dt * xi
        if keep_full:
            X[:, :, step + 1] = _softmax(z)

    final = _softmax(z)
    t = t0 + dt * np.arange(n_steps + 1)
    return SdeResult(t=t, X=X, final=final, sigma=sigma, dt=float(dt),
                     seed=seed, n_paths=n_paths)
