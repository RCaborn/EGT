"""Deterministic replicator dynamics.

Equation integrated
--------------------
For a population of ``n`` strategies with shares ``x`` on the probability
simplex ``Δ = { x ∈ ℝ^n : x_i ≥ 0, Σ_i x_i = 1 }`` and payoff matrix ``A``
(``A[i, j]`` = payoff to strategy ``i`` against strategy ``j``):

    dx_i/dt = x_i · ( (A x)_i − φ(x) ),      φ(x) = (x · A x) / (Σ_j x_j)

This is the standard replicator equation (Hofbauer & Sigmund 1998, *Evolutionary
Games and Population Dynamics*, eq. 7.1; Weibull 1995). The only deviation from
the textbook form is that the mean payoff ``φ`` is written as a **mass-weighted**
average — divided by ``Σ_j x_j`` rather than assuming it equals 1.

On the simplex (``Σ x = 1``) this is *identical* to the textbook equation. Off
the simplex it makes the vector field exactly tangent to the constraint surface:

    Σ_i dx_i/dt = (Σ_i x_i (A x)_i) − φ(x)·(Σ_i x_i)
                = (x · A x) − [(x · A x)/(Σ x)]·(Σ x) = 0   for all x.

So total mass is conserved to integrator roundoff and the simplex is invariant,
which is exactly what verification gate 4 checks. (The naive form ``φ = x·A x``
only conserves mass *on* the simplex, leaving the constraint to drift under the
integrator.) This identity holds for every payoff matrix, so it does not change
any on-simplex result.

Method
------
Adaptive Runge–Kutta (RK45) via :func:`scipy.integrate.solve_ivp`, per the
project solver stack. Tolerances are explicit arguments and are never left to
the library default (CLAUDE.md: Reproducibility).

Confidence: high. The replicator equation and the mass-conservation identity
above are textbook; the equation is nonlinear and is integrated numerically —
no closed-form trajectory is attempted (CLAUDE.md: The non-negotiable rule).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult


def average_payoff(x: NDArray[np.float64], A: NDArray[np.float64]) -> float:
    """Mass-weighted mean payoff φ(x) = (x · A x) / Σ_j x_j.

    Equal to the population-average payoff x·A x on the simplex (Σ x = 1).
    """
    total = float(np.sum(x))
    return float(x @ (A @ x)) / total


def replicator_rhs(
    t: float,
    x: NDArray[np.float64],
    A: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Right-hand side of the replicator ODE, dx/dt.

    Parameters
    ----------
    t:
        Time. Unused — the system is autonomous — but required by the
        :func:`scipy.integrate.solve_ivp` callback signature.
    x:
        Current state, shape ``(n,)``.
    A:
        Payoff matrix, shape ``(n, n)``; ``A[i, j]`` is the payoff to strategy
        ``i`` against strategy ``j``.

    Returns
    -------
    ndarray, shape ``(n,)``
        ``dx/dt = x ⊙ (A x − φ(x))`` with the mass-weighted mean φ.
    """
    fitness = A @ x                       # (A x)_i: expected payoff to strategy i
    total = np.sum(x)
    phi = (x @ fitness) / total           # mass-weighted average payoff φ(x)
    return x * (fitness - phi)


def simulate(
    A: ArrayLike,
    x0: ArrayLike,
    t_span: tuple[float, float],
    *,
    t_eval: Optional[Sequence[float]] = None,
    method: str = "RK45",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
    dense_output: bool = False,
) -> OdeResult:
    """Integrate the replicator dynamics for payoff matrix ``A`` from ``x0``.

    Integrates ``dx_i/dt = x_i ((A x)_i − φ(x))`` with RK45 (see module
    docstring for the equation and the mass-weighting of φ).

    Parameters
    ----------
    A:
        Payoff matrix, shape ``(n, n)``.
    x0:
        Initial state, shape ``(n,)``. Must be non-negative with positive total
        mass; it is normalised onto the simplex (``Σ x0 = 1``) before integration.
    t_span:
        ``(t0, tf)`` integration interval.
    t_eval:
        Optional times at which to store the solution. If ``None``, the solver
        chooses the output points.
    method:
        :func:`scipy.integrate.solve_ivp` method. Default ``'RK45'``; use
        ``'BDF'`` / ``'LSODA'`` for stiff systems (CLAUDE.md: Solver stack).
    rtol, atol:
        Relative / absolute tolerances. Explicit by design — defaults are tight
        (1e-10 / 1e-12) so that the simplex-invariance gate is met; tune
        deliberately and record the values used.
    max_step:
        Optional cap on the internal step size.
    dense_output:
        If ``True``, attach a continuous interpolant (``sol.sol``).

    Returns
    -------
    scipy.integrate.OdeResult
        ``sol.t`` shape ``(m,)``, ``sol.y`` shape ``(n, m)``.

    Raises
    ------
    ValueError
        If ``A`` is not square, ``x0`` has the wrong shape, is negative, or has
        non-positive total mass.
    RuntimeError
        If the integration reports failure (surfaced immediately, not buried —
        CLAUDE.md: Tool-failure protocol).
    """
    A = np.asarray(A, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix; got shape {A.shape}.")
    n = A.shape[0]
    if x0.shape != (n,):
        raise ValueError(f"x0 must have shape ({n},) to match A; got {x0.shape}.")
    if np.any(x0 < 0.0):
        raise ValueError("x0 must be non-negative (a point on the simplex).")
    total = float(np.sum(x0))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("x0 must have positive, finite total mass.")
    x0 = x0 / total  # project onto the simplex so Σ x0 = 1 up to roundoff

    sol = solve_ivp(
        replicator_rhs,
        t_span,
        x0,
        method=method,
        t_eval=t_eval,
        args=(A,),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense_output,
    )
    if not sol.success:
        raise RuntimeError(f"Replicator integration failed: {sol.message}")
    return sol
