"""Frequency-dependent Moran process (finite-population dynamics).

Model
-----
A well-mixed population of fixed size ``N`` with two strategies, ``0`` and ``1``.
The state ``i`` is the number of strategy-0 individuals (``0 <= i <= N``); ``0``
and ``N`` are absorbing. Payoffs use the 2x2 matrix ``M`` with the same
convention as :mod:`egt.games` (``M[s, t]`` = payoff to strategy ``s`` against
``t``). Following Nowak, Sasaki, Taylor & Fudenberg (2004, *Nature* 428:646) and
Nowak (2006, *Evolutionary Dynamics*, ch. 6-7), payoffs exclude self-interaction:

    f0(i) = [ M00 (i-1) + M01 (N-i) ] / (N-1)        (strategy-0 payoff)
    f1(i) = [ M10  i    + M11 (N-i-1) ] / (N-1)      (strategy-1 payoff)

Effective fitness uses selection intensity ``w in [0, 1]`` (linear mapping,
Nowak 2004):

    F_s(i) = 1 - w + w * f_s(i)

``w = 0`` is neutral drift; ``w = 1`` makes fitness equal payoff. Fitness must
stay positive — scale payoffs or lower ``w`` if not (enforced).

One elementary Moran step: pick a reproducer with probability proportional to
fitness, and a uniformly random individual to die; the offspring replaces the
dead one. The state moves by at most +/-1, giving a birth-death chain:

    T+(i) = [ i F0 / (i F0 + (N-i) F1) ] * (N-i)/N        i -> i+1
    T-(i) = [ (N-i) F1 / (i F0 + (N-i) F1) ] * i/N        i -> i-1

Closed-form results used here (textbook, hence permitted by CLAUDE.md)
--------------------------------------------------------------------
The ratio ``gamma_i = T-(i)/T+(i) = F1(i)/F0(i)`` reduces the fixation
probability to a gambler's-ruin sum (Karlin & Taylor 1975; Nowak 2006 eq. 6.20).
Starting from ``i`` strategy-0 individuals, the probability of reaching ``N``:

    rho_i = ( 1 + sum_{k=1}^{i-1} prod_{j=1}^{k} gamma_j )
            -----------------------------------------------
            ( 1 + sum_{k=1}^{N-1} prod_{j=1}^{k} gamma_j )

Special cases checked in the verification gates:
  * Neutral (w=0): gamma_j = 1, so rho_1 = 1/N.
  * Constant selection (M00=M01=r, M10=M11=1, w=1): r-fold advantage gives
    rho_1 = (1 - 1/r) / (1 - 1/r^N)   (Nowak 2006 eq. 6.21).

Link to the deterministic core: the drift ``T+(i) - T-(i)`` has the sign of
``F0(i) - F1(i) = w (f0(i) - f1(i))``, i.e. the same sign as the replicator
velocity at ``x = i/N`` (up to the O(1/N) self-interaction correction). The
N -> infinity deterministic limit of this process is the replicator equation
(Traulsen, Claussen & Hauert 2005).

Reproducibility: every stochastic routine takes an explicit ``rng`` or ``seed``;
there is no implicit global RNG state (CLAUDE.md: Reproducibility).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Type alias: accept a Generator, an integer seed, or None.
RngLike = Union[np.random.Generator, int, None]


def _as_generator(rng: RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _validate(N: int, M: NDArray[np.float64], w: float) -> None:
    if N < 2:
        raise ValueError(f"N must be >= 2; got {N}.")
    if M.shape != (2, 2):
        raise ValueError(f"Moran process here is 2-strategy; M must be 2x2, got {M.shape}.")
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"Selection intensity w must be in [0, 1]; got {w}.")


def fitnesses(N: int, M: ArrayLike, w: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Effective fitnesses ``(F0, F1)`` at interior states ``i = 1 .. N-1``.

    Returns two arrays of length ``N-1`` indexed so that element ``k`` is state
    ``i = k + 1``.

    Raises
    ------
    ValueError
        If any effective fitness is non-positive (the gambler's-ruin ratio is
        then ill-defined — rescale payoffs or reduce ``w``).
    """
    M = np.asarray(M, dtype=np.float64)
    _validate(N, M, w)
    i = np.arange(1, N, dtype=np.float64)  # 1 .. N-1
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    f0 = (a * (i - 1.0) + b * (N - i)) / (N - 1.0)
    f1 = (c * i + d * (N - i - 1.0)) / (N - 1.0)
    F0 = 1.0 - w + w * f0
    F1 = 1.0 - w + w * f1
    if np.any(F0 <= 0.0) or np.any(F1 <= 0.0):
        raise ValueError(
            "Non-positive effective fitness encountered; rescale payoffs or reduce w."
        )
    return F0, F1


def fixation_probability(N: int, M: ArrayLike, w: float, i0: int = 1) -> float:
    """Exact probability that strategy 0 fixates, starting from ``i0`` copies.

    Uses the gambler's-ruin formula (see module docstring). Exact up to floating
    point; the products ``prod gamma_j`` are accumulated iteratively.

    Parameters
    ----------
    N:
        Population size (>= 2).
    M:
        2x2 payoff matrix.
    w:
        Selection intensity in [0, 1].
    i0:
        Initial number of strategy-0 individuals (1 <= i0 <= N-1). Default 1 (a
        single mutant), giving the standard fixation probability ``rho``.
    """
    if not (1 <= i0 <= N - 1):
        raise ValueError(f"i0 must satisfy 1 <= i0 <= N-1; got {i0} with N={N}.")
    F0, F1 = fitnesses(N, M, w)
    gamma = F1 / F0  # gamma_j for j = 1 .. N-1

    # cumulative_products[k] = prod_{j=1}^{k} gamma_j, for k = 1 .. N-1
    partial = 1.0
    cum_sum_to = np.empty(N, dtype=np.float64)  # cum_sum_to[k] = sum_{m=1}^{k} prod...
    cum_sum_to[0] = 0.0
    running = 0.0
    for k in range(1, N):
        partial *= gamma[k - 1]
        running += partial
        cum_sum_to[k] = running

    numerator = 1.0 + cum_sum_to[i0 - 1]
    denominator = 1.0 + cum_sum_to[N - 1]
    return float(numerator / denominator)


def fixation_probability_constant_selection(r: float, N: int) -> float:
    """Closed-form Moran fixation probability for a single mutant of fitness ``r``.

    ``rho = (1 - 1/r) / (1 - 1/r^N)`` (Nowak 2006, eq. 6.21). For ``r = 1`` this
    is the neutral value ``1/N``. Provided for the verification gate.
    """
    if r <= 0:
        raise ValueError(f"Relative fitness r must be > 0; got {r}.")
    if N < 2:
        raise ValueError(f"N must be >= 2; got {N}.")
    if r == 1.0:
        return 1.0 / N
    return (1.0 - 1.0 / r) / (1.0 - 1.0 / r**N)


def simulate_fixation(
    N: int,
    M: ArrayLike,
    w: float,
    i0: int = 1,
    *,
    rng: RngLike = None,
) -> bool:
    """Run one realisation to absorption; return ``True`` if strategy 0 fixates.

    Simulates the *embedded jump chain*: from state ``i`` the next change is to
    ``i+1`` with probability ``F0(i) / (F0(i) + F1(i))`` and to ``i-1``
    otherwise. The no-op (``i -> i``) steps of the full Moran chain do not affect
    which absorbing state is reached, so omitting them leaves the fixation
    probability exact while avoiding wasted draws. (For fixation *times* the
    no-ops would matter; this routine targets fixation probability.)
    """
    if not (1 <= i0 <= N - 1):
        raise ValueError(f"i0 must satisfy 1 <= i0 <= N-1; got {i0} with N={N}.")
    generator = _as_generator(rng)
    F0, F1 = fitnesses(N, M, w)
    p_up = F0 / (F0 + F1)  # index k -> state i = k+1

    i = i0
    while 0 < i < N:
        if generator.random() < p_up[i - 1]:
            i += 1
        else:
            i -= 1
    return i == N


def estimate_fixation_probability(
    N: int,
    M: ArrayLike,
    w: float,
    i0: int = 1,
    *,
    n_runs: int,
    seed: RngLike = None,
) -> tuple[float, float]:
    """Monte-Carlo estimate of the fixation probability and its standard error.

    Returns ``(p_hat, std_error)`` where ``p_hat`` is the fraction of ``n_runs``
    realisations in which strategy 0 fixated and ``std_error`` is the binomial
    standard error ``sqrt(p_hat (1 - p_hat) / n_runs)``. Seeded for reproducibility.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1; got {n_runs}.")
    generator = _as_generator(seed)
    # Precompute p_up once; reuse across runs for speed.
    F0, F1 = fitnesses(N, M, w)
    p_up = F0 / (F0 + F1)

    wins = 0
    for _ in range(n_runs):
        i = i0
        while 0 < i < N:
            if generator.random() < p_up[i - 1]:
                i += 1
            else:
                i -= 1
        if i == N:
            wins += 1
    p_hat = wins / n_runs
    std_error = float(np.sqrt(p_hat * (1.0 - p_hat) / n_runs))
    return p_hat, std_error
