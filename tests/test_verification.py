"""Verification gates for the deterministic replicator solver.

These are the canonical results from CLAUDE.md that any integrator must
reproduce *before* it is trusted for the dissertation. If a test here fails,
the solver is wrong — fix the solver, do not loosen the gate.

All results are textbook (Maynard Smith 1982; Hofbauer & Sigmund 1998,
*Evolutionary Games and Population Dynamics*; Weibull 1995). Tolerances are
explicit and were set with margin against measured behaviour, not guessed.
"""
from __future__ import annotations

import numpy as np
import pytest

from egt.games import hawk_dove, prisoners_dilemma, rock_paper_scissors
from egt.moran import (
    estimate_fixation_probability,
    fitnesses,
    fixation_probability,
    fixation_probability_constant_selection,
)
from egt.replicator import replicator_rhs, simulate
from egt.stochastic import simulate_replicator_sde

# Integrator tolerances used throughout the gates (explicit, never defaulted).
RTOL = 1e-10
ATOL = 1e-12


def _count_upward_crossings(series: np.ndarray, level: float) -> int:
    """Number of times ``series`` crosses ``level`` from below — one per cycle."""
    centred = series - level
    return int(np.sum((centred[:-1] < 0.0) & (centred[1:] >= 0.0)))


# --------------------------------------------------------------------------- #
# Gate 1 — Hawk–Dove: interior fixed point at x* = V/C, from multiple starts. #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hawk0", [0.02, 0.2, 0.5, 0.8, 0.98])
def test_hawk_dove_interior_fixed_point(hawk0: float) -> None:
    V, C = 2.0, 5.0  # V < C => interior ESS exists
    x_star = V / C
    A = hawk_dove(V, C)

    sol = simulate(A, [hawk0, 1.0 - hawk0], (0.0, 200.0), rtol=RTOL, atol=ATOL)
    hawk_final = sol.y[0, -1]

    assert abs(hawk_final - x_star) < 1e-4, (
        f"Hawk share converged to {hawk_final}, expected V/C={x_star}"
    )


# --------------------------------------------------------------------------- #
# Gate 2 — Rock–Paper–Scissors: closed orbits, H = x1*x2*x3 conserved.        #
# --------------------------------------------------------------------------- #
def test_rps_closed_orbits_conserve_product() -> None:
    A = rock_paper_scissors(1.0)
    x0 = np.array([0.4, 0.35, 0.25])  # off-centre interior start

    # Period ~ 11 time units; 1500 units => ~136 cycles, comfortably >= 100.
    T = 1500.0
    t_eval = np.linspace(0.0, T, 15001)
    sol = simulate(A, x0, (0.0, T), t_eval=t_eval, rtol=RTOL, atol=ATOL)
    x = sol.y

    # >= 100 cycles, detected as upward crossings of the centre value 1/3.
    cycles = _count_upward_crossings(x[0], 1.0 / 3.0)
    assert cycles >= 100, f"only {cycles} cycles in T={T}; need >= 100"

    # H = x1*x2*x3 conserved to within integrator tolerance over the whole run.
    H = x[0] * x[1] * x[2]
    rel_drift = np.max(np.abs(H - H[0])) / H[0]
    assert rel_drift < 1e-6, f"H drifted by relative {rel_drift:.2e}"

    # Orbit stays interior (no strategy dies on a closed orbit).
    assert x.min() > 1e-6, f"trajectory left the interior: min component {x.min():.2e}"


# --------------------------------------------------------------------------- #
# Gate 3 — Prisoner's Dilemma: defection fixates from any interior start.     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("coop0", [0.99, 0.75, 0.5, 0.25, 0.01])
def test_prisoners_dilemma_defection_fixates(coop0: float) -> None:
    A = prisoners_dilemma()  # canonical (T,R,P,S) = (5,3,1,0)

    # Cooperation decays at rate >= 1, so T=50 drives it far below 1e-6.
    sol = simulate(A, [coop0, 1.0 - coop0], (0.0, 50.0), rtol=RTOL, atol=ATOL)
    coop_final = sol.y[0, -1]
    defect_final = sol.y[1, -1]

    assert coop_final < 1e-6, f"cooperation survived: {coop_final:.3e}"
    assert defect_final > 1.0 - 1e-6, f"defection did not fixate: {defect_final}"


# --------------------------------------------------------------------------- #
# Gate 4 — Simplex invariance: Sum x = 1 to 1e-10, no component drifts < 0.   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "A, x0, T",
    [
        (hawk_dove(2.0, 5.0), [0.02, 0.98], 200.0),
        (hawk_dove(2.0, 5.0), [0.5, 0.5], 200.0),
        (rock_paper_scissors(1.0), [0.4, 0.35, 0.25], 1500.0),
        (prisoners_dilemma(), [0.99, 0.01], 50.0),
        (prisoners_dilemma(), [0.01, 0.99], 50.0),
    ],
)
def test_simplex_invariance(A: np.ndarray, x0: list[float], T: float) -> None:
    t_eval = np.linspace(0.0, T, 2001)
    sol = simulate(A, x0, (0.0, T), t_eval=t_eval, rtol=RTOL, atol=ATOL)
    x = sol.y

    sum_drift = np.max(np.abs(x.sum(axis=0) - 1.0))
    assert sum_drift < 1e-10, f"Sum x deviated from 1 by {sum_drift:.2e}"

    # A dip of ~1e-12 at a fixation boundary is atol-scale roundoff, not the
    # trajectory leaving the simplex; anything below -1e-11 would be.
    assert x.min() >= -1e-11, f"a component drifted negative: {x.min():.2e}"


# =========================================================================== #
# Verification gates for the finite-population Moran process (egt.moran).      #
# These are the closed-form results CLAUDE.md permits because they are         #
# textbook (Nowak 2006; Nowak, Sasaki, Taylor & Fudenberg 2004).               #
# =========================================================================== #

# --- Moran gate 1: neutral fixation probability is exactly 1/N. ------------- #
@pytest.mark.parametrize("N", [2, 5, 10, 50, 100])
def test_moran_neutral_fixation_is_one_over_N(N: int) -> None:
    M = np.array([[1.0, 1.0], [1.0, 1.0]])  # payoffs irrelevant when w = 0
    rho = fixation_probability(N, M, w=0.0)
    assert abs(rho - 1.0 / N) < 1e-12, f"neutral rho={rho}, expected {1.0 / N}"


# --- Moran gate 2: constant selection matches (1-1/r)/(1-1/r^N). ------------ #
@pytest.mark.parametrize("N, r", [(10, 1.1), (20, 1.5), (50, 0.8), (100, 2.0)])
def test_moran_constant_selection_closed_form(N: int, r: float) -> None:
    # M00=M01=r, M10=M11=1 with w=1 gives strategy-0 fitness r against fitness 1.
    M = np.array([[r, r], [1.0, 1.0]])
    rho_chain = fixation_probability(N, M, w=1.0)
    rho_closed = fixation_probability_constant_selection(r, N)
    assert abs(rho_chain - rho_closed) < 1e-12, (
        f"chain={rho_chain}, closed-form={rho_closed}"
    )


def test_moran_monte_carlo_matches_closed_form() -> None:
    # Seeded Monte Carlo agrees with the exact fixation probability.
    N, r = 20, 1.5
    M = np.array([[r, r], [1.0, 1.0]])
    rho = fixation_probability_constant_selection(r, N)
    p_hat, se = estimate_fixation_probability(N, M, 1.0, n_runs=200_000, seed=12345)
    assert abs(p_hat - rho) < 4.0 * se, f"MC {p_hat:.5f}+/-{se:.5f} vs exact {rho:.5f}"


# --- Moran gate 3: drift sign matches the (verified) replicator velocity. --- #
def test_moran_drift_sign_matches_replicator() -> None:
    # Strategy 0 strictly dominates (a>c and b>d): the birth-death drift must
    # point the same way as the deterministic replicator at every interior state.
    N = 50
    M = np.array([[4.0, 3.0], [2.0, 1.0]])
    F0, F1 = fitnesses(N, M, w=0.5)
    for k, i in enumerate(range(1, N)):
        drift_sign = np.sign(F0[k] - F1[k])
        x = np.array([i / N, 1.0 - i / N])
        repl_sign = np.sign(replicator_rhs(0.0, x, M)[0])
        assert drift_sign == repl_sign, f"sign mismatch at i={i}"


# --- Moran gate 4: the 1/3 law (Nowak, Sasaki, Taylor & Fudenberg 2004). ---- #
# In a coordination game with unstable interior equilibrium x*, under weak
# selection and large N a single strategy-0 mutant is favoured (rho > 1/N) iff
# x* < 1/3. Tested with cases comfortably either side of the threshold.
@pytest.mark.parametrize(
    "a, b, c, d",
    [
        (7.0, 1.0, 1.0, 2.0),  # x* = 1/7  < 1/3  -> favoured
        (8.0, 1.0, 1.0, 2.0),  # x* = 1/8  < 1/3  -> favoured
        (3.0, 1.0, 1.0, 5.0),  # x* = 2/3  > 1/3  -> disfavoured
        (2.0, 1.0, 1.0, 4.0),  # x* = 3/4  > 1/3  -> disfavoured
    ],
)
def test_moran_one_third_law(a: float, b: float, c: float, d: float) -> None:
    N, w = 500, 0.005  # large N, weak selection
    M = np.array([[a, b], [c, d]])
    x_star = (d - b) / ((a - c) + (d - b))  # unstable interior equilibrium (share of 0)
    rho = fixation_probability(N, M, w)
    assert (rho > 1.0 / N) == (x_star < 1.0 / 3.0), (
        f"x*={x_star:.3f}, rho={rho:.3e}, 1/N={1.0 / N:.3e}"
    )


# =========================================================================== #
# Verification gates for the stochastic replicator (egt.stochastic).           #
# Fudenberg & Harris (1992); Imhof (2005). Log-score Euler-Maruyama.           #
# =========================================================================== #

# --- SDE gate 1: neutral game gives an EXACT Gaussian logit. ----------------- #
# With A = 0 the log scores are constant-coefficient SDEs, so the scheme is
# exact and u = log(x0/x1) ~ N(u0 - (s0^2 - s1^2) T / 2, (s0^2 + s1^2) T). The
# predicted mean depends on the -sigma^2/2 Ito correction, so this gate fails
# loudly if that term is wrong (a wrong correction shifts the mean by ~90 SE).
def test_sde_neutral_game_gaussian_logit() -> None:
    A = np.zeros((2, 2))
    s0, s1 = 0.7, 0.4
    sigma = np.array([s0, s1])
    x0 = np.array([0.5, 0.5])
    T, dt, n_paths = 1.0, 0.005, 40_000
    res = simulate_replicator_sde(A, x0, sigma, (0.0, T), dt,
                                  seed=1, n_paths=n_paths, keep_full=False)
    u = np.log(res.final[:, 0] / res.final[:, 1])

    mean_pred = 0.0 - 0.5 * (s0**2 - s1**2) * T
    var_pred = (s0**2 + s1**2) * T
    se_mean = np.sqrt(var_pred / n_paths)
    assert abs(u.mean() - mean_pred) < 4.0 * se_mean, (
        f"mean {u.mean():.4f} vs predicted {mean_pred:.4f} (Ito term?)"
    )
    assert abs(u.var() / var_pred - 1.0) < 0.05, (
        f"variance ratio {u.var() / var_pred:.4f}"
    )


# --- SDE gate 2: zero-noise limit reproduces the deterministic replicator. --- #
def test_sde_zero_noise_matches_deterministic() -> None:
    A = hawk_dove(2.0, 5.0)
    x0 = np.array([0.8, 0.2])
    T = 20.0
    sde = simulate_replicator_sde(A, x0, 0.0, (0.0, T), dt=0.005, seed=0, n_paths=1)
    ode = simulate(A, x0, (0.0, T), rtol=RTOL, atol=ATOL)
    assert abs(sde.final[0, 0] - ode.y[0, -1]) < 1e-4, (
        f"sde={sde.final[0, 0]}, ode={ode.y[0, -1]}"
    )


# --- SDE gate 3: simplex invariance and positivity under noise. -------------- #
def test_sde_simplex_invariance_under_noise() -> None:
    A = hawk_dove(2.0, 5.0)
    res = simulate_replicator_sde(A, np.array([0.5, 0.5]), np.array([0.5, 0.3]),
                                  (0.0, 30.0), dt=0.01, seed=7, n_paths=300)
    sums = res.X.sum(axis=1)  # (n_paths, n_steps+1)
    assert np.max(np.abs(sums - 1.0)) < 1e-10, "Sum x left the simplex under noise"
    assert res.X.min() > 0.0, "a component became non-positive under noise"
