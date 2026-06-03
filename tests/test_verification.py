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
from egt.replicator import simulate

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
