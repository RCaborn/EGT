"""Unit tests for egt.replicator (RHS algebra, mass conservation, validation).

These are narrower than the verification gates: they pin the right-hand side
and the input contract so regressions are caught at the source.
"""
from __future__ import annotations

import numpy as np
import pytest

from egt.games import rock_paper_scissors
from egt.replicator import average_payoff, replicator_rhs, simulate


def test_rhs_matches_explicit_formula() -> None:
    # Asymmetric A and a non-simplex x to exercise the mass-weighted mean.
    A = np.array([[1.0, 2.0, 3.0],
                  [0.0, 1.0, -1.0],
                  [2.0, -2.0, 0.0]])
    x = np.array([0.2, 0.5, 0.3])
    fitness = A @ x
    phi = (x @ fitness) / x.sum()
    expected = x * (fitness - phi)
    np.testing.assert_allclose(replicator_rhs(0.0, x, A), expected, rtol=0, atol=0)


def test_rhs_conserves_mass_off_simplex() -> None:
    # Sum of dx/dt must be exactly 0 for ANY x (the tangency identity), so the
    # constraint Sum x = const cannot drift under the field. Test off the simplex.
    A = np.array([[0.0, -1.0, 2.0],
                  [3.0, 0.0, -1.0],
                  [-2.0, 1.0, 0.0]])
    for x in (np.array([0.2, 0.5, 0.3]),       # on simplex
              np.array([0.7, 0.9, 0.4]),       # mass 2.0
              np.array([1.0, 0.0, 0.5])):      # on a face, mass 1.5
        assert abs(np.sum(replicator_rhs(0.0, x, A))) < 1e-15


def test_average_payoff_is_quadratic_form_on_simplex() -> None:
    A = rock_paper_scissors(1.0)
    x = np.array([0.5, 0.3, 0.2])
    assert abs(average_payoff(x, A) - x @ (A @ x)) < 1e-15


def test_rest_point_is_stationary() -> None:
    # RPS centre (1/3,1/3,1/3) is a rest point: dx/dt = 0.
    A = rock_paper_scissors(1.0)
    x = np.full(3, 1.0 / 3.0)
    np.testing.assert_allclose(replicator_rhs(0.0, x, A), 0.0, atol=1e-15)


def test_simulate_normalises_initial_condition() -> None:
    # x0 need not sum to 1; it is projected onto the simplex.
    A = rock_paper_scissors(1.0)
    sol = simulate(A, [2.0, 1.0, 1.0], (0.0, 1.0))  # mass 4 -> (0.5,0.25,0.25)
    np.testing.assert_allclose(sol.y[:, 0], [0.5, 0.25, 0.25], atol=1e-12)


@pytest.mark.parametrize(
    "A, x0, match",
    [
        (np.ones((2, 3)), [0.5, 0.5], "square"),
        (np.eye(3), [0.5, 0.5], "shape"),
        (np.eye(2), [-0.1, 1.1], "non-negative"),
        (np.eye(2), [0.0, 0.0], "positive"),
    ],
)
def test_simulate_rejects_bad_input(A: np.ndarray, x0: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        simulate(A, x0, (0.0, 1.0))
