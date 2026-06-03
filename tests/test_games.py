"""Unit tests for egt.games payoff-matrix constructors."""
from __future__ import annotations

import numpy as np
import pytest

from egt.games import hawk_dove, prisoners_dilemma, rock_paper_scissors


def test_hawk_dove_structure() -> None:
    V, C = 2.0, 5.0
    A = hawk_dove(V, C)
    expected = np.array([[(V - C) / 2.0, V],
                         [0.0, V / 2.0]])
    np.testing.assert_allclose(A, expected)


@pytest.mark.parametrize("V, C", [(0.0, 5.0), (-1.0, 5.0), (2.0, 0.0), (2.0, -3.0)])
def test_hawk_dove_rejects_nonpositive_params(V: float, C: float) -> None:
    with pytest.raises(ValueError):
        hawk_dove(V, C)


def test_rps_is_zero_sum_and_antisymmetric() -> None:
    A = rock_paper_scissors(1.0)
    # Zero-sum symmetric game: A is antisymmetric (A^T = -A).
    np.testing.assert_allclose(A.T, -A)
    assert np.all(np.diag(A) == 0.0)


def test_rps_scaling() -> None:
    np.testing.assert_allclose(rock_paper_scissors(2.5), 2.5 * rock_paper_scissors(1.0))


def test_rps_rejects_nonpositive_stake() -> None:
    with pytest.raises(ValueError):
        rock_paper_scissors(0.0)


def test_prisoners_dilemma_ordering_enforced() -> None:
    # Valid ordering T>R>P>S passes.
    prisoners_dilemma(5.0, 3.0, 1.0, 0.0)
    # Violations raise.
    with pytest.raises(ValueError):
        prisoners_dilemma(T=2.0, R=3.0, P=1.0, S=0.0)  # T < R


def test_prisoners_dilemma_structure() -> None:
    A = prisoners_dilemma(5.0, 3.0, 1.0, 0.0)
    np.testing.assert_allclose(A, np.array([[3.0, 0.0], [5.0, 1.0]]))
