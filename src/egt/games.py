"""Canonical payoff matrices.

These are the games used by the verification gates (CLAUDE.md). Payoff matrices
are *experimental inputs*: each constructor takes the game parameters explicitly
and does not bake in convenient numbers to make a result come out clean.

Convention: ``A[i, j]`` is the payoff to strategy ``i`` when it meets strategy
``j``. All results below are textbook (Maynard Smith 1982; Hofbauer & Sigmund
1998; Weibull 1995).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hawk_dove(V: float, C: float) -> NDArray[np.float64]:
    """Hawk–Dove game. Strategy order: (0) Hawk, (1) Dove.

        A = [[(V − C)/2,   V  ],
             [    0     ,  V/2 ]]

    For ``0 < V < C`` the dynamics have a unique interior rest point with Hawk
    share ``x* = V / C`` (Maynard Smith 1982; Hofbauer & Sigmund 1998, sec. 7).
    This is verification gate 1.

    Parameters
    ----------
    V:
        Value of the contested resource (V > 0).
    C:
        Cost of escalated conflict (C > 0). The interior ESS exists for V < C.
    """
    if V <= 0 or C <= 0:
        raise ValueError(f"Hawk–Dove requires V > 0 and C > 0; got V={V}, C={C}.")
    return np.array(
        [[(V - C) / 2.0, V],
         [0.0, V / 2.0]],
        dtype=np.float64,
    )


def rock_paper_scissors(a: float = 1.0) -> NDArray[np.float64]:
    """Zero-sum Rock–Paper–Scissors. Order: (0) Rock, (1) Paper, (2) Scissors.

        A = a · [[ 0, −1,  1],
                 [ 1,  0, −1],
                 [−1,  1,  0]]

    Win = +a, loss = −a, tie = 0. The interior rest point is (1/3, 1/3, 1/3) and
    trajectories are closed orbits on which the Shahshahani-invariant
    ``H = x1·x2·x3`` is conserved (Hofbauer & Sigmund 1998, sec. 7.5). This is
    verification gate 2.

    Parameters
    ----------
    a:
        Stake of a win/loss (a > 0). Scales the speed of the orbit, not its shape.
    """
    if a <= 0:
        raise ValueError(f"Rock–Paper–Scissors requires a > 0; got a={a}.")
    return a * np.array(
        [[0.0, -1.0, 1.0],
         [1.0, 0.0, -1.0],
         [-1.0, 1.0, 0.0]],
        dtype=np.float64,
    )


def prisoners_dilemma(
    T: float = 5.0,
    R: float = 3.0,
    P: float = 1.0,
    S: float = 0.0,
) -> NDArray[np.float64]:
    """Prisoner's Dilemma. Strategy order: (0) Cooperate, (1) Defect.

        A = [[R, S],
             [T, P]]

    Requires the ordering ``T > R > P > S``. Defection strictly dominates
    cooperation, so it fixates from any interior start (verification gate 3).

    Parameters
    ----------
    T, R, P, S:
        Temptation, Reward, Punishment, Sucker payoffs. Default (5, 3, 1, 0) is
        the canonical Axelrod parameterisation, which also satisfies 2R > T + S.
    """
    if not (T > R > P > S):
        raise ValueError(
            f"Prisoner's Dilemma requires T > R > P > S; got T={T}, R={R}, P={P}, S={S}."
        )
    return np.array(
        [[R, S],
         [T, P]],
        dtype=np.float64,
    )
