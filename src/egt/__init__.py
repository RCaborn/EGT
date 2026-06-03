"""EGT — evolutionary game theory simulator.

Public surface kept deliberately small: the deterministic replicator
integrator and the canonical payoff matrices used by the verification gates.
"""
from __future__ import annotations

from egt.games import hawk_dove, prisoners_dilemma, rock_paper_scissors
from egt.moran import (
    estimate_fixation_probability,
    fitnesses,
    fixation_probability,
    fixation_probability_constant_selection,
    simulate_fixation,
)
from egt.replicator import average_payoff, replicator_rhs, simulate

__all__ = [
    # replicator (deterministic)
    "average_payoff",
    "replicator_rhs",
    "simulate",
    # games
    "hawk_dove",
    "rock_paper_scissors",
    "prisoners_dilemma",
    # moran (finite population)
    "fitnesses",
    "fixation_probability",
    "fixation_probability_constant_selection",
    "simulate_fixation",
    "estimate_fixation_probability",
]

__version__ = "0.0.1"
