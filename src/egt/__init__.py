"""EGT — evolutionary game theory simulator.

Public surface kept deliberately small: the deterministic replicator
integrator and the canonical payoff matrices used by the verification gates.
"""
from __future__ import annotations

from egt.games import hawk_dove, prisoners_dilemma, rock_paper_scissors
from egt.replicator import average_payoff, replicator_rhs, simulate

__all__ = [
    "average_payoff",
    "replicator_rhs",
    "simulate",
    "hawk_dove",
    "rock_paper_scissors",
    "prisoners_dilemma",
]

__version__ = "0.0.1"
