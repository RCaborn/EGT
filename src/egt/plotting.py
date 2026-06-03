"""Plotting helpers for the replicator dynamics.

Figure builders return a :class:`matplotlib.figure.Figure` (object-oriented API,
no global pyplot state, so they work headless and are safe to unit-test). They
deliberately do not save anything — use :func:`egt.provenance.save_figure` so the
provenance sidecar is always written.

The three builders here mirror the verification gates so the figures are a visual
re-statement of results that are already checked numerically in
``tests/test_verification.py``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

from egt.games import hawk_dove, prisoners_dilemma, rock_paper_scissors
from egt.replicator import simulate

# --- ternary (3-strategy simplex) projection ------------------------------- #
# Corners: strategy 0 -> (0, 0), strategy 1 -> (1, 0), strategy 2 -> top.
_C0 = np.array([0.0, 0.0])
_C1 = np.array([1.0, 0.0])
_C2 = np.array([0.5, np.sqrt(3.0) / 2.0])


def simplex_to_xy(x: np.ndarray) -> np.ndarray:
    """Map points on the 3-simplex to 2D barycentric coordinates.

    Parameters
    ----------
    x:
        Array of shape ``(3,)`` or ``(3, m)`` with columns summing to 1.

    Returns
    -------
    ndarray
        Shape ``(2,)`` or ``(2, m)`` of plane coordinates.
    """
    x = np.asarray(x, dtype=np.float64)
    return _C0[:, None] * x[0] + _C1[:, None] * x[1] + _C2[:, None] * x[2] if x.ndim == 2 \
        else _C0 * x[0] + _C1 * x[1] + _C2 * x[2]


def _draw_ternary_frame(ax, labels: tuple[str, str, str]) -> None:
    triangle = np.column_stack([_C0, _C1, _C2, _C0])
    ax.plot(triangle[0], triangle[1], color="0.6", lw=1.0)
    for corner, label, (dx, dy, ha, va) in zip(
        (_C0, _C1, _C2),
        labels,
        [(-0.02, -0.04, "right", "top"),
         (0.02, -0.04, "left", "top"),
         (0.0, 0.03, "center", "bottom")],
    ):
        ax.annotate(label, corner, xytext=(corner[0] + dx, corner[1] + dy),
                    ha=ha, va=va, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


# --- Gate 1: Hawk-Dove convergence to x* = V/C ----------------------------- #
def hawk_dove_convergence_figure(
    V: float = 2.0,
    C: float = 5.0,
    initial_hawk_shares: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    T: float = 20.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Figure:
    """Hawk share vs time from several starts, converging to x* = V/C."""
    A = hawk_dove(V, C)
    x_star = V / C
    t_eval = np.linspace(0.0, T, 1000)

    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    for h0 in initial_hawk_shares:
        sol = simulate(A, [h0, 1.0 - h0], (0.0, T), t_eval=t_eval, rtol=rtol, atol=atol)
        ax.plot(sol.t, sol.y[0], lw=1.3)
    ax.axhline(x_star, ls="--", color="k", lw=1.0,
               label=f"$x^* = V/C = {x_star:.3g}$")
    ax.set_xlabel("time")
    ax.set_ylabel("Hawk share $x_H$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Hawk–Dove convergence ($V={V:g}$, $C={C:g}$)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# --- Gate 2: RPS closed orbits and conserved H ----------------------------- #
def rps_orbits_figure(
    a: float = 1.0,
    initial_conditions: Sequence[Sequence[float]] = (
        (0.40, 0.35, 0.25),
        (0.45, 0.30, 0.25),
        (0.50, 0.25, 0.25),
    ),
    orbit_T: float = 50.0,
    conservation_T: float = 300.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Figure:
    """Left: closed RPS orbits on the simplex. Right: H = x1*x2*x3 vs time."""
    A = rock_paper_scissors(a)

    fig = Figure(figsize=(9.0, 4.2))
    ax_orbit = fig.add_subplot(1, 2, 1)
    _draw_ternary_frame(ax_orbit, ("Rock", "Paper", "Scissors"))

    t_orbit = np.linspace(0.0, orbit_T, 4000)
    for x0 in initial_conditions:
        sol = simulate(A, list(x0), (0.0, orbit_T), t_eval=t_orbit, rtol=rtol, atol=atol)
        xy = simplex_to_xy(sol.y)
        ax_orbit.plot(xy[0], xy[1], lw=1.0)
        start = simplex_to_xy(np.asarray(x0, dtype=np.float64))
        ax_orbit.plot(start[0], start[1], "o", ms=3, color="k")
    centre = simplex_to_xy(np.full(3, 1.0 / 3.0))
    ax_orbit.plot(centre[0], centre[1], "+", ms=9, color="k")
    ax_orbit.set_title("Closed orbits about $(1/3,1/3,1/3)$")

    ax_h = fig.add_subplot(1, 2, 2)
    t_cons = np.linspace(0.0, conservation_T, 6000)
    sol = simulate(A, list(initial_conditions[0]), (0.0, conservation_T),
                   t_eval=t_cons, rtol=rtol, atol=atol)
    H = sol.y[0] * sol.y[1] * sol.y[2]
    rel_dev = (H - H[0]) / H[0]
    rel_drift = float(np.max(np.abs(rel_dev)))
    # Plot the relative deviation against a fixed band so "conserved" reads as
    # flat; auto-scaling would zoom into roundoff noise and look like growth.
    ax_h.plot(sol.t, rel_dev, lw=1.0)
    ax_h.axhline(0.0, color="0.6", lw=0.8)
    ax_h.set_ylim(-1e-6, 1e-6)
    ax_h.set_xlabel("time")
    ax_h.set_ylabel(r"relative drift in $H = x_1 x_2 x_3$")
    ax_h.set_title(f"Invariant conserved (max drift {rel_drift:.1e})")
    ax_h.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    return fig


# --- Gate 3: Prisoner's Dilemma fixation of defection ---------------------- #
def prisoners_dilemma_fixation_figure(
    T_payoff: float = 5.0,
    R: float = 3.0,
    P: float = 1.0,
    S: float = 0.0,
    initial_coop_shares: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    T: float = 15.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Figure:
    """Cooperation share vs time (log scale) decaying to 0 from several starts."""
    A = prisoners_dilemma(T_payoff, R, P, S)
    t_eval = np.linspace(0.0, T, 1000)

    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    for c0 in initial_coop_shares:
        sol = simulate(A, [c0, 1.0 - c0], (0.0, T), t_eval=t_eval, rtol=rtol, atol=atol)
        ax.semilogy(sol.t, np.clip(sol.y[0], 1e-16, None), lw=1.3,
                    label=f"$x_C(0)={c0:g}$")
    ax.set_xlabel("time")
    ax.set_ylabel("Cooperator share $x_C$ (log)")
    ax.set_title(f"Prisoner's Dilemma: defection fixates ($T,R,P,S={T_payoff:g},{R:g},{P:g},{S:g}$)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
