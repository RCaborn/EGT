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
from egt.moran import (
    estimate_fixation_probability,
    fixation_probability_constant_selection,
)
from egt.replicator import simulate
from egt.stochastic import simulate_replicator_sde

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


# --- Moran process: fixation probability, closed form vs Monte Carlo -------- #
def moran_fixation_figure(
    N: int = 20,
    r_curve: Optional[np.ndarray] = None,
    r_points: Sequence[float] = (0.6, 0.8, 1.0, 1.25, 1.6, 2.0),
    n_runs: int = 50_000,
    seed: int = 20240603,
) -> Figure:
    """Moran fixation probability vs relative fitness r at population size N.

    The closed-form ``(1 - 1/r)/(1 - 1/r^N)`` (Nowak 2006) is drawn as a curve;
    seeded Monte-Carlo estimates with binomial error bars are overlaid. The
    neutral value ``1/N`` is marked. Each Monte-Carlo point uses an independent
    sub-stream derived from ``seed`` so the figure is reproducible.
    """
    if r_curve is None:
        r_curve = np.linspace(0.5, 2.0, 200)

    rho_curve = np.array([fixation_probability_constant_selection(float(r), N) for r in r_curve])

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(len(r_points))
    p_hat = np.empty(len(r_points))
    p_err = np.empty(len(r_points))
    for k, r in enumerate(r_points):
        M = np.array([[r, r], [1.0, 1.0]])
        rng = np.random.default_rng(child_seeds[k])
        p_hat[k], p_err[k] = estimate_fixation_probability(
            N, M, 1.0, n_runs=n_runs, seed=rng
        )

    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(r_curve, rho_curve, color="C0", lw=1.5,
            label=r"closed form $(1-1/r)/(1-1/r^N)$")
    ax.errorbar(r_points, p_hat, yerr=p_err, fmt="o", color="C3", ms=4,
                capsize=2, lw=1.0, label=f"Monte Carlo ($n={n_runs:,}$)")
    ax.axhline(1.0 / N, ls=":", color="0.5", lw=1.0, label=f"neutral $1/N={1.0/N:g}$")
    ax.axvline(1.0, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel("relative fitness $r$")
    ax.set_ylabel(r"fixation probability $\rho$")
    ax.set_title(f"Moran process fixation, $N={N}$")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# --- Stochastic replicator (aggregate shocks) ------------------------------- #
def stochastic_replicator_figure(
    V: float = 2.0,
    C: float = 5.0,
    sigma: float = 0.25,
    T: float = 30.0,
    dt: float = 0.01,
    n_paths: int = 2000,
    n_paths_shown: int = 12,
    sigma_neutral: Sequence[float] = (0.7, 0.4),
    seed: int = 20240603,
) -> Figure:
    """Two panels for the Fudenberg-Harris stochastic replicator.

    Left: Hawk-Dove Hawk-share sample paths under noise, the ensemble mean, and
    the deterministic trajectory, all relative to ``x* = V/C``.

    Right: for the neutral game (``A = 0``) the log-odds ``log(x0/x1)`` are exactly
    Gaussian; the simulated histogram is overlaid with the predicted density
    ``N(-(s0^2 - s1^2)T/2, (s0^2 + s1^2)T)`` -- a visual check of the Ito term.
    """
    seeds = np.random.SeedSequence(seed).spawn(2)

    fig = Figure(figsize=(9.0, 4.2))

    # Left: Hawk-Dove sample paths.
    A = hawk_dove(V, C)
    x_star = V / C
    x0 = np.array([0.85, 0.15])
    res = simulate_replicator_sde(A, x0, sigma, (0.0, T), dt,
                                  seed=np.random.default_rng(seeds[0]), n_paths=n_paths)
    ax0 = fig.add_subplot(1, 2, 1)
    for k in range(min(n_paths_shown, n_paths)):
        ax0.plot(res.t, res.X[k, 0], color="0.7", lw=0.5, alpha=0.7)
    ax0.plot(res.t, res.X.mean(axis=0)[0], color="C3", lw=1.6, label="ensemble mean")
    ode = simulate(A, x0, (0.0, T), t_eval=res.t).y[0]
    ax0.plot(res.t, ode, color="C0", lw=1.4, ls="--", label="deterministic")
    ax0.axhline(x_star, color="k", lw=0.8, ls=":", label=f"$x^*=V/C={x_star:g}$")
    ax0.set_xlabel("time")
    ax0.set_ylabel("Hawk share $x_H$")
    ax0.set_ylim(0.0, 1.0)
    ax0.set_title(rf"Hawk-Dove with noise ($\sigma={sigma:g}$)")
    ax0.legend(loc="best", fontsize=8)

    # Right: neutral-game logit distribution vs exact Gaussian.
    s0, s1 = float(sigma_neutral[0]), float(sigma_neutral[1])
    res_n = simulate_replicator_sde(np.zeros((2, 2)), np.array([0.5, 0.5]),
                                    np.array([s0, s1]), (0.0, T), dt,
                                    seed=np.random.default_rng(seeds[1]),
                                    n_paths=n_paths, keep_full=False)
    u = np.log(res_n.final[:, 0] / res_n.final[:, 1])
    mu = -0.5 * (s0**2 - s1**2) * T
    var = (s0**2 + s1**2) * T
    grid = np.linspace(u.min(), u.max(), 300)
    pdf = np.exp(-((grid - mu) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.hist(u, bins=60, density=True, color="0.8", edgecolor="none",
             label="simulated")
    ax1.plot(grid, pdf, color="C3", lw=1.6,
             label=r"exact $\mathcal{N}(\mu,\sigma^2)$")
    ax1.axvline(mu, color="k", lw=0.8, ls=":")
    ax1.set_xlabel(r"log-odds $\log(x_0/x_1)$ at $t=T$")
    ax1.set_ylabel("density")
    ax1.set_title(rf"Neutral game ($\sigma=({s0:g},{s1:g})$)")
    ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig
