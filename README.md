# EGT — Evolutionary Game Theory Simulator

Simulator for an MSc dissertation, built around the **replicator equation** and
(later) its stochastic / finite-population variants. The working rules for the
project — including the non-negotiable "integrate numerically, don't solve
symbolically" rule — live in [`CLAUDE.md`](CLAUDE.md) and are binding.

## Status

Deterministic replicator core **and** a finite-population (frequency-dependent
Moran) model, both passing their verification gates (see below). Figures are
generated with provenance sidecars. Stochastic *replicator* SDE (aggregate
shocks / Euler–Maruyama) and spatial models are not implemented yet.

## Layout

```
src/egt/            # library code (the only dissertation-grade code)
  replicator.py     # deterministic replicator ODE, integrated with solve_ivp (RK45)
  games.py          # canonical payoff matrices (Hawk-Dove, RPS, Prisoner's Dilemma)
  moran.py          # frequency-dependent Moran process (finite population), seeded
  plotting.py       # headless figure builders (return matplotlib Figures)
  provenance.py     # save_figure() + provenance sidecar (script, seed, commit)
scripts/            # regenerable entry points
  make_verification_figures.py
tests/              # pytest suite
  test_verification.py   # the canonical verification gates (replicator + Moran)
  test_replicator.py     # RHS algebra, mass conservation, input validation
  test_games.py          # payoff-matrix constructors
  test_moran.py          # fitness algebra, reproducibility, validation
  test_provenance.py     # sidecar contents, figure builders run
notebooks/          # exploratory only — nothing here is dissertation-grade
figures/            # vector (PDF/SVG) figures tracked; quick PNGs git-ignored
data/               # free-source data only (no Orbis)
```

## The replicator equation

For `n` strategies with shares `x` on the simplex and payoff matrix `A`
(`A[i, j]` = payoff to `i` against `j`):

```
dx_i/dt = x_i * ( (A x)_i - φ(x) ),   φ(x) = (x · A x) / (Σ_j x_j)
```

The mean payoff `φ` is written as a mass-weighted average; on the simplex this
is exactly the textbook replicator equation (Hofbauer & Sigmund 1998), and it
makes the vector field exactly tangent to the constraint so `Σ x = 1` is held to
integrator roundoff. The equation is nonlinear and is **only** integrated
numerically — no closed-form trajectories. See the docstring in
`src/egt/replicator.py` for the derivation.

## Verification gates

Any new solver must reproduce these before it is trusted (`CLAUDE.md`).

Deterministic replicator:

1. **Hawk-Dove** — interior fixed point at `x* = V/C` from multiple starts (tol 1e-4).
2. **Rock-Paper-Scissors** — closed orbits; `H = x1·x2·x3` conserved over ≥100 cycles.
3. **Prisoner's Dilemma** — defection fixates from any interior start.
4. **Simplex invariance** — `Σ x = 1` to 1e-10; no component drifts negative.

Finite-population Moran process (closed forms are textbook — Nowak 2006; Nowak
et al. 2004 — so permitted by `CLAUDE.md`):

5. **Neutral fixation** — `ρ = 1/N` exactly.
6. **Constant selection** — `ρ = (1 − 1/r)/(1 − 1/rᴺ)`; seeded Monte Carlo agrees.
7. **Replicator link** — birth-death drift sign matches the replicator velocity.
8. **The 1/3 law** — a mutant is favoured (`ρ > 1/N`) iff the unstable `x* < 1/3`.

## Setup & running the tests

```bash
python -m pip install -r requirements.txt   # exact pins for reproducibility
python -m pytest                            # src/ is on the path via pyproject.toml
```

Dependencies are pinned to exact versions in `requirements.txt`; deterministic
runs are intended to be bit-for-bit reproducible.

## Reproducibility

- Every stochastic run will take an explicit `seed` (no implicit RNG state).
- Numerical tolerances (`rtol`, `atol`, `dt`) are explicit, never defaulted silently.
- Every dissertation figure records its script, seed(s), and git commit hash.
