# CLAUDE.md — EGT Simulator Working Rules

Instructions for Claude Code when working in this dissertation repo. Read before touching any code or math.

## Project context

Masters-level dissertation. The empirical chapter is built on an evolutionary game theory (EGT) simulator centred on the **replicator equation** (and its stochastic / finite-population variants). Free-source data only — Orbis is not available (see auto-memory `project_data_access.md`).

## The non-negotiable rule

**The replicator equation is nonlinear. Do not attempt symbolic / closed-form solutions except in the degenerate cases listed below. Numerical integration is the default and only trusted path.**

If you find yourself producing pages of algebra for a general n-strategy replicator system, stop. You are pattern-matching, not solving.

### Degenerate cases where closed form IS acceptable
- 2-strategy symmetric games with constant payoffs (e.g. Hawk-Dove interior fixed point).
- Linear ODE subsystems extracted via change of variables that you can name and cite.
- Anything where you can point to a textbook (Hofbauer & Sigmund, Weibull, Sandholm) for the result.

Everywhere else: numerical.

## Solver stack (use these, don't reinvent)

| Problem type | Method | Library |
|---|---|---|
| Deterministic replicator ODE | RK4 / RK45 adaptive | `scipy.integrate.solve_ivp` with `method='RK45'` or `'LSODA'` |
| Stiff systems (mixed timescales) | BDF | `solve_ivp(method='BDF')` |
| Stochastic replicator / finite pop noise | Euler-Maruyama | hand-rolled, seeded |
| Moran / Wright-Fisher discrete process | Gillespie or direct sampling | `numpy.random.Generator` |
| PDEs (spatial replicator, if needed) | Finite difference, explicit unless stiffness forces implicit | hand-rolled |

Do not silently substitute. If a problem looks like it needs a different method, say so and ask.

## Verification gates — run these BEFORE trusting any new solver

Every new integrator must reproduce these canonical results, with a unit test, before being used in the dissertation:

1. **Hawk-Dove** (2-strategy): interior fixed point at x* = V/C, converged from multiple initial conditions, within tolerance 1e-4.
2. **Rock-Paper-Scissors** (zero-sum 3-strategy): closed orbits around (1/3, 1/3, 1/3); the Shahshahani-invariant H = x1·x2·x3 stays constant to within integrator tolerance over ≥100 cycles.
3. **Prisoner's Dilemma**: defection fixates from any interior start.
4. **Simplex invariance**: Σxi = 1 preserved to within 1e-10 over the full run; no xi drifts negative.

If a solver fails any of these, the solver is wrong. Fix it, don't paper over it.

## Reproducibility (this is a dissertation, not a hackathon)

- Every stochastic run takes an explicit `seed` argument. No implicit RNG state.
- Pin dependencies: `requirements.txt` with exact versions. Re-run must be bit-for-bit reproducible on deterministic runs.
- Every figure produced for the dissertation has: (a) the script that produced it, (b) the seed(s) used, (c) the git commit hash, written into the figure caption or a sidecar JSON.
- Numerical tolerances (`rtol`, `atol`, dt) are explicit, never defaulted silently.

## How to flag confidence

Per user preference: surface confidence explicitly. Concretely:

- If implementing a textbook result: state it and cite the source.
- If deriving something yourself: mark it as derived, show the derivation, and run the verification gates.
- If you're below ~90% confident on a piece of math, say so in the comment block above the code, not buried in chat.
- Symbolic manipulation of nonlinear systems: assume you are wrong until a numerical check agrees.

## What NOT to do

- Do not "solve" the replicator equation analytically for n ≥ 3 strategies with non-trivial payoffs.
- Do not use SymPy as a substitute for numerical integration on nonlinear systems. SymPy is fine for verifying Jacobians at fixed points, deriving linearisations, and checking algebra — not for trajectories.
- Do not write a custom RK4 when `solve_ivp` exists, unless there's a specific reason (e.g. SDE integration, exact step control for a paper claim). If you do roll one, unit-test it against `solve_ivp` on a known problem.
- Do not present a plot without first running the verification gates on the underlying solver.
- Do not invent payoff matrices to make a result come out clean. Payoff matrices are an experimental input; treat them as such.

## Tool-failure protocol

From auto-memory `feedback_flag_tool_failures_early.md`: if something breaks (a solver diverges, a library import fails, a data source returns empty), surface it immediately. Do not silently fall back to a degraded path and bury the caveat at the end.

## Output expectations

- Code: PEP 8, type hints on public functions, docstrings stating the equation being integrated and the method.
- Tests: `pytest`, one file per module, verification gates live in `tests/test_verification.py`.
- Figures: matplotlib, vector format (PDF/SVG) for the dissertation, PNG only for quick checks.
- Repo layout: `src/egt/`, `tests/`, `notebooks/` (exploratory only — nothing in notebooks is dissertation-grade until promoted to `src/`), `figures/`, `data/`.

## When to push back on me (Rory)

If I ask for something that violates the above — a closed-form replicator solution, a symbolic SDE result beyond the standard forms, a figure without a verification gate — refuse and say why. Cite this file.
