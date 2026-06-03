#!/usr/bin/env python3
"""Generate the canonical verification figures with provenance sidecars.

Each figure is a visual re-statement of a verification gate that is already
checked numerically in ``tests/test_verification.py``. Every output gets a
``<figure>.json`` sidecar recording the script, seed, git commit, parameters,
and library versions (CLAUDE.md: Reproducibility).

Usage
-----
    python scripts/make_verification_figures.py [--outdir figures] [--ext pdf]

The figures here are deterministic (ODE integration), so ``seed`` is recorded as
``None``; the pattern is in place for the stochastic figures to come.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (no install): put src/ on the path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from egt.plotting import (  # noqa: E402
    hawk_dove_convergence_figure,
    moran_fixation_figure,
    prisoners_dilemma_fixation_figure,
    rps_orbits_figure,
    stochastic_replicator_figure,
)
from egt.provenance import save_figure  # noqa: E402

_SCRIPT = "scripts/make_verification_figures.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="figures", type=Path)
    parser.add_argument("--ext", default="pdf", choices=["pdf", "svg", "png"],
                        help="pdf/svg are dissertation-grade; png is a quick check.")
    args = parser.parse_args()

    outdir: Path = args.outdir
    ext: str = args.ext

    # (name, builder, params, seed). seed is None for deterministic (ODE) figures
    # and an explicit integer for the stochastic (Monte-Carlo) figure.
    builders = [
        (
            "hawk_dove_convergence",
            hawk_dove_convergence_figure,
            {"V": 2.0, "C": 5.0,
             "initial_hawk_shares": [0.05, 0.25, 0.5, 0.75, 0.95],
             "T": 20.0, "rtol": 1e-10, "atol": 1e-12},
            None,
        ),
        (
            "rps_orbits",
            rps_orbits_figure,
            {"a": 1.0,
             "initial_conditions": [[0.40, 0.35, 0.25],
                                    [0.45, 0.30, 0.25],
                                    [0.50, 0.25, 0.25]],
             "orbit_T": 50.0, "conservation_T": 300.0,
             "rtol": 1e-10, "atol": 1e-12},
            None,
        ),
        (
            "prisoners_dilemma_fixation",
            prisoners_dilemma_fixation_figure,
            {"T_payoff": 5.0, "R": 3.0, "P": 1.0, "S": 0.0,
             "initial_coop_shares": [0.1, 0.3, 0.5, 0.7, 0.9],
             "T": 15.0, "rtol": 1e-10, "atol": 1e-12},
            None,
        ),
        (
            "moran_fixation",
            moran_fixation_figure,
            {"N": 20,
             "r_points": [0.6, 0.8, 1.0, 1.25, 1.6, 2.0],
             "n_runs": 50_000, "seed": 20240603},
            20240603,
        ),
        (
            "stochastic_replicator",
            stochastic_replicator_figure,
            {"V": 2.0, "C": 5.0, "sigma": 0.25, "T": 30.0, "dt": 0.01,
             "n_paths": 2000, "n_paths_shown": 12,
             "sigma_neutral": [0.7, 0.4], "seed": 20240603},
            20240603,
        ),
    ]

    for name, builder, params, seed in builders:
        fig = builder(**params)
        path = outdir / f"{name}.{ext}"
        sidecar = save_figure(fig, path, script=_SCRIPT, seed=seed, params=params)
        print(f"wrote {path}  (+ {sidecar.name})")


if __name__ == "__main__":
    main()
