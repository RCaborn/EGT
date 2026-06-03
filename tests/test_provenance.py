"""Tests for figure provenance and that the figure builders run."""
from __future__ import annotations

import json

from matplotlib.figure import Figure

from egt.plotting import (
    hawk_dove_convergence_figure,
    prisoners_dilemma_fixation_figure,
    rps_orbits_figure,
)
from egt.provenance import save_figure


def test_save_figure_writes_sidecar(tmp_path) -> None:
    fig = Figure()
    fig.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
    path = tmp_path / "demo.pdf"

    sidecar = save_figure(
        fig, path, script="tests/test_provenance.py",
        seed=123, params={"alpha": 0.5}, caption="demo",
    )

    assert path.exists()
    assert sidecar == path.with_suffix(".pdf.json")
    meta = json.loads(sidecar.read_text())
    for key in ("figure", "script", "git_commit", "seed", "params", "versions", "created_utc"):
        assert key in meta
    assert meta["figure"] == "demo.pdf"
    assert meta["seed"] == 123
    assert meta["params"] == {"alpha": 0.5}
    assert meta["versions"]["numpy"] != "unknown"


def test_figure_builders_return_figures() -> None:
    # Short horizons keep this fast; correctness is covered by the gates.
    assert isinstance(hawk_dove_convergence_figure(T=5.0), Figure)
    assert isinstance(rps_orbits_figure(orbit_T=12.0, conservation_T=20.0), Figure)
    assert isinstance(prisoners_dilemma_fixation_figure(T=5.0), Figure)
