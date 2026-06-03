"""Figure provenance.

CLAUDE.md (Reproducibility) requires every dissertation figure to record
(a) the script that produced it, (b) the seed(s) used, and (c) the git commit
hash. :func:`save_figure` writes the figure in a vector format and a sidecar
JSON capturing all three, plus the parameters, a UTC timestamp, and the pinned
library versions, so any figure can be regenerated exactly.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Optional

from matplotlib.figure import Figure

# Repo root = two levels up from src/egt/provenance.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit(repo_root: Path = _REPO_ROOT) -> tuple[str, bool]:
    """Return ``(commit_hash, is_dirty)`` for the repo, ('unknown', False) if not a repo."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ("unknown", False)
    try:
        # Ignore untracked files: the figure being written is itself untracked
        # at generation time, so only *tracked* source modifications should
        # flag the provenance as dirty.
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = False
    return (commit, dirty)


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("numpy", "scipy", "matplotlib"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    out["python"] = sys.version.split()[0]
    return out


def figure_metadata(
    figure_name: str,
    script: str,
    *,
    seed: Optional[int] = None,
    params: Optional[Mapping[str, Any]] = None,
    caption: Optional[str] = None,
) -> dict[str, Any]:
    """Build the provenance record written alongside a figure."""
    commit, dirty = _git_commit()
    meta: dict[str, Any] = {
        "figure": figure_name,
        "script": script,
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "params": dict(params) if params is not None else {},
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "versions": _versions(),
    }
    if caption is not None:
        meta["caption"] = caption
    return meta


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    script: str,
    seed: Optional[int] = None,
    params: Optional[Mapping[str, Any]] = None,
    caption: Optional[str] = None,
    stamp: bool = True,
) -> Path:
    """Save ``fig`` to ``path`` and write a ``<path>.json`` provenance sidecar.

    Parameters
    ----------
    fig:
        The matplotlib figure to save. Format is inferred from the extension;
        use ``.pdf`` or ``.svg`` for dissertation-grade vector output.
    path:
        Output path for the figure.
    script:
        Identifier of the script that produced the figure (e.g. its repo-relative
        path). Recorded verbatim in the sidecar.
    seed:
        RNG seed used, if any. ``None`` for fully deterministic figures.
    params:
        Parameters that define the figure (payoff values, initial conditions,
        tolerances, ...). Recorded so the figure can be regenerated.
    caption:
        Optional caption text stored in the sidecar.
    stamp:
        If ``True``, render a small provenance footnote (name, short commit,
        script) onto the figure itself.

    Returns
    -------
    Path
        The path to the sidecar JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = figure_metadata(path.name, script, seed=seed, params=params, caption=caption)

    if stamp:
        commit = meta["git_commit"]
        short = commit[:10] + ("-dirty" if meta["git_dirty"] else "")
        fig.text(
            0.005,
            0.005,
            f"{path.name} · {short} · {script}",
            fontsize=5,
            color="0.5",
            ha="left",
            va="bottom",
        )

    fig.savefig(path, bbox_inches="tight")
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return sidecar
