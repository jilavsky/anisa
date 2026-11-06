"""
The MCP tools as plain functions: path guard plus :mod:`anisa.api`. No ``mcp`` import, so they
are tested without it and the server module stays a thin registration layer.

All file access is confined to one folder tree: ``ANISA_DATA_ROOT`` if set, else the current
working directory of the server. Paths may be relative to it. A path that resolves outside the
root (``..``, absolute elsewhere, or a symlink leading out) is refused.
"""

from __future__ import annotations

import os
from pathlib import Path

from anisa import api

__all__ = ["TOOLS", "data_root", "resolve"]

ROOT_VARIABLE = "ANISA_DATA_ROOT"
#: Cap on Monte-Carlo runs an agent may request in one call (each is a full fit).
MAX_MONTE_CARLO_RUNS = 200


def data_root() -> Path:
    """The folder tree the tools may read and write."""
    return Path(os.environ.get(ROOT_VARIABLE) or Path.cwd()).expanduser().resolve()


def resolve(path: str | Path, *, must_exist: bool = True) -> Path:
    """``path`` made absolute inside :func:`data_root`.

    Raises:
        PermissionError: the path is outside the data root.
        FileNotFoundError: ``must_exist`` and the path does not exist.
    """
    root = data_root()
    candidate = Path(path).expanduser()
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError(f"{path} is outside the data root {root} (set {ROOT_VARIABLE})")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def list_files(folder: str = ".", recursive: bool = True) -> dict:
    """Anisa project (``*.anisa.json``) and results (``*.anisa.h5``) files below ``folder``."""
    base = resolve(folder)
    if not base.is_dir():
        raise NotADirectoryError(base)
    pattern = "**/*" if recursive else "*"
    found = sorted(
        p
        for p in base.glob(pattern)
        if p.name.endswith((".anisa.json", ".anisa.h5")) and p.is_file()
    )
    root = data_root()
    return {
        "root": str(root),
        "files": [
            {
                "path": p.relative_to(root).as_posix(),
                "kind": "project" if p.suffix == ".json" else "results",
            }
            for p in found
        ],
    }


def describe(file: str) -> dict:
    """Populations, directions, fit flags and saved scans of a project or results file."""
    return api.describe(resolve(file))


def calculate(file: str, results: str | None = None) -> dict:
    """Calculate the model along every direction; chi-squared against the data if present."""
    target = resolve(results, must_exist=False) if results else None
    return api.calculate(resolve(file), results=target)


def fit(
    file: str,
    results: str | None = None,
    monte_carlo_runs: int = 0,
    max_nfev: int | None = None,
    update_project: bool = False,
) -> dict:
    """Fit the parameters flagged for fitting; writes a results file next to the project."""
    if monte_carlo_runs > MAX_MONTE_CARLO_RUNS:
        raise ValueError(f"monte_carlo_runs is limited to {MAX_MONTE_CARLO_RUNS} per call")
    target = resolve(results, must_exist=False) if results else None
    return api.fit(
        resolve(file),
        results=target,
        monte_carlo_runs=monte_carlo_runs,
        max_nfev=max_nfev,
        update_project=update_project,
    )


def anisotropy(
    file: str,
    scan: int | None = None,
    sweep: str | None = None,
    fixed: float | None = None,
    q_values: list[float] | None = None,
    n_points: int = 37,
    with_measured: bool = True,
) -> dict:
    """Model intensity as the direction of Q is swept at fixed |Q| (see anisa.api.anisotropy)."""
    if n_points > 721:
        raise ValueError("n_points is limited to 721")
    return api.anisotropy(
        resolve(file),
        scan=scan,
        sweep=sweep,
        fixed=fixed,
        q_values=q_values,
        n_points=n_points,
        with_measured=with_measured,
    )


def read_results(file: str) -> dict:
    """Parameters with errors, derived quantities and fit summary of a results file."""
    return api.summary(resolve(file))


def export_bernardyn(
    file: str, package: str, recipe: str = "raw_iq", overwrite: bool = False
) -> dict:
    """Write data and model curves of all directions into a Bernardyn graph package."""
    return api.export_bernardyn(
        resolve(file), resolve(package, must_exist=False), recipe=recipe, overwrite=overwrite
    )


def list_test_systems() -> list[dict]:
    """The synthetic test systems (known truth) that write_test_system can create."""
    return api.list_test_systems()


def write_test_system(
    name: str, folder: str = "testdata", fmt: str = "h5", noise: float = 0.02, seed: int = 0
) -> dict:
    """Write a test system (data, truth project, perturbed start project, TRUTH.txt)."""
    return api.write_test_system(
        name, resolve(folder, must_exist=False), fmt=fmt, noise=noise, seed=seed
    )


#: name -> function; the server registers each as ``anisa_<name>``.
TOOLS = {
    "list_files": list_files,
    "describe": describe,
    "calculate": calculate,
    "fit": fit,
    "anisotropy": anisotropy,
    "read_results": read_results,
    "export_bernardyn": export_bernardyn,
    "list_test_systems": list_test_systems,
    "write_test_system": write_test_system,
}
