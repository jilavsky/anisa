"""
Programmatic and batch interface: JSON in, JSON out.

Every function takes paths and plain numbers and returns only JSON-serialisable values (dicts,
lists, floats, strings), so the command line (:mod:`anisa.cli`) and the MCP server call exactly
the same code. The configuration is the project file the GUI saves (``*.anisa.json``), or a
results file (``*.anisa.h5``), which carries its own data.

No Qt, no matplotlib: this layer runs headless.

Functions: :func:`describe`, :func:`calculate`, :func:`fit`, :func:`anisotropy`,
:func:`list_test_systems`, :func:`write_test_system`.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from anisa import testsets
from anisa.core.anisotropy import AnisotropyScan, calculate_scan, measured_points
from anisa.core.derived import mean_volume
from anisa.core.engine import Engine
from anisa.core.fitting import (
    FitResult,
    active_parameters,
    monte_carlo,
    statistics_from_result,
)
from anisa.core.fitting import (
    fit as _fit,
)
from anisa.core.model import AnisaModel
from anisa.io.project import load_project, save_project
from anisa.io.results import RESULTS_SUFFIX, load_results, read_summary, save_results

__all__ = [
    "anisotropy",
    "calculate",
    "describe",
    "fit",
    "list_test_systems",
    "load_model",
    "summary",
    "write_test_system",
]


def load_model(path: str | Path) -> AnisaModel:
    """Read a project (``.json``) or results (``.h5``) file."""
    path = Path(path).expanduser()
    if path.suffix.lower() in (".h5", ".hdf5"):
        return load_results(path)
    return load_project(path)


def _number(value) -> float | None:
    """JSON has no NaN/inf: map them to ``None``."""
    value = float(value)
    return value if math.isfinite(value) else None


def _numbers(array) -> list:
    return [_number(v) for v in np.asarray(array, dtype=float).ravel()]


def _default_results_path(project: Path) -> Path:
    name = project.name
    for suffix in (".anisa.json", ".anisa.h5", ".json", ".h5"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return project.with_name(f"{name}{RESULTS_SUFFIX}")


def _statistics(model: AnisaModel, statistics) -> dict:
    return {
        "chi_squared": _number(statistics.chi_squared),
        "n_points": int(statistics.n_points),
        "reduced_chi_squared": _number(statistics.reduced),
        "directions": [
            {
                "index": i + 1,
                "label": d.label,
                "n_points": int(statistics.direction_points[i]),
                "reduced_chi_squared": _number(statistics.direction_reduced(i)),
            }
            for i, d in enumerate(model.directions)
        ],
    }


def describe(path: str | Path) -> dict:
    """Summary of a project or results file: populations, directions, fit flags."""
    path = Path(path).expanduser()
    model = load_model(path)
    return {
        "file": str(path.resolve()),
        "formula": model.settings.formula,
        "populations": [
            {
                "name": p.name,
                "radius": p.radius,
                "beta": p.beta,
                "volume_fraction": p.volume_fraction,
                "delta_rho": p.delta_rho,
                "size_distribution": p.size_dist.type,
                "interference": p.use_interference,
                "mean_volume": mean_volume(p),
            }
            for p in model.populations
        ],
        "directions": [
            {
                "index": i + 1,
                "label": d.label,
                "alpha_q": d.alpha_q,
                "omega_q": d.omega_q,
                "background": d.background,
                "n_points": None if d.q is None else int(len(d.q)),
                "use": d.use,
            }
            for i, d in enumerate(model.directions)
        ],
        "fit_parameters": [p.label for p in active_parameters(model)],
        "scans": [s.title() for s in model.scans],
    }


def calculate(project: str | Path, *, results: str | Path | None = None) -> dict:
    """Calculate the model along every direction and report chi-squared against the data.

    Args:
        project: project or results file.
        results: if given, write the results file there.

    Returns:
        ``{"statistics": ..., "surface_area": [...], "results_file": path or None}``.
    """
    model = load_model(project)
    engine = Engine()
    q_list = [
        np.asarray(d.q, dtype=float) if d.q is not None else np.geomspace(1e-3, 0.3, 120)
        for d in model.directions
    ]
    result = engine.evaluate(model, q_list)
    report = {
        "statistics": _statistics(model, statistics_from_result(model, result)),
        "surface_area": _numbers(result.surface_areas),
        "number_density": _numbers(result.number_densities),
        "results_file": None,
    }
    if results is not None:
        report["results_file"] = str(save_results(model, results, result=result))
    return report


def _fit_report(model: AnisaModel, result: FitResult) -> dict:
    mc = result.monte_carlo
    parameters = []
    for k, p in enumerate(result.parameters):
        parameters.append(
            {
                "label": p.parameter.label,
                "owner": p.parameter.owner,
                "index": p.parameter.index + 1,
                "path": p.parameter.path,
                "start": _number(p.start),
                "value": _number(p.value),
                "std": _number(p.std),
                "monte_carlo_std": None if mc is None else _number(mc.std[k]),
                "at_lower_limit": bool(p.at_lower),
                "at_upper_limit": bool(p.at_upper),
            }
        )
    return {
        "success": bool(result.success),
        "cancelled": bool(result.cancelled),
        "message": result.message,
        "chi_squared": _number(result.chi_squared),
        "reduced_chi_squared": _number(result.reduced_chi_squared),
        "n_points": int(result.n_points),
        "n_evaluations": int(result.n_evaluations),
        "elapsed_seconds": _number(result.elapsed_seconds),
        "parameters": parameters,
        "directions": _statistics(model, result.statistics)["directions"],
    }


def fit(
    project: str | Path,
    *,
    results: str | Path | None = None,
    monte_carlo_runs: int = 0,
    max_nfev: int | None = None,
    update_project: bool = False,
    workers: int | None = None,
) -> dict:
    """Fit the parameters flagged for fitting in ``project`` to its data.

    Args:
        project: project or results file with data in every direction to fit.
        results: results file to write (default: next to the project, ``NAME.anisa.h5``).
        monte_carlo_runs: if >= 2, also estimate uncertainties by refitting that many times.
        max_nfev: limit on model evaluations.
        update_project: write the fitted parameters back into the project file (a ``.json``
            project only; never done silently).
        workers: processes for the Monte Carlo runs (default one per CPU).

    Returns:
        The fit report: success, chi-squared, every parameter with start, value and errors,
        chi-squared per direction, and the path of the results file.

    Raises:
        ValueError: nothing flagged for fitting, or no data.
    """
    project = Path(project).expanduser()
    model = load_model(project)
    if not active_parameters(model):
        raise ValueError("no parameter is marked for fitting (set 'fit' in the project file)")
    if not any(d.use and d.fit_mask().any() for d in model.directions):
        raise ValueError("there are no data to fit: no direction has points in its Q range")
    engine = Engine()
    result = _fit(model, engine=engine, max_nfev=max_nfev)
    if monte_carlo_runs >= 2:
        result.monte_carlo = monte_carlo(model, result, monte_carlo_runs, workers=workers)
    target = Path(results) if results is not None else _default_results_path(project)
    report = _fit_report(model, result)
    report["results_file"] = str(save_results(model, target, fit=result, engine=engine))
    report["project_updated"] = None
    if update_project:
        if project.suffix.lower() in (".h5", ".hdf5"):
            raise ValueError("update_project needs a .anisa.json project, not a results file")
        report["project_updated"] = str(save_project(model, project))
    return report


def anisotropy(
    project: str | Path,
    *,
    scan: int | None = None,
    sweep: str | None = None,
    fixed: float | None = None,
    q_values: list[float] | None = None,
    n_points: int = 73,
    with_measured: bool = True,
) -> dict:
    """Model intensity as the direction of Q is swept (see :mod:`anisa.core.anisotropy`).

    Either choose a scan saved in the project (``scan``, 1-based) or define one with
    ``sweep`` (``"omega"`` or ``"alpha"``), ``fixed`` [deg] and ``q_values`` [1/A].

    Returns:
        ``{"title", "angles", "q_values", "intensity": [q][angle], "populations":
        [population][q][angle], "measured": [{q, angles, intensity, sigma, labels}]}``.
    """
    model = load_model(project)
    if sweep is not None:
        if fixed is None or not q_values:
            raise ValueError("a scan needs sweep, fixed and q_values")
        the_scan = AnisotropyScan(
            sweep=sweep, fixed=float(fixed), q_values=tuple(q_values), n_points=n_points
        )
    else:
        scans = model.scans
        if not scans:
            raise ValueError("the project has no saved scan; pass sweep, fixed and q_values")
        index = 1 if scan is None else scan
        if not 1 <= index <= len(scans):
            raise ValueError(f"scan must be between 1 and {len(scans)}")
        the_scan = copy.deepcopy(scans[index - 1])
    outcome = calculate_scan(model, the_scan, Engine())
    measured = []
    if with_measured:
        for q in the_scan.q_values:
            points = measured_points(model, the_scan, q)
            measured.append(
                {
                    "q": q,
                    "angles": _numbers(points.angles),
                    "intensity": _numbers(points.intensity),
                    "sigma": _numbers(points.sigma),
                    "labels": points.labels,
                }
            )
    return {
        "title": the_scan.title(),
        "sweep": the_scan.sweep,
        "fixed": the_scan.fixed,
        "angles": _numbers(outcome.angles),
        "q_values": list(the_scan.q_values),
        "intensity": [_numbers(row) for row in outcome.intensity],
        "populations": [[_numbers(row) for row in pop] for pop in outcome.populations],
        "measured": measured,
    }


def list_test_systems() -> list[dict]:
    """The synthetic test systems: name, title and description."""
    out = []
    for name in testsets.list_systems():
        system = testsets.make_system(name)
        out.append({"name": name, "title": system.title, "description": system.description})
    return out


def write_test_system(
    name: str, folder: str | Path, *, fmt: str = "h5", noise: float = 0.02, seed: int = 0
) -> dict:
    """Write one test system (data, truth, perturbed start, TRUTH.txt) and return the paths."""
    truth = testsets.write_test_system(name, folder, fmt, relative_noise=noise, seed=seed)
    return {
        "folder": str(truth.parent),
        "truth": str(truth),
        "start": str(truth.with_name(f"{name}_start.anisa.json")),
        "description": str(truth.with_name("TRUTH.txt")),
    }


def summary(results: str | Path) -> dict:
    """Parameters, errors and derived quantities of a results file as plain dicts."""
    data = read_summary(results)
    return {
        "version": data.version,
        "parameters": {
            name: {k: (_number(v) if isinstance(v, float) else v) for k, v in entry.items()}
            for name, entry in data.parameters.items()
        },
        "derived": {
            k: (_numbers(v) if isinstance(v, np.ndarray) else _number(v))
            for k, v in data.derived.items()
        },
        "fit": {k: (_number(v) if isinstance(v, float) else v) for k, v in data.fit.items()},
    }
