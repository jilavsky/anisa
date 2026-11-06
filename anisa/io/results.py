"""
Results file ``<name>.anisa.h5``: data, model and parameters of all directions in one file.

A fit spans several data files (one per direction of Q), so the results are written as one
self-contained NXcanSAS file (PLAN.md §6, format approved 2026-11-04)::

    /                                   attrs: creator, anisa_version, format, schema_version
    /entry                              NXentry
        dir01 … dirNN                   NXsubentry / SASentry, one per direction
            sasdata                     NXdata / SASdata: Q, I, Idev (the measured data)
            anisa_results               NXprocess: Q, I_model, I_populations, residuals, fit_mask,
                                        background; attrs alpha_q_deg, omega_q_deg, label,
                                        population_names; fit_quality (pyIrena group)
            (attrs source_path, source_internal_path, source_sha256 when the data came from a file)
    /anisa                              NXentry: the shared results
        params/<pop01_radius, …>        value; attrs label, units, fit, limit_low, limit_high
        params_std/<same names>         standard error of fitted parameters (NaN if not fitted)
        derived/                        surface_area, number_density, mean_volume per population
        orientation/popNN/              alpha_deg, p_alpha, omega_deg, b_omega (for plotting)
        fit/                            chi_squared, n_points, reduced_chi_squared, ... (if fitted)
        attrs _pyirena_config           the complete model (pyIrena setup envelope, tool "anisa")

Every NXcanSAS reader opens each direction's data (``pyirena.io.discover_scattering`` lists the
``dirNN`` entries). :func:`load_results` restores the whole session, including the data arrays,
from this one file. Group and dataset names are frozen once released; readers supply defaults
for anything absent.

Units as everywhere in Anisa: Q 1/A, I 1/cm, radius A, contrast cm^-2, angles degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from pyirena.core.fit_metrics import fit_quality_metrics
from pyirena.io.nxcansas_fit_quality import read_fit_quality, write_fit_quality
from pyirena.io.setup_config import read_setup_config, write_setup_config

import anisa
from anisa.core import derived
from anisa.core.engine import Engine, ModelResult
from anisa.core.fitting import FitResult, active_parameters
from anisa.core.model import POPULATION_PARAMETERS, AnisaModel

__all__ = [
    "RESULTS_SUFFIX",
    "ResultsSummary",
    "load_results",
    "read_summary",
    "save_results",
]

RESULTS_SUFFIX = ".anisa.h5"
FORMAT = "anisa_results"
SCHEMA_VERSION = 1
TOOL = "anisa"
_CONFIG_GROUP = "anisa"

_UNITS = {
    "radius": "angstrom",
    "volume_fraction": "1",
    "beta": "1",
    "delta_rho": "1/cm^2",
    "eta": "angstrom",
    "pack": "1",
    "background": "1/cm",
}


def _results_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.name.lower().endswith((".h5", ".hdf5")) else Path(f"{path}{RESULTS_SUFFIX}")


def parameter_name(owner: str, index: int, path: str) -> str:
    """Dataset name of a parameter: ``pop01_radius``, ``dir02_background`` (1-based, 2 digits)."""
    return f"{'pop' if owner == 'population' else 'dir'}{index + 1:02d}_{path}"


def _text(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=str(value))


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _nx(group: h5py.Group, nx_class: str, can_sas: str | None = None) -> h5py.Group:
    group.attrs["NX_class"] = nx_class
    if can_sas:
        group.attrs["canSAS_class"] = can_sas
    return group


def _write_direction(
    entry: h5py.Group,
    index: int,
    model: AnisaModel,
    result: ModelResult,
    n_parameters: int,
    names: list[str],
) -> None:
    direction = model.directions[index]
    calculated = result.directions[index]
    sub = _nx(entry.create_group(f"dir{index + 1:02d}"), "NXsubentry", "SASentry")
    sub.attrs["default"] = "sasdata"
    sub.attrs["title"] = direction.label or f"Direction {index + 1}"
    _text(sub, "definition", "NXcanSAS")
    _text(sub, "title", direction.label or f"Direction {index + 1}")
    _text(sub, "run", "anisa")
    if direction.source is not None:
        sub.attrs["source_path"] = direction.source.path
        sub.attrs["source_internal_path"] = direction.source.internal_path or ""
        sub.attrs["source_sha256"] = direction.source.fingerprint

    has_data = direction.q is not None and direction.intensity is not None
    if has_data:
        sasdata = _nx(sub.create_group("sasdata"), "NXdata", "SASdata")
        sasdata.attrs["signal"] = "I"
        sasdata.attrs["I_axes"] = "Q"
        q = sasdata.create_dataset("Q", data=np.asarray(direction.q, dtype=float))
        q.attrs.update({"units": "1/angstrom", "long_name": "Q"})
        i = sasdata.create_dataset("I", data=np.asarray(direction.intensity, dtype=float))
        i.attrs.update({"units": "1/cm", "long_name": "Intensity"})
        if direction.uncertainty is not None:
            i.attrs["uncertainties"] = "Idev"
            idev = sasdata.create_dataset(
                "Idev", data=np.asarray(direction.uncertainty, dtype=float)
            )
            idev.attrs.update({"units": "1/cm", "long_name": "Uncertainties"})

    process = _nx(sub.create_group("anisa_results"), "NXprocess")
    process.attrs.update(
        {
            "program": "anisa",
            "version": anisa.__version__,
            "alpha_q_deg": float(direction.alpha_q),
            "omega_q_deg": float(direction.omega_q),
            "label": direction.label,
            "population_names": "\n".join(names),
        }
    )
    process.create_dataset("Q", data=calculated.q).attrs["units"] = "1/angstrom"
    process.create_dataset("I_model", data=calculated.intensity).attrs["units"] = "1/cm"
    process.create_dataset("I_populations", data=calculated.populations).attrs["units"] = "1/cm"
    process.create_dataset("background", data=float(calculated.background)).attrs["units"] = "1/cm"
    if has_data:
        mask = (
            direction.fit_mask()
            if direction.uncertainty is not None
            else np.ones(np.asarray(direction.q).shape, bool)
        )
        process.create_dataset("fit_mask", data=mask.astype(np.int8))
        if direction.uncertainty is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                residuals = (direction.intensity - calculated.intensity) / direction.uncertainty
            process.create_dataset("residuals", data=residuals).attrs["long_name"] = "(I-model)/dI"
            metrics = fit_quality_metrics(
                np.asarray(direction.q)[mask],
                np.asarray(direction.intensity)[mask],
                calculated.intensity[mask],
                np.asarray(direction.uncertainty)[mask],
                n_parameters,
            )
            write_fit_quality(process, metrics)


def save_results(
    model: AnisaModel,
    path: str | Path,
    *,
    result: ModelResult | None = None,
    fit: FitResult | None = None,
    engine: Engine | None = None,
) -> Path:
    """Write ``model`` with its data and calculated curves to a results file.

    Args:
        model: the model, with data in its directions.
        path: output file; ``.anisa.h5`` is appended unless the name ends in ``.h5``/``.hdf5``.
        result: calculated model, if already at hand (it must belong to ``model`` as it is now);
            otherwise it is calculated here, at each direction's own Q values.
        fit: the last fit, for the standard errors and the fit summary.
        engine: engine to calculate with.

    Returns:
        The path written. An existing file is replaced atomically.
    """
    path = _results_path(path).resolve()
    if result is None:
        q_list = [
            np.asarray(d.q, dtype=float) if d.q is not None else np.geomspace(1e-3, 0.3, 120)
            for d in model.directions
        ]
        result = (engine or Engine()).evaluate(model, q_list)
    if len(result.directions) != len(model.directions):
        raise ValueError("result does not match the model: different number of directions")

    state = model.to_dict()
    state["schema_version"] = SCHEMA_VERSION
    for direction in state["directions"]:
        source = direction.get("source")
        if source and source.get("path"):
            source["path"] = str(Path(source["path"]).expanduser().resolve())
    names = [p.name or f"Population {i + 1}" for i, p in enumerate(model.populations)]
    active = active_parameters(model)
    std = {}
    if fit is not None:
        std = {
            parameter_name(r.parameter.owner, r.parameter.index, r.parameter.path): r.std
            for r in fit.parameters
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with h5py.File(temporary, "w") as f:
            f.attrs.update(
                {
                    "default": "entry",
                    "creator": "anisa",
                    "anisa_version": anisa.__version__,
                    "format": FORMAT,
                    "schema_version": SCHEMA_VERSION,
                    "file_name": path.name,
                    "file_time": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            )
            entry = _nx(f.create_group("entry"), "NXentry", "SASentry")
            _text(entry, "definition", "NXsas")
            for index in range(len(model.directions)):
                _write_direction(entry, index, model, result, len(active), names)

            shared = _nx(f.create_group(_CONFIG_GROUP), "NXentry")
            _write_shared(shared, model, result, fit, std, state)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _write_shared(
    shared: h5py.Group,
    model: AnisaModel,
    result: ModelResult,
    fit: FitResult | None,
    std: dict[str, float],
    state: dict,
) -> None:
    params = _nx(shared.create_group("params"), "NXcollection")
    params_std = _nx(shared.create_group("params_std"), "NXcollection")

    def put(owner: str, index: int, path: str, label: str, value: float, spec) -> None:
        name = parameter_name(owner, index, path)
        dataset = params.create_dataset(name, data=float(value))
        dataset.attrs["label"] = label
        dataset.attrs["units"] = _UNITS.get(path, "deg" if path.endswith("par1") else "")
        dataset.attrs["fit"] = bool(spec.fit) if spec is not None else False
        dataset.attrs["limit_low"] = -math.inf if spec is None or spec.min is None else spec.min
        dataset.attrs["limit_high"] = math.inf if spec is None or spec.max is None else spec.max
        params_std.create_dataset(name, data=float(std.get(name, math.nan)))

    for i, pop in enumerate(model.populations):
        for path in POPULATION_PARAMETERS:
            put(
                "population",
                i,
                path,
                f"{pop.name or f'Population {i + 1}'}: {path}",
                pop.get_parameter(path),
                pop.fit_specs.get(path),
            )
    for i, direction in enumerate(model.directions):
        put(
            "direction",
            i,
            "background",
            f"{direction.label or f'Direction {i + 1}'}: background",
            direction.background,
            direction.fit_specs.get("background"),
        )

    out = _nx(shared.create_group("derived"), "NXcollection")
    out.create_dataset("surface_area", data=result.surface_areas).attrs["units"] = "cm^2/cm^3"
    out.create_dataset("number_density", data=result.number_densities).attrs["units"] = "cm^-3"
    out.create_dataset(
        "mean_volume", data=[derived.mean_volume(p) for p in model.populations]
    ).attrs["units"] = "angstrom^3"
    out.create_dataset("total_surface_area", data=result.total_surface_area)

    orientation = _nx(shared.create_group("orientation"), "NXcollection")
    alpha = np.linspace(0.0, 90.0, 181)
    omega = np.linspace(0.0, 360.0, 361)
    for i, pop in enumerate(model.populations):
        group = _nx(orientation.create_group(f"pop{i + 1:02d}"), "NXcollection")
        group.create_dataset("alpha_deg", data=alpha)
        group.create_dataset("p_alpha", data=pop.alpha_dist.alpha_values(np.radians(alpha)))
        group.create_dataset("omega_deg", data=omega)
        group.create_dataset("b_omega", data=pop.omega_dist.omega_values(np.radians(omega)))
        group.attrs["note"] = "unnormalised factors p(alpha) and b(omega) of the orientation law"

    if fit is not None:
        group = _nx(shared.create_group("fit"), "NXcollection")
        group.create_dataset("chi_squared", data=float(fit.chi_squared))
        group.create_dataset("reduced_chi_squared", data=float(fit.reduced_chi_squared))
        group.create_dataset("n_points", data=int(fit.n_points))
        group.create_dataset("n_evaluations", data=int(fit.n_evaluations))
        group.create_dataset("elapsed_seconds", data=float(fit.elapsed_seconds))
        group.attrs["success"] = bool(fit.success)
        group.attrs["message"] = str(fit.message)
        if fit.monte_carlo is not None:
            group.attrs["monte_carlo_runs"] = int(fit.monte_carlo.n_requested)
    write_setup_config(shared, TOOL, state)


# -- reading -----------------------------------------------------------------------------------


@dataclass
class ResultsSummary:
    """What a results file records besides the arrays.

    Attributes:
        parameters: ``name -> {"value", "std", "label", "units", "fit", "limit_low",
            "limit_high"}`` for every parameter (names as in :func:`parameter_name`).
        derived: ``surface_area``, ``number_density``, ``mean_volume`` per population, and
            ``total_surface_area``.
        fit: fit summary (empty if the results were saved without a fit).
        direction_quality: pyIrena fit-quality dict of each direction (``None`` where absent).
        version: Anisa version that wrote the file.
    """

    parameters: dict[str, dict] = field(default_factory=dict)
    derived: dict[str, object] = field(default_factory=dict)
    fit: dict[str, object] = field(default_factory=dict)
    direction_quality: list[dict | None] = field(default_factory=list)
    version: str = ""


def _check_format(f: h5py.File, path: Path) -> None:
    if _decode(f.attrs.get("format", "")) != FORMAT or _CONFIG_GROUP not in f:
        raise ValueError(f"{path.name} is not an Anisa results file")


def load_results(path: str | Path) -> AnisaModel:
    """Restore the model, with its data arrays, from a results file.

    The arrays come from the file itself, so the original data files are not needed; each
    direction keeps its ``source`` record (path and fingerprint of the original).

    Raises:
        FileNotFoundError, ValueError: missing file, or not an Anisa results file.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        _check_format(f, path)
    state = read_setup_config(path, _CONFIG_GROUP, TOOL)
    if state is None:
        raise ValueError(f"{path.name} has no stored model")
    model = AnisaModel.from_dict(state)
    with h5py.File(path, "r") as f:
        for index, direction in enumerate(model.directions):
            sasdata = f.get(f"entry/dir{index + 1:02d}/sasdata")
            if sasdata is None or "Q" not in sasdata or "I" not in sasdata:
                direction.q = direction.intensity = direction.uncertainty = None
                continue
            direction.q = np.asarray(sasdata["Q"][()], dtype=float)
            direction.intensity = np.asarray(sasdata["I"][()], dtype=float)
            direction.uncertainty = (
                np.asarray(sasdata["Idev"][()], dtype=float) if "Idev" in sasdata else None
            )
    return model


def read_summary(path: str | Path) -> ResultsSummary:
    """Parameters with standard errors, derived quantities and fit quality from a results file."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = ResultsSummary()
    with h5py.File(path, "r") as f:
        _check_format(f, path)
        summary.version = _decode(f.attrs.get("anisa_version", ""))
        shared = f[_CONFIG_GROUP]
        for name, dataset in shared["params"].items():
            summary.parameters[name] = {
                "value": float(dataset[()]),
                "std": float(shared["params_std"][name][()])
                if name in shared["params_std"]
                else math.nan,
                "label": _decode(dataset.attrs.get("label", "")),
                "units": _decode(dataset.attrs.get("units", "")),
                "fit": bool(dataset.attrs.get("fit", False)),
                "limit_low": float(dataset.attrs.get("limit_low", -math.inf)),
                "limit_high": float(dataset.attrs.get("limit_high", math.inf)),
            }
        if "derived" in shared:
            for name, dataset in shared["derived"].items():
                value = dataset[()]
                summary.derived[name] = float(value) if np.ndim(value) == 0 else np.asarray(value)
        if "fit" in shared:
            fit = shared["fit"]
            summary.fit = {name: dataset[()].item() for name, dataset in fit.items()}
            for key in fit.attrs:
                if key != "NX_class":
                    value = fit.attrs[key]
                    summary.fit[key] = value.item() if isinstance(value, np.generic) else value
        for key in sorted(k for k in f["entry"] if k.startswith("dir")):
            process = f["entry"][key].get("anisa_results")
            summary.direction_quality.append(
                read_fit_quality(process) if process is not None else None
            )
    return summary
