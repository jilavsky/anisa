"""
Reading the measured data of a direction.

Anisa reads data **only through pyIrena's public reader** (``pyirena.io.discover_scattering`` /
``load_scattering``), so it accepts exactly what pyIrena accepts: NXcanSAS and simple HDF5
(``.h5 .hdf5 .hdf .nxs``) and text curves (``.dat .txt .csv``), and files are never modified.

Anisa needs **desmeared** data: the model has no slit smearing, and for an anisotropic sample
slit-smeared data cannot be desmeared as if it were isotropic (PLAN.md section 11). A
slit-smeared curve can still be loaded, with a warning.

Units: Q in 1/Angstrom, intensity in 1/cm. NXcanSAS files carry their own units; text files are
read with the ``q_unit`` the caller gives. A curve without an uncertainty column gets
``error_fraction * intensity`` (pyIrena's convention, default 5 %).
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import h5py
import numpy as np
from pyirena.io import ScatteringLocation, discover_scattering, load_scattering

from anisa.core.model import DataSource, Direction

__all__ = [
    "content_fingerprint",
    "discover_curves",
    "find_wavelength",
    "load_direction",
    "read_wavelength",
    "reload_direction_data",
]

_WAVELENGTH_PATHS = ("instrument/wavelength", "instrument/beam/incident_wavelength")
_ANGSTROM_PER_UNIT = {"a": 1.0, "å": 1.0, "angstrom": 1.0, "ang": 1.0, "nm": 10.0, "pm": 0.01}


def content_fingerprint(path: str | Path) -> str:
    """SHA-256 of the file content (hex). Unlike pyIrena's fingerprint (path, size and mtime) it
    survives copying and moving a file, and changes if the data are edited."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wavelength(path: str | Path) -> float | None:
    """Wavelength [Angstrom] stored in an HDF5 file's instrument group, or ``None``.

    Only formulas 4a and 4b need a wavelength. Text files and files without one return ``None``.
    """
    path = Path(path)
    if path.suffix.lower() not in (".h5", ".hdf5", ".hdf", ".nxs"):
        return None
    try:
        with h5py.File(path, "r") as handle:
            for entry in handle.values():
                if not isinstance(entry, h5py.Group):
                    continue
                for relative in _WAVELENGTH_PATHS:
                    node = entry.get(relative)
                    if isinstance(node, h5py.Dataset) and node.size == 1:
                        value = float(np.asarray(node[()]).ravel()[0])
                        unit = node.attrs.get("units", "angstrom")
                        unit = (unit.decode() if isinstance(unit, bytes) else str(unit)).strip()
                        value *= _ANGSTROM_PER_UNIT.get(unit.lower(), 1.0)
                        if np.isfinite(value) and value > 0:
                            return value
    except OSError:
        return None
    return None


def discover_curves(path: str | Path) -> list[ScatteringLocation]:
    """Every selectable curve in a file (an HDF5 file may hold several), without reading arrays."""
    return discover_scattering(path)


def _select_location(
    path: Path, internal_path: str | None, variant: str | None
) -> ScatteringLocation:
    locations = discover_scattering(path)
    if internal_path is not None:
        for location in locations:
            if location.internal_path == internal_path:
                return location
        raise ValueError(
            f"{path.name} has no curve at {internal_path!r}; available: "
            f"{[loc.internal_path for loc in locations]}"
        )
    if variant is not None:
        locations = [loc for loc in locations if loc.variant == variant] or locations
    if len(locations) != 1:
        raise ValueError(
            f"{path.name} contains {len(locations)} curves; pass internal_path= to choose one "
            f"of {[loc.internal_path for loc in locations]}"
        )
    return locations[0]


def _fill(direction: Direction, location: ScatteringLocation, source: DataSource) -> None:
    record = load_scattering(location, q_unit=source.q_unit, error_fraction=source.error_fraction)
    q = np.asarray(record.q, dtype=float)
    intensity = np.asarray(record.intensity, dtype=float)
    if record.intensity_unit.replace(" ", "").lower() not in ("1/cm", "cm-1", "cm^-1", "1/cm."):
        warnings.warn(
            f"{location.path.name}: intensity unit is {record.intensity_unit!r}, not 1/cm; "
            "absolute-scale results (volume fractions) are only meaningful in 1/cm",
            UserWarning,
            stacklevel=3,
        )
    if record.uncertainty is None:
        sigma = source.error_fraction * np.abs(intensity)
    else:
        sigma = np.asarray(record.uncertainty, dtype=float)
    order = np.argsort(q, kind="stable")  # plots and ranges assume ascending Q
    direction.q, direction.intensity, direction.uncertainty = (
        q[order],
        intensity[order],
        sigma[order],
    )
    if location.variant == "slit-smeared":
        warnings.warn(
            f"{location.display_name} is slit-smeared; Anisa models desmeared data only",
            UserWarning,
            stacklevel=3,
        )


def load_direction(
    path: str | Path,
    *,
    alpha_q: float = 0.0,
    omega_q: float = 0.0,
    internal_path: str | None = None,
    variant: str | None = None,
    q_unit: str = "1/A",
    error_fraction: float = 0.05,
    label: str | None = None,
) -> Direction:
    """Read a curve into a new :class:`~anisa.core.model.Direction`.

    Args:
        path: data file.
        alpha_q, omega_q: direction of Q in the sample frame [deg]. Not stored in the data files;
            the caller knows the sample orientation.
        internal_path: HDF5 group of the curve, required when the file has more than one.
        variant: prefer curves of this variant (``"default"`` skips slit-smeared copies).
        q_unit: unit of Q in a text file: ``"1/A"``, ``"1/nm"``, ...
        error_fraction: relative uncertainty for curves without one.
        label: defaults to the file name (and the curve name for multi-curve files).

    Raises:
        FileNotFoundError, ValueError: missing file, unsupported suffix, ambiguous or unknown
            curve, unreadable data.
    """
    path = Path(path).expanduser().resolve()
    location = _select_location(path, internal_path, variant)
    source = DataSource(
        path=str(path),
        internal_path=location.internal_path,
        variant=location.variant,
        fingerprint=content_fingerprint(path),
        q_unit=q_unit,
        error_fraction=error_fraction,
    )
    direction = Direction(
        label=label or location.display_name,
        alpha_q=alpha_q,
        omega_q=omega_q,
        source=source,
    )
    _fill(direction, location, source)
    return direction


def reload_direction_data(direction: Direction, base_dir: str | Path | None = None) -> bool:
    """(Re)read the arrays of a direction from its :class:`DataSource`.

    Args:
        direction: direction whose ``source`` is set. Its ``q``, ``intensity`` and
            ``uncertainty`` are replaced on success and left untouched on failure.
        base_dir: folder a relative ``source.path`` is resolved against (the project folder).

    Returns:
        ``True`` if the data were loaded, ``False`` if there is no source or the file could not
        be read. A missing or unreadable file warns instead of raising, so a project whose data
        moved still opens. A file whose content changed since the project was saved loads, with
        a warning.
    """
    source = direction.source
    if source is None:
        return False
    path = Path(source.path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    if not path.is_file():
        warnings.warn(f"data file not found: {path}", UserWarning, stacklevel=2)
        return False
    try:
        if source.fingerprint and content_fingerprint(path) != source.fingerprint:
            warnings.warn(
                f"{path.name} changed since the project was saved (different content)",
                UserWarning,
                stacklevel=2,
            )
        location = _select_location(path.resolve(), source.internal_path, source.variant)
        _fill(direction, location, source)
    except (ValueError, OSError, KeyError) as error:
        warnings.warn(f"could not read {path.name}: {error}", UserWarning, stacklevel=2)
        return False
    return True


def find_wavelength(directions) -> float | None:
    """First wavelength [Angstrom] stored in the data files of ``directions``, or ``None``."""
    for direction in directions:
        if direction.source is not None:
            value = read_wavelength(direction.source.path)
            if value is not None:
                return value
    return None
