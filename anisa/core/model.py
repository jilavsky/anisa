"""
The Anisa model: directions, populations and calculation settings.

These dataclasses are what the GUI edits, what the JSON project file stores and what the engine
(`anisa.core.engine`) evaluates. Angles are in **degrees**, radii in Angstrom, ``delta_rho`` in
cm**-2 (not squared), background in cm**-1, as in the Igor package and the GUI.

``to_dict`` / ``from_dict`` supply a default for every field, so project files written by older
versions keep loading (saved-file compatibility is a hard rule). JSON keys are snake_case; the
mapping to the Igor variable names (``Pop1_Radius`` ...) is in ``docs/parameters.md``.

Fittable parameters are plain floats on the objects. Their fit flag and limits live in
``fit_specs`` (a dict keyed by parameter path), so the engine never sees fitting machinery and
the fitting module (Phase 3) can enumerate and set parameters through ``parameter_paths`` /
``get_parameter`` / ``set_parameter``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from anisa.core.formfactor import FORMULAS
from anisa.core.orientation import AngularDistribution, OrientationDistribution
from anisa.core.sizedist import SizeDistribution

SCHEMA_VERSION = 1

#: Default (min, max) used for a parameter when it is first marked for fitting.
#: ``None`` means unbounded. Lower bounds keep the physics valid (positive sizes, p >= 0).
DEFAULT_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "radius": (10.0, 1.0e7),
    "volume_fraction": (1.0e-6, 1.0),
    "beta": (0.02, 50.0),
    "eta": (1.0, 1.0e5),
    "pack": (0.0, 16.0),
    "alpha_par1": (0.0, 90.0),
    "alpha_par2": (0.0, 200.0),
    "alpha_par3": (0.0, 100.0),
    "omega_par1": (0.0, 360.0),
    "omega_par2": (0.0, 200.0),
    "omega_par3": (0.0, 100.0),
    "background": (None, None),
}

POPULATION_PARAMETERS = tuple(p for p in DEFAULT_LIMITS if p != "background")


def _num_or_none(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isinf(value) else value


@dataclass
class FitSpec:
    """Whether a parameter is varied by the fit, and between which limits (None = unbounded)."""

    fit: bool = False
    min: float | None = None
    max: float | None = None

    def to_dict(self) -> dict:
        return {"fit": self.fit, "min": self.min, "max": self.max}

    @classmethod
    def from_dict(cls, data: dict | None, default_path: str | None = None) -> FitSpec:
        lo, hi = DEFAULT_LIMITS.get(default_path or "", (None, None))
        data = data or {}
        return cls(
            fit=bool(data.get("fit", False)),
            min=_num_or_none(data["min"]) if "min" in data else lo,
            max=_num_or_none(data["max"]) if "max" in data else hi,
        )


def _fit_specs_to_dict(values: dict[str, float], specs: dict[str, FitSpec]) -> dict:
    out = {}
    for path, value in values.items():
        spec = specs.get(path) or FitSpec.from_dict(None, path)
        out[path] = {"value": value, **spec.to_dict()}
    return out


def _read_parameters(data: dict, defaults: dict[str, float], paths) -> tuple[dict, dict]:
    values, specs = {}, {}
    for path in paths:
        entry = data.get(path, defaults[path])
        if isinstance(entry, dict):
            values[path] = float(entry.get("value", defaults[path]))
            specs[path] = FitSpec.from_dict(entry, path)
        else:
            values[path] = float(entry)
            specs[path] = FitSpec.from_dict(None, path)
    return values, specs


@dataclass
class Population:
    """One population of identical-shape spheroids (Igor ``PopN_*``).

    Attributes:
        name: free-text label.
        radius: equatorial radius R [A]; location of the size distribution.
        volume_fraction: volume fraction of the sample occupied by this population.
        beta: aspect ratio; semi-axes are (R, R, beta R). < 1 oblate, > 1 prolate.
        delta_rho: scattering contrast [cm**-2] (not squared).
        size_dist: radius distribution (default: monodisperse).
        alpha_dist, omega_dist: orientation distribution factors p(alpha) and b(omega).
        use_interference: apply the Irena interference factor 1 / (1 + pack 3 j1(Q eta)/(Q eta)).
        eta: interference distance [A].
        pack: interference packing strength.
        fit_specs: fit flags and limits, keyed by parameter path (see ``parameter_paths``).
    """

    name: str = ""
    radius: float = 1000.0
    volume_fraction: float = 0.1
    beta: float = 1.0
    delta_rho: float = 1.0e11
    size_dist: SizeDistribution = field(default_factory=SizeDistribution)
    alpha_dist: AngularDistribution = field(default_factory=AngularDistribution)
    omega_dist: AngularDistribution = field(default_factory=AngularDistribution)
    use_interference: bool = False
    eta: float = 100.0
    pack: float = 0.0
    fit_specs: dict[str, FitSpec] = field(default_factory=dict)

    # -- parameter access (used by fitting and the GUI) -----------------------------------

    def parameter_paths(self) -> tuple[str, ...]:
        return POPULATION_PARAMETERS

    def get_parameter(self, path: str) -> float:
        if path.startswith(("alpha_par", "omega_par")):
            dist = self.alpha_dist if path.startswith("alpha") else self.omega_dist
            return float(getattr(dist, path.split("_")[1]))
        if path not in POPULATION_PARAMETERS:
            raise KeyError(path)
        return float(getattr(self, path))

    def set_parameter(self, path: str, value: float) -> None:
        if path.startswith(("alpha_par", "omega_par")):
            dist = self.alpha_dist if path.startswith("alpha") else self.omega_dist
            setattr(dist, path.split("_")[1], float(value))
        elif path in POPULATION_PARAMETERS:
            setattr(self, path, float(value))
        else:
            raise KeyError(path)

    def fit_spec(self, path: str) -> FitSpec:
        if path not in self.fit_specs:
            self.fit_specs[path] = FitSpec.from_dict(None, path)
        return self.fit_specs[path]

    # -- cache keys -----------------------------------------------------------------------

    def shape_key(self) -> tuple:
        """Everything that changes the *shape* of this population's I(Q).

        Excludes the volume fraction (which only scales the result) and the interference
        parameters (which are applied afterwards), so changing them reuses cached results.
        """
        return (
            float(self.radius),
            float(self.beta),
            float(self.delta_rho),
            self.size_dist.key(),
            self.alpha_dist.key(),
            self.omega_dist.key(),
        )

    # -- serialisation --------------------------------------------------------------------

    def to_dict(self) -> dict:
        values = {p: self.get_parameter(p) for p in POPULATION_PARAMETERS}
        out = {
            "name": self.name,
            "delta_rho": self.delta_rho,
            "size_dist": self.size_dist.to_dict(),
            "alpha_dist": {
                "mode": self.alpha_dist.mode,
                "table": list(self.alpha_dist.table),
            },
            "omega_dist": {
                "mode": self.omega_dist.mode,
                "table": list(self.omega_dist.table),
            },
            "use_interference": self.use_interference,
        }
        out.update(_fit_specs_to_dict(values, self.fit_specs))
        return out

    @classmethod
    def from_dict(cls, data: dict | None) -> Population:
        data = data or {}
        default = cls()
        defaults = {p: default.get_parameter(p) for p in POPULATION_PARAMETERS}
        values, specs = _read_parameters(data, defaults, POPULATION_PARAMETERS)
        pop = cls(
            name=str(data.get("name", "")),
            delta_rho=float(data.get("delta_rho", default.delta_rho)),
            size_dist=SizeDistribution.from_dict(data.get("size_dist")),
            alpha_dist=AngularDistribution.from_dict(data.get("alpha_dist")),
            omega_dist=AngularDistribution.from_dict(data.get("omega_dist")),
            use_interference=bool(data.get("use_interference", False)),
            fit_specs=specs,
        )
        for path, value in values.items():
            pop.set_parameter(path, value)
        return pop

    @property
    def orientation(self) -> OrientationDistribution:
        return OrientationDistribution(self.alpha_dist, self.omega_dist)


@dataclass
class DataSource:
    """Where a direction's measured data came from, so a project can reload them.

    The project file stores this reference, never the arrays. ``fingerprint`` is a SHA-256 of the
    file *content*, so a moved or copied file still matches and an edited one is detected.

    Attributes:
        path: data file (HDF5 or text). Resolved against the project folder when relative.
        internal_path: HDF5 group of the curve inside the file (``None`` for text files).
        variant: ``"default"``, ``"slit-smeared"``, ``"text"`` or ``"simple-hdf5"``.
        fingerprint: SHA-256 of the file content, hex.
        q_unit: unit of Q in a text file (ignored for NXcanSAS, which carries its own).
        error_fraction: relative uncertainty assumed for files that have none.
    """

    path: str = ""
    internal_path: str | None = None
    variant: str = "default"
    fingerprint: str = ""
    q_unit: str = "1/A"
    error_fraction: float = 0.05

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "internal_path": self.internal_path,
            "variant": self.variant,
            "fingerprint": self.fingerprint,
            "q_unit": self.q_unit,
            "error_fraction": self.error_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> DataSource | None:
        if not data or not data.get("path"):
            return None
        default = cls()
        return cls(
            path=str(data["path"]),
            internal_path=data.get("internal_path"),
            variant=str(data.get("variant", default.variant)),
            fingerprint=str(data.get("fingerprint", "")),
            q_unit=str(data.get("q_unit", default.q_unit)),
            error_fraction=float(data.get("error_fraction", default.error_fraction)),
        )


@dataclass
class Direction:
    """One measured 1D cut I(Q) along a direction of Q in the sample frame (Igor ``DirN_*``).

    Attributes:
        label: free-text label (e.g. the file name).
        alpha_q: polar angle of Q from the sample z axis [deg].
        omega_q: azimuth of Q from the sample x axis [deg].
        background: constant background [cm**-1]. In Igor it was subtracted from the data and
            could not be fitted; in Anisa it is added to the model and can be fitted.
        q, intensity, uncertainty: measured data (1/A, 1/cm, 1/cm); ``None`` until loaded.
        source: where the data came from (see :class:`DataSource`); ``None`` for data that was
            set directly, e.g. synthetic data.
        q_min, q_max: Q range used by the fit [1/A]; ``None`` means no limit on that side.
        use: include this direction in the fit.
        fit_specs: fit flags and limits (``background``).
    """

    label: str = ""
    alpha_q: float = 0.0
    omega_q: float = 0.0
    background: float = 0.0
    q: np.ndarray | None = None
    intensity: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    source: DataSource | None = None
    q_min: float | None = None
    q_max: float | None = None
    use: bool = True
    fit_specs: dict[str, FitSpec] = field(default_factory=dict)

    def fit_mask(self) -> np.ndarray:
        """Boolean mask of the data points the fit uses.

        A point is used if Q, intensity and uncertainty are finite, Q is positive and inside
        ``[q_min, q_max]``, and the uncertainty is positive (a point with zero uncertainty would
        have infinite weight). Returns an empty array when no data are loaded.
        """
        if self.q is None or self.intensity is None:
            return np.zeros(0, dtype=bool)
        q = np.asarray(self.q, dtype=float)
        mask = np.isfinite(q) & (q > 0) & np.isfinite(self.intensity)
        if self.uncertainty is not None:
            sigma = np.asarray(self.uncertainty, dtype=float)
            mask &= np.isfinite(sigma) & (sigma > 0)
        if self.q_min is not None:
            mask &= q >= self.q_min
        if self.q_max is not None:
            mask &= q <= self.q_max
        return mask

    def geometry_key(self) -> tuple:
        return (float(self.alpha_q), float(self.omega_q))

    def fit_spec(self, path: str = "background") -> FitSpec:
        if path not in self.fit_specs:
            self.fit_specs[path] = FitSpec.from_dict(None, path)
        return self.fit_specs[path]

    def to_dict(self) -> dict:
        """Settings and the data source. Data arrays are stored by the results writer, never in
        the project."""
        out = {
            "label": self.label,
            "alpha_q": self.alpha_q,
            "omega_q": self.omega_q,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "use": self.use,
            "source": self.source.to_dict() if self.source else None,
        }
        out.update(_fit_specs_to_dict({"background": self.background}, self.fit_specs))
        return out

    @classmethod
    def from_dict(cls, data: dict | None) -> Direction:
        data = data or {}
        values, specs = _read_parameters(data, {"background": 0.0}, ("background",))
        return cls(
            label=str(data.get("label", "")),
            alpha_q=float(data.get("alpha_q", 0.0)),
            omega_q=float(data.get("omega_q", 0.0)),
            background=values["background"],
            q_min=_num_or_none(data.get("q_min")),
            q_max=_num_or_none(data.get("q_max")),
            source=DataSource.from_dict(data.get("source")),
            use=bool(data.get("use", True)),
            fit_specs=specs,
        )


@dataclass
class CalcSettings:
    """How the model is evaluated (Igor ``UseOfFormula4``, ``IntegrationStepsIn*``, ``Wavelength``).

    Attributes:
        formula: ``"4c"`` (diffraction limit, default, fastest), ``"4a"`` or ``"4b"`` (with
            refraction; need ``wavelength``).
        n_alpha: Gauss-Legendre nodes in cos(alpha) for the orientation integral.
        n_omega: equally spaced nodes in omega.
        points_per_period: nodes per oscillation of the form factor along K; sets the K grid.
        wavelength: X-ray / neutron wavelength [A]; only used by formulas 4a and 4b.
        n_threads: worker threads; 0 means one per CPU.
    """

    formula: str = "4c"
    n_alpha: int = 64
    n_omega: int = 128
    points_per_period: int = 48
    wavelength: float = 1.0
    n_threads: int = 0

    def validate(self) -> None:
        if self.formula not in FORMULAS:
            raise ValueError(f"unknown formula {self.formula!r}; expected one of {FORMULAS}")
        if self.n_alpha < 2 or self.n_omega < 2:
            raise ValueError("n_alpha and n_omega must be at least 2")
        if self.points_per_period < 4:
            raise ValueError("points_per_period must be at least 4")
        if self.formula != "4c" and not self.wavelength > 0:
            raise ValueError("formulas 4a and 4b need a positive wavelength")

    def key(self) -> tuple:
        """Everything that changes a result (the thread count does not)."""
        wavelength = float(self.wavelength) if self.formula != "4c" else 0.0
        return (
            self.formula,
            int(self.n_alpha),
            int(self.n_omega),
            int(self.points_per_period),
            wavelength,
        )

    def to_dict(self) -> dict:
        return {
            "formula": self.formula,
            "n_alpha": self.n_alpha,
            "n_omega": self.n_omega,
            "points_per_period": self.points_per_period,
            "wavelength": self.wavelength,
            "n_threads": self.n_threads,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> CalcSettings:
        data = data or {}
        d = cls()
        return cls(
            formula=str(data.get("formula", d.formula)),
            n_alpha=int(data.get("n_alpha", d.n_alpha)),
            n_omega=int(data.get("n_omega", d.n_omega)),
            points_per_period=int(data.get("points_per_period", d.points_per_period)),
            wavelength=float(data.get("wavelength", d.wavelength)),
            n_threads=int(data.get("n_threads", d.n_threads)),
        )


@dataclass
class AnisaModel:
    """A complete model: any number of directions and populations, and the settings."""

    directions: list[Direction] = field(default_factory=list)
    populations: list[Population] = field(default_factory=list)
    settings: CalcSettings = field(default_factory=CalcSettings)
    scans: list = field(default_factory=list)  # list[AnisotropyScan], see anisa.core.anisotropy

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "settings": self.settings.to_dict(),
            "directions": [d.to_dict() for d in self.directions],
            "populations": [p.to_dict() for p in self.populations],
            "scans": [s.to_dict() for s in self.scans],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> AnisaModel:
        data = data or {}
        return cls(
            directions=[Direction.from_dict(d) for d in data.get("directions", [])],
            populations=[Population.from_dict(p) for p in data.get("populations", [])],
            settings=CalcSettings.from_dict(data.get("settings")),
            scans=[_scan_from_dict(x) for x in data.get("scans", [])],
        )


def _scan_from_dict(data: dict):
    from anisa.core.anisotropy import AnisotropyScan  # deferred: anisotropy imports this module

    return AnisotropyScan.from_dict(data)
