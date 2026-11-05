"""
Size distribution of the equatorial radius of one population.

Replaces Igor AnisoSAS's two ad hoc distributions (a 9-point triangle and a truncated "Gauss"
whose FWHM was really a 1/e half-width) by the Irena distributions in
``pyirena.core.distributions``: Gauss, log-normal, Schulz-Zimm, LSW and Ardell, with the same
parameter names as Irena Modeling. (Anisa integrates over radius with a uniform mid-point grid,
not Irena's equal-probability grid; see :func:`radius_classes`.)

The population's ``radius`` is the location parameter of the distribution (``mean_size`` for
Gauss, log-normal and Schulz-Zimm; ``location`` for LSW and Ardell), so fitting the radius moves
the whole distribution. The remaining parameters live in ``params``:

========== =====================================
type       ``params``
========== =====================================
none       (monodisperse; no parameters)
gauss      ``width`` [A]
lognormal  ``min_size`` [A], ``sdeviation``
schulz_zimm ``width`` [A]
lsw        (none)
ardell     ``parameter`` (2 < m < 3)
========== =====================================

``weighting`` says what the distribution describes, as in Irena Modeling (``use_number_dist``):
``"volume"`` (default) means the distribution is the volume fraction per radius bin;
``"number"`` means the number of particles per radius bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from pyirena.core import distributions as irena_dist

from anisa.core.formfactor import spheroid_volume

DISTRIBUTION_TYPES = ("none", "gauss", "lognormal", "schulz_zimm", "lsw", "ardell")

#: Parameters (besides the radius) each type needs, with defaults.
PARAMETER_DEFAULTS: dict[str, dict[str, float]] = {
    "none": {},
    "gauss": {"width": 100.0},
    "lognormal": {"min_size": 10.0, "sdeviation": 0.3},
    "schulz_zimm": {"width": 100.0},
    "lsw": {},
    "ardell": {"parameter": 2.5},
}

#: Name Irena gives the location parameter of each type.
_LOCATION_NAME = {
    "gauss": "mean_size",
    "lognormal": "mean_size",
    "schulz_zimm": "mean_size",
    "lsw": "location",
    "ardell": "location",
}


@dataclass
class SizeDistribution:
    """Radius distribution of a population (see the module docstring)."""

    type: str = "none"
    params: dict[str, float] = field(default_factory=dict)
    n_bins: int = 100
    weighting: str = "volume"

    def validate(self) -> None:
        if self.type not in DISTRIBUTION_TYPES:
            raise ValueError(f"unknown size distribution {self.type!r}; use {DISTRIBUTION_TYPES}")
        if self.weighting not in ("volume", "number"):
            raise ValueError("size distribution weighting must be 'volume' or 'number'")
        if self.type != "none" and self.n_bins < 2:
            raise ValueError("a size distribution needs at least 2 radius bins")

    def full_params(self) -> dict[str, float]:
        """``params`` with the defaults filled in for every parameter the type needs."""
        merged = dict(PARAMETER_DEFAULTS[self.type])
        merged.update({k: float(v) for k, v in self.params.items() if k in merged})
        return merged

    def key(self) -> tuple:
        return (
            self.type,
            tuple(sorted(self.full_params().items())),
            int(self.n_bins),
            self.weighting,
        )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "params": self.full_params(),
            "n_bins": self.n_bins,
            "weighting": self.weighting,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> SizeDistribution:
        data = data or {}
        return cls(
            type=str(data.get("type", "none")),
            params={k: float(v) for k, v in dict(data.get("params", {})).items()},
            n_bins=int(data.get("n_bins", 100)),
            weighting=str(data.get("weighting", "volume")),
        )


#: Probability omitted in each tail when the radius range is found.
DEFAULT_TAIL_PRECISION = 1e-5


@lru_cache(maxsize=256)
def _range(dist_type: str, items: tuple, tail: float) -> tuple[float, float]:
    params = dict(items)
    grid = irena_dist.generate_radius_grid(dist_type, params, n_bins=60, tail_precision=tail)
    lo, hi = float(grid[0]), float(grid[-1])
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError(f"size distribution {dist_type!r} has no usable radius range")
    return lo, hi


def _located_params(dist: SizeDistribution, radius: float) -> tuple:
    params = dist.full_params()
    params[_LOCATION_NAME[dist.type]] = float(radius)
    return tuple(sorted(params.items()))


def radius_range(
    dist: SizeDistribution, radius: float, tail_precision: float = DEFAULT_TAIL_PRECISION
) -> tuple[float, float]:
    """Radii ``(lo, hi)`` [A] between which all but ``2 * tail_precision`` of the distribution
    lies. A monodisperse population returns ``(radius, radius)``."""
    dist.validate()
    if radius <= 0:
        raise ValueError("radius must be positive")
    if dist.type == "none":
        return float(radius), float(radius)
    return _range(dist.type, _located_params(dist, radius), float(tail_precision))


def volume_density(dist: SizeDistribution, radius: float, beta: float, r) -> np.ndarray:
    """Unnormalised density of the **volume fraction** over radius, evaluated at radii ``r``.

    For a volume distribution this is the distribution itself; for a number distribution it is
    multiplied by the particle volume. Only ratios matter: callers normalise on their own grid.
    """
    r = np.asarray(r, dtype=float)
    if dist.type == "none":
        raise ValueError("a monodisperse population has no density")
    items = _located_params(dist, radius)
    density = np.asarray(irena_dist.pdf(dist.type, r, dict(items)), dtype=float)
    if dist.weighting == "number":
        density = density * spheroid_volume(r, beta)
    return density


def radius_classes(
    dist: SizeDistribution,
    radius: float,
    beta: float,
    n: int | None = None,
    tail_precision: float = DEFAULT_TAIL_PRECISION,
) -> tuple[np.ndarray, np.ndarray]:
    """Radii and **volume-fraction weights** of a population's size classes.

    The classes are the mid-points of ``n`` equal cells between :func:`radius_range` limits. On
    a smooth distribution the mid-point rule converges faster than any power of ``n`` once the
    cells are narrower than half a period of the form-factor oscillation (``pi / Q``), which is
    why this replaces Irena's equal-probability grid (made for its NNLS G-matrix, it converges
    only to first order as a quadrature).

    Args:
        dist: the size distribution.
        radius: the population radius R [A]; the location parameter of the distribution.
        beta: aspect ratio (needed to turn a number distribution into volume weights).
        n: number of classes; defaults to ``dist.n_bins``.
        tail_precision: probability omitted in each tail.

    Returns:
        ``(radii, weights)``, each of shape ``(n,)``. ``weights`` sum to exactly 1 and give the
        share of the population's volume fraction carried by each class. A monodisperse
        population returns ``([radius], [1.0])``.
    """
    dist.validate()
    if dist.type == "none":
        return np.array([float(radius)]), np.array([1.0])
    n = int(dist.n_bins if n is None else n)
    lo, hi = radius_range(dist, radius, tail_precision)
    cell = (hi - lo) / n
    radii = lo + (np.arange(n) + 0.5) * cell
    density = volume_density(dist, radius, beta, radii)
    total = density.sum()
    if not (total > 0 and np.isfinite(total)):
        raise ValueError(f"size distribution {dist.type!r} has no weight on its radius grid")
    return radii, density / total
