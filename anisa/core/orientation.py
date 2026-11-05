"""
Orientation distribution of the particle symmetry axes, P(alpha, omega) = p(alpha) b(omega).

This is the Igor ``PopN_AlphaDist`` / ``PopN_OmegaDist`` pair (Allen, eq. 3). Two input modes
exist for each factor, as in Igor:

* **parametric** -- with Par1 in degrees, Par2 >= 0 and Par3 >= 0::

      p(alpha) = Par3 + |cos(alpha - Par1)| ** Par2           alpha in [0, 90 deg]
      b(omega) = Par3 + |cos((omega - Par1) / 2)| ** Par2     omega in [0, 360 deg)

  Par1 is the position of the peak, Par2 its sharpness (Par2 = 0 gives a flat, isotropic
  distribution) and Par3 an isotropic floor.
* **table** -- ``n`` bin values on equal bins over [0, pi/2] (alpha) or [0, 2 pi) (omega),
  interpolated linearly between bin centres (constant beyond the first and last centre for
  alpha; periodic for omega).

Normalisation (differs from Igor, which used a 500x oversampled trapezoid for p and the
trapezoid rule for b): the weights are normalised **on the integration grid itself**, so the
numerical weights sum to exactly one for any distribution and any grid, and the intensity scale
cannot drift with the grid.

The weight of a grid node is the probability that a particle axis lies in the solid angle that
node represents. P is therefore a density per steradian over the upper hemisphere, which is
what a pole figure shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

HALF_PI = 0.5 * np.pi
TWO_PI = 2.0 * np.pi


@lru_cache(maxsize=16)
def _make_grid(n_alpha: int, n_omega: int) -> OrientationGrid:
    # Gauss-Legendre nodes are expensive for large n, so grids are cached.
    x, w = roots_legendre(n_alpha)  # ~6x faster than numpy's leggauss at n ~ 1000
    cos_alpha = 0.5 * (x + 1.0)  # [-1, 1] -> [0, 1]
    omega = (np.arange(n_omega) + 0.5) * (TWO_PI / n_omega)
    grid = OrientationGrid(
        cos_alpha=cos_alpha,
        alpha=np.arccos(cos_alpha),
        alpha_weight=0.5 * w,
        omega=omega,
        omega_weight=np.full(n_omega, 1.0 / n_omega),
    )
    for array in (grid.cos_alpha, grid.alpha, grid.alpha_weight, grid.omega, grid.omega_weight):
        array.flags.writeable = False
    return grid


@dataclass(frozen=True)
class OrientationGrid:
    """Quadrature grid over particle-axis orientations.

    ``cos(alpha)`` uses Gauss-Legendre nodes on [0, 1], which integrates the ``sin(alpha)``
    measure exactly; ``omega`` uses equally spaced mid-points on the periodic interval, which
    is spectrally accurate for smooth periodic integrands.

    Attributes:
        cos_alpha: nodes in cos(alpha), shape ``(n_alpha,)``, ascending.
        alpha: polar angles [rad] of those nodes.
        alpha_weight: Gauss-Legendre weights (sum to 1) for the integral over cos(alpha).
        omega: azimuths [rad], shape ``(n_omega,)``.
        omega_weight: equal weights (sum to 1).
    """

    cos_alpha: np.ndarray
    alpha: np.ndarray
    alpha_weight: np.ndarray
    omega: np.ndarray
    omega_weight: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.alpha.size, self.omega.size)

    @classmethod
    def make(cls, n_alpha: int, n_omega: int) -> OrientationGrid:
        """Grid with ``n_alpha`` x ``n_omega`` nodes. Grids are shared and read-only."""
        if n_alpha < 2 or n_omega < 2:
            raise ValueError("the orientation grid needs at least 2 nodes in alpha and omega")
        return _make_grid(int(n_alpha), int(n_omega))


@dataclass
class AngularDistribution:
    """One factor of the orientation distribution: ``p(alpha)`` or ``b(omega)``.

    Attributes:
        mode: ``"parametric"`` or ``"table"``.
        par1: peak position [deg] (parametric).
        par2: sharpness exponent, >= 0 (parametric); 0 means isotropic.
        par3: isotropic floor, >= 0 (parametric).
        table: bin values (table mode), at least 2, all >= 0 and not all 0.
    """

    mode: str = "parametric"
    par1: float = 0.0
    par2: float = 0.0
    par3: float = 0.0
    table: tuple[float, ...] = ()

    def validate(self) -> None:
        if self.mode == "parametric":
            if self.par2 < 0 or self.par3 < 0:
                raise ValueError("orientation parameters par2 and par3 must be >= 0")
        elif self.mode == "table":
            values = np.asarray(self.table, dtype=float)
            if values.size < 2:
                raise ValueError("an orientation table needs at least 2 bins")
            if np.any(values < 0) or not np.any(values > 0):
                raise ValueError("orientation table values must be >= 0 and not all zero")
        else:
            raise ValueError(f"unknown orientation mode {self.mode!r}")

    def required_nodes(self, role: str) -> int:
        """Nodes needed in ``alpha`` (``role="alpha"``) or ``omega`` to resolve this distribution.

        The weights are always normalised on the grid, so an under-resolved sharp peak does not
        fail: it silently becomes a broader one. These rules (checked in
        ``tests/test_orientation_resolution.py``) keep the quadrature error of moments of the
        orientation distribution below about 1e-4:

        * parametric, ``par2 = n``: the peak has width ``sigma ~ 1 / sqrt(n)`` rad in alpha and
          ``2 / sqrt(n)`` in omega. Omega (periodic trapezoid rule) converges exponentially and
          needs ``3 sqrt(n)`` nodes. In alpha, Gauss-Legendre in cos(alpha) is sparsest at the
          pole, where it needs ``14 sqrt(n)``; a peak at polar angle ``a0`` away from the pole
          needs about ``3 sqrt(n) / sin(a0)``.
        * table: 16 nodes per bin, since the linear interpolation has a kink at every bin centre.
        """
        if role not in ("alpha", "omega"):
            raise ValueError("role must be 'alpha' or 'omega'")
        if self.mode == "table":
            return 16 * len(self.table)
        root_n = math.sqrt(max(self.par2, 0.0))
        if role == "omega":
            return math.ceil(3.0 * root_n)
        sin_peak = abs(math.sin(math.radians(self.par1)))
        factor = 14.0 if sin_peak < 3.0 / 14.0 else 3.0 / sin_peak
        return math.ceil(factor * root_n)

    def key(self) -> tuple:
        """Hashable summary of everything that changes the distribution (cache key)."""
        if self.mode == "table":
            return ("table", tuple(float(v) for v in self.table))
        return ("parametric", float(self.par1), float(self.par2), float(self.par3))

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "par1": self.par1,
            "par2": self.par2,
            "par3": self.par3,
            "table": list(self.table),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> AngularDistribution:
        data = data or {}
        return cls(
            mode=str(data.get("mode", "parametric")),
            par1=float(data.get("par1", 0.0)),
            par2=float(data.get("par2", 0.0)),
            par3=float(data.get("par3", 0.0)),
            table=tuple(float(v) for v in data.get("table", ())),
        )

    # -- evaluation -----------------------------------------------------------------------

    def _table_values(self, angle: np.ndarray, period: float, periodic: bool) -> np.ndarray:
        values = np.asarray(self.table, dtype=float)
        n = values.size
        centres = (np.arange(n) + 0.5) * (period / n)
        if periodic:
            return np.interp(np.mod(angle, period), centres, values, period=period)
        return np.interp(angle, centres, values)

    def alpha_values(self, alpha: np.ndarray) -> np.ndarray:
        """Unnormalised ``p(alpha)`` at polar angles ``alpha`` [rad]."""
        alpha = np.asarray(alpha, dtype=float)
        if self.mode == "table":
            return self._table_values(alpha, HALF_PI, periodic=False)
        return self.par3 + np.abs(np.cos(alpha - np.radians(self.par1))) ** self.par2

    def omega_values(self, omega: np.ndarray) -> np.ndarray:
        """Unnormalised ``b(omega)`` at azimuths ``omega`` [rad]."""
        omega = np.asarray(omega, dtype=float)
        if self.mode == "table":
            return self._table_values(omega, TWO_PI, periodic=True)
        return self.par3 + np.abs(np.cos(0.5 * (omega - np.radians(self.par1)))) ** self.par2


@dataclass
class OrientationDistribution:
    """``P(alpha, omega) = p(alpha) b(omega)`` for one population."""

    alpha: AngularDistribution
    omega: AngularDistribution

    def __init__(
        self, alpha: AngularDistribution | None = None, omega: AngularDistribution | None = None
    ):
        self.alpha = alpha if alpha is not None else AngularDistribution()
        self.omega = omega if omega is not None else AngularDistribution()

    def key(self) -> tuple:
        return (self.alpha.key(), self.omega.key())

    def required_nodes(self) -> tuple[int, int]:
        """``(n_alpha, n_omega)`` needed to resolve the sharpness of this distribution."""
        return self.alpha.required_nodes("alpha"), self.omega.required_nodes("omega")

    def weights(self, grid: OrientationGrid) -> np.ndarray:
        """Node weights, shape ``grid.shape``, summing to exactly 1.

        ``W[i, j]`` is the probability that a particle axis lies in the part of the hemisphere
        represented by node ``(alpha_i, omega_j)``.
        """
        self.alpha.validate()
        self.omega.validate()
        p = self.alpha.alpha_values(grid.alpha) * grid.alpha_weight
        b = self.omega.omega_values(grid.omega) * grid.omega_weight
        total_p, total_b = p.sum(), b.sum()
        if not (total_p > 0 and total_b > 0 and np.isfinite(total_p * total_b)):
            raise ValueError("orientation distribution has no weight on the integration grid")
        return np.outer(p / total_p, b / total_b)

    def density(self, alpha: np.ndarray, omega: np.ndarray, *, n_alpha=256, n_omega=512):
        """Probability density per steradian P(alpha, omega) on the upper hemisphere.

        Normalised so that the integral over the hemisphere (solid angle element
        ``sin(alpha) d alpha d omega``) is 1. Used for orientation maps and pole figures.
        """
        grid = OrientationGrid.make(n_alpha, n_omega)
        p_norm = float(np.sum(self.alpha.alpha_values(grid.alpha) * grid.alpha_weight))
        b_norm = float(np.sum(self.omega.omega_values(grid.omega) * grid.omega_weight))
        # integral of p b over the hemisphere = (p_norm) (b_norm) 2 pi, since the GL weights
        # already carry the sin(alpha) measure and the omega weights sum to 1.
        scale = 1.0 / (p_norm * b_norm * TWO_PI)
        return self.alpha.alpha_values(alpha) * self.omega.omega_values(omega) * scale
