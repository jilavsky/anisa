"""
Azimuthal scans: the model intensity at fixed |Q| as the direction of Q is swept.

This is Igor's "model anisotropy" tool (``ASAS_IntAnisotropy``), without its limits (three fixed
orientations X, Y, Z with up to six Q values each). A scan sweeps one angle of Q over 0-360 deg while
the other is held fixed:

* ``sweep="omega"``: omega_Q runs 0-360 deg at fixed alpha_Q (a circle around the sample normal;
  at ``fixed = 90`` it is the azimuthal circle in the substrate plane, Igor's "X" orientation);
* ``sweep="alpha"``: alpha_Q runs 0-360 deg at fixed omega_Q (a great circle through the normal;
  Igor's "Y" and "Z" orientations are ``fixed = 0`` and ``fixed = 90``).

Each scan has any number of Q values; the result is I(angle) for every one.

**Measured data on the same path.** A measured direction is a cut along (alpha_Q, omega_Q), so
the cuts of a data set sample I(angle) at a fixed Q wherever they lie on a scan's path.
:func:`measured_points` interpolates every such cut at the scan's Q (log-log) and returns the
points, so the model can be compared with the data without a separate file format. Because
``I(Q) = I(-Q)`` (axes are undirected) each point also stands for the opposite direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from anisa.core import geometry
from anisa.core.engine import Engine
from anisa.core.model import AnisaModel, Direction

__all__ = [
    "AnisotropyResult",
    "AnisotropyScan",
    "MeasuredPoints",
    "calculate_scan",
    "measured_points",
]

#: A measured direction counts as lying on a scan path within this many degrees.
PATH_TOLERANCE_DEG = 1.0


@dataclass
class AnisotropyScan:
    """One azimuthal scan (see the module docstring).

    Attributes:
        label: free-text name.
        sweep: ``"omega"`` (omega_Q swept, alpha_Q fixed) or ``"alpha"`` (alpha_Q swept).
        fixed: the fixed angle [deg].
        q_values: |Q| values [1/A] at which I(angle) is calculated.
        n_points: number of angles from 0 to 360 deg inclusive (so a polar plot closes).
    """

    label: str = ""
    sweep: str = "omega"
    fixed: float = 90.0
    q_values: tuple[float, ...] = (0.002, 0.01)
    n_points: int = 73

    def validate(self) -> None:
        if self.sweep not in ("omega", "alpha"):
            raise ValueError("sweep must be 'omega' or 'alpha'")
        if self.n_points < 3:
            raise ValueError("a scan needs at least 3 points")
        if not self.q_values or any(not (q > 0 and math.isfinite(q)) for q in self.q_values):
            raise ValueError("a scan needs at least one positive Q value")

    def angles(self) -> np.ndarray:
        """The swept angle [deg], 0 to 360 inclusive."""
        return np.linspace(0.0, 360.0, int(self.n_points))

    def geometry(self) -> list[tuple[float, float]]:
        """``(alpha_q, omega_q)`` [deg] of every point of the scan."""
        if self.sweep == "omega":
            return [(float(self.fixed), float(a)) for a in self.angles()]
        return [(float(a), float(self.fixed)) for a in self.angles()]

    def title(self) -> str:
        if self.label:
            return self.label
        swept, fixed = ("ω_Q", "α_Q") if self.sweep == "omega" else ("α_Q", "ω_Q")
        return f"{swept} scan at {fixed} = {self.fixed:g}°"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "sweep": self.sweep,
            "fixed": self.fixed,
            "q_values": list(self.q_values),
            "n_points": self.n_points,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> AnisotropyScan:
        data = data or {}
        default = cls()
        return cls(
            label=str(data.get("label", "")),
            sweep=str(data.get("sweep", default.sweep)),
            fixed=float(data.get("fixed", default.fixed)),
            q_values=tuple(float(q) for q in data.get("q_values", default.q_values)),
            n_points=int(data.get("n_points", default.n_points)),
        )


@dataclass
class AnisotropyResult:
    """Model of one scan.

    Attributes:
        scan: the scan.
        angles: swept angle [deg], shape ``(n_angles,)``.
        intensity: total model intensity [1/cm] (no background), shape ``(n_q, n_angles)``.
        populations: each population's contribution, shape ``(n_populations, n_q, n_angles)``.
    """

    scan: AnisotropyScan
    angles: np.ndarray
    intensity: np.ndarray
    populations: np.ndarray


@dataclass
class MeasuredPoints:
    """Measured intensity at one Q on a scan's path.

    ``angles`` [deg] and ``intensity`` [1/cm, background subtracted] are sorted by angle;
    ``sigma`` is the uncertainty interpolated like the intensity; ``labels`` names the direction
    each point comes from.
    """

    angles: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    labels: list[str]


def calculate_scan(
    model: AnisaModel, scan: AnisotropyScan, engine: Engine | None = None
) -> AnisotropyResult:
    """Model intensity along ``scan``: every population, every Q, every angle.

    The model's directions and backgrounds are not used; its populations and settings are.
    """
    scan.validate()
    engine = engine or Engine()
    q = np.asarray(scan.q_values, dtype=float)
    geometry = scan.geometry()
    probe = AnisaModel(
        directions=[Direction(alpha_q=a, omega_q=o) for a, o in geometry],
        populations=model.populations,
        settings=model.settings,
    )
    result = engine.evaluate(probe, [q] * len(geometry))
    n_pop = len(model.populations)
    populations = np.zeros((n_pop, q.size, len(geometry)))
    for k, direction in enumerate(result.directions):
        populations[:, :, k] = direction.populations
    return AnisotropyResult(
        scan=scan,
        angles=scan.angles(),
        intensity=populations.sum(axis=0),
        populations=populations,
    )


def _swept_angles(scan: AnisotropyScan, direction: Direction) -> list[float]:
    """Swept-angle values [deg] at which ``direction`` lies on the scan path.

    The direction ``u`` and its inverse ``-u`` (same intensity) are each tested against the
    scan's circle, with a tolerance of :data:`PATH_TOLERANCE_DEG`, using vectors rather than
    case analysis:

    * omega scan at alpha_Q = a: the circle ``z . v = cos a``; the angle is ``atan2(v_y, v_x)``;
    * alpha scan at omega_Q = w: the great circle ``v . n = 0`` with ``n`` perpendicular to the
      plane of the normal and azimuth ``w``; the angle is ``atan2(v . e, v_z)``, ``e`` being the
      in-plane unit vector at azimuth ``w``.

    A scan whose circle degenerates to a point (omega scan at a pole) matches nothing.
    """
    u = geometry.q_unit_vector_deg(direction.alpha_q, direction.omega_q)
    tolerance = math.radians(PATH_TOLERANCE_DEG)
    found: list[float] = []
    for v in (u, -u):
        if scan.sweep == "omega":
            alpha0 = math.acos(math.cos(math.radians(scan.fixed)))  # folded into [0, 180]
            if math.sin(alpha0) < 1e-6:
                continue
            if abs(math.acos(float(np.clip(v[2], -1.0, 1.0))) - alpha0) <= tolerance:
                found.append(math.degrees(math.atan2(v[1], v[0])) % 360.0)
        else:
            w = math.radians(scan.fixed)
            e = np.array([math.cos(w), math.sin(w), 0.0])
            n = np.array([-math.sin(w), math.cos(w), 0.0])
            if abs(float(v @ n)) <= math.sin(tolerance):
                found.append(math.degrees(math.atan2(float(v @ e), float(v[2]))) % 360.0)
    unique: list[float] = []
    for angle in sorted(found):
        if not unique or abs(angle - unique[-1]) > 0.01:
            unique.append(angle)
    if len(unique) > 1 and abs(unique[0] + 360.0 - unique[-1]) <= 0.01:
        unique.pop()  # 0 and 360 are the same point
    return unique


def measured_points(model: AnisaModel, scan: AnisotropyScan, q: float) -> MeasuredPoints:
    """Measured intensities of the directions that lie on ``scan``'s path, at ``|Q| = q``.

    Each direction with data (and ``use`` set) is interpolated at ``q`` (log-log, inside its Q
    range only), its background subtracted, and placed at the swept angle where it lies on the
    path (and at the opposite direction, which has the same intensity).
    """
    angles: list[float] = []
    values: list[float] = []
    sigmas: list[float] = []
    labels: list[str] = []
    for index, d in enumerate(model.directions):
        if not d.use or d.q is None or d.intensity is None:
            continue
        positions = _swept_angles(scan, d)
        if not positions:
            continue
        mask = d.fit_mask() if d.uncertainty is not None else np.isfinite(d.q) & (d.q > 0)
        mask &= np.asarray(d.intensity) > 0
        qd, idat = np.asarray(d.q)[mask], np.asarray(d.intensity)[mask]
        if qd.size < 2 or not qd.min() <= q <= qd.max():
            continue
        order = np.argsort(qd)
        value = float(np.exp(np.interp(np.log(q), np.log(qd[order]), np.log(idat[order]))))
        sigma = math.nan
        if d.uncertainty is not None:
            sd = np.asarray(d.uncertainty)[mask]
            sigma = float(np.interp(np.log(q), np.log(qd[order]), sd[order]))
        value -= d.background
        for angle in positions:
            angles.append(angle)
            values.append(value)
            sigmas.append(sigma)
            labels.append(d.label or f"Dir {index + 1}")
    order = np.argsort(angles)
    return MeasuredPoints(
        angles=np.asarray(angles, dtype=float)[order],
        intensity=np.asarray(values, dtype=float)[order],
        sigma=np.asarray(sigmas, dtype=float)[order],
        labels=[labels[i] for i in order],
    )
