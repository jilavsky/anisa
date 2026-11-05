"""
Geometry of the sample frame.

Conventions (Allen, "Scattering models for SBUSAXS and MSANS studies of coatings", 2001):

* The sample frame has its z axis along the substrate normal **n**.
* A direction is given by a polar angle ``alpha`` from z and an azimuth ``omega`` from x,
  so the unit vector is ``(sin(alpha) cos(omega), sin(alpha) sin(omega), cos(alpha))``.
* The scattering vector Q points along (``alpha_q``, ``omega_q``) in this frame. Q is taken to
  lie in the sample plane, i.e. the small tilt of Q out of the detector plane (phi_s / 2) is
  ignored, as in Allen's notes.
* A particle's symmetry axis points along (``alpha``, ``omega``). Axes are undirected, so
  ``alpha`` in [0, pi/2] and ``omega`` in [0, 2 pi) cover every axis exactly once.
* ``X = cos(eta)`` is the cosine of the angle between the particle axis and Q (Allen, eq. 1):
  ``X = sin(alpha_q) sin(alpha) cos(omega - omega_q) + cos(alpha_q) cos(alpha)``.

Every function takes **radians** unless the name says ``_deg``. The GUI and the JSON project
file use degrees; the conversion happens at the model boundary (``anisa.core.model``).

The engine works with unit vectors of Q rather than with angles, so measurement geometries that
are not a single (alpha_q, omega_q) -- slit-smeared data and 2D detector images, see PLAN.md
sections 10 and 11 -- only need a different way to produce those vectors.
"""

from __future__ import annotations

import numpy as np


def unit_vector(alpha, omega) -> np.ndarray:
    """Unit vector(s) for polar angle ``alpha`` and azimuth ``omega`` [rad].

    Args:
        alpha: polar angle(s) from the sample z axis.
        omega: azimuth(s) from the sample x axis. Broadcast against ``alpha``.

    Returns:
        Array of shape ``(..., 3)``.
    """
    alpha = np.asarray(alpha, dtype=float)
    omega = np.asarray(omega, dtype=float)
    alpha, omega = np.broadcast_arrays(alpha, omega)
    sin_a = np.sin(alpha)
    return np.stack((sin_a * np.cos(omega), sin_a * np.sin(omega), np.cos(alpha)), axis=-1)


def q_unit_vector_deg(alpha_q_deg, omega_q_deg) -> np.ndarray:
    """Unit vector of Q for angles given in degrees (the GUI / JSON convention)."""
    return unit_vector(np.radians(alpha_q_deg), np.radians(omega_q_deg))


def angles_from_unit_vector(vec) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`unit_vector`: returns ``(alpha, omega)`` in radians.

    ``omega`` is returned in [0, 2 pi). It is arbitrary (0) where the vector lies on the z axis.
    """
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec, axis=-1)
    if np.any(norm == 0.0):
        raise ValueError("cannot take the direction of a zero vector")
    alpha = np.arccos(np.clip(vec[..., 2] / norm, -1.0, 1.0))
    omega = np.mod(np.arctan2(vec[..., 1], vec[..., 0]), 2.0 * np.pi)
    return alpha, omega


def cos_eta(q_hat, alpha, omega) -> np.ndarray:
    """``X = cos(eta)`` between Q and particle axes (Allen, eq. 1).

    Args:
        q_hat: unit vector of Q in the sample frame, shape ``(3,)``.
        alpha: polar angle(s) of the particle axis [rad].
        omega: azimuth(s) of the particle axis [rad], broadcast against ``alpha``.

    Returns:
        Array with the broadcast shape of ``alpha`` and ``omega``.
    """
    q_hat = np.asarray(q_hat, dtype=float)
    if q_hat.shape != (3,):
        raise ValueError(f"q_hat must have shape (3,), got {q_hat.shape}")
    return np.clip(unit_vector(alpha, omega) @ q_hat, -1.0, 1.0)
