"""
Quantities derived from a population: surface area, number density and mean volume.

Units follow the Igor package: surface area in cm**2 / cm**3 (= cm**-1), number density in
cm**-3, volumes in A**3. Size distributions are included: each radius class carries a share
``v`` of the population's volume fraction ``phi`` (see ``anisa.core.sizedist``), so a class has
number density ``phi v / V`` and surface density ``phi v S / V``.
"""

from __future__ import annotations

import numpy as np

from anisa.core.formfactor import spheroid_volume
from anisa.core.model import Population
from anisa.core.sizedist import radius_classes

#: 1 / A -> 1 / cm
PER_ANGSTROM_TO_PER_CM = 1.0e8
#: 1 / A**3 -> 1 / cm**3
PER_A3_TO_PER_CM3 = 1.0e24


def spheroid_surface_area(radius, beta: float):
    """Surface area [A**2] of a spheroid with semi-axes (R, R, beta R).

    Oblate (beta < 1): ``2 pi R**2 (1 + beta**2 / e ln((1 + e) / beta))``, ``e = sqrt(1 - beta**2)``.
    Prolate (beta > 1): ``2 pi R**2 (1 + beta / e asin(e))``, ``e = sqrt(1 - 1/beta**2)``.
    A sphere (beta = 1) gives ``4 pi R**2``. (Igor's ``ASAS_CalcOnePopSfcArea`` writes the same
    expression as ``4 pi beta chi R**2``.)
    """
    radius = np.asarray(radius, dtype=float)
    if abs(beta - 1.0) < 1e-9:
        g = 1.0  # limit of both expressions below
    elif beta < 1.0:
        e = np.sqrt(1.0 - beta * beta)
        g = beta * beta / e * np.log((1.0 + e) / beta)
    else:
        e = np.sqrt(1.0 - 1.0 / (beta * beta))
        g = beta / e * np.arcsin(e)
    return 2.0 * np.pi * radius**2 * (1.0 + g)


def surface_area(pop: Population) -> float:
    """Total surface area of the population [cm**2 / cm**3]."""
    radii, share = radius_classes(pop.size_dist, pop.radius, pop.beta)
    per_particle = spheroid_surface_area(radii, pop.beta) / spheroid_volume(radii, pop.beta)
    return float(PER_ANGSTROM_TO_PER_CM * pop.volume_fraction * np.sum(share * per_particle))


def number_density(pop: Population) -> float:
    """Number of particles per cm**3 of sample."""
    radii, share = radius_classes(pop.size_dist, pop.radius, pop.beta)
    return float(
        PER_A3_TO_PER_CM3 * pop.volume_fraction * np.sum(share / spheroid_volume(radii, pop.beta))
    )


def mean_volume(pop: Population) -> float:
    """Number-averaged particle volume [A**3] (= phi / number density, converted)."""
    radii, share = radius_classes(pop.size_dist, pop.radius, pop.beta)
    return float(1.0 / np.sum(share / spheroid_volume(radii, pop.beta)))
