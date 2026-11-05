"""
The orientation grid must resolve the orientation distribution, not only the form factor.

Weights are normalised on the grid, so an under-resolved sharp peak does not fail loudly: it
silently becomes a broader peak. ``AngularDistribution.required_nodes`` sets a floor on the grid
from the sharpness of each distribution; these tests check the floor against a fine reference
and show what happens without it. Moments of X = cos(eta) with even powers are used because the
cross sections depend on X only through X**2, where the integrand is analytic (<|X|> has a kink
and converges more slowly, which says nothing about the engine).
"""

from __future__ import annotations

import numpy as np
import pytest

from anisa.core import geometry
from anisa.core.engine import resolved_grid_size
from anisa.core.model import CalcSettings
from anisa.core.orientation import AngularDistribution, OrientationDistribution, OrientationGrid

DIRECTIONS = [(0.0, 0.0), (40.0, 20.0), (90.0, 70.0), (20.0, 100.0)]


def moments(dist, n_alpha, n_omega, direction):
    grid = OrientationGrid.make(n_alpha, n_omega)
    w = dist.weights(grid)
    x = geometry.cos_eta(
        geometry.q_unit_vector_deg(*direction), grid.alpha[:, None], grid.omega[None, :]
    )
    return np.array([np.sum(w * x**2), np.sum(w * x**4), np.sum(w * x**6)])


def worst_error(dist, n_alpha, n_omega, ref_alpha, ref_omega):
    errors = []
    for direction in DIRECTIONS:
        ref = moments(dist, ref_alpha, ref_omega, direction)
        errors.append(np.max(np.abs(moments(dist, n_alpha, n_omega, direction) - ref) / ref))
    return max(errors)


@pytest.mark.parametrize("peak", [0.0, 10.0, 30.0, 70.0, 90.0])
@pytest.mark.parametrize("par2", [30.0, 3000.0, 30000.0])
def test_alpha_floor_resolves_sharp_peaks(peak, par2):
    dist = OrientationDistribution(AngularDistribution(par1=peak, par2=par2), AngularDistribution())
    n_alpha = resolved_grid_size(CalcSettings(), 0.0, dist.required_nodes())[0]
    assert worst_error(dist, n_alpha, 8, 3000, 8) < 1e-4


@pytest.mark.parametrize("par2", [30.0, 3000.0, 30000.0])
def test_omega_floor_resolves_sharp_peaks(par2):
    dist = OrientationDistribution(
        AngularDistribution(par1=40.0, par2=2.0), AngularDistribution(par1=130.0, par2=par2)
    )
    n_omega = resolved_grid_size(CalcSettings(), 0.0, dist.required_nodes())[1]
    assert worst_error(dist, 16, n_omega, 16, 12000) < 1e-4


def test_without_the_floor_a_sharp_pole_peak_is_silently_broadened():
    """The failure the floor prevents: par2 = 20000 has sigma = 0.007 rad, far below the node
    spacing of a 64-node Gauss-Legendre grid at the pole, so the grid's <alpha^2> is ~7x too
    large and nothing complains."""
    dist = OrientationDistribution(
        AngularDistribution(par1=0.0, par2=20000.0), AngularDistribution()
    )
    grid = OrientationGrid.make(64, 16)
    w = dist.weights(grid).sum(axis=1)
    assert w.sum() == pytest.approx(1.0)  # still "normalised"
    assert np.sum(w * grid.alpha**2) > 5 * (2.0 / 20000.0)  # analytic <alpha^2> = 2 / par2
    # and with the floor it is right
    n_alpha = resolved_grid_size(CalcSettings(), 0.0, dist.required_nodes())[0]
    grid = OrientationGrid.make(n_alpha, 16)
    w = dist.weights(grid).sum(axis=1)
    assert np.sum(w * grid.alpha**2) == pytest.approx(2.0 / 20000.0, rel=1e-2)


def test_table_floor_scales_with_the_number_of_bins():
    table = AngularDistribution(mode="table", table=tuple(range(1, 11)))
    assert table.required_nodes("alpha") == 160
    assert table.required_nodes("omega") == 160


def test_smooth_distributions_stay_at_the_user_defaults():
    flat = OrientationDistribution()
    assert resolved_grid_size(CalcSettings(), 0.0, flat.required_nodes()) == (64, 128)


def test_required_nodes_validates_role():
    with pytest.raises(ValueError):
        AngularDistribution().required_nodes("polar")
