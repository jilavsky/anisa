"""Geometry (Allen eq. 1) and orientation distributions."""

from __future__ import annotations

import numpy as np
import pytest

from anisa.core import geometry
from anisa.core.orientation import (
    AngularDistribution,
    OrientationDistribution,
    OrientationGrid,
)

# -- geometry ---------------------------------------------------------------------------------


def test_unit_vector_round_trip():
    alpha = np.radians([0.0, 10.0, 45.0, 90.0, 120.0])
    omega = np.radians([0.0, 30.0, 200.0, 359.0, 90.0])
    a2, o2 = geometry.angles_from_unit_vector(geometry.unit_vector(alpha, omega))
    np.testing.assert_allclose(a2, alpha, atol=1e-12)
    # omega is undefined on the z axis (alpha = 0)
    np.testing.assert_allclose(np.cos(o2[1:]), np.cos(omega[1:]), atol=1e-12)


def test_cos_eta_matches_allen_equation_1():
    """cos eta = cos(wQ) cos(w) sin(aQ) sin(a) + sin(wQ) sin(w) sin(aQ) sin(a) + cos(aQ) cos(a)."""
    rng = np.random.default_rng(3)
    a, w = rng.uniform(0, np.pi / 2, 50), rng.uniform(0, 2 * np.pi, 50)
    for aq, wq in ((0.3, 1.1), (1.5, 4.0), (0.0, 0.0), (np.pi / 2, 2.0)):
        q_hat = geometry.unit_vector(aq, wq)
        expected = (
            np.cos(wq) * np.cos(w) * np.sin(aq) * np.sin(a)
            + np.sin(wq) * np.sin(w) * np.sin(aq) * np.sin(a)
            + np.cos(aq) * np.cos(a)
        )
        np.testing.assert_allclose(geometry.cos_eta(q_hat, a, w), expected, atol=1e-14)


def test_cos_eta_special_cases():
    z = geometry.q_unit_vector_deg(0.0, 0.0)
    assert geometry.cos_eta(z, 0.0, 0.0) == pytest.approx(1.0)  # Q along z, axis along z
    assert geometry.cos_eta(z, np.pi / 2, 1.0) == pytest.approx(0.0, abs=1e-15)  # axis in plane
    x = geometry.q_unit_vector_deg(90.0, 0.0)
    assert geometry.cos_eta(x, np.pi / 2, 0.0) == pytest.approx(1.0)  # both along x


def test_cos_eta_rejects_bad_q_hat():
    with pytest.raises(ValueError):
        geometry.cos_eta(np.array([1.0, 0.0]), 0.1, 0.1)


# -- orientation grid and weights -------------------------------------------------------------


@pytest.mark.parametrize("shape", [(8, 8), (64, 128), (200, 33)])
def test_grid_weights_sum_to_one(shape):
    grid = OrientationGrid.make(*shape)
    assert grid.shape == shape
    assert grid.alpha_weight.sum() == pytest.approx(1.0, abs=1e-14)
    assert grid.omega_weight.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.all((grid.alpha > 0) & (grid.alpha < np.pi / 2))
    assert np.all((grid.omega > 0) & (grid.omega < 2 * np.pi))


def test_grid_is_shared_and_read_only():
    assert OrientationGrid.make(16, 32) is OrientationGrid.make(16, 32)
    with pytest.raises(ValueError):
        OrientationGrid.make(16, 32).alpha[0] = 0.0


def test_grid_integrates_solid_angle_exactly():
    """With a flat distribution, E[cos(alpha)] over the hemisphere is exactly 1/2."""
    grid = OrientationGrid.make(32, 64)
    w = OrientationDistribution().weights(grid)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.sum(w * grid.cos_alpha[:, None]) == pytest.approx(0.5, abs=1e-14)


@pytest.mark.parametrize(
    "alpha_dist, omega_dist",
    [
        (AngularDistribution(), AngularDistribution()),
        (AngularDistribution(par1=30, par2=10, par3=0.1), AngularDistribution(par1=200, par2=3)),
        (AngularDistribution(mode="table", table=(1, 4, 2, 0.5)), AngularDistribution()),
        (AngularDistribution(), AngularDistribution(mode="table", table=(0, 1, 3, 1, 0, 0))),
    ],
)
def test_weights_normalised_exactly_for_any_distribution(alpha_dist, omega_dist):
    grid = OrientationGrid.make(48, 96)
    w = OrientationDistribution(alpha_dist, omega_dist).weights(grid)
    assert w.shape == grid.shape
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.all(w >= 0)


def test_isotropic_second_moment_is_one_third_for_any_q_direction():
    """For isotropic axes <cos^2 eta> = 1/3, whatever the direction of Q."""
    grid = OrientationGrid.make(40, 80)
    w = OrientationDistribution().weights(grid)
    for aq, wq in ((0.0, 0.0), (37.0, 111.0), (90.0, 250.0)):
        q_hat = geometry.q_unit_vector_deg(aq, wq)
        x = geometry.cos_eta(q_hat, grid.alpha[:, None], grid.omega[None, :])
        assert np.sum(w * x**2) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_parametric_alpha_peaks_where_par1_says():
    grid = OrientationGrid.make(200, 16)
    for peak in (0.0, 40.0, 90.0):
        dist = AngularDistribution(par1=peak, par2=30)
        p = dist.alpha_values(grid.alpha)
        assert np.degrees(grid.alpha[np.argmax(p)]) == pytest.approx(peak, abs=2.0)


def test_parametric_omega_peaks_where_par1_says_and_is_periodic():
    omega = np.linspace(0, 2 * np.pi, 721)
    dist = AngularDistribution(par1=100.0, par2=8)
    b = dist.omega_values(omega)
    assert np.degrees(omega[np.argmax(b)]) == pytest.approx(100.0, abs=1.0)
    assert b[0] == pytest.approx(b[-1])  # periodic


def test_par2_zero_is_isotropic_including_where_cosine_vanishes():
    alpha = np.array([0.0, np.pi / 2, 0.3])
    np.testing.assert_allclose(AngularDistribution(par1=90, par2=0).alpha_values(alpha), 1.0)


def test_table_interpolates_between_bin_centres_and_wraps_in_omega():
    dist = AngularDistribution(mode="table", table=(0.0, 1.0, 0.0, 0.0))
    centres = (np.arange(4) + 0.5) * 2 * np.pi / 4
    np.testing.assert_allclose(dist.omega_values(centres), [0, 1, 0, 0], atol=1e-14)
    # halfway between the last centre (0) and the first (0), across the 2 pi seam
    assert dist.omega_values(np.array([0.0]))[0] == pytest.approx(0.0)
    assert dist.omega_values(np.array([2 * np.pi]))[0] == pytest.approx(dist.omega_values(0.0))


@pytest.mark.parametrize(
    "bad",
    [
        AngularDistribution(par2=-1.0),
        AngularDistribution(par3=-0.1),
        AngularDistribution(mode="table", table=(1.0,)),
        AngularDistribution(mode="table", table=(0.0, 0.0)),
        AngularDistribution(mode="table", table=(1.0, -1.0)),
        AngularDistribution(mode="bogus"),
    ],
)
def test_invalid_distributions_are_rejected(bad):
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        OrientationDistribution(bad, AngularDistribution()).weights(OrientationGrid.make(8, 8))


def test_density_integrates_to_one_over_the_hemisphere():
    dist = OrientationDistribution(
        AngularDistribution(par1=30, par2=6, par3=0.1), AngularDistribution(par1=60, par2=2)
    )
    n_a, n_o = 400, 800
    c = (np.arange(n_a) + 0.5) / n_a  # midpoint in cos(alpha): flat measure for solid angle
    alpha = np.arccos(c)
    omega = (np.arange(n_o) + 0.5) * 2 * np.pi / n_o
    density = dist.density(alpha[:, None], omega[None, :])
    total = density.sum() * (1.0 / n_a) * (2 * np.pi / n_o)  # d(omega) d(cos alpha)
    assert total == pytest.approx(1.0, abs=1e-4)


def test_distribution_dict_round_trip_and_defaults():
    d = AngularDistribution(par1=12.5, par2=3.0, par3=0.2)
    assert AngularDistribution.from_dict(d.to_dict()) == d
    t = AngularDistribution(mode="table", table=(1.0, 2.0, 3.0))
    assert AngularDistribution.from_dict(t.to_dict()) == t
    assert AngularDistribution.from_dict(None) == AngularDistribution()
    assert AngularDistribution.from_dict({"par2": 4}).par2 == 4.0


def test_cache_keys_distinguish_distributions():
    a = AngularDistribution(par1=1, par2=2, par3=3)
    assert a.key() == AngularDistribution(par1=1, par2=2, par3=3).key()
    assert a.key() != AngularDistribution(par1=1, par2=2, par3=3.1).key()
    assert a.key() != AngularDistribution(mode="table", table=(1.0, 2.0)).key()
