"""Azimuthal scans: geometry, symmetries, agreement with directions and with measured cuts."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from anisa import demo
from anisa.core.anisotropy import (
    AnisotropyScan,
    calculate_scan,
    measured_points,
)
from anisa.core.engine import Engine
from anisa.core.model import AnisaModel, Direction, Population
from anisa.core.orientation import AngularDistribution
from anisa.io.project import load_project, save_project


def disks(**kwargs) -> Population:
    """Flat disks with their axes along the sample normal."""
    return Population(
        radius=1000.0,
        beta=0.1,
        volume_fraction=0.05,
        delta_rho=2e11,
        alpha_dist=AngularDistribution(par1=0.0, par2=40.0),
        **kwargs,
    )


# -- the scan object --------------------------------------------------------------------------


def test_scan_angles_and_geometry():
    scan = AnisotropyScan(sweep="omega", fixed=90.0, n_points=37)
    angles = scan.angles()
    assert angles[0] == 0.0 and angles[-1] == 360.0 and angles.size == 37  # closed loop
    assert scan.geometry()[10] == (90.0, 100.0)  # alpha fixed, omega swept
    other = AnisotropyScan(sweep="alpha", fixed=30.0, n_points=37)
    assert other.geometry()[10] == (100.0, 30.0)  # omega fixed, alpha swept


@pytest.mark.parametrize(
    "bad",
    [
        AnisotropyScan(sweep="polar"),
        AnisotropyScan(n_points=2),
        AnisotropyScan(q_values=()),
        AnisotropyScan(q_values=(0.01, -1.0)),
        AnisotropyScan(q_values=(float("nan"),)),
    ],
)
def test_invalid_scans_are_rejected(bad):
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        calculate_scan(AnisaModel(populations=[Population()]), bad)


def test_scan_titles_and_dict_round_trip():
    scan = AnisotropyScan(sweep="alpha", fixed=0.0, q_values=(0.001, 0.02), n_points=19, label="")
    assert "α_Q scan" in scan.title() and "ω_Q = 0" in scan.title()
    assert AnisotropyScan(label="mine").title() == "mine"
    assert AnisotropyScan.from_dict(scan.to_dict()) == scan
    assert AnisotropyScan.from_dict(None) == AnisotropyScan()
    assert AnisotropyScan.from_dict({"fixed": 12}).sweep == "omega"


def test_scans_are_saved_in_the_project(tmp_path):
    model = AnisaModel(
        populations=[Population()],
        scans=[AnisotropyScan(sweep="alpha", fixed=45.0, q_values=(0.005,)), AnisotropyScan()],
    )
    loaded = load_project(save_project(model, tmp_path / "scans"))
    assert loaded.scans == model.scans
    assert AnisaModel.from_dict({"populations": []}).scans == []  # older files have none


# -- physics of the model scans ---------------------------------------------------------------


def test_axially_symmetric_sample_is_flat_in_omega():
    """Axes distributed independently of omega: scanning omega_Q at fixed alpha_Q changes nothing."""
    model = AnisaModel(populations=[disks()])
    for fixed in (30.0, 90.0):
        result = calculate_scan(
            model, AnisotropyScan("", "omega", fixed, (0.002, 0.01, 0.03), n_points=25)
        )
        spread = (
            result.intensity.max(axis=1) - result.intensity.min(axis=1)
        ) / result.intensity.mean(axis=1)
        assert np.all(spread < 1e-4), spread


def test_inversion_symmetry_I_of_Q_equals_I_of_minus_Q():
    """Axes are undirected, so rotating the scan direction by 180 deg changes nothing."""
    pop = Population(
        radius=800.0,
        beta=0.2,
        volume_fraction=0.04,
        delta_rho=2e11,
        alpha_dist=AngularDistribution(par1=60.0, par2=15.0),
        omega_dist=AngularDistribution(par1=40.0, par2=6.0),  # NOT symmetric under omega + 180
    )
    model = AnisaModel(populations=[pop])
    scan = AnisotropyScan("", "alpha", 25.0, (0.004, 0.015), n_points=73)  # 5 deg steps
    result = calculate_scan(model, scan)
    shifted = np.roll(result.intensity[:, :-1], -36, axis=1)  # angle + 180 deg
    np.testing.assert_allclose(shifted, result.intensity[:, :-1], rtol=2e-4)
    # at alpha_Q = 90 the omega scan is its own inversion circle
    flat = calculate_scan(model, AnisotropyScan("", "omega", 90.0, (0.004, 0.015), n_points=73))
    np.testing.assert_allclose(
        np.roll(flat.intensity[:, :-1], -36, axis=1), flat.intensity[:, :-1], rtol=2e-4
    )
    # ...and the scan is genuinely anisotropic (so the symmetry test is not vacuous)
    assert flat.intensity.max() / flat.intensity.min() > 1.5


def test_scan_points_equal_direct_evaluation_of_the_same_direction():
    model = demo.demo_model()
    scan = AnisotropyScan("", "alpha", 90.0, (0.003, 0.02), n_points=19)
    result = calculate_scan(model, scan)
    engine = Engine()
    for k in (0, 4, 9, 13, 18):
        alpha, omega = scan.geometry()[k]
        probe = AnisaModel(
            [Direction(alpha_q=alpha, omega_q=omega)], model.populations, model.settings
        )
        direct = engine.evaluate(probe, [np.array(scan.q_values)]).directions[0]
        np.testing.assert_allclose(
            result.intensity[:, k], direct.populations.sum(axis=0), rtol=1e-12
        )
        np.testing.assert_allclose(result.populations[:, :, k], direct.populations, rtol=1e-12)


def test_disks_with_axes_along_z_scatter_far_more_along_z_at_high_q():
    """An alpha scan through the normal: Q along the disk axis probes the thin direction."""
    model = AnisaModel(populations=[disks()])
    result = calculate_scan(model, AnisotropyScan("", "alpha", 0.0, (0.01,), n_points=73))
    along, across = result.intensity[0, 0], result.intensity[0, 18]  # alpha = 0 and 90 deg
    assert along > 100 * across
    np.testing.assert_allclose(
        result.intensity[0, 0], result.intensity[0, 36], rtol=1e-4
    )  # 0 == 180


def test_populations_sum_to_the_total_and_background_is_not_included():
    model = demo.demo_model()  # directions carry a 0.05 background
    result = calculate_scan(model, AnisotropyScan("", "omega", 45.0, (0.01,), n_points=13))
    np.testing.assert_allclose(result.populations.sum(axis=0), result.intensity)
    assert result.populations.shape == (3, 1, 13)
    assert result.angles.shape == (13,)


# -- measured points on the path --------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    return demo.synthesize(demo.demo_model(), relative_noise=0.02, seed=3)


def interpolated(direction: Direction, q: float) -> float:
    return float(np.exp(np.interp(np.log(q), np.log(direction.q), np.log(direction.intensity))))


def test_measured_cuts_on_an_alpha_scan_appear_with_their_inverses(synthetic):
    scan = AnisotropyScan("", "alpha", 0.0, (0.01,))
    points = measured_points(synthetic, scan, 0.01)
    # normal (0,0), in-plane x (90,0) and tilted (45,0) lie on the omega = 0 great circle,
    # each together with its inverse; in-plane y (90,90) does not
    np.testing.assert_allclose(points.angles, [0, 45, 90, 180, 225, 270])
    assert sorted(set(points.labels)) == ["along normal", "in plane, x", "tilted 45 deg"]
    normal = synthetic.directions[0]
    expected = interpolated(normal, 0.01) - normal.background
    first = [i for i, label in enumerate(points.labels) if label == "along normal"]
    np.testing.assert_allclose(points.intensity[first], expected)
    assert np.all(np.isfinite(points.sigma)) and np.all(points.sigma > 0)


def test_measured_cuts_on_an_omega_scan(synthetic):
    points = measured_points(synthetic, AnisotropyScan("", "omega", 90.0, (0.01,)), 0.01)
    np.testing.assert_allclose(points.angles, [0, 90, 180, 270])  # x and y, each with inverse
    assert sorted(set(points.labels)) == ["in plane, x", "in plane, y"]


def test_measured_points_match_the_model_scan_within_the_noise(synthetic):
    """The overlay logic and the scan geometry must agree: a model fitted to these data
    (here: the truth) passes through the measured points at the same angles."""
    scan = AnisotropyScan("", "alpha", 0.0, (0.004, 0.02), n_points=73)
    result = calculate_scan(synthetic, scan)
    for iq, q in enumerate(scan.q_values):
        points = measured_points(synthetic, scan, q)
        assert points.angles.size == 6
        for angle, value, sigma in zip(points.angles, points.intensity, points.sigma, strict=True):
            model_value = result.intensity[iq, int(round(angle / 5.0))]
            assert abs(value - model_value) < 5 * sigma, (q, angle, value, model_value, sigma)


def test_measured_points_respect_use_range_background_and_tolerance(synthetic):
    scan = AnisotropyScan("", "omega", 90.0, (0.01,))
    model = copy.deepcopy(synthetic)
    base = measured_points(model, scan, 0.01)
    model.directions[1].use = False
    assert set(measured_points(model, scan, 0.01).labels) == {"in plane, y"}
    model.directions[1].use = True
    model.directions[1].background += 1.0  # background is subtracted from the data
    changed = measured_points(model, scan, 0.01)
    k = changed.labels.index("in plane, x")
    assert changed.intensity[k] == pytest.approx(
        base.intensity[base.labels.index("in plane, x")] - 1.0
    )
    assert measured_points(model, scan, 5.0).angles.size == 0  # Q outside the data range
    model.directions[2].alpha_q = 90.5  # inside the 1 degree tolerance
    assert "in plane, y" in measured_points(model, scan, 0.01).labels
    model.directions[2].alpha_q = 93.0  # outside it
    assert "in plane, y" not in measured_points(model, scan, 0.01).labels
    model.directions[0].q = model.directions[0].intensity = None  # no data
    assert (
        "along normal" not in measured_points(model, AnisotropyScan("", "alpha", 0.0), 0.01).labels
    )


def test_an_omega_scan_through_the_pole_has_no_measured_points(synthetic):
    """At alpha_Q = 0 every omega is the same direction: the scan is a point."""
    assert (
        measured_points(synthetic, AnisotropyScan("", "omega", 0.0, (0.01,)), 0.01).angles.size == 0
    )
    flat = calculate_scan(
        AnisaModel(populations=[disks()]), AnisotropyScan("", "omega", 0.0, (0.01,), 7)
    )
    assert np.ptp(flat.intensity) / flat.intensity.mean() < 1e-12
    assert math.isfinite(flat.intensity[0, 0])
