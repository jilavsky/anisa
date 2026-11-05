"""Size distributions: radius range, classes, volume vs number weighting."""

from __future__ import annotations

import numpy as np
import pytest
from pyirena.core import distributions as irena_dist

from anisa.core.formfactor import spheroid_volume
from anisa.core.sizedist import (
    DISTRIBUTION_TYPES,
    PARAMETER_DEFAULTS,
    SizeDistribution,
    radius_classes,
    radius_range,
    volume_density,
)

CASES = [
    ("gauss", {"width": 150.0}),
    ("lognormal", {"min_size": 10.0, "sdeviation": 0.4}),
    ("schulz_zimm", {"width": 200.0}),
    ("lsw", {}),
    ("ardell", {"parameter": 2.5}),
]


def test_all_types_have_defaults():
    assert set(PARAMETER_DEFAULTS) == set(DISTRIBUTION_TYPES)


def test_monodisperse_is_a_single_class():
    radii, weights = radius_classes(SizeDistribution(), 321.0, 0.5)
    np.testing.assert_array_equal(radii, [321.0])
    np.testing.assert_array_equal(weights, [1.0])
    assert radius_range(SizeDistribution(), 321.0) == (321.0, 321.0)


@pytest.mark.parametrize("dist_type, params", CASES)
@pytest.mark.parametrize("weighting", ["volume", "number"])
def test_classes_are_uniform_with_weights_summing_to_one(dist_type, params, weighting):
    dist = SizeDistribution(dist_type, params, 50, weighting)
    radii, weights = radius_classes(dist, 1000.0, 0.4)
    lo, hi = radius_range(dist, 1000.0)
    assert radii.size == weights.size == 50
    assert weights.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(np.diff(radii), (hi - lo) / 50, rtol=1e-12)  # uniform cells
    assert lo < radii[0] and radii[-1] < hi


@pytest.mark.parametrize("dist_type, params", CASES)
def test_range_leaves_only_the_tails_out(dist_type, params):
    dist = SizeDistribution(dist_type, params)
    lo, hi = radius_range(dist, 1000.0)
    located = dict(params)
    located["location" if dist_type in ("lsw", "ardell") else "mean_size"] = 1000.0
    inside = irena_dist.cdf(dist_type, hi, located) - irena_dist.cdf(dist_type, lo, located)
    assert inside > 1.0 - 1e-3


def test_gauss_classes_are_centred_on_the_radius():
    dist = SizeDistribution("gauss", {"width": 100.0}, 200)
    radii, weights = radius_classes(dist, 1000.0, 0.5)
    assert np.sum(weights * radii) == pytest.approx(1000.0, rel=1e-6)
    sigma = np.sqrt(np.sum(weights * (radii - 1000.0) ** 2))
    assert sigma == pytest.approx(100.0, rel=1e-3)  # the Irena 'width' is the sigma


def test_number_weighting_multiplies_by_volume():
    vol = SizeDistribution("gauss", {"width": 150.0}, 50, "volume")
    num = SizeDistribution("gauss", {"width": 150.0}, 50, "number")
    r, w_vol = radius_classes(vol, 1000.0, 0.4)
    _, w_num = radius_classes(num, 1000.0, 0.4)
    expected = w_vol * spheroid_volume(r, 0.4)
    np.testing.assert_allclose(w_num, expected / expected.sum(), rtol=1e-12)
    # larger particles weigh more once the distribution counts particles, not volume
    assert np.sum(w_num * r) > np.sum(w_vol * r)


def test_volume_density_requires_a_distribution():
    with pytest.raises(ValueError):
        volume_density(SizeDistribution(), 100.0, 1.0, np.array([100.0]))


def test_location_parameter_follows_the_radius():
    """Fitting the radius must move the whole distribution."""
    dist = SizeDistribution("gauss", {"width": 50.0}, 100)
    for radius in (400.0, 1500.0):
        r, w = radius_classes(dist, radius, 1.0)
        assert np.sum(r * w) == pytest.approx(radius, rel=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        SizeDistribution("bogus"),
        SizeDistribution("gauss", {"width": 10.0}, 1),
        SizeDistribution("gauss", {"width": 10.0}, 50, "mass"),
    ],
)
def test_invalid_size_distributions_are_rejected(bad):
    with pytest.raises(ValueError):
        radius_classes(bad, 100.0, 1.0)


def test_non_positive_radius_is_rejected():
    with pytest.raises(ValueError):
        radius_range(SizeDistribution("gauss", {"width": 5.0}), 0.0)


def test_dict_round_trip_fills_defaults():
    d = SizeDistribution("lognormal", {"sdeviation": 0.5}, 80, "number")
    back = SizeDistribution.from_dict(d.to_dict())
    assert back.key() == d.key()
    assert back.full_params() == {"min_size": 10.0, "sdeviation": 0.5}
    assert SizeDistribution.from_dict(None).type == "none"
    assert SizeDistribution.from_dict({"type": "gauss"}).full_params() == {"width": 100.0}


def test_key_ignores_unknown_parameters():
    a = SizeDistribution("gauss", {"width": 10.0})
    b = SizeDistribution("gauss", {"width": 10.0, "irrelevant": 5.0})
    assert a.key() == b.key()
