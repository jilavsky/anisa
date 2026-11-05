"""Model dataclasses (JSON round trip, defaults, parameters) and derived quantities."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import integrate

from anisa.core import derived
from anisa.core.model import (
    DEFAULT_LIMITS,
    POPULATION_PARAMETERS,
    AnisaModel,
    CalcSettings,
    Direction,
    FitSpec,
    Population,
)
from anisa.core.orientation import AngularDistribution
from anisa.core.sizedist import SizeDistribution, radius_range, volume_density

# -- serialisation ----------------------------------------------------------------------------


def _busy_model() -> AnisaModel:
    pop = Population(
        name="cracks",
        radius=850.0,
        volume_fraction=0.037,
        beta=0.12,
        delta_rho=2.3e11,
        size_dist=SizeDistribution(
            "lognormal", {"min_size": 20.0, "sdeviation": 0.35}, 80, "number"
        ),
        alpha_dist=AngularDistribution(par1=77.0, par2=9.0, par3=0.2),
        omega_dist=AngularDistribution(mode="table", table=(1.0, 2.0, 5.0, 2.0)),
        use_interference=True,
        eta=240.0,
        pack=1.5,
    )
    pop.fit_spec("radius").fit = True
    pop.fit_spec("radius").min, pop.fit_spec("radius").max = 100.0, 5000.0
    pop.fit_spec("alpha_par2").fit = True
    pop.fit_spec("volume_fraction").max = None  # explicitly unbounded
    d = Direction(label="dir A", alpha_q=35.0, omega_q=190.0, background=0.12)
    d.fit_spec().fit = True
    return AnisaModel(
        [d, Direction(label="dir B", alpha_q=90.0)],
        [pop, Population(name="spheres", beta=1.0)],
        CalcSettings(formula="4b", n_alpha=40, wavelength=0.59, n_threads=2),
    )


def test_model_round_trips_through_json_unchanged():
    model = _busy_model()
    text = json.dumps(model.to_dict())  # must be valid JSON: no inf/nan
    back = AnisaModel.from_dict(json.loads(text))
    assert back.to_dict() == model.to_dict()
    pop = back.populations[0]
    assert pop.fit_spec("radius").fit and pop.fit_spec("radius").min == 100.0
    assert pop.fit_spec("volume_fraction").max is None
    assert pop.omega_dist.table == (1.0, 2.0, 5.0, 2.0)
    assert pop.size_dist.weighting == "number"
    assert back.directions[0].fit_spec().fit
    assert back.settings.formula == "4b"


def test_missing_fields_get_defaults_so_old_files_keep_loading():
    assert AnisaModel.from_dict(None).populations == []
    pop = Population.from_dict({"radius": 500.0})  # bare number, nothing else
    default = Population()
    assert pop.radius == 500.0
    assert pop.beta == default.beta and pop.volume_fraction == default.volume_fraction
    assert pop.fit_spec("radius").min == DEFAULT_LIMITS["radius"][0]
    d = Direction.from_dict({})
    assert (d.alpha_q, d.omega_q, d.background, d.use) == (0.0, 0.0, 0.0, True)
    s = CalcSettings.from_dict({"formula": "4a"})
    assert s.formula == "4a" and s.n_alpha == CalcSettings().n_alpha


def test_unknown_keys_in_files_are_ignored():
    model = AnisaModel.from_dict({"future_feature": 1, "populations": [{"radius": 1.0, "x": 2}]})
    assert model.populations[0].radius == 1.0


def test_parameter_accessors_cover_every_path():
    pop = Population()
    for i, path in enumerate(POPULATION_PARAMETERS):
        value = 1.5 + i
        pop.set_parameter(path, value)
        assert pop.get_parameter(path) == value
    assert pop.alpha_dist.par1 == pop.get_parameter("alpha_par1")
    assert pop.omega_dist.par3 == pop.get_parameter("omega_par3")
    with pytest.raises(KeyError):
        pop.get_parameter("nope")
    with pytest.raises(KeyError):
        pop.set_parameter("nope", 1.0)


def test_fit_spec_defaults_come_from_the_limits_table():
    for path in ("radius", "alpha_par2", "pack"):
        spec = FitSpec.from_dict(None, path)
        assert (spec.min, spec.max) == DEFAULT_LIMITS[path]
        assert spec.fit is False
    assert FitSpec.from_dict({"min": float("inf")}).min is None


def test_shape_key_ignores_volume_fraction_and_interference():
    a = Population(radius=100.0, volume_fraction=0.1)
    b = Population(radius=100.0, volume_fraction=0.3, use_interference=True, eta=5.0, pack=2.0)
    assert a.shape_key() == b.shape_key()
    assert a.shape_key() != Population(radius=101.0).shape_key()
    assert a.shape_key() != Population(radius=100.0, beta=0.5).shape_key()
    assert (
        a.shape_key()
        != Population(radius=100.0, alpha_dist=AngularDistribution(par2=1.0)).shape_key()
    )


def test_settings_key_ignores_threads_and_wavelength_for_4c():
    base = CalcSettings()
    assert base.key() == CalcSettings(n_threads=7, wavelength=3.0).key()
    assert (
        CalcSettings(formula="4a", wavelength=1.0).key()
        != CalcSettings(formula="4a", wavelength=2.0).key()
    )
    assert base.key() != CalcSettings(n_alpha=65).key()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"formula": "9z"},
        {"n_alpha": 1},
        {"points_per_period": 2},
        {"formula": "4a", "wavelength": 0.0},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CalcSettings(**kwargs).validate()


def test_no_limit_on_directions_or_populations():
    model = AnisaModel(
        [Direction(alpha_q=float(i)) for i in range(40)], [Population() for _ in range(25)]
    )
    back = AnisaModel.from_dict(model.to_dict())
    assert (len(back.directions), len(back.populations)) == (40, 25)


# -- derived quantities -----------------------------------------------------------------------


def test_sphere_surface_to_volume_is_three_over_r():
    pop = Population(radius=250.0, beta=1.0, volume_fraction=0.2)
    assert derived.surface_area(pop) == pytest.approx(1e8 * 0.2 * 3.0 / 250.0, rel=1e-12)


@pytest.mark.parametrize("beta", [0.05, 0.3, 0.9, 1.0, 1.1, 2.0, 12.0])
def test_spheroid_surface_area_matches_numerical_integration(beta):
    """S = 2 pi R^2 Int_0^pi sin(t) sqrt(cos^2 t + beta^2 sin^2 t) dt for semi-axes (R, R, beta R)."""
    r = 37.0
    numeric, _ = integrate.quad(
        lambda t: math.sin(t) * math.sqrt(math.cos(t) ** 2 + beta**2 * math.sin(t) ** 2),
        0.0,
        math.pi,
        epsabs=0,
        epsrel=1e-12,
    )
    assert float(derived.spheroid_surface_area(r, beta)) == pytest.approx(
        2.0 * math.pi * r * r * numeric, rel=1e-9
    )


def test_surface_area_is_continuous_through_beta_equal_one():
    below = derived.spheroid_surface_area(10.0, 1.0 - 1e-6)
    above = derived.spheroid_surface_area(10.0, 1.0 + 1e-6)
    sphere = derived.spheroid_surface_area(10.0, 1.0)
    assert below == pytest.approx(sphere, rel=1e-5)
    assert above == pytest.approx(sphere, rel=1e-5)


def test_number_density_and_mean_volume_monodisperse():
    pop = Population(radius=100.0, beta=0.5, volume_fraction=0.05)
    volume = 4.0 / 3.0 * math.pi * 0.5 * 100.0**3
    assert derived.mean_volume(pop) == pytest.approx(volume, rel=1e-12)
    assert derived.number_density(pop) == pytest.approx(1e24 * 0.05 / volume, rel=1e-12)


@pytest.mark.parametrize("weighting", ["volume", "number"])
def test_polydisperse_number_density_matches_quadrature(weighting):
    """n = 1e24 phi Int v(R) / V(R) dR over the normalised volume density v."""
    dist = SizeDistribution("gauss", {"width": 120.0}, 100, weighting)
    pop = Population(radius=800.0, beta=0.4, volume_fraction=0.05, size_dist=dist)
    lo, hi = radius_range(dist, 800.0)

    def density(r):
        return float(volume_density(dist, 800.0, 0.4, np.array([r]))[0])

    total, _ = integrate.quad(density, lo, hi, epsrel=1e-12)
    inverse_volume, _ = integrate.quad(
        lambda r: density(r) / (4.0 / 3.0 * math.pi * 0.4 * r**3), lo, hi, epsrel=1e-12
    )
    assert derived.number_density(pop) == pytest.approx(
        1e24 * 0.05 * inverse_volume / total, rel=1e-6
    )


def test_polydispersity_raises_surface_to_volume():
    mono = Population(radius=500.0, beta=0.5)
    poly = Population(radius=500.0, beta=0.5, size_dist=SizeDistribution("gauss", {"width": 150.0}))
    assert derived.surface_area(poly) > derived.surface_area(mono)  # <1/R> > 1/<R>
