"""Form factors: formula 4c in closed form, and 4a / 4b including refraction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from anisa.core import formfactor as ff


def test_sphere_amplitude_limits_and_values():
    assert ff.sphere_amplitude(0.0) == pytest.approx(1.0)
    assert ff.sphere_amplitude(1e-9) == pytest.approx(1.0)
    # first zero of 3 j1(x)/x is the first root of tan x = x, x = 4.4934...
    assert abs(ff.sphere_amplitude(4.493409457909064)) < 1e-12
    # 3 (sin x - x cos x) / x^3 at x = pi is 3 / pi^2
    assert ff.sphere_amplitude(np.pi) == pytest.approx(3.0 / np.pi**2)


def test_sphere_amplitude_matches_closed_form_across_range():
    # The closed form 3 (sin x - x cos x) / x^3 cancels badly at small x, so it is the
    # reference only from x = 0.05 up; below that the Taylor series is exact to double precision.
    x = np.geomspace(0.05, 500.0, 2000)
    closed = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    np.testing.assert_allclose(ff.sphere_amplitude(x), closed, rtol=1e-11, atol=1e-15)
    small = np.geomspace(1e-6, 0.05, 500)
    series = 1.0 - small**2 / 10 + small**4 / 280 - small**6 / 15120 + small**8 / 1330560
    np.testing.assert_allclose(ff.sphere_amplitude(small), series, rtol=1e-13)


def test_sphere_amplitude_sq_is_the_square_and_vectorised():
    x = np.array([[0.1, 1.0], [10.0, 0.0]])
    np.testing.assert_allclose(ff.sphere_amplitude_sq(x), ff.sphere_amplitude(x) ** 2)


def test_spheroid_k_and_volume():
    assert ff.spheroid_k(0.3, 0.0) == pytest.approx(1.0)  # axis perpendicular to Q
    assert ff.spheroid_k(0.3, 1.0) == pytest.approx(0.3)  # axis along Q
    assert ff.spheroid_k(1.0, 0.7) == pytest.approx(1.0)  # sphere
    assert ff.spheroid_volume(10.0, 2.0) == pytest.approx(4 / 3 * np.pi * 2 * 1000.0)


def test_4c_equals_igor_bessel_form():
    """Igor wrote 4c as V^2 drho^2 (9 pi / 2) [J_3/2(x) / x^1.5]^2; it is the same function."""
    q = np.geomspace(1e-3, 0.2, 40)
    k = ff.spheroid_k(0.4, np.linspace(0, 1, 5))
    r, beta, drho = 300.0, 0.4, 1e-5
    x = r * np.outer(q, k)
    igor = (
        (ff.spheroid_volume(r, beta) * drho) ** 2 * 4.5 * np.pi * (special.jv(1.5, x) / x**1.5) ** 2
    )
    np.testing.assert_allclose(ff.cross_section_4c(q, k, r, beta, drho), igor, rtol=1e-10)


def test_4c_forward_scattering_is_volume_squared_contrast_squared():
    got = ff.cross_section_4c(np.array([0.0]), np.array([1.0]), 100.0, 2.0, 3e-6)
    assert got[0, 0] == pytest.approx((ff.spheroid_volume(100.0, 2.0) * 3e-6) ** 2)


# -- refraction -------------------------------------------------------------------------------

Q = np.geomspace(2e-6, 2e-4, 25)
KS = ff.spheroid_k(0.3, np.linspace(0, 1, 6))


@pytest.mark.parametrize("formula", ["4a", "4b"])
def test_refraction_formulas_reduce_to_4c_when_contrast_vanishes(formula):
    r, beta, lam = 800.0, 0.3, 1.0
    q = np.linspace(0.0005, 0.05, 30)
    for drho in (1e-9, 1e-7):
        refr = ff.cross_section(formula, q, KS, r, beta, drho, lam)
        diff = ff.cross_section_4c(q, KS, r, beta, drho)
        # the first correction is O(nu_0^2); nu_0 = 2 R drho lambda. The absolute floor covers
        # deep zero crossings, where 1 - exp(i y) ~ y loses a few digits to cancellation.
        nu0 = 2 * r * drho * lam
        np.testing.assert_allclose(refr, diff, rtol=20 * nu0**2 + 1e-7, atol=1e-8 * np.max(diff))


@pytest.mark.parametrize(
    "r, beta",
    [(20000.0, 0.3), (100000.0, 0.3), (100000.0, 2.0), (300000.0, 0.5)],
)
def test_series_with_factorial_matches_the_integral(r, beta):
    """4b needs the 1/m! that the printed formula lacks: with it, 4b equals 4a to ~1e-12.

    Parameters give nu_0 = 2 R drho lambda from 0.24 to 3.6, c = beta nu_0 / K up to 3.6.
    """
    lam, drho = 0.6, 1e-5
    k = ff.spheroid_k(beta, np.linspace(0, 1, 6))
    q = Q / (r / 1e5)
    a = ff.cross_section_4a(q, k, r, beta, drho, lam)
    b = ff.cross_section_4b(q, k, r, beta, drho, lam)
    np.testing.assert_allclose(b, a, rtol=1e-9)


def test_printed_series_without_factorial_is_wrong():
    """Documents why Anisa does not follow Allen's printed eq. 4b / the Igor code.

    Even at small refraction (c < 1) the series without 1/m! is off by percent; it diverges for
    c > 1. (Igor never ran it: the Gamma table it needs was never created.)
    """
    r, beta, lam, drho = 20000.0, 0.3, 0.6, 1e-5
    k = ff.spheroid_k(beta, np.linspace(0, 1, 6))
    q = Q / (r / 1e5)
    exact = ff.cross_section_4a(q, k, r, beta, drho, lam)
    c = beta * 2 * r * drho * lam / k
    a = np.maximum(r * np.outer(q, k), 1e-12)
    total = np.zeros(a.shape, complex)
    for m in range(1, 31):
        order = 1 + m / 2
        total += (
            (1j * c)[None, :] ** m
            * 2 ** (m / 2)
            * special.gamma(order)
            * special.jv(order, a)
            / a**order
        )
    printed = (2 * np.pi / lam * r**2 * k[None, :] * np.abs(total)) ** 2
    assert np.max(np.abs(printed - exact) / exact) > 0.01


def test_4b_falls_back_to_4a_for_large_c():
    r, beta, lam, drho = 100000.0, 0.5, 0.6, 2e-4  # nu_0 = 24, c = beta nu_0 / K from 12 to 24
    k = ff.spheroid_k(beta, np.linspace(0, 1, 6))
    assert np.max(beta * 2 * r * drho * lam / k) > ff.C_SERIES_MAX
    q = Q / (r / 1e5)
    np.testing.assert_allclose(
        ff.cross_section_4b(q, k, r, beta, drho, lam),
        ff.cross_section_4a(q, k, r, beta, drho, lam),
        rtol=1e-9,
    )


def test_4a_is_converged_in_its_quadrature():
    r, beta, lam, drho = 800.0, 0.3, 0.6, 1e-5
    q = np.linspace(0.0005, 0.05, 40)
    base = ff.cross_section_4a(q, KS, r, beta, drho, lam)
    original = ff._n_theta
    try:
        ff._n_theta = lambda a_max, c_max: 3 * original(a_max, c_max)
        finer = ff.cross_section_4a(q, KS, r, beta, drho, lam)
    finally:
        ff._n_theta = original
    np.testing.assert_allclose(base, finer, rtol=1e-9)


def test_refraction_reduces_forward_scattering():
    """Refraction (phase shift) lowers the cross section below the diffraction limit."""
    r, beta, lam, drho = 100000.0, 0.5, 0.6, 1e-5
    k = np.array([1.0])
    q = np.array([1e-7])
    assert (
        ff.cross_section_4a(q, k, r, beta, drho, lam)[0, 0]
        < ff.cross_section_4c(q, k, r, beta, drho)[0, 0]
    )


def test_cross_section_rejects_unknown_formula():
    with pytest.raises(ValueError):
        ff.cross_section("4z", np.array([0.1]), np.array([1.0]), 10.0, 1.0, 1e-5)
