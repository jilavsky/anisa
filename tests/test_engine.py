"""
Engine validation against independent references (PLAN.md section 9):

* analytic limits (sphere, isotropic average, aligned particles, direction independence),
* a brute-force orientation sum that shares no code with the engine's K binning,
* scipy quadrature over radius for polydisperse spheres,
* pyIrena's spheroid form factor,
* consistency between the engine's computation paths, caching and threading.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyirena.core.form_factors import spheroid_ff
from scipy import integrate, special

from anisa.core import engine as eng
from anisa.core.engine import Engine, calculate, q_bands, resolved_grid_size
from anisa.core.model import AnisaModel, CalcSettings, Direction, Population
from anisa.core.orientation import AngularDistribution
from anisa.core.sizedist import SizeDistribution, radius_range, volume_density

DRHO = 2.0e11  # cm^-2
DRHO_A = DRHO * 1e-16  # A^-2


def amp2(x):
    return (3.0 * special.spherical_jn(1, x) / x) ** 2


def one(pop, direction, q, **settings):
    model = AnisaModel([direction], [pop], CalcSettings(**settings))
    return Engine().evaluate(model, [q]).directions[0].intensity


def isotropic_reference(q, radius, beta):
    """<A^2(Q R K)> over isotropic axes (X uniform on [0, 1]) by 4000-node Gauss-Legendre."""
    x, w = np.polynomial.legendre.leggauss(4000)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    k = np.sqrt(1.0 + (beta**2 - 1.0) * x**2)
    return amp2(q[:, None] * radius * k[None, :]) @ w


def brute_force(pop, direction, q, n_alpha=600, n_omega=1200):
    """Dense mid-point orientation sum, evaluating the amplitude at every node (formula 4c,
    monodisperse). Independent of the engine's K binning and its grid-size rules."""
    c = (np.arange(n_alpha) + 0.5) / n_alpha
    alpha = np.arccos(c)
    omega = (np.arange(n_omega) + 0.5) * 2.0 * np.pi / n_omega
    w = np.outer(pop.alpha_dist.alpha_values(alpha), pop.omega_dist.omega_values(omega))
    w /= w.sum()
    aq, oq = np.radians(direction.alpha_q), np.radians(direction.omega_q)
    x = np.sin(aq) * np.sin(alpha)[:, None] * np.cos(omega - oq)[None, :] + np.cos(aq) * c[:, None]
    k = np.sqrt(1.0 + (pop.beta**2 - 1.0) * x * x).ravel()
    wf = w.ravel()
    volume = 4.0 / 3.0 * np.pi * pop.beta * pop.radius**3
    out = np.array([np.sum(wf * amp2(qi * pop.radius * k)) for qi in q])
    return 1e8 * pop.volume_fraction * volume * (pop.delta_rho * 1e-16) ** 2 * out


Q = np.geomspace(2e-4, 0.12, 60)
TILTED = Direction(alpha_q=53.0, omega_q=20.0)
ANISOTROPIC = dict(
    alpha_dist=AngularDistribution(par1=25.0, par2=6.0, par3=0.02),
    omega_dist=AngularDistribution(par1=80.0, par2=3.0, par3=0.05),
)

# -- analytic limits --------------------------------------------------------------------------


def test_sphere_matches_closed_form_and_forward_limit():
    pop = Population(radius=400.0, beta=1.0, volume_fraction=0.07, delta_rho=DRHO, **ANISOTROPIC)
    q = np.geomspace(1e-4, 0.1, 50)
    volume = 4.0 / 3.0 * np.pi * 400.0**3
    expected = 1e8 * 0.07 * volume * DRHO_A**2 * amp2(q * 400.0)
    np.testing.assert_allclose(one(pop, TILTED, q), expected, rtol=1e-12)
    # I(0) = 1e8 phi V drho^2  [cm^-1]
    forward = one(pop, TILTED, np.array([1e-9]))[0]
    assert forward == pytest.approx(1e8 * 0.07 * volume * DRHO_A**2, rel=1e-12)


@pytest.mark.parametrize("beta", [0.1, 0.4, 2.5, 8.0])
def test_isotropic_average_matches_independent_quadrature(beta):
    pop = Population(radius=700.0, beta=beta, volume_fraction=0.05, delta_rho=DRHO)
    q = np.geomspace(1e-4, 0.1, 80)
    volume = 4.0 / 3.0 * np.pi * beta * 700.0**3
    expected = 1e8 * 0.05 * volume * DRHO_A**2 * isotropic_reference(q, 700.0, beta)
    got = one(pop, Direction(alpha_q=37.0, omega_q=111.0), q)
    # design target: 1e-4 relative at the worst Q point with the default settings
    np.testing.assert_allclose(got, expected, rtol=1e-4)
    assert np.median(np.abs(got - expected) / expected) < 1e-6


def test_isotropic_average_agrees_with_pyirena_spheroid():
    """pyIrena's spheroid_ff uses 50 Gauss-Legendre nodes: good to a few 1e-4 up to Q R ~ 80."""
    pop = Population(radius=800.0, beta=0.25, volume_fraction=0.05, delta_rho=DRHO)
    q = np.geomspace(1e-4, 0.1, 100)
    expected = 1e8 * 0.05 * DRHO_A**2 * spheroid_ff(q, 800.0, 0.25)
    np.testing.assert_allclose(one(pop, TILTED, q), expected, rtol=2e-3)


def test_isotropic_intensity_does_not_depend_on_the_direction_of_q():
    pop = Population(radius=500.0, beta=0.3, volume_fraction=0.1, delta_rho=DRHO)
    q = np.geomspace(1e-4, 0.1, 40)
    reference = one(pop, Direction(alpha_q=0.0), q)
    for aq, oq in ((90.0, 0.0), (37.0, 111.0), (62.0, 290.0)):
        np.testing.assert_allclose(
            one(pop, Direction(alpha_q=aq, omega_q=oq), q), reference, rtol=1e-4
        )


def test_aligned_particles_scatter_with_the_axis_along_or_across_q():
    """Axes all along z: Q along z sees X = 1 (K = beta), Q across z sees X = 0 (K = 1)."""
    beta, radius = 0.3, 600.0
    sharp = dict(alpha_dist=AngularDistribution(par1=0.0, par2=20000.0))
    pop = Population(radius=radius, beta=beta, volume_fraction=0.05, delta_rho=DRHO, **sharp)
    q = np.geomspace(2e-4, 0.01, 40)  # Q R up to 6: the form factor varies slowly enough
    volume = 4.0 / 3.0 * np.pi * beta * radius**3
    prefactor = 1e8 * 0.05 * volume * DRHO_A**2
    along = one(pop, Direction(alpha_q=0.0), q)
    across = one(pop, Direction(alpha_q=90.0, omega_q=0.0), q)
    # a finite sharpness fills the zeros of A^2 (first zero at Q R K = 4.49), where a purely
    # relative comparison is meaningless, hence the absolute floor
    for got, x in ((along, q * radius * beta), (across, q * radius)):
        expected = prefactor * amp2(x)
        np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-4 * expected.max())
    assert np.all(along >= across)  # an oblate particle's long dimension lies across its axis


def test_oblate_particle_scatters_more_across_than_along_at_high_q():
    """Oblate disks (beta < 1) with axes along z: Q along z probes the thin dimension, so the
    scattering extends to much higher Q than for Q across z."""
    pop = Population(
        radius=800.0,
        beta=0.1,
        volume_fraction=0.05,
        delta_rho=DRHO,
        alpha_dist=AngularDistribution(par1=0.0, par2=3000.0),
    )
    q = np.array([0.02])
    along = one(pop, Direction(alpha_q=0.0), q)[0]
    across = one(pop, Direction(alpha_q=90.0), q)[0]
    assert along > 100 * across


# -- brute force ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "beta, radius",
    [(0.1, 3000.0), (0.4, 300.0), (2.5, 1500.0)],
)
def test_anisotropic_tilted_direction_matches_brute_force(beta, radius):
    pop = Population(radius=radius, beta=beta, volume_fraction=0.05, delta_rho=DRHO, **ANISOTROPIC)
    q = np.geomspace(2e-4, 0.12, 40)
    expected = brute_force(pop, TILTED, q)
    got = one(pop, TILTED, q)
    # The brute force is itself good to ~1e-4 at the worst Q point (it moves by 3e-4 when its own
    # grid is doubled), so this is a check at that level, not tighter.
    np.testing.assert_allclose(got, expected, rtol=5e-4)
    assert np.median(np.abs(got - expected) / expected) < 1e-4  # limited by the reference


def test_table_orientation_matches_brute_force():
    pop = Population(
        radius=500.0,
        beta=0.2,
        volume_fraction=0.05,
        delta_rho=DRHO,
        alpha_dist=AngularDistribution(mode="table", table=(0.2, 1.0, 3.0, 0.5, 0.1)),
        omega_dist=AngularDistribution(mode="table", table=(1.0, 4.0, 1.0, 0.2)),
    )
    q = np.geomspace(2e-4, 0.1, 30)
    # tables have kinks, so the brute force needs a fine grid too
    np.testing.assert_allclose(
        one(pop, TILTED, q), brute_force(pop, TILTED, q, 900, 1800), rtol=5e-4
    )


# -- polydisperse -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist",
    [
        SizeDistribution("gauss", {"width": 150.0}),
        SizeDistribution("lognormal", {"min_size": 10.0, "sdeviation": 0.4}),
        SizeDistribution("schulz_zimm", {"width": 100.0}),
        SizeDistribution("gauss", {"width": 150.0}, weighting="number"),
    ],
)
def test_polydisperse_spheres_match_scipy_quadrature(dist):
    """Spheres: I = 1e8 phi drho^2 Int v(R) V(R) A^2(Q R) dR, integrated independently."""
    pop = Population(radius=1000.0, beta=1.0, volume_fraction=0.04, delta_rho=DRHO, size_dist=dist)
    lo, hi = radius_range(dist, 1000.0)

    def density(r):
        return float(volume_density(dist, 1000.0, 1.0, np.array([r]))[0])

    norm, _ = integrate.quad(density, lo, hi, epsrel=1e-12)
    q = np.geomspace(5e-4, 0.1, 12)
    expected = np.empty(q.size)
    for i, qi in enumerate(q):
        value, _ = integrate.quad(
            lambda r, qi=qi: density(r) * 4 / 3 * np.pi * r**3 * amp2(qi * r),
            lo,
            hi,
            limit=400,
            epsrel=1e-10,
        )
        expected[i] = 1e8 * 0.04 * DRHO_A**2 * value / norm
    np.testing.assert_allclose(one(pop, TILTED, q), expected, rtol=1e-4)


def test_log_space_path_matches_direct_sum_over_sizes_and_orientations():
    """The convolution in ln(R K) and the direct (size x K-node) sum are different algorithms
    for the same integral."""
    pop = Population(
        radius=900.0,
        beta=0.3,
        volume_fraction=0.05,
        delta_rho=DRHO,
        **ANISOTROPIC,
        size_dist=SizeDistribution("gauss", {"width": 120.0}),
    )
    settings = CalcSettings()
    q = np.geomspace(5e-4, 0.1, 25)
    engine = Engine()
    log_space = engine._band_log_space(q, TILTED.geometry_key(), pop, settings)
    direct = engine._band_classes(q, TILTED.geometry_key(), pop, settings)
    np.testing.assert_allclose(log_space, direct, rtol=2e-4)


def test_size_distribution_smooths_the_form_factor_oscillations():
    sharp = Population(radius=1000.0, beta=1.0, delta_rho=DRHO)
    broad = Population(
        radius=1000.0,
        beta=1.0,
        delta_rho=DRHO,
        size_dist=SizeDistribution("gauss", {"width": 200.0}),
    )
    q = np.linspace(0.02, 0.06, 400)
    ratio_sharp = one(sharp, TILTED, q)
    ratio_broad = one(broad, TILTED, q)
    # monodisperse spheres have deep minima; a 20 % size distribution fills them in
    assert (
        ratio_sharp.min() / ratio_sharp.max() < 1e-3 * ratio_broad.min() / ratio_broad.max() + 1e-2
    )
    assert ratio_broad.min() > 20 * ratio_sharp.min()


# -- refraction formulas through the engine ---------------------------------------------------


def test_refraction_formulas_agree_with_4c_at_small_contrast_through_the_engine():
    pop = Population(radius=600.0, beta=0.4, volume_fraction=0.05, delta_rho=1e8, **ANISOTROPIC)
    q = np.geomspace(5e-4, 0.05, 25)
    base = one(pop, TILTED, q, formula="4c")
    for formula in ("4a", "4b"):
        np.testing.assert_allclose(
            one(pop, TILTED, q, formula=formula, wavelength=1.0),
            base,
            rtol=1e-5,
            atol=1e-8 * base.max(),
        )


def test_4a_and_4b_agree_at_strong_refraction_through_the_engine():
    pop = Population(
        radius=50000.0, beta=0.4, volume_fraction=0.05, delta_rho=1e11, **ANISOTROPIC
    )  # nu_0 = 2 R drho lambda = 0.6 at 0.6 A
    q = np.geomspace(1e-5, 4e-4, 12)
    a = one(pop, TILTED, q, formula="4a", wavelength=0.6)
    b = one(pop, TILTED, q, formula="4b", wavelength=0.6)
    c = one(pop, TILTED, q, formula="4c")
    np.testing.assert_allclose(a, b, rtol=1e-6)
    assert np.max(np.abs(a - c) / c) > 0.01  # refraction matters here


# -- model composition ------------------------------------------------------------------------


def test_populations_add_and_background_is_added_not_subtracted():
    p1 = Population(radius=300.0, beta=0.5, volume_fraction=0.02, delta_rho=DRHO)
    p2 = Population(radius=900.0, beta=2.0, volume_fraction=0.01, delta_rho=DRHO, **ANISOTROPIC)
    d = Direction(alpha_q=40.0, omega_q=10.0, background=0.37)
    q = np.geomspace(1e-3, 0.05, 20)
    both = Engine().evaluate(AnisaModel([d], [p1, p2]), [q]).directions[0]
    alone1 = one(p1, Direction(alpha_q=40.0, omega_q=10.0), q)
    alone2 = one(p2, Direction(alpha_q=40.0, omega_q=10.0), q)
    np.testing.assert_allclose(both.populations[0], alone1, rtol=1e-12)
    np.testing.assert_allclose(both.populations[1], alone2, rtol=1e-12)
    np.testing.assert_allclose(both.intensity, alone1 + alone2 + 0.37, rtol=1e-12)
    assert both.background == 0.37


def test_no_populations_gives_only_the_background():
    d = Direction(background=1.25)
    out = Engine().evaluate(AnisaModel([d], []), [np.array([0.01, 0.1])]).directions[0]
    np.testing.assert_array_equal(out.intensity, [1.25, 1.25])
    assert out.populations.shape == (0, 2)


def test_interference_factor_is_applied_per_population():
    plain = Population(radius=300.0, beta=0.5, volume_fraction=0.05, delta_rho=DRHO)
    packed = Population(
        radius=300.0,
        beta=0.5,
        volume_fraction=0.05,
        delta_rho=DRHO,
        use_interference=True,
        eta=500.0,
        pack=2.0,
    )
    q = np.geomspace(1e-3, 0.05, 20)
    factor = 1.0 + 2.0 * 3.0 * special.spherical_jn(1, q * 500.0) / (q * 500.0)
    np.testing.assert_allclose(one(packed, TILTED, q), one(plain, TILTED, q) / factor, rtol=1e-12)
    off = Population(**{**packed.__dict__, "use_interference": False})
    np.testing.assert_allclose(one(off, TILTED, q), one(plain, TILTED, q), rtol=1e-12)


def test_many_directions_and_populations_have_no_limit():
    rng = np.random.default_rng(5)
    pops = [
        Population(
            radius=200.0 + 60 * i,
            beta=0.3 + 0.2 * i,
            volume_fraction=0.01,
            delta_rho=DRHO,
            alpha_dist=AngularDistribution(par1=rng.uniform(0, 90), par2=rng.uniform(0, 8)),
        )
        for i in range(8)
    ]
    dirs = [
        Direction(alpha_q=float(a), omega_q=float(o))
        for a, o in zip(np.linspace(0, 90, 23), np.linspace(0, 340, 23), strict=True)
    ]
    q = np.geomspace(1e-3, 0.05, 30)
    result = Engine().evaluate(AnisaModel(dirs, pops), [q] * 23)
    assert len(result.directions) == 23
    assert all(
        d.populations.shape == (8, 30) and np.all(np.isfinite(d.intensity))
        for d in result.directions
    )
    assert result.surface_areas.shape == (8,) and result.number_densities.shape == (8,)
    assert result.total_surface_area == pytest.approx(result.surface_areas.sum())


def test_directions_use_their_own_q_grid_and_measured_q_by_default():
    pop = Population(radius=300.0, beta=0.5, delta_rho=DRHO)
    d1 = Direction(alpha_q=10.0, q=np.array([0.001, 0.01]))
    d2 = Direction(alpha_q=80.0, q=np.array([0.002, 0.004, 0.02]))
    result = Engine().evaluate(AnisaModel([d1, d2], [pop]))
    assert [r.intensity.size for r in result.directions] == [2, 3]
    with pytest.raises(ValueError, match="no q values"):
        Engine().evaluate(AnisaModel([Direction()], [pop]))
    with pytest.raises(ValueError, match="one q array per direction"):
        Engine().evaluate(AnisaModel([d1, d2], [pop]), [np.array([0.1])])


def test_unsorted_duplicate_and_zero_q_are_handled():
    pop = Population(radius=400.0, beta=0.3, volume_fraction=0.05, delta_rho=DRHO, **ANISOTROPIC)
    q_sorted = np.geomspace(1e-3, 0.08, 30)
    reference = one(pop, TILTED, q_sorted)
    perm = np.random.default_rng(2).permutation(q_sorted.size)
    np.testing.assert_allclose(one(pop, TILTED, q_sorted[perm]), reference[perm], rtol=1e-12)
    q = np.array([0.0, 0.01, 0.01, 0.05])
    out = one(pop, TILTED, q)
    assert np.all(np.isfinite(out)) and out[1] == out[2]
    volume = 4.0 / 3.0 * np.pi * 0.3 * 400.0**3
    assert out[0] == pytest.approx(1e8 * 0.05 * volume * DRHO_A**2, rel=1e-9)  # I(0)
    assert (
        one(
            pop,
            TILTED,
            np.array([]),
        ).size
        == 0
    )


# -- caching and threading --------------------------------------------------------------------


def test_cache_does_not_change_results_and_is_used():
    pop = Population(radius=500.0, beta=0.3, volume_fraction=0.05, delta_rho=DRHO, **ANISOTROPIC)
    model = AnisaModel([TILTED, Direction(alpha_q=10.0)], [pop])
    q = [Q, Q]
    engine = Engine()
    first = engine.evaluate(model, q)
    entries = len(engine._unit)
    second = engine.evaluate(model, q)
    assert len(engine._unit) == entries  # nothing recomputed
    for a, b in zip(first.directions, second.directions, strict=True):
        np.testing.assert_array_equal(a.intensity, b.intensity)
    np.testing.assert_array_equal(
        calculate(model, q).directions[0].intensity, first.directions[0].intensity
    )


def test_volume_fraction_and_interference_changes_reuse_the_cache():
    pop = Population(radius=500.0, beta=0.3, volume_fraction=0.05, delta_rho=DRHO, **ANISOTROPIC)
    model = AnisaModel([TILTED], [pop])
    engine = Engine()
    base = engine.evaluate(model, [Q]).directions[0].intensity
    entries = len(engine._unit)
    pop.volume_fraction = 0.1
    doubled = engine.evaluate(model, [Q]).directions[0].intensity
    pop.use_interference, pop.pack, pop.eta = True, 1.0, 300.0
    engine.evaluate(model, [Q])
    assert len(engine._unit) == entries
    np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-12)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: setattr(p, "radius", p.radius * 1.05),
        lambda p: setattr(p, "beta", p.beta * 1.05),
        lambda p: setattr(p, "delta_rho", p.delta_rho * 1.1),
        lambda p: setattr(p.alpha_dist, "par2", p.alpha_dist.par2 + 1.0),
        lambda p: setattr(p.omega_dist, "par1", p.omega_dist.par1 + 10.0),
        lambda p: setattr(p, "size_dist", SizeDistribution("gauss", {"width": 50.0})),
    ],
)
def test_changing_a_shape_parameter_changes_the_result(change):
    pop = Population(radius=500.0, beta=0.3, volume_fraction=0.05, delta_rho=DRHO, **ANISOTROPIC)
    model = AnisaModel([TILTED], [pop])
    engine = Engine()
    before = engine.evaluate(model, [Q]).directions[0].intensity.copy()
    change(pop)
    after = engine.evaluate(model, [Q]).directions[0].intensity
    assert not np.allclose(before, after, rtol=1e-6)


def test_results_do_not_depend_on_the_number_of_threads():
    rng = np.random.default_rng(9)
    pops = [
        Population(
            radius=300.0 + 100 * i,
            beta=0.2 + 0.7 * i,
            volume_fraction=0.02,
            delta_rho=DRHO,
            alpha_dist=AngularDistribution(par1=rng.uniform(0, 90), par2=rng.uniform(1, 9)),
            size_dist=SizeDistribution("gauss", {"width": 40.0 + 20 * i}),
        )
        for i in range(4)
    ]
    dirs = [Direction(alpha_q=float(a)) for a in (0, 25, 50, 75, 90)]
    q = [np.geomspace(1e-3, 0.08, 40)] * 5
    serial = Engine().evaluate(AnisaModel(dirs, pops, CalcSettings(n_threads=1)), q)
    parallel = Engine().evaluate(AnisaModel(dirs, pops, CalcSettings(n_threads=8)), q)
    for a, b in zip(serial.directions, parallel.directions, strict=True):
        np.testing.assert_array_equal(a.intensity, b.intensity)


# -- resolution rules and helpers -------------------------------------------------------------


def test_q_bands_partition_all_indices_and_respect_the_ratio():
    q = np.random.default_rng(1).permutation(np.geomspace(1e-4, 0.3, 200))
    bands = q_bands(q)
    covered = np.sort(np.concatenate(bands))
    np.testing.assert_array_equal(covered, np.arange(q.size))
    for band in bands:
        assert (
            q[band].max() / max(q[band].min(), 1e-300) <= eng.BAND_RATIO * 1.0001 or band.size == 1
        )


def test_q_bands_edge_cases():
    assert q_bands(np.array([])) == []
    assert len(q_bands(np.array([0.1]))) == 1
    assert sum(b.size for b in q_bands(np.array([0.0, 0.0, 0.5]))) == 3
    assert sum(b.size for b in q_bands(np.full(5, 0.1))) == 5


def test_grid_size_grows_with_span_and_is_quantised():
    s = CalcSettings()
    small = resolved_grid_size(s, 5.0)
    assert small == (s.n_alpha, s.n_omega)  # user settings are a lower bound
    large = resolved_grid_size(s, 400.0)
    assert large[0] >= 400 and large[1] >= 800
    assert resolved_grid_size(s, 400.0) == resolved_grid_size(s, 401.0)  # quantised: cache hits
    assert eng._quantise(129) % 8 == 0 and eng._quantise(129) >= 129


def test_unresolvable_grid_warns_instead_of_failing_silently():
    with pytest.warns(RuntimeWarning, match="capped"):
        n_alpha, n_omega = resolved_grid_size(CalcSettings(), 5000.0)
    assert n_alpha * n_omega <= eng.MAX_GRID_NODES * 1.0001
