"""Synthetic demo data: reproducible, statistically right, physically anisotropic."""

from __future__ import annotations

import numpy as np
import pytest

from anisa import demo
from anisa.core.engine import Engine
from anisa.io.data import load_direction
from anisa.io.project import load_project


def test_demo_model_has_the_documented_structure():
    model = demo.demo_model()
    assert [p.name for p in model.populations] == [
        "intrasplat cracks",
        "interlamellar pores",
        "globular voids",
    ]
    assert len(model.directions) == 4
    assert model.directions[0].q is None  # no data until synthesize()


def test_synthesize_is_reproducible_and_seed_dependent():
    a = demo.synthesize(demo.demo_model(), seed=3)
    b = demo.synthesize(demo.demo_model(), seed=3)
    c = demo.synthesize(demo.demo_model(), seed=4)
    for da, db, dc in zip(a.directions, b.directions, c.directions, strict=True):
        np.testing.assert_array_equal(da.intensity, db.intensity)
        assert not np.array_equal(da.intensity, dc.intensity)
        assert np.all(da.uncertainty > 0) and np.all(np.isfinite(da.intensity))


def test_noise_matches_the_stored_uncertainty():
    """(data - truth) / uncertainty must be standard normal, or fits would see biased chi^2."""
    model = demo.demo_model()
    truth = Engine().evaluate(model, [np.geomspace(1e-4, 0.08, 120)] * 4)
    demo.synthesize(model, relative_noise=0.05, seed=11)
    z = np.concatenate(
        [
            (d.intensity - t.intensity) / d.uncertainty
            for d, t in zip(model.directions, truth.directions, strict=True)
        ]
    )
    assert abs(z.mean()) < 0.15 and 0.9 < z.std() < 1.1  # 480 points
    assert np.max(np.abs(z)) < 4.5


def test_demo_is_anisotropic_where_the_physics_says():
    """Flat pores parallel to the substrate scatter far more along the normal than in plane;
    in the substrate plane x and y are equivalent (axes isotropic in omega)."""
    model = demo.synthesize(demo.demo_model(), relative_noise=0.005, seed=0)
    along, in_x, in_y, tilted = (d.intensity for d in model.directions)
    mid = slice(40, 90)
    assert np.median(along[mid] / in_x[mid]) > 5.0
    np.testing.assert_allclose(in_x[mid], in_y[mid], rtol=0.05)
    assert np.all(tilted[mid] < along[mid])


@pytest.mark.parametrize("fmt", ["h5", "dat"])
def test_written_files_reload_to_the_same_data(tmp_path, fmt):
    model = demo.synthesize(demo.demo_model(), seed=2)
    paths = demo.write_files(model, tmp_path, fmt)
    assert [p.name for p in paths] == [f"demo_dir{i:02d}.{fmt}" for i in range(1, 5)]
    for direction, path in zip(model.directions, paths, strict=True):
        again = load_direction(path, alpha_q=direction.alpha_q, omega_q=direction.omega_q)
        np.testing.assert_allclose(again.q, direction.q, rtol=1e-9)
        np.testing.assert_allclose(again.intensity, direction.intensity, rtol=1e-9)
        np.testing.assert_allclose(again.uncertainty, direction.uncertainty, rtol=1e-9)
        assert direction.source.fingerprint == again.source.fingerprint


def test_write_files_needs_data_and_a_valid_format(tmp_path):
    with pytest.raises(ValueError, match="no data"):
        demo.write_files(demo.demo_model(), tmp_path)
    with pytest.raises(ValueError, match="fmt"):
        demo.write_files(demo.synthesize(demo.demo_model()), tmp_path, "pdf")


def test_perturb_model_moves_parameters_sets_fit_flags_and_keeps_the_original():
    truth = demo.synthesize(demo.demo_model(), seed=0)
    before = truth.to_dict()
    start = demo.perturb_model(truth, seed=5, fraction=0.25)
    assert truth.to_dict() == before  # the original is untouched
    for t, s in zip(truth.populations, start.populations, strict=True):
        for path in ("radius", "volume_fraction"):
            ratio = s.get_parameter(path) / t.get_parameter(path)
            assert ratio != 1.0 and abs(ratio - 1.0) <= 0.25
            assert s.fit_spec(path).fit and not t.fit_spec(path).fit
    assert not start.populations[2].fit_spec("alpha_par2").fit  # isotropic population
    again = demo.perturb_model(truth, seed=5, fraction=0.25)
    assert again.to_dict() == start.to_dict()  # deterministic
    np.testing.assert_array_equal(start.directions[0].intensity, truth.directions[0].intensity)


def test_demo_project_is_complete_and_loadable(tmp_path):
    project = demo.write_demo_project(tmp_path / "d")
    start = project.with_name("demo_start.anisa.json")
    assert start.exists()
    truth = load_project(project)
    guess = load_project(start)
    assert all(d.q is not None for d in truth.directions + guess.directions)
    np.testing.assert_array_equal(truth.directions[0].intensity, guess.directions[0].intensity)
    assert truth.populations[0].radius != guess.populations[0].radius
    assert guess.populations[0].fit_spec("radius").fit


def test_command_line_writes_the_demo(tmp_path, capsys):
    assert (
        demo.main([str(tmp_path / "cli"), "--format", "dat", "--noise", "0.02", "--seed", "7"]) == 0
    )
    assert "demo.anisa.json" in capsys.readouterr().out
    assert len(list((tmp_path / "cli").glob("demo_dir*.dat"))) == 4
    assert load_project(tmp_path / "cli" / "demo.anisa.json").directions[3].q is not None


# -- test systems ---------------------------------------------------------------------------

from anisa import testsets  # noqa: E402
from anisa.core.fitting import fit  # noqa: E402


@pytest.mark.parametrize("name", testsets.list_systems())
def test_every_test_system_writes_and_reloads(tmp_path, name):
    truth = testsets.write_test_system(name, tmp_path, "dat")
    folder = tmp_path / name
    assert (folder / "TRUTH.txt").read_text().startswith(testsets.make_system(name).title)
    model = load_project(truth)
    start = load_project(folder / f"{name}_start.anisa.json")
    assert len(model.directions) == len(start.directions) >= 6
    assert all(d.q is not None and d.source is not None for d in model.directions)
    assert any(p.fit_spec(path).fit for p in start.populations for path in p.parameter_paths())


def test_unknown_system_is_rejected():
    with pytest.raises(ValueError, match="unknown test system"):
        testsets.make_system("nope")


def test_many_has_twelve_directions_and_spheres_are_isotropic():
    assert len(testsets.make_system("many").model.directions) == 12
    system = testsets.make_system("spheres")
    q = np.geomspace(system.q_min, system.q_max, 30)
    result = Engine().evaluate(system.model, [q] * len(system.model.directions))
    for direction in result.directions[1:]:
        np.testing.assert_allclose(direction.intensity, result.directions[0].intensity, rtol=1e-6)


def test_prolate_and_oblate_have_opposite_anisotropy():
    """Along the axis a prolate particle is long (steeper decay), an oblate one thin (slower)."""
    q = np.array([0.03])
    ratios = {}
    for name in ("prolate", "oblate"):
        model = testsets.make_system(name).model
        r = Engine().evaluate(model, [q] * len(model.directions))
        ratios[name] = r.directions[0].intensity[0] / r.directions[3].intensity[0]  # normal / plane
    assert ratios["prolate"] < 0.2 and ratios["oblate"] > 5.0


def test_interference_system_differs_from_the_dilute_one():
    system = testsets.make_system("interference")
    q = np.geomspace(2e-3, 0.05, 40)
    with_s = Engine().evaluate(system.model, [q] * 6).directions[3].intensity
    system.model.populations[0].use_interference = False
    without = Engine().evaluate(system.model, [q] * 6).directions[3].intensity
    assert with_s[0] < 0.4 * without[0]  # S(0) = 1/(1 + pack) = 0.25
    assert np.max(with_s / without) > 1.3  # the correlation hump


@pytest.mark.parametrize("name", testsets.list_systems())
def test_fit_recovers_each_test_system(tmp_path, name):
    """A perturbed start returns to the truth within a few sigma, chi^2/N ~ 1."""
    truth_path = testsets.write_test_system(name, tmp_path, "dat")
    truth = load_project(truth_path)
    start = load_project(truth_path.with_name(f"{name}_start.anisa.json"))
    result = fit(start, engine=Engine())
    assert 0.8 < result.reduced_chi_squared < 1.3
    for item in result.parameters:
        par = item.parameter
        true_value = truth.populations[par.index].get_parameter(par.path)
        assert abs(item.value - true_value) < 5.0 * item.std, (par.label, item.value, true_value)
