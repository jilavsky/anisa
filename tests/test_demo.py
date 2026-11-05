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
