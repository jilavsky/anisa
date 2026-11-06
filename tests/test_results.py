"""Results file: round trip, standard errors, and readability by pyIrena's NXcanSAS readers."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from pyirena.io import discover_scattering, load_scattering

from anisa import testsets
from anisa.core.engine import Engine
from anisa.core.fitting import fit
from anisa.io.project import load_project
from anisa.io.results import (
    RESULTS_SUFFIX,
    load_results,
    parameter_name,
    read_summary,
    save_results,
)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    folder = tmp_path_factory.mktemp("results")
    truth = testsets.write_test_system("prolate_oblate", folder, "dat")
    model = load_project(truth.with_name("prolate_oblate_start.anisa.json"))
    result = fit(model, engine=Engine())
    return model, result, folder


def test_suffix_is_appended_and_file_written_atomically(fitted, tmp_path):
    model, _, _ = fitted
    path = save_results(model, tmp_path / "out")
    assert path.name == "out" + RESULTS_SUFFIX and path.is_file()
    assert not list(tmp_path.glob(".*.tmp"))
    assert save_results(model, tmp_path / "plain.h5").name == "plain.h5"


def test_round_trip_restores_model_and_data_without_the_source_files(fitted, tmp_path):
    model, result, folder = fitted
    path = save_results(model, tmp_path / "r", fit=result)
    back = load_results(path)
    assert back.to_dict() == model.to_dict()
    for a, b in zip(model.directions, back.directions, strict=True):
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.intensity, b.intensity)
        np.testing.assert_array_equal(a.uncertainty, b.uncertainty)
        assert b.source is not None and b.source.fingerprint == a.source.fingerprint


def test_file_has_one_sasentry_per_direction_readable_by_pyirena(fitted, tmp_path):
    model, result, _ = fitted
    path = save_results(model, tmp_path / "r", fit=result)
    locations = discover_scattering(path)
    assert len(locations) == len(model.directions)
    for location, direction in zip(locations, model.directions, strict=True):
        record = load_scattering(location)
        np.testing.assert_allclose(record.q, direction.q)
        np.testing.assert_allclose(record.intensity, direction.intensity)


def test_model_curves_are_stored_and_consistent(fitted, tmp_path):
    model, result, _ = fitted
    path = save_results(model, tmp_path / "r", fit=result)
    expected = Engine().evaluate(model, [d.q for d in model.directions])
    with h5py.File(path, "r") as f:
        for i, calculated in enumerate(expected.directions, start=1):
            process = f[f"entry/dir{i:02d}/anisa_results"]
            np.testing.assert_allclose(process["I_model"][()], calculated.intensity)
            np.testing.assert_allclose(
                process["I_populations"][()].sum(axis=0) + process["background"][()],
                process["I_model"][()],
                rtol=1e-12,
            )
            assert process.attrs["alpha_q_deg"] == model.directions[i - 1].alpha_q
            assert "fit_quality" in process
            residuals = process["residuals"][()]
            assert 0.7 < np.sqrt(np.mean(residuals**2)) < 1.4  # a good fit: chi ~ 1


def test_summary_has_values_errors_limits_and_derived_quantities(fitted, tmp_path):
    model, result, _ = fitted
    summary = read_summary(save_results(model, tmp_path / "r", fit=result))
    name = parameter_name("population", 1, "radius")
    entry = summary.parameters[name]
    assert entry["value"] == pytest.approx(model.populations[1].radius)
    assert (
        entry["fit"]
        and entry["std"] > 0
        and entry["limit_low"] < entry["value"] < entry["limit_high"]
    )
    assert np.isnan(summary.parameters[parameter_name("population", 1, "beta")]["std"])
    assert summary.fit["reduced_chi_squared"] == pytest.approx(result.reduced_chi_squared)
    assert summary.derived["surface_area"].shape == (2,)
    assert summary.derived["total_surface_area"] == pytest.approx(
        summary.derived["surface_area"].sum()
    )
    assert len(summary.direction_quality) == len(model.directions)
    assert summary.direction_quality[0]["n_valid"] > 0


def test_directions_without_data_are_kept(tmp_path):
    model = testsets.make_system("spheres").model  # no data attached
    path = save_results(model, tmp_path / "nodata")
    back = load_results(path)
    assert len(back.directions) == 6 and all(d.q is None for d in back.directions)


def test_rejects_files_that_are_not_results(tmp_path):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_group("entry")
    with pytest.raises(ValueError, match="not an Anisa results file"):
        load_results(other)
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "missing.h5")
    with pytest.raises(ValueError, match="not an Anisa results file"):
        read_summary(other)


def test_mismatched_result_is_refused(fitted, tmp_path):
    model, _, _ = fitted
    many = testsets.make_system("many").model  # twelve directions against the model's six
    q = np.geomspace(1e-3, 0.3, 10)
    other = Engine().evaluate(many, [q] * len(many.directions))
    with pytest.raises(ValueError, match="does not match"):
        save_results(model, tmp_path / "x", result=other)
