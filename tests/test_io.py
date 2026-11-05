"""Data loading through pyIrena, and project files."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from anisa import demo
from anisa.core.model import AnisaModel, DataSource, Direction, Population
from anisa.io.data import (
    content_fingerprint,
    discover_curves,
    find_wavelength,
    load_direction,
    read_wavelength,
    reload_direction_data,
)
from anisa.io.project import PROJECT_SUFFIX, load_project, save_project

Q = np.geomspace(1e-4, 0.1, 40)
I = 1e3 * Q**-2.5  # noqa: E741 - intensity
DI = 0.04 * I


@pytest.fixture
def demo_dir(tmp_path):
    """A written demo project (HDF5 data) and its folder."""
    demo.write_demo_project(tmp_path / "proj")
    return tmp_path / "proj"


def write_text(path: Path, q=Q, i=I, di=DI):
    cols = (q, i) if di is None else (q, i, di)
    np.savetxt(path, np.column_stack(cols), header="Q I dI", fmt="%.12e")
    return path


def write_simple_h5(path: Path, groups: dict[str, tuple]):
    with h5py.File(path, "w") as handle:
        for name, (q, i) in groups.items():
            group = handle.create_group(name)
            group["Q"], group["I"] = q, i
    return path


# -- fit mask ---------------------------------------------------------------------------------


def test_fit_mask_excludes_bad_points_and_applies_the_q_range():
    q = np.array([0.0, 0.001, 0.002, np.nan, 0.01, 0.02, 0.05, -1.0])
    i = np.array([1.0, 2.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0])
    sigma = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, np.nan, 1.0])
    d = Direction(q=q, intensity=i, uncertainty=sigma)
    # Q <= 0, non-finite Q or I, zero or NaN uncertainty are all dropped
    np.testing.assert_array_equal(
        d.fit_mask(), [False, True, False, False, False, True, False, False]
    )
    d.q_min, d.q_max = 0.0015, 0.03
    np.testing.assert_array_equal(
        d.fit_mask(), [False, False, False, False, False, True, False, False]
    )


def test_fit_mask_without_data_or_uncertainty():
    assert Direction().fit_mask().size == 0
    d = Direction(q=np.array([0.1, 0.2]), intensity=np.array([1.0, 2.0]))
    np.testing.assert_array_equal(d.fit_mask(), [True, True])


# -- loading ----------------------------------------------------------------------------------


def test_load_text_and_hdf5_give_the_same_curve(tmp_path):
    txt = write_text(tmp_path / "c.dat")
    from pyirena.io.nxcansas_unified import create_nxcansas_file

    h5 = tmp_path / "c.h5"
    create_nxcansas_file(h5, Q, I, DI, sample_name="c")
    for path in (txt, h5):
        d = load_direction(path, alpha_q=30.0, omega_q=45.0)
        np.testing.assert_allclose(d.q, Q, rtol=1e-11)
        np.testing.assert_allclose(d.intensity, I, rtol=1e-11)
        np.testing.assert_allclose(d.uncertainty, DI, rtol=1e-11)
        assert (d.alpha_q, d.omega_q) == (30.0, 45.0)
        assert d.source.fingerprint == content_fingerprint(path)
        assert Path(d.source.path) == path.resolve()


def test_label_defaults_to_the_file_name_and_can_be_set(tmp_path):
    path = write_text(tmp_path / "sample_x.dat")
    assert load_direction(path).label == "sample_x.dat"
    assert load_direction(path, label="my cut").label == "my cut"


def test_missing_uncertainty_uses_the_error_fraction(tmp_path):
    path = write_text(tmp_path / "noerr.dat", di=None)
    d = load_direction(path, error_fraction=0.1)
    np.testing.assert_allclose(d.uncertainty, 0.1 * d.intensity, rtol=1e-9)
    assert d.source.error_fraction == 0.1


def test_text_q_unit_is_converted_to_inverse_angstrom(tmp_path):
    path = write_text(tmp_path / "nm.dat", q=Q * 10.0)  # the same curve with Q in 1/nm
    d = load_direction(path, q_unit="1/nm")
    np.testing.assert_allclose(d.q, Q, rtol=1e-10)
    assert d.source.q_unit == "1/nm"


def test_descending_q_is_sorted_ascending_with_its_uncertainty(tmp_path):
    path = write_text(tmp_path / "desc.dat", q=Q[::-1], i=I[::-1], di=DI[::-1])
    d = load_direction(path)
    np.testing.assert_allclose(d.q, Q, rtol=1e-11)
    np.testing.assert_allclose(d.intensity, I, rtol=1e-11)
    np.testing.assert_allclose(d.uncertainty, DI, rtol=1e-11)


def test_files_with_several_curves_need_a_choice(tmp_path):
    path = write_simple_h5(tmp_path / "two.h5", {"a": (Q, I), "b": (Q, 2 * I)})
    curves = discover_curves(path)
    assert len(curves) == 2
    with pytest.raises(ValueError, match="2 curves"):
        load_direction(path)
    first = load_direction(path, internal_path=curves[0].internal_path)
    second = load_direction(path, internal_path=curves[1].internal_path)
    assert np.allclose(second.intensity, 2 * first.intensity) or np.allclose(
        first.intensity, 2 * second.intensity
    )
    with pytest.raises(ValueError, match="no curve at"):
        load_direction(path, internal_path="/nonexistent")


def test_slit_smeared_curves_load_with_a_warning(tmp_path):
    path = write_simple_h5(tmp_path / "smr.h5", {"entry_smr": (Q, I)})
    with pytest.warns(UserWarning, match="slit-smeared"):
        d = load_direction(path)
    assert d.source.variant == "slit-smeared"


def test_non_cm_intensity_unit_warns(tmp_path):
    from pyirena.io.nxcansas_unified import create_nxcansas_file

    path = tmp_path / "u.h5"
    create_nxcansas_file(path, Q, I, DI, sample_name="u")
    with h5py.File(path, "r+") as handle:
        handle["entry/u/sasdata/I"].attrs["units"] = "arb"
    with pytest.warns(UserWarning, match="not 1/cm"):
        load_direction(path)


def test_bad_inputs_raise_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_direction(tmp_path / "nope.dat")
    bad = tmp_path / "x.xyz"
    bad.write_text("1 2 3")
    with pytest.raises(ValueError, match="unsupported"):
        load_direction(bad)


def test_fingerprint_follows_content_not_location(tmp_path):
    a = write_text(tmp_path / "a.dat")
    b = tmp_path / "copy.dat"
    shutil.copy(a, b)
    assert content_fingerprint(a) == content_fingerprint(b)
    write_text(b, i=I * 1.01)
    assert content_fingerprint(a) != content_fingerprint(b)


# -- wavelength -------------------------------------------------------------------------------


def test_wavelength_is_read_from_hdf5_and_absent_for_text(tmp_path):
    from pyirena.io.nxcansas_unified import create_nxcansas_file

    h5 = tmp_path / "w.h5"
    create_nxcansas_file(h5, Q, I, DI, sample_name="w", metadata={"instrument.wavelength": 0.59})
    assert read_wavelength(h5) == pytest.approx(0.59)
    assert read_wavelength(write_text(tmp_path / "w.dat")) is None
    assert read_wavelength(tmp_path / "missing.h5") is None


def test_wavelength_units_are_converted(tmp_path):
    from pyirena.io.nxcansas_unified import create_nxcansas_file

    h5 = tmp_path / "nm.h5"
    create_nxcansas_file(h5, Q, I, DI, sample_name="w", metadata={"instrument.wavelength": 0.059})
    with h5py.File(h5, "r+") as handle:
        handle["entry/instrument/wavelength"].attrs["units"] = "nm"
    assert read_wavelength(h5) == pytest.approx(0.59)


def test_find_wavelength_over_directions(demo_dir):
    model = load_project(demo_dir / "demo.anisa.json")
    assert find_wavelength(model.directions) == pytest.approx(model.settings.wavelength)
    assert find_wavelength([Direction()]) is None


# -- projects ---------------------------------------------------------------------------------


def test_project_round_trip_restores_model_and_data(demo_dir):
    original = load_project(demo_dir / "demo.anisa.json")
    again_path = save_project(original, demo_dir / "again")
    assert again_path.name == "again" + PROJECT_SUFFIX
    again = load_project(again_path)
    assert again.to_dict() == original.to_dict()
    for a, b in zip(original.directions, again.directions, strict=True):
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.intensity, b.intensity)
        np.testing.assert_array_equal(a.uncertainty, b.uncertainty)
        assert a.source.fingerprint == b.source.fingerprint


def test_project_file_holds_references_not_arrays(demo_dir):
    text = (demo_dir / "demo.anisa.json").read_text()
    document = json.loads(text)  # strict JSON: no NaN / Infinity
    assert document["_anisa_project"]["format"] == "anisa-project"
    directions = document["model"]["directions"]
    assert len(directions) == 4
    for direction in directions:
        assert set(direction) >= {"alpha_q", "omega_q", "source", "background"}
        assert not any(isinstance(v, list) and len(v) > 20 for v in direction.values())
        assert not Path(direction["source"]["path"]).is_absolute()  # relative to the project
        assert Path(direction["source"]["path_absolute"]).is_absolute()
        assert len(direction["source"]["fingerprint"]) == 64
    assert len(text) < 30_000


def test_project_survives_moving_the_whole_folder(demo_dir, tmp_path):
    moved = tmp_path / "elsewhere"
    shutil.copytree(demo_dir, moved)
    shutil.rmtree(demo_dir)  # the absolute paths in the file no longer exist
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = load_project(moved / "demo.anisa.json")
    assert all(d.q is not None for d in model.directions)


def test_project_survives_moving_only_the_project_file(demo_dir, tmp_path):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(demo_dir / "demo.anisa.json", lonely / "demo.anisa.json")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = load_project(lonely / "demo.anisa.json")  # data found through path_absolute
    assert all(d.q is not None for d in model.directions)
    # saving re-derives the relative path from the file's new home
    saved = json.loads(save_project(model, lonely / "resaved").read_text())
    assert saved["model"]["directions"][0]["source"]["path"].startswith("..")


def test_changed_data_file_warns_but_loads(demo_dir):
    victim = demo_dir / "demo_dir02.h5"
    with h5py.File(victim, "r+") as handle:
        handle["entry/dir02/sasdata/I"][0] *= 2.0
    with pytest.warns(UserWarning, match="changed since the project was saved"):
        model = load_project(demo_dir / "demo.anisa.json")
    assert model.directions[1].q is not None


def test_missing_data_file_warns_and_the_project_still_opens(demo_dir):
    (demo_dir / "demo_dir03.h5").unlink()
    with pytest.warns(UserWarning, match="not found"):
        model = load_project(demo_dir / "demo.anisa.json")
    assert [d.q is None for d in model.directions] == [False, False, True, False]
    assert model.directions[2].source is not None  # the reference is kept for a later reload


def test_load_data_false_keeps_references_only(demo_dir):
    model = load_project(demo_dir / "demo.anisa.json", load_data=False)
    assert all(d.q is None and d.source is not None for d in model.directions)
    assert reload_direction_data(model.directions[0], base_dir=demo_dir)
    assert model.directions[0].q is not None


def test_directions_without_a_source_keep_no_data(tmp_path):
    model = AnisaModel([Direction(q=Q, intensity=I, uncertainty=DI, alpha_q=20.0)], [Population()])
    loaded = load_project(save_project(model, tmp_path / "nosource"))
    assert loaded.directions[0].source is None and loaded.directions[0].q is None
    assert loaded.directions[0].alpha_q == 20.0
    assert not reload_direction_data(loaded.directions[0])


def test_q_range_and_use_flag_round_trip(tmp_path):
    d = Direction(alpha_q=10.0, q_min=0.001, q_max=0.05, use=False)
    loaded = load_project(save_project(AnisaModel([d], []), tmp_path / "range"))
    assert (loaded.directions[0].q_min, loaded.directions[0].q_max) == (0.001, 0.05)
    assert loaded.directions[0].use is False


def test_bad_project_files_are_rejected(tmp_path):
    notjson = tmp_path / "a.json"
    notjson.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_project(notjson)
    other = tmp_path / "b.json"
    other.write_text('{"hello": 1}')
    with pytest.raises(ValueError, match="not an Anisa project"):
        load_project(other)
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "missing.json")


def test_newer_project_schema_warns(tmp_path):
    path = save_project(AnisaModel(), tmp_path / "new")
    document = json.loads(path.read_text())
    document["_anisa_project"]["schema_version"] = 999
    document["model"]["some_future_key"] = {"x": 1}
    path.write_text(json.dumps(document))
    with pytest.warns(UserWarning, match="newer Anisa"):
        load_project(path)


def test_save_is_atomic_and_leaves_no_temporary_files(tmp_path, monkeypatch):
    path = save_project(AnisaModel(populations=[Population(radius=111.0)]), tmp_path / "keep")
    before = path.read_text()
    import anisa.io.project as project_module

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(project_module.json, "dump", explode)
    with pytest.raises(RuntimeError):
        save_project(AnisaModel(populations=[Population(radius=999.0)]), path)
    assert path.read_text() == before  # the old project is intact
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # and no temp file is left


def test_saved_project_has_normal_file_permissions(tmp_path):
    path = save_project(AnisaModel(), tmp_path / "perm")
    assert path.stat().st_mode & 0o044, "project must be readable by group/others per the umask"


def test_data_source_dict_round_trip():
    s = DataSource(
        path="a.h5", internal_path="entry/x/sasdata", fingerprint="ab" * 32, q_unit="1/nm"
    )
    assert DataSource.from_dict(s.to_dict()) == s
    assert DataSource.from_dict(None) is None and DataSource.from_dict({"path": ""}) is None
