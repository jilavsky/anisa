"""Bernardyn package export (skipped where Bernardyn is not installed)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

pytest.importorskip("bernardyn")

from anisa import testsets  # noqa: E402
from anisa.io.bernardyn_export import export_bernardyn_package  # noqa: E402
from anisa.io.project import load_project  # noqa: E402


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    truth = testsets.write_test_system("prolate", tmp_path_factory.mktemp("bx"), "dat")
    return load_project(truth)


def test_package_holds_data_and_model_of_every_direction(model, tmp_path):
    path = export_bernardyn_package(model, tmp_path / "plot", title="prolate")
    assert path.is_file() and path.suffix != ""
    with h5py.File(path, "r") as f:
        assert f.attrs["bernardyn_format"]
        datasets = f["datasets"]
        assert len(datasets) == 2 * len(model.directions)  # data and model for each direction


def test_overwrite_is_refused_by_default(model, tmp_path):
    export_bernardyn_package(model, tmp_path / "p")
    with pytest.raises(FileExistsError):
        export_bernardyn_package(model, tmp_path / "p")
    export_bernardyn_package(model, tmp_path / "p", overwrite=True, recipe="porod")


def test_no_data_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no direction has data"):
        export_bernardyn_package(testsets.make_system("prolate").model, tmp_path / "x")


def test_model_curve_in_the_package_matches_the_engine(model, tmp_path):
    from anisa.core.engine import Engine

    path = export_bernardyn_package(model, tmp_path / "m")
    expected = Engine().evaluate(model, [d.q for d in model.directions]).directions[0].intensity
    with h5py.File(path, "r") as f:
        found = False
        for group in f["datasets"].values():
            if (
                "data" in group
                and "I" in group["data"]
                and np.allclose(group["data"]["I"][()], expected)
            ):
                found = True
        assert found


def test_cli_and_api_export(model, tmp_path, capsys):
    import json

    from anisa import cli

    truth = tmp_path / "t.anisa.json"
    from anisa.io.project import save_project

    save_project(model, truth)
    assert (
        cli.main(["export-bernardyn", str(truth), str(tmp_path / "c"), "--recipe", "kratky"]) == 0
    )
    document = json.loads(capsys.readouterr().out)
    assert document["recipe"] == "kratky" and document["package"]
    assert cli.main(["export-bernardyn", str(truth), str(tmp_path / "c")]) == 2  # exists
    assert "already exists" in capsys.readouterr().err
