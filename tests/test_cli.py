"""``anisa.api`` and the command line: same results headless as the library gives."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from anisa import api, cli, testsets
from anisa.io.project import load_project


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cli")
    paths = api.write_test_system("prolate_oblate", folder, fmt="dat")
    return paths


def run(capsys, *argv) -> tuple[int, object]:
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip().startswith(("{", "[")) else out


def test_every_api_result_is_json_serialisable(system, tmp_path):
    json.dumps(api.describe(system["start"]), allow_nan=False)
    json.dumps(api.calculate(system["truth"]), allow_nan=False)
    json.dumps(
        api.anisotropy(system["truth"], sweep="alpha", fixed=0.0, q_values=[0.01], n_points=13),
        allow_nan=False,
    )
    report = api.fit(
        system["start"], results=tmp_path / "r.anisa.h5", monte_carlo_runs=3, workers=1
    )
    json.dumps(report, allow_nan=False)
    json.dumps(api.summary(report["results_file"]), allow_nan=False)


def test_info_lists_populations_directions_and_flags(system, capsys):
    code, doc = run(capsys, "info", system["start"])
    assert code == 0
    assert [p["name"] for p in doc["populations"]] == [
        "needles, strongly aligned",
        "platelets, weakly aligned",
    ]
    assert len(doc["directions"]) == 6 and doc["fit_parameters"]


def test_calc_of_the_truth_gives_chi_squared_near_one(system, capsys, tmp_path):
    out = tmp_path / "calc.anisa.h5"
    code, doc = run(capsys, "calc", system["truth"], "--results", str(out))
    assert code == 0 and out.is_file()
    assert 0.85 < doc["statistics"]["reduced_chi_squared"] < 1.15
    assert len(doc["surface_area"]) == 2


def test_fit_recovers_the_truth_and_writes_results(system, capsys, tmp_path):
    out = tmp_path / "fit.anisa.h5"
    code, doc = run(capsys, "fit", system["start"], "--results", str(out))
    assert code == 0 and doc["success"] and out.is_file()
    assert 0.8 < doc["reduced_chi_squared"] < 1.3
    truth = load_project(system["truth"])
    for p in doc["parameters"]:
        true_value = truth.populations[p["index"] - 1].get_parameter(p["path"])
        assert abs(p["value"] - true_value) < 5 * p["std"]
    # the CLI result is the library result
    assert api.summary(out)["fit"]["reduced_chi_squared"] == pytest.approx(
        doc["reduced_chi_squared"]
    )


def test_fit_text_report_and_update_project(system, capsys, tmp_path):
    import shutil

    copy = tmp_path / "copy.anisa.json"
    shutil.copy(system["start"], copy)
    # project data paths are absolute fallbacks, so the copy still finds its data
    code = cli.main(
        ["fit", str(copy), "--text", "--update-project", "--results", str(tmp_path / "x")]
    )
    text = capsys.readouterr().out
    assert code == 0 and text.startswith("Fit converged") and "->" in text
    updated = load_project(copy)
    start = load_project(system["start"])
    assert updated.populations[1].radius != start.populations[1].radius
    assert abs(updated.populations[1].radius - 450.0) < 5.0


def test_fit_without_flags_is_a_clean_error(system, capsys, tmp_path):
    model = load_project(system["start"])
    for p in model.populations:
        for spec in p.fit_specs.values():
            spec.fit = False
    from anisa.io.project import save_project

    path = save_project(model, tmp_path / "noflags.anisa.json")
    assert cli.main(["fit", str(path)]) == 2
    assert "no parameter is marked for fitting" in capsys.readouterr().err


def test_missing_file_is_exit_status_2(capsys, tmp_path):
    assert cli.main(["info", str(tmp_path / "nope.anisa.json")]) == 2
    assert "nope" in capsys.readouterr().err


def test_anisotropy_scan_and_csv(system, capsys, tmp_path):
    csv_path = tmp_path / "scan.csv"
    code, doc = run(
        capsys, "anisotropy", system["truth"], "--sweep", "alpha", "--fixed", "0",
        "--q", "0.01", "0.05", "--points", "37", "--csv", str(csv_path),
    )  # fmt: skip
    assert code == 0 and len(doc["angles"]) == 37 and len(doc["intensity"]) == 2
    assert len(doc["populations"]) == 2
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 38 and lines[0].startswith("angle_deg,I(Q=0.01)")
    # needles aligned along the normal scatter least along the normal... and the scan is
    # periodic: 0 deg and 360 deg are the same direction
    assert doc["intensity"][0][0] == pytest.approx(doc["intensity"][0][-1], rel=1e-9)
    assert any(len(m["angles"]) for m in doc["measured"])  # the 0 deg line has data on it


def test_anisotropy_needs_a_scan_definition(system):
    with pytest.raises(ValueError, match="no saved scan"):
        api.anisotropy(system["truth"])
    with pytest.raises(ValueError, match="sweep, fixed and q_values"):
        api.anisotropy(system["truth"], sweep="omega")


def test_testdata_list_and_write(capsys, tmp_path):
    code, doc = run(capsys, "testdata", "--list")
    assert code == 0 and [d["name"] for d in doc] == testsets.list_systems()
    code, doc = run(capsys, "testdata", str(tmp_path), "--system", "oblate", "--format", "dat")
    assert code == 0 and (tmp_path / "oblate" / "TRUTH.txt").is_file()
    assert cli.main(["testdata"]) == 2


def test_python_dash_m_anisa_runs(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "anisa", "testdata", "--list"], capture_output=True, text=True
    )
    assert done.returncode == 0 and "prolate" in done.stdout
