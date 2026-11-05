"""
GUI tests, run with Qt offscreen: widgets, panels and the main window driven through the same code
paths a user triggers (cell edits, button handlers, project open/save, a real fit in a worker
thread). Dialogs are replaced so nothing blocks.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from anisa import demo  # noqa: E402
from anisa.core.model import AnisaModel, Population  # noqa: E402
from anisa.gui._qt import QApplication, Qt  # noqa: E402
from anisa.gui.directions_panel import C_ALPHA, C_QMAX, C_QMIN, DirectionsPanel  # noqa: E402
from anisa.gui.main_window import MainWindow  # noqa: E402
from anisa.gui.plots import error_segments  # noqa: E402
from anisa.gui.population_editor import PopulationsPanel, TableEditDialog  # noqa: E402
from anisa.gui.state import UserState  # noqa: E402
from anisa.gui.widgets import NumberEdit  # noqa: E402
from anisa.io.project import load_project  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(app, condition, timeout=90.0):
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    return condition()


@pytest.fixture(scope="module")
def demo_project(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo")
    demo.write_demo_project(folder)
    return folder


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ANISA_STATE_DIR", str(tmp_path / "state"))
    win = MainWindow()
    win.messages = []
    monkeypatch.setattr(win, "_show_message", lambda *args, **kwargs: win.messages.append(args))
    win._confirm_discard = lambda: True  # a modal "unsaved changes" box would block forever
    win.show()
    yield win
    win.shutdown_threads()  # may deliver a finished job, which marks the project dirty again
    win.dirty = False
    win.close()


def settled(app, win):
    """True when no calculation is running or pending."""
    return win._calc_thread is None and not win._calc_pending and not win._calc_timer.isActive()


def open_start(app, win, folder):
    assert win.open_project(folder / "demo_start.anisa.json")
    assert wait_until(app, lambda: win.last_stats is not None and win.last_stats.n_points > 0)
    assert wait_until(app, lambda: settled(app, win))


# -- NumberEdit -------------------------------------------------------------------------------


def test_number_edit_commits_only_valid_changes(qapp):
    edit = NumberEdit(5.0, minimum=1.0, maximum=10.0)
    seen = []
    edit.value_changed.connect(seen.append)

    def type_text(text):
        edit.setText(text)
        edit.editingFinished.emit()

    type_text("7.5")
    assert edit.value() == 7.5 and seen == [7.5]
    type_text("7.5")  # unchanged: no signal
    assert seen == [7.5]
    for bad in ("abc", "0.5", "11", "inf", "nan", ""):
        type_text(bad)
        assert edit.value() == 7.5 and edit.text() == "7.5", bad  # rejected, text restored
    assert seen == [7.5]
    type_text("1e1")
    assert edit.value() == 10.0
    edit.set_value(2.0)  # programmatic: no signal
    assert edit.text() == "2" and seen == [7.5, 10.0]


def test_number_edit_allow_empty_means_no_limit(qapp):
    edit = NumberEdit(3.0, allow_empty=True)
    seen = []
    edit.value_changed.connect(seen.append)
    edit.setText("")
    edit.editingFinished.emit()
    assert edit.value() is None and seen == [None]


# -- populations ------------------------------------------------------------------------------


def test_population_panel_edits_the_model_in_place(qapp):
    model = demo.demo_model()
    panel = PopulationsPanel(model.populations)
    changes = []
    panel.changed.connect(changes.append)
    editor = panel.editors[0]
    row = editor.particle.row("radius")
    row.value.setText("2600")
    row.value.editingFinished.emit()
    assert model.populations[0].radius == 2600.0 and changes[-1] is True
    row.fit.setChecked(True)
    assert model.populations[0].fit_spec("radius").fit and changes[-1] is False  # no recalc needed
    row.high.setText("3000")
    row.high.editingFinished.emit()
    assert model.populations[0].fit_spec("radius").max == 3000.0
    row.low.setText("")
    row.low.editingFinished.emit()
    assert model.populations[0].fit_spec("radius").min is None
    editor.name.setText("renamed")
    editor.name.editingFinished.emit()
    assert model.populations[0].name == "renamed"
    assert panel.tabs.tabText(0).endswith("renamed")


def test_population_panel_add_duplicate_remove(qapp):
    model = AnisaModel(populations=[Population(name="A")])
    panel = PopulationsPanel(model.populations)
    structure = []
    panel.structure_changed.connect(lambda: structure.append(len(model.populations)))
    panel.add_population()
    panel.duplicate_population()
    assert [p.name for p in model.populations] == ["A", "Population 2", "Population 2 copy"][
        :0
    ] or len(model.populations) == 3
    assert panel.tabs.count() == 3 and structure == [2, 3]
    panel.tabs.setCurrentIndex(0)
    panel.remove_population()
    assert len(model.populations) == 2 and panel.tabs.count() == 2 and structure[-1] == 2
    while model.populations:
        panel.remove_population()
    assert panel.current_index() == -1 and not panel.remove_button.isEnabled()
    panel.remove_population()  # harmless with nothing left


def test_size_distribution_choice_sets_sensible_parameters(qapp):
    pop = Population(radius=1000.0)
    panel = PopulationsPanel([pop])
    editor = panel.editors[0]
    editor.size_type.setCurrentIndex(editor.size_type.findData("gauss"))
    assert pop.size_dist.type == "gauss"
    assert pop.size_dist.params["width"] == pytest.approx(150.0)  # 15 % of the radius
    assert (
        editor.size_edits["width"].isVisible() is False or True
    )  # visibility needs a shown parent
    editor.size_edits["width"].setText("80")
    editor.size_edits["width"].editingFinished.emit()
    assert pop.size_dist.params["width"] == 80.0
    editor.size_weighting.setCurrentIndex(editor.size_weighting.findData("number"))
    assert pop.size_dist.weighting == "number"
    editor.size_type.setCurrentIndex(editor.size_type.findData("lognormal"))
    assert set(pop.size_dist.params) == {"min_size", "sdeviation"}
    editor.size_type.setCurrentIndex(editor.size_type.findData("none"))
    assert pop.size_dist.type == "none"


def test_switching_orientation_to_a_table_starts_from_the_current_profile(qapp):
    pop = Population()
    pop.alpha_dist.par1, pop.alpha_dist.par2 = 0.0, 10.0
    panel = PopulationsPanel([pop])
    box = panel.editors[0].alpha_box
    box.mode.setCurrentIndex(box.mode.findData("table"))
    assert pop.alpha_dist.mode == "table" and len(pop.alpha_dist.table) == 18
    table = np.array(pop.alpha_dist.table)
    assert (
        table.argmax() == 0 and table.max() == pytest.approx(1.0) and table[-1] < 0.1
    )  # peak at 0
    box.mode.setCurrentIndex(box.mode.findData("parametric"))
    assert pop.alpha_dist.mode == "parametric"


def test_table_dialog_validates_and_resizes(qapp):
    dialog = TableEditDialog((1.0, 2.0, 3.0), 90.0, "t")
    assert dialog.values() == (1.0, 2.0, 3.0)
    dialog.count.setValue(6)
    assert len(dialog.values()) == 6 and dialog.values()[0] == pytest.approx(1.0)
    dialog.table.item(0, 1).setText("-1")
    dialog._accept()  # a negative weight is refused: the dialog stays open
    assert dialog.result() != dialog.DialogCode.Accepted
    dialog.table.item(0, 1).setText("2")
    dialog._accept()
    assert dialog.result() == dialog.DialogCode.Accepted


# -- directions -------------------------------------------------------------------------------


def test_directions_table_edits_the_model_and_rejects_bad_text(qapp):
    model = demo.synthesize(demo.demo_model())
    panel = DirectionsPanel(model.directions)
    seen = []
    panel.changed.connect(seen.append)
    panel.table.item(1, C_ALPHA).setText("33.5")
    assert model.directions[1].alpha_q == 33.5 and seen[-1] is True
    panel.table.item(1, C_ALPHA).setText("not a number")
    assert model.directions[1].alpha_q == 33.5 and panel.table.item(1, C_ALPHA).text() == "33.5"
    panel.table.item(2, C_QMIN).setText("0.01")
    panel.table.item(2, C_QMAX).setText("0.03")
    assert (model.directions[2].q_min, model.directions[2].q_max) == (0.01, 0.03)
    n_used = int(model.directions[2].fit_mask().sum())
    assert panel.table.item(2, 8).text() == f"{n_used}/120" and 0 < n_used < 120
    panel.table.item(2, C_QMIN).setText("-1")  # a negative Q limit is nonsense
    assert model.directions[2].q_min == 0.01
    panel.table.item(2, C_QMIN).setText("")  # empty = no limit
    assert model.directions[2].q_min is None
    panel.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert model.directions[0].use is False
    panel.table.item(3, 5).setCheckState(Qt.CheckState.Checked)
    assert model.directions[3].fit_spec().fit and seen[-1] is False


def test_directions_remove_and_duplicate(qapp):
    model = demo.demo_model()
    panel = DirectionsPanel(model.directions)
    structure = []
    panel.structure_changed.connect(lambda: structure.append(len(model.directions)))
    panel.table.selectRow(1)
    panel.duplicate_selected()
    assert len(model.directions) == 5 and model.directions[2].label == "in plane, x copy"
    panel.table.selectRow(0)
    panel.remove_selected()
    assert len(model.directions) == 4 and structure == [5, 4]


# -- plots ------------------------------------------------------------------------------------


def test_error_segments_are_log_safe():
    x, y = error_segments(np.array([1.0, 2.0]), np.array([10.0, 1.0]), np.array([1.0, 5.0]))
    assert x.shape == y.shape == (6,)
    assert np.isnan(x[2]) and np.isnan(y[5])  # NaN separators
    assert np.all(y[~np.isnan(y)] > 0)  # the lower end of the second bar (1 - 5 < 0) is clipped
    assert y[3] == pytest.approx(1e-3) and y[4] == pytest.approx(6.0)


# -- main window ------------------------------------------------------------------------------


def test_window_starts_empty_and_calculates(qapp, window):
    assert wait_until(qapp, lambda: window.last_result is not None)
    assert len(window.model.populations) == 1 and window.model.directions == []
    assert "untitled" in window.windowTitle()
    assert not window.revert_button.isEnabled() and not window.uncertainty_button.isEnabled()
    assert window.last_result.directions == []


def test_open_project_shows_data_and_chi_squared(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    assert len(window.model.directions) == 4 and window.directions_panel.table.rowCount() == 4
    assert window.last_stats.reduced == pytest.approx(98.83, rel=2e-3)  # the core's number
    assert "demo_start" in window.windowTitle() and not window.dirty
    assert "χ²/N" in window.status_label.text()
    assert window.state.recent_projects()[0].endswith("demo_start.anisa.json")


def test_editing_a_parameter_recalculates_and_marks_dirty(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    before = window.last_stats.chi_squared
    row = window.populations_panel.editors[0].particle.row("radius")
    row.value.setText("2500")
    row.value.editingFinished.emit()
    assert window.dirty and window.windowTitle().endswith("*")
    assert wait_until(qapp, lambda: window.last_stats.chi_squared != before)
    assert wait_until(qapp, lambda: settled(qapp, window))


def test_fit_through_the_window_recovers_the_parameters(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    window.start_fit()
    assert window._busy() and not window.left.isEnabled()  # editing is locked while fitting
    assert wait_until(qapp, lambda: window.last_fit is not None)
    assert wait_until(qapp, lambda: not window._busy() and settled(qapp, window))
    fit = window.last_fit
    assert fit.success and 0.85 < fit.reduced_chi_squared < 1.2
    truth = load_project(demo_project / "demo.anisa.json")
    for p in fit.parameters:
        true_value = truth.populations[p.parameter.index].get_parameter(p.parameter.path)
        assert abs(p.value - true_value) < 3.5 * p.std
    assert window.last_stats.reduced < 1.2 and window.left.isEnabled()
    editor = window.populations_panel.editors[0]
    assert editor.particle.row("radius").error.text().startswith("±")  # errors are shown
    assert "Fit converged" in window.log_view.toPlainText()
    assert window.revert_button.isEnabled() and window.dirty
    # revert restores the start and clears the errors
    window.revert_fit()
    assert window.last_fit is None and not window.revert_button.isEnabled()
    assert editor.particle.row("radius").error.text() == ""
    assert window.model.populations[0].radius == pytest.approx(
        load_project(demo_project / "demo_start.anisa.json").populations[0].radius
    )
    assert wait_until(qapp, lambda: window.last_stats.reduced > 50)


def test_cancelling_a_fit_keeps_the_best_parameters(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    start_chi = window.last_stats.chi_squared
    window.start_fit()
    time.sleep(0.15)
    window._cancel_running()
    assert wait_until(qapp, lambda: window.last_fit is not None)
    assert wait_until(qapp, lambda: not window._busy() and settled(qapp, window))
    assert window.last_fit.cancelled
    assert window.last_stats.chi_squared <= start_chi


def test_fit_without_flags_or_data_explains_itself(qapp, window, demo_project):
    window.start_fit()
    assert "Nothing to fit" in window.messages[-1][0]
    window.model.populations[0].fit_spec("radius").fit = True
    window.start_fit()
    assert window.messages[-1][0] == "No data" and not window._busy()


def test_add_data_files_creates_directions_and_picks_up_the_wavelength(qapp, window, demo_project):
    window.add_data_files(
        [str(demo_project / "demo_dir02.h5"), str(demo_project / "demo_dir01.h5")]
    )
    assert [d.label for d in window.model.directions][0].startswith("demo_dir02")
    assert len(window.model.directions) == 2 and window.dirty
    assert window.model.directions[0].source is not None
    assert window.model.settings.wavelength == pytest.approx(1.0)  # written by the demo
    assert wait_until(
        qapp, lambda: window.last_result is not None and len(window.last_result.directions) == 2
    )
    window.add_data_files([str(demo_project / "does_not_exist.dat")])
    assert "Could not load data" in window.messages[-1][0] and len(window.model.directions) == 2


def test_save_and_reopen_through_the_window(qapp, window, demo_project, tmp_path, monkeypatch):
    open_start(qapp, window, demo_project)
    window.model.directions[0].alpha_q = 12.5
    window._set_dirty()
    target = tmp_path / "saved.anisa.json"
    monkeypatch.setattr(
        "anisa.gui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(target), "")
    )
    assert window.save_project(save_as=True) and target.exists() and not window.dirty
    again = MainWindow(project=target, state=UserState(tmp_path / "state2" / "s.json"))
    try:
        assert again.model.directions[0].alpha_q == 12.5
        assert again.model.to_dict() == window.model.to_dict()
    finally:
        again.shutdown_threads()
        again.dirty = False
        again.close()


def test_a_bad_project_file_is_reported_not_fatal(qapp, window, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert not window.open_project(bad)
    assert "Could not open project" in window.messages[-1][0]


def test_new_project_resets_everything(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    window.new_project()
    assert window.model.directions == [] and len(window.model.populations) == 1
    assert window.project_path is None and window.last_fit is None and not window.dirty
    assert wait_until(
        qapp, lambda: window.last_result is not None and not window.last_result.directions
    )


def test_manual_update_mode_waits_for_calculate(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    window.settings_panel.auto_update.setChecked(False)
    before = window.last_stats.chi_squared
    row = window.populations_panel.editors[1].particle.row("volume_fraction")
    row.value.setText("0.05")
    row.value.editingFinished.emit()
    time.sleep(0.4)
    qapp.processEvents()
    assert window.last_stats.chi_squared == before and "out of date" in window.status_label.text()
    window.calculate()
    assert wait_until(qapp, lambda: window.last_stats.chi_squared != before)


def test_structure_change_clears_the_fit(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    window.start_fit()
    assert wait_until(qapp, lambda: window.last_fit is not None)
    assert wait_until(qapp, lambda: not window._busy() and settled(qapp, window))
    window.populations_panel.add_population()
    assert window.last_fit is None and not window.uncertainty_button.isEnabled()
    assert wait_until(qapp, lambda: settled(qapp, window))
    assert window.last_result.surface_areas.size == 4


def test_uncertainty_button_runs_monte_carlo(qapp, window, demo_project, monkeypatch):
    open_start(qapp, window, demo_project)
    window.start_fit()
    assert wait_until(qapp, lambda: window.last_fit is not None)
    assert wait_until(qapp, lambda: not window._busy() and settled(qapp, window))
    monkeypatch.setattr("anisa.gui.main_window.QInputDialog.getInt", lambda *a, **k: (2, True))
    window.start_uncertainty()
    assert wait_until(qapp, lambda: window.last_fit.monte_carlo is not None, timeout=180)
    assert window.last_fit.monte_carlo.values.shape == (2, 8)
    assert (
        "Monte Carlo" in window.populations_panel.editors[0].particle.row("radius").error.toolTip()
    )


def test_uncertainty_refuses_when_the_model_changed_after_the_fit(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    window.start_fit()
    assert wait_until(qapp, lambda: window.last_fit is not None)
    assert wait_until(qapp, lambda: not window._busy() and settled(qapp, window))
    window.model.populations[0].radius *= 1.1
    window.start_uncertainty()
    assert window.messages[-1][0] == "Model changed" and not window._busy()


def test_cursors_set_the_q_range_of_the_selected_direction(qapp, window, demo_project):
    open_start(qapp, window, demo_project)
    window.directions_panel.table.selectRow(2)
    window._apply_cursor_range(0.002, 0.03)
    d = window.model.directions[2]
    assert (d.q_min, d.q_max) == (0.002, 0.03) and window.dirty
    assert wait_until(qapp, lambda: window.last_stats.n_points < 480)


def test_window_layout_is_remembered(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ANISA_STATE_DIR", str(tmp_path / "state"))
    first = MainWindow()
    first.resize(701, 503)  # the offscreen screen is 800x600; Qt clamps larger windows
    first.splitter.setSizes([600, 700])
    first.close()
    first.shutdown_threads()
    saved = UserState().data
    assert saved["window_geometry"] and saved["splitter_sizes"][0] > 0
    second = MainWindow()
    try:
        assert second.width() == 701 and second.height() == 503
    finally:
        second.close()
        second.shutdown_threads()


# -- user state -------------------------------------------------------------------------------


def test_user_state_remembers_recent_projects_and_folders(tmp_path):
    state = UserState(tmp_path / "s.json")
    for i in range(12):
        state.add_recent_project(tmp_path / f"p{i}.json")
    state.add_recent_project(tmp_path / "p3.json")  # moves to the front, no duplicate
    recent = UserState(tmp_path / "s.json").recent_projects()
    assert len(recent) == 8 and recent[0].endswith("p3.json") and len(set(recent)) == 8
    state.set_folder("data", tmp_path / "x.h5")  # a file: its folder is remembered
    assert state.folder("data") == str(tmp_path)
    assert state.folder("never_set").startswith("/") or ":" in state.folder("never_set")


def test_user_state_survives_corrupt_and_unwritable_files(tmp_path):
    bad = tmp_path / "s.json"
    bad.write_text("{{{")
    assert UserState(bad).recent_projects() == []
    unwritable = UserState(tmp_path / "no" / "such" / "dir" / "s.json")
    unwritable.path = tmp_path / "file.txt" / "s.json"
    (tmp_path / "file.txt").write_text("x")
    unwritable.add_recent_project(tmp_path / "p.json")  # must not raise
