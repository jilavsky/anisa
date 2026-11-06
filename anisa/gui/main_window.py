"""
The Anisa main window.

Left: directions (data), calculation settings, populations, and the Calculate / Fit buttons.
Right: I(Q) with model and residuals, the orientation distributions, and a log.

The window owns the :class:`~anisa.core.model.AnisaModel` and edits it in place. Calculations and
fits run on a snapshot in a worker thread (:mod:`anisa.gui.workers`); the fit result is copied
back when it finishes. While a fit runs the editing panels are disabled.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

import anisa
from anisa import demo, testsets
from anisa.core.engine import Engine
from anisa.core.fitting import (
    FitResult,
    FitStatistics,
    MonteCarloResult,
    active_parameters,
    apply_values,
    revert,
)
from anisa.core.model import AnisaModel, Population
from anisa.gui._qt import (
    QAction,
    QApplication,
    QByteArray,
    QDesktopServices,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QKeySequence,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    Qt,
    QTabWidget,
    QTimer,
    QUrl,
    QVBoxLayout,
    QWidget,
)
from anisa.gui.anisotropy_tab import AnisotropyTab, default_q_values
from anisa.gui.directions_panel import CurveChooserDialog, DirectionsPanel, SettingsPanel
from anisa.gui.plots import IQPlotWidget, OrientationPlotWidget
from anisa.gui.population_editor import PopulationsPanel
from anisa.gui.state import UserState
from anisa.gui.workers import (
    CalcOutput,
    TaskThread,
    anisotropy_job,
    calculation_job,
    fit_job,
    monte_carlo_job,
)
from anisa.io.data import discover_curves, find_wavelength, load_direction
from anisa.io.project import PROJECT_SUFFIX, load_project, save_project
from anisa.io.results import RESULTS_SUFFIX, load_results, save_results

log = logging.getLogger(__name__)

DOCS_URL = "https://github.com/jilavsky/anisa"
DATA_FILTER = "Scattering data (*.h5 *.hdf5 *.hdf *.nxs *.dat *.txt *.csv);;All files (*)"
PROJECT_FILTER = (
    f"Anisa projects and results (*{PROJECT_SUFFIX} *.json *{RESULTS_SUFFIX});;All files (*)"
)
RESULTS_FILTER = f"Anisa results (*{RESULTS_SUFFIX});;All files (*)"
DEFAULT_Q_RANGE = (1.0e-4, 0.1)

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """Anisa: modelling of small-angle scattering from anisotropic particle populations."""

    def __init__(self, project: str | Path | None = None, state: UserState | None = None):
        super().__init__()
        self.state = state or UserState()
        self.model = AnisaModel(populations=[Population(name="Population 1")])
        self.engine = Engine()
        self.project_path: Path | None = None
        self.dirty = False
        self.last_result = None
        self.last_stats: FitStatistics | None = None
        self.last_fit: FitResult | None = None
        self._calc_thread: TaskThread | None = None
        self._calc_pending = False
        self._out_of_date = False
        self._calc_seconds: float | None = None
        self._calc_started = 0.0
        self._fit_thread: TaskThread | None = None
        self._mc_thread: TaskThread | None = None
        self._aniso_thread: TaskThread | None = None
        self._aniso_pending = False
        self._aniso_stale = True
        self._threads: list[TaskThread] = []

        self.resize(1480, 900)
        self._build_ui()
        self._build_menus()
        self._calc_timer = QTimer(self)
        self._calc_timer.setSingleShot(True)
        self._calc_timer.setInterval(120)
        self._calc_timer.timeout.connect(self.calculate)
        self._aniso_timer = QTimer(self)
        self._aniso_timer.setSingleShot(True)
        self._aniso_timer.setInterval(150)
        self._aniso_timer.timeout.connect(self.calculate_anisotropy)
        self._enable_drops()
        self._restore_window_state()
        self._update_title()
        self._refresh_all()
        self.calculate()
        if project:
            self.open_project(project)

    # ==========================================================================================
    # construction
    # ==========================================================================================

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(500)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        self.left = left

        directions_box = QGroupBox("Directions (data)")
        directions_layout = QVBoxLayout(directions_box)
        self.directions_panel = DirectionsPanel(self.model.directions)
        directions_layout.addWidget(self.directions_panel)
        left_layout.addWidget(directions_box)

        self.settings_box = QGroupBox("Calculation settings")
        self.settings_box.setCheckable(True)
        self.settings_box.setChecked(False)
        settings_layout = QVBoxLayout(self.settings_box)
        self.settings_panel = SettingsPanel(self.model.settings)
        settings_layout.addWidget(self.settings_panel)
        self.settings_panel.setVisible(False)
        self.settings_box.toggled.connect(self._settings_toggled)
        self._settings_toggled(False)
        left_layout.addWidget(self.settings_box)

        populations_box = QGroupBox("Populations")
        populations_layout = QVBoxLayout(populations_box)
        self.populations_panel = PopulationsPanel(self.model.populations)
        populations_layout.addWidget(self.populations_panel)
        left_layout.addWidget(populations_box, 1)

        buttons = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate")
        self.fit_button = QPushButton("Fit")
        self.revert_button = QPushButton("Revert fit")
        self.uncertainty_button = QPushButton("Uncertainty…")
        self.calculate_button.setToolTip("Recalculate the model (Ctrl+R)")
        self.fit_button.setToolTip("Fit the parameters marked 'Fit?' to the data (Ctrl+F)")
        self.revert_button.setToolTip("Return to the parameter values before the last fit")
        self.uncertainty_button.setToolTip("Monte-Carlo estimate of the parameter errors")
        for button in (
            self.calculate_button,
            self.fit_button,
            self.revert_button,
            self.uncertainty_button,
        ):
            buttons.addWidget(button)
        left_layout.addLayout(buttons)

        progress_row = QHBoxLayout()
        self.progress_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.cancel_button = QPushButton("Cancel")
        progress_row.addWidget(self.progress_label, 1)
        progress_row.addWidget(self.progress)
        progress_row.addWidget(self.cancel_button)
        self.progress_widgets = (self.progress_label, self.progress, self.cancel_button)
        for widget in self.progress_widgets:
            widget.setVisible(False)
        left_layout.addLayout(progress_row)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        self.iq_plot = IQPlotWidget()
        self.orientation_plot = OrientationPlotWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = self.log_view.font()
        font.setFamily("Menlo, Consolas, monospace")
        font.setStyleHint(font.StyleHint.Monospace)
        self.log_view.setFont(font)
        self.anisotropy_tab = AnisotropyTab(self.model.scans)
        self.tabs.addTab(self.iq_plot, "I(Q)")
        self.tabs.addTab(self.anisotropy_tab, "Anisotropy")
        self.tabs.addTab(self.orientation_plot, "Orientation")
        self.tabs.addTab(self.log_view, "Log")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([540, 940])
        self.splitter = splitter

        self.status_label = QLabel("")
        self.calc_label = QLabel("")
        self.calc_bar = QProgressBar()  # indeterminate: shown while anything is (re)calculated
        self.calc_bar.setRange(0, 0)
        self.calc_bar.setTextVisible(False)
        self.calc_bar.setFixedSize(90, 12)
        self.calc_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.calc_label)
        self.statusBar().addPermanentWidget(self.calc_bar)
        self.statusBar().addPermanentWidget(self.status_label)

        # -- wiring ------------------------------------------------------------------------
        self.directions_panel.changed.connect(self._on_edit)
        self.directions_panel.structure_changed.connect(self._on_structure_changed)
        self.directions_panel.current_changed.connect(self._on_direction_selected)
        self.directions_panel.add_requested.connect(self.add_data_dialog)
        self.populations_panel.changed.connect(self._on_edit)
        self.populations_panel.structure_changed.connect(self._on_structure_changed)
        self.populations_panel.current_changed.connect(self._on_population_selected)
        self.settings_panel.changed.connect(lambda: self._on_edit(True))
        self.settings_panel.display_changed.connect(self._refresh_plots)
        self.settings_panel.auto_update.toggled.connect(self._auto_update_toggled)
        self.iq_plot.cursors_applied.connect(self._apply_cursor_range)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.anisotropy_tab.changed.connect(self._scans_edited)
        self.anisotropy_tab.selection_changed.connect(self._schedule_anisotropy)
        self.calculate_button.clicked.connect(self.calculate)
        self.fit_button.clicked.connect(self.start_fit)
        self.revert_button.clicked.connect(self.revert_fit)
        self.uncertainty_button.clicked.connect(self.start_uncertainty)
        self.cancel_button.clicked.connect(self._cancel_running)

    def _settings_toggled(self, expanded: bool) -> None:
        """Show or hide the settings; collapsed, the box shrinks to its title."""
        self.settings_panel.setVisible(expanded)
        title_height = self.settings_box.fontMetrics().height() + 14
        self.settings_box.setMaximumHeight(16777215 if expanded else title_height)

    def _action(self, menu, text, slot, shortcut=None, tip=None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self._action(file_menu, "&New project", self.new_project, "Ctrl+N")
        self._action(file_menu, "&Open project…", self.open_project_dialog, "Ctrl+O")
        self.recent_menu = file_menu.addMenu("Open &recent")
        self._action(file_menu, "&Save project", self.save_project, "Ctrl+S")
        self._action(
            file_menu, "Save project &as…", lambda: self.save_project(True), "Ctrl+Shift+S"
        )
        self._action(
            file_menu,
            "Save &results (HDF5)…",
            self.save_results_dialog,
            "Ctrl+E",
            tip="One file with the data, model curves, parameters and errors of all directions",
        )
        file_menu.addSeparator()
        self._action(file_menu, "Add &data files…", self.add_data_dialog, "Ctrl+D")
        self._action(
            file_menu,
            "Create &demo data…",
            self.create_demo_dialog,
            tip="Write synthetic multi-direction data with known parameters and open it",
        )
        test_menu = file_menu.addMenu("Create &test system")
        test_menu.setToolTip("Synthetic data with known parameters, from simple to complicated")
        for name in testsets.list_systems():
            title = testsets.make_system(name).title
            self._action(
                test_menu,
                f"{name}: {title}",
                lambda _checked=False, n=name: self.create_test_system_dialog(n),
                tip="Write the data, the truth and a perturbed start (see TRUTH.txt) and open it",
            )
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, "Ctrl+Q")
        model_menu = bar.addMenu("&Model")
        self._action(model_menu, "&Calculate", self.calculate, "Ctrl+R")
        self._action(model_menu, "&Fit", self.start_fit, "Ctrl+F")
        self._action(model_menu, "&Revert fit", self.revert_fit)
        self._action(model_menu, "&Uncertainty (Monte Carlo)…", self.start_uncertainty)
        help_menu = bar.addMenu("&Help")
        self._action(help_menu, "&Documentation (GitHub)", self._open_docs, "F1")
        self._action(help_menu, "&About Anisa", self._about)
        self._rebuild_recent_menu()

    def _enable_drops(self) -> None:
        try:
            from pyirena.gui.file_drop import enable_file_drop

            enable_file_drop(self.directions_panel.table, self.add_data_files)
            enable_file_drop(self.iq_plot, self.add_data_files)
        except Exception:  # drag and drop is a convenience only
            log.debug("drag and drop unavailable", exc_info=True)

    def _restore_window_state(self) -> None:
        """Restore the window geometry and splitter from the previous session (best effort)."""
        try:
            geometry = self.state.data.get("window_geometry")
            if geometry:
                self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
            sizes = self.state.data.get("splitter_sizes")
            if (
                isinstance(sizes, list)
                and len(sizes) == 2
                and all(isinstance(n, int) for n in sizes)
            ):
                self.splitter.setSizes(sizes)
        except Exception:  # remembering the layout is a convenience only
            log.debug("could not restore the window layout", exc_info=True)

    def _save_window_state(self) -> None:
        try:
            self.state.data["window_geometry"] = bytes(self.saveGeometry().toBase64()).decode(
                "ascii"
            )
            self.state.data["splitter_sizes"] = [int(n) for n in self.splitter.sizes()]
            self.state.save()
        except Exception:
            log.debug("could not save the window layout", exc_info=True)

    # ==========================================================================================
    # helpers
    # ==========================================================================================

    def log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {text}")

    def _update_title(self) -> None:
        name = self.project_path.name if self.project_path else "untitled"
        self.setWindowTitle(f"Anisa {anisa.__version__} — {name}{'*' if self.dirty else ''}")

    def _set_dirty(self, dirty: bool = True) -> None:
        self.dirty = dirty
        self._update_title()

    def _reap_threads(self) -> None:
        """Forget threads that have fully stopped (never from inside their own signal)."""
        self._threads = [t for t in self._threads if t.isRunning()]

    def _busy(self) -> bool:
        """A fit or Monte-Carlo job is in progress. Explicit state, cleared by the job's own
        ``finished`` handler: ``QThread.isRunning`` is still False between creating a thread and
        starting it, and still True while ``finished`` is being delivered."""
        return self._fit_thread is not None or self._mc_thread is not None

    def _selected_direction(self) -> int:
        return self.directions_panel.current_index()

    def _selected_population(self) -> int:
        return self.populations_panel.current_index()

    def _default_q(self) -> np.ndarray:
        low, high = DEFAULT_Q_RANGE
        qs = [
            np.asarray(d.q)[np.asarray(d.q) > 0] for d in self.model.directions if d.q is not None
        ]
        qs = [q for q in qs if q.size]
        if qs:
            low, high = min(q.min() for q in qs), max(q.max() for q in qs)
        return np.geomspace(low, high, 240)

    def _q_list(self) -> list[np.ndarray]:
        default = self._default_q()
        return [d.q if d.q is not None else default for d in self.model.directions]

    def _show_message(self, title: str, text: str, icon=QMessageBox.Icon.Warning) -> None:
        box = QMessageBox(icon, title, text, QMessageBox.StandardButton.Ok, self)
        box.exec()

    # ==========================================================================================
    # reacting to edits
    # ==========================================================================================

    def _on_edit(self, recalc: bool) -> None:
        self._set_dirty()
        if recalc:
            self._clear_errors()
            self._schedule_calculation()
            self._schedule_anisotropy()

    def _on_structure_changed(self) -> None:
        self._set_dirty()
        self.last_fit = None
        self._clear_errors()
        self._refresh_buttons()
        self._refresh_plots()
        self._schedule_calculation()
        self._schedule_anisotropy()

    def _on_direction_selected(self, index: int) -> None:
        if 0 <= index < len(self.model.directions):
            d = self.model.directions[index]
            if d.q_min and d.q_max:
                self.iq_plot.set_cursors(d.q_min, d.q_max)
            elif d.q is not None and len(d.q):
                q = np.asarray(d.q)[np.asarray(d.q) > 0]
                if q.size:
                    self.iq_plot.set_cursors(q.min() * 1.2, q.max() / 1.2)
        self._refresh_plots()

    def _on_population_selected(self, _index: int) -> None:
        self._refresh_plots()

    def _auto_update_toggled(self, checked: bool) -> None:
        if checked:
            self.calculate()

    def _apply_cursor_range(self, q_min: float, q_max: float) -> None:
        index = self._selected_direction()
        if not 0 <= index < len(self.model.directions):
            self.statusBar().showMessage("Select a direction in the table first", 4000)
            return
        d = self.model.directions[index]
        d.q_min, d.q_max = float(q_min), float(q_max)
        self.directions_panel.rebuild()
        self._on_edit(True)

    def _clear_errors(self) -> None:
        for editor in self.populations_panel.editors:
            editor.set_errors({})

    # ==========================================================================================
    # refreshing the display
    # ==========================================================================================

    def _refresh_all(self) -> None:
        self.directions_panel.set_directions(self.model.directions)
        self.populations_panel.set_populations(self.model.populations)
        self.settings_panel.set_settings(self.model.settings)
        self.anisotropy_tab.set_scans(self.model.scans)
        self._refresh_buttons()
        self._refresh_plots()

    def _refresh_buttons(self) -> None:
        busy = self._busy()
        self.fit_button.setEnabled(not busy)
        self.calculate_button.setEnabled(not busy)
        self.revert_button.setEnabled(not busy and self.last_fit is not None)
        self.uncertainty_button.setEnabled(not busy and self.last_fit is not None)
        self.left.setEnabled(not busy)
        for widget in self.progress_widgets:
            widget.setVisible(busy)

    def _refresh_plots(self) -> None:
        self.iq_plot.update_plot(
            self.model,
            self.last_result,
            selected_direction=self._selected_direction(),
            selected_population=self._selected_population(),
            show_populations=self.settings_panel.show_populations.isChecked(),
        )
        self.orientation_plot.update_plot(self.model, self._selected_population())

    def _update_status(self) -> None:
        stats = self.last_stats
        parts = []
        if stats is not None and stats.n_points:
            parts.append(f"χ²/N = {stats.reduced:.4g}  ({stats.n_points} points)")
        if self.last_result is not None and len(self.last_result.surface_areas):
            parts.append(f"surface area {self.last_result.total_surface_area:.4g} cm²/cm³")
        text = "   |   ".join(parts)
        self.status_label.setText(text)
        self.iq_plot.status.setText(text)
        if stats is not None and stats.n_points:
            lines = [
                f"{self.model.directions[i].label or f'Dir {i + 1}'}: χ²/N = {stats.direction_reduced(i):.4g}"
                for i, n in enumerate(stats.direction_points)
                if n
            ]
            self.status_label.setToolTip("\n".join(lines))

    # ==========================================================================================
    # calculation
    # ==========================================================================================

    def _calculating(self) -> bool:
        """True while a model or scan calculation runs or is waiting to start."""
        return (
            self._calc_thread is not None
            or self._aniso_thread is not None
            or self._calc_timer.isActive()
            or self._aniso_timer.isActive()
            or self._calc_pending
            or self._aniso_pending
        )

    def _update_calc_indicator(self) -> None:
        """Status-bar cue: a moving bar while calculating, then how long the last update took.

        Without it a fast model gives no sign that the plot is current, and a slow one (many
        directions, large grids) no sign that it is still working.
        """
        if self._calculating():
            self.calc_bar.setVisible(True)
            self.calc_label.setStyleSheet("")
            self.calc_label.setText("Calculating…")
        elif self._out_of_date:
            self.calc_bar.setVisible(False)
            self.calc_label.setStyleSheet("color: #c06000; font-weight: bold;")
            self.calc_label.setText("model out of date: press Calculate")
        else:
            self.calc_bar.setVisible(False)
            self.calc_label.setStyleSheet("")
            seconds = self._calc_seconds
            self.calc_label.setText("" if seconds is None else f"Up to date ({seconds:.2f} s)")

    def _schedule_calculation(self) -> None:
        if self.settings_panel.auto_update.isChecked():
            self._out_of_date = False
            self._calc_timer.start()
        else:
            self._out_of_date = True
        self._update_calc_indicator()

    def calculate(self) -> None:
        """Calculate the model along every direction (in a worker thread)."""
        self._calc_timer.stop()
        if self._calc_thread is not None and self._calc_thread.isRunning():
            self._calc_pending = True
            return
        self._reap_threads()
        self._out_of_date = False
        self._calc_started = time.perf_counter()
        snapshot = copy.deepcopy(self.model)
        thread = TaskThread(calculation_job(snapshot, self.engine, self._q_list()))
        thread.done.connect(self._calculation_done)
        thread.failed.connect(self._calculation_failed)
        thread.finished.connect(self._calculation_finished)
        self._calc_thread = thread
        self._threads.append(thread)
        thread.start()
        self._update_calc_indicator()

    def _calculation_done(self, output: CalcOutput) -> None:
        # a structure change while the worker ran makes the result stale: recalculate
        if len(output.result.directions) != len(
            self.model.directions
        ) or output.result.surface_areas.size != len(self.model.populations):
            self._calc_pending = True
            return
        self.last_result = output.result
        self.last_stats = output.statistics
        self._calc_seconds = time.perf_counter() - self._calc_started
        self._update_status()
        self._refresh_plots()
        self.directions_panel.refresh_row_values()

    def _calculation_failed(self, message: str) -> None:
        self.log(f"Calculation failed: {message}")
        self.status_label.setText(f"calculation failed: {message}")
        self._calc_seconds = None

    def _calculation_finished(self) -> None:
        # The thread object stays in self._threads: dropping the last reference inside its own
        # finished signal would destroy a QThread that is still shutting down. _reap_threads()
        # prunes it later.
        self._calc_thread = None
        if self._calc_pending:
            self._calc_pending = False
            self._schedule_calculation()
        self._update_calc_indicator()

    # ==========================================================================================
    # anisotropy scans
    # ==========================================================================================

    def _anisotropy_visible(self) -> bool:
        return self.tabs.currentWidget() is self.anisotropy_tab

    def _tab_changed(self, _index: int) -> None:
        if self._anisotropy_visible() and self._aniso_stale:
            self.calculate_anisotropy()

    def _scans_edited(self) -> None:
        self._set_dirty()
        self._schedule_anisotropy()

    def _schedule_anisotropy(self) -> None:
        """The scan result is out of date; recompute now if the tab is showing, else on demand."""
        self._aniso_stale = True
        if self._anisotropy_visible():
            self._aniso_timer.start()
        self._update_calc_indicator()

    def _ensure_default_scan(self) -> None:
        """Offer a first scan once there are data: three Q values across the measured range."""
        if not self.model.scans and any(d.q is not None for d in self.model.directions):
            was_dirty = self.dirty
            self.anisotropy_tab.add_scan(default_q_values(self.model))
            self._set_dirty(was_dirty)  # an automatic scan is not a user edit

    def calculate_anisotropy(self) -> None:
        """Calculate the selected scan (worker thread)."""
        self._aniso_timer.stop()
        scan = self.anisotropy_tab.current_scan()
        if scan is None:
            self._aniso_stale = False
            self.anisotropy_tab.update_result(None, self.model)
            return
        if self._aniso_thread is not None:
            self._aniso_pending = True
            return
        self._reap_threads()
        self._aniso_stale = False
        thread = TaskThread(
            anisotropy_job(copy.deepcopy(self.model), copy.deepcopy(scan), self.engine)
        )
        thread.done.connect(lambda result: self.anisotropy_tab.update_result(result, self.model))
        thread.failed.connect(self._anisotropy_failed)
        thread.finished.connect(self._anisotropy_finished)
        self._aniso_thread = thread
        self._threads.append(thread)
        thread.start()
        self._update_calc_indicator()

    def _anisotropy_failed(self, message: str) -> None:
        self.log(f"Anisotropy calculation failed: {message}")
        self.anisotropy_tab.message.setText(f"calculation failed: {message}")

    def _anisotropy_finished(self) -> None:
        self._aniso_thread = None
        if self._aniso_pending:
            self._aniso_pending = False
            self._schedule_anisotropy()
        self._update_calc_indicator()

    # ==========================================================================================
    # fit, revert, uncertainty
    # ==========================================================================================

    def _have_data(self) -> bool:
        return any(d.use and d.fit_mask().any() for d in self.model.directions)

    def start_fit(self) -> None:
        if self._busy():
            return
        if not active_parameters(self.model):
            self._show_message(
                "Nothing to fit",
                "No parameter is marked for fitting.\n\nTick 'Fit?' next to the parameters to "
                "vary (and set their limits) in the populations, or 'Fit bkg' for a background.",
                QMessageBox.Icon.Information,
            )
            return
        if not self._have_data():
            self._show_message(
                "No data",
                "There are no data to fit: add data files (Add data…), and make sure the "
                "directions are marked 'Use' with a Q range that contains points.",
                QMessageBox.Icon.Information,
            )
            return
        self._reap_threads()
        snapshot = copy.deepcopy(self.model)
        self._fit_snapshot = snapshot
        thread = TaskThread(fit_job(snapshot, self.engine))
        thread.progress.connect(self._fit_progress)
        thread.done.connect(self._fit_done)
        thread.failed.connect(self._fit_failed)
        thread.finished.connect(self._fit_finished)
        self._fit_thread = thread
        self._threads.append(thread)
        self.progress.setRange(0, 0)
        self.progress_label.setText("Fitting…")
        self.log("Fit started")
        self._refresh_buttons()
        thread.start()

    def _fit_progress(self, n_evaluations: int, reduced_chi: float) -> None:
        self.progress_label.setText(
            f"Fitting…  {n_evaluations} evaluations,  χ²/dof = {reduced_chi:.4g}"
        )

    def _fit_done(self, result: FitResult) -> None:
        parameters = [p.parameter for p in result.parameters]
        apply_values(self.model, parameters, result.values())
        self.last_fit = result
        self.log(result.report([d.label for d in self.model.directions]))
        self.populations_panel.refresh()
        self.directions_panel.refresh_row_values()
        self._show_errors(result)
        self._set_dirty()
        self.calculate()
        self._schedule_anisotropy()
        self.statusBar().showMessage(
            f"Fit {'cancelled' if result.cancelled else 'finished'}: χ²/dof = {result.reduced_chi_squared:.4g}",
            8000,
        )

    def _fit_failed(self, message: str) -> None:
        self.log(f"Fit failed: {message}")
        self._show_message("Fit failed", message)

    def _fit_finished(self) -> None:
        self._fit_thread = None
        self._refresh_buttons()

    def _mc_finished(self) -> None:
        self._mc_thread = None
        self._refresh_buttons()

    def _show_errors(self, result: FitResult, monte_carlo: MonteCarloResult | None = None) -> None:
        """Show the parameter errors next to the fitted values."""
        per_population: dict[int, dict[str, float]] = {}
        for k, p in enumerate(result.parameters):
            if p.parameter.owner != "population":
                continue
            std = monte_carlo.std[k] if monte_carlo is not None else p.std
            per_population.setdefault(p.parameter.index, {})[p.parameter.path] = float(std)
        source = "Monte Carlo" if monte_carlo is not None else "from the fit covariance"
        for index, editor in enumerate(self.populations_panel.editors):
            editor.set_errors(per_population.get(index, {}), source)

    def revert_fit(self) -> None:
        if self.last_fit is None or self._busy():
            return
        revert(self.model, self.last_fit)
        self.last_fit = None
        self._clear_errors()
        self.populations_panel.refresh()
        self.directions_panel.refresh_row_values()
        self._set_dirty()
        self._refresh_buttons()
        self.log("Reverted to the parameters before the fit")
        self.calculate()
        self._schedule_anisotropy()

    def start_uncertainty(self) -> None:
        if self.last_fit is None or self._busy():
            return
        if not self._fit_is_current():
            self._show_message(
                "Model changed",
                "The parameters changed since the fit. Run the fit again (or revert), then ask "
                "for the uncertainty.",
            )
            return
        runs, ok = QInputDialog.getInt(
            self,
            "Monte-Carlo uncertainty",
            "Number of refits of re-drawn data (each takes about as long as a fit):",
            10,
            2,
            1000,
        )
        if not ok:
            return
        self._reap_threads()
        thread = TaskThread(monte_carlo_job(self.model, self.last_fit, runs))
        thread.progress.connect(
            lambda done, total: self.progress_label.setText(
                f"Monte Carlo…  {done} / {int(total)} refits"
            )
        )
        thread.done.connect(self._uncertainty_done)
        thread.failed.connect(lambda message: self._show_message("Uncertainty failed", message))
        thread.finished.connect(self._mc_finished)
        self._mc_thread = thread
        self._threads.append(thread)
        self.progress.setRange(0, 0)
        self.progress_label.setText("Monte Carlo…")
        self.log(f"Monte-Carlo uncertainty started ({runs} runs)")
        self._refresh_buttons()
        thread.start()

    def _parameter_value(self, parameter) -> float:
        if parameter.owner == "population":
            return self.model.populations[parameter.index].get_parameter(parameter.path)
        return self.model.directions[parameter.index].background

    def _uncertainty_done(self, mc: MonteCarloResult) -> None:
        if self.last_fit is None:
            return
        self.last_fit.monte_carlo = mc
        self._show_errors(self.last_fit, mc)
        self.log(
            f"Monte Carlo finished: {mc.values.shape[0]} of {mc.n_requested} runs succeeded\n"
            + self.last_fit.report([d.label for d in self.model.directions])
        )

    def _cancel_running(self) -> None:
        for thread in (self._fit_thread, self._mc_thread):
            if thread is not None and thread.isRunning():
                thread.cancel()
        self.progress_label.setText("Cancelling…")

    # ==========================================================================================
    # data and projects
    # ==========================================================================================

    def add_data_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add data files", self.state.folder("data"), DATA_FILTER
        )
        if paths:
            self.add_data_files(paths)

    def add_data_files(self, paths: list[str]) -> None:
        """Load each file as a new direction (asks which curve if a file has several)."""
        if self._busy():
            return
        added = 0
        first_new = not any(d.q is not None for d in self.model.directions)
        for path in paths:
            try:
                internal = self._choose_curve(path)
                if internal is False:
                    continue
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    direction = load_direction(path, internal_path=internal)
                for w in caught:
                    self.log(f"{Path(path).name}: {w.message}")
            except Exception as error:
                self.log(f"Could not load {path}: {error}")
                self._show_message("Could not load data", f"{Path(path).name}:\n{error}")
                continue
            self.directions_panel.append_direction(direction)
            self.log(
                f"Loaded {Path(path).name}: {len(direction.q)} points, "
                f"Q {direction.q.min():.3g} – {direction.q.max():.3g} Å⁻¹. "
                "Set the angles α_Q, ω_Q of this direction in the table."
            )
            added += 1
        if added:
            self.state.set_folder("data", paths[0])
            if first_new:
                wavelength = find_wavelength(self.model.directions)
                if wavelength:
                    self.model.settings.wavelength = wavelength
                    self.settings_panel.refresh()
                self._reset_views()
            self._ensure_default_scan()
            self._on_structure_changed()

    def _choose_curve(self, path: str):
        """The ``internal_path`` to load: ``None`` for single-curve files, ``False`` if cancelled."""
        curves = discover_curves(path)
        if len(curves) <= 1:
            return None
        dialog = CurveChooserDialog(
            Path(path).name, [c.internal_path or c.display_name for c in curves], self
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return False
        return curves[dialog.selected()].internal_path

    def _reset_views(self) -> None:
        from pyirena.gui.sas_plot import reset_user_view

        reset_user_view(self.iq_plot.main, self.iq_plot.resid)

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        box = QMessageBox(
            QMessageBox.Icon.Question,
            "Unsaved changes",
            "The project has unsaved changes.",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            self,
        )
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            return self.save_project()
        return choice == QMessageBox.StandardButton.Discard

    def new_project(self) -> None:
        if self._busy() or not self._confirm_discard():
            return
        self.model = AnisaModel(populations=[Population(name="Population 1")])
        self.project_path = None
        self.last_fit = None
        self.last_result = None
        self.last_stats = None
        self.engine.clear_cache()
        self._set_dirty(False)
        self._refresh_all()
        self._clear_errors()
        self.calculate()
        self._schedule_anisotropy()

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", self.state.folder("project"), PROJECT_FILTER
        )
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> bool:
        if self._busy() or not self._confirm_discard():
            return False
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                from_results = str(path).lower().endswith((".h5", ".hdf5"))
                model = load_results(path) if from_results else load_project(path)
        except (OSError, ValueError) as error:
            self.log(f"Could not open {path}: {error}")
            self._show_message("Could not open project", str(error))
            return False
        self.model = model
        self.project_path = None if from_results else Path(path).resolve()
        opened = Path(path).resolve()
        self.last_fit = None
        self.last_result = None
        self.last_stats = None
        self.engine.clear_cache()
        if not from_results:  # a results file is not a project: Save project asks for a name
            self.state.add_recent_project(opened)
        self.state.set_folder("project", opened)
        self._rebuild_recent_menu()
        self._set_dirty(False)
        self._refresh_all()
        self._clear_errors()
        self._reset_views()
        for w in caught:
            self.log(f"{opened.name}: {w.message}")
        if caught:
            self._show_message(
                "Project opened with warnings", "\n".join(str(w.message) for w in caught)
            )
        self.log(
            f"Opened {opened}" + (" (results file; data taken from it)" if from_results else "")
        )
        self._ensure_default_scan()
        self.calculate()
        self._schedule_anisotropy()
        return True

    def save_project(self, save_as: bool = False) -> bool:
        path = self.project_path
        if save_as or path is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                "Save project",
                str(Path(self.state.folder("project")) / f"anisa{PROJECT_SUFFIX}"),
                PROJECT_FILTER,
            )
            if not chosen:
                return False
            path = Path(chosen)
        try:
            self.project_path = save_project(self.model, path)
        except OSError as error:
            self._show_message("Could not save project", str(error))
            return False
        self.state.add_recent_project(self.project_path)
        self.state.set_folder("project", self.project_path)
        self._rebuild_recent_menu()
        self._set_dirty(False)
        self.log(f"Saved {self.project_path}")
        return True

    def _fit_is_current(self) -> bool:
        """True if the parameters still have the values of the last fit (so its errors apply)."""
        if self.last_fit is None:
            return False
        current = [self._parameter_value(p.parameter) for p in self.last_fit.parameters]
        return bool(np.allclose(current, self.last_fit.values(), rtol=1e-9, atol=0))

    def save_results_dialog(self) -> bool:
        """Write data, model curves, parameters and errors of all directions to one HDF5 file."""
        if self._busy():
            return False
        if not self.model.directions:
            self._show_message(
                "Nothing to save", "Add data or directions first.", QMessageBox.Icon.Information
            )
            return False
        stem = self.project_path.name.removesuffix(PROJECT_SUFFIX) if self.project_path else "anisa"
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save results",
            str(Path(self.state.folder("results")) / f"{stem}{RESULTS_SUFFIX}"),
            RESULTS_FILTER,
        )
        if not chosen:
            return False
        try:
            path = save_results(
                self.model,
                chosen,
                fit=self.last_fit if self._fit_is_current() else None,
                engine=self.engine,
            )
        except (OSError, ValueError) as error:
            self.log(f"Could not save results: {error}")
            self._show_message("Could not save results", str(error))
            return False
        self.state.set_folder("results", path)
        self.log(f"Saved results {path}")
        self.statusBar().showMessage(f"Saved results to {path}", 6000)
        return True

    def create_demo_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Folder for the synthetic demo data", self.state.folder("demo")
        )
        if not folder:
            return
        try:
            project = demo.write_demo_project(folder)
        except (OSError, ValueError) as error:
            self._show_message("Could not write demo data", str(error))
            return
        self.state.set_folder("demo", folder)
        self.log(
            f"Wrote demo data to {folder}: demo.anisa.json is the truth, demo_start.anisa.json a "
            "perturbed start with fit flags set. Opening the start; press Fit."
        )
        self.open_project(project.with_name("demo_start.anisa.json"))

    def create_test_system_dialog(self, name: str) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            f"Folder for the test system '{name}' (a subfolder is created)",
            self.state.folder("demo"),
        )
        if not folder:
            return
        try:
            truth = testsets.write_test_system(name, folder)
        except (OSError, ValueError) as error:
            self._show_message("Could not write the test system", str(error))
            return
        self.state.set_folder("demo", folder)
        self.log(
            f"Wrote test system '{name}' to {truth.parent}: {truth.name} is the truth, "
            f"{name}_start.anisa.json a perturbed start with fit flags set, TRUTH.txt describes "
            "it. Opening the start; press Fit."
        )
        self.open_project(truth.with_name(f"{name}_start.anisa.json"))

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self.state.recent_projects() if Path(p).is_file()]
        self.recent_menu.setEnabled(bool(recent))
        for path in recent:
            action = QAction(path, self)
            action.triggered.connect(lambda _checked=False, p=path: self.open_project(p))
            self.recent_menu.addAction(action)

    # ==========================================================================================
    # help, closing
    # ==========================================================================================

    def _open_docs(self) -> None:
        QDesktopServices.openUrl(QUrl(DOCS_URL))

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About Anisa",
            f"<b>Anisa {anisa.__version__}</b><br>Small-angle scattering from anisotropic "
            "particle populations.<br>Successor of the Igor Pro AnisoSAS macros.<br><br>"
            f"<a href='{DOCS_URL}'>{DOCS_URL}</a>",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._busy():
            reply = QMessageBox.question(
                self, "Fit running", "A fit is still running. Cancel it and quit?"
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if not self._confirm_discard():
            event.ignore()
            return
        self.shutdown_threads()
        self._save_window_state()
        event.accept()

    def shutdown_threads(self) -> None:
        """Stop and wait for every background job (used on close and by the tests)."""
        self._calc_timer.stop()
        self._aniso_timer.stop()
        for thread in list(self._threads):
            thread.cancel()
        for thread in list(self._threads):
            thread.wait(10_000)
        with contextlib.suppress(RuntimeError):
            QApplication.processEvents()
