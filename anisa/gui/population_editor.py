"""
Editor for one population (Igor's tab on the AnisoSAS panel), and the tabbed list of them.

Sections: particle (radius, volume fraction, beta, contrast), size distribution, orientation
(p(alpha) and b(omega), parametric or table) and interference, plus read-only derived values.
Everything edits the :class:`~anisa.core.model.Population` in place and emits ``changed``.
"""

from __future__ import annotations

import copy

import numpy as np

from anisa.core import derived
from anisa.core.model import Population
from anisa.core.orientation import AngularDistribution
from anisa.core.sizedist import DISTRIBUTION_TYPES, PARAMETER_DEFAULTS
from anisa.gui._qt import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    Signal,
)
from anisa.gui.widgets import NumberEdit, ParameterGrid, ParamSpec

__all__ = ["PopulationEditor", "PopulationsPanel", "TableEditDialog"]

SIZE_TYPE_LABELS = {
    "none": "None (monodisperse)",
    "gauss": "Gauss",
    "lognormal": "Log-normal",
    "schulz_zimm": "Schulz-Zimm",
    "lsw": "LSW",
    "ardell": "Ardell",
}
SIZE_PARAM_LABELS = {
    "width": "Width σ (Å)",
    "min_size": "Min size (Å)",
    "sdeviation": "σ (log)",
    "parameter": "m  (2 < m < 3)",
}
#: Fraction of the radius used as the initial width when a width-type distribution is chosen.
DEFAULT_RELATIVE_WIDTH = 0.15


class TableEditDialog(QDialog):
    """Edit the bin values of an orientation table."""

    def __init__(self, values, span_deg: float, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.span_deg = span_deg
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Relative weight of each angular bin. Values are interpolated linearly between\n"
                "bin centres and normalised automatically; they must be ≥ 0, not all zero."
            )
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Number of bins:"))
        self.count = QSpinBox()
        self.count.setRange(2, 360)
        self.count.setValue(len(values))
        self.count.valueChanged.connect(self._resize)
        row.addWidget(self.count)
        row.addStretch(1)
        layout.addLayout(row)
        self.table = QTableWidget(len(values), 2)
        self.table.setHorizontalHeaderLabels(["Angular range (°)", "Weight"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._fill(list(values))
        self.resize(360, 460)

    def _fill(self, values: list[float]) -> None:
        n = len(values)
        self.table.setRowCount(n)
        for i, value in enumerate(values):
            lo, hi = i * self.span_deg / n, (i + 1) * self.span_deg / n
            label = QTableWidgetItem(f"{lo:.4g} – {hi:.4g}")
            label.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(i, 0, label)
            self.table.setItem(i, 1, QTableWidgetItem(f"{value:.6g}"))

    def _current(self) -> list[float]:
        out = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 1)
            try:
                out.append(float(item.text()))
            except (ValueError, AttributeError):
                out.append(float("nan"))
        return out

    def _resize(self, n: int) -> None:
        old = [v if np.isfinite(v) else 1.0 for v in self._current()]
        grid = (np.arange(n) + 0.5) / n
        source = (np.arange(len(old)) + 0.5) / len(old)
        self._fill([float(v) for v in np.interp(grid, source, old)])

    def _accept(self) -> None:
        values = self._current()
        if not all(np.isfinite(v) and v >= 0 for v in values) or not any(v > 0 for v in values):
            self.table.setToolTip("Every weight must be a number ≥ 0 and at least one > 0")
            self.table.setStyleSheet("QTableWidget { border: 2px solid #c0392b; }")
            return
        self.accept()

    def values(self) -> tuple[float, ...]:
        return tuple(self._current())


class _AngularBox(QGroupBox):
    """p(alpha) or b(omega): parametric (peak, sharpness, floor) or table."""

    changed = Signal(bool)

    ROLES = {
        "alpha": ("Polar angle  p(α)", 90.0, 18, "Peak α (°)"),
        "omega": ("Azimuth  b(ω)", 360.0, 36, "Peak ω (°)"),
    }

    def __init__(self, pop: Population, role: str, parent: QWidget | None = None):
        title, self.span, self.default_bins, peak = self.ROLES[role]
        super().__init__(title, parent)
        self.pop, self.role = pop, role
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("Parametric", "parametric")
        self.mode.addItem("Table", "table")
        top.addWidget(QLabel("Input:"))
        top.addWidget(self.mode)
        top.addStretch(1)
        layout.addLayout(top)
        specs = [
            ParamSpec(
                f"{role}_par1",
                peak,
                "Position of the peak of the distribution",
            ),
            ParamSpec(
                f"{role}_par2",
                "Sharpness n",
                "0 = isotropic; p = floor + |cos(angle − peak)|^n",
                minimum=0.0,
            ),
            ParamSpec(
                f"{role}_par3",
                "Isotropic floor",
                "Constant added to the peak: the share of randomly oriented particles",
                minimum=0.0,
            ),
        ]
        self.grid = ParameterGrid(pop, specs)
        self.grid.changed.connect(self.changed)
        layout.addWidget(self.grid)
        self.table_row = QWidget()
        row = QHBoxLayout(self.table_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.table_button = QPushButton("Edit table…")
        self.table_label = QLabel("")
        row.addWidget(self.table_button)
        row.addWidget(self.table_label, 1)
        layout.addWidget(self.table_row)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.table_button.clicked.connect(self.edit_table)
        self.refresh()

    @property
    def dist(self) -> AngularDistribution:
        return self.pop.alpha_dist if self.role == "alpha" else self.pop.omega_dist

    def refresh(self) -> None:
        self.grid.set_owner(self.pop)
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0 if self.dist.mode == "parametric" else 1)
        self.mode.blockSignals(False)
        self._update_visibility()

    def _update_visibility(self) -> None:
        parametric = self.dist.mode == "parametric"
        self.grid.setVisible(parametric)
        self.table_row.setVisible(not parametric)
        self.table_label.setText(f"{len(self.dist.table)} bins")

    def _profile_table(self) -> tuple[float, ...]:
        """The current parametric profile sampled in bins, as the starting point of a table."""
        n = self.default_bins
        centres = np.radians((np.arange(n) + 0.5) * self.span / n)
        parametric = AngularDistribution(
            "parametric", self.dist.par1, self.dist.par2, self.dist.par3
        )
        values = (
            parametric.alpha_values(centres)
            if self.role == "alpha"
            else parametric.omega_values(centres)
        )
        return tuple(float(v) for v in values / np.max(values))

    def _mode_changed(self) -> None:
        mode = self.mode.currentData()
        self.dist.mode = mode
        if mode == "table" and len(self.dist.table) < 2:
            self.dist.table = self._profile_table()
        self._update_visibility()
        self.changed.emit(True)

    def edit_table(self) -> None:
        dialog = TableEditDialog(self.dist.table, self.span, f"Edit table: {self.title()}", self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.dist.table = dialog.values()
            self._update_visibility()
            self.changed.emit(True)


class PopulationEditor(QWidget):
    """All settings of one population. ``changed(bool)``: True if the calculation changes."""

    changed = Signal(bool)

    def __init__(self, pop: Population, parent: QWidget | None = None):
        super().__init__(parent)
        self.pop = pop
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name = QLineEdit(pop.name)
        self.name.setPlaceholderText("optional label")
        self.name.editingFinished.connect(self._name_edited)
        name_row.addWidget(self.name, 1)
        layout.addLayout(name_row)

        # -- particle ---------------------------------------------------------------------
        box = QGroupBox("Particle  (spheroid, semi-axes R, R, β·R)")
        box_layout = QVBoxLayout(box)
        self.particle = ParameterGrid(
            pop,
            [
                ParamSpec("radius", "Radius R (Å)", "Equatorial radius", minimum=1.0),
                ParamSpec(
                    "volume_fraction",
                    "Volume fraction",
                    "Volume fraction of the sample",
                    minimum=0.0,
                ),
                ParamSpec(
                    "beta",
                    "Aspect ratio β",
                    "β < 1: oblate (disk / crack).  β > 1: prolate (cigar).  β = 1: sphere",
                    minimum=0.01,
                    maximum=100.0,
                ),
            ],
        )
        self.particle.changed.connect(self._edited)
        box_layout.addWidget(self.particle)
        form = QFormLayout()
        self.delta_rho = NumberEdit(pop.delta_rho, minimum=0.0)
        self.delta_rho.setToolTip("Scattering contrast Δρ in cm⁻² (not squared)")
        self.delta_rho.value_changed.connect(self._delta_rho_edited)
        form.addRow("Contrast Δρ (cm⁻²):", self.delta_rho)
        box_layout.addLayout(form)
        layout.addWidget(box)

        # -- size distribution ------------------------------------------------------------
        size_box = QGroupBox("Size distribution")
        size_layout = QFormLayout(size_box)
        self.size_type = QComboBox()
        for key in DISTRIBUTION_TYPES:
            self.size_type.addItem(SIZE_TYPE_LABELS[key], key)
        self.size_type.currentIndexChanged.connect(self._size_type_changed)
        size_layout.addRow("Type:", self.size_type)
        self.size_edits: dict[str, NumberEdit] = {}
        self.size_labels: dict[str, QLabel] = {}
        for name, text in SIZE_PARAM_LABELS.items():
            edit = NumberEdit(0.0, minimum=0.0)
            edit.value_changed.connect(lambda v, n=name: self._size_param_edited(n, v))
            label = QLabel(text)
            self.size_edits[name], self.size_labels[name] = edit, label
            size_layout.addRow(label, edit)
        self.size_weighting = QComboBox()
        self.size_weighting.addItem("Volume distribution", "volume")
        self.size_weighting.addItem("Number distribution", "number")
        self.size_weighting.currentIndexChanged.connect(self._size_weighting_changed)
        self.size_weighting_label = QLabel("Describes:")
        size_layout.addRow(self.size_weighting_label, self.size_weighting)
        self.size_bins = QSpinBox()
        self.size_bins.setRange(8, 2000)
        self.size_bins.setToolTip(
            "Minimum number of radius cells in the integration (more are used at high Q)"
        )
        self.size_bins.valueChanged.connect(self._size_bins_changed)
        self.size_bins_label = QLabel("Radius cells:")
        size_layout.addRow(self.size_bins_label, self.size_bins)
        layout.addWidget(size_box)

        # -- orientation ------------------------------------------------------------------
        self.alpha_box = _AngularBox(pop, "alpha")
        self.omega_box = _AngularBox(pop, "omega")
        for angular in (self.alpha_box, self.omega_box):
            angular.changed.connect(self._edited)
            layout.addWidget(angular)

        # -- interference -----------------------------------------------------------------
        interference = QGroupBox("Interference")
        interference_layout = QVBoxLayout(interference)
        self.use_interference = QCheckBox("Apply 1 / (1 + pack · 3 j₁(Qη)/(Qη))")
        self.use_interference.toggled.connect(self._interference_toggled)
        interference_layout.addWidget(self.use_interference)
        self.interference_grid = ParameterGrid(
            pop,
            [
                ParamSpec("eta", "Distance η (Å)", "Correlation distance", minimum=1.0),
                ParamSpec("pack", "Packing", "Interference strength", minimum=0.0),
            ],
        )
        self.interference_grid.changed.connect(self._edited)
        interference_layout.addWidget(self.interference_grid)
        layout.addWidget(interference)

        # -- derived values ---------------------------------------------------------------
        self.derived = QLabel("")
        self.derived.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.derived)
        layout.addStretch(1)
        self.refresh()

    # -- model -> widgets ------------------------------------------------------------------

    def refresh(self) -> None:
        """Show the current model values (after a fit, revert or load). Emits nothing."""
        pop = self.pop
        self.name.setText(pop.name)
        self.particle.set_owner(pop)
        self.delta_rho.set_value(pop.delta_rho)
        sd = pop.size_dist
        self.size_type.blockSignals(True)
        self.size_type.setCurrentIndex(self.size_type.findData(sd.type))
        self.size_type.blockSignals(False)
        params = sd.full_params()
        for name, edit in self.size_edits.items():
            if name in params:
                edit.set_value(params[name])
        self.size_weighting.blockSignals(True)
        self.size_weighting.setCurrentIndex(self.size_weighting.findData(sd.weighting))
        self.size_weighting.blockSignals(False)
        self.size_bins.blockSignals(True)
        self.size_bins.setValue(sd.n_bins)
        self.size_bins.blockSignals(False)
        self.alpha_box.refresh()
        self.omega_box.refresh()
        self.use_interference.blockSignals(True)
        self.use_interference.setChecked(pop.use_interference)
        self.use_interference.blockSignals(False)
        self.interference_grid.set_owner(pop)
        self._update_visibility()
        self.update_derived()

    def _update_visibility(self) -> None:
        sd = self.pop.size_dist
        needed = set(PARAMETER_DEFAULTS[sd.type])
        for name in self.size_edits:
            self.size_edits[name].setVisible(name in needed)
            self.size_labels[name].setVisible(name in needed)
        polydisperse = sd.type != "none"
        for widget in (
            self.size_weighting,
            self.size_weighting_label,
            self.size_bins,
            self.size_bins_label,
        ):
            widget.setVisible(polydisperse)
        self.interference_grid.setEnabled(self.pop.use_interference)

    def update_derived(self) -> None:
        try:
            text = (
                f"Mean particle volume: {derived.mean_volume(self.pop):.4g} Å³\n"
                f"Number density: {derived.number_density(self.pop):.4g} cm⁻³\n"
                f"Surface area: {derived.surface_area(self.pop):.4g} cm²/cm³"
            )
        except (ValueError, ArithmeticError):
            text = "(derived values unavailable for these parameters)"
        self.derived.setText(text)

    def set_errors(self, errors: dict[str, float | None], tooltip: str = "") -> None:
        self.particle.set_errors(errors, tooltip)
        self.alpha_box.grid.set_errors(errors, tooltip)
        self.omega_box.grid.set_errors(errors, tooltip)
        self.interference_grid.set_errors(errors, tooltip)

    # -- widgets -> model ------------------------------------------------------------------

    def _edited(self, recalc: bool) -> None:
        self.update_derived()
        self.changed.emit(recalc)

    def _name_edited(self) -> None:
        if self.name.text() != self.pop.name:
            self.pop.name = self.name.text()
            self.changed.emit(False)

    def _delta_rho_edited(self, value) -> None:
        if value is not None:
            self.pop.delta_rho = value
            self._edited(True)

    def _size_type_changed(self) -> None:
        sd = self.pop.size_dist
        new_type = self.size_type.currentData()
        defaults = dict(PARAMETER_DEFAULTS[new_type])
        if "width" in defaults and "width" not in sd.params:
            defaults["width"] = max(DEFAULT_RELATIVE_WIDTH * self.pop.radius, 1.0)
        sd.type = new_type
        sd.params = {k: float(sd.params.get(k, v)) for k, v in defaults.items()}
        for name, edit in self.size_edits.items():
            if name in sd.params:
                edit.set_value(sd.params[name])
        self._update_visibility()
        self._edited(True)

    def _size_param_edited(self, name: str, value) -> None:
        if value is not None:
            self.pop.size_dist.params[name] = value
            self._edited(True)

    def _size_weighting_changed(self) -> None:
        self.pop.size_dist.weighting = self.size_weighting.currentData()
        self._edited(True)

    def _size_bins_changed(self, value: int) -> None:
        self.pop.size_dist.n_bins = int(value)
        self._edited(True)

    def _interference_toggled(self, checked: bool) -> None:
        self.pop.use_interference = checked
        self._update_visibility()
        self._edited(True)


class PopulationsPanel(QWidget):
    """Tabs of :class:`PopulationEditor`, with buttons to add, duplicate and remove populations.

    ``changed(bool)``: a parameter was edited (True: the calculation changes).
    ``structure_changed``: a population was added or removed.
    ``current_changed(int)``: the selected population (-1 if there is none).
    """

    changed = Signal(bool)
    structure_changed = Signal()
    current_changed = Signal(int)

    def __init__(self, populations: list[Population], parent: QWidget | None = None):
        super().__init__(parent)
        self.populations = populations
        self.editors: list[PopulationEditor] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.duplicate_button = QPushButton("Duplicate")
        self.remove_button = QPushButton("Remove")
        for button in (self.add_button, self.duplicate_button, self.remove_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.current_changed)
        layout.addWidget(self.tabs, 1)
        self.add_button.clicked.connect(self.add_population)
        self.duplicate_button.clicked.connect(self.duplicate_population)
        self.remove_button.clicked.connect(self.remove_population)
        self.rebuild()

    def set_populations(self, populations: list[Population]) -> None:
        self.populations = populations
        self.rebuild()

    def rebuild(self) -> None:
        """Recreate every tab from ``self.populations``."""
        current = self.tabs.currentIndex()
        self.tabs.blockSignals(True)
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        self.editors = []
        for pop in self.populations:
            editor = PopulationEditor(pop)
            editor.changed.connect(self._editor_changed)
            self.editors.append(editor)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(editor)
            self.tabs.addTab(scroll, "")
        self.update_titles()
        self.tabs.blockSignals(False)
        if self.populations:
            self.tabs.setCurrentIndex(min(max(current, 0), len(self.populations) - 1))
        self._update_buttons()
        self.current_changed.emit(self.tabs.currentIndex())

    def update_titles(self) -> None:
        for i, pop in enumerate(self.populations):
            self.tabs.setTabText(i, f"{i + 1}. {pop.name or 'Population'}")

    def _update_buttons(self) -> None:
        has = bool(self.populations)
        self.duplicate_button.setEnabled(has)
        self.remove_button.setEnabled(has)

    def refresh(self) -> None:
        for editor in self.editors:
            editor.refresh()
        self.update_titles()

    def current_index(self) -> int:
        return self.tabs.currentIndex() if self.populations else -1

    def _editor_changed(self, recalc: bool) -> None:
        self.update_titles()
        self.changed.emit(recalc)

    def add_population(self) -> None:
        self.populations.append(Population(name=f"Population {len(self.populations) + 1}"))
        self.rebuild()
        self.tabs.setCurrentIndex(len(self.populations) - 1)
        self.structure_changed.emit()

    def duplicate_population(self) -> None:
        index = self.current_index()
        if index < 0:
            return
        clone = copy.deepcopy(self.populations[index])
        clone.name = f"{clone.name or 'Population'} copy"
        self.populations.insert(index + 1, clone)
        self.rebuild()
        self.tabs.setCurrentIndex(index + 1)
        self.structure_changed.emit()

    def remove_population(self) -> None:
        index = self.current_index()
        if index < 0:
            return
        del self.populations[index]
        self.rebuild()
        self.structure_changed.emit()
