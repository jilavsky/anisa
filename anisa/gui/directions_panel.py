"""
Table of directions (Igor's Dir 1 ... Dir 6, here without a limit) and the calculation settings.

Each direction row: use, label, alpha_Q, omega_Q, background, fit background, Q range. Editing a
cell edits the :class:`~anisa.core.model.Direction` in place and emits ``changed``.
"""

from __future__ import annotations

import copy
import math

from anisa.core.model import CalcSettings, Direction
from anisa.gui._qt import (
    QAbstractItemView,
    QCheckBox,
    QColor,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Signal,
)
from anisa.gui.plots import direction_color
from anisa.gui.widgets import NumberEdit, format_number

__all__ = ["CurveChooserDialog", "DirectionsPanel", "SettingsPanel"]

COLUMNS = [
    "Use",
    "Label",
    "α_Q (°)",
    "ω_Q (°)",
    "Background",
    "Fit bkg",
    "Q min",
    "Q max",
    "Points",
]
C_USE, C_LABEL, C_ALPHA, C_OMEGA, C_BKG, C_FITBKG, C_QMIN, C_QMAX, C_POINTS = range(9)


class CurveChooserDialog(QDialog):
    """Pick one curve of a file that holds several."""

    def __init__(self, file_name: str, curves: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Choose a curve in {file_name}")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{file_name} contains {len(curves)} curves. Choose one:"))
        self.list = QListWidget()
        self.list.addItems(curves)
        self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self.list)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 300)

    def selected(self) -> int:
        return self.list.currentRow()


class DirectionsPanel(QWidget):
    """The directions table with Add / Remove / Duplicate.

    Signals:
        changed(bool): a value was edited (True: the calculation changes).
        structure_changed: a direction was added, removed or reordered.
        current_changed(int): the selected direction (-1 if none).
        add_requested: the user pressed "Add data…".
    """

    changed = Signal(bool)
    structure_changed = Signal()
    current_changed = Signal(int)
    add_requested = Signal()

    def __init__(self, directions: list[Direction], parent: QWidget | None = None):
        super().__init__(parent)
        self.directions = directions
        self._filling = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add data…")
        self.add_button.setToolTip(
            "Load data files, one direction each (you can also drag files here)"
        )
        self.remove_button = QPushButton("Remove")
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.setToolTip(
            "Copy the selected direction (e.g. to model another angle)"
        )
        for button in (self.add_button, self.remove_button, self.duplicate_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(C_LABEL, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(210)
        layout.addWidget(self.table)
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(
            lambda: self.current_changed.emit(self.current_index())
        )
        self.add_button.clicked.connect(self.add_requested)
        self.remove_button.clicked.connect(self.remove_selected)
        self.duplicate_button.clicked.connect(self.duplicate_selected)
        self.rebuild()

    # -- model -> table --------------------------------------------------------------------

    def set_directions(self, directions: list[Direction]) -> None:
        self.directions = directions
        self.rebuild()

    def rebuild(self) -> None:
        current = self.current_index()
        self._filling = True
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.directions))
        for row, direction in enumerate(self.directions):
            self._fill_row(row, direction)
        self.table.blockSignals(False)
        self._filling = False
        if self.directions:
            self.table.selectRow(min(max(current, 0), len(self.directions) - 1))
        self._update_buttons()
        self.current_changed.emit(self.current_index())

    def _text_item(self, text: str, editable: bool = True) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item

    def _check_item(self, checked: bool, tip: str) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsSelectable
        )
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setToolTip(tip)
        return item

    def _fill_row(self, row: int, d: Direction) -> None:
        self.table.setItem(row, C_USE, self._check_item(d.use, "Include in the fit"))
        label = self._text_item(d.label)
        label.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setForeground(QColor(direction_color(row)))
        label.setToolTip(d.source.path if d.source else "no data file (model only)")
        self.table.setItem(row, C_LABEL, label)
        self.table.setItem(row, C_ALPHA, self._text_item(format_number(d.alpha_q)))
        self.table.setItem(row, C_OMEGA, self._text_item(format_number(d.omega_q)))
        self.table.setItem(row, C_BKG, self._text_item(format_number(d.background)))
        self.table.setItem(
            row,
            C_FITBKG,
            self._check_item(d.fit_spec().fit, "Fit the background of this direction"),
        )
        self.table.setItem(row, C_QMIN, self._text_item(format_number(d.q_min)))
        self.table.setItem(row, C_QMAX, self._text_item(format_number(d.q_max)))
        self.table.setItem(row, C_POINTS, self._text_item(self._points_text(d), editable=False))

    @staticmethod
    def _points_text(d: Direction) -> str:
        if d.q is None:
            return "—"
        return f"{int(d.fit_mask().sum())}/{len(d.q)}"

    def refresh_row_values(self) -> None:
        """Re-read values and point counts after a fit or revert (keeps the selection)."""
        self._filling = True
        self.table.blockSignals(True)
        for row, d in enumerate(self.directions):
            self.table.item(row, C_BKG).setText(format_number(d.background))
            self.table.item(row, C_POINTS).setText(self._points_text(d))
        self.table.blockSignals(False)
        self._filling = False

    # -- table -> model --------------------------------------------------------------------

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        row, column = item.row(), item.column()
        d = self.directions[row]
        recalc = True
        if column == C_USE:
            d.use = item.checkState() == Qt.CheckState.Checked
        elif column == C_FITBKG:
            d.fit_spec().fit = item.checkState() == Qt.CheckState.Checked
            recalc = False
        elif column == C_LABEL:
            d.label = item.text()
            recalc = False
        elif column in (C_ALPHA, C_OMEGA, C_BKG, C_QMIN, C_QMAX):
            self._edit_number(d, column, item)
            if column in (C_QMIN, C_QMAX):
                self._filling = True
                self.table.item(row, C_POINTS).setText(self._points_text(d))
                self._filling = False
        self.changed.emit(recalc)

    def _edit_number(self, d: Direction, column: int, item: QTableWidgetItem) -> None:
        text = item.text().strip()
        optional = column in (C_QMIN, C_QMAX)
        try:
            value = None if (optional and not text) else float(text)
            if value is not None and not math.isfinite(value):
                raise ValueError
            if optional and value is not None and value <= 0:
                raise ValueError
        except ValueError:
            value = None
            self._restore_cell(item, d, column)
            return
        if column == C_ALPHA:
            d.alpha_q = value
        elif column == C_OMEGA:
            d.omega_q = value
        elif column == C_BKG:
            d.background = value
        elif column == C_QMIN:
            d.q_min = value
        else:
            d.q_max = value

    def _restore_cell(self, item: QTableWidgetItem, d: Direction, column: int) -> None:
        old = {
            C_ALPHA: d.alpha_q,
            C_OMEGA: d.omega_q,
            C_BKG: d.background,
            C_QMIN: d.q_min,
            C_QMAX: d.q_max,
        }
        self._filling = True
        item.setText(format_number(old[column]))
        self._filling = False

    # -- buttons ---------------------------------------------------------------------------

    def current_index(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else (self.table.currentRow() if self.directions else -1)

    def _update_buttons(self) -> None:
        has = bool(self.directions)
        self.remove_button.setEnabled(has)
        self.duplicate_button.setEnabled(has)

    def remove_selected(self) -> None:
        index = self.current_index()
        if 0 <= index < len(self.directions):
            del self.directions[index]
            self.rebuild()
            self.structure_changed.emit()

    def duplicate_selected(self) -> None:
        index = self.current_index()
        if 0 <= index < len(self.directions):
            clone = copy.deepcopy(self.directions[index])
            clone.label = f"{clone.label} copy" if clone.label else "copy"
            self.directions.insert(index + 1, clone)
            self.rebuild()
            self.table.selectRow(index + 1)
            self.structure_changed.emit()

    def append_direction(self, direction: Direction) -> None:
        self.directions.append(direction)
        self.rebuild()
        self.table.selectRow(len(self.directions) - 1)
        self.structure_changed.emit()


class SettingsPanel(QWidget):
    """Calculation settings and display options. ``changed`` fires after any edit."""

    changed = Signal()
    display_changed = Signal()

    FORMULAS = [
        ("4c", "4c  diffraction limit (fast)"),
        ("4a", "4a  with refraction, integral"),
        ("4b", "4b  with refraction, series"),
    ]

    def __init__(self, settings: CalcSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self.settings = settings
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.formula = QComboBox()
        for key, text in self.FORMULAS:
            self.formula.addItem(text, key)
        self.formula.setToolTip(
            "Allen's equation 4. 4a/4b include refraction (large pores) and need the wavelength."
        )
        form.addRow("Formula:", self.formula)
        self.wavelength = NumberEdit(settings.wavelength, minimum=1e-6)
        self.wavelength.setToolTip("Wavelength in Å (only used by formulas 4a and 4b)")
        form.addRow("Wavelength (Å):", self.wavelength)
        self.n_alpha = QSpinBox()
        self.n_alpha.setRange(8, 4000)
        self.n_alpha.setSingleStep(8)
        self.n_alpha.setToolTip(
            "Minimum α nodes of the orientation integral; raised automatically when needed"
        )
        self.n_omega = QSpinBox()
        self.n_omega.setRange(8, 4000)
        self.n_omega.setSingleStep(8)
        self.n_omega.setToolTip(
            "Minimum ω nodes of the orientation integral; raised automatically when needed"
        )
        grid = QHBoxLayout()
        grid.addWidget(self.n_alpha)
        grid.addWidget(QLabel("×"))
        grid.addWidget(self.n_omega)
        form.addRow("Integration α × ω:", grid)
        self.threads = QSpinBox()
        self.threads.setRange(0, 256)
        self.threads.setSpecialValueText("auto")
        self.threads.setToolTip("Worker threads; auto = one per CPU")
        form.addRow("Threads:", self.threads)
        self.auto_update = QCheckBox("Update automatically")
        self.auto_update.setChecked(True)
        self.auto_update.setToolTip("Recalculate the model after every change")
        self.show_populations = QCheckBox("Show single populations")
        self.show_populations.setToolTip(
            "Draw the selected population as a dashed line in every direction"
        )
        self.show_populations.setChecked(True)
        form.addRow(self.auto_update)
        form.addRow(self.show_populations)
        self.refresh()
        self.formula.currentIndexChanged.connect(self._formula_changed)
        self.wavelength.value_changed.connect(self._wavelength_changed)
        self.n_alpha.valueChanged.connect(self._n_alpha_changed)
        self.n_omega.valueChanged.connect(self._n_omega_changed)
        self.threads.valueChanged.connect(self._threads_changed)
        self.show_populations.toggled.connect(lambda _checked: self.display_changed.emit())

    def set_settings(self, settings: CalcSettings) -> None:
        self.settings = settings
        self.refresh()

    def refresh(self) -> None:
        s = self.settings
        for widget in (self.formula, self.n_alpha, self.n_omega, self.threads):
            widget.blockSignals(True)
        self.formula.setCurrentIndex(self.formula.findData(s.formula))
        self.n_alpha.setValue(s.n_alpha)
        self.n_omega.setValue(s.n_omega)
        self.threads.setValue(s.n_threads)
        for widget in (self.formula, self.n_alpha, self.n_omega, self.threads):
            widget.blockSignals(False)
        self.wavelength.set_value(s.wavelength)
        self.wavelength.setEnabled(s.formula != "4c")

    def _formula_changed(self) -> None:
        self.settings.formula = self.formula.currentData()
        self.wavelength.setEnabled(self.settings.formula != "4c")
        self.changed.emit()

    def _wavelength_changed(self, value) -> None:
        if value is not None:
            self.settings.wavelength = value
            self.changed.emit()

    def _n_alpha_changed(self, value: int) -> None:
        self.settings.n_alpha = int(value)
        self.changed.emit()

    def _n_omega_changed(self, value: int) -> None:
        self.settings.n_omega = int(value)
        self.changed.emit()

    def _threads_changed(self, value: int) -> None:
        self.settings.n_threads = int(value)
