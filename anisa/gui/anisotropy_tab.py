"""
The Anisotropy tab: azimuthal scans of the model with the measured cuts overlaid.

A scan sweeps one angle of Q over 0-360 deg at fixed |Q| (see :mod:`anisa.core.anisotropy`); the
tab edits the list of scans (any number, any number of Q values each) and draws the selected one
either as polar plots (one per Q, like Igor's) or as I(angle) on a log axis.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg

from anisa.core.anisotropy import AnisotropyResult, AnisotropyScan, measured_points
from anisa.core.model import AnisaModel
from anisa.gui._qt import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)
from anisa.gui.plots import direction_color
from anisa.gui.widgets import NumberEdit

__all__ = ["AnisotropyTab", "default_q_values", "polar_xy", "parse_q_values"]

MAX_POLAR_PLOTS = 6
POLAR_COLUMNS = 3
RING_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def polar_xy(angle_deg: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian coordinates of polar data: 0 deg to the right, counter-clockwise."""
    theta = np.radians(np.asarray(angle_deg, dtype=float))
    radius = np.asarray(radius, dtype=float)
    return radius * np.cos(theta), radius * np.sin(theta)


def parse_q_values(text: str) -> tuple[float, ...] | None:
    """Q values from comma / space separated text; ``None`` if any is not a positive number."""
    parts = text.replace(",", " ").replace(";", " ").split()
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        return None
    if not values or any(not (v > 0 and math.isfinite(v)) for v in values):
        return None
    return values


def default_q_values(model: AnisaModel) -> tuple[float, ...]:
    """Three Q values spread (in log) over the middle of the measured range."""
    qs = [np.asarray(d.q)[np.asarray(d.q) > 0] for d in model.directions if d.q is not None]
    qs = [q for q in qs if q.size]
    if not qs:
        return (0.002, 0.01)
    low, high = min(q.min() for q in qs), max(q.max() for q in qs)
    return tuple(float(f"{q:.3g}") for q in np.geomspace(low, high, 7)[[1, 3, 5]])


def _font(point_size: float):
    """The application font at ``point_size`` (a font built from an empty family renders oddly)."""
    font = QApplication.font()
    font.setPointSizeF(point_size)
    return font


def _nice_ticks(maximum: float) -> list[float]:
    return [maximum * f for f in RING_FRACTIONS]


class AnisotropyTab(QWidget):
    """Controls and plots for the azimuthal scans. ``changed`` fires after a scan is edited."""

    changed = Signal()
    selection_changed = Signal()

    def __init__(self, scans: list[AnisotropyScan], parent: QWidget | None = None):
        super().__init__(parent)
        self.scans = scans
        self._result: AnisotropyResult | None = None
        self._model: AnisaModel | None = None
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        row = QHBoxLayout()
        self.scan_box = QComboBox()
        self.scan_box.setMinimumWidth(220)
        self.add_button = QPushButton("Add scan")
        self.duplicate_button = QPushButton("Duplicate")
        self.remove_button = QPushButton("Remove")
        for widget in (self.scan_box, self.add_button, self.duplicate_button, self.remove_button):
            row.addWidget(widget)
        row.addStretch(1)
        self.view = QComboBox()
        self.view.addItem("Polar plots", "polar")
        self.view.addItem("I(angle), log axis", "linear")
        self.show_data = QCheckBox("Measured cuts")
        self.show_data.setChecked(True)
        self.show_data.setToolTip(
            "Overlay the measured directions that lie on this scan path, interpolated at each Q "
            "(background subtracted). Each also stands for the opposite direction."
        )
        self.show_populations = QCheckBox("Populations")
        self.show_populations.setToolTip("Draw each population as a dashed line")
        row.addWidget(QLabel("View:"))
        row.addWidget(self.view)
        row.addWidget(self.show_data)
        row.addWidget(self.show_populations)
        layout.addLayout(row)

        form = QHBoxLayout()
        self.sweep = QComboBox()
        self.sweep.addItem("sweep ω_Q  (α_Q fixed)", "omega")
        self.sweep.addItem("sweep α_Q  (ω_Q fixed)", "alpha")
        self.sweep.setToolTip(
            "ω_Q at α_Q = 90°: azimuthal circle in the sample plane.  α_Q at ω_Q = 0° or 90°: a "
            "circle through the sample normal (Igor's Y and Z)."
        )
        self.fixed = NumberEdit(90.0)
        self.fixed.setFixedWidth(70)
        self.fixed_label = QLabel("α_Q =")
        self.points = QSpinBox()
        self.points.setRange(5, 1441)
        self.points.setToolTip("Number of angles from 0° to 360°")
        self.q_text = QLineEdit()
        self.q_text.setToolTip("Q values in Å⁻¹, separated by commas or spaces")
        self.q_text.setMinimumWidth(240)
        form.addWidget(self.sweep)
        form.addWidget(self.fixed_label)
        form.addWidget(self.fixed)
        form.addWidget(QLabel("°   points:"))
        form.addWidget(self.points)
        form.addWidget(QLabel("   Q (Å⁻¹):"))
        form.addWidget(self.q_text, 1)
        layout.addLayout(form)

        self.message = QLabel("")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message)
        self.stack = QStackedWidget()
        self.polar_view = pg.GraphicsLayoutWidget()
        self.polar_view.setBackground("w")
        self.linear_view = pg.GraphicsLayoutWidget()
        self.linear_view.setBackground("w")
        self.stack.addWidget(self.polar_view)
        self.stack.addWidget(self.linear_view)
        layout.addWidget(self.stack, 1)

        self.scan_box.currentIndexChanged.connect(self._scan_selected)
        self.add_button.clicked.connect(self.add_scan)
        self.duplicate_button.clicked.connect(self.duplicate_scan)
        self.remove_button.clicked.connect(self.remove_scan)
        self.sweep.currentIndexChanged.connect(self._sweep_changed)
        self.fixed.value_changed.connect(self._fixed_changed)
        self.points.valueChanged.connect(self._points_changed)
        self.q_text.editingFinished.connect(self._q_edited)
        self.view.currentIndexChanged.connect(self._redraw)
        self.show_data.toggled.connect(self._redraw)
        self.show_populations.toggled.connect(self._redraw)
        self.rebuild()

    # -- the scan list ---------------------------------------------------------------------

    def set_scans(self, scans: list[AnisotropyScan]) -> None:
        self.scans = scans
        self._result = None
        self.rebuild()

    def current_index(self) -> int:
        return self.scan_box.currentIndex() if self.scans else -1

    def current_scan(self) -> AnisotropyScan | None:
        index = self.current_index()
        return self.scans[index] if 0 <= index < len(self.scans) else None

    def rebuild(self) -> None:
        current = self.scan_box.currentIndex()
        self._loading = True
        self.scan_box.clear()
        for i, scan in enumerate(self.scans):
            self.scan_box.addItem(f"{i + 1}. {scan.title()}")
        if self.scans:
            self.scan_box.setCurrentIndex(min(max(current, 0), len(self.scans) - 1))
        self._loading = False
        self._load_controls()
        self._update_enabled()
        self._redraw()

    def _update_enabled(self) -> None:
        has = bool(self.scans)
        for widget in (
            self.duplicate_button,
            self.remove_button,
            self.sweep,
            self.fixed,
            self.points,
            self.q_text,
        ):
            widget.setEnabled(has)
        self.message.setText(
            "" if has else "Add a scan to see how the model depends on the direction of Q."
        )

    def _load_controls(self) -> None:
        scan = self.current_scan()
        if scan is None:
            return
        self._loading = True
        self.sweep.setCurrentIndex(self.sweep.findData(scan.sweep))
        self.fixed_label.setText("α_Q =" if scan.sweep == "omega" else "ω_Q =")
        self.fixed.set_value(scan.fixed)
        self.points.setValue(scan.n_points)
        self.q_text.setText(", ".join(f"{q:g}" for q in scan.q_values))
        self._loading = False

    def _edited(self) -> None:
        scan = self.current_scan()
        if scan is not None:
            self.scan_box.setItemText(
                self.current_index(), f"{self.current_index() + 1}. {scan.title()}"
            )
        self.changed.emit()

    def add_scan(self, q_values: tuple[float, ...] | None = None) -> None:
        # The circle through the sample normal is where textured samples show their anisotropy
        # (Igor's Y and Z); an omega scan in the sample plane is flat for axially symmetric ones.
        scan = AnisotropyScan(
            sweep="alpha",
            fixed=0.0,
            q_values=q_values if isinstance(q_values, tuple) else (0.002, 0.01),
        )
        self.scans.append(scan)
        self.rebuild()
        self.scan_box.setCurrentIndex(len(self.scans) - 1)
        self.changed.emit()

    def duplicate_scan(self) -> None:
        scan = self.current_scan()
        if scan is None:
            return
        clone = AnisotropyScan.from_dict(scan.to_dict())
        self.scans.insert(self.current_index() + 1, clone)
        self.rebuild()
        self.scan_box.setCurrentIndex(self.current_index() + 1)
        self.changed.emit()

    def remove_scan(self) -> None:
        index = self.current_index()
        if index < 0:
            return
        del self.scans[index]
        self._result = None
        self.rebuild()
        self.changed.emit()

    # -- editing ---------------------------------------------------------------------------

    def _scan_selected(self) -> None:
        if self._loading:
            return
        self._result = None
        self._load_controls()
        self._redraw()
        self.selection_changed.emit()

    def _sweep_changed(self) -> None:
        scan = self.current_scan()
        if scan is None or self._loading:
            return
        scan.sweep = self.sweep.currentData()
        self.fixed_label.setText("α_Q =" if scan.sweep == "omega" else "ω_Q =")
        self._edited()

    def _fixed_changed(self, value) -> None:
        scan = self.current_scan()
        if scan is not None and value is not None and not self._loading:
            scan.fixed = float(value)
            self._edited()

    def _points_changed(self, value: int) -> None:
        scan = self.current_scan()
        if scan is not None and not self._loading:
            scan.n_points = int(value)
            self._edited()

    def _q_edited(self) -> None:
        scan = self.current_scan()
        if scan is None or self._loading:
            return
        values = parse_q_values(self.q_text.text())
        if values is None:
            self.q_text.setText(", ".join(f"{q:g}" for q in scan.q_values))  # reject, restore
            return
        if values != scan.q_values:
            scan.q_values = values
            self.q_text.setText(", ".join(f"{q:g}" for q in values))
            self._edited()

    # -- drawing ---------------------------------------------------------------------------

    def update_result(self, result: AnisotropyResult | None, model: AnisaModel) -> None:
        self._result, self._model = result, model
        self._redraw()

    def _redraw(self) -> None:
        self.stack.setCurrentIndex(0 if self.view.currentData() == "polar" else 1)
        self.polar_view.clear()
        self.linear_view.clear()
        result, model = self._result, self._model
        if result is None or model is None or result.scan != self.current_scan():
            return
        if self.view.currentData() == "polar":
            self._draw_polar(result, model)
        else:
            self._draw_linear(result, model)

    def _measured(self, model: AnisaModel, scan: AnisotropyScan, q: float):
        if not self.show_data.isChecked():
            return None
        points = measured_points(model, scan, q)
        return points if points.angles.size else None

    def _draw_linear(self, result: AnisotropyResult, model: AnisaModel) -> None:
        plot = self.linear_view.addPlot()
        plot.setLogMode(y=True)
        plot.setLabel("bottom", "swept angle  (°)")
        plot.setLabel("left", "I  (cm⁻¹)")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.getAxis("bottom").setTickSpacing(90, 30)
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.setXRange(0, 360, padding=0.01)
        legend = plot.addLegend(offset=(-10, 10))
        legend.setLabelTextColor("k")
        for iq, q in enumerate(result.scan.q_values):
            color = direction_color(iq)
            good = result.intensity[iq] > 0
            plot.plot(
                result.angles[good],
                result.intensity[iq][good],
                pen=pg.mkPen(color, width=2),
                name=f"Q = {q:g} Å⁻¹",
            )
            if self.show_populations.isChecked():
                for ip in range(result.populations.shape[0]):
                    part = result.populations[ip, iq]
                    ok = part > 0
                    plot.plot(
                        result.angles[ok],
                        part[ok],
                        pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                    )
            points = self._measured(model, result.scan, q)
            if points is not None:
                ok = points.intensity > 0
                plot.plot(
                    points.angles[ok],
                    points.intensity[ok],
                    pen=None,
                    symbol="o",
                    symbolSize=8,
                    symbolBrush=color,
                    symbolPen=pg.mkPen("k", width=0.5),
                )

    def _draw_polar(self, result: AnisotropyResult, model: AnisaModel) -> None:
        shown = list(enumerate(result.scan.q_values))[:MAX_POLAR_PLOTS]
        for position, (iq, q) in enumerate(shown):
            plot = self.polar_view.addPlot(
                row=position // POLAR_COLUMNS, col=position % POLAR_COLUMNS
            )
            self._polar_plot(plot, result, model, iq, q)
        if len(result.scan.q_values) > MAX_POLAR_PLOTS:
            self.message.setText(
                f"Showing the first {MAX_POLAR_PLOTS} of {len(result.scan.q_values)} Q values "
                "(the linear view shows all)."
            )
        else:
            self.message.setText("")

    def _polar_plot(self, plot: pg.PlotItem, result, model, iq: int, q: float) -> None:
        color = direction_color(iq)
        angles = result.angles
        curves = [(angles, result.intensity[iq])]
        points = self._measured(model, result.scan, q)
        radius_max = float(np.nanmax(result.intensity[iq]))
        if points is not None:
            radius_max = max(radius_max, float(np.nanmax(points.intensity)))
        if not radius_max > 0:
            return
        plot.setTitle(f"Q = {q:g} Å⁻¹", size="10pt")
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setAspectLocked(True)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        grid = pg.mkPen("#bbbbbb", width=1)
        theta = np.linspace(0, 2 * np.pi, 181)
        for ring in _nice_ticks(radius_max):
            plot.plot(ring * np.cos(theta), ring * np.sin(theta), pen=grid)
            label = pg.TextItem(f"{ring:.3g}", color="#777777", anchor=(0, 1))
            label.setFont(_font(8))
            label.setPos(ring * math.cos(math.radians(45)), ring * math.sin(math.radians(45)))
            plot.addItem(label)
        for angle in range(0, 360, 30):
            x, y = polar_xy(np.array([angle, angle]), np.array([0.0, radius_max]))
            plot.plot(x, y, pen=grid)
            tip = pg.TextItem(f"{angle}°", color="#555555", anchor=(0.5, 0.5))
            tip.setFont(_font(9))
            tx, ty = polar_xy(np.array([angle]), np.array([radius_max * 1.1]))
            tip.setPos(float(tx[0]), float(ty[0]))
            plot.addItem(tip)
        if self.show_populations.isChecked():
            for ip in range(result.populations.shape[0]):
                x, y = polar_xy(angles, result.populations[ip, iq])
                plot.plot(x, y, pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
        for a, r in curves:
            x, y = polar_xy(a, r)
            plot.plot(x, y, pen=pg.mkPen(color, width=2.5))
        if points is not None:
            x, y = polar_xy(points.angles, points.intensity)
            plot.plot(
                x,
                y,
                pen=None,
                symbol="o",
                symbolSize=8,
                symbolBrush=color,
                symbolPen=pg.mkPen("k", width=0.5),
            )
        limit = radius_max * 1.4
        plot.setXRange(-limit, limit, padding=0)
        plot.setYRange(-limit, limit, padding=0)
