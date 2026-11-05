"""
Plots: I(Q) of all directions with the model and residuals, and the orientation distributions.

Built on pyIrena's plot helpers (``make_sas_plot``: log-log axes, R = pi/Q axis, export menu, zoom
preservation; the cursor and range helpers), with per-direction colours added here. The data
curves are drawn directly instead of with ``plot_iq_data`` because that function resets the view
whenever its data change, which would happen once per direction on every redraw.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyirena.gui.sas_plot import (
    get_cursor_q_range,
    make_cursors,
    make_sas_plot,
    note_plotted_data,
    set_cursor_q_range,
    set_robust_y_range,
    user_view_active,
)

from anisa.core.engine import ModelResult
from anisa.core.model import AnisaModel
from anisa.gui._qt import QHBoxLayout, QLabel, QPushButton, Qt, QVBoxLayout, QWidget, Signal

__all__ = ["IQPlotWidget", "OrientationPlotWidget", "direction_color", "error_segments"]

#: Colour-blind-friendly cycle; the same colour is used for a direction in the table and plots.
PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#000000",
    "#3b5998",
)


def direction_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    c = pg.mkColor(color)
    return c.red(), c.green(), c.blue(), alpha


def error_segments(q: np.ndarray, intensity: np.ndarray, sigma: np.ndarray):
    """Vertical error bars as one NaN-separated polyline ``(x, y)`` for ``connect="finite"``.

    The lower end is clipped to a positive value so the bar survives the log axis.
    """
    q, intensity, sigma = (np.asarray(a, dtype=float) for a in (q, intensity, sigma))
    low = np.maximum(intensity - sigma, intensity * 1e-3)
    high = intensity + sigma
    n = q.size
    x = np.full(3 * n, np.nan)
    y = np.full(3 * n, np.nan)
    x[0::3] = q
    x[1::3] = q
    y[0::3] = low
    y[1::3] = high
    return x, y


class IQPlotWidget(QWidget):
    """I(Q) with data, model and (optionally) one population per direction, plus residuals."""

    cursors_applied = Signal(float, float)  # q_min, q_max chosen with the cursors

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.status = QLabel("")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cursor_button = QPushButton("Use cursors as Q range of selected direction")
        self.cursor_button.setToolTip(
            "Drag the red and blue cursors, then press this to set Q min / Q max of the direction "
            "selected in the table"
        )
        self.cursor_button.clicked.connect(self._apply_cursors)
        bar.addWidget(self.status, 1)
        bar.addWidget(self.cursor_button)
        layout.addLayout(bar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        layout.addWidget(self.graphics, 1)
        self.main = make_sas_plot(
            self.graphics,
            0,
            0,
            x_label="Q  (Å⁻¹)",
            y_label="I  (cm⁻¹)",
            parent_widget=self,
            jpeg_default_name="anisa_iq",
        )
        self.resid = make_sas_plot(
            self.graphics,
            1,
            0,
            x_label="Q  (Å⁻¹)",
            y_label="(I − model) / σ",
            log_x=True,
            log_y=False,
            x_link=self.main,
            parent_widget=self,
            jpeg_default_name="anisa_residuals",
        )
        self.graphics.ci.layout.setRowStretchFactor(0, 4)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        legend = self.main.addLegend(offset=(-10, 10))
        legend.setLabelTextColor("k")
        self.resid.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#555555", width=1)))
        self._items: list[tuple[pg.PlotItem, object]] = []
        self._cursors = None

    # -- drawing ---------------------------------------------------------------------------

    def _add(self, plot: pg.PlotItem, *args, **kwargs):
        item = plot.plot(*args, **kwargs)
        self._items.append((plot, item))
        return item

    def _clear(self) -> None:
        for plot, item in self._items:
            plot.removeItem(item)
        self._items.clear()

    def update_plot(
        self,
        model: AnisaModel,
        result: ModelResult | None,
        *,
        selected_direction: int = -1,
        selected_population: int = -1,
        show_populations: bool = False,
    ) -> None:
        """Redraw everything. Keeps the user's zoom unless the data or model range changed."""
        self._clear()
        all_q: list[np.ndarray] = []
        all_i: list[np.ndarray] = []
        for i, direction in enumerate(model.directions):
            color = direction_color(i)
            selected = i == selected_direction
            dim = 90 if direction.use else 40
            calc = result.directions[i] if result and i < len(result.directions) else None
            if direction.q is not None and direction.intensity is not None:
                q = np.asarray(direction.q, dtype=float)
                intensity = np.asarray(direction.intensity, dtype=float)
                mask = direction.fit_mask()
                ok = np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
                inside, outside = ok & mask, ok & ~mask
                name = direction.label or f"Dir {i + 1}"
                if inside.any():
                    self._add(
                        self.main,
                        q[inside],
                        intensity[inside],
                        pen=None,
                        symbol="o",
                        symbolSize=6 if selected else 5,
                        symbolPen=None,
                        symbolBrush=_rgba(color, 230 if direction.use else dim),
                        name=name,
                    )
                if outside.any():  # outside the Q range or without a usable uncertainty
                    self._add(
                        self.main,
                        q[outside],
                        intensity[outside],
                        pen=None,
                        symbol="o",
                        symbolSize=5,
                        symbolBrush=None,
                        symbolPen=pg.mkPen(_rgba(color, 150)),
                        name=None if inside.any() else name,
                    )
                if direction.uncertainty is not None and inside.any():
                    x, y = error_segments(
                        q[inside], intensity[inside], np.asarray(direction.uncertainty)[inside]
                    )
                    self._add(self.main, x, y, pen=pg.mkPen(_rgba(color, 110)), connect="finite")
                all_q.append(q[ok])
                all_i.append(intensity[ok])
            if calc is not None:
                good = (calc.q > 0) & (calc.intensity > 0) & np.isfinite(calc.intensity)
                if good.sum() >= 2:
                    self._add(
                        self.main,
                        calc.q[good],
                        calc.intensity[good],
                        pen=pg.mkPen(color, width=3 if selected else 2),
                    )
                    all_q.append(calc.q[good])
                    all_i.append(calc.intensity[good])
                if show_populations and 0 <= selected_population < calc.populations.shape[0]:
                    part = calc.populations[selected_population] + calc.background
                    ok_part = (calc.q > 0) & (part > 0) & np.isfinite(part)
                    if ok_part.sum() >= 2:
                        self._add(
                            self.main,
                            calc.q[ok_part],
                            part[ok_part],
                            pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashLine),
                        )
                if (
                    direction.q is not None
                    and direction.intensity is not None
                    and direction.uncertainty is not None
                ):
                    mask = direction.fit_mask()
                    z = (np.asarray(direction.intensity) - calc.intensity)[mask] / np.asarray(
                        direction.uncertainty
                    )[mask]
                    self._add(
                        self.resid,
                        np.asarray(direction.q)[mask],
                        z,
                        pen=None,
                        symbol="o",
                        symbolSize=4,
                        symbolPen=None,
                        symbolBrush=_rgba(color, 200),
                    )
        if all_q:
            q_all = np.concatenate(all_q)
            i_all = np.concatenate(all_i)
            new_data = note_plotted_data(self.main, q_all, i_all)
            if new_data or not user_view_active(self.main):
                set_robust_y_range(self.main, i_all)
                if not user_view_active(self.main):
                    self.main.setXRange(
                        np.log10(q_all.min()) - 0.1, np.log10(q_all.max()) + 0.1, padding=0
                    )
            self._ensure_cursors(q_all.min(), q_all.max())
        self.resid.enableAutoRange(axis="y")

    # -- cursors ---------------------------------------------------------------------------

    def _ensure_cursors(self, q_min: float, q_max: float) -> None:
        if self._cursors is None:
            self._cursors = make_cursors(self.main, q_min, q_max)

    def set_cursors(self, q_min: float | None, q_max: float | None) -> None:
        """Move the cursors to a direction's fit range (no-op until data are drawn)."""
        if self._cursors is None or q_min is None or q_max is None or not 0 < q_min < q_max:
            return
        self._cursors = set_cursor_q_range(self.main, *self._cursors, q_min, q_max)

    def _apply_cursors(self) -> None:
        low, high = get_cursor_q_range(*self._cursors) if self._cursors else (None, None)
        if low is not None and high is not None:
            self.cursors_applied.emit(low, high)


class OrientationPlotWidget(QWidget):
    """p(alpha) and b(omega) of every population (the Igor "probability graph").

    Both are shown normalised as in Igor: the integral of p(alpha) sin(alpha) over [0, 90 deg]
    is 1, and the integral of b(omega) over [0, 360 deg] is 1.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        layout.addWidget(self.graphics)
        self.alpha_plot = self.graphics.addPlot(row=0, col=0)
        self.omega_plot = self.graphics.addPlot(row=0, col=1)
        for plot, x_label, y_label, x_max in (
            (self.alpha_plot, "α  (°)", "p(α)   ∫ p sin α dα = 1", 90),
            (self.omega_plot, "ω  (°)", "b(ω)   ∫ b dω = 1", 360),
        ):
            plot.setLabel("bottom", x_label)
            plot.setLabel("left", y_label)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setXRange(0, x_max, padding=0.02)
            plot.setMouseEnabled(x=True, y=True)
            plot.getAxis("left").enableAutoSIPrefix(False)
        legend = self.alpha_plot.addLegend(offset=(-10, 10))
        legend.setLabelTextColor("k")

    def update_plot(self, model: AnisaModel, selected_population: int = -1) -> None:
        for plot in (self.alpha_plot, self.omega_plot):
            plot.clear()
        alpha = np.linspace(0.0, np.pi / 2, 721)
        omega = np.linspace(0.0, 2 * np.pi, 1441)
        y_max = {self.alpha_plot: 0.0, self.omega_plot: 0.0}
        for i, pop in enumerate(model.populations):
            try:
                p = pop.alpha_dist.alpha_values(alpha)
                b = pop.omega_dist.omega_values(omega)
                p = p / np.trapezoid(p * np.sin(alpha), alpha)
                b = b / np.trapezoid(b, omega)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                continue
            color = direction_color(i)
            width = 3 if i == selected_population else 1.5
            name = f"{i + 1}. {pop.name}" if pop.name else f"Population {i + 1}"
            self.alpha_plot.plot(np.degrees(alpha), p, pen=pg.mkPen(color, width=width), name=name)
            self.omega_plot.plot(np.degrees(omega), b, pen=pg.mkPen(color, width=width))
            y_max[self.alpha_plot] = max(y_max[self.alpha_plot], float(p.max()))
            y_max[self.omega_plot] = max(y_max[self.omega_plot], float(b.max()))
        for plot, top in y_max.items():
            if top > 0:  # constant curves would otherwise get a meaningless auto range
                plot.setYRange(0.0, 1.15 * top, padding=0)
