"""
Reusable input widgets: a number field and a grid of fittable parameters.

The grid mirrors pyIrena's parameter layout (``Fit? | Parameter | Value | Min | Max | +- error``)
and edits a model object in place: anything with ``get_parameter`` / ``set_parameter`` /
``fit_spec`` (a :class:`~anisa.core.model.Population`). Panels stay thin; all numerics live in
``anisa.core``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anisa.gui._qt import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    Qt,
    QWidget,
    Signal,
)

__all__ = ["NumberEdit", "ParamSpec", "ParameterGrid", "ParameterRow", "format_number"]


def format_number(value: float | None) -> str:
    """Shortest readable text for a parameter value ('' for None)."""
    if value is None:
        return ""
    if value == 0:
        return "0"
    return f"{value:.6g}"


class NumberEdit(QLineEdit):
    """A line edit holding one number.

    ``value_changed`` fires only when the committed value actually changes (Enter or focus loss).
    Text that is not a number, or outside ``[minimum, maximum]``, is rejected: the field returns
    to the previous value. With ``allow_empty`` an empty field means ``None`` (no limit).
    """

    value_changed = Signal(object)

    def __init__(
        self,
        value: float | None = None,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        allow_empty: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.minimum, self.maximum, self.allow_empty = minimum, maximum, allow_empty
        self._value: float | None = value
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setText(format_number(value))
        self.editingFinished.connect(self._commit)

    def value(self) -> float | None:
        return self._value

    def set_value(self, value: float | None) -> None:
        """Show ``value`` without emitting ``value_changed``."""
        self._value = value
        self.setText(format_number(value))

    def _commit(self) -> None:
        text = self.text().strip()
        if not text:
            new = None if self.allow_empty else self._value
        else:
            try:
                new = float(text)
            except ValueError:
                new = self._value
            else:
                bad = not math.isfinite(new)
                bad = bad or (self.minimum is not None and new < self.minimum)
                bad = bad or (self.maximum is not None and new > self.maximum)
                if bad:
                    new = self._value
        self.setText(format_number(new))
        if new != self._value:
            self._value = new
            self.value_changed.emit(new)


@dataclass(frozen=True)
class ParamSpec:
    """One row of a :class:`ParameterGrid`."""

    path: str
    label: str
    tooltip: str = ""
    minimum: float | None = None  # hard limit on the value itself (not on the fit limits)
    maximum: float | None = None


class ParameterRow:
    def __init__(self, spec: ParamSpec, widgets: list[QWidget], fit, value, low, high, error):
        self.spec = spec
        self.widgets = widgets
        self.fit: QCheckBox = fit
        self.value: NumberEdit = value
        self.low: NumberEdit = low
        self.high: NumberEdit = high
        self.error: QLabel = error


class ParameterGrid(QWidget):
    """Rows of ``[fit?] label value min max +-error`` editing ``owner`` in place.

    ``changed(bool)`` is emitted after every edit; the argument says whether the model result
    changes (a new value) or only the fit settings do (fit flag, limits).
    """

    changed = Signal(bool)

    HEADERS = ("Fit?", "", "Value", "Min", "Max", "± σ")

    def __init__(self, owner, specs: list[ParamSpec], parent: QWidget | None = None):
        super().__init__(parent)
        self.owner = owner
        self._rows: dict[str, ParameterRow] = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        for column, text in enumerate(self.HEADERS):
            header = QLabel(f"<small>{text}</small>")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(header, 0, column)
        for row, spec in enumerate(specs, start=1):
            self._add_row(grid, row, spec)
        grid.setColumnStretch(1, 1)
        self.refresh()

    # -- construction ----------------------------------------------------------------------

    def _add_row(self, grid: QGridLayout, row: int, spec: ParamSpec) -> None:
        fit = QCheckBox()
        fit.setToolTip("Vary this parameter when fitting")
        label = QLabel(spec.label)
        if spec.tooltip:
            label.setToolTip(spec.tooltip)
        value = NumberEdit(0.0, minimum=spec.minimum, maximum=spec.maximum)
        low = NumberEdit(None, allow_empty=True)
        high = NumberEdit(None, allow_empty=True)
        for edit, tip in ((low, "Lower fit limit (empty = none)"), (high, "Upper fit limit")):
            edit.setToolTip(tip)
            edit.setFixedWidth(64)
        value.setFixedWidth(84)
        error = QLabel("")
        error.setMinimumWidth(62)
        error.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        widgets = [fit, label, value, low, high, error]
        for column, widget in enumerate(widgets):
            grid.addWidget(widget, row, column)
        entry = ParameterRow(spec, widgets, fit, value, low, high, error)
        self._rows[spec.path] = entry
        fit.toggled.connect(lambda checked, p=spec.path: self._fit_toggled(p, checked))
        value.value_changed.connect(lambda v, p=spec.path: self._value_edited(p, v))
        low.value_changed.connect(lambda v, p=spec.path: self._limit_edited(p, "min", v))
        high.value_changed.connect(lambda v, p=spec.path: self._limit_edited(p, "max", v))

    def row(self, path: str) -> ParameterRow:
        """The widgets of one parameter (``fit``, ``value``, ``low``, ``high``, ``error``)."""
        return self._rows[path]

    # -- model -> widgets ------------------------------------------------------------------

    def set_owner(self, owner) -> None:
        self.owner = owner
        self.refresh()

    def refresh(self) -> None:
        """Show the current values of the owner (emits nothing)."""
        for path, row in self._rows.items():
            spec = self.owner.fit_spec(path)
            row.value.set_value(self.owner.get_parameter(path))
            row.low.set_value(spec.min)
            row.high.set_value(spec.max)
            row.fit.blockSignals(True)
            row.fit.setChecked(bool(spec.fit))
            row.fit.blockSignals(False)

    def set_row_visible(self, path: str, visible: bool) -> None:
        for widget in self._rows[path].widgets:
            widget.setVisible(visible)

    def set_errors(self, errors: dict[str, float | None], tooltip: str = "") -> None:
        """Show ``+- error`` per parameter path (``None`` / missing clears the label)."""
        for path, row in self._rows.items():
            error = errors.get(path)
            if error is None or not math.isfinite(error):
                row.error.setText("")
                row.error.setToolTip("")
            else:
                row.error.setText(f"± {error:.3g}")
                row.error.setToolTip(tooltip)

    # -- widgets -> model ------------------------------------------------------------------

    def _value_edited(self, path: str, value: float | None) -> None:
        if value is not None:
            self.owner.set_parameter(path, value)
            self.changed.emit(True)

    def _limit_edited(self, path: str, which: str, value: float | None) -> None:
        setattr(self.owner.fit_spec(path), which, value)
        self.changed.emit(False)

    def _fit_toggled(self, path: str, checked: bool) -> None:
        self.owner.fit_spec(path).fit = checked
        self.changed.emit(False)
