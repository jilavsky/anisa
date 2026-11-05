"""
Background jobs, so the window stays responsive while the model is calculated or fitted.

A job runs on a snapshot (deep copy) of the model, never on the objects the GUI is editing, so
nothing is shared between the GUI thread and the worker except the thread-safe
:class:`~anisa.core.engine.Engine` cache. Results come back as Qt signals.
"""

from __future__ import annotations

import copy
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass

from anisa.core.anisotropy import AnisotropyResult, AnisotropyScan, calculate_scan
from anisa.core.engine import Engine, ModelResult
from anisa.core.fitting import (
    FitResult,
    FitStatistics,
    MonteCarloResult,
    fit,
    monte_carlo,
    statistics_from_result,
)
from anisa.core.model import AnisaModel
from anisa.gui._qt import QThread, Signal

log = logging.getLogger(__name__)

__all__ = [
    "CalcOutput",
    "TaskThread",
    "anisotropy_job",
    "calculation_job",
    "fit_job",
    "monte_carlo_job",
]


class TaskThread(QThread):
    """Runs ``job(thread)`` off the GUI thread.

    Signals: ``progress(int, float)`` (job specific), ``done(object)`` with the job's return
    value, ``failed(str)`` with a one-line error (the traceback goes to the log).
    """

    progress = Signal(int, float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, job: Callable[[TaskThread], object], parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:  # executed in the worker thread
        try:
            result = self._job(self)
        except Exception as error:
            log.error("background job failed:\n%s", traceback.format_exc())
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.done.emit(result)


@dataclass
class CalcOutput:
    """What a calculation job returns: the model curves and chi-squared against the data."""

    result: ModelResult
    statistics: FitStatistics


def calculation_job(model: AnisaModel, engine: Engine, q_list: list):
    """Evaluate ``model`` at ``q_list`` (one Q array per direction)."""

    def job(thread: TaskThread) -> CalcOutput:
        result = engine.evaluate(model, q_list)
        return CalcOutput(result, statistics_from_result(model, result))

    return job


def fit_job(model: AnisaModel, engine: Engine):
    """Fit ``model`` in place. Returns the :class:`FitResult`; ``thread.cancel()`` stops it."""

    def job(thread: TaskThread) -> FitResult:
        return fit(
            model,
            engine=engine,
            progress=lambda n, chi: thread.progress.emit(int(n), float(chi)),
            cancel=lambda: thread.cancelled,
        )

    return job


def monte_carlo_job(model: AnisaModel, result: FitResult, n_runs: int, seed: int = 0):
    """Monte-Carlo uncertainty of a fitted model (worker processes inside the job)."""
    snapshot = copy.deepcopy(model)

    def job(thread: TaskThread) -> MonteCarloResult:
        return monte_carlo(
            snapshot,
            result,
            n_runs,
            seed=seed,
            progress=lambda done, total: thread.progress.emit(int(done), float(total)),
            cancel=lambda: thread.cancelled,
        )

    return job


def anisotropy_job(model: AnisaModel, scan: AnisotropyScan, engine: Engine):
    """Model along one azimuthal scan; returns an :class:`AnisotropyResult`."""

    def job(thread: TaskThread) -> AnisotropyResult:
        return calculate_scan(model, scan, engine)

    return job
