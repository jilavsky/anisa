"""
Least-squares fitting of the model to the measured data of all directions at once.

The residual vector is ``(I_data - I_model) / sigma`` over every point inside the fit mask of
every direction that has data and is marked ``use``, concatenated (Igor: one long wave and
``FuncFit`` with weights 1/sigma). The optimiser is ``scipy.optimize.least_squares`` (trust-region
reflective, box constraints from each parameter's limits).

Which parameters vary is decided by each parameter's ``FitSpec.fit`` flag, per population and per
direction. (Igor's ``ASAS_FitCreateParametersWvs`` read population 1's "use parameters" flags for
every population; here each population uses its own.) A parameter that does not apply is skipped
rather than varied uselessly: orientation parameters of a population whose distribution is a
table, and ``eta`` / ``pack`` of a population without interference.

Finite differences use a **relative step of 1e-3**. The model is smooth, but its numerical
discretisation (grid sizes, K nodes) changes in integer steps, which leaves ~1e-6 relative noise;
scipy's default step (~1e-8) would be dominated by it. Only the population whose parameter moved
is recomputed (see ``anisa.core.engine``), so a Jacobian column is cheap.

Uncertainties:

* ``covariance`` from the Jacobian at the solution, ``(J^T J)^-1`` scaled by the reduced
  chi-squared (the usual correction when the absolute scale of the uncertainties is not fully
  trusted; for correct uncertainties the factor is about 1);
* :func:`monte_carlo`: refit data re-drawn within their uncertainties, which also captures
  non-linearity and correlated parameters.
"""

from __future__ import annotations

import copy
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from anisa.core.engine import Engine
from anisa.core.model import AnisaModel, Direction, Population

__all__ = [
    "FitCancelled",
    "FitParameter",
    "FitResult",
    "FitStatistics",
    "MonteCarloResult",
    "ParameterResult",
    "active_parameters",
    "apply_values",
    "fit",
    "goodness_of_fit",
    "monte_carlo",
    "revert",
    "statistics_from_result",
]

#: Relative finite-difference step (see the module docstring).
DIFF_STEP = 1.0e-3


class FitCancelled(Exception):
    """Raised inside the fit when the ``cancel`` callback returns True."""


@dataclass(frozen=True)
class FitParameter:
    """One value the fit may vary.

    Attributes:
        owner: ``"population"`` or ``"direction"``.
        index: index into ``model.populations`` or ``model.directions``.
        path: parameter path, e.g. ``"radius"``, ``"alpha_par2"``, ``"background"``.
        label: human-readable name, e.g. ``"Pop 2 (cracks): radius"``.
        value: current value.
        lower, upper: limits (``-inf`` / ``inf`` where unbounded).
    """

    owner: str
    index: int
    path: str
    label: str
    value: float
    lower: float
    upper: float


@dataclass
class ParameterResult:
    """A fitted parameter: ``std`` is ``nan`` when it could not be estimated."""

    parameter: FitParameter
    start: float
    value: float
    std: float = math.nan
    at_lower: bool = False
    at_upper: bool = False


@dataclass(frozen=True)
class FitStatistics:
    """Chi-squared of the current model against the data (Igor ``ASAS_CalcChiSquare``).

    ``direction_chi_squared[i]`` and ``direction_points[i]`` are ``None`` / 0 for directions
    without data or not in use. ``reduced`` is ``chi_squared / n_points`` as in Igor; the
    degrees-of-freedom version is in :class:`FitResult`.
    """

    chi_squared: float
    n_points: int
    direction_chi_squared: tuple[float | None, ...]
    direction_points: tuple[int, ...]

    @property
    def reduced(self) -> float:
        return self.chi_squared / self.n_points if self.n_points else math.nan

    def direction_reduced(self, index: int) -> float:
        n = self.direction_points[index]
        chi = self.direction_chi_squared[index]
        return chi / n if n and chi is not None else math.nan


@dataclass
class FitResult:
    """Outcome of :func:`fit`.

    ``chi_squared`` is the sum of squared weighted residuals; ``reduced_chi_squared`` divides by
    the degrees of freedom ``n_points - n_parameters``. ``covariance`` is scaled as described in
    the module docstring and ordered like ``parameters``.
    """

    success: bool
    cancelled: bool
    message: str
    parameters: list[ParameterResult]
    chi_squared: float
    reduced_chi_squared: float
    n_points: int
    n_evaluations: int
    elapsed_seconds: float
    statistics: FitStatistics
    covariance: np.ndarray | None = None
    monte_carlo: MonteCarloResult | None = field(default=None, repr=False)

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def degrees_of_freedom(self) -> int:
        return self.n_points - self.n_parameters

    def report(self, direction_labels: list[str] | None = None) -> str:
        """Plain-text summary: status, chi-squared, every parameter (start -> fit +- error) and
        chi-squared per direction. ``direction_labels`` names the directions (default Dir 1 ...)."""
        state = (
            "cancelled" if self.cancelled else ("converged" if self.success else "NOT converged")
        )
        lines = [
            f"Fit {state}: {self.message}",
            f"{self.n_points} points, {self.n_parameters} parameters, "
            f"{self.n_evaluations} evaluations, {self.elapsed_seconds:.1f} s",
            f"chi2 = {self.chi_squared:.6g}, chi2/dof = {self.reduced_chi_squared:.4g}",
            "",
        ]
        width = max((len(p.parameter.label) for p in self.parameters), default=0)
        for p in self.parameters:
            error = f" +- {p.std:.3g}" if np.isfinite(p.std) else ""
            flag = (
                "  [at lower limit]" if p.at_lower else ("  [at upper limit]" if p.at_upper else "")
            )
            if self.monte_carlo is not None:
                k = self.parameters.index(p)
                if np.isfinite(self.monte_carlo.std[k]):
                    error += f"  (Monte Carlo +- {self.monte_carlo.std[k]:.3g})"
            lines.append(
                f"  {p.parameter.label:<{width}}  {p.start:.6g} -> {p.value:.6g}{error}{flag}"
            )
        stats = self.statistics
        lines += ["", "chi2 per point by direction:"]
        for i, n in enumerate(stats.direction_points):
            if n:
                name = (
                    direction_labels[i]
                    if direction_labels and direction_labels[i]
                    else f"Dir {i + 1}"
                )
                lines.append(f"  {name}: {stats.direction_reduced(i):.4g}  ({n} points)")
        return "\n".join(lines)

    def start_values(self) -> list[float]:
        return [p.start for p in self.parameters]

    def values(self) -> list[float]:
        return [p.value for p in self.parameters]


@dataclass
class MonteCarloResult:
    """Parameter spread from refitting data re-drawn within their uncertainties.

    ``values`` has shape ``(n_runs, n_parameters)`` (only successful runs); ``std`` is its
    standard deviation per parameter (ddof = 1).
    """

    values: np.ndarray
    std: np.ndarray
    n_requested: int
    n_failed: int


# -- parameters -------------------------------------------------------------------------------


def _bound(value: float | None, default: float) -> float:
    return default if value is None else float(value)


def _population_label(index: int, pop: Population, path: str) -> str:
    name = f" ({pop.name})" if pop.name else ""
    return f"Pop {index + 1}{name}: {path}"


def _applies(pop: Population, path: str) -> bool:
    if path.startswith("alpha_par"):
        return pop.alpha_dist.mode == "parametric"
    if path.startswith("omega_par"):
        return pop.omega_dist.mode == "parametric"
    if path in ("eta", "pack"):
        return pop.use_interference
    return True


def active_parameters(model: AnisaModel) -> list[FitParameter]:
    """The parameters the fit will vary, in a fixed order (populations first, then directions).

    A parameter is active if its ``fit`` flag is set and it applies to the population as
    configured (see the module docstring).
    """
    out: list[FitParameter] = []
    for i, pop in enumerate(model.populations):
        for path in pop.parameter_paths():
            spec = pop.fit_specs.get(path)
            if spec is None or not spec.fit or not _applies(pop, path):
                continue
            out.append(
                FitParameter(
                    "population",
                    i,
                    path,
                    _population_label(i, pop, path),
                    pop.get_parameter(path),
                    _bound(spec.min, -math.inf),
                    _bound(spec.max, math.inf),
                )
            )
    for i, direction in enumerate(model.directions):
        spec = direction.fit_specs.get("background")
        if spec is not None and spec.fit:
            name = f" ({direction.label})" if direction.label else ""
            out.append(
                FitParameter(
                    "direction",
                    i,
                    "background",
                    f"Dir {i + 1}{name}: background",
                    float(direction.background),
                    _bound(spec.min, -math.inf),
                    _bound(spec.max, math.inf),
                )
            )
    return out


def apply_values(model: AnisaModel, parameters: list[FitParameter], values) -> None:
    """Write ``values`` (one per parameter) into ``model`` in place."""
    for parameter, value in zip(parameters, values, strict=True):
        if parameter.owner == "population":
            model.populations[parameter.index].set_parameter(parameter.path, float(value))
        else:
            model.directions[parameter.index].background = float(value)


def revert(model: AnisaModel, result: FitResult) -> None:
    """Restore the parameter values the fit started from (Igor "Reverse fit")."""
    apply_values(model, [p.parameter for p in result.parameters], result.start_values())


# -- residuals --------------------------------------------------------------------------------


def _fit_directions(model: AnisaModel) -> list[tuple[int, Direction, np.ndarray]]:
    """``(index, direction, mask)`` of every direction the fit uses."""
    used = []
    for i, direction in enumerate(model.directions):
        if not direction.use:
            continue
        mask = direction.fit_mask()
        if mask.any():
            used.append((i, direction, mask))
    return used


class _Problem:
    """Residuals of a model against its data, with cached masks and Q arrays."""

    def __init__(self, model: AnisaModel, engine: Engine):
        self.model = model
        self.engine = engine
        self.used = _fit_directions(model)
        if not self.used:
            raise ValueError("there are no data to fit: load data into at least one direction")
        self.q = [np.ascontiguousarray(d.q[m], dtype=float) for _, d, m in self.used]
        self.intensity = [np.asarray(d.intensity, dtype=float)[m] for _, d, m in self.used]
        sigma = []
        for _, d, m in self.used:
            if d.uncertainty is None:
                raise ValueError(f"direction {d.label or '?'} has no uncertainty")
            sigma.append(np.asarray(d.uncertainty, dtype=float)[m])
        self.sigma = sigma
        # a model made of the used directions only, sharing populations and settings
        self.sub = AnisaModel([d for _, d, _ in self.used], model.populations, model.settings)
        self.n_points = int(sum(q.size for q in self.q))

    def per_direction(self) -> list[np.ndarray]:
        result = self.engine.evaluate(self.sub, self.q)
        return [
            (data - calc.intensity) / sigma
            for data, calc, sigma in zip(self.intensity, result.directions, self.sigma, strict=True)
        ]

    def residuals(self) -> np.ndarray:
        return np.concatenate(self.per_direction())


def goodness_of_fit(model: AnisaModel, engine: Engine | None = None) -> FitStatistics:
    """Chi-squared per direction and in total for the model as it stands."""
    problem = _Problem(model, engine or Engine())
    parts = problem.per_direction()
    chi = [None] * len(model.directions)
    points = [0] * len(model.directions)
    for (index, _, _), r in zip(problem.used, parts, strict=True):
        chi[index] = float(np.sum(r * r))
        points[index] = int(r.size)
    return FitStatistics(
        chi_squared=float(sum(c for c in chi if c is not None)),
        n_points=problem.n_points,
        direction_chi_squared=tuple(chi),
        direction_points=tuple(points),
    )


def statistics_from_result(model: AnisaModel, result) -> FitStatistics:
    """Chi-squared of an already evaluated model (``result`` from ``Engine.evaluate`` at each
    direction's own measured Q), without evaluating it again.

    Directions without data, not in use, or evaluated on a different Q grid are left out.
    """
    n = len(model.directions)
    chi: list[float | None] = [None] * n
    points = [0] * n
    for i, (direction, calc) in enumerate(zip(model.directions, result.directions, strict=False)):
        if not direction.use or direction.q is None or direction.intensity is None:
            continue
        if direction.uncertainty is None or calc.q.shape != np.shape(direction.q):
            continue
        mask = direction.fit_mask()
        if not mask.any():
            continue
        z = (np.asarray(direction.intensity)[mask] - calc.intensity[mask]) / np.asarray(
            direction.uncertainty
        )[mask]
        chi[i] = float(np.dot(z, z))
        points[i] = int(z.size)
    return FitStatistics(
        chi_squared=float(sum(c for c in chi if c is not None)),
        n_points=sum(points),
        direction_chi_squared=tuple(chi),
        direction_points=tuple(points),
    )


# -- the fit ----------------------------------------------------------------------------------


def fit(
    model: AnisaModel,
    *,
    engine: Engine | None = None,
    max_nfev: int | None = None,
    progress: Callable[[int, float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    apply: bool = True,
) -> FitResult:
    """Fit the parameters whose ``fit`` flag is set to the data of all directions.

    Args:
        model: model with data. Modified in place: on return it holds the best parameters found
            (``apply=True``), or the start values (``apply=False``).
        engine: engine to evaluate with (keep one across calls to reuse its cache).
        max_nfev: limit on model evaluations (default ``100 * n_parameters``).
        progress: ``progress(n_evaluations, reduced_chi_squared)`` after every evaluation. Called
            from the fitting thread.
        cancel: polled after every evaluation; if it returns True the fit stops and returns the
            best parameters found so far with ``cancelled=True``.
        apply: write the result into ``model`` (default).

    Raises:
        ValueError: nothing to fit (no data, or no parameter marked for fitting), or a parameter
            with ``min >= max``.
    """
    engine = engine or Engine()
    parameters = active_parameters(model)
    if not parameters:
        raise ValueError("no parameter is marked for fitting")
    for p in parameters:
        if not p.lower < p.upper:
            raise ValueError(f"{p.label}: lower limit {p.lower} is not below upper limit {p.upper}")
    problem = _Problem(model, engine)
    n_params = len(parameters)

    lower = np.array([p.lower for p in parameters])
    upper = np.array([p.upper for p in parameters])
    start = np.array([p.value for p in parameters], dtype=float)
    # the trust-region method needs a strictly feasible start
    margin = 1e-9 * np.maximum(1.0, np.abs(start))
    x0 = np.clip(start, lower + margin, upper - margin)

    state = {"nfev": 0, "best_chi": math.inf, "best_x": x0.copy()}
    started = time.perf_counter()

    def residuals(x: np.ndarray) -> np.ndarray:
        apply_values(model, parameters, x)
        r = problem.residuals()
        chi = float(np.dot(r, r))
        state["nfev"] += 1
        if chi < state["best_chi"]:
            state["best_chi"], state["best_x"] = chi, x.copy()
        if progress is not None:
            progress(state["nfev"], chi / max(problem.n_points - n_params, 1))
        if cancel is not None and cancel():
            raise FitCancelled
        return r

    cancelled = False
    solution = None
    message = ""
    try:
        solution = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            diff_step=DIFF_STEP,
            max_nfev=max_nfev or 100 * n_params,
        )
        best = solution.x
        message = solution.message
        success = bool(solution.success)
    except FitCancelled:
        cancelled, success = True, False
        best, message = state["best_x"], "cancelled by the user"
    except BaseException:
        apply_values(model, parameters, start)  # never leave the model half-fitted
        raise

    apply_values(model, parameters, best)
    residual = problem.residuals()
    chi = float(np.dot(residual, residual))
    dof = problem.n_points - n_params
    reduced = chi / dof if dof > 0 else math.nan

    covariance = None
    std = np.full(n_params, math.nan)
    if solution is not None and dof > 0:
        jac = solution.jac
        try:
            covariance = np.linalg.pinv(jac.T @ jac) * reduced
            # pinv of an ill-conditioned matrix is only symmetric to rounding noise
            covariance = 0.5 * (covariance + covariance.T)
            std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        except np.linalg.LinAlgError:
            covariance = None

    results = []
    for k, p in enumerate(parameters):
        tol = 1e-8 * max(1.0, abs(best[k]))
        results.append(
            ParameterResult(
                parameter=p,
                start=p.value,
                value=float(best[k]),
                std=float(std[k]),
                at_lower=bool(best[k] - p.lower <= tol),
                at_upper=bool(p.upper - best[k] <= tol),
            )
        )
    statistics = goodness_of_fit(model, engine)
    outcome = FitResult(
        success=success,
        cancelled=cancelled,
        message=str(message),
        parameters=results,
        chi_squared=chi,
        reduced_chi_squared=reduced,
        n_points=problem.n_points,
        n_evaluations=state["nfev"],
        elapsed_seconds=time.perf_counter() - started,
        statistics=statistics,
        covariance=covariance,
    )
    if not apply:
        apply_values(model, parameters, start)
    return outcome


# -- Monte-Carlo uncertainty ------------------------------------------------------------------


def _pin_blas_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _monte_carlo_run(args) -> tuple[list[float] | None, str]:
    """One refit of re-drawn data. Top-level so worker processes can import it."""
    model, seed = args
    rng = np.random.default_rng(seed)
    for direction in model.directions:
        if direction.intensity is not None and direction.uncertainty is not None:
            sigma = np.asarray(direction.uncertainty, dtype=float)
            direction.intensity = np.asarray(direction.intensity, dtype=float) + np.where(
                np.isfinite(sigma), sigma, 0.0
            ) * rng.standard_normal(sigma.shape)
    model.settings.n_threads = 1  # the parallelism is over runs
    try:
        result = fit(model)
    except Exception as error:  # a failed run is reported, not fatal
        return None, str(error)
    return [p.value for p in result.parameters], result.message


def monte_carlo(
    model: AnisaModel,
    result: FitResult,
    n_runs: int = 10,
    *,
    seed: int = 0,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> MonteCarloResult:
    """Parameter uncertainty by refitting data re-drawn within their uncertainties.

    Each run adds ``sigma * N(0, 1)`` to the measured intensities and refits from the best-fit
    values in ``result``. ``model`` is not modified.

    Args:
        model: the fitted model (with data).
        result: the :class:`FitResult` of the fit that produced it; fixes which parameters vary.
        n_runs: number of refits.
        seed: seed of the random generator (each run gets its own child seed).
        workers: worker processes (``None``: one per CPU up to ``n_runs``; ``1``: in this
            process). Processes use the ``spawn`` start method with BLAS pinned to one thread, as
            in pyIrena's Modeling tool, and fall back to serial if they cannot start.
        progress: ``progress(runs_done, n_runs)``.
        cancel: polled between runs; returns what finished so far.

    Raises:
        ValueError: ``n_runs < 2``.
    """
    if n_runs < 2:
        raise ValueError("a Monte-Carlo estimate needs at least 2 runs")
    parameters = [p.parameter for p in result.parameters]
    seeds = np.random.SeedSequence(seed).generate_state(n_runs)
    jobs = []
    for s in seeds:
        trial = copy.deepcopy(model)
        apply_values(trial, parameters, result.values())
        jobs.append((trial, int(s)))

    n_workers = workers if workers is not None else min(os.cpu_count() or 1, n_runs)
    outputs: list[tuple[list[float] | None, str]] = []

    def run_serial(items):
        for job in items:
            if cancel is not None and cancel():
                break
            outputs.append(_monte_carlo_run(job))
            if progress is not None:
                progress(len(outputs), n_runs)

    if n_workers > 1:
        import multiprocessing

        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_pin_blas_threads,
            ) as pool:
                futures = [pool.submit(_monte_carlo_run, job) for job in jobs]
                for future in futures:
                    if cancel is not None and cancel():
                        for pending in futures:
                            pending.cancel()
                        break
                    outputs.append(future.result())
                    if progress is not None:
                        progress(len(outputs), n_runs)
        except (OSError, RuntimeError):
            outputs.clear()
            run_serial(jobs)
    else:
        run_serial(jobs)

    good = [values for values, _ in outputs if values is not None]
    failed = len(outputs) - len(good)
    if len(good) >= 2:
        values = np.array(good)
        std = values.std(axis=0, ddof=1)
    else:
        values = np.array(good).reshape(len(good), len(parameters))
        std = np.full(len(parameters), math.nan)
    return MonteCarloResult(values=values, std=std, n_requested=n_runs, n_failed=failed)
