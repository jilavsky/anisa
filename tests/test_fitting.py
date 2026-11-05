"""Least-squares fitting: parameter selection, recovery of known parameters, bounds, revert,
cancel, uncertainties."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from anisa import demo
from anisa.core.engine import Engine
from anisa.core.fitting import (
    active_parameters,
    apply_values,
    fit,
    goodness_of_fit,
    monte_carlo,
    revert,
)
from anisa.core.model import AnisaModel, Population
from anisa.core.orientation import AngularDistribution


@pytest.fixture(scope="module")
def truth():
    return demo.synthesize(demo.demo_model(), seed=0)


@pytest.fixture(scope="module")
def fitted(truth):
    """The demo fit from a perturbed start, done once and shared (it takes about two seconds)."""
    model = demo.perturb_model(truth, seed=1, fraction=0.25)
    result = fit(model)
    return model, result


# -- which parameters vary --------------------------------------------------------------------


def test_active_parameters_follow_the_flags_in_a_fixed_order():
    model = demo.demo_model()
    assert active_parameters(model) == []
    model.populations[1].fit_spec("radius").fit = True
    model.populations[0].fit_spec("beta").fit = True
    model.directions[2].fit_spec().fit = True
    params = active_parameters(model)
    assert [(p.owner, p.index, p.path) for p in params] == [
        ("population", 0, "beta"),
        ("population", 1, "radius"),
        ("direction", 2, "background"),
    ]
    assert params[1].label == "Pop 2 (interlamellar pores): radius"
    assert params[2].label == "Dir 3 (in plane, y): background"
    assert params[1].value == model.populations[1].radius


def test_limits_default_to_the_table_and_none_means_infinite():
    model = demo.demo_model()
    pop = model.populations[0]
    pop.fit_spec("radius").fit = True
    pop.fit_spec("volume_fraction").fit = True
    pop.fit_spec("volume_fraction").max = None
    radius, vf = active_parameters(model)
    assert (radius.lower, radius.upper) == (10.0, 1.0e7)
    assert vf.lower == 1e-6 and vf.upper == math.inf
    model.directions[0].fit_spec().fit = True
    assert active_parameters(model)[-1].lower == -math.inf


def test_parameters_that_do_not_apply_are_skipped():
    pop = Population(
        alpha_dist=AngularDistribution(mode="table", table=(1.0, 2.0, 1.0)), use_interference=False
    )
    for path in ("alpha_par1", "alpha_par2", "omega_par2", "eta", "pack", "radius"):
        pop.fit_spec(path).fit = True
    names = [p.path for p in active_parameters(AnisaModel([], [pop]))]
    assert names == ["radius", "omega_par2"]  # table alpha and no interference are skipped
    pop.use_interference = True
    assert {"eta", "pack"} <= {p.path for p in active_parameters(AnisaModel([], [pop]))}


def test_each_population_uses_its_own_orientation_mode():
    """Igor read population 1's 'use parameters' flags for every population (a bug)."""
    parametric = Population(alpha_dist=AngularDistribution(par1=0.0, par2=10.0))
    table = Population(alpha_dist=AngularDistribution(mode="table", table=(1.0, 3.0, 1.0)))
    for pop in (parametric, table):
        pop.fit_spec("alpha_par2").fit = True
    params = active_parameters(AnisaModel([], [parametric, table]))
    assert [(p.index, p.path) for p in params] == [(0, "alpha_par2")]
    # and in the opposite order the answer follows the population, not its position
    params = active_parameters(AnisaModel([], [table, parametric]))
    assert [(p.index, p.path) for p in params] == [(1, "alpha_par2")]


# -- recovering known parameters --------------------------------------------------------------


def test_fit_recovers_the_true_parameters_from_a_perturbed_start(truth, fitted):
    model, result = fitted
    assert result.success and not result.cancelled
    assert result.n_parameters == 8 and result.n_points == 480
    # data noise equals the stored uncertainty, so a correct fit has chi^2 per dof ~ 1
    assert 0.85 < result.reduced_chi_squared < 1.2
    for p in result.parameters:
        true_value = truth.populations[p.parameter.index].get_parameter(p.parameter.path)
        assert p.std > 0 and np.isfinite(p.std)
        assert abs(p.value - true_value) < 3.5 * p.std, p.parameter.label
        assert abs(p.value / true_value - 1.0) < 0.1, p.parameter.label
        assert not p.at_lower and not p.at_upper
    # the model itself carries the result
    assert model.populations[1].radius == pytest.approx(result.parameters[3].value)
    assert result.statistics.reduced < 1.2


def test_fit_improves_chi_squared_enormously(truth, fitted):
    _, result = fitted
    start = demo.perturb_model(truth, seed=1, fraction=0.25)
    assert goodness_of_fit(start).chi_squared > 50 * result.chi_squared


def test_fit_recovers_the_background(truth):
    model = copy.deepcopy(truth)
    for direction in model.directions:
        direction.fit_spec().fit = True
        direction.background = 0.4  # truth is 0.05
    result = fit(model)
    assert result.success
    for direction, p in zip(model.directions, result.parameters, strict=True):
        assert abs(direction.background - 0.05) < 4 * p.std
        assert p.parameter.path == "background"
    assert 0.85 < result.reduced_chi_squared < 1.2


def test_per_direction_statistics_add_up(fitted):
    model, result = fitted
    stats = result.statistics
    assert sum(c for c in stats.direction_chi_squared if c is not None) == pytest.approx(
        stats.chi_squared
    )
    assert sum(stats.direction_points) == stats.n_points == result.n_points
    assert all(0.7 < stats.direction_reduced(i) < 1.4 for i in range(4))
    assert stats.chi_squared == pytest.approx(result.chi_squared)


def test_directions_not_in_use_or_without_data_are_left_out(truth):
    model = copy.deepcopy(truth)
    model.directions[1].use = False
    model.directions[3].q = model.directions[3].intensity = model.directions[3].uncertainty = None
    stats = goodness_of_fit(model)
    assert stats.direction_chi_squared[1] is None and stats.direction_chi_squared[3] is None
    assert stats.direction_points == (120, 0, 120, 0)
    assert stats.n_points == 240


# -- bounds, revert, cancel -------------------------------------------------------------------


def test_limits_are_respected_and_reported(truth):
    model = copy.deepcopy(truth)
    pop = model.populations[1]
    pop.radius = 3500.0
    spec = pop.fit_spec("radius")
    spec.fit, spec.min, spec.max = True, 3000.0, 3800.0  # the truth, 4000, is out of reach
    result = fit(model)
    (p,) = result.parameters
    assert p.at_upper and not p.at_lower
    assert 3000.0 <= pop.radius <= 3800.0 + 1e-6
    assert pop.radius == pytest.approx(3800.0)


def test_a_start_outside_the_limits_is_pulled_inside(truth):
    model = copy.deepcopy(truth)
    pop = model.populations[0]
    spec = pop.fit_spec("volume_fraction")
    spec.fit, spec.min, spec.max = True, 0.005, 0.02
    pop.volume_fraction = 0.5  # far outside
    result = fit(model)
    assert 0.005 <= pop.volume_fraction <= 0.02 and result.n_evaluations > 1


def test_revert_restores_the_start_values(truth):
    model = demo.perturb_model(truth, seed=2, fraction=0.2)
    before = model.to_dict()
    result = fit(model)
    assert model.to_dict() != before
    revert(model, result)
    assert model.to_dict() == before


def test_apply_false_leaves_the_model_untouched(truth):
    model = demo.perturb_model(truth, seed=2, fraction=0.2)
    before = model.to_dict()
    result = fit(model, apply=False)
    assert model.to_dict() == before
    assert result.reduced_chi_squared < 2  # but the fit itself ran
    apply_values(model, [p.parameter for p in result.parameters], result.values())
    assert goodness_of_fit(model).reduced < 2


def test_cancel_returns_the_best_parameters_found_so_far(truth):
    model = demo.perturb_model(truth, seed=1, fraction=0.25)
    start_chi = goodness_of_fit(model).chi_squared
    seen = []

    def cancel():
        return len(seen) >= 25

    result = fit(model, progress=lambda n, chi: seen.append((n, chi)), cancel=cancel)
    assert result.cancelled and not result.success
    assert result.n_evaluations == 25
    assert [n for n, _ in seen] == list(range(1, 26))  # progress after every evaluation
    assert result.chi_squared < start_chi  # best so far, not the last trial
    assert goodness_of_fit(model).chi_squared == pytest.approx(result.chi_squared)
    assert result.covariance is None


def test_max_nfev_limits_the_work(truth):
    model = demo.perturb_model(truth, seed=1, fraction=0.25)
    result = fit(model, max_nfev=6)
    assert result.n_evaluations <= 6 * (1 + result.n_parameters)  # jacobian columns add calls
    assert not result.success


def test_problems_are_reported_clearly(truth):
    with pytest.raises(ValueError, match="no parameter is marked"):
        fit(copy.deepcopy(truth))
    flagged = copy.deepcopy(truth)
    flagged.populations[0].fit_spec("radius").fit = True
    for d in flagged.directions:
        d.q = d.intensity = d.uncertainty = None
    with pytest.raises(ValueError, match="no data"):
        fit(flagged)
    bad = copy.deepcopy(truth)
    spec = bad.populations[0].fit_spec("radius")
    spec.fit, spec.min, spec.max = True, 500.0, 500.0
    with pytest.raises(ValueError, match="not below"):
        fit(bad)


def test_a_failure_leaves_the_model_at_its_start_values(truth, monkeypatch):
    model = demo.perturb_model(truth, seed=1, fraction=0.25)
    before = model.to_dict()
    calls = {"n": 0}
    original = Engine.evaluate

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 12:
            raise RuntimeError("boom")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Engine, "evaluate", flaky)
    with pytest.raises(RuntimeError):
        fit(model)
    assert model.to_dict() == before


# -- uncertainties ----------------------------------------------------------------------------


def test_covariance_is_consistent_with_the_reported_errors(fitted):
    _, result = fitted
    cov = result.covariance
    assert cov.shape == (8, 8)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-8, atol=0)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-12 * np.abs(cov).max())
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), [p.std for p in result.parameters])


def test_errors_do_not_depend_on_the_absolute_scale_of_the_uncertainties(truth):
    """Scaling sigma changes chi^2 but must not change the parameter errors (the covariance is
    scaled by the reduced chi^2), so a pessimistic or optimistic sigma is harmless."""
    a = demo.perturb_model(truth, seed=3, fraction=0.1)
    b = copy.deepcopy(a)
    for d in b.directions:
        d.uncertainty = 3.0 * d.uncertainty
    ra, rb = fit(a), fit(b)
    assert rb.reduced_chi_squared == pytest.approx(ra.reduced_chi_squared / 9.0, rel=1e-2)
    for pa, pb in zip(ra.parameters, rb.parameters, strict=True):
        assert pb.std == pytest.approx(pa.std, rel=2e-2)
        assert pb.value == pytest.approx(pa.value, rel=1e-3)


def test_monte_carlo_agrees_with_the_covariance(fitted):
    model, result = fitted
    before = model.to_dict()
    mc = monte_carlo(model, result, n_runs=8, seed=5, workers=1)
    assert model.to_dict() == before  # the model is not modified
    assert mc.values.shape == (8, 8) and mc.n_failed == 0
    ratio = mc.std / np.array([p.std for p in result.parameters])
    # eight runs estimate a standard deviation to ~25 %: this only catches gross disagreement
    assert np.all((ratio > 0.25) & (ratio < 4.0)), ratio


def test_monte_carlo_is_reproducible_for_a_seed(fitted):
    model, result = fitted
    a = monte_carlo(model, result, n_runs=2, seed=5, workers=1)
    b = monte_carlo(model, result, n_runs=2, seed=5, workers=1)
    c = monte_carlo(model, result, n_runs=2, seed=6, workers=1)
    np.testing.assert_array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


def test_monte_carlo_in_worker_processes(fitted):
    model, result = fitted
    mc = monte_carlo(model, result, n_runs=2, seed=1, workers=2)
    assert mc.values.shape == (2, 8) and mc.n_failed == 0
    serial = monte_carlo(model, result, n_runs=2, seed=1, workers=1)
    np.testing.assert_allclose(mc.values, serial.values, rtol=1e-6)  # same seeds, same answers


def test_monte_carlo_progress_cancel_and_validation(fitted):
    model, result = fitted
    done = []
    monte_carlo(
        model, result, n_runs=3, workers=1, progress=lambda n, total: done.append((n, total))
    )
    assert done == [(1, 3), (2, 3), (3, 3)]
    partial = monte_carlo(model, result, n_runs=4, workers=1, cancel=lambda: True)
    assert partial.values.shape[0] == 0 and np.all(np.isnan(partial.std))
    with pytest.raises(ValueError, match="at least 2"):
        monte_carlo(model, result, n_runs=1)
