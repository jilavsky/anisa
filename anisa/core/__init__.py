"""
Numerical model for anisotropic SAS. numpy/scipy only (plus pyIrena's size distributions);
no Qt, no matplotlib.

Typical use::

    import numpy as np
    from anisa.core import AnisaModel, Direction, Population, Engine
    from anisa.core.orientation import AngularDistribution

    pop = Population(radius=800.0, beta=0.2, volume_fraction=0.03, delta_rho=2e11,
                     alpha_dist=AngularDistribution(par1=0.0, par2=12.0))
    model = AnisaModel([Direction(alpha_q=0.0), Direction(alpha_q=90.0)], [pop])
    q = np.geomspace(1e-4, 0.1, 200)
    result = Engine().evaluate(model, [q, q])
    result.directions[0].intensity        # I(Q) in 1/cm along the first direction

Fitting (parameters with ``fit_spec(path).fit = True``; data in ``Direction.q`` etc.)::

    from anisa.core import fit
    result = fit(model)                   # model now holds the best parameters
    result.reduced_chi_squared, [(p.parameter.label, p.value, p.std) for p in result.parameters]
"""

from anisa.core.engine import DirectionResult, Engine, ModelResult, calculate
from anisa.core.fitting import (
    FitResult,
    FitStatistics,
    fit,
    goodness_of_fit,
    monte_carlo,
    revert,
)
from anisa.core.model import (
    AnisaModel,
    CalcSettings,
    Direction,
    FitSpec,
    Population,
)
from anisa.core.orientation import AngularDistribution, OrientationDistribution
from anisa.core.sizedist import SizeDistribution

__all__ = [
    "AngularDistribution",
    "AnisaModel",
    "CalcSettings",
    "Direction",
    "DirectionResult",
    "Engine",
    "FitResult",
    "FitSpec",
    "FitStatistics",
    "ModelResult",
    "OrientationDistribution",
    "Population",
    "SizeDistribution",
    "calculate",
    "fit",
    "goodness_of_fit",
    "monte_carlo",
    "revert",
]
