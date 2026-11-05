"""
Convergence study for the Anisa engine (PLAN.md section 9, item 4).

Compares the engine with an independent brute-force orientation sum (a dense mid-point rule in
(cos alpha, omega), evaluating the sphere amplitude directly at every node; no K binning, no
shared code with the engine except the orientation distribution). Reports, for several shapes,
sizes and orientation distributions, how the maximum and median relative error behave as the
engine's grid settings are refined. Used to choose the defaults in ``CalcSettings`` and the
grid-resolution constants in ``anisa.core.engine``.

Run:  python benchmarks/convergence.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy import special

from anisa.core import engine as eng
from anisa.core.model import AnisaModel, CalcSettings, Direction, Population
from anisa.core.orientation import AngularDistribution


def brute_force(pop: Population, direction: Direction, q: np.ndarray, na: int, no: int):
    """Independent reference for formula 4c, monodisperse."""
    c = (np.arange(na) + 0.5) / na  # mid-point rule in cos(alpha)
    alpha = np.arccos(c)
    omega = (np.arange(no) + 0.5) * 2.0 * np.pi / no
    w = np.outer(pop.alpha_dist.alpha_values(alpha), pop.omega_dist.omega_values(omega))
    w /= w.sum()
    aq, oq = np.radians(direction.alpha_q), np.radians(direction.omega_q)
    x = np.sin(aq) * np.sin(alpha)[:, None] * np.cos(omega - oq)[None, :] + np.cos(aq) * c[:, None]
    k = np.sqrt(1.0 + (pop.beta**2 - 1.0) * x * x).ravel()
    wf = w.ravel()
    volume = 4.0 / 3.0 * np.pi * pop.beta * pop.radius**3
    out = np.empty(q.size)
    for i, qi in enumerate(q):
        arg = qi * pop.radius * k
        amp = 3.0 * special.spherical_jn(1, arg) / arg
        out[i] = np.sum(wf * amp * amp)
    return 1e8 * pop.volume_fraction * volume * (pop.delta_rho * 1e-16) ** 2 * out


def run(pop, direction, q, **settings):
    model = AnisaModel([direction], [pop], CalcSettings(**settings))
    return eng.Engine().evaluate(model, [q]).directions[0].intensity


def study(quick: bool) -> None:
    cases = []
    for beta, radius, par2 in itertools.product((0.1, 0.4, 2.5), (300.0, 3000.0), (0.0, 6.0, 40.0)):
        if quick and (beta, radius, par2) not in ((0.1, 3000.0, 40.0), (2.5, 300.0, 6.0)):
            continue
        pop = Population(
            radius=radius,
            beta=beta,
            volume_fraction=0.05,
            alpha_dist=AngularDistribution(par1=25.0, par2=par2, par3=0.02),
            omega_dist=AngularDistribution(par1=80.0, par2=min(par2, 10.0), par3=0.05),
        )
        cases.append((pop, Direction(alpha_q=53.0, omega_q=20.0)))

    q = np.geomspace(2e-4, 0.12, 60)
    print("Reference: brute force, 1200 x 2400 orientation nodes, no binning.")
    print("Error = relative difference; 'max' is the worst Q point (deep minima inflate it).\n")
    header = f"{'beta':>5} {'R':>6} {'par2':>5} {'span':>6} | " + " | ".join(
        f"{label:^15}" for label in ("default", "ppp 24", "grid x0.5", "grid x2")
    )
    print(header)
    print("-" * len(header))
    for pop, direction in cases:
        ref = brute_force(pop, direction, q, 1200, 2400)
        span = q.max() * pop.radius * abs(pop.beta - 1.0)
        cells = []
        for _label, kwargs, factors in (
            ("default", {}, (1.0, 2.0)),
            ("ppp 24", {"points_per_period": 24}, (1.0, 2.0)),
            ("grid x0.5", {"n_alpha": 32, "n_omega": 64}, (0.5, 1.0)),
            ("grid x2", {"n_alpha": 128, "n_omega": 256}, (2.0, 4.0)),
        ):
            eng.ALPHA_NODES_PER_SPAN, eng.OMEGA_NODES_PER_SPAN = factors
            err = np.abs(run(pop, direction, q, **kwargs) - ref) / ref
            cells.append(f"{err.max():7.1e}/{np.median(err):7.1e}")
        print(
            f"{pop.beta:5.2f} {pop.radius:6.0f} {pop.alpha_dist.par2:5.0f} {span:6.0f} | "
            + " | ".join(cells)
        )
    print("\ncells show max/median relative error")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="two cases only")
    study(parser.parse_args().quick)
