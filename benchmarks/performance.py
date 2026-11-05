"""
Timing of the Anisa engine for the scenarios in PLAN.md section 3.

Run:  python benchmarks/performance.py
Targets (formula 4c, default grids, this reference is a laptop-class machine):
  1 direction x 1 population x 300 Q points ........ <= 10 ms
  15 directions x 5 populations x 300 Q points ..... < 1 s per full evaluation
"""

from __future__ import annotations

import os
import time

import numpy as np

from anisa.core.engine import Engine
from anisa.core.model import AnisaModel, CalcSettings, Direction, Population
from anisa.core.orientation import AngularDistribution
from anisa.core.sizedist import SizeDistribution


def make_model(
    n_dir: int, n_pop: int, size_dist: str, radius_scale: float = 1.0, **settings
) -> AnisaModel:
    """``radius_scale`` < 1 gives smaller particles (smaller form-factor phase span at a given Q)."""
    rng = np.random.default_rng(1)
    pops = []
    for i in range(n_pop):
        radius = radius_scale * (500.0 + 400.0 * i)
        dist = (
            SizeDistribution("gauss", {"width": 0.15 * radius})
            if size_dist == "gauss"
            else SizeDistribution()
        )
        pops.append(
            Population(
                radius=radius,
                beta=[0.15, 0.4, 2.5, 5.0, 0.8][i % 5],
                volume_fraction=0.02,
                size_dist=dist,
                alpha_dist=AngularDistribution(par1=rng.uniform(0, 90), par2=rng.uniform(2, 20)),
                omega_dist=AngularDistribution(par1=rng.uniform(0, 360), par2=rng.uniform(0, 6)),
            )
        )
    dirs = [
        Direction(alpha_q=a, omega_q=o)
        for a, o in zip(np.linspace(0, 90, n_dir), np.linspace(0, 330, n_dir), strict=True)
    ]
    return AnisaModel(dirs, pops, CalcSettings(**settings))


def time_it(model: AnisaModel, q: np.ndarray, repeat: int = 3) -> tuple[float, float]:
    """Return (cold, warm) seconds: first evaluation, and a re-evaluation of the same model."""
    engine = Engine()
    qs = [q] * len(model.directions)
    t0 = time.perf_counter()
    engine.evaluate(model, qs)
    cold = time.perf_counter() - t0
    warm = min(_t(engine, model, qs) for _ in range(repeat))
    return cold, warm


def _t(engine, model, qs):
    t0 = time.perf_counter()
    engine.evaluate(model, qs)
    return time.perf_counter() - t0


def volume_fraction_change(model: AnisaModel, q: np.ndarray) -> float:
    """Seconds to re-evaluate after changing only one volume fraction (what a fit step does)."""
    engine = Engine()
    qs = [q] * len(model.directions)
    engine.evaluate(model, qs)
    model.populations[0].volume_fraction *= 1.1
    return _t(engine, model, qs)


if __name__ == "__main__":
    q = np.geomspace(1e-4, 0.1, 300)
    print(f"CPU count: {os.cpu_count()}, Q points: {q.size}\n")
    print(f"{'scenario':<52} {'cold':>9} {'cached':>9}")
    # radius_scale 0.4: R = 200 ... 720 A (phase span <= ~300 at Q = 0.1); 1.0: R up to 2100 A,
    # a deliberately harsh case (span up to ~840, orientation grids of ~1.4 M nodes).
    scenarios = [
        ("1 dir x 1 pop, monodisperse", 1, 1, "none", 1.0, {}),
        ("1 dir x 1 pop, Gauss size dist", 1, 1, "gauss", 1.0, {}),
        ("6 dir x 3 pop, Gauss, typical sizes", 6, 3, "gauss", 0.4, {}),
        ("15 dir x 5 pop, Gauss, typical sizes", 15, 5, "gauss", 0.4, {}),
        ("15 dir x 5 pop, Gauss, typical sizes, 1 thread", 15, 5, "gauss", 0.4, {"n_threads": 1}),
        ("15 dir x 5 pop, monodisperse, large sizes", 15, 5, "none", 1.0, {}),
        ("15 dir x 5 pop, Gauss, large sizes", 15, 5, "gauss", 1.0, {}),
        ("15 dir x 5 pop, Gauss, large sizes, 1 thread", 15, 5, "gauss", 1.0, {"n_threads": 1}),
    ]
    for label, n_dir, n_pop, dist, scale, kw in scenarios:
        cold, warm = time_it(make_model(n_dir, n_pop, dist, scale, **kw), q)
        print(f"{label:<52} {cold * 1e3:7.1f}ms {warm * 1e3:7.2f}ms")
    model = make_model(15, 5, "gauss", 0.4)
    print(
        f"\n15 x 5 Gauss typical, after changing one volume fraction: "
        f"{volume_fraction_change(model, q) * 1e3:.1f} ms"
    )
