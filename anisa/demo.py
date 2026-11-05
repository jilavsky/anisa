"""
Synthetic multi-direction data with known parameters, for testing without real data.

The demo model follows the example in Allen's notes: a thermal-spray coating with three void
populations,

* **intrasplat cracks**: flat (oblate) cracks standing perpendicular to the substrate, so the
  disk axes lie in the substrate plane (alpha ~ 90 deg);
* **interlamellar pores**: flat pores parallel to the substrate, so the disk axes point along
  the substrate normal (alpha ~ 0 deg);
* **globular voids**: spheres with a size distribution, which scatter isotropically,

measured along four directions of Q: along the normal, in the substrate plane along x and y, and
tilted by 45 degrees. A fit can then be checked against the truth.

Command line::

    python -m anisa.demo OUTPUT_FOLDER [--format h5|dat] [--noise 0.03] [--seed 0]

writes one data file per direction, ``demo.anisa.json`` (the true model) and
``demo_start.anisa.json`` (perturbed starting values with fit flags set).
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
from pyirena.io import discover_scattering
from pyirena.io.nxcansas_unified import create_nxcansas_file

from anisa.core.engine import Engine
from anisa.core.model import AnisaModel, DataSource, Direction, Population
from anisa.core.orientation import AngularDistribution
from anisa.core.sizedist import SizeDistribution
from anisa.io.data import content_fingerprint
from anisa.io.project import save_project

__all__ = [
    "demo_model",
    "perturb_model",
    "synthesize",
    "write_demo_project",
    "write_files",
]


def demo_model() -> AnisaModel:
    """The three-population, four-direction demo model (no data attached yet)."""
    delta_rho = 2.5e11  # cm^-2, void in a ceramic
    populations = [
        Population(
            name="intrasplat cracks",
            radius=2500.0,
            beta=0.08,
            volume_fraction=0.012,
            delta_rho=delta_rho,
            alpha_dist=AngularDistribution(par1=90.0, par2=25.0),
        ),
        Population(
            name="interlamellar pores",
            radius=4000.0,
            beta=0.06,
            volume_fraction=0.02,
            delta_rho=delta_rho,
            alpha_dist=AngularDistribution(par1=0.0, par2=30.0),
        ),
        Population(
            name="globular voids",
            radius=1500.0,
            beta=1.0,
            volume_fraction=0.01,
            delta_rho=delta_rho,
            size_dist=SizeDistribution("gauss", {"width": 300.0}),
        ),
    ]
    directions = [
        Direction(label="along normal", alpha_q=0.0, omega_q=0.0, background=0.05),
        Direction(label="in plane, x", alpha_q=90.0, omega_q=0.0, background=0.05),
        Direction(label="in plane, y", alpha_q=90.0, omega_q=90.0, background=0.05),
        Direction(label="tilted 45 deg", alpha_q=45.0, omega_q=0.0, background=0.05),
    ]
    return AnisaModel(directions, populations)


def synthesize(
    model: AnisaModel,
    q: np.ndarray | None = None,
    *,
    relative_noise: float = 0.03,
    noise_floor: float = 0.0,
    seed: int = 0,
    engine: Engine | None = None,
) -> AnisaModel:
    """Attach noisy synthetic data to every direction of ``model`` (in place) and return it.

    The data are the model intensity (background included) plus Gaussian noise of standard
    deviation ``relative_noise * I + noise_floor`` [1/cm], which is also stored as the
    uncertainty. Reproducible for a given ``seed``.

    Args:
        model: model with the true parameters.
        q: Q grid [1/A], shared by all directions; default 120 log-spaced points, 1e-4 to 0.08.
        relative_noise: relative standard deviation of the noise (and of the uncertainty).
        noise_floor: absolute noise added in quadrature-free form [1/cm].
        seed: seed of the random generator.
        engine: engine to use (a fresh one by default).
    """
    if q is None:
        q = np.geomspace(1.0e-4, 0.08, 120)
    q = np.asarray(q, dtype=float)
    result = (engine or Engine()).evaluate(model, [q] * len(model.directions))
    rng = np.random.default_rng(seed)
    for direction, calculated in zip(model.directions, result.directions, strict=True):
        sigma = relative_noise * calculated.intensity + noise_floor
        direction.q = q.copy()
        direction.intensity = calculated.intensity + sigma * rng.standard_normal(q.size)
        direction.uncertainty = sigma
        direction.source = None
    return model


def write_files(model: AnisaModel, folder: str | Path, fmt: str = "h5") -> list[Path]:
    """Write each direction's data to its own file and point ``direction.source`` at it.

    Args:
        model: model whose directions carry data (see :func:`synthesize`).
        folder: output folder, created if needed.
        fmt: ``"h5"`` (NXcanSAS, written with pyIrena's writer) or ``"dat"`` (text: Q, I, dI).

    Returns:
        The files written, in direction order.
    """
    if fmt not in ("h5", "dat"):
        raise ValueError("fmt must be 'h5' or 'dat'")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, direction in enumerate(model.directions, start=1):
        if direction.q is None or direction.intensity is None or direction.uncertainty is None:
            raise ValueError(f"direction {index} has no data; call synthesize() first")
        path = folder / f"demo_dir{index:02d}.{fmt}"
        if fmt == "h5":
            create_nxcansas_file(
                path,
                direction.q,
                direction.intensity,
                direction.uncertainty,
                sample_name=f"dir{index:02d}",
                metadata={
                    "title": f"Anisa synthetic data, {direction.label}",
                    "instrument.wavelength": float(model.settings.wavelength),
                    "notes.alpha_q_deg": float(direction.alpha_q),
                    "notes.omega_q_deg": float(direction.omega_q),
                },
            )
        else:
            header = (
                f"Anisa synthetic data, {direction.label}\n"
                f"alpha_q = {direction.alpha_q} deg, omega_q = {direction.omega_q} deg, "
                f"written {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                "Q [1/A]   I [1/cm]   dI [1/cm]"
            )
            np.savetxt(
                path,
                np.column_stack((direction.q, direction.intensity, direction.uncertainty)),
                header=header,
                fmt="%.10e",
            )
        (location,) = discover_scattering(path)
        direction.source = DataSource(
            path=str(path.resolve()),
            internal_path=location.internal_path,
            variant=location.variant,
            fingerprint=content_fingerprint(path),
        )
        paths.append(path)
    return paths


def perturb_model(model: AnisaModel, seed: int = 1, fraction: float = 0.25) -> AnisaModel:
    """A copy of ``model`` with its main parameters moved away from the truth and marked to fit.

    Radius, volume fraction and the alpha-distribution sharpness of every population are
    multiplied by a random factor within ``1 +- fraction``; their ``fit`` flags are set. Data and
    sources are kept. Used to check that a fit recovers the true parameters.
    """
    rng = np.random.default_rng(seed)
    start = copy.deepcopy(model)
    for population in start.populations:
        paths = ["radius", "volume_fraction"]
        if population.alpha_dist.par2 > 0:
            paths.append("alpha_par2")
        for path in paths:
            factor = 1.0 + fraction * rng.uniform(-1.0, 1.0)
            population.set_parameter(path, population.get_parameter(path) * factor)
            population.fit_spec(path).fit = True
    return start


def write_demo_project(
    folder: str | Path,
    fmt: str = "h5",
    *,
    relative_noise: float = 0.03,
    seed: int = 0,
) -> Path:
    """Write a complete demo: data files, the true model and a perturbed starting model.

    Returns the path of ``demo.anisa.json``; ``demo_start.anisa.json`` is written next to it.
    """
    folder = Path(folder)
    model = synthesize(demo_model(), relative_noise=relative_noise, seed=seed)
    write_files(model, folder, fmt)
    truth = save_project(model, folder / "demo.anisa.json")
    save_project(perturb_model(model, seed=seed + 1), folder / "demo_start.anisa.json")
    return truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write synthetic anisotropic SAS demo data.")
    parser.add_argument("folder", help="output folder (created if needed)")
    parser.add_argument("--format", choices=("h5", "dat"), default="h5")
    parser.add_argument("--noise", type=float, default=0.03, help="relative noise (default 0.03)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    project = write_demo_project(
        args.folder, args.format, relative_noise=args.noise, seed=args.seed
    )
    print(f"wrote {project} and demo_start.anisa.json with data files in {Path(args.folder)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
