"""
A catalogue of synthetic test systems with known parameters, from simple to complicated.

Real data have no known answer, so these are what a fit, a plot or a code change is checked
against. Every system is a complete :class:`~anisa.core.model.AnisaModel` (the truth), measured
along the same six directions of Q (four polar angles in the plane of the sample normal and two
azimuths in the substrate plane), with a stated contrast and volume fraction, plus a perturbed
copy with fit flags set (the starting point of a fit).

=================  ===============================================================================
``prolate``        needles, aspect 7, axes mildly aligned along the sample normal
``oblate``         platelets, aspect 1/6.7, axes mildly aligned along the sample normal
``prolate_oblate`` one prolate and one oblate population, strongly and weakly aligned, different
                   sizes and contrasts, the oblate axes lying in the substrate plane along x
``interference``   dense prolate population (12 vol %) with the Irena interference factor
``spheres``        isotropic polydisperse spheres (all directions must agree)
``many``           the ``prolate_oblate`` system measured along 12 directions (speed check)
=================  ===============================================================================

Each system is written by :func:`write_test_system` as a folder with the data files, the truth
(``NAME.anisa.json``), the start (``NAME_start.anisa.json``) and a plain-text ``TRUTH.txt``.
Command line::

    python -m anisa.testsets FOLDER [--system NAME ...] [--format h5|dat] [--noise 0.02]

Units are those of the package: Q in 1/A, I in 1/cm, radius in A, contrast in cm^-2, angles in
degrees. Spheroids have semi-axes (R, R, beta R): beta > 1 is prolate, beta < 1 oblate; the
alpha distribution is that of the spheroid *axis* measured from the sample normal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anisa import demo
from anisa.core.model import AnisaModel, Direction, Population
from anisa.core.orientation import AngularDistribution
from anisa.core.sizedist import SizeDistribution
from anisa.io.project import save_project

__all__ = [
    "TestSystem",
    "list_systems",
    "make_system",
    "truth_text",
    "write_test_system",
]

#: Directions shared by the systems: (label, alpha_Q, omega_Q) in degrees.
STANDARD_DIRECTIONS = (
    ("Q along normal (alpha 0)", 0.0, 0.0),
    ("alpha 30, omega 0", 30.0, 0.0),
    ("alpha 60, omega 0", 60.0, 0.0),
    ("in plane, omega 0 (x)", 90.0, 0.0),
    ("in plane, omega 45", 90.0, 45.0),
    ("in plane, omega 90 (y)", 90.0, 90.0),
)

BACKGROUND = 0.01  # cm^-1, small flat background in every direction


@dataclass
class TestSystem:
    """A named test system: the truth model without data, and how to measure it."""

    __test__ = False  # not a pytest class, despite the name

    name: str
    title: str
    description: str
    model: AnisaModel
    q_min: float
    q_max: float
    #: parameters the start model perturbs and fits, besides each population's own choice
    fit_paths: tuple[str, ...] = ("radius", "volume_fraction", "alpha_par2")


def _directions(extra_alpha: tuple[float, ...] = ()) -> list[Direction]:
    rows = list(STANDARD_DIRECTIONS) + [(f"alpha {a:g}, omega 0", a, 0.0) for a in extra_alpha]
    return [
        Direction(label=label, alpha_q=a, omega_q=w, background=BACKGROUND) for label, a, w in rows
    ]


def _prolate() -> TestSystem:
    population = Population(
        name="needles, aspect 7",
        radius=60.0,  # equatorial; the long semi-axis is 420 A
        beta=7.0,
        volume_fraction=0.01,
        delta_rho=1.0e11,
        alpha_dist=AngularDistribution(par1=0.0, par2=4.0),
    )
    return TestSystem(
        "prolate",
        "Prolate spheroids, mildly aligned along the normal",
        "One population, aspect 7 (equatorial radius 60 A, long semi-axis 420 A), volume "
        "fraction 1 %, contrast 1e11 cm^-2. The axes follow |cos alpha|^4 around the sample "
        "normal and are uniform in azimuth. Along the normal Q probes the length (intensity "
        "falls at lower Q), in the substrate plane the width, so the plane directions are "
        "brighter at high Q, and the three in-plane azimuths must agree.",
        AnisaModel(_directions(), [population]),
        2.0e-3,
        0.3,
    )


def _oblate() -> TestSystem:
    population = Population(
        name="platelets, aspect 1/6.7",
        radius=400.0,  # equatorial; the thin semi-axis is 60 A
        beta=0.15,
        volume_fraction=0.01,
        delta_rho=1.0e11,
        alpha_dist=AngularDistribution(par1=0.0, par2=4.0),
    )
    return TestSystem(
        "oblate",
        "Oblate spheroids, mildly aligned along the normal",
        "One population, aspect 1/6.7 (equatorial radius 400 A, thin semi-axis 60 A), volume "
        "fraction 1 %, contrast 1e11 cm^-2. The disk axes follow |cos alpha|^4 around the "
        "sample normal: the disks lie flat. Along the normal Q sees the thickness (a slow decay, "
        "bright at high Q), in plane the large diameter (a steep, early decay), the opposite of "
        "the prolate case.",
        AnisaModel(_directions(), [population]),
        1.0e-3,
        0.3,
    )


def _prolate_oblate(
    extra_alpha: tuple[float, ...] = (), name: str = "prolate_oblate"
) -> TestSystem:
    prolate = Population(
        name="needles, strongly aligned",
        radius=50.0,
        beta=8.0,  # long semi-axis 400 A
        volume_fraction=0.006,
        delta_rho=1.0e11,
        alpha_dist=AngularDistribution(par1=0.0, par2=30.0),
    )
    oblate = Population(
        name="platelets, weakly aligned",
        radius=450.0,
        beta=0.12,  # thin semi-axis 54 A
        volume_fraction=0.012,
        delta_rho=0.6e11,
        alpha_dist=AngularDistribution(par1=90.0, par2=3.0),
        omega_dist=AngularDistribution(par1=0.0, par2=3.0),
    )
    return TestSystem(
        name,
        "A prolate and an oblate population with different alignment",
        "Needles (aspect 8, equatorial radius 50 A, 0.6 vol %, contrast 1e11) strongly aligned "
        "along the normal (|cos alpha|^30), plus platelets (aspect 1/8.3, equatorial radius "
        "450 A, 1.2 vol %, contrast 0.6e11) whose axes lie in the substrate plane along x, only "
        "weakly aligned (|cos|^3 in both alpha and omega). The platelets dominate at low Q, the "
        "needles at high Q; the in-plane azimuths differ because the platelet axes prefer x.",
        AnisaModel(_directions(extra_alpha), [prolate, oblate]),
        1.0e-3,
        0.3,
    )


def _interference() -> TestSystem:
    population = Population(
        name="dense needles",
        radius=100.0,
        beta=5.0,  # long semi-axis 500 A
        volume_fraction=0.12,
        delta_rho=1.0e11,
        alpha_dist=AngularDistribution(par1=0.0, par2=6.0),
        use_interference=True,
        eta=450.0,
        pack=3.0,
    )
    return TestSystem(
        "interference",
        "Dense prolate population with interference",
        "Aspect 5 (equatorial radius 100 A), volume fraction 12 %, contrast 1e11 cm^-2, axes "
        "|cos alpha|^6 around the normal. The Irena interference factor with distance eta = "
        "450 A and packing 3 suppresses the low-Q intensity (S(0) = 1/(1+3) = 0.25) and adds a "
        "correlation hump near Q = 4.5/eta = 0.01 1/A. Without it the model is the "
        "dilute one, so a fit that ignores interference gets volume fraction and size wrong.",
        AnisaModel(_directions(), [population]),
        2.0e-3,
        0.3,
        fit_paths=("radius", "volume_fraction", "alpha_par2", "eta", "pack"),
    )


def _spheres() -> TestSystem:
    population = Population(
        name="polydisperse spheres",
        radius=150.0,
        beta=1.0,
        volume_fraction=0.02,
        delta_rho=1.0e11,
        size_dist=SizeDistribution("lognormal", {"min_size": 10.0, "sdeviation": 0.25}),
    )
    return TestSystem(
        "spheres",
        "Isotropic polydisperse spheres",
        "Spheres, mean radius 150 A with a Irena log-normal size distribution (min size 10 A, sdeviation 0.25), volume "
        "fraction 2 %, contrast 1e11 cm^-2. The scattering is isotropic: all six directions "
        "must agree within noise. A sanity check for size, volume fraction and contrast.",
        AnisaModel(_directions(), [population]),
        2.0e-3,
        0.3,
        fit_paths=("radius", "volume_fraction"),
    )


_BUILDERS = {
    "prolate": _prolate,
    "oblate": _oblate,
    "prolate_oblate": _prolate_oblate,
    "interference": _interference,
    "spheres": _spheres,
    "many": lambda: _many(),
}


def _many() -> TestSystem:
    system = _prolate_oblate(extra_alpha=(10.0, 20.0, 40.0, 50.0, 70.0, 80.0), name="many")
    system.title = "Twelve directions of Q (speed check)"
    system.description = (
        "The prolate/oblate system measured along twelve directions of Q: the standard six plus "
        "alpha_Q = 10, 20, 40, 50, 70 and 80 deg at omega_Q = 0. Meant for judging how the GUI "
        "and the fit behave as directions are added. " + system.description
    )
    return system


def list_systems() -> list[str]:
    """Names of the available systems."""
    return list(_BUILDERS)


def make_system(name: str) -> TestSystem:
    """A fresh copy of the named system (without data)."""
    try:
        return _BUILDERS[name]()
    except KeyError:
        raise ValueError(f"unknown test system {name!r}; choose from {list_systems()}") from None


def truth_text(system: TestSystem, relative_noise: float, seed: int) -> str:
    """Plain-text description of the truth: what was simulated and every parameter value."""
    lines = [system.title, "=" * len(system.title), "", system.description, ""]
    for index, pop in enumerate(system.model.populations, start=1):
        shape = "prolate" if pop.beta > 1 else "oblate" if pop.beta < 1 else "spherical"
        lines += [
            f"Population {index}: {pop.name} ({shape})",
            f"  Radius            {pop.radius:g} A (equatorial)",
            f"  Beta              {pop.beta:g}   (semi-axes R, R, beta R)",
            f"  Volume Fract      {pop.volume_fraction:g}",
            f"  Delta rho         {pop.delta_rho:g} cm^-2",
            f"  Size distribution {pop.size_dist.type} {dict(pop.size_dist.params)}",
            f"  P alpha           par1 {pop.alpha_dist.par1:g} deg, par2 {pop.alpha_dist.par2:g}, "
            f"par3 {pop.alpha_dist.par3:g}",
            f"  B omega           par1 {pop.omega_dist.par1:g} deg, par2 {pop.omega_dist.par2:g}, "
            f"par3 {pop.omega_dist.par3:g}",
        ]
        if pop.use_interference:
            lines.append(f"  Interference      eta {pop.eta:g} A, pack {pop.pack:g}")
        lines.append("")
    lines.append(f"Directions (alpha_Q, omega_Q in deg; each has background {BACKGROUND:g} 1/cm):")
    for d in system.model.directions:
        lines.append(f"  {d.label:<28} {d.alpha_q:6.1f} {d.omega_q:6.1f}")
    lines += [
        "",
        f"Data: Q {system.q_min:g} to {system.q_max:g} 1/A, 120 log-spaced points, relative "
        f"noise {relative_noise:g} (also the stored uncertainty), seed {seed}.",
        "The *_start.anisa.json project has radius, volume fraction and orientation sharpness "
        "(plus eta and pack where used) moved by up to 25 % and set to fit.",
        "",
        "Correct fit result: every fitted parameter returns to the values above within about "
        "two standard deviations, and chi^2/N is about 1.",
    ]
    return "\n".join(lines) + "\n"


def write_test_system(
    system: TestSystem | str,
    folder: str | Path,
    fmt: str = "h5",
    *,
    relative_noise: float = 0.02,
    seed: int = 0,
) -> Path:
    """Write one system into ``folder / NAME`` and return the path of its truth project.

    The folder gets the per-direction data files, ``NAME.anisa.json`` (truth),
    ``NAME_start.anisa.json`` (perturbed start, fit flags set) and ``TRUTH.txt``.
    """
    if isinstance(system, str):
        system = make_system(system)
    target = Path(folder) / system.name
    target.mkdir(parents=True, exist_ok=True)
    q = np.geomspace(system.q_min, system.q_max, 120)
    model = demo.synthesize(system.model, q, relative_noise=relative_noise, seed=seed)
    demo.write_files(model, target, fmt, prefix=system.name)
    truth = save_project(model, target / f"{system.name}.anisa.json")
    start = demo.perturb_model(model, seed=seed + 1, paths=system.fit_paths)
    save_project(start, target / f"{system.name}_start.anisa.json")
    (target / "TRUTH.txt").write_text(truth_text(system, relative_noise, seed), encoding="utf-8")
    return truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write synthetic test systems with known truth.")
    parser.add_argument("folder", help="output folder; one subfolder per system")
    parser.add_argument(
        "--system", action="append", choices=list_systems(), help="system (default: all)"
    )
    parser.add_argument("--format", choices=("h5", "dat"), default="h5")
    parser.add_argument("--noise", type=float, default=0.02, help="relative noise (default 0.02)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for name in args.system or list_systems():
        path = write_test_system(
            name, args.folder, args.format, relative_noise=args.noise, seed=args.seed
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
