"""
Command line: ``anisa COMMAND``. A thin layer over :mod:`anisa.api`; no Qt.

====================  =====================================================================
``anisa info FILE``     populations, directions, fit flags of a project or results file
``anisa calc FILE``     calculate the model, report chi-squared (``--results`` writes HDF5)
``anisa fit FILE``      fit the flagged parameters, write ``NAME.anisa.h5`` (``--mc N``)
``anisa anisotropy``    I(angle) scan of a saved scan, or ``--sweep/--fixed/--q`` (``--csv``)
``anisa export-bernardyn``  data and model of all directions as a Bernardyn package
``anisa testdata DIR``  write the synthetic test systems (``--system NAME``, ``--list``)
====================  =====================================================================

FILE is a project (``*.anisa.json``, the file the GUI saves) or a results file
(``*.anisa.h5``). Each command prints a JSON document to stdout (``--text`` prints a short
human-readable report for ``fit``); progress and errors go to stderr. Exit status: 0 success,
1 the fit did not converge, 2 bad input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from anisa import __version__, api

__all__ = ["main"]


def _emit(document) -> None:
    json.dump(document, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")


def _fit_text(report: dict) -> str:
    state = "converged" if report["success"] else "NOT converged"
    lines = [
        f"Fit {state}: {report['message']}",
        f"chi2/dof = {report['reduced_chi_squared']:.4g}, {report['n_points']} points, "
        f"{report['n_evaluations']} evaluations, {report['elapsed_seconds']:.1f} s",
        "",
    ]
    width = max((len(p["label"]) for p in report["parameters"]), default=0)
    for p in report["parameters"]:
        error = "" if p["std"] is None else f" +- {p['std']:.3g}"
        lines.append(f"  {p['label']:<{width}}  {p['start']:.6g} -> {p['value']:.6g}{error}")
    lines += ["", f"results: {report['results_file']}"]
    return "\n".join(lines)


def _write_csv(path: str, scan: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_deg"] + [f"I(Q={q:g})" for q in scan["q_values"]])
        for k, angle in enumerate(scan["angles"]):
            writer.writerow([angle] + [row[k] for row in scan["intensity"]])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anisa", description="Small-angle scattering from anisotropic spheroid populations."
    )
    parser.add_argument("--version", action="version", version=f"anisa {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="describe a project or results file")
    info.add_argument("file")

    calc = commands.add_parser("calc", help="calculate the model and report chi-squared")
    calc.add_argument("file")
    calc.add_argument("--results", help="also write a results file")

    fit = commands.add_parser("fit", help="fit the parameters flagged for fitting")
    fit.add_argument("file")
    fit.add_argument("--results", help="results file (default: NAME.anisa.h5 next to FILE)")
    fit.add_argument("--mc", type=int, default=0, metavar="N", help="Monte-Carlo runs (>= 2)")
    fit.add_argument("--max-nfev", type=int, help="limit on model evaluations")
    fit.add_argument("--workers", type=int, help="processes for the Monte-Carlo runs")
    fit.add_argument(
        "--update-project", action="store_true", help="write the fitted values into FILE (.json)"
    )
    fit.add_argument("--text", action="store_true", help="print a readable report, not JSON")

    scan = commands.add_parser("anisotropy", help="I(angle) at fixed |Q|")
    scan.add_argument("file")
    scan.add_argument("--scan", type=int, help="number (1-based) of a scan saved in the project")
    scan.add_argument("--sweep", choices=("omega", "alpha"), help="angle to sweep")
    scan.add_argument("--fixed", type=float, help="the fixed angle [deg]")
    scan.add_argument("--q", type=float, nargs="+", help="|Q| values [1/A]")
    scan.add_argument("--points", type=int, default=73)
    scan.add_argument("--no-measured", action="store_true", help="skip the measured cuts")
    scan.add_argument("--csv", help="write the total intensity as CSV to this file")

    export = commands.add_parser("export-bernardyn", help="data and model as a Bernardyn package")
    export.add_argument("file")
    export.add_argument("package")
    export.add_argument("--recipe", default="raw_iq", help="Bernardyn recipe (default raw_iq)")
    export.add_argument("--overwrite", action="store_true")

    data = commands.add_parser("testdata", help="write synthetic test systems with known truth")
    data.add_argument("folder", nargs="?")
    data.add_argument("--system", action="append", help="system name (default: all)")
    data.add_argument("--format", choices=("h5", "dat"), default="h5")
    data.add_argument("--noise", type=float, default=0.02)
    data.add_argument("--seed", type=int, default=0)
    data.add_argument("--list", action="store_true", help="list the systems and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "info":
            _emit(api.describe(args.file))
        elif args.command == "calc":
            _emit(api.calculate(args.file, results=args.results))
        elif args.command == "fit":
            report = api.fit(
                args.file,
                results=args.results,
                monte_carlo_runs=args.mc,
                max_nfev=args.max_nfev,
                update_project=args.update_project,
                workers=args.workers,
            )
            print(_fit_text(report)) if args.text else _emit(report)
            return 0 if report["success"] else 1
        elif args.command == "anisotropy":
            scan = api.anisotropy(
                args.file,
                scan=args.scan,
                sweep=args.sweep,
                fixed=args.fixed,
                q_values=args.q,
                n_points=args.points,
                with_measured=not args.no_measured,
            )
            if args.csv:
                _write_csv(args.csv, scan)
            _emit(scan)
        elif args.command == "export-bernardyn":
            _emit(
                api.export_bernardyn(
                    args.file, args.package, recipe=args.recipe, overwrite=args.overwrite
                )
            )
        elif args.command == "testdata":
            if args.list:
                _emit(api.list_test_systems())
            elif not args.folder:
                print("anisa testdata: a folder is required (or use --list)", file=sys.stderr)
                return 2
            else:
                names = args.system or [s["name"] for s in api.list_test_systems()]
                _emit(
                    [
                        api.write_test_system(
                            name, args.folder, fmt=args.format, noise=args.noise, seed=args.seed
                        )
                        for name in names
                    ]
                )
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"anisa {args.command}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
