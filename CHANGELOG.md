# Changelog

All notable changes to Anisa are documented here. Versions follow
[PEP 440](https://peps.python.org/pep-0440/); the version in `pyproject.toml` is authoritative.

## [Unreleased]

## [0.0.1b1] - 2026-11-06

First beta release (version number reset from the development placeholder 0.1.0.dev0).

### Added
- **Bernardyn export (Phase 9)**: File → Export Bernardyn package…, `anisa export-bernardyn FILE
  PACKAGE [--recipe raw_iq|guinier|porod|kratky…]` and the `anisa_export_bernardyn` MCP tool
  put data and model curves of all directions on one graph in a self-contained package.
  Bernardyn stays an optional import (public headless API only).
- **MCP server (Phase 8)**, `anisa-mcp` (`anisa[mcp]`, mcp 1.x): eight `anisa_` tools over
  `anisa.api` (list, describe, calculate, fit with Monte Carlo, anisotropy scan, read results,
  test systems). File access confined to `ANISA_DATA_ROOT`; requests capped. In-process and
  stdio round-trip tests.
- **`anisa.api` and the `anisa` command (Phase 7)**: `info`, `calc`, `fit` (`--mc N`,
  `--update-project`, results file written next to the project), `anisotropy` (saved or ad hoc
  scan, `--csv`) and `testdata`. JSON in and out, the same functions the MCP server will call.
- **Results file (Phase 6)**, `anisa.io.results` and File → Save results (HDF5)…: one
  `<name>.anisa.h5` with an NXcanSAS `SASentry` per direction (data, `I_model`, per-population
  curves, residuals, pyIrena fit-quality), shared parameters with limits and standard errors,
  derived quantities, orientation profiles and the full model (pyIrena setup envelope).
  pyIrena opens every direction; File → Open reads the file back with its data. 9 more tests.
- **Test systems with known truth**, `anisa.testsets` (`python -m anisa.testsets FOLDER`, and
  File → Create test system in the GUI): prolate, oblate, prolate + oblate with different
  alignment, dense prolate with interference, isotropic spheres, and a 12-direction case. Each
  writes data, a truth project, a perturbed start project and a `TRUTH.txt`. A fit from the start
  recovers every parameter within 5σ (a test).
- **Calculation indicator** in the status bar: a moving bar and "Calculating…" from the moment an
  edit is pending, then "Up to date (0.12 s)"; "model out of date" when auto-update is off.
- **Anisotropy view (Phase 5)**: `anisa.core.anisotropy` (scans of α_Q or ω_Q at any number of Q
  values, saved in the project; measured cuts that lie on a scan path are interpolated and
  overlaid) and an Anisotropy tab with polar and I(angle) views. 35 more tests (290 in total).
- **GUI (Phase 4)**, `anisa-gui`: main window with a directions table, population tabs (parameters
  with fit flags, limits and errors, size distribution, orientation tables, interference), I(Q)
  with residuals and fit-range cursors, orientation-distribution plots, fit log, project
  open/save/recent, drag-and-drop data, demo data generator, background calculation and fitting
  with progress and cancel, revert, Monte-Carlo uncertainty. 28 GUI tests (255 in total).
- **Fitting (Phase 3)**, `anisa.core.fitting`: `fit` (least squares, bounds, per-population
  flags, progress/cancel), `revert`, `goodness_of_fit`, covariance and `monte_carlo` uncertainties.
  The formula-4c sum over size classes is now one vectorised contraction (fit 1.75x faster).
- **Data I/O and project files (Phase 2)**:
  - `anisa.io.data`: `load_direction` / `reload_direction_data` through pyIrena's public reader
    (NXcanSAS, simple HDF5, text); content fingerprints; wavelength from file metadata.
  - `Direction.q_min` / `q_max` / `fit_mask()` and `DataSource` (data reference).
  - `anisa.io.project`: `save_project` / `load_project` (`*.anisa.json`, atomic, relative data
    paths with absolute fallback, no arrays in the file).
  - `anisa.demo` (`python -m anisa.demo FOLDER`): synthetic three-population, four-direction
    data with known parameters, plus a perturbed starting model.
  - 39 more tests (204 in total).
- **Core engine (Phase 1)**, `anisa.core`:
  - Sample-frame geometry (Allen eq. 1); orientation distributions p(α)·b(ω), parametric or
    table, normalised exactly on the integration grid.
  - Formulas 4c (closed form), 4a (integral) and 4b (series, with the `1/m!` that Allen's printed
    formula and the Igor code lack), all with refraction where applicable.
  - Irena size distributions (Gauss, log-normal, Schulz–Zimm, LSW, Ardell), volume or number
    weighted; interference; surface area, number density.
  - `Engine`: K binning, log-space convolution for 4c with size distributions, Q bands, caching,
    thread pool. No limit on the number of directions or populations.
  - Model dataclasses with JSON round trip (`to_dict` / `from_dict` with defaults).
  - 165 tests against independent references; `benchmarks/convergence.py` and
    `benchmarks/performance.py`; `docs/parameters.md` (Igor ↔ Anisa).
- Project initialised:
  - Packaging (`pyproject.toml`, setuptools, flat `anisa/` layout).
  - Conda environment `anisa` (`environment.yml`), compatible with the pyIrena and Bernardyn
    environments.
  - PySide6-only Qt shim (`anisa/gui/_qt.py`).
  - Package contract tests.
  - GitHub Actions workflows for tests and PyPI publishing.
- `PLAN.md`:
  - Analysis of the Igor AnisoSAS code and the list of its bugs, all to be fixed.
  - Performance strategy.
  - Architecture, GUI design, results format, phases and validation plan.
  - Research note on fitting slit-smeared anisotropic data.
  - Decisions of 2026-11-04.
  - Plan for 3D visualisation of the fitted microstructure (§14, Phase 12).
