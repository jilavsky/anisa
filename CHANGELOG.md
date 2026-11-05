# Changelog

All notable changes to Anisa are documented here. Versions follow
[PEP 440](https://peps.python.org/pep-0440/); the version in `pyproject.toml` is authoritative.

## [Unreleased]

### Added
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
