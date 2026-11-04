# Changelog

All notable changes to Anisa are documented here. Versions follow
[PEP 440](https://peps.python.org/pep-0440/); the version in `pyproject.toml` is authoritative.

## [Unreleased]

### Added
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
