# CLAUDE.md — Anisa

Anisa models SAS from anisotropic spheroid populations, measured along several Q directions. It
is the Python successor of Igor AnisoSAS. Jan's cross-repo conventions are in
`~/.claude/CLAUDE.md` and are not repeated here.

## Read first
- **[PLAN.md](PLAN.md)** — the plan, phases, decisions (§12), to-dos (§13). §2 describes what
  the Igor code computes; §2.8 lists the Igor bugs and how Anisa fixes them.
- `Igor source/ASAS_*.ipf` — the original Igor code. Files use CR line endings; read them
  with `tr '\r' '\n' < file`.
- `Igor source/Theory AnisoSAS.pdf` — Allen's theory notes. They are local only (gitignored,
  absent in fresh clones). Equation numbers (1)–(7) are used in docstrings.

## Layout
- `anisa/core` — numpy/scipy only. `anisa/io` — data via `pyirena.io`, JSON, HDF5.
- `anisa/gui` — PySide6, imported only through `anisa/gui/_qt.py`.
- Tests are in `tests/`. `tests/test_package.py` enforces the layering and the Qt-import rule.

## Rules specific to this repo
- **Anisa does not aim to match Igor AnisoSAS numerically** (Jan, 2026-11-04). This overrides
  the global "validated against Igor" rule for this repo only. Keep Igor's physics and parameter
  names, fix its bugs and ad hoc numerics (PLAN.md §2.8), and treat Igor results as a sanity
  check. Anisa's own analytic, convergence and regression tests (PLAN.md §9) are the reference:
  a change that moves them must be an explained fix.
- No hard limits on the number of directions, populations or Q values.
- GUI labels keep the Igor parameter names (Radius, Volume Fract, Beta, Delta rho,
  P alpha par 1…). JSON keys are snake_case; the mapping is in PLAN.md §4.1.
- Reuse pyIrena (`pyirena.io`, `pyirena.gui.sas_plot`, `fit_metrics`, `setup_config`) before
  writing a parallel implementation.
- The environment architecture is: conda for Python and numpy/scipy/h5py, pip for Qt. Never
  add PyQt6.

## Commands
```bash
conda activate anisa
pytest                 # tests
ruff check anisa tests # lint
```
