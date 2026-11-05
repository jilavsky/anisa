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
- `anisa/core` — numpy/scipy only (plus `pyirena.core.distributions` for size distributions).
  `engine.py` is the calculation engine (its module docstring explains the algorithms);
  `model.py` the dataclasses. `anisa/io` — `data.py` (reads only through `pyirena.io`),
  `project.py` (`*.anisa.json`); results HDF5 comes in Phase 6. `anisa/demo.py` — synthetic data
  (`python -m anisa.demo`), also the fixture for fitting tests.
- `benchmarks/` — `convergence.py` (accuracy against an independent brute force) and
  `performance.py`. Re-run both after any change to the engine's numerics.
- `docs/parameters.md` — Igor ↔ Anisa parameter mapping.
- `anisa/gui` — PySide6, imported only through `anisa/gui/_qt.py`.
- Tests are in `tests/`. `tests/test_package.py` enforces the layering and the Qt-import rule.

## Rules specific to this repo
- **Anisa does not aim to match Igor AnisoSAS numerically** (Jan, 2026-11-04). This overrides
  the global "validated against Igor" rule for this repo only. Keep Igor's physics and parameter
  names, fix its bugs and ad hoc numerics (PLAN.md §2.8), and treat Igor results as a sanity
  check. Anisa's own analytic, convergence and regression tests (PLAN.md §9) are the reference:
  a change that moves them must be an explained fix.
- No hard limits on the number of directions, populations or Q values.
- **Grid sizes in `CalcSettings` are lower bounds.** The engine raises them to resolve the form
  factor and the orientation distribution. Weights are normalised on the grid, so an
  under-resolved sharp distribution does not fail, it silently gets broader: never remove the
  `required_nodes` floor, and test any new distribution type for it.
- Tests compare against independent references (brute force, scipy quadrature, analytic
  limits), not against the engine's own output. Tolerances are tied to a stated accuracy target
  (1e-4 relative at the worst Q point); tighten the code, not the tolerance, if they fail.
- **Qt threading rules (each learned from a bug here).** Workers run on a deep copy of the model,
  never on the objects the GUI edits. Never drop the last Python reference to a `QThread` inside
  its own `finished` handler (the thread is still shutting down: "Destroyed while thread is
  running"); keep it in `_threads` and prune with `_reap_threads()`. "Busy" is explicit state set
  before `start()` and cleared in the job's `finished` handler, never `isRunning()`. Tests must
  never open a modal dialog (stub `_show_message` / `_confirm_discard`): offscreen it hangs
  forever.
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
