# Anisa — development plan

Anisa models small-angle scattering from **anisotropic particle populations**: spheroids with an
orientation distribution, measured as 1D cuts along several directions of Q in the sample frame.
It is a Python successor to the Igor Pro "AnisoSAS" macros (`Igor source/`, theory by A. J. Allen,
NIST 2001). It must:

- keep the Igor physics and the parameters users know, but **fix its bugs and ad hoc numerics**
  (Igor results are a sanity check, not a bit-for-bit target — decision 2026-11-04, §12),
- be much faster (vectorised and multicore),
- have no fixed limit on the number of directions or populations,
- install from PyPI into the same environment as pyIrena and Bernardyn,
- have three surfaces: **GUI (priority)**, command line with JSON configuration, and an MCP server.

Status: **Phase 0 (repository and environment) done; Phase 1 next.** Decisions taken so far
are in §12, open to-dos in §13.

---

## 1. Goals and non-goals

**Goals for the first testable release (Phases 1–6)**
- Same physics and same parameters as Igor AnisoSAS (formulas 4a/4b/4c, size distributions,
  orientation distributions, interference, surface area, fitting, azimuthal anisotropy).
- Data input identical to pyIrena (NXcanSAS HDF5 and ASCII through `pyirena.io`).
- A pyIrena-style GUI the user can test end to end with real data.
- Calculation fast enough that "update automatically" is usable for 10–15 directions.

**Later**
- Command line and JSON batch runs, MCP server for agents (Aida).
- Bernardyn package export, launching from pyIrena.
- Direct modelling of 2D detector images.

**Data:** desmeared (pinhole-like) 1D cuts only for now. On USAXS that means 2D-collimated
USAXS, which is planned to return. Slit-smeared anisotropic data is a research item (§11); it
would make the package much more useful.

**Not planned:** multiple-scattering (MSANS) analysis. Allen's notes leave it unfinished.

---

## 2. What the Igor code does (physics to keep; bugs are fixed, see §2.8)

Source: `Igor source/ASAS_*.ipf` (v1.1–1.2, 2005/2017). Theory: `Igor source/Theory AnisoSAS.pdf`.
Igor package folder: `root:Packages:AnisoSAS`.

### 2.1 Geometry
- Sample frame: z = substrate normal **n**. A direction is (α polar from z, ω azimuth from x).
- Each *population* is a set of axially symmetric spheroids with radii R₀, R₀, βR₀ (β<1 oblate,
  β>1 prolate). Their symmetry axes follow an orientation distribution P(α,ω) = p(α)·b(ω).
- Each *direction* (data set) is one 1D cut I(Q) with Q along (α_Q, ω_Q) in the sample frame
  (degrees in the GUI). Q is assumed to lie in the sample plane; Allen ignores the φs/2 tilt.
- cos η = X = cos ω_Q cos ω sin α_Q sin α + sin ω_Q sin ω sin α_Q sin α + cos α_Q cos α   (eq. 1)

### 2.2 Orientational average (eq. 3) — `ASAS_CalcIntOf1Scatterer`
dσ/dΩ(Q) = Σ_α Σ_ω b(ω) p(α) sin α · F(Q, X(α,ω)) · Δω Δα
- **Grid:**
  - α = 0…π/2 in `IntegrationStepsInAlpha` steps (default 90, so 91 points with both ends).
  - ω = 0…2π in `IntegrationStepsInOmega` steps (default 200, so 201 points with both ends).
  - Plain rectangle sum with both endpoints included, so ω = 0 and 2π are counted twice.
- **p(α):** a wave of `PAlphaSteps` points (default 100) spanning 0…π/2, read by linear
  interpolation. Normalised so ∫p(α) sin α dα = 1, using a trapezoid on a 500× oversampled copy.
- **b(ω):** a wave of `BOmegaSteps` points spanning 0…2π, read by linear interpolation.
  Normalised so ∫b dω = 1 (trapezoid `area`).
- **Parametric forms** (when "Use parameters" is checked; Par1 in degrees):
  - p(α) = Par3 + |cos(α − Par1)|^Par2
  - b(ω) = Par3 + |cos((ω − Par1)/2)|^Par2
  - Otherwise the user edits the table of bin values directly.
- **Lookup table for F(Q,X).** F is not evaluated at every grid point. Instead:
  1. Build a table over X ∈ [−1, 1] with N = floor(16·(1 + |Q R₀ (β−1)|/π)) points.
     If N > 400, N is set to 300 (but 300 < N ≤ 400 is kept).
  2. Smooth it with Igor `Smooth max(2, floor(N/10))`: that many passes of the [1,2,1]/4
     binomial filter with "bounce" end handling (Igor default, confirmed in the WaveMetrics docs).
  3. Read it by linear interpolation at X(α,ω).

### 2.3 Single-particle cross-section F(Q,X) — `UseOfFormula4`
Here K(β,X) = √(1 + (β²−1)X²), k = 2π/λ, and ν₀ = 2R₀Δρλ (Δρ in cm⁻², converted with 1e-16).
1. **4c — diffraction limit (default, fastest), with size distribution:**
   F = Σᵢ cᵢ Vᵢ² Δρ² (9π/2)[J₃/₂(QRᵢK)/(QRᵢK)^{3/2}]² / Σᵢ cᵢ,  with cᵢ = wᵢ V₀/Vᵢ.
   - The bracket is exactly the sphere amplitude [3(sin x − x cos x)/x³]², so Python uses the
     closed form.
   - **λ drops out of 4c entirely.**
   - Triangular distribution (default): 9 radii Rᵢ = R₀(1 + i·FWHM/4) for i = −4…4, with
     weights wᵢ = 0.2 − 0.04|i|.
   - "Gauss" distribution: an odd number of bins (`GaussSDNumBins`), step 1.5·R₀·GaussSDFWHM/N,
     weight exp(−(R−R₀)²/(R₀·GaussSDFWHM)²). The "FWHM" is really the 1/e half-width, and the
     range is truncated at ±0.75 of it (fixed in Anisa, §2.8).
2. **4a — full integral with refraction:**
   f = i k R₀² K ∫₀¹ J₀(QR₀Kξ)(1 − exp[iβν₀√(1−ξ²)/K]) ξ dξ, and F = |f|².
   The ξ grid has floor(20·QR₀K/6.2 + 40) points, integrated by trapezoid. No size distribution.
3. **4b — infinite series with refraction:**
   Σₘ (iβν₀/K)ᵐ 2^{m/2} Γ(1+m/2) J_{1+m/2}(x)/x^{1+m/2}, summed up to m = 328. Convergence
   (1e-3) is checked every 4 terms. No size distribution.

### 2.4 Population and direction totals — `ASAS_CalcOnePopDir`, `ASAS_CalcAllIntensities`
- **Units:** V₀ = 4/3 π β R₀³ [Å³], N = 1e24·φ/V₀ [cm⁻³], and I_pop = F·N·1e-16 [cm⁻¹].
- **Optional interference:** I_pop /= (1 + Pack · 3(sin Qη − Qη cos Qη)/(Qη)³)
  (the Irena/Unified form).
- **Direction total:** I_dir(Q) = Σ_pop I_pop(Q).
- **Background** is subtracted from the data, never added to the model, and never fitted.
- **Surface area** per population: S = 4πβχR₀² · φ/V₀ · 1e8 [cm²/cm³], with the spheroid χ(β)
  factor.
- **Caching:** a population is recomputed only if its inputs changed (a string compare of every
  5th point). Changes to volume fraction alone are applied by rescaling.

### 2.5 Fitting — `ASAS_FitModelToData`
- All directions are concatenated into one vector and fitted with Igor `FuncFit`
  (Levenberg–Marquardt), weights 1/σ, box constraints from Min/Max.
- **Fittable per population:** VolumeFraction, Radius, PAlphaPar1–3, BOmegaPar1–3, InterfETA,
  InterfPack.
- **Not fittable:** β, Δρ, background, and distributions entered as tables.
- "Reverse fit" restores the pre-fit values. χ² is reported per direction and in total (χ²/N).

### 2.6 Anisotropy (azimuthal) tool — `ASAS_IntAnisotropy.ipf`
- Three fixed sample orientations (X, Y, Z), each with up to 6 Q values:
  - X: α_Q fixed (`Ani1_AlphaFixed`), ω_Q swept 0…360° (`NumberOmegaPoints`, default 32).
  - Y and Z: ω_Q fixed (0° and 90°), α_Q swept 0…360° (`NumberAlphaPoints`, default 32).
- The model is shown on polar plots. Experimental I(angle) can be overlaid, optionally rescaled
  to the model maximum.

### 2.7 Inputs
- USAXS desmeared data (`DSM_Int/Qvec/Error`) or multiple-scattering-corrected data (`M_DSM_*`)
  from Igor folders.
- The wavelength is read from the wave note of direction 1.
- No slit smearing in the model.
- **Hard limits:** 6 directions, 5 populations, and for anisotropy 6 Q values × 3 orientations.

### 2.8 Issues found in the Igor code — all fixed in Anisa (decision 2026-11-04)
Anisa keeps Igor's physics and parameters, but not its bugs or ad hoc numerics. With the same
parameters, Anisa results can therefore differ from Igor.

| # | Where | Igor behaviour | Anisa |
|---|---|---|---|
| 1 | `ASAS_FitCreateParametersWvs` | Uses `Pop1_UsePAlphaParam` / `Pop1_UseBOmegaParam` for *every* population | Each population uses its own flags |
| 2 | `ASAS_CalcOneMterm` | Formula 4b reads `GammWv`, but the line creating it is commented out in `ASAS_Initialize`, so 4b cannot run | Correct implementation (scipy `gamma`, `jv`); validated against 4a for β<1 |
| 3 | `ASAS_CalcIntOf1Scatterer` | Lookup-table size from a heuristic; N>400 becomes 300, but 300<N≤400 is kept | F evaluated on an X grid fine enough to resolve its oscillations (node count from QR₀\|β−1\|) |
| 4 | 4c size distribution | Normalising by Σcᵢ gives I = φ Σwᵢ Vᵢ Fᵢ / (V₀ Σ wᵢ/Vᵢ), which does not conserve volume fraction for broad distributions | Volume-weighted: I = φΔρ² Σ wᵢVᵢPᵢ with Σwᵢ = 1, for every distribution |
| 5 | Lookup smoothing | Binomial smoothing over X averages away oscillations (hidden polydispersity) | No smoothing; polydispersity comes only from the explicit size distribution |
| 6 | Gauss size distribution | "FWHM" is the 1/e half-width; range truncated at ±0.75 of it | Irena distributions from `pyirena.core.distributions` (Gauss, log-normal, Schulz–Zimm, LSW, Ardell), with Irena parameter names and radius grids; triangular kept for continuity |
| 7 | Background | Subtracted from data, not fitted | Part of the model, fittable |
| 8 | `ASAS_Initialize` | `RadiusMax` defaults to 1000, the same as Radius (the comment says 10000) | Sensible defaults |
| 9 | Integration | Rectangle sum including both ω endpoints (0 and 2π) | Periodic grid in ω; α quadrature with the sin α weight (Gauss–Legendre in cos α or midpoint), chosen by convergence tests. Distributions normalised on the same quadrature, so Σ W = 1 exactly |

**Consequence for users of the Igor version:** with a narrow size distribution, Anisa shows
form-factor oscillations that Igor's smoothing (item 5) removed. Refitting old Igor results may
need a broader size distribution. Document this in the user docs.

---

## 3. Where the time goes, and how Anisa avoids it

**Igor's cost:** for every direction × population × Q point, Igor builds a form-factor lookup
table and then interpolates it at all 91 × 201 = 18,291 (α,ω) grid points. With 6 directions,
3 populations and 300 Q points, that is ~10⁸ interpolations per model evaluation, repeated for
every Jacobian column of a fit.

**What can be computed once instead of per Q point:**
- **The X grid and the weights.** X(α,ω) and W(α,ω) = p(α)b(ω) sin α ΔαΔω do not depend on Q.
  Per direction, W can be binned once onto a fixed X grid, so each Q point costs a short dot
  product.
- **The 4c form factor.** It depends on Q and X only through Q·K(β,X), with a closed form for
  every radius in the size distribution. It is cheap to evaluate directly on a fine X grid.
- **The 4a/4b form factor.** It depends on Q, R₀, β, ν₀ and X, but **not on the direction**.
  It can be computed once per population and shared by all directions with the same Q grid.

**Prototype measurement** (this Mac, numpy, formula 4c, Igor-default grids, 300 Q points,
1 direction × 1 population, `/tmp` benchmark):

| Variant | Time | Agreement with exact Igor algorithm |
|---|---|---|
| Straight numpy port of the Igor algorithm | ~40 ms | (reference) |
| Weights binned once onto 4001 fixed X nodes | ~21 ms | 7e-6 relative |
| Same, 1001 nodes | ~19 ms | 9e-5 relative |

The remaining time is a Python loop over Q; vectorising it is the next step. The Igor run time
for the same case is the speed baseline, recorded with the Igor reference data (§13).

**Parallel strategy, in order of use:**
1. **Numpy vectorisation** — most of the gain, with no overhead.
2. **`ThreadPoolExecutor` over independent (direction × population) blocks.** Numpy and scipy
   release the GIL in large array operations; there is no pickling and it is safe inside the GUI.
3. **`ProcessPoolExecutor`** (spawn context, BLAS pinned to one thread, as in
   `pyirena/core/modeling.py`) for Monte-Carlo uncertainty runs and later 2D images.
4. **Fit Jacobian:** each fitted parameter belongs to one population, so a finite-difference
   column only recomputes that population. Columns run in parallel.
5. **Caching:** each population result is keyed by a hash of exactly the inputs it depends on
   (replacing Igor's every-5th-point string compare). Changes to volume fraction alone rescale.

**Numba** is not used anywhere in the pyIrena family and would be a new dependency. Consider it
only if benchmarks after steps 1–5 still fall short, and ask first.

**Performance targets** (formula 4c, default integration grids):
- 1 direction × 1 population × 300 Q points: ≤ 10 ms.
- 15 directions × 5 populations: < 1 s per full model evaluation, so auto-update stays usable.

---

## 4. Architecture

Flat package, house layering: `core` (numpy/scipy only) ← `io` ← `api` ← {`gui`, `cli`, `mcp`}.
No Qt or matplotlib below `gui`.

```
anisa/
  core/
    geometry.py      X(α,ω), integration grids, lab↔sample frame (needed for §10 and §11)
    orientation.py   p(α), b(ω): parametric / table modes, normalised on the integration grid
    formfactor.py    formulas 4a, 4b, 4c; size distributions via pyirena.core.distributions
    engine.py        orientational average, per-population cache, parallel execution
    model.py         dataclasses: Direction, Population, Parameter, CalcSettings, AnisaModel
    fitting.py       scipy least_squares (TRF, bounds), χ², revert, Monte-Carlo uncertainty
    derived.py       surface area, number density, scatterer volume
    anisotropy.py    azimuthal profiles I(angle) at fixed Q
  io/
    data.py          load directions via pyirena.io (discover_scattering / load_scattering)
    project.py       JSON project/config: to_dict / from_dict with defaults for every field
    results.py       HDF5 results writer/reader (format: §6)
  api/               JSON-serialisable functions shared by CLI and MCP (Phase 7)
  cli.py             `anisa` command (argparse, no new dependency) (Phase 7)
  mcp/server.py      FastMCP server `anisa-mcp`, tools prefixed `anisa_` (Phase 8)
  demo.py            synthetic multi-direction data with known parameters (testing without data)
  gui/
    _qt.py           the single PySide6 import point (no PyQt6 fallback, per house standard)
    app.py           `anisa-gui` entry point
    main_window.py   splitter: controls left, graph tabs right
    directions.py    directions table widget
    populations.py   population tabs and parameter grid
    orientation_editor.py  p(α)/b(ω) editor (parametric or table) with preview plot
    anisotropy_tab.py      azimuthal / polar plots
    workers.py       QThread workers for calculate / fit, with progress and cancel
tests/
```

### 4.1 Model and JSON configuration
One data model, used by the GUI, the CLI and MCP, and saved as both project JSON and an HDF5
attribute. JSON keys are snake_case; GUI labels keep the Igor names users know. The full mapping
table goes in `docs/parameters.md`.

| JSON key | Igor name | Unit |
|---|---|---|
| `directions[i].alpha_q`, `.omega_q` | `DirN_AlphaQ`, `DirN_OmegaQ` | deg |
| `directions[i].background` (+ `fit`, `limits`) | `DirN_Background` (not fittable in Igor) | cm⁻¹ |
| `populations[j].radius` | `PopN_Radius` | Å |
| `populations[j].volume_fraction` | `PopN_VolumeFraction` | — |
| `populations[j].beta` | `PopN_Beta` (aspect ratio) | — |
| `populations[j].delta_rho` | `PopN_DeltaRho` | cm⁻² |
| `populations[j].size_dist` {`type`, Irena distribution parameters, `n_bins`} | `PopN_UseTriangularDist`, `_FWHM`, `_GaussSDFWHM`, `_GaussSDNumBins` (replaced by the Irena set, §2.8) | — |
| `populations[j].alpha_dist` {`mode`, `n_bins`, `par1..3`, `table`} | `PopN_UsePAlphaParam`, `_PAlphaSteps`, `_PAlphaPar1..3`, `_AlphaDist` | deg for par1 |
| `populations[j].omega_dist` {…} | `PopN_UseBOmegaParam`, `_BOmegaSteps`, `_BOmegaPar1..3`, `_OmegaDist` | deg for par1 |
| `populations[j].interference`, `eta`, `pack` | `PopN_UseInterference`, `_InterfETA`, `_InterfPack` | Å, — |
| `settings.formula` = `4c`/`4a`/`4b` | `UseOfFormula4` = 1/2/3 | — |
| `settings.n_alpha_steps`, `n_omega_steps` | `IntegrationStepsInAlpha/Omega` | — |
| `settings.wavelength` | `Wavelength` | Å |

Every fittable value is `{value, fit, min, max}`. Lists of directions and populations have no
length limit.

### 4.2 Reuse from pyIrena (confirmed in pyIrena 1.2.1)
| Need | pyIrena piece | Notes |
|---|---|---|
| Find and read data | `pyirena.io.discover_scattering`, `load_scattering` → `ScatteringRecord` (q, I, σ, dQ, units, metadata, sha256 fingerprint) | Public, Qt-free, built for companion apps (Bernardyn uses it) |
| File list filter / sort / types | `pyirena.gui.file_filter.make_file_matcher`, `pyirena.core.file_sorting`, `pyirena.core.file_types` | Same behaviour as the pyIrena Data Selector |
| Load one file in a panel | `pyirena.gui.data_loading.DataFileLoaderRow` (signal `data_loaded`) | Internal widget; see note below |
| Log-log plots, cursors, export | `pyirena.gui.sas_plot.make_sas_plot`, `plot_iq_data`, `plot_iq_model`, `make_cursors` | pyqtgraph |
| Saving settings into HDF5 | `pyirena.io.setup_config.write_setup_config` / `read_setup_config` | JSON envelope attribute |
| Fit-quality block | `pyirena.core.fit_metrics.fit_quality_metrics` + `pyirena.io.nxcansas_fit_quality.write_fit_quality` | Same `fit_quality/` group as the other tools |
| Size distributions | `pyirena.core.distributions` (`pdf`, `cdf`, `generate_radius_grid`; Gauss, log-normal, Schulz–Zimm, LSW, Ardell) | Same parameters as Irena Modeling |
| Cross-check of the physics | `pyirena.core.form_factors.spheroid_ff` (Gauss–Legendre orientation average) | The isotropic limit must match it |

**Policy (decision 2026-11-04): reuse pyIrena as much as possible.** The GUI widgets and
`sas_plot` are internal to pyIrena, not documented API, so pin `pyirena>=1.2.1,<2`. When Anisa
needs a pyIrena piece that is not public, promote it to documented public API in pyIrena (as
`pyirena.io.scattering` was for Bernardyn) rather than copy it. Those changes go to the pyIrena
repository as separate, coordinated commits.

---

## 5. GUI (priority)

Mirrors the pyIrena panel layout. A horizontal `QSplitter` holds a scrollable control column on
the left (≈ 400–460 px) and graph tabs on the right. Settings persist as JSON (pyIrena contract:
`save_state()` / `load_state()` / `_collect_state()` / `_apply_state()`), in `~/.anisa/`.

**Left column, top to bottom**
1. Title, help button, and a File menu: New / Open / Save project (JSON).
2. **Directions table**, unlimited rows. Columns: use ✓ | data (file : dataset) | α_Q° | ω_Q° |
   background | fit bkg | Q min | Q max | colour.
   - Buttons: Add files… (multi-select and drag-and-drop; HDF5 files with several datasets get
     the pyIrena picker), Remove, Duplicate, Up/Down.
   - Wavelength field, filled from file metadata when present. It only matters for 4a/4b.
3. **Populations**, as a tab per population with "+" to add and ✕ to remove (unlimited). Each tab:
   - Δρ, β, and the size distribution (triangular / Gauss with its widths).
   - A parameter grid with columns `Fit? | Parameter | Value | min | max | ± std` for Radius,
     Volume fraction, P-alpha par 1–3, B-omega par 1–3, interference η and Pack.
   - Orientation distributions: "use parameters" or "table", an Edit table… dialog, and a small
     live preview of p(α) and b(ω).
   - Derived values: scatterer volume, number density, surface area (cm²/cm³).
4. **Calculation settings:** formula 4c / 4a / 4b, α steps, ω steps, number of threads.
5. **Actions:** Calculate, Fit, Revert fit, ☐ Update automatically, ☐ Show single populations,
   Save results, Export…; a status line with χ² and calculation time.

**Right side, tabs**
- **I(Q):** log-log plot of all directions (one colour each), model lines, optional dashed lines
  per population, Q-range cursors, and a residuals plot underneath.
- **Orientation:** p(α) and b(ω) for one or all populations (the Igor "probability graph").
- **Anisotropy:** any number of azimuthal scans (not fixed X/Y/Z with ≤ 6 Q).
  - Each scan is either "α_Q fixed, sweep ω_Q" or "ω_Q fixed, sweep α_Q", with a list of Q values
    and a number of points.
  - Shown as linear I(angle) plots and as polar plots drawn in pyqtgraph.
  - Experimental I(angle) can be overlaid, optionally scaled to the model.

Calculations and fits run in a `QThread` worker with progress and cancel, so the window stays
responsive. Igor's "Reverse fit" becomes "Revert fit", with the same behaviour.

**Testing without data:** `anisa.demo` writes synthetic NXcanSAS files for several directions
from known parameters (with noise). The GUI can then be checked end to end, and a fit must
recover the parameters, before real data is available.

---

## 6. Saving results (format approved 2026-11-04)

A fit spans several data files (one per direction), so pyIrena's "one `entry/<tool>_results`
group per data file" does not map one-to-one.

**Proposal**
1. **Primary:** one self-contained results file, `<name>.anisa.h5`, in NXcanSAS form:
   - **One `SASentry` per direction** (`dir01`, `dir02`, …). Each holds the fitted data
     (Q, I, Idev, Qdev), α_Q and ω_Q, the source file and dataset path with its sha256
     fingerprint, and an `anisa_results` (NXprocess) group with `I_model`, the per-population
     curves, residuals and background.
   - **One `anisa` NXentry** holding the shared results: `params/<name>` with `limit_low` /
     `limit_high` (pyIrena convention), `params_std/`, `derived/` (surface areas, number
     densities), the orientation distributions, `fit_quality/` (pyIrena helper), and the full
     configuration as a JSON attribute (pyIrena `write_setup_config` envelope).
   - Any NXcanSAS reader (pyIrena Data Explorer, Bernardyn via `pyirena.io`) can open each
     direction's data. "Open project" restores the whole session from the file.
2. **Optional:** also write `entry/anisa_results` back into each source file, so pyIrena's
   per-file tools see the model next to the data. Off by default.
3. **Project JSON** (no arrays, just paths and fingerprints) for quick save/load and for the CLI.
4. **Bernardyn package** (Phase 9): write data plus model curves for all directions as
   `scattering_curve_v1` datasets through Bernardyn's `save_package`, or through the headless
   `create_plot` once Bernardyn can read `anisa` results. Bernardyn's result reader currently
   accepts only six hard-coded pyIrena analyses. Bernardyn stays an optional import, not a
   dependency.

Group and dataset names are frozen once released, and `from_dict()` supplies defaults for every
field.

---

## 7. Command line, JSON batch, MCP (after the GUI)

- **Batch pattern:** config JSON → `anisa.api` → results HDF5, as in `pyirena.batch`.
  - Commands: `anisa calc config.json`, `anisa fit config.json [--mc N]`,
    `anisa anisotropy config.json`.
  - The config is the same JSON the GUI saves.
- **`anisa.api`:** functions that take and return only JSON-serialisable values (paths, numbers,
  lists). Both the CLI and MCP call these.
- **MCP:** `anisa-mcp` (FastMCP, `mcp>=1.0,<2.0` like pyIrena and Bernardyn), stdio, tools
  prefixed `anisa_`. Data access limited to a root folder, as with `PYIRENA_DATA_ROOT`.
  Registered in Aida with `aida mcp server add anisa --command …`; a dedicated
  `aida mcp add-anisa` helper can follow.

---

## 8. Phases

| Phase | Content | Done when |
|---|---|---|
| **0** ✅ | Repository, packaging, environment, CI skeleton, this plan | `conda env create` works; `import anisa`, ruff and pytest pass |
| **1** | Core engine: geometry (sample frame now, lab frame designed in), orientation, 4a/4b/4c, Irena size distributions, interference, totals, surface area; caching; threads | Validation §9 items 1–3 pass; convergence tests set the default grids; benchmark recorded |
| **2** | Data I/O through `pyirena.io`; project JSON round trip; `anisa.demo` synthetic data | Load real and synthetic files; JSON save/load is lossless |
| **3** | Fitting: least_squares with bounds, fittable background, revert, χ² per direction, Monte-Carlo uncertainty | Synthetic round trip recovers parameters |
| **4** | GUI: main window, directions, populations, settings, I(Q) and orientation tabs, workers | Jan can load data, calculate and fit interactively |
| **5** | Anisotropy tab (azimuthal and polar, experimental overlay) | Igor's X/Y/Z scans are reproducible as special cases |
| **6** | Results HDF5 (§6) | Save, reopen, and read the data in pyIrena / Bernardyn |
| — | **Milestone: testing release for Jan with real data** | |
| 7 | `anisa.api`, CLI, JSON batch | Same result headless as in the GUI |
| 8 | MCP server, Aida registration, skill file | An agent can run a fit end to end |
| 9 | Bernardyn package export; launch from pyIrena (needs a small pyIrena plugin hook) | — |
| 10 | 2D detector images (research, §10) | — |
| 11 | Slit-smeared anisotropic data (research, §11) | A forward model validated on synthetic data |

---

## 9. Validation

1. **Analytic limits**
   - β = 1 with any orientation distribution gives the sphere form factor × φΔρ²V for every
     direction.
   - An isotropic distribution with any β gives the same result for every direction, and it
     must match `pyirena.core.form_factors.spheroid_ff`.
2. **Formula consistency:** 4a → 4c as ν₀ → 0; 4b = 4a where the series converges (β < 1).
3. **Normalisation:** Σ W = 1 exactly for every orientation distribution, and Σ wᵢ = 1 for
   every size distribution.
4. **Convergence:** results stop changing as the α, ω and X grids are refined; the defaults are
   the coarsest grids within a stated tolerance (e.g. 1e-4 relative).
5. **Fit round trip:** synthetic data with noise, starting from perturbed values.
6. **Anisa regression values:** once items 1–5 pass, store reference outputs of Anisa itself.
   A later change that moves them must be explained (a fix) or it is a regression.
7. **Igor comparison (sanity check, §13):** expect agreement only where Igor's approximations
   are small (broad size distributions, fine grids); document the differences.
8. **Performance regression test** (marked, not run by default in CI).

---

## 10. 2D detector images (later, design kept open now)

The core takes arrays of (Q, α_Q, ω_Q), not "a direction". A 1D cut, an azimuthal scan and a
detector image are then just different samplings. With binned orientation weights, a 2D image
becomes a matrix product: I(q, φ) = F(q, X_nodes) · H(φ, X_nodes)ᵀ.

Problems to solve then:
- **Lab → sample rotation** for sample stages: which angles are recorded and how.
- **The Q tilt out of the sample plane** (φs/2). Negligible for USAXS/SAXS, but not at WAXS
  angles.
- **Detector geometry and calibration input:** reuse NXcanSAS 2D or Matilda outputs rather
  than adding pyFAI.
- **Masks, binning, and fitting cost** with 10⁵–10⁶ pixels: fit on azimuthal sectors first.

---

## 11. Slit-smeared anisotropic data (research)

Igor AnisoSAS needs desmeared data, and so does Anisa for now. On USAXS that means 2D-collimated
USAXS. A forward model for slit-smeared data looks possible, and it would make Anisa much more
useful.

**Why the usual approach fails.** For an isotropic sample, slit smearing is a 1D integral,
I_s(q) = (1/ℓ)∫₀^ℓ I(√(q²+l²)) dl, so the data can be desmeared first and fitted afterwards.
For an anisotropic sample, every point along the slit has a **different Q direction**:
- Q = q·ê_scan + l·ê_slit (small angle, Q ⟂ beam). ê_scan is the Bonse–Hart scan direction and
  ê_slit the slit-length direction, both in the lab frame.
- With the sample orientation R (lab → sample frame), the direction along the slit is
  d(l) = R(q·ê_scan + l·ê_slit)/√(q²+l²).
- So I_s(q) = (1/2ℓ)∫₋ℓ^ℓ I(√(q²+l²), d(l)) dl. Both signs of l are needed: ±l give different
  directions unless the sample is symmetric about the scan direction.

Desmearing such data as if it were isotropic is wrong. Fitting the smeared model directly is
not.

**What Anisa needs for it:**
- **The core already fits.** It evaluates I at arrays of (|Q|, α_Q, ω_Q), so a smeared point is
  a weighted sum over ~50–100 directions along the slit. That is affordable with the vectorised
  engine.
- **Per-measurement geometry:** the sample orientation (rotation angles recorded by the stage),
  ê_scan and ê_slit, and the slit length (`ScatteringRecord` already carries `slit_length`).
- **Information content:** each point mixes directions, so measurements at several sample
  rotations (e.g. about the beam axis) are needed to recover the anisotropy. Synthetic tests
  will show how many.

**Design consequence now:** `core/geometry.py` gets the lab ↔ sample frame from the start
(Phase 1). A `Direction` can then be given either as explicit (α_Q, ω_Q) or as a lab geometry
(sample orientation + scan direction + optional slit). The same frame serves 2D images (§10).

---

## 12. Decisions (Jan, 2026-11-04)
1. **Results format (§6) approved:** a standalone `<name>.anisa.h5` with one SASentry per
   direction, following Bernardyn's self-contained package logic. Write-back into source files
   is optional.
2. **Fix the Igor bugs and leave Igor behind.** There is no reason to match Igor results that
   include bugs; this is the time to do it right. There is no Igor-compatible mode (§2.8).
3. **Igor reference data:** deferred until test data exists (generated or real) — see §13.
4. **Data:** desmeared data only, for now. Slit-smeared anisotropic data is research (§11).
   2D-collimated USAXS is planned to return.
5. **Reuse pyIrena as much as possible**; promote pieces to public pyIrena API when needed
   (§4.2).
6. **Allen's theory notes** (`Igor source/Theory AnisoSAS.*`) stay local: listed in
   `.gitignore`, not tracked. The Igor `.ipf` code is tracked as reference.

---

## 13. To-do (outside the phase plan)
- [ ] **Igor reference data** (Jan). Once synthetic or real test data exists, run Igor
      AnisoSAS on a few fixed configurations and save the experiment (`.pxp`, readable through
      `igor2`). Record the Igor run time (the speed baseline).
      - Configurations: oblate and prolate; isotropic and parametric; 4c and 4a; with and
        without interference.
      - Use: sanity check, §9 item 7.
- [ ] **Real user data** (Jan): desmeared multi-direction data sets with known sample
      orientations.
- [ ] **pyIrena public API:** list the pyIrena internals Anisa uses (Phase 4) and promote them.
