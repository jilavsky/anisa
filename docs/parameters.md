# Parameters: Igor AnisoSAS ↔ Anisa

JSON keys are snake_case; the GUI keeps the Igor labels. Angles are in degrees, radii in Å,
contrast in cm⁻² (Δρ, **not** squared), Q in Å⁻¹, intensity in cm⁻¹.

## Directions (Igor `DirN_*`, up to 6 → unlimited)

| JSON (`directions[i]`) | Igor | Unit | Notes |
|---|---|---|---|
| `alpha_q` | `DirN_AlphaQ` | deg | Polar angle of Q from the sample z axis (substrate normal) |
| `omega_q` | `DirN_OmegaQ` | deg | Azimuth of Q from the sample x axis |
| `background` (+ `fit`, `min`, `max`) | `DirN_Background` | cm⁻¹ | Igor **subtracted** it from the data and could not fit it. Anisa **adds** it to the model and can fit it |
| `label`, `use` | — | | label (e.g. file name), include in fit |
| *(data)* `q`, `intensity`, `uncertainty` | `DirN_IntWvName` … | | not stored in the project file; see the results format (PLAN.md §6) |

## Populations (Igor `PopN_*`, up to 5 → unlimited)

| JSON (`populations[j]`) | Igor | Unit | Notes |
|---|---|---|---|
| `radius` (+ fit) | `PopN_Radius` | Å | Equatorial radius R; location parameter of the size distribution |
| `volume_fraction` (+ fit) | `PopN_VolumeFraction` | — | |
| `beta` (+ fit) | `PopN_Beta` | — | Aspect ratio, semi-axes (R, R, βR). β < 1 oblate, β > 1 prolate. Fittable in Anisa (not in Igor) |
| `delta_rho` | `PopN_DeltaRho` | cm⁻² | Contrast, not squared. With formula 4c it is degenerate with `volume_fraction`, so it is not fitted |
| `size_dist` | `PopN_UseTriangularDist`, `_FWHM`, `_UseGaussSizeDist`, `_GaussSDFWHM`, `_GaussSDNumBins` | | **Replaced**, see below |
| `alpha_dist` (`mode`, `table`) + `alpha_par1/2/3` | `PopN_UsePAlphaParam`, `_AlphaDist`, `_PAlphaPar1/2/3` | | p(α), see below |
| `omega_dist` (`mode`, `table`) + `omega_par1/2/3` | `PopN_UseBOmegaParam`, `_OmegaDist`, `_BOmegaPar1/2/3` | | b(ω), see below |
| `use_interference`, `eta` (+ fit), `pack` (+ fit) | `PopN_UseInterference`, `_InterfETA`, `_InterfPack` | Å, — | Factor `1 / (1 + pack · 3 j₁(Qη)/(Qη))` |

Every fittable value is stored as `{"value": …, "fit": false, "min": …, "max": …}` (`null` =
unbounded). A bare number is also accepted when reading.

### Orientation distributions

`P(α, ω) = p(α) · b(ω)` for the symmetry axis of the particles, α ∈ [0°, 90°] and
ω ∈ [0°, 360°). Axes are undirected, so this covers every axis exactly once.

* **Parametric** (as in Igor; par1 in degrees, par2 ≥ 0, par3 ≥ 0):
  `p(α) = par3 + |cos(α − par1)|^par2`, `b(ω) = par3 + |cos((ω − par1)/2)|^par2`.
  `par1` is the peak position, `par2` the sharpness (0 = isotropic), `par3` an isotropic floor.
* **Table**: `n` bin values on equal bins, linear interpolation between bin centres (periodic
  in ω).
* The weights are normalised **on the integration grid**, so they sum to exactly 1.
* The grid is refined automatically for sharp distributions
  (`AngularDistribution.required_nodes`): without that a very sharp peak would silently become
  a broader one.

### Size distributions

Igor's 9-point triangle and its truncated "Gauss" (whose FWHM was really a 1/e half-width) are
replaced by the Irena distributions from `pyirena.core.distributions`, with the Irena parameter
names. The population `radius` supplies the location parameter, so fitting the radius moves the
whole distribution.

| `size_dist.type` | `params` |
|---|---|
| `none` (default, monodisperse) | — |
| `gauss` | `width` [Å] (the σ) |
| `lognormal` | `min_size` [Å], `sdeviation` |
| `schulz_zimm` | `width` [Å] |
| `lsw` | — |
| `ardell` | `parameter` (2 < m < 3) |

`weighting` is `"volume"` (default: the distribution is volume fraction per radius) or
`"number"`, as `use_number_dist` in Irena Modeling. `n_bins` (default 100) is the minimum number
of radius cells; the engine uses more at high Q. It is a quadrature resolution, not a binning of
the data.

## Calculation settings (Igor `UseOfFormula4`, `IntegrationStepsIn*`, `Wavelength`)

| JSON (`settings`) | Igor | Default | Notes |
|---|---|---|---|
| `formula` | `UseOfFormula4` (1 = 4c, 2 = 4a, 3 = 4b) | `"4c"` | 4c: diffraction limit (fastest). 4a/4b: with refraction, need `wavelength` |
| `n_alpha`, `n_omega` | `IntegrationStepsInAlpha` (90), `…InOmega` (200) | 64, 128 | **Lower bounds**: the engine raises them to resolve the form factor and the orientation distribution |
| `points_per_period` | — | 48 | K-grid nodes per 2π of form-factor phase (cubic binning, error ∝ nodes⁻⁴) |
| `wavelength` | `Wavelength` | 1.0 Å | only for 4a/4b |
| `n_threads` | — | 0 | 0 = one per CPU |

## What changed relative to Igor, and why

See PLAN.md §2.8 for the full list. In short: Anisa keeps Igor's physics but not its bugs or ad
hoc numerics, so results for the same parameters can differ from Igor's. The largest effect is
that Igor smoothed the form-factor oscillations in X with a binomial filter (hidden
polydispersity); Anisa has only the explicit size distribution. A fit of old Igor results may
therefore need a broader size distribution.
