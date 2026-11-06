# Anisa

Small-angle scattering (SAXS / USAXS / SANS) modelling of **anisotropic particle populations**.

Anisa fits populations of spheroids with orientation distributions to several 1D cuts I(Q),
each measured along a different direction of Q in the sample frame. It is the Python successor
of the Igor Pro AnisoSAS macros (theory: A. J. Allen, NIST 2001). Compared with the Igor
version it is faster, uses multiple cores, and has no fixed limit on the number of directions
or populations.

> **Status: beta (0.0.1b1).** The engine, data input, fitting, GUI, results files, command line,
> MCP server and Bernardyn export are done and tested on synthetic data; it has not yet been
> validated on real measurements, so expect changes. See
> [PLAN.md](https://github.com/jilavsky/anisa/blob/main/PLAN.md) for the plan and progress.

Anisa is part of the pyIrena family:
- **[pyIrena](https://github.com/jilavsky/pyirena)** — data input (NXcanSAS HDF5 and ASCII) and
  shared GUI pieces.
- **[Bernardyn](https://github.com/jilavsky/Bernardyn)** — publication plots.
- **Aida** — agent workbench, which will use the Anisa MCP server.

All of them install into the same Python environment.

## Installation

Not on PyPI yet. From a source checkout:

```bash
conda env create -f environment.yml     # creates the "anisa" environment
conda activate anisa
```

To add Anisa to an existing pyIrena or Bernardyn environment instead:

```bash
pip install -e ".[gui]"
```

Python 3.10–3.13. The GUI uses PySide6 (never PyQt6).

## Using the GUI

```bash
anisa-gui                      # or: python -m anisa.gui
```

1. *File → Create test system* writes a synthetic data set with known parameters (prolate,
   oblate, both with different alignment, interference, spheres, 12 directions; `TRUTH.txt`
   lists the answer) and opens it with perturbed starting values; press **Fit** to see it
   recover them. *Create demo data…* does the same for the three-population coating example. (Or *Add data…* /
   drag files onto the directions table to load your own: NXcanSAS HDF5 or text.)
2. Give each direction its angles α_Q (from the sample normal) and ω_Q (azimuth) in the table.
3. Set the particle parameters of each population, tick **Fit?** for those to vary, and press
   **Fit**. The errors appear next to the fitted values; *Uncertainty…* refits re-drawn data for a
   Monte-Carlo estimate; *Revert fit* returns to the values before the fit.
4. The **Anisotropy** tab shows the model as a function of the direction of Q (polar plots, or
   I(angle)) with your measured cuts on top, which is the quickest check that the orientation
   distributions make sense.

After a fit, *File → Save results (HDF5)…* writes one `NAME.anisa.h5` with the data, model
curves, parameters and errors of all directions (an NXcanSAS entry per direction, so pyIrena
opens each); *File → Open* reads it back.

## Command line

```bash
anisa testdata folder --system prolate_oblate        # synthetic data with known truth
anisa info    folder/prolate_oblate/prolate_oblate_start.anisa.json
anisa fit     folder/prolate_oblate/prolate_oblate_start.anisa.json --mc 20 --text
anisa calc    project.anisa.json --results out.anisa.h5
anisa export-bernardyn project.anisa.json plot     # needs Bernardyn installed
anisa anisotropy project.anisa.json --sweep alpha --fixed 0 --q 0.01 0.05 --csv scan.csv
```

The project file is the one the GUI saves; every command prints JSON (`anisa.api` has the same
functions for Python).

## MCP server (agents)

```bash
pip install "anisa[mcp]"
ANISA_DATA_ROOT=/path/to/data anisa-mcp          # stdio server
aida mcp server add anisa --command anisa-mcp    # register with Aida
```

Tools (`anisa_` prefix): `list_files`, `describe`, `calculate`, `fit`, `anisotropy`,
`read_results`, `list_test_systems`, `write_test_system`. They are the `anisa.api` functions;
file access is confined to `ANISA_DATA_ROOT` (default: the working directory).

## Example

```python
import numpy as np
from anisa.core import AnisaModel, Direction, Population, Engine
from anisa.core.orientation import AngularDistribution

# oblate disks (beta = 0.2) with their axes along the sample z axis
pop = Population(radius=800.0, beta=0.2, volume_fraction=0.03, delta_rho=2e11,
                 alpha_dist=AngularDistribution(par1=0.0, par2=12.0))
model = AnisaModel([Direction(alpha_q=0.0), Direction(alpha_q=90.0)], [pop])

q = np.geomspace(1e-4, 0.1, 200)
result = Engine().evaluate(model, [q, q])
along_z, across_z = (d.intensity for d in result.directions)   # I(Q) in 1/cm
```

Any number of directions and populations is allowed. Parameters and their Igor names:
[docs/parameters.md](https://github.com/jilavsky/anisa/blob/main/docs/parameters.md).

Synthetic multi-direction data with known parameters, for trying things out:

```bash
python -m anisa.demo demo_data            # NXcanSAS files + demo.anisa.json (the truth)
```

```python
from anisa.core import fit
from anisa.io.project import load_project

model = load_project("demo_data/demo_start.anisa.json")   # perturbed start, fit flags set
result = fit(model)                                        # model now holds the best parameters
for p in result.parameters:
    print(p.parameter.label, p.value, "+-", p.std)
```

## Units

- Q in Å⁻¹.
- Intensity in cm⁻¹ (absolute).
- Radii in Å.
- Scattering contrast Δρ in cm⁻² (Δρ, not Δρ²).
- Angles in degrees.

## License

MIT — see [LICENSE](LICENSE).
