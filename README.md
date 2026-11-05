# Anisa

Small-angle scattering (SAXS / USAXS / SANS) modelling of **anisotropic particle populations**.

Anisa fits populations of spheroids with orientation distributions to several 1D cuts I(Q),
each measured along a different direction of Q in the sample frame. It is the Python successor
of the Igor Pro AnisoSAS macros (theory: A. J. Allen, NIST 2001). Compared with the Igor
version it is faster, uses multiple cores, and has no fixed limit on the number of directions
or populations.

> **Status: early development (pre-alpha).** The calculation engine, data input and project
> files are done and tested; fitting, the GUI, the command line and the MCP server are next. See [PLAN.md](PLAN.md)
> for the plan and progress.

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
[docs/parameters.md](docs/parameters.md).

Synthetic multi-direction data with known parameters, for trying things out:

```bash
python -m anisa.demo demo_data            # NXcanSAS files + demo.anisa.json (the truth)
```

```python
from anisa.io.project import load_project
model = load_project("demo_data/demo.anisa.json")   # settings + data, read back through pyIrena
```

## Units

- Q in Å⁻¹.
- Intensity in cm⁻¹ (absolute).
- Radii in Å.
- Scattering contrast Δρ in cm⁻² (Δρ, not Δρ²).
- Angles in degrees.

## License

MIT — see [LICENSE](LICENSE).
