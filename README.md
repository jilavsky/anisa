# Anisa

Small-angle scattering (SAXS / USAXS / SANS) modelling of **anisotropic particle populations**.

Anisa fits populations of spheroids with orientation distributions to several 1D cuts I(Q),
each measured along a different direction of Q in the sample frame. It is the Python successor
of the Igor Pro AnisoSAS macros (theory: A. J. Allen, NIST 2001). Compared with the Igor
version it is faster, uses multiple cores, and has no fixed limit on the number of directions
or populations.

> **Status: early development (pre-alpha).** See [PLAN.md](PLAN.md) for the plan and progress.

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

## Units

- Q in Å⁻¹.
- Intensity in cm⁻¹ (absolute).
- Radii in Å.
- Scattering contrast Δρ in cm⁻² (Δρ, not Δρ²).
- Angles in degrees.

## License

MIT — see [LICENSE](LICENSE).
