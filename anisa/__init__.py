"""
Anisa: small-angle scattering modelling of anisotropic particle populations.

Python successor of the Igor Pro AnisoSAS macros (J. Ilavsky, theory by A. J. Allen,
NIST 2001). Populations of axially symmetric spheroids with orientation distributions
p(alpha) * b(omega) are modelled against 1D cuts measured along several directions of Q
in the sample frame.

Layers:
    core: numerical model (numpy/scipy only)
    io:   data input through pyirena.io, project JSON, HDF5 results
    gui:  PySide6 GUI (optional extra ``anisa[gui]``)

See PLAN.md for the development plan.
"""

# Single-source the version from the installed package metadata
# (pyproject.toml is the source of truth).
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("anisa")
except PackageNotFoundError:  # pragma: no cover - only in uninstalled source trees
    __version__ = "0.0.0+unknown"

__author__ = "Jan Ilavsky"
