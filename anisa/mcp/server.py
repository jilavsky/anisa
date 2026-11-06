"""MCP stdio server: ``anisa-mcp`` (or ``python -m anisa.mcp.server``).

Tools are prefixed ``anisa_`` and wrap :mod:`anisa.mcp.tools`. File access is limited to the
folder tree in ``ANISA_DATA_ROOT`` (default: the working directory).

Register with Aida: ``aida mcp server add anisa --command anisa-mcp`` (set ``ANISA_DATA_ROOT``
in the server's environment to the folder holding the data).
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'mcp' package (1.x) is required to run anisa-mcp. "
        "Install with: pip install 'anisa[mcp]'  (mcp 2.0 removed mcp.server.fastmcp)."
    ) from exc

from anisa.mcp.tools import TOOLS

INSTRUCTIONS = (
    "Tools for modelling and fitting small-angle scattering from populations of anisotropic "
    "spheroids measured along several directions of Q. All tool names start with 'anisa_'. "
    "Typical workflow: anisa_list_files to find projects (*.anisa.json) or results "
    "(*.anisa.h5); anisa_describe to see populations, directions and which parameters are "
    "flagged for fitting; anisa_calculate for chi-squared of the current model; anisa_fit to "
    "fit (it writes a results file and returns every parameter with start, value and standard "
    "error; check 'success' and 'reduced_chi_squared', which should be near 1; values at "
    "'at_lower_limit'/'at_upper_limit' are not determined by the data); anisa_anisotropy for "
    "I(angle) at fixed |Q| against the measured cuts; anisa_read_results to read a saved fit. "
    "To try the tools without data, anisa_write_test_system creates synthetic data with "
    "known truth (see TRUTH.txt) and a perturbed start project to fit. Units: Q in 1/A, "
    "intensity in 1/cm, radius in A, contrast (delta_rho) in cm^-2, angles in degrees. "
    "Spheroid semi-axes are (R, R, beta R): beta > 1 prolate, beta < 1 oblate. alpha is the "
    "angle from the sample normal, omega the azimuth. Files are confined to ANISA_DATA_ROOT."
)

mcp = FastMCP("anisa", instructions=INSTRUCTIONS)

for _name, _function in TOOLS.items():
    mcp.tool(name=f"anisa_{_name}")(_function)


def main() -> None:
    """Run the server on stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
