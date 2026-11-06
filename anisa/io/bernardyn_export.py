"""
Export to a Bernardyn graph package: measured curves and model curves of all directions on one
graph, for publication-quality plotting and sharing.

Bernardyn is an *optional* import (it is not a dependency of Anisa); only its public headless
API (``bernardyn.api.create_plot``) is used, so nothing here depends on Bernardyn internals. The
data and model curves are written to NXcanSAS files in a temporary folder, plotted by
Bernardyn's recipe, and embedded in the package, which is therefore self-contained.

The intensities are the *total* model with its background, as measured: Bernardyn's views
(I(Q), Guinier, Porod, Kratky, ...) apply to both data and model alike.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pyirena.io import discover_scattering
from pyirena.io.nxcansas_unified import create_nxcansas_file

from anisa.core.engine import Engine, ModelResult
from anisa.core.model import AnisaModel
from anisa.io.results import save_results

__all__ = ["export_bernardyn_package"]


def export_bernardyn_package(
    model: AnisaModel,
    package_path: str | Path,
    *,
    recipe: str = "raw_iq",
    title: str | None = None,
    result: ModelResult | None = None,
    engine: Engine | None = None,
    overwrite: bool = False,
) -> Path:
    """Write data and model curves of every direction into a Bernardyn package.

    Args:
        model: model with data in at least one direction.
        package_path: output (Bernardyn appends its package suffix if missing).
        recipe: Bernardyn recipe id (``raw_iq``, ``guinier``, ``porod``, ``kratky``, ...).
        title: graph title.
        result: calculated model if at hand (else calculated here at the data's Q).
        engine: engine to calculate with.
        overwrite: replace an existing package.

    Returns:
        The path of the package written.

    Raises:
        ImportError: Bernardyn is not installed.
        ValueError: no direction has data.
    """
    try:
        from bernardyn.api import PlotRequest, Presentation, ScatteringInput, create_plot
    except ImportError as error:  # pragma: no cover - exercised only without Bernardyn
        raise ImportError(
            "Bernardyn is required for this export (pip install bernardyn)"
        ) from error

    with_data = [i for i, d in enumerate(model.directions) if d.q is not None]
    if not with_data:
        raise ValueError("no direction has data to export")
    if result is None:
        q_list = [d.q if d.q is not None else np.geomspace(1e-3, 0.3, 50) for d in model.directions]
        result = (engine or Engine()).evaluate(model, q_list)

    with tempfile.TemporaryDirectory(prefix="anisa_bernardyn_") as folder:
        folder = Path(folder)
        data_file = save_results(model, folder / "data.anisa.h5", result=result)
        locations = {
            int(Path(loc.internal_path).parts[-2][3:]) - 1: loc
            for loc in discover_scattering(data_file)
        }
        inputs = []
        for i in with_data:
            label = model.directions[i].label or f"Direction {i + 1}"
            inputs.append(
                ScatteringInput(data_file, internal_path=locations[i].internal_path, label=label)
            )
        for i in with_data:
            calculated = result.directions[i]
            label = model.directions[i].label or f"Direction {i + 1}"
            path = folder / f"model_dir{i + 1:02d}.h5"
            create_nxcansas_file(
                path, calculated.q, calculated.intensity, None, sample_name=f"model{i + 1:02d}"
            )
            (location,) = discover_scattering(path)
            inputs.append(
                ScatteringInput(
                    path, internal_path=location.internal_path, label=f"{label} (model)"
                )
            )
        request = PlotRequest(
            recipe_id=recipe,
            inputs=tuple(inputs),
            presentation=Presentation(title=title or "Anisa: data and model"),
            package_path=Path(package_path),
            overwrite=overwrite,
        )
        outcome = create_plot(request)
    return outcome.package_path
