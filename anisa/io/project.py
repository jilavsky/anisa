"""
Anisa project files: the whole model as JSON, without the data arrays.

A project stores the settings, the directions (angles, background, Q range, fit flags) and
populations, and for every direction a reference to its data file: path, curve inside the file,
and a content fingerprint. Arrays are read back from the data files on load, so a project stays
small and human-readable, and the measured data have a single home. (The self-contained results
file, which does carry the data, is a different format: PLAN.md section 6.)

Layout::

    {"_anisa_project": {"format": "anisa-project", "schema_version": 1,
                        "saved_by": "anisa 0.1.0", "saved_at": "..."},
     "model": {...AnisaModel.to_dict()...}}

Data paths are stored relative to the project file when possible (so a project folder can be
moved or shared with its data), together with the absolute path as a fallback (so the project file
alone can be moved). Files are written atomically.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import anisa
from anisa.core.model import SCHEMA_VERSION, AnisaModel
from anisa.io.data import reload_direction_data

__all__ = ["PROJECT_SUFFIX", "load_project", "save_project"]

PROJECT_SUFFIX = ".anisa.json"
FORMAT = "anisa-project"
_HEADER = "_anisa_project"


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.suffix.lower() == ".json" else path.with_name(path.name + PROJECT_SUFFIX)


def _relative(path: Path, folder: Path) -> str | None:
    try:
        return os.path.relpath(path, folder)
    except ValueError:  # different drive on Windows
        return None


def save_project(model: AnisaModel, path: str | Path) -> Path:
    """Write ``model`` to a project file. Returns the path written.

    ``.anisa.json`` is appended unless ``path`` already ends in ``.json``.
    """
    path = _project_path(path).resolve()
    document = {
        _HEADER: {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "saved_by": f"anisa {anisa.__version__}",
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "model": model.to_dict(),
    }
    for direction in document["model"]["directions"]:
        source = direction.get("source")
        if source:
            absolute = Path(source["path"]).expanduser().resolve()
            relative = _relative(absolute, path.parent)
            source["path"] = relative if relative is not None else str(absolute)
            source["path_absolute"] = str(absolute)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write next to the target and rename, so a crash never leaves a half-written project. The
    # temporary file is created with open() (not mkstemp, which makes it private, mode 0600) so
    # the project gets the permissions the user's umask gives any new file.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _resolve_sources(model_dict: dict, folder: Path) -> None:
    for direction in model_dict.get("directions", []):
        source = direction.get("source")
        if not source or not source.get("path"):
            continue
        stored = Path(source["path"]).expanduser()
        candidate = stored if stored.is_absolute() else folder / stored
        fallback = source.pop("path_absolute", None)
        if not candidate.is_file() and fallback and Path(fallback).is_file():
            candidate = Path(fallback)
        source["path"] = str(candidate.resolve() if candidate.is_file() else candidate)


def load_project(path: str | Path, *, load_data: bool = True) -> AnisaModel:
    """Read a project file.

    Args:
        path: project file.
        load_data: also read the arrays of every direction from its data file. A data file that
            is missing, unreadable, or changed since the project was saved produces a
            ``UserWarning`` (the project still opens; that direction has no data).

    Raises:
        FileNotFoundError: the project file does not exist.
        ValueError: the file is not an Anisa project or is not valid JSON.
    """
    path = Path(path).expanduser().resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name} is not valid JSON: {error}") from error
    header = document.get(_HEADER) if isinstance(document, dict) else None
    if not isinstance(header, dict) or header.get("format") != FORMAT:
        raise ValueError(f"{path.name} is not an Anisa project file")
    if int(header.get("schema_version", 1)) > SCHEMA_VERSION:
        warnings.warn(
            f"{path.name} was written by a newer Anisa ({header.get('saved_by', '?')}); "
            "unknown settings are ignored",
            UserWarning,
            stacklevel=2,
        )
    model_dict = document.get("model", {})
    _resolve_sources(model_dict, path.parent)
    model = AnisaModel.from_dict(model_dict)
    if load_data:
        for direction in model.directions:
            reload_direction_data(direction, base_dir=path.parent)
    return model
