"""
Package contract tests: version, layering, Qt import discipline, pyIrena API surface.

The source-scanning tests read files rather than importing them, so they run on a
machine with no Qt at all.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import anisa

PACKAGE = Path(anisa.__file__).parent
REPO = PACKAGE.parent
SHIM = PACKAGE / "gui" / "_qt.py"

#: ``from PySide6…`` / ``import PyQt6…`` at any indentation.
BINDING_IMPORT = re.compile(r"^\s*(?:from|import)\s+(PySide[26]|PyQt[456])\b", re.M)
#: Qt, pyqtgraph or matplotlib imported anywhere.
GUI_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(PySide[26]|PyQt[456]|pyqtgraph|matplotlib|anisa\.gui|pyirena\.gui)\b",
    re.M,
)


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_version_matches_pyproject():
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(REPO / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert anisa.__version__ == declared


def test_only_the_shim_imports_a_qt_binding():
    offenders = {
        str(p.relative_to(PACKAGE)): sorted(set(BINDING_IMPORT.findall(p.read_text("utf-8"))))
        for p in _python_files(PACKAGE)
        if p != SHIM and BINDING_IMPORT.search(p.read_text("utf-8"))
    }
    assert not offenders, f"import Qt through anisa.gui._qt, not directly: {offenders}"


def test_no_pyqt_anywhere():
    """House standard: PySide6 only. PyQt6 collides with PySide6 in a shared env."""
    for path in _python_files(PACKAGE):
        assert "PyQt" not in "".join(BINDING_IMPORT.findall(path.read_text("utf-8"))), path


@pytest.mark.parametrize("layer", ["core", "io"])
def test_core_and_io_have_no_gui_imports(layer):
    offenders = [
        str(p.relative_to(PACKAGE))
        for p in _python_files(PACKAGE / layer)
        if GUI_IMPORT.search(p.read_text("utf-8"))
    ]
    assert not offenders, f"{layer} must stay numpy/scipy/h5py only: {offenders}"


@pytest.mark.parametrize("module", ["api.py", "cli.py", "testsets.py", "demo.py"])
def test_headless_modules_have_no_gui_imports(module):
    assert not GUI_IMPORT.search((PACKAGE / module).read_text("utf-8")), module


def test_pyirena_reader_api_available():
    """Anisa reads data through pyIrena's public, Qt-free reader."""
    from pyirena.io import ScatteringRecord, discover_scattering, load_scattering

    assert callable(discover_scattering) and callable(load_scattering)
    assert {"q", "intensity", "uncertainty"} <= set(ScatteringRecord.__dataclass_fields__)


def test_qt_shim_loads_pyside6(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from anisa.gui import _qt

    assert _qt.QT_BINDING == "PySide6"
    app = _qt.QApplication.instance() or _qt.QApplication([])
    assert app is not None


def test_declared_console_scripts_resolve_to_callables():
    """A script declared in pyproject.toml must point at something that exists."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(REPO / "pyproject.toml", "rb") as fh:
        scripts = tomllib.load(fh)["project"].get("scripts", {})
    assert "anisa-gui" in scripts
    for name, target in scripts.items():
        module_name, _, function = target.partition(":")
        source = REPO / (module_name.replace(".", "/") + ".py")
        assert source.exists(), f"{name}: {source} does not exist"
        assert re.search(rf"^def {function}\(", source.read_text("utf-8"), re.M), (
            f"{name}: no {function}()"
        )
