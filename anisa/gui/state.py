"""
Small per-user GUI state: recent projects and last folders.

Stored as JSON in ``~/.anisa/state.json`` (``$ANISA_STATE_DIR`` overrides the folder, which the
tests use). Everything is best effort: a missing, unreadable or read-only state file never stops
the GUI, it only means nothing is remembered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MAX_RECENT = 8


def state_dir() -> Path:
    override = os.environ.get("ANISA_STATE_DIR")
    return Path(override) if override else Path.home() / ".anisa"


class UserState:
    """Recent projects and last-used folders."""

    def __init__(self, path: Path | None = None):
        self.path = path or state_dir() / "state.json"
        self.data: dict = {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self.data = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            self.data = {}

    # -- recent projects -------------------------------------------------------------------

    def recent_projects(self) -> list[str]:
        return [p for p in self.data.get("recent_projects", []) if isinstance(p, str)]

    def add_recent_project(self, path: str | Path) -> None:
        path = str(Path(path).resolve())
        recent = [p for p in self.recent_projects() if p != path]
        self.data["recent_projects"] = [path, *recent][:MAX_RECENT]
        self.save()

    # -- last folders ----------------------------------------------------------------------

    def folder(self, key: str) -> str:
        value = self.data.get(f"folder_{key}", "")
        # Path("") is the current directory and "exists", so an unset folder must be excluded
        # explicitly rather than by is_dir().
        if isinstance(value, str) and value and Path(value).is_dir():
            return value
        return str(Path.home())

    def set_folder(self, key: str, path: str | Path) -> None:
        folder = Path(path)
        folder = folder if folder.is_dir() else folder.parent
        self.data[f"folder_{key}"] = str(folder)
        self.save()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError:
            pass
