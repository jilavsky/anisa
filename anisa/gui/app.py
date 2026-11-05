"""Entry point of the Anisa GUI: ``anisa-gui [project.anisa.json]``."""

from __future__ import annotations

import argparse
import sys

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anisa-gui", description="Anisa: anisotropic SAS modelling."
    )
    parser.add_argument("project", nargs="?", help="project file (*.anisa.json) to open")
    args, qt_args = parser.parse_known_args(argv)
    from anisa.gui._qt import QApplication
    from anisa.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Anisa")
    app.setOrganizationName("anisa")
    window = MainWindow(project=args.project)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
