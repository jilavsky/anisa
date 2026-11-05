"""Test configuration: Qt runs offscreen, and GUI state never touches the user's home folder."""

import os

# Must be set before anything imports PySide6.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
