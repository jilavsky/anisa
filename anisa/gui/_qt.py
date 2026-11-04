"""
anisa.gui._qt — the single Qt import point for the whole ``anisa.gui`` package.

PySide6 only (house standard): PyQt6 is never used, because the two bindings
collide in one environment shared with pyIrena and Bernardyn.

Two import styles are supported:

* flat names — ``from anisa.gui._qt import QWidget, Qt, Signal``
* submodules — ``from anisa.gui._qt import QtWidgets, QtCore, QtGui``

Add a missing class here rather than importing PySide6 in another module;
``tests/test_package.py`` enforces that.
"""

import os

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Signal, Slot
except ImportError as _pyside_error:  # pragma: no cover - only without PySide6
    # ImportError also covers an installed-but-unloadable PySide6 (architecture
    # mismatch, shiboken6 skew); `pyirena-doctor` (installed with pyirena) tells
    # the two cases apart.
    raise ImportError(
        "Anisa GUI needs PySide6. Install it with:  pip install anisa[gui]\n"
        "If PySide6 is installed but fails to load, run `pyirena-doctor` for a diagnosis."
    ) from _pyside_error

QT_BINDING = "PySide6"

# pyqtgraph picks its own binding and tries PyQt6 first; pin it to ours so a stale
# PyQt6 in the environment cannot be picked up. An explicit user setting wins.
os.environ.setdefault("PYQTGRAPH_QT_LIB", QT_BINDING)

# --- QtWidgets ---------------------------------------------------------------
QAbstractItemView = QtWidgets.QAbstractItemView
QApplication = QtWidgets.QApplication
QButtonGroup = QtWidgets.QButtonGroup
QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QDialog = QtWidgets.QDialog
QDialogButtonBox = QtWidgets.QDialogButtonBox
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QFrame = QtWidgets.QFrame
QGridLayout = QtWidgets.QGridLayout
QGroupBox = QtWidgets.QGroupBox
QHBoxLayout = QtWidgets.QHBoxLayout
QHeaderView = QtWidgets.QHeaderView
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QListWidget = QtWidgets.QListWidget
QListWidgetItem = QtWidgets.QListWidgetItem
QMainWindow = QtWidgets.QMainWindow
QMenu = QtWidgets.QMenu
QMenuBar = QtWidgets.QMenuBar
QMessageBox = QtWidgets.QMessageBox
QPlainTextEdit = QtWidgets.QPlainTextEdit
QProgressBar = QtWidgets.QProgressBar
QPushButton = QtWidgets.QPushButton
QRadioButton = QtWidgets.QRadioButton
QScrollArea = QtWidgets.QScrollArea
QSizePolicy = QtWidgets.QSizePolicy
QSpinBox = QtWidgets.QSpinBox
QSplitter = QtWidgets.QSplitter
QStatusBar = QtWidgets.QStatusBar
QTabWidget = QtWidgets.QTabWidget
QTableWidget = QtWidgets.QTableWidget
QTableWidgetItem = QtWidgets.QTableWidgetItem
QToolBar = QtWidgets.QToolBar
QToolButton = QtWidgets.QToolButton
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

# --- QtCore ------------------------------------------------------------------
Qt = QtCore.Qt
QObject = QtCore.QObject
QThread = QtCore.QThread
QTimer = QtCore.QTimer
QUrl = QtCore.QUrl

# --- QtGui -------------------------------------------------------------------
QAction = QtGui.QAction
QCloseEvent = QtGui.QCloseEvent
QColor = QtGui.QColor
QDesktopServices = QtGui.QDesktopServices
QDoubleValidator = QtGui.QDoubleValidator
QFont = QtGui.QFont
QIcon = QtGui.QIcon
QIntValidator = QtGui.QIntValidator
QKeySequence = QtGui.QKeySequence
QPen = QtGui.QPen

__all__ = [
    "QtWidgets",
    "QtCore",
    "QtGui",
    "Signal",
    "Slot",
    "QT_BINDING",
    # QtWidgets
    "QAbstractItemView",
    "QApplication",
    "QButtonGroup",
    "QCheckBox",
    "QComboBox",
    "QDialog",
    "QDialogButtonBox",
    "QDoubleSpinBox",
    "QFileDialog",
    "QFormLayout",
    "QFrame",
    "QGridLayout",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMenu",
    "QMenuBar",
    "QMessageBox",
    "QPlainTextEdit",
    "QProgressBar",
    "QPushButton",
    "QRadioButton",
    "QScrollArea",
    "QSizePolicy",
    "QSpinBox",
    "QSplitter",
    "QStatusBar",
    "QTabWidget",
    "QTableWidget",
    "QTableWidgetItem",
    "QToolBar",
    "QToolButton",
    "QVBoxLayout",
    "QWidget",
    # QtCore
    "Qt",
    "QObject",
    "QThread",
    "QTimer",
    "QUrl",
    # QtGui
    "QAction",
    "QCloseEvent",
    "QColor",
    "QDesktopServices",
    "QDoubleValidator",
    "QFont",
    "QIcon",
    "QIntValidator",
    "QKeySequence",
    "QPen",
]
