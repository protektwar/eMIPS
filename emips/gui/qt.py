"""Qt binding shim: PyQt6 (preferred), else PySide6.

Only the Qt6 APIs common to both are used; ``Signal`` is the binding's
signal class (``pyqtSignal`` / ``Signal``).
"""

from __future__ import annotations

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
    API = "PyQt6"
except ImportError:                                       # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets           # type: ignore
    Signal = QtCore.Signal                                 # type: ignore
    API = "PySide6"

__all__ = ["API", "QtCore", "QtGui", "QtWidgets", "Signal"]
