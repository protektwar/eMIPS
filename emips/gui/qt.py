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



def multimedia():
    """The binding's QtMultimedia (sound: the Sound pane), or None when it
    is not installed (it is an extra package on some systems)."""
    try:
        if API == "PyQt6":
            from PyQt6 import QtMultimedia
        else:                                             # pragma: no cover
            from PySide6 import QtMultimedia               # type: ignore
    except ImportError:
        return None
    return QtMultimedia


__all__ = ["API", "QtCore", "QtGui", "QtWidgets", "Signal", "multimedia"]
