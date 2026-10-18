"""Graphical front end (PLAN M8): ``python3 -m emips gui [prog.asm]``.

:mod:`emips.gui.session` is the Qt-free model (tested headless); the Qt
widgets (PyQt6, or PySide6 through :mod:`emips.gui.qt`) only draw it.
Needs PyQt6 (or PySide6): the system's package (``python3-pyqt6``) or
``pip install ".[gui]"`` in a virtual environment.
"""
