"""The P8700's pipeline seen from inside (LF14): each hart's window
(dispatched, not graduated: queued, executing, complete — wrong-path
instructions marked), the issue queues and busy units, the front end and
the predictors.  Everything comes from :class:`emips.pipe8700.P8700Pipeline`
(``rows`` / ``lines``) — nothing is re-executed here."""

from __future__ import annotations

from .i18n import tr
from .. import coredraw
from ..session import Session
from . import theme
from .qt import QtWidgets
from .widgets import _set, mono_font

T = theme.colors

_STATE_STAGE = coredraw.STATE_STAGE


class Core8700View(QtWidgets.QWidget):
    """The window table above, the core's state (queues, units, front
    ends, predictors) below."""

    COLUMNS = coredraw.P8700_COLUMNS

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel()
        lay.addWidget(self.title)
        self.window = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.window.setFont(mono_font())
        self.window.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.window.setHorizontalHeaderLabels([tr(c) for c in self.COLUMNS])
        self.window.verticalHeader().setVisible(False)
        self.window.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.window, 3)
        self.state = QtWidgets.QPlainTextEdit()
        self.state.setReadOnly(True)
        self.state.setFont(mono_font())
        lay.addWidget(self.state, 2)

    def refresh(self, s: Session) -> None:
        v = coredraw.p8700_view(s, tr)
        if v is None:
            return
        self.title.setText(v["title"])
        self.window.setRowCount(len(v["rows"]))
        c = T()
        for r, (cells, band) in enumerate(v["rows"]):
            bg = c.error_line if band == "error" else c.stage[band[5:]][0]
            for col, text in enumerate(cells):
                _set(self.window, r, col, text, bg)
        self.window.resizeColumnsToContents()
        self.state.setPlainText(v["state"])
