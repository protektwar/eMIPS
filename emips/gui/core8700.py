"""The P8700's pipeline seen from inside (LF14): each hart's window
(dispatched, not graduated: queued, executing, complete — wrong-path
instructions marked), the issue queues and busy units, the front end and
the predictors.  Everything comes from :class:`emips.pipe8700.P8700Pipeline`
(``rows`` / ``lines``) — nothing is re-executed here."""

from __future__ import annotations

from .i18n import tr
from ..session import Session
from . import theme
from .qt import QtWidgets
from .widgets import _set, mono_font

T = theme.colors

#: the stage colour a window entry gets by its state
_STATE_STAGE = {"queued": "ID", "dispatched": "ID", "executing": "EX", "complete": "WB"}


class Core8700View(QtWidgets.QWidget):
    """The window table above, the core's state (queues, units, front
    ends, predictors) below."""

    COLUMNS = ("hart", "#", "pc", "instruction", "unit", "state")

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
        p = getattr(s.pipeline, "core", None)
        if p is None:
            return
        self.title.setText(tr("P8700 core: cycle {c}, {n} graduated — the window "
                              "(dispatched instructions, oldest first)").format(
                                  c=p.cycles, n=p.retired))
        rows = p.rows()
        self.window.setRowCount(len(rows))
        c = T()
        for r, row in enumerate(rows):
            state = tr(row["state"]) + (f" ({row['left']})" if row["left"] else "")
            flags = [tr("wrong path")] if row["ghost"] else []
            flags += [tr("mispredicted")] if row["mispredicted"] else []
            flags += [tr("bonded")] if row["bonded"] else []
            if flags:
                state += " — " + ", ".join(flags)
            bg = c.error_line if row["ghost"] else c.stage[_STATE_STAGE[row["state"]]][0]
            for col, text in enumerate((str(row["hart"]), str(row["seq"]),
                                        f"{row['pc']:08x}", row["text"], row["unit"], state)):
                _set(self.window, r, col, text, bg)
        self.window.resizeColumnsToContents()
        self.state.setPlainText("\n".join(p.lines("queues") + p.lines("predict")[1:]
                                          + [ln for ln in p.lines("window")[1:]
                                             if ln.startswith("hart")]))
