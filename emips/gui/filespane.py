"""The Files pane (PLAN M27 phase S3): the program's file area
(:mod:`emips.files`) — the files given with ``--file`` / Settings >
Exceptions & I/O > Files and the ones the run created, each with its size
and state, the one chosen shown (text, or a hex dump), the descriptors
open; Save File writes it as the run has left it.  Everything comes from
:meth:`Session.file_lines` / :meth:`Session.file_data`."""

from __future__ import annotations

from ..files import dump_bytes
from ..session import Session
from .i18n import tr
from .qt import QtWidgets, Signal
from .widgets import mono_font


class FilesView(QtWidgets.QWidget):
    #: Save File: the file's name in the area
    save_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        split = QtWidgets.QSplitter()
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(lambda _: self._show())
        split.addWidget(self.list)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(mono_font())
        split.addWidget(self.text)
        split.setStretchFactor(1, 3)
        lay.addWidget(split, 1)
        row = QtWidgets.QHBoxLayout()
        self.open = QtWidgets.QLabel()
        self.open.setFont(mono_font())
        row.addWidget(self.open, 1)
        self.save = QtWidgets.QPushButton(tr("Save File…"))
        self.save.setToolTip(tr("Write the chosen file as the run has left it"))
        self.save.clicked.connect(self._save)
        row.addWidget(self.save)
        lay.addLayout(row)

    def _names(self) -> list[str]:
        area = self._session.file_area if self._session is not None else None
        return sorted(area.data) if area is not None else []

    def refresh(self, s: Session) -> None:
        self._session = s
        area = s.file_area
        names = self._names()
        if not names:
            self.note.setText(tr("No files. Give a program files to open in Configure > "
                                 "Settings > Exceptions & I/O > Files (or --file NAME=PATH): "
                                 "MARS's syscalls 13-16 and RARS's open / read / write / "
                                 "close / lseek reach only these, and every run starts from "
                                 "them."))
        else:
            self.note.setText(tr("The program's files: syscalls 13-16 (MARS) or 1024 / 63 / "
                                 "64 / 57 / 62 (RARS) reach only these; every run starts "
                                 "from the given ones."))
        current = self.list.currentItem().data(256) if self.list.currentItem() else None
        self.list.blockSignals(True)
        self.list.clear()
        for name in names:
            item = QtWidgets.QListWidgetItem(
                f"{name}  ({len(area.data[name])} B, {tr(area.state(name))})")
            item.setData(256, name)
            self.list.addItem(item)
        if names:
            self.list.setCurrentRow(names.index(current) if current in names else 0)
        self.list.blockSignals(False)
        opened = s.services.files.open_lines() if s.services is not None and area else []
        self.open.setText("   ".join(opened))
        self._show()

    def _show(self) -> None:
        item = self.list.currentItem()
        self.save.setEnabled(item is not None)
        if item is None or self._session is None:
            self.text.setPlainText("")
            return
        self.text.setPlainText("\n".join(dump_bytes(self._session.file_data(item.data(256)))))

    def _save(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self.save_requested.emit(item.data(256))
