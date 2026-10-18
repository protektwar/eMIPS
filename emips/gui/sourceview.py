"""The Source pane (PLAN M13 phase 3): the C file of a compiled program
(an ELF with a DWARF line table) at the line the selected core is on —
following it from file to file —, breakpoints on C lines (click the
gutter), and the call stack (Session.backtrace: the calls not returned
yet; click a frame to see its line).  For an assembled program the
editor is the source: the pane shows the call stack only.

Everything comes from the Session (source_position, source_file,
line_addresses, backtrace)."""

from __future__ import annotations

from pathlib import Path

from ..session import Session
from . import theme
from .i18n import tr
from .qt import QtCore, QtWidgets, Signal
from .widgets import CodeEditor, mono_font


class SourceView(QtWidgets.QWidget):
    """C source + call stack."""

    #: a breakpoint was set / cleared: the window redraws
    breakpoints_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        self._file: str | None = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.title = QtWidgets.QLabel()
        lay.addWidget(self.title)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.code = CodeEditor()
        self.code.setReadOnly(True)
        self.code.breakpoint_clicked.connect(self._toggle)
        split.addWidget(self.code)
        stack = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(stack)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QtWidgets.QLabel(tr("Call stack (innermost first)")))
        self.frames = QtWidgets.QListWidget()
        self.frames.setFont(mono_font())
        self.frames.currentRowChanged.connect(self._frame_chosen)
        sl.addWidget(self.frames)
        split.addWidget(stack)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        self._frames: list[dict] = []

    # -- drawing --------------------------------------------------------------------

    def refresh(self, s: Session) -> None:
        self._session = s
        if s.machine is None or s.program is None:
            self._show_file(None, [])
            self.title.setText(tr("no program"))
            self.frames.clear()
            return
        self._frames = s.backtrace()
        self.frames.blockSignals(True)
        self.frames.clear()
        for f in self._frames:
            where = ""
            if f["line"] is not None:
                where = f"  {Path(f['line'][0]).name}:{f['line'][1]}"
            elif f["source"] is not None and s.kind == "asm":
                where = "  " + tr("line {n}").format(n=f["source"].line)
            self.frames.addItem(f"#{f['level']}  0x{f['pc']:08x}  {f['function'] or '??'}"
                                f"{where}")
        self.frames.blockSignals(False)
        if not s.program.c_lines:
            self._show_file(None, [])
            self.title.setText(tr("an assembled program: its source is the editor; the "
                                  "call stack is below"))
            return
        pc = s.current_pc()
        pos = s.source_position(pc) if pc is not None else None
        if pos is not None:
            self._show_at(pos[0], pos[1])
        else:
            got = s.source_file()
            if got is not None and got[0] != self._file:
                self._show_file(got[0], got[1])
            self._mark(None)
            self.title.setText(tr("{file} — the next instruction has no C line (start-up, "
                                  "library or handler code)").format(
                file=Path(self._file).name if self._file else "?"))

    def _show_at(self, file: str, line: int, current: bool = True) -> None:
        s = self._session
        if file != self._file:
            got = s.source_file(next(iter(s.line_addresses(file, 1)), None))
            self._show_file(file, got[1] if got and got[0] == file else [])
        self._mark(line if current else None, None if current else line)
        self.title.setText(f"{Path(file).name}:{line}")

    def _show_file(self, file: str | None, text: list[str]) -> None:
        self._file = file
        self.code.setPlainText("\n".join(text) if text else
                               (tr("({file}: cannot read it)").format(file=file) if file
                                else ""))

    def _mark(self, line: int | None, other: int | None = None) -> None:
        s = self._session
        lines: dict[int, str] = {}
        if line:
            lines[line] = theme.colors().current_line
        if other:
            lines[other] = theme.colors().valid_fill
        self.code.highlight(lines)
        if line or other:
            self.code.show_line(line or other)
        bps = set()
        if s is not None and s.program is not None and self._file:
            for a in s.breakpoints:
                pos = s.program.c_lines.get(a)
                if pos is not None and pos[0] == self._file:
                    bps.add(pos[1])
        self.code.breakpoint_lines = bps
        self.code.gutter.update()

    # -- clicks ------------------------------------------------------------------------

    def _toggle(self, line: int) -> None:
        s = self._session
        if s is None or s.program is None or not self._file:
            return
        addrs = [a for a in s.line_addresses(self._file, line)
                 if s.program.c_lines.get(a) == (self._file, line)]
        if not addrs:
            return
        if any(a in s.breakpoints for a in addrs):
            s.breakpoints.difference_update(addrs)
        else:
            s.breakpoints.update(addrs)
        self.breakpoints_changed.emit()

    def _frame_chosen(self, row: int) -> None:
        if not 0 <= row < len(self._frames):
            return
        f = self._frames[row]
        if f["line"] is not None:
            self._show_at(f["line"][0], f["line"][1], current=row == 0)
