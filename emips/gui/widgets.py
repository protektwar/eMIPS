"""Qt widgets of the GUI: source editor, registers, memory, console,
pipeline chart and the instruction-details panel.  They only read a
:class:`~emips.gui.session.Session`."""

from __future__ import annotations

import html
from typing import Any

from . import theme
from .i18n import N_, tr
from .qt import QtCore, QtGui, QtWidgets, Signal
from ..mmu import cca_name
from ..session import Session

Qt = QtCore.Qt

T = theme.colors

STAGE_NAMES = {"IF": N_("fetch"), "ID": N_("decode"), "EX": N_("execute"),
               "MEM": N_("memory"), "WB": N_("write-back")}


def mono_font() -> QtGui.QFont:
    """A monospace font (built on demand: needs the QApplication)."""
    f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    f.setPointSize(10)
    return f


# --------------------------------------------------------------- source editor

class _Gutter(QtWidgets.QWidget):
    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, event) -> None:
        self.editor.paint_gutter(event)

    def mousePressEvent(self, event) -> None:
        block = self.editor.cursorForPosition(QtCore.QPoint(0, int(event.position().y()))).block()
        self.editor.breakpoint_clicked.emit(block.blockNumber() + 1)


class CodeEditor(QtWidgets.QPlainTextEdit):
    """Assembly source with line numbers, breakpoint dots (click the
    gutter) and highlighted lines (current instruction / pipeline stages)."""

    breakpoint_clicked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setFont(mono_font())
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.gutter = _Gutter(self)
        self.breakpoint_lines: set[int] = set()
        #: a multi-core system: line -> the other cores whose next
        #: instruction is on it (drawn in the gutter, left of the numbers)
        self.core_marks: dict[int, list[int]] = {}
        #: room for the marks (on while the program has several cores)
        self.show_core_marks = False
        self.blockCountChanged.connect(lambda _: self._update_margin())
        self.updateRequest.connect(self._on_update)
        self._update_margin()
        self._tab_stops()

    # -- editing helpers (Edit menu) -------------------------------------------

    def _tab_stops(self) -> None:
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 8)

    def set_point_size(self, size: int) -> None:
        """Editor font size (View > Zoom)."""
        f = self.font()
        f.setPointSize(max(6, min(size, 40)))
        self.setFont(f)
        self._tab_stops()
        self._update_margin()

    def _line_range(self) -> tuple[int, int]:
        """First and last line (0-based) touched by the selection / cursor."""
        c, doc = self.textCursor(), self.document()
        first = doc.findBlock(c.selectionStart()).blockNumber()
        end = doc.findBlock(c.selectionEnd())
        last = end.blockNumber()
        if c.hasSelection() and c.selectionEnd() == end.position() and last > first:
            last -= 1                           # selection ends at column 0
        return first, last

    def edit_lines(self, change) -> None:
        """Replace the selected lines by ``change(lines)`` as one undo step,
        keeping them selected."""
        if self.isReadOnly():
            return
        doc = self.document()
        first, last = self._line_range()
        lines = [doc.findBlockByNumber(i).text() for i in range(first, last + 1)]
        new = change(lines)
        if new == lines:
            return
        start = doc.findBlockByNumber(first).position()
        end_block = doc.findBlockByNumber(last)
        cur = QtGui.QTextCursor(doc)
        cur.setPosition(start)
        cur.setPosition(end_block.position() + end_block.length() - 1,
                        QtGui.QTextCursor.MoveMode.KeepAnchor)
        cur.beginEditBlock()
        cur.insertText("\n".join(new))
        cur.endEditBlock()
        sel = QtGui.QTextCursor(doc)
        sel.setPosition(start)
        sel.setPosition(start + len("\n".join(new)), QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(sel)

    def toggle_comment(self) -> None:
        """Comment the selected lines with ``#`` (or uncomment them when
        they all are comments)."""
        def change(lines: list[str]) -> list[str]:
            code = [ln for ln in lines if ln.strip()]
            if code and all(ln.lstrip().startswith("#") for ln in code):
                out = []
                for ln in lines:
                    i = ln.find("#")
                    if i < 0:
                        out.append(ln)
                        continue
                    j = i + 1 + (ln[i + 1:i + 2] == " ")
                    out.append(ln[:i] + ln[j:])
                return out
            return [f"# {ln}" if ln.strip() else ln for ln in lines]
        self.edit_lines(change)

    def indent(self) -> None:
        self.edit_lines(lambda lines: ["\t" + ln if ln.strip() else ln for ln in lines])

    def unindent(self) -> None:
        def strip(ln: str) -> str:
            if ln.startswith("\t"):
                return ln[1:]
            return ln[min(len(ln) - len(ln.lstrip(" ")), 8):]
        self.edit_lines(lambda lines: [strip(ln) for ln in lines])

    def go_to_line(self, line: int) -> None:
        self.show_line(max(1, min(line, self.blockCount())))
        self.setFocus()

    def current_line(self) -> int:
        return self.textCursor().blockNumber() + 1

    def marks_width(self) -> int:
        return self.fontMetrics().horizontalAdvance("C000") + 4 if self.show_core_marks else 0

    def gutter_width(self) -> int:
        digits = max(3, len(str(self.blockCount())))
        return 18 + self.fontMetrics().horizontalAdvance("9") * digits + self.marks_width()

    def set_core_marks(self, marks: dict[int, list[int]], show: bool) -> None:
        """Mark where the other cores are (``{line: [cores]}``)."""
        if show != self.show_core_marks:
            self.show_core_marks = show
            self._update_margin()
        self.core_marks = marks
        self.gutter.update()

    def _update_margin(self) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _on_update(self, rect, dy) -> None:
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.gutter.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def gutter_colors(self) -> tuple[QtGui.QColor, QtGui.QColor]:
        """(background, number colour) of the line-number gutter, from the
        editor's palette: a shade off the text background, numbers halfway
        between text and background — so light and dark themes both work."""
        pal = self.palette()
        base = pal.color(QtGui.QPalette.ColorRole.Base)
        text = pal.color(QtGui.QPalette.ColorRole.Text)
        bg = base.lighter(135) if base.lightness() < 128 else base.darker(106)
        if bg == base:                          # pure black: lighter() has no effect
            bg = QtGui.QColor(28, 28, 28)
        mid = QtGui.QColor((text.red() + bg.red()) // 2, (text.green() + bg.green()) // 2,
                           (text.blue() + bg.blue()) // 2)
        return bg, mid

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.PaletteChange:
            self.gutter.update()

    def paint_gutter(self, event) -> None:
        painter = QtGui.QPainter(self.gutter)
        bg, number = self.gutter_colors()
        painter.fillRect(event.rect(), bg)
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        height = round(self.blockBoundingRect(block).height())
        mw = self.marks_width()
        bold = QtGui.QFont(self.font())
        bold.setBold(True)
        while block.isValid() and top <= event.rect().bottom():
            line = block.blockNumber() + 1
            if line in self.core_marks:
                painter.setPen(number)
                painter.setFont(bold)
                painter.drawText(2, top, mw, height, int(Qt.AlignmentFlag.AlignLeft),
                                 "C" + "".join(str(c) for c in self.core_marks[line]))
                painter.setFont(self.font())
            if line in self.breakpoint_lines:
                painter.setBrush(QtGui.QColor(T().breakpoint))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(mw + 4, top + height // 2 - 5, 10, 10)
            painter.setPen(number)
            painter.drawText(0, top, self.gutter.width() - 4, height,
                             int(Qt.AlignmentFlag.AlignRight), str(line))
            block = block.next()
            top += height
        painter.end()

    def highlight(self, lines: dict[int, str]) -> None:
        """``{line: colour}`` full-width line highlights."""
        sels = []
        for line, color in lines.items():
            block = self.document().findBlockByNumber(line - 1)
            if not block.isValid():
                continue
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.format.setBackground(QtGui.QColor(color))
            sel.format.setForeground(QtGui.QColor(T().highlight_text))
            sel.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            sel.cursor = QtGui.QTextCursor(block)
            sels.append(sel)
        self.setExtraSelections(sels)

    def show_line(self, line: int) -> None:
        block = self.document().findBlockByNumber(line - 1)
        if block.isValid():
            cursor = QtGui.QTextCursor(block)
            self.setTextCursor(cursor)
            self.ensureCursorVisible()


class FindBar(QtWidgets.QWidget):
    """Find / replace under the editor (Ctrl+F, Ctrl+H; Esc closes)."""

    def __init__(self, editor: CodeEditor) -> None:
        super().__init__()
        self.editor = editor
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(2, 2, 2, 2)
        self.find_text = QtWidgets.QLineEdit()
        self.find_text.setPlaceholderText(tr("Find"))
        self.replace_text = QtWidgets.QLineEdit()
        self.replace_text.setPlaceholderText(tr("Replace with"))
        self.case = QtWidgets.QCheckBox(tr("Match case"))
        self.status = QtWidgets.QLabel()
        prev_b = QtWidgets.QPushButton(tr("Previous"))
        next_b = QtWidgets.QPushButton(tr("Next"))
        self.repl_b = QtWidgets.QPushButton(tr("Replace"))
        self.all_b = QtWidgets.QPushButton(tr("Replace all"))
        close_b = QtWidgets.QToolButton()
        close_b.setText("✕")
        close_b.setToolTip(tr("Close (Esc)"))
        grid.addWidget(self.find_text, 0, 0)
        grid.addWidget(prev_b, 0, 1)
        grid.addWidget(next_b, 0, 2)
        grid.addWidget(self.case, 0, 3)
        grid.addWidget(close_b, 0, 4)
        grid.addWidget(self.replace_text, 1, 0)
        grid.addWidget(self.repl_b, 1, 1)
        grid.addWidget(self.all_b, 1, 2)
        grid.addWidget(self.status, 1, 3, 1, 2)
        self.find_text.returnPressed.connect(self.find_next)
        self.replace_text.returnPressed.connect(self.replace)
        prev_b.clicked.connect(self.find_previous)
        next_b.clicked.connect(self.find_next)
        self.repl_b.clicked.connect(self.replace)
        self.all_b.clicked.connect(self.replace_all)
        close_b.clicked.connect(self.close_bar)
        self.hide()

    def open(self, replace: bool = False) -> None:
        sel = self.editor.textCursor().selectedText()
        if sel and "\u2029" not in sel:
            self.find_text.setText(sel)
        for w in (self.replace_text, self.repl_b, self.all_b):
            w.setVisible(replace and not self.editor.isReadOnly())
        self.status.clear()
        self.show()
        self.find_text.setFocus()
        self.find_text.selectAll()

    def close_bar(self) -> None:
        self.hide()
        self.editor.setFocus()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close_bar()
        else:
            super().keyPressEvent(event)

    def _flags(self, backward: bool = False):
        f = QtGui.QTextDocument.FindFlag(0)
        if self.case.isChecked():
            f |= QtGui.QTextDocument.FindFlag.FindCaseSensitively
        if backward:
            f |= QtGui.QTextDocument.FindFlag.FindBackward
        return f

    def find_next(self, backward: bool = False) -> bool:
        text = self.find_text.text()
        if not text:
            return False
        ed = self.editor
        if ed.find(text, self._flags(backward)):
            self.status.clear()
            return True
        c = ed.textCursor()                      # wrap around
        c.movePosition(QtGui.QTextCursor.MoveOperation.End if backward
                       else QtGui.QTextCursor.MoveOperation.Start)
        ed.setTextCursor(c)
        found = ed.find(text, self._flags(backward))
        self.status.setText(tr("wrapped around") if found else tr("not found"))
        return found

    def find_previous(self) -> bool:
        return self.find_next(backward=True)

    def _matches(self, s: str) -> bool:
        t = self.find_text.text()
        return s == t if self.case.isChecked() else s.lower() == t.lower()

    def replace(self) -> None:
        c = self.editor.textCursor()
        if c.hasSelection() and self._matches(c.selectedText()):
            c.insertText(self.replace_text.text())
        self.find_next()

    def replace_all(self) -> int:
        text = self.find_text.text()
        if not text or self.editor.isReadOnly():
            return 0
        doc, n = self.editor.document(), 0
        cur = QtGui.QTextCursor(doc)
        cur.beginEditBlock()
        found = doc.find(text, 0, self._flags())
        while not found.isNull():
            found.insertText(self.replace_text.text())
            n += 1
            found = doc.find(text, found.position(), self._flags())
        cur.endEditBlock()
        self.status.setText(tr("{n} replaced").format(n=n))
        return n


def _swatch(color: str, text: str) -> str:
    return (f'<span style="background-color:{color}; color:{T().highlight_text};">'
            f'&nbsp;{html.escape(text).replace(" ", "&nbsp;")}&nbsp;</span>')


class EditorLegend(QtWidgets.QLabel):
    """What the editor's line colours mean, for the current mode."""

    def __init__(self) -> None:
        super().__init__()
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.set_mode("instruction")

    def set_mode(self, mode: str) -> None:
        dot = (f'<span style="color:{T().breakpoint};">&#9679;</span> '
               + html.escape(tr("breakpoint (click a line number)")))
        if mode == "pipeline":
            stages = " ".join(_swatch(T().stage[st][0], f"{st} {tr(name)}")
                              for st, name in STAGE_NAMES.items())
            text = (html.escape(tr("Line colour = pipeline stage of its instruction in "
                                   "the last cycle:")) + f"<br>{stages}<br>{dot}")
        else:
            text = (f"{_swatch(T().current_line, tr('next instruction'))} "
                    + html.escape(tr("not executed yet (Step runs it)"))
                    + f" &nbsp; {_swatch(T().error_line, tr('assembly error'))}<br>{dot}")
        self.setText(text)


# --------------------------------------------------------------- tables

def _table(headers: list[str]) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setFont(mono_font())
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _signed(value: int, bits: int) -> int:
    """A ``bits``-wide register value as a signed number."""
    return value - (1 << bits) if value >> (bits - 1) & 1 else value


def _set(t: QtWidgets.QTableWidget, r: int, c: int, text: str,
         bg: str | None = None, bold: bool = False) -> None:
    item = t.item(r, c)
    if item is None:
        item = QtWidgets.QTableWidgetItem()
        t.setItem(r, c, item)
    item.setText(text)
    item.setBackground(QtGui.QBrush(QtGui.QColor(bg)) if bg else QtGui.QBrush())
    item.setForeground(QtGui.QBrush(QtGui.QColor(T().highlight_text)) if bg else QtGui.QBrush())
    f = item.font()
    f.setBold(bold)
    item.setFont(f)


class RegisterView(QtWidgets.QTabWidget):
    """GPRs (changed ones highlighted), PC/HI/LO/CP0/FCSR, the FPRs, the
    interrupt system and shadow sets, and EJTAG (Debug Mode)."""

    #: the Debug tab's button: a debug interrupt (the DINT input)
    dint_requested = Signal()
    #: the Debug tab's probe boxes: (connected, ProbTrap)
    probe_changed = Signal(bool, bool)

    def __init__(self) -> None:
        super().__init__()
        self.gpr = _table([tr("reg"), tr("hex"), tr("decimal")])
        self.gpr.setRowCount(32)
        self.special = _table([tr("reg"), tr("hex")])
        self.fpr = _table([tr("reg"), tr("hex"), tr("float / double")])
        self.fpr.setRowCount(32)
        self.addTab(self.gpr, tr("GPRs"))
        self.addTab(self.special, tr("PC / CP0"))
        self.addTab(self.fpr, tr("FPU"))
        # the DSP Module (Session.dsp_rows): DSPControl field by field, ac0-ac3
        self.dsp = _table([tr("reg"), tr("hex"), tr("meaning")])
        self.dsp_tab = self.addTab(self.dsp, tr("DSP"))
        self.setTabToolTip(self.dsp_tab, tr("The DSP Module: DSPControl and the four "
                                            "accumulators (Status.MX enables it)"))
        self._dsp_prev: list[int] = []
        # the interrupt system (mode, request, next vector: Session.
        # interrupt_lines) and the GPR shadow sets (Release 2+, SRSCtl / SRSMap)
        self.sets = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.sets)
        lay.setContentsMargins(2, 2, 2, 2)
        self.int_text = QtWidgets.QLabel()
        self.int_text.setWordWrap(True)
        self.int_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.int_text)
        self.set_box = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(self.set_box)
        box.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("Shadow set")))
        self.set_choice = QtWidgets.QComboBox()
        row.addWidget(self.set_choice, 1)
        box.addLayout(row)
        self.set_info = QtWidgets.QLabel()
        self.set_info.setWordWrap(True)
        box.addWidget(self.set_info)
        self.set_table = _table([tr("reg"), tr("hex"), tr("decimal")])
        self.set_table.setRowCount(32)
        box.addWidget(self.set_table, 1)
        lay.addWidget(self.set_box, 1)
        self.sets_tab = self.addTab(self.sets, tr("Interrupts"))
        # EJTAG (Session.debug_lines): Debug Mode, Debug / DEPC / DESAVE, the DCR
        self.debug = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.debug)
        lay.setContentsMargins(2, 2, 2, 2)
        self.debug_text = QtWidgets.QLabel()
        self.debug_text.setWordWrap(True)
        self.debug_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.debug_text)
        self.dint = QtWidgets.QPushButton(tr("Debug Interrupt (DINT)"))
        self.dint.setToolTip(tr("Request a debug interrupt, as an EJTAG probe does: the core "
                                "enters Debug Mode at the debug vector before its next "
                                "instruction"))
        self.dint.clicked.connect(self.dint_requested.emit)
        lay.addWidget(self.dint, 0, Qt.AlignmentFlag.AlignLeft)
        self.probe_box = QtWidgets.QCheckBox(tr("EJTAG probe connected (ECR.ProbEn)"))
        self.probe_box.setToolTip(tr("A probe on the TAP serves EJTAG memory (dmseg); "
                                     "changing it restarts the program"))
        self.trap_box = QtWidgets.QCheckBox(tr("Debug handler from the probe (ProbTrap: "
                                               "0xff200200)"))
        for box in (self.probe_box, self.trap_box):
            box.toggled.connect(self._probe_toggled)
            lay.addWidget(box)
        lay.addStretch(1)
        self.debug_tab = self.addTab(self.debug, tr("Debug"))
        self._session: Session | None = None
        self._digits = 8                        # (MIPS64: 16 — the columns follow)
        self.set_choice.currentIndexChanged.connect(self._show_set)

    def refresh(self, s: Session) -> None:
        if s.machine is None:                   # no program: nothing to show
            for table in (self.gpr, self.fpr):
                for r in range(table.rowCount()):
                    for c in range(1, table.columnCount()):
                        _set(table, r, c, "")
            self.special.setRowCount(0)
            self.setTabVisible(self.sets_tab, False)
            self.setTabVisible(self.dsp_tab, False)
            self.setTabVisible(self.debug_tab, False)
            return
        self._session = s
        self._refresh_sets(s)
        self._refresh_dsp(s)
        self.setTabVisible(self.debug_tab, True)
        self.debug_text.setText("\n".join(ln.strip() for ln in s.debug_lines()))
        self.dint.setEnabled(not s.machine.debug_mode and not s.halted)
        for box, on in ((self.probe_box, s.probe is not None),
                        (self.trap_box, s.probe is not None and s.probe.trap)):
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
        self.trap_box.setEnabled(s.probe is not None)
        d = s.hex_digits()                      # a GPR's (MIPS64: 16 digits)
        for i, (name, value, changed) in enumerate(s.registers()):
            bg = T().current_line if changed else None
            _set(self.gpr, i, 0, name, bg, changed)
            _set(self.gpr, i, 1, f"0x{value:0{d}x}", bg, changed)
            _set(self.gpr, i, 2, str(_signed(value, 4 * d)), bg)
        rows = s.special_registers()
        self.special.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            _set(self.special, i, 0, name)
            _set(self.special, i, 1, f"0x{value:0{s.hex_digits(name)}x}")
        if d != self._digits:                   # MIPS32 <-> MIPS64: the values' width
            self._digits = d
            for t in (self.gpr, self.special, self.set_table):
                t.resizeColumnToContents(0)
                t.resizeColumnToContents(1)
        import struct
        fpu = s.machine.has_fpu
        self.setTabToolTip(self.indexOf(self.fpr), "" if fpu else tr(
            "This core has no FPU (Config1.FP = 0): Status.CU1 reads 0, every FPU "
            "instruction is Coprocessor Unusable"))
        self.fpr.setEnabled(fpu)
        for i, raw in enumerate(s.machine.fpr):
            if not fpu:
                _set(self.fpr, i, 0, f"$f{i}")
                _set(self.fpr, i, 1, tr("no FPU"))
                _set(self.fpr, i, 2, "")
                continue
            lo = raw & 0xFFFFFFFF
            single = struct.unpack(">f", lo.to_bytes(4, "big"))[0]
            dbl = struct.unpack(">d", raw.to_bytes(8, "big"))[0]
            _set(self.fpr, i, 0, f"$f{i}")
            _set(self.fpr, i, 1, f"0x{raw:016x}")
            _set(self.fpr, i, 2, f"{single:.7g} / {dbl:.15g}")


    def _refresh_dsp(self, s: Session) -> None:
        """The DSP tab, when the core has the DSP Module; a value that
        changed since the last refresh is highlighted."""
        self.setTabVisible(self.dsp_tab, bool(s.machine.dsp))
        if not s.machine.dsp:
            return
        rows = s.dsp_rows()
        self.dsp.setRowCount(len(rows))
        changed = len(self._dsp_prev) == len(rows)
        for i, (name, value, note) in enumerate(rows):
            hot = changed and self._dsp_prev[i] != value
            bg = T().current_line if hot else None
            _set(self.dsp, i, 0, name, bg, hot)
            digits = 16 if s.wide and name.startswith("ac") else 8
            _set(self.dsp, i, 1, f"0x{value:0{digits}x}" if not name.startswith("  ")
                 else (f"{value:04b}" if "ccond" in name else
                       f"0x{value:02x}" if "ouflag" in name else str(value)), bg, hot)
            _set(self.dsp, i, 2, tr(note) if note else "", bg)
        self._dsp_prev = [v for _, v, _ in rows]

    def _probe_toggled(self, *_args) -> None:
        on = self.probe_box.isChecked()
        self.probe_changed.emit(on, on and self.trap_box.isChecked())

    def _refresh_sets(self, s: Session) -> None:
        st = s.interrupt_state()
        n = st["sets"]
        self.setTabVisible(self.sets_tab, True)
        self.int_text.setText("\n".join(          # (SRSMap: under the set chooser)
            ln.strip() for ln in s.interrupt_lines() if "SRSMap:" not in ln))
        self.set_box.setVisible(n > 1)
        if n == 1:
            return
        want = self.set_choice.currentIndex()
        self.set_choice.blockSignals(True)
        self.set_choice.clear()
        for k in range(n):
            mark = [tr("current") if k == st["css"] else "",
                    tr("previous") if k == st["pss"] and k != st["css"] else ""]
            label = ", ".join(x for x in mark if x)
            self.set_choice.addItem(tr("set {n}").format(n=k) + (f" ({label})" if label
                                                                  else ""))
        self.set_choice.setCurrentIndex(want if 0 <= want < n else st["css"])
        self.set_choice.blockSignals(False)
        self.set_info.setText(
            tr("EIC mode: the controller gives the set (SRSCtl.EICSS {e})").format(
                e=st["eicss"]) if st["mode"] == "eic" else
            tr("SRSMap, vector → set: {m}").format(
                m=" ".join(f"{v}→{k}" for v, k in enumerate(st["srsmap"]))))
        self._show_set()

    def _show_set(self, *_args) -> None:
        s = self._session
        k = self.set_choice.currentIndex()
        if s is None or s.machine is None or k < 0 or k >= len(s.machine.shadow):
            return
        d = s.hex_digits()
        for i, (name, value) in enumerate(s.shadow_set(k)):
            _set(self.set_table, i, 0, name)
            _set(self.set_table, i, 1, f"0x{value:0{d}x}")
            _set(self.set_table, i, 2, str(_signed(value, 4 * d)))


class MemoryView(QtWidgets.QWidget):
    """Hex dump (4 words per row) from an address or a segment, by
    virtual address (as the program sees it: through the data cache, when
    it holds the line) or physical address (the RAM, after the MMU,
    bypassing the caches).  MIPS64: 64-bit addresses, and doublewords (2
    per row, as LD loads them) when *doublewords* is on."""

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.segment = QtWidgets.QComboBox()
        self.segment.addItems([".data (0x10010000)", ".text (0x00400000)",
                               tr("stack ($sp)"), "MMIO (0xffff0000)"])
        self.address = QtWidgets.QLineEdit()
        self.address.setPlaceholderText(tr("address, e.g. 0x10010000"))
        self.physical = QtWidgets.QCheckBox(tr("physical"))
        self.physical.setToolTip(tr(
            "Show physical addresses: the RAM after the MMU translated the "
            "virtual address, without the caches (a typed address is then "
            "physical)"))
        self.doubles = QtWidgets.QCheckBox(tr("doublewords"))
        self.doubles.setToolTip(tr("MIPS64: show 64-bit doublewords (as LD loads them) "
                                   "instead of words"))
        self.doubles.setChecked(True)
        self.doubles.setVisible(False)
        bar.addWidget(self.segment)
        bar.addWidget(self.address)
        bar.addWidget(self.physical)
        bar.addWidget(self.doubles)
        lay.addLayout(bar)
        self.table = _table([tr("address"), "+0", "+4", "+8", "+c", "ascii"])
        lay.addWidget(self.table)
        self.segment.currentIndexChanged.connect(lambda _: self.address.clear())
        self._session: Session | None = None
        self.address.returnPressed.connect(lambda: self._session and self.refresh(self._session))
        self.physical.toggled.connect(lambda _: self._session and self.refresh(self._session))
        self.doubles.toggled.connect(lambda _: self._session and self.refresh(self._session))
        self._layout: bool | None = None         # doublewords shown?

    def start(self, s: Session) -> int:
        text = self.address.text().strip()
        if text:
            try:
                return int(text, 0)
            except ValueError:
                pass
        idx = self.segment.currentIndex()
        va = [0x10010000, 0x00400000, (s.machine.read_reg(29) - 64) if s.machine else 0,
              0xFFFF0000][idx]
        if self.physical.isChecked() and s.machine is not None:
            pa = s.machine.translate(va).pa
            return va if pa is None else pa
        return va

    def refresh(self, s: Session) -> None:
        self._session = s
        if s.machine is None:
            self.table.setRowCount(0)
            return
        rows = s.memory_words(self.start(s), 32 if self.segment.currentIndex() != 3 else 1,
                              physical=self.physical.isChecked())
        self.doubles.setVisible(s.wide)
        doubles = s.wide and self.doubles.isChecked()
        fresh = doubles != self._layout
        if fresh:                                # the columns: 4 words or 2 doublewords
            self._layout = doubles
            heads = ["+0", "+8"] if doubles else ["+0", "+4", "+8", "+c"]
            self.table.setColumnCount(len(heads) + 2)
            self.table.setHorizontalHeaderLabels([tr("address")] + heads + ["ascii"])
        self.table.setRowCount(len(rows))
        big = s.machine.big_endian
        for r, (addr, words) in enumerate(rows):
            _set(self.table, r, 0, f"0x{addr:016x}" if addr >> 32 else f"0x{addr:08x}")
            chars = "".join("    " if w is None else
                            "".join(chr(b) if 32 <= b < 127 else "." for b in w.to_bytes(4, "big"))
                            for w in words)
            if doubles:                          # as LD loads them (the byte order)
                cells = []
                for hi, lo in ((words[0], words[1]), (words[2], words[3])):
                    if not big:
                        hi, lo = lo, hi
                    cells.append("-" * 16 if hi is None or lo is None else f"{hi:08x}{lo:08x}")
            else:
                cells = ["--------" if w is None else f"{w:08x}" for w in words]
            for c, text in enumerate(cells):
                _set(self.table, r, c + 1, text)
            _set(self.table, r, len(cells) + 1, chars)
        if fresh and rows:                       # (16-digit cells: wide enough)
            for c in range(self.table.columnCount() - 1):
                self.table.resizeColumnToContents(c)


class ConsoleView(QtWidgets.QWidget):
    """Program output, a line for read_* syscalls and keys for the MMIO
    keyboard."""

    submit_line = Signal(str)
    submit_keys = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(mono_font())
        lay.addWidget(self.out)
        bar = QtWidgets.QHBoxLayout()
        self.line = QtWidgets.QLineEdit()
        self.line.setPlaceholderText(tr("input for read_int / read_string ... (Enter)"))
        self.keys = QtWidgets.QLineEdit()
        self.keys.setPlaceholderText(tr("MMIO keyboard: type, Enter sends the keys "
                                        "(and Enter)"))
        bar.addWidget(self.line)
        bar.addWidget(self.keys)
        lay.addLayout(bar)
        self.line.returnPressed.connect(self._line)
        self.keys.returnPressed.connect(self._keys)

    def _line(self) -> None:
        self.submit_line.emit(self.line.text())
        self.line.clear()

    def _keys(self) -> None:
        # a keyboard: the Enter key that sends the line is a key too ('\n')
        self.submit_keys.emit(self.keys.text() + "\n")
        self.keys.clear()

    def refresh(self, s: Session) -> None:
        if self.out.toPlainText() != s.output:
            self.out.setPlainText(s.output)
            self.out.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def ask_input(self, ask: bool, focus: bool = True) -> None:
        self.asking = ask
        self.line.setStyleSheet(f"background: {T().current_line}; color: {T().highlight_text};"
                                if ask else "")
        if ask and focus:
            self.line.setFocus()

    def restyle(self) -> None:
        if getattr(self, "asking", False):
            self.ask_input(True, focus=False)


class DisassemblyView(QtWidgets.QTableWidget):
    """Address, word, disassembly, source line; current / breakpoints marked."""

    def __init__(self) -> None:
        super().__init__(0, 4)
        self.setHorizontalHeaderLabels([tr("address"), tr("word"), tr("instruction"),
                                        tr("line")])
        self.verticalHeader().setVisible(False)
        self.setFont(mono_font())
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.horizontalHeader().setStretchLastSection(True)
        self._rows: list[int] = []
        self._cache: list = []
        self._shown: int | None = None

    def refresh(self, s: Session) -> None:
        rows = s.disassembly()
        if rows != self._cache:                 # new program / register names
            self._cache = rows
            self.setRowCount(len(rows))
            self._rows = [a for a, *_ in rows]
            c_lines = s.c_text.splitlines() if s.c_text is not None else None
            last = None
            prog = s.program
            for r, (a, w, text, line) in enumerate(rows):
                _set(self, r, 0, f"0x{a:08x}")
                _set(self, r, 1, prog.encoding(a, w).rstrip() if prog is not None
                     else f"{w:08x}")           # (nanoMIPS: its halfwords)
                _set(self, r, 2, text)
                if line is None:
                    _set(self, r, 3, "")
                elif c_lines is not None:       # a C program: the C line's text, once
                    code = c_lines[line - 1].strip() if 0 < line <= len(c_lines) else ""
                    _set(self, r, 3, f"{line}: {code}" if line != last else f"{line}")
                else:
                    _set(self, r, 3, str(line))
                last = line
            self.resizeColumnsToContents()
            self._shown = None
        stage_of = {pc: st for st, pc in s.stage_pcs().items()}
        cur = s.current_pc()
        if cur != self._shown and cur in self._rows:   # keep the next instruction in view
            self._shown = cur
            self.scrollToItem(self.item(self._rows.index(cur), 0),
                              QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible)
        for r, a in enumerate(self._rows):
            bg = None
            if a in stage_of:
                bg = T().stage[stage_of[a]][0]
            elif a == cur:
                bg = T().current_line
            for c in range(4):
                item = self.item(r, c)
                item.setBackground(QtGui.QBrush(QtGui.QColor(bg)) if bg else QtGui.QBrush())
                fg = T().breakpoint if a in s.breakpoints else T().highlight_text if bg else None
                item.setForeground(QtGui.QBrush(QtGui.QColor(fg)) if fg else QtGui.QBrush())


class PipelineChart(QtWidgets.QTableWidget):
    """Instructions (rows) x cycles (columns), coloured by stage; ``--``
    marks a stall in ID.  Shows the last ``window`` cycles."""

    def __init__(self, window: int = 24) -> None:
        super().__init__(0, 0)
        self.window = window
        self.setFont(mono_font())
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setDefaultSectionSize(20)
        self.horizontalHeader().setDefaultSectionSize(34)

    def refresh(self, s: Session) -> None:
        if s.pipeline is None:
            self.setRowCount(0)
            self.setColumnCount(0)
            return
        hist = s.pipeline.history[-self.window:]
        rows: dict[int, tuple[int, str, dict[int, str]]] = {}
        cycles = [t.cycle for t in hist]
        slots = getattr(s.pipeline, "slots", None)
        if slots is not None:                   # the 24K: every instruction's stages
            first = cycles[0] if cycles else 0
            for sl in reversed(slots):
                cells = {c: st for c, st in sl.timeline.items() if c >= first}
                if cells:
                    rows[sl.seq] = (sl.pc, sl.text, cells)
                elif sl.fetched < first - 64:
                    break
        else:
            for t in hist:
                for st, v in t.stages.items():
                    if v is not None:
                        cell = "--" if st == "ID" and t.stall else st
                        rows.setdefault(v.seq, (v.pc, v.text, {}))[2][t.cycle] = cell
        seqs = sorted(rows)[-20:]
        self.setColumnCount(len(cycles))
        self.setRowCount(len(seqs))
        self.setHorizontalHeaderLabels([str(c) for c in cycles])
        self.setVerticalHeaderLabels([f"{rows[q][0]:08x}  {rows[q][1]}" for q in seqs])
        for r, q in enumerate(seqs):
            for c, cyc in enumerate(cycles):
                cell = rows[q][2].get(cyc, "")
                bg = T().error_line if cell in ("--", "xx") else T().stage.get(cell, (None,))[0]
                _set(self, r, c, cell, bg)
        if cycles:
            self.scrollToItem(self.item(max(len(seqs) - 1, 0), len(cycles) - 1))


def fields_html(fields) -> str:
    """The bit fields of an instruction word as a row of coloured cells:
    name [msb:lsb], the bits, their meaning."""
    e = html.escape
    cells = "".join(
        f"<td bgcolor='{T().fields[i % 8]}' style='padding:2px 6px'>"
        # the text colour on the span itself: a <td> colour is not honoured
        # by every style, and theme-white text on light cells is unreadable
        f"<span style='color:{T().highlight_text}'><b>{e(f.name)}</b> [{f.msb}:{f.lsb}]"
        f"<br><tt>{f.bits}</tt><br>{e(f.meaning)}</span></td>"
        for i, f in enumerate(fields))
    return f"<table cellspacing=2><tr>{cells}</tr></table>"


class InstrPanel(QtWidgets.QTextBrowser):
    """What happened inside the last executed instruction (InstrView):
    the bit fields, the ALU, the address, the branch and the effects."""

    def __init__(self) -> None:
        super().__init__()
        self.setFont(mono_font())

    def show_view(self, v: Any) -> None:
        if v is None:
            self.setHtml("<i>" + html.escape(tr("Step to see what happens inside an "
                                                "instruction.")) + "</i>")
            return
        e = html.escape
        w = getattr(v, "digits", 8)                     # a register's (MIPS64: 16)
        slot = " — <b>delay slot</b>" if v.in_delay_slot else ""
        own = v.format in ("MIPS16e", "microMIPS")      # no MIPS32 twin
        parts = [f"<h3><tt>0x{v.pc:0{w}x}: {e(v.source_text or v.text)}</tt></h3>"]
        if v.compressed:
            parts.append(f"<p>{e(v.compressed)}{slot if own else ''}</p>")
        if not own:
            parts.append(f"<p>{v.format}-format word <tt>0x{v.word:08x}</tt>"
                         + (f" (<tt>{e(v.text)}</tt>)" if v.compressed else "") + slot + "</p>")
        parts.append(fields_html(v.fields))
        if v.alu:
            a = v.alu
            parts.append("<h4>ALU</h4><p>" + e(a.op) + "<br>"
                         + (f"a = <tt>0x{a.a.value:0{w}x}</tt> from {e(a.a.source)}<br>" if a.a else "")
                         + (f"b = <tt>0x{a.b.value:0{w}x}</tt> from {e(a.b.source)}<br>" if a.b else "")
                         + (f"result = <tt>0x{a.result:0{w}x}</tt>" if a.result is not None else "")
                         + (" (zero)" if a.zero else "") + (" <b>OVERFLOW</b>" if a.overflow else "")
                         + "</p>")
        if v.address:
            ad = v.address
            parts.append(f"<h4>Address</h4><p>base <tt>0x{ad.base.value:0{w}x}</tt> + {ad.offset}"
                         f" = <tt>0x{ad.address:0{w}x}</tt> — {ad.access} of {ad.size} byte(s), "
                         f"{ad.segment} ({ad.privilege}, {'mapped' if ad.mapped else 'unmapped'})"
                         + ("" if ad.aligned else " <b>misaligned</b>")
                         + (f"<br>physical <tt>0x{ad.physical:08x}</tt> — {e(ad.how)}; "
                            f"attribute {ad.cca} = {cca_name(ad.cca)}"
                            if ad.physical is not None else "") + "</p>")
        if v.branch:
            br = v.branch
            what = (f"taken → <tt>0x{br.target:0{w}x}</tt>" if br.taken
                    else "not taken" + (", delay slot nullified" if br.likely else ""))
            parts.append(f"<h4>Branch</h4><p>{e(br.condition)}: {what}"
                         + (f"; link $ra = <tt>0x{br.link:0{w}x}</tt>" if br.link is not None else "")
                         + "</p>")
        eff = [f"memory read <tt>0x{a:08x}</tt> = <tt>0x{val:0{2 * n}x}</tt>"
               for a, n, val in v.mem_reads]
        eff += [f"memory write <tt>0x{a:08x}</tt>: <tt>0x{o:0{2 * n}x}</tt> → <tt>0x{nw:0{2 * n}x}</tt>"
                for a, n, o, nw in v.mem_writes]
        eff += [f"{e(name)}: <tt>0x{o:0{d}x}</tt> → <tt>0x{nw:0{d}x}</tt>"
                for name, o, nw in v.reg_writes
                for d in (w if name[0] == "$" and name[1] != "f" or name in ("hi", "lo")
                          else 8,)]
        if eff:
            parts.append("<h4>Effects</h4><p>" + "<br>".join(eff) + "</p>")
        if v.exception:
            parts.append(f"<h4 style='color:{T().breakpoint}'>Exception: {e(v.exception)}</h4>")
        self.setHtml("".join(parts))
