"""The Tools panes (dockable): instruction reference, encoder / decoder
and number converter.  Thin Qt views over :mod:`emips.gui.tools`."""

from __future__ import annotations

import html

from ..assembler import AsmError
from ..isa import ISADecodeError
from .. import tools
from .i18n import N_, tr
from .qt import QtCore, QtWidgets, Signal
from . import theme
from .widgets import fields_html, mono_font

Qt = QtCore.Qt
E = html.escape
EXEC_TEXT = {
    "EXEC": N_("simulated"),
    "HINT": N_("simulated as a no-op (no caches, prefetch or hazards are modelled)"),
    "DEFER": N_("decoded but not simulated yet (the simulation stops)"),
    "ILLEGAL": N_("raises Reserved Instruction in this model (paired-single / COP2)"),
}


def _exec_text(bucket: str) -> str:
    return tr(EXEC_TEXT.get(bucket, bucket))


def _cell(text: str, color: str) -> str:
    return (f"<td bgcolor='{color}' style='padding:2px 5px'><span style='color:"
            f"{theme.colors().highlight_text}'>{text}</span></td>")


def _error(text: str) -> str:
    return f"<p style='color:{theme.colors().red}'>{E(text)}</p>"


def encoding_html(row_name: str, release: str) -> str:
    """A row's encoding as cells: field name [msb:lsb] over the bits
    (0 / 1 fixed, letters = the operand's bits)."""
    cells = "".join(
        _cell(f"<b>{E(name)}</b> [{msb}:{lsb}]<br><tt>{E(bits)}</tt>",
              theme.colors().fields[i % len(theme.colors().fields)])
        for i, (msb, lsb, name, bits) in enumerate(tools.template_fields(row_name,
                                                                          release)))
    return f"<table cellspacing=2><tr>{cells}</tr></table>"


class _Pane(QtWidgets.QWidget):
    """Common context: the session's release, register names, strict mode."""

    def __init__(self) -> None:
        super().__init__()
        self.release, self.abi, self.pseudo = "R5", True, True

    def set_context(self, release: str, abi: bool = True, pseudo: bool = True) -> None:
        changed = (release, abi, pseudo) != (self.release, self.abi, self.pseudo)
        self.release, self.abi, self.pseudo = release, abi, pseudo
        if changed:
            self.update_view()

    def update_view(self) -> None:          # pragma: no cover - overridden
        pass


class ReferenceView(_Pane):
    """Every instruction of the selected release: search, syntax, encoding,
    releases, how the simulator treats it.  Double-click (or *Insert*)
    puts the syntax in the editor."""

    insert_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText(tr("search: mnemonic, description or family "
                                          "(e.g. lw, branch, float)"))
        self.search.setClearButtonEnabled(True)
        self.count = QtWidgets.QLabel()
        bar.addWidget(self.search)
        bar.addWidget(self.count)
        lay.addLayout(bar)
        split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([tr("syntax"), tr("description"),
                                              tr("format"), tr("since")])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setFont(mono_font())
        self.details = QtWidgets.QTextBrowser()
        split.addWidget(self.table)
        split.addWidget(self.details)
        split.setSizes([300, 220])
        lay.addWidget(split)
        self.insert = QtWidgets.QPushButton(tr("Insert into editor"))
        lay.addWidget(self.insert)
        self.rows: list[tools.RefRow] = []
        self.shown: list[tools.RefRow] = []
        self.search.textChanged.connect(lambda _: self._filter())
        self.table.currentCellChanged.connect(lambda r, *_: self._show(r))
        self.table.cellDoubleClicked.connect(lambda r, _c: self._insert(r))
        self.insert.clicked.connect(lambda: self._insert(self.table.currentRow()))
        self.update_view()

    def update_view(self) -> None:
        self.rows = tools.reference(self.release)
        self._filter()

    def _filter(self) -> None:
        q = self.search.text()
        self.shown = [r for r in self.rows if r.matches(q)]
        t = self.table
        t.setRowCount(len(self.shown))
        for i, r in enumerate(self.shown):
            for c, text in enumerate((r.syntax, r.description, r.format, r.since)):
                item = t.item(i, c)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    t.setItem(i, c, item)
                item.setText(text)
        t.resizeColumnToContents(0)
        self.count.setText(tr("{n} of {total} ({release})").format(
            n=len(self.shown), total=len(self.rows), release=self.release))
        if self.shown:
            t.setCurrentCell(0, 0)
            self._show(0)
        else:
            self.details.clear()

    def _show(self, row: int) -> None:
        if not 0 <= row < len(self.shown):
            return
        r = self.shown[row]
        facts = [tr("format {fmt}, family {family}").format(fmt=r.format, family=r.family),
                 tr("since MIPS32 {since}").format(since=r.since)
                 + (", " + tr("removed in {rel}").format(rel=r.removed_in)
                    if r.removed_in else ""),
                 _exec_text(r.execution)]
        if r.delay_slot:
            facts.append(tr("has a branch delay slot: the next instruction always "
                            "executes"))
        if r.privileged:
            facts.append(tr("privileged: needs kernel mode (or Status.CU0)"))
        self.details.setHtml(
            f"<h3><tt>{E(r.syntax)}</tt></h3><p>{E(r.description)}</p>"
            + encoding_html(r.name, self.release)
            + "<p><small>" + E(tr("0 / 1: fixed bits (they select the instruction); "
                                  "letters: the bits of the operand named above "
                                  "them")) + "</small></p><ul>"
            + "".join(f"<li>{E(f)}</li>" for f in facts) + "</ul>")

    def _insert(self, row: int) -> None:
        if 0 <= row < len(self.shown):
            self.insert_requested.emit(self.shown[row].syntax)


class EncoderView(_Pane):
    """Type a word (0x...) to decode it, or an instruction to encode it."""

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText(tr("0x01095020   or   add $t2, $t0, $t1"))
        self.input.setFont(mono_font())
        self.input.setClearButtonEnabled(True)
        self.pc = QtWidgets.QLineEdit("0x00400000")
        self.pc.setFont(mono_font())
        self.pc.setToolTip(tr("address of the instruction (for branches and jumps)"))
        form.addRow(tr("Word or instruction"), self.input)
        form.addRow(tr("At address"), self.pc)
        lay.addLayout(form)
        self.output = QtWidgets.QTextBrowser()
        lay.addWidget(self.output)
        self.input.textChanged.connect(lambda _: self.update_view())
        self.pc.textChanged.connect(lambda _: self.update_view())
        self.update_view()

    def update_view(self) -> None:
        text = self.input.text()
        if not text.strip():
            self.output.setHtml("<p>" + E(tr(
                "Decode: type a 32-bit word in hex (0x8fa80004), binary (0b...) or "
                "decimal. Encode: type an instruction (lw $t0, 4($sp)); a "
                "pseudo-instruction shows every machine word it becomes.")) + "</p>")
            return
        try:
            pc = int(self.pc.text().strip() or "0", 0) & 0xFFFFFFFF
        except ValueError:
            pc = 0x00400000
        try:
            mode, words = tools.encode_or_decode(text, release=self.release, pc=pc,
                                                 abi=self.abi, pseudo=self.pseudo)
        except (AsmError, ValueError, ISADecodeError, KeyError) as exc:
            msg = str(exc)
            if isinstance(exc, ISADecodeError):
                msg = tr("not a MIPS32 {release} instruction (Reserved Instruction "
                         "exception)").format(release=self.release)
            self.output.setHtml(_error(msg))
            return
        head = (tr("decoded") if mode == "decode" else
                tr("assembled into {n} machine word(s)").format(n=len(words)))
        parts = [f"<p><i>{E(head)}</i></p>"]
        for i, d in enumerate(words):
            parts.append(
                f"<h3><tt>0x{pc + 4 * i:08x}: 0x{d.word:08x} &nbsp; {E(d.text)}</tt></h3>"
                f"<p>{E(d.description)} — {E(d.format)}-format, {E(d.name)}; "
                f"{E(_exec_text(d.execution))}</p>"
                + fields_html(d.fields)
                + f"<p><tt>{tools.group_bits(d.word, 32)}</tt></p>")
        self.output.setHtml("".join(parts))


class ConverterView(QtWidgets.QWidget):
    """A number in every representation, with the IEEE-754 fields."""

    def __init__(self) -> None:
        super().__init__()
        self.nan2008 = False
        lay = QtWidgets.QVBoxLayout(self)
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText(tr("42   -7   0x2a   0b101010   'A'   1.5   "
                                         "-0.1   inf   nan"))
        self.input.setFont(mono_font())
        self.input.setClearButtonEnabled(True)
        lay.addWidget(self.input)
        self.output = QtWidgets.QTextBrowser()
        lay.addWidget(self.output)
        self.input.textChanged.connect(lambda _: self.update_view())
        self.update_view()

    def set_nan2008(self, on: bool) -> None:
        if on != self.nan2008:
            self.nan2008 = on
            self.update_view()

    def update_view(self) -> None:
        text = self.input.text()
        if not text.strip():
            self.output.setHtml("<p>" + E(tr(
                "Type a number: decimal, hex (0x), binary (0b), octal (0o), a "
                "character ('A') or a floating-point value (1.5, 2e-3, inf, nan).")) + "</p>")
            return
        try:
            n = tools.convert(text, nan2008=self.nan2008)
        except ValueError as exc:
            self.output.setHtml(_error(str(exc)))
            return
        rows = [(tr("hex"), n.hex), (tr("unsigned"), str(n.unsigned)),
                (tr("signed (two's complement)"), str(n.signed)),
                (tr("binary"), n.binary), (tr("octal"), n.octal),
                (tr("bytes as text"), n.chars)]
        table = "".join(f"<tr><td>{E(k)}</td><td><tt>{E(v)}</tt></td></tr>" for k, v in rows)
        parts = [f"<p><i>{E(n.source)}, {n.width} {E(tr('bits'))}</i></p>",
                 f"<table cellpadding=3>{table}</table>"]
        for fp, title in ((n.single, tr("as an IEEE-754 single (float)")),
                          (n.double, tr("as an IEEE-754 double"))):
            if fp is not None:
                parts.append(self._float_html(fp, title))
        if n.notes:
            parts.append("<ul>" + "".join(f"<li>{E(x)}</li>" for x in n.notes) + "</ul>")
        self.output.setHtml("".join(parts))

    @staticmethod
    def _float_html(fp: tools.FloatParts, title: str) -> str:
        ew = 8 if fp.width == 32 else 11
        fw = fp.width - 1 - ew
        cells = (_cell(f"<b>{E(tr('sign'))}</b><br><tt>{fp.sign}</tt>", theme.colors().fields[0])
                 + _cell(f"<b>{E(tr('exponent'))}</b> ({fp.exponent})<br>"
                         f"<tt>{fp.exponent:0{ew}b}</tt>", theme.colors().fields[1])
                 + _cell(f"<b>{E(tr('fraction'))}</b><br><tt>{fp.fraction:0{fw}b}</tt>",
                         theme.colors().fields[2]))
        value = tools.fmt_float(fp.value)
        return (f"<h4>{E(title)}: <tt>0x{fp.bits:0{fp.width // 4}x}</tt></h4>"
                f"<table cellspacing=2><tr>{cells}</tr></table>"
                f"<p>{E(fp.kind)}: {E(fp.formula)}<br>"
                f"{E(tr('value'))} = <tt>{E(value)}</tt></p>")
