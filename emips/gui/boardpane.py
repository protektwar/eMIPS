"""The Board pane (PLAN M15): the MIPS Malta as it sits on the desk — its
eight LEDs (LEDBAR), the eight-character ASCII display, the DIP switches
S2, the ON/NMI button — and the GT-64120 system controller: its interrupt
cause, the last bus error, its address decode (what answers where) — and
the IDE disks on the PIIX4 (their state, the last commands; Save Disk
writes what a program wrote).  On the flat board (every physical address
RAM) it says so.

Everything comes from the board model (:meth:`Session.board_lines`, the
:class:`emips.malta.Fpga` registers): the pane only draws it."""

from __future__ import annotations

from ..session import Session
from . import theme
from .i18n import tr
from .qt import QtCore, QtWidgets, Signal
from .widgets import mono_font

Qt = QtCore.Qt


class BoardView(QtWidgets.QWidget):
    #: the ON/NMI button was pressed
    nmi_requested = Signal()
    #: Save Disk: the slot (pri0 ...)
    save_disk_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("LEDs")))
        self.leds: list[QtWidgets.QLabel] = []
        for i in range(7, -1, -1):
            led = QtWidgets.QLabel()
            led.setFixedSize(18, 18)
            led.setToolTip(tr("LEDBAR bit {n}").format(n=i))
            row.addWidget(led)
            self.leds.insert(0, led)
        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel(tr("Display")))
        self.display = QtWidgets.QLabel()
        font = mono_font()
        font.setPointSize(font.pointSize() + 6)
        font.setBold(True)
        self.display.setFont(font)
        self.display.setToolTip(tr("The ASCII display: ASCIIWORD (a word in hex) or "
                                   "ASCIIPOS0-7 (one character each)"))
        row.addWidget(self.display)
        row.addSpacing(16)
        self.switches = QtWidgets.QLabel()
        self.switches.setToolTip(tr("DIP switch S2 (the SWITCH register): set in "
                                    "Configure > Settings > Processor"))
        row.addWidget(self.switches)
        row.addStretch(1)
        self.nmi = QtWidgets.QPushButton(tr("ON/NMI"))
        self.nmi.setToolTip(tr("The board's ON/NMI button: a Non Maskable Interrupt "
                               "(NMISTATUS.ONNMI)"))
        self.nmi.clicked.connect(self.nmi_requested)
        row.addWidget(self.nmi)
        lay.addLayout(row)
        disks = QtWidgets.QHBoxLayout()
        self.disk_label = QtWidgets.QLabel(tr("IDE disk"))
        disks.addWidget(self.disk_label)
        self.disk = QtWidgets.QComboBox()
        self.disk.setToolTip(tr("The disks of Configure > Settings > Processor"))
        disks.addWidget(self.disk)
        self.save_disk = QtWidgets.QPushButton(tr("Save Disk…"))
        self.save_disk.setToolTip(tr("Write the disk as the program left it to a file "
                                     "(every run starts from its image)"))
        self.save_disk.clicked.connect(
            lambda: self.disk.currentData() and self.save_disk_requested.emit(
                self.disk.currentData()))
        disks.addWidget(self.save_disk)
        disks.addStretch(1)
        lay.addLayout(disks)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(mono_font())
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self.text, 1)

    def refresh(self, s: Session) -> None:
        board = s.board if s.machine is not None else None
        malta = board is not None
        for w in (*self.leds, self.display, self.switches, self.nmi):
            w.setVisible(malta)
        slots = list(board.disks) if malta else []
        for w in (self.disk_label, self.disk, self.save_disk):
            w.setVisible(bool(slots))
        if [self.disk.itemData(i) for i in range(self.disk.count())] != slots:
            self.disk.clear()
            for slot in slots:
                self.disk.addItem(board.disks[slot].name, slot)
        if not malta:
            self.note.setText(tr("Flat board: every physical address is RAM (the MARS "
                                 "layout). Configure > Settings > Processor > Board "
                                 "chooses the MIPS Malta: its memory map, bus errors, "
                                 "LEDs and display."))
            self.text.setPlainText("")
            return
        c = theme.colors()
        f = board.fpga
        for i, led in enumerate(self.leds):
            fill = c.red if f.ledbar >> i & 1 else c.bubble
            led.setStyleSheet(f"background: {fill}; border: 1px solid {c.idle_line}; "
                              "border-radius: 9px;")
        self.display.setText(f.display.replace(" ", " "))
        self.display.setStyleSheet(f"background: {c.part_fill}; color: {c.highlight_text}; "
                                   "padding: 2px 6px;")
        self.switches.setText(tr("S2 = 0x{v:02x}").format(v=board.switches))
        self.note.setText(tr("MIPS Malta, CoreLV card (GT-64120 system controller): "
                             "SDRAM, the monitor flash and the board registers; "
                             "anything else is a bus error."))
        lines = s.board_lines()
        if board.disks or s.disk_notes:
            lines += ["", tr("IDE disk commands:")] + s.disk_lines(states=False)
        self.text.setPlainText("\n".join(lines))
