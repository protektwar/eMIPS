"""The part window of the Datapath: click a part (PC, ALU, forwarding
mux A, ...) to see what it is and what it does in the cycle on the
screen; the window follows every step until closed.  The text comes
from :mod:`emips.datapathinfo` (the trace, never recomputed); the ALU's
gates from :mod:`emips.alulogic` (derived from the traced operands and
checked against the traced result)."""

from __future__ import annotations

import html

from .. import datapathinfo as dpi
from .. import partlogic as pl
from ..session import Session
from . import theme
from .i18n import tr
from .logicview import SceneView
from .qt import QtCore, QtWidgets
from .widgets import mono_font

Qt = QtCore.Qt
E = html.escape


class PartDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.key: str | None = None
        self._session: Session | None = None
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.bit_label = QtWidgets.QLabel(tr("ALU bit slice:"))
        self.bit = QtWidgets.QSpinBox()
        self.bit.setRange(0, 31)
        self.bit.setToolTip(tr("Which of the 32 one-bit ALU slices to show gate by gate"))
        self.bit.valueChanged.connect(lambda _: self._again())
        bar.addWidget(self.bit_label)
        bar.addWidget(self.bit)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self.drawings = QtWidgets.QTabWidget()
        self.pla_main = SceneView()
        self.pla_second = SceneView()
        self.pla_alu = SceneView()
        self.slice = SceneView()
        self.view = QtWidgets.QTextBrowser()
        self.view.setOpenLinks(False)
        self.split.addWidget(self.drawings)
        self.split.addWidget(self.view)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        lay.addWidget(self.split)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        lay.addWidget(buttons)
        self.resize(860, 820)

    def show_part(self, s: Session, key: str) -> None:
        self.key = key
        self.refresh(s)
        self.show()
        self.raise_()
        self.activateWindow()

    def _again(self) -> None:
        if self._session is not None:
            self.refresh(self._session)

    def refresh(self, s: Session) -> None:
        self._session = s
        if self.key is None:
            return
        info = dpi.part_info(s, self.key)
        self.setWindowTitle(info.heading)
        mono = mono_font().family()
        where = (tr("In {stage} now: {text}").format(stage=E(info.part.stage),
                                                     text=f"<code>{E(info.instruction)}</code>")
                 if info.instruction else tr("{stage} holds a bubble").format(
                     stage=E(info.part.stage)))
        now = "".join(f"<li style='font-family:{mono}'>{E(line)}</li>" for line in info.now)
        out = pl.part_logic(s, self.key, self.bit.value(), tr)
        if out.width:                           # (MIPS64 / RV64: 64 slices)
            self.bit.setRange(0, out.width - 1)
            self.bit.setToolTip(tr("Which of the {n} one-bit ALU slices to show gate by "
                                   "gate").format(n=out.width))
        self._drawings(out)
        self.bit.setVisible(bool(out.width))
        self.bit_label.setVisible(bool(out.width))
        self.view.document().setDefaultStyleSheet(self._style())
        self.view.setHtml(
            f"<h3>{E(info.heading)}</h3><p>{where}</p>"
            f"<p><b>{E(tr('What happens in this cycle'))}</b></p><ul>{now}</ul>"
            f"{out.html}"
            f"<p><b>{E(tr('How it works'))}</b></p><p>{E(info.part.inside)}</p>")

    @staticmethod
    def _style() -> str:
        """emips.partlogic's classes in the theme."""
        c = theme.colors()
        return (f".chosen {{ background-color: {c.hit_fill}; color: {c.highlight_text}; }}"
                f" .mono {{ font-family: '{mono_font().family()}'; }}"
                f" .mismatch {{ color: {c.red}; }}")

    def _drawings(self, out) -> None:
        """The part's drawings in the tabs (hidden when it has none)."""
        tabs = self.drawings
        current = tabs.currentIndex()
        while tabs.count():
            tabs.removeTab(0)
        for name, scene in out.drawings:
            view = (self.slice if name == tr("ALU bit slice") else
                    self.pla_alu if name == tr("ALU control") else
                    self.pla_main if name == tr("Opcode decoder") else self.pla_second)
            view.show_scene(scene)
            tabs.addTab(view, name)
        tabs.setVisible(tabs.count() > 0)
        if 0 <= current < tabs.count():
            tabs.setCurrentIndex(current)
