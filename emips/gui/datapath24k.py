"""The MIPS 24K pipeline drawn (PLAN M10 phase 5): its nine stages, the
fetch unit (instruction buffer, branch history table, return stack), the
multiply / divide, FPU and load / store units and the CP0 writes on their
way.

Each cycle the stages holding an instruction are filled in their colour
(the colours of the textbook stage doing the same work), the stage where
an interlock holds the pipeline is red, a redirect of the fetch (a
misprediction, an exception, ERET) lights the redirect wire.  Everything
comes from :meth:`Session.stage_details`, the pipeline's last
:class:`~emips.pipeline.CycleTrace` and :mod:`emips.coreview` — nothing
is re-executed here.  Hover a part for what it does.
"""

from __future__ import annotations

from .. import coredraw as CD
from ..coredraw import COL, H, STAGES, W
from ..datapathinfo import PARTS_24K
from ..session import Session
from . import theme
from .datapath import _font, _Part, _Value, _Wire, colour
from .qt import QtCore, QtGui, QtWidgets, Signal

Qt = QtCore.Qt
T = theme.colors
PARTS = CD.PARTS
STALL_STAGE = CD.STALL_STAGE
_x = CD.x_of


class Datapath24KView(QtWidgets.QGraphicsView):
    """The 24K's pipeline as a diagram (emips.coredraw's layout);
    :meth:`refresh` from ``coredraw.draw24k`` — the web page draws the same.
    A click on a part emits :attr:`part_clicked` (its key in
    :data:`emips.datapathinfo.PARTS_24K`); :meth:`show_numbers` numbers
    them (1-13)."""

    part_clicked = Signal(str)

    def __init__(self) -> None:
        self.scene_ = QtWidgets.QGraphicsScene(0, 0, W, H)
        super().__init__(self.scene_)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        self.numbers_on = False
        self._build()

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        key = item.data(0) if item is not None else None
        if key and event.button() == Qt.MouseButton.LeftButton:
            self.part_clicked.emit(key)
            return
        super().mousePressEvent(event)

    def show_numbers(self, on: bool) -> None:
        self.numbers_on = on
        for item in self.badges:
            item.setVisible(on)

    def _clickable(self, key: str, items, at: tuple[float, float]) -> None:
        """Make ``items`` open part ``key``; a number badge at ``at``."""
        info = PARTS_24K[key]
        c = T()
        for item in items:
            item.setData(0, key)
            item.setCursor(Qt.CursorShape.PointingHandCursor)
            tip = item.toolTip()
            item.setToolTip((tip + "\n" if tip else "") + "(click: what happens inside)")
        x0, y0 = at
        dot = self.scene_.addEllipse(x0 - 11, y0 - 11, 22, 22,
                                     QtGui.QPen(QtGui.QColor(c.red), 1.5),
                                     QtGui.QBrush(QtGui.QColor(c.red_fill)))
        num = self._text(0, 0, str(info.number), 9, True, c.highlight_text)
        br = num.boundingRect()
        num.setPos(x0 - br.width() / 2, y0 - br.height() / 2)
        for item in (dot, num):
            item.setZValue(5)
            item.setData(0, key)
            item.setCursor(Qt.CursorShape.PointingHandCursor)
            item.setToolTip(f"{info.number}: {info.title} (click: what happens inside)")
            item.setVisible(self.numbers_on)
            self.badges.append(item)

    def restyle(self) -> None:
        self.scene_.clear()
        self._build()

    def _fit(self) -> None:
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit()

    # -- construction ------------------------------------------------------------------

    def _text(self, x, y, text, size=9, bold=False, color=None):
        t = self.scene_.addSimpleText(text)
        t.setFont(_font(size, bold))
        t.setBrush(QtGui.QBrush(QtGui.QColor(color or T().text)))
        t.setPos(x, y)
        t.setZValue(3)
        return t

    def _build(self) -> None:
        s, c = self.scene_, T()
        L = CD.layout24k()
        s.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(c.canvas)))
        self.bands, self.heads, self.titles = {}, {}, {}
        self.parts: dict[str, _Part] = {}
        self.badges: list = []
        for st in STAGES:
            x = _x(st)
            band = s.addRect(x, 0, COL - 8, 290, QtGui.QPen(Qt.PenStyle.NoPen),
                             QtGui.QBrush(QtGui.QColor(c.bubble)))
            band.setZValue(-2)
            self.bands[st] = band
            self.heads[st] = self._text(x + 6, 6, st, 13, True, c.stage[st][1])
            self.titles[st] = self._text(x + 6, 30, "", 8, True)
            px, py, w, h, label, tip = L["parts"][st]
            self.parts[st] = _Part(s, "rect", px, py, w, h, label, tip)
            self._clickable(f"24k:{st}", [self.parts[st].item, self.parts[st].label, band],
                            (px, py))
        self.wires: dict[str, _Wire] = {k: _Wire(s, pts, tip)
                                        for k, (pts, tip) in L["wires"].items()}
        self.values: dict[str, _Value] = {k: _Value(s, x, y, size)
                                          for k, (x, y, size) in L["values"].items()}
        texts = {k: self._text(x, y, text, size, bold, colour(col))
                 for k, (x, y, size, bold, col, text) in L["texts"].items()}
        self.ib: list[QtWidgets.QGraphicsSimpleTextItem] = []
        self.ib_boxes = []
        for x, y, w, h in L["ib"]:
            box = s.addRect(x, y, w, h, QtGui.QPen(QtGui.QColor(c.idle_line)),
                            QtGui.QBrush(QtGui.QColor(c.part_fill)))
            box.setZValue(1)
            self.ib_boxes.append(box)
            self.ib.append(self._text(x + 6, y + 3, "", 8))
        blocks = {k: self._text(x, y, "", size, bold) for k, (x, y, size, bold)
                  in L["blocks"].items()}
        self.predict, self.units, self.hazards = (blocks["predict"], blocks["units"],
                                                  blocks["hazards"])
        self.status, self.events = blocks["status"], blocks["events"]
        R = L["regions"]
        self._clickable("24k:buffer", [texts["ib_head"]] + self.ib + self.ib_boxes,
                        tuple(R["24k:buffer"][:2]))
        self._clickable("24k:predict", [texts["predict_head"], self.predict],
                        tuple(R["24k:predict"][:2]))
        self._clickable("24k:units", [texts["units_head"], self.units],
                        tuple(R["24k:units"][:2]))
        self._clickable("24k:hazards", [texts["hazards_head"], self.hazards],
                        tuple(R["24k:hazards"][:2]))

    # -- refresh ---------------------------------------------------------------------------

    def refresh(self, s: Session) -> None:
        """Show ``coredraw.draw24k(s)``."""
        d = CD.draw24k(s)
        c = T()
        for st in STAGES:
            self.bands[st].setBrush(QtGui.QBrush(QtGui.QColor(colour(d["bands"][st]))))
            self.titles[st].setText(d["titles"][st])
            color, fill = d["parts"][st]
            self.parts[st].set(colour(color), colour(fill))
        for key, (text, color) in d["values"].items():
            self.values[key].set(text, colour(color))
        for key, (color, width) in d["wires"].items():
            if color is None:
                self.wires[key].set(None)
            else:
                self.wires[key].set(colour(color), width or 2.2)
        for (text, fill), item, box in zip(d["ib"], self.ib, self.ib_boxes):
            item.setText(text)
            box.setBrush(QtGui.QBrush(QtGui.QColor(colour(fill) or c.part_fill)))
        for key, item in (("predict", self.predict), ("units", self.units),
                          ("hazards", self.hazards), ("status", self.status),
                          ("events", self.events)):
            text, color = d["blocks"][key]
            item.setText(text)
            item.setBrush(QtGui.QBrush(QtGui.QColor(colour(color) or c.text)))
