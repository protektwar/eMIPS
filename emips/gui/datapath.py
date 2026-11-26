"""The animated datapath (PLAN M8): the classic 5-stage MIPS pipeline
(Patterson & Hennessy) drawn with Qt graphics items.

Layout: one column per stage, separated by the pipeline registers, and
horizontal *lanes* for the signals that travel through them — control,
PC + 4 / branch target, operand A, operand B, immediate, destination
register, forwarding / hazards, and the two write-back lanes that return
to the register file.

Each cycle (pipeline mode) the components and wires an instruction uses
light up in its stage's colour; forwarding paths and stalls are red;
values are written on the wires (PC, instruction word, register names,
immediate, ALU operands with their source and the result, memory address
and data, register write).  In instruction mode the one executed
instruction is shown in all stages, single-cycle style.

Everything drawn comes from :meth:`Session.stage_details` /
:attr:`Session.last_view` (the per-cycle trace) and the decoded word —
nothing is re-executed here.  Hover a component for an explanation.
"""

from __future__ import annotations

from .. import datapathdraw as DD
from ..datapathdraw import BARS, H, STAGE_X, STAGES, W
from ..datapathinfo import PARTS as _PARTS
from .qt import QtCore, QtGui, QtWidgets, Signal
from ..session import Session
from . import theme
from .widgets import mono_font

Qt = QtCore.Qt
T = theme.colors


def _font(size: int = 9, bold: bool = False) -> QtGui.QFont:
    f = mono_font()
    f.setPointSize(size)
    f.setBold(bold)
    return f


def colour(token: str | None) -> str | None:
    """A drawing's colour token (emips.datapathdraw) in the theme."""
    if token is None:
        return None
    c = T()
    if token.startswith("stage:"):
        return c.stage[token[6:]][1]
    if token.startswith("band:"):
        return c.stage[token[5:]][0]
    return {"red": c.red, "red_fill": c.red_fill, "muted": c.muted_text,
            "bubble": c.bubble}[token]


class _Part:
    """A component: rectangle, ellipse, ALU or mux, with a label and a
    tooltip."""

    def __init__(self, scene, kind, x, y, w, h, label, tip) -> None:
        self.box = (x, y, w, h)
        if kind == "ellipse":
            self.item = scene.addEllipse(x, y, w, h)
        elif kind == "alu":
            pts = [(0, 0), (1, 0.3), (1, 0.7), (0, 1), (0, 0.6), (0.3, 0.5), (0, 0.4)]
            self.item = scene.addPolygon(QtGui.QPolygonF(
                [QtCore.QPointF(x + w * a, y + h * b) for a, b in pts]))
        elif kind == "mux":
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(x, y, w, h), w / 2, w / 2)
            self.item = scene.addPath(path)
        else:
            self.item = scene.addRect(x, y, w, h)
        self.item.setToolTip(tip)
        self.item.setZValue(2)
        self.label = scene.addSimpleText(label)
        self.label.setFont(_font(8 if kind == "mux" else 9, kind != "mux"))
        self.label.setZValue(3)
        self.label.setToolTip(tip)
        self.rename(label)
        self.set(None)

    def rename(self, label: str, tip: str | None = None) -> None:
        """Change the label (centred again) and the tooltip."""
        x, y, w, h = self.box
        self.label.setText(label)
        br = self.label.boundingRect()
        self.label.setPos(x + (w - br.width()) / 2, y + (h - br.height()) / 2)
        if tip is not None:
            self.item.setToolTip(tip)
            self.label.setToolTip(tip)

    def show(self, on: bool) -> None:
        self.item.setVisible(on)
        self.label.setVisible(on)

    def set(self, color: str | None, fill: str | None = None) -> None:
        c = T()
        if color is None:
            self.item.setPen(QtGui.QPen(QtGui.QColor(c.idle_line), 1.2))
            self.item.setBrush(QtGui.QBrush(QtGui.QColor(c.part_fill)))
            self.label.setBrush(QtGui.QBrush(QtGui.QColor(c.idle_text)))
        else:
            self.item.setPen(QtGui.QPen(QtGui.QColor(color), 2.4))
            self.item.setBrush(QtGui.QBrush(QtGui.QColor(fill or c.part_fill)))
            self.label.setBrush(QtGui.QBrush(QtGui.QColor(c.text)))


class _Wire:
    """An orthogonal polyline ending in an arrowhead."""

    def __init__(self, scene, points, tip: str = "") -> None:
        path = QtGui.QPainterPath(QtCore.QPointF(*points[0]))
        for p in points[1:]:
            path.lineTo(QtCore.QPointF(*p))
        self.item = scene.addPath(path)
        self.item.setToolTip(tip)
        (x0, y0), (x1, y1) = points[-2], points[-1]
        if x1 != x0:
            d = 1 if x1 > x0 else -1
            tri = [(x1, y1), (x1 - 7 * d, y1 - 4), (x1 - 7 * d, y1 + 4)]
        else:
            d = 1 if y1 > y0 else -1
            tri = [(x1, y1), (x1 - 4, y1 - 7 * d), (x1 + 4, y1 - 7 * d)]
        self.head = scene.addPolygon(QtGui.QPolygonF([QtCore.QPointF(*p) for p in tri]))
        self.set(None)

    def set(self, color: str | None, width: float = 2.2) -> None:
        c = QtGui.QColor(color or T().idle_line)
        self.item.setPen(QtGui.QPen(c, width if color else 1.1))
        z = 1.5 if color == T().red else 1 if color else 0
        self.item.setZValue(z)
        self.head.setPen(QtGui.QPen(c, 1))
        self.head.setBrush(QtGui.QBrush(c))
        self.head.setZValue(z)

    def show(self, on: bool) -> None:
        self.item.setVisible(on)
        self.head.setVisible(on)


class _Value:
    """A value label on a white background (readable over the wires)."""

    def __init__(self, scene, x, y, size: int = 9) -> None:
        self.bg = scene.addRect(0, 0, 0, 0, QtGui.QPen(Qt.PenStyle.NoPen),
                                QtGui.QBrush(QtGui.QColor(*T().value_bg)))
        self.bg.setZValue(4)
        self.text = scene.addSimpleText("")
        self.text.setFont(_font(size))
        self.text.setZValue(5)
        self.x, self.y = x, y
        self.set("")

    def set(self, text: str, color: str | None = None) -> None:
        self.text.setText(text)
        self.text.setBrush(QtGui.QBrush(QtGui.QColor(color or T().text)))
        self.text.setPos(self.x, self.y)
        br = self.text.boundingRect()
        self.bg.setRect(self.x - 2, self.y - 1, br.width() + 4, br.height() + 2)
        self.bg.setVisible(bool(text))


class DatapathView(QtWidgets.QGraphicsView):
    """Scene with parts, wires and value labels (emips.datapathdraw's
    layout); :meth:`refresh` colours them from ``datapathdraw.draw`` — the
    web page draws the same.  A click on a part emits :attr:`part_clicked`
    (its key in :data:`emips.datapathinfo.PARTS`); :meth:`show_numbers`
    puts the guide's part numbers (1-18) on the drawing."""

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
        """The guide's part numbers on the drawing (View menu)."""
        self.numbers_on = on
        for item in self.badges:
            item.setVisible(on)

    def restyle(self) -> None:
        """Rebuild the scene in the current theme colours (the window then
        refreshes it)."""
        self.scene_.clear()
        self._build()

    # -- fit to the pane -----------------------------------------------------

    def _fit(self) -> None:
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit()

    # -- construction --------------------------------------------------------

    def _text(self, x, y, text, size=9, bold=False, color=None):
        t = self.scene_.addSimpleText(text)
        t.setFont(_font(size, bold))
        t.setBrush(QtGui.QBrush(QtGui.QColor(color or T().text)))
        t.setPos(x, y)
        t.setZValue(3)
        return t

    def _build(self) -> None:
        s = self.scene_
        c = T()
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(c.canvas)))
        self.bands, self.heads, self.titles = {}, {}, {}
        for st, (x0, x1) in STAGE_X.items():
            band = s.addRect(x0, 0, x1 - x0, H - 40, QtGui.QPen(Qt.PenStyle.NoPen))
            band.setZValue(-2)
            self.bands[st] = band
            self.heads[st] = self._text(x0 + 8, 6, st, 13, True, c.stage[st][1])
            self.titles[st] = self._text(x0 + 8, 30, "", 10, True)
        for name, x in BARS.items():
            r = s.addRect(x, 60, 12, 680, QtGui.QPen(QtGui.QColor(c.bar_stroke)),
                          QtGui.QBrush(QtGui.QColor(c.bar_fill)))
            r.setZValue(1)
            r.setToolTip(f"{name} pipeline register: holds what the next stage "
                         "needs, latched at the end of every cycle")
            self._text(x - 18, 744, name, 8, False, c.muted_text)
        self.parts: dict[str, _Part] = {}
        for key, (kind, x, y, w, h, label, tip) in DD.PARTS.items():
            p = self.parts[key] = _Part(s, kind, x, y, w, h, label, tip)
            for item in (p.item, p.label):          # clickable: what happens inside
                item.setData(0, key)
                item.setCursor(Qt.CursorShape.PointingHandCursor)
                item.setToolTip(item.toolTip() + DD.CLICK)
        self.wires: dict[str, _Wire] = {k: _Wire(s, pts, tip)
                                        for k, (pts, tip) in DD.WIRES.items()}
        self.ports = {t: self._text(x, y, t, 7, False, c.idle_text)
                      for t, (x, y) in DD.PORTS.items()}
        self.wires["rd_pass"].show(False)
        self._shown: dict[str, tuple] = {}
        self.values: dict[str, _Value] = {k: _Value(s, x, y, size)
                                          for k, (x, y, size) in DD.VALUES.items()}
        # the guide's part numbers (View > Datapath Part Numbers)
        self.badges = []
        for key, (number, title) in DD.layout()["numbers"].items():
            x0, y0, _, _ = self.parts[key].box
            dot = s.addEllipse(x0 - 11, y0 - 11, 22, 22, QtGui.QPen(QtGui.QColor(c.red), 1.5),
                               QtGui.QBrush(QtGui.QColor(c.red_fill)))
            num = self._text(0, 0, str(number), 9, True, c.highlight_text)
            br = num.boundingRect()
            num.setPos(x0 - br.width() / 2, y0 - br.height() / 2)
            for item in (dot, num):
                item.setZValue(5)
                item.setData(0, key)
                item.setCursor(Qt.CursorShape.PointingHandCursor)
                item.setToolTip(f"{number}: {title} (click: what happens inside)")
                item.setVisible(self.numbers_on)
                self.badges.append(item)
        # legend
        x = 1000
        for st in STAGES:
            r = s.addRect(x, 768, 14, 14, QtGui.QPen(QtGui.QColor(c.stage[st][1])),
                          QtGui.QBrush(QtGui.QColor(c.stage[st][0])))
            r.setZValue(3)
            self._text(x + 18, 767, st, 8)
            x += 62
        self._text(x, 767, "red: forwarding / stall", 8, False, c.red)

    # -- refresh -------------------------------------------------------------

    def refresh(self, s: Session) -> None:
        """Show ``datapathdraw.draw(s)``."""
        d = DD.draw(s)
        for key, (color, fill, label, tip, shown) in d.parts.items():
            p = self.parts[key]
            p.set(colour(color), colour(fill))
            if (label, tip) != self._shown.get(key):    # (rename only on a change)
                self._shown[key] = (label, tip)
                p.rename(label, tip)
            if p.item.isVisible() != shown:
                p.show(shown)
        for key, (color, shown) in d.wires.items():
            w = self.wires[key]
            w.set(colour(color))
            if w.item.isVisible() != shown:
                w.show(shown)
        for key, (text, color) in d.values.items():
            self.values[key].set(text, colour(color))
        for st in STAGES:
            self.titles[st].setText(d.titles[st])
            self.bands[st].setBrush(QtGui.QBrush(QtGui.QColor(colour(d.bands[st]))))
        for k, name in d.ports.items():
            if self.ports[k].text() != name:
                self.ports[k].setText(name)
