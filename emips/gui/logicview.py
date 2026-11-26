"""Gate-level drawings for the part window (the Datapath: click Control
or the ALU): the Qt side of :mod:`emips.partlogic`, whose :class:`Scene`s
(a PLA as Patterson & Hennessy draw the MIPS control, one ALU bit slice in
gate symbols) the web page draws too.  Lines carrying a 1 are lit (the
colour of the stage), 0s are grey."""

from __future__ import annotations

from . import theme
from .datapath import _font
from .qt import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt
T = theme.colors


class _Canvas(QtWidgets.QGraphicsView):
    """A scene that fits the width of its view (scrolls vertically)."""

    def __init__(self) -> None:
        self.scene_ = QtWidgets.QGraphicsScene()
        super().__init__(self.scene_)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)

    def _fit(self) -> None:
        r = self.scene_.itemsBoundingRect().adjusted(-8, -8, 8, 8)
        self.setSceneRect(r)
        if r.width() > 0:
            scale = min(1.0, (self.viewport().width() - 4) / r.width())
            self.resetTransform()
            self.scale(scale, scale)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit()

    # drawing helpers -------------------------------------------------------------

    def colour(self, token: str | None) -> str:
        c = T()
        if token == "lit":
            return self.lit
        if token is None or token == "text":
            return c.text
        return {"idle_line": c.idle_line, "idle_text": c.idle_text, "muted": c.muted_text,
                "part_fill": c.part_fill, "bar_stroke": c.bar_stroke}[token]



class SceneView(_Canvas):
    """Draws a :class:`emips.partlogic.Scene` (a PLA, an ALU bit slice)."""

    def __init__(self) -> None:
        super().__init__()
        self.lit = T().stage["EX"][1]

    def show_scene(self, scene) -> None:
        self.scene_.clear()
        c = T()
        token = scene.lit
        self.lit = c.stage[token[6:]][1] if token.startswith("stage:") else c.text
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(c.canvas)))
        for op in scene.ops:
            getattr(self, f"_{op[0]}")(*op[1:])
        self._fit()

    def _line(self, x0, y0, x1, y1, color, width, z):
        item = self.scene_.addLine(x0, y0, x1, y1,
                                   QtGui.QPen(QtGui.QColor(self.colour(color)), width))
        item.setZValue(z)

    def _text(self, x, y, s, size, bold, color, center, rot):
        t = self.scene_.addSimpleText(s)
        t.setFont(_font(size, bold))
        t.setBrush(QtGui.QBrush(QtGui.QColor(self.colour(color))))
        if center:
            x -= t.boundingRect().width() / 2
        if rot:
            t.setRotation(rot)
        t.setPos(x, y)
        t.setZValue(4)

    def _dot(self, x, y, r, color):
        col = QtGui.QColor(self.colour(color))
        d = self.scene_.addEllipse(x - r, y - r, 2 * r, 2 * r, QtGui.QPen(col),
                                   QtGui.QBrush(col))
        d.setZValue(3)

    def _rect(self, x, y, w, h, color):
        pen = QtGui.QPen(QtGui.QColor(self.colour(color)), 1, Qt.PenStyle.DashLine)
        self.scene_.addRect(x, y, w, h, pen)

    def _gate(self, kind, x, y, w, h, color, width):
        pen = QtGui.QPen(QtGui.QColor(self.colour(color)), width)
        brush = QtGui.QBrush(QtGui.QColor(T().part_fill))
        path = QtGui.QPainterPath()
        if kind == "and":
            path.moveTo(x, y)
            path.lineTo(x + w / 2, y)
            path.arcTo(x, y, w, h, 90, -180)
            path.lineTo(x, y + h)
            path.closeSubpath()
        elif kind in ("or", "xor"):
            path.moveTo(x, y)
            path.quadTo(x + w * 0.6, y, x + w, y + h / 2)
            path.quadTo(x + w * 0.6, y + h, x, y + h)
            path.quadTo(x + w * 0.3, y + h / 2, x, y)
            if kind == "xor":
                path.moveTo(x - 6, y)
                path.quadTo(x - 6 + w * 0.3, y + h / 2, x - 6, y + h)
        elif kind == "not":
            path.moveTo(x, y)
            path.lineTo(x + w - 6, y + h / 2)
            path.lineTo(x, y + h)
            path.closeSubpath()
            path.addEllipse(x + w - 6, y + h / 2 - 3, 6, 6)
        elif kind == "mux":
            path.moveTo(x, y)
            path.lineTo(x + w, y + h * 0.18)
            path.lineTo(x + w, y + h * 0.82)
            path.lineTo(x, y + h)
            path.closeSubpath()
        else:
            path.addRoundedRect(x, y, w, h, 6, 6)
        item = self.scene_.addPath(path, pen, brush)
        item.setZValue(2)


#: (the part window's views: one class draws every scene)
PLAView = SliceView = SceneView
