"""Gate-level drawings for the part window (the Datapath: click Control
or the ALU): a PLA as Patterson & Hennessy draw the MIPS control (the
AND plane: input columns, true and inverted, crossed by the product-term
rows, a dot where a term takes a literal; the OR plane: the term rows
crossing the output columns), and one ALU bit slice in gate symbols.

Lines carrying a 1 are lit (the colour of the stage), 0s are grey; the
values come from :mod:`emips.controllogic` / :mod:`emips.alulogic`
(evaluated on the instruction the trace shows)."""

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

    def line(self, x0, y0, x1, y1, on: bool | None, width: float = 1.4):
        c = T()
        col = QtGui.QColor(self.lit if on else c.idle_line)
        item = self.scene_.addLine(x0, y0, x1, y1, QtGui.QPen(col, width + (1 if on else 0)))
        item.setZValue(2 if on else 1)
        return item

    def text(self, x, y, s, size=8, bold=False, color=None, center=False):
        t = self.scene_.addSimpleText(s)
        t.setFont(_font(size, bold))
        t.setBrush(QtGui.QBrush(QtGui.QColor(color or T().text)))
        if center:
            br = t.boundingRect()
            x -= br.width() / 2
        t.setPos(x, y)
        t.setZValue(4)
        return t

    def dot(self, x, y, on: bool, r: float = 3.2):
        col = QtGui.QColor(self.lit if on else T().idle_text)
        d = self.scene_.addEllipse(x - r, y - r, 2 * r, 2 * r, QtGui.QPen(col),
                                   QtGui.QBrush(col))
        d.setZValue(3)
        return d


class PLAView(_Canvas):
    """A PLA, its inputs set to ``x`` (input 0 = the most significant bit)."""

    COL, ROW = 18, 13

    def __init__(self) -> None:
        super().__init__()
        self.lit = T().stage["ID"][1]

    def show_pla(self, pla, x: int | None, title: str, lit: str,
                 labels: dict[str, str] | None = None) -> None:
        self.scene_.clear()
        c = T()
        self.lit = lit
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(c.canvas)))
        n, terms, outs = pla.n, pla.terms, pla.outputs
        hit, values = pla.evaluate(x) if x is not None else ([False] * len(terms), {})
        bits = [None if x is None else (x >> (n - 1 - i)) & 1 for i in range(n)]
        C, R = self.COL, self.ROW
        top = 66 + max(len(name) for name in pla.inputs) * 6
        and_w = 2 * n * C
        or_x0 = and_w + 40
        bottom = top + max(1, len(terms)) * R + 10
        self.text(0, 0, title, 10, True)
        self.text(0, 16, "AND plane: a dot ANDs that input (true, or inverted ~) into the "
                         "product term of its row", 7, False, c.muted_text)
        self.text(0, 28, "OR plane: a dot ORs the term into the output of its column; lit "
                         "= 1", 7, False, c.muted_text)
        # input columns: true and inverted
        for i, name in enumerate(pla.inputs):
            for k, inv in enumerate((False, True)):
                cx = (2 * i + k) * C + C / 2
                on = None if bits[i] is None else bool(bits[i] ^ inv)
                self.line(cx, top - 6, cx, bottom, on)
                if not inv:
                    t = self.text(0, 0, name, 7, True)
                    t.setRotation(-90)
                    t.setPos(cx + C / 2 - 6, top - 30)
                    if bits[i] is not None:
                        self.text(cx, top - 20, str(bits[i]), 7, False,
                                  self.lit if bits[i] else c.muted_text, center=True)
                        self.text(cx + C, top - 20, str(1 - bits[i]), 7, False,
                                  self.lit if not bits[i] else c.muted_text, center=True)
                if inv:
                    self.text(cx, top - 19 if bits[i] is None else top - 8, "~", 7, True,
                              c.muted_text, center=True)
        # output columns
        for j, name in enumerate(outs):
            cx = or_x0 + j * C + C / 2
            on = values.get(name) if x is not None else None
            self.line(cx, top - 6, cx, bottom, on)
            label = (labels or {}).get(name, name)
            t = self.text(cx - 4, bottom + 4, label, 7, True,
                          self.lit if on else None)
            t.setRotation(90)
            t.setPos(cx + 5, bottom + 4)
            if x is not None:
                self.text(cx, top - 19, "1" if on else "0", 7, False,
                          self.lit if on else c.muted_text, center=True)
        # term rows
        for k, term in enumerate(terms):
            y = top + k * R + R / 2
            self.line(0, y, or_x0 + len(outs) * C, y, hit[k] if x is not None else None,
                      0.8)
            for i in range(n):
                lit = pla.literal(term, i)
                if lit is not None:
                    cx = (2 * i + (0 if lit else 1)) * C + C / 2
                    self.dot(cx, y, hit[k])
            for j, name in enumerate(outs):
                if k in pla.uses.get(name, []):
                    self.dot(or_x0 + j * C + C / 2, y, hit[k])
        # the planes' frames
        pen = QtGui.QPen(QtGui.QColor(c.bar_stroke), 1, Qt.PenStyle.DashLine)
        self.scene_.addRect(-4, top - 8, and_w + 8, bottom - top + 12, pen)
        self.scene_.addRect(or_x0 - 4, top - 8, len(outs) * C + 8, bottom - top + 12, pen)
        self.text(and_w / 2, bottom + 6, f"AND plane: {len(terms)} product terms", 7, False,
                  c.muted_text, center=True)
        self._fit()


class SliceView(_Canvas):
    """One bit of the ALU in gate symbols."""

    def __init__(self) -> None:
        super().__init__()
        self.lit = T().stage["EX"][1]

    # gate symbols ----------------------------------------------------------------

    def _gate(self, kind: str, x, y, w, h, on: bool, label: str = ""):
        c = T()
        col = QtGui.QColor(self.lit if on else c.idle_line)
        pen = QtGui.QPen(col, 2 if on else 1.4)
        brush = QtGui.QBrush(QtGui.QColor(c.part_fill))
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
        if label:
            self.text(x + w / 2, y + h / 2 - 6, label, 8, True, center=True)

    def wire(self, pts, on: bool, label: str = "", value: int | None = None):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.line(x0, y0, x1, y1, on)
        if label:
            x, y = pts[0]
            self.text(x - 4, y - 13, label + ("" if value is None else f" = {value}"), 7,
                      False, self.lit if on else None)

    def show_slice(self, sl, ctl, msb: bool, lit: str) -> None:
        """Bit ``sl`` (an :class:`emips.alulogic.Slice`) under ALU control
        ``ctl``; ``msb``: bit 31, with Overflow and Set.  A wire is lit when
        it carries a 1."""
        self.scene_.clear()
        c = T()
        self.lit = lit
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(c.canvas)))
        op = ctl.operation
        self.text(0, -44, f"ALU bit {sl.i}: Ainvert {ctl.ainvert}, Bnegate {ctl.bnegate}, "
                          f"Operation {op:03b}" + (f", Unsigned {ctl.unsigned}"
                                                   if msb else ""), 10, True)
        # the inputs and their inverting muxes (Ainvert, Bnegate)
        for name, val, y, sel, out, label in (
                ("a", sl.a, 40, ctl.ainvert, sl.a_in, "Ainvert"),
                ("b", sl.b, 130, ctl.bnegate, sl.b_in, "Bnegate")):
            self.wire([(0, y), (70, y), (70, y + 6), (90, y + 6)], val, f"{name}{sl.i}", val)
            self.wire([(40, y), (40, y + 30), (50, y + 30)], val)
            self._gate("not", 50, y + 22, 26, 16, not val)
            self.wire([(76, y + 30), (90, y + 30)], 1 - val)
            self._gate("mux", 90, y - 4, 22, 46, out)
            self.text(94, y - 2, "0", 6, False, c.muted_text)
            self.text(94, y + 22, "1", 6, False, c.muted_text)
            self.wire([(56, y - 20), (101, y - 20), (101, y - 1)], sel, label, sel)
        ax, bx = 140, 156
        self.wire([(112, 59), (ax, 59)], sl.a_in)
        self.wire([(ax, 48), (ax, 298)], sl.a_in)
        self.text(116, 62, f"a' = {sl.a_in}", 7, False, self.lit if sl.a_in else None)
        self.wire([(112, 149), (bx, 149)], sl.b_in)
        self.wire([(bx, 64), (bx, 314)], sl.b_in)
        self.text(116, 152, f"b' = {sl.b_in}", 7, False, self.lit if sl.b_in else None)
        # AND, OR, the full adder and XOR (top to bottom, in the order of
        # the mux inputs they feed, so no two routes cross)
        gx = 210
        outs = {}
        for kind, y, val in (("and", 30, sl.and_), ("or", 85, sl.or_), ("xor", 290, sl.xor_)):
            self.wire([(ax, y + 8), (gx, y + 8)], sl.a_in)
            self.wire([(bx, y + 24), (gx, y + 24)], sl.b_in)
            self._gate(kind, gx, y, 44, 32, val)
            self.text(gx + 13, y + 9, kind.upper(), 7, True)
            outs[kind] = (gx + 44, y + 16, val)
        fy = 170
        self.wire([(ax, fy + 16), (gx, fy + 16)], sl.a_in)
        self.wire([(bx, fy + 40), (gx, fy + 40)], sl.b_in)
        self._gate("box", gx, fy, 64, 64, sl.sum_, "+")
        self.wire([(gx + 32, fy - 20), (gx + 32, fy)], sl.cin)
        self.text(gx + 38, fy - 22, f"CarryIn = {sl.cin}"
                  + (" (= Bnegate)" if sl.i == 0 else f" (bit {sl.i - 1}'s CarryOut)"), 7,
                  False, self.lit if sl.cin else None)
        self.wire([(gx + 32, fy + 64), (gx + 32, fy + 90)], sl.cout)
        self.text(gx + 38, fy + 76, f"CarryOut = {sl.cout}"
                  + (f" (to bit {sl.i + 1})" if sl.i < 31 else ""), 7, False,
                  self.lit if sl.cout else None)
        outs["sum"] = (gx + 64, fy + 32, sl.sum_)
        # the result mux: 0 AND, 1 OR, 2 sum, 3 Less, 4 XOR, 5 shifter
        mx, my, mh = 460, 30, 300
        self._gate("mux", mx, my, 34, mh, sl.result)
        ins = {k: my + 30 + k * 46 for k in range(6)}
        routes = [(0, "and", 380), (1, "or", 388), (2, "sum", 396), (4, "xor", 404)]
        for k, key, xv in routes:
            x0, y0, val = outs[key]
            self.wire([(x0, y0), (xv, y0), (xv, ins[k]), (mx, ins[k])], val)
        less_y, sh_y = 380, 410
        self.wire([(240, less_y), (420, less_y), (420, ins[3]), (mx, ins[3])], sl.less)
        self.text(240, less_y - 13, f"Less = {sl.less}" + (" (Set, from bit 31)" if sl.i == 0
                                                           else " (0)"), 7, False,
                  self.lit if sl.less else None)
        self.wire([(240, sh_y), (428, sh_y), (428, ins[5]), (mx, ins[5])], sl.shift)
        self.text(240, sh_y - 13, f"shifter bit = {sl.shift}", 7, False,
                  self.lit if sl.shift else None)
        names = ["AND", "OR", "sum", "Less", "XOR", "shifter"]
        for k in range(6):
            chosen = k == op
            self.text(mx + 4, ins[k] - 7, f"{k} {names[k]}" if chosen else str(k), 7, chosen,
                      self.lit if chosen else c.muted_text)
        self.wire([(mx + 17, my - 22), (mx + 17, my + 6)], True)
        self.text(mx + 22, my - 26, f"Operation {op:03b} ({names[op] if op < 6 else '?'})", 7,
                  True, self.lit)
        ry = my + mh / 2
        self.wire([(mx + 34, ry), (mx + 140, ry)], sl.result)
        self.text(mx + 44, ry - 15, f"Result{sl.i} = {sl.result}", 8, True,
                  self.lit if sl.result else None)
        if msb:                                 # Overflow and Set (bit 31 only),
            ox, oy = 600, 360                   # fed by named stubs (net labels)
            ovf = sl.cin ^ sl.cout
            setv = (1 - sl.cout) if ctl.unsigned else sl.sum_ ^ ovf
            for name, val, y in (("CarryIn31", sl.cin, oy + 8), ("CarryOut31", sl.cout, oy + 26),
                                 ("Sum31", sl.sum_, oy + 88), ("Overflow", ovf, oy + 106)):
                self.wire([(ox - 34, y), (ox, y)], val)
                self.text(ox - 34 - 7 * len(name), y - 7, name, 7, False,
                          self.lit if val else None)
            self._gate("xor", ox, oy, 42, 34, ovf)
            self.wire([(ox + 42, oy + 17), (ox + 120, oy + 17)], ovf)
            self.text(ox + 48, oy + 2, f"Overflow = {ovf}", 8, True, self.lit if ovf else None)
            sy = oy + 80
            self._gate("xor", ox, sy, 42, 34, sl.sum_ ^ ovf)
            self.wire([(ox + 42, sy + 17), (ox + 120, sy + 17)], setv)
            self.text(ox + 48, sy + 2, f"Set = {setv}", 8, True, self.lit if setv else None)
            self.text(ox - 34, sy + 44, "Set = Sum31 XOR Overflow (Unsigned: NOT CarryOut31)",
                      7, False, c.muted_text)
            self.text(ox - 34, sy + 56, "-> bit 0's Less", 7, False, c.muted_text)
        self._fit()
