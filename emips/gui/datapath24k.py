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

from .. import coreview as cv
from ..datapathinfo import PARTS_24K
from ..session import Session
from . import theme
from .datapath import _font, _Part, _Value, _Wire
from .qt import QtCore, QtGui, QtWidgets, Signal

Qt = QtCore.Qt
T = theme.colors
W, H = 1500, 790
STAGES = ("IF", "IS", "IT", "RF", "AG", "EX", "MS", "ER", "WB")
COL = 166                                    # column pitch
X0 = 4
Y_PART, H_PART = 112, 96
Y_WIRE = Y_PART + H_PART // 2

PARTS = {
    "IF": ("I-cache\nITLB, BHT",
           "IF: the I-cache and its tags are read, the ITLB translates the PC, the "
           "branch history table is read — for a 64-bit bundle (two instructions)"),
    "IS": ("hit / way\npredict",
           "IS: I-cache hit and way select; conditional branches are predicted from "
           "the BHT, jr $31 from the return stack"),
    "IT": ("instruction\nbuffer",
           "IT: the 8-entry instruction buffer (bypassed while empty); the branch "
           "target is computed here"),
    "RF": ("registers\ndecode",
           "RF: register file read, decode; a load / store waits here for its base "
           "register (the skewed ALU: an ALU result reaches AG a cycle late)"),
    "AG": ("address\nadder",
           "AG: the data address is added; an instruction waits here for its ALU "
           "operands (load data, MFC0, MUL / MFHI results) or a busy unit"),
    "EX": ("ALU, DTLB\nbranch",
           "EX: the ALU, the DTLB, the D-cache access starts; branches are resolved "
           "(a misprediction redirects the fetch)"),
    "MS": ("D-cache\nhit, align",
           "MS: D-cache hit and way select, load align; a waiting access holds the "
           "pipeline here"),
    "ER": ("exceptions\ncomplete",
           "ER: exceptions and interrupts are taken here; the instruction completes"),
    "WB": ("register\nwrite", "WB: the register file is written"),
}

#: where a stall kind holds the pipeline
STALL_STAGE = {"memory": "MS", "fetch": "IT", "alu-agen": "RF"}


def _x(st: str) -> int:
    return X0 + STAGES.index(st) * COL


class Datapath24KView(QtWidgets.QGraphicsView):
    """The 24K's pipeline as a diagram; :meth:`refresh` from the session.
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
        s.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(c.canvas)))
        self.bands, self.heads, self.titles = {}, {}, {}
        self.parts: dict[str, _Part] = {}
        self.wires: dict[str, _Wire] = {}
        self.values: dict[str, _Value] = {}
        self.badges: list = []
        for st in STAGES:
            x = _x(st)
            band = s.addRect(x, 0, COL - 8, 290, QtGui.QPen(Qt.PenStyle.NoPen),
                             QtGui.QBrush(QtGui.QColor(c.bubble)))
            band.setZValue(-2)
            self.bands[st] = band
            self.heads[st] = self._text(x + 6, 6, st, 13, True, c.stage[st][1])
            self.titles[st] = self._text(x + 6, 30, "", 8, True)
            label, tip = PARTS[st]
            self.parts[st] = _Part(s, "rect", x + 12, Y_PART, COL - 32, H_PART, label, tip)
            self.values[st] = _Value(s, x + 8, Y_PART + H_PART + 10, 8)
            self._clickable(f"24k:{st}", [self.parts[st].item, self.parts[st].label, band],
                            (x + 12, Y_PART))
        for a, b in zip(STAGES, STAGES[1:]):
            xa = _x(a) + COL - 20
            self.wires[f"{a}-{b}"] = _Wire(s, [(xa, Y_WIRE), (_x(b) + 12, Y_WIRE)])
        # the fetch unit's group and the ALU / LSU pipe
        self._text(X0 + 6, 270, "fetch unit (IFU): two instructions a cycle", 8, False,
                   c.muted_text)
        self._text(_x("RF") + 6, 270, "integer pipeline (ALU) and load / store unit (LSU)",
                   8, False, c.muted_text)
        # redirect: from EX back to IF (misprediction, exception, ERET)
        ex, iff = _x("EX") + COL // 2, _x("IF") + COL // 2
        self.wires["redirect"] = _Wire(s, [(ex, Y_PART), (ex, 80), (iff, 80), (iff, Y_PART)],
                                       "redirect: a misprediction (from EX, fetched in "
                                       "the cycle after MS), an exception (ER), ERET")
        self.values["redirect"] = _Value(s, iff + 40, 62, 8)
        # bypasses into AG / EX
        ag = _x("AG") + COL // 2
        self.wires["bypass_ex"] = _Wire(s, [(_x("EX") + 40, Y_PART + H_PART),
                                            (_x("EX") + 40, 236), (ag, 236),
                                            (ag, Y_PART + H_PART)],
                                        "bypass EX -> AG: an ALU result for the next "
                                        "instruction's ALU (no wait), its address a "
                                        "cycle later")
        self.wires["bypass_ms"] = _Wire(s, [(_x("MS") + 40, Y_PART + H_PART),
                                            (_x("MS") + 40, 252), (ag - 30, 252),
                                            (ag - 30, Y_PART + H_PART)],
                                        "bypass MS -> AG: load data (one cycle to an "
                                        "ALU user, two to an address)")
        # the instruction buffer
        ib_head = self._text(X0 + 6, 300, "instruction buffer (IS / IT)", 9, True)
        self.ib: list[QtWidgets.QGraphicsSimpleTextItem] = []
        self.ib_boxes = []
        for i in range(10):
            y = 322 + i * 22
            box = s.addRect(X0 + 6, y, 3 * COL - 30, 20, QtGui.QPen(QtGui.QColor(c.idle_line)),
                            QtGui.QBrush(QtGui.QColor(c.part_fill)))
            box.setZValue(1)
            self.ib_boxes.append(box)
            self.ib.append(self._text(X0 + 12, y + 3, "", 8))
        self._clickable("24k:buffer", [ib_head] + self.ib + self.ib_boxes, (X0 + 6, 300))
        # predictors
        px = _x("RF")
        head = self._text(px + 6, 300, "branch prediction", 9, True)
        self.predict = self._text(px + 6, 322, "", 8)
        self._clickable("24k:predict", [head, self.predict], (px + 6, 300))
        # units and hazards
        ux = _x("EX")
        head = self._text(ux + 6, 300, "units", 9, True)
        self.units = self._text(ux + 6, 322, "", 8)
        self._clickable("24k:units", [head, self.units], (ux + 6, 300))
        head = self._text(ux + 6, 470, "CP0 writes on their way (hazards)", 9, True)
        self.hazards = self._text(ux + 6, 492, "", 8)
        self._clickable("24k:hazards", [head, self.hazards], (ux + 6, 470))
        # status and events
        self.status = self._text(X0 + 6, 610, "", 10, True)
        self.events = self._text(X0 + 6, 636, "", 8, False, c.muted_text)
        self._text(X0 + 6, H - 22, "stage colours: fetch (IF IS IT), decode (RF), execute "
                   "(AG EX), memory (MS), completion (ER WB); red: where the pipeline is "
                   "held, the redirect wire", 8, False, c.muted_text)

    # -- refresh ---------------------------------------------------------------------------

    def refresh(self, s: Session) -> None:
        c = T()
        for st in STAGES:
            self.bands[st].setBrush(QtGui.QBrush(QtGui.QColor(c.bubble)))
            self.titles[st].setText("")
            self.parts[st].set(None)
            self.values[st].set("")
        for w in self.wires.values():
            w.set(None)
        self.values["redirect"].set("")
        p = s.pipeline
        if p is None or not cv.is_24k(p):
            self.status.setText("pipeline mode with the 24K model shows it here")
            for t in (self.predict, self.units, self.hazards, self.events):
                t.setText("")
            for item in self.ib:
                item.setText("")
            return
        details = s.stage_details()
        ev = details.pop("_events", {})
        trace = p.history[-1] if p.history else None
        stall = ev.get("stall") or ""
        kind = stall.split(":", 1)[0] if stall else ""
        held = STALL_STAGE.get(kind, "AG") if stall else None
        for st in STAGES:
            d = details.get(st) or {}
            if d.get("pc") is None:
                self.titles[st].setText("(bubble)")
                continue
            self.bands[st].setBrush(QtGui.QBrush(QtGui.QColor(c.stage[st][0])))
            self.titles[st].setText(_clip(d.get("text", ""), 21))
            self.parts[st].set(c.stage[st][1])
            self.values[st].set(self._value(st, d))
        for a, b in zip(STAGES, STAGES[1:]):
            if (details.get(b) or {}).get("pc") is not None:
                self.wires[f"{a}-{b}"].set(c.stage[b][1])
        if held is not None and held in self.parts:
            self.parts[held].set(c.red, c.red_fill)
            self.values[held].set(_clip(stall.split(":", 1)[-1].strip(), 24), c.red)
        events = ev.get("events") or []
        redirect = next((e for e in events if "kill the younger" in e or
                         "fetch the handler" in e or "return to" in e), None)
        if redirect is not None:
            self.wires["redirect"].set(c.red)
            self.values["redirect"].set(_clip(redirect, 60), c.red)
        if trace is not None:
            view = details.get("EX", {}).get("view")
            if view is not None and view.alu is not None and view.alu.a is not None:
                self.wires["bypass_ex"].set(c.stage["EX"][1], 1.6)
        # the buffer
        rows = cv.buffer_rows(p)
        for i, (item, box) in enumerate(zip(self.ib, self.ib_boxes)):
            if i < len(rows):
                r = rows[i]
                item.setText(f"{r.where} {r.pc:08x} {_clip(r.text, 26)}"
                             + (" (slot)" if r.in_slot else ""))
                box.setBrush(QtGui.QBrush(QtGui.QColor(c.stage[r.where][0])))
            else:
                item.setText("")
                box.setBrush(QtGui.QBrush(QtGui.QColor(c.part_fill)))
        self.predict.setText(_predict_text(p))
        self.units.setText("\n".join(f"{r.unit:4s} {'BUSY' if r.busy else 'free'}  "
                                     f"{_clip(r.state, 60)}" for r in cv.unit_rows(p)))
        hz = cv.hazard_rows(p)
        self.hazards.setText("\n".join(
            f"{h.register} by {_clip(h.producer, 22)}: seen from cycle {h.visible['other']}"
            for h in hz[:5]) or "none")
        status = f"cycle {ev.get('cycle', p.cycles)}   {cv.fetch_state(p)}"
        if stall:
            status += f"   STALL — {stall}"
        self.status.setText(_clip(status, 150))
        self.status.setBrush(QtGui.QBrush(QtGui.QColor(c.red if stall else c.text)))
        self.events.setText("\n".join(_clip(e, 150) for e in events[:6]))

    @staticmethod
    def _value(st: str, d: dict) -> str:
        v = d.get("view")
        if st == "IF":
            return f"PC {d['pc']:08x}"
        if v is None:
            return ""
        if st == "AG" and v.address is not None:
            return f"addr {v.address.address:08x}"
        if st == "EX":
            if v.exception:
                return f"EXCEPTION {v.exception}"
            if v.branch is not None:
                return "taken" if v.branch.taken else "not taken"
            if v.alu is not None and v.alu.result is not None:
                return f"= {v.alu.result:08x}"
        if st == "MS" and (v.mem_reads or v.mem_writes):
            if v.mem_reads:
                a, n, val = v.mem_reads[0]
                return f"read {val:0{2 * n}x}"
            a, n, _, new = v.mem_writes[0]
            return f"write {new:0{2 * n}x}"
        if st == "WB" and v.reg_writes:
            name, _, new = v.reg_writes[0]
            return f"{name} <- {new:08x}"
        return ""


def _predict_text(p) -> str:
    lines = []
    if p.last_prediction is not None:
        pc, text, how, nxt = p.last_prediction
        lines.append(f"last: {_clip(text, 26)}")
        lines.append(f"  {how} -> " + (f"{nxt:08x}" if nxt is not None else "wait"))
    if p.last_resolution is not None:
        pc, text, taken, right = p.last_resolution
        lines.append(f"resolved: {_clip(text, 22)}")
        lines.append(f"  {'taken' if taken else 'not taken'}, "
                     + ("right" if right else "MISPREDICTED"))
    lines.append(f"{p.predicted} resolved, {p.mispredicts} mispredicted")
    rows = cv.bht_rows(p)
    lines.append(f"BHT: {len(rows)} of 512 used")
    for i, n, state in rows[-3:]:
        lines.append(f"  [{i:3d}] {state}")
    lines.append("return stack: " + " ".join(
        ("[" if top else "") + f"{a:08x}" + ("]" if top else "") for _, a, top in cv.rps_rows(p)))
    return "\n".join(lines)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"
