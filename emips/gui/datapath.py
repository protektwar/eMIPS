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

from typing import Any

from ..datapathinfo import PARTS as _PARTS
from ..datapathinfo import control_of as _control_of, dest_of as _dest_of
from ..datapathinfo import nano_of as _nano_of
from ..datapathinfo import details as _details_of, taken as _taken_of
from ..datapathinfo import tlb_lines as _tlb_lines_of, word_of as _word
from ..syntax import gpr_name
from .qt import QtCore, QtGui, QtWidgets, Signal
from ..session import Session
from . import theme
from .widgets import mono_font

Qt = QtCore.Qt
W, H = 1500, 790
STAGES = ("IF", "ID", "EX", "MEM", "WB")
STAGE_X = {"IF": (0, 270), "ID": (282, 600), "EX": (612, 920),
           "MEM": (932, 1170), "WB": (1182, 1500)}
BARS = {"IF/ID": 270, "ID/EX": 600, "EX/MEM": 920, "MEM/WB": 1170}
T = theme.colors

# lanes (y)
Y_BR, Y_ADD, Y_CTRL = 112, 165, 92
Y_A, Y_B, Y_IMM, Y_DST = 300, 370, 450, 525
Y_FWD, Y_WB, Y_WREG = 610, 700, 725


def _font(size: int = 9, bold: bool = False) -> QtGui.QFont:
    f = mono_font()
    f.setPointSize(size)
    f.setBold(bold)
    return f


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
    """Scene with parts, wires and value labels; :meth:`refresh` colours
    them from the session.  A click on a part emits :attr:`part_clicked`
    (its key in :data:`emips.datapathinfo.PARTS`); :meth:`show_numbers`
    puts the guide's part numbers (1-17) on the drawing."""

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

        def part(key, *a):
            p = self.parts[key] = _Part(s, *a)
            for item in (p.item, p.label):          # clickable: what happens inside
                item.setData(0, key)
                item.setCursor(Qt.CursorShape.PointingHandCursor)
                item.setToolTip(item.toolTip() + "\n(click: what happens inside)")

        # IF
        part("pcmux", "mux", 16, 270, 22, 70, "M", "PC source: PC + 4, or the "
             "branch / jump target decided in ID")
        part("pc", "rect", 52, 283, 42, 44, "PC", "Program counter: address of "
             "the instruction being fetched")
        part("add4", "ellipse", 112, 145, 44, 40, "+4", "Adds 4: address of the "
             "next instruction (PC + 4)")
        part("imem", "rect", 120, 255, 124, 110, "Instruction\nmemory",
             "Instruction memory: reads the 32-bit word at PC")
        # ID
        part("control", "ellipse", 322, Y_CTRL - 24, 116, 48, "Control",
             "Main control unit: decodes the opcode / function into the control "
             "signals carried along the pipeline")
        part("regs", "rect", 344, 270, 126, 155, "Registers",
             "Register file: two read ports (rs, rt) in ID, one write port used "
             "by WB (written in the first half of the cycle, read in the second)")
        part("sext", "ellipse", 356, Y_IMM - 20, 96, 40, "Sign-\nextend",
             "Sign- (or zero-) extends the 16-bit immediate to 32 bits")
        part("btarget", "ellipse", 512, Y_ADD - 20, 76, 40, "Branch\ntarget",
             "PC + 4 + (offset << 2): the branch target, computed in ID")
        part("cmp", "ellipse", 506, 316, 44, 40, "=", "Branch comparator in ID: "
             "decides the branch (operands may be forwarded from EX/MEM)")
        part("hazard", "rect", 300, Y_FWD - 23, 150, 46, "Hazard\ndetection",
             "Hazard detection unit: stalls ID (bubble into EX) when an operand "
             "is not ready yet (load-use, branch operands, multiply/divide busy)")
        # EX
        part("fwda", "mux", 640, 272, 24, 62, "A", "Forwarding mux A: register "
             "value, or the result forwarded from EX/MEM or MEM/WB")
        part("fwdb", "mux", 640, 352, 24, 62, "B", "Forwarding mux B: register "
             "value, or the result forwarded from EX/MEM or MEM/WB")
        part("alusrc", "mux", 708, 372, 24, 62, "Src", "ALUSrc mux: operand b is "
             "the register value or the sign-extended immediate")
        part("alu", "alu", 768, 262, 84, 176, "ALU", "Arithmetic-logic unit")
        part("regdst", "mux", 740, 500, 24, 54, "Dst", "RegDst mux: destination "
             "register is rt (I-type) or rd (R-type), or $ra for links")
        part("fwdu", "rect", 680, Y_FWD - 23, 160, 46, "Forwarding\nunit",
             "Forwarding unit: compares the destination registers in EX/MEM and "
             "MEM/WB with the operands in EX and drives muxes A and B")
        # MEM
        part("dmem", "rect", 984, 270, 136, 155, "Data\nmemory",
             "Data memory: loads read and stores write at the ALU-computed address")
        # the level below the L1 caches (shown with caches; one unified L2,
        # drawn under each L1 as in Patterson & Hennessy's cache figures)
        part("below_if", "rect", 120, 600, 124, 40, "L2 cache", "")
        part("below_mem", "rect", 984, 600, 136, 40, "L2 cache", "")
        # WB
        part("wbmux", "mux", 1240, 312, 26, 156, "WB", "MemtoReg mux: write back "
             "the loaded data or the ALU result")

        self.wires: dict[str, _Wire] = {}

        def wire(key, pts, tip=""):
            self.wires[key] = _Wire(s, pts, tip)

        # IF
        wire("pcmux_pc", [(38, 305), (52, 305)])
        wire("pc_imem", [(94, 305), (120, 305)])
        wire("pc_add", [(102, 305), (102, Y_ADD), (112, Y_ADD)])
        wire("add_ifid", [(156, Y_ADD), (270, Y_ADD)], "PC + 4")
        wire("add_pcmux", [(164, Y_ADD), (164, 132), (8, 132), (8, 286), (16, 286)],
             "PC + 4 back to the PC mux")
        wire("imem_ifid", [(244, 310), (270, 310)], "the instruction word")
        wire("refill_if", [(232, 600), (232, 365)], "a missing line comes from the level below")
        # ID
        wire("ifid_ctrl", [(282, Y_CTRL), (322, Y_CTRL)], "opcode / funct")
        wire("ctrl_idex", [(438, Y_CTRL), (600, Y_CTRL)], "control signals")
        wire("ifid_rs", [(282, 288), (344, 288)], "rs")
        wire("ifid_rt", [(282, 318), (344, 318)], "rt")
        wire("rd1_idex", [(470, Y_A), (600, Y_A)], "read data 1 (rs)")
        wire("rd2_idex", [(470, Y_B), (600, Y_B)], "read data 2 (rt)")
        wire("ifid_sext", [(282, Y_IMM), (356, Y_IMM)], "immediate (16 bits)")
        wire("sext_idex", [(452, Y_IMM), (600, Y_IMM)], "immediate (32 bits)")
        wire("sext_bt", [(476, Y_IMM), (476, 214), (550, 214), (550, Y_ADD + 20)],
             "offset for the branch target")
        wire("ifid_dst", [(282, Y_DST), (600, Y_DST)], "rt and rd register numbers")
        wire("ifid_bt", [(282, Y_ADD), (512, Y_ADD)], "PC + 4")
        wire("rd1_cmp", [(486, Y_A), (486, 326), (506, 326)])
        wire("rd2_cmp", [(494, Y_B), (494, 346), (506, 346)])
        wire("bt_pcmux", [(588, Y_ADD), (594, Y_ADD), (594, Y_BR), (4, Y_BR), (4, 324),
                          (16, 324)], "branch / jump target to the PC mux")
        wire("ifid_hazard", [(282, Y_FWD), (300, Y_FWD)])
        # EX
        wire("idex_fwda", [(612, Y_A), (640, Y_A)])
        wire("fwda_alu", [(664, Y_A), (768, Y_A)], "ALU operand a")
        wire("idex_fwdb", [(612, Y_B), (640, Y_B)])
        wire("fwdb_src", [(664, 386), (708, 386)])
        wire("imm_src", [(612, Y_IMM), (690, Y_IMM), (690, 418), (708, 418)], "immediate")
        wire("src_alu", [(732, 403), (768, 403)], "ALU operand b")
        wire("store_data", [(682, 386), (682, 470), (920, 470)], "store data (rt)")
        wire("alu_exmem", [(852, 350), (920, 350)], "ALU result")
        wire("idex_dst", [(612, Y_DST), (740, Y_DST)])
        wire("dst_exmem", [(764, Y_DST + 2), (920, Y_DST + 2)], "destination register")
        wire("fwd_ex_a", [(932, 350), (948, 350), (948, 572), (630, 572), (630, 290),
                          (640, 290)], "forwarded from EX/MEM")
        wire("fwd_ex_b", [(932, 350), (948, 350), (948, 572), (630, 572), (630, 372),
                          (640, 372)], "forwarded from EX/MEM")
        wire("fwd_wb_a", [(1290, 390), (1290, 586), (622, 586), (622, 312), (640, 312)],
             "forwarded from MEM/WB")
        wire("fwd_wb_b", [(1290, 390), (1290, 586), (622, 586), (622, 396), (640, 396)],
             "forwarded from MEM/WB")
        wire("fwd_ex_id", [(948, 350), (948, 228), (500, 228), (500, 336), (506, 336)],
             "forwarded from EX/MEM to the branch comparator in ID")
        # MEM
        wire("addr", [(932, 350), (960, 350), (960, 300), (984, 300)], "address")
        wire("wdata", [(932, 470), (968, 470), (968, 390), (984, 390)], "write data")
        wire("rdata", [(1120, 330), (1170, 330)], "read data")
        wire("bypass", [(960, 350), (960, 450), (1170, 450)], "ALU result (not a load)")
        wire("dst_memwb", [(932, Y_DST + 2), (1170, Y_DST + 2)])
        wire("refill_mem", [(1108, 600), (1108, 425)],
             "a missing line comes from the level below (a dirty victim goes down)")
        # WB
        wire("memwb_mem", [(1182, 330), (1240, 330)])
        wire("memwb_alu", [(1182, 450), (1240, 450)])
        wire("wb_data", [(1266, 390), (1320, 390), (1320, Y_WB), (318, Y_WB), (318, 405),
                         (344, 405)], "write-back data to the register file")
        wire("wb_reg", [(1182, Y_DST + 2), (1340, Y_DST + 2), (1340, Y_WREG), (308, Y_WREG),
                        (308, 382), (344, 382)], "write-back register number")

        # port labels on the register file and data memory
        for x, y, t in ((347, 280, "rs"), (347, 310, "rt"), (347, 374, "write reg"),
                        (347, 396, "write data"), (436, 292, "rd1"), (436, 362, "rd2"),
                        (987, 292, "addr"), (987, 382, "wdata"), (1084, 322, "rdata")):
            self._text(x, y, t, 7, False, c.idle_text)

        self.values: dict[str, _Value] = {}
        for key, x, y, size in (
                ("pc", 48, 332, 9), ("word", 124, 370, 9), ("ctrl", 290, 118, 8),
                ("regs", 290, 252, 8), ("imm", 290, 474, 8), ("branch", 290, 190, 8),
                ("aluop", 772, 240, 9), ("a", 616, 248, 8), ("b", 616, 438, 8),
                ("res", 856, 316, 8), ("dst", 768, 532, 8), ("mem", 940, 434, 8),
                ("wb", 1190, 556, 9), ("status", 10, 764, 10), ("events", 470, 646, 8),
                ("icache", 20, 392, 8), ("dcache", 940, 540, 8),
                ("below_if", 20, 646, 8), ("below_mem", 990, 646, 8)):
            self.values[key] = _Value(s, x, y, size)
        # the guide's part numbers (View > Datapath Part Numbers)
        self.badges = []
        seen = set()
        for key, p in self.parts.items():
            info = _PARTS.get(key)
            if info is None or info.number is None or info.number in seen:
                continue
            seen.add(info.number)
            x0, y0, _, _ = p.box
            dot = s.addEllipse(x0 - 11, y0 - 11, 22, 22, QtGui.QPen(QtGui.QColor(c.red), 1.5),
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
        for p in self.parts.values():
            p.set(None)
        for w in self.wires.values():
            w.set(None)
        for v in self.values.values():
            v.set("")
        for st in STAGES:
            self.titles[st].setText("")
            self.bands[st].setBrush(QtGui.QBrush(QtGui.QColor(T().bubble)))
        self._cache_parts(s)
        nano = bool(getattr(s.machine, "nano", False))
        if self.parts["add4"].label.text() != ("+2/4/6" if nano else "+4"):
            self.parts["add4"].rename(           # nanoMIPS: 16 / 32 / 48-bit instructions
                *(("+2/4/6", "Adds the instruction's size (2, 4 or 6 bytes): the address "
                   "of the next instruction\n(click: what happens inside)") if nano else
                  ("+4", "Adds 4: address of the next instruction (PC + 4)\n(click: what "
                   "happens inside)")))
        if s.machine is None:
            self.values["status"].set("no program loaded — open a file or pick File > Examples")
            return
        details, ev = self._details(s)
        for st in STAGES:
            d = details.get(st) or {}
            if d.get("pc") is None:
                self.titles[st].setText("(bubble)" if s.pipeline is not None else "")
                continue
            self.bands[st].setBrush(QtGui.QBrush(QtGui.QColor(T().stage[st][0])))
            self.titles[st].setText(_clip(d["text"], 28))
            if d.get("control") is None:
                d = dict(d, control=_control_of(s, d))
            if st in ("IF", "ID") or _word(s, d) != 0:     # a nop does nothing past ID
                getattr(self, f"_{st.lower()}")(s, d, ev)
            if st in ("IF", "MEM") and (d.get("cache") or d.get("tlb")):
                self._cache(st, d.get("cache") or [], d.get("wait", 0), d.get("tlb") or [])
        events = ev.get("events") or []
        for e in events:                      # forwarding not seen in an operand
            if "EX/MEM -> EX" in e:
                self._light(["fwdu"], T().red)
            elif "MEM/WB -> EX" in e:
                self._light(["fwdu"], T().red)
        status = ev.get("label") or ""
        if ev.get("cycle") is not None:
            status = f"cycle {ev['cycle']}"
        if ev.get("stall"):
            status += f"   STALL — {ev['stall']}"
            if not ev["stall"].startswith(("memory", "I-miss")):   # not the hazard unit
                self.parts["hazard"].set(T().red, T().red_fill)
                self.wires["ifid_hazard"].set(T().red)
        self.values["status"].set(status, T().red if ev.get("stall") else None)
        self.values["events"].set("\n".join(_clip(e, 84) for e in events[:3]),
                                  T().muted_text)

    def _details(self, s: Session) -> tuple[dict, dict]:
        return _details_of(s)

    # caches ------------------------------------------------------------------

    def _cache_parts(self, s: Session) -> None:
        """Name the memories after the caches the machine has; the boxes
        of the level below show only with caches."""
        caches = s.machine.caches if s.machine is not None else None
        below = None
        if caches is not None and caches.all:
            below = ("L2 cache" if caches.l2 is not None and not caches.l2_bypass
                     else "L3 cache" if caches.l3 is not None else "Memory")
        for st, key, l1, what in (("if", "imem", "i", "Instruction"),
                                  ("mem", "dmem", "d", "Data")):
            c = getattr(caches, l1) if caches is not None else None
            if c is not None:
                self.parts[key].rename(
                    f"{c.name}\n{c.geometry.spec()}",
                    f"{c.name} ({c.geometry.describe()}): a hit delivers in the cycle; "
                    "a miss fills the line from the level below first. See the "
                    "Caches pane for its sets and ways.")
            else:
                self.parts[key].rename(
                    f"{what}\nmemory",
                    "Instruction memory: reads the 32-bit word at PC" if l1 == "i" else
                    "Data memory: loads read and stores write at the ALU-computed address")
            self.parts[f"below_{st}"].show(below is not None)
            self.wires[f"refill_{st}"].show(below is not None)
            if below is not None:
                self.parts[f"below_{st}"].rename(
                    below, f"{below}: where misses are served from (one unified level, "
                    "drawn under each L1 cache); uncached accesses go straight to memory")

    @staticmethod
    def _tlb_lines(translations: list) -> list[str]:
        """What the micro-TLB / TLB (or BAT) did for the access."""
        return _tlb_lines_of(translations)[:3]

    def _cache(self, st: str, accesses: list, wait: int, translations: list = ()) -> None:
        """Show what the micro-TLB and the caches did for the access in
        IF / MEM."""
        key, val = ("imem", "icache") if st == "IF" else ("dmem", "dcache")
        low = "below_if" if st == "IF" else "below_mem"
        c = T()
        lines, below = self._tlb_lines(translations), []
        for a in accesses:
            if a.level in ("I", "D"):
                if a.hit:
                    lines.append(f"{a.level}$ hit: set {a.set} way {a.way}")
                else:
                    lines.append(f"{a.level}$ MISS ({a.miss or 'no allocate'})")
                    lines.append(f"  set {a.set}" + (f" -> way {a.way}" if a.way is not None
                                                     else ""))
                    self._light([key], c.red, c.red_fill)
            elif a.level == "U":
                lines.append("no cache: memory" if a.miss == "no cache"
                             else "uncached: memory")
            else:
                name = "L2" if a.level == "S" else "L3"
                below.append(f"{name} {a.kind}: " + ("hit" if a.hit else "miss"))
        if wait:
            lines.append(f"waiting: {wait} more cycle(s)")
        bad = any("MISS" in t or "TLB " in t and t.startswith("  ") for t in lines)
        self.values[val].set("\n".join(lines), c.red if bad else None)
        if below:
            self._light([f"refill_{st.lower()}", low], c.red if wait else c.stage[st][1])
            self.values[low].set("\n".join(below[:3]))

    def _light(self, keys, color, fill=None) -> None:
        for k in keys:
            if k in self.parts:
                self.parts[k].set(color, fill)
            else:
                self.wires[k].set(color)

    @staticmethod
    def _taken(d, ev) -> bool:
        return _taken_of(d, ev)

    # per stage -------------------------------------------------------------

    def _if(self, s, d, ev) -> None:
        c = T().stage["IF"][1]
        self._light(["pcmux", "pc", "imem", "add4", "pcmux_pc", "pc_imem", "pc_add",
                     "add_ifid", "imem_ifid"], c)
        if not self._taken(d, ev):
            self._light(["add_pcmux"], c)
        self.values["pc"].set(f"PC {d['pc']:0{s.hex_digits('pc')}x}")
        word = _word(s, d)
        i = _nano_of(s, d)
        if i is not None:                       # nanoMIPS: its halfwords
            self.values["word"].set(" ".join(f"{(i.raw >> (16 * k)) & 0xFFFF:04x}"
                                             for k in range(i.size // 2 - 1, -1, -1)))
        elif d.get("compressed"):               # MIPS16e / microMIPS: the encoding, recoded
            raw = d["compressed"].split()[-1][2:]
            self.values["word"].set(f"{raw} -> {word:08x}" if word is not None else raw)
        elif word is not None:
            self.values["word"].set(f"{word:08x}")

    def _id(self, s, d, ev) -> None:
        c = T().stage["ID"][1]
        ctl = d.get("control") or {}
        self._light(["control", "regs", "ifid_ctrl", "ctrl_idex", "ifid_rs", "ifid_rt",
                     "rd1_idex", "rd2_idex", "ifid_dst"], c)
        if ctl:
            items = [f"{k}={_short(v)}" for k, v in ctl.items() if k != "ALUOp"]
            self.values["ctrl"].set("\n".join("  ".join(items[i:i + 3])
                                               for i in range(0, len(items), 3)))
        word = _word(s, d)
        i = _nano_of(s, d)
        if i is not None:                       # nanoMIPS: what the format reads
            from ..simnano import uses
            reads = [gpr_name(n, s.machine.names) for n in uses(i)[0]]
            self.values["regs"].set("read " + ("  ".join(reads[:3]) + (" …" if len(reads) > 3
                                                                      else "")
                                               if reads else "-"))
            if ctl.get("ALUSrc") == "imm" or ctl.get("Branch"):
                imm = _nano_imm(i)
                if imm is not None:
                    self._light(["sext", "ifid_sext", "sext_idex"], c)
                    self.values["imm"].set(f"imm {imm}")
        elif word is not None:
            rs, rt, imm = (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
            abi = "n64" if s.wide else True     # (MIPS64: the n64 names)
            self.values["regs"].set(f"rs={gpr_name(rs, abi)}  rt={gpr_name(rt, abi)}")
            if ctl.get("ALUSrc") == "imm" or ctl.get("Branch"):
                self._light(["sext", "ifid_sext", "sext_idex"], c)
                simm = imm - 0x10000 if imm & 0x8000 else imm
                self.values["imm"].set(f"imm {simm}")
        if ctl.get("Branch") or ctl.get("Jump"):
            self._light(["btarget", "ifid_bt", "cmp", "rd1_cmp", "rd2_cmp", "sext_bt"], c)
            if self._taken(d, ev):
                self._light(["bt_pcmux"], c)
            v = d.get("view")
            if v is not None and v.branch is not None:
                br = v.branch
                self.values["branch"].set(
                    _clip(br.condition, 26) + ("\ntaken -> " + f"{br.target:0{s.hex_digits()}x}"
                                               if br.taken else "\nnot taken"))
        if any("-> ID" in e for e in ev.get("events") or []):
            self._light(["fwd_ex_id"], T().red)

    def _ex(self, s, d, ev) -> None:
        c = T().stage["EX"][1]
        self._light(["fwda", "fwdb", "alu", "idex_fwda", "idex_fwdb", "fwda_alu",
                     "fwdb_src", "alusrc", "src_alu", "alu_exmem", "idex_dst"], c)
        v = d.get("view")
        if v is None:
            return
        if v.alu is not None:
            a, b = v.alu.a, v.alu.b
            self.values["aluop"].set(_clip(v.alu.op, 30))
            if a is not None:
                self.values["a"].set(f"a {a.value:0{s.hex_digits()}x} ({_src(a.source)})")
                self._forward("a", a.source)
            if b is not None:
                self.values["b"].set(f"b {b.value:0{s.hex_digits()}x} ({_src(b.source)})")
                if "immediate" in b.source or "offset" in b.source:
                    self._light(["imm_src"], c)
                self._forward("b", b.source)
            if v.alu.result is not None:
                self.values["res"].set(f"= {v.alu.result:0{s.hex_digits()}x}"
                                       + ("\nOVERFLOW" if v.alu.overflow else ""))
        if v.mem_writes:
            self._light(["store_data"], c)
        dest = _dest(s, d)
        if dest is not None:
            self._light(["regdst", "dst_exmem"], c)
            self.values["dst"].set(f"dest {dest[0]}")
        if v.exception:
            self.values["res"].set(f"EXCEPTION\n{v.exception}", T().red)

    def _forward(self, operand: str, source: str) -> None:
        if "EX/MEM" in source:
            self._light([f"fwd_ex_{operand}", "fwdu"], T().red)
        elif "MEM/WB" in source:
            self._light([f"fwd_wb_{operand}", "fwdu"], T().red)

    def _mem(self, s, d, ev) -> None:
        c = T().stage["MEM"][1]
        v = d.get("view")
        if _dest(s, d) is not None:
            self._light(["dst_memwb"], c)
        if v is None or not (v.mem_reads or v.mem_writes):
            if _dest(s, d) is not None:
                self._light(["bypass"], c)
            return
        self._light(["dmem", "addr"], c)
        lines = []
        for a, n, val in v.mem_reads[:2]:
            lines.append(f"read  {a:08x}\n   = {val:0{2 * n}x}")
            self._light(["rdata"], c)
        for a, n, o, nw in v.mem_writes[:2]:
            lines.append(f"write {a:08x}\n  <- {nw:0{2 * n}x}")
            self._light(["wdata"], c)
        if v.address is not None and v.address.physical is not None:
            lines.append(f"PA    {v.address.physical:08x}")
        self.values["mem"].set("\n".join(lines))

    def _wb(self, s, d, ev) -> None:
        c = T().stage["WB"][1]
        dest = _dest(s, d)
        if dest is None:
            return
        v = d.get("view")
        load = bool((d.get("control") or {}).get("MemtoReg")) or (
            v is not None and bool(v.mem_reads) and not v.mem_writes)
        self._light(["wbmux", "wb_data", "wb_reg",
                     "memwb_mem" if load else "memwb_alu"], c)
        name, value = dest
        self.values["wb"].set(f"{name} <- {value:0{s.hex_digits()}x}" if value is not None
                              else f"{name} written")


def _dest(s, d):
    return _dest_of(d, s.machine.names)


#: the decoded arguments that are a nanoMIPS instruction's immediate, by
#: preference (emips.nanomips's decode steps)
_NANO_IMMEDIATES = ("imm", "offset", "u", "s", "shift")


def _nano_imm(i) -> int | None:
    from ..tracenano import _sx
    for k in _NANO_IMMEDIATES:
        if k in i.f:
            v = i.f[k]
            return _sx(v & 0xFFFFFFFF, 32) if k in ("offset", "s", "imm") else v
    return None


def _src(source: str) -> str:
    if "forwarded" in source:
        return "fwd " + ("EX/MEM" if "EX/MEM" in source else "MEM/WB")
    if "immediate" in source:
        return "imm"
    if "offset" in source:
        return "offset"
    if "shift" in source:
        return "sa"
    return source.split(" ")[0]


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _short(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)
