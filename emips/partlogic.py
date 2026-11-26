"""The gates in the datapath's part window, Qt-free (M29 W2): the PLAs of
the main control and the ALU control as Patterson & Hennessy draw them, one
ALU bit slice in gate symbols, and the text that goes with them — the
window's ``gui/partdialog.py`` / ``gui/logicview.py`` and the web page draw
the same :class:`Scene`s and show the same text.

A :class:`Scene` is a list of drawing operations (lines, texts, dots,
rectangles, gate symbols) with colour tokens: ``"lit"`` (the stage's
colour: a line carrying a 1), ``"idle_line"``, ``"idle_text"``, ``"text"``,
``"muted"``, ``"part_fill"``, ``"bar_stroke"``.  Each view resolves them in
its theme; a text's (x, y) is its top-left corner (rotated about it), or
its top centre with ``center``.

The values come from :mod:`emips.controllogic` / :mod:`emips.alulogic`
(evaluated on the instruction the trace shows); the text's HTML uses the
classes ``chosen`` (the row the Operation mux passes on), ``mono`` and
``mismatch``."""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from . import alulogic as al
from . import controllogic as cl
from . import datapathinfo as dpi
from . import nanocontrol as nc

E = html.escape


@dataclass
class Scene:
    """One drawing: ``ops`` in drawing order (``z``: lit lines over idle)."""
    lit: str                                   # the stage colour token ("stage:ID" ...)
    ops: list = field(default_factory=list)

    def line(self, x0, y0, x1, y1, on, width: float = 1.4) -> None:
        self.ops.append(["line", x0, y0, x1, y1, "lit" if on else "idle_line",
                         width + (1 if on else 0), 2 if on else 1])

    def text(self, x, y, s, size=8, bold=False, color=None, center=False, rot=0) -> None:
        self.ops.append(["text", x, y, str(s), size, bold, color or "text", center, rot])

    def dot(self, x, y, on, r: float = 3.2) -> None:
        self.ops.append(["dot", x, y, r, "lit" if on else "idle_text"])

    def frame(self, x, y, w, h) -> None:              # (a dashed box)
        self.ops.append(["rect", x, y, w, h, "bar_stroke"])

    def gate(self, kind: str, x, y, w, h, on, label: str = "") -> None:
        """and / or / xor / not / mux / box, outlined in "lit" when its output
        is 1."""
        self.ops.append(["gate", kind, x, y, w, h, "lit" if on else "idle_line",
                         2 if on else 1.4])
        if label:
            self.text(x + w / 2, y + h / 2 - 6, label, 8, True, center=True)

    def wire(self, pts, on, label: str = "", value: int | None = None) -> None:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.line(x0, y0, x1, y1, on)
        if label:
            x, y = pts[0]
            self.text(x - 4, y - 13, label + ("" if value is None else f" = {value}"), 7,
                      False, "lit" if on else None)

    def to_json(self) -> dict:
        return {"lit": self.lit, "ops": self.ops}


# -- a PLA ----------------------------------------------------------------------------------

COL, ROW = 18, 13


def pla_scene(pla, x: int | None, title: str, lit: str,
              labels: dict[str, str] | None = None) -> Scene:
    """A PLA, its inputs set to ``x`` (input 0 = the most significant bit):
    the AND plane (input columns, true and inverted, crossed by the product
    terms, a dot where a term takes a literal), the OR plane (the terms
    crossing the output columns); lit = 1."""
    sc = Scene(lit)
    n, terms, outs = pla.n, pla.terms, pla.outputs
    hit, values = pla.evaluate(x) if x is not None else ([False] * len(terms), {})
    bits = [None if x is None else (x >> (n - 1 - i)) & 1 for i in range(n)]
    C, R = COL, ROW
    top = 66 + max(len(name) for name in pla.inputs) * 6
    and_w = 2 * n * C
    or_x0 = and_w + 40
    bottom = top + max(1, len(terms)) * R + 10
    sc.text(0, 0, title, 10, True)
    sc.text(0, 16, "AND plane: a dot ANDs that input (true, or inverted ~) into the "
                   "product term of its row", 7, False, "muted")
    sc.text(0, 28, "OR plane: a dot ORs the term into the output of its column; lit "
                   "= 1", 7, False, "muted")
    for i, name in enumerate(pla.inputs):          # input columns: true and inverted
        for k, inv in enumerate((False, True)):
            cx = (2 * i + k) * C + C / 2
            on = None if bits[i] is None else bool(bits[i] ^ inv)
            sc.line(cx, top - 6, cx, bottom, on)
            if not inv:
                sc.text(cx + C / 2 - 6, top - 30, name, 7, True, rot=-90)
                if bits[i] is not None:
                    sc.text(cx, top - 20, str(bits[i]), 7, False,
                            "lit" if bits[i] else "muted", center=True)
                    sc.text(cx + C, top - 20, str(1 - bits[i]), 7, False,
                            "lit" if not bits[i] else "muted", center=True)
            if inv:
                sc.text(cx, top - 19 if bits[i] is None else top - 8, "~", 7, True,
                        "muted", center=True)
    for j, name in enumerate(outs):                # output columns
        cx = or_x0 + j * C + C / 2
        on = values.get(name) if x is not None else None
        sc.line(cx, top - 6, cx, bottom, on)
        sc.text(cx + 5, bottom + 4, (labels or {}).get(name, name), 7, True,
                "lit" if on else None, rot=90)
        if x is not None:
            sc.text(cx, top - 19, "1" if on else "0", 7, False, "lit" if on else "muted",
                    center=True)
    for k, term in enumerate(terms):               # term rows
        y = top + k * R + R / 2
        sc.line(0, y, or_x0 + len(outs) * C, y, hit[k] if x is not None else None, 0.8)
        for i in range(n):
            lit_ = pla.literal(term, i)
            if lit_ is not None:
                sc.dot((2 * i + (0 if lit_ else 1)) * C + C / 2, y, hit[k])
        for j, name in enumerate(outs):
            if k in pla.uses.get(name, []):
                sc.dot(or_x0 + j * C + C / 2, y, hit[k])
    sc.frame(-4, top - 8, and_w + 8, bottom - top + 12)       # the planes' frames
    sc.frame(or_x0 - 4, top - 8, len(outs) * C + 8, bottom - top + 12)
    sc.text(and_w / 2, bottom + 6, f"AND plane: {len(terms)} product terms", 7, False,
            "muted", center=True)
    return sc


# -- an ALU bit slice -----------------------------------------------------------------------

def slice_scene(sl, ctl, msb: bool, lit: str, top: int = 31, width: int = 32) -> Scene:
    """Bit ``sl`` (an :class:`emips.alulogic.Slice`) under ALU control
    ``ctl``; ``msb``: bit ``top`` (31; MIPS64: 63, a Word operation 31),
    with Overflow and Set.  A wire is lit when it carries a 1."""
    sc = Scene(lit)
    op = ctl.operation
    sc.text(0, -44, f"ALU bit {sl.i}: Ainvert {ctl.ainvert}, Bnegate {ctl.bnegate}, "
                    f"Operation {op:03b}" + (f", Unsigned {ctl.unsigned}" if msb else ""),
            10, True)
    # the inputs and their inverting muxes (Ainvert, Bnegate)
    for name, val, y, sel, out, label in (
            ("a", sl.a, 40, ctl.ainvert, sl.a_in, "Ainvert"),
            ("b", sl.b, 130, ctl.bnegate, sl.b_in, "Bnegate")):
        sc.wire([(0, y), (70, y), (70, y + 6), (90, y + 6)], val, f"{name}{sl.i}", val)
        sc.wire([(40, y), (40, y + 30), (50, y + 30)], val)
        sc.gate("not", 50, y + 22, 26, 16, not val)
        sc.wire([(76, y + 30), (90, y + 30)], 1 - val)
        sc.gate("mux", 90, y - 4, 22, 46, out)
        sc.text(94, y - 2, "0", 6, False, "muted")
        sc.text(94, y + 22, "1", 6, False, "muted")
        sc.wire([(56, y - 20), (101, y - 20), (101, y - 1)], sel, label, sel)
    ax, bx = 140, 156
    sc.wire([(112, 59), (ax, 59)], sl.a_in)
    sc.wire([(ax, 48), (ax, 298)], sl.a_in)
    sc.text(116, 62, f"a' = {sl.a_in}", 7, False, "lit" if sl.a_in else None)
    sc.wire([(112, 149), (bx, 149)], sl.b_in)
    sc.wire([(bx, 64), (bx, 314)], sl.b_in)
    sc.text(116, 152, f"b' = {sl.b_in}", 7, False, "lit" if sl.b_in else None)
    # AND, OR, the full adder and XOR (top to bottom, in the order of the mux
    # inputs they feed, so no two routes cross)
    gx = 210
    outs = {}
    for kind, y, val in (("and", 30, sl.and_), ("or", 85, sl.or_), ("xor", 290, sl.xor_)):
        sc.wire([(ax, y + 8), (gx, y + 8)], sl.a_in)
        sc.wire([(bx, y + 24), (gx, y + 24)], sl.b_in)
        sc.gate(kind, gx, y, 44, 32, val)
        sc.text(gx + 13, y + 9, kind.upper(), 7, True)
        outs[kind] = (gx + 44, y + 16, val)
    fy = 170
    sc.wire([(ax, fy + 16), (gx, fy + 16)], sl.a_in)
    sc.wire([(bx, fy + 40), (gx, fy + 40)], sl.b_in)
    sc.gate("box", gx, fy, 64, 64, sl.sum_, "+")
    sc.wire([(gx + 32, fy - 20), (gx + 32, fy)], sl.cin)
    sc.text(gx + 38, fy - 22, f"CarryIn = {sl.cin}"
            + (" (= Bnegate)" if sl.i == 0 else f" (bit {sl.i - 1}'s CarryOut)"), 7,
            False, "lit" if sl.cin else None)
    sc.wire([(gx + 32, fy + 64), (gx + 32, fy + 90)], sl.cout)
    sc.text(gx + 38, fy + 76, f"CarryOut = {sl.cout}"
            + (f" (to bit {sl.i + 1})" if sl.i < width - 1 else ""), 7, False,
            "lit" if sl.cout else None)
    outs["sum"] = (gx + 64, fy + 32, sl.sum_)
    # the result mux: 0 AND, 1 OR, 2 sum, 3 Less, 4 XOR, 5 shifter
    mx, my, mh = 460, 30, 300
    sc.gate("mux", mx, my, 34, mh, sl.result)
    ins = {k: my + 30 + k * 46 for k in range(6)}
    for k, key, xv in ((0, "and", 380), (1, "or", 388), (2, "sum", 396), (4, "xor", 404)):
        x0, y0, val = outs[key]
        sc.wire([(x0, y0), (xv, y0), (xv, ins[k]), (mx, ins[k])], val)
    less_y, sh_y = 380, 410
    sc.wire([(240, less_y), (420, less_y), (420, ins[3]), (mx, ins[3])], sl.less)
    sc.text(240, less_y - 13, f"Less = {sl.less}" + (f" (Set, from bit {top})" if sl.i == 0
                                                     else " (0)"), 7, False,
            "lit" if sl.less else None)
    sc.wire([(240, sh_y), (428, sh_y), (428, ins[5]), (mx, ins[5])], sl.shift)
    sc.text(240, sh_y - 13, f"shifter bit = {sl.shift}", 7, False,
            "lit" if sl.shift else None)
    names = ["AND", "OR", "sum", "Less", "XOR", "shifter"]
    for k in range(6):
        chosen = k == op
        sc.text(mx + 4, ins[k] - 7, f"{k} {names[k]}" if chosen else str(k), 7, chosen,
                "lit" if chosen else "muted")
    sc.wire([(mx + 17, my - 22), (mx + 17, my + 6)], True)
    sc.text(mx + 22, my - 26, f"Operation {op:03b} ({names[op] if op < 6 else '?'})", 7,
            True, "lit")
    ry = my + mh / 2
    sc.wire([(mx + 34, ry), (mx + 140, ry)], sl.result)
    sc.text(mx + 44, ry - 15, f"Result{sl.i} = {sl.result}", 8, True,
            "lit" if sl.result else None)
    if msb:                                     # Overflow and Set (bit top only),
        ox, oy = 600, 360                       # fed by named stubs (net labels)
        ovf = sl.cin ^ sl.cout
        setv = (1 - sl.cout) if ctl.unsigned else sl.sum_ ^ ovf
        for name, val, y in ((f"CarryIn{top}", sl.cin, oy + 8),
                             (f"CarryOut{top}", sl.cout, oy + 26),
                             (f"Sum{top}", sl.sum_, oy + 88), ("Overflow", ovf, oy + 106)):
            sc.wire([(ox - 34, y), (ox, y)], val)
            sc.text(ox - 34 - 7 * len(name), y - 7, name, 7, False, "lit" if val else None)
        sc.gate("xor", ox, oy, 42, 34, ovf)
        sc.wire([(ox + 42, oy + 17), (ox + 120, oy + 17)], ovf)
        sc.text(ox + 48, oy + 2, f"Overflow = {ovf}", 8, True, "lit" if ovf else None)
        sy = oy + 80
        sc.gate("xor", ox, sy, 42, 34, sl.sum_ ^ ovf)
        sc.wire([(ox + 42, sy + 17), (ox + 120, sy + 17)], setv)
        sc.text(ox + 48, sy + 2, f"Set = {setv}", 8, True, "lit" if setv else None)
        sc.text(ox - 34, sy + 44, f"Set = Sum{top} XOR Overflow (Unsigned: NOT "
                                  f"CarryOut{top})", 7, False, "muted")
        sc.text(ox - 34, sy + 56, "-> bit 0's Less", 7, False, "muted")
    return sc


# -- the part window's gates and text --------------------------------------------------------

@dataclass
class PartLogic:
    """What the part window shows beside the part's description:
    ``drawings`` ((tab name, Scene)), ``html`` (the text about them) and
    ``width`` (the ALU's slices: 32 / 64; 0 without slices)."""
    drawings: list = field(default_factory=list)
    html: str = ""
    width: int = 0


def part_logic(s, key: str, bit: int = 0, tr=lambda t: t) -> PartLogic:
    """The gates of part ``key`` for the cycle on the screen: the main
    control's decoders in ID (Control), the ALU control and bit ``bit``'s
    slice in EX (the ALU), the text that explains them."""
    out = PartLogic()
    msa = bool(getattr(s.machine, "msa", False))
    logic = al.alu_logic(dpi.alu_view(s, key), s.release, s.options.arch, msa,
                         s.options.rv_extensions)
    if logic is not None and logic.slices:
        out.width = logic.width
        bit = max(0, min(bit, logic.width - 1))
    if s.machine is not None and s.machine.rv:
        text = _rv(s, key, logic, bit, out, tr)
    else:
        text = _mips(s, key, logic, bit, out, msa, tr)
    out.html = text + (_alu_html(logic, bit, tr) if logic is not None else "")
    return out


def _mips(s, key, logic, bit, out, msa, tr) -> str:
    nano = bool(getattr(s.machine, "nano", False))
    word = None if nano else dpi.control_word(s, key)
    insn = dpi.control_nano(s, key) if nano else None
    text = ""
    if insn is not None:
        text = _nano_main(insn, out, tr)
    if word is not None:
        mc = cl.main_control(s.release, s.options.arch, msa)
        ev = cl.evaluate_main(mc, word)
        lit = "stage:ID"
        out.drawings.append([tr("Opcode decoder"), pla_scene(
            mc.opcode, word >> 26, f"Main control: the opcode decoder (op = {word >> 26:06b})",
            lit)])
        if ev.escape is not None:
            esc = ev.escape
            out.drawings.append([tr("{name} decoder").format(name=esc.escape), pla_scene(
                esc.pla, cl._bits(word, esc.fields),
                f"{esc.escape} decoder ({', '.join(esc.fields)}): enabled by the opcode "
                f"decoder's {esc.escape} line", lit)])
        sig = ", ".join(f"{k}={int(v)}" for k, v in ev.signals.items())
        text = (f"<p><b>{E(tr('The control signals'))}</b> (the opcode decoder's "
                + (f"OR the {E(ev.escape.escape)} decoder's " if ev.escape else "")
                + f"outputs): <code>{E(sig)}</code>; ALUOp = {ev.aluop:04b} "
                f"({E(al.aluop_why(ev.aluop, s.wide))})</p>")
    if logic is not None and logic.control is not None:
        ctl = logic.control
        lit = "stage:EX"
        view = dpi.alu_view(s, key)
        if nano:                                # ALUOp names the function; ALUSrc: ShVar
            src = nc.evaluate_main(nc.main_control(), view.word, view.size).signals["ALUSrc"]
            scene = pla_scene(nc.alu_control(), nc.alu_input(ctl.aluop, src),
                              f"ALU control: ALUOp = {ctl.aluop:04b}, ALUSrc = {int(src)}", lit)
        else:
            pla = cl.alu_control(s.release, s.options.arch, msa)
            scene = pla_scene(pla, cl.alu_input(ctl.aluop, view.word),
                              f"ALU control: ALUOp = {ctl.aluop:04b}, funct = "
                              f"{view.word & 0x3F:06b}", lit)
        out.drawings.append([tr("ALU control"), scene])
        if logic.slices:
            out.drawings.append([tr("ALU bit slice"), slice_scene(
                logic.slices[bit], ctl, bit == logic.top, lit, logic.top, logic.width)])
    return text


def _rv(s, key, logic, bit, out, tr) -> str:
    """A RISC-V hart (emips.rvcontrol): the opcode decoder and its secondary
    decoder in ID, the ALU control and a slice in EX."""
    from . import rvcontrol as rc
    xlen = s.machine.xlen
    exts = s.options.rv_extensions
    text = ""
    word = dpi.control_rv(s, key)
    if word is not None:
        mc = rc.main_control(xlen, exts)
        ev = rc.evaluate_main(mc, word)
        lit = "stage:ID"
        op = rc.opcode(word)
        out.drawings.append([tr("Opcode decoder"), pla_scene(
            mc.opcode, op, f"Main control: the opcode decoder (bits 6..0 = {op:07b}, "
            f"{rc.OPCODES.get(op >> 2, '?')})", lit)])
        if ev.escape is not None:
            esc = ev.escape
            out.drawings.append([tr("{name} decoder").format(name=esc.escape), pla_scene(
                esc.pla, ev.x, f"{esc.escape} decoder ({', '.join(esc.fields)}): enabled by "
                f"the opcode decoder's {esc.escape} line", lit)])
        sig = ", ".join(f"{k}={int(v)}" for k, v in ev.signals.items())
        text = (f"<p><b>{E(tr('The control signals'))}</b> (the opcode decoder's "
                + (f"OR the {E(ev.escape.escape)} decoder's " if ev.escape else "")
                + f"outputs): <code>{E(sig)}</code>; ALUOp = {ev.aluop:03b} "
                f"({E(rc.ALUOP_WHY.get(ev.aluop, ''))}); rd is always bits 11..7: no RegDst</p>")
    if logic is not None and logic.control is not None:
        ctl = logic.control
        lit = "stage:EX"
        view = dpi.alu_view(s, key)
        w = rc.view_word(view, xlen, exts)
        out.drawings.append([tr("ALU control"), pla_scene(
            rc.alu_control(xlen, exts), rc.alu_input(ctl.aluop, w),
            f"ALU control: ALUOp = {ctl.aluop:03b}, bits 30 29 27 25 = {(w >> 30) & 1}"
            f"{(w >> 29) & 1}{(w >> 27) & 1}{(w >> 25) & 1}, funct3 = {(w >> 12) & 7:03b}",
            lit)])
        if logic.slices:
            out.drawings.append([tr("ALU bit slice"), slice_scene(
                logic.slices[bit], ctl, bit == logic.top, lit, logic.top, logic.width)])
    return text


def _nano_main(insn, out, tr) -> str:
    """The main control of a nanoMIPS core (emips.nanocontrol): the major
    opcode decoder and the pool's decoder."""
    mc = nc.main_control()
    ev = nc.evaluate_main(mc, insn.raw, insn.size)
    maj = nc.major(insn.raw, insn.size)
    lit = "stage:ID"
    out.drawings.append([tr("Opcode decoder"), pla_scene(
        mc.opcode, maj, f"Main control: the major opcode decoder (bits 15..10 of the first "
        f"halfword = {maj:06b})", lit)])
    if ev.escape is not None:
        esc = ev.escape
        out.drawings.append([tr("{name} decoder").format(name=esc.escape), pla_scene(
            esc.pla, ev.x, f"{esc.escape} decoder ({', '.join(esc.fields)}): enabled by the "
            f"opcode decoder's {esc.escape} line", lit)])
    sig = ", ".join(f"{k}={int(v)}" for k, v in ev.signals.items())
    dst = (f"RegDst = {ev.regdst:03b} ({nc.REGDST[ev.regdst]})"
           if ev.signals["RegWrite"] else "RegDst: no register written")
    return (f"<p><b>{E(tr('The control signals'))}</b> (the opcode decoder's "
            + (f"OR the {E(ev.escape.escape)} decoder's " if ev.escape else "")
            + f"outputs): <code>{E(sig)}</code>; {E(dst)}; ALUOp = {ev.aluop:04b} "
            f"({E(nc.ALUOP_WHY.get(ev.aluop, ''))})</p>")


def _alu_html(lg: al.AluLogic, bit: int, tr) -> str:
    out = [f"<p><b>{E(tr('Inside the ALU'))}</b> — {E(lg.instruction)}</p>"]
    ctl = lg.control
    if ctl is not None and not ctl.other:
        funct = f" + funct {ctl.funct}" if ctl.funct else ""
        out.append(f"<p>ALU control: ALUOp = {ctl.aluop:0{ctl.aluop_bits}b} "
                   f"({E(ctl.aluop_why)}){funct} "
                   f"&rarr; Ainvert {ctl.ainvert}, Bnegate {ctl.bnegate}, Operation "
                   f"{ctl.operation:03b} = {E(al.OPERATION.get(ctl.operation, '?'))}"
                   + (f", Unsigned {ctl.unsigned}" if ctl.operation == 3 else "")
                   + (f", Word {ctl.word}" if lg.width == 64 else "")
                   + (f", Sh32 {ctl.sh32}" if lg.width == 64 and ctl.operation == 5 else "")
                   + " (the ALU control PLA above)</p>")
    rows = al.rows(lg)
    if rows:
        cells = []
        n = lg.width
        for label, value, chosen in rows:
            cls = " class='chosen'" if chosen else ""
            bold = ("<b>", "</b>") if label == "result" or chosen else ("", "")
            cells.append(f"<tr{cls}><td>{bold[0]}{E(label)}{bold[1]}</td>"
                         f"<td class='mono'>{bold[0]}{al.binary(value, n)}{bold[1]}</td>"
                         f"<td class='mono'>0x{value & ((1 << n) - 1):0{n // 4}x}</td></tr>")
        t = lg.top
        out.append("<table cellspacing=0 cellpadding=2>"
                   f"<tr><td></td><td><i>bit {lg.width - 1} &hellip; bit 0</i></td>"
                   "<td></td></tr>" + "".join(cells) + "</table>")
        out.append(f"<p>{E(tr('The highlighted row is what the Operation mux passes on.'))}"
                   f" Zero = NOR of the {lg.width} result bits = {lg.zero}; Overflow = "
                   f"carry into bit {t} XOR carry out = {lg.overflow}; Set = {lg.set_} "
                   f"(bit {t}'s sum corrected by overflow, wired to bit 0's Less)</p>")
        out.append(_slice_html(lg.slices[bit], ctl, lg.top, lg.width))
    if lg.shifter:
        n = lg.width
        cells = "".join(f"<tr><td>{E(label)}</td><td class='mono'>{al.binary(v, n)}</td></tr>"
                        for label, v in lg.shifter)
        stages = (f"5 stages of {n} 2:1 muxes" if n == 32 else
                  "6 stages of 64 2:1 muxes; a Word shift: the low word's 5, then bit 31 "
                  "sign-extended")
        out.append(f"<p>Barrel shifter ({stages}), input "
                   f"<code>{al.binary(lg.a if lg.instruction.split()[0] != 'lui' else lg.b, n)}"
                   f"</code>:</p><table cellpadding=2>{cells}</table>")
    out += [f"<p>{E(note)}</p>" for note in lg.notes]
    if lg.result is not None and not lg.matches:
        out.append(f"<p class='mismatch'>The gates give 0x{lg.result:0{lg.width // 4}x}, "
                   f"the trace recorded 0x{(lg.traced or 0):0{lg.width // 4}x}.</p>")
    return "".join(out)


def _slice_html(sl: al.Slice, ctl, top: int = 31, width: int = 32) -> str:
    less = (f"Less = {sl.less}" + (f" (Set, from bit {top})" if sl.i == 0 else " (0)"))
    op = ctl.operation
    lines = [
        f"a{sl.i} = {sl.a}, b{sl.i} = {sl.b}",
        f"Ainvert {ctl.ainvert}: a' = a XOR {ctl.ainvert} = {sl.a_in};  "
        f"Bnegate {ctl.bnegate}: b' = b XOR {ctl.bnegate} = {sl.b_in}",
        f"AND gate: a' AND b' = {sl.and_}",
        f"OR gate:  a' OR b'  = {sl.or_}",
        f"XOR gate: a' XOR b' = {sl.xor_}",
        f"full adder: sum = a' XOR b' XOR carry-in ({sl.cin}) = {sl.sum_}",
        f"            carry out = a'b' OR a'c OR b'c = {sl.cout}"
        + ("  (into bit {})".format(sl.i + 1) if sl.i < width - 1 else "  (out of the ALU)"),
        less,
        f"shifter bit {sl.i} = {sl.shift}",
        f"mux (Operation {op:03b}: {al.OPERATION.get(op, '?')}) -> result bit {sl.i} "
        f"= {sl.result}",
    ]
    if sl.i == top:
        ovf = sl.cin ^ sl.cout
        lines.append(f"bit {top} only: Overflow = carry-in {sl.cin} XOR carry-out {sl.cout}"
                     f" = {ovf}")
        lines.append(f"             Set = sum {sl.sum_} XOR Overflow {ovf} = "
                     f"{sl.sum_ ^ ovf}  -> bit 0's Less (Unsigned: NOT carry-out = "
                     f"{1 - sl.cout})")
    body = "<br>".join(E(x).replace("  ", "&nbsp;&nbsp;") for x in lines)
    return f"<p><b>Bit {sl.i}, one ALU slice:</b></p><p class='mono'>{body}</p>"
