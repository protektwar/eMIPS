"""The part window of the Datapath: click a part (PC, ALU, forwarding
mux A, ...) to see what it is and what it does in the cycle on the
screen; the window follows every step until closed.  The text comes
from :mod:`emips.datapathinfo` (the trace, never recomputed); the ALU's
gates from :mod:`emips.alulogic` (derived from the traced operands and
checked against the traced result)."""

from __future__ import annotations

import html

from .. import alulogic as al
from .. import controllogic as cl
from .. import datapathinfo as dpi
from .. import nanocontrol as nc
from ..session import Session
from . import theme
from .i18n import tr
from .logicview import PLAView, SliceView
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
        self.pla_main = PLAView()
        self.pla_second = PLAView()
        self.pla_alu = PLAView()
        self.slice = SliceView()
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
        logic = al.alu_logic(dpi.alu_view(s, self.key), s.release, s.options.arch)
        if logic is not None:                   # (MIPS64: 64 slices)
            self.bit.setRange(0, logic.width - 1)
            self.bit.setToolTip(tr("Which of the {n} one-bit ALU slices to show gate by "
                                   "gate").format(n=logic.width))
        inside = self._alu_html(logic, mono) if logic is not None else ""
        inside = self._drawings(s, logic) + inside
        has_slices = logic is not None and bool(logic.slices)
        self.bit.setVisible(has_slices)
        self.bit_label.setVisible(has_slices)
        self.view.setHtml(
            f"<h3>{E(info.heading)}</h3><p>{where}</p>"
            f"<p><b>{E(tr('What happens in this cycle'))}</b></p><ul>{now}</ul>"
            f"{inside}"
            f"<p><b>{E(tr('How it works'))}</b></p><p>{E(info.part.inside)}</p>")

    # -- the drawings -----------------------------------------------------------------

    def _drawings(self, s: Session, logic) -> str:
        """Fill the drawing tabs for the part (hidden when it has none);
        returns the text that goes with them."""
        c = theme.colors()
        tabs = self.drawings
        current = tabs.currentIndex()
        while tabs.count():
            tabs.removeTab(0)
        html_ = ""
        nano = bool(getattr(s.machine, "nano", False))
        word = None if nano else dpi.control_word(s, self.key)
        insn = dpi.control_nano(s, self.key) if nano else None
        if insn is not None:
            html_ = self._nano_main(insn, c.stage["ID"][1])
        if word is not None:
            mc = cl.main_control(s.release, s.options.arch)
            ev = cl.evaluate_main(mc, word)
            lit = c.stage["ID"][1]
            self.pla_main.show_pla(mc.opcode, word >> 26,
                                   f"Main control: the opcode decoder (op = {word >> 26:06b})",
                                   lit)
            tabs.addTab(self.pla_main, tr("Opcode decoder"))
            if ev.escape is not None:
                esc = ev.escape
                x = cl._bits(word, esc.fields)
                self.pla_second.show_pla(
                    esc.pla, x, f"{esc.escape} decoder ({', '.join(esc.fields)}): enabled by "
                    f"the opcode decoder's {esc.escape} line", lit)
                tabs.addTab(self.pla_second, tr("{name} decoder").format(name=esc.escape))
            sig = ", ".join(f"{k}={int(v)}" for k, v in ev.signals.items())
            html_ = (f"<p><b>{E(tr('The control signals'))}</b> (the opcode decoder's "
                     + (f"OR the {E(ev.escape.escape)} decoder's " if ev.escape else "")
                     + f"outputs): <code>{E(sig)}</code>; ALUOp = {ev.aluop:04b} "
                     f"({E(al.aluop_why(ev.aluop, s.wide))})</p>")
        if logic is not None and logic.control is not None:
            ctl = logic.control
            lit = c.stage["EX"][1]
            view = dpi.alu_view(s, self.key)
            if nano:                            # ALUOp names the function; ALUSrc: ShVar
                src = nc.evaluate_main(nc.main_control(), view.word,
                                       view.size).signals["ALUSrc"]
                self.pla_alu.show_pla(nc.alu_control(), nc.alu_input(ctl.aluop, src),
                                      f"ALU control: ALUOp = {ctl.aluop:04b}, ALUSrc = "
                                      f"{int(src)}", lit)
            else:
                pla = cl.alu_control(s.release, s.options.arch)
                self.pla_alu.show_pla(pla, cl.alu_input(ctl.aluop, view.word),
                                      f"ALU control: ALUOp = {ctl.aluop:04b}, funct = "
                                      f"{view.word & 0x3F:06b}", lit)
            tabs.addTab(self.pla_alu, tr("ALU control"))
            if logic.slices:
                i = self.bit.value()
                self.slice.show_slice(logic.slices[i], ctl, i == logic.top, lit, logic.top,
                                      logic.width)
                tabs.addTab(self.slice, tr("ALU bit slice"))
        self.drawings.setVisible(tabs.count() > 0)
        if 0 <= current < tabs.count():
            tabs.setCurrentIndex(current)
        return html_

    def _nano_main(self, insn, lit: str) -> str:
        """The main control of a nanoMIPS core (emips.nanocontrol): the major
        opcode decoder and the pool's decoder."""
        tabs = self.drawings
        mc = nc.main_control()
        ev = nc.evaluate_main(mc, insn.raw, insn.size)
        maj = nc.major(insn.raw, insn.size)
        self.pla_main.show_pla(mc.opcode, maj, f"Main control: the major opcode decoder "
                               f"(bits 15..10 of the first halfword = {maj:06b})", lit)
        tabs.addTab(self.pla_main, tr("Opcode decoder"))
        if ev.escape is not None:
            esc = ev.escape
            self.pla_second.show_pla(
                esc.pla, ev.x, f"{esc.escape} decoder ({', '.join(esc.fields)}): enabled "
                f"by the opcode decoder's {esc.escape} line", lit)
            tabs.addTab(self.pla_second, tr("{name} decoder").format(name=esc.escape))
        sig = ", ".join(f"{k}={int(v)}" for k, v in ev.signals.items())
        dst = (f"RegDst = {ev.regdst:03b} ({nc.REGDST[ev.regdst]})"
               if ev.signals["RegWrite"] else "RegDst: no register written")
        return (f"<p><b>{E(tr('The control signals'))}</b> (the opcode decoder's "
                + (f"OR the {E(ev.escape.escape)} decoder's " if ev.escape else "")
                + f"outputs): <code>{E(sig)}</code>; {E(dst)}; ALUOp = {ev.aluop:04b} "
                f"({E(nc.ALUOP_WHY.get(ev.aluop, ''))})</p>")

    # -- inside the ALU ---------------------------------------------------------------

    def _alu_html(self, lg: al.AluLogic, mono: str) -> str:
        c = theme.colors()
        out = [f"<p><b>{E(tr('Inside the ALU'))}</b> — {E(lg.instruction)}</p>"]
        ctl = lg.control
        if ctl is not None and not ctl.other:
            funct = f" + funct {ctl.funct}" if ctl.funct else ""
            out.append(f"<p>ALU control: ALUOp = {ctl.aluop:04b} ({E(ctl.aluop_why)}){funct} "
                       f"&rarr; Ainvert {ctl.ainvert}, Bnegate {ctl.bnegate}, Operation "
                       f"{ctl.operation:03b} = {E(al.OPERATION.get(ctl.operation, '?'))}"
                       + (f", Unsigned {ctl.unsigned}" if ctl.operation == 3 else "")
                       + (f", Word {ctl.word}" if lg.width == 64 else "")
                       + (f", Sh32 {ctl.sh32}" if lg.width == 64 and ctl.operation == 5
                          else "")
                       + " (the ALU control PLA above)</p>")
        rows = al.rows(lg)
        if rows:
            cells = []
            for label, value, chosen in rows:
                style = (f" style='background:{c.hit_fill};color:{c.highlight_text}'"
                         if chosen else "")
                bold = ("<b>", "</b>") if label == "result" or chosen else ("", "")
                n = lg.width
                cells.append(f"<tr{style}><td>{bold[0]}{E(label)}{bold[1]}</td>"
                             f"<td style='font-family:{mono}'>{bold[0]}{al.binary(value, n)}"
                             f"{bold[1]}</td><td style='font-family:{mono}'>"
                             f"0x{value & ((1 << n) - 1):0{n // 4}x}</td></tr>")
            t = lg.top
            out.append("<table cellspacing=0 cellpadding=2>"
                       f"<tr><td></td><td><i>bit {lg.width - 1} &hellip; bit 0</i></td>"
                       "<td></td></tr>" + "".join(cells) + "</table>")
            out.append(f"<p>{E(tr('The highlighted row is what the Operation mux passes on.'))}"
                       f" Zero = NOR of the {lg.width} result bits = {lg.zero}; Overflow = "
                       f"carry into bit {t} XOR carry out = {lg.overflow}; Set = {lg.set_} "
                       f"(bit {t}'s sum corrected by overflow, wired to bit 0's Less)</p>")
            out.append(self._slice_html(lg.slices[self.bit.value()], ctl, mono, lg.top,
                                        lg.width))
        if lg.shifter:
            n = lg.width
            cells = "".join(f"<tr><td>{E(label)}</td><td style='font-family:{mono}'>"
                            f"{al.binary(v, n)}</td></tr>" for label, v in lg.shifter)
            stages = (f"5 stages of {n} 2:1 muxes" if n == 32 else
                      "6 stages of 64 2:1 muxes; a Word shift: the low word's 5, then bit 31 "
                      "sign-extended")
            out.append(f"<p>Barrel shifter ({stages}), input "
                       f"<code>{al.binary(lg.a if lg.instruction.split()[0] != 'lui' else lg.b, n)}"
                       f"</code>:</p><table cellpadding=2>{cells}</table>")
        out += [f"<p>{E(n)}</p>" for n in lg.notes]
        if lg.result is not None and not lg.matches:
            out.append(f"<p style='color:{c.red}'>The gates give 0x{lg.result:0{lg.width // 4}x}, "
                       f"the trace recorded 0x{(lg.traced or 0):0{lg.width // 4}x}.</p>")
        return "".join(out)

    @staticmethod
    def _slice_html(sl: al.Slice, ctl, mono: str, top: int = 31, width: int = 32) -> str:
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
        return (f"<p><b>Bit {sl.i}, one ALU slice:</b></p>"
                f"<p style='font-family:{mono}'>{body}</p>")
