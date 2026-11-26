"""The 5-stage datapath's drawing, Qt-free (PLAN M8; M29 W2): the layout of
the classic pipeline (Patterson & Hennessy) — parts, wires, port names,
value places — and :func:`draw`, what one refresh shows: which parts and
wires light up and in what colour, the values on the wires, the stage
titles.  The window's ``gui/datapath.py`` (Qt graphics items) and the web
page (SVG) both draw from it, so they show the same thing.

Colours are tokens, resolved by each view in its theme: ``None`` (idle),
``"stage:IF"`` ... (a stage's line colour), ``"band:IF"`` (its fill),
``"bubble"``, ``"red"``, ``"red_fill"``, ``"muted"``.

Layout: one column per stage, separated by the pipeline registers, and
horizontal *lanes* for the signals that travel through them — control,
PC + 4 / branch target, operand A, operand B, immediate, destination
register, forwarding / hazards, and the two write-back lanes that return
to the register file.  Each cycle (pipeline mode) the components and wires
an instruction uses light up in its stage's colour; forwarding paths and
stalls are red; values are written on the wires.  In instruction mode the
one executed instruction is shown in all stages, single-cycle style.

Everything comes from :meth:`Session.stage_details` / ``last_view`` (the
per-cycle trace) and the decoded word — nothing is re-executed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .datapathinfo import PARTS as INFO
from .datapathinfo import control_of, coprocessor_of, coprocessor_reads
from .datapathinfo import dest_of, details as details_of, nano_of, rv_of
from .datapathinfo import taken as taken_of, tlb_lines as tlb_lines_of, word_of
from .syntax import gpr_name

W, H = 1500, 790
STAGES = ("IF", "ID", "EX", "MEM", "WB")
STAGE_X = {"IF": (0, 270), "ID": (282, 600), "EX": (612, 920),
           "MEM": (932, 1170), "WB": (1182, 1500)}
BARS = {"IF/ID": 270, "ID/EX": 600, "EX/MEM": 920, "MEM/WB": 1170}

# lanes (y)
Y_BR, Y_ADD, Y_CTRL = 112, 165, 92
Y_A, Y_B, Y_IMM, Y_DST = 300, 370, 450, 525
Y_FWD, Y_WB, Y_WREG = 610, 700, 725

#: key -> (kind, x, y, w, h, label, tip); kinds rect / ellipse / alu / mux
PARTS: dict[str, tuple] = {
    # IF
    "pcmux": ("mux", 16, 270, 22, 70, "M", "PC source: PC + 4, or the branch / jump "
              "target decided in ID"),
    "pc": ("rect", 52, 283, 42, 44, "PC", "Program counter: address of the instruction "
           "being fetched"),
    "add4": ("ellipse", 112, 145, 44, 40, "+4", "Adds 4: address of the next instruction "
             "(PC + 4)"),
    "imem": ("rect", 120, 255, 124, 110, "Instruction\nmemory",
             "Instruction memory: reads the 32-bit word at PC"),
    # ID
    "control": ("ellipse", 322, Y_CTRL - 24, 116, 48, "Control",
                "Main control unit: decodes the opcode / function into the control "
                "signals carried along the pipeline"),
    "regs": ("rect", 344, 270, 126, 155, "Registers",
             "Register file: two read ports (rs, rt) in ID, one write port used by WB "
             "(written in the first half of the cycle, read in the second)"),
    "sext": ("ellipse", 356, Y_IMM - 20, 96, 40, "Sign-\nextend",
             "Sign- (or zero-) extends the 16-bit immediate to 32 bits"),
    "btarget": ("ellipse", 512, Y_ADD - 20, 76, 40, "Branch\ntarget",
                "PC + 4 + (offset << 2): the branch target, computed in ID"),
    "cmp": ("ellipse", 506, 316, 44, 40, "=", "Branch comparator in ID: decides the "
            "branch (operands may be forwarded from EX/MEM)"),
    "hazard": ("rect", 300, Y_FWD - 23, 150, 46, "Hazard\ndetection",
               "Hazard detection unit: stalls ID (bubble into EX) when an operand is not "
               "ready yet (load-use, branch operands, multiply/divide busy)"),
    # EX
    "fwda": ("mux", 640, 272, 24, 62, "A", "Forwarding mux A: register value, or the "
             "result forwarded from EX/MEM or MEM/WB"),
    "fwdb": ("mux", 640, 352, 24, 62, "B", "Forwarding mux B: register value, or the "
             "result forwarded from EX/MEM or MEM/WB"),
    "alusrc": ("mux", 708, 372, 24, 62, "Src", "ALUSrc mux: operand b is the register "
               "value or the sign-extended immediate"),
    "alu": ("alu", 768, 262, 84, 176, "ALU", "Arithmetic-logic unit"),
    "vunit": ("rect", 784, 542, 124, 30, "FPU / MSA unit",
              "The coprocessor beside the ALU: the FPU's and MSA's registers ($fN = the "
              "low half of $wN) and execution units; it runs FPU and MSA instructions "
              "element by element"),
    "regdst": ("mux", 740, 500, 24, 54, "Dst", "RegDst mux: destination register is rt "
               "(I-type) or rd (R-type), or $ra for links, or bits 10..6 (MSA's COPY_S / "
               "COPY_U / CFCMSA)"),
    "fwdu": ("rect", 680, Y_FWD - 23, 160, 46, "Forwarding\nunit",
             "Forwarding unit: compares the destination registers in EX/MEM and MEM/WB "
             "with the operands in EX and drives muxes A and B"),
    # MEM
    "dmem": ("rect", 984, 270, 136, 155, "Data\nmemory",
             "Data memory: loads read and stores write at the ALU-computed address"),
    # the level below the L1 caches (shown with caches; one unified L2,
    # drawn under each L1 as in Patterson & Hennessy's cache figures)
    "below_if": ("rect", 120, 600, 124, 40, "L2 cache", ""),
    "below_mem": ("rect", 984, 600, 136, 40, "L2 cache", ""),
    # WB
    "wbmux": ("mux", 1240, 312, 26, 156, "WB", "MemtoReg mux: write back the loaded "
              "data or the ALU result"),
}

#: key -> (points, tip)
WIRES: dict[str, tuple] = {
    # IF
    "pcmux_pc": ([(38, 305), (52, 305)], ""),
    "pc_imem": ([(94, 305), (120, 305)], ""),
    "pc_add": ([(102, 305), (102, Y_ADD), (112, Y_ADD)], ""),
    "add_ifid": ([(156, Y_ADD), (270, Y_ADD)], "PC + 4"),
    "add_pcmux": ([(164, Y_ADD), (164, 132), (8, 132), (8, 286), (16, 286)],
                  "PC + 4 back to the PC mux"),
    "imem_ifid": ([(244, 310), (270, 310)], "the instruction word"),
    "refill_if": ([(232, 600), (232, 365)], "a missing line comes from the level below"),
    # ID
    "ifid_ctrl": ([(282, Y_CTRL), (322, Y_CTRL)], "opcode / funct"),
    "ctrl_idex": ([(438, Y_CTRL), (600, Y_CTRL)], "control signals"),
    "ifid_rs": ([(282, 288), (344, 288)], "rs"),
    "ifid_rt": ([(282, 318), (344, 318)], "rt"),
    "rd1_idex": ([(470, Y_A), (600, Y_A)], "read data 1 (rs)"),
    "rd2_idex": ([(470, Y_B), (600, Y_B)], "read data 2 (rt)"),
    "ifid_sext": ([(282, Y_IMM), (356, Y_IMM)], "immediate (16 bits)"),
    "sext_idex": ([(452, Y_IMM), (600, Y_IMM)], "immediate (32 bits)"),
    "sext_bt": ([(476, Y_IMM), (476, 214), (550, 214), (550, Y_ADD + 20)],
                "offset for the branch target"),
    "ifid_dst": ([(282, Y_DST), (600, Y_DST)], "rt and rd register numbers"),
    "ifid_bt": ([(282, Y_ADD), (512, Y_ADD)], "PC + 4"),
    "rd1_cmp": ([(486, Y_A), (486, 326), (506, 326)], ""),
    "rd2_cmp": ([(494, Y_B), (494, 346), (506, 346)], ""),
    "bt_pcmux": ([(588, Y_ADD), (594, Y_ADD), (594, Y_BR), (4, Y_BR), (4, 324), (16, 324)],
                 "branch / jump target to the PC mux"),
    "ifid_hazard": ([(282, Y_FWD), (300, Y_FWD)], ""),
    # EX
    "idex_fwda": ([(612, Y_A), (640, Y_A)], ""),
    "fwda_alu": ([(664, Y_A), (768, Y_A)], "ALU operand a"),
    "idex_fwdb": ([(612, Y_B), (640, Y_B)], ""),
    "fwdb_src": ([(664, 386), (708, 386)], ""),
    "imm_src": ([(612, Y_IMM), (690, Y_IMM), (690, 418), (708, 418)], "immediate"),
    "src_alu": ([(732, 403), (768, 403)], "ALU operand b"),
    "store_data": ([(682, 386), (682, 470), (920, 470)], "store data (rt)"),
    "alu_exmem": ([(852, 350), (920, 350)], "ALU result"),
    "idex_dst": ([(612, Y_DST), (740, Y_DST)], ""),
    "dst_exmem": ([(764, Y_DST + 2), (920, Y_DST + 2)], "destination register"),
    "fwd_ex_a": ([(932, 350), (948, 350), (948, 572), (630, 572), (630, 290), (640, 290)],
                 "forwarded from EX/MEM"),
    "fwd_ex_b": ([(932, 350), (948, 350), (948, 572), (630, 572), (630, 372), (640, 372)],
                 "forwarded from EX/MEM"),
    "fwd_wb_a": ([(1290, 390), (1290, 586), (622, 586), (622, 312), (640, 312)],
                 "forwarded from MEM/WB"),
    "fwd_wb_b": ([(1290, 390), (1290, 586), (622, 586), (622, 396), (640, 396)],
                 "forwarded from MEM/WB"),
    "fwd_ex_id": ([(948, 350), (948, 228), (500, 228), (500, 336), (506, 336)],
                  "forwarded from EX/MEM to the branch comparator in ID"),
    # MEM
    "addr": ([(932, 350), (960, 350), (960, 300), (984, 300)], "address"),
    "wdata": ([(932, 470), (968, 470), (968, 390), (984, 390)], "write data"),
    "rdata": ([(1120, 330), (1170, 330)], "read data"),
    "bypass": ([(960, 350), (960, 450), (1170, 450)], "ALU result (not a load)"),
    "dst_memwb": ([(932, Y_DST + 2), (1170, Y_DST + 2)], ""),
    "refill_mem": ([(1108, 600), (1108, 425)],
                   "a missing line comes from the level below (a dirty victim goes down)"),
    # WB
    "memwb_mem": ([(1182, 330), (1240, 330)], ""),
    "memwb_alu": ([(1182, 450), (1240, 450)], ""),
    "wb_data": ([(1266, 390), (1320, 390), (1320, Y_WB), (318, Y_WB), (318, 405), (344, 405)],
                "write-back data to the register file"),
    "wb_reg": ([(1182, Y_DST + 2), (1340, Y_DST + 2), (1340, Y_WREG), (308, Y_WREG),
                (308, 382), (344, 382)], "write-back register number"),
    # RISC-V: rd passes straight through (no RegDst mux)
    "rd_pass": ([(740, Y_DST), (764, Y_DST + 2)], "rd: bits 11..7 in every format"),
}

#: port labels on the register file and data memory: name -> (x, y)
PORTS = {"rs": (347, 280), "rt": (347, 310), "write reg": (347, 374),
         "write data": (347, 396), "rd1": (436, 292), "rd2": (436, 362),
         "addr": (987, 292), "wdata": (987, 382), "rdata": (1084, 322)}

#: value labels: key -> (x, y, font size)
VALUES = {"pc": (48, 332, 9), "word": (124, 370, 9), "ctrl": (290, 118, 8),
          "regs": (290, 252, 8), "imm": (290, 474, 8), "branch": (290, 190, 8),
          "aluop": (772, 240, 9), "a": (616, 248, 8), "b": (616, 438, 8),
          "res": (856, 316, 8), "dst": (768, 532, 8), "mem": (940, 434, 8),
          "wb": (1190, 556, 9), "status": (10, 764, 10), "events": (470, 646, 8),
          "icache": (20, 392, 8), "dcache": (940, 540, 8),
          "below_if": (20, 646, 8), "below_mem": (990, 646, 8), "vunit": (786, 573, 7)}

#: what the parts are called on each architecture (key -> (label, tip));
#: RISC-V: P&H's RISC-V edition, Figure 4.65
LABELS = {
    "mips": {"add4": ("+4", "Adds 4: address of the next instruction (PC + 4)"),
             "sext": ("Sign-\nextend", "Sign- (or zero-) extends the 16-bit immediate "
                                       "to 32 bits"),
             "vunit": ("FPU / MSA unit", "The coprocessor beside the ALU: the FPU's and "
                       "MSA's registers ($fN = the low half of $wN) and execution units"),
             "btarget": ("Branch\ntarget", "PC + 4 + (offset << 2): the branch target, "
                         "computed in ID"),
             "control": ("Control", "Main control unit: decodes the opcode / function "
                         "into the control signals carried along the pipeline")},
    "nano": {"add4": ("+2/4/6", "Adds the instruction's size (2, 4 or 6 bytes): the "
                      "address of the next instruction")},
    "rv": {"add4": ("+4/+2", "Adds the instruction's size: 4, or 2 for a C (compressed) "
                    "instruction"),
           "sext": ("Imm\nGen", "Immediate generator: assembles the I / S / B / U / "
                    "J-type immediate from its bit fields, sign-extended from bit 31"),
           "vunit": ("FPU (F / D)", "The floating-point unit beside the ALU: f0-f31, "
                     "fcsr and the F / D execution units"),
           "btarget": ("Branch\ntarget", "PC + offset (in bytes, always even: the "
                       "textbook's shift left 1): the branch / jal target, computed in "
                       "ID; jalr: rs1 + offset"),
           "control": ("Control", "Main control unit: decodes the opcode (bits 6..0) "
                       "into the control signals; funct3 / funct7 go to the ALU "
                       "control")},
}
#: register-file port names (MIPS: rs / rt; RISC-V: rs1 / rs2)
PORT_NAMES = {"mips": ("rs", "rt"), "nano": ("rs", "rt"), "rv": ("rs1", "rs2")}

CLICK = "\n(click: what happens inside)"


def layout() -> dict:
    """The static drawing (the page builds its SVG from it once): sizes,
    stages, pipeline registers, parts, wires, ports, value places, the
    guide's part numbers."""
    numbers, seen = {}, set()
    for key in PARTS:
        info = INFO.get(key)
        if info is not None and info.number is not None and info.number not in seen:
            seen.add(info.number)
            numbers[key] = [info.number, info.title]
    return {"W": W, "H": H, "stages": {k: list(v) for k, v in STAGE_X.items()},
            "bars": BARS,
            "parts": {k: [kind, x, y, w, h, label, tip + CLICK]
                      for k, (kind, x, y, w, h, label, tip) in PARTS.items()},
            "wires": {k: [[list(p) for p in pts], tip] for k, (pts, tip) in WIRES.items()},
            "ports": {k: list(v) for k, v in PORTS.items()},
            "values": {k: list(v) for k, v in VALUES.items()},
            "numbers": numbers}


@dataclass
class Drawing:
    """One refresh: what each part, wire, value, stage shows (colour tokens)."""
    #: key -> [line colour, fill colour, label, tip, shown]
    parts: dict = field(default_factory=dict)
    #: key -> [colour, shown]
    wires: dict = field(default_factory=dict)
    #: key -> [text, colour]
    values: dict = field(default_factory=dict)
    titles: dict = field(default_factory=dict)  # stage -> its instruction ("" / "(bubble)")
    bands: dict = field(default_factory=dict)   # stage -> "band:ST" or "bubble"
    ports: dict = field(default_factory=dict)   # "rs" / "rt" -> the shown name
    kind: str = "mips"                          # mips / nano / rv

    def to_json(self) -> dict:
        return {"parts": self.parts, "wires": self.wires, "values": self.values,
                "titles": self.titles, "bands": self.bands, "ports": self.ports,
                "kind": self.kind}


def draw(s) -> Drawing:
    """What the datapath shows for the session's current state."""
    return _Draw(s).d


class _Draw:
    def __init__(self, s) -> None:
        self.s = s
        d = self.d = Drawing()
        for key, (kind, x, y, w, h, label, tip) in PARTS.items():
            d.parts[key] = [None, None, label, tip + CLICK, True]
        for key in WIRES:
            d.wires[key] = [None, True]
        for key in VALUES:
            d.values[key] = ["", None]
        for st in STAGES:
            d.titles[st] = ""
            d.bands[st] = "bubble"
        m = s.machine
        self._relabel("rv" if m is not None and m.rv else
                      "nano" if getattr(m, "nano", False) else "mips")
        self._cache_parts()
        if m is None:
            self.value("status", "no program loaded — open a file or pick File > Examples")
            return
        details, ev = details_of(s)
        for st in STAGES:
            dd = details.get(st) or {}
            if dd.get("pc") is None:
                d.titles[st] = "(bubble)" if s.pipeline is not None else ""
                continue
            d.bands[st] = f"band:{st}"
            d.titles[st] = _clip(dd["text"], 28)
            if dd.get("control") is None:
                dd = dict(dd, control=control_of(s, dd))
            if st in ("IF", "ID") or not _is_nop(s, dd):    # a nop does nothing past ID
                getattr(self, f"_{st.lower()}")(dd, ev)
            if st in ("IF", "MEM") and (dd.get("cache") or dd.get("tlb")):
                self._cache(st, dd.get("cache") or [], dd.get("wait", 0), dd.get("tlb") or [])
        events = ev.get("events") or []
        for e in events:                      # forwarding not seen in an operand
            if "EX/MEM -> EX" in e or "MEM/WB -> EX" in e:
                self.light(["fwdu"], "red")
        status = ev.get("label") or ""
        if ev.get("cycle") is not None:
            status = f"cycle {ev['cycle']}"
        if ev.get("stall"):
            status += f"   STALL — {ev['stall']}"
            if not ev["stall"].startswith(("memory", "I-miss")):   # not the hazard unit
                self.light(["hazard"], "red", "red_fill")
                self.light(["ifid_hazard"], "red")
        self.value("status", status, "red" if ev.get("stall") else None)
        self.value("events", "\n".join(_clip(e, 84) for e in events[:3]), "muted")

    # -- helpers ----------------------------------------------------------------------------

    def light(self, keys, color, fill=None) -> None:
        for k in keys:
            if k in self.d.parts:
                self.d.parts[k][0] = color
                self.d.parts[k][1] = fill
            else:
                self.d.wires[k][0] = color

    def off(self, keys) -> None:
        self.light(keys, None)

    def value(self, key: str, text: str, color: str | None = None) -> None:
        self.d.values[key] = [text, color]

    def rename(self, key: str, label: str, tip: str | None = None) -> None:
        p = self.d.parts[key]
        p[2] = label
        if tip is not None:
            p[3] = tip

    def show(self, key: str, on: bool) -> None:
        (self.d.parts.get(key) or self.d.wires[key])[-1] = on

    def _relabel(self, kind: str) -> None:
        """Name the parts for the architecture."""
        self.d.kind = kind
        for key in LABELS["mips"]:
            label, tip = LABELS[kind].get(key) or LABELS["mips"][key]
            self.rename(key, label, tip + CLICK)
        rs, rt = PORT_NAMES[kind]
        self.d.ports = {"rs": rs, "rt": rt}
        rv = kind == "rv"
        self.show("regdst", not rv)             # RISC-V: rd needs no mux
        self.show("rd_pass", rv)

    def _cache_parts(self) -> None:
        """Name the memories after the caches the machine has; the boxes
        of the level below show only with caches."""
        s = self.s
        caches = s.machine.caches if s.machine is not None else None
        below = None
        if caches is not None and caches.all:
            below = ("L2 cache" if caches.l2 is not None and not caches.l2_bypass
                     else "L3 cache" if caches.l3 is not None else "Memory")
        for st, key, l1, what in (("if", "imem", "i", "Instruction"),
                                  ("mem", "dmem", "d", "Data")):
            c = getattr(caches, l1) if caches is not None else None
            if c is not None:
                self.rename(key, f"{c.name}\n{c.geometry.spec()}",
                            f"{c.name} ({c.geometry.describe()}): a hit delivers in the "
                            "cycle; a miss fills the line from the level below first. See "
                            "the Caches pane for its sets and ways." + CLICK)
            else:
                self.rename(key, f"{what}\nmemory",
                            ("Instruction memory: reads the 32-bit word at PC" if l1 == "i"
                             else "Data memory: loads read and stores write at the "
                             "ALU-computed address") + CLICK)
            self.show(f"below_{st}", below is not None)
            self.show(f"refill_{st}", below is not None)
            if below is not None:
                self.rename(f"below_{st}", below,
                            f"{below}: where misses are served from (one unified level, "
                            "drawn under each L1 cache); uncached accesses go straight to "
                            "memory" + CLICK)

    def _cache(self, st: str, accesses: list, wait: int, translations: list = ()) -> None:
        """What the micro-TLB and the caches did for the access in IF / MEM."""
        key, val = ("imem", "icache") if st == "IF" else ("dmem", "dcache")
        low = "below_if" if st == "IF" else "below_mem"
        lines, below = tlb_lines_of(translations)[:3], []
        for a in accesses:
            if a.level in ("I", "D"):
                if a.hit:
                    lines.append(f"{a.level}$ hit: set {a.set} way {a.way}")
                else:
                    lines.append(f"{a.level}$ MISS ({a.miss or 'no allocate'})")
                    lines.append(f"  set {a.set}" + (f" -> way {a.way}" if a.way is not None
                                                     else ""))
                    self.light([key], "red", "red_fill")
            elif a.level == "U":
                lines.append("no cache: memory" if a.miss == "no cache"
                             else "uncached: memory")
            else:
                name = "L2" if a.level == "S" else "L3"
                below.append(f"{name} {a.kind}: " + ("hit" if a.hit else "miss"))
        if wait:
            lines.append(f"waiting: {wait} more cycle(s)")
        bad = any("MISS" in t or "TLB " in t and t.startswith("  ") for t in lines)
        self.value(val, "\n".join(lines), "red" if bad else None)
        if below:
            self.light([f"refill_{st.lower()}", low], "red" if wait else f"stage:{st}")
            self.value(low, "\n".join(below[:3]))

    # -- per stage --------------------------------------------------------------------------

    def _if(self, d, ev) -> None:
        s, c = self.s, "stage:IF"
        self.light(["pcmux", "pc", "imem", "add4", "pcmux_pc", "pc_imem", "pc_add",
                    "add_ifid", "imem_ifid"], c)
        if not taken_of(d, ev):
            self.light(["add_pcmux"], c)
        self.value("pc", f"PC {d['pc']:0{s.hex_digits('pc')}x}")
        word = word_of(s, d)
        i = nano_of(s, d)
        r = rv_of(s, d)
        if r is not None:                       # RISC-V: a word, or a C halfword
            self.value("word", f"{r.raw:08x}" if r.size == 4 else
                       f"{r.raw:04x} (C: {r.base.name})")
        elif i is not None:                     # nanoMIPS: its halfwords
            self.value("word", " ".join(f"{(i.raw >> (16 * k)) & 0xFFFF:04x}"
                                        for k in range(i.size // 2 - 1, -1, -1)))
        elif d.get("compressed"):               # MIPS16e / microMIPS: the encoding, recoded
            raw = d["compressed"].split()[-1][2:]
            self.value("word", f"{raw} -> {word:08x}" if word is not None else raw)
        elif word is not None:
            self.value("word", f"{word:08x}")

    def _id(self, d, ev) -> None:
        s, c = self.s, "stage:ID"
        ctl = d.get("control") or {}
        self.light(["control", "regs", "ifid_ctrl", "ctrl_idex", "ifid_rs", "ifid_rt",
                    "rd1_idex", "rd2_idex", "ifid_dst"], c)
        if ctl:
            items = [f"{k}={_short(v)}" for k, v in ctl.items() if k != "ALUOp"]
            self.value("ctrl", "\n".join("  ".join(items[i:i + 3])
                                         for i in range(0, len(items), 3)))
        word = word_of(s, d)
        i = nano_of(s, d)
        r = rv_of(s, d)
        gpr_reads = ("regs", "ifid_rs", "ifid_rt", "rd1_idex", "rd2_idex")
        if r is not None:                       # RISC-V: rs1 / rs2 and the Imm Gen
            from .riscv import FPR_NAMES, GPR_NAMES as RV_GPRS
            from .tracerv import f_registers
            fp = f_registers(r)
            reads = [(FPR_NAMES if k in fp else RV_GPRS)[r.f[k]]
                     for k in ("rs1", "rs2", "rs3") if k in r.f]
            self.value("regs", "read " + ("  ".join(reads) or "-"))
            if fp and all(k in fp for k in ("rs1", "rs2", "rs3") if k in r.f):
                self.off(gpr_reads)             # (no x register read)
            if "imm" in r.f and (ctl.get("ALUSrc") == "imm" or ctl.get("Branch")
                                 or ctl.get("Jump")):
                self.light(["sext", "ifid_sext", "sext_idex"], c)
                self.value("imm", f"imm {r.f['imm']}")
        elif i is not None:                     # nanoMIPS: what the format reads
            from .simnano import uses
            reads = [gpr_name(n, s.machine.names) for n in uses(i)[0]]
            self.value("regs", "read " + ("  ".join(reads[:3]) + (" …" if len(reads) > 3
                                                                 else "")
                                          if reads else "-"))
            if ctl.get("ALUSrc") == "imm" or ctl.get("Branch"):
                imm = _nano_imm(i)
                if imm is not None:
                    self.light(["sext", "ifid_sext", "sext_idex"], c)
                    self.value("imm", f"imm {imm}")
        elif coprocessor_reads(s, word) is not None:    # FPU / MSA: its own registers
            unit, reads = coprocessor_reads(s, word)
            self.value("regs", f"{'MSA' if unit == 'msa' else 'FPU'} reads "
                       + (" ".join(reads[:4]) or "-"))
            if not any(x.startswith("$") and x[1:2] not in ("w", "f") for x in reads):
                self.off(gpr_reads)             # (no GPR read)
            if ctl.get("ALUSrc") == "imm" or ctl.get("Branch"):
                self.light(["sext", "ifid_sext", "sext_idex"], c)
        elif word is not None:
            rs, rt, imm = (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
            abi = "n64" if s.wide else True     # (MIPS64: the n64 names)
            self.value("regs", f"rs={gpr_name(rs, abi)}  rt={gpr_name(rt, abi)}")
            if ctl.get("ALUSrc") == "imm" or ctl.get("Branch"):
                self.light(["sext", "ifid_sext", "sext_idex"], c)
                simm = imm - 0x10000 if imm & 0x8000 else imm
                self.value("imm", f"imm {simm}")
        if ctl.get("Branch") or ctl.get("Jump"):
            self.light(["btarget", "ifid_bt", "cmp", "rd1_cmp", "rd2_cmp", "sext_bt"], c)
            if taken_of(d, ev):
                self.light(["bt_pcmux"], c)
            v = d.get("view")
            if v is not None and v.branch is not None:
                br = v.branch
                self.value("branch", _clip(br.condition, 26)
                           + ("\ntaken -> " + f"{br.target:0{s.hex_digits()}x}"
                              if br.taken else "\nnot taken"))
        if any("-> ID" in e for e in ev.get("events") or []):
            self.light(["fwd_ex_id"], "red")

    def _ex(self, d, ev) -> None:
        s, c = self.s, "stage:EX"
        v = d.get("view")
        unit = coprocessor_of(s, v)
        if unit is not None:                    # the FPU / MSA unit does the work
            self.light(["vunit"], c)
            lanes = getattr(v, "lanes", None)
            self.value("vunit", _clip(f"{lanes.size}-bit x {128 // lanes.size}"
                                      if lanes is not None and hasattr(lanes, "size")
                                      else "FPU", 20))
            if v.alu is None and not (v.mem_reads or v.mem_writes):
                if v.exception:
                    self.value("res", f"EXCEPTION\n{v.exception}", "red")
                dest = _dest(s, d)
                if dest is not None:
                    self.light(["rd_pass" if s.machine.rv else "regdst", "dst_exmem"], c)
                    self.value("dst", f"dest {dest[0]}")
                return
        self.light(["fwda", "fwdb", "alu", "idex_fwda", "idex_fwdb", "fwda_alu",
                    "fwdb_src", "alusrc", "src_alu", "alu_exmem", "idex_dst"], c)
        if v is None:
            return
        if v.alu is not None:
            a, b = v.alu.a, v.alu.b
            self.value("aluop", _clip(v.alu.op, 30))
            if a is not None:
                self.value("a", f"a {a.value:0{s.hex_digits()}x} ({_src(a.source)})")
                self._forward("a", a.source)
            if b is not None:
                self.value("b", f"b {b.value:0{s.hex_digits()}x} ({_src(b.source)})")
                if "immediate" in b.source or "offset" in b.source:
                    self.light(["imm_src"], c)
                self._forward("b", b.source)
            if v.alu.result is not None:
                self.value("res", f"= {v.alu.result:0{s.hex_digits()}x}"
                           + ("\nOVERFLOW" if v.alu.overflow else ""))
        if v.mem_writes:
            self.light(["store_data"], c)
        dest = _dest(s, d)
        if dest is not None:
            self.light(["rd_pass" if s.machine.rv else "regdst", "dst_exmem"], c)
            self.value("dst", f"dest {dest[0]}")
        if v.exception:
            self.value("res", f"EXCEPTION\n{v.exception}", "red")

    def _forward(self, operand: str, source: str) -> None:
        if "EX/MEM" in source:
            self.light([f"fwd_ex_{operand}", "fwdu"], "red")
        elif "MEM/WB" in source:
            self.light([f"fwd_wb_{operand}", "fwdu"], "red")

    def _mem(self, d, ev) -> None:
        s, c = self.s, "stage:MEM"
        v = d.get("view")
        if _dest(s, d) is not None:
            self.light(["dst_memwb"], c)
        if v is None or not (v.mem_reads or v.mem_writes):
            if _dest(s, d) is not None:
                self.light(["bypass"], c)
            return
        self.light(["dmem", "addr"], c)
        lines = []
        for a, n, val in v.mem_reads[:2]:
            lines.append(f"read  {a:08x}\n   = {val:0{2 * n}x}")
            self.light(["rdata"], c)
        for a, n, o, nw in v.mem_writes[:2]:
            lines.append(f"write {a:08x}\n  <- {nw:0{2 * n}x}")
            self.light(["wdata"], c)
        if v.address is not None and v.address.physical is not None:
            lines.append(f"PA    {v.address.physical:08x}")
        self.value("mem", "\n".join(lines))

    def _wb(self, d, ev) -> None:
        s, c = self.s, "stage:WB"
        dest = _dest(s, d)
        if dest is None:
            return
        v = d.get("view")
        load = bool((d.get("control") or {}).get("MemtoReg")) or (
            v is not None and bool(v.mem_reads) and not v.mem_writes)
        self.light(["wbmux", "wb_data", "wb_reg", "memwb_mem" if load else "memwb_alu"], c)
        name, value = dest
        self.value("wb", f"{name} <- {value:0{s.hex_digits()}x}" if value is not None
                   else f"{name} written")


def _dest(s, d):
    return dest_of(d, s.machine.names)


def _is_nop(s, d) -> bool:
    """A nop (MIPS: the word 0; RISC-V: addi x0, x0, 0 / c.nop) does
    nothing past ID."""
    if s.machine is not None and s.machine.rv:
        r = rv_of(s, d)
        return r is not None and r.raw in (0x00000013, 0x0001)
    return word_of(s, d) == 0


#: the decoded arguments that are a nanoMIPS instruction's immediate, by
#: preference (emips.nanomips's decode steps)
_NANO_IMMEDIATES = ("imm", "offset", "u", "s", "shift")


def _nano_imm(i) -> int | None:
    from .tracenano import _sx
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
