"""The control logic of a RISC-V hart's datapath as gates (PLAN M25 phase
4c; the GUI: click Control or the ALU): the **main control unit** and the
**ALU control**, as PLAs, generated from the hart's instruction table
(:mod:`emips.riscv`) as :mod:`emips.controllogic` does for MIPS32.  Qt-free.

Patterson & Hennessy's RISC-V edition draws both for its subset (4.4,
Figures 4.12 / 4.22 and Appendix C): the main control decodes the
**opcode** (bits 6..0) into ALUSrc, MemtoReg, RegWrite, MemRead,
MemWrite, Branch and a 2-bit ALUOp (00 add — loads, stores —, 01 subtract
— beq —, 10 the funct fields decide); the ALU control turns ALUOp,
funct7 and funct3 into the ALU's lines.  eMIPS builds the same kind of
logic for every instruction of the hart (RV32 / RV64, M, A, F, D, Zicsr,
Zifencei, Zba, Zbb ...):

* The **opcode PLA** (op6..op0) makes the signals of every major opcode
  whose instructions all share them; an opcode whose instructions differ
  (SYSTEM: ecall vs csrrw; OP-FP: fadd writes an f register, feq an x
  one; OP-IMM: addi vs clz ...) has an *escape* line enabling a
  **secondary PLA** on the fixed bits that tell them apart (named by the
  instruction's bit numbers: ``b25`` = bit 25), built as cubes and
  minimized (:func:`emips.logic.build_cubes`), as eMIPS's nanoMIPS logic.
* A **C instruction** reaches the main control as its 32-bit expansion
  (the decoder expands it first: unprivileged ch. 28) — the same gates.
* The signals: those of :func:`emips.rvpipeline.control_signals` —
  RegWrite (an x register; rd is always bits 11..7, so there is no
  RegDst), ALUSrc, MemtoReg, MemRead, MemWrite, Branch, Jump — and a
  3-bit **ALUOp** (``ALUOP_WHY``; the textbook's 2 bits name too little
  of RISC-V): 000 add (loads, stores, the AMOs' address, lui / auipc —
  the ALU's operand a is then 0 / the pc), 001 branch (the comparator in
  ID decides; the ALU subtracts as the textbook's beq), 010 R-type: funct7
  and funct3 decide, 011 I-type: funct3 (and bits 30 / 29 of a shift)
  decide, 100 / 101 the same for RV64's word operations (OP-32,
  OP-IMM-32), 111 not the ALU (CSRs, the FPU, jumps' links, fences, system).

ALU control: a PLA from ALUOp and the instruction's bits 30, 29, 27, 25
(the funct7 values that tell RISC-V's operations apart: 0000000,
0100000 sub / sra / andn, 0000001 M, 0010000 sh1add, 0000101 min / max,
0110000 rol / ror / clz, 0000100 zext.h) and funct3 (14..12) to the lines
of :mod:`emips.alulogic`'s ALU (``controllogic.ALU_OUTPUTS``, with Word on
RV64): Bnegate also inverts b for and / or / xor — andn, orn and xnor are
the textbook gates with b inverted; multiplies, divides, Zba's shifted
adds, min / max, rol and the bit-count instructions are other units
(OtherUnit).  ``tests/test_rvcontrol.py`` checks both against every row.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

from . import riscv as R
from .controllogic import ALU_OUTPUTS, ALU_OUTPUTS64, alu_lines
from .logic import PLA, Term, build, build_cubes, sharp

SIGNALS = ["ALUSrc", "MemtoReg", "RegWrite", "MemRead", "MemWrite", "Branch", "Jump",
           "ALUOp2", "ALUOp1", "ALUOp0"]
#: ALUOp code -> what the main control asks for
ALUOP_WHY = {0b000: "add: a load / store address (rs1 + offset), an AMO's (rs1), lui "
                    "(0 + the immediate), auipc (the pc + the immediate)",
             0b001: "branch: the comparator in ID decides (the ALU subtracts, as the "
                    "textbook's beq)",
             0b010: "R-type: funct7 and funct3 decide",
             0b011: "I-type: funct3 decides (bits 30 / 29: srai, rori)",
             0b100: "R-type word (RV64 OP-32): funct7 and funct3, a 32-bit result "
                    "sign-extended",
             0b101: "I-type word (RV64 OP-IMM-32): funct3, a 32-bit result sign-extended",
             0b111: "not the ALU (CSRs, the FPU, a jump's link, fences, system)"}
#: the major opcodes (bits 6..0 >> 2; unpriv. Table 70)
OPCODES = {0b00000: "LOAD", 0b00001: "LOAD-FP", 0b00011: "MISC-MEM", 0b00100: "OP-IMM",
           0b00101: "AUIPC", 0b00110: "OP-IMM-32", 0b01000: "STORE", 0b01001: "STORE-FP",
           0b01011: "AMO", 0b01100: "OP", 0b01101: "LUI", 0b01110: "OP-32",
           0b10000: "MADD", 0b10001: "MSUB", 0b10010: "NMSUB", 0b10011: "NMADD",
           0b10100: "OP-FP", 0b11000: "BRANCH", 0b11001: "JALR", 0b11011: "JAL",
           0b11100: "SYSTEM"}
#: the ALU control's inputs (I30 ... = instruction bits)
ALU_INPUTS = ["ALUOp2", "ALUOp1", "ALUOp0", "I30", "I29", "I27", "I25", "F2", "F1", "F0"]
_FUNCT7_BITS = (30, 29, 27, 25)
_LOADS = {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "flw", "fld",
          "sb", "sh", "sw", "sd", "fsw", "fsd"}
#: R / I-type rows -> the ALU function (controllogic.alu_lines' names)
_FUNCTION = {"add": "add", "sub": "sub", "and": "and", "or": "or", "xor": "xor",
             "slt": "slt", "sltu": "sltu", "sll": "sllv", "srl": "srlv", "sra": "srav",
             "addi": "add", "andi": "and", "ori": "or", "xori": "xor", "slti": "slt",
             "sltiu": "sltu", "slli": "sll", "srli": "srl", "srai": "sra",
             "andn": "andn", "orn": "orn", "xnor": "xnor", "ror": "rotrv", "rori": "rotr",
             "addw": "addw", "subw": "subw", "sllw": "sllv.w", "srlw": "srlv.w",
             "sraw": "srav.w", "addiw": "addw", "slliw": "sll.w", "srliw": "srl.w",
             "sraiw": "sra.w", "rorw": "rotrv.w", "roriw": "rotr.w"}
#: andn / orn / xnor: the textbook gates with b inverted (Bnegate)
_INVERTED = {"andn": "and", "orn": "or", "xnor": "xor"}
_PATTERNS = (0x0A5294A5, 0x10842108, 0x07FFFFFF, 0x3C3C3C3C, 0x5A5A5A5A)


def word32(i) -> int:
    """The 32-bit instruction the control decodes: a C instruction's
    expansion (unpriv. ch. 28), else the instruction."""
    return i.raw if i.size == 4 else R.encode(i.base, i.f)


def opcode(word: int) -> int:
    return word & 0x7F


# -- a row's signals -------------------------------------------------------------------

@lru_cache(maxsize=None)
def probe(isa: R.RVISA, row: R.Row) -> R.RV:
    """An instruction of ``row`` (the decoder's choice) with non-zero
    register fields."""
    rng = random.Random(row.name)
    for k in range(600):
        p = _PATTERNS[k] if k < len(_PATTERNS) else rng.getrandbits(32)
        raw = row.match | (p & ~row.mask & 0xFFFFFFFF)
        try:
            i = isa.decode(raw, 4)
        except R.RVDecodeError:
            continue
        if i.row is row:
            return i
    raise ValueError(f"no encoding decodes to {row.name}")     # pragma: no cover


def aluop(name: str, op: int) -> int:
    """The ALUOp code of a row (its name, its opcode bits 6..0)."""
    major = op >> 2
    if name in _LOADS or major == 0b01011 or name in ("lui", "auipc"):
        return 0b000
    if major == 0b11000:
        return 0b001
    return {0b01100: 0b010, 0b00100: 0b011, 0b01110: 0b100, 0b00110: 0b101}.get(major, 0b111)


@lru_cache(maxsize=None)
def row_signals(isa: R.RVISA, row: R.Row) -> dict[str, bool]:
    """The main control's signals for a row — the pipeline's control signals
    of an instance (:func:`emips.rvpipeline.control_signals`), whatever its
    register numbers (rd = x0 is still RegWrite: the format writes rd)."""
    from .rvpipeline import analyze, control_signals
    i = probe(isa, row)
    c = control_signals(i, analyze(i))
    writes_x = bool(c["RegWrite"])
    code = aluop(row.name, row.match & 0x7F)
    return {"ALUSrc": c["ALUSrc"] == "imm", "MemtoReg": bool(c["MemtoReg"]),
            "RegWrite": writes_x, "MemRead": bool(c["MemRead"]),
            "MemWrite": bool(c["MemWrite"]), "Branch": bool(c["Branch"]),
            "Jump": bool(c["Jump"]), "ALUOp2": bool(code & 4), "ALUOp1": bool(code & 2),
            "ALUOp0": bool(code & 1)}


# -- the main control ---------------------------------------------------------------------

@dataclass
class Secondary:
    escape: str                 # the opcode's name (unpriv. Table 70)
    opcode: int
    bits: list[int]             # the instruction bits it reads (MSB first)
    pla: PLA

    @property
    def fields(self) -> list[str]:
        return [f"b{b}" for b in self.bits]

    def x(self, word: int) -> int:
        v = 0
        for b in self.bits:
            v = (v << 1) | ((word >> b) & 1)
        return v


@dataclass
class MainControl:
    opcode: PLA
    secondary: dict[str, Secondary]
    xlen: int
    extensions: frozenset


def _secondary(isa: R.RVISA, name: str, op: int, rows: list[R.Row]) -> Secondary:
    used = 0
    for r in rows:
        used |= r.mask & ~0x7F & 0xFFFFFFFF
    bits = [b for b in range(31, 6, -1) if used >> b & 1]
    n = len(bits)

    def cube(r: R.Row) -> Term:
        value = mask = 0
        for k, b in enumerate(bits):
            pos = 1 << (n - 1 - k)
            if r.mask >> b & 1:
                mask |= pos
                value |= pos if r.match >> b & 1 else 0
        return value, mask

    table: list[tuple[Term, dict[str, bool]]] = []
    before: list[Term] = []
    for r in sorted(rows, key=lambda r: -r.specificity):     # (the decoder's order)
        c = cube(r)
        region = [c]
        for b in before:
            region = [piece for part in region for piece in sharp(part, b)]
        before.append(c)
        sig = row_signals(isa, r)
        table += [(piece, sig) for piece in region]
    pla = build_cubes(f"{name} decoder", [f"b{b}" for b in bits], SIGNALS, table)
    keep = [k for k in range(n) if any(t[1] >> (n - 1 - k) & 1 for t in pla.terms)]
    if len(keep) != n:                          # (inputs no product term reads)
        def squeeze(x: int) -> int:
            v = 0
            for k in keep:
                v = (v << 1) | (x >> (n - 1 - k) & 1)
            return v
        pla = PLA(pla.name, [pla.inputs[k] for k in keep], pla.outputs,
                  [(squeeze(v), squeeze(m)) for v, m in pla.terms], pla.uses)
        bits = [bits[k] for k in keep]
    return Secondary(name, op, bits, pla)


@lru_cache(maxsize=None)
def main_control(xlen: int = 64, extensions: str = "") -> MainControl:
    """The main control unit of a hart (its 32-bit rows: C instructions
    reach it expanded), as PLAs."""
    from .options import RV_EXTENSIONS
    isa = R.RVISA.for_core(xlen, extensions or RV_EXTENSIONS)
    by_op: dict[int, list[R.Row]] = {}
    for r in isa.rows:
        if r.size == 4 and r.mask & 0x7F == 0x7F:
            by_op.setdefault(r.match & 0x7F, []).append(r)
    opcode_sig: dict[int, dict[str, bool]] = {}
    escapes: dict[int, tuple[str, list[R.Row]]] = {}
    for op, rows in sorted(by_op.items()):
        sigs = [row_signals(isa, r) for r in rows]
        if all(s == sigs[0] for s in sigs):
            opcode_sig[op] = sigs[0]
        else:
            escapes[op] = (OPCODES.get(op >> 2, f"opcode {op:07b}"), rows)
    names = [escapes[op][0] for op in sorted(escapes)]
    table = {}
    for op in range(128):
        if op in escapes:
            table[op] = dict({o: False for o in SIGNALS},
                             **{n: n == escapes[op][0] for n in names})
        elif op in opcode_sig:
            table[op] = dict(opcode_sig[op], **{n: False for n in names})
    pla = build("opcode decoder", [f"op{i}" for i in range(6, -1, -1)], SIGNALS + names, table)
    secondary = {name: _secondary(isa, name, op, rows)
                 for op, (name, rows) in sorted(escapes.items())}
    return MainControl(pla, secondary, xlen, isa.extensions)


@dataclass
class MainEval:
    word: int
    opcode_terms: list[bool]
    opcode_out: dict[str, bool]
    escape: Secondary | None
    x: int | None
    secondary_terms: list[bool]
    secondary_out: dict[str, bool]
    signals: dict[str, bool]

    @property
    def aluop(self) -> int:
        return sum(int(self.signals[f"ALUOp{i}"]) << i for i in range(3))


def evaluate_main(mc: MainControl, word: int) -> MainEval:
    hit, out = mc.opcode.evaluate(opcode(word))
    esc = next((s for s in mc.secondary.values() if out.get(s.escape)), None)
    s_hit: list[bool] = []
    s_out = {o: False for o in SIGNALS}
    x = None
    if esc is not None:
        x = esc.x(word)
        s_hit, s_out = esc.pla.evaluate(x)
    final = {o: out[o] or s_out.get(o, False) for o in SIGNALS}
    return MainEval(word, hit, out, esc, x, s_hit, s_out, final)


# -- the ALU control ---------------------------------------------------------------------

def alu_input(code: int, word: int) -> int:
    v = code
    for b in _FUNCT7_BITS:
        v = (v << 1) | ((word >> b) & 1)
    return (v << 3) | ((word >> 12) & 7)


def alu_function(name: str) -> str:
    """A row's ALU function (controllogic.alu_lines' names; "other":
    another unit)."""
    f = _FUNCTION.get(name)
    if f is not None:
        return f
    if name in _LOADS or name.startswith(("amo", "lr.", "sc.")) or name in ("lui", "auipc"):
        return "add"
    return "other"


def lines(function: str, wide: bool) -> dict[str, bool]:
    """The ALU control's outputs for a function (andn / orn / xnor: Bnegate
    with and / or / xor)."""
    base = _INVERTED.get(function)
    out = alu_lines(base or function, wide)
    if base is not None:
        out["Bnegate"] = True
    return out


@lru_cache(maxsize=None)
def alu_control(xlen: int = 64, extensions: str = "") -> PLA:
    """The ALU control of a hart, as a PLA: every row's (ALUOp, bits 30 / 29
    / 27 / 25, funct3) -> its function's lines; the rest don't care."""
    from .options import RV_EXTENSIONS
    isa = R.RVISA.for_core(xlen, extensions or RV_EXTENSIONS)
    wide = xlen == 64
    table: dict[int, dict[str, bool]] = {}
    for r in isa.rows:
        if r.size != 4 or r.mask & 0x7F != 0x7F:
            continue
        code = aluop(r.name, r.match & 0x7F)
        if code in (0b001, 0b111):
            fn = "sub" if code == 0b001 else "other"
        else:
            fn = alu_function(r.name)
        out = lines(fn, wide)
        i = probe(isa, r)
        word = i.raw
        if code == 0b000:                       # (one function: every funct don't care)
            for x in range(1 << 7):
                table.setdefault(code << 7 | x, out)
            continue
        if code == 0b001:
            for x in range(1 << 7):
                table.setdefault(code << 7 | x, out)
            continue
        key = alu_input(code, word)
        relevant = [b for b in _FUNCT7_BITS if r.mask >> b & 1] + [14, 13, 12]
        free = [k for k, b in enumerate(list(_FUNCT7_BITS) + [14, 13, 12])
                if b not in relevant]
        for combo in range(1 << len(free)):     # (bits the row does not fix: any value)
            k = key
            for j, pos in enumerate(free):
                bit = 1 << (6 - pos)
                k = (k & ~bit) | (bit if combo >> j & 1 else 0)
            table.setdefault(k, out)
    for x in range(1 << 7):                     # 111: another unit, whatever funct says
        table[0b111 << 7 | x] = lines("other", wide)
    return build("ALU control", ALU_INPUTS, ALU_OUTPUTS64 if wide else ALU_OUTPUTS, table)


def control_for(word: int, xlen: int = 64, extensions: str = ""):
    """The ALU control of a (32-bit, expanded) instruction word (an
    :class:`emips.alulogic.AluControl`)."""
    from .alulogic import AluControl
    ev = evaluate_main(main_control(xlen, extensions), word)
    _, out = alu_control(xlen, extensions).evaluate(alu_input(ev.aluop, word))
    funct = (f"funct7 {word >> 25:07b} funct3 {(word >> 12) & 7:03b}"
             if ev.aluop in (0b010, 0b011, 0b100, 0b101) else None)
    return AluControl(ev.aluop, funct, out, ALUOP_WHY.get(ev.aluop, ""), aluop_bits=3)


def view_word(view, xlen: int, extensions: str = "") -> int:
    """The 32-bit word the control decodes for a traced instruction (a C
    instruction: its expansion)."""
    from .options import RV_EXTENSIONS
    if view.size == 4:
        return view.word
    isa = R.RVISA.for_core(xlen, (extensions or RV_EXTENSIONS) + "_c")
    return word32(isa.decode(view.word, 2))
