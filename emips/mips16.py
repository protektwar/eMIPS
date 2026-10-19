"""MIPS16e: the 16-bit instruction set (PLAN M17 phase 1).

Reference: *MIPS32 Architecture for Programmers Vol. IV-a: The MIPS16e
Application-Specific Extension to the MIPS32 Architecture* (MD00076, rev.
2.63): 3.6 the registers (Xlat), 3.7 / 3.8 the ISA mode, 3.10 PC-relative
instructions (Table 3.14: the base PC), 3.11 EXTEND, 3.13 jumps and
branches, 3.14 the formats, 3.15 the encodings (Tables 3.18-3.27), 3.16
the instruction stream, ch. 4 every instruction; the 24K (MD00343 1.2.1.11,
ch. 13): "a decompressor converts the MIPS16e 16-bit instructions ... into
the 32-bit equivalents".  binutils ``opcodes/mips16-opc.c`` gives GNU's
syntax; ``tests/test_mips16.py`` decodes every halfword against
``objdump -m mips:16``.

eMIPS does what the 24K's recoder does: an instruction with a MIPS32
equivalent is executed *as* that MIPS32 instruction (:func:`expand`, so
traces, views, counters and exceptions are the MIPS32 ones); the others
(PC-relative, SAVE / RESTORE, the branches and jumps, JAL / JALX) have
their own executors (:data:`EXECUTORS`, entries ``M16.*``).  Only the
MIPS32 MIPS16e: the MIPS64-only encodings (β) and the reserved ones (∗)
are Reserved Instruction, as is ASMACRO (no implementation-defined
macros) and EXTEND before a non-extensible instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .isa import ISADecodeError

M32 = 0xFFFFFFFF
#: Xlat (Figure 4-1): the 3-bit register fields -> GPR numbers
XLAT = (16, 17, 2, 3, 4, 5, 6, 7)
#: the registers SAVE / RESTORE's xsregs field adds, in order
XSREGS = (18, 19, 20, 21, 22, 23, 30)
EXTEND = 0b11110


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


@dataclass(frozen=True)
class M16:
    """A decoded MIPS16e instruction.  ``form`` names its encoding (e.g.
    ``ADDIUSP`` = ADDIU rx, sp, imm), ``f`` its operands: GPR numbers
    after Xlat (``rx``, ``ry``, ``rz``, ``r32``), ``imm`` = the value the
    instruction uses (scaled and extended as the manual says), ``code``,
    SAVE / RESTORE's ``ra s0 s1 xsregs aregs frame``, JAL's ``target``."""

    form: str
    size: int
    extended: bool
    raw: int
    f: dict = field(default_factory=dict, compare=False)

    @property
    def mnemonic(self) -> str:
        return MNEMONIC.get(self.form, self.form.lower())


#: the forms that print as another mnemonic
MNEMONIC = {"ADDIU8": "addiu", "ADDIU3": "addiu", "ADJSP": "addiu", "ADDIUSP": "addiu",
            "ADDIUPC": "addiu", "LWSP": "lw", "LWPC": "lw", "SWSP": "sw", "SWRASP": "sw",
            "JRRA": "jr", "JRCRA": "jrc", "MOV32R": "move", "MOVR32": "move"}

#: may an EXTEND precede it (Table 3.15)
EXTENSIBLE = {"ADDIUSP", "ADDIUPC", "B", "BEQZ", "BNEZ", "SLL", "SRL", "SRA", "ADDIU3",
              "ADDIU8", "SLTI", "SLTIU", "BTEQZ", "BTNEZ", "SWRASP", "ADJSP", "SAVE",
              "RESTORE", "LI", "CMPI", "LB", "LH", "LWSP", "LW", "LBU", "LHU", "LWPC",
              "SB", "SH", "SWSP", "SW"}
#: extended formats' fields fixed at 0 (3.14.13-3.14.19): (instruction
#: halfword mask, EXTEND mask); other values are MIPS16e2 instructions or
#: undefined: Reserved Instruction (binutils decodes no MIPS16e there)
_RI_ZERO = (0x00E0, 0)
_EXT_ZERO = {**dict.fromkeys(("ADDIUSP", "ADDIUPC", "BEQZ", "BNEZ", "ADDIU8", "SLTI",
                              "SLTIU", "LI", "CMPI", "LWSP", "LWPC", "SWSP", "BTEQZ",
                              "BTNEZ", "SWRASP", "ADJSP"), _RI_ZERO),
             "B": (0x07E0, 0), **dict.fromkeys(("SLL", "SRL", "SRA"), (0x001C, 0x001F))}
#: the branches (compact: no delay slot) and the jumps with a delay slot
BRANCHES = {"B", "BEQZ", "BNEZ", "BTEQZ", "BTNEZ"}
JUMPS_DELAYED = {"JR", "JRRA", "JALR", "JAL", "JALX"}
JUMPS_COMPACT = {"JRC", "JRCRA", "JALRC"}

_RR = {2: "SLT", 3: "SLTU", 4: "SLLV", 6: "SRLV", 7: "SRAV", 10: "CMP", 11: "NEG",
       12: "AND", 13: "OR", 14: "XOR", 15: "NOT", 16: "MFHI", 18: "MFLO", 24: "MULT",
       25: "MULTU", 26: "DIV", 27: "DIVU"}
_CNVT = {0: "ZEB", 1: "ZEH", 4: "SEB", 5: "SEH"}
_JR = {0: "JR", 1: "JRRA", 2: "JALR", 4: "JRC", 5: "JRCRA", 6: "JALRC"}
_LOADS = {16: ("LB", 0), 17: ("LH", 1), 19: ("LW", 2), 20: ("LBU", 0), 21: ("LHU", 1),
          24: ("SB", 0), 25: ("SH", 1), 27: ("SW", 2)}


class M16DecodeError(ISADecodeError):
    """No MIPS32 MIPS16e instruction (the Machine: Reserved Instruction);
    ``why`` says which rule."""

    def __init__(self, why: str, word: int = 0) -> None:
        Exception.__init__(self, f"no MIPS16e instruction: {why}")
        self.word = word
        self.why = why


def _ri(why: str) -> ISADecodeError:
    return M16DecodeError(why)


def decode(h: int, second: Callable[[], int] | None = None) -> M16:
    """Decode the instruction whose first halfword is ``h``; ``second()``
    gives the next halfword (EXTEND and JAL / JALX are 32 bits).  An
    encoding that is no MIPS32 MIPS16e instruction raises
    :class:`ISADecodeError` (the Machine: Reserved Instruction)."""
    h &= 0xFFFF
    op = h >> 11
    if op == 3:                                  # JAL / JALX: 32 bits
        if second is None:
            raise _ri("JAL / JALX needs its second halfword")
        lo = second() & 0xFFFF
        target = (((h & 0x1F) << 21) | (((h >> 5) & 0x1F) << 16) | lo) << 2
        return M16("JALX" if h >> 10 & 1 else "JAL", 4, False, (h << 16) | lo,
                   {"target": target})
    if op == EXTEND:
        if second is None:
            raise _ri("EXTEND needs the instruction it extends")
        i = second() & 0xFFFF
        return _decode(i, ext=h)
    return _decode(h, ext=None)


def _decode(h: int, ext: int | None) -> M16:
    op = h >> 11
    rx, ry, rz = XLAT[(h >> 8) & 7], XLAT[(h >> 5) & 7], XLAT[(h >> 2) & 7]
    extended = ext is not None
    raw = (ext << 16) | h if extended else h
    size = 4 if extended else 2
    imm16 = (((ext & 0x1F) << 11) | (((ext >> 5) & 0x3F) << 5) | (h & 0x1F)) if extended else 0

    def done(form: str, **f) -> M16:
        if extended:
            if form not in EXTENSIBLE:
                raise _ri(f"{form} is not extensible (EXTEND: Reserved Instruction)")
            zero = _EXT_ZERO.get(form)
            if zero and (h & zero[0] or ext & zero[1]):
                raise _ri(f"extended {form}: a field the format fixes at 0 is not 0")
        return M16(form, size, extended, raw, f)

    def simm(bits: int, scale: int, signed: bool) -> int:
        """The unextended immediate (its low ``bits``), or the extended 16."""
        if extended:
            return _sx(imm16, 16)
        v = h & ((1 << bits) - 1)
        return (_sx(v, bits) if signed else v) << scale

    if op == 0:
        return done("ADDIUSP", rx=rx, imm=simm(8, 2, False))
    if op == 1:
        return done("ADDIUPC", rx=rx, imm=simm(8, 2, False))
    if op == 2:
        return done("B", imm=(_sx(imm16, 16) if extended else _sx(h & 0x7FF, 11)) << 1)
    if op in (4, 5):
        return done("BEQZ" if op == 4 else "BNEZ", rx=rx,
                    imm=(_sx(imm16, 16) if extended else _sx(h & 0xFF, 8)) << 1)
    if op == 6:
        f = h & 3
        if f == 1:
            raise _ri("DSLL: MIPS64 only")
        if extended:
            if ext >> 5 & 1:                     # s5: a 64-bit shift amount
                raise _ri("a shift amount above 31: MIPS64 only")
            sa = (ext >> 6) & 0x1F
        else:
            sa = (h >> 2) & 7 or 8
        return done({0: "SLL", 2: "SRL", 3: "SRA"}[f], rx=rx, ry=ry, sa=sa)
    if op == 8:
        if h >> 4 & 1:
            raise _ri("DADDIU: MIPS64 only")
        imm = _sx(((ext & 0xF) << 11) | (((ext >> 4) & 0x7F) << 4) | (h & 0xF), 15) \
            if extended else _sx(h & 0xF, 4)
        return done("ADDIU3", rx=rx, ry=ry, imm=imm)
    if op == 9:
        return done("ADDIU8", rx=rx, imm=simm(8, 0, True))
    if op in (10, 11):
        return done("SLTI" if op == 10 else "SLTIU", rx=rx, imm=simm(8, 0, False))
    if op == 12:
        funct = (h >> 8) & 7
        if funct in (0, 1):
            return done("BTEQZ" if funct == 0 else "BTNEZ",
                        imm=(_sx(imm16, 16) if extended else _sx(h & 0xFF, 8)) << 1)
        if funct == 2:
            return done("SWRASP", imm=simm(8, 2, False))
        if funct == 3:
            return done("ADJSP", imm=simm(8, 3, True))
        if funct == 4:
            return _svrs(h, ext, done)
        if funct == 5:
            r32 = ((h >> 5) & 7) | (((h >> 3) & 3) << 3)
            return done("MOV32R", r32=r32, rz=XLAT[h & 7])
        if funct == 7:
            return done("MOVR32", ry=ry, r32=h & 0x1F)
        raise _ri("I8 funct 6: reserved")
    if op == 13:
        return done("LI", rx=rx, imm=(imm16 if extended else h & 0xFF))
    if op == 14:
        return done("CMPI", rx=rx, imm=(imm16 if extended else h & 0xFF))
    if op in _LOADS:
        form, scale = _LOADS[op]
        return done(form, rx=rx, ry=ry, imm=simm(5, scale, False))
    if op == 18:
        return done("LWSP", rx=rx, imm=simm(8, 2, False))
    if op == 22:
        return done("LWPC", rx=rx, imm=simm(8, 2, False))
    if op == 26:
        return done("SWSP", rx=rx, imm=simm(8, 2, False))
    if op == 28:
        if extended:
            raise _ri("ASMACRO: no implementation-defined macros")
        f = h & 3
        if f not in (1, 3):
            raise _ri("DADDU / DSUBU: MIPS64 only")
        return done("ADDU" if f == 1 else "SUBU", rx=rx, ry=ry, rz=rz)
    if op == 29:
        return _rr(h, done, rx, ry)
    raise _ri(f"major opcode {op}: MIPS64 only or reserved")


def _rr(h: int, done, rx: int, ry: int) -> M16:
    funct = h & 0x1F
    sel = (h >> 5) & 7
    if funct == 0:
        form = _JR.get(sel)
        if form is None:
            raise _ri("J(AL)R(C) ry 3 / 7: reserved")
        if form in ("JRRA", "JRCRA") and (h >> 8) & 7:
            raise _ri("JR ra: the rx field must be 0")
        return done(form, rx=rx)
    if funct in (1, 5):
        return done("SDBBP" if funct == 1 else "BREAK", code=(h >> 5) & 0x3F)
    if funct == 17:
        form = _CNVT.get(sel)
        if form is None:
            raise _ri("CNVT: MIPS64 only or reserved")
        return done(form, rx=rx)
    form = _RR.get(funct)
    if form is None:
        raise _ri(f"RR funct {funct}: MIPS64 only or reserved")
    if form in ("MFHI", "MFLO") and sel:
        raise _ri(f"{form}: bits 7..5 must be 0")
    return done(form, rx=rx, ry=ry)


def _svrs(h: int, ext: int | None, done) -> M16:
    f = {"ra": h >> 6 & 1, "s0": h >> 5 & 1, "s1": h >> 4 & 1}
    if ext is None:
        frame = h & 0xF
        f.update(xsregs=0, aregs=0, frame=(frame or 16) * 8)
    else:
        f.update(xsregs=(ext >> 8) & 7, aregs=ext & 0xF,
                 frame=((((ext >> 4) & 0xF) << 4) | (h & 0xF)) * 8)
        if f["aregs"] == 0xF:
            raise _ri("SAVE / RESTORE aregs 1111: UNPREDICTABLE (Reserved here)")
    return done("SAVE" if h >> 7 & 1 else "RESTORE", **f)


# ------------------------------------------------------------ the registers SAVE uses

#: aregs -> (incoming arguments, static registers) (the tables of SAVE / RESTORE)
def args_statics(aregs: int) -> tuple[int, int]:
    args = {0: 0, 1: 0, 2: 0, 3: 0, 11: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2,
            12: 3, 13: 3, 14: 4}[aregs]
    statics = {0: 0, 4: 0, 8: 0, 12: 0, 14: 0, 1: 1, 5: 1, 9: 1, 13: 1, 2: 2, 6: 2,
               10: 2, 3: 3, 7: 3, 11: 4}[aregs]
    return args, statics


# ------------------------------------------------------------ the disassembler

def _reg(n: int, abi: bool) -> str:
    from .syntax import gpr_name
    return gpr_name(n, abi)


def text(i: M16, pc: int, *, abi: bool = True) -> str:
    """GNU's syntax (objdump ``-m mips:16``, with ``$`` and ``, ``)."""
    r = lambda n: _reg(n, abi)               # noqa: E731
    f, m = i.f, i.mnemonic
    form = i.form
    if form in ("ADDIU8", "LI", "CMPI", "SLTI", "SLTIU"):
        return f"{m} {r(f['rx'])}, {f['imm']}"
    if form == "ADDIU3":
        return f"addiu {r(f['ry'])}, {r(f['rx'])}, {f['imm']}"
    if form == "ADJSP":
        return f"addiu {r(29)}, {f['imm']}"
    if form in ("ADDIUSP", "ADDIUPC"):
        base = r(29) if form == "ADDIUSP" else "$pc"
        return f"addiu {r(f['rx'])}, {base}, {f['imm']}"
    if form in ("LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW"):
        return f"{m} {r(f['ry'])}, {f['imm']}({r(f['rx'])})"
    if form in ("LWSP", "SWSP"):
        return f"{m} {r(f['rx'])}, {f['imm']}({r(29)})"
    if form == "LWPC":
        return f"lw {r(f['rx'])}, {f['imm']}($pc)"
    if form == "SWRASP":
        return f"sw {r(31)}, {f['imm']}({r(29)})"
    if form in BRANCHES:
        t = branch_target(i, pc)
        reg = f"{r(f['rx'])}, " if "rx" in f else ""
        return f"{m} {reg}0x{t:08x}"
    if form in ("JAL", "JALX"):
        return f"{m} 0x{jump_target(i, pc):08x}"
    if form in ("JR", "JRC"):
        return f"{m} {r(f['rx'])}"
    if form in ("JRRA", "JRCRA"):
        return f"{m} {r(31)}"
    if form in ("JALR", "JALRC"):                       # (ra is implied)
        return f"{m} {r(f['rx'])}"
    if form in ("SLL", "SRL", "SRA"):
        return f"{m} {r(f['rx'])}, {r(f['ry'])}, {f['sa']}"
    if form in ("SLLV", "SRLV", "SRAV"):
        return f"{m} {r(f['ry'])}, {r(f['rx'])}"
    if form in ("ADDU", "SUBU"):
        return f"{m} {r(f['rz'])}, {r(f['rx'])}, {r(f['ry'])}"
    if form in ("MFHI", "MFLO", "ZEB", "ZEH", "SEB", "SEH"):
        return f"{m} {r(f['rx'])}"
    if form == "MOV32R":
        return f"move {r(f['r32'])}, {r(f['rz'])}" if (f["r32"], f["rz"]) != (0, 16) else "nop"
    if form == "MOVR32":
        return f"move {r(f['ry'])}, {r(f['r32'])}"
    if form in ("DIV", "DIVU"):                         # (GNU as: 2 operands = its macro)
        return f"{m} {r(0)}, {r(f['rx'])}, {r(f['ry'])}"
    if form in ("BREAK", "SDBBP"):
        return f"{m} {f['code']}" if f["code"] else m
    if form in ("SAVE", "RESTORE"):
        return f"{m} {save_list(i, abi)}"
    return f"{m} {r(f['rx'])}, {r(f['ry'])}"


def save_list(i: M16, abi: bool = True) -> str:
    """SAVE / RESTORE's operands as objdump prints them: the argument
    registers, the frame size, ra, the s-registers, the statics."""
    r = lambda n: _reg(n, abi)               # noqa: E731
    f = i.f
    args, statics = args_statics(f["aregs"])
    out = []
    if args:
        out.append(r(4) if args == 1 else f"{r(4)}-{r(3 + args)}")
    out.append(str(f["frame"]))
    if f["ra"]:
        out.append(r(31))
    sregs = ([16] if f["s0"] else []) + ([17] if f["s1"] else []) + list(XSREGS[:f["xsregs"]])
    if sregs:
        out.extend(_ranges(sregs, r))
    if statics:
        out.append(r(7) if statics == 1 else f"{r(8 - statics)}-{r(7)}")
    return ", ".join(out)


def _ranges(regs: list[int], r) -> list[str]:
    """s0, s1, s2-s7, s8 as objdump groups them (s0-s1, s2-s3 ...)."""
    out, start, prev = [], None, None
    for n in regs:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(r(start) if start == prev else f"{r(start)}-{r(prev)}")
            start = prev = n
    if start is not None:
        out.append(r(start) if start == prev else f"{r(start)}-{r(prev)}")
    return out


# ------------------------------------------------------------ targets

def branch_target(i: M16, pc: int) -> int:
    """B / BEQZ / BNEZ / BTEQZ / BTNEZ: PC + 2 (+ 4 extended) + offset."""
    return (pc + i.size + i.f["imm"]) & M32


def jump_target(i: M16, pc: int) -> int:
    """JAL / JALX: the 256 MB region of the delay slot (PC + 4)."""
    return (((pc + 4) & 0xF0000000) | i.f["target"]) & M32


# ------------------------------------------------------------ the recoder

def _i(op: int, rs: int, rt: int, imm: int) -> int:
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def _r(rs: int, rt: int, rd: int, sa: int, funct: int, op: int = 0) -> int:
    return (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


#: loads / stores: the MIPS32 opcodes
_MEM_OP = {"LB": 0x20, "LH": 0x21, "LW": 0x23, "LBU": 0x24, "LHU": 0x25,
           "SB": 0x28, "SH": 0x29, "SW": 0x2B}
_ALU2 = {"AND": 0x24, "OR": 0x25, "XOR": 0x26}
_SHIFT = {"SLL": 0, "SRL": 2, "SRA": 3}
_SHIFTV = {"SLLV": 4, "SRLV": 6, "SRAV": 7}
_HILO = {"MULT": 0x18, "MULTU": 0x19, "DIV": 0x1A, "DIVU": 0x1B}
#: the forms without a MIPS32 twin: run by sim's ``M16.*`` executors
SPECIAL = BRANCHES | JUMPS_DELAYED | JUMPS_COMPACT | {"ADDIUPC", "LWPC", "SAVE", "RESTORE"}


def expand(i: M16) -> int | None:
    """The MIPS32 instruction the 24K's recoder turns ``i`` into (MD00343
    1.2.1.11), or None for the :data:`SPECIAL` forms.  The condition-code
    register T is GPR 24 (Table 3.1): CMP / CMPI / SLT* write it."""
    f, form = i.f, i.form
    if form in SPECIAL:
        return None
    if form == "ADDIU8":
        return _i(0x09, f["rx"], f["rx"], f["imm"])
    if form == "ADDIU3":
        return _i(0x09, f["rx"], f["ry"], f["imm"])
    if form == "ADJSP":
        return _i(0x09, 29, 29, f["imm"])
    if form == "ADDIUSP":
        return _i(0x09, 29, f["rx"], f["imm"])
    if form == "LI":
        return _i(0x0D, 0, f["rx"], f["imm"])           # ORI: zero-extended
    if form == "CMPI":
        return _i(0x0E, f["rx"], 24, f["imm"])          # XORI T
    if form in ("SLTI", "SLTIU"):
        return _i(0x0A if form == "SLTI" else 0x0B, f["rx"], 24, f["imm"])
    if form in _MEM_OP:
        return _i(_MEM_OP[form], f["rx"], f["ry"], f["imm"])
    if form in ("LWSP", "SWSP"):
        return _i(0x23 if form == "LWSP" else 0x2B, 29, f["rx"], f["imm"])
    if form == "SWRASP":
        return _i(0x2B, 29, 31, f["imm"])
    if form in ("ADDU", "SUBU"):
        return _r(f["rx"], f["ry"], f["rz"], 0, 0x21 if form == "ADDU" else 0x23)
    if form in ("SLT", "SLTU"):
        return _r(f["rx"], f["ry"], 24, 0, 0x2A if form == "SLT" else 0x2B)
    if form == "CMP":
        return _r(f["rx"], f["ry"], 24, 0, 0x26)       # XOR T
    if form == "NEG":
        return _r(0, f["ry"], f["rx"], 0, 0x23)        # SUBU rx, $0, ry
    if form == "NOT":
        return _r(f["ry"], 0, f["rx"], 0, 0x27)        # NOR rx, ry, $0
    if form in _ALU2:
        return _r(f["rx"], f["ry"], f["rx"], 0, _ALU2[form])
    if form in _SHIFT:
        return _r(0, f["ry"], f["rx"], f["sa"], _SHIFT[form])
    if form in _SHIFTV:
        return _r(f["rx"], f["ry"], f["ry"], 0, _SHIFTV[form])
    if form in ("MFHI", "MFLO"):
        return _r(0, 0, f["rx"], 0, 0x10 if form == "MFHI" else 0x12)
    if form in _HILO:
        return _r(f["rx"], f["ry"], 0, 0, _HILO[form])
    if form in ("ZEB", "ZEH"):
        return _i(0x0C, f["rx"], f["rx"], 0xFF if form == "ZEB" else 0xFFFF)
    if form in ("SEB", "SEH"):
        return _r(0, f["rx"], f["rx"], 0x10 if form == "SEB" else 0x18, 0x20, op=0x1F)
    if form == "MOV32R":
        return _r(f["rz"], 0, f["r32"], 0, 0x25)       # OR r32, rz, $0
    if form == "MOVR32":
        return _r(f["r32"], 0, f["ry"], 0, 0x25)       # OR ry, r32, $0
    if form == "BREAK":
        return (f["code"] << 16) | 0x0D
    if form == "SDBBP":
        return (0x1C << 26) | (f["code"] << 6) | 0x3F
    raise AssertionError(form)


#: synthetic isaref-like entries of the SPECIAL forms (the Machine
#: dispatches ``M16.<form>``; the views know them by ``isa``)
_FAMILY = {**dict.fromkeys(BRANCHES, "branch_compact"),
           **dict.fromkeys(JUMPS_COMPACT, "jump_compact"),
           **dict.fromkeys(JUMPS_DELAYED, "jump"),
           "ADDIUPC": "imm", "LWPC": "load", "SAVE": "store", "RESTORE": "load"}
ENTRIES = {form: {"name": f"M16.{form}", "mnemonic": MNEMONIC.get(form, form.lower()),
                  "family": fam, "status": "core", "pic32": False, "isa": "mips16e",
                  "decode": {"lead": True}, "releases": ["R1", "R2", "R3", "R5"]}
           for form, fam in _FAMILY.items()}


# ------------------------------------------------------------ the assembler

#: GPR -> the 3-bit field (the inverse of Xlat)
FIELD = {r: k for k, r in enumerate(XLAT)}
NOP = 0x6500


class AsmError16(ValueError):
    """A MIPS16e statement that cannot be encoded (the assembler adds the
    line)."""


class Unresolved(Exception):
    """A label not known yet (pass 1): size the short form."""


def _gpr(tok: str) -> int:
    from .syntax import parse_gpr
    r = parse_gpr(tok)
    if r is None:
        raise AsmError16(f"register expected, got {tok!r}")
    return r


def _r3(tok: str) -> int:
    r = _gpr(tok)
    if r not in FIELD:
        raise AsmError16(f"{tok} is not a MIPS16e register ($s0, $s1, $v0, $v1, $a0-$a3)")
    return FIELD[r]


def _is_reg(tok: str, name: str | None = None) -> bool:
    from .syntax import parse_gpr
    t = tok.strip().lower()
    if name is not None:
        return t == name or parse_gpr(t) == {"$sp": 29, "$ra": 31}.get(name)
    return parse_gpr(t) is not None


def _fits(v: int, lo: int, hi: int, step: int = 1) -> bool:
    return lo <= v <= hi and v % step == 0


def _ext(imm: int, h: int) -> list[int]:
    """EXT-I / EXT-RI / EXT-RRI / EXT-I8: imm[10:5] imm[15:11] in the
    EXTEND, imm[4:0] in the instruction."""
    imm &= 0xFFFF
    return [0xF000 | (((imm >> 5) & 0x3F) << 5) | ((imm >> 11) & 0x1F), h | (imm & 0x1F)]


def _mem(op: int, tok: str):
    """``off(base)`` -> (offset expression, base token)."""
    import re
    m = re.fullmatch(r"(.*)\(\s*(\$\w+)\s*\)", tok.strip())
    if not m:
        raise AsmError16(f"offset(base) expected, got {tok!r}")
    return (m.group(1).strip() or "0"), m.group(2)


def encode(mn: str, ops: list[str], pc: int, value: Callable[[str], int], *,
           reorder: bool = True, long: bool = False) -> list[list[int]]:
    """GNU as's MIPS16e (binutils ``mips16-opc.c`` syntax): the
    instructions (each a list of halfwords) a statement assembles to at
    ``pc``.  The short form when the operand fits, else the EXTENDed one
    (GNU as does the same); ``long`` forces the extended one (a branch /
    PC-relative label too far for the short form: the assembler relaxes).
    ``value(expr)`` raises :class:`Unresolved` for a label not known yet.
    ``.set reorder``: JR / JALR become JRC / JALRC, JAL / JALX get a NOP
    in the delay slot (GNU as fills it with an earlier instruction where
    it can; eMIPS never moves instructions)."""
    n = len(ops)

    def val(expr: str) -> int | None:
        try:
            return value(expr)
        except Unresolved:
            return None

    def pick(v: int | None, short_ok: Callable[[int], bool], short, ext) -> list[list[int]]:
        if v is not None and not long and short_ok(v):
            return [short(v)]
        if v is None and not long:              # pass 1: the short size
            return [short(0)]
        return [ext(0 if v is None else v)]

    def need(k: int) -> None:
        if len(ops) != k:
            raise AsmError16(f"{mn} takes {k} operands")

    if mn == "nop":
        need(0)
        return [[NOP]]
    if mn == "move":
        need(2)
        d, s = _gpr(ops[0]), _gpr(ops[1])
        if d in FIELD:                           # MOVE ry, r32
            return [[0x6700 | (FIELD[d] << 5) | s]]
        if s in FIELD:                           # MOVE r32, rz
            return [[0x6500 | ((d & 7) << 5) | ((d >> 3) << 3) | FIELD[s]]]
        raise AsmError16("move needs a MIPS16e register on one side")
    if mn in ("li", "cmpi", "cmp", "slti", "sltiu", "slt", "sltu") and n == 2 and \
            not _is_reg(ops[1]):
        op = {"li": 13, "cmpi": 14, "cmp": 14, "slti": 10, "slt": 10, "sltiu": 11,
              "sltu": 11}[mn]
        rx = _r3(ops[0])
        signed_ext = op in (10, 11)
        v = val(ops[1])
        if v is not None and not (0 <= v <= 0xFFFF if not signed_ext else
                                  -0x8000 <= v <= 0xFFFF):
            raise AsmError16(f"{mn}: {v} out of range")
        return pick(v, lambda x: 0 <= x <= 255, lambda x: [(op << 11) | (rx << 8) | x],
                    lambda x: _ext(x, (op << 11) | (rx << 8)))
    if mn in ("cmp", "slt", "sltu"):
        need(2)
        f = {"cmp": 10, "slt": 2, "sltu": 3}[mn]
        return [[0xE800 | (_r3(ops[0]) << 8) | (_r3(ops[1]) << 5) | f]]
    if mn in ("addiu", "addu"):
        return _addiu(mn, ops, pc, val, pick)
    if mn == "subu":
        if n == 2:
            ops = [ops[0], ops[0], ops[1]]
        rz, rx, ry = (_r3(o) for o in ops)
        return [[0xE003 | (rx << 8) | (ry << 5) | (rz << 2)]]
    if mn in ("lb", "lbu", "lh", "lhu", "lw", "sb", "sh", "sw"):
        return _loadstore(mn, ops, pc, val, pick)
    if mn == "la":
        need(2)
        rx = _r3(ops[0])
        v = val(ops[1])
        off = None if v is None else v - (pc & ~3)
        return pick(off, lambda x: _fits(x, 0, 1020, 4),
                    lambda x: [(1 << 11) | (rx << 8) | (x >> 2)],
                    lambda x: _branch_ext(x, (1 << 11) | (rx << 8), 0xFFFF))
    if mn in ("b", "beqz", "bnez", "bteqz", "btnez"):
        return _branches(mn, ops, pc, val, pick)
    if mn in ("jal", "jalx"):
        need(1)
        v = val(ops[0])
        t = 0 if v is None else v
        if v is not None:
            if t & 3:
                raise AsmError16(f"{mn}: the target 0x{t:08x} is not word-aligned")
            if (t ^ (pc + 4)) & 0xF0000000:
                raise AsmError16(f"{mn}: the target is outside the 256 MB region")
        idx = (t >> 2) & 0x3FFFFFF
        h = 0x1800 | ((mn == "jalx") << 10) | (((idx >> 16) & 0x1F) << 5) | (idx >> 21)
        out = [[h, idx & 0xFFFF]]
        return out + ([[NOP]] if reorder else [])
    if mn in ("jr", "jrc", "jalr", "jalrc", "j"):
        if mn == "j":
            mn = "jr"
        if mn in ("jalr", "jalrc") and n == 2:
            if not _is_reg(ops[0], "$ra"):
                raise AsmError16(f"{mn}: the link register is $ra")
            ops = ops[1:]
        need(1)
        compact = mn.endswith("c") or reorder
        link = mn.startswith("jal")
        if not link and _is_reg(ops[0], "$ra"):
            return [[0xE8A0 if compact else 0xE820]]
        sel = (4 if compact else 0) | (2 if link else 0)
        return [[0xE800 | (_r3(ops[0]) << 8) | (sel << 5)]]
    if mn in ("sll", "srl", "sra", "sllv", "srlv", "srav"):
        return _shifts(mn, ops, val, pick)
    if mn in ("and", "or", "xor", "neg", "not"):
        if mn in ("neg", "not") and n == 1:
            ops = [ops[0], ops[0]]
        need(2)
        f = {"and": 12, "or": 13, "xor": 14, "neg": 11, "not": 15}[mn]
        return [[0xE800 | (_r3(ops[0]) << 8) | (_r3(ops[1]) << 5) | f]]
    if mn in ("mfhi", "mflo"):
        need(1)
        return [[0xE800 | (_r3(ops[0]) << 8) | (16 if mn == "mfhi" else 18)]]
    if mn in ("mult", "multu", "div", "divu"):
        if mn.startswith("div") and n == 3:
            if _gpr(ops[0]) != 0:
                raise AsmError16(f"{mn} rd form: the first operand must be $zero")
            ops = ops[1:]
        need(2)
        f = {"mult": 24, "multu": 25, "div": 26, "divu": 27}[mn]
        return [[0xE800 | (_r3(ops[0]) << 8) | (_r3(ops[1]) << 5) | f]]
    if mn in ("zeb", "zeh", "seb", "seh"):
        need(1)
        sel = {"zeb": 0, "zeh": 1, "seb": 4, "seh": 5}[mn]
        return [[0xE811 | (_r3(ops[0]) << 8) | (sel << 5)]]
    if mn in ("break", "sdbbp"):
        code = val(ops[0]) if n else 0
        if code is None or not 0 <= code <= 63:
            raise AsmError16(f"{mn}: the code is 0..63")
        return [[0xE800 | (code << 5) | (5 if mn == "break" else 1)]]
    if mn in ("save", "restore"):
        return [_save(mn, ops, val)]
    raise AsmError16(f"{mn} is not a MIPS16e instruction")


def _addiu(mn, ops, pc, val, pick):
    n = len(ops)
    if n == 3 and _is_reg(ops[2]):               # addu rz, rx, ry
        rz, rx, ry = (_r3(o) for o in ops)
        return [[0xE001 | (rx << 8) | (ry << 5) | (rz << 2)]]
    if n == 2 and _is_reg(ops[1]):               # addu rz, ry = addu rz, rz, ry
        rz, ry = _r3(ops[0]), _r3(ops[1])
        return [[0xE001 | (rz << 8) | (ry << 5) | (rz << 2)]]
    if n == 3 and _is_reg(ops[0], "$sp") and _is_reg(ops[1], "$sp"):
        ops = [ops[0], ops[2]]
        n = 2
    if n == 2 and _is_reg(ops[0], "$sp"):        # ADJSP
        v = val(ops[1])
        return pick(v, lambda x: _fits(x, -1024, 1016, 8),
                    lambda x: [0x6300 | ((x >> 3) & 0xFF)], lambda x: _ext(x, 0x6300))
    if n == 2:                                    # ADDIU8
        rx = _r3(ops[0])
        v = val(ops[1])
        return pick(v, lambda x: _fits(x, -128, 127),
                    lambda x: [(9 << 11) | (rx << 8) | (x & 0xFF)],
                    lambda x: _ext(x, (9 << 11) | (rx << 8)))
    if n != 3:
        raise AsmError16(f"{mn} takes 2 or 3 operands")
    rx = _r3(ops[0])
    if ops[1].strip().lower() == "$pc":           # ADDIUPC
        v = val(ops[2])
        return pick(v, lambda x: _fits(x, 0, 1020, 4),
                    lambda x: [(1 << 11) | (rx << 8) | (x >> 2)],
                    lambda x: _ext(x, (1 << 11) | (rx << 8)))
    if _is_reg(ops[1], "$sp"):                    # ADDIUSP
        v = val(ops[2])
        return pick(v, lambda x: _fits(x, 0, 1020, 4),
                    lambda x: [(rx << 8) | (x >> 2)], lambda x: _ext(x, rx << 8))
    ry, rxs = rx, _r3(ops[1])                     # ADDIU ry, rx, imm (RRI-A)
    v = val(ops[2])

    def ext15(x: int) -> list[int]:
        x &= 0x7FFF
        return [0xF000 | (((x >> 4) & 0x7F) << 4) | ((x >> 11) & 0xF),
                (8 << 11) | (rxs << 8) | (ry << 5) | (x & 0xF)]
    if v is not None and not -0x4000 <= v <= 0x3FFF:
        raise AsmError16(f"addiu: {v} is out of range (-16384 .. 16383)")
    return pick(v, lambda x: _fits(x, -8, 7),
                lambda x: [(8 << 11) | (rxs << 8) | (ry << 5) | (x & 0xF)], ext15)


def _loadstore(mn, ops, pc, val, pick):
    if len(ops) != 2:
        raise AsmError16(f"{mn} takes 2 operands")
    store = mn.startswith("s")
    if mn == "lw" and "(" not in ops[1]:          # lw rx, label (PC-relative)
        rx = _r3(ops[0])
        v = val(ops[1])
        off = None if v is None else v - (pc & ~3)
        return pick(off, lambda x: _fits(x, 0, 1020, 4),
                    lambda x: [(22 << 11) | (rx << 8) | (x >> 2)],
                    lambda x: _branch_ext(x, (22 << 11) | (rx << 8), 0xFFFF))
    expr, base = _mem(0, ops[1])
    v = val(expr)
    if v is not None and not -0x8000 <= v <= 0x7FFF:
        raise AsmError16(f"{mn}: offset {v} out of range")
    b = base.strip().lower()
    if b == "$pc":
        if mn != "lw":
            raise AsmError16(f"{mn}: no PC-relative form")
        rx = _r3(ops[0])
        return pick(v, lambda x: _fits(x, 0, 1020, 4),
                    lambda x: [(22 << 11) | (rx << 8) | (x >> 2)],
                    lambda x: _ext(x, (22 << 11) | (rx << 8)))
    if _is_reg(base, "$sp") and mn in ("lw", "sw"):
        if store and _is_reg(ops[0], "$ra"):      # SW ra, off(sp)
            return pick(v, lambda x: _fits(x, 0, 1020, 4),
                        lambda x: [0x6200 | (x >> 2)], lambda x: _ext(x, 0x6200))
        rx = _r3(ops[0])
        op = 26 if store else 18
        return pick(v, lambda x: _fits(x, 0, 1020, 4),
                    lambda x: [(op << 11) | (rx << 8) | (x >> 2)],
                    lambda x: _ext(x, (op << 11) | (rx << 8)))
    op, scale = {"lb": (16, 0), "lh": (17, 1), "lw": (19, 2), "lbu": (20, 0),
                 "lhu": (21, 1), "sb": (24, 0), "sh": (25, 1), "sw": (27, 2)}[mn]
    ry, rx = _r3(ops[0]), _r3(base)
    word = (op << 11) | (rx << 8) | (ry << 5)
    return pick(v, lambda x: _fits(x, 0, 31 << scale, 1 << scale),
                lambda x: [word | (x >> scale)], lambda x: _ext(x, word))


def _branch_ext(x: int, h: int, _mask: int) -> list[int]:
    if not -0x8000 <= x <= 0x7FFF:
        raise AsmError16(f"offset {x} out of range of the extended form")
    return _ext(x, h)


def _branches(mn, ops, pc, val, pick):
    reg = mn in ("beqz", "bnez")
    if len(ops) != (2 if reg else 1):
        raise AsmError16(f"{mn} takes {2 if reg else 1} operand(s)")
    v = val(ops[-1])
    if reg:
        rx = _r3(ops[0])
        h = ((4 if mn == "beqz" else 5) << 11) | (rx << 8)
        lo, hi, bits = -256, 254, 0xFF
    elif mn == "b":
        h, lo, hi, bits = 2 << 11, -2048, 2046, 0x7FF
    else:
        h = 0x6000 if mn == "bteqz" else 0x6100
        lo, hi, bits = -256, 254, 0xFF

    def short(x: int) -> list[int]:
        return [h | ((x >> 1) & bits)]

    def ext(x: int) -> list[int]:
        if x & 1 or not -0x10000 <= x <= 0xFFFE:
            raise AsmError16(f"{mn}: the target is out of range")
        return _ext(x >> 1, h)
    if v is None:
        return pick(None, lambda x: True, short, ext)
    if v & 1:
        raise AsmError16(f"{mn}: the target is not halfword-aligned")
    off2, off4 = v - (pc + 2), v - (pc + 4)
    if pick(off2, lambda x: _fits(x, lo, hi, 2), lambda x: [0], lambda x: [0, 0]) == [[0]]:
        return [short(off2)]
    return [ext(off4)]


def _shifts(mn, ops, val, pick):
    if mn.endswith("v") or (len(ops) == 2 and _is_reg(ops[1])):
        if len(ops) != 2:
            raise AsmError16(f"{mn} takes 2 operands")
        f = {"sll": 4, "srl": 6, "sra": 7, "sllv": 4, "srlv": 6, "srav": 7}[mn]
        return [[0xE800 | (_r3(ops[1]) << 8) | (_r3(ops[0]) << 5) | f]]
    if len(ops) == 2:
        ops = [ops[0], ops[0], ops[1]]
    if len(ops) != 3:
        raise AsmError16(f"{mn} takes 3 operands")
    rx, ry = _r3(ops[0]), _r3(ops[1])
    f = {"sll": 0, "srl": 2, "sra": 3}[mn]
    v = val(ops[2])
    if v is not None and not 0 <= v <= 31:
        raise AsmError16(f"{mn}: the shift amount is 0..31")
    word = (6 << 11) | (rx << 8) | (ry << 5) | f
    return pick(v, lambda x: 1 <= x <= 8, lambda x: [word | ((x & 7) << 2)],
                lambda x: [0xF000 | (x << 6), word])


def _save(mn: str, ops: list[str], val) -> list[int]:
    """SAVE / RESTORE: ``[a0-a3 args,] frame, [ra,] [s0, s1, s2-s8 ...,]
    [statics]`` in any order GNU as accepts (args before the frame size,
    statics after)."""
    from .syntax import parse_gpr
    ra = s0 = s1 = 0
    xs, args, statics, frame = 0, 0, 0, None
    sregs: list[int] = []
    for tok in ops:
        t = tok.strip()
        if "-" in t and t.startswith("$"):
            a, b = (parse_gpr(x.strip()) for x in t.split("-"))
            if a is None or b is None:
                raise AsmError16(f"{mn}: bad register range {t!r}")
            regs = list(range(a, b + 1)) if not (a <= 23 < b == 30) else \
                list(range(a, 24)) + [30]
        elif t.startswith("$"):
            r = parse_gpr(t)
            if r is None:
                raise AsmError16(f"{mn}: bad register {t!r}")
            regs = [r]
        else:
            frame = val(t)
            if frame is None:
                raise AsmError16(f"{mn}: the frame size must be a constant")
            continue
        for r in regs:
            if r == 31:
                ra = 1
            elif 4 <= r <= 7:
                if frame is None:
                    args = max(args, r - 3)
                else:
                    statics = max(statics, 8 - r)
            elif r in (16, 17, 18, 19, 20, 21, 22, 23, 30):
                sregs.append(r)
            else:
                raise AsmError16(f"{mn}: ${r} cannot be saved")
    if frame is None:
        frame = 0 if (args or statics) else 0
    s0 = int(16 in sregs)
    s1 = int(17 in sregs)
    extra = sorted(r for r in sregs if r not in (16, 17))
    xs = 0
    for k, r in enumerate(XSREGS):
        if r in extra:
            xs = k + 1
    if extra and extra != list(XSREGS[:xs]):
        raise AsmError16(f"{mn}: the extra s-registers are $s2 up to one in order ($s2-$s8)")
    aregs = _AREGS.get((args, statics))
    if aregs is None:
        raise AsmError16(f"{mn}: {args} arguments + {statics} statics cannot be encoded")
    if frame % 8 or not 0 <= frame <= 2040:
        raise AsmError16(f"{mn}: the frame size is a multiple of 8, 0 .. 2040")
    s = 1 if mn == "save" else 0
    word = 0x6400 | (s << 7) | (ra << 6) | (s0 << 5) | (s1 << 4)
    if not xs and not aregs and 8 <= frame <= 128:
        return [word | ((frame // 8) & 0xF)]
    fs = frame // 8
    return [0xF000 | (xs << 8) | (((fs >> 4) & 0xF) << 4) | aregs, word | (fs & 0xF)]


#: (incoming arguments, statics) -> aregs (the inverse of args_statics)
_AREGS = {args_statics(a): a for a in range(15)}
