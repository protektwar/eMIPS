"""nanoMIPS32: the decoder and the disassembler (PLAN M23 phase 1).

nanoMIPS is the instruction set of the MIPS I7200: a variable-length ISA
of 16-, 32- and 48-bit instructions based on Release 6, with its own
encodings, register lists and assembler syntax, and with no delay slots
and no forbidden slots (MD01247 ch. 1).  It is an architecture of its
own, not an ASE of MIPS32: a nanoMIPS core runs nanoMIPS code only.

Reference: *MIPS Architecture Base: nanoMIPS32 Instruction Set Technical
Reference Manual* (MD01247) rev. 01.01 -- ch. 3 the formats (their
encoding diagrams and decode pseudocode), 4.7 ``decode_gpr``, A.4 the
argument conventions, App. B the opcode map.  binutils' nanoMIPS port
(MediaTek GNU Tools 2025.09-02: ``opcodes/nanomips-opc.c`` /
``nanomips-dis.c``) gives GNU's syntax and the aliases objdump prints
by default; ``tests/test_nanomips.py`` compares both with its objdump.

A row is one hardware format (``ADDIU[R1.SP]``), written as in the
manual's diagrams, most significant bit first: fixed bits, named fields
(``rt:5``, ``s[20:1]`` -- the manual's names; a slice puts its bits at
those positions of the argument, A.4.1) and ``x:N``, bits the hardware
ignores (A.4.3).  Its ``dec`` steps are the format's decode pseudocode
(``rt=gpr3(rt3) rs=29 imm=u``; a step may read what an earlier one
wrote), so ``NM.f`` holds the arguments the instruction's Operation
pseudocode reads.  The formats of one instruction share its ``name``
(the manual's page: ``ADDIU``, ``RESTORE``, ``LL``), which is what the
Machine will execute (phase 2).  Where an instruction page computes an
argument at the top of its Operation instead (BEQIC's offset), the row's
steps do it, so that every branch row has its ``offset``.

A 32-bit encoding is ``first halfword << 16 | second``, a 48-bit one
``h0 << 32 | h1 << 16 | h2``: the diagrams' bit numbers (A.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .isa import ISADecodeError

M32 = 0xFFFFFFFF

#: the p32 ABI's register names (DN00179 ch. 3; objdump prints these)
GPR_NAMES = ["zero", "at", "t4", "t5", "a0", "a1", "a2", "a3",
             "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3",
             "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
             "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"]

#: decode_gpr() (MD01247 4.7): the encoded register fields -> GPR numbers
GPR_LISTS = {
    "gpr3": (16, 17, 18, 19, 4, 5, 6, 7),
    "gpr3.src.store": (0, 17, 18, 19, 4, 5, 6, 7),
    "gpr1": (4, 5),
    "gpr2.reg1": (4, 5, 6, 7),
    "gpr2.reg2": (5, 6, 7, 8),
    "gpr4": (8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23),
    "gpr4.zero": (8, 9, 10, 0, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23),
}


def gpr_name(i: int, abi: bool = True) -> str:
    """``$a0``; ``abi=False``: ``$4``."""
    return f"${GPR_NAMES[i]}" if abi else f"${i}"


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


#: the decode steps' functions (the names of the manual's pseudocode where
#: it has one: gpr3 = decode_gpr(x, 'gpr3'), sx = sign_extend)
_FUNCS: dict[str, Callable[..., int]] = {
    "gpr3": lambda v: GPR_LISTS["gpr3"][v],
    "gpr3s": lambda v: GPR_LISTS["gpr3.src.store"][v],
    "gpr1": lambda v: GPR_LISTS["gpr1"][v],
    "gpr2a": lambda v: GPR_LISTS["gpr2.reg1"][v],
    "gpr2b": lambda v: GPR_LISTS["gpr2.reg2"][v],
    "gpr4": lambda v: GPR_LISTS["gpr4"][v],
    "gpr4z": lambda v: GPR_LISTS["gpr4.zero"][v],
    "sx": _sx,
    "neg": lambda v: -v,
    "add": lambda v, n: v + n,
    "sub": lambda v, n: v - n,
    "eight": lambda v: v or 8,                       # count3 / shift3: 0 means 8
    "andi16": lambda v: 0xFF if v == 12 else 0xFFFF if v == 13 else v,
    "li16": lambda v: -1 if v == 127 else v,
    "rt1": lambda v: 31 if v else 30,                # SAVE[16] / RESTORE.JRC[16]
    "acc": lambda v: v >> 2,                         # MFTR / MTTR: rs = 4 * ac + n
}

_STEP = re.compile(r"(\w+)=(?:(-?\d+)|(\w+)\((\w+)(?:,(-?\d+|\w+))?\)|(\w+))$")
_TOKEN = re.compile(r"([01]+)$|([a-z]\w*):(\d+)$|([a-z]\w*)\[(\d+)(?::(\d+))?\]$")


@dataclass(frozen=True)
class Row:
    """One hardware format.  ``form`` the manual's name (``LW[GP16]``),
    ``name`` its instruction (``LW``), ``size`` 2 / 4 / 6 bytes, ``mask`` /
    ``value`` the fixed bits, ``texts`` the renderings (condition,
    text, alias) in the order objdump tries them, ``nms`` whether the
    nanoMIPS subset has it, ``need`` what else the core must have
    (``eva``, ``crc``, ``ginv``, ``tlb``, ``debug``)."""

    form: str
    name: str
    size: int
    mask: int
    value: int
    template: str
    #: (argument, ((shift, width, at), ...)), "x" = the ignored bits
    fields: tuple
    #: (target, function | None, argument, second argument: a constant
    #: or an argument's name)
    steps: tuple
    dec: str
    when: Callable[[dict], bool] | None
    texts: tuple
    nms: bool = True
    need: str | None = None
    cond: str | None = None

    def extract(self, raw: int) -> dict:
        f: dict = {}
        for name, pieces in self.fields:
            v = 0
            for shift, width, at in pieces:
                v |= ((raw >> shift) & ((1 << width) - 1)) << at
            f[name] = v
        for target, fn, arg, const in self.steps:
            if fn is None:
                f[target] = f[arg] if arg is not None else const
            elif const is None:
                f[target] = _FUNCS[fn](f[arg])
            else:
                f[target] = _FUNCS[fn](f[arg], f[const] if isinstance(const, str) else const)
        return f


def _parse(template: str) -> tuple[int, int, int, tuple]:
    toks = template.split()
    widths = []
    for t in toks:
        m = _TOKEN.match(t)
        if m is None:
            raise AssertionError(f"bad token {t!r} in {template!r}")
        if m.group(1):
            widths.append(len(m.group(1)))
        elif m.group(2):
            widths.append(int(m.group(3)))
        else:
            hi, lo = int(m.group(5)), int(m.group(6) or m.group(5))
            widths.append(hi - lo + 1)
    size = sum(widths)
    pos, mask, value = size, 0, 0
    pieces: dict[str, list] = {}
    xat = 0
    for t, w in zip(toks, widths):
        pos -= w
        m = _TOKEN.match(t)
        if m.group(1):
            mask |= ((1 << w) - 1) << pos
            value |= int(m.group(1), 2) << pos
        elif m.group(2):
            pieces.setdefault(m.group(2), []).append((pos, w, 0))
        else:
            pieces.setdefault(m.group(4), []).append((pos, w, int(m.group(6) or m.group(5))))
    # the x fields, concatenated (MSB first) into one value
    if "x" in pieces:
        xs, xat = [], 0
        for shift, w, _ in reversed(pieces["x"]):
            xs.append((shift, w, xat))
            xat += w
        pieces["x"] = list(reversed(xs))
    return size, mask, value, tuple((n, tuple(p)) for n, p in pieces.items())


def _steps(dec: str) -> tuple:
    out = []
    for s in dec.split():
        m = _STEP.match(s)
        if m is None:
            raise AssertionError(f"bad decode step {s!r}")
        target, const, fn, arg, n, copy = m.groups()
        if const is not None:
            out.append((target, None, None, int(const)))
        elif fn is not None:
            if fn not in _FUNCS:
                raise AssertionError(f"unknown decode function {fn!r}")
            out.append((target, fn, arg, None if n is None else
                        n if n[0].isalpha() else int(n)))
        else:
            out.append((target, None, copy, None))
    return tuple(out)


def _R(form: str, template: str, text: str, dec: str = "", *,
       when: str | None = None,
       alias: tuple = (), name: str | None = None, nms: bool = True,
       need: str | None = None) -> Row:
    size, mask, value, fields = _parse(template)
    if size not in (16, 32, 48):
        raise AssertionError(f"{form}: template of {size} bits")
    texts = tuple((c, t, True) for c, t in alias) + ((None, text, False),)
    test = None
    if when is not None:                    # a condition on the decoded fields
        code = compile(when, form, "eval")
        test = lambda f: eval(code, {"__builtins__": {}}, f)     # noqa: E731
    return Row(form, name or form.split("[")[0].split("/")[0], size // 8, mask, value,
               template, fields, _steps(dec), dec, test, texts, nms, need, when)


#: an alias objdump always prefers
_ALWAYS = None


# ------------------------------------------------------------------ 16-bit
#
# MD01247 App. B.3 (P16) and the pools under it; ch. 3 the formats.

_ROWS16 = [
    # P16.MV / P16.RI (B.18, B.47, B.65)
    _R("MOVE", "000100 rt:5 rs:5", "move {r:rt}, {r:rs}", when="rt != 0"),
    _R("SYSCALL[16]", "000100 00000 01 0 code:2", "syscall {x:code}",
       alias=((lambda f: f["code"] == 0, "syscall"),), name="SYSCALL"),
    _R("BREAK[16]", "000100 00000 10 code:3", "break {x:code}",
       alias=((lambda f: f["code"] == 0, "break"),), name="BREAK"),
    _R("SDBBP[16]", "000100 00000 11 code:3", "sdbbp {x:code}",
       alias=((lambda f: f["code"] == 0, "sdbbp"),), name="SDBBP", need="debug"),
    _R("LW[16]", "000101 rt3:3 rs3:3 u[5:2]", "lw {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3(rt3) rs=gpr3(rs3) offset=u"),
    _R("BC[16]", "000110 s[9:1] s[10]", "bc {b:offset}", "offset=sx(s,11)"),
    # P16.SR (B.19)
    _R("SAVE[16]", "000111 rt1:1 0 u[7:4] count:4", "save {d:u}{L:list}",
       "rt=rt1(rt1) gp=0"),
    _R("RESTORE.JRC[16]", "000111 rt1:1 1 u[7:4] count:4", "restore.jrc {d:u}{L:list}",
       "rt=rt1(rt1) gp=0 jr=1", name="RESTORE"),
    # P16.SHIFT (B.20)
    _R("SLL[16]", "001100 rt3:3 rs3:3 0 shift3:3", "sll {r:rt}, {r:rs}, {d:shift}",
       "rt=gpr3(rt3) rs=gpr3(rs3) shift=eight(shift3)"),
    _R("SRL[16]", "001100 rt3:3 rs3:3 1 shift3:3", "srl {r:rt}, {r:rs}, {d:shift}",
       "rt=gpr3(rt3) rs=gpr3(rs3) shift=eight(shift3)"),
    _R("LW[SP]", "001101 rt:5 u[6:2]", "lw {r:rt}, {d:offset}({r:rs})", "rs=29 offset=u"),
    _R("BALC[16]", "001110 s[9:1] s[10]", "balc {b:offset}", "offset=sx(s,11)"),
    # P16.4X4 (B.21)
    _R("ADDU[4X4]", "001111 rt4[3] 0 rt4[2:0] rs4[3] 0 rs4[2:0]",
       "addu {r:dst}, {r:src1}, {r:src2}",
       "dst=gpr4(rt4) src1=gpr4(rt4) src2=gpr4(rs4)", nms=False),
    _R("MUL[4X4]", "001111 rt4[3] 0 rt4[2:0] rs4[3] 1 rs4[2:0]",
       "mul {r:dst}, {r:src1}, {r:src2}",
       "dst=gpr4(rt4) src1=gpr4(rt4) src2=gpr4(rs4)", nms=False),
    # P16C / POOL16C_0 / POOL16C_00 (B.22, B.48, B.66)
    _R("NOT[16]", "010100 rt3:3 rs3:3 00 0 0", "not {r:rt}, {r:rs}",
       "rt=gpr3(rt3) rs=gpr3(rs3)", name="NOT"),
    _R("XOR[16]", "010100 rt3:3 rs3:3 01 0 0", "xor {r:rt}, {r:rs}",
       "rt=gpr3(rt3) rs=gpr3(rs3) rd=rt",
       alias=((_ALWAYS, "xor {r:rd}, {r:rt}, {r:rs}"),)),
    _R("AND[16]", "010100 rt3:3 rs3:3 10 0 0", "and {r:rt}, {r:rs}",
       "rt=gpr3(rt3) rs=gpr3(rs3) rd=rt",
       alias=((_ALWAYS, "and {r:rd}, {r:rt}, {r:rs}"),)),
    _R("OR[16]", "010100 rt3:3 rs3:3 11 0 0", "or {r:rt}, {r:rs}",
       "rt=gpr3(rt3) rs=gpr3(rs3) rd=rt",
       alias=((_ALWAYS, "or {r:rd}, {r:rt}, {r:rs}"),)),
    _R("LWXS[16]", "010100 rt3:3 rs3:3 rd3:3 1", "lwxs {r:rd}, {r:rs}({r:rt})",
       "rd=gpr3(rd3) rs=gpr3(rs3) rt=gpr3(rt3)"),
    _R("LW[GP16]", "010101 rt3:3 u[8:2]", "lw {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3(rt3) rs=28 offset=u"),
    # P16.LB (B.23)
    _R("LB[16]", "010111 rt3:3 rs3:3 00 u:2", "lb {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3(rt3) rs=gpr3(rs3) offset=u"),
    _R("SB[16]", "010111 rtz3:3 rs3:3 01 u:2", "sb {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3s(rtz3) rs=gpr3(rs3) offset=u"),
    _R("LBU[16]", "010111 rt3:3 rs3:3 10 u:2", "lbu {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3(rt3) rs=gpr3(rs3) offset=u"),
    # P16.A1 (B.24)
    _R("ADDIU[R1.SP]", "011100 rt3:3 1 u[7:2]", "addiu {r:rt}, {r:rs}, {d:imm}",
       "rt=gpr3(rt3) rs=29 imm=u"),
    _R("LW[4X4]", "011101 rt4[3] u[2] rt4[2:0] rs4[3] u[3] rs4[2:0]",
       "lw {r:rt}, {d:offset}({r:rs})", "rt=gpr4(rt4) rs=gpr4(rs4) offset=u", nms=False),
    # P16.LH (B.25)
    _R("LH[16]", "011111 rt3:3 rs3:3 0 u[2:1] 0", "lh {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3(rt3) rs=gpr3(rs3) offset=u"),
    _R("SH[16]", "011111 rtz3:3 rs3:3 0 u[2:1] 1", "sh {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3s(rtz3) rs=gpr3(rs3) offset=u"),
    _R("LHU[16]", "011111 rt3:3 rs3:3 1 u[2:1] 0", "lhu {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3(rt3) rs=gpr3(rs3) offset=u"),
    # P16.A2 / P.ADDIU[RS5] (B.26, B.49)
    _R("ADDIU[R2]", "100100 rt3:3 rs3:3 0 u[4:2]", "addiu {r:rt}, {r:rs}, {d:imm}",
       "rt=gpr3(rt3) rs=gpr3(rs3) imm=u"),
    _R("NOP[16]", "100100 00000 x:1 1 x:3", "nop", name="NOP"),
    _R("ADDIU[RS5]", "100100 rt:5 s[3] 1 s[2:0]", "addiu {r:rt}, {d:imm}",
       "rs=rt imm=sx(s,4)", when="rt != 0",
       alias=((_ALWAYS, "addiu {r:rt}, {r:rs}, {d:imm}"),)),
    _R("SW[16]", "100101 rtz3:3 rs3:3 u[5:2]", "sw {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3s(rtz3) rs=gpr3(rs3) offset=u"),
    _R("BEQZC[16]", "100110 rt3:3 s[6:1] s[7]", "beqzc {r:rt}, {b:offset}",
       "rt=gpr3(rt3) offset=sx(s,8)"),
    # P16.ADDU (B.27)
    _R("ADDU[16]", "101100 rt3:3 rs3:3 rd3:3 0", "addu {r:dst}, {r:src1}, {r:src2}",
       "dst=gpr3(rd3) src1=gpr3(rs3) src2=gpr3(rt3)"),
    _R("SUBU[16]", "101100 rt3:3 rs3:3 rd3:3 1", "subu {r:rd}, {r:rs}, {r:rt}",
       "rd=gpr3(rd3) rs=gpr3(rs3) rt=gpr3(rt3)"),
    _R("SW[SP]", "101101 rt:5 u[6:2]", "sw {r:rt}, {d:offset}({r:rs})", "rs=29 offset=u"),
    _R("BNEZC[16]", "101110 rt3:3 s[6:1] s[7]", "bnezc {r:rt}, {b:offset}",
       "rt=gpr3(rt3) offset=sx(s,8)"),
    _R("MOVEP", "101111 rtz4[3] rd2[0] rtz4[2:0] rsz4[3] rd2[1] rsz4[2:0]",
       "movep {r:dst1}, {r:dst2}, {r:src1}, {r:src2}",
       "dst1=gpr2a(rd2) dst2=gpr2b(rd2) src1=gpr4z(rsz4) src2=gpr4z(rtz4)", nms=False),
    _R("LI[16]", "110100 rt3:3 eu:7", "li {r:rt}, {d:s}", "rt=gpr3(rt3) s=li16(eu)"),
    _R("SW[GP16]", "110101 rtz3:3 u[8:2]", "sw {r:rt}, {d:offset}({r:rs})",
       "rt=gpr3s(rtz3) rs=28 offset=u"),
    # P16.BR / P16.JRC / P16.BR1 (B.28, B.50, B.51)
    _R("JRC", "110110 rt:5 0 0000", "jrc {r:rt}"),
    _R("JALRC[16]", "110110 rt:5 1 0000", "jalrc {r:src}", "src=rt dst=31", name="JALRC"),
    _R("BEQC[16]", "110110 rt3:3 rs3:3 u[4:1]", "beqc {r:rs}, {r:rt}, {b:offset}",
       "rs=gpr3(rs3) rt=gpr3(rt3) offset=u",
       when="rs3 < rt3 and u != 0", nms=False),
    _R("BNEC[16]", "110110 rt3:3 rs3:3 u[4:1]", "bnec {r:rs}, {r:rt}, {b:offset}",
       "rs=gpr3(rs3) rt=gpr3(rt3) offset=u",
       when="rs3 >= rt3 and u != 0", nms=False),
    _R("ANDI[16]", "111100 rt3:3 rs3:3 eu:4", "andi {r:rt}, {r:rs}, {x:u}",
       "rt=gpr3(rt3) rs=gpr3(rs3) u=andi16(eu)"),
    _R("SW[4X4]", "111101 rtz4[3] u[2] rtz4[2:0] rs4[3] u[3] rs4[2:0]",
       "sw {r:rt}, {d:offset}({r:rs})", "rt=gpr4z(rtz4) rs=gpr4(rs4) offset=u", nms=False),
    _R("MOVEP[REV]", "111111 rt4[3] rd2[0] rt4[2:0] rs4[3] rd2[1] rs4[2:0]",
       "movep {r:dst1}, {r:dst2}, {r:src1}, {r:src2}",
       "dst1=gpr4(rs4) dst2=gpr4(rt4) src1=gpr2a(rd2) src2=gpr2b(rd2)", name="MOVEP",
       nms=False),
]


# ------------------------------------------------------------------ 32-bit
#
# MD01247 App. B.2 (P32) and the pools under it.

_RRR = "001000 rt:5 rs:5 rd:5 x:1"          # _POOL32A0's three-register rows

_ROWS32 = [
    # P.ADDIU / P.RI / P.SYSCALL (B.4, B.29, B.52)
    _R("SIGRIE", "000000 00000 00 code:19", "sigrie {x:code}"),
    _R("SYSCALL[32]", "000000 00000 01 0 code:18", "syscall {x:code}",
       alias=((lambda f: f["code"] == 0, "syscall"),), name="SYSCALL"),
    _R("BREAK[32]", "000000 00000 10 code:19", "break {x:code}",
       alias=((lambda f: f["code"] == 0, "break"),), name="BREAK"),
    _R("SDBBP[32]", "000000 00000 11 code:19", "sdbbp {x:code}",
       alias=((lambda f: f["code"] == 0, "sdbbp"),), name="SDBBP", need="debug"),
    _R("ADDIU[32]", "000000 rt:5 rs:5 u:16", "addiu {r:rt}, {r:rs}, {d:imm}", "imm=u",
       when="rt != 0",
       alias=((lambda f: f["rs"] == 0, "li {r:rt}, {d:imm}"),)),
    _R("ADDIUPC[32]", "000001 rt:5 s[20:1] s[21]", "addiupc {r:rt}, {d:imm}",
       "s=sx(s,22) imm=add(s,4)", alias=((_ALWAYS, "lapc {r:rt}, {b:s}"),), name="ADDIUPC"),
    _R("MOVE.BALC", "000010 rtz4[3] rd1:1 rtz4[2:0] s[20:1] s[21]",
       "move.balc {r:rd}, {r:rt}, {b:offset}",
       "rd=gpr1(rd1) rt=gpr4z(rtz4) offset=sx(s,22)", nms=False),
    # P.BAL (B.6)
    _R("BC[32]", "001010 0 s[24:1] s[25]", "bc {b:offset}", "offset=sx(s,26)"),
    _R("BALC[32]", "001010 1 s[24:1] s[25]", "balc {b:offset}", "offset=sx(s,26)"),
    # P.GP.W (B.7)
    _R("ADDIU[GP.W]", "010000 rt:5 u[20:2] 00", "addiu {r:rt}, {r:rs}, {d:imm}",
       "imm=u rs=28"),
    _R("LW[GP]", "010000 rt:5 u[20:2] 10", "lw {r:rt}, {d:offset}({r:rs})", "rs=28 offset=u"),
    _R("SW[GP]", "010000 rt:5 u[20:2] 11", "sw {r:rt}, {d:offset}({r:rs})", "rs=28 offset=u"),
    # P.GP.BH / P.GP.LH / P.GP.SH (B.8, B.32, B.33)
    _R("LB[GP]", "010001 rt:5 000 u:18", "lb {r:rt}, {d:offset}({r:rs})", "rs=28 offset=u"),
    _R("SB[GP]", "010001 rt:5 001 u:18", "sb {r:rt}, {d:offset}({r:rs})", "rs=28 offset=u"),
    _R("LBU[GP]", "010001 rt:5 010 u:18", "lbu {r:rt}, {d:offset}({r:rs})", "rs=28 offset=u"),
    _R("ADDIU[GP.B]", "010001 rt:5 011 u:18", "addiu {r:rt}, {r:rs}, {d:imm}", "imm=u rs=28"),
    _R("LH[GP]", "010001 rt:5 100 u[17:1] 0", "lh {r:rt}, {d:offset}({r:rs})", "rs=28 offset=u"),
    _R("LHU[GP]", "010001 rt:5 100 u[17:1] 1", "lhu {r:rt}, {d:offset}({r:rs})",
       "rs=28 offset=u"),
    _R("SH[GP]", "010001 rt:5 101 u[17:1] 0", "sh {r:rt}, {d:offset}({r:rs})", "rs=28 offset=u"),
    # P.J / P.BALRSC (B.9, B.34)
    _R("JALRC[32]", "010010 rt:5 rs:5 0000 x:12", "jalrc {r:dst}, {r:src}", "src=rs dst=rt",
       name="JALRC",
       alias=((lambda f: f["rt"] == 0 and f["x"] == 0, "jrc {r:src}"),
              (lambda f: f["rt"] == 31 and f["x"] == 0, "jalrc {r:src}"))),
    _R("JALRC.HB", "010010 rt:5 rs:5 0001 x:12", "jalrc.hb {r:rt}, {r:rs}",
       alias=((lambda f: f["rt"] == 0 and f["x"] == 0, "jrc.hb {r:rs}"),
              (lambda f: f["rt"] == 31 and f["x"] == 0, "jalrc.hb {r:rs}"))),
    _R("BRSC", "010010 00000 rs:5 1000 x:12", "brsc {r:rs}"),
    _R("BALRSC", "010010 rt:5 rs:5 1000 x:12", "balrsc {r:rt}, {r:rs}",
       when="rt != 0"),
    # P.U12 (B.11)
    _R("ORI", "100000 rt:5 rs:5 0000 u:12", "ori {r:rt}, {r:rs}, {x:u}"),
    _R("XORI", "100000 rt:5 rs:5 0001 u:12", "xori {r:rt}, {r:rs}, {x:u}"),
    _R("ANDI[32]", "100000 rt:5 rs:5 0010 u:12", "andi {r:rt}, {r:rs}, {x:u}"),
    _R("SLTI", "100000 rt:5 rs:5 0100 u:12", "slti {r:rt}, {r:rs}, {d:u}"),
    _R("SLTIU", "100000 rt:5 rs:5 0101 u:12", "sltiu {r:rt}, {r:rs}, {d:u}"),
    _R("SEQI", "100000 rt:5 rs:5 0110 u:12", "seqi {r:rt}, {r:rs}, {d:u}"),
    _R("ADDIU[NEG]", "100000 rt:5 rs:5 1000 u:12", "addiu {r:rt}, {r:rs}, {n:u}", "imm=neg(u)",
       alias=((lambda f: f["rs"] == 0, "li {r:rt}, {n:u}"),)),
    # P.SR / PP.SR (B.35, B.57)
    _R("SAVE[32]", "100000 rt:5 0 count:4 0011 u[11:3] gp:1 00", "save {d:u}{L:list}"),
    _R("RESTORE[32]", "100000 rt:5 0 count:4 0011 u[11:3] gp:1 10", "restore {d:u}{L:list}",
       "jr=0", name="RESTORE"),
    _R("RESTORE.JRC[32]", "100000 rt:5 0 count:4 0011 u[11:3] gp:1 11",
       "restore.jrc {d:u}{L:list}", "jr=1", name="RESTORE"),
    # P.SHIFT / P.SLL (B.36, B.58)
    _R("NOP[32]", "100000 00000 x:5 1100 x:3 0000 00000", "nop", name="NOP"),
    _R("EHB", "100000 00000 x:5 1100 x:3 0000 00011", "ehb"),
    _R("PAUSE", "100000 00000 x:5 1100 x:3 0000 00101", "pause"),
    _R("SYNC", "100000 00000 stype:5 1100 x:3 0000 00110", "sync {x:stype}",
       alias=((lambda f: f["x"] == 0 and f["stype"] == 0x04, "sync_wmb"),
              (lambda f: f["x"] == 0 and f["stype"] == 0x10, "sync_mb"),
              (lambda f: f["x"] == 0 and f["stype"] == 0x11, "sync_acquire"),
              (lambda f: f["x"] == 0 and f["stype"] == 0x12, "sync_release"),
              (lambda f: f["x"] == 0 and f["stype"] == 0x13, "sync_rmb"),
              (lambda f: f["x"] == 0 and f["stype"] == 0x14, "sync_ginv"),
              (lambda f: f["x"] == 0 and f["stype"] == 0, "sync"))),
    _R("SLL[32]", "100000 rt:5 rs:5 1100 x:3 0000 shift:5", "sll {r:rt}, {r:rs}, {d:shift}",
       name="SLL"),
    _R("SRL[32]", "100000 rt:5 rs:5 1100 x:3 0010 shift:5", "srl {r:rt}, {r:rs}, {d:shift}",
       name="SRL"),
    _R("SRA", "100000 rt:5 rs:5 1100 x:3 0100 shift:5", "sra {r:rt}, {r:rs}, {d:shift}"),
    _R("ROTR", "100000 rt:5 rs:5 1100 x:3 0110 shift:5", "rotr {r:rt}, {r:rs}, {d:shift}"),
    # P.ROTX / P.INS / P.EXT (B.37-B.39)
    _R("ROTX", "100000 rt:5 rs:5 1101 0 shiftx[4:1] stripe:1 0 shift:5",
       "rotx {r:rt}, {r:rs}, {d:shift}, {d:shiftx}, {d:stripe}", nms=False,
       alias=((lambda f: (f["shift"], f["shiftx"], f["stripe"]) == (7, 8, 1),
               "bitrevb {r:rt}, {r:rs}"),
              (lambda f: (f["shift"], f["shiftx"], f["stripe"]) == (15, 16, 0),
               "bitrevh {r:rt}, {r:rs}"),
              (lambda f: (f["shift"], f["shiftx"], f["stripe"]) == (31, 0, 0),
               "bitrevw {r:rt}, {r:rs}"),
              (lambda f: (f["shift"], f["shiftx"], f["stripe"]) == (8, 24, 0),
               "byterevh {r:rt}, {r:rs}"),
              (lambda f: (f["shift"], f["shiftx"], f["stripe"]) == (24, 8, 0),
               "byterevw {r:rt}, {r:rs}"),
              (lambda f: f["stripe"] == 0, "rotx {r:rt}, {r:rs}, {d:shift}, {d:shiftx}"))),
    _R("INS", "100000 rt:5 rs:5 1110 0 msbd:5 0 lsb:5", "ins {r:rt}, {r:rs}, {d:pos}, {d:size}",
       "pos=lsb size=sub(msbd,lsb) size=add(size,1)", nms=False),
    _R("EXT", "100000 rt:5 rs:5 1111 0 msbd:5 0 lsb:5", "ext {r:rt}, {r:rs}, {d:pos}, {d:size}",
       "pos=lsb size=add(msbd,1)", nms=False),
    # P.LS.U12 / P.PREF[U12] (B.12, B.40)
    _R("LB[U12]", "100001 rt:5 rs:5 0000 u:12", "lb {r:rt}, {d:offset}({r:rs})", "offset=u"),
    _R("SB[U12]", "100001 rt:5 rs:5 0001 u:12", "sb {r:rt}, {d:offset}({r:rs})", "offset=u"),
    _R("LBU[U12]", "100001 rt:5 rs:5 0010 u:12", "lbu {r:rt}, {d:offset}({r:rs})", "offset=u"),
    _R("SYNCI[U12]", "100001 11111 rs:5 0011 u:12", "synci {d:offset}({r:rs})",
       "offset=u is_eva=0", name="SYNCI"),
    _R("PREF[U12]", "100001 hint:5 rs:5 0011 u:12", "pref {x:hint}, {d:offset}({r:rs})",
       "offset=u is_eva=0", when="hint != 31", name="PREF"),
    _R("LH[U12]", "100001 rt:5 rs:5 0100 u:12", "lh {r:rt}, {d:offset}({r:rs})", "offset=u"),
    _R("SH[U12]", "100001 rt:5 rs:5 0101 u:12", "sh {r:rt}, {d:offset}({r:rs})", "offset=u"),
    _R("LHU[U12]", "100001 rt:5 rs:5 0110 u:12", "lhu {r:rt}, {d:offset}({r:rs})", "offset=u"),
    _R("LW[U12]", "100001 rt:5 rs:5 1000 u:12", "lw {r:rt}, {d:offset}({r:rs})", "offset=u"),
    _R("SW[U12]", "100001 rt:5 rs:5 1001 u:12", "sw {r:rt}, {d:offset}({r:rs})", "offset=u"),
    # P.BR1 (B.13)
    _R("BEQC[32]", "100010 rt:5 rs:5 00 s[13:1] s[14]", "beqc {r:rs}, {r:rt}, {b:offset}",
       "offset=sx(s,15)",
       alias=((lambda f: f["rs"] == 0, "beqzc {r:rt}, {b:offset}"),
              (lambda f: f["rt"] == 0, "beqzc {r:rs}, {b:offset}"))),
    _R("BGEC", "100010 rt:5 rs:5 10 s[13:1] s[14]", "bgec {r:rs}, {r:rt}, {b:offset}",
       "offset=sx(s,15)",
       alias=((lambda f: f["rs"] == 0, "blezc {r:rt}, {b:offset}"),
              (lambda f: f["rt"] == 0, "bgezc {r:rs}, {b:offset}"))),
    _R("BGEUC", "100010 rt:5 rs:5 11 s[13:1] s[14]", "bgeuc {r:rs}, {r:rt}, {b:offset}",
       "offset=sx(s,15)"),
    # P.LS.S9 / P.LS.S0 / P.PREF[S9] (B.14, B.42, B.59)
    _R("LB[S9]", "101001 rt:5 rs:5 s[8] 0000 0 00 s[7:0]", "lb {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)"),
    _R("SB[S9]", "101001 rt:5 rs:5 s[8] 0001 0 00 s[7:0]", "sb {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)"),
    _R("LBU[S9]", "101001 rt:5 rs:5 s[8] 0010 0 00 s[7:0]", "lbu {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)"),
    _R("SYNCI[S9]", "101001 11111 rs:5 s[8] 0011 0 00 s[7:0]", "synci {d:offset}({r:rs})",
       "offset=sx(s,9) is_eva=0", name="SYNCI"),
    _R("PREF[S9]", "101001 hint:5 rs:5 s[8] 0011 0 00 s[7:0]",
       "pref {x:hint}, {d:offset}({r:rs})", "offset=sx(s,9) is_eva=0",
       when="hint != 31", name="PREF"),
    _R("LH[S9]", "101001 rt:5 rs:5 s[8] 0100 0 00 s[7:0]", "lh {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)"),
    _R("SH[S9]", "101001 rt:5 rs:5 s[8] 0101 0 00 s[7:0]", "sh {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)"),
    _R("LHU[S9]", "101001 rt:5 rs:5 s[8] 0110 0 00 s[7:0]", "lhu {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)"),
    _R("LW[S9]", "101001 rt:5 rs:5 s[8] 1000 0 00 s[7:0]", "lw {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)"),
    _R("SW[S9]", "101001 rt:5 rs:5 s[8] 1001 0 00 s[7:0]", "sw {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)"),
    # P.LS.S1 / P.LL / P.SC (B.43, B.60, B.61)
    _R("UALH", "101001 rt:5 rs:5 s[8] 0100 0 01 s[7:0]", "ualh {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", nms=False),
    _R("UASH", "101001 rt:5 rs:5 s[8] 0101 0 01 s[7:0]", "uash {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", nms=False),
    _R("CACHE", "101001 op:5 rs:5 s[8] 0111 0 01 s[7:0]", "cache {x:op}, {d:offset}({r:rs})",
       "offset=sx(s,9) is_eva=0"),
    _R("LL", "101001 rt:5 rs:5 s[8] 1010 0 01 s[7:2] 00", "ll {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9) nbytes=4 is_eva=0"),
    _R("LLWP", "101001 rt:5 rs:5 x:1 1010 0 01 ru:5 x:1 01", "llwp {r:rt}, {r:ru}, ({r:rs})",
       "offset=0 nbytes=8 is_eva=0", name="LL"),
    _R("SC", "101001 rt:5 rs:5 s[8] 1011 0 01 s[7:2] 00", "sc {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9) nbytes=4 is_eva=0"),
    _R("SCWP", "101001 rt:5 rs:5 x:1 1011 0 01 ru:5 x:1 01", "scwp {r:rt}, {r:ru}, ({r:rs})",
       "offset=0 nbytes=8 is_eva=0", name="SC"),
    # P.LS.E0 / P.PREFE / P.LLE / P.SCE (B.44, B.62-B.64): EVA, kernel only
    _R("LBE", "101001 rt:5 rs:5 s[8] 0000 0 10 s[7:0]", "lbe {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", need="eva"),
    _R("SBE", "101001 rt:5 rs:5 s[8] 0001 0 10 s[7:0]", "sbe {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", need="eva"),
    _R("LBUE", "101001 rt:5 rs:5 s[8] 0010 0 10 s[7:0]", "lbue {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", need="eva"),
    _R("SYNCIE", "101001 11111 rs:5 s[8] 0011 0 10 s[7:0]", "syncie {d:offset}({r:rs})",
       "offset=sx(s,9) is_eva=1", name="SYNCI", need="eva"),
    _R("PREFE", "101001 hint:5 rs:5 s[8] 0011 0 10 s[7:0]", "prefe {x:hint}, {d:offset}({r:rs})",
       "offset=sx(s,9) is_eva=1", when="hint != 31", name="PREF", need="eva"),
    _R("LHE", "101001 rt:5 rs:5 s[8] 0100 0 10 s[7:0]", "lhe {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", need="eva"),
    _R("SHE", "101001 rt:5 rs:5 s[8] 0101 0 10 s[7:0]", "she {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", need="eva"),
    _R("LHUE", "101001 rt:5 rs:5 s[8] 0110 0 10 s[7:0]", "lhue {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", need="eva"),
    _R("CACHEE", "101001 op:5 rs:5 s[8] 0111 0 10 s[7:0]", "cachee {x:op}, {d:offset}({r:rs})",
       "offset=sx(s,9) is_eva=1", name="CACHE", need="eva"),
    _R("LWE", "101001 rt:5 rs:5 s[8] 1000 0 10 s[7:0]", "lwe {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", need="eva"),
    _R("SWE", "101001 rt:5 rs:5 s[8] 1001 0 10 s[7:0]", "swe {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9)", need="eva"),
    _R("LLE", "101001 rt:5 rs:5 s[8] 1010 0 10 s[7:2] 00", "lle {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9) nbytes=4 is_eva=1", name="LL", need="eva"),
    _R("LLWPE", "101001 rt:5 rs:5 x:1 1010 0 10 ru:5 x:1 01", "llwpe {r:rt}, {r:ru}, ({r:rs})",
       "offset=0 nbytes=8 is_eva=1", name="LL", need="eva"),
    _R("SCE", "101001 rt:5 rs:5 s[8] 1011 0 10 s[7:2] 00", "sce {r:rt}, {d:offset}({r:rs})",
       "offset=sx(s,9) nbytes=4 is_eva=1", name="SC", need="eva"),
    _R("SCWPE", "101001 rt:5 rs:5 x:1 1011 0 10 ru:5 x:1 01", "scwpe {r:rt}, {r:ru}, ({r:rs})",
       "offset=0 nbytes=8 is_eva=1", name="SC", need="eva"),
    # P.LS.WM / P.LS.UAWM (B.45, B.46)
    _R("LWM", "101001 rt:5 rs:5 s[8] count3:3 0 1 00 s[7:0]",
       "lwm {r:rt}, {d:offset}({r:rs}), {d:count}", "offset=sx(s,9) count=eight(count3)",
       nms=False),
    _R("SWM", "101001 rt:5 rs:5 s[8] count3:3 1 1 00 s[7:0]",
       "swm {r:rt}, {d:offset}({r:rs}), {d:count}", "offset=sx(s,9) count=eight(count3)",
       nms=False),
    _R("UALWM", "101001 rt:5 rs:5 s[8] count3:3 0 1 01 s[7:0]",
       "ualwm {r:rt}, {d:offset}({r:rs}), {d:count}", "offset=sx(s,9) count=eight(count3)",
       nms=False, alias=((lambda f: f["count"] == 1, "ualw {r:rt}, {d:offset}({r:rs})"),)),
    _R("UASWM", "101001 rt:5 rs:5 s[8] count3:3 1 1 01 s[7:0]",
       "uaswm {r:rt}, {d:offset}({r:rs}), {d:count}", "offset=sx(s,9) count=eight(count3)",
       nms=False, alias=((lambda f: f["count"] == 1, "uasw {r:rt}, {d:offset}({r:rs})"),)),
    # P.BR2 (B.15)
    _R("BNEC[32]", "101010 rt:5 rs:5 00 s[13:1] s[14]", "bnec {r:rs}, {r:rt}, {b:offset}",
       "offset=sx(s,15)",
       alias=((lambda f: f["rs"] == 0, "bnezc {r:rt}, {b:offset}"),
              (lambda f: f["rt"] == 0, "bnezc {r:rs}, {b:offset}"))),
    _R("BLTC", "101010 rt:5 rs:5 10 s[13:1] s[14]", "bltc {r:rs}, {r:rt}, {b:offset}",
       "offset=sx(s,15)",
       alias=((lambda f: f["rs"] == 0, "bgtzc {r:rt}, {b:offset}"),
              (lambda f: f["rt"] == 0, "bltzc {r:rs}, {b:offset}"))),
    _R("BLTUC", "101010 rt:5 rs:5 11 s[13:1] s[14]", "bltuc {r:rs}, {r:rt}, {b:offset}",
       "offset=sx(s,15)"),
    # P.BRI (B.16): the offset is computed at the top of each Operation
    _R("BEQIC", "110010 rt:5 000 u:7 s[10:1] s[11]", "beqic {r:rt}, {d:u}, {b:offset}",
       "offset=sx(s,12)"),
    _R("BBEQZC", "110010 rt:5 001 x:1 bit:6 s[10:1] s[11]", "bbeqzc {r:rt}, {d:bit}, {b:offset}",
       "offset=sx(s,12)", nms=False),
    _R("BGEIC", "110010 rt:5 010 u:7 s[10:1] s[11]", "bgeic {r:rt}, {d:u}, {b:offset}",
       "offset=sx(s,12)"),
    _R("BGEIUC", "110010 rt:5 011 u:7 s[10:1] s[11]", "bgeiuc {r:rt}, {d:u}, {b:offset}",
       "offset=sx(s,12)"),
    _R("BNEIC", "110010 rt:5 100 u:7 s[10:1] s[11]", "bneic {r:rt}, {d:u}, {b:offset}",
       "offset=sx(s,12)"),
    _R("BBNEZC", "110010 rt:5 101 x:1 bit:6 s[10:1] s[11]", "bbnezc {r:rt}, {d:bit}, {b:offset}",
       "offset=sx(s,12)", nms=False),
    _R("BLTIC", "110010 rt:5 110 u:7 s[10:1] s[11]", "bltic {r:rt}, {d:u}, {b:offset}",
       "offset=sx(s,12)"),
    _R("BLTIUC", "110010 rt:5 111 u:7 s[10:1] s[11]", "bltiuc {r:rt}, {d:u}, {b:offset}",
       "offset=sx(s,12)"),
    # P.LUI (B.17)
    _R("LUI", "111000 rt:5 s[20:12] s[30:21] 0 s[31]", "lui {r:rt}, {h:imm}", "imm=sx(s,32)"),
    _R("ALUIPC", "111000 rt:5 s[20:12] s[30:21] 1 s[31]", "aluipc {r:rt}, {p:offset}",
       "offset=sx(s,32)"),
    # _POOL32A0_0 (B.53) and its pools P.TRAP / P.CMOVE / P.SLTU / P.DVP (B.67-B.69, B.75)
    _R("TEQ", "001000 rt:5 rs:5 code:5 0 0000000 000", "teq {r:rs}, {r:rt}, {x:code}",
       nms=False, alias=((lambda f: f["code"] == 0, "teq {r:rs}, {r:rt}"),)),
    _R("TNE", "001000 rt:5 rs:5 code:5 1 0000000 000", "tne {r:rs}, {r:rt}, {x:code}",
       nms=False, alias=((lambda f: f["code"] == 0, "tne {r:rs}, {r:rt}"),)),
    _R("SEB", "001000 rt:5 rs:5 x:6 0000001 000", "seb {r:rt}, {r:rs}", nms=False),
    _R("SLLV", _RRR + " 0000010 000", "sllv {r:rd}, {r:rs}, {r:rt}"),
    _R("MUL[32]", _RRR + " 0000011 000", "mul {r:dst}, {r:src1}, {r:src2}",
       "dst=rd src1=rs src2=rt", name="MUL"),
    _R("SEH", "001000 rt:5 rs:5 x:6 0001001 000", "seh {r:rt}, {r:rs}"),
    _R("SRLV", _RRR + " 0001010 000", "srlv {r:rd}, {r:rs}, {r:rt}"),
    _R("MUH", _RRR + " 0001011 000", "muh {r:rd}, {r:rs}, {r:rt}"),
    _R("SRAV", _RRR + " 0010010 000", "srav {r:rd}, {r:rs}, {r:rt}"),
    _R("MULU", _RRR + " 0010011 000", "mulu {r:rd}, {r:rs}, {r:rt}"),
    _R("ROTRV", _RRR + " 0011010 000", "rotrv {r:rd}, {r:rs}, {r:rt}"),
    _R("MUHU", _RRR + " 0011011 000", "muhu {r:rd}, {r:rs}, {r:rt}"),
    _R("ADD", _RRR + " 0100010 000", "add {r:rd}, {r:rs}, {r:rt}", nms=False),
    _R("DIV", _RRR + " 0100011 000", "div {r:rd}, {r:rs}, {r:rt}"),
    _R("ADDU[32]", _RRR + " 0101010 000", "addu {r:dst}, {r:src1}, {r:src2}",
       "dst=rd src1=rs src2=rt", name="ADDU",
       alias=((lambda f: f["rt"] == 0 and f["x"] == 0, "move {r:rd}, {r:rs}"),)),
    _R("MOD", _RRR + " 0101011 000", "mod {r:rd}, {r:rs}, {r:rt}"),
    _R("SUB", _RRR + " 0110010 000", "sub {r:rd}, {r:rs}, {r:rt}", nms=False,
       alias=((lambda f: f["rs"] == 0 and f["x"] == 0, "neg {r:rd}, {r:rt}"),)),
    _R("DIVU", _RRR + " 0110011 000", "divu {r:rd}, {r:rs}, {r:rt}"),
    _R("RDHWR", "001000 rt:5 hs:5 sel:5 x:1 0111000 000", "rdhwr {r:rt}, {c:hs}, {d:sel}",
       nms=False, alias=((lambda f: f["x"] == 0 and f["sel"] == 0, "rdhwr {r:rt}, {c:hs}"),
                         (lambda f: f["x"] == 0, "rdhwr {r:rt}, {c:hs}, {d:sel}"))),
    _R("SUBU[32]", _RRR + " 0111010 000", "subu {r:rd}, {r:rs}, {r:rt}", name="SUBU",
       alias=((lambda f: f["rs"] == 0 and f["x"] == 0, "negu {r:rd}, {r:rt}"),)),
    _R("MODU", _RRR + " 0111011 000", "modu {r:rd}, {r:rs}, {r:rt}"),
    _R("MOVZ", "001000 rt:5 rs:5 rd:5 0 1000010 000", "movz {r:rd}, {r:rs}, {r:rt}"),
    _R("MOVN", "001000 rt:5 rs:5 rd:5 1 1000010 000", "movn {r:rd}, {r:rs}, {r:rt}"),
    _R("AND[32]", _RRR + " 1001010 000", "and {r:rd}, {r:rs}, {r:rt}", name="AND"),
    _R("OR[32]", _RRR + " 1010010 000", "or {r:rd}, {r:rs}, {r:rt}", name="OR",
       alias=((lambda f: f["rt"] == 0 and f["x"] == 0, "move {r:rd}, {r:rs}"),)),
    _R("NOR", _RRR + " 1011010 000", "nor {r:rd}, {r:rs}, {r:rt}",
       alias=((lambda f: f["rt"] == 0 and f["x"] == 0, "not {r:rd}, {r:rs}"),)),
    _R("XOR[32]", _RRR + " 1100010 000", "xor {r:rd}, {r:rs}, {r:rt}", name="XOR"),
    _R("SLT", _RRR + " 1101010 000", "slt {r:rd}, {r:rs}, {r:rt}"),
    _R("DVP", "001000 rt:5 x:5 00000 0 1110010 000", "dvp {r:rt}",
       alias=((lambda f: f["rt"] == 0, "dvp"),)),
    _R("EVP", "001000 rt:5 x:5 00000 1 1110010 000", "evp {r:rt}",
       alias=((lambda f: f["rt"] == 0, "evp"),)),
    _R("SLTU", _RRR + " 1110010 000", "sltu {r:rd}, {r:rs}, {r:rt}",
       when="rd != 0"),
    _R("SOV", _RRR + " 1111010 000", "sov {r:rd}, {r:rs}, {r:rt}"),
    # _POOL32A0_1 (B.54) / CRC32 (B.70)
    _R("MFC0", "001000 rt:5 c0s:5 sel:5 x:1 0000110 000", "mfc0 {r:rt}, {c:c0s}, {d:sel}",
       alias=((lambda f: f["sel"] == 0, "mfc0 {r:rt}, {c:c0s}"),)),
    _R("MFHC0", "001000 rt:5 c0s:5 sel:5 x:1 0000111 000", "mfhc0 {r:rt}, {c:c0s}, {d:sel}",
       alias=((lambda f: f["sel"] == 0, "mfhc0 {r:rt}, {c:c0s}"),)),
    _R("MTC0", "001000 rt:5 c0s:5 sel:5 x:1 0001110 000", "mtc0 {r:rt}, {c:c0s}, {d:sel}",
       alias=((lambda f: f["sel"] == 0, "mtc0 {r:rt}, {c:c0s}"),)),
    _R("MTHC0", "001000 rt:5 c0s:5 sel:5 x:1 0001111 000", "mthc0 {r:rt}, {c:c0s}, {d:sel}",
       alias=((lambda f: f["sel"] == 0, "mthc0 {r:rt}, {c:c0s}"),)),
    _R("CRC32B", "001000 rt:5 rs:5 x:3 000 1111 1 01 000", "crc32b {r:rt}, {r:rs}", need="crc"),
    _R("CRC32H", "001000 rt:5 rs:5 x:3 001 1111 1 01 000", "crc32h {r:rt}, {r:rs}", need="crc"),
    _R("CRC32W", "001000 rt:5 rs:5 x:3 010 1111 1 01 000", "crc32w {r:rt}, {r:rs}", need="crc"),
    _R("CRC32CB", "001000 rt:5 rs:5 x:3 100 1111 1 01 000", "crc32cb {r:rt}, {r:rs}",
       need="crc"),
    _R("CRC32CH", "001000 rt:5 rs:5 x:3 101 1111 1 01 000", "crc32ch {r:rt}, {r:rs}",
       need="crc"),
    _R("CRC32CW", "001000 rt:5 rs:5 x:3 110 1111 1 01 000", "crc32cw {r:rt}, {r:rs}",
       need="crc"),
    # _POOL32A7 (B.31): P.LSX / PP.LSX / PP.LSXS (B.55, B.71, B.72)
    _R("LBX", "001000 rt:5 rs:5 rd:5 0000 0 000 111", "lbx {r:rd}, {r:rs}({r:rt})"),
    _R("SBX", "001000 rt:5 rs:5 rd:5 0001 0 000 111", "sbx {r:rd}, {r:rs}({r:rt})", nms=False),
    _R("LBUX", "001000 rt:5 rs:5 rd:5 0010 0 000 111", "lbux {r:rd}, {r:rs}({r:rt})"),
    _R("LHX", "001000 rt:5 rs:5 rd:5 0100 0 000 111", "lhx {r:rd}, {r:rs}({r:rt})"),
    _R("SHX", "001000 rt:5 rs:5 rd:5 0101 0 000 111", "shx {r:rd}, {r:rs}({r:rt})", nms=False),
    _R("LHUX", "001000 rt:5 rs:5 rd:5 0110 0 000 111", "lhux {r:rd}, {r:rs}({r:rt})"),
    _R("LWX", "001000 rt:5 rs:5 rd:5 1000 0 000 111", "lwx {r:rd}, {r:rs}({r:rt})"),
    _R("SWX", "001000 rt:5 rs:5 rd:5 1001 0 000 111", "swx {r:rd}, {r:rs}({r:rt})", nms=False),
    _R("LHXS", "001000 rt:5 rs:5 rd:5 0100 1 000 111", "lhxs {r:rd}, {r:rs}({r:rt})"),
    _R("SHXS", "001000 rt:5 rs:5 rd:5 0101 1 000 111", "shxs {r:rd}, {r:rs}({r:rt})",
       nms=False),
    _R("LHUXS", "001000 rt:5 rs:5 rd:5 0110 1 000 111", "lhuxs {r:rd}, {r:rs}({r:rt})"),
    _R("LWXS[32]", "001000 rt:5 rs:5 rd:5 1000 1 000 111", "lwxs {r:rd}, {r:rs}({r:rt})",
       name="LWXS"),
    _R("SWXS", "001000 rt:5 rs:5 rd:5 1001 1 000 111", "swxs {r:rd}, {r:rs}({r:rt})",
       nms=False),
    _R("LSA", "001000 rt:5 rs:5 rd:5 u2:2 x:3 001 111", "lsa {r:rd}, {r:rs}, {r:rt}, {d:u2}"),
    _R("EXTW", "001000 rt:5 rs:5 rd:5 shift:5 011 111", "extw {r:rd}, {r:rs}, {r:rt}, {d:shift}"),
    # POOL32Axf_4 (B.73)
    _R("CLO", "001000 rt:5 rs:5 0100101 100 111 111", "clo {r:rt}, {r:rs}", nms=False),
    _R("CLZ", "001000 rt:5 rs:5 0101101 100 111 111", "clz {r:rt}, {r:rs}", nms=False),
    # POOL32Axf_5 groups 0, 1, 3 / ERETx (B.74, B.76-B.79)
    _R("TLBP", "001000 x:10 00 00001 101 111 111", "tlbp", need="tlb"),
    _R("TLBINV", "001000 x:10 00 00011 101 111 111", "tlbinv", need="tlb"),
    _R("GINVT", "001000 x:3 type:2 rs:5 00 00111 101 111 111", "ginvt {r:rs}, {d:type}",
       need="ginv"),
    _R("TLBR", "001000 x:10 00 01001 101 111 111", "tlbr", need="tlb"),
    _R("TLBINVF", "001000 x:10 00 01011 101 111 111", "tlbinvf", need="tlb"),
    _R("GINVI", "001000 x:5 rs:5 00 01111 101 111 111", "ginvi {r:rs}", need="ginv"),
    _R("TLBWI", "001000 x:10 00 10001 101 111 111", "tlbwi", need="tlb"),
    _R("TLBWR", "001000 x:10 00 11001 101 111 111", "tlbwr", need="tlb"),
    _R("DI", "001000 rt:5 x:5 01 00011 101 111 111", "di {r:rt}",
       alias=((lambda f: f["rt"] == 0 and f["x"] == 0, "di"),)),
    _R("EI", "001000 rt:5 x:5 01 01011 101 111 111", "ei {r:rt}",
       alias=((lambda f: f["rt"] == 0 and f["x"] == 0, "ei"),)),
    _R("WAIT", "001000 code:10 11 00001 101 111 111", "wait {x:code}",
       alias=((lambda f: f["code"] == 0, "wait"),)),
    _R("RDPGPR", "001000 rt:5 rs:5 11 10000 101 111 111", "rdpgpr {r:rt}, {r:rs}"),
    _R("DERET", "001000 x:10 11 10001 101 111 111", "deret", need="debug"),
    _R("WRPGPR", "001000 rt:5 rs:5 11 11000 101 111 111", "wrpgpr {r:rt}, {r:rs}"),
    _R("ERET", "001000 x:9 0 11 11001 101 111 111", "eret", "nc=0"),
    _R("ERETNC", "001000 x:9 1 11 11001 101 111 111", "eretnc", "nc=1", name="ERET"),
]

# ------------------------------------------------------------------ 48-bit
#
# MD01247 App. B.10 (P48I)

_ROWS48 = [
    _R("LI[48]", "011000 rt:5 00000 s[15:0] s[31:16]", "li {r:rt}, {x:s}", "s=sx(s,32)",
       name="LI", nms=False),
    _R("ADDIU[48]", "011000 rt:5 00001 s[15:0] s[31:16]", "addiu {r:rt}, {r:rs}, {d:imm}",
       "imm=sx(s,32) rs=rt", nms=False),
    _R("ADDIU[GP48]", "011000 rt:5 00010 s[15:0] s[31:16]", "addiu {r:rt}, {r:rs}, {d:imm}",
       "imm=sx(s,32) rs=28", nms=False),
    _R("ADDIUPC[48]", "011000 rt:5 00011 s[15:0] s[31:16]", "addiupc {r:rt}, {d:imm}",
       "s=sx(s,32) imm=add(s,6)", alias=((_ALWAYS, "lapc {r:rt}, {b:s}"),), name="ADDIUPC",
       nms=False),
    _R("LWPC[48]", "011000 rt:5 01011 s[15:0] s[31:16]", "lwpc {r:rt}, {b:offset}",
       "offset=sx(s,32)", name="LWPC", nms=False),
    _R("SWPC[48]", "011000 rt:5 01111 s[15:0] s[31:16]", "swpc {r:rt}, {b:offset}",
       "offset=sx(s,32)", name="SWPC", nms=False),
]


#: the labelled cells of App. B's opcode map that this decoder leaves
#: reserved: (size, mask, value, label).  DSP and MT are the ASEs of
#: MD01249 / MD01255 (their formats fill most of these cells: phases 5a /
#: 5b); CP1 / CP2 / UDI / VZ / MCU / MIPS64 have no nanoMIPS32 definition
#: in MD01247.
REGIONS = [
    (4, 0xFC000000, 0xA0000000, "CP1"),                     # P32 101000
    (4, 0xFC000000, 0xC0000000, "MIPS64"),                  # P32 110000
    (4, 0xFC000007, 0x20000001, "UDI"),                     # P32A 001
    (4, 0xFC000007, 0x20000002, "CP2"),                     # P32A 010
    (4, 0xFC000007, 0x20000003, "UDI"),                     # P32A 011
    (4, 0xFC000007, 0x20000005, "DSP"),                     # P32A 101 (_POOL32A5)
    (4, 0xFC000003, 0x40000001, "MIPS64"),                  # P.GP.W 01
    (4, 0xFC1C0000, 0x44180000, "CP1"),                     # P.GP.BH 110
    (4, 0xFC1C0000, 0x441C0000, "MIPS64"),                  # P.GP.BH 111
    (4, 0xFC00F000, 0x80009000, "MIPS64"),                  # P.U12 1001-1011
    (4, 0xFC00E000, 0x8000A000, "MIPS64"),
    (4, 0xFC10F000, 0x80103000, "CP1"),                     # P.SR bit 20
    (4, 0xFC00F100, 0x8000C100, "MIPS64"),                  # P.SHIFT 1xxx
    (4, 0xFC00E820, 0x8000E020, "MIPS64"),                  # P.INS / P.EXT 01, 11
    (4, 0xFC00E820, 0x8000E800, "MIPS64"),                  # 10
    (4, 0xFC00F000, 0x84007000, "MIPS64"),                  # P.LS.U12 0111
    (4, 0xFC00E000, 0x8400A000, "CP1"),                     # 1010, 1011
    (4, 0xFC00E000, 0x8400C000, "MIPS64"),                  # 1100, 1101
    (4, 0xFC00E000, 0x8400E000, "CP1"),                     # 1110, 1111
    (4, 0xFC1FC000, 0x88004000, "CP1"),                     # P.BR3A 00000 / 00001
    (4, 0xFC1FC000, 0x88014000, "CP1"),
    (4, 0xFC1FC000, 0x88024000, "CP2"),                     # 00010 / 00011
    (4, 0xFC1FC000, 0x88034000, "CP2"),
    (4, 0xFC1FC000, 0x88044000, "DSP"),                     # 00100 (BPOSGE32C)
    (4, 0xFC000600, 0xA4000600, "MIPS64"),                  # P.LS.S9 110 / 111
    (4, 0xFC007F00, 0xA4003800, "MIPS64"),                  # P.LS.S0 0111
    (4, 0xFC007700, 0xA4005000, "CP1"),                     # 1010, 1110
    (4, 0xFC007700, 0xA4005800, "CP1"),                     # 1011, 1111
    (4, 0xFC007700, 0xA4006000, "MIPS64"),                  # 1100 / 1101
    (4, 0xFC007700, 0xA4006800, "MIPS64"),
    (4, 0xFC007F00, 0xA4001100, "MCU"),                     # P.LS.S1 0010
    (4, 0xFC006700, 0xA4004100, "CP2"),                     # 1000, 1001, 1100, 1101
    (4, 0xFC007700, 0xA4007100, "MIPS64"),                  # 1110 / 1111
    (4, 0xFFFC0000, 0x000C0000, "VZ"),                      # P.SYSCALL bit 18
    (4, 0xFC000BFF, 0x20000BE8, "MIPS64"),                  # CRC32 011 / 111
    (4, 0xFC00077F, 0x20000387, "MIPS64"),                  # PP.LSX 0111 (both)
    (4, 0xFC00073F, 0x20000507, "CP1"),                     # 1010, 1011 (both)
    (4, 0xFC00063F, 0x20000607, "MIPS64"),                  # 1100, 1101 (both)
    (4, 0xFC00063F, 0x20000707, "CP1"),                     # 1110, 1111 (both)
    (4, 0xFC0001FF, 0x2000007F, "DSP"),                     # POOL32Axf 001
    (4, 0xFC0001FF, 0x200000BF, "DSP"),                     # 010
    (4, 0xFC0001FF, 0x200001FF, "DSP"),                     # 111
    (4, 0xFC00FFFF, 0x2000017F, "VZ"),                      # POOL32Axf_5 group0 ??000
    (4, 0xFC00FFFF, 0x2000117F, "VZ"),
    (4, 0xFC00FFFF, 0x2000217F, "VZ"),
    (4, 0xFC00FFFF, 0x2000317F, "VZ"),
    (4, 0xFC00FFFF, 0x2000057F, "VZ"),                      # ??010
    (4, 0xFC00FFFF, 0x2000157F, "VZ"),
    (4, 0xFC00FFFF, 0x20000D7F, "VZ"),                      # 00110
    (4, 0xFC00FFFF, 0x2000D37F, "MCU"),                     # group3 01001
    (2, 0xFFFC, 0x100C, "VZ"),                              # P16.SYSCALL bit 2
    (6, 0xFC1F << 32, 0x6011 << 32, "MIPS64"),              # P48I
    (6, 0xFC1F << 32, 0x6014 << 32, "MIPS64"),
    (6, 0xFC1F << 32, 0x601B << 32, "MIPS64"),
    (6, 0xFC1F << 32, 0x601F << 32, "MIPS64"),
]


def _a01(row: int, col: int) -> int:
    """_POOL32A0_1's cell (B.54): bits 9:6 = row, 5 = 1, 4:3 = col."""
    return 0x20000000 | row << 6 | 1 << 5 | col << 3


REGIONS += [(4, 0xFC0003FF, _a01(row, col), label) for row, col, label in (
    (2, 2, "VZ"), (2, 3, "VZ"), (3, 2, "VZ"), (3, 3, "VZ"), (4, 2, "MIPS64"),
    (5, 2, "MIPS64"), (6, 2, "VZ"), (7, 2, "VZ"), (8, 1, "MT"), (8, 2, "MT"), (8, 3, "MT"),
    (9, 1, "MT"), (9, 2, "MT"), (9, 3, "MT"), (10, 2, "MT"))]

#: POOL32Axf_4's DSP and CP2 cells (B.73): bits 15:9 -> label
_AXF4 = {**{(r << 3) | 0: "DSP" for r in (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15)},
         **{(r << 3) | 1: "DSP" for r in (0, 1, 7, 9, 11, 13)},
         **{(r << 3) | 6: "CP2" for r in (4, 5, 6, 7, 8, 9, 12, 13)}}


def region(raw: int, size: int) -> str | None:
    """The opcode map's label of a reserved encoding (``DSP``, ``MT`` ...),
    ``None`` for a plain reserved one."""
    for s, mask, value, label in REGIONS:
        if s == size and raw & mask == value:
            return label
    if size == 4 and raw & 0xFC0001FF == 0x2000013F:          # POOL32Axf_4
        return _AXF4.get((raw >> 9) & 0x7F)
    return None


# ------------------------------------------------------------------ the DSP ASE
#
# MD01249 rev. 0.04 (the nanoMIPS DSP ASE: Revision 3 of the DSP Module on
# nanoMIPS32, ch. 6 the pages), generated from binutils' nanomips-opc.c D32
# entries (emips/data/nanomips_asm.json; tests/test_nanomips_dsp.py checks
# every row against that table and the manual's encoding diagrams).  The
# fields are named as the manual's diagrams name them (rt rs rd ac sa
# mask imm shift size); each page runs its MIPS32 twin (emips.simnano).
# The accumulator forms of MULT ... MTLO are binutils' MULT[DSP] ...
# Decoded always (as objdump does); a core without the ASE raises RI.

_ROWS_DSP = [
    _R("ABSQ_S.PH", "001000 rt:5 rs:5 0001000100111111", "absq_s.ph {r:rt}, {r:rs}", need="dsp"),
    _R("ABSQ_S.QB", "001000 rt:5 rs:5 0000000100111111", "absq_s.qb {r:rt}, {r:rs}", need="dsp"),
    _R("ABSQ_S.W", "001000 rt:5 rs:5 0010000100111111", "absq_s.w {r:rt}, {r:rs}", need="dsp"),
    _R("ADDQ.PH", "001000 rt:5 rs:5 rd:5 00000001101", "addq.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDQH.PH", "001000 rt:5 rs:5 rd:5 00001001101", "addqh.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDQH.W", "001000 rt:5 rs:5 rd:5 00010001101", "addqh.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDQH_R.PH", "001000 rt:5 rs:5 rd:5 10001001101", "addqh_r.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDQH_R.W", "001000 rt:5 rs:5 rd:5 10010001101", "addqh_r.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDQ_S.PH", "001000 rt:5 rs:5 rd:5 10000001101", "addq_s.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDQ_S.W", "001000 rt:5 rs:5 rd:5 x:1 1100000101", "addq_s.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDSC", "001000 rt:5 rs:5 rd:5 x:1 1110000101", "addsc {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDU.PH", "001000 rt:5 rs:5 rd:5 00100001101", "addu.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDU.QB", "001000 rt:5 rs:5 rd:5 00011001101", "addu.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDUH.QB", "001000 rt:5 rs:5 rd:5 00101001101", "adduh.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDUH_R.QB", "001000 rt:5 rs:5 rd:5 10101001101", "adduh_r.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDU_S.PH", "001000 rt:5 rs:5 rd:5 10100001101", "addu_s.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDU_S.QB", "001000 rt:5 rs:5 rd:5 10011001101", "addu_s.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("ADDWC", "001000 rt:5 rs:5 rd:5 x:1 1111000101", "addwc {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("APPEND", "001000 rt:5 rs:5 sa:5 01000010101", "append {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("BITREV", "001000 rt:5 rs:5 0011000100111111", "bitrev {r:rt}, {r:rs}", need="dsp"),
    _R("CMP.EQ.PH", "001000 rt:5 rs:5 x:6 0000000101", "cmp.eq.ph {r:rs}, {r:rt}", need="dsp"),
    _R("CMP.LE.PH", "001000 rt:5 rs:5 x:6 0010000101", "cmp.le.ph {r:rs}, {r:rt}", need="dsp"),
    _R("CMP.LT.PH", "001000 rt:5 rs:5 x:6 0001000101", "cmp.lt.ph {r:rs}, {r:rt}", need="dsp"),
    _R("CMPGDU.EQ.QB", "001000 rt:5 rs:5 rd:5 x:1 0110000101", "cmpgdu.eq.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("CMPGDU.LE.QB", "001000 rt:5 rs:5 rd:5 x:1 1000000101", "cmpgdu.le.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("CMPGDU.LT.QB", "001000 rt:5 rs:5 rd:5 x:1 0111000101", "cmpgdu.lt.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("CMPGU.EQ.QB", "001000 rt:5 rs:5 rd:5 x:1 0011000101", "cmpgu.eq.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("CMPGU.LE.QB", "001000 rt:5 rs:5 rd:5 x:1 0101000101", "cmpgu.le.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("CMPGU.LT.QB", "001000 rt:5 rs:5 rd:5 x:1 0100000101", "cmpgu.lt.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("CMPU.EQ.QB", "001000 rt:5 rs:5 x:6 1001000101", "cmpu.eq.qb {r:rs}, {r:rt}", need="dsp"),
    _R("CMPU.LE.QB", "001000 rt:5 rs:5 x:6 1011000101", "cmpu.le.qb {r:rs}, {r:rt}", need="dsp"),
    _R("CMPU.LT.QB", "001000 rt:5 rs:5 x:6 1010000101", "cmpu.lt.qb {r:rs}, {r:rt}", need="dsp"),
    _R("DPA.W.PH", "001000 rt:5 rs:5 ac:2 00000010111111", "dpa.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPAQX_S.W.PH", "001000 rt:5 rs:5 ac:2 10001010111111", "dpaqx_s.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPAQX_SA.W.PH", "001000 rt:5 rs:5 ac:2 11001010111111", "dpaqx_sa.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPAQ_S.W.PH", "001000 rt:5 rs:5 ac:2 00001010111111", "dpaq_s.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPAQ_SA.L.W", "001000 rt:5 rs:5 ac:2 01001010111111", "dpaq_sa.l.w $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPAU.H.QBL", "001000 rt:5 rs:5 ac:2 10000010111111", "dpau.h.qbl $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPAU.H.QBR", "001000 rt:5 rs:5 ac:2 11000010111111", "dpau.h.qbr $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPAX.W.PH", "001000 rt:5 rs:5 ac:2 01000010111111", "dpax.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPS.W.PH", "001000 rt:5 rs:5 ac:2 00010010111111", "dps.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPSQX_S.W.PH", "001000 rt:5 rs:5 ac:2 10011010111111", "dpsqx_s.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPSQX_SA.W.PH", "001000 rt:5 rs:5 ac:2 11011010111111", "dpsqx_sa.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPSQ_S.W.PH", "001000 rt:5 rs:5 ac:2 00011010111111", "dpsq_s.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPSQ_SA.L.W", "001000 rt:5 rs:5 ac:2 01011010111111", "dpsq_sa.l.w $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPSU.H.QBL", "001000 rt:5 rs:5 ac:2 10010010111111", "dpsu.h.qbl $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPSU.H.QBR", "001000 rt:5 rs:5 ac:2 11010010111111", "dpsu.h.qbr $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("DPSX.W.PH", "001000 rt:5 rs:5 ac:2 01010010111111", "dpsx.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("EXTP", "001000 rt:5 size:5 ac:2 10011001111111", "extp {r:rt}, $ac{d:ac}, {d:size}", need="dsp"),
    _R("EXTPDP", "001000 rt:5 size:5 ac:2 11011001111111", "extpdp {r:rt}, $ac{d:ac}, {d:size}", need="dsp"),
    _R("EXTPDPV", "001000 rt:5 rs:5 ac:2 11100010111111", "extpdpv {r:rt}, $ac{d:ac}, {r:rs}", need="dsp"),
    _R("EXTPV", "001000 rt:5 rs:5 ac:2 10100010111111", "extpv {r:rt}, $ac{d:ac}, {r:rs}", need="dsp"),
    _R("EXTR.W", "001000 rt:5 sa:5 ac:2 00111001111111", "extr.w {r:rt}, $ac{d:ac}, {d:sa}", need="dsp"),
    _R("EXTRV.W", "001000 rt:5 rs:5 ac:2 00111010111111", "extrv.w {r:rt}, $ac{d:ac}, {r:rs}", need="dsp"),
    _R("EXTRV_R.W", "001000 rt:5 rs:5 ac:2 01111010111111", "extrv_r.w {r:rt}, $ac{d:ac}, {r:rs}", need="dsp"),
    _R("EXTRV_RS.W", "001000 rt:5 rs:5 ac:2 10111010111111", "extrv_rs.w {r:rt}, $ac{d:ac}, {r:rs}", need="dsp"),
    _R("EXTRV_S.H", "001000 rt:5 rs:5 ac:2 11111010111111", "extrv_s.h {r:rt}, $ac{d:ac}, {r:rs}", need="dsp"),
    _R("EXTR_R.W", "001000 rt:5 sa:5 ac:2 01111001111111", "extr_r.w {r:rt}, $ac{d:ac}, {d:sa}", need="dsp"),
    _R("EXTR_RS.W", "001000 rt:5 sa:5 ac:2 10111001111111", "extr_rs.w {r:rt}, $ac{d:ac}, {d:sa}", need="dsp"),
    _R("EXTR_S.H", "001000 rt:5 sa:5 ac:2 11111001111111", "extr_s.h {r:rt}, $ac{d:ac}, {d:sa}", need="dsp"),
    _R("INSV", "001000 rt:5 rs:5 0100000100111111", "insv {r:rt}, {r:rs}", need="dsp"),
    _R("MADD[DSP]", "001000 rt:5 rs:5 ac:2 00101010111111", "madd $ac{d:ac}, {r:rs}, {r:rt}", name="MADD", need="dsp"),
    _R("MADDU[DSP]", "001000 rt:5 rs:5 ac:2 01101010111111", "maddu $ac{d:ac}, {r:rs}, {r:rt}", name="MADDU", need="dsp"),
    _R("MAQ_S.W.PHL", "001000 rt:5 rs:5 ac:2 01101001111111", "maq_s.w.phl $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("MAQ_S.W.PHR", "001000 rt:5 rs:5 ac:2 00101001111111", "maq_s.w.phr $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("MAQ_SA.W.PHL", "001000 rt:5 rs:5 ac:2 11101001111111", "maq_sa.w.phl $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("MAQ_SA.W.PHR", "001000 rt:5 rs:5 ac:2 10101001111111", "maq_sa.w.phr $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("MFHI[DSP]", "001000 rt:5 x:5 ac:2 00000001111111", "mfhi {r:rt}, $ac{d:ac}", name="MFHI", need="dsp"),
    _R("MFLO[DSP]", "001000 rt:5 x:5 ac:2 01000001111111", "mflo {r:rt}, $ac{d:ac}", name="MFLO", need="dsp"),
    _R("MODSUB", "001000 rt:5 rs:5 rd:5 x:1 1010010101", "modsub {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MSUB[DSP]", "001000 rt:5 rs:5 ac:2 10101010111111", "msub $ac{d:ac}, {r:rs}, {r:rt}", name="MSUB", need="dsp"),
    _R("MSUBU[DSP]", "001000 rt:5 rs:5 ac:2 11101010111111", "msubu $ac{d:ac}, {r:rs}, {r:rt}", name="MSUBU", need="dsp"),
    _R("MTHI[DSP]", "001000 x:5 rs:5 ac:2 10000001111111", "mthi {r:rs}, $ac{d:ac}", name="MTHI", need="dsp"),
    _R("MTHLIP", "001000 x:5 rs:5 ac:2 00001001111111", "mthlip {r:rs}, $ac{d:ac}", need="dsp"),
    _R("MTLO[DSP]", "001000 x:5 rs:5 ac:2 11000001111111", "mtlo {r:rs}, $ac{d:ac}", name="MTLO", need="dsp"),
    _R("MUL.PH", "001000 rt:5 rs:5 rd:5 00000101101", "mul.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULEQ_S.W.PHL", "001000 rt:5 rs:5 rd:5 x:1 0000100101", "muleq_s.w.phl {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULEQ_S.W.PHR", "001000 rt:5 rs:5 rd:5 x:1 0001100101", "muleq_s.w.phr {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULEU_S.PH.QBL", "001000 rt:5 rs:5 rd:5 x:1 0010010101", "muleu_s.ph.qbl {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULEU_S.PH.QBR", "001000 rt:5 rs:5 rd:5 x:1 0011010101", "muleu_s.ph.qbr {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULQ_RS.PH", "001000 rt:5 rs:5 rd:5 x:1 0100010101", "mulq_rs.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULQ_RS.W", "001000 rt:5 rs:5 rd:5 x:1 0110010101", "mulq_rs.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULQ_S.PH", "001000 rt:5 rs:5 rd:5 x:1 0101010101", "mulq_s.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULQ_S.W", "001000 rt:5 rs:5 rd:5 x:1 0111010101", "mulq_s.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULSA.W.PH", "001000 rt:5 rs:5 ac:2 10110010111111", "mulsa.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULSAQ_S.W.PH", "001000 rt:5 rs:5 ac:2 11110010111111", "mulsaq_s.w.ph $ac{d:ac}, {r:rs}, {r:rt}", need="dsp"),
    _R("MULT[DSP]", "001000 rt:5 rs:5 ac:2 00110010111111", "mult $ac{d:ac}, {r:rs}, {r:rt}", name="MULT", need="dsp"),
    _R("MULTU[DSP]", "001000 rt:5 rs:5 ac:2 01110010111111", "multu $ac{d:ac}, {r:rs}, {r:rt}", name="MULTU", need="dsp"),
    _R("MUL_S.PH", "001000 rt:5 rs:5 rd:5 10000101101", "mul_s.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PACKRL.PH", "001000 rt:5 rs:5 rd:5 x:1 0110101101", "packrl.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PICK.PH", "001000 rt:5 rs:5 rd:5 x:1 1000101101", "pick.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PICK.QB", "001000 rt:5 rs:5 rd:5 x:1 0111101101", "pick.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PRECEQ.W.PHL", "001000 rt:5 rs:5 0101000100111111", "preceq.w.phl {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEQ.W.PHR", "001000 rt:5 rs:5 0110000100111111", "preceq.w.phr {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEQU.PH.QBL", "001000 rt:5 rs:5 0111000100111111", "precequ.ph.qbl {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEQU.PH.QBLA", "001000 rt:5 rs:5 0111001100111111", "precequ.ph.qbla {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEQU.PH.QBR", "001000 rt:5 rs:5 1001000100111111", "precequ.ph.qbr {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEQU.PH.QBRA", "001000 rt:5 rs:5 1001001100111111", "precequ.ph.qbra {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEU.PH.QBL", "001000 rt:5 rs:5 1011000100111111", "preceu.ph.qbl {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEU.PH.QBLA", "001000 rt:5 rs:5 1011001100111111", "preceu.ph.qbla {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEU.PH.QBR", "001000 rt:5 rs:5 1101000100111111", "preceu.ph.qbr {r:rt}, {r:rs}", need="dsp"),
    _R("PRECEU.PH.QBRA", "001000 rt:5 rs:5 1101001100111111", "preceu.ph.qbra {r:rt}, {r:rs}", need="dsp"),
    _R("PRECR.QB.PH", "001000 rt:5 rs:5 rd:5 x:1 0001101101", "precr.qb.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PRECRQ.PH.W", "001000 rt:5 rs:5 rd:5 x:1 0011101101", "precrq.ph.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PRECRQ.QB.PH", "001000 rt:5 rs:5 rd:5 x:1 0010101101", "precrq.qb.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PRECRQU_S.QB.PH", "001000 rt:5 rs:5 rd:5 x:1 0101101101", "precrqu_s.qb.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PRECRQ_RS.PH.W", "001000 rt:5 rs:5 rd:5 x:1 0100101101", "precrq_rs.ph.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("PRECR_SRA.PH.W", "001000 rt:5 rs:5 sa:5 01111001101", "precr_sra.ph.w {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("PRECR_SRA_R.PH.W", "001000 rt:5 rs:5 sa:5 11111001101", "precr_sra_r.ph.w {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("RADDU.W.QB", "001000 rt:5 rs:5 1111000100111111", "raddu.w.qb {r:rt}, {r:rs}", need="dsp"),
    _R("RDDSP", "001000 rt:5 mask:7 00011001111111", "rddsp {r:rt}, {x:mask}", alias=((lambda f: f["mask"] == 0x7F, "rddsp {r:rt}"),), need="dsp"),
    _R("REPL.PH", "001000 rt:5 imm:10 x:1 0000111101", "repl.ph {r:rt}, {d:simm}", "simm=sx(imm,10)", need="dsp"),
    _R("REPL.QB", "001000 rt:5 imm:8 x:1 010111111111", "repl.qb {r:rt}, {x:imm}", need="dsp"),
    _R("REPLV.PH", "001000 rt:5 rs:5 0000001100111111", "replv.ph {r:rt}, {r:rs}", need="dsp"),
    _R("REPLV.QB", "001000 rt:5 rs:5 0001001100111111", "replv.qb {r:rt}, {r:rs}", need="dsp"),
    _R("SHILO", "001000 x:4 shift:6 ac:2 x:4 0000011101", "shilo $ac{d:ac}, {d:s}", "s=sx(shift,6)", need="dsp"),
    _R("SHILOV", "001000 x:5 rs:5 ac:2 01001001111111", "shilov $ac{d:ac}, {r:rs}", need="dsp"),
    _R("SHLL.PH", "001000 rt:5 rs:5 sa:4 001110110101", "shll.ph {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHLL.QB", "001000 rt:5 rs:5 sa:3 0100001111111", "shll.qb {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHLLV.PH", "001000 rt:5 rs:5 rd:5 01110001101", "shllv.ph {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHLLV.QB", "001000 rt:5 rs:5 rd:5 x:1 1110010101", "shllv.qb {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHLLV_S.PH", "001000 rt:5 rs:5 rd:5 11110001101", "shllv_s.ph {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHLLV_S.W", "001000 rt:5 rs:5 rd:5 x:1 1111010101", "shllv_s.w {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHLL_S.PH", "001000 rt:5 rs:5 sa:4 101110110101", "shll_s.ph {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHLL_S.W", "001000 rt:5 rs:5 sa:5 x:1 1111110101", "shll_s.w {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHRA.PH", "001000 rt:5 rs:5 sa:4 001100110101", "shra.ph {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHRA.QB", "001000 rt:5 rs:5 sa:3 0000111111111", "shra.qb {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHRAV.PH", "001000 rt:5 rs:5 rd:5 00110001101", "shrav.ph {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHRAV.QB", "001000 rt:5 rs:5 rd:5 00111001101", "shrav.qb {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHRAV_R.PH", "001000 rt:5 rs:5 rd:5 10110001101", "shrav_r.ph {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHRAV_R.QB", "001000 rt:5 rs:5 rd:5 10111001101", "shrav_r.qb {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHRAV_R.W", "001000 rt:5 rs:5 rd:5 x:1 1011010101", "shrav_r.w {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHRA_R.PH", "001000 rt:5 rs:5 sa:4 x:1 11100110101", "shra_r.ph {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHRA_R.QB", "001000 rt:5 rs:5 sa:3 1000111111111", "shra_r.qb {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHRA_R.W", "001000 rt:5 rs:5 sa:5 x:1 1011110101", "shra_r.w {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHRL.PH", "001000 rt:5 rs:5 sa:4 001111111111", "shrl.ph {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHRL.QB", "001000 rt:5 rs:5 sa:3 1100001111111", "shrl.qb {r:rt}, {r:rs}, {d:sa}", need="dsp"),
    _R("SHRLV.PH", "001000 rt:5 rs:5 rd:5 x:1 1100010101", "shrlv.ph {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SHRLV.QB", "001000 rt:5 rs:5 rd:5 x:1 1101010101", "shrlv.qb {r:rd}, {r:rt}, {r:rs}", need="dsp"),
    _R("SUBQ.PH", "001000 rt:5 rs:5 rd:5 01000001101", "subq.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBQH.PH", "001000 rt:5 rs:5 rd:5 01001001101", "subqh.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBQH.W", "001000 rt:5 rs:5 rd:5 01010001101", "subqh.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBQH_R.PH", "001000 rt:5 rs:5 rd:5 11001001101", "subqh_r.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBQH_R.W", "001000 rt:5 rs:5 rd:5 11010001101", "subqh_r.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBQ_S.PH", "001000 rt:5 rs:5 rd:5 11000001101", "subq_s.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBQ_S.W", "001000 rt:5 rs:5 rd:5 x:1 1101000101", "subq_s.w {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBU.PH", "001000 rt:5 rs:5 rd:5 01100001101", "subu.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBU.QB", "001000 rt:5 rs:5 rd:5 01011001101", "subu.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBUH.QB", "001000 rt:5 rs:5 rd:5 01101001101", "subuh.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBUH_R.QB", "001000 rt:5 rs:5 rd:5 11101001101", "subuh_r.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBU_S.PH", "001000 rt:5 rs:5 rd:5 11100001101", "subu_s.ph {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("SUBU_S.QB", "001000 rt:5 rs:5 rd:5 11011001101", "subu_s.qb {r:rd}, {r:rs}, {r:rt}", need="dsp"),
    _R("WRDSP", "001000 rt:5 mask:7 01011001111111", "wrdsp {r:rt}, {x:mask}", alias=((lambda f: f["mask"] == 0x7F, "wrdsp {r:rt}"),), need="dsp"),
    _R("BPOSGE32C", "100010 x:5 00100 01 s[13:1] s[14]", "bposge32c {b:offset}",
       "offset=sx(s,15)", need="dsp"),
]

# ------------------------------------------------------------------ the MT ASE
#
# MD01255 rev. 1.17 (the MIPS MT Module, MD00378, on nanoMIPS32: ch. 5 the
# eight instructions' encodings; MFTR / MTTR have a 5-bit sel and no rx).
# The texts are objdump's: binutils' MT32 idioms (mftc0, mftgpr, mftlo
# $ac1, mftdsp, cftc1 ...) come before MFTR / MTTR in its table.


def _tr(mn: str) -> tuple:
    """MFTR / MTTR's idioms (MD01255 Table 5.1 / 5.2, binutils' order)."""
    t = "mft" if mn == "mftr" else "mtt"
    c = "cft" if mn == "mftr" else "ctt"
    return (
        (lambda f: not f["u"] and not f["h"] and not f["sel"], f"{t}c0 {{r:rt}}, {{c:rs}}"),
        (lambda f: not f["u"] and not f["h"], f"{t}c0 {{r:rt}}, {{c:rs}}, {{d:sel}}"),
        (lambda f: not f["u"] and not f["sel"], f"{t}hc0 {{r:rt}}, {{c:rs}}"),
        (lambda f: not f["u"], f"{t}hc0 {{r:rt}}, {{c:rs}}, {{d:sel}}"),
        (lambda f: f["u"] and not f["h"] and f["sel"] == 0, f"{t}gpr {{r:rt}}, {{r:rs}}"),
        (lambda f: f["u"] and not f["h"] and f["sel"] == 1 and f["rs"] == 0, f"{t}lo {{r:rt}}"),
        (lambda f: f["u"] and not f["h"] and f["sel"] == 1 and f["rs"] == 1, f"{t}hi {{r:rt}}"),
        (lambda f: f["u"] and not f["h"] and f["sel"] == 1 and f["rs"] in (4, 8, 12),
         f"{t}lo {{r:rt}}, $ac{{d:ac}}"),
        (lambda f: f["u"] and not f["h"] and f["sel"] == 1 and f["rs"] in (5, 9, 13),
         f"{t}hi {{r:rt}}, $ac{{d:ac}}"),
        (lambda f: f["u"] and not f["h"] and f["sel"] == 1 and f["rs"] == 16,
         f"{t}dsp {{r:rt}}"),
        (lambda f: f["u"] and f["sel"] == 2 and not f["h"], f"{t}c1 {{r:rt}}, $f{{d:rs}}"),
        (lambda f: f["u"] and f["sel"] == 2, f"{t}hc1 {{r:rt}}, $f{{d:rs}}"),
        (lambda f: f["u"] and f["sel"] == 3 and not f["h"], f"{c}c1 {{r:rt}}, {{c:rs}}"),
        (lambda f: f["u"] and f["sel"] == 4 and not f["h"], f"{t}c2 {{r:rt}}, {{c:rs}}"),
        (lambda f: f["u"] and f["sel"] == 4, f"{t}hc2 {{r:rt}}, {{c:rs}}"),
        (lambda f: f["u"] and f["sel"] == 5 and not f["h"], f"{c}c2 {{r:rt}}, {{c:rs}}"),
    )


_ROWS_MT = [
    _R("DMT", "001000 rt:5 00001 00001 0 1010110 000", "dmt {r:rt}",
       alias=((lambda f: f["rt"] == 0, "dmt"),), need="mt"),
    _R("EMT", "001000 rt:5 00001 00001 1 1010110 000", "emt {r:rt}",
       alias=((lambda f: f["rt"] == 0, "emt"),), need="mt"),
    _R("DVPE", "001000 rt:5 00000 00001 0 1010110 000", "dvpe {r:rt}",
       alias=((lambda f: f["rt"] == 0, "dvpe"),), need="mt"),
    _R("EVPE", "001000 rt:5 00000 00001 1 1010110 000", "evpe {r:rt}",
       alias=((lambda f: f["rt"] == 0, "evpe"),), need="mt"),
    _R("FORK", "001000 rt:5 rs:5 rd:5 x:1 1000101 000", "fork {r:rd}, {r:rs}, {r:rt}",
       need="mt"),
    _R("YIELD", "001000 rt:5 rs:5 x:6 1001101 000", "yield {r:rt}, {r:rs}",
       alias=((lambda f: f["rt"] == 0 and f["x"] == 0, "yield {r:rs}"),), need="mt"),
    _R("MFTR", "001000 rt:5 rs:5 sel:5 u:1 100011 h:1 000",
       "mftr {r:rt}, {r:rs}, {d:u}, {d:sel}, {d:h}", "ac=acc(rs)",
       alias=_tr("mftr"), need="mt"),
    _R("MTTR", "001000 rt:5 rs:5 sel:5 u:1 100111 h:1 000",
       "mttr {r:rt}, {r:rs}, {d:u}, {d:sel}, {d:h}", "ac=acc(rs)",
       alias=_tr("mttr"), need="mt"),
]

# ------------------------------------------------------------------ the table

class _Table:
    """The rows by (size, major opcode), most specific first; two rows of
    the same specificity matching one encoding are a build error unless
    both have a ``when`` (A.4's "with ..." qualifiers)."""

    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows
        self.by_major: dict[tuple[int, int], list[Row]] = {}
        for r in rows:
            major = r.value >> (r.size * 8 - 6)
            self.by_major.setdefault((r.size, major), []).append(r)
        for lst in self.by_major.values():
            lst.sort(key=lambda r: -bin(r.mask).count("1"))
            for i, a in enumerate(lst):
                for b in lst[i + 1:]:
                    if not (a.value ^ b.value) & a.mask & b.mask and \
                            bin(a.mask).count("1") == bin(b.mask).count("1") and \
                            (a.when is None or b.when is None):
                        raise AssertionError(f"{a.form} and {b.form} overlap")
        self.by_form = {r.form: r for r in rows}

    def match(self, raw: int, size: int) -> tuple[Row, dict] | None:
        for r in self.by_major.get((size, raw >> (size * 8 - 6)), ()):
            if raw & r.mask == r.value:
                f = r.extract(raw)
                if r.when is None or r.when(f):
                    return r, f
        return None


ROWS = _ROWS16 + _ROWS32 + _ROWS48 + _ROWS_DSP + _ROWS_MT
#: the MT ASE's formats (MD01255)
MT_FORMS = frozenset(r.form for r in _ROWS_MT)
#: the DSP ASE's formats and pages (MD01249)
DSP_FORMS = frozenset(r.form for r in _ROWS_DSP)
DSP_NAMES = frozenset(r.name for r in _ROWS_DSP)
TABLE = _Table(ROWS)
FORMS = {r.form for r in ROWS}
NAMES = {r.name for r in ROWS}


def size_of(h: int) -> int:
    """2, 4 or 6: the size of the instruction whose first halfword is ``h``
    (App. B.1: bit 12 = 1 is P16; the P32 major opcode 011000 is P48I)."""
    if h & 0x1000:
        return 2
    return 6 if h & 0xFC00 == 0x6000 else 4


@dataclass(frozen=True)
class NM:
    """A decoded nanoMIPS instruction: ``form`` (:class:`Row` name),
    ``size`` 2 / 4 / 6, ``raw`` the encoding (A.2: the first halfword is
    the most significant), ``f`` the arguments after the decode steps."""

    form: str
    size: int
    raw: int
    row: Row = field(compare=False, repr=False)
    f: dict = field(default_factory=dict, compare=False)


class NMDecodeError(ISADecodeError):
    """No nanoMIPS instruction (the Machine: Reserved Instruction);
    ``region`` names the opcode map's cell (``DSP``, ``MT`` ...) when it
    has a label."""

    def __init__(self, why: str, word: int = 0, region: str | None = None) -> None:
        Exception.__init__(self, f"no nanoMIPS instruction: {why}")
        self.word = word
        self.why = why
        self.region = region


def decode(h: int, more: Callable[[], int] | None = None) -> NM:
    """Decode the instruction whose first halfword is ``h``; ``more()``
    gives the following halfwords one at a time (one for a 32-bit
    instruction, two for a 48-bit one).  A reserved encoding raises
    :class:`NMDecodeError`."""
    h &= 0xFFFF
    size = size_of(h)
    raw = h
    for _ in range(size // 2 - 1):
        if more is None:
            raise NMDecodeError(f"a {size * 8}-bit instruction needs its other halfwords", h)
        raw = (raw << 16) | (more() & 0xFFFF)
    return decode_raw(raw, size)


def decode_raw(raw: int, size: int) -> NM:
    """Decode ``raw``, an encoding of ``size`` bytes (A.2's numbering)."""
    hit = TABLE.match(raw, size)
    if hit is None:
        label = region(raw, size)
        what = f"the opcode map's {label} cell" if label else "reserved"
        raise NMDecodeError(f"0x{raw:0{size * 2}x}: {what}", raw, label)
    row, f = hit
    return NM(row.form, size, raw, row, f)


def encode(row: Row, fields: Mapping[str, int]) -> int:
    """The encoding of ``row`` with these raw fields (the template's names:
    ``rt3``, ``s``, ``u`` ...; missing fields and ``x`` are 0): the inverse
    of :meth:`Row.extract`'s field step."""
    raw = row.value
    for name, pieces in row.fields:
        v = fields.get(name, 0)
        for shift, width, at in pieces:
            raw |= ((v >> at) & ((1 << width) - 1)) << shift
    return raw


def halfwords(i: NM) -> list[int]:
    """The instruction's halfwords in memory order."""
    return [(i.raw >> (16 * k)) & 0xFFFF for k in range(i.size // 2 - 1, -1, -1)]


# ------------------------------------------------------------------ text

def reglist(rt: int, count: int, gp: int) -> list[int]:
    """The registers of SAVE / RESTORE: the Operation's ``this_rt`` for
    i = 0 .. count - 1 ($31 wraps to $16; with ``gp`` the last is $28)."""
    return [28 if gp and i + 1 == count else rt + i if rt + i < 32 else rt + i - 16
            for i in range(count)]


def _runs(regs: list[int]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for x in regs:
        if out and out[-1][1] + 1 == x:
            out[-1][1] = x
        else:
            out.append([x, x])
    return [(a, b) for a, b in out]


def _list_text(f: dict, r: Callable[[int], str]) -> str:
    """The register list as GNU writes it: ``fp, ra, s0-s3, gp`` (binutils
    ``nanomips_print_save_restore``: fp and ra on their own, then one
    range, then gp -- the last register's gp bit, or a list from fp / ra
    that wraps round to $28)."""
    rt, count, gp = f["rt"], f["count"], f["gp"]
    if count == 0:
        return ""
    fp = rt == 30
    ra = rt == 31 or (rt == 30 and count > 1)
    gp = 1 if gp or rt + count == 45 else 0
    n = count - gp
    first = rt
    for flag in (fp, ra):
        if flag and n > 0:
            first = (first & 0x10) | ((first + 1) % 32)
            n -= 1
    parts = [r(30)] if fp else []
    if ra:
        parts.append(r(31))
    if n > 0:
        if first + n - 1 > 31:
            # a list from below fp that wraps round: binutils reads past its
            # register names here, so its runs are written out
            for a, b in _runs(reglist(rt, count, f["gp"])[len(parts):len(parts) + n]):
                parts.append(r(a) if a == b else f"{r(a)}-{r(b)}")
        else:
            parts.append(r(first) if n == 1 else f"{r(first)}-{r(first + n - 1)}")
    if gp:
        parts.append(r(28))
    return ", " + ", ".join(parts)


_FIELD = re.compile(r"\{(\w):(\w+)\}")


def _render(text: str, i: NM, pc: int, r: Callable[[int], str],
            symbols: Mapping[int, str] | None) -> str:
    f = i.f

    def one(m: re.Match) -> str:
        kind, name = m.groups()
        if kind == "L":
            return _list_text(f, r)
        v = f[name]
        if kind == "r":
            return r(v)
        if kind == "d":
            return str(v)
        if kind == "x":
            return f"0x{v & M32:x}"
        if kind == "n":
            return f"-{v}"
        if kind == "c":
            return f"${v}"
        if kind == "b":
            a = (pc + i.size + v) & M32
            return symbols[a] if symbols is not None and a in symbols else f"0x{a:08x}"
        if kind == "h":
            return f"%hi(0x{v & M32:x})" if v else "0x0"
        if kind == "p":
            return f"0x{(v >> 12) & 0xFFFFF:x}"
        raise AssertionError(kind)

    out = _FIELD.sub(one, text)
    mn, _, ops = out.partition(" ")
    return f"{mn} {ops.lstrip(', ')}" if ops else mn


def text(i: NM, pc: int = 0, *, abi: bool = True, aliases: bool = True,
         symbols: Mapping[int, str] | None = None) -> str:
    """The instruction at ``pc`` in GNU syntax with ``$`` registers: by
    default as objdump prints it (its aliases: ``li``, ``move``, ``lapc``,
    ``beqzc`` ...); ``aliases=False`` the instruction itself (``objdump -M
    no-aliases``).  ``symbols`` names branch / PC-relative targets."""
    r = lambda n: gpr_name(n, abi)      # noqa: E731
    for cond, t, alias in i.row.texts:
        if alias and not aliases:
            continue
        if cond is None or cond(i.f):
            return _render(t, i, pc, r, symbols)
    raise AssertionError(i.form)


def target(i: NM, pc: int) -> int | None:
    """The branch / PC-relative address of ``i`` at ``pc``, if it has one
    (``effective_address(CPU.next_pc, offset)``)."""
    for _, t, _ in i.row.texts:
        m = re.search(r"\{b:(\w+)\}", t)
        if m:
            return (pc + i.size + i.f[m.group(1)]) & M32
    return None
