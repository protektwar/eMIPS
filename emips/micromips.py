"""microMIPS32: the 16 / 32-bit instruction set (PLAN M17 phase 2).

Reference: *MIPS Architecture for Programmers Volume II-B: The
microMIPS32 Instruction Set* (MD00582) rev. 5.04 (Release 3 / 5) and rev.
6.05 (Release 6): 3.2-3.6 the ISA mode and the offsets, ch. 4 the
formats and the instruction stream, ch. 5 every instruction (5.1 the
16-bit re-encodings, Table 5.3 their register and immediate values, 5.4
the new instructions, 5.5 the recoded MIPS32 ones), ch. 6 the opcode map;
Vol. III (MD00090 6.02) 9.25 BadInstr, 9.38.1 EPC, Config3.ISA /
ISAOnExc.  binutils ``opcodes/micromips-opc.c`` gives GNU's syntax and,
with ``objdump -m mips:micromips``, an independent decoder for
``tests/test_micromips.py``; QEMU's M14K runs the instructions in
``tests/test_micromips_unicorn.py``.

Like :mod:`emips.mips16`, an instruction that is a re-encoding of a
MIPS32 one runs *as* that instruction (:func:`expand` builds its
"twin": the same fields in the MIPS32 encoding, so traces, views,
counters and exceptions are the MIPS32 ones); the instructions with no
MIPS32 twin — the branches and jumps (offsets in halfwords, links past a
16- or 32-bit delay slot), ADDIUPC, LWP / SWP, LWM / SWM, LWXS, MOVEP,
JRADDIUSP — have their own executors (``sim._MM_EXECUTORS``, entries
``MM.*``).

A row of the tables is a template written as in the manual's encoding
diagrams, most significant bit first: ``0`` / ``1`` fixed, a letter a
field (the letters of one row are its own names).  The twin template
(32 bits) uses the same letters: each group of a letter there receives
the field's value (after the row's maps: the 3-bit register numbers of
Table 5.4 / 5.5, the encoded immediates of Table 5.3, scaling, sign
extension), cut to its width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .isa import ISA, ISADecodeError

M32 = 0xFFFFFFFF

#: Table 5.4: the 3-bit register fields -> GPR numbers
GPR3 = (16, 17, 2, 3, 4, 5, 6, 7)
#: Table 5.5: SB16 / SH16 / SW16's source register
GPR3_SRC = (0, 17, 2, 3, 4, 5, 6, 7)
#: MOVEP's sources (Table 5.3; binutils ``reg_mn_map``)
MOVEP_SRC = (0, 17, 2, 3, 16, 18, 19, 20)
#: MOVEP's destination pairs (Table 5.3)
MOVEP_DEST = ((5, 6), (5, 7), (6, 7), (4, 21), (4, 22), (4, 5), (4, 6), (4, 7))
#: ADDIUR2's immediate (Table 5.3)
ADDIUR2_IMM = (1, 4, 8, 12, 16, 20, 24, -1)
#: ANDI16's immediate (Table 5.3)
ANDI16_IMM = (128, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 32768, 65535)
#: LWM32 / SWM32's reglist (5.4 LWM32): 1-9 s-registers (s8 = GPR 30 the
#: ninth), bit 4 = ra too
SREGS = (16, 17, 18, 19, 20, 21, 22, 23, 30)


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _addiusp(v: int) -> int:
    """ADDIUSP's 9-bit field -> the word offset (Table 5.3: -258..-3,
    2..257; QEMU micromips_translate.c.inc)."""
    if v <= 1:
        return 256 + v
    if v <= 255:
        return v
    return v - 512 if v <= 509 else v - 768


#: field maps: (value, width) -> value
_MAPS: dict[str, Callable[[int, int], int]] = {
    "g": lambda v, w: GPR3[v],
    "z": lambda v, w: GPR3_SRC[v],
    "n": lambda v, w: MOVEP_SRC[v],
    "sx": _sx,
    "x2": lambda v, w: v << 1,
    "x4": lambda v, w: v << 2,
    "lbu16": lambda v, w: -1 if v == 15 else v,
    "li16": lambda v, w: -1 if v == 127 else v,
    "sa8": lambda v, w: v or 8,
    "andi16": lambda v, w: ANDI16_IMM[v],
    "addiur2": lambda v, w: ADDIUR2_IMM[v],
    "addiusp": lambda v, w: _addiusp(v) << 2,
}


def _runs(template: str, letter: str) -> list[tuple[int, int]]:
    """The groups of ``letter`` in ``template``: (shift, width), most
    significant first."""
    out, size, k = [], len(template), 0
    while k < size:
        if template[k] == letter:
            j = k
            while j < size and template[j] == letter:
                j += 1
            out.append((size - j, j - k))
            k = j
        else:
            k += 1
    return out


@dataclass(frozen=True)
class Row:
    """One encoding: ``form`` its name (the MIPS32 row name for a recoded
    instruction, the manual's for the others: ``ADDU16``, ``LWSP``, ``LWP``
    ...), ``size`` 2 or 4 bytes, ``mask`` / ``value`` the fixed bits."""

    form: str
    size: int
    mask: int
    value: int
    template: str
    twin: str | None
    maps: dict
    syntax: str | None
    fields: tuple = ()
    #: the twin: (fixed bits, {letter: [(shift, width), ...]})
    twin_fields: tuple | None = None
    #: the twin template as written (spaces end groups)
    twin_source: str | None = None
    #: fields -> True when this row is the one (Release 6 groups told
    #: apart by comparing rs and rt, as in MIPS32 Release 6)
    constraint: Callable[[dict], bool] | None = None
    #: the twin is a Release 5 row (a field wider than in MIPS32 R6)
    r5_twin: bool = False

    def extract(self, raw: int) -> dict:
        f = {}
        for letter, runs in self.fields:
            v = w = 0
            for shift, width in runs:
                v = (v << width) | ((raw >> shift) & ((1 << width) - 1))
                w += width
            for name in self.maps.get(letter, ()):
                v = _MAPS[name](v, w)
            f[letter] = v
        return f


def _row(form: str, template: str, twin: str | None = None, maps: str = "",
         syntax: str | None = None, *, when: Callable[[dict], bool] | None = None,
         r5: bool = False) -> Row:
    t = template.replace(" ", "")
    if len(t) not in (16, 32):
        raise AssertionError(f"{form}: template of {len(t)} bits")
    mask = value = 0
    for c in t:
        mask = (mask << 1) | (c in "01")
        value = (value << 1) | (c == "1")
    letters = sorted({c for c in t if c not in "01"})
    m = {}
    for spec in maps.split():
        k, v = spec.split("=")
        m[k] = tuple(v.split(","))
    tw = twin.replace(" ", "") if twin is not None else None
    if tw is not None and len(tw) != 32:
        raise AssertionError(f"{form}: twin of {len(tw)} bits")
    return Row(form, len(t) // 8, mask, value, t, tw, m, syntax,
               tuple((c, _runs(t, c)) for c in letters),
               _twin_fields(twin) if twin is not None else None, twin, when, r5)


def _twin_fields(twin: str) -> tuple[int, dict]:
    """The twin's fixed bits and its letters' groups; a space ends a group
    (``ddddd ddddd`` = two fields that both receive d)."""
    fixed, groups, pos = 0, {}, 32
    for token in twin.split():
        k = 0
        while k < len(token):
            c, j = token[k], k
            while j < len(token) and token[j] == c:
                j += 1
            width = j - k
            pos -= width
            if c == "1":
                fixed |= ((1 << width) - 1) << pos
            elif c != "0":
                groups.setdefault(c, []).append((pos, width))
            k = j
    return fixed, groups


# ------------------------------------------------------------ the pre-Release 6 table
#
# MD00582 rev. 5.04 ch. 5 / 6.  The 32-bit rows' rt is bits 25:21 and rs
# 20:16 (the reverse of MIPS32); the twins put each field back.

_R = _row
#: the 16-bit forms
_PRE16 = [
    _R("ADDU16", "000001 ddd ttt sss 0", "000000 sssss ttttt ddddd 00000 100001", "d=g t=g s=g",
       "addu {d}, {s}, {t}"),
    _R("SUBU16", "000001 ddd ttt sss 1", "000000 sssss ttttt ddddd 00000 100011", "d=g t=g s=g",
       "subu {d}, {s}, {t}"),
    _R("SLL16", "001001 ddd ttt aaa 0", "00000000000 ttttt ddddd aaaaa 000000", "d=g t=g a=sa8",
       "sll {d}, {t}, {a}"),
    _R("SRL16", "001001 ddd ttt aaa 1", "00000000000 ttttt ddddd aaaaa 000010", "d=g t=g a=sa8",
       "srl {d}, {t}, {a}"),
    _R("NOT16", "010001 0000 ttt sss", "000000 sssss 00000 ttttt 00000 100111", "t=g s=g",
       "not {t}, {s}"),
    _R("XOR16", "010001 0001 ttt sss", "000000 sssss ttttt ttttt 00000 100110", "t=g s=g",
       "xor {t}, {t}, {s}"),
    _R("AND16", "010001 0010 ttt sss", "000000 sssss ttttt ttttt 00000 100100", "t=g s=g",
       "and {t}, {t}, {s}"),
    _R("OR16", "010001 0011 ttt sss", "000000 sssss ttttt ttttt 00000 100101", "t=g s=g",
       "or {t}, {t}, {s}"),
    _R("LWM16", "010001 0100 nn oooo", None, "o=x4", "lwm {L}, {o}($sp)"),
    _R("SWM16", "010001 0101 nn oooo", None, "o=x4", "swm {L}, {o}($sp)"),
    _R("JR16", "010001 01100 sssss", None, "", "jr {s}"),
    _R("JRC", "010001 01101 sssss", None, "", "jrc {s}"),
    _R("JALR16", "010001 01110 sssss", None, "", "jalr {s}"),
    _R("JALRS16", "010001 01111 sssss", None, "", "jalrs {s}"),
    _R("MFHI16", "010001 10000 sssss", "0000000000000000 sssss 00000 010000", "", "mfhi {s}"),
    _R("MFLO16", "010001 10010 sssss", "0000000000000000 sssss 00000 010010", "", "mflo {s}"),
    _R("BREAK16", "010001 101000 cccc", "000000 000000cccc 0000000000 001101", "", "break {c}"),
    _R("SDBBP16", "010001 101100 cccc", "011100 0000000000 000000cccc 111111", "", "sdbbp {c}"),
    _R("JRADDIUSP", "010001 11000 iiiii", None, "i=x4", "jraddiusp {i}"),
    _R("MOVE16", "000011 ddddd sssss", "000000 sssss 00000 ddddd 00000 100101", "",
       "move {d}, {s}"),
    _R("ANDI16", "001011 ddd sss iiii", "001100 sssss ddddd iiiiiiiiiiiiiiii",
       "d=g s=g i=andi16", "andi {d}, {s}, {i}"),
    _R("ADDIUS5", "010011 ddddd iiii 0", "001001 ddddd ddddd iiiiiiiiiiiiiiii", "i=sx",
       "addiu {d}, {d}, {i}"),
    _R("ADDIUSP", "010011 iiiiiiiii 1", "001001 11101 11101 iiiiiiiiiiiiiiii", "i=addiusp",
       "addiu $sp, $sp, {i}"),
    _R("ADDIUR2", "011011 ddd sss iii 0", "001001 sssss ddddd iiiiiiiiiiiiiiii",
       "d=g s=g i=addiur2", "addiu {d}, {s}, {i}"),
    _R("ADDIUR1SP", "011011 ddd iiiiii 1", "001001 11101 ddddd iiiiiiiiiiiiiiii", "d=g i=x4",
       "addiu {d}, $sp, {i}"),
    _R("LBU16", "000010 ddd bbb oooo", "100100 bbbbb ddddd oooooooooooooooo",
       "d=g b=g o=lbu16", "lbu {d}, {o}({b})"),
    _R("LHU16", "001010 ddd bbb oooo", "100101 bbbbb ddddd oooooooooooooooo", "d=g b=g o=x2",
       "lhu {d}, {o}({b})"),
    _R("LW16", "011010 ddd bbb oooo", "100011 bbbbb ddddd oooooooooooooooo", "d=g b=g o=x4",
       "lw {d}, {o}({b})"),
    _R("LWSP", "010010 ddddd ooooo", "100011 11101 ddddd oooooooooooooooo", "o=x4",
       "lw {d}, {o}($sp)"),
    _R("LWGP", "011001 ddd ooooooo", "100011 11100 ddddd oooooooooooooooo", "d=g o=sx,x4",
       "lw {d}, {o}($gp)"),
    _R("SB16", "100010 ttt bbb oooo", "101000 bbbbb ttttt oooooooooooooooo", "t=z b=g",
       "sb {t}, {o}({b})"),
    _R("SH16", "101010 ttt bbb oooo", "101001 bbbbb ttttt oooooooooooooooo", "t=z b=g o=x2",
       "sh {t}, {o}({b})"),
    _R("SW16", "111010 ttt bbb oooo", "101011 bbbbb ttttt oooooooooooooooo", "t=z b=g o=x4",
       "sw {t}, {o}({b})"),
    _R("SWSP", "110010 ttttt ooooo", "101011 11101 ttttt oooooooooooooooo", "o=x4",
       "sw {t}, {o}($sp)"),
    _R("B16", "110011 oooooooooo", None, "o=sx,x2", "b {T}"),
    _R("BEQZ16", "100011 sss ooooooo", None, "s=g o=sx,x2", "beqz {s}, {T}"),
    _R("BNEZ16", "101011 sss ooooooo", None, "s=g o=sx,x2", "bnez {s}, {T}"),
    _R("LI16", "111011 ddd iiiiiii", "001001 00000 ddddd iiiiiiiiiiiiiiii", "d=g i=li16",
       "li {d}, {i}"),
    _R("MOVEP", "100001 eee ttt sss 0", None, "t=n s=n", "movep {e}, {s}, {t}"),
]

# 32-bit, POOL32A (000000): ALU, shifts, POOL32Axf
_A3 = {"SLLV": ("0000010000", "000000 sssss ttttt ddddd 00000 000100"),
       "SRLV": ("0001010000", "000000 sssss ttttt ddddd 00000 000110"),
       "SRAV": ("0010010000", "000000 sssss ttttt ddddd 00000 000111"),
       "ROTRV": ("0011010000", "000000 sssss ttttt ddddd 00001 000110"),
       "ADD": ("0100010000", "000000 sssss ttttt ddddd 00000 100000"),
       "ADDU": ("0101010000", "000000 sssss ttttt ddddd 00000 100001"),
       "SUB": ("0110010000", "000000 sssss ttttt ddddd 00000 100010"),
       "SUBU": ("0111010000", "000000 sssss ttttt ddddd 00000 100011"),
       "MUL": ("1000010000", "011100 sssss ttttt ddddd 00000 000010"),
       "AND": ("1001010000", "000000 sssss ttttt ddddd 00000 100100"),
       "OR": ("1010010000", "000000 sssss ttttt ddddd 00000 100101"),
       "NOR": ("1011010000", "000000 sssss ttttt ddddd 00000 100111"),
       "XOR": ("1100010000", "000000 sssss ttttt ddddd 00000 100110"),
       "SLT": ("1101010000", "000000 sssss ttttt ddddd 00000 101010"),
       "SLTU": ("1110010000", "000000 sssss ttttt ddddd 00000 101011"),
       "MOVN": ("0000011000", "000000 sssss ttttt ddddd 00000 001011"),
       "MOVZ": ("0001011000", "000000 sssss ttttt ddddd 00000 001010")}
# shifts by sa: rt (25:21) = rs (20:16) shifted
_SH = {"SLL": ("0000000000", "00000000000 sssss ttttt aaaaa 000000"),
       "SRL": ("0001000000", "00000000000 sssss ttttt aaaaa 000010"),
       "SRA": ("0010000000", "00000000000 sssss ttttt aaaaa 000011"),
       "ROTR": ("0011000000", "00000000001 sssss ttttt aaaaa 000010")}
# POOL32Axf with rt, rs: minor (bits 15:6), twin
_AXF = {"MULT": ("1000101100", "000000 sssss ttttt 00000 00000 011000"),
        "MULTU": ("1001101100", "000000 sssss ttttt 00000 00000 011001"),
        "DIV": ("1010101100", "000000 sssss ttttt 00000 00000 011010"),
        "DIVU": ("1011101100", "000000 sssss ttttt 00000 00000 011011"),
        "MADD": ("1100101100", "011100 sssss ttttt 00000 00000 000000"),
        "MADDU": ("1101101100", "011100 sssss ttttt 00000 00000 000001"),
        "MSUB": ("1110101100", "011100 sssss ttttt 00000 00000 000100"),
        "MSUBU": ("1111101100", "011100 sssss ttttt 00000 00000 000101"),
        "SEB": ("0010101100", "01111100000 sssss ttttt 10000 100000"),
        "SEH": ("0011101100", "01111100000 sssss ttttt 11000 100000"),
        "CLO": ("0100101100", "011100 sssss ttttt ttttt 00000 100001"),
        "CLZ": ("0101101100", "011100 sssss ttttt ttttt 00000 100000"),
        "RDHWR": ("0110101100", "01111100000 ttttt sssss 00000 111011"),
        "WSBH": ("0111101100", "01111100000 sssss ttttt 00010 100000"),
        "MFC2": ("0100110100", "01001000000 ttttt sssss 00000000000"),
        "MTC2": ("0101110100", "01001000100 ttttt sssss 00000000000"),
        "MFHC2": ("1000110100", "01001000011 ttttt sssss 00000000000"),
        "MTHC2": ("1001110100", "01001000111 ttttt sssss 00000000000"),
        "CFC2": ("1100110100", "01001000010 ttttt sssss 00000000000"),
        "CTC2": ("1101110100", "01001000110 ttttt sssss 00000000000"),
        "RDPGPR": ("1110000101", "01000001010 sssss ttttt 00000000000"),
        "WRPGPR": ("1111000101", "01000001110 sssss ttttt 00000000000")}
# POOL32Axf with no register / rs only (bits 25:21 = 0)
_AXF0 = {"TLBP": ("0000001101", "01000010000000000000000000001000"),
         "TLBR": ("0001001101", "01000010000000000000000000000001"),
         "TLBWI": ("0010001101", "01000010000000000000000000000010"),
         "TLBWR": ("0011001101", "01000010000000000000000000000110"),
         "TLBINV": ("0100001101", "01000010000000000000000000000011"),
         "TLBINVF": ("0101001101", "01000010000000000000000000000100"),
         "DERET": ("1110001101", "01000010000000000000000000011111"),
         "ERET": ("1111001101", "01000010000000000000000000011000")}
_AXF1 = {"MFHI": ("0000110101", "0000000000000000 sssss 00000 010000"),
         "MFLO": ("0001110101", "0000000000000000 sssss 00000 010010"),
         "MTHI": ("0010110101", "000000 sssss 000000000000000 010001"),
         "MTLO": ("0011110101", "000000 sssss 000000000000000 010011"),
         "DI": ("0100011101", "01000001011 sssss 0110000000000000"),
         "EI": ("0101011101", "01000001011 sssss 0110000000100000"),
         "SYNC": ("0110101101", "000000000000000000000 sssss 001111")}
_TRAPS = {"TEQ": ("000000", "110100"), "TGE": ("001000", "110000"),
          "TGEU": ("010000", "110001"), "TLT": ("100000", "110010"),
          "TLTU": ("101000", "110011"), "TNE": ("110000", "110110")}
# the 16-bit-immediate major opcodes: microMIPS major, MIPS32 opcode
_IMM = {"ADDI": ("000100", "001000"), "ADDIU": ("001100", "001001"),
        "ORI": ("010100", "001101"), "XORI": ("011100", "001110"),
        "SLTI": ("100100", "001010"), "SLTIU": ("101100", "001011"),
        "ANDI": ("110100", "001100")}
_MEM = {"LBU": ("000101", "100100"), "LHU": ("001101", "100101"), "LB": ("000111", "100000"),
        "LH": ("001111", "100001"), "LW": ("111111", "100011"), "SB": ("000110", "101000"),
        "SH": ("001110", "101001"), "SW": ("111110", "101011")}
_FMEM = {"LWC1": ("100111", "110001"), "LDC1": ("101111", "110101"),
         "SWC1": ("100110", "111001"), "SDC1": ("101110", "111101")}
# POOL32B / POOL32C with a 12-bit offset: bits 15:12, MIPS32 opcode
_MEM12 = {"LWC2": ("001000", "0000", "110010"), "SWC2": ("001000", "1000", "111010"),
          "LDC2": ("001000", "0010", "110110"), "SDC2": ("001000", "1010", "111110"),
          "LWL": ("011000", "0000", "100010"), "SWL": ("011000", "1000", "101010"),
          "LWR": ("011000", "0001", "100110"), "SWR": ("011000", "1001", "101110"),
          "LL": ("011000", "0011", "110000"), "SC": ("011000", "1011", "111000")}
# EVA (LD-EVA / ST-EVA): bits 15:9, the MIPS32 function
_EVA = {"LBUE": ("0110000", "101000"), "LHUE": ("0110001", "101001"),
        "LWLE": ("0110010", "011001"), "LWRE": ("0110011", "011010"),
        "LBE": ("0110100", "101100"), "LHE": ("0110101", "101101"),
        "LLE": ("0110110", "101110"), "LWE": ("0110111", "101111"),
        "SWLE": ("1010000", "100001"), "SWRE": ("1010001", "100010"),
        "SBE": ("1010100", "011100"), "SHE": ("1010101", "011101"),
        "SCE": ("1010110", "011110"), "SWE": ("1010111", "011111")}
# POOL32I: bits 25:21 (rs at 20:16)
_TRAPI = {"TLTI": ("01000", "01010"), "TGEI": ("01001", "01000"),
          "TLTIU": ("01010", "01011"), "TGEIU": ("01011", "01001"),
          "TNEI": ("01100", "01110"), "TEQI": ("01110", "01100")}
#: the branches and jumps (no twin): POOL32I bits 25:21
_BRANCHI = {"BLTZ": "00000", "BLTZAL": "00001", "BGEZ": "00010", "BGEZAL": "00011",
            "BLEZ": "00100", "BNEZC": "00101", "BGTZ": "00110", "BEQZC": "00111",
            "BLTZALS": "10001", "BGEZALS": "10011"}

_PRE32 = [
    *[_R(n, f"000000 ttttt sssss aaaaa 0{m}", t) for n, (m, t) in _SH.items()],
    *[_R(n, f"000000 ttttt sssss ddddd 0{m}", t) for n, (m, t) in _A3.items()],
    _R("LWXS", "000000 ttttt bbbbb ddddd 0 0100011000", None, "", "lwxs {d}, {t}({b})"),
    _R("INS", "000000 ttttt sssss mmmmm ppppp 001100", "011111 sssss ttttt mmmmm ppppp 000100"),
    _R("EXT", "000000 ttttt sssss mmmmm ppppp 101100", "011111 sssss ttttt mmmmm ppppp 000000"),
    _R("BREAK", "000000 cccccccccc qqqqqqqqqq 000111", "000000 cccccccccc qqqqqqqqqq 001101"),
    _R("COP2", "000000 ccccccccccccccccccccccc 010", "0100101 00 ccccccccccccccccccccccc"),
    *[_R(n, f"000000 ttttt sssss {m} 111100", t) for n, (m, t) in _AXF.items()],
    *[_R(n, f"000000 00000 00000 {m} 111100", t) for n, (m, t) in _AXF0.items()],
    *[_R(n, f"000000 00000 sssss {m} 111100", t) for n, (m, t) in _AXF1.items()],
    *[_R(n, f"000000 ttttt sssss cccc {m} 111100",
         f"000000 sssss ttttt 000000 cccc {f}") for n, (m, f) in _TRAPS.items()],
    _R("MFC0", "000000 ttttt sssss 00 lll 00011 111100",
       "01000000000 ttttt sssss 00000000 lll"),
    _R("MTC0", "000000 ttttt sssss 00 lll 01011 111100",
       "01000000100 ttttt sssss 00000000 lll"),
    # (the code: the low bits of MIPS32's 20 / 19-bit field)
    _R("SYSCALL", "000000 cccccccccc 1000101101 111100", "000000 0000000000 cccccccccc 001100"),
    _R("WAIT", "000000 cccccccccc 1001001101 111100", "0100001 000000000 cccccccccc 100000"),
    _R("SDBBP", "000000 cccccccccc 1101101101 111100", "011100 0000000000 cccccccccc 111111"),
    _R("JALR", "000000 ttttt sssss 0000111100 111100", None, "", "jalr {t}, {s}"),
    _R("JALR.HB", "000000 ttttt sssss 0001111100 111100", None, "", "jalr.hb {t}, {s}"),
    _R("JALRS", "000000 ttttt sssss 0100111100 111100", None, "", "jalrs {t}, {s}"),
    _R("JALRS.HB", "000000 ttttt sssss 0101111100 111100", None, "", "jalrs.hb {t}, {s}"),
    # POOL32B (001000) / POOL32C (011000)
    *[_R(n, f"{maj} ttttt bbbbb {m} oooooooooooo", f"{op} bbbbb ttttt oooooooooooooooo", "o=sx")
      for n, (maj, m, op) in _MEM12.items()],
    _R("LWP", "001000 ttttt bbbbb 0001 oooooooooooo", None, "o=sx", "lwp {t}, {o}({b})"),
    _R("SWP", "001000 ttttt bbbbb 1001 oooooooooooo", None, "o=sx", "swp {t}, {o}({b})"),
    _R("LWM32", "001000 nnnnn bbbbb 0101 oooooooooooo", None, "o=sx", "lwm {L}, {o}({b})"),
    _R("SWM32", "001000 nnnnn bbbbb 1101 oooooooooooo", None, "o=sx", "swm {L}, {o}({b})"),
    _R("CACHE", "001000 hhhhh bbbbb 0110 oooooooooooo", "101111 bbbbb hhhhh oooooooooooooooo",
       "o=sx"),
    _R("PREF", "011000 hhhhh bbbbb 0010 oooooooooooo", "110011 bbbbb hhhhh oooooooooooooooo",
       "o=sx"),
    *[_R(n, f"011000 ttttt bbbbb {m} ooooooooo", f"011111 bbbbb ttttt ooooooooo 0 {f}", "o=sx")
      for n, (m, f) in _EVA.items()],
    _R("PREFE", "011000 hhhhh bbbbb 1010010 ooooooooo", "011111 bbbbb hhhhh ooooooooo 0 100011",
       "o=sx"),
    _R("CACHEE", "011000 hhhhh bbbbb 1010011 ooooooooo", "011111 bbbbb hhhhh ooooooooo 0 011011",
       "o=sx"),
    # POOL32I (010000)
    *[_R(n, f"010000 {m} sssss oooooooooooooooo", None, "o=sx,x2") for n, m in _BRANCHI.items()],
    *[_R(n, f"010000 {m} sssss iiiiiiiiiiiiiiii", f"000001 sssss {f} iiiiiiiiiiiiiiii")
      for n, (m, f) in _TRAPI.items()],
    _R("LUI", "010000 01101 sssss iiiiiiiiiiiiiiii", "00111100000 sssss iiiiiiiiiiiiiiii"),
    _R("SYNCI", "010000 10000 bbbbb oooooooooooooooo", "000001 bbbbb 11111 oooooooooooooooo",
       "o=sx"),
    _R("BC2F", "010000 10100 ccc 00 oooooooooooooooo", None, "o=sx,x2"),
    _R("BC2T", "010000 10101 ccc 00 oooooooooooooooo", None, "o=sx,x2"),
    _R("BC1F", "010000 11100 ccc 00 oooooooooooooooo", None, "o=sx,x2"),
    _R("BC1T", "010000 11101 ccc 00 oooooooooooooooo", None, "o=sx,x2"),
    # the major opcodes
    *[_R(n, f"{maj} ttttt sssss iiiiiiiiiiiiiiii", f"{op} sssss ttttt iiiiiiiiiiiiiiii",
         "" if n in ("ORI", "XORI", "ANDI") else "i=sx") for n, (maj, op) in _IMM.items()],
    *[_R(n, f"{maj} ttttt bbbbb oooooooooooooooo", f"{op} bbbbb ttttt oooooooooooooooo", "o=sx")
      for n, (maj, op) in {**_MEM, **_FMEM}.items()],
    _R("BEQ", "100101 ttttt sssss oooooooooooooooo", None, "o=sx,x2"),
    _R("BNE", "101101 ttttt sssss oooooooooooooooo", None, "o=sx,x2"),
    _R("J", "110101 jjjjjjjjjjjjjjjjjjjjjjjjjj", None, "j=x2", "j {J}"),
    _R("JAL", "111101 jjjjjjjjjjjjjjjjjjjjjjjjjj", None, "j=x2", "jal {J}"),
    _R("JALS", "011101 jjjjjjjjjjjjjjjjjjjjjjjjjj", None, "j=x2", "jals {J}"),
    _R("JALX", "111100 jjjjjjjjjjjjjjjjjjjjjjjjjj", None, "j=x4", "jalx {J}"),
    _R("ADDIUPC", "011110 ttt iiiiiiiiiiiiiiiiiiiiiii", None, "t=g i=sx,x4",
       "addiu {t}, $pc, {i}"),
]

# POOL32F (010101): the FPU.  fmt S / D / PS, the MIPS32 fmt field
_FMT3 = {"S": ("00", "10000"), "D": ("01", "10001"), "PS": ("10", "10110")}
#: the conditions of C.cond.fmt (Table 5.17 / 5.18 order)
CONDS = ("F", "UN", "EQ", "UEQ", "OLT", "ULT", "OLE", "ULE", "SF", "NGLE", "SEQ", "NGL",
         "LT", "NGE", "LE", "NGT")
# fd, fs, ft (three registers): bits 10:6 (fmt in 9:8), the MIPS32 function
_F3 = {"ADD": ("110000", "000000"), "SUB": ("110000", "000001"),
       "MUL": ("110000", "000010"), "DIV": ("110000", "000011")}
_F3_MINOR = {"ADD": "00", "SUB": "01", "MUL": "10", "DIV": "11"}


def _fpu_rows() -> list[Row]:
    rows = []
    # ADD / SUB / MUL / DIV .fmt: fmt at 9:8, op at 7:6, 110000 (PS: no DIV)
    for op, (low, funct) in _F3.items():
        for fmt, (bits, mfmt) in _FMT3.items():
            if (op, fmt) == ("DIV", "PS"):
                continue
            rows.append(_R(f"{op}.{fmt}", f"010101 TTTTT SSSSS DDDDD 0 {bits} {_F3_MINOR[op]} {low}",
                           f"010001 {mfmt} TTTTT SSSSS DDDDD {funct}"))
    # MOVN / MOVZ .fmt: rt at 25:21, fs, fd; fmt at 9:8, 0 / 1 at 6, 111000
    for op, bit, funct in (("MOVN", "0", "010011"), ("MOVZ", "1", "010010")):
        for fmt, (bits, mfmt) in _FMT3.items():
            rows.append(_R(f"{op}.{fmt}", f"010101 ttttt SSSSS DDDDD 0 {bits} 0{bit} 111000",
                           f"010001 {mfmt} ttttt SSSSS DDDDD {funct}"))
    # MOVF / MOVT .fmt: ft = fd at 25:21, fs, cc 15:13; fmt at 10:9, tf at 8
    for op, tf in (("MOVF", "0"), ("MOVT", "1")):
        for fmt, (bits, mfmt) in _FMT3.items():
            rows.append(_R(f"{op}.{fmt}", f"010101 TTTTT SSSSS ccc 00 {bits} 00 {tf} 100000",
                           f"010001 {mfmt} ccc 0{tf} SSSSS TTTTT 010001"))
    # MADD / MSUB / NMADD / NMSUB .fmt: ft fs fd fr, the low 6 bits
    for op, sub, low, funct in (("MADD", "0", "001", "100"), ("MSUB", "1", "001", "101"),
                                ("NMADD", "0", "010", "110"), ("NMSUB", "1", "010", "111")):
        for fmt, f2, f3 in (("S", "00", "000"), ("D", "01", "001"), ("PS", "10", "110")):
            code = sub + f2 + low
            rows.append(_R(f"{op}.{fmt}", f"010101 TTTTT SSSSS DDDDD RRRRR {code}",
                           f"010011 RRRRR TTTTT SSSSS DDDDD {funct}{f3}"))
    rows.append(_R("ALNV.PS", "010101 TTTTT SSSSS DDDDD xxxxx 011001",
                   "010011 xxxxx TTTTT SSSSS DDDDD 011110"))
    for op, m, funct in (("PLL.PS", "0010000000", "101100"), ("PLU.PS", "0011000000", "101101"),
                         ("PUL.PS", "0100000000", "101110"), ("PUU.PS", "0101000000", "101111")):
        rows.append(_R(op, f"010101 TTTTT SSSSS DDDDD 0{m}",
                       f"01000110110 TTTTT SSSSS DDDDD {funct}"))
    rows.append(_R("CVT.PS.S", "010101 TTTTT SSSSS DDDDD 00110000000",
                   "01000110000 TTTTT SSSSS DDDDD 100110"))
    # indexed loads / stores: index rt at 25:21, base 20:16, fd / fs 15:11
    for op, m, low in (("LWXC1", "00001001000", "00000 DDDDD 000000"),
                       ("LDXC1", "00011001000", "00000 DDDDD 000001"),
                       ("LUXC1", "00101001000", "00000 DDDDD 000101"),
                       ("SWXC1", "00010001000", "DDDDD 00000 001000"),
                       ("SDXC1", "00100001000", "DDDDD 00000 001001"),
                       ("SUXC1", "00110001000", "DDDDD 00000 001101")):
        rows.append(_R(op, f"010101 ttttt bbbbb DDDDD {m}", f"010011 bbbbb ttttt {low}"))
    rows.append(_R("PREFX", "010101 ttttt bbbbb hhhhh 00110100000",
                   "010011 bbbbb ttttt hhhhh 00000 001111"))
    # C.cond.fmt: ft fs, cc 15:13, fmt 11:10, cond 9:6
    for k, cond in enumerate(CONDS):
        for fmt, (bits, mfmt) in _FMT3.items():
            rows.append(_R(f"C.{cond}.{fmt}", f"010101 TTTTT SSSSS ccc 0 {bits} {k:04b} 111100",
                           f"010001 {mfmt} TTTTT SSSSS ccc 00 11 {k:04b}"))
    # POOL32Fxf two-register forms: ft (= the destination) 25:21, fs 20:16,
    # the 10-bit minor at 15:6, 111011 / 1111011
    one = {  # name: (minor 15:6 with fmt bits, MIPS32 fmt, function)
        "MOV.S": ("0000000001", "10000", "000110"), "MOV.D": ("0010000001", "10001", "000110"),
        "MOV.PS": ("0100000001", "10110", "000110"),
        "ABS.S": ("0000001101", "10000", "000101"), "ABS.D": ("0010001101", "10001", "000101"),
        "ABS.PS": ("0100001101", "10110", "000101"),
        "NEG.S": ("0000101101", "10000", "000111"), "NEG.D": ("0010101101", "10001", "000111"),
        "NEG.PS": ("0100101101", "10110", "000111"),
        "CVT.D.S": ("0001001101", "10000", "100001"), "CVT.D.W": ("0011001101", "10100", "100001"),
        "CVT.D.L": ("0101001101", "10101", "100001"),
        "CVT.S.D": ("0001101101", "10001", "100000"), "CVT.S.W": ("0011101101", "10100", "100000"),
        "CVT.S.L": ("0101101101", "10101", "100000"),
    }
    two = {  # 111011 group (bits 5:0 = 111011, bit 6 = 0)
        "CVT.L.S": ("0000000100", "10000", "100101"), "CVT.L.D": ("0100000100", "10001", "100101"),
        "CVT.W.S": ("0000100100", "10000", "100100"), "CVT.W.D": ("0100100100", "10001", "100100"),
        "RSQRT.S": ("0000001000", "10000", "010110"), "RSQRT.D": ("0100001000", "10001", "010110"),
        "SQRT.S": ("0000101000", "10000", "000100"), "SQRT.D": ("0100101000", "10001", "000100"),
        "RECIP.S": ("0001001000", "10000", "010101"), "RECIP.D": ("0101001000", "10001", "010101"),
        "FLOOR.L.S": ("0000001100", "10000", "001011"),
        "FLOOR.L.D": ("0100001100", "10001", "001011"),
        "FLOOR.W.S": ("0000101100", "10000", "001111"),
        "FLOOR.W.D": ("0100101100", "10001", "001111"),
        "CEIL.L.S": ("0001001100", "10000", "001010"), "CEIL.L.D": ("0101001100", "10001", "001010"),
        "CEIL.W.S": ("0001101100", "10000", "001110"), "CEIL.W.D": ("0101101100", "10001", "001110"),
        "TRUNC.L.S": ("0010001100", "10000", "001001"),
        "TRUNC.L.D": ("0110001100", "10001", "001001"),
        "TRUNC.W.S": ("0010101100", "10000", "001101"),
        "TRUNC.W.D": ("0110101100", "10001", "001101"),
        "ROUND.L.S": ("0011001100", "10000", "001000"),
        "ROUND.L.D": ("0111001100", "10001", "001000"),
        "ROUND.W.S": ("0011101100", "10000", "001100"),
        "ROUND.W.D": ("0111101100", "10001", "001100"),
        "CVT.S.PL": ("0010000100", "10110", "101000"), "CVT.S.PU": ("0010100100", "10110", "100000"),
    }
    for op, (m, mfmt, funct) in one.items():
        rows.append(_R(op, f"010101 TTTTT SSSSS {m} 111011",
                       f"010001 {mfmt} 00000 SSSSS TTTTT {funct}"))
    for op, (m, mfmt, funct) in two.items():
        rows.append(_R(op, f"010101 TTTTT SSSSS {m} 111011",
                       f"010001 {mfmt} 00000 SSSSS TTTTT {funct}"))
    # GPR <-> FPU: rt 25:21, fs 20:16
    for op, m, rsf in (("MFC1", "0010000000", "00000"), ("MTC1", "0010100000", "00100"),
                       ("MFHC1", "0011000000", "00011"), ("MTHC1", "0011100000", "00111"),
                       ("CFC1", "0001000000", "00010"), ("CTC1", "0001100000", "00110")):
        rows.append(_R(op, f"010101 ttttt SSSSS {m} 111011",
                       f"010001 {rsf} ttttt SSSSS 00000000000"))
    # MOVF / MOVT (GPR): rt (dest) 25:21, rs 20:16, cc 15:13
    rows.append(_R("MOVF", "010101 ttttt sssss ccc 0000101111011",
                   "000000 sssss ccc 00 ttttt 00000 000001"))
    rows.append(_R("MOVT", "010101 ttttt sssss ccc 0100101111011",
                   "000000 sssss ccc 01 ttttt 00000 000001"))
    return rows


PRE_ROWS: list[Row] = _PRE16 + _PRE32 + _fpu_rows()


# ------------------------------------------------------------ decoding

#: the 16-bit major opcodes (Table 6.2: bits 12:10 of the first halfword
#: 1, 2 or 3); everything else is 32 bits
def size_of(h: int) -> int:
    """2 or 4: the size of the instruction whose first halfword is ``h``
    (MD00582 ch. 4 / Table 6.2: the major opcode decides)."""
    return 2 if ((h >> 10) & 7) in (1, 2, 3) else 4


class _Table:
    """The rows of one ISA level, by major opcode, most specific first;
    two rows matching one encoding with the same specificity are a build
    error (as in :mod:`emips.isa`)."""

    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows
        self.by_major: dict[int, list[Row]] = {}
        for r in rows:
            major = r.value >> (r.size * 8 - 6)
            self.by_major.setdefault(major, []).append(r)
        for lst in self.by_major.values():
            lst.sort(key=lambda r: -bin(r.mask).count("1"))
            for i, a in enumerate(lst):
                for b in lst[i + 1:]:
                    if a.size == b.size and not (a.value ^ b.value) & a.mask & b.mask and \
                            bin(a.mask).count("1") == bin(b.mask).count("1") and \
                            a.constraint is None and b.constraint is None:
                        raise AssertionError(f"{a.form} and {b.form} overlap")
        self.by_form = {}
        for r in rows:
            self.by_form.setdefault(r.form, r)

    def match(self, raw: int, size: int) -> Row | None:
        major = raw >> (size * 8 - 6)
        for r in self.by_major.get(major, ()):
            if r.size == size and raw & r.mask == r.value and (
                    r.constraint is None or r.constraint(r.extract(raw))):
                return r
        return None


PRE = _Table(PRE_ROWS)


# ------------------------------------------------------------ the Release 6 table
#
# MD00582 rev. 6.05: the 16-bit branches and jumps are compact (BC16,
# BEQZC16, BNEZC16, JRC16, JALRC16, JRCADDIUSP), POOL16C is laid out
# anew (MOVEP moves there), ADDU16 / SUBU16 have rs in 9:7 and rd in 3:1;
# no delay slots, no HI / LO, no branch-likely or FPU condition codes.
# Where the page's diagram is an image (ALIGN, CACHE) QEMU's mips32r6
# decoder (micromips_translate.c.inc) and LLVM agree on the fields.

def _ne0(k):
    return lambda f: f[k] != 0


_R6_16 = [
    _R("ADDU16", "000001 sss ttt ddd 0", "000000 sssss ttttt ddddd 00000 100001", "d=g t=g s=g",
       "addu {d}, {s}, {t}"),
    _R("SUBU16", "000001 sss ttt ddd 1", "000000 sssss ttttt ddddd 00000 100011", "d=g t=g s=g",
       "subu {d}, {s}, {t}"),
    *[r for r in _PRE16 if r.form in (
        "SLL16", "SRL16", "MOVE16", "ANDI16", "ADDIUS5", "ADDIUSP", "ADDIUR2", "ADDIUR1SP",
        "LBU16", "LHU16", "LW16", "LWSP", "LWGP", "SB16", "SH16", "SW16", "SWSP", "LI16")],
    _R("NOT16", "010001 ttt sss 0000", "000000 sssss 00000 ttttt 00000 100111", "t=g s=g",
       "not {t}, {s}"),
    _R("AND16", "010001 ttt sss 0001", "000000 sssss ttttt ttttt 00000 100100", "t=g s=g",
       "and {t}, {t}, {s}"),
    _R("XOR16", "010001 ttt sss 1000", "000000 sssss ttttt ttttt 00000 100110", "t=g s=g",
       "xor {t}, {t}, {s}"),
    _R("OR16", "010001 ttt sss 1001", "000000 sssss ttttt ttttt 00000 100101", "t=g s=g",
       "or {t}, {t}, {s}"),
    _R("LWM16", "010001 nn oooo 0010", None, "o=x4", "lwm {L}, {o}($sp)"),
    _R("SWM16", "010001 nn oooo 1010", None, "o=x4", "swm {L}, {o}($sp)"),
    _R("JRC16", "010001 sssss 00011", None, "", "jrc {s}"),
    _R("JALRC16", "010001 sssss 01011", None, "", "jalrc {s}"),
    _R("JRCADDIUSP", "010001 iiiii 10011", None, "i=x4", "jrcaddiusp {i}"),
    _R("BREAK16", "010001 cccc 011011", "000000 000000cccc 0000000000 001101", "", "break {c}"),
    _R("SDBBP16", "010001 cccc 111011", "000000 0000000000 000000cccc 001110", "", "sdbbp {c}"),
    _R("MOVEP", "010001 eee ttt s 1 ss", None, "t=n s=n", "movep {e}, {s}, {t}"),
    _R("BC16", "110011 oooooooooo", None, "o=sx,x2", "bc {T}"),
    _R("BEQZC16", "100011 sss ooooooo", None, "s=g o=sx,x2", "beqzc {s}, {T}"),
    _R("BNEZC16", "101011 sss ooooooo", None, "s=g o=sx,x2", "bnezc {s}, {T}"),
]

#: POOL32A ALU ops of Release 6 (minor 9:0), the MIPS32 R6 twin
_A3_R6 = {**{n: v for n, v in _A3.items() if n not in ("MUL", "MOVN", "MOVZ")},
          "MUL@R6": ("0000011000", "000000 sssss ttttt ddddd 00010 011000"),
          "MUH": ("0001011000", "000000 sssss ttttt ddddd 00011 011000"),
          "MULU": ("0010011000", "000000 sssss ttttt ddddd 00010 011001"),
          "MUHU": ("0011011000", "000000 sssss ttttt ddddd 00011 011001"),
          "DIV@R6": ("0100011000", "000000 sssss ttttt ddddd 00010 011010"),
          "MOD": ("0101011000", "000000 sssss ttttt ddddd 00011 011010"),
          "DIVU@R6": ("0110011000", "000000 sssss ttttt ddddd 00010 011011"),
          "MODU": ("0111011000", "000000 sssss ttttt ddddd 00011 011011"),
          "SELEQZ": ("0101000000", "000000 sssss ttttt ddddd 00000 110101"),
          "SELNEZ": ("0110000000", "000000 sssss ttttt ddddd 00000 110111")}
_AXF_R6 = {**{n: v for n, v in _AXF.items() if n in (
    "SEB", "SEH", "WSBH", "MFC2", "MTC2", "MFHC2", "MTHC2", "CFC2", "CTC2", "RDPGPR",
    "WRPGPR")},
    "CLO@R6": ("0100101100", "000000 sssss 00000 ttttt 00001 010001"),
    "CLZ@R6": ("0101101100", "000000 sssss 00000 ttttt 00001 010000"),
    "BITSWAP": ("0000101100", "01111100000 ttttt sssss 00000 100000")}
_AXF1_R6 = {n: v for n, v in _AXF1.items() if n in ("DI", "EI", "SYNC")}
#: the compact branches grouped in one opcode: (major, the rs (20:16) /
#: rt (25:21) rule) as QEMU decodes them (MD00582 6.05 B<cond>C)
_POP = {
    "BOVC": ("011101", lambda f: f["s"] >= f["t"]),
    "BEQZALC": ("011101", lambda f: f["s"] == 0 and f["t"] != 0),
    "BEQC": ("011101", lambda f: 0 < f["s"] < f["t"]),
    "BNVC": ("011111", lambda f: f["s"] >= f["t"]),
    "BNEZALC": ("011111", lambda f: f["s"] == 0 and f["t"] != 0),
    "BNEC": ("011111", lambda f: 0 < f["s"] < f["t"]),
    "BLEZALC": ("110000", lambda f: f["s"] == 0 and f["t"] != 0),
    "BGEZALC": ("110000", lambda f: f["s"] != 0 and f["s"] == f["t"]),
    "BGEUC": ("110000", lambda f: not (f["s"] == 0 and f["t"] != 0) and
              not (f["s"] != 0 and f["s"] == f["t"])),
    "BGTZALC": ("111000", lambda f: f["s"] == 0 and f["t"] != 0),
    "BLTZALC": ("111000", lambda f: f["s"] != 0 and f["s"] == f["t"]),
    "BLTUC": ("111000", lambda f: not (f["s"] == 0 and f["t"] != 0) and
              not (f["s"] != 0 and f["s"] == f["t"])),
    "BGTZC": ("110101", lambda f: f["s"] == 0 and f["t"] != 0),
    "BLTZC": ("110101", lambda f: f["s"] != 0 and f["s"] == f["t"]),
    "BLTC": ("110101", lambda f: not (f["s"] == 0 and f["t"] != 0) and
             not (f["s"] != 0 and f["s"] == f["t"])),
    "BLEZC": ("111101", lambda f: f["s"] == 0 and f["t"] != 0),
    "BGEZC": ("111101", lambda f: f["s"] != 0 and f["s"] == f["t"]),
    "BGEC": ("111101", lambda f: not (f["s"] == 0 and f["t"] != 0) and
             not (f["s"] != 0 and f["s"] == f["t"])),
}

_R6_32 = [
    *[_R(n, f"000000 ttttt sssss aaaaa 0{m}", t) for n, (m, t) in _SH.items()],
    *[_R(n, f"000000 ttttt sssss ddddd 0{m}", t) for n, (m, t) in _A3_R6.items()],
    _R("RDHWR@R6", "000000 ttttt sssss 00 lll 0 0111000000", "01111100000 ttttt sssss 00 lll 111011"),
    _R("LSA", "000000 ttttt sssss ddddd aa 000 001111", "000000 sssss ttttt ddddd 000 aa 000101"),
    _R("ALIGN", "000000 ttttt sssss ddddd bb 000 011111", "011111 sssss ttttt ddddd 010 bb 100000"),
    *[r for r in _PRE32 if r.form in ("INS", "EXT", "BREAK", "COP2", "SYSCALL", "WAIT", "MFC0",
                                      "MTC0", "LWP", "SWP", "LWM32", "SWM32", "LUI", "TEQ",
                                      "TGE", "TGEU", "TLT", "TLTU", "TNE", "ADDIU", "ORI", "XORI",
                                      "SLTI", "SLTIU", "ANDI", "LBU", "LHU", "LB", "LH", "LW",
                                      "SB", "SH", "SW", "LWC1", "LDC1", "SWC1", "SDC1", "LBUE",
                                      "LHUE", "LBE", "LHE", "LLE", "LWE", "SBE", "SHE", "SCE",
                                      "SWE", "PREFE", "CACHEE") and r.form != "LUI"],
    _R("SIGRIE", "000000 0000 cccccccccccccccc 111111", "0000010000010111 cccccccccccccccc"),
    # Release 6 multi-threading (MD00582 6.05 DVP / EVP: POOL32Axf, rt in 25:21)
    _R("DVP", "000000 ttttt 00000 0001100101 111100", "01000001011 ttttt 0000000000100100"),
    _R("EVP", "000000 ttttt 00000 0011100101 111100", "01000001011 ttttt 0000000000000100"),
    _R("SDBBP@R6", "000000 cccccccccc 1101101101 111100", "000000 0000000000 cccccccccc 001110"),
    *[_R(n, f"000000 ttttt sssss {m} 111100", t) for n, (m, t) in _AXF_R6.items()],
    *[_R(n, f"000000 00000 00000 {m} 111100", t) for n, (m, t) in _AXF0.items()],
    *[_R(n, f"000000 00000 sssss {m} 111100", t) for n, (m, t) in _AXF1_R6.items()],
    _R("JALRC", "000000 ttttt sssss 0000111100 111100", None, "", "jalrc {t}, {s}"),
    _R("JALRC.HB", "000000 ttttt sssss 0001111100 111100", None, "", "jalrc.hb {t}, {s}"),
    # POOL32B / POOL32C: Release 6's offsets (COP2 11 bits, LL / SC / PREF 9)
    *[_R(n + "@R6", f"001000 ttttt bbbbb {m} 0 ooooooooooo",
         f"01001001{k} ttttt bbbbb ooooooooooo", "o=sx")
      for n, m, k in (("LWC2", "0000", "010"), ("SWC2", "1000", "011"),
                      ("LDC2", "0010", "110"), ("SDC2", "1010", "111"))],
    _R("CACHE", "001000 hhhhh bbbbb 0110 oooooooooooo", "101111 bbbbb hhhhh oooooooooooooooo",
       "o=sx", r5=True),
    _R("PREF@R6", "011000 hhhhh bbbbb 0010 000 ooooooooo", "011111 bbbbb hhhhh ooooooooo 0 110101",
       "o=sx"),
    _R("LL@R6", "011000 ttttt bbbbb 0011 000 ooooooooo", "011111 bbbbb ttttt ooooooooo 0 110110",
       "o=sx"),
    _R("SC@R6", "011000 ttttt bbbbb 1011 000 ooooooooo", "011111 bbbbb ttttt ooooooooo 0 100110",
       "o=sx"),
    # POOL32I
    _R("BC1EQZC", "010000 01000 TTTTT oooooooooooooooo", None, "o=sx,x2"),
    _R("BC1NEZC", "010000 01001 TTTTT oooooooooooooooo", None, "o=sx,x2"),
    _R("BC2EQZC", "010000 01010 TTTTT oooooooooooooooo", None, "o=sx,x2"),
    _R("BC2NEZC", "010000 01011 TTTTT oooooooooooooooo", None, "o=sx,x2"),
    _R("SYNCI", "010000 01100 bbbbb oooooooooooooooo", "000001 bbbbb 11111 oooooooooooooooo",
       "o=sx"),
    # majors
    _R("AUI", "000100 ttttt sssss iiiiiiiiiiiiiiii", "001111 sssss ttttt iiiiiiiiiiiiiiii"),
    _R("ADDIUPC", "011110 ttttt 00 iiiiiiiiiiiiiiiiiii", None, "i=sx,x4", "addiupc {t}, {i}"),
    _R("LWPC", "011110 ttttt 01 ooooooooooooooooooo", None, "o=sx,x4", "lwpc {t}, {o}"),
    _R("AUIPC", "011110 ttttt 11110 iiiiiiiiiiiiiiii", "111011 ttttt 11110 iiiiiiiiiiiiiiii"),
    _R("ALUIPC", "011110 ttttt 11111 iiiiiiiiiiiiiiii", "111011 ttttt 11111 iiiiiiiiiiiiiiii"),
    _R("BC", "100101 oooooooooooooooooooooooooo", None, "o=sx,x2", "bc {T}"),
    _R("BALC", "101101 oooooooooooooooooooooooooo", None, "o=sx,x2", "balc {T}"),
    _R("BEQZC", "100000 sssss ooooooooooooooooooooo", None, "o=sx,x2", when=_ne0("s")),
    _R("JIC", "100000 00000 ttttt iiiiiiiiiiiiiiii", None, "i=sx", "jic {t}, {i}"),
    _R("BNEZC", "101000 sssss ooooooooooooooooooooo", None, "o=sx,x2", when=_ne0("s")),
    _R("JIALC", "101000 00000 ttttt iiiiiiiiiiiiiiii", None, "i=sx", "jialc {t}, {i}"),
    *[_R(n, f"{maj} ttttt sssss oooooooooooooooo", None, "o=sx,x2", when=cond)
      for n, (maj, cond) in _POP.items()],
]
for _r in _R6_32:
    if _r.form in ("BEQZALC", "BNEZALC", "BLEZALC", "BGTZALC", "BGTZC", "BLEZC"):
        object.__setattr__(_r, "syntax", None)


def _fpu_rows_r6() -> list[Row]:
    rows = []
    fmt2 = {"S": ("00", "10000"), "D": ("01", "10001")}
    for op, (low, funct) in _F3.items():
        for fmt, (bits, mfmt) in fmt2.items():
            rows.append(_R(f"{op}.{fmt}", f"010101 TTTTT SSSSS DDDDD 0 {bits} {_F3_MINOR[op]} {low}",
                           f"010001 {mfmt} TTTTT SSSSS DDDDD {funct}"))
    # fd = op(fs, ft): fmt at 9:8, the low 8 bits
    three = {"MADDF": ("110111000", "011000"), "MSUBF": ("111111000", "011001"),
             "SEL": ("010111000", "010000"), "SELEQZ": ("000111000", "010100"),
             "SELNEZ": ("001111000", "010111"), "MIN": ("000000011", "011100"),
             "MAX": ("000001011", "011110"), "MINA": ("000100011", "011101"),
             "MAXA": ("000101011", "011111")}
    for op, (m, funct) in three.items():
        for fmt, (bits, mfmt) in fmt2.items():
            rows.append(_R(f"{op}.{fmt}", f"010101 TTTTT SSSSS DDDDD {bits} {m}",
                           f"010001 {mfmt} TTTTT SSSSS DDDDD {funct}"))
    # fd = op(fs): fs 25:21, fd 20:16
    for op, (m, funct) in {"RINT": ("000100000", "011010"),
                           "CLASS": ("001100000", "011011")}.items():
        for fmt, (bits, mfmt) in fmt2.items():
            rows.append(_R(f"{op}.{fmt}", f"010101 SSSSS DDDDD 00000 {bits} {m}",
                           f"010001 {mfmt} 00000 SSSSS DDDDD {funct}"))
    # CMP.condn.fmt: fd 15:11, condn 10:6
    from .isa import ISA
    for e in ISA.for_release("R6").instructions:
        name = e["name"]
        if name.startswith("CMP.") and name[-2:] in (".S", ".D"):
            cond = int(e["encode"][-5:], 2)
            low = "000101" if name.endswith(".S") else "010101"
            mfmt = "10100" if name.endswith(".S") else "10101"
            rows.append(_R(name, f"010101 TTTTT SSSSS DDDDD {cond:05b} {low}",
                           f"010001 {mfmt} TTTTT SSSSS DDDDD 0 {cond:05b}"))
    keep = ("MOV.", "ABS.", "NEG.", "CVT.", "RSQRT.", "SQRT.", "RECIP.", "FLOOR.", "CEIL.",
            "TRUNC.", "ROUND.", "MFC1", "MTC1", "MFHC1", "MTHC1", "CFC1", "CTC1")
    rows += [r for r in _fpu_rows() if r.form.startswith(keep) and not r.form.endswith(".PS")
             and r.form not in ("CVT.S.PL", "CVT.S.PU", "CVT.PS.S")]
    return rows


R6_ROWS: list[Row] = _R6_16 + _R6_32 + _fpu_rows_r6()
R6 = _Table(R6_ROWS)


# ------------------------------------------------------------ the DSP Module
#
# MD00764 rev. 3.01 (the DSP Module for microMIPS32; pre-Release 6 only: it
# gives no Release 6 encoding), generated from binutils micromips-opc.c's
# D32 / D33 entries: each row's twin is its MIPS32 DSP row (emips.dsp), the
# fields matched by operand position (tests/test_micromips.py checks them
# against objdump and GNU as).  BPOSGE32 has its own executor (a microMIPS
# branch: the offset in halfwords).

_DSP32 = [
    _R("ABSQ_S.PH", "000000 ddddd ttttt 0001000100111100", "01111100000 ttttt ddddd 01001010010"),
    _R("ABSQ_S.QB", "000000 ddddd ttttt 0000000100111100", "01111100000 ttttt ddddd 00001010010"),
    _R("ABSQ_S.W", "000000 ddddd ttttt 0010000100111100", "01111100000 ttttt ddddd 10001010010"),
    _R("ADDQ.PH", "000000 ttttt sssss ddddd 00000001101", "011111 sssss ttttt ddddd 01010010000"),
    _R("ADDQH.PH", "000000 ttttt sssss ddddd 00001001101", "011111 sssss ttttt ddddd 01000011000"),
    _R("ADDQH.W", "000000 ttttt sssss ddddd 00010001101", "011111 sssss ttttt ddddd 10000011000"),
    _R("ADDQH_R.PH", "000000 ttttt sssss ddddd 10001001101", "011111 sssss ttttt ddddd 01010011000"),
    _R("ADDQH_R.W", "000000 ttttt sssss ddddd 10010001101", "011111 sssss ttttt ddddd 10010011000"),
    _R("ADDQ_S.PH", "000000 ttttt sssss ddddd 10000001101", "011111 sssss ttttt ddddd 01110010000"),
    _R("ADDQ_S.W", "000000 ttttt sssss ddddd 01100000101", "011111 sssss ttttt ddddd 10110010000"),
    _R("ADDSC", "000000 ttttt sssss ddddd 01110000101", "011111 sssss ttttt ddddd 10000010000"),
    _R("ADDU.PH", "000000 ttttt sssss ddddd 00100001101", "011111 sssss ttttt ddddd 01000010000"),
    _R("ADDU.QB", "000000 ttttt sssss ddddd 00011001101", "011111 sssss ttttt ddddd 00000010000"),
    _R("ADDUH.QB", "000000 ttttt sssss ddddd 00101001101", "011111 sssss ttttt ddddd 00000011000"),
    _R("ADDUH_R.QB", "000000 ttttt sssss ddddd 10101001101", "011111 sssss ttttt ddddd 00010011000"),
    _R("ADDU_S.PH", "000000 ttttt sssss ddddd 10100001101", "011111 sssss ttttt ddddd 01100010000"),
    _R("ADDU_S.QB", "000000 ttttt sssss ddddd 10011001101", "011111 sssss ttttt ddddd 00100010000"),
    _R("ADDWC", "000000 ttttt sssss ddddd 01111000101", "011111 sssss ttttt ddddd 10001010000"),
    _R("APPEND", "000000 ttttt sssss aaaaa 01000010101", "011111 sssss ttttt aaaaa 00000110001"),
    _R("BALIGN", "000000 ttttt sssss kk 00100010111100", "011111 sssss ttttt 000 kk 10000110001"),
    _R("BITREV", "000000 ddddd ttttt 0011000100111100", "01111100000 ttttt ddddd 11011010010"),
    _R("BPOSGE32", "0100001101100000 oooooooooooooooo", None, "o=sx,x2"),
    _R("CMP.EQ.PH", "000000 ttttt sssss 0000000000000101", "011111 sssss ttttt 0000001000010001"),
    _R("CMP.LE.PH", "000000 ttttt sssss 0000000010000101", "011111 sssss ttttt 0000001010010001"),
    _R("CMP.LT.PH", "000000 ttttt sssss 0000000001000101", "011111 sssss ttttt 0000001001010001"),
    _R("CMPGDU.EQ.QB", "000000 ttttt sssss ddddd 00110000101", "011111 sssss ttttt ddddd 11000010001"),
    _R("CMPGDU.LE.QB", "000000 ttttt sssss ddddd 01000000101", "011111 sssss ttttt ddddd 11010010001"),
    _R("CMPGDU.LT.QB", "000000 ttttt sssss ddddd 00111000101", "011111 sssss ttttt ddddd 11001010001"),
    _R("CMPGU.EQ.QB", "000000 ttttt sssss ddddd 00011000101", "011111 sssss ttttt ddddd 00100010001"),
    _R("CMPGU.LE.QB", "000000 ttttt sssss ddddd 00101000101", "011111 sssss ttttt ddddd 00110010001"),
    _R("CMPGU.LT.QB", "000000 ttttt sssss ddddd 00100000101", "011111 sssss ttttt ddddd 00101010001"),
    _R("CMPU.EQ.QB", "000000 ttttt sssss 0000001001000101", "011111 sssss ttttt 0000000000010001"),
    _R("CMPU.LE.QB", "000000 ttttt sssss 0000001011000101", "011111 sssss ttttt 0000000010010001"),
    _R("CMPU.LT.QB", "000000 ttttt sssss 0000001010000101", "011111 sssss ttttt 0000000001010001"),
    _R("DPA.W.PH", "000000 ttttt sssss AA 00000010111100", "011111 sssss ttttt 000 AA 00000110000"),
    _R("DPAQX_S.W.PH", "000000 ttttt sssss AA 10001010111100", "011111 sssss ttttt 000 AA 11000110000"),
    _R("DPAQX_SA.W.PH", "000000 ttttt sssss AA 11001010111100", "011111 sssss ttttt 000 AA 11010110000"),
    _R("DPAQ_S.W.PH", "000000 ttttt sssss AA 00001010111100", "011111 sssss ttttt 000 AA 00100110000"),
    _R("DPAQ_SA.L.W", "000000 ttttt sssss AA 01001010111100", "011111 sssss ttttt 000 AA 01100110000"),
    _R("DPAU.H.QBL", "000000 ttttt sssss AA 10000010111100", "011111 sssss ttttt 000 AA 00011110000"),
    _R("DPAU.H.QBR", "000000 ttttt sssss AA 11000010111100", "011111 sssss ttttt 000 AA 00111110000"),
    _R("DPAX.W.PH", "000000 ttttt sssss AA 01000010111100", "011111 sssss ttttt 000 AA 01000110000"),
    _R("DPS.W.PH", "000000 ttttt sssss AA 00010010111100", "011111 sssss ttttt 000 AA 00001110000"),
    _R("DPSQX_S.W.PH", "000000 ttttt sssss AA 10011010111100", "011111 sssss ttttt 000 AA 11001110000"),
    _R("DPSQX_SA.W.PH", "000000 ttttt sssss AA 11011010111100", "011111 sssss ttttt 000 AA 11011110000"),
    _R("DPSQ_S.W.PH", "000000 ttttt sssss AA 00011010111100", "011111 sssss ttttt 000 AA 00101110000"),
    _R("DPSQ_SA.L.W", "000000 ttttt sssss AA 01011010111100", "011111 sssss ttttt 000 AA 01101110000"),
    _R("DPSU.H.QBL", "000000 ttttt sssss AA 10010010111100", "011111 sssss ttttt 000 AA 01011110000"),
    _R("DPSU.H.QBR", "000000 ttttt sssss AA 11010010111100", "011111 sssss ttttt 000 AA 01111110000"),
    _R("DPSX.W.PH", "000000 ttttt sssss AA 01010010111100", "011111 sssss ttttt 000 AA 01001110000"),
    _R("EXTP", "000000 ttttt ZZZZZ AA 10011001111100", "011111 ZZZZZ ttttt 000 AA 00010111000"),
    _R("EXTPDP", "000000 ttttt ZZZZZ AA 11011001111100", "011111 ZZZZZ ttttt 000 AA 01010111000"),
    _R("EXTPDPV", "000000 ttttt sssss AA 11100010111100", "011111 sssss ttttt 000 AA 01011111000"),
    _R("EXTPV", "000000 ttttt sssss AA 10100010111100", "011111 sssss ttttt 000 AA 00011111000"),
    _R("EXTR.W", "000000 ttttt fffff AA 00111001111100", "011111 fffff ttttt 000 AA 00000111000"),
    _R("EXTRV.W", "000000 ttttt sssss AA 00111010111100", "011111 sssss ttttt 000 AA 00001111000"),
    _R("EXTRV_R.W", "000000 ttttt sssss AA 01111010111100", "011111 sssss ttttt 000 AA 00101111000"),
    _R("EXTRV_RS.W", "000000 ttttt sssss AA 10111010111100", "011111 sssss ttttt 000 AA 00111111000"),
    _R("EXTRV_S.H", "000000 ttttt sssss AA 11111010111100", "011111 sssss ttttt 000 AA 01111111000"),
    _R("EXTR_R.W", "000000 ttttt fffff AA 01111001111100", "011111 fffff ttttt 000 AA 00100111000"),
    _R("EXTR_RS.W", "000000 ttttt fffff AA 10111001111100", "011111 fffff ttttt 000 AA 00110111000"),
    _R("EXTR_S.H", "000000 ttttt fffff AA 11111001111100", "011111 fffff ttttt 000 AA 01110111000"),
    _R("INSV", "000000 ttttt sssss 0100000100111100", "011111 sssss ttttt 0000000000001100"),
    _R("LBUX", "000000 xxxxx bbbbb ddddd 01000100101", "011111 bbbbb xxxxx ddddd 00110001010"),
    _R("LHX", "000000 xxxxx bbbbb ddddd 00101100101", "011111 bbbbb xxxxx ddddd 00100001010"),
    _R("LWX", "000000 xxxxx bbbbb ddddd 00110100101", "011111 bbbbb xxxxx ddddd 00000001010"),
    _R("MADD@DSP", "000000 ttttt sssss AA 00101010111100", "011100 sssss ttttt 000 AA 00000000000"),
    _R("MADDU@DSP", "000000 ttttt sssss AA 01101010111100", "011100 sssss ttttt 000 AA 00000000001"),
    _R("MAQ_S.W.PHL", "000000 ttttt sssss AA 01101001111100", "011111 sssss ttttt 000 AA 10100110000"),
    _R("MAQ_S.W.PHR", "000000 ttttt sssss AA 00101001111100", "011111 sssss ttttt 000 AA 10110110000"),
    _R("MAQ_SA.W.PHL", "000000 ttttt sssss AA 11101001111100", "011111 sssss ttttt 000 AA 10000110000"),
    _R("MAQ_SA.W.PHR", "000000 ttttt sssss AA 10101001111100", "011111 sssss ttttt 000 AA 10010110000"),
    _R("MFHI@DSP", "00000000000 ddddd AA 00000001111100", "000000000 AA 00000 ddddd 00000010000"),
    _R("MFLO@DSP", "00000000000 ddddd AA 01000001111100", "000000000 AA 00000 ddddd 00000010010"),
    _R("MODSUB", "000000 ttttt sssss ddddd 01010010101", "011111 sssss ttttt ddddd 10010010000"),
    _R("MSUB@DSP", "000000 ttttt sssss AA 10101010111100", "011100 sssss ttttt 000 AA 00000000100"),
    _R("MSUBU@DSP", "000000 ttttt sssss AA 11101010111100", "011100 sssss ttttt 000 AA 00000000101"),
    _R("MTHI@DSP", "00000000000 sssss AA 10000001111100", "000000 sssss 00000000 AA 00000010001"),
    _R("MTHLIP", "00000000000 sssss AA 00001001111100", "011111 sssss 00000000 AA 11111111000"),
    _R("MTLO@DSP", "00000000000 sssss AA 11000001111100", "000000 sssss 00000000 AA 00000010011"),
    _R("MUL.PH", "000000 ttttt sssss ddddd 00000101101", "011111 sssss ttttt ddddd 01100011000"),
    _R("MULEQ_S.W.PHL", "000000 ttttt sssss ddddd 00000100101", "011111 sssss ttttt ddddd 11100010000"),
    _R("MULEQ_S.W.PHR", "000000 ttttt sssss ddddd 00001100101", "011111 sssss ttttt ddddd 11101010000"),
    _R("MULEU_S.PH.QBL", "000000 ttttt sssss ddddd 00010010101", "011111 sssss ttttt ddddd 00110010000"),
    _R("MULEU_S.PH.QBR", "000000 ttttt sssss ddddd 00011010101", "011111 sssss ttttt ddddd 00111010000"),
    _R("MULQ_RS.PH", "000000 ttttt sssss ddddd 00100010101", "011111 sssss ttttt ddddd 11111010000"),
    _R("MULQ_RS.W", "000000 ttttt sssss ddddd 00110010101", "011111 sssss ttttt ddddd 10111011000"),
    _R("MULQ_S.PH", "000000 ttttt sssss ddddd 00101010101", "011111 sssss ttttt ddddd 11110010000"),
    _R("MULQ_S.W", "000000 ttttt sssss ddddd 00111010101", "011111 sssss ttttt ddddd 10110011000"),
    _R("MULSA.W.PH", "000000 ttttt sssss AA 10110010111100", "011111 sssss ttttt 000 AA 00010110000"),
    _R("MULSAQ_S.W.PH", "000000 ttttt sssss AA 11110010111100", "011111 sssss ttttt 000 AA 00110110000"),
    _R("MULT@DSP", "000000 ttttt sssss AA 00110010111100", "000000 sssss ttttt 000 AA 00000011000"),
    _R("MULTU@DSP", "000000 ttttt sssss AA 01110010111100", "000000 sssss ttttt 000 AA 00000011001"),
    _R("MUL_S.PH", "000000 ttttt sssss ddddd 10000101101", "011111 sssss ttttt ddddd 01110011000"),
    _R("PACKRL.PH", "000000 ttttt sssss ddddd 00110101101", "011111 sssss ttttt ddddd 01110010001"),
    _R("PICK.PH", "000000 ttttt sssss ddddd 01000101101", "011111 sssss ttttt ddddd 01011010001"),
    _R("PICK.QB", "000000 ttttt sssss ddddd 00111101101", "011111 sssss ttttt ddddd 00011010001"),
    _R("PRECEQ.W.PHL", "000000 ddddd ttttt 0101000100111100", "01111100000 ttttt ddddd 01100010010"),
    _R("PRECEQ.W.PHR", "000000 ddddd ttttt 0110000100111100", "01111100000 ttttt ddddd 01101010010"),
    _R("PRECEQU.PH.QBL", "000000 ddddd ttttt 0111000100111100", "01111100000 ttttt ddddd 00100010010"),
    _R("PRECEQU.PH.QBLA", "000000 ddddd ttttt 0111001100111100", "01111100000 ttttt ddddd 00110010010"),
    _R("PRECEQU.PH.QBR", "000000 ddddd ttttt 1001000100111100", "01111100000 ttttt ddddd 00101010010"),
    _R("PRECEQU.PH.QBRA", "000000 ddddd ttttt 1001001100111100", "01111100000 ttttt ddddd 00111010010"),
    _R("PRECEU.PH.QBL", "000000 ddddd ttttt 1011000100111100", "01111100000 ttttt ddddd 11100010010"),
    _R("PRECEU.PH.QBLA", "000000 ddddd ttttt 1011001100111100", "01111100000 ttttt ddddd 11110010010"),
    _R("PRECEU.PH.QBR", "000000 ddddd ttttt 1101000100111100", "01111100000 ttttt ddddd 11101010010"),
    _R("PRECEU.PH.QBRA", "000000 ddddd ttttt 1101001100111100", "01111100000 ttttt ddddd 11111010010"),
    _R("PRECR.QB.PH", "000000 ttttt sssss ddddd 00001101101", "011111 sssss ttttt ddddd 01101010001"),
    _R("PRECRQ.PH.W", "000000 ttttt sssss ddddd 00011101101", "011111 sssss ttttt ddddd 10100010001"),
    _R("PRECRQ.QB.PH", "000000 ttttt sssss ddddd 00010101101", "011111 sssss ttttt ddddd 01100010001"),
    _R("PRECRQU_S.QB.PH", "000000 ttttt sssss ddddd 00101101101", "011111 sssss ttttt ddddd 01111010001"),
    _R("PRECRQ_RS.PH.W", "000000 ttttt sssss ddddd 00100101101", "011111 sssss ttttt ddddd 10101010001"),
    _R("PRECR_SRA.PH.W", "000000 ttttt sssss aaaaa 01111001101", "011111 sssss ttttt aaaaa 11110010001"),
    _R("PRECR_SRA_R.PH.W", "000000 ttttt sssss aaaaa 11111001101", "011111 sssss ttttt aaaaa 11111010001"),
    _R("PREPEND", "000000 ttttt sssss aaaaa 01001010101", "011111 sssss ttttt aaaaa 00001110001"),
    _R("RADDU.W.QB", "000000 ddddd sssss 1111000100111100", "011111 sssss 00000 ddddd 10100010000"),
    _R("RDDSP", "000000 ddddd 0 mmmmmm 00011001111100", "011111 mmmmmmmmmm ddddd 10010111000"),
    _R("REPL.PH", "000000 iiiiiiiiii ddddd 00000111101", "011111 iiiiiiiiii ddddd 01010010010"),
    _R("REPL.QB", "000000 ddddd iiiiiiii 0010111111100", "01111100 iiiiiiii ddddd 00010010010"),
    _R("REPLV.PH", "000000 ddddd ttttt 0000001100111100", "01111100000 ttttt ddddd 01011010010"),
    _R("REPLV.QB", "000000 ddddd ttttt 0001001100111100", "01111100000 ttttt ddddd 00011010010"),
    _R("SHILO", "0000000000 ffffff AA 00000000011101", "011111 ffffff 0000000 AA 11010111000"),
    _R("SHILOV", "00000000000 sssss AA 01001001111100", "011111 sssss 00000000 AA 11011111000"),
    _R("SHLL.PH", "000000 ddddd ttttt ffff 001110110101", "0111110 ffff ttttt ddddd 01000010011"),
    _R("SHLL.QB", "000000 ddddd ttttt fff 0100001111100", "01111100 fff ttttt ddddd 00000010011"),
    _R("SHLLV.PH", "000000 ttttt sssss ddddd 01110001101", "011111 sssss ttttt ddddd 01010010011"),
    _R("SHLLV.QB", "000000 ttttt sssss ddddd 01110010101", "011111 sssss ttttt ddddd 00010010011"),
    _R("SHLLV_S.PH", "000000 ttttt sssss ddddd 11110001101", "011111 sssss ttttt ddddd 01110010011"),
    _R("SHLLV_S.W", "000000 ttttt sssss ddddd 01111010101", "011111 sssss ttttt ddddd 10110010011"),
    _R("SHLL_S.PH", "000000 ddddd ttttt ffff 101110110101", "0111110 ffff ttttt ddddd 01100010011"),
    _R("SHLL_S.W", "000000 ddddd ttttt fffff 01111110101", "011111 fffff ttttt ddddd 10100010011"),
    _R("SHRA.PH", "000000 ddddd ttttt ffff 001100110101", "0111110 ffff ttttt ddddd 01001010011"),
    _R("SHRA.QB", "000000 ddddd ttttt fff 0000111111100", "01111100 fff ttttt ddddd 00100010011"),
    _R("SHRAV.PH", "000000 ttttt sssss ddddd 00110001101", "011111 sssss ttttt ddddd 01011010011"),
    _R("SHRAV.QB", "000000 ttttt sssss ddddd 00111001101", "011111 sssss ttttt ddddd 00110010011"),
    _R("SHRAV_R.PH", "000000 ttttt sssss ddddd 10110001101", "011111 sssss ttttt ddddd 01111010011"),
    _R("SHRAV_R.QB", "000000 ttttt sssss ddddd 10111001101", "011111 sssss ttttt ddddd 00111010011"),
    _R("SHRAV_R.W", "000000 ttttt sssss ddddd 01011010101", "011111 sssss ttttt ddddd 10111010011"),
    _R("SHRA_R.PH", "000000 ddddd ttttt ffff 011100110101", "0111110 ffff ttttt ddddd 01101010011"),
    _R("SHRA_R.QB", "000000 ddddd ttttt fff 1000111111100", "01111100 fff ttttt ddddd 00101010011"),
    _R("SHRA_R.W", "000000 ddddd ttttt fffff 01011110101", "011111 fffff ttttt ddddd 10101010011"),
    _R("SHRL.PH", "000000 ddddd ttttt ffff 001111111100", "0111110 ffff ttttt ddddd 11001010011"),
    _R("SHRL.QB", "000000 ddddd ttttt fff 1100001111100", "01111100 fff ttttt ddddd 00001010011"),
    _R("SHRLV.PH", "000000 ttttt sssss ddddd 01100010101", "011111 sssss ttttt ddddd 11011010011"),
    _R("SHRLV.QB", "000000 ttttt sssss ddddd 01101010101", "011111 sssss ttttt ddddd 00011010011"),
    _R("SUBQ.PH", "000000 ttttt sssss ddddd 01000001101", "011111 sssss ttttt ddddd 01011010000"),
    _R("SUBQH.PH", "000000 ttttt sssss ddddd 01001001101", "011111 sssss ttttt ddddd 01001011000"),
    _R("SUBQH.W", "000000 ttttt sssss ddddd 01010001101", "011111 sssss ttttt ddddd 10001011000"),
    _R("SUBQH_R.PH", "000000 ttttt sssss ddddd 11001001101", "011111 sssss ttttt ddddd 01011011000"),
    _R("SUBQH_R.W", "000000 ttttt sssss ddddd 11010001101", "011111 sssss ttttt ddddd 10011011000"),
    _R("SUBQ_S.PH", "000000 ttttt sssss ddddd 11000001101", "011111 sssss ttttt ddddd 01111010000"),
    _R("SUBQ_S.W", "000000 ttttt sssss ddddd 01101000101", "011111 sssss ttttt ddddd 10111010000"),
    _R("SUBU.PH", "000000 ttttt sssss ddddd 01100001101", "011111 sssss ttttt ddddd 01001010000"),
    _R("SUBU.QB", "000000 ttttt sssss ddddd 01011001101", "011111 sssss ttttt ddddd 00001010000"),
    _R("SUBUH.QB", "000000 ttttt sssss ddddd 01101001101", "011111 sssss ttttt ddddd 00001011000"),
    _R("SUBUH_R.QB", "000000 ttttt sssss ddddd 11101001101", "011111 sssss ttttt ddddd 00011011000"),
    _R("SUBU_S.PH", "000000 ttttt sssss ddddd 11100001101", "011111 sssss ttttt ddddd 01101010000"),
    _R("SUBU_S.QB", "000000 ttttt sssss ddddd 11011001101", "011111 sssss ttttt ddddd 00101010000"),
    _R("WRDSP", "000000 sssss 0 mmmmmm 01011001111100", "011111 sssss mmmmmmmmmm 10011111000"),
]
DSP = _Table(_DSP32)
#: microMIPS64's DSP Module (MD00765 rev. 3.02): microMIPS32's rows and
#: LDX (POOL32S), whose twin is MIPS64's LDX (MD00375)
_DSP64 = [_R("LDX", "010110 xxxxx bbbbb ddddd 00110100101",
             "011111 bbbbb xxxxx ddddd 01000001010")]
DSP64 = _Table(_DSP32 + _DSP64)
#: the DSP Module level each form needs (emips.isa.ISA.DSP_LEVEL)
DSP_LEVEL = {r.form: 1 if r.form == "BPOSGE32" else
             ISA.DSP_LEVEL[ISA.for_release("R5", 2).get(r.form)["ase"]] for r in _DSP32}
DSP_LEVEL["LDX"] = 1
# ------------------------------------------------------------ the MT Module
#
# MD00768 rev. 1.12 (Vol. IV-f for microMIPS32) ch. 5: POOL32Axf for DMT /
# EMT / DVPE / EVPE / YIELD, POOL32A minors for FORK / MFTR / MTTR.  The
# 32-bit rows' rt is bits 25:21, rs 20:16 as everywhere in microMIPS:
# YIELD rs, rt writes rs (the MIPS32 rd), MFTR rs, rt writes rs, MTTR rt,
# rs writes the target's rs (the MIPS32 rd).  The FORK diagram labels
# bits 6:0 with a 6-bit field (100111): bit 6 = 0 (bits 10:6 = 0).  No
# binutils / QEMU knows these encodings: they are pinned to the manual.

_MT32 = [
    _R("DMT", "000000 ttttt 00000 0000010101 111100", "01000001011 ttttt 0000101111000001"),
    _R("DVPE", "000000 ttttt 00000 0001010101 111100", "01000001011 ttttt 0000000000000001"),
    _R("EMT", "000000 ttttt 00000 0010010101 111100", "01000001011 ttttt 0000101111100001"),
    _R("EVPE", "000000 ttttt 00000 0011010101 111100", "01000001011 ttttt 0000000000100001"),
    _R("FORK", "000000 ttttt sssss ddddd 00000 100111", "011111 sssss ttttt ddddd 00000 001000"),
    _R("YIELD", "000000 ttttt sssss 00 00100110 111100", "011111 ttttt 00000 sssss 00000 001001"),
    _R("MFTR", "000000 ttttt sssss xxxxx u h 00 lll 1110",
       "01000001000 ttttt sssss xxxxx u h 0 lll"),
    _R("MTTR", "000000 ttttt sssss xxxxx u h 00 lll 0110",
       "01000001100 ttttt sssss xxxxx u h 0 lll"),
]
MT = _Table(_MT32)
#: the MT Module's forms (decoded only by a core with it)
MT_FORMS = frozenset(r.form for r in _MT32)
#: the assembler's pre-Release 6 forms (the DSP / MT Modules' included:
#: the MIPS32 assembler checks .set dsp / mt; decode keeps them apart)
PRE_ASM = _Table(PRE_ROWS + _DSP32 + _MT32)


# ------------------------------------------------------------ microMIPS64
#
# MD00594 (Vol. II-B for microMIPS64) rev. 5.04 / 6.05: the microMIPS32
# instructions plus the doubleword ones — POOL32S (010110: the ALU, shifts,
# bit fields, POOL32Sxf), DADDIU32 (010111), LD32 / SD32 (110111 / 110110),
# POOL32C's LDL ... LWU, POOL32B's LDP / SDP / LDM / SDM, POOL32Fxf's DMFC1
# / DMTC1, POOL32Axf's DMFC2 / DMTC2.  The 16-bit instructions are the
# microMIPS32 ones (32-bit operations, sign-extended).  Every recoded row's
# twin is its MIPS64 row (the MIPS64 table of the core's release decodes
# it); rt is bits 25:21 and rs 20:16 as everywhere in microMIPS.  binutils
# (micromips-opc.c's I3 rows, ``objdump -m mips:micromips``) agrees on the
# pre-Release 6 encodings; Release 6 is pinned to the 6.05 pages (see
# DEVIATIONS "microMIPS64": its POOL32S table and DLSA page disagree).

#: POOL32S three-register forms: the minor (bits 10:0), the MIPS64 twin
_S3 = {"DSLLV": ("00000010000", "000000 sssss ttttt ddddd 00000 010100"),
       "DSRLV": ("00001010000", "000000 sssss ttttt ddddd 00000 010110"),
       "DSRAV": ("00010010000", "000000 sssss ttttt ddddd 00000 010111"),
       "DROTRV": ("00011010000", "000000 sssss ttttt ddddd 00001 010110"),
       "DADD": ("00100010000", "000000 sssss ttttt ddddd 00000 101100"),
       "DADDU": ("00101010000", "000000 sssss ttttt ddddd 00000 101101"),
       "DSUB": ("00110010000", "000000 sssss ttttt ddddd 00000 101110"),
       "DSUBU": ("00111010000", "000000 sssss ttttt ddddd 00000 101111")}
#: POOL32S shifts by sa: rt (25:21) = rs (20:16) shifted
_S_SH = {"DSLL": ("00000000000", "00000000000 sssss ttttt aaaaa 111000"),
         "DSLL32": ("00000001000", "00000000000 sssss ttttt aaaaa 111100"),
         "DSRL": ("00001000000", "00000000000 sssss ttttt aaaaa 111010"),
         "DSRL32": ("00001001000", "00000000000 sssss ttttt aaaaa 111110"),
         "DSRA": ("00010000000", "00000000000 sssss ttttt aaaaa 111011"),
         "DSRA32": ("00010001000", "00000000000 sssss ttttt aaaaa 111111"),
         "DROTR": ("00011000000", "00000000001 sssss ttttt aaaaa 111010"),
         "DROTR32": ("00011001000", "00000000001 sssss ttttt aaaaa 111110")}
#: POOL32S bit fields (bits 5:0), the MIPS64 function
_S_BITS = {"DINSM": ("000100", "000101"), "DINS": ("001100", "000111"),
           "DEXTU": ("010100", "000010"), "DEXTM": ("100100", "000001"),
           "DEXT": ("101100", "000011"), "DINSU": ("110100", "000110")}
#: POOL32Sxf (bits 5:0 = 111100) with rt, rs: minor 15:6, twin
_SXF = {"DSBH": ("0111101100", "01111100000 sssss ttttt 00010 100100"),
        "DSHD": ("1111101100", "01111100000 sssss ttttt 00101 100100")}
_SXF_PRE = {"DCLO": ("0100101100", "011100 sssss ttttt ttttt 00000 100101"),
            "DCLZ": ("0101101100", "011100 sssss ttttt ttttt 00000 100100"),
            "DMULT": ("1000101100", "000000 sssss ttttt 00000 00000 011100"),
            "DMULTU": ("1001101100", "000000 sssss ttttt 00000 00000 011101"),
            "DDIV": ("1010101100", "000000 sssss ttttt 00000 00000 011110"),
            "DDIVU": ("1011101100", "000000 sssss ttttt 00000 00000 011111")}
#: POOL32C with a 12-bit offset: bits 15:12, the MIPS64 opcode
_C12 = {"LDL": ("0100", "011010"), "LDR": ("0101", "011011"), "LLD": ("0111", "110100"),
        "SDL": ("1100", "101100"), "SDR": ("1101", "101101"), "LWU": ("1110", "100111"),
        "SCD": ("1111", "111100")}


def _rows64(pre: bool) -> list[Row]:
    """The microMIPS64 rows added to the microMIPS32 table of a pre-Release
    6 (``pre``) or a Release 6 core."""
    rows = [
        *[_R(n, f"010110 ttttt sssss ddddd {m}", t) for n, (m, t) in _S3.items()],
        *[_R(n, f"010110 ttttt sssss aaaaa {m}", t) for n, (m, t) in _S_SH.items()],
        *[_R(n, f"010110 ttttt sssss mmmmm ppppp {m}", f"011111 sssss ttttt mmmmm ppppp {f}")
          for n, (m, f) in _S_BITS.items()],
        *[_R(n, f"010110 ttttt sssss {m} 111100", t) for n, (m, t) in _SXF.items()],
        _R("DMFC0", "010110 ttttt sssss 00 lll 00011 111100",
           "01000000001 ttttt sssss 00000000 lll"),
        _R("DMTC0", "010110 ttttt sssss 00 lll 01011 111100",
           "01000000101 ttttt sssss 00000000 lll"),
        _R("DMFC1", "010101 ttttt SSSSS 0010010000 111011", "01000100001 ttttt SSSSS 00000000000"),
        _R("DMTC1", "010101 ttttt SSSSS 0010110000 111011", "01000100101 ttttt SSSSS 00000000000"),
        _R("DMFC2", "000000 ttttt sssss 0110110100 111100", "01001000001 ttttt sssss 00000000000"),
        _R("DMTC2", "000000 ttttt sssss 0111110100 111100", "01001000101 ttttt sssss 00000000000"),
        _R("DADDIU", "010111 ttttt sssss iiiiiiiiiiiiiiii", "011001 sssss ttttt iiiiiiiiiiiiiiii",
           "i=sx"),
        _R("LD", "110111 ttttt bbbbb oooooooooooooooo", "110111 bbbbb ttttt oooooooooooooooo",
           "o=sx"),
        _R("SD", "110110 ttttt bbbbb oooooooooooooooo", "111111 bbbbb ttttt oooooooooooooooo",
           "o=sx"),
        _R("LDP", "001000 ttttt bbbbb 0100 oooooooooooo", None, "o=sx", "ldp {t}, {o}({b})"),
        _R("SDP", "001000 ttttt bbbbb 1100 oooooooooooo", None, "o=sx", "sdp {t}, {o}({b})"),
        _R("LDM", "001000 nnnnn bbbbb 0111 oooooooooooo", None, "o=sx", "ldm {L}, {o}({b})"),
        _R("SDM", "001000 nnnnn bbbbb 1111 oooooooooooo", None, "o=sx", "sdm {L}, {o}({b})"),
    ]
    if pre:
        return rows + [
            *[_R(n, f"010110 ttttt sssss {m} 111100", t) for n, (m, t) in _SXF_PRE.items()],
            _R("DADDI", "010110 ttttt sssss iiiiiiiiii 011100",
               "011000 sssss ttttt iiiiiiiiiiiiiiii", "i=sx"),
            *[_R(n, f"011000 ttttt bbbbb {m} oooooooooooo", f"{op} bbbbb ttttt oooooooooooooooo",
                 "o=sx") for n, (m, op) in _C12.items()],
        ]
    # Release 6 (6.05): DMUL ... DMODU, DLSA, DALIGN, DBITSWAP, DCLO / DCLZ
    # (the R6 MIPS64 encodings), DAUI / DAHI / DATI, LDPC / LWUPC; LWU and
    # LLD keep their 12-bit offset (the 6.05 LLD page; MIPS64 R6's LLD has
    # 9 bits: the twin is the Release 5 row), SCD has 9 bits
    md = {"DMUL": ("000011000", "00010 011100"), "DMUH": ("001011000", "00011 011100"),
          "DMULU": ("010011000", "00010 011101"), "DMUHU": ("011011000", "00011 011101"),
          "DDIV@R6": ("100011000", "00010 011110"), "DMOD": ("101011000", "00011 011110"),
          "DDIVU@R6": ("110011000", "00010 011111"), "DMODU": ("111011000", "00011 011111")}
    return rows + [
        *[_R(n, f"010110 ttttt sssss ddddd 00 {m}", f"000000 sssss ttttt ddddd {t}")
          for n, (m, t) in md.items()],
        _R("DLSA", "010110 ttttt sssss ddddd aa 100 000000",
           "000000 sssss ttttt ddddd 000 aa 010101"),
        _R("DALIGN", "010110 ttttt sssss ddddd bbb 00 011100",
           "011111 sssss ttttt ddddd 01 bbb 100100"),
        _R("DBITSWAP", "010110 ttttt sssss 0000101100 111100",
           "01111100000 ttttt sssss 00000 100100"),
        _R("DCLO@R6", "010110 ttttt sssss 0100101100 111100",
           "000000 sssss 00000 ttttt 00001 010011"),
        _R("DCLZ@R6", "010110 ttttt sssss 0101101100 111100",
           "000000 sssss 00000 ttttt 00001 010010"),
        _R("DAUI", "111100 ttttt sssss iiiiiiiiiiiiiiii", "011101 sssss ttttt iiiiiiiiiiiiiiii"),
        _R("DAHI", "010000 10001 sssss iiiiiiiiiiiiiiii", "000001 sssss 00110 iiiiiiiiiiiiiiii"),
        _R("DATI", "010000 10000 sssss iiiiiiiiiiiiiiii", "000001 sssss 11110 iiiiiiiiiiiiiiii"),
        _R("LDPC", "011110 ttttt 110 oooooooooooooooooo", "111011 ttttt 110 oooooooooooooooooo"),
        _R("LWUPC", "011110 ttttt 10 ooooooooooooooooooo", None, "o=sx,x4", "lwupc {t}, {o}"),
        _R("LWU", "011000 ttttt bbbbb 1110 oooooooooooo", "100111 bbbbb ttttt oooooooooooooooo",
           "o=sx"),
        _R("LLD", "011000 ttttt bbbbb 0111 oooooooooooo", "110100 bbbbb ttttt oooooooooooooooo",
           "o=sx", r5=True),
        _R("SCD@R6", "011000 ttttt bbbbb 1111 000 ooooooooo",
           "011111 bbbbb ttttt ooooooooo 0 100111", "o=sx"),
    ]


PRE64_ROWS: list[Row] = PRE_ROWS + _rows64(True)
R6_64_ROWS: list[Row] = R6_ROWS + _rows64(False)
PRE64 = _Table(PRE64_ROWS)
R6_64 = _Table(R6_64_ROWS)
#: the assembler's pre-Release 6 microMIPS64 forms (the DSP Module's:
#: MD00765; the MT Module's: MD00768's encodings, its twins the MIPS64 rows)
PRE64_ASM = _Table(PRE64_ROWS + _DSP32 + _DSP64 + _MT32)
#: the forms only microMIPS64 has
FORMS64 = frozenset(r.form for r in _rows64(True) + _rows64(False))


@dataclass(frozen=True)
class MM:
    """A decoded microMIPS instruction: ``form`` (:class:`Row` name),
    ``size`` 2 / 4, ``raw`` the encoding (a 32-bit one: first halfword in
    bits 31:16), ``f`` the fields after the row's maps."""

    form: str
    size: int
    raw: int
    row: Row = field(compare=False, repr=False)
    f: dict = field(default_factory=dict, compare=False)


class MMDecodeError(ISADecodeError):
    """No microMIPS instruction (the Machine: Reserved Instruction)."""

    def __init__(self, why: str, word: int = 0) -> None:
        Exception.__init__(self, f"no microMIPS instruction: {why}")
        self.word = word
        self.why = why


def table_for(release: str = "R5", arch: str = "mips32") -> _Table:
    """The base table of a microMIPS32 / microMIPS64 core of ``release``."""
    if arch == "mips64":
        return R6_64 if release == "R6" else PRE64
    return R6 if release == "R6" else PRE


def decode(h: int, second: Callable[[], int] | None = None, *, release: str = "R5",
           dsp: int = 0, mt: bool = False, arch: str = "mips32") -> MM:
    """Decode the instruction whose first halfword is ``h``; ``second()``
    gives the next halfword for a 32-bit one.  An encoding that is no
    instruction of this release raises :class:`ISADecodeError`.  ``dsp``:
    the DSP Module level whose encodings decode (MD00764: Releases 2-5);
    ``mt``: the MT Module's (MD00768: Releases 2-5); ``arch`` "mips64":
    microMIPS64 (MD00594)."""
    h &= 0xFFFF
    size = size_of(h)
    if size == 4:
        if second is None:
            raise MMDecodeError("a 32-bit instruction needs its second halfword", h)
        raw = (h << 16) | (second() & 0xFFFF)
    else:
        raw = h
    row = table_for(release, arch).match(raw, size)
    if row is None and dsp and release != "R6" and size == 4:
        row = (DSP64 if arch == "mips64" else DSP).match(raw, size)
        if row is not None and DSP_LEVEL[row.form] > dsp:
            row = None
    if row is None and mt and release != "R6" and size == 4:
        row = MT.match(raw, size)
    if row is None:
        raise MMDecodeError(f"0x{raw:0{size * 2}x}: reserved or not in microMIPS"
                            f"{'64' if arch == 'mips64' else '32'}"
                            f"{' R6' if release == 'R6' else ''}", raw)
    f = row.extract(raw)
    if row.form in ("LWM32", "SWM32", "LDM", "SDM") and not reglist_ok(f["n"]):
        raise MMDecodeError(f"{row.form}: reglist {f['n']:05b} is reserved", raw)
    return MM(row.form, size, raw, row, f)


# ------------------------------------------------------------ the recoder

def expand(i: MM) -> int | None:
    """The MIPS32 twin of ``i`` (the same operation and fields in the
    MIPS32 encoding), or None for the forms with their own executors."""
    tf = i.row.twin_fields
    if tf is None:
        return None
    word, groups = tf
    for letter, v in i.f.items():
        for shift, width in groups.get(letter, ()):
            word |= (v & ((1 << width) - 1)) << shift
    return word


#: the forms without a twin and their family (sim's has_delay_slot / is_cti)
_FAMILY = {
    **dict.fromkeys(("B16", "BEQZ16", "BNEZ16", "BEQ", "BNE", "BLTZ", "BLTZAL", "BGEZ",
                     "BGEZAL", "BLEZ", "BGTZ", "BLTZALS", "BGEZALS"), "branch"),
    **dict.fromkeys(("BC1F", "BC1T"), "branch_fpu"),
    **dict.fromkeys(("BC2F", "BC2T"), "cop2"),     # (no COP2: CpU(2))
    **dict.fromkeys(("BEQZC", "BNEZC"), "branch_compact"),
    **dict.fromkeys(("JR16", "JALR16", "JALRS16", "JALR", "JALR.HB", "JALRS", "JALRS.HB",
                     "J", "JAL", "JALS", "JALX"), "jump"),
    **dict.fromkeys(("JRC", "JRADDIUSP"), "jump_compact"),
    # Release 6
    **dict.fromkeys(("BC16", "BEQZC16", "BNEZC16", "BC", "BALC", "BEQC", "BNEC", "BLTC", "BGEC",
                     "BLTUC", "BGEUC", "BOVC", "BNVC", "BLTZC", "BLEZC", "BGEZC", "BGTZC",
                     "BEQZALC", "BNEZALC", "BLTZALC", "BLEZALC", "BGEZALC", "BGTZALC",
                     "BC1EQZC", "BC1NEZC"), "branch_compact"),
    **dict.fromkeys(("JRC16", "JALRC16", "JALRC", "JALRC.HB", "JIC", "JIALC", "JRCADDIUSP"),
                    "jump_compact"),
    **dict.fromkeys(("BC2EQZC", "BC2NEZC"), "cop2"),
    "LWPC": "load",
    "ADDIUPC": "imm", "LWXS": "load", "LWP": "load", "LWM16": "load", "LWM32": "load",
    "SWP": "store", "SWM16": "store", "SWM32": "store", "MOVEP": "alu",
    "BPOSGE32": "branch",                           # the DSP Module (MD00764)
    "LDP": "load", "LDM": "load", "SDP": "store", "SDM": "store",   # microMIPS64
    "LWUPC": "load",
}
#: the short-slot jumps / branches (a 16-bit delay slot: link = PC + size + 2)
SHORT_SLOT = {"JALS", "JALRS", "JALRS.HB", "JALRS16", "BLTZALS", "BGEZALS"}
ENTRIES = {form: {"name": f"MM.{form}", "mnemonic": form.lower(), "family": fam,
                  "status": "core", "pic32": False, "isa": "micromips",
                  "decode": {"lead": True}, "releases": ["R2", "R3", "R5", "R6"]}
           for form, fam in _FAMILY.items()}
ENTRIES["BPOSGE32"].update(ase="dsp", releases=["R2", "R3", "R5"])   # Status.MX
for _f in ("LDP", "LDM", "SDP", "SDM", "LWUPC"):    # 64-bit operations (MD00594)
    ENTRIES[_f].update(arch="mips64", releases=[], releases64=["R3", "R5", "R6"])
ENTRIES["LWUPC"].update(releases64=["R6"])


def link(i: MM, pc: int) -> int:
    """The return address a linking branch / jump writes: past it and its
    delay slot (16 bits for the S forms, else 32; MD00582 5.4 / 5.5),
    with the ISA mode (1) in bit 0."""
    slot = 2 if i.form in SHORT_SLOT else 4
    return (pc + i.size + slot) | 1


def branch_target(i: MM, pc: int, mask: int = M32) -> int:
    """PC-relative: the instruction after the branch + the offset (the
    field shifted by 1: halfwords); ``mask`` the GPR width."""
    return (pc + i.size + i.f["o"]) & mask


def jump_target(i: MM, pc: int, mask: int = M32) -> int:
    """J / JAL / JALS: the 128 MB region of the delay slot || instr_index
    << 1; JALX: the 256 MB region || target << 2 (MD00582 5.4; MD00594:
    the region is PC 63:27 / 63:28)."""
    keep = 0x0FFFFFFF if i.form == "JALX" else 0x07FFFFFF
    return (((pc + 4) & ~keep) | i.f["j"]) & mask


def reglist(i: MM) -> list[int]:
    """LWM / SWM's registers in memory order (5.4 LWM32 / LWM16)."""
    n = i.f["n"]                                # (LDM / SDM: the same lists, MD00594)
    if i.form in ("LWM16", "SWM16"):
        return [16, 17, 18, 19][:n + 1] + [31]
    regs = list(SREGS[:n & 0xF])
    return regs + [31] if n & 0x10 else regs


def reglist_ok(n: int) -> bool:
    """LWM32 / SWM32's reglist: 1-9 s-registers, with or without ra, or ra
    alone (5.4 LWM32: all other combinations reserved)."""
    return 1 <= n & 0xF <= 9 or n == 0x10


# ------------------------------------------------------------ the disassembler

def _reg(n: int, abi: bool | str) -> str:
    from .syntax import gpr_name
    return gpr_name(n, abi)


def _list_text(regs: list[int], abi: bool) -> str:
    """s0-s2, ra / s0-s7, s8, ra as GNU writes them."""
    s = [r for r in regs if r != 31]
    out = []
    if s:
        if 30 in s:
            body = [r for r in s if r != 30]
            out.append(_reg(16, abi) if len(body) == 1 else f"{_reg(16, abi)}-{_reg(body[-1], abi)}")
            out.append(_reg(30, abi))
        else:
            out.append(_reg(16, abi) if len(s) == 1 else f"{_reg(16, abi)}-{_reg(s[-1], abi)}")
    if 31 in regs:
        out.append(_reg(31, abi))
    return ", ".join(out)


def text(i: MM, pc: int, *, abi: bool | str = True, release: str = "R5",
         arch: str = "mips32") -> str:
    """The instruction in GNU syntax: a recoded one as its MIPS32 (MIPS64)
    twin (``emips.disasm``), the others in binutils ``micromips-opc.c``
    syntax; microMIPS64 names the registers as n64 does."""
    if abi is True and arch == "mips64":
        abi = "n64"
    r = lambda n: _reg(n, abi)               # noqa: E731
    syn = i.row.syntax
    if i.form == "MOVE16" and i.f["d"] == i.f["s"] == 0:
        return "nop"                            # (NOP16, as binutils prints it)
    if syn is None:
        from .disasm import disassemble
        tw = expand(i)
        if tw is None:
            return _special_text(i, pc, r)
        rel = "R6" if release == "R6" and not i.row.r5_twin else "R5"
        out = disassemble(tw, pc, abi=abi, style="gas", dsp=2 if i.form in DSP_LEVEL else 0,
                          mt=i.form in MT_FORMS, release=rel,
                          isa=ISA.for_release(rel, 2 if i.form in DSP_LEVEL else 0,
                                              i.form in MT_FORMS, arch="mips64")
                          if arch == "mips64" else None)
        if i.form.endswith("@DSP") and "$ac" not in out:     # ac0: still the DSP form
            mn, _, ops = out.partition(" ")
            out = (f"{mn} $ac0, {ops}" if mn in ("mult", "multu", "madd", "maddu", "msub",
                                                 "msubu") else f"{out}, $ac0")
        if i.form.startswith(("MOVF", "MOVT")) and "$fcc" not in out:
            out += ", $fcc0"                    # (binutils: the cc is not optional here)
        return out
    f = i.f
    if i.form in ("JALRC", "JALRC.HB"):
        if f["t"] == 0:
            return f"jrc{'.hb' if i.form.endswith('.HB') else ''} {r(f['s'])}"
        if f["t"] == 31:
            return f"{i.form.lower()} {r(f['s'])}"
    if i.form in ("ADDIUPC", "LWPC", "LWUPC") and release == "R6":   # (the address, as R6)
        k = "i" if i.form == "ADDIUPC" else "o"
        return f"{i.form.lower()} {r(f['t'])}, 0x{((pc & ~3) + f[k]) & M32:08x}"
    if i.form in ("JALR", "JALR.HB", "JALRS", "JALRS.HB"):
        if f["t"] == 0 and i.form in ("JALR", "JALR.HB"):
            return f"jr{'.hb' if i.form.endswith('.HB') else ''} {r(f['s'])}"
        if f["t"] == 0:                         # (binutils: JRS = JALRS $0)
            return f"jrs{'.hb' if i.form.endswith('.HB') else ''} {r(f['s'])}"
        if f["t"] == 31:
            return f"{i.form.lower()} {r(f['s'])}"
    if i.form in ("BREAK16", "SDBBP16") and f["c"] == 0:
        return i.form[:-2].lower()
    out = syn
    for key, val in (("{T}", lambda: f"0x{branch_target(i, pc):08x}"),
                     ("{J}", lambda: f"0x{jump_target(i, pc):08x}"),
                     ("{L}", lambda: _list_text(reglist(i), abi)),
                     ("{e}", lambda: ", ".join(r(x) for x in MOVEP_DEST[f["e"]]))):
        if key in out:
            out = out.replace(key, val())
    for letter in ("d", "s", "t", "b"):
        if "{" + letter + "}" in out:
            out = out.replace("{" + letter + "}", r(f[letter]))
    for letter in ("o", "i", "a", "c"):
        if "{" + letter + "}" in out:
            out = out.replace("{" + letter + "}", str(f[letter]))
    return out


def _special_text(i: MM, pc: int, r) -> str:
    f, form = i.f, i.form
    t = f"0x{branch_target(i, pc):08x}" if "o" in f else ""
    alias = {("BEQZC", 0): "bc", ("BGEZAL", 0): "bal", ("BGEZALS", 0): "bals",
             ("BGEZ", 0): "b"}.get((form, f.get("s")))
    if alias:                                   # (binutils' aliases of $zero forms)
        return f"{alias} {t}"
    if form in ("BEQ", "BNE"):
        if f["t"] == 0:
            return ("b " if form == "BEQ" and f["s"] == 0 else
                    f"{form.lower()}z {r(f['s'])}, ") + t
        return f"{form.lower()} {r(f['s'])}, {r(f['t'])}, {t}"
    if form in ("BEQZC", "BNEZC", "BLTZ", "BGEZ", "BLEZ", "BGTZ", "BLTZAL", "BGEZAL",
                "BLTZALS", "BGEZALS"):
        return f"{form.lower()} {r(f['s'])}, 0x{branch_target(i, pc):08x}"
    if form in ("BEQC", "BNEC", "BLTC", "BGEC", "BLTUC", "BGEUC", "BOVC", "BNVC"):
        return f"{form.lower()} {r(f['s'])}, {r(f['t'])}, {t}"
    if form in ("BEQZALC", "BNEZALC", "BLEZALC", "BGEZALC", "BLTZALC", "BGTZALC", "BLTZC",
                "BLEZC", "BGEZC", "BGTZC"):
        return f"{form.lower()} {r(f['t'])}, {t}"
    if form in ("BC1EQZC", "BC1NEZC"):
        return f"{form.lower()} $f{f['T']}, {t}"
    if form in ("BC2EQZC", "BC2NEZC"):
        return f"{form.lower()} ${f['T']}, {t}"
    if form == "BPOSGE32":
        return f"bposge32 {t}"
    if form in ("BC1F", "BC1T", "BC2F", "BC2T"):
        cc = f"$fcc{f['c']}, " if form.startswith("BC1") and f["c"] else \
            (f"$cc{f['c']}, " if f["c"] else "")
        return f"{form.lower()} {cc}0x{branch_target(i, pc):08x}"
    raise AssertionError(form)




# ------------------------------------------------------------ the assembler
#
# GNU as (binutils micromips-opc.c): an instruction takes the first of its
# forms whose operands fit — the 16-bit ones first.  A recoded form is
# found from the MIPS32 encoding of the instruction (the assembler's own
# MIPS32 encoder): the twin read backwards, every field through the
# inverse of its map.

NOP16, NOP32 = 0x0C00, 0x00000000


class AsmErrorMM(ValueError):
    """A microMIPS statement that cannot be encoded (the assembler adds the
    line)."""


class Unresolved(Exception):
    """A label not known yet (pass 1): size the short form."""


#: the maps whose values can be negative (the twin's field is signed)
_SIGNED = {"sx", "lbu16", "li16", "addiusp", "addiur2"}


def _unmap(name: str, v: int, w: int) -> int | None:
    """The inverse of the map ``name`` (None: no encoding)."""
    if name == "g":
        return GPR3.index(v) if v in GPR3 else None
    if name == "z":
        return GPR3_SRC.index(v) if v in GPR3_SRC else None
    if name == "n":
        return MOVEP_SRC.index(v) if v in MOVEP_SRC else None
    if name == "sx":
        return v & ((1 << w) - 1) if -(1 << (w - 1)) <= v < (1 << (w - 1)) else None
    if name in ("x2", "x4"):
        k = 2 if name == "x2" else 4
        return v // k if v % k == 0 else None
    if name == "lbu16":
        return 15 if v == -1 else v if 0 <= v <= 14 else None
    if name == "li16":
        return 127 if v == -1 else v if 0 <= v <= 126 else None
    if name == "sa8":
        return 0 if v == 8 else v if 1 <= v <= 7 else None
    if name == "andi16":
        return ANDI16_IMM.index(v) if v in ANDI16_IMM else None
    if name == "addiur2":
        return ADDIUR2_IMM.index(v) if v in ADDIUR2_IMM else None
    if name == "addiusp":
        if v % 4:
            return None
        k = v // 4
        return (k if 2 <= k <= 255 else k - 256 if k in (256, 257) else
                k + 512 if -256 <= k <= -3 else k + 768 if k in (-258, -257) else None)
    raise AssertionError(name)


def _invert(row: Row, twin: tuple[int, int, dict], word: int) -> int | None:
    """The encoding of ``row`` whose twin is the MIPS32 ``word`` (None:
    the word's fields do not fit this form)."""
    fixed, fmask, groups = twin
    if word & fmask != fixed:
        return None
    raw = row.value
    for letter, runs in row.fields:
        width = sum(w for _, w in runs)
        gs = groups.get(letter, ())
        vals = {(word >> shift) & ((1 << w) - 1) for shift, w in gs}
        if len(vals) > 1:
            return None
        u = vals.pop() if vals else 0
        maps = row.maps.get(letter, ())
        v = _sx(u, gs[0][1]) if gs and any(m in _SIGNED for m in maps) else u
        for name in reversed(maps):
            v = _unmap(name, v, width)
            if v is None:
                return None
        if not 0 <= v < (1 << width):
            return None
        for shift, w in reversed(runs):
            raw |= (v & ((1 << w) - 1)) << shift
            v >>= w
    return raw


def _twin_spec(twin: str) -> tuple[int, int, dict]:
    fixed, groups = _twin_fields(twin)
    fmask = 0
    pos = 32
    for token in twin.split():
        for c in token:
            pos -= 1
            if c in "01":
                fmask |= 1 << pos
    return fixed, fmask, groups


#: MIPS32 mnemonic -> its 16-bit forms in binutils' order: (form, the twin
#: they are read from; None = the row's own)
_ENC16_SPEC: dict[str, list[tuple[str, str | None]]] = {
    "addu": [("MOVE16", "000000 sssss 00000 ddddd 00000 100001"),
             ("MOVE16", "000000 00000 sssss ddddd 00000 100001"), ("ADDU16", None)],
    "or": [("MOVE16", None), ("MOVE16", "000000 00000 sssss ddddd 00000 100101"),
           ("OR16", None), ("OR16", "000000 ttttt sssss ttttt 00000 100101")],
    "subu": [("SUBU16", None)],
    "and": [("AND16", None), ("AND16", "000000 ttttt sssss ttttt 00000 100100")],
    "xor": [("XOR16", None), ("XOR16", "000000 ttttt sssss ttttt 00000 100110")],
    "nor": [("NOT16", "000000 00000 sssss ttttt 00000 100111"), ("NOT16", None)],
    "addiu": [("MOVE16", "001001 sssss ddddd 0000000000000000"), ("ADDIUR1SP", None),
              ("ADDIUR2", None), ("ADDIUSP", None), ("ADDIUS5", None)],
    "ori": [("MOVE16", "001101 sssss ddddd 0000000000000000")],
    "andi": [("ANDI16", None)],
    "sll": [("SLL16", None)], "srl": [("SRL16", None)],
    "lw": [("LW16", None), ("LWSP", None), ("LWGP", None)],
    "sw": [("SW16", None), ("SWSP", None)],
    "sb": [("SB16", None)], "sh": [("SH16", None)],
    "lbu": [("LBU16", None)], "lhu": [("LHU16", None)],
    "mfhi": [("MFHI16", None)], "mflo": [("MFLO16", None)],
    "break": [("BREAK16", None)], "sdbbp": [("SDBBP16", None)],
}


#: MIPS32 alias rows (refinements of another row) -> the row they encode as
_ALIAS_ROWS = {"NOP": "SLL", "SSNOP": "SLL", "EHB": "SLL", "PAUSE": "SLL"}


#: microMIPS64: GNU as's MOVE16 forms of ADDU rd, rs, $0 / ADDIU rd, rs, 0
#: are a 64-bit copy, not ADDU / ADDIU's sign extension of the low word
#: (MD00594 MOVE16, ADDU): eMIPS keeps the 32-bit instruction (DEVIATIONS
#: "microMIPS64")
_NOT_MOVE64 = {"addu", "addiu"}


class _Encoder:
    """The forms of one table, for the assembler."""

    def __init__(self, table: _Table, arch: str = "mips32") -> None:
        self.table = table
        self.by_twin32 = {r.form: (r, _twin_spec(r.twin_source))
                          for r in table.rows if r.size == 4 and r.twin is not None}
        self.enc16 = {}
        for mn, lst in _ENC16_SPEC.items():
            out = []
            for form, tw in lst:
                if arch == "mips64" and mn in _NOT_MOVE64 and form == "MOVE16":
                    continue
                row = table.by_form.get(form)
                if row is not None:
                    out.append((row, _twin_spec(tw if tw is not None else row.twin_source)))
            self.enc16[mn] = out


@dataclass(frozen=True)
class Insn:
    """One encoded instruction: 2 or 4 bytes."""

    size: int
    raw: int

    @property
    def halves(self) -> list[int]:
        return [self.raw >> 16, self.raw & 0xFFFF] if self.size == 4 else [self.raw]


def _gpr(tok: str) -> int:
    from .syntax import parse_gpr
    r = parse_gpr(tok.strip())
    if r is None:
        raise AsmErrorMM(f"register expected, got {tok!r}")
    return r


def _mem(tok: str) -> tuple[str, str]:
    import re
    m = re.fullmatch(r"(.*)\(\s*(\$\w+)\s*\)", tok.strip())
    if not m:
        raise AsmErrorMM(f"offset(base) expected, got {tok!r}")
    return (m.group(1).strip() or "0"), m.group(2)


def _reglist(tokens: list[str]) -> list[int]:
    """``$s0-$s2, $ra`` -> [16, 17, 18, 31]."""
    regs: list[int] = []
    for tok in tokens:
        if "-" in tok:
            a, b = tok.split("-", 1)
            lo, hi = _gpr(a), _gpr(b)
            if hi < lo:
                raise AsmErrorMM(f"bad register range {tok!r}")
            regs.extend(range(lo, hi + 1))
        else:
            regs.append(_gpr(tok))
    return regs


def _fits(v: int, bits: int, scale: int = 1) -> bool:
    return v % scale == 0 and -(1 << (bits - 1)) <= v // scale < (1 << (bits - 1))


class Encoder:
    """GNU as's microMIPS (pre-Release 6): :meth:`encode` gives the
    instruction a (macro-expanded) statement assembles to at ``pc``;
    ``arch`` "mips64": microMIPS64 (MD00594; the MIPS64 assembler's
    encodings read back through the twins)."""

    def __init__(self, release: str = "R5", arch: str = "mips32") -> None:
        self.release = release
        self.arch = arch
        self.table = (R6_64 if release == "R6" else PRE64_ASM) if arch == "mips64" else \
            R6 if release == "R6" else PRE_ASM
        self._enc = _Encoder(self.table, arch)

    def _raw(self, form: str, **fields) -> Insn:
        row = self.table.by_form[form]
        raw = row.value
        for letter, runs in row.fields:
            v = fields.get(letter, 0)
            for shift, w in reversed(runs):
                raw |= (v & ((1 << w) - 1)) << shift
                v >>= w
        return Insn(row.size, raw)

    def encode(self, mn: str, ops: list[str], pc: int, value: Callable[[str], int],
               mips32: Callable[[str, list[str]], int], *, force: int = 0,
               long: bool = False) -> Insn:
        """``force``: the size a delay slot demands (2 / 4, 0 = any);
        ``long``: a branch too far for its 16-bit form (relaxed);
        ``value(expr)`` raises :class:`Unresolved` for a label not known
        yet; ``mips32(mn, ops)`` = the MIPS32 encoding."""
        short_ok = force != 4 and not long
        if mn in ("rddsp", "wrdsp") and len(ops) == 1:
            ops = ops + ["0x3f"]                # (microMIPS: the 6 mask bits, GNU as)
        special = self._special(mn, ops, pc, value, short_ok)
        if special is not None:
            insn = special
        elif mn == "nop" and not ops:
            insn = Insn(2, NOP16) if force != 4 else Insn(4, NOP32)
        else:
            word = mips32(mn, ops)
            insn = None
            if force != 4:
                for row, twin in self._enc.enc16.get(mn, ()):
                    raw = _invert(row, twin, word)
                    if raw is not None:
                        insn = Insn(2, raw)
                        break
            if insn is None:
                from .isa import ISA, ISADecodeError
                try:
                    isa = ISA.for_release(self.release, 2, self.release != "R6",
                                          arch=self.arch)
                    name = isa.decode(word).name
                except ISADecodeError:
                    name = ""
                name = _ALIAS_ROWS.get(name, name)
                base = mn.split("@")[0].upper()
                if f"{base}@DSP" in DSP_LEVEL and any(
                        o.strip().lower().startswith("$ac") for o in ops):
                    name = f"{base}@DSP"         # (also $ac0: its own encoding here)
                if self.release == "R6":
                    name = {"LUI": "AUI"}.get(name, name)
                entry = self._enc.by_twin32.get(name) or self._enc.by_twin32.get(
                    name.split("@")[0])
                raw = _invert(*entry, word) if entry is not None else None
                if raw is None:
                    raise AsmErrorMM(f"{mn}: no microMIPS encoding for these operands"
                                     if entry is not None else
                                     f"{mn} does not exist in microMIPS"
                                     f"{'64' if self.arch == 'mips64' else '32'}")
                insn = Insn(4, raw)
        if force and insn.size != force:
            raise AsmErrorMM(f"{mn}: the delay slot needs a {8 * force}-bit instruction")
        return insn

    # -- branches, jumps, the microMIPS-only instructions ----------------------

    def _offset(self, expr: str, pc: int, size: int, bits: int, value) -> int | None:
        """The branch offset field to ``expr`` from a ``size``-byte branch
        at ``pc`` (None: out of range / odd; Unresolved propagates)."""
        delta = ((value(expr) & M32) - (pc + size)) & M32
        delta = delta - (1 << 32) if delta >> 31 else delta
        if delta & 1 or not -(1 << bits) <= delta < (1 << bits):
            return None
        return (delta >> 1) & ((1 << bits) - 1)

    def _branch(self, form16: str | None, form32: str, regs: dict, expr: str, pc: int,
                value, short_ok: bool, bits16: int = 0) -> Insn:
        try:
            if form16 is not None and short_ok:
                off = self._offset(expr, pc, 2, bits16, value)
                if off is not None:
                    return self._raw(form16, o=off, **regs)
            off = self._offset(expr, pc, 4, 16, value)
        except Unresolved:
            if form16 is not None and short_ok:
                return Insn(2, 0)                   # pass 1: the short form
            return Insn(4, 0)
        if off is None:
            raise AsmErrorMM(f"branch target {expr} out of range (or odd)")
        return self._raw(form32, o=off, **regs)

    def _special(self, mn: str, ops: list[str], pc: int, value, short_ok: bool) -> Insn | None:
        if self.release == "R6":
            return self._special6(mn, ops, pc, value, short_ok)
        n = len(ops)

        def need(k: int) -> None:
            if n != k:
                raise AsmErrorMM(f"{mn} takes {k} operands")
        if mn == "b":
            need(1)
            return self._branch("B16", "BEQ", {}, ops[0], pc, value, short_ok, 10)
        if mn == "bc":
            need(1)
            return self._branch(None, "BEQZC", {}, ops[0], pc, value, False)
        if mn == "bposge32":                    # the DSP Module (MD00764)
            need(1)
            return self._branch(None, "BPOSGE32", {}, ops[0], pc, value, False)
        if mn in ("beqz", "bnez"):
            need(2)
            rs = _gpr(ops[0])
            f16 = ("BEQZ16" if mn == "beqz" else "BNEZ16") if rs in GPR3 else None
            regs16 = {"s": GPR3.index(rs)} if rs in GPR3 else {}
            return self._branch16(f16, regs16, "BEQ" if mn == "beqz" else "BNE",
                                  {"s": rs}, ops[1], pc, value, short_ok)
        if mn in ("beq", "bne"):
            need(3)
            rs, rt = _gpr(ops[0]), _gpr(ops[1])
            z = rs if rt == 0 else rt if rs == 0 else None
            f16 = None
            if z is not None and z in GPR3:
                f16 = "BEQZ16" if mn == "beq" else "BNEZ16"
            regs16 = {"s": GPR3.index(z)} if f16 else {}
            return self._branch16(f16, regs16, mn.upper(), {"s": rs, "t": rt}, ops[2], pc,
                                  value, short_ok)
        if mn in ("beqzc", "bnezc", "bgez", "bltz", "blez", "bgtz", "bgezal", "bltzal",
                  "bgezals", "bltzals"):
            need(2)
            rs = _gpr(ops[0])
            if mn in ("bgezal", "bltzal", "bgezals", "bltzals") and rs == 31:
                raise AsmErrorMM(f"{mn}: rs must not be $ra (the link register)")
            return self._branch(None, mn.upper(), {"s": rs}, ops[1], pc, value, False)
        if mn in ("bal", "bals"):
            need(1)
            return self._branch(None, "BGEZAL" if mn == "bal" else "BGEZALS", {}, ops[0], pc,
                                value, False)
        if mn in ("bc1f", "bc1t", "bc2f", "bc2t"):
            if n not in (1, 2):
                raise AsmErrorMM(f"{mn} takes [cc,] target")
            cc = 0
            if n == 2:
                import re
                m = re.fullmatch(r"\$(?:fcc|cc)?([0-7])", ops[0].strip())
                if not m:
                    raise AsmErrorMM(f"condition code expected, got {ops[0]!r}")
                cc = int(m.group(1))
            return self._branch(None, mn.upper(), {"c": cc}, ops[-1], pc, value, False)
        if mn in ("j", "jal", "jals", "jalx") and n == 1 and not ops[0].strip().startswith("$"):
            try:
                target = value(ops[0]) & M32
            except Unresolved:
                return Insn(4, 0)
            region = 0xF0000000 if mn == "jalx" else 0xF8000000
            if mn == "jalx" and target & 3 or target & 1:
                raise AsmErrorMM(f"{mn}: target 0x{target:08x} not "
                                 f"{'word' if mn == 'jalx' else 'halfword'}-aligned")
            if (target ^ (pc + 4)) & region:
                raise AsmErrorMM(f"{mn}: target 0x{target:08x} outside the "
                                 f"{'256' if mn == 'jalx' else '128'} MB region")
            index = (target & ~region) >> (2 if mn == "jalx" else 1)
            return self._raw(mn.upper(), j=index)
        if mn in ("jr", "j"):
            need(1)
            rs = _gpr(ops[0])
            return self._raw("JR16", s=rs) if short_ok else self._raw("JALR", t=0, s=rs)
        if mn in ("jr.hb", "jrs", "jrs.hb"):
            need(1)
            form = {"jr.hb": "JALR.HB", "jrs": "JALRS", "jrs.hb": "JALRS.HB"}[mn]
            return self._raw(form, t=0, s=_gpr(ops[0]))
        if mn in ("jalr", "jalrs", "jalr.hb", "jalrs.hb", "jal") and n in (1, 2):
            rd, rs = (31, _gpr(ops[0])) if n == 1 else (_gpr(ops[0]), _gpr(ops[1]))
            if rd == rs:
                raise AsmErrorMM(f"{mn}: rd and rs must be different")
            base = "jalr" if mn == "jal" else mn
            if rd == 31 and short_ok and base in ("jalr", "jalrs"):
                return self._raw("JALR16" if base == "jalr" else "JALRS16", s=rs)
            return self._raw(base.upper(), t=rd, s=rs)
        if mn == "jrc":
            need(1)
            return self._raw("JRC", s=_gpr(ops[0]))
        if mn == "jraddiusp":
            need(1)
            v = value(ops[0])
            if v % 4 or not 0 <= v <= 124:
                raise AsmErrorMM("jraddiusp: 0..124, a multiple of 4")
            return self._raw("JRADDIUSP", i=v // 4)
        if mn in ("ldp", "sdp", "ldm", "sdm") and self.arch != "mips64":
            raise AsmErrorMM(f"{mn} is a microMIPS64 instruction (--arch mips64)")
        if mn in ("lwp", "swp", "ldp", "sdp"):
            need(2)
            rd = _gpr(ops[0])
            off, base = _mem(ops[1])
            v = value(off)
            if not _fits(v, 12):
                raise AsmErrorMM(f"{mn}: offset {v} out of range (12 bits)")
            if rd == 31 or mn in ("lwp", "ldp") and rd == _gpr(base):
                raise AsmErrorMM(f"{mn}: rd must not be $ra" + ("" if rd == 31 else
                                                              " or the base register"))
            return self._raw(mn.upper(), t=rd, b=_gpr(base), o=v & 0xFFF)
        if mn in ("ldm", "sdm"):                # (no 16-bit form: MD00594)
            return self._multiple(mn, ops, value, False)
        if mn in ("lwm", "swm"):
            return self._multiple(mn, ops, value, short_ok)
        if mn == "lwxs":
            need(2)
            index, base = _mem(ops[1])
            return self._raw("LWXS", d=_gpr(ops[0]), t=_gpr(index), b=_gpr(base))
        if mn == "movep":
            need(4)
            dest = (_gpr(ops[0]), _gpr(ops[1]))
            src = (_gpr(ops[2]), _gpr(ops[3]))
            if dest not in MOVEP_DEST or not all(r in MOVEP_SRC for r in src):
                raise AsmErrorMM("movep: these registers cannot be encoded")
            return self._raw("MOVEP", e=MOVEP_DEST.index(dest), s=MOVEP_SRC.index(src[0]),
                             t=MOVEP_SRC.index(src[1]))
        if mn == "addiupc" or mn == "addiu" and n == 3 and ops[1].strip() == "$pc":
            rs = _gpr(ops[0])
            v = value(ops[-1])
            if rs not in GPR3 or not _fits(v, 23, 4):
                raise AsmErrorMM("addiupc: $2-$7, $16, $17 and a multiple of 4 within "
                                 "+-16 MB")
            return self._raw("ADDIUPC", t=GPR3.index(rs), i=(v >> 2) & 0x7FFFFF)
        return None

    # -- Release 6 (MD00582 6.05): compact branches and jumps only --------------

    #: register-register compact branches; (signed / unsigned) the names
    #: GNU-style aliases swap to (BGTC rt, rs = BLTC rs, rt ...)
    _RR6 = {"beqc": "BEQC", "bnec": "BNEC", "bltc": "BLTC", "bgec": "BGEC", "bltuc": "BLTUC",
            "bgeuc": "BGEUC", "bovc": "BOVC", "bnvc": "BNVC"}
    _RR6_SWAP = {"bgtc": "BLTC", "blec": "BGEC", "bgtuc": "BLTUC", "bleuc": "BGEUC"}
    #: register-to-zero compact branches (the register in rt, 25:21)
    _RZ6 = {"bltzc": "BLTZC", "blezc": "BLEZC", "bgezc": "BGEZC", "bgtzc": "BGTZC",
            "beqzalc": "BEQZALC", "bnezalc": "BNEZALC", "bltzalc": "BLTZALC",
            "blezalc": "BLEZALC", "bgezalc": "BGEZALC", "bgtzalc": "BGTZALC"}
    #: how each group's encoding puts the registers (rs 20:16, rt 25:21)
    _RZ6_RS = {"BLTZC", "BGEZC", "BGEZALC", "BLTZALC"}      # rs = rt

    def _special6(self, mn: str, ops: list[str], pc: int, value, short_ok: bool) -> Insn | None:
        n = len(ops)

        def need(k: int) -> None:
            if n != k:
                raise AsmErrorMM(f"{mn} takes {k} operands")

        def compact(form16, regs16, form32, regs32, expr, bits16=0, bits32=16) -> Insn:
            try:
                if form16 is not None and short_ok:
                    off = self._offset(expr, pc, 2, bits16, value)
                    if off is not None:
                        return self._raw(form16, o=off, **regs16)
                off = self._offset(expr, pc, 4, bits32, value)
            except Unresolved:
                return Insn(2, 0) if form16 is not None and short_ok else Insn(4, 0)
            if off is None:
                raise AsmErrorMM(f"branch target {expr} out of range (or odd)")
            return self._raw(form32, o=off, **regs32)
        if mn in ("b", "bc"):
            need(1)
            return compact("BC16", {}, "BC", {}, ops[0], 10, 26)
        if mn in ("bal", "balc"):
            need(1)
            return compact(None, {}, "BALC", {}, ops[0], 0, 26)
        if mn in ("beqz", "bnez", "beqzc", "bnezc") or mn in ("beq", "bne") and n == 3 and (
                _gpr(ops[0]) == 0 or _gpr(ops[1]) == 0):
            if mn in ("beq", "bne"):
                reg = ops[1] if _gpr(ops[0]) == 0 else ops[0]
                ops = [reg, ops[2]]
                n = 2
            need(2)
            rs = _gpr(ops[0])
            eq = mn.startswith("beq")
            if rs == 0:
                if eq:
                    return compact("BC16", {}, "BC", {}, ops[1], 10, 26)
                raise AsmErrorMM(f"{mn} $zero: never taken (no encoding)")
            f16 = ("BEQZC16" if eq else "BNEZC16") if rs in GPR3 else None
            regs16 = {"s": GPR3.index(rs)} if f16 else {}
            return compact(f16, regs16, "BEQZC" if eq else "BNEZC", {"s": rs}, ops[1], 7, 21)
        if mn in ("beq", "bne") or mn in self._RR6 or mn in self._RR6_SWAP:
            need(3)
            a, b = _gpr(ops[0]), _gpr(ops[1])
            form = {"beq": "BEQC", "bne": "BNEC"}.get(mn) or self._RR6.get(mn)
            if form is None:
                form, (a, b) = self._RR6_SWAP[mn], (b, a)
            if form in ("BEQC", "BNEC"):
                if a == b or 0 in (a, b):
                    raise AsmErrorMM(f"{mn}: two different registers, not $zero (else "
                                     "BEQZC / BNEZC)")
                s_, t_ = min(a, b), max(a, b)             # rs < rt
            elif form in ("BOVC", "BNVC"):
                s_, t_ = max(a, b), min(a, b)             # rs >= rt
            else:                                         # BLTC ...: rs, rt ordered
                if a == 0 and b != 0 or a != 0 and a == b:
                    raise AsmErrorMM(f"{mn}: rs $zero / rs = rt encode the compare-with-"
                                     "zero branches (BLEZALC, BGEZALC ...)")
                s_, t_ = a, b
            return compact(None, {}, form, {"s": s_, "t": t_}, ops[2])
        if mn in self._RZ6 or mn in ("bgez", "bltz", "blez", "bgtz"):
            need(2)
            form = self._RZ6.get(mn) or self._RZ6[mn + "c"]
            rt = _gpr(ops[0])
            if rt == 0:
                raise AsmErrorMM(f"{mn}: the register must not be $zero")
            regs = {"t": rt, "s": rt if form in self._RZ6_RS else 0}
            return compact(None, {}, form, regs, ops[1])
        if mn in ("bc1eqzc", "bc1nezc", "bc2eqzc", "bc2nezc"):
            need(2)
            import re
            m = re.fullmatch(r"\$f?(\d+)", ops[0].strip())
            if not m or int(m.group(1)) > 31:
                raise AsmErrorMM(f"{mn}: register expected, got {ops[0]!r}")
            return compact(None, {}, mn.upper(), {"T": int(m.group(1))}, ops[1])
        if mn in ("jr", "jrc"):
            need(1)
            rs = _gpr(ops[0])
            return self._raw("JRC16", s=rs) if short_ok else self._raw("JALRC", t=0, s=rs)
        if mn in ("jr.hb", "jrc.hb"):
            need(1)
            return self._raw("JALRC.HB", t=0, s=_gpr(ops[0]))
        if mn in ("jalr", "jalrc", "jalr.hb", "jalrc.hb"):
            if n not in (1, 2):
                raise AsmErrorMM(f"{mn} takes [rd,] rs")
            rd, rs = (31, _gpr(ops[0])) if n == 1 else (_gpr(ops[0]), _gpr(ops[1]))
            if rd == rs:
                raise AsmErrorMM(f"{mn}: rd and rs must be different")
            if not mn.endswith(".hb") and rd == 31 and short_ok:
                return self._raw("JALRC16", s=rs)
            return self._raw("JALRC.HB" if mn.endswith(".hb") else "JALRC", t=rd, s=rs)
        if mn in ("jic", "jialc"):
            need(2)
            v = value(ops[1])
            if not -0x8000 <= v <= 0x7FFF:
                raise AsmErrorMM(f"{mn}: offset {v} out of range (16 bits)")
            return self._raw(mn.upper(), t=_gpr(ops[0]), i=v & 0xFFFF)
        if mn in ("jraddiusp", "jrcaddiusp"):
            need(1)
            v = value(ops[0])
            if v % 4 or not 0 <= v <= 124:
                raise AsmErrorMM(f"{mn}: 0..124, a multiple of 4")
            return self._raw("JRCADDIUSP", i=v // 4)
        if mn in ("addiupc", "lwpc", "lwupc"):            # an address (as MIPS32 R6)
            if mn == "lwupc" and self.arch != "mips64":
                raise AsmErrorMM("lwupc is a microMIPS64 instruction (--arch mips64)")
            need(2)
            try:
                target = value(ops[1]) & M32
            except Unresolved:
                return Insn(4, 0)
            delta = (target - (pc & ~3)) & M32
            delta = delta - (1 << 32) if delta >> 31 else delta
            if not _fits(delta, 19, 4):
                raise AsmErrorMM(f"{mn}: address {ops[1]} not reachable (+-1 MB, x4)")
            k = "i" if mn == "addiupc" else "o"
            return self._raw(mn.upper(), t=_gpr(ops[0]), **{k: (delta >> 2) & 0x7FFFF})
        if mn == "lld" and self.arch == "mips64":         # 6.05 LLD: a 12-bit offset
            need(2)
            off, base = _mem(ops[1])
            v = value(off)
            if not _fits(v, 12):
                raise AsmErrorMM(f"lld: offset {v} out of range (12 bits)")
            return self._raw("LLD", t=_gpr(ops[0]), b=_gpr(base), o=v & 0xFFF)
        if mn == "ldpc" and self.arch == "mips64":        # (PC & ~7) + offset << 3
            need(2)
            try:
                target = value(ops[1]) & M32
            except Unresolved:
                return Insn(4, 0)
            delta = (target - (pc & ~7)) & M32
            delta = delta - (1 << 32) if delta >> 31 else delta
            if not _fits(delta, 18, 8):
                raise AsmErrorMM(f"ldpc: address {ops[1]} not reachable (+-1 MB, x8)")
            return self._raw("LDPC", t=_gpr(ops[0]), o=(delta >> 3) & 0x3FFFF)
        if mn in ("j", "jal", "jals", "jalx", "bgezal", "bltzal", "bals", "bgezals", "bltzals",
                  "beql", "bnel", "jalrs", "jrs"):
            raise AsmErrorMM(f"{mn} does not exist in microMIPS32 Release 6 (no delay slots: "
                             "use the compact branches / jumps)")
        if mn == "cache":                           # (12 bits, as before Release 6)
            need(2)
            off, base = _mem(ops[1])
            op, v = value(ops[0]), value(off)
            if not 0 <= op <= 31 or not _fits(v, 12):
                raise AsmErrorMM("cache: op 0..31, a 12-bit offset")
            return self._raw("CACHE", h=op, b=_gpr(base), o=v & 0xFFF)
        if mn in ("lwp", "swp", "lwm", "swm", "movep", "ldp", "sdp", "ldm", "sdm"):
            r = self.release
            self.release = "R5"                 # (the same syntax; Release 6's own rows)
            try:
                insn = Encoder._special(self, mn, ops, pc, value, short_ok)
            finally:
                self.release = r
            return insn
        return None

    def _multiple(self, mn: str, ops: list[str], value, short_ok: bool) -> Insn:
        """LWM / SWM (the 16-bit form when it fits) and LDM / SDM."""
        if len(ops) < 2:
            raise AsmErrorMM(f"{mn} takes a register list and offset(base)")
        regs = _reglist(ops[:-1])
        off, base = _mem(ops[-1])
        v, b = value(off), _gpr(base)
        s_regs = [r for r in regs if r != 31]
        ra = 31 in regs
        if s_regs != list(SREGS[:len(s_regs)]) or len(regs) != len(set(regs)) or \
                not s_regs and not ra:
            raise AsmErrorMM(f"{mn}: the list must be s0[-s7][, s8][, ra] or ra")
        if mn in ("lwm", "swm") and short_ok and b == 29 and ra and 1 <= len(s_regs) <= 4 \
                and v % 4 == 0 and 0 <= v <= 60:
            return self._raw(mn.upper() + "16", n=len(s_regs) - 1, o=v // 4)
        if not _fits(v, 12):
            raise AsmErrorMM(f"{mn}: offset {v} out of range (12 bits)")
        return self._raw(mn.upper() + ("32" if mn in ("lwm", "swm") else ""),
                         n=len(s_regs) | (0x10 if ra else 0), b=b, o=v & 0xFFF)

    def _branch16(self, f16, regs16, f32, regs32, expr, pc, value, short_ok) -> Insn:
        try:
            if f16 is not None and short_ok:
                off = self._offset(expr, pc, 2, 7, value)
                if off is not None:
                    return self._raw(f16, o=off, **regs16)
            off = self._offset(expr, pc, 4, 16, value)
        except Unresolved:
            return Insn(2, 0) if f16 is not None and short_ok else Insn(4, 0)
        if off is None:
            raise AsmErrorMM(f"branch target {expr} out of range (or odd)")
        return self._raw(f32, o=off, **regs32)

