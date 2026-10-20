"""Assembly syntax of every instruction row (PLAN M4).

One operand specification per ``isaref`` row, shared by the assembler
(:mod:`emips.assembler`) and the disassembler (:mod:`emips.disasm`), so
the two are inverses of each other.  The syntax is the GNU ``as`` / MARS
one: ``add $t0, $t1, $t2``, ``lw $t0, 8($sp)``, ``c.eq.s $fcc1, $f2, $f4``.

An operand spec is a tuple of operand kinds (see :data:`KINDS`); a kind
in ``[...]`` is optional (it may be omitted from the source and gets its
default).  :func:`check_coverage` proves that, for every row, the
operands plus the template's fixed bits cover all 32 bits — so every
encodable word has exactly one textual form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .isa import ISA

M32 = 0xFFFFFFFF

# --------------------------------------------------------------- registers

GPR_NAMES = ["zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
             "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
             "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
             "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"]
_GPR = {n: i for i, n in enumerate(GPR_NAMES)}
_GPR["s8"] = 30


def parse_gpr(tok: str) -> int | None:
    """``$t0`` / ``$8`` / ``$zero`` -> 8 / 8 / 0; ``None`` if not a GPR."""
    t = tok.strip()
    if not t.startswith("$"):
        return None
    t = t[1:]
    if t.isdigit() and int(t) < 32:
        return int(t)
    return _GPR.get(t.lower())


def parse_fpr(tok: str) -> int | None:
    m = re.fullmatch(r"\$f(\d+)", tok.strip(), re.I)
    return int(m.group(1)) if m and int(m.group(1)) < 32 else None


def parse_numreg(tok: str) -> int | None:
    """Numbered register of another file (CP0, FCR, HWR, COP2): ``$12``."""
    m = re.fullmatch(r"\$(\d+)", tok.strip())
    return int(m.group(1)) if m and int(m.group(1)) < 32 else None


def parse_fcc(tok: str) -> int | None:
    """FPU condition code: ``$fcc3`` or a plain ``3``."""
    t = tok.strip()
    m = re.fullmatch(r"\$fcc([0-7])", t, re.I)
    if m:
        return int(m.group(1))
    return int(t) if t.isdigit() and int(t) < 8 else None


def gpr_name(i: int, abi: bool = True) -> str:
    return f"${GPR_NAMES[i]}" if abi else f"${i}"


# ------------------------------------------------------------ operand kinds

@dataclass(frozen=True)
class Kind:
    """An operand kind: what it looks like and which bits it fills."""

    name: str
    #: (lsb, width) bit fields this operand owns, LSB-relative.
    fields: tuple[tuple[int, int], ...]


def _k(name: str, *fields: tuple[int, int]) -> Kind:
    return Kind(name, tuple(fields))


#: Every operand kind.  ``fields`` are (bit 0-based LSB, width).
KINDS: dict[str, Kind] = {k.name: k for k in [
    _k("rs", (21, 5)), _k("rt", (16, 5)), _k("rd", (11, 5)),
    _k("fs", (11, 5)), _k("ft", (16, 5)), _k("fd", (6, 5)), _k("fr", (21, 5)),
    _k("sa", (6, 5)),
    _k("imm", (0, 16)),            # signed 16-bit immediate
    _k("uimm", (0, 16)),           # unsigned 16-bit immediate
    _k("mem", (0, 16), (21, 5)),   # offset(base)
    _k("mem9", (7, 9), (21, 5)),   # EVA: 9-bit offset(base)
    _k("idx", (16, 5), (21, 5)),   # index(base) (COP1X)
    _k("br", (0, 16)),             # PC-relative branch target
    _k("jt", (0, 26)),             # J-type target
    _k("cc18", (18, 3)),           # FPU cc in bits 20..18 (BC1, MOVF)
    _k("cc8", (8, 3)),             # FPU cc in bits 10..8 (C.cond)
    _k("c2cc", (18, 3)),           # COP2 condition (BC2*): $cc3
    _k("code20", (6, 20)),         # SYSCALL / SDBBP code
    _k("code19", (6, 19)),         # WAIT code
    _k("bcode", (6, 20)),          # BREAK code1[, code2] (25..16, 15..6)
    _k("tcode", (6, 10)),          # trap code
    _k("sel", (0, 3)),             # CP0 select
    _k("c0rd", (11, 5)),           # CP0 register number ($12)
    _k("fcr", (11, 5)),            # FPU control register ($31)
    _k("hwr", (11, 5)),            # hardware register (RDHWR)
    _k("c2rt", (16, 5)),           # COP2 register in rt (LWC2 ...)
    _k("impl16", (0, 16)),         # COP2 implementation field
    _k("cofun", (0, 25)),          # COP2 co-function
    _k("hint", (16, 5)),           # CACHE op / PREF hint
    _k("hint11", (11, 5)),         # PREFX hint
    _k("stype", (6, 5)),           # SYNC stype
    _k("pos_ext", (6, 5), (11, 5)),  # EXT: pos -> lsb, size -> msbd
    _k("pos_ins", (6, 5), (11, 5)),  # INS: pos -> lsb, size -> msb
    # Release 6
    _k("br21", (0, 21)),           # compact branch, 21-bit offset
    _k("br26", (0, 26)),           # BC / BALC, 26-bit offset
    _k("rsd", (21, 5)),            # destination GPR in the rs field (PCREL)
    _k("pc19", (0, 19)),           # ADDIUPC / LWPC: an address, PC-relative /4
    _k("sa2", (6, 2)),             # LSA shift 1..4 (stored minus 1)
    _k("bp", (6, 2)),              # ALIGN byte position
    _k("code16", (0, 16)),         # SIGRIE code
    _k("mem11", (0, 11), (11, 5)),  # COP2 R6 loads/stores: offset11(base)
    _k("sel6", (6, 3)),            # R6 RDHWR select
    # the DSP Module (MD00374)
    _k("ac", (11, 2)),             # accumulator $ac0..$ac3 in bits 12..11
    _k("ac21", (21, 2)),           # MFHI / MFLO: the accumulator in bits 22..21
    _k("dsa3", (21, 3)),           # .QB shift amount 0..7
    _k("dsa4", (21, 4)),           # .PH shift amount 0..15
    _k("dsa5", (21, 5)),           # .W shift amount 0..31 (EXTR: the shift)
    _k("dsize", (21, 5)),          # EXTP / EXTPDP: the size 0..31
    _k("shilo", (20, 6)),          # SHILO: signed shift -32..31
    _k("rmask", (16, 10)),         # RDDSP mask (bits 5..0 used)
    _k("wmask", (11, 10)),         # WRDSP mask (bits 5..0 used)
    _k("dimm10", (16, 10)),        # REPL.PH: signed 10-bit immediate
    _k("dimm8", (16, 8)),          # REPL.QB: 8-bit immediate
    _k("dsa", (11, 5)),            # APPEND / PREPEND / PRECR_SRA: sa in 15..11
    _k("bp2", (11, 2)),            # BALIGN byte position
    # the MT Module (MD00378): MFTR / MTTR
    _k("trt", (16, 5)),            # MFTR: the target register number in rt ($2)
    _k("trd", (11, 5)),            # MTTR: the target register number in rd ($2)
    _k("u1", (5, 1)),              # u: 0 = CP0, 1 = the register files
    _k("h1", (4, 1)),              # h: the high half of a 64-bit register
    _k("rx", (6, 5)),              # rx: COP2 register number bits (GNU as: 0)
    # MIPS64 (PLAN M22): the doubleword bit fields (MIPS64 Vol. II DEXT ...
    # DINSU: pos / size -> lsb and msbd / msb, each row its own ranges)
    _k("pos_dext", (6, 5), (11, 5)),   # DEXT:  lsb = pos, msbd = size - 1
    _k("pos_dextm", (6, 5), (11, 5)),  # DEXTM: lsb = pos, msbd = size - 33
    _k("pos_dextu", (6, 5), (11, 5)),  # DEXTU: lsb = pos - 32, msbd = size - 1
    _k("pos_dins", (6, 5), (11, 5)),   # DINS:  lsb = pos, msb = pos + size - 1
    _k("pos_dinsm", (6, 5), (11, 5)),  # DINSM: lsb = pos, msb = pos + size - 33
    _k("pos_dinsu", (6, 5), (11, 5)),  # DINSU: lsb = pos - 32, msb = pos + size - 33
    _k("bp3", (6, 3)),             # DALIGN byte position 0..7
    _k("pc18", (0, 18)),           # LDPC: an address, (PC & ~7) + offset * 8
    _k("rs2"),                     # DAHI / DATI: rs again (the manual's "DAHI rs, rs")
]}


# ----------------------------------------------------- per-row operand specs

_FPU_3 = re.compile(r"(ADD|SUB|MUL|DIV)\.(S|D|PS)|(PLL|PLU|PUL|PUU)\.PS")
_FPU_2 = re.compile(r"(ABS|NEG|MOV|SQRT|RECIP|RSQRT)\.(S|D|PS)"
                    r"|(CVT|ROUND|TRUNC|CEIL|FLOOR)\..*")
_FPU_4 = re.compile(r"N?M(ADD|SUB)\.(S|D|PS)")

_EXPLICIT: dict[str, tuple[str, ...]] = {
    # integer
    "LUI": ("rt", "uimm"),
    "ANDI": ("rt", "rs", "uimm"), "ORI": ("rt", "rs", "uimm"),
    "XORI": ("rt", "rs", "uimm"),
    "SLL": ("rd", "rt", "sa"), "SRL": ("rd", "rt", "sa"),
    "SRA": ("rd", "rt", "sa"), "ROTR": ("rd", "rt", "sa"),
    "SLLV": ("rd", "rt", "rs"), "SRLV": ("rd", "rt", "rs"),
    "SRAV": ("rd", "rt", "rs"), "ROTRV": ("rd", "rt", "rs"),
    "SEB": ("rd", "rt"), "SEH": ("rd", "rt"), "WSBH": ("rd", "rt"),
    "CLO": ("rd", "rs"), "CLZ": ("rd", "rs"),          # rt = rd (manual)
    "EXT": ("rt", "rs", "pos_ext"), "INS": ("rt", "rs", "pos_ins"),
    "MULT": ("rs", "rt"), "MULTU": ("rs", "rt"), "DIV": ("rs", "rt"),
    "DIVU": ("rs", "rt"), "MADD": ("rs", "rt"), "MADDU": ("rs", "rt"),
    "MSUB": ("rs", "rt"), "MSUBU": ("rs", "rt"),
    "MFHI": ("rd",), "MFLO": ("rd",), "MTHI": ("rs",), "MTLO": ("rs",),
    "MOVF": ("rd", "rs", "[cc18]"), "MOVT": ("rd", "rs", "[cc18]"),
    # branches / jumps
    "B": ("br",), "BAL": ("br",), "NAL": ("br",),
    "BEQ": ("rs", "rt", "br"), "BNE": ("rs", "rt", "br"),
    "BEQL": ("rs", "rt", "br"), "BNEL": ("rs", "rt", "br"),
    "J": ("jt",), "JAL": ("jt",), "JALX": ("jt",),
    "JR": ("rs",), "JR.HB": ("rs",),
    "JALR": ("[rd]", "rs"), "JALR.HB": ("[rd]", "rs"),
    # traps / system
    "SYSCALL": ("[code20]",), "SDBBP": ("[code20]",), "BREAK": ("[bcode]",),
    "WAIT": ("[code19]",), "SYNC": ("[stype]",),
    "MFC0": ("rt", "c0rd", "[sel]"), "MTC0": ("rt", "c0rd", "[sel]"),
    "DI": ("[rt]",), "EI": ("[rt]",), "RDHWR": ("rt", "hwr"),
    "RDPGPR": ("rd", "rt"), "WRPGPR": ("rd", "rt"),
    "CACHE": ("hint", "mem"), "PREF": ("hint", "mem"),
    "CACHEE": ("hint", "mem9"), "PREFE": ("hint", "mem9"),
    "PREFX": ("hint11", "idx"), "SYNCI": ("mem",),
    # FPU moves and memory
    "MFC1": ("rt", "fs"), "MTC1": ("rt", "fs"), "MFHC1": ("rt", "fs"),
    "MTHC1": ("rt", "fs"), "CFC1": ("rt", "fcr"), "CTC1": ("rt", "fcr"),
    "LWC1": ("ft", "mem"), "SWC1": ("ft", "mem"), "LDC1": ("ft", "mem"),
    "SDC1": ("ft", "mem"),
    "LWXC1": ("fd", "idx"), "LDXC1": ("fd", "idx"), "LUXC1": ("fd", "idx"),
    "SWXC1": ("fs", "idx"), "SDXC1": ("fs", "idx"), "SUXC1": ("fs", "idx"),
    "BC1F": ("[cc18]", "br"), "BC1T": ("[cc18]", "br"),
    "BC1FL": ("[cc18]", "br"), "BC1TL": ("[cc18]", "br"),
    "ALNV.PS": ("fd", "fs", "ft", "rs"), "CVT.PS.S": ("fd", "fs", "ft"),
    # COP2 (no COP2 on this core; encodings only)
    "COP2": ("cofun",),
    "MFC2": ("rt", "impl16"), "MTC2": ("rt", "impl16"),
    "CFC2": ("rt", "impl16"), "CTC2": ("rt", "impl16"),
    "MFHC2": ("rt", "impl16"), "MTHC2": ("rt", "impl16"),
    "LWC2": ("c2rt", "mem"), "SWC2": ("c2rt", "mem"),
    "LDC2": ("c2rt", "mem"), "SDC2": ("c2rt", "mem"),
    "BC2F": ("[c2cc]", "br"), "BC2T": ("[c2cc]", "br"),
    "BC2FL": ("[c2cc]", "br"), "BC2TL": ("[c2cc]", "br"),
}


_R6: dict[str, tuple[str, ...]] = {
    "BC": ("br26",), "BALC": ("br26",),
    "BEQZC": ("rs", "br21"), "BNEZC": ("rs", "br21"),
    "JIC": ("rt", "imm"), "JIALC": ("rt", "imm"),
    "BOVC": ("rs", "rt", "br"), "BNVC": ("rs", "rt", "br"),
    "BEQC": ("rs", "rt", "br"), "BNEC": ("rs", "rt", "br"),
    "BGEC": ("rs", "rt", "br"), "BLTC": ("rs", "rt", "br"),
    "BGEUC": ("rs", "rt", "br"), "BLTUC": ("rs", "rt", "br"),
    **{n: ("rt", "br") for n in ("BEQZALC", "BNEZALC", "BLEZALC", "BGEZALC",
                                 "BGTZALC", "BLTZALC", "BLEZC", "BGEZC",
                                 "BGTZC", "BLTZC")},
    "BC1EQZ": ("ft", "br"), "BC1NEZ": ("ft", "br"),
    "BC2EQZ": ("c2rt", "br"), "BC2NEZ": ("c2rt", "br"),
    "AUI": ("rt", "rs", "uimm"), "AUIPC": ("rsd", "uimm"), "ALUIPC": ("rsd", "uimm"),
    "ADDIUPC": ("rsd", "pc19"), "LWPC": ("rsd", "pc19"),
    "ALIGN": ("rd", "rs", "rt", "bp"), "BITSWAP": ("rd", "rt"),
    "LSA": ("rd", "rs", "rt", "sa2"),
    "CLZ@R6": ("rd", "rs"), "CLO@R6": ("rd", "rs"),
    "RDHWR@R6": ("rt", "hwr", "[sel6]"),
    "SDBBP@R6": ("[code20]",), "SIGRIE": ("[code16]",),
    "LL@R6": ("rt", "mem9"), "SC@R6": ("rt", "mem9"),
    "CACHE@R6": ("hint", "mem9"), "PREF@R6": ("hint", "mem9"),
    **{f"{n}@R6": ("c2rt", "mem11") for n in ("LWC2", "SWC2", "LDC2", "SDC2")},
    **{f"{n}.{f}": ("fd", "fs") for n in ("RINT", "CLASS") for f in ("S", "D")},
}


#: the MIPS64 rows (PLAN M22, MIPS64 Vol. II): like their MIPS32
#: counterparts; DEXTM / DEXTU / DINSM / DINSU print as dext / dins with the
#: field's real pos and size (the manual: "always specify the DEXT
#: mnemonic"; binutils prints them so)
_MIPS64: dict[str, tuple[str, ...]] = {
    **{n: ("rd", "rt", "sa") for n in ("DSLL", "DSLL32", "DSRL", "DSRL32", "DSRA",
                                       "DSRA32", "DROTR", "DROTR32")},
    **{n: ("rd", "rt", "rs") for n in ("DSLLV", "DSRLV", "DSRAV", "DROTRV")},
    "DSBH": ("rd", "rt"), "DSHD": ("rd", "rt"), "DBITSWAP": ("rd", "rt"),
    "DCLO": ("rd", "rs"), "DCLZ": ("rd", "rs"),          # rt = rd (manual)
    "DCLO@R6": ("rd", "rs"), "DCLZ@R6": ("rd", "rs"),
    "DMULT": ("rs", "rt"), "DMULTU": ("rs", "rt"), "DDIV": ("rs", "rt"),
    "DDIVU": ("rs", "rt"),
    "DEXT": ("rt", "rs", "pos_dext"), "DEXTM": ("rt", "rs", "pos_dextm"),
    "DEXTU": ("rt", "rs", "pos_dextu"), "DINS": ("rt", "rs", "pos_dins"),
    "DINSM": ("rt", "rs", "pos_dinsm"), "DINSU": ("rt", "rs", "pos_dinsu"),
    "DMFC0": ("rt", "c0rd", "[sel]"), "DMTC0": ("rt", "c0rd", "[sel]"),
    "DMFC1": ("rt", "fs"), "DMTC1": ("rt", "fs"),
    "DMFC2": ("rt", "impl16"), "DMTC2": ("rt", "impl16"),
    "DALIGN": ("rd", "rs", "rt", "bp3"), "DAUI": ("rt", "rs", "uimm"),
    "DAHI": ("rs", "rs2", "uimm"), "DATI": ("rs", "rs2", "uimm"),
    "DLSA": ("rd", "rs", "rt", "sa2"),
    "LDPC": ("rsd", "pc18"), "LWUPC": ("rsd", "pc19"),
    "LLD@R6": ("rt", "mem9"), "SCD@R6": ("rt", "mem9"),
}


def _dsp_specs() -> dict[str, tuple[str, ...]]:
    """The DSP Module rows' operands, in the manual's (and GNU as') order."""
    spec: dict[str, tuple[str, ...]] = {}

    def put(kinds: tuple[str, ...], names: str) -> None:
        for n in names.split():
            spec[n] = kinds

    put(("rd", "rs", "rt"), """
        ADDQ.PH ADDQ_S.PH ADDQ_S.W ADDQH.PH ADDQH_R.PH ADDQH.W ADDQH_R.W ADDSC
        ADDU.QB ADDU_S.QB ADDU.PH ADDU_S.PH ADDUH.QB ADDUH_R.QB ADDWC
        SUBQ.PH SUBQ_S.PH SUBQ_S.W SUBQH.PH SUBQH_R.PH SUBQH.W SUBQH_R.W
        SUBU.QB SUBU_S.QB SUBU.PH SUBU_S.PH SUBUH.QB SUBUH_R.QB MODSUB
        CMPGU.EQ.QB CMPGU.LT.QB CMPGU.LE.QB CMPGDU.EQ.QB CMPGDU.LT.QB CMPGDU.LE.QB
        PICK.QB PICK.PH PACKRL.PH PRECRQ.QB.PH PRECR.QB.PH PRECRQ.PH.W
        PRECRQ_RS.PH.W PRECRQU_S.QB.PH
        MULEU_S.PH.QBL MULEU_S.PH.QBR MULQ_RS.PH MULEQ_S.W.PHL MULEQ_S.W.PHR
        MULQ_S.PH MUL.PH MUL_S.PH MULQ_S.W MULQ_RS.W""")
    put(("rd", "rt"), """
        ABSQ_S.QB ABSQ_S.PH ABSQ_S.W BITREV REPLV.QB REPLV.PH
        PRECEQ.W.PHL PRECEQ.W.PHR PRECEQU.PH.QBL PRECEQU.PH.QBR PRECEQU.PH.QBLA
        PRECEQU.PH.QBRA PRECEU.PH.QBL PRECEU.PH.QBR PRECEU.PH.QBLA PRECEU.PH.QBRA""")
    put(("rs", "rt"), """
        CMPU.EQ.QB CMPU.LT.QB CMPU.LE.QB CMP.EQ.PH CMP.LT.PH CMP.LE.PH""")
    put(("rd", "rt", "rs"), """
        SHLLV.QB SHRLV.QB SHRAV.QB SHRAV_R.QB SHLLV.PH SHRAV.PH SHLLV_S.PH
        SHRAV_R.PH SHRLV.PH SHLLV_S.W SHRAV_R.W""")
    put(("rd", "rt", "dsa3"), "SHLL.QB SHRL.QB SHRA.QB SHRA_R.QB")
    put(("rd", "rt", "dsa4"), "SHLL.PH SHLL_S.PH SHRA.PH SHRA_R.PH SHRL.PH")
    put(("rd", "rt", "dsa5"), "SHLL_S.W SHRA_R.W")
    put(("ac", "rs", "rt"), """
        DPA.W.PH DPS.W.PH MULSA.W.PH DPAU.H.QBL DPAQ_S.W.PH DPSQ_S.W.PH
        MULSAQ_S.W.PH DPAU.H.QBR DPAX.W.PH DPSX.W.PH DPSU.H.QBL DPAQ_SA.L.W
        DPSQ_SA.L.W DPSU.H.QBR MAQ_SA.W.PHL MAQ_SA.W.PHR MAQ_S.W.PHL MAQ_S.W.PHR
        DPAQX_S.W.PH DPSQX_S.W.PH DPAQX_SA.W.PH DPSQX_SA.W.PH
        MULT@DSP MULTU@DSP MADD@DSP MADDU@DSP MSUB@DSP MSUBU@DSP""")
    put(("rt", "ac", "dsa5"), "EXTR.W EXTR_R.W EXTR_RS.W EXTR_S.H")
    put(("rt", "ac", "dsize"), "EXTP EXTPDP")
    put(("rt", "ac", "rs"), """
        EXTRV.W EXTRV_R.W EXTRV_RS.W EXTRV_S.H EXTPV EXTPDPV""")
    put(("ac", "shilo"), "SHILO")
    put(("ac", "rs"), "SHILOV")
    put(("rs", "ac"), "MTHLIP MTHI@DSP MTLO@DSP")
    put(("rd", "ac21"), "MFHI@DSP MFLO@DSP")
    put(("rt", "rs"), "INSV")
    put(("rd", "rs"), "RADDU.W.QB")
    put(("rd", "idx"), "LWX LHX LBUX")
    put(("rt", "rs", "dsa"), "APPEND PREPEND PRECR_SRA.PH.W PRECR_SRA_R.PH.W")
    put(("rt", "rs", "bp2"), "BALIGN")
    put(("rd", "dimm10"), "REPL.PH")
    put(("rd", "dimm8"), "REPL.QB")
    put(("rd", "[rmask]"), "RDDSP")
    put(("rs", "[wmask]"), "WRDSP")
    put(("br",), "BPOSGE32 BPOSGE32C")
    return spec


#: the DSP Module rows' operand kinds (PLAN M18)
DSP_SPECS: dict[str, tuple[str, ...]] = _dsp_specs()

#: the MT Module rows' operand kinds (PLAN M19; binutils' MT32 syntax:
#: ``mftr rd, rt, u, sel, h`` / ``mttr rt, rd, u, sel, h``, the target
#: register as a number; rx, which GNU as leaves 0, as an optional last)
MT_SPECS: dict[str, tuple[str, ...]] = {
    "DMT": ("[rt]",), "EMT": ("[rt]",), "DVPE": ("[rt]",), "EVPE": ("[rt]",),
    "FORK": ("rd", "rs", "rt"), "YIELD": ("[rd]", "rs"),
    "MFTR": ("rd", "trt", "u1", "sel", "h1", "[rx]"),
    "MTTR": ("rt", "trd", "u1", "sel", "h1", "[rx]"),
}


def operand_spec(entry: dict) -> tuple[str, ...]:
    """Operand kinds of a row (``[kind]`` = optional)."""
    name, fam = entry["name"], entry["family"]
    if entry.get("ase") == "mt":
        return MT_SPECS[name]
    if entry.get("ase"):
        return DSP_SPECS[name]
    if entry.get("arch") == "mips64" and name in _MIPS64:
        return _MIPS64[name]
    if name in _EXPLICIT:
        return _EXPLICIT[name]
    if name in _R6:
        return _R6[name]
    if name.startswith("CMP.") or "." in name and name.split(".")[0] in (
            "SEL", "SELEQZ", "SELNEZ", "MADDF", "MSUBF", "MIN", "MAX", "MINA", "MAXA"):
        return ("fd", "fs", "ft")
    if fam in ("load", "store"):                   # EVA forms: SPECIAL3
        return ("rt", "mem9") if entry["encode"].startswith("011111") \
            else ("rt", "mem")
    if fam == "branch":
        return ("rs", "br")                        # BGEZ ... BLTZALL
    if fam == "trap":
        return ("rs", "imm") if name.endswith("I") or name.endswith("IU") \
            else ("rs", "rt", "[tcode]")
    if fam == "imm":
        return ("rt", "rs", "imm")
    if fam in ("alu", "muldiv"):
        return ("rd", "rs", "rt")                  # ADD ..., MOVZ/N, MUL
    if name.startswith("C."):
        return ("[cc8]", "fs", "ft")
    if re.fullmatch(r"(MOVF|MOVT)\.(S|D|PS)", name):
        return ("fd", "fs", "[cc18]")
    if re.fullmatch(r"(MOVZ|MOVN)\.(S|D|PS)", name):
        return ("fd", "fs", "rt")
    if _FPU_4.fullmatch(name):
        return ("fd", "fr", "fs", "ft")
    if _FPU_3.fullmatch(name):
        return ("fd", "fs", "ft")
    if _FPU_2.fullmatch(name):
        return ("fd", "fs")
    if name in ("NOP", "SSNOP", "EHB", "PAUSE", "ERET", "DERET") \
            or name.startswith("TLB"):
        return ()
    raise KeyError(f"no assembly syntax for {name}")


#: the value an omitted optional operand stands for (others: 0): GNU as
#: encodes ``rddsp rd`` / ``wrdsp rs`` with every mask bit set
DEFAULTS: dict[str, int] = {"rmask": 0x3FF, "wmask": 0x3FF}


def kind_of(spec_item: str) -> tuple[str, bool]:
    """``'[rd]'`` -> (``'rd'``, optional=True)."""
    if spec_item.startswith("["):
        return spec_item[1:-1], True
    return spec_item, False


#: Rows whose manual syntax ties two fields together (``CLO rd, rs`` has
#: ``rt = rd``): encoder copies, coverage treats the field as covered.
TIED_FIELDS: dict[str, tuple[tuple[int, int], ...]] = {
    "CLO": ((16, 5),), "CLZ": ((16, 5),), "DCLO": ((16, 5),), "DCLZ": ((16, 5),),
    # R6: the register is in both rs and rt (rs == rt selects the row)
    **{n: ((21, 5),) for n in ("BGEZALC", "BLTZALC", "BGEZC", "BLTZC")},
}


def check_coverage(isa: ISA | None = None) -> list[str]:
    """Rows whose operands + fixed bits do not cover the word exactly once
    (must be empty)."""
    isa = isa or ISA()
    problems = []
    for e in isa.instructions:
        name = e["name"]
        covered = e["decode"]["mask"]
        fields = [f for item in operand_spec(e)
                  for f in KINDS[kind_of(item)[0]].fields]
        fields += list(TIED_FIELDS.get(name, ()))
        for lsb, width in fields:
            bits = ((1 << width) - 1) << lsb
            if covered & bits:
                problems.append(f"{name}: operand field ({lsb},{width}) "
                                f"overlaps fixed/other bits")
            covered |= bits
        if covered != M32:
            problems.append(f"{name}: bits {~covered & M32:#010x} not covered")
    return problems
