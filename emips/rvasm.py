"""The RISC-V assembler (PLAN M25 phases 3a / 3b): GNU as 2.42 for RISC-V, ported
from its source (``docs`` keeps ``gas/config/tc-riscv.c`` / ``.h``): one
source file in, the object :mod:`emips.rvld` links.  Names follow gas's
functions; port a behaviour from its source, never from a guess.

* **Instructions** (``riscv_ip``): binutils' opcode table (``riscv.json``:
  the entries of eMIPS's extensions and the macros, in table order); the
  entries of a mnemonic are tried in order, each operand letter of their
  ``args`` parsed as gas does, ``match_func`` checked last — so a
  compressible instruction becomes its C form when C is on (the C
  entries come first), ``.option norvc`` skips 16-bit entries.
* **Macros** (``macro``): ``li`` (``load_const``: lui / addi / slli, each
  through ``md_assemble`` again — compressible), ``la`` / ``lla`` (auipc
  + addi, ``%pcrel_hi`` / ``%pcrel_lo`` of an internal label), ``lb sym``
  ... ``sw r, sym, t`` (auipc + the access), ``call`` / ``tail`` /
  ``jump`` (auipc + jalr, R_RISCV_CALL_PLT), ``sext.b`` / ``sext.h`` /
  ``zext.h`` / ``zext.w`` without Zbb (two shifts).
* **Branches and jumps** go in relaxable frags (``add_relaxed_insn``):
  ``relaxed_branch_length`` picks 2 (an RVC form in reach), 4, 6 (an RVC
  branch inverted over a jal) or 8 bytes (a branch inverted over a jal);
  gas's ``relax_segment`` loop (symbol values of later frags are last
  pass's) and ``md_convert_frag_branch``.
* **Relocations:** RISC-V gas leaves nearly every one to the linker
  (``tc-riscv.h``: TC_FORCE_RELOCATION_LOCAL); this object keeps each fix
  as (offset, type, expression) and :mod:`emips.rvld` applies them at
  the final addresses — the same bytes.
* **Alignment** in code (``riscv_frag_align_code`` / ``riscv_handle_align``):
  with ``relax`` (gas's default, ``-mrelax``) the worst-case nops and an
  R_RISCV_ALIGN for the linker; without it the padding itself (a zero
  byte, at most one c.nop, then nops); nothing at all below the
  instruction alignment.  With ``-mrelax`` (the option's final state) the
  other code padding is zero bytes: ``frag_align_code``'s NOP_OPCODE.
* **The object** (``_object``: ``fixup_segment`` / ``md_apply_fix`` /
  ``tc_gen_reloc``): a fix against a local absolute symbol becomes a
  constant, one against a local equate its target plus offset; every
  relaxable relocation with a symbol, assembled with ``-mrelax``, gets an
  R_RISCV_RELAX after it; the relocations sorted by offset, stably
  (GAS_SORT_RELOCS); ``.size`` values, the RVC flag (``riscv_set_rvc``:
  set by C in the architecture or ``.option rvc``, never cleared).
* **Directives:** sections and subsections (``.text 1``: gas's frchains,
  concatenated in order; a code section's end padded to its alignment, as
  write.c's ``subsegs_finish_section``), symbols, data (``.byte`` ...
  ``.dword``, ``.float`` / ``.double``, strings, ``.space`` / ``.fill``),
  alignment (``.align`` = ``.p2align`` on RISC-V, ``.balign``), ``.equ`` /
  ``.set``, ``.option rvc / norvc / relax / norelax / push / pop / pic /
  nopic / arch``, ``.insn`` (``s_riscv_insn``: ``riscv_insn_types`` and
  ``riscv_ip_hardcode``'s values, 2 to 22 bytes); ``.type`` / ``.size`` /
  ``.file`` / ``.attribute`` / ``.cfi_*`` are accepted and change no byte
  here.  The rest of gas's (``.macro``, ``.rept``, ``.if``, ``.include``,
  ``.org``, ``.comm``, ``.uleb128`` ...) are refused by name
  (``_UNSUPPORTED``).

Choices: ``sim.DEVIATIONS["RISC-V assembler"]``.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

from . import gasread
from .assembler import AsmError
from .nanoasm import _unescape_gas, _LABEL_DEF, _ASSIGN
from .riscv import _DATA, closure, parse_extensions, sp_base

M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF

# ------------------------------------------------------------------ the tables


@dataclass(frozen=True)
class Opcode:
    """One entry of binutils' riscv_opcodes (an instruction or a macro)."""
    i: int
    name: str
    xlen: int
    classes: tuple
    args: str
    match: int
    mask: int
    func: str | None
    macro: str | None = None
    cname: str = "I"                            # binutils' INSN_CLASS_ name
    eew64: bool = False                         # INSN_V_EEW64: no zve32x-only form

    @property
    def length(self) -> int:
        return insn_length(self.match)


def insn_length(insn: int) -> int:
    """riscv_insn_length (include/opcode/riscv.h): 2, 4, 6, 8, 10-22."""
    if insn & 0x3 != 0x3:
        return 2
    if insn & 0x1F != 0x1F:
        return 4
    if insn & 0x3F == 0x1F:
        return 6
    if insn & 0x7F == 0x3F:
        return 8
    if insn & 0x7F == 0x7F and insn & 0x7000 != 0x7000:
        return 10 + ((insn >> 11) & 0xE)
    return 2


def _opcodes() -> dict[str, list[Opcode]]:
    b = _DATA["binutils"]
    rows = [Opcode(e["i"], e["name"], e["xlen"], tuple(tuple(c) for c in e["class"]),
                   e["args"], e["match"], e["mask"],
                   None if e["func"] in (None, "NULL") else e["func"], None, e["cname"],
                   e.get("eew64", False))
            for e in b["entries"]]
    rows += [Opcode(e["i"], e["name"], e["xlen"], tuple(tuple(c) for c in e["class"]),
                    e["args"], e["match"], 0,
                    None if e["func"] in (None, "NULL") else e["func"], e["macro"],
                    e["cname"])
             for e in b["macros"]]
    out: dict[str, list[Opcode]] = {}
    for r in sorted(rows, key=lambda r: r.i):
        out.setdefault(r.name, []).append(r)
    return out


OPCODES = _opcodes()
#: riscv_insn_types: .insn TYPE OPERANDS (match = mask = 0, match_opcode)
INSN_TYPES: dict[str, list[Opcode]] = {}
for _k, _e in enumerate(_DATA["binutils"]["insn_types"]):
    INSN_TYPES.setdefault(_e["name"], []).append(
        Opcode(_k, _e["name"], _e["xlen"], tuple(tuple(c) for c in _e["class"]), _e["args"],
               0, 0, "match_opcode", None, _e["cname"]))
#: opcode_name_list (gas): the names .insn takes for its opcode field
OPCODE_NAMES = {"C0": 0x0, "C1": 0x1, "C2": 0x2, "LOAD": 0x03, "LOAD_FP": 0x07,
                "CUSTOM_0": 0x0B, "MISC_MEM": 0x0F, "OP_IMM": 0x13, "AUIPC": 0x17,
                "OP_IMM_32": 0x1B, "STORE": 0x23, "STORE_FP": 0x27, "CUSTOM_1": 0x2B,
                "AMO": 0x2F, "OP": 0x33, "LUI": 0x37, "OP_32": 0x3B, "MADD": 0x43,
                "MSUB": 0x47, "NMADD": 0x4F, "NMSUB": 0x4B, "OP_FP": 0x53, "OP_V": 0x57,
                "CUSTOM_2": 0x5B, "BRANCH": 0x63, "JALR": 0x67, "JAL": 0x6F, "SYSTEM": 0x73,
                "CUSTOM_3": 0x7B}
#: the .insn fields: letter(s) -> (bits, shift, gas's message)
_INSN_FIELDS = {
    "F7": (0x7F, 25, "bad value for funct7 field, value must be 0...127"),
    "F3": (0x7, 12, "bad value for funct3 field, value must be 0...7"),
    "F2": (0x3, 25, "bad value for funct2 field, value must be 0...3"),
    "CF6": (0x3F, 10, "bad value for compressed funct6 field, value must be 0...63"),
    "CF4": (0xF, 12, "bad value for compressed funct4 field, value must be 0...15"),
    "CF3": (0x7, 13, "bad value for compressed funct3 field, value must be 0...7"),
    "CF2": (0x3, 5, "bad value for compressed funct2 field, value must be 0...3"),
}

GPR_ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2",
           "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
           "s10", "s11", "t3", "t4", "t5", "t6"]
FPR_ABI = ["ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1",
           "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
           "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"]
GPRS = {**{f"x{n}": n for n in range(32)}, **{n: k for k, n in enumerate(GPR_ABI)}, "fp": 8}
FPRS = {**{f"f{n}": n for n in range(32)}, **{n: k for k, n in enumerate(FPR_ABI)}}
#: riscv_rm / riscv_pred_succ (opcodes/riscv-opc.c)
VECR = {f"v{n}": n for n in range(32)}
VECM = {"v0.t": 0}
VSEW = ["e8", "e16", "e32", "e64", None, None, None, None]
VLMUL = ["m1", "m2", "m4", "m8", None, "mf8", "mf4", "mf2"]
VTA = ["tu", "ta"]
VMA = ["mu", "ma"]
RM = ["rne", "rtz", "rdn", "rup", "rmm", None, None, "dyn"]
PRED_SUCC = [None, "w", "r", "rw", "o", "ow", "or", "orw", "i", "iw", "ir", "irw", "io",
             "iow", "ior", "iorw"]
#: the privileged spec gas assumes (DEFAULT_RISCV_PRIV_SPEC "1.11": bfd's
#: PRIV_SPEC_CLASS_1P11, index 3 of the table builder's PRIV list)
PRIV_1P11 = 3
ASM_CSRS = _DATA["binutils"]["asm_csrs"]


#: bfd/cpu-riscv.c riscv_priv_specs (the classes: 1 1.9.1 ... 4 1.12)
PRIV_SPECS = {"1.9.1": 1, "1.10": 2, "1.11": 3, "1.12": 4}


def csr_address(name: str, priv: int = PRIV_1P11) -> int | None:
    """reg_csr_lookup_internal / riscv_csr_address: the CSR named ``name``
    for the privileged spec ``priv`` (gas's default 1.11; the .attribute
    priv_spec tags choose another), else its first definition."""
    entries = ASM_CSRS.get(name)
    if not entries:
        return None
    for num, check, define, abort in entries:
        if not check or (priv >= define and priv < abort):
            return num
    return entries[0][0]


# --------------------------------------------------------- operand encoding


def _x(v: int, s: int, n: int) -> int:
    return (v >> s) & ((1 << n) - 1)


def enc_i(v: int) -> int:
    return _x(v, 0, 12) << 20


def enc_s(v: int) -> int:
    return _x(v, 0, 5) << 7 | _x(v, 5, 7) << 25


def enc_b(v: int) -> int:
    return _x(v, 1, 4) << 8 | _x(v, 5, 6) << 25 | _x(v, 11, 1) << 7 | _x(v, 12, 1) << 31


def enc_u(v: int) -> int:
    return _x(v, 12, 20) << 12


def enc_j(v: int) -> int:
    return _x(v, 1, 10) << 21 | _x(v, 11, 1) << 20 | _x(v, 12, 8) << 12 | _x(v, 20, 1) << 31


def enc_ci(v: int) -> int:
    return _x(v, 0, 5) << 2 | _x(v, 5, 1) << 12


def enc_cb(v: int) -> int:
    return (_x(v, 1, 2) << 3 | _x(v, 3, 2) << 10 | _x(v, 5, 1) << 2 | _x(v, 6, 2) << 5
            | _x(v, 8, 1) << 12)


def enc_cj(v: int) -> int:
    return (_x(v, 1, 3) << 3 | _x(v, 4, 1) << 11 | _x(v, 5, 1) << 2 | _x(v, 6, 1) << 7
            | _x(v, 7, 1) << 6 | _x(v, 8, 2) << 9 | _x(v, 10, 1) << 8 | _x(v, 11, 1) << 12)


def high_part(v: int) -> int:
    """RISCV_CONST_HIGH_PART."""
    return (v + 0x800) & ~0xFFF


#: the 'C' immediates: (encode, valid) — VALID_* as integer ranges
_C_IMM = {
    "5": (lambda v: _x(v, 0, 2) << 5 | _x(v, 2, 3) << 10, lambda v: 0 <= v < 32),
    "6": (lambda v: _x(v, 0, 6) << 7, lambda v: 0 <= v < 64),
    "8": (lambda v: _x(v, 0, 8) << 5, lambda v: 0 <= v < 256),
    "j": (enc_ci, lambda v: v != 0 and -32 <= v < 32),
    "o": (enc_ci, lambda v: -32 <= v < 32),
    ">": (enc_ci, None),
    "k": (lambda v: _x(v, 2, 1) << 6 | _x(v, 3, 3) << 10 | _x(v, 6, 1) << 5,
          lambda v: 0 <= v <= 124 and v % 4 == 0),
    "l": (lambda v: _x(v, 3, 3) << 10 | _x(v, 6, 2) << 5,
          lambda v: 0 <= v <= 248 and v % 8 == 0),
    "m": (lambda v: _x(v, 2, 3) << 4 | _x(v, 5, 1) << 12 | _x(v, 6, 2) << 2,
          lambda v: 0 <= v <= 252 and v % 4 == 0),
    "n": (lambda v: _x(v, 3, 2) << 5 | _x(v, 5, 1) << 12 | _x(v, 6, 3) << 2,
          lambda v: 0 <= v <= 504 and v % 8 == 0),
    "K": (lambda v: _x(v, 2, 1) << 6 | _x(v, 3, 1) << 5 | _x(v, 4, 2) << 11 | _x(v, 6, 4) << 7,
          lambda v: v != 0 and 0 <= v <= 1020 and v % 4 == 0),
    "L": (lambda v: _x(v, 4, 1) << 6 | _x(v, 5, 1) << 2 | _x(v, 6, 1) << 5 | _x(v, 7, 2) << 3
          | _x(v, 9, 1) << 12,
          lambda v: v != 0 and -512 <= v <= 496 and v % 16 == 0),
    "M": (lambda v: _x(v, 2, 4) << 9 | _x(v, 6, 2) << 7,
          lambda v: 0 <= v <= 252 and v % 4 == 0),
    "N": (lambda v: _x(v, 3, 3) << 10 | _x(v, 6, 3) << 7,
          lambda v: 0 <= v <= 504 and v % 8 == 0),
}
#: the 'C' immediates that may be written as a bare base register: (rs1)
_C_IMPLICIT_ZERO = frozenset("klmnMN")

RVC_BRANCH_REACH, RVC_JUMP_REACH = 1 << 9, 1 << 12
BRANCH_REACH, JUMP_REACH = 1 << 13, 1 << 21
MATCH_JAL, MATCH_BEQ, MATCH_BNE = 0x6F, 0x63, 0x1063
MATCH_C_J, MASK_C_J, MATCH_C_JAL, MASK_C_JAL = 0xA001, 0xE003, 0x2001, 0xE003
MATCH_C_BEQZ, MASK_C_BEQZ, MATCH_C_BNEZ = 0xC001, 0xE003, 0xE001
NOP32, NOP16 = 0x13, 0x0001


#: binutils 2.45 tc-riscv.c's MIPS operands ('Xm' + c): (the message's word,
#: the largest value (alignment 1) or the alignment of an offset below 128)
_MIPS_OPERANDS = {"@": ("hint", 31, 1), "#": ("immediate", 511, 1), "$": ("LDP offset", 0, 8),
                  "%": ("LWP offset", 0, 4), "^": ("SDP offset", 0, 8),
                  "&": ("SWP offset", 0, 4), "*": ("MTLBWR level", 7, 1)}
#: ... and the fields they go to: (mask, shift in the word, value shift) —
#: include/opcode/riscv.h OP_MASK_MIPS_* / OP_SH_MIPS_*
_MIPS_FIELDS = {"@": ((0x1F, 7, 0),), "#": ((0x1FF, 20, 0),), "$": ((0xF, 23, 3),),
                "%": ((0x1F, 22, 2),), "^": ((0x3, 10, 3), (0x3, 25, 5)),
                "&": ((0x7, 9, 2), (0x3, 25, 5)), "*": ((0x7, 20, 0),)}


#: FLI.[HSDQ]'s constants (opcodes/riscv-opc.c riscv_fli_symval /
#: riscv_fli_numval: floats; 0.0 for the symbolic ones)
FLI_SYMVAL = [None, "min"] + [None] * 28 + ["inf", "nan"]
FLI_NUMVAL = [-1.0, 0.0, 2.0 ** -16, 2.0 ** -15, 2.0 ** -8, 2.0 ** -7, 0.0625, 0.125, 0.25,
              0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5,
              3.0, 4.0, 8.0, 16.0, 128.0, 256.0, 2.0 ** 15, 2.0 ** 16, 0.0, 0.0]
_STRTOF = re.compile(r"\s*([+-]?(?:0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)"
                     r"(?:[pP][+-]?\d+)?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|"
                     r"inf(?:inity)?|nan(?:\([0-9A-Za-z_]*\))?))", re.I)


def _regno_to_reg_list(regno: int) -> int:
    """tc-riscv.c (2.45) regno_to_reg_list: ra 4, s0 5, s1 6, s2-s9 7-14,
    s11 15, else 0."""
    if regno == 1:
        return 4
    if regno in (8, 9):
        return 5 + regno - 8
    if 18 <= regno < 26:
        return 7 + regno - 18
    return 15 if regno == 27 else 0


def _reglist(text: str) -> int:
    """tc-riscv.c (2.45) reglist_lookup_internal: "{ra}", "{ra, s0}",
    "{ra, s0-sN}" or numeric "{x1, x8-x9, x18-xN}" -> the rlist (0:
    invalid)."""
    names = [[sub for sub in tok.split("-") if sub][:2]
             for tok in text.split(",") if tok][:3]     # (strtok: empty tokens skipped)

    def look(name: str) -> tuple[int | None, int]:
        m = re.match(r"[A-Za-z_.$][A-Za-z0-9_.$]*", name)
        r = GPRS.get(m.group(0)) if m else None
        return r, _regno_to_reg_list(r) if r is not None else 0
    reg_list = 0
    numeric = False
    for i, pair in enumerate(names):
        first = pair[0]
        if i == 0:
            numeric = first[0] == "x"
            r, reg_list = look(first)
            if not reg_list or r != 1:
                return 0
        elif i == 1:
            r, reg_list = look(first)
            if (first[0] == "x") != numeric or not reg_list or r != 8:
                return 0
            if len(pair) == 1:
                return reg_list
            second = pair[1]
            r, reg_list = look(second)
            if (second[0] == "x") != numeric or not reg_list or r <= 8 or (numeric and r != 9):
                return 0
        else:
            if not numeric or first[0] != "x":
                return 0
            r, reg_list = look(first)
            if not reg_list or r != 18:
                return 0
            if len(pair) == 1:
                return reg_list
            second = pair[1]
            r, reg_list = look(second)
            if second[0] != "x" or not reg_list or r <= 18:
                return 0
    return reg_list


def _strtof(s: str, a: int) -> tuple[float | None, int]:
    """C's strtof at ``s[a:]``: (the float's value, where it ends); None
    when nothing converts or it overflows (errno)."""
    m = _STRTOF.match(s, a)
    if not m:
        return None, a
    text = m.group(1)
    try:
        if text.lstrip("+-")[:2].lower() == "0x":
            mant = text if "p" in text.lower() else text + "p0"
            v = float.fromhex(mant)
        else:
            v = float(text)
        v = struct.unpack("<f", struct.pack("<f", v))[0]
    except (OverflowError, ValueError):
        return None, m.end()
    return v, m.end()


def _match_func(op: Opcode, insn: int) -> bool:
    """opcodes/riscv-opc.c's match functions."""
    fn = op.func
    if fn is None:
        return True

    def opcode() -> bool:
        return ((insn ^ op.match) & op.mask) == 0

    rd, rs1, rs2 = insn >> 7 & 31, insn >> 15 & 31, insn >> 20 & 31
    crs2 = insn >> 2 & 31
    ci = (insn >> 2 & 31) | (-(insn >> 12 & 1) << 5)

    def rd_nonzero() -> bool:
        return (op.macro is not None or opcode()) and rd != 0

    if fn == "match_opcode":
        return opcode()
    if fn == "match_rs1_eq_rs2":
        return opcode() and rs1 == rs2
    if fn == "match_rd_nonzero":
        return rd_nonzero()
    if fn == "match_rs1_nonzero":
        return rs1 != 0
    if fn == "match_c_add":
        return rd_nonzero() and crs2 != 0
    if fn == "match_c_add_with_hint":
        return opcode() and crs2 != 0
    if fn == "match_c_nop":
        return opcode() and rd == 0
    if fn == "match_c_addi16sp":
        return opcode() and rd == 2
    if fn == "match_c_lui":
        return rd_nonzero() and rd != 2 and ci != 0
    if fn == "match_c_lui_with_hint":
        return opcode() and rd != 2 and ci != 0
    if fn == "match_c_addi4spn":
        return opcode() and (insn >> 5 & 0xFF) != 0
    if fn == "match_c_slli":
        return opcode() and ci != 0
    if fn == "match_slli_as_c_slli":
        return rd_nonzero() and ci != 0
    if fn == "match_c_slli64":
        return opcode() and ci == 0
    if fn == "match_srxi_as_c_srxi":
        return opcode() and ci != 0
    if fn == "match_sreg1_not_eq_sreg2":                # (2.45: Zcmp / Zcmt)
        return opcode() and (insn >> 7 & 7) != (insn >> 2 & 7)
    if fn == "match_cm_jt":
        return opcode() and (insn >> 2 & 0xFF) < 32
    if fn == "match_cm_jalt":
        return opcode() and 32 <= (insn >> 2 & 0xFF) < 256
    if fn == "match_vs1_eq_vs2":
        return opcode() and (insn >> 15 & 31) == (insn >> 20 & 31)
    if fn == "match_vd_eq_vs1_eq_vs2":
        return opcode() and (insn >> 7 & 31) == (insn >> 15 & 31) == (insn >> 20 & 31)
    if fn == "match_rs2_rd_even":                       # (2.45: AMOCAS.D on RV32, .Q)
        return opcode() and not ((insn >> 20 | insn >> 7) & 1)
    if fn == "match_rd_x1x5_opcode":                    # (2.45: Zicfiss)
        return opcode() and rd in (1, 5)
    if fn == "match_rs1_x1x5_opcode":
        return opcode() and rs1 in (1, 5)
    if fn == "match_rs2_x1x5_opcode":
        return opcode() and rs2 in (1, 5)
    raise AssertionError(fn)


# ------------------------------------------------------------- expressions


@dataclass(eq=False)
class Sym:
    """A gas symbol: a label (section, frag, offset), an absolute value,
    an equate, or undefined."""
    name: str
    section: "Section | None" = None
    frag: "Frag | None" = None
    where: int = 0
    absolute: int | None = None
    equated: tuple | None = None                # (Expr) for .equ of an expression
    external: bool = False
    weak: bool = False
    line: int = 0
    size: "Expr | None" = None                  # .size (st_size, evaluated at the end)
    type: str = ""                              # .type: "function" / "object" / ...
    common: int | None = None                   # .comm: SHN_COMMON's size (ld allocates it)
    common_align: int = 0                       # its alignment (a power of 2, in bytes)
    local: bool = False                         # .local (obj_elf_local)
    tls: bool = False                           # STT_TLS (.tls_common, .type @tls_object)
    weakref: bool = False                       # .weakref's alias

    @property
    def defined(self) -> bool:
        if self.equated is not None:
            e = self.equated[0]
            return e.op == "constant" or (e.sym is not None and e.sym.defined)
        return self.absolute is not None or self.frag is not None or self.common is not None

    def segment(self):
        if self.equated is not None:
            e = self.equated[0]
            return "abs" if e.op == "constant" else (e.sym.segment() if e.sym else None)
        if self.absolute is not None:
            return "abs"
        return self.section

    def value(self) -> int:
        """S_GET_VALUE (relaxation time: frag addresses in the section)."""
        if self.equated is not None:
            e = self.equated[0]
            return e.num + (e.sym.value() if e.sym is not None else 0)
        if self.absolute is not None:
            return self.absolute
        if self.frag is None:
            return 0
        return self.frag.address + self.where


@dataclass
class Expr:
    """expressionS: ``op`` "constant", "symbol" (sym + num), "subtract"
    (sym - sub + num), "big", "absent" or "illegal"."""
    op: str = "absent"
    num: int = 0
    sym: Sym | None = None
    sub: Sym | None = None

    def copy(self) -> "Expr":
        return Expr(self.op, self.num, self.sym, self.sub)


_NAME = re.compile(r"[A-Za-z_.$][\w.$]*")
_NUMBER = re.compile(r"0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+")
_RANK = {"*": 8, "/": 8, "%": 8, "<<": 8, ">>": 8, "|": 7, "!": 7, "^": 7, "&": 7,
         "+": 5, "-": 5, "==": 4, "<>": 4, "!=": 4, "<": 4, "<=": 4, ">=": 4, ">": 4,
         "&&": 3, "||": 2}


def _sx64(v: int) -> int:
    v &= M64
    return v - (1 << 64) if v >> 63 else v


class _Expressions:
    """gas's expr.c at a position of a string (``parse`` -> (Expr, end))."""

    def __init__(self, asm: "Assembler") -> None:
        self.asm = asm
        #: .eqv: symbols stay symbols (read.c s_set: "no attempt is made to
        #: evaluate the expression or any part of it immediately")
        self.late = False

    def parse(self, s: str, i: int, rank: int = 0) -> tuple[Expr, int]:
        left, i = self._operand(s, i)
        while True:
            j = _skip(s, i)
            op = None
            for cand in ("<<", ">>", "==", "<>", "!=", "<=", ">=", "&&", "||",
                         "*", "/", "%", "|", "!", "^", "&", "+", "-", "<", ">"):
                if s.startswith(cand, j):
                    op = cand
                    break
            if op is None or _RANK[op] <= rank:
                return left, i
            right, i = self.parse(s, j + len(op), _RANK[op])
            if right.op == "absent":
                self.asm.warn("missing operand; zero assumed")
                right = Expr("constant", 0)
            left = self._combine(op, left, right)

    def _operand(self, s: str, i: int) -> tuple[Expr, int]:
        i = _skip(s, i)
        if i >= len(s):
            return Expr("absent"), i
        c = s[i]
        if c == "(":
            e, i = self.parse(s, i + 1)
            i = _skip(s, i)
            if i < len(s) and s[i] == ")":
                return e, i + 1
            return Expr("illegal"), i
        if c in "-~!+":
            e, i = self._operand(s, i + 1)
            if c == "+":
                return e, i
            if e.op == "symbol" and c == "-":
                return Expr("illegal"), i
            if e.op != "constant":
                return (e if e.op in ("absent", "illegal") else Expr("illegal")), i
            n = {"-": -e.num, "~": ~e.num, "!": int(not e.num)}[c]
            return Expr("constant", _sx64(n)), i
        if c == "'":
            j = i + 1
            if j >= len(s):
                return Expr("constant", 0), j
            ch = s[j]
            if ch == "\\" and j + 1 < len(s):
                j += 1
                ch = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}.get(s[j], s[j])
            j += 1
            if j < len(s) and s[j] == "'":
                j += 1
            return Expr("constant", ord(ch) & 0xFF), j
        m = _NUMBER.match(s, i)
        if m:
            text, end = m.group(0), m.end()
            if text.startswith(("0x", "0X")):
                n = int(text, 16)
            elif text.startswith(("0b", "0B")):
                n = int(text[2:], 2)
            elif len(text) > 1 and text[0] == "0":
                n = int(text, 8)
            else:
                n = int(text)
            if n >= 1 << 64:
                return Expr("big", n), end
            return Expr("constant", _sx64(n)), end
        if c == "." and not (i + 1 < len(s) and re.match(r"[\w.$]", s[i + 1])):
            return Expr("symbol", 0, self.asm.dot()), i + 1
        m = _NAME.match(s, i)
        if m:
            return self._symbol(m.group(0)), m.end()
        return Expr("absent"), i

    def _symbol(self, name: str) -> Expr:
        sym = self.asm.symbol_ref(name)
        if self.late:
            return Expr("symbol", 0, sym)
        if sym.equated is not None and sym.equated[0].op == "constant":
            return Expr("constant", sym.equated[0].num)
        if sym.absolute is not None:
            return Expr("constant", _sx64(sym.absolute))
        return Expr("symbol", 0, sym)

    def _combine(self, op: str, a: Expr, b: Expr) -> Expr:
        if a.op == "constant" and b.op == "constant":
            x, y = a.num, b.num
            if op in ("/", "%") and y == 0:
                raise ValueError("division by zero")
            r = {"*": lambda: x * y, "/": lambda: int(x / y),
                 "%": lambda: x - int(x / y) * y,
                 "<<": lambda: x << (y & 63) if y >= 0 else 0,
                 ">>": lambda: (x & M64) >> (y & 63) if y >= 0 else 0,
                 "|": lambda: x | y, "!": lambda: x | ~y, "^": lambda: x ^ y,
                 "&": lambda: x & y, "+": lambda: x + y, "-": lambda: x - y,
                 "==": lambda: -int(x == y), "!=": lambda: -int(x != y),
                 "<>": lambda: -int(x != y), "<": lambda: -int(x < y),
                 "<=": lambda: -int(x <= y), ">=": lambda: -int(x >= y),
                 ">": lambda: -int(x > y), "&&": lambda: int(bool(x and y)),
                 "||": lambda: int(bool(x or y))}[op]()
            return Expr("constant", _sx64(r))
        if "illegal" in (a.op, b.op) or "absent" in (a.op, b.op):
            return Expr("illegal")
        if op == "+":
            if a.op in ("symbol", "subtract") and b.op == "constant":
                return Expr(a.op, _sx64(a.num + b.num), a.sym, a.sub)
            if a.op == "constant" and b.op in ("symbol", "subtract"):
                return Expr(b.op, _sx64(a.num + b.num), b.sym, b.sub)
        if op == "-":
            if a.op in ("symbol", "subtract") and b.op == "constant":
                return Expr(a.op, _sx64(a.num - b.num), a.sym, a.sub)
            if a.op == "symbol" and b.op == "symbol":
                return self.asm.difference(a, b)
        return Expr("illegal")


def _skip(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t":
        i += 1
    return i


# ------------------------------------------------------- sections and frags


class Section:
    def __init__(self, name: str) -> None:
        self.name = name
        self.frags: list[Frag] = []             # (the current subsection's chain)
        self.chains: dict[int, list[Frag]] = {0: self.frags}     # gas's frchains
        self.sub = 0
        self.align = 0
        self.code = name == ".text" or name.startswith(".text.")
        self.bss = name in (".bss", ".sbss", ".tbss") or name.startswith((".bss.", ".sbss.",
                                                                           ".tbss."))
        self.tls = name in (".tdata", ".tbss") or name.startswith((".tdata.", ".tbss."))

    def __repr__(self) -> str:
        return f"<{self.name}>"


class Frag:
    """A gas frag: ``fix`` the fixed bytes, then the variable part:
    ``kind`` "fill" (``repeat`` x ``pattern``), "align" / "align_code"
    (padding to 1 << ``power``, at most ``max_skip``), "branch" (a
    relaxable branch: ``var`` bytes, ``subtype``), "org" (rs_org: up to
    ``symbol`` + ``offset`` with ``fill``), "leb128" (rs_leb128: ``expr``,
    ``subtype`` 1 = signed)."""

    def __init__(self, section: Section) -> None:
        self.section = section
        self.fix = bytearray()
        self.kind = "fill"
        self.repeat = 0
        self.pattern = b""
        self.power = 0
        self.max_skip = 0
        self.fill = b"\0"
        self.var = 0
        self.subtype = 0
        self.symbol: Sym | None = None
        self.offset = 0
        self.address = 0
        self.line = 0
        self.pad = b""                          # the converted variable part
        self.expr: "Expr | None" = None         # rs_leb128's value

    @property
    def size(self) -> int:
        return len(self.fix) + len(self.pad)


@dataclass
class Fix:
    """A fixup: ``reloc`` (gas's BFD code without BFD_RELOC_ / RISCV_)."""
    frag: Frag
    where: int
    size: int
    expr: Expr
    reloc: str
    line: int
    relax: bool = False                         # fx_tcbit: assembled with -mrelax


@dataclass
class Insn:
    """riscv_cl_insn."""
    mo: Opcode
    opcode: int

    @property
    def length(self) -> int:
        return insn_length(self.opcode)


class _NoMatch(Exception):
    pass


@dataclass
class Options:
    """riscv_opts (the per-.option state)."""
    rvc: bool = True
    relax: bool = True
    pic: bool = False
    exts: frozenset = frozenset()


def _relax_encode(uncond: bool, rvc: bool, length: int) -> int:
    return 0xC0000000 | (1 if uncond else 0) | (2 if rvc else 0) | length << 2


def _relax_align(address: int, power: int) -> int:
    mask = (1 << power) - 1
    return (-address) & mask


# --------------------------------------------------------------- the assembler


class Assembler:
    """GNU as for RISC-V (see the module docstring)."""

    def __init__(self, xlen: int = 64, extensions: str = "imafdc_zicsr_zifencei_zba_zbb",
                 relax: bool = True, strict: bool = False) -> None:
        self.xlen = xlen
        #: eMIPS's strict mode: GNU's macros refused (their expansion named)
        self.strict = strict
        self.exts = closure(parse_extensions(extensions) | {"i"})
        self.default_relax = relax
        self.expr = _Expressions(self)

    # -- the source ---------------------------------------------------------

    def assemble(self, source: str, path: str | None = None,
                 include_dirs: tuple[str, ...] = ()) -> "AsmObject":
        self.opts = Options(rvc=bool({"c", "zca"} & self.exts), relax=self.default_relax,
                            exts=self.exts)
        self.rvc_flag = self.opts.rvc            # EF_RISCV_RVC (riscv_set_rvc never clears it)
        self.stack: list[Options] = []
        self.symbols: dict[str, Sym] = {}
        self.deferred: dict[str, Sym] = {}
        self.sections: dict[str, Section] = {}
        self.fixes: list[Fix] = []
        self.insns: list[tuple[Frag, int, int, int]] = []       # (frag, where, size, line)
        self.warnings: list[tuple[int, str]] = []
        self.fake = 0
        self.prev_section: tuple[Section, int] | None = None
        self.section_stack: list[tuple[Section, int, tuple | None]] = []   # .pushsection
        self.abs_offset: int | None = None      # in the absolute section (.struct)
        self.attributes: dict[int, int | str] = {}                       # .attribute
        self.reloc_list: list = []              # .reloc: (section, offset, type, expr, line)
        self.started = False                     # an instruction assembled (start_assemble)
        self.priv_spec = PRIV_1P11               # default_priv_spec
        self.priv_checked = False
        self.section = self._section(".text")
        self.record_alignment(self.section, 1 if self.opts.rvc else 2)
        self.frag = self.section.frags[-1]
        lines = source.splitlines()
        self.lines = lines
        self.reader = gasread.Reader(source, _Hooks(self), path=path, include_dirs=include_dirs,
                                     warn=lambda n, msg: self.warnings.append((n, msg)))
        for st in self.reader:
            self.line, self.raw = st.line, st.raw
            try:
                self._statement(st.text)
            except AsmError:
                raise
            except (ValueError, KeyError, _NoMatch) as exc:
                raise AsmError(st.line, str(exc), st.raw) from None
        self.line, self.raw = len(lines), ""
        self._relax_all()
        return self._object()

    def error(self, msg: str):
        raise AsmError(self.line, msg, self.raw)

    def warn(self, msg: str) -> None:
        if (self.line, msg) not in self.warnings:
            self.warnings.append((self.line, msg))

    def _statement(self, text: str) -> None:
        while True:
            m = _LABEL_DEF.match(text)
            if not m:
                break
            self.define_label(m.group(1))
            text = text[m.end():]
        text = text.strip()
        if not text:
            return
        m = _ASSIGN.match(text)
        if m:
            self._equate(m.group(1), text[m.end():].strip())
            return
        if text.startswith("."):
            m = re.match(r"\.[\w.$]+", text)
            name, rest = m.group(0), text[m.end():].strip()
            if name.lower() in _DIRECTIVES:
                self.directive(name.lower(), rest)
                return
            if name.lower() in _UNSUPPORTED:
                self.error(f"directive {name.lower()} is not supported by the eMIPS RISC-V "
                           "assembler")
            self.error(f"unknown pseudo-op: `{name.lower()}'")
        text = _scrub(text)
        name, sep, rest = text.partition(" ")
        self.md_assemble(name.lower() + sep + rest, source=True)   # (read.c: lowercased)

    # -- symbols ---------------------------------------------------------------

    def symbol(self, name: str) -> Sym:
        s = self.symbols.get(name)
        if s is None:
            s = self.deferred.pop(name, None) or Sym(name)
            s.refs = True
            self.symbols[name] = s
        return s

    def symbol_ref(self, name: str) -> Sym:
        """A name in an operand: a register's name (riscv_parse_name) is a
        symbol only committed if the instruction uses the entry."""
        if name in self.symbols or name not in GPRS or not self.probing:
            return self.symbol(name)
        s = self.deferred.get(name)
        if s is None:
            s = self.deferred[name] = Sym(name)
            s.refs = True
        self.used_deferred.add(name)
        return s

    def dot(self) -> Sym:
        if self.abs_offset is not None:
            s = Sym(".", absolute=self.abs_offset)
        else:
            s = Sym(".", self.section, self.frag, len(self.frag.fix))
        s.refs = True
        return s

    def define_label(self, name: str) -> None:
        s = self.symbol(name)
        if s.defined:
            self.error(f"symbol `{name}' is already defined")
        if self.abs_offset is not None:         # (.struct: an offset)
            s.absolute, s.line = self.abs_offset, self.line
            return
        s.section, s.frag, s.where, s.line = self.section, self.frag, len(self.frag.fix), self.line

    def difference(self, a: Expr, b: Expr) -> Expr:
        """expr.c's O_subtract of two symbols: an equate to constants folds;
        otherwise kept for the linker (md_allow_local_subtract is 0 on
        RISC-V)."""
        return Expr("subtract", _sx64(a.num - b.num), a.sym, b.sym)

    def _equate(self, name: str, text: str) -> None:
        e, end = self.expr.parse(text, 0)
        if e.op in ("absent", "illegal") or text[end:].strip():
            self.error(f"bad expression `{text}'")
        s = self.symbol(name)
        if e.op == "constant":
            s.equated, s.absolute = None, e.num
        else:
            s.equated = (e,)

    # -- sections and frags ----------------------------------------------------------

    def _section(self, name: str) -> Section:
        sec = self.sections.get(name)
        if sec is None:
            sec = self.sections[name] = Section(name)
            f = Frag(sec)
            sec.frags.append(f)
        return sec

    def switch_section(self, name: str, sub: int = 0) -> None:
        """obj_elf_section_change_hook + subseg_set."""
        self.prev_section = (self.section, self.section.sub)
        self._subseg_set(self._section(name), sub)

    def _subseg_set(self, sec: Section, sub: int) -> None:
        """subseg_set (``.previous`` untouched); leaves the absolute section."""
        self.abs_offset = None
        self.section = sec
        if sub not in sec.chains:
            sec.chains[sub] = [Frag(sec)]
        sec.sub, sec.frags = sub, sec.chains[sub]
        self.frag = sec.frags[-1]

    def frag_new(self) -> Frag:
        f = Frag(self.section)
        f.line = self.line
        self.section.frags.append(f)
        self.frag = f
        return f

    def frag_more(self, data: bytes) -> int:
        if self.abs_offset is not None:         # (emit_expr in absolute_section)
            if any(data):
                self.error("attempt to store value in absolute section")
            self.abs_offset += len(data)
            return 0
        where = len(self.frag.fix)
        self.frag.fix += data
        return where

    def record_alignment(self, sec: Section, power: int) -> None:
        sec.align = max(sec.align, power)

    # -- instructions -------------------------------------------------------------

    def md_assemble(self, text: str, source: bool = False) -> None:
        if not self.priv_checked:               # (start_assemble: riscv_set_default_priv_spec)
            self.priv_checked = True
            major, minor, rev = (int(self.attributes.get(t, 0)) for t in (8, 10, 12))
            if (major, minor, rev) != (0, 0, 0):
                name = f"{major}.{minor}.{rev}" if rev else f"{major}.{minor}"
                if name not in PRIV_SPECS:
                    self.error(f"unknown default privileged spec `{major}.{minor}.{rev}' set "
                               "by privileged elf attributes")
                self.priv_spec = PRIV_SPECS[name]
        insn, imm, reloc = self.riscv_ip(text)
        if source and self.strict and insn.mo.macro is not None:
            self.error(f"`{insn.mo.name}' is a pseudo-instruction (strict mode): "
                       f"{_STRICT_HINTS[insn.mo.macro]}")
        if insn.mo.macro is not None:
            self.macro(insn, imm, reloc)
        else:
            self.append_insn(insn, imm, reloc)

    def _supported(self, mo: Opcode) -> bool:
        """riscv_multi_subset_supports."""
        if mo.xlen and mo.xlen != self.xlen:
            return False
        return any(all(e in self.opts.exts for e in alt) for alt in mo.classes)

    def _missing_ext(self, cname: str) -> str:
        """riscv_multi_subset_supports_ext (bfd/elfxx-riscv.c) for eMIPS's
        classes."""
        has = self.opts.exts.__contains__
        if cname in ("F_AND_C", "D_AND_C"):
            fd, z = cname[0].lower(), "zc" + cname[0].lower()
            if not has(fd):
                return f"{fd}' and `c', or `{fd}' and `{z}" if not (has("c") or has(z)) else fd
            return f"c' or `{z}"
        if cname == "ZIHINTNTL_AND_C":
            if not has("zihintntl"):
                return ("zihintntl' and `c', or `zihintntl' and `zca"
                        if not (has("c") or has("zca")) else "zihintntl")
            return "c' or `zca"
        for both, (x, y, z, w) in (("ZFHMIN_AND_D_INX", ("zfhmin", "d", "zhinxmin", "zdinx")),
                                   ("ZFHMIN_AND_Q_INX", ("zfhmin", "q", "zhinxmin", "zqinx"))):
            if cname == both:
                for have, need in ((x, y), (y, x), (z, w), (w, z)):
                    if has(have):
                        return need
                return f"{x}' and `{y}', or `{z}' and `{w}"
        for both, (x, y) in (("D_AND_ZFA", ("d", "zfa")), ("Q_AND_ZFA", ("q", "zfa")),
                             ("ZFH_AND_ZFA", ("zfh", "zfa"))):
            if cname == both:
                if not has(x) and not has(y):
                    return f"{x}' and `{y}"
                return x if not has(x) else y
        if cname == "ZFH_OR_ZVFH_AND_ZFA":
            if not has("zfa"):
                return ("zfh' and `zfa', or `zvfh' and `zfa"
                        if not (has("zfh") or has("zvfh")) else "zfa")
            return "zfh' or `zvfh"
        return {"ZMMUL": "m' or `zmmul", "C": "c' or `zca",
                "V": "v' or `zve64x' or `zve32x", "ZVEF": "v' or `zve64d' or `zve64f' or `zve32f",
                "ZVKNHA_OR_ZVKNHB": "zvknha' or `zvknhb", "F_INX": "f' or `zfinx",
                "D_INX": "d' or `zdinx", "ZBB_OR_ZBKB": "zbb' or `zbkb",
                "ZBC_OR_ZBKC": "zbc' or `zbkc", "Q_INX": "q' or `zqinx",
                "ZFH_INX": "zfh' or `zhinx", "ZFHMIN_INX": "zfhmin' or `zhinxmin"}.get(
                    cname, cname.lower())

    def riscv_ip(self, text: str, table: dict | None = None) -> tuple[Insn, Expr, str | None]:
        """gas's riscv_ip: the first entry of the mnemonic whose operands
        all match (``table``: OPCODES, or INSN_TYPES for .insn)."""
        name, _, args = text.partition(" ")
        mos = (OPCODES if table is None else table).get(name)
        if not mos:
            self.error(f"unrecognized opcode `{text}'")
        error = f"unrecognized opcode `{text}'"
        missing = None
        self.probing = True
        try:
            for mo in mos:
                if not self._supported(mo):
                    if mo.xlen and mo.xlen != self.xlen:
                        continue
                    missing = self._missing_ext(mo.cname)
                    continue
                error = f"illegal operands `{text}'"
                missing = None
                self.used_deferred: set[str] = set()
                self.ip_error = None
                try:
                    ip, imm, reloc = self._match(mo, args)
                except _NoMatch:
                    if self.ip_error:
                        error = f"{self.ip_error} `{text}'"
                    continue
                for n in self.used_deferred:            # commit deferred symbols
                    if n in self.deferred and n not in self.symbols:
                        self.symbols[n] = self.deferred.pop(n)
                return ip, imm, reloc
        finally:
            self.probing = False
        if missing is not None:
            self.error(f"unrecognized opcode `{text}', extension `{missing}' required")
        self.error(error)

    def _small(self, s: str, i: int, percent: dict) -> tuple[Expr, str | None, int, int]:
        """my_getSmallExpression: (expression, its %reloc, operators used,
        the end)."""
        reloc = None
        index = -1
        depth = 0
        while True:
            index += 1
            crux, crux_depth = i, depth
            while i < len(s) and s[i] in " \t(":
                if s[i] == "(":
                    depth += 1
                i += 1
            if not (i < len(s) and s[i] == "%" and index < 1):
                break
            hit = None
            for op, r in percent.items():
                if s[i + 1:i + 1 + len(op)].lower() == op:
                    j = i + 1 + len(op)
                    while j < len(s) and s[j] in " \t":
                        j += 1
                    if j < len(s) and s[j] == "(":
                        hit = (j, r)
                        break
            if hit is None:
                break
            i, reloc = hit
        if i < len(s) and s[i] == "%":
            return Expr("illegal"), reloc, 0, i
        e, j = self.expr.parse(s, crux)
        while crux_depth > 0 and j < len(s) and s[j] in ") \t":
            if s[j] == ")":
                crux_depth -= 1
            j += 1
        if crux_depth > 0:
            raise ValueError("unclosed '('")
        return e, reloc, index, j

    def _normalize(self, e: Expr) -> None:
        """normalize_constant_expr: RV32 sign-extends 32-bit constants."""
        if self.xlen > 32:
            return
        if e.op in ("constant", "symbol") and ((e.num & ~M32) == 0 or (e.num & ~M32) == ~M32):
            e.num = ((e.num & M32) ^ 0x80000000) - 0x80000000

    def _reg(self, s: str, i: int, cls: dict) -> tuple[int, int] | None:
        m = _NAME.match(s, i)
        if not m:
            return None
        n = cls.get(m.group(0))
        return None if n is None else (n, m.end())

    def _match(self, mo: Opcode, s: str) -> tuple[Insn, Expr, str | None]:
        ip = Insn(mo, mo.match)
        imm = Expr("absent")
        reloc: str | None = None
        percent: dict = {}
        a = 0
        args = mo.args
        k = 0

        def fail():
            raise _NoMatch

        def ins(v: int, mask: int, sh: int) -> None:   # (INSERT_OPERAND: replaces the field)
            ip.opcode = (ip.opcode & ~(mask << sh)) | (v & mask) << sh

        def gpr() -> int:
            nonlocal a
            r = self._reg(s, a, GPRS)
            if r is None:
                fail()
            a = r[1]
            return r[0]

        def fpr() -> int:                       # (Zfinx: the x registers' names)
            nonlocal a
            r = self._reg(s, a, GPRS if "zfinx" in self.opts.exts else FPRS)
            if r is None:
                fail()
            a = r[1]
            return r[0]

        def small(pct: dict) -> tuple[Expr, str | None, int, int]:
            return self._small(s, a, pct)

        def field(spec: tuple) -> None:
            """.insn's funct fields (my_getSmallExpression, a constant)."""
            nonlocal a, imm
            bits, sh, message = spec
            e, r, n, end = small(_PCT_NULL)
            if n or e.op != "constant" or not 0 <= e.num <= bits:
                self.error(message)
            ins(e.num, bits, sh)
            a = end
            imm = Expr("absent")

        def implicit_zero() -> bool:
            if a < len(s) and s[a] == "(" and "(" not in s[a + 1:]:
                return True
            return False

        while True:
            a = _skip(s, a)
            ch = args[k] if k < len(args) else "\0"
            if ch == "\0":
                if mo.macro is None:
                    if not _match_func(mo, ip.opcode):
                        fail()
                    if ip.length == 2 and not self.opts.rvc:
                        fail()
                    if mo.eew64 and "zve32x" in self.opts.exts and "zve64x" not in self.opts.exts:
                        self.ip_error = "illegal opcode for zve32x"
                        fail()
                elif mo.func is not None and not _match_func(mo, ip.opcode):
                    fail()
                if a != len(s):
                    fail()
                return ip, imm, reloc
            k += 1
            if ch == "C":
                c = args[k]
                k += 1
                if c == "s":
                    r = gpr()
                    if not 8 <= r <= 15:
                        fail()
                    ins(r % 8, 7, 7)
                elif c == "w":
                    r = gpr()
                    if (ip.opcode >> 7 & 7) + 8 != r:
                        fail()
                elif c == "t":
                    r = gpr()
                    if not 8 <= r <= 15:
                        fail()
                    ins(r % 8, 7, 2)
                elif c == "x":
                    r = gpr()
                    if (ip.opcode >> 2 & 7) + 8 != r:
                        fail()
                elif c == "U":
                    if gpr() != ip.opcode >> 7 & 31:
                        fail()
                elif c == "V":
                    ins(gpr(), 31, 2)
                elif c == "c":
                    if gpr() != 2:
                        fail()
                elif c == "z":
                    if gpr() != 0:
                        fail()
                elif c in _C_IMM:
                    if c in _C_IMPLICIT_ZERO and implicit_zero():
                        imm = Expr("constant", 0)
                        continue
                    e, r, n, end = small(percent)
                    if n or e.op != "constant":
                        fail()
                    v = e.num
                    enc, valid = _C_IMM[c]
                    if c == ">":
                        if not 0 <= v < self.xlen:
                            fail()
                    elif not valid(v):
                        fail()
                    ip.opcode |= enc(v)
                    a = end
                    imm = Expr("absent")
                elif c in ("u", "v"):
                    percent = _PCT_U
                    e, r, n, end = small(percent)
                    if n:
                        fail()
                    if c == "v":
                        if e.num & 0xFFF or ((e.num & M32) ^ 0x80000000) - 0x80000000 != e.num:
                            fail()
                        e = Expr(e.op, (e.num & M32) >> 12, e.sym, e.sub)
                    v = e.num
                    if (e.op != "constant" or v <= 0 or v >= 1 << 20
                            or (32 <= v < (1 << 20) - 32)):
                        fail()
                    ip.opcode |= enc_ci(v)
                    a = end
                    imm = Expr("absent")
                elif c == "p":
                    reloc = "12_PCREL"
                    imm, a = self.expr.parse(s, a)
                elif c == "a":
                    imm, a = self.expr.parse(s, a)
                    reloc = "JMP"
                elif c == "S":
                    r = fpr()
                    if not 8 <= r <= 15:
                        fail()
                    ins(r % 8, 7, 7)
                elif c == "D":
                    r = fpr()
                    if not 8 <= r <= 15:
                        fail()
                    ins(r % 8, 7, 2)
                elif c == "T":
                    ins(fpr(), 31, 2)
                elif c == "F":
                    field(_INSN_FIELDS["CF" + args[k]])
                    k += 1
                else:
                    raise AssertionError(f"C{c}")
                continue
            if ch == "F":
                field(_INSN_FIELDS["F" + args[k]])
                k += 1
                continue
            if ch == "V":                               # RVV (30)
                c = args[k]
                k += 1
                if c in "dstuv0T":
                    r = self._reg(s, a, VECR)
                    if r is None:
                        fail()
                    regno, a = r
                    if c == "0":                        # v[m]adc / v[m]sbc / v[f]merge
                        if regno:
                            fail()
                    elif c == "T":                      # vmsge's temporary: in VFUNCT6
                        if not regno:
                            fail()
                        ins(regno, 0x3F, 26)
                    else:
                        for sh in {"d": (7,), "s": (15,), "t": (20,), "u": (15, 20),
                                   "v": (7, 15, 20)}[c]:
                            ins(regno, 31, sh)
                    continue
                if c in "bc":                           # vtypei (my_getVsetvliExpression)
                    e, a = self._vtype(s, a)
                    self._absolute(mo, e)
                    top = 1023 if c == "b" else 2047
                    if not 0 <= e.num <= top:
                        self.error(f"bad value for {'vsetivli' if c == 'b' else 'vsetvli'} "
                                   f"immediate field, value must be 0..{top}")
                    ins(e.num, top, 20)
                    imm = Expr("absent")
                    continue
                if c in "ijkl":                         # vector immediates
                    e, a = self.expr.parse(s, a)
                    self._absolute(mo, e)
                    v = e.num
                    lo, hi = {"i": (-16, 15), "j": (0, 31), "k": (-15, 16), "l": (0, 63)}[c]
                    if not lo <= v <= hi:
                        self.error(f"bad value for vector immediate field, value must be "
                                   f"{lo}...{hi}")
                    if c == "l":                        # ENCODE_RVV_VI_UIMM6
                        ins(v, 31, 15)
                        ins(v >> 5, 1, 26)
                    else:
                        ins(v - 1 if c == "k" else v, 31, 15)
                    imm = Expr("absent")
                    continue
                if c == "m":                            # optional vector mask
                    if a == len(s):
                        ins(1, 1, 25)
                        continue
                    if s[a] == ",":
                        r = self._reg(s, a + 1, VECM)
                        if r is not None:
                            a = r[1]
                            continue
                    fail()
                if c == "M":                            # required vector mask
                    r = self._reg(s, a, VECM)
                    if r is None:
                        fail()
                    a = r[1]
                    continue
                raise AssertionError(f"operand V{c} of {mo.name}")
            if ch == "O":
                bits = 127 if args[k] == "4" else 3
                m = _NAME.match(s, a)
                if m and m.group(0) in OPCODE_NAMES:          # my_getOpcodeExpression
                    v, a = OPCODE_NAMES[m.group(0)], m.end()
                else:
                    e, r, n, end = small(_PCT_NULL)
                    v = e.num if not n and e.op == "constant" else -1
                    a = end
                if args[k] == "4" and not (0 <= v < 128 and v & 3 == 3):
                    self.error("bad value for opcode field, value must be 0...127 and lower "
                               "2 bits must be 0x3")
                if args[k] == "2" and not 0 <= v < 3:
                    self.error("bad value for opcode field, value must be 0...2")
                ins(v, bits, 0)
                k += 1
                imm = Expr("absent")
                continue
            if ch == ",":
                if a < len(s) and s[a] == ",":
                    a += 1
                    continue
                fail()
            if ch in "(){}":
                if a < len(s) and s[a] == ch:
                    a += 1
                    continue
                fail()
            if ch == "W" and args[k:k + 1] == "c":      # Zcb / Zcmp / Zcmt (2.42 / 2.45)
                c = args[k + 1]
                k += 2
                if c in "bh":                           # c.lbu / c.sb, c.lh / c.lhu / c.sh
                    if implicit_zero():                 # riscv_handle_implicit_zero_offset
                        imm = Expr("constant", 0)
                        continue
                    e, r, n, end = small(percent)
                    if n or e.op != "constant" or not (
                            0 <= e.num <= 3 if c == "b" else e.num in (0, 2)):
                        fail()
                    v = e.num
                    ip.opcode |= (v & 1) << 6 | (v >> 1 & 1) << 5 if c == "b" else (v >> 1) << 5
                    a = end
                    imm = Expr("absent")
                    continue
                if c == "f":                            # (andi's 255: matched, not encoded)
                    e, r, n, end = small(percent)
                    if n or e.op != "constant" or e.num != 255:
                        fail()
                    a = end
                    imm = Expr("absent")
                    continue
                if c == "r":                            # PUSH / POP's register list
                    end = s.find("}", a)
                    if end < 0:
                        self.error("cannot find `}' for cm.push/cm.pop")
                    rl = _reglist(s[a:end])
                    if not rl:
                        fail()
                    a = end
                    ins(rl, 0xF, 4)
                    continue
                if c == "p":                            # the stack adjustment -> spimm
                    e, r, n, end = small(percent)
                    if n or e.op != "constant":
                        fail()
                    v = e.num
                    if mo.match == 0xB802:              # (cm.push: a negative adjustment)
                        v = -v
                    v -= sp_base(ip.opcode >> 4 & 0xF, self.xlen)
                    if v not in (0, 16, 32, 48):
                        fail()
                    ip.opcode |= (v >> 4) << 2
                    a = end
                    imm = Expr("absent")
                    continue
                if c in "12":                           # an s0-s7 register
                    r = gpr()
                    if not (r in (8, 9) or 18 <= r <= 23):
                        fail()
                    ins(r % 8, 7, 7 if c == "1" else 2)
                    continue
                if c in "Ii":                           # cm.jt / cm.jalt's index
                    e, r, n, end = small(percent)
                    lo, hi = (0, 31) if c == "I" else (32, 255)
                    if e.op != "constant" or not lo <= e.num <= hi:
                        self.error(f"bad index value for cm.{'jt' if c == 'I' else 'jalt'}, "
                                   f"range: [{lo}, {hi}]")
                    ins(e.num, 0xFF, 2)
                    a = end
                    imm = Expr("absent")
                    continue
                raise AssertionError(f"operand Wc{c} of {mo.name}")
            if ch == "W" and args[k:k + 2] == "if":     # Zicbop's prefetch offset (19.6.3)
                k += 2
                if implicit_zero():
                    imm = Expr("constant", 0)
                    continue
                e, a = self.expr.parse(s, a)            # my_getExpression
                self._absolute(mo, e)                   # check_absolute_expr
                if e.num & 0x1F or not -2048 <= e.num < 2048:
                    self.error(f"improper prefetch offset ({e.num})")
                ins((e.num & ~0x1F) >> 5, 0x7F, 25)     # ENCODE_STYPE_IMM (imm[4:0] = 0)
                imm = Expr("absent")
                continue
            if ch == "W" and args[k:k + 2] == "fv":     # Zfa: FLI's constant (25.1)
                k += 2
                r = self._arg_lookup(s, a, FLI_SYMVAL)
                if r is None:
                    v, end = _strtof(s, a)
                    if v is None or v == 0.0 or v not in FLI_NUMVAL:
                        self.error("bad fli constant operand, supported constants must be "
                                   "in decimal or hexadecimal floating-point literal form")
                    r = FLI_NUMVAL.index(v), end
                regno, a = r
                ins(regno, 31, 15)                      # INSERT_OPERAND (RS1)
                continue
            if ch == "X" and args[k:k + 1] == "m":      # (2.45: the MIPS vendor operands)
                c = args[k + 1]
                k += 2
                e, a = self.expr.parse(s, a)            # my_getExpression
                self._absolute(mo, e)                   # check_absolute_expr
                v = e.num & M64                         # ((unsigned long) ...)
                what, top, align = _MIPS_OPERANDS[c]
                if v > top if align == 1 else (v >= 1 << 7 or v % align):
                    self.error(f"Improper {what} amount ({v})")
                for mask, sh, shift in _MIPS_FIELDS[c]:  # INSERT_OPERAND
                    ins(v >> shift, mask, sh)
                imm = Expr("absent")
                continue
            if ch in "yY":                               # scalar crypto: bs, rnum
                e, a = self.expr.parse(s, a)
                self._absolute(mo, e)
                v, top = e.num & M64, 3 if ch == "y" else 10
                if v > top:
                    self.error(f"Improper {'bs' if ch == 'y' else 'rnum'} immediate ({v})")
                ins(v, 3, 30) if ch == "y" else ins(v, 0xF, 20)
                imm = Expr("absent")
                continue
            if ch in "<>Z":
                e, a = self.expr.parse(s, a)
                self._absolute(mo, e)
                v = e.num
                if ch == "<" and not 0 <= v <= 31:
                    self.error(f"improper shift amount ({v & M64})")
                if ch == ">" and not 0 <= v < self.xlen:
                    self.error(f"improper shift amount ({v & M64})")
                if ch == "Z" and not 0 <= v <= 31:
                    self.error(f"improper CSRxI immediate ({v & M64})")
                if ch == "Z":
                    ins(v, 31, 15)
                else:
                    ins(v, 31 if ch == "<" else 63, 20)
                continue
            if ch == "E":
                m = _NAME.match(s, a)
                num = csr_address(m.group(0), self.priv_spec) if m else None
                if num is not None:
                    ins(num, 0xFFF, 20)
                    a = m.end()
                    continue
                e, a = self.expr.parse(s, a)
                if e.op == "symbol":
                    self.error(f"unknown CSR `{e.sym.name}'")
                self._absolute(mo, e)
                if not 0 <= e.num <= 0xFFF:
                    self.error(f"improper CSR address ({e.num & M64})")
                ins(e.num, 0xFFF, 20)
                continue
            if ch == "m":
                v = self._arg_lookup(s, a, RM)
                if v is None:
                    fail()
                ins(v[0], 7, 12)
                a = v[1]
                continue
            if ch in "PQ":
                v = self._arg_lookup(s, a, PRED_SUCC)
                if v is None:
                    fail()
                ins(v[0], 15, 24 if ch == "P" else 20)
                a = v[1]
                continue
            if ch in "dstr":
                r = gpr()
                if a < len(s) and s[a] == " ":
                    a += 1
                ins(r, 31, {"d": 7, "s": 15, "t": 20, "r": 27}[ch])
                continue
            if ch in "DSTUR":
                r = fpr()
                if a < len(s) and s[a] == " ":
                    a += 1
                if ch == "U":
                    ins(r, 31, 15)
                    ins(r, 31, 20)
                else:
                    ins(r, 31, {"D": 7, "S": 15, "T": 20, "R": 27}[ch])
                continue
            if ch == "I":
                imm, a = self.expr.parse(s, a)
                if imm.op not in ("big", "constant"):
                    fail()
                self._normalize(imm)
                continue
            if ch in "AB":
                imm, a = self.expr.parse(s, a)
                self._normalize(imm)
                if ch == "A" and imm.op != "symbol":
                    fail()
                if ch == "B" and imm.op not in ("symbol", "constant"):
                    fail()
                if imm.op == "symbol":
                    reloc = "32"
                continue
            if ch in "jqo01":
                if ch == "j":
                    percent, reloc = _PCT_I, "LO12_I"
                elif ch == "q":
                    percent, reloc = _PCT_S, "LO12_S"
                elif ch == "o":
                    percent, reloc = _PCT_I, "LO12_I"
                elif ch == "1":
                    percent = _PCT_R
                if ch in "qo0" and implicit_zero():
                    imm = Expr("constant", 0)
                    continue
                e, r, n, end = small(percent)
                if r is not None:
                    reloc = r
                if not n:
                    self._normalize(e)
                    if (e.op != "constant" or (ch == "0" and e.num != 0) or ch == "1"
                            or not -2048 <= e.num < 2048):
                        fail()
                imm = e
                a = end
                continue
            if ch == "p":
                reloc = "12_PCREL"
                imm, a = self.expr.parse(s, a)
                continue
            if ch == "u":
                percent = _PCT_U
                e, r, n, end = small(percent)
                if r is not None:
                    reloc = r
                if not n:
                    if e.op != "constant":
                        fail()
                    if not 0 <= e.num < 1 << 20:
                        self.error("lui expression not in range 0..1048575")
                    reloc = "HI20"
                    e = Expr("constant", e.num << 12)
                imm = e
                a = end
                continue
            if ch == "a":
                imm, a = self.expr.parse(s, a)
                reloc = "JMP"
                continue
            if ch == "c":
                imm, a = self.expr.parse(s, a)
                if s[a:] == "@plt":
                    a += 4
                reloc = "CALL_PLT"
                continue
            if ch == "z":
                e, r, n, end = small(percent)
                if n or e.op != "constant" or e.num != 0:
                    fail()
                a = end
                imm = Expr("absent")
                continue
            raise AssertionError(f"operand {ch!r} of {mo.name}")

    def _arg_lookup(self, s: str, a: int, array: list) -> tuple[int, int] | None:
        end = s.find(",", a)
        text = s[a:] if end < 0 else s[a:end]
        if not text:
            return None
        for i, name in enumerate(array):
            if name is not None and name == text:
                return i, a + len(text)
        return None

    def _vtype(self, s: str, a: int) -> tuple[Expr, int]:
        """my_getVsetvliExpression: e8 / m1 / ta / ma names (each optional,
        in that order, each followed by an optional comma), else an
        expression."""
        v, found = 0, False
        for names, sh in ((VSEW, 3), (VLMUL, 0), (VTA, 6), (VMA, 7)):
            r = self._arg_lookup(s, a, names)
            if r is not None:
                a = r[1]
                if a < len(s) and s[a] == ",":
                    a += 1
                v |= r[0] << sh
                found = True
        if found:
            return Expr("constant", v), a
        return self.expr.parse(s, a)

    def _absolute(self, mo: Opcode, e: Expr) -> None:
        """check_absolute_expr."""
        if e.op == "big":
            self.error("unsupported large constant")
        if e.op != "constant":
            self.error(f"instruction {mo.name} requires absolute expression")
        self._normalize(e)

    # -- output (append_insn) ------------------------------------------------------------

    def _install(self, insn: Insn) -> int:
        n = insn.length                         # (install_insn: insn_length bytes)
        where = self.frag_more((insn.opcode & ((1 << 8 * n) - 1)).to_bytes(n, "little"))
        self.insns.append((self.frag, where, n, self.line))
        return where

    def append_insn(self, ip: Insn, ep: Expr, reloc: str | None) -> None:
        self.started = True                     # (start_assemble: .attribute arch is too late)
        if reloc is not None:
            if reloc in ("12_PCREL", "JMP"):
                best = ip.length
                worst = 8
                frag = self.frag
                where = self._install(ip)
                frag.kind = "branch"
                frag.var = best
                frag.subtype = _relax_encode(reloc == "JMP", best == 2, worst)
                frag.symbol = ep.sym if ep.op == "symbol" else None
                frag.offset = ep.num
                frag.line = self.line
                frag.insn_where = where
                frag.insn_bytes = bytes(frag.fix[where:])
                del frag.fix[where:]
                self.insns[-1] = (frag, where, -1, self.line)     # (sized at conversion)
                self.frag_new()
                return
            where = self._install(ip)
            self._fix(where, ip.length if reloc not in ("32",) else 4, ep, reloc)
        else:
            self._install(ip)
        if reloc in ("HI20", "PCREL_HI20", "TPREL_HI20", "TPREL_ADD"):
            self.frag_new()                     # (frag_wane: the linker may change it)

    def _fix(self, where: int, size: int, e: Expr, reloc: str) -> None:
        if e.op == "constant" and reloc in ("HI20", "LO12_I", "LO12_S", "32"):
            b = self.frag.fix
            word = int.from_bytes(b[where:where + 4], "little")
            v = e.num
            word |= {"HI20": enc_u(high_part(v)), "LO12_I": enc_i(v), "LO12_S": enc_s(v),
                     "32": v & M32}[reloc]
            b[where:where + 4] = (word & M32).to_bytes(4, "little")
            return
        if e.op not in ("symbol", "subtract", "constant"):    # (a constant: against *ABS*)
            self.error("relocation against a non-symbol")
        self.fixes.append(Fix(self.frag, where, size, e.copy(), reloc, self.line,
                              self.opts.relax))

    # -- macros ---------------------------------------------------------------------------

    def macro_build(self, ep: Expr | None, name: str, fmt: str, *args) -> None:
        """macro_build: the first non-RVC entry of ``name``; INSERT_OPERAND
        replaces a field (jalr's first such entry presets rd = ra)."""
        mo = next(m for m in OPCODES[name] if m.length == 4 and m.macro is None)
        ip = Insn(mo, mo.match)
        r = None
        it = iter(args)

        def put(sh: int, mask: int, v: int) -> None:
            ip.opcode = ip.opcode & ~(mask << sh) | (v & mask) << sh

        i = 0
        while i < len(fmt):
            c = fmt[i]
            i += 1
            if c == "V":                        # RVV: Vd, Vs, Vt, Vm (-1 = unmasked)
                c = fmt[i]
                i += 1
                if c == "m":
                    reg = next(it)
                    put(25, 1, 1 if reg == -1 else 0)
                else:
                    put({"d": 7, "s": 15, "t": 20}[c], 31, next(it))
            elif c == "d":
                put(7, 31, next(it))
            elif c == "s":
                put(15, 31, next(it))
            elif c == "t":
                put(20, 31, next(it))
            elif c in "juq":
                r = next(it)
        self.append_insn(ip, ep, r)

    def vector_macro(self, ip: Insn) -> None:
        """vector_macro: vmsge[u].vx (30.13.8: there is no vmsge{u}.vx)."""
        o = ip.opcode
        vd, vs1, vs2 = o >> 7 & 31, o >> 15 & 31, o >> 20 & 31
        vm, vtemp = o >> 25 & 1, o >> 26 & 0x3F
        vmslt = "vmsltu.vx" if ip.mo.match else "vmslt.vx"
        b = self.macro_build
        if vm:                                  # unmasked
            b(None, vmslt, "Vd,Vt,sVm", vd, vs2, vs1, -1)
            b(None, "vmnand.mm", "Vd,Vt,Vs", vd, vd, vd)
        elif vtemp != 0:                        # masked, with a temporary
            if vd == vm:
                b(None, vmslt, "Vd,Vt,sVm", vtemp, vs2, vs1, -1)
                b(None, "vmandnot.mm", "Vd,Vt,Vs", vd, vm, vtemp)
            else:
                b(None, vmslt, "Vd,Vt,sVm", vtemp, vs2, vs1, -1)
                b(None, "vmandnot.mm", "Vd,Vt,Vs", vtemp, vm, vtemp)
                b(None, "vmandnot.mm", "Vd,Vt,Vs", vd, vd, vm)
                b(None, "vmor.mm", "Vd,Vt,Vs", vd, vtemp, vd)
        elif vd != vm:                          # masked: vd may overlap vs2
            b(None, vmslt, "Vd,Vt,sVm", vd, vs2, vs1, vm)
            b(None, "vmxor.mm", "Vd,Vt,Vs", vd, vd, vm)
        else:
            self.error("must provide temp if destination overlaps mask")

    def make_internal_label(self) -> Sym:
        self.fake += 1
        s = Sym(f".L0 {self.fake}", self.section, self.frag, len(self.frag.fix))
        s.refs = True
        return s

    def pcrel_access(self, dest: int, temp: int, ep: Expr, lo_insn: str, lo_fmt: str,
                     hi: str, lo: str) -> None:
        ep2 = Expr("symbol", 0, self.make_internal_label())
        self.macro_build(ep, "auipc", "d,u", temp, hi)
        self.macro_build(ep2, lo_insn, lo_fmt, dest, temp, lo)

    def riscv_call(self, dest: int, temp: int, ep: Expr, reloc: str) -> None:
        self.macro_build(ep, "auipc", "d,u", temp, reloc)
        self.macro_build(None, "jalr", "d,s", dest, temp)
        self.frag_new()

    def load_const(self, reg: int, ep: Expr) -> None:
        if ep.op != "constant":
            self.error("unsupported large constant")
        v = ep.num
        lower = ((v & 0xFFF) ^ 0x800) - 0x800
        upper = v - lower
        if self.xlen > 32 and not (-(1 << 31) <= v < 1 << 31):
            shift = 12
            while ((upper >> shift) & 1) == 0:
                shift += 1
            up = (upper & M64) - (1 << 64) if upper & 1 << 63 else upper    # int64_t
            self.load_const(reg, Expr("constant", up >> shift))
            self.md_assemble(f"slli x{reg},x{reg},0x{shift:x}")
            if lower != 0:
                self.md_assemble(f"addi x{reg},x{reg},{lower}")
        else:
            hi = 0
            if upper != 0:
                self.md_assemble(f"lui x{reg},0x{(upper & M32) >> 12:x}")
                hi = reg
            if lower != 0 or hi == 0:
                add = "addiw" if self.xlen == 64 else "addi"
                self.md_assemble(f"{add} x{reg},x{hi},{lower}")

    def riscv_ext(self, dest: int, src: int, shift: int, sign: bool) -> None:
        self.md_assemble(f"slli x{dest},x{src},{shift:#x}")
        self.md_assemble(f"sr{'a' if sign else 'l'}i x{dest},x{dest},{shift:#x}")

    def macro(self, ip: Insn, ep: Expr, reloc: str | None) -> None:
        rd, rs1, rs2 = ip.opcode >> 7 & 31, ip.opcode >> 15 & 31, ip.opcode >> 20 & 31
        m = ip.mo.macro
        if m == "M_LI":
            if ep.op == "big":
                self.error("unsupported large constant")
            self.load_const(rd, ep)
        elif m in ("M_LA", "M_LLA", "M_LGA"):
            if not -(1 << 31) <= ep.num < 1 << 31:
                self.error("offset too large")
            if ep.op == "constant":
                self.load_const(rd, ep)
            elif (self.opts.pic and m == "M_LA") or m == "M_LGA":
                self.pcrel_access(rd, rd, ep, "lw" if self.xlen == 32 else "ld", "d,s,j",
                                  "GOT_HI20", "PCREL_LO12_I")
            else:
                self.pcrel_access(rd, rd, ep, "addi", "d,s,j", "PCREL_HI20", "PCREL_LO12_I")
        elif m == "M_LA_TLS_GD":
            self.pcrel_access(rd, rd, ep, "addi", "d,s,j", "TLS_GD_HI20", "PCREL_LO12_I")
        elif m == "M_LA_TLS_IE":
            self.pcrel_access(rd, rd, ep, "lw" if self.xlen == 32 else "ld", "d,s,j",
                              "TLS_GOT_HI20", "PCREL_LO12_I")
        elif m == "M_Lx":
            self.pcrel_access(rd, rd, ep, ip.mo.name, "d,s,j", "PCREL_HI20", "PCREL_LO12_I")
        elif m == "M_FLx":
            self.pcrel_access(rd, rs1, ep, ip.mo.name, "d,s,j", "PCREL_HI20", "PCREL_LO12_I")
        elif m == "M_Sx_FSx":
            self.pcrel_access(rs2, rs1, ep, ip.mo.name, "t,s,q", "PCREL_HI20", "PCREL_LO12_S")
        elif m == "M_CALL":
            self.riscv_call(rd, rs1, ep, reloc)
        elif m == "M_EXTH":
            self.riscv_ext(rd, rs1, self.xlen - 16, ip.mo.name[0] == "s")
        elif m == "M_ZEXTW":
            self.riscv_ext(rd, rs1, self.xlen - 32, False)
        elif m == "M_SEXTB":
            self.riscv_ext(rd, rs1, self.xlen - 8, True)
        elif m == "M_VMSGE":
            self.vector_macro(ip)
        else:
            self.error(f"internal: macro {ip.mo.name} not implemented")

    # -- directives -----------------------------------------------------------------------

    def directive(self, d: str, rest: str) -> None:
        fn = _DIRECTIVES[d]
        fn(self, d, rest)

    def _expr(self, text: str) -> Expr:
        e, end = self.expr.parse(text, 0)
        if text[end:].strip():
            self.error(f"junk at end of line, first unrecognized character is `{text[end:].strip()[0]}'")
        return e

    def _absolute_value(self, text: str) -> int:
        e = self._expr(text)
        if e.op != "constant":
            self.error("expected an absolute expression")
        return e.num

    def d_section(self, d: str, rest: str) -> None:
        if d in (".section", ".sect", ".sect.s", ".section.s"):
            name, more = self._section_name(rest)
            self.switch_section(name)
            self._section_flags(self.section, more)
            return
        if d == ".previous":
            if self.prev_section is not None:
                p, sub = self.prev_section
                self.switch_section(p.name, sub)
            return
        else:
            name = d
            sub = self._absolute_value(rest) if rest.strip() else 0
            self.switch_section(name, sub)
            return

    def d_globl(self, d: str, rest: str) -> None:
        for n in (x.strip() for x in rest.split(",")):
            if n:
                s = self.symbol(n)
                s.external = True
                if d == ".weak":
                    s.weak = True

    def d_ignore(self, d: str, rest: str) -> None:
        pass

    def d_type(self, d: str, rest: str) -> None:
        """obj_elf_type: STT_FUNC / STT_OBJECT ... (ld's relaxation reads it)."""
        name, _, kind = rest.partition(",")
        kind = kind.strip().lstrip("@%#").strip('"').lower()
        kind = {"stt_func": "function", "stt_object": "object", "stt_tls": "tls_object",
                "stt_notype": "notype"}.get(kind, kind)
        if kind not in ("function", "object", "tls_object", "notype", "common",
                        "gnu_indirect_function", "gnu_unique_object"):
            self.error(f"unrecognized symbol type \"{kind}\"")
        self.symbol(name.strip()).type = kind

    def d_size(self, d: str, rest: str) -> None:
        """obj_elf_size: st_size, evaluated when the object is written."""
        name, _, value = rest.partition(",")
        if not value.strip():
            self.error("expected comma after name in .size")
        self.symbol(name.strip()).size = self._expr(value.strip())

    def d_equ(self, d: str, rest: str) -> None:
        name, _, value = rest.partition(",")
        name = name.strip()
        if d in (".equiv", ".eqv") and self.symbols.get(name) is not None \
                and self.symbols[name].defined:     # (read.c s_set: .eqv is .equiv, evaluated late)
            self.error(f"symbol `{name}' is already defined")
        self.expr.late = d == ".eqv"
        try:
            self._equate(name, value.strip())
        finally:
            self.expr.late = False

    def d_option(self, d: str, rest: str) -> None:
        """s_riscv_option."""
        o = self.opts
        name = rest.strip()
        if name == "rvc":
            o.exts = o.exts | {"c"}
            o.rvc = self.rvc_flag = True
        elif name == "norvc":
            o.exts = o.exts - {"c"}
            o.rvc = False
        elif name == "pic":
            o.pic = True
        elif name == "nopic":
            o.pic = False
        elif name == "relax":
            o.relax = True
        elif name == "norelax":
            o.relax = False
        elif name in ("csr-check", "no-csr-check"):
            pass
        elif name.startswith("arch,"):
            spec = name[5:].strip()
            if spec[0] in "+-":
                exts = set(o.exts)
                for part in spec.split(","):
                    part = part.strip()
                    if part.startswith("+"):
                        exts |= set(parse_extensions(part[1:]))
                    elif part.startswith("-"):
                        exts -= set(parse_extensions(part[1:]))
                o.exts = closure(exts)
            else:
                body = re.sub(r"^rv(32|64)", "", spec)
                o.exts = closure(parse_extensions(body) | {"i"})
            o.rvc = bool({"c", "zca"} & o.exts)
            self.rvc_flag |= o.rvc
        elif name == "push":
            self.stack.append(Options(o.rvc, o.relax, o.pic, o.exts))
        elif name == "pop":
            if not self.stack:
                self.error(".option pop with no .option push")
            self.opts = self.stack.pop()
        else:
            self.warn(f"unrecognized .option directive: {name}")

    def d_data(self, d: str, rest: str) -> None:
        size = _DATA_SIZES[d]
        for part in _split_args(rest):
            if not part:
                continue
            e = self._expr(part)
            self.emit_value(e, size)

    def emit_value(self, e: Expr, size: int) -> None:
        if (e.op != "constant" or e.num) and self.section.bss and self.abs_offset is None:
            self.error(f"attempt to store non-zero value in section `{self.section.name}'")
        if e.op == "big":
            self.frag_more((e.num & ((1 << (8 * size)) - 1)).to_bytes(size, "little"))
            return
        if e.op == "constant":
            self.frag_more((e.num & ((1 << (8 * size)) - 1)).to_bytes(size, "little"))
            return
        if e.op in ("symbol", "subtract"):
            where = self.frag_more(bytes(size))
            self.fixes.append(Fix(self.frag, where, size, e.copy(), f"DATA{size}", self.line,
                                  self.opts.relax))
            return
        self.error("bad expression")

    def d_float(self, d: str, rest: str) -> None:
        fmt = "<f" if d in (".float", ".single") else "<d"
        for part in _split_args(rest):
            text = part.strip()
            if re.match(r"0[fFdDrR]", text):
                text = text[2:]
            self.frag_more(struct.pack(fmt, float(text)))

    def d_string(self, d: str, rest: str) -> None:
        for part in _split_args(rest):
            part = part.strip()
            if not (part.startswith('"') and part.endswith('"')):
                self.error("expected string")
            data = _unescape_gas(part[1:-1])
            self.frag_more(data + (b"\0" if d in (".asciz", ".string") else b""))

    def d_space(self, d: str, rest: str) -> None:
        """s_space (``.space`` / ``.skip`` / ``.zero``; ``.ds.*`` and
        ``.dcb.*`` with their element size): a constant count, a fill byte —
        or each element emitted when the fill is wider; a count the
        assembler learns later (``.space end - start``) is an rs_space frag."""
        mult = _SPACE_MULT.get(d, 0)
        parts = _split_args(rest)
        if not parts:
            self.error("missing expression")
        e = self._expr(parts[0])
        val = self._expr(parts[1]) if len(parts) > 1 else Expr("constant", 0)
        bss = self.section.bss and self.abs_offset is None
        if (val.op != "constant" or not -0x80 <= val.num <= 0xFF
                or (mult not in (0, 1) and val.num)) and self.abs_offset is None and not bss:
            if e.op != "constant":
                self.error("unsupported variable size or fill value")
            if not 0 <= e.num <= 1 << 10:
                self.error(f"size value for space directive too large: {e.num & M64:x}")
            for _ in range(e.num):
                self.emit_value(val, mult or 1)
            return
        if e.op == "constant":
            if e.num < 0:
                self.warn(".space repeat count is negative, ignored")
                return
            if e.num == 0:
                self.warn(".space repeat count is zero, ignored")
                return
            total = e.num * max(mult, 1)
            if self.abs_offset is not None:
                if val.op != "constant" or val.num:
                    self.warn("ignoring fill value in absolute section")
                self.abs_offset += total
                return
            fill = 0
            if val.op != "constant" or val.num:
                if bss:
                    self.warn(f"ignoring fill value in section `{self.section.name}'")
                else:
                    fill = val.num & 0xFF
            self.frag_more(bytes([fill]) * total)
            return
        if self.abs_offset is not None:
            self.error("space allocation too complex in absolute section")
        f = self.frag                           # (rs_space: the size known later)
        f.kind, f.expr, f.line = "space", e, self.line
        f.fill = bytes([val.num & 0xFF]) if val.op == "constant" and not bss else b"\0"
        self.frag_new()

    def d_insn(self, d: str, rest: str) -> None:
        """s_riscv_insn: ``.insn TYPE OPERANDS`` (riscv_insn_types), else
        riscv_ip_hardcode: ``.insn [LENGTH, ]VALUE``."""
        text = _scrub(rest)
        try:
            insn, imm, reloc = self.riscv_ip(text, INSN_TYPES)
        except AsmError as exc:
            insn, imm, reloc = self._hardcode(text, exc), Expr("absent"), None
        self.append_insn(insn, imm, reloc)

    def _hardcode(self, text: str, error: AsmError) -> Insn:
        """riscv_ip_hardcode."""
        values: list[int] = []
        a = 0
        big = False
        while True:
            e, a = self.expr.parse(text, a)
            if e.op == "big":
                values.append(e.num)
                big = True
            elif e.op == "constant":
                values.append(e.num & M64)
            elif not values:
                raise error
            else:
                self.error("values must be constant")
            if not (a < len(text) and text[a] == "," and len(values) < 2 and not big):
                break
            a = _skip(text, a + 1)
        if a != len(text):
            self.error("unrecognized values")
        value = values[-1]
        n = insn_length(value & M64)
        if len(values) == 2 and values[0] != n:
            self.error("value conflicts with instruction length")
        if value >> (8 * n):
            self.error("value conflicts with instruction length")
        return Insn(Opcode(-1, ".insn", 0, (), "", value, 0, "match_opcode"), value)

    def d_fill(self, d: str, rest: str) -> None:
        parts = _split_args(rest)
        rep = self._absolute_value(parts[0])
        size = self._absolute_value(parts[1]) if len(parts) > 1 else 1
        value = self._absolute_value(parts[2]) if len(parts) > 2 else 0
        size = min(size, 8)
        chunk = (value & M32).to_bytes(4, "little")[:size] if size <= 4 else \
            (value & M32).to_bytes(4, "little") + bytes(size - 4)
        if rep > 0:
            self.frag_more(chunk * rep)

    def d_align(self, d: str, rest: str) -> None:
        parts = _split_args(rest) if rest else []
        if not parts:
            n = 0
        else:
            n = self._absolute_value(parts[0])
        fill = None
        if len(parts) > 1 and parts[1].strip():
            fill = self._absolute_value(parts[1])
        max_skip = self._absolute_value(parts[2]) if len(parts) > 2 and parts[2].strip() else 0
        if d == ".balign":
            if n and n & (n - 1):
                self.error("alignment not a power of 2")
            power = n.bit_length() - 1 if n else 0
        else:
            power = n
        self.do_align(power, fill, max_skip)

    def do_align(self, power: int, fill: int | None, max_skip: int) -> None:
        """read.c do_align -> md_do_align (riscv_frag_align_code) / frag_align."""
        if power and fill is None and self.section.code:
            if self.riscv_frag_align_code(power):
                self.record_alignment(self.section, power)
                return
            f = self.frag
            f.kind, f.power, f.max_skip = "align_code", power, max_skip
            f.line = self.line
            self.frag_new()
        elif power:
            f = self.frag
            f.kind, f.power, f.max_skip = "align", power, max_skip
            f.fill = bytes([fill & 0xFF]) if fill is not None else b"\0"
            self.frag_new()
        self.record_alignment(self.section, power)

    # -- LF12: the rest of GNU as's directives (read.c, obj-elf.c, tc-riscv.c) ----------

    def _section_name(self, rest: str) -> tuple[str, str]:
        """obj_elf_section_name: a quoted name or the characters up to a
        comma / blank; and what follows."""
        rest = rest.strip()
        if rest.startswith('"'):
            j = rest.find('"', 1)
            if j < 0:
                self.error("missing close quote")
            return rest[1:j], rest[j + 1:]
        m = re.match(r"[^\s,]*", rest)
        if not m.group(0):
            self.error("missing name")
        return m.group(0), rest[m.end():]

    def _section_flags(self, sec: Section, rest: str) -> None:
        """The flags string and @type of .section / .pushsection: "x" makes
        code, @nobits a bss section (obj_elf_parse_section_letters)."""
        m = re.match(r'\s*,\s*"([^"]*)"\s*(?:,\s*[@%]?(\w+))?', rest)
        if not m:
            return
        flags, kind = m.group(1), (m.group(2) or "").lower()
        if "x" in flags:
            sec.code = True
        if "T" in flags:
            sec.tls = True
        if kind == "nobits":
            sec.bss = True
        elif kind == "progbits":
            sec.bss = False

    def d_pushsection(self, d: str, rest: str) -> None:
        """obj_elf_section (push = 1): the section stack keeps the current
        and previous section; an optional subsection number first."""
        name, more = self._section_name(rest)
        sub = 0
        m = re.match(r"\s*,\s*(\d[^,]*)", more)
        if m:
            sub = self._absolute_value(m.group(1))
            more = more[m.end():]
        self.section_stack.append((self.section, self.section.sub, self.prev_section,
                                   self.abs_offset))
        self.switch_section(name, sub)
        self._section_flags(self.section, more)

    def d_popsection(self, d: str, rest: str) -> None:
        if not self.section_stack:
            self.warn(".popsection without corresponding .pushsection; ignored")
            return
        sec, sub, prev, abs_ = self.section_stack.pop()
        self.prev_section = prev
        self._subseg_set(sec, sub)
        self.abs_offset = abs_

    def d_subsection(self, d: str, rest: str) -> None:
        """obj_elf_subsection: the current section, subsection N."""
        sub = self._absolute_value(rest)
        self.prev_section = (self.section, self.section.sub)
        self._subseg_set(self.section, sub)

    def d_struct(self, d: str, rest: str) -> None:
        """s_struct (.struct / .offset): the absolute section from an offset —
        labels become constants, data only advances the offset."""
        self.prev_section = (self.section, self.section.sub)
        self.abs_offset = self._absolute_value(rest) if rest.strip() else 0

    def d_org(self, d: str, rest: str) -> None:
        """do_org: an rs_org frag up to SYMBOL + OFFSET (this section or
        absolute) with the fill byte."""
        parts = _split_args(rest)
        if not parts:
            self.error("missing expression")
        e = self._expr(parts[0])
        fill = self._absolute_value(parts[1]) & 0xFF if len(parts) > 1 else 0
        if self.abs_offset is not None:
            if e.op != "constant":
                self.error("invalid segment")
            self.abs_offset = e.num
            return
        if e.op == "constant":
            sym, off = None, e.num
        elif e.op == "symbol" and not e.sym.defined:
            # get_known_segmented_expression: "zero assumed" — but do_org keeps
            # the symbol, so the frag reaches its value after all
            self.warn(f"symbol \"{e.sym.name}\" undefined; zero assumed")
            sym, off = e.sym, 0
        elif e.op == "symbol" and e.sym.segment() in (self.section, "abs"):
            sym, off = e.sym, e.num
        else:
            seg = e.sym.segment() if e.op == "symbol" and e.sym.defined else None
            self.error(f"invalid segment \"{seg.name if isinstance(seg, Section) else '*UND*'}\"")
        f = self.frag
        f.kind, f.symbol, f.offset, f.fill, f.line = "org", sym, off, bytes([fill]), self.line
        self.frag_new()

    def _comm_name_size(self, rest: str) -> tuple[Sym, int, str]:
        """s_comm_internal: NAME[,] SIZE and what follows."""
        m = re.match(r"\s*(\"[^\"]*\"|[A-Za-z_.$][\w.$]*)\s*,?", rest)
        if not m:
            self.error("expected symbol name")
        name = m.group(1).strip('"')
        args = _split_args(rest[m.end():])
        if not args or not args[0]:
            self.error("missing size expression")
        e = self._expr(args[0])
        if e.op != "constant":
            self.error("bad or irreducible absolute expression")
        size = e.num
        if size < 0 or size >> self.xlen:
            self.warn(f"size ({size}) out of range, ignored")
            return None, 0, ""
        sym = self.symbol(name)
        if (sym.defined or sym.equated is not None) and sym.common is None:
            self.error(f"symbol `{name}' is already defined")
        if sym.common is not None and sym.common != size:
            self.warn(f"size of \"{name}\" is already {sym.common}; not changing to {size}")
            size = sym.common
        return sym, size, ",".join(args[1:])

    def _parse_align(self, text: str, bytes_: bool) -> int:
        """parse_align: the alignment after the size (bytes, made a power
        of two, for ``bytes_``)."""
        if not text.strip():
            self.error("expected alignment after size")
        a = self._absolute_value(text)
        if a < 0:
            self.warn("alignment negative; 0 assumed")
            a = 0
        if bytes_ and a:
            if a & (a - 1):
                self.error("alignment not a power of 2")
            a = a.bit_length() - 1
        return a

    def _bss_alloc(self, sym: Sym, size: int, power: int) -> None:
        """bss_alloc: .bss subsection 1, aligned, the symbol there."""
        cur, sub, abs_ = self.section, self.section.sub, self.abs_offset
        self._subseg_set(self._section(".bss"), 1)
        if power:
            self.record_alignment(self.section, power)
            f = self.frag
            f.kind, f.power, f.fill = "align", power, b"\0"
            self.frag_new()
        sym.section, sym.frag, sym.where, sym.line = self.section, self.frag, len(self.frag.fix), \
            self.line
        sym.common = None
        self.frag_more(bytes(size))
        sym.size = Expr("constant", size)
        self._subseg_set(cur, sub)
        self.abs_offset = abs_

    def d_comm(self, d: str, rest: str) -> None:
        """obj_elf_common / elf_common_parse (.comm, .common, .tls_common;
        .xcom: read.c's s_comm, no alignment): a common symbol for ld to
        allocate in .bss — or, after .local, a .bss object here."""
        sym, size, more = self._comm_name_size(rest)
        if sym is None:
            return
        power = 0
        if d != ".xcom" and more.strip():
            if more.strip().startswith('"'):    # (.common sym, size, "bss")
                if more.strip().strip('"').lstrip(".") not in ("bss", "data"):
                    self.error(f"bad .common segment {more.strip()}")
                sym.local = False
            else:
                power = self._parse_align(more, sym.local)
        elif more.strip():
            self.error(f"junk at end of line, first unrecognized character is `{more.strip()[0]}'")
        if d == ".tls_common":
            sym.tls = True
        if sym.local:
            self._bss_alloc(sym, size, power)
            sym.external = False
            return
        if not power:                           # (bfd: st_value of an unaligned common =
            power = 1                           # its size up to a power of 2, at most 16)
            while power < size and power < 16:
                power <<= 1
        sym.common, sym.common_align = size, power      # (st_value: ld uses its log2)
        sym.external = True

    def d_lcomm(self, d: str, rest: str) -> None:
        """obj_elf_lcomm: a local .bss object, aligned by its size
        (TC_IMPLICIT_LCOMM_ALIGNMENT)."""
        sym, size, more = self._comm_name_size(rest)
        if sym is None:
            return
        if more.strip():
            self.error(f"junk at end of line, first unrecognized character is `{more.strip()[0]}'")
        power = 3 if size >= 8 else 2 if size >= 4 else 1 if size >= 2 else 0
        self._bss_alloc(sym, size, power)

    def d_local(self, d: str, rest: str) -> None:
        """obj_elf_local: not external; a later .comm is a local object."""
        for n in (x.strip() for x in _split_args(rest)):
            if n:
                s = self.symbol(n)
                s.external = False
                s.local = True

    def d_leb128(self, d: str, rest: str) -> None:
        """s_riscv_leb128 -> s_leb128 / emit_leb128_expr: a constant now; a
        label difference an rs_leb128 frag (and, with -mrelax, ld's
        R_RISCV_SET_ULEB128 / SUB_ULEB128 pair: riscv_insert_uleb128_fixes)."""
        sign = d == ".sleb128"
        parts = _split_args(rest) or [""]
        if len(parts) > 1:                      # (s_riscv_leb128: one expression, then
            self.error("junk at end of line, first unrecognized character is `,'")
        for part in parts:                      # demand_empty_rest_of_line)
            e = self._expr(part) if part else Expr("absent")
            if sign and e.op != "constant":
                self.error("non-constant .sleb128 is not supported")
            if not sign and e.op not in ("constant", "subtract", "big"):
                self.error(".uleb128 only supports constant or subtract expressions")
            if e.op in ("constant", "big"):
                v = e.num if (e.op == "big" or sign) else e.num & M64
                if self.section.bss and v:
                    self.error(f"attempt to store non-zero value in section `{self.section.name}'")
                self.frag_more(_leb128(v, sign))
                continue
            f = self.frag
            f.kind, f.expr, f.subtype, f.line = "leb128", e, int(sign), self.line
            self.frag_new()

    def d_octa(self, d: str, rest: str) -> None:
        """cons (16): 128-bit values (a negative one sign-extended)."""
        for part in _split_args(rest):
            if not part:
                continue
            e = self._expr(part)
            if e.op not in ("constant", "big"):
                self.error("bad expression")
            self.frag_more((e.num & ((1 << 128) - 1)).to_bytes(16, "little"))

    def d_incbin(self, d: str, rest: str) -> None:
        """s_incbin: a file's bytes (``skip``, ``count``), looked up as
        .include does (emips.gasread)."""
        m = re.match(r'\s*"((?:\\.|[^"\\])*)"', rest)
        if not m:
            self.error("expected string")
        name = _unescape_gas(m.group(1)).decode("latin-1")
        args = _split_args(rest[m.end():].lstrip().lstrip(","))
        skip = self._absolute_value(args[0]) if args and args[0] else 0
        count = self._absolute_value(args[1]) if len(args) > 1 and args[1] else 0
        if len(args) > 1 and args[1] and count == 0:
            self.warn(f".incbin count zero, ignoring `{name}'")
        path = self.reader.find(name)
        if path is None:
            self.error(f"file not found: {name}")
        with open(path, "rb") as fh:
            data = fh.read()
        if not count:
            count = len(data) - skip
        if skip < 0 or count < 0 or skip + count > len(data):
            self.error(f"skip ({skip}) or count ({count}) invalid for file size ({len(data)})")
        self.frag_more(data[skip:skip + count])

    def d_float16(self, d: str, rest: str) -> None:
        """float_cons ('h'): IEEE binary16."""
        for part in _split_args(rest):
            text = part.strip()
            if re.match(r"0[fFdDrRhH]", text):
                text = text[2:]
            self.frag_more(struct.pack("<e", float(text)))

    def d_strings(self, d: str, rest: str) -> None:
        """stringer (.string8 / 16 / 32 / 64): each character a little-endian
        unit of that width, then a zero unit."""
        width = {".string8": 1, ".string16": 2, ".string32": 4, ".string64": 8}[d]
        for part in _split_args(rest):
            part = part.strip()
            if not (part.startswith('"') and part.endswith('"')):
                self.error("expected string")
            out = b"".join(b.to_bytes(width, "little")
                           for b in _unescape_gas(part[1:-1]) + b"\0")
            self.frag_more(out)

    def d_dc(self, d: str, rest: str) -> None:
        """cons / float_cons for .dc, .dc.b / .w / .l / .a / .s / .d."""
        if d in (".dc.s", ".dc.d"):
            self.d_float(".float" if d == ".dc.s" else ".double", rest)
            return
        size = {".dc": 2, ".dc.b": 1, ".dc.w": 2, ".dc.l": 4, ".dc.a": self.xlen // 8}[d]
        self.d_data(f".{size}byte" if size > 1 else ".byte", rest)

    def d_align_fill(self, d: str, rest: str) -> None:
        """s_align with a 2- / 4-byte fill pattern (.balignw / .balignl /
        .p2alignw / .p2alignl)."""
        parts = _split_args(rest) if rest else []
        n = self._absolute_value(parts[0]) if parts and parts[0] else 0
        if d.startswith(".balign"):
            if n and n & (n - 1):
                self.error("alignment not a power of 2")
            power = n.bit_length() - 1 if n else 0
        else:
            power = n
        width = 2 if d.endswith("w") else 4
        fill = self._absolute_value(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        max_skip = self._absolute_value(parts[2]) if len(parts) > 2 and parts[2].strip() else 0
        if fill is None:
            self.do_align(power, None, max_skip)
            return
        if power:
            f = self.frag
            f.kind, f.power, f.max_skip = "align", power, max_skip
            f.fill = (fill & ((1 << (8 * width)) - 1)).to_bytes(width, "little")
            self.frag_new()
        self.record_alignment(self.section, power)

    def d_attribute(self, d: str, rest: str) -> None:
        """s_riscv_attribute / obj_elf_vendor_attribute: Tag_RISCV_arch sets
        the ISA (riscv_set_arch; not after an instruction), the others are
        kept for the ELF's .riscv.attributes."""
        parts = _split_args(rest)
        if len(parts) != 2:
            self.error("expected comma")
        tag_text, value = parts[0].strip(), parts[1].strip()
        tag = _ATTRIBUTE_TAGS.get(tag_text)
        if tag is None:
            try:
                tag = self._absolute_value(tag_text)
            except AsmError:
                self.error(f"attribute name not recognised: {tag_text}")
        string = bool(tag & 1)                  # (riscv_elf_obj_attrs_arg_type: odd = a string)
        if string:
            if not (value.startswith('"') and value.endswith('"')):
                self.error("bad string constant")
            v: int | str = _unescape_gas(value[1:-1]).decode("latin-1")
        else:
            v = self._absolute_value(value)
        self.attributes[tag] = v
        if tag == 5:                            # Tag_RISCV_arch
            if self.started:
                self.error("architecture elf attributes must set before any instructions")
            m = re.match(r"rv(32|64)", v.lower())
            if not m:
                self.error(f"{v}: ISA string must begin with rv32 or rv64")
            if int(m.group(1)) != self.xlen:
                self.error(f"{v}: the architecture is rv{self.xlen} (eMIPS assembles for "
                           "the core's XLEN)")
            try:
                exts = closure(parse_extensions(_arch_extensions(v)) | {"i"})
            except ValueError as exc:
                self.error(str(exc))
            self.exts = exts
            self.opts.exts = exts
            self.opts.rvc = bool({"c", "zca"} & exts)
            if self.opts.rvc:
                self.rvc_flag = True
        elif tag in (6, 8, 10) and self.started:
            self.error("privileged elf attributes must set before any instructions")

    def d_reloc(self, d: str, rest: str) -> None:
        """s_reloc: ``.reloc OFFSET, TYPE[, EXPRESSION]`` — a relocation ld
        applies (TYPE an R_RISCV_ name, or BFD_RELOC_NONE / 32 / 64); a
        constant offset is in the current section."""
        parts = _split_args(rest)
        off = self._expr(parts[0]) if parts and parts[0] else Expr("absent")
        if off.op in ("absent", "illegal", "big"):
            self.error("missing or bad offset expression")
        if len(parts) < 2 or not parts[1]:
            self.error("missing reloc type")
        name = parts[1].strip()
        if name.upper().startswith("BFD_RELOC_"):
            kind = {"NONE": "NONE", "32": "DATA4", "64": "DATA8"}.get(name[10:].upper())
        else:
            kind = _ELF_RELOCS.get(name.upper())
        if kind is None:
            self.error("unrecognized reloc type")
        e = self._expr(parts[2]) if len(parts) > 2 and parts[2] else Expr("absent")
        if e.op in ("illegal", "big"):
            self.error("bad reloc expression")
        if off.op == "constant":
            where = (self.section, off.num, None)
        elif off.op == "symbol":
            where = (None, off.num, off.sym)
        else:
            self.error("missing or bad offset expression")
        self.reloc_list.append((where, kind, e, self.line))

    def d_dtprel(self, d: str, rest: str) -> None:
        """s_dtprel: a TLS symbol's offset from the DTV pointer (ld:
        address - the TLS segment - 0x800)."""
        e = self._expr(rest)
        if e.op != "symbol":
            self.error(f"unsupported use of {d}")
        size = 8 if d == ".dtpreldword" else 4
        where = self.frag_more(bytes(size))
        self.fixes.append(Fix(self.frag, where, size, e.copy(), f"TLS_DTPREL{8 * size}", self.line))

    def d_lsym(self, d: str, rest: str) -> None:
        """s_lsym: NAME, EXPRESSION (a local symbol: never global)."""
        name, _, value = rest.partition(",")
        name = name.strip()
        if not value.strip():
            self.error("expected comma after name")
        if name in self.symbols and self.symbols[name].defined:
            self.error(f"symbol `{name}' is already defined")
        self._equate(name, value.strip())

    def d_weakref(self, d: str, rest: str) -> None:
        """s_weakref: ALIAS, TARGET — the alias stands for the target, which a
        reference through the alias makes a weak one."""
        alias, _, target = rest.partition(",")
        alias, target = alias.strip(), target.strip()
        if not alias or not target:
            self.error("expected comma after name")
        a = self.symbol(alias)
        if a.defined:
            self.error(f"symbol `{alias}' is already defined")
        t = self.symbol(target)
        if t is a:
            self.error(f"recursive use of weakref `{alias}'")
        a.equated, a.weakref = (Expr("symbol", 0, t),), True

    def d_symver(self, d: str, rest: str) -> None:
        """obj_elf_symver: NAME, NAME2@[@]NODE — the versioned name an alias
        of NAME (a static link has no version scripts)."""
        name, _, versioned = rest.partition(",")
        name, versioned = name.strip(), versioned.strip().split(",")[0].strip()
        if "@" not in versioned:
            self.error(f"missing version name in `{versioned}' for symbol `{name}'")
        alias = self.symbol(versioned)
        alias.equated = (Expr("symbol", 0, self.symbol(name)),)

    def d_abort(self, d: str, rest: str) -> None:
        self.error(".abort detected.  Abandoning ship.")

    def d_nop(self, d: str, rest: str) -> None:
        """s_nop: ``nop`` instructions (at least one) until N bytes; .nops is
        rs_space_nop, which RISC-V's gas does not implement."""
        if d == ".nops":
            self.error("unimplemented .nops directive")
        n = self._absolute_value(rest) if rest.strip() else 0
        start = self._here()
        while True:
            self.md_assemble("nop")
            if self._here() - start >= n:
                break

    def _here(self) -> int:
        """Bytes emitted in this frag chain so far (frag_offset_ignore_align_p)."""
        return sum(len(f.fix) for f in self.section.frags)

    def d_float_x(self, d: str, rest: str) -> None:
        """float_cons ('x') / s_float_space: the 80-bit extended format, 10
        bytes (.ds.x reserves 12); .dcb.s / .dcb.d / .dcb.x COUNT, VALUE
        repeat a float."""
        if d.startswith(".dcb."):
            parts = _split_args(rest)
            count = self._absolute_value(parts[0])
            value = parts[1] if len(parts) > 1 else "0"
            kind = d[-1]
            for _ in range(max(count, 0)):
                self._float_bytes(kind, value)
            return
        for part in _split_args(rest):
            self._float_bytes("x", part)

    def _float_bytes(self, kind: str, text: str) -> None:
        text = text.strip()
        if re.match(r"0[fFdDrRxX]", text):
            text = text[2:]
        if kind == "s":
            self.frag_more(struct.pack("<f", float(text)))
        elif kind == "d":
            self.frag_more(struct.pack("<d", float(text)))
        else:
            self.frag_more(_extended(text))

    def riscv_frag_align_code(self, n: int) -> bool:
        bytes_ = 1 << n
        insn_align = 2 if self.opts.rvc else 4
        worst = bytes_ - insn_align
        if bytes_ <= insn_align:
            return True
        if not self.opts.relax:
            return False
        where = self.frag_more(_nops(worst))
        self.fixes.append(Fix(self.frag, where, 0, Expr("constant", worst), "ALIGN", self.line,
                              True))
        self.frag_new()
        return True

    # -- relaxation (write.c relax_segment, riscv_relax_frag) -----------------------------------

    def relaxed_branch_length(self, f: Frag, update: bool) -> int:
        jump = f.subtype & 1
        rvc = f.subtype & 2
        length = 4 if jump else 8
        s = f.symbol
        if s is not None and s.defined and not s.weak and s.segment() is f.section:
            val = s.value() + f.offset - (f.address + len(f.fix))
            rng = RVC_JUMP_REACH if jump else RVC_BRANCH_REACH
            if rvc and (val + rng // 2) % (1 << 64) < rng:
                length = 2
            elif (val + BRANCH_REACH // 2) % (1 << 64) < BRANCH_REACH:
                length = 4
            elif not jump and rvc:
                length = 6
        if update:
            f.subtype = _relax_encode(bool(jump), bool(rvc), length)
        return length

    def _relax_all(self) -> None:
        for sec in self.sections.values():     # the subsections in order (subsegs_finish)
            sec.frags = [f for n in sorted(sec.chains) for f in sec.chains[n]]
        for sec in self.sections.values():
            if sec.code and sec.align:          # write.c subsegs_finish_section: the code
                f = sec.frags[-1]               # section's end padded (frag_align_code)
                f.kind, f.power, f.max_skip, f.line = "align_code", sec.align, 0, self.line
                self.section, self.frag = sec, f
                self.frag_new()
        pass_ = 0
        while True:                             # (relax_and_size_all_segments)
            changed = False
            for sec in self.sections.values():
                changed |= self._relax_segment(sec, pass_)
            pass_ += 1
            if not changed or pass_ > 64:
                break
        if self.opts.relax:                     # (riscv_md_finish: riscv_insert_uleb128_fixes)
            for sec in self.sections.values():
                for f in sec.frags:
                    if f.kind == "leb128" and f.expr.op == "subtract":
                        self.fixes.append(Fix(f, len(f.fix), 0, Expr("symbol", f.expr.num,
                                                                     f.expr.sym),
                                              "SET_ULEB128", f.line))
                        self.fixes.append(Fix(f, len(f.fix), 0, Expr("symbol", 0, f.expr.sub),
                                              "SUB_ULEB128", f.line))
        for sec in self.sections.values():
            self._convert(sec)

    def _relax_segment(self, sec: Section, pass_: int) -> bool:
        """write.c relax_segment: a first guess of every frag's address, then
        passes until nothing grows; returns whether an address moved since
        the last call (the caller repeats over the sections while one did)."""
        frags = sec.frags
        address = 0
        for f in frags:
            f.address = address
            address += len(f.fix)
            if f.kind == "align" or f.kind == "align_code":
                off = _relax_align(address, f.power)
                if f.max_skip and off > f.max_skip:
                    off = 0
                if f.kind == "align" and len(f.fill) > 1 and off % len(f.fill):
                    raise AsmError(f.line, f"alignment padding ({off} bytes) not a multiple of "
                                   f"{len(f.fill)}", self._source_line(f.line))
                address += off
            elif f.kind == "branch":
                f.var = self.relaxed_branch_length(f, False)
                address += f.var
            elif f.kind == "leb128":
                f.var = 1                       # (initial guess: 1, write.c)
                address += 1
            elif f.kind in ("org", "space"):
                f.var = 0
        ret = False
        iterations = max(len(frags) * len(frags), len(frags))
        while True:
            stretch = 0
            stretched = False
            for k, f in enumerate(frags):
                was = f.address
                f.address += stretch
                growth = 0
                if f.kind in ("align", "align_code"):
                    old = _relax_align(was + len(f.fix), f.power)
                    new = _relax_align(f.address + len(f.fix), f.power)
                    if f.max_skip:
                        old = 0 if old > f.max_skip else old
                        new = 0 if new > f.max_skip else new
                    growth = new - old
                elif f.kind == "branch":
                    old = f.var
                    f.var = self.relaxed_branch_length(f, True)
                    growth = f.var - old
                elif f.kind == "org":
                    target = f.offset + (f.symbol.value() if f.symbol is not None else 0)
                    after = frags[k + 1].address + stretch
                    growth = target - after
                    if f.address + len(f.fix) > target:
                        growth = 0
                        if pass_ < 2:
                            ret = True
                        else:
                            raise AsmError(f.line, "attempt to move .org backwards",
                                           self._source_line(f.line))
                    else:
                        f.var = target - (f.address + len(f.fix))
                elif f.kind == "space":
                    amount, ok = self._space_amount(f.expr)
                    if not ok:
                        raise AsmError(f.line, ".space, .nops or .fill specifies non-absolute "
                                       "value", self._source_line(f.line))
                    if amount < 0:
                        if pass_ < 2:
                            ret = True
                        else:
                            self.warnings.append((f.line, ".space, .nops or .fill with "
                                                  "negative value, ignored"))
                            f.kind, f.var = "fill", 0
                    else:
                        growth = was + len(f.fix) + amount - frags[k + 1].address
                        f.var = amount
                elif f.kind == "leb128":
                    size = len(_leb128(self._leb_value(f), bool(f.subtype)))
                    growth = size - f.var
                    f.var = size
                if growth:
                    stretch += growth
                    stretched = True
            iterations -= 1
            if not stretched or iterations <= 0:
                break
        for f in frags:
            if getattr(f, "last_address", None) != f.address:
                f.last_address = f.address
                ret = True
        return ret

    def _source_line(self, n: int) -> str:
        return self.lines[n - 1] if 0 < n <= len(self.lines) else ""

    def _space_amount(self, e: Expr) -> tuple[int, bool]:
        """An rs_space frag's size: an absolute value (a difference of two
        symbols of one section)."""
        if e.op == "constant":
            return e.num, True
        if e.op == "subtract" and e.sym.defined and e.sub.defined \
                and e.sym.segment() is e.sub.segment():
            return e.num + e.sym.value() - e.sub.value(), True
        return 0, False

    def _leb_value(self, f: Frag) -> int:
        """resolve_symbol_value of an rs_leb128 frag's expression."""
        e = f.expr
        if e.op == "subtract":
            if not (e.sym.defined and e.sub.defined):
                undef = e.sym if not e.sym.defined else e.sub
                raise AsmError(f.line, f"leb128 operand is an undefined symbol: {undef.name}",
                               self._source_line(f.line))
            v = e.num + e.sym.value() - e.sub.value()
        else:
            v = e.num + (e.sym.value() if e.sym is not None else 0)
        return v if f.subtype else v & M64

    def _convert(self, sec: Section) -> None:
        """md_convert_frag_branch and the alignment fills (riscv_handle_align)."""
        for f in sec.frags:
            if f.kind == "branch":
                self._convert_branch(f)
            elif f.kind == "align_code":
                n = _relax_align(f.address + len(f.fix), f.power)
                if f.max_skip and n > f.max_skip:
                    n = 0
                # riscv_handle_align fills only without -mrelax (the option's
                # final state); else frag_align_code's NOP_OPCODE byte, 0
                f.pad = bytes(n) if self.opts.relax else _nops_norelax(n)
            elif f.kind == "align":
                n = _relax_align(f.address + len(f.fix), f.power)
                if f.max_skip and n > f.max_skip:
                    n = 0
                f.pad = (f.fill * n)[:n] if len(f.fill) > 1 else f.fill * n
            elif f.kind in ("org", "space"):
                f.pad = f.fill[:1] * f.var
            elif f.kind == "leb128":
                f.pad = _leb128(self._leb_value(f), bool(f.subtype))

    def _convert_branch(self, f: Frag) -> None:
        length = f.subtype >> 2 & 0xF
        uncond = f.subtype & 1
        rvc = f.subtype & 2
        buf = bytearray(f.insn_bytes)
        exp = Expr("symbol", f.offset, f.symbol) if f.symbol is not None else \
            Expr("constant", f.offset)
        out = bytearray()
        fixes: list[tuple[int, int, str]] = []
        if rvc and length in (8, 4):
            insn = int.from_bytes(buf[:2], "little")
            rs1 = 8 + (insn >> 7 & 7)
            if insn & MASK_C_J == MATCH_C_J:
                insn = MATCH_JAL
            elif insn & MASK_C_JAL == MATCH_C_JAL:
                insn = MATCH_JAL | 1 << 7
            elif insn & MASK_C_BEQZ == MATCH_C_BEQZ:
                insn = MATCH_BEQ | rs1 << 15
            elif insn & MASK_C_BEQZ == MATCH_C_BNEZ:
                insn = MATCH_BNE | rs1 << 15
            else:                               # (an .insn cb / cj: gas aborts)
                raise AsmError(f.line, "a compressed .insn branch or jump cannot reach its "
                                       "target (GNU as 2.42 aborts here)", "")
            buf = bytearray(insn.to_bytes(4, "little"))
        if rvc and length == 6:
            insn = int.from_bytes(buf[:2], "little")
            insn ^= MATCH_C_BEQZ ^ MATCH_C_BNEZ
            insn |= enc_cb(6)
            out += insn.to_bytes(2, "little")
            fixes.append((2, 4, "JMP"))
            out += MATCH_JAL.to_bytes(4, "little")
        elif rvc and length == 2:
            out += buf[:2]
            fixes.append((0, 2, "RVC_JUMP" if uncond else "RVC_BRANCH"))
        elif length == 8:
            insn = int.from_bytes(buf[:4], "little")
            insn ^= MATCH_BEQ ^ MATCH_BNE
            insn |= enc_b(8)
            out += insn.to_bytes(4, "little")
            fixes.append((4, 4, "JMP"))
            out += MATCH_JAL.to_bytes(4, "little")
        else:                                   # 4
            out += buf[:4]
            fixes.append((0, 4, "JMP" if uncond else "12_PCREL"))
        f.pad = bytes(out)
        base = len(f.fix)
        for off, size, reloc in fixes:
            fx = Fix(f, base + off, size, exp, reloc, f.line, False)
            fx.in_pad = True
            self.fixes.append(fx)
        # the listing: the branch's instructions
        k = next(i for i, t in enumerate(self.insns) if t[0] is f and t[2] == -1)
        if len(out) == 8 or len(out) == 6:
            first = 4 if len(out) == 8 else 2
            self.insns[k] = (f, base, first, f.line)
            self.insns.insert(k + 1, (f, base + first, 4, f.line))
        else:
            self.insns[k] = (f, base, len(out), f.line)

    # -- the object --------------------------------------------------------------------------

    def _object(self) -> "AsmObject":
        from .rvld import ObjSection, Reloc
        obj = AsmObject(self.xlen, self)
        for name, sec in self.sections.items():
            o = ObjSection(name, sec.align, sec.code, sec.bss)
            o.tls = sec.tls
            data = bytearray()
            for f in sec.frags:
                f.address = len(data)
                data += f.fix + f.pad
            o.data = data
            obj.sections[name] = o
        for fx in self.fixes:                   # (md_apply_fix / tc_gen_reloc)
            o = obj.sections[fx.frag.section.name]
            e = fx.expr
            sym = e.sym if e.op in ("symbol", "subtract") else None
            sub = e.sub if e.op == "subtract" else None
            addend = e.num
            while sym is not None and not sym.external and sym.equated is not None \
                    and sym.equated[0].op in ("symbol", "constant") and sub is None:
                q = sym.equated[0]                # (adjust_reloc_syms: an equate's target)
                addend += q.num
                sym = q.sym
            if sym is not None and sub is None and not sym.external \
                    and sym.absolute is not None:
                addend += sym.absolute            # (fixup_segment: a local absolute symbol)
                sym = None
            at = fx.frag.address + fx.where
            o.relocs.append(Reloc(at, fx.reloc, sym, addend, fx.size, fx.line, sub))
            if fx.reloc in _RELAXABLE and fx.relax and sym is not None:
                o.relocs.append(Reloc(at, "RELAX", None, 0, 0, fx.line))
        for (sec, off, sym), kind, e, line in self.reloc_list:     # (write_relocs: .reloc's,
            if sym is not None:                 # after the fixups at their address)
                if not sym.defined or not isinstance(sym.segment(), Section):
                    raise AsmError(line, "bad offset expression", self._source_line(line))
                sec, off = sym.segment(), off + sym.value()
            target = e.sym if e.op in ("symbol", "subtract") else None
            size = _DATA_RELOC_SIZE.get(kind, 0)
            obj.sections[sec.name].relocs.append(
                Reloc(off, kind, target, e.num if e.op != "absent" else 0, size, line,
                      e.sub if e.op == "subtract" else None))
        for o in obj.sections.values():         # GAS_SORT_RELOCS: by address, stable
            o.relocs.sort(key=lambda r: r.offset)
        for frag, where, size, line in self.insns:
            obj.sections[frag.section.name].insns.append((frag.address + where, size, line))
        for s in self.symbols.values():         # .size: st_size (elf_frob_symbol)
            if s.size is not None:
                obj.sizes[s] = self._size_value(s)
        obj.rvc = self.rvc_flag
        return obj

    def _size_value(self, s: Sym) -> int:
        e = s.size
        if e.op == "constant":
            return e.num
        if e.op == "subtract" and e.sym.segment() is e.sub.segment():
            return e.num + e.sym.value() - e.sub.value()
        if e.op == "symbol" and e.sym.segment() == "abs":
            return e.num + e.sym.value()
        raise AsmError(s.line, f".size expression for {s.name} does not evaluate to a "
                               "constant", "")


@dataclass
class AsmObject:
    """What :mod:`emips.rvld` links: sections (bytes, relocations, the
    instructions' offsets), the symbols (``asm.symbols``)."""
    xlen: int
    asm: Assembler
    sections: dict = field(default_factory=dict)
    sizes: dict = field(default_factory=dict)       # Sym -> st_size
    rvc: bool = False                               # EF_RISCV_RVC

    @property
    def symbols(self) -> dict[str, Sym]:
        return self.asm.symbols


#: strict mode: what each of GNU as's macros expands to
_STRICT_HINTS = {
    "M_LI": "lui and addi / addiw (and slli for 64-bit constants)",
    "M_LA": "auipc rd, %pcrel_hi(sym) and addi rd, rd, %pcrel_lo(label of the auipc)",
    "M_LLA": "auipc rd, %pcrel_hi(sym) and addi rd, rd, %pcrel_lo(label of the auipc)",
    "M_LGA": "auipc rd, %got_pcrel_hi(sym) and a load (%pcrel_lo)",
    "M_Lx": "auipc rd, %pcrel_hi(sym) and the load with %pcrel_lo(label of the auipc)",
    "M_FLx": "auipc rt, %pcrel_hi(sym) and the load with %pcrel_lo(label of the auipc)",
    "M_Sx_FSx": "auipc rt, %pcrel_hi(sym) and the store with %pcrel_lo(label of the auipc)",
    "M_CALL": "auipc rt, %pcrel_hi(sym) and jalr rd, %pcrel_lo(label)(rt) — or jal",
    "M_EXTH": "slli and srli / srai",
    "M_ZEXTW": "slli and srli",
    "M_SEXTB": "slli and srai",
    "M_VMSGE": "vmslt.vx / vmsltu.vx and vmnand.mm, or with a mask vmxor.mm / vmandn.mm / vmor.mm",
    "M_LA_TLS_GD": "auipc rd, %tls_gd_pcrel_hi(sym) and addi (%pcrel_lo)",
    "M_LA_TLS_IE": "auipc rd, %tls_ie_pcrel_hi(sym) and a load (%pcrel_lo)",
}

#: md_apply_fix's relaxable relocations (an R_RISCV_RELAX follows them with
#: -mrelax; GOT_HI20 only from la / lga, which eMIPS's linker refuses)
_RELAXABLE = frozenset(("HI20", "LO12_I", "LO12_S", "TPREL_HI20", "TPREL_LO12_I",
                        "TPREL_LO12_S", "TPREL_ADD", "CALL", "CALL_PLT", "PCREL_HI20",
                        "PCREL_LO12_I", "PCREL_LO12_S"))

#: the relocation operators (percent_op_utype / itype / stype / rtype)
_PCT_U = {"tprel_hi": "TPREL_HI20", "pcrel_hi": "PCREL_HI20", "got_pcrel_hi": "GOT_HI20",
          "tls_ie_pcrel_hi": "TLS_GOT_HI20", "tls_gd_pcrel_hi": "TLS_GD_HI20", "hi": "HI20"}
_PCT_I = {"lo": "LO12_I", "tprel_lo": "TPREL_LO12_I", "pcrel_lo": "PCREL_LO12_I"}
_PCT_S = {"lo": "LO12_S", "tprel_lo": "TPREL_LO12_S", "pcrel_lo": "PCREL_LO12_S"}
_PCT_R = {"tprel_add": "TPREL_ADD"}
_PCT_NULL: dict = {}


def _nops(n: int) -> bytes:
    """riscv_make_nops: a zero byte if odd, at most one c.nop, then nops."""
    out = bytearray()
    i = 0
    if n % 2:
        out.append(0)
        i = 1
    if (n - i) % 4 == 2:
        out += NOP16.to_bytes(2, "little")
        i += 2
    while i < n:
        out += NOP32.to_bytes(4, "little")
        i += 4
    return bytes(out)


def _nops_norelax(n: int) -> bytes:
    """riscv_handle_align without relaxation: the excess over a multiple of
    4 (riscv_make_nops), then 4-byte nops."""
    if n <= 0:
        return b""
    excess = n % 4
    return _nops(excess) + NOP32.to_bytes(4, "little") * ((n - excess) // 4)


class _Hooks:
    """What emips.gasread asks the assembler (``.if``, ``.ifdef``, a macro's
    name, comments)."""

    onechar = True

    def __init__(self, asm: Assembler) -> None:
        self.asm = asm

    def absolute(self, text: str) -> int:
        return self.asm._absolute_value(text)

    def defined(self, name: str) -> bool:
        """cond.c: defined, or equated (a referenced, unset symbol is not)."""
        s = self.asm.symbols.get(name)
        return s is not None and (s.defined or s.equated is not None or s.absolute is not None)

    def known(self, directive: str) -> bool:
        return directive in _DIRECTIVES or directive in _UNSUPPORTED

    def strip(self, line: str) -> str:
        return _strip_comment(line)


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote and (len(out) < 2 or out[-2] != "\\"):
                quote = None
        elif ch == '"':
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _split_args(text: str) -> list[str]:
    parts, cur, quote, depth = [], [], False, 0
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            cur.append(ch)
            if ch == "\\" and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
            elif ch == '"':
                quote = False
        elif ch == '"':
            quote = True
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur).strip())
    return parts if any(parts) else []


_SYMCH = re.compile(r"[\w.$]")


def _scrub(text: str) -> str:
    """app.c's scrubbing of an instruction: the mnemonic, one space, then
    the operands with whitespace kept only between two name characters."""
    text = text.strip()
    m = re.match(r"\S+", text)
    name, rest = m.group(0), text[m.end():].strip()
    if not rest:
        return name
    out = []
    i = 0
    quote = False
    while i < len(rest):
        ch = rest[i]
        if quote:
            out.append(ch)
            if ch == '"':
                quote = False
            i += 1
            continue
        if ch == '"':
            quote = True
            out.append(ch)
            i += 1
            continue
        if ch == "'":                       # (LEX_IS_ONECHAR_QUOTE: 'x' -> its number)
            c = rest[i + 1] if i + 1 < len(rest) else "\0"
            i += 2
            if c == "\\" and i < len(rest):
                c = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}.get(rest[i], rest[i])
                i += 1
            if i < len(rest) and rest[i] == "'":
                i += 1
            out.append(str(ord(c) & 0xFF))
            continue
        if ch in " \t":
            j = i
            while j < len(rest) and rest[j] in " \t":
                j += 1
            if out and j < len(rest) and _SYMCH.match(out[-1]) and (
                    _SYMCH.match(rest[j]) or rest[j] == "'"):   # (state 10: foo 'b')
                out.append(" ")
            i = j
            continue
        out.append(ch)
        i += 1
    return name + " " + "".join(out)


def _leb128(v: int, sign: bool) -> bytes:
    """output_leb128: ``v`` unsigned (a 64-bit value) or signed."""
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        done = (v == 0 and not b & 0x40) or (v == -1 and b & 0x40) if sign else v == 0
        out.append(b | (0 if done else 0x80))
        if done:
            return bytes(out)


def _extended(text: str) -> bytes:
    """atof_ieee's 'x': the 80-bit extended format (1 sign, 15 exponent,
    64 significand bits with the integer bit), rounded to nearest even."""
    from decimal import Decimal, InvalidOperation
    from fractions import Fraction
    t = text.strip().lower()
    neg = t.startswith("-")
    t = t.lstrip("+-")
    if t in ("inf", "infinity"):
        return (1 << 63).to_bytes(8, "little") + (0x7FFF | neg << 15).to_bytes(2, "little")
    if t == "nan":
        return (0xC000000000000000).to_bytes(8, "little") + (0x7FFF).to_bytes(2, "little")
    try:
        f = Fraction(Decimal(t))
    except InvalidOperation:
        raise ValueError(f"bad floating literal: {text}") from None
    if f == 0:
        return bytes(8) + (neg << 15).to_bytes(2, "little")
    e = f.numerator.bit_length() - f.denominator.bit_length()
    if Fraction(2) ** e > f:
        e -= 1
    while Fraction(2) ** (e + 1) <= f:
        e += 1
    exp = e + 16383
    if exp <= 0:                                # (denormal: exponent 0, no integer bit)
        scaled = f * 2 ** (63 + 16382)
        exp = 0
    else:
        scaled = f * Fraction(2) ** (63 - e)
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2 * r > scaled.denominator or (2 * r == scaled.denominator and q & 1):
        q += 1
    if q >> 64:
        q >>= 1
        exp += 1
    if exp >= 0x7FFF:
        q, exp = 1 << 63, 0x7FFF
    return q.to_bytes(8, "little") + (exp | neg << 15).to_bytes(2, "little")


def _arch_extensions(arch: str) -> str:
    """An ISA string with versions (``rv64i2p1_m2p0_zicsr2p0``) as
    parse_extensions reads it (``im_zicsr``)."""
    t = arch.lower()[4:]
    parts = t.split("_")
    out = ["".join(re.findall(r"([a-z])(?:\d+(?:p\d+)?)?", parts[0]))]
    for p in parts[1:]:
        if p:
            out.append(re.sub(r"(?<=[a-z])\d+(?:p\d+)?$", "", p))
    return "_".join(out)


#: R_RISCV_ names (bfd's howto table) -> the linker's reloc names
_ELF_RELOCS = {**{n: n for n in (
    "NONE", "CALL", "CALL_PLT", "PCREL_HI20", "PCREL_LO12_I", "PCREL_LO12_S", "HI20",
    "LO12_I", "LO12_S", "ADD8", "ADD16", "ADD32", "ADD64", "SUB6", "SUB8", "SUB16", "SUB32",
    "SUB64", "SET6", "SET8", "SET16", "SET32", "32_PCREL", "RVC_BRANCH", "RVC_JUMP",
    "RVC_LUI", "RELAX", "ALIGN", "GPREL_I", "GPREL_S", "SET_ULEB128", "SUB_ULEB128",
    "GOT_HI20", "TLS_GOT_HI20", "TLS_GD_HI20", "TPREL_HI20", "TPREL_LO12_I", "TPREL_LO12_S",
    "TPREL_ADD", "TPREL_I", "TPREL_S", "TLS_DTPMOD32", "TLS_DTPMOD64", "TLS_DTPREL32",
    "TLS_DTPREL64", "TLS_TPREL32", "TLS_TPREL64", "COPY", "JUMP_SLOT", "RELATIVE",
    "IRELATIVE")},
    "BRANCH": "12_PCREL", "JAL": "JMP", "32": "DATA4", "64": "DATA8"}
_ELF_RELOCS = {"R_RISCV_" + k: v for k, v in _ELF_RELOCS.items()}
_DATA_RELOC_SIZE = {"DATA4": 4, "DATA8": 8, "ADD8": 1, "ADD16": 2, "ADD32": 4, "ADD64": 8,
                    "SUB6": 1, "SUB8": 1, "SUB16": 2, "SUB32": 4, "SUB64": 8, "SET6": 1,
                    "SET8": 1, "SET16": 2, "SET32": 4, "32_PCREL": 4}
#: s_space's element sizes (.ds.x / .ds.p: 12 bytes)
_SPACE_MULT = {".dcb": 2, ".dcb.b": 1, ".dcb.l": 4, ".dcb.w": 2, ".ds": 2, ".ds.b": 1,
               ".ds.d": 8, ".ds.l": 4, ".ds.p": 12, ".ds.s": 4, ".ds.w": 2, ".ds.x": 12}
#: riscv_convert_symbolic_attribute (and the Tag_RISCV_ forms)
_ATTRIBUTE_TAGS = {**{n: t for n, t in (("arch", 5), ("priv_spec", 8), ("priv_spec_minor", 10),
                                        ("priv_spec_revision", 12), ("unaligned_access", 6),
                                        ("stack_align", 4))},
                   **{"Tag_RISCV_" + n: t for n, t in (("arch", 5), ("priv_spec", 8),
                                                       ("priv_spec_minor", 10),
                                                       ("priv_spec_revision", 12),
                                                       ("unaligned_access", 6),
                                                       ("stack_align", 4))}}
#: listing and debug directives: nothing in a static program's bytes
_NO_EFFECT = (".eject", ".list", ".nolist", ".llen", ".lflags", ".page", ".plen", ".psize",
              ".sbttl", ".title", ".ttl", ".nopage", ".format", ".noformat", ".spc", ".name",
              ".debug", ".xref", ".stabs", ".stabn", ".stabd", ".xstabs", ".desc", ".line",
              ".loc_mark_labels", ".version", ".gnu_attribute", ".attach_to_group",
              ".linkonce", ".vtable_entry", ".vtable_inherit", ".internal",
              ".this_gcc_requires_the_gnu_assembler")
_DATA_SIZES = {".byte": 1, ".half": 2, ".hword": 2, ".short": 2, ".2byte": 2, ".word": 4, ".long": 4,
               ".4byte": 4, ".int": 4, ".dword": 8, ".quad": 8, ".8byte": 8}
#: GNU as's directives eMIPS leaves out (an error naming them)
_UNSUPPORTED = frozenset((".mri", "..mri", ".rva"))
_DIRECTIVES = {
    **{d: Assembler.d_section for d in (".text", ".data", ".bss", ".section", ".previous")},
    **{d: Assembler.d_globl for d in (".globl", ".global", ".weak")},
    ".type": Assembler.d_type,
    ".size": Assembler.d_size,
    **{d: Assembler.d_ignore for d in (".file", ".ident", ".hidden", ".protected", ".extern",
                                        ".cfi_startproc", ".cfi_endproc", ".loc",
                                        ".cfi_def_cfa_offset", ".cfi_offset",
                                        ".cfi_def_cfa", ".cfi_def_cfa_register",
                                        ".cfi_restore", ".cfi_sections",
                                        ".variant_cc", ".func", ".endfunc")},
    **{d: Assembler.d_equ for d in (".equ", ".set", ".equiv", ".eqv")},
    ".option": Assembler.d_option,
    **{d: Assembler.d_data for d in _DATA_SIZES},
    **{d: Assembler.d_float for d in (".float", ".single", ".double")},
    **{d: Assembler.d_string for d in (".ascii", ".asciz", ".string")},
    **{d: Assembler.d_space for d in (".space", ".zero", ".skip")},
    ".fill": Assembler.d_fill,
    ".insn": Assembler.d_insn,
    **{d: Assembler.d_align for d in (".align", ".p2align", ".balign")},
    # LF12
    **{d: Assembler.d_section for d in (".sect", ".sect.s", ".section.s")},
    ".pushsection": Assembler.d_pushsection,
    ".popsection": Assembler.d_popsection,
    ".subsection": Assembler.d_subsection,
    **{d: Assembler.d_struct for d in (".struct", ".offset")},
    ".org": Assembler.d_org,
    **{d: Assembler.d_comm for d in (".comm", ".common", ".common.s", ".xcom", ".tls_common")},
    ".lcomm": Assembler.d_lcomm,
    ".local": Assembler.d_local,
    ".xdef": Assembler.d_globl,
    **{d: Assembler.d_leb128 for d in (".uleb128", ".sleb128")},
    ".octa": Assembler.d_octa,
    ".incbin": Assembler.d_incbin,
    ".float16": Assembler.d_float16,
    **{d: Assembler.d_strings for d in (".string8", ".string16", ".string32", ".string64")},
    **{d: Assembler.d_dc for d in (".dc", ".dc.b", ".dc.w", ".dc.l", ".dc.a", ".dc.s", ".dc.d")},
    **{d: Assembler.d_float_x for d in (".dc.x", ".dcb.s", ".dcb.d", ".dcb.x")},
    **{d: Assembler.d_space for d in _SPACE_MULT},
    **{d: Assembler.d_align_fill for d in (".balignw", ".balignl", ".p2alignw", ".p2alignl")},
    ".attribute": Assembler.d_attribute,
    ".lsym": Assembler.d_lsym,
    **{d: Assembler.d_dtprel for d in (".dtprelword", ".dtpreldword")},
    ".reloc": Assembler.d_reloc,
    ".weakref": Assembler.d_weakref,
    ".symver": Assembler.d_symver,
    ".abort": Assembler.d_abort,
    **{d: Assembler.d_nop for d in (".nop", ".nops")},
    **{d: Assembler.d_ignore for d in _NO_EFFECT},
}


#: the extensions programs are assembled for (simrv.DEFAULT_EXTENSIONS)
DEFAULT_EXTENSIONS = "imafdc_zicsr_zifencei_zicntr_zihpm_zba_zbb"
#: RARS's initial sp (decision 10)
BOOT_SP = 0x7FFFEFFC


def assemble(source: str, *, arch: str = "rv64", extensions: str = DEFAULT_EXTENSIONS,
             relax: bool = True, link_relax: bool = True, bases: dict | None = None,
             strict: bool = False, path: str | None = None, include_dirs: tuple[str, ...] = ()):
    """Assemble and link a RISC-V ``source`` (``arch`` "rv32" / "rv64")
    into a :class:`emips.assembler.Program` (``arch`` set, ``rv`` = the
    instructions' sizes).  ``relax``: gas's -mrelax (its default) /
    -mno-relax; ``link_relax``: ld's default / --no-relax."""
    from . import rvld
    from .assembler import Program, SourceLine
    if arch not in ("rv32", "rv64"):
        raise ValueError(f"arch must be rv32 or rv64, not {arch!r}")
    asm = Assembler(32 if arch == "rv32" else 64, extensions, relax, strict)
    obj = asm.assemble(source, path, include_dirs)
    try:
        linked = rvld.link(obj, bases, link_relax)
    except rvld.LinkError as exc:
        raise AsmError(exc.line, exc.message,
                       asm.lines[exc.line - 1] if 0 < exc.line <= len(asm.lines) else "") from None
    prog = Program(endian="little", release="RV", arch=arch)
    prog.image = linked.image
    prog.rv_extensions = extensions
    for a, size, line in linked.insns:
        prog.rv[a] = size
        prog.source_map[a] = SourceLine(line, asm.lines[line - 1] if line else "")
    prog.symbols.update(linked.symbols)
    prog.segments = {n.lstrip("."): span for n, span in linked.segments.items()}   # (text ...)
    for name in ("main", "_start", "__start"):
        if name in prog.symbols:
            prog.entry = prog.symbols[name]
            break
    else:
        prog.entry = obj.sections[".text"].address
    prog.warnings = list(asm.warnings)
    prog.printed = list(asm.reader.printed)
    prog.tls = linked.tls
    prog.rv_attributes = dict(asm.attributes)
    return prog
