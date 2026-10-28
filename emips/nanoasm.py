"""The nanoMIPS assembler (PLAN M23 phase 3): GNU ``as`` + ``ld`` for one
source file.

nanoMIPS has its own assembler syntax (MD01247 names, the p32 ABI's
register names, size suffixes ``[16]`` / ``[32]`` / ``[48]`` and format
suffixes ``[gp]`` / ``[u12]`` / ...), and its assembler picks the size:
GNU ``as`` matches each instruction against the entries of binutils'
opcode table that have its mnemonic, **in table order**, and keeps the
first that fits (``tc-nanomips.c: match_insns``) -- so ``addiu $a0, $a0,
4`` is the 16-bit ADDIU[R2], ``li $a0, 1000`` ADDIU[32] and ``li $a0,
0x12345678`` LI[48].  A branch to a label of the same section is matched
in its 16-bit form and *relaxed*: GNU ``as`` iterates over the section's
fragments until every 16-bit branch reaches its target, else it becomes
the 32-bit branch (``relax_segment`` / ``nanomips_relax_frag`` /
``md_convert_frag``).  Everything that needs an address (``la``, a load
from a label, a branch to another section) leaves a relocation that the
linker resolves.

This module repeats both tools, step for step, on their own sources
(MediaTek's nanoMIPS GNU toolchain release 2025.09-02: gas's
``config/tc-nanomips.c``, the generic ``write.c`` / ``expr.c``, binutils'
opcode table ``opcodes/nanomips-opc.c`` -- generated into
``emips/data/nanomips_asm.json``, never edited by hand -- and gold's
``nanomips.cc`` for the relocations):

- the operand tokens (``nanomips_parse_arguments``), GNU expressions
  (``expr.c``'s operator ranks: ``| ^ & !`` bind tighter than ``+ -``),
  ``%hi`` / ``%lo`` / ``%gprel`` / ``%pcrel_hi`` operators;
- ``match_insns`` / ``match_insn`` / every ``match_*_operand``, with the
  second, *lax* pass;
- the macros (``macro``: ``li``, ``la``, a load / store from a label or
  with a large offset, ``b<cond>`` with a constant, ``seq`` ...,
  ``load_register``, ``macro_la`` / ``macro_ld_st`` / ``macro_build``),
  including the gp-relative / absolute alternatives a forward reference
  leaves to relaxation (``relax_start`` / ``relax_switch``);
- fragments, ``append_insn``'s relaxable branches and ``addiu`` of a
  label difference, ``.align`` in code (``nanomips_handle_align``: a
  NOP16 then NOP32s), GNU's directives;
- the object it writes (``write.c``'s ``fixup_segment``, ``tc_gen_reloc``,
  the code sections' relocations sorted by offset): what gas resolves
  itself, what stays a relocation and against which symbol;
- ``.linkrelax`` (``--linkrelax``): the object is marked for the
  linker's transformations -- every reference in code stays a relocation
  against its label, ``.align`` leaves R_NANOMIPS_ALIGN (+ FILL / MAX),
  ``.set nolinkrelax`` / ``linkrelax`` R_NANOMIPS_NORELAX / RELAX, a
  size forced by a suffix R_NANOMIPS_INSN16 / INSN32 / FIXED, a data
  directive's address explicit relocations (a label difference in code:
  R_NANOMIPS_NEG + R_NANOMIPS_32 unless nothing the linker may change
  lies between the labels -- gas's ``link_var`` frags, ``mark_if_
  invariable_branch``);
- the link, :mod:`emips.nanold` (gold): the sections laid out at the
  eMIPS segment bases, the expansions and (``link={"relax": True}`` =
  ``--relax``) relaxations of a ``.linkrelax`` object, then each
  relocation as gold applies it (value, field, overflow check);
  ``Program.link_log`` = what gold's ``--debug=target`` prints.

Not modelled: PIC / PID / TLS (GOT relocations), the balc stubs
(``-mbalc-stubs``, off by default), jump tables (``.jumptable``),
``.insn``, ``.reloc``, the DSP / MT / MCU / MSA / VZ instructions (phase
5).  ``tests/test_nanomips_asm.py`` compares the bytes with MediaTek's
``nanomips-elf-as`` + ``ld`` (``EMIPS_NANOMIPS_TOOLS``).

Sections and their eMIPS addresses: ``.text`` (code), ``.data``
(``.rdata`` / ``.rodata``, ``.sdata`` / ``.sbss`` (``_gp`` = the start
of the small data, as the toolchain's ``uhi32.ld``), ``.bss`` follow it
in that order), and eMIPS's ``.ktext`` / ``.kdata`` (the kernel
segments of the MIPS32 assembler).
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from . import gasread, nanold
from .assembler import AsmError, Program, SEGMENT_BASES, SourceLine, _strip_comment

M32 = 0xFFFFFFFF

_TABLES = json.loads((Path(__file__).parent / "data" / "nanomips_asm.json").read_text())

# ------------------------------------------------------------------ tables

ASE = {"DSP": 0x1, "DSP64": 0x2, "EVA": 0x4, "MCU": 0x8, "MT": 0x10, "VIRT": 0x20,
       "VIRT64": 0x40, "MSA": 0x80, "MSA64": 0x100, "CRC": 0x200, "CRC64": 0x400,
       "GINV": 0x800, "GINV_VIRT": 0x1000, "xNMS": 0x2000, "TLB": 0x4000}
#: the opcode table's ASE names (binutils_opcodes_nanomips-opc.c)
_TABLE_ASE = {"D32": "DSP", "MT32": "MT", "MC": "MCU", "EVA": "EVA", "MSA": "MSA",
              "MSA64": "MSA64", "XPA": None, "GINV": "GINV", "CRC": "CRC",
              "xNMS": "xNMS", "TLB": "TLB"}
#: gas's .set / -m ASE names (nanomips_ases)
SET_ASES = {"dsp": "DSP", "dspr3": "DSP", "eva": "EVA", "ginv": "GINV", "crc": "CRC",
            "mcu": "MCU", "msa": "MSA", "mt": "MT", "tlb": "TLB", "virt": "VIRT"}


@dataclass(frozen=True)
class Opcode:
    """One entry of binutils' nanoMIPS opcode table."""
    index: int
    name: str
    suffix: str
    args: str
    match: int
    mask: int
    macro: str | None
    pinfo: frozenset
    pinfo2: frozenset
    isa32: bool                       # membership I38
    ase: int                          # the ASE bits it needs (0: none)

    @property
    def length(self) -> int:
        """insn_length: 6, 4 or 2 bytes (a macro: 4)."""
        if self.macro is None and self.mask >> 16 == 0 and self.match >> 10 == 0x18:
            return 6
        return 4 if self.mask >> 16 else 2

    @property
    def is_macro(self) -> bool:
        return self.macro is not None


def _opcodes() -> list[Opcode]:
    out = []
    for i, e in enumerate(_TABLES["opcodes"]):
        ase = 0
        unknown = False
        for a in e["ase"]:
            name = _TABLE_ASE.get(a, "?")
            if name is None or name == "?":
                unknown = True
            else:
                ase |= ASE[name]
        if unknown:
            ase = -1                      # never enabled
        out.append(Opcode(i, e["name"], e["suffix"], e["args"], e.get("match", 0),
                          e.get("mask", 0), e.get("macro"), frozenset(e["pinfo"]),
                          frozenset(e["pinfo2"]), "I38" in e["membership"], ase))
    return out


OPCODES = _opcodes()
#: name -> the first entry of that name; name + suffix -> its first entry
_FIRST: dict[str, int] = {}
for _o in OPCODES:
    _FIRST.setdefault(_o.name, _o.index)
for _o in OPCODES:
    if _o.suffix:
        _FIRST.setdefault(_o.name + _o.suffix, _o.index)


@dataclass(frozen=True)
class Operand:
    """decode_nanomips_operand's descriptor of one operand string."""
    code: str
    type: str
    size: int
    lsb: int
    size_top: int = 0
    lsb_top: int = 0
    max_val: int = 0
    bias: int = 0
    shift: int = 0
    print_hex: bool = False
    int_map: tuple | None = None
    reg_type: str | None = None
    reg_map: tuple | None = None
    reg1_map: tuple | None = None
    reg2_map: tuple | None = None
    add_lsb: bool = False
    opsize: int = 0
    greater_than_ok: bool = False
    less_than_ok: bool = False
    equal_ok: bool = False
    zero_ok: bool = False
    const_ok: bool = False
    expr_ok: bool = False

    def insert(self, insn: int, uval: int) -> int:
        """nanomips_insert_operand."""
        bottom = self.size - self.size_top
        mask = (1 << bottom) - 1
        insn &= ~(mask << self.lsb)
        insn |= (uval & mask) << self.lsb
        mask = (1 << self.size_top) - 1
        insn &= ~(mask << self.lsb_top)
        insn |= ((uval & (mask << bottom)) >> bottom) << self.lsb_top
        return insn & M32

    def extract(self, insn: int) -> int:
        """nanomips_extract_operand."""
        bottom = self.size - self.size_top
        uval = (insn >> self.lsb_top) & ((1 << self.size_top) - 1)
        uval <<= bottom
        return uval | ((insn >> self.lsb) & ((1 << bottom) - 1))

    @property
    def mask(self) -> int:
        """nanomips_operand_mask."""
        m = ((1 << self.size_top) - 1) << self.lsb_top
        return m | ((1 << (self.size - self.size_top)) - 1) << self.lsb

    def int_max(self) -> int:
        return (self.max_val + self.bias) << self.shift

    def int_min(self) -> int:
        return self.int_max() - (((1 << self.size) - 1) << self.shift)

    @property
    def optional(self) -> bool:
        """nanomips_optional_operand_p."""
        return (self.type in ("OPTIONAL_REG", "REPEAT_PREV_REG")
                or (self.type != "INT" and self.size == 0 and self.lsb == 0))


def _operands() -> dict[str, Operand]:
    out = {}
    for code, d in _TABLES["operands"].items():
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
        out[code] = Operand(code, **kw)
    return out


OPERANDS = _operands()


def operand_code(args: str, i: int) -> str:
    """The operand string starting at ``args[i]`` (one or two characters)."""
    return args[i:i + 2] if args[i] in "+m-`" else args[i]


# --------------------------------------------------------------- registers
#
# tc-nanomips.c: the register symbols of md_begin (RTYPE_* | number)

RNUM_MASK = 0x3FF
RTYPE_NUM, RTYPE_FPU, RTYPE_FCC, RTYPE_VEC = 0x400, 0x800, 0x1000, 0x2000
RTYPE_GP, RTYPE_CP0, RTYPE_PC, RTYPE_ACC = 0x4000, 0x8000, 0x10000, 0x20000
RTYPE_CCC, RTYPE_MSA = 0x40000, 0x80000
RTYPE_CP0SEL_EVEN, RTYPE_CP0SEL_ODD, RTYPE_CP0SEL_LO16 = 0x100000, 0x200000, 0x400000
RTYPE_CP0SEL = RTYPE_CP0SEL_EVEN | RTYPE_CP0SEL_ODD | RTYPE_CP0SEL_LO16
RTYPE_HWR = 0x800000
CP0SEL_SHIFT = 5
_SELMASK = {RTYPE_CP0SEL_EVEN: 0x55555555, RTYPE_CP0SEL_EVEN | RTYPE_CP0SEL_LO16: 0xAAAA,
            RTYPE_CP0SEL_ODD: 0xAAAAAAAA, RTYPE_CP0SEL_ODD | RTYPE_CP0SEL_LO16: 0x5555,
            RTYPE_CP0SEL_LO16: 0xFFFF, RTYPE_CP0SEL: 0xFFFFFFFF}
#: the select masks' register types (md_begin's switch)
_SELTYPE = {0x55555555: RTYPE_CP0SEL_EVEN, 0x5555: RTYPE_CP0SEL_ODD | RTYPE_CP0SEL_LO16,
            0xAAAAAAAA: RTYPE_CP0SEL_ODD, 0xAAAA: RTYPE_CP0SEL_EVEN | RTYPE_CP0SEL_LO16,
            0xFFFF: RTYPE_CP0SEL_LO16, 0xFFFFFFFF: RTYPE_CP0SEL}

#: the p32 ABI's names (DN00179 ch. 3)
GPR_NAMES = ["zero", "at", "t4", "t5", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
             "t0", "t1", "t2", "t3", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
             "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"]


def _registers() -> dict[str, int]:
    r: dict[str, int] = {}
    for i in range(32):
        r[f"${i}"] = RTYPE_NUM | i
    for i, n in enumerate(GPR_NAMES):
        r[f"${n}"] = RTYPE_GP | i
    r["$AT"] = RTYPE_GP | 1
    for i in range(32):
        r[f"$r{i}"] = RTYPE_GP | i
    for i in range(32):
        r[f"$f{i}"] = RTYPE_FPU | i
    for i in range(8):
        r[f"$fcc{i}"] = RTYPE_FCC | i
    for i in range(8):
        r[f"$cc{i}"] = RTYPE_FCC | RTYPE_CCC | i
    for i in range(4):
        r[f"$ac{i}"] = RTYPE_ACC | i
    for i in range(32):
        r[f"$w{i}"] = RTYPE_MSA | i
    for name, num, sel in _TABLES["cp0"]:
        r.setdefault(name, RTYPE_CP0 | num << CP0SEL_SHIFT | sel)
    for name, num, selmask in _TABLES["cp0sel"]:
        r.setdefault(name, num | _SELTYPE[selmask])
    for name, num, sel, cp0num, cp0sel in _TABLES["hwr"]:
        if name in r:
            if cp0num != 0xFFFFFFFF or cp0sel != 0xFFFFFFFF:
                r[name] |= RTYPE_HWR
        else:
            r[name] = RTYPE_HWR | num << CP0SEL_SHIFT | sel
    return r


REGISTERS = _registers()
_HWR = [tuple(h) for h in _TABLES["hwr"]]

ZERO, ATREG, SP, GP, RA = 0, 1, 29, 28, 31

# ------------------------------------------------------------- expressions


@dataclass
class Sym:
    """A gas symbol: a label (section, frag, offset in the frag), an
    absolute value, or not defined (yet)."""
    name: str
    section: "Section | None" = None
    frag: "Frag | None" = None
    where: int = 0
    absolute: int | None = None
    #: an equate to another symbol: (symbol, addend)
    equated: tuple | None = None
    line: int = 0
    #: S_IS_EXTERNAL (.globl) / S_IS_WEAK (.weak)
    external: bool = False
    weak: bool = False

    @property
    def defined(self) -> bool:
        if self.equated is not None:
            return self.equated[0].defined
        return self.absolute is not None or self.frag is not None

    def resolve(self) -> tuple["Sym", int]:
        """(the label or absolute symbol this one stands for, addend)."""
        sym, add = self, 0
        for _ in range(64):
            if sym.equated is None:
                return sym, add
            nxt, n = sym.equated
            sym, add = nxt, add + n
        raise ValueError(f"symbol {self.name} is defined in terms of itself")

    def segment(self):
        """S_GET_SEGMENT: a Section, "abs" or None (undefined)."""
        s, _ = self.resolve()
        if s.absolute is not None:
            return "abs"
        return s.section

    def value(self) -> int:
        """S_GET_VALUE: the section-relative value (relaxation time)."""
        s, add = self.resolve()
        if s.absolute is not None:
            return s.absolute + add
        if s.frag is None:
            return add
        return s.frag.address + s.where + add


@dataclass
class Expr:
    """expressionS: ``op`` "constant" (num), "symbol" (sym + num),
    "subtract" (sym - sub + num), "register" (num = the register
    symbol), "absent", "big" (a value wider than 64 bits) or "illegal"."""
    op: str = "absent"
    num: int = 0
    sym: Sym | None = None
    sub: Sym | None = None
    #: a difference of two labels (symbol_constant_p: no equates)
    plain: bool = False
    #: a difference's symbols as written (before their equates resolve)
    orig: tuple | None = None

    def copy(self) -> "Expr":
        return Expr(self.op, self.num, self.sym, self.sub, self.plain, self.orig)


#: expr.c's op_rank
_RANK = {"*": 8, "/": 8, "%": 8, "<<": 8, ">>": 8, "|": 7, "!": 7, "^": 7, "&": 7,
         "+": 5, "-": 5, "==": 4, "<>": 4, "!=": 4, "<": 4, "<=": 4, ">=": 4, ">": 4,
         "&&": 3, "||": 2}
_NAME_START = re.compile(r"[A-Za-z_.$]")
_NAME = re.compile(r"[A-Za-z_.$][\w.$]*")
_NUMBER = re.compile(r"0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+")


def _sx64(v: int) -> int:
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >> 63 else v


class _Expressions:
    """GNU ``expr.c``: an expression at a position of a string."""

    def __init__(self, asm: "Assembler") -> None:
        self.asm = asm

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
            raise ValueError("missing ')'")
        if c in "-~!+":
            e, i = self._operand(s, i + 1)
            if c == "+":
                return e, i
            if e.op != "constant":
                if e.op in ("absent", "register"):
                    return e, i
                raise ValueError("expression too complex")
            n = {"-": -e.num, "~": ~e.num, "!": int(not e.num)}[c]
            return Expr("constant", _sx64(n)), i
        if c == "'":
            # a character constant: app.c makes 'c' (or 'c; \\b \\f \\n
            # \\r \\t escapes, process_escape) its decimal value
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
            text = m.group(0)
            end = m.end()
            # a numeric local label reference: 1b / 1f (renamed before)
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
            name = m.group(0)
            if name in REGISTERS:
                return Expr("register", REGISTERS[name]), m.end()
            return self._symbol(name), m.end()
        return Expr("absent"), i

    def _symbol(self, name: str) -> Expr:
        sym = self.asm.symbol(name)
        target, add = sym.resolve()
        if target.absolute is not None:
            return Expr("constant", _sx64(target.absolute + add))
        return Expr("symbol", 0, sym)

    def _combine(self, op: str, a: Expr, b: Expr) -> Expr:
        if a.op == "register" or b.op == "register":
            return Expr("register", a.num if a.op == "register" else b.num)
        if a.op == "constant" and b.op == "constant":
            x, y = a.num, b.num
            if op in ("/", "%") and y == 0:
                raise ValueError("division by zero")
            r = {"*": lambda: x * y,
                 "/": lambda: int(x / y),
                 "%": lambda: x - int(x / y) * y,
                 "<<": lambda: x << (y & 63) if y >= 0 else 0,
                 ">>": lambda: (x & ((1 << 64) - 1)) >> (y & 63) if y >= 0 else 0,
                 "|": lambda: x | y, "!": lambda: x | ~y, "^": lambda: x ^ y,
                 "&": lambda: x & y, "+": lambda: x + y, "-": lambda: x - y,
                 "==": lambda: -int(x == y), "!=": lambda: -int(x != y),
                 "<>": lambda: -int(x != y), "<": lambda: -int(x < y),
                 "<=": lambda: -int(x <= y), ">=": lambda: -int(x >= y),
                 ">": lambda: -int(x > y), "&&": lambda: int(bool(x and y)),
                 "||": lambda: int(bool(x or y))}[op]()
            return Expr("constant", _sx64(r))
        if op == "+":
            if a.op == "symbol" and b.op == "constant":
                return Expr("symbol", _sx64(a.num + b.num), a.sym)
            if a.op == "constant" and b.op == "symbol":
                return Expr("symbol", _sx64(a.num + b.num), b.sym)
            if a.op == "subtract" and b.op == "constant":
                return Expr("subtract", _sx64(a.num + b.num), a.sym, a.sub, a.plain, a.orig)
            if a.op == "constant" and b.op == "subtract":
                return Expr("subtract", _sx64(a.num + b.num), b.sym, b.sub, b.plain, b.orig)
        if op == "-":
            if a.op == "symbol" and b.op == "constant":
                return Expr("symbol", _sx64(a.num - b.num), a.sym)
            if a.op == "subtract" and b.op == "constant":
                return Expr("subtract", _sx64(a.num - b.num), a.sym, a.sub, a.plain, a.orig)
            if a.op == "symbol" and b.op == "symbol":
                return self.asm.difference(a, b)
        raise ValueError("expression too complex")


def _skip(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t":
        i += 1
    return i


#: nanomips_percent_op, in its order
_PERCENT = [("%lo", "LO12"), ("%call16", "GOT_CALL"), ("%got_call", "GOT_CALL"),
            ("%got_disp", "GOT_DISP"), ("%got_page", "GOT_PAGE"),
            ("%got_ofst", "GOT_OFST"), ("%got", "GOT_DISP"), ("%gp_rel", "GPREL18"),
            ("%gprel", "GPREL18"), ("%gprel_hi", "GPREL_HI20"),
            ("%gprel_lo", "GPREL_LO12"), ("%gprel32", "GPREL_I32"), ("%hi", "HI20"),
            ("%pcrel_hi", "PCREL_HI20"), ("%pcrel_lo", "LO12"),
            ("%got_pcrel_hi", "GOTPC_HI20"), ("%got_pcrel_lo", "GOT_LO12"),
            ("%got_lo", "GOT_LO12"), ("%got_pcrel32", "GOTPC_I32"), ("%tlsgd", "TLS_GD"),
            ("%tlsld", "TLS_LD"), ("%dtprel", "TLS_DTPREL12"), ("%tprel", "TLS_TPREL12"),
            ("%gottprel", "TLS_GOTTPREL"), ("%gottprel_pc32", "TLS_GOTTPREL_PC_I32"),
            ("%call_hi", None), ("%call_lo", None)]
#: relocation operators eMIPS has no use for (no GOT, no TLS: no PIC code)
_UNSUPPORTED = frozenset({"GOT_CALL", "GOT_DISP", "GOT_PAGE", "GOT_OFST", "GOTPC_HI20",
                          "GOT_LO12", "GOTPC_I32", "TLS_GD", "TLS_LD", "TLS_DTPREL12",
                          "TLS_TPREL12", "TLS_GOTTPREL", "TLS_GOTTPREL_PC_I32"})


# ----------------------------------------------------------- relocations
#
# gas works with BFD relocation codes (here without the BFD_RELOC_ /
# NANOMIPS_ prefixes); bfd/elfnn-nanomips.c maps them to the R_NANOMIPS_*
# types an object file carries (and gold applies).

HOWTOS = _TABLES["howtos"]
BFD_TO_R = dict(_TABLES["bfd_relocs"], LO16="LO12", HI16="HI20", HI16_S="HI20",
                GPREL16="GPREL18")

_HI = frozenset({"HI16_S", "HI20", "GOTPC_HI20", "GPREL_HI20"})
_LO = frozenset({"LO16", "LO12", "GPREL_LO12", "GOT_LO12"})
_GPREL = frozenset({"GPREL16", "GPREL7_S2", "GPREL18", "GPREL19_S2", "GPREL16_S2",
                    "GPREL18_S3", "GPREL17_S1", "TLS_GD", "TLS_LD"})
_PCREL16 = frozenset({"4_PCREL_S1", "7_PCREL_S1", "10_PCREL_S1"})
_PCREL32 = frozenset({"32_PCREL", "11_PCREL_S1", "14_PCREL_S1", "21_PCREL_S1",
                      "25_PCREL_S1", "PCREL_HI20", "GOTPC_HI20"})
_PCREL48 = frozenset({"PC_I32", "GOTPC_I32", "TLS_GOTTPREL_PC_I32"})
_R48 = frozenset({"I32", "PC_I32", "GPREL_I32", "GOTPC_I32", "TLS_GD_I32", "TLS_LD_I32",
                  "TLS_DTPREL_I32", "TLS_TPREL_I32", "TLS_GOTTPREL_PC_I32"})
_FLEX = frozenset({"GPREL18", "TLS_LD", "TLS_GD", "TLS_TPREL12", "TLS_DTPREL12"})
_GPREL_I32 = frozenset({"GPREL_I32", "TLS_LD_I32", "TLS_GD_I32"})


def _pcrel_branch(r) -> bool:
    return r in _PCREL16 or (r in _PCREL32 and r not in ("PCREL_HI20", "GOTPC_HI20"))


def _pcrel(r) -> bool:
    return r in _PCREL16 or r in _PCREL32 or r in _PCREL48


def _howto(r: str) -> dict:
    return HOWTOS[BFD_TO_R[r]]


# the frag relaxation subtypes (tc-nanomips.c: relax_nanomips_subtype)
RT_NONE, RT_PC4_S1, RT_PC7_S1, RT_PC10_S1, RT_PC11_S1, RT_PC14_S1 = range(6)
RT_PC21_S1, RT_PC25_S1, RT_ADDIU, RT_BRANCH_UCND, RT_BRANCH_CNDZ = range(6, 11)
RT_BRANCH_CND = 11
_RT_OF = {"4_PCREL_S1": RT_PC4_S1, "7_PCREL_S1": RT_PC7_S1, "10_PCREL_S1": RT_PC10_S1,
          "11_PCREL_S1": RT_PC11_S1, "14_PCREL_S1": RT_PC14_S1,
          "21_PCREL_S1": RT_PC21_S1, "25_PCREL_S1": RT_PC25_S1}


@dataclass(frozen=True)
class Relax:
    """A pending relaxation in place of a relocation (BFD_RELOC_UNUSED + RT)."""
    rt: int


def _rt_reloc(rt: int, toofar16: bool) -> str | None:
    """frag_subtype_to_reloc."""
    for name, v in _RT_OF.items():
        if v == rt:
            return name
    if rt == RT_BRANCH_UCND:
        return "25_PCREL_S1" if toofar16 else "10_PCREL_S1"
    if rt == RT_BRANCH_CNDZ:
        return "14_PCREL_S1" if toofar16 else "7_PCREL_S1"
    if rt == RT_BRANCH_CND:
        return "14_PCREL_S1" if toofar16 else "4_PCREL_S1"
    return None


# ------------------------------------------------------- frags and fixups


class Section:
    def __init__(self, name: str) -> None:
        self.name = name
        self.frags: list[Frag] = []
        self.align = 0                      # record_alignment: log2
        self.code = name in (".text", ".ktext") or name.startswith(".text.")
        self.address = 0                    # where the link puts it
        self.size = 0

    def __repr__(self) -> str:
        return f"<{self.name}>"


class Frag:
    """A gas fragment: ``fix`` the fixed part, then a variable part --
    ``kind`` "fill" (none; or ``repeat`` copies of ``pattern``: .space),
    "align" / "align_code" (padding to 1 << ``power``, at most
    ``max_skip`` bytes), "md" (a relaxable instruction, ``var`` bytes, or
    the two sequences of a relax_start / relax_switch macro)."""

    def __init__(self, section: Section, line: int) -> None:
        self.section = section
        self.fix = bytearray()
        self.kind = "fill"
        self.var = 0
        self.max = 0
        self.subtype = 0
        self.symbol: Sym | None = None
        self.offset = 0
        self.address = 0
        self.line = line
        self.repeat = 0
        self.pattern = b""
        self.power = 0
        self.max_skip = 0
        self.fill = b"\0"
        #: bytes of the variable part (the relaxable instruction)
        self.literal = bytearray()
        #: a relaxable sequence: (first, second) sizes, use the second
        self.first = self.second = 0
        self.use_second = False
        #: line of each byte range of this frag: (offset, size, line)
        self.lines: list[tuple[int, int, int, int]] = []
        #: tc_frag_data: link-time variable (a relocation the linker may
        #: transform), the bytes it may add / remove, the first fixup
        self.link_var = False
        self.relax_sop = 0
        self.relax_sink = 0
        self.first_fix: "Fix | None" = None
        #: the R_NANOMIPS_ALIGN of an .align (its symbol's size: the padding)
        self.align_fix: "Fix | None" = None

    @property
    def size(self) -> int:
        """The frag's size after conversion."""
        if self.kind == "fill":
            return len(self.fix) + self.repeat * len(self.pattern)
        return len(self.fix) + self.var


@dataclass
class Fix:
    frag: Frag
    where: int
    size: int
    expr: Expr
    reloc: str                 # BFD code (gas level)
    line: int
    done: bool = False
    #: the sequence of a relax_start / relax_switch pair it belongs to (1 / 2)
    seq: int = 0
    insn_size: int = 0
    #: an ALIGN / FILL / MAX placeholder: its absolute symbol (name, value, size)
    place: tuple | None = None


@dataclass
class Insn:
    """nanomips_cl_insn."""
    mo: Opcode
    opcode: int
    ext: int = 0
    complete: bool = False

    @property
    def length(self) -> int:
        return self.mo.length


class _Arg:
    """nanomips_arg_info: the state while matching one entry."""

    def __init__(self, insn: Insn, tokens: list, lax: bool) -> None:
        self.insn = insn
        self.tokens = tokens
        self.pos = 0
        self.opnum = 0
        self.argnum = 1
        self.last_regno = 0xFFFFFFFF
        self.dest_regno = 0xFFFFFFFF
        self.last_op_int = 0
        self.lax = lax
        self.seen_at = False
        self.select_mask = 0

    @property
    def token(self):
        return self.tokens[self.pos]


# operand tokens (nanomips_operand_token_type)
T_REG, T_REG_RANGE, T_INTEGER, T_FLOAT, T_CHAR, T_DOUBLE, T_END = range(7)


@dataclass
class Token:
    type: int
    value: object = None              # register symval / (expr, relocs) / char / (r1, r2)


class _MatchError(Exception):
    pass


class _NoFormat(AssertionError):
    """macro_build found no entry with that operand format."""



# --------------------------------------------------------------- options


@dataclass
class Options:
    """nanomips_set_options (the part eMIPS uses)."""
    ase: int = ASE["xNMS"] | ASE["TLB"]          # -march=32r6 (gas's default)
    noreorder: int = 0
    at: int = ATREG
    nomacro: bool = False
    insn32: bool = False
    pcrel: bool = False
    mc_model: str = "auto"
    linkrelax: bool = False

    def copy(self) -> "Options":
        return Options(self.ase, self.noreorder, self.at, self.nomacro, self.insn32,
                       self.pcrel, self.mc_model, self.linkrelax)


#: macro_build's operand formats that depend on the options
def _brk_fmt(o: Options) -> str:
    return "+J" if o.insn32 else "mK"


ADDIU_FMT = "+t,r,j"
BITOP_IMM_FMT = "t,r,g"
OP_IMM_FMT = "t,r,i"
SHFT_FMT = "t,r,<"
TRAP_FMT = "s,t,|"
BCONDZ1_FMT, BCONDZ2_FMT = "s,p", "t,p"
B_FMT = "+u"
LUI_FMT = "t,u"
MOVE_FMT = "mb,mj"
LWGP_FMT = "t,.(ma)"
ADDIUGP_FMT, ADDIUGPS_FMT = "t,ma,.", "t,ma,+1"
LL_SC_FMT = "t,+p(b)"
LLP_SCP_FMT = "t,e,(b)"
ISA_SIGNED_LDST_FMT = "t,+j(b)"
ISA_UNSIGNED_LDST_FMT = "t,o(b)"
ISA_MULTI_LDST_FMT = "t,+j(b),|"
PREF_FMT = "k,+j(b)"
ISA_OFFBITS, ISA_SIGNED_OFFBITS, ISA_ADD_OFFBITS = 12, 9, 16
MAX_GPREL_OFFSET = 0x1FFFC
GPR_SMAX, GPR_SMIN = 0x7FFFFFFF, -0x80000000


def _is_sext32(x: int) -> bool:
    return (x & ~0x7FFFFFFF) == 0 or (x & ~0x7FFFFFFF) == ~0x7FFFFFFF


def _is_zext32(x: int) -> bool:
    return (x & ~0xFFFFFFFF) == 0 or (x & ~0xFFFFFFFF) == ~0xFFFFFFFF


def _sext16_uint(x: int) -> bool:
    return (x & ~0xFFFF) == 0


def _normalize(x: int) -> int:
    """normalize_constant_expr: a zero-extended 32-bit value sign-extended."""
    if _is_zext32(x):
        return ((x & M32) ^ 0x80000000) - 0x80000000
    return x


def _offset_high_part(value: int, offbits: int) -> int:
    if offbits == 0:
        return value
    bias = 1 << (offbits - 1)
    low = bias * 2 - 1
    if offbits == ISA_OFFBITS:
        return value & ~low
    return (value + bias) & ~low


def _offset_high_unsigned(value: int, offbits: int) -> int:
    if offbits == 0:
        return value
    return value & ~((1 << (offbits - 1)) * 2 - 1)


_SMALL_DATA = (".sdata", ".sbss", ".ssdata", ".ssbss")


def _small_data_section(name: str) -> bool:
    return (name in _SMALL_DATA or name.startswith((".sdata.", ".sbss.", ".ssdata.",
                                                    ".ssbss.", ".gnu.linkonce.sb.",
                                                    ".gnu.linkonce.s.",
                                                    ".gnu.linkonce.ssb.",
                                                    ".gnu.linkonce.ss.")))


def _special_small_data_section(name: str) -> bool:
    return name in (".ssdata", ".ssbss") or name.startswith(
        (".ssdata.", ".ssbss.", ".gnu.linkonce.ssb.", ".gnu.linkonce.ss."))


#: the section order of the data segment (the toolchain's uhi32.ld: small
#: data after _gp, then .bss)
_DATA_ORDER = (".data", ".rdata", ".rodata", ".ssdata", ".sdata", ".ssbss", ".sbss", ".bss")


# -------------------------------------------------------------- assembler


class _Core:
    """One run of ``nanomips-elf-as`` + ``nanomips-elf-ld`` (see the
    module docstring).  ``nms``: assemble for the nanoMIPS subset (gas
    ``-march=32r6s``: no xNMS instructions); ``crc`` / ``ginv`` / ``eva``
    / ``tlb`` / ``dsp`` / ``mt``: the ASEs (``-mcrc`` / ``-mginv`` /
    ``-meva`` / ``-mdsp`` / ``-mmt``; TLB is on by default, as with gas's
    ``32r6``)."""

    def __init__(self, *, endian: str = "big", bases: dict[str, int] | None = None,
                 nms: bool = False, crc: bool = False, ginv: bool = False,
                 eva: bool = False, tlb: bool = True, dsp: bool = False, mt: bool = False,
                 pseudo: bool = True,
                 reorder: bool = True, linkrelax: bool = False,
                 link: "nanold.LinkOptions | None" = None) -> None:
        if endian not in ("big", "little"):
            raise ValueError("endian must be 'big' or 'little'")
        self.endian = endian
        #: macros allowed (False: eMIPS's strict mode)
        self.pseudo = pseudo
        self.reorder = reorder
        self.big = endian == "big"
        self.bases = dict(SEGMENT_BASES, **(bases or {}))
        ase = (0 if nms else ASE["xNMS"]) | (ASE["TLB"] if tlb else 0)
        ase |= (ASE["CRC"] if crc else 0) | (ASE["GINV"] if ginv else 0)
        ase |= (ASE["EVA"] if eva else 0) | (ASE["DSP"] if dsp else 0) | (ASE["MT"] if mt else 0)
        self.file_opts = Options(ase=ase)
        self.g_switch = 8
        #: gas --linkrelax; ld's options
        self.linkrelax_option = linkrelax
        self.link_options = link or nanold.LinkOptions()

    # -- the run -------------------------------------------------------------

    def assemble(self, source: str, path: str | None = None,
                 include_dirs: tuple[str, ...] = ()) -> Program:
        """Assemble and link ``source``."""
        self._run(source, path, include_dirs)
        prog = self._finish()
        prog.printed = list(self.reader.printed)
        return prog

    def _run(self, source: str, path: str | None = None,
             include_dirs: tuple[str, ...] = ()) -> None:
        self.linkrelax = False                 # gas's linkrelax: the object may be relaxed
        self.file_checked = False              # file_nanomips_opts_checked
        self.toggle_linkrelax = False          # toggle_linkrelax_p
        self.assembling_insn = False           # nanomips_assembling_insn
        self.relaxed = False                   # relax_segment has run (frag addresses)
        self.align_symidx = 1                  # create_align_relocs' symidx
        if self.linkrelax_option:              # --linkrelax = s_linkrelax
            self.s_linkrelax()
        self.opts = self.file_opts.copy()
        self.opts.noreorder = 0 if self.reorder else 1
        self.built: list[str] | None = None
        self.opts_stack: list[Options] = []
        self.symbols: dict[str, Sym] = {}
        self.sections: dict[str, Section] = {}
        self.fixups: list[Fix] = []
        self.warnings: list[tuple[int, str]] = []
        self.line = 0
        self.raw = ""
        self.auto_align = True
        self.labels: list[Sym] = []          # labels at the current point (insn_labels)
        self.history: Insn | None = None
        self.label_counter = 0               # nanomips_target_label
        self.label_name = ""
        self.relax_seq = 0                   # nanomips_relax.sequence
        self.relax_sizes = [0, 0]
        self.relax_symbol: Sym | None = None
        self.relax_first_fix: Fix | None = None
        self.macro_insns = [0, 0]
        self.code_lines: dict[tuple[int, int], int] = {}   # (frag id, offset) -> line
        self.prev_section: Section | None = None
        self.section = self._section(".text")
        self.frag = self.section.frags[-1]
        lines = source.splitlines()
        self._lines = lines
        self.reader = gasread.Reader(source, _Hooks(self), path=path, include_dirs=include_dirs,
                                     warn=lambda n, msg: self.warnings.append((n, msg)))
        for st in self.reader:
            self.line, self.raw = st.line, st.raw
            try:
                self._statement(st.text)
            except AsmError:
                raise
            except (ValueError, KeyError, _MatchError) as exc:
                raise AsmError(st.line, str(exc), st.raw) from None

    def error(self, msg: str):
        raise AsmError(self.line, msg, self.raw)

    def s_linkrelax(self) -> None:
        """The .linkrelax directive (and --linkrelax)."""
        if self.file_checked:
            self.error(".linkrelax is not permitted after generating code")
        self.linkrelax = True
        self.file_opts.linkrelax = True
        if hasattr(self, "opts"):
            self.opts.linkrelax = True

    def warn(self, msg: str) -> None:
        self.warnings.append((self.line, msg))

    # -- symbols --------------------------------------------------------------

    def symbol(self, name: str) -> Sym:
        s = self.symbols.get(name)
        if s is None:
            s = self.symbols[name] = Sym(name)
        return s

    def dot(self) -> Sym:
        """A temporary symbol at the current location (``.``)."""
        return Sym(".", self.section, self.frag, len(self.frag.fix), line=self.line)

    def define_label(self, name: str) -> Sym:
        s = self.symbol(name)
        if s.defined:
            self.error(f"symbol `{name}' is already defined")
        s.section, s.frag, s.where, s.line = self.section, self.frag, len(self.frag.fix), self.line
        self.labels.append(s)
        return s

    def difference(self, a: Expr, b: Expr) -> Expr:
        """``a - b`` of two symbols (expr.c): a constant when both are in
        the same section with only fixed-size frags between them."""
        sa, na = a.sym.resolve()
        sb, nb = b.sym.resolve()
        num = a.num - b.num + na - nb
        if a.sym is b.sym:                     # x - x: 0 even before x is defined
            return Expr("constant", _sx64(a.num - b.num))
        if sa.absolute is not None and sb.absolute is not None:
            return Expr("constant", _sx64(sa.absolute - sb.absolute + num))
        if (sa.frag is not None and sb.frag is not None and sa.section is sb.section
                and self.md_allow_local_subtract(sa, sb)):
            off = self._fixed_offset(sb.frag, sa.frag)
            if off is not None:
                return Expr("constant", _sx64(off + sa.where - sb.where + num))
        plain = a.sym.equated is None and b.sym.equated is None and a.num == b.num == 0
        return Expr("subtract", _sx64(num), sa, sb, plain, (a.sym, b.sym))

    def md_allow_local_subtract(self, left: Sym, right: Sym) -> bool:
        """nanomips_allow_local_subtract: in code, with ``.linkrelax``, a
        data directive's label difference stays symbolic unless nothing
        the linker may change lies between the labels."""
        if not right.section.code or left is right or self.assembling_insn:
            return True
        if not self.linkrelax:
            return True
        return self.allow_local_subtract_symbols(left, right, True)

    def _fixed_offset(self, f1: "Frag", f2: "Frag") -> int | None:
        """frag_offset_fixed_p: the distance from f1 to f2 if only fixed
        frags lie between them (in either order)."""
        for a, b, sign in ((f1, f2, 1), (f2, f1, -1)):
            frags = a.section.frags
            i = frags.index(a)
            off = 0
            for f in frags[i:]:
                if f is b:
                    return sign * off
                if f is self.frag or f.kind != "fill":
                    if f is self.frag and f is not b:
                        break
                    if f.kind != "fill":
                        break
                off += len(f.fix) + f.repeat * len(f.pattern)
        return None

    # -- sections and frags -----------------------------------------------------

    def _section(self, name: str) -> Section:
        s = self.sections.get(name)
        if s is None:
            s = self.sections[name] = Section(name)
            s.frags.append(Frag(s, self.line))
        return s

    def switch_section(self, name: str) -> None:
        self.flush_pending_output()
        self.prev_section = self.section
        self.section = self._section(name)
        self.frag = self.section.frags[-1]
        self.auto_align = True

    def frag_close(self) -> None:
        """frag_new: start a new frag after the current one."""
        f = Frag(self.section, self.line)
        self.section.frags.append(f)
        self.frag = f

    def frag_more(self, data: bytes) -> int:
        at = len(self.frag.fix)
        self.frag.fix += data
        return at

    def record_alignment(self, power: int) -> None:
        self.section.align = max(self.section.align, power)

    # -- statements ------------------------------------------------------------

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
            self._assign(m.group(1), text[m.end():])
            return
        if text.startswith("."):
            name, _, rest = text.partition(" ")
            if "\t" in name:
                name, _, more = name.partition("\t")
                rest = more + " " + rest
            self.directive(name.lower(), rest.strip())
            return
        m = _NAME.match(text)                   # read.c lowercases the opcode's name
        if m:
            text = m.group(0).lower() + text[m.end():]
        self.md_assemble(text)

    # -- instructions (md_assemble) -------------------------------------------

    def md_assemble(self, text: str) -> None:
        self.file_checked = True
        self.assembling_insn = True
        try:
            self._md_assemble(text)
        finally:
            self.assembling_insn = False

    def _md_assemble(self, text: str) -> None:
        if not self.section.code:
            if self.section.name in (".bss", ".sbss") or self.section.name.startswith(".bss."):
                self.error(f"attempt to store non-zero value in section `{self.section.name}'")
        self.imm_expr = Expr("absent")
        self.offset_expr = Expr("absent")
        self.offset_reloc = [None, None, None]
        self.macro_lines = self.line
        insn = self.ip(text)
        if insn.mo.is_macro:
            self.macro_insns = [0, 0]
            if not self.pseudo:                 # strict mode: name what it expands to
                self.built = []
                self.macro(insn)
                main = " + ".join(n for n, q in self.built if q != 1) or "nothing"
                gp = " + ".join(n for n, q in self.built if q == 1)
                self.error(f"`{text.strip()}' is a macro (GNU as makes it {main}"
                           + (f"; {gp} for a symbol in the small data" if gp else "")
                           + "): strict mode wants the machine instructions")
            try:
                self.macro(insn)
            except _NoFormat as exc:
                # gas: "Internal error in macro_build" (an expansion needs an
                # instruction the core does not have)
                self.error(f"GNU as cannot expand `{text.strip()}' here ({exc}: it stops "
                           "with an internal error)")
            if self.opts.nomacro and (self.macro_insns[0] > 1 and self.macro_insns[1] > 1):
                if "INSN2_MACRO" in insn.mo.pinfo2:
                    self.warn("macro instruction expanded into multiple instructions")
                else:
                    self.error("macro instruction expanded into multiple instructions")
        else:
            if self.offset_expr.op != "absent":
                self.append_insn(insn, self.offset_expr, self.offset_reloc)
            else:
                self.append_insn(insn, None, [None, None, None])

    def lookup(self, name: str):
        """nanomips_lookup_insn -> (first entry, forced length, forced format)."""
        forced_format = "[" in name and "]" in name[name.index("["):]
        i = _FIRST.get(name)
        if i is not None:
            return i, 0, forced_format
        dot = name.find(".", name.find("]") if "]" in name else 0)
        end = dot if dot >= 0 else len(name)
        suffix = {"16": 2, "32": 4, "48": 6}.get(name[end - 2:end]) if end >= 3 else None
        if suffix:
            # (gas walks on past the entries of that name until one has
            # the length: nanomips_lookup_insn)
            stripped = name[:end - 2] + name[end:]
            i = _FIRST.get(stripped)
            while i is not None and i < len(OPCODES):
                if OPCODES[i].length == suffix:
                    return i, suffix, forced_format
                if forced_format:
                    break
                i += 1
        return None, 0, forced_format

    def ip(self, text: str) -> Insn:
        """nanomips_ip: match ``text`` against the opcode table."""
        end = 0
        while end < len(text) and text[end] not in " \t":
            end += 1
        mnemonic = text[:end]
        first, self.forced_length, self.forced_format = self.lookup(mnemonic)
        if first is None:
            self.error("unrecognized opcode `" + text.strip() + "'")
        name = OPCODES[first].name
        fmt = {"li.s": "f", "li.d": "d"}.get(name)
        self._err = None                        # insn_error: (min_argnum, template, text)
        tokens = self.parse_arguments(text[end:], fmt)
        for lax in (False, True):
            insn = self.match_insns(first, tokens, lax)
            if insn is not None:
                if isinstance(insn, str):
                    self.error(insn + " `" + text.strip() + "'")
                return insn
        if self._err is None or self._err[1] is None:
            self._set_error(0, "invalid operands")
        self.error(self._err[2] + " `" + text.strip() + "'")

    def _set_error(self, argnum: int, template: str, text: str | None = None) -> bool:
        """set_insn_error_format: errors against later arguments win; two
        different errors against one argument give up on it."""
        text = template if text is None else text
        e = self._err
        if argnum == 0:
            if e is not None and e[1] is not None:
                return False
        else:
            mn = e[0] if e is not None else 0
            if argnum < mn:
                return False
            if e is not None and argnum == mn and e[1] is not None and e[1] != template:
                self._err = (mn + 1, None, None)
                return False
        self._err = (argnum, template, text)
        return True

    def opcode_valid(self, mo: Opcode, ase: int | None = None) -> bool:
        """is_opcode_valid (nanomips_opcode_is_member for 32r6)."""
        if mo.isa32:
            return True
        ase = self.opts.ase if ase is None else ase
        return mo.ase > 0 and (ase & mo.ase) == mo.ase

    def _isa_name(self) -> str:
        """nanomips_cpu_info_from_isa: the subset when xNMS is off."""
        return "32r6" if self.opts.ase & ASE["xNMS"] else "32r6s"

    def _isa_ase(self) -> int:
        return ASE["xNMS"] | ASE["TLB"] if self.opts.ase & ASE["xNMS"] else 0

    def size_valid(self, mo: Opcode) -> bool:
        if self.opts.insn32 and not mo.is_macro and mo.length != 4:
            return False
        if not self.forced_length:
            return True
        if mo.is_macro:
            return False
        return self.forced_length == mo.length

    def match_insns(self, first: int, tokens: list, lax: bool):
        """match_insns: the first entry of that name (and suffix, when
        forced) that matches; a string for an ISA / size error."""
        fmo = OPCODES[first]
        seen_isa = seen_size = seen_ase = False
        i = first
        while i < len(OPCODES) and OPCODES[i].name == fmo.name:
            mo = OPCODES[i]
            i += 1
            if self.forced_format and mo.suffix != fmo.suffix:
                continue
            if self.opcode_valid(mo):
                seen_isa = True
                if self.size_valid(mo):
                    seen_size = True
                    insn = self.match_insn(mo, tokens, lax)
                    if insn is not None:
                        return insn
            elif self.opcode_valid(mo, self._isa_ase() | self.opts.ase):
                seen_ase = True                 # is_opcode_valid_def_ase
        if not seen_isa:
            if seen_ase:
                return "opcode belongs to a disabled ASE:"
            # match_invalid_for_isa (-march names the ISA: 32r6 or 32r6s)
            return f"opcode not supported on this ISA: {self._isa_name()}"
        if not seen_size:
            if self.opts.insn32:
                return "opcode not supported in the `insn32' mode"
            return f"unrecognized {8 * self.forced_length}-bit version of nanoMIPS opcode"
        return None

    # -- operand tokens ---------------------------------------------------------

    def parse_arguments(self, s: str, float_format=None) -> list:
        """nanomips_parse_arguments."""
        toks: list[Token] = []
        i = _skip(s, 0)
        while i < len(s):
            i = self._argument_token(s, i, toks, float_format)
            i = _skip(s, i)
        toks.append(Token(T_END))
        return toks

    def _register_at(self, s: str, i: int):
        m = _NAME.match(s, i)
        if m and m.group(0) in REGISTERS:
            return REGISTERS[m.group(0)], m.end()
        return None, i

    def _argument_token(self, s: str, i: int, toks: list, float_format) -> int:
        if s[i] == "(":                         # "(" + a register: a base
            j = _skip(s, i + 1)
            dec = s.startswith("--", j)
            if dec:
                j = _skip(s, j + 2)
            reg, k = self._register_at(s, j)
            if reg is not None:
                toks.append(Token(T_CHAR, "("))
                if dec:
                    toks.append(Token(T_DOUBLE, "-"))
                toks.append(Token(T_REG, reg))
                k = _skip(s, k)
                if s.startswith("++", k):
                    toks.append(Token(T_DOUBLE, "+"))
                    k += 2
                return k
        if s[i] in "),":
            toks.append(Token(T_CHAR, s[i]))
            return i + 1
        reg, j = self._register_at(s, i)
        if reg is not None:
            j = _skip(s, j)
            if j < len(s) and s[j] == "-":
                reg2, k = self._register_at(s, _skip(s, j + 1))
                if reg2 is None:
                    raise _MatchError("invalid register range")
                toks.append(Token(T_REG_RANGE, (reg, reg2)))
                return k
            toks.append(Token(T_REG, reg))
            if j < len(s) and s[j] == "[":
                raise _MatchError("vector element must be constant")
            return j
        if float_format:
            m = re.compile(r"[-+]?(\d+\.?\d*([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?|inf|nan)",
                           re.I).match(s, i)
            if m:
                v = float(m.group(0))
                data = struct.pack(">f" if float_format == "f" else ">d", v)
                if not self.big:
                    data = data[::-1]
                toks.append(Token(T_FLOAT, data))
                return m.end()
        expr, relocs, j = self.small_expression(s, i)
        toks.append(Token(T_INTEGER, (expr, relocs)))
        return j

    def small_expression(self, s: str, i: int):
        """my_getSmallExpression: one optional relocation operator, then
        the expression."""
        relocs = [None, None, None]
        reversed_ = []
        depth = 0
        crux, crux_depth = i, 0
        j = i
        while True:
            crux, crux_depth = j, depth
            while j < len(s) and s[j] in " \t(":
                if s[j] == "(":
                    depth += 1
                j += 1
            if j < len(s) and s[j] == "%" and len(reversed_) < 1:
                r, k = self._percent(s, j)
                if r is not False:
                    reversed_.append(r)
                    j = k
                    continue
            break
        expr, j = _Expressions(self).parse(s, crux)
        while crux_depth > 0 and j < len(s) and s[j] in ") \t":
            if s[j] == ")":
                crux_depth -= 1
            j += 1
        if crux_depth > 0:
            raise _MatchError("unclosed '('")
        for k, r in enumerate(reversed(reversed_)):
            relocs[k] = r
        return expr, relocs, j

    def _percent(self, s: str, j: int):
        for text, r in _PERCENT:
            if s[j:j + len(text)].lower() == text:
                nxt = s[j + len(text):j + len(text) + 1]
                if nxt not in (" ", "\t", "("):
                    continue
                if r is None or r in _UNSUPPORTED:
                    raise _MatchError(f"relocation {text} is not supported (no PIC / TLS "
                                      "code in eMIPS)")
                return r, j + len(text)
        return False, j

    # -- matching one entry (match_insn) -----------------------------------------

    def match_insn(self, mo: Opcode, tokens: list, lax: bool) -> Insn | None:
        self.imm_expr = Expr("absent")
        self.offset_expr = Expr("absent")
        self.offset_reloc = [None, None, None]
        insn = Insn(mo, mo.match)
        arg = _Arg(insn, tokens, lax)
        if "INSN2_MTTGPR_RC1" in mo.pinfo2:
            return None
        args = mo.args
        a = 0
        n = len(args)
        while True:
            if arg.token.type == T_END:
                # a unary instruction given one operand: the source is the
                # destination (an optional operand restarts the tokens)
                if a < n and (args[a] == "," or (arg.opnum == 0 and args[a])):
                    at = a + (1 if arg.opnum else 0)
                    operand = OPERANDS.get(operand_code(args, at)) if at < n else None
                    if operand is not None and operand.optional:
                        arg.pos = 0
                        arg.argnum = 1
                        a += 1
                        continue
                    if operand is not None and operand.type == "CP0SEL":
                        return insn
                    if operand is not None and operand.type == "DONT_CARE":
                        insn.opcode = operand.insert(insn.opcode, 0)
                        self._completed(arg)
                        return insn
                if args[a:] in ("(b)", "(c)"):        # an elided base register: $0
                    a += 3
                if args[a:a + 2] in ("+N", "mK"):      # an optional register suffix
                    a += 2
                if a < n:
                    return None
                self._completed(arg)
                return insn
            if a >= n:
                return None
            c = args[a]
            if c in "(),":
                if not self._match_char(arg, c):
                    return None
                a += 1
                continue
            if c == "#":
                a += 1
                if arg.token.type == T_DOUBLE and arg.token.value == args[a]:
                    arg.pos += 1
                    a += 1
                    continue
                return None
            arg.opnum += 1
            if c == "I":
                v = self._match_const_int(arg)
                if v is None:
                    return None
                self.imm_expr = Expr("constant", _normalize(v))
                a += 1
                continue
            if c == "A":
                if arg.token.type == T_CHAR and arg.token.value == "(":
                    self.offset_expr = Expr("constant", 0)
                else:
                    e = self._match_expression(arg, self.offset_reloc)
                    if e is None:
                        return None
                    self.offset_expr = e
                    if e.op in ("constant", "symbol") and _is_zext32(e.num):
                        e.num = _normalize(e.num)
                a += 1
                continue
            if c in "FLfl":
                if not self._match_float(arg, {"F": 8, "L": 8, "f": 4, "l": 4}[c],
                                         c in "Ff"):
                    return None
                a += 1
                continue
            code = operand_code(args, a)
            operand = OPERANDS[code]
            if operand.type in ("PCREL", "NON_ZERO_PCREL_S1", "HI20_PCREL", "INT_WORD",
                                "UINT_WORD", "PC_WORD", "INT", "GPREL_WORD"):
                r = {"+u": "25_PCREL_S1", "+r": "21_PCREL_S1", "+Q": "I32", "+R": "I32",
                     "a": "25_PCREL_S1", "p": "14_PCREL_S1", "~": "11_PCREL_S1"}.get(code)
                if code in ("mD", "mE", "mF"):
                    if self.forced_length:
                        r = {"mD": "10_PCREL_S1", "mE": "7_PCREL_S1", "mF": "4_PCREL_S1"}[code]
                    else:
                        r = Relax({"mD": RT_BRANCH_UCND, "mE": RT_BRANCH_CNDZ,
                                   "mF": RT_BRANCH_CND}[code])
                if r is not None:
                    self.offset_reloc[0] = r
            if c in "+m-`":
                a += 1
            if (operand.optional and args[a + 1:a + 2] == ","
                    and not _dontcare(args, a + 2)
                    and (arg.token.type != T_REG
                         or arg.tokens[arg.pos + 1].type == T_END)):
                arg.pos = 0                    # elided: the same as the first operand
                arg.argnum = 1
            if operand.type == "COPY_BITS":
                a += 1
            if not self._match_operand(arg, operand):
                return None
            if self.offset_reloc[0] == "GPREL18":
                self.offset_reloc[0] = self._gprel_for_insn(mo)
            a += 1

    def _completed(self, arg: _Arg) -> None:
        if arg.seen_at:
            if self.opts.at == ATREG:
                self.warn('used $at without ".set noat"')
            else:
                self.warn(f'used ${self.opts.at} with ".set at=${self.opts.at}"')

    def _gprel_for_insn(self, mo: Opcode):
        if self.forced_length == 2 or (self.forced_format and mo.length == 2):
            return "GPREL7_S2"
        if mo.length == 6:
            return "GPREL_I32"
        table = {"lw[gp]": "GPREL19_S2", "sw[gp]": "GPREL19_S2", "lh[gp]": "GPREL17_S1",
                 "lhu[gp]": "GPREL17_S1", "sh[gp]": "GPREL17_S1", "lb[gp]": "GPREL18",
                 "lbu[gp]": "GPREL18", "sb[gp]": "GPREL18", "addiu.b": "GPREL18",
                 "addiu[gp.b]": "GPREL18", "addiu.w": "GPREL19_S2",
                 "addiu[gp.w]": "GPREL19_S2", "ldc1[gp]": "GPREL16_S2",
                 "sdc1[gp]": "GPREL16_S2", "l.d[gp]": "GPREL16_S2", "s.d[gp]": "GPREL16_S2",
                 "swc1[gp]": "GPREL16_S2", "lwc1[gp]": "GPREL16_S2",
                 "s.s[gp]": "GPREL16_S2", "l.s[gp]": "GPREL16_S2",
                 "lwu[gp]": "GPREL16_S2", "ld[gp]": "GPREL19_S2", "sd[gp]": "GPREL19_S2"}
        return table.get(mo.name + mo.suffix)

    # -- the operand matchers ------------------------------------------------------

    def _out_of_range(self, arg: _Arg) -> None:
        self._set_error(arg.argnum, "operand %d out of range", f"operand {arg.argnum} out of range")

    def _match_char(self, arg: _Arg, ch: str) -> bool:
        t = arg.token
        if t.type == T_CHAR and t.value == ch:
            arg.pos += 1
            if ch == ",":
                arg.argnum += 1
            return True
        return False

    def _match_expression(self, arg: _Arg, relocs: list) -> Expr | None:
        t = arg.token
        if t.type == T_CHAR and t.value == "(":
            relocs[0] = relocs[1] = relocs[2] = None
            return Expr("constant", 0)
        if t.type == T_INTEGER and t.value[0].op == "register":
            self._set_error(arg.argnum, "register value used as expression")
            return None
        if t.type == T_INTEGER:
            e, r = t.value
            relocs[:] = list(r)
            arg.pos += 1
            return e.copy()
        self._set_error(arg.argnum, "operand %d must be an immediate expression",
                        f"operand {arg.argnum} must be an immediate expression")
        return None

    def _match_const_int(self, arg: _Arg) -> int | None:
        r = [None, None, None]
        e = self._match_expression(arg, r)
        if e is None:
            return None
        if r[0] is None and e.op == "constant":
            return e.num
        self._set_error(arg.argnum, "operand %d must be constant",
                        f"operand {arg.argnum} must be constant")
        return None

    def _reg_types(self, mo: Opcode, t: str) -> int:
        """convert_reg_type."""
        if t == "FP":
            return RTYPE_FPU
        if t == "ACC":
            return RTYPE_ACC
        if t == "COPRO":
            return RTYPE_NUM | RTYPE_CP0 if mo.name.endswith("0") else RTYPE_NUM
        if t == "HW":
            return RTYPE_HWR
        if t == "MSA":
            return RTYPE_MSA
        if t == "MSA_CTRL":
            return RTYPE_NUM
        if t == "CP0":
            return RTYPE_CP0
        if t == "CP0SEL":
            return RTYPE_CP0SEL | RTYPE_NUM
        if t == "HWRSEL":
            return RTYPE_HWR | RTYPE_NUM
        return RTYPE_GP

    def _match_regno(self, arg: _Arg, t: str, symval: int) -> int | None:
        if not symval & self._reg_types(arg.insn.mo, t):
            return None
        if t in ("CP0SEL", "HWRSEL"):
            arg.select_mask = _SELMASK.get(symval & RTYPE_CP0SEL, 0xFFFFFFFF)
        if t == "HW":
            for name, num, sel, cnum, csel in _HWR:
                if (symval & RNUM_MASK) == ((cnum << CP0SEL_SHIFT | csel) & 0xFFFFFFFF) \
                        and cnum != 0xFFFFFFFF:
                    symval = (symval & ~RNUM_MASK) | (num << CP0SEL_SHIFT) | sel
                    break
        elif t == "HWRSEL" and (symval & ~RNUM_MASK) != RTYPE_NUM:
            if (symval & RNUM_MASK) == 25:
                symval = (symval & ~RNUM_MASK) | 4
            else:
                return None
        regno = symval & RNUM_MASK
        if self.opts.at and t == "GP" and regno == self.opts.at:
            arg.seen_at = True
        return regno

    def _match_reg(self, arg: _Arg, t: str) -> int | None:
        tok = arg.token
        if tok.type == T_REG:
            r = self._match_regno(arg, t, tok.value)
            if r is not None:
                arg.pos += 1
                return r
        return None

    def _match_reg_range(self, arg: _Arg, t: str):
        r = self._match_reg(arg, t)
        if r is not None:
            return r, r
        tok = arg.token
        if tok.type == T_REG_RANGE:
            r1 = self._match_regno(arg, t, tok.value[0])
            r2 = self._match_regno(arg, t, tok.value[1])
            if r1 is not None and r2 is not None and r1 <= r2:
                arg.pos += 1
                return r1, r2
        return None

    def _insert(self, arg: _Arg, op: Operand, uval: int) -> None:
        arg.insn.opcode = op.insert(arg.insn.opcode, uval)

    def _match_operand(self, arg: _Arg, op: Operand) -> bool:
        t = op.type
        if t == "INT":
            return self._match_int(arg, op)
        if t == "MAPPED_INT":
            v = self._match_const_int(arg)
            if v is None:
                return False
            if v not in op.int_map:
                self._out_of_range(arg)
                return False
            self._insert(arg, op, op.int_map.index(v))
            return True
        if t == "MSB":
            size = self._match_const_int(arg)
            if size is None:
                return False
            lo, hi_ = op.bias, op.bias + (1 << op.size) - 1
            high = size + arg.last_op_int
            sval = high if op.add_lsb else size
            if size < 0 or high > op.opsize or sval < lo or sval > hi_:
                self._out_of_range(arg)
                return False
            self._insert(arg, op, sval - lo)
            return True
        if t in ("REG", "OPTIONAL_REG"):
            return self._match_reg_operand(arg, op)
        if t == "REG_PAIR":
            r1 = self._match_reg(arg, op.reg_type)
            if r1 is None or not self._match_char(arg, ","):
                return False
            r2 = self._match_reg(arg, op.reg_type)
            if r2 is None:
                return False
            for u in range(1 << op.size):
                if op.reg1_map[u] == r1 and op.reg2_map[u] == r2:
                    self._insert(arg, op, u)
                    return True
            return False
        if t == "PCREL":
            r = [None, None, None]
            e = self._match_expression(arg, r)
            if e is None or r[0] is not None:
                return False
            self.offset_expr = e
            return True
        if t == "SAVE_RESTORE_LIST":
            return self._match_save_restore(arg, op)
        if t == "SAVE_RESTORE_FP_LIST":
            rr = self._match_reg_range(arg, "FP")
            if rr is None or rr[0] != 0 or rr[1] > 15:
                return False
            self._insert(arg, op, rr[1] - rr[0])
            return True
        if t == "REPEAT_DEST_REG":
            r = self._match_reg(arg, "GP")
            return r is not None and r == arg.dest_regno
        if t == "REPEAT_PREV_REG":
            r = self._match_reg(arg, "GP")
            return r is not None and r == arg.last_regno
        if t == "CHECK_PREV":
            r = self._match_reg(arg, "GP")
            if r is None or (not op.zero_ok and r == 0):
                return False
            if ((op.less_than_ok and r < arg.last_regno)
                    or (op.greater_than_ok and r > arg.last_regno)
                    or (op.equal_ok and r == arg.last_regno)):
                arg.last_regno = r
                self._insert(arg, op, r)
                return True
            return False
        if t == "NON_ZERO_REG":
            r = self._match_reg(arg, "GP")
            if r is None:
                return False
            if r == 0:
                pinfo = arg.insn.mo.pinfo
                kind = ("target" if (arg.opnum == 1 and "WR_1" in pinfo)
                        or (arg.opnum == 2 and "WR_2" in pinfo) else "source")
                self._set_error(arg.argnum, "the %s register must not be $0%s",
                                f"the {kind} register must not be $0")
                return False
            arg.last_regno = r
            self._insert(arg, op, r)
            return True
        if t == "NON_ZERO_PCREL_S1":
            r = [None, None, None]
            e = self._match_expression(arg, r)
            if e is None or r[0] is not None:
                return False
            self.offset_expr = e
            if e.sym is None and e.op == "constant" and e.num == 0:
                self._set_error(arg.argnum, "the PC relative offset must not be $0")
                return False
            self._insert(arg, op, (1 << op.size) - 1)
            return True
        if t == "HI20_PCREL":
            e = self._match_expression(arg, self.offset_reloc)
            if e is None:
                return False
            self.offset_expr = e
            r = self.offset_reloc[0]
            if r is None or (r in _HI and e.op == "constant"):
                uval = e.num & M32
                if r in _HI:
                    uval >>= 12
                self._insert_hi20(arg, uval)
                self.offset_expr = Expr("absent")
                return True
            return r in ("PCREL_HI20", "GOTPC_HI20")
        if t == "HI20_INT":
            e = self._match_expression(arg, self.offset_reloc)
            if e is None:
                return False
            self.offset_expr = e
            r = self.offset_reloc[0]
            if r is None and e.op == "constant":
                if e.num & ~0xFFFF == 0:
                    uval = e.num << 4
                else:
                    self._out_of_range(arg)
                    return False
            elif r in _HI:
                uval = (e.num >> 12) & M32
            else:
                if arg.lax:
                    self.offset_reloc[0] = "HI20"
                return arg.lax
            if e.op == "constant":
                self.offset_expr = Expr("absent")
                self._insert_hi20(arg, uval)
            return True
        if t == "HI20_SCALE":
            v = self._match_const_int(arg)
            if v is None or v & 0xFFF:
                return False
            self.offset_expr = Expr("absent")
            self._insert_hi20(arg, (v >> 12) & M32)
            return True
        if t == "MAPPED_CHECK_PREV":
            last_u = _encode_reg(op, arg.last_regno)
            if not self._match_reg_operand(arg, op):
                return False
            if not op.zero_ok and arg.last_regno == 0:
                return False
            u = _encode_reg(op, arg.last_regno)
            return ((op.less_than_ok and u < last_u) or (op.greater_than_ok and u > last_u)
                    or (op.equal_ok and u == last_u))
        if t in ("UINT_WORD", "INT_WORD"):
            e = self._match_expression(arg, self.offset_reloc)
            if e is None:
                return False
            self.offset_expr = e
            r = self.offset_reloc[0]
            if r is not None and r not in _R48 and r not in _FLEX:
                return False
            if r is None:
                self.offset_reloc[0] = "I32"
            if e.op != "constant" and e.sub is None:
                arg.insn.ext = 0
                return True
            if ((self.forced_length == 6 or self.forced_format or e.num > 0xFFFF
                 or e.num < -4095) and e.op != "big"):
                arg.last_op_int = e.num
                arg.insn.ext = 0
                return True
            return False
        if t == "PC_WORD":
            e = self._match_expression(arg, self.offset_reloc)
            if e is None:
                return False
            self.offset_expr = e
            r = self.offset_reloc[0]
            if r is not None and r not in _R48:
                return False
            if r is None:
                self.offset_reloc[0] = "PC_I32"
            arg.insn.ext = 0
            return True
        if t == "GPREL_WORD":
            e = self._match_expression(arg, self.offset_reloc)
            if e is None:
                return False
            self.offset_expr = e
            r = self.offset_reloc[0]
            if r is not None and r not in _GPREL_I32 and r not in _GPREL:
                return False
            if r is None:
                self.offset_reloc[0] = "GPREL_I32"
            arg.insn.ext = 0
            return True
        if t == "DONT_CARE":
            return False
        if t == "NEG_INT":
            v = self._match_const_int(arg)
            if v is None:
                return False
            sv = _s32(v)
            if sv > 0 or sv < -4095:
                return False
            self._insert(arg, op, -sv)
            return True
        if t == "IMM_INT":
            if not self._match_int(arg, op):
                return False
            return (self.offset_expr.op in ("absent", "constant")
                    and self.offset_reloc[0] is None)
        if t == "IMM_WORD":
            e = self._match_expression(arg, self.offset_reloc)
            if e is None:
                return False
            self.offset_expr = e
            if e.op != "constant" or self.offset_reloc[0] is not None:
                return False
            sval = _s32(e.num - op.bias)
            arg.last_op_int = sval
            arg.insn.ext = (((sval >> 16) & 0xFFFF) | (sval << 16)) & M32
            self.offset_expr = Expr("absent")
            return True
        if t == "BASE_CHECK_OFFSET":
            r = self._match_reg(arg, "GP")
            if r is None:
                return False
            oe = self.offset_expr.op
            if ((op.const_ok and oe in ("constant", "absent"))
                    or (op.expr_ok and oe in ("symbol", "absent"))
                    or r == 0 or self.offset_reloc[0] in _LO):
                arg.last_regno = r
                self._insert(arg, op, r)
                return True
            return False
        if t == "COPY_BITS":
            src = Operand("", "INT", op.size, op.lsb)
            dst = Operand("", "INT", op.size_top, op.lsb_top)
            arg.insn.opcode = dst.insert(arg.insn.opcode, src.extract(arg.insn.opcode))
            return True
        if t == "CP0SEL":
            v = self._match_const_int(arg)
            if v is None or v < 0 or v >= 32 or not ((1 << v) & arg.select_mask):
                return False
            self._insert(arg, op, v)
            return True
        raise AssertionError(f"operand type {t}")

    def _insert_hi20(self, arg: _Arg, uval: int) -> None:
        shuffle = Operand("", "INT", 19, 10, 10, 0)
        enc = Operand("", "INT", 20, 2, 1, 0)
        uval = shuffle.insert(uval & 0x80000, uval & 0x7FFFF)
        arg.insn.opcode = enc.insert(arg.insn.opcode, uval)

    def _match_reg_operand(self, arg: _Arg, op: Operand) -> bool:
        r = self._match_reg(arg, op.reg_type or "GP")
        if r is None:
            return False
        if op.reg_map:
            if r not in op.reg_map:
                return False
            u = op.reg_map.index(r)
        else:
            u = r
        if op.size > 0 and u >= 1 << op.size:
            self._out_of_range(arg)
            return False
        arg.last_regno = r
        if arg.opnum == 1:
            arg.dest_regno = r
        self._insert(arg, op, u)
        return True

    def _match_int(self, arg: _Arg, op: Operand) -> bool:
        """match_int_operand."""
        factor = 1 << op.shift
        min_val, max_val = _s32(op.int_min()), _s32(op.int_max())
        if _relocatable_int(op):
            e = self._match_expression(arg, self.offset_reloc)
            if e is None:
                return False
            self.offset_expr = e
            r = self.offset_reloc[0]
            if r is not None and r not in _R48:
                if r in _FLEX:
                    return self._flex_ok(op, r)
                if r == "LO12" and (self.forced_length == 2
                                    or (self.forced_format and arg.insn.length == 2)):
                    r = self.offset_reloc[0] = "LO4_S2"
                h = _howto(r)
                self._out_of_range(arg)
                return (h["dst_mask"] & op.mask) == h["dst_mask"]
            if e.op != "constant":
                if op.size == 16 and e.op == "subtract":
                    self.offset_reloc[0] = Relax(RT_ADDIU)
                    return True
                if e.op == "symbol":
                    self._out_of_range(arg)
                    return False
                self.offset_reloc[0] = "LO12"
                return arg.lax
            sval = _s32(e.num) if min_val < 0 else e.num
            self.offset_expr = Expr("absent")
            if sval > max_val:
                max_val = ((1 << op.size) - 1) << op.shift
                if not arg.lax and sval <= max_val:
                    return False
        else:
            sval = self._match_const_int(arg)
            if sval is None:
                return False
            if min_val < 0 and _is_zext32(sval):
                sval = _s32(sval)
        arg.last_op_int = sval
        if sval < min_val or sval > max_val or sval % factor:
            self._out_of_range(arg)
            return False
        uval = ((sval & M32) >> op.shift) - op.bias
        self._insert(arg, op, uval & M32)
        return True

    def _flex_ok(self, op: Operand, r: str) -> bool:
        """match_flex_reloc_for_int_operand."""
        rl = {"GPREL18": ("GPREL18", "GPREL19_S2", "GPREL18_S3", "GPREL16_S2",
                          "GPREL17_S1", "GPREL7_S2")}.get(r, (r,))
        for x in rl:
            h = _howto(x)
            if (h["dst_mask"] & op.mask) == h["dst_mask"] and h["bitsize"] == op.size:
                return True
        return False

    def _match_save_restore(self, arg: _Arg, op: Operand) -> bool:
        mode16 = arg.insn.length == 2
        gp = False
        count = 0
        first = last = RNUM_MASK + 1
        while True:
            rr = self._match_reg_range(arg, "GP")
            if rr is None:
                return False
            r1, r2 = rr
            first_range = False
            if r1 <= 29 <= r2:
                self._set_error(r1, "%s register list should not contain $sp%s",
                                f"{arg.insn.mo.name} register list should not contain $sp")
                return False
            if first == RNUM_MASK + 1:
                first = last = r1
                first_range = True
            if r1 != ((first & 0x10) | (last + 1) % 32) or first_range:
                if r1 == 28 and not mode16 and self.opts.ase & ASE["xNMS"]:
                    gp = True
                elif r1 == 28 and not self.opts.ase & ASE["xNMS"]:
                    self._set_error(count, "non-contiguous %s of $gp is not available for "
                                    "nanoMIPS subset%s",
                                    f"non-contiguous {arg.insn.mo.name} of $gp is not "
                                    "available for nanoMIPS subset")
                    return False
                elif r1 == 28 and mode16:
                    self._set_error(count, "non-contiguous %s of $gp is not available for "
                                    "16-bit instructions%s",
                                    f"non-contiguous {arg.insn.mo.name} of $gp is not "
                                    "available for 16-bit instructions")
                    return False
                elif not first_range:
                    if r1 == 30 and first != 30:
                        self._set_error(count, "$fp must be the first register in the list")
                    elif r1 == 31 and first not in (30, 31):
                        self._set_error(count, "$ra must be the first register in the list")
                    elif last == 31 and r1 != 16:
                        self._set_error(count + 1, "%s must be followed by %s",
                                        f"${GPR_NAMES[last]} must be followed by $s0")
                    else:
                        self._set_error(r1, "register list not contiguous between %s and %s",
                                        "register list not contiguous between "
                                        f"${GPR_NAMES[last]} and ${GPR_NAMES[r1]}")
                    return False
            last = r2
            count += r2 - r1 + 1
            if not self._match_char(arg, ","):
                break
        if count >= 16:
            self._set_error(count, "%s can handle a maximum of 16 register, %d requested",
                            f"{arg.insn.mo.name} can handle a maximum of 16 register, "
                            f"{count} requested")
            return False
        if mode16 and first < 30:
            self._set_error(count + 1, "first register for 16-bit %s must be $fp or $ra%s",
                            f"first register for 16-bit {arg.insn.mo.name} must be "
                            "$fp or $ra")
            return False
        if count * 4 > arg.last_op_int:
            self.warn("frame too small for save/restore register list")
        if mode16:
            opval = ((first & 1) << 4) | count
        else:
            opval = (first << 6) | (count << 1) | int(gp)
        self._insert(arg, op, opval)
        return True

    def _match_float(self, arg: _Arg, length: int, using_gprs: bool) -> bool:
        t = arg.token
        if t.type != T_FLOAT:
            self._set_error(arg.argnum, "floating-point expression required")
            return False
        data = t.value
        arg.pos += 1
        if length == 4:
            self.imm_expr = Expr("constant", int.from_bytes(data, "big" if self.big
                                                            else "little"))
            self.offset_expr = Expr("absent")
            return True
        hi = int.from_bytes(data[:4] if self.big else data[4:], "big" if self.big else "little")
        lo = int.from_bytes(data[4:] if self.big else data[:4], "big" if self.big else "little")
        self.imm_expr = Expr("constant", hi)
        self.offset_expr = Expr("constant", lo) if lo else Expr("absent")
        return True


def _s32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v >> 31 else v


def _relocatable_int(op: Operand) -> bool:
    """match_relocatable_int_operand."""
    return ((op.size == 7 and op.shift == 2) or (op.size == 4 and op.shift == 2)
            or op.size in (12, 14, 16, 17, 18, 19, 20))


def _encode_reg(op: Operand, regno: int) -> int:
    """nanomips_encode_reg_operand: the field value (1 << size if none)."""
    for u in range(1 << op.size):
        if op.reg_map[u] == regno:
            return u
    return 1 << op.size


def _dontcare(args: str, i: int) -> bool:
    return args[i:i + 1] == "-" and "A" <= args[i + 1:i + 2] <= "Q"


class _Emit:
    """append_insn and the frags it fills (mixed into the Assembler)."""

    # -- emitting an instruction (append_insn) --------------------------------------

    def _bytes(self, value: int, length: int) -> bytes:
        """write_compressed_insn: halfwords, most significant first, each
        in the target's byte order."""
        out = bytearray()
        for i in range(0, length, 2):
            h = (value >> ((length - i - 2) * 8)) & 0xFFFF
            out += h.to_bytes(2, "big" if self.big else "little")
        return bytes(out)

    def _insn_bytes(self, insn: Insn) -> bytes:
        if insn.length == 6:
            return self._bytes(insn.opcode, 2) + self._bytes(insn.ext, 4)
        return self._bytes(insn.opcode, insn.length)

    def _note_line(self, frag: Frag, where: int, size: int) -> None:
        frag.lines.append((where, size, self.macro_lines, self.relax_seq))

    def toggle_point(self, frag: Frag, where: int) -> None:
        """toggle_linkrelax: a pending ``.set [no]linkrelax`` takes effect
        here, with an R_NANOMIPS_RELAX / R_NANOMIPS_NORELAX."""
        if self.toggle_linkrelax:
            self.opts.linkrelax = not self.opts.linkrelax
            self.toggle_linkrelax = False
            self.fixups.append(Fix(frag, where, 0, Expr("absent"),
                                   "RELAX" if self.opts.linkrelax else "NORELAX", self.line))

    def add_fixed_insn(self, insn: Insn) -> tuple[Frag, int]:
        where = self.frag_more(self._insn_bytes(insn))
        self.toggle_point(self.frag, where)            # move_insn
        self._note_line(self.frag, where, insn.length)
        return self.frag, where

    def add_relaxed_insn(self, insn: Insn, max_chars: int, var: int, subtype: int,
                         symbol: Sym | None, offset: int) -> tuple[Frag, int]:
        f = self.frag
        where = len(f.fix)
        self.toggle_point(f, where)                    # move_insn
        f.literal = bytearray(self._insn_bytes(insn))
        f.kind = "md"
        f.max, f.var, f.subtype = max_chars, var, subtype
        f.symbol, f.offset = symbol, offset
        f.line = self.line
        self._note_line(f, where, var)
        self.frag_close()
        return f, where

    def append_insn(self, insn: Insn, expr: Expr | None, relocs: list,
                    expansion: bool = False) -> None:
        mo = insn.mo
        r0 = relocs[0]
        if expr is None:
            insn.complete = True
        elif (not isinstance(r0, Relax) and relocs[1] is None and relocs[2] is None
              and expr.op == "constant"):
            v = expr.num
            if r0 in ("HI20", "GPREL_HI20"):
                insn.opcode |= ((((v >> 12) & 0x1FF) << 12) | (((v >> 21) & 0x3FF) << 2)
                                | ((v >> 31) & 1))
                insn.complete = True
            elif r0 in ("LO12", "GOT_LO12"):
                insn.opcode |= v & 0xFFF
                insn.complete = True
            elif r0 == "IMM16":
                insn.opcode |= v & 0xFFFF
                insn.complete = True
            elif r0 == "NEG12":
                insn.opcode |= (-v) & 0xFFF
                insn.complete = True
            elif r0 in ("I32", "GPREL_I32"):
                insn.ext = (((v >> 16) & 0xFFFF) | (v << 16)) & M32
                insn.complete = True
            elif r0 == "GOT_CALL":
                insn.complete = True
            elif r0 == "SIGNED_9":
                insn.opcode |= (v & 0xFF) | ((v & 0x100) << 7)
                insn.complete = True
            insn.opcode &= M32
        h = self.history
        if (h is not None and "CTC" in h.mo.pinfo2 and self.opts.noreorder
                and mo.name != "nop" and _compact_branch(h)):
            self.error(f"unable to convert `{h.mo.name}' to its compact form because it has "
                       f"a non NOP instruction (`{mo.name}') in its delay slot.  Please move "
                       "the delay slot instruction before the branch and disable noreorder.")
        if (h is not None and self.opts.noreorder
                and h.mo.name in ("bal", "jal", "bgezal", "bltzal") and mo.name == "nop"):
            self.error(f"unable to convert `{h.mo.name}' to its compact form because the "
                       "link register would have a different value.  Please move the delay "
                       "slot instruction before the branch and disable noreorder.")
        fixfrag = fixwhere = None
        if expr is not None and isinstance(r0, Relax) and r0.rt >= RT_BRANCH_UCND:
            al = "WR_31" in mo.pinfo
            mx = mo.length if self.forced_format else 4
            rf = RF_VARIABLE if self.opts.linkrelax else RF_NORELAX
            if not self.forced_format:
                fixfrag, fixwhere = self.add_relaxed_insn(
                    insn, mx, mo.length, _md_encode(r0.rt, al, rf), expr.sym, expr.num)
                relocs = [None, None, None]
            else:
                fixfrag, fixwhere = self.add_fixed_insn(insn)
                relocs = [_rt_reloc(r0.rt, False), None, None]
        elif (expr is not None and isinstance(r0, Relax) and r0.rt == RT_ADDIU
              and expr.op == "subtract" and not self.forced_format):
            fixed = RF_FIXED if (self.opts.insn32 or self.forced_length == 4) else RF_NONE
            sym = Sym("<expr>")
            sym.expr = expr.copy()            # make_expr_symbol
            fixfrag, fixwhere = self.add_relaxed_insn(
                insn, 6, 4, _md_encode(RT_ADDIU, False, fixed), sym, 0)
            relocs = [None, None, None]
        else:
            if self.relax_seq:
                self.relax_sizes[self.relax_seq - 1] += mo.length
            if self.relax_seq != 2:
                self.macro_insns[0] += 1
            if self.relax_seq != 1:
                self.macro_insns[1] += 1
            if (isinstance(r0, str) and _pcrel_branch(r0) and expr is not None
                    and expr.op != "constant"):
                fix_type = (RF_NORELAX if (not self.opts.linkrelax or self.forced_length
                                            or self.forced_format) else RF_VARIABLE)
                fixfrag, fixwhere = self.add_relaxed_insn(
                    insn, mo.length, mo.length,
                    _md_encode(_RT_OF.get(r0, 0), "WR_31" in mo.pinfo, fix_type),
                    expr.sym, expr.num)
            else:
                fixfrag, fixwhere = self.add_fixed_insn(insn)
        if not insn.complete and isinstance(relocs[0], str):
            where = fixwhere + (2 if relocs[0] in _R48 else 0)
            fx = Fix(fixfrag, where, 4, expr.copy() if expr is not None else Expr("absent"),
                     relocs[0], self.line, seq=self.relax_seq)
            fx.insn_size = mo.length
            self.fixups.append(fx)
            if fixfrag.first_fix is None:
                fixfrag.first_fix = fx
            if self.relax_seq and self.relax_first_fix is None:
                self.relax_first_fix = fx
            for r in relocs[1:]:
                if r is not None:
                    raise _MatchError("composite relocations are not supported")
            if self.opts.linkrelax:
                if self.forced_format:
                    self.fixups.append(Fix(fixfrag, fixwhere + (2 if mo.length == 6 else 0), 0,
                                           Expr("absent"), "FIXED", self.line,
                                           seq=self.relax_seq))
                elif self.forced_length in (2, 4):
                    self.fixups.append(Fix(fixfrag, fixwhere, 0, Expr("absent"),
                                           "INSN16" if self.forced_length == 2 else "INSN32",
                                           self.line, seq=self.relax_seq))
        if not self.forced_length and not self.forced_format and not insn.complete:
            self.record_insn_sops(fixfrag, relocs[0])
        self.history = insn
        if _compact_branch(insn) and "UBR" in mo.pinfo2 and "CTC" not in mo.pinfo2:
            self.flush_pending_output()
        self.labels = []

    def flush_pending_output(self) -> None:
        self.history = None
        self.labels = []

    # -- link-time variability (record_insn_sops, get_reloc_sop / sink) ---------------

    def get_reloc_sop(self, r: str | None) -> int:
        """The bytes a link-time transformation may add to an instruction
        with relocation ``r`` (None: BFD_RELOC_UNUSED, the fall-back)."""
        nmf, nmf32, nms, _ = _RELOC_SOPS.get(r or "", _RELOC_SOPS["NONE"])
        if self.opts.ase & ASE["xNMS"]:
            return nmf32 if self.opts.insn32 else nmf
        return nms

    @staticmethod
    def get_reloc_sink(r: str | None) -> int:
        return _RELOC_SOPS.get(r or "", _RELOC_SOPS["NONE"])[3]

    def subtype_sop(self, subtype: int) -> int:
        """frag_subtype_to_relax_sop / _sink (of the reloc the subtype
        stands for)."""
        return self.get_reloc_sop(_rt_reloc(_md_type(subtype), bool(subtype & TOOFAR16)))

    def subtype_sink(self, subtype: int) -> int:
        return self.get_reloc_sink(_rt_reloc(_md_type(subtype), bool(subtype & TOOFAR16)))

    def record_insn_sops(self, frag: Frag, r0) -> None:
        """record_insn_sops: the frag holds an instruction the linker may
        change (an unconsumed relaxation stands for the frag's subtype)."""
        if isinstance(r0, Relax):
            rsop, rsink = (self.subtype_sop(_md_type(frag.subtype)),
                           self.subtype_sink(_md_type(frag.subtype)))
        else:
            rsop, rsink = self.get_reloc_sop(r0), self.get_reloc_sink(r0)
        if rsop + rsink > 0:
            frag.link_var = True
            frag.relax_sop += rsop
            frag.relax_sink += rsink

    # -- relaxable macro sequences (relax_start / relax_switch / relax_end) -----------

    def relax_start(self, sym: Sym) -> None:
        self.relax_seq = 1
        self.relax_symbol = sym
        self.relax_sizes = [0, 0]
        self.relax_first_fix = None

    def relax_switch(self) -> None:
        self.relax_seq = 2

    def relax_end(self) -> None:
        f = self.frag
        f.kind = "md"
        f.var = f.max = 0
        f.subtype = 0
        f.first, f.second = self.relax_sizes
        f.symbol, f.offset = self.relax_symbol, 0
        f.line = self.line
        self.frag_close()
        self.relax_seq = 0
        self.relax_sizes = [0, 0]


RF_NONE, RF_FIXED, RF_VARIABLE, RF_NORELAX = range(4)

#: tc-nanomips.c reloc_sops: the bytes a link-time transformation of an
#: instruction with this relocation may add (full ISA, full ISA with
#: insn32, the subset) and remove; "NONE" is the fall-back maximum
_RELOC_SOPS = {
    "4_PCREL_S1": (8, 12, 12, 0), "7_PCREL_S1": (8, 12, 10, 2),
    "10_PCREL_S1": (6, 10, 8, 2), "11_PCREL_S1": (8, 12, 10, 4),
    "21_PCREL_S1": (6, 12, 4, 4), "25_PCREL_S1": (4, 8, 6, 4),
    "14_PCREL_S1": (8, 12, 10, 4), "GPREL19_S2": (6, 8, 8, 2),
    "GPREL18_S3": (6, 8, 8, 0), "GPREL16_S2": (6, 8, 8, 0), "GPREL18": (6, 8, 8, 0),
    "GPREL17_S1": (6, 8, 8, 0), "GPREL7_S2": (8, 10, 10, 0), "GOT_CALL": (6, 8, 8, 4),
    "GOT_PAGE": (2, 4, 4, 4), "GOT_OFST": (6, 8, 8, 0), "GOT_DISP": (6, 8, 8, 0),
    "HI20": (0, 0, 0, 2), "LO12": (0, 0, 0, 4), "PCREL_HI20": (0, 0, 0, 2),
    "GOTPC_HI20": (0, 0, 0, 4), "GOT_LO12": (0, 0, 0, 4), "I32": (0, 0, 0, 4),
    "TLS_GD": (2, 8, 0, 0), "TLS_GD_I32": (0, 0, 0, 2), "TLS_LD": (2, 8, 0, 0),
    "TLS_LD_I32": (0, 0, 0, 2), "TLS_DTPREL12": (6, 8, 0, 0), "TLS_DTPREL16": (6, 8, 0, 4),
    "TLS_DTPREL_I32": (0, 0, 0, 6), "TLS_TPREL12": (6, 8, 0, 0), "TLS_TPREL16": (6, 8, 0, 4),
    "TLS_TPREL_I32": (0, 0, 0, 6), "TLS_GOTTPREL": (2, 4, 0, 4),
    "TLS_GOTTPREL_PC_I32": (0, 0, 0, 6), "PC_I32": (0, 0, 0, 4), "GOTPC_I32": (0, 0, 0, 2),
    "GPREL_I32": (0, 0, 0, 4), "GPREL_HI20": (0, 0, 0, 2), "GPREL_LO12": (0, 0, 0, 4),
    "JALR16": (2, 2, 2, 2), "JALR32": (0, 0, 0, 4), "INSN16": (0, 0, 0, 0),
    "INSN32": (0, 0, 0, 0), "FIXED": (0, 0, 0, 0), "ALIGN": (0, 0, 0, 0),
    "RELAX": (0, 0, 0, 0), "NORELAX": (0, 0, 0, 0), "SAVERESTORE": (0, 0, 0, 4),
    "JUMPTABLE_LOAD": (0, 0, 0, 0), "NOTRAMP": (0, 0, 0, 0), "NONE": (32, 32, 32, 4)}


def _md_encode(rt: int, link: bool, rf: int) -> int:
    """RELAX_MD_ENCODE."""
    return 0x20000000 | (rt & 0xFF) | (0x100 if link else 0) | (rf << 15 if rf else 0)


def _md_type(s: int) -> int:
    return s & 0xFF


TOOFAR16, TOOFAR32, NEGOFF, TOOFAR_FINAL = 0x1000, 0x2000, 0x4000, 0x20000


def _md_rf(s: int) -> int:
    return (s & 0x18000) >> 15


def _compact_branch(insn: Insn) -> bool:
    """compact_branch_p: a branch or jump (nanoMIPS: every one is compact)."""
    return "UBR" in insn.mo.pinfo2 or "CBR" in insn.mo.pinfo2


def _format_fixed(search: str, args: str) -> bool:
    """macro_match_format_fixed."""
    if search == args:
        return True
    if args.startswith(search) and len(args) - len(search) >= 3:
        return _dontcare(args, len(search) + 1)
    return False


#: macro_match_reloc: the generic relocation of a macro operand by its format
def _macro_reloc(fmt: str, i: int, r):
    if r == "LO16":
        c = fmt[i]
        if c in "og":
            r = "LO12"
        elif c == "j":
            r = "IMM16"
        elif c == "h":
            r = "NEG12"
    elif r == "GPREL16":
        c2 = fmt[i:i + 2]
        r = {".": "GPREL19_S2"}.get(fmt[i]) or {"+2": "GPREL16_S2", "+1": "GPREL18",
                                                  "+3": "GPREL17_S1",
                                                  "+4": "GPREL18_S3"}[c2]
    return {"GPREL16": "GPREL18", "HI16": "HI20", "HI16_S": "HI20", "LO16": "LO12"}.get(r, r)


class _Macros:
    """tc-nanomips.c's macro layer, mixed into the Assembler."""

    def macro_build(self, ep: Expr | None, name: str, fmt: str, *args) -> None:
        """macro_build: the first entry ``name`` whose operands are ``fmt``."""
        i = _FIRST[name]
        mo = None
        while i < len(OPCODES) and OPCODES[i].name == name:
            o = OPCODES[i]
            if (_format_fixed(fmt, o.args) and not o.is_macro and self.opcode_valid(o)
                    and self.size_valid(o)):
                mo = o
                break
            i += 1
        if mo is None:
            raise _NoFormat(f"macro_build: no {name} {fmt}")
        if self.built is not None:
            self.built.append((name + mo.suffix, self.relax_seq))
        insn = Insn(mo, mo.match)
        r = [None, None, None]
        it = iter(args)
        k = 0
        while k < len(fmt):
            c = fmt[k]
            if c in ",()z":
                k += 1
                continue
            if c in "igjh~o.":
                r = self._read_relocs(it)
                r[0] = _macro_reloc(fmt, k, r[0])
                k += 1
                continue
            if c == "u":
                r = self._read_relocs(it)
                r[0] = _macro_reloc(fmt, k, r[0])
                k += 1
                continue
            if c == "+":
                c2 = fmt[k + 1]
                if c2 in "mj1234KQRS" and ep is not None:
                    r = self._read_relocs(it)
                    r[0] = _macro_reloc(fmt, k, r[0])
                    if c2 not in "QRS":
                        k += 2
                        continue
                elif c2 in "ru":
                    r[0] = "25_PCREL_S1" if c2 == "u" else "21_PCREL_S1"
                    k += 2
                    continue
            elif c == "p":
                r[0] = _macro_reloc(fmt, k, "14_PCREL_S1")
                k += 1
                continue
            elif c == "m" and fmt[k + 1] == "Q":
                r = self._read_relocs(it)
                r[0] = _macro_reloc(fmt, k, r[0])
                k += 2
                continue
            elif c == "-" and fmt[k + 1] == "i":
                r = self._read_relocs(it)
            op = OPERANDS[operand_code(fmt, k)]
            if op.type in ("INT_WORD", "UINT_WORD", "PC_WORD", "GPREL_WORD", "IMM_WORD"):
                next(it, None)                  # (va_arg past the arguments: ignored)
                insn.ext = 0
                k += 2 if c in "+m-`" else 1
                continue
            uval = next(it)
            if op.type == "NEG_INT":
                insn.opcode = op.insert(insn.opcode, -uval)
            else:
                insn.opcode = op.insert(insn.opcode, uval)
            k += 2 if c in "+m-`" else 1
        self.append_insn(insn, ep, r, True)

    def _read_relocs(self, it) -> list:
        """macro_read_relocs: a single code, or -1 then three."""
        v = next(it)
        if v == -1:
            r = [next(it), next(it), next(it)]
        else:
            r = [v, None, None]
        if r[0] is None:
            r[0] = "LO16"
        return r

    def load_register(self, reg: int, ep: Expr, dbl: bool = False) -> None:
        """load_register: the fewest instructions for a constant."""
        if ep.op == "big":
            self.error(f"number (0x{ep.num:x}) larger than 32 bits")
        ep = ep.copy()
        if not dbl:
            ep.num = _normalize(ep.num)
        v = ep.num
        if _s32(v) >= -1 and v <= 126 and (reg >> 2 == 1 or reg >> 2 == 4):
            self.macro_build(None, "li", "md,mI", reg, _s32(v))
            return
        if _sext16_uint(v):
            self.macro_build(ep, "addiu", ADDIU_FMT, reg, 0, "IMM16")
            return
        if _offset_high_unsigned(-v, ISA_OFFBITS) == 0:
            self.macro_build(ep, "addiu", "t,r,h", reg, 0, "NEG12")
            return
        if _is_sext32(v):
            r0 = self.offset_reloc[0]
            if (self.opts.ase & ASE["xNMS"] and v & 0xFFF
                    and (r0 is None or (isinstance(r0, str) and _pcrel(r0)))
                    and not self.opts.insn32):
                self.macro_build(ep, "li", "mp,+Q", reg, "I32")
            else:
                self.macro_build(ep, "lui", LUI_FMT, reg, "HI20")
                if v & 0xFFF and r0 not in _HI:
                    self.macro_build(ep, "ori", BITOP_IMM_FMT, reg, reg, "LO12")
            return
        self.error(f"number (0x{v & ((1 << 64) - 1):x}) larger than 32 bits")

    def move_register(self, dest: int, source: int) -> None:
        if not self.opts.insn32:
            self.macro_build(None, "move", MOVE_FMT, dest, source)
        else:
            self.macro_build(None, "or", "d,v,t", dest, source, 0)

    def branch_rs(self, kind: str, ep: Expr, sreg: int) -> None:
        """macro_build_branch_rs."""
        self.macro_build(ep, kind, BCONDZ2_FMT if kind in ("bgtz", "blez") else BCONDZ1_FMT,
                         sreg)

    def branch_rsrt(self, kind: str, ep: Expr, sreg: int, treg: int) -> None:
        """macro_build_branch_rsrt (kind beq / bne)."""
        if treg == 0:
            self.macro_build(ep, kind, "s,t,p", treg, sreg)
        else:
            self.macro_build(ep, kind, "s,t,p", sreg, treg)

    def branch_rtim(self, kind: str, ep: Expr, sreg: int, imm: Expr) -> None:
        """macro_build_branch_rtim."""
        self.macro_build(ep, kind, "t,+9,~", sreg, imm.num, "11_PCREL_S1")

    # the offset checks (small_offset_p ...)

    def small_offset_p(self, rng: int, align: int, offbits: int) -> bool:
        r0 = self.offset_reloc[0]
        if offbits == 16:
            if rng < align and r0 is not None:
                return True
            if r0 in _GPREL:
                return True
        e = self.offset_expr
        return (e.op == "constant" and _offset_high_part(e.num, offbits) == 0
                and _offset_high_part(e.num + rng, offbits) == 0
                and e.num % align == 0 and rng % align == 0)

    def small_poffset_p(self, rng: int, align: int, offbits: int) -> bool:
        e = self.offset_expr
        if (e.op == "constant" and e.num >= 0 and e.num + rng < (1 << offbits)
                and e.num % align == 0 and rng % align == 0):
            return True
        return self.offset_reloc[0] in _LO

    def small_noffset_p(self, rng: int, offbits: int) -> bool:
        e = self.offset_expr
        return e.op == "constant" and e.num < 0 and -e.num + rng < (1 << offbits)

    def small_add_offset_p(self, rng: int) -> bool:
        return (self.small_poffset_p(rng, 1, ISA_ADD_OFFBITS)
                or self.small_noffset_p(rng, ISA_OFFBITS))

    def expression_valid(self, e: Expr, rtype) -> bool:
        """macro_expression_valid."""
        if e.op not in ("constant", "symbol") or (e.op == "symbol" and rtype in _HI):
            self.error("expression too complex")
        if e.op == "constant" and rtype in _HI:
            e.num = _offset_high_part(e.num, ISA_OFFBITS)
        if not _is_sext32(e.num):
            self.error(f"number (0x{e.num & ((1 << 64) - 1):x}) larger than 32 bits")
        return True

    def need_relax(self, sym: Sym | None, before: bool) -> bool:
        """nopic_need_relax: True = the symbol cannot be reached off $gp."""
        if sym is None:
            return False
        if self.g_switch <= 0:
            return True
        s, _ = sym.resolve()
        if s.name in ("eprol", "etext", "_gp", "edata", "_fbss", "_fdata", "_ftext", "end",
                      "_gp_disp"):
            return True
        if not s.defined and ((before and s.value() == 0)
                              or (s.value() != 0 and s.value() <= self.g_switch)):
            return False
        seg = s.segment()
        name = seg.name if isinstance(seg, Section) else ("*ABS*" if seg == "abs" else "*UND*")
        return not _small_data_section(name)

    def pcrel_la(self, dest: int, e: Expr, r) -> None:
        if self.opts.ase & ASE["xNMS"] and r is None and not self.opts.insn32:
            self.macro_build(e, "lapc", "mp,+S", dest, "PC_I32")
        else:
            self.macro_build(e, "aluipc", "t,+K", dest, "PCREL_HI20")
            self.macro_build(e, "addiu", ADDIU_FMT, dest, dest, "LO12")

    def absolute_la(self, dest: int) -> None:
        if self.opts.ase & ASE["xNMS"] and self.offset_reloc[0] is None and not self.opts.insn32:
            self.macro_build(self.offset_expr, "li", "mp,+Q", dest, "I32")
        else:
            self.macro_build(self.offset_expr, "lui", LUI_FMT, dest, "HI20")
            if self.offset_reloc[0] not in _HI:
                self.macro_build(self.offset_expr, "ori", BITOP_IMM_FMT, dest, dest, "LO12")

    def macro_la(self, op: list, breg: int, gprel_ok: bool) -> bool:
        """macro_la; returns whether $at was used."""
        used_at = False
        rel = self.offset_reloc
        if self.small_poffset_p(0, 1, ISA_ADD_OFFBITS):
            self.macro_build(self.offset_expr, "addiu", ADDIU_FMT, op[0], breg, -1, *rel)
            return False
        if self.small_noffset_p(0, ISA_OFFBITS):
            self.macro_build(self.offset_expr, "addiu", "t,r,h", op[0], breg, -1, *rel)
            return False
        if self.opts.at and op[0] == breg:
            tempreg = self.opts.at
            used_at = True
        else:
            tempreg = op[0]
        self.expression_valid(self.offset_expr, rel[0])
        e = self.offset_expr
        if e.op == "constant":
            self.load_register(tempreg, e)
        else:
            if (gprel_ok and (e.num & ((1 << 64) - 1)) <= MAX_GPREL_OFFSET
                    and not self.need_relax(e.sym, True)
                    and self.opts.mc_model in ("medium", "auto")):
                seg = e.sym.segment()
                self.relax_start(e.sym)
                fmt = (ADDIUGPS_FMT if isinstance(seg, Section)
                       and _special_small_data_section(seg.name) else ADDIUGP_FMT)
                self.macro_build(e, "addiu", fmt, op[0], GP, "GPREL16")
                self.relax_switch()
            if self.opts.pcrel:
                self.pcrel_la(op[0], e, rel[0])
            else:
                self.absolute_la(op[0])
            if self.relax_seq:
                self.relax_end()
        if breg != 0:
            self.macro_build(None, "addu", "d,v,t", op[0], tempreg, breg)
        return used_at

    def absolute_ld_st(self, s: str, fmt: str, op: list, tempreg: int, breg: int) -> None:
        self.macro_build(self.offset_expr, "lui", LUI_FMT, tempreg, "HI20")
        if breg != 0:
            self.macro_build(None, "addu", "d,v,t", tempreg, tempreg, breg)
        self.macro_build(self.offset_expr, s, fmt, op[0], "LO16", tempreg)

    def pcrel_ld_st(self, s: str, fmt: str, op: list, tempreg: int, coproc: bool) -> None:
        if (self.opts.ase & ASE["xNMS"] and self.offset_reloc[0] is None
                and not self.opts.insn32 and not coproc and s[0] in "sl"
                and s[1:2] in ("w", "d") and len(s) == 2):
            self.macro_build(self.offset_expr, s[:2] + "pc", "mp,+S", op[0], "PC_I32")
        else:
            self.macro_build(self.offset_expr, "aluipc", "t,+K", tempreg, "PCREL_HI20")
            self.macro_build(self.offset_expr, s, fmt, op[0], "LO12", tempreg)

    def macro_ld_st(self, s: str, fmt: str, op: list, gpfmt: str | None, align: int,
                    offbits: int, breg: int, tempreg: int, coproc: bool = False) -> bool:
        """macro_ld_st; returns whether $at was used."""
        used_at = False
        rel = self.offset_reloc
        e = self.offset_expr
        if self.small_add_offset_p(0):
            if ((offbits == ISA_OFFBITS or s.startswith("ua"))
                    and self.small_poffset_p(0, align, offbits)):
                self.macro_build(e, s, fmt, op[0], rel[0], breg)
            elif self.small_offset_p(0, align, offbits):
                self.macro_build(None, s, fmt, op[0], e.num, breg)
            else:
                if s[0] == "l" and op[0] == breg and op[0] != 0:
                    tempreg = breg
                elif tempreg == self.opts.at:
                    used_at = True
                if self.small_poffset_p(0, 1, ISA_ADD_OFFBITS):
                    self.macro_build(e, "addiu", ADDIU_FMT, tempreg, breg, -1, *rel)
                elif self.small_noffset_p(0, ISA_OFFBITS):
                    self.macro_build(e, "addiu", "t,r,h", tempreg, breg, -1, *rel)
                self.macro_build(None, s, fmt, op[0], 0, tempreg)
            return used_at
        if tempreg == self.opts.at:
            used_at = True
        self.expression_valid(e, rel[0])
        if e.op == "constant":
            if (op[0] == breg and op[0] != 0 and self.opts.ase & ASE["xNMS"]
                    and _offset_high_part(e.num, offbits) != e.num):
                used_at = False
                self.macro_build(e, "addiu", "mp,mt,+R", op[0], op[0], "I32")
                self.macro_build(None, s, fmt, op[0], 0, op[0])
            else:
                hi = Expr("constant", _offset_high_part(e.num, offbits))
                if offbits != ISA_OFFBITS and hi.num & 0xFFF:
                    hi = e.copy()
                e.num -= hi.num
                self.load_register(tempreg, hi)
                if breg != 0:
                    self.macro_build(None, "addu", "d,v,t", tempreg, tempreg, breg)
                if offbits == ISA_OFFBITS:
                    self.macro_build(e, s, fmt, op[0], "LO12", tempreg)
                else:
                    self.macro_build(None, s, fmt, op[0], e.num, tempreg)
        elif offbits != ISA_OFFBITS:
            t = [tempreg]
            used_at |= self.macro_la(t, 0, not s.startswith("ua"))
            if breg != 0:
                self.macro_build(None, "addu", "d,v,t", tempreg, tempreg, breg)
            self.macro_build(None, s, fmt, op[0], 0, tempreg)
        else:
            if ((e.num & ((1 << 64) - 1)) <= MAX_GPREL_OFFSET
                    and not self.need_relax(e.sym, True)
                    and self.opts.mc_model in ("medium", "auto")):
                self.relax_start(e.sym)
                try:
                    self.macro_build(e, s, gpfmt or fmt, op[0], "GPREL16", GP)
                except _NoFormat:
                    # GNU as gives ``sh`` the byte format "t,+1(ma)" here and
                    # stops with an internal error; eMIPS keeps the absolute
                    # sequence it uses for a label defined earlier
                    self.relax_seq = 0
                else:
                    self.relax_switch()
            if self.opts.pcrel:
                self.pcrel_ld_st(s, fmt, op, tempreg, coproc)
            else:
                self.absolute_ld_st(s, fmt, op, tempreg, breg)
            if self.relax_seq:
                self.relax_end()
        return used_at

    def macro_ldp_stp(self, s: str, fmt: str, op: list, align: int, breg: int,
                      tempreg: int) -> bool:
        used_at = tempreg == self.opts.at
        e = self.offset_expr
        rel = self.offset_reloc
        if self.small_add_offset_p(0) and e.num != 0:
            if breg == 0:
                self.load_register(tempreg, e)
            elif self.small_poffset_p(0, 1, ISA_ADD_OFFBITS):
                self.macro_build(e, "addiu", ADDIU_FMT, tempreg, breg, -1, *rel)
            elif self.small_noffset_p(0, ISA_OFFBITS):
                self.macro_build(e, "addiu", "t,r,h", tempreg, breg, -1, *rel)
            self.macro_build(None, s, fmt, op[0], op[1], tempreg)
            return used_at
        self.expression_valid(e, rel[0])
        if e.op == "constant":
            if e.num & (align - 1) and breg == 0:
                self.error(f"number (0x{e.num:x}) not aligned to {align} bytes")
            if e.num == 0:
                used_at = False
                tempreg = breg
            else:
                self.load_register(tempreg, e)
                if breg != 0:
                    self.macro_build(None, "addu", "d,v,t", tempreg, tempreg, breg)
            self.macro_build(None, s, fmt, op[0], op[1], tempreg)
        else:
            t = [tempreg]
            used_at |= self.macro_la(t, 0, True)
            if breg != 0:
                self.macro_build(None, "addu", "d,v,t", tempreg, tempreg, breg)
            self.macro_build(None, s, fmt, op[0], op[1], tempreg)
        return used_at

    def ld_st_pair(self, s: str, fmt: str, op: list, breg: int, ep: Expr, r,
                   multi_ok: bool) -> None:
        if multi_ok and self.opts.ase & ASE["xNMS"]:
            self.macro_build(ep, s + "m", ISA_MULTI_LDST_FMT, op[0], r, breg, 2)
        else:
            self.macro_build(ep, s, fmt, op[0], r, breg)
            ep.num += 4
            self.macro_build(ep, s, fmt, op[0] + 1, r, breg)

    def macro_ldd_std(self, s: str, fmt: str, op: list, align: int, offbits: int,
                      breg: int) -> bool:
        used_at = False
        at = self.opts.at
        rel = self.offset_reloc
        if self.small_add_offset_p(0):
            ep = self.offset_expr
            expr1 = Expr("constant", 1)
            if offbits == ISA_OFFBITS and self.small_poffset_p(4, align, ISA_OFFBITS):
                rel[0] = "LO12"
            elif offbits == ISA_SIGNED_OFFBITS and self.small_offset_p(4, align,
                                                                       ISA_SIGNED_OFFBITS):
                rel[0] = "SIGNED_9"
            else:
                if self.small_poffset_p(0, align, ISA_ADD_OFFBITS):
                    self.macro_build(self.offset_expr, "addiu", ADDIU_FMT, at, breg, rel[0])
                elif self.small_noffset_p(0, ISA_OFFBITS):
                    self.macro_build(self.offset_expr, "addiu", "t,r,h", at, breg, rel[0])
                expr1 = Expr("constant", 0)
                ep = expr1
                breg = at
                used_at = True
                rel[0] = None
            if "lw" in s and op[0] == breg:
                ep.num += 4
                self.macro_build(ep, s, fmt, op[0] + 1, rel[0], breg)
                ep.num -= 4
                self.macro_build(ep, s, fmt, op[0], rel[0], breg)
            else:
                self.ld_st_pair(s, fmt, op, breg, ep, rel[0],
                                self.small_offset_p(0, align, ISA_SIGNED_OFFBITS)
                                or expr1.num == 0)
            return used_at
        e = self.offset_expr
        self.expression_valid(e, rel[0])
        if (e.op == "symbol" and (e.num & ((1 << 64) - 1)) <= MAX_GPREL_OFFSET
                and not s.startswith("ua")
                and breg == 0 and (rel[0] in _GPREL or rel[0] is None)
                and not self.need_relax(e.sym, True)):
            self.relax_start(e.sym)
            self.macro_build(e, s, "t,.(ma)", op[0], "GPREL16", GP)
            e.num += 4
            self.macro_build(e, s, "t,.(ma)", op[0] + 1, "GPREL16", GP)
            self.relax_switch()
            e.num -= 4
        used_at = True
        if self.opts.pcrel:
            self._pcrel_ldd_std(s, fmt, op, offbits)
        else:
            self._absolute_ldd_std(s, fmt, op, breg, offbits)
        if self.relax_seq:
            self.relax_end()
        return used_at

    def _absolute_ldd_std(self, s, fmt, op, breg, offbits) -> None:
        at = self.opts.at
        e = self.offset_expr
        if self.opts.ase & ASE["xNMS"] and not self.opts.insn32:
            self.macro_build(e, "li", "mp,+Q", at, "I32")
            self.offset_expr = e = Expr("constant", 0)
        else:
            self.macro_build(e, "lui", LUI_FMT, at, "HI20")
            if (offbits != ISA_OFFBITS
                    or _offset_high_part(e.num, ISA_OFFBITS)
                    != _offset_high_part(e.num + 4, ISA_OFFBITS)):
                self.macro_build(e, "ori", BITOP_IMM_FMT, at, at, "LO12")
                self.offset_expr = e = Expr("constant", 0)
        if breg != 0:
            self.macro_build(None, "addu", "d,v,t", at, breg, at)
        self.ld_st_pair(s, fmt, op, at, e, "LO12", e.op == "constant")

    def _pcrel_ldd_std(self, s, fmt, op, offbits) -> None:
        at = self.opts.at
        e = self.offset_expr
        if offbits != ISA_OFFBITS:
            an = Expr("constant", 4)
            self.pcrel_la(at, e, self.offset_reloc[0])
            self.ld_st_pair(s, fmt, op, at, an, "SIGNED_9", True)
        elif self.opts.ase & ASE["xNMS"] and self.offset_reloc[0] is None \
                and not self.opts.insn32:
            insn = "lwpc" if s == "lw" else "swpc"
            self.macro_build(e, insn, "mp,+S", op[0], "PC_I32")
            e.num += 4
            self.macro_build(e, insn, "mp,+S", op[0] + 1, "PC_I32")
        else:
            self.macro_build(e, "aluipc", "t,+K", at, "PCREL_HI20")
            self.ld_st_pair(s, fmt, op, at, e, "LO12", False)

    def set_at(self, reg: int, unsigned: bool) -> None:
        if _offset_high_part(self.imm_expr.num, ISA_OFFBITS) == 0:
            self.macro_build(self.imm_expr, "sltiu" if unsigned else "slti", OP_IMM_FMT,
                             self.opts.at, reg, "LO16")
        else:
            self.load_register(self.opts.at, self.imm_expr)
            self.macro_build(None, "sltu" if unsigned else "slt", "d,v,t", self.opts.at,
                             reg, self.opts.at)

    def label_expr(self) -> Expr:
        """nanomips_label_expr: the macro's local label (defined later)."""
        if not self.label_name:
            self.label_name = f"L\037{self.label_counter}"
        return Expr("symbol", 0, self.symbol(self.label_name))

    def add_label(self) -> None:
        self.define_label(self.label_expr().sym.name)
        self.label_counter += 1
        self.label_name = ""

    def macro(self, ip: Insn) -> None:
        """macro: expand a macro entry (tc-nanomips.c)."""
        mo = ip.mo
        op: list[int] = []
        a = 0
        while a < len(mo.args):
            c = mo.args[a]
            if c in ",()":
                a += 1
                continue
            if c == "#":
                a += 2
                continue
            o = OPERANDS.get(operand_code(mo.args, a))
            op.append(o.extract(ip.opcode) if o is not None else -1)
            a += 2 if c in "+m-" else 1
        op += [-1] * (6 - len(op))
        m = mo.macro
        s = mo.name
        at = self.opts.at
        used_at = False
        imm = self.imm_expr
        off = self.offset_expr
        expr1 = Expr("constant", 1)
        mb = self.macro_build

        if m == "M_ABS":
            lab = self.label_expr()
            if op[0] != op[1]:
                self.move_register(op[0], op[1])
            mb(lab, "bgezc", BCONDZ1_FMT, op[1])
            mb(None, "subu", "d,v,t", op[0], 0, op[1])
            self.add_label()
        elif m in ("M_ADD_I", "M_ADDU_I"):
            s, s2 = "addiu", ("add" if m == "M_ADD_I" else "addu")
            if m == "M_ADD_I" and imm.num != 0:
                used_at = True
                self.load_register(at, imm)
                mb(None, s2, "d,v,t", op[0], op[1], at)
            elif _offset_high_unsigned(imm.num, 16) == 0:
                mb(imm, s, ADDIU_FMT, op[0], op[1], "IMM16")
            elif _offset_high_unsigned(-imm.num, 12) == 0:
                mb(imm, s, "t,r,h", op[0], op[1], "NEG12")
            elif self.opts.ase & ASE["xNMS"] and op[0] == op[1] and not self.opts.insn32:
                mb(imm, s, "mp,mt,+R", op[0], op[0], "I32")
            else:
                used_at = True
                self.load_register(at, imm)
                mb(None, s2, "d,v,t", op[0], op[1], at)
        elif m in ("M_AND_I", "M_OR_I", "M_NOR_I", "M_XOR_I"):
            s, s2 = {"M_AND_I": ("andi", "and"), "M_OR_I": ("ori", "or"),
                     "M_NOR_I": ("", "nor"), "M_XOR_I": ("xori", "xor")}[m]
            if m != "M_NOR_I" and _offset_high_unsigned(imm.num, 12) == 0:
                mb(imm, s, BITOP_IMM_FMT, op[0], op[1], "LO16")
            else:
                used_at = True
                self.load_register(at, imm)
                mb(None, s2, "d,v,t", op[0], op[1], at)
        elif m in ("M_BEQ_I", "M_BNE_I"):
            kind = "beq" if m == "M_BEQ_I" else "bne"
            if imm.num == 0:
                self.branch_rsrt(kind, off, op[0], 0)
            elif _offset_high_unsigned(imm.num, 7) == 0:
                self.branch_rtim("beqic" if m == "M_BEQ_I" else "bneic", off, op[0], imm)
            else:
                used_at = True
                self.load_register(at, imm)
                self.branch_rsrt(kind, off, op[0], at)
        elif m in ("M_BGE", "M_BGT", "M_BLE", "M_BLT"):
            z1, z0, swap, kind = {"M_BGE": ("bgez", "blez", False, "beq"),
                                  "M_BGT": ("bgtz", "bltz", True, "bne"),
                                  "M_BLE": ("blez", "bgez", True, "beq"),
                                  "M_BLT": ("bltz", "bgtz", False, "bne")}[m]
            if op[1] == 0:
                self.branch_rs(z1, off, op[0])
            elif op[0] == 0:
                self.branch_rs(z0, off, op[1])
            else:
                used_at = True
                a1, a2 = (op[1], op[0]) if swap else (op[0], op[1])
                mb(None, "slt", "d,v,t", at, a1, a2)
                self.branch_rsrt(kind, off, at, 0)
        elif m in ("M_BGEZAL", "M_BLTZAL"):
            lab = self.label_expr()
            mb(lab, "bltzc" if m == "M_BGEZAL" else "bgezc", BCONDZ1_FMT, op[0])
            mb(off, "bal", B_FMT)
            self.add_label()
        elif m in ("M_BGT_I", "M_BGE_I"):
            imm = imm.copy()
            if m == "M_BGT_I":
                if imm.num >= GPR_SMAX:
                    mb(None, "nop", "")
                    return self._macro_done(used_at)
                imm.num += 1
            if imm.num == 0:
                self.branch_rs("bgez", off, op[0])
            elif imm.num == 1:
                self.branch_rs("bgtz", off, op[0])
            elif imm.num <= GPR_SMIN:
                self.warn(f"branch {mo.name} is always true")
                mb(off, "b", B_FMT)
            elif _offset_high_unsigned(imm.num, 7) == 0:
                self.branch_rtim("bgeic", off, op[0], imm)
            else:
                used_at = True
                self.imm_expr = imm
                self.set_at(op[0], False)
                self.branch_rsrt("beq", off, at, 0)
        elif m in ("M_BGTU_I", "M_BGEU_I"):
            imm = imm.copy()
            if m == "M_BGTU_I":
                if op[0] == 0 or imm.num == -1:
                    mb(None, "nop", "")
                    return self._macro_done(used_at)
                imm.num += 1
            if imm.num == 0:
                self.warn(f"branch {mo.name} is always true")
                mb(off, "b", B_FMT)
            elif imm.num == 1:
                self.branch_rsrt("bne", off, op[0], 0)
            elif _offset_high_unsigned(imm.num, 7) == 0:
                self.branch_rtim("bgeiuc", off, op[0], imm)
            else:
                used_at = True
                self.imm_expr = imm
                self.set_at(op[0], True)
                self.branch_rsrt("beq", off, at, 0)
        elif m == "M_BGTU":
            if op[1] == 0:
                self.branch_rsrt("bne", off, op[0], 0)
            elif op[0] == 0:
                mb(None, "nop", "")
            else:
                used_at = True
                mb(None, "sltu", "d,v,t", at, op[1], op[0])
                self.branch_rsrt("bne", off, at, 0)
        elif m in ("M_BLE_I", "M_BLT_I"):
            imm = imm.copy()
            if m == "M_BLE_I":
                if imm.num >= GPR_SMAX:
                    self.warn(f"branch {mo.name} is always true")
                    mb(off, "b", B_FMT)
                    return self._macro_done(used_at)
                imm.num += 1
            if imm.num == 0:
                self.branch_rs("bltz", off, op[0])
            elif imm.num == 1:
                self.branch_rs("blez", off, op[0])
            elif _offset_high_unsigned(imm.num, 7) == 0:
                self.branch_rtim("bltic", off, op[0], imm)
            else:
                used_at = True
                self.imm_expr = imm
                self.set_at(op[0], False)
                self.branch_rsrt("bne", off, at, 0)
        elif m == "M_BLEU":
            if op[1] == 0:
                self.branch_rsrt("beq", off, op[0], 0)
            elif op[0] == 0:
                self.warn(f"branch {mo.name} is always true")
                mb(off, "b", B_FMT)
            else:
                used_at = True
                mb(None, "sltu", "d,v,t", at, op[1], op[0])
                self.branch_rsrt("beq", off, at, 0)
        elif m in ("M_BLEU_I", "M_BLTU_I"):
            imm = imm.copy()
            if m == "M_BLEU_I":
                if op[0] == 0 or imm.num == -1:
                    self.warn(f"branch {mo.name} is always true")
                    mb(off, "b", B_FMT)
                    return self._macro_done(used_at)
                imm.num += 1
            if imm.num == 0:
                mb(None, "nop", "")
            elif imm.num == 1:
                self.branch_rsrt("beq", off, op[0], 0)
            elif _offset_high_unsigned(imm.num, 7) == 0:
                self.branch_rtim("bltiuc", off, op[0], imm)
            else:
                used_at = True
                self.imm_expr = imm
                self.set_at(op[0], True)
                self.branch_rsrt("bne", off, at, 0)
        elif m == "M_BLTU":
            if op[1] == 0:
                mb(None, "nop", "")
            elif op[0] == 0:
                self.branch_rsrt("bne", off, 0, op[1])
            else:
                used_at = True
                mb(None, "sltu", "d,v,t", at, op[0], op[1])
                self.branch_rsrt("bne", off, at, 0)
        elif m == "M_EXT":
            if 31 - op[2] - op[3] > 0:
                mb(None, "sll", SHFT_FMT, op[0], op[1], 31 - op[2] - op[3])
                mb(None, "srl", SHFT_FMT, op[0], op[0], 31 - op[3])
            else:
                mb(None, "srl", SHFT_FMT, op[0], op[1], op[2])
        elif m == "M_INS":
            if op[0] == op[1] and op[2] == 0:
                mb(None, "nop", "")
                return self._macro_done(used_at)
            if op[0] == op[1]:
                used_at = True
                mb(None, "extw", "d,s,t,+I", at, op[0], op[0], op[2])
                mb(None, "extw", "d,s,t,+I", op[0], at, op[0], 1 + op[3] - op[2])
            elif op[2] == 0 and op[3] == 31:
                self.move_register(op[0], op[1])
                return self._macro_done(used_at)
            else:
                if op[2] != 0:
                    mb(None, "extw", "d,s,t,+I", op[0], op[0], op[0], op[2])
                mb(None, "extw", "d,s,t,+I", op[0], op[0], op[1], 1 + op[3] - op[2])
            if op[3] < 31:
                mb(None, "extw", "d,s,t,+I", op[0], op[0], op[0], 31 - op[3])
        elif m in ("M_DIV_I", "M_DIVU_I", "M_REM_3I", "M_MOD_I", "M_MODU_I"):
            if m in ("M_REM_3I", "M_MOD_I"):
                s = "mod"
            elif m == "M_MODU_I":
                s = "modu"
            if imm.num == 0:
                self.warn("divide by zero")
                mb(None, "break", _brk_fmt(self.opts), 7)
            elif imm.num == 1:
                self.move_register(op[0], op[1] if s.startswith("div") else 0)
            elif imm.num == -1 and not s.endswith("u"):
                if s.startswith("div"):
                    mb(None, "neg", "d,w", op[0], op[1])
                else:
                    self.move_register(op[0], 0)
            else:
                used_at = True
                self.load_register(at, imm)
                mb(None, s, "d,v,t", op[0], op[1], at)
        elif m == "M_LA_AB":
            used_at = self.macro_la(op, op[2], True)
        elif m == "M_J_A":
            mb(off, "bc", B_FMT)
        elif m == "M_JAL_A":
            mb(off, "jal", B_FMT)
        elif m in ("M_LB_AC", "M_LBU_AC", "M_LH_AC", "M_LHU_AC", "M_LW_AC"):
            gpfmt = {"b": "t,+1(ma)", "d": "t,+4(ma)", "h": "t,+3(ma)"}.get(
                s[1], "t,+2(ma)" if s[2:3] == "u" else LWGP_FMT)
            used_at = self._ld(s, ISA_UNSIGNED_LDST_FMT, op, gpfmt, 1, ISA_OFFBITS)
        elif m == "M_LL_AC":
            used_at = self._ld(s, LL_SC_FMT, op, None, 4, ISA_SIGNED_OFFBITS)
        elif m == "M_LLWP_AC":
            breg = op[3]
            if op[0] != 0 and op[0] != breg:
                tempreg = op[0]
            elif op[1] != 0 and op[1] != breg:
                tempreg = op[1]
            else:
                tempreg = at
            used_at = self.macro_ldp_stp(s, LLP_SCP_FMT, op, 8, breg, tempreg)
        elif m in ("M_LBUE_AC", "M_LHUE_AC", "M_LBE_AC", "M_LHE_AC", "M_SHE_AC", "M_SWE_AC",
                   "M_LWE_AC", "M_SBE_AC"):
            used_at = self.macro_ld_st(s, ISA_SIGNED_LDST_FMT, op, None, 1,
                                       ISA_SIGNED_OFFBITS, op[2], at)
        elif m in ("M_SB_AC", "M_SH_AC"):
            used_at = self.macro_ld_st(s, ISA_UNSIGNED_LDST_FMT, op, "t,+1(ma)", 1,
                                       ISA_OFFBITS, op[2], at)
        elif m == "M_SW_AC":
            used_at = self.macro_ld_st(s, ISA_UNSIGNED_LDST_FMT, op, LWGP_FMT, 1, ISA_OFFBITS,
                                       op[2], at)
        elif m in ("M_LLE_AC", "M_SCE_AC", "M_SC_AC"):
            used_at = self.macro_ld_st(s, LL_SC_FMT, op, None, 4, ISA_SIGNED_OFFBITS,
                                       op[2], at)
        elif m == "M_SCWP_AC":
            used_at = self.macro_ldp_stp(s, LLP_SCP_FMT, op, 8, op[3], at)
        elif m in ("M_CACHE_AC", "M_PREF_AC", "M_CACHEE_AC", "M_PREFE_AC"):
            used_at = self.macro_ld_st(s, PREF_FMT, op, None, 1, ISA_SIGNED_OFFBITS,
                                       op[2], at)
        elif m in ("M_SWM_AC", "M_SDM_AC", "M_LWM_AC", "M_LDM_AC"):
            used_at = self.macro_ld_st(s, ISA_SIGNED_LDST_FMT + ",|", op, None, 1,
                                       ISA_SIGNED_OFFBITS, op[2], at)
        elif m in ("M_LWX_AB", "M_LHX_AB", "M_LHUX_AB", "M_LBX_AB", "M_LBUX_AB"):
            s = {"M_LWX_AB": "lwx", "M_LHX_AB": "lhx", "M_LHUX_AB": "lhux",
                 "M_LBX_AB": "lbx", "M_LBUX_AB": "lbux"}[m]
            if op[0] == 0 or op[0] == op[2]:
                used_at = self._st_indexed(s, op)
            else:
                t = [op[0]]
                self.macro_la(t, 0, True)
                mb(None, s, "d,s(t)", op[0], op[0], op[2])
        elif m in ("M_SWX_AB", "M_SHX_AB", "M_SBX_AB"):
            s = {"M_SWX_AB": "swx", "M_SHX_AB": "shx", "M_SBX_AB": "sbx"}[m]
            used_at = self._st_indexed(s, op)
        elif m in ("M_LDX_AB", "M_SDX_AB"):
            s = "lwx" if m == "M_LDX_AB" else "swx"
            if not at:
                self.error('macro used $at after ".set noat"')
            used_at = True
            self.imm_expr = Expr("constant", 4)
            self.macro_la([at], 0, True)
            mb(None, s, "d,s(t)", op[0], at, op[2])
            mb(self.imm_expr, "addiu", ADDIU_FMT if self.opts.insn32 else "mp,mk,mQ",
               at, at, "LO12")
            mb(None, s, "d,s(t)", op[0] + 1, at, op[2])
        elif m == "M_JRADDIUSP":
            self.offset_expr = imm.copy()
            if self.small_poffset_p(0, 1, ISA_ADD_OFFBITS):
                mb(imm, "addiu", ADDIU_FMT, SP, SP, "IMM16")
            elif self.small_noffset_p(0, ISA_OFFBITS):
                mb(imm, "addiu", ADDIU_FMT, SP, SP, "NEG12")
            else:
                used_at = True
                self.load_register(at, imm)
                mb(None, "addu", "d,v,t", SP, SP, at)
            mb(None, "jrc", "s" if self.opts.insn32 else "mp", RA)
        elif m == "M_LI":
            if imm.op != "absent":
                self.load_register(op[0], imm)
            else:
                used_at = self.macro_la(op, 0, False)
        elif m in ("M_LI_S", "M_LI_D"):
            if imm.op == "constant" and m == "M_LI_S":
                self.load_register(op[0], imm)
            elif imm.op == "constant":
                hreg, lreg = (op[0], op[0] + 1) if self.big else (op[0] + 1, op[0])
                if hreg <= 31:
                    self.load_register(hreg, imm)
                if lreg <= 31:
                    if off.op == "absent":
                        self.move_register(lreg, 0)
                    else:
                        self.load_register(lreg, off)
            else:
                self.error(f"{mo.name}: a constant that needs memory is not supported")
        elif m in ("M_LD_AC", "M_SD_AC"):
            s = "lw" if m == "M_LD_AC" else "sw"
            used_at = self.macro_ldd_std(s, ISA_UNSIGNED_LDST_FMT, op, 1, ISA_OFFBITS, op[2])
        elif m == "M_MUL":
            mb(None, "multu", "s,t", op[1], op[2])
            mb(None, "mflo", "s" if self.opts.insn32 else "mj", op[0])
        elif m == "M_MUL_I":
            used_at = True
            self.load_register(at, imm)
            mb(None, "mul", "d,v,t", op[0], op[1], at)
        elif m == "M_ROL":
            if op[0] == op[1]:
                tempreg, used_at = at, True
            else:
                tempreg = op[0]
            mb(None, "negu", "d,w", tempreg, op[2])
            mb(None, "rorv", "d,v,t", op[0], op[1], tempreg)
        elif m == "M_ROL_I":
            rot = imm.num & 0x1F
            mb(None, "ror", SHFT_FMT, op[0], op[1], (32 - rot) & 0x1F)
        elif m == "M_ROR_I":
            mb(None, "ror", SHFT_FMT, op[0], op[1], imm.num & 0x1F)
        elif m == "M_SEQ":
            if op[1] == 0:
                mb(expr1, "sltiu", OP_IMM_FMT, op[0], op[2], "LO16")
            elif op[2] == 0:
                mb(expr1, "sltiu", OP_IMM_FMT, op[0], op[1], "LO16")
            else:
                mb(None, "xor", "d,v,t", op[0], op[1], op[2])
                mb(expr1, "sltiu", OP_IMM_FMT, op[0], op[0], "LO16")
        elif m == "M_SEQ_I":
            if imm.num == 0:
                mb(expr1, "sltiu", OP_IMM_FMT, op[0], op[1], "LO16")
            elif op[1] == 0:
                self.warn(f"instruction {mo.name}: result is always false")
                self.move_register(op[0], 0)
            elif _offset_high_unsigned(imm.num, ISA_OFFBITS) == 0:
                mb(imm, "seqi", OP_IMM_FMT, op[0], op[1], "LO16")
            else:
                if _offset_high_unsigned(-imm.num, ISA_ADD_OFFBITS) == 0:
                    neg = Expr("constant", -imm.num)
                    mb(neg, "addiu", ADDIU_FMT, op[0], op[1], "LO16")
                else:
                    used_at = True
                    self.load_register(at, imm)
                    mb(None, "xor", "d,v,t", op[0], op[1], at)
                mb(expr1, "sltiu", OP_IMM_FMT, op[0], op[0], "LO16")
        elif m in ("M_SGE", "M_SGEU"):
            mb(None, "slt" if m == "M_SGE" else "sltu", "d,v,t", op[0], op[1], op[2])
            mb(expr1, "xori", BITOP_IMM_FMT, op[0], op[0], "LO16")
        elif m in ("M_SGE_I", "M_SGEU_I"):
            ins = "slti" if m == "M_SGE_I" else "sltiu"
            if _offset_high_unsigned(imm.num, ISA_OFFBITS) == 0:
                mb(imm, ins, OP_IMM_FMT, op[0], op[1], "LO16")
            else:
                used_at = True
                self.load_register(at, imm)
                mb(None, {"slti": "slt", "sltiu": "sltu"}[ins], "d,v,t", op[0], op[1], at)
            mb(expr1, "xori", BITOP_IMM_FMT, op[0], op[0], "LO16")
        elif m in ("M_SGT", "M_SGTU"):
            mb(None, "slt" if m == "M_SGT" else "sltu", "d,v,t", op[0], op[2], op[1])
        elif m in ("M_SGT_I", "M_SGTU_I"):
            used_at = True
            self.load_register(at, imm)
            mb(None, "slt" if m == "M_SGT_I" else "sltu", "d,v,t", op[0], at, op[1])
        elif m in ("M_SLE", "M_SLEU"):
            mb(None, "slt" if m == "M_SLE" else "sltu", "d,v,t", op[0], op[2], op[1])
            mb(expr1, "xori", BITOP_IMM_FMT, op[0], op[0], "LO16")
        elif m in ("M_SLE_I", "M_SLEU_I"):
            used_at = True
            self.load_register(at, imm)
            mb(None, "slt" if m == "M_SLE_I" else "sltu", "d,v,t", op[0], at, op[1])
            mb(expr1, "xori", BITOP_IMM_FMT, op[0], op[0], "LO16")
        elif m in ("M_SLT_I", "M_SLTU_I"):
            ins = "slti" if m == "M_SLT_I" else "sltiu"
            if _offset_high_unsigned(imm.num, ISA_OFFBITS) == 0:
                mb(imm, ins, OP_IMM_FMT, op[0], op[1], "LO16")
            else:
                used_at = True
                self.load_register(at, imm)
                mb(None, {"slti": "slt", "sltiu": "sltu"}[ins], "d,v,t", op[0], op[1], at)
        elif m == "M_SNE":
            if op[1] == 0:
                mb(None, "sltu", "d,v,t", op[0], 0, op[2])
            elif op[2] == 0:
                mb(None, "sltu", "d,v,t", op[0], 0, op[1])
            else:
                mb(None, "xor", "d,v,t", op[0], op[1], op[2])
                mb(None, "sltu", "d,v,t", op[0], 0, op[0])
        elif m == "M_SNE_I":
            if imm.num == 0:
                mb(None, "sltu", "d,v,t", op[0], 0, op[1])
            elif op[1] == 0:
                self.warn(f"instruction {mo.name}: result is always true")
                self.load_register(op[0], expr1)
            else:
                if _offset_high_unsigned(imm.num, ISA_ADD_OFFBITS) == 0:
                    mb(imm, "xori", BITOP_IMM_FMT, op[0], op[1], "LO16")
                elif _offset_high_unsigned(-imm.num, ISA_ADD_OFFBITS) == 0:
                    neg = Expr("constant", -imm.num)
                    mb(neg, "addiu", ADDIU_FMT, op[0], op[1], "LO16")
                else:
                    used_at = True
                    self.load_register(at, imm)
                    mb(None, "xor", "d,v,t", op[0], op[1], op[2])
                mb(None, "sltu", "d,v,t", op[0], 0, op[0])
        elif m in ("M_SUB_I", "M_SUBU_I"):
            s2 = "sub" if m == "M_SUB_I" else "subu"
            neg = Expr("constant", -imm.num)
            if m == "M_SUBU_I" and _offset_high_unsigned(-imm.num, 16) == 0:
                mb(neg, "addiu", ADDIU_FMT, op[0], op[1], "IMM16")
            elif m == "M_SUBU_I" and _offset_high_unsigned(imm.num, 12) == 0:
                mb(neg, "addiu", "t,r,h", op[0], op[1], "NEG12")
            elif (m == "M_SUBU_I" and self.opts.ase & ASE["xNMS"] and op[0] == op[1]
                  and not self.opts.insn32):
                mb(neg, "addiu", "mp,mt,+R", op[0], op[0], "I32")
            else:
                used_at = True
                self.load_register(at, imm)
                mb(None, s2, "d,v,t", op[0], op[1], at)
        elif m in ("M_TEQ_I", "M_TNE_I"):
            used_at = True
            self.load_register(at, imm)
            mb(None, s, "s,t", op[0], at)
        elif m in ("M_ULH_AC", "M_ULW_AC", "M_USH_AC", "M_USW_AC"):
            s = {"M_ULH_AC": "ualh", "M_ULW_AC": "ualw", "M_USH_AC": "uash",
                 "M_USW_AC": "uasw"}[m]
            used_at = self.macro_ld_st(s, ISA_SIGNED_LDST_FMT, op, None, 1,
                                       ISA_SIGNED_OFFBITS, op[2], at)
        elif m in ("M_ULD_AC", "M_USD_AC"):
            s = "ualw" if m == "M_ULD_AC" else "uasw"
            used_at = self.macro_ldd_std(s, ISA_SIGNED_LDST_FMT, op, 1, ISA_SIGNED_OFFBITS,
                                         op[2])
        else:
            self.error(f"macro {mo.name} not implemented yet")
        self._macro_done(used_at)

    def _macro_done(self, used_at: bool) -> None:
        if not self.opts.at and used_at:
            self.error('macro used $at after ".set noat"')

    def _ld(self, s: str, fmt: str, op: list, gpfmt, align: int, offbits: int) -> bool:
        """macro()'s ``ld:`` path (a load uses its own target as the scratch
        register unless that is the base or $0)."""
        if op[2] == op[0] or op[0] == 0:
            return self.macro_ld_st(s, fmt, op, gpfmt, align, offbits, op[2], self.opts.at)
        return self.macro_ld_st(s, fmt, op, gpfmt, align, offbits, op[2], op[0])

    def _st_indexed(self, s: str, op: list) -> bool:
        if not self.opts.at:
            self.error('macro used $at after ".set noat"')
        t = [self.opts.at]
        self.macro_la(t, 0, True)
        self.macro_build(None, s, "d,s(t)", op[0], self.opts.at, op[2])
        return True


class _Directives:
    """GNU as's directives (read.c, obj-elf.c, tc-nanomips.c)."""

    _IGNORED = frozenset({".hidden", ".protected", ".internal", ".type", ".size", ".ent", ".aent", ".end", ".frame",
                          ".mask", ".fmask", ".file", ".loc", ".ident", ".cpload",
                          ".cplocal", ".cprestore", ".cpreturn", ".cpadd", ".extern",
                          ".verstamp", ".bgnb", ".endb", ".gnu_attribute", ".cfi_startproc",
                          ".cfi_endproc", ".cfi_def_cfa", ".cfi_offset", ".cfi_def_cfa_offset",
                          ".cfi_restore", ".cfi_sections", ".cfi_def_cfa_register",
                          ".nan", ".abicalls", ".option", ".previous_dummy"})

    def directive(self, d: str, rest: str) -> None:
        if d in self._IGNORED:
            return
        if d in (".globl", ".global", ".weak", ".local"):
            # S_SET_EXTERNAL / S_SET_WEAK / S_CLEAR_EXTERNAL
            for name in _split_args(rest):
                name = name.strip()
                if not name:
                    continue
                sym = self.symbol(name)
                if d == ".weak":
                    sym.weak = True
                elif d == ".local":
                    sym.external = sym.weak = False
                else:
                    sym.external = True
            return
        if d == ".linkrelax":
            self.s_linkrelax()
            return
        if d in (".text", ".data", ".bss", ".rdata", ".sdata", ".sbss", ".ktext", ".kdata"):
            if rest:
                v = self.absolute(rest)
                if v != 0:
                    self.error(f"{d} {rest}: subsections are not supported (eMIPS lays the "
                               "segments out itself)")
            self.switch_section(d)
            return
        if d == ".section":
            name = re.split(r"[\s,]", rest.strip(), 1)[0].strip('"')
            if not name:
                self.error(".section needs a name")
            flags = rest.split(",")[1].strip().strip('"') if "," in rest else ""
            self.switch_section(name)
            if "x" in flags:
                self.section.code = True
            return
        if d == ".previous":
            if self.prev_section is None:
                self.error(".previous without a corresponding .section")
            prev = self.prev_section
            self.switch_section(prev.name)
            return
        if d == ".set":
            self.set_directive(rest)
            return
        if d == ".module":
            if self.file_checked:
                self.error(".module is not permitted after generating code")
            if not self.code_option(rest.strip()):
                self.error(f".module used with unrecognized symbol: {rest.strip()}")
            self.file_opts = self.opts.copy()
            return
        if d in (".equ", ".equiv", ".eqv"):
            name, _, value = rest.partition(",")
            name = name.strip()
            if d == ".equiv" and self.symbols.get(name) is not None \
                    and self.symbols[name].defined:
                self.error(f"symbol `{name}' is already defined")
            self._assign(name, value)
            return
        if d in (".byte", ".hword", ".half", ".short", ".2byte", ".word", ".int", ".long",
                 ".4byte", ".dword", ".quad", ".8byte"):
            size = {".byte": 1, ".hword": 2, ".half": 2, ".short": 2, ".2byte": 2, ".word": 4,
                    ".int": 4, ".long": 4, ".4byte": 4, ".dword": 8, ".quad": 8,
                    ".8byte": 8}[d]
            self.cons(size, rest, auto=d not in (".2byte", ".4byte", ".8byte"))
            return
        if d in (".float", ".single", ".double"):
            self.float_cons(d, rest)
            return
        if d in (".ascii", ".asciz", ".asciiz", ".string"):
            self.stringer(rest, zero=d != ".ascii")
            return
        if d in (".space", ".skip", ".zero"):
            parts = _split_args(rest)
            n = self.absolute(parts[0])
            fill = self.absolute(parts[1]) if len(parts) > 1 and d != ".zero" else 0
            self.space(n, fill)
            return
        if d == ".fill":
            # s_fill: ``repeat`` copies of a ``size``-byte pattern; the value
            # fills at most its first 4 bytes, the rest are zeros
            parts = _split_args(rest)
            repeat = self.absolute(parts[0])
            size = self.absolute(parts[1]) if len(parts) > 1 and parts[1] else 1
            value = self.absolute(parts[2]) if len(parts) > 2 and parts[2] else 0
            if size > 8:
                self.warn(".fill size clamped to 8")
                size = 8
            if repeat > 0 and size > 0:
                n = min(size, 4)
                pat = (value & ((1 << (8 * n)) - 1)).to_bytes(n, "big" if self.big
                                                              else "little")
                self.flush_pending_output_keep_labels()
                f = self.frag
                f.repeat = repeat
                f.pattern = pat + bytes(size - n)
                self.frag_close()
                self.labels = []
            return
        if d == ".align":
            self.s_align(rest)
            return
        if d in (".balign", ".p2align"):
            parts = _split_args(rest)
            n = self.absolute(parts[0]) if parts and parts[0] else 0
            if d == ".balign":
                if n & (n - 1):
                    self.error("alignment not a power of 2")
                power = n.bit_length() - 1 if n else 0
            else:
                power = n
            fill = self.absolute(parts[1]) if len(parts) > 1 and parts[1] else None
            mx = self.absolute(parts[2]) if len(parts) > 2 and parts[2] else 0
            self.flush_pending_output()        # read.c do_align: the labels stay
            if self.opts.linkrelax and self.section.code and power > 1:   # md_do_align
                self.create_align_relocs(power, 0 if fill is None else fill & 0xFF,
                                         0 if fill is None else 1, mx)
            self.do_align(power, fill, mx)
            return
        if d in _NOT_MODELLED:
            self.error(f"directive {d} is not supported by the eMIPS nanoMIPS assembler")
        self.error(f"unknown pseudo-op: `{d}'")

    #: the directives this assembler handles itself (emips.gasread passes them
    #: on; the md table wins over gas's: MIPS's .end ends a function)
    KNOWN = frozenset((".2byte", ".4byte", ".8byte", ".align", ".ascii", ".asciiz", ".asciz",
                       ".balign", ".bss", ".byte", ".data", ".double", ".dword", ".equ",
                       ".equiv", ".eqv", ".fill", ".float", ".global", ".globl", ".half",
                       ".hword", ".int", ".kdata", ".ktext", ".linkrelax", ".local", ".long",
                       ".module", ".p2align", ".previous", ".quad", ".rdata", ".sbss",
                       ".sdata", ".section", ".set", ".short", ".single", ".skip", ".space",
                       ".string", ".text", ".weak", ".word", ".zero"))

    def absolute(self, text: str) -> int:
        e, j = _Expressions(self).parse(text.strip(), 0)
        if e.op != "constant" or text.strip()[j:].strip():
            self.error(f"bad or irreducible absolute expression `{text.strip()}'")
        return e.num

    def _assign(self, name: str, value: str) -> None:
        name = name.strip()
        e, j = _Expressions(self).parse(value.strip(), 0)
        if value.strip()[j:].strip():
            self.error(f"junk at end of line: `{value.strip()[j:].strip()}'")
        sym = self.symbol(name)
        if sym.frag is not None:
            self.error(f"symbol `{name}' is already defined")
        sym.absolute = None
        sym.equated = None
        if e.op == "constant":
            sym.absolute = e.num
        elif e.op == "symbol":
            sym.equated = (e.sym, e.num)
        else:
            self.error(f"invalid value for symbol `{name}'")

    def set_directive(self, rest: str) -> None:
        self.file_checked = True               # file_check_options
        name = rest.strip()
        if "," in name:
            sym, _, value = name.partition(",")
            self._assign(sym, value)
            return
        o = self.opts
        if name == "reorder":
            o.noreorder = 0
        elif name == "noreorder":
            o.noreorder = 1
        elif name == "macro":
            o.nomacro = False
        elif name == "nomacro":
            o.nomacro = True
        elif name == "push":
            self.opts_stack.append(o.copy())
        elif name == "pop":
            if not self.opts_stack:
                self.error(".set pop with no .set push")
            saved = self.opts_stack.pop()
            if saved.linkrelax != o.linkrelax:  # the R_NANOMIPS_[NO]RELAX flips it
                self.toggle_linkrelax = True
                saved.linkrelax = o.linkrelax
            self.opts = saved
        elif name == "nolinkrelax":
            self.toggle_linkrelax = self.linkrelax and o.linkrelax
        elif name == "linkrelax":
            self.toggle_linkrelax = self.linkrelax and not o.linkrelax
        elif name == "arch=default":
            o.ase = (o.ase & ~ASE["xNMS"]) | (self.file_opts.ase & ASE["xNMS"])
        elif name in ("legacyregs", "nolegacyregs"):
            if name == "legacyregs":
                self.error(".set legacyregs is not supported (the p32 names only)")
        elif not self.code_option(name):
            self.warn(f"tried to set unrecognized symbol: {name}")

    def code_option(self, name: str) -> bool:
        """parse_code_option."""
        o = self.opts
        if name.startswith("at="):
            r = REGISTERS.get(name[3:])
            if r is None or not r & (RTYPE_NUM | RTYPE_GP):
                self.error(f"unrecognized register name `{name[3:]}'")
            o.at = r & RNUM_MASK
        elif name == "at":
            o.at = ATREG
        elif name == "noat":
            o.at = ZERO
        elif name in ("softfloat", "hardfloat", "singlefloat", "doublefloat"):
            pass
        elif name in ("mips16", "MIPS-16"):
            self.error("nanoMIPS does not support MIPS16 ASE")
        elif name == "micromips":
            self.error("micromips is incompatible with nanoMIPS")
        elif name.startswith("no") and name[2:] in SET_ASES:
            o.ase &= ~ASE[SET_ASES[name[2:]]]
        elif name in SET_ASES:
            o.ase |= ASE[SET_ASES[name]]
        elif name.startswith("mips") or name.startswith("arch=mips"):
            self.error(f"cannot change ISA from nanoMIPS to {name.split('=')[-1]}")
        elif name.startswith("arch="):
            arch = name[5:]
            table = {"32r6": ASE["xNMS"] | ASE["TLB"], "32r6s": 0,
                     "i7200": ASE["xNMS"] | ASE["TLB"] | ASE["DSP"] | ASE["EVA"] | ASE["MT"],
                     "nms1": 0}
            if arch not in table:
                self.error(f"unknown architecture {arch}")
            o.ase = table[arch]
        elif name == "insn32":
            o.insn32 = True
        elif name == "noinsn32":
            o.insn32 = False
        elif name.startswith("mcmodel="):
            if name[8:] not in ("auto", "medium", "large"):
                self.error(f"invalid memory model setting {name}")
            o.mc_model = name[8:]
        elif name.startswith("pic") or name == "nopic" or name in ("pid", "nopid"):
            if name not in ("nopic", "pic0", "nopid"):
                self.error(f".set {name}: position-independent code is not modelled")
        elif name == "pcrel":
            o.pcrel = True
        elif name == "nopcrel":
            o.pcrel = False
        else:
            return name in ("oddspreg", "nomips16", "noMIPS-16", "nomicromips", "nobopt",
                            "noforbidden-slots", "move", "nomove", "novolatile", "volatile")
        return True

    # data

    def _auto_align(self, power: int) -> None:
        if power > 0 and self.auto_align:
            self.nanomips_align(power, None)

    def cons(self, size: int, rest: str, auto: bool = True) -> None:
        """s_cons / cons: integers or address expressions of ``size`` bytes."""
        self.flush_pending_output_keep_labels()
        if auto:
            self._auto_align(size.bit_length() - 1)
        for part in _split_args(rest):
            if not part:
                continue
            e, j = _Expressions(self).parse(part, 0)
            if part[j:].strip():
                self.error(f"junk at end of line: `{part[j:].strip()}'")
            self.emit_value(e, size)
        self.labels = []

    def emit_value(self, e: Expr, size: int) -> None:
        order = "big" if self.big else "little"
        if e.op == "constant":
            v = e.num & ((1 << (8 * size)) - 1)
            self.frag_more(v.to_bytes(size, order))
        elif e.op in ("symbol", "subtract"):
            if size not in (1, 2, 4):
                self.error(f"cannot represent a {size}-byte relocation")
            where = self.frag_more(bytes(size))
            r = {1: "8", 2: "16", 4: "32"}[size]
            # nanomips_parse_cons_expression: with .linkrelax an address
            # (a difference only of code labels) is written as explicit
            # relocations -- a plain label difference is left to write.c
            if (self.linkrelax and (e.op != "subtract" or _in_code(e.sym) or _in_code(e.sub))
                    and not (e.op == "subtract" and e.plain and e.num == 0)):
                self.cons_fix_new(where, size, e)
                return
            self.fixups.append(Fix(self.frag, where, size, e.copy(), r, self.line))
        elif e.op == "big":
            self.error("bignum invalid")
        else:
            self.error("illegal operand")

    def cons_fix_new(self, where: int, size: int, e: Expr) -> None:
        """nanomips_cons_fix_new: ``sym + n`` -> one relocation, ``a - b + n``
        -> R_NANOMIPS_NEG against ``b`` (addend n) then one against ``a``."""
        r = {1: "UNSIGNED_8", 2: "UNSIGNED_16", 4: "32"}[size]
        f = self.frag
        fixes = []
        if e.op == "subtract":
            left, right = e.orig if e.orig else (e.sym, e.sub)
            if left.equated is not None:
                self.error("Expression too complex to relocate!")
            sub, addend = right, e.num
            while sub.equated is not None:          # an equate's own expression wins
                sub, n = sub.equated
                addend = -n
            fixes.append(Fix(f, where, size, Expr("symbol", addend, sub), "NEG", self.line))
            fixes.append(Fix(f, where, size, Expr("symbol", 0, left), r, self.line))
        else:
            sym, addend = e.sym, e.num
            while sym.equated is not None:
                sym, addend = sym.equated
            fixes.append(Fix(f, where, size, Expr("symbol", addend, sym), r, self.line))
        for fx in fixes:
            self.fixups.append(fx)
            if f.first_fix is None:
                f.first_fix = fx

    def float_cons(self, d: str, rest: str) -> None:
        self.flush_pending_output_keep_labels()
        if self.auto_align:
            self.nanomips_align(3 if d == ".double" else 2, None)
        fmt = (">" if self.big else "<") + ("d" if d == ".double" else "f")
        for part in _split_args(rest):
            if part:
                try:
                    v = float(part)
                except ValueError:
                    self.error(f"bad floating-point constant `{part}'")
                self.frag_more(struct.pack(fmt, v))
        self.labels = []

    def stringer(self, rest: str, zero: bool) -> None:
        self.flush_pending_output_keep_labels()
        for part in _split_args(rest):
            part = part.strip()
            if not (len(part) >= 2 and part[0] == '"' and part[-1] == '"'):
                self.error("expected a string in quotes")
            self.frag_more(_unescape_gas(part[1:-1]) + (b"\0" if zero else b""))
        self.labels = []

    def space(self, n: int, fill: int) -> None:
        """s_space: an rs_fill frag of ``n`` copies of the fill byte."""
        self.flush_pending_output_keep_labels()
        if n < 0:
            self.warn(".space repeat count is negative, ignored")
            return
        if n == 0:
            return
        f = self.frag
        f.repeat = n
        f.pattern = bytes([fill & 0xFF])
        self.frag_close()
        self.labels = []

    def s_align(self, rest: str) -> None:
        """tc-nanomips.c s_align: ``.align N[, fill]`` (2**N)."""
        parts = _split_args(rest)
        temp = self.absolute(parts[0]) if parts and parts[0] else 0
        if temp > 28:
            self.warn(f"alignment too large, {28} assumed")
            temp = 28
        elif temp < 0:
            self.warn("alignment negative, 0 assumed")
            temp = 0
        fill = self.absolute(parts[1]) if len(parts) > 1 and parts[1] else None
        if temp:
            if self.opts.linkrelax and self.section.code and temp > 1:
                self.create_align_relocs(temp, fill or 0, 0 if fill is None else 1, 0)
            self.auto_align = True
            self.nanomips_align(temp, fill)
        else:
            self.auto_align = False

    def create_align_relocs(self, power: int, fill: int, fill_length: int,
                            max_fill: int) -> None:
        """create_align_relocs: an R_NANOMIPS_ALIGN at the padding's start
        (its absolute symbol: the power of two; its size, set when the
        padding is laid out: the bytes of padding), R_NANOMIPS_FILL /
        R_NANOMIPS_MAX for a fill value / a maximum."""
        f = self.frag
        where = len(f.fix)
        name = f"__reloc_align_\2_{self.align_symidx}"
        self.align_symidx += 1
        fx = Fix(f, where, 0, Expr("absent"), "ALIGN", self.line, place=(name, power, 0))
        self.fixups.append(fx)
        if f.first_fix is None:
            f.first_fix = fx
        f.align_fix = fx
        f.link_var = True
        f.relax_sop = f.relax_sink = power - 2
        if fill_length:
            self.fixups.append(Fix(f, where, 0, Expr("absent"), "FILL", self.line,
                                   place=(f"__reloc_fill_\2_{fill:x}", fill, fill_length)))
        if max_fill:
            self.fixups.append(Fix(f, where, 0, Expr("absent"), "MAX", self.line,
                                   place=(f"__reloc_max_\2_{max_fill:x}", max_fill, 0)))

    def nanomips_align(self, to: int, fill) -> None:
        labels = list(self.labels)
        self.flush_pending_output()
        self.do_align(to, fill, 0)
        self.toggle_point(self.frag, len(self.frag.fix))       # nanomips_move_labels
        for s in labels:                       # (once: the list is cleared)
            s.frag, s.where = self.frag, len(self.frag.fix)

    def do_align(self, power: int, fill, max_skip: int) -> None:
        """frag_align / frag_align_code."""
        f = self.frag
        f.kind = "align_code" if fill is None and self.section.code else "align"
        f.power = power
        f.max_skip = max_skip
        f.fill = bytes([(fill or 0) & 0xFF])
        f.line = self.line
        self.frag_close()
        self.record_alignment(power)

    def flush_pending_output_keep_labels(self) -> None:
        self.history = None


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


def _unescape_gas(body: str) -> bytes:
    """A string literal's bytes (read.c next_char_of_string: \\n \\t ...,
    octal \\NNN, hex \\xHH)."""
    out, i = bytearray(), 0
    simple = {"n": 10, "t": 9, "r": 13, "b": 8, "f": 12, "\\": 92, '"': 34, "'": 39, "v": 11,
              "a": 7, "e": 27}
    while i < len(body):
        ch = body[i]
        if ch != "\\" or i + 1 >= len(body):
            out += ch.encode("latin-1")
            i += 1
            continue
        c = body[i + 1]
        if c in simple:
            out.append(simple[c])
            i += 2
        elif c in "01234567":
            j = i + 1
            v = 0
            while j < len(body) and j < i + 4 and body[j] in "01234567":
                v = v * 8 + int(body[j])
                j += 1
            out.append(v & 0xFF)
            i = j
        elif c in "xX":
            j = i + 2
            v = 0
            while j < len(body) and body[j] in "0123456789abcdefABCDEF":
                v = v * 16 + int(body[j], 16)
                j += 1
            out.append(v & 0xFF)
            i = j
        else:
            out += c.encode("latin-1")
            i += 2
    return bytes(out)


#: a label at the start of a statement; ``name = value``
_LABEL_DEF = re.compile(r"\s*([A-Za-z_.$][\w.$]*|\.L\d+\.\d+)\s*:(?!=)")
_ASSIGN = re.compile(r"\s*([A-Za-z_.$][\w.$]*)\s*=(?!=)")


_NOP16, _NOP32 = 0x9008, 0x8000C000
#: md_convert_frag: the 32-bit form of a 16-bit branch's registers
_TO32 = (16, 17, 18, 19, 4, 5, 6, 7)


def _relax_align(address: int, power: int) -> int:
    mask = (1 << power) - 1
    return ((address + mask) & ~mask) - address


class _Link:
    """write.c's relaxation and frag conversion, then the link (gold)."""

    # -- relaxation (relax_segment / md_estimate_size_before_relax / nanomips_relax_frag)

    def _relax_all(self) -> None:
        passes = 0
        while True:
            changed = False
            for sec in self.sections.values():
                if self._relax_segment(sec):
                    changed = True
            passes += 1
            if not changed or passes > 1000:
                break

    def _relax_segment(self, sec: Section) -> bool:
        frags = sec.frags
        address = 0
        for f in frags:
            f.address = address
            address += len(f.fix)
            if f.kind == "fill":
                address += f.repeat * len(f.pattern)
            elif f.kind in ("align", "align_code"):
                off = _relax_align(address, f.power)
                if f.max_skip and off > f.max_skip:
                    off = 0
                address += off
            elif f.kind == "md":
                address += self._estimate(f, sec)
        iterations = max(len(frags) * len(frags), len(frags))
        while True:
            stretch = 0
            stretched = False
            for f in frags:
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
                elif f.kind == "md":
                    growth = self._relax_frag(f, sec)
                if growth:
                    stretch += growth
                    stretched = True
            iterations -= 1
            if not stretched or iterations <= 0:
                break
        if stretched:
            raise AsmError(0, "Infinite loop encountered whilst attempting to compute the "
                           f"addresses of symbols in section {sec.name}")
        ret = False
        for f in frags:
            if getattr(f, "last_address", None) != f.address:
                f.last_address = f.address
                ret = True
        return ret

    def _estimate(self, f: Frag, sec: Section) -> int:
        if f.subtype & 0xE0000000 == 0x20000000:
            length = f.var
            t = _md_type(f.subtype)
            if t == RT_ADDIU:
                length = self._addiu_length(f, False)
            elif t >= RT_BRANCH_UCND:
                length = self._branch_length(f, sec, False)
            f.var = length
            return length
        if self.need_relax(f.symbol, False):
            f.use_second = True
            f.var = -f.first
        else:
            f.var = -f.second
        return f.var

    def _relax_frag(self, f: Frag, sec: Section) -> int:
        if f.subtype & 0xE0000000 != 0x20000000:
            return 0
        old = f.var
        t = _md_type(f.subtype)
        if t == RT_ADDIU:
            new = self._addiu_length(f, True)
        elif t >= RT_BRANCH_UCND:
            new = self._branch_length(f, sec, True)
        else:
            new = old
        f.var = new
        if t != RT_ADDIU and f.symbol is not None:
            self.mark_if_invariable_branch(f, sec)
        return new - old

    # -- link-time variability (mark_if_invariable_branch, variable_frag_p ...) ------

    def mark_if_invariable_branch(self, f: Frag, sec: Section) -> None:
        """Mark a branch frag link-time FIXED / VARIABLE (RF_*) from the
        frags between it and its target and the bytes the linker may
        add / remove there."""
        st = f.subtype
        sym = f.symbol
        base, _ = sym.resolve()
        if (sym.external or sym.weak or base.segment() is not sec or st & TOOFAR_FINAL):
            if _md_rf(st) != RF_NORELAX:
                f.subtype = (st & ~0x18000) | RF_NORELAX << 15
                f.link_var = True
            return
        addr = self._sym_value(sym)
        max_off = addr - f.address - len(f.fix)
        if _md_type(st) >= RT_BRANCH_UCND and not st & TOOFAR16:
            max_off -= 2
        else:
            max_off -= f.var
        min_off = max_off
        variable = False
        frags = sec.frags
        i = frags.index(f)
        if f.address <= addr <= f.address + len(f.fix):
            variable = f.link_var
        elif addr > f.address + len(f.fix):
            target = base.frag
            stop = frags.index(target) + 1
            for g in frags[i + 1:stop]:
                if g.link_var and addr > g.address:
                    max_off += g.relax_sop
                    min_off -= g.relax_sink
                    variable = True
                elif (g.subtype & 0xE0000000 == 0x20000000 and _md_rf(g.subtype) == RF_VARIABLE
                      and addr > g.address + len(g.fix)):
                    max_off += self.subtype_sop(g.subtype)
                    min_off -= g.var
                    variable = True
        else:
            j = frags.index(base.frag)
            while j < len(frags) and frags[j] is not f:
                g = frags[j]
                if g.link_var:
                    max_off -= g.relax_sop
                    min_off += g.relax_sink
                    variable = True
                elif g.subtype & 0xE0000000 == 0x20000000 and _md_rf(g.subtype) == RF_VARIABLE:
                    max_off -= self.subtype_sop(g.subtype)
                    min_off += g.var
                    variable = True
                j += 1
            if j < len(frags) and frags[j].link_var:
                max_off -= frags[j].relax_sop
                min_off += frags[j].relax_sink
                variable = True
        fixed = not variable and self.branch_in_range(st, max_off)
        variable = variable and not self.branch_in_range(st, max_off)
        if _md_type(st) >= RT_BRANCH_UCND and st & TOOFAR16 and not variable:
            variable = self.branch_in_range(st & ~TOOFAR16, min_off)
        if fixed:
            f.subtype = (st & ~0x18000) | RF_FIXED << 15
        elif variable:
            f.subtype = (st & ~0x18000) | RF_VARIABLE << 15
        else:
            f.subtype = st & ~0x18000

    @staticmethod
    def branch_in_range(subtype: int, offset: int) -> bool:
        r = _rt_reloc(_md_type(subtype), bool(subtype & TOOFAR16))
        if r == "4_PCREL_S1":
            lo, hi = 2, 30
        elif r is not None:
            hi = (1 << HOWTOS[BFD_TO_R[r]]["bitsize"]) - 1
            lo = -hi - 1
        else:
            lo = hi = 0
        return lo <= offset <= hi

    @staticmethod
    def variable_frag_p(f: Frag) -> bool:
        return f.link_var or (f.subtype & 0xE0000000 == 0x20000000
                              and _md_rf(f.subtype) == RF_VARIABLE)

    def _fr_fix(self, f: Frag) -> int:
        """fr_fix: the current frag's is 0 until it is closed."""
        return 0 if f is self.frag and not self.relaxed else len(f.fix)

    def part_fixed_frag_p(self, f: Frag, offset: int) -> bool:
        if offset > f.address + self._fr_fix(f):
            return False
        return (not f.link_var or f.first_fix is None
                or offset <= f.first_fix.where + f.address)

    def allow_local_subtract_symbols(self, left: Sym, right: Sym, pre_relax: bool) -> bool:
        """nanomips_allow_local_subtract_symbols: may gas compute
        ``left - right`` itself (nothing the linker changes lies between)?"""
        lf, rf = left.frag, right.frag
        if lf is rf:
            high, low = left.value(), right.value()
            if high < low:
                high, low = low, high
            if lf.first_fix is not None:
                chain = [fx for fx in self.fixups if fx.frag.section is lf.section]
                k = next(n for n, fx in enumerate(chain) if fx is lf.first_fix)
                while k < len(chain) and (chain[k].frag.address + chain[k].where < low
                                          and chain[k].frag.address + chain[k].where < high):
                    k += 1
                if k < len(chain) and chain[k].frag.address + chain[k].where < high:
                    return False
            return not pre_relax
        if left.segment() is right.segment():
            if left.value() > right.value():
                fragp, ifragp = lf, rf
                fixed_final = self.part_fixed_frag_p(fragp, left.value())
            else:
                fragp, ifragp = rf, lf
                fixed_final = self.part_fixed_frag_p(fragp, right.value())
            frags = lf.section.frags
            j = frags.index(ifragp)
            while j < len(frags) and frags[j] is not fragp and not self.variable_frag_p(frags[j]):
                j += 1
            return (j < len(frags) and frags[j] is fragp
                    and (fixed_final or not self.variable_frag_p(frags[j])))
        return False

    def _sym_value(self, sym: Sym) -> int:
        """S_GET_VALUE during relaxation (section-relative)."""
        e = getattr(sym, "expr", None)
        if e is None:
            return sym.value()
        if e.op == "subtract":
            if e.sym.segment() is not e.sub.segment():
                raise AsmError(self.line, "can't resolve the difference of symbols in "
                               "different sections")
            return e.sym.value() - e.sub.value() + e.num
        return e.sym.value() + e.num if e.op == "symbol" else e.num

    def _branch_length(self, f: Frag, sec: Section, update: bool) -> int:
        """relaxed_16bit_branch_length."""
        st = f.subtype
        if st & TOOFAR_FINAL:
            return f.var
        sym = f.symbol
        t = _md_type(st)
        if sym is not None and sym.defined and not sym.weak and sym.segment() is sec:
            val = self._sym_value(sym) + f.offset
            val -= f.address + len(f.fix) + f.var
            if t == RT_BRANCH_UCND:
                toofar = val < -(0x200 << 1) or val >= (0x200 << 1)
            elif t == RT_BRANCH_CNDZ:
                toofar = val < -(0x40 << 1) or val >= (0x40 << 1)
            elif t == RT_BRANCH_CND:
                toofar = (val <= 0 or val > 30
                          or (val == 2 and st & TOOFAR16 and self._frag_match(f, sym)))
            else:
                toofar = True
            if update and toofar and f.var == 2:
                if ((t == RT_BRANCH_UCND and val in (-0x402, 0x400))
                        or (t == RT_BRANCH_CNDZ and val in (-0x82, 0x80))
                        or (t == RT_BRANCH_CND and val == 32)):
                    f.subtype = st = st | TOOFAR16 | TOOFAR_FINAL
        else:
            toofar = True
        if update and toofar != bool(st & TOOFAR16):
            f.subtype = st | TOOFAR16 if toofar else st & ~TOOFAR16
        return 4 if toofar else 2

    def _frag_match(self, head: Frag, sym: Sym) -> bool:
        """nanomips_frag_match: the target is the start of the next frag."""
        s, _ = sym.resolve()
        target = s.frag
        frags = head.section.frags
        i = frags.index(head)
        nxt = frags[i + 1] if i + 1 < len(frags) else None
        if target is nxt:
            return self._sym_value(sym) == target.address
        j = i + 1
        while j < len(frags):
            h = frags[j]
            if h is target:
                break
            empty = (len(h.fix) == 0 and (h.kind == "fill" and not h.repeat
                                          or h.kind == "md" and h.var == 0))
            if not (empty or h.kind == "align_code"):
                break
            if target.line > 0 and h.line > target.line:
                break
            j += 1
        return j < len(frags) and frags[j] is target

    def _addiu_length(self, f: Frag, update: bool) -> int:
        """relaxed_addiu_length."""
        st = f.subtype
        sval = self._sym_value(f.symbol) + f.offset
        insn = int.from_bytes(self._halfwords(f.literal[:4]), "big")
        rt, rs = (insn >> 21) & 0x1F, (insn >> 16) & 0x1F
        fixed = _md_rf(st) == RF_FIXED
        toofar16, toofar32, negoff = True, False, False
        gpr3 = lambda r: r >> 2 in (1, 4)            # noqa: E731
        if rt == rs and -8 <= sval <= 7 and not fixed:
            toofar16, negoff = False, True
        elif gpr3(rs) and gpr3(rt) and 0 <= sval <= 28 and sval % 4 == 0 and not fixed:
            toofar16 = False
        elif gpr3(rt) and rs == 0 and -1 <= _s32(sval) <= 126 and not fixed:
            toofar16 = False
        else:
            negoff = -4096 < sval < 0
        if (toofar16 and not fixed and (sval > 0xFFFF or sval < -4096)
                and (rs == rt or rs == 0) and self.opts.ase & ASE["xNMS"]):
            toofar32 = True
        if update:
            st = (st | TOOFAR16) if toofar16 else (st & ~TOOFAR16)
            st = (st | TOOFAR32) if toofar32 else (st & ~TOOFAR32)
            st = (st | NEGOFF) if negoff else (st & ~NEGOFF)
            f.subtype = st
        return 6 if toofar32 else 4 if toofar16 else 2

    def _halfwords(self, data: bytes) -> bytes:
        """The bytes of an instruction as its big-endian value."""
        if self.big:
            return bytes(data)
        out = bytearray()
        for i in range(0, len(data), 2):
            out += data[i:i + 2][::-1]
        return bytes(out)

    # -- conversion (md_convert_frag, cvt_frag_to_fill, nanomips_handle_align) ------------

    def _convert(self, sec: Section) -> None:
        frags = sec.frags
        for i, f in enumerate(frags):
            nxt = frags[i + 1].address if i + 1 < len(frags) else None
            if f.kind == "md":
                if f.subtype & 0xE0000000 == 0x20000000:
                    self._convert_md(f)
                else:
                    self._convert_seq(f)
            elif f.kind in ("align", "align_code"):
                end = nxt if nxt is not None else f.address + len(f.fix) + _relax_align(
                    f.address + len(f.fix), f.power)
                pad = end - f.address - len(f.fix)
                if f.align_fix is not None:
                    name, power, _ = f.align_fix.place
                    f.align_fix.place = (name, power, pad)
                if f.kind == "align_code":
                    excess = pad % 4
                    lead = b""
                    if excess == 3:
                        lead = b"\0" + (self._bytes(_NOP16, 2) if not self.opts.insn32
                                        else b"\0\0")
                    elif excess == 2:
                        lead = self._bytes(_NOP16, 2) if not self.opts.insn32 else b"\0\0"
                    elif excess == 1:
                        lead = b"\0"
                    at = len(f.fix)
                    f.fix += lead
                    if excess >= 2 and not self.opts.insn32:
                        f.nops.append((at + excess - 2, 2))
                    for k in range((pad - excess) // 4):
                        f.nops.append((len(f.fix), 4))
                        f.fix += self._bytes(_NOP32, 4)
                else:
                    f.fix += f.fill * pad
                f.kind = "fill"
            elif f.kind == "fill" and f.repeat:
                f.fix += f.pattern * f.repeat
                f.repeat = 0
            f.var = 0

    def _convert_seq(self, f: Frag) -> None:
        """The unused alternative of a relax_start / relax_switch macro goes."""
        first, second = f.first, f.second
        end = len(f.fix)
        start = end - first - second
        if f.use_second:
            del f.fix[start:start + first]
        else:
            del f.fix[end - second:end]
        for fx in self.fixups:
            if fx.frag is f and fx.seq:
                if fx.seq == 1:
                    fx.done = fx.done or f.use_second
                elif f.use_second:
                    fx.where -= first
                else:
                    fx.done = True
        lines = []
        for where, size, line, seq in f.lines:
            if seq == 1 and f.use_second or seq == 2 and not f.use_second:
                continue
            lines.append((where - first if seq == 2 else where, size, line, 0))
        f.lines = lines
        f.kind = "fill"

    def _convert_md(self, f: Frag) -> None:
        st = f.subtype
        t = _md_type(st)
        base = len(f.fix)
        if t == RT_ADDIU:
            insn = int.from_bytes(self._halfwords(f.literal[:4]), "big")
            rt, rs = (insn >> 21) & 0x1F, (insn >> 16) & 0x1F
            sval = self._sym_value(f.symbol) + f.offset
            if not st & TOOFAR16 and not st & TOOFAR32:
                if st & NEGOFF:                    # ADDIU[32] -> ADDIU[RS5]
                    insn = 0x9008 | (rt << 5) | (sval & 0x8) << 1 | (sval & 0x7)
                elif rs == 0:                      # LI[32] -> LI[16]
                    insn = 0xD000 | (rt << 7) | (sval & 0x7F)
                else:                              # ADDIU[32] -> ADDIU[R2]
                    insn = 0x9000 | (rt & 7) << 7 | (rs & 7) << 4 | (sval & 0x1F) >> 2
                f.fix += self._bytes(insn & 0xFFFF, 2)
            elif st & TOOFAR32:
                op = 0x6001 | rt << 5 if rs == rt else 0x6000 | rt << 5
                f.fix += self._bytes(op, 2)
                f.fix += self._bytes((((sval >> 16) & 0xFFFF) | (sval << 16)) & M32, 4)
            else:
                if st & NEGOFF:                    # ADDIU[32] -> ADDIU[NEG]
                    insn = (insn & ~0xFFF) | 0x80008000 | (-sval & 0xFFF)
                else:
                    insn = (insn & ~0xFFFF) | (sval & 0xFFFF)
                    if sval & ~0xFFFF:
                        self.error_at(f.line, f"Operand 3 out of range `addiu $r{rt},$r{rs},"
                                      f"{sval}")
                f.fix += self._bytes(insn & M32, 4)
            self._resize_line(f, base, len(f.fix) - base)
            return
        if t < RT_BRANCH_UCND:                    # a fixed-size branch with its relocation
            f.fix += f.literal
            return
        toofar = bool(st & TOOFAR16)
        rtype = ("10_PCREL_S1", "7_PCREL_S1", "4_PCREL_S1", "25_PCREL_S1", "14_PCREL_S1",
                 "14_PCREL_S1")[t - RT_BRANCH_UCND + (3 if toofar else 0)]
        fx = Fix(f, base, 2, Expr("symbol", f.offset, f.symbol), rtype, f.line)
        fx.insn_size = 4 if toofar else 2
        self.fixups.append(fx)
        if f.first_fix is None:
            f.first_fix = fx
        insn = int.from_bytes(self._halfwords(f.literal[:2]), "big")
        if not toofar:
            f.fix += f.literal[:2]
            return
        if insn & 0xFC00 == 0xD800:                # beqc[16] / bnec[16]
            rt, rs = (insn & 0x0380) >> 7, (insn & 0x0070) >> 4
            new = 0x88000000 if rt > rs else 0xA8000000
            new |= _TO32[rt] << 21 | _TO32[rs] << 16
        elif insn & 0xDC00 == 0x9800:              # beqzc[16] / bnezc[16]
            rt = _TO32[(insn & 0x0380) >> 7]
            new = 0x88000000 | (((insn & 0x2000) >> 13) << 29) | rt << 21
        elif insn & 0xDC00 == 0x1800:              # bc[16] / balc[16]
            new = 0x28000000 | (((insn & 0x2000) >> 13) << 25)
        else:
            raise AssertionError(f"branch 0x{insn:04x}")
        f.fix += self._bytes(new, 4)
        self._resize_line(f, base, 4)

    def _resize_line(self, f: Frag, where: int, size: int) -> None:
        f.lines = [(w, size if w == where else s, line, seq) for w, s, line, seq in f.lines]

    def error_at(self, line: int, msg: str):
        raise AsmError(line, msg)

    # -- the object (write.c: fixup_segment, tc_gen_reloc; nanomips_sort_relocs_final) -----

    def _object(self) -> nanold.Object:
        """What ``nanomips-elf-as`` writes: each section's bytes, its
        relocations (code sections' sorted by offset, the placeholders after
        the instruction's own), the symbols; ``EF_NANOMIPS_LINKRELAX`` and
        the .nanoMIPS.abiflags ASEs."""
        obj = nanold.Object(big=self.big, linkrelax=self.linkrelax,
                            nmf=bool(self.file_opts.ase & ASE["xNMS"]))
        self._osec: dict[int, nanold.ObjSection] = {}
        self._osym: dict[int, nanold.Symbol] = {}
        for sec in self.sections.values():
            data = bytearray()
            o = nanold.ObjSection(sec.name, data, sec.align, sec.code)
            for f in sec.frags:
                assert f.address == len(data), (sec.name, f.address, len(data))
                data += f.fix
                for where, size, line, _seq in f.lines:
                    o.entries.append(nanold.Entry(f.address + where, size, line))
                for where, size in f.nops:
                    o.entries.append(nanold.Entry(f.address + where, size, f.line, True))
            o.entries.sort(key=lambda e: e.offset)
            o.section_symbol = nanold.Symbol(sec.name, o, 0, is_section=True)
            self._osec[id(sec)] = o
            obj.sections.append(o)
        for name, s in self.symbols.items():
            if s.defined:
                base, add = s.resolve()
                if base.frag is not None:
                    o = self._osec[id(base.section)]
                    sym = nanold.Symbol(name, o, base.frag.address + base.where + add,
                                        is_global=s.external or s.weak)
                    o.symbols.append(sym)
                else:
                    sym = nanold.Symbol(name, None, base.absolute + add)
                self._osym[id(s)] = sym
        gp = self.symbols.get("_gp")
        if gp is not None and gp.defined:
            obj.gp_symbol = self._osym[id(gp)]
        for fx in self.fixups:
            if fx.done:
                continue
            o = self._osec[id(fx.frag.section)]
            off = fx.frag.address + fx.where
            for r in self.gen_reloc(fx, o, off):
                o.relocs.append(r)
        for o in obj.sections:
            if o.code:
                o.relocs.sort(key=_reloc_order)
        return obj

    def _label(self, s: Sym) -> nanold.Symbol:
        """The object's symbol for a (resolved) label."""
        sym = self._osym.get(id(s))
        if sym is None:
            if s.frag is not None:
                o = self._osec[id(s.section)]
                sym = nanold.Symbol(s.name, o, s.frag.address + s.where)
                o.symbols.append(sym)
            elif s.name == "_gp":
                sym = nanold.Symbol("_gp", None, 0, script=True)
            else:
                sym = nanold.Symbol(s.name, None, 0, defined=False)
            self._osym[id(s)] = sym
        return sym

    def gen_reloc(self, fx: Fix, o: nanold.ObjSection, off: int) -> list:
        """fixup_segment + md_apply_fix + tc_gen_reloc for one fixup: the
        relocations it leaves (or none: gas applied it itself)."""
        r = fx.reloc
        if r in ("ALIGN", "FILL", "MAX"):
            name, value, size = fx.place
            if value == 0:              # BFD writes an absolute 0 as symbol 0 (STN_UNDEF)
                return [nanold.Reloc(off, r, None, 0, fx.line)]
            return [nanold.Reloc(off, r, nanold.Symbol(name, None, value, size), 0, fx.line)]
        if r in ("RELAX", "NORELAX", "INSN16", "INSN32", "FIXED"):
            return [nanold.Reloc(off, r, None, 0, fx.line)]
        e = fx.expr
        if e.op == "subtract":
            add, sub = e.sym, e.sub
            if add.segment() is sub.segment() and isinstance(add.segment(), Section):
                forced = (self.linkrelax and add.section.code and sub.section.code
                          and not self.allow_local_subtract_symbols(add, sub, False))
                if not forced:              # write.c resolves it
                    value = add.value() - sub.value() + e.num
                    return self._apply_now(o, off, fx, value)
                return [nanold.Reloc(off, "NEG", self._label(sub), 0, fx.line),
                        nanold.Reloc(off, self._r(fx), self._label(add), 0, fx.line)]
            if sub.segment() == "abs":
                return self.gen_reloc(Fix(fx.frag, fx.where, fx.size, Expr(
                    "symbol", e.num - sub.value(), add), fx.reloc, fx.line), o, off)
            self.error_at(fx.line, f"can't resolve `{add.name}' {{{_segname(add)} section}} - "
                          f"`{sub.name}' {{{_segname(sub)} section}}")
        if e.op in ("symbol",):
            base, add = e.sym.resolve()
            add += e.num
            if base.absolute is not None and not self._forced_abs(r):
                return self._apply_now(o, off, fx, base.absolute + add)
            if base.absolute is not None:
                return [nanold.Reloc(off, self._r(fx), None, base.absolute + add, fx.line)]
            if (base.frag is not None and not base.external and not base.weak
                    and self.fix_adjustable(fx, base)):
                # adjust_reloc_syms: against the section's symbol + the offset
                o2 = self._osec[id(base.section)]
                return [nanold.Reloc(off, self._r(fx), o2.section_symbol,
                                     base.frag.address + base.where + add, fx.line)]
            return [nanold.Reloc(off, self._r(fx), self._label(base), add, fx.line)]
        if self._forced_abs(r):             # a constant PC-relative: against *ABS*
            return [nanold.Reloc(off, self._r(fx), None, e.num, fx.line)]
        return self._apply_now(o, off, fx, e.num)

    def fix_adjustable(self, fx: Fix, base: Sym) -> bool:
        """nanomips_fix_adjustable: PC-relative relocations and (with
        .linkrelax, or a used symbol) those against code stay against the
        symbol, so the linker can move code without changing addends."""
        if _pcrel(fx.reloc):
            return False
        return not base.section.code       # (a code symbol a relocation uses is used)

    def _r(self, fx: Fix) -> str:
        """The R_NANOMIPS_* type (gold's name) of a fixup's BFD code."""
        r = BFD_TO_R.get(fx.reloc)
        if r is None or r not in nanold.RELOCS and r not in _GOLD_NAMES:
            self.error_at(fx.line, f"cannot represent BFD_RELOC_{fx.reloc} relocation in this "
                          "object file format")
        return _GOLD_NAMES.get(r, r)

    @staticmethod
    def _forced_abs(r: str) -> bool:
        """nanomips_force_relocation: kept against an absolute symbol too."""
        return _pcrel_branch(r) or r in ("PCREL_HI20", "PC_I32", "GOTPC_HI20", "GOTPC_I32",
                                         "GOT_LO12")

    def _apply_now(self, o: nanold.ObjSection, off: int, fx: Fix, value: int) -> list:
        """md_apply_fix of a fixup gas resolves: the value into the field."""
        if fx.reloc in ("8", "16", "32") or BFD_TO_R.get(fx.reloc) in (
                "UNSIGNED_8", "UNSIGNED_16", "SIGNED_8", "SIGNED_16", "32", "NEG"):
            size = fx.size
            if size < 8:
                mask = (-1 << (size * 8)) & ((1 << 64) - 1)
                v = value & ((1 << 64) - 1)
                if v & mask not in (0, mask):
                    self.error_at(fx.line, f"value of {_sx64(value)} too large for field of "
                                  f"{size} bytes at {fx.frag.address + fx.where:#x}")
            o.data[off:off + size] = (value & ((1 << (8 * size)) - 1)).to_bytes(
                size, "big" if self.big else "little")
            return []
        r = self._r(fx)
        if r.startswith("GPREL") or r.startswith("GOT"):
            self.error_at(fx.line, "unsupported constant in relocation")
        nanold.Linker(nanold.Object(self.big), self.bases).apply(o.data, off, r, value & M32)
        return []

    def write_object(self, source: str) -> nanold.Object:
        """Assemble ``source`` only (``nanomips-elf-as``): the object the
        linker gets."""
        self._run(source)
        return self._gas_end()

    def _gas_end(self) -> nanold.Object:
        """write.c: relax_segment, md_convert_frag, fixup_segment; the object."""
        for name, s in self.symbols.items():
            if not s.defined and getattr(s, "expr", None) is None and name != "_gp":
                used = any(_uses(fx.expr, s) for fx in self.fixups) or any(
                    f.symbol is s for sec in self.sections.values() for f in sec.frags)
                if used:
                    raise AsmError(s.line or self.line, f"undefined reference to `{name}'")
        # subsegs_finish_section: a code section ends aligned to its own
        # alignment (padded with NOPs), a data section with an empty .align 0
        for sec in self.sections.values():
            self.section, self.frag = sec, sec.frags[-1]
            if sec.code:
                f = self.frag
                f.kind, f.power, f.max_skip, f.line = "align_code", sec.align, 0, 0
                self.frag_close()
        self.relaxed = True
        self._relax_all()
        for sec in self.sections.values():
            for f in sec.frags:
                f.nops = []
        for sec in self.sections.values():
            self._convert(sec)
        return self._object()

    def _finish(self) -> Program:
        obj = self._gas_end()
        try:
            linker = nanold.link(obj, self.bases, self.link_options)
        except nanold.LinkError as exc:
            raise AsmError(exc.line, exc.message) from None
        prog = Program(endian=self.endian, release="R6", arch="nanomips")
        prog.nm = {}
        for o in obj.sections:
            for i, b in enumerate(o.data):
                prog.image[o.address + i] = b
            for e in o.entries:
                a = o.address + e.offset
                prog.nm[a] = e.size
                if e.nop:
                    prog.source_map.setdefault(a, SourceLine(e.line, self._lines[e.line - 1]
                                                             if e.line else ""))
                else:
                    prog.source_map[a] = SourceLine(e.line, self._lines[e.line - 1])
        for name, s in self.symbols.items():
            if s.defined and not name.startswith("L\037"):
                prog.symbols[name] = linker.value(self._osym[id(s)], 0)
        prog.symbols.setdefault("_gp", linker.gp & M32)
        prog.segments = linker.segments
        for name in ("main", "__start", "_start"):
            if name in prog.symbols:
                prog.entry = prog.symbols[name]
                break
        else:
            prog.entry = self.bases["text"]
        prog.warnings = list(self.warnings)
        prog.link_log = list(linker.log)
        needs = set()
        for a in prog.nm:
            row = prog.nm_insn(a).row
            if row.need is not None:
                needs.add("ginvt" if row.form == "GINVT" else
                          "ginvi" if row.form == "GINVI" else row.need)
        prog.needs = frozenset(needs)
        return prog


#: bfd's names of the R_NANOMIPS_* types that gold's elfcpp names otherwise
_GOLD_NAMES = {"PCHI20": "PC_HI20"}


def _reloc_order(r: nanold.Reloc) -> tuple:
    """nanomips_sort_relocs_compare: by offset; at one offset the
    instruction's relocation (R_NANOMIPS_NEG first), then the placeholders
    by number."""
    prop = nanold.RELOCS[r.type]
    if prop.placeholder:
        return (r.offset, 1, prop.number)
    return (r.offset, 0, 0 if r.type == "NEG" else 1)


def _in_code(s: Sym) -> bool:
    """S_GET_SEGMENT (s)->flags & SEC_CODE (an undefined symbol: no)."""
    seg = s.segment()
    return isinstance(seg, Section) and seg.code


def _segname(s: Sym) -> str:
    seg = s.segment()
    return "*UND*" if seg is None else "*ABS*" if seg == "abs" else seg.name


def _uses(e: Expr, s: Sym) -> bool:
    return e.sym is s or e.sub is s


class Assembler(_Core, _Emit, _Macros, _Directives, _Link):
    __doc__ = _Core.__doc__


#: GNU as's directives eMIPS's nanoMIPS assembler leaves out (an error naming them)
_NOT_MODELLED = frozenset((".insn", ".jumptable", ".cpsetup", ".dtprelword", ".dtpreldword",
                           ".ehword", ".org", ".origin", ".comm", ".lcomm", ".incbin",
                           ".sleb128", ".uleb128", ".octa"))


class _Hooks:
    """What emips.gasread asks the assembler."""

    onechar = True

    def __init__(self, asm: Assembler) -> None:
        self.asm = asm

    def absolute(self, text: str) -> int:
        return self.asm.absolute(text)

    def defined(self, name: str) -> bool:
        s = self.asm.symbols.get(name)
        return s is not None and (s.defined or s.equated is not None)

    def known(self, directive: str) -> bool:
        return (directive in self.asm.KNOWN or directive in self.asm._IGNORED
                or directive in _NOT_MODELLED)

    def strip(self, line: str) -> str:
        return _strip_comment(line)


def assemble(source: str, *, endian: str = "big", bases: dict[str, int] | None = None,
             nms: bool = False, crc: bool = False, ginv: bool = False, eva: bool = False,
             tlb: bool = True, dsp: bool = False, mt: bool = False, pseudo: bool = True,
             reorder: bool = True,
             linkrelax: bool = False,
             link: "nanold.LinkOptions | dict | None" = None, path: str | None = None,
             include_dirs: tuple[str, ...] = ()) -> Program:
    """Assemble and link a nanoMIPS ``source`` (see the module docstring).
    ``nms`` / ``crc`` / ``ginv`` / ``eva`` / ``tlb`` / ``dsp`` / ``mt``: the
    core's options (the instructions GNU as accepts: ``-march=32r6s``,
    ``-mcrc``, ``-mdsp``, ``-mmt`` ...);
    ``pseudo=False``: strict mode (no macros); ``reorder=False``: start in
    ``.set noreorder``; ``linkrelax``: gas ``--linkrelax`` (= ``.linkrelax``:
    the linker may transform the code); ``link``: ld's options
    (``nanold.LinkOptions`` or its fields: ``relax`` = ``--relax`` ...)."""
    if isinstance(link, dict):
        link = nanold.LinkOptions(**link)
    return Assembler(endian=endian, bases=bases, nms=nms, crc=crc, ginv=ginv, eva=eva,
                     tlb=tlb, dsp=dsp, mt=mt, pseudo=pseudo, reorder=reorder,
                     linkrelax=linkrelax,
                     link=link).assemble(source, path, include_dirs)
