"""Two-pass MIPS32 assembler (PLAN M4): ``.asm`` source -> memory image.

Syntax is GNU ``as`` / MARS-compatible (see :mod:`emips.syntax` for every
instruction's operands).  The assembler produces a :class:`Program`: the
bytes of each segment, the symbol table, a **source map** (address ->
source line, so the GUI / CLI can highlight the executing line) and an
entry point; it can load itself into a :class:`emips.sim.Machine` or be
written as Intel HEX (and read back by :mod:`emips.hexload`).

Segments and defaults (MARS / SPIM layout):

=========  ============  =============================================
``.text``  0x00400000    user code
``.data``  0x10010000    user data
``.ktext`` 0x80000180    kernel code — the general exception vector
``.kdata`` 0x90000000    kernel data
=========  ============  =============================================

Directives: ``.text/.data/.ktext/.kdata [addr]``, ``.word``, ``.half``,
``.byte``, ``.float``, ``.double``, ``.ascii``, ``.asciiz``, ``.space``,
``.align``, ``.eqv/.equ NAME, value``, ``.set reorder|noreorder|at|noat``;
``.globl/.global/.ent/.end/.type/.size/.extern`` are accepted and ignored.
``.word``/``.half``/``.float``/``.double`` align themselves (as in MARS
and GNU ``as``).

Expressions: integers (``42``, ``-3``, ``0x2A``, ``0b101``, ``'a'``),
symbols, ``+ - * / << >> & | ( )``, ``%hi(x)`` / ``%lo(x)`` (``%hi`` is
carry-adjusted for the sign-extended ``%lo``).

Pseudo-instructions, expanded exactly as GNU ``as`` does (``$at`` is the
scratch register; using one under ``.set noat`` is an error):
``nop, move, li, la, b, beqz, bnez, blt, bltu, bgt, bgtu, ble, bleu,
bge, bgeu, neg, negu, not``, and loads / stores with a label or a 32-bit
address (``lw $t0, value``).  ``div/divu rs, rt`` is the machine
instruction (MARS / SPIM meaning); GNU ``as`` would expand it into a
divide-by-zero-checking macro.

**Delay slots.**  The machine always executes the instruction after a
branch / jump.  Under ``.set reorder`` (the default, as in GNU ``as``)
the assembler puts a ``nop`` in every delay slot, so code written for
simulators without delay slots (MARS default) still runs correctly; under
``.set noreorder`` the next source instruction *is* the delay slot.
Unlike GNU ``as``, the assembler never moves instructions into the slot.

**Options** (teaching): ``pseudo=False`` is *strict mode* — every
pseudo-instruction above is an error that names the machine instructions
to write instead (``nop``, ``b``, ``bal`` and the R6 ``jr`` stay: they are
encodings of real instructions); ``reorder=False`` starts the source in
``.set noreorder`` (the source can still switch with ``.set``).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Callable

from . import cp0 as C
from .isa import ISA
from .syntax import (
    KINDS,
    kind_of,
    operand_spec,
    parse_fcc,
    parse_fpr,
    parse_gpr,
    parse_numreg,
)

M32 = 0xFFFFFFFF
AT = 1

SEGMENT_BASES = {
    "text": 0x00400000,
    "data": 0x10010000,
    "ktext": 0x80000180,
    "kdata": 0x90000000,
}
#: Register values set by :meth:`Program.load` with ``boot=True`` (MARS).
BOOT_SP = 0x7FFFEFFC
BOOT_GP = 0x10008000


class AsmError(Exception):
    """An assembly error at a source line."""

    def __init__(self, line: int, message: str, text: str = "") -> None:
        super().__init__(f"line {line}: {message}" + (f"  [{text.strip()}]" if text else ""))
        self.line = line
        self.message = message
        self.text = text


@dataclass
class SourceLine:
    line: int
    text: str


@dataclass
class Program:
    """Result of :func:`assemble`."""

    endian: str
    #: MIPS32 release the program was assembled for
    release: str = "R5"
    #: address -> byte
    image: dict[int, int] = field(default_factory=dict)
    #: symbol -> address / value
    symbols: dict[str, int] = field(default_factory=dict)
    #: instruction address -> the source line it came from
    source_map: dict[int, SourceLine] = field(default_factory=dict)
    #: segment name -> (lowest, highest + 1) address used
    segments: dict[str, tuple[int, int]] = field(default_factory=dict)
    entry: int = SEGMENT_BASES["text"]
    #: a compiled program (an ELF, emips.elf): instruction address ->
    #: (source file, line) from its DWARF line table
    c_lines: dict[int, tuple[str, int]] = field(default_factory=dict)
    #: ... (file, line) -> the addresses where that line's statements start
    c_stmts: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    #: ... (start, end, name) of its functions (STT_FUNC symbols)
    functions: list[tuple[int, int, str]] = field(default_factory=list)
    #: MIPS16e code (emips.mips16): instruction address -> its size (2 / 4)
    m16: dict[int, int] = field(default_factory=dict)
    #: microMIPS code (emips.micromips): instruction address -> its size
    mm: dict[int, int] = field(default_factory=dict)

    def word(self, addr: int) -> int:
        data = bytes(self.image.get(addr + i, 0) for i in range(4))
        return struct.unpack(">I" if self.endian == "big" else "<I", data)[0]

    def labels(self) -> dict[int, str]:
        """address -> label (first defined), for the disassembler."""
        out: dict[int, str] = {}
        for name, addr in self.symbols.items():
            if not name.startswith(".L"):          # numeric local labels: no names
                out.setdefault(addr, name)
        return out

    def load(self, machine, *, boot: bool = True) -> None:
        """Copy the image into ``machine`` and point its PC at the entry.

        ``boot=True`` plays the boot firmware: ``Status`` = CU1 only (kernel
        mode, FPU usable, BEV = ERL = 0 so exceptions go to ``EBase +
        0x180`` = 0x80000180, where ``.ktext`` code lives), ``Config.K0 =
        KU = 3`` (kseg0 / kuseg cacheable), ``K23 = 2`` (kseg2 / kseg3
        uncached: devices), the caches initialized (every line invalid:
        what boot ROM code does with Index Store Tag and ``TagLo = 0``),
        ``$sp`` = 0x7FFFEFFC, ``$gp`` = 0x10008000 (MARS defaults).  With
        a TLB (:mod:`emips.tlb`) only ``Config.K0`` exists; the firmware
        initializes the TLB (every entry unique and invalid) and, unless
        the program brings its own (code at 0x80000000-0x800000FF),
        installs its TLB refill handler at EBase + 0x000
        (:data:`emips.tlb.FIRMWARE_REFILL`: kuseg at VA + 1 GB, kseg2 /
        kseg3 at PA = VA, like the fixed mapping).  The image goes where
        that mapping puts it even when the program has its own handler
        (whose page tables then know where the program is).
        ``boot=False`` keeps the hardware reset state (caches undefined,
        TLB not initialised).

        The image is in virtual addresses; every byte is written to the
        physical address the MMU gives it in the state the program starts
        in (with the fixed mapping, kuseg is at VA + 1 GB after boot, at
        VA itself in the reset state, where Status.ERL = 1)."""
        from .exceptions import UnmappedAddress
        if machine.big_endian != (self.endian == "big"):
            raise ValueError(f"program is {self.endian}-endian, machine is not")
        fixed = getattr(machine.mmu, "mt", 3) == 3
        if boot:
            from .tlb import boot_cca, firmware_refill, user_window
            cca = boot_cca(machine)                    # 3; a CPS core: 5, coherent
            # (Release 6: Status.FR is 1 and read-only — 64-bit FPRs only)
            fr = machine.status & C.ST_FR if machine.release == "R6" else 0
            machine._cp0_set(C.STATUS, C.ST_CU1 | fr)
            config = machine.cp0_read(*C.CONFIG) & ~((7 << 28) | (7 << 25) | 7)
            ku_k23 = (2 << 28) | (cca << 25) if fixed else 0
            machine._cp0_set(C.CONFIG, config | ku_k23 | cca)
            if machine.caches is not None:
                machine.caches.initialize()
            if not fixed:
                own = any(0x80000000 <= a < 0x80000100 for a in self.image)
                tlb = machine.mmu.mt in (1, 4)              # BAT: no refill handler
                machine.mmu.boot(machine, handler=tlb and not own)
                if tlb and not own:
                    fw = assemble(firmware_refill(cca, user_window(machine)),
                                  endian=self.endian,
                                  release=machine.release)
                    for addr, byte in fw.image.items():
                        machine.write_byte(addr, byte)
                    machine.firmware = fw
            board = getattr(machine, "board", None)
            if board is not None:                      # the board's firmware (YAMON)
                board.firmware(machine)
                machine.write_reg(29, board.stack_top())
                machine.write_reg(28, self.segments.get("data", (0x80200000,))[0] + 0x8000)
            else:
                machine.write_reg(29, BOOT_SP)
                machine.write_reg(28, BOOT_GP)
        for addr, byte in self.image.items():
            try:
                machine.write_byte(addr, byte)
            except UnmappedAddress:
                if boot and machine.mmu.mt in (1, 4):   # the firmware's convention
                    from .tlb import boot_cca, firmware_map, user_window
                    pa = firmware_map(addr, boot_cca(machine), user_window(machine))[0]
                    machine.phys_write_byte(pa, byte)
                    if machine.caches is not None:
                        machine.caches.backdoor_write(pa, byte)
                    continue
                raise UnmappedAddress(
                    f"cannot load the byte at 0x{addr:08x}: the address is mapped "
                    "through the TLB, which holds no translation for it (after a "
                    "reset the TLB is not initialised: put this code in kseg0 / "
                    "kseg1, or start with the boot firmware)",
                    name=None, word=None, addr=addr) from None
        if self.entry & 1:                             # a MIPS16e / microMIPS entry point
            want = "micromips" if self.mm else "mips16e"
            if getattr(machine, "compressed", None) != want:
                isa = "microMIPS" if self.mm else "MIPS16e"
                raise ValueError(f"the program starts in {isa} code: the core needs "
                                 f"{isa} (--compressed {want})")
            machine._jump(self.entry)
        else:
            machine.pc = self.entry

    def to_ihex(self) -> str:
        """Intel HEX (types 04 / 00 / 05 / 01) of the image."""
        lines: list[str] = []

        def rec(rtype: int, offset: int, data: bytes) -> None:
            body = bytes([len(data), offset >> 8, offset & 0xFF, rtype]) + data
            lines.append(":" + body.hex().upper()
                         + f"{(-sum(body)) & 0xFF:02X}")

        upper = None
        addrs = sorted(self.image)
        i = 0
        while i < len(addrs):
            start = addrs[i]
            chunk = bytearray([self.image[start]])
            i += 1
            while (i < len(addrs) and addrs[i] == start + len(chunk)
                   and len(chunk) < 16 and (addrs[i] & 0xFFFF) != 0):
                chunk.append(self.image[addrs[i]])
                i += 1
            if start >> 16 != upper:
                upper = start >> 16
                rec(0x04, 0, upper.to_bytes(2, "big"))
            rec(0x00, start & 0xFFFF, bytes(chunk))
        rec(0x05, 0, self.entry.to_bytes(4, "big"))
        rec(0x01, 0, b"")
        return "\n".join(lines) + "\n"

    def half(self, addr: int) -> int:
        data = bytes(self.image.get(addr + i, 0) for i in range(2))
        return struct.unpack(">H" if self.endian == "big" else "<H", data)[0]

    def m16_insn(self, addr: int):
        """The MIPS16e instruction at ``addr`` (emips.mips16.M16)."""
        from . import mips16
        return mips16.decode(self.half(addr), lambda: self.half(addr + 2))

    def mm_insn(self, addr: int):
        """The microMIPS instruction at ``addr`` (emips.micromips.MM)."""
        from . import micromips
        return micromips.decode(self.half(addr), lambda: self.half(addr + 2),
                                release=self.release)

    def listing(self, *, abi: bool = True) -> list[tuple[int, int, str, str]]:
        """``(address, word, disassembly, source)`` for every instruction
        (a MIPS16e one: its 16 / 32-bit encoding)."""
        from . import mips16
        from .disasm import disassemble
        labels = self.labels()
        isa = ISA.for_release(self.release)
        out = []
        for a, s in sorted(self.source_map.items()):
            if a in self.m16:
                i = self.m16_insn(a)
                out.append((a, i.raw, mips16.text(i, a, abi=abi), s.text.strip()))
            elif a in self.mm:
                from . import micromips
                i = self.mm_insn(a)
                out.append((a, i.raw, micromips.text(i, a, abi=abi, release=self.release),
                            s.text.strip()))
            else:
                out.append((a, self.word(a), disassemble(self.word(a), a, abi=abi, isa=isa,
                                                          symbols=labels), s.text.strip()))
        return out


# ---------------------------------------------------------------- lexing

_LABEL = re.compile(r"^\s*([A-Za-z_.$][\w.$]*)\s*:")
_NAME = re.compile(r"[A-Za-z_.$][\w.$]*$")
#: a numeric local label (GNU as: "1:"), and a reference to one ("1f" the
#: next definition, "1b" the latest)
_LOCAL = re.compile(r"^\s*(\d+)\s*:")
_LOCAL_REF = re.compile(r"(?<![\w.$'])(\d+)([fb])(?![\w.$])")


def _local_labels(texts: list[str]) -> list[str]:
    """Give GNU as's numeric local labels names: definition k of label N
    is ``.LN.k``; ``Nb`` refers to the latest definition at or before the
    line, ``Nf`` to the next one after it (outside quotes)."""
    defs: dict[str, list[int]] = {}
    heads: list[list[tuple[str, int]]] = []
    for i, t in enumerate(texts):
        found = []
        rest = t
        while True:
            m = _LOCAL.match(rest) or _LABEL.match(rest)
            if not m:
                break
            if m.re is _LOCAL:
                found.append((m.group(1), m.end()))
                defs.setdefault(m.group(1), []).append(i)
            rest = rest[m.end():]
        heads.append(found)
    if not defs:
        return texts
    seen: dict[str, int] = {}
    out = []
    for i, t in enumerate(texts):
        for num, _end in heads[i]:
            k = seen.get(num, 0)
            seen[num] = k + 1
            t = re.sub(rf"^(\s*(?:[A-Za-z_.$][\w.$]*\s*:\s*|\.L\d+\.\d+\s*:\s*)*){num}\s*:",
                       rf"\g<1>.L{num}.{k}:", t, count=1)

        def ref(m: re.Match, i: int = i) -> str:
            num, way = m.group(1), m.group(2)
            lines = defs.get(num)
            if lines is None:
                return m.group(0)
            if way == "b":
                ks = [k for k, line in enumerate(lines) if line <= i]
                return f".L{num}.{ks[-1]}" if ks else m.group(0)
            ks = [k for k, line in enumerate(lines) if line > i]
            return f".L{num}.{ks[0]}" if ks else m.group(0)

        parts = re.split(r"(\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])')", t)
        t = "".join(p if j % 2 else _LOCAL_REF.sub(ref, p) for j, p in enumerate(parts))
        out.append(t)
    return out


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote and (len(out) < 2 or out[-2] != "\\"):
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _split_operands(text: str) -> list[str]:
    parts, depth, cur, quote = [], 0, [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote and cur[-2:-1] != ["\\"]:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


_ESC = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}


def _unescape(body: str) -> bytes:
    out, i = bytearray(), 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            out += _ESC.get(body[i + 1], body[i + 1]).encode("latin-1")
            i += 2
        else:
            out += ch.encode("latin-1")
            i += 1
    return bytes(out)


# ------------------------------------------------------------- expressions

class _Unresolved(Exception):
    """A symbol is not (yet) defined."""


_TOKEN = re.compile(r"\s*(%hi|%lo|0[xX][0-9a-fA-F]+|0[bB][01]+|\d+|'(?:\\.|[^'])'"
                    r"|<<|>>|[A-Za-z_.$][\w.$]*|[-+*/()&|~])")


def _eval(expr: str, lookup: Callable[[str], int]) -> int:
    toks = []
    pos = 0
    expr = expr.strip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            raise ValueError(f"bad expression {expr!r}")
        toks.append(m.group(1))
        pos = m.end()
    i = 0

    def peek() -> str | None:
        return toks[i] if i < len(toks) else None

    def take() -> str:
        nonlocal i
        i += 1
        return toks[i - 1]

    def atom() -> int:
        t = take()
        if t == "(":
            v = binop(0)
            if take() != ")":
                raise ValueError("missing )")
            return v
        if t == "-":
            return -atom()
        if t == "+":
            return atom()
        if t == "~":
            return ~atom()
        if t in ("%hi", "%lo"):
            if take() != "(":
                raise ValueError(f"{t} needs (...)")
            v = binop(0)
            if take() != ")":
                raise ValueError("missing )")
            v &= M32
            return ((v + 0x8000) >> 16) & 0xFFFF if t == "%hi" else _sx16(v)
        if t[0] == "'":
            return ord(_unescape(t[1:-1]).decode("latin-1"))
        if t[:2].lower() == "0x":
            return int(t, 16)
        if t[:2].lower() == "0b":
            return int(t, 2)
        if t.isdigit():
            return int(t)
        return lookup(t)

    prec = {"|": 1, "&": 2, "<<": 3, ">>": 3, "+": 4, "-": 4, "*": 5, "/": 5}

    def binop(level: int) -> int:
        v = atom()
        while peek() in prec and prec[peek()] > level:
            op = take()
            r = binop(prec[op])
            v = {"|": v | r, "&": v & r, "<<": v << r, ">>": v >> r,
                 "+": v + r, "-": v - r, "*": v * r,
                 "/": int(v / r) if r else 0}[op]
        return v

    v = binop(0)
    if i != len(toks):
        raise ValueError(f"bad expression {expr!r}")
    return v


def _sx32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def _sx16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


# ------------------------------------------------------------ the assembler

#: (mnemonic -> handler) for pseudo-instructions; each returns a list of
#: (mnemonic, operands) real instructions.
_BRANCH_MACROS = {
    "blt": ("slt", False, "bne"), "bltu": ("sltu", False, "bne"),
    "bge": ("slt", False, "beq"), "bgeu": ("sltu", False, "beq"),
    "bgt": ("slt", True, "bne"), "bgtu": ("sltu", True, "bne"),
    "ble": ("slt", True, "beq"), "bleu": ("sltu", True, "beq"),
}

#: strict mode (``pseudo=False``): what to write instead of each pseudo-op
PSEUDO_HINTS = {
    "move": "or rd, rs, $zero",
    "neg": "sub rd, $zero, rs",
    "negu": "subu rd, $zero, rs",
    "not": "nor rd, rs, $zero",
    "beqz": "beq rs, $zero, label",
    "bnez": "bne rs, $zero, label",
    "li": "addiu rt, $zero, imm / ori rt, $zero, imm / lui rt, hi + ori rt, rt, lo",
    "la": "lui rt, %hi(label) + addiu rt, rt, %lo(label)",
    **{mn: f"{cmp} $at, ... + {br} $at, $zero, label"
       for mn, (cmp, _swap, br) in _BRANCH_MACROS.items()},
}

#: GNU as's macros of the ALU instructions (gas/config/tc-mips.c,
#: macro()): a constant in place of rt -> the immediate instruction when
#: it fits, else the constant in $at (as ``li``) and the register form
_ALU_IMMEDIATE = {"add": "addi", "addu": "addiu", "sub": "addi", "subu": "addiu",
                  "and": "andi", "or": "ori", "xor": "xori", "nor": "ori",
                  "slt": "slti", "sltu": "sltiu"}
#: shifts by a register written with the immediate mnemonic
_SHIFT_VARIABLE = {"sll": "sllv", "srl": "srlv", "sra": "srav", "rotr": "rotrv"}
#: immediates GNU as also takes as a 16-bit pattern (0x8000-0xFFFF = negative)
_SIGNED_IMMEDIATE = {"addi", "addiu", "slti", "sltiu"}
#: ``op rd, rt`` = ``op rd, rd, rt`` (GNU as): register forms ...
_SHORT_REGISTER = set(_ALU_IMMEDIATE) | set(_SHIFT_VARIABLE) | {"mul", "movn", "movz"}
#: ... and immediate forms
_SHORT_IMMEDIATE = set(_ALU_IMMEDIATE) | set(_SHIFT_VARIABLE) | {"mul", "addi", "addiu",
                                                                   "andi", "ori", "xori",
                                                                   "slti", "sltiu"}

_DIRECTIVES_IGNORED = {".globl", ".global", ".ent", ".end", ".type", ".size",
                       ".extern", ".file", ".section", ".previous", ".frame",
                       ".mask", ".fmask", ".abicalls", ".option"}


class _Relax(Exception):
    """A MIPS16e statement needs its long form: lay the program out again."""


@dataclass
class _Stmt:
    line: int
    text: str
    seg: str
    addr: int
    mnemonic: str
    ops: list[str]
    reorder: bool
    at_ok: bool
    size: int = 0
    #: R6: a nop is put in this compact branch's forbidden slot
    forbidden_nop: bool = False
    #: a MIPS16e statement (.set mips16)
    m16: bool = False
    #: a microMIPS statement (.set micromips)
    mm: bool = False
    #: microMIPS, noreorder: the size the previous branch's delay slot
    #: demands of this statement's first instruction (0 = any)
    slot: int = 0


class Assembler:
    def __init__(self, *, endian: str = "big", isa: ISA | None = None,
                 bases: dict[str, int] | None = None,
                 release: str | None = None, pseudo: bool = True,
                 reorder: bool = True) -> None:
        if endian not in ("big", "little"):
            raise ValueError("endian must be 'big' or 'little'")
        self.endian = endian
        #: pseudo-instructions allowed (False: strict mode)
        self.pseudo = pseudo
        #: initial ``.set reorder`` state (nop in every delay slot)
        self.reorder = reorder
        self.isa = isa or ISA.for_release(release)
        self.bases = dict(SEGMENT_BASES, **(bases or {}))
        #: instructions of the target release only, by mnemonic
        self._by_name = {e.get("mnemonic", e["name"]).lower(): e
                         for e in self.isa.instructions
                         if self.isa.in_release(e["name"])}
        self._all = {e.get("mnemonic", e["name"]).lower()
                     for e in self.isa.instructions}

    # -- public -------------------------------------------------------------

    def assemble(self, source: str) -> Program:
        #: MIPS16e statements (by line) too far for their short form
        self._m16_long: set[int] = set()
        for _ in range(64):                     # MIPS16e relaxation: sizes only grow
            try:
                return self._assemble_once(source)
            except _Relax:
                continue
        raise AsmError(0, "MIPS16e branch relaxation does not converge")

    def _assemble_once(self, source: str) -> Program:
        self.symbols: dict[str, int] = {}
        #: names defined by .eqv / .equ / .set NAME (not labels)
        self._consts: set[str] = set()
        #: labels of MIPS16e / microMIPS code (their address, taken as data,
        #: has bit 0 set)
        self._m16_labels: set[str] = set()
        self._mode16 = False
        self._modemm = False
        self._mm_enc = None
        self._seen_m16 = self._seen_mm = False
        self.prog = Program(endian=self.endian, release=self.isa.release)
        stmts = self._pass1(source)
        for st in stmts:
            self._emit(st)
        self.prog.symbols = dict(self.symbols)
        for name in ("main", "__start"):
            if name in self.symbols:
                self.prog.entry = self.symbols[name] | (name in self._m16_labels)
                break
        else:
            base = self.bases["text"]
            self.prog.entry = base | (base in self.prog.mm or base in self.prog.m16)
        return self.prog

    # -- pass 1: layout -------------------------------------------------------

    def _pass1(self, source: str) -> list[_Stmt]:
        lc = dict(self.bases)
        seg = "text"
        reorder, at_ok = self.reorder, True
        stmts: list[_Stmt] = []
        #: labels at the current location with nothing emitted after them
        #: yet: an auto-aligning directive moves them along (GNU as, MARS)
        self._pending: list[str] = []
        prev: _Stmt | None = None                 # previous instruction
        lines = source.splitlines()
        texts = _local_labels([_strip_comment(raw) for raw in lines])
        for n, raw in enumerate(lines, start=1):
            text = texts[n - 1]
            while True:
                m = _LABEL.match(text)
                if not m:
                    break
                name = m.group(1)
                if name in self.symbols:
                    raise AsmError(n, f"label {name!r} defined twice", raw)
                self.symbols[name] = lc[seg]
                self._pending.append(name)
                text = text[m.end():]
            text = text.strip()
            if not text:
                continue
            head, _, rest = text.partition(" ")
            if "\t" in head:
                head, _, more = head.partition("\t")
                rest = more + " " + rest
            mnemonic, ops = head.lower(), _split_operands(rest.strip())
            try:
                if mnemonic.startswith("."):
                    seg, reorder, at_ok = self._directive_layout(
                        n, raw, mnemonic, ops, rest, lc, seg, reorder, at_ok, stmts)
                    continue
                if seg in ("data", "kdata"):
                    raise AsmError(n, f"instruction in .{seg} segment", raw)
                if self._modemm:                # microMIPS: halfwords, sizes 2 / 4
                    if lc[seg] & 1:
                        raise AsmError(n, "microMIPS instruction not halfword-aligned", raw)
                    self._m16_labels.update(self._pending)
                    st = _Stmt(n, raw, seg, lc[seg], mnemonic, ops, reorder, at_ok, mm=True)
                    if prev is not None and prev.mm and not prev.reorder and \
                            prev.addr + prev.size == lc[seg]:
                        st.slot = self._mm_slot_need(prev)
                    st.size = sum(i.size for i in self._mm_encode(st, final=False))
                    stmts.append(st)
                    lc[seg] += st.size
                    self._pending = []
                    prev = st
                    continue
                if self._mode16:                # MIPS16e: halfwords, sizes 2 / 4
                    if lc[seg] & 1:
                        raise AsmError(n, "MIPS16e instruction not halfword-aligned", raw)
                    self._m16_labels.update(self._pending)
                    st = _Stmt(n, raw, seg, lc[seg], mnemonic, ops, reorder, at_ok, m16=True)
                    st.size = self._m16_size(st)
                    stmts.append(st)
                    lc[seg] += st.size
                    self._pending = []
                    prev = None
                    continue
                if lc[seg] & 3:
                    raise AsmError(n, "instruction not word-aligned" + (
                        " (after MIPS16e / microMIPS code: .align 2)" if self._m16_labels
                        else ""), raw)
                st = _Stmt(n, raw, seg, lc[seg], mnemonic, ops, reorder, at_ok)
                insns = self._expand(st)
                if (prev is not None and prev.reorder and prev.seg == seg
                        and prev.addr + prev.size == lc[seg]
                        and self._conditional_compact(self._expand(prev)[-1][0])
                        and self._cti(insns[0][0])):
                    # R6 forbidden slot: GNU as puts a nop there (.set reorder)
                    prev.forbidden_nop = True
                    prev.size += 4
                    lc[seg] += 4
                    st.addr = lc[seg]
                    for name in self._pending:
                        self.symbols[name] = lc[seg]
                st.size = 4 * len(insns)
                stmts.append(st)
                lc[seg] += st.size
                self._pending = []
                prev = st
            except AsmError:
                raise
            except (ValueError, KeyError) as exc:
                raise AsmError(n, str(exc), raw) from None
        return stmts

    def _directive_layout(self, n, raw, d, ops, rest, lc, seg, reorder,
                          at_ok, stmts):
        def stmt(size: int, align: int = 1) -> None:
            pad = (-lc[seg]) % align
            st = _Stmt(n, raw, seg, lc[seg] + pad, d, ops, reorder, at_ok, size)
            stmts.append(st)
            for name in self._pending:
                self.symbols[name] = st.addr
            self._pending = []
            lc[seg] += pad + size

        if d in (".text", ".data", ".ktext", ".kdata"):
            self._pending = []
            seg = d[1:]
            if ops:
                lc[seg] = self._const(ops[0], n)
            return seg, reorder, at_ok
        if d == ".set":
            word = ops[0].lower() if ops else ""
            if len(ops) == 2 and _NAME.match(ops[0]):
                self.symbols[ops[0]] = self._const(ops[1], n)
                self._consts.add(ops[0])
            elif word in ("reorder", "noreorder"):
                reorder = word == "reorder"
            elif word in ("at", "noat"):
                at_ok = word == "at"
            elif word in ("mips16", "nomips16"):
                if word == "mips16" and self.isa.release == "R6":
                    raise AsmError(n, "Release 6 has no MIPS16e", raw)
                if word == "mips16" and self._seen_mm:
                    raise AsmError(n, "MIPS16e after microMIPS code: no core has both", raw)
                self._mode16 = word == "mips16"
                self._seen_m16 |= self._mode16
            elif word in ("micromips", "nomicromips"):
                if word == "micromips" and self.isa.release in ("R1", "R2"):
                    raise AsmError(n, "microMIPS32 starts with Release 3 (--release R3, "
                                   "R5 or R6)", raw)
                if word == "micromips" and self._seen_m16:
                    raise AsmError(n, "microMIPS after MIPS16e code: no core has both", raw)
                self._modemm = word == "micromips"
                self._seen_mm |= self._modemm
            return seg, reorder, at_ok
        if d in (".eqv", ".equ"):
            if len(ops) != 2:
                raise AsmError(n, f"{d} NAME, value", raw)
            self.symbols[ops[0]] = self._const(ops[1], n)
            self._consts.add(ops[0])
            return seg, reorder, at_ok
        if d in _DIRECTIVES_IGNORED:
            return seg, reorder, at_ok
        if d == ".align":                       # moves pending labels too
            a = 1 << self._const(ops[0], n)
            pad = (-lc[seg]) % a
            if (self._mode16 or self._modemm) and seg in ("text", "ktext") and pad:
                # MIPS16e / microMIPS code: GNU as fills with 16-bit NOPs
                stmts.append(_Stmt(n, raw, seg, lc[seg], ".fill16", [], reorder, at_ok, pad,
                                   mm=self._modemm))
            lc[seg] += pad
            for name in self._pending:
                self.symbols[name] = lc[seg]
        elif d == ".space":
            stmt(self._const(ops[0], n))
        elif d == ".word":
            stmt(4 * len(ops), 4)
        elif d == ".half":
            stmt(2 * len(ops), 2)
        elif d == ".byte":
            stmt(len(ops))
        elif d == ".float":
            stmt(4 * len(ops), 4)
        elif d == ".double":
            stmt(8 * len(ops), 8)
        elif d in (".ascii", ".asciiz"):
            data = self._strings(rest, n, raw, d == ".asciiz")
            stmt(len(data))
        else:
            raise AsmError(n, f"unknown directive {d}", raw)
        return seg, reorder, at_ok

    def _strings(self, rest: str, n: int, raw: str, zero: bool) -> bytes:
        out = b""
        for part in _split_operands(rest):
            if len(part) < 2 or part[0] != '"' or part[-1] != '"':
                raise AsmError(n, "string literal expected", raw)
            out += _unescape(part[1:-1]) + (b"\0" if zero else b"")
        return out

    # -- expressions ----------------------------------------------------------

    def _lookup(self, name: str) -> int:
        if name in self.symbols:
            return self.symbols[name]
        raise _Unresolved(name)

    def _const(self, expr: str, n: int) -> int:
        try:
            return _eval(expr, self._lookup)
        except _Unresolved as exc:
            raise AsmError(n, f"undefined symbol {exc}") from None
        except ValueError as exc:
            raise AsmError(n, str(exc)) from None

    def _value(self, expr: str, st: _Stmt) -> int:
        try:
            return _eval(expr, self._lookup)
        except _Unresolved as exc:
            raise AsmError(st.line, f"undefined symbol {exc}", st.text) from None
        except ValueError as exc:
            raise AsmError(st.line, str(exc), st.text) from None

    def _is_constant(self, expr: str) -> bool:
        """True when ``expr`` uses no labels (only numbers and .eqv/.equ/.set
        constants).  Expansion sizes depend on this, so it gives the same
        answer in both passes."""
        names = [t for t in _TOKEN.findall(expr)
                 if re.fullmatch(r"[A-Za-z_.$][\w.$]*", t)]
        return all(t in self._consts for t in names)

    # -- pseudo-instruction expansion -------------------------------------------

    def _expand(self, st: _Stmt) -> list[tuple[str, list[str]]]:
        """Real instructions (mnemonic, operands) for a source statement,
        including the reorder-mode delay-slot ``nop``."""
        mn, ops = st.mnemonic, st.ops
        out: list[tuple[str, list[str]]]
        at = "$at"

        def need_at() -> None:
            if not st.at_ok:
                raise AsmError(st.line, f"{mn} needs $at, but .set noat", st.text)

        if not self.pseudo and mn in PSEUDO_HINTS:
            raise AsmError(st.line, f"{mn} is a pseudo-instruction (strict mode); "
                           f"write: {PSEUDO_HINTS[mn]}", st.text)

        if mn == "nop":
            out = [("nop", [])]
        elif mn == "move":
            self._nops(st, 2)
            out = [("or", [ops[0], ops[1], "$zero"])]
        elif mn in ("neg", "negu"):
            self._nops(st, 2)
            out = [("sub" if mn == "neg" else "subu", [ops[0], "$zero", ops[1]])]
        elif mn == "not":
            self._nops(st, 2)
            out = [("nor", [ops[0], ops[1], "$zero"])]
        elif mn in ("beqz", "bnez"):
            self._nops(st, 2)
            out = [("beq" if mn == "beqz" else "bne", [ops[0], "$zero", ops[1]])]
        elif mn in _BRANCH_MACROS:
            self._nops(st, 3)
            cmp, swap, br = _BRANCH_MACROS[mn]
            need_at()
            a, b = (ops[1], ops[0]) if swap else (ops[0], ops[1])
            out = [(cmp, [at, a, b]), (br, [at, "$zero", ops[2]])]
        elif mn == "li":
            self._nops(st, 2)
            out = self._li(st, ops[0], ops[1])
        elif mn == "la":
            self._nops(st, 2)
            out = self._li(st, ops[0], ops[1])
        elif ((mn in self._FP_MEM or mn in self._by_name
               and self._by_name[mn]["family"] in ("load", "store")
               and "mem" in operand_spec(self._by_name[mn]))
              and len(ops) == 2 and self._mem_needs_macro(ops[1], st)):
            if not self.pseudo:
                raise AsmError(st.line, f"{mn} with a label or 32-bit address is a "
                               "pseudo-instruction (strict mode); write: lui $at, "
                               f"%hi(label) + {mn} rt, %lo(label)($at)", st.text)
            out = self._mem_macro(mn, ops, st)
        elif mn in ("jr", "jr.hb") and mn not in self._by_name:
            # Release 6: JR is JALR with rd = $zero (GNU as does the same)
            out = [(mn.replace("jr", "jalr"), ["$zero"] + ops)]
        elif mn in ("div", "divu") and len(ops) == 3 and \
                self._by_name.get(mn, {}).get("name", "") in ("DIV", "DIVU"):
            if parse_gpr(ops[0]) != 0:
                raise AsmError(st.line, f"{mn} rd form: first operand must be $zero", st.text)
            out = [(mn, ops[1:])]
        elif (alu := self._alu_macro(st, need_at)) is not None:
            out = alu
        elif mn in self._by_name:
            out = [(mn, ops)]
        elif mn in self._all:
            raise AsmError(st.line, f"{mn} does not exist in MIPS32 "
                           f"{self.isa.release}", st.text)
        else:
            raise AsmError(st.line, f"unknown instruction {mn!r}", st.text)

        last = out[-1][0]
        if st.reorder and last in self._by_name and self._has_delay_slot(last):
            out.append(("nop", []))
        return out

    def _alu_macro(self, st: _Stmt, need_at) -> list[tuple[str, list[str]]] | None:
        """GNU as's forms of the ALU instructions, expanded exactly as it
        does: ``op rd, rt`` = ``op rd, rd, rt``; a constant in place of a
        register (``or rd, rs, 0x12345``: ``ori`` when it fits, else the
        constant in ``$at``, ``sub`` / ``subu`` as an add of -imm, ``nor``
        as ``ori`` + ``nor``, ``mul`` as ``mult`` + ``mflo``); ``sll rd, rt,
        rs`` = ``sllv``; ``addiu rt, rs, 0xFFFF`` = -1; ``rotr`` by any
        constant, modulo 32.  None: not one of them (the instruction as
        written)."""
        mn, ops = st.mnemonic, list(st.ops)
        if mn not in self._by_name or not 2 <= len(ops) <= 3:
            return None
        if ops[-1].strip().startswith("$") and parse_gpr(ops[-1]) is None:
            return None                         # a bad register: the plain error
        short = len(ops) == 2
        if short:
            reg = parse_gpr(ops[1]) is not None
            if mn not in (_SHORT_REGISTER if reg else _SHORT_IMMEDIATE):
                return None
            self._strict(st, f"{mn} {ops[0]}, {ops[0]}, {ops[1]}")
            ops = [ops[0], ops[0], ops[1]]
        rd, rs, third = ops
        if parse_gpr(third) is not None:
            if mn in _SHIFT_VARIABLE and _SHIFT_VARIABLE[mn] in self._by_name:
                self._strict(st, f"{_SHIFT_VARIABLE[mn]} {rd}, {rs}, {third}")
                return [(_SHIFT_VARIABLE[mn], ops)]
            return [(mn, ops)] if short else None
        if mn == "rotr" and self._is_constant(third) and \
                not 0 <= self._value(third, st) <= 31:          # GNU as: modulo 32
            v = self._value(third, st) & 31
            self._strict(st, f"rotr {rd}, {rs}, {v}")
            return [("rotr", [rd, rs, str(v)])]
        if mn in _SIGNED_IMMEDIATE:
            if self._is_constant(third) and 0x8000 <= self._value(third, st) <= 0xFFFF:
                return [(mn, [rd, rs, str(_sx16(self._value(third, st)))])]
            return [(mn, ops)] if short else None
        if mn not in _ALU_IMMEDIATE and not (mn == "mul" and "mult" in self._by_name):
            return [(mn, ops)] if short else None
        if not self._is_constant(third):
            raise AsmError(st.line, f"{mn}: the third operand must be a register or a "
                           "constant", st.text)
        v = _sx32(self._value(third, st))
        imm = _ALU_IMMEDIATE.get(mn)
        if mn in ("and", "or", "xor") and 0 <= v <= 0xFFFF:
            self._strict(st, f"{imm} {rd}, {rs}, {v}")
            return [(imm, [rd, rs, str(v)])]
        if mn == "nor" and 0 <= v <= 0xFFFF:
            self._strict(st, f"ori {rd}, {rs}, {v} + nor {rd}, {rd}, $zero")
            return [("ori", [rd, rs, str(v)]), ("nor", [rd, rd, "$zero"])]
        if mn in ("add", "addu", "slt", "sltu") and imm in self._by_name \
                and -0x8000 <= v <= 0x7FFF:
            self._strict(st, f"{imm} {rd}, {rs}, {v}")
            return [(imm, [rd, rs, str(v)])]
        if mn in ("sub", "subu") and imm in self._by_name and -0x8000 <= -v <= 0x7FFF:
            self._strict(st, f"{imm} {rd}, {rs}, {-v}")
            return [(imm, [rd, rs, str(-v)])]
        self._strict(st, f"li $at, {third} + " +
                     (f"mult {rs}, $at + mflo {rd}" if mn == "mul" else f"{mn} {rd}, {rs}, $at"))
        need_at()
        load = self._li(st, "$at", str(v & M32))
        if mn == "mul":
            return load + [("mult", [rs, "$at"]), ("mflo", [rd])]
        return load + [(mn, [rd, rs, "$at"])]

    def _strict(self, st: _Stmt, real: str) -> None:
        """Strict mode: a GNU as macro is an error naming the real
        instructions."""
        if not self.pseudo:
            raise AsmError(st.line, f"{st.mnemonic} with these operands is a "
                           f"pseudo-instruction (strict mode); write: {real}", st.text)

    def _nops(self, st: _Stmt, count: int) -> None:
        if len(st.ops) != count:
            raise AsmError(st.line, f"{st.mnemonic} takes {count} operands", st.text)

    def _cti(self, mn: str) -> bool:
        e = self._by_name.get(mn)
        return e is not None and (e["family"] in (
            "branch", "branch_fpu", "jump", "branch_compact", "jump_compact")
            or e["name"].startswith("BC2"))

    def _conditional_compact(self, mn: str) -> bool:
        e = self._by_name.get(mn)
        return e is not None and e["family"] == "branch_compact" and \
            e["name"] not in ("BC", "BALC")

    def _has_delay_slot(self, mn: str) -> bool:
        e = self._by_name[mn]
        return e["family"] in ("branch", "jump", "branch_fpu") or \
            mn.upper().startswith("BC2")

    def _li(self, st: _Stmt, rt: str, expr: str) -> list[tuple[str, list[str]]]:
        """``li`` / ``la``: a constant gets the shortest GNU ``as`` form; an
        address expression always ``lui %hi`` + ``addiu %lo``."""
        if not self._is_constant(expr):
            if self._isa_bit(expr):              # MIPS16e code: bit 0 = the ISA mode
                expr = f"({expr}) + 1"
            return [("lui", [rt, f"%hi({expr})"]),
                    ("addiu", [rt, rt, f"%lo({expr})"])]
        v = self._value(expr, st) & M32
        if v < 0x8000 or v >= 0xFFFF8000:
            return [("addiu", [rt, "$zero", str(_sx16(v))])]
        if v <= 0xFFFF:
            return [("ori", [rt, "$zero", str(v)])]
        out = [("lui", [rt, str(v >> 16)])]
        if v & 0xFFFF:
            out.append(("ori", [rt, rt, str(v & 0xFFFF)]))
        return out

    def _mem_needs_macro(self, op: str, st: _Stmt) -> bool:
        m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", op.strip())
        if not m:
            return True                             # label / absolute address
        off = m.group(1).strip()
        if not off or "%lo" in off:
            return False
        if not self._is_constant(off):
            return True
        v = self._value(off, st)
        return not -0x8000 <= v <= 0x7FFF

    #: FPU loads / stores that also accept a label or 32-bit address
    _FP_MEM = {"lwc1", "swc1", "ldc1", "sdc1"}

    #: loads whose rt may serve as the address scratch (rt is overwritten
    #: anyway, and not merged like LWL/LWR) — GNU as does the same
    _RT_SCRATCH = {"lb", "lbu", "lh", "lhu", "lw", "ll"}

    def _mem_macro(self, mn: str, ops: list[str], st: _Stmt):
        m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", ops[1].strip())
        expr, base = (m.group(1), m.group(2).strip()) if m else (ops[1], None)
        rt = parse_gpr(ops[0])
        tmp = "$at"
        if (mn in self._RT_SCRATCH and rt not in (None, 0)
                and (base is None or parse_gpr(base) != rt)):
            tmp = ops[0]
        elif not st.at_ok:
            raise AsmError(st.line, f"{mn} needs $at, but .set noat", st.text)
        out = [("lui", [tmp, f"%hi({expr})"])]
        if base:
            out.append(("addu", [tmp, tmp, base]))
        out.append((mn, [ops[0], f"%lo({expr})({tmp})"]))
        return out

    # -- pass 2: emit -----------------------------------------------------------

    def _emit(self, st: _Stmt) -> None:
        if st.mnemonic.startswith("."):
            self._emit_data(st)
            return
        if st.m16:
            self._emit16(st)
            return
        if st.mm:
            self._emitmm(st)
            return
        insns = self._expand(st)
        if st.forbidden_nop:
            insns = insns + [("nop", [])]
        if 4 * len(insns) != st.size:
            raise AsmError(st.line, "expansion size changed between passes "
                           "(use constants defined before use)", st.text)
        for k, (mn, ops) in enumerate(insns):
            addr = st.addr + 4 * k
            word = self._encode(mn, ops, addr, st)
            self._put(addr, word, 4)
            self.prog.source_map[addr] = SourceLine(st.line, st.text)
            self._mark(st.seg, addr, 4)

    # -- MIPS16e ------------------------------------------------------------------

    def _m16_value(self, final: bool):
        from .mips16 import Unresolved

        def value(expr: str) -> int:
            try:
                return _eval(expr, self._lookup)
            except _Unresolved:
                if final:
                    raise
                raise Unresolved(expr) from None
        return value

    def _m16_encode(self, st: _Stmt, final: bool) -> list[list[int]]:
        from . import mips16
        try:
            value = self._m16_value(final)
            if st.mnemonic == "la":                 # a MIPS16e label: the ISA bit (GNU as)
                plain = value
                value = lambda e: plain(e) | self._isa_bit(e)     # noqa: E731
            return mips16.encode(st.mnemonic, st.ops, st.addr, value,
                                 reorder=st.reorder, long=st.line in self._m16_long)
        except _Unresolved as exc:
            raise AsmError(st.line, f"undefined symbol {exc}", st.text) from None
        except mips16.AsmError16 as exc:
            raise AsmError(st.line, str(exc), st.text) from None
        except ValueError as exc:
            raise AsmError(st.line, str(exc), st.text) from None

    def _m16_size(self, st: _Stmt) -> int:
        """Pass 1: the bytes of a MIPS16e statement (the short form while a
        label is not known yet, unless relaxation made it long)."""
        return sum(2 * len(i) for i in self._m16_encode(st, final=False))

    def _emit16(self, st: _Stmt) -> None:
        insns = self._m16_encode(st, final=True)
        size = sum(2 * len(i) for i in insns)
        if size != st.size:
            if size > st.size and st.line not in self._m16_long:
                self._m16_long.add(st.line)     # too far for the short form: again
                raise _Relax
            raise AsmError(st.line, "MIPS16e size changed between passes "
                           "(use constants defined before use)", st.text)
        addr = st.addr
        for hws in insns:
            for k, h in enumerate(hws):
                self._put(addr + 2 * k, h, 2)
            self.prog.source_map[addr] = SourceLine(st.line, st.text)
            self.prog.m16[addr] = 2 * len(hws)
            self._mark(st.seg, addr, 2 * len(hws))
            addr += 2 * len(hws)

    # -- microMIPS ----------------------------------------------------------------

    #: written as is (no MIPS32 macro): branches and jumps (the reorder
    #: rules below), the microMIPS-only instructions
    _MM_DIRECT = {"b", "bc", "beqz", "bnez", "beq", "bne", "beqzc", "bnezc", "bgez", "bltz",
                  "blez", "bgtz", "bgezal", "bltzal", "bgezals", "bltzals", "bal", "bals",
                  "bc1f", "bc1t", "bc2f", "bc2t", "j", "jal", "jals", "jalx", "jr", "jr.hb",
                  "jrs", "jrs.hb", "jalr", "jalr.hb", "jalrs", "jalrs.hb", "jrc", "jraddiusp",
                  "lwp", "swp", "lwm", "swm", "lwxs", "movep", "addiupc",
                  # Release 6
                  "balc", "beqc", "bnec", "bltc", "bgec", "bltuc", "bgeuc", "bovc", "bnvc",
                  "bgtc", "blec", "bgtuc", "bleuc", "bltzc", "blezc", "bgezc", "bgtzc",
                  "beqzalc", "bnezalc", "bltzalc", "blezalc", "bgezalc", "bgtzalc", "bc1eqzc",
                  "bc1nezc", "bc2eqzc", "bc2nezc", "jalrc", "jalrc.hb", "jrc.hb", "jic",
                  "jialc", "jrcaddiusp", "lwpc"}
    #: Release 6 also writes these as is
    _MM_DIRECT_R6 = {"cache"}
    #: delay slots of 32 bits (BD32) / 16 bits (BD16); the other branches
    #: and jumps take either (MD00582 5.4 / 5.5)
    _MM_SLOT32 = {"jal", "jalr", "jalr.hb", "jalx", "bgezal", "bltzal", "bal"}
    _MM_SLOT16 = {"jals", "jalrs", "jalrs.hb", "bgezals", "bltzals", "bals"}
    _MM_SLOT_ANY = {"b", "beqz", "bnez", "beq", "bne", "bgez", "bltz", "blez", "bgtz",
                    "bc1f", "bc1t", "bc2f", "bc2t", "j", "jr", "jr.hb", "jrs", "jrs.hb"}

    def _mm_expand(self, st: _Stmt) -> list[tuple[str, list[str], int]]:
        """(mnemonic, operands, forced size) of a microMIPS statement:
        GNU as's choices — the MIPS32 macros, and with ``.set reorder``
        (eMIPS never moves an instruction into a delay slot) B / BEQZ /
        BNEZ / JR as the compact BC / BEQZC / BNEZC / JRC, the other
        branches and jumps followed by a NOP of their delay slot's size."""
        mn, ops = st.mnemonic, list(st.ops)
        if mn == "li" and len(ops) == 2 and self._is_constant(ops[1]):
            v = self._value(ops[1], st)
            if parse_gpr(ops[0]) in (16, 17, 2, 3, 4, 5, 6, 7) and -1 <= v <= 126 \
                    and st.slot != 4:
                return [("li16", ops, st.slot)]
        if mn in self._MM_DIRECT or self.isa.release == "R6" and mn in self._MM_DIRECT_R6:
            out = [(mn, ops)]
        else:
            reorder, st.reorder = st.reorder, False
            pseudo = self.pseudo
            if mn in ("move", "not"):           # real: MOVE16, NOT16
                self.pseudo = True
            try:
                out = self._expand(st)
            finally:
                st.reorder, self.pseudo = reorder, pseudo
        # GNU as builds its macros from 32-bit forms: an instruction that is
        # not the statement itself (or MOVE / NOT / NEG(U), real here) is 32 bits
        real = mn in ("move", "not", "neg", "negu") or len(out) == 1 and out[0][0] == mn \
            and out[0][1] in (ops, ops[:1] + ops)
        res = [(m, o, st.slot if k == 0 and (real or st.slot == 4) else 0 if real else 4)
               for k, (m, o) in enumerate(out)]
        if not real and st.slot == 2:
            raise AsmError(st.line, f"{mn}: a macro in a 16-bit delay slot", st.text)
        last, lops, _ = res[-1]
        if st.reorder and self.isa.release != "R6":        # (Release 6: no delay slots)
            zero = [r for r in lops[:2] if parse_gpr(r) == 0] if last in ("beq", "bne") else []
            if last == "b":
                res[-1] = ("bc", lops, res[-1][2])
            elif last in ("beqz", "bnez"):
                res[-1] = (last + "c", lops, res[-1][2])
            elif zero and len(lops) == 3:
                other = lops[1] if parse_gpr(lops[0]) == 0 else lops[0]
                res[-1] = ("beqzc" if last == "beq" else "bnezc", [other, lops[2]], res[-1][2])
            elif last == "jr":
                res[-1] = ("jrc", lops, res[-1][2])
            elif last in self._MM_SLOT32 or last in self._MM_SLOT16 or \
                    last in self._MM_SLOT_ANY:
                res.append(("nop", [], 4 if last in self._MM_SLOT32 else 2))
        return res

    def _mm_slot_need(self, prev: _Stmt) -> int:
        """The size the delay slot after ``prev`` (noreorder) demands."""
        if self.isa.release == "R6":
            return 0
        last = self._mm_expand(prev)[-1][0]
        return 4 if last in self._MM_SLOT32 else 2 if last in self._MM_SLOT16 else 0

    def _mm_label_dep(self, ops: list[str]) -> bool:
        """An operand that uses a label (not a branch target): GNU as
        leaves a relocation there, so the 32-bit form."""
        for op in ops:
            names = [t for t in _TOKEN.findall(op)
                     if re.fullmatch(r"[A-Za-z_.][\w.$]*", t)]
            if any(t not in self._consts for t in names):
                return True
        return False

    def _mm_encode(self, st: _Stmt, final: bool) -> list:
        from . import micromips as MM
        if self._mm_enc is None or self._mm_enc.release != self.isa.release:
            self._mm_enc = MM.Encoder(self.isa.release)
        enc = self._mm_enc
        value = self._cvalue(final, MM.Unresolved)
        out, addr = [], st.addr
        long = st.line in self._m16_long
        for k, (mn, ops, force) in enumerate(self._mm_expand(st)):
            branch = mn in self._MM_DIRECT and mn not in (
                "lwp", "swp", "lwm", "swm", "lwxs", "movep") or mn in ("beq", "bne")
            try:
                if mn == "li16":
                    insn = enc._raw("LI16", d=MM.GPR3.index(parse_gpr(ops[0])),
                                    i=self._value(ops[1], st) & 0x7F)
                elif not branch and self._mm_label_dep(ops):
                    if not final:
                        insn = MM.Insn(4, 0)
                    else:
                        insn = enc.encode(mn, ops, addr, value, self._mm_mips32(addr, st),
                                          force=4)
                        if force == 2:
                            raise MM.AsmErrorMM(f"{mn}: the delay slot needs a 16-bit "
                                                "instruction")
                else:
                    insn = enc.encode(mn, ops, addr, value, self._mm_mips32(addr, st),
                                      force=force, long=long)
            except _Unresolved as exc:
                raise AsmError(st.line, f"undefined symbol {exc}", st.text) from None
            except MM.AsmErrorMM as exc:
                raise AsmError(st.line, str(exc), st.text) from None
            except ValueError as exc:
                raise AsmError(st.line, str(exc), st.text) from None
            out.append(insn)
            addr += insn.size
        return out

    def _mm_mips32(self, addr: int, st: _Stmt):
        return lambda mn, ops: self._encode(mn, ops, addr, st)

    def _cvalue(self, final: bool, unresolved):
        def value(expr: str) -> int:
            try:
                return _eval(expr, self._lookup)
            except _Unresolved:
                if final:
                    raise
                raise unresolved(expr) from None
        return value

    def _emitmm(self, st: _Stmt) -> None:
        insns = self._mm_encode(st, final=True)
        size = sum(i.size for i in insns)
        if size != st.size:
            if size > st.size and st.line not in self._m16_long:
                self._m16_long.add(st.line)     # a branch too far for the 16-bit form
                raise _Relax
            raise AsmError(st.line, "microMIPS size changed between passes "
                           "(use constants defined before use)", st.text)
        addr = st.addr
        for insn in insns:
            for k, h in enumerate(insn.halves):
                self._put(addr + 2 * k, h, 2)
            self.prog.source_map[addr] = SourceLine(st.line, st.text)
            self.prog.mm[addr] = insn.size
            self._mark(st.seg, addr, insn.size)
            addr += insn.size

    def _isa_bit(self, expr: str) -> int:
        """1 for the address of MIPS16e code taken as data (``.word f`` /
        ``la`` / ``li``): what the linker does for a MIPS16 symbol, so a
        JALR / JR to it switches the ISA mode."""
        return 1 if expr.strip() in self._m16_labels else 0

    def _put(self, addr: int, value: int, size: int) -> None:
        data = value.to_bytes(size, "big" if self.endian == "big" else "little")
        for i, b in enumerate(data):
            self.prog.image[(addr + i) & M32] = b

    def _mark(self, seg: str, addr: int, size: int) -> None:
        lo, hi = self.prog.segments.get(seg, (addr, addr))
        self.prog.segments[seg] = (min(lo, addr), max(hi, addr + size))

    def _emit_data(self, st: _Stmt) -> None:
        d, addr = st.mnemonic, st.addr
        if d == ".fill16":                      # MIPS16e / microMIPS alignment: NOPs
            from .micromips import NOP16
            from .mips16 import NOP
            for k in range(0, st.size, 2):
                self._put(addr + k, NOP16 if st.mm else NOP, 2)
        elif d == ".space":
            for i in range(st.size):
                self.prog.image[addr + i] = 0
        elif d in (".word", ".half", ".byte"):
            size = {".word": 4, ".half": 2, ".byte": 1}[d]
            for i, op in enumerate(st.ops):
                v = self._value(op, st) | (self._isa_bit(op) if size == 4 else 0)
                if not -(1 << (8 * size - 1)) <= v < (1 << (8 * size)):
                    raise AsmError(st.line, f"value {v} does not fit {d}", st.text)
                self._put(addr + i * size, v & ((1 << (8 * size)) - 1), size)
        elif d in (".float", ".double"):
            fmt = ("f" if d == ".float" else "d")
            fmt = (">" if self.endian == "big" else "<") + fmt
            size = struct.calcsize(fmt)
            for i, op in enumerate(st.ops):
                for j, b in enumerate(struct.pack(fmt, float(op))):
                    self.prog.image[addr + i * size + j] = b
        elif d in (".ascii", ".asciiz"):
            data = self._strings(_strip_comment(st.text).split(d, 1)[1], st.line,
                                 st.text, d == ".asciiz")
            for i, b in enumerate(data):
                self.prog.image[addr + i] = b
        self._mark(st.seg, addr, st.size)

    # -- encoding one real instruction -------------------------------------------

    def _encode(self, mn: str, ops: list[str], addr: int, st: _Stmt) -> int:
        e = self._by_name[mn]
        spec = operand_spec(e)
        word = e["decode"]["value"]
        name = e["name"]

        def err(msg: str) -> AsmError:
            return AsmError(st.line, f"{mn}: {msg}", st.text)

        if name in ("EXT", "INS"):
            if len(ops) != 4:
                raise err("expects rt, rs, pos, size")
            pos, size = self._value(ops[2], st), self._value(ops[3], st)
            if not (0 <= pos <= 31 and 1 <= size <= 32 and pos + size <= 32):
                raise err(f"bad pos {pos} / size {size}")
            hi = size - 1 if name == "EXT" else pos + size - 1
            return (word | self._operand("rt", ops[0], addr, st, err)
                    | self._operand("rs", ops[1], addr, st, err)
                    | (hi << 11) | (pos << 6))
        if name == "BREAK":
            if len(ops) > 2:
                raise err("takes at most 2 codes")
            codes = [self._value(o, st) for o in ops] + [0, 0]
            for c in codes[:2]:
                if not 0 <= c < 1024:
                    raise err("code must be 0..1023")
            return word | (codes[0] << 16) | (codes[1] << 6)

        items = [kind_of(s) for s in spec]
        n_req = sum(1 for _, opt in items if not opt)
        n_opt = len(items) - n_req
        extra = len(ops) - n_req
        if not 0 <= extra <= n_opt:
            want = f"{n_req}" + (f"..{n_req + n_opt}" if n_opt else "")
            raise err(f"expects {want} operands, got {len(ops)}")
        it = iter(ops)
        for kind, optional in items:
            if optional:
                if extra <= 0:
                    word |= self._default(kind, name)
                    continue
                extra -= 1
            word |= self._operand(kind, next(it), addr, st, err)
        if name in ("CLO", "CLZ"):                     # rt = rd
            word |= ((word >> 11) & 31) << 16
        word = self._r6_fields(name, word, err)
        if name in ("BGEZAL", "BLTZAL", "BGEZALL", "BLTZALL") and (word >> 21) & 31 == 31:
            raise err("rs must not be $ra (the link register)")
        if name.startswith("JALR") and (word >> 11) & 31 == (word >> 21) & 31:
            raise err("rd and rs must be different")
        return word

    @staticmethod
    def _r6_fields(name: str, word: int, err) -> int:
        """Release 6 compact branches whose encodings differ only by how
        rs compares with rt: fill / order the fields as GNU as does, and
        reject operands the encoding cannot express."""
        rs, rt = (word >> 21) & 31, (word >> 16) & 31
        if name in ("BGEZALC", "BLTZALC", "BGEZC", "BLTZC"):
            word |= rt << 21                        # register in rs and rt
        elif name in ("BEQC", "BNEC") and rs > rt or \
                name in ("BOVC", "BNVC") and rs < rt:
            word = (word & ~(0x3FF << 16)) | (rt << 21) | (rs << 16)   # swap
        from .isa import _constraint_ok
        entry = ISA.for_release("R6").get(name) if "@" not in name else None
        if entry is not None and not _constraint_ok(entry, word):
            raise err("these registers cannot be encoded (e.g. $zero, or "
                      "the same register twice)")
        return word

    @staticmethod
    def _default(kind: str, name: str) -> int:
        if kind == "rd" and name.startswith("JALR"):
            return 31 << 11
        return 0

    def _operand(self, kind: str, tok: str, addr: int, st: _Stmt, err) -> int:
        def reg(parse, what: str) -> int:
            r = parse(tok)
            if r is None:
                raise err(f"{what} expected, got {tok!r}")
            return r

        def num(lo: int, hi: int, what: str) -> int:
            v = self._value(tok, st)
            if not lo <= v <= hi:
                raise err(f"{what} {v} out of range {lo}..{hi}")
            return v

        lsb = KINDS[kind].fields[0][0]
        if kind in ("rs", "rt", "rd"):
            return reg(parse_gpr, "GPR") << lsb
        if kind in ("fs", "ft", "fd", "fr"):
            return reg(parse_fpr, "FPR") << lsb
        if kind in ("c0rd", "fcr", "hwr", "c2rt"):
            r = parse_numreg(tok)
            if r is None:
                r = num(0, 31, "register number")
            return r << lsb
        if kind in ("sa", "hint", "hint11", "stype"):
            return num(0, 31, kind) << lsb
        if kind == "sel":
            return num(0, 7, "select")
        if kind == "sel6":
            return num(0, 7, "select") << 6
        if kind == "rsd":
            return reg(parse_gpr, "GPR") << 21
        if kind in ("br21", "br26"):
            bits = 21 if kind == "br21" else 26
            target = self._value(tok, st) & M32
            delta = _sx32(target - (addr + 4))
            if delta & 3:
                raise err(f"branch target 0x{target:08x} not word-aligned")
            if not -(1 << (bits + 1)) <= delta < (1 << (bits + 1)):
                raise err(f"branch target 0x{target:08x} out of range")
            return (delta >> 2) & ((1 << bits) - 1)
        if kind == "pc19":                  # an address, PC-relative (GNU as)
            delta = _sx32((self._value(tok, st) & M32) - addr)
            if delta & 3 or not -(1 << 20) <= delta < (1 << 20):
                raise err(f"address {tok} not reachable PC-relative (+-1 MB, x4)")
            return (delta >> 2) & 0x7FFFF
        if kind == "sa2":
            return (num(1, 4, "shift") - 1) << 6
        if kind == "bp":
            return num(0, 3, "byte position") << 6
        if kind == "code16":
            return num(0, 0xFFFF, "code")
        if kind == "mem11":
            m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", tok.strip())
            if not m or parse_gpr(m.group(2)) is None:
                raise err(f"offset(base) expected, got {tok!r}")
            off = self._value(m.group(1), st) if m.group(1).strip() else 0
            if not -1024 <= off <= 1023:
                raise err(f"11-bit offset {off} out of range")
            return (off & 0x7FF) | (parse_gpr(m.group(2)) << 11)
        if kind == "c2cc":
            m = re.fullmatch(r"\$cc([0-7])", tok.strip())
            return (int(m.group(1)) if m else num(0, 7, "COP2 condition")) << lsb
        if kind == "tcode":
            return num(0, 1023, "code") << 6
        if kind == "code20":
            return num(0, (1 << 20) - 1, "code") << 6
        if kind == "code19":
            return num(0, (1 << 19) - 1, "code") << 6
        if kind == "imm":
            return num(-0x8000, 0x7FFF, "immediate") & 0xFFFF
        if kind == "uimm":
            return num(0, 0xFFFF, "immediate")
        if kind in ("impl16",):
            return num(0, 0xFFFF, "impl")
        if kind == "cofun":
            return num(0, (1 << 25) - 1, "cofun")
        if kind in ("cc18", "cc8"):
            cc = parse_fcc(tok)
            if cc is None:
                raise err(f"FPU condition code expected, got {tok!r}")
            return cc << lsb
        if kind in ("mem", "mem9", "idx"):
            m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", tok.strip())
            if not m:
                raise err(f"offset(base) expected, got {tok!r}")
            base = parse_gpr(m.group(2))
            if base is None:
                raise err(f"base register expected, got {m.group(2)!r}")
            if kind == "idx":
                index = parse_gpr(m.group(1))
                if index is None:
                    raise err(f"index register expected, got {m.group(1)!r}")
                return (index << 16) | (base << 21)
            off = self._value(m.group(1), st) if m.group(1).strip() else 0
            if kind == "mem":
                if not -0x8000 <= off <= 0x7FFF:
                    raise err(f"offset {off} out of range")
                return (off & 0xFFFF) | (base << 21)
            if not -256 <= off <= 255:
                raise err(f"9-bit offset {off} out of range")
            return ((off & 0x1FF) << 7) | (base << 21)
        if kind == "br":
            target = self._value(tok, st) & M32
            delta = _sx32(target - (addr + 4))
            if delta & 3:
                raise err(f"branch target 0x{target:08x} not word-aligned")
            if not -0x20000 <= delta <= 0x1FFFC:
                raise err(f"branch target 0x{target:08x} out of range")
            return (delta >> 2) & 0xFFFF
        if kind == "jt":
            target = self._value(tok, st) & M32
            if target & 3:
                raise err(f"jump target 0x{target:08x} not word-aligned")
            if (target ^ (addr + 4)) & 0xF0000000:
                raise err(f"jump target 0x{target:08x} outside the 256 MB region")
            return (target >> 2) & 0x3FFFFFF
        raise err(f"internal: unknown operand kind {kind}")


def assemble(source: str, *, endian: str = "big", isa: ISA | None = None,
             bases: dict[str, int] | None = None,
             release: str | None = None, pseudo: bool = True,
             reorder: bool = True) -> Program:
    """Assemble ``source`` for a MIPS32 ``release`` (default R5); raises
    :class:`AsmError` on the first error.  ``pseudo=False``: strict mode;
    ``reorder=False``: start in ``.set noreorder``."""
    return Assembler(endian=endian, isa=isa, bases=bases, release=release,
                     pseudo=pseudo, reorder=reorder).assemble(source)
