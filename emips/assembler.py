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

    def word(self, addr: int) -> int:
        data = bytes(self.image.get(addr + i, 0) for i in range(4))
        return struct.unpack(">I" if self.endian == "big" else "<I", data)[0]

    def labels(self) -> dict[int, str]:
        """address -> label (first defined), for the disassembler."""
        out: dict[int, str] = {}
        for name, addr in self.symbols.items():
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
            from .tlb import boot_cca, firmware_refill
            cca = boot_cca(machine)                    # 3; a CPS core: 5, coherent
            machine._cp0_set(C.STATUS, C.ST_CU1)
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
                    fw = assemble(firmware_refill(cca), endian=self.endian,
                                  release=machine.release)
                    for addr, byte in fw.image.items():
                        machine.write_byte(addr, byte)
                    machine.firmware = fw
            machine.write_reg(29, BOOT_SP)
            machine.write_reg(28, BOOT_GP)
        for addr, byte in self.image.items():
            try:
                machine.write_byte(addr, byte)
            except UnmappedAddress:
                if boot and machine.mmu.mt in (1, 4):   # the firmware's convention
                    from .tlb import firmware_map
                    pa = firmware_map(addr)[0]
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

    def listing(self, *, abi: bool = True) -> list[tuple[int, int, str, str]]:
        """``(address, word, disassembly, source)`` for every instruction."""
        from .disasm import disassemble
        labels = self.labels()
        isa = ISA.for_release(self.release)
        return [(a, self.word(a), disassemble(self.word(a), a, abi=abi, isa=isa,
                                              symbols=labels), s.text.strip())
                for a, s in sorted(self.source_map.items())]


# ---------------------------------------------------------------- lexing

_LABEL = re.compile(r"^\s*([A-Za-z_.$][\w.$]*)\s*:")
_NAME = re.compile(r"[A-Za-z_.$][\w.$]*$")


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

_DIRECTIVES_IGNORED = {".globl", ".global", ".ent", ".end", ".type", ".size",
                       ".extern", ".file", ".section", ".previous", ".frame",
                       ".mask", ".fmask", ".abicalls", ".option"}


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
        self.symbols: dict[str, int] = {}
        #: names defined by .eqv / .equ / .set NAME (not labels)
        self._consts: set[str] = set()
        self.prog = Program(endian=self.endian, release=self.isa.release)
        stmts = self._pass1(source)
        for st in stmts:
            self._emit(st)
        self.prog.symbols = dict(self.symbols)
        for name in ("main", "__start"):
            if name in self.symbols:
                self.prog.entry = self.symbols[name]
                break
        else:
            self.prog.entry = self.bases["text"]
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
        for n, raw in enumerate(source.splitlines(), start=1):
            text = _strip_comment(raw)
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
                if lc[seg] & 3:
                    raise AsmError(n, "instruction not word-aligned", raw)
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
            lc[seg] += (-lc[seg]) % a
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

    def _put(self, addr: int, value: int, size: int) -> None:
        data = value.to_bytes(size, "big" if self.endian == "big" else "little")
        for i, b in enumerate(data):
            self.prog.image[(addr + i) & M32] = b

    def _mark(self, seg: str, addr: int, size: int) -> None:
        lo, hi = self.prog.segments.get(seg, (addr, addr))
        self.prog.segments[seg] = (min(lo, addr), max(hi, addr + size))

    def _emit_data(self, st: _Stmt) -> None:
        d, addr = st.mnemonic, st.addr
        if d == ".space":
            for i in range(st.size):
                self.prog.image[addr + i] = 0
        elif d in (".word", ".half", ".byte"):
            size = {".word": 4, ".half": 2, ".byte": 1}[d]
            for i, op in enumerate(st.ops):
                v = self._value(op, st)
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
