"""RISC-V instructions: the table, the decoder, the disassembler (PLAN M25
phase 1b).

The rows are ``emips/data/riscv.json``'s (generated from RISC-V
International's riscv-opcodes; never edited by hand): per XLEN the
instructions of RV32I / RV64I, M, A, F, D, C, Zicsr, Zifencei, Zba, Zbb
and the privileged ones, each a mask / match and its operand arguments.
The normative text is The RISC-V Instruction Set Manual, Volume I:
Unprivileged Architecture, 20250508; where a section is cited, it is
that manual's.

- :func:`size_of` — the length of an instruction from its first halfword
  (1.5 "Base Instruction-Length Encoding": bits 1:0 != 11 → 16 bits, else
  bits 4:2 != 111 → 32 bits; the longer encodings are reserved here).
- :class:`RVISA` — the rows one core decodes: an XLEN and a set of
  extensions (``RVISA.for_core(64, "imafdc_zicsr_zifencei_zba_zbb")``).
  The most specific matching row wins (c.nop before c.addi, c.jr before
  c.mv, c.addi16sp before c.lui); no two rows of one ISA match a word
  with the same specificity (checked when the ISA is built).
- :meth:`RVISA.decode` → :class:`RV`: the row, the size, the raw bits and
  the operand values ``f`` (registers ``rd`` / ``rs1`` / ``rs2`` / ``rs3``,
  immediates assembled and sign-extended as 2.3 / 2.4 and 28.3 (the C
  formats) define them: ``imm`` (a U-type's already shifted by 12),
  ``shamt``, ``csr``, ``zimm``, ``rm``, ``aq`` / ``rl``, ``pred`` /
  ``succ`` / ``fm``).  A compressed instruction carries its expansion
  (decision 5, 28.4-28.7: "expands to"): ``i.base`` is the 32-bit row it
  is, ``i.f`` the expansion's operands (``c.addi16sp`` = ``addi`` with
  ``rd`` = ``rs1`` = 2).  Reserved encodings raise :class:`RVDecodeError`
  (an illegal instruction); HINTs decode as their instruction (they
  write nothing: ``c.li x0, 5`` = ``addi x0, x0, 5``) — :data:`ZERO_RULES`.
- :func:`text` / :class:`Disassembler` — objdump's text (decision 4):
  binutils 2.42's table walked as ``riscv-dis.c`` does
  (``riscv_disassemble_insn``: from the first entry of the word's bucket,
  the first entry whose match function, XLEN and extension class fit;
  aliases unless ``aliases=False`` = ``-M no-aliases``; ABI register names
  unless ``numeric=True`` = ``-M numeric``; ``print_insn_args``' operand
  letters), operands separated by ", " (eMIPS's convention; objdump
  writes ","), targets ``0x...``.  :class:`Disassembler` keeps objdump's
  state between instructions: a ``lui`` / ``auipc`` / ``c.lui`` remembers
  its register so that the next ``addi`` / load / ``jalr`` on it gets the
  address as a comment (``addi a0, a0, 16 # 0x10010010``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from .isa import ISADecodeError

_DATA = json.loads((Path(__file__).parent / "data" / "riscv.json").read_text())

#: the integer registers' ABI names (psABI 1.0 ch. 1; objdump's, s0 not fp)
GPR_NAMES = ("zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
             "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
             "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
             "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6")
#: the floating-point registers' ABI names
FPR_NAMES = ("ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
             "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
             "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
             "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11")
#: the rounding-mode field (14.2 Table 14.1; objdump's names)
RM_NAMES = ("rne", "rtz", "rdn", "rup", "rmm", None, None, "dyn")
#: FENCE predecessor / successor sets (2.7: I O R W from bit 3 down)
PRED_SUCC = (None, "w", "r", "rw", "o", "ow", "or", "orw",
             "i", "iw", "ir", "irw", "io", "iow", "ior", "iorw")
#: CSR number -> the name objdump prints (binutils 2.42, privileged 1.12)
CSR_NAMES = {int(k): v for k, v in _DATA["binutils"]["csrs"].items()}

#: an operand argument of riscv-opcodes -> (the value it makes, the
#: instruction bits (hi, lo) and the value bit they start at, the value's
#: sign width when the argument completes a signed immediate, a register
#: offset) — 2.3 Figure 2.4 / 28.3 Table 28.x.  Arguments that make two
#: values (``rd_rs1_p``) list both names.
_ARGS: dict[str, tuple[tuple[str, ...], tuple[tuple[int, int, int], ...], int, int]] = {
    "rd": (("rd",), ((11, 7, 0),), 0, 0),
    "rs1": (("rs1",), ((19, 15, 0),), 0, 0),
    "rs2": (("rs2",), ((24, 20, 0),), 0, 0),
    "rs3": (("rs3",), ((31, 27, 0),), 0, 0),
    "rm": (("rm",), ((14, 12, 0),), 0, 0),
    "aq": (("aq",), ((26, 26, 0),), 0, 0),
    "rl": (("rl",), ((25, 25, 0),), 0, 0),
    "fm": (("fm",), ((31, 28, 0),), 0, 0),
    "pred": (("pred",), ((27, 24, 0),), 0, 0),
    "succ": (("succ",), ((23, 20, 0),), 0, 0),
    "csr": (("csr",), ((31, 20, 0),), 0, 0),
    "zimm5": (("zimm",), ((19, 15, 0),), 0, 0),
    "shamtw": (("shamt",), ((24, 20, 0),), 0, 0),
    "shamtd": (("shamt",), ((25, 20, 0),), 0, 0),
    # I / S / B / U / J immediates (2.3 Figure 2.4)
    "imm12": (("imm",), ((31, 20, 0),), 12, 0),
    "imm12hi": (("imm",), ((31, 25, 5),), 12, 0),
    "imm12lo": (("imm",), ((11, 7, 0),), 0, 0),
    "bimm12hi": (("imm",), ((31, 31, 12), (30, 25, 5)), 13, 0),
    "bimm12lo": (("imm",), ((11, 8, 1), (7, 7, 11)), 0, 0),
    "imm20": (("imm",), ((31, 12, 12),), 32, 0),
    "jimm20": (("imm",), ((31, 31, 20), (30, 21, 1), (20, 20, 11), (19, 12, 12)), 21, 0),
    # compressed registers: rd' / rs1' / rs2' = x8-x15 (28.2 Table 28.1)
    "rd_p": (("rd",), ((4, 2, 0),), 0, 8),
    "rs1_p": (("rs1",), ((9, 7, 0),), 0, 8),
    "rs2_p": (("rs2",), ((4, 2, 0),), 0, 8),
    "rd_rs1_p": (("rd", "rs1"), ((9, 7, 0),), 0, 8),
    "rd_rs1_n0": (("rd", "rs1"), ((11, 7, 0),), 0, 0),
    "rd_n0": (("rd",), ((11, 7, 0),), 0, 0),
    "rd_n2": (("rd",), ((11, 7, 0),), 0, 0),
    "rs1_n0": (("rs1",), ((11, 7, 0),), 0, 0),
    "c_rs1_n0": (("rs1",), ((11, 7, 0),), 0, 0),
    "c_rs2_n0": (("rs2",), ((6, 2, 0),), 0, 0),
    "c_rs2": (("rs2",), ((6, 2, 0),), 0, 0),
    # compressed immediates (28.3-28.6)
    "c_nzuimm10": (("imm",), ((12, 11, 4), (10, 7, 6), (6, 6, 2), (5, 5, 3)), 0, 0),
    "c_uimm7lo": (("imm",), ((6, 6, 2), (5, 5, 6)), 0, 0),
    "c_uimm7hi": (("imm",), ((12, 10, 3),), 0, 0),
    "c_uimm8lo": (("imm",), ((6, 5, 6),), 0, 0),
    "c_uimm8hi": (("imm",), ((12, 10, 3),), 0, 0),
    "c_nzimm6lo": (("imm",), ((6, 2, 0),), 0, 0),
    "c_nzimm6hi": (("imm",), ((12, 12, 5),), 6, 0),
    "c_imm6lo": (("imm",), ((6, 2, 0),), 0, 0),
    "c_imm6hi": (("imm",), ((12, 12, 5),), 6, 0),
    "c_nzimm10hi": (("imm",), ((12, 12, 9),), 10, 0),
    "c_nzimm10lo": (("imm",), ((6, 6, 4), (5, 5, 6), (4, 3, 7), (2, 2, 5)), 0, 0),
    "c_nzimm18hi": (("imm",), ((12, 12, 17),), 18, 0),
    "c_nzimm18lo": (("imm",), ((6, 2, 12),), 0, 0),
    "c_imm12": (("imm",), ((12, 12, 11), (11, 11, 4), (10, 9, 8), (8, 8, 10), (7, 7, 6),
                           (6, 6, 7), (5, 3, 1), (2, 2, 5)), 12, 0),
    "c_bimm9lo": (("imm",), ((6, 5, 6), (4, 3, 1), (2, 2, 5)), 0, 0),
    "c_bimm9hi": (("imm",), ((12, 12, 8), (11, 10, 3)), 9, 0),
    "c_nzuimm5": (("shamt",), ((6, 2, 0),), 0, 0),
    "c_nzuimm6lo": (("shamt",), ((6, 2, 0),), 0, 0),
    "c_nzuimm6hi": (("shamt",), ((12, 12, 5),), 0, 0),
    "c_uimm8splo": (("imm",), ((6, 4, 2), (3, 2, 6)), 0, 0),
    "c_uimm8sphi": (("imm",), ((12, 12, 5),), 0, 0),
    "c_uimm8sp_s": (("imm",), ((12, 9, 2), (8, 7, 6)), 0, 0),
    "c_uimm9splo": (("imm",), ((6, 5, 3), (4, 2, 6)), 0, 0),
    "c_uimm9sphi": (("imm",), ((12, 12, 5),), 0, 0),
    "c_uimm9sp_s": (("imm",), ((12, 10, 3), (9, 7, 6)), 0, 0),
    # the P8700's custom instructions (MD01502 App. B; binutils 2.45's
    # include/opcode/riscv.h EXTRACT_MIPS_*): rd2 of LDP / LWP, the pairs'
    # unsigned offsets, PREF's hint and 9-bit unsigned offset
    "mips_rd2": (("rd2",), ((31, 27, 0),), 0, 0),
    "mips_ldp_imm": (("imm",), ((26, 23, 3),), 0, 0),
    "mips_lwp_imm": (("imm",), ((26, 22, 2),), 0, 0),
    "mips_sdp_imm_hi": (("imm",), ((26, 25, 5),), 0, 0),
    "mips_sdp_imm_lo": (("imm",), ((11, 10, 3),), 0, 0),
    "mips_swp_imm_hi": (("imm",), ((26, 25, 5),), 0, 0),
    "mips_swp_imm_lo": (("imm",), ((11, 9, 2),), 0, 0),
    "mips_hint": (("hint",), ((11, 7, 0),), 0, 0),
    "mips_imm9": (("imm",), ((28, 20, 0),), 0, 0),
    "mips_level": (("level",), ((22, 20, 0),), 0, 0),
}

#: what a zero in a "nonzero" argument means (28.3-28.8 and Table 28.x
#: "RVC HINT instructions"): "reserved" = an illegal instruction, "hint" =
#: the instruction itself (it changes nothing).  A rule names the row and
#: the value that must not be zero.
ZERO_RULES: dict[tuple[str, str], str] = {
    ("c.addi4spn", "imm"): "reserved",          # 28.5: nzuimm = 0 reserved
    ("c.addi16sp", "imm"): "reserved",          # 28.5: nzimm = 0 reserved
    ("c.lui", "imm"): "reserved",               # 28.5: nzimm = 0 reserved
    ("c.lwsp", "rd"): "reserved",               # 28.3: rd = x0 reserved
    ("c.ldsp", "rd"): "reserved",
    ("c.jr", "rs1"): "reserved",                # 28.4: rs1 = x0 reserved
    ("c.addiw", "rd"): "reserved",              # 28.5: rd = x0 reserved
    ("c.lui", "rd"): "hint",                    # Table 28.x: rd = x0 HINT
    ("c.li", "rd"): "hint",
    ("c.addi", "rd"): "hint",
    ("c.addi", "imm"): "hint",                  # c.addi rd, 0
    ("c.nop", "imm"): None,                     # c.nop with imm != 0: a HINT
    ("c.mv", "rd"): "hint",
    ("c.add", "rd"): "hint",
    ("c.slli", "rd"): "hint",
    ("c.slli", "shamt"): "hint",                # shamt = 0 (c.slli64)
    ("c.srli", "shamt"): "hint",
    ("c.srai", "shamt"): "hint",
}

#: every compressed row -> (the 32-bit instruction it expands to, the
#: operands the expansion fixes) — 28.3-28.8
EXPAND: dict[str, tuple[str, dict[str, int]]] = {
    "c.addi4spn": ("addi", {"rs1": 2}),
    "c.fld": ("fld", {}), "c.lw": ("lw", {}), "c.flw": ("flw", {}), "c.ld": ("ld", {}),
    "c.fsd": ("fsd", {}), "c.sw": ("sw", {}), "c.fsw": ("fsw", {}), "c.sd": ("sd", {}),
    "c.nop": ("addi", {"rd": 0, "rs1": 0}),
    "c.addi": ("addi", {}),
    "c.jal": ("jal", {"rd": 1}),
    "c.addiw": ("addiw", {}),
    "c.li": ("addi", {"rs1": 0}),
    "c.addi16sp": ("addi", {"rd": 2, "rs1": 2}),
    "c.lui": ("lui", {}),
    "c.srli": ("srli", {}), "c.srai": ("srai", {}), "c.andi": ("andi", {}),
    "c.sub": ("sub", {}), "c.xor": ("xor", {}), "c.or": ("or", {}), "c.and": ("and", {}),
    "c.subw": ("subw", {}), "c.addw": ("addw", {}),
    "c.j": ("jal", {"rd": 0}),
    "c.beqz": ("beq", {"rs2": 0}), "c.bnez": ("bne", {"rs2": 0}),
    "c.slli": ("slli", {}),
    "c.fldsp": ("fld", {"rs1": 2}), "c.lwsp": ("lw", {"rs1": 2}),
    "c.flwsp": ("flw", {"rs1": 2}), "c.ldsp": ("ld", {"rs1": 2}),
    "c.jr": ("jalr", {"rd": 0, "imm": 0}),
    "c.mv": ("add", {"rs1": 0}),
    "c.ebreak": ("ebreak", {}),
    "c.jalr": ("jalr", {"rd": 1, "imm": 0}),
    "c.add": ("add", {}),
    "c.fsdsp": ("fsd", {"rs1": 2}), "c.swsp": ("sw", {"rs1": 2}),
    "c.fswsp": ("fsw", {"rs1": 2}), "c.sdsp": ("sd", {"rs1": 2}),
}

#: the extensions every core has (the privileged base: mret / wfi)
BASE_EXTENSIONS = frozenset({"i", "priv"})
#: what an extension brings with it (binutils' implicit subsets: the
#: classes of its table are tested on the closure)
IMPLIED = {"g": ("i", "m", "a", "f", "d", "zicsr", "zifencei"), "m": ("zmmul",),
           "d": ("f",), "f": ("zicsr",), "zicntr": ("zicsr",), "zihpm": ("zicsr",)}


def closure(extensions: Iterable[str]) -> frozenset:
    """The extensions with everything they imply (to a fixpoint)."""
    out = set(extensions)
    todo = list(out)
    while todo:
        for e in IMPLIED.get(todo.pop(), ()):
            if e not in out:
                out.add(e)
                todo.append(e)
    return frozenset(out)


def size_of(h: int) -> int:
    """The instruction's length in bytes from its first halfword (1.5):
    2 or 4; 0 for the reserved longer encodings (48 bits and up)."""
    if h & 3 != 3:
        return 2
    if h & 0x1C != 0x1C:
        return 4
    return 0


class RVDecodeError(ISADecodeError):
    """No instruction of the core (an illegal-instruction exception):
    ``why`` = "reserved" (a reserved encoding of a known instruction),
    "length" (a 48-bit or longer encoding) or "unknown"."""

    def __init__(self, why: str, word: int = 0, size: int = 4, name: str = "") -> None:
        Exception.__init__(self, f"no RISC-V instruction ({why}{': ' + name if name else ''}): "
                                 f"0x{word:0{size * 2}x}")
        self.word = word
        self.why = why
        self.size = size
        self.name = name


@dataclass(frozen=True)
class Row:
    """One instruction of riscv-opcodes (see the module docstring)."""
    name: str
    xlen: tuple[int, ...]
    size: int
    mask: int
    match: int
    args: tuple[str, ...]
    ext: tuple[str, ...]
    file: str
    specificity: int = field(compare=False)

    @property
    def compressed(self) -> bool:
        return self.size == 2


ROWS: tuple[Row, ...] = tuple(
    Row(r["name"], tuple(r["xlen"]), r["size"], r["mask"], r["match"], tuple(r["args"]),
        tuple(r["ext"]), r["file"], bin(r["mask"]).count("1"))
    for r in _DATA["rows"])
NAMES = frozenset(r.name for r in ROWS)
assert set(EXPAND) == {r.name for r in ROWS if r.compressed}, \
    {r.name for r in ROWS if r.compressed} ^ set(EXPAND)


@dataclass(frozen=True)
class RV:
    """A decoded instruction: its row (the compressed one for a C
    instruction), size in bytes, raw bits, the operands ``f`` (the
    expansion's for a C instruction) and ``base``, the 32-bit row it is."""
    row: Row
    size: int
    raw: int
    f: Mapping[str, int]
    base: Row

    @property
    def name(self) -> str:
        return self.row.name


def _fields(row: Row, raw: int) -> dict[str, int]:
    f: dict[str, int] = {}
    sign = 0
    for a in row.args:
        names, segs, width, offset = _ARGS[a]
        v = 0
        for hi, lo, pos in segs:
            v |= ((raw >> lo) & ((1 << (hi - lo + 1)) - 1)) << pos
        for n in names:
            f[n] = f.get(n, 0) | (v + offset if offset else v)
        if width:
            sign = width
    if sign and f["imm"] >> (sign - 1) & 1:
        f["imm"] -= 1 << sign
    return f


class RVISA:
    """The instructions of one core: ``xlen`` 32 / 64 and its extensions
    (letters and Z names; ``s`` = supervisor mode's sret / sfence.vma,
    ``sdext`` = Debug Mode's dret).  Use :meth:`for_core` (cached)."""

    def __init__(self, xlen: int, extensions: Iterable[str]) -> None:
        if xlen not in (32, 64):
            raise ValueError(f"xlen must be 32 or 64, not {xlen!r}")
        exts = set(closure(set(extensions) | BASE_EXTENSIONS))
        self.xlen = xlen
        self.extensions = frozenset(exts)
        self.rows = tuple(r for r in ROWS if xlen in r.xlen and set(r.ext) <= exts)
        self.by_name = {r.name: r for r in self.rows}
        self._buckets: dict[int, list[Row]] = {}
        for r in self.rows:
            for key in self._keys(r):
                self._buckets.setdefault(key, []).append(r)
        for rows in self._buckets.values():
            rows.sort(key=lambda r: -r.specificity)
        for a in self.rows:                     # equal specificity never overlaps
            for b in self.rows:
                if (a is not b and a.size == b.size and a.specificity == b.specificity
                        and (a.match ^ b.match) & a.mask & b.mask == 0):
                    raise ValueError(f"{a.name} and {b.name} overlap (xlen {xlen})")
        self._cache: dict[tuple[int, int], RV] = {}

    @staticmethod
    def _key(raw: int, size: int) -> int:
        """The bucket: the quadrant + funct3 of a 16-bit word, the opcode of
        a 32-bit one."""
        return (raw & 3) | ((raw >> 11) & 0x1C) if size == 2 else 0x100 | (raw & 0x7F)

    def _keys(self, r: Row) -> list[int]:
        if r.size == 2:
            base = r.match & 3
            known = (r.mask >> 13) & 7
            return [base | (f3 << 2) for f3 in range(8) if (f3 ^ (r.match >> 13)) & known == 0]
        return [0x100 | op for op in range(128) if (op ^ r.match) & r.mask & 0x7F == 0]

    @classmethod
    @lru_cache(maxsize=None)
    def for_core(cls, xlen: int, extensions: str = "imafdc_zicsr_zifencei_zicntr") -> "RVISA":
        """The ISA of an ``extensions`` string (``imafdc_zicsr_zba``: single
        letters, then Z names separated by ``_``; ``g`` = imafd_zicsr_
        zifencei)."""
        return cls(xlen, parse_extensions(extensions))

    def decode(self, raw: int, size: int | None = None) -> RV:
        """Decode ``raw`` (a 16-bit or a 32-bit instruction; the 32-bit
        one as ``first halfword | second << 16``)."""
        if size is None:
            size = size_of(raw & 0xFFFF)
        key = (raw, size)
        i = self._cache.get(key)
        if i is not None:
            return i
        if size not in (2, 4):
            raise RVDecodeError("length", raw, 4)
        raw &= (1 << (8 * size)) - 1
        for r in self._buckets.get(self._key(raw, size), ()):
            if raw & r.mask == r.match:
                i = self._make(r, raw)
                break
        else:
            raise RVDecodeError("unknown", raw, size)
        if len(self._cache) > 65536:
            self._cache.clear()
        self._cache[key] = i
        return i

    def _make(self, r: Row, raw: int) -> RV:
        f = _fields(r, raw)
        if not r.compressed:
            return RV(r, 4, raw, f, r)
        for (name, value), what in ZERO_RULES.items():
            if name == r.name and what == "reserved" and f.get(value) == 0:
                raise RVDecodeError("reserved", raw, 2, r.name)
        base_name, fixed = EXPAND[r.name]
        base = self.by_name.get(base_name)
        if base is None:                        # (an ISA without the expansion)
            raise RVDecodeError("unknown", raw, 2, r.name)
        f.update(fixed)
        return RV(r, 2, raw, f, base)


def parse_extensions(text: str) -> frozenset[str]:
    """``imafdc_zicsr_zifencei`` -> {"i", "m", ..., "zicsr", "zifencei"}
    (``g`` expanded; an ``rv32`` / ``rv64`` prefix is ignored)."""
    text = text.lower().strip()
    for p in ("rv32", "rv64"):
        if text.startswith(p):
            text = text[len(p):]
    parts = text.split("_")
    out: set[str] = set()
    for i, p in enumerate(parts):
        if not p:
            continue
        if i == 0 and not p.startswith(("z", "s", "x")):
            for ch in p:
                out |= set(IMPLIED["g"]) if ch == "g" else {ch}
            continue
        out.add(p)
    return frozenset(out)


# ---------------------------------------------------------------- the text

@dataclass(frozen=True)
class _Entry:
    i: int
    name: str
    xlen: int
    classes: tuple[frozenset[str], ...]
    args: str
    match: int
    mask: int
    func: str
    alias: bool


_ENTRIES = tuple(
    _Entry(e["i"], e["name"], e["xlen"], tuple(frozenset(c) for c in e["class"]), e["args"],
           e["match"], e["mask"], e["func"], e["alias"])
    for e in _DATA["binutils"]["entries"])
_START = {int(k): v for k, v in _DATA["binutils"]["start"].items()}


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _x(w: int, s: int, n: int) -> int:
    return (w >> s) & ((1 << n) - 1)


# riscv.h's EXTRACT_* (what objdump prints; equal to _fields by construction)
def _itype(w: int) -> int: return _sx(w >> 20, 12)                                  # noqa: E704
def _stype(w: int) -> int: return _sx(_x(w, 7, 5) | (_x(w, 25, 7) << 5), 12)         # noqa: E704
def _ci(w: int) -> int: return _sx(_x(w, 2, 5) | (_x(w, 12, 1) << 5), 6)             # noqa: E704


def _match(e: _Entry, w: int) -> bool:
    if (w ^ e.match) & e.mask:
        return False
    fn = e.func
    if fn == "match_opcode":
        return True
    rd = _x(w, 7, 5)
    if fn == "match_rd_nonzero":
        return rd != 0
    if fn == "match_rs1_eq_rs2":
        return _x(w, 15, 5) == _x(w, 20, 5)
    if fn == "match_c_add":
        return rd != 0 and _x(w, 2, 5) != 0
    if fn == "match_c_add_with_hint":
        return _x(w, 2, 5) != 0
    if fn == "match_c_nop":
        return rd == 0
    if fn == "match_c_addi16sp":
        return rd == 2
    if fn == "match_c_lui":
        return rd != 0 and rd != 2 and _ci(w) != 0
    if fn == "match_c_lui_with_hint":
        return rd != 2 and _ci(w) != 0
    if fn == "match_c_addi4spn":
        return (_x(w, 6, 1) << 2 | _x(w, 5, 1) << 3 | _x(w, 11, 2) << 4 | _x(w, 7, 4) << 6) != 0
    if fn in ("match_c_slli", "match_srxi_as_c_srxi"):
        return _ci(w) != 0
    if fn == "match_slli_as_c_slli":
        return rd != 0 and _ci(w) != 0
    if fn == "match_c_slli64":
        return _ci(w) == 0
    raise AssertionError(fn)


class Disassembler:
    """objdump's disassembler for one core (see the module docstring):
    ``text(raw, size, pc)`` for one instruction after another, keeping the
    ``lui`` / ``auipc`` state of ``riscv-dis.c``; :meth:`reset` forgets it
    (objdump: a new section)."""

    def __init__(self, xlen: int, extensions: Iterable[str] | str = "imafdc_zicsr_zifencei_zicntr",
                 *, aliases: bool = True, numeric: bool = False,
                 symbols: Mapping[int, str] | None = None) -> None:
        exts = parse_extensions(extensions) if isinstance(extensions, str) else set(extensions)
        exts = closure(set(exts) | {"i"})
        self.xlen = xlen
        self.extensions = frozenset(exts)
        self.aliases = aliases
        self.numeric = numeric
        self.symbols = symbols
        self._entries = tuple(e for e in _ENTRIES
                              if (e.xlen in (0, xlen)) and (aliases or not e.alias)
                              and any(c <= exts for c in e.classes))
        self.reset()

    def reset(self) -> None:
        self.hi: dict[int, int] = {}

    def _gpr(self, n: int) -> str:
        return f"x{n}" if self.numeric else GPR_NAMES[n]

    def _fpr(self, n: int) -> str:
        return f"f{n}" if self.numeric else FPR_NAMES[n]

    def _addr(self, a: int) -> str:
        a &= (1 << self.xlen) - 1
        if self.symbols and a in self.symbols:
            return self.symbols[a]
        return f"0x{a:x}"

    def text(self, raw: int, size: int | None = None, pc: int = 0) -> str:
        """The instruction ``raw`` at ``pc`` (``size`` from its bits when
        not given); a word no entry matches prints as objdump's ``.insn``."""
        if size is None:
            size = size_of(raw & 0xFFFF) or 4
        raw &= (1 << (8 * size)) - 1
        bucket = raw & (0x3 if size == 2 else 0x7F)
        start = _START.get(bucket)
        if start is not None:
            for e in self._entries:
                if e.i < start or not _match(e, raw):
                    continue
                return self._render(e, raw, pc)
        halves = [raw >> 16, raw & 0xFFFF] if size == 4 else [raw]
        while halves and halves[0] == 0:        # objdump skips leading zero halfwords
            halves.pop(0)
        return f".insn {size}, 0x" + "".join(f"{h:04x}" for h in halves)

    # riscv-dis.c: maybe_print_address
    def _maybe(self, note: list, base: int, offset: int, wide: bool) -> None:
        if base in self.hi:
            hi = self.hi.pop(base)
            a = (hi if base else 0) + offset
        elif base == 4 or base == 0:            # X_TP or x0
            a = offset
        else:
            return
        if wide:
            a = _sx(a, 32)
        if self.xlen == 32:
            a &= 0xFFFFFFFF
        note.append(a)

    def _render(self, e: _Entry, w: int, pc: int) -> str:
        out: list[str] = []
        note: list[int] = []
        rs1, rd = _x(w, 15, 5), _x(w, 7, 5)
        args = e.args
        k = 0
        is_jalr = w & 0x707F == 0x67
        while k < len(args):
            ch = args[k]
            if ch == "C":
                k += 1
                c = args[k]
                if c in "sw":
                    out.append(self._gpr(_x(w, 7, 3) + 8))
                elif c in "tx":
                    out.append(self._gpr(_x(w, 2, 3) + 8))
                elif c == "U":
                    out.append(self._gpr(rd))
                elif c == "c":
                    out.append(self._gpr(2))
                elif c == "V":
                    out.append(self._gpr(_x(w, 2, 5)))
                elif c in "oj":
                    imm = _ci(w)
                    if w & 0xE003 == 0x0001 and rd != 0:            # c.addi
                        self._maybe(note, rd, imm, False)
                    if self.xlen == 64 and w & 0xE003 == 0x2001 and rd != 0:   # c.addiw
                        self._maybe(note, rd, imm, True)
                    out.append(str(imm))
                elif c == "k":
                    out.append(str(_x(w, 6, 1) << 2 | _x(w, 10, 3) << 3 | _x(w, 5, 1) << 6))
                elif c == "l":
                    out.append(str(_x(w, 10, 3) << 3 | _x(w, 5, 2) << 6))
                elif c == "m":
                    out.append(str(_x(w, 4, 3) << 2 | _x(w, 12, 1) << 5 | _x(w, 2, 2) << 6))
                elif c == "n":
                    out.append(str(_x(w, 5, 2) << 3 | _x(w, 12, 1) << 5 | _x(w, 2, 3) << 6))
                elif c == "K":
                    out.append(str(_x(w, 6, 1) << 2 | _x(w, 5, 1) << 3 | _x(w, 11, 2) << 4
                                   | _x(w, 7, 4) << 6))
                elif c == "L":
                    out.append(str(_sx(_x(w, 6, 1) << 4 | _x(w, 2, 1) << 5 | _x(w, 5, 1) << 6
                                       | _x(w, 3, 2) << 7 | _x(w, 12, 1) << 9, 10)))
                elif c == "M":
                    out.append(str(_x(w, 9, 4) << 2 | _x(w, 7, 2) << 6))
                elif c == "N":
                    out.append(str(_x(w, 10, 3) << 3 | _x(w, 7, 3) << 6))
                elif c == "p":
                    off = _sx(_x(w, 3, 2) << 1 | _x(w, 10, 2) << 3 | _x(w, 2, 1) << 5
                              | _x(w, 5, 2) << 6 | _x(w, 12, 1) << 8, 9)
                    out.append(self._addr(pc + off))
                elif c == "a":
                    off = _sx(_x(w, 3, 3) << 1 | _x(w, 11, 1) << 4 | _x(w, 2, 1) << 5
                              | _x(w, 7, 1) << 6 | _x(w, 6, 1) << 7 | _x(w, 9, 2) << 8
                              | _x(w, 8, 1) << 10 | _x(w, 12, 1) << 11, 12)
                    out.append(self._addr(pc + off))
                elif c == "u":
                    out.append(f"0x{_ci(w) & 0xFFFFF:x}")
                elif c == ">":
                    out.append(f"0x{_ci(w) & 0x3F:x}")
                elif c == "<":
                    out.append(f"0x{_ci(w) & 0x1F:x}")
                elif c == "T":
                    out.append(self._fpr(_x(w, 2, 5)))
                elif c == "D":
                    out.append(self._fpr(_x(w, 2, 3) + 8))
                else:
                    raise AssertionError(f"C{c}")
            elif ch in ",()":
                out.append(ch)
            elif ch == "0":
                if k == len(args) - 1:
                    out.append("0")
            elif ch == "s":
                if is_jalr:
                    self._maybe(note, rs1, _itype(w), False)
                out.append(self._gpr(rs1))
            elif ch == "t":
                out.append(self._gpr(_x(w, 20, 5)))
            elif ch == "u":
                out.append(f"0x{(w >> 12) & 0xFFFFF:x}")
            elif ch == "m":
                out.append(RM_NAMES[_x(w, 12, 3)] or "unknown")
            elif ch == "P":
                out.append(PRED_SUCC[_x(w, 24, 4)] or "unknown")
            elif ch == "Q":
                out.append(PRED_SUCC[_x(w, 20, 4)] or "unknown")
            elif ch in "oj":
                if ch == "o":
                    self._maybe(note, rs1, _itype(w), False)
                if (w & 0x707F == 0x13 and rs1 != 0) or is_jalr:     # addi / jalr
                    self._maybe(note, rs1, _itype(w), False)
                if self.xlen == 64 and w & 0x707F == 0x1B and rs1 != 0:   # addiw
                    self._maybe(note, rs1, _itype(w), True)
                out.append(str(_itype(w)))
            elif ch == "q":
                self._maybe(note, rs1, _stype(w), False)
                out.append(str(_stype(w)))
            elif ch == "a":
                off = _sx(_x(w, 21, 10) << 1 | _x(w, 20, 1) << 11 | _x(w, 12, 8) << 12
                          | _x(w, 31, 1) << 20, 21)
                out.append(self._addr(pc + off))
            elif ch == "p":
                off = _sx(_x(w, 8, 4) << 1 | _x(w, 25, 6) << 5 | _x(w, 7, 1) << 11
                          | _x(w, 31, 1) << 12, 13)
                out.append(self._addr(pc + off))
            elif ch == "d":
                if w & 0x7F == 0x17:                                    # auipc
                    self.hi[rd] = pc + _sx(w & 0xFFFFF000, 32)
                elif w & 0x7F == 0x37:                                  # lui
                    self.hi[rd] = _sx(w & 0xFFFFF000, 32)
                elif w & 0xE003 == 0x6001:                              # c.lui
                    self.hi[rd] = _ci(w) << 12
                out.append(self._gpr(rd))
            elif ch == "z":
                out.append(self._gpr(0))
            elif ch == ">":
                out.append(f"0x{_x(w, 20, 6):x}")
            elif ch == "<":
                out.append(f"0x{_x(w, 20, 5):x}")
            elif ch in "SU":
                out.append(self._fpr(rs1))
            elif ch == "T":
                out.append(self._fpr(_x(w, 20, 5)))
            elif ch == "D":
                out.append(self._fpr(rd))
            elif ch == "R":
                out.append(self._fpr(_x(w, 27, 5)))
            elif ch == "E":
                csr = _x(w, 20, 12)
                out.append(CSR_NAMES.get(csr) or f"0x{csr:x}")
            elif ch == "Z":
                out.append(str(rs1))
            elif ch == "r":                                     # (2.45: rs3 as an x register)
                out.append(self._gpr(_x(w, 27, 5)))
            elif ch == "X" and args[k + 1:k + 2] == "m":        # 2.45: the MIPS operands
                k += 2
                c = args[k]
                if c == "@":
                    out.append(f"0x{_x(w, 7, 5):x}")
                elif c == "#":
                    out.append(f"0x{_x(w, 20, 9):x}")
                elif c == "$":
                    out.append(str(_x(w, 23, 4) << 3))
                elif c == "%":
                    out.append(str(_x(w, 22, 5) << 2))
                elif c == "^":
                    out.append(str(_x(w, 25, 2) << 5 | _x(w, 10, 2) << 3))
                elif c == "&":
                    out.append(str(_x(w, 25, 2) << 5 | _x(w, 9, 3) << 2))
                elif c == "*":                                  # (MTLBWR's level: MD01502)
                    out.append(str(_x(w, 20, 3)))
                else:
                    raise AssertionError(f"Xm{c}")
            else:
                raise AssertionError(f"operand {ch!r} of {e.name}")
            k += 1
        ops = "".join(out).replace(",", ", ")
        s = f"{e.name} {ops}" if ops else e.name
        if note:
            s += f" # {self._addr(note[-1])}"
        return s


def text(i: RV | int, pc: int = 0, *, xlen: int = 64,
         extensions: Iterable[str] | str = "imafdc_zicsr_zifencei_zicntr_zba_zbb",
         aliases: bool = True, numeric: bool = False) -> str:
    """One instruction's text with no context (no address comments): an
    :class:`RV` or raw bits."""
    raw = i.raw if isinstance(i, RV) else i
    size = i.size if isinstance(i, RV) else None
    return _disassembler(xlen, frozenset(parse_extensions(extensions) if isinstance(extensions, str)
                                         else extensions), aliases, numeric).text(raw, size, pc)


@lru_cache(maxsize=32)
def _disassembler(xlen: int, exts: frozenset[str], aliases: bool, numeric: bool) -> Disassembler:
    class _Stateless(Disassembler):
        def text(self, raw, size=None, pc=0):
            self.hi = {}
            return super().text(raw, size, pc)
    return _Stateless(xlen, exts, aliases=aliases, numeric=numeric)


def encode(row: Row | str, fields: Mapping[str, int], *, xlen: int = 64) -> int:
    """The bits of ``row`` with the operand values ``fields`` (named as
    :class:`RV`'s ``f``; for a compressed row its own operands, before the
    expansion) — for tests; the assembler is phase 3."""
    if isinstance(row, str):
        cand = [r for r in ROWS if r.name == row and xlen in r.xlen]
        if not cand:
            raise KeyError(row)
        row = cand[0]
    raw = row.match
    for a in row.args:
        names, segs, _w, offset = _ARGS[a]
        v = fields[names[0]] - offset
        for hi, lo, pos in segs:
            width = hi - lo + 1
            raw |= ((v >> pos) & ((1 << width) - 1)) << lo
    if raw & row.mask != row.match:
        raise ValueError(f"{row.name}: the operands change fixed bits")
    return raw
