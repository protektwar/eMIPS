"""Disassembler (PLAN M4): instruction word -> assembly text.

Uses :func:`emips.isa.ISA.decode` to identify the row and the shared
operand table (:mod:`emips.syntax`) to print its operands, so that
:func:`emips.assembler.assemble` reads the text back to the same word.

Syntax: GNU ``as`` / MARS — lower-case mnemonic, ABI register names
(``$t0``; ``abi=False`` prints ``$8``), FPU registers ``$f2``, condition
codes ``$fcc1``, CP0 / FPU-control / hardware registers as ``$N``, signed
decimal immediates (hex for the logical ``andi/ori/xori/lui``), branch
and jump targets as absolute addresses or as symbol names.  Optional
operands are omitted when they hold their default (``jalr $t0`` for
``rd = $ra``, ``c.eq.s $f2, $f4`` for ``cc = 0``).

``style="gas"`` prints the forms GNU ``as`` takes verbatim where MARS
differs: ``div $zero, $a0, $a1`` (``div $a0, $a1`` is a checking macro in
GNU ``as`` but the machine instruction in MARS / SPIM), and always the
condition code of ``movf/movt[.fmt]``.
"""

from __future__ import annotations

from typing import Mapping

from .isa import ISA, ISADecodeError
from .syntax import DEFAULTS, KINDS, gpr_name, kind_of, operand_spec

M32 = 0xFFFFFFFF

def _bits(word: int, lsb: int, width: int) -> int:
    return (word >> lsb) & ((1 << width) - 1)


def _sx(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def branch_target(word: int, pc: int) -> int:
    return (pc + 4 + (_sx(word & 0xFFFF, 16) << 2)) & M32


def jump_target(word: int, pc: int) -> int:
    return ((pc + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2)


def _addr(value: int, symbols: Mapping[int, str] | None) -> str:
    if symbols is not None and value in symbols:
        return symbols[value]
    return f"0x{value:08x}"


def operand_text(kind: str, word: int, pc: int, *, abi: bool = True,
                 symbols: Mapping[int, str] | None = None) -> str:
    """Text of one operand of ``word``."""
    g = lambda lsb: gpr_name(_bits(word, lsb, 5), abi)   # noqa: E731
    if kind in ("rs", "rt", "rd"):
        return g(KINDS[kind].fields[0][0])
    if kind in ("fs", "ft", "fd", "fr"):
        return f"$f{_bits(word, KINDS[kind].fields[0][0], 5)}"
    if kind in ("c0rd", "fcr", "hwr", "c2rt", "trt", "trd"):
        return f"${_bits(word, KINDS[kind].fields[0][0], 5)}"
    if kind in ("sa", "sel", "hint", "hint11", "stype", "tcode", "code20",
                "code19", "sel6"):
        lsb, width = KINDS[kind].fields[0]
        return str(_bits(word, lsb, width))
    if kind == "imm":
        return str(_sx(word & 0xFFFF, 16))
    if kind == "uimm":
        v = word & 0xFFFF
        return f"0x{v:x}" if v > 9 else str(v)
    if kind in ("impl16", "cofun"):
        lsb, width = KINDS[kind].fields[0]
        return f"0x{_bits(word, lsb, width):x}"
    if kind == "mem":
        return f"{_sx(word & 0xFFFF, 16)}({g(21)})"
    if kind == "mem9":
        return f"{_sx(_bits(word, 7, 9), 9)}({g(21)})"
    if kind == "idx":
        return f"{g(16)}({g(21)})"
    if kind == "br":
        return _addr(branch_target(word, pc), symbols)
    if kind in ("br21", "br26"):
        bits = 21 if kind == "br21" else 26
        off = _sx(word & ((1 << bits) - 1), bits) << 2
        return _addr((pc + 4 + off) & M32, symbols)
    if kind == "rsd":
        return g(21)
    if kind == "pc19":                      # ADDIUPC / LWPC: the address
        return _addr((pc + (_sx(word & 0x7FFFF, 19) << 2)) & M32, symbols)
    if kind == "sa2":
        return str(_bits(word, 6, 2) + 1)
    if kind == "bp":
        return str(_bits(word, 6, 2))
    if kind == "code16":
        return str(word & 0xFFFF)
    if kind == "mem11":
        return f"{_sx(word & 0x7FF, 11)}({g(11)})"
    if kind == "jt":
        return _addr(jump_target(word, pc), symbols)
    if kind in ("ac", "ac21"):
        return f"$ac{_bits(word, KINDS[kind].fields[0][0], 2)}"
    if kind in ("dsa3", "dsa4", "dsa5", "dsize", "dsa", "bp2", "u1", "h1", "rx"):
        lsb, width = KINDS[kind].fields[0]
        return str(_bits(word, lsb, width))
    if kind in ("rmask", "wmask", "dimm8"):
        lsb, width = KINDS[kind].fields[0]
        v = _bits(word, lsb, width)
        return f"0x{v:x}" if v > 9 else str(v)
    if kind == "dimm10":
        return str(_sx(_bits(word, 16, 10), 10))
    if kind == "shilo":
        return str(_sx(_bits(word, 20, 6), 6))
    if kind in ("cc18", "cc8"):
        return f"$fcc{_bits(word, KINDS[kind].fields[0][0], 3)}"
    if kind == "c2cc":
        return f"$cc{_bits(word, 18, 3)}"
    if kind == "bcode":
        c1, c2 = _bits(word, 16, 10), _bits(word, 6, 10)
        return f"{c1}, {c2}" if c2 else str(c1)
    if kind == "pos_ext":
        pos, msbd = _bits(word, 6, 5), _bits(word, 11, 5)
        return f"{pos}, {msbd + 1}"
    if kind == "pos_ins":
        pos, msb = _bits(word, 6, 5), _bits(word, 11, 5)
        return f"{pos}, {msb - pos + 1}"
    if kind.startswith("pos_d"):                # MIPS64 DEXT ... DINSU: pos, size
        pos, size = mips64_field(kind, _bits(word, 6, 5), _bits(word, 11, 5))
        return f"{pos}, {size}"
    if kind == "bp3":
        return str(_bits(word, 6, 3))
    if kind == "pc18":                          # LDPC: (PC & ~7) + offset * 8
        return _addr(((pc & ~7) + (_sx(word & 0x3FFFF, 18) << 3)) & M32, symbols)
    if kind == "rs2":                           # DAHI / DATI: rs, written twice
        return g(21)
    raise KeyError(kind)


def mips64_field(kind: str, lsb: int, high: int) -> tuple[int, int]:
    """(pos, size) of a MIPS64 DEXT / DEXTM / DEXTU / DINS / DINSM / DINSU
    from its lsb and msbd / msb fields (MIPS64 Vol. II)."""
    if kind == "pos_dext":
        return lsb, high + 1
    if kind == "pos_dextm":
        return lsb, high + 33
    if kind == "pos_dextu":
        return lsb + 32, high + 1
    if kind == "pos_dins":
        return lsb, high - lsb + 1
    if kind == "pos_dinsm":
        return lsb, high + 32 - lsb + 1
    return lsb + 32, high - lsb + 1             # pos_dinsu


def _is_default(kind: str, name: str, word: int) -> bool:
    if kind == "rd" and name.startswith("JALR"):
        return _bits(word, 11, 5) == 31
    if kind == "bcode":
        return _bits(word, 6, 20) == 0
    lsb, width = KINDS[kind].fields[0]
    return _bits(word, lsb, width) == DEFAULTS.get(kind, 0)


def disassemble(word: int, pc: int = 0, *, isa: ISA | None = None,
                abi: bool = True, symbols: Mapping[int, str] | None = None,
                style: str = "mars", release: str | None = None,
                dsp: int = 0, mt: bool = False) -> str:
    """Assembly text of one instruction word at address ``pc``.

    A word that does not decode is printed as ``.word 0x........``.
    ``symbols`` maps addresses to label names for branch / jump targets;
    ``dsp`` = the DSP Module level whose instructions decode, ``mt`` = the
    MT Module's (``isa``'s when given).
    """
    isa = isa or ISA.for_release(release, dsp, mt)
    try:
        dec = isa.decode(word & M32)
    except ISADecodeError:
        return f".word 0x{word & M32:08x}"
    name = dec.name
    parts: list[str] = []
    for item in operand_spec(dec.entry):
        kind, optional = kind_of(item)
        if optional and _is_default(kind, name, word) and not (
                style == "gas" and name.startswith(("MOVF", "MOVT"))):
            continue                       # GNU as wants movf/movt's cc
        parts.append(operand_text(kind, word, pc, abi=abi, symbols=symbols))
    if style == "gas" and name in ("DIV", "DIVU", "DDIV", "DDIVU"):
        parts.insert(0, gpr_name(0, abi))
    mnemonic = dec.entry.get("mnemonic", name).lower()
    return f"{mnemonic} {', '.join(parts)}" if parts else mnemonic


#: (word, pc, ISA) -> its text (text_at)
_TEXTS: dict[tuple, str] = {}


def text_at(word: int, pc: int, isa: ISA) -> str:
    """``disassemble(word, pc, isa=isa)``, remembered: a pure function of
    the three, asked for every instruction the trace and the pipelines
    run (loops ask for the same ones again)."""
    key = (word, pc, isa)
    text = _TEXTS.get(key)
    if text is None:
        if len(_TEXTS) >= 1 << 16:
            _TEXTS.clear()
        text = _TEXTS[key] = disassemble(word, pc, isa=isa)
    return text


def disassemble_range(machine, start: int, count: int, **kw) -> list[tuple[int, int, str]]:
    """``(address, word, text)`` for ``count`` words from ``start`` of a
    :class:`emips.sim.Machine` (uses its byte order)."""
    m = machine
    out = []
    for i in range(count):
        addr = (start + 4 * i) & M32
        word = m.read_word(addr)
        out.append((addr, word, disassemble(word, addr, **kw)))
    return out
