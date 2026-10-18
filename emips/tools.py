"""Instruction reference, encoder / decoder and number converter — the
GUI Tools menu (M8 step 3) and the CLI debugger commands (M7), free of Qt.

Everything comes from the real tables and code paths — the reference from
``isaref.json`` / ``releases.json`` (via :class:`~emips.isa.ISA`), the
encoder from the assembler, the decoder from :meth:`ISA.decode` and
:func:`~emips.trace.decode_fields` — so what a student reads here is what
the simulator does.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from . import fpu
from .assembler import AsmError, assemble
from .disasm import disassemble
from .isa import ISA, Decode, ISADecodeError
from .sim import PRIVILEGED_NAMES, classify, has_delay_slot
from .syntax import kind_of, operand_spec
from .trace import Field, decode_fields, instruction_format

M32 = 0xFFFFFFFF

#: how each operand kind is written in the syntax column
KIND_TEXT = {
    "mem": "offset(base)", "mem9": "offset9(base)", "mem11": "offset11(base)",
    "idx": "index(base)", "br": "label", "br21": "label", "br26": "label",
    "jt": "target", "uimm": "imm", "cc18": "cc", "cc8": "cc", "c2cc": "cc",
    "code20": "code", "code19": "code", "bcode": "code", "tcode": "code",
    "code16": "code", "c0rd": "$cp0reg", "fcr": "$fcr", "hwr": "$hwr",
    "c2rt": "$c2reg", "impl16": "impl", "cofun": "cofun", "hint11": "hint",
    "pos_ext": "pos, size", "pos_ins": "pos, size", "rsd": "rs", "pc19": "label",
    "sa2": "sa", "sel6": "sel",
}

def syntax_text(entry: dict) -> str:
    """``ADD rd, rs, rt`` — the assembly form of a row."""
    mn = entry.get("mnemonic", entry["name"]).lower()
    try:
        spec = operand_spec(entry)
    except KeyError:
        return mn
    ops = []
    for item in spec:
        kind, optional = kind_of(item)
        text = KIND_TEXT.get(kind, kind)
        ops.append(f"[{text}]" if optional else text)
    return f"{mn} {', '.join(ops)}" if ops else mn


@lru_cache(maxsize=1)
def _releases() -> dict:
    return json.loads(resources.files("emips").joinpath("data/releases.json").read_text())


@dataclass(frozen=True)
class RefRow:
    """One instruction of the reference."""

    name: str                 # row name (MUL@R6 ...)
    mnemonic: str
    syntax: str
    description: str
    format: str               # R / I / J / FR / FI / COP
    family: str
    encoding: str             # the 32-char template (0/1 fixed, letters = fields)
    releases: tuple[str, ...]
    since: str
    removed_in: str | None
    execution: str            # EXEC / HINT / DEFER / ILLEGAL
    privileged: bool
    delay_slot: bool

    def matches(self, query: str) -> bool:
        q = query.strip().lower()
        return not q or any(q in s.lower() for s in (
            self.mnemonic, self.description, self.family, self.syntax))


def reference(release: str = "R5") -> list[RefRow]:
    """Every instruction of a MIPS32 release, sorted by mnemonic."""
    isa = ISA.for_release(release)
    rel = _releases()
    rows = []
    for e in isa.instructions:
        if not isa.in_release(e["name"]):
            continue
        value = e["decode"]["value"]
        rows.append(RefRow(
            name=e["name"], mnemonic=e.get("mnemonic", e["name"]),
            syntax=syntax_text(e), description=e.get("desc") or "",
            format=instruction_format(value), family=e["family"],
            encoding=e["encode"], releases=tuple(e.get("releases", ())),
            since=rel["since"].get(e["name"], "R1"),
            removed_in=rel["removed_in"].get(e["name"]),
            execution=classify(e), privileged=e["name"] in PRIVILEGED_NAMES,
            delay_slot=has_delay_slot(e)))     # compact branches: own family
    return sorted(rows, key=lambda r: (r.mnemonic.lower(), r.name))


def template_fields(row_name: str, release: str = "R5") -> list[tuple[int, int, str, str]]:
    """``(msb, lsb, name, bits)`` of a row's encoding: the fixed bits and
    the operand fields named by their role (rs / base / offset ...), the
    same split as the Instruction panel; operand bits shown as letters."""
    isa = ISA.for_release(release)
    entry = next(e for e in isa.instructions if e["name"] == row_name)
    template = entry["encode"]
    out = []
    for f in decode_fields(Decode(entry["decode"]["value"], entry)):
        out.append((f.msb, f.lsb, f.name, template[31 - f.msb:32 - f.lsb]))
    return out


# ---------------------------------------------------------------- decoder

@dataclass
class Decoded:
    word: int
    text: str                 # disassembly
    name: str                 # row name
    description: str
    format: str
    fields: list[Field]
    execution: str


def parse_word(text: str) -> int:
    """A 32-bit word: ``0x...`` / ``0b...`` / plain hex (8 digits) or a
    decimal number."""
    t = text.strip().replace("_", "").replace(" ", "")
    if not t:
        raise ValueError("type an instruction word (0x...) or an instruction")
    if re.fullmatch(r"[0-9a-fA-F]{8}", t):
        v = int(t, 16)
    else:
        v = int(t, 0)
    if not -(1 << 31) <= v <= M32:
        raise ValueError(f"{text.strip()} does not fit in 32 bits")
    return v & M32


def decode_word(word: int, *, release: str = "R5", pc: int = 0x00400000,
                abi: bool = True) -> Decoded:
    isa = ISA.for_release(release)
    dec = isa.decode(word)                      # ISADecodeError if reserved
    e = dec.entry
    return Decoded(word, disassemble(word, pc, isa=isa, abi=abi), e["name"],
                   e.get("desc") or "", instruction_format(word), decode_fields(dec),
                   classify(e))


def encode_text(text: str, *, release: str = "R5", pc: int = 0x00400000,
                pseudo: bool = True, abi: bool = True) -> list[Decoded]:
    """Assemble one source line at ``pc``: every machine word it becomes
    (a pseudo-instruction can be several), decoded.  Branch / jump targets
    are absolute addresses or ``.``-free numbers; no delay-slot nop is
    added.  Raises :class:`AsmError`."""
    line = text.strip()
    if not line:
        raise AsmError(1, "type an instruction")
    prog = assemble(f".text 0x{pc & M32:08x}\n{line}\n", release=release,
                    pseudo=pseudo, reorder=False)
    words = [prog.word(a) for a in sorted(prog.source_map)]
    if not words:
        raise AsmError(1, "no instruction on this line")
    return [decode_word(w, release=release, pc=pc + 4 * i, abi=abi)
            for i, w in enumerate(words)]


def encode_or_decode(text: str, **kw) -> tuple[str, list[Decoded]]:
    """A number decodes, anything else is assembled: (``"decode"`` /
    ``"encode"``, words).  Raises ValueError / AsmError / ISADecodeError."""
    try:
        word = parse_word(text)
    except ValueError:
        return "encode", encode_text(text, **kw)
    kw.pop("pseudo", None)
    return "decode", [decode_word(word, **kw)]


# -------------------------------------------------------- number converter

@dataclass
class FloatParts:
    bits: int
    width: int                # 32 or 64
    sign: int
    exponent: int             # raw biased field
    fraction: int
    kind: str                 # normal / subnormal / zero / infinity / NaN
    value: float
    formula: str


#: the default (quiet) NaN a MIPS FPU produces (emips.fpu): legacy encoding
#: (before Release 6, FCSR.NAN2008 = 0) and IEEE 754-2008 (Release 6)
DEFAULT_NAN = {(32, n): fpu.default_nan(fpu.S, n) for n in (False, True)} | \
    {(64, n): fpu.default_nan(fpu.D, n) for n in (False, True)}


def float_parts(bits: int, width: int = 32, nan2008: bool = False) -> FloatParts:
    """The IEEE fields of ``bits``.  NaNs are classified the MIPS way:
    legacy (``nan2008=False``, before Release 6) a NaN whose fraction MSB
    is 1 is *signalling*; IEEE 754-2008 (Release 6) the reverse."""
    ew, fw, bias = (8, 23, 127) if width == 32 else (11, 52, 1023)
    sign = bits >> (width - 1) & 1
    exp = bits >> fw & ((1 << ew) - 1)
    frac = bits & ((1 << fw) - 1)
    value = struct.unpack(">f" if width == 32 else ">d",
                          bits.to_bytes(width // 8, "big"))[0]
    s = "-" if sign else "+"
    if exp == (1 << ew) - 1:
        quiet = bool(frac >> (fw - 1)) == nan2008
        kind = "infinity" if frac == 0 else ("NaN (quiet)" if quiet else
                                               "NaN (signalling)")
        formula = kind
    elif exp == 0:
        kind = "zero" if frac == 0 else "subnormal"
        formula = f"{s}0" if frac == 0 else f"{s}0.fraction × 2^{1 - bias}"
    else:
        kind = "normal"
        formula = f"{s}1.fraction × 2^({exp} − {bias}) = {s}1.fraction × 2^{exp - bias}"
    return FloatParts(bits, width, sign, exp, frac, kind, value, formula)


@dataclass
class Number:
    """Every view of a value (32-bit unless the input needs 64)."""

    source: str               # what was typed, as understood
    width: int                # 32 or 64
    bits: int                 # the raw bits
    unsigned: int
    signed: int
    hex: str
    binary: str               # grouped in nibbles
    octal: str
    chars: str                # bytes as ASCII (big-endian order), '.' if not printable
    single: FloatParts | None = None      # the 32 bits as an IEEE single
    double: FloatParts | None = None      # 64 bits as an IEEE double
    notes: list[str] = field(default_factory=list)


def group_bits(bits: int, width: int) -> str:
    """Binary, in groups of four."""
    s = format(bits, f"0{width}b")
    return " ".join(s[i:i + 4] for i in range(0, width, 4))


def fmt_float(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return repr(x)


def convert(text: str, nan2008: bool = False) -> Number:
    """Understand ``text`` (decimal, ``0x``, ``0b``, ``0o``, ``'c'``, a
    float like ``1.5`` / ``-2e-3`` / ``inf`` / ``nan``) and show it every
    way.  A float is stored as the nearest IEEE single (and double); ``nan``
    is the MIPS default NaN of the encoding (``nan2008``: Release 6)."""
    t = text.strip().replace("_", "")
    if not t:
        raise ValueError("type a number: 42, -7, 0x2a, 0b101010, 'A' or 1.5")
    notes: list[str] = []
    m = re.fullmatch(r"'(.)'", t)
    if m:
        bits, width, source = ord(m.group(1)), 32, f"character {t}"
        if bits > 0xFF:
            notes.append("not an ASCII / Latin-1 character: its Unicode code point")
    else:
        try:
            v = int(t, 0)
        except ValueError:
            v = None
        if v is None:
            try:
                x = float(t)
            except ValueError:
                raise ValueError(f"not a number: {text.strip()!r}") from None
            if math.isnan(x):
                single, dbits = DEFAULT_NAN[32, nan2008], DEFAULT_NAN[64, nan2008]
                notes.append("the default NaN of a MIPS FPU ("
                             + ("IEEE 754-2008 encoding, Release 6)" if nan2008 else
                                "legacy encoding: quiet NaNs have fraction MSB 0)"))
            else:
                single = struct.unpack(">I", struct.pack(">f", x))[0] \
                    if abs(x) < 3.5e38 or math.isinf(x) else None
                dbits = struct.unpack(">Q", struct.pack(">d", x))[0]
            if single is None:
                notes.append("too large for single precision: shown as a double")
                return _number(dbits, 64, f"float {t}", notes, nan2008)
            n = _number(single, 32, f"float {t}", notes, nan2008)
            n.double = float_parts(dbits, 64, nan2008)
            stored = n.single.value
            if stored != x and not math.isnan(x):
                n.notes.append(f"single precision cannot hold {t} exactly: stored "
                               f"{fmt_float(stored)} (error {stored - x:+.3g})")
            return n
        if v < -(1 << 63) or v >= 1 << 64:
            raise ValueError("larger than 64 bits")
        width = 32 if -(1 << 31) <= v <= M32 else 64
        bits = v & ((1 << width) - 1)
        source = f"integer {v}"
        if width == 64:
            notes.append("needs 64 bits (e.g. a double, or HI:LO)")
    return _number(bits, width, source, notes, nan2008)


def _number(bits: int, width: int, source: str, notes: list[str],
            nan2008: bool = False) -> Number:
    signed = bits - (1 << width) if bits >> (width - 1) else bits
    raw = bits.to_bytes(width // 8, "big")
    n = Number(source=source, width=width, bits=bits, unsigned=bits, signed=signed,
               hex=f"0x{bits:0{width // 4}x}", binary=group_bits(bits, width),
               octal=f"0o{bits:o}",
               chars="".join(chr(b) if 32 <= b < 127 else "." for b in raw),
               notes=list(notes))
    if width == 32:
        n.single = float_parts(bits, 32, nan2008)
    else:
        n.double = float_parts(bits, 64, nan2008)
    return n


__all__ = ["RefRow", "reference", "syntax_text", "template_fields", "Decoded",
           "parse_word", "decode_word", "encode_text", "encode_or_decode",
           "Number", "FloatParts", "convert", "float_parts", "group_bits", "fmt_float",
           "ISADecodeError"]
