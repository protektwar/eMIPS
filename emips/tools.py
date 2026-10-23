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
from .msa import H as _HALF                 # (MSA's binary16: the converter's half)
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
    "pos_ext": "pos, size", "pos_ins": "pos, size",
    **{f"pos_{k}": "pos, size" for k in ("dext", "dextm", "dextu", "dins", "dinsm", "dinsu")},
    "bp3": "bp", "pc18": "label", "rs2": "rs", "rsd": "rs", "pc19": "label",
    "sa2": "sa", "sel6": "sel",
    # MSA (PLAN M24): MD00866's operand names
    "rd6": "rd", "rs11": "rs", "mcs": "cs", "mcd": "cd",
    "m3": "m", "m4": "m", "m5": "m", "m6": "m",
    **{f"vmem_{d}": "s10(rs)" for d in "bhwd"},
    **{f"elm{k}": "ws[n]" for k in (1, 2, 3, 4)},
    **{f"elmd{k}": "wd[n]" for k in (1, 2, 3, 4)},
    "ws0": "ws[0]", "elmr": "ws[rt]",
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
    #: an Application-Specific Extension's row: "dsp" / "dspr2" (the DSP
    #: Module, MD00374: Revision 1 / 2), "mt" (the MT Module), "msa" (MSA,
    #: MD00866), "" for the base architecture
    ase: str = ""

    def matches(self, query: str) -> bool:
        q = query.strip().lower()
        return not q or any(q in s.lower() for s in (
            self.mnemonic, self.description, self.family, self.syntax,
            _ASE_TEXT.get(self.ase, "")))


def search(rows: list[RefRow], query: str) -> list[RefRow]:
    """The rows matching ``query``, best first: the mnemonic itself, then
    mnemonics starting with it, then mnemonics containing it, then the
    matches in a description / family / syntax (``lw`` lists LW before
    FCAF.D, "Compare Always False"); each group in the reference's order."""
    q = query.strip().lower()

    def rank(r: RefRow) -> int:
        mn = r.mnemonic.lower()
        return 0 if mn == q else 1 if mn.startswith(q) else 2 if q in mn else 3
    return sorted((r for r in rows if r.matches(query)), key=rank)


#: what a reference search for an ASE's rows may type
_ASE_TEXT = {"dsp": "dsp module", "dspr2": "dsp module", "mt": "mt module",
             "msa": "msa simd"}


def reference(release: str = "R5", arch: str = "mips32") -> list[RefRow]:
    """Every instruction of a MIPS32 (``arch="mips64"``: MIPS64) release,
    sorted by mnemonic.  MIPS64: the DSP Module's rows are MD00375's (with
    LDX), the MT Module's MD00378's.  ``arch="nanomips"``: every nanoMIPS
    format (MD01247 ch. 2, emips.nanomips)."""
    if arch == "nanomips":
        return _nano_reference()
    isa = ISA.for_release(release, arch=arch)
    rel = _releases()
    wide = arch == "mips64"
    rows = []
    for e in isa.instructions:
        if not isa.in_release(e["name"]):
            continue
        rels = e.get("releases64" if wide else "releases", ())
        value = e["decode"]["value"]
        rows.append(RefRow(
            name=e["name"], mnemonic=e.get("mnemonic", e["name"]),
            syntax=syntax_text(e), description=e.get("desc") or "",
            format=instruction_format(value), family=e["family"],
            encoding=e["encode"], releases=tuple(rels),
            since=rel["since64" if wide else "since"].get(e["name"]) or rels[0],
            removed_in=rel["removed_in"].get(e["name"]),
            execution=classify(e), privileged=e["name"] in PRIVILEGED_NAMES,
            delay_slot=has_delay_slot(e),      # compact branches: own family
            ase=e.get("ase", "")))
    return sorted(rows, key=lambda r: (r.mnemonic.lower(), r.name))


def template_fields(row_name: str, release: str = "R5", arch: str = "mips32"
                    ) -> list[tuple[int, int, str, str]]:
    """``(msb, lsb, name, bits)`` of a row's encoding: the fixed bits and
    the operand fields named by their role (rs / base / offset ...), the
    same split as the Instruction panel; operand bits shown as letters
    (nanoMIPS: one per token of the format's diagram, ``row_name`` its
    form)."""
    if arch == "nanomips":
        return _nano_template(row_name)
    isa = ISA.for_release(release, arch=arch)
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
    #: its size in bytes (a nanoMIPS instruction: 2, 4 or 6)
    size: int = 4

    def encoding(self, arch: str = "mips32") -> str:
        """The encoding as a listing prints it (nanoMIPS: its halfwords)."""
        from .cli import encoding
        return encoding(self.word, self.size, arch).rstrip() if arch == "nanomips" \
            else f"0x{self.word:08x}"


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


def _dsp(release: str, dsp: int | None, arch: str = "mips32") -> int:
    """The DSP Module level the tools decode with: ``None`` = all the
    release allows (Revision 2 from Release 2 on: the reference tool knows
    every instruction; a core's Machine decodes only what it has)."""
    return dsp if dsp is not None else (0 if release == "R1" else 2)


def _msa(release: str, msa: bool | None) -> bool:
    """MSA's rows decode in the tools: ``None`` = when the release can have
    it (Release 5 and 6, PLAN M24)."""
    return msa if msa is not None else release in ("R5", "R6")


def decode_word(word: int, *, release: str = "R5", pc: int = 0x00400000,
                abi: bool = True, dsp: int | None = None, arch: str = "mips32",
                size: int | None = None, msa: bool | None = None) -> Decoded:
    """Decode a machine word (nanoMIPS: an encoding of ``size`` bytes, by
    default what its first halfword says — :func:`parse_nano`)."""
    if arch == "nanomips":
        return _nano_decode(word, size, pc, abi)
    isa = ISA.for_release(release, _dsp(release, dsp, arch), arch=arch,
                          msa=_msa(release, msa))
    dec = isa.decode(word)                      # ISADecodeError if reserved
    e = dec.entry
    return Decoded(word, disassemble(word, pc, isa=isa, abi=abi), e["name"],
                   e.get("desc") or "", instruction_format(word), decode_fields(dec),
                   classify(e))


def encode_text(text: str, *, release: str = "R5", pc: int = 0x00400000,
                pseudo: bool = True, abi: bool = True,
                dsp: int | None = None, arch: str = "mips32") -> list[Decoded]:
    """Assemble one source line at ``pc``: every machine word it becomes
    (a pseudo-instruction can be several), decoded.  Branch / jump targets
    are absolute addresses or ``.``-free numbers; no delay-slot nop is
    added.  Raises :class:`AsmError`."""
    line = text.strip()
    if not line:
        raise AsmError(1, "type an instruction")
    if arch == "nanomips":                  # GNU as + ld (emips.nanoasm): .text at pc
        prog = assemble(f".text\n{line}\n", pseudo=pseudo, arch=arch,
                        bases={"text": pc & M32},
                        nano={"crc": True, "ginv": True, "eva": True, "tlb": True})
        out = []
        for a in sorted(prog.nm):
            i = prog.nm_insn(a)
            out.append(_nano_decode(i.raw, i.size, a, abi))
        if not out:
            raise AsmError(1, "no instruction on this line")
        return out
    prog = assemble(f".text 0x{pc & M32:08x}\n{line}\n", release=release,
                    pseudo=pseudo, reorder=False, arch=arch)
    words = [prog.word(a) for a in sorted(prog.source_map)]
    if not words:
        raise AsmError(1, "no instruction on this line")
    return [decode_word(w, release=release, pc=pc + 4 * i, abi=abi, dsp=dsp, arch=arch)
            for i, w in enumerate(words)]


def encode_or_decode(text: str, **kw) -> tuple[str, list[Decoded]]:
    """A number decodes, anything else is assembled: (``"decode"`` /
    ``"encode"``, words).  Raises ValueError / AsmError / ISADecodeError."""
    nano = kw.get("arch") == "nanomips"
    try:
        word, size = parse_nano(text) if nano else (parse_word(text), None)
    except ValueError:
        return "encode", encode_text(text, **kw)
    kw.pop("pseudo", None)
    return "decode", [decode_word(word, size=size, **kw)]


# ---------------------------------------------------------------- nanoMIPS

def parse_nano(text: str) -> tuple[int, int]:
    """A nanoMIPS encoding (PLAN M23): hex halfwords — ``d205``, ``00a4
    0007``, ``0x00a40007``, ``6080 5678 1234`` — or a number; (its value,
    its size in bytes: what the first halfword says, MD01247 3.1)."""
    from . import nanomips
    t = text.strip().replace("_", "")
    if not t:
        raise ValueError("type an instruction encoding (0x...) or an instruction")
    digits = t[2:] if t.lower().startswith("0x") else t
    if re.fullmatch(r"(?:[0-9a-fA-F]{4}\s*){1,3}", digits):
        digits = re.sub(r"\s+", "", digits)
        n = len(digits) // 2
        value = int(digits, 16)
    else:
        value = int(t, 0)
        n = 2 if value < 1 << 16 else 4 if value < 1 << 32 else 6
    if n not in (2, 4, 6) or not 0 <= value < 1 << (8 * n):
        raise ValueError(f"{text.strip()} is not a 16, 32 or 48-bit encoding")
    size = nanomips.size_of(value >> (8 * n - 16))
    if size != n:
        raise ValueError(f"the first halfword 0x{value >> (8 * n - 16):04x} starts a "
                         f"{8 * size}-bit instruction, not a {8 * n}-bit one")
    return value, size


def _nano_decode(raw: int, size: int | None, pc: int, abi: bool) -> Decoded:
    from . import nanomips
    from .simnano import PRIVILEGED, entry_name
    from .tracenano import fields
    if size is None:
        size = 2 if raw < 1 << 16 else 4 if raw < 1 << 32 else 6
    i = nanomips.decode_raw(raw, size)          # NMDecodeError if reserved
    row = i.row
    return Decoded(raw, nanomips.text(i, pc, abi=abi), row.form,
                   f"{row.name}, nanoMIPS {8 * size}-bit",
                   row.form, fields(i, "p32" if abi else False),
                   "privileged" if entry_name(row) in PRIVILEGED else "EXEC", size)


_NANO_KIND = {"r": "", "d": "", "x": "", "n": "", "c": "", "b": "label", "h": "", "p": "",
              "L": ""}


@lru_cache(maxsize=1)
def _nano_reference() -> list[RefRow]:
    from . import nanomips
    from .simnano import _FAMILY, PRIVILEGED, entry_name
    rows = []
    for r in nanomips.ROWS:
        text = r.texts[-1][1]
        syntax = re.sub(r"\{(\w):(\w+)\}",
                        lambda m: _NANO_KIND.get(m.group(1)) or m.group(2), text)
        notes = [] if r.nms else ["not in NMS"]
        rows.append(RefRow(
            name=r.form, mnemonic=r.name.lower(), syntax=syntax,
            description=f"{r.form}: {8 * r.size}-bit" + (f", {', '.join(notes)}" if notes
                                                          else ""),
            format=f"{8 * r.size}-bit", family=_FAMILY[r.name], encoding=r.template,
            releases=("R6",), since="R6", removed_in=None, execution="EXEC",
            privileged=entry_name(r) in PRIVILEGED, delay_slot=False, ase=r.need or ""))
    return sorted(rows, key=lambda x: (x.mnemonic, x.name))


def _nano_template(form: str) -> list[tuple[int, int, str, str]]:
    from . import nanomips
    row = nanomips.TABLE.by_form[form]
    out, pos = [], row.size * 8
    for t in row.template.split():
        m = nanomips._TOKEN.match(t)
        if m.group(1):
            w, name, bits = len(m.group(1)), "op", m.group(1)
        else:
            name = m.group(2) or m.group(4)
            w = int(m.group(3)) if m.group(2) else \
                int(m.group(5)) - int(m.group(6) or m.group(5)) + 1
            bits = name[0] * w
        pos -= w
        out.append((pos + w - 1, pos, name if m.group(1) or m.group(4) is None
                    else t, bits))
    return out


# -------------------------------------------------------- number converter

@dataclass
class FloatParts:
    bits: int
    width: int                # 16 (MSA's half precision), 32 or 64
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
    ew, fw, bias = {16: (5, 10, 15), 32: (8, 23, 127), 64: (11, 52, 1023)}[width]
    sign = bits >> (width - 1) & 1
    exp = bits >> fw & ((1 << ew) - 1)
    frac = bits & ((1 << fw) - 1)
    value = struct.unpack({16: ">e", 32: ">f", 64: ">d"}[width],
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
    #: an IEEE half (binary16: MSA's FEXDO.H / FEXUPL / FEXUPR, MD00866
    #: 3.5.2): the low 16 bits of a value that fits them, or a float's
    #: nearest half
    half: FloatParts | None = None
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
            if math.isnan(x):
                n.half = float_parts(fpu.default_nan(_HALF, nan2008), 16, nan2008)
            else:
                try:
                    n.half = float_parts(struct.unpack(">H", struct.pack(">e", x))[0], 16,
                                         nan2008)
                except OverflowError:               # (beyond 65504: no half)
                    n.half = None
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
        if bits <= 0xFFFF:
            n.half = float_parts(bits, 16, nan2008)
    else:
        n.double = float_parts(bits, 64, nan2008)
    return n


__all__ = ["RefRow", "reference", "syntax_text", "template_fields", "Decoded",
           "parse_word", "decode_word", "encode_text", "encode_or_decode",
           "Number", "FloatParts", "convert", "float_parts", "group_bits", "fmt_float",
           "ISADecodeError"]
