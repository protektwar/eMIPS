"""The RISC-V linker (PLAN M25 phase 3a): GNU ld 2.42's static link of one
object from :mod:`emips.rvasm`, after ``bfd/elfnn-riscv.c``
(``riscv_elf_relocate_section`` / ``perform_relocation``) and ld's default
linker script for RISC-V ELF (``ld --verbose``), with eMIPS's addresses.

The layout (RARS's memory map, decision 10 of the plan): ``.text`` (and
``.text.*``) at ``bases["text"]`` (0x00400000), ``.rodata`` after it;
``.data``, ``.sdata`` (``.srodata`` first), ``.sbss``, ``.bss`` from
``bases["data"]`` (0x10010000), each output section aligned to its input's
alignment; ``__global_pointer$`` = MIN(__SDATA_BEGIN__ + 0x800,
MAX(__DATA_BEGIN__ + 0x800, __BSS_END__ - 0x800)) as ld's script computes it
(``__BSS_END__`` aligned to XLEN / 8).  ``tests/test_rvasm_gnu.py``'s
linker script mirrors it — change both together.

Relocations, with ld's checks ("relocation truncated to fit"): BRANCH,
JAL, RVC_BRANCH, RVC_JUMP, CALL / CALL_PLT (auipc + jalr), PCREL_HI20 /
PCREL_LO12_I / _S (the low part from the PCREL_HI20 at the address its
symbol names), HI20 / LO12_I / _S, 32 / 64 and label differences (gas's
ADD / SUB pairs).  Relaxation (R_RISCV_RELAX, R_RISCV_ALIGN): phase 3b.
"""

from __future__ import annotations

from dataclasses import dataclass, field

M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
TEXT_BASE, DATA_BASE = 0x00400000, 0x10010000


@dataclass
class Reloc:
    offset: int
    type: str
    expr: object
    size: int
    line: int
    relax: bool = False


@dataclass
class ObjSection:
    name: str
    align: int
    code: bool
    bss: bool
    data: bytearray = field(default_factory=bytearray)
    relocs: list = field(default_factory=list)
    insns: list = field(default_factory=list)       # (offset, size, line)
    address: int = 0


class LinkError(Exception):
    def __init__(self, line: int, message: str) -> None:
        super().__init__(message)
        self.line = line
        self.message = message


#: output sections in ld's order: (name, input name patterns, segment)
LAYOUT = [
    (".text", ((".text",), (".text.",)), "text"),
    (".rodata", ((".rodata",), (".rodata.",)), "text"),
    (".data", ((".data",), (".data.",)), "data"),
    (".sdata", ((".srodata", ".sdata"), (".srodata.", ".sdata.")), "data"),
    (".sbss", ((".sbss",), (".sbss.",)), "data"),
    (".bss", ((".bss",), (".bss.",)), "data"),
]


def _output_of(name: str) -> str | None:
    for out, (exact, prefixes), _seg in LAYOUT:
        if name in exact or name.startswith(prefixes):
            return out
    return None


@dataclass
class Linked:
    """The result: section addresses, symbol values, the image."""
    xlen: int
    sections: dict
    symbols: dict
    gp: int
    image: dict = field(default_factory=dict)
    segments: dict = field(default_factory=dict)


def _align(v: int, power: int) -> int:
    a = 1 << power
    return (v + a - 1) & ~(a - 1)


def layout(obj, bases: dict | None = None) -> tuple[dict, int]:
    """Place the sections (see the module docstring); returns their
    addresses and __global_pointer$."""
    bases = {"text": TEXT_BASE, "data": DATA_BASE, **(bases or {})}
    by_out: dict[str, list] = {out: [] for out, _p, _s in LAYOUT}
    for name, sec in obj.sections.items():
        out = _output_of(name)
        if out is None:
            if sec.data or sec.relocs:
                raise LinkError(0, f"section `{name}' has no place in eMIPS's memory map")
            continue
        by_out[out].append(sec)
    for v in by_out.values():                       # .srodata before .sdata, then by name
        v.sort(key=lambda s: (0 if s.name.startswith(".srodata") else 1)
               if s.name.startswith((".srodata", ".sdata")) else 0)
    dot = {"text": bases["text"], "data": bases["data"]}
    marks: dict[str, int] = {}
    for out, _p, seg in LAYOUT:
        secs = by_out[out]
        align = max([s.align for s in secs], default=0)
        start = _align(dot[seg], align)
        if out == ".data":
            marks["__DATA_BEGIN__"] = start
        if out == ".sdata":
            marks["__SDATA_BEGIN__"] = start
        addr = start
        for s in secs:
            addr = _align(addr, s.align)
            s.address = addr
            addr += len(s.data)
        if out == ".bss" and addr != start:
            addr = _align(addr, 3 if obj.xlen == 64 else 2)
        dot[seg] = addr if secs else dot[seg]
    bss_end = _align(dot["data"], 3 if obj.xlen == 64 else 2)
    gp = min(marks["__SDATA_BEGIN__"] + 0x800,
             max(marks["__DATA_BEGIN__"] + 0x800, bss_end - 0x800))
    return marks, gp


class Linker:
    def __init__(self, obj, bases: dict | None = None) -> None:
        self.obj = obj
        self.xlen = obj.xlen
        self.bases = bases
        self.marks, self.gp = layout(obj, bases)
        self.pcrel_hi: dict[int, int] = {}          # auipc address -> its value (S + A - P)

    def sym_value(self, s, line: int) -> int:
        """A symbol's final address (an equate: its expression's)."""
        if s.equated is not None:
            e = s.equated[0]
            return e.num + (self.sym_value(e.sym, line) if e.sym is not None else 0) - \
                (self.sym_value(e.sub, line) if getattr(e, "sub", None) is not None else 0)
        if s.absolute is not None:
            return s.absolute
        if s.frag is not None:
            return self.obj.sections[s.section.name].address + s.frag.address + s.where
        special = {"__global_pointer$": self.gp, **self.marks}
        if s.name in special:
            return special[s.name]
        if s.weak:
            return 0
        raise LinkError(line, f"undefined reference to `{s.name}'")

    def value(self, e, line: int) -> int:
        v = e.num
        if e.sym is not None:
            v += self.sym_value(e.sym, line)
        if getattr(e, "sub", None) is not None:
            v -= self.sym_value(e.sub, line)
        return v

    def link(self) -> Linked:
        obj = self.obj
        # the PCREL_HI20 values first (riscv_record_pcrel_hi_reloc)
        for sec in obj.sections.values():
            for r in sec.relocs:
                if r.type in ("PCREL_HI20", "GOT_HI20"):
                    p = sec.address + r.offset
                    self.pcrel_hi[p] = self.value(r.expr, r.line) - p
        for sec in obj.sections.values():
            for r in sec.relocs:
                self.apply(sec, r)
        mask = M32 if self.xlen == 32 else M64
        symbols = {name: self.sym_value(s, 0) & mask for name, s in obj.symbols.items()
                   if s.defined and not name.startswith(".L")}
        symbols.setdefault("__global_pointer$", self.gp)
        out = Linked(self.xlen, obj.sections, symbols, self.gp)
        for sec in obj.sections.values():
            if sec.bss:
                continue
            for i, b in enumerate(sec.data):
                out.image[sec.address + i] = b
        for name, sec in obj.sections.items():
            if sec.data or sec.bss:
                out.segments[name] = (sec.address, sec.address + len(sec.data))
        return out

    def _overflow(self, r: Reloc, sym_text: str) -> None:
        raise LinkError(r.line, f"relocation truncated to fit: R_RISCV_{_RNAME.get(r.type, r.type)}"
                                f" against `{sym_text}'")

    def apply(self, sec: ObjSection, r: Reloc) -> None:
        from .rvasm import enc_b, enc_cb, enc_cj, enc_i, enc_j, enc_s, enc_u, high_part
        p = sec.address + r.offset
        t = r.type
        d = sec.data
        o = r.offset

        def get(n: int) -> int:
            return int.from_bytes(d[o:o + n], "little")

        def put(n: int, v: int) -> None:
            d[o:o + n] = (v & ((1 << (8 * n)) - 1)).to_bytes(n, "little")

        name = r.expr.sym.name if getattr(r.expr, "sym", None) is not None else "*ABS*"
        if t == "ALIGN" or t == "RELAX":
            return
        if t in ("TPREL_HI20", "TPREL_LO12_I", "TPREL_LO12_S", "TPREL_ADD", "TLS_GOT_HI20",
                 "TLS_GD_HI20", "GOT_HI20"):
            raise LinkError(r.line, f"R_RISCV_{_RNAME.get(t, t)}: thread-local storage and the "
                                    "GOT are not supported (a static link of position-"
                                    "dependent code)")
        if t.startswith("DATA"):
            n = int(t[4:])
            if getattr(r.expr, "sub", None) is None and n in (1, 2):
                raise LinkError(r.line, f"cannot represent BFD_RELOC_{8 * n} relocation in "
                                        "object file")
            put(n, self.value(r.expr, r.line))
            return
        v = self.value(r.expr, r.line)
        if t == "12_PCREL":
            v -= p
            if v & 1 or not -4096 <= v < 4096:
                self._overflow(r, name)
            put(4, get(4) & ~enc_b(-1) | enc_b(v))
        elif t == "JMP":
            v -= p
            if v & 1 or not -(1 << 20) <= v < 1 << 20:
                self._overflow(r, name)
            put(4, get(4) & ~enc_j(-1) | enc_j(v))
        elif t == "RVC_BRANCH":
            v -= p
            if v & 1 or not -256 <= v < 256:
                self._overflow(r, name)
            put(2, get(2) & ~enc_cb(-1) | enc_cb(v))
        elif t == "RVC_JUMP":
            v -= p
            if v & 1 or not -2048 <= v < 2048:
                self._overflow(r, name)
            put(2, get(2) & ~enc_cj(-1) | enc_cj(v))
        elif t in ("CALL", "CALL_PLT"):
            v -= p
            if not self._utype_ok(high_part(v)):
                self._overflow(r, name)
            put(4, get(4) & ~enc_u(-1) | enc_u(high_part(v)))
            o += 4
            put(4, get(4) & ~enc_i(-1) | enc_i(v))
        elif t == "PCREL_HI20":
            v -= p
            if not self._utype_ok(high_part(v)):
                self._overflow(r, name)
            put(4, get(4) & ~enc_u(-1) | enc_u(high_part(v)))
        elif t in ("PCREL_LO12_I", "PCREL_LO12_S"):
            hi = self.pcrel_hi.get(v)
            if hi is None:
                raise LinkError(r.line, "dangerous relocation: %pcrel_lo missing matching "
                                        "%pcrel_hi")
            enc = enc_i if t.endswith("_I") else enc_s
            put(4, get(4) & ~enc(-1) | enc(hi))
        elif t == "HI20":
            if not self._utype_ok(high_part(v)):
                self._overflow(r, name)
            put(4, get(4) & ~enc_u(-1) | enc_u(high_part(v)))
        elif t in ("LO12_I", "LO12_S"):
            enc = enc_i if t.endswith("_I") else enc_s
            put(4, get(4) & ~enc(-1) | enc(v))
        elif t == "32":
            put(4, v)
        else:
            raise LinkError(r.line, f"unsupported relocation R_RISCV_{t}")

    def _utype_ok(self, v: int) -> bool:
        """VALID_UTYPE_IMM of a 64-bit value: what lui / auipc can make
        (sign-extended from bit 31; RV32: any 32-bit value)."""
        if self.xlen == 32:
            v &= M32
            v = v - (1 << 32) if v >> 31 else v
        return -(1 << 31) <= v < 1 << 31


#: gas's BFD codes -> the R_RISCV_ names
_RNAME = {"12_PCREL": "BRANCH", "JMP": "JAL", "DATA4": "32", "DATA8": "64"}


def link(obj, bases: dict | None = None) -> Linked:
    return Linker(obj, bases).link()
