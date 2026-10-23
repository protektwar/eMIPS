"""The nanoMIPS link (PLAN M23 phase 3): gold's nanoMIPS target for the one
object file ``emips.nanoasm`` assembles.

GNU ``as`` leaves an object: each section's bytes, its relocations and
the symbols defined in it.  ``ld`` (gold, MediaTek's nanoMIPS toolchain
release 2025.09-02: ``gold/nanomips.cc``, ``nanomips-insn.def``,
``nanomips-reloc.def``, ``layout.cc``) lays the sections out, and -- for
an object assembled with
``.linkrelax`` (``EF_NANOMIPS_LINKRELAX``) -- may change the code
first:

- *expansions* (always, unless ``--no-expand``): an instruction whose
  relocation no longer fits becomes a longer form or a sequence --
  ``bc[16]`` -> ``bc``, ``beqc`` too far -> ``bnec`` over a ``bc``, a
  ``bc`` beyond its 32 MB -> ``lapc[48]`` + ``jrc[16]``, ``lw[gp]`` past
  the 2 MB of the small data -> ``addiugp[48]`` + ``lw``;
- *relaxations* (``--relax``): the other way round -- ``bc`` ->
  ``bc[16]``, ``beqc`` -> ``beqc[16]`` / ``beqzc[16]``, ``lapc[48]`` ->
  ``lapc``, ``lw[gp]`` -> ``lw[gp16]``, ``addiugp[48]`` ->
  ``addiu[gp.b]`` / ``addiu[gp.w]``.

Each pass (``Target_nanomips::do_relax``) walks every code section's
relocations against the layout of the previous pass: an instruction that
changes size inserts or deletes bytes after itself
(``update_content``), which moves every later relocation, label and
symbol of its section, and the first ``R_NANOMIPS_ALIGN`` after it is
padded again (``Nanomips_transformations::align``).  Relaxation passes
run while something changes, then expansion passes; then each
relocation is applied (``Target_nanomips::Relocate``: the value, the
field, the overflow check -- a pair at one offset, ``R_NANOMIPS_NEG`` +
``R_NANOMIPS_32``, combines) and the branches a transformation created
are filled in (``relocate_branch``).

The transformation tables are generated from gold's ``.def`` files into
``emips/data/nanomips_asm.json`` (never edited by hand).  Not
modelled (eMIPS's assembler never emits their relocations): the GOT /
TLS / PIC transformations, ``R_NANOMIPS_SAVERESTORE`` and the GP-setup
removal (``finalize_gpsetup_optimizations``), ``--finalize-relocs``,
sorting by reference.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path

M32 = 0xFFFFFFFF
NOP16 = 0x9008
NOP32 = 0x8000C000

_GOLD = json.loads((Path(__file__).parent / "data" / "nanomips_asm.json").read_text())["gold"]


@dataclass(frozen=True)
class RelocProperty:
    """nanomips-reloc.def: an R_NANOMIPS_* type."""
    name: str
    number: int
    kind: str                   # STATIC / PLACEHOLDER / DYNAMIC
    size: int                   # the instruction's size in bits (0: data)
    bitsize: int
    mask: int                   # the bits that find the instruction

    @property
    def placeholder(self) -> bool:
        return self.kind == "PLACEHOLDER"


RELOCS = {name: RelocProperty(name, **p) for name, p in _GOLD["relocs"].items()}


# -- nanomips-insn-property.cc: the register functions -------------------------

def _valid_reg(r: int) -> bool:
    return 4 <= r <= 7 or 16 <= r <= 19


def _valid_st_src_reg(r: int) -> bool:
    return r == 0 or 4 <= r <= 7 or 17 <= r <= 19


_GPR3 = (16, 17, 18, 19, 4, 5, 6, 7)
_GPR3_ST = (0, 17, 18, 19, 4, 5, 6, 7)
_GPR4_ZERO = (8, 9, 10, 0, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)

_CHECK = {"valid_reg": _valid_reg, "valid_st_src_reg": _valid_st_src_reg}
_CONVERT = {"convert_reg": lambda r: _GPR3[r], "convert_st_src_reg": lambda r: _GPR3_ST[r]}
_EXTRACT = {
    "move_balc_treg_32": lambda i: _GPR4_ZERO[((i >> 21) & 0x7) | ((i >> 22) & 0x8)],
    "move_balc_dreg_32": lambda i: ((i >> 24) & 0x1) + 4,
    "ext_sres_fields": lambda i: ((i >> 3) & 0x1FF) | ((i >> 12) & 0x3E00),
}


def _ins_sres(_t: int, fields: int, data: int) -> int:
    return data | (fields & 0x3E00) << 12 | (fields & 0x1FF) << 3


def _ins_sres16(_t: int, fields: int, data: int) -> int:
    rt = (fields & 0x3E00) >> 9
    return data | (0 if rt == 30 else 1 << 9) | (fields & 0x1FF) << 3


_INSERT = {"ins_sres_fields": _ins_sres, "ins_sres16_fields": _ins_sres16,
           "ins_bposge32c_fields": lambda _t, _s, data: data | 1 << 18}


def _extract(f):
    if f is None:
        return None
    if isinstance(f, str):
        return _EXTRACT[f]
    pos, bits = f["pos"], f["bits"]
    return lambda insn: (insn >> pos) & ((1 << bits) - 1)


def _insert(f):
    if f is None:
        return None
    if isinstance(f, str):
        return _INSERT[f]
    pos, mask, treg = f["pos"], (1 << f["bits"]) - 1, f["from"] == "TREG"
    return lambda t, s, data: data | ((t if treg else s) & mask) << pos


@dataclass(frozen=True)
class InsnTemplate:
    """Nanomips_insn_template: one instruction of a transformation."""
    name: str
    data: int
    reloc: str
    size: int                   # bytes
    treg: object
    sreg: object

    def encode(self, treg: int, sreg: int) -> int:
        data = self.data
        if self.treg is not None:
            data = self.treg(treg, sreg, data)
        if self.sreg is not None:
            data = self.sreg(treg, sreg, data)
        return data & M32


@dataclass(frozen=True)
class Transform:
    """Nanomips_transform_template: the instructions for some relocations."""
    relocs: frozenset
    insns: tuple

    @property
    def size(self) -> int:
        return sum(t.size for t in self.insns)


class InsnProperty:
    """Nanomips_insn_property: an instruction gold may transform."""

    def __init__(self, p: dict) -> None:
        self.name = p["name"]
        self.opcode = p["opcode"]
        self._treg = _extract(p.get("treg"))
        self._sreg = _extract(p.get("sreg"))
        self._valid_treg = _CHECK.get(p.get("valid_treg"))
        self._valid_sreg = _CHECK.get(p.get("valid_sreg"))
        self._convert_treg = _CONVERT.get(p.get("convert_treg"))
        self._convert_sreg = _CONVERT.get(p.get("convert_sreg"))
        self.transforms: dict[str, list[Transform]] = {}
        self.relocs: set[str] = set()
        for t in p["transforms"]:
            tmpl = Transform(frozenset(t["relocs"]), tuple(
                InsnTemplate(i["name"], i["data"], i["reloc"], i["size"], _insert(i["treg"]),
                             _insert(i["sreg"])) for i in t["insns"]))
            self.transforms.setdefault(t["type"], []).append(tmpl)
            self.relocs |= tmpl.relocs

    def treg(self, insn: int) -> int:
        return self._treg(insn)

    def sreg(self, insn: int) -> int:
        return self._sreg(insn)

    def convert_treg(self, insn: int) -> int:
        return self._convert_treg(self.treg(insn))

    def convert_sreg(self, insn: int) -> int:
        return self._convert_sreg(self.sreg(insn))

    def valid_treg(self, insn: int) -> bool:
        return self._valid_treg(self.treg(insn))

    def valid_sreg(self, insn: int) -> bool:
        return self._valid_sreg(self.sreg(insn))

    def valid_regs(self, insn: int) -> bool:
        return self.valid_treg(insn) and self.valid_sreg(insn)

    def has_transform(self, ttype: str, r_type: str) -> bool:
        return any(r_type in t.relocs for t in self.transforms.get(ttype, ()))

    def get_transform(self, ttype: str, r_type: str) -> Transform:
        for t in self.transforms.get(ttype, ()):
            if r_type in t.relocs:
                return t
        raise AssertionError(f"{self.name}: no {ttype} transformation for {r_type}")


INSNS = {p["opcode"]: InsnProperty(p) for p in _GOLD["insns"]}


def insn_property(insn: int, mask: int, r_type: str) -> InsnProperty | None:
    """Nanomips_insn_property_table::get_insn_property."""
    p = INSNS.get(insn & mask)
    return p if p is not None and r_type in p.relocs else None


# -- the object --------------------------------------------------------------------

@dataclass(eq=False)
class Symbol:
    """A symbol of the object: defined in ``section`` at offset ``value``
    (gold moves it when the code before it changes size), or absolute
    (``section`` None: ``value`` itself; an ALIGN / FILL / MAX symbol's
    ``size`` is gas's padding / the fill's length)."""
    name: str
    section: "ObjSection | None" = None
    value: int = 0
    size: int = 0
    defined: bool = True
    #: defined by the linker script (_gp)
    script: bool = False
    #: STB_GLOBAL / STB_WEAK (gold prints "global symbol"); the section's
    #: own STT_SECTION symbol (gold never moves it: its addend says where)
    is_global: bool = False
    is_section: bool = False


@dataclass(eq=False)
class Reloc:
    """An Elf32_Rela: ``offset`` in its section, ``type`` (the R_NANOMIPS_
    name without the prefix), ``sym`` (None: the absolute section's
    symbol, i.e. the value is the addend), ``addend``; ``line`` is the
    source line (for messages)."""
    offset: int
    type: str
    sym: Symbol | None
    addend: int = 0
    line: int = 0


@dataclass
class Branch:
    """A Cond_branch: the conditional branch of a transformed one (it now
    jumps over the ``bc`` that follows it)."""
    reloc: str
    offset: int
    destination: int


@dataclass
class Entry:
    """What occupies some bytes of a code section, for the listing: an
    instruction of source line ``line`` (``nop``: alignment padding)."""
    offset: int
    size: int
    line: int
    nop: bool = False


@dataclass(eq=False)
class ObjSection:
    name: str
    data: bytearray
    align: int = 0                       # log2
    code: bool = False
    relocs: list[Reloc] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    branches: list[Branch] = field(default_factory=list)
    address: int = 0
    #: the STT_SECTION symbol (relocations gas converted: section + offset)
    section_symbol: "Symbol | None" = None


@dataclass
class Object:
    """The ELF relocatable file: its sections in section-index order, the
    ELF header's EF_NANOMIPS_LINKRELAX, the .nanoMIPS.abiflags ASE xNMS
    (``nmf``: the full nanoMIPS ISA, not the subset) and EF_NANOMIPS_PCREL /
    PID / PIC (eMIPS assembles none of those)."""
    big: bool
    sections: list[ObjSection] = field(default_factory=list)
    linkrelax: bool = False
    nmf: bool = True
    pcrel: bool = False
    pid: bool = False
    pic: bool = False
    #: the program's own definition of _gp, if any
    gp_symbol: Symbol | None = None

    def section(self, name: str) -> ObjSection | None:
        for s in self.sections:
            if s.name == name:
                return s
        return None


@dataclass
class LinkOptions:
    """gold's nanoMIPS options (gold/options.h; the defaults are ld's)."""
    relax: bool = False                  # --relax
    expand: bool = True                  # --[no-]expand
    insn32: bool = False                 # --insn32
    relax_lo12: bool = False             # --relax-lo12
    strict_address_modes: bool = True    # --[no-]strict-address-modes
    expand_reg: int = 1                  # --expand-reg
    fix_nmips_hw110880: bool = False
    fix_nmips_hw113064: bool = False
    fix_nmips_hw142543: bool = False


class LinkError(Exception):
    def __init__(self, line: int, message: str) -> None:
        super().__init__(message)
        self.line = line
        self.message = message


# -- arithmetic ------------------------------------------------------------------

def _signed(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def _overflow_signed(bits: int, v: int) -> bool:
    """Bits<valsize>::has_overflow32."""
    s = _signed(v)
    return not -(1 << (bits - 1)) <= s < (1 << (bits - 1))


def _overflow_unsigned(bits: int, v: int) -> bool:
    """Bits<valsize>::has_unsigned_overflow32."""
    return (v & M32) >= (1 << bits)


def _align_address(address: int, align: int) -> int:
    return (address + align - 1) & ~(align - 1)


RELAX, EXPAND = "relax", "expand"

TT_NONE = "NONE"


class Linker:
    """Target_nanomips for one object: ``link()`` = Layout's relaxation
    loop, then relocate_section for every section."""

    def __init__(self, obj: Object, bases: dict[str, int],
                 options: LinkOptions | None = None) -> None:
        self.obj = obj
        self.bases = bases
        self.o = options or LinkOptions()
        self.order = "big" if obj.big else "little"
        self.gp = 0
        self.segments: dict[str, tuple[int, int]] = {}
        #: what --debug=target prints: one line per transformation / padding change
        self.log: list[str] = []

    # -- layout (the eMIPS linker script) ------------------------------------------

    def layout(self) -> None:
        """Each section's address: the code sections at the text base;
        at the data base .data, .rdata, .rodata (and other data
        sections), then -- uhi32.ld's order -- ``. = ALIGN(8); _gp = .``,
        the small data (.ssdata, .ssbss, .sdata), ``. = ALIGN(4)``,
        .sbss, .bss; .ktext / .kdata at their own bases."""
        secs = {s.name: s for s in self.obj.sections}
        segs: dict[str, tuple[int, int]] = {}

        def place(names, addr):
            for name in names:
                sec = secs.get(name)
                if sec is not None:
                    addr = _align_address(addr, 1 << sec.align)
                    sec.address = addr
                    addr += len(sec.data)
            return addr

        def segment(seg, names, base):
            end = place(names, base)
            if end > base:
                segs[seg] = (base, end)

        code = [s.name for s in self.obj.sections if s.code and s.name != ".ktext"]
        segment("text", code, self.bases["text"])
        data = [".data", ".rdata", ".rodata"] + [
            s.name for s in self.obj.sections
            if not s.code and s.name not in DATA_ORDER and s.name != ".kdata"]
        base = self.bases["data"]
        addr = place(data, base)
        addr = _align_address(addr, 8)
        self.gp = addr
        addr = place([".ssdata", ".ssbss", ".sdata"], addr)
        addr = _align_address(addr, 4)
        addr = place([".sbss", ".bss"], addr)
        if any(len(secs[n].data) for n in secs if n in data or n in DATA_ORDER):
            segs["data"] = (base, addr)
        segment("ktext", [".ktext"], self.bases["ktext"])
        segment("kdata", [".kdata"], self.bases["kdata"])
        if self.obj.gp_symbol is not None:
            self.gp = self.value(self.obj.gp_symbol, 0)
        self.segments = segs

    def value(self, sym: Symbol | None, addend: int) -> int:
        """Symbol_value::value: the symbol's address + the addend."""
        if sym is None:
            return addend & M32
        if sym.script:
            return (self.gp + addend) & M32
        if sym.section is None:
            return (sym.value + addend) & M32
        return (sym.section.address + sym.value + addend) & M32

    # -- the relaxation loop ------------------------------------------------------

    def link(self) -> None:
        self.layout()
        if (self.obj.linkrelax and not self.obj.pic
                and (self.o.relax or self.o.expand)):      # do_may_relax, safe_to_modify
            state = RELAX if self.o.relax else EXPAND
            passes = 0
            while True:
                passes += 1
                again = False
                while True:                                 # do_relax
                    for sec in self.obj.sections:
                        if sec.code and sec.data and sec.relocs:
                            again |= self.scan(sec, state)
                    if not again and state == RELAX and self.o.expand:
                        state = EXPAND
                        continue
                    break
                if not again:
                    break
                self.layout()
        for sec in self.obj.sections:
            self.relocate_section(sec)
        for sec in self.obj.sections:
            for b in sec.branches:
                self.relocate_branch(sec, b)

    # -- Target_nanomips::scan_reloc_section_for_transform ------------------------

    def scan(self, sec: ObjSection, state: str) -> bool:
        again = do_align = seen_norelax = False
        relax = state == RELAX
        count = len(sec.relocs)
        for i in range(count):
            r = sec.relocs[i]
            if r.type == "NORELAX":
                seen_norelax = True
                continue
            if r.type == "RELAX":
                seen_norelax = False
                continue
            if seen_norelax:
                continue
            if r.type == "ALIGN":
                if do_align:
                    self.align(sec, i, count)
                    do_align = False
                continue
            prop = RELOCS[r.type]
            insn_size = prop.size
            if insn_size == 0:
                continue
            r_offset = r.offset - 2 if insn_size == 48 else r.offset
            insn = self.read_insn(sec.data, r_offset, insn_size)
            ip = (self.relax_find(insn, prop.mask, r.type) if relax
                  else self.expand_find(insn, prop.mask, r.type))
            if ip is None:
                continue
            if self.forced_length(sec, r_offset, count, i):
                continue
            if r.sym is not None and not r.sym.defined:
                continue                       # compute_final_value fails
            address = (sec.address + r_offset) & M32
            ttype = (self.relax_type(ip, r, insn, address) if relax
                     else self.expand_type(ip, r, insn, address))
            if ttype == TT_NONE:
                continue
            tmpl = ip.get_transform(ttype, r.type)
            grow = tmpl.size - insn_size // 8
            if grow:
                again = True
                do_align = True
                self.update_content(sec, r_offset + insn_size // 8, grow, False)
            self.transform(sec, tmpl, ip, ttype, i, insn)
            self.log.append(f"{sec.name}+{r_offset:#x}: {ip.name} is "
                            + ("removed" if not tmpl.insns else "transformed into "
                               + ", ".join(t.name for t in tmpl.insns))
                            + f" for {self.sym_name(r)}")
        return again

    @staticmethod
    def sym_name(r: Reloc) -> str:
        """How gold names a relocation's symbol."""
        if r.sym is None:
            return f"local symbol '*ABS*+{r.addend & M32:#x}'"
        if r.sym.is_section:
            return f"local symbol '{r.sym.name}+{r.addend & M32:#x}'"
        return f"{'global' if r.sym.is_global else 'local'} symbol '{r.sym.name}'"

    def read_insn(self, data: bytearray, offset: int, bits: int) -> int:
        """read_nanomips_insn: a 16-bit / 48-bit one's first halfword, a
        32-bit one's two halfwords (first << 16 | second)."""
        h0 = int.from_bytes(data[offset:offset + 2], self.order)
        if bits in (16, 48):
            return h0
        return h0 << 16 | int.from_bytes(data[offset + 2:offset + 4], self.order)

    def write_insn(self, data: bytearray, offset: int, insn: int, size: int) -> None:
        if size == 2:
            data[offset:offset + 2] = (insn & 0xFFFF).to_bytes(2, self.order)
        elif size == 4:
            data[offset:offset + 2] = ((insn >> 16) & 0xFFFF).to_bytes(2, self.order)
            data[offset + 2:offset + 4] = (insn & 0xFFFF).to_bytes(2, self.order)
        else:                       # 48 bits: the first halfword, the immediate cleared
            data[offset:offset + 2] = (insn & 0xFFFF).to_bytes(2, self.order)
            data[offset + 2:offset + 6] = bytes(4)

    @staticmethod
    def forced_length(sec: ObjSection, offset: int, count: int, relnum: int) -> bool:
        """is_forced_insn_length: a FIXED / INSN32 / INSN16 at the same offset."""
        for k in range(relnum + 1, count):
            r = sec.relocs[k]
            if r.offset != offset:
                break
            if r.type in ("FIXED", "INSN32", "INSN16"):
                return True
        return False

    # -- update_content ----------------------------------------------------------

    def update_content(self, sec: ObjSection, address: int, count: int,
                       no_old_padding: bool) -> None:
        """Insert ``count`` bytes at ``address`` (add_bytes: the gap keeps
        the bytes that were there) or delete ``-count`` bytes before it,
        and move what follows: relocations, the branches of earlier
        transformations, the section's symbols."""
        d = sec.data
        if count > 0:
            d[address:address] = d[address:address + count].ljust(count, b"\0")
        else:
            del d[address + count:address]
        for r in sec.relocs:
            if r.offset >= address:
                if (no_old_padding and r.offset == address
                        and r.type in ("ALIGN", "FILL", "MAX")):
                    continue
                r.offset += count
        for b in sec.branches:
            if b.offset >= address:
                b.offset += count
            if b.destination >= address:
                b.destination += count
        for s in sec.symbols:
            if s.value >= address:
                s.value += count
        entries = sec.entries                  # (sorted by offset)
        for k in range(bisect.bisect_left(entries, address, key=lambda e: e.offset),
                       len(entries)):
            entries[k].offset += count

    # -- Nanomips_transformations::align -------------------------------------------

    def find_fill_max(self, sec: ObjSection, offset: int, relnum: int, count: int):
        fill, mx, fill_size = NOP16, M32, 2
        for k in range(relnum + 1, count):
            r = sec.relocs[k]
            if r.offset != offset:
                break
            if r.type == "FILL":         # (symbol 0: value 0, size 0)
                fill, fill_size = (r.sym.value, r.sym.size) if r.sym else (0, 0)
            elif r.type == "MAX":
                mx = r.sym.value if r.sym else 0
        return fill, mx, fill_size

    def align(self, sec: ObjSection, relnum: int, count: int) -> None:
        r = sec.relocs[relnum]
        sym = r.sym                     # value: the power of two, size: the padding
        address = (sec.address + r.offset) & M32
        new_padding = (_align_address(address, 1 << sym.value) - address) & M32
        old_padding = sym.size
        fill, mx, fill_size = self.find_fill_max(sec, r.offset, relnum, count)
        if new_padding > mx:
            new_padding = 0
        if new_padding == old_padding:
            return
        grow = new_padding - old_padding
        start = r.offset
        padding = [e for e in sec.entries if e.nop and start <= e.offset < start + old_padding]
        if grow < 0 and new_padding >= 2:
            at = start + new_padding - 2
            if self.read_insn(sec.data, at, 32) == NOP32:
                self.write_insn(sec.data, at, NOP16, 2)
                self.log.append(f"{sec.name}+{at:#x}: nop[32] is replaced with nop[16]")
        self.update_content(sec, start + old_padding, grow, old_padding == 0)
        sym.size = new_padding
        self.log.append(f"{sec.name}+{start + old_padding:#x}: {abs(grow)} bytes are "
                        f"{'added' if grow > 0 else 'removed'} due to the alignment "
                        "requirement")
        end = start + new_padding
        kept = [e for e in sec.entries if not any(e is p for p in padding)]
        for e in padding:
            if e.offset + e.size <= end:
                kept.append(e)
            elif e.offset + 2 == end:
                kept.append(Entry(e.offset, 2, e.line, True))
        if grow > 0:
            if fill_size > grow:
                fill, fill_size = NOP16, 2
            if fill_size not in (1, 2, 4, 6):
                raise LinkError(r.line, "internal error in write_insn: ld cannot pad this "
                                f".align again (its fill is {fill_size} bytes: a fill of 0 "
                                "loses its symbol)")
            at = start + old_padding
            for j in range(0, grow, fill_size):
                if fill_size == 1:
                    sec.data[at + j] = fill & 0xFF
                else:
                    self.write_insn(sec.data, at + j, fill, fill_size)
                    kept.append(Entry(at + j, fill_size, r.line, True))
        sec.entries = sorted(kept, key=lambda e: e.offset)

    # -- Nanomips_transformations::transform --------------------------------------

    def transform(self, sec: ObjSection, tmpl: Transform, ip: InsnProperty, ttype: str,
                  relnum: int, insn: int) -> None:
        r = sec.relocs[relnum]
        t = r.type
        r_offset = r.offset
        xreg = self.o.expand_reg
        if t == "GPREL_I32":
            r_offset -= 2
            treg, sreg = ip.treg(insn), 0
        elif t in ("PC_I32", "I32"):
            r_offset -= 2
            treg = ip.treg(insn)
            treg, sreg = treg << 5 | treg, xreg
        elif t == "PC25_S1":
            treg = sreg = xreg
        elif t == "PC21_S1":
            treg = ip.treg(insn)
            sreg = ip.sreg(insn) if ip.name[0] == "m" else treg
        elif t == "PC14_S1":
            if ip.name == "bposge32c":
                treg = sreg = 0
            else:
                treg, sreg = ip.treg(insn), ip.sreg(insn)
                if ttype == "PCREL16":
                    t16, s16 = treg & 7, sreg & 7
                    if ip.name.startswith("beqc"):
                        if s16 > t16:
                            treg, sreg = sreg, treg
                    elif ip.name.startswith("bnec"):
                        if s16 < t16:
                            treg, sreg = sreg, treg
                    else:
                        raise AssertionError(ip.name)
        elif t == "PC10_S1":
            treg = sreg = 0
        elif t in ("PC11_S1", "GOT_OFST", "LO12"):
            treg, sreg = ip.treg(insn), ip.sreg(insn)
        elif t in ("GPREL19_S2", "GPREL18", "GPREL17_S1"):
            treg, sreg = ip.treg(insn), xreg
        elif t in ("PC7_S1", "GPREL7_S2"):
            treg, sreg = ip.convert_treg(insn), 0
        elif t in ("PC4_S1", "LO4_S2"):
            treg, sreg = ip.convert_treg(insn), ip.convert_sreg(insn)
        else:
            raise AssertionError(f"no transformation for R_NANOMIPS_{t}")
        if t == "PC14_S1" and ttype == "PCREL32_TRAMP":
            sec.branches.append(Branch(t, r_offset, r_offset + 6))
            sec.branches.append(Branch("PC10_S1", r_offset + 4, r_offset + 4 + 6))
        elif t in ("PC14_S1", "PC11_S1") and ttype not in ("PCREL16", "PCREL16_ZERO",
                                                             "PCREL32_TRAMP"):
            sec.branches.append(Branch(t, r_offset, r_offset + 8))
        if ttype in ("DISCARD", "SAVERESTORE16"):
            r.offset, r.type, r.sym, r.addend = 0, "NONE", None, 0
        new_reloc = False
        offset = r_offset
        # the listing: the instruction's entry -> one per new instruction
        k = bisect.bisect_left(sec.entries, r_offset, key=lambda e: e.offset)
        while k < len(sec.entries) and sec.entries[k].offset == r_offset and sec.entries[k].nop:
            k += 1
        line = r.line
        if k < len(sec.entries) and sec.entries[k].offset == r_offset:
            line = sec.entries.pop(k).line
        new = []
        for tp in tmpl.insns:
            if tp.reloc != "NONE":
                at = offset + 2 if tp.size == 6 else offset
                if not new_reloc:
                    r.type, r.offset = tp.reloc, at
                    new_reloc = True
                else:
                    sec.relocs.append(Reloc(at, tp.reloc, r.sym, r.addend, r.line))
            self.write_insn(sec.data, offset, tp.encode(treg, sreg), tp.size)
            new.append(Entry(offset, tp.size, line))
            offset += tp.size
        sec.entries[k:k] = new

    # -- Nanomips_relax_insn ------------------------------------------------------

    def relax_find(self, insn: int, mask: int, r_type: str) -> InsnProperty | None:
        if self.o.insn32 and r_type not in ("GPREL_I32", "PC_I32", "PC21_S1", "GOT_CALL",
                                            "GOT_DISP", "GOT_PAGE", "GOT_OFST", "JALR32",
                                            "JALR16", "SAVERESTORE"):
            return None
        p = insn_property(insn, mask, r_type)
        if p is None:
            return None
        if r_type == "GPREL_I32":
            ok = p.has_transform("GPREL32", r_type) or p.has_transform("GPREL32_WORD", r_type)
        elif r_type == "PC_I32":
            ok = p.has_transform("PCREL32", r_type)
        elif r_type == "PC25_S1":
            ok = p.has_transform("PCREL16", r_type)
        elif r_type == "PC21_S1":
            ok = self.obj.pic and p.has_transform("DISCARD", r_type)
        elif r_type == "PC14_S1":
            if p.name == "bposge32c":
                return None
            sreg = p.sreg(insn)
            if (not p.has_transform("PCREL16", r_type) or not self.obj.nmf
                    or (not p.valid_regs(insn)
                        and not (p.has_transform("PCREL16_ZERO", r_type)
                                 and p.valid_treg(insn) and sreg == 0))):
                return None
            ok = not (p.name.startswith("beqc") and p.treg(insn) == sreg)
        elif r_type == "GPREL19_S2":
            ok = p.has_transform("GPREL16", r_type) and p.valid_treg(insn)
        elif r_type == "LO12":
            ok = (self.o.relax_lo12 and p.has_transform("ABS16", r_type)
                  and p.valid_regs(insn))
        else:
            ok = False         # the GOT / JALR / SAVERESTORE ones (not modelled)
        return p if ok else None

    def relax_type(self, p: InsnProperty, r: Reloc, insn: int, address: int) -> str:
        t = r.type
        sv = self.value(r.sym, r.addend)
        if t == "GPREL_I32":
            value = (sv - self.gp) & M32
            if value & 3 and not _overflow_unsigned(18, value):
                return "GPREL32"
            if not value & 3 and not _overflow_unsigned(21, value):
                return "GPREL32_WORD"
            return TT_NONE
        if t in ("PC_I32", "PC25_S1"):
            value = (sv - address - (6 if t == "PC_I32" else 4)) & M32
            if value & 0x80000000:
                value = (value + 2) & M32     # a backward target does not move
            if t == "PC_I32":
                return "PCREL32" if not value & 1 and not _overflow_signed(22, value) \
                    else TT_NONE
            return "PCREL16" if not _overflow_signed(11, value) else TT_NONE
        if t == "PC21_S1":
            return TT_NONE                     # the GP-setup removal (not modelled)
        if t == "PC14_S1":
            value = (sv - address - 4) & M32
            if p.name == "bposge32c":
                return TT_NONE
            sreg = p.sreg(insn)
            if value != 0 and sreg != 0 and not _overflow_unsigned(5, value):
                return "PCREL16"
            if sreg == 0 and not _overflow_signed(8, value):
                return "PCREL16_ZERO"
            return TT_NONE
        if t == "GPREL19_S2":
            value = (sv - self.gp) & M32
            return "GPREL16" if not _overflow_unsigned(9, value) else TT_NONE
        if t == "LO12":
            value = sv & 0xFFF
            return "ABS16" if not value & 3 and not _overflow_unsigned(6, value) \
                else TT_NONE
        return TT_NONE

    # -- Nanomips_expand_insn -----------------------------------------------------

    _EXPANDABLE = frozenset({"PC25_S1", "PC21_S1", "PC14_S1", "PC11_S1", "GPREL19_S2",
                             "GPREL18", "GPREL17_S1", "PC10_S1", "PC7_S1", "PC4_S1",
                             "GPREL7_S2", "LO4_S2", "PC_I32", "I32"})

    def expand_find(self, insn: int, mask: int, r_type: str) -> InsnProperty | None:
        if r_type not in self._EXPANDABLE:
            return None              # (the GOT / TLS ones: not modelled)
        return insn_property(insn, mask, r_type)

    def _hw110880(self, value: int) -> bool:
        if not self.o.fix_nmips_hw110880:
            return False
        return (value & 0x50016400) == 0x50012400 and (
            (value >> 24) & 0x7 == 0x1 or (value >> 24) & 0x2 == 0x2)

    def _hw113064(self, value: int) -> bool:
        s = _signed(value)
        return self.o.fix_nmips_hw113064 and (s <= _signed(0xFE01FFFE) or s >= 0x1FE0000)

    def _hw142543(self, address: int, target: int) -> bool:
        return (self.o.fix_nmips_hw142543 and address & 0x1FFFE == 0x1FFFE
                and target & 0x1FFFF == 0)

    def expand_type(self, p: InsnProperty, r: Reloc, insn: int, address: int) -> str:
        t = r.type
        sv = self.value(r.sym, r.addend)
        pc4 = (sv - address - 4) & M32
        pc2 = (sv - address - 2) & M32
        gprel = (sv - self.gp) & M32
        if t == "PC25_S1":
            if not _overflow_signed(26, pc4):
                if self._hw113064(pc4) or self._hw142543(address, sv):
                    return "PCREL_NMF"
                return TT_NONE
        elif t == "PC21_S1":
            if not _overflow_signed(22, pc4):
                if self._hw142543(address, sv):
                    return "PCREL32_LONG"
                if not pc4 & 1:
                    return TT_NONE
        elif t == "PC14_S1":
            if not _overflow_signed(15, pc4):
                return TT_NONE
        elif t == "PC11_S1":
            if not _overflow_signed(12, pc4):
                return TT_NONE
        elif t == "GPREL19_S2":
            if not _overflow_unsigned(21, gprel):
                return TT_NONE
        elif t in ("GPREL18", "GPREL17_S1"):
            if not _overflow_unsigned(18, gprel):
                return TT_NONE
        elif t == "PC10_S1":
            if not _overflow_signed(11, pc2):
                return TT_NONE
        elif t == "PC7_S1":
            if not _overflow_signed(8, pc2):
                return TT_NONE
        elif t == "PC4_S1":
            if pc2 != 0 and not _overflow_unsigned(5, pc2):
                return TT_NONE
        elif t == "GPREL7_S2":
            if not _overflow_unsigned(9, gprel):
                return TT_NONE
        elif t == "LO4_S2":
            v = sv & 0xFFF
            if not v & 3 and not _overflow_unsigned(6, v):
                return TT_NONE
        elif t == "PC_I32":
            if not self._hw110880((sv - address - 6) & M32):
                return TT_NONE
        elif t == "I32":
            if not self._hw110880(sv):
                return TT_NONE
        return self._expand_kind(p, insn, t)

    def _expand_kind(self, p: InsnProperty, insn: int, t: str) -> str:
        """Nanomips_expand_insn::expand_type."""
        nmf, pcrel = self.obj.nmf, self.obj.pcrel
        if t == "PC25_S1":
            if nmf:
                return "PCREL_NMF" if pcrel else "ABS_NMF"
            if not pcrel:
                return "ABS32_LONG" if self.o.insn32 else "ABS16_LONG"
            return "PCREL32_LONG" if self.o.insn32 else "PCREL16_LONG"
        if t == "PC21_S1":
            if p.name[0] == "m":
                return "PCREL32_LONG"
            if nmf:
                return "PCREL_NMF" if pcrel else "ABS_NMF"
            return "PCREL32_LONG" if pcrel else "ABS32_LONG"
        if t == "PC14_S1" and p.name == "bposge32c":
            return "PCREL32_TRAMP"
        if t in ("PC14_S1", "PC11_S1"):
            return "PCREL32_LONG"
        if t in ("GPREL19_S2", "GPREL18", "GPREL17_S1"):
            if p.name[0] == "a":
                if nmf:
                    return "GPREL_NMF"
                if self.o.strict_address_modes:
                    return "GPREL_LONG"
                return "PCREL32_LONG" if pcrel else "ABS32_LONG"
            if self.o.strict_address_modes:
                return "GPREL32_NMF" if nmf else "GPREL_LONG"
            if nmf and p.has_transform("PCREL_NMF", t):
                return "PCREL_NMF"
            return "PCREL32_LONG" if pcrel else "ABS32_LONG"
        if t in ("PC10_S1", "PC7_S1"):
            return "PCREL32"
        if t == "PC4_S1":
            return "BNEC32" if p.sreg(insn) >= p.treg(insn) else "BEQC32"
        if t == "GPREL7_S2":
            return "GPREL32_WORD"
        if t == "LO4_S2":
            return "ABS32"
        if t in ("PC_I32", "I32"):
            return "IMM48_FIX" if p.has_transform("IMM48_FIX", t) else TT_NONE
        raise AssertionError(t)

    # -- Target_nanomips::Relocate --------------------------------------------------

    def relocate_section(self, sec: ObjSection) -> None:
        calculate_only = False
        calculated = 0
        n = len(sec.relocs)
        for i, r in enumerate(sec.relocs):
            prop = RELOCS[r.type]
            if prop.placeholder:
                continue
            nxt = sec.relocs[i + 1] if i + 1 < n else None
            addend = calculated if calculate_only else r.addend
            calculate_only = (nxt is not None and r.offset == nxt.offset
                              and nxt.type != "NONE" and not RELOCS[nxt.type].placeholder)
            if r.sym is not None and not r.sym.defined:
                raise LinkError(r.line, f"undefined reference to `{r.sym.name}'")
            P = (sec.address + r.offset) & M32
            S_A = self.value(r.sym, addend)
            t = r.type
            if t == "NONE":
                value = 0
            elif t == "ASHIFTR_1":
                value = (_signed(S_A) >> 1) & M32
            elif t == "NEG":
                value = (addend - self.value(r.sym, 0)) & M32
            elif t == "PC_HI20":
                value = (S_A - ((P + 4) & ~0xFFF)) & M32
            elif t == "PC32":
                value = (S_A - P) & M32
            elif t in ("PC_I32", "PC25_S1", "PC21_S1", "PC14_S1", "PC11_S1", "PC10_S1",
                       "PC7_S1", "PC4_S1"):
                value = (S_A - (P + (2 if prop.size == 16 else 4))) & M32
            elif t.startswith("GPREL"):
                value = (S_A - self.gp) & M32
            else:
                value = S_A
            if calculate_only:
                calculated = value
                continue
            status = self.apply(sec.data, r.offset, t, value)
            if status == "overflow":
                raise LinkError(r.line, f"relocation overflow: R_NANOMIPS_{t} against "
                                f"{self.sym_name(r)}: cannot encode a value of {value:#x} "
                                f"in a field of {prop.bitsize} bits")
            if status == "unaligned":
                raise LinkError(r.line, f"unaligned relocation value: R_NANOMIPS_{t} "
                                f"against {self.sym_name(r)}: value {value:#x} is not "
                                "aligned")

    def relocate_branch(self, sec: ObjSection, b: Branch) -> None:
        """Target_nanomips::relocate_branch."""
        size = 2 if RELOCS[b.reloc].size == 16 else 4
        value = (b.destination - (b.offset + size)) & M32
        if self.apply(sec.data, b.offset, b.reloc, value) is not None:
            raise AssertionError("a relaxed branch overflows")

    def apply(self, d: bytearray, w: int, t: str, value: int) -> str | None:
        """Nanomips_relocate_functions: write the field, check it."""
        order = self.order

        def rd16(o):
            return int.from_bytes(d[o:o + 2], order)

        def wr16(o, v):
            d[o:o + 2] = (v & 0xFFFF).to_bytes(2, order)

        def rd32():
            return rd16(w) << 16 | rd16(w + 2)

        def wr32(v):
            wr16(w, v >> 16)
            wr16(w + 2, v)

        def signed(bits):
            return "overflow" if _overflow_signed(bits, value) else None

        def unsigned(bits):
            return "overflow" if _overflow_unsigned(bits, value) else None

        def relpc(fieldsize, valsize):
            reloc = (value & ~1) | ((value >> (valsize - 1)) & 1)
            mask = (1 << (valsize - 1)) - 1
            if fieldsize == 16:
                wr16(w, (rd16(w) & ~mask) | (reloc & mask))
            else:
                wr32((rd32() & ~mask) | (reloc & mask))
            return "unaligned" if value & 1 else signed(valsize)

        def relgp(valsize, mask, align):
            wr32((rd32() & ~mask) | (value & mask))
            return "unaligned" if value & (align - 1) else unsigned(valsize)

        if t == "NONE":
            return None
        if t in ("32", "NEG", "ASHIFTR_1", "PC32"):
            d[w:w + 4] = value.to_bytes(4, order)
            return signed(32) if t == "PC32" else None
        if t in ("UNSIGNED_16", "SIGNED_16"):
            d[w:w + 2] = (value & 0xFFFF).to_bytes(2, order)
            return unsigned(16) if t == "UNSIGNED_16" else signed(16)
        if t in ("UNSIGNED_8", "SIGNED_8"):
            d[w] = value & 0xFF
            return unsigned(8) if t == "UNSIGNED_8" else signed(8)
        if t in ("I32", "PC_I32", "GPREL_I32"):
            if self._hw110880(value):
                return "unaligned"
            wr16(w, value)
            wr16(w + 2, value >> 16)
            return signed(32)
        if t in ("HI20", "PC_HI20", "GPREL_HI20"):
            wr32((rd32() & 0xFFE00002) | (value & 0x1FF000) | ((value >> 19) & 0xFFC)
                 | ((value >> 31) & 1))
            return None
        if t in ("LO12", "GPREL_LO12"):
            wr32((rd32() & ~0xFFF) | (value & 0xFFF))
            return None
        if t == "LO4_S2":
            v = value & 0xFFF
            wr16(w, (rd16(w) & ~0xF) | ((v >> 2) & 0xF))
            return "unaligned" if v & 3 else ("overflow" if v >= 64 else None)
        if t == "PC25_S1":
            return relpc(32, 26)
        if t == "PC21_S1":
            return relpc(32, 22)
        if t == "PC14_S1":
            return relpc(32, 15)
        if t == "PC11_S1":
            return relpc(32, 12)
        if t == "PC10_S1":
            return relpc(16, 11)
        if t == "PC7_S1":
            return relpc(16, 8)
        if t == "PC4_S1":
            wr16(w, (rd16(w) & ~0xF) | ((value >> 1) & 0xF))
            return "unaligned" if value & 1 else unsigned(5)
        if t == "GPREL19_S2":
            return relgp(21, 0x1FFFFC, 4)
        if t == "GPREL18_S3":
            return relgp(21, 0x1FFFF8, 8)
        if t == "GPREL18":
            return relgp(18, 0x3FFFF, 1)
        if t == "GPREL17_S1":
            return relgp(18, 0x3FFFE, 2)
        if t == "GPREL16_S2":
            return relgp(18, 0x3FFFC, 4)
        if t == "GPREL7_S2":
            wr16(w, (rd16(w) & ~0x7F) | ((value >> 2) & 0x7F))
            return "unaligned" if value & 3 else unsigned(9)
        raise LinkError(0, f"unsupported reloc R_NANOMIPS_{t}")


#: the data sections the layout places itself
DATA_ORDER = (".data", ".rdata", ".rodata", ".ssdata", ".sdata", ".ssbss", ".sbss", ".bss")


def link(obj: Object, bases: dict[str, int], options: LinkOptions | None = None) -> Linker:
    """Lay out, transform (``.linkrelax`` objects) and relocate ``obj``;
    the sections' ``data`` / ``address`` / ``entries`` and the symbols'
    values are the linked program's."""
    linker = Linker(obj, bases, options)
    linker.link()
    return linker
