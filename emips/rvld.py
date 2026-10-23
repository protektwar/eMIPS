"""The RISC-V linker (PLAN M25 phase 3): GNU ld 2.42's static link of one
object from :mod:`emips.rvasm`, after ``bfd/elfnn-riscv.c``
(``riscv_elf_relocate_section`` / ``perform_relocation``, the relaxation
of ``_bfd_riscv_relax_section``), ``ld/ldlang.c`` (``lang_relax_sections``)
and ld's default linker script for RISC-V ELF (``ld --verbose``), with
eMIPS's addresses.

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
symbol names; ``riscv_zero_pcrel_hi_reloc``), HI20 / LO12_I / _S, 32 / 64
and label differences (gas's ADD / SUB pairs), and the relaxations'
GPREL_I / _S and RVC_LUI.

Relaxation (``relax``: ld's default; False = ``--no-relax``, which still
runs pass 1, as riscvelf.em's TARGET_ENABLE_RELAXATION does): pass 0 makes
trips over the input sections in layout order — each section relaxed
before it is placed, so with this trip's addresses for the sections before
it and the last trip's for itself and those after (``_relax_call``: auipc +
jalr → c.j / c.jal / jal / jalr near zero; ``_relax_lui``: lui deleted for
a gp- or x0-relative access, lui → c.lui; ``_relax_pc``: auipc deleted, its
%pcrel_lo gp-relative), the deletions marked (R_RISCV_DELETE) and done at
the section's end — until a trip changes nothing; pass 1 then deletes the
excess nops of every R_RISCV_ALIGN at once.  Only relocations followed by
an R_RISCV_RELAX (gas -mrelax) are relaxed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
TEXT_BASE, DATA_BASE = 0x00400000, 0x10010000
NOP32, NOP16 = 0x00000013, 0x0001
MATCH_JAL, MATCH_JALR, MATCH_C_J, MATCH_C_JAL = 0x6F, 0x67, 0xA001, 0x2001
MATCH_C_LUI, MATCH_C_LI, MATCH_LUI, MASK_AUIPC = 0x6001, 0x4001, 0x37, 0x7F
X_RA, X_SP, X_GP = 1, 2, 3
ELF_MAXPAGESIZE = 0x1000
#: the relocations ld's pass 0 relaxes (when an R_RISCV_RELAX follows)
_RELAX_TYPES = {"CALL": "call", "CALL_PLT": "call", "HI20": "lui", "LO12_I": "lui",
                "LO12_S": "lui", "PCREL_HI20": "pc", "PCREL_LO12_I": "pc",
                "PCREL_LO12_S": "pc", "TPREL_HI20": "tls", "TPREL_ADD": "tls",
                "TPREL_LO12_I": "tls", "TPREL_LO12_S": "tls"}


@dataclass(eq=False)
class Reloc:
    """One relocation: gas's BFD code without BFD_RELOC_ / RISCV_ (12_PCREL,
    JMP, DATA4 ...) or ld's (GPREL_I, RVC_LUI, DELETE, NONE); ``sym`` the
    symbol (None = STN_UNDEF: gas's *ABS*), ``sub`` the subtracted symbol of
    an ADD / SUB pair."""
    offset: int
    type: str
    sym: object
    addend: int
    size: int
    line: int
    sub: object = None


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
    done: bool = False                               # sec_flg0: an ALIGN was handled


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
    """The result: section addresses, symbol values, the image, the
    instructions (address, size, source line)."""
    xlen: int
    sections: dict
    symbols: dict
    gp: int
    image: dict = field(default_factory=dict)
    segments: dict = field(default_factory=dict)
    insns: list = field(default_factory=list)


def _align(v: int, power: int) -> int:
    a = 1 << power
    return (v + a - 1) & ~(a - 1)


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _valid_i(v: int) -> bool:
    """VALID_ITYPE_IMM of a bfd_vma (64 bits)."""
    return -2048 <= _sx(v, 64) < 2048


def _valid_j(v: int) -> bool:
    v = _sx(v, 64)
    return -(1 << 20) <= v < 1 << 20 and not v & 1


def _valid_cj(v: int) -> bool:
    v = _sx(v, 64)
    return -2048 <= v < 2048 and not v & 1


def _high(v: int) -> int:
    """RISCV_CONST_HIGH_PART (64 bits)."""
    return (v + 0x800) & ~0xFFF & M64


def _valid_lui16(v: int) -> bool:
    """VALID_CITYPE_LUI_IMM: a non-zero 6-bit field that gives v back."""
    f = (v >> 12) & 0x3F
    return f != 0 and (_sx(f, 6) << 12) & M64 == v & M64


class Linker:
    def __init__(self, obj, bases: dict | None = None, relax: bool = True) -> None:
        self.obj = obj
        self.xlen = obj.xlen
        self.bases = {"text": TEXT_BASE, "data": DATA_BASE, **(bases or {})}
        self.relax = relax
        self.rvc = obj.rvc                                  # EF_RISCV_RVC
        # the symbol table: section-defined symbols -> [section, offset]; st_size
        self.loc: dict = {}
        self.size: dict = {}
        for sec in obj.sections.values():
            for r in sec.relocs:
                for s in (r.sym, r.sub):
                    self._place(s)
        for s in obj.symbols.values():
            self._place(s)
        self.by_out: dict[str, list] = {out: [] for out, _p, _s in LAYOUT}
        for name, sec in obj.sections.items():
            out = _output_of(name)
            if out is None:
                if sec.data or sec.relocs:
                    raise LinkError(0, f"section `{name}' has no place in eMIPS's memory map")
                continue
            self.by_out[out].append(sec)
        for v in self.by_out.values():          # .srodata before .sdata
            v.sort(key=lambda s: (0 if s.name.startswith(".srodata") else 1)
                   if s.name.startswith((".srodata", ".sdata")) else 0)
        self.inputs = [s for out, _p, _g in LAYOUT for s in self.by_out[out]]
        self.out_align = {out: max([s.align for s in secs], default=0)
                          for out, secs in self.by_out.items()}
        self.out_addr: dict[str, int] = {}
        self.marks: dict[str, int] = {}
        self.gp = 0
        self.max_alignment: int | None = None
        self.max_alignment_for_gp: int | None = None
        self.first_section = None

    def _place(self, s) -> None:
        if s is None or s in self.loc:
            return
        if s.frag is not None and s.equated is None and s.absolute is None:
            self.loc[s] = [self.obj.sections[s.section.name], s.frag.address + s.where]
            self.size[s] = self.obj.sizes.get(s, 0)

    # -- layout (lang_size_sections; the script's symbols) -----------------------------

    def _size_sections(self, relax_pass: int | None = None) -> bool:
        """Place every input section in layout order; with ``relax_pass``
        each is relaxed first, at the address it had."""
        again = False
        dot = {"text": self.bases["text"], "data": self.bases["data"]}
        marks: dict[str, int] = {}
        # lang_reset_memory_regions: every output section's size is 0 until
        # this pass has sized it (the relaxation sees that)
        self.out_size = dict.fromkeys(self.by_out, 0)
        for out, _p, seg in LAYOUT:
            secs = self.by_out[out]
            start = _align(dot[seg], self.out_align[out])
            self.out_addr[out] = start
            if out == ".data":
                marks["__DATA_BEGIN__"] = start
            if out == ".sdata":
                marks["__SDATA_BEGIN__"] = start
            addr = start
            for s in secs:
                if relax_pass is not None and self.relax_section(s, relax_pass):
                    again = True
                addr = _align(addr, s.align)
                s.address = addr
                addr += len(s.data)
            if out == ".bss" and addr != start:
                addr = _align(addr, 3 if self.xlen == 64 else 2)
            if secs:
                self.out_size[out] = addr - start
                dot[seg] = addr
        bss_end = _align(dot["data"], 3 if self.xlen == 64 else 2)
        self.marks = marks
        self.gp = min(marks["__SDATA_BEGIN__"] + 0x800,
                      max(marks["__DATA_BEGIN__"] + 0x800, bss_end - 0x800))
        return again

    # -- symbols ---------------------------------------------------------------------

    def sym_value(self, s, line: int) -> int:
        """A symbol's address (an equate: its expression's)."""
        if s.equated is not None:
            e = s.equated[0]
            return e.num + (self.sym_value(e.sym, line) if e.sym is not None else 0) - \
                (self.sym_value(e.sub, line) if getattr(e, "sub", None) is not None else 0)
        if s.absolute is not None:
            return s.absolute
        if s in self.loc:
            sec, off = self.loc[s]
            return sec.address + off
        special = {"__global_pointer$": self.gp, **self.marks}
        if s.name in special:
            return special[s.name]
        if s.weak:
            return 0
        raise LinkError(line, f"undefined reference to `{s.name}'")

    def _undefined_weak(self, s) -> bool:
        return (s is not None and s.weak and not s.defined
                and s.name not in ("__global_pointer$", "__DATA_BEGIN__", "__SDATA_BEGIN__"))

    def value(self, r: Reloc) -> int:
        """S + A (- the subtracted symbol)."""
        v = r.addend
        if r.sym is not None:
            v += self.sym_value(r.sym, r.line)
        if r.sub is not None:
            v -= self.sym_value(r.sub, r.line)
        return v

    # -- relaxation (_bfd_riscv_relax_section) -----------------------------------------

    def _max_alignment(self, gp: int) -> int:
        """_bfd_riscv_get_max_alignment: every output section's (gp != 0:
        those with an end within reach of gp)."""
        power = 0
        for out, _p, _s in LAYOUT:
            if not self.by_out[out]:
                continue
            addr, size = self.out_addr[out], self.out_size[out]
            if gp and not (_valid_i(addr - gp) or _valid_i(addr + size - gp)):
                continue
            power = max(power, self.out_align[out])
        return 1 << power

    def relax_section(self, sec: ObjSection, relax_pass: int) -> bool:
        if sec.done or not sec.relocs or sec.bss:
            return False
        if relax_pass == 0 and not self.relax:   # --no-relax: the alignment pass only
            return False
        if self.first_section is None:
            self.first_section = sec
        elif self.first_section is sec:
            self.max_alignment_for_gp = None
        if self.max_alignment is None:
            self.max_alignment = self._max_alignment(0)
        pcgp = {"hi": [], "lo": []}
        again = [False]
        relocs = sec.relocs
        i = 0
        while i < len(relocs):
            rel = relocs[i]
            if relax_pass == 0:
                kind = _RELAX_TYPES.get(rel.type)
                if kind is None or not (i + 1 < len(relocs) and relocs[i + 1].type == "RELAX"
                                        and relocs[i + 1].offset == rel.offset):
                    i += 1
                    continue
                i += 1                          # (the R_RISCV_RELAX: reused by deletions)
                delete = self._delete_piecewise
            elif rel.type == "ALIGN":
                kind, delete = "align", self._delete_immediate
            else:
                i += 1
                continue
            s = rel.sym
            undefined_weak = False
            reserve = 0
            if s is None:                       # STN_UNDEF: the reloc's own place
                sym_sec, symval = sec, rel.offset
            elif s in self.loc:
                sym_sec, symval = self.loc[s]
                size = self.size.get(s, 0)
                if not (s.external and s.type == "function"):
                    reserve = size - rel.addend if 0 <= rel.addend <= size else 0
            elif self._undefined_weak(s):
                if kind not in ("lui", "pc"):   # (a call of one: not relaxed)
                    i += 1
                    continue
                undefined_weak, sym_sec, symval = True, None, 0
            elif s.defined or s.name in self.marks or s.name == "__global_pointer$":
                symval, sym_sec = self.sym_value(s, rel.line), "abs"
            else:                               # undefined: refused when relocating
                i += 1
                continue
            symval += rel.addend
            if isinstance(sym_sec, ObjSection):
                symval += sym_sec.address
            getattr(self, "_relax_" + kind)(sec, sym_sec, relocs[i], rel, symval & M64,
                                            reserve, again, pcgp, undefined_weak, delete)
            i += 1
        self._resolve_delete_relocs(sec)
        return again[0]

    def _relax_call(self, sec, sym_sec, nxt, rel, symval, reserve, again, pcgp,
                    undefined_weak, delete) -> None:
        """_bfd_riscv_relax_call: auipc + jalr -> c.j / c.jal / jal / jalr."""
        d = sec.data
        foff = (symval - (sec.address + rel.offset)) & M64
        near_zero = (symval + 2048) & M64 < 4096
        max_alignment = self.max_alignment
        if _valid_j(foff):
            if isinstance(sym_sec, ObjSection) and \
                    _output_of(sym_sec.name) == _output_of(sec.name):
                max_alignment = 1 << self.out_align[_output_of(sec.name)]
            foff = (foff - max_alignment if _sx(foff, 64) < 0 else foff + max_alignment) & M64
        if not _valid_j(foff) and not near_zero:
            return
        o = rel.offset
        rd = int.from_bytes(d[o + 4:o + 8], "little") >> 7 & 31
        rvc = self.rvc and _valid_cj(foff) and (rd == 0 or (rd == X_RA and self.xlen == 32))
        if rvc:
            rel.type, insn, n = "RVC_JUMP", (MATCH_C_J if rd == 0 else MATCH_C_JAL), 2
        elif _valid_j(foff):
            rel.type, insn, n = "JMP", MATCH_JAL | rd << 7, 4
        else:                                   # near zero: jalr rd, x0, addr
            rel.type, insn, n = "LO12_I", MATCH_JALR | rd << 7, 4
        d[o:o + n] = insn.to_bytes(n, "little")
        again[0] = True
        delete(sec, o + n, 8 - n, nxt)

    def _gp_alignment(self, sym_sec) -> int:
        """The alignment relax_lui / relax_pc allow for: gp's output section's
        when the symbol is in it, else the sections near gp's."""
        if isinstance(sym_sec, ObjSection) and _output_of(sym_sec.name) == self.gp_section:
            return 1 << self.out_align[self.gp_section]
        if self.max_alignment_for_gp is None:
            self.max_alignment_for_gp = self._max_alignment(self.gp)
        return self.max_alignment_for_gp

    #: __global_pointer$'s output section (None: an absolute symbol, ld's
    #: result for MIN / MAX of values in different sections)
    gp_section: str | None = None

    def _in_gp_reach(self, symval: int, alignment: int, reserve: int,
                     undefined_weak: bool) -> bool:
        gp = self.gp
        return (undefined_weak or _valid_i(symval)
                or (symval >= gp and _valid_i(symval - gp + alignment + reserve))
                or (symval < gp and _valid_i(symval - gp - alignment - reserve)))

    def _relax_lui(self, sec, sym_sec, nxt, rel, symval, reserve, again, pcgp,
                   undefined_weak, delete) -> None:
        """_bfd_riscv_relax_lui: a gp- / x0-relative access; lui -> c.lui."""
        alignment = self.max_alignment
        if not undefined_weak and self.gp:
            alignment = self._gp_alignment(sym_sec)
        if self._in_gp_reach(symval, alignment, reserve, undefined_weak):
            if rel.type == "LO12_I":
                rel.type = "GPREL_I"
            elif rel.type == "LO12_S":
                rel.type = "GPREL_S"
            else:                               # HI20: the lui goes, the reloc with it
                again[0] = True
                delete(sec, rel.offset, 4, rel)
            return
        if (self.rvc and rel.type == "HI20" and _valid_lui16(_high(symval))
                and _valid_lui16((_high(symval) + ELF_MAXPAGESIZE) & M64)):
            d, o = sec.data, rel.offset
            lui = int.from_bytes(d[o:o + 4], "little")
            if lui >> 7 & 31 in (0, X_SP):
                return
            d[o:o + 4] = ((lui & (31 << 7)) | MATCH_C_LUI).to_bytes(4, "little")
            rel.type = "RVC_LUI"
            again[0] = True
            delete(sec, o + 2, 2, nxt)

    def _relax_tls(self, *args) -> None:
        """TPREL: thread-local storage is refused when relocating."""

    def _relax_pc(self, sec, sym_sec, nxt, rel, symval, reserve, again, pcgp,
                  undefined_weak, delete) -> None:
        """_bfd_riscv_relax_pc: the auipc deleted, its %pcrel_lo gp-relative."""
        hi = None
        if rel.type in ("PCREL_LO12_I", "PCREL_LO12_S"):
            base = sym_sec.address if isinstance(sym_sec, ObjSection) else 0
            hi_sec_off = (symval - base - rel.addend) & M64
            hi = next((h for h in pcgp["hi"] if h["off"] == hi_sec_off), None)
            if hi is None:
                pcgp["lo"].append(hi_sec_off)
                return
            symval, sym_sec, undefined_weak = hi["addr"], hi["sec"], hi["weak"]
        else:                                   # PCREL_HI20
            if not undefined_weak and isinstance(sym_sec, ObjSection) and sym_sec.code:
                return                          # (code may move out of reach later)
            if rel.offset in pcgp["lo"]:
                return
        alignment = self.max_alignment
        if not undefined_weak and self.gp:
            alignment = self._gp_alignment(sym_sec)
        if not self._in_gp_reach(symval, alignment, reserve, undefined_weak):
            return
        if hi is not None:
            rel.type = "GPREL_I" if rel.type == "PCREL_LO12_I" else "GPREL_S"
            rel.sym = hi["sym"]
            rel.addend += hi["addend"]
            return
        pcgp["hi"].append({"off": rel.offset, "addend": rel.addend, "addr": symval,
                           "sym": rel.sym, "sec": sym_sec, "weak": undefined_weak})
        again[0] = True
        delete(sec, rel.offset, 4, rel)

    def _relax_align(self, sec, sym_sec, nxt, rel, symval, reserve, again, pcgp,
                     undefined_weak, delete) -> None:
        """_bfd_riscv_relax_align: the nops beyond the alignment deleted;
        nothing more in this section is relaxed afterwards."""
        alignment = 1
        while alignment <= rel.addend:
            alignment *= 2
        symval -= rel.addend
        nop_bytes = ((symval - 1) & ~(alignment - 1)) + alignment - symval
        sec.done = True
        if rel.addend < nop_bytes:
            raise LinkError(rel.line, f"{nop_bytes} bytes required for alignment to "
                                      f"{alignment}-byte boundary, but only {rel.addend} "
                                      "present")
        rel.type = "NONE"
        if nop_bytes == rel.addend:
            return
        d, o = sec.data, rel.offset
        pos = 0
        while pos < nop_bytes & -4:
            d[o + pos:o + pos + 4] = NOP32.to_bytes(4, "little")
            pos += 4
        if nop_bytes % 4:
            d[o + pos:o + pos + 2] = NOP16.to_bytes(2, "little")
        delete(sec, o + nop_bytes, rel.addend - nop_bytes, None)

    # -- deleting bytes ----------------------------------------------------------------

    def _delete_piecewise(self, sec, addr: int, count: int, rel) -> None:
        """_riscv_relax_delete_piecewise: the reloc becomes R_RISCV_DELETE."""
        rel.type, rel.offset, rel.addend, rel.sym, rel.sub = "DELETE", addr, count, None, None

    def _delete_immediate(self, sec, addr: int, count: int, rel) -> None:
        if rel is not None:
            rel.type = "NONE"
        self._delete_bytes(sec, addr, count)

    def _delete_bytes(self, sec, addr: int, count: int) -> None:
        """_riscv_relax_delete_bytes: relocs in (addr, end), symbols in
        (addr, end] move down; a symbol spanning the hole shrinks.  (The
        pcgp table is never passed here in 2.42: deletions are piecewise.)"""
        end = len(sec.data)
        del sec.data[addr:addr + count]
        for r in sec.relocs:
            if addr < r.offset < end:
                r.offset -= count
        for s, place in self.loc.items():
            if place[0] is not sec:
                continue
            v, size = place[1], self.size.get(s, 0)
            if addr < v <= end:
                place[1] = v - count
            elif v <= addr < v + size <= end:
                self.size[s] = size - count
        insns = []
        for off, n, line in sec.insns:          # (the listing's instructions)
            if addr <= off < addr + count:
                continue
            if off < addr < off + n:
                n -= min(count, off + n - addr)
            insns.append((off - count if off > addr else off, n, line))
        sec.insns = insns

    def _resolve_delete_relocs(self, sec) -> None:
        """riscv_relax_resolve_delete_relocs: every R_RISCV_DELETE, in order."""
        for r in list(sec.relocs):
            if r.type == "DELETE":
                self._delete_bytes(sec, r.offset, r.addend)
                r.type = "NONE"

    # -- the link ----------------------------------------------------------------------

    def link(self) -> Linked:
        self._size_sections()
        for relax_pass in (0, 1):               # lang_relax_sections (relax_pass = 2)
            while self._size_sections(relax_pass):
                pass
        self._size_sections()                   # (the final sizing)
        for sec in self.inputs:
            self.relocate_section(sec)
        mask = M32 if self.xlen == 32 else M64
        symbols = {}
        for name, s in self.obj.symbols.items():
            if s.defined and not name.startswith(".L"):
                symbols[name] = self.sym_value(s, 0) & mask
        symbols.setdefault("__global_pointer$", self.gp)
        out = Linked(self.xlen, self.obj.sections, symbols, self.gp)
        for sec in self.obj.sections.values():
            if sec.bss:
                continue
            for i, b in enumerate(sec.data):
                out.image[sec.address + i] = b
            out.insns += [(sec.address + off, n, line) for off, n, line in sec.insns]
        for name, sec in self.obj.sections.items():
            if sec.data or sec.bss:
                out.segments[name] = (sec.address, sec.address + len(sec.data))
        return out

    # -- relocating (riscv_elf_relocate_section) ---------------------------------------

    def _overflow(self, r: Reloc) -> None:
        name = r.sym.name if r.sym is not None else "*ABS*"
        raise LinkError(r.line, f"relocation truncated to fit: "
                                f"R_RISCV_{_RNAME.get(r.type, r.type)} against `{name}'")

    def relocate_section(self, sec: ObjSection) -> None:
        from .rvasm import enc_b, enc_cb, enc_cj, enc_i, enc_j, enc_s, enc_u, high_part
        d = sec.data
        mask = M32 if self.xlen == 32 else M64
        pcrel_hi: dict[int, int] = {}
        lo_relocs: list = []

        def get(o: int, n: int) -> int:
            return int.from_bytes(d[o:o + n], "little")

        def put(o: int, n: int, v: int) -> None:
            d[o:o + n] = (v & ((1 << (8 * n)) - 1)).to_bytes(n, "little")

        for r in sec.relocs:
            t, o = r.type, r.offset
            p = sec.address + o
            if t in ("ALIGN", "RELAX", "NONE", "DELETE"):
                continue
            if t in ("TPREL_HI20", "TPREL_LO12_I", "TPREL_LO12_S", "TPREL_ADD", "TLS_GOT_HI20",
                     "TLS_GD_HI20", "GOT_HI20"):
                raise LinkError(r.line, f"R_RISCV_{_RNAME.get(t, t)}: thread-local storage and "
                                        "the GOT are not supported (a static link of position-"
                                        "dependent code)")
            if t.startswith("DATA"):
                n = int(t[4:])
                if r.sub is None and n in (1, 2):
                    raise LinkError(r.line, f"cannot represent BFD_RELOC_{8 * n} relocation in "
                                            "object file")
                put(o, n, self.value(r))
                continue
            if t in ("PCREL_LO12_I", "PCREL_LO12_S"):
                # (found by the label's address; the addend is the target's)
                label = self.sym_value(r.sym, r.line) if r.sym is not None else 0
                lo_relocs.append((r, label & mask))
                continue
            v = self.value(r)
            if t == "12_PCREL":
                v -= p
                if v & 1 or not -4096 <= v < 4096:
                    self._overflow(r)
                put(o, 4, get(o, 4) & ~enc_b(-1) | enc_b(v))
            elif t == "JMP":
                v -= p
                if v & 1 or not -(1 << 20) <= v < 1 << 20:
                    self._overflow(r)
                put(o, 4, get(o, 4) & ~enc_j(-1) | enc_j(v))
            elif t == "RVC_BRANCH":
                v -= p
                if v & 1 or not -256 <= v < 256:
                    self._overflow(r)
                put(o, 2, get(o, 2) & ~enc_cb(-1) | enc_cb(v))
            elif t == "RVC_JUMP":
                v -= p
                if v & 1 or not -2048 <= v < 2048:
                    self._overflow(r)
                put(o, 2, get(o, 2) & ~enc_cj(-1) | enc_cj(v))
            elif t in ("CALL", "CALL_PLT"):
                if self._undefined_weak(r.sym):     # (x0 as the base: the address 0)
                    put(o + 4, 4, get(o + 4, 4) & ~(31 << 15))
                    v = p + r.addend
                v -= p
                if not self._utype_ok(high_part(v)):
                    self._overflow(r)
                put(o, 4, get(o, 4) & ~enc_u(-1) | enc_u(high_part(v)))
                put(o + 4, 4, get(o + 4, 4) & ~enc_i(-1) | enc_i(v))
            elif t == "PCREL_HI20":
                if not self._zero_pcrel_hi(d, o, p, v):
                    v -= p
                    if not self._utype_ok(high_part(v)):
                        self._overflow(r)
                pcrel_hi[p & mask] = v
                put(o, 4, get(o, 4) & ~enc_u(-1) | enc_u(high_part(v)))
            elif t == "HI20":
                if not self._utype_ok(high_part(v)):
                    self._overflow(r)
                put(o, 4, get(o, 4) & ~enc_u(-1) | enc_u(high_part(v)))
            elif t in ("LO12_I", "LO12_S"):
                enc = enc_i if t.endswith("_I") else enc_s
                put(o, 4, get(o, 4) & ~enc(-1) | enc(v))
            elif t in ("GPREL_I", "GPREL_S"):
                x0 = _valid_i(v)
                if not x0 and not _valid_i(v - self.gp):
                    self._overflow(r)
                insn = get(o, 4) & ~(31 << 15)
                if not x0:
                    v -= self.gp
                    insn |= X_GP << 15
                enc = enc_i if t.endswith("_I") else enc_s
                put(o, 4, insn & ~enc(-1) | enc(v))
            elif t == "RVC_LUI":
                hi = _high(v)
                if hi == 0:                     # (relaxed below 0x800: c.li rd, 0)
                    put(o, 2, ((get(o, 2) & ~MATCH_C_LUI) | MATCH_C_LI) & ~0x107C)
                elif not _valid_lui16(hi):
                    self._overflow(r)
                else:
                    f = (hi >> 12) & 0x3F
                    put(o, 2, get(o, 2) & ~0x107C | (f & 31) << 2 | (f >> 5) << 12)
            else:
                raise LinkError(r.line, f"unsupported relocation R_RISCV_{t}")
        for r, label in lo_relocs:                  # riscv_resolve_pcrel_lo_relocs
            hi = pcrel_hi.get(label)
            if hi is None:
                raise LinkError(r.line, "dangerous relocation: %pcrel_lo missing matching "
                                        "%pcrel_hi")
            if high_part(hi) != high_part(hi + r.addend):
                raise LinkError(r.line, "dangerous relocation: %pcrel_lo overflow with an "
                                        "addend")
            enc = enc_i if r.type.endswith("_I") else enc_s
            put(r.offset, 4, get(r.offset, 4) & ~enc(-1) | enc(hi + r.addend))

    def _zero_pcrel_hi(self, d, o: int, pc: int, addr: int) -> bool:
        """riscv_zero_pcrel_hi_reloc: an auipc that cannot reach becomes a
        lui of the absolute address (RV64)."""
        from .rvasm import high_part
        if self.xlen == 32 or self._utype_ok(high_part(addr - pc)):
            return False
        if not self._utype_ok(high_part(addr)):
            return False
        insn = int.from_bytes(d[o:o + 4], "little")
        d[o:o + 4] = ((insn & ~MASK_AUIPC) | MATCH_LUI).to_bytes(4, "little")
        return True

    def _utype_ok(self, v: int) -> bool:
        """VALID_UTYPE_IMM of a 64-bit value: what lui / auipc can make
        (sign-extended from bit 31; RV32: any 32-bit value)."""
        if self.xlen == 32:
            v &= M32
            v = v - (1 << 32) if v >> 31 else v
        return -(1 << 31) <= v < 1 << 31


#: gas's BFD codes -> the R_RISCV_ names
_RNAME = {"12_PCREL": "BRANCH", "JMP": "JAL", "DATA4": "32", "DATA8": "64"}


def link(obj, bases: dict | None = None, relax: bool = True) -> Linked:
    return Linker(obj, bases, relax).link()
