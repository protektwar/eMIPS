"""ELF32 MIPS executables (PLAN M13): the loadable segments, the entry, the
symbols, the MIPS32 release the code was built for, and — from the DWARF
line table — the source line of every instruction.  Qt-free, no
dependencies.

References: the *System V ABI* (ELF header, program / section headers,
symbol table), its *MIPS RISC Processor Supplement* (EM_MIPS = 8, the
``e_flags`` bits: EF_MIPS_ARCH 31:28 — 5 MIPS32, 7 MIPS32 Release 2, 9
MIPS32 Release 6 —, EF_MIPS_NAN2008 0x400, EF_MIPS_PIC / CPIC 2 / 4,
EF_MIPS_ARCH_ASE_MICROMIPS 0x02000000, EF_MIPS_ARCH_ASE_M16 0x04000000;
as binutils' ``include/elf/mips.h``), and the *DWARF Debugging
Information Format* versions 2-5, section 6.2 "Line Number Information"
(the line number program and its state machine; version 5's directory /
file tables with their content descriptions).

What eMIPS loads: an executable (``ET_EXEC``) of 32 bits, either byte
order; every ``PT_LOAD`` segment — ``p_filesz`` bytes from the file, the
rest of ``p_memsz`` zeros (``.bss``) — at its virtual address (the
loader puts each byte where the MMU maps it, as for assembled programs);
``e_entry``.  Relocatable objects (``ET_REL``: link them first), shared
objects and 64-bit files are refused with the reason; MIPS16e and
microMIPS code is listed and run in its ISA mode (``Symbol.isa``,
:meth:`ElfImage.isa_ranges`).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

EM_MIPS = 8
ET_EXEC = 2
PT_LOAD = 1
SHT_SYMTAB = 2
#: the MIPS ABI flags section (Elf_MIPS_ABIFlags_v0) and its DSP ASE bits
SHT_MIPS_ABIFLAGS = 0x7000002A
AFL_ASE_DSP, AFL_ASE_DSPR2, AFL_ASE_DSPR3 = 0x1, 0x2, 0x2000
STT_OBJECT, STT_FUNC = 1, 2
#: e_flags (binutils include/elf/mips.h)
EF_MIPS_PIC, EF_MIPS_CPIC = 0x2, 0x4
EF_MIPS_NAN2008 = 0x400
EF_MIPS_MICROMIPS = 0x02000000
EF_MIPS_M16 = 0x04000000
#: st_other: the symbol is MIPS16e / microMIPS code
STO_MIPS16, STO_MICROMIPS = 0xF0, 0x80
EF_MIPS_ARCH = 0xF0000000
#: EF_MIPS_ARCH -> eMIPS release (MIPS32 Release 2 binaries run on the
#: full pre-R6 set, R5: every later release keeps their instructions)
ARCH_RELEASE = {0x50000000: "R1", 0x70000000: "R5", 0x90000000: "R6"}
ARCH_NAMES = {0x50000000: "MIPS32", 0x70000000: "MIPS32 Release 2",
              0x90000000: "MIPS32 Release 6", 0x60000000: "MIPS64",
              0x80000000: "MIPS64 Release 2", 0xA0000000: "MIPS64 Release 6"}


class ElfError(ValueError):
    """Not an ELF eMIPS can run (the message says why)."""


@dataclass
class Symbol:
    name: str
    value: int
    size: int
    kind: str                       # "func", "object" or "other"
    #: the code's ISA (st_other: STO_MIPS16 0xF0, STO_MICROMIPS 0x80)
    isa: str = "mips32"


@dataclass
class LineRow:
    """A row of the DWARF line table: the instruction at ``address``
    starts (or continues) source line ``line`` of ``file``."""
    address: int
    file: str
    line: int
    is_stmt: bool = True
    end: bool = False               # end_sequence: the address after the code


@dataclass
class ElfImage:
    path: str
    endian: str                     # "big" / "little"
    entry: int
    flags: int
    #: virtual address -> byte (every PT_LOAD segment, .bss included)
    image: dict[int, int] = field(default_factory=dict)
    #: (vaddr, memsz, flags "RWX") of each loadable segment
    segments: list[tuple[int, int, str]] = field(default_factory=list)
    #: (name, address, size, executable) of each allocated section
    sections: list[tuple[str, int, int, bool]] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    lines: list[LineRow] = field(default_factory=list)
    #: things that may surprise (a PIC binary, no line table ...)
    notes: list[str] = field(default_factory=list)
    #: the DSP Module level its code needs (.MIPS.abiflags ases: AFL_ASE_DSP
    #: 1, AFL_ASE_DSPR2 / DSPR3 2; binutils include/elf/mips.h)
    dsp: int = 0

    @property
    def release(self) -> str:
        return ARCH_RELEASE.get(self.flags & EF_MIPS_ARCH, "R5")

    @property
    def arch_name(self) -> str:
        return ARCH_NAMES.get(self.flags & EF_MIPS_ARCH,
                              f"EF_MIPS_ARCH 0x{(self.flags >> 28) & 0xF:x}")

    @property
    def mips16(self) -> bool:
        return bool(self.flags & EF_MIPS_M16)

    @property
    def micromips(self) -> bool:
        return bool(self.flags & EF_MIPS_MICROMIPS)

    def isa_ranges(self) -> list[tuple[int, int, str]]:
        """``(start, end, isa)`` of the code: each executable range split
        at its symbols, a region taking the ISA of the symbol it starts at
        (what objdump does: STO_MIPS16 / STO_MICROMIPS mark 16-bit code)."""
        out = []
        # (at one address a MIPS16e / microMIPS symbol beats a MIPS32 one:
        # the linker's _ftext / __start next to the code's own)
        syms = sorted(((s.value, s.isa) for s in self.symbols),
                      key=lambda v: (v[0], v[1] != "mips32"))
        for start, stop in self.code_ranges():
            marks = [(a, isa) for a, isa in syms if start <= a < stop]
            cur, isa = start, "mips32"
            for a, s_isa in marks:
                if a > cur:
                    out.append((cur, a, isa))
                cur, isa = a, s_isa
            if stop > cur:
                out.append((cur, stop, isa))
        merged: list[tuple[int, int, str]] = []
        for r in out:
            if merged and merged[-1][2] == r[2] and merged[-1][1] == r[0]:
                merged[-1] = (merged[-1][0], r[1], r[2])
            else:
                merged.append(r)
        return merged

    def code_ranges(self) -> list[tuple[int, int]]:
        """``(start, end)`` of the code: the executable sections (else the
        executable segments)."""
        code = [(a, a + n) for _name, a, n, x in self.sections if x]
        return code or [(a, a + n) for a, n, f in self.segments if "X" in f]

    def statement_starts(self) -> dict[tuple[str, int], list[int]]:
        """(file, line) -> the addresses where a statement of that line
        begins: line-table rows with is_stmt whose line differs from the
        row before (a line spread over several places has several)."""
        out: dict[tuple[str, int], list[int]] = {}
        prev = None
        for r in self.lines:
            key = (r.file, r.line)
            if r.end:
                prev = None
                continue
            if r.is_stmt and key != prev:
                out.setdefault(key, [])
                if r.address not in out[key]:
                    out[key].append(r.address)
            prev = key
        for v in out.values():
            v.sort()
        return out

    def source_lines(self) -> dict[int, tuple[str, int]]:
        """Instruction address -> ``(file, line)``: the line-table row in
        force at every 4-byte address of the code it covers (every 2-byte
        one with MIPS16e / microMIPS code)."""
        out: dict[int, tuple[str, int]] = {}
        rows = self.lines
        step = 2 if self.mips16 or self.micromips else 4
        for k, r in enumerate(rows):
            if r.end or k + 1 >= len(rows):
                continue
            nxt = rows[k + 1]
            stop = nxt.address if nxt.address > r.address else r.address + step
            for a in range(r.address, stop, step):
                out[a] = (r.file, r.line)
        return out


# ------------------------------------------------------------------ reading

def read_elf(path: str | Path) -> ElfImage:
    """Parse the executable at ``path`` (see the module docstring)."""
    data = Path(path).read_bytes()
    return parse_elf(data, str(path))


def parse_elf(data: bytes, path: str = "<elf>") -> ElfImage:
    if data[:4] != b"\x7fELF":
        raise ElfError("not an ELF file")
    if data[4] != 1:
        raise ElfError("a 64-bit ELF (eMIPS runs MIPS32: ELFCLASS32)")
    if data[5] not in (1, 2):
        raise ElfError("unknown byte order (EI_DATA)")
    end = "<" if data[5] == 1 else ">"
    endian = "little" if data[5] == 1 else "big"
    (e_type, e_machine, _ver, e_entry, e_phoff, e_shoff, e_flags, _ehsize,
     e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx) = struct.unpack_from(
        end + "HHIIIIIHHHHHH", data, 16)
    if e_machine != EM_MIPS:
        raise ElfError(f"not a MIPS executable (e_machine {e_machine})")
    if e_type != ET_EXEC:
        kind = {1: "a relocatable object (.o): link it first",
                3: "a shared object / position-independent executable"}.get(
            e_type, f"ELF type {e_type}")
        raise ElfError(f"{kind} — eMIPS loads executables (ET_EXEC)")
    arch = e_flags & EF_MIPS_ARCH
    if arch in (0x60000000, 0x80000000, 0xA0000000, 0x00000000 + 0x30000000,
                0x40000000):
        raise ElfError(f"{ARCH_NAMES.get(arch, 'a MIPS III / IV / 64-bit')} code: "
                       "eMIPS runs MIPS32")
    img = ElfImage(path, endian, e_entry, e_flags)
    if e_flags & EF_MIPS_M16:
        img.notes.append("MIPS16e code (EF_MIPS_ARCH_ASE_M16): the core gets MIPS16e")
    if e_flags & EF_MIPS_MICROMIPS:
        img.notes.append("microMIPS code (EF_MIPS_ARCH_ASE_MICROMIPS): the core gets "
                         "microMIPS")
    if e_flags & (EF_MIPS_PIC | EF_MIPS_CPIC):
        img.notes.append("position-independent (abicalls) code: it expects $gp / $t9 "
                         "set up by a dynamic loader — build with -mno-abicalls -fno-pic")
    for i in range(e_phnum):
        (p_type, p_offset, p_vaddr, _paddr, p_filesz, p_memsz, p_flags,
         _align) = struct.unpack_from(end + "IIIIIIII", data, e_phoff + i * e_phentsize)
        if p_type != PT_LOAD or not p_memsz:
            continue
        chunk = data[p_offset:p_offset + p_filesz]
        for k in range(p_memsz):
            img.image[(p_vaddr + k) & 0xFFFFFFFF] = chunk[k] if k < len(chunk) else 0
        rwx = "".join(c for c, bit in (("R", 4), ("W", 2), ("X", 1)) if p_flags & bit)
        img.segments.append((p_vaddr, p_memsz, rwx))
    if not img.segments:
        raise ElfError("no loadable segment (PT_LOAD)")
    sections = _sections(data, end, e_shoff, e_shentsize, e_shnum, e_shstrndx)
    for name, sh_type, flags, addr, off, size, link, _info, entsize in sections:
        if flags & 2 and size:                         # SHF_ALLOC (SHF_EXECINSTR 4)
            img.sections.append((name, addr, size, bool(flags & 4)))
        if sh_type == SHT_SYMTAB:
            img.symbols = _symbols(data, end, off, size, entsize, sections[link])
    by_name = {s[0]: s for s in sections}
    flags = next((s for s in sections if s[1] == SHT_MIPS_ABIFLAGS and s[5] >= 24), None)
    if flags is not None:                          # Elf_MIPS_ABIFlags_v0.ases
        ases = struct.unpack_from(end + "I", data, flags[4] + 12)[0]
        img.dsp = 2 if ases & (AFL_ASE_DSPR2 | AFL_ASE_DSPR3) else 1 if ases & AFL_ASE_DSP else 0
        if img.dsp:
            img.notes.append("DSP Module code (.MIPS.abiflags: "
                             + ("Revision 2" if img.dsp > 1 else "Revision 1")
                             + "): the core gets the DSP Module")
    if ".debug_line" in by_name:
        s = by_name[".debug_line"]
        line_str = by_name.get(".debug_line_str")
        dstr = by_name.get(".debug_str")
        try:
            img.lines = read_line_table(
                data[s[4]:s[4] + s[5]], end,
                data[line_str[4]:line_str[4] + line_str[5]] if line_str else b"",
                data[dstr[4]:dstr[4] + dstr[5]] if dstr else b"")
        except (struct.error, IndexError, ValueError) as exc:
            img.notes.append(f"the DWARF line table could not be read ({exc})")
    else:
        img.notes.append("no DWARF line table (build with -g to see the source lines)")
    return img


def _sections(data: bytes, end: str, off: int, entsize: int, n: int, strndx: int) -> list:
    raw = [struct.unpack_from(end + "IIIIIIIIII", data, off + i * entsize) for i in range(n)]
    if not raw:
        return []
    names = raw[strndx] if strndx < len(raw) else None
    out = []
    for (name, sh_type, flags, addr, offset, size, link, info, _align, ents) in raw:
        text = _cstr(data, names[4] + name) if names else ""
        out.append((text, sh_type, flags, addr, offset, size, link, info, ents))
    return out


def _cstr(data: bytes, at: int) -> str:
    stop = data.index(b"\0", at)
    return data[at:stop].decode("utf-8", "replace")


def _symbols(data: bytes, end: str, off: int, size: int, entsize: int, strtab) -> list[Symbol]:
    out = []
    for k in range(size // (entsize or 16)):
        name, value, sz, info, other, shndx = struct.unpack_from(
            end + "IIIBBH", data, off + k * (entsize or 16))
        kind = {STT_FUNC: "func", STT_OBJECT: "object"}.get(info & 0xF, "other")
        if not name or shndx == 0 or (info & 0xF) in (3, 4):   # SECTION, FILE
            continue
        isa = {STO_MIPS16: "mips16e", STO_MICROMIPS: "micromips"}.get(other & 0xF0, "mips32")
        out.append(Symbol(_cstr(data, strtab[4] + name), value & ~1 if isa != "mips32"
                          else value, sz, kind, isa))
    return out


# ------------------------------------------------------------------ DWARF lines

def _uleb(b: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = b[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, i


def _sleb(b: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = b[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            if byte & 0x40:
                value -= 1 << shift
            return value, i


def read_line_table(sec: bytes, end: str, line_str: bytes = b"",
                    debug_str: bytes = b"") -> list[LineRow]:
    """Every row of every line-number program in ``.debug_line`` (DWARF
    6.2; versions 2-5, 32-bit DWARF)."""
    rows: list[LineRow] = []
    i = 0
    while i + 4 <= len(sec):
        unit_len, = struct.unpack_from(end + "I", sec, i)
        if unit_len == 0xFFFFFFFF:
            raise ValueError("64-bit DWARF")
        start, unit_end = i + 4, i + 4 + unit_len
        rows += _line_program(sec[start:unit_end], end, line_str, debug_str)
        i = unit_end
    return rows


def _line_program(u: bytes, end: str, line_str: bytes, debug_str: bytes) -> list[LineRow]:
    version, = struct.unpack_from(end + "H", u, 0)
    i = 2
    if version >= 5:
        i += 2                                       # address_size, seg_sel_size
    header_len, = struct.unpack_from(end + "I", u, i)
    i += 4
    prog = i + header_len
    min_inst = u[i]
    i += 1
    if version >= 4:
        i += 1                                       # maximum_operations_per_instruction
    default_stmt, line_base = u[i], struct.unpack_from("b", u, i + 1)[0]
    line_range, opcode_base = u[i + 2], u[i + 3]
    i += 4
    std_lengths = list(u[i:i + opcode_base - 1])
    i += opcode_base - 1
    if version >= 5:
        dirs, i = _entries(u, i, end, line_str, debug_str)
        files, i = _entries(u, i, end, line_str, debug_str)
        dirnames = [d.get(1, "") for d in dirs]
        filenames = [_join(dirnames, f.get(1, "?"), f.get(2, 0)) for f in files]
        base = 0                                     # DWARF 5: file 0 is the unit's
    else:
        dirnames = [""]
        while u[i]:
            stop = u.index(b"\0", i)
            dirnames.append(u[i:stop].decode("utf-8", "replace"))
            i = stop + 1
        i += 1
        filenames = [""]
        while u[i]:
            stop = u.index(b"\0", i)
            name = u[i:stop].decode("utf-8", "replace")
            i = stop + 1
            d, i = _uleb(u, i)
            _, i = _uleb(u, i)
            _, i = _uleb(u, i)
            filenames.append(_join(dirnames, name, d))
        base = 1
    i = prog
    rows: list[LineRow] = []

    def reset():
        return {"address": 0, "file": base, "line": 1, "is_stmt": bool(default_stmt)}

    st = reset()

    def emit(endseq: bool = False) -> None:
        f = st["file"]
        name = filenames[f] if 0 <= f < len(filenames) else f"file{f}"
        # (MIPS16e / microMIPS code: GCC puts the ISA bit in bit 0 of the address)
        rows.append(LineRow(st["address"] & ~1, name, st["line"], st["is_stmt"], endseq))

    while i < len(u):
        op = u[i]
        i += 1
        if op >= opcode_base:                        # special opcode
            adj = op - opcode_base
            st["address"] += (adj // line_range) * min_inst
            st["line"] += line_base + adj % line_range
            emit()
        elif op == 0:                                # extended
            length, i = _uleb(u, i)
            sub = u[i]
            body = u[i + 1:i + length]
            i += length
            if sub == 1:                             # end_sequence
                emit(True)
                st = reset()
            elif sub == 2:                           # set_address
                st["address"], = struct.unpack_from(end + "I", body, 0)
            elif sub == 3:                           # define_file (v2-4)
                stop = body.index(b"\0")
                filenames.append(body[:stop].decode("utf-8", "replace"))
        elif op == 1:                                # copy
            emit()
        elif op == 2:                                # advance_pc
            n, i = _uleb(u, i)
            st["address"] += n * min_inst
        elif op == 3:                                # advance_line
            n, i = _sleb(u, i)
            st["line"] += n
        elif op == 4:                                # set_file
            st["file"], i = _uleb(u, i)
        elif op == 5:                                # set_column
            _, i = _uleb(u, i)
        elif op == 6:                                # negate_stmt
            st["is_stmt"] = not st["is_stmt"]
        elif op in (7, 10, 11):                      # basic_block, prologue / epilogue
            pass
        elif op == 8:                                # const_add_pc
            st["address"] += ((255 - opcode_base) // line_range) * min_inst
        elif op == 9:                                # fixed_advance_pc
            n, = struct.unpack_from(end + "H", u, i)
            i += 2
            st["address"] += n
        elif op == 12:                               # set_isa
            _, i = _uleb(u, i)
        else:                                        # unknown standard opcode
            for _ in range(std_lengths[op - 1]):
                _, i = _uleb(u, i)
    return rows


def _join(dirs: list[str], name: str, d: int) -> str:
    if name.startswith("/") or not (0 <= d < len(dirs)) or not dirs[d]:
        return os.path.normpath(name)
    return os.path.normpath(f"{dirs[d]}/{name}")


#: DW_FORM codes used in DWARF 5 line-table entry formats
_FORMS = {0x08: "string", 0x0E: "strp", 0x1F: "line_strp", 0x0F: "udata",
          0x0B: "data1", 0x05: "data2", 0x06: "data4", 0x07: "data8",
          0x1E: "data16", 0x09: "block"}


def _entries(u: bytes, i: int, end: str, line_str: bytes,
             debug_str: bytes) -> tuple[list[dict], int]:
    """A DWARF 5 directory / file table: its entry format, then the
    entries (content type -> value: 1 path, 2 directory index, ...)."""
    nfmt = u[i]
    i += 1
    fmt = []
    for _ in range(nfmt):
        ctype, i = _uleb(u, i)
        form, i = _uleb(u, i)
        fmt.append((ctype, form))
    count, i = _uleb(u, i)
    out = []
    for _ in range(count):
        entry = {}
        for ctype, form in fmt:
            kind = _FORMS.get(form)
            if kind == "string":
                stop = u.index(b"\0", i)
                value = u[i:stop].decode("utf-8", "replace")
                i = stop + 1
            elif kind in ("strp", "line_strp"):
                off, = struct.unpack_from(end + "I", u, i)
                i += 4
                value = _cstr(line_str if kind == "line_strp" else debug_str, off)
            elif kind == "udata":
                value, i = _uleb(u, i)
            elif kind in ("data1", "data2", "data4", "data8", "data16"):
                n = {"data1": 1, "data2": 2, "data4": 4, "data8": 8, "data16": 16}[kind]
                value = int.from_bytes(u[i:i + n], "little" if end == "<" else "big")
                i += n
            elif kind == "block":
                n, i = _uleb(u, i)
                value = u[i:i + n]
                i += n
            else:
                raise ValueError(f"DW_FORM 0x{form:x} in a line-table header")
            entry[ctype] = value
        out.append(entry)
    return out, i
