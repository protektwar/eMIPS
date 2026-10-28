"""MIPS executables (PLAN M13; ELF64 for MIPS64: M22 phase 4b): the
loadable segments, the entry, the symbols, the release the code was built
for, and — from the DWARF line table — the source line of every
instruction.  Qt-free, no dependencies.

References: the *System V ABI* (ELF header, program / section headers,
symbol table), its *MIPS RISC Processor Supplement* (EM_MIPS = 8, the
``e_flags`` bits: EF_MIPS_ARCH 31:28 — 5 MIPS32, 7 MIPS32 Release 2, 9
MIPS32 Release 6 —, EF_MIPS_NAN2008 0x400, EF_MIPS_PIC / CPIC 2 / 4,
EF_MIPS_ARCH_ASE_MICROMIPS 0x02000000, EF_MIPS_ARCH_ASE_M16 0x04000000;
as binutils' ``include/elf/mips.h``), and the *DWARF Debugging
Information Format* versions 2-5, section 6.2 "Line Number Information"
(the line number program and its state machine; version 5's directory /
file tables with their content descriptions).

What eMIPS loads: an executable (``ET_EXEC``) of either byte order —
ELFCLASS32 (o32: MIPS32) or ELFCLASS64 (the n64 ABI: MIPS64; EF_MIPS_ARCH
64 / 64R2 / 64R6, and MIPS III / IV / V code as MIPS64 Release 1, its
superset; the 64-bit headers of the ABI's ELF-64 Object File Format) —;
every ``PT_LOAD`` segment — ``p_filesz`` bytes from the file, the rest of
``p_memsz`` zeros (``.bss``) — at its virtual address (the loader puts
each byte where the MMU maps it, as for assembled programs); ``e_entry``.
A 64-bit address in the 32-bit compatibility space is kept as its 32-bit
form (``0xFFFF_FFFF_8000_0180`` -> ``0x8000_0180``), as the MIPS64
assembler (``-msym32``) keeps program addresses.  Relocatable objects
(``ET_REL``: link them first), shared objects and n32 executables (an
ELF32 with 64-bit code) are refused with the reason; MIPS16e and
microMIPS code is listed and run in its ISA mode (``Symbol.isa``,
:meth:`ElfImage.isa_ranges`).  DWARF 2-5, 32- or 64-bit, with 4- or
8-byte addresses.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

EM_MIPS = 8
#: RISC-V (the RISC-V ELF psABI 1.0, ch. 8): e_flags RVC, the float ABI
#: (soft / single / double / quad), RVE, TSO; .riscv.attributes (8.6:
#: Tag_RISCV_arch = the ISA string the code was built for)
EM_RISCV = 243
EF_RISCV_RVC, EF_RISCV_FLOAT_ABI, EF_RISCV_RVE, EF_RISCV_TSO = 0x1, 0x6, 0x8, 0x10
RISCV_FLOAT_ABIS = {0: "soft", 2: "single", 4: "double", 6: "quad"}
SHT_RISCV_ATTRIBUTES = 0x70000003
TAG_RISCV_ARCH = 5
#: nanoMIPS (the nanoMIPS ELF ABI supplement DN00179 5.1-5.3): e_flags
#: EF_NANOMIPS_PIC / PID / ABI (15:12, P32 = 1) / ARCH (31:28, 32R6 = 0),
#: the .nanoMIPS.abiflags section and its ASE bits
EM_NANOMIPS = 249
EF_NANOMIPS_PIC, EF_NANOMIPS_PID = 0x2, 0x8
EF_NANOMIPS_ABI, E_NANOMIPS_ABI_P32 = 0xF000, 0x1000
EF_NANOMIPS_ARCH = 0xF0000000
SHT_NANOMIPS_ABIFLAGS = 0x70000000
NANOMIPS_ASE_CRC, NANOMIPS_ASE_GINV, NANOMIPS_ASE_XNMS = 0x8000, 0x20000, 0x40000
NANOMIPS_ASE_DSPR3, NANOMIPS_ASE_MT = 0x2000, 0x40
#: ... the ASEs eMIPS's nanoMIPS core has no model of
NANOMIPS_ASES_MISSING = {0x8: "the MCU ASE", 0x100: "the Virtualization ASE", 0x200: "MSA"}
ET_EXEC = 2
PT_LOAD = 1
PT_TLS = 7
SHT_SYMTAB = 2
#: the MIPS ABI flags section (Elf_MIPS_ABIFlags_v0) and its DSP ASE bits
SHT_MIPS_ABIFLAGS = 0x7000002A
AFL_ASE_DSP, AFL_ASE_DSPR2, AFL_ASE_DSPR3 = 0x1, 0x2, 0x2000
#: ... MSA (binutils include/elf/mips.h AFL_ASE_MSA) and the FP ABIs
#: (Val_GNU_MIPS_ABI_FP_*: the ABI flags' fp_abi byte) whose o32 code
#: needs Status.FR = 0 — MSA's vector registers need FR = 1 (MD00866 3.1);
#: n64's "double" is 64-bit FPRs already
AFL_ASE_MSA = 0x200
FP_ABI_FR0 = {1: "double precision, 32-bit FPRs (-mfp32)", 2: "single precision",
              4: "the old 64-bit ABI (-mfp64 on o32 before 'FPXX')"}
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
#: ... of a 64-bit (ELFCLASS64) executable: MIPS III / IV / V and MIPS64 run
#: on MIPS64 Release 1 (MIPS64 includes them), 64R2 on R2-R5, 64R6 on R6
ARCH_RELEASE64 = {0x20000000: "R1", 0x30000000: "R1", 0x40000000: "R1",
                  0x60000000: "R1", 0x80000000: "R5", 0xA0000000: "R6"}
ARCH_NAMES = {0x50000000: "MIPS32", 0x70000000: "MIPS32 Release 2",
              0x90000000: "MIPS32 Release 6", 0x60000000: "MIPS64",
              0x80000000: "MIPS64 Release 2", 0xA0000000: "MIPS64 Release 6",
              0x20000000: "MIPS III", 0x30000000: "MIPS IV", 0x40000000: "MIPS V"}
_ABOVE_4G = ("code or data above the 32-bit compatibility space: it needs 64-bit "
             "addressing (Config.AT = 2: a TLB MMU, --mmu tlb)")
#: e_flags EF_MIPS_ABI2: the n32 ABI (64-bit registers, 32-bit pointers)
EF_MIPS_ABI2 = 0x20
M32, M64 = 0xFFFFFFFF, (1 << 64) - 1


def va32(addr: int) -> int:
    """A 64-bit address in the 32-bit compatibility space as its 32-bit
    form (what the MIPS64 assembler's -msym32 programs use); any other
    address as it is."""
    addr &= M64
    return addr & M32 if addr < 0x80000000 or addr >= 0xFFFFFFFF80000000 else addr


class ElfError(ValueError):
    """Not an ELF eMIPS can run (the message says why)."""


@dataclass
class Symbol:
    name: str
    value: int
    size: int
    kind: str                       # "func", "object" or "other"
    #: the code's ISA (st_other: STO_MIPS16 0xF0, STO_MICROMIPS 0x80;
    #: "nanomips" in an EM_NANOMIPS file)
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
    #: MSA code (.MIPS.abiflags ases: AFL_ASE_MSA): the core gets MSA and
    #: Status.FR = 1 (the FP ABI must allow it: FPXX, FP64, FP64A)
    msa: bool = False
    #: e_flags EF_MIPS_NAN2008: the code's NaNs are IEEE 754-2008's
    nan2008: bool = False
    #: "mips32" (ELFCLASS32, o32), "mips64" (ELFCLASS64, n64) or "nanomips"
    #: (EM_NANOMIPS, p32)
    arch: str = "mips32"
    #: nanoMIPS: what the code needs of the core (.nanoMIPS.abiflags: "crc",
    #: "ginvt", "xnms" = the full ISA, not the nanoMIPS subset) — the names
    #: of emips.assembler.Program.needs
    needs: frozenset = frozenset()
    #: RISC-V: PT_TLS (address, p_filesz, p_memsz): the TLS template
    tls: tuple | None = None
    #: RISC-V ("rv32" / "rv64", EM_RISCV): the extensions the code was built
    #: for (Tag_RISCV_arch as eMIPS's extension string: "imafdc_zicsr ...")
    rv_extensions: str = ""

    @property
    def rv(self) -> bool:
        return self.arch in ("rv32", "rv64")

    @property
    def release(self) -> str:
        if self.arch == "nanomips":             # (nanoMIPS is Release 6 based)
            return "R6"
        if self.rv:
            return "R5"                         # (no MIPS release: the session's default)
        table = ARCH_RELEASE64 if self.arch == "mips64" else ARCH_RELEASE
        return table.get(self.flags & EF_MIPS_ARCH, "R5")

    @property
    def arch_name(self) -> str:
        if self.arch == "nanomips":
            return "nanoMIPS32 Release 6"
        if self.rv:
            return f"{self.arch.upper()} {self.rv_extensions}, " \
                f"{RISCV_FLOAT_ABIS[self.flags & EF_RISCV_FLOAT_ABI]}-float ABI"
        return ARCH_NAMES.get(self.flags & EF_MIPS_ARCH,
                              f"EF_MIPS_ARCH 0x{(self.flags >> 28) & 0xF:x}")

    @property
    def mips16(self) -> bool:
        return self.arch in ("mips32", "mips64") and bool(self.flags & EF_MIPS_M16)

    @property
    def micromips(self) -> bool:
        return self.arch in ("mips32", "mips64") and bool(self.flags & EF_MIPS_MICROMIPS)

    def isa_ranges(self) -> list[tuple[int, int, str]]:
        """``(start, end, isa)`` of the code: each executable range split
        at its symbols, a region taking the ISA of the symbol it starts at
        (what objdump does: STO_MIPS16 / STO_MICROMIPS mark 16-bit code);
        nanoMIPS code is all "nanomips"."""
        if self.arch == "nanomips" or self.rv:
            return [(a, b, "riscv" if self.rv else "nanomips") for a, b in self.code_ranges()]
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
        one with MIPS16e / microMIPS / nanoMIPS code)."""
        out: dict[int, tuple[str, int]] = {}
        rows = self.lines
        step = 2 if self.mips16 or self.micromips or self.arch == "nanomips" or (
            self.rv and self.flags & EF_RISCV_RVC) else 4
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
    """Parse the executable ``data`` (see the module docstring); raises
    :class:`ElfError` with the reason (a cut or damaged file too)."""
    try:
        return _parse(data, path)
    except (struct.error, IndexError, ValueError) as exc:
        if isinstance(exc, ElfError):
            raise
        raise ElfError(f"a damaged or cut ELF file ({exc})") from None


def _parse(data: bytes, path: str) -> ElfImage:
    if data[:4] != b"\x7fELF":
        raise ElfError("not an ELF file")
    if data[4] not in (1, 2):
        raise ElfError("unknown ELF class (EI_CLASS)")
    wide = data[4] == 2                              # ELFCLASS64: the n64 ABI
    if data[5] not in (1, 2):
        raise ElfError("unknown byte order (EI_DATA)")
    end = "<" if data[5] == 1 else ">"
    endian = "little" if data[5] == 1 else "big"
    (e_type, e_machine, _ver, e_entry, e_phoff, e_shoff, e_flags, _ehsize,
     e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx) = struct.unpack_from(
        end + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH"), data, 16)
    nano = e_machine == EM_NANOMIPS
    if e_machine not in (EM_MIPS, EM_NANOMIPS, EM_RISCV):
        raise ElfError(f"not a MIPS or RISC-V executable (e_machine {e_machine})")
    if e_type != ET_EXEC:
        kind = {1: "a relocatable object (.o): link it first",
                3: "a shared object / position-independent executable"}.get(
            e_type, f"ELF type {e_type}")
        raise ElfError(f"{kind} — eMIPS loads executables (ET_EXEC)")
    if nano:
        return _parse_nano(data, path, end, endian, wide, e_entry, e_phoff, e_shoff, e_flags,
                           e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx)
    if e_machine == EM_RISCV:
        return _parse_rv(data, path, end, endian, wide, e_entry, e_phoff, e_shoff, e_flags,
                         e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx)
    arch = e_flags & EF_MIPS_ARCH
    if not wide and (arch in ARCH_RELEASE64 or e_flags & EF_MIPS_ABI2):
        raise ElfError(f"{ARCH_NAMES.get(arch, '64-bit')} code in a 32-bit ELF (the n32 "
                       "ABI: 64-bit registers, 32-bit pointers): eMIPS runs o32 (ELF32, "
                       "MIPS32) and n64 (ELF64, MIPS64) executables")
    if wide and arch not in ARCH_RELEASE64:
        raise ElfError(f"{ARCH_NAMES.get(arch, f'EF_MIPS_ARCH 0x{arch >> 28:x}')} code in a "
                       "64-bit ELF: eMIPS runs MIPS64 (and MIPS III / IV / V) code there")
    img = ElfImage(path, endian, va32(e_entry) if wide else e_entry, e_flags,
                   arch="mips64" if wide else "mips32")
    img.nan2008 = bool(e_flags & EF_MIPS_NAN2008)
    if e_flags & EF_MIPS_M16:
        img.notes.append("MIPS16e code (EF_MIPS_ARCH_ASE_M16): the core gets MIPS16e")
    if e_flags & EF_MIPS_MICROMIPS:
        img.notes.append("microMIPS code (EF_MIPS_ARCH_ASE_MICROMIPS): the core gets "
                         "microMIPS")
    if e_flags & (EF_MIPS_PIC | EF_MIPS_CPIC):
        img.notes.append("position-independent (abicalls) code: it expects $gp / $t9 "
                         "set up by a dynamic loader — build with -mno-abicalls -fno-pic")
    for i in range(e_phnum):
        if wide:                                     # Elf64_Phdr
            (p_type, p_flags, p_offset, p_vaddr, _paddr, p_filesz, p_memsz,
             _align) = struct.unpack_from(end + "IIQQQQQQ", data, e_phoff + i * e_phentsize)
            if p_type == PT_LOAD and p_memsz and 0x80000000 <= p_vaddr <= M32:
                raise ElfError(f"a segment at 0x{p_vaddr:016x}: eMIPS keeps 32-bit "
                               "addresses for the compatibility space (0x80000000 is "
                               "kseg0, 0xFFFFFFFF80000000): xuseg 0x0000000080000000-"
                               "0x00000000FFFFFFFF cannot be loaded — link below "
                               "0x80000000 or above 4 GB")
            p_vaddr = va32(p_vaddr)
        else:
            (p_type, p_offset, p_vaddr, _paddr, p_filesz, p_memsz, p_flags,
             _align) = struct.unpack_from(end + "IIIIIIII", data, e_phoff + i * e_phentsize)
        if p_type != PT_LOAD or not p_memsz:
            continue
        if wide and p_vaddr > M32 and not img.notes.count(_ABOVE_4G):
            img.notes.append(_ABOVE_4G)
        chunk = data[p_offset:p_offset + p_filesz]
        amask = M64 if p_vaddr > M32 else M32
        for k in range(p_memsz):
            img.image[(p_vaddr + k) & amask] = chunk[k] if k < len(chunk) else 0
        rwx = "".join(c for c, bit in (("R", 4), ("W", 2), ("X", 1)) if p_flags & bit)
        img.segments.append((p_vaddr, p_memsz, rwx))
    if not img.segments:
        raise ElfError("no loadable segment (PT_LOAD)")
    sections = _sections(data, end, e_shoff, e_shentsize, e_shnum, e_shstrndx, wide)
    _read_sections(img, data, end, sections, wide)
    flags = next((s for s in sections if s[1] == SHT_MIPS_ABIFLAGS and s[5] >= 24), None)
    if flags is not None:                          # Elf_MIPS_ABIFlags_v0.ases
        ases = struct.unpack_from(end + "I", data, flags[4] + 12)[0]
        img.dsp = 2 if ases & (AFL_ASE_DSPR2 | AFL_ASE_DSPR3) else 1 if ases & AFL_ASE_DSP else 0
        if img.dsp:
            img.notes.append("DSP Module code (.MIPS.abiflags: "
                             + ("Revision 2" if img.dsp > 1 else "Revision 1")
                             + "): the core gets the DSP Module")
        img.msa = bool(ases & AFL_ASE_MSA)
        fp_abi = data[flags[4] + 7]
        if img.msa and not wide and fp_abi in FP_ABI_FR0:   # (n64: "double" = FR 1)
            raise ElfError(f"MSA code built for the FP ABI {FP_ABI_FR0[fp_abi]}: MSA's "
                           "vector registers need Status.FR = 1 (MD00866 3.1) — build "
                           "with -mfp64 (emips cc --msa does)")
        if img.msa:
            img.notes.append("MSA code (.MIPS.abiflags): the core gets MSA, Status.FR = 1")
    _read_lines(img, data, end, sections, wide)
    return img


def _parse_nano(data: bytes, path: str, end: str, endian: str, wide: bool, e_entry: int,
                e_phoff: int, e_shoff: int, e_flags: int, e_phentsize: int, e_phnum: int,
                e_shentsize: int, e_shnum: int, e_shstrndx: int) -> ElfImage:
    """A nanoMIPS executable (DN00179 ch. 5): ELFCLASS32, the p32 ABI,
    nanoMIPS32 Release 6; what .nanoMIPS.abiflags says the code needs."""
    if wide or e_flags & EF_NANOMIPS_ARCH:
        raise ElfError("a nanoMIPS64 executable (E_NANOMIPS_ARCH_64R6, the p64 ABI): eMIPS "
                       "runs nanoMIPS32 (p32)")
    img = ElfImage(path, endian, e_entry, e_flags, arch="nanomips")
    if e_flags & EF_NANOMIPS_ABI not in (0, E_NANOMIPS_ABI_P32):
        raise ElfError(f"nanoMIPS ABI {(e_flags & EF_NANOMIPS_ABI) >> 12} (EF_NANOMIPS_ABI): "
                       "eMIPS runs the p32 ABI's executables")
    if e_flags & (EF_NANOMIPS_PIC | EF_NANOMIPS_PID):
        img.notes.append("position-independent code / data (EF_NANOMIPS_PIC / PID): it "
                         "expects a loader to set $gp and move it — build without -fpic / "
                         "-mpid")
    for i in range(e_phnum):
        (p_type, p_offset, p_vaddr, _paddr, p_filesz, p_memsz, p_flags,
         _align) = struct.unpack_from(end + "IIIIIIII", data, e_phoff + i * e_phentsize)
        if p_type != PT_LOAD or not p_memsz:
            continue
        chunk = data[p_offset:p_offset + p_filesz]
        for k in range(p_memsz):
            img.image[(p_vaddr + k) & M32] = chunk[k] if k < len(chunk) else 0
        rwx = "".join(c for c, bit in (("R", 4), ("W", 2), ("X", 1)) if p_flags & bit)
        img.segments.append((p_vaddr, p_memsz, rwx))
    if not img.segments:
        raise ElfError("no loadable segment (PT_LOAD)")
    sections = _sections(data, end, e_shoff, e_shentsize, e_shnum, e_shstrndx)
    _read_sections(img, data, end, sections)
    flags = next((s for s in sections if s[1] == SHT_NANOMIPS_ABIFLAGS and s[5] >= 24), None)
    if flags is not None:                         # the ABIFlags structure's ases (5.2)
        ases = struct.unpack_from(end + "I", data, flags[4] + 12)[0]
        needs = {name for bit, name in ((NANOMIPS_ASE_CRC, "crc"), (NANOMIPS_ASE_GINV, "ginvt"),
                                        (NANOMIPS_ASE_XNMS, "xnms"),
                                        (NANOMIPS_ASE_DSPR3, "dsp"),
                                        (NANOMIPS_ASE_MT, "mt")) if ases & bit}
        img.needs = frozenset(needs)
        missing = [what for bit, what in NANOMIPS_ASES_MISSING.items() if ases & bit]
        if missing:
            raise ElfError(f"the code uses {' and '.join(missing)} (.nanoMIPS.abiflags): "
                           "eMIPS's nanoMIPS core has no model of "
                           + ("it" if len(missing) == 1 else "them"))
    _read_lines(img, data, end, sections)
    return img


def _parse_rv(data: bytes, path: str, end: str, endian: str, wide: bool, e_entry: int,
              e_phoff: int, e_shoff: int, e_flags: int, e_phentsize: int, e_phnum: int,
              e_shentsize: int, e_shnum: int, e_shstrndx: int) -> ElfImage:
    """A RISC-V executable (the psABI ch. 8): RV32 (ELFCLASS32) or RV64
    (ELFCLASS64), little-endian; the extensions from .riscv.attributes."""
    if endian != "little":
        raise ElfError("a big-endian RISC-V executable: eMIPS's RISC-V harts are "
                       "little-endian (mstatus.MBE = 0)")
    if e_flags & EF_RISCV_RVE:
        raise ElfError("an RV32E / RV64E executable (EF_RISCV_RVE: 16 registers): eMIPS's "
                       "harts are RV32I / RV64I")
    img = ElfImage(path, endian, e_entry, e_flags, arch="rv64" if wide else "rv32")
    for i in range(e_phnum):
        if wide:
            (p_type, p_flags, p_offset, p_vaddr, _paddr, p_filesz, p_memsz,
             _align) = struct.unpack_from(end + "IIQQQQQQ", data, e_phoff + i * e_phentsize)
        else:
            (p_type, p_offset, p_vaddr, _paddr, p_filesz, p_memsz, p_flags,
             _align) = struct.unpack_from(end + "IIIIIIII", data, e_phoff + i * e_phentsize)
        if p_type == PT_TLS and p_memsz:        # (the TLS template: thread 0's tp)
            img.tls = (p_vaddr, p_filesz, p_memsz)
        if p_type != PT_LOAD or not p_memsz:
            continue
        chunk = data[p_offset:p_offset + p_filesz]
        for k in range(p_memsz):
            img.image[p_vaddr + k] = chunk[k] if k < len(chunk) else 0
        rwx = "".join(c for c, bit in (("R", 4), ("W", 2), ("X", 1)) if p_flags & bit)
        img.segments.append((p_vaddr, p_memsz, rwx))
    if not img.segments:
        raise ElfError("no loadable segment (PT_LOAD)")
    if img.tls is not None:                     # (.tbss has no room of its own in ld's
        base, filesz, memsz = img.tls           # layout: eMIPS uses the block in place)
        if any(base + k in img.image for k in range(filesz, memsz)):
            img.notes.append("the TLS block's .tbss overlaps what the linker put after "
                             ".tdata: tp points at the image, so those variables share "
                             "memory (eMIPS's emips_rv.ld reserves the room)")
    sections = _sections(data, end, e_shoff, e_shentsize, e_shnum, e_shstrndx, wide)
    _read_sections(img, data, end, sections, wide)
    attrs = next((s for s in sections if s[1] == SHT_RISCV_ATTRIBUTES), None)
    arch = _riscv_arch(data[attrs[4]:attrs[4] + attrs[5]]) if attrs is not None else None
    if arch is None:
        img.rv_extensions = "imac" if e_flags & EF_RISCV_RVC else "ima"
        img.notes.append("no .riscv.attributes (Tag_RISCV_arch): the extensions are guessed "
                         "from e_flags")
    else:
        img.rv_extensions = arch
    _read_lines(img, data, end, sections, wide)
    return img


def _riscv_arch(attrs: bytes) -> str | None:
    """Tag_RISCV_arch of a .riscv.attributes section (the psABI 8.6: 'A',
    then <length, "riscv", Tag_File, length, attributes>), as eMIPS's
    extension string (versions dropped: "rv64i2p1_m2p0_zicsr2p0" ->
    "im_zicsr"); None when it has none."""
    import re
    if not attrs or attrs[0] != ord("A"):
        return None
    i = 1
    while i + 4 <= len(attrs):
        length = struct.unpack_from("<I", attrs, i)[0]
        end_ = i + length
        name_end = attrs.index(b"\0", i + 4)
        vendor = attrs[i + 4:name_end].decode("ascii", "replace")
        j = name_end + 1
        while vendor == "riscv" and j < end_:
            tag, j = _uleb(attrs, j)
            size = struct.unpack_from("<I", attrs, j)[0]
            sub_end, j = j - 1 + size, j + 4
            if tag != 1:                        # (Tag_File only)
                j = sub_end
                continue
            while j < sub_end:
                tag, j = _uleb(attrs, j)
                if tag == TAG_RISCV_ARCH or (tag > 32 and tag & 1):     # NTBS
                    k = attrs.index(b"\0", j)
                    text = attrs[j:k].decode("ascii", "replace")
                    j = k + 1
                    if tag == TAG_RISCV_ARCH:
                        body = re.sub(r"^rv(32|64)", "", text.lower())
                        parts = [re.sub(r"\d+p\d+$", "", p) for p in body.split("_")]
                        letters = "".join(re.sub(r"\d+p\d+", "", p) for p in parts
                                          if len(re.sub(r"\d+p\d+", "", p)) == 1
                                          or p == parts[0])
                        multi = [p for p in parts[1:] if len(p) > 1]
                        return "_".join([letters] + multi)
                else:
                    _v, j = _uleb(attrs, j)
        i = end_
    return None


def _read_sections(img: ElfImage, data: bytes, end: str, sections: list,
                   wide: bool = False) -> None:
    """The allocated sections and the symbols."""
    fold = wide and not img.rv                         # (MIPS64: -msym32's addresses)
    for name, sh_type, flags, addr, off, size, link, _info, entsize in sections:
        if flags & 2 and size:                         # SHF_ALLOC (SHF_EXECINSTR 4)
            img.sections.append((name, va32(addr) if fold else addr, size, bool(flags & 4)))
        if sh_type == SHT_SYMTAB:
            img.symbols = _symbols(data, end, off, size, entsize, sections[link], wide,
                                   img.arch == "nanomips", rv=img.rv)


def _read_lines(img: ElfImage, data: bytes, end: str, sections: list,
                wide: bool = False) -> None:
    """The DWARF line table (or a note that there is none)."""
    by_name = {s[0]: s for s in sections}
    if ".debug_line" in by_name:
        s = by_name[".debug_line"]
        line_str = by_name.get(".debug_line_str")
        dstr = by_name.get(".debug_str")
        try:
            img.lines = read_line_table(
                data[s[4]:s[4] + s[5]], end,
                data[line_str[4]:line_str[4] + line_str[5]] if line_str else b"",
                data[dstr[4]:dstr[4] + dstr[5]] if dstr else b"")
            if wide and not img.rv:
                for row in img.lines:
                    row.address = va32(row.address)
        except (struct.error, IndexError, ValueError) as exc:
            img.notes.append(f"the DWARF line table could not be read ({exc})")
    else:
        img.notes.append("no DWARF line table (build with -g to see the source lines)")


def _sections(data: bytes, end: str, off: int, entsize: int, n: int, strndx: int,
              wide: bool = False) -> list:
    fmt = end + ("IIQQQQIIQQ" if wide else "IIIIIIIIII")    # Elf64_Shdr / Elf32_Shdr
    raw = [struct.unpack_from(fmt, data, off + i * entsize) for i in range(n)]
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


def _symbols(data: bytes, end: str, off: int, size: int, entsize: int, strtab,
             wide: bool = False, nano: bool = False, rv: bool = False) -> list[Symbol]:
    out = []
    step = entsize or (24 if wide else 16)
    for k in range(size // step):
        if wide:                                     # Elf64_Sym
            name, info, other, shndx, value, sz = struct.unpack_from(
                end + "IBBHQQ", data, off + k * step)
            if not rv:
                value = va32(value)
        else:
            name, value, sz, info, other, shndx = struct.unpack_from(
                end + "IIIBBH", data, off + k * step)
        kind = {STT_FUNC: "func", STT_OBJECT: "object"}.get(info & 0xF, "other")
        if not name or shndx == 0 or (info & 0xF) in (3, 4):   # SECTION, FILE
            continue
        isa = "nanomips" if nano else "riscv" if rv else {
            STO_MIPS16: "mips16e", STO_MICROMIPS: "micromips"}.get(other & 0xF0, "mips32")
        out.append(Symbol(_cstr(data, strtab[4] + name), value & ~1 if isa in (
            "mips16e", "micromips") else value, sz, kind, isa))
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
    6.2; versions 2-5, 32- or 64-bit DWARF: 7.4)."""
    rows: list[LineRow] = []
    i = 0
    while i + 4 <= len(sec):
        unit_len, = struct.unpack_from(end + "I", sec, i)
        off = 4                                      # 32-bit DWARF: 4-byte offsets
        i += 4
        if unit_len == 0xFFFFFFFF:                   # 64-bit DWARF (7.4)
            unit_len, = struct.unpack_from(end + "Q", sec, i)
            off = 8
            i += 8
        start, unit_end = i, i + unit_len
        rows += _line_program(sec[start:unit_end], end, line_str, debug_str, off)
        i = unit_end
    return rows


def _line_program(u: bytes, end: str, line_str: bytes, debug_str: bytes,
                  off: int = 4) -> list[LineRow]:
    version, = struct.unpack_from(end + "H", u, 0)
    i = 2
    if version >= 5:
        i += 2                                       # address_size, seg_sel_size
    header_len, = struct.unpack_from(end + ("Q" if off == 8 else "I"), u, i)
    i += off
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
        dirs, i = _entries(u, i, end, line_str, debug_str, off)
        files, i = _entries(u, i, end, line_str, debug_str, off)
        dirnames = [d.get(1, "") for d in dirs]
        filenames = [_join(dirnames, f.get(1, "?"), f.get(2, 0)) for f in files]
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
    i = prog
    rows: list[LineRow] = []

    def reset():                                     # (file = 1 in every version: DWARF
        #                                              5 6.2.2 Table 6.4 — entry 1 of a v5 table)
        return {"address": 0, "file": 1, "line": 1, "is_stmt": bool(default_stmt)}

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
            elif sub == 2:                           # set_address (4 or 8 bytes)
                st["address"], = struct.unpack_from(end + ("Q" if len(body) >= 8 else "I"),
                                                    body, 0)
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
             debug_str: bytes, size: int = 4) -> tuple[list[dict], int]:
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
                off, = struct.unpack_from(end + ("Q" if size == 8 else "I"), u, i)
                i += size                            # an offset: 4 or 8 bytes (DWARF 7.4)
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
