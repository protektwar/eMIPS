"""An ELF executable of an assembled program (PLAN M16): what GDB reads
for an assembly program — its byte order and architecture, the segments,
the labels as symbols and, in a DWARF line table, the source lines.

Reference: the System V ABI (ELF32: the header, program headers, section
headers, the symbol table), the MIPS supplement (EM_MIPS 8, EF_MIPS_ARCH),
DWARF 2 6.2 (the line number program); checked by reading it back with
:mod:`emips.elf` and binutils' ``readelf``.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .elf import EM_MIPS, ET_EXEC

#: EF_MIPS_ARCH by release (32 / 32R2 / 32R6) and EF_MIPS_NOREORDER-free o32
ARCH_FLAGS = {"R1": 0x50000000, "R6": 0x90000000}
ARCH_R2 = 0x70000000
SECTIONS = (("text", ".text", True), ("ktext", ".ktext", True),
            ("data", ".data", False), ("kdata", ".kdata", False))
SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB = 1, 2, 3
SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR = 1, 2, 4
PT_LOAD, PF_X, PF_W, PF_R = 1, 1, 2, 4


class _Strings:
    def __init__(self) -> None:
        self.data = bytearray(b"\0")
        self.at: dict[str, int] = {}

    def add(self, s: str) -> int:
        if s not in self.at:
            self.at[s] = len(self.data)
            self.data += s.encode() + b"\0"
        return self.at[s]


def _uleb(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _sleb(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        done = (n == 0 and not b & 0x40) or (n == -1 and b & 0x40)
        out.append(b | (0 if done else 0x80))
        if done:
            return bytes(out)


def line_program(lines: dict[int, int], source: Path, order: str) -> bytes:
    """A DWARF 2 .debug_line unit: one sequence per run of consecutive
    instructions, a row per instruction (address -> line)."""
    end = "<" if order == "little" else ">"
    std_lengths = bytes([0, 1, 1, 1, 1, 0, 0, 0, 1])   # opcodes 1-9 (opcode_base 10)
    header = (struct.pack(f"{end}B", 4)                # minimum_instruction_length
              + struct.pack(f"{end}B", 1)              # default_is_stmt
              + struct.pack(f"{end}b", -5)             # line_base
              + struct.pack(f"{end}B", 14)             # line_range
              + struct.pack(f"{end}B", 10)             # opcode_base
              + std_lengths
              + str(source.parent).encode() + b"\0" + b"\0"     # include_directories
              + source.name.encode() + b"\0" + _uleb(1) + _uleb(0) + _uleb(0) + b"\0")
    prog = bytearray()
    addrs = sorted(lines)
    runs: list[list[int]] = []
    for a in addrs:
        if runs and runs[-1][-1] + 4 == a:
            runs[-1].append(a)
        else:
            runs.append([a])
    for run in runs:
        prog += b"\0" + _uleb(5) + b"\x02" + struct.pack(f"{end}I", run[0])  # set_address
        line = 1
        for i, a in enumerate(run):
            if i:
                prog += b"\x02" + _uleb(1)       # advance_pc 1 (x 4 bytes)
            delta = lines[a] - line
            if delta:
                prog += b"\x03" + _sleb(delta)   # advance_line
                line = lines[a]
            prog += b"\x01"                      # copy
        prog += b"\x02" + _uleb(1) + b"\0" + _uleb(1) + b"\x01"   # end_sequence
    unit = struct.pack(f"{end}H", 2) + struct.pack(f"{end}I", len(header)) + header + prog
    return struct.pack(f"{end}I", len(unit)) + unit


def debug_info(source: Path, low: int, high: int, order: str) -> tuple[bytes, bytes]:
    """A DWARF 2 compilation unit for the line table (GDB reads lines only
    through one): (.debug_info, .debug_abbrev)."""
    e = "<" if order == "little" else ">"
    abbrev = (_uleb(1) + _uleb(0x11) + b"\0"                    # DW_TAG_compile_unit, no children
              + _uleb(0x25) + _uleb(0x08)                        # producer: string
              + _uleb(0x13) + _uleb(0x05)                        # language: data2
              + _uleb(0x03) + _uleb(0x08)                        # name: string
              + _uleb(0x1B) + _uleb(0x08)                        # comp_dir: string
              + _uleb(0x10) + _uleb(0x06)                        # stmt_list: data4
              + _uleb(0x11) + _uleb(0x01)                        # low_pc: addr
              + _uleb(0x12) + _uleb(0x01)                        # high_pc: addr
              + b"\0\0" + b"\0")
    die = (_uleb(1) + b"eMIPS assembler\0" + struct.pack(f"{e}H", 0x8001)  # DW_LANG_Mips_Assembler
           + source.name.encode() + b"\0" + str(source.parent).encode() + b"\0"
           + struct.pack(f"{e}III", 0, low, high))
    body = struct.pack(f"{e}HIB", 2, 0, 4) + die
    return struct.pack(f"{e}I", len(body)) + body, abbrev


def write_elf(program, *, source: Path | None = None) -> bytes:
    """ELF32 ``ET_EXEC`` bytes of ``program`` (an :class:`emips.assembler.
    Program`); ``source``: the assembly file (for the line table)."""
    order = program.endian
    e = "<" if order == "little" else ">"
    secs = []                                    # (name, addr, data, exec)
    for seg, name, ex in SECTIONS:
        span = program.segments.get(seg)
        if not span or span[1] <= span[0]:
            continue
        lo, hi = span
        data = bytes(program.image.get(a, 0) for a in range(lo, hi))
        secs.append((name, lo, data, ex))
    shstr = _Strings()
    strtab = _Strings()
    names = [s[0] for s in secs] + [".symtab", ".strtab", ".debug_line", ".debug_info",
                                    ".debug_abbrev", ".shstrtab"]
    for n in names:
        shstr.add(n)
    # symbols: labels in a section, constants absolute
    syms = [struct.pack(f"{e}IIIBBH", 0, 0, 0, 0, 0, 0)]
    for name, value in sorted(program.symbols.items(), key=lambda kv: kv[1]):
        if name.startswith(".L"):
            continue
        shndx = next((k + 1 for k, (_, lo, data, _x) in enumerate(secs)
                      if lo <= value < lo + len(data)), 0xFFF1)       # SHN_ABS
        stt = 0                                    # STT_NOTYPE, as GNU as does
        # a label of MIPS16e / microMIPS code: STO_MIPS16 / STO_MICROMIPS (GDB
        # and objdump take the ISA from it; the value stays even, as GNU ld's)
        other = 0xF0 if value in program.m16 else 0x80 if value in program.mm else 0
        syms.append(struct.pack(f"{e}IIIBBH", strtab.add(name), value & 0xFFFFFFFF, 0,
                                (1 << 4) | stt, other, shndx))      # STB_GLOBAL
    symtab = b"".join(syms)
    lines = {a: sl.line for a, sl in program.source_map.items()}
    src = source or Path("program.asm")
    debug_line = line_program(lines, src, order) if lines else b""
    info, abbrev = debug_info(src, min(lines), max(lines) + 4, order) if lines else (b"", b"")
    # layout: header, program headers, section data, section headers
    ehsize, phsize, shsize = 52, 32, 40
    off = ehsize + phsize * len(secs)
    chunks: list[tuple[int, bytes]] = []
    placed = []
    for name, lo, data, ex in secs:
        off += (-off) % 4
        placed.append(off)
        chunks.append((off, data))
        off += len(data)
    extra = []
    for data in (symtab, bytes(strtab.data), debug_line, info, abbrev, bytes(shstr.data)):
        off += (-off) % 4
        extra.append(off)
        chunks.append((off, data))
        off += len(data)
    off += (-off) % 4
    shoff = off
    nsec = 1 + len(secs) + 6
    flags = ARCH_FLAGS.get(program.release, ARCH_R2) | 0x1000   # EF_MIPS_ABI_O32 (ABI2 0)
    if program.m16:
        flags |= 0x04000000                      # EF_MIPS_ARCH_ASE_M16
    if program.mm:
        flags |= 0x02000000                      # EF_MIPS_ARCH_ASE_MICROMIPS
    ident = b"\x7fELF" + bytes([1, 2 if order == "big" else 1, 1, 0]) + bytes(8)
    out = bytearray(ident)
    out += struct.pack(f"{e}HHIIIIIHHHHHH", ET_EXEC, EM_MIPS, 1, program.entry & 0xFFFFFFFF,
                       ehsize, shoff, flags, ehsize, phsize, len(secs), shsize, nsec,
                       nsec - 1)
    for (name, lo, data, ex), o in zip(secs, placed):
        out += struct.pack(f"{e}IIIIIIII", PT_LOAD, o, lo, lo & 0x1FFFFFFF, len(data),
                           len(data), PF_R | (PF_X if ex else PF_W), 4)
    for o, data in chunks:
        out += bytes(o - len(out))
        out += data
    out += bytes(shoff - len(out))
    out += bytes(shsize)                         # the null section
    for (name, lo, data, ex), o in zip(secs, placed):
        out += struct.pack(f"{e}IIIIIIIIII", shstr.add(name), SHT_PROGBITS,
                           SHF_ALLOC | (SHF_EXECINSTR if ex else SHF_WRITE), lo, o,
                           len(data), 0, 0, 4, 0)
    strtab_index = len(secs) + 2
    out += struct.pack(f"{e}IIIIIIIIII", shstr.add(".symtab"), SHT_SYMTAB, 0, 0, extra[0],
                       len(symtab), strtab_index, 1, 4, 16)
    out += struct.pack(f"{e}IIIIIIIIII", shstr.add(".strtab"), SHT_STRTAB, 0, 0, extra[1],
                       len(strtab.data), 0, 0, 1, 0)
    out += struct.pack(f"{e}IIIIIIIIII", shstr.add(".debug_line"), SHT_PROGBITS, 0, 0,
                       extra[2], len(debug_line), 0, 0, 1, 0)
    out += struct.pack(f"{e}IIIIIIIIII", shstr.add(".debug_info"), SHT_PROGBITS, 0, 0,
                       extra[3], len(info), 0, 0, 1, 0)
    out += struct.pack(f"{e}IIIIIIIIII", shstr.add(".debug_abbrev"), SHT_PROGBITS, 0, 0,
                       extra[4], len(abbrev), 0, 0, 1, 0)
    out += struct.pack(f"{e}IIIIIIIIII", shstr.add(".shstrtab"), SHT_STRTAB, 0, 0, extra[5],
                       len(shstr.data), 0, 0, 1, 0)
    return bytes(out)
