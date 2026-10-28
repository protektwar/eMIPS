"""Two-pass MIPS32 assembler (PLAN M4): ``.asm`` source -> memory image.

Syntax is GNU ``as`` / MARS-compatible (see :mod:`emips.syntax` for every
instruction's operands).  The assembler produces a :class:`Program`: the
bytes of each segment, the symbol table, a **source map** (address ->
source line, so the GUI / CLI can highlight the executing line) and an
entry point; it can load itself into a :class:`emips.sim.Machine` or be
written as Intel HEX (and read back by :mod:`emips.hexload`).

Segments and defaults (MARS / SPIM layout):

=========  ============  =============================================
``.text``  0x00400000    user code
``.data``  0x10010000    user data
``.ktext`` 0x80000180    kernel code — the general exception vector
``.kdata`` 0x90000000    kernel data
=========  ============  =============================================

Directives: ``.text/.data/.ktext/.kdata [addr]``, ``.word``, ``.half``,
``.byte``, ``.float``, ``.double``, ``.ascii``, ``.asciiz``, ``.space``,
``.align``, ``.eqv/.equ NAME, value``, ``.set reorder|noreorder|at|noat``;
``.globl/.global/.ent/.end/.type/.size/.extern`` are accepted and ignored.
``.word``/``.half``/``.float``/``.double`` align themselves (as in MARS
and GNU ``as``).

**MIPS64** (an ISA of ``arch="mips64"``, PLAN M22): GNU ``as`` with
``-mabi=64 -msym32`` — the n64 register names (``$a4``-``$a7`` = ``$8``-
``$11``, ``$t0``-``$t3`` = ``$12``-``$15``; ``$t4``-``$t7`` do not
exist), symbols of 32 bits (every eMIPS segment is in the compatibility
space), ``.dword`` (a label's address sign-extended), ``.set mips64*`` /
``.set mips32*`` (the doubleword instructions allowed or not), and the
doubleword macros: ``dli`` (gas's ``load_register``), ``dla`` / ``la``
(``lui`` + ``daddiu``), ``dneg`` / ``dnegu``, the constant / short /
variable forms of ``dadd`` / ``daddu`` / ``dsub`` / ``dsubu`` / ``dsll``
/ ``dsrl`` / ``dsra`` / ``drotr`` (a shift by 32-63: the ``*32``
instruction), ``dmul`` (Release 6: an instruction), label loads with a
base register (``daddu``).

Expressions: integers (``42``, ``-3``, ``0x2A``, ``0b101``, ``'a'``),
symbols, ``+ - * / << >> & | ( )``, ``%hi(x)`` / ``%lo(x)`` (``%hi`` is
carry-adjusted for the sign-extended ``%lo``).

Pseudo-instructions, expanded exactly as GNU ``as`` does (``$at`` is the
scratch register; using one under ``.set noat`` is an error):
``nop, move, li, la, b, beqz, bnez, blt, bltu, bgt, bgtu, ble, bleu,
bge, bgeu, neg, negu, not``, and loads / stores with a label or a 32-bit
address (``lw $t0, value``).  ``div/divu rs, rt`` is the machine
instruction (MARS / SPIM meaning); GNU ``as`` would expand it into a
divide-by-zero-checking macro.

**Delay slots.**  The machine always executes the instruction after a
branch / jump.  Under ``.set reorder`` (the default, as in GNU ``as``)
the assembler puts a ``nop`` in every delay slot, so code written for
simulators without delay slots (MARS default) still runs correctly; under
``.set noreorder`` the next source instruction *is* the delay slot.
Unlike GNU ``as``, the assembler never moves instructions into the slot.

**Options** (teaching): ``pseudo=False`` is *strict mode* — every
pseudo-instruction above is an error that names the machine instructions
to write instead (``nop``, ``b``, ``bal`` and the R6 ``jr`` stay: they are
encodings of real instructions); ``reorder=False`` starts the source in
``.set noreorder`` (the source can still switch with ``.set``).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Callable

from . import cp0 as C
from .isa import ISA
from .syntax import (
    DEFAULTS,
    KINDS,
    kind_of,
    operand_spec,
    parse_fcc,
    parse_fpr,
    parse_gpr,
    parse_numreg,
)

M32 = 0xFFFFFFFF
AT = 1

SEGMENT_BASES = {
    "text": 0x00400000,
    "data": 0x10010000,
    "ktext": 0x80000180,
    "kdata": 0x90000000,
}
#: Register values set by :meth:`Program.load` with ``boot=True`` (MARS).
BOOT_SP = 0x7FFFEFFC
#: ... on a MIPS64 core: 16-byte aligned, as the n64 ABI wants it (sd / ld
#: of a stack slot; MARS's 0x7FFFEFFC is not even doubleword-aligned)
BOOT_SP64 = 0x7FFFEFF0
BOOT_GP = 0x10008000


class AsmError(Exception):
    """An assembly error at a source line."""

    def __init__(self, line: int, message: str, text: str = "") -> None:
        super().__init__(f"line {line}: {message}" + (f"  [{text.strip()}]" if text else ""))
        self.line = line
        self.message = message
        self.text = text


@dataclass
class SourceLine:
    line: int
    text: str


@dataclass
class Program:
    """Result of :func:`assemble`."""

    endian: str
    #: MIPS32 release the program was assembled for
    release: str = "R5"
    #: address -> byte
    image: dict[int, int] = field(default_factory=dict)
    #: symbol -> address / value
    symbols: dict[str, int] = field(default_factory=dict)
    #: instruction address -> the source line it came from
    source_map: dict[int, SourceLine] = field(default_factory=dict)
    #: segment name -> (lowest, highest + 1) address used
    segments: dict[str, tuple[int, int]] = field(default_factory=dict)
    entry: int = SEGMENT_BASES["text"]
    #: a compiled program (an ELF, emips.elf): instruction address ->
    #: (source file, line) from its DWARF line table
    c_lines: dict[int, tuple[str, int]] = field(default_factory=dict)
    #: ... (file, line) -> the addresses where that line's statements start
    c_stmts: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    #: ... (start, end, name) of its functions (STT_FUNC symbols)
    functions: list[tuple[int, int, str]] = field(default_factory=list)
    #: MIPS16e code (emips.mips16): instruction address -> its size (2 / 4)
    m16: dict[int, int] = field(default_factory=dict)
    #: microMIPS code (emips.micromips): instruction address -> its size
    mm: dict[int, int] = field(default_factory=dict)
    #: the DSP Module level its instructions need (0: none, 1: the DSP
    #: Module, 2: Revision 2 — emips.isa.ISA ``dsp``)
    dsp: int = 0
    #: it uses the MT Module's instructions (emips.mt, MD00378)
    mt: bool = False
    #: it uses MSA's instructions (MD00866; LSA / DLSA before Release 6 too)
    msa: bool = False
    #: the architecture it was made for: "mips32", "mips64" (assembled
    #: for MIPS64, an ELF64) or "nanomips" (emips.nanoasm)
    arch: str = "mips32"
    #: nanoMIPS code (emips.nanoasm): instruction address -> its size (2 / 4 / 6)
    nm: dict[int, int] = field(default_factory=dict)
    #: RISC-V code (emips.rvasm): instruction address -> its size (2 / 4)
    rv: dict[int, int] = field(default_factory=dict)
    #: ... the extensions it was assembled for (the listing's disassembler)
    rv_extensions: str = ""
    #: the assembler's warnings: (line, message)
    warnings: list = field(default_factory=list)
    #: the strings of ``.print`` (GNU as prints them while it assembles)
    printed: list = field(default_factory=list)
    #: RISC-V: the TLS block (address, initialized bytes, size) — the boot
    #: points tp at it (thread 0 runs in the image itself)
    tls: tuple | None = None
    #: RISC-V: the ``.attribute`` tags (tag -> value) for .riscv.attributes
    rv_attributes: dict = field(default_factory=dict)
    #: what the nanoMIPS linker changed (gold's --debug=target lines)
    link_log: list = field(default_factory=list)
    #: nanoMIPS: what its instructions need of the core besides the base
    #: ISA: "crc", "ginvi", "ginvt", "eva", "tlb", "debug" (from
    #: emips.nanomips.Row.need)
    needs: frozenset = frozenset()

    def word(self, addr: int) -> int:
        data = bytes(self.image.get(addr + i, 0) for i in range(4))
        return struct.unpack(">I" if self.endian == "big" else "<I", data)[0]

    def labels(self) -> dict[int, str]:
        """address -> label (first defined), for the disassembler."""
        out: dict[int, str] = {}
        for name, addr in self.symbols.items():
            if not name.startswith(".L"):          # numeric local labels: no names
                out.setdefault(addr, name)
        return out

    def _load_rv(self, machine, boot: bool = True) -> None:
        """A RISC-V program (emips.rvasm): the image at its addresses
        (physical: an M-mode hart, satp Bare) and the PC at the entry.
        ``boot`` plays the firmware (decision 11's "machine" boot): sp and
        gp set (RARS's sp; gp = __global_pointer$, where the linker's
        gp-relative accesses assume it, as GNU's crt0 sets it) and
        tp the TLS block (LF12: the image's .tdata / .tbss are thread 0's;
        the RISC-V TLS ABI's variant I puts tp at the block's start), and
        mstatus.FS = Initial (the F / D registers usable, as a MIPS boot
        sets CU1) and, with a vector unit, mstatus.VS = Initial (LF6: the
        vector registers usable, 30.3.2) and, with Smrnmi, mnstatus.NMIE = 1
        (LF8: interrupts possible, priv. 8.3 leaves it to the boot code);
        without it the hart stays in its reset state (priv. 3.4: FS / VS
        Off, sp / gp 0, NMIE 0) with the program at the reset vector."""
        if not getattr(machine, "rv", False) or machine.arch != self.arch:
            raise ValueError(f"a {self.arch} program needs a {self.arch} core")
        for a, b in self.image.items():
            machine.phys_write_byte(a, b)
        machine.pc = self.entry & machine.amask
        if not boot:
            return
        machine.write_reg(2, 0x7FFFEFFC if self.arch == "rv32" else 0x7FFFEFF0)
        machine.write_reg(3, self.symbols.get("__global_pointer$", BOOT_GP))
        if self.tls is not None:                # (LF12: thread 0's tp = the TLS block)
            machine.write_reg(4, self.tls[0])
        if machine.flen:
            from .rvcsr import FS_SHIFT
            machine.csrs.mstatus |= 1 << FS_SHIFT          # Initial
        if machine.vector is not None:
            from .rvcsr import VS_SHIFT
            machine.csrs.mstatus |= 1 << VS_SHIFT          # Initial
        if machine.csrs.smrnmi:
            from .rvcsr import NMIE
            machine.csrs.mnstatus |= NMIE

    def load(self, machine, *, boot: bool = True) -> None:
        """Copy the image into ``machine`` and point its PC at the entry.

        ``boot=True`` plays the boot firmware: ``Status`` = CU1 only (kernel
        mode, FPU usable, BEV = ERL = 0 so exceptions go to ``EBase +
        0x180`` = 0x80000180, where ``.ktext`` code lives), ``Config.K0 =
        KU = 3`` (kseg0 / kuseg cacheable), ``K23 = 2`` (kseg2 / kseg3
        uncached: devices), the caches initialized (every line invalid:
        what boot ROM code does with Index Store Tag and ``TagLo = 0``),
        ``$sp`` = 0x7FFFEFFC, ``$gp`` = 0x10008000 (MARS defaults; MIPS64:
        ``$sp`` = 0x7FFFEFF0, 16-byte aligned for n64).  With
        a TLB (:mod:`emips.tlb`) only ``Config.K0`` exists; the firmware
        initializes the TLB (every entry unique and invalid) and, unless
        the program brings its own (code at 0x80000000-0x800000FF),
        installs its TLB refill handler at EBase + 0x000
        (:data:`emips.tlb.FIRMWARE_REFILL`: kuseg at VA + 1 GB, kseg2 /
        kseg3 at PA = VA, like the fixed mapping) — on a MIPS64 core with
        64-bit addresses also the XTLB one at EBase + 0x080
        (:data:`emips.tlb.FIRMWARE_XTLB_REFILL`) and Status.KX / SX / UX;
        MIPS64: Status.FR = 1 (the n64 ABI).  A nanoMIPS program (PLAN M23)
        boots the same way on a nanoMIPS core: no FPU (Status 0), the
        refill handler in nanoMIPS code (``firmware_refill(arch=)``), ``$gp``
        = its ``_gp`` (the start of the small data, as the toolchain's
        linker scripts put it); with the I7200's MPU (:mod:`emips.mpu`) the
        firmware enables the CDMM and, in the MPU build, makes kuseg /
        kseg0's segments cacheable (its Config.K0 / KU / K23 read 0).  The
        image goes where
        that mapping puts it even when the program has its own handler
        (whose page tables then know where the program is).
        ``boot=False`` keeps the hardware reset state (caches undefined,
        TLB not initialised).

        The image is in virtual addresses; every byte is written to the
        physical address the MMU gives it in the state the program starts
        in (with the fixed mapping, kuseg is at VA + 1 GB after boot, at
        VA itself in the reset state, where Status.ERL = 1)."""
        from .exceptions import UnmappedAddress
        if self.arch in ("rv32", "rv64"):
            self._load_rv(machine, boot)
            return
        if machine.big_endian != (self.endian == "big"):
            raise ValueError(f"program is {self.endian}-endian, machine is not")
        if (self.arch == "nanomips") != bool(getattr(machine, "nano", False)):
            raise ValueError("a nanoMIPS program needs a nanoMIPS core (--arch nanomips)"
                             if self.arch == "nanomips" else
                             "a nanoMIPS core runs nanoMIPS programs only")
        # (the I7200's MPU, emips.mpu: Config.MT 3 too, but PA = VA and no
        # Config.K0 / KU / K23 — its segments hold the attributes)
        mpu = getattr(machine.mmu, "segments", False)
        fixed = getattr(machine.mmu, "mt", 3) == 3 and not mpu
        if self.dsp > getattr(machine, "dsp", 0):
            raise ValueError("the program uses DSP Module " + (
                "Revision 2 " if self.dsp > 1 else "") + "instructions: the core needs "
                f"the DSP Module (--dsp {'dspr2' if self.dsp > 1 else 'dsp'})")
        if self.mt and getattr(machine, "mt", None) is None:
            raise ValueError("the program uses MT Module instructions: the core needs "
                             "the MT Module (--mt VPES:TCS)")
        if self.msa and not getattr(machine, "msa", False):
            raise ValueError("the program uses MSA instructions: the core needs MSA "
                             "(--msa)")
        if boot:
            from .tlb import boot_cca, firmware_refill, firmware_xtlb_refill, user_window
            cca = boot_cca(machine)                    # 3; a CPS core: 5, coherent
            # (Release 6: Status.FR is 1 and read-only — 64-bit FPRs only)
            fr = machine.status & C.ST_FR if machine.release == "R6" else 0
            # (no FPU: CU1 reads 0, the 24Kc's MD00343 Table 6.12; the DSP
            # Module is enabled like the FPU: Status.MX, MD00374 3.11)
            machine._cp0_set(C.STATUS, (C.ST_CU1 if machine.has_fpu else 0) | fr
                             | (C.ST_MX if getattr(machine, "dsp", 0) else 0))
            if machine.wide:                           # MIPS64: FR (n64), KX / SX / UX
                machine.boot64()
            if self.msa and getattr(machine, "msa", False):
                # MSA in use: 64-bit FPRs (FR, MD00866 3.3.2) and Config5.MSAEn
                # (3.2) — a reset-state program enables it itself (the lazy
                # switch: MSA Disabled until it does)
                machine._cp0_set(C.STATUS, machine.status | (C.ST_FR if machine.has_fpu else 0))
                machine._cp0_set((16, 5), machine.cp0_read(16, 5) | (1 << 27))
                wrp = getattr(machine, "msa_wrp", None)
                if wrp is not None:                    # partitioning: map $w0 ...
                    for n in range(len(wrp.store)):     # (as many as there are)
                        wrp.write(machine, 6, n)
            config = machine.cp0_read(*C.CONFIG) & ~((7 << 28) | (7 << 25) | 7)
            ku_k23 = (2 << 28) | (cca << 25) if fixed else 0
            machine._cp0_set(C.CONFIG, config | ku_k23 | (0 if mpu else cca))
            if machine.caches is not None:
                machine.caches.initialize()
            if machine.cdmm is not None:               # the CDMM, an MPU's segments
                machine.mmu.boot(machine, handler=False)
            elif not fixed:
                own = any(0x80000000 <= a < 0x80000100 for a in self.image)
                tlb = machine.mmu.mt in (1, 4)              # BAT: no refill handler
                machine.mmu.boot(machine, handler=tlb and not own)
                if tlb and not own:                     # (MIPS64 with 64-bit addresses:
                    text = firmware_refill(cca, user_window(machine),   # + the XTLB
                                           arch=machine.arch)           # Refill handler;
                    if getattr(machine, "at", 1) == 2:                  # nanoMIPS: its
                        text += firmware_xtlb_refill(cca, user_window(machine))   # code)
                    nano = {"nms": machine.nms, "tlb": True} if machine.nano else None
                    fw = assemble(text, endian=self.endian, release=machine.release,
                                  arch=machine.arch, nano=nano,
                                  bases={"ktext": 0x80000000} if nano else None)
                    for addr, byte in fw.image.items():
                        machine.write_byte(addr, byte)
                    machine.firmware = fw
            board = getattr(machine, "board", None)
            if board is not None:                      # the board's firmware (YAMON)
                board.firmware(machine)
                machine.write_reg(29, board.stack_top())
                machine.write_reg(28, self.board_gp())
            else:
                machine.write_reg(29, BOOT_SP64 if machine.wide else BOOT_SP)
                # (nanoMIPS: _gp, the start of the small data, as the
                # toolchain's linker scripts put it)
                machine.write_reg(28, self.symbols.get("_gp", BOOT_GP) if machine.nano
                                  else BOOT_GP)
        for addr, byte in self.image.items():
            try:
                machine.write_byte(addr, byte)
            except UnmappedAddress:
                if boot and machine.mmu.mt in (1, 4):   # the firmware's convention
                    from .tlb import boot_cca, firmware_map, user_window
                    pa = firmware_map(addr, boot_cca(machine), user_window(machine))[0]
                    machine.phys_write_byte(pa, byte)
                    if machine.caches is not None:
                        machine.caches.backdoor_write(pa, byte)
                    continue
                raise UnmappedAddress(
                    f"cannot load the byte at 0x{addr:08x}: the address is mapped "
                    "through the TLB, which holds no translation for it (after a "
                    "reset the TLB is not initialised: put this code in kseg0 / "
                    "kseg1, or start with the boot firmware)",
                    name=None, word=None, addr=addr) from None
        if self.entry & 1:                             # a MIPS16e / microMIPS entry point
            want = "micromips" if self.mm else "mips16e"
            if getattr(machine, "compressed", None) != want:
                isa = "microMIPS" if self.mm else "MIPS16e"
                raise ValueError(f"the program starts in {isa} code: the core needs "
                                 f"{isa} (--compressed {want})")
            machine._jump(machine.to_pc(self.entry))   # (MIPS64: sign-extended)
        else:
            machine.pc = machine.to_pc(self.entry)  # (MIPS64: sign-extended)

    def board_gp(self) -> int:
        """$gp on a board (the descriptor's): the data segment + 0x8000 as
        MIPS's -G small data wants it; nanoMIPS: ``_gp``, the start of the
        small data (the p32 ABI's offsets from $gp are unsigned)."""
        if self.arch == "nanomips" and "_gp" in self.symbols:
            return self.symbols["_gp"]
        return self.segments.get("data", (0x80200000,))[0] + 0x8000

    def size_at(self, addr: int) -> int:
        """The size in bytes of the instruction at ``addr`` (2 / 4 / 6)."""
        return (self.nm.get(addr) or self.m16.get(addr) or self.mm.get(addr)
                or self.rv.get(addr) or 4)

    def encoding(self, addr: int, word: int) -> str:
        """The instruction at ``addr`` (its listing word) as a listing
        prints it (emips.cli.encoding: a nanoMIPS one in halfwords)."""
        from .cli import encoding
        return encoding(word, self.size_at(addr), self.arch)

    def nm_insn(self, addr: int):
        """The nanoMIPS instruction at ``addr`` (emips.nanomips.NM)."""
        from . import nanomips
        size = self.nm[addr]
        raw = 0
        for k in range(0, size, 2):
            raw = raw << 16 | self.half(addr + k)
        return nanomips.decode_raw(raw, size)

    def to_ihex(self) -> str:
        """Intel HEX (types 04 / 00 / 05 / 01) of the image."""
        lines: list[str] = []

        def rec(rtype: int, offset: int, data: bytes) -> None:
            body = bytes([len(data), offset >> 8, offset & 0xFF, rtype]) + data
            lines.append(":" + body.hex().upper()
                         + f"{(-sum(body)) & 0xFF:02X}")

        upper = None
        addrs = sorted(self.image)
        i = 0
        while i < len(addrs):
            start = addrs[i]
            chunk = bytearray([self.image[start]])
            i += 1
            while (i < len(addrs) and addrs[i] == start + len(chunk)
                   and len(chunk) < 16 and (addrs[i] & 0xFFFF) != 0):
                chunk.append(self.image[addrs[i]])
                i += 1
            if start >> 16 != upper:
                upper = start >> 16
                rec(0x04, 0, upper.to_bytes(2, "big"))
            rec(0x00, start & 0xFFFF, bytes(chunk))
        rec(0x05, 0, self.entry.to_bytes(4, "big"))
        rec(0x01, 0, b"")
        return "\n".join(lines) + "\n"

    def half(self, addr: int) -> int:
        data = bytes(self.image.get(addr + i, 0) for i in range(2))
        return struct.unpack(">H" if self.endian == "big" else "<H", data)[0]

    def m16_insn(self, addr: int):
        """The MIPS16e instruction at ``addr`` (emips.mips16.M16)."""
        from . import mips16
        return mips16.decode(self.half(addr), lambda: self.half(addr + 2), arch=self.arch)

    def mm_insn(self, addr: int):
        """The microMIPS instruction at ``addr`` (emips.micromips.MM)."""
        from . import micromips
        return micromips.decode(self.half(addr), lambda: self.half(addr + 2),
                                release=self.release, dsp=self.dsp, mt=self.mt,
                                arch=self.arch)

    def listing(self, *, abi: bool = True) -> list[tuple[int, int, str, str]]:
        """``(address, word, disassembly, source)`` for every instruction
        (a MIPS16e one: its 16 / 32-bit encoding)."""
        from . import mips16
        from .disasm import disassemble
        labels = self.labels()
        out = []
        if self.arch in ("rv32", "rv64"):          # RISC-V: objdump's text, its walk
            from .riscv import Disassembler
            d = Disassembler(32 if self.arch == "rv32" else 64,
                             self.rv_extensions or "imafdc_zicsr", numeric=not abi,
                             symbols=labels)
            for a, s in sorted(self.source_map.items()):
                n = self.rv.get(a, 4)
                raw = sum(self.image.get(a + k, 0) << 8 * k for k in range(n))
                out.append((a, raw, d.text(raw, n, a), s.text.strip()))
            return out
        if self.arch == "nanomips":                # (a nanoMIPS one: 16 / 32 / 48 bits)
            from . import nanomips
            for a, s in sorted(self.source_map.items()):
                i = self.nm_insn(a)
                out.append((a, i.raw, nanomips.text(i, a, abi=abi, symbols=labels),
                            s.text.strip()))
            return out
        isa = ISA.for_release(self.release, self.dsp, self.mt, arch=self.arch)
        for a, s in sorted(self.source_map.items()):
            if a in self.m16:
                i = self.m16_insn(a)
                out.append((a, i.raw, mips16.text(i, a, abi=abi, arch=self.arch), s.text.strip()))
            elif a in self.mm:
                from . import micromips
                i = self.mm_insn(a)
                out.append((a, i.raw, micromips.text(i, a, abi=abi, release=self.release,
                                                     arch=self.arch), s.text.strip()))
            else:
                out.append((a, self.word(a), disassemble(self.word(a), a, abi=abi, isa=isa,
                                                          symbols=labels), s.text.strip()))
        return out


# ---------------------------------------------------------------- lexing

_LABEL = re.compile(r"^\s*([A-Za-z_.$][\w.$]*)\s*:")
_NAME = re.compile(r"[A-Za-z_.$][\w.$]*$")
#: Release 6 compact branch-and-link on a register: GPR 31 as the source is
#: UNPREDICTABLE (Vol. II-A rev. 6.06, B{LE,GE,GT,LT,EQ,NE}ZALC)
_ZALC = frozenset({"BLEZALC", "BGEZALC", "BGTZALC", "BLTZALC", "BEQZALC", "BNEZALC"})
#: a DSP Module accumulator operand
_AC = re.compile(r"\$ac([0-3])", re.I)


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote and (len(out) < 2 or out[-2] != "\\"):
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _split_operands(text: str) -> list[str]:
    parts, depth, cur, quote = [], 0, [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote and cur[-2:-1] != ["\\"]:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


_ESC = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}


def _unescape(body: str) -> bytes:
    out, i = bytearray(), 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            out += _ESC.get(body[i + 1], body[i + 1]).encode("latin-1")
            i += 2
        else:
            out += ch.encode("latin-1")
            i += 1
    return bytes(out)


# ------------------------------------------------------------- expressions

class _Unresolved(Exception):
    """A symbol is not (yet) defined."""


_TOKEN = re.compile(r"\s*(%hi|%lo|0[xX][0-9a-fA-F]+|0[bB][01]+|\d+|'(?:\\.|[^'])'"
                    r"|<<|>>|==|!=|<>|<=|>=|&&|\|\||[A-Za-z_.$][\w.$]*|[-+*/%()&|^~!<>])")


def _eval(expr: str, lookup: Callable[[str], int]) -> int:
    toks = []
    pos = 0
    expr = expr.strip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            raise ValueError(f"bad expression {expr!r}")
        toks.append(m.group(1))
        pos = m.end()
    i = 0

    def peek() -> str | None:
        return toks[i] if i < len(toks) else None

    def take() -> str:
        nonlocal i
        i += 1
        return toks[i - 1]

    def atom() -> int:
        t = take()
        if t == "(":
            v = binop(0)
            if take() != ")":
                raise ValueError("missing )")
            return v
        if t == "-":
            return -atom()
        if t == "+":
            return atom()
        if t == "~":
            return ~atom()
        if t == "!":
            return int(not atom())
        if t in ("%hi", "%lo"):
            if take() != "(":
                raise ValueError(f"{t} needs (...)")
            v = binop(0)
            if take() != ")":
                raise ValueError("missing )")
            v &= M32
            return ((v + 0x8000) >> 16) & 0xFFFF if t == "%hi" else _sx16(v)
        if t[0] == "'":
            return ord(_unescape(t[1:-1]).decode("latin-1"))
        if t[:2].lower() == "0x":
            return int(t, 16)
        if t[:2].lower() == "0b":
            return int(t, 2)
        if t.isdigit():
            return int(t)
        return lookup(t)

    # gas's operator ranks (expr.c op_rank): the bitwise operators bind
    # tighter than + and -; a comparison is -1 when true
    prec = {"*": 8, "/": 8, "%": 8, "<<": 8, ">>": 8, "|": 7, "^": 7, "&": 7, "!": 7,
            "+": 5, "-": 5, "==": 4, "!=": 4, "<>": 4, "<": 4, "<=": 4, ">": 4, ">=": 4,
            "&&": 3, "||": 2}

    def binop(level: int) -> int:
        v = atom()
        while peek() in prec and prec[peek()] > level:
            op = take()
            r = binop(prec[op])
            if op in ("/", "%") and not r:
                raise ValueError("division by zero")
            v = {"|": lambda: v | r, "&": lambda: v & r, "^": lambda: v ^ r,
                 "!": lambda: v | ~r, "<<": lambda: v << r, ">>": lambda: v >> r,
                 "+": lambda: v + r, "-": lambda: v - r, "*": lambda: v * r,
                 "/": lambda: int(v / r), "%": lambda: v - int(v / r) * r,
                 "==": lambda: -(v == r), "!=": lambda: -(v != r), "<>": lambda: -(v != r),
                 "<": lambda: -(v < r), "<=": lambda: -(v <= r), ">": lambda: -(v > r),
                 ">=": lambda: -(v >= r), "&&": lambda: int(bool(v and r)),
                 "||": lambda: int(bool(v or r))}[op]()
        return v

    v = binop(0)
    if i != len(toks):
        raise ValueError(f"bad expression {expr!r}")
    return v


def _sx32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def _sx16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


M64 = (1 << 64) - 1


def load_register(reg: str, value: int, dbl: bool) -> list[tuple[str, list[str]]]:
    """GNU as's ``load_register`` (gas/config/tc-mips.c): the instructions
    that put the constant ``value`` in ``reg`` — ``li`` (``dbl`` False: a
    32-bit value, sign-extended; more is an error, as in gas) or ``dli``
    (``dbl`` True: 64 bits)."""
    if not dbl and 0 <= value <= M32:           # normalize_constant_expr
        value = _sx32(value)
    if -0x8000 <= value < 0x8000:
        return [("addiu", [reg, "$zero", str(value)])]
    if 0 <= value < 0x10000:
        return [("ori", [reg, "$zero", str(value)])]
    if -(1 << 31) <= value < (1 << 31):
        out = [("lui", [reg, str((value >> 16) & 0xFFFF)])]
        if value & 0xFFFF:
            out.append(("ori", [reg, reg, str(value & 0xFFFF)]))
        return out
    if not dbl:
        raise ValueError(f"number (0x{value & M64:x}) larger than 32 bits"
                         + (" (dli loads 64 bits)" if -(1 << 63) <= value <= M64 else ""))
    if not -(1 << 63) <= value <= M64:
        raise ValueError(f"number {value:#x} larger than 64 bits")

    def shift(mn: str, n: int, src: str) -> tuple[str, list[str]]:
        return (mn + "32", [reg, src, str(n - 32)]) if n >= 32 else (mn, [reg, src, str(n)])

    u = value & M64
    hi32, lo32 = u >> 32, u & M32
    out: list[tuple[str, list[str]]] = []
    freg = "$zero"
    if hi32:
        if hi32 == M32:
            if lo32 & 0xFFFF8000 == 0xFFFF8000:
                return [("addiu", [reg, "$zero", str(_sx16(lo32))])]
            if lo32 & 0x80000000:
                out = [("lui", [reg, str(lo32 >> 16)])]
                if lo32 & 0xFFFF:
                    out.append(("ori", [reg, reg, str(lo32 & 0xFFFF)]))
                return out
        for n in range(17, 64 - 16 + 1):        # a 16-bit value shifted left
            himask, lomask = ((0xFFFF >> (32 - n), (0xFFFF << n) & M32) if n < 32
                              else (0xFFFF << (n - 32), 0))
            if not hi32 & ~himask and not lo32 & ~lomask:
                tmp = (hi32 << (32 - n)) | (lo32 >> n) if n < 32 else hi32 >> (n - 32)
                return [("ori", [reg, "$zero", str(tmp & 0xFFFF)]), shift("dsll", n, reg)]
        hi, lo = hi32, lo32                     # the lowest one bit
        if lo:
            bit = (lo & -lo).bit_length() - 1
            if bit:
                lo = (lo >> bit) | ((hi & ((1 << bit) - 1)) << (32 - bit))
                hi >>= bit
        else:
            bit = 32 + (hi & -hi).bit_length() - 1
            lo, hi = hi >> (bit - 32), 0
        if (hi == 0 and not (lo + 1) & lo) or (lo == M32 and not (hi + 1) & hi):
            top = 32 - hi32.bit_length()        # COUNT_TOP_ZEROES
            if top:                             # a run of ones: -1, shifted
                out = [("addiu", [reg, "$zero", "-1"])]
                if bit:
                    out.append(shift("dsll", bit + top, reg))
                out.append(shift("dsrl", top, reg))
                return out
        out = load_register(reg, _sx32(hi32), False)
        freg = reg
    if not lo32 & 0xFFFF0000:
        if freg != "$zero":
            out.append(("dsll32", [reg, freg, "0"]))
    else:
        if freg == "$zero" and lo32 == M32:
            return [("lui", [reg, "65535"]), ("dsrl32", [reg, reg, "0"])]
        if freg != "$zero":
            out.append(("dsll", [reg, freg, "16"]))
        out.append(("ori", [reg, freg, str(lo32 >> 16)]))
        out.append(("dsll", [reg, reg, "16"]))
        freg = reg
    if lo32 & 0xFFFF:
        out.append(("ori", [reg, freg, str(lo32 & 0xFFFF)]))
    return out


#: the n64 register names that differ from o32's (GNU as N32N64_SYMBOLIC_
#: REGISTER_NAMES): $a4-$a7 / $ta0-$ta3 = $8-$11, $t0-$t3 = $12-$15
N64_NAMES = {**{f"a{i}": 4 + i for i in range(4, 8)}, **{f"ta{i}": 8 + i for i in range(4)},
             **{f"t{i}": 12 + i for i in range(4)}}
_N64_REG = re.compile(r"(?<![\w$])\$(t[0-7]|ta[0-3]|a[4-7])(?![\w$])", re.I)


def n64_operand(tok: str) -> str:
    """An operand with its n64 register names written as numbers; ``$t4`` -
    ``$t7`` (o32 names only) are an error."""
    def sub(m: re.Match) -> str:
        name = m.group(1).lower()
        if name not in N64_NAMES:
            raise ValueError(f"${name} is not an n64 register name (the MIPS64 ABI "
                             f"has $t0-$t3 = $12-$15, $a4-$a7 = $8-$11)")
        return f"${N64_NAMES[name]}"
    return _N64_REG.sub(sub, tok)


# ------------------------------------------------------------ the assembler

#: (mnemonic -> handler) for pseudo-instructions; each returns a list of
#: (mnemonic, operands) real instructions.
_BRANCH_MACROS = {
    "blt": ("slt", False, "bne"), "bltu": ("sltu", False, "bne"),
    "bge": ("slt", False, "beq"), "bgeu": ("sltu", False, "beq"),
    "bgt": ("slt", True, "bne"), "bgtu": ("sltu", True, "bne"),
    "ble": ("slt", True, "beq"), "bleu": ("sltu", True, "beq"),
}

#: strict mode (``pseudo=False``): what to write instead of each pseudo-op
PSEUDO_HINTS = {
    "move": "or rd, rs, $zero",
    "neg": "sub rd, $zero, rs",
    "negu": "subu rd, $zero, rs",
    "not": "nor rd, rs, $zero",
    "beqz": "beq rs, $zero, label",
    "bnez": "bne rs, $zero, label",
    "li": "addiu rt, $zero, imm / ori rt, $zero, imm / lui rt, hi + ori rt, rt, lo",
    "la": "lui rt, %hi(label) + addiu rt, rt, %lo(label)",
    "dli": "addiu / ori / lui and dsll / dsll32 / ori, as GNU as",
    "dla": "lui rt, %hi(label) + daddiu rt, rt, %lo(label)",
    "dneg": "dsub rd, $zero, rs",
    "dnegu": "dsubu rd, $zero, rs",
    **{mn: f"{cmp} $at, ... + {br} $at, $zero, label"
       for mn, (cmp, _swap, br) in _BRANCH_MACROS.items()},
}

#: GNU as's macros of the ALU instructions (gas/config/tc-mips.c,
#: macro()): a constant in place of rt -> the immediate instruction when
#: it fits, else the constant in $at (as ``li``) and the register form
_ALU_IMMEDIATE = {"add": "addi", "addu": "addiu", "sub": "addi", "subu": "addiu",
                  "and": "andi", "or": "ori", "xor": "xori", "nor": "ori",
                  "slt": "slti", "sltu": "sltiu",
                  # MIPS64 (PLAN M22)
                  "dadd": "daddi", "daddu": "daddiu", "dsub": "daddi", "dsubu": "daddiu"}
#: shifts by a register written with the immediate mnemonic
_SHIFT_VARIABLE = {"sll": "sllv", "srl": "srlv", "sra": "srav", "rotr": "rotrv",
                   "dsll": "dsllv", "dsrl": "dsrlv", "dsra": "dsrav", "drotr": "drotrv"}
#: immediates GNU as also takes as a 16-bit pattern (0x8000-0xFFFF = negative)
_SIGNED_IMMEDIATE = {"addi", "addiu", "slti", "sltiu", "daddi", "daddiu"}
#: ``op rd, rt`` = ``op rd, rd, rt`` (GNU as): register forms ...
_SHORT_REGISTER = set(_ALU_IMMEDIATE) | set(_SHIFT_VARIABLE) | {"mul", "movn", "movz",
                                                                  "dmul"}
#: ... and immediate forms
_SHORT_IMMEDIATE = set(_ALU_IMMEDIATE) | set(_SHIFT_VARIABLE) | {"mul", "addi", "addiu",
                                                                   "andi", "ori", "xori",
                                                                   "slti", "sltiu", "dmul",
                                                                   "daddi", "daddiu"}
#: the constant GNU as puts in $at for these is 64 bits on a MIPS64 core
#: (``load_register(..., HAVE_64BIT_GPRS)``); add / sub's is 32 bits
_DBL_CONSTANT = {"and", "or", "xor", "nor", "slt", "sltu", "dadd", "daddu", "dsub",
                 "dsubu", "dmul"}
#: doubleword shifts by 32-63: the ``*32`` instruction (GNU as)
_SHIFT32 = {"dsll", "dsrl", "dsra", "drotr"}

_DIRECTIVES_IGNORED = {".globl", ".global", ".ent", ".end", ".type", ".size",
                       ".extern", ".file", ".section", ".previous", ".frame",
                       ".mask", ".fmask", ".abicalls", ".option"}


class _Hooks:
    """What emips.gasread asks the assembler (``.if``, ``.ifdef``, a macro's
    name, comments)."""

    onechar = True
    KNOWN = frozenset((".text", ".data", ".ktext", ".kdata", ".set", ".eqv", ".equ", ".align",
                       ".space", ".word", ".half", ".byte", ".float", ".double", ".dword",
                       ".8byte", ".ascii", ".asciiz", *_DIRECTIVES_IGNORED))

    def __init__(self, asm: "Assembler") -> None:
        self.asm = asm

    def absolute(self, text: str) -> int:
        return self.asm._const(text, self.asm.reader.current.line)

    def defined(self, name: str) -> bool:
        return name in self.asm.symbols

    def known(self, directive: str) -> bool:
        return directive in self.KNOWN

    def strip(self, line: str) -> str:
        return _strip_comment(line)


class _Relax(Exception):
    """A MIPS16e statement needs its long form: lay the program out again."""


#: The MT Module's MFTR / MTTR idioms (MD00378 Tables 5.1 / 5.2, GNU as'
#: MT32 entries): mnemonic -> (instruction, u, sel, h, the target's kind,
#: its fixed register number or None).  ``"gpr"`` / ``"fpr"`` / ``"num"``:
#: the operand names the target; ``"ac"``: an optional $acN (rt / rd = 4N +
#: the fixed number); None: no target operand.
_MT_IDIOMS: dict[str, tuple[str, int, int, int, str | None, int | None]] = {
    "mftc0": ("mftr", 0, 0, 0, "num", None), "mttc0": ("mttr", 0, 0, 0, "num", None),
    "mftgpr": ("mftr", 1, 0, 0, "gpr", None), "mttgpr": ("mttr", 1, 0, 0, "gpr", None),
    "mftlo": ("mftr", 1, 1, 0, "ac", 0), "mttlo": ("mttr", 1, 1, 0, "ac", 0),
    "mfthi": ("mftr", 1, 1, 0, "ac", 1), "mtthi": ("mttr", 1, 1, 0, "ac", 1),
    "mftacx": ("mftr", 1, 1, 0, "ac", 2), "mttacx": ("mttr", 1, 1, 0, "ac", 2),
    "mftdsp": ("mftr", 1, 1, 0, None, 16), "mttdsp": ("mttr", 1, 1, 0, None, 16),
    "mftc1": ("mftr", 1, 2, 0, "fpr", None), "mttc1": ("mttr", 1, 2, 0, "fpr", None),
    "mfthc1": ("mftr", 1, 2, 1, "fpr", None), "mtthc1": ("mttr", 1, 2, 1, "fpr", None),
    "cftc1": ("mftr", 1, 3, 0, "num", None), "cttc1": ("mttr", 1, 3, 0, "num", None),
    "mftc2": ("mftr", 1, 4, 0, "num", None), "mttc2": ("mttr", 1, 4, 0, "num", None),
    "mfthc2": ("mftr", 1, 4, 1, "num", None), "mtthc2": ("mttr", 1, 4, 1, "num", None),
    "cftc2": ("mftr", 1, 5, 0, "num", None), "cttc2": ("mttr", 1, 5, 0, "num", None),
}


def _mt_idiom(mn: str, ops: list[str], err) -> tuple[str, list[str]]:
    """``mftc0 rd, rt[, sel]`` ... -> the MFTR / MTTR it stands for."""
    insn, u, sel, h, kind, fixed = _MT_IDIOMS[mn]
    if not ops:
        raise err("expects a GPR")
    gpr, rest = ops[0], [o.strip() for o in ops[1:]]
    if kind == "num" and mn.endswith("c0") and len(rest) == 2:
        sel = None                              # mftc0 rd, rt, sel
    want = {"num": 1, "gpr": 1, "fpr": 1, "ac": (0, 1), None: 0}[kind]
    n = len(rest) - (1 if sel is None else 0)
    if n not in (want if isinstance(want, tuple) else (want,)):
        raise err(f"expects {1 + (want if isinstance(want, int) else 1)} operands")
    if kind == "gpr":
        r = parse_gpr(rest[0])
        if r is None:
            raise err(f"GPR expected, got {rest[0]!r}")
        target = str(r)
    elif kind == "fpr":
        r = parse_fpr(rest[0]) if rest[0].lower().startswith("$f") else parse_numreg(rest[0])
        if r is None:
            raise err(f"FPR expected, got {rest[0]!r}")
        target = str(r)
    elif kind == "ac":
        a = 0
        if rest:
            m = _AC.fullmatch(rest[0])
            if not m:
                raise err(f"accumulator $ac0..$ac3 expected, got {rest[0]!r}")
            a = int(m.group(1))
        target = str(4 * a + fixed)
    elif kind is None:
        target = str(fixed)
    else:
        target = rest[0]
    return insn, [gpr, target, str(u), rest[1] if sel is None else str(sel), str(h)]


@dataclass
class _Stmt:
    line: int
    text: str
    seg: str
    addr: int
    mnemonic: str
    ops: list[str]
    reorder: bool
    at_ok: bool
    size: int = 0
    #: R6: a nop is put in this compact branch's forbidden slot
    forbidden_nop: bool = False
    #: a MIPS16e statement (.set mips16)
    m16: bool = False
    #: a microMIPS statement (.set micromips)
    mm: bool = False
    #: microMIPS, noreorder: the size the previous branch's delay slot
    #: demands of this statement's first instruction (0 = any)
    slot: int = 0
    #: the DSP Module level ``.set dsp`` / ``dspr2`` / ``nodsp`` allows here
    dsp: int = 2
    #: ``.set mt`` / ``nomt``: the MT Module's instructions are allowed
    mt: bool = True
    #: ``.set msa`` / ``nomsa``: MSA's instructions are allowed
    msa: bool = True


class Assembler:
    def __init__(self, *, endian: str = "big", isa: ISA | None = None,
                 bases: dict[str, int] | None = None,
                 release: str | None = None, pseudo: bool = True,
                 reorder: bool = True, arch: str = "mips32") -> None:
        if endian not in ("big", "little"):
            raise ValueError("endian must be 'big' or 'little'")
        self.endian = endian
        #: pseudo-instructions allowed (False: strict mode)
        self.pseudo = pseudo
        #: initial ``.set reorder`` state (nop in every delay slot)
        self.reorder = reorder
        self.isa = isa or ISA.for_release(release, arch=arch)
        self.bases = dict(SEGMENT_BASES, **(bases or {}))
        #: instructions of the target release only, by mnemonic (the DSP
        #: Module's accumulator forms by row name: ``mult@dsp``)
        self._by_name = {(e["name"] if "@DSP" in e["name"]
                          else e.get("mnemonic", e["name"])).lower(): e
                         for e in self.isa.instructions
                         if self.isa.in_release(e["name"])}
        for e in self.isa.instructions:           # MIPS64 dextm ... dinsu by their names
            if e["name"] in _DFIELD and self.isa.in_release(e["name"]):
                self._by_name.setdefault(e["name"].lower(), e)
        self._all = {e.get("mnemonic", e["name"]).lower()
                     for e in self.isa.instructions}
        #: a MIPS64 target (PLAN M22): n64 names, the doubleword macros
        self.wide = self.isa.arch == "mips64"
        #: the doubleword rows (``.set mips32*`` forbids them)
        self._rows64 = {mn for mn, e in self._by_name.items() if e.get("arch") == "mips64"}

    # -- public -------------------------------------------------------------

    def assemble(self, source: str, path: str | None = None,
                 include_dirs: tuple[str, ...] = ()) -> Program:
        self._path, self._include_dirs = path, include_dirs
        #: MIPS16e statements (by line) too far for their short form
        self._m16_long: set[int] = set()
        for _ in range(64):                     # MIPS16e relaxation: sizes only grow
            try:
                return self._assemble_once(source)
            except _Relax:
                continue
        raise AsmError(0, "MIPS16e branch relaxation does not converge")

    def _assemble_once(self, source: str) -> Program:
        self.symbols: dict[str, int] = {}
        #: names defined by .eqv / .equ / .set NAME (not labels)
        self._consts: set[str] = set()
        #: labels of MIPS16e / microMIPS code (their address, taken as data,
        #: has bit 0 set)
        self._m16_labels: set[str] = set()
        self._mode16 = False
        self._modemm = False
        self._mm_enc = None
        self._seen_m16 = self._seen_mm = False
        self._dsp_set = 2                       # .set dsp / dspr2 / nodsp
        self._mt_set = True                     # .set mt / nomt
        self._msa_set = True                    # .set msa / nomsa
        self._isa64 = True                      # .set mips64* / mips32* (MIPS64)
        self.prog = Program(endian=self.endian, release=self.isa.release,
                            arch=self.isa.arch)
        stmts = self._pass1(source)
        for st in stmts:
            self._emit(st)
        self.prog.printed = list(self.reader.printed)
        self.prog.symbols = dict(self.symbols)
        for name in ("main", "__start"):
            if name in self.symbols:
                self.prog.entry = self.symbols[name] | (name in self._m16_labels)
                break
        else:
            base = self.bases["text"]
            self.prog.entry = base | (base in self.prog.mm or base in self.prog.m16)
        return self.prog

    # -- pass 1: layout -------------------------------------------------------

    def _pass1(self, source: str) -> list[_Stmt]:
        lc = dict(self.bases)
        seg = "text"
        reorder, at_ok = self.reorder, True
        stmts: list[_Stmt] = []
        #: labels at the current location with nothing emitted after them
        #: yet: an auto-aligning directive moves them along (GNU as, MARS)
        self._pending: list[str] = []
        prev: _Stmt | None = None                 # previous instruction
        from . import gasread
        self.reader = gasread.Reader(source, _Hooks(self), path=self._path,
                                     include_dirs=self._include_dirs,
                                     warn=lambda n, msg: self.prog.warnings.append((n, msg)))
        for gst in self.reader:
            n, text = gst.line, gst.text
            raw = gst.text.strip() if gst.expanded else gst.raw
            while True:
                m = _LABEL.match(text)
                if not m:
                    break
                name = m.group(1)
                if name in self.symbols:
                    raise AsmError(n, f"label {name!r} defined twice", raw)
                self.symbols[name] = lc[seg]
                self._pending.append(name)
                text = text[m.end():]
            text = text.strip()
            if not text:
                continue
            head, _, rest = text.partition(" ")
            if "\t" in head:
                head, _, more = head.partition("\t")
                rest = more + " " + rest
            if self._consts:                    # (gas: a .set / .equ symbol's value now)
                rest = self._current_constants(head.lower(), rest)
            mnemonic, ops = head.lower(), _split_operands(rest.strip())
            try:
                if self.wide and not mnemonic.startswith("."):
                    ops = [n64_operand(o) for o in ops]
                    if not self._isa64 and mnemonic in self._rows64:
                        raise AsmError(n, f"{mnemonic} is a MIPS64 instruction: not "
                                       "after .set mips32 (.set mips64)", raw)
                if mnemonic.startswith("."):
                    seg, reorder, at_ok = self._directive_layout(
                        n, raw, mnemonic, ops, rest, lc, seg, reorder, at_ok, stmts)
                    continue
                if seg in ("data", "kdata"):
                    raise AsmError(n, f"instruction in .{seg} segment", raw)
                if self._modemm:                # microMIPS: halfwords, sizes 2 / 4
                    if lc[seg] & 1:
                        raise AsmError(n, "microMIPS instruction not halfword-aligned", raw)
                    self._m16_labels.update(self._pending)
                    st = _Stmt(n, raw, seg, lc[seg], mnemonic, ops, reorder, at_ok, mm=True,
                               msa=self._msa_set)
                    if prev is not None and prev.mm and not prev.reorder and \
                            prev.addr + prev.size == lc[seg]:
                        st.slot = self._mm_slot_need(prev)
                    st.size = sum(i.size for i in self._mm_encode(st, final=False))
                    stmts.append(st)
                    lc[seg] += st.size
                    self._pending = []
                    prev = st
                    continue
                if self._mode16:                # MIPS16e: halfwords, sizes 2 / 4
                    if lc[seg] & 1:
                        raise AsmError(n, "MIPS16e instruction not halfword-aligned", raw)
                    self._m16_labels.update(self._pending)
                    st = _Stmt(n, raw, seg, lc[seg], mnemonic, ops, reorder, at_ok, m16=True)
                    st.size = self._m16_size(st)
                    stmts.append(st)
                    lc[seg] += st.size
                    self._pending = []
                    prev = None
                    continue
                if lc[seg] & 3:
                    raise AsmError(n, "instruction not word-aligned" + (
                        " (after MIPS16e / microMIPS code: .align 2)" if self._m16_labels
                        else ""), raw)
                st = _Stmt(n, raw, seg, lc[seg], mnemonic, ops, reorder, at_ok,
                           dsp=self._dsp_set, mt=self._mt_set, msa=self._msa_set)
                insns = self._expand(st)
                if (prev is not None and prev.reorder and prev.seg == seg
                        and prev.addr + prev.size == lc[seg]
                        and self._conditional_compact(self._expand(prev)[-1][0])
                        and self._cti(insns[0][0])):
                    # R6 forbidden slot: GNU as puts a nop there (.set reorder)
                    prev.forbidden_nop = True
                    prev.size += 4
                    lc[seg] += 4
                    st.addr = lc[seg]
                    for name in self._pending:
                        self.symbols[name] = lc[seg]
                st.size = 4 * len(insns)
                stmts.append(st)
                lc[seg] += st.size
                self._pending = []
                prev = st
            except AsmError:
                raise
            except (ValueError, KeyError) as exc:
                raise AsmError(n, str(exc), raw) from None
        return stmts

    def _current_constants(self, mnemonic: str, rest: str) -> str:
        """The operands with every .set / .equ / .eqv name replaced by the
        value it has here: gas reads a constant symbol when the statement is
        assembled (``.set n, n + 1`` in a ``.rept`` gives each copy its own
        value) — this assembler evaluates operands in its second pass."""
        define = mnemonic in (".set", ".eqv", ".equ")
        if define:                              # (NAME, VALUE: only the value)
            name, sep, value = rest.partition(",")
            if not sep:
                return rest
            return name + sep + self._current_constants("", value)
        parts = re.split(r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'])')", rest)

        def sub(m: re.Match) -> str:
            name = m.group(0)
            return str(self.symbols[name]) if name in self._consts else name

        return "".join(p if k % 2 else re.sub(r"(?<![\w.$])[A-Za-z_.$][\w.$]*", sub, p)
                       for k, p in enumerate(parts))

    def _directive_layout(self, n, raw, d, ops, rest, lc, seg, reorder,
                          at_ok, stmts):
        def stmt(size: int, align: int = 1) -> None:
            pad = (-lc[seg]) % align
            st = _Stmt(n, raw, seg, lc[seg] + pad, d, ops, reorder, at_ok, size)
            stmts.append(st)
            for name in self._pending:
                self.symbols[name] = st.addr
            self._pending = []
            lc[seg] += pad + size

        if d in (".text", ".data", ".ktext", ".kdata"):
            self._pending = []
            seg = d[1:]
            if ops:
                v = self._const(ops[0], n)
                if not -(1 << 31) <= v <= M32 and not (0xFFFFFFFF80000000 <= v <= M64):
                    raise AsmError(n, f"segment address 0x{v & M64:x} is not a 32-bit "
                                   "address (MIPS64 programs are -msym32: every address in "
                                   "the 32-bit compatibility space, 0xFFFFFFFF80000000 = "
                                   "0x80000000)", raw)
                lc[seg] = v & M32
            return seg, reorder, at_ok
        if d == ".set":
            word = ops[0].lower() if ops else ""
            if len(ops) == 2 and _NAME.match(ops[0]):
                self.symbols[ops[0]] = self._const(ops[1], n)
                self._consts.add(ops[0])
            elif word in ("reorder", "noreorder"):
                reorder = word == "reorder"
            elif word in ("at", "noat"):
                at_ok = word == "at"
            elif word in ("mips16", "nomips16"):
                if word == "mips16" and self.isa.release == "R6":
                    raise AsmError(n, "Release 6 has no MIPS16e", raw)
                if word == "mips16" and self._seen_mm:
                    raise AsmError(n, "MIPS16e after microMIPS code: no core has both", raw)
                self._mode16 = word == "mips16"
                self._seen_m16 |= self._mode16
            elif word in ("micromips", "nomicromips"):
                if word == "micromips" and self.isa.release == "R1":
                    raise AsmError(n, "microMIPS32 needs Release 2+ (the M14K is a Release 2 "
                                   "core with microMIPS)", raw)
                if word == "micromips" and self.wide and self.isa.release == "R2":
                    raise AsmError(n, "microMIPS64 needs Release 3+ (MD00594 3.1)", raw)
                if word == "micromips" and self._seen_m16:
                    raise AsmError(n, "microMIPS after MIPS16e code: no core has both", raw)
                self._modemm = word == "micromips"
                self._seen_mm |= self._modemm
            elif word in ("dsp", "dspr2", "dspr3", "nodsp", "nodspr2", "nodspr3"):
                # GNU as: which DSP Module instructions are allowed
                if not word.startswith("no") and self.isa.release == "R1":
                    raise AsmError(n, "the DSP Module needs Release 2+ (MD00374 3.1)", raw)
                self._dsp_set = (0 if word == "nodsp" else 1 if word.startswith("nodspr")
                                 else 1 if word == "dsp" else 2)
            elif word.startswith(("mips32", "mips64")) or word == "mips0":
                # GNU as: the ISA level; here the doubleword instructions'
                if word.startswith("mips64") and not self.wide:
                    raise AsmError(n, f".set {word}: this is a MIPS32 program; assemble "
                                   "it for MIPS64 (--arch mips64, or a line "
                                   "'# eMIPS: --arch mips64')", raw)
                self._isa64 = not word.startswith("mips32")
            elif word in ("mt", "nomt"):        # GNU as: the MT Module's instructions
                if word == "mt" and self.isa.release in ("R1", "R6"):
                    raise AsmError(n, "the MT Module needs Release 2-5 (MD00378 1.1; "
                                   "Release 6 has none)", raw)
                self._mt_set = word == "mt"
            elif word in ("msa", "nomsa"):      # GNU as: MSA's instructions
                if word == "msa" and self.isa.release not in ("R5", "R6"):
                    raise AsmError(n, "MSA needs Release 5+ (MD00866 1.1)", raw)
                self._msa_set = word == "msa"
            return seg, reorder, at_ok
        if d in (".eqv", ".equ"):
            if len(ops) != 2:
                raise AsmError(n, f"{d} NAME, value", raw)
            self.symbols[ops[0]] = self._const(ops[1], n)
            self._consts.add(ops[0])
            return seg, reorder, at_ok
        if d in _DIRECTIVES_IGNORED:
            return seg, reorder, at_ok
        if d == ".align":                       # moves pending labels too
            a = 1 << self._const(ops[0], n)
            pad = (-lc[seg]) % a
            if (self._mode16 or self._modemm) and seg in ("text", "ktext") and pad:
                # MIPS16e / microMIPS code: GNU as fills with 16-bit NOPs
                stmts.append(_Stmt(n, raw, seg, lc[seg], ".fill16", [], reorder, at_ok, pad,
                                   mm=self._modemm))
            lc[seg] += pad
            for name in self._pending:
                self.symbols[name] = lc[seg]
        elif d == ".space":
            stmt(self._const(ops[0], n))
        elif d == ".word":
            stmt(4 * len(ops), 4)
        elif d == ".half":
            stmt(2 * len(ops), 2)
        elif d == ".byte":
            stmt(len(ops))
        elif d == ".float":
            stmt(4 * len(ops), 4)
        elif d == ".double":
            stmt(8 * len(ops), 8)
        elif d in (".dword", ".8byte"):
            stmt(8 * len(ops), 8)
        elif d in (".ascii", ".asciiz"):
            data = self._strings(rest, n, raw, d == ".asciiz")
            stmt(len(data))
        else:
            raise AsmError(n, f"unknown directive {d}", raw)
        return seg, reorder, at_ok

    def _strings(self, rest: str, n: int, raw: str, zero: bool) -> bytes:
        out = b""
        for part in _split_operands(rest):
            if len(part) < 2 or part[0] != '"' or part[-1] != '"':
                raise AsmError(n, "string literal expected", raw)
            out += _unescape(part[1:-1]) + (b"\0" if zero else b"")
        return out

    # -- expressions ----------------------------------------------------------

    def _lookup(self, name: str) -> int:
        if name in self.symbols:
            return self.symbols[name]
        raise _Unresolved(name)

    def _const(self, expr: str, n: int) -> int:
        try:
            return _eval(expr, self._lookup)
        except _Unresolved as exc:
            raise AsmError(n, f"undefined symbol {exc}") from None
        except ValueError as exc:
            raise AsmError(n, str(exc)) from None

    def _value(self, expr: str, st: _Stmt) -> int:
        try:
            return _eval(expr, self._lookup)
        except _Unresolved as exc:
            raise AsmError(st.line, f"undefined symbol {exc}", st.text) from None
        except ValueError as exc:
            raise AsmError(st.line, str(exc), st.text) from None

    def _is_constant(self, expr: str) -> bool:
        """True when ``expr`` uses no labels (only numbers and .eqv/.equ/.set
        constants).  Expansion sizes depend on this, so it gives the same
        answer in both passes."""
        names = [t for t in _TOKEN.findall(expr)
                 if re.fullmatch(r"[A-Za-z_.$][\w.$]*", t)]
        return all(t in self._consts for t in names)

    # -- pseudo-instruction expansion -------------------------------------------

    def _expand(self, st: _Stmt) -> list[tuple[str, list[str]]]:
        """Real instructions (mnemonic, operands) for a source statement,
        including the reorder-mode delay-slot ``nop``."""
        mn, ops = st.mnemonic, st.ops
        out: list[tuple[str, list[str]]]
        at = "$at"

        def need_at() -> None:
            if not st.at_ok:
                raise AsmError(st.line, f"{mn} needs $at, but .set noat", st.text)

        if not self.pseudo and mn in PSEUDO_HINTS:
            raise AsmError(st.line, f"{mn} is a pseudo-instruction (strict mode); "
                           f"write: {PSEUDO_HINTS[mn]}", st.text)

        if mn == "nop":
            out = [("nop", [])]
        elif mn == "move":
            self._nops(st, 2)
            out = [("or", [ops[0], ops[1], "$zero"])]
        elif mn in ("neg", "negu"):
            self._nops(st, 2)
            out = [("sub" if mn == "neg" else "subu", [ops[0], "$zero", ops[1]])]
        elif mn in ("dneg", "dnegu") and self.wide:
            self._nops(st, 2)
            out = [("dsub" if mn == "dneg" else "dsubu", [ops[0], "$zero", ops[1]])]
        elif mn == "not":
            self._nops(st, 2)
            out = [("nor", [ops[0], ops[1], "$zero"])]
        elif mn in ("beqz", "bnez"):
            self._nops(st, 2)
            out = [("beq" if mn == "beqz" else "bne", [ops[0], "$zero", ops[1]])]
        elif mn in _BRANCH_MACROS:
            self._nops(st, 3)
            cmp, swap, br = _BRANCH_MACROS[mn]
            need_at()
            a, b = (ops[1], ops[0]) if swap else (ops[0], ops[1])
            out = [(cmp, [at, a, b]), (br, [at, "$zero", ops[2]])]
        elif mn == "li":
            self._nops(st, 2)
            out = self._li(st, ops[0], ops[1])
        elif mn == "la" and not self.wide:
            self._nops(st, 2)
            out = self._li(st, ops[0], ops[1])
        elif mn in ("la", "dla", "dli") and self.wide:
            self._nops(st, 2)
            out = self._dla(st, mn, ops[0], ops[1])
        elif ((mn in self._FP_MEM or mn in self._by_name
               and self._by_name[mn]["family"] in ("load", "store")
               and "mem" in operand_spec(self._by_name[mn]))
              and len(ops) == 2 and self._mem_needs_macro(ops[1], st)):
            if not self.pseudo:
                raise AsmError(st.line, f"{mn} with a label or 32-bit address is a "
                               "pseudo-instruction (strict mode); write: lui $at, "
                               f"%hi(label) + {mn} rt, %lo(label)($at)", st.text)
            out = self._mem_macro(mn, ops, st)
        elif mn in ("jr", "jr.hb") and mn not in self._by_name:
            # Release 6: JR is JALR with rd = $zero (GNU as does the same)
            out = [(mn.replace("jr", "jalr"), ["$zero"] + ops)]
        elif mn in ("div", "divu", "ddiv", "ddivu") and len(ops) == 3 and \
                self._by_name.get(mn, {}).get("name", "") in ("DIV", "DIVU", "DDIV", "DDIVU"):
            if parse_gpr(ops[0]) != 0:
                raise AsmError(st.line, f"{mn} rd form: first operand must be $zero", st.text)
            out = [(mn, ops[1:])]
        elif self._dsp_form(mn, ops) != mn:
            out = [(mn + "@dsp", ops)]          # mult $ac1, ...: the DSP Module's form
        elif mn == "balign" and len(ops) == 3 and self._is_constant(ops[2]) and \
                self._value(ops[2], st) in (0, 2):
            # GNU as: bp 0 / 2 are UNPREDICTABLE (MD00374 BALIGN): a nop / packrl.ph
            out = ([("nop", [])] if self._value(ops[2], st) == 0
                   else [("packrl.ph", [ops[0], ops[0], ops[1]])])
        elif mn in _MT_IDIOMS and "mftr" in self._by_name:
            out = [_mt_idiom(mn, ops, lambda m: AsmError(st.line, f"{mn}: {m}", st.text))]
        elif (alu := self._alu_macro(st, need_at)) is not None:
            out = alu
        elif mn in self._by_name:
            out = [(mn, ops)]
        elif mn in self._all:
            raise AsmError(st.line, f"{mn} does not exist in "
                           f"{'MIPS64' if self.wide else 'MIPS32'} {self.isa.release}",
                           st.text)
        elif not self.wide and mn in _mips64_mnemonics():
            raise AsmError(st.line, f"{mn} is a MIPS64 instruction: assemble the program "
                           "for MIPS64 (--arch mips64, or a line '# eMIPS: --arch "
                           "mips64')", st.text)
        else:
            raise AsmError(st.line, f"unknown instruction {mn!r}", st.text)

        last = out[-1][0]
        if st.reorder and last in self._by_name and self._has_delay_slot(last):
            out.append(("nop", []))
        return out

    def _alu_macro(self, st: _Stmt, need_at) -> list[tuple[str, list[str]]] | None:
        """GNU as's forms of the ALU instructions, expanded exactly as it
        does: ``op rd, rt`` = ``op rd, rd, rt``; a constant in place of a
        register (``or rd, rs, 0x12345``: ``ori`` when it fits, else the
        constant in ``$at``, ``sub`` / ``subu`` as an add of -imm, ``nor``
        as ``ori`` + ``nor``, ``mul`` as ``mult`` + ``mflo``); ``sll rd, rt,
        rs`` = ``sllv``; ``addiu rt, rs, 0xFFFF`` = -1; ``rotr`` by any
        constant, modulo 32.  None: not one of them (the instruction as
        written)."""
        mn, ops = st.mnemonic, list(st.ops)
        if (mn not in self._by_name and not (mn == "dmul" and self.wide)) \
                or not 2 <= len(ops) <= 3:
            return None
        if mn == "dmul" and "dmult" not in self._by_name:
            return None                         # Release 6: DMUL, no macro (GNU as)
        if ops[-1].strip().startswith("$") and parse_gpr(ops[-1]) is None:
            return None                         # a bad register: the plain error
        short = len(ops) == 2
        if short:
            reg = parse_gpr(ops[1]) is not None
            if mn not in (_SHORT_REGISTER if reg else _SHORT_IMMEDIATE):
                return None
            self._strict(st, f"{mn} {ops[0]}, {ops[0]}, {ops[1]}")
            ops = [ops[0], ops[0], ops[1]]
        rd, rs, third = ops
        if parse_gpr(third) is not None:
            if mn in _SHIFT_VARIABLE and _SHIFT_VARIABLE[mn] in self._by_name:
                self._strict(st, f"{_SHIFT_VARIABLE[mn]} {rd}, {rs}, {third}")
                return [(_SHIFT_VARIABLE[mn], ops)]
            if mn == "dmul" and "dmul" not in self._by_name:   # before Release 6
                self._strict(st, f"dmultu {rs}, {third} + mflo {rd}")
                return [("dmultu", [rs, third]), ("mflo", [rd])]     # (GNU as: dmultu)
            return [(mn, ops)] if short else None
        if mn == "rotr" and self._is_constant(third) and \
                not 0 <= self._value(third, st) <= 31:          # GNU as: modulo 32
            v = self._value(third, st) & 31
            self._strict(st, f"rotr {rd}, {rs}, {v}")
            return [("rotr", [rd, rs, str(v)])]
        if mn in _SHIFT32 and self._is_constant(third):
            v = self._value(third, st)
            if mn == "drotr":                   # GNU as: modulo 64
                v &= 63
            if 32 <= v <= 63:                   # the *32 instruction (GNU as)
                self._strict(st, f"{mn}32 {rd}, {rs}, {v - 32}")
                return [(mn + "32", [rd, rs, str(v - 32)])]
            if mn == "drotr" or short:
                return [(mn, [rd, rs, str(v)])]
            return None
        if mn in _SIGNED_IMMEDIATE:
            if self._is_constant(third) and 0x8000 <= self._value(third, st) <= 0xFFFF:
                return [(mn, [rd, rs, str(_sx16(self._value(third, st)))])]
            return [(mn, ops)] if short else None
        if mn not in _ALU_IMMEDIATE and not (mn == "mul" and "mult" in self._by_name) \
                and not (mn == "dmul" and self.wide):
            return [(mn, ops)] if short else None
        if not self._is_constant(third):
            raise AsmError(st.line, f"{mn}: the third operand must be a register or a "
                           "constant", st.text)
        dbl = self.wide and mn in _DBL_CONSTANT        # a 64-bit constant (GNU as)
        v = raw = self._value(third, st)
        if not dbl:
            if not -(1 << 31) <= v <= M32:              # GNU as: "larger than 32 bits"
                raise AsmError(st.line, f"{mn}: number {v:#x} larger than 32 bits",
                               st.text)
            if not self.wide:       # (MIPS64: gas fits the immediate to the number
                v = _sx32(v)        # as written, and loads its 32 bits sign-extended)
        imm = _ALU_IMMEDIATE.get(mn)
        if mn in ("and", "or", "xor") and 0 <= v <= 0xFFFF:
            self._strict(st, f"{imm} {rd}, {rs}, {v}")
            return [(imm, [rd, rs, str(v)])]
        if mn == "nor" and 0 <= v <= 0xFFFF:
            self._strict(st, f"ori {rd}, {rs}, {v} + nor {rd}, {rd}, $zero")
            return [("ori", [rd, rs, str(v)]), ("nor", [rd, rd, "$zero"])]
        if mn in ("add", "addu", "slt", "sltu", "dadd", "daddu") and imm in self._by_name \
                and -0x8000 <= v <= 0x7FFF:
            self._strict(st, f"{imm} {rd}, {rs}, {v}")
            return [(imm, [rd, rs, str(v)])]
        if mn in ("sub", "subu", "dsub", "dsubu") and imm in self._by_name \
                and -0x8000 <= -v <= 0x7FFF:
            self._strict(st, f"{imm} {rd}, {rs}, {-v}")
            return [(imm, [rd, rs, str(-v)])]
        mul = {"mul": "mult", "dmul": "dmult"}.get(mn)
        if mul is not None and mul not in self._by_name:  # Release 6: MUL / DMUL rd, rs, rt
            mul = None
        self._strict(st, f"{'dli' if dbl else 'li'} $at, {third} + " +
                     (f"{mul} {rs}, $at + mflo {rd}" if mul else f"{mn} {rd}, {rs}, $at"))
        need_at()
        try:
            load = (load_register("$at", v, True) if dbl
                    else self._li(st, "$at", str(raw & M32)))
        except ValueError as exc:
            raise AsmError(st.line, f"{mn}: {exc}", st.text) from None
        if mul:
            return load + [(mul, [rs, "$at"]), ("mflo", [rd])]
        return load + [(mn, [rd, rs, "$at"])]

    def _strict(self, st: _Stmt, real: str) -> None:
        """Strict mode: a GNU as macro is an error naming the real
        instructions."""
        if not self.pseudo:
            raise AsmError(st.line, f"{st.mnemonic} with these operands is a "
                           f"pseudo-instruction (strict mode); write: {real}", st.text)

    def _nops(self, st: _Stmt, count: int) -> None:
        if len(st.ops) != count:
            raise AsmError(st.line, f"{st.mnemonic} takes {count} operands", st.text)

    def _cti(self, mn: str) -> bool:
        e = self._by_name.get(mn)
        return e is not None and (e["family"] in (
            "branch", "branch_fpu", "jump", "branch_compact", "jump_compact")
            or e["name"].startswith("BC2"))

    def _conditional_compact(self, mn: str) -> bool:
        e = self._by_name.get(mn)
        return e is not None and e["family"] == "branch_compact" and \
            e["name"] not in ("BC", "BALC")

    def _has_delay_slot(self, mn: str) -> bool:
        e = self._by_name[mn]
        return e["family"] in ("branch", "jump", "branch_fpu", "msa_branch") or \
            mn.upper().startswith("BC2")

    def _li(self, st: _Stmt, rt: str, expr: str) -> list[tuple[str, list[str]]]:
        """``li`` / ``la``: a constant gets the shortest GNU ``as`` form; an
        address expression always ``lui %hi`` + ``addiu %lo``."""
        if not self._is_constant(expr):
            if self._isa_bit(expr):              # MIPS16e code: bit 0 = the ISA mode
                expr = f"({expr}) + 1"
            return [("lui", [rt, f"%hi({expr})"]),
                    ("addiu", [rt, rt, f"%lo({expr})"])]
        v = self._value(expr, st)
        if not -(1 << 31) <= v <= M32:          # GNU as: "larger than 32 bits"
            raise AsmError(st.line, f"li: number {v:#x} larger than 32 bits" + (
                " (dli loads 64 bits)" if self.wide else ""), st.text)
        v &= M32
        if v < 0x8000 or v >= 0xFFFF8000:
            return [("addiu", [rt, "$zero", str(_sx16(v))])]
        if v <= 0xFFFF:
            return [("ori", [rt, "$zero", str(v)])]
        out = [("lui", [rt, str(v >> 16)])]
        if v & 0xFFFF:
            out.append(("ori", [rt, rt, str(v & 0xFFFF)]))
        return out

    def _dla(self, st: _Stmt, mn: str, rt: str, expr: str) -> list[tuple[str, list[str]]]:
        """MIPS64 ``la`` / ``dla`` / ``dli`` (GNU as, ``-mabi=64 -msym32``):
        a label ``lui %hi`` + ``daddiu %lo``; a constant the 64-bit value
        (``load_register``)."""
        if not self._is_constant(expr):
            if mn == "dli":
                raise AsmError(st.line, "dli takes a constant (a label: dla)", st.text)
            if self._isa_bit(expr):              # microMIPS code: bit 0 = the ISA mode
                expr = f"({expr}) + 1"
            return [("lui", [rt, f"%hi({expr})"]), ("daddiu", [rt, rt, f"%lo({expr})"])]
        try:
            return load_register(rt, self._value(expr, st), True)
        except ValueError as exc:
            raise AsmError(st.line, f"{mn}: {exc}", st.text) from None

    def _mem_needs_macro(self, op: str, st: _Stmt) -> bool:
        m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", op.strip())
        if not m:
            return True                             # label / absolute address
        off = m.group(1).strip()
        if not off or "%lo" in off:
            return False
        if not self._is_constant(off):
            return True
        v = self._value(off, st)
        return not -0x8000 <= v <= 0x7FFF

    #: FPU loads / stores that also accept a label or 32-bit address
    _FP_MEM = {"lwc1", "swc1", "ldc1", "sdc1"}

    #: loads whose rt may serve as the address scratch (rt is overwritten
    #: anyway, and not merged like LWL/LWR) — GNU as does the same
    _RT_SCRATCH = {"lb", "lbu", "lh", "lhu", "lw", "ll", "ld", "lwu", "lld"}

    def _mem_macro(self, mn: str, ops: list[str], st: _Stmt):
        m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", ops[1].strip())
        expr, base = (m.group(1), m.group(2).strip()) if m else (ops[1], None)
        rt = parse_gpr(ops[0])
        tmp = "$at"
        if (mn in self._RT_SCRATCH and rt not in (None, 0)
                and (base is None or parse_gpr(base) != rt)):
            tmp = ops[0]
        elif not st.at_ok:
            raise AsmError(st.line, f"{mn} needs $at, but .set noat", st.text)
        out = [("lui", [tmp, f"%hi({expr})"])]
        if base:                                # (MIPS64: a 64-bit add, GNU as)
            out.append(("daddu" if self.wide else "addu", [tmp, tmp, base]))
        out.append((mn, [ops[0], f"%lo({expr})({tmp})"]))
        return out

    # -- pass 2: emit -----------------------------------------------------------

    def _emit(self, st: _Stmt) -> None:
        if st.mnemonic.startswith("."):
            self._emit_data(st)
            return
        if st.m16:
            self._emit16(st)
            return
        if st.mm:
            self._emitmm(st)
            return
        insns = self._expand(st)
        if st.forbidden_nop:
            insns = insns + [("nop", [])]
        if 4 * len(insns) != st.size:
            raise AsmError(st.line, "expansion size changed between passes "
                           "(use constants defined before use)", st.text)
        for k, (mn, ops) in enumerate(insns):
            addr = st.addr + 4 * k
            word = self._encode(mn, ops, addr, st)
            self._put(addr, word, 4)
            self.prog.source_map[addr] = SourceLine(st.line, st.text)
            self._mark(st.seg, addr, 4)

    # -- MIPS16e ------------------------------------------------------------------

    def _m16_value(self, final: bool):
        from .mips16 import Unresolved

        def value(expr: str) -> int:
            try:
                return _eval(expr, self._lookup)
            except _Unresolved:
                if final:
                    raise
                raise Unresolved(expr) from None
        return value

    def _m16_encode(self, st: _Stmt, final: bool) -> list[list[int]]:
        from . import mips16
        try:
            value = self._m16_value(final)
            if st.mnemonic in ("la", "dla"):        # a MIPS16e label: the ISA bit (GNU as)
                plain = value
                value = lambda e: plain(e) | self._isa_bit(e)     # noqa: E731
            return mips16.encode(st.mnemonic, st.ops, st.addr, value,
                                 reorder=st.reorder, long=st.line in self._m16_long,
                                 wide=self.wide)
        except _Unresolved as exc:
            raise AsmError(st.line, f"undefined symbol {exc}", st.text) from None
        except mips16.AsmError16 as exc:
            raise AsmError(st.line, str(exc), st.text) from None
        except ValueError as exc:
            raise AsmError(st.line, str(exc), st.text) from None

    def _m16_size(self, st: _Stmt) -> int:
        """Pass 1: the bytes of a MIPS16e statement (the short form while a
        label is not known yet, unless relaxation made it long)."""
        return sum(2 * len(i) for i in self._m16_encode(st, final=False))

    def _emit16(self, st: _Stmt) -> None:
        insns = self._m16_encode(st, final=True)
        size = sum(2 * len(i) for i in insns)
        if size != st.size:
            if size > st.size and st.line not in self._m16_long:
                self._m16_long.add(st.line)     # too far for the short form: again
                raise _Relax
            raise AsmError(st.line, "MIPS16e size changed between passes "
                           "(use constants defined before use)", st.text)
        addr = st.addr
        for hws in insns:
            for k, h in enumerate(hws):
                self._put(addr + 2 * k, h, 2)
            self.prog.source_map[addr] = SourceLine(st.line, st.text)
            self.prog.m16[addr] = 2 * len(hws)
            self._mark(st.seg, addr, 2 * len(hws))
            addr += 2 * len(hws)

    # -- microMIPS ----------------------------------------------------------------

    #: written as is (no MIPS32 macro): branches and jumps (the reorder
    #: rules below), the microMIPS-only instructions
    _MM_DIRECT = {"b", "bc", "beqz", "bnez", "beq", "bne", "beqzc", "bnezc", "bgez", "bltz",
                  "blez", "bgtz", "bgezal", "bltzal", "bgezals", "bltzals", "bal", "bals",
                  "bc1f", "bc1t", "bc2f", "bc2t", "j", "jal", "jals", "jalx", "jr", "jr.hb",
                  "jrs", "jrs.hb", "jalr", "jalr.hb", "jalrs", "jalrs.hb", "jrc", "jraddiusp",
                  "lwp", "swp", "lwm", "swm", "lwxs", "movep", "addiupc", "bposge32",
                  "ldp", "sdp", "ldm", "sdm",          # microMIPS64 (MD00594)
                  *(f"{b}.{df}" for b in ("bz", "bnz") for df in "vbhwd"),   # MSA
                  # Release 6
                  "balc", "beqc", "bnec", "bltc", "bgec", "bltuc", "bgeuc", "bovc", "bnvc",
                  "bgtc", "blec", "bgtuc", "bleuc", "bltzc", "blezc", "bgezc", "bgtzc",
                  "beqzalc", "bnezalc", "bltzalc", "blezalc", "bgezalc", "bgtzalc", "bc1eqzc",
                  "bc1nezc", "bc2eqzc", "bc2nezc", "jalrc", "jalrc.hb", "jrc.hb", "jic",
                  "jialc", "jrcaddiusp", "lwpc", "lwupc"}
    #: Release 6 also writes these as is
    _MM_DIRECT_R6 = {"cache"}
    #: delay slots of 32 bits (BD32) / 16 bits (BD16); the other branches
    #: and jumps take either (MD00582 5.4 / 5.5)
    _MM_SLOT32 = {"jal", "jalr", "jalr.hb", "jalx", "bgezal", "bltzal", "bal"}
    _MM_SLOT16 = {"jals", "jalrs", "jalrs.hb", "bgezals", "bltzals", "bals"}
    _MM_SLOT_ANY = {"b", "beqz", "bnez", "beq", "bne", "bgez", "bltz", "blez", "bgtz",
                    "bc1f", "bc1t", "bc2f", "bc2t", "j", "jr", "jr.hb", "jrs", "jrs.hb",
                    "bposge32",
                    # MSA (MD00870): compact — but binutils gives them a delay slot
                    # (CBD): GNU as puts a NOP after them under .set reorder
                    *(f"{b}.{df}" for b in ("bz", "bnz") for df in "vbhwd")}

    def _mm_expand(self, st: _Stmt) -> list[tuple[str, list[str], int]]:
        """(mnemonic, operands, forced size) of a microMIPS statement:
        GNU as's choices — the MIPS32 macros, and with ``.set reorder``
        (eMIPS never moves an instruction into a delay slot) B / BEQZ /
        BNEZ / JR as the compact BC / BEQZC / BNEZC / JRC, the other
        branches and jumps followed by a NOP of their delay slot's size."""
        mn, ops = st.mnemonic, list(st.ops)
        if mn == "li" and len(ops) == 2 and self._is_constant(ops[1]):
            v = self._value(ops[1], st)
            if parse_gpr(ops[0]) in (16, 17, 2, 3, 4, 5, 6, 7) and -1 <= v <= 126 \
                    and st.slot != 4:
                return [("li16", ops, st.slot)]
        if mn in self._MM_DIRECT or self.isa.release == "R6" and mn in self._MM_DIRECT_R6:
            out = [(mn, ops)]
        else:
            reorder, st.reorder = st.reorder, False
            pseudo = self.pseudo
            if mn in ("move", "not"):           # real: MOVE16, NOT16
                self.pseudo = True
            try:
                out = self._expand(st)
            finally:
                st.reorder, self.pseudo = reorder, pseudo
        # GNU as builds its macros from 32-bit forms: an instruction that is
        # not the statement itself (or MOVE / NOT / NEG(U), real here) is 32 bits
        # — but MFHI / MFLO / BREAK, which it builds with their 16-bit formats
        # (tc-mips.c MFHL_FMT "mj", BRK_FMT "mF")
        real = mn in ("move", "not", "neg", "negu") or len(out) == 1 and out[0][0] == mn \
            and out[0][1] in (ops, ops[:1] + ops)
        res = [(m, o, st.slot if k == 0 and (real or st.slot == 4) else
                0 if real or m in ("mfhi", "mflo", "break") else 4)
               for k, (m, o) in enumerate(out)]
        if not real and st.slot == 2:
            raise AsmError(st.line, f"{mn}: a macro in a 16-bit delay slot", st.text)
        last, lops, _ = res[-1]
        if st.reorder and self.isa.release != "R6":        # (Release 6: no delay slots)
            zero = [r for r in lops[:2] if parse_gpr(r) == 0] if last in ("beq", "bne") else []
            if last == "b":
                res[-1] = ("bc", lops, res[-1][2])
            elif last in ("beqz", "bnez"):
                res[-1] = (last + "c", lops, res[-1][2])
            elif zero and len(lops) == 3:
                other = lops[1] if parse_gpr(lops[0]) == 0 else lops[0]
                res[-1] = ("beqzc" if last == "beq" else "bnezc", [other, lops[2]], res[-1][2])
            elif last == "jr":
                res[-1] = ("jrc", lops, res[-1][2])
            elif last in self._MM_SLOT32 or last in self._MM_SLOT16 or \
                    last in self._MM_SLOT_ANY:
                res.append(("nop", [], 4 if last in self._MM_SLOT32 else 2))
        return res

    def _mm_slot_need(self, prev: _Stmt) -> int:
        """The size the delay slot after ``prev`` (noreorder) demands."""
        if self.isa.release == "R6":
            return 0
        last = self._mm_expand(prev)[-1][0]
        return 4 if last in self._MM_SLOT32 else 2 if last in self._MM_SLOT16 else 0

    def _mm_label_dep(self, ops: list[str]) -> bool:
        """An operand that uses a label (not a branch target): GNU as
        leaves a relocation there, so the 32-bit form."""
        for op in ops:
            names = [t for t in _TOKEN.findall(op)
                     if re.fullmatch(r"[A-Za-z_.][\w.$]*", t)]
            if any(t not in self._consts for t in names):
                return True
        return False

    def _mm_encode(self, st: _Stmt, final: bool) -> list:
        from . import micromips as MM
        if self._mm_enc is None or self._mm_enc.release != self.isa.release:
            self._mm_enc = MM.Encoder(self.isa.release, self.isa.arch)
        enc = self._mm_enc
        value = self._cvalue(final, MM.Unresolved)
        out, addr = [], st.addr
        long = st.line in self._m16_long
        for k, (mn, ops, force) in enumerate(self._mm_expand(st)):
            branch = mn in self._MM_DIRECT and mn not in (
                "lwp", "swp", "lwm", "swm", "lwxs", "movep", "ldp", "sdp", "ldm", "sdm") \
                or mn in ("beq", "bne")
            try:
                if mn == "li16":
                    insn = enc._raw("LI16", d=MM.GPR3.index(parse_gpr(ops[0])),
                                    i=self._value(ops[1], st) & 0x7F)
                elif not branch and self._mm_label_dep(ops):
                    if not final:
                        insn = MM.Insn(4, 0)
                    else:
                        insn = enc.encode(mn, ops, addr, value, self._mm_mips32(addr, st),
                                          force=4)
                        if force == 2:
                            raise MM.AsmErrorMM(f"{mn}: the delay slot needs a 16-bit "
                                                "instruction")
                else:
                    insn = enc.encode(mn, ops, addr, value, self._mm_mips32(addr, st),
                                      force=force, long=long)
            except _Unresolved as exc:
                raise AsmError(st.line, f"undefined symbol {exc}", st.text) from None
            except MM.AsmErrorMM as exc:
                raise AsmError(st.line, str(exc), st.text) from None
            except ValueError as exc:
                raise AsmError(st.line, str(exc), st.text) from None
            out.append(insn)
            addr += insn.size
        return out

    def _mm_mips32(self, addr: int, st: _Stmt):
        return lambda mn, ops: self._encode(self._dsp_form(mn, ops), ops, addr, st)

    def _dsp_form(self, mn: str, ops: list[str]) -> str:
        """``mult $ac1, ...`` / ``mfhi $t0, $ac1``: the DSP Module's
        accumulator row (``mult@dsp``), else ``mn``."""
        dsp = mn + "@dsp"
        if dsp in self._by_name and (any(_AC.fullmatch(o.strip()) for o in ops)
                                     or len(ops) == len(operand_spec(self._by_name[dsp]))):
            return dsp
        return mn

    def _cvalue(self, final: bool, unresolved):
        def value(expr: str) -> int:
            try:
                return _eval(expr, self._lookup)
            except _Unresolved:
                if final:
                    raise
                raise unresolved(expr) from None
        return value

    def _emitmm(self, st: _Stmt) -> None:
        insns = self._mm_encode(st, final=True)
        size = sum(i.size for i in insns)
        if size != st.size:
            if size > st.size and st.line not in self._m16_long:
                self._m16_long.add(st.line)     # a branch too far for the 16-bit form
                raise _Relax
            raise AsmError(st.line, "microMIPS size changed between passes "
                           "(use constants defined before use)", st.text)
        addr = st.addr
        for insn in insns:
            for k, h in enumerate(insn.halves):
                self._put(addr + 2 * k, h, 2)
            self.prog.source_map[addr] = SourceLine(st.line, st.text)
            self.prog.mm[addr] = insn.size
            self._mark(st.seg, addr, insn.size)
            addr += insn.size

    def _isa_bit(self, expr: str) -> int:
        """1 for the address of MIPS16e code taken as data (``.word f`` /
        ``la`` / ``li``): what the linker does for a MIPS16 symbol, so a
        JALR / JR to it switches the ISA mode."""
        return 1 if expr.strip() in self._m16_labels else 0

    def _put(self, addr: int, value: int, size: int) -> None:
        data = value.to_bytes(size, "big" if self.endian == "big" else "little")
        for i, b in enumerate(data):
            self.prog.image[(addr + i) & M32] = b

    def _mark(self, seg: str, addr: int, size: int) -> None:
        lo, hi = self.prog.segments.get(seg, (addr, addr))
        self.prog.segments[seg] = (min(lo, addr), max(hi, addr + size))

    def _emit_data(self, st: _Stmt) -> None:
        d, addr = st.mnemonic, st.addr
        if d == ".fill16":                      # MIPS16e / microMIPS alignment: NOPs
            from .micromips import NOP16
            from .mips16 import NOP
            for k in range(0, st.size, 2):
                self._put(addr + k, NOP16 if st.mm else NOP, 2)
        elif d == ".space":
            for i in range(st.size):
                self.prog.image[addr + i] = 0
        elif d in (".word", ".half", ".byte"):
            size = {".word": 4, ".half": 2, ".byte": 1}[d]
            for i, op in enumerate(st.ops):
                v = self._value(op, st) | (self._isa_bit(op) if size == 4 else 0)
                if not -(1 << (8 * size - 1)) <= v < (1 << (8 * size)):
                    raise AsmError(st.line, f"value {v} does not fit {d}", st.text)
                self._put(addr + i * size, v & ((1 << (8 * size)) - 1), size)
        elif d in (".dword", ".8byte"):          # (GNU as) 8 bytes each
            for i, op in enumerate(st.ops):
                v = self._value(op, st)
                if self.wide and not self._is_constant(op):
                    v = _sx32(v)                # an address, as a MIPS64 pointer
                if not -(1 << 63) <= v <= M64:
                    raise AsmError(st.line, f"value {v} does not fit {d}", st.text)
                self._put(addr + i * 8, v & M64, 8)
        elif d in (".float", ".double"):
            fmt = ("f" if d == ".float" else "d")
            fmt = (">" if self.endian == "big" else "<") + fmt
            size = struct.calcsize(fmt)
            for i, op in enumerate(st.ops):
                for j, b in enumerate(struct.pack(fmt, float(op))):
                    self.prog.image[addr + i * size + j] = b
        elif d in (".ascii", ".asciiz"):
            data = self._strings(_strip_comment(st.text).split(d, 1)[1], st.line,
                                 st.text, d == ".asciiz")
            for i, b in enumerate(data):
                self.prog.image[addr + i] = b
        self._mark(st.seg, addr, st.size)

    # -- encoding one real instruction -------------------------------------------

    def _encode(self, mn: str, ops: list[str], addr: int, st: _Stmt) -> int:
        e = self._by_name[mn]
        spec = operand_spec(e)
        word = e["decode"]["value"]
        name = e["name"]

        def err(msg: str) -> AsmError:
            return AsmError(st.line, f"{mn.split('@')[0]}: {msg}", st.text)

        if e.get("ase") == "mt":                # the MT Module (MD00378)
            if not st.mt:
                raise err("an MT Module instruction, after .set nomt")
            self.prog.mt = True
        elif e.get("ase") == "msa":             # MSA (MD00866)
            if not st.msa:
                raise err("an MSA instruction, after .set nomsa")
            self.prog.msa = True
        elif e.get("ase"):                      # the DSP Module (MD00374)
            level = ISA.DSP_LEVEL[e["ase"]]
            if level > st.dsp:
                raise err("a DSP Module Revision 2 instruction (.set dsp allows "
                          "Revision 1 only)" if st.dsp else
                          "a DSP Module instruction, after .set nodsp")
            self.prog.dsp = max(self.prog.dsp, level)

        if name in _DFIELD:                     # MIPS64 DEXT ... DINSU
            if len(ops) != 4:
                raise err("expects rt, rs, pos, size")
            pos, size = self._value(ops[2], st), self._value(ops[3], st)
            row = name if mn in ("dextm", "dextu", "dinsm", "dinsu") else \
                _dfield_row(name[:4], pos, size)
            if row is None or not _DFIELD[row][0](pos, size):
                raise err(f"bad pos {pos} / size {size}")
            lsb, high = _DFIELD[row][1](pos, size)
            return (self.isa.get(row)["decode"]["value"]
                    | self._operand("rt", ops[0], addr, st, err)
                    | self._operand("rs", ops[1], addr, st, err)
                    | (high << 11) | (lsb << 6))
        if name in ("DAHI", "DATI"):            # MIPS64 R6: "DAHI rs, rs, immediate"
            if len(ops) != 3 or parse_gpr(ops[0]) != parse_gpr(ops[1]):
                raise err("expects rs, rs, immediate (the same register twice)")
            return (word | self._operand("rs", ops[0], addr, st, err)
                    | self._operand("uimm", ops[2], addr, st, err))
        if name in ("EXT", "INS"):
            if len(ops) != 4:
                raise err("expects rt, rs, pos, size")
            pos, size = self._value(ops[2], st), self._value(ops[3], st)
            if not (0 <= pos <= 31 and 1 <= size <= 32 and pos + size <= 32):
                raise err(f"bad pos {pos} / size {size}")
            hi = size - 1 if name == "EXT" else pos + size - 1
            return (word | self._operand("rt", ops[0], addr, st, err)
                    | self._operand("rs", ops[1], addr, st, err)
                    | (hi << 11) | (pos << 6))
        if name == "BREAK":
            if len(ops) > 2:
                raise err("takes at most 2 codes")
            codes = [self._value(o, st) for o in ops] + [0, 0]
            for c in codes[:2]:
                if not 0 <= c < 1024:
                    raise err("code must be 0..1023")
            return word | (codes[0] << 16) | (codes[1] << 6)

        items = [kind_of(s) for s in spec]
        n_req = sum(1 for _, opt in items if not opt)
        n_opt = len(items) - n_req
        extra = len(ops) - n_req
        if not 0 <= extra <= n_opt:
            want = f"{n_req}" + (f"..{n_req + n_opt}" if n_opt else "")
            raise err(f"expects {want} operands, got {len(ops)}")
        it = iter(ops)
        for kind, optional in items:
            if optional:
                if extra <= 0:
                    word |= self._default(kind, name)
                    continue
                extra -= 1
            word |= self._operand(kind, next(it), addr, st, err)
        if name in ("CLO", "CLZ", "DCLO", "DCLZ"):     # rt = rd
            word |= ((word >> 11) & 31) << 16
        word = self._r6_fields(name, word, err)
        if name in ("BGEZAL", "BLTZAL", "BGEZALL", "BLTZALL") and (word >> 21) & 31 == 31:
            raise err("rs must not be $ra (the link register)")
        if name in _ZALC and (word >> 16) & 31 == 31:   # R6: UNPREDICTABLE (GNU as: error)
            raise err("rt must not be $ra (the link register)")
        if name.startswith("JALR") and (word >> 11) & 31 == (word >> 21) & 31:
            raise err("rd and rs must be different")
        return word

    @staticmethod
    def _r6_fields(name: str, word: int, err) -> int:
        """Release 6 compact branches whose encodings differ only by how
        rs compares with rt: fill / order the fields as GNU as does, and
        reject operands the encoding cannot express."""
        rs, rt = (word >> 21) & 31, (word >> 16) & 31
        if name in ("BGEZALC", "BLTZALC", "BGEZC", "BLTZC"):
            word |= rt << 21                        # register in rs and rt
        elif name in ("BEQC", "BNEC") and rs > rt or \
                name in ("BOVC", "BNVC") and rs < rt:
            word = (word & ~(0x3FF << 16)) | (rt << 21) | (rs << 16)   # swap
        from .isa import _constraint_ok
        # (the MIPS64 table has every row: MIPS32's and the doubleword ones)
        entry = ISA.for_release("R6", arch="mips64")._by_name.get(name) \
            if "@" not in name else None
        if entry is not None and not _constraint_ok(entry, word):
            raise err("these registers cannot be encoded (e.g. $zero, or "
                      "the same register twice)")
        return word

    @staticmethod
    def _default(kind: str, name: str) -> int:
        if kind == "rd" and name.startswith("JALR"):
            return 31 << 11
        return DEFAULTS.get(kind, 0) << KINDS[kind].fields[0][0]

    def _operand(self, kind: str, tok: str, addr: int, st: _Stmt, err) -> int:
        def reg(parse, what: str) -> int:
            r = parse(tok)
            if r is None:
                raise err(f"{what} expected, got {tok!r}")
            return r

        def num(lo: int, hi: int, what: str) -> int:
            v = self._value(tok, st)
            if not lo <= v <= hi:
                raise err(f"{what} {v} out of range {lo}..{hi}")
            return v

        lsb = KINDS[kind].fields[0][0]
        if kind in ("rs", "rt", "rd"):
            return reg(parse_gpr, "GPR") << lsb
        if kind in ("fs", "ft", "fd", "fr"):
            return reg(parse_fpr, "FPR") << lsb
        if kind in ("c0rd", "fcr", "hwr", "c2rt", "trt", "trd"):
            r = parse_numreg(tok)
            if r is None and kind in ("trt", "trd"):
                r = parse_gpr(tok)              # (binutils: a GPR operand, any name)
            if r is None:
                r = num(0, 31, "register number")
            return r << lsb
        if kind in ("u1", "h1"):
            return num(0, 1, kind[0]) << lsb
        if kind == "rx":
            return num(0, 31, "rx") << lsb
        if kind in ("sa", "hint", "hint11", "stype"):
            return num(0, 31, kind) << lsb
        if kind == "sel":
            return num(0, 7, "select")
        if kind == "sel6":
            return num(0, 7, "select") << 6
        if kind == "rsd":
            return reg(parse_gpr, "GPR") << 21
        if kind in ("br21", "br26"):
            bits = 21 if kind == "br21" else 26
            target = self._value(tok, st) & M32
            delta = _sx32(target - (addr + 4))
            if delta & 3:
                raise err(f"branch target 0x{target:08x} not word-aligned")
            if not -(1 << (bits + 1)) <= delta < (1 << (bits + 1)):
                raise err(f"branch target 0x{target:08x} out of range")
            return (delta >> 2) & ((1 << bits) - 1)
        if kind == "pc19":                  # an address, PC-relative (GNU as)
            delta = _sx32((self._value(tok, st) & M32) - addr)
            if delta & 3 or not -(1 << 20) <= delta < (1 << 20):
                raise err(f"address {tok} not reachable PC-relative (+-1 MB, x4)")
            return (delta >> 2) & 0x7FFFF
        if kind == "pc18":                  # LDPC: (PC & ~7) + offset * 8 (MIPS64 R6)
            delta = _sx32((self._value(tok, st) & M32) - (addr & ~7))
            if delta & 7 or not -(1 << 20) <= delta < (1 << 20):
                raise err(f"address {tok} not reachable PC-relative (+-1 MB, x8)")
            return (delta >> 3) & 0x3FFFF
        if kind == "sa2":
            return (num(1, 4, "shift") - 1) << 6
        if kind == "bp":
            return num(0, 3, "byte position") << 6
        if kind == "bp3":
            return num(0, 7, "byte position") << 6
        if kind == "code16":
            return num(0, 0xFFFF, "code")
        if kind == "mem11":
            m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", tok.strip())
            if not m or parse_gpr(m.group(2)) is None:
                raise err(f"offset(base) expected, got {tok!r}")
            off = self._value(m.group(1), st) if m.group(1).strip() else 0
            if not -1024 <= off <= 1023:
                raise err(f"11-bit offset {off} out of range")
            return (off & 0x7FF) | (parse_gpr(m.group(2)) << 11)
        if kind == "c2cc":
            m = re.fullmatch(r"\$cc([0-7])", tok.strip())
            return (int(m.group(1)) if m else num(0, 7, "COP2 condition")) << lsb
        if kind == "tcode":
            return num(0, 1023, "code") << 6
        if kind == "code20":
            return num(0, (1 << 20) - 1, "code") << 6
        if kind == "code19":
            return num(0, (1 << 19) - 1, "code") << 6
        if kind == "imm":
            return num(-0x8000, 0x7FFF, "immediate") & 0xFFFF
        if kind == "uimm":
            return num(0, 0xFFFF, "immediate")
        if kind in ("impl16",):
            return num(0, 0xFFFF, "impl")
        if kind == "cofun":
            return num(0, (1 << 25) - 1, "cofun")
        if kind in ("cc18", "cc8"):
            cc = parse_fcc(tok)
            if cc is None:
                raise err(f"FPU condition code expected, got {tok!r}")
            return cc << lsb
        if kind in ("mem", "mem9", "idx"):
            m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", tok.strip())
            if not m:
                raise err(f"offset(base) expected, got {tok!r}")
            base = parse_gpr(m.group(2))
            if base is None:
                raise err(f"base register expected, got {m.group(2)!r}")
            if kind == "idx":
                index = parse_gpr(m.group(1))
                if index is None:
                    raise err(f"index register expected, got {m.group(1)!r}")
                return (index << 16) | (base << 21)
            off = self._value(m.group(1), st) if m.group(1).strip() else 0
            if kind == "mem":
                if not -0x8000 <= off <= 0x7FFF:
                    raise err(f"offset {off} out of range")
                return (off & 0xFFFF) | (base << 21)
            if not -256 <= off <= 255:
                raise err(f"9-bit offset {off} out of range")
            return ((off & 0x1FF) << 7) | (base << 21)
        if kind == "br":
            target = self._value(tok, st) & M32
            delta = _sx32(target - (addr + 4))
            if delta & 3:
                raise err(f"branch target 0x{target:08x} not word-aligned")
            if not -0x20000 <= delta <= 0x1FFFC:
                raise err(f"branch target 0x{target:08x} out of range")
            return (delta >> 2) & 0xFFFF
        if kind in ("ac", "ac21"):
            m = _AC.fullmatch(tok.strip())
            if not m:
                raise err(f"accumulator $ac0..$ac3 expected, got {tok!r}")
            return int(m.group(1)) << lsb
        if kind in ("dsa3", "dsa4", "dsa5", "dsize", "dsa", "bp2", "rmask", "wmask",
                    "dimm8"):
            width = KINDS[kind].fields[0][1]
            return num(0, (1 << width) - 1, {"dsize": "size", "bp2": "byte position",
                                             "rmask": "mask", "wmask": "mask",
                                             "dimm8": "immediate"}.get(kind, "shift")) << lsb
        if kind == "dimm10":
            return (num(-512, 511, "immediate") & 0x3FF) << lsb
        if kind == "shilo":
            return (num(-32, 31, "shift") & 0x3F) << lsb
        if kind in _MSA_KINDS:                  # MSA (MD00866 3.6)
            return self._msa_operand(kind, tok, st, err)
        if kind == "jt":
            target = self._value(tok, st) & M32
            if target & 3:
                raise err(f"jump target 0x{target:08x} not word-aligned")
            if (target ^ (addr + 4)) & 0xF0000000:
                raise err(f"jump target 0x{target:08x} outside the 256 MB region")
            return (target >> 2) & 0x3FFFFFF
        raise err(f"internal: unknown operand kind {kind}")


    def _msa_operand(self, kind: str, tok: str, st: _Stmt, err) -> int:
        """An MSA operand (MD00866 3.6, GNU as's syntax): ``$w5``, ``$w5[3]``,
        ``$w5[$t0]``, ``-16($a0)`` (a byte offset, a multiple of the element
        size), the control registers as ``$1`` (objdump's ``msa_csr`` names
        too), the immediates."""
        t = tok.strip()
        fields = KINDS[kind].fields
        lsb = fields[0][0]

        def wreg(text: str) -> int:
            m = _WREG.fullmatch(text.strip())
            if not m:
                raise err(f"MSA register $w0..$w31 expected, got {text.strip()!r}")
            return int(m.group(1))

        def num(text: str, lo: int, hi: int, what: str) -> int:
            v = self._value(text, st)
            if not lo <= v <= hi:
                raise err(f"{what} {v} out of range {lo}..{hi}")
            return v

        if kind in ("wd", "ws", "wt"):
            return wreg(t) << lsb
        if kind in ("rd6", "rs11"):
            r = parse_gpr(t)
            if r is None:
                raise err(f"GPR expected, got {t!r}")
            return r << lsb
        if kind in ("mcs", "mcd"):
            r = parse_numreg(t)
            if r is None and t.lower().lstrip("$") in MSA_CONTROL:
                r = MSA_CONTROL.index(t.lower().lstrip("$"))
            if r is None:
                r = num(t, 0, 31, "MSA control register")
            return r << lsb
        if kind == "u5":
            return num(t, 0, 31, "immediate") << lsb
        if kind == "s5":
            return (num(t, -16, 15, "immediate") & 31) << lsb
        if kind in ("m3", "m4", "m5", "m6", "u8"):
            width = fields[0][1]
            return num(t, 0, (1 << width) - 1, "immediate") << lsb
        if kind == "s10":
            return (num(t, -512, 511, "immediate") & 0x3FF) << lsb
        if kind.startswith("vmem_"):
            m = re.fullmatch(r"(.*)\((\s*\$[\w]+\s*)\)", t)
            base = parse_gpr(m.group(2)) if m else None
            if base is None:
                raise err(f"offset(base) expected, got {t!r}")
            off = self._value(m.group(1), st) if m.group(1).strip() else 0
            scale = {"b": 1, "h": 2, "w": 4, "d": 8}[kind[-1]]
            if off % scale:
                raise err(f"offset {off} is not a multiple of {scale} (the element size)")
            if not -512 <= off // scale <= 511:
                raise err(f"offset {off} out of range {-512 * scale}..{511 * scale}")
            return ((off // scale) & 0x3FF) << 16 | base << 11
        m = re.fullmatch(r"(.*?)\[(.*)\]", t)
        if not m:
            raise err(f"an element $wN[n] expected, got {t!r}")
        reg, index = wreg(m.group(1)), m.group(2).strip()
        if kind == "ws0":
            if self._value(index, st) != 0:
                raise err(f"the source element must be [0], got [{index}]")
            return reg << lsb
        if kind == "elmr":
            r = parse_gpr(index)
            if r is None:
                raise err(f"a GPR index $wN[$rt] expected, got [{index}]")
            return reg << lsb | r << 16
        width = fields[1][1]
        return reg << lsb | num(index, 0, (1 << width) - 1, "element index") << 16


#: the operand kinds of MSA's rows (emips.syntax.msa_spec)
_MSA_KINDS = frozenset({"wd", "ws", "wt", "rd6", "rs11", "mcs", "mcd", "u5", "s5", "m3",
                        "m4", "m5", "m6", "u8", "s10", "ws0", "elmr",
                        *(f"vmem_{d}" for d in "bhwd"), *(f"elm{k}" for k in (1, 2, 3, 4)),
                        *(f"elmd{k}" for k in (1, 2, 3, 4))})
_WREG = re.compile(r"\$w([12]?[0-9]|3[01])", re.I)
#: MSA's control registers as objdump names them (MD00866 3.4: 0-7)
MSA_CONTROL = ("msa_ir", "msa_csr", "msa_access", "msa_save", "msa_modify",
               "msa_request", "msa_map", "msa_unmap")


#: MIPS64 (PLAN M22, MIPS64 Vol. II DEXT ... DINSU): row -> (pos / size
#: in its range, (pos, size) -> (lsb, msbd or msb))
_DFIELD = {
    "DEXT": (lambda p, n: 0 <= p < 32 and 1 <= n <= 32, lambda p, n: (p, n - 1)),
    "DEXTM": (lambda p, n: 0 <= p < 32 and 33 <= n <= 64 and p + n <= 64,
              lambda p, n: (p, n - 33)),
    "DEXTU": (lambda p, n: 32 <= p < 64 and 1 <= n <= 32 and p + n <= 64,
              lambda p, n: (p - 32, n - 1)),
    "DINS": (lambda p, n: 0 <= p < 32 and 1 <= n <= 32 and p + n <= 32,
             lambda p, n: (p, p + n - 1)),
    "DINSM": (lambda p, n: 0 <= p < 32 and 2 <= n <= 64 and 32 < p + n <= 64,
              lambda p, n: (p, p + n - 33)),
    "DINSU": (lambda p, n: 32 <= p < 64 and 1 <= n <= 32 and p + n <= 64,
              lambda p, n: (p - 32, p + n - 33)),
}


def _dfield_row(base: str, pos: int, size: int) -> str | None:
    """The DEXT / DINS encoding for ``pos`` / ``size`` (as GNU as chooses:
    the manual's "the assembler will ... emit DEXT, DEXTM, or DEXTU as
    appropriate"); None when no form takes them."""
    for row in (base, base + "M", base + "U"):
        if _DFIELD[row][0](pos, size):
            return row
    return None


def _mips64_mnemonics() -> frozenset[str]:
    """The doubleword mnemonics (and macros): a hint in a MIPS32 program."""
    global _M64_MNEMONICS
    if _M64_MNEMONICS is None:
        isa = ISA.for_release("R5", arch="mips64")
        _M64_MNEMONICS = frozenset(
            {e.get("mnemonic", e["name"]).lower() for e in isa.instructions
             if e.get("arch") == "mips64"} | {"dli", "dla", "dneg", "dnegu", "dmul"}
            | {e.get("mnemonic", e["name"]).lower()
               for e in ISA.for_release("R6", arch="mips64").instructions
               if e.get("arch") == "mips64"})
    return _M64_MNEMONICS


_M64_MNEMONICS: frozenset[str] | None = None


def assemble(source: str, *, endian: str = "big", isa: ISA | None = None,
             bases: dict[str, int] | None = None,
             release: str | None = None, pseudo: bool = True,
             reorder: bool = True, arch: str = "mips32",
             nano: dict | None = None, rv: dict | None = None, path: str | None = None,
             include_dirs: tuple[str, ...] = ()) -> Program:
    """Assemble ``source`` for a MIPS32 ``release`` (default R5; ``arch``
    "mips64": MIPS64; "nanomips": :mod:`emips.nanoasm`, GNU's nanoMIPS
    assembler and linker, Release 6 only); raises :class:`AsmError` on the
    first error.  ``pseudo=False``: strict mode; ``reorder=False``: start
    in ``.set noreorder``; ``nano``: :func:`emips.nanoasm.assemble`'s
    options (the core's ``nms`` / ``crc`` ..., ``linkrelax``, ``link``);
    ``arch`` "rv32" / "rv64": :mod:`emips.rvasm` (GNU as + ld for RISC-V,
    little-endian), ``rv`` its ``extensions`` / ``relax``.  ``path``: the
    source's file (``.include`` / ``.incbin`` look in its folder, then in
    ``include_dirs``: emips.gasread)."""
    if arch in ("rv32", "rv64"):
        from . import rvasm
        rv = rv or {}
        relax = rv.get("relax", True)
        return rvasm.assemble(source, arch=arch,
                              extensions=rv.get("extensions", rvasm.DEFAULT_EXTENSIONS),
                              relax=relax, link_relax=relax, bases=bases, strict=not pseudo,
                              path=path, include_dirs=include_dirs)
    if arch == "nanomips":
        if release not in (None, "R6"):
            raise ValueError(f"nanoMIPS is Release 6 based: no {release}")
        from . import nanoasm
        return nanoasm.assemble(source, endian=endian, bases=bases, pseudo=pseudo,
                                reorder=reorder, path=path, include_dirs=include_dirs,
                                **(nano or {}))
    return Assembler(endian=endian, isa=isa, bases=bases, release=release,
                     pseudo=pseudo, reorder=reorder, arch=arch).assemble(
                         source, path=path, include_dirs=include_dirs)
