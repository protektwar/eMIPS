"""The Galileo GT-64120 system controller of the Malta's CoreLV card
(PLAN M15 phase 1).

References: *GT-64120 System Controller for R4xxx/R5xxx/R7000*, Product
Preview rev. 0.2 (Galileo, 1997): chapter 4 (address space decoding: the CPU resource
groups and the device sub-decoders, 4.5 decoding errors, Table 12 the
reset map), chapter 11 (interrupts), chapter 18 (the registers); the
CoreLV User's Manual (MD00007 rev. 02.08: 3.2 the programming notes, 3.5
the interrupt wiring); QEMU's ``hw/pci-host/gt64120.c`` for the reset
values the datasheet leaves to strapping.

The CPU address goes through two decode stages (4.1.1): a **resource
group** — SCS[1:0], SCS[3:2] (SDRAM), CS[2:0], CS[3] & BootCS (the
device bus), PCI_0 I/O, PCI_0 Memory 0 / 1, the internal registers — is
hit when address bits 35:28 equal the Low register's bits 14:7 and bits
27:21 lie between the Low register's bits 6:0 and the High register's;
then a **device sub-decoder** (bits 27:20 between its Low and High
registers) picks the SDRAM bank or device chip select.  An address no
group decodes is a *CPU out of range* error (Bus Error Address register,
Cause.CPUOut), one a group decodes but no sub-decoder is a *memory out of
range* error (Address Decode Error register, Cause.MemOut); a read gets a
bus error, a write only the interrupt (4.5).

The internal registers are little-endian (Linux reads them with
``le32_to_cpu``; the CoreLV manual calls the byte swap a big-endian CPU
sees a GT-64120 bug): a big-endian program swaps what it writes.

Data between the CPU and PCI, and between PCI and the SDRAM, keeps its
byte addresses while the CPU Configuration Endianess bit (12) and the
PCI_0 Command ByteSwap bit (0) agree — their reset values (3.5, Tables 6
/ 42, 14.1); when they differ the byte lanes of every 32-bit word are
reversed (a swapped BAR in PCI function 1 reverses them for its window).
Configuration data of the *other* devices on PCI keeps its value
(Table 6: a big-endian CPU reads a device's IDs unswapped; Linux reads
them raw), the GT-64120's own configuration header is little-endian
like its registers (Linux: ``GT_READ`` for slot 0).

As a PCI target (7.3.5, 5.2) it decodes in the same two stages as for
the CPU: a Base Address Register of its configuration header (BAR0
SCS[1:0], BAR1 SCS[3:2], BAR2 CS[2:0], BAR3 CS[3] / BootCS, BAR4 the
internal registers; the swapped BARs of function 1) with its Bank Size
register (0xC08-0xC14) selects the group — enabled in 0xC3C, with Memory
Enable in the header's Command — the address is remapped (0xC48-0xC64),
then bits 27:20 go to the group's device sub-decoders; a BAR hit no
sub-decoder takes is retried forever, no BAR hit is a master abort for
the PCI master.
"""

from __future__ import annotations

import bisect
from typing import Callable

M32 = 0xFFFFFFFF

# -- registers (chapter 18) ----------------------------------------------------
CPU_CONFIG = 0x000
SCS10LD, SCS10HD, SCS32LD, SCS32HD = 0x008, 0x010, 0x018, 0x020
CS20LD, CS20HD, CS3BOOTLD, CS3BOOTHD = 0x028, 0x030, 0x038, 0x040
PCI0IOLD, PCI0IOHD, PCI0M0LD, PCI0M0HD = 0x048, 0x050, 0x058, 0x060
ISD = 0x068
BUSERR_LO, BUSERR_HI = 0x070, 0x078
PCI0M1LD, PCI0M1HD = 0x080, 0x088
PCI1IOLD, PCI1IOHD, PCI1M0LD, PCI1M0HD, PCI1M1LD, PCI1M1HD = \
    0x090, 0x098, 0x0A0, 0x0A8, 0x0B0, 0x0B8
MULTI = 0x120
SCS0LD, SCS0HD, SCS1LD, SCS1HD = 0x400, 0x404, 0x408, 0x40C
SCS2LD, SCS2HD, SCS3LD, SCS3HD = 0x410, 0x414, 0x418, 0x41C
CS0LD, CS0HD, CS1LD, CS1HD = 0x420, 0x424, 0x428, 0x42C
CS2LD, CS2HD, CS3LD, CS3HD = 0x430, 0x434, 0x438, 0x43C
BOOTLD, BOOTHD = 0x440, 0x444
SDRAM_CFG = 0x448
ADERR = 0x470
INT_CAUSE, CPU_INT_MASK, PCI0_INT_MASK = 0xC18, 0xC1C, 0xC24
PCI0_IACK = 0xC34
PCI0_CFGADDR, PCI0_CFGDATA = 0xCF8, 0xCFC
#: PCI_0 Command (0xC00): bit 0 ByteSwap (0: swap), reset = CPU Config bit 12
PCI0_CMD = 0xC00
#: PCI_0 Bank Size registers (7.6.1): RAS[1:0], RAS[3:2], CS[2:0], CS[3] & BootCS
PCI0_BS = (0xC08, 0xC0C, 0xC10, 0xC14)
#: PCI_0 Base Address Registers' Enable (0xC3C; a 1 disables): bit 0 swapped
#: CS3/Boot, 1 swapped RAS[3:2], 2 swapped RAS[1:0], 3 internal I/O,
#: 4 internal memory, 5 CS[3]/Boot, 6 CS[2:0], 7 RAS[3:2], 8 RAS[1:0]
PCI0_BAR_EN = 0xC3C
#: the PCI remap registers: RAS[1:0], RAS[3:2], CS[2:0], CS3/Boot, then the
#: swapped RAS[1:0], RAS[3:2], CS3/Boot
PCI0_REMAP = (0xC48, 0xC4C, 0xC50, 0xC54, 0xC58, 0xC5C, 0xC64)
#: the target BARs: (function, header offset, group, bank size register,
#: remap register, enable bit, swapped)
PCI_BARS = ((0, 0x10, "scs10", 0xC08, 0xC48, 8, False),
            (0, 0x14, "scs32", 0xC0C, 0xC4C, 7, False),
            (0, 0x18, "cs20", 0xC10, 0xC50, 6, False),
            (0, 0x1C, "cs3boot", 0xC14, 0xC54, 5, False),
            (1, 0x10, "scs10", 0xC08, 0xC58, 2, True),
            (1, 0x14, "scs32", 0xC0C, 0xC5C, 1, True),
            (1, 0x1C, "cs3boot", 0xC14, 0xC64, 0, True))
#: PCI_0 Configuration Address (0xCF8): ConfigEn, bus 23:16, device 15:11,
#: function 10:8, register 7:2
CFG_EN = 1 << 31

#: CPU Interface Configuration (0x000): Endianess, sampled at reset
CPU_ENDIAN_LITTLE = 1 << 12

# Interrupt Cause (0xC18) bits (Table 18.15)
INT_SUM, MEM_OUT, DMA_OUT, CPU_OUT = 1 << 0, 1 << 1, 1 << 2, 1 << 3
MAS_ABORT0, TAR_ABORT0 = 1 << 18, 1 << 19
CPU_INT_BITS = 0x1F << 21           # set by the CPU writing 0 (to PCI)
PCI_INT_BITS = 0xF << 26            # set by PCI, cleared by the CPU writing 0
CPU_INT_SUM, PCI_INT_SUM = 1 << 30, 1 << 31
#: bits that reach Interrupt* (CPUIntSum): 29:26 and 20:1
TO_CPU = PCI_INT_BITS | 0x1FFFFE
#: bits that reach Int0* (PCIIntSum): 25:1
TO_PCI = 0x3FFFFFE

#: the CPU decode groups: name -> (Low register, High register)
GROUPS = {"scs10": (SCS10LD, SCS10HD), "scs32": (SCS32LD, SCS32HD),
          "cs20": (CS20LD, CS20HD), "cs3boot": (CS3BOOTLD, CS3BOOTHD),
          "pci0io": (PCI0IOLD, PCI0IOHD), "pci0m0": (PCI0M0LD, PCI0M0HD),
          "pci0m1": (PCI0M1LD, PCI0M1HD)}
#: each group's device sub-decoders: name -> (Low register, High register)
SUBDECODERS = {"scs10": {"scs0": (SCS0LD, SCS0HD), "scs1": (SCS1LD, SCS1HD)},
               "scs32": {"scs2": (SCS2LD, SCS2HD), "scs3": (SCS3LD, SCS3HD)},
               "cs20": {"cs0": (CS0LD, CS0HD), "cs1": (CS1LD, CS1HD), "cs2": (CS2LD, CS2HD)},
               "cs3boot": {"cs3": (CS3LD, CS3HD), "boot": (BOOTLD, BOOTHD)}}
#: the internal register space (Table 13: 4 KB)
REGS_SIZE = 0x1000

#: reset values (the datasheet's chapter 18, Table 12; strapped values as
#: QEMU's model has them: one GT, PCI_0 only)
RESET = {
    MULTI: 0x00000003,
    SCS10LD: 0x000, SCS10HD: 0x07, SCS32LD: 0x008, SCS32HD: 0x0F,
    CS20LD: 0x0E0, CS20HD: 0x70, CS3BOOTLD: 0x0F8, CS3BOOTHD: 0x7F,
    PCI0IOLD: 0x080, PCI0IOHD: 0x0F, PCI0M0LD: 0x090, PCI0M0HD: 0x1F,
    ISD: 0x0A0, PCI0M1LD: 0x790, PCI0M1HD: 0x1F,
    PCI1IOLD: 0x100, PCI1IOHD: 0x0F, PCI1M0LD: 0x110, PCI1M0HD: 0x1F,
    PCI1M1LD: 0x120, PCI1M1HD: 0x2F,
    0x0D0: 0x000, 0x0D8: 0x008, 0x0E0: 0x0E0, 0x0E8: 0x0F8,      # remaps
    0x0F0: 0x080, 0x0F8: 0x090, 0x100: 0x790, 0x108: 0x100, 0x110: 0x110, 0x118: 0x120,
    SCS0LD: 0x00, SCS0HD: 0x07, SCS1LD: 0x08, SCS1HD: 0x0F,
    SCS2LD: 0x10, SCS2HD: 0x17, SCS3LD: 0x18, SCS3HD: 0x1F,
    CS0LD: 0xC0, CS0HD: 0xC7, CS1LD: 0xC8, CS1HD: 0xCF,
    CS2LD: 0xD0, CS2HD: 0xDF, CS3LD: 0xF0, CS3HD: 0xFB,
    BOOTLD: 0xFC, BOOTHD: 0xFF,
    SDRAM_CFG: 0x00000200, ADERR: 0xFFFFFFFF,
    0xC08: 0x00FFF000, 0xC0C: 0x00FFF000, 0xC10: 0x01FFF000, 0xC14: 0x00FFF000,
    PCI0_BAR_EN: 0x0000000F,
    0xC48: 0x00000000, 0xC4C: 0x01000000, 0xC50: 0x1C000000, 0xC54: 0x1F000000,
    0xC58: 0x00000000, 0xC5C: 0x01000000, 0xC64: 0x1F000000,
}
#: the configuration header's reset values (ch. 18, "PCI Configuration"):
#: function 0 (normal BARs), function 1 (byte-swapped BARs)
HEADER_RESET = {
    0: {0x00: 0x462011AB, 0x04: 0x02800000, 0x08: 0x06000010, 0x0C: 0x00000000,
        0x10: 0x00000008, 0x14: 0x01000008, 0x18: 0x1C000008, 0x1C: 0x1F000008,
        0x20: 0x14000000, 0x24: 0x14000001},
    1: {0x00: 0x462011AB, 0x04: 0x02800000, 0x08: 0x06000010, 0x0C: 0x00000000,
        0x10: 0x00000008, 0x14: 0x01000008, 0x1C: 0x1F000008},
}
#: which bits of a register a write changes (the rest stays)
_LOW15, _HIGH7, _DEV8 = 0x7FFF, 0x7F, 0xFF
WRITE_MASK = {**{lo: _LOW15 for lo, _ in GROUPS.values()},
              **{hi: _HIGH7 for _, hi in GROUPS.values()},
              PCI1IOLD: _LOW15, PCI1IOHD: _HIGH7, PCI1M0LD: _LOW15, PCI1M0HD: _HIGH7,
              PCI1M1LD: _LOW15, PCI1M1HD: _HIGH7, ISD: _LOW15,
              **{r: _DEV8 for sub in SUBDECODERS.values() for pair in sub.values()
                 for r in pair},
              CPU_CONFIG: 0x0003FFFF, MULTI: 0, ADERR: 0, BUSERR_LO: M32, BUSERR_HI: 0xF,
              CPU_INT_MASK: TO_CPU, PCI0_INT_MASK: TO_PCI, PCI0_CMD: 0xF,
              **{r: 0xFFFFF000 for r in PCI0_BS}, PCI0_BAR_EN: 0x1FF,
              **{r: 0xFFFFF000 for r in PCI0_REMAP}}
_DECODE = set(WRITE_MASK) - {CPU_CONFIG, MULTI, ADERR, BUSERR_LO, BUSERR_HI,
                             CPU_INT_MASK, PCI0_INT_MASK, PCI0_CMD, PCI0_BAR_EN,
                             *PCI0_BS, *PCI0_REMAP}


class GT64120:
    """The system controller's registers, its CPU-side address decode,
    errors and interrupt output (see the module docstring)."""

    def __init__(self, *, big_endian: bool = True) -> None:
        self.big_endian = big_endian
        #: Interrupt* changed: ``on_irq(level)`` (the Malta: CPU INT3)
        self.on_irq: Callable[[bool], None] | None = None
        #: the decode changed (the board recomputes where RAM is)
        self.on_decode: Callable[[], None] | None = None
        #: PCI_0: a configuration read / write (bus, device, function,
        #: register[, value, byte mask]) — None: no device (master abort)
        self.pci_config_read: Callable[[int, int, int, int], int | None] | None = None
        self.pci_config_write: Callable[[int, int, int, int, int, int], bool] | None = None
        #: PCI_0 interrupt acknowledge (the vector)
        self.pci_iack: Callable[[], int] | None = None
        self.reset()

    def reset(self) -> None:
        self.regs: dict[int, int] = dict(RESET)
        self.regs[CPU_CONFIG] = 0 if self.big_endian else CPU_ENDIAN_LITTLE
        self.regs[PCI0_CMD] = 0 if self.big_endian else 1      # = the Endianess pin
        #: the configuration header, by function and register
        self.header = {fn: dict(regs) for fn, regs in HEADER_RESET.items()}
        self.irq = False
        self._build()

    # -- byte order (3.5, 14.1) -----------------------------------------------------------

    @property
    def lanes_reversed(self) -> bool:
        """The CPU Endianess bit and PCI ByteSwap differ: PCI data has its
        byte lanes reversed within each 32-bit word (Tables 6 / 42)."""
        return bool(self.regs[CPU_CONFIG] & CPU_ENDIAN_LITTLE) != bool(self.regs[PCI0_CMD] & 1)

    def _cfgdata_order(self) -> str:
        """How the bytes of PCI_0 Configuration Data map to the value:
        the GT-64120's own header little-endian (like its registers), any
        other device's keeps its value in the CPU's byte order (Table 6),
        reversed when the lanes are."""
        addr = self.regs.get(PCI0_CFGADDR, 0)
        if addr >> 16 & 0xFF == 0 and addr >> 11 & 0x1F == 0:
            return "little"
        big = not self.regs[CPU_CONFIG] & CPU_ENDIAN_LITTLE
        return "big" if big != self.lanes_reversed else "little"

    # -- its configuration header (ch. 18 "PCI Configuration") --------------------------

    def _bar(self, fn: int, reg: int) -> tuple | None:
        for bar in PCI_BARS:
            if bar[0] == fn and bar[1] == reg:
                return bar
        return None

    def config_read(self, fn: int, reg: int) -> int | None:
        """A configuration read of the GT-64120's header (None: no such
        function).  A BAR reads its base without the Bank Size bits (7.6.1:
        how software sizes it), bit 3 Prefetch."""
        regs = self.header.get(fn)
        if regs is None:
            return None
        bar = self._bar(fn, reg)
        if bar is not None:
            size = self.regs[bar[3]] | 0xFFF
            return regs.get(reg, 0) & ~size & M32 | 0x8
        if fn == 0 and reg == 0x20:                  # internal registers: 4 KB
            return regs[0x20] & 0xFFFFF000
        if fn == 0 and reg == 0x24:
            return regs[0x24] & 0xFFFFF000 | 1
        return regs.get(reg, 0)

    def config_write(self, fn: int, reg: int, value: int, mask: int) -> bool:
        regs = self.header.get(fn)
        if regs is None:
            return False
        old = regs.get(reg, 0)
        value = old & ~mask | value & mask
        if reg == 0x04:                              # Command: IOEn MEMEn MasEn MemWrInv
            regs[reg] = old & ~0x157 | value & 0x157  # PErrEn SErrEn
        elif self._bar(fn, reg) is not None or (fn == 0 and reg in (0x20, 0x24)):
            regs[reg] = value & 0xFFFFF000 | (old & 0xFFF)
            bar = self._bar(fn, reg)
            if bar is not None:                      # 5.2.1: the remap follows the BAR
                self.regs[bar[4]] = value & 0xFFFFF000
        elif reg in (0x0C, 0x3C):                    # latency timer / interrupt line
            regs[reg] = old & ~0x0000FF00 | value & 0x0000FF00 if reg == 0x0C else \
                old & ~0xFF | value & 0xFF
        return True

    # -- the PCI target decode (7.3.5, 5.2) --------------------------------------------

    def slave_decode(self, addr: int) -> tuple | str | None:
        """A PCI memory access at ``addr``: (group, device, CPU-side
        physical address, swapped), ``"retry"`` (a BAR hit no sub-decoder
        takes), or None (no BAR: master abort).  The physical address is
        the remapped address in the group's place of the CPU map (the
        decoders are shared: the same SDRAM byte the CPU reaches there)."""
        if not self.header[0][0x04] & 0x2:           # Memory Enable
            return None
        enable = self.regs[PCI0_BAR_EN]
        for fn, reg, group, bs, remap, bit, swapped in PCI_BARS:
            if enable >> bit & 1:
                continue
            mask = self.regs[bs] | 0xFFF
            if (addr ^ self.header[fn].get(reg, 0)) & ~mask & M32:
                continue
            r = self.regs[remap] & ~mask & M32 | addr & mask
            for dev, (lo, hi) in SUBDECODERS[group].items():
                if self.regs[lo] & _DEV8 <= (r >> 20) & 0xFF <= self.regs[hi] & _DEV8:
                    page = (self.regs[GROUPS[group][0]] & _LOW15) >> 7 << 28
                    return group, dev, page | r & 0x0FFFFFFF, swapped
            return "retry"
        if not enable & 0x10:                        # BAR4: the internal registers
            base = self.header[0][0x20] & 0xFFFFF000
            if base <= addr < base + REGS_SIZE:
                return "regs", None, addr - base, False
        return None

    # -- the decode ----------------------------------------------------------------

    @property
    def base(self) -> int:
        """Where the internal registers are (ISD: address bits 35:21)."""
        return (self.regs[ISD] & _LOW15) << 21

    def group_range(self, name: str) -> tuple[int, int]:
        lo, hi = GROUPS[name]
        low, high = self.regs[lo] & _LOW15, self.regs[hi] & _HIGH7
        page = (low >> 7) << 28
        return page | (low & 0x7F) << 21, page | high << 21 | 0x1FFFFF

    def _build(self) -> None:
        """Turn the registers into sorted address intervals (start, end,
        group, device) — device None: the group hits, no sub-decoder does."""
        spans: list[tuple[int, int, str, str | None]] = []
        base = self.base
        spans.append((base, base + REGS_SIZE - 1, "regs", None))
        for name in GROUPS:
            g0, g1 = self.group_range(name)
            if g0 > g1 or g0 > M32:
                continue
            g1 = min(g1, M32)
            subs = SUBDECODERS.get(name)
            if subs is None:                     # PCI windows: no sub-decode
                spans.append((g0, g1, name, name))
                continue
            page = g0 & ~0x0FFFFFFF
            cuts = []
            for dev, (lo, hi) in subs.items():
                d0 = page | (self.regs[lo] & _DEV8) << 20
                d1 = page | (self.regs[hi] & _DEV8) << 20 | 0xFFFFF
                a, b = max(d0, g0), min(d1, g1)
                if a <= b:
                    cuts.append((a, b, dev))
            cuts.sort()
            at = g0
            for a, b, dev in cuts:
                if a < at:                       # overlapping sub-decoders: the
                    a = at                       # first wins (4.5: unpredictable)
                if a > b:
                    continue
                if a > at:
                    spans.append((at, a - 1, name, None))
                spans.append((a, b, name, dev))
                at = b + 1
            if at <= g1:
                spans.append((at, g1, name, None))
        out: list[tuple[int, int, str, str | None]] = []
        for s in sorted(spans, key=lambda t: (t[0], t[2] != "regs")):
            if out and s[0] <= out[-1][1]:       # overlapping groups (4.5): the
                if s[1] <= out[-1][1]:           # earlier start (registers first) wins
                    continue
                s = (out[-1][1] + 1, *s[1:])
            out.append(s)
        self.spans = out
        self._starts = [s[0] for s in out]
        if self.on_decode is not None:
            self.on_decode()

    def decode(self, pa: int) -> tuple[str, str | None] | None:
        """(group, device) decoding ``pa``; None: no group (CPU out of
        range); device None: no sub-decoder (memory out of range)."""
        i = bisect.bisect_right(self._starts, pa) - 1
        if i >= 0:
            s = self.spans[i]
            if pa <= s[1]:
                return s[2], s[3]
        return None

    def ranges(self, device: str) -> list[tuple[int, int]]:
        """The address ranges a sub-decoder (or PCI window) answers."""
        return [(s[0], s[1]) for s in self.spans if s[3] == device]

    # -- errors and the interrupt (4.5, chapter 11) ----------------------------------

    def cpu_out(self, pa: int) -> None:
        """An address no group decodes (Bus Error Address, CPUOut)."""
        self.regs[BUSERR_LO] = pa & M32
        self.regs[BUSERR_HI] = 0
        self.cause(CPU_OUT)

    def mem_out(self, pa: int) -> None:
        """A group hit no sub-decoder took (Address Decode Error, MemOut)."""
        self.regs[ADERR] = pa & M32
        self.cause(MEM_OUT)

    def cause(self, bits: int) -> None:
        self.regs[INT_CAUSE] = self.regs.get(INT_CAUSE, 0) | bits
        self._update_irq()

    def _cause_value(self) -> int:
        c = self.regs.get(INT_CAUSE, 0) & 0x3FFFFFFE
        if c:
            c |= INT_SUM
        if c & self.regs.get(CPU_INT_MASK, 0) & TO_CPU:
            c |= CPU_INT_SUM
        if c & self.regs.get(PCI0_INT_MASK, 0) & TO_PCI:
            c |= PCI_INT_SUM
        return c

    def _update_irq(self) -> None:
        level = bool(self._cause_value() & CPU_INT_SUM)
        if level != self.irq:
            self.irq = level
            if self.on_irq is not None:
                self.on_irq(level)

    # -- register access (bytes, little-endian) ----------------------------------------

    def read(self, offset: int, *, side_effects: bool = False) -> int:
        """The 32-bit register at ``offset``; ``side_effects``: a real read
        (a configuration cycle, an interrupt acknowledge)."""
        offset &= REGS_SIZE - 4
        if offset == INT_CAUSE:
            return self._cause_value()
        if offset == PCI0_CFGDATA:
            return self._config(None, 0xFFFFFFFF, side_effects)
        if offset == PCI0_IACK:                  # 7.5.1: a PCI interrupt acknowledge
            if side_effects and self.pci_iack is not None:
                return self.pci_iack() & 0xFF
            return 0
        return self.regs.get(offset, 0)

    def _config(self, value: int | None, mask: int, side_effects: bool) -> int:
        """A configuration cycle (7.3): type 0 on bus 0 (the GT-64120 is
        device 0; the board's devices answer through the hooks)."""
        addr = self.regs.get(PCI0_CFGADDR, 0)
        bus, dev, fn, reg = addr >> 16 & 0xFF, addr >> 11 & 0x1F, addr >> 8 & 7, addr & 0xFC
        if not addr & CFG_EN:
            return 0xFFFFFFFF if value is None else 0
        if value is None:
            got = self.pci_config_read(bus, dev, fn, reg) if self.pci_config_read else None
            if got is None:
                if side_effects:
                    self.cause(MAS_ABORT0)        # nobody answered: master abort
                return 0xFFFFFFFF
            return got
        ok = self.pci_config_write(bus, dev, fn, reg, value, mask) if self.pci_config_write \
            else False
        if not ok and side_effects:
            self.cause(MAS_ABORT0)
        return 0

    def write(self, offset: int, value: int) -> None:
        offset &= REGS_SIZE - 4
        value &= M32
        if offset == INT_CAUSE:                  # 11.1: write 0 to clear
            old = self.regs.get(INT_CAUSE, 0)
            keep = old & ~(PCI_INT_BITS | 0x1FFFFE) | (old & value & (PCI_INT_BITS | 0x1FFFFE))
            keep |= ~value & CPU_INT_BITS        # the CPU raises a PCI interrupt
            self.regs[INT_CAUSE] = keep & 0x3FFFFFFE
            self._update_irq()
            return
        mask = WRITE_MASK.get(offset)
        if mask is None:                         # not modelled: kept as written
            self.regs[offset] = value
            return
        self.regs[offset] = (self.regs.get(offset, 0) & ~mask) | (value & mask)
        if offset in _DECODE:
            self._build()
        elif offset in (CPU_INT_MASK, PCI0_INT_MASK):
            self._update_irq()

    def load_bytes(self, offset: int, n: int, *, side_effects: bool = True) -> bytes:
        """``n`` bytes of the register space from ``offset`` (bus order);
        each register is read once."""
        out = bytearray()
        words: dict[int, int] = {}
        for k in range(offset, offset + n):
            reg = k & ~3
            if reg not in words:
                words[reg] = self.read(reg, side_effects=side_effects)
            lane = k & 3
            if reg == PCI0_CFGDATA and self._cfgdata_order() == "big":
                lane = 3 - lane
            out.append((words[reg] >> (8 * lane)) & 0xFF)
        return bytes(out)

    def store_bytes(self, offset: int, data: bytes) -> None:
        k = offset
        while k < offset + len(data):
            reg = k & ~3
            value = self.read(reg) if reg != INT_CAUSE else 0x3FFFFFFE | CPU_INT_BITS
            mask = 0
            big = reg == PCI0_CFGDATA and self._cfgdata_order() == "big"
            while k < offset + len(data) and k & ~3 == reg:
                byte = data[k - offset]
                lane = 3 - (k & 3) if big else k & 3
                value = (value & ~(0xFF << 8 * lane)) | byte << 8 * lane
                mask |= 0xFF << 8 * lane
                k += 1
            if reg == PCI0_CFGDATA:
                self._config(value, mask, True)
            else:
                self.write(reg, value)

    def describe(self) -> list[str]:
        """The decode as a table (views)."""
        out = [f"internal registers 0x{self.base:08x} (ISD 0x{self.regs[ISD]:03x})"]
        for s0, s1, group, dev in self.spans:
            if group == "regs":
                continue
            out.append(f"0x{s0:08x}-0x{s1:08x}  {group:<8} "
                       + (dev if dev is not None else "(no sub-decoder: MemOut)"))
        return out
