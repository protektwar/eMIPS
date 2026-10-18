"""The MIPS Malta board with a CoreLV card (PLAN M15).

References: *MIPS Malta User's Manual* (MD00048 rev. 01.07: Table 3.1 the
physical memory map, chapter 3 the board registers, Table 5.3 the CPU
interrupts), the *CoreLV User's Manual* (MD00007 rev. 02.08), the YAMON
manuals (MD00008 Table 3.1: the map YAMON programs), the GT-64120
datasheet (:mod:`emips.gt64120`); QEMU's ``hw/mips/malta.c`` and Linux's
``arch/mips/mti-malta`` for what the manuals leave open.

The physical address space as the GT-64120 decodes it (after the boot
firmware, as YAMON sets it up)::

    0x00000000  SDRAM (the ram option, at most 256 MB)       SCS[0]
    0x10000000  PCI memory 0                                 (no devices: phase 2)
    0x18000000  PCI I/O: ISA port P at 0x18000000 + P        (phase 2)
    0x18200000  PCI memory 1
    0x1BE00000  GT-64120 registers (at reset: 0x14000000)
    0x1E000000  Monitor flash, 4 MB                          BootCS -> CBUS
    0x1F000000  board (FPGA) registers                       BootCS -> CBUS
    0x1FC00000  boot: the monitor flash again (0x1FC00010: REVISION)

Everything else is a bus error on a read (IBE / DBE) and a GT-64120
interrupt on CPU INT3 (COREHI) for any access.  :class:`MaltaBus` is the
one device that covers every address the SDRAM does not: it routes the
access through the GT-64120's decode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ata as ATA
from . import gt64120 as GT
from .exceptions import BusError
from .piix4 import CPU_HZ, PIIX4
from .superio import LD_SERIAL1, LD_SERIAL2, SuperIO
from .uart16550 import UART16550

M32 = 0xFFFFFFFF

FLASH_BASE, FLASH_SIZE = 0x1E000000, 4 << 20
BOOT_BASE = 0x1FC00000
REVISION_ADDR = 0x1FC00010
FPGA_BASE, FPGA_SIZE = 0x1F000000, 1 << 20
#: REVISION (MD00048 3.2): FPGRV 0, CORID 1 (CoreLV), CORRV 0, PROID 2
#: (Malta), PRORV 0 — what QEMU's Malta reads
REVISION = (1 << 10) | (2 << 4)
#: board register offsets in the FPGA (MD00048 Tables 3.4-3.18)
NMISTATUS, NMIACK = 0x024, 0x104
SWITCH, STATUS, JMPRS = 0x200, 0x208, 0x210
LEDBAR, ASCIIWORD = 0x408, 0x410
ASCIIPOS = tuple(0x418 + 8 * i for i in range(8))
SOFTRES, BRKRES = 0x500, 0x508
CBUS_UART, CBUS_UART_END = 0x900, 0x940          # tty2: 8 registers 8 bytes apart
GPOUT, GPINP = 0xA00, 0xA08
I2CINP, I2COE, I2COUT, I2CSEL = 0xB00, 0xB08, 0xB10, 0xB18
#: SOFTRES: the value that resets the board (GORESET)
GORESET = 0x42
#: STATUS bits
ST_MFWR, ST_S54, ST_S53, ST_BIGEND = 1 << 4, 1 << 3, 1 << 2, 1 << 1
#: CPU interrupt lines (Table 5.3): INTn = the core's HW n (IP n + 2)
INT_SB, INT_SMI, INT_CBUS_UART, INT_COREHI, INT_CORELO = 0, 1, 2, 3, 4
#: the most SDRAM the Malta maps at 0 (MD00048 p. 14)
MAX_RAM = 256 << 20
#: the assembler's segments on the Malta: everything in kseg0 SDRAM, like
#: a YAMON application (MD00008 ch. 5; YAMON's own example starts at
#: 0x80100000): the exception vectors at the bottom, kernel data at 64 KB,
#: code at 1 MB, data at 2 MB
SEGMENT_BASES = {"text": 0x80100000, "data": 0x80200000,
                 "ktext": 0x80000180, "kdata": 0x80010000}
#: where the firmware puts the PIIX4's bus master IDE registers (I/O space)
BMIBA = 0x1000
#: with a CPS: the GIC shared interrupt the south bridge's INTR drives (Linux's
#: malta.dts "i8259 ... interrupts = <GIC_SHARED 3>", QEMU's malta.c)
GIC_I8259 = 3
#: the IDE drive positions (MD00048 6.12: J18 primary, J19 / the
#: CompactFlash socket J20 secondary): (channel, device)
DISK_SLOTS = {"pri0": (0, 0), "pri1": (0, 1), "sec0": (1, 0), "sec1": (1, 1)}
_SIZE_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def parse_size(text: str) -> int:
    """``64K``, ``8M``, ``1G`` or a number of bytes."""
    t = text.strip().upper().removesuffix("B")
    unit = t[-1:] if t[-1:] in _SIZE_UNITS and t[-1:] else ""
    try:
        n = int(t[:-1] if unit else t, 0) * _SIZE_UNITS[unit]
    except ValueError:
        raise ValueError(f"not a size: {text!r} (64K, 8M, 1G)") from None
    if not ATA.SECTOR <= n <= ATA.MAX_SECTORS * ATA.SECTOR:
        raise ValueError(f"a disk of 512 bytes - 128 GB, not {text}")
    return n


def parse_disks(spec: str) -> list[tuple[str, str]]:
    """The ``disks`` option: ``[SLOT=]SOURCE,...`` -> [(slot, source)];
    SLOT pri0 / pri1 / sec0 / sec1 (default: the next free one, pri0
    first), SOURCE an image file or ``blank:SIZE`` (an empty disk)."""
    out: list[tuple[str, str]] = []
    for item in (x.strip() for x in spec.split(",")):
        if not item:
            continue
        slot, sep, source = item.partition("=")
        if not sep or slot.strip().lower() not in DISK_SLOTS:
            slot, source = "", item
        slot = slot.strip().lower()
        if not slot:
            free = [k for k in DISK_SLOTS if k not in {s for s, _ in out}]
            if not free:
                raise ValueError("at most four IDE disks (pri0 pri1 sec0 sec1)")
            slot = free[0]
        if slot in {s for s, _ in out}:
            raise ValueError(f"two disks in {slot}")
        source = source.strip()
        if not source:
            raise ValueError(f"{slot}: no image")
        if source.lower().startswith("blank:"):
            parse_size(source[6:])
        out.append((slot, source))
    return out


def load_media(source: str) -> ATA.Media:
    """A disk's sectors from its SOURCE (``blank:SIZE`` or a file)."""
    if source.lower().startswith("blank:"):
        return ATA.Media(None, parse_size(source[6:]) // ATA.SECTOR)
    data = Path(source).expanduser().read_bytes()
    if not data:
        raise ValueError(f"{source}: an empty disk image")
    return ATA.Media.of(data)


class Fpga:
    """The Malta's board registers (the CBUS FPGA, MD00048 chapter 3):
    32-bit registers on 64-bit boundaries."""

    def __init__(self, board: "MaltaBoard") -> None:
        self.board = board
        self.reset()

    def reset(self) -> None:
        self.ledbar = 0
        self.ascii = [" "] * 8
        self.brkres = 0x0A
        self.gpout = 0
        self.i2coe = 0
        self.i2cout = 0x3
        self.i2csel = 0x01
        self.nmi_button = False
        self.reset_requests = 0

    def read(self, off: int, *, side_effects: bool = False) -> int:
        b = self.board
        if CBUS_UART <= off < CBUS_UART_END:     # the 16C550's byte in the low lane
            u = b.cbus_uart
            reg = (off - CBUS_UART) >> 3
            return u.read(reg) if side_effects else u.peek(reg)
        if off == NMISTATUS:
            return int(self.nmi_button)
        if off == SWITCH:
            return b.switches & 0xFF
        if off == STATUS:
            return ST_MFWR | (ST_BIGEND if b.big_endian else 0)
        if off == JMPRS:
            return 0
        if off == LEDBAR:
            return self.ledbar
        if off == BRKRES:
            return self.brkres
        if off == GPOUT:
            return self.gpout
        if off == I2CINP:                        # the bus lines idle high
            sda = 0 if self.i2coe & 1 and not self.i2cout & 1 else 1
            scl = 0 if self.i2coe & 2 and not self.i2cout & 2 else 1
            return sda | scl << 1
        if off == I2COE:
            return self.i2coe
        if off == I2COUT:
            return self.i2cout
        if off == I2CSEL:
            return self.i2csel
        return 0                                 # write-only / not decoded: 0

    def write(self, off: int, value: int) -> None:
        if CBUS_UART <= off < CBUS_UART_END:
            self.board.cbus_uart.write((off - CBUS_UART) >> 3, value & 0xFF)
            return
        if off == NMIACK:
            if value & 1:
                self.nmi_button = False
        elif off == LEDBAR:
            self.ledbar = value & 0xFF
        elif off == ASCIIWORD:
            self.ascii = list(f"{value & M32:08X}")
        elif off in ASCIIPOS:
            ch = value & 0xFF
            self.ascii[ASCIIPOS.index(off)] = chr(ch) if 0x20 <= ch < 0x7F else " "
        elif off == SOFTRES:
            if value & 0xFF == GORESET:
                self.reset_requests += 1
                self.board.software_reset()
        elif off == BRKRES:
            self.brkres = value & 0xFF
        elif off == GPOUT:
            self.gpout = value & 0xFF
        elif off == I2COE:
            self.i2coe = value & 3
        elif off == I2COUT:
            self.i2cout = value & 3
        elif off == I2CSEL:
            self.i2csel = value & 1

    @property
    def display(self) -> str:
        return "".join(self.ascii)


class MaltaBus:
    """The device on every core's bus that answers whatever the SDRAM
    does not: the GT-64120's decode (see the module docstring)."""

    def __init__(self, board: "MaltaBoard") -> None:
        self.board = board

    def covers(self, pa: int) -> bool:
        for lo, hi in self.board.ram:
            if lo <= pa <= hi:
                return False
        return True

    def _route(self, pa: int) -> tuple[str, Any]:
        b = self.board
        hit = b.gt.decode(pa)
        if hit is None:
            return "cpu-out", None
        group, dev = hit
        if group == "regs":
            return "regs", pa - b.gt.base
        if dev is None:
            return "mem-out", None
        if group in ("scs10", "scs32"):
            return "no-sdram", None              # a bank without chips
        if group.startswith("pci"):
            return "pci", group
        return "cbus", pa                        # the device bus = the CBUS (MD00007 3.2)

    def _cbus_bytes(self, pa: int, n: int, side_effects: bool = False) -> bytes | None:
        """The CBUS (MD00048 Table 3.1): flash, its boot alias, the board
        registers; nothing else answers (reads 0)."""
        b = self.board
        if REVISION_ADDR <= pa < REVISION_ADDR + 4:
            return _lanes(REVISION, pa - REVISION_ADDR, n, b.big_endian)
        if FPGA_BASE <= pa < FPGA_BASE + FPGA_SIZE:
            off = pa - FPGA_BASE
            if side_effects:
                b.sync()
            return _lanes(b.fpga.read(off & ~3, side_effects=side_effects), off & 3, n,
                          b.big_endian)
        flash = b.flash_offset(pa)
        if flash is not None:
            return bytes(b.m.phys_read_byte(FLASH_BASE + flash + i) for i in range(n))
        return bytes(n)

    def load(self, m, pa: int, size: int) -> int:
        kind, arg = self._route(pa)
        gt = self.board.gt
        if kind == "cpu-out":
            gt.cpu_out(pa)
            raise BusError(pa, "read")
        if kind == "mem-out":
            gt.mem_out(pa)
            raise BusError(pa, "read")
        data = self._bytes(kind, arg, pa, size, side_effects=True)
        return int.from_bytes(data, "big" if m.big_endian else "little")

    def _bytes(self, kind: str, arg: Any, pa: int, size: int, *, side_effects: bool) -> bytes:
        b = self.board
        gt = b.gt
        if kind == "regs":
            if side_effects:
                b.sync()
            return gt.load_bytes(arg, size, side_effects=side_effects)
        if kind == "pci":
            if arg == "pci0io":                  # PCI I/O: the PIIX4 takes every ISA port
                port, rev = _pci_lanes(pa - gt.group_range("pci0io")[0], size,
                                       gt.lanes_reversed)
                if side_effects:
                    b.sync()
                    out = b.piix4.read_bytes(port, size)
                    b.touched()
                else:
                    out = b.piix4.read_bytes(port, size, peek=True)
                return out[::-1] if rev else out
            if side_effects:                     # PCI memory: no device, master abort
                gt.cause(GT.MAS_ABORT0)
            return b"\xff" * size
        if kind == "cbus":
            return self._cbus_bytes(pa, size, side_effects) or bytes(size)
        return bytes(size)                       # no SDRAM chips there

    def store(self, m, pa: int, value: int, size: int) -> None:
        kind, arg = self._route(pa)
        b = self.board
        gt = b.gt
        data = (value & ((1 << 8 * size) - 1)).to_bytes(size, "big" if m.big_endian else "little")
        if kind == "cpu-out":                    # a write: the interrupt only (4.5)
            gt.cpu_out(pa)
        elif kind == "mem-out":
            gt.mem_out(pa)
        elif kind == "regs":
            b.sync()
            gt.store_bytes(arg, data)
            b.touched()
        elif kind == "pci" and arg == "pci0io":
            b.sync()
            port, rev = _pci_lanes(pa - gt.group_range("pci0io")[0], size, gt.lanes_reversed)
            b.piix4.write_bytes(port, data[::-1] if rev else data)
            b.touched()
        elif kind == "pci":
            gt.cause(GT.MAS_ABORT0)
        elif kind == "cbus" and FPGA_BASE <= pa < FPGA_BASE + FPGA_SIZE:
            b.sync()
            off = pa - FPGA_BASE
            reg = off & ~3
            old = _lanes(b.fpga.read(reg), 0, 4, b.big_endian)
            word = bytearray(old)
            word[off & 3: (off & 3) + size] = data[:4 - (off & 3)]
            b.fpga.write(reg, int.from_bytes(word, "big" if b.big_endian else "little"))
            b.touched()
        # the flash: read-only here (the CFI command set is not modelled)

    def peek(self, m, pa: int, size: int) -> int | None:
        """The value a load would get, without side effects; None: a bus
        error."""
        kind, arg = self._route(pa)
        if kind in ("cpu-out", "mem-out"):
            return None
        data = self._bytes(kind, arg, pa, size, side_effects=False)
        return int.from_bytes(data, "big" if m.big_endian else "little")

    def tick(self, m, cycles: int) -> None:
        b = self.board
        if m is not b.m:                         # one clock: the core with the pins
            return
        b.now += cycles
        if b.due is not None and b.now >= b.due:
            b.sync()


def _pci_lanes(addr: int, size: int, reversed_: bool) -> tuple[int, bool]:
    """Where a CPU access of ``size`` bytes at ``addr`` lands on PCI: with
    the byte lanes reversed (the GT-64120's swap bits disagree) byte k of
    a word goes to byte 3 - k."""
    if not reversed_:
        return addr, False
    return (addr & ~3) | (4 - (addr & 3) - size), True


def _lanes(word: int, k: int, n: int, big: bool) -> bytes:
    """Bytes ``k .. k + n`` of a 32-bit register on the bus."""
    return word.to_bytes(4, "big" if big else "little")[k:k + n]


class MaltaBoard:
    """A Malta with a CoreLV card: the GT-64120, the SDRAM, the CBUS
    (monitor flash, board registers)."""

    name = "malta"

    def __init__(self, *, ram: int = 128 << 20, big_endian: bool = True,
                 switches: int = 0, cpu_hz: int = CPU_HZ) -> None:
        if not 1 << 20 <= ram <= MAX_RAM:
            raise ValueError(f"the Malta maps 1 MB - 256 MB of SDRAM at 0, not {ram >> 20} MB")
        self.ram_size = ram
        self.big_endian = big_endian
        self.switches = switches
        self.gt = GT.GT64120(big_endian=big_endian)
        self.cpu_hz = cpu_hz
        self.cbus_uart = UART16550("tty2 (CBUS UART, 16C550)", clock=3_686_400, cpu_hz=cpu_hz,
                                   gate_out2=False)
        self.cbus_uart.on_irq = lambda level: self._line(INT_CBUS_UART, level)
        self.piix4 = PIIX4(cpu_hz)
        self.superio = SuperIO(cpu_hz)
        self.piix4.isa_devices.append(self.superio)
        self.superio.set_irq = self.piix4.pics.set_irq
        self.piix4.pics.on_intr = lambda level: self._line(INT_SB, level)
        self.gt.pci_config_read = self._config_read
        self.gt.pci_config_write = self._config_write
        self.gt.pci_iack = self.piix4.pics.inta
        self.now = 0                             # the board's clock: core 0's cycles
        self.due: int | None = 0                 # the next device event
        self.lines_out = 0                       # the CPU interrupt lines it drives
        self.fpga = Fpga(self)
        self.bus = MaltaBus(self)
        self.m = None                            # the core whose pins it drives (core 0)
        self.machines: list = []
        #: the CPS of a multi-core Malta (join_cps), None: one core
        self.cps = None
        self.booted = False                      # the firmware ran (once, any core)
        self.ram: list[tuple[int, int]] = []
        self.gt.on_decode = self._ram_changed
        self.gt.on_irq = self._gt_irq
        #: the IDE disks by slot (pri0 ...): :meth:`attach_disk`
        self.disks: dict[str, ATA.ATADisk] = {}
        for bm in self.piix4.bm:                 # PCI masters reach memory here
            bm.dma = self.dma_access
            bm.target = self.dma_target
        self._ram_changed()

    def attach(self, m, *, pins: bool = True) -> None:
        """Put the board on ``m``'s bus (last: other devices first);
        ``pins``: its interrupt lines go to this core (core 0)."""
        m.board = self
        m.devices.append(self.bus)
        cpu = getattr(m, "cpu_num", 0)            # a core restarted by the CPC: replaced
        self.machines = [x for x in self.machines if getattr(x, "cpu_num", 0) != cpu] + [m]
        if pins:
            self.m = m
            if self.cps is None:                  # the lines it already drives
                for n in range(6):
                    if self.lines_out >> n & 1:
                        m.assert_irq(n)
            self._gt_irq(self.gt.irq)

    def join_cps(self, cps) -> None:
        """The cores are a CPS (a multi-core Malta, as Linux and QEMU know
        it): the CM decodes the GCR / CPC / GIC before the GT-64120 (their
        devices are first on every core's bus); the south bridge's INTR goes
        to GIC shared interrupt 3, the board's other lines (SMI, the CBUS
        UART, COREHI, CORELO) to core 0 directly, ORed with the GIC's pins
        (Linux keeps COREHI a CPU interrupt with a GIC)."""
        self.cps = cps
        for n in range(6):
            if self.lines_out >> n & 1:
                if self.m is not None:
                    self.m.clear_irq(n)
                self._line(n, True)

    def cps_firmware(self) -> None:
        """After the CPS firmware: GIC source 3 (the 8259s) level, active
        high, enabled, to core 0's pin 0 — Cause.IP2, where INT0 is
        without a GIC."""
        from .gic import MAP_TO_PIN
        g = self.cps.gic
        bit = 1 << GIC_I8259
        g.trig &= ~bit
        g.pol |= bit
        g.mask |= bit
        g.map_pin[GIC_I8259] = MAP_TO_PIN | 0
        g.map_vp[GIC_I8259] = 1
        g.update()

    def attach_disk(self, slot: str, media: ATA.Media, name: str = "") -> ATA.ATADisk:
        """A disk on an IDE cable, powered on with the board (BSY until it
        has spun up, unless the firmware waits for it)."""
        ch, dev = DISK_SLOTS[slot]
        disk = ATA.ATADisk(media, name=name or slot, serial=f"EMIPS-{slot.upper()}",
                           hz=self.cpu_hz)
        self.piix4.ide[ch].attach(dev, disk)
        disk.power_on(ready=False)
        self.disks[slot] = disk
        self.touched()
        return disk

    def _ram_changed(self) -> None:
        """Where SDRAM answers: the SDRAM sub-decoders' ranges that hold
        chips (the first ``ram_size`` bytes)."""
        out = []
        for dev in ("scs0", "scs1", "scs2", "scs3"):
            for lo, hi in self.gt.ranges(dev):
                hi = min(hi, self.ram_size - 1)
                if lo <= hi:
                    out.append((lo, hi))
        self.ram = sorted(out)

    def _gt_irq(self, level: bool) -> None:
        self._line(INT_COREHI, level)

    def _line(self, n: int, level: bool) -> None:
        """A CPU interrupt input INTn (MD00048 Table 5.3) of the core with
        the pins."""
        bit = 1 << n
        self.lines_out = self.lines_out | bit if level else self.lines_out & ~bit
        if self.cps is not None:
            if n == INT_SB:
                self.cps.gic.set_input(GIC_I8259, level)
            else:
                self.cps.gic.set_extra(0, n, level)
        elif self.m is not None:
            (self.m.assert_irq if level else self.m.clear_irq)(n)

    # -- time and the PCI bus -----------------------------------------------------------

    @property
    def console(self) -> UART16550:
        """The serial console: tty0 (Linux's ttyS0, YAMON's tty0)."""
        return self.superio.uarts[0]

    def sync(self) -> None:
        """Bring the devices up to the board's time (before an access, at
        an event): between events only their clocks move."""
        now = self.now
        if self.due is not None and now >= self.due:
            self.piix4.advance(now)
            self.cbus_uart.advance(now)
            self.touched()
            return
        self.piix4.now = now
        self.cbus_uart.now = now
        for u in self.superio.uarts:
            u.now = now

    def touched(self) -> None:
        """A device changed: when is the next event?"""
        t = [d for d in (self.piix4.next_due(), self.cbus_uart.next_due()) if d is not None]
        self.due = min(t) if t else None

    #: PCI bus 0: the GT-64120 is device 0, the PIIX4 device 10 (MD00048
    #: Table 5.1: IDSEL AD20); slots, Ethernet and audio are empty here
    PIIX4_DEVICE = 10

    def _config_read(self, bus: int, dev: int, fn: int, reg: int) -> int | None:
        if bus != 0:
            return None
        if dev == 0:
            return self.gt.config_read(fn, reg)
        if dev == self.PIIX4_DEVICE and fn < len(self.piix4.functions):
            return self.piix4.functions[fn].read(reg)
        return None

    def _config_write(self, bus: int, dev: int, fn: int, reg: int, value: int,
                      mask: int) -> bool:
        if bus != 0:
            return False
        if dev == 0:
            return self.gt.config_write(fn, reg, value, mask)
        if dev == self.PIIX4_DEVICE and fn < len(self.piix4.functions):
            self.piix4.functions[fn].write(reg, value, mask)
            return True
        return False

    # -- PCI masters: memory through the GT-64120's target interface (7.3.5) ------------

    def dma_access(self, addr: int, n: int, data: bytes | None = None) -> bytes | str:
        """A PCI master reads ``n`` bytes at PCI address ``addr`` (``data``
        None) or writes ``data``: the bytes read (in address order) /
        b"" for a write, or "abort" (no BAR: master abort) / "retry" (the
        GT-64120 retries: a BAR hit no device sub-decoder takes).  The
        SDRAM is reached below every cache and the write buffer: DMA on
        the Malta is not coherent with the CPU's caches."""
        out = bytearray()
        k = 0
        while k < n:
            a = (addr + k) & M32
            hit = self.gt.slave_decode(a)
            if hit is None:
                return "abort"
            if hit == "retry":
                return "retry"
            group, dev, pa, swapped = hit
            step = min(n - k, 0x1000 - (a & 0xFFF))   # one decode per 4 KB
            rev = self.gt.lanes_reversed != swapped
            if (not rev and group in ("scs10", "scs32") and self.cps is not None
                    and self.cps.iocu
                    and any(lo <= pa and pa + step - 1 <= hi for lo, hi in self.ram)):
                cm = self.cps.cm                 # through the IOCU: coherent
                if data is None:
                    out += cm.iocu_read(pa, step)
                else:
                    cm.iocu_write(pa, bytes(data[k:k + step]))
                k += step
                continue
            for i in range(step):
                where = (pa + i) ^ 3 if rev else pa + i
                if data is None:
                    out.append(self._dma_byte(group, where))
                else:
                    self._dma_store(group, where, data[k + i])
            k += step
        if data is None:
            return bytes(out)
        self.touched()
        return b""

    def dma_target(self, addr: int) -> str:
        """Would the GT-64120 take a PCI memory access at ``addr``: "ok",
        "abort" (no BAR) or "retry"."""
        hit = self.gt.slave_decode(addr & M32)
        return "abort" if hit is None else ("retry" if hit == "retry" else "ok")

    def _dma_byte(self, group: str, pa: int) -> int:
        if group == "regs":
            return self.gt.load_bytes(pa & 0xFFF, 1)[0]
        if group in ("scs10", "scs32"):
            if any(lo <= pa <= hi for lo, hi in self.ram):
                return self.m.phys_read_byte(pa) if self.m is not None else 0
            return 0                             # a bank without chips
        return (self.bus._cbus_bytes(pa, 1, True) or b"\0")[0]

    def _dma_store(self, group: str, pa: int, byte: int) -> None:
        if group == "regs":
            self.gt.store_bytes(pa & 0xFFF, bytes((byte,)))
        elif group in ("scs10", "scs32"):
            if self.m is not None and any(lo <= pa <= hi for lo, hi in self.ram):
                self.m.phys_write_byte(pa, byte)
        elif FPGA_BASE <= pa < FPGA_BASE + FPGA_SIZE:
            off = pa - FPGA_BASE
            word = bytearray(_lanes(self.fpga.read(off & ~3), 0, 4, self.big_endian))
            word[off & 3] = byte
            self.fpga.write(off & ~3, int.from_bytes(word, "big" if self.big_endian
                                                     else "little"))

    def flash_offset(self, pa: int) -> int | None:
        if FLASH_BASE <= pa < FLASH_BASE + FLASH_SIZE:
            return pa - FLASH_BASE
        if BOOT_BASE <= pa < BOOT_BASE + FLASH_SIZE:
            return pa - BOOT_BASE
        return None

    def storage(self, pa: int) -> int:
        """Where a loader / debugger write at ``pa`` is kept: SDRAM as is,
        the flash (and its boot alias) in the flash; anything else is not
        memory (a :class:`BusError`)."""
        if not self.bus.covers(pa):
            return pa
        off = self.flash_offset(pa)
        if off is not None and self.bus._route(pa)[0] == "cbus":
            return FLASH_BASE + off
        raise BusError(pa, "load", f"0x{pa:08x} is not memory on the Malta board "
                       f"(SDRAM: 0x00000000-0x{self.ram_size - 1:08x})")

    def firmware(self, m) -> None:
        """What YAMON does before it runs a program (MD00008 Table 3.1,
        MD00007 3.2.1; QEMU's boot stub): the GT-64120 registers at
        0x1BE00000, SDRAM bank 0 = the whole SDRAM, the device bus as one
        BootCS region 0x1E000000-0x1FFFFFFF (the monitor flash and the
        board), PCI I/O at 0x18000000 (2 MB), PCI memory at 0x10000000
        (128 MB) and 0x18200000 (60 MB); the unused groups moved above 4 GB."""
        if self.booted:                          # a CPS loads every core: once
            return
        self.booted = True
        g = self.gt
        top = self.ram_size - 1
        park = 0x7F80                            # bits 35:28 = 0xFF: beyond 32-bit PAs
        writes = [
            (GT.SCS10LD, 0x000), (GT.SCS10HD, (top >> 21) & 0x7F),
            (GT.SCS0LD, 0x00), (GT.SCS0HD, top >> 20),
            (GT.SCS1LD, 0xFF), (GT.SCS1HD, 0x00),
            (GT.SCS32LD, park), (GT.SCS32HD, 0x00),
            (GT.SCS2LD, 0xFF), (GT.SCS2HD, 0x00), (GT.SCS3LD, 0xFF), (GT.SCS3HD, 0x00),
            (GT.CS20LD, park), (GT.CS20HD, 0x00),
            (GT.CS3BOOTLD, 0x1E000000 >> 21), (GT.CS3BOOTHD, 0x7F),
            (GT.CS3LD, 0xFF), (GT.CS3HD, 0x00), (GT.BOOTLD, 0xE0), (GT.BOOTHD, 0xFF),
            (GT.PCI0IOLD, 0x18000000 >> 21), (GT.PCI0IOHD, 0x40),
            (GT.PCI0M0LD, 0x10000000 >> 21), (GT.PCI0M0HD, 0x3F),
            (GT.PCI0M1LD, 0x18200000 >> 21), (GT.PCI0M1HD, 0x5E),
            (GT.ISD, 0x1BE00000 >> 21),
        ]
        for reg, value in writes:
            g.write(reg, value)
        # PCI interrupt routing (PIRQA-D to IRQ 10, 10, 11, 11: QEMU's boot stub)
        isa = self.piix4.functions[0]
        isa.write(0x60, 0x0B0B0A0A)
        # the Super I/O serial ports as YAMON has them (tty0 3F8h IRQ 4,
        # tty1 2F8h IRQ 3), 38400 8N1 with FIFOs (MD00008: 38400,n,8,1,hw)
        self.superio.configure(LD_SERIAL1, 0x3F8, 4)
        self.superio.configure(LD_SERIAL2, 0x2F8, 3)
        for u, divisor in ((self.superio.uarts[0], 3), (self.superio.uarts[1], 3),
                           (self.cbus_uart, 6)):
            u.write(3, 0x80)                     # LCR.DLAB
            u.write(0, divisor)
            u.write(1, 0)
            u.write(3, 0x03)                     # 8 data bits, no parity, 1 stop bit
            u.write(2, 0x07)                     # FIFOs on, emptied
            u.write(4, 0x03)                     # DTR, RTS (OUT2 off: no interrupts yet)
        # the IDE function: I/O space and bus master enabled, both channels
        # decoded with compatible timing (5.1.3, 5.1.10, Table 14); the
        # disks have spun up (YAMON waits for them)
        ide = self.piix4.functions[1]
        ide.write(0x04, 0x05, 0xFF)
        ide.write(0x40, 0x80008000)
        ide.write(0x20, BMIBA)                   # the bus master block (5.1.9)
        # the GT-64120 as a PCI target (7.3.5): Memory / Master Enable, BAR0
        # (SCS[1:0]) at PCI 0 over the whole SDRAM — PCI address = physical
        # address, as Linux's Malta DMA expects; the BARs that would overlap
        # it (SCS[3:2], CS[2:0], CS[3] / BootCS) disabled, BAR4 left at reset
        g.config_write(0, 0x04, 0x06, 0xFF)
        size = 1 << (self.ram_size - 1).bit_length()
        g.write(GT.PCI0_BS[0], (size - 1) & 0xFFFFF000)
        g.config_write(0, 0x10, 0, 0xFFFFFFFF)
        g.write(GT.PCI0_BAR_EN, 0x0F | 0xE0)
        for disk in self.disks.values():
            disk.power_on(ready=True)
        self.touched()

    def nmi_button(self) -> None:
        """The ON/NMI button (MD00048 3.3, Table 5.3): NMISTATUS.ONNMI and
        an edge on the CPU's NMI input (core 0)."""
        self.fpga.nmi_button = True
        if self.m is not None:
            self.m.nmi()

    def software_reset(self) -> None:
        """SOFTRES = GORESET: the whole board resets, as the reset button
        does (MD00048 3.6).  eMIPS ends the program there (Run → Reset
        starts it again: the boot firmware loads it anew)."""
        for m in self.machines:
            m.halted = True

    def user_window(self) -> tuple[int, int]:
        """Where the boot firmware maps kuseg with a TLB: the top
        power-of-two part of the SDRAM, half of it at most (base, size) —
        the kernel keeps the bottom (vectors, its data, code at 1 MB)."""
        return user_window(self.ram_size)

    def stack_top(self) -> int:
        """``$sp`` as YAMON starts an application: 4 words below the top of
        the SDRAM (MD00008 Table 5.1), through kseg0."""
        return 0x80000000 + self.ram_size - 16

    def reset(self) -> None:
        """Power-on / SOFTRES: the GT-64120, the PIIX4, the Super I/O, the
        board registers."""
        self.gt.reset()
        self.fpga.reset()
        self.piix4.reset()
        self.superio.reset()
        self.cbus_uart.reset()
        self.now = 0
        self.touched()

    def lines(self) -> list[str]:
        """The board for the debugger's ``board`` (views)."""
        g = self.gt
        f = self.fpga
        leds = "".join("#" if f.ledbar >> i & 1 else "." for i in range(7, -1, -1))
        cause = g.read(GT.INT_CAUSE)
        out = [f"Malta (CoreLV, GT-64120), {self.cpu_hz // 1_000_000} MHz core, "
               f"{self.ram_size >> 20} MB SDRAM, "
               f"{'big' if self.big_endian else 'little'}-endian, switches S2 = "
               f"0x{self.switches:02x}",
               f"LEDs [{leds}]  ASCII display [{f.display}]",
               f"GT-64120 interrupt cause 0x{cause:08x}, CPU mask "
               f"0x{g.read(GT.CPU_INT_MASK):08x} -> INT3 {'on' if g.irq else 'off'}; "
               f"bus error address 0x{g.read(GT.BUSERR_LO):08x}, decode error "
               f"0x{g.read(GT.ADERR):08x}"]
        out += ["   " + line for line in g.describe()]
        out += self.piix4.pics.describe()
        out.append(self.piix4.rtc.describe(self.now))
        c0 = self.piix4.pit.counters[0]
        if c0.loaded_at is not None:
            out.append(f"timer counter 0: mode {c0.mode}, every {c0.period} ticks of 1.19 MHz "
                       f"(IRQ0)")
        out += self.superio.describe()
        out.append(self.cbus_uart.describe() + " — on the board (INT2)")
        ide = self.piix4.functions[1].space
        for i, ch in enumerate(self.piix4.ide):
            on = ide[0x04] & 1 and ide[0x41 + 2 * i] & 0x80
            for line in ch.describe():
                out.append(line + ("" if on else " — the channel is not decoded "
                                   "(PCICMD.IOSE / IDETIM.IDE)"))
            bm = self.piix4.bm[i]
            if ch.present() or bm.status & 1 or bm.done_bytes:
                out.append("   " + bm.describe())
        for u in (self.superio.uarts[1], self.cbus_uart):   # tty0 is the console
            if u.output:
                tail = u.output[-60:].replace("\n", "\\n")
                out.append(f"   {u.name.split()[0]} sent: {tail}")
        return out


def user_window(ram: int) -> tuple[int, int]:
    """(base, size) of kuseg's window in ``ram`` bytes of SDRAM."""
    size = 1 << ((ram // 2).bit_length() - 1)
    return ram - size, size


#: the assembler's segments for a user-mode program on the Malta (a TLB
#: maps kuseg into the SDRAM's top: emips.tlb.firmware_refill): the MARS
#: user segments, the kernel's as for kernel mode
USER_SEGMENT_BASES = {"text": 0x00400000, "data": 0x10010000,
                      "ktext": 0x80000180, "kdata": 0x80010000}
#: $sp of a user-mode program (MARS: kuseg)
USER_SP = 0x7FFFEFFC


def make_board(spec: str, *, big_endian: bool = True):
    """``flat`` (None: every physical address is RAM) or
    ``malta[:RAM_MB[:SWITCHES[:MHZ]]]`` (MHZ: the core's clock, default 10)."""
    parts = spec.lower().split(":")
    if parts[0] in ("", "flat"):
        if len(parts) > 1:
            raise ValueError("flat takes no parameters")
        return None
    if parts[0] != "malta" or len(parts) > 4:
        raise ValueError(f"unknown board {spec!r} (flat | malta[:RAM_MB[:SWITCHES[:MHZ]]])")
    ram = int(parts[1], 0) << 20 if len(parts) > 1 and parts[1] else 128 << 20
    sw = int(parts[2], 0) & 0xFF if len(parts) > 2 and parts[2] else 0
    mhz = int(parts[3], 0) if len(parts) > 3 and parts[3] else CPU_HZ // 1_000_000
    if not 1 <= mhz <= 1000:
        raise ValueError(f"a core clock of 1 - 1000 MHz, not {mhz}")
    return MaltaBoard(ram=ram, big_endian=big_endian, switches=sw, cpu_hz=mhz * 1_000_000)
