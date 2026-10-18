"""An ATA hard disk and the IDE channel it sits on (PLAN M15 phase 3).

Reference: *AT Attachment with Packet Interface Extension (ATA/ATAPI-4)*,
T13 1153D revision 18 (1998) — the generation of the PIIX4 (1997):
7 the registers (Status, Error, Device/Head, Device Control; the Data
register is 16 bits wide), 6.2 the CHS / LBA addressing and the IDENTIFY
DEVICE words that describe it, 8 the commands (their inputs, normal and
error outputs), 9 the protocols — 9.1 the signature, 9.2 / 9.3 the power-on
/ hardware and software resets, 9.5 EXECUTE DEVICE DIAGNOSTIC, 9.6 device
selection, 9.7 / 9.8 PIO data in / out, 9.9 non-data, 9.16 one device on
a cable; 4.3.1 the host's pull-down on DD7 (an empty cable reads 7Fh).

The disk is a device not implementing the PACKET Command feature set:
the General feature set (EXECUTE DEVICE DIAGNOSTIC, FLUSH CACHE, IDENTIFY
DEVICE, INITIALIZE DEVICE PARAMETERS, NOP, READ / WRITE BUFFER, READ /
WRITE DMA, READ / WRITE MULTIPLE, READ / WRITE SECTOR(S), READ VERIFY
SECTOR(S), SEEK, SET FEATURES, SET MULTIPLE MODE) and the Power Management
feature set (CHECK POWER MODE, IDLE, IDLE IMMEDIATE, SLEEP, STANDBY,
STANDBY IMMEDIATE, the Standby timer).  Its media is a disk image (the
``disks`` option): 28-bit LBA, a default CHS translation of up to 16
heads and 63 sectors per track.  There is no write cache (data is on the
media when a write completes), PIO modes 0-4, multiword DMA 0-2 and Ultra
DMA 0-2 are reported (the DMA data moves when the PIIX4's bus master runs).

Time is the board's cycles (``hz`` per second): a command keeps BSY set
for the times in :data:`TIMING_US` (eMIPS's, shorter than a real disk's
milliseconds — what ATA-4 leaves to the device; ``sim.DEVIATIONS``).
"""

from __future__ import annotations

from typing import Callable

SECTOR = 512
#: the largest CHS-addressable capacity (6.2) and the 28-bit LBA limit
MAX_CHS_SECTORS = 16_514_064
MAX_SECTORS = 1 << 28

# Status (7.15)
BSY, DRDY, DF, DSC, DRQ, ERR = 0x80, 0x40, 0x20, 0x10, 0x08, 0x01
# Error (7.11; the media access commands' error outputs)
UNC, MC, IDNF, MCR, ABRT, NM = 0x40, 0x20, 0x10, 0x08, 0x04, 0x02
# Device Control (7.9)
NIEN, SRST = 0x02, 0x04
# Device/Head (7.10)
LBA, DEV = 0x40, 0x10
#: what an undriven bus reads: DD7 pulled down by the host (4.3.1 note 3),
#: the other lines float high
FLOAT = 0x7F
#: the command block registers by their offset in the block (DA2:DA0)
DATA, ERROR, FEATURES, NSECTOR, SECTOR_NUM, LCYL, HCYL, SELECT, STATUS, COMMAND = \
    0, 1, 1, 2, 3, 4, 5, 6, 7, 7

#: the commands (8; annex E) and their names
COMMANDS = {
    0x00: "NOP", 0x10: "RECALIBRATE", 0x20: "READ SECTOR(S)", 0x21: "READ SECTOR(S)",
    0x30: "WRITE SECTOR(S)", 0x31: "WRITE SECTOR(S)", 0x40: "READ VERIFY SECTOR(S)",
    0x41: "READ VERIFY SECTOR(S)", 0x70: "SEEK", 0x90: "EXECUTE DEVICE DIAGNOSTIC",
    0x91: "INITIALIZE DEVICE PARAMETERS", 0xC4: "READ MULTIPLE", 0xC5: "WRITE MULTIPLE",
    0xC6: "SET MULTIPLE MODE", 0xC8: "READ DMA", 0xC9: "READ DMA", 0xCA: "WRITE DMA",
    0xCB: "WRITE DMA", 0xE0: "STANDBY IMMEDIATE", 0xE1: "IDLE IMMEDIATE", 0xE2: "STANDBY",
    0xE3: "IDLE", 0xE4: "READ BUFFER", 0xE5: "CHECK POWER MODE", 0xE6: "SLEEP",
    0xE7: "FLUSH CACHE", 0xE8: "WRITE BUFFER", 0xEC: "IDENTIFY DEVICE", 0xEF: "SET FEATURES",
}
#: the most sectors a READ / WRITE MULTIPLE block may have (IDENTIFY word 47)
MAX_MULTIPLE = 16
#: the DMA mode after power-on (8.37.10: one is selected at all times; which
#: one is vendor specific — eMIPS: multiword DMA mode 2)
DEFAULT_DMA = ("mwdma", 2)
#: how long the disk is busy (microseconds; eMIPS's choices, see the module
#: docstring): a command without media access, the first sector of a media
#: access (seek + rotation), each further sector, writing a block to the
#: media, a software reset, power-on (spin-up), standby -> active (spin-up)
TIMING_US = {"command": 5, "access": 100, "sector": 10, "write": 100, "reset": 400,
             "power_on": 2000, "spin_up": 1000, "diagnostic": 400}
#: the Standby timer (Table 14): the Sector Count value -> seconds (None: off)


def standby_seconds(value: int) -> float | None:
    if value == 0:
        return None
    if value <= 240:
        return value * 5.0
    if value <= 251:
        return (value - 240) * 30 * 60.0
    return {252: 21 * 60.0, 253: 8 * 3600.0, 254: None, 255: 21 * 60 + 15.0}[value]


def default_chs(sectors: int) -> tuple[int, int, int]:
    """The default CHS translation (6.2.1): 16 heads and 63 sectors per
    track as the BIOSs expect, smaller for a small disk; cylinders such
    that C x H x S does not exceed the capacity (at most 16,383 when the
    capacity is at least 16,514,064 sectors)."""
    spt = min(63, sectors)
    heads = max(1, min(16, sectors // spt))
    cyls = min(sectors, MAX_CHS_SECTORS) // (heads * spt)
    return max(1, min(cyls, 16383)), heads, spt


def size_text(nbytes: int) -> str:
    """512 B, 64 KB, 1.5 MB, 8 GB."""
    for unit, shift in (("GB", 30), ("MB", 20), ("KB", 10)):
        if nbytes >= 1 << shift:
            v = nbytes / (1 << shift)
            return f"{v:.0f} {unit}" if v == int(v) else f"{v:.1f} {unit}"
    return f"{nbytes} B"


def _ascii_words(text: str, n: int) -> list[int]:
    """An IDENTIFY string field: two characters per word, the first in
    bits 15:8 (8.12.8), padded with spaces."""
    raw = text.encode("ascii", "replace")[:2 * n].ljust(2 * n, b" ")
    return [raw[2 * i] << 8 | raw[2 * i + 1] for i in range(n)]


class Media:
    """A disk's sectors: the image it started from (never changed: a
    reset starts from it again) and the sectors written since; bytes
    beyond the image's end read 0 (a file that is not a whole number of
    sectors is padded)."""

    def __init__(self, base: bytes | None, sectors: int) -> None:
        if not 1 <= sectors <= MAX_SECTORS:
            raise ValueError("a disk of 1 sector - 2^28 sectors (128 GB: 28-bit LBA)")
        self.base = base or b""
        self.sectors = sectors
        self.written: dict[int, bytes] = {}

    @classmethod
    def of(cls, image: bytes) -> "Media":
        return cls(bytes(image), max(1, -(-len(image) // SECTOR)))

    def read(self, lba: int, n: int) -> bytes:
        out = bytearray()
        for k in range(lba, lba + n):
            w = self.written.get(k)
            if w is None:
                w = self.base[k * SECTOR:(k + 1) * SECTOR].ljust(SECTOR, b"\0")
            out += w
        return bytes(out)

    def write(self, lba: int, data: bytes) -> None:
        for i in range(len(data) // SECTOR):
            self.written[lba + i] = bytes(data[i * SECTOR:(i + 1) * SECTOR])

    def image(self) -> bytes:
        """The whole disk as it is now (what a save writes)."""
        out = bytearray(self.base.ljust(self.sectors * SECTOR, b"\0")[:self.sectors * SECTOR])
        for k, data in self.written.items():
            out[k * SECTOR:(k + 1) * SECTOR] = data
        return bytes(out)


class ATADisk:
    """One ATA device (Device 0 or 1 on its cable).  ``media``: its
    sectors (:class:`Media`, or the bytes of an image); the channel
    delivers the register accesses."""

    def __init__(self, media: "Media | bytes", *, name: str = "disk",
                 serial: str = "EMIPS-0001", hz: int = 10_000_000) -> None:
        if not isinstance(media, Media):
            media = Media.of(bytes(media))
        self.media = media
        self.sectors = media.sectors
        self.name = name
        self.serial = serial
        self.hz = hz
        self.default = default_chs(self.sectors)
        self.channel: "IDEChannel | None" = None
        self.number = 0                          # Device 0 / 1 (set by the channel)
        #: what it did (the debugger's ``disk``): (cycle, text)
        self.log: list[tuple[int, str]] = []
        self.reads = self.writes = 0             # sectors moved
        self.power_on(ready=True)

    # -- time ------------------------------------------------------------------------

    def cycles(self, key: str, n: int = 1) -> int:
        return max(1, TIMING_US[key] * n * self.hz // 1_000_000)

    @property
    def now(self) -> int:
        return self.channel.now() if self.channel is not None else 0

    def _after(self, cycles: int, then: Callable[[], None]) -> None:
        self.due = self._time() + cycles
        self._then = then

    def advance(self, now: int) -> None:
        """Run the device up to cycle ``now`` (its timed steps)."""
        while self.due is not None and self.due <= now:
            then, self._then = self._then, None
            at, self.due = self.due, None
            self._clock = at
            if then is not None:
                then()
        self._clock = None
        self._standby_check(now)

    def next_due(self) -> int | None:
        t = [d for d in (self.due, self._standby_at()) if d is not None]
        return min(t) if t else None

    def _time(self) -> int:
        return self._clock if self._clock is not None else self.now

    # -- resets ------------------------------------------------------------------------

    def power_on(self, *, ready: bool) -> None:
        """Power-on / hardware reset (9.2): the default CHS translation, no
        multiple mode, PIO default mode; BSY until the device is ready
        (``ready``: it already is — the firmware waited for it)."""
        self.features = self.nsector = self.sector = self.lcyl = self.hcyl = 0
        self.select = 0
        self.ctl = 0
        self.error = 0x01
        self.status = 0
        self.current = self.default              # (cylinders, heads, sectors per track)
        self.chs_valid = True
        self.multiple = 0                        # READ / WRITE MULTIPLE block (0: disabled)
        self.pio_mode, self.dma_mode = 0, DEFAULT_DMA   # SET FEATURES transfer modes
        self.look_ahead = True
        self.revert = True                       # revert to power-on defaults at SRST
        self.power = "active"                    # active / idle / standby / sleep
        self.standby_timer: float | None = None
        self.last_command = 0
        self.intrq = False
        self.buffer = bytearray(SECTOR)          # the sector buffer (READ / WRITE BUFFER)
        self.data = bytearray()                  # the block DRQ transfers
        self.pos = 0
        self.direction: str | None = None        # "in" (to the host) / "out" / "dma-in"...
        self.lba = 0                             # the next sector of a transfer
        self.remaining = 0                       # sectors still to transfer
        self.block = 1
        self.block_sectors = 0
        self.command = None
        self.due: int | None = None
        self._then: Callable[[], None] | None = None
        self._clock: int | None = None
        self.dmarq = False
        if ready:
            self._reset_done()
        else:
            self.status = BSY
            self._after(self.cycles("power_on"), self._reset_done)

    def _reset_done(self) -> None:
        """The end of a reset or a diagnostic: the diagnostic code in
        Error, the signature (9.1), BSY cleared, DRDY set (9.2.1 m-q)."""
        self.error = 0x01                        # passed (Table 10)
        self.nsector, self.sector, self.lcyl, self.hcyl, self.select = 1, 1, 0, 0, 0
        self.status = DRDY | DSC
        self.command = None
        self.data = bytearray()
        self.direction = None
        self._changed()

    def software_reset(self, srst: bool) -> None:
        """The SRST bit (9.3): while it is 1 the device is held BSY (any
        command stops, Sleep ends); when it is cleared the device resets
        and clears BSY after :data:`TIMING_US` ``reset``.  With reverting
        to defaults enabled (SET FEATURES CCh, the default) the settings
        go back to their power-on values; the Ultra DMA mode never does."""
        if srst:
            self.status = BSY
            self.intrq = False
            self.data = bytearray()
            self.direction = None
            self.dmarq = False
            self.due = None
            self._then = None
            if self.power == "sleep":
                self.power = "standby"           # 6.8.2: Sleep -> Standby after a reset
            self._changed()
            return
        if self.revert:
            udma = self.dma_mode if self.dma_mode and self.dma_mode[0] == "udma" else None
            self.current, self.chs_valid = self.default, True
            self.multiple = 0
            self.pio_mode = 0
            self.dma_mode = udma or DEFAULT_DMA
        self._after(self.cycles("reset"), self._reset_done)

    # -- the interrupt -----------------------------------------------------------------

    def _changed(self) -> None:
        if self.channel is not None:
            self.channel.update()

    def _interrupt(self) -> None:
        self.intrq = True
        self._changed()

    # -- addressing (6.2) ----------------------------------------------------------------

    def _address(self) -> int | None:
        """The starting sector the registers give (None: not a valid
        CHS address in the current translation)."""
        if self.select & LBA:
            return (self.select & 0x0F) << 24 | self.hcyl << 16 | self.lcyl << 8 | self.sector
        if not self.chs_valid:
            return None
        cyls, heads, spt = self.current
        cyl, head, sec = self.hcyl << 8 | self.lcyl, self.select & 0x0F, self.sector
        if cyl >= cyls or head >= heads or not 1 <= sec <= spt:
            return None
        return (cyl * heads + head) * spt + sec - 1

    def _set_address(self, lba: int) -> None:
        """The registers say sector ``lba`` (in the command's addressing)."""
        if self.select & LBA:
            self.sector, self.lcyl, self.hcyl = lba & 0xFF, lba >> 8 & 0xFF, lba >> 16 & 0xFF
            self.select = self.select & 0xF0 | lba >> 24 & 0x0F
            return
        cyls, heads, spt = self.current
        cyl, rest = divmod(lba, heads * spt)
        head, sec = divmod(rest, spt)
        self.sector, self.lcyl, self.hcyl = sec + 1, cyl & 0xFF, cyl >> 8 & 0xFF
        self.select = self.select & 0xF0 | head

    def _count(self) -> int:
        return self.nsector or 256

    # -- the registers -----------------------------------------------------------------

    def read_reg(self, reg: int) -> int:
        """A command block register (Status read clears a pending
        interrupt: 7.15.4)."""
        if reg == STATUS:
            value = self.status
            if self.intrq:
                self.intrq = False
                self._changed()
                if self.power == "sleep-pending":
                    self.power = "sleep"         # 8.40: the Status read lets it sleep
            return value
        return self.peek_reg(reg)

    def peek_reg(self, reg: int) -> int:
        return {ERROR: self.error, NSECTOR: self.nsector, SECTOR_NUM: self.sector,
                LCYL: self.lcyl, HCYL: self.hcyl, SELECT: self.select,
                STATUS: self.status}[reg]

    def write_reg(self, reg: int, value: int) -> None:
        """A command block register other than Command (7.4-7.14): ignored
        while BSY (7.15.6.1) — but for the DEV bit, which both devices
        always latch (7.10.4: it takes effect when written), so they never
        disagree about which one is selected."""
        if self.status & BSY:
            if reg == SELECT:
                self.select = self.select & ~DEV | value & DEV
            return
        if reg == FEATURES:
            self.features = value
        elif reg == NSECTOR:
            self.nsector = value
        elif reg == SECTOR_NUM:
            self.sector = value
        elif reg == LCYL:
            self.lcyl = value
        elif reg == HCYL:
            self.hcyl = value
        elif reg == SELECT:
            self.select = value

    # -- the data register (PIO) -------------------------------------------------------

    def read_data(self) -> int:
        """A 16-bit Data register read (7.7): the next word of the block,
        DD7:0 = the byte at the lower offset (the order of the sector)."""
        if self.direction != "in" or not self.status & DRQ:
            return 0xFFFF                         # 7.7.3: only with DRQ (indeterminate)
        word = self.data[self.pos] | self.data[self.pos + 1] << 8
        self.pos += 2
        if self.pos >= len(self.data):
            self._block_read()
        return word

    def peek_data(self) -> int | None:
        if self.direction != "in" or not self.status & DRQ:
            return None
        return self.data[self.pos] | self.data[self.pos + 1] << 8

    def write_data(self, word: int) -> None:
        if self.direction != "out" or not self.status & DRQ:
            return
        self.data[self.pos] = word & 0xFF
        self.data[self.pos + 1] = word >> 8 & 0xFF
        self.pos += 2
        if self.pos >= len(self.data):
            self._block_written()

    # -- commands ----------------------------------------------------------------------

    def command_write(self, code: int) -> None:
        """The Command register (7.4): the command starts; writing it
        clears a pending interrupt.  While BSY the write is ignored; while
        DRQ the transfer in progress is abandoned (indeterminate in 7.4.3:
        eMIPS's choice)."""
        if self.power == "sleep" or self.status & BSY:
            return
        self.intrq = False
        self.data = bytearray()
        self.direction = None
        self.dmarq = False
        self.status = (self.status | BSY) & ~(DRQ | ERR | DF)
        self.error = 0
        self.command = code
        self.last_command = self._time()
        self._log(f"{COMMANDS.get(code, 'unknown')} ({code:02X}h)")
        self._changed()
        handler = {
            0x20: self._read_sectors, 0x21: self._read_sectors,
            0xC4: self._read_multiple, 0x30: self._write_sectors, 0x31: self._write_sectors,
            0xC5: self._write_multiple, 0x40: self._verify, 0x41: self._verify,
            0x70: self._seek, 0x90: self._diagnostic, 0x91: self._init_params,
            0xC6: self._set_multiple, 0xEC: self._identify, 0xEF: self._set_features,
            0xE4: self._read_buffer, 0xE8: self._write_buffer, 0xE7: self._flush,
            0xE5: self._check_power, 0xE0: self._standby_now, 0xE1: self._idle_now,
            0xE2: self._standby, 0xE3: self._idle, 0xE6: self._sleep,
            0xC8: self._dma, 0xC9: self._dma, 0xCA: self._dma, 0xCB: self._dma,
        }.get(code)
        if handler is None:                      # NOP and every unsupported code
            self._after(self.cycles("command"), lambda: self._abort())
            return
        handler()

    def _log(self, text: str) -> None:
        self.log.append((self._time(), text))
        del self.log[:-64]

    # -- endings -------------------------------------------------------------------------

    def _complete(self, *, interrupt: bool = True) -> None:
        """Command completion without an error (9.9): BSY and DRQ cleared."""
        self.status = DRDY | DSC
        self.direction = None
        self.command = None
        if interrupt:
            self._interrupt()
        else:
            self._changed()

    def _abort(self, bits: int = ABRT) -> None:
        """Command completion with an error: the Error bits, ERR (and
        the interrupt)."""
        self.error = bits
        self.status = DRDY | DSC | ERR
        self.direction = None
        self.data = bytearray()
        self.dmarq = False
        self.command = None
        self._log(f"  error {self._error_text(bits)}")
        self._interrupt()

    @staticmethod
    def _error_text(bits: int) -> str:
        names = [n for b, n in ((UNC, "UNC"), (MC, "MC"), (IDNF, "IDNF"), (MCR, "MCR"),
                                (ABRT, "ABRT"), (NM, "NM")) if bits & b]
        return "|".join(names) or f"{bits:02X}h"

    def _spin_up(self) -> int:
        """A media access from Standby spins the disk up (6.8.1)."""
        extra = self.cycles("spin_up") if self.power == "standby" else 0
        self.power = "active"
        return extra

    # -- PIO data in (9.7) -------------------------------------------------------------

    def _read_sectors(self) -> None:
        self._start_read(1)

    def _read_multiple(self) -> None:
        if not self.multiple:
            self._after(self.cycles("command"), self._abort)
            return
        self._start_read(self.multiple)

    def _start_read(self, block: int) -> None:
        lba = self._address()
        if lba is None or lba >= self.sectors:
            self._after(self.cycles("access"), lambda: self._abort(IDNF))
            return
        self.lba, self.remaining, self.block = lba, self._count(), block
        self._after(self.cycles("access") + self._spin_up(), self._next_read_block)

    def _next_read_block(self) -> None:
        """The next block is in the buffer: DRQ, BSY cleared, the
        interrupt (9.7).  A sector beyond the disk ends the command with
        IDNF and the registers at that sector (8.27.6)."""
        n = min(self.block, self.remaining)
        if self.lba + n > self.sectors:
            if self.lba >= self.sectors:
                self._set_address(self.lba)
                self._abort(IDNF)
                return
            n = self.sectors - self.lba          # the good ones first
        self.data = bytearray(self.media.read(self.lba, n))
        self.pos = 0
        self.direction = "in"
        self.block_sectors = n
        self.status = DRDY | DSC | DRQ
        self._interrupt()

    def _block_read(self) -> None:
        """The host read the whole block (9.7 C)."""
        if self.command in (0xEC, 0xE4):         # IDENTIFY DEVICE, READ BUFFER
            self._complete(interrupt=False)
            return
        n = self.block_sectors
        self._set_address(self.lba + n - 1)      # the last sector transferred
        self.lba += n
        self.remaining -= n
        self.reads += n
        self.nsector = self.remaining & 0xFF
        if self.remaining == 0:
            self._complete(interrupt=False)      # PIO in: no interrupt at the end
            return
        self.status = BSY | DRDY | DSC
        self.direction = None
        self._changed()
        self._after(self.cycles("sector", min(self.block, self.remaining)),
                    self._next_read_block)

    # -- PIO data out (9.8) ------------------------------------------------------------

    def _write_sectors(self) -> None:
        self._start_write(1)

    def _write_multiple(self) -> None:
        if not self.multiple:
            self._after(self.cycles("command"), self._abort)
            return
        self._start_write(self.multiple)

    def _start_write(self, block: int) -> None:
        lba = self._address()
        if lba is None or lba >= self.sectors:
            self._after(self.cycles("command"), lambda: self._abort(IDNF))
            return
        self.lba, self.remaining, self.block = lba, self._count(), block
        self._after(self.cycles("command") + self._spin_up(),
                    lambda: self._want_block(interrupt=False))

    def _want_block(self, *, interrupt: bool) -> None:
        """Ready for the next block: DRQ, BSY cleared; the interrupt for
        every block but the first (9.8)."""
        if self.lba >= self.sectors:
            self._set_address(self.lba)
            self._abort(IDNF)
            return
        n = min(self.block, self.remaining, self.sectors - self.lba)
        self.data = bytearray(n * SECTOR)
        self.pos = 0
        self.direction = "out"
        self.block_sectors = n
        self.status = DRDY | DSC | DRQ
        if interrupt:
            self._interrupt()
        else:
            self._changed()

    def _block_written(self) -> None:
        """The host wrote the whole block: BSY, DRQ cleared, the device
        writes it to the media (9.8)."""
        self.status = BSY | DRDY | DSC
        self.direction = None
        self._changed()
        self._after(self.cycles("write"), self._block_on_media)

    def _block_on_media(self) -> None:
        n = self.block_sectors
        if self.command == 0xE8:                 # WRITE BUFFER
            self.buffer[:] = self.data
            self._complete()
            return
        self.media.write(self.lba, bytes(self.data))
        self.writes += n
        self._set_address(self.lba + n - 1)
        self.lba += n
        self.remaining -= n
        self.nsector = self.remaining & 0xFF
        if self.remaining == 0:
            self._complete()
        else:
            self._want_block(interrupt=True)

    # -- DMA (9.10; the data moves when the PIIX4's bus master runs) ---------------

    def _dma(self) -> None:
        lba = self._address()
        if lba is None or lba >= self.sectors or lba + self._count() > self.sectors:
            self._after(self.cycles("access"), lambda: self._abort(IDNF))
            return
        self.lba, self.remaining = lba, self._count()
        self.direction = "dma-in" if self.command in (0xC8, 0xC9) else "dma-out"
        self.pos = 0
        self._after(self.cycles("access") + self._spin_up(), self._dma_ready)

    def _dma_ready(self) -> None:
        """DMARQ: the data is ready / wanted; BSY stays set (9.10)."""
        if self.direction == "dma-in":
            self.data = bytearray(self.media.read(self.lba, self.remaining))
        else:
            self.data = bytearray(self.remaining * SECTOR)
        self.pos = 0
        self.status = BSY | DRDY | DSC | DRQ
        self.dmarq = True
        self._changed()

    def dma_read(self, n: int) -> bytes:
        """The bus master takes up to ``n`` bytes (device -> memory)."""
        if not self.dmarq or self.direction != "dma-in":
            return b""
        out = bytes(self.data[self.pos:self.pos + n])
        self.pos += len(out)
        if self.pos >= len(self.data):
            self._dma_done()
        return out

    def dma_write(self, data: bytes) -> int:
        """The bus master gives bytes (memory -> device); how many it took."""
        if not self.dmarq or self.direction != "dma-out":
            return 0
        n = min(len(data), len(self.data) - self.pos)
        self.data[self.pos:self.pos + n] = data[:n]
        self.pos += n
        if self.pos >= len(self.data):
            self._dma_done()
        return n

    def _dma_done(self) -> None:
        self.dmarq = False
        n = self.remaining
        if self.direction == "dma-out":
            self.media.write(self.lba, bytes(self.data))
            self.writes += n
        else:
            self.reads += n
        self._set_address(self.lba + n - 1)
        self.nsector = 0
        self.lba += n
        self.remaining = 0
        self.status = BSY | DRDY | DSC
        self._changed()
        self._after(self.cycles("write" if self.direction == "dma-out" else "command"),
                    self._complete)

    # -- non-data commands (9.9) -------------------------------------------------------

    def _verify(self) -> None:
        lba = self._address()
        if lba is None or lba >= self.sectors:
            self._after(self.cycles("access"), lambda: self._abort(IDNF))
            return
        n = self._count()
        good = min(n, self.sectors - lba)

        def done() -> None:
            self.reads += good
            if good < n:                         # the first sector beyond the disk
                self._set_address(lba + good)
                self._abort(IDNF)
                return
            self._set_address(lba + good - 1)
            self.nsector = 0
            self._complete()
        self._after(self.cycles("access") + self.cycles("sector", good) + self._spin_up(), done)

    def _seek(self) -> None:
        lba = self._address()
        if lba is None or lba >= self.sectors:
            self._after(self.cycles("access"), lambda: self._abort(IDNF))
            return
        self._after(self.cycles("access") + self._spin_up(), self._complete)

    def _diagnostic(self) -> None:
        """EXECUTE DEVICE DIAGNOSTIC (9.5): both devices run it (the
        channel calls the other); Device 0 interrupts."""
        self._run_diagnostic()
        other = self.channel.other(self) if self.channel is not None else None
        if other is not None:
            other._run_diagnostic()

    def _run_diagnostic(self) -> None:
        self.intrq = False
        self.data = bytearray()
        self.direction = None
        self.status = BSY
        if self.power == "sleep":
            self.power = "standby"

        def done() -> None:
            self._reset_done()
            if self.number == 0:
                self._interrupt()
        self._after(self.cycles("diagnostic"), done)

    def _init_params(self) -> None:
        """INITIALIZE DEVICE PARAMETERS (8.16): a new CHS translation of
        Sector Count sectors per track and the Device/Head head bits + 1
        heads; one that gives no cylinder is not supported (aborted; media
        accesses by CHS then fail with IDNF until a valid one is set)."""
        spt, heads = self.nsector, (self.select & 0x0F) + 1

        def done() -> None:
            cyls = min(self.sectors, MAX_CHS_SECTORS) // (heads * spt) if spt else 0
            if not spt or cyls == 0:
                self.chs_valid = False
                self._abort()
                return
            self.current, self.chs_valid = (min(cyls, 65535), heads, spt), True
            self._complete()
        self._after(self.cycles("command"), done)

    def _set_multiple(self) -> None:
        n = self.nsector

        def done() -> None:
            if n not in (2, 4, 8, 16) or n > MAX_MULTIPLE:
                self._abort()
                return
            self.multiple = n
            self._complete()
        self._after(self.cycles("command"), done)

    def _set_features(self) -> None:
        """SET FEATURES (8.37): transfer modes (03h, Table 20), read
        look-ahead (AAh / 55h), reverting to defaults (CCh / 66h); write
        cache (02h / 82h) and advanced power management (05h / 85h) are
        not implemented (command aborted), nor any other subcommand."""
        sub, value = self.features, self.nsector

        def done() -> None:
            if sub == 0x03:
                kind, mode = value >> 3, value & 7
                if kind == 0 and mode in (0, 1):
                    self.pio_mode = 0
                elif kind == 1 and mode <= 4:
                    self.pio_mode = mode
                elif kind == 4 and mode <= 2:
                    self.dma_mode = ("mwdma", mode)
                elif kind == 8 and mode <= 2:
                    self.dma_mode = ("udma", mode)
                else:
                    self._abort()
                    return
            elif sub in (0xAA, 0x55):
                self.look_ahead = sub == 0xAA
            elif sub in (0xCC, 0x66):
                self.revert = sub == 0xCC
            else:
                self._abort()
                return
            self._complete()
        self._after(self.cycles("command"), done)

    def _identify(self) -> None:
        def ready() -> None:
            self.lba, self.remaining, self.block = 0, 1, 1
            self.data = bytearray(b"".join(w.to_bytes(2, "little") for w in self.identify()))
            self.pos = 0
            self.direction = "in"
            self.block_sectors = 1
            self.status = DRDY | DSC | DRQ
            self._interrupt()
        self._after(self.cycles("command"), ready)

    def _read_buffer(self) -> None:
        def ready() -> None:
            self.lba, self.remaining, self.block = 0, 1, 1
            self.data = bytearray(self.buffer)
            self.pos = 0
            self.direction = "in"
            self.block_sectors = 1
            self.status = DRDY | DSC | DRQ
            self._interrupt()
        self._after(self.cycles("command"), ready)

    def _write_buffer(self) -> None:
        def ready() -> None:
            self.lba, self.remaining, self.block = 0, 1, 1
            self._want_block(interrupt=False)
        self._after(self.cycles("command"), ready)

    def _flush(self) -> None:
        self._after(self.cycles("command"), self._complete)   # no write cache

    # -- power management (6.8) ---------------------------------------------------------

    def _check_power(self) -> None:
        def done() -> None:
            self.nsector = {"standby": 0x00, "idle": 0x80}.get(self.power, 0xFF)
            self._complete()
        self._after(self.cycles("command"), done)

    def _go(self, mode: str, timer: int | None) -> None:
        def done() -> None:
            if timer is not None:
                self.standby_timer = standby_seconds(timer)
            self.power = mode
            self.last_command = self._time()
            self._complete()
        self._after(self.cycles("command"), done)

    def _standby_now(self) -> None:
        self._go("standby", None)

    def _idle_now(self) -> None:
        self._go("idle", None)

    def _standby(self) -> None:
        self._go("standby", self.nsector)

    def _idle(self) -> None:
        self._go("idle", self.nsector)

    def _sleep(self) -> None:
        """SLEEP (8.40): the interrupt; the Status read that clears it puts
        the device to sleep — then it answers nothing until a reset."""
        def done() -> None:
            self.power = "sleep-pending"
            self._complete()
        self._after(self.cycles("command"), done)

    def _standby_at(self) -> int | None:
        if self.standby_timer is None or self.power not in ("active", "idle") \
                or self.status & BSY or self.direction is not None:
            return None
        return self.last_command + int(self.standby_timer * self.hz)

    def _standby_check(self, now: int) -> None:
        at = self._standby_at()
        if at is not None and at <= now:
            self.power = "standby"
            self._log("Standby timer: Standby mode")

    @property
    def asleep(self) -> bool:
        return self.power == "sleep"

    # -- IDENTIFY DEVICE (8.12, Table 11) ----------------------------------------------

    def identify(self) -> list[int]:
        w = [0] * 256
        cyls, heads, spt = self.default
        w[0] = 0x0040                            # an ATA device, not removable
        w[1], w[3], w[6] = cyls, heads, spt
        w[10:20] = _ascii_words(self.serial, 10)
        w[23:27] = _ascii_words("eMIPS 1", 4)
        w[27:47] = _ascii_words(f"eMIPS ATA disk {size_text(self.sectors * SECTOR)}", 20)
        w[47] = 0x8000 | MAX_MULTIPLE
        w[49] = 0x2F00                           # Standby timer values, IORDY, LBA, DMA
        w[50] = 0x4000
        w[51] = 0x0200                           # PIO mode 2 timing (words 64-68: 3, 4)
        w[53] = 0x0007 if self.chs_valid else 0x0006
        if self.chs_valid:
            c, h, s = self.current
            w[54], w[55], w[56] = c, h, s
            cap = c * h * s
            w[57], w[58] = cap & 0xFFFF, cap >> 16
        w[59] = (0x0100 | self.multiple) if self.multiple else 0
        w[60], w[61] = self.sectors & 0xFFFF, self.sectors >> 16
        mw = self.dma_mode[1] if self.dma_mode and self.dma_mode[0] == "mwdma" else None
        ud = self.dma_mode[1] if self.dma_mode and self.dma_mode[0] == "udma" else None
        w[63] = 0x0007 | ((0x100 << mw) if mw is not None else 0)
        w[64] = 0x0003                           # PIO modes 3 and 4
        w[65], w[66], w[67], w[68] = 120, 120, 120, 120
        w[80] = 0x001E                           # ATA-1 .. ATA/ATAPI-4
        w[82] = 0x7048                           # NOP, READ / WRITE BUFFER, look-ahead, PM
        w[83] = w[84] = w[87] = 0x4000
        w[85] = 0x7008 | (0x0040 if self.look_ahead else 0)
        w[88] = 0x0007 | ((0x100 << ud) if ud is not None else 0)
        return w

    def describe(self) -> str:
        c, h, s = self.current
        mode = self.power.replace("sleep-pending", "active")
        state = "BSY" if self.status & BSY else ("DRQ" if self.status & DRQ else "ready")
        n = lambda k, what: f"{k} {what}{'' if k == 1 else 's'}"   # noqa: E731
        return (f"{self.name}: {n(self.sectors, 'sector')} "
                f"({size_text(self.sectors * SECTOR)}), CHS {c}/{h}/{s}, {mode}, status "
                f"0x{self.status:02x} ({state}), error 0x{self.error:02x}, "
                f"{n(self.reads, 'sector')} read, {self.writes} written")


class IDEChannel:
    """One IDE cable (primary or secondary): Device 0 and Device 1, the
    registers every device sees (7.1), the INTRQ of the selected device
    (5.2.9) on ``irq`` (IRQ 14 / 15), the 9.16 rules for one device."""

    def __init__(self, name: str, irq: int, now: Callable[[], int]) -> None:
        self.name = name
        self.irq_line = irq
        self.now = now
        self.devices: list[ATADisk | None] = [None, None]
        self.on_irq: Callable[[int, bool], None] | None = None
        self.level = False

    def attach(self, n: int, disk: ATADisk | None) -> None:
        self.devices[n] = disk
        if disk is not None:
            disk.channel = self
            disk.number = n
        self.update()

    def other(self, disk: ATADisk) -> ATADisk | None:
        return self.devices[1 - disk.number]

    def present(self) -> list[ATADisk]:
        return [d for d in self.devices if d is not None]

    def selected(self) -> int:
        """The DEV bit of the device that holds the register (both have
        the same value when both are there: 7.1)."""
        d = self.devices[0] or self.devices[1]
        return 1 if d is not None and d.select & DEV else 0

    def _responder(self, *, status: bool) -> ATADisk | None:
        """Who drives the bus on a read (9.16): the selected device; with
        Device 0 alone and Device 1 selected, Device 0 for every register
        but Status / Alternate Status (those read 00h)."""
        n = self.selected()
        d = self.devices[n]
        if d is not None:
            return None if d.asleep else d
        if n == 1 and self.devices[0] is not None and not status:
            return None if self.devices[0].asleep else self.devices[0]
        return None

    # -- the command block --------------------------------------------------------

    def read(self, reg: int, *, peek: bool = False) -> int:
        """An 8-bit command block register (offsets 1-7)."""
        status = reg == STATUS
        d = self._responder(status=status)
        if d is None:
            return 0x00 if (status and self.selected() == 1 and self.devices[0] is not None
                            and not self.devices[0].asleep) else FLOAT
        return d.peek_reg(reg) if peek else d.read_reg(reg)

    def write(self, reg: int, value: int) -> None:
        value &= 0xFF
        if reg == COMMAND:
            n = self.selected()
            d = self.devices[n]
            if d is not None:
                d.command_write(value)
            elif value == 0x90 and self.devices[1 - n] is not None:
                self.devices[1 - n].command_write(value)   # 9.16.1 (3)
            return
        for d in self.present():                 # 7.1: the registers go to both
            if not d.asleep:
                d.write_reg(reg, value)
        if reg == SELECT:
            self.update()                        # INTRQ follows the selection (5.2.9)

    def _target(self) -> ATADisk | None:
        """Who takes a write other than Command: the selected device, or
        Device 0 alone for Device 1 (9.16.1 (2))."""
        n = self.selected()
        d = self.devices[n] or (self.devices[0] if n == 1 else None)
        return None if d is None or d.asleep else d

    def read_data(self, *, peek: bool = False) -> int:
        d = self._responder(status=False)
        if d is None:
            return 0xFF00 | FLOAT                # nothing drives DD15:0
        if peek:
            w = d.peek_data()
            return 0xFFFF if w is None else w
        return d.read_data()

    def write_data(self, word: int) -> None:
        d = self._target()
        if d is not None:
            d.write_data(word)

    # -- the control block ---------------------------------------------------------

    def read_alt_status(self) -> int:
        """Alternate Status (7.3): Status without clearing the interrupt."""
        return self.read(STATUS, peek=True)

    def write_control(self, value: int) -> None:
        """Device Control (7.9): every device sees nIEN and SRST."""
        for d in self.present():
            old = d.ctl
            d.ctl = value & (NIEN | SRST)
            if (old ^ d.ctl) & SRST:
                d.software_reset(bool(d.ctl & SRST))
        self.update()

    # -- time and INTRQ --------------------------------------------------------------

    def next_due(self) -> int | None:
        t = [x for x in (d.next_due() for d in self.present()) if x is not None]
        return min(t) if t else None

    def advance(self, now: int) -> None:
        for d in self.present():
            d.advance(now)

    def update(self) -> None:
        """INTRQ (5.2.9): driven by the selected device with nIEN = 0
        (tri-stated otherwise: the host's pull-down holds it low)."""
        d = self.devices[self.selected()]
        level = bool(d is not None and d.intrq and not d.ctl & NIEN and not d.asleep)
        if level != self.level:
            self.level = level
            if self.on_irq is not None:
                self.on_irq(self.irq_line, level)

    def describe(self) -> list[str]:
        out = []
        for n, d in enumerate(self.devices):
            if d is not None:
                out.append(f"{self.name} device {n}: {d.describe()}")
        return out
