"""The PIIX4's bus master IDE function (PLAN M15 phase 4).

Reference: the PIIX4 datasheet (Intel 290562-001): 5.1.3 PCICMD.BME,
5.1.4 PCISTS.MAS, 5.1.9 BMIBA (16 bytes of I/O space), 5.2.1 BMICX (Start
/ Stop, Read / Write Control), 5.2.2 BMISX (Active, DMA Error, IDE
Interrupt Status, the DMA-capable bits; Table 15), 5.2.3 BMIDTPX, 9.4 the
Physical Region Descriptor table and the operation steps; ATA/ATAPI-4
9.10 (the device's side: DMARQ, the interrupt at the end) and its
multiword / Ultra DMA cycle times (Tables 32 / 34) for the rate.

A channel runs while Start / Stop is 1 and PCICMD.BME allows it to
master the PCI bus: it reads a PRD (8 bytes: the region's address, its
byte count, EOT in bit 31 of the second word), moves the region between
memory and the device whenever the device asserts DMARQ for the
direction Read / Write Control asks, then reads the next PRD; after the
EOT region it clears Active.  Memory is reached through the GT-64120's
PCI target interface (``MaltaBoard.dma_access``): not coherent with the
CPU's caches.  A master abort (no BAR decodes the address) sets DMA Error
and PCISTS.MAS and stops the channel; IDE Interrupt Status latches every
rising edge of the channel's interrupt line.
"""

from __future__ import annotations

from typing import Callable

from . import ata as ATA

# BMICX / BMISX bits (5.2.1 / 5.2.2)
SSBM, RWCON = 0x01, 0x08
ACTIVE, DMA_ERROR, IDEINTS, CAP0, CAP1 = 0x01, 0x02, 0x04, 0x20, 0x40
#: bytes per second of each transfer mode (ATA-4: multiword DMA cycle
#: times 480 / 150 / 120 ns per 16-bit word; Ultra DMA 16.7 / 25 / 33.3 MB/s)
RATES = {("mwdma", 0): 2 * 1_000_000_000 // 480, ("mwdma", 1): 2 * 1_000_000_000 // 150,
         ("mwdma", 2): 2 * 1_000_000_000 // 120, ("udma", 0): 16_666_666,
         ("udma", 1): 25_000_000, ("udma", 2): 33_333_333}
#: the most a transfer step moves (the PIIX4's line buffer is 16 dwords:
#: 64 bytes per burst; eMIPS moves up to a sector per event)
CHUNK = ATA.SECTOR


class BusMasterChannel:
    """One channel's registers and engine (primary or secondary)."""

    def __init__(self, name: str, channel: ATA.IDEChannel, hz: int) -> None:
        self.name = name
        self.channel = channel
        self.hz = hz
        #: PCI memory: ``dma(addr, n, data=None)`` -> bytes / b"" / "abort" / "retry"
        self.dma: Callable[..., bytes | str] | None = None
        #: would a PCI memory access at ``addr`` be taken? "ok" / "abort" / "retry"
        self.target: Callable[[int], str] = lambda addr: "ok"
        #: may the function master the bus (PCICMD.BME)?
        self.bme: Callable[[], bool] = lambda: True
        #: the UDMACTL enable bit of drive n (5.1.12)
        self.udma_enabled: Callable[[int], bool] = lambda n: False
        #: a master abort happened (-> PCISTS.MAS)
        self.on_master_abort: Callable[[], None] | None = None
        self.now: Callable[[], int] = lambda: 0
        self.reset()

    def reset(self) -> None:
        self.cmd = 0
        self.status = 0
        self.dtp = 0
        self.prd: tuple[int, int, bool] | None = None   # (address, bytes left, EOT)
        self.prd_index = 0
        self.done_bytes = 0
        self.due: int | None = None
        self.retrying = False
        self.level = False
        self._last_eot = False
        self._burst: int | None = None              # when the chunk on the cable is done
        self.log: list[str] = []

    # -- registers (byte offsets 0-7 of the channel's block) ------------------------------

    def read(self, off: int) -> int:
        if off == 0:
            return self.cmd
        if off == 2:
            return self.status
        if 4 <= off <= 7:
            return self.dtp >> 8 * (off - 4) & 0xFF
        return 0                                    # reserved

    def write(self, off: int, value: int) -> None:
        value &= 0xFF
        if off == 0:
            old = self.cmd
            if self.status & ACTIVE and old & SSBM:  # RWCON fixed while running
                value = value & ~RWCON | old & RWCON
            self.cmd = value & (SSBM | RWCON)
            if self.cmd & SSBM and not old & SSBM:
                self._start()
            elif old & SSBM and not self.cmd & SSBM:
                self._stop()
        elif off == 2:
            self.status = (self.status & ~(CAP0 | CAP1) | value & (CAP0 | CAP1)) \
                & ~(value & (DMA_ERROR | IDEINTS))
        elif 4 <= off <= 7:
            shift = 8 * (off - 4)
            self.dtp = (self.dtp & ~(0xFF << shift) | value << shift) & 0xFFFFFFFC

    # -- the engine (9.4) ----------------------------------------------------------------------

    def _start(self) -> None:
        """Start: Active, the first PRD is read (step 4)."""
        self.status |= ACTIVE
        self.prd = None
        self.prd_index = 0
        self.done_bytes = 0
        self.retrying = False
        self._burst = None
        self.log.append(f"start: {'write to' if self.cmd & RWCON else 'read from'} memory, "
                        f"PRD table 0x{self.dtp:08x}")
        del self.log[:-16]
        self.due = self.now()

    def _stop(self) -> None:
        """Stop (5.2.1): the state is lost; Active clears."""
        if self.status & ACTIVE:
            self.log.append(f"stopped after {self.done_bytes} bytes")
        self.status &= ~ACTIVE
        self.prd = None
        self.due = None
        self.retrying = False
        self._burst = None

    def _device(self) -> ATA.ATADisk | None:
        return self.channel.devices[self.channel.selected()]

    def _wants(self) -> bool:
        """The selected device asks for DMA in the direction set (DMARQ)."""
        d = self._device()
        if d is None or not d.dmarq:
            return False
        return d.direction == ("dma-in" if self.cmd & RWCON else "dma-out")

    def next_due(self) -> int | None:
        if not self.status & ACTIVE or self.retrying or not self.bme():
            return None
        if self.prd is not None and not self._wants():
            return None                              # waits for DMARQ (the device's event)
        return self.due if self.due is not None else self.now()

    def advance(self, now: int) -> None:
        """Run the channel up to cycle ``now``."""
        while True:
            t = self.next_due()
            if t is None or t > now:
                return
            self.due = None
            self._step(t)

    def _abort(self, what: str) -> None:
        self.status = self.status & ~ACTIVE | DMA_ERROR
        self.prd = None
        self._burst = None
        self.log.append(f"master abort ({what}): DMA Error, stopped")
        if self.on_master_abort is not None:
            self.on_master_abort()

    def _step(self, t: int) -> None:
        t = max(t, self.now())                       # never before now (DMARQ came late)
        if self.prd is None:                         # read the next PRD (8 bytes)
            if self.prd_index and self._last_eot:
                self.status &= ~ACTIVE               # the table is done (9.4 step 6)
                self.log.append(f"PRD table done: {self.done_bytes} bytes")
                return
            at = (self.dtp + 8 * self.prd_index) & 0xFFFFFFFF
            raw = self.dma(at, 8) if self.dma is not None else "abort"
            if raw == "retry":
                self.retrying = True
                self.log.append(f"PRD at 0x{at:08x}: retried by the target (no device)")
                return
            if not isinstance(raw, bytes):
                self._abort(f"PRD at 0x{at:08x}")
                return
            base = int.from_bytes(raw[0:4], "little") & 0xFFFFFFFE
            count = int.from_bytes(raw[4:6], "little") & 0xFFFE or 0x10000
            eot = bool(raw[7] & 0x80)
            self.prd = (base, count, eot)
            self._last_eot = eot
            self.prd_index += 1
            self.log.append(f"PRD {self.prd_index - 1}: 0x{base:08x}, {count} bytes"
                            + (", EOT" if eot else ""))
            self.due = t
            return
        if not self._wants():
            self._burst = None
            return
        base, left, eot = self.prd
        d = self._device()
        n = min(left, CHUNK)
        if self._burst is None:                      # the chunk takes its time on the cable
            self._burst = t + max(1, n * self.hz // self.rate())
            self.due = self._burst
            return
        self._burst = None
        where = self.target(base)                    # the line buffer is small: the
        if where == "retry":                         # first burst finds out
            self.retrying = True
            self.log.append(f"0x{base:08x}: retried by the target (no device)")
            return
        if where != "ok":
            self._abort(f"0x{base:08x}")
            return
        if self.cmd & RWCON:                         # device -> memory
            data = d.dma_read(n)
            res = self.dma(base, len(data), data) if self.dma is not None else "abort"
        else:                                        # memory -> device
            got = self.dma(base, n) if self.dma is not None else "abort"
            res = got
            if isinstance(got, bytes):
                taken = d.dma_write(got)
                data = got[:taken]
        if res == "retry":
            self.retrying = True
            self.log.append(f"0x{base:08x}: retried by the target (no device)")
            return
        if not isinstance(res, bytes):
            self._abort(f"0x{base:08x}")
            return
        moved = len(data)
        self.done_bytes += moved
        left -= moved
        self.prd = (base + moved, left, eot) if left else None
        self.due = t

    def rate(self) -> int:
        """Bytes per second: the device's DMA mode (Ultra DMA only with
        the drive's UDMACTL bit)."""
        d = self._device()
        mode = d.dma_mode if d is not None and d.dma_mode else ("mwdma", 0)
        if mode[0] == "udma" and not self.udma_enabled(d.number if d else 0):
            mode = ("mwdma", 2)
        return RATES.get(mode, RATES[("mwdma", 0)])

    # -- the interrupt (5.2.2) -----------------------------------------------------------

    def irq_changed(self, level: bool) -> None:
        """The channel's interrupt line: a rising edge sets IDE Interrupt
        Status (it stays 0 when cleared while the line is still high)."""
        if level and not self.level:
            self.status |= IDEINTS
        self.level = level

    def describe(self) -> str:
        state = ("active" if self.status & ACTIVE else "idle") + \
            (", retried" if self.retrying else "")
        return (f"{self.name} bus master: {state}, command 0x{self.cmd:02x} "
                f"({'write to' if self.cmd & RWCON else 'read from'} memory), status "
                f"0x{self.status:02x}, PRD table 0x{self.dtp:08x}, {self.done_bytes} bytes moved")
