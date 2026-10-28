"""Xilinx's AXI Central DMA (PG034 v4.1, the AXI CDMA product guide): the DMA-capable device of the P8700 system (LF15).

The P8700's Boston is a Xilinx FPGA board; this is the memory-to-memory
DMA engine of its IP catalogue.  Simple mode (SA, DA, BTT: writing BTT
starts a transfer) and scatter-gather mode (CURDESC / TAILDESC; a chain of
64-byte-aligned descriptors: NXTDESC, SA, DA, CONTROL.BTT, STATUS with
Cmplt / DMADecErr / DMASlvErr / DMAIntErr written back), interrupt on
complete with the SG threshold and the delay timer, the error interrupt,
soft reset (Tables 2-3 to 2-14, 3-1 to 3-9).  Configured with a 64-bit
address space (the *_MSB registers; a write of TAILDESC_PNTR_MSB starts SG)
and 64-bit data: it moves 8 bytes a clock.

Its AXI master reaches memory through the IOCU when the cluster has one
(MD01502 5.9: coherent — the cores' caches are snooped) or else directly
(non-coherent: what the data caches hold is not seen, stale data can be
read).  An address beyond the physical space, or an MMIO region while
GCR_CONTROL bit 13 keeps the IOCU out, is a decode error.

The device sits at :data:`BASE` (the uncached MMIO page) and raises APLIC
source :data:`SOURCE`.  Choices: ``sim.DEVIATIONS["P8700 memory system"]``.
"""

from __future__ import annotations

from .rvplatform import Device

BASE, SIZE, SOURCE = 0xFFFF1000, 0x1000, 5
#: data moved a clock (C_M_AXI_DATA_WIDTH 64)
RATE = 8
#: CDMACR / CDMASR bits (Tables 2-4 / 2-5)
RESET, TAILPNTREN, SGMODE, KEYHOLE_RD, KEYHOLE_WR, CYCLIC = 1 << 2, 1 << 1, 1 << 3, 1 << 4, \
    1 << 5, 1 << 6
IOC_IRQ, DLY_IRQ, ERR_IRQ = 1 << 12, 1 << 13, 1 << 14
IDLE, SGINCLD, DMAINTERR, DMASLVERR, DMADECERR = 1 << 1, 1 << 3, 1 << 4, 1 << 5, 1 << 6
SGINTERR, SGSLVERR, SGDECERR = 1 << 8, 1 << 9, 1 << 10
CR_RW = 0xFFFF0000 | ERR_IRQ | DLY_IRQ | IOC_IRQ | CYCLIC | KEYHOLE_WR | KEYHOLE_RD | SGMODE
#: descriptor STATUS (Table 3-9)
CMPLT, D_DEC, D_SLV, D_INT = 1 << 31, 1 << 30, 1 << 29, 1 << 28
BTT_MASK = (1 << 26) - 1


class DmaError(Exception):
    """An AXI error response: "dec" (no slave there) or "slv"."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


class AxiCdma(Device):
    """The CDMA of a P8700 cluster (``cl``: emips.p8700cps.Cluster)."""

    base, size = BASE, SIZE

    def __init__(self, cl) -> None:
        self.cl = cl
        self.reset()

    def reset(self) -> None:
        self.cr = TAILPNTREN | 1 << 16           # (IRQThreshold 1)
        self.sr = IDLE | SGINCLD | 1 << 16
        self.cur = self.tail = self.sa = self.da = 0
        self.btt = 0
        self._job: dict | None = None             # the transfer in progress
        self._threshold = 1
        self._delay = 0
        self._paused = False                      # (SG: at the tail descriptor)
        self.log: list[str] = []
        self.moved = 0
        self._irq = False
        self._last = getattr(self, "_last", 0)

    # -- the register file -----------------------------------------------------------

    def reg_read(self, m, off: int, side_effects: bool = True) -> int:
        if off == 0x00:
            return self.cr
        if off == 0x04:
            return (self.sr & ~(0xFF << 24 | 0xFF << 16)) | self._delay << 24 \
                | self._threshold << 16
        regs = {0x08: self.cur & 0xFFFFFFFF, 0x0C: self.cur >> 32,
                0x10: self.tail & 0xFFFFFFFF, 0x14: self.tail >> 32,
                0x18: self.sa & 0xFFFFFFFF, 0x1C: self.sa >> 32,
                0x20: self.da & 0xFFFFFFFF, 0x24: self.da >> 32, 0x28: self.btt}
        return regs.get(off, 0)

    def reg_write(self, m, off: int, value: int, size: int) -> None:
        idle = bool(self.sr & IDLE)
        if off == 0x00:
            if value & RESET:                     # (soft reset: every register)
                self.reset()
                self._update_irq()
                return
            old = self.cr
            new = (old & ~CR_RW) | (value & CR_RW)
            if not value >> 16 & 0xFF:            # (a threshold of 0: no effect)
                new = (new & ~(0xFF << 16)) | (old & 0xFF << 16)
            else:
                self._threshold = value >> 16 & 0xFF
            if not idle:                          # (SGMode only while idle)
                new = (new & ~SGMODE) | (old & SGMODE)
            if old & SGMODE and not new & SGMODE:
                self.cur = self.tail = 0
                self.sr &= ~DLY_IRQ
            self.cr = new
            self._update_irq()
        elif off == 0x04:                         # (Err_Irq / Dly_Irq / IOC_Irq: W1C)
            self.sr &= ~(value & (ERR_IRQ | DLY_IRQ | IOC_IRQ))
            self._update_irq()
        elif off in (0x08, 0x0C):
            if idle and self.cr & SGMODE:
                self.cur = _half(self.cur, off == 0x0C, value) & ~0x3F
        elif off in (0x10, 0x14):
            if self.cr & SGMODE:
                self.tail = _half(self.tail, off == 0x14, value) & ~0x3F
                if off == 0x14:                   # (64-bit address space: the MSB starts)
                    self._sg_start()
        elif off in (0x18, 0x1C):
            self.sa = _half(self.sa, off == 0x1C, value)
        elif off in (0x20, 0x24):
            self.da = _half(self.da, off == 0x24, value)
        elif off == 0x28:
            self.btt = value & BTT_MASK
            if idle and not self.cr & SGMODE:
                self._start(self.sa, self.da, self.btt, None)

    # -- the engine -------------------------------------------------------------------

    def _start(self, sa: int, da: int, btt: int, desc: int | None) -> None:
        self.sr &= ~IDLE
        if btt == 0:                              # (Table 2-14 note 3)
            self._fail("int", desc)
            return
        self._job = {"sa": sa, "da": da, "left": btt, "done": 0, "desc": desc}
        self.log.append(f"{'SG ' if desc is not None else ''}transfer {btt} bytes "
                        f"0x{sa:x} -> 0x{da:x}")
        del self.log[:-8]

    def _sg_start(self) -> None:
        if not self.sr & IDLE and not self._paused:
            return                                # (running: the pause point moves)
        if self.sr & (SGINTERR | SGSLVERR | SGDECERR | DMAINTERR | DMASLVERR | DMADECERR):
            return                                # (halted by an error: a reset clears it)
        if self._paused:                          # (resume after the old tail)
            self._paused = False
            nxt = self._read_word(self.cur, 8)
            if nxt is None:
                return
            self.cur = nxt & ~0x3F
        self._delay = 0
        self._fetch()

    def _fetch(self) -> None:
        """Read the descriptor at CURDESC_PNTR and start its transfer."""
        d = self.cur
        try:
            words = self._read(d, 32)
        except DmaError as e:
            self.sr |= SGDECERR if e.kind == "dec" else SGSLVERR
            self._halt()
            return
        w = [int.from_bytes(words[k:k + 4], "little") for k in range(0, 32, 4)]
        status = w[7]
        if status & CMPLT and not self.cr & CYCLIC:   # (a stale descriptor)
            self.sr |= SGINTERR
            self._halt()
            return
        sa = w[2] | w[3] << 32
        da = w[4] | w[5] << 32
        self._start(sa, da, w[6] & BTT_MASK, d)

    def tick(self, m, cycles: int) -> None:
        """Every hart ticks its devices: the CDMA follows the cluster's clock
        (the Session's steps), or hart 0's when nothing drives that."""
        now = self.cl.clock
        if now:
            cycles, self._last = now - self._last, now
        elif m is not self.cl.machines[0]:
            return
        for _ in range(max(0, cycles)):
            job = self._job
            if job is None:
                if self.sr & IDLE and self.cr & SGMODE and self.cr >> 24 and \
                        self._threshold != self.cr >> 16 & 0xFF:
                    self._delay += 1               # (the delay timer: idle, events pending)
                    if self._delay >= self.cr >> 24:
                        self._delay = 0
                        self._threshold = self.cr >> 16 & 0xFF
                        self.sr |= DLY_IRQ
                        self._update_irq()
                return
            n = min(RATE, job["left"])
            src = job["sa"] + (0 if self.cr & KEYHOLE_RD else job["done"])
            dst = job["da"] + (0 if self.cr & KEYHOLE_WR else job["done"])
            try:
                self._write(dst, self._read(src, n))
            except DmaError as e:
                self._fail(e.kind, job["desc"])
                return
            job["done"] += n
            job["left"] -= n
            self.moved += n
            if not job["left"]:
                self._finish(job)

    def _finish(self, job: dict) -> None:
        self._job = None
        d = job["desc"]
        if d is None:                             # (simple mode)
            self.sr |= IDLE | IOC_IRQ
            self._update_irq()
            return
        try:                                      # (the status word written back)
            self._write(d + 0x1C, CMPLT.to_bytes(4, "little"))
        except DmaError as e:
            self.sr |= SGDECERR if e.kind == "dec" else SGSLVERR
            self._halt()
            return
        self._threshold -= 1
        if self._threshold <= 0:
            self._threshold = self.cr >> 16 & 0xFF or 1
            self.sr |= IOC_IRQ
            self._update_irq()
        if d == self.tail and not self.cr & CYCLIC:
            self._paused = True                   # (the pause point)
            self.sr |= IDLE
            return
        nxt = self._read_word(d, 8)
        if nxt is None:
            return
        self.cur = nxt & ~0x3F
        self._fetch()

    def _read_word(self, d: int, n: int) -> int | None:
        try:
            return int.from_bytes(self._read(d, n), "little")
        except DmaError as e:
            self.sr |= SGDECERR if e.kind == "dec" else SGSLVERR
            self._halt()
            return None

    def _fail(self, kind: str, desc: int | None) -> None:
        """A DataMover error: the engine halts gracefully (Table 2-5)."""
        self._job = None
        self.sr |= {"dec": DMADECERR, "slv": DMASLVERR, "int": DMAINTERR}[kind]
        if desc is not None:
            self.cur = desc
            bit = {"dec": D_DEC, "slv": D_SLV, "int": D_INT}[kind]
            try:
                self._write(desc + 0x1C, bit.to_bytes(4, "little"))
            except DmaError:
                pass
        self.log.append(f"error: DMA {kind} error")
        self._halt()

    def _halt(self) -> None:
        self._job = None
        self.sr |= IDLE | ERR_IRQ
        self._update_irq()

    def _update_irq(self) -> None:
        level = bool(self.sr & self.cr & (IOC_IRQ | DLY_IRQ | ERR_IRQ))
        if level != self._irq:
            self._irq = level
            ic = self.cl.ic
            if ic is not None:
                ic.set_input(SOURCE, level)

    # -- the AXI master: the IOCU or memory --------------------------------------------

    def _target_ok(self, pa: int, n: int) -> None:
        m = self.cl.machines[0]
        if pa < 0 or pa + n > 1 << m.pabits:
            raise DmaError("dec")
        cm = self.cl.cm
        if self.cl.iocu and self.cl.gcr.regs.get(0x10, 0) >> 13 & 1 \
                and cm.mmio(pa, True) is not None:
            raise DmaError("dec")                 # (5.9: IOCU requests kept out of MMIO)

    def _device(self, pa: int):
        m = self.cl.machines[0]
        return next((d for d in m.devices if d is not self and d.covers(pa)), None)

    def _read(self, pa: int, n: int) -> bytes:
        self._target_ok(pa, n)
        dev = self._device(pa)
        m = self.cl.machines[0]
        if dev is not None:
            v = dev.load(m, pa, n) if n in (1, 2, 4, 8) else 0
            return v.to_bytes(n, "little")
        if self.cl.iocu:
            return self.cl.cm.iocu_read(pa, n)
        return bytes(m.phys_read_byte(pa + k) for k in range(n))

    def _write(self, pa: int, data: bytes) -> None:
        n = len(data)
        self._target_ok(pa, n)
        dev = self._device(pa)
        m = self.cl.machines[0]
        if dev is not None:
            if n in (1, 2, 4, 8):
                dev.store(None if self.cl.iocu else m, pa, int.from_bytes(data, "little"), n)
            return
        if self.cl.iocu:
            self.cl.cm.iocu_write(pa, data)
            return
        for k, b in enumerate(data):
            m.phys_write_byte(pa + k, b)

    # -- views ------------------------------------------------------------------------------

    def lines(self) -> list[str]:
        state = "idle" if self.sr & IDLE else "busy"
        if self._job is not None:
            j = self._job
            state = f"busy: {j['done']} of {j['done'] + j['left']} bytes"
        mode = "scatter-gather" if self.cr & SGMODE else "simple"
        errs = [n for b, n in ((DMADECERR, "DMADecErr"), (DMASLVERR, "DMASlvErr"),
                               (DMAINTERR, "DMAIntErr"), (SGDECERR, "SGDecErr"),
                               (SGSLVERR, "SGSlvErr"), (SGINTERR, "SGIntErr")) if self.sr & b]
        via = "the IOCU (coherent)" if self.cl.iocu else "memory directly (not coherent)"
        out = [f"AXI CDMA at 0x{BASE:08x} (APLIC source {SOURCE}): {mode} mode, {state}; "
               f"through {via}; {self.moved} bytes moved",
               f"  CDMACR 0x{self.cr:08x}, CDMASR 0x{self.reg_read(None, 4):08x}"
               + (f" ({', '.join(errs)})" if errs else "")
               + f", SA 0x{self.sa:x}, DA 0x{self.da:x}, BTT {self.btt}"
               + (f", CURDESC 0x{self.cur:x}, TAILDESC 0x{self.tail:x}"
                  if self.cr & SGMODE else "")]
        out += [f"  {line}" for line in self.log[-3:]]
        return out


def _half(old: int, high: bool, value: int) -> int:
    value &= 0xFFFFFFFF
    return (old & 0xFFFFFFFF) | value << 32 if high else (old & ~0xFFFFFFFF) | value
