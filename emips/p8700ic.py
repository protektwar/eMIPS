"""The MIPS P8700's interrupt controller (PLAN M25 phase 5d2; MD01502 rev.
1.84 ch. 8; QEMU's Boston P8700 board for what the manual leaves to the
configuration: 53 APLIC sources, 3 priority bits, the UART on source 4).

:class:`Controller` sits in the cluster's GCR block (emips.p8700cps), its
hart index = mhartid[11:0] = CORENUM << 4 | HARTNUM (8.1.2 / 8.1.3.2):

- **ACLINT machine region** (GCR_BASE + 0x50000, 8.2.1): MSIP[i] (+ 4 i: bit
  0, a machine software interrupt), MTIMECMP[i] (+ 0x4000 + 8 i; reset all
  ones — the MIPS choice of 8.2.1.2), compared with the CPC's MTIME_REG
  (8.1.3.1: mtime lives in the CPC), WDCFG[i] (+ 0xC000 + 4 i) and
  WDCSR[i] (+ 0xD000 + 4 i): the watchdog of 8.1.4.
- **ACLINT supervisor region** (+ 0x6C000, 8.2.2): SETSSIP[i] (writing 1
  sets the hart's mip.SSIP; reads 0).
- **The APLIC** (8.1.2, 8.3; emips.rvplatform's direct-mode domains):
  the machine domain at + 0x40000, the supervisor domain at + 0x60000, the
  custom block at + 0x4C000 — setnmie / setnmienum / clrnmie / clrnmienum
  (8.3.3.18-21): a root-domain source with nmie = 1 and its interrupt
  enable 0 signals an NMI to target[k]'s hart when it becomes pending
  (IPRIO ignored).
- **The watchdog** (8.1.4, 8.2.1.3-4): enabled by WDCSR.Enable; a write of
  WDCSR loads the counter with WTOCNT; it counts down when bit 8 (WDFRQ + 1)
  of mtime rises; at zero the stage-1 event (S1WTO set, the counter
  reloaded), at zero again with S1WTO the stage-2 event (S2WTO).  Events: 0
  interrupt (mip bit 25), 1 NMI, 2 a reset of the cluster's harts, 3 the
  SoC pin (no SoC: nothing), 4 (stage 1) an interval timer.

Outputs go to the harts as the standard lines (MSIP 3, MTIP 7, SEIP 9,
MEIP 11), which the P8700's mipsintctl routes (emips.p8700), plus mip bit
25 and the NMI input (``RVMachine.nmi``).  Choices:
``sim.DEVIATIONS["P8700 interrupt controller"]``.
"""

from __future__ import annotations

from typing import Any

from .rvplatform import APLIC, MEI, MSI, MTI, SEI, SSI, Domain

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
#: offsets in the GCR block (Table 5.24)
APLIC_M, APLIC_CUSTOM, ACLINT_M, APLIC_S, ACLINT_S = (0x40000, 0x4C000, 0x50000, 0x60000,
                                                     0x6C000)
SIZES = {APLIC_M: 0xC000, APLIC_CUSTOM: 0x1000, ACLINT_M: 0x10000, APLIC_S: 0xC000,
         ACLINT_S: 0x4000}
#: QEMU's Boston P8700 (hw/riscv/cps.h, boston-aia.c)
SOURCES, IPRIO_BITS, UART_SOURCE = 0x35, 3, 4
#: the watchdog's interrupt (8.2.1.3: "bit 25 of the mip CSR")
WDT_IRQ = 25
INTERRUPT, NMI, RESET, TOPLEVEL, INTERVAL = 0, 1, 2, 3, 4


class _Domain(Domain):
    """An APLIC domain at its place in the GCR block (moves with GCR_BASE)."""

    IPRIOLEN = IPRIO_BITS

    def __init__(self, aplic, off: int, level: int, child) -> None:
        super().__init__(aplic, 0, level, child)
        self.off = off
        self.size = SIZES[off]

    @property
    def base(self) -> int:                         # type: ignore[override]
        return self.a.p.cl.base + self.off

    @base.setter
    def base(self, v: int) -> None:
        pass


class _APLIC(APLIC):
    """The APLIC with the P8700's domains, NMI routing and sparse hart
    indices."""

    def __init__(self, ic: "Controller") -> None:
        self.p = ic
        self.n = SOURCES
        self.wire = [False] * (SOURCES + 1)
        self.s = _Domain(self, APLIC_S, 1, None)
        self.m = _Domain(self, APLIC_M, 3, self.s)
        self.nmie = [False] * (SOURCES + 1)
        self._nmi_sent = [False] * (SOURCES + 1)

    def update(self) -> None:
        ic = self.p
        m_dom = self.m
        for s in range(1, self.n + 1):          # (8.1.2: NMI routing, the root only)
            if self.nmie[s] and not m_dom.enabled[s] and m_dom.active(s) \
                    and m_dom.pending[s]:
                if not self._nmi_sent[s]:
                    self._nmi_sent[s] = True
                    hart = ic.hart(m_dom.target[s] >> 18)
                    if hart is not None:
                        hart.nmi_pending = True
            else:
                self._nmi_sent[s] = False
        for h, hart in ic.indexed():
            hart.set_interrupt(MEI, self.m.signal(h))
            if 1 in hart.modes:
                hart.set_interrupt(SEI, self.s.signal(h))


class Controller:
    """See the module docstring."""

    def __init__(self, cl: Any) -> None:
        self.cl = cl
        n = 1 << 12
        self.msip: dict[int, int] = {}
        self.mtimecmp: dict[int, int] = {}
        self.wdcfg: dict[int, int] = {}
        self.wdcsr: dict[int, int] = {}
        self.wdcount: dict[int, int] = {}
        self._wd_bits: dict[int, int] = {}
        self.max_index = n
        self.aplic = _APLIC(self)
        self._mtime = 0
        self._next_mtip = M64               # (the smallest MTIMECMP: no compare below it)

    # -- the harts -------------------------------------------------------------------

    def indexed(self) -> list[tuple[int, Any]]:
        """(hart index, machine) of every hart (mhartid[11:0])."""
        cl = self.cl
        return [(c << 4 | h, cl.machines[c * cl.harts + h])
                for c in range(cl.cores) for h in range(cl.harts)]

    @property
    def harts(self) -> list:
        """The APLIC's hart table (index = hart index; None: no hart)."""
        top = max(i for i, _ in self.indexed()) + 1
        out: list = [None] * top
        for i, m in self.indexed():
            out[i] = m
        return out

    def hart(self, index: int):
        c, h = index >> 4, index & 0xF
        cl = self.cl
        if c < cl.cores and h < cl.harts:
            return cl.machines[c * cl.harts + h]
        return None

    def exists(self, index: int) -> bool:
        return self.hart(index) is not None

    # -- the bus (through the GCR) ----------------------------------------------------------

    def covers(self, off: int) -> bool:
        return any(o <= off < o + s for o, s in SIZES.items())

    def load(self, m, off: int, size: int) -> int:
        word = off & ~3
        if size == 8:
            return self._get(m, word) | self._get(m, word + 4) << 32
        return (self._get(m, word) >> (8 * (off & 3))) & ((1 << (8 * size)) - 1)

    def store(self, m, off: int, value: int, size: int) -> None:
        if size == 8 and not off & 7:
            self._put(m, off, value & M32, 4)
            self._put(m, off + 4, value >> 32 & M32, 4)
            return
        word, shift = off & ~3, 8 * (off & 3)
        if size < 4:                         # (a part of a word: merged)
            mask = ((1 << (8 * size)) - 1) << shift
            value = (self._get(m, word, side_effects=False) & ~mask) | (value << shift) & mask
        self._put(m, word, value & M32, size)

    def _get(self, m, off: int, side_effects: bool = True) -> int:
        if APLIC_M <= off < APLIC_M + SIZES[APLIC_M]:
            return self.aplic.m.reg_read(m, off - APLIC_M, side_effects)
        if APLIC_S <= off < APLIC_S + SIZES[APLIC_S]:
            return self.aplic.s.reg_read(m, off - APLIC_S, side_effects)
        if APLIC_CUSTOM <= off < APLIC_CUSTOM + SIZES[APLIC_CUSTOM]:
            r = off - APLIC_CUSTOM
            if r < 0x80:                     # setnmie[k]: the NMI enables
                return self._nmi_bits(r // 4)
            return 0
        if ACLINT_M <= off < ACLINT_M + SIZES[ACLINT_M]:
            r = off - ACLINT_M
            if r < 0x4000:
                i = r // 4
                return self.msip.get(i, 0) if self.exists(i) else 0
            if r < 0xC000:
                i, half = divmod(r - 0x4000, 8)
                if not self.exists(i):
                    return 0
                return (self.mtimecmp.get(i, M64) >> (8 * half)) & M32
            if r < 0xD000:
                i = (r - 0xC000) // 4
                return self.wdcfg.get(i, 0) if self.exists(i) else 0
            if r < 0xE000:
                i = (r - 0xD000) // 4
                return self.wdcsr.get(i, 0) if self.exists(i) else 0
        return 0                             # (SETSSIP and the rest read 0)

    def _nmi_bits(self, w: int) -> int:
        a = self.aplic
        return sum(1 << b for b in range(32) if 0 < 32 * w + b <= a.n and a.nmie[32 * w + b])

    def _put(self, m, off: int, value: int, size: int) -> None:
        if APLIC_M <= off < APLIC_M + SIZES[APLIC_M]:
            self.aplic.m.reg_write(m, off - APLIC_M, value, 4 if size >= 4 else size)
            return
        if APLIC_S <= off < APLIC_S + SIZES[APLIC_S]:
            self.aplic.s.reg_write(m, off - APLIC_S, value, 4 if size >= 4 else size)
            return
        if APLIC_CUSTOM <= off < APLIC_CUSTOM + SIZES[APLIC_CUSTOM]:
            self._nmi_write(off - APLIC_CUSTOM, value)
            return
        if ACLINT_S <= off < ACLINT_S + SIZES[ACLINT_S]:
            i = (off - ACLINT_S) // 4
            hart = self.hart(i)
            if hart is not None and value & 1:
                hart.csrs.mip_sw |= 1 << SSI
            return
        if ACLINT_M <= off < ACLINT_M + SIZES[ACLINT_M]:
            r = off - ACLINT_M
            if r < 0x4000:
                i = r // 4
                if self.exists(i):
                    self.msip[i] = value & 1
                    self.hart(i).set_interrupt(MSI, bool(value & 1))
            elif r < 0xC000:
                i, half = divmod(r - 0x4000, 8)
                if self.exists(i):
                    sh = 8 * half
                    old = self.mtimecmp.get(i, M64)
                    self.mtimecmp[i] = (old & ~(M32 << sh)) | value << sh
                    self._next_mtip = min(self.mtimecmp.values())
                    self.timers(force=True)
            elif r < 0xD000:
                i = (r - 0xC000) // 4
                if self.exists(i):
                    self.wdcfg[i] = value & 0x3FF
            elif r < 0xE000:
                i = (r - 0xD000) // 4
                if self.exists(i):
                    self.wdcsr[i] = value & 0x3FFD
                    self.wdcount[i] = (value >> 4) & 0x3FF       # (a kick: WTOCNT)
                    self._wd_bits[i] = self._wd_bit(i)

    def _nmi_write(self, r: int, value: int) -> None:
        a = self.aplic
        if r < 0x80:                         # setnmie[k]
            for b in range(32):
                s = 32 * (r // 4) + b
                if value >> b & 1 and 0 < s <= a.n:
                    a.nmie[s] = True
        elif r == 0xDC:                      # setnmienum
            if 0 < value <= a.n:
                a.nmie[value] = True
        elif 0x100 <= r < 0x180:             # clrnmie[k]
            for b in range(32):
                s = 32 * ((r - 0x100) // 4) + b
                if value >> b & 1 and 0 < s <= a.n:
                    a.nmie[s] = False
        elif r == 0x1DC:                     # clrnmienum
            if 0 < value <= a.n:
                a.nmie[value] = False
        a.update()

    # -- inputs and time ----------------------------------------------------------------------

    def set_input(self, src: int, level: bool) -> None:
        self.aplic.set_input(src, level)

    def timers(self, force: bool = False) -> None:
        """MTIP = mtime >= MTIMECMP (8.2.1.2); the watchdogs."""
        now = self.cl.mtime()
        if now == self._mtime and not force:
            return
        self._mtime = now
        if now >= self._next_mtip or force:
            for i, hart in self.indexed():
                hart.set_interrupt(MTI, now >= self.mtimecmp.get(i, M64))
        for i in list(self.wdcsr):
            if self.wdcsr[i] & 1:
                self._watchdog(i)

    def _wd_bit(self, i: int) -> int:
        b = 8 * (((self.wdcfg.get(i, 0) >> 8) & 3) + 1)
        return (self.cl.mtime() >> b) if b < 64 else 0

    def _watchdog(self, i: int) -> None:
        """8.1.4: one count down per rise of mtime bit 8 (WDFRQ + 1)."""
        now = self._wd_bit(i)
        rises = now - self._wd_bits.get(i, now)
        self._wd_bits[i] = now
        for _ in range(max(rises, 0)):
            if self.wdcount.get(i, 0):
                self.wdcount[i] -= 1
            if self.wdcount.get(i, 0) == 0:
                self._timeout(i)

    def _timeout(self, i: int) -> None:
        cfg, csr = self.wdcfg.get(i, 0), self.wdcsr.get(i, 0)
        s1 = cfg & 0xF
        reload = (csr >> 4) & 0x3FF
        if s1 == INTERVAL:                   # (an interval timer: no stage 2)
            self.wdcount[i] = reload
            self._event(i, INTERRUPT)
            return
        if not csr & 4:                      # stage 1
            self.wdcsr[i] = csr | 4
            self.wdcount[i] = reload
            self._event(i, s1)
        elif not csr & 8:                    # stage 2
            self.wdcsr[i] = csr | 8
            self._event(i, (cfg >> 4) & 0xF)

    def _event(self, i: int, ev: int) -> None:
        hart = self.hart(i)
        if hart is None:
            return
        if ev == INTERRUPT:
            hart.set_interrupt(WDT_IRQ, True)
        elif ev == NMI:
            hart.nmi_pending = True
        elif ev == RESET:
            for c in range(self.cl.cores):
                self.cl.reset_core(c)
        # (TOPLEVEL: the SoC's pin — no SoC here)

    def update(self) -> None:
        self.aplic.update()

    # -- views ----------------------------------------------------------------------------------

    def lines(self) -> list[str]:
        a = self.aplic
        out = ["P8700 interrupt controller (APLIC 53 sources, direct delivery; ACLINT):"]
        for i, hart in self.indexed():
            cmp_ = self.mtimecmp.get(i, M64)
            wd = self.wdcsr.get(i, 0)
            out.append(f"  hart index 0x{i:x}: MSIP {self.msip.get(i, 0)}, MTIMECMP "
                       f"{'-' if cmp_ == M64 else cmp_}, M {int(a.m.signal(i))} / S "
                       f"{int(a.s.signal(i))}"
                       f"{f', watchdog {self.wdcount.get(i, 0)} left (WDCSR 0x{wd:x})' if wd & 1 else ''}")
        on = [s for s in range(1, a.n + 1) if a.m.sourcecfg[s] or a.s.sourcecfg[s]]
        for s in on:
            d = a.owner(s)
            out.append(f"  source {s}: {'M' if d is a.m else 'S'} mode {d.mode(s)}, "
                       f"{'pending ' if d.pending[s] else ''}"
                       f"{'enabled ' if d.enabled[s] else ''}"
                       f"{'NMI ' if a.nmie[s] else ''}-> hart 0x{d.target[s] >> 18:x}")
        return out
