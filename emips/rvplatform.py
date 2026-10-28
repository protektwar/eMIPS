"""The RISC-V platform's timer and interrupt controllers (PLAN M25 phase
2d): the ACLINT (RISC-V Advanced Core Local Interruptor 1.0-rc4: MTIMER,
MSWI, SSWI), the PLIC (RISC-V Platform-Level Interrupt Controller 1.0.0)
and the APLIC (The RISC-V Advanced Interrupt Architecture 1.0 ch. 4,
direct delivery mode).

:class:`Platform` puts them on a set of harts (:class:`emips.simrv.
RVMachine`) at Spike's / QEMU virt's addresses (:data:`MAP`), so programs
written for those run unchanged:

- **CLINT** at 0x02000000 — the ACLINT's MSWI (msip, offset 0) and
  MTIMER (mtimecmp at 0x4000, mtime at 0xBFF8): the SiFive CLINT layout
  the ACLINT keeps (1.1).  mtime counts every ``timer_div`` cycles of hart
  0 (the platform's fixed frequency; ``time`` reads it, priv. 3.2.1) and
  is writable; MTIP = mtime >= mtimecmp (2.3).  The **SSWI** (4) at
  0x02F00000 (QEMU virt's): a write of 1 to SETSSIP sets the hart's
  mip.SSIP; it reads 0.
- **PLIC** at 0x0C000000 (with ``controller="plic"``, the default): 31
  sources, priorities of 4 bits (0 = never, ties to the lower ID), the
  pending array, per-context enables, thresholds and claim / complete
  (chs. 4-8); contexts 2h = hart h's M-mode, 2h + 1 its S-mode
  (Spike's); gateways (1.2) level- or edge-triggered by source
  (:meth:`PLIC.set_input`): one request until completion.  Outputs: MEIP /
  SEIP.
- **APLIC** (``controller="aplic"``): a machine-level root domain at
  0x0C000000 and its supervisor-level child at 0x0D000000 (Spike's),
  direct delivery (DM, BE read-only 0: no IMSIC), 31 sources with
  the source modes of Table 4.2, delegation to the child (D), setip /
  in_clrip / setie / clrie (and by number), the pending rules of 4.7,
  target (hart index, IPRIO of 8 bits: smaller = higher, 0 written as 1),
  IDC structures per hart (idelivery, iforce, ithreshold, topi, claimi).
  Outputs: MEIP from the root, SEIP from the child; the top's priority to
  the hart (``m.ext_priority``: the AIA's Tables 5.3 / 5.5).
- **IMSICs** (``controller="imsic"``, LF9; AIA chs. 3 / 4.9): an IMSIC per
  hart (emips.rvaia) and the APLIC in **MSI delivery mode** (DM read-only
  1, as Spike: the harts' files have no eidelivery 0x40000000): an active
  source pending and enabled with the domain's IE is forwarded as an MSI —
  a 32-bit little-endian store of target.EIID to the address of 4.9.1
  (mmsiaddrcfg / mmsiaddrcfgh / smsiaddrcfg / smsiaddrcfgh: the IMSIC
  layout at reset, writable until L) — and its pending bit cleared;
  genmsi sends one at once (Busy never seen); 4.7's MSI-mode pending
  rules for level-sensitive sources.

All registers are 32-bit (mtime / mtimecmp: 64-bit, accessible as two
words); accesses of other sizes to the PLIC / APLIC are ignored (AIA 4.5:
"must otherwise ignore").  Choices: ``sim.DEVIATIONS["RISC-V"]``.
"""

from __future__ import annotations

from typing import Iterable

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

#: the platform's addresses (Spike's riscv/platform.h; SSWI: QEMU virt)
MAP = {"clint": 0x02000000, "sswi": 0x02F00000, "plic": 0x0C000000,
       "aplic_m": 0x0C000000, "aplic_s": 0x0D000000}
CLINT_SIZE = 0xC000
PLIC_SIZE = 0x1000000
APLIC_SIZE = 0x8000
SOURCES = 31                        # (Spike's PLIC_NDEV)
MSI, MTI, SSI, MEI, SEI = 3, 7, 1, 11, 9


class Device:
    """A memory-mapped device of a hart (sim.Machine.devices)."""

    base = 0
    size = 0

    def covers(self, pa: int) -> bool:
        return self.base <= pa < self.base + self.size

    def tick(self, m, cycles: int) -> None:
        pass

    # byte-wise access to registers (``reg_read(offset)`` / ``reg_write``
    # of 32-bit words; 64-bit registers as two words)
    def load(self, m, pa: int, size: int) -> int:
        off = pa - self.base
        word = off & ~3
        if size == 8:
            return self.reg_read(m, word) | self.reg_read(m, word + 4) << 32
        v = self.reg_read(m, word)
        return (v >> (8 * (off & 3))) & ((1 << (8 * size)) - 1)

    def store(self, m, pa: int, value: int, size: int) -> None:
        off = pa - self.base
        word = off & ~3
        if size == 8:
            self.reg_write(m, word, value & M32, 4)
            self.reg_write(m, word + 4, value >> 32 & M32, 4)
            return
        if size == 4:
            self.reg_write(m, word, value & M32, 4)
            return
        shift = 8 * (off & 3)
        mask = ((1 << (8 * size)) - 1) << shift
        old = self.reg_read(m, word, side_effects=False)
        self.reg_write(m, word, (old & ~mask) | ((value << shift) & mask), size)

    def reg_read(self, m, off: int, side_effects: bool = True) -> int:
        return 0

    def reg_write(self, m, off: int, value: int, size: int) -> None:
        pass


# ---------------------------------------------------------------- the ACLINT

class CLINT(Device):
    """The ACLINT's MSWI + MTIMER in the SiFive CLINT layout (see the module
    docstring)."""

    def __init__(self, platform: "Platform") -> None:
        self.p = platform
        self.base = MAP["clint"]
        self.size = CLINT_SIZE
        n = len(platform.harts)
        self.msip = [0] * n
        #: MTIMECMP resets to an unknown value (2.3): eMIPS's = all ones
        #: (no timer interrupt until software sets it)
        self.mtimecmp = [M64] * n
        self._base_time = 0
        self._base_cycles = 0
        self.timer_div = platform.timer_div

    # -- time ------------------------------------------------------------------------

    def clock(self) -> int:
        return self.p.harts[0].cycles

    def mtime(self) -> int:
        return (self._base_time + (self.clock() - self._base_cycles) // self.timer_div) & M64

    def set_mtime(self, v: int) -> None:
        self._base_time = v & M64
        self._base_cycles = self.clock()
        self.update()

    def update(self) -> None:
        now = self.mtime()
        for h, m in enumerate(self.p.harts):
            m.set_interrupt(MTI, now >= self.mtimecmp[h])
            m.set_interrupt(MSI, bool(self.msip[h]))

    def tick(self, m, cycles: int) -> None:
        if m is self.p.harts[0]:
            now = self.mtime()
            for h, hart in enumerate(self.p.harts):
                hart.set_interrupt(MTI, now >= self.mtimecmp[h])

    # -- registers -------------------------------------------------------------------

    def reg_read(self, m, off: int, side_effects: bool = True) -> int:
        n = len(self.p.harts)
        if off < 0x4000:
            h = off // 4
            return self.msip[h] if h < n else 0
        if off < 0xBFF8:
            h, half = divmod(off - 0x4000, 8)
            return (self.mtimecmp[h] >> (8 * half)) & M32 if h < n else 0
        if off < 0xC000:
            return (self.mtime() >> (8 * (off - 0xBFF8))) & M32
        return 0

    def reg_write(self, m, off: int, value: int, size: int) -> None:
        n = len(self.p.harts)
        if off < 0x4000:
            h = off // 4
            if h < n:
                self.msip[h] = value & 1        # (31 upper bits wired to 0: 3.2)
        elif off < 0xBFF8:
            h, half = divmod(off - 0x4000, 8)
            if h < n:
                sh = 8 * half
                self.mtimecmp[h] = (self.mtimecmp[h] & ~(M32 << sh)) | (value & M32) << sh
        elif off < 0xC000:
            sh = 8 * (off - 0xBFF8)
            self.set_mtime((self.mtime() & ~(M32 << sh)) | (value & M32) << sh)
        self.update()


class SSWI(Device):
    """The ACLINT's SSWI (4): SETSSIP sets the hart's mip.SSIP (an edge);
    it reads 0."""

    def __init__(self, platform: "Platform") -> None:
        self.p = platform
        self.base = MAP["sswi"]
        self.size = 0x4000

    def reg_write(self, m, off: int, value: int, size: int) -> None:
        h = off // 4
        if h < len(self.p.harts) and value & 1:
            self.p.harts[h].csrs.mip_sw |= 1 << SSI


# ---------------------------------------------------------------- the PLIC

class PLIC(Device):
    """PLIC 1.0 (see the module docstring)."""

    PRIO_BITS = 4

    def __init__(self, platform: "Platform", sources: int = SOURCES) -> None:
        self.p = platform
        self.base = MAP["plic"]
        self.size = PLIC_SIZE
        self.n = sources
        self.priority = [0] * (sources + 1)
        self.pending = [False] * (sources + 1)
        self.claimed = [False] * (sources + 1)       # in flight: the gateway waits
        self.level = [False] * (sources + 1)         # the input wires
        self.edge = [False] * (sources + 1)          # (gateway kind per source)
        self.contexts = 2 * len(platform.harts)
        self.enable = [[False] * (sources + 1) for _ in range(self.contexts)]
        self.threshold = [0] * self.contexts

    def set_input(self, src: int, level: bool) -> None:
        """A source's wire (1.2: the gateway turns the first assertion — or
        rising edge — into one request until the completion)."""
        old = self.level[src]
        self.level[src] = level
        if not self.claimed[src] and (level and (not self.edge[src] or not old)):
            self.pending[src] = True
        self.update()

    def _best(self, ctx: int) -> int:
        best, prio = 0, 0
        for s in range(1, self.n + 1):
            if self.pending[s] and self.enable[ctx][s] and self.priority[s] > self.threshold[ctx] \
                    and self.priority[s] > prio:
                best, prio = s, self.priority[s]
        return best

    def update(self) -> None:
        for h, m in enumerate(self.p.harts):
            m.set_interrupt(MEI, self._best(2 * h) != 0)
            if 1 in m.modes:
                m.set_interrupt(SEI, self._best(2 * h + 1) != 0)

    def reg_read(self, m, off: int, side_effects: bool = True) -> int:
        if off < 0x1000:
            s = off // 4
            return self.priority[s] if 0 < s <= self.n else 0
        if off < 0x1080:
            w = (off - 0x1000) // 4
            return sum(1 << b for b in range(32) if 0 < 32 * w + b <= self.n
                       and self.pending[32 * w + b])
        if 0x2000 <= off < 0x200000:
            ctx, w = divmod(off - 0x2000, 0x80)
            w //= 4
            if ctx < self.contexts and w < 32:
                return sum(1 << b for b in range(32) if 0 < 32 * w + b <= self.n
                           and self.enable[ctx][32 * w + b])
            return 0
        if off >= 0x200000:
            ctx, reg = divmod(off - 0x200000, 0x1000)
            if ctx >= self.contexts:
                return 0
            if reg == 0:
                return self.threshold[ctx]
            if reg == 4:                        # claim (8)
                s = self._best(ctx)
                if s and side_effects:
                    self.pending[s] = False
                    self.claimed[s] = True
                    self.update()
                return s
        return 0

    def reg_write(self, m, off: int, value: int, size: int) -> None:
        if size != 4:
            return
        mask = (1 << self.PRIO_BITS) - 1
        if off < 0x1000:
            s = off // 4
            if 0 < s <= self.n:
                self.priority[s] = value & mask
        elif 0x2000 <= off < 0x200000:
            ctx, w = divmod(off - 0x2000, 0x80)
            w //= 4
            if ctx < self.contexts and w < 32:
                for b in range(32):
                    s = 32 * w + b
                    if 0 < s <= self.n:
                        self.enable[ctx][s] = bool(value >> b & 1)
        elif off >= 0x200000:
            ctx, reg = divmod(off - 0x200000, 0x1000)
            if ctx >= self.contexts:
                return
            if reg == 0:
                self.threshold[ctx] = value & mask
            elif reg == 4:                      # complete (8): an enabled source only
                s = value
                if 0 < s <= self.n and self.enable[ctx][s] and self.claimed[s]:
                    self.claimed[s] = False
                    if self.level[s] and not self.edge[s]:
                        self.pending[s] = True  # (still asserted: a new request)
        self.update()


# ---------------------------------------------------------------- the APLIC

class Domain(Device):
    """One APLIC interrupt domain (AIA 4.5-4.9): direct delivery mode, or
    MSI delivery mode when its APLIC forwards by MSIs (``APLIC.msi``)."""

    IPRIOLEN = 8
    genmsi = 0

    def __init__(self, aplic: "APLIC", base: int, level: int, child: "Domain | None") -> None:
        self.a = aplic
        self.base = base
        self.size = APLIC_SIZE
        self.level = level                      # 3 = machine, 1 = supervisor
        self.child = child
        n = aplic.n
        self.ie = False
        self.sourcecfg = [0] * (n + 1)
        self.pending = [False] * (n + 1)
        self.enabled = [False] * (n + 1)
        self.target = [0] * (n + 1)
        harts = len(aplic.p.harts)
        self.idelivery = [0] * harts
        self.iforce = [0] * harts
        self.ithreshold = [0] * harts

    # -- the sources -----------------------------------------------------------------

    def visible(self, s: int) -> bool:
        """sourcecfg[s] exists here: always in the root, in the child when
        the root delegates ``s`` (4.5.2)."""
        return self.level == 3 or self.a.owner(s) is self

    def active(self, s: int) -> bool:
        cfg = self.sourcecfg[s]
        return self.a.owner(s) is self and not cfg & 0x400 and cfg & 7 != 0

    def mode(self, s: int) -> int:
        return self.sourcecfg[s] & 7 if self.active(s) else 0

    def rectified(self, s: int) -> bool:
        sm = self.mode(s)
        if sm in (4, 6):
            return self.a.wire[s]
        if sm in (5, 7):
            return not self.a.wire[s]
        return False

    def wire_changed(self, s: int, old_rect: bool) -> None:
        sm = self.mode(s)
        new = self.rectified(s)
        if sm in (4, 5) and new and not old_rect:
            self.pending[s] = True              # 4.7: an edge
        elif sm in (6, 7) and self.a.msi:       # 4.7, MSI mode: set on the rising edge,
            if new and not old_rect:            # cleared while low
                self.pending[s] = True
            elif not new:
                self.pending[s] = False
        elif sm in (6, 7):
            self.pending[s] = new               # 4.7: level, direct mode: a copy

    def _set_pending(self, s: int, v: bool) -> None:
        """A write that sets / clears a pending bit, "if possible" (4.7)."""
        sm = self.mode(s)
        if sm in (6, 7):
            if self.a.msi and (not v or self.rectified(s)):
                self.pending[s] = v             # (MSI mode: set only while high)
            return                              # level, direct: only the wire decides
        if sm:
            self.pending[s] = v

    # -- delivery ----------------------------------------------------------------------

    def topi(self, h: int) -> int:
        best, prio = 0, 0
        for s in range(1, self.a.n + 1):
            if not (self.pending[s] and self.enabled[s] and self.active(s)):
                continue
            if self.target[s] >> 18 != h:
                continue
            p = self.target[s] & 0xFF
            if self.ithreshold[h] and p >= self.ithreshold[h]:
                continue
            if best == 0 or p < prio:
                best, prio = s, p
        return best << 16 | prio if best else 0

    def signal(self, h: int) -> bool:
        if self.a.msi:
            return False                        # (MSI mode: no wire to the harts)
        return self.ie and bool(self.idelivery[h]) and bool(self.iforce[h] or self.topi(h))

    def forward(self) -> None:
        """MSI delivery mode (4.9): each active source pending and enabled
        with IE goes as an MSI to its target, its pending bit cleared."""
        if not self.ie:
            return
        for s in range(1, self.a.n + 1):
            if self.pending[s] and self.enabled[s] and self.active(s):
                self.pending[s] = False
                t = self.target[s]
                self.a.send_msi(self, t >> 18, (t >> 12) & 0x3F, t & 0x7FF)

    # -- registers ---------------------------------------------------------------------

    def reg_read(self, m, off: int, side_effects: bool = True) -> int:
        n = self.a.n
        if off == 0:
            return 0x80000000 | (0x100 if self.ie else 0) | (4 if self.a.msi else 0)
        if 0x1BC0 <= off < 0x1BD0:              # (4.5.3 / 4.5.4: the root's only)
            return self.a.msiaddrcfg[(off - 0x1BC0) // 4] \
                if self.a.msi and self.level == 3 else 0
        if off == 0x3000:                       # genmsi (4.5.15; Busy never seen)
            return self.genmsi if self.a.msi else 0
        if off < 0x1000:
            s = off // 4
            return self.sourcecfg[s] if 0 < s <= n and self.visible(s) else 0
        if 0x1C00 <= off < 0x1C80:
            return self._bits(off - 0x1C00, lambda s: self.pending[s] and self.active(s))
        if 0x1D00 <= off < 0x1D80:
            return self._bits(off - 0x1D00, lambda s: self.rectified(s))
        if 0x1E00 <= off < 0x1E80:
            return self._bits(off - 0x1E00, lambda s: self.enabled[s] and self.active(s))
        if 0x3004 <= off < 0x4000:
            s = (off - 0x3000) // 4
            return self.target[s] if s <= n and self.active(s) else 0
        if off >= 0x4000:
            h, reg = divmod(off - 0x4000, 32)
            if h >= len(self.idelivery) or self.a.msi:
                return 0
            if reg == 0:
                return self.idelivery[h]
            if reg == 4:
                return self.iforce[h]
            if reg == 8:
                return self.ithreshold[h]
            if reg in (0x18, 0x1C):
                t = self.topi(h)
                if reg == 0x1C and side_effects:  # claimi (4.8.1.5)
                    if t:
                        s = t >> 16
                        if self.mode(s) not in (6, 7):
                            self.pending[s] = False
                    else:
                        self.iforce[h] = 0
                    self.a.update()
                return t
        return 0

    def _bits(self, off: int, pred) -> int:
        w = off // 4
        return sum(1 << b for b in range(32) if 0 < 32 * w + b <= self.a.n and pred(32 * w + b))

    def reg_write(self, m, off: int, value: int, size: int) -> None:
        if size != 4:
            return                              # (4.5: other sizes are ignored)
        n = self.a.n
        if off == 0:
            self.ie = bool(value & 0x100)       # (DM read-only: 0, or 1 with IMSICs; BE 0)
        elif 0x1BC0 <= off < 0x1BD0:
            if self.a.msi and self.level == 3:
                self.a.write_msiaddrcfg((off - 0x1BC0) // 4, value)
        elif off == 0x3000:
            if self.a.msi:                      # (an extempore MSI: not IE-gated)
                self.genmsi = value & (0x3FFF << 18 | 0x7FF)
                self.a.send_msi(self, self.genmsi >> 18, 0, self.genmsi & 0x7FF)
        elif off < 0x1000:
            s = off // 4
            if 0 < s <= n and self.visible(s):
                old = self.rectified(s)
                if value & 0x400:               # D: only with a child (index 0: WLRL)
                    v = 0x400 if self.child is not None else 0
                else:
                    sm = value & 7
                    v = sm if sm in (0, 1, 4, 5, 6, 7) else 0
                self.sourcecfg[s] = v
                if not self.active(s):
                    self.pending[s] = self.enabled[s] = False
                    self.target[s] = 0
                else:
                    self.wire_changed(s, old)
                if self.child is not None and not v & 0x400:
                    self.child.forget(s)
        elif 0x1C00 <= off < 0x1C80:
            self._each(off - 0x1C00, value, lambda s: self._set_pending(s, True))
        elif off == 0x1CDC or off == 0x2000:     # setipnum / setipnum_le
            if 0 < value <= n and self.active(value):
                self._set_pending(value, True)
        elif off == 0x2004:                     # setipnum_be (BE read-only 0: swapped)
            v = int.from_bytes(value.to_bytes(4, "little"), "big")
            if 0 < v <= n and self.active(v):
                self._set_pending(v, True)
        elif 0x1D00 <= off < 0x1D80:
            self._each(off - 0x1D00, value, lambda s: self._set_pending(s, False))
        elif off == 0x1DDC:
            if 0 < value <= n and self.active(value):
                self._set_pending(value, False)
        elif 0x1E00 <= off < 0x1E80:
            self._each(off - 0x1E00, value, lambda s: self.enabled.__setitem__(s, True))
        elif off == 0x1EDC:
            if 0 < value <= n and self.active(value):
                self.enabled[value] = True
        elif 0x1F00 <= off < 0x1F80:
            self._each(off - 0x1F00, value, lambda s: self.enabled.__setitem__(s, False))
        elif off == 0x1FDC:
            if 0 < value <= n and self.active(value):
                self.enabled[value] = False
        elif 0x3004 <= off < 0x4000:
            s = (off - 0x3000) // 4
            if s <= n and self.active(s):
                hart = (value >> 18) & 0x3FFF
                if self.a.msi:                  # (4.5.16: Hart, Guest, EIID)
                    guest = (value >> 12) & 0x3F if self.level == 1 and self.a.guests else 0
                    self.target[s] = hart << 18 | guest << 12 | value & 0x7FF
                else:
                    prio = value & ((1 << self.IPRIOLEN) - 1) or 1
                    self.target[s] = hart << 18 | prio
        elif off >= 0x4000 and not self.a.msi:
            h, reg = divmod(off - 0x4000, 32)
            if h < len(self.idelivery):
                if reg == 0:
                    self.idelivery[h] = value & 1
                elif reg == 4:
                    self.iforce[h] = value & 1
                elif reg == 8:
                    self.ithreshold[h] = value & ((1 << self.IPRIOLEN) - 1)
        self.a.update()

    def _each(self, off: int, value: int, fn) -> None:
        w = off // 4
        for b in range(32):
            s = 32 * w + b
            if value >> b & 1 and 0 < s <= self.a.n and self.active(s):
                fn(s)

    def forget(self, s: int) -> None:
        """Source ``s`` is no longer delegated here: it reads 0 until written."""
        self.sourcecfg[s] = 0
        self.pending[s] = self.enabled[s] = False
        self.target[s] = 0
        if self.child is not None:
            self.child.forget(s)


class APLIC:
    """The APLIC: a machine-level root domain with one supervisor-level child
    (see the module docstring)."""

    #: MSI delivery mode (the harts have IMSICs); the guest index field exists
    msi = False
    guests = False

    def __init__(self, platform: "Platform", sources: int = SOURCES, msi: bool = False) -> None:
        self.p = platform
        self.n = sources
        self.wire = [False] * (sources + 1)
        self.s = Domain(self, MAP["aplic_s"], 1, None)
        self.m = Domain(self, MAP["aplic_m"], 3, self.s)
        self.msi = msi
        if msi:                                 # (4.5.3 / 4.5.4: the IMSIC layout at reset)
            from .rvaia import IMSIC_M_BASE, IMSIC_S_BASE, S_STRIDE, PAGE
            n = len(platform.harts)
            lhxw = max(0, (n - 1).bit_length())
            self.msiaddrcfg = [IMSIC_M_BASE >> 12, lhxw << 12,
                               IMSIC_S_BASE >> 12, ((S_STRIDE // PAGE).bit_length() - 1) << 20]
            self.guests = any(h.csrs.has_h for h in platform.harts)

    #: mmsiaddrcfgh's and smsiaddrcfgh's writable fields (4.5.3 / 4.5.4)
    _CFG_MASKS = (M32, 1 << 31 | 0x1F << 24 | 7 << 20 | 7 << 16 | 0xF << 12 | 0xFFF,
                  M32, 7 << 20 | 0xFFF)

    def write_msiaddrcfg(self, k: int, value: int) -> None:
        """A root-domain write of mmsiaddrcfg(h) / smsiaddrcfg(h): ignored
        once mmsiaddrcfgh.L is set (4.5.3)."""
        if self.msiaddrcfg[1] >> 31:
            return
        self.msiaddrcfg[k] = value & self._CFG_MASKS[k]

    def msi_address(self, dom: Domain, hart: int, guest: int) -> int:
        """4.9.1: the target address of an MSI from ``dom`` to hart index
        ``hart`` (and guest file ``guest`` from a supervisor domain)."""
        mcfg, mcfgh, scfg, scfgh = self.msiaddrcfg
        hhxs, lhxw, hhxw = mcfgh >> 24 & 0x1F, mcfgh >> 12 & 0xF, mcfgh >> 16 & 7
        g = (hart >> lhxw) & ((1 << hhxw) - 1)
        h = hart & ((1 << lhxw) - 1)
        if dom.level == 3:
            ppn, lhxs = (mcfgh & 0xFFF) << 32 | mcfg, mcfgh >> 20 & 7
            return (ppn | g << (hhxs + 12) | h << lhxs) << 12
        ppn, lhxs = (scfgh & 0xFFF) << 32 | scfg, scfgh >> 20 & 7
        return (ppn | g << (hhxs + 12) | h << lhxs | guest) << 12

    def send_msi(self, dom: Domain, hart: int, guest: int, eiid: int) -> None:
        """An MSI: the store of ``eiid`` (4.9.1: little-endian, 32 bits)."""
        self.p.msi_write(self.msi_address(dom, hart, guest), eiid)

    def owner(self, s: int) -> Domain:
        """The domain source ``s`` belongs to (the child when delegated)."""
        if self.m.sourcecfg[s] & 0x400:
            return self.s
        return self.m

    def set_input(self, src: int, level: bool) -> None:
        d = self.owner(src)
        old = d.rectified(src)
        self.wire[src] = level
        d.wire_changed(src, old)
        self.update()

    def update(self) -> None:
        if self.msi:
            self.m.forward()
            self.s.forward()
            return
        for h, hart in enumerate(self.p.harts):
            hart.set_interrupt(MEI, self.m.signal(h))
            if 1 in hart.modes:
                hart.set_interrupt(SEI, self.s.signal(h))

    def priority(self, h: int, level: str) -> int:
        """Direct mode: the priority of hart ``h``'s top interrupt at
        ``level`` (the AIA's external interrupt priority: Tables 5.3 / 5.5)."""
        d = self.m if level == "m" else self.s
        t = d.topi(h)
        return t & 0xFF if t and d.signal(h) else 0


# ---------------------------------------------------------------- the platform

class Platform:
    """The timer and interrupt controllers of a set of harts (see the module
    docstring).  ``controller``: "plic" or "aplic"; ``timer_div``: core
    cycles per mtime tick."""

    def __init__(self, harts: Iterable, *, controller: str = "plic", sources: int = SOURCES,
                 timer_div: int = 2) -> None:
        if controller not in ("plic", "aplic", "imsic"):
            raise ValueError(f"controller must be 'plic', 'aplic' or 'imsic', not {controller!r}")
        self.harts = list(harts)
        self.timer_div = timer_div
        self.clint = CLINT(self)
        self.sswi = SSWI(self)
        self.controller = controller
        self.imsics = []
        self.pages = None
        if controller == "plic":
            self.plic = PLIC(self, sources)
            self.aplic = None
            devices = [self.clint, self.sswi, self.plic]
        elif controller == "aplic":
            self.plic = None
            self.aplic = APLIC(self, sources)
            devices = [self.clint, self.sswi, self.aplic.m, self.aplic.s]
            for h, m in enumerate(self.harts):
                m.ext_priority = (lambda h: lambda level: self.aplic.priority(h, level))(h)
        else:
            from .rvaia import IMSIC, IMSICPages
            self.plic = None
            self.imsics = [IMSIC(m, h) for h, m in enumerate(self.harts)]
            self.pages = IMSICPages(self.imsics)
            self.aplic = APLIC(self, sources, msi=True)
            devices = [self.clint, self.sswi, self.aplic.m, self.aplic.s, self.pages]
        self.devices = devices
        for k, m in enumerate(self.harts):
            m.devices[:0] = devices
            m.io_regions = [(d.base, d.base + d.size) for d in devices
                            if d is not self.pages] + (self.pages.regions if self.pages else [])
            m.timer = self.clint
            if self.imsics:
                m.imsic = self.imsics[k]
                m.imsic_regions = self.pages.regions
                m.csrs.rebuild()                # (mtopei / stopei / vstopei exist now)
        self.clint.update()

    def msi_write(self, pa: int, value: int) -> None:
        """A 32-bit little-endian store to physical ``pa`` by a bus master
        (an APLIC's MSI): a device's register, else memory (not through the
        harts' caches: as a device's DMA)."""
        for d in self.devices:
            if d.covers(pa):
                d.store(self.harts[0], pa, value & M32, 4)
                return
        for k in range(4):
            self.harts[0].phys_write_byte(pa + k, value >> (8 * k) & 0xFF)

    def set_input(self, src: int, level: bool) -> None:
        """Interrupt source ``src``'s wire (a device's interrupt output)."""
        (self.plic or self.aplic).set_input(src, level)
