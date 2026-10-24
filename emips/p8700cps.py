"""The MIPS P8700 cluster (PLAN M25 phase 5d; MD01502 rev. 1.84 ch. 5 the
Coherence Manager, 5.13 / 5.14 its registers, ch. 6 the CPC, 6.5.10 hart run
/ suspend; QEMU's Boston board model — ``hw/riscv/cps.c``,
``hw/misc/riscv_cmgcr.c`` / ``riscv_cpc.c`` — where the manual is silent).

:class:`Cluster` is 1-6 cores of 1-2 harts (1.1: "up to two threads, or
harts") on one memory: ``machines`` = the harts, hart ``i`` = core ``i //
harts``, hart ``i % harts``; mhartid = CORENUM << 4 | HARTNUM (7.1.3; one
cluster).  A core's harts share its FTLB (2.6; emips.p8700mmu
``share_ftlb``) and its caches; the LR reservation of a hart is lost when
another hart stores to its 64-byte granule (3.2: "a reservation by one
hart will be cleared by any ownership request by any other hart or core
for the same 64B coherence granule").

The GCRs (:class:`GCR`) are one 512 KB block at GCR_BASE — reset: the
first naturally aligned block below the reset PC (5.14.1.2) — with the
blocks of Table 5.24 eMIPS has so far:

- **GCR.Global** (5.14.1): CONFIG (PCORES = cores - 1, ADDR_REGIONS 8,
  NUM_CLUSTERS 1, DBU_PRESENT), GCR_BASE (bits 47:19 writable; a 32-bit
  write of the upper half waits in a shadow until the lower half is
  written), CONTROL (stored; ARB_PRI_RAISE_CNT 32 at reset), REV (0x0A00:
  QEMU's), CPC_STATUS (1), ACCESS (cores 7:0, IOCUs 23:16; a core whose bit
  is 0 cannot write the GCR / CPC), SCRATCH0 / 1, SEM (5.14.1.31), the
  CM's error registers (ERR_CONTROL, ERR_MASK / CAUSE / ADDR / MULT: the
  Coherence Manager's), CACHE_REV, L2_CONFIG (the L2's shape, L2_BYPASS,
  the cache-op write enables), L2_RAM_CONFIG (HCI_DONE), the L2 cache-op
  state machine (L2SM_COP, L2SM_TAG_ADDR_COP: 3.6), the L2 tag / data / ECC
  registers, the MMIO-region and L2-prefetch registers stored.
- **GCR.Core** (5.14.3): Hn_RESET_BASE (0x2000 + 0x100 core + 8 hart; 47:12
  R/W, RESET_BASE_MODE reads 1) and COH_EN (0x20F8 + 0x100 core).
- **CPC.Global** (5.14.4): REVISION, ROCC_CTL (a core's bit set by its
  reset, write 1 to clear), MTIME_REG (the cluster's mtime: one tick every
  ``timer_div`` clocks unless TIMECTL.MSTOP; written by the
  synchronisation procedure of 8.1.3.1), TIMECTL, HRTIME (the clock), the
  sequencer / rail / reset-length / clock registers stored.
- **CPC.Core** (5.14.5, 6.5.10): CMD (1 ClockOff, 2 PwrDown, 3 PwrUp, 4
  Reset; ClockOff / PwrDown wait while COH_EN = 1), STAT_CONF (SEQ_STATE:
  D0, U2, U5, U6 = coherent), VP_STOP (W1C: a running hart stops — its
  state kept), VP_RUN (W1S: a stopped hart starts afresh at its
  RESET_BASE), VP_RUNNING.  A core powered up or reset runs hart 0 (the
  si_core_vc_run_init pins: hart 0's 1).

The caches (ch. 3): :class:`P8700CM` (``cm``) — MESI between the cores'
L1 data caches, the shared L2 (write-back, write-allocate), FENCE.I on
every I-cache; each core's L1s are shared by its harts.

The ACLINT / APLIC at GCR_BASE + 0x50000 / 0x40000 / ...: ``ic``
(emips.p8700ic), its timer compares done when the clock moves.  Choices:
``sim.DEVIATIONS["P8700 cluster"]``.
"""

from __future__ import annotations

from typing import Any, Callable

from .coherence import CoherenceManager

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
GCR_SIZE = 0x80000                      # (5.13: a 512 KB block, the AIA CM's)
MAX_CORES, MAX_HARTS = 6, 2
REV = 0x0A00                            # (QEMU's gcr-rev: the manual has none)
CPC_REVISION = 0x0100
CACHE_REVISION = 0x0100                 # (GCR_CACHE_REV: "from configuration")
#: GCR.Global (5.14.1)
CONFIG, BASE, CONTROL, REVR = 0x0000, 0x0008, 0x0010, 0x0030
ERR_CONTROL, ERR_MASK, ERR_CAUSE, ERR_ADDR, ERR_MULT = 0x38, 0x40, 0x48, 0x50, 0x58
CUSTOM_STATUS, CPC_STATUS, ACCESS = 0x0068, 0x00F0, 0x0120
SCRATCH0, SCRATCH1, SEM = 0x0280, 0x0288, 0x0640
CACHE_REV, L2_CONFIG, L2_RAM_CONFIG = 0x00E0, 0x0130, 0x0240
L2_TAG_ADDR, L2_TAG_STATE, L2_DATA, L2_ECC = 0x0600, 0x0608, 0x0610, 0x0618
L2SM_COP, L2SM_TAG_ADDR_COP = 0x0620, 0x0628
#: their writable bits (5.14.1.25-28: what an L2 Load / Store Tag & Data
#: cache op would use; rev. 1.84 has no instruction that runs one — stored)
L2_STORED = {L2_TAG_ADDR: 0xFFFFFFFF8000, L2_TAG_STATE: 0x7FFF00000FFF, L2_DATA: M64,
             L2_ECC: 1 << 63 | 0xFF << 32 | 1 << 31 | 0xFF}
#: GCR_L2_CONFIG (5.14.1.15): REG_EXISTS, the COP write enables (26:24,
#: COP_LRU_WE 1 at reset), L2_BYPASS (20)
L2_CONFIG_RW, L2_BYPASS = 0x7 << 24 | 1 << 20, 1 << 20
#: the L2 cache-op state machine (5.14.1.29 / 3.6): types and results
COP_FLUSH, COP_INIT_TAG, COP_INIT, COP_HIT_INV, COP_HIT_WB_INV, COP_HIT_WB, COP_FETCH_LOCK = (
    0, 1, 2, 4, 5, 6, 7)
COP_DONE, COP_DONE_ERR = 1, 2
L2_PFT_CONTROL, L2_PFT_CONTROL_B = 0x0300, 0x0308
TIMEOUT_LIMIT, MMIO_REQ_LIMIT = 0x0650, 0x06F8
MMIO = tuple(0x0700 + 0x10 * k + h for k in range(8) for h in (0, 8))
STORED = (CONTROL, SCRATCH0, SCRATCH1, L2_PFT_CONTROL, L2_PFT_CONTROL_B, TIMEOUT_LIMIT,
          MMIO_REQ_LIMIT) + MMIO
#: the error registers (5.14.1.5-9) -> the Coherence Manager's attributes
CM_ERRORS = {ERR_MASK: "error_mask", ERR_CAUSE: "error_cause", ERR_ADDR: "error_addr",
             ERR_MULT: "error_mult"}
CONTROL_RESET = 32 << 25                # (ARB_PRI_RAISE_CNT)
#: GCR.Core (5.14.3)
CORE_BLOCK, CORE_STRIDE, COH_EN = 0x2000, 0x100, 0xF8
#: CPC.Global (5.14.4) and CPC.Core (5.14.5)
CPC = 0x8000
SEQDEL, RAIL, RESETLEN, CPC_REV, CC_CTL, PWRUP_CTL, RES_REL, ROCC_CTL, PRESCALE = (
    0x8008, 0x8010, 0x8018, 0x8020, 0x8028, 0x8030, 0x8038, 0x8040, 0x8048)
MTIME, TIMECTL, CLK_GATE_DIS, HRTIME, CPC_CONFIG = 0x8050, 0x8058, 0x8060, 0x8090, 0x8138
CPC_STORED = (SEQDEL, RAIL, RESETLEN, CC_CTL, PWRUP_CTL, RES_REL, PRESCALE, CLK_GATE_DIS)
CPC_CORE, CMD, STAT_CONF, VP_STOP, VP_RUN, VP_RUNNING = 0xA000, 0x00, 0x08, 0x20, 0x28, 0x30
#: CPC commands and sequencer states (Table 5.108 / 5.109)
CLOCK_OFF, PWR_DOWN, PWR_UP, RESET = 1, 2, 3, 4
D0, U2, U5, U6 = 0, 3, 6, 7
STATE_NAMES = {D0: "D0 (powered down)", U2: "U2 (clock off)", U5: "U5 (non-coherent)",
               U6: "U6 (coherent)"}


class GCR:
    """The cluster's register block: a device every hart sees (5.13)."""

    def __init__(self, cl: "Cluster") -> None:
        self.cl = cl
        self.regs: dict[int, int] = dict.fromkeys(STORED, 0)
        self.regs[CONTROL] = CONTROL_RESET
        self.err_control = 2                      # (L2_ECC_EN)
        self.l2_config = 1 << 26                  # (COP_LRU_WE)
        self.l2_regs = dict.fromkeys(L2_STORED, 0)
        self.cop = 0                              # (L2SM_COP: RESULT, MODE, TYPE)
        self.cop_range = 0                        # (L2SM_TAG_ADDR_COP)
        self.access = 0xFF00FF                    # (5.14.1.14: everyone may write)
        self.sem = 0
        self._base_hi: int | None = None          # (GCR_BASE's 32-bit write shadow)

    # -- the bus -------------------------------------------------------------------

    def covers(self, pa: int) -> bool:
        return self.cl.base <= pa < self.cl.base + GCR_SIZE

    def tick(self, m, cycles: int) -> None:
        pass

    def load(self, m, pa: int, size: int) -> int:
        off = pa - self.cl.base
        ic = self.cl.ic
        if ic is not None and ic.covers(off):    # (the interrupt controller: 32-bit)
            return ic.load(m, off, size)
        word = off & ~7
        return (self.get(m, word) >> (8 * (off & 7))) & ((1 << (8 * size)) - 1)

    def store(self, m, pa: int, value: int, size: int) -> None:
        off = pa - self.cl.base
        ic = self.cl.ic
        if ic is not None and ic.covers(off):    # (ch. 8: writable by every requester)
            ic.store(m, off, value, size)
            return
        word, shift = off & ~7, 8 * (off & 7)
        mask = (((1 << (8 * size)) - 1) << shift) & M64
        if not self._may_write(m, word):
            return
        self.put(m, word, (value << shift) & mask, mask)

    def _may_write(self, m, off: int) -> bool:
        """5.14.1.14: ACCESS gates the GCR and CPC (not the interrupt
        controller) by requesting core."""
        if off >= 0x10000:
            return True
        return bool(self.access >> self.cl.core_of(m) & 1)

    # -- the registers ---------------------------------------------------------------

    def get(self, m, off: int) -> int:
        cl = self.cl
        if off == CONFIG:
            return (cl.cores - 1) | 8 << 16 | 1 << 23 | 1 << 40
        if off == BASE:
            return cl.base
        if off == REVR:
            return REV
        if off == CPC_STATUS:
            return 1
        if off == ACCESS:
            return self.access
        if off == SEM:
            return self.sem
        if off == CUSTOM_STATUS:
            return 0
        cm = cl.cm
        if off == ERR_CONTROL:              # (5.14.1.5: L2 ECC always supported)
            return self.err_control | 1
        if off in CM_ERRORS:
            return getattr(cm, CM_ERRORS[off])
        if off == CACHE_REV:
            return CACHE_REVISION
        if off == L2_CONFIG:
            return 1 << 31 | self.l2_config | cm.l2_config() & ~L2_BYPASS
        if off == L2_RAM_CONFIG:            # (PRESENT, HCI_DONE: 5.14.1.20 / 3.5)
            return 1 << 31 | 1 << 30
        if off in L2_STORED:
            return self.l2_regs[off]
        if off == L2SM_COP:
            return 1 << 31 | self.cop
        if off == L2SM_TAG_ADDR_COP:
            return self.cop_range
        if off in self.regs:
            return self.regs[off]
        if CORE_BLOCK <= off < CORE_BLOCK + MAX_CORES * CORE_STRIDE:
            c, r = divmod(off - CORE_BLOCK, CORE_STRIDE)
            if c >= cl.cores:
                return 0
            if r == COH_EN:
                return cl.coh_en[c]
            if r < 8 * cl.harts and not r & 7:
                return cl.reset_base[c * cl.harts + r // 8] | 2     # (RESET_BASE_MODE)
            return 0
        if CPC <= off < CPC + 0x8000:
            return cl.cpc_get(off)
        return 0

    def put(self, m, off: int, value: int, mask: int) -> None:
        cl = self.cl
        if off == BASE:                     # (5.14.1.2: 47:19; the upper half shadowed)
            if mask == M32 << 32:
                self._base_hi = value >> 32
                return
            hi = self._base_hi if mask == M32 and self._base_hi is not None else \
                cl.base >> 32
            self._base_hi = None
            new = (hi << 32 | value & M32) if mask == M32 else (cl.base & ~mask) | value
            cl.move((new & ((1 << 48) - 1)) & ~(GCR_SIZE - 1))
            return
        if off == ACCESS:
            self.access = ((self.access & ~mask) | value) & 0xFF00FF
            return
        if off == SEM:                      # (5.14.1.31: a set lock refuses a new owner)
            v = value & M32
            if not (v >> 31 and self.sem >> 31):
                self.sem = v
            return
        cm = cl.cm
        if off == ERR_CONTROL:              # (MEM_PARITY_EN, L2_ECC_EN)
            self.err_control = ((self.err_control & ~mask) | value) & 0x22
            cm.ecc_checking(bool(self.err_control & 2))
            return
        if off in CM_ERRORS:
            self._put_error(off, value, mask)
            return
        if off == L2_CONFIG:
            self.l2_config = ((self.l2_config & ~mask) | value) & L2_CONFIG_RW
            cm.bypass = bool(self.l2_config & L2_BYPASS)
            return
        if off in L2_STORED:
            self.l2_regs[off] = ((self.l2_regs[off] & ~mask) | value) & L2_STORED[off]
            return
        if off == L2SM_COP:
            self._cop(m, value, mask)
            return
        if off == L2SM_TAG_ADDR_COP:        # (written only while the state machine idles)
            self.cop_range = ((self.cop_range & ~mask) | value) & (M64 & ~0x3F)
            return
        if off in self.regs:
            self.regs[off] = (self.regs[off] & ~mask) | value
            return
        if CORE_BLOCK <= off < CORE_BLOCK + MAX_CORES * CORE_STRIDE:
            c, r = divmod(off - CORE_BLOCK, CORE_STRIDE)
            if c >= cl.cores:
                return
            if r == COH_EN:
                cl.set_coherent(c, bool(value & 1) if mask & 1 else bool(cl.coh_en[c]))
            elif r < 8 * cl.harts and not r & 7:
                i = c * cl.harts + r // 8
                old = cl.reset_base[i]
                cl.reset_base[i] = ((old & ~mask) | value) & 0xFFFFFFFFF000
            return
        if CPC <= off < CPC + 0x8000:
            cl.cpc_put(off, value, mask)

    def _put_error(self, off: int, value: int, mask: int) -> None:
        """5.14.1.6-9: ERR_MASK / ERR_ADDR R/W; ERR_CAUSE and ERR_MULT clear
        when software writes the recorded ERR_TYPE (63:58) back."""
        cm = self.cl.cm
        if off in (ERR_CAUSE, ERR_MULT):
            kind = cm.error_cause >> 58
            if mask >> 58 == 0x3F and kind and (value >> 58) & 0x3F == kind:
                cm.error_cause = cm.error_mult = 0
                cm.error_changed()
            return
        name = CM_ERRORS[off]
        width = M64 if off == ERR_MASK else (1 << 48) - 1
        setattr(cm, name, ((getattr(cm, name) & ~mask) | value) & width)
        if off == ERR_MASK:
            cm.error_changed()

    def domain(self, core: int) -> int:
        """The coherence domain as the CM's bits (emips.coherence): a core
        with COH_EN = 1 sees every other coherent core (5.14.3.x: COH_EN)."""
        cl = self.cl
        if not cl.coh_en[core]:
            return 0
        return sum(1 << j for j in range(cl.cores) if cl.coh_en[j])

    # -- the L2 cache-op state machine (5.14.1.29 / 30, 3.6) ---------------------------

    def _cop(self, m, value: int, mask: int) -> None:
        """TYPE (4:2) is written only while idle; CMD 1 = START runs the
        operation (eMIPS: at once, RESULT = DONE), 3 = ABORT (nothing runs
        long enough to abort)."""
        cop = self.cop
        if cop & 0x20:                      # (RUNNING: never seen — the operation is done)
            return
        typ = ((cop & ~mask) | value) >> 2 & 7 if mask & 0x1C else cop >> 2 & 7
        cmd = value & 3 if mask & 3 else 0
        self.cop = (cop & 0x1C0) | typ << 2
        if cmd == 1:
            ok = self.cl.cm.l2_operation(typ, (self.cop_range & ((1 << 48) - 1)) & ~0x3F,
                                         self.cop_range >> 48, getattr(m, "caches", None))
            self.cop = (COP_DONE if ok else COP_DONE_ERR) << 6 | typ << 2


class P8700CM(CoherenceManager):
    """The CM3.7's coherence and its L2 (MD01502 3.3.5 / 3.3.7 / 3.4, ch. 5):
    eMIPS's Coherence Manager (MESI between the cores' L1 data caches,
    interventions, the L2 behind them) with what the P8700 has — no CM
    address regions (its MMIO regions route uncached requests), the L2
    write-back *write-allocate* (3.3.8), GCR_L2_CONFIG.L2_BYPASS, the
    L2 cache-op state machine; caches initialized by hardware at reset
    (3.5; the L1s too: no instruction can initialize them)."""

    def __init__(self, cl: "Cluster", l2=None) -> None:
        # (3.3.8 / 5.14.1.5: the L2's SEC-DED is always there)
        super().__init__(cl, l2=l2, ecc=True, hardware_init=True, seed=0x8700)
        self.error_mask = 0                 # (5.14.1.6: reset 0)
        #: GCR_L2_CONFIG.L2_BYPASS: every request goes to memory
        self.bypass = False
        if self.l2 is not None:
            self.l2.write_allocate = True
            self.l2.on_corrected = self._l2_corrected

    def _l2_corrected(self, cache, s: int, w: int, k: int, bit: int) -> None:
        """A corrected L2 error: recorded in the requesting core's
        mipscacheerr (ARRAY L2Data)."""
        from .p8700err import CORRECTED, L2_DATA
        errs = getattr(self.current, "errs", None)
        if errs is not None:
            errs.record(CORRECTED, L2_DATA, bits=bit, way=w, index=s, word=k)

    #: FENCE.I invalidates every core's I-cache (3.3.10)
    fence_i_global = True

    def l2_policy(self, pa: int) -> str:
        return "bypass" if self.bypass else "wb"

    def _region_attr(self, pa: int) -> tuple[int, bool]:
        return 0, True

    @property
    def memory_sync0(self) -> bool:         # (no SYNCCTL in the CM3.7's CONTROL)
        return False

    def drop_l2(self, pa: int) -> bool:
        return False

    def l2_only_sync(self, pa: int) -> bool:
        return False

    def describe(self) -> list[str]:
        out = []
        cl = self.cps
        for j in sorted(self.cores):
            coh = self._coherence(j)
            frm = [str(i) for i in sorted(self.cores) if i != j and coh >> i & 1]
            out.append(f"core {j}: COH_EN {cl.coh_en[j]}, "
                       + ("coherent" if coh >> j & 1 else "not coherent")
                       + (f", interventions from core(s) {', '.join(frm)}" if frm else ""))
        out.append(", ".join(f"{k} {v}" for k, v in self.counts.items()))
        if self.l2 is not None:
            out.append(f"shared {self.l2.name}: {self.l2.geometry.describe()}, write-back "
                       "write-allocate" + (" (L2_BYPASS: bypassed)" if self.bypass else ""))
        out.append(self.describe_error())
        return out

    def ecc_checking(self, on: bool) -> None:
        """GCR_ERR_CONTROL.L2_ECC_EN (phase 5e2)."""
        for c in self.secondary:
            c.check = on and self.ecc

    def l2_operation(self, typ: int, start: int, lines: int, req=None) -> bool:
        """The L2SM's operation (5.14.1.29, 3.6.1 / 3.6.2): 0 a full flush
        (every dirty line written back, every line invalidated), 1 / 2 the
        full initialization (tags; tags and data), 4-7 Hit Invalidate / Hit
        Writeback Invalidate / Hit Writeback / Fetch and Lock on ``lines``
        lines from ``start``.  Returns False on an error (DONE - ERR)."""
        from .cache import Caches
        c = self.l2
        if c is None:
            return True
        if req is None and self.cores:
            req = self.current or next(iter(self.cores.values()))
        if req is not None:
            self.claim(req)
        if typ in (COP_FLUSH, COP_INIT_TAG, COP_INIT):
            for s in range(c.sets):
                for w in range(c.ways):
                    ln = c.lines[s][w]
                    if typ == COP_FLUSH and ln.valid and ln.dirty:
                        c.write_back(s, w, c.below)
                    Caches._invalidate(c, s, w)
                    if typ == COP_INIT:
                        ln.data[:] = bytes(c.line)
                        c._set_data_parity(ln, 0, c.line)
            return True
        if typ in (COP_HIT_INV, COP_HIT_WB_INV, COP_HIT_WB, COP_FETCH_LOCK) and req is not None:
            for k in range(min(lines, 1 << 16)):
                pa = (start + k * c.line) & M32
                req._cache_op(typ << 2 | 3, pa, pa, 5)
        return True


class Cluster:
    """See the module docstring.  ``factory(cluster, core, hart, pc)`` makes a
    hart in its reset state starting at ``pc``; ``on_start(i, m)`` hears of a
    hart made afresh (VP_RUN, a core reset)."""

    #: what the Coherence Manager asks of its system (emips.coherence)
    cm3, iocu = True, False

    def __init__(self, cores: int, harts: int, factory: Callable[..., Any], reset_pc: int,
                 *, timer_div: int = 2, on_start: Callable[[int, Any], None] | None = None,
                 caches: Callable[[int], Any] | None = None, l2=None) -> None:
        if not 1 <= cores <= MAX_CORES or not 1 <= harts <= MAX_HARTS:
            raise ValueError(f"the P8700 has 1-{MAX_CORES} cores of 1-{MAX_HARTS} harts "
                             "(MD01502 1.1)")
        self.cores, self.harts = cores, harts
        self.factory, self.on_start = factory, on_start
        self.timer_div = timer_div
        #: GCR_BASE (5.14.1.2: the 512 KB block below the reset PC)
        self.base = ((reset_pc & ~(GCR_SIZE - 1)) - GCR_SIZE) & ((1 << 48) - 1)
        n = cores * harts
        self.reset_base = [reset_pc & 0xFFFFFFFFF000] * n
        self.coh_en = [0] * cores
        #: CPC: per-core state, last command, the running harts (VP_RUNNING)
        self.state = [U5] + [D0] * (cores - 1)
        self.cmd = [PWR_UP] + [0] * (cores - 1)
        self.pending: list[int | None] = [None] * cores
        self.running_mask = [1] + [0] * (cores - 1)
        self.rocc = 0
        self.cpc_regs: dict[int, int] = dict.fromkeys(CPC_STORED, 0)
        self.timectl = 0
        self._clock = 0
        self.ic = None
        self._mtime = 0                     # (mtime = _mtime + (clock - _since) // div)
        self._since = 0
        self.gcr = GCR(self)
        self.timer = _Timer(self)
        #: the CM3.7: coherence, the L2, the error registers
        self.cm = P8700CM(self, l2)
        #: each core's L1s, shared by its harts (3.3: "The L1 instruction and
        #: data caches are shared by all Harts in the same core")
        self._caches = [caches(c) if caches is not None else None for c in range(cores)]
        from .p8700err import CoreErrors, equip
        #: each core's mipscacheerr / mipserrctl (emips.p8700err)
        self.errors = [CoreErrors() for _ in range(cores)]
        for c, ch in enumerate(self._caches):
            if ch is not None:
                equip(ch, self.errors[c])   # (the ECC L1s)
                self.cm.attach(c, ch)
                ch.alias_free = True        # (3.3.2: "logic to correct virtual aliasing")
        self.cm.ecc_checking(True)          # (GCR_ERR_CONTROL.L2_ECC_EN resets to 1)
        self.ftlbs: list[Any] = [None] * cores
        self.machines: list[Any] = [None] * n
        for i in range(n):
            self.machines[i] = self._make(i, reset_pc)
        from .p8700ic import Controller
        #: the ACLINT / APLIC (ch. 8; emips.p8700ic)
        self.ic = Controller(self)

    @property
    def clock(self) -> int:
        """The cluster's clock (the Session: one per step)."""
        return self._clock

    @clock.setter
    def clock(self, v: int) -> None:
        self._clock = v
        if self.ic is not None:
            self.ic.timers()

    # -- harts -----------------------------------------------------------------------

    def core_of(self, m) -> int:
        p = getattr(m, "p8700", None)
        return p.core if p is not None else 0

    def index(self, core: int, hart: int) -> int:
        return core * self.harts + hart

    def _make(self, i: int, pc: int) -> Any:
        c, h = divmod(i, self.harts)
        m = self.factory(self, c, h, pc, self._caches[c])
        p = m.p8700
        p.core, p.hart = c, h
        m.hartid = p.hartid
        m.cluster = self
        m.devices.insert(0, self.gcr)
        m.timer = self.timer
        self._io(m)
        mmu = m.mmu
        if hasattr(mmu, "ecc_check"):       # (2.2: the FTLB's ECC reports to the core)
            errs = self.errors[c]
            mmu.errs = errs
            mmu.ecc_check = errs.pe
            errs.mmus = [x for x in errs.mmus if x is not None] + [mmu]
        if hasattr(mmu, "share_ftlb"):      # (2.6: a core's harts share its FTLB)
            if self.ftlbs[c] is None:
                self.ftlbs[c] = mmu
            else:
                mmu.share_ftlb(self.ftlbs[c])
        return m

    def _io(self, m) -> None:
        """The GCR block is I/O for the hart's PMAs (uncached)."""
        m.io_regions = [r for r in m.io_regions if r[1] - r[0] != GCR_SIZE]
        m.io_regions.append((self.base, self.base + GCR_SIZE))

    def move(self, base: int) -> None:
        self.base = base
        for m in self.machines:
            self._io(m)

    def start(self, i: int) -> None:
        """Hart ``i`` starts afresh at its RESET_BASE (VP_RUN, 6.5.10)."""
        m = self._make(i, self.reset_base[i])
        self.machines[i] = m
        c, h = divmod(i, self.harts)
        self.running_mask[c] |= 1 << h
        if self.on_start is not None:
            self.on_start(i, m)
        self._lines()

    def _lines(self) -> None:
        """A hart made afresh sees the controller's outputs."""
        if self.ic is not None:
            self.ic.update()
            self.ic.timers(force=True)
            for idx, hart in self.ic.indexed():
                hart.set_interrupt(3, bool(self.ic.msip.get(idx, 0)))

    def running(self) -> list[int]:
        """The harts that execute: their core is in U5 / U6 and VP_RUNNING."""
        out = []
        for c in range(self.cores):
            if self.state[c] in (U5, U6):
                out += [self.index(c, h) for h in range(self.harts)
                        if self.running_mask[c] >> h & 1]
        return out

    def runs(self, i: int) -> bool:
        c, h = divmod(i, self.harts)
        return self.state[c] in (U5, U6) and bool(self.running_mask[c] >> h & 1)

    def boot(self) -> None:
        """The kernel / user boot (OpenSBI's boston_nascent_init and HSM
        hart_start): every core powered up and coherent, every hart running."""
        for c in range(self.cores):
            self.state[c], self.cmd[c] = U6, PWR_UP
            self.coh_en[c] = 1
            self.running_mask[c] = (1 << self.harts) - 1

    # -- LR / SC (3.2) ------------------------------------------------------------------

    def stored(self, m, pa: int) -> None:
        """Hart ``m`` stores at ``pa``: the other harts lose a reservation in
        its 64-byte granule."""
        for o in self.machines:
            if o is not m and o.reservation is not None and o.lr_granule == pa >> 6:
                o.reservation = None

    # -- the CPC (5.14.4 / 5.14.5) --------------------------------------------------------

    def mtime(self) -> int:
        if self.timectl & 1:                # (MSTOP)
            return self._mtime
        return (self._mtime + (self.clock - self._since) // self.timer_div) & M64

    def _set_mtime(self, v: int) -> None:
        self._mtime, self._since = v & M64, self.clock

    def cpc_get(self, off: int) -> int:
        if off == CPC_REV:
            return CPC_REVISION
        if off == ROCC_CTL:
            return self.rocc
        if off == MTIME:
            return self.mtime()
        if off == TIMECTL:
            return self.timectl
        if off == HRTIME:
            return self.clock & M64
        if off in self.cpc_regs:
            return self.cpc_regs[off]
        if CPC_CORE <= off < CPC_CORE + MAX_CORES * CORE_STRIDE:
            c, r = divmod(off - CPC_CORE, CORE_STRIDE)
            if c >= self.cores:
                return 0
            if r == CMD:
                return self.cmd[c]
            if r == STAT_CONF:
                return self.stat_conf(c)
            if r == VP_RUNNING:
                return self.running_mask[c]
        return 0

    def stat_conf(self, c: int) -> int:
        """Table 5.109: SEQ_STATE, COH_EN, PWUP_POLICY (core 0: 2, power up;
        the others 0), CMD."""
        return (self.state[c] << 19 | self.coh_en[c] << 11 | (2 if c == 0 else 0) << 8
                | 1 << 13 | 1 << 12 | self.cmd[c])

    def cpc_put(self, off: int, value: int, mask: int) -> None:
        if off == ROCC_CTL:
            self.rocc &= ~(value & mask)
            return
        if off == MTIME:
            self._set_mtime((self.mtime() & ~mask) | value)
            return
        if off == TIMECTL:
            stop = (value & mask) & 1
            if stop and not self.timectl & 1:
                self._mtime = self.mtime()
            elif not stop and self.timectl & 1:
                self._since = self.clock
            self.timectl = ((self.timectl & ~mask) | value) & 0xF
            return
        if off in self.cpc_regs:
            self.cpc_regs[off] = (self.cpc_regs[off] & ~mask) | value
            return
        if CPC_CORE <= off < CPC_CORE + MAX_CORES * CORE_STRIDE:
            c, r = divmod(off - CPC_CORE, CORE_STRIDE)
            if c >= self.cores:
                return
            v = value & mask
            if r == CMD and mask & 0xF:
                self.command(c, v & 0xF)
            elif r == VP_STOP and mask & 0xF:
                self.vp_stop(c, v & ((1 << self.harts) - 1))
            elif r == VP_RUN and mask & 0xF:
                self.vp_run(c, v & ((1 << self.harts) - 1))

    def command(self, c: int, cmd: int) -> None:
        """Table 5.108; reserved codes are not loaded.  ClockOff / PwrDown /
        Reset of a coherent core wait until COH_EN = 0 ("only successful if
        core_coherence_enable inactive"; Reset: "if in non-coherent mode")."""
        if cmd not in (CLOCK_OFF, PWR_DOWN, PWR_UP, RESET):
            return
        self.cmd[c] = cmd
        if cmd != PWR_UP and self.coh_en[c]:
            self.pending[c] = cmd
            return
        self._do(c, cmd)

    def _do(self, c: int, cmd: int) -> None:
        self.pending[c] = None
        if cmd == CLOCK_OFF:
            self.state[c] = U2
        elif cmd == PWR_DOWN:
            self.state[c] = D0
            self.running_mask[c] = 0
        elif cmd == PWR_UP and self.state[c] == U2:
            self.state[c] = U6 if self.coh_en[c] else U5     # (the clock back: state kept)
        elif cmd == PWR_UP and self.state[c] in (U5, U6):
            pass                            # (already up)
        else:                               # (power up from D0, or a reset)
            self.reset_core(c)
            self.cmd[c] = PWR_UP

    def reset_core(self, c: int) -> None:
        """The core's harts reset; hart 0 runs (si_core_vc_run_init)."""
        self.state[c] = U6 if self.coh_en[c] else U5
        self.rocc |= 1 << c
        self.running_mask[c] = 0
        for h in range(self.harts):
            i = self.index(c, h)
            self.machines[i] = self._make(i, self.reset_base[i])
            if self.on_start is not None:
                self.on_start(i, self.machines[i])
        self.running_mask[c] = 1
        self._lines()

    def set_coherent(self, c: int, on: bool) -> None:
        self.coh_en[c] = int(on)
        if self.state[c] in (U5, U6):
            self.state[c] = U6 if on else U5
        if not on and self.pending[c] is not None:
            self._do(c, self.pending[c])

    def vp_stop(self, c: int, bits: int) -> None:
        self.running_mask[c] &= ~bits

    def vp_run(self, c: int, bits: int) -> None:
        if self.state[c] not in (U5, U6):
            return                          # (a core that does not execute)
        for h in range(self.harts):
            if bits >> h & 1 and not self.running_mask[c] >> h & 1:
                self.start(self.index(c, h))

    # -- views ---------------------------------------------------------------------------

    def hart_name(self, i: int) -> str:
        c, h = divmod(i, self.harts)
        return f"hart {i} (core {c}, mhartid 0x{c << 4 | h:x})"

    def lines(self) -> list[str]:
        out = [f"P8700 cluster: {self.cores} core(s) x {self.harts} hart(s), GCR_BASE "
               f"0x{self.base:x}, mtime {self.mtime()}"
               f"{' (stopped)' if self.timectl & 1 else ''}"]
        for c in range(self.cores):
            harts = ", ".join(
                f"hart {h} {'running' if self.running_mask[c] >> h & 1 else 'stopped'} "
                f"(RESET_BASE 0x{self.reset_base[self.index(c, h)]:x})"
                for h in range(self.harts))
            wait = f", waiting command {self.pending[c]}" if self.pending[c] else ""
            out.append(f"  core {c}: {STATE_NAMES.get(self.state[c], self.state[c])}, "
                       f"COH_EN {self.coh_en[c]}{wait}; {harts}")
        return out


class _Timer:
    """The cluster's mtime for the harts' ``time`` (RVMachine.timer)."""

    def __init__(self, cl: Cluster) -> None:
        self.cl = cl

    def mtime(self) -> int:
        return self.cl.mtime()
