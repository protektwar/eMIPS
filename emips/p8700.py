"""The MIPS P8700's own hart registers (PLAN M25 phase 5b; MD01502, the
P8700 / P8700-F Multiprocessing System Programmer's Guide rev. 1.80:
ch. 7 the CSRs, ch. 11.1 the performance counters).

A :class:`P8700` object is the P8700-specific state of one hart
(``RVMachine(core="p8700")`` → ``m.p8700``); :data:`CSRS` are its CSRs,
which replace or join the generic hart's (:mod:`emips.rvcsr`):

- **Identification** (7.1): mvendorid = BANK 2, OFFSET 0x27 (JEDEC: MIPS
  Technologies, 0x127); marchid = MSB 1, CLASS 2 (a P-class core), UARCH
  1 (QEMU's mips-p8700: 0x8000000000000201); mimpid MAJOR / MINOR / PATCH
  / CONFIGID "from configuration" — 0 (QEMU's too); mhartid = CLUSTERNUM
  19:16 | CORENUM 11:4 | HARTNUM 3:0 (:func:`hartid`).
- **time** (7.2): not implemented.  A read the counter enables allow is a
  custom *Read Time* exception (cause 26, through mipstvec) for the
  firmware to emulate, or an illegal instruction with mipsconfig6.RDTM_II.
- **No AMOs** (rev. 1.84 1.10.2.2: "traps and emulates AMO arithmetic
  with LR/SC"; mipsconfig6.AMO_II): an AMO whose address checks pass is a custom exception with
  mcause 2 (through mipstvec), or with AMO_II an illegal instruction
  (mtvec, delegable).  LR / SC work.
- **mipstvec** (7.3.1): the vector of the custom exceptions — always
  M-mode, never delegated; MODE 0 direct, 1 vectored (BASE + 4 x cause).
  mipstval / mipsscratch (7.3.2 / 7.3.3) are read / write (the software
  table walk writes mipstval: emips.p8700mmu).
- **mipsintctl** (7.3.6): MIPS hardware interrupts 5..0 (MEI, MSI, MTI,
  SEI, STI, VSEI) go to their standard mip bits, or with the bit clear to
  mip[20] / [19] / VSTIP (6) / [18] / [17] / [16].  The interrupt
  priority is the AIA's default order (riscv-interrupts 1.0 5.1: 23, the
  watchdog's 25, 20, MEI, MSI, MTI, SEI, SSI, STI, VSEI, VSTI, LCOFI, 19,
  18, 17, 16).
- **mipsconfig5.ERL** (7.3.9): no interrupts, every access uncached;
  MRET / SRET clear it (NMI / Cache Error set it: emips.p8700ic /
  emips.p8700err).
  MDIAGL is write-1-only.  The software table walker's fields (LEVEL,
  ATYPE, XATP, G, MTW / MPTW / MPPPS): :meth:`P8700.tlb_miss`,
  :meth:`P8700.m_trap`, :meth:`P8700.mret` (emips.p8700mmu).
- **mipsconfig6** AMO_II / RDTM_II / PRI, **mipsconfig7** (DMALN / DULS:
  misaligned loads and stores trap; the rest are performance features
  eMIPS does not model, DHTW / FTLB64 / FTLBP for emips.p8700mmu, or the
  caches' bits of phase 5e),
  **mipsconfig8-11** (the prefetch trackers' registers, through 8 / 9 /
  10; PFEN), **mipsconfig1** (the L1 shapes and L2C, from the caches),
  **mipscacheerr / mipserrctl** (the core's: emips.p8700err),
  **mipsdsprambase** (stored).
- **pmacfg0-15** (7.3.16): PMA *k* belongs to PMP entry *k* (OpenSBI's
  ``mips_p8700_pmp_set``): the lowest-numbered PMP entry whose range
  holds the address gives its CCA — 0 cacheable, 2 uncached, 3 uncached
  accelerated (MIPS CCA 3 / 2 / 7); the reserved codes are uncached.
  No entry: the platform's PMAs.
- **The performance counters** (11.1): mhpmevent3-6 / mhpmcounter3-6
  (:class:`Counters`): OF, MINH / SINH / UINH, PCTD, EVENT (Table 11.4,
  :data:`EVENTS`); an overflow with OF clear sets OF and the local
  counter-overflow interrupt (mip.LCOFIP, 13).

Choices: ``sim.DEVIATIONS["P8700 hart"]``.  Sources: MD01502 rev. 1.80,
QEMU's ``target/riscv/mips_csr.c`` and mips-p8700 CPU, OpenSBI's
``platform/generic/mips``, Linux's Zalrsc patch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rvcsr import CSR, M64

if TYPE_CHECKING:
    from .rvcsr import CSRFile
    from .simrv import RV, RVMachine

#: the P8700's extensions (1.2: RV64GC Zba Zbb, and binutils' vendor ones)
EXTENSIONS = ("imafdc_zicsr_zifencei_zicntr_zihpm_zba_zbb_xmipscmov_xmipslsp_xmipscbop"
              "_xmipsexectl_xmipsstw")
#: 7.1.5 / 7.1.1 (QEMU's MIPS_VENDOR_ID and marchid)
MVENDORID = 0x2 << 7 | 0x27
CLASS_P, UARCH = 0x02, 0x01
MARCHID = 1 << 63 | CLASS_P << 8 | UARCH
MIMPID = 0

#: the custom exception codes (Table 7.4) and interrupt (Table 7.5)
INST_TLB_MISS, LOAD_TLB_MISS, READ_TIME, STORE_TLB_MISS = 24, 25, 26, 27
CACHE_ERROR = 48
SATP_TLB_MISS = 60
IMPRECISE_BUS_ERROR = 23
CAUSE_NAMES = {INST_TLB_MISS: "instruction TLB miss", LOAD_TLB_MISS: "load TLB miss",
               READ_TIME: "read time", STORE_TLB_MISS: "store TLB miss",
               CACHE_ERROR: "cache error", SATP_TLB_MISS: "satp TLB miss"}
INTERRUPT_NAMES = {IMPRECISE_BUS_ERROR: "imprecise bus error interrupt",
                   13: "local counter-overflow interrupt",
                   16: "custom interrupt 16 (VSEI routed)", 17: "custom interrupt 17 (STI routed)",
                   18: "custom interrupt 18 (SEI routed)", 19: "custom interrupt 19 (MSI routed)",
                   20: "custom interrupt 20 (MEI routed)", 6: "VSTIP (MTI routed)",
                   10: "VSEIP", 25: "watchdog interrupt"}

#: mipsconfig5 (7.3.9): the software table walker's fields (phase 5c)
ERL, MPTW, MTW, MPPPS, MDIAGL = 1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 5
LEVEL_SHIFT, XATP_SHIFT, ATYPE_SHIFT, G5 = 7, 10, 12, 1 << 14
CONFIG5_RW = ERL | MPTW | MTW | MPPPS | G5 | 3 << ATYPE_SHIFT
#: mipsconfig6 (7.3.10)
PRI, RDTM_II, AMO_II = 1 << 0, 1 << 1, 1 << 2
#: mipsconfig7 (7.3.11): read-only HCI 31, TL 8, reserved 24 / 10
DMALN, DULS = 1 << 9, 1 << 15
CONFIG7_MASK = 0x7FFFFFFF & ~(1 << 24) & ~(1 << 10) & ~(1 << 8)

#: mipsintctl (7.3.6): MIPS hardware interrupt n -> (its standard mip
#: bit, its bit when the routing bit is clear); the platform's lines come
#: as standard bits (MEIP 11, MSIP 3, MTIP 7, SEIP 9, STIP 5, VSEIP 10)
ROUTES = {5: (11, 20), 4: (3, 19), 3: (7, 6), 2: (9, 18), 1: (5, 17), 0: (10, 16)}
_BY_LINE = {std: (n, custom) for n, (std, custom) in ROUTES.items()}
INTCTL_RESET = 0x3F
#: the mip / mie bits the hardware lines may reach (+ the watchdog's 25,
#: MD01502 8.2.1.3), the counter overflow
WDT = 25
LINE_BITS = sum(1 << b for pair in ROUTES.values() for b in pair) | 1 << WDT
LCOFI = 13
#: the interrupt priority (riscv-interrupts 1.0 5.1's default order; the
#: custom 25 below 23)
PRIORITY = (23, 25, 20, 11, 3, 7, 9, 1, 5, 10, 6, 13, 19, 18, 17, 16)

#: the prefetch trackers (7.3.12-7.3.14): 2 groups x 4 entries, registers
#: 0-4 of 3 / 4 / 6 / 15 / 6 bits
TRACKER_BITS = (3, 4, 6, 15, 6)

#: pmacfg CCA codes (7.3.16.17) -> the MIPS CCA eMIPS's caches use
PMA_CCA = {0: 3, 2: 2, 3: 7}


def hartid(cluster: int, core: int, hart: int) -> int:
    """mhartid (7.1.3)."""
    return (cluster & 0xF) << 16 | (core & 0xFF) << 4 | (hart & 0xF)


class P8700:
    """The P8700 state of one hart (see the module docstring)."""

    def __init__(self, m: "RVMachine", cluster: int = 0, core: int = 0, hart: int = 0) -> None:
        self.m = m
        self.cluster, self.core, self.hart = cluster, core, hart
        self.reset()

    def reset(self) -> None:
        """The reset values (7.3: "Undefined" ones are 0, mipsintctl the
        standard routing — DEVIATIONS)."""
        self.mipstvec = self.mipstval = self.mipsscratch = 0
        self._dspram_base = 0
        #: mipscacheerr / mipserrctl when the hart has no core's (emips.p8700err)
        self._errors = None
        self.mipsintctl = INTCTL_RESET
        self.config5 = self.config6 = self.config7 = 0
        self.config8 = self.config9 = self._config11 = 0
        self.tracker = [[0] * len(TRACKER_BITS) for _ in range(8)]
        self.pmacfg = [0] * 64
        #: the hardware interrupt lines (standard mip bits)
        self.lines = 0
        self.counters = Counters(self.m)
        self._pma: tuple | None = None
        self._pma_version = 0

    @property
    def hartid(self) -> int:
        return hartid(self.cluster, self.core, self.hart)

    # -- interrupts -------------------------------------------------------------

    def set_line(self, bit: int, level: bool) -> int:
        """A platform interrupt line (a standard mip bit) changed: the mip
        bits the lines reach now (mipsintctl's routing)."""
        self.lines = self.lines | 1 << bit if level else self.lines & ~(1 << bit)
        return self.routed()

    def routed(self) -> int:
        out = 0
        lines = self.lines
        for bit in range(lines.bit_length()):
            if lines >> bit & 1:
                route = _BY_LINE.get(bit)
                if route is None:
                    out |= 1 << bit
                else:
                    n, custom = route
                    out |= 1 << (bit if self.mipsintctl >> n & 1 else custom)
        return out

    def write_intctl(self, v: int) -> None:
        self.mipsintctl = v & 0x3F
        self.m.irq_lines = self.routed()

    @property
    def erl(self) -> bool:
        return bool(self.config5 & ERL)

    # -- errors (7.3.4 / 7.3.5) ------------------------------------------------------

    def errors(self):
        """The core's mipscacheerr / mipserrctl (emips.p8700err: per core)."""
        errs = getattr(self.m.caches, "errs", None)
        if errs is not None:
            return errs
        if self._errors is None:
            from .p8700err import CoreErrors
            self._errors = e = CoreErrors()
            mmu = getattr(self.m, "mmu", None)
            if hasattr(mmu, "ecc_check"):       # (a hart on its own: its FTLB reports here)
                e.mmus, mmu.errs = [mmu], e
        return self._errors

    # -- the PMAs (7.3.16) --------------------------------------------------------

    def pma(self, pa: int) -> int | None:
        """The CCA ERL or the pmacfg of the PMP entry holding ``pa`` gives
        (None: the platform's PMAs)."""
        if self.config5 & ERL:
            return 2
        p = self.m.pmp
        key = (p.version, self._pma_version)
        if self._pma is None or self._pma[0] != key:
            regions = []
            for k in range(p.entries):
                rng = p._range(k)
                if rng is not None:
                    regions.append((rng[0], rng[1], PMA_CCA.get(self.pmacfg[k] & 7, 2)))
            self._pma = (key, regions)
        for lo, hi, cca in self._pma[1]:
            if lo <= pa < hi:
                return cca
        return None

    def read_pmacfg(self, n: int) -> int:
        per = self.m.xlen // 8
        return sum(self.pmacfg[n * 4 + k] << (8 * k) for k in range(per))

    def write_pmacfg(self, n: int, v: int) -> None:
        for k in range(self.m.xlen // 8):
            self.pmacfg[n * 4 + k] = (v >> (8 * k)) & 0x0F      # (7:4 reserved: 0)
        self._pma_version += 1

    # -- mipsconfig ---------------------------------------------------------------

    def config1(self) -> int:
        """7.3.8: the L1 shapes (sets 2 ** (S + 6), 7 = 32; line 2 ** (L +
        1); ways A + 1) and L2C, from the hart's caches."""
        c = self.m.caches
        v = 0
        if c is None:
            return v
        for cache, shift in ((c.i, 16), (c.d, 7)):
            g = getattr(cache, "geometry", None) if cache is not None else None
            if g is None:
                continue
            sets = 7 if g.sets == 32 else max(g.sets.bit_length() - 7, 0)
            line = g.line.bit_length() - 2
            v |= (sets << 6 | line << 3 | (g.ways - 1)) << shift
        if c.l2 is not None:
            v |= 1 << 31
        return v

    def write_config5(self, v: int) -> None:
        """7.3.9: ERL, MPTW / MTW / MPPPS, G, ATYPE R/W; LEVEL WARL (the
        levels Sv39 / Sv48 use: 0-3); XATP reads 0 (no H extension: only
        satp); MDIAGL write-1-only; PGVA / MPPV (H only) 0."""
        level = (v >> LEVEL_SHIFT) & 7
        if level > 3:
            level = (self.config5 >> LEVEL_SHIFT) & 7
        self.config5 = (v & CONFIG5_RW) | level << LEVEL_SHIFT | (self.config5 | v) & MDIAGL

    def write_config7(self, v: int) -> None:
        self.config7 = v & CONFIG7_MASK
        mmu = getattr(self.m, "mmu", None)
        if hasattr(mmu, "config7_written"):
            mmu.config7_written(self.m)

    def tlb_miss(self, s) -> None:
        """A software-walk TLB miss is taken (7.3.2 / 7.3.9): mipstval = the
        missing PTE's address, LEVEL / ATYPE / XATP (satp) / G."""
        from .p8700mmu import ATYPE
        self.mipstval = s.pte_addr
        c = self.config5 & ~(G5 | 3 << ATYPE_SHIFT | 3 << XATP_SHIFT | 7 << LEVEL_SHIFT)
        self.config5 = (c | (G5 if s.g else 0) | ATYPE[s.kind] << ATYPE_SHIFT
                        | s.level << LEVEL_SHIFT)

    def m_trap(self, custom: bool, mpp: int) -> None:
        """An M-mode trap (7.3.9): MPPPS = MPP was S on a mipstvec exception
        or with MTW = 1; MPTW = MTW; MTW = 0."""
        c = self.config5
        if custom or c & MTW:
            c = (c & ~MPPPS) | (MPPPS if mpp == 1 else 0)
        c = (c & ~MPTW) | (MPTW if c & MTW else 0)
        self.config5 = c & ~MTW

    def mret(self) -> bool:
        """MRET (7.3.9): MTW = MPTW, ERL = 0; True when MPPPS asks MPP = S
        (MPPPS is then cleared)."""
        c = self.config5
        s_mode = bool(c & MPPPS)
        c = (c & ~(MTW | MPPPS | ERL)) | (MTW if c & MPTW else 0)
        self.config5 = c
        return s_mode

    def write_config8(self, v: int) -> None:
        self.config8 = v & 0x7FF

    def l1pf(self):
        """The core's L1 prefetcher (emips.p8700mem; a cluster's core with a
        data cache), None otherwise."""
        c = self.m.caches
        return getattr(c, "l1pf", None) if c is not None else None

    def _tracker(self) -> tuple[list[int], int] | None:
        sel = self.config8 & 0xFF
        if sel >= len(TRACKER_BITS):
            return None
        k = self.config8 >> 8 & 7
        pf = self.l1pf()
        return (pf.trackers[k].regs if pf is not None else self.tracker[k]), sel

    @property
    def config11(self) -> int:
        """mipsconfig11.PFEN (7.3.15): bit k enables tracker k (the core's)."""
        pf = self.l1pf()
        return pf.pfen if pf is not None else self._config11

    @config11.setter
    def config11(self, v: int) -> None:
        pf = self.l1pf()
        if pf is not None:
            pf.pfen = v & 0xFF
        self._config11 = v & 0xFF

    @property
    def mipsdsprambase(self) -> int:
        sp = self.m.spram                       # (per core: its harts share it)
        return sp.raw if sp is not None else self._dspram_base

    @mipsdsprambase.setter
    def mipsdsprambase(self, v: int) -> None:
        """7.3.7: the core's DSPRAM moves to the base (emips.p8700mem)."""
        self._dspram_base = v & ((1 << 64) - 1)
        sp = self.m.spram
        if sp is not None:
            sp.set_base(self._dspram_base)

    def write_config9(self, v: int) -> None:
        self.config9 = v & 0x7FFF
        t = self._tracker()
        if t is not None:
            regs, sel = t
            regs[sel] = v & ((1 << TRACKER_BITS[sel]) - 1)

    def config10(self) -> int:
        t = self._tracker()
        return 0 if t is None else t[0][t[1]]

    # -- the custom exceptions ----------------------------------------------------

    def read_time(self, insn: int) -> None:
        """A read of ``time`` the counter enables allow (7.2, 7.3.10)."""
        from .simrv import ILLEGAL, RVTrap
        if self.config6 & RDTM_II:
            raise RVTrap(ILLEGAL, insn)
        raise RVTrap(READ_TIME, insn, custom=True)

    def amo(self, insn: int) -> None:
        """An AMO whose address checks passed (7.3.10 AMO_II)."""
        from .simrv import ILLEGAL, RVTrap
        raise RVTrap(ILLEGAL, insn, custom=not self.config6 & AMO_II)

    @property
    def misaligned_traps(self) -> bool:
        return bool(self.config7 & (DMALN | DULS))

    def vector(self, cause: int) -> int:
        """The PC of a custom exception (7.3.1)."""
        base = self.mipstvec & ~3
        return base + 4 * cause if self.mipstvec & 3 == 1 else base


# ------------------------------------------------------------------ counters

#: Table 11.4: event -> (name, what eMIPS counts it from): "retire" (each
#: instruction), "cycle" (each clock), "trap", "pipeline" (the 5-stage
#: pipeline's cycle), "cache" (the caches' and the TLBs' counters), "" (never happens on
#: this model: listed so the field accepts it)
EVENTS: dict[int, tuple[str, str]] = {
    1: ("num_grad", "retire"), 2: ("one_grad", "cycle"), 3: ("two_grad", "cycle"),
    4: ("three_grad", "cycle"), 5: ("four_grad", "cycle"), 6: ("no_grad", "cycle"),
    7: ("alu_grad", "retire"), 8: ("lsu_grad", "retire"), 9: ("cti_grad", "retire"),
    10: ("mdu_grad", "retire"), 11: ("fpu_grad", "retire"), 12: ("load_grad", "retire"),
    13: ("store_grad", "retire"), 14: ("no_isu", "cycle"), 15: ("one_isu", "cycle"),
    16: ("two_isu", "cycle"), 17: ("three_isu", "cycle"), 18: ("four_isu", "cycle"),
    19: ("isu_block", "pipeline"), 20: ("dec_stall", "pipeline"), 21: ("dmap_stall", ""),
    22: ("ibfr_empty", "pipeline"), 23: ("itrkr_num_replay", "pipeline"), 24: ("br_grad", "retire"),
    25: ("br_miss_grad", "pipeline"), 26: ("jr_ret_grad", "retire"),
    27: ("jr_ret_miss_grad", "pipeline"), 28: ("jr_grad", "retire"),
    29: ("jr_miss_grad", "pipeline"), 30: ("br_t_grad", "retire"),
    31: ("br_nt_grad", "retire"), 32: ("redirect", "pipeline"),
    33: ("num_exceptions", "trap"), 34: ("exe_redir", "pipeline"), 35: ("load_blocked", "pipeline"),
    36: ("sync_blocked", "pipeline"), 37: ("rftch_inbnd_oel", ""), 38: ("rftch_inbnd_misalgn", "pipeline"),
    64: ("dtlb_lookup", "cache"), 65: ("dtlb_miss_new", "cache"), 66: ("dtlb_miss_merge", ""),
    67: ("bond_load", "pipeline"), 68: ("bond_store", "pipeline"), 69: ("total_dcache_lookups", "cache"),
    70: ("loads_dcache_lookup", "cache"), 71: ("stores_dcache_lookup", "cache"),
    72: ("total_dcache_misses", "cache"), 73: ("load_dcache_misses", "cache"),
    74: ("store_dcache_misses", "cache"), 75: ("smb_full", ""),
    76: ("DCACHE load Retries", ""),
    **{77 + k: (f"L1 Prefetch from Prefetch tracker {k}", "prefetch") for k in range(8)},
    85: ("rftch_inbnd_prd", ""), 128: ("all harts stalled", "pipeline"),
    129: ("utb_access", "cache"), 130: ("utlb MMU wait stalls", ""),
    131: ("utb_miss", "cache"), 132: ("ica_access", "cache"),
    133: ("ica_miss", "cache"), 134: ("ibuff_cred_stall", ""),
    135: ("PCBuffer credit stalls", ""), 136: ("hart stalls", "pipeline"),
    137: ("redirect_stall", "pipeline"), 138: ("ica_miss_stall", "pipeline"),
    139: ("uncached_stall", ""), 140: ("prb_full", ""), 141: ("l1btb_rps_hit", "pipeline"),
    142: ("l1btb_prs_mispred", "pipeline"), 143: ("rps_hit_l1btb_miss", "pipeline"),
    144: ("ica_way_mispred", ""), 145: ("jtlb_miss", "cache"),
    146: ("no_spec_fetch_maar", ""), 147: ("fetch_kill_cti", "pipeline"),
    148: ("no_ifu_2_idu_inst", "pipeline"), 149: ("l1btb_non_rps_hit", "pipeline"),
    150: ("l1btb_non_rps_mispred", "pipeline"), 151: ("jal_b_hit_l1btb_miss", "pipeline"),
    152: ("l1btb_hit_masked", ""), 160: ("PDTrace backpressure stall", ""),
    161: ("PDTrace backpressure stall cycles", ""), 162: ("PDTrace overflow", ""),
    163: ("PDTrace LSU backpressure stall", ""),
}
#: mhpmevent (Figure 11.1): OF 63, MINH 62, SINH 61, UINH 60, PCTD 55, EVENT 7:0
OF, MINH, SINH, UINH, PCTD = 1 << 63, 1 << 62, 1 << 61, 1 << 60, 1 << 55
_INHIBIT = {3: MINH, 1: SINH, 0: UINH}

_LOADS = frozenset({"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "flw", "fld", "lr.w", "lr.d",
                    "mips.ldp", "mips.lwp"})
_STORES = frozenset({"sb", "sh", "sw", "sd", "fsw", "fsd", "sc.w", "sc.d", "mips.sdp",
                     "mips.swp"})
_BRANCHES = frozenset({"beq", "bne", "blt", "bge", "bltu", "bgeu"})


def _classes(i: "RV", pc: int, nxt: int) -> tuple[int, ...]:
    """The retire events of instruction ``i`` (Table 11.4 7-13, 24-31)."""
    row = i.base
    name, ext = row.name, row.ext
    out = [1]
    if name in _LOADS:
        out += [8, 12]
    elif name in _STORES:
        out += [8, 13]
    elif name.startswith("amo") or name == "mips.pref":
        out.append(8)
    elif name in _BRANCHES:
        out += [9, 24, 30 if nxt != (pc + i.size) & M64 else 31]
    elif name in ("jal", "jalr"):
        out.append(9)
        if name == "jalr":
            out.append(28)
            f = i.f
            if f.get("rd", 0) == 0 and f.get("rs1", 0) in (1, 5) and f.get("imm", 0) == 0:
                out.append(26)                  # (a return: unprivileged 2.5.1's hints)
    elif "m" in ext:
        out.append(10)
    elif "f" in ext or "d" in ext:
        out.append(11)
    elif name.startswith(("csr", "fence")) or name in ("ecall", "ebreak", "mret", "sret",
                                                         "wfi", "sfence.vma", "dret"):
        pass
    else:
        out.append(7)
    return tuple(out)


#: the TLB events (Table 11.4; emips.p8700mmu's counters): (buffer, hits /
#: misses)
_TLB_KEYS = {64: ("dtlb", (0, 1)), 65: ("dtlb", (1,)), 129: ("itlb", (0, 1)),
             131: ("itlb", (1,)), 145: ("tlb", (1,))}
_CACHE_KEYS = {69: ("d", ("load", "store"), True, True), 70: ("d", ("load",), True, True),
               71: ("d", ("store",), True, True), 72: ("d", ("load", "store"), False, True),
               73: ("d", ("load",), False, True), 74: ("d", ("store",), False, True),
               132: ("i", ("fetch",), True, True), 133: ("i", ("fetch",), False, True)}


class Counters:
    """mhpmevent3-6 / mhpmcounter3-6 of one hart (11.1)."""

    FIRST, N = 3, 4

    def __init__(self, m: "RVMachine") -> None:
        self.m = m
        self.event = [0] * self.N
        self.count = [0] * self.N
        #: some counter has a modelled event
        self.armed = False
        #: instructions graduated in this clock cycle
        self.graduated = 0
        #: the sampled sources' last totals (cache / TLB events)
        self._last: dict[int, int] = {}
        #: the P8700 pipeline (emips.pipe8700) drives the issue / graduation
        #: events: an instruction's retire events wait in :attr:`last` for it
        #: to graduate (:meth:`graduate`)
        self.pipeline = False
        self.last: tuple | None = None

    def write_event(self, k: int, v: int) -> None:
        ev = v & 0xFF
        if ev not in EVENTS:
            ev = 0                              # (EVENT is WARL: the table's codes)
        self.event[k] = (v & (OF | MINH | SINH | UINH | PCTD)) | ev
        self._last.pop(k, None)
        self._rearm()

    def _rearm(self) -> None:
        self.armed = any(EVENTS.get(e & 0xFF, ("", ""))[1] for e in self.event)
        for k, e in enumerate(self.event):
            if EVENTS.get(e & 0xFF, ("", ""))[1] == "cache":
                self._last[k] = self._sample(e & 0xFF)

    def _counts(self, k: int) -> bool:
        m = self.m
        if m.csrs.mcountinhibit >> (self.FIRST + k) & 1:
            return False
        return not self.event[k] & _INHIBIT.get(m.priv, 0)

    def add(self, event: int, n: int = 1) -> None:
        """``n`` occurrences of ``event``."""
        for k, e in enumerate(self.event):
            if e & 0xFF == event and n and self._counts(k):
                self._bump(k, n)

    def _bump(self, k: int, n: int) -> None:
        v = self.count[k] + n
        if v > M64:
            if not self.event[k] & OF:          # 11.1.2 / Sscofpmf: an interrupt once
                self.event[k] |= OF
                self.m.csrs.mip_sw |= 1 << LCOFI
            v &= M64
        self.count[k] = v

    # -- the sources -------------------------------------------------------------

    def retired(self, i: "RV", pc: int, nxt: int) -> None:
        if self.pipeline:                       # (counted when it graduates)
            self.last = _classes(i, pc, nxt)
            return
        self.graduated += 1
        for ev in _classes(i, pc, nxt):
            self.add(ev)

    def graduate(self, events: tuple | None) -> None:
        """An instruction graduates in the P8700 pipeline: its retire
        events (taken from :attr:`last` when it executed)."""
        for ev in events or ():
            self.add(ev)

    def trap(self) -> None:
        self.add(33)

    def tick(self, cycles: int) -> None:
        """The cycle events (``cycles`` clocks; the graduations happened
        in the last of them) and the sampled ones."""
        g, self.graduated = self.graduated, 0
        if self.pipeline:                       # (the pipeline counts 2-6 / 14-18)
            cycles = 0
        elif cycles > 1:
            self.add(6, cycles - 1)
            self.add(14, cycles - 1)
        if not cycles:
            pass
        elif g:
            self.add(1 + min(g, 4), 1)          # one_grad ... four_grad
            self.add(14 + min(g, 4), 1)         # one_isu ... four_isu
        else:
            self.add(6)
            self.add(14)
        for k, e in enumerate(self.event):
            ev = e & 0xFF
            if EVENTS.get(ev, ("", ""))[1] == "cache":
                now = self._sample(ev)
                d = now - self._last.get(k, now)
                self._last[k] = now
                if d > 0 and self._counts(k):
                    self._bump(k, d)

    def _sample(self, ev: int) -> int:
        m = self.m
        st = getattr(m.mmu, "stats", None)
        if ev in _TLB_KEYS:                     # (the MMU's buffers: emips.p8700mmu)
            if not isinstance(st, dict):
                return getattr(m.mmu, "misses", 0) if ev == 145 else 0
            name, which = _TLB_KEYS[ev]
            return sum(st[name][k] for k in which)
        c = m.caches
        if c is None:
            return 0
        which, kinds, hits, misses = _CACHE_KEYS[ev]
        cache = c.i if which == "i" else c.d
        if cache is None:
            return 0
        k = cache.stats.kinds
        return sum((k[x][0] if hits else 0) + (k[x][1] if misses else 0) for x in kinds
                   if x in k)

    def lines(self) -> list[str]:
        """The counters (the views)."""
        out = []
        for k in range(self.N):
            e = self.event[k]
            name, src = EVENTS.get(e & 0xFF, ("none", ""))
            flags = " ".join(f for f, b in (("OF", OF), ("MINH", MINH), ("SINH", SINH),
                                             ("UINH", UINH)) if e & b)
            note = "" if not e & 0xFF or src else " (not modelled: never counts)"
            out.append(f"mhpmcounter{self.FIRST + k} = {self.count[k]}  event {e & 0xFF} "
                       f"{name}{note}{'  ' + flags if flags else ''}")
        return out


# ------------------------------------------------------------------ the CSRs

def _p(c: "CSRFile") -> bool:
    return c.m.p8700 is not None


def _attr(name: str):
    return lambda c: getattr(c.m.p8700, name)


def _store(name: str, mask: int = M64):
    return lambda c, v: setattr(c.m.p8700, name, v & mask)


def _w_tvec(c: "CSRFile", v: int) -> None:
    c.m.p8700.mipstvec = v & ~2                 # (MODE 2 / 3 reserved: WARL, as mtvec)


def _r_counter(k: int):
    return lambda c: c.m.p8700.counters.count[k]


def _w_counter(k: int):
    def w(c: "CSRFile", v: int) -> None:
        c.m.p8700.counters.count[k] = v & M64
    return w


def _r_event(k: int):
    return lambda c: c.m.p8700.counters.event[k]


def _w_event(k: int):
    return lambda c, v: c.m.p8700.counters.write_event(k, v)


CSRS: list[CSR] = [
    CSR(0xF11, "mvendorid", _p, lambda c: MVENDORID),
    CSR(0xF12, "marchid", _p, lambda c: MARCHID),
    CSR(0xF13, "mimpid", _p, lambda c: MIMPID),
    CSR(0xF14, "mhartid", _p, lambda c: c.m.p8700.hartid),
    CSR(0xC01, "time", _p, lambda c: c.m.mtime),     # (an instruction: CSRFile.check)
    CSR(0x7C0, "mipstvec", _p, _attr("mipstvec"), _w_tvec),
    CSR(0x7C3, "mipstval", _p, _attr("mipstval"), _store("mipstval")),
    CSR(0x7C4, "mipsscratch", _p, _attr("mipsscratch"), _store("mipsscratch")),
    CSR(0x7C5, "mipscacheerr", _p, lambda c: c.m.p8700.errors().cacheerr,
        lambda c, v: setattr(c.m.p8700.errors(), "cacheerr", v & 0xFFFFFFFF)),
    CSR(0x7C6, "mipserrctl", _p, lambda c: c.m.p8700.errors().errctl,
        lambda c, v: c.m.p8700.errors().write_errctl(v)),
    CSR(0x7CB, "mipsintctl", _p, _attr("mipsintctl"), lambda c, v: c.m.p8700.write_intctl(v)),
    CSR(0x7CC, "mipsdsprambase", _p, _attr("mipsdsprambase"), _store("mipsdsprambase")),
    CSR(0x7CF, "mipsdebug", _p, lambda c: c.m.hybrid.read(),       # (7.4; LF16: the
        lambda c, v: c.m.hybrid.write(v)),                          # number is assumed)
    CSR(0x7D1, "mipsconfig1", _p, lambda c: c.m.p8700.config1()),
    CSR(0x7D5, "mipsconfig5", _p, _attr("config5"), lambda c, v: c.m.p8700.write_config5(v)),
    CSR(0x7D6, "mipsconfig6", _p, _attr("config6"),
        _store("config6", PRI | RDTM_II | AMO_II)),
    CSR(0x7D7, "mipsconfig7", _p, _attr("config7"), lambda c, v: c.m.p8700.write_config7(v)),
    CSR(0x7D8, "mipsconfig8", _p, _attr("config8"), lambda c, v: c.m.p8700.write_config8(v)),
    CSR(0x7D9, "mipsconfig9", _p, _attr("config9"), lambda c, v: c.m.p8700.write_config9(v)),
    CSR(0x7DA, "mipsconfig10", _p, lambda c: c.m.p8700.config10(), lambda c, v: None),
    CSR(0x7DB, "mipsconfig11", _p, _attr("config11"), _store("config11", 0xFF)),
    *(CSR(0x7E0 + n, f"pmacfg{n}", lambda c, n=n: _p(c) and (c.m.xlen == 32 or n % 2 == 0),
          lambda c, n=n: c.m.p8700.read_pmacfg(n),
          lambda c, v, n=n: c.m.p8700.write_pmacfg(n, v)) for n in range(16)),
    *(CSR(0x323 + k, f"mhpmevent{3 + k}", _p, _r_event(k), _w_event(k))
      for k in range(Counters.N)),
    *(CSR(0xB03 + k, f"mhpmcounter{3 + k}", _p, _r_counter(k), _w_counter(k))
      for k in range(Counters.N)),
    *(CSR(0xC03 + k, f"hpmcounter{3 + k}", lambda c: _p(c) and c.zihpm, _r_counter(k))
      for k in range(Counters.N)),
]
NAMES = {c.num: c.name for c in CSRS}
