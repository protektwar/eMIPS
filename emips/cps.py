"""A MIPS Coherent Processing System (PLAN M11 phase 1): N cores sharing
physical memory and the devices, with the Coherence Manager's Global
Control Registers (GCR) and the Cluster Power Controller (CPC) of the
1004K / interAptiv / P5600 generation (CM2).

Sources: *Boot-MIPS: Example Boot Code for MIPS Cores* (MD00901 rev.
1.20: the CPS boot flow and its addresses — GCR at 0x1FBF8000, CPC at
0x1BDE0000, GIC at 0x1BDC0000), the CM2 / CPC register layouts of the
Linux MIPS CPS support (``arch/mips/include/asm/mips-cm.h``,
``mips-cpc.h``, ``arch/mips/kernel/smp-cps.c``), the MIPS CPS training
notes, checked after M12 against the P5600 manual (MD01025: ch. 10 the
CPC, ch. 11 the CM2).

GCR (32 KB at GCR_BASE)::

    0x0000 GCR_CONFIG     PCORES (7:0) = cores - 1, NUMIOCU, ADDR_REGIONS
    0x0008 GCR_BASE       the GCRs' physical address (31:15)
    0x0010 GCR_CONTROL    (SYNCCTL 16, SYNCDIS 5, IVU_EN 4, SHST_EN 3,
                          PARK_EN 2, MMIO_LIMIT_DIS 1, SPEC_READ_EN 0 = 1)
    0x0020 GCR_ACCESS     (a requester may write the GCRs; reset 0xFF)
    0x0030 GCR_REV        CM2 = 6.0
    0x0080 GCR_GIC_BASE   0x0088 GCR_CPC_BASE  (base, bit 0 = enable)
    0x0090.. GCR_REGn_BASE / MASK          0x00D0 / 0x00F0 GIC / CPC status
    0x000C, 0x0054, 0x0064, 0x0074, 0x0084, 0x008C, 0x0094 ... 0x00CC: the
           _UPPER halves of the base registers (40-bit physical addresses,
           MD01025 / MD01138 Table 6.5: PA 39:32; GCR_BASE_UPPER 35:32)
    0x0060 GCR_CUSTOM_BASE  0x0068 GCR_CUSTOM_STATUS (no custom block)
    0x2000 core-local block (the accessing core), 0x4000 core-other block
           (the core in the local GCR_Cx_OTHER.CORENUM):
           0x08 COHERENCE  0x10 CONFIG  0x18 OTHER  0x20 RESET_BASE
           0x28 ID  0x30 RESET_EXT_BASE  (0x00 is reserved in a CM2: the
           CM1's RESET_RELEASE — a core starts through the CPC)

CPC (32 KB at GCR_CPC_BASE when enabled)::

    0x0000 CPC_ACCESS  0x0008 SEQDEL  0x0010 RAIL  0x0018 RESETLEN
    0x0020 REVISION
    0x2000 core-local / 0x4000 core-other (CPC_Cx_OTHER.CORENUM 23:16):
           0x00 CMD (1 ClockOff, 2 PwrDown, 3 PwrUp, 4 Reset; reads the
                last executed command, PwrUp after a Reset)
           0x08 STAT_CONF (SEQSTATE 22:19: D0 powered down, U2 clock off
                after power-up, U5 / U6 (non-)coherent execution, D2 clock
                off from execution; bit 10 = 1, PWUP_POLICY 9:8, the most
                recently placed command 3:0)  0x10 OTHER

A core that is powered up or reset starts at its reset vector
(GCR_Cx_RESET_BASE, 0xBFC00000 after reset; its BEV exception vectors
move with it) in the hardware reset state (a fresh core: CP0 reset
values, caches and TLB undefined), non-coherent (SEQSTATE U5); it reads
U6 once it has joined the coherence domain (GCR_Cx_COHERENCE, as Linux's
``cps-vec.S`` does on a CM2).  ClockOff / PwrDown / Reset of a core in
the domain wait until it leaves it (MD01025 10.4: "the command remains
inactive until the protocol barriers subside"; Reset "if in non-coherent
mode").  ClockOff of a running core: D2 (PwrUp gives its clock back,
its state kept); of a powered-down one: U2 (power on, clock off; PwrUp
or Reset then reset it).  At power-on core 0 runs and the others are
powered down (D0; PWUP_POLICY): "only one processor should be powered up
when power is first applied" (Boot-MIPS 4.32).

eMIPS choices (``sim.DEVIATIONS`` "CPS"): the CPC sequences at once (a
command takes effect before the next cycle, no SEQDEL / RAIL / RESETLEN
delays, no transitional states, a command is never preempted); the
reset base is taken when a core starts (GCR_Cx_RESET_EXT_BASE.
UseExceptionBase = 1: GCR_Cx_RESET_BASE with bits 31:30 forced to 10,
else 0xBFC00000; the BEV overlay fields are stored, not used); the
address regions (GCR_REGn) are stored, used only for the L2's CCA
override / DROP_L2; MT cores (``vpes``): VP = core * VPEs + VPE; no
IOCU.  The GIC is :mod:`emips.gic` (GCR_GIC_STATUS.EX = 1, GCR_GIC_BASE
enables it).
"""

from __future__ import annotations

from typing import Any, Callable

M32 = 0xFFFFFFFF
#: the cores of a CPS: up to six (the P5600 / P6600, MD01024 / MD01137;
#: the 1004K / interAptiv up to four)
MAX_CORES = 6
#: the VPs (cores x VPEs) the boot firmware's GIC set-up serves (Boot-MIPS
#: init_gic: sources 24 + n / 32 + n for VP n < 8; the interAptiv's 4 x 2)
MAX_VPS = 8
#: Boot-MIPS (MD00901 4.5): the platform addresses (physical)
GCR_BASE_PA = 0x1FBF8000
CPC_BASE_PA = 0x1BDE0000
GIC_BASE_PA = 0x1BDC0000
GCR_SIZE = CPC_SIZE = 0x8000
#: the CM revision (Linux CM_REV_CM2 = 6.0)
CM2_REV = 0x0600
#: the reset vector of every core (GCR_Cx_RESET_BASE after reset)
RESET_VECTOR = 0xBFC00000

# GCR offsets
GCR_CONFIG, GCR_BASE, GCR_CONTROL, GCR_ACCESS, GCR_REV = 0x00, 0x08, 0x10, 0x20, 0x30
GCR_GIC_BASE, GCR_CPC_BASE = 0x80, 0x88
GCR_REGS = tuple(range(0x90, 0xD0, 8))                 # REG0..3 BASE / MASK
#: the CM's own registers (emips.coherence): offset -> CoherenceManager field
GCR_ERROR_MASK, GCR_ERROR_CAUSE = 0x40, 0x48
GCR_ERROR_ADDR, GCR_ERROR_MULT, GCR_L2_ONLY_SYNC_BASE = 0x50, 0x58, 0x70
#: the _UPPER registers (MD01138 Table 6.5): offset -> writable bits (PA
#: 39:32; GCR_BASE_UPPER 4:0, a 36-bit GCR base)
GCR_BASE_UPPER, GCR_ERROR_ADDR_UPPER, GCR_CUSTOM_BASE_UPPER = 0x0C, 0x54, 0x64
GCR_L2_ONLY_SYNC_BASE_UPPER, GCR_GIC_BASE_UPPER, GCR_CPC_BASE_UPPER = 0x74, 0x84, 0x8C
UPPER = {GCR_BASE_UPPER: 0x1F, GCR_ERROR_ADDR_UPPER: 0xFF, GCR_CUSTOM_BASE_UPPER: 0xFF,
         GCR_L2_ONLY_SYNC_BASE_UPPER: 0xFF, GCR_GIC_BASE_UPPER: 0xFF,
         GCR_CPC_BASE_UPPER: 0xFF, **{o: 0xFF for o in range(0x94, 0xD0, 8)}}
#: the custom GCR block's base (31:16; GGU_EN reads 0: none is attached,
#: GCR_CUSTOM_STATUS.GGU_EX = 0)
GCR_CUSTOM_BASE, GCR_CUSTOM_STATUS = 0x60, 0x68
_CM_REGS = {GCR_ERROR_MASK: "error_mask",
            GCR_ERROR_CAUSE: "error_cause", GCR_ERROR_ADDR: "error_addr",
            GCR_ERROR_MULT: "error_mult", GCR_L2_ONLY_SYNC_BASE: "l2_sync_base"}
GCR_GIC_STATUS, GCR_CPC_STATUS = 0xD0, 0xF0
CORE_LOCAL, CORE_OTHER = 0x2000, 0x4000
CX_COHERENCE, CX_CONFIG, CX_OTHER = 0x08, 0x10, 0x18
CX_RESET_BASE, CX_ID, CX_RESET_EXT_BASE = 0x20, 0x28, 0x30
# CPC offsets
CPC_ACCESS, CPC_SEQDEL, CPC_RAIL, CPC_RESETLEN = 0x00, 0x08, 0x10, 0x18
CPC_REVISION = 0x20
#: GCR_CONTROL fields (MD01025 Table 11.23): R/W bits, reset value
GCR_CONTROL_MASK, GCR_CONTROL_RESET = 0x000100FF, 0x1
CTL_SYNCCTL, CTL_SHST_EN = 1 << 16, 1 << 3
#: GCR_Cx_RESET_EXT_BASE (Table 11.65): EVAReset 31, UseExceptionBase 30,
#: BEVExceptionBaseMask 27:20, BEVExceptionBasePA 7:1, PRESENT 0 (reads 1)
EXT_BASE_MASK, EXT_BASE_RESET, EXT_UEB = 0xC0FF00FE, 0x40000001, 1 << 30
CPC_CX_CMD, CPC_CX_STAT_CONF, CPC_CX_OTHER = 0x00, 0x08, 0x10
#: CPC_Cx_CMD
CMD_CLOCKOFF, CMD_PWRDOWN, CMD_PWRUP, CMD_RESET = 1, 2, 3, 4
#: CPC_Cx_STAT_CONF.SEQSTATE (Linux mips-cpc.h)
SEQ_D0, SEQ_U2, SEQ_U3, SEQ_U5, SEQ_U6, SEQ_D2 = 0x0, 0x3, 0x4, 0x6, 0x7, 0xA
SEQ_NAMES = {0x0: "D0 powered down", 0x1: "U0", 0x2: "U1", 0x3: "U2 clock off",
             0x4: "U3 reset", 0x5: "U4", 0x6: "U5 non-coherent execution",
             0x7: "U6 coherent execution", 0x8: "D1", 0x9: "D3", 0xA: "D2"}
#: the CPC revision this model reports (major 1, minor 0)
CPC_REV = 0x0100
#: the CM's transaction targets (its error registers; mips-cm.c cm2_tr)
TR_GCR, TR_GIC, TR_CPC = 1, 2, 5

# -- the CM3 (the I6400: MD01196 ch. 5-7, Linux mips-cm.h / mips-cpc.h CM3
# fields, QEMU's Boston CPS; the P8700's CM3.5 pages for VP_RUN / RESET_BASE)
#: the CM3's revision (Linux CM_REV_CM3 = 8.0; QEMU's gcr-rev default)
CM3_REV = 0x0800
#: VPs a CM3 system can have: GIC_SH_MAPn_VP's 24 bits (MD01196 7.2.5.1,
#: 6 cores x 4 VP numbers)
MAX_VPS_CM3 = 24
#: CM3 global registers (MD01196 5.2-5.12, Linux mips-cm.h): the access
#: privilege register moved to 0x120 (cores 5:0, IOCUs 17:16; reset 0x3003F)
GCR_ACCESS_CM3 = 0x120
GCR_L2_CONFIG, GCR_SYS_CONFIG2 = 0x130, 0x150
GCR_L2_PFT_CONTROL, GCR_L2_PFT_CONTROL_B = 0x300, 0x308
GCR_SEM, GCR_BEV_BASE = 0x640, 0x680
GCR_MMIO_REQ_LIMIT = 0x6F8
#: GCR_MMIOn_BOTTOM / TOP (5.12.1: 0x700-0x738, bounds 47:16)
GCR_MMIO = tuple(range(0x700, 0x740, 8))
#: GCR_SEM.SEM_LOCK (5.8)
SEM_LOCK = 1 << 31
#: GCR_CONFIG (5.2): an Inter-Thread Communication Unit is present (31)
CONFIG_ITU = 1 << 31
#: GCR_Cx_COHERENCE.COHEN (Linux CM3_GCR_Cx_COHERENCE_COHEN; MD01196 5.6)
COHEN = 1
#: GCR_CL_REDIRECT (5.5.1, per VP): CORE 13:8, VP 2:0, BLOCK 25:24 (0 the
#: core-local block of that core / VP, 1 the global block)
REDIRECT_MASK = 0x03003F07
#: GCR_CL_RESET_BASE (per VP): BEVExcBase 31:12, RESET_BASE_MODE 1 (5.5.2:
#: 0 = the lower 512 MB, 1 = anywhere in the 4 GB)
RESET_BASE_MODE = 1 << 1
#: the ring-bus IDs a core-other access can redirect to (Table 5.1)
RING_IOCU0, RING_IOCU1, RING_MEMORY, RING_CM, RING_DBU = 0x10, 0x11, 0x1A, 0x20, 0x23
#: the CPC's CM3 registers (MD01196 6.2): global, then per domain
CPC_CC_CTL, CPC_PWRUP_CTL, CPC_RES_REL, CPC_ROCC_CTL, CPC_PRESCALE = 0x28, 0x30, 0x38, 0x40, 0x48
CPC_CX_CC_CTL, CPC_VP_STOP, CPC_VP_RUN, CPC_VP_RUNNING, CPC_RAM_SLEEP = 0x18, 0x20, 0x28, 0x30, 0x50
#: CPC_CC_CTL: SET_CLK_RATIO 8, CLK_CHANGE_ACTIVE 10, CC_DELAY 29:20 (reset 80)
CC_SET, CC_DELAY_MASK, CC_DELAY_RESET = 1 << 8, 0x3FF << 20, 80 << 20
#: CPC_PRESCALE_CC_CTL (6.2.7): CLK_PRESCALE 7:0 (written while CHANGE_EN
#: 8 is set), PRESCALE_CLK_RATIO 23:16 (read-only: the ratio in use)
PRESCALE_EN = 1 << 8
#: CPC_ROCC_CTL (Figure 6.4): bit n core n, 16 the CM, 17 the DBU, RESET_CAUSE
#: 30:29 (0 cold, 1 external warm, 2 watchdog), 31 the CPC
ROCC_CM, ROCC_DBU, ROCC_CPC = 1 << 16, 1 << 17, 1 << 31
CAUSE_COLD, CAUSE_WARM, CAUSE_WATCHDOG = 0, 1, 2


class _Registers:
    """A block of 32-bit registers at word offsets, read and written a byte
    at a time in the accessing core's byte order (as every MIPS device).
    A CM3's registers are 64 bits wide (``wide``): a doubleword access
    reaches a whole register, a word access its low (offset + 0) or high
    (offset + 4) half in either byte order — what Linux's 32-bit and 64-bit
    accessors both rely on (``mips-cm.h`` mips_cm_is64, ``mips-gic.h``),
    the CM2's 32-bit registers and their _UPPER halves."""

    wide = False

    @staticmethod
    def _bytes(value: int, big: bool) -> list[int]:
        b = [(value >> s) & 0xFF for s in (24, 16, 8, 0)]
        return b if big else b[::-1]

    def read(self, m, off: int, size: int) -> int:
        if size == 8 and self.wide and not off & 7:
            return self.get(m, off) | (self.get(m, off + 4) << 32)
        out = 0
        big = m.big_endian
        for i in range(size):
            o = off + i
            byte = self._bytes(self.get(m, o & ~3), big)[o & 3]
            out = (out << 8) | byte if big else out | (byte << (8 * i))
        return out

    def write(self, m, off: int, value: int, size: int) -> None:
        if size == 8 and self.wide and not off & 7:
            self.put(m, off + 4, (value >> 32) & M32)    # (the high half first: a
            self.put(m, off, value & M32)                # low-half write acts)
            return
        big = m.big_endian
        words: dict[int, list[int]] = {}
        for i in range(size):
            o = off + i
            shift = 8 * (size - 1 - i) if big else 8 * i
            word = o & ~3
            if word not in words:
                words[word] = self._bytes(self.get(m, word), big)
            words[word][o & 3] = (value >> shift) & 0xFF
        for word, b in words.items():
            if not big:
                b = b[::-1]
            self.put(m, word, (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3])

    def get(self, m, off: int) -> int:         # pragma: no cover - overridden
        raise NotImplementedError

    def put(self, m, off: int, value: int) -> None:   # pragma: no cover
        raise NotImplementedError


class GCR(_Registers):
    """The Coherence Manager's Global Control Registers (a bus device): a
    CM2's, or with ``cps.cm3`` a CM3's (64-bit registers, per-VP
    GCR_CL_REDIRECT / RESET_BASE, the I6400's)."""

    def __init__(self, cps: "CPS") -> None:
        self.cps = cps
        n = cps.n
        self.cm3 = self.wide = cps.cm3
        self.base = GCR_BASE_PA
        #: GCR_BASE 7:0: the default region's CCA override (7:5), its enable
        #: (4) and CM_DEFAULT_TARGET (1:0; 0 = memory); a CM3: the target only
        self.base_attr = 0
        # MD01025 Tables 11.23 / 11.25: SPEC_READ_EN = 1, every requester may write
        self.regs = {GCR_CONTROL: GCR_CONTROL_RESET, GCR_ACCESS: 0xFF, GCR_GIC_BASE: 0,
                     GCR_CPC_BASE: 0, GCR_CUSTOM_BASE: 0}
        for r in GCR_REGS:
            self.regs[r] = 0
        if self.cm3:                               # MD01196 5.4: all cores and IOCUs
            self.regs.update({GCR_ACCESS_CM3: 0x3003F, GCR_L2_PFT_CONTROL: 0,
                              GCR_L2_PFT_CONTROL_B: 0, GCR_SEM: 0, GCR_BEV_BASE: 0,
                              GCR_BEV_BASE + 4: 0, GCR_MMIO_REQ_LIMIT: 0})
            for r in GCR_MMIO:
                self.regs[r] = self.regs[r + 4] = 0
        #: the _UPPER registers (Undefined at reset but GCR_BASE_UPPER's
        #: MIPS default 0: eMIPS 0); a CM3's high halves (PA 47:32)
        self.upper = dict.fromkeys(UPPER, 0)
        self.upper_mask = ({o: 0xFFFF for o in UPPER} if self.cm3 else UPPER)
        #: per core: COHERENCE, OTHER (CM2), RESET_BASE (CM2), RESET_EXT_BASE
        self.core = [{CX_COHERENCE: 0, CX_OTHER: 0, CX_RESET_BASE: RESET_VECTOR,
                      CX_RESET_EXT_BASE: EXT_BASE_RESET} for i in range(n)]
        #: a CM3: per VP (the CPS's VP index) GCR_CL_REDIRECT and RESET_BASE
        #: (MD01196 5.5.1: "Both of these registers are instantiated per-VP")
        self.vp = [{CX_OTHER: 0, CX_RESET_BASE: RESET_VECTOR} for _ in range(cps.nvp)] \
            if self.cm3 else []

    # -- the bus ------------------------------------------------------------------------

    def covers(self, pa: int) -> bool:
        return self.full_base <= pa < self.full_base + GCR_SIZE

    @property
    def full_base(self) -> int:
        """GCR_BASE_UPPER:GCR_BASE — above eMIPS's 32-bit physical
        addresses when the upper bits are not 0."""
        return (self.upper[GCR_BASE_UPPER] << 32) | self.base

    def place(self, low: int, high: int) -> int | None:
        """A block's 40-bit (CM3: 48-bit) physical address from a base
        register (bit 0 = enable) and its _UPPER half, None while it is
        disabled."""
        return ((self.upper[high] << 32) | (self.regs[low] & ~0x7FFF)) \
            if self.regs[low] & 1 else None

    def may_write(self, m) -> bool:
        """GCR_ACCESS (a CM3's at 0x120: MD01196 5.4) lets this core write
        the GCRs and the CPC."""
        access = self.regs[GCR_ACCESS_CM3 if self.cm3 else GCR_ACCESS]
        return bool(access >> m.core_num & 1)

    def load(self, m, pa: int, size: int) -> int:
        return self.read(m, pa - self.base, size)

    def store(self, m, pa: int, value: int, size: int) -> None:
        if not self.may_write(m):
            return                                 # this core may not write the GCRs
        self.write(m, pa - self.base, value, size)

    def tick(self, m, cycles: int) -> None:
        pass

    # -- registers ----------------------------------------------------------------------

    def redirect(self, m) -> int:
        """The accessing VP's GCR_CL_REDIRECT (CM3) — where its core-other
        blocks (the GCR's, the CPC's) look."""
        return self.vp[m.cpu_num][CX_OTHER] if m.cpu_num < len(self.vp) else 0

    def _block(self, m, off: int) -> tuple[int | None, int | None, int]:
        """``(core, VP index, offset in its block)`` for a core-local /
        core-other offset — the VP only for a CM3's per-VP registers;
        ``(None, None, off)`` for a global one (also a CM3 redirect to the
        global block, BLOCK = 1); core -1: nothing there (reads 0)."""
        cps = self.cps
        if CORE_LOCAL <= off < CORE_LOCAL + 0x2000:
            return m.core_num, m.cpu_num if self.cm3 else None, off - CORE_LOCAL
        if CORE_OTHER <= off < CORE_OTHER + 0x2000:
            o = off - CORE_OTHER
            if not self.cm3:
                core = (self.core[m.core_num][CX_OTHER] >> 16) & 0xFFFF
                return (core if core < cps.n else -1), None, o
            r = self.redirect(m)
            block = (r >> 24) & 3
            if block == 1:
                return None, None, o
            core, k = (r >> 8) & 0x3F, r & 7
            if block or core >= cps.n or k >= cps.vpes:
                return -1, None, o
            return core, core * cps.vpes + k, o
        return None, None, off

    def domain(self, core: int) -> int:
        """GCR_Cx_COHERENCE as the CM2's domain bits (bit j = core j's
        interventions): a CM3's COHEN = 1 puts the core in one domain with
        every core whose COHEN is set (its directory, MD01196 5.1)."""
        v = self.core[core][CX_COHERENCE]
        if not self.cm3:
            return v
        if not v & COHEN:
            return 0
        return (sum(1 << j for j in range(self.cps.n) if self.core[j][CX_COHERENCE] & COHEN)
                | (1 << 4 if self.cps.iocu else 0))        # (the IOCU's bit, as a CM2's)

    def get(self, m, off: int) -> int:
        core, v, o = self._block(m, off)
        if core is not None:
            if core < 0:
                return 0
            if o == CX_CONFIG:                     # PVPE 9:0 = the core's VPEs - 1
                return self.cps.vpes - 1           # (MD00904 GCR_CL_CONFIG; no IOCU)
            if o == CX_ID:
                return core
            if v is not None and o in (CX_OTHER, CX_RESET_BASE):
                return self.vp[v][o]
            if self.cm3 and o == CX_OTHER:         # (a CM3 has no per-core OTHER)
                return 0
            return self.core[core].get(o, 0)
        if o == GCR_CONFIG:                        # 4 regions, NUMIOCU 11:8, PCORES
            return ((4 << 16) | (int(self.cps.iocu) << 8) | ((self.cps.n - 1) & 0xFF)
                    | (CONFIG_ITU if self.cm3 and self.cps.itu else 0))
        if o == GCR_BASE:
            return self.base | self.base_attr
        cm = self.cps.cm
        if self.cm3 and o - 4 in _CM_REGS and o - 4 != GCR_L2_ONLY_SYNC_BASE:
            return getattr(cm, _CM_REGS[o - 4]) >> 32      # (a 64-bit error register)
        if o in self.upper:
            return self.upper[o]
        if o in _CM_REGS:
            return getattr(cm, _CM_REGS[o]) & M32
        if self.cm3:
            if o == GCR_REV:
                return CM3_REV
            if o == GCR_ACCESS or o in GCR_REGS:   # (moved / no CM2 regions: MMIO ones)
                return 0
            if o == GCR_L2_CONFIG:
                return cm.l2_config()
            if o == GCR_SYS_CONFIG2:               # MAXVPW: 4 VP numbers a core
                from .vp import MAX_VP
                return MAX_VP
        if o == GCR_REV:
            return CM2_REV
        if o == GCR_GIC_STATUS:
            return 1                               # EX: a GIC is connected
        if o == GCR_CPC_STATUS:
            return 1                               # EX: a CPC is connected
        return self.regs.get(o, 0)

    def put(self, m, off: int, value: int) -> None:
        core, v, o = self._block(m, off)
        if core is not None:
            if core < 0:
                return
            if v is not None and o in (CX_OTHER, CX_RESET_BASE):
                self.vp[v][o] = value & (REDIRECT_MASK if o == CX_OTHER
                                         else 0xFFFFF000 | RESET_BASE_MODE)
                return
            regs = self.core[core]
            if o == CX_COHERENCE:
                regs[o] = value & (COHEN if self.cm3 else 0xFF)
                self.cps.coherence_changed(core)
            elif o == CX_OTHER and not self.cm3:   # GCR_CL_ / GCR_CO_OTHER: R/W
                regs[o] = value & 0xFFFF0000
            elif o == CX_RESET_BASE and not self.cm3:
                regs[o] = value & 0xFFFFF000
            elif o == CX_RESET_EXT_BASE:
                regs[o] = (value & EXT_BASE_MASK) | 1          # PRESENT reads 1
            return
        if self.cm3 and o - 4 in _CM_REGS and o - 4 != GCR_L2_ONLY_SYNC_BASE:
            self._put_cm3_error(o - 4, value, high=True)    # (a 64-bit error register)
        elif o == GCR_BASE:
            self.base = value & 0xFFFF8000
            self.base_attr = value & (0x3 if self.cm3 else 0xF3)
            for c in self.cps.vps:              # (every VPE's CMGCRBase)
                c.cmgcr_moved(self.full_base)
        elif o in self.upper:
            self.upper[o] = value & self.upper_mask[o]
            if o == GCR_BASE_UPPER:
                for c in self.cps.vps:
                    c.cmgcr_moved(self.full_base)
            elif o == GCR_GIC_BASE_UPPER:
                self._place_gic()
            elif o == GCR_CPC_BASE_UPPER:
                self.cps.cpc.base = self.place(GCR_CPC_BASE, GCR_CPC_BASE_UPPER)
        elif o == GCR_CUSTOM_BASE:
            self.regs[o] = value & 0xFFFF0000       # GGU_EN: read-only 0 (GGU_EX = 0)
        elif self.cm3 and o in _CM_REGS and o != GCR_L2_ONLY_SYNC_BASE:
            self._put_cm3_error(o, value, high=False)
        elif o in _CM_REGS:                        # ERROR_CAUSE / MULT: R/W (Table
            mask = {GCR_L2_ONLY_SYNC_BASE: 0xFFFFF001,  # 11.29: a type is kept until
                    GCR_ERROR_MULT: 0x1F}.get(o, M32)   # written 0)
            setattr(self.cps.cm, _CM_REGS[o], value & mask)
            if o in (GCR_ERROR_CAUSE, GCR_ERROR_MASK):
                self.cps.cm.error_changed()
        elif o in GCR_REGS and not self.cm3:       # REGn_BASE / REGn_MASK
            self.regs[o] = value & (0xFFFF0000 if (o - 0x90) % 16 == 0 else 0xFFFF00F7)
        elif o == GCR_ACCESS and not self.cm3:
            self.regs[o] = value & 0xFF
        elif o == GCR_CONTROL:
            self.regs[o] = value & (GCR_CONTROL_MASK | (0x3000 if self.cm3 else 0))
        elif o == GCR_GIC_BASE:
            self.regs[o] = value & 0xFFFE0001
            self._place_gic()
        elif o == GCR_CPC_BASE:
            self.regs[o] = value & 0xFFFF8001
            self.cps.cpc.base = self.place(GCR_CPC_BASE, GCR_CPC_BASE_UPPER)
        elif self.cm3:
            self._put_cm3(o, value)

    def _put_cm3(self, o: int, value: int) -> None:
        """The CM3's other global registers (MD01196 5.4-5.12)."""
        regs = self.regs
        if o == GCR_ACCESS_CM3:                    # ACCESS_EN: cores 5:0, IOCUs 17:16
            regs[o] = value & 0x3003F
        elif o == GCR_L2_PFT_CONTROL:              # NPFT 7:0 = 0 (no prefetch unit):
            regs[o] = value & 0xFFFFF100           # PAGEMASK 31:12, PFTEN 8 stored
        elif o == GCR_L2_PFT_CONTROL_B:            # CEN 8, PORT_ID 7:0
            regs[o] = value & 0x1FF
        elif o == GCR_SEM:                         # 5.8: a lock write is inhibited
            if not (value & SEM_LOCK and regs[o] & SEM_LOCK):   # while it is held
                regs[o] = value & M32
        elif o in (GCR_BEV_BASE, GCR_BEV_BASE + 4):
            regs[o] = value & (0xFFFFF000 if o == GCR_BEV_BASE else 0xFFFF)
        elif o == GCR_MMIO_REQ_LIMIT:
            regs[o] = value & 0xFF
        elif any(r <= o < r + 8 for r in GCR_MMIO):    # bounds 47:16 + the low controls
            regs[o] = value & (0xFFFF if o & 4 else M32)

    def _put_cm3_error(self, o: int, value: int, high: bool) -> None:
        """A CM3 error register (64 bits, MD01196 5.10): GCR_ERROR_CAUSE /
        MULT clear when software writes the recorded ERR_TYPE (63:58) back;
        the address and mask are R/W."""
        cm = self.cps.cm
        name = _CM_REGS[o]
        old = getattr(cm, name)
        if o in (GCR_ERROR_CAUSE, GCR_ERROR_MULT):
            if high and (value >> 26) & 0x3F == (cm.error_cause >> 58) and cm.error_cause >> 58:
                cm.error_cause = 0
                cm.error_mult = 0
                cm.error_changed()
            return
        new = ((old & M32) | ((value & M32) << 32)) if high else \
            ((old & ~M32) | (value & M32))
        setattr(cm, name, new & ((1 << 64) - 1 if o == GCR_ERROR_MASK else (1 << 48) - 1))
        if o == GCR_ERROR_MASK:
            cm.error_changed()

    def _place_gic(self) -> None:
        gic = self.cps.gic
        gic.base = self.place(GCR_GIC_BASE, GCR_GIC_BASE_UPPER)
        if gic.base is not None:
            gic.base &= ~0x1FFFF                    # (a 128 KB block)
        gic.update()


class CPC(_Registers):
    """The Cluster Power Controller (a bus device, at GCR_CPC_BASE when the
    GCR enables it): a CM2's, or a CM3's (64-bit registers; another
    domain through the accessing VP's GCR_CL_REDIRECT, MD01196 6.2.4; the
    VP run / stop registers)."""

    def __init__(self, cps: "CPS") -> None:
        self.cps = cps
        self.cm3 = self.wide = cps.cm3
        self.base: int | None = None              # disabled until GCR_CPC_BASE.EN
        # MD01025 Table 10.8: ACCESS resets to 0xFF; SEQDEL / RAIL / RESETLEN
        # are 10-bit fields (their delays: not modelled)
        self.regs = {CPC_ACCESS: 0xFF, CPC_SEQDEL: 0, CPC_RAIL: 0, CPC_RESETLEN: 0}
        if self.cm3:                              # MD01196 6.2.14.1: MICROSTEP 0x002
            del self.regs[CPC_ACCESS]             # (6.2.5: GCR_ACCESS gates the CPC)
            self.regs.update({CPC_SEQDEL: 0x002, CPC_CC_CTL: CC_DELAY_RESET,
                              CPC_PWRUP_CTL: 0, CPC_RES_REL: 0, CPC_PRESCALE: 0})
        self.other = [0] * cps.n
        #: a CM3's per-domain clock / RAM registers: (domain, offset) -> value
        self.domain_regs: dict[tuple[int, int], int] = {}
        #: CPC_ROCC_CTL: at power-on the CPC, the CM and core 0 came out of
        #: a cold reset (Figure 6.4)
        self.rocc = ROCC_CPC | ROCC_CM | 1
        #: the Debug Unit's power domain (sequencer state, last command)
        self.dbu_state, self.dbu_cmd = SEQ_D0, 0

    def covers(self, pa: int) -> bool:
        return self.base is not None and self.base <= pa < self.base + CPC_SIZE

    def load(self, m, pa: int, size: int) -> int:
        return self.read(m, pa - self.base, size)

    def store(self, m, pa: int, value: int, size: int) -> None:
        if not (self.cps.gcr.may_write(m) if self.cm3 else self.regs[CPC_ACCESS] >> m.core_num & 1):
            return
        self.write(m, pa - self.base, value, size)

    def tick(self, m, cycles: int) -> None:
        pass

    def _core(self, m, off: int) -> tuple[int | None, int]:
        """``(domain, offset in its block)``: a core's number, a CM3's ring
        ID of another domain (the CM, the DBU, an IOCU, memory), -1 for
        nothing there; ``(None, off)`` for a global register."""
        if CORE_LOCAL <= off < CORE_LOCAL + 0x2000:
            return m.core_num, off - CORE_LOCAL
        if CORE_OTHER <= off < CORE_OTHER + 0x2000:
            o = off - CORE_OTHER
            if not self.cm3:
                return (self.other[m.core_num] >> 16) & 0xFF, o
            target = (self.cps.gcr.redirect(m) >> 8) & 0x3F
            if target < self.cps.n or target in (RING_CM, RING_DBU, RING_MEMORY) \
                    or (self.cps.iocu and target == RING_IOCU0):
                return target, o
            return -1, o
        return None, off

    def get(self, m, off: int) -> int:
        core, o = self._core(m, off)
        if core is not None:
            if self.cm3:
                return self._get_domain(core, o)
            if core >= self.cps.n:
                return 0
            if o == CPC_CX_CMD:
                return self.cps.last_cmd[core]
            if o == CPC_CX_STAT_CONF:
                return self.stat_conf(core)
            if o == CPC_CX_OTHER:
                return self.other[core]
            return 0
        if o == CPC_REVISION:
            return CPC_REV
        if self.cm3 and o == CPC_ROCC_CTL:
            return self.rocc
        if self.cm3 and o == CPC_PRESCALE:         # the ratio in use reads back in 23:16
            v = self.regs[o]
            return v | ((v & 0xFF) << 16)
        return self.regs.get(o, 0)

    def _get_domain(self, d: int, o: int) -> int:
        cps = self.cps
        if d < 0:
            return 0
        if o in (CPC_CX_CC_CTL, CPC_RAM_SLEEP):
            return self.domain_regs.get((d, o), 0)
        if d < cps.n:
            if o == CPC_CX_CMD:
                return cps.last_cmd[d]
            if o == CPC_CX_STAT_CONF:
                return self.stat_conf(d)
            if o == CPC_VP_RUNNING:
                return cps.vp_mask[d] if cps.state[d] in (SEQ_U5, SEQ_D2) else 0
            return 0                               # (VP_STOP / VP_RUN: write-only)
        if d == RING_CM:                           # 6.2.9: up while a core is up
            up = any(st != SEQ_D0 for st in cps.state)
            return ((SEQ_U5 if up else SEQ_D0) << 19) if o == CPC_CX_STAT_CONF else 0
        if d == RING_DBU:
            if o == CPC_CX_CMD:
                return self.dbu_cmd
            if o == CPC_CX_STAT_CONF:
                return (self.dbu_state << 19) | (1 << 17) | (1 << 16) | self.dbu_cmd
        return 0

    def put(self, m, off: int, value: int) -> None:
        core, o = self._core(m, off)
        if core is not None:
            if self.cm3:
                self._put_domain(core, o, value)
                return
            if core >= self.cps.n:
                return
            if o == CPC_CX_CMD:
                self.cps.command(core, value & 0xF)
            elif o == CPC_CX_OTHER:                # CPC_CL_ / CPC_CO_OTHER: R/W
                self.other[core] = value & 0x00FF0000
            return
        if self.cm3:
            self._put_global(o, value)
        elif o in self.regs:
            self.regs[o] = value & (0xFF if o == CPC_ACCESS else 0x3FF)

    def _put_domain(self, d: int, o: int, value: int) -> None:
        cps = self.cps
        if d < 0:
            return
        if o == CPC_CX_CC_CTL:                     # CLK_RATIO 3:0, CHANGE_EN 8 (6.2.8)
            self.domain_regs[(d, o)] = value & 0x10F
        elif o == CPC_RAM_SLEEP:                   # 6.2.12: the disables, the delays
            self.domain_regs[(d, o)] = value & 0x80FF80FF
        elif d < cps.n:
            if o == CPC_CX_CMD:
                cps.command(d, value & 0xF)
            elif o == CPC_VP_STOP:
                cps.vp_stop(d, value & 0xF)
            elif o == CPC_VP_RUN:
                cps.vp_run(d, value & 0xF)
        elif d == RING_DBU and o == CPC_CX_CMD:    # its own power domain (Figure 6.1)
            cmd = value & 0xF
            if cmd in (CMD_PWRUP, CMD_RESET):
                self.dbu_state, self.dbu_cmd = SEQ_U5, CMD_PWRUP
                self.rocc |= ROCC_DBU               # (it comes out of a reset)
            elif cmd == CMD_PWRDOWN:
                self.dbu_state, self.dbu_cmd = SEQ_D0, cmd
            elif cmd == CMD_CLOCKOFF:
                self.dbu_state = SEQ_D2 if self.dbu_state == SEQ_U5 else SEQ_U2
                self.dbu_cmd = cmd

    def _put_global(self, o: int, value: int) -> None:
        regs = self.regs
        if o in (CPC_SEQDEL, CPC_RAIL, CPC_RESETLEN, CPC_RES_REL):
            regs[o] = value & 0x3FF
        elif o == CPC_CC_CTL:                      # the change sequence ends at once
            regs[o] = value & CC_DELAY_MASK        # (SET_CLK_RATIO, ACTIVE read 0)
        elif o == CPC_ROCC_CTL:                    # the "occurred" bits: write 1 to clear
            self.rocc &= ~(value & (ROCC_CPC | ROCC_DBU | ROCC_CM | 0x3F))
        elif o == CPC_PRESCALE:                    # 6.2.7: the ratio with CHANGE_EN
            if value & PRESCALE_EN:
                regs[o] = value & (PRESCALE_EN | 0xFF)
            else:
                regs[o] &= 0xFF
        # CPC_PWRUP_CTL (6.2.9): the CM is operational while a core runs:
        # "setting this bit has no meaning and the register write is ignored"

    def core_reset(self, core: int, cause: int | None = None) -> None:
        """A core came out of reset: CPC_ROCC_CTL records it (and why)."""
        self.rocc |= 1 << core
        if cause is not None:
            self.rocc = (self.rocc & ~(3 << 29)) | (cause << 29)

    def stat_conf(self, core: int) -> int:
        """CPC_Cx_STAT_CONF (MD01025 Table 10.17): SEQSTATE 22:19, CLKGAT_IMPL
        / PWRDN_IMPL (17 / 16), bit 10 (reserved, reads 1), PWUP_POLICY 9:8
        (hardwired: core 0 powers up at cold start, 2; the others not, 0),
        CMD 3:0 = the most recently placed command (a waiting one too)."""
        cps = self.cps
        return ((cps.seq_state(core) << 19) | (1 << 17) | (1 << 16) | (1 << 10)
                | ((2 if core == 0 else 0) << 8) | cps.placed[core])


class CPS:
    """The cores and the platform.  ``factory(cps, core)`` builds a core in
    its hardware reset state: a Machine on the shared physical memory, or
    — an MT core (the 1004K / interAptiv, ``vpes`` = 2) — the list of its
    VPEs' Machines (VPE n of core c: ``cpu = c * vpes + n``, the GIC's
    processor number, MD00904 Table 9.1); the CPS puts the GCR and the CPC
    first on their buses.  ``on_start(core, machine)`` is called when the
    CPC / GCR (re)start a core (with its first VPE).  At power-on core 0
    runs and the others are powered down.

    ``cm3``: the I6400's CM3 (MD01196): 64-bit registers, Release 6 cores
    of ``vpes`` virtual processors (emips.vp: the factory builds a core's
    VPs and their VPCore; ``vp_factory(cps, core, k)`` one VP the CPC
    starts again), VP numbers core x 4 + VP (the GIC's, EBase.CPUNum),
    the VPs that run a core's VP_RUNNING mask (after a core reset: VP 0)."""

    def __init__(self, n: int, factory: Callable[["CPS", int], Any],
                 on_start: Callable[[int, Any], None] | None = None, *,
                 l2=None, l3=None, ecc: bool = False, hardware_init: bool = False,
                 iocu: bool = False, vpes: int = 1, cm3: bool = False,
                 vp_factory: Callable[["CPS", int, int], Any] | None = None,
                 itu: bool = False) -> None:
        from .coherence import CoherenceManager
        from .vp import MAX_VP
        if not 1 <= n <= MAX_CORES:
            raise ValueError(f"a CPS has 1 to {MAX_CORES} cores, not {n}")
        if cm3 and not 1 <= vpes <= MAX_VP:
            raise ValueError(f"a CM3 core has 1 to {MAX_VP} VPs, not {vpes}")
        if not cm3 and n * vpes > MAX_VPS:
            raise ValueError(f"{n} cores of {vpes} VPEs: the GIC's boot set-up routes "
                             f"the IPIs of {MAX_VPS} VPs")
        self.n = n
        #: the CM3 (the I6400) instead of a CM2
        self.cm3 = cm3
        #: VPEs (the MT Module) or Release 6 VPs (a CM3) per core, and the
        #: GIC's processors (VPs)
        self.vpes = vpes
        self.nvp = n * vpes
        #: an I/O Coherence Unit (GCR_CONFIG.NUMIOCU = 1): the I/O masters
        #: behind it (the Malta's PCI DMA) make coherent requests
        self.iocu = iocu
        #: the cores have an ITU (a CM3's GCR_CONFIG bit 31)
        self.itu = itu
        self.factory = factory
        self.vp_factory = vp_factory
        self.on_start = on_start
        #: a CM3: the VPs that run, per core (CPC VP_RUNNING; reset: VP 0)
        self.vp_mask = [1] * n
        self.gcr = GCR(self)
        self.cpc = CPC(self)
        #: the system's clock: cycles since power-on (the Session counts them;
        #: a core started later keeps Machine.clock_offset to it)
        self.clock = 0
        #: the Coherence Manager: MESI between the data caches, the shared
        #: L2 / L3 (Geometry ``l2`` / ``l3``), its error registers
        self.cm = CoherenceManager(self, l2=l2, l3=l3, ecc=ecc, hardware_init=hardware_init)
        from .gic import GIC
        #: the Global Interrupt Controller (at GCR_GIC_BASE once enabled)
        self.gic = GIC(self)
        #: the platform devices every core's bus has
        self.devices: list = [self.gcr, self.cpc, self.gic]
        #: the cores (a powered-down core keeps its last Machine): the
        #: Machine of each core's VPE 0, and every VPE (the GIC's VPs)
        self.cores: list = []
        self.vps: list = [None] * self.nvp
        #: the sequencer state per core: SEQ_D0, SEQ_U2 (powered, clock off),
        #: SEQ_D2 (clock off from execution, state kept) or SEQ_U5 (executing; it
        #: reads U6 while the core is in the coherence domain)
        self.state = [SEQ_U5] + [SEQ_D0] * (n - 1)
        #: the last CPC command executed per core (CPC_Cx_CMD reads it)
        self.last_cmd = [0] * n
        #: the most recently placed command per core (STAT_CONF.CMD)
        self.placed = [0] * n
        #: a ClockOff / PwrDown waiting for the core to leave the coherence
        #: domain ("the command remains inactive until the protocol barriers
        #: subside")
        self.pending: list[int | None] = [None] * n
        for i in range(n):
            self.cores.append(self._build(i)[0])

    def _build(self, core: int) -> list:
        """A core in its reset state: ``factory(cps, core)`` builds the
        Machine(s) (its caches attached to :attr:`cm` first); returns its
        VPEs."""
        built = self.factory(self, core)
        ms = list(built) if isinstance(built, (list, tuple)) else [built]
        for k, m in enumerate(ms):
            self._install(core, k, m, attach=False)
        for m in ms:
            self.gic.attach(m)
        self.vp_mask[core] = 1
        return ms

    def _install(self, core: int, k: int, m, attach: bool = True) -> None:
        m.core_num = core
        m.join_system(self)
        m.devices[:0] = self.devices
        self.vps[core * self.vpes + k] = m
        if attach:
            self.gic.attach(m)

    def vps_of(self, core: int) -> list:
        """The Machines of ``core``'s VPEs."""
        return self.vps[core * self.vpes:(core + 1) * self.vpes]

    def vp_core(self, v: int) -> int:
        return v // self.vpes

    def vp_id(self, v: int) -> int:
        """VP ``v``'s number in the hardware: the GIC's, EBase.CPUNum (a
        CM3: core x 4 + VP, emips.vp; else ``v``)."""
        if not self.cm3:
            return v
        from .vp import vp_number
        return vp_number(*divmod(v, self.vpes))

    def vp_index(self, vid: int) -> int | None:
        """The VP whose hardware number is ``vid``, None if there is none."""
        if not self.cm3:
            return vid if 0 <= vid < self.nvp else None
        from .vp import MAX_VP
        core, k = divmod(vid, MAX_VP)
        return core * self.vpes + k if core < self.n and k < self.vpes else None

    # -- power ---------------------------------------------------------------------------

    def running(self) -> list[int]:
        """The cores that execute (non-coherent / coherent execution)."""
        return [i for i in range(self.n) if self.state[i] == SEQ_U5]

    def running_vps(self) -> list[int]:
        """The VPs that execute: every VP(E) of a running core (MT VPEs:
        the MT processor still chooses, emips.mt); a CM3: those in the
        core's VP_RUNNING mask that no other VP's DVP holds."""
        vpes = self.vpes
        if not self.cm3:
            return [c * vpes + k for c in self.running() for k in range(vpes)]
        out = []
        for c in self.running():
            mask = self.vp_mask[c]
            vc = self.vps[c * vpes].vpcore
            for k in range(vpes):
                if mask >> k & 1 and (vc is None or not vc.disabled(k)):
                    out.append(c * vpes + k)
        return out

    def vp_runs(self, v: int) -> bool:
        """Does VP ``v`` execute now (a CM3: its core runs, it is in the
        core's VP_RUNNING mask, no other VP's DVP holds it)?"""
        c, k = divmod(v, self.vpes)
        if self.state[c] != SEQ_U5:
            return False
        if not self.cm3:
            return True
        vc = self.vps[v].vpcore if self.vps[v] is not None else None
        return bool(self.vp_mask[c] >> k & 1) and (vc is None or not vc.disabled(k))

    def held_vps(self) -> list[int]:
        """A CM3's VPs that a DVP holds: the core's clock still runs for
        them (Vol. II 6.06 EVP: "the system timer continues to count")."""
        if not self.cm3:
            return []
        vpes = self.vpes
        return [c * vpes + k for c in self.running() for k in range(vpes)
                if self.vp_mask[c] >> k & 1 and self.vps[c * vpes].vpcore is not None
                and self.vps[c * vpes].vpcore.disabled(k)]

    def coherent(self, core: int) -> bool:
        """In the coherence domain (GCR_Cx_COHERENCE.COHDOMAINEN != 0; a
        CM3's COHEN)."""
        return bool(self.gcr.core[core][CX_COHERENCE])

    def seq_state(self, core: int) -> int:
        """CPC_Cx_STAT_CONF.SEQSTATE: U5 non-coherent execution, U6 coherent
        execution once the core has joined the coherence domain."""
        st = self.state[core]
        return SEQ_U6 if st == SEQ_U5 and self.coherent(core) else st

    def reset_base(self, core: int, k: int = 0) -> int:
        """Where VP ``k`` of ``core`` starts: GCR_Cx_RESET_BASE (a CM3's per
        VP; RESET_BASE_MODE = 1: anywhere in the 4 GB, MD01196 5.5.2) with
        UseExceptionBase (Table 11.65: bits 31:30 forced to 10, the lower
        512 MB), else 0xBFC00000."""
        regs = self.gcr.core[core]
        rb = self.gcr.vp[core * self.vpes + k][CX_RESET_BASE] if self.cm3 \
            else regs[CX_RESET_BASE]
        if self.cm3 and rb & RESET_BASE_MODE:
            return rb & 0xFFFFF000
        if regs[CX_RESET_EXT_BASE] & EXT_UEB:     # SI_UseExceptionBase (Table 11.65)
            return (rb & 0x3FFFF000) | 0x80000000
        return RESET_VECTOR

    def _place(self, m, base: int, pc: int | None) -> None:
        m.clock_offset = self.clock - m.cycles     # its cycles count from now
        m.bev_base = (base + 0x200) & M32          # its BEV vectors follow it
        # (a 32-bit virtual address, MD01138 Table 6.62: MIPS64 sign-extends)
        m.pc = m.to_pc((pc if pc is not None else base) & M32)

    def start(self, core: int, pc: int | None = None, cause: int | None = None) -> Any:
        """Reset ``core`` and let it run: a fresh core (the hardware reset
        state) at its reset vector GCR_Cx_RESET_BASE, non-coherent (a CM3:
        each VP at its own, VP 0 running — "the value in this register is
        reset whenever the associated core is reset", MD01196 6.2.11)."""
        ms = self._build(core)
        for k, m in enumerate(ms):
            self._place(m, self.reset_base(core, k), pc)
        m = ms[0]
        self.cores[core] = m
        self.state[core] = SEQ_U5
        self.gic.reattach(core)
        self.pending[core] = None
        self.cpc.core_reset(core, cause)
        if self.on_start is not None:           # each VPE (VP core * vpes + n)
            for k, vm in enumerate(ms):
                self.on_start(core * self.vpes + k, vm)
        return m

    def vp_run(self, core: int, mask: int) -> None:
        """CPC VP_RUN (MD01196 6.2.11; the P8700's CM3.5 Table 90): a VP not
        running starts "from the reset vector" — a fresh VP (its own reset
        state; the core's caches, FTLB and ITU go on) at its RESET_BASE.  A
        core that does not execute (powered down, clock off from power-up)
        has nothing to start: its reset decides."""
        if self.state[core] not in (SEQ_U5, SEQ_D2):
            return
        for k in range(self.vpes):
            if mask >> k & 1 and not self.vp_mask[core] >> k & 1:
                self.start_vp(core, k)
                self.vp_mask[core] |= 1 << k

    def vp_stop(self, core: int, mask: int) -> None:
        """CPC VP_STOP: running VPs stop (the suspend state; a VP_RUN starts
        them again from the reset vector)."""
        if self.state[core] in (SEQ_U5, SEQ_D2):
            self.vp_mask[core] &= ~mask & ((1 << self.vpes) - 1)

    def start_vp(self, core: int, k: int) -> Any:
        """VP ``k`` of a running core starts again: a fresh Machine from
        ``vp_factory`` at its reset base."""
        if self.vp_factory is None:
            raise ValueError("this CPS cannot start a VP alone")
        m = self.vp_factory(self, core, k)
        self._install(core, k, m)
        self._place(m, self.reset_base(core, k), None)
        if k == 0:
            self.cores[core] = m
        v = core * self.vpes + k
        self.gic.reattach_vp(v)
        if self.on_start is not None:
            self.on_start(v, m)
        return m

    def command(self, core: int, cmd: int) -> None:
        """CPC_Cx_CMD (MD01025 10.3.4.1, 10.4).  Reserved codes do nothing.
        ClockOff / PwrDown / Reset of a core in coherent execution wait
        until it leaves the domain.  ClockOff: a running core stops its
        clock (D2, its state kept), a powered-down one powers up with its
        clock off (U2).  PwrDown: D0.  PwrUp: D2 gets its clock back (U5 /
        U6), D0 / U2 are reset and run, a running core: no effect.  Reset:
        reset and run; then CMD reads PwrUp."""
        if cmd not in (CMD_CLOCKOFF, CMD_PWRDOWN, CMD_PWRUP, CMD_RESET):
            return
        self.placed[core] = cmd
        st = self.state[core]
        if cmd != CMD_PWRUP and st == SEQ_U5 and self.coherent(core):
            self.pending[core] = cmd
            return
        self.pending[core] = None
        self.last_cmd[core] = cmd
        if cmd == CMD_CLOCKOFF:
            if st == SEQ_U5:
                self.state[core] = SEQ_D2
            elif st == SEQ_D0:
                self.state[core] = SEQ_U2
        elif cmd == CMD_PWRDOWN:
            self.state[core] = SEQ_D0
        elif cmd == CMD_PWRUP:
            if st == SEQ_D2:
                self.state[core] = SEQ_U5
            elif st != SEQ_U5:
                self.start(core)
        else:                                          # Reset
            self.start(core)
            self.last_cmd[core] = CMD_PWRUP            # "reads as PwrUp cmd"

    def coherence_changed(self, core: int) -> None:
        """GCR_Cx_COHERENCE was written: a waiting command may go now."""
        if self.pending[core] is not None and not self.coherent(core):
            self.command(core, self.pending[core])

    def boot(self) -> None:
        """What the boot firmware does on a CPS (Boot-MIPS init_cm /
        init_cpc / release_mp): every core may write the GCRs and the CPC,
        the CPC is enabled at :data:`CPC_BASE_PA`, every core is up and in
        the coherence domain (a CM3: every VP runs)."""
        if self.cm3:                                   # MD01196 5.4: all requesters
            self.gcr.regs[GCR_ACCESS_CM3] = 0x3003F
        else:
            self.gcr.regs[GCR_ACCESS] = 0xFF           # every requester (the reset value)
        self.gcr.regs[GCR_CPC_BASE] = CPC_BASE_PA | 1
        self.cpc.base = CPC_BASE_PA
        from .gic import GIC_BASE_PA
        self.gcr.regs[GCR_GIC_BASE] = GIC_BASE_PA | 1
        self.gic.boot()
        for i in range(self.n):
            self.gcr.core[i][CX_COHERENCE] = COHEN if self.cm3 else 0xFF
            self.state[i] = SEQ_U5
            self.vp_mask[i] = (1 << self.vpes) - 1

    # -- the bus hooks of the cores (emips.sim) ------------------------------------------

    def registers_at(self, pa: int) -> bool:
        """Is ``pa`` a GCR / CPC register?"""
        return self.gcr.covers(pa) or self.cpc.covers(pa) or self.gic.covers(pa)

    def register_error(self, m, pa: int, cca: int, write: bool) -> bool:
        """A GCR / GIC / CPC access with an attribute other than uncached.
        A single beat — uncached accelerated, a write-through store, any
        access of a core without a data cache — reaches the register like
        an uncached one: returns False.  A line fill (a load, a write-back
        store miss: write-allocate) is a CM error (MD01025 Table 11.11,
        GC_RD / COH_RD) and does nothing (a load returns 0); an ERROR
        response makes the core take a data bus error (imprecise): returns
        True."""
        if cca == 7 or (write and cca == 0) or not m._uses_dcache(cca):
            return False
        target = TR_GCR if self.gcr.covers(pa) else TR_GIC if self.gic.covers(pa) else TR_CPC
        if self.cm.register_access_error(m, pa, cca, write, target):
            m.bus_error_pending = True
        return True

    def store_check(self, m, pa: int, size: int, cca: int, value: int = 0) -> bool:
        """Before a store of core ``m``: True when it must not happen (a
        GCR / CPC store with a cached attribute: a CM error; an uncached
        store to the L2-only SYNC region: a barrier, not a write).  A store
        the other cores can see (any but one kept in this core's data cache
        without coherence) clears their LLbit for its 32-byte block — and
        always the LLbit of the VP(E)s that share this core's caches."""
        if cca != 2 and self.registers_at(pa):
            if self.register_error(m, pa, cca, True):
                return True
            dev = next(d for d in self.devices if d.covers(pa))
            dev.store(m, pa, value, size)       # a single beat: the register
            return True
        if cca == 2 and self.cm.l2_only_sync(pa):
            return True
        c = m.caches
        hidden = (c is not None and c.d is not None and m._uses_dcache(cca)
                  and not c.coherent(cca))
        block = pa & ~31
        for other in self.vps:
            if other is not m and other.lr_valid and other.lr_pa is not None \
                    and other.lr_pa & ~31 == block \
                    and (not hidden or other.core_num == m.core_num):
                other.lr_valid = False
        return False

    def describe(self) -> list[str]:
        out = []
        for i in range(self.n):
            m = self.cores[i]
            st = self.seq_state(i)
            line = (f"core {i}: {SEQ_NAMES.get(st, st)}, "
                    f"pc 0x{m.pc:0{16 if m.wide else 8}x}"
                    + (" (halted)" if m.halted else ""))
            if self.cm3 and self.vpes > 1:
                run = [str(self.vp_id(i * self.vpes + k)) for k in range(self.vpes)
                       if self.vp_mask[i] >> k & 1]
                line += f", VPs running: {', '.join(run) or 'none'}"
            out.append(line)
        return out
