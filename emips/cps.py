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


class _Registers:
    """A block of 32-bit registers at word offsets, read and written a byte
    at a time in the accessing core's byte order (as every MIPS device)."""

    @staticmethod
    def _bytes(value: int, big: bool) -> list[int]:
        b = [(value >> s) & 0xFF for s in (24, 16, 8, 0)]
        return b if big else b[::-1]

    def read(self, m, off: int, size: int) -> int:
        out = 0
        big = m.big_endian
        for i in range(size):
            o = off + i
            byte = self._bytes(self.get(m, o & ~3), big)[o & 3]
            out = (out << 8) | byte if big else out | (byte << (8 * i))
        return out

    def write(self, m, off: int, value: int, size: int) -> None:
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
    """The Coherence Manager's Global Control Registers (a bus device)."""

    def __init__(self, cps: "CPS") -> None:
        self.cps = cps
        n = cps.n
        self.base = GCR_BASE_PA
        #: GCR_BASE 7:0: the default region's CCA override (7:5), its enable
        #: (4) and CM_DEFAULT_TARGET (1:0; 0 = memory)
        self.base_attr = 0
        # MD01025 Tables 11.23 / 11.25: SPEC_READ_EN = 1, every requester may write
        self.regs = {GCR_CONTROL: GCR_CONTROL_RESET, GCR_ACCESS: 0xFF, GCR_GIC_BASE: 0,
                     GCR_CPC_BASE: 0}
        for r in GCR_REGS:
            self.regs[r] = 0
        #: per core: COHERENCE, OTHER, RESET_BASE, RESET_EXT_BASE
        self.core = [{CX_COHERENCE: 0, CX_OTHER: 0, CX_RESET_BASE: RESET_VECTOR,
                      CX_RESET_EXT_BASE: EXT_BASE_RESET} for i in range(n)]

    # -- the bus ------------------------------------------------------------------------

    def covers(self, pa: int) -> bool:
        return self.base <= pa < self.base + GCR_SIZE

    def load(self, m, pa: int, size: int) -> int:
        return self.read(m, pa - self.base, size)

    def store(self, m, pa: int, value: int, size: int) -> None:
        if not self.regs[GCR_ACCESS] >> m.core_num & 1:
            return                                 # this core may not write the GCRs
        self.write(m, pa - self.base, value, size)

    def tick(self, m, cycles: int) -> None:
        pass

    # -- registers ----------------------------------------------------------------------

    def _block(self, m, off: int) -> tuple[int | None, int]:
        """``(core, offset in its block)`` for a core-local / core-other
        offset; ``(None, off)`` for a global one."""
        if CORE_LOCAL <= off < CORE_LOCAL + 0x2000:
            return m.core_num, off - CORE_LOCAL
        if CORE_OTHER <= off < CORE_OTHER + 0x2000:
            return (self.core[m.core_num][CX_OTHER] >> 16) & 0xFFFF, off - CORE_OTHER
        return None, off

    def get(self, m, off: int) -> int:
        core, o = self._block(m, off)
        if core is not None:
            if core >= self.cps.n:
                return 0
            if o == CX_CONFIG:                     # PVPE 9:0 = the core's VPEs - 1
                return self.cps.vpes - 1           # (MD00904 GCR_CL_CONFIG; no IOCU)
            if o == CX_ID:
                return core
            return self.core[core].get(o, 0)
        if o == GCR_CONFIG:                        # 4 regions, NUMIOCU 11:8, PCORES
            return (4 << 16) | (int(self.cps.iocu) << 8) | ((self.cps.n - 1) & 0xFF)
        if o == GCR_BASE:
            return self.base | self.base_attr
        cm = self.cps.cm
        if o in _CM_REGS:
            return getattr(cm, _CM_REGS[o])
        if o == GCR_REV:
            return CM2_REV
        if o == GCR_GIC_STATUS:
            return 1                               # EX: a GIC is connected
        if o == GCR_CPC_STATUS:
            return 1                               # EX: a CPC is connected
        return self.regs.get(o, 0)

    def put(self, m, off: int, value: int) -> None:
        core, o = self._block(m, off)
        if core is not None:
            if core >= self.cps.n:
                return
            regs = self.core[core]
            if o == CX_COHERENCE:
                regs[o] = value & 0xFF
                self.cps.coherence_changed(core)
            elif o == CX_OTHER:                    # GCR_CL_ / GCR_CO_OTHER: R/W
                regs[o] = value & 0xFFFF0000
            elif o == CX_RESET_BASE:
                regs[o] = value & 0xFFFFF000
            elif o == CX_RESET_EXT_BASE:
                regs[o] = (value & EXT_BASE_MASK) | 1          # PRESENT reads 1
            return
        if o == GCR_BASE:
            self.base = value & 0xFFFF8000
            self.base_attr = value & 0xF3
            for c in self.cps.cores:
                c.cmgcr_moved(self.base)
        elif o in _CM_REGS:                        # ERROR_CAUSE / MULT: R/W (Table
            mask = {GCR_L2_ONLY_SYNC_BASE: 0xFFFFF001,  # 11.29: a type is kept until
                    GCR_ERROR_MULT: 0x1F}.get(o, M32)   # written 0)
            setattr(self.cps.cm, _CM_REGS[o], value & mask)
            if o in (GCR_ERROR_CAUSE, GCR_ERROR_MASK):
                self.cps.cm.error_changed()
        elif o in GCR_REGS:                        # REGn_BASE / REGn_MASK
            self.regs[o] = value & (0xFFFF0000 if (o - 0x90) % 16 == 0 else 0xFFFF00F7)
        elif o == GCR_ACCESS:
            self.regs[o] = value & 0xFF
        elif o == GCR_CONTROL:
            self.regs[o] = value & GCR_CONTROL_MASK
        elif o == GCR_GIC_BASE:
            self.regs[o] = value & 0xFFFE0001
            gic = self.cps.gic
            gic.base = value & 0xFFFE0000 if value & 1 else None
            gic.update()
        elif o == GCR_CPC_BASE:
            self.regs[o] = value & 0xFFFF8001
            self.cps.cpc.base = value & 0xFFFF8000 if value & 1 else None
        elif o in self.regs:
            self.regs[o] = value & M32


class CPC(_Registers):
    """The Cluster Power Controller (a bus device, at GCR_CPC_BASE when the
    GCR enables it)."""

    def __init__(self, cps: "CPS") -> None:
        self.cps = cps
        self.base: int | None = None              # disabled until GCR_CPC_BASE.EN
        # MD01025 Table 10.8: ACCESS resets to 0xFF; SEQDEL / RAIL / RESETLEN
        # are 10-bit fields (their delays: not modelled)
        self.regs = {CPC_ACCESS: 0xFF, CPC_SEQDEL: 0, CPC_RAIL: 0, CPC_RESETLEN: 0}
        self.other = [0] * cps.n

    def covers(self, pa: int) -> bool:
        return self.base is not None and self.base <= pa < self.base + CPC_SIZE

    def load(self, m, pa: int, size: int) -> int:
        return self.read(m, pa - self.base, size)

    def store(self, m, pa: int, value: int, size: int) -> None:
        if not self.regs[CPC_ACCESS] >> m.core_num & 1:
            return
        self.write(m, pa - self.base, value, size)

    def tick(self, m, cycles: int) -> None:
        pass

    def _core(self, m, off: int) -> tuple[int | None, int]:
        if CORE_LOCAL <= off < CORE_LOCAL + 0x2000:
            return m.core_num, off - CORE_LOCAL
        if CORE_OTHER <= off < CORE_OTHER + 0x2000:
            return (self.other[m.core_num] >> 16) & 0xFF, off - CORE_OTHER
        return None, off

    def get(self, m, off: int) -> int:
        core, o = self._core(m, off)
        if core is not None:
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
        return self.regs.get(o, 0)

    def put(self, m, off: int, value: int) -> None:
        core, o = self._core(m, off)
        if core is not None:
            if core >= self.cps.n:
                return
            if o == CPC_CX_CMD:
                self.cps.command(core, value & 0xF)
            elif o == CPC_CX_OTHER:                # CPC_CL_ / CPC_CO_OTHER: R/W
                self.other[core] = value & 0x00FF0000
            return
        if o in self.regs:
            self.regs[o] = value & (0xFF if o == CPC_ACCESS else 0x3FF)

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
    runs and the others are powered down."""

    def __init__(self, n: int, factory: Callable[["CPS", int], Any],
                 on_start: Callable[[int, Any], None] | None = None, *,
                 l2=None, l3=None, ecc: bool = False, hardware_init: bool = False,
                 iocu: bool = False, vpes: int = 1) -> None:
        from .coherence import CoherenceManager
        if not 1 <= n <= 4:
            raise ValueError(f"a CPS has 1 to 4 cores, not {n}")
        self.n = n
        #: VPEs per core (the MT Module) and the GIC's processors (VPs)
        self.vpes = vpes
        self.nvp = n * vpes
        #: an I/O Coherence Unit (GCR_CONFIG.NUMIOCU = 1): the I/O masters
        #: behind it (the Malta's PCI DMA) make coherent requests
        self.iocu = iocu
        self.factory = factory
        self.on_start = on_start
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
            m.core_num = core
            m.join_system(self)
            m.devices[:0] = self.devices
            self.vps[core * self.vpes + k] = m
        for m in ms:
            self.gic.attach(m)
        return ms

    def vps_of(self, core: int) -> list:
        """The Machines of ``core``'s VPEs."""
        return self.vps[core * self.vpes:(core + 1) * self.vpes]

    def vp_core(self, v: int) -> int:
        return v // self.vpes

    # -- power ---------------------------------------------------------------------------

    def running(self) -> list[int]:
        """The cores that execute (non-coherent / coherent execution)."""
        return [i for i in range(self.n) if self.state[i] == SEQ_U5]

    def coherent(self, core: int) -> bool:
        """In the coherence domain (GCR_Cx_COHERENCE.COHDOMAINEN != 0)."""
        return bool(self.gcr.core[core][CX_COHERENCE])

    def seq_state(self, core: int) -> int:
        """CPC_Cx_STAT_CONF.SEQSTATE: U5 non-coherent execution, U6 coherent
        execution once the core has joined the coherence domain."""
        st = self.state[core]
        return SEQ_U6 if st == SEQ_U5 and self.coherent(core) else st

    def start(self, core: int, pc: int | None = None) -> Any:
        """Reset ``core`` and let it run: a fresh core (the hardware reset
        state) at its reset vector GCR_Cx_RESET_BASE, non-coherent."""
        ms = self._build(core)
        regs = self.gcr.core[core]
        if regs[CX_RESET_EXT_BASE] & EXT_UEB:     # SI_UseExceptionBase (Table 11.65)
            base = (regs[CX_RESET_BASE] & 0x3FFFF000) | 0x80000000
        else:
            base = RESET_VECTOR
        for m in ms:
            m.clock_offset = self.clock - m.cycles     # its cycles count from now
            m.bev_base = (base + 0x200) & M32          # its BEV vectors follow it
            m.pc = (pc if pc is not None else base) & M32
        m = ms[0]
        self.cores[core] = m
        self.state[core] = SEQ_U5
        self.gic.reattach(core)
        self.pending[core] = None
        if self.on_start is not None:           # each VPE (VP core * vpes + n)
            for k, vm in enumerate(ms):
                self.on_start(core * self.vpes + k, vm)
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
        the coherence domain."""
        self.gcr.regs[GCR_ACCESS] = 0xFF               # every requester (the reset value)
        self.gcr.regs[GCR_CPC_BASE] = CPC_BASE_PA | 1
        self.cpc.base = CPC_BASE_PA
        from .gic import GIC_BASE_PA
        self.gcr.regs[GCR_GIC_BASE] = GIC_BASE_PA | 1
        self.gic.boot()
        for i in range(self.n):
            self.gcr.core[i][CX_COHERENCE] = 0xFF
            self.state[i] = SEQ_U5

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
        without coherence) clears their LLbit for its 32-byte block."""
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
        if not hidden:
            block = pa & ~31
            for other in self.vps:
                if other is not m and other.lr_valid and other.lr_pa is not None \
                        and other.lr_pa & ~31 == block:
                    other.lr_valid = False
        return False

    def describe(self) -> list[str]:
        out = []
        for i in range(self.n):
            m = self.cores[i]
            st = self.seq_state(i)
            out.append(f"core {i}: {SEQ_NAMES.get(st, st)}, "
                       f"pc 0x{m.pc:08x}" + (" (halted)" if m.halted else ""))
        return out
