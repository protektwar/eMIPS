"""The Global Interrupt Controller of a MIPS Coherent Processing System
(PLAN M11 phase 4).

Sources: the MIPS CPS training notes ("Global Interrupt Controller": what
every register does), Boot-MIPS (MD00901 4.28 init_gic: the GIC at
0x1BDC0000 enabled through GCR_GIC_BASE, a 40-source system, sources
24-39 rising-edge for inter-processor interrupts, 24 + n and 32 + n to
VPE n; GIC_VPEi_CTL's routable bits; main.c's IPIs through GIC_SH_WEDGE),
Linux (``arch/mips/include/asm/mips-gic.h`` — offsets and fields) and,
checked after M12 against it, the P5600 manual (MD01025 chapter 12:
the register tables, reset values — every source MAP_TO_PIN pin 0 of no
core, the local mask 0x7F, the local maps WD → NMI, TIMER / PERFCTR / FDC
→ pin 5, the others pin 0 — the watchdog (12.3.7.2, 12.6.4), the
COMPARE equality, software interrupts through the GIC only in EIC mode).

Layout (128 KB at GCR_GIC_BASE when GIC_EN)::

    0x00000 shared: CONFIG (COUNTSTOP 28, COUNTBITS 27:24 = 8: 64 bits,
            NUMINTERRUPTS 23:16 = groups of 8 - 1, PVPS 6:0 = VPs - 1),
            COUNTER 0x10 / 0x14, POL 0x100, TRIG 0x180, DUAL 0x200,
            WEDGE 0x280 (31 = set / clear, 30:0 = source), RMASK 0x300,
            SMASK 0x380, MASK 0x400, PEND 0x480 (a bit per source),
            MAPn_PIN 0x500 + 4n (31 to pin, 30 to NMI, 29 to YQ, 5:0 pin),
            MAPn_VP 0x2000 + 0x20 n (a bit per VP)
    0x08000 the accessing core's local registers, 0x0C000 the "other"
            core's (GIC_VL_OTHER): CTL 0x00, PEND 0x04, MASK 0x08, RMASK
            0x0C, SMASK 0x10, the maps WD 0x40 COMPARE 0x44 TIMER 0x48 FDC
            0x4C PERFCTR 0x50 SWINT0 0x54 SWINT1 0x58, OTHER 0x80, IDENT
            0x88, WD_CONFIG0 / COUNT0 / INITIAL0 0x90 / 0x94 / 0x98,
            COMPARE 0xA0 / 0xA4, EIC shadow sets 0x100
    0x10000 user-mode visible: the counter (0x0 / 0x4)

A source is **level** (TRIG 0: pending while its input is at the POL
level) or **edge** (TRIG 1: a POL edge — or both with DUAL — latches it
pending; a WEDGE write with bit 31 sets it, without clears it).  A
pending, enabled (MASK) source drives pin ``MAP & 63`` (HW0..HW5 =
Cause.IP2..IP7) of every VP set in its MAPn_VP, or their NMI input.  The
local COMPARE interrupt (the shared counter reaching the VP's COMPARE;
writing COMPARE clears it) goes through the VP's local mask and its map.

**Routable local interrupts** (M12; the MIPS GIC training notes,
"Local Interrupt Control"): GIC_VPEi_CTL's TIMER / PERFCNT / SWINT
_ROUTABLE bits are preset when the core is built; eMIPS builds them
routable (``GIC(routable=True)``; FDC: no FDC).  A routable interrupt is
not hardwired to a pin: the core sends its request out (Cause.TI, PCI,
IP0, IP1 → ``Machine.irq_outputs``), it is pending in the VP's local PEND
and goes, when enabled in the local MASK, where its map says.  (The
software interrupts still act inside the core too, Vol. III Table 6.2.)

**EIC mode** (Vol. III 6.1.1.3; the P5600 manual MD01025 12.3.9 /
12.5.3.15; the notes): GIC_VPEi_CTL.EIC_MODE, the only writable CTL bit,
drives the core's SI_EICPresent (Config3.VEIC).  Then a map's field "represents
one less than the EIC interrupt level" (MD01025 12.5.3.15; Linux
``GIC_PIN_TO_VEC_OFFSET``): the GIC sends the core the highest pending
enabled level as RIPL (0 = none), the vector = RIPL (EIC option 1, as the
24K), with the shadow set of GIC_VPEi_EIC_SHADOW_SET[vector] (the 1004K
generation; the P5600 has no shadow sets).  The P5600's GIC_COREi_CTL has
no routable bits (its local interrupts always go through the GIC); eMIPS
follows the 1004K-generation GIC, where they are a build option.

eMIPS choices (``sim.DEVIATIONS`` "GIC"): 40 sources as Boot-MIPS's
example; the counter counts system cycles; no MT ASE (no yield
qualifiers); the watchdog counts system cycles (the rate is not given)
and stops while its core sleeps unless WAIT; REVISIONID reads 0 (as QEMU).
"""

from __future__ import annotations

from .cps import SEQ_U5, SEQ_U6, _Registers

M32 = 0xFFFFFFFF
#: Boot-MIPS: the GIC's physical address
GIC_BASE_PA = 0x1BDC0000
GIC_SIZE = 0x20000
#: external interrupt sources (Boot-MIPS's example system: 5 groups of 8)
SOURCES = 40
#: what the eMIPS devices are wired to
SRC_CONSOLE_TX, SRC_CONSOLE_RX = 0, 1
#: the Coherence Manager's error interrupt (MD01025 Table 11.11 "Signal
#: Interrupt"; where it is wired is the system's choice: eMIPS source 2)
SRC_CM_ERROR = 2
#: Boot-MIPS: sources 24 + n and 32 + n interrupt VPE n (IPIs)
FIRST_IPI = 24

SHARED, LOCAL, OTHER, USER = 0x0, 0x8000, 0xC000, 0x10000
SH_CONFIG, SH_COUNTER_LO, SH_COUNTER_HI, SH_REVISION = 0x00, 0x10, 0x14, 0x20
SH_POL, SH_TRIG, SH_DUAL, SH_WEDGE = 0x100, 0x180, 0x200, 0x280
SH_RMASK, SH_SMASK, SH_MASK, SH_PEND = 0x300, 0x380, 0x400, 0x480
SH_MAP_PIN, SH_MAP_VP = 0x500, 0x2000
COUNTSTOP = 1 << 28
MAP_TO_PIN, MAP_TO_NMI, MAP_TO_YQ = 1 << 31, 1 << 30, 1 << 29
WEDGE_RW = 1 << 31
# local registers
VL_CTL, VL_PEND, VL_MASK, VL_RMASK, VL_SMASK = 0x00, 0x04, 0x08, 0x0C, 0x10
VL_MAP = 0x40                               # WD .. SWINT1: 0x40 .. 0x58
VL_OTHER, VL_IDENT = 0x80, 0x88
VL_WD = (0x90, 0x94, 0x98)
VL_COMPARE_LO, VL_COMPARE_HI = 0xA0, 0xA4
VL_EIC_SS = 0x100
#: local interrupts (pend / mask bits, Linux mips_gic_local_interrupt)
LOCAL_WD, LOCAL_COMPARE, LOCAL_TIMER, LOCAL_PERFCTR = 0, 1, 2, 3
LOCAL_SWINT0, LOCAL_SWINT1, LOCAL_FDC = 4, 5, 6
LOCAL_NAMES = ("watchdog", "GIC count/compare", "CP0 timer", "performance counter",
               "software 0", "software 1", "fast debug channel")
#: GIC_VPEi_WD_CONFIG fields (MD01025 Table 12.52) and its TYPE values
WD_RESET, WD_INTR, WD_WAIT, WD_DEBUG, WD_TYPE, WD_START = 0x80, 0x40, 0x20, 0x10, 0xE, 1
WD_ONE_TRIP, WD_SECOND, WD_PIT = 0, 1, 2
#: CPC sequencer states in which a core is up (its watchdog counts)
_AWAKE = (SEQ_U5, SEQ_U6)
#: GIC_VPEi_CTL (Linux GIC_VX_CTL_*): EIC_MODE (R/W), the routable bits (RO)
CTL_EIC, CTL_TIMER, CTL_PERFCNT, CTL_SWINT, CTL_FDC = 1, 2, 4, 8, 16
#: the core's own requests (Machine.irq_outputs order) -> local interrupt
_CORE_LOCAL = (LOCAL_TIMER, LOCAL_PERFCTR, LOCAL_SWINT0, LOCAL_SWINT1)
#: map register index (0x40 + 4 i) of each local interrupt (FDC is 4th)
_MAP_INDEX = {LOCAL_WD: 0, LOCAL_COMPARE: 1, LOCAL_TIMER: 2, LOCAL_FDC: 3,
              LOCAL_PERFCTR: 4, LOCAL_SWINT0: 5, LOCAL_SWINT1: 6}


class _VP:
    """One VP's local registers (a core: one VPE each)."""

    def __init__(self) -> None:
        self.mask = 0x7F                    # MD01025 Table 12.45: every bit resets to 1
        self.pend = 0
        #: interrupt lines a board drives into the core directly, ORed with
        #: the GIC's pins (the Malta's COREHI etc.; ignored in EIC mode)
        self.extra = 0
        # Table 12.49: WD -> NMI; COMPARE, SWINTx -> pin 0; TIMER, PERFCTR (and
        # FDC) -> pin 5 (the map order: WD COMPARE TIMER FDC PERFCTR SWINT0 SWINT1)
        self.maps = [MAP_TO_NMI, MAP_TO_PIN, MAP_TO_PIN | 5, MAP_TO_PIN | 5,
                     MAP_TO_PIN | 5, MAP_TO_PIN, MAP_TO_PIN]
        self.other = 0
        #: the watchdog (12.3.7.2 / 12.6.4): WD_CONFIG (WDRESET 7, WDINTR 6,
        #: WAIT 5, DEBUG 4, TYPE 3:1, WD_START 0), WD_COUNT, WD_INITIAL
        self.wd_config = 0
        self.wd_count = 0
        self.wd_initial = 0
        #: counting down (WD_START, not stopped by a one-trip expiry)
        self.wd_running = False
        #: second countdown mode: the first expiry happened (the next resets)
        self.wd_tripped = False
        self.compare = (1 << 64) - 1
        self.eic_ss = [0] * 64
        #: GIC_VPEi_CTL.EIC_MODE: the core's SI_EICPresent
        self.eic = False
        #: the outputs last driven: pin bits (EIC mode: the RIPL), NMI
        self.pins = 0
        self.nmi = False


class GIC(_Registers):
    """The GIC of a :class:`~emips.cps.CPS` (a bus device of every core)."""

    def __init__(self, cps, routable: bool = True) -> None:
        self.cps = cps
        self.n = SOURCES
        #: GIC_VPEi_CTL's routable bits (a build option): the cores'
        #: timer, performance counter and software interrupts
        self.ctl_routable = (CTL_TIMER | CTL_PERFCNT | CTL_SWINT) if routable else 0
        #: the GIC appears at ``base`` when GCR_GIC_BASE.EN is set
        self.base: int | None = None
        self.countstop = False
        self.counter = 0
        #: the system clock the counter was last brought up to
        self._clock = 0
        self.pol = self.trig = self.dual = 0
        self.mask = 0
        #: edge sources: the latched edges; level sources: the input levels
        self.edges = 0
        self.inputs = 0
        self.map_pin = [MAP_TO_PIN] * self.n
        self.map_vp = [0] * self.n
        self.vps = [_VP() for _ in range(cps.n)]

    # -- the bus ---------------------------------------------------------------------

    def covers(self, pa: int) -> bool:
        return self.base is not None and self.base <= pa < self.base + GIC_SIZE

    def load(self, m, pa: int, size: int) -> int:
        self._sync(m)
        return self.read(m, pa - self.base, size)

    def store(self, m, pa: int, value: int, size: int) -> None:
        self._sync(m)
        self.write(m, pa - self.base, value, size)
        self.update()

    def tick(self, m, cycles: int) -> None:
        self._sync(m)

    # -- time: the shared counter, the local compares ------------------------------------

    def _sync(self, m) -> None:
        """Bring the counter up to the system's clock (the core ``m``'s
        cycles + its offset); a counter passing a VP's COMPARE latches its
        COMPARE interrupt; the watchdogs count down."""
        now = m.cycles + getattr(m, "clock_offset", 0)
        if now <= self._clock:
            return
        delta, self._clock = now - self._clock, now
        hit = False
        if not self.countstop:
            old = self.counter
            self.counter = (old + delta) & ((1 << 64) - 1)
            for vp in self.vps:
                if old < vp.compare <= self.counter:
                    vp.pend |= 1 << LOCAL_COMPARE
                    hit = True
        reset = False
        for v, vp in enumerate(self.vps):
            if vp.wd_running and self._wd_counts(v):
                fired, reset_now = self._wd_run(vp, delta)
                hit = hit or fired
                reset = reset or reset_now
        if hit:
            self.update()
        if reset:
            self._wd_reset()

    def _check_compare(self, vp: _VP) -> None:
        """MD01025 12.3.7.1: "when the count value equals the compare value"
        — also when software writes one equal to the other."""
        if vp.compare == self.counter:
            vp.pend |= 1 << LOCAL_COMPARE

    # -- the watchdogs (MD01025 12.3.7.2, 12.6.4) -------------------------------------------

    def _wd_counts(self, v: int) -> bool:
        """WAIT = 0: no countdown while the core sleeps (WAIT, a CPC low
        power state); DEBUG: no debug mode yet (EJTAG, M14)."""
        vp = self.vps[v]
        if vp.wd_config & WD_WAIT:
            return True
        cps = self.cps
        if v >= len(cps.cores) or cps.state[v] not in _AWAKE:
            return False
        return not cps.cores[v].waiting

    def _wd_run(self, vp: _VP, cycles: int) -> tuple[bool, bool]:
        """Count ``cycles`` down; at 0 (Table 12.52 TYPE): one trip — the
        interrupt, stop; second countdown — the interrupt and reload, a
        second expiry before software reloads it resets every core; PIT —
        the interrupt, reload, go on.  Returns (interrupt, reset)."""
        fired = False
        while cycles >= vp.wd_count:
            cycles -= vp.wd_count
            vp.wd_config |= WD_INTR
            vp.pend |= 1 << LOCAL_WD
            fired = True
            kind = (vp.wd_config >> 1) & 7
            if kind == WD_SECOND:
                if vp.wd_tripped:
                    vp.wd_running = False
                    vp.wd_count = 0
                    return True, True
                vp.wd_tripped = True
            elif kind != WD_PIT:                    # one trip (reserved: as one trip)
                vp.wd_running = False
                vp.wd_count = 0
                return True, False
            vp.wd_count = vp.wd_initial
            if vp.wd_count == 0:                    # INITIAL 0: expires at once again
                vp.wd_count = 1
        vp.wd_count -= cycles
        return fired, False

    def _wd_reset(self) -> None:
        """Second-countdown expiry: SI_Reset to every core that is up
        (WDRESET says so; it survives the reset — the GIC is not reset)."""
        cps = self.cps
        for vp in self.vps:
            vp.wd_config |= WD_RESET
        for core in range(cps.n):
            if cps.state[core] in _AWAKE:
                cps.start(core)

    # -- sources -----------------------------------------------------------------------------

    def set_input(self, src: int, level: bool) -> None:
        """A device drives source ``src`` (an edge source latches the
        edges its POL / DUAL select)."""
        bit = 1 << src
        old = bool(self.inputs & bit)
        self.inputs = self.inputs | bit if level else self.inputs & ~bit
        if self.trig & bit and old != level:
            rising = level
            if self.dual & bit or rising == bool(self.pol & bit):
                self.edges |= bit
        self.update()

    def set_extra(self, core: int, line: int, level: bool) -> None:
        """A board's direct interrupt line ``line`` (HW0..5) of ``core``."""
        if core < len(self.vps):
            vp = self.vps[core]
            vp.extra = vp.extra | 1 << line if level else vp.extra & ~(1 << line)
            self.update()

    def pending(self) -> int:
        """GIC_SH_PEND: level sources at their POL level, latched edges."""
        level = ~self.trig & ~(self.inputs ^ self.pol)
        return ((level | (self.trig & self.edges)) & ((1 << self.n) - 1))

    def wedge(self, value: int) -> None:
        src = value & ~WEDGE_RW & M32               # INTERRUPT: bits 30:0 (Table 12.30)
        if src >= self.n:
            return
        if not self.trig >> src & 1:
            return                               # only edge sources: the notes
        if value & WEDGE_RW:
            self.edges |= 1 << src
        else:
            self.edges &= ~(1 << src)

    # -- the outputs -------------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.base is not None

    def local_pending(self, v: int) -> int:
        """GIC_VPEi_PEND: the latched COMPARE interrupt and the requests the
        core sends out (timer, performance counter, software 0 / 1) when
        they are routable."""
        vp = self.vps[v]
        pend = vp.pend
        cores = self.cps.cores
        if v < len(cores):
            out = cores[v].irq_outputs()
            for bit, local in zip(out, _CORE_LOCAL):
                if local in (LOCAL_SWINT0, LOCAL_SWINT1):
                    # MD01025 12.3.8.7: "only used by the GIC in EIC mode. In
                    # non-EIC mode they are routed internally within the CPU"
                    routable = vp.eic and self.ctl_routable & CTL_SWINT
                else:
                    routable = self.ctl_routable & (CTL_TIMER if local == LOCAL_TIMER
                                                    else CTL_PERFCNT)
                if bit and routable:
                    pend |= 1 << local
        return pend

    def update(self) -> None:
        """Drive every core's interrupt lines and NMI input from the
        pending, enabled sources and local interrupts: pins HW0..HW5, or in
        EIC mode the highest vector as RIPL with its shadow set."""
        active = self.pending() & self.mask if self.enabled else 0
        cores = self.cps.cores
        for v, vp in enumerate(self.vps):
            pins, nmi, ripl = 0, False, 0
            if self.enabled:
                maps = [self.map_pin[src] for src in range(self.n)
                        if active >> src & 1 and self.map_vp[src] >> v & 1]
                local = self.local_pending(v) & vp.mask
                maps += [vp.maps[_MAP_INDEX[i]] for i in range(7) if local >> i & 1]
                for mapping in maps:
                    pins, nmi = _route(mapping, pins, nmi)
                    if mapping & MAP_TO_PIN:
                        ripl = max(ripl, min((mapping & 0x3F) + 1, 63))
            lines = ripl if vp.eic else pins | vp.extra
            if v < len(cores):
                m = cores[v]
                ss = vp.eic_ss[ripl] if vp.eic else 0
                if (lines, ss) != (m.irq_lines, m.eic_ss):
                    m.set_irq_lines(lines, ss)
                if nmi and not vp.nmi:
                    m.nmi()                      # an edge on the NMI input
            vp.pins, vp.nmi = lines, nmi

    def attach(self, m) -> None:
        """Wire a core to its VP: its timer / performance counter go out
        to the GIC when routable, its requests update the outputs, its
        SI_EICPresent follows EIC_MODE."""
        m.local_routed = bool(self.ctl_routable & (CTL_TIMER | CTL_PERFCNT))
        m.on_irq_outputs = self.update
        m.set_eic_present(self.vps[m.cpu_num].eic)
        m._update_ip()

    def _eic_vps(self, vps: int) -> bool:
        """Are the VPs of a MAPn_VP in EIC mode?"""
        return any(vps >> v & 1 and vp.eic for v, vp in enumerate(self.vps)) and \
            all(vp.eic for v, vp in enumerate(self.vps) if vps >> v & 1)

    def reattach(self, core: int) -> None:
        """A core was (re)started: its pins start low; drive them again."""
        self.vps[core].pins, self.vps[core].nmi = 0, False
        self.update()

    # -- registers ---------------------------------------------------------------------------

    def get(self, m, off: int) -> int:
        if off >= USER:                                  # user-mode visible
            o = off - USER
            return self._counter_word(o == 4) if o in (0, 4) else 0
        if off >= LOCAL:
            v = m.cpu_num if off < OTHER else self.vps[m.cpu_num].other
            return self._get_vp(v, off & 0x3FFF) if v < len(self.vps) else 0
        if off == SH_CONFIG:
            return ((COUNTSTOP if self.countstop else 0) | (8 << 24)
                    | ((self.n // 8 - 1) << 16) | (len(self.vps) - 1))
        if off in (SH_COUNTER_LO, SH_COUNTER_HI):
            return self._counter_word(off == SH_COUNTER_HI)
        if off == SH_REVISION:
            return 0
        bits = {SH_POL: self.pol, SH_TRIG: self.trig, SH_DUAL: self.dual,
                SH_MASK: self.mask, SH_PEND: self.pending()}
        for base, value in bits.items():
            if base <= off < base + 0x20:
                return (value >> (8 * (off - base))) & M32
        if SH_MAP_PIN <= off < SH_MAP_PIN + 4 * self.n:
            return self.map_pin[(off - SH_MAP_PIN) // 4]
        if SH_MAP_VP <= off < SH_MAP_VP + 0x20 * self.n:
            src, word = divmod(off - SH_MAP_VP, 0x20)
            return self.map_vp[src] if word == 0 else 0
        return 0                          # WEDGE, RMASK, SMASK: write-only

    def _counter_word(self, high: bool) -> int:
        return (self.counter >> 32) & M32 if high else self.counter & M32

    def _get_vp(self, v: int, o: int) -> int:
        vp = self.vps[v]
        if o == VL_CTL:
            return self.ctl_routable | (CTL_EIC if vp.eic else 0)
        if o == VL_PEND:
            return self.local_pending(v)
        if o == VL_MASK:
            return vp.mask
        if VL_MAP <= o < VL_MAP + 0x1C:
            return vp.maps[(o - VL_MAP) // 4]
        if o == VL_OTHER:
            return vp.other
        if o == VL_IDENT:
            return v
        if o == VL_COMPARE_LO:
            return vp.compare & M32
        if o == VL_COMPARE_HI:
            return (vp.compare >> 32) & M32
        if VL_EIC_SS <= o < VL_EIC_SS + 0x100:
            return vp.eic_ss[(o - VL_EIC_SS) // 4]
        if o == VL_WD[0]:
            return vp.wd_config
        if o == VL_WD[1]:
            return vp.wd_count & M32
        if o == VL_WD[2]:
            return vp.wd_initial
        return 0                          # RMASK / SMASK (write-only)

    def put(self, m, off: int, value: int) -> None:
        value &= M32
        if off >= USER:
            return                        # read-only
        if off >= LOCAL:
            v = m.cpu_num if off < OTHER else self.vps[m.cpu_num].other
            if v < len(self.vps):
                self._put_vp(v, off & 0x3FFF, value)
            return
        if off == SH_CONFIG:
            self.countstop = bool(value & COUNTSTOP)
            return
        if off in (SH_COUNTER_LO, SH_COUNTER_HI):
            if self.countstop:            # MD01025 12.5.3.2: written only while stopped
                shift = 32 if off == SH_COUNTER_HI else 0
                self.counter = (self.counter & ~(M32 << shift)) | (value << shift)
                for vp in self.vps:
                    self._check_compare(vp)
            return
        if off == SH_WEDGE:
            self.wedge(value)
            return
        for base, name in ((SH_POL, "pol"), (SH_TRIG, "trig"), (SH_DUAL, "dual")):
            if base <= off < base + 0x20:
                shift = 8 * (off - base)
                cur = getattr(self, name)
                new = (cur & ~(M32 << shift)) | (value << shift)
                setattr(self, name, new & ((1 << self.n) - 1))
                return
        if SH_RMASK <= off < SH_RMASK + 0x20:
            self.mask &= ~(value << (8 * (off - SH_RMASK)))
            return
        if SH_SMASK <= off < SH_SMASK + 0x20:
            self.mask |= (value << (8 * (off - SH_SMASK))) & ((1 << self.n) - 1)
            return
        if SH_MAP_PIN <= off < SH_MAP_PIN + 4 * self.n:
            self.map_pin[(off - SH_MAP_PIN) // 4] = value & (MAP_TO_PIN | MAP_TO_NMI
                                                             | MAP_TO_YQ | 0x3F)
            return
        if SH_MAP_VP <= off < SH_MAP_VP + 0x20 * self.n:
            src, word = divmod(off - SH_MAP_VP, 0x20)
            if word == 0:
                self.map_vp[src] = value & ((1 << len(self.vps)) - 1)

    def _put_vp(self, v: int, o: int, value: int) -> None:
        vp = self.vps[v]
        if o == VL_CTL:                   # only EIC_MODE is writable
            vp.eic = bool(value & CTL_EIC)
            if v < len(self.cps.cores):
                self.cps.cores[v].set_eic_present(vp.eic)
        elif o == VL_RMASK:
            vp.mask &= ~value & 0x7F
        elif o == VL_SMASK:
            vp.mask |= value & 0x7F
        elif VL_MAP <= o < VL_MAP + 0x1C:
            vp.maps[(o - VL_MAP) // 4] = value & (MAP_TO_PIN | MAP_TO_NMI | MAP_TO_YQ | 0x3F)
        elif o == VL_OTHER:
            vp.other = value & 0xFFFF               # VPENUM 15:0 (Table 12.50)
        elif o in (VL_COMPARE_LO, VL_COMPARE_HI):
            shift = 32 if o == VL_COMPARE_HI else 0
            vp.compare = (vp.compare & ~(M32 << shift)) | (value << shift)
            vp.pend &= ~(1 << LOCAL_COMPARE)          # writing COMPARE clears it
            self._check_compare(vp)
        elif o == VL_WD[0]:                           # WD_CONFIG (Table 12.52)
            clear = value & (WD_RESET | WD_INTR)      # R/WC: a 1 clears
            vp.wd_config = ((vp.wd_config & ~clear & (WD_RESET | WD_INTR))
                            | (value & (WD_WAIT | WD_DEBUG | WD_TYPE | WD_START)))
            if not vp.wd_config & WD_INTR:
                vp.pend &= ~(1 << LOCAL_WD)
            if value & WD_START:                      # reload INITIAL and start
                vp.wd_count, vp.wd_running, vp.wd_tripped = vp.wd_initial, True, False
            else:
                vp.wd_running = False
        elif o == VL_WD[2]:                           # WD_INITIAL (UNPREDICTABLE while
            vp.wd_initial = value                     # running: just stored)
        elif VL_EIC_SS <= o < VL_EIC_SS + 0x100:
            vp.eic_ss[(o - VL_EIC_SS) // 4] = value & 0xF

    # -- what the boot firmware does --------------------------------------------------------

    def boot(self) -> None:
        """Boot-MIPS init_gic: the GIC enabled at :data:`GIC_BASE_PA`;
        sources 24-39 rising-edge, enabled, 24 + n and 32 + n to VPE n (the
        IPIs; their pin: the reset value, pin 0); and, for eMIPS's
        console, its TX / RX sources level active-high, enabled, to core
        0's pins 0 / 1 (Cause.IP2 / IP3, as without a GIC)."""
        self.base = GIC_BASE_PA
        ipis = ((1 << 16) - 1) << FIRST_IPI
        self.pol |= ipis
        self.trig |= ipis
        self.mask |= ipis
        for n in range(8):
            for src in (FIRST_IPI + n, FIRST_IPI + 8 + n):
                self.map_vp[src] = (1 << n) & ((1 << len(self.vps)) - 1)
        self.pol |= 1 << SRC_CM_ERROR      # the CM's error interrupt: level, active
        for src, pin in ((SRC_CONSOLE_TX, 0), (SRC_CONSOLE_RX, 1)):  # high (masked)
            self.pol |= 1 << src
            self.mask |= 1 << src
            self.map_pin[src] = MAP_TO_PIN | pin
            self.map_vp[src] = 1
        # (the core's timer / counters reset mapped to pin 5 = IP7 and enabled,
        # MD01025 Tables 12.45 / 12.49: nothing to do for them)
        self.update()

    def describe(self) -> list[str]:
        """The GIC for the debugger: the counter, the sources in use, every
        core's local interrupts and pins."""
        if not self.enabled:
            return ["GIC: disabled (GCR_GIC_BASE.GIC_EN = 0)"]
        out = [f"GIC at 0x{self.base:08x}: {self.n} sources, counter {self.counter}"
               + (" (stopped)" if self.countstop else "")]
        pend = self.pending()
        for src in range(self.n):
            bit = 1 << src
            if not self.mask & bit:
                continue
            kind = ("edge" + (", both" if self.dual & bit else
                              ", rising" if self.pol & bit else ", falling")
                    if self.trig & bit else
                    "level, active " + ("high" if self.pol & bit else "low"))
            out.append(f"  source {src:2d}: {kind}, "
                       f"{'enabled' if self.mask & bit else 'masked'}, "
                       f"{'PENDING' if pend & bit else 'idle'}, "
                       f"{_target(self.map_pin[src], self._eic_vps(self.map_vp[src]))} of VP(s) "
                       f"{_vps(self.map_vp[src])}")
        for v, vp in enumerate(self.vps):
            pend = self.local_pending(v)
            local = ", ".join(f"{LOCAL_NAMES[i]}{' PENDING' if pend >> i & 1 else ''}"
                              f" -> {_target(vp.maps[_MAP_INDEX[i]], vp.eic)}"
                              for i in range(7) if vp.mask >> i & 1)
            lines = (f"EIC mode, RIPL {vp.pins}" + (f" (shadow set {vp.eic_ss[vp.pins]})"
                                                    if vp.pins else "")
                     if vp.eic else f"pins {_bits(vp.pins)}")
            out.append(f"  VP {v}: {lines}"
                       + (", NMI" if vp.nmi else "")
                       + f"; local enabled: {local or 'none'}; COMPARE "
                       + (f"{vp.compare}" if vp.compare != (1 << 64) - 1 else "max"))
        return out


def _route(mapping: int, pins: int, nmi: bool) -> tuple[int, bool]:
    if mapping & MAP_TO_NMI:
        nmi = True
    if mapping & MAP_TO_PIN and (mapping & 0x3F) < 6:
        pins |= 1 << (mapping & 0x3F)
    return pins, nmi


def _target(mapping: int, eic: bool = False) -> str:
    """Where a map sends an interrupt (``eic``: to a VP in EIC mode)."""
    if mapping & MAP_TO_NMI:
        return "NMI"
    if mapping & MAP_TO_PIN:
        pin = mapping & 0x3F
        if eic:
            return f"EIC vector {min(pin + 1, 63)}"
        return f"pin {pin} (IP{pin + 2})" if pin < 6 else f"pin {pin} (no such pin)"
    return "nothing"


def _vps(mask: int) -> str:
    return ", ".join(str(i) for i in range(8) if mask >> i & 1) or "none"


def _bits(mask: int) -> str:
    return ", ".join(f"IP{i + 2}" for i in range(6) if mask >> i & 1) or "none"
