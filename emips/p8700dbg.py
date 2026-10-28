"""The P8700's Hybrid Debug, the hart's side (LF16; MD01502 rev. 1.84 1.2.2
"MIPS EJTAG with RISC-V Trace", 4.3, 7.4 the Hybrid Debug register, 13.1.4
the per-hart resources; the P8700 datasheet's "Core Debug Support"; MIPS
EJTAG 5.06, MD00047, ch. 2 / 3 / 5 for the mechanisms the manual names but
does not describe; the I6400's MD01196 Table 4.4 for the probe's vector).

The *MIPS Hybrid Debug Specification* the manual refers to is not public:
every choice below is in ``LF.md`` (LF16) and ``sim.DEVIATIONS["P8700
debug"]``.

**One Debug Mode, EJTAG's** (:class:`HartDebug`, ``m.hybrid``): entered by
a debug exception — the RISC-V causes of dcsr.cause (ebreak with
dcsr.ebreakm / s / u, an Sdtrig trigger with action 1, dcsr.step, a halt
request) and the MIPS ones (a DINT: CPC DBG_DBRK_REG or the probe's
EjtagBrk — dcsr.cause haltreq; the breakpoint controller's DIB / DDB —
cause trigger) —, the hart *executes* a debug handler with M-mode rights
(no interrupts, NMIs, triggers or single step) and leaves with DRET.
dpc is DEPC; the vector is dmseg's 0xFFFFFFFF_FF200200 when the DBU's
ProbTrap is set, else the hart's RESET_BASE + 0x480.  An exception in
Debug Mode changes only dpc and the Hybrid Debug register (DEXCCODE = the
RISC-V cause; DINT / DBS cleared) and enters the vector again (MD00047
2.4.3).

**The Hybrid Debug register** (``mipsdebug``, Table 7.45; at CSR 0x7CF —
not published): DM, CACHEEP / DBUSEP (R/W1: a pending imprecise error
deferred while IEXI), IEXI (set on entry, cleared on DRET; R/W), DBGVER 1,
DEXCCODE, DINT, DBS; DIBIMPR reads 0.

**dseg** (Debug Mode, before translation): 0xFFFFFFFF_FF200000 -
0xFFFFFFFF_FF3FFFFF (EJTAG's dseg sign-extended): dmseg (the DBU's RAM,
dmxseg, while the probe's ProbEn is set; else the access waits for a probe
forever) and drseg (8-byte registers: the DCR at 0 — IntE / NMIE gate the
hart's interrupts and NMIs outside Debug Mode, NMIpend, IB, DB, ProbEn —,
the breakpoints at 0x1000 / 0x2000, the hart's trace encoder at 0x8000).

**The breakpoint controller** (:class:`Breakpoints`, one in the cluster):
EJTAG's 8 instruction and 8 data breakpoints, the same registers in every
hart's drseg ("shared between all harts", 13.1.3), the match and the IBS
/ DBS status per hart ("each hart has its own independent breakpoint
control logic", 13.1.4.1); TCuse / TC (IBCn / DBCn 22, 31:24) select a hart
by its index in the cluster, ASIDuse compares satp.ASID[7:0].  A break
(BE) is precise: Debug Mode before the instruction / the access, dpc =
the instruction.
"""

from __future__ import annotations

from typing import Any

from . import ejtag as EJ
from .exceptions import ProbeWaitError

M64 = (1 << 64) - 1
#: dseg (EJTAG's, sign-extended to 64 bits)
DSEG_BASE = 0xFFFFFFFFFF200000
DRSEG_BASE = 0xFFFFFFFFFF300000
DSEG_END = 0xFFFFFFFFFF3FFFFF
#: the debug vector with the probe's ProbTrap (MD01196 Table 4.4), and its
#: offset from RESET_BASE without (MD00047 / the 24K's CPS rule)
PROBE_VECTOR = 0xFFFFFFFFFF200200
VECTOR_OFFSET = 0x480
#: the Hybrid Debug register (Table 7.45) and its bits
MIPSDEBUG = 0x7CF
DM, CACHEEP, DBUSEP, IEXI = 1 << 30, 1 << 22, 1 << 21, 1 << 20
DBGVER = 1
DEXC_SHIFT, DINT, DBS = 10, 1 << 5, 1 << 0
#: the drseg block of the hart's trace encoder (LF16c: emips.p8700trace)
TRACE_BASE, TRACE_SIZE = 0x8000, 0x1000
#: the breakpoint controller's size (13.1.3)
BREAKPOINTS = 8


def in_dseg(va: int) -> bool:
    return DSEG_BASE <= va <= DSEG_END


class Breakpoints(EJ.EJTAG):
    """The cluster's 8 instruction / 8 data breakpoints (EJTAG's registers,
    64 bits wide); per-hart state lives in each :class:`HartDebug`."""

    ib_count = db_count = BREAKPOINTS

    def __init__(self, s_mode: bool = True) -> None:
        self.m = None
        self.wide, self.mask, self.bus = True, M64, 8
        self.s_mode = s_mode
        #: the harts that look at these breakpoints (re-armed on a change)
        self.harts: list[HartDebug] = []
        self.ib_on = self.db_on = False
        self.reset()

    @property
    def asid_sup(self) -> bool:
        """satp's ASID with S-mode (paging)."""
        return self.s_mode

    @property
    def tc_use(self) -> bool:
        return True

    def changed(self) -> None:
        self.ib_on = self.ib_armed
        self.db_on = self.db_armed

    def names(self) -> dict[int, str]:
        """drseg register names by offset (the views)."""
        out = {0x0000: "DCR", EJ.IBS: "IBS", EJ.DBS: "DBS"}
        for n in range(self.ib_count):
            for off, name in ((EJ.IBA, "IBA"), (EJ.IBM, "IBM"), (EJ.IBASID, "IBASID"),
                              (EJ.IBC, "IBC")):
                out[0x1100 + 0x100 * n + off] = f"{name}{n}"
        for n in range(self.db_count):
            for off, name in ((EJ.DBA, "DBA"), (EJ.DBM, "DBM"), (EJ.DBASID, "DBASID"),
                              (EJ.DBC, "DBC"), (EJ.DBV, "DBV")):
                out[0x2100 + 0x100 * n + off] = f"{name}{n}"
        return out


class HartDebug:
    """One hart's Hybrid Debug state (see the module docstring)."""

    def __init__(self, m: Any, bp: Breakpoints | None = None, index: int = 0,
                 dbu: Any = None) -> None:
        self.m = m
        self.bp = bp if bp is not None else Breakpoints("s" in getattr(m, "modes", "msu"))
        self.bp.harts.append(self)
        #: the hart's index in the cluster (IBCn / DBCn's TC)
        self.index = index
        #: the cluster's Debug Unit (emips.p8700dbu), or None
        self.dbu = dbu
        #: the hart's trace encoder in drseg (LF16c), or None
        self.trace = None
        self.reset()

    def reset(self) -> None:
        self.iexi = False
        self.cacheep = self.dbusep = False
        self.dexccode = 0
        self.dint = self.dbs = False
        #: DCR (MD00047 Table 3.1): IntE, NMIE and SRstE as written
        self.dcr = EJ.DCR_INTE | EJ.DCR_NMIE | EJ.DCR_SRE
        self.ibs_bs = self.dbs_bs = 0
        #: a value compare a load waits for: (breakpoints, lanes)
        self._vc: tuple[int, int] | None = None

    # -- the probe (emips.p8700dbu) ----------------------------------------------------

    @property
    def prob_en(self) -> bool:
        return self.dbu is not None and self.dbu.prob_en

    @property
    def prob_trap(self) -> bool:
        return self.dbu is not None and self.dbu.prob_trap

    def vector(self) -> int:
        if self.prob_trap:
            return PROBE_VECTOR & self.m.amask
        cl = self.m.cluster                     # (the hart's RESET_BASE now: GCR.Core)
        base = cl.reset_base[self.index] if cl is not None else self.m.reset_pc
        return (base + VECTOR_OFFSET) & self.m.amask

    # -- Debug Mode ---------------------------------------------------------------------

    def entered(self, cause: int) -> int:
        """A debug exception (dcsr.cause ``cause``) put the hart in Debug
        Mode: the register's state; returns the vector."""
        from .simrv import CAUSE_HALTREQ, CAUSE_STEP
        self.iexi = True
        self.dint = cause == CAUSE_HALTREQ
        self.dbs = cause == CAUSE_STEP
        return self.vector()

    def exception(self, cause: int, pc: int) -> None:
        """An exception in Debug Mode (MD00047 2.4.3): dpc = the
        instruction, DEXCCODE, the cause bits cleared, the vector again."""
        m = self.m
        self.dexccode = cause & 0x3F
        self.dint = self.dbs = False
        m.csrs.dpc = pc & m.amask & ~1
        m.exception_log.append({"name": "exception in Debug Mode", "cause": cause,
                                "epc": pc, "debug": True, "priv": m.priv})
        m.pc = self.vector()

    def left(self) -> None:
        """DRET: IEXI cleared (a deferred imprecise error is taken now)."""
        self.iexi = False

    def defers(self) -> bool:
        """An imprecise error waits while IEXI = 1 (MD00047 2.4.2): CACHEEP."""
        if self.iexi:
            self.cacheep = True
            return True
        return False

    # -- the Hybrid Debug register ----------------------------------------------------------

    def read(self) -> int:
        return ((DM if self.m.debug_mode else 0) | (CACHEEP if self.cacheep else 0)
                | (DBUSEP if self.dbusep else 0) | (IEXI if self.iexi else 0)
                | DBGVER << 16 | self.dexccode << DEXC_SHIFT | (DINT if self.dint else 0)
                | (DBS if self.dbs else 0))

    def write(self, v: int) -> None:
        """IEXI R/W, CACHEEP / DBUSEP R/W1 (writing 1 sets them: the error
        is taken when IEXI clears); the rest read-only."""
        self.iexi = bool(v & IEXI)
        self.cacheep |= bool(v & CACHEEP)
        self.dbusep |= bool(v & DBUSEP)

    # -- the DCR --------------------------------------------------------------------------

    def read_dcr(self) -> int:
        v = self.dcr | EJ.DCR_IB | EJ.DCR_DB
        if self.m.nmi_pending:
            v |= EJ.DCR_NMIP
        if self.prob_en:
            v |= EJ.DCR_PE
        return v

    @property
    def inte(self) -> bool:
        return bool(self.dcr & EJ.DCR_INTE)

    @property
    def nmie(self) -> bool:
        return bool(self.dcr & EJ.DCR_NMIE)

    # -- drseg -----------------------------------------------------------------------------

    def register(self, off: int) -> int:
        """The drseg register at ``off`` (8-byte aligned; no side effects)."""
        if off == 0:
            return self.read_dcr()
        bp = self.bp
        if off == EJ.IBS:
            return int(bp.asid_sup) << 30 | bp.ib_count << 24 | self.ibs_bs
        if off == EJ.DBS:
            return int(bp.asid_sup) << 30 | bp.db_count << 24 | self.dbs_bs
        if TRACE_BASE <= off < TRACE_BASE + TRACE_SIZE:
            t = self.trace
            return t.get(off - TRACE_BASE) if t is not None else 0
        return bp.register(off)

    def set_register(self, off: int, v: int) -> None:
        v &= M64
        if off == 0:
            w = EJ.DCR_INTE | EJ.DCR_NMIE | EJ.DCR_SRE
            self.dcr = (self.dcr & ~w) | (v & w)
            return
        bp = self.bp
        if off == EJ.IBS:                       # (BS: R/W0, MD00047 Table 5.3)
            self.ibs_bs &= v | ~((1 << bp.ib_count) - 1)
            return
        if off == EJ.DBS:
            self.dbs_bs &= v | ~((1 << bp.db_count) - 1)
            return
        if TRACE_BASE <= off < TRACE_BASE + TRACE_SIZE:
            if self.trace is not None:
                self.trace.put(off - TRACE_BASE, v & 0xFFFFFFFF)
            return
        bp.set_register(off, v)

    def names(self) -> dict[int, str]:
        return self.bp.names()

    # -- dseg accesses ------------------------------------------------------------------------

    @staticmethod
    def covers(va: int) -> bool:
        return DSEG_BASE <= va <= DSEG_END

    def _dmseg(self, va: int, size: int) -> int:
        """The DBU RAM offset of a dmseg access; without the probe's ProbEn
        it waits for a probe forever (MD00047 2.2.2.1)."""
        if not self.prob_en:
            raise ProbeWaitError(
                f"access to dmseg 0x{va:016x} with no probe (the DBU's ProbEn = 0): it waits "
                "for a probe forever (MD00047 2.2.2.1)", name=None, word=None, addr=va)
        return va - DSEG_BASE

    def _trace_reg(self, off: int) -> bool:
        return TRACE_BASE <= off < TRACE_BASE + TRACE_SIZE

    def load(self, va: int, size: int) -> int:
        """A load from dseg (little-endian; drseg: the bytes of an 8-byte
        register — the trace encoder's are 4 bytes)."""
        if va >= DRSEG_BASE and self._trace_reg(va - DRSEG_BASE):
            off = va - DRSEG_BASE
            v = self.register(off & ~3) & 0xFFFFFFFF
            if size == 8:
                v |= (self.register((off & ~3) + 4) & 0xFFFFFFFF) << 32
                return v
            return (v >> 8 * (off & 3)) & ((1 << 8 * size) - 1)
        if va >= DRSEG_BASE:
            off = va - DRSEG_BASE
            return (self.register(off & ~7) >> 8 * (off & 7)) & ((1 << 8 * size) - 1)
        off = self._dmseg(va, size)
        return self.dbu.ram_read(off, size)

    def store(self, va: int, value: int, size: int) -> None:
        value &= (1 << 8 * size) - 1
        if va >= DRSEG_BASE and self._trace_reg(va - DRSEG_BASE):
            off = va - DRSEG_BASE
            if size == 8:
                self.set_register(off & ~3, value & 0xFFFFFFFF)
                self.set_register((off & ~3) + 4, value >> 32)
                return
            base, sh = off & ~3, 8 * (off & 3)
            mask = ((1 << 8 * size) - 1) << sh
            self.set_register(base, (self.register(base) & ~mask) | (value << sh))
            return
        if va >= DRSEG_BASE:
            off = va - DRSEG_BASE
            base, sh = off & ~7, 8 * (off & 7)
            mask = ((1 << 8 * size) - 1) << sh
            self.set_register(base, (self.register(base) & ~mask) | (value << sh))
            return
        off = self._dmseg(va, size)
        self.dbu.ram_write(off, value, size)

    def fetch_half(self, va: int) -> int:
        """A fetch from dseg (drseg: UNDEFINED — a simulator halt)."""
        if va >= DRSEG_BASE:
            from .exceptions import FetchError
            raise FetchError(f"instruction fetch from drseg (0x{va:016x}) is UNDEFINED "
                             "(MD00047 Table 2.4)", name=None, word=None, addr=va)
        off = self._dmseg(va, 2)
        return self.dbu.ram_read(off, 2)

    def peek(self, va: int, size: int) -> int | None:
        """What a load would read, for the views (None: nothing answers)."""
        if va >= DRSEG_BASE:
            return self.load(va, size)
        if not self.prob_en:
            return None
        return self.dbu.ram_read(va - DSEG_BASE, size)

    # -- the breakpoints (MD00047 5.3) ------------------------------------------------------

    def _asid(self) -> int:
        m = self.m
        return (m.csrs.satp >> (22 if m.xlen == 32 else 44)) & 0xFF

    def instruction(self, pc: int) -> None:
        """Before the instruction at ``pc`` (Non-Debug Mode): IB_match; a
        break enters Debug Mode before it executes."""
        from .simrv import CAUSE_TRIGGER, EnterDebug
        hits, brk = self.bp.instruction_match(pc & self.m.amask, self._asid(), self.index)
        if hits:
            self.ibs_bs |= hits
        if brk:
            raise EnterDebug(CAUSE_TRIGGER)

    def data(self, addr: int, size: int, store: bool, value: int | None = None) -> None:
        """An access's address (``value``: a store's data) — the precise
        breaks; a load's value compare waits for its data (:meth:`loaded`)."""
        from .simrv import CAUSE_TRIGGER, EnterDebug
        k = addr & 7
        lanes = (((1 << size) - 1) << k) & 0xFF
        precise, brk, vcmp = self.bp.data_match(addr & self.m.amask, lanes, 8, store,
                                                self._asid(), self.index)
        if precise:
            self.dbs_bs |= precise
        if vcmp:
            if store:
                data = (value or 0) << 8 * k
                hits = sum(1 << n for n in range(self.bp.db_count)
                           if vcmp >> n & 1 and self.bp.value_match(n, lanes, 8, data))
                if hits:
                    self.dbs_bs |= hits
                    brk = brk or any(self.bp.dbc[n] & EJ.BC_BE for n in range(self.bp.db_count)
                                     if hits >> n & 1)
            else:
                self._vc = (vcmp, lanes)
        if brk:
            self._vc = None
            raise EnterDebug(CAUSE_TRIGGER)

    def loaded(self, addr: int, size: int, value: int) -> None:
        """A load's data for the value compares it waits for: precise here
        (the register is not written yet)."""
        from .simrv import CAUSE_TRIGGER, EnterDebug
        vcmp, lanes = self._vc
        self._vc = None
        data = value << 8 * (addr & 7)
        hits = sum(1 << n for n in range(self.bp.db_count)
                   if vcmp >> n & 1 and self.bp.value_match(n, lanes, 8, data))
        if hits:
            self.dbs_bs |= hits
            if any(self.bp.dbc[n] & EJ.BC_BE for n in range(self.bp.db_count) if hits >> n & 1):
                raise EnterDebug(CAUSE_TRIGGER)

    # -- views -------------------------------------------------------------------------------

    def lines(self) -> list[str]:
        m = self.m
        out = [f"Hybrid Debug (mipsdebug 0x{self.read():08x}): "
               + ("Debug Mode" if m.debug_mode else "not in Debug Mode")
               + f", IEXI {int(self.iexi)}, DEXCCODE {self.dexccode}"
               + (", DINT" if self.dint else "") + (", DBS" if self.dbs else "")
               + (", CACHEEP" if self.cacheep else "") + (", DBUSEP" if self.dbusep else "")]
        out.append(f"  vector 0x{self.vector():x} ({'dmseg: the probe' if self.prob_trap else 'RESET_BASE + 0x480'}), "
                   f"DCR 0x{self.read_dcr():08x} (IntE {int(self.inte)}, NMIE {int(self.nmie)}, "
                   f"ProbEn {int(self.prob_en)})")
        bp = self.bp
        on = [n for n in range(bp.ib_count) if bp.ibc[n] & (EJ.BC_BE | EJ.BC_TE)]
        for n in on:
            c = bp.ibc[n]
            who = f" hart {c >> EJ.BC_TC_SHIFT & 0xFF}" if c & EJ.BC_TCUSE else ""
            out.append(f"  IB{n}: 0x{bp.iba[n]:x} mask 0x{bp.ibm[n]:x}{who}"
                       f"{' break' if c & EJ.BC_BE else ''}{' trigger' if c & EJ.BC_TE else ''}"
                       f"{' (hit)' if self.ibs_bs >> n & 1 else ''}")
        on = [n for n in range(bp.db_count) if bp.dbc[n] & (EJ.BC_BE | EJ.BC_TE)]
        for n in on:
            c = bp.dbc[n]
            kind = "store" if c & EJ.DBC_NOLB else "load" if c & EJ.DBC_NOSB else "load/store"
            who = f" hart {c >> EJ.BC_TC_SHIFT & 0xFF}" if c & EJ.BC_TCUSE else ""
            value = "" if (c >> EJ.DBC_BLM_SHIFT & 0xFF) == 0xFF else f" value 0x{bp.dbv[n]:x}"
            out.append(f"  DB{n}: 0x{bp.dba[n]:x} mask 0x{bp.dbm[n]:x} {kind}{value}{who}"
                       f"{' break' if c & EJ.BC_BE else ''}"
                       f"{' (hit)' if self.dbs_bs >> n & 1 else ''}")
        if len(out) == 2:
            out.append(f"  no hardware breakpoints ({bp.ib_count} instruction, "
                       f"{bp.db_count} data in the cluster)")
        return out
