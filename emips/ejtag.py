"""EJTAG debug support (PLAN M14): Debug Mode.

References: *MIPS EJTAG Specification* (MD00047, revision 5.06), chapter 2
"EJTAG Processor Core Extensions" and chapter 3 "Debug Control Register";
the 24K core's implementation (MD00343 chapter 11 "EJTAG Debug Support in
the 24K Core", EJTAG 3.20; Debug register Table 6.36, DCR Table 11.1,
exception priorities Table 5.1, vectors Tables 5.5 / 5.6).

* **Debug Mode** is entered only through a debug exception and left by
  DERET (or a reset).  It has the rights of kernel mode
  (:attr:`emips.sim.Machine.kernel_mode`), with interrupts and NMIs
  blocked (2.5), and the dseg segment.
* **Debug exceptions** (Table 2.8; this module's codes are
  :data:`emips.cp0.DSS` ... :data:`emips.cp0.DDBS`): Debug Single Step
  (``Debug.SSt``), Debug Interrupt (:meth:`Machine.debug_interrupt`, the
  DINT input), Debug Breakpoint (SDBBP).  DEPC = the restart PC (the
  branch for a delay slot, ``Debug.DBD``), the Debug cause bit, Halt /
  Doze, ``IEXI = DM = 1``, PC = the debug vector 0xBFC00480 (a CPS core:
  its reset base + 0x480, MD00343 Table 5.6).
* **Exceptions in Debug Mode** (2.4): only DEPC and Debug change (DBD,
  the cause bits cleared, ``DExcCode``, IEXI); the processor re-enters
  the debug vector and stays in Debug Mode.  Interrupts, NMI, single
  step and debug interrupts are blocked.
* **Imprecise errors** (2.4.2): with ``Debug.IEXI = 1`` a pending data
  bus error / cache error is deferred and shown in ``DBusEP`` /
  ``CacheEP`` (R/W1); taken when IEXI is cleared.  The 24K's IBusEP and
  MCheckP read 0 (its fetch bus errors and machine checks are precise).
* **dseg** (2.2.2, Table 2.2): 0xFF200000 - 0xFF3FFFFF in Debug Mode,
  unmapped and uncached: dmseg (the probe's memory) and drseg (the debug
  registers: the DCR at offset 0; word accesses only).  ``Debug.LSNM =
  1`` sends loads / stores there to kseg3 memory again.  dmseg accesses
  are processor accesses through the core's TAP (:mod:`emips.tap`): a
  probe serves them, without one they wait forever
  (:class:`ProbeWaitError`).
* The DCR (Chapter 3; 24K Table 11.1): IntE / NMIE mask interrupts / NMI
  in Non-Debug Mode, NMIpend shows a held NMI, ENM the endianness, IB /
  DB the hardware breakpoints, PCR the PC sampling rate, SRstE, ProbEn.
* A 64-bit (MIPS64) core (MD00047 2.2.2.2, 5.2, 7.2): every drseg register
  is 64 bits wide (IBA / IBM / DBA / DBM / DBV compare 64-bit addresses and
  data; no DBVH — 0x2124 is DBV's upper half), drseg takes doubleword
  transactions (other sizes UNDEFINED: eMIPS reads / writes those bytes of
  the doubleword), the data breakpoints see the 8 byte lanes of a
  doubleword bus, and PCsample holds a 64-bit PC (Figure 7.2).
"""

from __future__ import annotations

from . import cp0 as C
from .exceptions import FetchError, ProbeWaitError

M32 = 0xFFFFFFFF

DEBUG, DEPC, DESAVE = C.DEBUG, C.DEPC, C.DESAVE

# ------------------------------------------------------- Debug (CP0 23, 0)
DBD = 1 << 31
DM = 1 << 30
NODCR = 1 << 29
LSNM = 1 << 28
DOZE = 1 << 27
HALT = 1 << 26
COUNTDM = 1 << 25
IBUSEP = 1 << 24
MCHECKP = 1 << 23
CACHEEP = 1 << 22
DBUSEP = 1 << 21
IEXI = 1 << 20
DDBSIMPR = 1 << 19
DDBLIMPR = 1 << 18
EJTAGVER_SHIFT = 15
DEXC_SHIFT = 10
NOSST = 1 << 9
SST = 1 << 8
DINT = 1 << 5
DIB = 1 << 4
DDBS = 1 << 3
DDBL = 1 << 2
DBP = 1 << 1
DSS = 1 << 0
#: the bits a debug exception sets one of and an exception in Debug Mode
#: clears (2.3.4 / 2.4.3)
CAUSE_BITS = DDBSIMPR | DDBLIMPR | DINT | DIB | DDBS | DDBL | DBP | DSS
#: the pending imprecise errors software may set by writing 1 (the 24K's)
W1_BITS = CACHEEP | DBUSEP

#: the cause bits by name (the views)
CAUSE_NAMES = ((DSS, "DSS"), (DBP, "DBp"), (DDBL, "DDBL"), (DDBS, "DDBS"), (DIB, "DIB"),
               (DINT, "DINT"), (DDBLIMPR, "DDBLImpr"), (DDBSIMPR, "DDBSImpr"))
#: the Debug bit of each debug exception
EXC_BIT = {C.DSS: DSS, C.DINT: DINT, C.DDBLIMPR: DDBLIMPR, C.DDBSIMPR: DDBSIMPR,
           C.DIB: DIB, C.DBP: DBP, C.DDBL: DDBL, C.DDBS: DDBS}
#: DExcCode of an SDBBP in Debug Mode (Bp) and of a cache error (CacheErr,
#: MD00343 Table 6.24)
DEXC_BP, DEXC_CACHEERR = 9, 30

# ----------------------------------------------------------------- dseg
DSEG_BASE = 0xFF200000
DRSEG_BASE = 0xFF300000
DSEG_END = 0xFF3FFFFF
#: the debug vector, from the BEV base (0xBFC00200 + 0x280 = 0xBFC00480;
#: a CPS core: SI_ExceptionBase + 0x480)
DEBUG_VECTOR_OFFSET = 0x280
#: the debug vector when the probe provides the handler (ECR.ProbTrap = 1)
PROBE_VECTOR = 0xFF200200

# ------------------------------------------------------------- the DCR
DCR_ENM = 1 << 29
DCR_DB = 1 << 17
DCR_IB = 1 << 16
DCR_IVM = 1 << 15
DCR_DVM = 1 << 14
DCR_CBT = 1 << 10
DCR_PCS = 1 << 9
DCR_PCR_SHIFT = 6
DCR_PCSE = 1 << 5
DCR_INTE = 1 << 4
DCR_NMIE = 1 << 3
DCR_NMIP = 1 << 2
DCR_SRE = 1 << 1
DCR_PE = 1 << 0
#: the bits software writes (24K Table 11.1: PCR, INTE, NMIE, SRE)
DCR_WRITE = (7 << DCR_PCR_SHIFT) | DCR_INTE | DCR_NMIE | DCR_SRE

# ------------------------------------------------- hardware breakpoints
#: the 24K's 4 instruction and 2 data breakpoints (MD00343 11.2)
IB_COUNT, DB_COUNT = 4, 2
IBS, DBS = 0x1000, 0x2000
#: register offsets in a breakpoint's 0x100 block (Tables 11.7 / 11.13)
IBA, IBM, IBASID, IBC = 0x00, 0x08, 0x10, 0x18
DBA, DBM, DBASID, DBC, DBV, DBVH = 0x00, 0x08, 0x10, 0x18, 0x20, 0x24
#: IBCn / DBCn bits
BC_BE = 1 << 0
BC_TE = 1 << 2
BC_ASIDUSE = 1 << 23
DBC_NOLB = 1 << 12
DBC_NOSB = 1 << 13
DBC_BAI_SHIFT = 14
DBC_BLM_SHIFT = 4
#: the writable bits (24K Tables 11.12 / 11.18; ASIDuse only with a TLB)
IBC_WRITE = BC_TE | BC_BE
DBC_WRITE = (0xFF << DBC_BAI_SHIFT) | DBC_NOSB | DBC_NOLB | (0xFF << DBC_BLM_SHIFT) \
    | BC_TE | BC_BE
#: with the MT Module (the 34K, MD00534 Tables 11.12 / 11.18; MD00047 5.3):
#: the TC to match (31:24) and TCuse
BC_TCUSE = 1 << 22
BC_TC_SHIFT = 24
BC_TC_WRITE = (0xFF << BC_TC_SHIFT) | BC_TCUSE


def _ib_names() -> dict[int, str]:
    out = {IBS: "IBS"}
    for n in range(IB_COUNT):
        for off, name in ((IBA, "IBA"), (IBM, "IBM"), (IBASID, "IBASID"), (IBC, "IBC")):
            out[0x1100 + 0x100 * n + off] = f"{name}{n}"
    out[DBS] = "DBS"
    for n in range(DB_COUNT):
        for off, name in ((DBA, "DBA"), (DBM, "DBM"), (DBASID, "DBASID"), (DBC, "DBC"),
                          (DBV, "DBV"), (DBVH, "DBVH")):
            out[0x2100 + 0x100 * n + off] = f"{name}{n}"
    return out


#: drseg register names by offset (the views, the debugger)
DRSEG_NAMES = {0x0000: "DCR", **_ib_names()}


def in_dseg(va: int) -> bool:
    return DSEG_BASE <= va <= DSEG_END


def segment(va: int) -> str:
    return "drseg" if va >= DRSEG_BASE else "dmseg"


class EJTAG:
    """A core's EJTAG unit: the memory-mapped debug registers of drseg
    and the dmseg window to the probe (none connected: ProbEn = 0)."""

    def __init__(self, m) -> None:
        self.m = m
        #: a 64-bit core: 64-bit registers, doubleword drseg transactions
        self.wide = bool(getattr(m, "wide", False))
        self.mask = (1 << 64) - 1 if self.wide else M32
        self.bus = 8 if self.wide else 4
        self.reset()

    def reset(self) -> None:
        """Reset values (24K Table 11.1): PCR = 7, IntE = NMIE = SRstE = 1,
        both breakpoint kinds implemented (IB, DB), PC sampling (PCS).  The
        breakpoints: TE = BE = 0; the rest is UNDEFINED at reset (0 here)."""
        self.dcr = (7 << DCR_PCR_SHIFT) | DCR_INTE | DCR_NMIE | DCR_SRE | DCR_IB | DCR_DB \
            | DCR_PCS
        #: the core's TAP (emips.tap.TAP; set by TAP.processor_reset), or None
        self.tap = None
        #: PC sampling (MD00047 ch. 7, 24K 11.6): the PCsample register
        #: (New, PC 32:1, ASID 40:33; MIPS64: PC 64:1, ASID 72:65), the
        #: cycle counter, a sample due
        self.pc_sample = 0
        self.pc_count = 0
        self.sample_due = False
        self.ibs_bs = 0
        self.iba = [0] * IB_COUNT
        self.ibm = [0] * IB_COUNT
        self.ibasid = [0] * IB_COUNT
        self.ibc = [0] * IB_COUNT
        self.dbs_bs = 0
        self.dba = [0] * DB_COUNT
        self.dbm = [0] * DB_COUNT
        self.dbasid = [0] * DB_COUNT
        self.dbc = [0] * DB_COUNT
        self.dbv = [0] * DB_COUNT
        self.dbvh = [0] * DB_COUNT

    @property
    def asid_sup(self) -> bool:
        """ASID compares (IBS / DBS.ASIDsup): with a TLB MMU only (24K
        Tables 11.8 / 11.14: "Fixed MMU - 0, TLB - 1")."""
        return getattr(self.m.mmu, "mt", 3) in (1, 4)

    @property
    def ib_armed(self) -> bool:
        return any(c & (BC_BE | BC_TE) for c in self.ibc)

    @property
    def db_armed(self) -> bool:
        return any(c & (BC_BE | BC_TE) for c in self.dbc)

    # -- the DCR -------------------------------------------------------------

    @property
    def prob_en(self) -> bool:
        return self.tap is not None and self.tap.prob_en

    @property
    def prob_trap(self) -> bool:
        return self.tap is not None and self.tap.prob_trap

    # -- PC sampling -----------------------------------------------------------

    @property
    def sample_period(self) -> int:
        """2^(5 + DCR.PCR) cycles (32 ... 4096)."""
        return 1 << (5 + ((self.dcr >> DCR_PCR_SHIFT) & 7))

    def tick(self, cycles: int) -> None:
        """Cycles pass: a sample is due every :attr:`sample_period` (the
        24K samples always, 11.6; PCSe reads 0 there)."""
        self.pc_count += cycles
        if self.pc_count >= self.sample_period:
            self.pc_count %= self.sample_period
            self.sample_due = True

    def sample(self, pc: int) -> None:
        """The graduating instruction at ``pc`` is sampled: New = 1; with
        the MT Module its TC above the ASID (MD00047 Figures 7.1 / 7.2:
        48:41, MIPS64 80:73 — the 34K always includes it, MD00534 11.6)."""
        self.sample_due = False
        m = self.m
        asid = m.cp0[(10, 0)]["value"] & 0xFF
        top = 1 + 8 * self.bus
        self.pc_sample = 1 | ((pc & self.mask) << 1) | (asid << top) | (
            m._tc_id() << (top + 8) if m.mt is not None else 0)

    def read_dcr(self) -> int:
        """The DCR as software reads it: ENM from the core's byte order,
        NMIpend = an NMI held, ProbEn from the probe."""
        value = self.dcr
        if self.m.big_endian:
            value |= DCR_ENM
        if self.m.nmi_pending:
            value |= DCR_NMIP
        if self.prob_en:
            value |= DCR_PE
        return value

    @property
    def int_enabled(self) -> bool:
        return bool(self.dcr & DCR_INTE)

    @property
    def nmi_enabled(self) -> bool:
        return bool(self.dcr & DCR_NMIE)

    # -- drseg ---------------------------------------------------------------

    def register(self, offset: int) -> int:
        """The drseg register at ``offset`` (no side effects); an
        unimplemented one reads 0 (UNPREDICTABLE, 2.2.2.2)."""
        if offset == 0x0000:
            return self.read_dcr()
        sup = self.asid_sup
        if offset == IBS:
            return (int(sup) << 30) | (IB_COUNT << 24) | self.ibs_bs
        if offset == DBS:
            return (int(sup) << 30) | (DB_COUNT << 24) | self.dbs_bs
        n, r = (offset >> 8) & 0xF, offset & 0xFF
        if offset & 0xF000 == 0x1000 and 1 <= n <= IB_COUNT:
            n -= 1
            regs = {IBA: self.iba, IBM: self.ibm, IBC: self.ibc,
                    IBASID: self.ibasid if sup else None}
            if r in regs:
                return regs[r][n] if regs[r] is not None else 0
        if offset & 0xF000 == 0x2000 and 1 <= n <= DB_COUNT:
            n -= 1
            regs = {DBA: self.dba, DBM: self.dbm, DBC: self.dbc, DBV: self.dbv,
                    DBVH: self.dbvh, DBASID: self.dbasid if sup else None}
            if r in regs:
                return regs[r][n] if regs[r] is not None else 0
        return 0

    def set_register(self, offset: int, value: int) -> None:
        """A write to drseg; unimplemented registers ignore it."""
        value &= self.mask
        if offset == 0x0000:
            if (self.dcr ^ value) & (7 << DCR_PCR_SHIFT):
                self.pc_count = 0               # a new rate restarts the count (7.2)
            self.dcr = (self.dcr & ~DCR_WRITE) | (value & DCR_WRITE)
            return
        sup = self.asid_sup
        asid = (BC_ASIDUSE if sup else 0) | (BC_TC_WRITE if self.m.mt is not None else 0)
        if offset == IBS:                     # BS: R/W on the 24K (Table 11.8)
            self.ibs_bs = value & ((1 << IB_COUNT) - 1)
        elif offset == DBS:                     # BS: R/W0 (Table 11.14)
            self.dbs_bs &= value | ~((1 << DB_COUNT) - 1)
        else:
            n, r = (offset >> 8) & 0xF, offset & 0xFF
            if offset & 0xF000 == 0x1000 and 1 <= n <= IB_COUNT:
                n -= 1
                if r == IBA:
                    self.iba[n] = value
                elif r == IBM:
                    self.ibm[n] = value
                elif r == IBASID and sup:
                    self.ibasid[n] = value & 0xFF
                elif r == IBC:
                    self.ibc[n] = value & (IBC_WRITE | asid)
            elif offset & 0xF000 == 0x2000 and 1 <= n <= DB_COUNT:
                n -= 1
                if r == DBA:
                    self.dba[n] = value
                elif r == DBM:
                    self.dbm[n] = value
                elif r == DBASID and sup:
                    self.dbasid[n] = value & 0xFF
                elif r == DBC:
                    self.dbc[n] = value & (DBC_WRITE | asid)
                elif r == DBV:
                    self.dbv[n] = value
                elif r == DBVH:
                    self.dbvh[n] = value
        self.m.debug_hw_changed()

    # -- matching (MD00047 5.3; MD00343 11.2.5) ------------------------------------

    def instruction_match(self, pc: int, asid: int, tc: int = 0) -> tuple[int, bool]:
        """IB_match of every enabled instruction breakpoint for the
        instruction at ``pc`` (of thread context ``tc``: the MT Module's
        TCuse): (the matching ones as a BS mask, does one of them break
        (BE)).  No MIPS16e: ISAMode is 0, so IBA / IBM bit 0 (ISA / ISAM)
        take part as an ordinary address bit."""
        hits, brk = 0, False
        for n in range(IB_COUNT):
            c = self.ibc[n]
            if not c & (BC_BE | BC_TE):
                continue
            if c & BC_ASIDUSE and asid != self.ibasid[n]:
                continue
            if c & BC_TCUSE and tc != c >> BC_TC_SHIFT & 0xFF:
                continue
            if (self.ibm[n] | ~(pc ^ self.iba[n])) & self.mask != self.mask:
                continue
            hits |= 1 << n
            brk = brk or bool(c & BC_BE)
        return hits, brk

    def data_match(self, addr: int, lanes: int, width: int, store: bool,
                   asid: int, tc: int = 0) -> tuple[int, bool, int]:
        """DB_addr_match of every enabled data breakpoint for an access (of
        thread context ``tc``): (the precise ones — no value compare — as a
        BS mask, does one of them break, the ones that still need the data
        value)."""
        full = (1 << width) - 1
        precise, brk, value = 0, False, 0
        for n in range(DB_COUNT):
            c = self.dbc[n]
            if not c & (BC_BE | BC_TE) or c & (DBC_NOSB if store else DBC_NOLB):
                continue
            if c & BC_ASIDUSE and asid != self.dbasid[n]:
                continue
            if c & BC_TCUSE and tc != c >> BC_TC_SHIFT & 0xFF:
                continue
            if (self.dbm[n] | ~(addr ^ self.dba[n])) & self.mask != self.mask:
                continue
            bai = (c >> DBC_BAI_SHIFT) & full
            if not ~bai & lanes:
                continue
            blm = (c >> DBC_BLM_SHIFT) & full
            if (blm | bai | ~lanes) & full == full:   # DB_no_value_compare
                precise |= 1 << n
                brk = brk or bool(c & BC_BE)
            else:
                value |= 1 << n
        return precise, brk, value

    def value_match(self, n: int, lanes: int, width: int, data: int) -> bool:
        """DB_value_match of breakpoint ``n``: every compared byte lane of
        the data bus equals DBVn (DBVHn: bits 63:32 of a doubleword)."""
        c = self.dbc[n]
        dbv = self.dbv[n] | (self.dbvh[n] << 32)
        skip = (c >> DBC_BAI_SHIFT) | (c >> DBC_BLM_SHIFT)
        for k in range(width):
            if lanes >> k & 1 and not skip >> k & 1 and \
                    (data >> 8 * k) & 0xFF != (dbv >> 8 * k) & 0xFF:
                return False
        return True

    def _lanes(self, va: int, size: int) -> int:
        """Shift of the low byte of a ``size``-byte access at ``va`` in its
        word (a 64-bit core: doubleword) in the core's byte order."""
        bus = self.bus
        k = va & (bus - 1)
        return 8 * (bus - size - k) if self.m.big_endian else 8 * k

    def _reg_offset(self, va: int) -> int:
        """The drseg register an access at ``va`` reaches (word / doubleword
        aligned)."""
        return (va - DRSEG_BASE) & 0xFFFFF & ~(self.bus - 1)

    # -- accesses (Debug Mode, dseg) --------------------------------------------

    def load(self, va: int, size: int) -> int:
        """A load from dseg (``Debug.LSNM = 0``).  drseg: only word
        transactions are allowed — doubleword on a 64-bit core (other sizes
        are UNDEFINED; eMIPS reads those bytes of the register)."""
        if va >= DRSEG_BASE:
            word = self.register(self._reg_offset(va))
            return (word >> self._lanes(va, size)) & ((1 << 8 * size) - 1)
        return self._probe("load", va, size)

    def store(self, va: int, value: int, size: int) -> None:
        if va >= DRSEG_BASE:
            off = self._reg_offset(va)
            shift = self._lanes(va, size)
            mask = ((1 << 8 * size) - 1) << shift
            old = self.register(off)
            self.set_register(off, (old & ~mask) | ((value << shift) & mask))
            return
        self._probe("store", va, size, value)

    def fetch(self, va: int) -> int:
        """An instruction fetch from dseg (always dseg in Debug Mode; a
        fetch from drseg is UNDEFINED: a simulator halt here)."""
        if va >= DRSEG_BASE:
            raise FetchError(f"instruction fetch from drseg (0x{va:08x}) is UNDEFINED "
                             "(MD00047 Table 2.4)", name=None, word=None, addr=va)
        return self._probe("fetch", va, 4)

    def peek(self, va: int, size: int) -> int | None:
        """What a load would read, for the views (no side effects): drseg's
        registers; dmseg from a connected probe's memory (None without
        one: nothing would answer)."""
        if va >= DRSEG_BASE:
            word = self.register(self._reg_offset(va))
            return (word >> self._lanes(va, size)) & ((1 << 8 * size) - 1)
        tap = self.tap
        if tap is None or not tap.prob_en or tap.probe is None:
            return None
        mem = tap.probe.mem
        data = [mem.get(va - DSEG_BASE + i, 0) for i in range(size)]
        return int.from_bytes(bytes(data), "big" if self.m.big_endian else "little")

    def _probe(self, kind: str, va: int, size: int, value: int = 0) -> int:
        """A processor access to dmseg through the TAP (emips.tap)."""
        if self.tap is None:
            what = {"fetch": "fetch from", "load": "load from", "store": "store to"}[kind]
            raise ProbeWaitError(
                f"{what} dmseg 0x{va:08x} with no EJTAG probe (DCR.ProbEn = 0): the access "
                "waits for a probe forever (MD00047 2.2.2.1)", name=None, word=None, addr=va)
        return self.tap.processor_access(kind, va, size, value)
