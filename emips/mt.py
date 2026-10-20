"""The MIPS MT Module: VPEs and thread contexts (PLAN M19).

Sources: *MIPS Architecture for Programmers Volume IV-f: The MIPS MT
Module for the MIPS32 Architecture* (MD00378, revision 1.12) — chapter 3
(thread contexts, VPEs), 4 (the exception model), 5 (DMT, EMT, DVPE,
EVPE, FORK, YIELD, MFTR, MTTR), 6 (the registers), 7 (WAIT, LL / SC),
8 (multiple VPEs), 10 (EJTAG); the 34K Software User's Manual (MD00534
rev. 1.13) for what an implementation chooses: chapter 7 (reset values,
the 16 YQMask bits, SRSConf0 without SRSConf1, the per-TC Status /
EntryHi / UserLocal / Debug fields, TLB writes that would duplicate an
entry dropped without a Machine Check) and chapter 13 (the basic
round-robin policy manager).

The model
---------

A *processor* (:class:`Processor`) holds 1-2 VPEs and 1-9 thread
contexts (TCs), as the 34K (MVPConf0.PVPE, PTC).  Each **VPE is an
:class:`emips.sim.Machine`** with its own CP0 (its own EBase.CPUNum,
Status, Cause, TLB, Count / Compare, interrupt lines ...); the VPEs of a
processor share the physical memory and the caches.  Each **TC** owns
what MD00378 3.1 lists: the GPRs, HI / LO, the DSP accumulators and
DSPControl, the PC (with its delay-slot / ISA-mode state), the LLbit,
the per-TC fields of Status (CU3..CU0, MX, KSU), EntryHi.ASID,
UserLocal, Debug.SSt / OffLine, and the TC registers (TCStatus, TCBind,
TCRestart, TCHalt, TCContext ...).

The TC that issues is bound to its VPE's Machine (:meth:`Processor.bind`
swaps the per-TC state in and out — architecturally invisible), so the
Machine's executors run unchanged on the right thread.  One cycle
(:meth:`Processor.cycle`) issues one instruction of one TC, the 34K's
fine-grained dispatch with the basic round-robin policy manager (MD00534
13.2.1: every TC at one priority, round-robin among the runnable ones);
every VPE's clock advances one cycle.

A TC issues when (MD00378 3.6, 4.1, 6.2, 6.5, 6.6, 6.11, 10.3):
it is activated (TCStatus.A) and not halted (TCHalt.H) or off-line
(Debug.OffLine); multi-VPE execution is enabled (MVPControl.EVP) and its
VPE is activated (VPEConf0.VPA) — or EVP = 0 and it is the one TC that
cleared EVP (TC 0 after reset); its VPE is not restricted to one TC
(Status.EXL / ERL, Debug Mode, VPEControl.TE = 0), or it is that VPE's
VPEConf0.XTC; it is not blocked (WAIT: until an interrupt its VPE takes
on a non-IXMT TC; YIELD: until a YIELD qualifier input it waits on is
set).  While a VPE is in Debug Mode only its TC runs (10.3).

What eMIPS chooses where the manuals leave it open is listed in
``sim.DEVIATIONS["MT Module"]``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

from . import cp0 as C
from .exceptions import Blocked

M32 = 0xFFFFFFFF

# ----------------------------------------------------------------- registers
MVPCONTROL, MVPCONF0, MVPCONF1 = (0, 1), (0, 2), (0, 3)
VPECONTROL, VPECONF0, VPECONF1 = (1, 1), (1, 2), (1, 3)
YQMASK, VPESCHEDULE, VPESCHEFBACK, VPEOPT = (1, 4), (1, 5), (1, 6), (1, 7)
TCSTATUS, TCBIND, TCRESTART, TCHALT = (2, 1), (2, 2), (2, 3), (2, 4)
TCCONTEXT, TCSCHEDULE, TCSCHEFBACK = (2, 5), (2, 6), (2, 7)
SRSCONF0 = (6, 1)

#: the MT registers (the Machine sends MFC0 / MTC0 of these here)
REGISTERS = frozenset({MVPCONTROL, MVPCONF0, MVPCONF1, VPECONTROL, VPECONF0,
                       VPECONF1, YQMASK, VPESCHEDULE, VPESCHEFBACK, VPEOPT,
                       TCSTATUS, TCBIND, TCRESTART, TCHALT, TCCONTEXT,
                       TCSCHEDULE, TCSCHEFBACK, SRSCONF0})
#: their names, for the views (the CP0 table spells two of them wrong)
NAMES = {MVPCONTROL: "MVPControl", MVPCONF0: "MVPConf0", MVPCONF1: "MVPConf1",
         VPECONTROL: "VPEControl", VPECONF0: "VPEConf0", VPECONF1: "VPEConf1",
         YQMASK: "YQMask", VPESCHEDULE: "VPESchedule", VPESCHEFBACK: "VPEScheFBack",
         VPEOPT: "VPEOpt", TCSTATUS: "TCStatus", TCBIND: "TCBind",
         TCRESTART: "TCRestart", TCHALT: "TCHalt", TCCONTEXT: "TCContext",
         TCSCHEDULE: "TCSchedule", TCSCHEFBACK: "TCScheFBack", SRSCONF0: "SRSConf0"}

# MVPControl (MD00534 Table 7.4)
EVP, VPC, STLB, CPA = 1 << 0, 1 << 1, 1 << 2, 1 << 3
# MVPConf0 (Table 7.5)
MVPCONF0_M, TLBS, GS, PCP, TCA = 1 << 31, 1 << 29, 1 << 28, 1 << 27, 1 << 15
PVPE_SHIFT = 10
# MVPConf1 (Table 7.6)
C1F = 1 << 30
# VPEControl (Table 7.8)
YSI, GSI, TE = 1 << 21, 1 << 20, 1 << 15
EXCPT_SHIFT = 16
# VPEConf0 (Table 7.9)
VPECONF0_M, MVP, VPA = 1 << 31, 1 << 1, 1 << 0
XTC_SHIFT = 21
ICS, DCS, SCS, TCS = 1 << 16, 1 << 17, 1 << 18, 1 << 19
# TCStatus (Table 7.15)
TMX, TDS, DT, DA, A, IXMT = 1 << 27, 1 << 21, 1 << 20, 1 << 15, 1 << 13, 1 << 10
RNST_SHIFT, TKSU_SHIFT = 23, 11
# TCBind (Table 7.16)
CURTC_SHIFT, TBE = 21, 1 << 17
#: Config7.WII (MD00534 Table 7.47): Wait IE / IXMT Ignore, read-only 1
CONFIG7, CFG7_WII = (16, 7), 1 << 31
# Debug: the per-TC bits (MD00378 10.2; the 34K's OffLine is bit 7)
DEBUG_SST, DEBUG_OFFLINE = 1 << 8, 1 << 7

#: Thread exception sub-codes (VPEControl.EXCPT, MD00378 Table 4.2)
UNDERFLOW, OVERFLOW, INVALID_YQ, GATING, YIELD_SCHED, GS_SCHED = range(6)
EXCPT_NAMES = {UNDERFLOW: "Thread Underflow", OVERFLOW: "Thread Overflow",
               INVALID_YQ: "Invalid YIELD Qualifier", GATING: "Gating Storage",
               YIELD_SCHED: "YIELD Scheduler", GS_SCHED: "GS Scheduler"}
#: the 34K's YIELD qualifier inputs (MD00534 7.2.8: 16 mask bits)
YQ_BITS = 0xFFFF
#: SRSConf0's programmable, unassigned field value (Table 7.24)
SRS_INVALID = 0x3FE
#: the basic round-robin policy manager implements no scheduling
#: register: reads -1, writes ignored (MD00534 13.2.1)
NO_PM = frozenset({VPESCHEDULE, VPESCHEFBACK, TCSCHEDULE, TCSCHEFBACK})
#: the 34K's reference policy managers (MD00534 13.2)
POLICY_MANAGERS = ("rr", "wrr", "wrr2")
#: what each implements (Table 13.1 - 13.4): TCSchedule's writable bits
#: (WRR: STP, GRP; WRR2: also the throttles), VPEScheFBack (WRR2 only)
TCSCHEDULE_MASK = {"wrr": 0x0000000B, "wrr2": 0x00FFFFFB}
#: VPESchedule.GPO
GPO = 1 << 5
#: TCSchedule fields
STP = 1 << 3
#: the PM inputs a throttle's queue enable bit selects (bit 0 = QE[4] /
#: QE[14]: PM_wbb_avail, then ldq, fsb, sys)
PM_INPUTS = ("wbb", "ldq", "fsb", "sys")
#: the 34K's buffer sizes the PM sees (VPEScheFBack: SYSS, WBBS, FSBS,
#: LDQS; MD00534 9.4: 4 WBB entries; the 24K model's 8-line FSB and
#: 9-entry LDQ; no system buffer)
PM_SIZES = {"sys": 0, "wbb": 4, "fsb": 8, "ldq": 9}

#: Table 13.5: rotation count (1-15) -> the priority of group 0, 1, 2, 3
ROTATION = {
    1: (0, 1, 2, 3), 2: (1, 0, 3, 2), 3: (0, 1, 2, 3), 4: (1, 3, 0, 2),
    5: (0, 1, 2, 3), 6: (1, 0, 3, 2), 7: (0, 2, 1, 3), 8: (3, 1, 2, 0),
    9: (0, 2, 1, 3), 10: (1, 0, 3, 2), 11: (0, 1, 2, 3), 12: (1, 3, 0, 2),
    13: (0, 1, 2, 3), 14: (1, 0, 3, 2), 15: (0, 2, 1, 3),
}

#: Status / EntryHi fields instantiated per TC (MD00378 3.1: CU3..CU0,
#: MX, KSU; EntryHi.ASID)
STATUS_TC = 0xF0000000 | C.ST_MX | (3 << C.ST_KSU_SHIFT)
ENTRYHI = (10, 0)
ASID = 0xFF

#: the Machine attributes a TC owns (its program counter with the delay
#: slot / forbidden slot / ISA-mode state, HI / LO, DSPControl, WAIT,
#: the LLbit, single step)
_TC_ATTRS = ("pc", "hi", "lo", "dspctl", "slot_armed", "slot_target", "_forbidden",
             "_redirect", "isa_mode", "_m16", "_mm", "_m16_base", "_insn_raw",
             "_slot_raw", "_slot_back", "waiting", "wait_pc", "slept", "lr_addr",
             "lr_pa", "lr_valid", "_linked", "_sst", "ss_pending")
_FRESH = {"pc": 0, "hi": 0, "lo": 0, "dspctl": 0, "slot_armed": False,
          "slot_target": None, "_forbidden": False, "_redirect": None, "isa_mode": 0,
          "_m16": None, "_mm": None, "_m16_base": 0, "_insn_raw": None,
          "_slot_raw": None, "_slot_back": 4, "waiting": False, "wait_pc": 0,
          "slept": False, "lr_addr": None, "lr_pa": None, "lr_valid": False,
          "_linked": False, "_sst": False, "ss_pending": False}


def parse_spec(spec: str) -> tuple[int, int] | None:
    """``"none"`` / ``""`` -> None; ``"VPES:TCS"`` -> (VPEs, TCs): 1-2
    VPEs and VPEs..9 TCs, as the 34K (MVPConf0.PVPE 0-1, PTC 0-8)."""
    spec = (spec or "").strip().lower()
    if spec in ("", "none", "off", "no"):
        return None
    parts = spec.split(":")
    try:
        vpes = int(parts[0])
        tcs = int(parts[1]) if len(parts) > 1 else vpes
    except ValueError:
        raise ValueError(f"MT spec {spec!r}: VPES:TCS, e.g. 2:5") from None
    if len(parts) > 2 or not 1 <= vpes <= 2 or not vpes <= tcs <= 9:
        raise ValueError(f"MT spec {spec!r}: 1-2 VPEs and VPEs..9 TCs (the 34K)")
    return vpes, tcs


class _FPUContext:
    """An FPU context no VPE has bound (VPEConf1.NCP1 = 0 on both)."""

    def __init__(self, fpr: list[int], fcsr: int) -> None:
        self.fpr, self.fcsr = fpr, fcsr


class TC:
    """A thread context: its GPRs, its share of the Machine's state while
    another TC is bound, and its TC registers."""

    def __init__(self, n: int, vpe: int) -> None:
        self.n = n
        #: TCBind.CurVPE
        self.vpe = vpe
        self.gprs: list[int] = [0] * 32
        self.dsp_hi: list[int] = [0] * 4
        self.dsp_lo: list[int] = [0] * 4
        #: the Machine attributes (_TC_ATTRS) while not bound
        self.ctx: dict[str, Any] = dict(_FRESH)
        #: Status CU3..CU0 / MX / KSU, EntryHi.ASID, UserLocal, Debug SSt /
        #: OffLine while not bound
        self.status_bits = 0
        self.asid = 0
        self.userlocal = 0
        self.debug_bits = 0
        #: TCStatus fields of the TC itself
        self.active = n == 0                    # A: 1 for TC 0 (Table 7.15)
        self.da = False
        self.dt = False
        self.ixmt = False
        #: TCHalt.H: 0 for TC 0, 1 for the others (Table 7.18)
        self.halted = n != 0
        self.tbe = False
        self.context = 0                        # TCContext
        #: blocked on a YIELD: the qualifier bits it waits for (0: not)
        self.yield_wait = 0
        #: blocked on gating storage: the access (emips.itc), else None
        self.gs_wait = None
        #: RNST kept while halted: why it was blocked (MD00534 7.2.16)
        self.halted_rnst = 0
        #: TCStatus was written by the instruction issuing (DT as written)
        self.dt_written = False
        #: TCSchedule (WRR / WRR2: STP, GRP, the throttles) and
        #: TCScheFBack (the instructions it completed, saturating)
        self.schedule = 0
        self.fback = 0

    def __repr__(self) -> str:
        return f"TC{self.n}"


class Processor:
    """The MT processor: VPE Machines + TCs + the MVP registers and the
    dispatcher.  ``vpes``: the Machines (built with ``mt=True``; VPE n's
    EBase.CPUNum = n), ``tcs``: the number of TCs, ``vpe0_max``: the last
    TC bound to VPE 0 at reset (SI_Vpe0MaxTC; default: all but one TC
    per further VPE)."""

    def __init__(self, vpes: list, tcs: int, vpe0_max: int | None = None,
                 pm: str = "rr", itc: Any = None) -> None:
        if not 1 <= len(vpes) <= 2 or not len(vpes) <= tcs <= 9:
            raise ValueError("1-2 VPEs and VPEs..9 TCs (the 34K)")
        if pm not in POLICY_MANAGERS:
            raise ValueError(f"policy manager {pm!r}: one of {', '.join(POLICY_MANAGERS)}")
        self.vpes = vpes
        #: the policy manager (MD00534 13.2): "rr", "wrr", "wrr2"
        self.pm = pm
        #: the gating storage: the 34K's ITC (emips.itc; None: MVPConf0.GS = 0);
        #: an ITC object: one the cores of a CPS share (the 1004K's)
        from .itc import ITC
        self.itc = ITC(*itc) if isinstance(itc, tuple) else itc
        #: the VPE whose TC issues in the current cycle (an error's VPE)
        self.last_vp = 0
        #: VPESchedule.GPO (one per processor, reset 1: fixed priorities)
        self.gpo = True
        #: the group rotation (13.2.7): the count (1-15) of the next
        #: rotation, the priorities of the groups now, cycles it is held
        self.priority = ROTATION[1]
        self._hold: int | None = None
        self.rotation = 2
        #: per group: the TC that issued last (the round-robin inside it)
        self.group_last: list[TC | None] = [None] * 4
        #: the policy manager's resource inputs PM_sys_avail (set from the
        #: debugger; the others come from the model: queue_avail())
        self.sys_avail = 7
        #: the 34K pipeline's LDQ / FSB availability when it runs (None:
        #: the instruction-level model has no load queue)
        self.pipeline_avail: Callable[[], dict[str, int]] | None = None
        #: the FPU context (MVPConf1.PCP1): which VPE has it (VPEConf1.NCP1)
        self.fpu = any(m.has_fpu for m in vpes)
        self.fpu_owner = next((i for i, m in enumerate(vpes) if m.has_fpu), None)
        self._fpu_store = None
        self._own_tlbs: list = []
        if vpe0_max is None:
            vpe0_max = tcs - len(vpes)
        self.tcs = [TC(i, 0 if i <= vpe0_max else 1) for i in range(tcs)]
        #: the TC bound to each VPE's Machine
        self.current: list[TC] = []
        #: MVPControl (EVP, VPC, STLB, CPA)
        self.mvpcontrol = 0
        #: the TC allowed to run while EVP = 0 (reset: TC 0; then the one
        #: whose DVPE / MTC0 cleared EVP)
        self.solo: TC | None = self.tcs[0]
        #: per VPE: VPEControl (YSI, GSI, EXCPT, TE, TargTC), VPEConf0 MVP /
        #: VPA / XTC, YQMask, VPEOpt, SRSConf0's three fields
        self.vpecontrol = [0] * len(vpes)
        self.mvp = [i == 0 for i in range(len(vpes))]
        self.vpa = [i == 0 for i in range(len(vpes))]
        self.xtc = [next(t.n for t in self.tcs if t.vpe == v) for v in range(len(vpes))]
        self.yqmask = [0] * len(vpes)
        self.vpeopt = [0] * len(vpes)
        self.srs = [[SRS_INVALID] * 3 for _ in vpes]
        #: the YIELD qualifier inputs (the 34K's yield_qualifier pins): from
        #: the debugger / a device, from the GIC (a CPS), both
        self.yq_ext = self.yq_gic = 0
        self.yq = 0
        #: the TC that issues in the next cycle (prepare()), the last one
        self.next: TC | None = None
        self.last: TC | None = None
        #: the TC in Debug Mode (scheduling stops, MD00378 10.3)
        self.debug_tc: TC | None = None
        #: the Machine issuing an MTTR (its MVP decides what may be written)
        self._issuer = None
        #: a debugger (GDB's vCont) lets only these TCs issue (None: all)
        self.only: set | None = None
        #: pipeline mode: its 34K pipeline (emips.pipe34k), built by the Session
        self.pipeline = None
        #: told when software stops a TC (TCHalt = 1, TCStatus.A = 0: "stop")
        #: or lets it go on (a restart, a release, an activation: "go") —
        #: the 34K pipeline: f(tc, what)
        self.listener: Callable[[TC, str], None] | None = None
        #: cycles with nothing to issue
        self.idle_cycles = 0
        self.issued = [0] * tcs
        for v, m in enumerate(vpes):
            self._attach(m, v)

    # -- set-up ----------------------------------------------------------------

    def _attach(self, m, v: int) -> None:
        """VPE ``v`` = Machine ``m``: its first TC takes over the Machine's
        state; Config3.MT / ULRI, EBase.CPUNum, Debug.OffLine writable."""
        m.mt = self
        m.vpe = v
        m._mt_regs = REGISTERS
        tc = next(t for t in self.tcs if t.vpe == v)
        tc.gprs = m.shadow[0]
        tc.dsp_hi, tc.dsp_lo = m.dsp_hi, m.dsp_lo
        tc.ctx = {}
        self.current.append(tc)
        m._ixmt = False
        m._wmask[C.DEBUG] = m._wmask.get(C.DEBUG, M32) | DEBUG_OFFLINE
        cfg7 = m.cp0.get(CONFIG7)
        if cfg7 is not None:                    # WII: an interrupt ends a WAIT (10.2.1)
            cfg7["value"] |= CFG7_WII
        if self.itc is not None:                # ErrCtl.ITC: the CACHE pseudo-tags
            from .itc import ERRCTL_ITC
            m.itc = self.itc
            m._wmask[C.ERRCTL] = m._wmask.get(C.ERRCTL, M32) | ERRCTL_ITC
        if hasattr(m.mmu, "mt_drop"):           # MD00378 6.25.3: no Machine Check
            m.mmu.mt_drop = True
            m.mmu.index_p = True                # Index.P writable (MD00534 5.4.4)
        c = m.caches
        if c is not None and v == 0:            # the shared L1s: way exclusion (CPA)
            for cache, kind in ((c.i, "I"), (c.d, "D")):
                if cache is not None:
                    cache.excluded = (lambda c=c, k=kind: self._excluded(c, k))

    def firmware(self) -> None:
        """The boot firmware's Status for every TC (CU1, MX, KSU as it set
        them up for the TC it runs on each VPE): FORKed threads can use the
        FPU / the DSP Module like the first one.  The MT state stays at
        reset (EVP = 0, TE = 0, the other TCs halted, DA = 0)."""
        for tc in self.tcs:
            if not self.is_bound(tc):
                m = self.vpes[tc.vpe]
                tc.status_bits = m.cp0[C.STATUS]["value"] & STATUS_TC
                tc.asid = m.cp0[ENTRYHI]["value"] & ASID

    # -- binding -------------------------------------------------------------------

    def bind(self, tc: TC) -> None:
        """Make ``tc`` the TC its VPE's Machine runs (save the bound one)."""
        m = self.vpes[tc.vpe]
        old = self.current[tc.vpe]
        if old is tc:
            return
        self._save(m, old)
        self._load(m, tc)
        self.current[tc.vpe] = tc

    def _save(self, m, tc: TC) -> None:
        tc.ctx = {a: getattr(m, a) for a in _TC_ATTRS}
        tc.status_bits = m.cp0[C.STATUS]["value"] & STATUS_TC
        tc.asid = m.cp0[ENTRYHI]["value"] & ASID
        tc.userlocal = m.cp0[C.USERLOCAL]["value"]
        tc.debug_bits = m.cp0[C.DEBUG]["value"] & (DEBUG_SST | DEBUG_OFFLINE)

    def _load(self, m, tc: TC) -> None:
        for a, v in tc.ctx.items():
            setattr(m, a, v)
        tc.ctx = {}
        m.shadow[0] = tc.gprs
        css = m.srs_field(C.SRS_CSS)
        m.regs = m.shadow[css] if css < len(m.shadow) else m.shadow[0]
        m.dsp_hi, m.dsp_lo = tc.dsp_hi, tc.dsp_lo
        cell = m.cp0[C.STATUS]
        cell["value"] = cell["value"] & ~STATUS_TC | tc.status_bits
        cell = m.cp0[ENTRYHI]
        cell["value"] = cell["value"] & ~ASID | tc.asid
        m.cp0[C.USERLOCAL]["value"] = tc.userlocal
        cell = m.cp0[C.DEBUG]
        cell["value"] = cell["value"] & ~(DEBUG_SST | DEBUG_OFFLINE) | tc.debug_bits
        m._ixmt = tc.ixmt

    @contextmanager
    def bound(self, tc: TC):
        """``tc`` bound to its VPE for a moment (MFTR / MTTR, the views)."""
        old = self.current[tc.vpe]
        self.bind(tc)
        try:
            yield self.vpes[tc.vpe]
        finally:
            self.bind(old)

    def is_bound(self, tc: TC) -> bool:
        return self.current[tc.vpe] is tc

    def attr(self, tc: TC, name: str) -> Any:
        """A per-TC Machine attribute of ``tc`` (live when it is bound)."""
        return getattr(self.vpes[tc.vpe], name) if self.is_bound(tc) else tc.ctx[name]

    def set_attr(self, tc: TC, name: str, value: Any) -> None:
        if self.is_bound(tc):
            setattr(self.vpes[tc.vpe], name, value)
        else:
            tc.ctx[name] = value

    def tc_of(self, m) -> TC:
        """The TC bound to Machine ``m``."""
        return self.current[m.vpe]

    # -- scheduling ------------------------------------------------------------------

    @property
    def evp(self) -> bool:
        return bool(self.mvpcontrol & EVP)

    def exclusive(self, v: int) -> bool:
        """VPE ``v`` runs only its XTC: Status.EXL / ERL, Debug Mode, or
        VPEControl.TE = 0 (MD00378 6.5, 6.6)."""
        m = self.vpes[v]
        return bool(m.cp0[C.STATUS]["value"] & (C.ST_EXL | C.ST_ERL) or m._dm
                    or not self.vpecontrol[v] & TE)

    def _wakes(self, tc: TC) -> bool:
        """Something ends ``tc``'s WAIT: an exception its VPE takes before
        the next instruction, or an active interrupt signal even when
        Status.IE / EXL or TCStatus.IXMT keep it from being taken (the 34K's
        Config7.WII = 1, MD00534 10.2.1)."""
        with self.bound(tc) as m:               # (its own single step, IXMT)
            return m.interrupt_request() is not None or m.interrupt_signal()

    def eligible(self, tc: TC) -> bool:
        """Can ``tc`` issue now (see the module docstring)?  No side
        effects.  (Asked for every TC twice a step: :meth:`evp`,
        :meth:`is_bound`, :meth:`exclusive` and :meth:`attr` are written
        out here.)"""
        v = tc.vpe
        if self.debug_tc is not None:
            return tc is self.debug_tc
        if self.only is not None and tc not in self.only:
            return False                        # a debugger resumes only these
        if not tc.active or tc.halted:
            return False
        m = self.vpes[v]
        bound = self.current[v] is tc
        if (m.cp0[C.DEBUG]["value"] if bound else tc.debug_bits) & DEBUG_OFFLINE:
            return False
        if not self.mvpcontrol & EVP:
            if tc is not self.solo:
                return False
        elif not self.vpa[v]:
            return False
        if tc.n != self.xtc[v] and (m.cp0[C.STATUS]["value"] & (C.ST_EXL | C.ST_ERL)
                                    or m._dm or not self.vpecontrol[v] & TE):
            return False                        # exclusive(v): only its XTC
        if tc.yield_wait:
            return bool(self.yq & tc.yield_wait)
        if tc.gs_wait is not None:
            return self.itc is not None and self.itc.ready(tc.gs_wait)
        if m.waiting if bound else tc.ctx["waiting"]:
            return self._wakes(tc)
        return True

    def _dint(self) -> TC | None:
        """A VPE with a Debug Interrupt request: any of its TCs serves it,
        halted or not (MD00378 10.3) — the bound one."""
        found = None
        for v, m in enumerate(self.vpes):
            if m._dm:
                return None                     # ignored while one is in Debug Mode
            if found is None and m.dint_pending:
                found = self.current[v]
        return found

    # -- the policy manager (MD00534 13.2) -------------------------------------------

    def queue_avail(self) -> dict[str, int]:
        """The PM_*_avail inputs (0-7 free entries): the write back buffer's
        free entries (MD00534 9.4: 4), the fill-store buffer's and the load
        queue's while the memory timing / the 34K pipeline run (else all
        free), PM_sys_avail as set."""
        m = self.vpes[0]
        c = m.caches
        wbb = PM_SIZES["wbb"]
        fsb = PM_SIZES["fsb"]
        if c is not None and c.write_buffer is not None:
            wbb -= len(c.write_buffer.entries)
        t = c.timing if c is not None else None
        if t is not None and t.active:
            fsb -= sum(1 for r in t.pending.values() if r > t.cursor())
        avail = {"sys": self.sys_avail, "wbb": wbb, "fsb": fsb, "ldq": PM_SIZES["ldq"]}
        if self.pipeline_avail is not None:
            avail.update(self.pipeline_avail())
        return {k: max(0, min(7, v)) for k, v in avail.items()}

    def effective(self, tc: TC, avail: dict[str, int] | None = None) -> tuple[int, bool]:
        """``tc``'s group and stop priority: TCSchedule.GRP / STP, or a
        throttle's while it is active (WRR2, 13.2.3.1: an enabled queue
        input <= its threshold — 7 = always; throttle 0 first)."""
        sch = tc.schedule
        if self.pm == "wrr2":
            if avail is None:
                avail = self.queue_avail()
            for qe, th, grp, stp in ((4, 8, 11, 13), (14, 18, 21, 23)):
                enabled = (sch >> qe) & 0xF
                if not enabled:
                    continue
                limit = (sch >> th) & 7
                if limit == 7 or any(enabled >> i & 1 and avail[k] <= limit
                                     for i, k in enumerate(PM_INPUTS)):
                    return (sch >> grp) & 3, bool(sch >> stp & 1)
        return sch & 3, bool(sch & STP)

    def _groups(self, cands: list[TC] | None = None) -> dict[int, list[TC]]:
        """The eligible TCs (or ``cands``) by effective group, without the
        stopped ones."""
        avail = self.queue_avail() if self.pm == "wrr2" else None
        out: dict[int, list[TC]] = {}
        for tc in (cands if cands is not None else
                   [t for t in self.tcs if self.eligible(t)]):
            g, stp = self.effective(tc, avail)
            if not stp:
                out.setdefault(g, []).append(tc)
        return out

    def _priorities(self) -> tuple[int, ...]:
        return (0, 1, 2, 3) if self.gpo else self.priority

    def _winner(self, groups: dict[int, list[TC]]) -> int | None:
        if not groups:
            return None
        prio = self._priorities()
        return max(groups, key=lambda g: prio[g])

    def pick(self) -> TC | None:
        """The TC that issues next (no side effects): the dispatch
        scheduler's round-robin among the eligible TCs after the last that
        issued — with WRR / WRR2 inside the runnable group of the highest
        priority (13.1)."""
        if self.debug_tc is not None and not self.vpes[self.debug_tc.vpe]._dm:
            self.debug_tc = None
        d = self._dint()
        if d is not None:
            return d
        if self.pm == "rr" or self.debug_tc is not None:
            tcs = self.tcs
            start = self.last.n + 1 if self.last is not None else 0
            for tc in tcs[start:] + tcs[:start]:     # round-robin after the last
                if self.eligible(tc):
                    return tc
            return None
        return self.choose([t for t in self.tcs if self.eligible(t)])

    def choose(self, cands: list[TC]) -> TC | None:
        """The dispatch scheduler's choice among ``cands`` (TCs that can
        issue): round-robin after the last that issued, inside the group of
        the highest priority with WRR / WRR2 (MD00534 13.1).  The 34K
        pipeline asks it with the TCs that have an instruction ready."""
        n = len(self.tcs)
        if not cands:
            return None
        if self.pm == "rr" or self.debug_tc is not None:
            members = {t.n for t in cands}
            start = (self.last.n + 1) % n if self.last is not None else 0
            return next(self.tcs[(start + k) % n] for k in range(n)
                        if (start + k) % n in members)
        groups = self._groups(cands)
        g = self._winner(groups)
        if g is None:
            return None
        last = self.group_last[g]
        start = (last.n + 1) % n if last is not None else 0
        members = {t.n for t in groups[g]}
        return next(self.tcs[(start + k) % n] for k in range(n)
                    if (start + k) % n in members)

    def _rotate(self) -> None:
        """End of a cycle: the rotation held for as many cycles as the
        winning group had runnable TCs when it began, then the next count
        (13.2.7, Table 13.6)."""
        if self.pm == "rr" or self.gpo:
            return
        if self._hold is None:
            self._hold = 1
        self._hold -= 1
        if self._hold <= 0:
            self.priority = ROTATION[self.rotation]
            self.rotation = self.rotation % 15 + 1
            self._hold = None

    def _begin_rotation(self, cands: list[TC] | None = None) -> None:
        if self.pm != "rr" and not self.gpo and self._hold is None:
            groups = self._groups(cands)
            g = self._winner(groups)
            self._hold = len(groups[g]) if g is not None else 1

    def prepare(self) -> TC | None:
        """Pick the next TC and bind it: the VPE Machines then show what
        executes next (the views, breakpoints)."""
        self._begin_rotation()
        self.next = self.pick()
        if self.next is not None:
            self.bind(self.next)
        return self.next

    def cycle(self, run: Callable[[int], Any] | None = None) -> int | None:
        """One clock cycle: the next TC issues one instruction on its VPE
        (``run(vpe)``, default ``Machine.step``), every other VPE's time
        passes.  Returns the VPE that issued, or None (nothing could)."""
        self._begin_rotation()
        tc = self.next if self.next is not None and self.eligible(self.next) else self.pick()
        self.next = None
        if tc is None:
            self.idle_cycles += 1
            for m in self.vpes:
                m.advance(1)
            self._rotate()
            return None
        v = tc.vpe
        self.last_vp = v
        self.bind(tc)
        m = self.vpes[v]
        if m.caches is not None:
            m.caches.m = m
        before = self.before(tc)
        try:
            (run or (lambda i: self.vpes[i].step()))(v)
        finally:
            self.dispatched(tc)
            for other in self.vpes:
                if other is not m:
                    other.advance(1)
            self.after(tc, before)
            self._rotate()
        return v

    def before(self, tc: TC) -> tuple:
        """The state :meth:`after` compares with, taken before ``tc``'s
        instruction runs (bound on its VPE)."""
        m = self.vpes[tc.vpe]
        tc.yield_wait, tc.gs_wait = 0, None     # it issues (again, if it was blocked)
        tc.dt_written = False
        m.blocked = None
        return (self.exclusive(tc.vpe), m.cp0[C.STATUS]["value"], m._dm, m.waiting,
                len(m.exception_log))

    def dispatched(self, tc: TC) -> None:
        """``tc`` issued (the round-robin pointers, the issue count)."""
        self.last = tc
        if self.pm != "rr":
            self.group_last[self.effective(tc)[0]] = tc
        self.issued[tc.n] += 1

    def after(self, tc: TC, before: tuple) -> None:
        """After ``tc``'s instruction: blocked (RNST 2 / 3: not retired),
        or retired (TCScheFBack, DT — Table 7.15); a VPE that became
        exclusive runs this TC alone (XTC); Debug Mode stops the others."""
        m = self.vpes[tc.vpe]
        excl, status, dm, waiting, n_exc = before
        b, m.blocked = m.blocked, None
        if b is not None:
            if b.kind == "yield":
                tc.yield_wait = b.wait
            else:
                tc.gs_wait = b.wait
        elif len(m.exception_log) == n_exc and not waiting:
            if tc.fback < M32:
                tc.fback += 1                   # TCScheFBack: completed (13.2.5)
            if (not tc.dt_written and not status & (C.ST_EXL | C.ST_ERL) and not dm):
                tc.dt = True                    # an instruction retired (Table 7.15)
        if not excl and self.exclusive(tc.vpe):
            self.xtc[tc.vpe] = tc.n             # exception / DMT: this TC runs alone
        if m._dm and not dm:
            self.debug_tc = tc

    def issuing(self) -> list[int]:
        """The VPE whose TC issues next (prepare()), as a list."""
        return [self.next.vpe] if self.next is not None else []

    def alive(self, v: int) -> bool:
        """VPE ``v`` has a TC that runs or waits (or a debug request)."""
        m = self.vpes[v]
        if m._dm or m.dint_pending:
            return True
        evp = self.mvpcontrol & EVP
        if evp and not self.vpa[v]:
            return False
        for t in self.tcs:
            if t.vpe == v and t.active and not t.halted and (evp or t is self.solo):
                return True
        return False

    def set_yield_inputs(self, value: int) -> None:
        """The YIELD qualifier inputs (YQ, the 34K's 16 pins) as a device /
        the debugger drives them; ORed with what the GIC drives."""
        self.yq_ext = value & YQ_BITS
        self.yq = self.yq_ext | self.yq_gic

    def set_gic_yield_inputs(self, value: int) -> None:
        """The GIC's MAP_TO_YQ sources of this core (MD00904 9.5.3.13)."""
        self.yq_gic = value & YQ_BITS
        self.yq = self.yq_ext | self.yq_gic

    # -- the MT registers -------------------------------------------------------------

    def _mvp(self, m) -> bool:
        """The issuing VPE may manage others: VPEConf0.MVP, or Debug Mode
        (MD00378 10.3)."""
        i = self._issuer if self._issuer is not None else m
        return self.mvp[i.vpe] or i._dm

    def tcstatus(self, tc: TC) -> int:
        m = self.vpes[tc.vpe]
        if self.is_bound(tc):
            status, asid = m.cp0[C.STATUS]["value"], m.cp0[ENTRYHI]["value"] & ASID
        else:
            status, asid = tc.status_bits, tc.asid
        if not self._tlb(m):
            asid = 0                            # TASID: read-only 0 with an FMT
        return ((status & 0xF0000000) | (TMX if status & C.ST_MX else 0)
                | (self.rnst(tc) << RNST_SHIFT) | (TDS if self._tds(tc) else 0)
                | (DT if tc.dt else 0) | (DA if tc.da else 0) | (A if tc.active else 0)
                | (((status >> C.ST_KSU_SHIFT) & 3) << TKSU_SHIFT)
                | (IXMT if tc.ixmt else 0) | asid)

    @staticmethod
    def _tlb(m) -> bool:
        return getattr(m.mmu, "mt", 3) in (1, 4)

    def rnst(self, tc: TC) -> int:
        """TCStatus.RNST: 0 running, 1 blocked on WAIT, 2 on YIELD (3:
        gating storage); while halted, why it was blocked."""
        if tc.halted:
            return tc.halted_rnst
        if tc.yield_wait:
            return 2
        if tc.gs_wait is not None:
            return 3
        return 1 if self.attr(tc, "waiting") else 0

    def _tds(self, tc: TC) -> bool:
        """Halted with a delay slot next (Table 7.15)."""
        return tc.halted and bool(self.attr(tc, "slot_armed"))

    def tcrestart(self, tc: TC) -> int:
        """Where the TC goes on: its PC (the branch when a delay slot is
        next, TDS), bit 0 = the ISA mode (MD00534 7.2.15.1)."""
        pc = self.attr(tc, "pc")
        if self.attr(tc, "slot_armed"):
            pc = (pc - self.attr(tc, "_slot_back")) & M32
        return (pc & ~1) | self.attr(tc, "isa_mode")

    def _restart(self, tc: TC, value: int) -> None:
        """A TCRestart write (or a FORK): the TC goes on at ``value`` (bit
        0 = the ISA mode with a compressed ISA); TDS and the LLbit clear
        (MD00378 7.2)."""
        m = self.vpes[tc.vpe]
        isa = value & 1 if m.compressed is not None else 0
        for name, v in (("pc", value & ~1 if m.compressed is not None else value & M32),
                        ("isa_mode", isa), ("slot_armed", False), ("slot_target", None),
                        ("_forbidden", False), ("_redirect", None), ("waiting", False),
                        ("lr_valid", False), ("_slot_back", 4)):
            self.set_attr(tc, name, v)
        tc.yield_wait, tc.gs_wait = 0, None
        if self.listener is not None:
            self.listener(tc, "go")

    def mvpconf0(self) -> int:
        tlb = all(self._tlb(m) for m in self.vpes)
        return (MVPCONF0_M | (TLBS if tlb and len(self.vpes) > 1 else 0)
                | (GS if self.itc is not None else 0) | (PCP if len(self.vpes) > 1 else 0) | TCA
                | ((len(self.vpes) - 1) << PVPE_SHIFT) | (len(self.tcs) - 1))

    def mvpconf1(self) -> int:
        return (C1F | 1) if self.fpu else 0     # one FPU context (PCP1), 64-bit (C1F)

    def vpeconf0(self, v: int) -> int:
        m = self.vpes[v]
        shared = len(self.vpes) > 1
        caches = m.caches
        return (VPECONF0_M | (self.xtc[v] << XTC_SHIFT)
                | (ICS if shared and caches is not None and caches.i is not None else 0)
                | (DCS if shared and caches is not None and caches.d is not None else 0)
                | (SCS if shared and caches is not None and caches.l2 is not None else 0)
                | (TCS if shared and caches is not None and caches.l3 is not None else 0)
                | (MVP if self.mvp[v] else 0) | (VPA if self.vpa[v] else 0))

    def vpeconf1(self, v: int) -> int:
        return 1 if self.fpu_owner == v else 0   # NCP1: the FPU context

    # -- the configuration state (MVPControl.VPC; MD00534 7.2.2, 7.2.7, 5.4.4) ------------

    def _bind_fpu(self, v: int, on: bool) -> None:
        """VPEConf1.NCP1 written: the FPU context goes to VPE ``v`` (the
        other VPE's NCP1 clears, 7.2.7), or leaves it.  The FPRs / FCSR
        are the context's: they move with it; a VPE without one has
        Config1.FP = 0 and Status.CU1 / FR fixed at 0."""
        if not self.fpu:                        # PCP1 = 0: NCP1 cannot be set
            return
        owner = self.fpu_owner
        target = v if on else (None if owner == v else owner)
        if target == owner:
            return
        src = self.vpes[owner] if owner is not None else self._fpu_store
        for i, m in enumerate(self.vpes):
            has = i == target
            if has:
                m.fpr, m.fcsr = src.fpr, src.fcsr
            m.has_fpu = has
            cfg1 = m.cp0[C.CONFIG1]
            cfg1["value"] = cfg1["value"] | 1 if has else cfg1["value"] & ~1
            if has:
                m._wmask[C.STATUS] |= C.ST_CU1 | (C.ST_FR if m.release != "R1" else 0)
            else:
                m._wmask[C.STATUS] &= ~(C.ST_CU1 | C.ST_FR)
                m.cp0[C.STATUS]["value"] &= ~(C.ST_CU1 | C.ST_FR)
                for t in self.tcs:
                    if t.vpe == i and not self.is_bound(t):
                        t.status_bits &= ~C.ST_CU1
        if target is None:                      # unbound: kept for the next binding
            self._fpu_store = _FPUContext(src.fpr, src.fcsr)
        self.fpu_owner = target

    def _share_tlb(self, on: bool) -> None:
        """MVPControl.STLB (5.4.4): the VPEs' JTLBs as one array (at most
        64 entries: the rest are ignored), one VA / ASID space; Config1.
        MMU_Size follows (PTLBE = 0: by hardware).  Cleared: each VPE gets
        its own entries back, in order."""
        mmus = [m.mmu for m in self.vpes]
        if on:
            self._own_tlbs = [(mu.entries, mu.n) for mu in mmus]
            shared = [e for mu in mmus for e in mu.entries][:64]
            for mu in mmus:
                self._resize_tlb(mu, shared)
            mmus[0].peer, mmus[1].peer = mmus[1], mmus[0]
        else:
            shared = mmus[0].entries
            k = 0
            for mu, (own, n) in zip(mmus, self._own_tlbs):
                part = shared[k:k + n]
                k += n
                self._resize_tlb(mu, part + own[len(part):n])
                mu.peer = None

    @staticmethod
    def _resize_tlb(mu, entries: list) -> None:
        mu.entries = entries
        mu.n = mu.total = len(entries)
        mu._index_mask = (1 << max(6, (mu.total - 1).bit_length())) - 1
        mu.random = min(mu.random, mu.n - 1)
        mu._changed(None)
        m = mu.m
        if m is not None:
            cell = m.cp0[C.CONFIG1]
            cell["value"] = (cell["value"] & ~(0x3F << 25)) | (((mu.n - 1) & 0x3F) << 25)

    def _excluded(self, caches, kind: str) -> int:
        """The ways VPEOpt keeps from allocation for the VPE whose access
        refills (MVPControl.CPA, 7.2.11): IWX / DWX."""
        if not self.mvpcontrol & CPA:
            return 0
        m = caches.m
        v = getattr(m, "vpe", 0) if m is not None else 0
        opt = self.vpeopt[v]
        return (opt >> 8) & 0xF if kind == "I" else opt & 0xF

    def srsconf0(self, v: int) -> int:
        f = self.srs[v]
        return f[0] | (f[1] << 10) | (f[2] << 20)

    def read(self, m, key: tuple[int, int]) -> int:
        """MFC0 of an MT register on VPE ``m`` (its bound TC's)."""
        v = m.vpe
        tc = self.current[v]
        if key in NO_PM:
            return self._pm_read(tc, key)
        if key == MVPCONTROL:
            return self.mvpcontrol
        if key == MVPCONF0:
            return self.mvpconf0()
        if key == MVPCONF1:
            return self.mvpconf1()
        if key == VPECONTROL:
            return self.vpecontrol[v]
        if key == VPECONF0:
            return self.vpeconf0(v)
        if key == VPECONF1:
            return self.vpeconf1(v)
        if key == YQMASK:
            return self.yqmask[v]
        if key == VPEOPT:
            return self.vpeopt[v]
        if key == TCSTATUS:
            return self.tcstatus(tc)
        if key == TCBIND:
            return (tc.n << CURTC_SHIFT) | (TBE if tc.tbe else 0) | tc.vpe
        if key == TCRESTART:
            return self.tcrestart(tc)
        if key == TCHALT:
            return int(tc.halted)
        if key == TCCONTEXT:
            return tc.context
        if key == SRSCONF0:
            return self.srsconf0(v)
        raise KeyError(key)

    def write(self, m, key: tuple[int, int], value: int) -> None:
        """MTC0 of an MT register on VPE ``m`` (an MTTR: ``_issuer`` is
        the VPE issuing it)."""
        v = m.vpe
        tc = self.current[v]
        value &= M32
        mvp = self._mvp(m)
        if key in NO_PM:
            self._pm_write(tc, key, value)
            return
        if key == VPECONF1:
            if self.mvpcontrol & VPC and mvp:   # writable in the configuration state
                self._bind_fpu(v, bool(value & 0xFF))
            return
        if key in (MVPCONF0, MVPCONF1):
            return                              # read-only
        if key == MVPCONTROL:
            if not mvp:
                return
            old = self.mvpcontrol
            new = value & (EVP | VPC | CPA)
            stlb = old & STLB                   # STLB: writable only while VPC was set
            if old & VPC and self.mvpconf0() & TLBS:
                stlb = value & STLB
            self.mvpcontrol = new | stlb
            if bool(stlb) != bool(old & STLB):
                self._share_tlb(bool(stlb))
            if old & EVP and not value & EVP:   # only the TC that cleared it runs
                issuer = self._issuer if self._issuer is not None else m
                self.solo = self.current[issuer.vpe]
        elif key == VPECONTROL:
            self.vpecontrol[v] = (self.vpecontrol[v] & (7 << EXCPT_SHIFT)
                                  | value & (YSI | GSI | TE | 0xFF))
        elif key == VPECONF0:
            if not mvp:
                return
            if not self.vpa[v] and self._issuer is not None and self._issuer is not m:
                self.xtc[v] = ((value >> XTC_SHIFT) & 0xFF) % len(self.tcs)
            self.mvp[v] = bool(value & MVP)
            self.vpa[v] = bool(value & VPA)
        elif key == YQMASK:
            self.yqmask[v] = value & YQ_BITS
        elif key == VPEOPT:
            self.vpeopt[v] = value & 0xF0F
        elif key == TCSTATUS:
            self._write_tcstatus(tc, value)
        elif key == TCBIND:
            tc.tbe = bool(value & TBE)
            if self.mvpcontrol & VPC:           # TCA = 1: rebinding (MD00378 8.2)
                self._rebind(tc, (value & 0xF) % len(self.vpes))
        elif key == TCRESTART:
            self._restart(tc, value)
        elif key == TCHALT:
            self._halt(tc, bool(value & 1))
        elif key == TCCONTEXT:
            tc.context = value
        elif key == SRSCONF0:
            self._write_srsconf(v, value)

    def _pm_read(self, tc: TC, key: tuple[int, int]) -> int:
        """The policy manager's registers (MD00534 13.2.4 - 13.2.6); -1
        where it implements none."""
        if self.pm == "rr":
            return M32
        if key == TCSCHEDULE:
            return tc.schedule
        if key == TCSCHEFBACK:
            return tc.fback
        if key == VPESCHEDULE:
            return GPO if self.gpo else 0
        if self.pm == "wrr2":                   # VPEScheFBack: the sizes
            z = PM_SIZES
            return (z["sys"] << 12) | (z["wbb"] << 8) | (z["fsb"] << 4) | z["ldq"]
        return M32

    def _pm_write(self, tc: TC, key: tuple[int, int], value: int) -> None:
        if self.pm == "rr":
            return
        if key == TCSCHEDULE:
            tc.schedule = value & TCSCHEDULE_MASK[self.pm]
        elif key == TCSCHEFBACK:
            tc.fback = value & M32
        elif key == VPESCHEDULE:
            gpo = bool(value & GPO)
            if gpo != self.gpo:
                self.gpo = gpo
                self._hold = None

    def _write_tcstatus(self, tc: TC, value: int) -> None:
        m = self.vpes[tc.vpe]
        bits = ((value & 0xF0000000) | (C.ST_MX if value & TMX else 0)
                | (((value >> TKSU_SHIFT) & 3) << C.ST_KSU_SHIFT))
        mask = m._wmask.get(C.STATUS, M32) & STATUS_TC
        if self.is_bound(tc):
            cell = m.cp0[C.STATUS]
            cell["value"] = cell["value"] & ~mask | bits & mask
            if self._tlb(m):
                cell = m.cp0[ENTRYHI]
                cell["value"] = cell["value"] & ~ASID | value & ASID
            m._ixmt = bool(value & IXMT)
        else:
            tc.status_bits = tc.status_bits & ~mask | bits & mask
            if self._tlb(m):
                tc.asid = value & ASID
        tc.dt = bool(value & DT)
        tc.dt_written = True                    # (not set again by this instruction)
        tc.da = bool(value & DA)
        tc.ixmt = bool(value & IXMT)
        was = tc.active
        tc.active = bool(value & A)
        if not tc.active:
            tc.yield_wait, tc.gs_wait = 0, None
        if was != tc.active and self.listener is not None:
            self.listener(tc, "go" if tc.active else "stop")

    def _halt(self, tc: TC, h: bool) -> None:
        """TCHalt.H (Table 7.18): a halted TC stops at the next instruction
        to issue; one blocked on WAIT / YIELD restarts at that instruction,
        RNST keeping why."""
        if h and not tc.halted:
            tc.halted_rnst = self.rnst(tc)
            if self.listener is not None:       # (the 34K pipeline: its PC now)
                self.listener(tc, "stop")
            if self.attr(tc, "waiting"):
                self.set_attr(tc, "waiting", False)
                self.set_attr(tc, "pc", self.attr(tc, "wait_pc"))
            tc.yield_wait, tc.gs_wait = 0, None  # its PC is at the YIELD / the access
        elif not h:
            tc.halted_rnst = 0
            if tc.halted and self.listener is not None:
                self.listener(tc, "go")
        tc.halted = h

    def _rebind(self, tc: TC, v: int) -> None:
        """TCBind.CurVPE written while MVPControl.VPC = 1."""
        if v == tc.vpe:
            return
        old = tc.vpe
        if self.is_bound(tc):                   # its old VPE needs another TC bound
            spare = next((t for t in self.tcs if t.vpe == old and t is not tc), None)
            if spare is None:
                spare = TC(-1, old)             # no TC left: an empty context
            self.bind(spare)
        tc.vpe = v
        self.srs[old] = [SRS_INVALID if f == tc.n else f for f in self.srs[old]]
        self._refresh_srs(old)

    def _write_srsconf(self, v: int, value: int) -> None:
        """SRSConf0: SRS1-3 take a TC bound to this VPE (MD00378 6.20);
        0x3FE / 0x3FF = unassigned; HSS follows (6.25.1)."""
        for i in range(3):
            f = (value >> (10 * i)) & 0x3FF
            if f >= SRS_INVALID:
                self.srs[v][i] = SRS_INVALID
            elif f < len(self.tcs) and self.tcs[f].vpe == v:
                self.srs[v][i] = f
        self._refresh_srs(v)

    def _refresh_srs(self, v: int) -> None:
        """The VPE's shadow sets = its own TC's GPRs + the TCs SRSConf0
        names, up to the first unassigned field (SRSCtl.HSS)."""
        m = self.vpes[v]
        banks = [self.current[v].gprs]
        for f in self.srs[v]:
            if f == SRS_INVALID or self.tcs[f].vpe != v:
                break
            banks.append(self.tcs[f].gprs)
        m.shadow = banks
        m.hss = len(banks) - 1
        srs = m.cp0[C.SRSCTL]["value"]
        m.cp0[C.SRSCTL]["value"] = srs & ~(0xF << C.SRS_HSS) | (m.hss << C.SRS_HSS)
        if m.hss:
            m._wmask[C.SRSCTL] = (0xF << C.SRS_ESS) | (0xF << C.SRS_PSS)
            m._wmask[C.SRSMAP] = M32
        else:
            m._wmask[C.SRSCTL] = 0
            m._wmask[C.SRSMAP] = 0
        css = m.srs_field(C.SRS_CSS)
        m.regs = banks[css] if css < len(banks) else banks[0]

    # -- MFTR / MTTR (MD00378 Tables 5.1 / 5.2) ----------------------------------------

    def target(self, m) -> TC | None:
        """VPEControl.TargTC's TC, or None: above PTC, or bound to another
        VPE while the issuing one has no MVP."""
        n = self.vpecontrol[m.vpe] & 0xFF
        if n >= len(self.tcs):
            return None
        tc = self.tcs[n]
        if tc.vpe != m.vpe and not (self.mvp[m.vpe] or m._dm):
            return None
        return tc

    def mftr(self, m, tc: TC, rt: int, u: int, sel: int, h: int, rx: int) -> int:
        """MFTR's value (-1: not implemented / not accessible)."""
        t = self.vpes[tc.vpe]
        if u == 0:                              # CP0 register rt, sel of the target TC
            if (rt, sel) not in t.cp0 or h:
                return M32 if (rt, sel) not in t.cp0 else 0
            with self.bound(tc) as tm:
                return tm.cp0_read(rt, sel)
        if sel == 0:
            return tc.gprs[rt] & M32 if rt else 0
        if sel == 1:
            if rt == 0:
                return self.attr(tc, "lo") & M32
            if rt == 1:
                return self.attr(tc, "hi") & M32
            if not t.dsp or rt in (2, 6, 10, 14) or rt not in (4, 5, 8, 9, 12, 13, 16):
                return M32                      # no ACX on MIPS32; no DSP Module
            if rt == 16:
                return self.attr(tc, "dspctl") & M32
            a = rt >> 2
            return (tc.dsp_hi if rt & 1 else tc.dsp_lo)[a] & M32
        if sel == 2:
            if not t.has_fpu:
                return M32
            v = t.fpr[rt]
            return (v >> 32) & M32 if h else v & M32
        if sel == 3:
            if not t.has_fpu:
                return M32
            from .sim import fcr_read
            v = fcr_read(t, rt)
            return M32 if v is None else v
        return M32                              # COP2 (none) and reserved selects

    def mttr(self, m, tc: TC, rd: int, u: int, sel: int, h: int, rx: int, value: int) -> None:
        """MTTR: write ``value`` (no effect where MFTR would read -1)."""
        t = self.vpes[tc.vpe]
        value &= M32
        if u == 0:
            if (rd, sel) not in t.cp0 or h:
                return
            self._issuer = m
            try:
                with self.bound(tc) as tm:
                    tm.mtc0(rd, sel, value)
            finally:
                self._issuer = None
            return
        if sel == 0:
            if rd:
                tc.gprs[rd] = value
            return
        if sel == 1:
            if rd in (0, 1):
                self.set_attr(tc, "lo" if rd == 0 else "hi", value)
            elif t.dsp and rd == 16:
                from .dsp import DSPCONTROL_MASK
                self.set_attr(tc, "dspctl", value & DSPCONTROL_MASK)
            elif t.dsp and rd in (4, 5, 8, 9, 12, 13):
                (tc.dsp_hi if rd & 1 else tc.dsp_lo)[rd >> 2] = value
            return
        if sel == 2 and t.has_fpu:
            old = t.fpr[rd]
            t.fpr[rd] = (old & M32) | (value << 32) if h else (old & ~M32) | value
        elif sel == 3 and t.has_fpu:
            from .sim import fcr_write
            fcr_write(t, rd, value)             # an FPE it enables: the target's (6.29)

    # -- the views ----------------------------------------------------------------------

    def state(self, tc: TC) -> str:
        """Table 6.22: halted / free / running / blocked (WAIT, YIELD)."""
        if tc.halted:
            return "halted"
        if not tc.active:
            return "free"
        r = self.rnst(tc)
        return {0: "running", 1: "blocked (WAIT)", 2: "blocked (YIELD)",
                3: "blocked (gating storage)"}[r]

    def rows(self) -> list[dict[str, Any]]:
        """One row per TC for the views (side-effect free)."""
        out = []
        for tc in self.tcs:
            out.append({
                "tc": tc.n, "vpe": tc.vpe, "state": self.state(tc),
                "pc": self.tcrestart(tc) & ~1 if not self.is_bound(tc) or tc.halted
                else self.vpes[tc.vpe].pc,
                "bound": self.is_bound(tc), "next": tc is self.next,
                "da": tc.da, "ixmt": tc.ixmt, "dt": tc.dt, "issued": self.issued[tc.n],
                "eligible": self.eligible(tc),
                "xtc": self.xtc[tc.vpe] == tc.n,
                "yield": tc.yield_wait,
                "gs": str(tc.gs_wait) if tc.gs_wait is not None else None,
                "group": self.effective(tc)[0] if self.pm != "rr" else None,
                "stp": self.effective(tc)[1] if self.pm != "rr" else False,
            })
        return out

    def lines(self) -> list[str]:
        """The MT state as text (debugger ``mt``)."""
        ctl = self.mvpcontrol
        out = [f"MT Module: {len(self.vpes)} VPE(s), {len(self.tcs)} TC(s); "
               f"MVPControl 0x{ctl:08x} (EVP {int(bool(ctl & EVP))}, VPC "
               f"{int(bool(ctl & VPC))})" + ("" if self.evp else
                                             f" - only {self.solo} runs"
                                             if self.solo is not None else "")]
        for v in range(len(self.vpes)):
            vc = self.vpecontrol[v]
            m = self.vpes[v]
            excl = "" if not self.exclusive(v) else f", runs only TC{self.xtc[v]} (" + (
                "Debug Mode" if m._dm else "EXL/ERL" if m.cp0[C.STATUS]["value"] & 6
                else "TE = 0") + ")"
            out.append(f"VPE {v}: VPA {int(self.vpa[v])} MVP {int(self.mvp[v])} TE "
                       f"{int(bool(vc & TE))} TargTC {vc & 0xFF} EXCPT "
                       f"{(vc >> EXCPT_SHIFT) & 7} YQMask 0x{self.yqmask[v]:04x}{excl}")
        out.append(f"YIELD qualifier inputs 0x{self.yq:04x}; idle cycles {self.idle_cycles}")
        out.append(self.pm_line())
        for r in self.rows():
            flags = " ".join(f for f, on in (("DA", r["da"]), ("IXMT", r["ixmt"]),
                                              ("DT", r["dt"]), ("STP", r["stp"])) if on)
            mark = ">" if r["next"] else " "
            wait = f" waits for YQ 0x{r['yield']:04x}" if r["yield"] else ""
            if r["gs"]:
                wait = f" waits: {r['gs']}"
            group = f" group {r['group']}" if r["group"] is not None else ""
            out.append(f"{mark} TC{r['tc']} VPE {r['vpe']}  {r['state']:<24} pc 0x{r['pc']:08x}"
                       f"  issued {r['issued']:<6}{group} {flags}{wait}")
        return out

    def pm_line(self) -> str:
        """The policy manager's state (MD00534 13.2)."""
        if self.pm == "rr":
            return "Policy manager: basic round-robin (every TC at one priority)"
        name = {"wrr": "weighted round-robin", "wrr2": "weighted round-robin + throttles"}
        prio = self._priorities()
        how = ("fixed priorities (VPESchedule.GPO = 1)" if self.gpo else
               f"rotating, next count {self.rotation}")
        text = (f"Policy manager: {name[self.pm]}; group priorities "
                + " ".join(f"G{g}=P{prio[g]}" for g in range(4)) + f" ({how})")
        if self.pm == "wrr2":
            a = self.queue_avail()
            text += "; free: " + ", ".join(f"{k} {a[k]}" for k in ("ldq", "fsb", "wbb", "sys"))
        return text


class Cluster:
    """The MT processors of a CPS (the 1004K / interAptiv: one per core,
    its VPEs the GIC's processors ``core * vpes + n``, MD00904 Table 9.1).
    The Session drives it as one processor: a cycle issues one TC on every
    running core."""

    def __init__(self, cps) -> None:
        self.cps = cps
        #: the VP whose TC issued last (an error's VP)
        self.last_vp = 0

    @property
    def procs(self) -> list[Processor]:
        """Each core's processor (a restarted core: its new one)."""
        return [m.mt for m in self.cps.cores]

    @property
    def itc(self):
        return self.procs[0].itc

    def _base(self, core: int) -> int:
        return core * self.cps.vpes

    def firmware(self) -> None:
        for p in self.procs:
            p.firmware()

    def prepare(self) -> None:
        for c in self.cps.running():
            self.procs[c].prepare()

    def issuing(self) -> list[int]:
        return [self._base(c) + v for c in self.cps.running()
                for v in self.procs[c].issuing()]

    def alive(self, vp: int) -> bool:
        c = self.cps.vp_core(vp)
        return c in self.cps.running() and self.procs[c].alive(vp - self._base(c))

    def cycle(self, run: Callable[[int], Any]) -> None:
        """One cycle of every running core: each issues one TC."""
        for c in self.cps.running():
            p = self.procs[c]
            base = self._base(c)
            try:
                p.cycle(lambda v, b=base: run(b + v))
            finally:
                self.last_vp = base + p.last_vp

    def processor_of(self, vp: int) -> Processor:
        return self.procs[self.cps.vp_core(vp)]

    def lines(self) -> list[str]:
        out = []
        for c, p in enumerate(self.procs):
            out.append(f"core {c} (VPs {self._base(c)}-{self._base(c) + len(p.vpes) - 1}):")
            out += ["  " + line for line in p.lines()]
        return out


# ----------------------------------------------------------------- executors

def _rt(w: int) -> int:
    return (w >> 16) & 31


def _rd(w: int) -> int:
    return (w >> 11) & 31


def _rs(w: int) -> int:
    return (w >> 21) & 31


def _thread(m, sub: int, why: str):
    """A Thread exception (ExcCode 25) with VPEControl.EXCPT = ``sub``."""
    from .sim import _Arch
    p = m.mt
    v = m.vpe
    p.vpecontrol[v] = p.vpecontrol[v] & ~(7 << EXCPT_SHIFT) | (sub << EXCPT_SHIFT)
    return _Arch(C.THREAD, info={"why": f"{EXCPT_NAMES[sub]}: {why}", "excpt": sub})


def _e_mt_te(sc: int):
    def ex(m, w: int) -> int | None:
        """DMT / EMT rt: GPR[rt] <- VPEControl; TE <- sc."""
        p = m.mt
        m.write_reg(_rt(w), p.vpecontrol[m.vpe])
        p.vpecontrol[m.vpe] = p.vpecontrol[m.vpe] & ~TE | (TE if sc else 0)
        return None
    return ex


def _e_mt_evp(sc: int):
    def ex(m, w: int) -> int | None:
        """DVPE / EVPE rt: GPR[rt] <- MVPControl; EVP <- sc if MVP."""
        p = m.mt
        m.write_reg(_rt(w), p.mvpcontrol)
        if p.mvp[m.vpe] or m._dm:
            if not sc and p.evp:
                p.solo = p.current[m.vpe]
            p.mvpcontrol = p.mvpcontrol & ~EVP | (EVP if sc else 0)
        return None
    return ex


def _e_fork(m, w: int) -> int | None:
    """FORK rd, rs, rt (MD00378 FORK): a free dynamically allocatable TC of
    this VPE starts at GPR[rs] with its GPR[rd] = GPR[rt]; none: Thread
    Overflow."""
    p = m.mt
    start, data = m.read_reg(_rs(w)), m.read_reg(_rt(w))
    for t in p.tcs:
        if t.vpe == m.vpe and t.da and not t.halted and not t.active:
            break
    else:
        raise _thread(m, OVERFLOW, "no free, non-halted, dynamically allocatable TC")
    p._restart(t, start)
    if _rd(w):
        t.gprs[_rd(w)] = data
    t.userlocal = m.cp0[C.USERLOCAL]["value"]   # Config3.ULRI = 1
    status = m.cp0[C.STATUS]["value"]
    t.status_bits = t.status_bits & ~(3 << C.ST_KSU_SHIFT) | status & (3 << C.ST_KSU_SHIFT)
    t.asid = m.cp0[ENTRYHI]["value"] & ASID     # TASID: the forking TC's
    t.dt = t.da = t.active = True
    t.halted_rnst = 0
    me = p.current[m.vpe]
    t.schedule = t.schedule & ~3 | me.schedule & 3   # the parent's GRP (13.2.2)
    return None


def _e_yield(m, w: int) -> int | None:
    """YIELD rd, rs (MD00378 YIELD): 0 = deallocate, -1 = reschedule, -2 =
    poll, > 0 = wait for one of these YIELD qualifier inputs; rd <- the
    enabled inputs."""
    p = m.mt
    tc = p.current[m.vpe]
    v = m.vpe
    r = m.read_reg(_rs(w))
    sr = r - (1 << 32) if r & 0x80000000 else r
    if sr == 0:
        if not any(t.vpe == v and t is not tc and t.da and not t.halted and t.active
                   for t in p.tcs):
            raise _thread(m, UNDERFLOW, "no other activated, dynamically allocatable TC")
    elif sr > 0 and r & ~p.yqmask[v]:
        raise _thread(m, INVALID_YQ, f"0x{r:08x} has bits YQMask (0x{p.yqmask[v]:04x}) "
                      "does not enable")
    if sr != -2 and p.vpecontrol[v] & YSI and tc.dt:
        raise _thread(m, YIELD_SCHED, "VPEControl.YSI and TCStatus.DT are set")
    if sr == 0:
        tc.active = False                       # deallocated: this TC stops
    elif sr > 0 and not p.yq & r:
        raise Blocked("yield", r)               # (RNST 2) issues again with a qualifier
    m.write_reg(_rd(w), p.yq & p.yqmask[v])
    return None


def _e_mftr(m, w: int) -> int | None:
    """MFTR rd, rt, u, sel, h: GPR[rd] <- a register of VPEControl.TargTC."""
    p = m.mt
    tc = p.target(m)
    rt, rx, u, h, sel = _rt(w), (w >> 6) & 31, (w >> 5) & 1, (w >> 4) & 1, w & 7
    value = M32 if tc is None else p.mftr(m, tc, rt, u, sel, h, rx)
    m.write_reg(_rd(w), value)
    return None


def _e_mttr(m, w: int) -> int | None:
    """MTTR rt, rd, u, sel, h: a register of VPEControl.TargTC <- GPR[rt]."""
    p = m.mt
    tc = p.target(m)
    rd, rx, u, h, sel = _rd(w), (w >> 6) & 31, (w >> 5) & 1, (w >> 4) & 1, w & 7
    if tc is not None:
        p.mttr(m, tc, rd, u, sel, h, rx, m.read_reg(_rt(w)))
    return None


EXECUTORS: dict[str, Callable] = {
    "DMT": _e_mt_te(0), "EMT": _e_mt_te(1),
    "DVPE": _e_mt_evp(0), "EVPE": _e_mt_evp(1),
    "FORK": _e_fork, "YIELD": _e_yield, "MFTR": _e_mftr, "MTTR": _e_mttr,
}
#: the MT rows that need kernel mode or Status.CU0 (Coprocessor Unusable)
PRIVILEGED = frozenset({"DMT", "EMT", "DVPE", "EVPE", "MFTR", "MTTR"})
