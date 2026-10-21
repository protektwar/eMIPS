"""MIPS Trace: PDtrace and the Trace Control Block (PLAN M14 phase 4).

References: *MIPS PDtrace Interface and Trace Control Block Specification*
(MD00439 rev. 6.16: chapter 3 the trace information, chapter 4 the trace
formats TF1-TF6, chapter 5 the trace word, chapter 6 compression, chapters
7-10 the registers and triggers) as the 24K implements it (MD00343 11.7-11.13,
PDtrace 4.x; CP0 TraceControl / TraceControl2 / UserTraceData / TraceIBPC /
TraceDBPC 6.2.29-6.2.33).

* :class:`PDtrace` — the pipeline side, one per core (``Machine.pdt``).
  Each instruction that completes is *captured* when it executes
  (:meth:`PDtrace.capture`, from ``Machine._traced``: the Machine, both
  pipelines): whether it is traced (TraceOn and the mode / ASID match of
  TraceControl or TCBCONTROLA, or a hardware-breakpoint trigger,
  TraceIBPC / TraceDBPC), its completion indicator InsComp (I / IL / IS,
  IB / ILB / ISB on a PC discontinuity, IPC for a periodic
  synchronisation) and its trace records — a TMOAS (mode, ASID) at a start,
  a sync or a mode change, the PC (TPC: delta or full) when the new PC
  cannot be predicted from the program, load / store addresses (TLA / TSA:
  the first full, then deltas) and data (TD) as TraceControl2.Mode /
  TCBCONTROLC.Mode ask, user data (TU1 / TU2: UserTraceData).  The records
  wait in a FIFO and leave on the 32-bit PDtrace data bus one per cycle
  (:meth:`PDtrace.clock`); a full FIFO overflows (TF5, restart with a TMOAS
  and a full PC) unless IO (Inhibit Overflow) stalls the core.
* :class:`TCB` — the Trace Control Block, behind the core's TAP
  (TCBCONTROLA 0x10, TCBCONTROLB 0x11, TCBDATA 0x12, TCBCONTROLC 0x13; through
  TCBDATA: TCBCONFIG, TCBTW, TCBRDP, TCBWRP, TCBSTP, TCBTRIG0-3).  Every
  cycle it turns the PDtrace output into trace formats (TF1 stall, TF2
  instruction, TF3 / TF4 instruction + record, TF5 overflow, TF6 TCB
  information; without CA no TF1 and bit 0 dropped), packs them into 64-bit
  trace words (Type field, formats spilling into the next word) and writes
  them to its on-chip trace memory (Trace-To: wrapping; Trace-From: until
  full, BF).  Triggers: Debug Mode entry, the probe's trigger input; actions
  start / end / about on TCBCONTROLB.EN, a TF6, the probe's trigger output.
* :func:`decode_words` / :func:`regenerate` — trace regeneration software
  (what a debugger runs on the trace words with the program image): the
  formats, then the executed instructions with their addresses, data and
  mode changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1

# ------------------------------------------------- InsComp / TType (ch. 3)
NI, I, IL, IS, IPC, IB, ILB, ISB = range(8)
INSCOMP_NAMES = ("NI", "I", "IL", "IS", "IPC", "IB", "ILB", "ISB")
NT, TPC, TLA, TSA, TD, TMOAS, TU1, TU2 = range(8)
TTYPE_NAMES = ("NT", "TPC", "TLA", "TSA", "TD", "TMOAS", "TU1", "TU2")
#: processor operating modes in a TMOAS record (Table 3.4)
POM_KERNEL, POM_EXL, POM_ERL, POM_DEBUG, POM_SUPER, POM_USER = 0, 1, 2, 3, 4, 5
POM_NAMES = {0: "kernel", 1: "exception (EXL)", 2: "error (ERL)", 3: "debug",
             4: "supervisor", 5: "user"}
#: TraceControl2.Mode / TCBCONTROLC.Mode bits
MODE_PC, MODE_LA, MODE_SA, MODE_LD, MODE_SD = 1, 2, 4, 8, 16

# ------------------------------------------------------- CP0 (6.2.29-33)
TRACECONTROL, TRACECONTROL2, USERTRACEDATA = (23, 1), (23, 2), (23, 3)
TRACEIBPC, TRACEDBPC = (23, 4), (23, 5)
TC_TS = 1 << 31
TC_UT = 1 << 30
TC_TB = 1 << 27
TC_IO = 1 << 26
TC_D, TC_E, TC_K, TC_S, TC_U = 1 << 25, 1 << 24, 1 << 23, 1 << 22, 1 << 21
TC_ASIDM_SHIFT, TC_ASID_SHIFT = 13, 5
TC_G = 1 << 4
TC_TFCR, TC_TLSM, TC_TIM, TC_ON = 1 << 3, 1 << 2, 1 << 1, 1 << 0
#: TraceControl2: ValidModes = 2 (PC, addresses and data), PC tracing on
TC2_VALIDMODES = 2 << 5
TC2_MODE_SHIFT = 7
IBPC_IE = DBPC_DE = 1 << 28

# ------------------------------------------------------- TCB (11.9)
TCBCONTROLA, TCBCONTROLB, TCBDATA, TCBCONTROLC = 0x10, 0x11, 0x12, 0x13
TCB_INSTRUCTIONS = {TCBCONTROLA: "TCBCONTROLA", TCBCONTROLB: "TCBCONTROLB",
                    TCBDATA: "TCBDATA", TCBCONTROLC: "TCBCONTROLC"}
# TCBCONTROLA (Table 11.31)
TA_VMODES = 2 << 24
TA_ADW = 1 << 23
TA_SYP_SHIFT = 20
TA_TB, TA_IO = 1 << 19, 1 << 18
TA_D, TA_E, TA_S, TA_K, TA_U = 1 << 17, 1 << 16, 1 << 15, 1 << 14, 1 << 13
TA_ASID_SHIFT = 5
TA_G, TA_TFCR, TA_TLSM, TA_TIM, TA_ON = 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0
TA_WRITE = (7 << TA_SYP_SHIFT) | TA_TB | TA_IO | TA_D | TA_E | TA_S | TA_K | TA_U \
    | (0xFF << TA_ASID_SHIFT) | TA_G | TA_TFCR | TA_TLSM | TA_TIM | TA_ON
# TCBCONTROLB (Table 11.32)
TB_WE = 1 << 31
TB_REG_SHIFT = 21
TB_WR = 1 << 20
TB_RM, TB_TR, TB_BF = 1 << 16, 1 << 15, 1 << 14
TB_TM_SHIFT = 12
TB_TLSIF = 1 << 11
TB_CR_SHIFT = 8
TB_CA, TB_OFC, TB_EN = 1 << 2, 1 << 1, 1 << 0
# TCBCONTROLC (Table 11.35)
TC_MODE_SHIFT = 23
# TCBTRIGx (Table 11.41)
TR_TRACE = 1 << 23
TR_CHTRO, TR_PDTRO = 1 << 15, 1 << 14
TR_DM, TR_CHTRI, TR_PDTRI = 1 << 6, 1 << 5, 1 << 4
TR_TYPE_SHIFT = 2
TR_FO, TR_TR = 1 << 1, 1 << 0
TRIG_START, TRIG_END, TRIG_ABOUT, TRIG_INFO = 0, 1, 2, 3
TRIG_WRITE = (0xFF << 24) | TR_TRACE | TR_CHTRO | TR_PDTRO | TR_DM | TR_CHTRI | TR_PDTRI \
    | (3 << TR_TYPE_SHIFT) | TR_FO | TR_TR

#: the TCB eMIPS builds (24K options): 4 triggers, 8 KB of on-chip trace
#: memory (SZ 5: 2^(5+8) bytes = 1024 trace words), no off-chip interface
TRIGGERS = 4
TRACE_SZ = 5
TCB_REV = 1
#: TF6 TCBcodes (Table 4.1)
TF6_START, TF6_END, TF6_INFO = 0b0000, 0b0100, 0b1100
TF6_NOTRACE, TF6_BACKSTALL, TF6_BREAKPOINT = 0b0001, 0b0101, 0b1001
#: the PDtrace FIFO depth (the 24K's is not published: an eMIPS choice)
FIFO_DEPTH = 8
#: bits of the AD field (TCBCONTROLA.ADW = 1: the 24K's 32-bit PDO_AD bus)
ADW = 32
#: DataOrder bits in TF4 (PDtrace 4.x: 4)
DATAORDER_BITS = 4

#: the instructions PDtrace flags as possible function calls / returns
#: (Table 3.9; MIPS32)
FUNCTION_CR = frozenset({"JAL", "JALR", "JALR.HB", "JALX", "JR", "JR.HB", "ERET", "DERET",
                         "BALC", "JIALC", "JIC"})
#: register jumps and returns: the new PC cannot be predicted
_REGISTER_TARGET = frozenset({"JR", "JR.HB", "JALR", "JALR.HB", "JIC", "JIALC"})
_RETURNS = frozenset({"ERET", "DERET", "ERETNC"})
_LOADS = frozenset({"LWC1", "LDC1", "LWXC1", "LDXC1", "LUXC1", "LWC2", "LDC2", "LWPC",
                    "LWUPC", "LDPC"})
_STORES = frozenset({"SWC1", "SDC1", "SWXC1", "SDXC1", "SUXC1", "SWC2", "SDC2"})


def static_target(name: str, word: int, pc: int, mask: int = M32) -> int | None:
    """The target a branch or jump encodes (None: a register jump or not
    a control instruction); ``mask``: the PC's width (MIPS64: 64 bits)."""
    if name in ("J", "JAL", "JALX"):
        return ((pc + 4) & mask & ~0x0FFFFFFF) | ((word & 0x3FFFFFF) << 2)
    if name in ("BC", "BALC"):
        off, bits = word & 0x3FFFFFF, 26
    elif name in ("BEQZC", "BNEZC"):
        off, bits = word & 0x1FFFFF, 21
    else:
        off, bits = word & 0xFFFF, 16
    if off >> (bits - 1):
        off -= 1 << bits
    return (pc + 4 + (off << 2)) & mask


def _is_likely(name: str) -> bool:
    return name.endswith("L") and name not in ("BAL", "JAL", "BGEZAL", "BLTZAL", "SLL",
                                               "SRL", "DSLL") and name.startswith("B")


def mode_of(m) -> tuple[int, str]:
    """(POM, the mode's name for the enables): Debug, Exception (EXL /
    ERL), Kernel, Supervisor, User (MD00439 2.2)."""
    if m.debug_mode:
        return POM_DEBUG, "D"
    status = m.status
    if status & 4:
        return POM_ERL, "E"
    if status & 2:
        return POM_EXL, "E"
    ksu = (status >> 3) & 3
    if ksu == 1:
        return POM_SUPER, "S"
    if ksu == 2:
        return POM_USER, "U"
    return POM_KERNEL, "K"


@dataclass
class Record:
    """One transfer on the PDtrace data bus (PDO_TType / AD / TEnd /
    TMode)."""
    ttype: int
    ad: int
    tmode: int = 1
    tend: int = 1
    lsm: int = 0


@dataclass
class Packet:
    """What an instruction leaves for the cycle it completes in."""
    pc: int
    inscomp: int
    records: list[Record]
    fcr: int = 0
    im: int = 0
    trig: list[tuple[int, int, int]] = field(default_factory=list)   # (data?, on?, n)


@dataclass
class PDO:
    """The PDtrace interface in one cycle (what the TCB samples)."""
    inscomp: int = NI
    record: Record | None = None
    overflow: bool = False
    fcr: int = 0
    im: int = 0
    tracing: bool = False
    trig: list[tuple[int, int, int]] = field(default_factory=list)
    asleep: bool = False


class PDtrace:
    """The core's PDtrace module (see the module docstring)."""

    def __init__(self, m) -> None:
        self.m = m
        #: a 64-bit core (MIPS64): 64-bit PCs and addresses — full values of
        #: two 32-bit AD parts, or one when the value is a sign-extended word
        #: (MD00439 6.4, early TEnd); TMOAS.ISAM = 01
        self.wide = bool(getattr(m, "wide", False))
        self.mask = M64 if self.wide else M32
        #: the TCB on the core's TAP (set by TAP.processor_reset), or None
        self.tcb: TCB | None = None
        self.fifo: deque[Record] = deque()
        #: anything to do each cycle (refresh): keeps an idle core cheap
        self.active = False
        self.reset()

    def reset(self) -> None:
        self.fifo.clear()
        self.restart = True              # the next traced instruction: TMOAS + full PC
        self.tracing = False             # the last instruction was traced
        self.last_pc: int | None = None  # PC of the last traced instruction
        self.last_la: int | None = None
        self.last_sa: int | None = None
        self.mode: tuple[int, int] | None = None   # (POM, ASID) last traced
        self.sync_left = 0
        self.sync_due = False
        self.ctx: tuple | None = None    # what predicts the next PC
        self.trigger_on = False          # TriggerOn (EQ 2)
        self.overflowed = False
        self.packet: Packet | None = None
        self.tmoas_due = False           # an EntryHi write: a TMOAS

    def refresh(self) -> None:
        """Recompute :attr:`active` after a control changed: trace on, a
        FIFO to drain, a TCB that records or waits for a trigger."""
        tc = self.m.cp0[TRACECONTROL]["value"]
        t = self.tcb
        on = bool(tc & TC_ON) if tc & TC_TS else (t is not None and bool(t.control_a & TA_ON))
        self.active = on or bool(self.fifo) or (t is not None and (
            t.enabled or t.trig_in or any(x & (TR_DM | TR_CHTRI | TR_PDTRI) for x in t.trig)))

    # -- the controls (TraceControl or TCBCONTROLA, by TraceControl.TS) --------------

    def _controls(self) -> tuple[int, ...]:
        m = self.m
        tc = m.cp0[TRACECONTROL]["value"]
        if tc & TC_TS:                   # software control
            mode = (m.cp0[TRACECONTROL2]["value"] >> TC2_MODE_SHIFT) & 0x1F
            return (tc & TC_ON, tc & TC_G, (tc >> TC_ASID_SHIFT) & 0xFF,
                    (tc >> TC_ASIDM_SHIFT) & 0xFF,
                    {"U": tc & TC_U, "S": tc & TC_S, "K": tc & TC_K, "E": tc & TC_E,
                     "D": tc & TC_D}, mode | MODE_PC, tc & TC_TB, tc & TC_IO,
                    tc & TC_TFCR, tc & TC_TIM, tc & TC_TLSM)
        t = self.tcb
        if t is None:
            return (0, 0, 0, 0, {}, MODE_PC, 0, 0, 0, 0, 0)
        a = t.control_a
        mode = (t.control_c >> TC_MODE_SHIFT) & 0x1F
        return (a & TA_ON, a & TA_G, (a >> TA_ASID_SHIFT) & 0xFF, 0,
                {"U": a & TA_U, "S": a & TA_S, "K": a & TA_K, "E": a & TA_E, "D": a & TA_D},
                mode | MODE_PC, a & TA_TB, a & TA_IO, a & TA_TFCR, a & TA_TIM, a & TA_TLSM)

    def settings(self) -> dict[str, Any]:
        """The controls in force, by name (views, regeneration)."""
        on, g, asid, asid_m, modes, mode, tb, io, tfcr, tim, tlsm = self._controls()
        return {"ts": bool(self.m.cp0[TRACECONTROL]["value"] & TC_TS), "on": bool(on),
                "g": bool(g), "asid": asid, "asid_mask": asid_m,
                "modes": "".join(k for k, v in modes.items() if v), "mode": mode,
                "tb": bool(tb), "io": bool(io), "fcr": bool(tfcr), "im": bool(tim),
                "lsm": bool(tlsm), "sync": self.sync_period}

    @property
    def on(self) -> bool:
        """TraceOn: the master switch (TraceControl.On or TCBCONTROLA.On)."""
        return bool(self._controls()[0])

    @property
    def sync_period(self) -> int:
        """2^(5 + SyP) cycles; SyP from TCBCONTROLA (TraceControl2.SyP
        mirrors it)."""
        syp = (self.tcb.control_a >> TA_SYP_SHIFT) & 7 if self.tcb is not None else 0
        return 1 << (5 + syp)

    # -- an instruction executes ---------------------------------------------------------

    def capture(self, pc: int, dec, word: int, result, accesses: list, in_slot: bool,
                mode: tuple[int, str], asid: int, triggers: list,
                controls: tuple | None = None) -> Packet | None:
        """Decide what the instruction at ``pc`` (executed in ``mode``, with
        the trace ``controls`` it started with) puts in the trace: a
        :class:`Packet`, or None when it is not traced (MD00439 3.1-3.8,
        EQ 1 / EQ 2)."""
        on, g, tasid, tasid_m, modes, tmode, tb, io, tfcr, tim, tlsm = \
            controls if controls is not None else self._controls()
        name = dec.name if dec is not None else ""
        # EQ 2 (simplified to the 24K's 11.10.2 / 11.10.3): breakpoint triggers
        enable = disable = False
        for _data, start, _n in triggers:
            enable = enable or bool(start)
            disable = disable or not start
        match = bool(on) and (bool(g) or ((tasid ^ asid) & ~tasid_m & 0xFF) == 0) and \
            bool(modes.get(mode[1]))
        traced = bool(on) and (match or enable or (self.trigger_on and not disable))
        self.trigger_on = bool(on) and (enable or (self.trigger_on and not disable))
        packet = None
        pom = (mode[0], asid)
        if traced:
            packet = self._packet(pc, name, word, result, accesses, in_slot, pom, tmode,
                                  tb, tfcr)
            packet.trig = triggers
        elif self.tracing:                         # leaving a traced mode: a TMOAS last
            self.fifo.append(Record(TMOAS, self._tmoas(pom, sync=False)))
            self.restart = True
            if triggers:
                packet = Packet(pc, NI, [], trig=triggers)
        elif triggers:
            packet = Packet(pc, NI, [], trig=triggers)
        self.tracing = traced
        self._follow(pc, name, word, result, in_slot)
        return packet

    def _packet(self, pc: int, name: str, word: int, result, accesses: list, in_slot: bool,
                pom: tuple[int, int], tmode: int, tb: int, tfcr: int) -> Packet:
        is_load = any(a[0] == "load" for a in accesses) or name in _LOADS \
            or name in ("LL", "LLE", "LL@R6")
        is_store = any(a[0] == "store" for a in accesses) or name in _STORES \
            or name.startswith("SC")
        records: list[Record] = []
        if self.restart:                           # start / restart (3.4, 3.6)
            records.append(Record(TMOAS, self._tmoas(pom, sync=False)))
            records += self._full(TPC, pc)
            branched = True
            self.restart = False
            self.last_la = self.last_sa = None
            self.sync_left = self.sync_period
            self.sync_due = False
        else:
            branched = self.last_pc is None or pc != (self.last_pc + 4) & self.mask
            if pom != self.mode or self.tmoas_due:      # a mode / ASID change (3.8)
                records.append(Record(TMOAS, self._tmoas(pom, sync=False)))
            if branched and (tb or not self._predicted(pc)):
                records += self._pc_record(pc)
        self.tmoas_due = False
        self.mode = pom
        if self.sync_due and not branched and not is_load and not is_store \
                and name not in FUNCTION_CR and not self._is_cti_name(name):
            inscomp = IPC                          # periodic synchronisation (3.5)
            records.append(Record(TMOAS, self._tmoas(pom, sync=True)))
            records += self._full(TPC, pc)
            self.sync_due = False
            self.sync_left = self.sync_period
            self.last_la = self.last_sa = None
        elif is_load:
            inscomp = ILB if branched else IL
        elif is_store:
            inscomp = ISB if branched else IS
        else:
            inscomp = IB if branched else I
        for kind, addr, size, value in accesses:
            addr = self.m._bp_va(addr)             # (MIPS64: the 64-bit address)
            if kind == "load" and tmode & MODE_LA:
                records += self._addr_record(TLA, addr)
            elif kind == "store" and tmode & MODE_SA:
                records += self._addr_record(TSA, addr)
            if kind == "load" and tmode & MODE_LD or kind == "store" and tmode & MODE_SD:
                records += _split(TD, value, size)
        self.last_pc = pc
        return Packet(pc, inscomp, records, fcr=int(bool(tfcr) and name in FUNCTION_CR))

    @staticmethod
    def _is_cti_name(name: str) -> bool:
        return name[:1] in ("B", "J") and name not in ("BREAK",) or name in _RETURNS

    def _tmoas(self, pom: tuple[int, int], *, sync: bool) -> int:
        """A TMOAS record (Figure 3.3): SYNC, PendL 0 (loads complete in
        order here), ISAM 00 (MIPS32; 01 MIPS64), POM, ASID (0 without a
        TLB)."""
        asid = pom[1] if getattr(self.m.mmu, "mt", 3) in (1, 4) else 0
        return (int(sync) << 15) | (int(self.wide) << 11) | (pom[0] << 8) | asid

    def _full(self, ttype: int, value: int, tmode: int = 1) -> list[Record]:
        """A full PC / address (TMode = 1), or a signed delta (TMode = 0),
        on the 32-bit AD bus: low bits first (6.1); a 64-bit value that is
        a sign-extended word goes as one part, its TEnd early (6.4)."""
        value &= self.mask
        if not self.wide or value < 0x80000000 or value >= 0xFFFFFFFF80000000:
            return [Record(ttype, value & M32, tmode=tmode)]
        return [Record(ttype, value & M32, tmode=tmode, tend=0),
                Record(ttype, value >> 32, tmode=tmode)]

    def _pc_record(self, pc: int) -> list[Record]:
        """TPC: the delta from the last traced PC, halved (6.1), when it
        fits in 16 bits, else the full PC."""
        if self.last_pc is not None:
            bits = 64 if self.wide else 32
            delta = ((pc - self.last_pc) & self.mask)
            if delta >> (bits - 1):
                delta -= 1 << bits
            delta >>= 1
            if -(1 << 15) <= delta < (1 << 15):
                return [Record(TPC, delta & M32, tmode=0)]
        return self._full(TPC, pc)

    def _addr_record(self, ttype: int, addr: int) -> list[Record]:
        """TLA / TSA: the first after a start or a sync full, then the
        delta from the last address of that kind (6.2)."""
        last = self.last_la if ttype == TLA else self.last_sa
        if ttype == TLA:
            self.last_la = addr
        else:
            self.last_sa = addr
        if last is None:
            return self._full(ttype, addr)
        return self._full(ttype, addr - last, tmode=0)

    def _predicted(self, pc: int) -> bool:
        """Can trace regeneration work out ``pc`` from the program?  After
        a taken branch / jump with an encoded target (its delay slot
        completed), a taken compact branch, a branch-likely whose slot was
        nullified — and only then (3.1)."""
        ctx = self.ctx
        if ctx is None:
            return False
        if ctx[0] == "after_slot" or ctx[0] == "compact":
            return pc == ctx[1]
        if ctx[0] == "likely":                     # its slot did not complete
            return pc == ctx[2]
        return False

    def _follow(self, pc: int, name: str, word: int, result, in_slot: bool) -> None:
        """Keep what predicts the next PC (the same rules as
        :func:`regenerate`)."""
        ctx = self.ctx
        if ctx is not None and ctx[0] in ("slot", "likely") and pc == ctx[3]:
            self.ctx = ("after_slot", ctx[1]) if ctx[1] is not None else None
            return
        if not name or name in _RETURNS:
            self.ctx = None
            return
        target = None if name in _REGISTER_TARGET else static_target(name, word, pc,
                                                                      self.mask)
        if name.startswith(("BC", "BEQZC", "BNEZC", "BEQC", "BNEC", "BLTC", "BGEC", "BLTUC",
                            "BGEUC", "BOVC", "BNVC", "BLEZC", "BGTZC", "BLTZC", "BGEZC",
                            "BEQZALC", "BNEZALC", "BLEZALC", "BGEZALC", "BGTZALC",
                            "BLTZALC", "BALC")) and not name.startswith(("BC1", "BC2")):
            self.ctx = ("compact", target)
        elif name in ("JIC", "JIALC"):
            self.ctx = None
        elif _is_likely(name) or name in ("BC1FL", "BC1TL", "BC2FL", "BC2TL"):
            self.ctx = ("likely", target, (pc + 8) & self.mask, (pc + 4) & self.mask)
        elif self._is_cti_name(name) and name not in _RETURNS:
            self.ctx = ("slot", target, None, (pc + 4) & self.mask)
        else:
            self.ctx = None

    # -- other sources of trace -----------------------------------------------------------

    def user_data(self, value: int) -> None:
        """A write to UserTraceData: a TU1 / TU2 record (TraceControl.UT),
        sent whenever TraceOn — even in a mode not traced (11.7.6.1)."""
        if self.on:
            ut = self.m.cp0[TRACECONTROL]["value"] & TC_UT
            self.fifo.append(Record(TU2 if ut else TU1, value & M32))

    def entryhi_written(self) -> None:
        if getattr(self.m.mmu, "mt", 3) in (1, 4):
            self.tmoas_due = True

    # -- the cycle -------------------------------------------------------------------

    def clock(self, packet: Packet | None, *, asleep: bool = False) -> None:
        """One core cycle: ``packet`` = the instruction completing now (or
        None); one record leaves the FIFO; the TCB samples the interface
        (``asleep``: in WAIT, no TF1, 4.1.1)."""
        pdo = PDO(asleep=asleep)
        if packet is not None:
            pdo.inscomp, pdo.fcr, pdo.im, pdo.trig = (packet.inscomp, packet.fcr, packet.im,
                                                      packet.trig)
            self.fifo.extend(packet.records)
        if len(self.fifo) > FIFO_DEPTH and not self._controls()[7]:
            self.fifo.clear()                      # overflow (3.6): TF5, restart
            self.restart = True
            pdo = PDO(overflow=True, tracing=True)
        elif self.fifo:
            pdo.record = self.fifo.popleft()
        pdo.tracing = (self.tracing or pdo.overflow or pdo.inscomp != NI
                       or pdo.record is not None)
        if self.tracing:
            self.sync_left -= 1
            if self.sync_left <= 0:
                self.sync_due = True
        if self.tcb is not None:
            self.tcb.sample(pdo, self.m)
        if not self.fifo:
            self.refresh()

    @property
    def backpressure(self) -> bool:
        """IO (Inhibit Overflow): the FIFO is too full — the core must
        stall a cycle (11.12.2)."""
        return len(self.fifo) > FIFO_DEPTH and bool(self._controls()[7])


def _split(ttype: int, value: int, size: int) -> list[Record]:
    """A value wider than the 32-bit AD bus goes low bits first (6.1);
    TEnd marks the last part."""
    if size <= 4:
        return [Record(ttype, value & M32)]
    return [Record(ttype, value & M32, tend=0), Record(ttype, (value >> 32) & M32)]


# ================================================================ the TCB

class TCB:
    """The Trace Control Block of a core's TAP (see the module docstring)."""

    def __init__(self) -> None:
        #: called after a register write (the core's PDtrace refreshes)
        self.on_change: Callable[[], None] | None = None
        self.power_up()

    def power_up(self) -> None:
        """Test-Logic-Reset / TRST (11.9.11): every register to its reset
        value, the trace memory pointers to 0."""
        self.control_a = TA_VMODES | TA_ADW
        self.control_b = 0
        self.control_c = 0
        self.trig = [0] * TRIGGERS
        self._trig_read = [0] * TRIGGERS        # the special read values
        self.words = TRACE_WORDS * [0]
        self.wrp = self.rdp = self.stp = 0
        self.count = 0                          # trace words held
        self.tw = 0                             # the trace word being built
        self.used = 0                           # its Trace bits used
        self.type = 1                           # its Type field
        self.started = False                    # a TW was begun since EN
        self.notrace = 0                        # no-trace cycles (TF6 0001)
        self.about = None                       # trace words left to an About end
        self.prev_dm = False
        self.trigger_out = 0                    # strobes on TR_TRIGOUT (the probe)
        self.trig_in = False                    # TR_TRIGIN edge waiting
        self.formats = 0                        # TFs written (views)
        self._rm_left = 0                       # words left to read with RM

    # -- the TAP side --------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.control_b & TB_EN)

    @property
    def ca(self) -> bool:
        return bool(self.control_b & TB_CA)

    @property
    def tlsif(self) -> bool:
        return bool(self.control_b & TB_TLSIF)

    def config(self) -> int:
        """TCBCONFIG (Table 11.36): TRIG, SZ, PiN 0, OnT, REV 1."""
        return (TRIGGERS << 21) | (TRACE_SZ << 17) | (1 << 5) | TCB_REV

    def capture(self, instruction: int) -> int:
        """Capture-DR of a TCB instruction."""
        if instruction == TCBCONTROLA:
            return self.control_a
        if instruction == TCBCONTROLB:
            return self.control_b & ~TB_WE
        if instruction == TCBCONTROLC:
            return self.control_c
        reg = (self.control_b >> TB_REG_SHIFT) & 0x1F
        if reg == 0:
            return self.config()
        if reg == 4:
            return self._read_tw()
        if reg == 5:
            return self.rdp
        if reg == 6:
            return self.wrp
        if reg == 7:
            return self.stp
        if 16 <= reg < 16 + TRIGGERS:
            return self._trig_value(reg - 16)
        return 0

    def length(self, instruction: int) -> int:
        if instruction == TCBDATA and (self.control_b >> TB_REG_SHIFT) & 0x1F == 4:
            return 64
        return 32

    def _read_tw(self) -> int:
        """TCBTW: the word at TCBRDP; with RM a read moves TCBRDP on (not
        when WR), and RM clears at the write pointer (11.9.2 / 11.9.6)."""
        if not self.control_b & TB_RM:
            return self.words[self.rdp // 8]
        if self._rm_left <= 0:
            self.control_b &= ~TB_RM
            return 0
        value = self.words[self.rdp // 8]
        if not self.control_b & TB_WR:
            self.rdp = (self.rdp + 8) % (TRACE_WORDS * 8)
            self._rm_left -= 1
            if self._rm_left <= 0:
                self.control_b &= ~TB_RM
        return value

    def update(self, instruction: int, value: int) -> None:
        """Update-DR of a TCB instruction."""
        if instruction == TCBCONTROLA:
            self.control_a = (self.control_a & ~TA_WRITE) | (value & TA_WRITE)
        elif instruction == TCBCONTROLB:
            if not value & TB_WE:                # written only with WE (Table 11.32)
                return
            keep = TB_BF | (self.control_b & TB_RM)
            write = (0x1F << TB_REG_SHIFT) | TB_WR | (3 << TB_TM_SHIFT) | TB_TLSIF \
                | (7 << TB_CR_SHIFT) | TB_CA | TB_EN
            was_en = self.enabled
            self.control_b = (self.control_b & keep) | (value & write)
            if value & TB_TR:                    # trace memory reset
                self.wrp = self.rdp = self.stp = self.count = 0
                self.control_b &= ~(TB_RM | TB_BF)
            if value & TB_RM and not self.control_b & TB_RM:
                self.control_b |= TB_RM          # read from the oldest word
                self.rdp = self.stp
                self._rm_left = self.count
            if was_en and not self.enabled:
                self._flush()
        elif instruction == TCBCONTROLC:
            self.control_c = value & (0x1F << TC_MODE_SHIFT)
        elif instruction == TCBDATA and self.control_b & TB_WR:
            reg = (self.control_b >> TB_REG_SHIFT) & 0x1F
            size = TRACE_WORDS * 8
            if reg == 4:
                self.words[self.rdp // 8] = value & ((1 << 64) - 1)
            elif reg == 5:
                self.rdp = value & (size - 8)
            elif reg == 6:
                self.wrp = value & (size - 8)
            elif reg == 7:
                self.stp = value & (size - 8)
            elif 16 <= reg < 16 + TRIGGERS:
                self.trig[reg - 16] = value & TRIG_WRITE
                self._trig_read[reg - 16] = 0
        if self.on_change is not None:
            self.on_change()

    def _trig_value(self, i: int) -> int:
        """TCBTRIGx as read: after it fired (TR) the source bits say which
        fired it, Type 11 / Trace 0 mark suppressed actions."""
        v = self.trig[i]
        r = self._trig_read[i]
        if v & TR_TR:
            v = (v & ~(TR_DM | TR_CHTRI | TR_PDTRI)) | (r & (TR_DM | TR_CHTRI | TR_PDTRI))
            if r & (3 << TR_TYPE_SHIFT):
                v |= 3 << TR_TYPE_SHIFT
            if r & TR_TRACE:
                v &= ~TR_TRACE
        return v

    # -- trace words -------------------------------------------------------------------

    def _put(self, value: int, length: int) -> None:
        """Add one trace format to the trace words (5.1)."""
        self.formats += 1
        while length:
            room = 60 - self.used
            take = min(room, length)
            self.tw |= (value & ((1 << take) - 1)) << (4 + self.used)
            self.used += take
            value >>= take
            length -= take
            if self.used == 60:
                self._store_tw(self.tw | self.type)
                self.tw = 0
                if length:                      # the rest starts the next word
                    k = -(-length // 4) * 4
                    self.type = 1 + k // 4
                    self.tw |= (value & ((1 << length) - 1)) << 4
                    self.used = k
                    length = 0
                else:
                    self.type = 1
                    self.used = 0

    def _flush(self) -> None:
        """EN off: the last trace word goes out, its free bits zero (the
        end-of-trace marker, 5.2)."""
        if self.used:
            self._store_tw(self.tw | self.type)
        self.tw = self.used = 0
        self.type = 1

    def _store_tw(self, word: int) -> None:
        """Into the on-chip trace memory (11.13): Trace-To wraps, keeping
        the newest; Trace-From stops when full (BF)."""
        tm = (self.control_b >> TB_TM_SHIFT) & 3
        if tm == 1 and self.count >= TRACE_WORDS:
            self.control_b |= TB_BF
            return
        self.words[self.wrp // 8] = word
        self.wrp = (self.wrp + 8) % (TRACE_WORDS * 8)
        if self.count < TRACE_WORDS:
            self.count += 1
        else:
            self.stp = self.wrp                 # wrapped: the oldest is next
        if tm == 1 and self.count >= TRACE_WORDS:
            self.control_b |= TB_BF
        if self.about is not None:
            self.about -= 1
            if self.about <= 0:
                self.about = None
                self.control_b &= ~TB_EN
                self._flush()

    # -- one cycle -------------------------------------------------------------------

    def probe_trigger(self) -> None:
        """A rising edge on TR_TRIGIN (the probe's trigger output)."""
        self.trig_in = True

    def sample(self, pdo: PDO, m) -> None:
        """The cycle's PDtrace output and the trigger sources (11.11)."""
        dm = m.debug_mode
        sources = (TR_DM if dm and not self.prev_dm else 0) | \
            (TR_PDTRI if self.trig_in else 0)
        self.prev_dm = dm
        self.trig_in = False
        before, after = [], []
        was = self.enabled
        if sources:
            before, after = self._fire(sources)
        if not (was or self.enabled):           # not recording: a TF6 has nowhere to go
            return
        for tf in before:                       # start / info: before the cycle's formats
            self._tf6(*tf)
        self._formats(pdo)                      # (an end trigger's cycle is still traced)
        for tf in after:
            self._tf6(*tf)
        if was and not self.enabled:
            self._flush()

    def _fire(self, sources: int) -> tuple[list, list]:
        """The trigger units (11.11.3-11.11.5): the lowest number wins for
        the prioritised actions (EN, TF6)."""
        before, after = [], []
        en_done = trace_done = False
        for i in range(TRIGGERS):
            t = self.trig[i]
            hit = t & sources & (TR_DM | TR_CHTRI | TR_PDTRI)
            if not hit or (t & TR_FO and t & TR_TR):
                continue
            self.trig[i] |= TR_TR
            self._trig_read[i] |= hit
            kind = (t >> TR_TYPE_SHIFT) & 3
            if t & TR_PDTRO:
                self.trigger_out += 1
            if kind != TRIG_INFO:
                if en_done:
                    self._trig_read[i] |= 3 << TR_TYPE_SHIFT     # suppressed
                else:
                    en_done = True
                    if kind == TRIG_START:
                        self.control_b |= TB_EN
                    elif kind == TRIG_END:
                        self.control_b &= ~TB_EN
                    elif self.about is None:
                        self.about = TRACE_WORDS // 2
                    else:
                        self._trig_read[i] |= 3 << TR_TYPE_SHIFT
            if t & TR_TRACE:
                if trace_done:
                    self._trig_read[i] |= TR_TRACE                # suppressed
                else:
                    trace_done = True
                    code = kind << 2
                    item = (code, (t >> 24) & 0xFF)
                    (before if kind in (TRIG_START, TRIG_INFO) else after).append(item)
        return before, after

    def _tf6(self, code: int, info: int) -> None:
        self._put(_strip(0b0100 | (code << 4) | (info << 8), 16, self.ca)[0],
                  16 - (0 if self.ca else 1))

    def _formats(self, pdo: PDO) -> None:
        """The trace formats of one cycle (4.1): TF5 overflow, TF1 stall,
        TF2 instruction, TF3 / TF4 instruction + record; TF6 for no-trace
        cycles and hardware-breakpoint triggers."""
        ca = self.ca
        if not pdo.tracing:                     # PDO_IamTracing = 0
            if ca and self.started:
                self.notrace += 1
                if self.notrace == 256:
                    self._tf6(TF6_NOTRACE, 0)
                    self.notrace = 0
            return
        if self.notrace and ca:
            self._tf6(TF6_NOTRACE, self.notrace & 0xFF)
        self.notrace = 0
        self.started = True
        if pdo.overflow:
            v, n = _strip(0b1100, 4, ca)
            self._put(v, n)
            return
        opt = self.tlsif
        rec = pdo.record
        if rec is None:
            if pdo.inscomp == NI:
                if ca and not pdo.asleep:
                    self._put(1, 1)             # TF1
            else:
                v, n = 0b10 | (pdo.inscomp << 2), 5
                if opt:
                    v |= (pdo.fcr << 5) | (pdo.im << 6)
                    n += 2
                v, n = _strip(v, n, ca)
                self._put(v, n)
        elif rec.ttype == TD and rec.tend:      # TF4: the last part of a data value
            v = (pdo.inscomp << 3) | (1 << 9) | (TD << 6) | (rec.tmode << 10)
            v |= (0 << 11) | ((rec.ad & M32) << (11 + DATAORDER_BITS))
            n = 11 + DATAORDER_BITS + ADW
            if opt:
                v |= (pdo.fcr << n) | (pdo.im << (n + 1)) | (rec.lsm << (n + 2))
                n += 3
            v, n = _strip(v, n, ca)
            self._put(v, n)
        else:                                   # TF3
            v = (pdo.inscomp << 3) | (rec.ttype << 6) | (rec.tend << 9) | (rec.tmode << 10)
            v |= (rec.ad & M32) << 11
            n = 11 + ADW
            if opt:
                v |= (pdo.fcr << n) | (pdo.im << (n + 1))
                n += 2
            v, n = _strip(v, n, ca)
            self._put(v, n)
        for data, start, num in pdo.trig:       # TF6 1001 after the instruction (4.1.6)
            self._tf6(TF6_BREAKPOINT, (data & 1) | ((start & 1) << 1) | ((num & 0xF) << 2))


TRACE_WORDS = 1 << (TRACE_SZ + 8 - 3)


def _strip(value: int, n: int, ca: bool) -> tuple[int, int]:
    """Without cycle accuracy bit 0 of every format is left out (5.1.1)."""
    return (value, n) if ca else (value >> 1, n - 1)


# ============================================================ regeneration

@dataclass
class Format:
    """One decoded trace format."""
    tf: int                     # 1..6
    inscomp: int = NI
    ttype: int = NT
    tend: int = 1
    tmode: int = 1
    ad: int = 0
    fcr: int = 0
    im: int = 0
    lsm: int = 0
    code: int = 0               # TF6
    info: int = 0


def format_text(f: Format) -> str:
    """One format as the debugger prints it."""
    if f.tf == 1:
        return "TF1 stall"
    if f.tf == 5:
        return "TF5 overflow"
    if f.tf == 6:
        return f"TF6 TCBcode {f.code:04b} TCBinfo 0x{f.info:02x}"
    text = f"TF{f.tf} {INSCOMP_NAMES[f.inscomp]}"
    if f.tf in (3, 4):
        text += (f" {TTYPE_NAMES[f.ttype]} 0x{f.ad:08x}" + ("" if f.tmode else " (delta)")
                 + ("" if f.tend else " (more)"))
    flags = [n for v, n in ((f.fcr, "FCR"), (f.im, "Im"), (f.lsm, "LSm")) if v]
    return text + (" " + " ".join(flags) if flags else "")


def _format_length(bits: int, have: int, ca: bool, opt: bool) -> int | None:
    """The length of the format whose first ``have`` bits are ``bits``
    (None: not known yet).  Without CA bit 0 was removed: the layout is
    read with it put back as 0."""
    s = 0 if ca else 1
    v, h = (bits, have) if ca else (bits << 1, have + 1)
    if ca:
        if h < 1:
            return None
        if v & 1:
            return 1                            # TF1
    if h < 2:
        return None
    if (v >> 1) & 1:
        return 5 + (2 if opt else 0) - s        # TF2
    if h < 3:
        return None
    if (v >> 2) & 1:                            # TF5 / TF6
        if h < 4:
            return None
        return (4 if (v >> 3) & 1 else 16) - s
    if h < 10:
        return None
    if ((v >> 6) & 7) == TD and (v >> 9) & 1:   # TF4
        return 11 + DATAORDER_BITS + ADW + (3 if opt else 0) - s
    return 11 + ADW + (2 if opt else 0) - s     # TF3


def _parse(bits: int, n: int, ca: bool, opt: bool) -> Format:
    v = bits if ca else bits << 1               # the full layout (bit 0 = 0)
    if ca and bits & 1:
        return Format(1)
    if (v >> 1) & 1:
        f = Format(2, inscomp=(v >> 2) & 7)
        if opt:
            f.fcr, f.im = (v >> 5) & 1, (v >> 6) & 1
        return f
    if (v >> 2) & 1:
        if (v >> 3) & 1:
            return Format(5)
        return Format(6, code=(v >> 4) & 0xF, info=(v >> 8) & 0xFF)
    ins, tt, tend, tmode = (v >> 3) & 7, (v >> 6) & 7, (v >> 9) & 1, (v >> 10) & 1
    if tt == TD and tend:
        base = 11 + DATAORDER_BITS
        f = Format(4, ins, tt, tend, tmode, (v >> base) & M32)
        if opt:
            k = base + ADW
            f.fcr, f.im, f.lsm = (v >> k) & 1, (v >> (k + 1)) & 1, (v >> (k + 2)) & 1
        return f
    f = Format(3, ins, tt, tend, tmode, (v >> 11) & M32)
    if opt:
        k = 11 + ADW
        f.fcr, f.im = (v >> k) & 1, (v >> (k + 1)) & 1
    return f


def decode_words(words: list[int], *, ca: bool = True, opt: bool = False) -> list[Format]:
    """Trace words (oldest first) -> trace formats (5.1, Table 5.1).  A
    run of 9 zero bits ends the trace (5.2)."""
    out: list[Format] = []
    part, plen = 0, 0                           # an unfinished format
    for w in words:
        typ = w & 0xF
        field_ = w >> 4
        if typ == 0:
            continue
        pos = 0
        if typ == 1:
            part, plen = 0, 0
        elif typ <= 14:
            k = (typ - 1) * 4
            if plen:                            # finish the straddling format
                total = None
                cont = field_ & ((1 << k) - 1)
                bits = part | (cont << plen)
                for have in range(plen, plen + k + 1):
                    total = _format_length(bits, have, ca, opt)
                    if total is not None:
                        break
                if total is not None and total <= plen + k:
                    out.append(_parse(bits & ((1 << total) - 1), total, ca, opt))
            part, plen = 0, 0
            pos = k
        while pos < 60:
            rest = field_ >> pos
            have = 60 - pos
            marker = 9 if ca else 8             # an empty TF3: the end of the trace (5.2)
            if have >= marker and rest & ((1 << marker) - 1) == 0:
                break
            total = None
            for h in range(1, have + 1):
                total = _format_length(rest, h, ca, opt)
                if total is not None:
                    break
            if total is None or total > have:
                part, plen = rest & ((1 << have) - 1), have
                break
            out.append(_parse(rest & ((1 << total) - 1), total, ca, opt))
            pos += total
    return out


@dataclass
class Step:
    """An executed instruction, regenerated."""
    pc: int
    inscomp: int
    load: int | None = None
    store: int | None = None
    data: list[int] = field(default_factory=list)
    mode: int | None = None
    asid: int | None = None
    sync: bool = False
    fcr: int = 0
    im: int = 0
    notes: list[str] = field(default_factory=list)


def regenerate(formats: list[Format], code_word: Callable[[int], int | None], isa,
               *, tb: bool = False, mode: int = MODE_PC,
               vector: Callable[[int], bool] = lambda a: False,
               wide: bool = False) -> list[Any]:
    """Trace regeneration: the formats + the static program image ->
    the executed instructions (:class:`Step`) and events (strings: user
    data, triggers, overflows).  ``tb`` / ``mode``: the trace settings
    used; ``vector``: the exception vectors (an exception entry in a
    place the program predicts is recognised by its TPC — a decoder
    rule, PDtrace does not mark exceptions); ``wide``: a 64-bit core's
    trace (64-bit PCs and addresses; a value sent in one 32-bit part is
    sign-extended, 6.4)."""
    mask = M64 if wide else M32
    ins: list[tuple[int, Format]] = []          # (position in the stream, format)
    recs: list[Format] = []
    where: list[int] = []                       # each record's position
    for n, f in enumerate(formats):
        if f.tf in (2, 3, 4) and f.inscomp != NI or f.tf in (5, 6):
            ins.append((n, f))
        if f.tf in (3, 4) and f.ttype != NT:
            recs.append(f)
            where.append(n)
    out: list[Any] = []
    q = 0                                       # the next record
    #: an overflow (TF5) drops the records still in the FIFO: an instruction
    #: before it never takes a record from after it (``bound``)
    overflows = [n for n, f in enumerate(formats) if f.tf == 5]
    bound = len(formats)
    pc: int | None = None
    ctx: tuple | None = None
    cur_mode: tuple[int, int] | None = None
    last_la = last_sa = None

    def take(kinds: tuple[int, ...]) -> Format | None:
        """The next record of these kinds, applying TMOAS / TU on the way."""
        nonlocal q, cur_mode
        while q < len(recs) and where[q] < bound:
            r = recs[q]
            if r.ttype in kinds:
                q += 1
                return r
            if r.ttype == TMOAS:
                q += 1
                cur_mode = ((r.ad >> 8) & 7, r.ad & 0xFF)
                out.append(f"mode {POM_NAMES.get(cur_mode[0], '?')}, ASID {cur_mode[1]}"
                           + (" (sync)" if r.ad >> 15 & 1 else ""))
                continue
            if r.ttype in (TU1, TU2):
                q += 1
                out.append(f"user data {'TU1' if r.ttype == TU1 else 'TU2'} 0x{r.ad:08x}")
                continue
            return None
        return None

    def value(r: Format, kinds: tuple[int, ...]) -> int:
        v, shift = r.ad, 32
        while not r.tend:
            r = take(kinds)
            if r is None:
                break
            v |= r.ad << shift
            shift += 32
        return v

    def parts(r: Format, kinds: tuple[int, ...]) -> tuple[int, int]:
        """A PC / address value and its width in bits (32 per part)."""
        v, shift = r.ad, 32
        while not r.tend:
            r = take(kinds)
            if r is None:
                break
            v |= r.ad << shift
            shift += 32
        return v, shift

    def full(v: int, bits: int) -> int:
        """A full value: on a 64-bit core sign-extended from its width."""
        return _sbits(v, bits) & mask if wide else v

    def delta(v: int, bits: int) -> int:
        return _sbits(v, bits)

    def peek_pc(at: int) -> Format | None:
        """The next TPC if this instruction (at position ``at``) sent it.
        The FIFO pops one record a cycle, in order, in the cycle of the
        push when it is empty: the next record is ours when it is at
        ``at`` (the FIFO was empty) or right after the previous record
        while that one was still queued at ``at``."""
        if q >= len(recs):
            return None
        prev = where[q - 1] if q else -1
        if not (where[q] == at if prev < at else where[q] == prev + 1):
            return None
        k = q
        while k < len(recs) and recs[k].ttype in (TMOAS, TU1, TU2):
            k += 1
        if k >= len(recs) or recs[k].ttype != TPC or where[k] - where[q] != k - q:
            return None
        return recs[k]

    for idx, f in ins:
        bound = next((n for n in overflows if n > idx), len(formats))
        if f.tf == 5:
            while q < len(recs) and where[q] < idx:     # what the overflow dropped the
                if take(()) is None:                    # rest of: TMOAS / TU still apply
                    q += 1
            out.append("overflow: trace lost, restarts")
            pc, ctx = None, None
            continue
        if f.tf == 6:
            names = {TF6_START: "trigger start", TF6_END: "trigger end",
                     TF6_INFO: "trigger info", TF6_NOTRACE: "no-trace cycles",
                     TF6_BACKSTALL: "back-stall cycles"}
            if f.code == TF6_BREAKPOINT:
                out.append(f"{'data' if f.info & 1 else 'instruction'} breakpoint "
                           f"{(f.info >> 2) & 0xF} turned trace {'on' if f.info & 2 else 'off'}")
            else:
                out.append(f"{names.get(f.code, f'TF6 code {f.code:04b}')} "
                           f"(TCBinfo 0x{f.info:02x})")
            continue
        branched = f.inscomp in (IB, ILB, ISB)
        if f.inscomp == IPC:                    # never a branch: sequential, then
            r = take((TPC,))                    # checked by the full PC (3.5)
            if r is not None:
                pc = full(*parts(r, (TPC,)))
            elif pc is not None:
                pc = (pc + 4) & mask
            step = Step(pc if pc is not None else 0, f.inscomp, sync=True)
        elif branched:
            predicted = None
            if ctx is not None and not tb:
                if ctx[0] in ("after_slot", "compact") and ctx[1] is not None:
                    predicted = ctx[1]
                elif ctx[0] == "likely":
                    predicted = ctx[2]
            r = None
            if predicted is not None and pc is not None:
                head = peek_pc(idx)
                if head is not None:                # (one part: early TEnd or MIPS32)
                    guess = full(head.ad, 32) if head.tmode else \
                        (pc + (_s32(head.ad) << 1)) & mask
                    if vector(guess) or not head.tend:
                        r = take((TPC,))
            else:
                r = take((TPC,))
            if r is not None:
                v, bits = parts(r, (TPC,))
                if r.tmode:
                    pc = full(v, bits)
                elif pc is not None:            # a delta needs a known PC
                    pc = (pc + (delta(v, bits) << 1)) & mask
            elif predicted is not None:
                pc = predicted
            elif pc is not None:                # its TPC is not in the trace (yet: the
                pc, ctx = None, None            # FIFO): lost until the next full PC
                out.append("PC not in the trace: lost until the next full PC")
            step = Step(pc if pc is not None else 0, f.inscomp)
        else:
            pc = (pc + 4) & mask if pc is not None else None
            step = Step(pc if pc is not None else 0, f.inscomp)
        step.fcr, step.im = f.fcr, f.im
        if cur_mode is not None:
            step.mode, step.asid = cur_mode
        if f.inscomp in (IL, ILB) or f.inscomp in (IS, ISB):
            load = f.inscomp in (IL, ILB)
            if mode & (MODE_LA if load else MODE_SA):
                kind = TLA if load else TSA
                r = take((kind,))
                if r is not None:
                    last = last_la if load else last_sa
                    v, bits = parts(r, (kind,))
                    a = full(v, bits) if r.tmode else None if last is None else \
                        (last + delta(v, bits)) & mask
                    if load:
                        step.load = last_la = a
                    else:
                        step.store = last_sa = a
            if mode & (MODE_LD if load else MODE_SD):
                r = take((TD,))
                if r is not None:
                    step.data.append(value(r, (TD,)))
        if pc is not None:
            out.append(step)
            try:
                word = code_word(pc)
            except Exception:                   # noqa: BLE001  no code there
                word = None
            name = ""
            if word is not None:
                try:
                    name = isa.decode(word).name
                except Exception:               # noqa: BLE001
                    name = ""
            ctx = _next_ctx(ctx, pc, name, word or 0, mask)
        if f.inscomp == IPC:
            last_la = last_sa = None
    bound = len(formats)
    while q < len(recs):                        # TMOAS / TU left at the end
        if take(()) is None:
            q += 1
    return out


def exception_vectors(m) -> Callable[[int], bool]:
    """The places an exception can enter on ``m`` (Vol. III 6.2.1; the
    debug vector; vectored interrupts up to 64 x 0x200): what
    :func:`regenerate` takes as ``vector`` (as the PC holds them: MIPS64
    sign-extended)."""
    mask = getattr(m, "amask", M32)
    ebase = m.cp0_read(15, 1) & ~0xFFF & mask
    bev = getattr(m, "bev_base", 0xBFC00200) & M32
    bases = {m.to_pc(b) if b <= M32 else b
             for b in (ebase, ebase ^ 0x20000000, 0x80000000, 0xA0000000, bev)}
    fixed = {m.to_pc(m.debug_vector() & M32), m.to_pc((bev - 0x200) & M32)}

    def vector(a: int) -> bool:
        if a in fixed:
            return True
        for b in bases:
            off = (a - b) & mask
            if off in (0, 0x100, 0x180) or 0x200 <= off < 0x200 + 64 * 0x200 and not off & 31:
                return True
        return False
    return vector


def _s32(v: int) -> int:
    return v - (1 << 32) if v >> 31 else v


def _sbits(v: int, bits: int) -> int:
    """``v`` (``bits`` wide) as a signed number."""
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


def _next_ctx(ctx: tuple | None, pc: int, name: str, word: int,
              mask: int = M32) -> tuple | None:
    """The decoder's twin of :meth:`PDtrace._follow`."""
    probe = PDtrace.__new__(PDtrace)
    probe.ctx = ctx
    probe.mask = mask
    PDtrace._follow(probe, pc, name, word, None, False)
    return probe.ctx
