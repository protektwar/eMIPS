"""The MIPS 24K pipeline (PLAN M10 phase 2): a second pipeline model next
to the textbook 5-stage one (:mod:`emips.pipeline`).

Reference: *MIPS32 24K Processor Core Family Software User's Manual*
(MD00343) chapter 2 "Pipeline of the 24K Core", 3.8 (the FPU's latencies,
Table 3.33), 6.2.25 (Config7); *Programming the MIPS32 24K Core Family*
(MD00355) 1.6, 3.7, B.4.5.

Stages (MIPS16e's IR / IK are not used: no MIPS16e)::

    IF  IS  IT  RF  AG  EX  MS  ER  WB
    |-- IFU --| |-------- ALU / LSU --------|

* **The fetch unit runs ahead.**  Each cycle IF fetches a 64-bit bundle
  (the two instructions of an aligned doubleword, one when the fetch
  address is its second word); IS detects the hit and predicts; IT holds
  the 8-entry **instruction buffer** and is bypassed while it is empty, so
  an instruction can be in RF two cycles after its IF.  The IFU stops when
  the buffer is full.  An I-cache miss stops it until the line comes
  (``emips.timing``); an ITLB miss costs ``UTLB`` cycles.
* **Branch prediction** (2.2.1, 2.2.2): conditional branches by a 512-entry
  bimodal **BHT** (two-bit counters indexed by VA 11:3; branch-likely too);
  unconditional ones (``B``, ``BAL``, ``J``, ``JAL``) statically taken;
  ``JR $31`` by the 4-entry **return prediction stack** (pushed by ``JAL``,
  ``JALR $31``, ``BGEZAL``).  Other jumps through a register are not
  predicted: the IFU waits after the delay slot until the jump has
  executed.  The target is computed in IT: a predicted-taken branch fetches
  its target three cycles after its own IF (one or two bubbles in the
  fetch stream, which the instruction buffer usually hides).  Config7.BP /
  RPS / BHT switch prediction off as on the 24K.
* **Branches resolve in EX**; a misprediction kills the younger
  instructions (the delay slot excepted) and the fetch restarts in the
  cycle after MS: a four-cycle bubble (Figures 2.5, 2.6).
* **The skewed ALU** (2.5, Tables 2.5 / 2.6): the ALU is in EX, the
  address adder in AG.  A result can be used in EX by the next instruction
  (EX -> AG bypass) but as a load / store base one cycle later; load data
  is there two cycles after the load's EX (one bubble before an ALU user,
  two before an address user); MFC0 / RDHWR results one cycle later still;
  MUL / MFHI / MFLO results four cycles after EX (Table 2.6).  A store's
  data and every other operand are needed in EX, the base / index in AG.
* **Execution is in EX, in program order** — the Machine's executors, run
  once per instruction, as in the 5-stage model; the pipeline only adds
  timing.  An instruction that raises an exception does not change state;
  the exception is taken when it reaches **ER** (the younger instructions,
  which have not executed, are killed).  An interrupt is attached to the
  instruction about to enter EX and taken in ER too.
* Memory: a load / store waiting for memory holds MS and everything
  behind it (blocking); multiply / divide and FPU latencies are those of
  Tables 2.2 and 3.33 (divide with early-in, by operand size and sign).
  The MDU / FPU / LSU structures (repeat rates, the MDU busy interlock,
  non-blocking misses with the FSB and LDQ, DTLB misses stalling only the
  LSU) come with M10 phase 3; the CP0 hazards with phase 4 (here, as in
  the Machine, every CP0 write is seen at once).

Every choice is in ``sim.DEVIATIONS`` ("24K pipeline").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import cp0 as C
from . import dsp as DSP
from .disasm import text_at
from .exceptions import FetchError, WatchpointHit
from .isa import Decode, ISADecodeError
from .pipeline import (_HAZARD_EVENTS, _MEM_EVENTS, CycleTrace, History, Info, StageView,
                       _pop_run, _rname, analyze, enter, fetch_compressed,
                       is_accumulator, mark_slot, stage_word)
from .sim import NULLIFY, Machine, _Arch
from .syntax import kind_of, operand_spec
from .timing import MEMORY_TIMING

M32 = 0xFFFFFFFF
STAGES = ("IF", "IS", "IT", "RF", "AG", "EX", "MS", "ER", "WB")
#: with MIPS16e code: "Three additional stages are conditionally added to
#: the fetch pipeline after the IS stage" (2.1): IR, IK (and IT)
STAGES16 = ("IF", "IS", "IR", "IK", "IT", "RF", "AG", "EX", "MS", "ER", "WB")
BACK = ("RF", "AG", "EX", "MS", "ER", "WB")
#: the instruction buffer (MD00343 1.2.1.5: "8-entry instruction buffer")
IB_ENTRIES = 8
#: instructions the fetch unit holds: the buffer and the IF / IS bundles
FRONT_ENTRIES = IB_ENTRIES + 4
BHT_ENTRIES = 512
RPS_ENTRIES = 4
#: Config7 (MD00343 6.2.25): BP / RPS / BHT disable the predictors, SL
#: makes load misses blocking, ULB uncached loads, NBLSU locks the LSU and
#: ALU pipelines together
CFG7_SL, CFG7_BHT, CFG7_RPS, CFG7_BP = 1, 1 << 1, 1 << 2, 1 << 3
CFG7_ULB, CFG7_NBLSU = 1 << 4, 1 << 5
#: the Load Data Queue (2.3.3: "up to 9 load misses")
LDQ_ENTRIES = 9
#: the fill-store buffer (2.3.3: "up to 8 independent cache lines")
FSB_ENTRIES = 8
#: MUL: "issued at the maximum rate of 3 every 5 cycles" (Table 2.4)
MUL_RATE = (3, 5)
CONFIG7 = (16, 7)

#: fetch -> redirect: a predicted target is fetched in the third cycle
#: after its branch (the target is computed in IT, 2.2.1.1)
TARGET_DELAY = 3
#: a branch resolved in EX redirects the fetch in MS: IF one cycle later
REDIRECT_DELAY = 2


# ------------------------------------------------------------------ latencies

#: Table 3.33 (24Kf FPU): latency in cycles by operation
_FP_LAT = {"ABS": 4, "NEG": 4, "ADD": 4, "SUB": 4, "MOV": 4, "MOVF": 4, "MOVN": 4,
           "MOVT": 4, "MOVZ": 4, "CEIL": 5, "FLOOR": 5, "ROUND": 5, "TRUNC": 5,
           "C": 2, "CMP": 4, "SEL": 4, "SELEQZ": 4, "SELNEZ": 4, "MIN": 4, "MAX": 4,
           "MINA": 4, "MAXA": 4, "CLASS": 4, "RINT": 4}


def fp_latency(name: str) -> int:
    """Cycles until an FPU result can be used (Table 3.33; the Release 6
    operations, which the 24K does not have, as the nearest 24K one)."""
    op, _, rest = name.partition(".")
    fmt = name.rsplit(".", 1)[-1]
    if op in ("MUL", "MADD", "MSUB", "NMADD", "NMSUB", "MADDF", "MSUBF"):
        return 5 if fmt == "D" else 4
    if op in ("DIV", "SQRT"):
        return 32 if fmt == "D" else 17
    if op == "RECIP":
        return 25 if fmt == "D" else 13
    if op == "RSQRT":
        return 35 if fmt == "D" else 17
    if op == "CVT":
        to = rest.split(".")[0]
        if to == "S" and fmt == "D":
            return 6
        return 5 if to in ("W", "L") else 4
    if name in ("LWC1", "LDC1", "LDXC1", "LUXC1", "LWXC1"):
        return 3
    if name in ("MTC1", "MFC1", "MTHC1", "MFHC1", "CTC1", "CFC1"):
        return 2
    return _FP_LAT.get(op, 4)


def fp_repeat(name: str) -> int:
    """Cycles before the FPU takes the next instruction (Table 3.33
    repeat rate: a double multiply passes M1 twice, divide / square root
    / reciprocal iterate in M1)."""
    op = name.split(".")[0]
    fmt = name.rsplit(".", 1)[-1]
    if op in ("MUL", "MADD", "MSUB", "NMADD", "NMSUB", "MADDF", "MSUBF"):
        return 2 if fmt == "D" else 1
    if op in ("DIV", "SQRT"):
        return 29 if fmt == "D" else 14
    if op == "RECIP":
        return 21 if fmt == "D" else 10
    if op == "RSQRT":
        return 31 if fmt == "D" else 14
    return 1


def divide_delay(name: str, rs: int, rt: int) -> int:
    """Table 2.2 / 2.4.2: DIVU 7 / 15 / 23 / 31 cycles by the size of the
    dividend (early-in: its upper 8 / 16 / 24 bits zero); DIV 2 more when
    an operand is negative (the sign-adjust stages)."""
    signed = name in ("DIV", "DIV@R6", "MOD")
    neg = signed and bool((rs | rt) & 0x80000000)
    mag = ((-rs) & M32) if signed and rs & 0x80000000 else rs
    delay = 7 if mag < 1 << 8 else 15 if mag < 1 << 16 else 23 if mag < 1 << 24 else 31
    return delay + (2 if neg else 0)


_MDU_GPR = frozenset({"MUL", "MUL@R6", "MUH", "MULU", "MUHU", "MFHI", "MFLO"})
_DIV_GPR = frozenset({"DIV@R6", "MOD", "DIVU@R6", "MODU"})
_DIV_HILO = frozenset({"DIV", "DIVU"})
_OTHER = frozenset({"MFC0", "RDHWR", "MFTC0"})
_MUL_GPR = frozenset({"MUL", "MUL@R6", "MUH", "MULU", "MUHU"})
_FP_STORES = frozenset({"SWC1", "SDC1", "SWXC1", "SDXC1", "SUXC1"})
_TLB_WRITES = frozenset({"TLBWI", "TLBWR"})
#: instructions that write CP0 state (the producers of Table 2.7)
_CP0_WRITERS = frozenset({"MTC0", "EI", "DI", "TLBR", "TLBP", "TLBWI", "TLBWR", "TLBINV",
                          "TLBINVF", "WRPGPR"})
#: instructions after which no execution hazard is left (2.8.2): EHB waits
#: for the CP0 writes, the others refetch what follows them
_BARRIERS = frozenset({"EHB", "ERET", "JR.HB", "JALR.HB", "DERET"})
_TLB_OPS = frozenset({"TLBP", "TLBR", "TLBWI", "TLBWR", "TLBINV", "TLBINVF"})
_COMPARE, _ENTRYHI = (11, 0), (10, 0)


@dataclass
class Hazard:
    """A CP0 write still on its way (Table 2.7): ``key`` is a register
    ``(reg, sel)`` or ``"tlb"`` (the TLB entries); instructions that
    execute fewer than ``spacing + 1`` cycles after ``ex`` see ``old``."""

    key: Any
    old: Any
    new: Any
    ex: int                      # the producer's EX cycle
    src: Any                     # the register an MTC0 wrote, or the instruction name
    text: str                    # the producer


_WATCH_KEYS = frozenset(C.WATCHLO + C.WATCHHI)


def hazard_spacing(h: Hazard, key: Any, consumer: str) -> int:
    """MD00343 Table 2.7: instructions between producer and consumer."""
    if h.key == "tlb":
        return 3 if consumer == "lsu" else 2       # TLBWR -> load / store 3, TLBP / TLBR 2
    if consumer == "int" and h.src == _COMPARE and key == C.CAUSE:
        return 4                                   # the timer interrupt still seen
    if consumer in ("cop", "dsp") and key == C.STATUS:
        return 4                                   # Status.CU / FR -> coprocessor; Status.MX
        #                                            -> DSP (MD00622 Table 2.10)
    if consumer == "lsu" and key in (_ENTRYHI, C.STATUS, C.DEBUG):
        return 3                                   # EntryHi.ASID, Status.ERL, Debug.LSNM
                                                   # (EJTAG Table 2.6) -> load / store
    if consumer == "srs" and key == C.SRSCTL:
        return 1                                   # SRSCtl.PSS -> RDPGPR / WRPGPR
    if consumer == "fetch" and key in _WATCH_KEYS:
        return 10                                  # WatchHi / WatchLo -> instruction fetch
    return 2                                       # MTC0 -> MFC0, ERET, EI / DI, any other


#: the DSP Module's ase values (emips.dsp)
_DSP_ASE = ("dsp", "dspr2")


def consumer_class(dec: Decode) -> str:
    fam, name = dec.entry["family"], dec.name
    if name in ("RDPGPR", "WRPGPR"):
        return "srs"
    if dec.entry.get("ase") in ("dsp", "dspr2"):
        return "dsp"                             # the DSP Module: Status.MX
    from .sim import FPU_FAMILIES
    if fam in FPU_FAMILIES or fam == "cop2" or name in (
            "MOVF", "MOVT", "PREFX", "LWC1", "LDC1", "SWC1", "SDC1", "LWXC1", "LDXC1",
            "LUXC1", "SWXC1", "SDXC1", "SUXC1"):
        return "cop"
    if _lsu(dec):
        return "lsu"
    return "other"


def _lsu(dec: Decode | None) -> bool:
    """Does the instruction go through the load / store unit?"""
    if dec is None:
        return False
    fam, name = dec.entry["family"], dec.name
    return fam in ("load", "store", "cache") or name.startswith("PREF") or name in (
        "SYNC", "SYNCI", "LWC1", "LDC1", "SWC1", "SDC1", "LWXC1", "LDXC1", "LUXC1",
        "SWXC1", "SDXC1", "SUXC1", "LL", "SC", "LL@R6", "SC@R6", "LLE", "SCE")
_MADD = frozenset({"MADD", "MADDU", "MSUB", "MSUBU"})


@dataclass(frozen=True)
class Timing24:
    """Where one instruction needs its operands and how late its results
    come (``ready`` is added to the cycle the instruction is in EX)."""

    at_ag: frozenset            # read in AG: the address base / index
    at_ex: frozenset            # read in EX: everything else
    writes: frozenset
    ready: int                  # GPR / FPR results: EX cycle + ready
    kind: str                   # "alu", "load", "mfc0", "mdu", "div", "fpu"
    #: the results that come at another time than ``ready`` (the DSP
    #: Module: an EXTR's GPR is late, its DSPControl is not)
    ready_by: dict = field(default_factory=dict)


def timing24(dec: Decode, info: Info) -> Timing24:
    """The 24K timing of an instruction (remembered for the MIPS32 rows: a
    function of the word, the row and its :class:`Info` only)."""
    e = dec.entry
    if e.get("isa") in ("mips16e", "micromips"):
        return _timing24(dec, info)
    key = (dec.word, e["name"], info)
    hit = _TIMED.get(key)
    if hit is None:
        if len(_TIMED) >= 1 << 16:
            _TIMED.clear()
        hit = _TIMED[key] = _timing24(dec, info)
    return hit


_TIMED: dict[tuple, Timing24] = {}


def _timing24(dec: Decode, info: Info) -> Timing24:
    e = dec.entry
    name, fam = e["name"], e["family"]
    w = dec.word
    base: set = set()
    if e.get("isa") in ("mips16e", "micromips"):     # no MIPS32 twin: SP is the base
        if name in ("M16.SAVE", "M16.RESTORE"):
            base.add(("r", 29))
    else:
        for item in operand_spec(e):
            kind = kind_of(item)[0]
            if kind in ("mem", "mem9"):
                base.add(("r", (w >> 21) & 31))
            elif kind == "idx":
                base |= {("r", (w >> 21) & 31), ("r", (w >> 16) & 31)}
    base.discard(("r", 0))
    reads = info.reads_ex | info.reads_id
    at_ag = frozenset(base & reads)
    at_ex = frozenset(reads - at_ag)
    fpu = fam.startswith("fpu") or fam in ("cop1_move",) or name in (
        "LWC1", "LDC1", "LDXC1", "LUXC1", "LWXC1")
    if e.get("ase") in ("dsp", "dspr2"):
        return _timing_dsp(name, at_ag, at_ex, info)
    if name in _OTHER:
        ready, kind = 3, "mfc0"
    elif name in _MDU_GPR:
        ready, kind = 5, "mdu"
    elif name in _DIV_GPR:
        ready, kind = 5, "div"
    elif name in _DIV_HILO:
        ready, kind = 1, "div"
    elif info.mdu_cycles or name in ("MTHI", "MTLO", "MADD", "MADDU", "MSUB", "MSUBU"):
        ready, kind = 1, "mdu"
    elif fpu:
        ready, kind = fp_latency(name), "fpu"
    elif info.ready == "MEM":
        ready, kind = 2, "load"
    else:
        ready, kind = 1, "alu"
    return Timing24(at_ag, at_ex, info.writes, ready, kind)


def _timing_dsp(name: str, at_ag, at_ex, info: Info) -> Timing24:
    """The DSP Module on the 24K model: the 24K family's delays, MD00622
    (1004K) Table 2.5 — a GPR from a multiply or an accumulator read (MUL*,
    EXT*, MFHI / MFLO) 4 cycles, from any other DSP instruction 1; the
    accumulators and DSPControl as the MDU's HI / LO (the pair delays:
    :meth:`Pipeline24K._waits_for`)."""
    other = {r: 1 for r in info.writes if r == "dspctl" or is_accumulator(r)}
    if name in DSP.GPR_MULTIPLIES or name in DSP.ACC_READERS or \
            name in ("MFHI@DSP", "MFLO@DSP"):
        return Timing24(at_ag, at_ex, info.writes, 5, "mdu", other)
    if info.mdu_cycles or name in DSP.ACC_SHIFTS or name in ("MTHI@DSP", "MTLO@DSP"):
        return Timing24(at_ag, at_ex, info.writes, 1, "mdu", other)
    if info.ready == "MEM":                      # LWX / LHX / LBUX: loads
        return Timing24(at_ag, at_ex, info.writes, 2, "load", other)
    return Timing24(at_ag, at_ex, info.writes, 2, "alu", other)


def _dsp_pair(pname: str, name: str) -> int:
    """MD00622 Table 2.5: the extra delay of an accumulator consumer
    ``name`` after the accumulator writer ``pname`` (same accumulator)."""
    if name in DSP.ACC_READERS or name in DSP.ACC_SHIFTS:
        return 3                                 # EXT*, SHILO*, MTHLIP: 3
    if name in DSP.MACS or name in _MADD:
        if pname in DSP.SATURATING_MACS or pname in DSP.ACC_SHIFTS or pname in (
                "MTHI@DSP", "MTLO@DSP", "MTHI", "MTLO"):
            return 1                             # *_SA, HI / LO writes -> MAC: 1
    return 0


# ------------------------------------------------------------------ in flight

@dataclass(eq=False)
class Slot24:
    seq: int
    pc: int
    word: int | None
    dec: Decode | None
    text: str
    info: Info | None
    t24: Timing24 | None
    fetched: int                      # its IF cycle
    avail: int                        # first cycle it can be in RF
    in_slot: bool = False             # a branch delay slot
    forbidden: bool = False           # R6 forbidden slot
    cti: str = ""                     # "branch" / "jump" / "" (control transfer)
    compact: bool = False
    pred: str = ""                    # how the IFU predicted it (cti only)
    pred_next: int | None = None      # what it fetched after the CTI (+ its slot)
    rps: bool = False                 # predicted by the return prediction stack
    fetch_exc: _Arch | None = None
    fetch_error: FetchError | None = None
    fetch_cache: list = field(default_factory=list)
    fetch_tlb: list = field(default_factory=list)
    fetch_runs: list = field(default_factory=list)
    executed: bool = False
    exc: _Arch | None = None          # taken when it reaches ER
    ex_cycle: int | None = None
    ready_at: dict = field(default_factory=dict)   # resource -> first usable cycle
    mem_wait: int = 0
    mem_runs: list = field(default_factory=list)
    mem_why: str = ""
    perf: tuple | None = None
    #: MIPS Trace: what it completes with at WB; its fetch missed the I-cache
    pdt: Any = None
    imiss: bool = False
    mispredicted: bool = False
    #: the next instruction of its stream (its delay slot / forbidden
    #: slot), once fetched — not seq + 1 when several TCs fetch (the 34K)
    next_seq: int | None = None
    #: the stage of every cycle (the chart)
    timeline: dict = field(default_factory=dict)
    #: MIPS16e (as emips.pipeline.Slot): the ISA mode, size, compressed
    #: form, the delay slot's branch; its recode cycle (IR)
    isa: int = 0
    size: int = 4
    insn: Any = None
    back: int = 4
    slot_raw: int | None = None
    decode_error: ISADecodeError | None = None
    recode: int = 0
    #: its last stage view (_sv)
    view: Any = None

    @property
    def addr(self) -> int:
        return self.pc | self.isa

    @property
    def fall(self) -> int:
        """The address after it (and its delay slot: 16 bits in MIPS16e)
        with the ISA mode."""
        if self.compact:
            return (self.pc + self.size) & M32 | self.isa
        return (self.pc + self.size + (2 if self.isa else 4)) & M32 | self.isa


def _sv(s: Slot24 | None) -> StageView | None:
    """The stage view of a slot (the same object while what it shows stays:
    a slot is in a stage view every cycle it is in flight)."""
    if s is None:
        return None
    v = s.view
    if v is None or v.mem_wait != s.mem_wait or v.text is not s.text:
        v = s.view = StageView(s.seq, s.pc, s.text, None, s.fetch_cache, 0, s.mem_wait,
                               s.fetch_tlb, *stage_word(s))
    return v


# ------------------------------------------------------------------ the pipeline

class Pipeline24K:
    """The 24K pipeline around a :class:`~emips.sim.Machine`.  Same use as
    :class:`emips.pipeline.Pipeline`: :meth:`cycle` returns a
    :class:`CycleTrace`; :meth:`run`, :meth:`drain`, :meth:`chart`,
    :attr:`stalls`, :attr:`retired`, :attr:`cycles`.  ``memory`` /
    ``nonblocking`` as there (the memory timing with caches)."""

    STAGES = STAGES
    FETCH_STAGE, MEM_STAGE = "IF", "MS"
    name = "24K"
    #: a debugger watchpoint stopped the instruction leaving AG this cycle
    #: (it stays in AG, unexecuted); the Session raises it after
    watch_hit: WatchpointHit | None = None

    def __init__(self, machine: Machine, *, trace: bool = True,
                 memory: dict[str, int] | None = None, nonblocking: bool = False,
                 **_ignored: Any) -> None:
        if machine.compressed == "micromips":
            raise ValueError("the 24K has no microMIPS (MD00343 1.2.1.11: MIPS16e); the "
                             "microMIPS cores (M14K, microAptiv) have a 5-stage pipeline: "
                             "use the 5-stage model")
        self.m = machine
        machine.tracing = trace
        self.timing = machine.caches.timing if machine.caches is not None else None
        if self.timing is not None:
            # the 24K's loads do not block (2.3.3) unless Config7.SL says so
            self.timing.configure(memory, nonblocking=True)
        self.utlb = machine.mmu if hasattr(machine.mmu, "take") else None
        if self.utlb is not None:
            self.utlb.timing = True
            self.utlb.utlb_cost = (memory or {}).get("UTLB", MEMORY_TIMING["UTLB"])
            self.utlb.take()
        self.nonblocking = nonblocking
        #: the back end; the front end is :attr:`queue` (IF / IS / IT)
        self.stage: dict[str, Slot24 | None] = {s: None for s in BACK}
        self.queue: list[Slot24] = []
        self.cycles = 0
        self.retired = 0
        self.stalls: dict[str, int] = {}
        self.history = History()
        self._seq = 0
        #: the fetch unit (fetch_isa: the ISA mode there, MIPS16e = 1)
        self.fetch_pc = machine.pc
        self.fetch_isa = machine.isa_mode
        #: MIPS16e: the first cycle the recoder (IR) is free
        self.recode_free = 0
        #: SAVE / RESTORE in EX until this cycle (one word per cycle)
        self.macro_until = 0
        self.fetching = True               # False: drain (exit, end of program)
        self.fetch_at = 1                  # first cycle the next fetch may happen
        self.blocked = False               # waiting for a jump to execute (or a fault)
        #: after the next instruction (a delay slot): ("target", pc, cycle)
        #: or ("wait", 0, 0)
        self.after_slot: tuple | None = None
        #: what the next fetched instruction is: "kill" (a nullified delay
        #: slot), "forbidden" (R6)
        self.next_marks: set[str] = set()
        self.last_fetched: Slot24 | None = None
        #: resource -> (first cycle its newest value can be used, producer)
        self.scoreboard: dict = {}
        #: after a misprediction: the last seq before the right path (the
        #: "branch mispredict stall cycles" event counts until it executes)
        self.recovering: int | None = None
        #: the units: first cycle each takes a new instruction
        self.lsu_free = 0                  # after a DTLB miss (the LSU waits alone)
        self.div_free = 0                  # the iterative divider
        self.fpu_free = 0                  # the FPU's M1 stage (repeat rates)
        self.tlb_until = 0                 # TLBWR / TLBWI / TLBR -> LSU, COP0 (Table 2.6)
        self.mul_starts: list[int] = []    # EX cycles of the recent MULs
        self.ldq: list[int] = []           # load misses on their way (data cycle)
        self.ifu_miss_until = 0            # an I-cache miss is being filled
        #: CP0 writes whose effect is not seen by everyone yet (Table 2.7)
        self.hazards: list[Hazard] = []
        self.bht = [1] * BHT_ENTRIES       # weakly not taken (reset: UNDEFINED)
        #: the views: BHT entries used, the last prediction / resolution
        self.bht_used: set[int] = set()
        self.last_prediction: tuple | None = None   # (pc, text, how, next)
        self.last_resolution: tuple | None = None   # (pc, text, taken, right)
        self.rps = [0] * RPS_ENTRIES
        self.rps_top = 0
        #: prediction counters (the views; the 24K's events are in emips.perf)
        self.predicted = self.mispredicts = 0
        self.slots = History(History.FIRST, 2 * History.LAST)   # the slots (the chart)
        self._fetch_notes: list[str] = []

    # -- predictors ------------------------------------------------------------------

    def _bht_index(self, pc: int) -> int:
        return (pc >> 3) & (BHT_ENTRIES - 1)

    def _rps_push(self, addr: int) -> None:
        self.rps_top = (self.rps_top + 1) % RPS_ENTRIES
        self.rps[self.rps_top] = addr & M32

    def _rps_pop(self) -> int:
        addr = self.rps[self.rps_top]
        self.rps_top = (self.rps_top - 1) % RPS_ENTRIES
        return addr

    def _predict(self, s: Slot24) -> tuple[str, int | None]:
        """``(how, next PC after the CTI and its delay slot)``; next =
        None: not predicted (the fetch waits for the jump to execute)."""
        if s.insn is not None:
            return self._predict16(s)
        name = s.dec.name
        w = s.word
        cfg7 = self.m.cp0_read(*CONFIG7)
        size = 4 if s.compact else 8
        fall = (s.pc + size) & M32
        rs, rt, rd = (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
        link = name in ("JAL", "JALX", "BGEZAL", "BAL", "BALC", "JIALC") or (
            name in ("JALR", "JALR.HB") and rd == 31)
        if link:
            self._rps_push(fall)
        if cfg7 & CFG7_BP:
            return "fetch waits (Config7.BP)", None
        if name in ("J", "JAL", "JALX"):          # (JALX: to MIPS16e)
            return "jump", ((s.pc + 4) & 0xF0000000) | ((w & 0x3FFFFFF) << 2) | (name == "JALX")
        if name in ("JR", "JR.HB", "JALR", "JALR.HB", "JIC", "JIALC"):
            ret = rs == 31 and (name in ("JR",) or (name == "JALR" and rd == 0))
            if ret and not cfg7 & CFG7_RPS:
                s.rps = True
                return "return stack", self._rps_pop()
            return "fetch waits (jump register)", None
        target = _branch_target(s)
        unconditional = name in ("B", "BAL", "BC", "BALC") or (
            name in ("BEQ",) and rs == 0 and rt == 0) or (name == "BGEZAL" and rs == 0)
        if unconditional:
            return "static taken", target
        if cfg7 & CFG7_BHT:
            return "static taken (Config7.BHT)", target
        i = self._bht_index(s.pc)
        self.bht_used.add(i)
        taken = self.bht[i] >= 2
        return ("BHT taken", target) if taken else ("BHT not taken", fall)

    def _predict16(self, s: Slot24) -> tuple[str, int | None]:
        """MIPS16e (2.1.3: predicted in IR): the branches (compact) by the
        BHT, B statically, JAL / JALX taken, JR $ra / JRC $ra by the
        return stack (JAL / JALX / JALR / JALRC push), the other jumps
        through a register not predicted.  Addresses carry the ISA mode."""
        from . import mips16 as M16
        name = s.dec.name[4:]
        cfg7 = self.m.cp0_read(*CONFIG7)
        fall = s.fall
        if name in ("JAL", "JALX", "JALR", "JALRC"):
            self._rps_push(fall)
        if cfg7 & CFG7_BP:
            return "fetch waits (Config7.BP)", None
        if name in ("JAL", "JALX"):
            target = M16.jump_target(s.insn, s.pc)
            return "jump", target if name == "JALX" else target | 1
        if name in ("JRRA", "JRCRA"):
            if not cfg7 & CFG7_RPS:
                s.rps = True
                return "return stack", self._rps_pop()
            return "fetch waits (jump register)", None
        if name in ("JR", "JALR", "JRC", "JALRC"):
            return "fetch waits (jump register)", None
        target = M16.branch_target(s.insn, s.pc) | 1
        if name == "B":
            return "static taken", target
        if cfg7 & CFG7_BHT:
            return "static taken (Config7.BHT)", target
        i = self._bht_index(s.pc)
        self.bht_used.add(i)
        return ("BHT taken", target) if self.bht[i] >= 2 else ("BHT not taken", fall)

    def _go(self, target: int) -> None:
        """Fetch from ``target`` next (bit 0: the ISA mode, with MIPS16e)."""
        target &= M32
        if self.m.compressed is not None:
            self.fetch_isa = target & 1
            target &= ~1
        self.fetch_pc = target

    def _bht_update(self, pc: int, taken: bool) -> None:
        i = self._bht_index(pc)
        self.bht[i] = min(self.bht[i] + 1, 3) if taken else max(self.bht[i] - 1, 0)

    # -- the fetch unit -------------------------------------------------------------

    def _fetch(self, t: int) -> list[Slot24]:
        """IF at cycle ``t``: one bundle along the predicted path.  After a
        branch or jump its delay slot is still fetched (in this bundle or
        the next), then :attr:`after_slot` applies: the predicted target
        from cycle ``fetched + 3`` on, or a wait for the jump to execute.
        MIPS16e (2.2.5): the instructions that start in the 64-bit bundle
        (one crossing its end is fetched whole), a bundle every other
        cycle, recoded in IR / IK (:meth:`_recode`)."""
        isa = self.fetch_isa
        if (not self.fetching or self.blocked or t < self.fetch_at
                or len(self.queue) > FRONT_ENTRIES - (4 if isa else 2)):
            return []
        pc = self.fetch_pc
        end = (pc & ~7) + 8
        out: list[Slot24] = []
        stopped = False
        a = pc
        while a < end and self.fetch_isa == isa:
            s = self._fetch_one(a, t)
            out.append(s)
            a = (a + s.size) & M32
            self.fetch_pc = a
            if "kill" in self.next_marks:             # a nullified delay slot
                s.timeline[t] = "IF"
            else:
                self.queue.append(s)
            if "forbidden" in self.next_marks:
                s.forbidden = True
            self.next_marks = set()
            if s.fetch_exc is not None or s.fetch_error is not None \
                    or s.decode_error is not None:
                self.blocked, stopped = True, True    # nothing to fetch past it
                break
            if self.after_slot is not None:           # s is the awaited delay slot
                what, target, at = self.after_slot
                self.after_slot = None
                if what == "wait":
                    self.blocked = True
                else:
                    self._go(target)
                    self.fetch_at = max(at, t + 1)
                stopped = True
                break
            if not s.cti:
                continue
            how, nxt = self._predict(s)
            s.pred, s.pred_next = how, nxt
            self.last_prediction = (s.pc, s.text, how, nxt)
            if nxt == s.fall:
                continue                              # predicted not taken
            if s.compact:                             # no delay slot
                if nxt is None:
                    self.blocked = True
                else:
                    self._go(nxt)
                    self.fetch_at = t + TARGET_DELAY
                stopped = True
                break
            self.after_slot = ("wait", 0, 0) if nxt is None else \
                ("target", nxt, t + TARGET_DELAY)
        if isa:
            self._recode(out, t)
        wait = sum(n for s in out for _, n in s.fetch_runs)
        if any(k == "I-miss" for s in out for k, _ in s.fetch_runs):
            self.ifu_miss_until = t + wait
        if not stopped or wait:
            # MIPS16e: "the processor fetches every other cycle" (2.2.5)
            self.fetch_at = max(self.fetch_at, t + (2 if isa else 1) + wait)
        if out:
            self.last_fetched = out[-1]
        return out

    def _recode(self, out: list[Slot24], t: int) -> None:
        """MIPS16e: the recoder takes 32 bits a cycle (Table 2.1): two
        16-bit instructions, or one when it is extended / JAL(X) or the
        next one is; IR is the cycle after IS, then IK, then RF."""
        cycle = max(t + 2, self.recode_free)
        wait = 0
        i = 0
        while i < len(out):
            pair = out[i].size == 2 and i + 1 < len(out) and out[i + 1].size == 2
            group = out[i:i + 2] if pair else out[i:i + 1]
            for s in group:
                wait += sum(n for _, n in s.fetch_runs)
                s.recode = cycle + wait
                s.avail = s.recode + 2
            cycle += 1
            i += len(group)
        self.recode_free = cycle + wait

    def _fetch_one(self, a: int, t: int) -> Slot24:
        m = self.m
        if self.fetch_isa:
            return self._fetch16(a, t)
        self._seq += 1
        seq = self._seq
        exc = err = None
        word = dec = None
        text = ""
        if a & 3 or not m.address_ok(a) or not m.mips32:
            exc, text = _Arch(C.ADEL, badvaddr=a), "(fetch: address error)"
        else:
            try:
                word = m.fetch_word(a)
            except FetchError as e:
                err, text = e, "(no code here)"
            except _Arch as e:
                exc = e
                text = "(fetch: " + ("TLB refill" if e.info.get("refill") else
                                     "cache error" if e.code == C.CACHE_ERROR else
                                     C.EXC_NAMES.get(e.code, str(e.code))) + ")"
        runs: list = []
        if self.utlb is not None:
            wait, notes = self.utlb.take()
            if wait:
                runs.append(("utlb", wait))
            self._fetch_notes += [f"fetch 0x{a:08x}: {n}" for n in notes]
        if self.timing is not None:
            _, _, notes = self.timing.take()
            runs += self.timing.taken_runs
            self._fetch_notes += [f"fetch 0x{a:08x}: {n}" for n in notes]
        info = t24 = None
        cti, compact = "", False
        if word is not None:
            try:
                dec = m.isa.decode(word)
            except ISADecodeError:
                text = f".word 0x{word:08x}"
            else:
                text = text_at(word, a, m.isa)
                info = analyze(dec)
                t24 = timing24(dec, info)
                cti = info.control
                compact = info.compact
        wait = sum(n for _, n in runs)
        s = Slot24(seq, a, word, dec, text, info, t24, t, t + 2 + wait, cti=cti,
                   compact=compact, fetch_exc=exc, fetch_error=err,
                   fetch_cache=m.fetch_accesses, fetch_tlb=m.fetch_tlb, fetch_runs=runs,
                   imiss=m.fetch_missed and word is not None)
        self._link(s)
        return s

    def _link(self, s: Slot24) -> None:
        last = self.last_fetched
        if last is not None:
            last.next_seq = s.seq
        if last is not None and last.cti and not last.compact:
            mark_slot(last, s)
        self.last_fetched = s
        self.slots.append(s)

    def _fetch16(self, a: int, t: int) -> Slot24:
        """One MIPS16e instruction (2 or 4 bytes) of the bundle; its
        recode cycle is set by :meth:`_recode`."""
        m = self.m
        p = fetch_compressed(self, m, a)
        runs: list = []
        if self.utlb is not None:
            wait, notes = self.utlb.take()
            if wait:
                runs.append(("utlb", wait))
            self._fetch_notes += [f"fetch 0x{a:08x}: {n}" for n in notes]
        if self.timing is not None:
            _, _, notes = self.timing.take()
            runs += self.timing.taken_runs
            self._fetch_notes += [f"fetch 0x{a:08x}: {n}" for n in notes]
        info = p.info
        s = Slot24(p.seq, a, p.word, p.dec, p.text, info,
                   timing24(p.dec, info) if p.dec is not None else None, t, t + 4,
                   cti=info.control if info else "", compact=info.compact if info else False,
                   fetch_exc=p.fetch_exc, fetch_error=p.fetch_error, fetch_cache=p.fetch_cache,
                   fetch_tlb=p.fetch_tlb, fetch_runs=runs, imiss=p.imiss, isa=1, size=p.size,
                   insn=p.insn, decode_error=p.decode_error)
        self._link(s)
        return s

    # -- readiness ---------------------------------------------------------------------

    def _unit_wait(self, s: Slot24, t: int) -> tuple[str, str] | None:
        """A unit ``s`` needs is still busy at cycle ``t``: the multiply /
        divide unit (Tables 2.2-2.4), the FPU (Table 3.33 repeat rates),
        the load / store unit (a DTLB miss, a full load queue), a TLB
        write (Table 2.6), or an older load still to deliver a register
        that ``s`` writes (Table 2.5: destination GPR dependency)."""
        if s.dec is None or s.t24 is None:
            return None
        if self.macro_until > t:
            return ("macro", "SAVE / RESTORE moves one word per cycle")
        name, writes = s.dec.name, s.t24.writes
        if name == "EHB" and any(t < h.ex + 3 for h in self.hazards):
            return ("ehb", "EHB waits until the CP0 writes before it are done")
        if name in _MUL_GPR:
            n, window = MUL_RATE
            if sum(1 for c in self.mul_starts if c > t - window) >= n:
                return ("mdu", f"{s.text}: at most {n} MULs every {window} cycles")
        if writes & {"hi", "lo"} or name in _DIV_GPR:
            busy = max([self.div_free if name in _DIV_HILO | _DIV_GPR else 0]
                       + [self.scoreboard[r][0] - 1 for r in ("hi", "lo")
                          if r in writes and r in self.scoreboard])
            if busy > t:
                return ("div" if self.div_free > t else "mdu",
                        f"{s.text}: the multiply / divide unit is busy")
        if s.t24.kind == "fpu" and name not in _FP_STORES and self.fpu_free > t:
            return ("fpu", f"{s.text}: the FPU is busy (repeat rate)")
        lsu = _lsu(s.dec)
        if lsu and self.lsu_free > t:
            return ("lsu", f"{s.text}: the load / store unit waits for the TLB")
        if (lsu or s.dec.entry["family"] == "cop0") and self.tlb_until > t:
            return ("tlb", f"{s.text}: waits for the TLB write / read (Table 2.6)")
        if lsu and s.info is not None and s.info.ready == "MEM" and name not in _FP_STORES:
            self.ldq = [c for c in self.ldq if c > t]
            if len(self.ldq) >= LDQ_ENTRIES:
                return ("ldq", f"{s.text}: the load data queue is full")
        for r in writes:
            entry = self.scoreboard.get(r)
            if entry is not None and entry[1].t24.kind == "load" and entry[0] > t + 3:
                return ("waw", f"{s.text}: {_rname(r)} is still to come from "
                                f"{entry[1].text}")
        return None

    # -- CP0 hazards (Table 2.7) ---------------------------------------------------------

    def _cp0_state(self) -> tuple[dict, tuple | None]:
        entries = getattr(self.m.mmu, "entries", None)
        return ({k: v["value"] for k, v in self.m.cp0.items()},
                tuple(entries) if isinstance(entries, list) else None)

    def _log_writes(self, s: Slot24, t: int, before) -> None:
        """Record what a CP0-writing instruction changed: the younger
        instructions within the spacing see the old values."""
        regs, entries = before
        now_regs, now_entries = self._cp0_state()
        src = ((s.word >> 11) & 31, s.word & 7) if s.dec.name == "MTC0" else s.dec.name
        for key, old in regs.items():
            new = now_regs[key]
            if new != old:
                self.hazards.append(Hazard(key, old, new, t, src, s.text))
        if entries is not None and any(a is not b for a, b in zip(entries, now_entries)):
            self.hazards.append(Hazard("tlb", entries, now_entries, t, s.dec.name, s.text))

    def _see(self, t: int, consumer: str, ev: list[str], s: Slot24) -> list:
        """Put in the values an instruction executing at cycle ``t`` still
        sees (the old ones of the writes within the spacing); returns what
        :meth:`_unsee` restores."""
        self.hazards = [h for h in self.hazards if t <= h.ex + 5]
        by_key: dict = {}
        for h in self.hazards:
            by_key.setdefault(h.key, []).append(h)
        swaps = []
        for key, hs in by_key.items():
            seen = hs[0].old
            for h in reversed(hs):
                if t >= h.ex + hazard_spacing(h, key, consumer) + 1:
                    seen = h.new
                    break
            current = self._get(key)
            if not _same(seen, current):
                self._put(key, seen)
                swaps.append((key, seen, current))
                h = hs[-1]
                if consumer != "int" and _reads(s.dec, s.word, key, consumer):
                    ev.append(f"CP0 hazard: {s.text} still sees the old "
                              f"{_key_name(self.m, key)} ({h.text} executed "
                              f"{t - h.ex} cycle{'s' if t - h.ex != 1 else ''} before)")
        return swaps

    def _unsee(self, swaps: list, written: frozenset = frozenset()) -> None:
        """Back to the current values — except what the instruction wrote
        itself (computed from the old values it saw: that stands)."""
        tlb = False
        for key, seen, current in swaps:
            tlb = tlb or key == "tlb"
            if key not in written and _same(self._get(key), seen):
                self._put(key, current)
            elif key == C.SRSCTL and key not in written:
                # an ERET switched sets from the PSS it saw: its CSS stands,
                # the software fields written meanwhile too
                self._put(key, (current & ~0xF) | (self._get(key) & 0xF))
        if tlb:
            self.m.mmu._changed()                    # no micro-TLB copy of an old entry

    def _get(self, key):
        if key == "tlb":
            return tuple(self.m.mmu.entries)
        return self.m.cp0[key]["value"]

    def _put(self, key, value) -> None:
        if key == "tlb":
            self.m.mmu.entries[:] = list(value)
            self.m.mmu._changed()
        else:
            self.m.cp0[key]["value"] = value

    def _waits_for(self, s: Slot24, regs, t: int) -> tuple[Slot24, Any] | None:
        """The producer an operand of ``s`` still waits for at cycle ``t``:
        an older instruction that has not executed yet (in AG), or the
        scoreboard of results still on their way (a MUL's result comes
        from the MDU after the MUL itself has left the pipeline)."""
        if not regs:
            return None
        name = s.dec.name if s.dec is not None else ""
        ag = self.stage["AG"]
        for r in regs:
            if ag is not None and ag is not s and ag.seq < s.seq and ag.t24 is not None \
                    and r in ag.t24.writes:
                return ag, r
            entry = self.scoreboard.get(r)
            if entry is None:
                continue
            ready, p = entry
            pname = p.dec.name
            if is_accumulator(r) and (s.dec.entry.get("ase") in _DSP_ASE
                                         or p.dec.entry.get("ase") in _DSP_ASE):
                ready += _dsp_pair(pname, name)  # the DSP Module: MD00622 Table 2.5
            elif r == "dspctl" and name == "RDDSP" and (pname in DSP.MACS
                                                        or pname in DSP.GPR_MULTIPLIES):
                ready += 4                       # ouflag from the MDU's A stage (2.4.2)
            elif pname in ("MTHI", "MTLO") and name in _MADD:
                ready += 1                       # Table 2.2: MTHI / MTLO -> MADD 1
            elif pname.startswith("C.") and name.startswith(("MOVF.", "MOVT.")):
                ready -= 1                       # Table 3.33: C.cond -> MOVF.fmt 1
            if ready > t:
                return p, r
        return None

    # -- one clock cycle ------------------------------------------------------------------

    def cycle(self) -> CycleTrace:
        m, st = self.m, self.stage
        self.cycles += 1
        t = self.cycles
        trace = CycleTrace(t, {})
        ev = trace.events
        stall: tuple[str, str] | None = None
        perf = m.perf.armed

        # WB: the instruction completes
        wb = st["WB"]
        if m.pdt.active or wb is not None and wb.pdt is not None:
            m.pdt.clock(wb.pdt if wb is not None else None,
                        asleep=m.waiting)        # MIPS Trace: what completes now
        if wb is not None:
            self.retired += 1
            if m.ejtag.sample_due:               # PC sampling: the graduating one
                m.ejtag.sample(wb.pc)
            if wb.perf is not None:
                m.perf.retire(*wb.perf)
        st["WB"] = None

        # MS: an access still waiting for memory holds MS and everything behind
        ms = st["MS"]
        held = ms is not None and ms.mem_wait > 0
        if held:
            ms.mem_wait -= 1
            why = _pop_run(ms.mem_runs)
            stall = ("memory", f"{ms.text} waits in MS ({ms.mem_why}; "
                               f"{ms.mem_wait + 1} more cycle{'s' if ms.mem_wait else ''})")
            if perf:
                m.perf.add("alu_stall")
                if why in _MEM_EVENTS:
                    m.perf.add(_MEM_EVENTS[why])
            st["WB"], st["ER"] = st["ER"], None
        else:
            st["WB"], st["ER"], st["MS"], st["EX"] = st["ER"], st["MS"], st["EX"], None

        # ER: a pending exception is taken now (the younger ones never ran)
        er = st["ER"]
        if er is not None and er.exc is not None:
            m.wake()
            enter(m, er)
            m.take_exception(er.exc, er.pc, er.word, er.dec, er.in_slot)
            name = C.EXC_NAMES.get(er.exc.code, str(er.exc.code))
            ev.append(f"exception {name} at 0x{er.pc:08x} ({er.text}) in ER: kill the "
                      f"younger instructions, fetch the handler at 0x{m.pc:08x}")
            er.timeline[t] = "ER"
            st["ER"] = None
            self._flush_after(er.seq - 1)
            self.last_fetched = None
            self._redirect(m.pc | m.isa_mode if m.compressed is not None else m.pc, t + 1)

        # AG -> EX: operands, interlocks, a pending exception older than it
        entered = None
        x, y, z = st["EX"], st["MS"], st["ER"]
        frozen = (x is not None and x.exc is not None or y is not None and y.exc is not None
                  or z is not None and z.exc is not None)
        ag = st["AG"]
        if ag is not None and not held and st["EX"] is None and not frozen:
            wait = self._waits_for(ag, ag.t24.at_ex if ag.t24 else (), t)
            busy = self._unit_wait(ag, t) if wait is None else None
            if wait is not None:
                p, r = wait
                kind = _kind(p)
                if kind == "load" and p.ex_cycle is not None and \
                        p.ready_at.get(r, 0) > p.ex_cycle + 2:
                    kind = "miss-wait"           # its data is late: a D-cache miss
                stall = (kind, f"{ag.text} waits in AG for {_rname(r)} from {p.text}")
            elif busy is not None:
                stall = busy
            elif m.waiting and m.interrupt_request() is None:
                stall = ("wait", "WAIT: sleeping until an interrupt")
            else:
                st["EX"], st["AG"] = ag, None
                entered = ag
                self._execute(ag, t, trace, ev)      # EX: in program order
                if self.watch_hit is not None:       # the debugger: it stays in AG
                    st["EX"], st["AG"], entered = None, ag, None
                    stall = ("watch", f"{ag.text} stopped by a debugger watchpoint")

        # RF -> AG: the address base / index
        rf = st["RF"]
        if rf is not None and st["AG"] is None and not held:
            wait = self._waits_for(rf, rf.t24.at_ag if rf.t24 else (), t)
            if wait is None:
                st["AG"], st["RF"] = rf, None
            elif stall is None:
                p, r = wait
                kind = "alu-agen" if _kind(p) == "alu" else _kind(p)
                stall = (kind, f"{rf.text} waits in RF for the address base "
                               f"{_rname(r)} from {p.text}")
        # IT / IS -> RF
        if st["RF"] is None and not held and self.queue and self.queue[0].avail <= t:
            st["RF"] = self.queue.pop(0)

        # IF: the next bundle
        fetched = self._fetch(t)

        # the views of this cycle
        for name in BACK:
            s = st[name]
            if s is not None:
                s.timeline[t] = name
        first: dict[str, Slot24] = {}            # the oldest in IS / IR / IK / IT
        for s in self.queue:
            if s.isa:                                # MIPS16e: IR / IK (2.1.3, 2.1.4)
                where = ("IF" if t == s.fetched else "IS" if t < s.recode else
                         "IR" if t == s.recode else "IK" if t == s.recode + 1 else "IT")
            elif t < s.avail - 1:
                where = "IF"
            elif t == s.avail - 1:
                where = "IS"
            else:
                where = "IT"
            s.timeline[t] = where
            if where not in first:
                first[where] = s
        stages = {n: _sv(st[n]) for n in BACK}
        stages["IF"] = _sv(fetched[0]) if fetched else None
        for n in ("IS", "IR", "IK", "IT"):
            stages[n] = _sv(first.get(n))
        trace.stages = {n: stages.get(n) for n in STAGES16}
        if stall is None and st["RF"] is None and self.fetching:
            stall = ("fetch", "the fetch unit has no instruction for RF")
        if stall is not None:
            kind, why = stall
            trace.stall = f"{kind}: {why}"
            self.stalls[kind] = self.stalls.get(kind, 0) + 1
        if perf:
            self._count_stalls(entered, stall)
        ev.extend(self._fetch_notes)
        self._fetch_notes = []
        m.advance(1)
        self.history.append(trace)
        return trace

    def _count_levels(self, t: int) -> None:
        """Per-cycle occupancy events (Table 6.47): outstanding D-cache /
        L2 misses, the FSB and LDQ fill levels (0-1 / 2-4 / 5+ of 8 or 9),
        the IFU fill buffer."""
        p, tm = self.m.perf, self.timing
        if tm is not None:
            out = [x for x in tm.outstanding if x[0] > t]
            if out:
                p.add("dcache_miss_cycles")
                caches = self.m.caches
                if caches is not None and caches.l2 is not None \
                        and any(level not in ("L2", "C2C") for _, level in out):
                    p.add("l2_miss_cycles")
            fsb = sum(1 for c in tm.pending.values() if c > t)
            p.add("fsb_low" if fsb <= 1 else "fsb_mid" if fsb <= 4 else "fsb_high")
        ldq = sum(1 for c in self.ldq if c > t)
        p.add("ldq_low" if ldq <= 1 else "ldq_mid" if ldq <= 4 else "ldq_high")
        if self.ifu_miss_until > t:
            p.add("ifu_fb")

    def _count_stalls(self, entered, stall) -> None:
        p = self.m.perf
        self._count_levels(self.cycles)
        if self.recovering is not None:
            if entered is not None and entered.seq > self.recovering:
                self.recovering = None
            else:
                p.add("mispredict_stall")
        if entered is None:
            p.add("stall")
        if stall is None:
            return
        kind = stall[0]
        if kind == "fetch":
            p.add("ifu_stall")
            head = self.queue[0] if self.queue else None
            if head is not None and head.fetch_runs and head.fetch_runs[-1][0] == "I-miss":
                p.add("icache_stall")
            return
        if kind == "memory":
            return                               # counted where it is held
        p.add("alu_stall")
        if kind == "miss-wait":                  # waits on a missing load's data
            p.add("load_use_stall")
        event = {"load": "load_use_stall", "alu-agen": "alu_agen_stall", "waw": "dcache_stall",
                 "mfc0": "other_stall", "mdu": "mdu_stall", "div": "mdu_stall",
                 "fpu": "fpu_stall", "lsu": "cache_fixup",
                 "ldq": "ldq_full_stall"}.get(kind) or _HAZARD_EVENTS.get(kind)
        if event:
            p.add(event)

    # -- execution ------------------------------------------------------------------------

    def _execute(self, s: Slot24, t: int, trace: CycleTrace, ev: list[str]) -> None:
        m = self.m
        late = m.pending_cache_error()
        if late is not None:
            m.wake()
            s.exc = late
            return
        swaps = self._see(t, "int", ev, s) if self.hazards else []
        pending = m.interrupt_request()
        self._unsee(swaps)
        if pending is not None:
            m.wake()
            s.exc = pending
            what = "interrupt" if pending.code == C.INT else C.EXC_NAMES[pending.code]
            ev.append(f"{what}: taken on {s.text} when it reaches ER")
            return
        if m._ihw or self.hazards:              # instruction breakpoints, fetch watch
            swaps = self._see(t, "fetch", ev, s) if self.hazards else []
            try:
                m.instruction_checks(s.pc)
            except _Arch as a:
                s.exc = a
                return
            finally:
                self._unsee(swaps)
        if s.fetch_error is not None:
            raise s.fetch_error
        if s.fetch_exc is not None:
            s.exc = s.fetch_exc
            return
        if s.decode_error is not None:
            raise s.decode_error
        if s.dec is None:
            if m.exceptions == "raise":
                m.isa.decode(s.word)                     # raises ISADecodeError
            s.exc = _Arch(C.RI)
            return
        name = s.dec.name
        rs, rt = m.read_reg((s.word >> 21) & 31), m.read_reg((s.word >> 16) & 31)
        cfg7 = m.cp0_read(*CONFIG7)
        if self.timing is not None:
            self.timing.nonblocking = not cfg7 & CFG7_SL
            self.timing.block_uncached = bool(cfg7 & CFG7_ULB)
        m.last_view = None
        m._forbidden = False
        res = None
        writer = name in _CP0_WRITERS
        before = self._cp0_state() if writer else None
        swaps = self._see(t, consumer_class(s.dec), ev, s) if self.hazards else []
        try:
            m.r6_slot_check(s.dec, s.in_slot, s.forbidden)
            m.fetch_missed = s.imiss
            enter(m, s)
            res = m.execute(s.pc, s.word, s.dec, in_slot=s.in_slot)
            s.pdt, m.pdt_packet = m.pdt_packet, None
        except WatchpointHit as w:              # before the access: nothing happened
            self.watch_hit = w
            m._redirect = None
            return
        except _Arch as a:
            if self.timing is not None:
                self.timing.take()                       # the access did not happen
            if self.utlb is not None:
                self.utlb.take()
            m._redirect = None
            s.exc = a
        finally:
            self._unsee(swaps, _written(s.dec, s.word) if s.exc is None else frozenset())
        if s.exc is None:
            if writer:
                self._log_writes(s, t, before)
            if name in _BARRIERS:
                self.hazards = []
        v = m.last_view
        if v is not None:
            trace.ex_view = v
            v.fetch, v.fetch_tlb = s.fetch_cache, s.fetch_tlb
        if s.exc is not None:
            return
        s.perf, m.last_events = m.last_events, None
        s.executed, s.ex_cycle = True, t
        ready = s.t24.ready
        if name in _DIV_HILO or name in _DIV_GPR:
            delay = divide_delay(name, rs, rt)
            ready = 1 + delay + (4 if name in _DIV_GPR else 0)
            self.div_free = t + 1 + delay
        elif name in _MUL_GPR:
            self.mul_starts = [c for c in self.mul_starts if c > t - MUL_RATE[1]] + [t]
        if s.t24.kind == "fpu" and name not in _FP_STORES:
            self.fpu_free = t + fp_repeat(name)
        if s.info.ex_cycles > 1 and s.insn is not None:   # SAVE / RESTORE: a word a cycle
            self.macro_until = t + s.info.ex_cycles
        if name in _TLB_WRITES:
            self.tlb_until = t + 3
        elif name == "TLBR":
            self.tlb_until = t + 2
        for r in s.t24.writes:
            s.ready_at[r] = t + s.t24.ready_by.get(r, ready)
            self.scoreboard[r] = (s.ready_at[r], s)
        self._memory_time(s, ev)
        if m.halted:                                     # exit service
            self.fetching = False
            self._flush_after(s.seq)
            ev.append(f"{s.text}: program exit, the pipeline drains")
            return
        if m._redirect is not None and not s.compact:    # ERET
            target = m._redirect & M32
            m._redirect = None
            ev.append(f"{s.text}: return to 0x{target:08x}, kill the younger "
                      "instructions")
            self._flush_after(s.seq)
            self._redirect(target, t + REDIRECT_DELAY)
            return
        if s.cti:
            self._resolve(s, res, t, ev)

    def _resolve(self, s: Slot24, res, t: int, ev: list[str]) -> None:
        """A branch / jump executed in EX: was the fetch right?"""
        m = self.m
        nxt = s.next_seq
        # the delay slot not fetched yet: it comes before the new target
        slot_missing = not s.compact and (nxt is None or self._find(nxt) is None)
        if s.compact:
            if m._redirect is not None:
                taken, actual = True, m._redirect & M32
            else:
                taken, actual = False, s.fall
                if m._forbidden:
                    self._mark(nxt, "forbidden")
            m._redirect = None
            m._forbidden = False
        elif res == NULLIFY:
            taken, actual = False, (s.pc + 8) & M32
            self._mark(nxt, "kill")
            ev.append(f"{s.text}: not taken, the delay slot is killed")
        else:
            taken = res is not None
            actual = res & M32 if taken else s.fall
        if s.pred.startswith("BHT") or s.pred.endswith("(Config7.BHT)"):
            self._bht_update(s.pc, taken)
        word = "taken" if taken else "not taken"
        keep = s.seq if s.compact or nxt is None else nxt
        if s.pred_next is None:                          # the fetch was waiting
            ev.append(f"{s.text}: {word} -> fetch 0x{actual:08x} (it was waiting: "
                      f"{s.pred})")
            if s.perf is not None and s.dec.name in ("JR", "JR.HB", "JALR") \
                    and (s.word >> 21) & 31 == 31 and not s.pred.startswith("fetch waits (C"):
                s.perf = (s.perf[0] + ("jr31_unpredicted",), s.perf[1])
            self._flush_after(keep)
            self._redirect(actual, t + REDIRECT_DELAY, after_slot=slot_missing)
            return
        self.predicted += 1
        self.last_resolution = (s.pc, s.text, taken, actual == s.pred_next)
        if actual == s.pred_next:
            ev.append(f"{s.text}: {word}, predicted right ({s.pred})")
            return
        s.mispredicted = True
        self.mispredicts += 1
        ev.append(f"{s.text}: {word} but predicted {s.pred}: kill the younger "
                  f"instructions, fetch 0x{actual:08x}")
        if s.perf is not None:
            extra = ("jr31_mispredict",) if s.rps else ("mispredict",)
            s.perf = (s.perf[0] + extra, s.perf[1])
        self._flush_after(keep)
        self._redirect(actual, t + REDIRECT_DELAY, after_slot=slot_missing)
        self.recovering = keep

    def _find(self, seq: int | None) -> Slot24 | None:
        if seq is None:
            return None
        for s in list(self.stage.values()) + self.queue:
            if s is not None and s.seq == seq:
                return s
        return None

    def _mark(self, seq: int | None, what: str) -> None:
        """Mark the next instruction (a delay / forbidden slot), fetched or not."""
        s = self._find(seq)
        if s is None:
            self.next_marks.add(what)
        elif what == "kill":
            for name in BACK:
                if self.stage[name] is s:
                    self.stage[name] = None
            if s in self.queue:
                self.queue.remove(s)
        else:
            s.forbidden = True

    def _flush_after(self, seq: int) -> None:
        """Kill every instruction younger than ``seq`` (the chart shows
        ``xx`` in the cycle it is killed)."""
        for name in BACK:
            s = self.stage[name]
            if s is not None and s.seq > seq:
                self.stage[name] = None
                s.timeline[self.cycles] = "xx"
        for s in self.queue:
            if s.seq > seq:
                s.timeline[self.cycles] = "xx"
        self.queue = [s for s in self.queue if s.seq <= seq]
        alive = [*self.queue, *(s for s in self.stage.values() if s is not None)]
        self.last_fetched = max(alive, key=lambda x: x.seq) if alive else None
        self.after_slot = None
        self.next_marks = set()

    def _redirect(self, target: int, at: int, *, after_slot: bool = False) -> None:
        """The fetch restarts at ``target`` from cycle ``at`` — after the
        delay slot when that is not fetched yet (it is next)."""
        self.blocked = False
        if after_slot:
            self.after_slot = ("target", target & M32, at)
            self.fetch_at = self.cycles + 1
        else:
            self._go(target)
            self.fetch_at = at

    def _memory_time(self, s: Slot24, ev: list[str]) -> None:
        """What the access costs: a wait in MS that holds the pipeline
        (a blocking miss, a full write buffer or fill-store buffer, SYNC,
        CACHE), a load's data coming later (a non-blocking miss: only its
        users wait), a DTLB miss (2.3.1: only the LSU waits — unless
        Config7.NBLSU locks the pipes together)."""
        if self.timing is None and self.utlb is None:
            return
        wait, extra, notes = self.timing.take() if self.timing is not None else (0, None, [])
        runs = list(self.timing.taken_runs) if self.timing is not None else []
        if extra is not None and extra > 0 and s.info is not None and s.info.ready == "MEM":
            self.ldq.append(s.ex_cycle + 1 + extra)
        if self.utlb is not None:
            more, tlb_notes = self.utlb.take()
            notes = tlb_notes + notes
            if more and self.m.cp0_read(*CONFIG7) & CFG7_NBLSU:
                wait += more                     # the pipes are locked together
                runs.insert(0, ("utlb", more))
            elif more:
                self.lsu_free = s.ex_cycle + 1 + more
                extra = (extra if extra else 1) + more
                notes.append(f"the LSU waits {more} cycles for the DTLB; younger "
                             "ALU instructions go on")
        s.mem_wait, s.mem_runs = wait, runs
        s.mem_why = notes[0] if notes else ""
        ev.extend(f"{s.text}: {n}" for n in notes)
        if extra is not None and extra > 0:
            for r in s.t24.writes:
                s.ready_at[r] = max(s.ready_at.get(r, 0), s.ex_cycle + 1 + extra)
                self.scoreboard[r] = (s.ready_at[r], s)

    # -- running ----------------------------------------------------------------------------

    def empty(self) -> bool:
        return not self.queue and all(v is None for v in self.stage.values())

    def next_slot(self) -> Slot24 | None:
        """The instruction that executes next (about to enter EX)."""
        for name in ("AG", "RF"):
            if self.stage[name] is not None:
                return self.stage[name]
        return self.queue[0] if self.queue else None

    def current_slot(self) -> Slot24 | None:
        return self.stage["EX"]

    def entering_ex(self) -> Slot24 | None:
        """The instruction in AG when it is the next to execute: no older
        one has an exception still to take (that would kill it)."""
        if any(x is not None and x.exc is not None
               for x in (self.stage["EX"], self.stage["MS"], self.stage["ER"])):
            return None
        return self.stage["AG"]

    def run(self, *, max_cycles: int = 100_000, until_pc: int | None = None) -> int:
        """Cycle until the instruction at ``until_pc`` is the next to
        execute (every older one has executed), the program exits and the
        pipeline is empty, or ``max_cycles``; returns cycles run."""
        start = self.cycles
        while self.cycles - start < max_cycles:
            if self.m.halted and self.empty():
                break
            nxt = self.entering_ex()
            if until_pc is not None and nxt is not None and nxt.pc == until_pc:
                break
            self.cycle()
        return self.cycles - start

    def drain(self, keep: set[int] | None = None, *, max_cycles: int = 1000) -> None:
        """Stop fetching, drop the instructions that have not executed and
        are not at an address in ``keep``, and cycle until empty."""
        self.fetching = False
        self.queue = [s for s in self.queue if keep is not None and s.pc in keep]
        for name in ("RF", "AG"):
            s = self.stage[name]
            if s is not None and (keep is None or s.pc not in keep):
                self.stage[name] = None
        for _ in range(max_cycles):
            if self.empty():
                return
            self.cycle()

    def chart(self, *, width: int = 60, last: bool = False) -> str:
        """Pipeline diagram: one row per instruction (also the killed
        ones), one column per cycle: IF IS IT RF AG EX MS ER WB."""
        cycles = [t.cycle for t in self.history]
        cycles = cycles[-width:] if last else cycles[:width]
        head = f"{'':10} {'instruction':28} " + " ".join(f"{c:>2}"[-2:] for c in cycles)
        lines = [head]
        for s in sorted(self.slots, key=lambda x: x.seq):
            if not any(c in s.timeline for c in cycles):
                continue
            lines.append(f"{s.pc:08x}   {s.text[:28]:28} "
                         + " ".join(s.timeline.get(c, "  ") for c in cycles))
        return "\n".join(lines)


def _reads(dec: Decode, word: int, key, consumer: str) -> bool:
    """Does the instruction use this CP0 state (worth a word in the trace)?"""
    name = dec.name
    if name in ("MFC0", "MTC0"):
        return key == ((word >> 11) & 31, word & 7)
    if name in ("EI", "DI"):
        return key == C.STATUS
    if name in ("ERET", "DERET"):
        return key in (C.EPC, C.STATUS, (30, 0))
    if name in _TLB_OPS:
        return key in ("tlb", (10, 0), (2, 0), (3, 0), (0, 0), (5, 0))
    if name == "RDHWR":
        return key == (9, 0)
    if consumer == "srs":
        return key == C.SRSCTL
    if consumer == "lsu":
        return key in ("tlb", _ENTRYHI, C.STATUS)
    return consumer == "cop" and key == C.STATUS


def _written(dec: Decode, word: int) -> frozenset:
    """The CP0 state an instruction writes (its result stands even when
    equal to what it saw)."""
    name = dec.name
    if name == "MTC0":
        return frozenset({((word >> 11) & 31, word & 7)})
    if name in ("EI", "DI"):
        return frozenset({C.STATUS})
    if name == "TLBR":
        return frozenset({(10, 0), (2, 0), (3, 0), (5, 0)})
    if name == "TLBP":
        return frozenset({(0, 0)})
    if name in ("TLBWI", "TLBWR", "TLBINV", "TLBINVF"):
        return frozenset({"tlb"})
    return frozenset()


def _same(a, b) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(x is y for x, y in zip(a, b))
    return a == b


def _key_name(m, key) -> str:
    if key == "tlb":
        return "TLB entries"
    return m.cp0[key].get("name") or f"CP0 {key[0]}.{key[1]}"


def _kind(p: Slot24) -> str:
    return p.t24.kind if p.t24 is not None else "alu"


def _branch_target(s: Slot24) -> int:
    """The PC-relative target of a branch (offset in bits 15:0; R6 BC /
    BALC 25:0, BEQZC / BNEZC 20:0)."""
    w, name = s.word, s.dec.name
    if name in ("BC", "BALC"):
        off, bits, base = w & 0x3FFFFFF, 26, s.pc + 4
    elif name in ("BEQZC", "BNEZC"):
        off, bits, base = w & 0x1FFFFF, 21, s.pc + 4
    else:
        off, bits, base = w & 0xFFFF, 16, s.pc + 4
    if off >> (bits - 1):
        off -= 1 << bits
    return (base + (off << 2)) & M32
