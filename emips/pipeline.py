"""Cycle-level 5-stage MIPS pipeline (PLAN M5).

The classic in-order pipeline — IF, ID, EX, MEM, WB — of the textbook MIPS
(Patterson & Hennessy) with the real MIPS32 choices:

* **branches and jumps resolve in ID**, with one **delay slot** (the
  instruction already in IF always executes; a not-taken branch-likely
  squashes it);
* **forwarding** EX/MEM → EX, MEM/WB → EX and EX/MEM → ID (branch
  comparator), switchable off for teaching (then operands wait for WB;
  the register file writes in the first half of a cycle and reads in the
  second);
* **hazard detection**: load-use (1 bubble), a branch that needs the
  result of the instruction in EX (1 bubble) or of a load (2 bubbles),
  HI/LO interlock against the multiply/divide unit;
* a **multiply/divide unit** working in parallel with the pipeline
  (``MULT/DIV/MADD/MSUB`` occupy it for their latency; ``MFHI/MFLO`` and
  the next MDU instruction wait), while ``MUL`` and FPU arithmetic hold
  EX for their latency;
* **precise exceptions and interrupts**: taken when the faulting /
  interrupted instruction is in EX; older instructions complete, younger
  ones are flushed; ``ERET`` flushes the instructions fetched after it;
* the clock: ``Count`` advances with the real cycles
  (:meth:`emips.sim.Machine.advance`).

Results are **identical** to :class:`emips.sim.Machine` (the architectural
reference): every instruction executes once, in program order, through the
same executors — when it enters EX, or in ID for branches and jumps.  The
pipeline adds the timing; ``tests/test_pipeline.py`` checks both.

* **memory timing** (with caches, :mod:`emips.timing`): an instruction
  miss holds IF until the line arrives; a data access that must wait (a
  miss, a full write buffer, ``SYNC``, a ``CACHE`` operation) holds MEM
  and freezes the stages behind it — the blocking cache of the textbook.
  ``nonblocking=True`` is the 24K's hit-under-miss: a missing load goes
  on and only an instruction that needs its register waits in ID.
  Without caches memory is ideal (one cycle).

Simplifications (documented model choices): no CP0 hazards (``EHB`` is
a no-op), store data is needed in EX (a load followed by a store of the
loaded value stalls one cycle), the FPU is a blocking EX unit, the
memory access is computed in EX and its waiting done in MEM.  An
imprecise L2 / L3 Cache Error recorded by an access is taken when the
next instruction reaches EX, like an interrupt.

A debugger stops at precise points (all older instructions executed, this
one not): :attr:`Pipeline.debug_stop` names the addresses it stops at, and
a branch there is not executed in ID (where the instruction before it
executes in the same cycle) but held to EX, then fetched again; a
debugger watchpoint holds its load / store in EX for a cycle
(:attr:`Pipeline.watch_hit`).  Without a debugger nothing changes.

Every :meth:`Pipeline.cycle` returns a :class:`CycleTrace` (stage contents,
control signals of the instruction in ID, stalls, forwarding paths,
branch decisions, flushes, exceptions) — what the CLI / GUI show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import cp0 as C
from .disasm import disassemble
from .exceptions import FetchError, WatchpointHit
from .isa import Decode, ISADecodeError
from .sim import NULLIFY, Machine, _Arch
from .syntax import kind_of, operand_spec
from .timing import MEMORY_TIMING

M32 = 0xFFFFFFFF
STAGES = ("IF", "ID", "EX", "MEM", "WB")

#: Default latencies in cycles (MIPS 24K-like; configurable per Pipeline).
LATENCY: dict[str, int] = {
    "MUL": 4,          # MUL rd: holds EX
    "MULT": 4,         # MULT/MULTU/MADD*/MSUB*: MDU busy
    "DIV": 35,         # DIV/DIVU: MDU busy
    "FP_ADD": 4, "FP_MUL": 4, "FP_MADD": 4, "FP_CVT": 4, "FP_CMP": 1,
    "FP_DIV_S": 17, "FP_DIV_D": 32, "FP_SQRT_S": 17, "FP_SQRT_D": 32,
}

_LOADS_TO_RT = {"LB", "LBU", "LH", "LHU", "LW", "LWL", "LWR", "LL", "LBE",
                "LBUE", "LHE", "LHUE", "LWE", "LWLE", "LWRE", "LLE"}
_LOADS_TO_RT |= {"LL@R6"}
_RT_DEST = (_LOADS_TO_RT | {"SC", "SCE", "SC@R6", "MFC0", "MFC1", "MFHC1",
                            "CFC1", "RDHWR", "RDHWR@R6", "DI", "EI", "EXT",
                            "INS", "LUI",
                            "AUI"})
_RT_DEST_AND_SRC = {"LWL", "LWR", "LWLE", "LWRE", "INS", "SC", "SCE", "SC@R6"}
_LINKS = {"JAL", "BAL", "BGEZAL", "BLTZAL", "BGEZALL", "BLTZALL", "NAL",
          # Release 6 compact (link = PC + 4)
          "BALC", "JIALC", "BEQZALC", "BNEZALC", "BLEZALC", "BGEZALC",
          "BGTZALC", "BLTZALC"}
_R6_MUL = {"MUL@R6", "MUH", "MULU", "MUHU"}
_R6_DIV = {"DIV@R6", "MOD", "DIVU@R6", "MODU"}
_MDU_OPS = {"MULT", "MULTU", "MADD", "MADDU", "MSUB", "MSUBU", "DIV", "DIVU"}
_FP_LOADS = {"LWC1", "LDC1", "LWXC1", "LDXC1", "LUXC1"}
_COND_MOVES = ("MOVZ.", "MOVN.", "MOVF.", "MOVT.")

Res = tuple[str, int] | str          # ("r", 5), ("f", 2), "hi", "lo", "fcc"


@dataclass(frozen=True)
class Info:
    """Static pipeline view of one instruction."""

    reads_id: frozenset          # needed in ID (branch / jump-register)
    reads_ex: frozenset          # needed in EX
    writes: frozenset
    ready: str                   # stage at whose end the result exists
    ex_cycles: int               # cycles spent in EX (>1: holds EX)
    mdu_cycles: int              # >0: occupies the multiply/divide unit
    control: str                 # resolved in ID: "branch" / "jump" / ""
    compact: bool = False        # R6 compact: no delay slot


def analyze(dec: Decode, latency: dict[str, int] = LATENCY) -> Info:
    """Registers read / written and timing class of a decoded row."""
    e = dec.entry
    name, fam = e["name"], e["family"]
    w = dec.word
    f5 = lambda lsb: (w >> lsb) & 31                     # noqa: E731
    reads: set = set()
    writes: set = set()
    for item in operand_spec(e):
        kind = kind_of(item)[0]
        if kind == "rs":
            reads.add(("r", f5(21)))
        elif kind == "rt":
            if fam == "imm" or name in _RT_DEST:
                writes.add(("r", f5(16)))
                if name in _RT_DEST_AND_SRC:
                    reads.add(("r", f5(16)))
            else:
                reads.add(("r", f5(16)))
        elif kind == "rd":
            writes.add(("r", f5(11)))
        elif kind == "rsd":                      # R6 PC-relative destination
            writes.add(("r", f5(21)))
        elif kind in ("mem", "mem9"):
            reads.add(("r", f5(21)))
        elif kind == "idx":
            reads |= {("r", f5(21)), ("r", f5(16))}
        elif kind in ("fs", "ft", "fr", "fd"):
            n = f5({"fs": 11, "ft": 16, "fr": 21, "fd": 6}[kind])
            dest = (kind == "fd" or (kind == "ft" and name in ("LWC1", "LDC1"))
                    or (kind == "fs" and name in ("MTC1", "MTHC1")))
            regs = {("f", n), ("f", n ^ 1)} if (".D" in name or name in (
                "LDC1", "SDC1", "LDXC1", "SDXC1", "LUXC1", "SUXC1")) else {("f", n)}
            (writes if dest else reads).update(regs)
            if dest and (name.startswith(_COND_MOVES + ("SEL.", "MADDF", "MSUBF"))
                         or name == "MTHC1"):
                reads.update(regs)
    if name in _LINKS:
        writes.add(("r", 31))
    if name in _MDU_OPS:
        writes |= {"hi", "lo"}
    if name in ("MADD", "MADDU", "MSUB", "MSUBU"):
        reads |= {"hi", "lo"}
    if name in ("MFHI",):
        reads.add("hi")
    if name in ("MFLO",):
        reads.add("lo")
    if name == "MTHI":
        writes.add("hi")
    if name == "MTLO":
        writes.add("lo")
    if name.startswith("C.") or name == "CTC1":
        writes.add("fcc")
    if name.startswith(("BC1", "MOVF", "MOVT")) or name == "CFC1":
        reads.add("fcc")
    writes.discard(("r", 0))
    reads.discard(("r", 0))

    control = ""
    if fam in ("branch", "branch_fpu", "branch_compact") or name.startswith("BC2"):
        control = "branch"
    elif fam in ("jump", "jump_compact"):
        control = "jump"
    reads_id = frozenset(reads) if control else frozenset()
    reads_ex = frozenset() if control else frozenset(reads)

    ready = "MEM" if (name in _LOADS_TO_RT or name in _FP_LOADS
                      or name in ("SC", "SCE", "SC@R6", "LWPC")) else "EX"
    ex_cycles, mdu = 1, 0
    if name == "MUL" or name in _R6_MUL:
        ex_cycles = latency["MUL"]
    elif name in _R6_DIV:
        ex_cycles = latency["DIV"]
    elif name in ("DIV", "DIVU"):
        mdu = latency["DIV"]
    elif name in _MDU_OPS:
        mdu = latency["MULT"]
    elif fam in ("fpu_arith", "fpu_cvt", "fpu_cmp"):
        op, fmt = name.split(".")[0], name.split(".")[-1]
        key = {"ADD": "FP_ADD", "SUB": "FP_ADD", "MUL": "FP_MUL",
               "MADD": "FP_MADD", "MSUB": "FP_MADD", "NMADD": "FP_MADD",
               "MADDF": "FP_MADD", "MSUBF": "FP_MADD", "CMP": "FP_CMP",
               "MIN": "FP_CMP", "MAX": "FP_CMP", "MINA": "FP_CMP",
               "MAXA": "FP_CMP", "CLASS": "FP_CMP",
               "NMSUB": "FP_MADD", "DIV": f"FP_DIV_{fmt}",
               "RECIP": f"FP_DIV_{fmt}", "SQRT": f"FP_SQRT_{fmt}",
               "RSQRT": f"FP_SQRT_{fmt}", "C": "FP_CMP"}.get(op, "FP_CVT")
        if op in ("ABS", "NEG"):
            key = "FP_CMP"
        ex_cycles = latency.get(key, 1)
    return Info(reads_id, reads_ex, frozenset(writes), ready,
                max(ex_cycles, 1), mdu, control,
                fam in ("branch_compact", "jump_compact"))


def control_signals(dec: Decode, info: Info) -> dict[str, Any]:
    """The textbook main-control signals for the instruction in ID."""
    e = dec.entry
    fam, name = e["family"], e["name"]
    spec = [kind_of(k)[0] for k in operand_spec(e)]
    gpr_write = any(r[0] == "r" for r in info.writes if isinstance(r, tuple))
    load = info.ready == "MEM"
    store = fam == "store" or name in ("SWC1", "SDC1", "SWXC1", "SDXC1", "SUXC1")
    return {
        "RegDst": "rd" if "rd" in spec else ("$ra" if name in _LINKS else "rt"),
        "ALUSrc": "imm" if any(k in spec for k in ("imm", "uimm", "mem", "mem9"))
                  else "reg",
        "MemtoReg": load,
        "RegWrite": gpr_write,
        "MemRead": load and name not in ("SC", "SCE"),
        "MemWrite": store,
        "Branch": info.control == "branch",
        "Jump": info.control == "jump",
        "ALUOp": name.lower(),
    }


# --------------------------------------------------------------- the pipe

@dataclass
class Slot:
    """An instruction in flight."""

    seq: int
    pc: int
    word: int | None
    dec: Decode | None
    text: str
    info: Info | None
    in_slot: bool = False             # this is a branch delay slot
    forbidden: bool = False           # R6 forbidden slot of a compact branch
    fetch_exc: _Arch | None = None    # AdEL / Cache Error on fetch, carried to EX
    fetch_error: FetchError | None = None
    pending: _Arch | None = None      # exception found in ID, taken in EX
    executed: bool = False
    busy: int = 0                     # extra EX cycles still to spend
    fetch_wait: int = 0               # cycles IF still waits for the instruction
    mem_wait: int = 0                 # cycles MEM still waits for the access
    mem_why: str = ""                 # why (the first timing note)
    #: the cache accesses of its fetch (emips.cache.CacheAccess; tracing)
    fetch_cache: list = field(default_factory=list)
    #: the translations of its fetch (emips.tlb.TLBAccess; tracing)
    fetch_tlb: list = field(default_factory=list)
    #: what its waits in IF / MEM are made of: [(stall kind, cycles)]
    fetch_runs: list = field(default_factory=list)
    mem_runs: list = field(default_factory=list)
    #: its performance counter events and Status (emips.perf), counted at WB
    perf: tuple | None = None
    #: MIPS Trace: what it completes with at WB; its fetch missed the I-cache
    pdt: Any = None
    imiss: bool = False
    #: a branch the debugger stops at: not executed in ID, so it reaches
    #: EX with everything older done (refetched when the run goes on)
    held: bool = False


@dataclass
class StageView:
    seq: int
    pc: int
    text: str
    control: dict[str, Any] | None = None
    #: the cache accesses that fetched it (tracing; see Slot.fetch_cache)
    fetch_cache: list = field(default_factory=list)
    #: cycles IF / MEM still wait for memory after this one
    fetch_wait: int = 0
    mem_wait: int = 0
    #: the translations of its fetch (tracing; see Slot.fetch_tlb)
    fetch_tlb: list = field(default_factory=list)


@dataclass
class CycleTrace:
    cycle: int
    stages: dict[str, StageView | None]
    events: list[str] = field(default_factory=list)
    stall: str | None = None
    #: what happened inside the instruction executed in EX / resolved in ID
    #: this cycle (emips.trace.InstrView; with trace=True)
    ex_view: Any = None
    id_view: Any = None


def _view(slot: Slot | None, control: bool = False) -> StageView | None:
    if slot is None:
        return None
    ctl = control_signals(slot.dec, slot.info) if control and slot.dec else None
    return StageView(slot.seq, slot.pc, slot.text, ctl, slot.fetch_cache,
                     slot.fetch_wait, slot.mem_wait, slot.fetch_tlb)


#: the performance counter event of a memory wait (timing / micro-TLB kind)
_MEM_EVENTS = {"D-miss": "dcache_stall", "coherence": "dcache_stall",
               "uncached": "uncached_stall",
               "write-buffer": "wbb_full_stall", "fsb": "fsb_full_stall",
               "sync": "sync_stall", "cacheop": "cacheop_stall", "utlb": "cache_fixup"}
#: ... of an interlock in ID
_HAZARD_EVENTS = {"load-use": "load_use_stall", "mdu": "mdu_stall",
                  "miss-wait": "dcache_stall"}


def _pop_run(runs: list) -> str | None:
    """Take one cycle off a list of ``(kind, cycles)`` waits; its kind."""
    if not runs:
        return None
    key, n = runs[0]
    if n <= 1:
        runs.pop(0)
    else:
        runs[0] = (key, n - 1)
    return key


def _rname(r: Res) -> str:
    if isinstance(r, tuple):
        from .syntax import gpr_name
        return gpr_name(r[1]) if r[0] == "r" else f"$f{r[1]}"
    return r.upper() if r != "fcc" else "FCC"


class Pipeline:
    """The 5-stage pipeline driving a :class:`~emips.sim.Machine`.

    ``forwarding=False`` disables all bypasses.  ``latency`` overrides
    entries of :data:`LATENCY`; ``memory`` entries of
    :data:`emips.timing.MEMORY_TIMING` (used when the machine has
    caches); ``nonblocking`` lets loads miss in the background.  The
    machine's PC is the first fetch address; afterwards the pipeline owns
    the program flow.
    """

    def __init__(self, machine: Machine, *, forwarding: bool = True,
                 latency: dict[str, int] | None = None, trace: bool = True,
                 memory: dict[str, int] | None = None, nonblocking: bool = False) -> None:
        self.m = machine
        machine.tracing = trace
        self.forwarding = forwarding
        self.latency = dict(LATENCY, **(latency or {}))
        #: the memory timing (None: ideal memory, no caches)
        self.timing = machine.caches.timing if machine.caches is not None else None
        if self.timing is not None:
            self.timing.configure(memory, nonblocking=nonblocking)
        #: the TLB's micro-TLB misses cost cycles (emips.tlb; with or
        #: without caches)
        self.utlb = machine.mmu if hasattr(machine.mmu, "take") else None
        if self.utlb is not None:
            self.utlb.timing = True
            self.utlb.utlb_cost = (memory or {}).get("UTLB", MEMORY_TIMING["UTLB"])
            self.utlb.take()
        self.nonblocking = nonblocking
        #: register -> cycle from which a missing load's value can be used
        self.reg_ready: dict = {}
        self._fetch_notes: list[str] = []
        self.stage: dict[str, Slot | None] = {s: None for s in STAGES}
        self.fetch_pc = machine.pc
        self.cycles = 0
        self.retired = 0
        self.mdu_busy = 0
        self.stalls: dict[str, int] = {}
        self._seq = 0
        self.history: list[CycleTrace] = []
        #: False: fetch nothing more (let the pipeline drain)
        self.fetching = True
        self.stage["IF"] = self._fetch()

    # -- fetch --------------------------------------------------------------

    def _fetch(self) -> Slot:
        slot = self._fetch_slot()
        notes: list[str] = []
        if self.utlb is not None:                # ITLB miss: the JTLB first
            slot.fetch_wait, notes = self.utlb.take()
            if slot.fetch_wait:
                slot.fetch_runs.append(("utlb", slot.fetch_wait))
        if self.timing is not None:
            wait, _, more = self.timing.take()
            slot.fetch_wait += wait
            slot.fetch_runs += self.timing.taken_runs
            notes = notes + more
        self._fetch_notes += [f"fetch 0x{slot.pc:08x}: {n}" for n in notes]
        return slot

    def _fetch_slot(self) -> Slot:
        m, pc = self.m, self.fetch_pc
        self.fetch_pc = (pc + 4) & M32
        self._seq += 1
        if pc & 3 or not m.address_ok(pc):
            return Slot(self._seq, pc, None, None, "(fetch: address error)",
                        None, fetch_exc=_Arch(C.ADEL, badvaddr=pc))
        try:
            word = m.fetch_word(pc)
        except FetchError as exc:
            return Slot(self._seq, pc, None, None, "(no code here)", None,
                        fetch_error=exc, fetch_cache=m.fetch_accesses,
                        fetch_tlb=m.fetch_tlb)
        except _Arch as exc:                    # TLB or Cache Error on the fetch
            what = ("TLB refill" if exc.info.get("refill") else
                    "cache error" if exc.code == C.CACHE_ERROR else
                    C.EXC_NAMES.get(exc.code, str(exc.code)))
            return Slot(self._seq, pc, None, None, f"(fetch: {what})", None,
                        fetch_exc=exc, fetch_cache=m.fetch_accesses,
                        fetch_tlb=m.fetch_tlb)
        try:
            dec = m.isa.decode(word)
        except ISADecodeError:
            return Slot(self._seq, pc, word, None, f".word 0x{word:08x}", None,
                        fetch_cache=m.fetch_accesses, fetch_tlb=m.fetch_tlb)
        return Slot(self._seq, pc, word, dec, disassemble(word, pc, isa=m.isa),
                    analyze(dec, self.latency), fetch_cache=m.fetch_accesses,
                    fetch_tlb=m.fetch_tlb, imiss=m.fetch_missed)

    # -- hazards ------------------------------------------------------------

    def _hazard(self, s: Slot) -> tuple[str, str] | None:
        """``(kind, explanation)`` when the instruction in ID cannot advance
        this cycle."""
        if s.info is None:
            return None
        ex, mem = self.stage["EX"], self.stage["MEM"]
        if s.info.mdu_cycles or s.info.reads_ex & {"hi", "lo"}:
            if self.mdu_busy:
                return ("mdu", f"multiply/divide unit busy "
                               f"({self.mdu_busy} more cycle(s))")
        needs = s.info.reads_ex | s.info.reads_id
        if self.reg_ready:
            for r in sorted(needs, key=str):
                ready = self.reg_ready.get(r, 0)
                if ready > self.cycles:
                    return ("miss-wait", f"{_rname(r)} from a load still missing "
                                         f"({ready - self.cycles} more cycle(s))")
        for prod, where in ((ex, "EX"), (mem, "MEM")):
            if prod is None or prod.info is None:
                continue
            common = needs & prod.info.writes - {"hi", "lo"}
            if not common:
                continue
            r = _rname(sorted(common, key=str)[0])
            if not self.forwarding:
                return ("no-forwarding", f"{r} not written back yet "
                                         f"({prod.text} in {where})")
            if s.info.reads_id & common:
                if where == "EX":
                    return ("branch", f"branch needs {r} from {prod.text} (in EX)")
                if prod.info.ready == "MEM":
                    return ("branch", f"branch needs {r} from load "
                                      f"{prod.text} (in MEM)")
            elif where == "EX" and prod.info.ready == "MEM":
                return ("load-use", f"{r} from {prod.text}")
        return None

    # -- one clock cycle ----------------------------------------------------

    def cycle(self) -> CycleTrace:
        m, st = self.m, self.stage
        self.cycles += 1
        trace = CycleTrace(self.cycles, {k: _view(v, k == "ID") for k, v in st.items()})
        ev = trace.events
        flush_to: int | None = None
        nullify = False
        branch_target: int | None = None

        # WB: retire (the performance counters count it); the multiply/divide
        # unit works on
        old_ex = st["EX"]
        if m.pdt.active or st["WB"] is not None and st["WB"].pdt is not None:
            m.pdt.clock(st["WB"].pdt if st["WB"] is not None else None,
                        asleep=m.waiting)        # MIPS Trace: what completes now
        if st["WB"] is not None:
            self.retired += 1
            if m.ejtag.sample_due:               # PC sampling: the graduating one
                m.ejtag.sample(st["WB"].pc)
            if st["WB"].perf is not None:
                m.perf.retire(*st["WB"].perf)
        if self.mdu_busy:
            self.mdu_busy -= 1
        # IF: an instruction miss being filled
        if_slot = st["IF"]
        if_wait = if_slot is not None and if_slot.fetch_wait > 0
        fetch_why = None
        if if_wait:
            if_slot.fetch_wait -= 1
            fetch_why = _pop_run(if_slot.fetch_runs)
        # MEM: an access still waiting freezes everything behind it
        mem = st["MEM"]
        if mem is not None and mem.mem_wait > 0:
            mem.mem_wait -= 1
            trace.stall = (f"memory: {mem.text} waits in MEM ({mem.mem_why}; "
                           f"{mem.mem_wait + 1} more cycle{'s' if mem.mem_wait else ''})")
            self.stalls["memory"] = self.stalls.get("memory", 0) + 1
            st["WB"] = None
            if m.perf.armed:                     # everything behind MEM waits
                why = _pop_run(mem.mem_runs)
                m.perf.add("stall")
                m.perf.add("alu_stall")
                if why in _MEM_EVENTS:
                    m.perf.add(_MEM_EVENTS[why])
            else:
                _pop_run(mem.mem_runs)
            self._flush_notes(ev)
            m.advance(1)
            self.history.append(trace)
            return trace

        # EX: execute (first EX cycle) — precise exceptions / interrupts here
        ex = st["EX"]
        late = None
        if ex is not None and (not ex.executed or ex.pending is not None):
            late = m.pending_cache_error()
        if late is not None:                             # imprecise cache error
            m.wake()
            flush_to = self._exception(ex, late, ev)
        elif ex is not None and (not ex.executed or ex.pending is not None):
            req = m.interrupt_request()
            if req is not None:
                m.wake()
                flush_to = self._exception(ex, req, ev)
            elif ex.fetch_error is not None:
                raise ex.fetch_error
            elif ex.held:                                # the debugger stopped here: go on
                self._pass = ex.pc
                flush_to = ex.pc
                ev.append(f"{ex.text}: held for the debugger, fetched again "
                          "(it executes in ID)")
            else:
                try:
                    if m._ihw and not ex.executed:       # hardware breakpoint / watch
                        m.instruction_checks(ex.pc)
                    exc = ex.fetch_exc or ex.pending
                    if exc is not None:
                        raise exc
                    if ex.dec is None:
                        if m.exceptions == "raise":
                            m.isa.decode(ex.word)        # raises ISADecodeError
                        raise _Arch(C.RI)
                    m.r6_slot_check(ex.dec, ex.in_slot, ex.forbidden)
                    if ex.info.control == "":
                        srcs = self._note_forwarding(ex, ex.info.reads_ex, "EX", ev)
                        m.last_view = None
                        try:
                            m.fetch_missed = ex.imiss
                            m.execute(ex.pc, ex.word, ex.dec, in_slot=ex.in_slot,
                                      sources=srcs)
                            ex.perf, m.last_events = m.last_events, None
                            ex.pdt, m.pdt_packet = m.pdt_packet, None
                        finally:
                            trace.ex_view = m.last_view
                            if trace.ex_view is not None:
                                trace.ex_view.fetch = ex.fetch_cache
                                trace.ex_view.fetch_tlb = ex.fetch_tlb
                        if m._redirect is not None:      # ERET
                            flush_to = m._redirect & M32
                            ev.append(f"{ex.text}: return to 0x{flush_to:08x}, "
                                      "flush IF/ID")
                        if m.halted:                     # exit syscall
                            self.fetching = False
                            st["IF"] = st["ID"] = None
                            ev.append(f"{ex.text}: program exit, pipeline drains")
                    self._memory_time(ex, ev)
                    ex.executed = True
                    ex.busy = ex.info.ex_cycles - 1
                    if ex.info.mdu_cycles:
                        # readable by an instruction entering EX L cycles on
                        self.mdu_busy = ex.info.mdu_cycles - 1
                except WatchpointHit as w:      # the debugger: before the access
                    self.watch_hit = w
                    ev.append(f"EX: {ex.text} stopped by a debugger watchpoint "
                              "(it executes in the next cycle)")
                except _Arch as a:
                    if self.timing is not None:
                        self.timing.take()               # the access did not happen
                    if self.utlb is not None:
                        self.utlb.take()
                    flush_to = self._exception(ex, a, ev)
        watched = self.watch_hit is not None and ex is not None and not ex.executed
        ex_holds = ex is not None and flush_to is None and (ex.busy > 0 or watched)
        if watched:
            pass
        elif ex_holds:
            ex.busy -= 1
            ev.append(f"EX busy: {ex.text} ({ex.busy + 1} more cycle"
                      f"{'s' if ex.busy else ''})")

        # ID: hazards, then branch / jump resolution
        stall: tuple[str, str] | None = None
        ids = st["ID"]
        if m.waiting and m.interrupt_request() is not None:
            m.wake()
            ev.append("WAIT: interrupt pending, wake up")
        if flush_to is None and ids is not None:
            if watched:
                stall = ("watch", f"{ex.text} stopped by a debugger watchpoint")
            elif ex_holds:
                stall = ("ex-busy", f"EX busy ({ex.text})")
            elif m.waiting:
                stall = ("wait", "WAIT: sleeping until an interrupt")
            else:
                stall = self._hazard(ids)
            # a debug exception is due (a single step completed in EX, an
            # imprecise data break): the branch must not execute here — it is
            # taken on the branch in EX
            if stall is None and ids.info and ids.info.control and not ids.executed \
                    and not m.debug_due and self._hold(ids, ev):
                pass
            elif stall is None and ids.info and ids.info.control and not ids.executed \
                    and not m.debug_due:
                srcs = self._note_forwarding(ids, ids.info.reads_id, "ID", ev)
                m.last_view = None
                m._forbidden = False
                try:
                    if m._ihw:                           # hardware breakpoint / watch
                        m.instruction_checks(ids.pc)
                    m.r6_slot_check(ids.dec, ids.in_slot, ids.forbidden)
                    m.fetch_missed = ids.imiss
                    res = m.execute(ids.pc, ids.word, ids.dec,
                                    in_slot=ids.in_slot, sources=srcs)
                    ids.perf, m.last_events = m.last_events, None
                    ids.pdt, m.pdt_packet = m.pdt_packet, None
                except _Arch as a:
                    ids.pending = a                      # taken in EX
                    res = None
                    m._redirect = None
                trace.id_view = m.last_view
                if trace.id_view is not None:
                    trace.id_view.fetch = ids.fetch_cache
                    trace.id_view.fetch_tlb = ids.fetch_tlb
                ids.executed = True
                nxt = st["IF"]
                if ids.info.compact:                     # R6: no delay slot
                    if m._redirect is not None:
                        branch_target = m._redirect & M32
                        m._redirect = None
                        nullify = True
                        ev.append(f"{ids.text}: taken -> 0x{branch_target:08x} "
                                  "(compact: no delay slot, fetched instruction squashed)")
                    elif ids.pending is None:
                        if nxt is not None and m._forbidden:
                            nxt.forbidden = True
                        ev.append(f"{ids.text}: not taken (next instruction is "
                                  "its forbidden slot)")
                    m._forbidden = False
                    res = None
                elif nxt is not None:
                    nxt.in_slot = True
                if ids.info.compact:
                    pass
                elif res == NULLIFY:
                    nullify = True
                    ev.append(f"{ids.text}: not taken, delay slot nullified")
                elif res is not None:
                    branch_target = res & M32
                    ev.append(f"{ids.text}: taken -> 0x{branch_target:08x} "
                              "(after the delay slot)")
                elif ids.info.control == "branch" and ids.pending is None:
                    ev.append(f"{ids.text}: not taken")
        if stall:
            kind, why = stall
            trace.stall = f"{kind}: {why}"
            self.stalls[kind] = self.stalls.get(kind, 0) + 1

        # advance the pipeline registers
        st["WB"] = st["MEM"]
        if flush_to is not None:
            st["MEM"] = ex if (ex is not None and ex.executed) else None
            st["EX"] = st["ID"] = st["IF"] = None
            self.fetch_pc = flush_to
            st["IF"] = self._fetch() if self.fetching else None
        elif ex_holds:
            st["MEM"] = None
        else:
            st["MEM"] = ex
            if stall:
                st["EX"] = None
            else:
                st["EX"] = st["ID"]
                if if_wait and not nullify:              # the instruction is not here yet
                    st["ID"] = None
                    trace.stall = (f"I-miss: fetch of 0x{if_slot.pc:08x} waits "
                                   f"({if_slot.fetch_wait + 1} more cycle"
                                   f"{'s' if if_slot.fetch_wait else ''})")
                    self.stalls["I-miss"] = self.stalls.get("I-miss", 0) + 1
                else:
                    st["ID"] = None if nullify else st["IF"]
                if branch_target is not None:
                    self.fetch_pc = branch_target
                if not if_wait or nullify:
                    st["IF"] = self._fetch() if self.fetching else None
        if m.perf.armed:
            self._count_stalls(old_ex, stall, ex_holds, fetch_why)
        self._flush_notes(ev)
        m.advance(1)
        self.history.append(trace)
        return trace

    def _hold(self, ids: Slot, ev: list[str]) -> bool:
        """A branch in ID the debugger stops at: it must not execute here,
        where the instruction in EX executes in the same cycle; it goes
        on to EX unexecuted, the precise point to stop at (as a debug
        exception due on a branch is taken in EX)."""
        if self.debug_stop is None:
            return False
        if ids.pc == self._pass:
            self._pass = None
            return False
        if not self.debug_stop(ids.pc):
            return False
        ids.held = True
        ev.append(f"{ids.text}: a debugger breakpoint: not resolved in ID, "
                  "it stops in EX")
        return True

    def _count_stalls(self, old_ex: Slot | None, stall, ex_holds: bool,
                      fetch_why: str | None) -> None:
        """This cycle's stall events for the performance counters (24K
        Table 6.47): nothing new entered EX ("stall"), ID got nothing from
        the fetch unit ("IFU stall", an I-cache miss), an interlock held
        the ALU pipeline ("ALU stall" and its cause)."""
        p, st = self.m.perf, self.stage
        if st["EX"] is None or st["EX"] is old_ex:
            p.add("stall")
        if st["ID"] is None and self.fetching:
            p.add("ifu_stall")
            if fetch_why == "I-miss":
                p.add("icache_stall")
            elif fetch_why == "uncached":
                p.add("uncached_stall")
        if stall or ex_holds:
            p.add("alu_stall")
            kind = stall[0] if stall else "ex-busy"
            if kind == "ex-busy":
                ex = st["EX"] if ex_holds else old_ex
                fpu = ex is not None and ex.dec is not None and "fpu" in ex.dec.entry["family"]
                p.add("fpu_stall" if fpu else "mdu_stall")
            elif kind in _HAZARD_EVENTS:
                p.add(_HAZARD_EVENTS[kind])

    def _flush_notes(self, ev: list[str]) -> None:
        ev.extend(self._fetch_notes)
        self._fetch_notes = []

    def _memory_time(self, s: Slot, ev: list[str]) -> None:
        """The cycles the instruction just executed must wait in MEM, and
        (non-blocking) when the register of a missing load is ready."""
        if self.timing is None and self.utlb is None:
            return
        wait, extra, notes = self.timing.take() if self.timing is not None else (0, None, [])
        runs = list(self.timing.taken_runs) if self.timing is not None else []
        if self.utlb is not None:                # DTLB miss: the LSU waits
            more, tlb_notes = self.utlb.take()
            wait += more
            notes = tlb_notes + notes
            if more:
                runs.insert(0, ("utlb", more))
        s.mem_wait = wait
        s.mem_runs = runs
        s.mem_why = notes[0] if notes else ""
        ev.extend(f"{s.text}: {n}" for n in notes)
        regs = [r for r in s.info.writes if isinstance(r, tuple)] if s.info else []
        for r in regs:
            self.reg_ready.pop(r, None)
        if extra is not None and extra > 0:
            for r in regs:
                self.reg_ready[r] = self.cycles + 1 + extra

    def _exception(self, s: Slot, exc: _Arch, ev: list[str]) -> int:
        m = self.m
        m.take_exception(exc, s.pc, s.word, s.dec, s.in_slot)
        name = C.EXC_NAMES.get(exc.code, str(exc.code))
        ev.append(f"exception {name} at 0x{s.pc:08x} ({s.text}): flush "
                  f"IF/ID/EX, fetch the handler at 0x{m.pc:08x}")
        return m.pc

    def _note_forwarding(self, s: Slot, needs, stage: str,
                         ev: list[str]) -> dict:
        """Record which operands come through a bypass; returns
        ``{resource: "forwarded EX/MEM -> EX from <instr>"}`` for the view."""
        sources: dict = {}
        if not self.forwarding:
            return sources
        for prod, path in ((self.stage["MEM"], "EX/MEM"),
                           (self.stage["WB"], "MEM/WB")):
            if prod is None or prod.info is None:
                continue
            for r in sorted(needs & prod.info.writes - {"hi", "lo"}, key=str):
                ev.append(f"forward {_rname(r)}: {path} -> {stage} "
                          f"({prod.text} -> {s.text})")
                sources[r] = f"forwarded {path} -> {stage} from {prod.text}"
                needs = needs - {r}
        return sources

    # -- running ------------------------------------------------------------

    STAGES = STAGES
    FETCH_STAGE, MEM_STAGE = "IF", "MEM"
    name = "5-stage"
    #: a debugger watchpoint stopped the instruction entering EX this
    #: cycle (it holds in EX, unexecuted); the Session raises it after
    watch_hit: WatchpointHit | None = None
    #: ``debug_stop(pc)``: does the debugger stop at this instruction (its
    #: breakpoints, run to cursor)?  A branch it stops at is held in ID
    debug_stop: Callable[[int], bool] | None = None
    #: the held branch being refetched: it executes in ID this time
    _pass: int | None = None

    def empty(self) -> bool:
        return all(v is None for v in self.stage.values())

    def next_slot(self) -> Slot | None:
        """The instruction that executes next: in EX not executed yet, else ID."""
        ex = self.stage["EX"]
        if ex is not None and not ex.executed:
            return ex
        return self.stage["ID"]

    def current_slot(self) -> Slot | None:
        return self.stage["EX"]

    def entering_ex(self) -> Slot | None:
        """The instruction in EX that has not executed yet (it executes in
        the next cycle), or None."""
        ex = self.stage["EX"]
        return ex if ex is not None and not ex.executed else None

    def run(self, *, max_cycles: int = 100_000, until_pc: int | None = None) -> int:
        """Cycle until the instruction at ``until_pc`` reaches EX (all older
        instructions have executed), the program exits and the pipeline is
        empty, or ``max_cycles``; returns cycles run."""
        start = self.cycles
        while self.cycles - start < max_cycles:
            if self.m.halted and all(v is None for v in self.stage.values()):
                break
            ex = self.stage["EX"]
            if until_pc is not None and ex is not None and ex.pc == until_pc:
                break
            self.cycle()
        return self.cycles - start

    def drain(self, keep: set[int] | None = None, *, max_cycles: int = 1000) -> None:
        """Stop fetching, drop the instructions in IF/ID/EX that are not
        at an address in ``keep`` (e.g. fetched past the end of the program)
        and cycle until the pipeline is empty."""
        self.fetching = False
        for s in ("IF", "ID", "EX"):
            slot = self.stage[s]
            if slot is not None and not slot.executed and (keep is None or slot.pc not in keep):
                self.stage[s] = None
        for _ in range(max_cycles):
            if all(v is None for v in self.stage.values()):
                return
            self.cycle()

    def chart(self, *, width: int = 60, last: bool = False) -> str:
        """Pipeline diagram: one row per instruction, one column per cycle
        (``IF ID EX ME WB``, ``--`` = stalled in ID).  Shows the first
        ``width`` cycles (``last``: the most recent ones) and the
        instructions present in them."""
        rows: dict[int, tuple[int, str, dict[int, str]]] = {}
        for t in self.history:
            for stage, v in t.stages.items():
                if v is None:
                    continue
                cell = "ME" if stage == "MEM" else stage
                if stage == "ID" and t.stall:
                    cell = "--"
                rows.setdefault(v.seq, (v.pc, v.text, {}))[2][t.cycle] = cell
        cycles = [t.cycle for t in self.history]
        cycles = cycles[-width:] if last else cycles[:width]
        head = f"{'':10} {'instruction':28} " + " ".join(f"{c:>2}"[-2:] for c in cycles)
        lines = [head]
        for seq in sorted(rows):
            pc, text, cells = rows[seq]
            if not any(c in cells for c in cycles):
                continue
            lines.append(f"{pc:08x}   {text[:28]:28} "
                         + " ".join(cells.get(c, "  ") for c in cycles))
        return "\n".join(lines)
