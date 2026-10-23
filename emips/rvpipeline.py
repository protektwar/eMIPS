"""The 5-stage pipeline of a RISC-V hart (PLAN M25 phase 4a).

Patterson & Hennessy's *Computer Organization and Design, RISC-V
Edition* (ch. 4) — IF, ID, EX, MEM, WB — with the choices of its section
4.8 ("Reducing the Delay of Branches") and the RISC-V ISA's own:

* **branches and jumps resolve in ID** (the comparator and the target
  adder moved there): the instruction fetched after them is predicted
  not taken; a taken branch, ``jal`` and ``jalr`` squash it (one bubble —
  RISC-V has no delay slot);
* **forwarding** EX/MEM → EX, MEM/WB → EX and EX/MEM → ID (the branch
  comparator, ``jalr``'s base), switchable off (operands then wait for
  WB; the register file writes in the first half of a cycle and reads in
  the second);
* **hazard detection** (4.7): a load followed by an instruction that uses
  its register (load-use: one bubble; a store's data too — the store data
  is needed in EX, as in the textbook's hazard unit), a branch / ``jalr``
  that needs the result of the instruction in EX (one bubble) or of a
  load (two);
* multiplies, divides and the F / D arithmetic hold EX for their
  latency (:data:`emips.pipeline.LATENCY`: ``MUL``, ``DIV``, ``FP_*``);
* **precise traps and interrupts**: an instruction's exception is taken
  when it reaches EX (older ones complete, younger ones are flushed); an
  interrupt is taken before the next instruction that would execute;
  ``mret`` / ``sret`` / ``dret``, ``ecall`` / ``ebreak``, ``wfi``,
  ``fence.i``, ``sfence.vma`` and every CSR write **serialize**: the
  instructions fetched after them are fetched again (what they change —
  privilege, translation, PMP, triggers, the counters — applies to every
  later fetch; fence.i's ordering of stores before fetches, unpriv.
  ch. 6);
* the clock: the hart's ``cycle`` / ``time`` follow the real cycles.

Results are **identical** to :class:`emips.simrv.RVMachine` (the
architectural reference): every instruction executes once, in program
order, through the same executors (:meth:`RVMachine.issue`) — when it
enters EX, or in ID for branches and jumps.  The pipeline adds the
timing; ``tests/test_rvpipeline.py`` checks both.  The memory timing of
the caches (:mod:`emips.timing`) is the MIPS pipeline's: an instruction
miss holds IF, a data access that waits holds MEM and freezes the stages
behind it; ``nonblocking`` lets a missing load go on until a reader needs
its register.

A debugger stops at precise points (:attr:`RVPipeline.debug_stop`): a
branch it stops at is not resolved in ID but held to EX, then fetched
again.  Debug Mode (the Debug spec's halt) stops the pipeline: what was
fetched is dropped, the hart resumes at dpc.

Every :meth:`RVPipeline.cycle` returns an :class:`emips.pipeline.CycleTrace`
— the stages, the control signals of the instruction in ID, stalls,
forwarding paths, branch decisions, flushes, traps — what the CLI / GUI
show (the MIPS pipeline's interface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .exceptions import FetchError, WatchpointHit
from .pipeline import LATENCY, CycleTrace, History, Info, Slot, StageView, _pop_run
from .riscv import GPR_NAMES, FPR_NAMES
from .tracerv import fp_operands

STAGES = ("IF", "ID", "EX", "MEM", "WB")

#: loads (their value exists at the end of MEM): the integer / F loads, LR,
#: SC (its result comes from the memory system) and the AMOs
_LOADS = {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "flw", "fld"}
_STORES = {"sb", "sh", "sw", "sd", "fsw", "fsd", "mips.sdp", "mips.swp"}
_BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
#: what serializes (see the module docstring), besides the CSR writes
_SERIALIZE = {"ecall": "ecall", "ebreak": "ebreak", "mret": "mret", "sret": "sret",
              "dret": "dret", "wfi": "wfi", "fence.i": "fence.i",
              "sfence.vma": "sfence.vma", "mips.ihb": "mips.ihb"}
#: the RARS services' registers (ecall: a7 the service, a0-a3 / fa0 in,
#: a0 / fa0 out; emips.rvservices)
_ECALL_READS = frozenset({("r", 17), ("r", 10), ("r", 11), ("r", 12), ("r", 13)})
_ECALL_WRITES = frozenset({("r", 10)})
#: rows whose register fields are reserved (unpriv. 2.7 / ch. 6: fence's rd and
#: rs1, fence.i's): they read and write no register
_NO_REGISTERS = {"fence", "fence.tso", "pause", "fence.i"}


@dataclass(frozen=True)
class RVInfo(Info):
    """:class:`emips.pipeline.Info` of a RISC-V instruction, and what it
    serializes for ("": nothing)."""

    serialize: str = ""


def _latency_of(name: str, latency: dict[str, int]) -> int:
    """The cycles a multiply / divide / F-D operation holds EX."""
    if name.startswith("mul"):
        return latency["MUL"]
    if name.startswith(("div", "rem")):
        return latency["DIV"]
    if not name.startswith("f") or name.startswith(("fence", "flw", "fld", "fsw", "fsd")):
        return 1
    op, fmt = (name.split(".") + ["s"])[:2]
    if op in ("fadd", "fsub"):
        return latency["FP_ADD"]
    if op == "fmul":
        return latency["FP_MUL"]
    if op in ("fmadd", "fmsub", "fnmadd", "fnmsub"):
        return latency["FP_MADD"]
    if op == "fdiv":
        return latency["FP_DIV_D" if fmt == "d" else "FP_DIV_S"]
    if op == "fsqrt":
        return latency["FP_SQRT_D" if fmt == "d" else "FP_SQRT_S"]
    if op == "fcvt":
        return latency["FP_CVT"]
    return latency["FP_CMP"]                    # compares, min / max, sign, moves, class


def _csr_writes(i) -> bool:
    """A CSR instruction that writes its CSR (Zicsr 6.1: CSRRS / CSRRC with
    rs1 = x0 and CSRRSI / CSRRCI with uimm = 0 only read)."""
    name = i.base.name
    if name in ("csrrw", "csrrwi"):
        return True
    if name in ("csrrs", "csrrc"):
        return i.f.get("rs1", 0) != 0
    if name in ("csrrsi", "csrrci"):
        return i.f.get("zimm", i.f.get("zimm5", 0)) != 0
    return False


_INFO: dict[tuple, RVInfo] = {}


def analyze(i, latency: dict[str, int] = LATENCY) -> RVInfo:
    """Registers read / written and the timing class of a decoded RISC-V
    instruction (``i``: an :class:`emips.riscv.RV`; a compressed one's
    operands are its expansion's)."""
    key = (i.raw, i.size, i.base.name, id(latency))
    hit = _INFO.get(key)
    if hit is not None:
        return hit
    name, f = i.base.name, i.f
    fp = fp_operands(name)

    def res(field: str):
        n = f.get(field)
        if n is None:
            return None
        if field in fp:
            return ("f", n)
        return ("r", n) if n else None

    reads = {r for r in (res("rs1"), res("rs2"), res("rs3")) if r is not None}
    writes = {r for r in (res("rd"), res("rd2")) if r is not None}
    if name in _NO_REGISTERS:                   # (their rd / rs1 fields are reserved)
        reads, writes = set(), set()
    control, reads_id, serialize = "", frozenset(), ""
    if name in _BRANCHES:
        control, reads_id, reads = "branch", frozenset(reads), set()
    elif name == "jal":
        control = "jump"
    elif name == "jalr":
        control, reads_id, reads = "jump", frozenset(reads), set()
    if name == "ecall":
        reads |= _ECALL_READS
        writes |= _ECALL_WRITES
    if name in _SERIALIZE:
        serialize = _SERIALIZE[name]
    elif name.startswith("csrr") and _csr_writes(i):
        serialize = "a CSR write"
    load = (name in _LOADS or name.startswith(("lr.", "sc.", "amo"))
            or name in ("mips.ldp", "mips.lwp"))
    if len(_INFO) >= 1 << 14:
        _INFO.clear()
    info = _INFO[key] = RVInfo(frozenset(reads_id), frozenset(reads), frozenset(writes),
                               "MEM" if load else "EX", _latency_of(name, latency), 0,
                               control, True, serialize)
    return info


def control_signals(i, info: RVInfo) -> dict[str, Any]:
    """The main control's signals for the instruction in ID (P&H RISC-V
    4.4, Figure 4.22, with the textbook's Branch extended by the jumps:
    ALUSrc, MemtoReg, RegWrite, MemRead, MemWrite, Branch; RegDst is
    always rd in RISC-V — "-" when nothing is written)."""
    name, f = i.base.name, i.f
    load = info.ready == "MEM" and not name.startswith("sc.")
    store = name in _STORES
    write = any(r[0] == "r" for r in info.writes) and name != "ecall"   # (an x register)
    imm = any(k.startswith("imm") or k in ("shamt", "zimm") for k in f) \
        or name in ("lui", "auipc")
    return {"RegDst": "rd" if write else "-",
            "ALUSrc": "imm" if imm else "reg" if "rs2" in f else "-",
            "MemtoReg": load, "RegWrite": write, "MemRead": load,
            "MemWrite": store or name.startswith(("sc.", "amo")),
            "Branch": info.control == "branch", "Jump": info.control == "jump",
            "ALUOp": name}


def _rname(r) -> str:
    kind, n = r
    return f"x{n} ({GPR_NAMES[n]})" if kind == "r" else f"f{n} ({FPR_NAMES[n]})"


class RVPipeline:
    """The 5-stage pipeline driving an :class:`emips.simrv.RVMachine`
    (the interface of :class:`emips.pipeline.Pipeline`: ``cycle``,
    ``run``, ``drain``, ``chart``, the stages, ``stalls``, ``retired``).

    ``forwarding=False`` disables every bypass; ``latency`` overrides
    entries of :data:`emips.pipeline.LATENCY`; ``memory`` /
    ``nonblocking`` the memory timing (with caches).  The hart's pc is
    the first fetch address; afterwards the pipeline owns the flow.
    """

    STAGES = STAGES
    FETCH_STAGE, MEM_STAGE = "IF", "MEM"
    name = "5-stage"
    #: a debugger watchpoint stopped the instruction entering EX this cycle
    watch_hit: WatchpointHit | None = None
    #: ``debug_stop(pc)``: does the debugger stop there (a branch it stops
    #: at is held in ID, not resolved)?
    debug_stop: Callable[[int], bool] | None = None
    _pass: int | None = None

    def __init__(self, machine, *, forwarding: bool = True,
                 latency: dict[str, int] | None = None, trace: bool = True,
                 memory: dict[str, int] | None = None, nonblocking: bool = False) -> None:
        self.m = machine
        machine.tracing = trace
        self.forwarding = forwarding
        self.latency = dict(LATENCY, **(latency or {}))
        self.timing = machine.caches.timing if machine.caches is not None else None
        if self.timing is not None:
            self.timing.configure(memory, nonblocking=nonblocking)
        self.nonblocking = nonblocking
        self.reg_ready: dict = {}
        self._fetch_notes: list[str] = []
        self.stage: dict[str, Slot | None] = {s: None for s in STAGES}
        self.fetch_pc = machine.pc
        self.cycles = 0
        self.retired = 0
        self.stalls: dict[str, int] = {}
        self._seq = 0
        self.history = History()
        self.fetching = True
        #: Debug Mode stopped the pipeline: fetch again at the pc on resume
        self.parked = False
        #: ID / IF stalled in the last cycle (the P8700's edge-counted events)
        self._was_stalled = self._was_fetch_stalled = False
        from .sessionrv import _disassembler
        self._dis = _disassembler(machine.xlen, machine.extensions, False)
        self.stage["IF"] = self._fetch()

    # -- fetch ----------------------------------------------------------------

    def _fetch(self) -> Slot:
        slot = self._fetch_slot()
        if self.timing is not None:
            wait, _, notes = self.timing.take()
            slot.fetch_wait = wait
            slot.fetch_runs = list(self.timing.taken_runs)
            self._fetch_notes += [f"fetch 0x{slot.pc:08x}: {n}" for n in notes]
        return slot

    def _fetch_slot(self) -> Slot:
        from .simrv import RVTrap
        m, pc = self.m, self.fetch_pc
        self._seq += 1
        m.fetch_accesses = []
        try:
            i = m.fetch(pc, triggers=False)      # (the execute triggers: when it issues)
        except RVTrap as t:
            from .simrv import CAUSE_NAMES
            self.fetch_pc = (pc + 2) & m.amask
            return Slot(self._seq, pc, None, None,
                        f"(fetch: {CAUSE_NAMES.get(t.cause, t.cause)})", None,
                        fetch_exc=t, fetch_cache=m.fetch_accesses)
        except FetchError as exc:
            self.fetch_pc = (pc + 2) & m.amask
            return Slot(self._seq, pc, None, None, "(no code here)", None,
                        fetch_error=exc, fetch_cache=m.fetch_accesses)
        self.fetch_pc = (pc + i.size) & m.amask
        return Slot(self._seq, pc, i.raw, None, self._dis.text(i.raw, i.size, pc),
                    analyze(i, self.latency), fetch_cache=m.fetch_accesses,
                    insn=i, size=i.size)

    def _goto(self, target: int) -> None:
        self.fetch_pc = target & self.m.amask

    # -- views ------------------------------------------------------------------

    @staticmethod
    def _view(slot: Slot | None, control: bool = False) -> StageView | None:
        if slot is None:
            return None
        views = slot.views
        if views is None:
            views = slot.views = [None, None]
        v = views[control]
        if v is None or v.fetch_wait != slot.fetch_wait or v.mem_wait != slot.mem_wait:
            ctl = (control_signals(slot.insn, slot.info)
                   if control and slot.insn is not None else None)
            comp = (f"RVC 16-bit 0x{slot.word:04x}"
                    if slot.size == 2 and slot.word is not None else "")
            v = views[control] = StageView(slot.seq, slot.pc, slot.text, ctl,
                                           slot.fetch_cache, slot.fetch_wait, slot.mem_wait,
                                           slot.fetch_tlb, slot.word, comp)
        return v

    # -- hazards ------------------------------------------------------------------

    def _hazard(self, s: Slot) -> tuple[str, str] | None:
        """``(kind, explanation)`` when the instruction in ID cannot advance."""
        if s.info is None:
            return None
        needs = s.info.reads_ex | s.info.reads_id
        if self.reg_ready:
            for r in sorted(needs):
                ready = self.reg_ready.get(r, 0)
                if ready > self.cycles:
                    return ("miss-wait", f"{_rname(r)} from a load still missing "
                                         f"({ready - self.cycles} more cycle(s))")
        for prod, where in ((self.stage["EX"], "EX"), (self.stage["MEM"], "MEM")):
            if prod is None or prod.info is None:
                continue
            common = needs & prod.info.writes
            if not common:
                continue
            r = _rname(sorted(common)[0])
            if not self.forwarding:
                return ("no-forwarding", f"{r} not written back yet "
                                         f"({prod.text} in {where})")
            if s.info.reads_id & common:
                if where == "EX":
                    return ("branch", f"{s.text} needs {r} from {prod.text} (in EX)")
                if prod.info.ready == "MEM":
                    return ("branch", f"{s.text} needs {r} from load {prod.text} (in MEM)")
            elif where == "EX" and prod.info.ready == "MEM":
                return ("load-use", f"{r} from {prod.text}")
        return None

    def _note_forwarding(self, s: Slot, needs, stage: str, ev: list[str]) -> dict:
        sources: dict = {}
        if not self.forwarding:
            return sources
        for prod, path in ((self.stage["MEM"], "EX/MEM"), (self.stage["WB"], "MEM/WB")):
            if prod is None or prod.info is None:
                continue
            for r in sorted(needs & prod.info.writes):
                ev.append(f"forward {_rname(r)}: {path} -> {stage} ({prod.text} -> {s.text})")
                sources[r] = f"forwarded {path} -> {stage} from {prod.text}"
                needs = needs - {r}
        return sources

    # -- issuing ------------------------------------------------------------------

    def _boundary(self, s: Slot, ev: list[str]) -> str | None:
        """Before the instruction ``s`` executes (the hart's step does the
        same before every instruction): a halt request (Debug Mode), an
        interrupt — "flush" —, or WFI still sleeping — "wait"."""
        from .simrv import CAUSE_HALTREQ, INTERRUPT_NAMES
        m = self.m
        if m.halt_request:
            m.halt_request = False
            m.pc = s.pc
            m.enter_debug(CAUSE_HALTREQ)
            ev.append(f"halt request before {s.text}: Debug Mode, flush the pipeline")
            return "flush"
        req = None if m._stepping else m.interrupt_request()
        if req is not None:
            m.take_interrupt(req, s.pc)
            ev.append(f"interrupt {INTERRUPT_NAMES.get(req[0], req[0])} before {s.text}: "
                      f"flush, fetch the handler at 0x{m.pc:08x}")
            return "flush"
        if m.waiting:
            if m.csrs.mip() & m.csrs.mie:
                m.waiting = False
                ev.append("WFI: an interrupt is pending, wake up")
            else:
                return "wait"
        return None

    def _issue(self, s: Slot, stage: str, ev: list[str], trace: CycleTrace) -> bool:
        """Execute ``s`` (in EX, or a branch / jump in ID); True when the
        flow leaves the predicted path (a trap, a taken branch, a jump,
        a serializing instruction, Debug Mode, the end) — then the hart's
        pc is where to fetch."""
        m = self.m
        if s.fetch_error is not None:
            raise s.fetch_error
        needs = s.info.reads_id if stage == "ID" else s.info.reads_ex if s.info else frozenset()
        m.rv_sources = self._note_forwarding(s, needs, stage, ev) if s.info else None
        logged = len(m.exception_log)
        m.last_view = None
        m.pc = s.pc
        retrigger = m.tm.fetch                   # (execute triggers: fetch it again)
        try:
            m.issue(s.pc, None if retrigger or s.insn is None else s.insn, s.fetch_exc,
                    clock=False)
        finally:
            m.rv_sources = None
            view = m.last_view
            if view is not None:
                view.fetch = s.fetch_cache
            if stage == "ID":
                trace.id_view = view
            else:
                trace.ex_view = view
        s.executed = True
        seq = (s.pc + s.size) & m.amask
        if len(m.exception_log) != logged or m.debug_mode:
            e = m.exception_log[-1] if m.exception_log else {}
            what = e.get("error") or e.get("name") or "a trap"
            if m.debug_mode:
                ev.append(f"{s.text}: Debug Mode — the pipeline stops")
            else:
                ev.append(f"{s.text}: {what} — flush, fetch the handler at 0x{m.pc:08x}")
            s.executed = e.get("debug") is None and "cause" not in e and not m.debug_mode
            return True
        if m.halted:
            ev.append(f"{s.text}: program exit, pipeline drains")
            return True
        if s.info is not None and s.info.control:
            if m.pc != seq:
                ev.append(f"{s.text}: taken -> 0x{m.pc:08x} (the next instruction "
                          "fetched is squashed)")
                return True
            ev.append(f"{s.text}: not taken")
            return False
        if s.info is not None and s.info.serialize:
            ev.append(f"{s.text}: {s.info.serialize} serializes — the instructions "
                      f"after it are fetched again from 0x{m.pc:08x}")
            return True
        return m.pc != seq

    # -- one clock cycle --------------------------------------------------------

    def _tick(self) -> None:
        self.m.advance(1)
        self.m.counter_written.clear()

    def cycle(self) -> CycleTrace:
        m, st = self.m, self.stage
        self.cycles += 1
        trace = CycleTrace(self.cycles, {k: self._view(v, k == "ID") for k, v in st.items()})
        ev = trace.events
        if st["WB"] is not None:
            self.retired += 1
        if_slot = st["IF"]
        if_wait = if_slot is not None and if_slot.fetch_wait > 0
        if if_wait:
            if_slot.fetch_wait -= 1
            _pop_run(if_slot.fetch_runs)
        mem = st["MEM"]
        if mem is not None and mem.mem_wait > 0:          # the access waits: all behind
            mem.mem_wait -= 1
            _pop_run(mem.mem_runs)
            trace.stall = (f"memory: {mem.text} waits in MEM ({mem.mem_why}; "
                           f"{mem.mem_wait + 1} more cycle{'s' if mem.mem_wait else ''})")
            self.stalls["memory"] = self.stalls.get("memory", 0) + 1
            st["WB"] = None
            p = m.p8700
            if p is not None and p.counters.armed and mem.insn is not None \
                    and mem.insn.base.name in _LOADS:
                p.counters.add(35)                 # (load_blocked: MD01502 Table 11.4)
            self._flush_notes(ev)
            self._tick()
            self.history.append(trace)
            return trace
        if self.parked:                                    # Debug Mode: halted
            st["WB"], st["MEM"] = st["MEM"], st["EX"] if st["EX"] and st["EX"].executed \
                else None
            st["EX"] = st["ID"] = st["IF"] = None
            if not m.debug_mode:                           # resumed: from dpc
                self.parked = False
                self._goto(m.pc)
                ev.append(f"resumed: fetch 0x{m.pc:08x}")
                if self.fetching:
                    st["IF"] = self._fetch()
            else:
                trace.stall = "debug: the hart is in Debug Mode"
            self._flush_notes(ev)
            self._tick()
            self.history.append(trace)
            return trace

        flush = False                                      # refetch from m.pc
        squash = False                                     # a branch in ID was taken
        stall: tuple[str, str] | None = None
        ex = st["EX"]
        ex_wait = False
        if ex is not None and not ex.executed:
            if ex.held:                                    # the debugger stopped here
                self._pass = ex.pc
                m.pc = ex.pc
                flush = True
                ev.append(f"{ex.text}: held for the debugger, fetched again "
                          "(it resolves in ID)")
            else:
                b = self._boundary(ex, ev)
                if b == "flush":
                    flush = True
                elif b == "wait":
                    ex_wait = True
                else:
                    try:
                        flush = self._issue(ex, "EX", ev, trace)
                    except WatchpointHit as w:
                        self.watch_hit = w
                        ev.append(f"EX: {ex.text} stopped by a debugger watchpoint "
                                  "(it executes in the next cycle)")
                        ex_wait = True
                    else:
                        if ex.executed:
                            self._memory_time(ex, ev)
                            ex.busy = ex.info.ex_cycles - 1 if ex.info else 0
        ex_holds = ex is not None and not flush and (ex_wait or ex.busy > 0)
        if ex_holds and not ex_wait:
            ex.busy -= 1
            ev.append(f"EX busy: {ex.text} ({ex.busy + 1} more cycle"
                      f"{'s' if ex.busy else ''})")

        ids = st["ID"]
        if not flush and ids is not None:
            if ex_holds:
                stall = (("wait", "WFI: sleeping until an interrupt") if ex_wait and
                         self.watch_hit is None else ("ex-busy", f"EX busy ({ex.text})"))
            else:
                stall = self._hazard(ids)
            if stall is None and ids.info is not None and ids.info.control \
                    and not ids.executed and not self._hold(ids, ev):
                b = self._boundary(ids, ev)
                if b == "flush":
                    flush = True
                    ids.executed = False
                    st["ID"] = None                       # (it was not executed)
                elif b == "wait":
                    stall = ("wait", "WFI: sleeping until an interrupt")
                else:
                    taken = self._issue(ids, "ID", ev, trace)
                    if not ids.executed or m.debug_mode:
                        flush = True                      # a trap in ID: precise here
                        if not ids.executed:
                            st["ID"] = None
                    elif taken:
                        squash = True
        if stall:
            kind, why = stall
            trace.stall = f"{kind}: {why}"
            self.stalls[kind] = self.stalls.get(kind, 0) + 1

        p = m.p8700                                        # (its performance counters)
        if p is not None and p.counters.armed:
            dropped = (st["IF"] is not None) if squash else \
                (st["IF"] is not None) + (st["ID"] is not None and not st["ID"].executed) \
                if flush else 0
            self._count(p.counters, ids, stall, flush or squash, if_wait, dropped)

        # the pipeline registers
        st["WB"] = st["MEM"]
        if flush:
            st["MEM"] = ex if ex is not None and ex.executed and not ex_holds else None
            if ex_holds:
                st["MEM"] = None
            keep_id = st["ID"] is not None and st["ID"].executed
            st["EX"] = st["ID"] if keep_id else (ex if ex_holds else None)
            st["ID"] = st["IF"] = None
            if m.debug_mode:
                self.parked = True
            elif m.halted:
                self.fetching = False
            else:
                self._goto(m.pc)
                if self.fetching:
                    st["IF"] = self._fetch()
        elif ex_holds:
            st["MEM"] = None
            if squash:                                   # (cannot happen: ID stalled)
                st["IF"] = None
        else:
            st["MEM"] = ex
            if stall:
                st["EX"] = None
            else:
                st["EX"] = st["ID"]
                if if_wait and not squash:
                    st["ID"] = None
                    trace.stall = (f"I-miss: fetch of 0x{if_slot.pc:08x} waits "
                                   f"({if_slot.fetch_wait + 1} more cycle"
                                   f"{'s' if if_slot.fetch_wait else ''})")
                    self.stalls["I-miss"] = self.stalls.get("I-miss", 0) + 1
                else:
                    st["ID"] = None if squash else st["IF"]
                if squash:
                    self._goto(m.pc)
                if not if_wait or squash:
                    st["IF"] = self._fetch() if self.fetching else None
        self._flush_notes(ev)
        self._tick()
        self.history.append(trace)
        return trace

    def _count(self, hpm, ids: Slot | None, stall, redirect: bool, if_wait: bool,
               dropped: int) -> None:
        """The P8700's pipeline events of this cycle (MD01502 Table 11.4;
        emips.p8700): 19 a stall of ID begins, 20 of IF (an I-cache miss),
        22 ID empty, 25 / 27 / 29 a taken branch / a return / a JALR
        redirects fetch (the 5-stage pipeline predicts not taken: each is
        a misprediction), 32 a redirect, 35 a load holds issue, 137 the
        fetched instructions a redirect drops, 138 a cycle IF waits for a
        miss."""
        kind = stall[0] if stall else None
        if ids is None:
            hpm.add(22)
        if kind and not self._was_stalled:
            hpm.add(19)
        self._was_stalled = bool(kind)
        if if_wait:
            if not self._was_fetch_stalled:
                hpm.add(20)
            hpm.add(138)
        self._was_fetch_stalled = if_wait
        if kind in ("load-use", "miss-wait"):
            hpm.add(35)
        if redirect:
            hpm.add(32)
            if dropped:
                hpm.add(137, dropped)
            if ids is not None and ids.executed and ids.insn is not None:
                name, f = ids.insn.base.name, ids.insn.f
                if name in _BRANCHES:
                    hpm.add(25)
                elif name == "jalr":
                    hpm.add(29)
                    if f.get("rd", 0) == 0 and f.get("rs1", 0) in (1, 5):
                        hpm.add(27)

    def _hold(self, ids: Slot, ev: list[str]) -> bool:
        """A branch the debugger stops at goes to EX unresolved (the
        precise point: everything older executed)."""
        if self.debug_stop is None:
            return False
        if ids.pc == self._pass:
            self._pass = None
            return False
        if not self.debug_stop(ids.pc):
            return False
        ids.held = True
        ev.append(f"{ids.text}: a debugger breakpoint: not resolved in ID, it stops in EX")
        return True

    def _flush_notes(self, ev: list[str]) -> None:
        ev.extend(self._fetch_notes)
        self._fetch_notes = []

    def _memory_time(self, s: Slot, ev: list[str]) -> None:
        if self.timing is None:
            return
        wait, extra, notes = self.timing.take()
        s.mem_wait = wait
        s.mem_runs = list(self.timing.taken_runs)
        s.mem_why = notes[0] if notes else ""
        ev.extend(f"{s.text}: {n}" for n in notes)
        regs = list(s.info.writes) if s.info else []
        for r in regs:
            self.reg_ready.pop(r, None)
        if extra is not None and extra > 0:
            for r in regs:
                self.reg_ready[r] = self.cycles + 1 + extra

    # -- running --------------------------------------------------------------------

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
        ex = self.stage["EX"]
        return ex if ex is not None and not ex.executed else None

    def run(self, *, max_cycles: int = 100_000, until_pc: int | None = None) -> int:
        start = self.cycles
        while self.cycles - start < max_cycles:
            if self.m.halted and self.empty():
                break
            ex = self.stage["EX"]
            if until_pc is not None and ex is not None and ex.pc == until_pc:
                break
            self.cycle()
        return self.cycles - start

    def drain(self, keep: set[int] | None = None, *, max_cycles: int = 1000) -> None:
        self.fetching = False
        for s in ("IF", "ID", "EX"):
            slot = self.stage[s]
            if slot is not None and not slot.executed and (keep is None or slot.pc not in keep):
                self.stage[s] = None
        for _ in range(max_cycles):
            if self.empty():
                return
            self.cycle()

    def chart(self, *, width: int = 60, last: bool = False) -> str:
        from .pipeline import Pipeline
        return Pipeline.chart(self, width=width, last=last)
