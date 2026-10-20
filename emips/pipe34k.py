"""The MIPS 34K pipeline (PLAN M19 phase 2): the 24K pipeline
(:mod:`emips.pipe24k`) with the MT Module's thread contexts.

Reference: *MIPS32 34K Processor Core Family Software User's Manual*
(MD00534 rev. 1.13) chapter 2 (2.1 the stages, 2.2 the fetch unit, 2.2.2
the return prediction stack, the per-TC ITLB entry) and chapter 13 (the
dispatch scheduler and the policy managers).

Stages::

    IF  IS  IT  RF  AG  EX  MS  ER  WB

* **IT** is the thread-select stage: "In a single TC configuration, this
  stage is bypassed" (2.1) — with one TC this is the 24K model; with more
  an instruction is in RF three cycles after its IF at the earliest, and
  a misprediction costs five cycles, not four (2.2, Figure 2.5).
* **Fetch**: before IF "a mux selects the TC that will start a fetch":
  each cycle one TC that can fetch — runnable, its **instruction buffer**
  (8 entries per TC, 2.2) not full, not waiting for a jump — fetches a
  64-bit bundle, round-robin among them.  A TC whose fetch misses the
  I-cache waits; the others go on.
* **Dispatch** (2.2, 13.1): in IT the dispatch scheduler chooses one of the
  TCs with an instruction ready, by the policy manager (round-robin, WRR's
  groups and rotation, WRR2's throttles: :meth:`emips.mt.Processor.choose`).
* **The skid buffer** (2.2): an instruction of a TC that waits for a
  load that missed — and the TC's younger instructions already issued —
  go back to its instruction buffer; the TC dispatches again when the data
  is there, the other TCs go on meanwhile.  The same happens to a TC that
  stops being runnable (WAIT, a blocked YIELD or ITC access, an exception
  that makes its VPE exclusive, TCHalt).
* **Shared**: the back end, the MDU / FPU / LSU, the BHT; the 4-entry
  return prediction stack only while one TC is runnable (2.2.2: "A TC
  begins to utilize the RPS when it is the only TC that is runnable").
* **Per TC**: the fetch state and instruction buffer, the register
  scoreboard (its own GPRs, HI / LO, accumulators); per VPE: its CP0 and
  the CP0 hazards (Table 2.7).

Execution is in EX, in program order per TC, by the Machine of the TC's
VPE with the TC bound (:meth:`emips.mt.Processor.bind`).  Every choice is
in ``sim.DEVIATIONS["34K pipeline"]``.
"""

from __future__ import annotations

from operator import attrgetter
from typing import Any

from . import cp0 as C
from .exceptions import Blocked
from .pipe24k import BACK, IB_ENTRIES, STAGES, STAGES16, Pipeline24K, Slot24, _kind, _sv
from .pipeline import CycleTrace, _pop_run, _rname, enter

M32 = 0xFFFFFFFF
#: the per-TC state the pipeline swaps in while it works on a TC
_FRONT = ("queue", "fetch_pc", "fetch_isa", "recode_free", "fetching", "fetch_at",
          "blocked", "after_slot", "next_marks", "last_fetched", "scoreboard",
          "recovering", "macro_until")
#: a front end holds the buffer and the IF / IS (/ IT) bundles
FRONT_ENTRIES = IB_ENTRIES + 4
#: the _FRONT attributes of an object, as a tuple (copied with one
#: ``__dict__.update``: no attribute is a property, _Front has no slots)
_front_values = attrgetter(*_FRONT)


class _Front:
    """One TC's fetch state, instruction buffer and scoreboard."""

    def __init__(self, pc: int, isa: int) -> None:
        self.queue: list[Slot24] = []
        self.fetch_pc = pc
        self.fetch_isa = isa
        self.recode_free = 0
        self.fetching = True
        self.fetch_at = 1
        self.blocked = False
        self.after_slot: tuple | None = None
        self.next_marks: set[str] = set()
        self.last_fetched: Slot24 | None = None
        self.scoreboard: dict = {}
        self.recovering: int | None = None
        self.macro_until = 0
        #: the TC's architectural state changed under the pipeline (a
        #: restart, a release): refetch from its PC before the next fetch
        self.stale = False


class Pipeline34K(Pipeline24K):
    """The 34K pipeline of an MT processor (:class:`emips.mt.Processor`):
    :meth:`cycle` returns a :class:`CycleTrace` as the other models."""

    STAGES = STAGES
    name = "34K"

    def __init__(self, processor, *, trace: bool = True, memory: dict | None = None,
                 nonblocking: bool = False, **_ignored: Any) -> None:
        self.p = processor
        self._tc = None
        super().__init__(processor.vpes[0], trace=trace, memory=memory,
                         nonblocking=nonblocking)
        for m in processor.vpes:
            m.tracing = trace
            mmu = m.mmu
            if hasattr(mmu, "take"):
                mmu.timing = True
                mmu.utlb_cost = self.utlb.utlb_cost if self.utlb is not None else 3
                mmu.take()
        #: the IT stage: with more than one TC (2.1)
        self.it = len(processor.tcs) > 1
        self.fronts: dict[int, _Front] = {}
        for tc in processor.tcs:
            self.fronts[tc.n] = _Front(processor.attr(tc, "pc"), processor.attr(tc, "isa_mode"))
        self.hazards_of = [[] for _ in processor.vpes]
        self.fetch_last: int = -1
        #: per TC: dispatch waits until this cycle (the skid buffer: the data
        #: of a missing load)
        self.held_until: dict[int, int] = {}
        #: replayed instructions (the 34K's "Replays" event)
        self.replays = 0
        self.dispatched = {tc.n: 0 for tc in processor.tcs}
        processor.listener = self._changed
        self.p.prepare()
        self._on(self.p.tcs[0])

    # -- the per-TC state ----------------------------------------------------------------

    def _on(self, tc) -> None:
        """Work on ``tc``: its front end, scoreboard, VPE (Machine, CP0
        hazards, micro-TLBs) and the TC bound there."""
        if self._tc is not tc:
            if self._tc is not None:
                self.fronts[self._tc.n].__dict__.update(zip(_FRONT, _front_values(self)))
                self.hazards_of[self._tc.vpe] = self.hazards
            f = self.fronts[tc.n]
            self.__dict__.update(zip(_FRONT, _front_values(f)))
            self.m = self.p.vpes[tc.vpe]
            self.hazards = self.hazards_of[tc.vpe]
            mmu = self.m.mmu
            self.utlb = mmu if hasattr(mmu, "take") else None
            self._tc = tc
        self.p.bind(tc)
        if self.m.caches is not None:
            self.m.caches.m = self.m

    def _changed(self, tc, what: str) -> None:
        """Software stopped ``tc`` (TCHalt, TCStatus.A = 0): its PC is the
        oldest instruction it has not executed (with its delay slot state);
        or let it go on (TCRestart, FORK, a release): it fetches from its
        PC.  Either way its fetch follows the architectural PC again."""
        if what == "stop":
            self._sync_pc(tc)
        self.fronts[tc.n].stale = True

    def _unexecuted(self, tc) -> list[Slot24]:
        f = self._front(tc)
        mine = [s for s in self.stage.values()
                if s is not None and getattr(s, "tc", None) is tc and not s.executed]
        return sorted(mine + list(f.queue), key=lambda s: s.seq)

    def _sync_pc(self, tc) -> None:
        """The TC's architectural PC from the pipeline: its oldest
        unexecuted instruction (a delay slot: armed, the instruction after
        it next), or where it fetches next."""
        p = self.p
        todo = self._unexecuted(tc)
        f = self.fronts[tc.n] if tc is not self._tc else None
        fetch_pc = f.fetch_pc if f is not None else self.fetch_pc
        after = f.after_slot if f is not None else self.after_slot
        if todo:
            s = todo[0]
            p.set_attr(tc, "pc", s.pc)
            p.set_attr(tc, "isa_mode", s.isa)
            if s.in_slot:
                nxt = todo[1].addr if len(todo) > 1 else (
                    after[1] if after is not None and after[0] == "target" else fetch_pc)
                p.set_attr(tc, "slot_armed", True)
                p.set_attr(tc, "slot_target", nxt & M32)
            else:
                p.set_attr(tc, "slot_armed", False)
                p.set_attr(tc, "slot_target", None)
        else:
            p.set_attr(tc, "pc", fetch_pc & M32)
            p.set_attr(tc, "slot_armed", False)

    def _front(self, tc) -> _Front:
        if tc is self._tc:
            self.fronts[tc.n].__dict__.update(zip(_FRONT, _front_values(self)))
        return self.fronts[tc.n]

    def _resync(self, tc, t: int) -> None:
        """Drop ``tc``'s instructions that have not executed and fetch from
        its architectural PC (the delay slot's branch target after it)."""
        self._on(tc)
        self._drop_unexecuted(tc)
        p = self.p
        self.queue = []
        self.blocked = False
        self.next_marks = set()
        self.last_fetched = None
        pc, isa = p.attr(tc, "pc"), p.attr(tc, "isa_mode")
        self.fetch_pc, self.fetch_isa = pc & M32, isa
        self.after_slot = None
        if p.attr(tc, "slot_armed"):
            target = p.attr(tc, "slot_target")
            self.after_slot = ("target", target, t) if target is not None else None
        self.fetching = True
        self.fetch_at = t
        self.fronts[tc.n].stale = False

    def _drop_unexecuted(self, tc) -> None:
        for name in ("RF", "AG"):
            s = self.stage[name]
            if s is not None and getattr(s, "tc", None) is tc:
                self.stage[name] = None
                s.timeline[self.cycles] = "xx"

    def _park(self, tc, t: int, why: str, ev: list[str]) -> None:
        """``tc`` cannot go on now: its issued, unexecuted instructions go
        back to its instruction buffer (2.2: the skid buffer), in order;
        the TC's PC is the oldest of them (TCRestart, a later resync)."""
        back = []
        for name in ("AG", "RF"):
            s = self.stage[name]
            if s is not None and getattr(s, "tc", None) is tc:
                self.stage[name] = None
                back.append(s)
        if not back:
            return
        self._on(tc)
        for s in back:
            s.avail = t + 1
            s.timeline[t] = "<<"
        self.queue = sorted(back + self.queue, key=lambda s: s.seq)
        self.replays += len(back)
        ev.append(f"TC{tc.n}: {why}: {len(back)} instruction{'s' if len(back) > 1 else ''} "
                  f"back to its instruction buffer")

    # -- the fetch unit -------------------------------------------------------------------

    def _fetchable(self, tc, t: int) -> bool:
        f = self._front(tc)
        return (self.p.eligible(tc) or bool(self.p.attr(tc, "waiting"))) and f.fetching \
            and not f.blocked and f.fetch_at <= t and len(f.queue) <= FRONT_ENTRIES - 2

    def _fetch_tc(self, t: int) -> list[Slot24]:
        """Before IF: one TC that can fetch, round-robin (2.2)."""
        tcs = self.p.tcs
        n = len(tcs)
        for k in range(1, n + 1):
            tc = tcs[(self.fetch_last + k) % n]
            if self.fronts[tc.n].stale:
                self._resync(tc, t)
            if not self._fetchable(tc, t):
                continue
            self._on(tc)
            out = self._fetch(t)
            if out:
                self.fetch_last = tc.n
                for s in out:
                    s.tc = tc
                    if self.it:                  # the IT stage (2.1)
                        s.avail += 1
                        s.text = f"[TC{tc.n}] {s.text}"
                return out
        return []

    def _predict(self, s: Slot24):
        """2.2.2: the return prediction stack only while one TC is runnable;
        else a JR $31 is not predicted (the fetch waits)."""
        name = s.dec.name if s.dec is not None else ""
        if name in ("JR", "JALR") and (s.word >> 21) & 31 == 31:
            if sum(1 for tc in self.p.tcs if self.p.eligible(tc)) > 1:
                return "fetch waits (the RPS is shared: several TCs run)", None
        return super()._predict(s)

    # -- readiness ------------------------------------------------------------------------

    def _waits_for(self, s: Slot24, regs, t: int):
        """As the 24K, with the producers of the same TC only."""
        if not regs:
            return None
        ag = self.stage["AG"]
        if ag is not None and getattr(ag, "tc", None) is not getattr(s, "tc", None):
            self.stage["AG"] = None             # another TC's: not a producer here
            try:
                return super()._waits_for(s, regs, t)
            finally:
                self.stage["AG"] = ag
        return super()._waits_for(s, regs, t)

    def _flush_after(self, seq: int) -> None:
        """Kill the current TC's instructions younger than ``seq``."""
        tc = self._tc
        others = {}
        for name in BACK:
            s = self.stage[name]
            if s is not None and getattr(s, "tc", None) is not tc:
                others[name] = s
                self.stage[name] = None
        try:
            super()._flush_after(seq)
        finally:
            for name, s in others.items():
                self.stage[name] = s
        mine = [*self.queue, *(s for s in self.stage.values()
                               if s is not None and getattr(s, "tc", None) is tc)]
        self.last_fetched = max(mine, key=lambda x: x.seq) if mine else None

    # -- one clock cycle ------------------------------------------------------------------

    def cycle(self) -> CycleTrace:
        p, st = self.p, self.stage
        self.cycles += 1
        t = self.cycles
        trace = CycleTrace(t, {})
        ev = trace.events
        stall: tuple[str, str] | None = None

        # WB: the instruction completes
        wb = st["WB"]
        if wb is not None:
            self.retired += 1
            m = self.p.vpes[wb.tc.vpe]
            if m.ejtag.sample_due:
                m.ejtag.sample(wb.pc)
            if wb.perf is not None:
                m.perf.retire(*wb.perf)
        st["WB"] = None

        # MS: an access still waiting for memory holds MS and everything behind
        ms = st["MS"]
        held = ms is not None and ms.mem_wait > 0
        if held:
            ms.mem_wait -= 1
            _pop_run(ms.mem_runs)
            stall = ("memory", f"{ms.text} waits in MS ({ms.mem_why}; "
                               f"{ms.mem_wait + 1} more cycle{'s' if ms.mem_wait else ''})")
            st["WB"], st["ER"] = st["ER"], None
        else:
            st["WB"], st["ER"], st["MS"], st["EX"] = st["ER"], st["MS"], st["EX"], None

        # ER: a pending exception is taken now (the TC's younger ones never ran)
        er = st["ER"]
        if er is not None and er.exc is not None:
            tc = er.tc
            self._on(tc)
            m = self.m
            before = p.before(tc)
            m.wake()
            enter(m, er)
            m.take_exception(er.exc, er.pc, er.word, er.dec, er.in_slot)
            p.after(tc, before)
            name = C.EXC_NAMES.get(er.exc.code, str(er.exc.code))
            ev.append(f"TC{tc.n}: exception {name} at 0x{er.pc:08x} ({er.text}) in ER: "
                      f"kill its younger instructions, fetch the handler at 0x{m.pc:08x}")
            er.timeline[t] = "ER"
            st["ER"] = None
            self._flush_after(er.seq - 1)
            self.last_fetched = None
            self._redirect(m.pc | m.isa_mode if m.compressed is not None else m.pc, t + 1)

        # the TCs that cannot go on: back to their instruction buffers
        for name in ("AG", "RF"):
            s = st[name]
            if s is not None and not p.eligible(s.tc):
                self._park(s.tc, t, "not runnable", ev)

        # AG -> EX
        x, y, z = st["EX"], st["MS"], st["ER"]
        frozen = (x is not None and x.exc is not None or y is not None and y.exc is not None
                  or z is not None and z.exc is not None)
        ag = st["AG"]
        if ag is not None and not held and st["EX"] is None and not frozen:
            self._on(ag.tc)
            wait = self._waits_for(ag, ag.t24.at_ex if ag.t24 else (), t)
            busy = self._unit_wait(ag, t) if wait is None else None
            if wait is not None:
                pr, r = wait
                kind = _kind(pr)
                late = kind == "load" and pr.ex_cycle is not None and \
                    pr.ready_at.get(r, 0) > pr.ex_cycle + 2
                if late and self.it:            # the skid buffer: let the others run
                    self.held_until[ag.tc.n] = pr.ready_at[r]
                    self._park(ag.tc, t, f"{_rname(r)} from {pr.text} (a load miss) is "
                                         "late", ev)
                    stall = ("miss-wait", f"{ag.text} waits for {_rname(r)}: replayed")
                else:
                    stall = ("miss-wait" if late else kind,
                             f"{ag.text} waits in AG for {_rname(r)} from {pr.text}")
            elif busy is not None:
                stall = busy
            else:
                st["EX"], st["AG"] = ag, None
                self._dispatch_ex(ag, t, trace, ev)
                if self.watch_hit is not None:
                    st["EX"], st["AG"] = None, ag
                    stall = ("watch", f"{ag.text} stopped by a debugger watchpoint")

        # RF -> AG
        rf = st["RF"]
        if rf is not None and st["AG"] is None and not held:
            self._on(rf.tc)
            wait = self._waits_for(rf, rf.t24.at_ag if rf.t24 else (), t)
            if wait is None:
                st["AG"], st["RF"] = rf, None
            elif stall is None:
                pr, r = wait
                kind = "alu-agen" if _kind(pr) == "alu" else _kind(pr)
                stall = (kind, f"{rf.text} waits in RF for the address base "
                               f"{_rname(r)} from {pr.text}")

        # IT -> RF: the dispatch scheduler (13.1)
        if st["RF"] is None and not held:
            ready = []
            for tc in p.tcs:
                f = self._front(tc)
                if f.queue and f.queue[0].avail <= t and self.held_until.get(tc.n, 0) <= t \
                        and p.eligible(tc):
                    ready.append(tc)
            p._begin_rotation(ready)
            tc = p.choose(ready)
            if tc is not None:
                self._on(tc)
                if self.m.waiting:              # its WAIT is over (an interrupt signal)
                    self.m.wake()
                st["RF"] = self.queue.pop(0)
                p.dispatched(tc)
                self.dispatched[tc.n] += 1
            p._rotate()

        # IF: the next bundle of one TC
        fetched = self._fetch_tc(t)

        # the views of this cycle
        for name in BACK:
            s = st[name]
            if s is not None:
                s.timeline[t] = name
        first: dict[str, Slot24] = {}            # the oldest in IS / IT (TC order)
        edge = 2 if self.it else 1
        for tc in p.tcs:
            for s in self._front(tc).queue:
                if t < s.avail - edge:
                    where = s.timeline.setdefault(t, "IF")
                elif t == s.avail - edge:
                    where = s.timeline[t] = "IS"
                else:
                    where = s.timeline[t] = "IT"
                if where not in first:
                    first[where] = s
        stages = {n: _sv(st[n]) for n in BACK}
        stages["IF"] = _sv(fetched[0]) if fetched else None
        for n in ("IS", "IT"):
            stages[n] = _sv(first.get(n))
        trace.stages = {n: stages.get(n) for n in STAGES16}
        if stall is None and st["RF"] is None and any(f.fetching for f in self.fronts.values()):
            stall = ("fetch", "no TC has an instruction for RF")
        if stall is not None:
            kind, why = stall
            trace.stall = f"{kind}: {why}"
            self.stalls[kind] = self.stalls.get(kind, 0) + 1
        ev.extend(self._fetch_notes)
        self._fetch_notes = []
        for m in p.vpes:
            m.advance(1)
        self.history.append(trace)
        return trace

    def _dispatch_ex(self, s: Slot24, t: int, trace: CycleTrace, ev: list[str]) -> None:
        """EX: the instruction runs on its TC (bound on its VPE); a blocked
        one (a YIELD, a gating storage access) goes back to the TC's buffer."""
        p, tc = self.p, s.tc
        self._on(tc)
        if self.m.waiting:                      # dispatched: its WAIT is over (WII)
            self.m.wake()
        before = p.before(tc)
        try:
            self._execute(s, t, trace, ev)
        except Blocked as b:
            self.m.blocked = b
            self.stage["EX"] = None
            s.timeline[t] = "<<"
            self.queue.insert(0, s)
            s.avail = t + 1
            ev.append(f"TC{tc.n}: {s.text} blocks ({b.kind}): back to its instruction buffer")
            self._park(tc, t, "blocked", ev)
        if s.exc is None:                       # (an exception: accounted in ER)
            p.after(tc, before)
        if s.exc is None and self.m.halted:
            for other in p.tcs:                  # the program ends: every TC drains
                f = self._front(other)
                f.fetching = False
                if other is not tc:
                    self._on(other)
                    self.fetching = False
                    self._drop_unexecuted(other)
                    self.queue = []
            self._on(tc)

    # -- running ----------------------------------------------------------------------------

    def empty(self) -> bool:
        return all(not self._front(tc).queue for tc in self.p.tcs) and \
            all(v is None for v in self.stage.values())

    def entering_ex(self) -> Slot24 | None:
        """The instruction in AG when it is the next to execute — not one of
        a TC that cannot go on (it goes back to its buffer next cycle: a TC
        that freed itself with YIELD may have fetched past the program)."""
        s = super().entering_ex()
        return s if s is None or self.p.eligible(s.tc) else None

    def next_slot(self) -> Slot24 | None:
        runnable = [tc for tc in self.p.tcs if self.p.eligible(tc)]
        for tcs in (runnable, self.p.tcs):
            for name in ("AG", "RF"):
                s = self.stage[name]
                if s is not None and s.tc in tcs:
                    return s
            for tc in tcs:
                q = self._front(tc).queue
                if q:
                    return q[0]
        return None

    def run(self, *, max_cycles: int = 100_000, until_pc: int | None = None) -> int:
        start = self.cycles
        while self.cycles - start < max_cycles:
            if any(m.halted for m in self.p.vpes) and self.empty():
                break
            nxt = self.entering_ex()
            if until_pc is not None and nxt is not None and nxt.pc == until_pc:
                break
            self.cycle()
        return self.cycles - start

    def drain(self, keep: set[int] | None = None, *, max_cycles: int = 1000) -> None:
        for tc in self.p.tcs:
            self._on(tc)
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
