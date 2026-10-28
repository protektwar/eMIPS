"""The MIPS P8700's own pipeline (LF14): out of order, quad issue, two
harts a core (simultaneous multithreading).

Sources: MD01502 rev. 1.84 — Figure 1.3 (the IFU; the IDU's 4-wide
decode, rename and dispatch; the IIU's 4-wide issue; the IEU's two ALU
pipes and one CTI pipe; the MDU; the LSU's load and store pipes; the
FPU's short and long pipes; the GRU's 4-wide graduation), 3.3.6 (load /
store bonding), 7.3.11 (mipsconfig7: the TAGE predictor, the L1 BTB, the
branch history, the jump register cache, the return prediction stack,
bonding, speculative issue after a load), ch. 10 (the harts: fetch of
four instructions of one hart a cycle, round robin; issue from both,
fair; graduation in order per hart; the shared buffers), Table 11.4 (the
events) and Table 12.1 (latencies, repeat rates, units).  What the manual
does not give comes from GCC's ``mips-p8700.md`` and LLVM's
``RISCVSchedMIPSP8700.td``: the issue
queues ALQ (the ALU pipe), AGQ (AL2, CTI, LSU) and FPQ (short, long) of
16 entries, a window of 96, issue width 4, a misprediction penalty of 8.
Where they differ from Table 12.1, the table wins.  The rest is assumed
and listed in ``sim.DEVIATIONS["P8700 pipeline"]`` (``LF.md`` LF14).

How it works.  Results are the hart's (:class:`emips.simrv.RVMachine`):
every instruction runs through its executors (:meth:`RVMachine.issue`)
once, in program order, **when it is dispatched** — traps, interrupts,
CSRs and the caches' contents follow program order.  Issue, execution,
completion and graduation are timing: an instruction issues from its
queue when its operands are ready (its producers' Table 12.1 latencies)
and its unit is free, completes after its latency, graduates in order.
The front end predicts (:class:`TAGE`, the BTB, the RPS, the JRC); after
a misprediction it goes on fetching the predicted path: those
instructions are dispatched as *ghosts* — they take window, queue and
unit slots, never execute, and are killed when the branch issues
(resolves).  A trap or a serializing instruction (``rvpipeline``'s list:
CSR writes, ``fence.i``, ``mret`` ...) stops fetching until it graduates.

:class:`P8700Pipeline` is one core (its harts); :class:`HartPipeline`
gives the Session one hart's view of it (``entering_ex``, ``next_slot``
...).  Every :meth:`P8700Pipeline.cycle` returns a
:class:`emips.pipeline.CycleTrace` with the stages IF (fetched), DR
(dispatched = executed), IS (issued), EX (executing), GR (graduated).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from .exceptions import FetchError, WatchpointHit
from .pipeline import CycleTrace, History, StageView
from .rvpipeline import _BRANCHES, _LOADS, _STORES, _sleep, analyze

STAGES = ("IF", "DR", "IS", "EX", "GR")

#: the core's sizes and front-end depths (LF.md LF14; ``latency=`` overrides)
P8700 = {
    "FETCH": 4,         # instructions fetched a cycle (one hart; ch. 10.1)
    "DISPATCH": 4,      # decoded / dispatched a cycle per hart (ch. 10.1)
    "ISSUE": 4,         # issued a cycle (Figure 1.3, LLVM IssueWidth)
    "GRADUATE": 4,      # graduated a cycle (Figure 1.3)
    "WINDOW": 96,       # instructions in flight (LLVM MicroOpBufferSize)
    "QUEUE": 16,        # each of ALQ / AGQ / FPQ (LLVM BufferSize)
    "IBUF": 16,         # the instruction buffer of a hart (assumed)
    "FRONT": 7,         # fetch to dispatch (penalty 8 = FRONT + the issue cycle)
    "DECODE": 4,        # fetch to decode (a decode-time redirect)
    "BTB": 64, "JRC": 32, "RPS": 16,
}

#: mipsconfig7 (Table 7.22)
DBP, DDBP, DGHR, DJRC, DRPS = 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 0
DSLD, DLBND, DSBND, RST, DL1B = 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 28

# -- Table 12.1 ---------------------------------------------------------------

#: AL2 only (the AGQ's second ALU)
_AL2 = {"auipc": 1, "mips.ccmov": 2, "clz": 2, "clzw": 2, "ctz": 2, "ctzw": 2, "cpop": 2,
        "cpopw": 2}
_MUL = {"mul": 4, "mulh": 4, "mulhsu": 4, "mulhu": 4, "mulw": 3}
_DIV = {"div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}
_GRU = {"ecall": 3, "ebreak": 3, "mret": 3, "sret": 3, "wfi": 3, "dret": 3,
        "mips.ehb": 1, "mips.ihb": 1}
#: the LSU's other operations (Table 12.1: n/a latency, rate 1)
_LSU_OTHER = {"fence", "fence.tso", "fence.i", "pause", "mips.pause", "sfence.vma",
              "mips.mtlbwr", "mips.pref", "prefetch.i", "prefetch.r", "prefetch.w",
              "cbo.clean", "cbo.flush", "cbo.inval", "cbo.zero"}
_FENCES = {"fence", "fence.tso", "fence.i", "sfence.vma", "pause", "mips.pause"}
_FP_SHORT = {"fclass": 1, "fsgnj": 1, "fsgnjn": 1, "fsgnjx": 1, "feq": 2, "flt": 2,
             "fle": 2, "fmin": 2, "fmax": 2, "fmv": 1}
_FP_LONG = {"fadd": 4, "fsub": 4, "fcvt": 4, "fmul": 5, "fmadd": 8, "fmsub": 8,
            "fnmadd": 8, "fnmsub": 8}
#: (minimum, maximum) of the data-dependent ones
_FP_DIV = {("fdiv", "s"): (11, 21), ("fdiv", "d"): (17, 37), ("fsqrt", "s"): (14, 22),
           ("fsqrt", "d"): (23, 24)}
_FP_MEM = {"flw": "LD", "fld": "LD", "fsw": "ST", "fsd": "ST"}


def unit_of(name: str) -> tuple[str | None, str, int]:
    """(queue, unit, latency) of an instruction by Table 12.1 (the
    expansion's name for a compressed one): queue "ALQ" / "AGQ" / "FPQ" /
    None (the IDU / GRU itself); unit ALU, AL2, CTI, LD, ST, MUL, DIV,
    FPS, FPL or GRU ("ALU" = either ALU pipe: ALQ's or AGQ's AL2); the
    latency of DIV / FDIV / FSQRT is their minimum (:func:`latency_of`)."""
    if name in _AL2:
        return "AGQ", "AL2", _AL2[name]
    if name in _BRANCHES or name in ("jal", "jalr"):
        return "AGQ", "CTI", 1
    if name in _MUL:
        return "ALQ", "MUL", _MUL[name]
    if name in _DIV:
        return "ALQ", "DIV", 7
    if name.startswith("csrr"):
        return "ALQ", "MUL", 4                  # (the MDU, rate 1)
    if name in _GRU:
        return None, "GRU", _GRU[name]
    if name in _FP_MEM:
        return "AGQ", _FP_MEM[name], 4 if _FP_MEM[name] == "LD" else 1
    if name in _LOADS or name.startswith("lr.") or name in ("mips.ldp", "mips.lwp"):
        return "AGQ", "LD", 4
    if name.startswith("sc."):
        return "AGQ", "ST", 4                   # (its rd: from the memory system)
    if name in _STORES or name in _LSU_OTHER:
        return "AGQ", "ST", 1
    if name.startswith("f"):
        op, fmt = (name.split(".") + ["s"])[:2]
        if (op, fmt) in _FP_DIV:
            return "FPQ", "FPL", _FP_DIV[op, fmt][0]
        if op in _FP_SHORT:
            return "FPQ", "FPS", _FP_SHORT[op]
        if op in _FP_LONG:
            return "FPQ", "FPL", _FP_LONG[op]
    return "ALQ", "ALU", 1


def _special(bits: int, width: int) -> bool:
    """A zero, infinite or NaN binary32 / binary64 value."""
    exp = bits >> (23 if width == 32 else 52) & (0xFF if width == 32 else 0x7FF)
    frac = bits & ((1 << (23 if width == 32 else 52)) - 1)
    return exp == (0xFF if width == 32 else 0x7FF) or (exp == 0 and frac == 0)


def latency_of(name: str, m, f: dict) -> int:
    """Table 12.1's latency of ``name`` with its operands as they are now
    (before it executes): DIV / REM 7 + one cycle per 4 significant bits
    of the dividend (7-22); FDIV / FSQRT the minimum for a zero, infinite
    or NaN operand, else the maximum (both assumed)."""
    q, unit, lat = unit_of(name)
    if unit == "DIV":
        v = m.regs[f.get("rs1", 0)] & m.amask
        if name.endswith("w"):
            v &= 0xFFFFFFFF
            v = (1 << 32) - v if v >> 31 and not name.endswith("uw") else v
        elif m.xlen == 64 and v >> 63 and not name.endswith("u"):
            v = (1 << 64) - v
        elif m.xlen == 32 and v >> 31 and not name.endswith("u"):
            v = (1 << 32) - v
        return min(22, 7 + max(0, v.bit_length() - 1) // 4)
    op, fmt = (name.split(".") + ["s"])[:2]
    if (op, fmt) in _FP_DIV:
        lo, hi = _FP_DIV[op, fmt]
        width = 32 if fmt == "s" else 64
        regs = [f[k] for k in ("rs1", "rs2") if k in f]
        vals = [m.fpr[r] & ((1 << width) - 1) for r in regs]
        return lo if any(_special(v, width) for v in vals) else hi
    return lat


# -- the predictors ------------------------------------------------------------

class TAGE:
    """A TAGE predictor (Seznec & Michaud 2006): a bimodal base of 2-bit
    counters and four tagged tables of 3-bit counters with 2-bit useful
    bits, indexed with geometric lengths of the global history (sizes:
    LF.md LF14, assumed — MD01502 names TAGE, 7.3.11, without them)."""

    BASE, ENTRIES, TAG = 4096, 1024, 9
    HIST = (5, 15, 44, 130)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.base = bytearray([1]) * self.BASE
        self.tables: list[list] = [[None] * self.ENTRIES for _ in self.HIST]
        self.updates = 0

    @staticmethod
    def _fold(h: int, length: int, bits: int) -> int:
        h &= (1 << length) - 1
        out = 0
        while h:
            out ^= h & ((1 << bits) - 1)
            h >>= bits
        return out

    def _keys(self, pc: int, h: int) -> tuple[list[int], list[int]]:
        idx, tags = [], []
        for t, length in enumerate(self.HIST):
            idx.append((pc >> 1 ^ pc >> 11 ^ self._fold(h, length, 10) ^ t * 0x155)
                       & (self.ENTRIES - 1))
            tags.append((pc >> 1 ^ self._fold(h, length, self.TAG)
                         ^ self._fold(h, length, self.TAG - 1) << 1) & ((1 << self.TAG) - 1))
        return idx, tags

    def predict(self, pc: int, h: int, history: bool = True) -> tuple[bool, tuple]:
        bi = pc >> 1 & (self.BASE - 1)
        base = self.base[bi] >= 2
        provider = alt = None
        idx, tags = self._keys(pc, h) if history else ([], [])
        if history:
            for t in range(len(self.HIST) - 1, -1, -1):
                e = self.tables[t][idx[t]]
                if e is not None and e[0] == tags[t]:
                    if provider is None:
                        provider = t
                    else:
                        alt = t
                        break
        if provider is None:
            pred = altpred = base
        else:
            pred = self.tables[provider][idx[provider]][1] >= 4
            altpred = self.tables[alt][idx[alt]][1] >= 4 if alt is not None else base
        return pred, (bi, idx, tags, provider, pred, altpred)

    def update(self, meta: tuple, taken: bool) -> None:
        bi, idx, tags, provider, pred, altpred = meta
        if provider is None:
            c = self.base[bi]
            self.base[bi] = min(3, c + 1) if taken else max(0, c - 1)
        else:
            e = self.tables[provider][idx[provider]]
            if e is not None and e[0] == tags[provider]:
                e[1] = min(7, e[1] + 1) if taken else max(0, e[1] - 1)
                if pred != altpred:
                    e[2] = min(3, e[2] + 1) if pred == taken else max(0, e[2] - 1)
        if pred != taken and idx:                # allocate in a longer history
            start = 0 if provider is None else provider + 1
            for t in range(start, len(self.HIST)):
                e = self.tables[t][idx[t]]
                if e is None or e[2] == 0:
                    self.tables[t][idx[t]] = [tags[t], 4 if taken else 3, 0]
                    break
            else:
                for t in range(start, len(self.HIST)):
                    e = self.tables[t][idx[t]]
                    if e is not None:
                        e[2] = max(0, e[2] - 1)
        self.updates += 1
        if self.updates % (1 << 18) == 0:        # (aging the useful bits)
            for table in self.tables:
                for e in table:
                    if e is not None:
                        e[2] >>= 1


def _link(r: int) -> bool:
    return r in (1, 5)


def ras_action(name: str, f: dict) -> str:
    """The return-address stack hint of a jump (unprivileged 2.5.1, Table
    3): "push", "pop", "pop-push" or ""."""
    rd, rs1 = f.get("rd", 0), f.get("rs1", 0)
    if name == "jal":
        return "push" if _link(rd) else ""
    if name != "jalr":
        return ""
    if not _link(rd):
        return "pop" if _link(rs1) else ""
    if not _link(rs1) or rs1 == rd:
        return "push"
    return "pop-push"


# -- an instruction in flight ------------------------------------------------------

class Op:
    """One instruction of a hart, from fetch to graduation (or its kill)."""

    __slots__ = ("seq", "hart", "pc", "insn", "size", "text", "info", "fetched", "decode_at",
                 "avail", "pred_next", "pred_how", "decoded", "tage", "hist", "rps_ck",
                 "fetch_exc", "fetch_error", "fetch_cache", "fetch_tlb", "fetch_wait",
                 "dispatched", "ghost", "queue", "unit", "lat", "srcs", "issued", "done",
                 "wake", "nospec", "replays", "mispredict", "actual", "trap", "serial",
                 "view", "killed", "graduated", "mem", "bond", "bonded", "bond_bad",
                 "mem_wait", "not_before", "timeline", "taken", "executed", "word",
                 "held", "busy", "perf")

    def __init__(self, seq: int, hart: int, pc: int) -> None:
        self.seq, self.hart, self.pc = seq, hart, pc
        self.insn = self.info = self.tage = self.rps_ck = None
        self.fetch_exc = self.fetch_error = None
        self.size, self.text, self.word = 4, "", None
        self.fetched = self.decode_at = self.avail = 0
        self.pred_next: int | None = None
        self.pred_how = ""
        self.decoded = False
        self.hist = 0
        self.fetch_cache: list = []
        self.fetch_tlb: list = []
        self.fetch_wait = 0
        self.dispatched = self.issued = self.done = self.wake = self.graduated = None
        self.ghost = self.nospec = self.mispredict = self.trap = self.serial = False
        self.killed = self.bond_bad = self.taken = self.executed = self.held = False
        self.queue: str | None = None
        self.unit, self.lat, self.busy = "", 1, 0
        self.srcs: list = []
        self.replays = 0
        self.actual: int | None = None
        self.view = None
        self.mem: list = []
        self.bond: Op | None = None             # (the second of a pair: its first)
        self.bonded: Op | None = None           # (the first of a pair: its second)
        self.mem_wait = 0
        self.perf = None
        self.not_before = 0
        self.timeline: dict[int, str] = {}

    def __repr__(self) -> str:
        return f"Op({self.seq}, h{self.hart}, 0x{self.pc:x}, {self.text!r})"

    @property
    def cti(self) -> bool:
        return self.info is not None and bool(self.info.control)


class _Hart:
    """One hart's front end and window."""

    def __init__(self, k: int, m) -> None:
        self.k, self.m = k, m
        self.fetch_pc = m.pc
        self.fetch_ready = 0                    # first cycle it may fetch again
        self.ibuf: deque[Op] = deque()
        self.rob: deque[Op] = deque()
        self.rename: dict = {}
        self.ghist = 0
        self.rps = [0] * P8700["RPS"]
        self.rps_top = 0
        self.hold: Op | None = None             # fetch waits until it graduates
        self.stop: Op | None = None             # fetch waits until it resolves
        self.wrong: Op | None = None            # mispredicted: ghosts follow it
        self.parked = False                     # Debug Mode
        self.done = False                       # exited
        self.pass_pc: int | None = None         # the debugger's stop, passed once
        self.cfg7 = 0
        self.devices: dict[int, bool] = {}      # page -> a device fetch
        self.was_blocked = self.was_fetch_stalled = False
        self.refilling = False
        self.last_view = None
        self.retired = 0


class P8700Pipeline:
    """One P8700 core: its harts (``machines``: :class:`RVMachine` s on the
    core's caches) share the front end's predictors, the queues, the
    units and the graduation width.  ``latency`` overrides :data:`P8700`;
    ``memory`` / ``nonblocking`` the memory timing (with caches)."""

    STAGES = STAGES
    FETCH_STAGE, MEM_STAGE = "IF", "DR"
    name = "p8700"
    watch_hit: WatchpointHit | None = None
    #: ``debug_stop(pc, hart)``: does the debugger stop there?  (dispatch
    #: stops before it: everything older executed, it not)
    debug_stop: Callable[[int, int], bool] | None = None
    #: is hart k running (the cluster's VP_RUNNING)?
    runs: Callable[[int], bool] | None = None

    def __init__(self, machines: list, *, latency: dict | None = None, trace: bool = True,
                 memory: dict | None = None, nonblocking: bool = False) -> None:
        self.cfg = dict(P8700, **{k: v for k, v in (latency or {}).items() if k in P8700})
        self.trace = trace
        self.harts = [_Hart(k, m) for k, m in enumerate(machines)]
        m0 = machines[0]
        self.timing = m0.caches.timing if m0.caches is not None else None
        if self.timing is not None:
            self.timing.configure(memory, nonblocking=nonblocking)
        self.memory = memory
        for h in self.harts:
            self._attach(h)
        self.tage = TAGE()
        self.btb: dict[int, tuple] = {}         # index -> (pc, target)
        self.jrc: dict[int, tuple] = {}
        self.queues: dict[str, list[Op]] = {"ALQ": [], "AGQ": [], "FPQ": []}
        self.busy: dict[str, int] = {}          # unpipelined unit -> free from cycle
        self.cycles = 0
        self.retired = 0
        self.stalls: dict[str, int] = {}
        #: (the Session reads ``stage["IF"]`` before the first cycle)
        self.stage: dict[str, Any] = {s: None for s in STAGES}
        self.history = History()
        self.slots = History()
        self._seq = 0
        self._fetch_turn = 0
        self._prio = 0
        self._steer = 0
        self.fetching = True
        self.stats = {"branches": 0, "mispredicted": 0, "replays": 0, "bonds": 0,
                      "ghosts": 0, "killed": 0}
        self._views: list = []                  # (hart, seq, view) of this cycle
        from .sessionrv import _disassembler
        self._dis = _disassembler(m0.xlen, m0.extensions, False)

    def _attach(self, h: _Hart) -> None:
        m = h.m
        m.tracing = self.trace
        p = m.p8700
        if p is not None:
            p.counters.pipeline = True          # (events from issue / graduation)
            h.cfg7 = p.config7
        h.utlb = m.mmu if hasattr(m.mmu, "take") else None
        if h.utlb is not None:
            from .timing import MEMORY_TIMING
            h.utlb.timing = True
            h.utlb.utlb_cost = (self.memory or {}).get("UTLB", MEMORY_TIMING["UTLB"])
            h.utlb.take()

    def attach(self, k: int, m) -> None:
        """Hart ``k`` restarted (the cluster's VP_RUN / a core reset): a
        fresh hart in its place, nothing of the old one in flight."""
        old = self.harts[k]
        self._kill(old, lambda op: True)
        for op in old.rob:
            op.killed = True
        h = self.harts[k] = _Hart(k, m)
        self._attach(h)

    # -- small helpers ---------------------------------------------------------------

    def _ev(self, ev: list[str], h: _Hart, text: str) -> None:
        ev.append(f"hart {h.k}: {text}" if len(self.harts) > 1 else text)

    def _stall(self, kind: str) -> None:
        self.stalls[kind] = self.stalls.get(kind, 0) + 1

    def _count(self, h: _Hart, event: int, n: int = 1) -> None:
        p = h.m.p8700
        if p is not None and p.counters.armed and n:
            p.counters.add(event, n)

    def _running(self, h: _Hart) -> bool:
        return self.runs is None or self.runs(h.k)

    # -- fetch ---------------------------------------------------------------------

    def _device(self, h: _Hart, pc: int) -> bool:
        """Does a fetch at ``pc`` read a device (not done speculatively)?"""
        m = h.m
        if not m.devices:
            return False
        page = pc >> 12
        hit = h.devices.get(page)
        if hit is None:
            try:
                pa = m.translate(pc).pa
            except Exception:                    # (no mapping: the fetch faults)
                pa = None
            hit = h.devices[page] = pa is not None and any(d.covers(pa) for d in m.devices)
        return hit

    def _fetch(self, ev: list[str], trace: CycleTrace) -> None:
        """One hart fetches up to FETCH instructions (round robin)."""
        n = len(self.harts)
        cyc = self.cycles
        for j in range(n):
            h = self.harts[(self._fetch_turn + j) % n]
            if h.done or h.parked or h.hold is not None or h.stop is not None \
                    or h.fetch_ready > cyc or not self._running(h) \
                    or len(h.ibuf) + self.cfg["FETCH"] > self.cfg["IBUF"]:
                continue
            self._fetch_turn = (h.k + 1) % n
            got = self._fetch_block(h, ev)
            if got:
                trace.stages["IF"] = self._view(got[0])
            return

    def _fetch_block(self, h: _Hart, ev: list[str]) -> list[Op]:
        from .simrv import RVTrap, CAUSE_NAMES
        m, cyc, cfg = h.m, self.cycles, self.cfg
        if m.caches is not None:
            m.caches.m = m
        out: list[Op] = []
        line = h.fetch_pc >> 6
        speculative = bool(h.rob or h.ibuf or h.wrong)
        while len(out) < cfg["FETCH"]:
            pc = h.fetch_pc
            if pc >> 6 != line:
                break
            if self._device(h, pc) and (speculative or out):
                break                            # (not before it is the next to run)
            self._seq += 1
            op = Op(self._seq, h.k, pc)
            op.fetched = cyc
            m.fetch_accesses = []
            m.fetch_tlb = []
            try:
                i = m.fetch(pc, triggers=False)
            except RVTrap as t:
                op.fetch_exc = t
                op.text = f"(fetch: {CAUSE_NAMES.get(t.cause, t.cause)})"
            except FetchError as exc:
                op.fetch_error = exc
                op.text = "(no code here)"
            op.fetch_cache = m.fetch_accesses
            op.fetch_tlb = list(m.fetch_tlb)
            if op.fetch_exc is not None or op.fetch_error is not None:
                op.pred_next = None
                out.append(op)
                h.stop = op                      # (nothing to predict after it)
                break
            op.insn, op.size, op.word = i, i.size, i.raw
            op.text = self._dis.text(i.raw, i.size, pc)
            op.info = analyze(i, _NO_LATENCY, m.vector)
            out.append(op)
            nxt = self._predict(h, op)
            if nxt is None:                      # an indirect jump nobody predicts
                h.stop = op
                break
            h.fetch_pc = nxt & m.amask
            if nxt != (pc + i.size) & m.amask:
                break                            # (a taken CTI ends the block)
        wait = 0
        notes: list[str] = []
        if self.timing is not None and out:
            wait, _, notes = self.timing.take()
        if h.utlb is not None and out:
            more, unotes = h.utlb.take()
            wait += more
            notes = unotes + notes
        for n_ in notes:
            self._ev(ev, h, f"fetch 0x{out[0].pc:08x}: {n_}")
        for op in out:
            op.fetch_wait = wait
            op.decode_at = cyc + cfg["DECODE"] + wait
            op.avail = cyc + cfg["FRONT"] + wait
            for c in range(cyc, cyc + 1 + wait):  # (IF: the fetch and its miss)
                op.timeline[c] = "IF"
            h.ibuf.append(op)
            self.slots.append(op)
        if wait:
            self._count(h, 20)
            self._count(h, 138, wait)
        h.fetch_ready = cyc + 1 + wait
        return out

    def _predict(self, h: _Hart, op: Op) -> int | None:
        """The next fetch address after ``op`` (None: an indirect jump with
        no prediction — fetch stops); predicts at fetch with a BTB hit,
        else leaves a decode-time redirect (:meth:`_decode`)."""
        i, pc, m = op.insn, op.pc, h.m
        seq = (pc + i.size) & m.amask
        name = i.base.name
        op.hist, op.rps_ck = h.ghist, (h.rps_top, h.rps[h.rps_top])
        if not op.cti:
            return seq
        cfg7 = h.cfg7
        if cfg7 & DBP:                           # no prediction: sequential
            op.pred_how = "none"
            op.pred_next = seq
            return seq
        f = i.f
        btb = None if cfg7 & DL1B else self.btb.get(pc >> 1 & (self.cfg["BTB"] - 1))
        hit = btb is not None and btb[0] == pc
        target: int | None
        if name in _BRANCHES:
            if cfg7 & DDBP:                      # static: backward taken
                taken = f.get("imm", 0) < 0
                op.tage = None
            else:
                taken, op.tage = self.tage.predict(pc, h.ghist, not cfg7 & DGHR)
            h.ghist = (h.ghist << 1 | taken) & ((1 << 130) - 1)
            target = (pc + f.get("imm", 0)) & m.amask if taken else seq
            op.pred_how = "tage" if not cfg7 & DDBP else "static"
        elif name == "jal":
            target = (pc + f.get("imm", 0)) & m.amask
            op.pred_how = "jal"
        else:                                    # jalr: the RPS or the JRC
            act = ras_action(name, f)
            target = None
            if act.startswith("pop") and not cfg7 & DRPS:
                target = h.rps[h.rps_top]
                h.rps_top = (h.rps_top - 1) % len(h.rps)
                op.pred_how = "rps"
            elif not cfg7 & DJRC:
                e = self.jrc.get(pc >> 1 & (self.cfg["JRC"] - 1))
                if e is not None and e[0] == pc:
                    target = e[1]
                    op.pred_how = "jrc"
        act = ras_action(name, f)
        if act.endswith("push"):
            h.rps_top = (h.rps_top + 1) % len(h.rps)
            h.rps[h.rps_top] = seq
        if target is None:
            op.pred_next = None
            return None
        target &= m.amask
        if target != seq and not hit:            # found at decode, not at fetch
            op.pred_next = seq
            op.decoded = False
            op.pred_how += ":decode"
            op.actual = target                   # (where decode sends fetch)
            return seq
        op.pred_next = target
        op.decoded = True
        if op.pred_how == "rps":
            self._count(h, 141)
        elif target != seq:
            self._count(h, 149)
        return target

    def _decode(self, h: _Hart, ev: list[str]) -> None:
        """Decode-time redirects: a CTI the BTB missed whose prediction is
        taken (MD01502 Table 11.4: jal_b_hit_l1btb_miss, rps_hit_l1btb_miss)."""
        cyc = self.cycles
        for idx, op in enumerate(h.ibuf):
            if op.decode_at > cyc:
                break
            if op.decoded or op.insn is None or not op.cti or not op.pred_how.endswith(":decode"):
                op.decoded = True
                continue
            op.decoded = True
            target = op.actual
            op.actual = None
            op.pred_next = target
            self._kill(h, lambda o, s=op.seq: o.seq > s, ibuf_only=True)
            self._restore(h, op, True)
            h.stop = None
            h.fetch_pc = target
            h.fetch_ready = cyc + 1
            self._count(h, 143 if op.pred_how.startswith("rps") else 151)
            self._ev(ev, h, f"decode: {op.text} predicted taken, the BTB missed: fetch "
                            f"0x{target:08x}")
            break

    # -- dispatch (= execution) --------------------------------------------------------

    def _boundary(self, h: _Hart, op: Op, ev: list[str]) -> str | None:
        """Before ``op`` executes (the hart's step does the same): a halt
        request, an L2 error, an interrupt — "flush" —, a WFI / WRS or
        the trace encoder — "wait"."""
        from .simrv import CAUSE_HALTREQ, INTERRUPT_NAMES
        m = h.m
        if m.halt_request and not m.debug_mode:
            m.halt_request = False
            m.pc = op.pc
            m.enter_debug(CAUSE_HALTREQ)
            self._ev(ev, h, f"halt request before {op.text}: Debug Mode")
            return "flush"
        if m.caches is not None and m.caches.pending is not None \
                and m.take_cache_error(op.pc):
            self._ev(ev, h, f"an L2 error before {op.text}: Cache Error, fetch the handler "
                            f"at 0x{m.pc:08x}")
            return "flush"
        req = None if m._stepping or m._step_next else m.interrupt_request()
        if req is not None:
            m.take_interrupt(req, op.pc)
            self._ev(ev, h, f"interrupt {INTERRUPT_NAMES.get(req[0], req[0])} before "
                            f"{op.text}: fetch the handler at 0x{m.pc:08x}")
            return "flush"
        if m.trace_stall and not m.trace_encoder.drain():
            return "wait"
        if m.waiting:
            if m.still_waiting():
                return "wait"
            self._ev(ev, h, "WFI: an interrupt is pending, wake up")
        return None

    def _queue_for(self, op: Op) -> tuple[str | None, str]:
        q, unit, _ = unit_of(op.insn.base.name) if op.insn is not None else ("ALQ", "ALU", 1)
        if unit == "ALU":                       # either ALU pipe: in turn (assumed)
            self._steer ^= 1
            if self._steer and len(self.queues["AGQ"]) < self.cfg["QUEUE"]:
                return "AGQ", "AL2"
            return "ALQ", "ALU"
        return q, unit

    def _window(self, h: _Hart) -> int:
        return self.cfg["WINDOW"] // max(1, len(self.harts))

    def _dispatch(self, h: _Hart, ev: list[str], trace: CycleTrace) -> str | None:
        """Up to DISPATCH instructions of hart ``h``; what stopped it."""
        cyc, cfg = self.cycles, self.cfg
        n = 0
        why = None
        prev: Op | None = None
        while n < cfg["DISPATCH"]:
            if not h.ibuf:
                why = "empty"
                break
            op = h.ibuf[0]
            if op.avail > cyc:
                why = "front end"
                break
            if len(h.rob) >= self._window(h):
                why = "window full"
                break
            if op.insn is None and h.wrong is not None:
                why = "wrong path"                # (a ghost's fetch fault: wait)
                break
            q, unit = self._queue_for(op) if op.insn is not None else (None, "GRU")
            if q is not None and len(self.queues[q]) >= cfg["QUEUE"]:
                why = f"{q} full"
                break
            ghost = h.wrong is not None
            if not ghost:
                if op.insn is not None and op.info.serialize and h.rob:
                    why = "serialize"             # (it waits to be the oldest)
                    break
                if self.debug_stop is not None and h.pass_pc != op.pc \
                        and self.debug_stop(op.pc, h.k):
                    op.held = True
                    h.pass_pc = op.pc            # (it goes on once the debugger stopped)
                    why = "debugger"
                    break
                if h.pass_pc == op.pc:
                    h.pass_pc = None
                b = self._boundary(h, op, ev)
                if b == "wait":
                    why = "wait"
                    break
                if b == "flush":
                    self._refetch(h, h.m.pc, ev)
                    why = "interrupt"
                    break
            elif op.insn is not None and op.info.serialize:
                why = "wrong path"
                break
            h.ibuf.popleft()
            op.queue, op.unit = q, unit
            op.dispatched = cyc
            op.ghost = ghost
            op.timeline[cyc] = "DR"
            if trace.stages["DR"] is None:
                trace.stages["DR"] = self._view(op)
            reads = (op.info.reads_ex | op.info.reads_id) if op.info else frozenset()
            op.srcs = [h.rename[r] for r in sorted(reads) if r in h.rename]
            if ghost:
                self.stats["ghosts"] += 1
                op.lat = unit_of(op.insn.base.name)[2]
            else:
                self._execute(h, op, ev, trace)
            if op.info is not None:
                for r in op.info.writes:
                    h.rename[r] = op
            h.rob.append(op)
            if q is not None:
                self.queues[q].append(op)
            else:
                op.done = cyc + op.lat
            if not ghost and op.insn is not None and prev is not None \
                    and not prev.ghost:
                self._bond(h, prev, op)
            prev = op
            n += 1
            if op.trap or op.serial or h.parked or h.done:
                break
        if n:
            h.refilling = False
        return why if n == 0 else None

    def _execute(self, h: _Hart, op: Op, ev: list[str], trace: CycleTrace) -> None:
        """The instruction runs through the hart (program order)."""
        m = h.m
        if m.caches is not None:
            m.caches.m = m
        if op.fetch_error is not None:
            raise op.fetch_error
        i = op.insn
        op.lat = latency_of(i.base.name, m, i.f) if i is not None else 1
        logged = len(m.exception_log)
        m.last_view = None
        m.pc = op.pc
        retrigger = m.tm.fetch
        try:
            m.issue(op.pc, None if retrigger or i is None else i, op.fetch_exc, clock=False)
        finally:
            view = m.last_view
            if view is not None:
                view.fetch = op.fetch_cache
                op.view = view
                self._views.append((h.k, op.seq, view))
                trace.ex_view = view
                h.last_view = view
        op.executed = True
        p = m.p8700
        if p is not None and p.counters.last is not None:
            op.perf, p.counters.last = p.counters.last, None
        seq = (op.pc + op.size) & m.amask
        halted = m.debug_mode and m.hybrid is None     # (the P8700's Debug Mode runs: LF16)
        if len(m.exception_log) != logged or halted:
            e = m.exception_log[-1] if m.exception_log else {}
            what = e.get("error") or e.get("name") or "a trap"
            op.trap = True
            op.lat = max(op.lat, 1)
            if halted:
                self._ev(ev, h, f"{op.text}: Debug Mode — the hart stops")
                self._park(h)
            else:
                self._ev(ev, h, f"{op.text}: {what} — fetch the handler at 0x{m.pc:08x} "
                                "when it graduates")
                self._hold(h, op)
            return
        if m.halted:
            self._ev(ev, h, f"{op.text}: program exit")
            h.done = True
            self._kill(h, lambda o: True, ibuf_only=True)
            return
        if i is not None and view is not None:
            op.mem = [(a, s) for a, s, *_ in getattr(view, "mem_reads", [])] + \
                [(a, s) for a, s, *_ in getattr(view, "mem_writes", [])]
        self._memory_time(h, op, ev)
        op.actual = m.pc
        if op.cti:
            op.taken = m.pc != seq
            if op.pred_next != m.pc:
                op.mispredict = True
                if op.pred_next is not None:
                    h.wrong = op                 # (ghosts follow until it resolves)
                    self._ev(ev, h, f"{op.text}: mispredicted (fetched 0x{op.pred_next:08x}, "
                                    f"goes to 0x{m.pc:08x}) — it resolves when it issues")
                else:
                    h.stop = op
        elif op.info is not None and op.info.serialize:
            op.serial = True
            self._ev(ev, h, f"{op.text}: {op.info.serialize} serializes — fetch again from "
                            f"0x{m.pc:08x} when it graduates")
            self._hold(h, op)
        elif m.pc != seq:                        # (a flow change nobody predicted)
            op.serial = True
            self._hold(h, op)

    def _memory_time(self, h: _Hart, op: Op, ev: list[str]) -> None:
        more, unotes = h.utlb.take() if h.utlb is not None else (0, [])
        for n_ in unotes:
            self._ev(ev, h, f"{op.text}: {n_}")
        wait = 0
        if self.timing is not None:
            wait, _, notes = self.timing.take()
            for n_ in notes:
                self._ev(ev, h, f"{op.text}: {n_}")
        op.mem_wait = wait + more

    def _bond(self, h: _Hart, a: Op, b: Op) -> None:
        """MD01502 3.3.6: two adjacent identical word / doubleword loads or
        stores, the same base, offsets +4 / +8, (loads) the first not
        writing the base, a cacheable first access."""
        na, nb = a.insn.base.name, b.insn.base.name
        if na != nb or na not in ("lw", "lwu", "ld", "sw", "sd") or a.bond or a.bonded:
            return
        load = na[0] == "l"
        if h.cfg7 & (DLBND if load else DSBND):
            return
        fa, fb = a.insn.f, b.insn.f
        size = 8 if na[1] == "d" else 4
        if fa.get("rs1") != fb.get("rs1") or fb.get("imm", 0) != fa.get("imm", 0) + size:
            return
        if load and fa.get("rd") == fa.get("rs1"):
            return
        if not a.mem or not b.mem or a.trap or b.trap:
            return
        from .rvmmu import pma_cca
        from .simrv import CCA_UNCACHED
        try:
            pa = h.m.translate(a.mem[0][0]).pa
        except Exception:
            pa = None
        if pa is None or pma_cca(h.m, pa) == CCA_UNCACHED:
            return
        a.bonded, b.bond = b, a
        block = 2 * size
        b.bond_bad = a.mem[0][0] // block != (b.mem[0][0] + size - 1) // block
        self.stats["bonds"] += 1

    def _hold(self, h: _Hart, op: Op) -> None:
        h.hold = op
        self._kill(h, lambda o: True, ibuf_only=True)

    def _park(self, h: _Hart) -> None:
        h.parked = True
        self._kill(h, lambda o: True, ibuf_only=True)

    def _refetch(self, h: _Hart, pc: int, ev: list[str]) -> None:
        """Fetch again from ``pc`` (an interrupt / a Debug Mode entry
        before the next instruction): the instruction buffer goes."""
        self._kill(h, lambda o: True, ibuf_only=True)
        h.stop = h.wrong = None
        if h.m.debug_mode and h.m.hybrid is None:
            h.parked = True
            return
        h.fetch_pc = pc & h.m.amask
        h.fetch_ready = self.cycles + 1
        h.devices.clear()
        h.refilling = True

    def refetch(self, k: int, pc: int) -> None:
        """A debugger moved hart ``k``'s pc: what it fetched and has not
        executed goes, fetching starts at ``pc``."""
        h = self.harts[k]
        if h.wrong is not None:
            self._kill(h, lambda o, s=h.wrong.seq: o.seq > s)
        self._refetch(h, pc, [])
        h.hold = None
        h.done = h.m.halted

    # -- issue / execute / complete ------------------------------------------------------

    def _ready(self, op: Op, cyc: int) -> bool:
        if op.not_before > cyc:
            return False
        dsld = self.harts[op.hart].cfg7 & DSLD
        for s in op.srcs:
            if s.killed or s.graduated is not None:
                continue
            if s.issued is None:
                return False
            t = s.done if (op.nospec or dsld or s.wake is None) else s.wake
            if t is None or t > cyc:
                return False
        h = self.harts[op.hart]
        name = op.insn.base.name if op.insn is not None else ""
        if op.unit == "LD" and op.mem and not op.ghost:     # (an older overlapping store)
            for o in h.rob:
                if o is op:
                    break
                if o.unit == "ST" and o.issued is None and o.mem and _overlap(o.mem, op.mem):
                    return False
        if name in _FENCES:                      # (older memory accesses complete)
            for o in h.rob:
                if o is op:
                    break
                if o.unit in ("LD", "ST") and (o.done is None or o.done > cyc):
                    op.busy += 1
                    return False
        return True

    def _issue(self, ev: list[str], trace: CycleTrace) -> list[Op]:
        cyc, cfg = self.cycles, self.cfg
        cands = [op for q in self.queues.values() for op in q
                 if op.issued is None and op.bond is None]
        prio = self._prio
        cands.sort(key=lambda o: (o.hart != prio, o.seq))
        used: set[str] = set()
        issued: list[Op] = []
        alq_mdu = False
        for op in cands:
            if len(issued) >= cfg["ISSUE"]:
                break
            unit = op.unit
            if unit in used or self.busy.get(unit, 0) > cyc:
                continue
            if unit in ("MUL", "DIV") and alq_mdu:
                continue
            if not self._ready(op, cyc):
                continue
            partner = op.bonded                  # (a bonded pair: one request)
            if partner is not None and partner.killed:
                op.bonded = partner.bond = None
                partner = None
            if partner is not None and not self._ready(partner, cyc):
                continue
            # a consumer that woke on a missing load: replayed
            if not self.harts[op.hart].cfg7 & DSLD and not op.nospec and any(
                    s.wake is not None and s.done is not None and s.wake <= cyc < s.done
                    and not s.killed and s.graduated is None for s in op.srcs):
                op.nospec = True
                op.replays += 1
                self.stats["replays"] += 1
                used.add(unit)
                op.timeline[cyc] = "rp"
                self._count(self.harts[op.hart], 23)
                self._ev(ev, self.harts[op.hart], f"{op.text}: replayed (its load missed)")
                issued.append(op)
                op.issued = None
                continue
            used.add(unit)
            if unit in ("MUL", "DIV"):
                alq_mdu = True
            self._start(op, cyc)
            issued.append(op)
            if partner is not None:
                self._start(partner, cyc)
                if op.bond_bad or partner.bond_bad:
                    h = self.harts[op.hart]
                    self._ev(ev, h, f"{op.text} + {partner.text}: the bond is not within "
                                    "its aligned block — the second is fetched again")
                    self._count(h, 38)
                    late = cyc + cfg["FRONT"] + 1
                    for o in h.rob:
                        if o.seq > op.seq:
                            o.not_before = max(o.not_before, late)
                            if o is partner:
                                o.issued = None
                                o.done = None
                                o.bond = None
                                op.bonded = None
            if trace.stages["IS"] is None:
                trace.stages["IS"] = self._view(op)
        for q in self.queues.values():
            q[:] = [o for o in q if o.issued is None and not o.killed]
        return [o for o in issued if o.issued is not None]

    def _start(self, op: Op, cyc: int) -> None:
        op.issued = cyc
        op.timeline[cyc] = "IS"
        lat = op.lat
        if op.unit == "LD":
            op.wake = cyc + lat
            op.done = cyc + lat + (op.mem_wait if not op.ghost else 0)
        elif op.unit == "ST" and op.lat == 1:
            op.done = cyc + 1 + (op.mem_wait if not op.ghost else 0)
        else:
            op.done = cyc + lat
        if op.unit == "DIV" or (op.unit == "FPL" and op.insn is not None
                                and op.insn.base.name.split(".")[0] in ("fdiv", "fsqrt")):
            self.busy[op.unit] = cyc + lat       # (not pipelined)

    def _resolve(self, issued: list[Op], ev: list[str]) -> None:
        """The CTIs issued this cycle resolve: a misprediction kills the
        younger instructions of its hart and redirects fetch."""
        for op in issued:
            if op.ghost or not op.mispredict or op.killed:
                continue
            h = self.harts[op.hart]
            self._kill(h, lambda o, s=op.seq: o.seq > s)
            self._restore(h, op, op.taken)
            h.wrong = None
            if h.stop is op or h.stop is not None and h.stop.seq > op.seq:
                h.stop = None
            h.fetch_pc = op.actual
            h.fetch_ready = self.cycles + 1
            h.refilling = True
            self._count(h, 32)
            self._count(h, 34)
            self._ev(ev, h, f"{op.text}: resolves — mispredicted, fetch 0x{op.actual:08x} "
                            f"(the younger instructions are killed)")

    def _restore(self, h: _Hart, op: Op, taken: bool) -> None:
        """The speculative history and RPS as they were after ``op`` (its
        actual outcome): what younger predictions did is undone."""
        h.ghist = op.hist
        if op.insn.base.name in _BRANCHES:
            h.ghist = (op.hist << 1 | taken) & ((1 << 130) - 1)
        h.rps_top = op.rps_ck[0]
        h.rps[h.rps_top] = op.rps_ck[1]
        act = ras_action(op.insn.base.name, op.insn.f)
        if act.startswith("pop"):
            h.rps_top = (h.rps_top - 1) % len(h.rps)
        if act.endswith("push"):
            h.rps_top = (h.rps_top + 1) % len(h.rps)
            h.rps[h.rps_top] = (op.pc + op.size) & h.m.amask

    def _kill(self, h: _Hart, which: Callable[[Op], bool], ibuf_only: bool = False) -> None:
        cyc = self.cycles
        n = 0
        for op in h.ibuf:
            if which(op):
                op.killed = True
                op.timeline[cyc] = "xx"
                n += 1
        h.ibuf = deque(o for o in h.ibuf if not o.killed)
        if not ibuf_only:
            for op in h.rob:
                if which(op):
                    op.killed = True
                    op.timeline[cyc] = "xx"
                    n += 1
            h.rob = deque(o for o in h.rob if not o.killed)
            for q in self.queues.values():
                q[:] = [o for o in q if not o.killed]
            h.rename = {}
            for o in h.rob:
                if o.info is not None:
                    for r in o.info.writes:
                        h.rename[r] = o
        if h.stop is not None and h.stop.killed:
            h.stop = None
        self.stats["killed"] += n
        if n:
            self._count(h, 147, n)

    # -- graduation -----------------------------------------------------------------------

    def _graduate(self, ev: list[str], trace: CycleTrace) -> dict[int, int]:
        cyc, cfg = self.cycles, self.cfg
        out: dict[int, int] = {h.k: 0 for h in self.harts}
        order = self.harts[self._prio:] + self.harts[:self._prio]
        total = 0
        progress = True
        while total < cfg["GRADUATE"] and progress:
            progress = False
            for h in order:
                if total >= cfg["GRADUATE"] or not h.rob:
                    continue
                op = h.rob[0]
                if op.ghost or op.done is None or op.done > cyc:
                    continue
                h.rob.popleft()
                op.graduated = cyc
                op.timeline[cyc] = "GR"
                total += 1
                out[h.k] += 1
                progress = True
                self._graduated(h, op, ev)
                if trace.stages["GR"] is None:
                    trace.stages["GR"] = self._view(op)
        return out

    def _graduated(self, h: _Hart, op: Op, ev: list[str]) -> None:
        self.retired += 1
        h.retired += 1
        if op.info is not None:
            for r in op.info.writes:
                if h.rename.get(r) is op:
                    del h.rename[r]
        p = h.m.p8700
        if p is not None and p.counters.armed:
            p.counters.graduate(op.perf)
            if op.mispredict and op.insn is not None:
                name = op.insn.base.name
                if name in _BRANCHES:
                    p.counters.add(25)
                elif name == "jalr":
                    p.counters.add(29)
                    if ras_action(name, op.insn.f).startswith("pop"):
                        p.counters.add(27)
                        if op.pred_how == "rps":
                            p.counters.add(142)
                    elif op.pred_how.startswith("jrc"):
                        p.counters.add(150)
            if op.bond is not None:
                p.counters.add(67 if op.insn.base.name[0] == "l" else 68)
        if op.insn is not None and op.cti and not op.trap:
            self._train(h, op)
        if h.hold is op:
            h.hold = None
            h.fetch_pc = h.m.pc
            h.fetch_ready = self.cycles + 1
            h.devices.clear()
            h.refilling = True
            h.ghist = op.hist
        if h.stop is op:
            h.stop = None

    def _train(self, h: _Hart, op: Op) -> None:
        """The predictors learn from a graduated CTI (MD01502 7.3.11)."""
        name = op.insn.base.name
        self.stats["branches"] += 1
        if op.mispredict:
            self.stats["mispredicted"] += 1
        if name in _BRANCHES and op.tage is not None:
            self.tage.update(op.tage, op.taken)
        if op.taken and not h.cfg7 & DL1B:
            self.btb[op.pc >> 1 & (self.cfg["BTB"] - 1)] = (op.pc, op.actual)
        if name == "jalr":
            self.jrc[op.pc >> 1 & (self.cfg["JRC"] - 1)] = (op.pc, op.actual)

    # -- one clock cycle --------------------------------------------------------------------

    def cycle(self) -> CycleTrace:
        self.cycles += 1
        cyc = self.cycles
        trace = CycleTrace(cyc, {s: None for s in STAGES})
        ev = trace.events
        self._views = []
        for h in self.harts:                    # mipsconfig7 (RST: reset TAGE on 0 -> 1)
            p = h.m.p8700
            if p is not None:
                if p.config7 & RST and not h.cfg7 & RST:
                    self.tage.reset()
                    ev.append("mipsconfig7.RST: the TAGE predictor is reset")
                h.cfg7 = p.config7
            if h.parked and not h.m.debug_mode:
                h.parked = False
                h.fetch_pc = h.m.pc
                h.fetch_ready = cyc
                self._ev(ev, h, f"resumed: fetch 0x{h.m.pc:08x}")
        grads = self._graduate(ev, trace)
        executing = [op for h in self.harts for op in h.rob
                     if op.issued is not None and op.done is not None and op.done > cyc]
        if executing:
            trace.stages["EX"] = self._view(min(executing, key=lambda o: o.seq))
        for op in executing:
            op.timeline.setdefault(cyc, "EX")
        issued = self._issue(ev, trace)
        self._resolve(issued, ev)
        whys: dict[int, str | None] = {}
        n = len(self.harts)
        for j in range(n):
            h = self.harts[(self._prio + j) % n]
            if not self._running(h) or h.parked or h.done:
                continue
            self._decode(h, ev)
            whys[h.k] = self._dispatch(h, ev, trace)
        self._fetch(ev, trace)
        for h in self.harts:                    # the waiting ones' timelines
            for op in h.rob:
                op.timeline.setdefault(cyc, "iq" if op.issued is None else "co")
            for op in h.ibuf:
                op.timeline.setdefault(cyc, "ib")
        self._events(grads, issued, whys)
        self._why(trace, grads, whys)
        self._prio = (self._prio + 1) % n
        for h in self.harts:
            h.m.advance(1)
            h.m.counter_written.clear()
        self.history.append(trace)
        return trace

    def _events(self, grads: dict[int, int], issued: list[Op], whys: dict) -> None:
        """The cycle events of Table 11.4, per hart."""
        isu = {h.k: 0 for h in self.harts}
        for op in issued:
            isu[op.hart] += 1
        stalled_all = bool(whys) and all(w for w in whys.values())
        for h in self.harts:
            p = h.m.p8700
            if p is None or not p.counters.armed:
                continue
            c = p.counters
            g = grads.get(h.k, 0)
            c.add(2 + min(g, 4) - 1 if g else 6)
            c.add(14 + min(isu[h.k], 4))
            why = whys.get(h.k)
            blocked = why in ("window full", "ALQ full", "AGQ full", "FPQ full")
            if blocked and not h.was_blocked:
                c.add(19)
            h.was_blocked = blocked
            if why in ("empty", "front end"):
                c.add(22)
                c.add(148)
            if why:
                c.add(136)
            if stalled_all:
                c.add(128)
            if h.rob and h.rob[0].unit == "LD" and h.rob[0].issued is not None \
                    and h.rob[0].done is not None and h.rob[0].done > self.cycles \
                    and h.rob[0].mem_wait:
                c.add(35)
            if any(o.insn is not None and o.insn.base.name in _FENCES and o.issued is None
                   and o.busy for o in h.rob):
                c.add(36)
            if h.refilling and why in ("empty", "front end"):
                c.add(137)

    def _why(self, trace: CycleTrace, grads: dict[int, int], whys: dict) -> None:
        """A cycle in which nothing graduated: why the oldest instruction
        waits (the stall kinds of ``stalls``)."""
        if any(grads.values()):
            return
        for h in self.harts:
            if not h.rob:
                why = whys.get(h.k)
                if h.done or why is None and not h.ibuf:
                    continue
                if h.m.waiting:
                    kind, text = "wait", _sleep(h.m)
                elif h.ibuf and h.ibuf[0].fetched + h.ibuf[0].fetch_wait >= self.cycles:
                    op = h.ibuf[0]
                    kind, text = "I-miss", (f"the fetch of 0x{op.pc:08x} waits for memory "
                                            f"({op.fetched + op.fetch_wait - self.cycles + 1}"
                                            " more cycle(s))")
                elif h.hold is not None or h.refilling or why in ("empty", "front end"):
                    kind, text = "front end", ("the front end refills after a redirect"
                                               if h.refilling else "fetch to dispatch")
                elif why == "debugger":
                    kind, text = "debugger", "held at a breakpoint"
                else:
                    kind, text = why or "front end", why or "nothing to dispatch"
            else:
                op = h.rob[0]
                if op.issued is None:
                    if op.dispatched == self.cycles:
                        kind, text = "front end", f"{op.text} was dispatched this cycle"
                    elif op.not_before > self.cycles:
                        kind, text = "refetch", f"{op.text} is fetched again (a bond)"
                    elif op.srcs and not self._ready_quiet(op):
                        s = next((s for s in op.srcs if s.issued is None or
                                  (s.done or 0) > self.cycles), None)
                        kind = "dependency"
                        text = f"{op.text} waits for {s.text if s else 'its operands'}"
                    else:
                        kind, text = "issue", f"{op.text} waits for {op.unit}"
                elif op.mem_wait and op.done > self.cycles:
                    kind, text = "memory", f"{op.text} waits for memory"
                else:
                    kind, text = "execute", f"{op.text} executes ({op.unit}, " \
                                            f"{op.done - self.cycles} more cycle(s))"
            self._stall(kind)
            trace.stall = f"{kind}: " + (f"hart {h.k}: " if len(self.harts) > 1 else "") + text
            return

    def _ready_quiet(self, op: Op) -> bool:
        busy = op.busy
        r = self._ready(op, self.cycles)
        op.busy = busy
        return r

    # -- views -------------------------------------------------------------------------------

    def _view(self, op: Op | None) -> StageView | None:
        if op is None:
            return None
        comp = f"RVC 16-bit 0x{op.word:04x}" if op.size == 2 and op.word is not None else ""
        text = op.text if not op.ghost else op.text + "  (wrong path)"
        return StageView(op.seq, op.pc, text, None, op.fetch_cache, op.fetch_wait,
                         op.mem_wait, op.fetch_tlb, op.word, comp)

    def views(self) -> list[tuple[int, int, Any]]:
        """(hart, seq, InstrView) of the instructions executed in the last
        cycle."""
        return list(self._views)

    def lines(self, what: str = "") -> list[str]:
        """The core's state (debugger ``core``): the window, the queues,
        the units, the front end, the predictors."""
        cyc = self.cycles
        out = [f"P8700 core: cycle {cyc}, {self.retired} graduated, "
               f"{len(self.harts)} hart(s)"]
        if what in ("", "window"):
            for h in self.harts:
                out.append(f"hart {h.k}: window {len(h.rob)}/{self._window(h)}, "
                           f"instruction buffer {len(h.ibuf)}/{self.cfg['IBUF']}, fetch "
                           f"0x{h.fetch_pc:08x}" + self._front(h))
                for op in list(h.rob)[:12]:
                    out.append("  " + self._row(op))
                if len(h.rob) > 12:
                    out.append(f"  ... {len(h.rob) - 12} more")
        if what in ("", "queues"):
            busy = ", ".join(f"{u} until {c}" for u, c in sorted(self.busy.items()) if c > cyc)
            out.append("queues: " + ", ".join(f"{q} {len(v)}/{self.cfg['QUEUE']}"
                                              for q, v in self.queues.items())
                       + (f"; busy: {busy}" if busy else ""))
        if what in ("", "predict"):
            st = self.stats
            rate = f"{100 * st['mispredicted'] / st['branches']:.1f}%" if st["branches"] \
                else "-"
            out.append(f"predictors: {st['branches']} CTIs graduated, {st['mispredicted']} "
                       f"mispredicted ({rate}); BTB {len(self.btb)}/{self.cfg['BTB']}, "
                       f"JRC {len(self.jrc)}/{self.cfg['JRC']}, TAGE "
                       f"{sum(e is not None for t in self.tage.tables for e in t)} tagged "
                       f"entries")
            for h in self.harts:
                depth = self.cfg["RPS"]
                top = [f"0x{h.rps[(h.rps_top - j) % depth]:x}" for j in range(3)]
                out.append(f"hart {h.k}: RPS top {', '.join(top)}; history "
                           f"{h.ghist & 0xFFFF:016b} (newest right)")
            out.append(f"replays {st['replays']}, bonds {st['bonds']}, wrong-path "
                       f"instructions dispatched {st['ghosts']}, killed {st['killed']}")
        return out

    def rows(self) -> list[dict[str, Any]]:
        """The window, oldest first per hart (the GUI's table): hart, seq,
        pc, text, unit, state ("queued" / "executing" / "complete"),
        cycles left, wrong path, mispredicted, bonded."""
        cyc = self.cycles
        out = []
        for h in self.harts:
            for op in h.rob:
                if op.issued is None:
                    state, left = "queued" if op.queue else "dispatched", None
                elif op.done is not None and op.done > cyc:
                    state, left = "executing", op.done - cyc
                else:
                    state, left = "complete", 0
                out.append({"hart": h.k, "seq": op.seq, "pc": op.pc, "text": op.text,
                            "unit": op.unit, "state": state, "left": left,
                            "ghost": op.ghost, "mispredicted": op.mispredict and not op.ghost,
                            "bonded": bool(op.bond or op.bonded)})
        return out

    def _front(self, h: _Hart) -> str:
        if h.done:
            return " (exited)"
        if h.parked:
            return " (Debug Mode)"
        if h.hold is not None:
            return f" (waits for {h.hold.text} to graduate)"
        if h.stop is not None:
            return f" (waits for {h.stop.text} to resolve)"
        if h.wrong is not None:
            return f" (wrong path after {h.wrong.text})"
        return ""

    def _row(self, op: Op) -> str:
        cyc = self.cycles
        if op.issued is None:
            state = "queued" if op.queue else "dispatched"
        elif op.done is not None and op.done > cyc:
            state = f"executing ({op.done - cyc} more)"
        else:
            state = "complete"
        flags = " wrong-path" if op.ghost else ""
        flags += " mispredicted" if op.mispredict and not op.ghost else ""
        flags += " bonded" if op.bond or op.bonded else ""
        return f"{op.seq:5} 0x{op.pc:08x} {op.text[:30]:30} {op.unit:4} {state}{flags}"

    # -- running ----------------------------------------------------------------------------

    def empty(self) -> bool:
        return all(not h.rob and not h.ibuf for h in self.harts) and \
            all(h.done or h.parked or not self._running(h) for h in self.harts)

    def hart_empty(self, k: int) -> bool:
        h = self.harts[k]
        return not h.rob and (h.done or not h.ibuf)

    def next_op(self, k: int) -> Op | None:
        """Hart ``k``'s next instruction to execute (dispatch), when known
        to be on the right path."""
        h = self.harts[k]
        if h.wrong is not None or h.hold is not None or h.done or h.parked or not h.ibuf \
                or h.stop is not None and h.stop.dispatched is not None:
            return None                         # (behind an unresolved jump: unknown)
        op = h.ibuf[0]
        return op if op.insn is not None or op.fetch_exc is not None \
            or op.fetch_error is not None else None

    def oldest(self, k: int) -> Op | None:
        h = self.harts[k]
        return h.rob[0] if h.rob else None

    def run(self, *, max_cycles: int = 100_000, until_pc: int | None = None) -> int:
        start = self.cycles
        while self.cycles - start < max_cycles:
            if all(h.m.halted for h in self.harts) and self.empty():
                break
            if until_pc is not None and any(
                    (op := self.next_op(h.k)) is not None and op.pc == until_pc
                    for h in self.harts):
                break
            self.cycle()
        return self.cycles - start

    def drain(self, keep: set[int] | None = None, *, max_cycles: int = 1000) -> None:
        self.fetching = False
        for h in self.harts:
            self._kill(h, lambda o: keep is None or o.pc not in keep, ibuf_only=True)
            h.done = True
        for _ in range(max_cycles):
            if all(not h.rob for h in self.harts):
                return
            self.cycle()

    def chart(self, *, width: int = 60, last: bool = False) -> str:
        """Pipeline diagram: one row per instruction (the killed ones too),
        one column per cycle: IF fetch, ib instruction buffer, DR dispatch
        (it executes), iq in a queue, IS issue, EX executing, co complete,
        GR graduate, rp replayed, xx killed."""
        cycles = [t.cycle for t in self.history]
        cycles = cycles[-width:] if last else cycles[:width]
        head = f"{'':10} {'instruction':28} " + " ".join(f"{c:>2}"[-2:] for c in cycles)
        lines = [head]
        multi = len(self.harts) > 1
        for op in sorted(self.slots, key=lambda x: x.seq):
            if not any(c in op.timeline for c in cycles):
                continue
            text = (f"{op.hart}:" if multi else "") + op.text + (" *" if op.ghost else "")
            lines.append(f"{op.pc:08x}   {text[:28]:28} "
                         + " ".join(op.timeline.get(c, "  ") for c in cycles))
        return "\n".join(lines)


#: analyze()'s latency table (unused: Table 12.1 decides; a stable key)
from .pipeline import LATENCY as _NO_LATENCY  # noqa: E402


def _overlap(a: list, b: list) -> bool:
    return any(x < y + m and y < x + n for x, n in a for y, m in b)


class HartPipeline:
    """One hart's view of its core's :class:`P8700Pipeline` (what the
    Session asks a pipeline): the next instruction to execute, the one
    executing, the cycle (the core's: :meth:`cycle` runs the whole core)."""

    def __init__(self, core: P8700Pipeline, k: int) -> None:
        self.core, self.k = core, k

    def __getattr__(self, name: str):
        return getattr(self.core, name)

    @property
    def watch_hit(self):
        return self.core.watch_hit

    @watch_hit.setter
    def watch_hit(self, v) -> None:
        self.core.watch_hit = v

    def cycle(self) -> CycleTrace:
        return self.core.cycle()

    def entering_ex(self):
        return self.core.next_op(self.k)

    def next_slot(self):
        return self.core.next_op(self.k)

    def current_slot(self):
        return self.core.oldest(self.k)

    def empty(self) -> bool:
        return self.core.hart_empty(self.k)
