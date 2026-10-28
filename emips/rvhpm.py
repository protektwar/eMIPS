"""The generic hart's hardware performance monitor (LF8; The RISC-V
Instruction Set Manual, Volume II: Privileged Architecture 20250508 3.1.10-
3.1.12, ch. 7 "Smcntrpmf", ch. 20 "Sscofpmf"; Volume I's Zihpm chapter).

With Zihpm, mhpmcounter3-31 count the event their mhpmevent selects.  The
manual leaves the events to the platform; eMIPS's mhpmevent event field is
the SBI PMU extension's ``event_idx`` zero-extended (SBI 3.0 11.4: "the
RISC-V platform hardware implementation may use the zero-extended
event_idx as the expected value", as QEMU's virt board does):

- **hardware general events** (type 0, Table 31): 1 cycles, 2 instructions
  retired, 3 / 4 L1 cache references / misses (instruction and data), 5
  branch instructions (conditional branches and jumps), 6 branch misses —
  the 5-stage pipeline predicts every branch not taken, so a taken branch
  or a jump redirects fetch: a misprediction in both models —, 7 / 10 bus
  / reference cycles (the platform timer's ticks), 8 / 9 stalled cycles in
  the front end (IF waits) / back end (ID or MEM holds) of the 5-stage
  pipeline (the instruction model has none);
- **hardware cache events** (type 1, Tables 32-34: ``cache_id << 3 | op <<
  1 | result``): L1D, L1I, LL (the L3, else the L2), DTLB, ITLB, BPU; op
  read / write / prefetch, result access / miss.  NODE never counts.

Another value is not an event: the field reads 0 (WARL).  Sscofpmf adds OF
and MINH / SINH / UINH / VSINH / VUINH (bits 63-58; a missing mode's bit
reads 0): an inhibited mode does not count, an overflow of the 64-bit
counter sets OF and — when OF was 0 — the local counter-overflow interrupt
(LCOFIP, mip bit 13).  An event belongs to the mode its instruction started
in (Smcntrpmf 7.3's rule, which mcycle / minstret follow too).

Spike's counters read 0 (``const_csr_t``): the counting is pinned to the
manual; Spike checks the CSRs' rules.  Choices: ``sim.DEVIATIONS["RISC-V
counters"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .riscv import RV
    from .simrv import RVMachine

M64 = 0xFFFFFFFFFFFFFFFF
#: mhpmevent's Sscofpmf bits (20.1) and Smcntrpmf's xINH (7.2.1)
OF, MINH, SINH, UINH, VSINH, VUINH = (1 << b for b in (63, 62, 61, 60, 59, 58))
#: the local counter-overflow interrupt (mip / mie bit 13)
LCOFI = 13
FIRST = 3

#: SBI 3.0 Table 31: the hardware general events eMIPS counts
GENERAL = {1: "cycles", 2: "instructions", 3: "cache references", 4: "cache misses",
           5: "branch instructions", 6: "branch misses", 7: "bus cycles",
           8: "stalled cycles (front end)", 9: "stalled cycles (back end)",
           10: "reference cycles"}
#: Table 32: the caches (6 NODE: no such cache in eMIPS)
CACHES = {0: "L1D", 1: "L1I", 2: "LL", 3: "DTLB", 4: "ITLB", 5: "BPU"}
OPS = {0: "read", 1: "write", 2: "prefetch"}
RESULTS = {0: "access", 1: "miss"}
#: every event_idx eMIPS accepts -> its name
EVENTS = {**GENERAL, **{1 << 16 | c << 3 | o << 1 | r: f"{CACHES[c]} {OPS[o]} {RESULTS[r]}"
                        for c in CACHES for o in OPS for r in RESULTS}}
EVENT_MASK = 0xFFFFF                            # (event_idx: 20 bits, SBI 11.1)

_BRANCHES = frozenset({"beq", "bne", "blt", "bge", "bltu", "bgeu"})
_JUMPS = frozenset({"jal", "jalr", "cm.jt", "cm.jalt", "cm.popret", "cm.popretz"})


def never(ev: int) -> bool:
    """An accepted event no eMIPS source produces: an L1I / ITLB write, an
    LL / DTLB / ITLB prefetch, a BPU write or prefetch (the counter stays)."""
    if ev >> 16 != 1:
        return False
    cid, op = ev >> 3 & 0x1FFF, ev >> 1 & 3
    return (cid in (1, 4) and op == 1) or (cid in (2, 3, 4) and op == 2) \
        or (cid == 5 and op != 0)


def _sampled(ev: int) -> bool:
    """An event read from the caches' / MMU's counters (not the BPU's)."""
    return ev in (3, 4) or ev >> 16 == 1 and (ev >> 3 & 0x1FFF) != 5


def inhibit_bit(priv: int, v: int) -> int:
    """The xINH bit of a mode (7.2.1 / 20.1)."""
    if priv == 3:
        return MINH
    if priv == 1:
        return VSINH if v else SINH
    return VUINH if v else UINH


def inh_mask(c) -> int:
    """The xINH bits this hart has (a missing mode's: read-only 0)."""
    mask = MINH
    if c.has_s:
        mask |= SINH
    if c.has_u:
        mask |= UINH
    if c.has_h:
        mask |= VSINH | VUINH
    return mask


class HPM:
    """mhpmcounter3-31 / mhpmevent3-31 of a generic hart with Zihpm."""

    def __init__(self, m: "RVMachine") -> None:
        self.m = m
        self.sscofpmf = "sscofpmf" in m.extensions
        self.event = [0] * 32
        self.count = [0] * 32
        #: some counter has an event
        self.armed = False
        #: the sampled sources' last totals (cache / TLB events), by counter
        self._last: dict[int, int] = {}
        #: the platform timer when last sampled (events 7 / 10)
        self._time = 0

    # -- the CSRs --------------------------------------------------------------------

    def event_mask(self) -> int:
        """mhpmevent's writable bits: the event field, with Sscofpmf OF and
        the xINH bits of the hart's modes."""
        return EVENT_MASK | ((OF | inh_mask(self.m.csrs)) if self.sscofpmf else 0)

    def write_event(self, n: int, v: int, mask: int = M64) -> None:
        """mhpmevent ``n`` (``mask``: the half an RV32 access writes)."""
        mask &= self.event_mask()
        new = (self.event[n] & ~mask) | (v & mask)
        if new & EVENT_MASK not in EVENTS:
            new &= ~EVENT_MASK                  # (not an event: 0, WARL)
        self.event[n] = new
        self._rearm()

    def write_count(self, n: int, v: int, mask: int = M64) -> None:
        self.count[n] = ((self.count[n] & ~mask) | (v & mask)) & M64
        self._last.pop(n, None)
        self._rearm()

    def _rearm(self) -> None:
        was = self.armed
        self.armed = any(self.event[n] & EVENT_MASK for n in range(FIRST, 32))
        if self.armed and not was:
            self._time = self.m.mtime
        for n in range(FIRST, 32):
            ev = self.event[n] & EVENT_MASK
            if _sampled(ev) and n not in self._last:
                self._last[n] = self._sample(ev)

    def overflowed(self) -> int:
        """scountovf's bits (20.2): the OF bits, bit n = counter n."""
        return sum(1 << n for n in range(FIRST, 32) if self.event[n] & OF)

    # -- counting --------------------------------------------------------------------

    def _counts(self, n: int, mode: tuple[int, int]) -> bool:
        if self.m.csrs.mcountinhibit >> n & 1:
            return False
        return not self.event[n] & inhibit_bit(*mode)

    def add(self, ev: int, k: int, mode: tuple[int, int]) -> None:
        """``k`` occurrences of event ``ev`` in ``mode`` (priv, V)."""
        for n in range(FIRST, 32):
            if self.event[n] & EVENT_MASK == ev and k and self._counts(n, mode):
                self._bump(n, k)

    def _bump(self, n: int, k: int) -> None:
        if f"hpm{n}" in self.m.counter_written:
            return                              # (3.1.10: a CSR write takes effect after
        v = self.count[n] + k                   # the writing instruction has completed)
        if v > M64:                             # (20.1: unsigned overflow of 64 bits)
            if self.sscofpmf and not self.event[n] & OF:
                self.event[n] |= OF
                self.m.csrs.mip_sw |= 1 << LCOFI
            v &= M64
        self.count[n] = v

    def retired(self, i: "RV", pc: int, nxt: int, mode: tuple[int, int]) -> None:
        """An instruction retired in ``mode`` (it started there)."""
        self.add(2, 1, mode)
        name = i.base.name
        if name in _BRANCHES or name in _JUMPS:
            taken = name in _JUMPS or nxt != (pc + i.size) & self.m.amask
            self.add(5, 1, mode)
            self.add(1 << 16 | 5 << 3, 1, mode)             # BPU read access
            if taken:
                self.add(6, 1, mode)
                self.add(1 << 16 | 5 << 3 | 1, 1, mode)     # BPU read miss

    def tick(self, cycles: int, mode: tuple[int, int]) -> None:
        """``cycles`` clock cycles in ``mode``: the cycle events, the
        platform timer's ticks meanwhile, the sampled cache / TLB events."""
        self.add(1, cycles, mode)
        now = self.m.mtime
        timer, self._time = now - self._time, now
        if timer > 0:
            self.add(7, timer, mode)
            self.add(10, timer, mode)
        for n in range(FIRST, 32):
            ev = self.event[n] & EVENT_MASK
            if _sampled(ev):
                now = self._sample(ev)
                d = now - self._last.get(n, now)
                self._last[n] = now
                if d > 0 and self._counts(n, mode):
                    self._bump(n, d)

    def stall(self, front: bool, mode: tuple[int, int]) -> None:
        """A stalled cycle of the 5-stage pipeline (8 front end, 9 back end)."""
        self.add(8 if front else 9, 1, mode)

    def _sample(self, ev: int) -> int:
        """The running total of a sampled event (the caches' and the
        MMU's counters, emips.cacheview's sources)."""
        m = self.m
        c = m.caches
        if ev in (3, 4):                        # L1 references / misses, I and D
            if c is None:
                return 0
            tot = 0
            for cache in (c.i, c.d):
                if cache is not None:
                    tot += cache.stats.misses if ev == 4 else cache.stats.accesses
            return tot
        cid, op, res = ev >> 3 & 0x1FFF, ev >> 1 & 3, ev & 1
        if cid in (3, 4):                       # the TLB: fetch / load / store
            st = getattr(m.mmu, "kind_stats", None)
            if st is None:
                return 0
            kind = "fetch" if cid == 4 else "store" if op == 1 else "load" if op == 0 else None
            if kind is None or (cid == 4 and op != 0):
                return 0
            h, mi = st.get(kind, (0, 0))
            return mi if res else h + mi
        if c is None:
            return 0
        if cid == 2:
            cache = c.l3 if c.l3 is not None else c.l2
            kinds = {0: ("read",), 1: ("write",), 2: ()}[op]
        else:
            cache = c.d if cid == 0 else c.i
            kinds = {0: ("load",) if cid == 0 else ("fetch",), 1: ("store",) if cid == 0
                     else (), 2: ("pref",)}[op]
        if cache is None:
            return 0
        k = cache.stats.kinds
        return sum((k[x][1] if res else k[x][0] + k[x][1]) for x in kinds if x in k)

    # -- the views -------------------------------------------------------------------

    def lines(self) -> list[str]:
        """The counters with an event (the debugger's ``perf``)."""
        out = []
        for n in range(FIRST, 32):
            e = self.event[n]
            if not e and not self.count[n]:
                continue
            ev = e & EVENT_MASK
            flags = " ".join(f for f, b in (("OF", OF), ("MINH", MINH), ("SINH", SINH),
                                             ("UINH", UINH), ("VSINH", VSINH),
                                             ("VUINH", VUINH)) if e & b)
            out.append(f"mhpmcounter{n} = {self.count[n]}  event 0x{ev:x} "
                       f"{EVENTS.get(ev, 'none')}{'  ' + flags if flags else ''}")
        return out or ["no counter has an event (write an SBI event_idx into mhpmevent3-31)"]
