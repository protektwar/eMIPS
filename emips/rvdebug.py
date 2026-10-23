"""RISC-V debug support in the hart (PLAN M25 phase 2e): the trigger module
(Sdtrig) and Debug Mode (Sdext), pinned to The RISC-V Debug Specification
1.0 (chs. 4 / 5; ``docs`` keeps the PDF).

:class:`Triggers` holds a hart's triggers (four, as Spike) and the CSRs
that reach them — tselect, tdata1-3, tinfo, tcontrol (only without
S-mode: 5.4's second solution), mcontext, scontext.  Every trigger
supports the types mcontrol (2), icount (3), itrigger (4), etrigger (5),
mcontrol6 (6) and disabled (15) (tinfo version 1).  The hart asks it

* before each instruction: :meth:`Triggers.icount` (5.7.13: a pending
  trigger fires, then the enabled ones count this instruction);
* for each 16-bit fetch parcel and the whole fetched instruction, each
  load (the address, then the data loaded) and each store (address and
  data at once): :meth:`Triggers.memory` (5.5, 5.7.11 / 5.7.12);
* after a trap: :meth:`Triggers.trap` (itrigger / etrigger, 5.7.14 /
  5.7.15).

A trigger that fires raises :class:`Fired` (timing *before*) or leaves it
in :attr:`Triggers.after` (timing *after*: the hart takes it once the
instruction has retired).  Action 0 is a breakpoint exception, action 1
Debug Mode (only with dmode = 1, writable in Debug Mode only); the trace
actions 2-4 set ``hit`` and nothing else (no trace encoder is attached:
5.3, "triggers that have an action other than 0 or 1 don't affect the
execution of the hart").

Where the specification leaves a choice, eMIPS takes Spike's
(``riscv/triggers.cc``): the trigger checks' order (an address match
before translation, the parcels, a chain matched by the same access),
5.4's solution 1 with S-mode (no action-0 match in M-mode with MIE = 0,
nor in S-mode with SIE = 0 when breakpoints are delegated) and the
tcontrol solution without it, an icount trigger counts an instruction as
it starts (its tdata1 then reads the count from before), the match modes
equal / napot (mcontrol6 only, maskmax6 = XLEN - 1) / ge / lt / mask low
/ mask high, size 0 only, load-data triggers fire after the load,
textra's mhselect 0 / 4 and sselect 0-2.  Choices:
``sim.DEVIATIONS["RISC-V"]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simrv import RVMachine

MCONTROL, ICOUNT, ITRIGGER, ETRIGGER, MCONTROL6, DISABLED = 2, 3, 4, 5, 6, 15
TYPES = (MCONTROL, ICOUNT, ITRIGGER, ETRIGGER, MCONTROL6, DISABLED)
#: tinfo (5.7.5): the supported types, version 1 (the ratified 1.0)
TINFO = sum(1 << t for t in TYPES) | 1 << 24
#: actions (Table 12)
BREAKPOINT, DEBUG_MODE = 0, 1
ACTION_MAX = 4                              # (2-4: trace on / off / notify)
#: tcontrol (5.7.6)
MTE, MPTE = 1 << 3, 1 << 7
#: match (5.7.12)
EQUAL, NAPOT, GE, LT, MASK_LOW, MASK_HIGH = 0, 1, 2, 3, 4, 5

#: dcsr (4.9.1)
DCSR_DEBUGVER = 4 << 28
EBREAKM, EBREAKS, EBREAKU = 1 << 15, 1 << 13, 1 << 12
CAUSE_SHIFT = 6
MPRVEN, STEP = 1 << 4, 1 << 2
#: dcsr.cause (Table 8)
CAUSE_EBREAK, CAUSE_TRIGGER, CAUSE_HALTREQ, CAUSE_STEP, CAUSE_RESETHALTREQ = 1, 2, 3, 4, 5


class Fired(Exception):
    """A trigger fired: its ``action`` and the tval of a breakpoint
    exception (the matching address; 0 for icount / itrigger / etrigger)."""

    def __init__(self, action: int, tval: int = 0, after: bool = False) -> None:
        super().__init__(action, tval)
        self.action = action
        self.tval = tval
        self.after = after


def _f(v: int, lo: int, width: int = 1) -> int:
    return v >> lo & ((1 << width) - 1)


@dataclass
class Trigger:
    """One trigger: its type and the fields of every type (a type change
    keeps tdata2 / tdata3, as Spike)."""
    type: int = DISABLED
    dmode: int = 0
    tdata2: int = 0
    # the mode filters, the action, hit (mcontrol6: the 2-bit {hit1, hit0})
    m: int = 0
    s: int = 0
    u: int = 0
    action: int = 0
    hit: int = 0
    # mcontrol / mcontrol6
    select: int = 0
    timing: int = 0
    chain: int = 0
    match: int = 0
    execute: int = 0
    store: int = 0
    load: int = 0
    # icount (count / pending as the current instruction found them: 5.7.13)
    count: int = 1
    pending: int = 0
    count_read: int = 1
    pending_read: int = 0
    # itrigger
    nmi: int = 0
    # textra32 / textra64 (5.7.17 / 5.7.18)
    mhvalue: int = 0
    mhselect: int = 0
    sbytemask: int = 0
    svalue: int = 0
    sselect: int = 0


class Triggers:
    """The trigger module of one hart (see the module docstring)."""

    def __init__(self, m: "RVMachine", count: int = 4) -> None:
        self.m = m
        self.xlen = m.xlen
        self.triggers = [Trigger() for _ in range(count)]
        self.tselect = 0
        self.tcontrol = 0
        self.mcontext = 0
        self.scontext = 0
        #: a timing-after match of the current instruction (taken when it
        #: retires; no other trigger is looked at until then)
        self.after: Fired | None = None
        self._armed()

    @property
    def count(self) -> int:
        return len(self.triggers)

    @property
    def has_s(self) -> bool:
        return 1 in self.m.modes

    @property
    def has_u(self) -> bool:
        return 0 in self.m.modes

    def _armed(self) -> None:
        """What the hart must look at (Spike's trigger_updated)."""
        ts = self.triggers
        mc = [t for t in ts if t.type in (MCONTROL, MCONTROL6)]
        self.fetch = any(t.execute for t in mc)
        self.loads = any(t.load for t in mc)
        self.stores = any(t.store for t in mc)
        self.icounts = [t for t in ts if t.type == ICOUNT]
        self.traps = any(t.type in (ITRIGGER, ETRIGGER) for t in ts)

    # -- the CSRs (5.7) -----------------------------------------------------------

    def write_tselect(self, v: int) -> None:
        """WARL: an index beyond the triggers keeps the old one (Spike)."""
        if v < self.count:
            self.tselect = v

    def read1(self) -> int:
        t = self.triggers[self.tselect]
        x = self.xlen
        v = t.type << (x - 4) | t.dmode << (x - 5)
        if t.type == MCONTROL:
            v |= (t.hit << 20 | t.select << 19 | t.timing << 18 | t.action << 12
                  | t.chain << 11 | t.match << 7 | t.m << 6 | t.s << 4 | t.u << 3
                  | t.execute << 2 | t.store << 1 | t.load)
        elif t.type == MCONTROL6:
            v |= ((t.hit >> 1) << 25 | (t.hit & 1) << 22 | t.select << 21 | t.action << 12
                  | t.chain << 11 | t.match << 7 | t.m << 6 | t.s << 4 | t.u << 3
                  | t.execute << 2 | t.store << 1 | t.load)
        elif t.type == ICOUNT:
            v |= (t.hit << 24 | t.count_read << 10 | t.m << 9 | t.pending_read << 8
                  | t.s << 7 | t.u << 6 | t.action)
        elif t.type in (ITRIGGER, ETRIGGER):
            v |= (t.hit << (x - 6) | t.m << 9 | t.s << 7 | t.u << 6 | t.action
                  | (t.nmi << 10 if t.type == ITRIGGER else 0))
        return v

    def write1(self, val: int) -> None:
        """tdata1 (5.7.2 and the types' WARL rules; Spike's module_t)."""
        i, ts, x, debug = self.tselect, self.triggers, self.xlen, self.m.debug_mode
        old = ts[i]
        if old.dmode and not debug:
            return                              # only Debug Mode writes it
        dmode = _f(val, x - 5)
        # 5.7.12 chain: zeroed when this trigger leaves dmode before one in it
        allow_chain = not (i + 1 < len(ts) and ts[i + 1].dmode and not dmode)
        if not debug:
            dmode = 0                           # dmode: writable in Debug Mode only
        if i > 0 and not ts[i - 1].dmode and ts[i - 1].chain and dmode:
            return                              # (a chain may not reach into dmode)
        typ = _f(val, x - 4, 4)
        t = Trigger(type=typ if typ in TYPES else DISABLED, dmode=dmode)
        t.mhvalue, t.mhselect, t.sbytemask, t.svalue, t.sselect = (
            old.mhvalue, old.mhselect, old.sbytemask, old.svalue, old.sselect)
        s, u = self.has_s, self.has_u

        def action(width: int) -> int:
            a = _f(val, 0 if width == 6 else 12, width)
            return BREAKPOINT if a > ACTION_MAX or (a == DEBUG_MODE and not dmode) else a

        if t.type in (MCONTROL, MCONTROL6):
            six = t.type == MCONTROL6
            t.hit = (_f(val, 25) << 1 | _f(val, 22)) if six else _f(val, 20)
            t.select = _f(val, 21 if six else 19)
            t.action = action(4)
            t.chain = _f(val, 11) if allow_chain else 0
            match = _f(val, 7, 4)
            t.match = match if match in (EQUAL, GE, LT, MASK_LOW, MASK_HIGH) or \
                (match == NAPOT and six) else EQUAL
            t.m, t.s, t.u = _f(val, 6), _f(val, 4) if s else 0, _f(val, 3) if u else 0
            t.execute, t.store, t.load = _f(val, 2), _f(val, 1), _f(val, 0)
            # timing: data-load triggers after, execute before, mcontrol6 before
            if t.select and t.load:
                t.timing = 1
            elif t.execute or six:
                t.timing = 0
            else:
                t.timing = _f(val, 18)
        elif t.type == ICOUNT:
            t.hit = _f(val, 24)
            t.count = t.count_read = _f(val, 10, 14)
            t.m, t.s, t.u = _f(val, 9), _f(val, 7) if s else 0, _f(val, 6) if u else 0
            t.pending = t.pending_read = _f(val, 8)
            t.action = action(6)
        elif t.type in (ITRIGGER, ETRIGGER):
            t.hit = _f(val, x - 6)
            t.nmi = _f(val, 10) if t.type == ITRIGGER else 0
            t.m, t.s, t.u = _f(val, 9), _f(val, 7) if s else 0, _f(val, 6) if u else 0
            t.action = action(6)
        ts[i] = t
        self._write2(t, old.tdata2)
        self._armed()

    def read2(self) -> int:
        return self.triggers[self.tselect].tdata2

    def write2(self, val: int) -> None:
        t = self.triggers[self.tselect]
        if t.dmode and not self.m.debug_mode:
            return
        self._write2(t, val)

    def _write2(self, t: Trigger, val: int) -> None:
        """5.7.12 match = 1: with all of bits maskmax6-1:0 set, bit maskmax6-1
        reads 0 (maskmax6 = XLEN - 1)."""
        t.tdata2 = val
        if t.type == MCONTROL6 and t.match == NAPOT:
            top = 1 << (self.xlen - 2)
            if val & (2 * top - 1) == 2 * top - 1:
                t.tdata2 = val & ~top

    def read3(self) -> int:
        t = self.triggers[self.tselect]
        mhselect = 4 if t.mhselect >= 4 else 0  # (without H: only 0 and 4)
        if self.xlen == 32:
            return (t.mhvalue << 26 | mhselect << 23 | t.sbytemask << 18 | t.svalue << 2
                    | t.sselect)
        return (t.mhvalue << 51 | mhselect << 48 | t.sbytemask << 36 | t.svalue << 2
                | t.sselect)

    def write3(self, val: int) -> None:
        t = self.triggers[self.tselect]
        if t.dmode and not self.m.debug_mode:
            return
        if self.xlen == 32:
            t.mhvalue, t.mhselect = _f(val, 26, 6), _f(val, 23, 3)
            t.sbytemask, svalue = _f(val, 18, 2), _f(val, 2, 16)
        else:
            t.mhvalue, t.mhselect = _f(val, 51, 13), _f(val, 48, 3)
            t.sbytemask, svalue = _f(val, 36, 4), _f(val, 2, 32)
        sselect = _f(val, 0, 2)
        t.svalue = svalue if self.has_s else 0
        t.sselect = sselect if self.has_s and sselect <= 2 else 0

    # -- matching ---------------------------------------------------------------------

    def _common(self, t: Trigger, prv: int) -> bool:
        """The mode filter (``prv``: the mode the event happened in), textra,
        and 5.4's re-entrancy rule for action 0 (the current mode)."""
        if not (t.m if prv == 3 else t.s if prv == 1 else t.u):
            return False
        if not self._textra(t):
            return False
        if t.action == BREAKPOINT:
            m = self.m
            c = m.csrs
            if self.has_s:
                if m.priv == 3 and not c.status_bit(3):            # MIE
                    return False
                if m.priv == 1 and c.medeleg >> 3 & 1 and not c.status_bit(1):   # SIE
                    return False
            elif m.priv == 3 and not self.tcontrol & MTE:
                return False
        return True

    def _textra(self, t: Trigger) -> bool:
        x = self.xlen
        if t.sselect == 1:                      # scontext, byte-masked
            mask = (1 << (16 if x == 32 else 32)) - 1
            for k in range(4):
                if t.sbytemask >> k & 1:
                    mask &= ~(0xFF << 8 * k)
            if self.scontext & mask != t.svalue & mask:
                return False
        elif t.sselect == 2:                    # satp.ASID (ASIDMAX 9 / 16)
            satp = self.m.csrs.satp
            asid = satp >> 22 & 0x1FF if x == 32 else satp >> 44 & 0xFFFF
            if asid != t.svalue & ((1 << (9 if x == 32 else 16)) - 1):
                return False
        if t.mhselect >= 4:                     # mcontext = mhvalue
            if self.mcontext & ((1 << (7 if x == 32 else 14)) - 1) != t.mhvalue:
                return False
        return True

    def _simple(self, t: Trigger, value: int) -> bool:
        x = self.xlen
        value &= (1 << x) - 1
        d = t.tdata2
        if t.match == EQUAL:
            return value == d
        if t.match == NAPOT:
            ones = ((d ^ (d + 1)).bit_length() - 1) + 1
            mask = ~((1 << ones) - 1) if ones < x else 0
            return value & mask == d & mask
        if t.match == GE:
            return value >= d
        if t.match == LT:
            return value < d
        half = x // 2
        low, high = d & ((1 << half) - 1), d >> half
        part = value & ((1 << half) - 1) if t.match == MASK_LOW else value >> half
        return part & high == low

    def _match(self, t: Trigger, op: str, addr: int, size: int,
               data: int | None) -> Fired | None:
        if not (t.execute if op == "execute" else t.store if op == "store" else t.load):
            return None
        if not self._common(t, self.m.priv):
            return None
        hit = (3 if t.timing else 1) if t.type == MCONTROL6 else 1
        if not t.select:
            for k in range(size):
                if self._simple(t, addr + k):
                    t.hit = hit
                    return Fired(t.action, addr + k, bool(t.timing))
            return None
        if size * 8 > self.xlen or data is None:
            return None
        if self._simple(t, data):
            t.hit = hit
            return Fired(t.action, addr, bool(t.timing))
        return None

    def memory(self, op: str, addr: int, size: int, data: int | None = None) -> None:
        """An access (``op`` execute / load / store) of ``size`` bytes at
        ``addr`` (with ``data``: the value loaded / stored / fetched).  Every
        trigger of a chain must match it (5.7.12); the highest action
        among the chains' ends wins (Debug Mode over a breakpoint)."""
        if self.after is not None or self.m.debug_mode:
            return
        best: Fired | None = None
        chain_ok = True
        for t in self.triggers:
            if not chain_ok:
                chain_ok = not t.chain
                continue
            if t.type not in (MCONTROL, MCONTROL6):
                chain_ok = True
                continue
            r = self._match(t, op, addr, size, data)
            if r is not None and not t.chain and r.action <= DEBUG_MODE and \
                    (best is None or best.action < r.action):
                best = r
            chain_ok = r is not None or not t.chain
        if best is None:
            return
        if best.after:
            self.after = best
        else:
            raise best

    def icount(self) -> Fired | None:
        """At the start of an instruction (5.7.13): a pending trigger
        enabled in the current mode fires (it clears pending, sets hit);
        then each enabled one counts the instruction (count 1 -> pending)
        unless one fired into Debug Mode.  tdata1 reads the counts as this
        instruction found them."""
        for t in self.icounts:
            t.count_read, t.pending_read = t.count, t.pending
        if self.m.debug_mode or not self.icounts:
            return None
        prv = self.m.priv
        best: Fired | None = None
        for t in self.icounts:
            if t.pending and self._common(t, prv):
                t.pending = 0
                t.hit = 1
                if t.action <= DEBUG_MODE and (best is None or best.action < t.action):
                    best = Fired(t.action, 0)
        if best is None or best.action != DEBUG_MODE:
            self.count_one(prv)
        return best

    def icount_trap(self, prv: int) -> None:
        """An interrupt taken from mode ``prv``: it counts (5.7.13 2.)."""
        for t in self.icounts:
            t.count_read, t.pending_read = t.count, t.pending
        if not self.m.debug_mode:
            self.count_one(prv)

    def count_one(self, prv: int) -> None:
        """One instruction (or trap) in mode ``prv`` for the icount
        triggers."""
        for t in self.icounts:
            if t.count and self._common(t, prv):
                if t.count == 1:
                    t.pending = 1
                t.count -= 1

    def trap(self, cause: int, interrupt: bool, prv: int) -> Fired | None:
        """After a trap from mode ``prv`` (5.7.14 / 5.7.15): an itrigger /
        etrigger with the cause's bit in tdata2 fires before the handler's
        first instruction."""
        if self.m.debug_mode or not self.traps:
            return None
        best: Fired | None = None
        for t in self.triggers:
            if t.type == (ITRIGGER if interrupt else ETRIGGER) and self._common(t, prv) \
                    and t.tdata2 >> cause & 1:
                t.hit = 1
                if t.action <= DEBUG_MODE and (best is None or best.action < t.action):
                    best = Fired(t.action, 0)
        return best

    # -- views ------------------------------------------------------------------------

    def lines(self) -> list[str]:
        names = {MCONTROL: "mcontrol", ICOUNT: "icount", ITRIGGER: "itrigger",
                 ETRIGGER: "etrigger", MCONTROL6: "mcontrol6", DISABLED: "disabled"}
        out = []
        for k, t in enumerate(self.triggers):
            modes = "".join(c for c, b in (("M", t.m), ("S", t.s), ("U", t.u)) if b)
            line = f"trigger {k}: {names[t.type]}"
            if t.type in (MCONTROL, MCONTROL6):
                ops = "".join(c for c, b in (("x", t.execute), ("r", t.load),
                                              ("w", t.store)) if b)
                line += (f" {ops or '-'} {'data' if t.select else 'address'}"
                         f" match {t.match} 0x{t.tdata2:x}{' chain' if t.chain else ''}")
            elif t.type == ICOUNT:
                line += f" count {t.count}{' pending' if t.pending else ''}"
            elif t.type in (ITRIGGER, ETRIGGER):
                line += f" causes 0x{t.tdata2:x}"
            if t.type != DISABLED:
                line += f" modes {modes or '-'} action {t.action}{' hit' if t.hit else ''}"
            out.append(line + (" (dmode)" if t.dmode else ""))
        return out
