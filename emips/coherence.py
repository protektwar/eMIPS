"""The Coherence Manager of a MIPS Coherent Processing System (PLAN M11
phase 2): MESI between the cores' L1 data caches, the shared L2 / L3
behind it, the address regions, the CM's error registers.

Sources: the MIPS CPS training notes ("CPS Introduction": the MESI
states, the coherent attributes 4 and 5, the intervention examples;
"Coherency Manager": GCR_CL_COHERENCE, the address regions and their CCA
override, the L2-only SYNC, the error types), Linux
(``arch/mips/include/asm/mips-cm.h`` register layouts,
``arch/mips/kernel/mips-cm.c`` the CM2 error decoding,
``arch/mips/mm/c-r4k.c``: "The MIPS Coherence Manager (CM) globalizes
address-based cache ops", ``arch/mips/kernel/cps-vec.S``: a core enters
the domain with GCR_CL_COHERENCE = 0xff).

The protocol, as the notes describe it::

    M  Modified   the only copy, dirty
    E  Exclusive  the only copy, clean (may be written without asking)
    S  Shared     a clean read-only copy; others may have one too
    I  Invalid

* A load miss with CCA 5 asks for the line **Shared** ("Coherent Read
  Share"): a core holding it Modified writes it back (to the L2 / memory)
  and sends it cache to cache, both end Shared; Exclusive -> Shared;
  with no other copy the requester gets it **Exclusive**.  CCA 4 asks
  for it Exclusive, like a store ("Coherent Read Own").
* A store miss asks for the line Exclusive: every other copy is
  invalidated (a Modified one gives its data cache to cache — the
  requester then holds it Modified); the store makes it Modified.
* A store hit on a Shared line first sends an **upgrade**: the other
  copies are invalidated.  A store hit on an Exclusive line just makes it
  Modified (no request).
* An instruction refill reads the newest copy without taking it
  ("Coherent Read Discard"); the instruction caches are not kept
  coherent with stores (SYNCI, which the CM globalizes, does that).
* Address-based CACHE operations (Hit Invalidate, Hit Writeback
  Invalidate, Hit Writeback; SYNCI) on a coherent address are done by
  every core of the domain.
* A core is in **coherent mode** when its own bit of GCR_Cx_COHERENCE
  is set; it receives the interventions of core ``i`` when bit ``i`` is
  set ("Setting a bit enables interventions from the corresponding
  Core").  Requests of a core not in coherent mode, and non-coherent
  attributes (0, 2, 3, 7), make no interventions.
* Another core's coherent store invalidating the line of an LL clears
  that core's LLbit (the SC then fails).

eMIPS choices (``sim.DEVIATIONS`` "Coherence Manager"): every request
completes at once, in the order the cores execute (core 0 first); a
Shared line is served by the L2 / memory, an Exclusive / Modified one
cache to cache (the CM's speculative memory read is cancelled); a
PrepareForStore (PREF 30) or a store from a core without a data cache
writes a Modified copy back before invalidating it; the CM's
inconsistent-state errors (17 / 18) are recorded and the request goes
on; the address regions' IOCU targets act as memory (no IOCU).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .timing import Bus
from .cache import (CACHED_CCA_CPS, CCA_UC, CCA_UCA, CCA_WT,
                    Cache, Geometry, Memory, SecondaryCache, _write_below)

M32 = 0xFFFFFFFF

#: GCR offsets of the CM (Linux mips-cm.h)
GCR_ERROR_MASK, GCR_ERROR_CAUSE = 0x40, 0x48
GCR_ERROR_ADDR, GCR_ERROR_MULT, GCR_L2_ONLY_SYNC_BASE = 0x50, 0x58, 0x70
#: CM2 error types (GCR_ERROR_CAUSE.ERRTYPE, 31:27; mips-cm.c cm2_causes)
ERR_GC_WR, ERR_GC_RD, ERR_COH_WR, ERR_COH_RD = 1, 2, 3, 4
ERR_INTVN_WR, ERR_INTVN_RD = 17, 18
#: the error types of requests that need a response (Table 11.11: "Return
#: SResp = ERROR if CM_ERROR_MASK[n] = 0")
ERR_NEEDS_RESPONSE = frozenset({2, 4, 6, 18, 24})
ERROR_NAMES = {1: "GC_WR_ERR", 2: "GC_RD_ERR", 3: "COH_WR_ERR", 4: "COH_RD_ERR",
               5: "MMIO_WR_ERR", 6: "MMIO_RD_ERR", 17: "INTVN_WR_ERR",
               18: "INTVN_RD_ERR", 24: "L2_RD_UNCORR", 25: "L2_WR_UNCORR",
               26: "L2_CORR"}
#: the CM2 commands (MCmd) of the requests (mips-cm.c cm2_cmd)
MCMD = {"write": 0x01, "read": 0x02, "own": 0x08, "share": 0x09, "discard": 0x0A,
        "upgrade": 0x0C, "writeback": 0x0D}
#: the target of a transaction (TR; mips-cm.c cm2_tr)
TR_GCR, TR_GIC, TR_CPC = 1, 2, 5
#: a line's state in an intervention error (C0..C3, cm2_core): 0 I, 2 S,
#: 4 M, 6 E; + 1 when it supplied the data
_STATE_CODE = {"I": 0, "S": 2, "M": 4, "E": 6}
#: region / default-region fields (GCR_REGn_MASK, GCR_BASE)
REG_CCAOVR_SHIFT, REG_CCAOVREN, REG_DROPL2, REG_CMTGT = 5, 1 << 4, 1 << 2, 3
GCR_REGS_BASE = (0x90, 0xA0, 0xB0, 0xC0)


def state_of(ln) -> str:
    """The MESI letter of a data-cache line."""
    if not ln.valid:
        return "I"
    if ln.dirty:
        return "M"
    return "S" if ln.shared else "E"


@dataclass
class _Holder:
    core: int
    caches: object
    cache: Cache
    s: int
    w: int

    @property
    def line(self):
        return self.cache.lines[self.s][self.w]


class CoherenceManager:
    """The CM2 of a :class:`~emips.cps.CPS`: the request ordering point of
    every core (bound with :meth:`attach`), the shared L2 / L3 and the
    memory port below them."""

    def __init__(self, cps, *, l2: Geometry | None = None, l3: Geometry | None = None,
                 ecc: bool = False, hardware_init: bool = False, seed: int = 0x1004) -> None:
        self.cps = cps
        #: core -> its emips.cache.Caches
        self.cores: dict[int, object] = {}
        self.l2 = self.l3 = None
        if l2 is not None:
            self.l2 = SecondaryCache("S", Geometry(l2.sets, l2.ways, l2.line, 2), ecc=ecc)
        if l3 is not None:
            self.l3 = SecondaryCache("T", Geometry(l3.sets, l3.ways, l3.line, 3), ecc=ecc)
        self.ecc = ecc and self.l2 is not None
        #: the system bus below the L2 / L3 (bound to the requesting core)
        self.memory = Memory(None)
        #: its timing, shared by the cores (the CM serializes their requests)
        self.bus = Bus(shared=True)
        if self.l3 is not None:
            self.l3.below = self.memory
        if self.l2 is not None:
            self.l2.below = self.l3 if self.l3 is not None else self.memory
            self.l2.policy = self.l2_policy
        rng = random.Random(seed)
        for c in self.secondary:
            c.randomize(rng) if not hardware_init else c.invalidate_all()
        #: the core whose request is being served
        self.current = None
        # the CM's registers
        # MD01025 Table 11.27: write errors interrupt, read errors get an
        # error response (the core takes a bus error)
        self.error_mask = 0x000A002A
        self.error_cause = 0
        self.error_addr = 0
        self.error_mult = 0
        self.l2_sync_base = 0
        #: what the CM did (for the views)
        self.counts = {"requests": 0, "interventions": 0, "cache-to-cache": 0,
                       "invalidations": 0, "upgrades": 0, "writebacks": 0,
                       "globalized": 0}

    @property
    def secondary(self) -> list[SecondaryCache]:
        return [c for c in (self.l2, self.l3) if c is not None]

    # -- the cores -----------------------------------------------------------------

    def attach(self, core: int, caches) -> None:
        """Bind a core's caches (before its Machine is built): they share
        the L2 / L3 and the memory port, and their coherent requests come
        here."""
        caches.cm = self
        caches.core = core
        caches.cached_cca = CACHED_CCA_CPS
        caches.l2, caches.l3 = self.l2, self.l3
        caches.ecc = self.ecc
        caches.memory = self.memory
        caches.timing.bus = self.bus             # one system bus behind the CM
        if caches.d is not None:
            caches.d.mesi = True
        self.cores[core] = caches
        self.current = None
        self.claim(caches)              # the views find memory before any request

    def claim(self, req) -> None:
        """The shared levels serve ``req`` now (its timing, its errors, its
        bus for the devices)."""
        if self.current is req:
            return
        self.current = req
        self.memory.c = req
        for c in self.secondary:
            c.timing = req.timing
            c.sink = req.record

    def first(self):
        """Where a coherent writeback goes: the L2, the L3 or memory."""
        if self.l2 is not None:
            return self.l2
        return self.l3 if self.l3 is not None else self.memory

    def _coherence(self, core: int) -> int:
        return self.cps.gcr.core[core][0x08]

    def coherent_mode(self, core: int) -> bool:
        """GCR_Cx_COHERENCE has the core's own bit."""
        return bool(self._coherence(core) >> core & 1)

    def snoopers(self, core: int) -> list[int]:
        """The cores that receive ``core``'s interventions: in coherent
        mode, accepting interventions from it, and powered."""
        from .cps import SEQ_D0
        out = []
        for j in sorted(self.cores):
            if j == core or self.cps.state[j] == SEQ_D0:
                continue
            coh = self._coherence(j)
            if coh >> j & 1 and coh >> core & 1:
                out.append(j)
        return out

    def _holders(self, core: int, pa: int, kind: str = "D") -> list[_Holder]:
        out = []
        for j in self.snoopers(core):
            caches = self.cores[j]
            c = caches.d if kind == "D" else caches.i
            if c is None:
                continue
            tag = c.tag_of(pa)
            for s in c.alias_sets(pa):
                w = c.find(s, tag)
                # locked lines are not coherent (the 1004K, Table 9.1)
                if w is not None and not c.lines[s][w].lock:
                    out.append(_Holder(j, caches, c, s, w))
        return out

    # -- requests -------------------------------------------------------------------

    def coherent_read(self, req, pa: int, n: int, intent: str) -> tuple[bytearray, int]:
        """A coherent line request of ``req`` (``intent``: "share", "own" or
        "discard"); sets ``req._fill = (shared, dirty, note)`` for the line
        it installs."""
        self.claim(req)
        self.counts["requests"] += 1
        holders = self._holders(req.core, pa)
        self.counts["interventions"] += len(self.snoopers(req.core))
        self._check(req, pa, holders, intent)
        src = next((h for h in holders if not h.line.shared), None)   # E or M
        notes = []
        data = None
        if src is not None:
            data = self._slice(src, pa, n)
        if src is not None and data is not None:
            self.counts["cache-to-cache"] += 1
            notes.append(f"core {src.core} had it {state_of(src.line)}: sent cache to cache")
        dirty = False
        if intent == "share":
            for h in holders:
                ln = h.line
                if ln.shared:
                    continue
                was = state_of(ln)
                if ln.dirty:
                    self._write_back(h)
                ln.shared = True
                h.cache._seal(ln)
                notes.append(f"core {h.core} {was} -> S"
                             + (" (written back to memory)" if was == "M" else ""))
            # GCR_CONTROL.SHST_EN (Table 11.23): always installed Shared
            shst = bool(self.cps.gcr.regs[0x10] & 0x8)
            shared = bool(holders) or shst
            notes.append("now Shared" + (" (SHST_EN)" if shst and not holders else "")
                         if shared else "no other copy: Exclusive")
        elif intent == "own":
            dirty = src is not None and src.line.dirty and data is not None
            for h in holders:
                self._invalidate(h)
                notes.append(f"core {h.core} -> I")
            shared = False
            notes.append("now Modified (the dirty data came along)" if dirty else "now Exclusive")
        else:                                          # discard: nothing changes
            shared = False
        req._fill = (shared, dirty, "; ".join(notes) if holders else
                     ("no other copy" if intent != "discard" else ""))
        if data is not None:
            req.read_below(pa, n, source="C2C")
            return data
        return req.read_below(pa, n)

    def upgrade(self, req, va: int, pa: int, s: int, w: int) -> None:
        """A store hit on a Shared line: invalidate the other copies."""
        self.claim(req)
        self.counts["requests"] += 1
        self.counts["upgrades"] += 1
        holders = self._holders(req.core, pa)
        self.counts["interventions"] += len(self.snoopers(req.core))
        self._check(req, pa, holders, "upgrade")
        for h in holders:
            self._invalidate(h)
        t = req.timing
        if t.active:
            t.coherence("upgrade: invalidating the other copies")
        if req.log is not None:
            text = ("upgrade S -> M: " + ", ".join(f"core {h.core} -> I" for h in holders)
                    if holders else "upgrade S -> M: no other copy left")
            for rec in reversed(req.log):
                if rec.level == "D":
                    rec.coherence = text
                    break

    def claim_line(self, req, pa: int, write_back: bool = True) -> None:
        """The line becomes ``req``'s alone (PrepareForStore, a store
        without a data cache): the other copies are invalidated, a
        Modified one written back first."""
        self.claim(req)
        self.counts["requests"] += 1
        for h in self._holders(req.core, pa):
            if write_back and h.line.dirty:
                self._write_back(h)
            self._invalidate(h)

    def global_op(self, req, kind: str, code: int, pa: int) -> str:
        """A Hit-type CACHE operation (or SYNCI) done in every core of the
        domain: D 4 Hit Invalidate, 5 Hit Writeback Invalidate, 6 Hit
        Writeback; I 4 Hit Invalidate."""
        self.claim(req)
        done = []
        for h in self._holders(req.core, pa, kind):
            ln = h.line
            before = state_of(ln) if kind == "D" else "V"
            if kind == "D" and code in (5, 6) and ln.dirty:
                self._write_back(h)
            if code in (4, 5):
                self._invalidate(h)
            done.append(f"core {h.core} {h.cache.name} {before} -> "
                        f"{state_of(ln) if kind == 'D' else ('V' if ln.valid else 'I')}")
        if done:
            self.counts["globalized"] += 1
        return ", ".join(done)

    def request_cycles(self, req, va: int, pa: int, why: str) -> int:
        """When a store waiting in ``req``'s fill-store buffer is globally
        performed: its request reaches the CM, which orders it and sends
        the interventions (``INTV`` cycles) — from then on no other core
        reads the old value (its own data arrives later, in the
        background)."""
        return req.timing.p["INTV"]

    def fill_cycles(self, req, va: int, pa: int, why: str) -> int:
        """When the line of a store waiting in the fill-store buffer is
        there (its own later loads of other bytes wait for it, and it holds
        its FSB entry until then): an upgrade ``UPG``; a miss the latency of
        whoever serves it (another core's E / M copy ``C2C``, the shared L2
        / L3, memory) plus the rest of the line's beats."""
        p = req.timing.p
        if why == "upgrade":
            return p["UPG"]
        beats = max((req.d.line + 7) // 8, 1)
        if any(not h.line.shared for h in self._holders(req.core, pa)):
            level = "C2C"
        elif self.l2 is not None and self.l2.probe(pa, pa) is not None:
            level = "L2"
        elif self.l3 is not None and self.l3.probe(pa, pa) is not None:
            level = "L3"
        else:
            level = "MEM"
        return p[level] + (beats - 1) * p["BEAT"]

    # -- line actions ---------------------------------------------------------------

    def _slice(self, h: _Holder, pa: int, n: int) -> tuple[bytearray, int] | None:
        c = h.cache
        off = pa & c.line_mask
        if off + n > c.line:
            return None
        ln = h.line
        return bytearray(ln.data[off:off + n]), (ln.known >> off) & ((1 << n) - 1)

    def _write_back(self, h: _Holder) -> None:
        """The CM moves a Modified line to the L2 / memory."""
        c, ln = h.cache, h.line
        _write_below(self.first(), c.line_pa(ln.tag, h.s), bytes(ln.data), ln.known, False)
        ln.dirty = False
        c._seal(ln)
        c.writebacks += 1
        self.counts["writebacks"] += 1

    def _invalidate(self, h: _Holder) -> None:
        c = h.cache
        line = c.line_pa(h.line.tag, h.s)
        h.caches._invalidate(c, h.s, h.w)
        self.counts["invalidations"] += 1
        m = h.caches.m
        if m is not None and m.lr_valid and m.lr_pa is not None \
                and m.lr_pa & ~c.line_mask == line:
            m.lr_valid = False                     # the LL's line is gone

    # -- views ----------------------------------------------------------------------

    def peek_owned(self, req, pa: int) -> int | None:
        """The byte of another core's Modified copy of ``pa`` (what a
        coherent load would get), no side effects."""
        for h in self._holders(req.core, pa):
            if h.line.dirty:
                return h.line.data[pa & h.cache.line_mask]
        return None

    def copies(self, pa: int) -> list[tuple[int, str]]:
        """``(core, MESI state)`` of every data-cache copy of ``pa``."""
        out = []
        for j in sorted(self.cores):
            d = self.cores[j].d
            if d is None:
                continue
            tag = d.tag_of(pa)
            for s in d.alias_sets(pa):
                w = d.find(s, tag)
                if w is not None:
                    out.append((j, state_of(d.lines[s][w])))
        return out

    def backdoor_write(self, pa: int, value: int) -> None:
        """A debugger / loader write: every core's copies, the shared
        levels' (once)."""
        for caches in self.cores.values():
            caches.own_backdoor_write(pa, value, [c for c in (caches.i, caches.d) if c])
        if self.cores:
            next(iter(self.cores.values())).own_backdoor_write(pa, value, self.secondary,
                                                              wbb=False)

    # -- the L2's address regions -----------------------------------------------------

    def l2_policy(self, pa: int) -> str:
        """How the L2 treats ``pa``: its address region's (GCR_BASE for the
        default region) CCA override, which applies only when the region's
        target is memory (Table 11.38: CM2_TARGET = 1; GCR_BASE's
        CM_DEFAULT_TARGET = 0): "wb", "wt" or "bypass"."""
        attr, default = self._region_attr(pa)
        memory = (attr & 3) == (0 if default else 1)
        if attr & REG_CCAOVREN and memory:
            cca = (attr >> REG_CCAOVR_SHIFT) & 7
            if cca in (CCA_UC, CCA_UCA):
                return "bypass"
            if cca == CCA_WT:
                return "wt"
        return "wb"

    def _region_attr(self, pa: int) -> tuple[int, bool]:
        """``(attributes, is the default region)`` of ``pa``'s address
        region (GCR_REGn_MASK, lowest-numbered match first) or GCR_BASE."""
        gcr = self.cps.gcr
        for b in GCR_REGS_BASE:
            mask = gcr.regs[b + 8]
            if mask & REG_CMTGT and (pa & mask & 0xFFFF0000) == (gcr.regs[b] & mask & 0xFFFF0000):
                return mask, False
        return gcr.base_attr, True

    @property
    def memory_sync0(self) -> bool:
        """GCR_CONTROL.SYNCCTL (MD01025 Table 11.23): 1 = SYNC 0 is a memory
        sync (it waits until memory has every write), 0 (reset) = an
        intervention sync (the coherent requests are done: no wait for
        memory)."""
        return bool(self.cps.gcr.regs[0x10] & (1 << 16))

    def drop_l2(self, pa: int) -> bool:
        """GCR_REGn_MASK.DROP_L2 (Table 11.38): "the CM2 drops the L2
        CacheOp write after it has been serialized" for the region."""
        return bool(self._region_attr(pa)[0] & REG_DROPL2)

    def l2_only_sync(self, pa: int) -> bool:
        """An uncached write to the L2-only SYNC region (CM2): a barrier
        in the L2, not a write."""
        base = self.l2_sync_base
        return bool(base & 1) and (pa & 0xFFFFF000) == (base & 0xFFFFF000)

    # -- errors ---------------------------------------------------------------------

    def error(self, kind: int, info: int, addr: int) -> bool:
        """Record a CM error (GCR_ERROR_CAUSE / ADDR, bits 2:0 = 0; a second
        one before software clears the cause goes to GCR_ERROR_MULT.ERR2ND).
        Returns True when the request gets an ERROR response (it needs a
        response and its GCR_ERROR_MASK bit is 0, Table 11.11: the core
        takes a bus error); with the bit set the CM interrupts instead."""
        if self.error_cause >> 27 == 0:
            self.error_cause = (kind << 27) | (info & 0x7FFFFFF)
            self.error_addr = addr & M32 & ~7
        elif not self.error_mult & 0x1F:
            self.error_mult = kind & 0x1F
        self.error_changed()
        return kind in ERR_NEEDS_RESPONSE and not self.error_mask >> kind & 1

    @property
    def error_irq(self) -> bool:
        """The CM's error interrupt: a recorded error whose type is enabled
        in GCR_ERROR_MASK."""
        kind = self.error_cause >> 27
        return bool(kind) and bool(self.error_mask >> kind & 1)

    def error_changed(self) -> None:
        """Drive the error interrupt (GIC source :data:`emips.gic.SRC_CM_ERROR`:
        where the system wires it is the integrator's choice)."""
        from .gic import SRC_CM_ERROR
        gic = getattr(self.cps, "gic", None)
        if gic is not None:
            gic.set_input(SRC_CM_ERROR, self.error_irq)

    def register_access_error(self, m, pa: int, cca: int, write: bool, target: int) -> bool:
        """A cached access to the GCR / GIC / CPC: its line fill (a store
        miss too: write-allocate) is a read of length > 1 — GC_RD_ERR (2),
        or COH_RD_ERR (4) for a coherent one (Table 11.11).  Returns True
        when the request gets an ERROR response."""
        c = m.caches
        if c is not None and c.coherent(cca):
            kind = ERR_COH_RD
            cmd = "share" if not write and (cca == 5 or getattr(m, "_linked", False)) \
                else "own"
        else:
            kind, cmd = ERR_GC_RD, "read"
        info = (cca << 15) | (target << 12) | (MCMD[cmd] << 7) | (m.cpu_num & 7)
        return self.error(kind, info, pa)

    def _check(self, req, pa: int, holders: list[_Holder], intent: str) -> None:
        """Inconsistent copies (two owners, an owner beside a sharer —
        software that changed attributes or domains without flushing):
        an intervention error (17 write / 18 read)."""
        owners = [h for h in holders if not h.line.shared]
        mine = None
        d = req.d
        if d is not None:
            tag = d.tag_of(pa)
            for s in d.alias_sets(pa):
                w = d.find(s, tag)
                if w is not None:
                    mine = state_of(d.lines[s][w])
        bad = len(owners) > 1 or (owners and len(owners) < len(holders))
        bad = bad or (intent == "upgrade" and owners)
        if not bad:
            return
        states = {h.core: state_of(h.line) for h in holders}
        if mine is not None:
            states[req.core] = mine
        # Tables 11.14-11.16: per core 3 bits from bit 9 — its state (11:10 ...)
        # and SResp (9: it supplied the data: the lowest-numbered owner);
        # bit 8: the request is a Store Conditional's
        supplier = min((h.core for h in owners), default=None)
        info = 0
        for k in range(4):
            info |= (_STATE_CODE[states.get(k, "I")] | (k == supplier)) << (9 + 3 * k)
        info |= int(getattr(req, "_atomic", False)) << 8
        info |= (MCMD.get(intent, 0) << 3) | (req.core & 7)
        # a read (CohRead / CohReadOwn: data back) needs a response: INTVN_RD;
        # an upgrade only invalidates: INTVN_WR
        kind = ERR_INTVN_WR if intent == "upgrade" else ERR_INTVN_RD
        if self.error(kind, info, pa) and req.m is not None:
            req.m.bus_error_pending = True

    def describe_error(self) -> str:
        kind = self.error_cause >> 27
        if not kind:
            return "no CM error"
        text = f"CM error {ERROR_NAMES.get(kind, kind)} at 0x{self.error_addr:08x}"
        if self.error_mult & 0x1F:
            text += f" (then {ERROR_NAMES.get(self.error_mult & 0x1F, self.error_mult & 0x1F)})"
        return text

    def describe(self) -> list[str]:
        """The CM for the debugger: the domain, the counters, the error."""
        out = []
        for j in sorted(self.cores):
            coh = self._coherence(j)
            mode = "coherent" if coh >> j & 1 else "not coherent"
            frm = [str(i) for i in sorted(self.cores) if i != j and coh >> i & 1]
            out.append(f"core {j}: GCR_COHERENCE 0x{coh:02x}, {mode}"
                       + (f", interventions from core(s) {', '.join(frm)}" if frm else ""))
        c = self.counts
        out.append(", ".join(f"{k} {v}" for k, v in c.items()))
        if self.l2 is not None:
            out.append(f"shared {self.l2.name}: {self.l2.geometry.describe()}"
                       + (f", {self.l3.name}: {self.l3.geometry.describe()}" if self.l3 else ""))
        out.append(self.describe_error())
        return out
