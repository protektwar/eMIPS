"""Memory timing of the cache hierarchy (PLAN M9 phase 4).

The caches (:mod:`emips.cache`) decide *what* an access does; this module
decides *how long* it takes, for the cycle-level pipeline
(:mod:`emips.pipeline`).  It never changes a result: the instruction-level
:class:`~emips.sim.Machine` ignores it (one cycle per instruction) and the
pipeline turns it on.

Costs, in CPU cycles beyond the one-cycle stage (:data:`MEMORY_TIMING`,
every one configurable):

======== ===================================================== ===========
``L2``   an L1 refill served by the L2, to the first doubleword  12
``L3``   ... by the L3                                           25
``MEM``  ... by memory; also an uncached read                    50
``BEAT`` each further doubleword of a line (64-bit bus)           1
``BYPASS`` added to uncached reads when an L2 is fitted           4
``WRITE`` the system taking one write (a write-buffer entry, or   10
         a store without write buffer)
``UTLB`` a micro-TLB (ITLB / DTLB) miss refilled from the JTLB     3
         (with the TLB MMU, :mod:`emips.tlb`; also without caches)
``C2C``  a coherent L1 miss served by another core's cache       14
         (cache to cache, :mod:`emips.coherence`), first dword
``UPG``  a store to a Shared line: the upgrade request that       8
         invalidates the other copies
``INTV`` a coherent request reaching the CM and its interventions 4
         the other cores (when a store waiting in the fill-store
         buffer becomes visible to them)
======== ===================================================== ===========

Sources: MD00343 2.2.4 ("the total miss penalty could range from a
minimum of 10-12 cycles for an L2 hit to 50 or more for an access to main
memory"), MD00355 3.4.1 (the L2 delivers "a burst of four 64-bit
doublewords ... the rest of the burst follows on consecutive clocks";
"it also adds 4 L2 cycles to the latency of all transactions which bypass
the L2").  ``L3`` and ``WRITE`` are eMIPS's assumptions (no 24K figure),
so are ``C2C``, ``UPG`` and ``INTV`` (no 1004K / interAptiv figure is
public: an intervention is a round trip through the CM, about an L2
access).

The model:

* **Blocking** (the textbook, default): a data miss holds the access in
  MEM — the whole pipeline waits for the line; an instruction miss holds
  IF.  **Non-blocking** (``nonblocking``, the 24K, MD00343 2.3.3): loads
  and stores that miss go on in the background (up to 8 lines in the
  Fill-Store Buffer); only an instruction that needs the loaded register
  waits.  Instruction misses always block.
* Refills share the bus: their data beats come one line after another
  (a line's first doubleword arrives ``L2`` / ``L3`` / ``MEM`` cycles
  after the request, the others ``BEAT`` apart).
* The **write buffer** has 4 entries; an entry leaving it occupies the
  system for ``WRITE`` + beats cycles, one after the other, and keeps
  its slot until then.  A store finding all 4 slots taken waits for the
  oldest ("write buffer full").  Without a write buffer every write to
  the system (an eviction, a write-through or uncached store) waits for
  ``WRITE`` + beats.  A read that sends an entry first waits for it.
* ``SYNC`` waits until every write has been taken and every fill done.
* ``CACHE`` operations cost what MD00355 Table 3.5 gives (Hit Invalidate
  3, Hit Writeback 4 / dirty 8, Hit Writeback Invalidate 5 / dirty 8,
  Index Store Tag 3, Fetch and Lock 3 / miss 7 + the fill); operations on
  the L2 / L3 travel through the write buffer like a write.
* A ``PREF`` fill is always in the background; an access to a line still
  being filled waits for it.
"""

from __future__ import annotations

#: Default costs (cycles); see the module docstring for their sources.
MEMORY_TIMING: dict[str, int] = {"L2": 12, "L3": 25, "MEM": 50, "BEAT": 1,
                                 "BYPASS": 4, "WRITE": 10, "UTLB": 3, "C2C": 14,
                                 "UPG": 8, "INTV": 4}
#: What each key means (for the views and the Settings dialog).
TIMING_NAMES: dict[str, str] = {
    "L2": "L1 miss served by the L2 (first doubleword)",
    "L3": "L1 miss served by the L3 (first doubleword)",
    "MEM": "L1 miss served by memory; uncached read",
    "BEAT": "each further doubleword of a line",
    "BYPASS": "extra for an uncached read past an L2",
    "WRITE": "the system taking one write",
    "UTLB": "micro-TLB miss: refill from the JTLB",
    "C2C": "coherent miss served by another core's cache",
    "UPG": "store to a Shared line: invalidating the others",
    "INTV": "coherent request: until the other cores see it",
}
#: Write buffer entries (MD00343 8.4) and Fill-Store Buffer lines (2.3.3).
WBB_SLOTS = 4
FSB_LINES = 8
#: CACHE operation costs, MD00355 Table 3.5 (cycles).
CACHEOP_COST = {"hit-inv": 3, "hit-wb-clean": 4, "hit-wb-dirty": 8,
                "hit-wbi-clean": 5, "hit-wbi-dirty": 8, "index-store-tag": 3,
                "lock-hit": 3, "lock-miss": 7}
#: Levels a read can be served by, in order of latency.
_LEVELS = ("", "C2C", "L2", "L3", "MEM")
_SOURCE = {"C2C": "another core's cache", "L2": "the L2", "L3": "the L3", "MEM": "memory"}


def _beats(n: int) -> int:
    """Doublewords of an ``n``-byte transfer on the 64-bit bus."""
    return max((n + 7) // 8, 1)


class Bus:
    """When the system bus is free again for reads / writes — one per core,
    or one for every core of a CPS (the Coherence Manager serializes their
    requests; ``shared``: kept in the system's clock, see
    :attr:`Timing.offset`)."""

    def __init__(self, shared: bool = False) -> None:
        self.read_free = 0
        self.write_free = 0
        self.shared = shared


class Timing:
    """The memory timing of one :class:`~emips.cache.Caches` (its
    ``timing``).  The pipeline sets :attr:`active`, the parameters and
    :attr:`nonblocking`, and after every fetch / execute calls
    :meth:`take`."""

    def __init__(self, caches) -> None:
        self.c = caches
        #: off for the instruction-level machine; the pipeline turns it on
        self.active = False
        self.p: dict[str, int] = dict(MEMORY_TIMING)
        self.nonblocking = False
        #: the system bus (emips.coherence shares one between the cores)
        self.bus = Bus()
        self.reset_state()

    @property
    def offset(self) -> int:
        """Core cycles -> the system's clock (a core of a CPS started late
        counts its cycles from its own reset)."""
        m = self.c.m
        return getattr(m, "clock_offset", 0) if m is not None else 0

    @property
    def read_free(self) -> int:
        """Core cycle at which the bus has returned the last refill."""
        return self.bus.read_free - self.offset

    @read_free.setter
    def read_free(self, value: int) -> None:
        if self.bus.shared:                     # never back in time for the others
            self.bus.read_free = max(self.bus.read_free, value + self.offset)
        else:
            self.bus.read_free = value

    @property
    def write_free(self) -> int:
        """Core cycle at which the system has taken the last write."""
        return self.bus.write_free - self.offset

    @write_free.setter
    def write_free(self, value: int) -> None:
        if self.bus.shared:
            self.bus.write_free = max(self.bus.write_free, value + self.offset)
        else:
            self.bus.write_free = value

    def configure(self, params: dict[str, int] | None = None, *,
                  nonblocking: bool = False) -> None:
        self.p = dict(MEMORY_TIMING, **(params or {}))
        self.nonblocking = nonblocking
        self.active = True
        self.reset_state()

    def reset_state(self) -> None:
        self.read_free = 0                      # the bus: free (see Bus)
        self.write_free = 0
        #: completion cycles of the writes in flight (write buffer slots)
        self.writes: list[int] = []
        #: line PA -> cycle its background fill completes (non-blocking, PREF)
        self.pending: dict[int, int] = {}
        #: totals for the views
        self.stall_cycles: dict[str, int] = {}
        self.taken_runs: list[tuple[str, int]] = []
        #: data reads on their way: [(cycle the data arrives, level)] —
        #: the 24K's "D$ miss cycles" / "L2 miss cycles" events
        self.outstanding: list[tuple[int, str]] = []
        #: cacheable data reads and their total latency (miss detection to
        #: data; the 24K's "request latency to read response" events)
        self.read_requests = 0
        self.read_latency = 0
        #: uncached loads block even when loads are non-blocking (the
        #: 24K's Config7.ULB)
        self.block_uncached = False
        self._clear()

    def _clear(self) -> None:
        self.delay = 0
        self.load_ready: int | None = None
        self.notes: list[str] = []
        #: the waits of the access, in order: [(stall kind, cycles)]
        self.runs: list[tuple[str, int]] = []
        self.kind = "load"
        self.level = 0
        self.after_write = 0

    # -- the clock -------------------------------------------------------------

    @property
    def now(self) -> int:
        return self.c.m.cycles

    def cursor(self) -> int:
        """When the current access can go on (now + its waits so far)."""
        return self.now + self.delay

    def take(self) -> tuple[int, int | None, list[str]]:
        """``(cycles the stage must hold, cycles until a non-blocking
        load's data arrives or None, notes)`` of the accesses since the
        last call; starts afresh."""
        now = self.now
        out = (self.delay, None if self.load_ready is None else self.load_ready - now,
               self.notes)
        #: the waits behind the last take(): [(stall kind, cycles)]
        self.taken_runs = self.runs
        if len(self.pending) > 64:
            self.pending = {k: v for k, v in self.pending.items() if v > now}
        self.writes = [d for d in self.writes if d > now]
        self._clear()
        return out

    def _wait_until(self, ready: int, why: str, key: str) -> None:
        extra = ready - self.cursor()
        if extra > 0:
            self.delay += extra
            self.runs.append((key, extra))
            self.stall_cycles[key] = self.stall_cycles.get(key, 0) + extra
            self.notes.append(f"{why}: {extra} cycle{'s' if extra != 1 else ''}")

    # -- the kind of access (set by the cache entry points) -----------------------

    def begin(self, kind: str) -> None:
        """``load`` / ``store`` / ``fetch`` / ``pref`` / ``cacheop`` / ``sync``."""
        if self.active:
            self.kind = kind
            self.level = 0

    def served(self, level: str) -> None:
        """A read was served by this level (the deepest one counts)."""
        if self.active:
            self.level = max(self.level, _LEVELS.index(level))

    # -- reads -------------------------------------------------------------------

    def _fsb(self) -> None:
        """Non-blocking: a new miss needs a free Fill-Store Buffer line."""
        busy = sorted(r for r in self.pending.values() if r > self.cursor())
        if len(busy) >= FSB_LINES:
            self._wait_until(busy[len(busy) - FSB_LINES], "fill-store buffer full", "fsb")

    def _deliver(self, ready: int, what: str, line: int | None,
                 key: str | None = None) -> None:
        """The data of a read arrives at ``ready``: wait for it, or (in
        the background) remember it."""
        background = self.kind == "pref" or (
            self.nonblocking and self.kind in ("load", "store")
            and not (key == "uncached" and self.block_uncached))
        if background:
            if line is not None:
                self.pending[line] = max(self.pending.get(line, 0), ready)
            if self.kind == "load":
                self.load_ready = max(self.load_ready or 0, ready)
                self.notes.append(f"{what}, in the background: data in "
                                  f"{ready - self.now} cycles")
            return
        key = key or {"fetch": "I-miss", "cacheop": "cacheop"}.get(self.kind, "D-miss")
        self._wait_until(ready, what, key)

    def refill(self, pa: int, n: int, cache: str = "D") -> None:
        """An L1 line of ``n`` bytes was read from below (after
        :meth:`served` named the level)."""
        if not self.active:
            return
        level = _LEVELS[self.level] or "MEM"
        if self.nonblocking and self.kind in ("load", "store") or self.kind == "pref":
            self._fsb()
        beats = _beats(n)
        start = max(self.cursor(), self.after_write)
        ready = max(start + self.p[level] + (beats - 1) * self.p["BEAT"],
                    self.read_free + beats * self.p["BEAT"])
        self.read_free = ready
        what = (f"{'I' if cache == 'I' else 'D'}-cache miss "
                f"({'prefetch' if self.kind == 'pref' else self.kind}): "
                f"line from {_SOURCE[level]}")
        if cache == "D":
            self.read_requests += 1
            self.read_latency += ready - self.now
            self._note_outstanding(ready, level)
        self._deliver(ready, what, pa if cache == "D" else None)

    def _note_outstanding(self, ready: int, level: str) -> None:
        now = self.now
        self.outstanding = [x for x in self.outstanding if x[0] > now]
        self.outstanding.append((ready, level))

    def uncached_read(self, n: int) -> None:
        """An uncached read (load or fetch) of ``n`` bytes, past the caches."""
        if not self.active:
            return
        lat = self.p["MEM"] + (self.p["BYPASS"] if self.c.l2 is not None else 0)
        start = max(self.cursor(), self.after_write)
        ready = max(start + lat, self.read_free + self.p["BEAT"])
        self.read_free = ready
        what = "uncached fetch" if self.kind == "fetch" else "uncached read"
        if self.kind != "fetch":
            self._note_outstanding(ready, "MEM")
        if self.nonblocking and self.kind == "load":
            self._fsb()
        self._deliver(ready, what, None, "uncached")

    def line_access(self, pa: int) -> None:
        """An access to an L1 data line that may still be in a background
        fill: it waits for it (a store merges into the fill: no wait)."""
        if not self.active or not self.pending:
            return
        ready = self.pending.get(pa)
        if ready is None or ready <= self.cursor() or self.kind == "store":
            return
        if self.nonblocking and self.kind == "load":
            self.load_ready = max(self.load_ready or 0, ready)
            return
        self._wait_until(ready, "line still being filled", "D-miss")

    # -- writes ------------------------------------------------------------------

    def _write_time(self, n: int) -> int:
        return self.p["WRITE"] + (_beats(n) - 1) * self.p["BEAT"]

    def reserve(self, what: str) -> None:
        """A write-buffer entry is needed (opened or sent at once)."""
        if not self.active:
            return
        wbb = self.c.wbb
        open_ = sum(e is not None for e in (wbb.wt, wbb.uca)) if wbb is not None else 0
        busy = sorted(d for d in self.writes if d > self.cursor())
        if open_ + len(busy) >= WBB_SLOTS:
            self._wait_until(busy[open_ + len(busy) - WBB_SLOTS], f"write buffer full ({what})",
                             "write-buffer")

    def issue(self, n: int) -> None:
        """An entry of ``n`` bytes leaves the write buffer for the system."""
        if not self.active:
            return
        start = max(self.cursor(), self.write_free)
        done = start + self._write_time(n)
        self.write_free = done
        self.writes.append(done)

    def ordered_read(self) -> None:
        """A read sent a matching write first: it waits for that write."""
        if self.active:
            self.after_write = max(self.after_write, self.write_free)

    def direct_write(self, n: int, what: str, key: str = "write") -> None:
        """No write buffer: the store waits until the system takes it
        (``key``: the stall kind, "uncached" for an uncached store)."""
        if not self.active:
            return
        start = max(self.cursor(), self.write_free)
        done = start + self._write_time(n)
        self.write_free = done
        self._wait_until(done, f"{what}, no write buffer", key)

    # -- barriers and maintenance ---------------------------------------------------

    def sync(self, memory: bool = True) -> None:
        """SYNC: every write taken, every fill done; ``memory = False`` (a
        CM2 intervention sync, GCR_CONTROL.SYNCCTL = 0): only the fills —
        the coherent requests — not the writes to memory."""
        if not self.active:
            return
        last = max([self.write_free if memory else 0] + list(self.pending.values()))
        self._wait_until(last, "sync: waiting for the writes and fills" if memory else
                         "sync (intervention): waiting for the coherent requests", "sync")

    def coherence(self, what: str) -> None:
        """A coherence request with no data (an upgrade): ``UPG`` cycles."""
        if self.active:
            self._deliver(self.cursor() + self.p["UPG"], what, None, "coherence")

    def fixed(self, cycles: int, what: str) -> None:
        """A CACHE operation's own cost (MD00355 Table 3.5)."""
        if self.active and cycles:
            self.delay += cycles
            self.runs.append(("cacheop", cycles))
            self.stall_cycles["cacheop"] = self.stall_cycles.get("cacheop", 0) + cycles
            self.notes.append(f"{what}: {cycles} cycles")
