"""The rest of the P8700's memory system (LF15): the CM's L2 prefetcher,
the cores' L1 data prefetchers, the DSPRAM and the MMIO regions.

Sources: MD01502 rev. 1.84 — 5.6 / 5.14.1.23-24 (GCR_L2_PFT_CONTROL / _B),
5.10 / 5.14.1.33-49 (MMIO regions), 7.3.7 (mipsdsprambase), 7.3.12-15
(mipsconfig8-11: the L1 prefetch trackers); the P6600's MD01138 6.3.2 for
how the CM's L2 prefetcher behaves (the P8700's manual only lists its
registers); OpenSBI's P8700 setup (``platform/generic/mips``) for
what the boot firmware programs; the 24K's MD00343 for the DSPRAM's rules
(the P8700 documents only its base register).  Everything the documents
leave open is assumed and listed in ``sim.DEVIATIONS["P8700 memory
system"]`` (``LF.md`` LF15).

* :class:`L2Prefetcher` (the CM's): watches the L2 lookups of the enabled
  ports (PORT_ID: cores, then IOCUs; instruction fetches only with CEN);
  a miss starts a tracker (one stream a page), a second miss on the next or
  the previous line gives it a direction, each further miss of the stream
  widens its window (up to :attr:`L2Prefetcher.WINDOW` lines) and every
  demand on the stream prefetches the window's lines ahead into the L2 —
  never past the OS page PAGE_MASK gives (MD01138 6.3.2).  A prefetch read
  fills the L2 in the background: it holds the bus for its beats, nobody
  waits for it.
* :class:`L1Prefetcher` (a core's): eight trackers (2 groups x 4, the
  registers of mipsconfig8-10, PFEN bit k = tracker k) learn the stride of
  the data cache's demand misses in a 4 KB page; with confidence 2 a
  tracker prefetches the next line of the stride into the L1 (as PREF 0)
  and counts event 77 + k.
* :class:`Dspram`: a core's data scratch-pad RAM at mipsdsprambase
  (aligned to its size): its loads and stores never go to the caches, are
  not coherent and no other core or the IOCU sees them; fetches go to
  memory.
* :func:`mmio_region`: the GCR's MMIO region an address falls in.
"""

from __future__ import annotations

from typing import Any

M64 = (1 << 64) - 1

# -- the L1 prefetch trackers (mipsconfig8-11) -------------------------------------

#: the registers of a tracker (mipsconfig8.PTRS 0-4) and their widths (7.3.13)
MISMATCH, CONFIDENCE, AGE, STRIDE, PRESCALER = range(5)
WIDTHS = (3, 4, 6, 15, 6)
TRACKER_NAMES = ("mismatch", "confidence", "age", "learn stride", "age prescaler")


class Tracker:
    """One L1 prefetch tracker: its five registers and what it follows."""

    __slots__ = ("regs", "page", "last", "valid", "tick", "issued")

    def __init__(self) -> None:
        self.regs = [0] * 5
        self.page = self.last = 0
        self.valid = False
        self.tick = 0
        self.issued = 0

    @property
    def stride(self) -> int:
        s = self.regs[STRIDE]
        return s - (1 << 15) if s >> 14 else s


class L1Prefetcher:
    """A core's L1 data prefetcher (MD01502 7.3.12-15; how it learns:
    assumed, LF.md LF15)."""

    GROUPS, ENTRIES = 2, 4
    #: the confidence from which a tracker prefetches
    CONFIDENT = 2

    def __init__(self) -> None:
        self.trackers = [Tracker() for _ in range(self.GROUPS * self.ENTRIES)]
        #: mipsconfig11.PFEN: bit k enables tracker k
        self.pfen = 0
        #: line PAs prefetched and not used yet (a demand hit on one trains)
        self.prefetched: dict[int, int] = {}
        self.issued = 0
        self.useful = 0

    def read(self, k: int, reg: int) -> int:
        return self.trackers[k].regs[reg] if 0 <= reg < 5 else 0

    def write(self, k: int, reg: int, value: int) -> None:
        if 0 <= reg < 5:
            self.trackers[k].regs[reg] = value & ((1 << WIDTHS[reg]) - 1)

    def observe(self, caches, va: int, pa: int, cca: int, miss: bool) -> None:
        """A demand load / store of the data cache at ``pa`` (``miss``: it
        missed the L1)."""
        if not self.pfen:
            return
        line = caches.d.line
        lp = pa // line
        if not miss:
            k = self.prefetched.pop(lp, None)
            if k is None:
                return
            self.useful += 1
        page = pa >> 12
        found = None
        for k, t in enumerate(self.trackers):
            if not (self.pfen >> k & 1 and t.valid):
                continue
            t.tick += 1                          # (aging: AGE every PRESCALER + 1 events)
            if t.tick > t.regs[PRESCALER]:
                t.tick = 0
                t.regs[AGE] = min(63, t.regs[AGE] + 1)
                if t.regs[AGE] == 63:            # (aged out)
                    t.valid = False
                    continue
            if found is None and t.page == page:
                found = k
        if found is None:
            self._allocate(page, pa)
            return
        t = self.trackers[found]
        s = pa - t.last
        if s == 0:
            return
        t.regs[AGE] = 0
        if s == t.stride:
            t.regs[CONFIDENCE] = min(15, t.regs[CONFIDENCE] + 1)
            t.regs[MISMATCH] = 0
        else:
            t.regs[MISMATCH] = min(7, t.regs[MISMATCH] + 1)
            t.regs[CONFIDENCE] = 0
            if -(1 << 14) <= s < 1 << 14:        # (a stride the 15-bit counter holds)
                t.regs[STRIDE] = s & 0x7FFF
        t.last = pa
        if t.regs[CONFIDENCE] < self.CONFIDENT:
            return
        stride = t.stride
        step = stride if abs(stride) >= line else (line if stride > 0 else -line)
        target = pa + step
        if target >> 12 != page or target < 0:
            return                               # (never past the page: no translation)
        tl = target // line
        d = caches.d
        if tl in self.prefetched or d.find(d.set_of(va + step), d.tag_of(target)) is not None:
            return
        caches.prefetch(0, va + step, target, cca)
        self.prefetched[tl] = found
        if len(self.prefetched) > 64:
            del self.prefetched[next(iter(self.prefetched))]
        t.issued += 1
        self.issued += 1
        m = caches.m
        p = getattr(m, "p8700", None)
        if p is not None and p.counters.armed:
            p.counters.add(77 + found)           # (L1 Prefetch from Prefetch tracker k)

    def _allocate(self, page: int, pa: int) -> None:
        free = [k for k, t in enumerate(self.trackers) if self.pfen >> k & 1 and not t.valid]
        if free:
            k = free[0]
        else:
            live = [k for k in range(len(self.trackers)) if self.pfen >> k & 1]
            if not live:
                return
            k = max(live, key=lambda j: self.trackers[j].regs[AGE])
        t = self.trackers[k]
        t.valid, t.page, t.last, t.tick = True, page, pa, 0
        t.regs[MISMATCH] = t.regs[CONFIDENCE] = t.regs[AGE] = t.regs[STRIDE] = 0

    def lines(self) -> list[str]:
        out = [f"L1 prefetcher: PFEN 0x{self.pfen:02x}, {self.issued} prefetched, "
               f"{self.useful} used"]
        for k, t in enumerate(self.trackers):
            if not (self.pfen >> k & 1):
                state = "off"
            elif not t.valid:
                state = "free"
            else:
                state = (f"page 0x{t.page << 12:x}, last 0x{t.last:x}, stride {t.stride}, "
                         f"confidence {t.regs[CONFIDENCE]}, mismatch {t.regs[MISMATCH]}, "
                         f"age {t.regs[AGE]}/{t.regs[PRESCALER] + 1}")
            out.append(f"  tracker {k} (group {k // 4}, entry {k % 4}): {state}"
                       + (f", {t.issued} prefetched" if t.issued else ""))
        return out


# -- the CM's L2 prefetcher -----------------------------------------------------------

class _Stream:
    __slots__ = ("page", "last", "dir", "window", "age", "issued")

    def __init__(self, page: int, line: int) -> None:
        self.page, self.last = page, line
        self.dir = self.window = self.age = self.issued = 0


class L2Prefetcher:
    """The CM's L2 prefetcher (MD01502 5.6; MD01138 6.3.2)."""

    #: NPFT (OpenSBI's Boston setup writes 0x10 there; the P6600: "up to 16
    #: streams"), the window's limit and the age a stream dies at (assumed)
    NPFT, WINDOW, AGE_OUT = 16, 8, 64

    def __init__(self, cm) -> None:
        self.cm = cm
        #: GCR_L2_PFT_CONTROL: PAGE_MASK 4 KB, PFTEN 0, NPFT ("from configuration")
        self.control = 0xFFFFF << 12 | self.NPFT
        #: GCR_L2_PFT_CONTROL_B: WRI_MODE 2, CEN 0, PORT_ID 0xFF
        self.control_b = 2 << 9 | 0xFF
        self.streams: list[_Stream] = []
        self.prefetched: set[int] = set()
        self.issued = 0
        self.useful = 0
        self._busy = False

    # -- the registers ---------------------------------------------------------------
    def get(self, which: int) -> int:
        if which == 0:
            return self.control
        idle = 0 if self.streams else 1 << 12
        return self.control_b | idle

    def put(self, which: int, value: int, mask: int) -> None:
        if which == 0:                           # (PAGE_MASK, PFTEN; NPFT read-only)
            w = 0xFFFFF << 12 | 1 << 8
            self.control = (self.control & ~(mask & w)) | (value & mask & w)
        else:                                    # (WRI_MODE, CEN, PORT_ID)
            w = 3 << 9 | 1 << 8 | 0xFF
            self.control_b = (self.control_b & ~(mask & w)) | (value & mask & w)
            if self.control_b >> 9 & 3 == 3:     # (reserved: kept as it was)
                self.control_b &= ~(1 << 9)

    @property
    def enabled(self) -> bool:
        return bool(self.control >> 8 & 1) and self.control & 0xFF > 0

    def page_bytes(self) -> int:
        mask = self.control >> 12 & 0xFFFFF
        zeros = (~mask & 0xFFFFF).bit_length()
        return 1 << (12 + zeros)

    # -- watching the L2 ---------------------------------------------------------------
    def port_of(self, req) -> int:
        core = getattr(req, "core", None)
        return core if isinstance(core, int) else self.cm.cps.cores   # (IOCU 0: after them)

    def observe(self, pa: int, miss: bool) -> None:
        """An L2 lookup of a read (SecondaryCache.observer)."""
        if self._busy or not self.enabled:
            return
        req = self.cm.current
        kind = getattr(req, "req_kind", "load")
        if not self.control_b >> self.port_of(req) & 1:
            return
        if kind == "fetch" and not self.control_b >> 8 & 1:
            return
        self._see(pa, miss)

    def write_invalidate(self, pa: int) -> None:
        """An IOCU's coherent full-line write (WRI_MODE 1 / 2: the stream
        trains the prefetcher as a miss would; 0: it does not)."""
        if self.enabled and self.control_b >> 9 & 3 and \
                self.control_b >> self.cm.cps.cores & 1:
            self._see(pa, True)

    def _see(self, pa: int, miss: bool) -> None:
        l2 = self.cm.l2
        line = pa // l2.line
        if not miss:
            if line not in self.prefetched:
                return
            self.prefetched.discard(line)
            self.useful += 1
        size = self.page_bytes()
        page = pa // size
        hit = None
        for s in self.streams:
            if s.page == page:
                hit = s
            else:
                s.age += 1
        self.streams = [s for s in self.streams if s.age < self.AGE_OUT]
        if hit is None:
            if miss:
                if len(self.streams) >= self.control & 0xFF:
                    self.streams.remove(max(self.streams, key=lambda s: s.age))
                self.streams.append(_Stream(page, line))
            return
        hit.age = 0
        d = line - hit.last
        if d in (1, -1) and hit.dir in (0, d):
            hit.dir = d
            if miss:                             # (the window grows until it hits)
                hit.window = min(self.WINDOW, hit.window + 1)
            hit.last = line
            per_page = size // l2.line
            for k in range(1, hit.window + 1):
                target = line + d * k
                if target // per_page != page:
                    break
                self._fill(target, hit)
        elif d:
            hit.last, hit.dir, hit.window = line, 0, 0

    def _fill(self, line: int, stream: _Stream) -> None:
        l2 = self.cm.l2
        pa = line * l2.line
        if line in self.prefetched or l2.probe(pa, pa) is not None:
            return
        req = self.cm.current
        t = getattr(req, "timing", None)
        active = t is not None and t.active
        if active:
            t.active = False                     # (no stall, not the demand's level)
        self._busy = True
        try:
            l2.lookup(pa, pa, l2.below, kind="pref")
        except Exception:                        # (a bus error: nothing prefetched)
            return
        finally:
            self._busy = False
            if active:
                t.active = True
        if active:
            t.background(l2.line)
        self.prefetched.add(line)
        if len(self.prefetched) > 1024:
            self.prefetched.pop()
        stream.issued += 1
        self.issued += 1

    def lines(self) -> list[str]:
        on = "on" if self.enabled else "off"
        cb = self.control_b
        out = [f"L2 prefetcher: {on} (PFTEN {self.control >> 8 & 1}, NPFT "
               f"{self.control & 0xFF}, pages {self.page_bytes() // 1024} KB, ports "
               f"0x{cb & 0xFF:02x}, code {'on' if cb >> 8 & 1 else 'off'}, WRI_MODE "
               f"{cb >> 9 & 3}): {self.issued} lines prefetched, {self.useful} used"]
        for k, s in enumerate(self.streams):
            way = {1: "up", -1: "down"}.get(s.dir, "learning")
            out.append(f"  stream {k}: line 0x{s.last * self.cm.l2.line:x}, {way}, window "
                       f"{s.window}, {s.issued} prefetched")
        return out


# -- the DSPRAM ----------------------------------------------------------------------

class Dspram:
    """A core's data scratch-pad RAM (MD01502 7.3.7: mipsdsprambase; the
    rest as the 24K's DSPRAM, assumed): ``size`` bytes at the base."""

    def __init__(self, size: int, pamask: int = M64) -> None:
        if size <= 0 or size & (size - 1):
            raise ValueError(f"a DSPRAM of {size} bytes: a power of two is needed")
        self.size = size
        self.pamask = pamask
        self.base = self.raw = 0
        self.data = bytearray(size)
        self.reads = self.writes = 0

    def set_base(self, value: int) -> None:
        self.raw = value
        self.base = value & self.pamask & ~(self.size - 1)

    def covers(self, pa: int) -> bool:
        return self.base <= pa < self.base + self.size

    def load(self, pas: list[int]) -> bytes:
        self.reads += 1
        b = self.base
        return bytes(self.data[pa - b] for pa in pas)

    def store(self, pas: list[int], data: bytes) -> None:
        self.writes += 1
        b = self.base
        for pa, v in zip(pas, data):
            self.data[pa - b] = v

    def lines(self) -> list[str]:
        return [f"DSPRAM: {self.size // 1024} KB at 0x{self.base:x}-"
                f"0x{self.base + self.size - 1:x}, {self.reads} reads, {self.writes} writes"]


# -- MMIO regions ----------------------------------------------------------------------

#: GCR_MMIOn_BOTTOM fields (5.14.1.34)
MMIO_EN, MMIO_DIS_LIMIT, MMIO_NONCOH = 1, 1 << 1, 1 << 6
PORTS = {0: "MEM", 8: "AUX0", 9: "AUX1", 10: "AUX2", 11: "AUX3"}
CCA_MATCH = ("any", "UC", "UCA", "UC or UCA")


def mmio_regions(regs: dict[int, int], count: int = 8) -> list[dict[str, Any]]:
    """The enabled MMIO regions (5.10.2): number, bottom / top (bits 47:16),
    CCA qualification, port, force non-coherent, request limit off."""
    out = []
    for k in range(count):
        b = regs.get(0x0700 + 0x10 * k, 0)
        t = regs.get(0x0708 + 0x10 * k, 0)
        if not b & MMIO_EN:
            continue
        lo = (b >> 16 & 0xFFFFFFFF) << 16
        hi = (t >> 16 & 0xFFFFFFFF) << 16 | 0xFFFF
        out.append({"n": k, "bottom": lo, "top": hi, "cca": b >> 8 & 3,
                    "port": b >> 2 & 0xF, "noncoh": bool(b & MMIO_NONCOH),
                    "no_limit": bool(b & MMIO_DIS_LIMIT)})
    return out


def mmio_region(regs: dict[int, int], pa: int, uncached: bool | None = None) -> dict | None:
    """The lowest-numbered enabled region ``pa`` falls in (5.10.2);
    ``uncached``: the request's CCA is UC / UCA (None: any request — a
    region qualified by CCA then needs it uncached)."""
    for r in mmio_regions(regs):
        if r["bottom"] <= pa <= r["top"]:
            if r["cca"] and not uncached:
                continue
            return r
    return None
