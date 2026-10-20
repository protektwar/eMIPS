"""The 34K's Inter-Thread Communication storage: gating storage (PLAN M19).

Sources: MD00378 rev. 1.12 (Vol. IV-f) 9.1 (gating storage: a blocked load
/ store can be aborted and re-issued; GSI + DT -> a GS Scheduler Thread
exception) and Appendix A (the ITC reference model: Empty / Full, the
views); the 34K Software User's Manual MD00534 rev. 1.13 chapter 12 (the
reference ITU: 4-entry FIFO cells and single-entry semaphore cells, the
tag, Table 12.2 the views, 12.4 ITCAddressMap0 / 1) and 7.2.3 / 7.2.60
(ErrCtl.ITC: CACHE Index Load / Store Tag D reach the two maps through
DTagLo); MD00427 rev. 1.64 3.3 (the maps at CACHE addresses 0 and 8, the
AddrMask sizes).

The ITC block decodes physical addresses: once ITCAddressMap0.En is set,
"all accesses to this physical address range will go to ITC" (MD00427
3.3.1) — before the caches and the bus.  A cell is ``128 << EntryGrain``
bytes apart from the next; address bits 6:3 choose the view (Table 12.2),
bit 2 is ignored (MD00378 A.3: 32-bit cells show in either word).

An access that cannot complete raises :class:`emips.exceptions.Blocked`
(``"gs"``, the :class:`Access`): the instruction has had no effect, the TC
waits (TCStatus.RNST = 3) until :meth:`ITC.ready` says it would complete
now, then issues it again (MD00378 9.1: an aborted access is re-issued).
What eMIPS chooses where the documents say "undefined" is in
``sim.DEVIATIONS["ITC"]``.
"""

from __future__ import annotations

from dataclasses import dataclass

M32 = 0xFFFFFFFF

#: Table 12.2: the views (address bits 6:3)
BYPASS, CONTROL, EF_SYNC, EF_TRY, PV_SYNC, PV_TRY = range(6)
VIEW_NAMES = {BYPASS: "bypass", CONTROL: "control", EF_SYNC: "E/F synchronized",
              EF_TRY: "E/F try", PV_SYNC: "P/V synchronized", PV_TRY: "P/V try"}
#: the cell sizes the reference ITU can be built with (12.2)
CELL_COUNTS = (0, 1, 2, 4, 8, 16)
#: the tag (Table 12.1)
TAG_E, TAG_F, TAG_T, TAG_FIFO = 1 << 0, 1 << 1, 1 << 16, 1 << 17
FIFO_DEPTH = 4
#: ITCAddressMap0 / 1 (Tables 12.3 - 12.6)
MAP0_EN = 1 << 0
MAP0_MASK = 0xFFFFFC01
MAP1_MASK = (0x7F << 10) | 7
#: ErrCtl.ITC (MD00534 Table 7.60)
ERRCTL_ITC = 1 << 26
#: the P/V counter: 16 bits, saturating at 0xffff (12.3.5)
PV_MAX = 0xFFFF


def parse_spec(spec: str) -> tuple[int, int] | None:
    """``"none"`` -> None; ``"FIFOS:SEMAPHORES"`` -> the cell counts (0, 1,
    2, 4, 8 or 16 each, MD00534 12.2); ``"auto"`` is the caller's."""
    spec = (spec or "").strip().lower()
    if spec in ("", "none", "off", "no"):
        return None
    parts = spec.split(":")
    try:
        fifos, sems = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        raise ValueError(f"ITC spec {spec!r}: FIFOS:SEMAPHORES, e.g. 4:8") from None
    if len(parts) > 2 or fifos not in CELL_COUNTS or sems not in CELL_COUNTS or not fifos + sems:
        raise ValueError(f"ITC spec {spec!r}: 0, 1, 2, 4, 8 or 16 FIFO cells and "
                         "semaphore cells, not both 0")
    return fifos, sems


#: "auto": the ITC eMIPS builds with the MT Module
AUTO = (4, 8)


@dataclass(frozen=True)
class Access:
    """A gating storage access that blocked: what it waits for."""
    cell: int
    view: int
    store: bool

    def __str__(self) -> str:
        return (f"{'store to' if self.store else 'load from'} ITC cell {self.cell} "
                f"({VIEW_NAMES.get(self.view, 'reserved')} view)")


class Cell:
    """One ITC cell: a 4-entry FIFO or a single-entry semaphore cell."""

    def __init__(self, fifo: bool) -> None:
        self.fifo = fifo
        self.words = [0] * (FIFO_DEPTH if fifo else 1)
        #: the oldest entry, the number of entries (FIFOPtr)
        self.head = 0
        self.count = 0
        #: the tag's E / F / T bits (reset UNDEFINED: eMIPS 0)
        self.empty = False
        self.full = False
        self.trap = False

    def tag(self) -> int:
        """The control view (Table 12.1)."""
        return (((2 << 28) | TAG_FIFO | (self.count << 18)) if self.fifo else 0) \
            | (TAG_T if self.trap else 0) | (TAG_F if self.full else 0) \
            | (TAG_E if self.empty else 0)

    def write_tag(self, value: int) -> None:
        self.trap = bool(value & TAG_T)
        self.full = bool(value & TAG_F)
        self.empty = bool(value & TAG_E)
        if self.empty and self.fifo:            # writing E = 1 resets FIFOPtr
            self.head = self.count = 0

    # the data (12.3): the oldest entry is read, a store fills the next
    @property
    def oldest(self) -> int:
        return self.words[self.head]

    def pop(self) -> int:
        """An E/F load: the oldest value; the cell empties with the last."""
        v = self.words[self.head]
        if self.fifo:
            self.head = (self.head + 1) % FIFO_DEPTH
            self.count = max(0, self.count - 1)
            self.empty = self.count == 0
        else:
            self.empty = True
        self.full = False
        return v

    def push(self, value: int) -> None:
        """An E/F store: the cell fills with the last free entry."""
        if self.fifo:
            self.words[(self.head + self.count) % FIFO_DEPTH] = value
            self.count = min(FIFO_DEPTH, self.count + 1)
            self.full = self.count == FIFO_DEPTH
        else:
            self.words[0] = value
            self.full = True
        self.empty = False

    def bypass_store(self, value: int) -> None:
        """Overwrites the newest FIFO entry (12.3.1)."""
        self.words[(self.head + max(self.count, 1) - 1) % len(self.words)] = value


class ITC:
    """The ITU of one MT processor (MD00534 ch. 12): ``fifos`` FIFO cells
    at cell numbers 0..N-1, then ``semaphores`` semaphore cells."""

    def __init__(self, fifos: int, semaphores: int) -> None:
        self.cells = [Cell(True) for _ in range(fifos)] + [Cell(False) for _ in range(semaphores)]
        self.fifos = fifos
        #: ITCAddressMap0 (BaseAddress, En: reset 0) and Map1's R/W fields
        self.map0 = 0
        self.map1_rw = 0
        #: loads / stores that reached the ITC (performance events 32)
        self.loads = self.stores = 0

    # -- the address maps (12.4) -------------------------------------------

    @property
    def map1(self) -> int:
        return (len(self.cells) << 20) | self.map1_rw

    def write_map(self, n: int, value: int) -> None:
        if n == 0:
            self.map0 = value & MAP0_MASK
        else:
            self.map1_rw = value & MAP1_MASK

    @property
    def enabled(self) -> bool:
        return bool(self.map0 & MAP0_EN)

    @property
    def span(self) -> int:
        """The block's size - 1 (AddrMask bits 16:10 drop BaseAddress bits)."""
        return (((self.map1_rw >> 10) & 0x7F) << 10) | 0x3FF

    def base(self) -> int:
        return self.map0 & ~self.span & 0xFFFFFC00

    def covers(self, pa: int) -> bool:
        return self.enabled and (pa & ~self.span & M32) == self.base()

    def decode(self, pa: int) -> tuple[int, int] | None:
        """(cell, view) at physical ``pa`` inside the block, None where no
        cell is (the stride's rest, cells above NumEntries)."""
        off = pa & self.span
        grain = 7 + (self.map1_rw & 7)
        cell, inside = off >> grain, off & ((1 << grain) - 1)
        if inside >= 128 or cell >= len(self.cells):
            return None
        return cell, (inside >> 3) & 0xF

    # -- accesses --------------------------------------------------------------

    def _gated(self, m, cell: Cell, view: int, store: bool) -> None:
        """The checks of a data view: the T bit -> a Gating Storage Thread
        exception (12.3.3-12.3.6); a sync access that would block -> with
        VPEControl.GSI and TCStatus.DT a GS Scheduler exception, else
        Blocked (MD00378 9.1)."""
        from . import mt as MT
        from .exceptions import Blocked
        if cell.trap:
            raise MT._thread(m, MT.GATING, f"the T bit of ITC cell {self.cells.index(cell)}")
        if self.would_block(cell, view, store):
            p = m.mt
            tc = p.current[m.vpe]
            if p.vpecontrol[m.vpe] & MT.GSI and tc.dt:
                raise MT._thread(m, MT.GS_SCHED, "VPEControl.GSI and TCStatus.DT are set")
            raise Blocked("gs", Access(self.cells.index(cell), view, store))

    @staticmethod
    def would_block(cell: Cell, view: int, store: bool) -> bool:
        if view == EF_SYNC:
            return cell.full if store else cell.empty
        if view == PV_SYNC and not store:
            return not cell.oldest & PV_MAX
        return False

    def ready(self, access: Access) -> bool:
        """The blocked ``access`` would complete now."""
        cell = self.cells[access.cell]
        return cell.trap or not self.would_block(cell, access.view, access.store)

    def load(self, m, pa: int) -> int:
        """A load (any size: the cell's 32-bit value, MD00534 12.2 —
        partial accesses are undefined)."""
        where = self.decode(pa)
        if where is None:
            return 0
        n, view = where
        cell = self.cells[n]
        if view == CONTROL:
            return cell.tag()
        if view == BYPASS:
            return cell.oldest
        if view in (EF_SYNC, EF_TRY):
            self._gated(m, cell, view, False)
            self.loads += 1
            if cell.empty:                      # the try view: 0
                return 0
            return cell.pop()
        if view in (PV_SYNC, PV_TRY):
            self._gated(m, cell, view, False)
            self.loads += 1
            v = cell.oldest
            if v & PV_MAX:                      # post-decrement
                cell.words[cell.head] = (v & ~PV_MAX) | ((v & PV_MAX) - 1)
            return v
        return 0                                # reserved views

    def store(self, m, pa: int, value: int) -> bool:
        """A store; returns whether it succeeded (SC: the E/F try view
        fails on a full cell — MD00378 7.2, A.2)."""
        where = self.decode(pa)
        if where is None:
            return True
        n, view = where
        cell = self.cells[n]
        value &= M32
        if view == CONTROL:
            cell.write_tag(value)
            return True
        if view == BYPASS:
            cell.bypass_store(value)
            return True
        if view in (EF_SYNC, EF_TRY):
            self._gated(m, cell, view, True)
            self.stores += 1
            if cell.full:                       # the try view: silently not stored
                return False
            cell.push(value)
            return True
        if view in (PV_SYNC, PV_TRY):
            self._gated(m, cell, view, True)
            self.stores += 1
            v = cell.oldest
            if v & PV_MAX < PV_MAX:             # saturating increment
                cell.words[cell.head] = (v & ~PV_MAX) | ((v & PV_MAX) + 1)
            return True
        return True

    def peek(self, pa: int) -> int:
        """What a bypass load would read (the views: no side effects)."""
        where = self.decode(pa)
        if where is None:
            return 0
        n, view = where
        cell = self.cells[n]
        return cell.tag() if view == CONTROL else cell.oldest

    # -- the CACHE pseudo-tags (MD00427 3.3.1) ---------------------------------

    def cache_op(self, m, op: int, va: int) -> str | None:
        """CACHE Index Load Tag D (5) / Index Store Tag D (9) with
        ErrCtl.ITC: address 0 = ITCAddressMap0, 8 = ITCAddressMap1, through
        DTagLo.  None: not an ITC operation."""
        if op not in (5, 9):
            return None
        n = (va >> 3) & 1
        name = f"ITCAddressMap{n}"
        if op == 5:
            m._cp0_set((28, 2), self.map0 if n == 0 else self.map1)
            return f"Index Load Tag D: {name} -> DTagLo"
        self.write_map(n, m.cp0_read(28, 2))
        return f"Index Store Tag D: DTagLo -> {name}"

    # -- the views ------------------------------------------------------------------

    def rows(self) -> list[dict]:
        out = []
        for i, c in enumerate(self.cells):
            out.append({"cell": i, "kind": "FIFO" if c.fifo else "semaphore",
                        "tag": c.tag(), "empty": c.empty, "full": c.full, "trap": c.trap,
                        "count": c.count, "data": [c.words[(c.head + k) % len(c.words)]
                                                   for k in range(len(c.words))],
                        "address": self.cell_address(i)})
        return out

    def cell_address(self, n: int) -> int | None:
        """The physical address of cell ``n``'s bypass view (None while the
        ITC is off or the cell is outside the block)."""
        if not self.enabled:
            return None
        off = n << (7 + (self.map1_rw & 7))
        return self.base() + off if off <= self.span else None

    def lines(self) -> list[str]:
        state = (f"on at 0x{self.base():08x}, {self.span + 1} bytes, cells every "
                 f"{128 << (self.map1_rw & 7)} bytes" if self.enabled else "off (ITCAddressMap0.En = 0)")
        out = [f"ITC: {self.fifos} FIFO + {len(self.cells) - self.fifos} semaphore cells; "
               f"{state}; Map0 0x{self.map0:08x} Map1 0x{self.map1:08x}; "
               f"{self.loads} loads, {self.stores} stores"]
        for r in self.rows():
            addr = f"0x{r['address']:08x}" if r["address"] is not None else "-"
            flags = "".join(f for f, on in (("E", r["empty"]), ("F", r["full"]),
                                            ("T", r["trap"])) if on) or "-"
            data = (" ".join(f"{w:08x}" for w in r["data"][:r["count"]]) if r["kind"] == "FIFO"
                    else f"{r['data'][0]:08x}")
            extra = f" ({r['count']} entries)" if r["kind"] == "FIFO" else ""
            out.append(f"  cell {r['cell']:>2} {r['kind']:<9} at {addr}  {flags:<3} "
                       f"{data}{extra}")
        return out
