"""The I6400's Inter-Thread Communication Unit: gating storage for the
virtual processors of a Release 6 core (PLAN M22 phase 6f).

Sources: the I6400 Programmer's Guide MD01196 rev. 1.00 chapter 11 — 11.1
(a 64-bit data path, 4- or 8-byte accesses, a 128 KB block), 11.1.1 (SAARI
/ SAAR: the block's base and its enable), 11.1.2 (ICR0: CELL_NUM, BLK_GRAIN,
the three error bits), 11.2 (single-entry and FIFO cells, Table 11.7 the
views in address bits 6:3, Table 11.8 the cell state, 11.2.4 / 11.2.5 the
cell index above bit 7 + BLK_GRAIN, bit 2 a word of the 64-bit cell), 11.3
(always uncached, aligned, SC on the E/F try view), 11.4.2 (one ITU, one
SAAR per core: every VP of the core reaches the same cells), 11.5 (the
errors: an error response the software sees as a bus error).  QEMU's
``hw/misc/mips_itu.c`` (the I6400 / Boston model) where the guide is
silent: SAAR bits 43:13 = PA 47:17, a Control write of E = 1 empties a
FIFO, a P/V load returns the count it decrements.

A blocking access (an E/F synchronized load of an empty cell, a store to a
full one, a P/V synchronized load of a zero count) raises
:class:`emips.exceptions.Blocked` (``"gs"``): it has had no effect, and
the VP issues it again until it completes.  What eMIPS chooses where the
guide is silent is in ``sim.DEVIATIONS["ITU"]``.
"""

from __future__ import annotations

from .exceptions import Blocked, BusError
from .itc import Access

M64 = (1 << 64) - 1
#: Table 11.7: the views (address bits 6:3); 6-14 behave as Bypass
BYPASS, CONTROL, EF_SYNC, EF_TRY, PV_SYNC, PV_TRY, ICR = 0, 1, 2, 3, 4, 5, 15
VIEW_NAMES = {BYPASS: "bypass", CONTROL: "control", EF_SYNC: "E/F synchronized",
              EF_TRY: "E/F try", PV_SYNC: "P/V synchronized", PV_TRY: "P/V try",
              ICR: "ICR0"}
#: the block (11.1: "a fixed size of 128 KBytes"; SAAR.SIZE = 17)
SIZE_LOG2 = 17
SIZE = 1 << SIZE_LOG2
#: SAAR (11.1.1.2; QEMU: bits 43:13 hold PA 47:17): ADDR, SIZE (5:1), EN (0)
SAAR_ADDR = ((1 << 44) - 1) & ~((1 << 13) - 1)
SAAR_SIZE = SIZE_LOG2 << 1
SAAR_EN = 1
#: at most 32 cells (Table 11.5 CELL_NUM)
MAX_CELLS = 32
#: FIFO depths a multi-entry cell can have (Table 11.8 FIFO_DEPTH 1-3)
DEPTHS = (2, 4, 8)
#: ICR0 (Table 11.5): CELL_NUM 23:16, BLK_GRAIN 10:8 (R/W), the errors (W1C)
ICR0_GRAIN_SHIFT, ICR0_GRAIN = 8, 7 << 8
ERROR_AXI, ERROR_PARITY, ERROR_EXEC = 1 << 2, 1 << 1, 1 << 0
#: Table 11.8: the cell state
STATE_EMPTY, STATE_FULL, STATE_FIFO = 1 << 0, 1 << 1, 1 << 17
#: the P/V count: the 16 low bits, saturating at 0xffff (Table 11.7)
PV_MAX = 0xFFFF
#: "auto": 4 FIFO cells of 4 entries and 8 single-entry cells (the 34K's ITU)
AUTO = (4, 8, 4)


def parse_spec(spec: str) -> tuple[int, int, int] | None:
    """``"none"`` -> None; ``"FIFOS:SINGLES[:DEPTH]"`` -> the cells (at most
    32 together, MD01196 Table 11.5; a FIFO of 2, 4 or 8 entries, Table 11.8:
    4 when not given); ``"auto"`` is the caller's."""
    spec = (spec or "").strip().lower()
    if spec in ("", "none", "off", "no"):
        return None
    parts = spec.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"ITU spec {spec!r}: FIFOS:SINGLES[:DEPTH], e.g. 4:8") from None
    fifos, singles = nums[0], nums[1] if len(nums) > 1 else 0
    depth = nums[2] if len(nums) > 2 else 4
    if len(nums) > 3 or fifos < 0 or singles < 0 or not 0 < fifos + singles <= MAX_CELLS \
            or depth not in DEPTHS:
        raise ValueError(f"ITU spec {spec!r}: FIFO and single-entry cells, 1 to {MAX_CELLS} "
                         f"together, FIFOs of {', '.join(map(str, DEPTHS))} entries")
    return fifos, singles, depth


class Cell:
    """One cell (Table 11.8): a single-entry cell or a FIFO of ``depth``
    64-bit entries, with its Empty / Full state (reset: Empty)."""

    def __init__(self, depth: int) -> None:
        self.fifo = depth > 1
        self.depth = depth
        self.words = [0] * depth
        self.head = 0                           # FIFO_RD_PTR: the oldest entry
        self.count = 0
        self.empty = True
        self.full = False

    def state(self) -> int:
        """The Control view: FIFO_DEPTH 29:28, FIFO_RD_PTR from bit 18,
        FIFO_TYPE 17, Full 1, Empty 0."""
        if self.fifo:
            log2 = self.depth.bit_length() - 1
            out = (log2 << 28) | (self.head << 18) | STATE_FIFO
        else:
            out = 0
        return out | (STATE_FULL if self.full else 0) | (STATE_EMPTY if self.empty else 0)

    def write_state(self, value: int) -> None:
        self.full = bool(value & STATE_FULL)
        self.empty = bool(value & STATE_EMPTY)
        if self.empty and self.fifo:            # (QEMU: E = 1 empties a FIFO)
            self.head = self.count = 0

    @property
    def oldest(self) -> int:
        return self.words[self.head]

    def pop(self) -> int:
        v = self.words[self.head]
        if self.fifo:
            self.head = (self.head + 1) % self.depth
            self.count -= 1
            self.empty = self.count == 0
        else:
            self.empty = True
        self.full = False
        return v

    def push(self, value: int) -> None:
        if self.fifo:
            self.words[(self.head + self.count) % self.depth] = value
            self.count += 1
            self.full = self.count == self.depth
        else:
            self.words[0] = value
            self.full = True
        self.empty = False


class ITU:
    """The ITU of one core: ``fifos`` FIFO cells (cell numbers 0..) of
    ``depth`` entries, then ``singles`` single-entry cells, every cell 64
    bits; at the base SAAR gives it once SAAR.EN is set."""

    #: what the Machine's gating-storage path needs (emips.sim): 64-bit cells
    width = 64
    bytes = 8
    mask = M64
    digits = 16

    def __init__(self, fifos: int, singles: int, depth: int = 4) -> None:
        if not 0 < fifos + singles <= MAX_CELLS or depth not in DEPTHS:
            raise ValueError("an ITU has 1 to 32 cells, FIFOs of 2, 4 or 8 entries")
        self.cells = [Cell(depth) for _ in range(fifos)] + [Cell(1) for _ in range(singles)]
        self.fifos = fifos
        self.depth = depth
        #: the block's physical address (SAAR.ADDR) and SAAR.EN
        self.base = 0
        self.enabled = False
        #: ICR0's BLK_GRAIN and error bits
        self.grain = 0
        self.errors = 0
        #: loads / stores that reached the cells
        self.loads = self.stores = 0

    # -- placement (SAAR, 11.4.1) --------------------------------------------------

    def place(self, saar: int) -> None:
        """SAAR was written: the block at PA 47:17 = SAAR 43:13, on with EN."""
        self.base = ((saar & SAAR_ADDR) >> 13) << SIZE_LOG2
        self.enabled = bool(saar & SAAR_EN)

    def covers(self, pa: int) -> bool:
        return self.enabled and self.base <= pa < self.base + SIZE

    @property
    def icr0(self) -> int:
        return (len(self.cells) << 16) | (self.grain << ICR0_GRAIN_SHIFT) | self.errors

    @property
    def index_bits(self) -> int:
        """The cell index's width: "the number of memory locations allocated
        must be on a power of 2" (11.2.5.3)."""
        return max(1, (len(self.cells) - 1).bit_length())

    def decode(self, pa: int) -> tuple[int, int]:
        """(cell, view) at physical ``pa``: the index from bit 7 + BLK_GRAIN
        (the bits above it up to 15, and below it down to 7, select nothing:
        11.2.5), the view in bits 6:3.  The cell may be past the last one."""
        off = pa - self.base
        cell = (off >> (7 + self.grain)) & ((1 << self.index_bits) - 1)
        return cell, (off >> 3) & 0xF

    # -- accesses -------------------------------------------------------------------

    def _error(self, pa: int, bit: int, why: str, store: bool) -> BusError:
        self.errors |= bit
        return BusError(pa, "write" if store else "read",
                        f"ITU 0x{pa:08x}: {why} (ICR0.{_ERROR_NAMES[bit]})")

    def _cell(self, pa: int, size: int, store: bool) -> tuple[Cell | None, int, int]:
        """The cell and view an access reaches, or the error response: an
        access of 1 or 2 bytes (11.5.1: only 32- and 64-bit transactions),
        a cell past the configured ones, the ICR view of a cell other than
        0 (11.5.3)."""
        if size not in (4, 8):
            raise self._error(pa, ERROR_AXI, f"a {size}-byte access", store)
        n, view = self.decode(pa)
        if n >= len(self.cells):
            raise self._error(pa, ERROR_EXEC, f"no cell {n}", store)
        if view == ICR:
            if n:
                raise self._error(pa, ERROR_EXEC, f"the ICR view of cell {n}", store)
            return None, n, view
        return self.cells[n], n, view

    def _state_check(self, pa: int, cell: Cell, view: int, store: bool) -> None:
        """The execution errors of 11.5.3: an E/F access to a single-entry
        cell whose E and F are equal, to a FIFO with both set; any P/V
        access to a FIFO, to a single-entry cell without E = F = 0."""
        if view in (EF_SYNC, EF_TRY):
            bad = cell.empty and cell.full if cell.fifo else cell.empty == cell.full
            if bad:
                raise self._error(pa, ERROR_EXEC, "an E/F access with E = "
                                  f"{int(cell.empty)}, F = {int(cell.full)}", store)
        elif cell.fifo:
            raise self._error(pa, ERROR_EXEC, "a P/V access to a FIFO cell", store)
        elif cell.empty or cell.full:
            raise self._error(pa, ERROR_EXEC, "a P/V access with E or F set", store)

    def would_block(self, cell: Cell, view: int, store: bool) -> bool:
        if view == EF_SYNC:
            return cell.full if store else cell.empty and not cell.full
        if view == PV_SYNC and not store:
            return not cell.words[0] & PV_MAX
        return False

    def ready(self, access: Access) -> bool:
        """The blocked ``access`` would complete now."""
        cell = self.cells[access.cell]
        return not self.would_block(cell, access.view, access.store)

    def load(self, m, pa: int, size: int = 8) -> int:
        """A load's 64-bit cell value (the Machine takes the word bit 2
        selects)."""
        cell, n, view = self._cell(pa, size, False)
        if cell is None:
            return self.icr0
        if view == CONTROL:
            return cell.state()
        if view in (EF_SYNC, EF_TRY):
            self._state_check(pa, cell, view, False)
            if self.would_block(cell, view, False):
                raise Blocked("gs", Access(n, view, False))
            self.loads += 1
            return 0 if cell.empty else cell.pop()     # the try view: 0
        if view in (PV_SYNC, PV_TRY):
            self._state_check(pa, cell, view, False)
            if self.would_block(cell, view, False):
                raise Blocked("gs", Access(n, view, False))
            self.loads += 1
            v = cell.words[0] & PV_MAX          # the 16 bits, zero-extended
            if v:                               # an atomic post-decrement
                cell.words[0] = v - 1
            return v
        return cell.oldest                      # Bypass (and the reserved views)

    def store(self, m, pa: int, value: int, size: int = 8) -> bool:
        """A store (``value`` the 64-bit cell value with the bytes in their
        place); returns whether it succeeded (SC on the E/F try view: a
        full cell fails, 11.3)."""
        cell, n, view = self._cell(pa, size, True)
        value &= M64
        if cell is None:                        # ICR0: BLK_GRAIN R/W, the errors W1C
            self.grain = (value & ICR0_GRAIN) >> ICR0_GRAIN_SHIFT
            self.errors &= ~(value & (ERROR_AXI | ERROR_PARITY | ERROR_EXEC))
            return True
        if view == CONTROL:
            cell.write_state(value)
            return True
        if view in (EF_SYNC, EF_TRY):
            self._state_check(pa, cell, view, True)
            if cell.full:
                if view == EF_SYNC:
                    raise Blocked("gs", Access(n, view, True))
                return False                    # the try view: silently not stored
            self.stores += 1
            cell.push(value)
            return True
        if view in (PV_SYNC, PV_TRY):
            self._state_check(pa, cell, view, True)
            self.stores += 1                    # the data ignored: an increment
            cell.words[0] = min(PV_MAX, (cell.words[0] & PV_MAX) + 1)
            return True
        # Bypass: the data, no state change (a FIFO: the newest entry, 11.2.1)
        cell.words[(cell.head + max(cell.count, 1) - 1) % cell.depth] = value
        return True

    def peek(self, pa: int) -> int:
        """What a Bypass / Control / ICR load would read (no side effects)."""
        n, view = self.decode(pa)
        if n >= len(self.cells):
            return 0
        if view == ICR:
            return self.icr0 if n == 0 else 0
        cell = self.cells[n]
        return cell.state() if view == CONTROL else cell.oldest

    # -- the views ------------------------------------------------------------------

    def cell_address(self, n: int) -> int | None:
        """The physical address of cell ``n``'s Bypass view, None while the
        ITU is off."""
        return self.base + (n << (7 + self.grain)) if self.enabled else None

    def rows(self) -> list[dict]:
        out = []
        for i, c in enumerate(self.cells):
            out.append({"cell": i, "kind": "FIFO" if c.fifo else "single",
                        "tag": c.state(), "empty": c.empty, "full": c.full, "trap": False,
                        "count": c.count if c.fifo else int(c.full),
                        "data": [c.words[(c.head + k) % c.depth] for k in range(c.depth)],
                        "address": self.cell_address(i)})
        return out

    def lines(self) -> list[str]:
        state = (f"on at 0x{self.base:08x}, {SIZE} bytes, cells every "
                 f"{128 << self.grain} bytes" if self.enabled else "off (SAAR.EN = 0)")
        singles = len(self.cells) - self.fifos
        errs = ", ".join(_ERROR_NAMES[b] for b in (ERROR_AXI, ERROR_PARITY, ERROR_EXEC)
                         if self.errors & b) or "none"
        out = [f"ITU: {self.fifos} FIFO cells of {self.depth} entries + {singles} "
               f"single-entry cells of 64 bits; {state}; ICR0 0x{self.icr0:08x} "
               f"(errors: {errs}); {self.loads} loads, {self.stores} stores"]
        for r in self.rows():
            addr = f"0x{r['address']:08x}" if r["address"] is not None else "-"
            flags = "".join(f for f, on in (("E", r["empty"]), ("F", r["full"])) if on) or "-"
            data = (" ".join(f"{w:016x}" for w in r["data"][:r["count"]])
                    if r["kind"] == "FIFO" else f"{r['data'][0]:016x}")
            extra = f" ({r['count']} entries)" if r["kind"] == "FIFO" else ""
            out.append(f"  cell {r['cell']:>2} {r['kind']:<6} at {addr}  {flags:<2} "
                       f"{data}{extra}")
        return out


_ERROR_NAMES = {ERROR_AXI: "ERROR_AXI", ERROR_PARITY: "ERROR_PARITY",
                ERROR_EXEC: "ERROR_EXEC"}
