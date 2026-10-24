"""The MIPS P8700's MMU (PLAN M25 phase 5c; MD01502 rev. 1.84 ch. 2, 7.3.2
/ 7.3.3 / 7.3.9 / 7.3.11, Table 7.4, Appendix B MTLBWR).

:class:`P8700MMU` is an :class:`emips.rvmmu.RVMMU` (Sv39 / Sv48 page tables,
the permission checks, SFENCE.VMA) with the P8700's translation buffers
instead of the generic TLB:

- **The VTLB** (2.1.4): ``vtlb`` dual entries (16 / 32 / 64), fully
  associative, binary-tree pseudo-LRU; every page size (4 KB, 64 KB, 2 MB,
  1 GB, 512 GB) and the non-leaf entries of the software walk.  A dual
  entry holds two adjacent pages of one size (the bit above the page
  offset picks the half); the tag is the VPN above it, the ASID (or G)
  and the level.
- **The FTLB** (2.1.5): 128 sets x 4 ways of dual entries of one page
  size — 4 KB, or 64 KB with mipsconfig7.FTLB64 (Table 2.1) —, "modified
  round robin" per set.  A write of that size goes to the FTLB, except one
  in 64 / 32 / 16 (mipsconfig7.FTLBP) that goes to the VTLB; every other
  size to the VTLB (2.7).
- **The micro-TLBs** (2.1.2 / 2.1.3): the ITLB (``itlb`` pages) and DTLB
  (``dtlb`` dual entries) of 4 KB / 64 KB pieces, LRU, cleared whenever
  the VTLB / FTLB change; a miss refills from them and costs the
  pipeline ``UTLB`` cycles (:meth:`take`).
- **Duplicates** (2.4): a write searches the whole VTLB and the FTLB set
  of its address and invalidates overlapping translations (a large page
  in the VTLB misses duplicates in other FTLB sets — as the manual says).
- **The hardware table walker** (2.7; mipsconfig7.DHTW = 0): on a miss it
  walks the page table from satp (through the caches) and writes the leaf
  to the FTLB / VTLB; an invalid PTE is the standard page fault.
- **The software walk** (DHTW = 1; 7.3.2 / 7.3.9, Table 7.4, MTLBWR): a
  miss is a custom exception through mipstvec — satpTLBMiss (60) when no
  non-leaf entry covers the address (the missing PTE is the root
  table's), else Instruction / Load / Store TLB Miss (24 / 25 / 27) with
  the PTE the deepest non-leaf entry points to — with mtval = the VA,
  mipstval = the missing PTE's physical address, mipsconfig5.LEVEL its
  level, ATYPE the access, XATP satp, G the non-leaf entry's G.  The
  handler reads the PTE and MTLBWR writes it at LEVEL for mtval: a leaf
  (a translation), a non-leaf PTE (a non-leaf entry), or an invalid one
  (an entry that faults: the retried access takes the standard page
  fault — as a MIPS refill handler's invalid entry does).
- **Svnapot** (64 KB pages, priv. ch. 15 — the P8700's TLBs and walker
  have them): a level-0 leaf with N = 1 and PPN[3:0] = 1000; any other N
  is a page fault.

The views (``describe`` / ``peek``) look in the TLBs without touching the
replacement state, else walk.  Choices: ``sim.DEVIATIONS["P8700 MMU"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mmu import Translation
from .rvmmu import (ACCESS_FAULT, FAULT, PA_BITS, PAGE, SCHEMES, A, D, G, PageFault, R,
                    RVMMU, Scheme, U, V, W, X)

#: Svnapot (priv. 15): the N bit, the 64 KB encoding
N_BIT = 1 << 63
NAPOT64 = 4
#: mipsconfig7 (7.3.11)
DHTW, FTLB64 = 1 << 7, 1 << 20
FTLBP_SHIFT = 18
#: FTLBP: one in N FTLB-sized writes goes to the VTLB (3: none)
FTLBP_EVERY = {0: 64, 1: 32, 2: 16, 3: 0}
#: the custom TLB-miss causes (Table 7.4) and mipsconfig5's ATYPE
MISS_CAUSE = {"fetch": 24, "load": 25, "store": 27}
SATP_MISS = 60
ATYPE = {"fetch": 0, "load": 1, "store": 3}


class SoftMiss(Exception):
    """A software-walk TLB miss (the custom exception's facts)."""

    def __init__(self, cause: int, pte_addr: int, level: int, g: bool, kind: str) -> None:
        super().__init__(cause)
        self.cause, self.pte_addr, self.level, self.g, self.kind = (cause, pte_addr, level,
                                                                     g, kind)


@dataclass
class Entry:
    """A dual entry: ``shift`` = log2 of the page in 4 KB pages (0, 4 =
    64 KB, 9, 18, 27); ``tag`` = the VPN above the pair; ``half`` = the two
    PTEs (None: empty); ``leaf`` False = a non-leaf entry (its PTEs point
    to the next table); ``level`` = the PTE's level."""
    asid: int                # -1: global
    tag: int
    shift: int
    level: int
    leaf: bool
    half: list = field(default_factory=lambda: [None, None])

    def covers(self, vpn: int, asid: int) -> bool:
        return (self.asid == -1 or self.asid == asid) and vpn >> (self.shift + 1) == self.tag

    def pte(self, vpn: int) -> int | None:
        return self.half[(vpn >> self.shift) & 1]

    @property
    def empty(self) -> bool:
        return self.half[0] is None and self.half[1] is None


class _Tree:
    """Binary-tree pseudo-LRU over ``n`` ways (n a power of two)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.bits = [0] * max(n - 1, 1)

    def touch(self, way: int) -> None:
        node, lo, hi = 0, 0, self.n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            right = way >= mid
            self.bits[node] = 0 if right else 1          # (point away from it)
            node = 2 * node + (2 if right else 1)
            lo, hi = (mid, hi) if right else (lo, mid)

    def victim(self) -> int:
        node, lo, hi = 0, 0, self.n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            right = self.bits[node] == 1
            node = 2 * node + (2 if right else 1)
            lo, hi = (mid, hi) if right else (lo, mid)
        return lo


class P8700MMU(RVMMU):
    """See the module docstring."""

    name = "P8700 MMU"

    def __init__(self, vtlb: int = 64, itlb: int = 16, dtlb: int = 32,
                 ftlb: tuple[int, int] = (128, 4)) -> None:
        super().__init__(vtlb)
        if vtlb not in (16, 32, 64) or itlb not in (8, 16) or dtlb not in (16, 32, 64):
            raise ValueError("the P8700's VTLB has 16 / 32 / 64 dual entries, its ITLB 8 / 16, "
                             "its DTLB 16 / 32 / 64 (MD01502 2.1.1)")
        self.nv, self.ni, self.nd = vtlb, itlb, dtlb
        self.sets, self.ways = ftlb
        self.vtlb: list[Entry | None] = [None] * vtlb
        self.vtree = _Tree(vtlb)
        self.ftlb: list[list[Entry | None]] = [[None] * self.ways for _ in range(self.sets)]
        self.fnext = [0] * self.sets                     # (round robin per set)
        self.itlb: list[tuple] = []                      # (asid, vpn piece, shift, entry)
        self.dtlb: list[tuple] = []
        self._ftlbp = 0
        #: statistics (the views, the performance counters)
        self.stats = {"itlb": [0, 0], "dtlb": [0, 0], "tlb": [0, 0], "walks": 0, "soft": 0,
                      "writes": 0}
        #: the pipeline's micro-TLB timing (emips.rvpipeline: take())
        self.timing = False
        self.utlb_cost = 3
        self.wait = 0
        self.notes: list[str] = []
        self.ftlb64 = False
        #: the MMUs sharing this FTLB (2.6: a core's harts; their micro-TLBs
        #: are cleared when it changes)
        self.group: list[P8700MMU] = [self]

    def share_ftlb(self, owner: "P8700MMU") -> None:
        """Use ``owner``'s FTLB (another hart of the core, 2.6)."""
        self.ftlb, self.fnext, self.group = owner.ftlb, owner.fnext, owner.group
        self.ftlb64 = owner.ftlb64
        self.group.append(self)

    # -- configuration (mipsconfig7) ------------------------------------------------

    def _config7(self, m) -> int:
        return m.p8700.config7 if m.p8700 is not None else 0

    def config7_written(self, m) -> None:
        """FTLB64 changed: the FTLB's entries are of the other size — gone."""
        big = bool(self._config7(m) & FTLB64)
        if big != self.ftlb64:
            for mmu in self.group:
                mmu.ftlb64 = big
            for ways in self.ftlb:          # (in place: the harts share it)
                ways[:] = [None] * self.ways
            self._changed()

    @property
    def fshift(self) -> int:
        return NAPOT64 if self.ftlb64 else 0

    def _fset(self, vpn: int, shift: int) -> int:
        return (vpn >> (shift + 1)) % self.sets

    # -- lookups ------------------------------------------------------------------

    def _find(self, vpn: int, asid: int, *, touch: bool, leaf: bool = True,
              level: int | None = None) -> Entry | None:
        """The entry (leaf, or a non-leaf one of ``level``) holding ``vpn``."""
        for k, e in enumerate(self.vtlb):
            if e is not None and e.leaf == leaf and (level is None or e.level == level) \
                    and e.covers(vpn, asid) and e.pte(vpn) is not None:
                if touch:
                    self.vtree.touch(k)
                return e
        if leaf:
            for e in self.ftlb[self._fset(vpn, self.fshift)]:
                if e is not None and e.covers(vpn, asid) and e.pte(vpn) is not None:
                    return e
        return None

    def _micro(self, kind: str, vpn: int, asid: int, *, record: bool) -> Entry | None:
        """The ITLB / DTLB (4 KB / 64 KB pieces, LRU); a miss refills from the
        VTLB / FTLB (which may miss too: None)."""
        name = "itlb" if kind == "fetch" else "dtlb"
        buf = self.itlb if kind == "fetch" else self.dtlb
        for k, (a, piece, sh, e) in enumerate(buf):
            if (a == -1 or a == asid) and vpn >> sh == piece:
                if record:
                    self.stats[name][0] += 1
                    buf.append(buf.pop(k))
                return e
        if record:
            self.stats[name][1] += 1
        e = self._find(vpn, asid, touch=record)
        if record:
            self.stats["tlb"][0 if e is not None else 1] += 1
            if e is not None:
                self._fill_micro(kind, vpn, e)
        return e

    def _fill_micro(self, kind: str, vpn: int, e: Entry) -> None:
        """A micro-TLB refill: the page's 4 KB / 64 KB piece, LRU."""
        buf = self.itlb if kind == "fetch" else self.dtlb
        sh = 0 if e.shift == 0 else NAPOT64
        buf.append((e.asid, vpn >> sh, sh, e))
        del buf[:-(self.ni if kind == "fetch" else 2 * self.nd)]
        if self.timing:
            self.wait += self.utlb_cost
            self.notes.append(f"{'ITLB' if kind == 'fetch' else 'DTLB'} miss: refilled from "
                              f"the {'FTLB' if self._in_ftlb(e) else 'VTLB'} "
                              f"({self.utlb_cost} cycles)")

    def _in_ftlb(self, e: Entry) -> bool:
        return any(e is x for x in self.ftlb[e.tag % self.sets])

    # -- writes ---------------------------------------------------------------------

    def _changed(self) -> None:
        """2.1.1: the micro-TLBs are cleared whenever the TLB changes (every
        hart's that shares the FTLB)."""
        for mmu in self.group:
            mmu.itlb.clear()
            mmu.dtlb.clear()

    def write(self, m, va: int, pte: int, level: int, asid: int, g: bool) -> str:
        """Put ``pte`` (of ``level``, for ``va``) into the VTLB / FTLB as the
        walker and MTLBWR do; returns where ("VTLB" / "FTLB")."""
        sch = self._scheme(m)
        vpn = (va >> PAGE) & ((1 << (sch.levels * sch.idx_bits)) - 1)
        leaf = bool(pte & (R | X)) or not pte & V
        shift = level * sch.idx_bits
        if leaf and level == 0 and pte & N_BIT and pte & V:
            shift = NAPOT64
        key = -1 if g or pte & G else asid
        tag = vpn >> (shift + 1)
        self.stats["writes"] += 1
        self._drop_duplicates(vpn, shift, key, leaf)
        to_ftlb = leaf and shift == self.fshift and self._ftlb_ok(m)
        if to_ftlb:
            s = tag % self.sets
            ways = self.ftlb[s]
            e = next((x for x in ways if x is not None and x.tag == tag and x.asid == key
                      and x.shift == shift), None)
            if e is None:
                w = next((k for k, x in enumerate(ways) if x is None), None)
                if w is None:
                    w = self.fnext[s]
                    self.fnext[s] = (w + 1) % self.ways
                e = ways[w] = Entry(key, tag, shift, level, True)
        else:
            e = next((x for x in self.vtlb if x is not None and x.tag == tag and x.asid == key
                      and x.shift == shift and x.leaf == leaf and x.level == level), None)
            if e is None:
                k = next((k for k, x in enumerate(self.vtlb) if x is None), None)
                if k is None:
                    k = self.vtree.victim()
                e = self.vtlb[k] = Entry(key, tag, shift, level, leaf)
                self.vtree.touch(k)
        e.half[(vpn >> shift) & 1] = pte
        self._changed()
        return "FTLB" if to_ftlb else "VTLB"

    def _ftlb_ok(self, m) -> bool:
        """mipsconfig7.FTLBP: one FTLB-sized write in N goes to the VTLB."""
        every = FTLBP_EVERY[(self._config7(m) >> FTLBP_SHIFT) & 3]
        if not every:
            return True
        self._ftlbp = (self._ftlbp + 1) % every
        return self._ftlbp != 0

    def _drop_duplicates(self, vpn: int, shift: int, asid: int, leaf: bool) -> None:
        """2.4: the whole VTLB and one FTLB set, overlapping translations of
        the same address space (the half that overlaps)."""
        lo, hi = vpn >> shift << shift, (vpn >> shift << shift) + (1 << shift)

        def clear(e: Entry) -> None:
            for h in (0, 1):
                start = ((e.tag << 1) | h) << e.shift
                if e.half[h] is not None and start < hi and lo < start + (1 << e.shift) \
                        and (e.asid == asid or -1 in (e.asid, asid)) and e.leaf == leaf \
                        and (leaf or e.shift == shift):     # (non-leaf levels nest)
                    e.half[h] = None
        for k, e in enumerate(self.vtlb):
            if e is not None:
                clear(e)
                if e.empty:
                    self.vtlb[k] = None
        if leaf:
            s = self._fset(vpn, self.fshift)
            for k, e in enumerate(self.ftlb[s]):
                if e is not None:
                    clear(e)
                    if e.empty:
                        self.ftlb[s][k] = None

    def sfence(self, m, va: int | None, asid: int | None) -> None:
        """SFENCE.VMA (12.2.1) on every buffer (the non-leaf entries too)."""
        sch = self._scheme(m)
        mask = (1 << (sch.levels * sch.idx_bits)) - 1
        vpn = None if va is None else (va >> PAGE) & mask

        def gone(e: Entry) -> bool:
            if asid is not None and (e.asid == -1 or e.asid != asid):
                return False
            return vpn is None or vpn >> (e.shift + 1) == e.tag or not e.leaf
        for k, e in enumerate(self.vtlb):
            if e is not None and gone(e):
                self.vtlb[k] = None
        for ways in self.ftlb:
            for k, e in enumerate(ways):
                if e is not None and gone(e):
                    ways[k] = None
        self._changed()
        super().sfence(m, va, asid)

    def flush_all(self) -> None:
        self.vtlb = [None] * self.nv
        for ways in self.ftlb:
            ways[:] = [None] * self.ways
        self._changed()

    # -- translation ----------------------------------------------------------------

    def _scheme(self, m) -> Scheme:
        return self._satp(m)[0] or SCHEMES[(m.xlen, 8)]

    def take(self) -> tuple[int, list[str]]:
        out = (self.wait, self.notes)
        self.wait, self.notes = 0, []
        return out

    def effective_priv(self, m, kind: str) -> int:          # type: ignore[override]
        """mipsconfig5.MTW (7.3.9): M-mode loads and stores with the walker's
        mapping (physical) and privilege (S, priv. 3.7.2)."""
        if self._mtw(m, kind):
            return 1
        return RVMMU.effective_priv(m, kind)

    @staticmethod
    def _mtw(m, kind: str) -> bool:
        p = m.p8700
        return (p is not None and kind != "fetch" and m.priv == 3 and m.dm_access is None
                and bool(p.config5 & 4))

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        kind = "store" if kind == "store" else "fetch" if kind == "fetch" else "load"
        if self._mtw(m, kind):                          # (the walker's mapping: physical)
            pa = va & m.amask
            if pa >> PA_BITS:
                from .simrv import RVTrap
                raise RVTrap(ACCESS_FAULT[kind], va)
            return pa, self._cca(m, pa)
        try:
            return super().xlate(m, va, kind)
        except SoftMiss as s:
            from .simrv import RVTrap
            t = RVTrap(s.cause, va, custom=True)
            t.tlb_miss = s
            raise t from None

    def _translate(self, m, va: int, kind: str, priv: int, sch: Scheme, asid: int,
                   root: int, *, record: bool) -> int:
        top = va >> (sch.va_bits - 1)
        if m.xlen == 64 and top not in (0, (1 << (64 - sch.va_bits + 1)) - 1):
            raise PageFault(FAULT[kind], "the VA's upper bits are not its sign")
        vpn = (va >> PAGE) & ((1 << (sch.levels * sch.idx_bits)) - 1)
        e = self._micro(kind, vpn, asid, record=True) if record else \
            self._find(vpn, asid, touch=False)
        if e is not None:
            pte = e.pte(vpn)
            self._leaf_ok(pte, e.level, sch, kind)
            self._check(m, pte, kind, priv, record=record)
            return self._addr(pte, e.level, va, sch)
        if record:
            self.misses += 1
        hw = not self._config7(m) & DHTW
        if record and not hw:                           # (the software walk: 7.3.9)
            self.stats["soft"] += 1
            raise self._soft_miss(vpn, asid, kind, sch, root)
        pte, level = self._walk8700(m, va, kind, sch, root, view=not record)
        self._check(m, pte, kind, priv, record=record)
        if record:
            self.walks += 1
            self.stats["walks"] += 1
            self.write(m, va, pte, level, asid, False)
            e = self._find(vpn, asid, touch=True)
            if e is not None:
                self._fill_micro(kind, vpn, e)
        return self._addr(pte, level, va, sch)

    def _soft_miss(self, vpn: int, asid: int, kind: str, sch: Scheme, root: int) -> SoftMiss:
        """The deepest non-leaf entry that covers the address decides the
        missing PTE (none: the root table's — satpTLBMiss)."""
        for level in range(1, sch.levels):
            e = self._find(vpn, asid, touch=True, leaf=False, level=level)
            if e is not None:
                pte = e.pte(vpn)
                base = ((pte >> 10) & ((1 << 44) - 1)) << PAGE
                idx = (vpn >> ((level - 1) * sch.idx_bits)) & ((1 << sch.idx_bits) - 1)
                return SoftMiss(MISS_CAUSE[kind], base + idx * sch.pte_size, level - 1,
                                e.asid == -1, kind)
        top = sch.levels - 1
        idx = (vpn >> (top * sch.idx_bits)) & ((1 << sch.idx_bits) - 1)
        return SoftMiss(SATP_MISS, (root << PAGE) + idx * sch.pte_size, top, False, kind)

    @staticmethod
    def _addr(pte: int, level: int, va: int, sch: Scheme) -> int:
        ppn = (pte >> 10) & ((1 << 44) - 1)
        if level == 0 and pte & N_BIT:                  # (Svnapot: 64 KB)
            low = (1 << (PAGE + NAPOT64)) - 1
        else:
            low = (1 << (PAGE + level * sch.idx_bits)) - 1
        return ((ppn << PAGE) & ~low) | (va & low)

    @staticmethod
    def _leaf_ok(pte: int, level: int, sch: Scheme, kind: str) -> None:
        """What makes a leaf PTE a page fault (12.3.2 step 3 / 6, Svnapot)."""
        if not pte & V or (pte & (R | W)) == W:
            raise PageFault(FAULT[kind], "an invalid PTE" if not pte & V else "W without R")
        if (pte >> 54) & 0x1FF:
            raise PageFault(FAULT[kind], "reserved PTE bits set (62:54)")
        ppn = (pte >> 10) & ((1 << 44) - 1)
        if pte & N_BIT:
            if level or ppn & 0xF != 0b1000:
                raise PageFault(FAULT[kind], "a reserved Svnapot encoding")
        elif ppn & ((1 << (level * sch.idx_bits)) - 1):
            raise PageFault(FAULT[kind], "a misaligned superpage")
        if not pte & (R | X):
            raise PageFault(FAULT[kind], "no leaf PTE")

    def _walk8700(self, m, va: int, kind: str, sch: Scheme, root: int,
                  view: bool = False) -> tuple[int, int]:
        """The hardware table walker (2.7): 12.3.2 with Svnapot."""
        base = root << PAGE
        for level in range(sch.levels - 1, -1, -1):
            idx = (va >> (PAGE + level * sch.idx_bits)) & ((1 << sch.idx_bits) - 1)
            addr = base + idx * sch.pte_size
            if addr >> PA_BITS:
                raise PageFault(ACCESS_FAULT[kind], "a PTE beyond physical memory")
            pte = m.load_pte(addr, sch.pte_size, view)
            if pte is None:
                raise PageFault(ACCESS_FAULT[kind], "PMP refuses the PTE's address")
            if not pte & V or (pte & (R | W)) == W:
                raise PageFault(FAULT[kind], "an invalid PTE" if not pte & V else "W without R")
            if pte & (R | X):
                self._leaf_ok(pte, level, sch, kind)
                return pte, level
            if pte >> 54 or pte & (D | A | U):
                raise PageFault(FAULT[kind], "a non-leaf PTE with N / reserved bits / D / A / U")
            base = ((pte >> 10) & ((1 << 44) - 1)) << PAGE
        raise PageFault(FAULT[kind], "no leaf PTE at level 0")

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        t = super().describe(m, va, kind)
        if t.pa is None or t.how.startswith(("M-mode", "satp.MODE")):
            return t
        scheme, asid, _ = self._satp(m)
        vpn = (va >> PAGE) & ((1 << (scheme.levels * scheme.idx_bits)) - 1)
        e = self._find(vpn, asid, touch=False)
        where = "not in the TLB" if e is None else \
            f"{'FTLB' if self._in_ftlb(e) else 'VTLB'} hit"
        return Translation(t.va, t.pa, t.cca, t.segment, t.mapped, f"{t.how}; {where}")

    # -- the views ----------------------------------------------------------------------

    SIZES = {0: "4 KB", NAPOT64: "64 KB", 9: "2 MB", 18: "1 GB", 27: "512 GB"}

    def rows(self, m) -> list[tuple[str, str, int, str, int, int, bool]]:
        """(where, ASID or "G", VPN of the half, page size, PPN, PTE, leaf)
        of every valid half (the TLB pane)."""
        out = []
        cells = [(f"VTLB {k}", e) for k, e in enumerate(self.vtlb) if e is not None]
        cells += [(f"FTLB {s}:{w}", e) for s, ways in enumerate(self.ftlb)
                  for w, e in enumerate(ways) if e is not None]
        for where, e in cells:
            for h in (0, 1):
                pte = e.half[h]
                if pte is not None:
                    out.append((where, "G" if e.asid == -1 else str(e.asid),
                                ((e.tag << 1) | h) << e.shift, self.SIZES.get(e.shift, "?"),
                                (pte >> 10) & ((1 << 44) - 1), pte, e.leaf))
        return out

    def lines(self, m) -> list[str]:
        sch = self._scheme(m)
        sizes = {0: "4K", NAPOT64: "64K", 9: "2M", 18: "1G", 27: "512G"}
        out = [f"P8700 MMU: VTLB {self.nv} dual entries, FTLB {self.sets} x {self.ways} "
               f"({'64K' if self.ftlb64 else '4K'} pages), ITLB {self.ni}, DTLB {self.nd}; "
               f"{'software walk (DHTW)' if self._config7(m) & DHTW else 'hardware walker'}"]

        def row(where: str, e: Entry) -> str:
            halves = []
            for h in (0, 1):
                pte = e.half[h]
                if pte is None:
                    halves.append("-")
                    continue
                va = (((e.tag << 1) | h) << e.shift) << PAGE
                if m.xlen == 64 and va >> (sch.va_bits - 1) & 1:
                    va |= ~((1 << sch.va_bits) - 1) & ((1 << 64) - 1)
                pa = ((pte >> 10) & ((1 << 44) - 1)) << PAGE
                flags = "".join(c for c, b in (("V", V), ("R", R), ("W", W), ("X", X), ("U", U),
                                               ("A", A), ("D", D)) if pte & b)
                halves.append(f"0x{va:x}->0x{pa:x} {flags}")
            kind = "" if e.leaf else " non-leaf"
            return (f"  {where}: {sizes.get(e.shift, e.shift)} level {e.level}{kind} "
                    f"{'G' if e.asid == -1 else f'ASID {e.asid}'}: " + " | ".join(halves))
        for k, e in enumerate(self.vtlb):
            if e is not None:
                out.append(row(f"VTLB {k}", e))
        for s, ways in enumerate(self.ftlb):
            for w, e in enumerate(ways):
                if e is not None:
                    out.append(row(f"FTLB {s}:{w}", e))
        st = self.stats
        out.append(f"  ITLB {st['itlb'][0]} hits / {st['itlb'][1]} misses, DTLB "
                   f"{st['dtlb'][0]} / {st['dtlb'][1]}, VTLB / FTLB {st['tlb'][0]} / "
                   f"{st['tlb'][1]}, walks {st['walks']}, software misses {st['soft']}, "
                   f"writes {st['writes']}")
        return out
