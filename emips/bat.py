"""Block Address Translation (``Config.MT = 2``, PLAN M9 phase 7).

Reference: *MIPS32 Architecture For Programmers Volume III* (MD00090,
Rev. 6.02) Appendix A.2 "Block Address Translation".

A BAT MMU relocates whole regions with base-and-bounds pairs instead of
pages.  Six entries, one per reference type (I = fetch, D = data) and
region (Table A.3)::

    0 I kuseg   1 D kuseg   2 I kseg2   3 D kseg2   4 I kseg3   5 D kseg3

Each holds a bounds VPN, a base PFN, C, D and V.  An access to a mapped
address picks its entry; above ``BoundsVPN || 0xFFF`` or with V = 0 it is
a **TLB Invalid** exception (TLBL / TLBS), a store with D = 0 a **TLB
Modified**; else ``PA = VA + (BasePFN << 12)`` (A.2.2).  kseg0 / kseg1
(and kuseg while Status.ERL = 1) are unmapped as always.

CP0 (A.2.3): Index selects the entry for TLBWI / TLBR, EntryHi[31:12]
is BoundsVPN, EntryLo0 holds BasePFN, C, D and V (TLB format).  Random,
EntryLo1, Context, PageMask and Wired are gone (reads 0, writes ignored —
Release 6's rule; UNDEFINED before), TLBP / TLBWR are Reserved
Instructions.  Config1.MMUSize = 5 (six entries).

eMIPS choices (``sim.DEVIATIONS``): all six entries are implemented; no
supervisor mode (as the fixed mapping); a BAT exception writes only
BadVAddr (EntryHi is the bounds register); the boot firmware loads the
entries so the map equals the fixed mapping's.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import cp0 as C
from .mmu import CCA_UNCACHED, M32, Translation, segment

INDEX, ENTRYLO0, ENTRYHI = (0, 0), (2, 0), (10, 0)
#: registers a BAT MMU does not have (A.2.3): read 0, writes ignored
GONE = frozenset({(1, 0), (3, 0), (4, 0), (5, 0), (6, 0)})
REGISTERS = frozenset({INDEX, ENTRYLO0, ENTRYHI}) | GONE
LO_V, LO_D = 2, 4
REGIONS = ("kuseg", "kseg2", "kseg3")


@dataclass(slots=True)
class BATEntry:
    bounds: int = 0          # BoundsVPN (VA >> 12)
    lo: int = 0              # EntryLo0 format: PFN 25:6, C, D, V

    @property
    def pfn(self) -> int:
        return (self.lo >> 6) & 0xFFFFF

    def describe(self) -> str:
        flags = "".join(f for f, b in (("V", LO_V), ("D", LO_D)) if self.lo & b) or "-"
        return (f"bounds 0x{self.bounds << 12 | 0xFFF:08x}, base 0x{self.pfn << 12:08x}, "
                f"C{(self.lo >> 3) & 7} {flags}")


def _arch(code: int, va: int, why: str):
    from .sim import _Arch
    return _Arch(code, badvaddr=va, info={"why": why})


class BATMMU:
    """Block Address Translation MMU (Vol. III A.2), six entries."""

    mt = 2
    name = "BAT"
    supervisor = False
    REGISTERS = REGISTERS
    address_ok = None

    def __init__(self) -> None:
        from .tlb import TLBStats
        self.entries = [BATEntry() for _ in range(6)]
        self.m = None
        #: tracing: the translations recorded since watch() (tlb.TLBAccess)
        self.log: list | None = None
        self.stats = TLBStats()

    def attach(self, m) -> None:
        self.m = m
        cp = m.cp0
        config = cp[C.CONFIG]["value"] & ~((7 << 28) | (7 << 25) | (7 << 7))
        cp[C.CONFIG]["value"] = config | (self.mt << 7)
        cp[C.CONFIG1]["value"] = (cp[C.CONFIG1]["value"] & ~(0x3F << 25)) | (5 << 25)
        for key in REGISTERS:
            cp[key]["value"] = 0
        m._wmask[C.CONFIG] = m._wmask.get(C.CONFIG, 7) & ~((7 << 28) | (7 << 25))

    def cp0_write(self, m, key: tuple[int, int], value: int) -> None:
        if key == INDEX:
            m.cp0[key]["value"] = value & 7
        elif key == ENTRYHI:
            m.cp0[key]["value"] = value & 0xFFFFF000
        elif key == ENTRYLO0:
            m.cp0[key]["value"] = value & ((0xFFFFF << 6) | 0x3F)
        # GONE: ignored

    # -- translation ------------------------------------------------------------------

    @staticmethod
    def entry_of(va: int, kind: str) -> int | None:
        """The BAT entry of a mapped address (Table A.3), None when unmapped."""
        seg = segment(va)
        if seg not in REGIONS:
            return None
        return 2 * REGIONS.index(seg) + (0 if kind == "fetch" else 1)

    def _unmapped(self, m, va: int) -> tuple[int, int] | None:
        if 0x80000000 <= va < 0xC0000000:
            if va < 0xA0000000:
                return va & 0x1FFFFFFF, m.cp0[C.CONFIG]["value"] & 7
            return va & 0x1FFFFFFF, CCA_UNCACHED
        if va < 0x80000000 and m.cp0[C.STATUS]["value"] & C.ST_ERL:
            return va, CCA_UNCACHED
        return None

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        va &= M32
        fixed = self._unmapped(m, va)
        if fixed is not None:
            return fixed
        i = self.entry_of(va, kind)
        e = self.entries[i]
        code = C.TLBS if kind == "store" else C.TLBL
        pa, cca = (va + (e.pfn << 12)) & M32, (e.lo >> 3) & 7
        rec = None
        if self.log is not None:
            from .tlb import TLBAccess
            rec = TLBAccess(kind, va, entry=i, pa=pa, cca=cca, source=e, mmu=self)
            self.log.append(rec)
        if va > (e.bounds << 12 | 0xFFF) or not e.lo & LO_V:
            why = "above the bounds" if e.lo & LO_V else "not valid (V = 0)"
            raise self._fault(rec, "TLB Invalid", code, va, i, why)
        if kind == "store" and not e.lo & LO_D:
            raise self._fault(rec, "TLB Modified", C.MOD, va, i, "not writable (D = 0)")
        self.stats.hits += 1
        return pa, cca

    def _fault(self, rec, name: str, code: int, va: int, i: int, why: str):
        self.stats.fault(name)
        if rec is not None:
            rec.fault, rec.why, rec.pa = name, why, None
        return _arch(code, va, f"BAT entry {i}: {why}")

    @staticmethod
    def entry_name(i: int) -> str:
        return f"BAT entry {i}"

    @staticmethod
    def entry_how(i: int) -> str:
        """``I kuseg`` / ``D kseg2`` ... (Table A.3)."""
        return f"{'I' if i % 2 == 0 else 'D'} {REGIONS[i // 2]}"

    # -- recording (tracing) ------------------------------------------------------------

    def watch(self) -> None:
        self.log = []

    def taken(self) -> list:
        out, self.log = self.log or [], None
        return out

    def reset_stats(self) -> None:
        from .tlb import TLBStats
        self.stats = TLBStats()

    def peek(self, m, va: int) -> tuple[int, int] | None:
        """``(pa, cca)`` as :meth:`describe` gives it (a data entry), no text."""
        va &= M32
        fixed = self._unmapped(m, va)
        if fixed is not None:
            return fixed
        e = self.entries[self.entry_of(va, "load")]
        if va > (e.bounds << 12 | 0xFFF) or not e.lo & LO_V:
            return None
        return (va + (e.pfn << 12)) & M32, (e.lo >> 3) & 7

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        va &= M32
        seg = segment(va)
        fixed = self._unmapped(m, va)
        if fixed is not None:
            how = ("Status.ERL = 1: kuseg unmapped, PA = VA, uncached" if seg == "kuseg"
                   else f"unmapped: PA = VA - 0x{va & 0xE0000000:08x}")
            return Translation(va, fixed[0], fixed[1], seg, False, how)
        i = self.entry_of(va, kind)
        e = self.entries[i]
        cca = (e.lo >> 3) & 7
        if va > (e.bounds << 12 | 0xFFF) or not e.lo & LO_V:
            return Translation(va, None, cca, seg, True,
                               f"BAT entry {i}: {'above the bounds' if e.lo & LO_V else 'not valid'}",
                               entry=i, fault="invalid")
        pa = (va + (e.pfn << 12)) & M32
        return Translation(va, pa, cca, seg, True,
                           f"BAT entry {i} ({'I' if i % 2 == 0 else 'D'} {seg}): "
                           f"PA = VA + 0x{e.pfn << 12:08x}", entry=i)

    def exception_entry(self, m, va: int) -> None:
        """A BAT exception sets only BadVAddr (done by the machine)."""

    # -- TLBWI / TLBR -----------------------------------------------------------------

    def tlbwi(self, m) -> None:
        i = m.cp0[INDEX]["value"] % 6
        self.entries[i] = BATEntry(m.cp0[ENTRYHI]["value"] >> 12,
                                   m.cp0[ENTRYLO0]["value"] & ~1)
        self.stats.writes += 1
        if self.log is not None:
            from .tlb import TLBAccess
            self.log.append(TLBAccess("write", 0, entry=i, mmu=self,
                                      note=f"TLBWI wrote BAT entry {i} ({self.entry_how(i)}): "
                                           f"{self.entries[i].describe()}"))

    def tlbr(self, m) -> None:
        e = self.entries[m.cp0[INDEX]["value"] % 6]
        m.cp0[ENTRYHI]["value"] = e.bounds << 12
        m.cp0[ENTRYLO0]["value"] = e.lo

    # -- firmware ----------------------------------------------------------------------

    def boot(self, m, *, handler: bool) -> None:
        """The boot firmware's map, the fixed mapping's: kuseg at VA + 1 GB
        (cacheable), kseg2 / kseg3 at PA = VA (uncached: the devices)."""
        from .tlb import boot_cca
        user = BATEntry(0x7FFFF, (0x40000 << 6) | (boot_cca(m) << 3) | LO_D | LO_V)
        k2 = BATEntry(0xDFFFF, (2 << 3) | LO_D | LO_V)
        k3 = BATEntry(0xFFFFF, (2 << 3) | LO_D | LO_V)
        self.entries = [user, BATEntry(user.bounds, user.lo), k2,
                        BATEntry(k2.bounds, k2.lo), k3, BATEntry(k3.bounds, k3.lo)]
