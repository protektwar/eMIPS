"""Address translation: virtual → physical (PLAN M9).

Reference: *MIPS32 Architecture For Programmers Volume III* (MD00090),
chapter "Virtual Memory" (the segments, address error checks) and
Appendix A.1 "Fixed Mapping MMU" (Rev. 6.02, pp. 313-315).

Every address a program uses is **virtual**.  Its top bits pick a
segment::

    kuseg 0x00000000-0x7FFFFFFF  user / kernel   mapped
    kseg0 0x80000000-0x9FFFFFFF  kernel          PA = VA - 0x80000000, Config.K0
    kseg1 0xA0000000-0xBFFFFFFF  kernel          PA = VA - 0xA0000000, uncached
    kseg2 0xC0000000-0xDFFFFFFF  kernel (sseg)   mapped
    kseg3 0xE0000000-0xFFFFFFFF  kernel          mapped

kseg0 and kseg1 are two windows on the same low 512 MB of physical
memory: one cached (by ``Config.K0``), one never cached.  The *mapped*
segments go through the MMU; :class:`FixedMappingMMU` is the
``Config.MT = 3`` organization (A.1.1):

* kuseg: PA = VA + 1 GB while ``Status.ERL = 0``; PA = VA (identity,
  uncached) while ``ERL = 1`` — so physical 0x20000000-0x3FFFFFFF cannot
  be reached with ERL = 0, and 0x80000000-0xBFFFFFFF not with ERL = 1;
* kseg2 / kseg3: PA = VA;
* cacheability: kuseg ``Config.KU``, kseg2 / kseg3 ``Config.K23``;
* no supervisor mode (``Status`` bit 3 is the reserved R0, reads 0).

Address errors (alignment, segment rights) are checked **before** the
translation, as the manual says; the fixed mapping itself never raises.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import cp0 as C

M32 = 0xFFFFFFFF

#: Cacheability and coherency attributes (Vol. III Table 9.12): 2 and 3
#: are required, the others are implementation-dependent.
CCA_UNCACHED = 2
CCA_CACHEABLE = 3
#: The attributes a core without caches implements: the two required
#: ones (with caches, the 24K's: :data:`emips.cache.CCA_24K`).
SUPPORTED_CCA = frozenset({CCA_UNCACHED, CCA_CACHEABLE})

#: ``Config`` cacheability fields: K0 = 2..0, KU = 27..25, K23 = 30..28.
K0_SHIFT, KU_SHIFT, K23_SHIFT = 0, 25, 28


def cca_name(cca: int) -> str:
    """What a cacheability attribute means (Vol. III Table 9.12; 0 and 7
    as the 24K, MD00343 Table 4.11; 4 / 5 as the cores of a CPS, the
    Coherence Manager notes — reserved on a 24K)."""
    return {0: "cacheable, write-through", CCA_UNCACHED: "uncached",
            CCA_CACHEABLE: "cacheable, write-back",
            4: "cacheable coherent, write-back, exclusive on a load miss (CPS)",
            5: "cacheable coherent, write-back (CPS)",
            7: "uncached accelerated"}.get(cca, "reserved")


def segment(va: int) -> str:
    """The segment of a 32-bit virtual address."""
    va &= M32
    if va < 0x80000000:
        return "kuseg"
    if va < 0xA0000000:
        return "kseg0"
    if va < 0xC0000000:
        return "kseg1"
    if va < 0xE0000000:
        return "kseg2"
    return "kseg3"


@dataclass(frozen=True)
class Translation:
    """One translated access, as the trace and the views show it.  ``pa``
    is ``None`` when nothing maps the address now (a TLB miss, ``fault =
    "refill"``, or an invalid page, ``"invalid"``)."""
    va: int
    pa: int | None
    cca: int                 # cacheability and coherency attribute
    segment: str
    mapped: bool             # went through the MMU (not kseg0 / kseg1)
    how: str                 # e.g. "fixed mapping: PA = VA + 0x40000000"
    entry: int | None = None # the TLB entry used
    fault: str = ""

    @property
    def cacheable(self) -> bool:
        return self.cca != CCA_UNCACHED


class FixedMappingMMU:
    """Fixed Mapping MMU (``Config.MT = 3``, Vol. III Appendix A.1)."""

    mt = 3
    name = "fixed mapping"
    #: Supervisor mode is not supported with a Fixed Mapping MMU (A.1.1).
    supervisor = False

    def pa(self, status: int, config: int, va: int) -> int:
        """Physical address of *va* (the fast path of :meth:`translate`)."""
        va &= M32
        if va < 0x80000000:
            return va if status & C.ST_ERL else va + 0x40000000
        if va < 0xC0000000:
            return va & 0x1FFFFFFF
        return va

    def cca(self, status: int, config: int, va: int) -> int:
        """Cacheability attribute of *va* (the fast path of :meth:`translate`)."""
        va &= M32
        if va < 0x80000000:
            return CCA_UNCACHED if status & C.ST_ERL else (config >> KU_SHIFT) & 7
        if va < 0xA0000000:
            return (config >> K0_SHIFT) & 7
        if va < 0xC0000000:
            return CCA_UNCACHED
        return (config >> K23_SHIFT) & 7

    # the machine-level interface every MMU has (see emips.tlb.TLBMMU)

    def attach(self, m) -> None:
        self.m = m

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        """``(pa, cca)`` of an access; the fixed mapping never raises."""
        status, config = m.cp0[C.STATUS]["value"], m.cp0[C.CONFIG]["value"]
        return self.pa(status, config, va), self.cca(status, config, va)

    def peek(self, m, va: int) -> tuple[int, int]:
        return self.xlate(m, va, "load")

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        return self.translate(m.cp0[C.STATUS]["value"], m.cp0[C.CONFIG]["value"], va)

    def translate(self, status: int, config: int, va: int) -> Translation:
        va &= M32
        seg = segment(va)
        pa = self.pa(status, config, va)
        if seg == "kuseg":
            if status & C.ST_ERL:
                return Translation(va, pa, CCA_UNCACHED, seg, True,
                                   "fixed mapping, Status.ERL = 1: PA = VA, uncached")
            return Translation(va, pa, (config >> KU_SHIFT) & 7, seg, True,
                               "fixed mapping: PA = VA + 0x40000000 (Config.KU)")
        if seg == "kseg0":
            return Translation(va, pa, (config >> K0_SHIFT) & 7, seg, False,
                               "unmapped: PA = VA - 0x80000000 (Config.K0)")
        if seg == "kseg1":
            return Translation(va, pa, CCA_UNCACHED, seg, False,
                               "unmapped: PA = VA - 0xA0000000, uncached")
        return Translation(va, pa, (config >> K23_SHIFT) & 7, seg, True,
                           "fixed mapping: PA = VA (Config.K23)")


def default_mmu() -> FixedMappingMMU:
    return FixedMappingMMU()


#: the MMU organizations :func:`make_mmu` builds (``Config.MT``)
MMU_KINDS = ("fixed", "bat", "tlb", "ftlb")
MMU_SPECS = "fixed | bat | tlb[:N] | ftlb[:N[:SETSxWAYS]], + sc / + pw"


def make_mmu(spec: str):
    """An MMU from a spec (the options, ``--mmu``)::

        fixed                   the Fixed Mapping MMU (Config.MT = 3)
        bat                     Block Address Translation (MT = 2)
        tlb[:N]                 a TLB of N dual entries (1..64, default 32;
                                the 24K's JTLB, MT = 1)
        ftlb[:N[:SETSxWAYS]]    a VTLB of N entries (default 64) and a fixed
                                page size TLB (default 128x4; the P5600's,
                                MT = 4)

    A TLB takes the options ``+sc`` (Segmentation Control and EVA,
    Release 3+) and ``+pw`` (the hardware page table walker, Release 5+),
    e.g. ``ftlb+sc+pw``."""
    text = (spec or "fixed").strip().lower().replace(" ", "")
    base, *flags = text.split("+")
    kind, _, rest = base.partition(":")
    if kind in ("fixed", "bat"):
        if rest or flags:
            raise ValueError(f"{kind} takes no size or options: {spec!r}")
        if kind == "fixed":
            return FixedMappingMMU()
        from .bat import BATMMU
        return BATMMU()
    if kind not in ("tlb", "ftlb") or not set(flags) <= {"sc", "pw"}:
        raise ValueError(f"unknown MMU {spec!r} ({MMU_SPECS})")
    from .tlb import TLBMMU
    parts = rest.split(":") if rest else []
    try:
        entries = int(parts[0]) if parts and parts[0] else (64 if kind == "ftlb" else 32)
        ftlb = None
        if kind == "ftlb":
            sets, ways = (int(x) for x in (parts[1] if len(parts) > 1 else "128x4").split("x"))
            ftlb = (sets, ways)
        elif len(parts) > 1:
            raise ValueError
    except ValueError:
        raise ValueError(f"bad MMU size in {spec!r} ({MMU_SPECS})") from None
    return TLBMMU(entries, ftlb=ftlb, sc="sc" in flags, pw="pw" in flags)
