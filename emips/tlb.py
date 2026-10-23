"""The TLB-based MMU (``Config.MT = 1``, PLAN M9 phase 6).

References: *MIPS32 Architecture For Programmers Volume III* (MD00090,
Rev. 6.02): 4.9 "TLB-Based Virtual Address Translation" (the lookup
pseudo code, 1 KB pages, RI / XI), 6.2.8 / 6.2.10 / 6.2.13 / 6.2.14 (the
Machine Check, TLB Refill, Invalid and Modified exceptions), chapter 9
(Index, Random, EntryLo0-1, Context, PageMask, PageGrain, Wired, BadInstr,
BadInstrP, EntryHi, Config1 / 3 / 4); *Volume II* (MD00086): TLBP, TLBR,
TLBWI, TLBWR, TLBINV, TLBINVF.  Where the architecture leaves a choice to
the implementation, the **24K** (MD00343 *24K Software User's Manual*
4.3-4.4, 2.2.3, 2.3.1, 6.2.1-6.2.11; MD00355 *Programming the 24K* 3.7):

* a **JTLB** of 16, 32 or 64 dual entries (any 1..64 here; default 32),
  pages 4 KB .. 256 MB (and 1 KB with ``PageGrain.ESP``, Release 2-5);
* every entry has a hidden "written" bit, cleared by reset and included
  in the match, so an uninitialised entry never matches (4.4.1);
* **write-compare**: a TLBWI / TLBWR is compared with every other entry;
  matching entries are disabled, unless both the match and the new entry
  are valid — then the write is aborted and a **Machine Check** is taken
  (``Status.TS``, Table 4.9).  So there are never multiple matches;
* ``Random`` is a clock-rate down-counter from N-1 to ``Wired``
  (MD00355 3.7.3);
* a 4-entry **ITLB** (true LRU) and an 8-entry **DTLB** (pseudo-LRU)
  hold 4 KB or 1 MB (sub)pages of JTLB entries; a miss costs the
  pipeline ``UTLB`` cycles (MD00355 3.7.2: "three extra clocks"); a JTLB
  write removes the copies of the entry written.  They never change a
  result, only time.

The Release 3+ options come from Vol. III: RI / XI (``Config3.RXI``,
``PageGrain.RIE / XIE / IEC``), ``EntryHi.EHINV`` with TLBINV / TLBINVF
(``Config4.IE``), BadInstr / BadInstrP (``Config3.BI / BP``); 1 KB pages
(``Config3.SP``) in Release 2-5.  Each is on for the releases that define
it (:class:`TLBMMU` arguments turn them off).  Release 6 removes Random
(TLBWR still uses the hidden counter), makes ``Index.P`` writable, ignores
an out-of-range Index or an illegal PageMask.

Address errors are checked before the translation (as with every MMU);
the TLB exceptions are raised as :class:`emips.sim._Arch` from
:meth:`TLBMMU.xlate` and their CP0 side effects (BadVAddr, Context,
EntryHi) happen when the exception is *taken* (:meth:`TLBMMU.exception_entry`).
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field

from . import cp0 as C
from . import ptw, segctl
from .mmu import CCA_UNCACHED, M32, Translation, segment

# --------------------------------------------------------------- registers
INDEX = (0, 0)
RANDOM = (1, 0)
ENTRYLO0, ENTRYLO1 = (2, 0), (3, 0)
CONTEXT = (4, 0)
PAGEMASK = (5, 0)
PAGEGRAIN = (5, 1)
WIRED = (6, 0)
BADINSTR, BADINSTRP = (8, 1), (8, 2)
ENTRYHI = (10, 0)
CONFIG4 = (16, 4)

#: the registers the TLB MMU owns (MTC0 goes through :meth:`TLBMMU.cp0_write`)
REGISTERS = frozenset({INDEX, RANDOM, ENTRYLO0, ENTRYLO1, CONTEXT, PAGEMASK,
                       PAGEGRAIN, WIRED, BADINSTR, BADINSTRP, ENTRYHI, C.STATUS})

# ------------------------------------------------------------------- fields
INDEX_P = 1 << 31
#: 24K: the Index / Random / Wired fields are bits 5:0 (MD00343 6.2.1)
INDEX_BITS = 0x3F
LO_G, LO_V, LO_D = 1, 2, 4
LO_C_SHIFT = 3
LO_RI, LO_XI = 1 << 31, 1 << 30
HI_ASID = 0xFF
HI_EHINV = 1 << 10
HI_VPN2 = 0xFFFFE000                     # VA 31:13
HI_VPN2X = 0x1800                        # VA 12:11 (1 KB pages)
PM_MASK = 0x1FFFE000                     # PageMask 28:13
PM_MASKX = 0x1800                        # PageMask 12:11
PG_RIE, PG_XIE, PG_ELPA, PG_ESP, PG_IEC = 1 << 31, 1 << 30, 1 << 29, 1 << 28, 1 << 27
ST_TS = 1 << 21                          # Status.TS: TLB shutdown (Machine Check)
CTX_PTEBASE = 0xFF800000                 # Context 31:23
CTX_BADVPN2 = 0x007FFFF0                 # Context 22:4

# MIPS64 (PLAN M22 phase 3; Vol. III 9.27 / 9.7 / 9.52): a TLB with 64-bit
# virtual addresses — EntryHi R (63:62) | Fill | VPN2 (SEGBITS-1:13),
# Context PTEBase 63:23, XContext PTEBase | R | BadVPN2 (SEGBITS-1:13)
M64 = (1 << 64) - 1
XCONTEXT = (20, 0)
HI_R = 3 << 62
#: the TLB MMU's registers that are 64 bits wide on MIPS64 (EntryHi, Context,
#: XContext; Segmentation Control's SegCtl0-2 and EBase; the walker's PWBase /
#: PWField / PWSize)
WIDE_KEYS = frozenset({(10, 0), (4, 0), XCONTEXT, (5, 2), (5, 3), (5, 4), (15, 1),
                       (5, 5), (5, 6), (5, 7)})
#: eMIPS's MIPS64 TLB: SEGBITS = 40 (the manual's default VPN2 width) and
#: PABITS = 32 (the EntryLo PFN of the 24K; xkphys windows of 4 GB)
SEGBITS, PABITS = 40, 32

#: Config3 bits (Vol. III Figure 9-50)
CFG3_M, CFG3_BP, CFG3_BI, CFG3_RXI, CFG3_SP = 1 << 31, 1 << 27, 1 << 26, 1 << 12, 1 << 4

#: the page sizes of the 24K: 4 KB .. 256 MB in powers of 4 (MD00343 4.4.2.1)
PAGE_SIZES = tuple(4096 << (2 * k) for k in range(9))

#: micro-TLB miss: cycles to refill it from the JTLB (MD00355 3.7.2)
UTLB_MISS = 3


def _lfsr_skips() -> tuple[int, ...]:
    """The Random perturbation of the 24K (MD00343 6.2.2: "an LFSR register
    is used which prevents the decrement pseudo-randomly", a 10-bit LFSR,
    4.4: "to reduce the possibility for a livelock").  Its taps are not
    published: eMIPS steps the maximal-length LFSR x^10 + x^7 + 1 once per
    cycle and skips the decrement in the cycles whose state has its low 4
    bits set — about 1 cycle in 16 ("decrements almost every cycle").
    One period (1023 cycles): 1 = the decrement is skipped."""
    state, out = 1, []
    for _ in range(1023):
        out.append(1 if state & 0xF == 0xF else 0)
        state = ((state << 1) | (((state >> 9) ^ (state >> 6)) & 1)) & 0x3FF
    assert state == 1                        # maximal length: back after 1023
    return tuple(out)


LFSR_SKIPS = _lfsr_skips()
#: skips before position i of the period (prefix sums)
_LFSR_CUM = tuple(sum(LFSR_SKIPS[:i]) for i in range(1024))


def size_name(size: int) -> str:
    for unit, n in (("MB", 1 << 20), ("KB", 1 << 10)):
        if size >= n:
            return f"{size // n} {unit}"
    return f"{size} B"


@dataclass(slots=True)
class TLBEntry:
    """One dual entry: the tag (PageMask, VPN2, G, ASID) and two halves
    (EntryLo0 / EntryLo1 without G).  ``mask`` is the don't-care mask in
    VA bits 28:11 (MaskX = 11 when 1 KB pages were not enabled at the
    write); ``written`` is the 24K's hidden bit; ``ehinv`` the Release 3
    VPN2-invalid bit."""
    pagemask: int = 0
    hi: int = 0                          # EntryHi VPN2 | VPN2X | ASID (as written)
    g: bool = False
    lo: list[int] = field(default_factory=lambda: [0, 0])
    small: bool = False                  # 1 KB pages enabled when written
    written: bool = False
    ehinv: bool = False
    #: the EntryHi bits a match compares (VPN2 | VPN2X; MIPS64: also R and
    #: VPN2 up to SEGBITS - 1)
    vmask: int = HI_VPN2 | HI_VPN2X

    @property
    def mask(self) -> int:
        m = self.pagemask & (PM_MASK | PM_MASKX)
        return m if self.small else m | PM_MASKX

    @property
    def vpn2(self) -> int:
        return self.hi & self.vmask & ~self.mask

    @property
    def asid(self) -> int:
        return self.hi & HI_ASID

    @property
    def evenodd(self) -> int:
        """The VA bit that picks the even or odd half (12 for 4 KB pages)."""
        return 10 + (self.mask >> 11).bit_length()

    @property
    def page_size(self) -> int:
        return 1 << self.evenodd

    @property
    def live(self) -> bool:
        return self.written and not self.ehinv

    def valid(self) -> bool:
        return bool((self.lo[0] | self.lo[1]) & LO_V)

    def matches(self, va: int, asid: int) -> bool:
        return (self.written and not self.ehinv
                and not (va ^ self.hi) & self.vmask & ~self.mask
                and (self.g or self.hi & HI_ASID == asid))

    def frame(self, half: int) -> int:
        """Physical base address of one half (32-bit physical addresses:
        PFN 25:6 = PA 31:12, or 27:6 = PA 31:10 with 1 KB pages)."""
        lo = self.lo[half]
        base = ((lo >> 6) & 0x3FFFFF) << 10 if self.small else ((lo >> 6) & 0xFFFFF) << 12
        return base & ~(self.page_size - 1) & M32

    def describe(self) -> str:
        if not self.written:
            return "never written"
        if self.ehinv:
            return "invalidated (EHINV)"
        va = self.vpn2
        span = 2 * self.page_size
        own = "global" if self.g else f"ASID {self.asid}"
        halves = []
        for h in (0, 1):
            lo = self.lo[h]
            flags = "".join(f for f, b in (("V", LO_V), ("D", LO_D), ("RI", LO_RI),
                                           ("XI", LO_XI)) if lo & b) or "-"
            halves.append(f"PA 0x{self.frame(h):08x} C{(lo >> LO_C_SHIFT) & 7} {flags}")
        top = (va + span - 1) & (M64 if va > M32 else M32)
        return (f"VA 0x{va:08x}-0x{top:08x} ({own}, "
                f"{size_name(self.page_size)} pages): " + " | ".join(halves))


@dataclass(slots=True)
class MicroEntry:
    """A micro-TLB entry: one 4 KB or 1 MB (sub)page of a JTLB entry."""
    vbase: int
    size: int
    asid: int
    g: bool
    index: int                           # the JTLB entry it was copied from
    pbase: int
    cca: int
    lo: int                              # the EntryLo half (D, RI, XI)

    def matches(self, va: int, asid: int) -> bool:
        return va - self.vbase < self.size and va >= self.vbase and (self.g or self.asid == asid)


class MicroTLB:
    """ITLB / DTLB: a few fully associative copies of JTLB translations,
    refilled by hardware (MD00343 2.2.3 / 2.3.1).  ``lru`` = true LRU (the
    ITLB), else the tree pseudo-LRU of the DTLB."""

    def __init__(self, name: str, size: int, *, lru: bool) -> None:
        self.name = name
        self.size = size
        self.lru = lru
        self.slots: list[MicroEntry | None] = [None] * size
        self.order: list[int] = list(range(size))     # LRU: least recent first
        self.tree = 0                                 # pseudo-LRU bits
        self.hits = self.misses = 0

    def lookup(self, va: int, asid: int) -> MicroEntry | None:
        for i, e in enumerate(self.slots):
            if e is not None and e.matches(va, asid):
                self.hits += 1
                self._touch(i)
                return e
        self.misses += 1
        return None

    def _touch(self, i: int) -> None:
        if self.lru:
            self.order.remove(i)
            self.order.append(i)
            return
        node, lo, hi = 0, 0, self.size            # point the tree away from i
        while hi - lo > 1 and node < self.size - 1:
            mid = (lo + hi) // 2
            if i < mid:
                self.tree |= 1 << node
                node, hi = 2 * node + 1, mid
            else:
                self.tree &= ~(1 << node)
                node, lo = 2 * node + 2, mid

    def _victim(self) -> int:
        for i, e in enumerate(self.slots):
            if e is None:
                return i
        if self.lru:
            return self.order[0]
        node, lo, hi = 0, 0, self.size
        while hi - lo > 1 and node < self.size - 1:
            mid = (lo + hi) // 2
            if self.tree >> node & 1:
                node, lo = 2 * node + 2, mid
            else:
                node, hi = 2 * node + 1, mid
        return lo

    def fill(self, e: MicroEntry) -> None:
        i = self._victim()
        self.slots[i] = e
        self._touch(i)

    def drop(self, index: int | None = None) -> None:
        """Remove the copies of JTLB entry ``index`` (``None``: all)."""
        for i, e in enumerate(self.slots):
            if e is not None and (index is None or e.index == index):
                self.slots[i] = None

    def reset_counters(self) -> None:
        self.hits = self.misses = 0


#: names of the TLB exceptions in the records (TLBAccess.fault)
FAULT_NAMES = {"refill": "TLB Refill", "invalid": "TLB Invalid", "mod": "TLB Modified",
               "ri": "TLB Read-Inhibit", "xi": "TLB Execute-Inhibit"}


@dataclass(slots=True)
class TLBAccess:
    """One translation (or TLB write) recorded while tracing
    (:meth:`TLBMMU.watch` / :meth:`TLBMMU.taken`; the BAT MMU too): the
    virtual address, the micro-TLB it went through, the entry that
    matched, the EntryLo half it used (PFN, C, flags) and the physical
    address — or the exception it raised.  ``kind`` is the access
    (``fetch`` / ``load`` / ``pcload`` / ``store`` / ``cacheop`` /
    ``pref``), or ``write`` / ``invalidate`` / ``probe`` / ``read`` for
    TLBWI / TLBWR / TLBINV* / TLBP / TLBR (``note`` says what happened)."""

    kind: str
    va: int
    asid: int = 0
    micro: str = ""                  # "ITLB" / "DTLB" ("" without micro-TLBs)
    micro_hit: bool = False
    entry: int | None = None         # the TLB / BAT entry used
    half: int = 0                    # 0 = even page (EntryLo0), 1 = odd (EntryLo1)
    pa: int | None = None
    cca: int | None = None
    fault: str = ""                  # FAULT_NAMES value, "" = translated
    why: str = ""
    walk: str = ""                   # what the page table walker did
    note: str = ""                   # write / invalidate: what changed
    #: the entry object used (TLBEntry / BATEntry: an entry write replaces
    #: the object, so it keeps the values of the translation) and the MMU
    #: (its names); the properties below are derived from them only when
    #: read, so recording stays cheap
    source: object = field(default=None, repr=False, compare=False)
    mmu: object = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        """``TLB entry 5``, ``FTLB set 3 way 1 (entry 70)``, ``BAT entry 1``."""
        if self.entry is None:
            return ""
        return self.mmu.entry_name(self.entry) if self.mmu is not None else f"entry {self.entry}"

    @property
    def page(self) -> int:
        """Page size (0: a BAT entry, base and bounds)."""
        e = self.source
        return e.page_size if isinstance(e, TLBEntry) else 4096 if e is None else 0

    @property
    def pfn(self) -> int | None:
        """EntryLo.PFN of the half used (the BAT entry's base PFN)."""
        e = self.source
        if isinstance(e, TLBEntry):
            return (e.lo[self.half] >> 6) & (0x3FFFFF if e.small else 0xFFFFF)
        return None if e is None else e.pfn

    @property
    def frame(self) -> int | None:
        """The physical address of the page (the BAT base)."""
        e = self.source
        if isinstance(e, TLBEntry):
            return e.frame(self.half)
        return None if e is None else e.pfn << 12

    @property
    def how(self) -> str:
        """The entry: ``global, 4 KB odd page, VD`` (BAT: ``D kuseg``)."""
        e = self.source
        if isinstance(e, TLBEntry):
            lo = e.lo[self.half]
            flags = "".join(f for f, b in (("V", LO_V), ("D", LO_D), ("RI", LO_RI),
                                           ("XI", LO_XI)) if lo & b) or "-"
            return (f"{'global' if e.g else f'ASID {e.asid}'}, {size_name(e.page_size)} "
                    f"{('even', 'odd')[self.half]} page, {flags}")
        if e is None or self.mmu is None:
            return ""
        return self.mmu.entry_how(self.entry)

    @property
    def vpn2(self) -> int:
        """VA bits above the page pair (``VA >> 13`` for 4 KB pages)."""
        return self.va >> max(self.page.bit_length(), 1)

    @property
    def offset(self) -> int:
        return self.va & (self.page - 1) if self.page else 0

    def describe(self) -> str:
        if self.note:                    # TLBWI / TLBWR / TLBP / TLBR / TLBINV*
            return self.note
        parts = []
        if self.micro:
            parts.append(f"{self.micro} {'hit' if self.micro_hit else 'miss'}")
        if self.walk:
            parts.append(self.walk)
        if self.entry is None:
            vpn = (f"R {self.va >> 62}, VPN2 0x{(self.va >> 13) & ((1 << (SEGBITS - 13)) - 1):05x}"
                   if self.va > M32 else f"VPN2 0x{self.vpn2:05x}")
            parts.append(f"no TLB entry maps {vpn} for ASID {self.asid}")
        else:
            parts.append(f"{self.name} ({self.how})" if self.how else self.name)
        if self.fault:
            if self.why and self.entry is not None:
                parts[-1] += f": {self.why}"
            parts.append(self.fault)
        elif self.pa is not None:
            if self.page:
                parts += [f"PFN 0x{self.pfn:x}", f"PA 0x{self.pa:08x} (C{self.cca})"]
            else:
                parts.append(f"VA + base 0x{self.frame:08x} = PA 0x{self.pa:08x} "
                             f"(C{self.cca})")
        return f"{self.kind} 0x{self.va:08x}: " + " -> ".join(parts)


@dataclass
class TLBStats:
    """What the TLB did since reset (or :meth:`TLBMMU.reset_stats`):
    ``hits`` / ``misses`` count the lookups in the TLB itself (micro-TLB
    misses; the micro-TLBs count their own), ``faults`` the exceptions by
    name, ``walks`` / ``walked`` the page table walks tried / that wrote
    an entry, ``writes`` the entries written (TLBWI, TLBWR, the walker)."""

    hits: int = 0
    misses: int = 0
    walks: int = 0
    walked: int = 0
    writes: int = 0
    faults: dict[str, int] = field(default_factory=dict)
    #: TLB lookups (micro-TLB misses) that found no valid entry: no match
    #: or V = 0, for fetches and for data (the 24K's JTLB miss events)
    jtlb_misses: dict[str, int] = field(default_factory=lambda: {"fetch": 0, "data": 0})
    #: data accesses to unmapped addresses (the 24K's DTLB access event
    #: counts them too)
    unmapped_data: int = 0

    def fault(self, name: str) -> None:
        self.faults[name] = self.faults.get(name, 0) + 1


def _arch(code: int, va: int, *, refill: bool = False, why: str = "", xtlb: bool = False):
    from .sim import _Arch
    return _Arch(code, badvaddr=va, info={"tlb": True, "refill": refill, "why": why,
                                          "xtlb": xtlb})


#: The boot firmware's TLB refill handler (``Program.load(boot=True)``
#: installs it at EBase + 0x000 when the program has none): it maps the
#: missing page pair as the fixed mapping MMU would — kuseg at VA + 1 GB,
#: cacheable; kseg2 / kseg3 at PA = VA, uncached (the MMIO devices) — in
#: global 4 KB pages, with a random entry (TLBWR).
def firmware_refill(cca: int = 3, window: tuple[int, int] | None = None,
                    arch: str = "mips32") -> str:
    """The boot firmware's refill handler mapping kuseg with attribute
    ``cca`` (3; a core of a CPS: 5, cacheable coherent).  ``window`` =
    (base, size) of a board's memory for kuseg (the Malta: the top
    power-of-two part of its SDRAM): PA = base + (VA mod size).
    ``arch`` "nanomips": the same handler in nanoMIPS code
    (:data:`FIRMWARE_REFILL_NANO`; assemble it with ``.ktext`` at
    0x80000000 — GNU as's ``.ktext N`` is a subsection)."""
    if arch == "nanomips":
        return _nano_refill(cca, window)
    if window is not None:
        base, size = window
        mask = (size - 1) & ~0x1FFF
        lo = 0x07 | cca << 3                        # C, D, V, G
        return f"""\
        .set noreorder
        .ktext 0x80000000
__tlb_refill:
        mfc0  $k1, $10           # EntryHi: VPN2 of the miss | ASID
        lui   $k0, 0x{mask >> 16:04x}        # the window: VA mod 0x{size:x}
        bltz  $k1, __tlb_k23     # VA >= 0x80000000: kseg2 / kseg3, PA = VA
        ori   $k0, $k0, 0x{mask & 0xFFFF:04x}   # (delay slot)
        and   $k0, $k0, $k1      # the page pair's offset in the window
        lui   $k1, 0x{base >> 16:04x}        # + its base 0x{base:08x}
        addu  $k0, $k0, $k1
        srl   $k0, $k0, 12
        sll   $k0, $k0, 6        # EntryLo PFN of the even page
        b     __tlb_set
        ori   $k0, $k0, 0x{lo:02x}     # (delay slot) C = {cca}, D, V, G
__tlb_k23:
        srl   $k0, $k1, 13
        sll   $k0, $k0, 7
        ori   $k0, $k0, 0x17     # C = 2 uncached, D, V, G
__tlb_set:
        mtc0  $k0, $2            # EntryLo0: the even page
        addiu $k0, $k0, 0x40     # the odd page: PFN + 1
        mtc0  $k0, $3            # EntryLo1
        mtc0  $zero, $5          # PageMask: 4 KB pages
        sll   $zero, $zero, 3    # ehb (Release 1: a nop)
        tlbwr                    # a random entry (Wired .. N-1)
        eret
"""
    return FIRMWARE_REFILL.replace(
        "xori  $k0, $k0, 0x08     # C = 3: cacheable, write-back",
        f"xori  $k0, $k0, 0x{(2 ^ cca) << 3:02x}     # C = {cca}: cacheable, write-back"
        + (", coherent" if cca in (4, 5) else ""))


FIRMWARE_REFILL = """\
        .set noreorder
        .ktext 0x80000000
__tlb_refill:
        mfc0  $k1, $10           # EntryHi: VPN2 of the miss | ASID
        srl   $k0, $k1, 13
        sll   $k0, $k0, 7        # EntryLo PFN of the even page: (VA >> 12) << 6
        bltz  $k1, __tlb_set     # VA >= 0x80000000: kseg2 / kseg3, PA = VA
        ori   $k0, $k0, 0x17     # (delay slot) C = 2 uncached, D, V, G
        lui   $k1, 0x0100        # kuseg: PA = VA + 0x40000000 (PFN + 0x40000)
        addu  $k0, $k0, $k1
        xori  $k0, $k0, 0x08     # C = 3: cacheable, write-back
__tlb_set:
        mtc0  $k0, $2            # EntryLo0: the even page
        addiu $k0, $k0, 0x40     # the odd page: PFN + 1
        mtc0  $k0, $3            # EntryLo1
        mtc0  $zero, $5          # PageMask: 4 KB pages
        sll   $zero, $zero, 3    # ehb (Release 1: a nop)
        tlbwr                    # a random entry (Wired .. N-1)
        eret
"""


#: The nanoMIPS boot firmware's refill handler (PLAN M23): FIRMWARE_REFILL
#: in nanoMIPS code — no delay slots (the uncached attribute is set before
#: the branch), BLTC for BLTZ, a real EHB; every instruction is in the
#: nanoMIPS subset (an NMS core runs it too)
FIRMWARE_REFILL_NANO = """\
        .ktext
__tlb_refill:
        mfc0  $k1, $10           # EntryHi: VPN2 of the miss | ASID
        srl   $k0, $k1, 13
        sll   $k0, $k0, 7        # EntryLo PFN of the even page: (VA >> 12) << 6
        ori   $k0, $k0, 0x17     # C = 2 uncached, D, V, G
        bltc  $k1, $zero, __tlb_set   # VA >= 0x80000000: kseg2 / kseg3, PA = VA
        li    $k1, 0x01000000    # kuseg: PA = VA + 0x40000000 (PFN + 0x40000)
        addu  $k0, $k0, $k1
        xori  $k0, $k0, 0x08     # C = 3: cacheable, write-back
__tlb_set:
        mtc0  $k0, $2            # EntryLo0: the even page
        addiu $k0, $k0, 0x40     # the odd page: PFN + 1
        mtc0  $k0, $3            # EntryLo1
        mtc0  $zero, $5          # PageMask: 4 KB pages
        ehb
        tlbwr                    # a random entry (Wired .. N-1)
        eret
"""


def _nano_refill(cca: int, window: tuple[int, int] | None) -> str:
    if window is None:
        return FIRMWARE_REFILL_NANO.replace(
            "xori  $k0, $k0, 0x08     # C = 3: cacheable, write-back",
            f"xori  $k0, $k0, 0x{(2 ^ cca) << 3:02x}     # C = {cca}: cacheable, write-back"
            + (", coherent" if cca in (4, 5) else ""))
    base, size = window
    mask = (size - 1) & ~0x1FFF
    lo = 0x07 | cca << 3                        # C, D, V, G
    return f"""\
        .ktext
__tlb_refill:
        mfc0  $k1, $10           # EntryHi: VPN2 of the miss | ASID
        srl   $k0, $k1, 13
        sll   $k0, $k0, 7
        ori   $k0, $k0, 0x17     # kseg2 / kseg3: PA = VA, C = 2 uncached, D, V, G
        bltc  $k1, $zero, __tlb_set
        li    $k0, 0x{mask:08x}  # the window: VA mod 0x{size:x}
        and   $k0, $k0, $k1      # the page pair's offset in the window
        li    $k1, 0x{base:08x}  # + its base
        addu  $k0, $k0, $k1
        srl   $k0, $k0, 12
        sll   $k0, $k0, 6        # EntryLo PFN of the even page
        ori   $k0, $k0, 0x{lo:02x}     # C = {cca}, D, V, G
__tlb_set:
        mtc0  $k0, $2            # EntryLo0: the even page
        addiu $k0, $k0, 0x40     # the odd page: PFN + 1
        mtc0  $k0, $3            # EntryLo1
        mtc0  $zero, $5          # PageMask: 4 KB pages
        ehb
        tlbwr                    # a random entry (Wired .. N-1)
        eret
"""


#: The MIPS64 boot firmware's XTLB refill handler (EBase + 0x080, Vol. III
#: Table 6.9: a refill of a reference with 64-bit addresses enabled): the
#: same mapping by the low word of the address — R = 0b11 (the kernel
#: regions, bit 63 set): PA = VA 31:0, uncached; else PA = VA 31:0 + 1 GB,
#: cacheable (firmware_map64).
FIRMWARE_XTLB_REFILL = """\
        .set noreorder
        .ktext 0x80000080
__xtlb_refill:
        dmfc0  $k1, $10          # EntryHi: R | VPN2 of the miss | ASID
        dsll32 $k0, $k1, 0       # its low word ...
        dsrl32 $k0, $k0, 13      # ... VA 31:13
        dsll   $k0, $k0, 7       # EntryLo PFN of the even page: (VA 31:12) << 6
        bltz   $k1, __xtlb_set   # R = 0b11 (bit 63): PA = VA 31:0
        ori    $k0, $k0, 0x17    # (delay slot) C = 2 uncached, D, V, G
        lui    $k1, 0x0100       # else PA = VA 31:0 + 0x40000000
        addu   $k0, $k0, $k1
        xori   $k0, $k0, 0x08    # C = 3: cacheable, write-back
__xtlb_set:
        mtc0   $k0, $2           # EntryLo0: the even page
        addiu  $k0, $k0, 0x40    # the odd page: PFN + 1
        mtc0   $k0, $3           # EntryLo1
        mtc0   $zero, $5         # PageMask: 4 KB pages
        sll    $zero, $zero, 3   # ehb (Release 1: a nop)
        tlbwr                    # a random entry (Wired .. N-1)
        eret
"""


def firmware_xtlb_refill(cca: int = 3, window: tuple[int, int] | None = None) -> str:
    """:data:`FIRMWARE_XTLB_REFILL` with attribute ``cca`` for the
    cacheable pages; ``window`` (a board's memory for the user pages, as
    :func:`firmware_refill`'s): PA = base + (VA 31:0 mod size)."""
    if window is not None:
        base, size = window
        mask = (size - 1) & ~0x1FFF
        lo = 0x07 | cca << 3                        # C, D, V, G
        return f"""\
        .set noreorder
        .ktext 0x80000080
__xtlb_refill:
        dmfc0  $k1, $10          # EntryHi: R | VPN2 of the miss | ASID
        bltz   $k1, __xtlb_k     # R = 0b11 (bit 63): PA = VA 31:0, uncached
        lui    $k0, 0x{mask >> 16:04x}       # (delay slot) the window: VA 31:0 mod 0x{size:x}
        ori    $k0, $k0, 0x{mask & 0xFFFF:04x}
        and    $k0, $k0, $k1     # the page pair's offset in the window
        lui    $k1, 0x{base >> 16:04x}       # + its base 0x{base:08x}
        addu   $k0, $k0, $k1
        srl    $k0, $k0, 12
        sll    $k0, $k0, 6       # EntryLo PFN of the even page
        b      __xtlb_set
        ori    $k0, $k0, 0x{lo:02x}    # (delay slot) C = {cca}, D, V, G
__xtlb_k:
        dsll32 $k0, $k1, 0       # its low word ...
        dsrl32 $k0, $k0, 13      # ... VA 31:13
        dsll   $k0, $k0, 7       # EntryLo PFN of the even page
        ori    $k0, $k0, 0x17    # C = 2 uncached, D, V, G
__xtlb_set:
        mtc0   $k0, $2           # EntryLo0: the even page
        addiu  $k0, $k0, 0x40    # the odd page: PFN + 1
        mtc0   $k0, $3           # EntryLo1
        mtc0   $zero, $5         # PageMask: 4 KB pages
        sll    $zero, $zero, 3   # ehb (Release 1: a nop)
        tlbwr                    # a random entry (Wired .. N-1)
        eret
"""
    return FIRMWARE_XTLB_REFILL.replace(
        "xori   $k0, $k0, 0x08    # C = 3: cacheable, write-back",
        f"xori   $k0, $k0, 0x{(2 ^ cca) << 3:02x}    # C = {cca}: cacheable, write-back")


def firmware_map64(va: int, cca: int = 3, window: tuple[int, int] | None = None
                   ) -> tuple[int, int, str]:
    """What the MIPS64 boot firmware's refill handlers map ``va`` to: bit
    63 set (the kernel regions: kseg2 / kseg3, xkseg): PA = VA 31:0,
    uncached; else PA = VA 31:0 + 1 GB (``window``: base + VA 31:0 mod
    size), cacheable."""
    if va >> 63:
        return va & M32, CCA_UNCACHED, "boot firmware refill handler: PA = VA 31:0, uncached"
    if window is not None:
        base, size = window
        return (base + (va & (size - 1)), cca,
                f"boot firmware refill handler: PA = 0x{base:08x} + (VA 31:0 mod 0x{size:x}), "
                "cacheable" + (" coherent" if cca in (4, 5) else ""))
    return (((va & M32) + 0x40000000) & M32, cca,
            "boot firmware refill handler: PA = VA 31:0 + 0x40000000, cacheable")


def segment64(va: int) -> str:
    """The MIPS64 segment of a 64-bit virtual address (Vol. III Table
    4.1): the 32-bit compatibility ones by their names, else xuseg /
    xsseg / xkphys / xkseg."""
    if va < 0x80000000 or va >= 0xFFFFFFFF80000000:
        return segment(va & M32)
    return ("xuseg", "xsseg", "xkphys", "xkseg")[va >> 62]


def user_window(m) -> tuple[int, int] | None:
    """Where the boot firmware maps kuseg on ``m``'s board (None: the flat
    board, VA + 1 GB)."""
    board = getattr(m, "board", None)
    return board.user_window() if board is not None else None


def boot_cca(m) -> int:
    """The attribute the boot firmware gives cacheable memory: 3, or 5
    (cacheable coherent) on a core of a CPS (Boot-MIPS: CCA 5 for coherent
    cores)."""
    return 5 if getattr(m, "system", None) is not None else 3


def firmware_map(va: int, cca: int = 3, window: tuple[int, int] | None = None
                 ) -> tuple[int, int, str]:
    """What :data:`FIRMWARE_REFILL` maps ``va`` to: ``(pa, cca, how)``
    (``window``: see :func:`firmware_refill`)."""
    if va < 0x80000000 and window is not None:
        base, size = window
        return (base + (va & (size - 1)), cca,
                f"boot firmware refill handler: PA = 0x{base:08x} + (VA mod 0x{size:x}), "
                "cacheable" + (" coherent" if cca in (4, 5) else ""))
    if va < 0x80000000:
        return ((va + 0x40000000) & M32, cca,
                "boot firmware refill handler: PA = VA + 0x40000000, cacheable"
                + (" coherent" if cca in (4, 5) else ""))
    return va, CCA_UNCACHED, "boot firmware refill handler: PA = VA, uncached"


#: FTLB page sizes (Config4.FTLBPageSize code -> bytes) eMIPS implements:
#: 4 KB and 16 KB, like the P5600 ("up to 16 KB per entry", MD01024)
FTLB_PAGE_SIZES = {1: 4096, 2: 16384}
#: Config6 (P5600, MD01025 Table 2.9): FTLBEn (15) enables the FTLB, FTLBP
#: (17:16) the share of TLBWRs of the FTLB page size that go to the VTLB
CONFIG6 = (16, 6)
CFG6_FTLBEN = 1 << 15
FTLBP_PERIOD = {0: 0, 1: 16, 2: 8, 3: 4}           # 1 in N goes to the VTLB


class TLBMMU:
    """TLB-based MMU with ``entries`` fully associative dual entries and
    the 24K micro-TLBs: ``Config.MT = 1`` (a JTLB), or ``MT = 4`` with an
    ``ftlb = (sets, ways)`` fixed-page-size TLB beside it (the entries are
    then the VTLB; Vol. III A.3).  ``sc`` adds Segmentation Control and
    EVA (Release 3+, :mod:`emips.segctl`), ``pw`` the hardware page table
    walker (Release 5+, :mod:`emips.ptw`).  Bind it to a machine by passing
    it as ``Machine(mmu=TLBMMU(...))``."""

    name = "TLB"
    supervisor = True

    def __init__(self, entries: int = 32, *, ftlb: tuple[int, int] | None = None,
                 sc: bool = False, pw: bool = False, rxi: bool | None = None,
                 small_pages: bool | None = None, invalidate: bool | None = None,
                 bad_instr: bool | None = None, itlb: int = 4, dtlb: int = 8,
                 seed: int = 0x7EB) -> None:
        if not 1 <= entries <= 64:
            raise ValueError(f"the TLB has 1..64 entries, not {entries}")
        if ftlb is not None:
            sets, ways = ftlb
            if sets < 1 or sets & (sets - 1) or sets > 32768 or not 2 <= ways <= 8:
                raise ValueError(f"FTLB sets must be a power of 2 (1..32768) and ways "
                                 f"2..8, not {sets} x {ways}")
        self.n = entries                    # the JTLB / VTLB
        self.ftlb = ftlb
        self.mt = 4 if ftlb else 1
        if ftlb:
            self.name = "VTLB + FTLB"
        self.fsets, self.fways = ftlb or (0, 0)
        self.total = self.n + self.fsets * self.fways
        #: Index / TLBP width with the FTLB on: 6 bits, more for a big FTLB
        self._index_mask = (1 << max(6, (self.total - 1).bit_length())) - 1
        #: the P5600's VTLB Random (MD01025 3.11): a 10-bit LFSR stepped by
        #: every VTLB write; TLBWRs of the FTLB page size (Config6.FTLBP)
        self.vtlb_lfsr = 1
        self.ftlbp_count = 0
        #: Config4.FTLBPageSize (kept while the FTLB is off)
        self.ftlb_ps = 1
        self._options = (rxi, small_pages, invalidate, bad_instr, sc, pw)
        self._features("R5")
        rng = _random.Random(seed)
        #: the TLB (VTLB first, then the FTLB way by way): contents
        #: undefined after reset (random from a fixed seed), the hidden
        #: written bit clear
        self.entries = [TLBEntry(pagemask=rng.choice((0, 0x6000, 0x1FE000)),
                                 hi=rng.getrandbits(32) & ~0x7FF,
                                 g=bool(rng.getrandbits(1)),
                                 lo=[rng.getrandbits(26) & ~1, rng.getrandbits(26) & ~1])
                        for _ in range(self.total)]
        self.itlb = MicroTLB("ITLB", itlb, lru=True)
        self.dtlb = MicroTLB("DTLB", dtlb, lru=False)
        self.random = entries - 1           # the hidden counter behind Random
        self.lfsr = 0                       # position in the LFSR period
        #: the boot firmware's mapping (its refill handler, for the views
        #: and the loader): None, or a function va -> (pa, cca, how)
        self.fallback = None
        #: micro-TLB timing for the pipeline (set by it)
        self.timing = False
        self.utlb_cost = UTLB_MISS
        self.wait = 0
        self.notes: list[str] = []
        self.m = None
        #: what the last page walk did (for the views)
        self.last_walk = ""
        #: tracing: the translations recorded since watch() (TLBAccess)
        self.log: list[TLBAccess] | None = None
        self.stats = TLBStats()
        self._memo: dict[tuple[int, int], int] = {}
        #: a VPE of an MT processor (emips.mt): a write that would make two
        #: valid entries match is dropped, no Machine Check (MD00378 6.25.3)
        self.mt_drop = False
        #: an MT processor (MD00534 5.4.4): Index.P is writable (software
        #: "parks" its Index); with MVPControl.STLB the other VPE's MMU,
        #: whose Index entry a TLBWR here never overwrites
        self.index_p = False
        self.peer: "TLBMMU | None" = None
        #: the other VPs' MMUs of a Release 6 core (emips.vp): one FTLB for
        #: the core, each VP its own VTLB (MD01196 12.2)
        self.ftlb_peers: list["TLBMMU"] = []
        self._width(False)

    #: translates 64-bit virtual addresses on a MIPS64 core (Config.AT = 2)
    va64 = True

    def _width(self, wide: bool) -> None:
        """The address width of the core it is bound to: MIPS32, or MIPS64
        with 64-bit addresses (R, VPN2 up to SEGBITS - 1, XContext)."""
        self.wide = wide
        vpn = ((1 << SEGBITS) - 1) & ~0x1FFF if wide else HI_VPN2
        #: EntryHi's R / VPN2 bits; the bits a TLB match compares
        self._hi_vpn = (HI_R | vpn) if wide else HI_VPN2
        self._vmask = self._hi_vpn | HI_VPN2X
        #: the virtual address width the TLB sees
        self._vamask = M64 if wide else M32
        for e in self.entries:
            e.vmask = self._vmask

    def _features(self, release: str) -> None:
        """The optional features, by default those of ``release``."""
        rxi, small_pages, invalidate, bad_instr, sc, pw = self._options
        r3 = release in ("R3", "R5", "R6")
        self.release = release
        #: RI / XI page protection (Config3.RXI; Release 3+)
        self.rxi = r3 if rxi is None else rxi or release == "R6"
        #: 1 KB pages (Config3.SP; Release 2-5, removed in Release 6; not
        #: with an FTLB, whose entries share one page size)
        self.small_pages = ((release in ("R2", "R3", "R5")) if small_pages is None
                            else small_pages and release != "R6") and not self.ftlb
        #: Segmentation Control and EVA (Config3.SC, Config5.EVA; Release 3+)
        self.sc = sc and r3
        self.eva = self.sc
        #: the hardware page table walker (Config3.PW; Release 5+)
        self.pw = pw and release in ("R5", "R6")
        #: EntryHi.EHINV, TLBINV, TLBINVF (Config4.IE = 3; Release 3+;
        #: required with an FTLB or Segmentation Control)
        self.invalidate = (r3 if invalidate is None else invalidate or release == "R6") \
            or bool(self.ftlb) or self.sc
        #: BadInstr / BadInstrP (Config3.BI / BP; Release 3+, required in 6)
        self.bad_instr = r3 if bad_instr is None else bad_instr or release == "R6"
        #: PageGrain exists (Release 2+ with 1 KB pages or RI / XI)
        self.pagegrain = release != "R1" and (self.rxi or self.small_pages)
        regs = set(REGISTERS)
        if self.ftlb:
            regs |= {CONFIG4, CONFIG6}      # FTLBPageSize, FTLBEn / FTLBP
        if self.sc:
            regs |= {segctl.SEGCTL0, segctl.SEGCTL1, segctl.SEGCTL2, segctl.CONFIG5, C.EBASE}
        if self.pw:
            regs |= ptw.REGISTERS
        #: the registers whose MTC0 goes through :meth:`cp0_write`
        self.REGISTERS = frozenset(regs)
        #: Machine.address_ok: the segment configurations, else fixed
        self.address_ok = self.segment_ok if self.sc else None

    # -- binding / reset ----------------------------------------------------------

    def attach(self, m) -> None:
        """Set the CP0 configuration of a machine using this MMU (the
        optional features follow its release)."""
        self.m = m
        self._features(m.release)
        wide = bool(getattr(m, "wide", False))
        self._width(wide)
        if wide:                            # XContext (20.0): the XTLB refill's pointer
            self.REGISTERS = self.REGISTERS | {XCONTEXT}
            m.cp0.setdefault(XCONTEXT, {"name": "XContext", "value": 0})
            m.cp0[XCONTEXT]["value"] = 0
        self.random = self.n - 1
        cp = m.cp0
        if self.ftlb:                       # Config6: the FTLB off after reset
            cp.setdefault(CONFIG6, {"name": "Config6", "value": 0})
            cp[CONFIG6]["value"] = 0
        config = cp[C.CONFIG]["value"] & ~((7 << 28) | (7 << 25) | (7 << 7))
        cp[C.CONFIG]["value"] = config | (self.mt << 7)
        cp[C.CONFIG1]["value"] = ((cp[C.CONFIG1]["value"] & ~(0x3F << 25))
                                  | (((self.n - 1) & 0x3F) << 25))
        cfg3 = cp[C.CONFIG3]["value"]
        if self.rxi:
            cfg3 |= CFG3_RXI
        if self.small_pages:
            cfg3 |= CFG3_SP
        if self.bad_instr:
            cfg3 |= CFG3_BI | CFG3_BP
        if self.sc:
            cfg3 |= segctl.CFG3_SC
        if self.pw:
            cfg3 |= ptw.CFG3_PW
        cfg4 = 0
        if self.invalidate:                 # IE = 3: TLBINV* on the whole TLB; with
            cfg4 |= (2 if self.ftlb else 3) << 29   # an FTLB 2: per Index (MD01025)
        self.ftlb_ps = 1
        if self.ftlb:                       # the P5600: Config5 present, three
            cfg4 |= (1 << 31) | (0x1C << 16)  # KScratch registers (31.2-31.4)
        if self.sc:
            cfg4 |= 1 << 31                 # Config5 present
            cp[segctl.CONFIG5]["value"] = segctl.CFG5_EVA
            for key, value in zip((segctl.SEGCTL0, segctl.SEGCTL1, segctl.SEGCTL2),
                                  segctl.RESET):
                cp[key]["value"] = value
        if cfg4:
            cfg3 |= CFG3_M                  # Config4 present
            cp[CONFIG4]["value"] = cfg4
        cp[C.CONFIG3]["value"] = cfg3
        if self.ftlb:
            self._ftlb_config(m)
        if self.pw:
            ptw.reset(m, self.release, p5600=bool(self.ftlb))
        cp[RANDOM]["value"] = 0 if self.release == "R6" else self.random
        cp[WIRED]["value"] = 0
        pg = 0
        if self.release == "R6":            # RIE / XIE read-only 1, IEC resets to 1
            pg = PG_RIE | PG_XIE | PG_IEC
        cp[PAGEGRAIN]["value"] = pg if self.pagegrain else 0
        for key in (INDEX, ENTRYLO0, ENTRYLO1, CONTEXT, PAGEMASK, ENTRYHI,
                    BADINSTR, BADINSTRP):
            cp[key]["value"] = 0
        # MTC0 masks: Config.K0 only (KU / K23 are fixed-mapping fields)
        m._wmask[C.CONFIG] = m._wmask.get(C.CONFIG, 7) & ~((7 << 28) | (7 << 25))
        m._wmask[C.STATUS] = m._wmask[C.STATUS] | (3 << C.ST_KSU_SHIFT) | ST_TS
        for key in (C.CONFIG1, C.CONFIG3, CONFIG4, segctl.CONFIG5):
            m._wmask[key] = 0

    # -- CP0 ------------------------------------------------------------------------

    @property
    def _esp(self) -> bool:
        return self.small_pages and bool(self.m.cp0[PAGEGRAIN]["value"] & PG_ESP)

    @property
    def ftlb_on(self) -> bool:
        """The FTLB is in use: present and Config6.FTLBEn (MD01025 3.3.1)."""
        return bool(self.ftlb) and self.m is not None \
            and bool(self.m.cp0[CONFIG6]["value"] & CFG6_FTLBEN)

    @property
    def index_mask(self) -> int:
        """Index / TLBP width: 6 bits without the FTLB (MD01025 2.2.1)."""
        return self._index_mask if self.ftlb_on else 0x3F

    @property
    def active_total(self) -> int:
        """The entries in use: the VTLB, and the FTLB while enabled."""
        return self.total if self.ftlb_on else self.n

    def _ftlb_config(self, m) -> None:
        """Config.MT and Config4 follow Config6.FTLBEn (MD01025 3.3.1-3.3.3):
        MT = 4, MMUExtDef = 3 and the FTLB fields while enabled; MT = 1,
        MMUExtDef = 1 and ExtVTLB = 0 while disabled."""
        cp = m.cp0
        on = bool(cp[CONFIG6]["value"] & CFG6_FTLBEN)
        cp[C.CONFIG]["value"] = (cp[C.CONFIG]["value"] & ~(7 << 7)) | ((4 if on else 1) << 7)
        cfg4 = cp[CONFIG4]["value"] & ~0xFFFF
        if self.release != "R6":
            cfg4 |= (3 if on else 1) << 14
        if on:
            cfg4 |= (self.ftlb_ps << 8) | ((self.fways - 2) << 4) | (self.fsets.bit_length() - 1)
        cp[CONFIG4]["value"] = cfg4
        self.mt = 4 if on else 1
        self.name = "VTLB + FTLB" if on else "VTLB (FTLB off)"
        self._changed()

    def _ix(self, value: int) -> int:
        """The entry an Index / Random value selects (out of range is
        UNDEFINED: the low address bits pick one)."""
        value &= self.index_mask
        return value % self.active_total

    @property
    def ftlb_page(self) -> int:
        """The page size of the FTLB entries (Config4.FTLBPageSize)."""
        return FTLB_PAGE_SIZES.get(self.ftlb_ps, 4096)

    @property
    def ftlb_pagemask(self) -> int:
        return ((2 * self.ftlb_page - 1) & PM_MASK)

    def ftlb_set(self, va: int) -> int:
        """The FTLB set of a virtual address (MD01025 3.13): the XOR of the
        VPN2 bits just above the page pair and the next ones — VA[19:13] ^
        VA[26:20] for 128 sets of 4 KB pages, VA[21:15] ^ VA[28:22] for
        16 KB (other set counts: the same rule, as many bits)."""
        bits = self.fsets.bit_length() - 1
        lo = va >> self.ftlb_page.bit_length()
        return (lo ^ (lo >> bits)) & (self.fsets - 1)

    def ftlb_index(self, way: int, s: int) -> int:
        """Index of an FTLB entry: ``{way, set} + VIndex`` (A.3.2)."""
        return self.n + way * self.fsets + s

    def cp0_write(self, m, key: tuple[int, int], value: int) -> None:
        """MTC0 to a register of the TLB MMU (masks and side effects)."""
        cell = m.cp0[key]
        old = cell["value"]
        r6 = self.release == "R6"
        if key == INDEX:
            ix = value & self.index_mask
            if r6 and ix >= self.total:      # R6: an out-of-range Index is ignored
                ix = old & self.index_mask
            p = old & INDEX_P
            if r6 and value & INDEX_P:       # R6: software may set P (not clear it)
                p = INDEX_P
            if self.index_p:                 # an MT processor: P is R/W
                p = value & INDEX_P
            new = p | ix
        elif key in (ENTRYLO0, ENTRYLO1):
            pg = m.cp0[PAGEGRAIN]["value"]
            mask = (0x3FFFFF if self._esp else 0xFFFFF) << 6 | 0x3F
            if self.rxi and pg & PG_RIE:
                mask |= LO_RI
            if self.rxi and pg & PG_XIE:
                mask |= LO_XI
            new = value & mask
        elif key == CONTEXT:                 # (MIPS64: PTEBase 63:23)
            base = (M64 & ~0x7FFFFF) if self.wide else CTX_PTEBASE
            new = (old & ~base) | (value & base)
        elif key == XCONTEXT:                # PTEBase 63:SEGBITS-7 (Vol. III 9.52)
            base = M64 & ~((1 << (SEGBITS - 7)) - 1)
            new = (old & ~base) | (value & base)
        elif key == PAGEMASK:
            new = self._pagemask(value, old)
        elif key == PAGEGRAIN:
            if not self.pagegrain:
                new = 0
            else:
                mask = 0
                if self.rxi and not r6:
                    mask |= PG_RIE | PG_XIE
                if self.rxi:
                    mask |= PG_IEC
                if self.small_pages:
                    mask |= PG_ESP
                new = (old & ~mask) | (value & mask)
        elif key == WIRED:
            new = value & self.index_mask
            if self.ftlb and new > self.n - 1:
                new = old                    # only VTLB entries can be wired (A.3.4)
            self.random = self.n - 1         # writing Wired resets Random
            m.cp0[RANDOM]["value"] = 0 if r6 else self.random
        elif key == CONFIG6:                 # FTLBP 17:16, FTLBEn 15 (MD01025 2.9)
            new = value & ((3 << 16) | CFG6_FTLBEN)
            if (new ^ old) & CFG6_FTLBEN:
                cell["value"] = new
                self._ftlb_config(m)
        elif key == ENTRYHI:                 # (MIPS64: R | VPN2; Fill reads 0)
            mask = self._hi_vpn | HI_ASID
            if self._esp:
                mask |= HI_VPN2X
            if self.invalidate:
                mask |= HI_EHINV
            new = value & mask
        elif key == C.STATUS:
            mask = m._wmask[C.STATUS]
            new = (old & ~mask) | (value & mask)
            if new & ST_TS and not old & ST_TS:
                new &= ~ST_TS                # TS: software can only clear it
        elif key == CONFIG4:                 # FTLBPageSize: 4 KB or 16 KB
            size = (value >> 8) & 0x1F
            if self.ftlb_on and size in FTLB_PAGE_SIZES and size != self.ftlb_ps:
                self.ftlb_ps = size          # (with the FTLB off: ExtVTLB, read-only)
                self._changed()
            new = (old & ~(0x1F << 8)) | (self.ftlb_ps << 8) if self.ftlb_on else old
        elif key == segctl.CONFIG5:          # K, CV writable; EVA read-only
            mask = segctl.CFG5_K | segctl.CFG5_CV
            new = (old & ~mask) | (value & mask)
        elif key in (segctl.SEGCTL0, segctl.SEGCTL1, segctl.SEGCTL2):
            new = value & 0xFE7FFE7F         # bits 8:7 of each CFG read 0
            if self.wide:                    # MIPS64 (Vol. III 9.14 / 9.15): SegCtl1.XAM
                new |= value & {segctl.SEGCTL1: segctl.XAM_MASK,     # 61:59,
                                segctl.SEGCTL2: segctl.XR_MASK}.get(key, 0)   # SegCtl2.XR
            if r6:                           # R6: unsupported C values ignored
                for shift in (0, 16):
                    if (value >> shift) & 7 not in m.supported_cca:
                        new = (new & ~(7 << shift)) | (old & (7 << shift))
            self._memo.clear()
        elif key == C.EBASE:                 # the write gate (Vol. III 9.41; MIPS64
            mask = 0x3FFFF000 | (1 << 11)    # 9.39: with WG bits 63:30)
            if value & (1 << 11):
                mask |= (M64 if self.wide else M32) & ~0x3FFFFFFF
            new = (old & ~mask) | (value & mask)
        elif key in ptw.REGISTERS:
            new = ptw.write(m, key, value, self.release)
        else:                                # Random, BadInstr, BadInstrP: read-only
            new = old
        cell["value"] = new & (M64 if self.wide and key in WIDE_KEYS else M32)

    def _pagemask(self, value: int, old: int) -> int:
        """PageMask: the 24K stores one bit per pair of Mask bits (MD00343
        4.3.1; the upper bit of each pair is kept); MaskX only with 1 KB
        pages enabled; bits above 28 (256 MB) read 0.  Release 6 ignores a
        value that is not a legal page size."""
        if self.release == "R6":
            legal = {(2 * size - 1) & PM_MASK for size in PAGE_SIZES}
            return value & PM_MASK if value & PM_MASK in legal else old
        out = 0
        for bit in range(13, 29, 2):
            if value >> (bit + 1) & 1:
                out |= 3 << bit
        if self._esp and value >> 12 & 1:
            out |= PM_MASKX
        return out

    def skipped(self, cycles: int) -> int:
        """How many of the next ``cycles`` cycles the LFSR skips the
        decrement of Random in (:data:`LFSR_SKIPS`); advances the LFSR."""
        p = self.lfsr
        full, rem = divmod(cycles, 1023)
        if p + rem <= 1023:
            part = _LFSR_CUM[p + rem] - _LFSR_CUM[p]
        else:
            part = _LFSR_CUM[1023] - _LFSR_CUM[p] + _LFSR_CUM[p + rem - 1023]
        self.lfsr = (p + cycles) % 1023
        return full * _LFSR_CUM[1023] + part

    def tick(self, m, cycles: int) -> None:
        """``cycles`` clock cycles pass: Random counts down from N-1 to
        Wired and wraps — a clock-rate down-counter whose decrement a
        10-bit LFSR skips now and then (MD00343 6.2.2), so a refill handler
        of a fixed length does not pick the same entry every time.  The P5600's
        Random (an FTLB MMU) does not follow the clock (MD01025 3.11:
        "modified after a VTLB write, or on a write to the Wired register",
        :meth:`_vtlb_random`)."""
        if self.ftlb:
            return
        wired = m.cp0[WIRED]["value"] & INDEX_BITS
        top = self.n - 1
        steps = cycles - self.skipped(cycles)
        if wired > top:
            self.random = top
        else:
            span = top - wired + 1
            self.random = wired + (self.random - wired - steps) % span
        if self.release != "R6":
            m.cp0[RANDOM]["value"] = self.random

    # -- segments -------------------------------------------------------------------

    def segment_ok(self, m, va: int) -> bool:
        """Segmentation Control: may this access use ``va`` (the AM of its
        Segment Configuration, Table 9.28; user rights for EVA)?"""
        _, am, _, _ = segctl.fields(segctl.cfg(m, segctl.region(va)))
        return segctl.allowed(am, segctl.level(m))

    def _unmapped(self, m, va: int, lvl: str | None = None) -> tuple[int, int] | None:
        """``(pa, cca)`` when ``va`` is not translated by the TLB: kseg0 /
        kseg1 and kuseg with ERL = 1 (legacy), or the unmapped Segment
        Configurations (Segmentation Control, 4.10); else None.  MIPS64:
        the same compatibility segments (sign-extended), and xkphys — the
        attribute from VA 61:59, PA = VA PABITS-1:0 (Vol. III 4.7)."""
        status = m.cp0[C.STATUS]["value"]
        if self.wide and 0x7FFFFFFF < va < 0xFFFFFFFF80000000:
            if va >> 62 == 2:                # xkphys; Segmentation Control: a region
                if self.sc and m.cp0[segctl.SEGCTL2]["value"] >> (56 + ((va >> 59) & 7)) & 1:
                    am = (m.cp0[segctl.SEGCTL1]["value"] >> 59) & 7   # of SegCtl2.XR
                    if segctl.mapped(am, lvl or segctl.level(m)):     # by SegCtl1.XAM
                        return None          # (4.12 xkphysRegionLookup: mapped)
                return va & ((1 << PABITS) - 1), (va >> 59) & 7
            return None                      # xuseg / xsseg / xkseg: mapped
        va &= M32
        if self.sc:
            n = segctl.region(va)
            c = segctl.cfg(m, n)
            pa, am, eu, cca = segctl.fields(c)
            if eu and status & C.ST_ERL:
                return segctl.unmapped_pa(c, n, va), CCA_UNCACHED
            if segctl.mapped(am, lvl or segctl.level(m)):
                return None
            if n == 3 and not m.cp0[segctl.CONFIG5]["value"] & segctl.CFG5_K:
                cca = m.cp0[C.CONFIG]["value"] & 7   # Config5.K = 0: Config.K0
            return segctl.unmapped_pa(c, n, va), cca
        if 0x80000000 <= va < 0xC0000000:
            if va < 0xA0000000:
                return va & 0x1FFFFFFF, m.cp0[C.CONFIG]["value"] & 7
            return va & 0x1FFFFFFF, CCA_UNCACHED
        if va < 0x80000000 and status & C.ST_ERL:
            return va, CCA_UNCACHED
        return None

    # -- lookup -----------------------------------------------------------------------

    def match(self, va: int, asid: int, check: bool = False) -> int | None:
        """The TLB entry that maps ``va`` for ``asid`` (no side effects):
        the VTLB / JTLB fully associatively, the FTLB (while enabled) in the
        set of ``va``.  ``check``: a lookup / TLBP of the P5600 MMU that
        finds several entries takes a Machine Check (MD01025 3.17 / 8.8.5:
        duplicates can exist there, a write does not check them all)."""
        key = (va >> 10, asid)
        hit = self._memo.get(key)
        if hit is None:
            hits = [i for i in range(self.n) if self.entries[i].matches(va, asid)]
            if self.ftlb_on:
                s = self.ftlb_set(va)
                hits += [self.ftlb_index(w, s) for w in range(self.fways)
                         if self.entries[self.ftlb_index(w, s)].matches(va, asid)]
            hit = hits[0] if hits else -1
            if len(hits) > 1 and self.ftlb:
                hit = -2 - hit                    # several: the first, flagged
            if len(self._memo) > 4096:
                self._memo.clear()
            self._memo[key] = hit
        if hit < -1:
            if check:
                from .sim import _Arch
                raise _Arch(C.MCHECK, info={"mccause": 1, "why": f"several TLB entries "
                                            f"map 0x{va:08x}"})
            hit = -2 - hit
        return None if hit < 0 else hit

    def _changed(self, index: int | None = None) -> None:
        """A TLB entry was written or invalidated: the micro-TLBs drop
        their copies of it (MD00343 2.2.3 / 2.3.1); an FTLB entry of a core
        with VPs: every VP sees it."""
        self._memo.clear()
        self.itlb.drop(index)
        self.dtlb.drop(index)
        if self.ftlb_peers and (index is None or index >= self.n):
            shared = self.entries[self.n:]
            for p in self.ftlb_peers:
                p.entries[p.n:] = shared
                p._memo.clear()
                p.itlb.drop(index)
                p.dtlb.drop(index)

    def share_ftlb(self, others: list["TLBMMU"]) -> None:
        """This MMU joins the FTLB of ``others`` (the VPs of its core, MD01196
        12.2: "The 512 dual-entry Fixed TLB (FTLB) is shared between all
        VPs"): it takes their FTLB entries, they see its writes."""
        group = [o for o in others if o.ftlb and o.n == self.n
                 and len(o.entries) == len(self.entries)] if self.ftlb else []
        if group:
            self.entries[self.n:] = group[0].entries[self.n:]
            self._memo.clear()
        for o in group:
            o.ftlb_peers = [p for p in o.ftlb_peers if p in group] + [self]
        self.ftlb_peers = group

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        """Translate an access (``fetch``, ``load``, ``pcload`` — a PC-relative
        load, ``store``, ``cacheop``, ``pref``): ``(pa, cca)``, or raise the
        TLB exception.  Goes through the micro-TLB, filling it from the TLB
        on a miss (Vol. III 4.9.4 pseudo code; 24K 2.2.3, 2.3.1); a miss in
        the TLB first runs the page table walker when there is one."""
        va &= self._vamask
        fixed = self._unmapped(m, va)
        if fixed is not None:
            if kind != "fetch":
                self.stats.unmapped_data += 1
            return fixed
        asid = m.cp0[ENTRYHI]["value"] & HI_ASID
        utlb = self.itlb if kind == "fetch" else self.dtlb
        u = utlb.lookup(va, asid)
        rec = None
        if self.log is not None:
            rec = TLBAccess(kind, va, asid, utlb.name, u is not None, mmu=self)
            self.log.append(rec)
        if u is None:
            ix = self.match(va, asid, check=True)
            store = kind == "store"
            code = C.TLBS if store else C.TLBL
            if ix is None and self.pw and not m.cp0[C.STATUS]["value"] & C.ST_EXL:
                self.stats.walks += 1
                if ptw.walk(self, m, va):
                    self.stats.walked += 1
                    ix = self.match(va, asid, check=True)
                    if rec is not None:
                        rec.walk = self.last_walk
                elif rec is not None:
                    rec.walk = "page walk: no valid PTE"
            if ix is None:
                self.stats.misses += 1
                self.stats.jtlb_misses["fetch" if kind == "fetch" else "data"] += 1
                raise self._fault(rec, "refill", code, va, "no TLB entry matches", refill=True,
                                  xtlb=self._xtlb(m, va))
            self.stats.hits += 1
            e = self.entries[ix]
            half = (va >> e.evenodd) & 1
            lo = e.lo[half]
            if rec is not None:
                self._record(rec, ix, e, half)
            if not lo & LO_V:
                self.stats.jtlb_misses["fetch" if kind == "fetch" else "data"] += 1
                raise self._fault(rec, "invalid", code, va, f"the {('even', 'odd')[half]} "
                                  "page is not valid (V = 0)", ix=ix)
            sub = e.page_size if e.page_size < 4096 else (1 << 20 if e.page_size >= 1 << 20 else 4096)
            pa = e.frame(half) | (va & (e.page_size - 1))
            u = MicroEntry(va & ~(sub - 1), sub, asid, e.g, ix, pa & ~(sub - 1),
                           (lo >> LO_C_SHIFT) & 7, lo)
            self._protect(m, u, va, kind, rec)
            utlb.fill(u)
            if self.timing:
                self.wait += self.utlb_cost
                self.notes.append(f"{utlb.name} miss: TLB lookup, {self.utlb_cost} cycles")
        else:
            if rec is not None:
                e = self.entries[u.index]
                self._record(rec, u.index, e, (va >> e.evenodd) & 1)
            self._protect(m, u, va, kind, rec)
        pa = u.pbase | (va - u.vbase)
        if rec is not None:
            rec.pa, rec.cca = pa, u.cca
        return pa, u.cca

    @staticmethod
    def _record(rec: TLBAccess, ix: int, e: TLBEntry, half: int) -> None:
        """Note the entry ``ix`` and the half a translation used."""
        rec.entry, rec.source, rec.half = ix, e, half

    def _xtlb(self, m, va: int) -> bool:
        """A refill of ``va`` uses the XTLB Refill vector (0x080): 64-bit
        addresses are enabled for the reference (Vol. III 6.2.10, Table
        4.2) — useg / xuseg by UX, sseg / kseg3 / xkseg by KX (note 2:
        not SX), xsseg by SX."""
        if not self.wide:
            return False
        status = m.cp0[C.STATUS]["value"]
        if va < 0x80000000:
            return bool(status & (1 << 5))                       # UX
        if va >= 0xFFFFFFFF80000000:
            return bool(status & (1 << 7))                       # KX
        return True                         # xuseg / xsseg / xkseg (enabled to get here)

    def _fault(self, rec: TLBAccess | None, what: str, code: int, va: int, why: str,
               *, refill: bool = False, ix: int | None = None, xtlb: bool = False):
        """The exception of a failed translation (counted and recorded)."""
        name = FAULT_NAMES[what]
        self.stats.fault(name)
        if rec is not None:
            rec.fault, rec.why, rec.pa = name, why, None
        where = "" if ix is None else f"TLB entry {ix}: "
        return _arch(code, va, refill=refill, why=where + why, xtlb=xtlb)

    def _protect(self, m, u: MicroEntry, va: int, kind: str,
                 rec: TLBAccess | None = None) -> None:
        """RI / XI (Release 3) and D (TLB Modified) of a valid page."""
        lo = u.lo
        if self.rxi:
            iec = m.cp0[PAGEGRAIN]["value"] & PG_IEC
            if lo & LO_RI and kind == "load" or (kind == "pcload" and lo & LO_RI and lo & LO_XI):
                raise self._fault(rec, "ri" if iec else "invalid", C.TLBRI if iec else C.TLBL,
                                  va, "read inhibit (RI = 1)", ix=u.index)
            if lo & LO_XI and kind == "fetch":
                raise self._fault(rec, "xi" if iec else "invalid", C.TLBXI if iec else C.TLBL,
                                  va, "execute inhibit (XI = 1)", ix=u.index)
        if kind == "store" and not lo & LO_D:
            raise self._fault(rec, "mod", C.MOD, va, "the page is not writable (D = 0)",
                              ix=u.index)

    # -- recording (tracing) ------------------------------------------------------------

    def watch(self) -> None:
        """Start recording translations (:class:`TLBAccess`)."""
        self.log = []

    def taken(self) -> list[TLBAccess]:
        """The records since :meth:`watch`; stops recording."""
        out, self.log = self.log or [], None
        return out

    def reset_stats(self) -> None:
        self.stats = TLBStats()
        self.itlb.reset_counters()
        self.dtlb.reset_counters()

    def _note(self, kind: str, va: int, note: str, entry: int | None = None) -> None:
        if self.log is not None:
            self.log.append(TLBAccess(kind, va & self._vamask, entry=entry, note=note,
                                      mmu=self))

    def walker_translation(self, m, va: int) -> tuple[int, int] | None:
        """``(pa, cca)`` of a page table walker read (kernel privilege, no
        exception, no micro-TLB): unmapped, or a valid TLB entry; else None.
        MIPS64: an address Kernel Mode may not use (Table 4.2: KX / SX /
        UX, the holes) is None too (the walk is aborted, 4.14)."""
        va &= self._vamask
        if self.wide and not self._kernel_ok(m, va):
            return None
        fixed = self._unmapped(m, va, segctl.KERNEL)
        if fixed is not None:
            return fixed
        ix = self.match(va, m.cp0[ENTRYHI]["value"] & HI_ASID)
        if ix is None:
            return None
        e = self.entries[ix]
        half = (va >> e.evenodd) & 1
        if not e.lo[half] & LO_V:
            return None
        return (e.frame(half) | (va & (e.page_size - 1))) & M32, (e.lo[half] >> LO_C_SHIFT) & 7

    def _kernel_ok(self, m, va: int) -> bool:
        """MIPS64: may a Kernel Mode load use ``va`` (Vol. III Table 4.2;
        Segmentation Control's xkphys regions, 4.12)?"""
        from .sim64 import ST_KX, ST_SX, ST_UX, compat, hole
        if compat(va):
            return True                      # (every Segment Configuration lets kernel in)
        if hole(va, SEGBITS, PABITS):
            return False
        st = m.cp0[C.STATUS]["value"]
        top = va >> 62
        if top == 2 and self.sc and m.cp0[segctl.SEGCTL2]["value"] >> (56 + ((va >> 59) & 7)) & 1:
            am = (m.cp0[segctl.SEGCTL1]["value"] >> 59) & 7
            need = ST_KX if am in (segctl.UK, segctl.MK, 6) else \
                ST_SX if am in (segctl.MSK, segctl.USK) else ST_UX
            return bool(st & need)
        return bool(st & (ST_UX if top == 0 else ST_SX if top == 1 else ST_KX))

    def peek(self, m, va: int) -> tuple[int, int] | None:
        """``(pa, cca)`` for a view (no micro-TLB, no counter, no
        exception); ``None`` when no valid entry maps it (the boot
        firmware's mapping when it would refill it).  The same answer as
        :meth:`describe`, without its text."""
        va &= self._vamask
        fixed = self._unmapped(m, va)
        if fixed is not None:
            return fixed
        ix = self.match(va, m.cp0[ENTRYHI]["value"] & HI_ASID)
        if ix is None:
            return None if self.fallback is None else self.fallback(va)[:2]
        e = self.entries[ix]
        half = (va >> e.evenodd) & 1
        lo = e.lo[half]
        if not lo & LO_V:
            return None
        return (e.frame(half) | (va & (e.page_size - 1))) & M32, (lo >> LO_C_SHIFT) & 7

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        """How ``va`` translates now, as the views show it."""
        va &= self._vamask
        seg = segment64(va) if self.wide else segment(va)
        fixed = self._unmapped(m, va)
        if fixed is not None:
            pa, cca = fixed
            if seg == "xkphys":
                how = (f"unmapped: PA = VA {PABITS - 1}:0, attribute {cca} from VA 61:59")
            elif self.sc:
                n = segctl.region(va)
                how = f"unmapped by {segctl.describe(n, segctl.cfg(m, n))}"
            elif seg == "kuseg":
                how = "Status.ERL = 1: kuseg unmapped, PA = VA, uncached"
            elif seg in ("kseg0", "ckseg0"):
                how = "unmapped: PA = VA - 0x80000000 (Config.K0)"
            else:
                how = "unmapped: PA = VA - 0xA0000000, uncached"
            return Translation(va, pa, cca, seg, False, how)
        asid = m.cp0[ENTRYHI]["value"] & HI_ASID
        ix = self.match(va, asid)
        if ix is None:
            if self.fallback is not None:
                pa, cca, how = self.fallback(va)
                return Translation(va, pa, cca, seg, True,
                                   f"not in the TLB yet; {how}", entry=None)
            return Translation(va, None, CCA_UNCACHED, seg, True,
                               f"TLB miss: no entry maps VPN2 0x{va >> 13:05x} "
                               f"for ASID {asid}", fault="refill")
        e = self.entries[ix]
        half = (va >> e.evenodd) & 1
        lo = e.lo[half]
        cca = (lo >> LO_C_SHIFT) & 7
        name = self.entry_name(ix)
        if not lo & LO_V:
            return Translation(va, None, cca, seg, True,
                               f"{name}: {('even', 'odd')[half]} page "
                               "not valid (V = 0)", entry=ix, fault="invalid")
        pa = e.frame(half) | (va & (e.page_size - 1))
        flags = "".join(f for f, b in (("D", LO_D), ("RI", LO_RI), ("XI", LO_XI))
                        if lo & b)
        return Translation(va, pa & M32, cca, seg, True,
                           f"{name} ({'global' if e.g else f'ASID {e.asid}'}, "
                           f"{size_name(e.page_size)} {('even', 'odd')[half]} page"
                           f"{', ' + flags if flags else ''}): PFN -> PA 0x{pa & M32:08x}",
                           entry=ix)

    def entry_name(self, ix: int) -> str:
        """``TLB entry 5``, or ``VTLB entry 5`` / ``FTLB set 3 way 1 (entry 70)``."""
        if not self.ftlb:
            return f"TLB entry {ix}"
        if ix < self.n:
            return f"VTLB entry {ix}"
        w, s = divmod(ix - self.n, self.fsets)
        return f"FTLB set {s} way {w} (entry {ix})"

    # -- exceptions -----------------------------------------------------------------------

    def exception_entry(self, m, va: int) -> None:
        """TLB Refill / Invalid / Modified / RI / XI taken: Context.BadVPN2
        and EntryHi.VPN2 (and VPN2X with 1 KB pages) get the failing VA;
        the ASID stays (Vol. III 6.2.10, Table 6.x; MD00343 Table 5.14).
        EntryLo0 / EntryLo1 are UNPREDICTABLE: left as they are."""
        cp = m.cp0
        ctx = cp[CONTEXT]["value"]
        cp[CONTEXT]["value"] = (ctx & ~CTX_BADVPN2) | ((va >> 13) << 4 & CTX_BADVPN2)
        hi = cp[ENTRYHI]["value"]
        keep = HI_ASID | HI_EHINV
        vpn = va & (self._hi_vpn | (HI_VPN2X if self._esp else 0))
        cp[ENTRYHI]["value"] = (hi & keep) | vpn
        if self.wide:                        # MIPS64 (Vol. III 9.52): XContext.R =
            xc = cp[XCONTEXT]["value"]       # VA 63:62, BadVPN2 = VA SEGBITS-1:13
            low = (1 << (SEGBITS - 7)) - 1
            cp[XCONTEXT]["value"] = ((xc & ~low) | (va >> 62) << (SEGBITS - 9)
                                     | ((va >> 13) & ((1 << (SEGBITS - 13)) - 1)) << 4)

    def machine_check(self, m, cause: int = 1) -> None:
        """Machine Check taken: Status.TS, PageGrain.MCCause = ``cause``
        (1 multiple hit, 3 / 4 a bad FTLB write, 5 a page walk huge-page
        pair without PTEvld)."""
        m.cp0[C.STATUS]["value"] |= ST_TS
        if self.pagegrain:
            pg = m.cp0[PAGEGRAIN]["value"]
            m.cp0[PAGEGRAIN]["value"] = (pg & ~0x1F) | cause

    # -- TLB instructions ------------------------------------------------------------------

    def tlbp(self, m) -> None:
        """TLBP: Index = the entry matching EntryHi (VPN2, ASID), or P = 1
        (the Index field is then UNPREDICTABLE: unchanged)."""
        hi = m.cp0[ENTRYHI]["value"]
        ix = self.match(hi & self._vmask, hi & HI_ASID, check=True)
        what = f"TLBP: EntryHi VPN2 0x{hi >> 13:05x} ASID {hi & HI_ASID}"
        if ix is not None:
            m.cp0[INDEX]["value"] = ix
            self._note("probe", hi, f"{what} matches {self.entry_name(ix)}: Index = {ix}", ix)
            return
        m.cp0[INDEX]["value"] = INDEX_P | (m.cp0[INDEX]["value"] & self.index_mask)
        self._note("probe", hi, f"{what} matches no entry: Index.P = 1")

    def tlbr(self, m) -> None:
        """TLBR: PageMask, EntryHi, EntryLo0 / EntryLo1 from entry Index
        (G in both EntryLos; an invalidated entry reads as EHINV = 1)."""
        i = self._ix(m.cp0[INDEX]["value"])
        e = self.entries[i]
        cp = m.cp0
        self._note("read", 0, f"TLBR read {self.entry_name(i)}: {e.describe()}", i)
        if self.invalidate and e.ehinv:
            cp[PAGEMASK]["value"] = 0
            cp[ENTRYHI]["value"] = HI_EHINV
            cp[ENTRYLO0]["value"] = cp[ENTRYLO1]["value"] = 0
            return
        cp[PAGEMASK]["value"] = e.pagemask & (PM_MASK | (PM_MASKX if e.small else 0))
        cp[ENTRYHI]["value"] = e.hi & (self._hi_vpn | (HI_VPN2X if e.small else 0) | HI_ASID)
        g = LO_G if e.g else 0
        cp[ENTRYLO0]["value"] = e.lo[0] | g
        cp[ENTRYLO1]["value"] = e.lo[1] | g

    def tlbwi(self, m) -> None:
        """TLBWI: write (or, with EntryHi.EHINV = 1, invalidate) entry Index.
        An FTLB entry needs the FTLB page size in PageMask and an EntryHi
        of its set, else Machine Check (PageGrain.MCCause 3 / 4, A.3.3)."""
        i = self._ix(m.cp0[INDEX]["value"])
        cp = m.cp0
        if self.invalidate and cp[ENTRYHI]["value"] & HI_EHINV:
            self.entries[i].ehinv = True
            self._changed(i)
            self._note("invalidate", 0, f"TLBWI with EntryHi.EHINV = 1: "
                                        f"{self.entry_name(i)} invalidated", i)
            return
        if i >= self.n:
            from .sim import _Arch
            if cp[PAGEMASK]["value"] != self.ftlb_pagemask:
                raise _Arch(C.MCHECK, info={"mccause": 3, "why": f"TLBWI to FTLB entry {i} "
                                            "with a page size the FTLB does not use"})
            w, s = divmod(i - self.n, self.fsets)
            if self.ftlb_set(cp[ENTRYHI]["value"]) != s:
                raise _Arch(C.MCHECK, info={"mccause": 4, "why": f"TLBWI: EntryHi does not "
                                            f"belong to FTLB set {s}"})
        self._write(m, i, cp[PAGEMASK]["value"], cp[ENTRYHI]["value"],
                    cp[ENTRYLO0]["value"], cp[ENTRYLO1]["value"], "TLBWI")

    def tlbwr(self, m) -> None:
        """TLBWR: write entry Random — or, with the FTLB page size in
        PageMask, a way of the FTLB set of EntryHi (A.3.3)."""
        cp = m.cp0
        self.write_random(m, cp[PAGEMASK]["value"], cp[ENTRYHI]["value"],
                          cp[ENTRYLO0]["value"], cp[ENTRYLO1]["value"])

    def write_random(self, m, pagemask: int, hi: int, lo0: int, lo1: int,
                     how: str = "TLBWR") -> int:
        """What TLBWR (and the page table walker) does with these values:
        the FTLB when the page size is the FTLB's — the way picked by a
        cycle counter, as the example of A.3.2.1 —, else the VTLB / JTLB
        entry Random.  Returns the entry written."""
        if self.ftlb_on and pagemask == self.ftlb_pagemask and self._to_ftlb(m):
            i = self.ftlb_index(m.cycles % self.fways, self.ftlb_set(hi))
        else:
            i = self._ix(self.random)
            peer = self.peer.m.cp0[INDEX]["value"] if self.peer is not None else INDEX_P
            if not peer & INDEX_P and i == peer & self.index_mask:
                # a shared TLB: never the other VPE's Index entry — the 34K
                # stalls until Random moves on (MD00534 5.4.4): the next one
                wired = m.cp0[WIRED]["value"] & INDEX_BITS
                i = self._ix(self.n - 1 if i <= wired else i - 1)
        if how != "TLBWR":
            self.last_walk = f"{how}: wrote {self.entry_name(i)}"
        self._write(m, i, pagemask, hi, lo0, lo1, how)
        return i

    def _to_ftlb(self, m) -> bool:
        """Config6.FTLBP (MD01025 Table 2.9): 0 every TLBWR of the FTLB page
        size goes to the FTLB, 1 / 2 / 3 one in 16 / 8 / 4 to the VTLB."""
        period = FTLBP_PERIOD[(m.cp0[CONFIG6]["value"] >> 16) & 3]
        if not period:
            return True
        self.ftlbp_count = (self.ftlbp_count + 1) % period
        return self.ftlbp_count != 0

    def _vtlb_random(self, m) -> None:
        """The P5600's Random after a VTLB write (MD01025 3.11: "pseudo-
        random LRU"; the algorithm is not published): a 10-bit LFSR
        (x^10 + x^7 + 1) step, folded into Wired..N-1."""
        s = self.vtlb_lfsr
        s = ((s << 1) | (((s >> 9) ^ (s >> 6)) & 1)) & 0x3FF
        self.vtlb_lfsr = s
        wired = m.cp0[WIRED]["value"] & INDEX_BITS
        top = self.n - 1
        self.random = top if wired > top else wired + s % (top - wired + 1)
        if self.release != "R6":
            m.cp0[RANDOM]["value"] = self.random

    def _write(self, m, i: int, pagemask: int, hi: int, lo0: int, lo1: int,
               how: str = "TLBWI") -> None:
        new = TLBEntry(pagemask=pagemask, hi=hi & ~HI_EHINV,
                       g=bool(lo0 & lo1 & LO_G), lo=[lo0 & ~LO_G, lo1 & ~LO_G],
                       small=self._esp, written=True, vmask=self._vmask)
        if self.ftlb:
            self._write_p5600(m, i, new, hi, how)
            return
        clash = []
        for j, e in enumerate(self.entries):
            if j == i or not e.live:
                continue
            mask = e.mask | new.mask
            if (e.hi ^ new.hi) & self._vmask & ~mask:
                continue
            if not (e.g or new.g or e.asid == new.asid):
                continue
            if e.valid() and new.valid():   # MD00343 Table 4.9: abort the write
                from .sim import _Arch
                why = (f"{how} to {self.entry_name(i)} matches valid "
                       f"{self.entry_name(j)}")
                if self.mt_drop:            # MT: dropped, no Machine Check (MD00534 5.4.4)
                    self._note("write", hi, f"{why}: write dropped (MT Module)", i)
                    return
                self._note("write", hi, f"{why}: write aborted, Machine Check", i)
                raise _Arch(C.MCHECK, info={"why": why})
            clash.append(j)
        for j in clash:                      # matching entries are disabled
            self.entries[j].written = False
            self._changed(j)
        self.entries[i] = new
        self._changed(i)
        self.stats.writes += 1
        if self.log is not None:
            note = f"{how} wrote {self.entry_name(i)}: {new.describe()}"
            if clash:
                note += "; disabled (matched it): " + ", ".join(
                    self.entry_name(j) for j in clash)
            self._note("write", hi, note, i)

    def _write_p5600(self, m, i: int, new: TLBEntry, hi: int, how: str) -> None:
        """A write with an FTLB (MD01025 3.17): "no machine check is
        generated and the older entries are just invalidated" — searched in
        the VTLB and the new entry's FTLB set only (a large VTLB page may
        leave duplicates: a later lookup / TLBP takes a Machine Check)."""
        scope = list(range(self.n))
        if self.ftlb_on:
            s = self.ftlb_set(new.hi)
            scope += [self.ftlb_index(w, s) for w in range(self.fways)]
        gone = []
        for j in scope:
            e = self.entries[j]
            if j == i or not e.live:
                continue
            mask = e.mask | new.mask
            if (e.hi ^ new.hi) & self._vmask & ~mask:
                continue
            if not (e.g or new.g or e.asid == new.asid):
                continue
            e.ehinv = True
            gone.append(j)
            self._changed(j)
        self.entries[i] = new
        self._changed(i)
        self.stats.writes += 1
        if i < self.n:
            self._vtlb_random(m)
        if self.log is not None:
            note = f"{how} wrote {self.entry_name(i)}: {new.describe()}"
            if gone:
                note += "; invalidated (duplicates): " + ", ".join(
                    self.entry_name(j) for j in gone)
            self._note("write", hi, note, i)

    def _inv_scope(self, m) -> list[int]:
        """The entries TLBINV / TLBINVF act on: all (Config4.IE = 3), or with
        an FTLB (IE = 2, MD01025 TLBINV): the VTLB when Index is in it, else
        the FTLB set of Index."""
        if not self.ftlb:
            return list(range(self.total))
        i = self._ix(m.cp0[INDEX]["value"])
        if i < self.n:
            return list(range(self.n))
        s = (i - self.n) % self.fsets
        return [self.ftlb_index(w, s) for w in range(self.fways)]

    def _scope_name(self, scope: list[int]) -> str:
        if len(scope) == self.total:
            return "every entry"
        if scope[0] < self.n:
            return "the VTLB"
        return f"FTLB set {(scope[0] - self.n) % self.fsets}"

    def tlbinv(self, m) -> None:
        """TLBINV: invalidate every non-global entry of EntryHi.ASID (with an
        FTLB: in the VTLB or one FTLB set, by Index)."""
        asid = m.cp0[ENTRYHI]["value"] & HI_ASID
        n = 0
        scope = self._inv_scope(m)
        for j in scope:
            e = self.entries[j]
            if e.asid == asid and not e.g:
                n += e.live
                e.ehinv = True
        self._changed()
        self._note("invalidate", 0, f"TLBINV: the non-global entries of ASID {asid} "
                                    f"in {self._scope_name(scope)} invalidated "
                                    f"({n} were in use)")

    def tlbinvf(self, m) -> None:
        """TLBINVF: invalidate every entry (with an FTLB: of the VTLB or of
        one FTLB set, by Index)."""
        scope = self._inv_scope(m)
        for j in scope:
            self.entries[j].ehinv = True
        self._changed()
        self._note("invalidate", 0, f"TLBINVF: {self._scope_name(scope)} invalidated")

    # -- firmware / timing ------------------------------------------------------------------

    def boot(self, m, *, handler: bool) -> None:
        """What the boot firmware does: every entry invalid — with EHINV
        when the core has it, else unique VPN2s in kseg0, which never match
        a mapped address (Vol. III 4.9.3 InitTLB) —, Wired = 0, PageMask =
        0, ASID 0; with ``handler`` its refill handler
        (:data:`FIRMWARE_REFILL`) is the page table."""
        kseg0 = 0xFFFFFFFF80000000 if self.wide else 0x80000000   # (MIPS64: ckseg0)
        for i in range(self.total):
            if self.invalidate:
                self.entries[i] = TLBEntry(written=True, ehinv=True, vmask=self._vmask)
            else:
                self.entries[i] = TLBEntry(hi=(kseg0 + (i << 13)) & self._hi_vpn,
                                           written=True, vmask=self._vmask)
        self._changed()
        for key in (PAGEMASK, ENTRYHI, ENTRYLO0, ENTRYLO1, INDEX, CONTEXT) + (
                (XCONTEXT,) if self.wide else ()):
            m.cp0[key]["value"] = 0
        self.cp0_write(m, WIRED, 0)
        if self.ftlb:                        # the OS turns the FTLB on (Linux:
            self.cp0_write(m, CONFIG6, CFG6_FTLBEN)   # Config6.FTLBEn on a P5600)
        cca = boot_cca(m)
        window = user_window(m)
        if self.wide:                        # the XTLB handler's mapping too
            self.fallback = (lambda va: firmware_map64(va, cca, window)) if handler else None
        else:
            self.fallback = (lambda va: firmware_map(va, cca, window)) if handler else None

    def take(self) -> tuple[int, list[str]]:
        """The micro-TLB miss cycles since the last call (the pipeline)."""
        out = (self.wait, self.notes)
        self.wait, self.notes = 0, []
        return out
