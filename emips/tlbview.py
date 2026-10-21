"""Views of the MMU (PLAN M9 phase 8), shared by the CLI debugger
(``tlb``, ``run --tlb-stats``) and the GUI TLB pane.

Qt-free and side-effect free: everything shown is read from the TLB
entries (:class:`~emips.tlb.TLBEntry`), the micro-TLB slots, the BAT
entries, the counters (:class:`~emips.tlb.TLBStats`) and the recorded
translations (:class:`~emips.tlb.TLBAccess`); nothing is translated
through the micro-TLBs (that would refill them and count).

How a TLB translates a virtual address (Vol. III 4.9; a dual entry maps
an even / odd pair of pages)::

    VA   | VPN2 (matched with EntryHi) | even/odd | offset in the page |
              -> TLB entry -> EntryLo0 / EntryLo1: PFN, C, D, V
    PA   | PFN (from the EntryLo)                 | offset             |
"""

from __future__ import annotations

from dataclasses import dataclass

from .bat import BATMMU
from .mmu import M32, cca_name, segment
from .tlb import (ENTRYHI, HI_ASID, INDEX, LO_D, LO_RI, LO_V, LO_XI, M64, PAGEMASK, RANDOM,
                  SEGBITS, TLBMMU, WIRED, TLBAccess, TLBEntry, size_name)

HALF = ("even", "odd")


def kind(m) -> str:
    """``tlb``, ``bat`` or ``fixed`` (no entries to show)."""
    if isinstance(m.mmu, TLBMMU):
        return "tlb"
    if isinstance(m.mmu, BATMMU):
        return "bat"
    return "fixed"


def lo_flags(lo: int) -> str:
    return "".join(f for f, b in (("V", LO_V), ("D", LO_D), ("RI", LO_RI), ("XI", LO_XI))
                   if lo & b) or "-"


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f} %" if whole else "-"


# ------------------------------------------------------------------ entries

@dataclass
class EntryRow:
    """One TLB entry as the views list it."""

    index: int
    name: str
    used: bool                  # written and not invalidated
    state: str                  # "", "never written", "invalidated (EHINV)"
    wired: bool
    random: bool                # the entry TLBWR writes next
    vpn2: int
    va: int                     # first VA of the pair
    span: int                   # bytes the pair maps
    page: int
    g: bool
    asid: int
    lo: tuple[int, int]
    pfn: tuple[int, int]
    frame: tuple[int, int]


def _pfn(e: TLBEntry, half: int) -> int:
    return (e.lo[half] >> 6) & (0x3FFFFF if e.small else 0xFFFFF)


def entry_row(mmu: TLBMMU, m, i: int) -> EntryRow:
    e = mmu.entries[i]
    wired = m.cp0[WIRED]["value"] & mmu.index_mask
    state = "" if e.live else ("invalidated (EHINV)" if e.written else "never written")
    return EntryRow(i, mmu.entry_name(i), e.live, state, i < wired and i < mmu.n,
                    i == mmu.random and i < mmu.n, e.vpn2 >> 13, e.vpn2, 2 * e.page_size,
                    e.page_size, e.g, e.asid, (e.lo[0], e.lo[1]), (_pfn(e, 0), _pfn(e, 1)),
                    (e.frame(0), e.frame(1)))


def rows(mmu: TLBMMU, m, *, everything: bool = False) -> list[EntryRow]:
    """Every entry (``everything``), else those in use, Wired and Random."""
    out = []
    for i in range(mmu.total):
        r = entry_row(mmu, m, i)
        if everything or r.used or r.wired or r.random:
            out.append(r)
    return out


def half_text(r: EntryRow, h: int) -> str:
    lo = r.lo[h]
    return f"PFN 0x{r.pfn[h]:05x} C{(lo >> 3) & 7} {lo_flags(lo)}"


def vpn_text(va: int) -> str:
    """``VPN2 0x00400`` (VA 31:13); MIPS64: ``R 3, VPN2 0x00001`` (VA 63:62,
    VA SEGBITS-1:13)."""
    if va > M32:
        return f"R {va >> 62}, VPN2 0x{(va >> 13) & ((1 << (SEGBITS - 13)) - 1):05x}"
    return f"VPN2 0x{va >> 13:05x}"


def _top(va: int, span: int) -> int:
    """The last address of a page pair (MIPS64: 64-bit VAs)."""
    return (va + span - 1) & (M64 if va > M32 else M32)


def _lo(mmu: TLBMMU, lo: int) -> int:
    """EntryLo as the core's software reads it (MIPS64: RI / XI at 63 / 62)."""
    if getattr(mmu, "wide", False):
        from .sim64 import lo64
        return lo64(lo)
    return lo


def row_text(r: EntryRow) -> str:
    marks = ("W" if r.wired else " ") + ("R" if r.random else " ")
    if not r.used:
        return f"{r.index:4d} {marks} {r.state}"
    owner = "G" if r.g else f"ASID {r.asid:3d}"
    return (f"{r.index:4d} {marks} VA 0x{r.va:08x}-0x{_top(r.va, r.span):08x} "
            f"{owner:<8} {size_name(r.page):>6}  "
            f"{half_text(r, 0):<24} | {half_text(r, 1)}")


def header(mmu: TLBMMU, m) -> str:
    cp = m.cp0
    wired = cp[WIRED]["value"] & mmu.index_mask
    used = sum(e.live for e in mmu.entries)
    shape = f"{mmu.n} dual entries" if not mmu.ftlb else (
        f"VTLB {mmu.n} entries + FTLB {mmu.fsets} sets x {mmu.fways} ways "
        f"({size_name(mmu.ftlb_page)} pages)")
    return (f"{mmu.name}: {shape}; {used} in use; Wired {wired}, Random {mmu.random}"
            f" (next TLBWR), ASID {cp[ENTRYHI]['value'] & HI_ASID}")


def format_tlb(m, *, everything: bool = False, limit: int = 80) -> str:
    mmu = m.mmu
    if kind(m) == "bat":
        return format_bat(m)
    if kind(m) == "fixed":
        return no_tlb(m)
    out = [header(mmu, m),
           f"{'#':>4}    {'VA range (page pair)':<24} {'owner':<8} {'page':>6}  "
           f"{'even page (EntryLo0)':<24} | odd page (EntryLo1)"]
    rs = rows(mmu, m, everything=everything)
    for r in rs[:limit]:
        out.append(row_text(r))
    if len(rs) > limit:
        out.append(f"  ... {len(rs) - limit} more")
    out.append("  W = wired (TLBWR never replaces it), R = Random (the entry TLBWR writes next)")
    return "\n".join(out)


def no_tlb(m) -> str:
    return (f"{getattr(m.mmu, 'name', 'fixed mapping')} MMU: no TLB (Config.MT = "
            f"{getattr(m.mmu, 'mt', 3)}); start with --mmu tlb, or set mmu tlb")


def format_entry(m, i: int) -> str:
    """One entry in detail: the tag, both halves, what TLBR would read."""
    mmu: TLBMMU = m.mmu
    e = mmu.entries[i]
    r = entry_row(mmu, m, i)
    out = [f"{mmu.entry_name(i)}" + (" (wired)" if r.wired else "")
           + (" (Random: TLBWR writes it next)" if r.random else "")]
    if not e.live:
        out.append(f"  {r.state}: matches nothing")
        return "\n".join(out)
    out.append(f"  tag: {vpn_text(r.va)} (VA 0x{r.va:08x}-0x{_top(r.va, r.span):08x}), "
               + ("global (G = 1: every ASID)" if e.g else f"ASID {e.asid}"))
    out.append(f"  page size {size_name(e.page_size)} (PageMask 0x{e.pagemask:08x}): "
               f"VA bit {e.evenodd} picks the even / odd page")
    for h in (0, 1):
        lo = e.lo[h] | e.g
        va0 = r.va + h * e.page_size
        cca = (lo >> 3) & 7
        out.append(f"  {HALF[h]:<4} EntryLo{h} 0x{_lo(mmu, lo):08x}: VA 0x{va0:08x} -> "
                   f"PFN 0x{r.pfn[h]:05x}"
                   f" = PA 0x{r.frame[h]:08x}; C{cca} ({cca_name(cca)}), "
                   f"{lo_flags(lo)}")
    out.append(f"  TLBR would read: PageMask 0x{e.pagemask:08x}, EntryHi "
               f"0x{e.hi & mmu._vamask:08x}, EntryLo0 0x{_lo(mmu, e.lo[0] | e.g):08x}, "
               f"EntryLo1 0x{_lo(mmu, e.lo[1] | e.g):08x}")
    return "\n".join(out)


def format_ftlb_set(m, s: int) -> str:
    mmu: TLBMMU = m.mmu
    lo, bits = mmu.ftlb_page.bit_length(), mmu.fsets.bit_length() - 1
    out = [f"FTLB set {s} ({mmu.fways} ways, {size_name(mmu.ftlb_page)} pages; the set "
           f"of a VA is VA[{lo + bits - 1}:{lo}] XOR VA[{lo + 2 * bits - 1}:{lo + bits}])"]
    for w in range(mmu.fways):
        out.append(row_text(entry_row(mmu, m, mmu.ftlb_index(w, s))))
    return "\n".join(out)


# ------------------------------------------------------------------ micro-TLBs

def micro_lines(utlb) -> list[str]:
    """The slots of a micro-TLB (age 0 = most recently used)."""
    out = [f"{utlb.name} ({utlb.size} entries, {'LRU' if utlb.lru else 'pseudo-LRU'}): "
           f"{utlb.hits} hits, {utlb.misses} misses ({pct(utlb.misses, utlb.hits + utlb.misses)})"]
    ages = micro_ages(utlb)
    for i, e in enumerate(utlb.slots):
        if e is None:
            out.append(f"  {i}: -")
            continue
        own = "G" if e.g else f"ASID {e.asid}"
        age = f" age {ages[i]}" if ages[i] is not None else ""
        out.append(f"  {i}: VA 0x{e.vbase:08x} ({size_name(e.size)}, {own}) -> PA "
                   f"0x{e.pbase:08x} C{e.cca} {lo_flags(e.lo)}  (from TLB entry {e.index}){age}")
    return out


def micro_ages(utlb) -> list[int | None]:
    """True LRU (the ITLB): the age of every slot; pseudo-LRU: None, the
    slot replaced next (no ordering beyond the tree) is marked by
    :func:`micro_victim`."""
    if not utlb.lru:
        return [None] * utlb.size
    order = list(reversed(utlb.order))         # most recent first
    return [order.index(i) if utlb.slots[i] is not None else None for i in range(utlb.size)]


def micro_victim(utlb) -> int:
    """The slot the next micro-TLB miss replaces (no side effects)."""
    return utlb._victim()


def format_micro(m) -> str:
    mmu = m.mmu
    if kind(m) != "tlb":
        return "no micro-TLBs: they come with a TLB (--mmu tlb)"
    out = []
    for utlb in (mmu.itlb, mmu.dtlb):
        out += micro_lines(utlb)
        out.append(f"  next miss replaces slot {micro_victim(utlb)}")
    out.append("They hold copies of TLB translations (4 KB or 1 MB pieces); a miss costs "
               f"{mmu.utlb_cost} cycles in the pipeline, a TLB write removes the copies.")
    return "\n".join(out)


# ------------------------------------------------------------------ BAT

def format_bat(m) -> str:
    mmu: BATMMU = m.mmu
    out = ["BAT MMU (Config.MT = 2): six base-and-bounds entries; PA = VA + base"]
    for i, e in enumerate(mmu.entries):
        out.append(f"  {i} {mmu.entry_how(i):<9} {e.describe()}")
    return "\n".join(out)


# ------------------------------------------------------------------ one address

@dataclass
class VASplit:
    """A virtual address cut as a TLB entry with pages of ``page`` bytes
    cuts it: ``(name, msb, lsb, value)`` fields."""

    va: int
    page: int
    #: MIPS64 (64-bit addresses): R VA[63:62], VPN2 up to VA[SEGBITS-1]
    wide: bool = False

    def fields(self) -> list[tuple[str, int, int, int]]:
        b = self.page.bit_length() - 1             # offset bits
        top = SEGBITS - 1 if self.wide else 31
        vpn = [("VPN2", top, b + 1, (self.va >> (b + 1)) & ((1 << (top - b)) - 1))]
        if self.wide:
            vpn.insert(0, ("R", 63, 62, self.va >> 62))
        return vpn + [("even/odd", b, b, (self.va >> b) & 1),
                      ("offset", b - 1, 0, self.va & (self.page - 1))]


def bits(value: int, msb: int, lsb: int) -> str:
    return format((value >> lsb) & ((1 << (msb - lsb + 1)) - 1), f"0{msb - lsb + 1}b")


def explain(m, va: int, access: str = "load") -> str:
    """How ``va`` translates right now, step by step: segment, the entry
    that matches, the half, PFN and PA — or the exception an access would
    take.  No side effects."""
    t = m.translate(va & m.amask)
    va = t.va                                   # (MIPS64: sign-extended)
    mmu = m.mmu
    wide = getattr(mmu, "wide", False)
    asid = m.cp0[ENTRYHI]["value"] & HI_ASID
    out = [f"0x{va:08x} ({t.segment if wide else segment(va)}): "
           + ("mapped" if t.mapped else "unmapped")]
    if not t.mapped:
        out.append(f"  {t.how}" + (": an access takes an Address Error exception"
                                   if t.fault == "address error" else ""))
        if t.pa is not None:
            out.append(f"  PA 0x{t.pa:08x}, C{t.cca} ({cca_name(t.cca)})")
        return "\n".join(out)
    if kind(m) == "bat":
        i = mmu.entry_of(va, access)
        e = mmu.entries[i]
        out.append(f"  BAT entry {i} ({mmu.entry_how(i)}): {e.describe()}")
        if t.pa is None:
            out.append(f"  {t.how}: an access takes a TLB Invalid exception")
        else:
            out.append(f"  PA = VA + base = 0x{va:08x} + 0x{e.pfn << 12:08x} = 0x{t.pa:08x}")
        return "\n".join(out)
    if kind(m) != "tlb":
        out.append(f"  {t.how}" + (f" -> PA 0x{t.pa:08x}" if t.pa is not None else ""))
        return "\n".join(out)
    ix = mmu.match(va, asid)
    if mmu.ftlb:
        out.append(f"  looked up: the {mmu.n} VTLB entries, and FTLB set {mmu.ftlb_set(va)} "
                   f"(VA[{mmu.ftlb_page.bit_length() + mmu.fsets.bit_length() - 2}:"
                   f"{mmu.ftlb_page.bit_length()}])")
    if ix is None:
        out.append(f"  no entry matches {vpn_text(va)} (4 KB pages) with ASID {asid}"
                   " or G = 1")
        if mmu._xtlb(m, va):
            out.append("  an access takes a TLB Refill exception at the XTLB Refill vector "
                       "(64-bit addresses: offset 0x080 with Status.EXL = 0, else 0x180)")
        else:
            out.append("  an access takes a TLB Refill exception (vector offset 0x000 with "
                       "Status.EXL = 0, else 0x180)")
        if getattr(mmu, "pw", False):
            out.append("  (the hardware page table walker tries the page table first, "
                       "when PWCtl.PWEn = 1)")
        if t.pa is not None:                 # the boot firmware's handler
            out.append(f"  (the boot firmware's refill handler then maps it: PA 0x{t.pa:08x}, "
                       f"C{t.cca} = {cca_name(t.cca)})")
        return "\n".join(out)
    e = mmu.entries[ix]
    sp = VASplit(va, e.page_size, wide)
    cells = []
    for name, msb, lsb, value in sp.fields():
        cells.append(f"{name} VA[{msb}:{lsb}] = 0x{value:x}" if msb != lsb
                     else f"{name} VA[{msb}] = {value}")
    out.append("  " + " | ".join(cells))
    out.append(f"  {mmu.entry_name(ix)} matches: {vpn_text(e.vpn2)}, "
               + ("global" if e.g else f"ASID {e.asid}") + f", {size_name(e.page_size)} pages")
    half = (va >> e.evenodd) & 1
    lo = e.lo[half] | e.g                     # as TLBR reads it (G in both)
    cca = (lo >> 3) & 7
    out.append(f"  {HALF[half]} page: EntryLo{half} = 0x{_lo(mmu, lo):08x}: "
               f"PFN 0x{_pfn(e, half):05x}, "
               f"C{cca} ({cca_name(cca)}), {lo_flags(lo)}")
    if t.pa is None:
        out.append("  V = 0: an access takes a TLB Invalid exception")
        return "\n".join(out)
    off = va & (e.page_size - 1)
    out.append(f"  PA = page 0x{e.frame(half):08x} + offset 0x{off:x} = 0x{t.pa:08x}")
    if not lo & LO_D:
        out.append("  D = 0: a store takes a TLB Modified exception")
    for utlb in (mmu.itlb, mmu.dtlb):
        held = [i for i, u in enumerate(utlb.slots) if u is not None and u.matches(va, asid)]
        if held:
            out.append(f"  {utlb.name} holds a copy (slot {held[0]})")
    return "\n".join(out)


# ------------------------------------------------------------------ statistics

def stats_lines(m) -> list[str]:
    mmu = m.mmu
    if kind(m) == "fixed":
        return [no_tlb(m)]
    st = mmu.stats
    out = [f"{mmu.name}"]
    if kind(m) == "tlb":
        look = st.hits + st.misses
        out.append(f"  TLB lookups {look} (micro-TLB misses): hits {st.hits} "
                   f"({pct(st.hits, look)}), misses {st.misses} ({pct(st.misses, look)})")
        for u in (mmu.itlb, mmu.dtlb):
            out.append(f"  {u.name}: {u.hits + u.misses} translations, hits {u.hits} "
                       f"({pct(u.hits, u.hits + u.misses)}), misses {u.misses}")
        if mmu.pw:
            out.append(f"  page table walks {st.walks}, refilled the TLB {st.walked}")
    else:
        out.append(f"  translations {st.hits + sum(st.faults.values())}")
    out.append(f"  entries written {st.writes}")
    if st.faults:
        out.append("  exceptions: " + ", ".join(f"{k} {v}" for k, v in sorted(st.faults.items())))
    else:
        out.append("  exceptions: none")
    return out


def format_stats(m) -> str:
    return "\n".join(stats_lines(m))


def reset_stats(m) -> None:
    if hasattr(m.mmu, "reset_stats"):
        m.mmu.reset_stats()


def last_access(accesses) -> TLBAccess | None:
    """The last translation (not a TLB instruction) of ``Session.tlb_accesses()``."""
    found = None
    for _where, a in accesses:
        if not a.note:
            found = a
    return found


def registers(m) -> list[tuple[str, int]]:
    """The CP0 registers a refill handler uses, for the pane."""
    from .tlb import CONTEXT, ENTRYLO0, ENTRYLO1
    cp = m.cp0
    names = (("Index", INDEX), ("Random", RANDOM), ("EntryHi", ENTRYHI), ("EntryLo0", ENTRYLO0),
             ("EntryLo1", ENTRYLO1), ("PageMask", PAGEMASK), ("Wired", WIRED),
             ("Context", CONTEXT), ("BadVAddr", (8, 0)))
    return [(n, cp[k]["value"]) for n, k in names]
