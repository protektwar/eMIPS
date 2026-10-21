"""Views of the caches (PLAN M9 phase 5), shared by the CLI debugger
(``cache``, ``run --cache-stats``) and the GUI cache pane.

Qt-free and side-effect free: everything shown is read from the cache
arrays (:class:`~emips.cache.Line`, the way-select RAM), the counters
(:class:`~emips.cache.CacheStats`) and the recorded accesses
(:class:`~emips.cache.CacheAccess`); nothing is looked up through the
caches (a lookup would update the LRU state and the counters).

How an address meets a cache (MD00343 8.5.1, MD00355 3.4.1)::

    L1 (VIPT)     VA  | ...        | index  | offset |
                  PA  | tag (PA[31:12])            |
    L2 / L3       PA  | tag        | index  | offset |
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import MISS_CLASSES, Cache, Caches, CacheAccess, WriteBuffer

M32 = 0xFFFFFFFF

#: Names the user may give a cache (CLI, GUI) -> the Caches attribute.
CACHE_KEYS = {"i": "i", "i$": "i", "icache": "i", "d": "d", "d$": "d", "dcache": "d",
              "l2": "l2", "s": "l2", "l3": "l3", "t": "l3"}
#: Short labels, in hierarchy order.
SHORT = {"I": "I$", "D": "D$", "S": "L2", "T": "L3"}


def cache_named(caches: Caches | None, name: str) -> Cache | None:
    """``I`` / ``D`` / ``L2`` / ``L3`` (also ``I$``, ``icache``, ...)."""
    if caches is None:
        return None
    key = CACHE_KEYS.get(name.strip().lower())
    return getattr(caches, key) if key else None


def short_name(c: Cache) -> str:
    return SHORT[c.kind]


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f} %" if whole else "-"


# ---------------------------------------------------------------- addresses

@dataclass
class Split:
    """An address cut into the fields a cache uses.  ``*_bits`` are
    ``(msb, lsb)``; ``index_from`` is "VA" for the L1s (virtually
    indexed), "PA" for the L2 / L3."""

    va: int
    pa: int
    tag: int
    index: int
    offset: int
    tag_bits: tuple[int, int]
    index_bits: tuple[int, int]
    offset_bits: tuple[int, int]
    index_from: str

    def fields(self) -> list[tuple[str, int, int, int, str]]:
        """``(name, msb, lsb, value, from)``, most significant first."""
        return [("tag", *self.tag_bits, self.tag, "PA"),
                ("index", *self.index_bits, self.index, self.index_from),
                ("offset", *self.offset_bits, self.offset, self.index_from)]


def split(c: Cache, va: int, pa: int) -> Split:
    ia = va if c.level == 1 else pa
    set_bits = c.sets.bit_length() - 1
    return Split(va & M32, pa & M32, c.tag_of(pa), c.set_of(ia), ia & c.line_mask,
                 (31, c.tag_shift), (c.line_shift + set_bits - 1, c.line_shift),
                 (c.line_shift - 1, 0), "VA" if c.level == 1 else "PA")


def _bits(value: int, msb: int, lsb: int) -> str:
    if msb < lsb:
        return ""
    return format((value >> lsb) & ((1 << (msb - lsb + 1)) - 1), f"0{msb - lsb + 1}b")


def format_split(c: Cache, va: int, pa: int) -> str:
    """How ``va`` / ``pa`` are cut for cache ``c``, with the bits."""
    sp = split(c, va, pa)
    kind = ("virtually indexed, physically tagged" if c.level == 1
            else "physically indexed and tagged")
    out = [f"{c.name} ({c.geometry.describe()}): {kind}"]
    for name, msb, lsb, value, src in sp.fields():
        if msb < lsb:
            out.append(f"  {name:<6} (none)")
            continue
        addr = sp.pa if src == "PA" else sp.va
        out.append(f"  {name:<6} {src}[{msb}:{lsb}] = {_bits(addr, msb, lsb)}"
                   f" = 0x{value:x} ({value})")
    if c.level == 1 and c.tag_shift < c.line_shift + c.sets.bit_length() - 1:
        out.append("  (index and tag overlap: the index bits above bit 11 come "
                   "from the virtual address, the tag holds them from the physical one)")
    return "\n".join(out)


# ---------------------------------------------------------------- contents

@dataclass
class WayView:
    way: int
    valid: bool
    dirty: bool
    lock: bool
    tag: int
    pa: int                 # the line's physical address (from tag and set)
    age: int                # 0 = most recently used ... ways - 1 = LRU
    data: bytes
    known: int              # bit i: byte i holds data written by someone
    shared: bool = False    # MESI S (a data cache of a CPS core)

    def mesi(self) -> str:
        """M / E / S / I (a data cache of a CPS core)."""
        return "I" if not self.valid else "M" if self.dirty else "S" if self.shared else "E"


def set_view(c: Cache, s: int) -> list[WayView]:
    order = c.order(s)                       # least -> most recently used
    out = []
    for w, ln in enumerate(c.lines[s]):
        out.append(WayView(w, ln.valid, bool(c.writable and ln.dirty), ln.lock, ln.tag,
                           c.line_pa(ln.tag, s), c.ways - 1 - order.index(w),
                           bytes(ln.data), ln.known, ln.shared))
    return out


def used_sets(c: Cache) -> list[int]:
    """Sets holding at least one valid line."""
    return [s for s in range(c.sets) if any(ln.valid for ln in c.lines[s])]


def state_letter(ln) -> str:
    return "I" if not ln.valid else "M" if ln.dirty else "S" if ln.shared else "E"


#: the MESI states (emips.coherence)
MESI_NAMES = {"M": "Modified", "E": "Exclusive", "S": "Shared", "I": "Invalid"}


def way_text(c: Cache, wv: WayView) -> str:
    """``V D L tag 0x50010 (line 0x50010000)``, or ``-`` when invalid."""
    if not wv.valid:
        return "-"
    if c.mesi:
        flags = wv.mesi() + ("L" if wv.lock else "-")
    else:
        flags = "V" + ("D" if wv.dirty else "-") + ("L" if wv.lock else "-")
    return f"{flags} tag 0x{wv.tag:x} (line 0x{wv.pa:08x})"


def format_set(c: Cache, s: int, *, data: bool = True) -> str:
    out = [f"{c.name} set {s}" + ("  (direct mapped)" if c.ways == 1 else
                                    f"  (age 0 = most recently used, {c.ways - 1} = "
                                    "least recently used: replaced next)")]
    for wv in set_view(c, s):
        out.append(f"  way {wv.way}  age {wv.age}  {way_text(c, wv)}")
        if data and wv.valid:
            for a in range(0, c.line, 16):
                cells = " ".join(f"{b:02x}" if wv.known >> (a + i) & 1 else ".."
                                 for i, b in enumerate(wv.data[a:a + 16]))
                out.append(f"      +{a:<3d} {cells}")
    return "\n".join(out)


def format_cache(c: Cache, *, limit: int = 16) -> str:
    """The sets holding valid lines (the first ``limit``)."""
    used = used_sets(c)
    valid = sum(ln.valid for row in c.lines for ln in row)
    out = [f"{c.name}: {c.geometry.describe()}; {valid} of {c.sets * c.ways} "
           f"lines valid, in {len(used)} set(s)"]
    for s in used[:limit]:
        ways = set_view(c, s)
        cells = "  ".join(f"w{wv.way}:{way_text(c, wv)}" for wv in ways if wv.valid)
        out.append(f"  set {s:4d}: {cells}")
    if len(used) > limit:
        out.append(f"  ... {len(used) - limit} more set(s) (cache {short_name(c)} SET)")
    return "\n".join(out)


def locate(m, caches: Caches, va: int) -> str:
    """Where the line of ``va`` is right now (no side effects)."""
    xl = m.translate(va)
    va &= m.vamask                              # (MIPS64 with 64-bit addresses: whole)
    if xl.pa is None:
        return f"0x{va:08x}: no physical address ({xl.how})"
    out = [f"0x{va:08x}: PA 0x{xl.pa:08x}, attribute {xl.cca}"
           + ("" if xl.cacheable else " (uncached: the caches are bypassed)")]
    for c in caches.all:
        sp = split(c, va, xl.pa)
        w = c.find(sp.index, sp.tag)
        state = "not cached" if w is None else f"in way {w}"
        if w is not None:
            ln = c.lines[sp.index][w]
            if c.mesi:
                state += f" ({MESI_NAMES[state_letter(ln)]})"
            else:
                state += " (dirty)" if c.writable and ln.dirty else ""
            state += " (locked)" if ln.lock else ""
        out.append(f"  {short_name(c)}: tag 0x{sp.tag:x}, set {sp.index}, "
                   f"offset {sp.offset}: {state}")
    if caches.cm is not None:
        copies = caches.cm.copies(xl.pa)
        out.append("  data caches of the cores: " + (", ".join(
            f"core {j} {MESI_NAMES[st]}" for j, st in copies) if copies else "no copy"))
    return "\n".join(out)


# ---------------------------------------------------------------- statistics

def stats_lines(c: Cache) -> list[str]:
    st = c.stats
    acc = st.accesses
    out = [f"{c.name} ({c.geometry.describe()})",
           f"  accesses {acc}   hits {st.hits} ({pct(st.hits, acc)})"
           f"   misses {st.misses} ({pct(st.misses, acc)})"]
    for kind, (h, m) in sorted(st.kinds.items()):
        out.append(f"    {kind:<6} hits {h:<8d} misses {m}")
    filled = sum(st.classes.values())
    if filled:
        out.append(f"  misses that filled a line: {filled} -- "
                   + ", ".join(f"{k} {st.classes[k]}" for k in MISS_CLASSES))
    if st.no_allocate:
        out.append(f"  misses that filled nothing: {st.no_allocate} "
                   "(no write allocate, SYNCI)")
    out.append(f"  evictions {c.evictions}, written back {c.writebacks}")
    if c.level > 1 and c.parity:
        out.append(f"  ECC: {c.corrected} single-bit error(s) corrected")
    return out


def wbb_lines(wbb: WriteBuffer) -> list[str]:
    out = [f"write buffer: {wbb.merges} merged, {wbb.gathers} gathered, "
           f"{wbb.flushes} flushed, {wbb.immediate} sent at once"]
    for e in wbb.entries:
        what = "write-through merge" if e.kind == "WT" else "uncached accelerated gather"
        out.append(f"  open {what} entry 0x{e.base:08x}: "
                   f"{bin(e.mask).count('1')} of 32 bytes")
    return out


def format_stats(caches: Caches | None) -> str:
    if caches is None or not caches.all:
        return "no caches (Settings > Caches, or --caches)"
    out = []
    for c in caches.all:
        out += stats_lines(c)
    if caches.wbb is not None:
        out += wbb_lines(caches.wbb)
    return "\n".join(out)


def access_text(acc: CacheAccess) -> str:
    return acc.describe()


def summary(accesses: list[CacheAccess]) -> str:
    """Compact: ``D$ miss (conflict) -> L2 hit``."""
    parts = []
    for a in accesses:
        if a.level == "U":
            parts.append("memory" if a.miss == "no cache" else "uncached")
            continue
        what = "hit" if a.hit else (f"miss ({a.miss})" if a.miss else "miss")
        parts.append(f"{SHORT[a.level]} {what}")
    return " -> ".join(parts)


# ---------------------------------------------------------------- the GUI pane

#: The pane shows every set of a cache up to this many (else only the
#: sets holding valid lines and the one last accessed).
ALL_SETS_LIMIT = 1024


def last_access(accesses, c: Cache) -> CacheAccess | None:
    """The last access to cache ``c`` among ``Session.cache_accesses()``."""
    found = None
    for _where, a in accesses:
        if a.level == c.kind:
            found = a
    return found


def pane_sets(c: Cache, *, all_sets: bool, focus: int | None) -> list[int]:
    """The sets the pane lists: all of them (up to :data:`ALL_SETS_LIMIT`),
    or those holding valid lines; the set ``focus`` (last accessed) always."""
    if all_sets and c.sets <= ALL_SETS_LIMIT:
        return list(range(c.sets))
    rows = used_sets(c)
    if focus is not None and focus not in rows:
        rows = sorted(rows + [focus])
    return rows


def cell_text(c: Cache, wv: WayView) -> str:
    """A way of a set in the pane's table."""
    if not wv.valid:
        return "-"
    flags = (wv.mesi() if c.mesi else "D" if wv.dirty else "") + ("L" if wv.lock else "")
    age = "" if c.ways == 1 else f" a{wv.age}"
    return f"{wv.tag:x}{age}" + (f" {flags}" if flags else "")


def cell_tip(c: Cache, s: int, wv: WayView, big_endian: bool = True) -> str:
    """Tooltip of a way: state, line address, data words."""
    if not wv.valid:
        return f"set {s} way {wv.way}: invalid"
    state = ["valid"] + (["dirty"] if wv.dirty else []) + (["locked"] if wv.lock else [])
    if c.mesi:
        state = [MESI_NAMES[wv.mesi()]] + (["locked"] if wv.lock else [])
    order = "big" if big_endian else "little"
    words = []
    for a in range(0, c.line, 4):
        if (wv.known >> a) & 0xF == 0xF:
            words.append(f"{int.from_bytes(wv.data[a:a + 4], order):08x}")
        else:
            words.append("........")
    rows = [" ".join(words[i:i + 4]) for i in range(0, len(words), 4)]
    age = "" if c.ways == 1 else (
        f"; age {wv.age} (0 = most recent"
        + (", replaced next)" if wv.age == c.ways - 1 else ")"))
    return (f"set {s} way {wv.way}: {', '.join(state)}; tag 0x{wv.tag:x} = line "
            f"0x{wv.pa:08x}{age}\n" + "\n".join(rows))
