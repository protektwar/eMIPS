"""The cache hierarchy holding real data (PLAN M9 phases 2-3): L1
instruction / data caches, L2 / L3 caches and the write buffer.

References: *MIPS32 Architecture For Programmers* Vol. II-A (CACHE,
PREF, SYNC, SYNCI), Vol. III (Config1 / Config2, the cacheability
attributes, the Cache Error exception) and, for what the architecture
leaves to the implementation, the *MIPS32 24K Processor Core Family
Software User's Manual* (MD00343, Rev. 03.11): chapter 8 "Caches of the
24K Core" (8.4 the write back buffer), the CACHE / PREF / SYNC pages of
chapter 12 and the Config / Config2 / ErrCtl / CacheErr / TagLo / DataLo
/ DataHi / Config7 registers of chapter 6; for the L2, *Programming the
MIPS32 24K Core Family* (MD00355, Rev. 04.63) 3.3.3 and 3.4.1-3.4.3 (the
SOC-it L2 controller's own manual, MD00525, is not public).

A cache is ``sets`` × ``ways`` **lines**; every line has a tag (bits of
its physical address), a valid, a dirty (data caches) and a lock bit and
holds ``line`` bytes of real data.  The L1 caches are **virtually indexed
and physically tagged** (24K 8.5.1): the set comes from the virtual
address, the tag from the physical one::

    virtual address    | ... | set index | byte in line |
    physical address   |   tag (PA[31:12], PA[31:11] for 2 KB ways)  |

The L2 / L3 caches are **physically indexed and tagged** (MD00355 3.4.1:
"unaffected by cache aliases")::

    physical address   |   tag   | set index | byte in line |

Replacement is least recently used, excluding valid locked lines
(24K 8.5.3); the LRU state of a set is the 24K way-select (WS) RAM: one
bit per pair of ways (Table 8.4).  An L2 / L3 with more than 4 ways uses
a pseudo-LRU (MD00355: "true LRU is impractical for an 8-way set
associative cache"): a binary tree of 7 bits.  Optional L1 parity
(24K 8.2.3, 8.3.2): one bit per tag, per data byte and per dirty bit;
with ``ErrCtl.PE = 1`` a mismatch found when the array is read is a
Cache Error exception.  Optional L2 / L3 ECC (MD00355 3.4.1): 8 check
bits per doubleword correct any 1-bit and detect any 2-bit error, tags
and dirty bits have a check bit; checked when ``ErrCtl.PE`` differs from
``ErrCtl.L2P`` (24K Table 6.49).  An uncorrectable L2 error is
**imprecise** (24K 5.8.12): the access completes with the bad data and
the exception is taken before the next instruction.

The way below the L1 caches is the 24K's bus interface unit::

    L1 D-cache --evictions, write-through, uncached, UCA--> write buffer
    L1 I/D refills, uncached reads ----------------------->     |
                                                                v
                                        L2 --> L3 --> memory and devices
                                  (uncached and UCA traffic bypasses them)

The **write buffer** (24K 8.4) has 4 entries of 32 bytes.  Evictions and
uncached stores leave it at once (flushed immediately); write-through
stores to one 32-byte block are merged into one entry while
``Config.MM = 1``, and uncached accelerated stores are gathered, until
a flush condition sends them on (see :class:`WriteBuffer`).  The L2 is
write-back without write allocate (MD00355 3.4.1: "write miss data is
just sent directly to the system memory") and read-allocates every
cacheable refill; it does not hold back-pointers to the L1s, so it is
neither inclusive nor exclusive (the 24K has no snoop port an L2 could
invalidate L1 lines through).

The caches are undefined after reset (random tags, data and LRU, from a
fixed seed so runs repeat), unless built with ``hardware_init`` (the
24K's ``Config7.HCI``: "a cache does not require initialization by
software", found on simulation models).  :meth:`Caches.initialize` is
what boot firmware does: Index Store Tag with ``TagLo = 0`` on every
index of every way of every cache.

What the caches did is observable (PLAN M9 phase 5) without changing
it: every cache keeps :class:`CacheStats` (hits and misses per kind of
access, misses classified by the "3 C's" of Hill and Smith,
evictions), and while :meth:`Caches.watch` is on every lookup is
recorded as a :class:`CacheAccess` (the trace and the views read them).
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from .timing import CACHEOP_COST, Timing

M32 = 0xFFFFFFFF

#: Cacheability attributes that use the caches (24K Table 4.11):
#: 0 = cacheable write-through, no write allocate; 3 = cacheable
#: write-back, write allocate.  2 (uncached), 7 (uncached accelerated)
#: and the reserved 1, 4, 5, 6 bypass them.
CCA_WT, CCA_UC, CCA_WB, CCA_UCA = 0, 2, 3, 7
CACHED_CCA = frozenset({CCA_WT, CCA_WB})
#: The attributes a 24K implements (Config.K0 / KU / K23 encodings).
CCA_24K = frozenset({0, 2, 3, 7})
#: The coherent attributes of a core in a Coherent Processing System
#: (1004K / interAptiv, emips.coherence): 4 = cacheable coherent
#: write-back, a load miss takes the line Exclusive; 5 = the same, a load
#: miss takes it Shared (Exclusive when no other cache has it).
CCA_CWBE, CCA_CWB = 4, 5
COHERENT_CCA = frozenset({CCA_CWBE, CCA_CWB})
#: Write-back attributes (write allocate, dirty lines).
WRITE_BACK_CCA = frozenset({CCA_WB, CCA_CWBE, CCA_CWB})
#: What a CPS core implements and caches.
CCA_CPS = CCA_24K | COHERENT_CCA
CACHED_CCA_CPS = CACHED_CCA | COHERENT_CCA

#: ErrCtl (CP0 26, 0) bits, 24K Table 6.49.
EC_PE, EC_PO, EC_WST, EC_SPR, EC_PCO = 1 << 31, 1 << 30, 1 << 29, 1 << 28, 1 << 27
EC_L2P = 1 << 23
EC_SE, EC_FE = 1 << 20, 1 << 19
EC_PCI_SHIFT, EC_PI_SHIFT = 13, 4

#: Config (CP0 16, 0) MM: write-through merging in the write buffer
#: (24K Table 6.28); Config2 L2B: L2 bypass (24K Table 6.31).
CONFIG_MM = 1 << 18
CONFIG2_L2B = 1 << 12

#: TagLo fields (24K Figures 6.41 / 6.44; L23TagLo uses the same ones).
TAG_V, TAG_D, TAG_L, TAG_P = 1 << 7, 1 << 6, 1 << 5, 1
#: A CPS core's coherent line state in DTagLo (the 1004K, MD00622 9.2.3
#: Table 9.1): V = 0, L = 1, D = 0 is SHARED; V = 1, L = 0: EXCLUSIVE (D =
#: 0) or MODIFIED (D = 1); V = 1, L = 1: locked (locked lines are not
#: coherent); V = 0, L = 1, D = 1 is reserved (eMIPS: invalid).
COHERENT_SHARED = TAG_L

#: CACHE op[17:16] (Vol. II-A Table 3.4) and op[20:18] names (24K Table 12.3).
CACHE_NAMES = {0: "I", 1: "D", 2: "T", 3: "S"}
_D_OPS = ("Index Writeback Invalidate", "Index Load Tag", "Index Store Tag",
          "(reserved)", "Hit Invalidate", "Hit Writeback Invalidate",
          "Hit Writeback", "Fetch and Lock")
OP_NAMES = {
    "I": ("Index Invalidate", "Index Load Tag", "Index Store Tag",
          "(reserved)", "Hit Invalidate", "Fill", "(reserved)",
          "Fetch and Lock"),
    "D": _D_OPS,
    "S": _D_OPS[:3] + ("Index Store Data",) + _D_OPS[4:],
    "T": _D_OPS[:3] + ("Index Store Data",) + _D_OPS[4:],
}
#: Which level each cache kind is.
LEVEL = {"I": 1, "D": 1, "S": 2, "T": 3}

#: The write buffer: 4 entries of 32 bytes (24K 8.4).
WBB_ENTRIES, WBB_LINE = 4, 32


def _parity(value: int) -> int:
    """Even parity bit: makes the number of ones, parity included, even."""
    return bin(value).count("1") & 1


def _log2(n: int) -> int:
    return n.bit_length() - 1


# ---------------------------------------------------------------------------
# SEC-DED ECC: a Hamming code over 64 data bits (7 check bits) plus an
# overall parity bit: corrects any single-bit and detects any double-bit
# error in the 72 stored bits (MD00355 3.4.1: "8-bit-per-doubleword ECC").
# ---------------------------------------------------------------------------

#: Hamming position (1..71, not a power of two) of each data bit.
_ECC_POS = [p for p in range(3, 72) if p & (p - 1)][:64]
_ECC_BIT = {p: i for i, p in enumerate(_ECC_POS)}
#: _ECC_BYTE[k][v]: the syndrome contribution of byte k having value v.
_ECC_BYTE = [[0] * 256 for _ in range(8)]
for _k in range(8):
    for _v in range(256):
        _h = 0
        for _b in range(8):
            if _v >> _b & 1:
                _h ^= _ECC_POS[8 * _k + _b]
        _ECC_BYTE[_k][_v] = _h


def ecc_encode(group: bytes) -> int:
    """The 8 check bits of up to 8 data bytes: Hamming bits 6:0, the
    overall parity in bit 7."""
    h = 0
    ones = 0
    for k, v in enumerate(group):
        h ^= _ECC_BYTE[k][v]
        ones ^= _parity(v)
    return h | ((ones ^ _parity(h)) << 7)


def ecc_check(group: bytes, check: int) -> tuple[str, bytes]:
    """Decode a stored group: ``("ok" | "corrected" | "uncorrectable",
    data)``; a corrected data bit is fixed in the returned data."""
    h, ones = 0, 0
    for k, v in enumerate(group):
        h ^= _ECC_BYTE[k][v]
        ones ^= _parity(v)
    syndrome = h ^ (check & 0x7F)
    overall = ones ^ _parity(check & 0x7F) ^ (check >> 7 & 1)
    if not syndrome and not overall:
        return "ok", bytes(group)
    if not overall:
        return "uncorrectable", bytes(group)          # an even number of flips
    if not syndrome or not syndrome & (syndrome - 1):
        return "corrected", bytes(group)              # a check bit flipped
    bit = _ECC_BIT.get(syndrome)
    if bit is None or bit >= 8 * len(group):
        return "uncorrectable", bytes(group)
    fixed = bytearray(group)
    fixed[bit // 8] ^= 1 << (bit % 8)
    return "corrected", bytes(fixed)


@dataclass(frozen=True)
class Geometry:
    """Sets per way × associativity × line size, as Config1 (level 1:
    IS/DS, IL/DL, IA/DA) or Config2 (level 2 / 3: SS/TS, SL/TL, SA/TA)
    encodes them (Vol. III Config1, Config2)."""
    sets: int
    ways: int
    line: int
    level: int = field(default=1, compare=False)

    def __post_init__(self) -> None:
        if self.level == 1:
            sets, lines, reg = (32, 64, 128, 256, 512, 1024, 2048, 4096), \
                (4, 8, 16, 32, 64, 128), "Config1.IS/DS"
        else:
            sets, lines, reg = tuple(64 << k for k in range(8)), \
                (4, 8, 16, 32, 64, 128, 256), "Config2.SS/TS"
        if self.sets not in sets:
            raise ValueError(f"sets per way must be {sets[0]}, {sets[1]}, ..., "
                             f"{sets[-1]} ({reg}), not {self.sets}")
        if self.line not in lines:
            raise ValueError(f"line size must be 4, 8, ..., {lines[-1]} bytes, "
                             f"not {self.line}")
        most = 8 if self.level == 1 else 16          # (Config2.SA: 8; the P8700's L2: 16)
        if not 1 <= self.ways <= most:
            raise ValueError(f"associativity must be 1..{most} ways, not {self.ways}")
        if self.level == 1 and self.way_bytes < 1024:
            raise ValueError(f"a way of {self.way_bytes} bytes is too small: "
                             f"TagLo holds tags of 1 KB ways and larger")

    @property
    def way_bytes(self) -> int:
        return self.sets * self.line

    @property
    def size(self) -> int:
        return self.way_bytes * self.ways

    def config1(self) -> tuple[int, int, int]:
        """(S, L, A) Config1 codes: sets 64 << S (S = 7: 32 sets), line
        2 << L bytes, A + 1 ways."""
        s = 7 if self.sets == 32 else _log2(self.sets // 64)
        return s, _log2(self.line) - 1, self.ways - 1

    def config2(self) -> tuple[int, int, int]:
        """(S, L, A) Config2 codes: sets 64 << S, line 2 << L, A + 1 ways."""
        return _log2(self.sets // 64), _log2(self.line) - 1, self.ways - 1

    def describe(self) -> str:
        ways = "direct mapped" if self.ways == 1 else f"{self.ways}-way"
        return (f"{_size_text(self.size)}, {ways}, {self.line}-byte lines, "
                f"{self.sets} sets")

    @classmethod
    def parse(cls, text: str, level: int = 1) -> "Geometry | None":
        """``none`` or ``SIZE[:WAYS[:LINE]]`` (``32K``, ``16K:2``,
        ``64K:4:32``, ``1M``); by default 4 ways (L1, the 24K) or 8 ways
        (L2 / L3, the SOC-it L2's largest) and 32-byte lines."""
        text = text.strip().lower()
        if text in ("", "none", "0"):
            return None
        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(f"cache {text!r}: expected SIZE[:WAYS[:LINE]]")
        size_s = parts[0].rstrip("b")
        mult = 1
        if size_s.endswith("k"):
            size_s, mult = size_s[:-1], 1024
        elif size_s.endswith("m"):
            size_s, mult = size_s[:-1], 1024 * 1024
        try:
            size = int(size_s) * mult
            ways = int(parts[1]) if len(parts) > 1 else (4 if level == 1 else 8)
            line = int(parts[2]) if len(parts) > 2 else 32
        except ValueError:
            raise ValueError(f"cache {text!r}: expected SIZE[:WAYS[:LINE]], "
                             f"e.g. 32K or 16K:2:32") from None
        if size <= 0 or ways <= 0 or line <= 0 or size % (ways * line):
            raise ValueError(f"cache {text!r}: {size} bytes is not a whole "
                             f"number of {ways}-way sets of {line}-byte lines")
        return cls(size // (ways * line), ways, line, level)

    def spec(self) -> str:
        size = self.size
        text = f"{size // (1024 * 1024)}M" if size % (1024 * 1024) == 0 else f"{size // 1024}K"
        return f"{text}:{self.ways}:{self.line}"


def _size_text(size: int) -> str:
    if size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)} MB"
    return f"{size // 1024} KB"


#: The 24K's caches come in 8, 16, 32 and 64 KB, 4-way, 32-byte lines.
GEOMETRY_24K = Geometry(256, 4, 32)
#: A SOC-it L2 as fitted to 24K systems: 256 KB, 8-way, 32-byte lines.
GEOMETRY_L2 = Geometry(1024, 8, 32, 2)


class CacheError(Exception):
    """An error found in a cache array (→ Cache Error exception).  The
    fields are the CacheErr fields (24K Tables 6.50 / 6.51)."""

    def __init__(self, cache: "Cache", *, data: bool = False, tag: bool = False,
                 way_select: bool = False, fatal: bool = False, way: int = 0,
                 index: int = 0, what: str = "") -> None:
        super().__init__(what)
        self.cache = cache
        self.data, self.tag, self.way_select, self.fatal = data, tag, way_select, fatal
        self.way, self.index, self.what = way, index, what

    def cacheerr(self) -> int:
        """The CacheErr register: the primary-cache format (EC = 0), or
        the secondary-cache format (EC = 1) for an L2 / L3 error."""
        if self.cache.level == 1:
            return ((int(self.cache.kind == "D") << 31) | (int(self.data) << 29)
                    | (int(self.tag) << 28) | (int(self.fatal) << 24)
                    | (int(self.way_select) << 22) | ((self.way & 3) << 20)
                    | (self.index & 0xFFFFF))
        return ((1 << 30) | (int(self.data) << 29) | (int(self.tag) << 28)
                | (int(self.fatal) << 24) | (int(self.way_select) << 22)
                | ((self.way & 7) << 19) | (self.index & 0x7FFFF))


class Line:
    """One cache line: tag, state bits, data and (with parity / ECC) the
    stored check bits.  ``known`` marks the bytes that came from memory
    that had been written (a fetch of an unknown byte is still a
    simulator halt, as without caches)."""

    __slots__ = ("tag", "valid", "dirty", "lock", "shared", "data", "known",
                 "tag_p", "data_p", "dirty_p", "pre", "pre_p", "ecc")

    def __init__(self, line: int) -> None:
        self.tag = 0
        self.valid = self.dirty = self.lock = False
        #: MESI (a data cache in a CPS): valid + shared = S, valid + dirty =
        #: M, valid alone = E; never set outside a CPS
        self.shared = False
        self.data = bytearray(line)
        self.known = 0
        self.tag_p = 0
        self.data_p = 0              # bit i = stored parity of byte i
        self.dirty_p = 0
        self.pre: list[int] = []     # I-cache precode, per doubleword
        self.pre_p: list[int] = []
        self.ecc: list[int] = []     # L2 / L3: check bits per doubleword

    def tag_parity(self) -> int:
        """Parity over the tag entry: PA bits, valid and lock (24K Fig. 8.1)."""
        return _parity((self.tag << 2) | (int(self.valid) << 1) | int(self.lock))


#: Names of the cache levels in the access records ("U": an uncached
#: access, straight to memory).
LEVEL_NAMES = {"I": "I-cache", "D": "D-cache", "S": "L2 cache", "T": "L3 cache",
               "U": "uncached"}
#: Miss classes (Hill & Smith 1989, "Evaluating Associativity in CPU
#: Caches"; Patterson & Hennessy 5.4: the 3 C's) and the fourth kind a
#: single core has: lines the program itself invalidated (CACHE, SYNCI,
#: PREF 25) -- the uniprocessor form of a coherence miss.
MISS_CLASSES = ("compulsory", "capacity", "conflict", "invalidated")


@dataclass(slots=True)
class CacheAccess:
    """One lookup, as the views show it (observed, never recomputed).

    ``kind``: load / store / fetch / pref / synci for an L1, read / write
    for an L2 / L3 (a line request or a write from the level above),
    load / store / fetch for an access that bypasses the caches (``level``
    "U"; ``miss`` "no cache" when its attribute is cacheable but no cache
    serves it).
    ``way`` is ``None`` for a miss that allocated nothing (write-through
    or L2 write miss, a locked way 0, ...).  ``miss`` is the miss class
    (:data:`MISS_CLASSES`) of a miss that filled a line; ``victim`` the
    physical address of the valid line it replaced and ``victim_dirty``
    whether that line was written back."""

    level: str
    kind: str
    va: int
    pa: int
    set: int | None = None
    way: int | None = None
    tag: int | None = None
    hit: bool = False
    miss: str | None = None
    victim: int | None = None
    victim_dirty: bool = False
    offset: int = 0
    #: what the Coherence Manager did for it (interventions), or None
    coherence: str | None = None

    @property
    def name(self) -> str:
        return LEVEL_NAMES[self.level]

    def describe(self) -> str:
        """One line: "D-cache load 0x... set 3 way 1: miss (compulsory)"."""
        if self.level == "U":
            if self.miss == "no cache":
                return f"{self.kind} 0x{self.pa:08x}: no cache for it, straight to memory"
            return f"uncached {self.kind} 0x{self.pa:08x}: straight to memory"
        where = f" set {self.set}" + (f" way {self.way}" if self.way is not None else "")
        if self.hit:
            what = "hit"
        elif self.miss is not None:
            what = f"MISS ({self.miss})"
            if self.victim is not None:
                what += (f", replaces line 0x{self.victim:08x}"
                         + (" (dirty: written back)" if self.victim_dirty else ""))
        else:
            what = "miss, no allocate"
        if self.coherence:
            what += f"; {self.coherence}"
        return f"{self.name} {self.kind} 0x{self.pa:08x}{where}: {what}"


class CacheStats:
    """Counters of one cache: hits / misses per kind of access, misses
    that filled a line classified as

    * **compulsory** -- the first reference to the line (since reset);
    * **capacity** -- a fully associative LRU cache with as many lines
      would have missed too;
    * **conflict** -- that fully associative cache would have hit: the
      line was thrown out only because its set was full;
    * **invalidated** -- the program removed it (CACHE, SYNCI, PREF 25).

    The fully associative cache is a shadow fed with the same references
    and the same allocation policy (Hill & Smith); misses that allocate
    nothing (write-through / L2 write misses, SYNCI) are not classified
    (:attr:`no_allocate`).  Lines placed by CACHE Fill / Fetch and Lock / PREF
    PrepareForStore are known to the shadow too."""

    def __init__(self, lines: int) -> None:
        self.capacity = lines
        #: kind -> [hits, misses]
        self.kinds: dict[str, list[int]] = {}
        self.classes = dict.fromkeys(MISS_CLASSES, 0)
        self.evictions = 0
        self._seen: set[int] = set()
        self._fa: OrderedDict[int, None] = OrderedDict()
        self._gone: set[int] = set()

    @property
    def hits(self) -> int:
        return sum(h for h, _ in self.kinds.values())

    @property
    def misses(self) -> int:
        return sum(m for _, m in self.kinds.values())

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def no_allocate(self) -> int:
        """Misses that filled nothing (no write allocate, SYNCI, ...)."""
        return self.misses - sum(self.classes.values())

    def reference(self, kind: str, line: int, hit: bool, fill: bool) -> str | None:
        """Count one lookup of the line at physical address ``line``;
        returns the miss class when it missed and filled."""
        k = self.kinds.get(kind)
        if k is None:
            k = self.kinds[kind] = [0, 0]
        fa = self._fa
        if hit:                                  # the common case: kept short
            k[0] += 1
            if line in fa:
                fa.move_to_end(line)
            else:
                self._seen.add(line)
                self._insert(line)
            return None
        k[1] += 1
        first = line not in self._seen
        if first:
            self._seen.add(line)
        in_fa = line in fa
        if in_fa:
            fa.move_to_end(line)
        elif fill:
            self._insert(line)
        if not fill:
            return None
        if line in self._gone:
            self._gone.discard(line)
            cls = "invalidated"
        elif first:
            cls = "compulsory"
        elif not in_fa:
            cls = "capacity"
        else:
            cls = "conflict"
        self.classes[cls] += 1
        return cls

    def _insert(self, line: int) -> None:
        fa = self._fa
        fa[line] = None
        if len(fa) > self.capacity:
            fa.popitem(last=False)

    def place(self, line: int) -> None:
        """A line was filled (by a lookup or by CACHE / PREF)."""
        self._seen.add(line)
        self._gone.discard(line)
        if line in self._fa:
            self._fa.move_to_end(line)
        else:
            self._insert(line)

    def forget(self, line: int) -> None:
        """The program invalidated a line."""
        if line in self._seen:
            self._gone.add(line)
            self._fa.pop(line, None)


def precode(word: int) -> int:
    """eMIPS's 3 precode bits of an instruction (the 24K stores 6 per
    doubleword but does not publish their encoding): bit 0 = branch or
    jump, bit 1 = register jump (JR / JALR), bit 2 = it links."""
    op = word >> 26
    funct, rt_ = word & 0x3F, (word >> 16) & 31
    if op == 0 and funct in (8, 9):
        return 0b011 | (0b100 if funct == 9 else 0)
    if op in (2, 3):
        return 0b001 | (0b100 if op == 3 else 0)
    if op in (4, 5, 6, 7, 20, 21, 22, 23):
        return 0b001
    if op == 1 and rt_ in (0, 1, 2, 3, 16, 17, 18, 19):
        return 0b001 | (0b100 if rt_ >= 16 else 0)
    if op in (0x11, 0x12) and (word >> 21) & 31 == 8:      # BC1 / BC2
        return 0b001
    return 0


class Cache:
    """One L1 cache (``kind`` "I" or "D") of the given :class:`Geometry`."""

    level = 1
    #: a data cache of a CPS core: its lines have MESI states
    mesi = False
    #: the bytes one check covers when a partial store must read them first
    #: (0: per-byte parity; the P8700's ECC: 4, emips.p8700err)
    WORD = 0
    #: the physical address space (LF13: the machine's PABITS; MIPS 32)
    pamask = M32

    def __init__(self, kind: str, geometry: Geometry, *, parity: bool = False) -> None:
        self.kind = kind
        self.geometry = g = geometry
        self.parity = parity
        #: holds data written by the CPU (has dirty bits)
        self.writable = kind != "I"
        self.sets, self.ways, self.line = g.sets, g.ways, g.line
        self.line_shift = _log2(g.line)
        self.line_mask = g.line - 1
        self.index_bit = _log2(g.way_bytes)            # IndexBit (CACHE)
        self.way_bits = (g.ways - 1).bit_length()      # Ceiling(Log2(A))
        #: tag = PA >> tag_shift: PA[31:12], or down to the way size
        #: for ways under 4 KB (the 24K's extra tag bit 11 of 8 KB caches)
        self.tag_shift = min(12, self.index_bit)
        self.pairs = _pairs(g.ways)
        self.plru = False
        self._touch_masks: list[tuple[int, int]] | None = None
        #: an MT processor's way exclusion (MVPControl.CPA, VPEOpt): a
        #: function -> the ways the refilling VPE may not allocate, or None
        self.excluded: Callable[[], int] | None = None
        self.lines = [[Line(g.line) for _ in range(g.ways)] for _ in range(g.sets)]
        #: way-select RAM: the LRU bits of each set
        self.ws = [0] * g.sets
        #: ErrCtl.PE as last seen by the machine (checks parity on reads)
        self.check = False
        self.hits = self.misses = self.fills = self.writebacks = self.evictions = 0
        self.stats = CacheStats(g.sets * g.ways)
        #: while watched (Caches.watch): every lookup as a CacheAccess
        self.log: list[CacheAccess] | None = None

    @property
    def name(self) -> str:
        return {"I": "I-cache", "D": "D-cache", "S": "L2 cache", "T": "L3 cache"}[self.kind]

    # -- address fields --------------------------------------------------

    def set_of(self, va: int) -> int:
        return (va >> self.line_shift) & (self.sets - 1)

    def tag_of(self, pa: int) -> int:
        return (pa & self.pamask) >> self.tag_shift

    def line_pa(self, tag: int, s: int) -> int:
        """Physical address of a line from its tag and set (the
        write-back address)."""
        low = (s << self.line_shift) & ((1 << self.tag_shift) - 1)
        return ((tag << self.tag_shift) | low) & self.pamask

    def alias_sets(self, pa: int) -> list[int]:
        """Every set a physical address can sit in: the index bits above
        bit 11 come from the virtual address (virtual aliases, 24K 8.3.1)."""
        s = (pa >> self.line_shift) & (self.sets - 1)
        if self.index_bit <= 12:
            return [s]
        low = (1 << (12 - self.line_shift)) - 1
        return [(s & low) | (h << (12 - self.line_shift))
                for h in range(1 << (self.index_bit - 12))]

    # -- LRU (way-select RAM, 24K Table 8.4; tree pseudo-LRU) -------------

    def newer(self, s: int, a: int, b: int) -> bool:
        """Was way ``a`` used more recently than way ``b``?"""
        if a == b:
            return False
        if self.plru:
            ws = self.ws[s]
            for (na, da), (_, db) in zip(_tree_path(a, self.tree), _tree_path(b, self.tree)):
                if da != db:
                    return (ws >> na & 1) == da
            return False
        i, j = (a, b) if a < b else (b, a)
        bit = (self.ws[s] >> self.pairs.index((i, j))) & 1
        return bool(bit) == (a == i)

    def touch(self, s: int, way: int) -> None:
        """Make ``way`` the most recently used of its set (the bits it sets
        and clears are remembered per way: :meth:`_touch_slow`)."""
        masks = self._touch_masks
        if masks is None:                   # (built on first use: SecondaryCache sets plru)
            every = (1 << 64) - 1
            masks = self._touch_masks = [
                (self._touch_slow(0, w, 0), every & ~self._touch_slow(0, w, every))
                for w in range(self.ways)]
        on, off = masks[way]
        self.ws[s] = (self.ws[s] | on) & ~off

    def _touch_slow(self, s: int, way: int, ws: int | None = None) -> int:
        """The WS bits of set ``s`` (or ``ws``) with ``way`` made the most
        recently used."""
        ws = self.ws[s] if ws is None else ws
        if self.plru:
            for node, d in _tree_path(way, self.tree):
                ws = ws | (1 << node) if d else ws & ~(1 << node)
        else:
            for b, (i, j) in enumerate(self.pairs):
                if i == way:
                    ws |= 1 << b
                elif j == way:
                    ws &= ~(1 << b)
        return ws

    def make_lru(self, s: int, way: int) -> None:
        ws = self.ws[s]
        if self.plru:
            for node, d in _tree_path(way, self.tree):
                ws = ws & ~(1 << node) if d else ws | (1 << node)
        else:
            for b, (i, j) in enumerate(self.pairs):
                if i == way:
                    ws &= ~(1 << b)
                elif j == way:
                    ws |= 1 << b
        self.ws[s] = ws

    def order(self, s: int) -> list[int]:
        """Ways from least to most recently used (for the views)."""
        return sorted(range(self.ways),
                      key=lambda w: sum(self.newer(s, w, o) for o in range(self.ways)))

    def victim(self, s: int, *, exclude_way0: bool = False) -> int:
        """The way a fill replaces: the least recently used one that is not
        a valid locked line; when every candidate is locked, one of the
        locked lines is displaced (24K 8.5.4).  An LRU state that names no
        single oldest way (an invalid WS encoding written by Index Store
        WS) gives the candidate older than the most others, lowest way
        first.  The pseudo-LRU tree is walked towards its older side,
        unless only the other side has a candidate."""
        ways = range(1 if exclude_way0 and self.ways > 1 else 0, self.ways)
        if self.excluded is not None:           # an MT VPE's VPEOpt.IWX / DWX
            ex = self.excluded()
            if ex:
                ways = [w for w in ways if not ex >> w & 1] or list(ways)
        cand = [w for w in ways if not (self.lines[s][w].valid and self.lines[s][w].lock)]
        if not cand:
            cand = list(ways)
        if self.plru:
            ws, lo, hi = self.ws[s], 0, self.tree
            while hi - lo > 1:
                mid = (lo + hi) // 2
                width = hi - lo
                node = self.tree // width - 1 + lo // width     # (heap order)
                older = 1 - (ws >> node & 1)                 # the side not used last
                sides = ((lo, mid), (mid, hi))
                a, b = sides[older]
                if not any(a <= w < b for w in cand):
                    a, b = sides[1 - older]
                lo, hi = a, b
            return lo
        return max(cand, key=lambda w: (sum(self.newer(s, o, w) for o in cand), -w))

    # -- parity ----------------------------------------------------------

    def _fail(self, err: CacheError) -> None:
        """An error found in an array: precise for the L1 caches."""
        raise err

    def _set_data_parity(self, ln: Line, start: int, count: int) -> None:
        if self.parity:
            for i in range(start, start + count):
                bit = 1 << i
                ln.data_p = ln.data_p | bit if _parity(ln.data[i]) else ln.data_p & ~bit

    def _check_tags(self, s: int) -> None:
        """Reading the tags of a set (every lookup) checks their parity."""
        for w, ln in enumerate(self.lines[s]):
            if ln.valid and ln.tag_p != ln.tag_parity():
                self._fail(CacheError(self, tag=True, way=w, index=s << self.line_shift,
                                      what=f"{self.name} tag parity error, set {s} way {w}"))
                return

    def _check_data(self, s: int, w: int, start: int, count: int, *,
                    fatal: bool = False) -> None:
        ln = self.lines[s][w]
        for i in range(start, start + count):
            if (ln.data_p >> i) & 1 != _parity(ln.data[i]):
                raise CacheError(self, data=True, fatal=fatal, way=w,
                                 index=(s << self.line_shift) | (i & ~7 & self.line_mask),
                                 what=f"data parity error, set {s} way {w} byte {i}")

    def _dirty(self, s: int, w: int, *, fatal: bool = False) -> bool:
        """Read a dirty bit (checks its parity in the way-select RAM)."""
        ln = self.lines[s][w]
        if self.check and self.parity and ln.dirty_p != int(ln.dirty):
            self._fail(CacheError(self, way_select=True, fatal=fatal, way=w,
                                  index=s << self.line_shift,
                                  what=f"{self.name} dirty-bit parity error, set {s} way {w}"))
        return ln.dirty

    def _seal(self, ln: Line) -> None:
        """Recompute the tag and dirty parity after a state change."""
        if self.parity:
            ln.tag_p = ln.tag_parity()
            ln.dirty_p = int(ln.dirty)

    # -- lookups ---------------------------------------------------------

    def find(self, s: int, tag: int) -> int | None:
        """Way of the valid line with this tag (no side effects); when
        Index Store Tag made two ways match, the lowest one."""
        for w, ln in enumerate(self.lines[s]):
            if ln.valid and ln.tag == tag:
                return w
        return None

    def probe(self, va: int, pa: int) -> Line | None:
        """The line holding ``pa`` (set from ``va``), no side effects."""
        w = self.find(self.set_of(va), self.tag_of(pa))
        return None if w is None else self.lines[self.set_of(va)][w]

    def lookup(self, va: int, pa: int, bus, *, allocate: bool = True,
               way0: bool = False, not_way0: bool = False,
               kind: str = "read", zero: bool = False) -> tuple[int, int | None]:
        """``(set, way)`` holding ``pa``; a miss fills a line when
        ``allocate`` (way ``None`` otherwise; ``zero``: with zeros, dirty —
        the caller writes all of it).  Counts hits and misses
        (:class:`CacheStats`, by ``kind`` of access) and, while watched,
        records a :class:`CacheAccess`."""
        s, tag = self.set_of(va), self.tag_of(pa)
        if self.check and self.parity:
            self._check_tags(s)
        w = self.find(s, tag)
        line = pa & ~self.line_mask & self.pamask
        if w is not None:
            self.hits += 1
            self.touch(s, w)
            self.stats.reference(kind, line, True, False)
            if self.log is not None:
                self.log.append(CacheAccess(self.kind, kind, va & M32, pa & self.pamask, s, w,
                                            tag, True, offset=pa & self.line_mask))
            return s, w
        self.misses += 1
        if allocate and way0 and self.lines[s][0].valid and self.lines[s][0].lock:
            allocate = False
        cls = self.stats.reference(kind, line, False, allocate)
        rec = None
        if self.log is not None:
            rec = CacheAccess(self.kind, kind, va & M32, pa & self.pamask, s, None, tag, False,
                              cls, offset=pa & self.line_mask)
            self.log.append(rec)
        if not allocate:
            return s, None
        w = 0 if way0 else self.victim(s, exclude_way0=not_way0)
        if rec is not None:
            rec.way = w
            old = self.lines[s][w]
            if old.valid:
                rec.victim = self.line_pa(old.tag, s)
                rec.victim_dirty = self.writable and old.dirty
        self.replace(s, w, tag, bus, zero=zero)
        return s, w

    def note_hit(self, kind: str, va: int, pa: int, s: int, w: int) -> None:
        """Count (and record) a hit found without :meth:`lookup`."""
        self.stats.reference(kind, pa & ~self.line_mask & self.pamask, True, False)
        if self.log is not None:
            self.log.append(CacheAccess(self.kind, kind, va & M32, pa & self.pamask, s, w,
                                        self.tag_of(pa), True, offset=pa & self.line_mask))

    def replace(self, s: int, w: int, tag: int, bus, *, zero: bool = False) -> None:
        """Evict way ``w`` of set ``s`` (writing a dirty victim back) and
        fill it with the line of ``tag`` (or zeros: PrepareForStore)."""
        ln = self.lines[s][w]
        if ln.valid:
            self.evictions += 1
            self.stats.evictions += 1
        if ln.valid and self.writable and self._dirty(s, w, fatal=True):
            if self.check and self.parity:
                if ln.tag_p != ln.tag_parity():
                    self._fail(CacheError(
                        self, tag=True, fatal=True, way=w, index=s << self.line_shift,
                        what=f"{self.name} tag parity error in a dirty victim, "
                             f"set {s} way {w}"))
                self._check_data(s, w, 0, self.line, fatal=True)
            self.write_back(s, w, bus)
        pa = self.line_pa(tag, s)
        if zero:
            ln.data[:] = bytes(self.line)
            ln.known = (1 << self.line) - 1
        else:
            data, known = bus.bus_read_line(pa, self.line)
            ln.data[:] = data
            ln.known = known
        ln.tag, ln.valid, ln.dirty, ln.lock = tag, True, zero, False
        ln.shared = False
        self._set_data_parity(ln, 0, self.line)
        if self.kind == "I":
            self._precode(ln, 0, self.line, bus)
        self._seal(ln)
        self.touch(s, w)
        self.fills += 1
        self.stats.place(pa)

    def line_data(self, s: int, w: int) -> bytes:
        """The data a write-back sends (L2 / L3: ECC-corrected)."""
        return bytes(self.lines[s][w].data)

    def write_back(self, s: int, w: int, bus) -> None:
        ln = self.lines[s][w]
        bus.bus_write_line(self.line_pa(ln.tag, s), bytearray(self.line_data(s, w)), ln.known)
        ln.dirty = False
        self._seal(ln)
        self.writebacks += 1

    def _precode(self, ln: Line, start: int, end: int, bus) -> None:
        n = max(self.line // 8, 1)
        if len(ln.pre) != n:
            ln.pre, ln.pre_p = [0] * n, [0] * n
        order = "big" if bus.big_endian else "little"
        for d in range(start // 8, (end + 7) // 8):
            if d >= n:
                break
            words = [int.from_bytes(ln.data[a:a + 4], order)
                     for a in range(8 * d, min(8 * d + 8, self.line), 4)]
            ln.pre[d] = sum(precode(word) << (3 * k) for k, word in enumerate(words))
            ln.pre_p[d] = _parity(ln.pre[d])

    # -- contents --------------------------------------------------------

    def invalidate_all(self) -> None:
        """Boot firmware: Index Store Tag, TagLo = 0, way by way and set by
        set (each stored invalid tag becomes the LRU way)."""
        for s in range(self.sets):
            for ln in self.lines[s]:
                ln.tag = 0
                ln.valid = ln.dirty = ln.lock = False
                self._seal(ln)
        # the LRU state every set ends in (one set computed, all copied)
        self.ws[0] = 0
        for w in range(self.ways):
            self.make_lru(0, w)
        self.ws[:] = [self.ws[0]] * self.sets
        self.hits = self.misses = self.fills = self.writebacks = self.evictions = 0
        self.stats = CacheStats(self.sets * self.ways)

    def randomize(self, rng: random.Random) -> None:
        """Power-up contents: every array bit undefined."""
        lru_bits = self.tree - 1 if self.plru else len(self.pairs)
        for s in range(self.sets):
            for ln in self.lines[s]:
                ln.tag = rng.getrandbits(32 - self.tag_shift)
                ln.valid = rng.random() < 0.5
                ln.dirty = self.writable and rng.random() < 0.5
                ln.lock = rng.random() < 0.05
                ln.data[:] = rng.randbytes(self.line)
                ln.known = (1 << self.line) - 1
                if self.parity:
                    ln.tag_p, ln.dirty_p = rng.getrandbits(1), rng.getrandbits(1)
                    ln.data_p = rng.getrandbits(self.line)
                    self._random_check_bits(ln, rng)
                if self.kind == "I":
                    n = max(self.line // 8, 1)
                    ln.pre = [rng.getrandbits(6) for _ in range(n)]
                    ln.pre_p = [rng.getrandbits(1) for _ in range(n)]
            self.ws[s] = rng.getrandbits(lru_bits) if lru_bits else 0

    def _random_check_bits(self, ln: Line, rng: random.Random) -> None:
        pass

    def inject(self, s: int, w: int, *, array: str = "data", byte: int = 0,
               bit: int = 0) -> None:
        """Fault injection: flip one bit of the data (``byte``, ``bit``),
        the tag, the dirty bit or (L2 / L3) the ECC check bits of a line
        *without* updating its check bits, as a particle strike does."""
        ln = self.lines[s][w]
        if array == "data":
            ln.data[byte] ^= 1 << bit
        elif array == "tag":
            ln.tag ^= 1 << bit
        elif array == "dirty":
            ln.dirty = not ln.dirty
        elif array == "ecc" and ln.ecc:
            ln.ecc[byte // 8] ^= 1 << bit
        else:
            raise ValueError(f"array must be data, tag, dirty or ecc (L2 / L3), "
                             f"not {array!r}")


def _tree_path(w: int, ways: int = 8) -> tuple[tuple[int, int], ...]:
    """(node, direction) from the root to way ``w`` in the pseudo-LRU tree
    of ``ways`` (8: 7 bits, 16: 15 bits) in heap order: node 0 splits the
    ways in halves, nodes 1-2 the halves, ... (8 ways: nodes 3-6 the
    pairs); a node bit names the side used last."""
    depth = ways.bit_length() - 1
    return tuple(((1 << d) - 1 + (w >> (depth - d)), w >> (depth - d - 1) & 1)
                 for d in range(depth))


class SecondaryCache(Cache):
    """An L2 (``kind`` "S") or L3 ("T") cache: physically indexed and
    tagged, write-back without write allocate, read-allocating, optional
    ECC; its errors are recorded and taken imprecisely
    (:attr:`Caches.pending`).  Below it is ``below`` (the next level or
    memory)."""

    def __init__(self, kind: str, geometry: Geometry, *, ecc: bool = False) -> None:
        super().__init__(kind, geometry, parity=ecc)
        self.level = LEVEL[kind]
        self.tag_shift = self.index_bit           # PIPT: no index bits in the tag
        self.plru = geometry.ways > 4
        #: the pseudo-LRU tree's ways (8 up to 8 ways, 16 above)
        self.tree = 1 << (geometry.ways - 1).bit_length()
        self.groups = max(self.line // 8, 1)
        self.group = min(8, self.line)
        if ecc:
            for row in self.lines:
                for ln in row:
                    ln.ecc = [ecc_encode(bytes(self.group))] * self.groups
        self.below = None
        #: where errors go (Caches.record)
        self.sink = None
        #: the hierarchy's emips.timing.Timing
        self.timing = None
        #: single-bit errors corrected (24K performance counter event 22)
        self.corrected = 0
        #: (cache, set, way, doubleword, bit) of each correction, or None
        self.on_corrected = None
        #: a CPS's shared L2: pa -> "wb" / "wt" / "bypass" (the CM's address
        #: regions, emips.coherence)
        self.policy = None
        #: a write miss allocates the line (the P8700's L2, MD01502 3.3.8: the
        #: rest of the line read from below unless the write covers it)
        self.write_allocate = False
        #: ``observer(pa, miss)`` after every line a read looks up (the
        #: P8700's L2 prefetcher, emips.p8700mem)
        self.observer = None
        #: ``perf(kind, hit, full, allocated)`` after every line looked up (the
        #: P8700 CM's performance counters, emips.p8700oci), or None
        self.perf = None

    def set_of(self, va: int) -> int:
        return (va >> self.line_shift) & (self.sets - 1)

    def alias_sets(self, pa: int) -> list[int]:
        return [self.set_of(pa)]

    def peek(self, pa: int) -> tuple[int, bool] | None:
        """(byte, known) of the copy of ``pa``, no side effects."""
        ln = self.probe(pa, pa)
        if ln is None:
            return None
        off = pa & self.line_mask
        return ln.data[off], bool(ln.known >> off & 1)

    # -- errors: recorded, taken imprecisely ---------------------------------

    def _fail(self, err: CacheError) -> None:
        if self.sink is not None:
            self.sink(err)

    def _set_data_parity(self, ln: Line, start: int, count: int) -> None:
        if self.parity:
            g = self.group
            for k in range(start // g, min((start + count + g - 1) // g, self.groups)):
                ln.ecc[k] = ecc_encode(bytes(ln.data[k * g:(k + 1) * g]))

    def _random_check_bits(self, ln: Line, rng: random.Random) -> None:
        ln.ecc = [rng.getrandbits(8) for _ in range(self.groups)]

    def _read_group(self, s: int, w: int, k: int) -> bytes:
        """Doubleword ``k`` of a line as the array delivers it: corrected
        (1-bit error) or raw with an error recorded (2-bit error)."""
        ln = self.lines[s][w]
        g = self.group
        raw = bytes(ln.data[k * g:(k + 1) * g])
        if not (self.check and self.parity):
            return raw
        status, data = ecc_check(raw, ln.ecc[k])
        if status == "corrected":
            self.corrected += 1
            if self.on_corrected is not None:
                bit = next((8 * i + (a ^ b).bit_length() - 1
                            for i, (a, b) in enumerate(zip(raw, data)) if a != b), 0x3F)
                self.on_corrected(self, s, w, k, bit)
        elif status == "uncorrectable":
            self._fail(CacheError(self, data=True, way=w,
                                  index=(s << self.line_shift) | (k * g),
                                  what=f"{self.name} uncorrectable ECC error, "
                                       f"set {s} way {w} doubleword {k}"))
        return data

    def _check_data(self, s: int, w: int, start: int, count: int, *,
                    fatal: bool = False) -> None:
        g = self.group
        for k in range(start // g, min((start + count + g - 1) // g, self.groups)):
            self._read_group(s, w, k)

    def line_data(self, s: int, w: int) -> bytes:
        if not (self.check and self.parity):
            return bytes(self.lines[s][w].data)
        saved = self.corrected
        data = b"".join(ecc_check(bytes(self.lines[s][w].data[k * self.group:
                                                              (k + 1) * self.group]),
                                  self.lines[s][w].ecc[k])[1]
                        for k in range(self.groups))
        self.corrected = saved                    # counted by _check_data
        return data

    # -- the port the level above uses ---------------------------------------

    @property
    def big_endian(self) -> bool:
        return self.below.big_endian

    def bus_read_line(self, pa: int, n: int) -> tuple[bytearray, int]:
        """A read request from above: every line it covers is looked up
        and, on a miss, filled (read allocate)."""
        if self.policy is not None and self.policy(pa) == "bypass":
            return self.below.bus_read_line(pa, n)
        out = bytearray(n)
        known = 0
        i = 0
        while i < n:
            a = (pa + i) & self.pamask
            off = a & self.line_mask
            k = min(n - i, self.line - off)
            before = self.misses
            s, w = self.lookup(a, a, self.below, kind="read")
            if self.misses == before and self.timing is not None:
                self.timing.served("L2" if self.kind == "S" else "L3")
            if self.observer is not None:
                self.observer(a, self.misses != before)
            if self.perf is not None:
                self.perf("read", self.misses == before, True, True)
            ln = self.lines[s][w]
            g = self.group
            for d in range(off // g, (off + k + g - 1) // g):
                data = self._read_group(s, w, d)
                lo, hi = max(off, d * g), min(off + k, (d + 1) * g)
                out[i + lo - off:i + hi - off] = data[lo - d * g:hi - d * g]
            known |= ((ln.known >> off) & ((1 << k) - 1)) << i
            i += k
        return out, known

    def bus_write_line(self, pa: int, data: bytes, known: int, *,
                       through: bool = False) -> None:
        """A write from above (an eviction, a write-through store): a hit
        updates the line (dirty unless ``through``), a miss goes on below
        (no write allocate) or, with :attr:`write_allocate`, fills the line
        first; ``through`` writes go on below as well."""
        if self.policy is not None:
            mode = self.policy(pa)
            if mode == "bypass":
                _write_below(self.below, pa, data, known, through)
                return
            through = through or mode == "wt"
        n = len(data)
        i = 0
        while i < n:
            a = (pa + i) & self.pamask
            off = a & self.line_mask
            k = min(n - i, self.line - off)
            part = (known >> i) & ((1 << k) - 1)
            if part:
                whole = k == self.line and part == (1 << k) - 1    # (nothing to read)
                before = self.misses
                s, w = self.lookup(a, a, self.below, kind="write",
                                   allocate=self.write_allocate and not through, zero=whole)
                if self.perf is not None:
                    self.perf("write", self.misses == before, whole, w is not None)
                if w is not None:
                    self._merge(s, w, off, data[i:i + k], part)
                    if not through:
                        self.lines[s][w].dirty = True
                        self._seal(self.lines[s][w])
                if w is None or through:
                    _write_below(self.below, a, data[i:i + k], part, through)
            i += k

    def _merge(self, s: int, w: int, off: int, data: bytes, known: int) -> None:
        """Write the known bytes into a line: a doubleword written only in
        part is read first (read-modify-write of its check bits)."""
        ln = self.lines[s][w]
        g = self.group
        full = (1 << g) - 1
        for d in range(off // g, (off + len(data) + g - 1) // g):
            mask = 0
            for b in range(g):
                j = d * g + b - off
                if 0 <= j < len(data) and known >> j & 1:
                    mask |= 1 << b
            if not mask:
                continue
            if mask != full and self.parity:
                ln.data[d * g:(d + 1) * g] = self._read_group(s, w, d)
            for b in range(g):
                if mask >> b & 1:
                    ln.data[d * g + b] = data[d * g + b - off]
            ln.known |= mask << (d * g)
            if self.parity:
                ln.ecc[d] = ecc_encode(bytes(ln.data[d * g:(d + 1) * g]))


def _write_below(port, pa: int, data: bytes, known: int, through: bool) -> None:
    if isinstance(port, SecondaryCache):
        port.bus_write_line(pa, data, known, through=through)
    else:
        port.bus_write_line(pa, data, known)


class Memory:
    """Physical memory and the devices on the system bus."""

    def __init__(self, caches: "Caches") -> None:
        self.c = caches

    @property
    def big_endian(self) -> bool:
        return self.c.m.big_endian

    def bus_read_line(self, pa: int, n: int) -> tuple[bytearray, int]:
        self.c.timing.served("MEM")
        m = self.c.m
        mem = m.mem.get_memory()
        data = bytearray(n)
        known = 0
        devices = m.devices
        for i in range(n):
            a = (pa + i) & self.c.pamask
            dev = next((d for d in devices if d.covers(a)), None) if devices else None
            if dev is not None:
                data[i] = dev.load(m, a, 1) & 0xFF
                known |= 1 << i
                continue
            raw = mem.get(format(a, "x"))
            if raw is not None:
                data[i] = int(raw, 16)
                known |= 1 << i
        return data, known

    def bus_write_line(self, pa: int, data: bytes, known: int) -> None:
        m = self.c.m
        for i, byte in enumerate(data):
            if known >> i & 1:
                a = (pa + i) & self.c.pamask
                dev = next((d for d in m.devices if d.covers(a)), None) if m.devices else None
                if dev is not None:
                    dev.store(m, a, byte, 1)
                else:
                    m.phys_write_byte(a, byte)

    def peek(self, pa: int) -> int | None:
        m = self.c.m
        board = m.board
        if board is not None and board.bus.covers(pa & self.c.pamask):
            return board.bus.peek(m, pa & self.c.pamask, 1)        # None: nothing there
        raw = m.mem.get_memory().get(format(pa & self.c.pamask, "x"))
        return None if raw is None else int(raw, 16)


class _Entry:
    """A write buffer entry: one 32-byte block, the bytes written so far
    (``mask``) and, for gathering, the next byte to fill."""

    __slots__ = ("kind", "base", "data", "mask", "ptr")

    def __init__(self, kind: str, base: int) -> None:
        self.kind, self.base = kind, base
        self.data = bytearray(WBB_LINE)
        self.mask = 0
        self.ptr = 0

    def put(self, off: int, data: bytes) -> None:
        self.data[off:off + len(data)] = data
        self.mask |= ((1 << len(data)) - 1) << off

    def byte(self, pa: int) -> int | None:
        off = pa - self.base
        if 0 <= off < WBB_LINE and self.mask >> off & 1:
            return self.data[off]
        return None


class WriteBuffer:
    """The 24K's write back buffer (MD00343 8.4, 8.4.1): 4 entries of 32
    bytes between the core and the system.

    * Evictions, uncached stores and L2 CACHE operations are flushed
      immediately (in this untimed model: they reach the next level at
      once; their cycles come with the memory timing, M9 phase 4).
    * Write-through stores are merged into one entry while
      ``Config.MM = 1`` ("only 1 WT merge can be active at any time");
      the entry is flushed when the 32 bytes are complete, when a read
      request matches its block, when a write-through store to another
      block comes, and by SYNC.
    * Uncached accelerated word / doubleword stores are gathered: one
      starting a 32-byte block reserves an entry, the following ones to
      the same block are written **sequentially** into it, whatever their
      word address; it is sent to the address of the first store when the
      last word is written, by a PREF Nudge (25) of its block, by SYNC, by
      a load of its block (address bits 31:5), by an uncached accelerated
      store to another block and by any exception.

    An open entry is not ordered with the stores that leave at once: an
    uncached store to the block of an open write-through entry reaches
    memory first and is then overwritten by the entry (real 24K
    behaviour, software orders them with SYNC).
    """

    def __init__(self, caches: "Caches") -> None:
        self.c = caches
        self.wt: _Entry | None = None
        self.uca: _Entry | None = None
        self.merges = self.gathers = self.flushes = self.immediate = 0
        #: the last flushes, newest last: (kind, base, bytes) for the views
        self.log: list[tuple[str, int, int]] = []

    def reset_counters(self) -> None:
        self.merges = self.gathers = self.flushes = self.immediate = 0
        self.log.clear()

    @property
    def entries(self) -> list[_Entry]:
        return [e for e in (self.wt, self.uca) if e is not None]

    def merging(self) -> bool:
        return bool(self.c.m.cp0_read(16, 0) & CONFIG_MM)

    def _note(self, kind: str, base: int, mask: int) -> None:
        self.flushes += 1
        self.log.append((kind, base, bin(mask).count("1")))
        del self.log[:-WBB_ENTRIES * 4]

    # -- the core side -------------------------------------------------------

    def evict(self, pa: int, data: bytes, known: int) -> None:
        """A D-cache eviction: complete at once, flushed immediately."""
        self.immediate += 1
        self._note("evict", pa, known)
        t = self.c.timing
        t.reserve("eviction")
        t.issue(len(data))
        self.c._send(pa, data, known, through=False)

    def write_through(self, pa: int, data: bytes) -> None:
        """A write-through store (24K 8.4: merged while Config.MM = 1)."""
        t = self.c.timing
        if not self.merging():
            self.immediate += 1
            self._note("WT", pa, (1 << len(data)) - 1)
            t.reserve("write-through store")
            t.issue(len(data))
            self.c._send(pa, data, (1 << len(data)) - 1, through=True)
            return
        i = 0
        while i < len(data):
            a = (pa + i) & self.c.pamask
            base, off = a & ~(WBB_LINE - 1), a & (WBB_LINE - 1)
            k = min(len(data) - i, WBB_LINE - off)
            if self.wt is not None and self.wt.base != base:
                self.flush_wt()                   # a WT store to another block
            if self.wt is None:
                t.reserve("write-through store")
                self.wt = _Entry("WT", base)
            else:
                self.merges += 1
            self.wt.put(off, data[i:i + k])
            if self.wt.mask == (1 << WBB_LINE) - 1:
                self.flush_wt()                   # the block is complete
            i += k

    def uncached_store(self, pa: int, data: bytes, cca: int, gather: bool) -> bool:
        """An uncached (accelerated) store; True when the buffer took it
        for gathering, False when it leaves at once (the caller does it)."""
        size = len(data)
        t = self.c.timing
        if cca != CCA_UCA or not gather or size not in (4, 8) or pa % size:
            return self._at_once(size)            # leaves at once
        base, off = pa & ~(WBB_LINE - 1), pa & (WBB_LINE - 1)
        e = self.uca
        if e is not None and e.base != base:
            self.flush_uca()                      # a UCA store to another block
            e = None
        if e is None:
            if off:                               # not at the start of a block
                return self._at_once(size)
            t.reserve("uncached accelerated store")
            e = self.uca = _Entry("UCA", base)
        elif size == 8 and e.ptr % 8:
            # a doubleword into a word position (24K 8.4.1: not allowed):
            # the gathered block goes out, this store leaves on its own
            self.flush_uca()
            return self._at_once(size)
        else:
            self.gathers += 1
        e.put(e.ptr, data)                        # sequentially, whatever the address
        e.ptr += size
        if e.ptr >= WBB_LINE:
            self.flush_uca()                      # the last word is written
        return True

    def _at_once(self, size: int) -> bool:
        """An uncached store that leaves at once (the caller writes it)."""
        self.immediate += 1
        t = self.c.timing
        t.reserve("uncached store")
        t.issue(size)
        return False

    def read_request(self, pa: int, n: int) -> None:
        """A read is about to leave the core: an open entry of its block
        goes out first (write-through: "the write command will be ordered
        ahead of the read command"; UCA: a load matching bits 31:5)."""
        lo, hi = pa & self.c.pamask, (pa + n - 1) & self.c.pamask
        for e in (self.wt, self.uca):
            if e is not None and e.base <= hi and lo < e.base + WBB_LINE:
                if e is self.wt:
                    self.flush_wt()
                else:
                    self.flush_uca()
                self.c.timing.ordered_read()

    def nudge(self, pa: int) -> None:
        """PREF 25 (Nudge) of the gathered block: explicit flush."""
        if self.uca is not None and self.uca.base == pa & ~(WBB_LINE - 1):
            self.flush_uca()

    def sync(self) -> None:
        """SYNC: every barrier type stops merging on all entries (24K
        SYNC: "the BIU will stop merging on all WBB entries")."""
        self.flush_wt()
        self.flush_uca()

    def exception(self) -> None:
        """An exception flushes the gathered UCA block (24K 8.4.1)."""
        self.flush_uca()

    # -- flushes -------------------------------------------------------------

    def flush_wt(self) -> None:
        e, self.wt = self.wt, None
        if e is not None:
            self._note("WT", e.base, e.mask)
            self.c.timing.issue(bin(e.mask).count("1"))
            self.c._send(e.base, bytes(e.data), e.mask, through=True)

    def flush_uca(self) -> None:
        e, self.uca = self.uca, None
        if e is not None:
            self._note("UCA", e.base, e.mask)
            self.c.timing.issue(bin(e.mask).count("1"))
            self.c.memory.bus_write_line(e.base, bytes(e.data), e.mask)

    # -- views ---------------------------------------------------------------

    def byte(self, pa: int, kind: str) -> int | None:
        e = self.wt if kind == "WT" else self.uca
        return None if e is None else e.byte(pa)

    def backdoor_write(self, pa: int, value: int) -> None:
        for e in (self.wt, self.uca):
            if e is not None and e.byte(pa) is not None:
                e.data[(pa & self.c.pamask) - e.base] = value & 0xFF


def _dword_byte(k: int, big_endian: bool) -> int:
    """Offset in a doubleword of its k-th least-significant byte."""
    return 7 - k if big_endian else k


def _pairs(ways: int) -> list[tuple[int, int]]:
    """Way pairs of the LRU bits, bit 0 first.  For 4 ways the 24K's
    order (Table 8.4: bit 5 = ways 0/2, 4 = 1/3, 3 = 0/3, 2 = 0/1,
    1 = 1/2, 0 = 2/3; a 1 means the lower way is the more recent)."""
    if ways == 4:
        return [(2, 3), (1, 2), (0, 1), (0, 3), (1, 3), (0, 2)]
    return [(i, j) for i in range(ways) for j in range(i + 1, ways)]


#: CP0 (register, select) of each cache's tag / data registers.
_TAGLO = {"I": (28, 0), "D": (28, 2), "S": (28, 4), "T": (28, 4)}

#: SYNC stypes (24K Table 12.1): the ordering barriers (the others are
#: completion barriers; reserved values act as 0, Vol. II-A SYNC)
SYNC_WMB, SYNC_MB, SYNC_ACQUIRE, SYNC_RELEASE, SYNC_RMB = 0x4, 0x10, 0x11, 0x12, 0x13
ORDERING_SYNC = frozenset({SYNC_WMB, SYNC_MB, SYNC_ACQUIRE, SYNC_RELEASE, SYNC_RMB})
#: ... after which no younger store may be performed before an older one
STORE_ORDERING_SYNC = frozenset({SYNC_WMB, SYNC_MB, SYNC_RELEASE})


@dataclass
class PendingStore:
    """A coherent store in the fill-store buffer, waiting for its line
    (ownership from the Coherence Manager): ``data`` maps byte offsets in
    the line to bytes (later stores to the line merge in)."""
    va: int
    line: int                 # physical address of the line
    cca: int
    ready: int                # core cycle at which the line arrives
    why: str                  # "miss" / "upgrade"
    data: dict[int, int] = field(default_factory=dict)


class Caches:
    """The cache hierarchy of a core and its CP0 interface (CACHE, SYNC,
    ErrCtl, TagLo / DataLo / DataHi): L1 instruction and data caches,
    optional L2 / L3 (``l2``, ``l3``: :class:`Geometry` of level 2 / 3,
    ``ecc``) and the write buffer (``write_buffer``).  Bound to a
    :class:`~emips.sim.Machine` by the machine itself."""

    def __init__(self, icache: Geometry | None = GEOMETRY_24K,
                 dcache: Geometry | None = GEOMETRY_24K, *, parity: bool = False,
                 hardware_init: bool = False, seed: int = 0x24C,
                 l2: Geometry | None = None, l3: Geometry | None = None,
                 ecc: bool = False, write_buffer: bool = False) -> None:
        if l3 is not None and l2 is None:
            raise ValueError("an L3 cache needs an L2 cache")
        self.i = Cache("I", icache, parity=parity) if icache is not None else None
        self.d = Cache("D", dcache, parity=parity) if dcache is not None else None
        self.l2 = self.l3 = None
        if l2 is not None:
            self.l2 = SecondaryCache("S", Geometry(l2.sets, l2.ways, l2.line, 2), ecc=ecc)
        if l3 is not None:
            self.l3 = SecondaryCache("T", Geometry(l3.sets, l3.ways, l3.line, 3), ecc=ecc)
        self.parity = parity
        self.ecc = ecc and self.l2 is not None
        self.hardware_init = hardware_init
        self.seed = seed
        self.m = None
        #: the machine's physical address space (set_pamask)
        self.pamask = M32
        self.memory = Memory(self)
        self.wbb = WriteBuffer(self) if write_buffer else None
        #: memory timing (emips.timing), switched on by the pipeline
        self.timing = Timing(self)
        #: an L2 / L3 error waiting to be taken (imprecise Cache Error)
        self.pending: CacheError | None = None
        #: Config2.L2B: the L2 is bypassed
        self.l2_bypass = False
        #: the P8700's L1 data prefetcher (emips.p8700mem) and what asks now
        self.l1pf = None
        self.req_kind = "load"
        #: the D-cache corrects virtual aliases (:meth:`_unalias`; the P8700)
        self.alias_free = False
        #: while watched: every access (CacheAccess), in the order made
        self.log: list[CacheAccess] | None = None
        #: a core of a CPS: the Coherence Manager (emips.coherence), which
        #: also owns the shared L2 / L3 and memory port; this core's number
        self.cm = None
        self.core = 0
        #: the attributes that go through the caches (a CPS core: + 4, 5)
        self.cached_cca = CACHED_CCA
        #: the coherent request a refill makes ("share" / "own" /
        #: "discard"), set around a lookup; what the CM answered
        self._intent: str | None = None
        #: the store being made is an SC's (emips.coherence error info)
        self._atomic = False
        self._fill: tuple[bool, bool, str] | None = None
        #: coherent stores waiting in the fill-store buffer for their line
        #: (a CPS core, pipeline with non-blocking caches): not globally
        #: performed yet — the other cores do not see them (M11 phase 3)
        self.fsb: list[PendingStore] = []
        for c in self.secondary:
            c.timing = self.timing
        if self.l3 is not None:
            self.l3.below = self.memory
            self.l3.sink = self.record
        if self.l2 is not None:
            self.l2.below = self.l3 if self.l3 is not None else self.memory
            self.l2.sink = self.record
        rng = random.Random(seed)
        for c in self.all:
            if not hardware_init:
                c.randomize(rng)
            else:
                c.invalidate_all()

    def set_pamask(self, mask: int) -> None:
        """The machine's physical address width (LF13): every level's tags
        and the lines below cover it."""
        self.pamask = mask
        for c in (self.i, self.d, self.l2, self.l3):
            if c is not None:
                c.pamask = mask

    @property
    def all(self) -> list[Cache]:
        return [c for c in (self.i, self.d, self.l2, self.l3) if c is not None]

    @property
    def secondary(self) -> list[SecondaryCache]:
        return [c for c in (self.l2, self.l3) if c is not None]

    @property
    def big_endian(self) -> bool:
        return self.m.big_endian

    def describe(self) -> str:
        part = [f"{c.name} {c.geometry.describe()}" for c in self.all]
        text = "; ".join(part) or "no caches"
        text += (", parity" if self.parity else "") + (", ECC" if self.ecc else "")
        if self.wbb is not None:
            text += f"; write buffer {WBB_ENTRIES} × {WBB_LINE} bytes"
        return text

    @property
    def synci_step(self) -> int:
        """RDHWR 1: the smallest line size of the caches SYNCI must
        synchronize; 0 without an instruction cache."""
        if self.i is None:
            return 0
        return min(c.line for c in (self.i, self.d) if c is not None)

    @property
    def cacheable(self) -> bool:
        """Do cacheable accesses go through some cache?"""
        return self.d is not None or self.l2 is not None

    # -- coherence (a CPS core, emips.coherence) ---------------------------------

    def coherent(self, cca: int) -> bool:
        """Is an access with attribute ``cca`` coherent (a coherent CCA, and
        this core in coherent mode: GCR_CL_COHERENCE has its own bit)?"""
        return self.cm is not None and cca in COHERENT_CCA and self.cm.coherent_mode(self.core)

    def _lookup(self, c: Cache, va: int, pa: int, kind: str, cca: int, intent: str,
                **kw) -> tuple[int, int | None]:
        """``c.lookup``; a coherent refill asks the Coherence Manager with
        ``intent`` and takes the MESI state it answers."""
        if self.alias_free and c is self.d:
            self._unalias(c, va, pa)
        if self.cm is None or not self.coherent(cca):
            return c.lookup(va, pa, self, kind=kind, **kw)
        self._intent, self._fill = intent, None
        log = self.log
        mark = len(log) if log is not None else 0
        try:
            s, w = c.lookup(va, pa, self, kind=kind, **kw)
        finally:
            self._intent = None
        self._apply_fill(c, s, w, log, mark)
        return s, w

    def _unalias(self, c: Cache, va: int, pa: int) -> None:
        """A D-cache that corrects virtual aliases (the P8700's, MD01502
        3.3.2): a line the access would miss in its set but that sits in
        another of the PA's sets (another virtual colour) is written back
        if dirty and invalidated there — the refill then brings it to the
        access's set, so a physical line is never in two sets."""
        tag = c.tag_of(pa)
        s = c.set_of(va)
        if c.find(s, tag) is not None:
            return
        for a in c.alias_sets(pa):
            if a == s:
                continue
            w = c.find(a, tag)
            if w is not None:
                if c._dirty(a, w):
                    c.write_back(a, w, self)
                self._invalidate(c, a, w)

    def _apply_fill(self, c: Cache, s: int, w: int | None, log, mark: int) -> None:
        fill, self._fill = self._fill, None
        if fill is None or w is None:
            return
        shared, dirty, note = fill
        ln = c.lines[s][w]
        if c.writable:
            ln.shared, ln.dirty = shared, dirty
            c._seal(ln)
        if log is not None and note:
            for rec in log[mark:]:
                if rec.level == c.kind and not rec.hit:
                    rec.coherence = note
                    break

    # -- observation (views, trace) ------------------------------------------

    def watch(self) -> None:
        """Start recording every access (see :meth:`taken`)."""
        self.log = log = []
        for c in (self.i, self.d, self.l2, self.l3):     # (self.all, without the list)
            if c is not None:
                c.log = log

    def taken(self) -> list[CacheAccess]:
        """Stop recording; the accesses recorded since :meth:`watch`."""
        log, self.log = self.log or [], None
        for c in (self.i, self.d, self.l2, self.l3):
            if c is not None:
                c.log = None
        return log

    def reset_stats(self) -> None:
        """Start the statistics again (the contents stay)."""
        for c in self.all:
            c.hits = c.misses = c.fills = c.writebacks = c.evictions = 0
            c.stats = CacheStats(c.sets * c.ways)
        if self.wbb is not None:
            self.wbb.reset_counters()

    def _uncached(self, kind: str, pa: int, no_cache: bool = False) -> None:
        if self.log is not None:
            self.log.append(CacheAccess("U", kind, pa & self.pamask, pa & self.pamask,
                                        miss="no cache" if no_cache else None))

    def initialize(self) -> None:
        """What boot firmware does before using kseg0 cached."""
        for c in self.all:
            c.invalidate_all()

    def set_checking(self, errctl: int) -> None:
        on = bool(errctl & EC_PE) and self.parity
        for c in (self.i, self.d):
            if c is not None:
                c.check = on
        l2_on = self.ecc and bool(errctl & EC_PE) != bool(errctl & EC_L2P)
        for c in self.secondary:
            c.check = l2_on

    def set_config2(self, value: int) -> None:
        self.l2_bypass = self.l2 is not None and bool(value & CONFIG2_L2B)

    # -- imprecise errors -----------------------------------------------------

    def record(self, err: CacheError) -> None:
        """An L2 / L3 error: kept until the exception is taken; a second
        one before that sets EF (24K Table 6.51)."""
        if self.pending is None:
            self.pending = err
        else:
            self.pending.fatal = True

    def take_pending(self) -> CacheError | None:
        err, self.pending = self.pending, None
        return err

    # -- the bus interface unit -------------------------------------------------

    def _first(self):
        """The level below the L1 caches for cacheable traffic."""
        if self.l2 is not None and not self.l2_bypass:
            return self.l2
        return self.l3 if self.l3 is not None else self.memory

    def bus_read_line(self, pa: int, n: int) -> tuple[bytearray, int]:
        """An L1 refill (or a cacheable read without an L1): the write
        buffer's matching entries go first, then the L2 / L3 / memory; a
        coherent refill goes through the Coherence Manager (interventions
        in the other cores' data caches)."""
        if self._intent is not None:
            intent, self._intent = self._intent, None        # one refill per lookup
            return self.cm.coherent_read(self, pa, n, intent)
        return self.read_below(pa, n)

    def read_below(self, pa: int, n: int, source: str | None = None
                   ) -> tuple[bytearray, int]:
        """A line from the levels below the L1s (``source``: the timing
        level that served it when not them: "C2C", another core's cache)."""
        if self.wbb is not None:
            self.wbb.read_request(pa, n)
        out = self._first().bus_read_line(pa, n) if source is None else None
        t = self.timing
        if t.active:
            if source is not None:
                t.served(source)
            t.refill(pa, n, "I" if t.kind == "fetch" else "D")
            t.level = 0
        return out

    def bus_write_line(self, pa: int, data: bytes, known: int) -> None:
        """A D-cache eviction (write-back)."""
        if self.wbb is not None:
            self.wbb.evict(pa, data, known)
        else:
            self.timing.direct_write(len(data), "victim write-back")
            self._send(pa, data, known, through=False)

    def _send(self, pa: int, data: bytes, known: int, *, through: bool) -> None:
        """A write leaving the write buffer for the L2 / L3 / memory."""
        _write_below(self._first(), pa, data, known, through)

    def read_request(self, pa: int, n: int, kind: str = "load", *,
                     no_cache: bool = False) -> None:
        """An uncached read (``kind`` load or fetch) is about to leave the
        core (write buffer ordering; uncached reads bypass the L2 / L3).
        ``no_cache``: the attribute is cacheable, but no cache serves it."""
        t = self.timing
        t.begin(kind)
        self._uncached(kind, pa, no_cache)
        if self.wbb is not None:
            self.wbb.read_request(pa, n)
        t.uncached_read(n)

    def uncached_store(self, pa: int, data: bytes, cca: int, gather: bool) -> bool:
        """True when the write buffer gathered the store (UCA)."""
        self.timing.begin("store")
        self._uncached("store", pa, cca in CACHED_CCA)
        if self.wbb is None:
            self.timing.direct_write(len(data), "uncached store", "uncached")
            return False
        return self.wbb.uncached_store(pa, data, cca, gather)

    def sync(self, stype: int = 0) -> None:
        """SYNC (24K Table 12.1 and its text): every type stops merging in
        the write buffer; a completion barrier (0, the CM's 2 / 3, any
        reserved stype) performs every waiting store and waits until all
        writes and fills are done; an ordering barrier waits for nothing
        ("the core will not wait for pending transactions"), but no younger
        store may be performed before an older one after SYNC_WMB / MB /
        RELEASE: the waiting stores are performed first (loads are
        processed in order: ACQUIRE / RMB need nothing more)."""
        if self.cm is not None:
            self.cm.claim(self)
        self.timing.begin("sync")
        if self.wbb is not None:
            self.wbb.sync()
        if stype in ORDERING_SYNC:
            if stype in STORE_ORDERING_SYNC:
                self.fsb_drain()
            return
        self.fsb_drain()
        # stype 0 on a CM2 with GCR_CONTROL.SYNCCTL = 0: an intervention sync
        self.timing.sync(memory=not (stype == 0 and self.cm is not None
                                     and not self.cm.memory_sync0))

    def exception_entry(self) -> None:
        if self.wbb is not None:
            self.wbb.exception()

    def nudge(self, pa: int) -> None:
        if self.wbb is not None:
            self.wbb.nudge(pa)

    # -- CPU accesses -----------------------------------------------------

    def read(self, va: int, pas: list[int], size: int, cca: int = CCA_WB) -> bytes:
        """Load bytes through the data cache (cacheable attribute ``cca``;
        coherent: a miss asks for the line Shared, 5, or Exclusive, 4)."""
        self.req_kind = "load"
        if self.cm is not None:
            self.cm.claim(self)
        d = self.d
        t = self.timing
        if t.active:
            t.begin("load")
        if d is None:                                    # straight to the L2
            coherent = self.coherent(cca)

            def line(pa: int, n: int) -> bytearray:
                if coherent:                             # the newest copy, not kept
                    self._intent = "discard"
                data = self.bus_read_line(pa, n)[0]
                self._intent = self._fill = None
                return data
            if pas == list(range(pas[0], pas[0] + size)):
                return bytes(line(pas[0], size))
            return bytes(line(pa, 1)[0] for pa in pas)
        off = va & d.line_mask
        if self.fsb and off + size <= d.line:
            fwd = self._fsb_forward(pas[0], size)
            if fwd is not None:
                return fwd
        if off + size <= d.line:
            # CCA 4 asks for the line Exclusive — but an LL always asks Shared
            # (P5600 SUM: "so that an LL itself does not clear the LL bit in
            # another CPU")
            linked = self.m is not None and getattr(self.m, "_linked", False)
            before = d.misses
            s, w = self._lookup(d, va, pas[0], "load", cca,
                                "own" if cca == CCA_CWBE and not linked else "share")
            if t.active:
                t.line_access(pas[0] & ~d.line_mask)
            if d.check and d.parity:
                d._check_data(s, w, off, size)
            out = bytes(d.lines[s][w].data[off:off + size])
            if self.l1pf is not None:                    # (the P8700's L1 prefetcher)
                self.l1pf.observe(self, va, pas[0], cca, d.misses != before)
            return out
        return bytes(self.read((va + i) & M32, [pas[i]], 1, cca)[0] for i in range(size))

    def write(self, va: int, pas: list[int], data: bytes, cca: int,
              atomic: bool = False) -> None:
        """Store bytes: write-back with write allocate (3; coherent 4 / 5:
        a miss asks for the line Exclusive, a hit on a Shared line first
        invalidates the other copies) or write-through without write
        allocate (0), 24K 8.5.2."""
        #: an SC's store (the CM's intervention-error info, bit 8)
        self._atomic = atomic
        self.req_kind = "store"
        if self.cm is not None:
            self.cm.claim(self)
        d = self.d
        size = len(data)
        t = self.timing
        if t.active:
            t.begin("sc" if atomic else "store")
        if self.cm is not None and d is not None:
            if atomic and self.fsb:
                self._fsb_drain_line(pas[0])
            elif (not atomic and t.active and t.nonblocking and self.coherent(cca)
                  and (va & d.line_mask) + size <= d.line
                  and self._fsb_store(va, pas[0], data, cca)):
                return
        if d is None:
            if self.coherent(cca):                       # no copy may stay stale
                for pa in sorted({pa & ~(WBB_LINE - 1) for pa in pas}):
                    self.cm.claim_line(self, pa, write_back=True)
            for i, pa in enumerate(pas):
                if cca == CCA_WT and self.wbb is not None:
                    self.wbb.write_through(pa, data[i:i + 1])
                elif self.wbb is not None:
                    t.reserve("store")
                    t.issue(1)
                    self._send(pa, data[i:i + 1], 1, through=False)
                else:
                    t.direct_write(1, "store")
                    self._send(pa, data[i:i + 1], 1, through=cca == CCA_WT)
            return
        off = va & d.line_mask
        if off + size > d.line:
            for i in range(size):
                self.write((va + i) & M32, [pas[i]], data[i:i + 1], cca)
            return
        before = d.misses
        s, w = self._lookup(d, va, pas[0], "store", cca, "own",
                            allocate=cca in WRITE_BACK_CCA)
        if t.active:
            t.line_access(pas[0] & ~d.line_mask)
        if w is not None:
            if d.check and d.parity and d.WORD and (off % d.WORD or size % d.WORD):
                d._check_data(s, w, off, size)           # (a word in part: read first)
            ln = d.lines[s][w]
            if ln.shared:                                # S -> M: an upgrade
                if self.coherent(cca):
                    self.cm.upgrade(self, va, pas[0], s, w)
                ln.shared = False
            ln.data[off:off + size] = data
            ln.known |= ((1 << size) - 1) << off
            d._set_data_parity(ln, off, size)
            if cca in WRITE_BACK_CCA:
                ln.dirty = True
                d._seal(ln)
            if self.l1pf is not None:                    # (the P8700's L1 prefetcher)
                self.l1pf.observe(self, va, pas[0], cca, d.misses != before)
        if cca == CCA_WT:
            if self.wbb is not None:
                self.wbb.write_through(pas[0], bytes(data))
            else:
                t.direct_write(size, "write-through store")
                self._send(pas[0], bytes(data), (1 << size) - 1, through=True)

    def fetch(self, va: int, pa: int, cca: int = CCA_WB) -> tuple[bytes, int]:
        """Instruction fetch through the I-cache: ``(4 bytes, known
        mask)``; a coherent refill reads the newest copy without taking it
        (Coherent Read Discard).  An L2 / L3 error found by the refill is
        taken at once (the instruction cannot run on bad bits)."""
        self.req_kind = "fetch"
        if self.cm is not None:
            self.cm.claim(self)
        c = self.i
        before = self.pending
        if self.timing.active:
            self.timing.begin("fetch")
        # an instruction refill is a non-coherent read (the CM's Legacy RD,
        # "used for all non-coherent reads, including code fetches"): it
        # reads the L2 / memory, never another core's data cache
        if c is None:
            data, known = self.bus_read_line(pa, 4)
            self._intent = self._fill = None
        else:
            s, w = self._lookup(c, va, pa, "fetch", cca, None)
            off = va & c.line_mask
            if c.check and c.parity:
                c._check_data(s, w, off, 4)
                d = off // 8
                ln = c.lines[s][w]
                if d < len(ln.pre) and ln.pre_p[d] != _parity(ln.pre[d]):
                    raise CacheError(c, data=True, way=w,
                                     index=(s << c.line_shift) | (off & ~7),
                                     what=f"precode parity error, set {s} way {w}")
            ln = c.lines[s][w]
            data, known = bytes(ln.data[off:off + 4]), (ln.known >> off) & 0xF
        if before is None and self.pending is not None:
            raise self.take_pending()
        return bytes(data), known

    # -- the fill-store buffer: coherent stores in flight (M11 phase 3) ----------

    def _fsb_entry(self, pa: int) -> PendingStore | None:
        line = pa & ~(self.d.line_mask) & self.pamask
        return next((e for e in self.fsb if e.line == line), None)

    def _fsb_store(self, va: int, pa: int, data: bytes, cca: int) -> bool:
        """A coherent store that needs the Coherence Manager (a miss, or a
        Shared line: an upgrade) waits in the fill-store buffer until its
        line arrives; younger loads and stores that hit go on.  False:
        the line is owned (E / M) — the store is performed now."""
        d, t = self.d, self.timing
        e = self._fsb_entry(pa)
        if e is None:
            ln = d.probe(va, pa)
            if ln is not None and not ln.shared:
                return False
            line = pa & ~d.line_mask & self.pamask
            why = "upgrade" if ln is not None else "miss"
            t._fsb()                                       # a free FSB line
            while len(self.fsb) >= 8:                      # the FSB: 8 lines
                self._fsb_perform(self.fsb[0])
            ready = t.cursor() + self.cm.request_cycles(self, va, pa, why)
            if self.fsb:                               # the CM takes them in order
                ready = max(ready, self.fsb[-1].ready)
            e = PendingStore(va & ~d.line_mask & M32, line, cca, ready, why)
            self.fsb.append(e)
            arrives = max(ready, t.cursor() + self.cm.fill_cycles(self, va, pa, why))
            t.pending[line] = max(t.pending.get(line, 0), arrives)
            note = (f"waits in the fill-store buffer ({why}): the CM orders it "
                    f"in {ready - t.now} cycles; until then the other cores see the "
                    "old value")
        else:
            note = "merged into the fill-store buffer entry of its line"
        off = pa & d.line_mask
        for i, byte in enumerate(data):
            e.data[off + i] = byte
        if self.log is not None:
            self.log.append(CacheAccess("D", "store", va & M32, pa & self.pamask, d.set_of(va), None,
                                        d.tag_of(pa), False, offset=off, coherence=note))
        return True

    def tick(self) -> None:
        """Every cycle: the stores whose line has arrived are performed."""
        if self.fsb:
            now = self.timing.now
            for e in [e for e in self.fsb if e.ready <= now]:
                self._fsb_perform(e)

    def _fsb_perform(self, e: PendingStore) -> None:
        """The store is globally performed: the CM request (interventions)
        and the bytes written into the line — no extra cycles (they were
        counted when it entered the buffer), nothing recorded."""
        self.fsb.remove(e)
        t, saved_log = self.timing, self.log
        active, kind = t.active, t.kind
        t.active = False
        if saved_log is not None:
            self.taken()
        try:
            d = self.d
            s, w = self._lookup(d, e.va, e.line, "store", e.cca, "own")
            ln = d.lines[s][w]
            if ln.shared:
                self.cm.upgrade(self, e.va, e.line, s, w)
                ln.shared = False
            for off, byte in e.data.items():
                ln.data[off] = byte
                ln.known |= 1 << off
                d._set_data_parity(ln, off, 1)
            ln.dirty = True
            d._seal(ln)
        finally:
            t.active, t.kind = active, kind
            if saved_log is not None:
                self.watch()
                self.log[:0] = saved_log

    def _fsb_drain_line(self, pa: int) -> None:
        e = self._fsb_entry(pa)
        if e is not None:
            self.timing.line_access(e.line)                # wait for the line
            self._fsb_perform(e)

    def fsb_drain(self) -> None:
        """Perform every waiting store, oldest first (a barrier)."""
        while self.fsb:
            self._fsb_perform(self.fsb[0])

    def _fsb_forward(self, pa: int, size: int) -> bytes | None:
        """A load of bytes all held by a waiting store: forwarded from the
        fill-store buffer; one that needs the rest of the line waits for it
        (the store is performed first)."""
        e = self._fsb_entry(pa)
        if e is None:
            return None
        off = pa & self.d.line_mask
        if all(off + i in e.data for i in range(size)):
            return bytes(e.data[off + i] for i in range(size))
        self._fsb_drain_line(pa)
        return None

    # -- views (no side effects) ------------------------------------------------

    def peek(self, cache: Cache | None, va: int, pa: int) -> int | None:
        """The cached byte a CPU access would see, no side effects
        (``None``: not cached)."""
        if cache is None:
            return None
        ln = cache.probe(va, pa)
        return None if ln is None else ln.data[va & cache.line_mask]

    def view_below(self, pa: int, cached: bool) -> int | None:
        """The byte a read request leaving the L1 caches now would return
        (``None``: never written): the write buffer's open entries go out
        first; a cacheable read is then served by the L2 / L3 (the UCA
        block went to memory only), an uncached one by memory."""
        wbb = self.wbb
        wt = wbb.byte(pa, "WT") if wbb is not None else None
        uca = wbb.byte(pa, "UCA") if wbb is not None else None
        if cached:
            if wt is not None:
                return wt
            levels = self.secondary if not self.l2_bypass else [c for c in (self.l3,) if c]
            for c in levels:
                hit = c.peek(pa)
                if hit is not None:
                    return hit[0] if hit[1] else None
            if uca is not None:
                return uca
            return self.memory.peek(pa)
        if uca is not None:
            return uca
        if wt is not None:
            return wt
        return self.memory.peek(pa)

    def view_data(self, va: int, pa: int, cached: bool,
                  coherent: bool = False) -> int | None:
        """The byte a load would return now (``None``: never written); a
        coherent load would get another core's Modified copy."""
        if cached:
            if self.fsb:
                e = self._fsb_entry(pa)
                if e is not None and (pa & self.d.line_mask) in e.data:
                    return e.data[pa & self.d.line_mask]
            byte = self.peek(self.d, va, pa)
            if byte is not None:
                return byte
            if coherent and self.cm is not None:
                byte = self.cm.peek_owned(self, pa)
                if byte is not None:
                    return byte
        return self.view_below(pa, cached)

    def backdoor_write(self, pa: int, value: int) -> None:
        """A debugger / loader write: every cached copy of ``pa`` (in any
        alias set, any level, the write buffer; in a CPS every core's)
        takes the new byte; nothing else changes."""
        if self.cm is not None:
            self.cm.backdoor_write(pa, value)
            return
        self.own_backdoor_write(pa, value, self.all)

    def own_backdoor_write(self, pa: int, value: int, caches: list, *,
                           wbb: bool = True) -> None:
        for c in caches:
            tag, off = c.tag_of(pa), pa & c.line_mask
            for s in c.alias_sets(pa):
                w = c.find(s, tag)
                if w is not None:
                    ln = c.lines[s][w]
                    ln.data[off] = value & 0xFF
                    ln.known |= 1 << off
                    c._set_data_parity(ln, off, 1)
                    if c.kind == "I":
                        c._precode(ln, off & ~7, (off & ~7) + 8, self)
        if wbb and self.wbb is not None:
            self.wbb.backdoor_write(pa, value)
        if wbb and self.fsb:
            e = self._fsb_entry(pa)
            if e is not None and (pa & self.d.line_mask) in e.data:
                e.data[pa & self.d.line_mask] = value & 0xFF

    # -- SYNCI, PREF -------------------------------------------------------

    def synci(self, va: int, pa: int, cca: int = CCA_WB) -> None:
        """SYNCI (24K: D-cache writeback and I-cache invalidate of the
        line); locked lines are left alone (Vol. II-A SYNCI).  Coherent:
        the Coherence Manager does it in every core of the domain (it
        globalizes address-based cache operations)."""
        if self.cm is not None:
            self.cm.claim(self)
            if self.coherent(cca):
                self.cm.global_op(self, "D", 6, pa)
                self.cm.global_op(self, "I", 4, pa)
        t = self.timing
        t.begin("cacheop")
        if self.d is not None:
            s, w = self.d.lookup(va, pa, self, allocate=False, kind="synci")
            if w is not None and not self.d.lines[s][w].lock and self.d._dirty(s, w):
                t.fixed(CACHEOP_COST["hit-wb-dirty"], "synci: data line written back")
                self.d.write_back(s, w, self)
            elif w is not None:
                t.fixed(CACHEOP_COST["hit-wb-clean"], "synci: data line clean")
        if self.i is not None:
            s, w = self.i.lookup(va, pa, self, allocate=False, kind="synci")
            if w is not None and not self.i.lines[s][w].lock:
                t.fixed(CACHEOP_COST["hit-inv"], "synci: instruction line invalidated")
                self._invalidate(self.i, s, w)

    def fence_i(self) -> None:
        """RISC-V FENCE.I (Zifencei 6.1; PLAN M25): this hart's later
        fetches see its earlier stores — every dirty D-cache line written
        back, every I-cache line invalidated, the write buffer drained (no
        24K cost table applies: no cycles are charged)."""
        if self.cm is not None:
            self.cm.claim(self)
        d = self.d
        if d is not None:
            for s in range(d.sets):
                for w in range(d.ways):
                    if d.lines[s][w].valid and d._dirty(s, w):
                        d.write_back(s, w, self)
        icaches = [self.i]
        if self.cm is not None and getattr(self.cm, "fence_i_global", False):
            # (the P8700, MD01502 3.3.10: "operates on all instruction caches in
            # a cluster")
            icaches = [c.i for c in self.cm.cores.values()]
        for ic in icaches:
            if ic is None:
                continue
            for s in range(ic.sets):
                for w in range(ic.ways):
                    if ic.lines[s][w].valid:
                        self._invalidate(ic, s, w)
        self.sync(0)

    def prefetch(self, hint: int, va: int, pa: int, cca: int) -> None:
        """PREF (24K Table 12.1): 0/1/6/7 fill the line (6/7 never into
        way 0), 4/5 fill way 0 (dropped when it is locked), 25 writes a
        dirty unlocked line back and invalidates it, 30 PrepareForStore
        allocates a zeroed dirty line on a miss; reserved hints and lines
        already present do nothing (a load hit only updates the LRU)."""
        d = self.d
        if d is None:
            return
        self.req_kind = "pref"
        if self.cm is not None:
            self.cm.claim(self)
        self.timing.begin("pref")
        s, tag = d.set_of(va), d.tag_of(pa)
        w = d.find(s, tag)
        if hint in (0, 1, 4, 5, 6, 7):
            if w is not None:
                if hint not in (4, 5):
                    d.touch(s, w)
                d.note_hit("pref", va, pa, s, w)
                return
            self._lookup(d, va, pa, "pref", cca,
                         "own" if hint in (1, 5, 7) or cca == CCA_CWBE else "share",
                         way0=hint in (4, 5), not_way0=hint in (6, 7))
        elif hint == 25:
            if w is not None and not d.lines[s][w].lock and d._dirty(s, w):
                d.write_back(s, w, self)
                self._invalidate(d, s, w)
        elif hint == 30 and cca in WRITE_BACK_CCA and w is None:
            if d.check and d.parity:
                d._check_tags(s)
            if self.coherent(cca):                       # the line becomes ours
                self.cm.claim_line(self, pa)
            d.replace(s, d.victim(s), tag, self, zero=True)

    # -- RISC-V's cache-block operations (unpriv. ch. 19, LF8) ----------------------

    def cbo(self, op: str, va: int, pa: int, size: int, cca: int) -> str:
        """CBO.CLEAN / CBO.FLUSH / CBO.INVAL (19.6.1) on the ``size``-byte
        block at ``va`` / ``pa``: each D-cache, L2 and L3 line the block
        covers — clean writes a dirty line back, flush writes back and
        invalidates, inval invalidates (a dirty line's data is lost) — and
        with a Coherence Manager every core's D-cache (the coherent caches:
        the instruction caches are not coherent in eMIPS, FENCE.I's job);
        clean / flush drain the write buffer: the block reaches memory, "a
        point common to all agents".  Hit-type CACHE operations' costs
        (MD00355 Table 3.5).  Returns what was done."""
        code = {"clean": 6, "flush": 5, "inval": 4}[op]
        name = {"clean": "clean", "flush": "flush", "inval": "invalidate"}[op]
        if self.cm is not None:
            self.cm.claim(self)
        t = self.timing
        t.begin("cacheop")
        done = []
        for c in (self.d, self.l2, self.l3):
            if c is None:
                continue
            bus = self if c.level == 1 else c.below
            for off in range(0, max(size, c.line), c.line):
                a, lpa = (va if c.level == 1 else pa) + off, pa + off
                s_, tag = c.set_of(a), c.tag_of(lpa)
                w = c.find(s_, tag)
                if w is None:
                    continue
                dirty = c.writable and c._dirty(s_, w)
                if c.level == 1 and t.active:
                    t.fixed(CACHEOP_COST["hit-inv" if code == 4 else
                                         ("hit-wb-dirty" if dirty else "hit-wb-clean")
                                         if code == 6 else
                                         ("hit-wbi-dirty" if dirty else "hit-wbi-clean")],
                            f"cbo.{op}: {c.name} line")
                done.append(self._hit_op(c, code, a, lpa, f"cbo.{op}", bus, cca))
        if self.cm is not None and self.coherent(cca):
            for off in range(0, size, self.d.line if self.d is not None else size):
                more = self.cm.global_op(self, "D", code, pa + off)
                if more:
                    done.append(f"globalized: {more}")
        if op != "inval":
            self.sync(0)                        # (down to memory)
        return f"cbo.{op}: " + ("; ".join(done) if done else f"no cached copy to {name}")

    def cbo_zero(self, va: int, pa: int, size: int, cca: int) -> bool:
        """CBO.ZERO (19.6.2) in a write-back D-cache whose lines the block
        holds whole: each line allocated zeroed and dirty without reading
        memory (as PREF 30 PrepareForStore) or zeroed where it is.  False:
        not done here (no D-cache, not write-back, lines larger than the
        block) — the caller stores the zeros."""
        d = self.d
        if d is None or cca not in WRITE_BACK_CCA or d.line > size:
            return False
        if self.cm is not None:
            self.cm.claim(self)
        self.timing.begin("store")
        for off in range(0, size, d.line):
            lva, lpa = va + off, pa + off
            s_, tag = d.set_of(lva), d.tag_of(lpa)
            w = d.find(s_, tag)
            if w is None:
                if d.check and d.parity:
                    d._check_tags(s_)
                if self.coherent(cca):          # the line becomes ours
                    self.cm.claim_line(self, lpa)
                d.replace(s_, d.victim(s_), tag, self, zero=True)
            else:
                self.write(lva & M32, [lpa + k for k in range(d.line)], bytes(d.line), cca)
        return True

    @staticmethod
    def _invalidate(c: Cache, s: int, w: int) -> None:
        ln = c.lines[s][w]
        if ln.valid:
            c.stats.forget(c.line_pa(ln.tag, s))
        ln.valid = ln.lock = False
        ln.dirty = ln.shared = False
        c._seal(ln)
        c.make_lru(s, w)

    # -- CACHE ------------------------------------------------------------

    def cache_op(self, op: int, va: int, pa: int, cca: int) -> str:
        """CACHE op, va (see :meth:`_cache_op`), with its cost in cycles
        (MD00355 Table 3.5; an L2 / L3 operation goes through the write
        buffer)."""
        if self.cm is not None:
            self.cm.claim(self)
            if op & 3 in (2, 3) and op >> 2 >= 4 and self.cm.drop_l2(pa):
                # GCR_REGn_MASK.DROP_L2 (MD01025 Table 11.38): the L2
                # CacheOp write is dropped after it has been serialized
                return "L2 / L3 CACHE operation dropped (the region's DROP_L2)"
        t = self.timing
        if not t.active:
            return self._cache_op(op, va, pa, cca)
        t.begin("cacheop")
        target, code = op & 3, op >> 2
        c = {0: self.i, 1: self.d, 2: self.l3, 3: self.l2}.get(target)
        hit = dirty = False
        if c is not None:
            a = va if c.level == 1 else pa
            if code >= 4:
                w = c.find(c.set_of(a), c.tag_of(pa)) if cca in self.cached_cca else None
                hit = w is not None
                dirty = hit and c.writable and c.lines[c.set_of(a)][w].dirty
            else:
                way = (a >> c.index_bit) & ((1 << c.way_bits) - 1)
                if way < c.ways:
                    ln = c.lines[c.set_of(a)][way]
                    hit, dirty = ln.valid, ln.valid and c.writable and ln.dirty
        done = self._cache_op(op, va, pa, cca)
        if c is None or done.endswith("no operation"):
            return done
        if c.level > 1:                                   # sent through the write buffer
            if self.wbb is not None:
                t.reserve("L2 / L3 CACHE operation")
                t.issue(0)
            else:
                t.direct_write(0, "L2 / L3 CACHE operation")
            return done
        if code == 7:
            cost = CACHEOP_COST["lock-hit" if hit else "lock-miss"]
        elif code == 5 and c.kind == "I":
            cost = CACHEOP_COST["lock-miss"]
        elif code == 5:
            cost = 0 if not hit else CACHEOP_COST["hit-wbi-dirty" if dirty else "hit-wbi-clean"]
        elif code == 6:
            cost = 0 if not hit else CACHEOP_COST["hit-wb-dirty" if dirty else "hit-wb-clean"]
        elif code == 4:
            cost = CACHEOP_COST["hit-inv"] if hit else 0
        elif code == 0 and dirty:
            cost = CACHEOP_COST["hit-wbi-dirty"]
        else:
            cost = CACHEOP_COST["index-store-tag"]
        t.fixed(cost, done)
        return done

    def _cache_op(self, op: int, va: int, pa: int, cca: int) -> str:
        """CACHE op, va (24K Table 12.3, Tables 0-1 / 12.4 for ErrCtl.WST /
        SPR).  An operation on a cache that does not exist, a reserved
        one, or an address operation on an uncached address does nothing
        (the Release 6 rule; UNDEFINED before).  The L2 / L3 are
        physically indexed: their index operations take the set and way
        from the physical address.  Returns what was done."""
        target = op & 3
        c = {0: self.i, 1: self.d, 2: self.l3, 3: self.l2}.get(target)
        if c is None:
            return f"no such cache ({CACHE_NAMES[target]}): no operation"
        code = op >> 2
        m = self.m
        errctl = m.cp0_read(26, 0)
        name = OP_NAMES[c.kind][code]
        bus = self if c.level == 1 else c.below
        if code >= 4:                                    # address operations
            if cca not in self.cached_cca:
                return f"{name}: uncached address, no operation"
            done = self._hit_op(c, code, va if c.level == 1 else pa, pa, name, bus, cca)
            if (c.level == 1 and self.coherent(cca)
                    and (code in (4, 5, 6) if c.kind == "D" else code == 4)):
                more = self.cm.global_op(self, c.kind, code, pa)
                if more:
                    done += f"; globalized: {more}"
            return done
        ia = va if c.level == 1 else pa                  # the index address
        way = (ia >> c.index_bit) & ((1 << c.way_bits) - 1)
        s, off = c.set_of(ia), ia & c.line_mask
        if way >= c.ways:
            return f"{name}: way {way} does not exist, no operation"
        if errctl & EC_SPR:
            if c.level > 1:
                return "ErrCtl.SPR: no secondary operations, no operation"
            return "ErrCtl.SPR: no scratchpad RAM, no operation"
        wst = bool(errctl & EC_WST)
        ln = c.lines[s][way]
        where = f"{c.name} set {s} way {way}"
        tl = _TAGLO[c.kind]
        if code == 0:
            if c.writable and ln.valid and c._dirty(s, way):
                if c.check and c.parity:
                    c._check_data(s, way, 0, c.line)
                c.write_back(s, way, bus)
            self._invalidate(c, s, way)
            return f"{name} {where}"
        if code == 1:
            if wst:
                m._cp0_set(tl, self._ws_value(c, s))
                return f"Index Load WS {c.name} set {s}"
            self._load_tag(c, s, way, off, errctl)
            return f"{name} {where}"
        if code == 2:
            taglo = m.cp0_read(*tl)
            if wst:
                self._store_ws(c, s, taglo, errctl)
                return f"Index Store WS {c.name} set {s}"
            if ln.valid:
                c.stats.forget(c.line_pa(ln.tag, s))
            ln.tag = (taglo & M32) >> c.tag_shift
            ln.valid = bool(taglo & TAG_V)
            ln.lock = bool(taglo & TAG_L)
            ln.dirty = c.writable and bool(taglo & TAG_D)
            ln.shared = False
            if self.cm is not None and c.kind == "D" and not taglo & TAG_V and taglo & TAG_L:
                # Table 9.1: SHARED (D = 1: reserved -> invalid)
                shared = not taglo & TAG_D
                ln.valid, ln.dirty, ln.shared, ln.lock = shared, False, shared, False
            c._seal(ln)
            if c.parity and errctl & EC_PO:
                ln.tag_p = taglo & TAG_P
            if ln.valid:
                c.touch(s, way)
            else:
                c.make_lru(s, way)
            return f"{name} {where}"
        if code == 3 and c.level > 1:
            if wst:                                      # Index Store ECC
                if not c.parity:
                    return f"Index Store ECC {where}: no ECC, no operation"
                ln.ecc[off // c.group % c.groups] = m.cp0_read(28, 5) & 0xFF
                return f"Index Store ECC {where}"
            self._store_data(c, s, way, off, errctl)
            return f"Index Store Data {where}"
        if code == 3 and wst:
            self._store_data(c, s, way, off, errctl)
            return f"Index Store Data {where}"
        return f"{name}: reserved, no operation"

    def _hit_op(self, c: Cache, code: int, va: int, pa: int, name: str, bus,
                cca: int = CCA_WB) -> str:
        coherent = c.level == 1 and self.coherent(cca)
        if code == 5 and c.kind == "I":                  # Fill
            s, tag = c.set_of(va), c.tag_of(pa)
            if c.check and c.parity:
                c._check_tags(s)
            w = c.find(s, tag)
            c.replace(s, c.victim(s) if w is None else w, tag, bus)
            self._intent = self._fill = None
            return f"Fill set {s}"
        if code == 7:                                    # Fetch and Lock
            s, tag = c.set_of(va), c.tag_of(pa)
            if c.check and c.parity:
                c._check_tags(s)
            w = c.find(s, tag)
            if w is None:
                saved = c.ws[s]
                w = c.victim(s)
                if coherent and c.kind == "D":
                    self._intent = "own" if cca == CCA_CWBE else "share"
                c.replace(s, w, tag, bus)
                self._apply_fill(c, s, w, None, 0)
                self._intent = None
                if c.kind == "I":
                    c.ws[s] = saved                      # I: "no update"
            elif c.writable:
                c.touch(s, w)
            c.lines[s][w].lock = True
            c.lines[s][w].shared = False             # locked lines are not coherent
            c._seal(c.lines[s][w])
            return f"Fetch and Lock {c.name} set {s} way {w}"
        if code == 6 and c.kind == "I":
            return f"{name}: reserved, no operation"
        s, tag = c.set_of(va), c.tag_of(pa)
        if c.check and c.parity:
            c._check_tags(s)
        w = c.find(s, tag)
        if w is None:
            return f"{name}: miss, nothing to do"
        if code in (5, 6) and c._dirty(s, w):
            if c.check and c.parity:
                c._check_data(s, w, 0, c.line)
            c.write_back(s, w, bus)
        if code in (4, 5):
            self._invalidate(c, s, w)
        return f"{name} {c.name} set {s} way {w}"

    def _ws_value(self, c: Cache, s: int) -> int:
        """xTagLo in WS test mode.  L1: WSLRU 15:10 (I and D), WSD 19:16
        and WSDP 23:20 (D: dirty bits and their parity, ways 0..3).
        L2 / L3 (eMIPS's layout, MD00525 is not public): LRU 14:8, dirty
        bits 23:16 and their parity 31:24 (ways 0..7)."""
        if c.level > 1:
            value = (c.ws[s] & 0x7F) << 8
            for w in range(c.ways):
                ln = c.lines[s][w]
                value |= int(ln.dirty) << (16 + w)
                if c.parity:
                    value |= ln.dirty_p << (24 + w)
            return value & M32
        value = (c.ws[s] & 0x3F) << 10
        if c.kind == "D":
            for w in range(min(c.ways, 4)):
                ln = c.lines[s][w]
                value |= int(ln.dirty) << (16 + w)
                if c.parity:
                    value |= ln.dirty_p << (20 + w)
        return value

    def _store_ws(self, c: Cache, s: int, taglo: int, errctl: int) -> None:
        if c.level > 1:
            bits = 7 if c.plru else len(c.pairs)
            c.ws[s] = (taglo >> 8) & ((1 << bits) - 1)
            dshift, pshift, n = 16, 24, c.ways
        else:
            keep = c.ws[s] & ~0x3F
            c.ws[s] = keep | ((taglo >> 10) & 0x3F & ((1 << len(c.pairs)) - 1))
            if c.kind != "D":
                return
            dshift, pshift, n = 16, 20, min(c.ways, 4)
        for w in range(n):
            ln = c.lines[s][w]
            ln.dirty = bool(taglo >> (dshift + w) & 1)
            ln.dirty_p = ((taglo >> (pshift + w) & 1) if c.parity and errctl & EC_PO
                          else int(ln.dirty))

    def _load_tag(self, c: Cache, s: int, way: int, off: int, errctl: int) -> None:
        """Index Load Tag: TagLo, the data at the byte index (I, L2 / L3:
        the doubleword into DataHi / DataLo, D: the word into DDataLo),
        and the precode / data parity into ErrCtl (L1)."""
        m = self.m
        ln = c.lines[s][way]
        ptag = (ln.tag << c.tag_shift) & M32
        taglo = (ptag | (TAG_V if ln.valid else 0) | (TAG_L if ln.lock else 0)
                 | (TAG_D if c.writable and ln.dirty else 0)
                 | (ln.tag_p if c.parity else 0))
        if self.cm is not None and c.kind == "D" and ln.valid and ln.shared:
            taglo = (taglo & ~(TAG_V | TAG_D)) | COHERENT_SHARED     # Table 9.1
        order = "big" if m.big_endian else "little"
        pe = c.parity and errctl & EC_PE
        keep = errctl & ~((0x3F << EC_PCI_SHIFT) | (0x1FF << EC_PI_SHIFT) | 0xF)
        if c.level > 1:
            m._cp0_set((28, 4), taglo)
            base = off & ~7 & c.line_mask
            raw = bytes(ln.data[base:base + 8]).ljust(8, b"\0")
            value = int.from_bytes(raw, order)
            m._cp0_set((29, 5), value >> 32)
            m._cp0_set((28, 5), value & M32)
        elif c.kind == "I":
            m._cp0_set((28, 0), taglo)
            base = off & ~7 & c.line_mask
            w0 = int.from_bytes(ln.data[base:base + 4], order)
            w1 = int.from_bytes(ln.data[base + 4:base + 8], order) if c.line >= 8 else 0
            hi, lo = (w0, w1) if m.big_endian else (w1, w0)
            m._cp0_set((29, 1), hi)
            m._cp0_set((28, 1), lo)
            d = base // 8
            pre = ln.pre[d] if d < len(ln.pre) else 0
            pi = 0
            if pe:                                       # bit k: k-th least-significant byte
                for k in range(min(8, c.line - base)):
                    pi |= ((ln.data_p >> (base + _dword_byte(k, m.big_endian))) & 1) << k
                pi |= (ln.pre_p[d] if d < len(ln.pre_p) else 0) << 8
            m._cp0_set((26, 0), keep | (pre << EC_PCI_SHIFT) | (pi << EC_PI_SHIFT))
        else:
            m._cp0_set((28, 2), taglo)
            base = off & ~3 & c.line_mask
            m._cp0_set((28, 3), int.from_bytes(ln.data[base:base + 4], order))
            pd = 0
            if pe:
                for k in range(4):                       # PD[0]: least-significant byte
                    i = base + (3 - k if m.big_endian else k)
                    pd |= ((ln.data_p >> i) & 1) << k
            m._cp0_set((26, 0), keep | pd)

    def _store_data(self, c: Cache, s: int, way: int, off: int, errctl: int) -> None:
        """Index Store Data: IDataHi / IDataLo into the doubleword (I,
        with ErrCtl.WST), DDataLo into the word (D, with WST),
        L23DataHi / L23DataLo into the doubleword (L2 / L3; the ECC is
        always generated)."""
        m = self.m
        ln = c.lines[s][way]
        order = "big" if m.big_endian else "little"
        if c.level > 1:
            base = off & ~7 & c.line_mask
            value = (m.cp0_read(29, 5) << 32) | m.cp0_read(28, 5)
            n = min(8, c.line - base)
            raw = value.to_bytes(8, order)
            ln.data[base:base + n] = raw[:n]
            ln.known |= ((1 << n) - 1) << base
            c._set_data_parity(ln, base, n)
            return
        if c.kind == "I":
            base = off & ~7 & c.line_mask
            hi, lo = m.cp0_read(29, 1), m.cp0_read(28, 1)
            w0, w1 = (hi, lo) if m.big_endian else (lo, hi)
            raw = w0.to_bytes(4, order) + w1.to_bytes(4, order)
            n = min(8, c.line - base)
            ln.data[base:base + n] = raw[:n]
            ln.known |= ((1 << n) - 1) << base
            c._precode(ln, base, base + n, self)
            d = base // 8
            if errctl & EC_PCO and d < len(ln.pre):
                ln.pre[d] = (errctl >> EC_PCI_SHIFT) & 0x3F
            c._set_data_parity(ln, base, n)
            if c.parity:
                if d < len(ln.pre_p):
                    ln.pre_p[d] = _parity(ln.pre[d])
                if errctl & EC_PO:
                    pi = (errctl >> EC_PI_SHIFT) & 0x1FF
                    for k in range(8):                   # bit k: k-th least-significant byte
                        i = base + _dword_byte(k, m.big_endian)
                        if i < base + n:
                            bit = 1 << i
                            ln.data_p = ln.data_p | bit if pi >> k & 1 else ln.data_p & ~bit
                    if d < len(ln.pre_p):
                        ln.pre_p[d] = pi >> 8 & 1
        else:
            base = off & ~3 & c.line_mask
            ln.data[base:base + 4] = m.cp0_read(28, 3).to_bytes(4, order)
            ln.known |= 0xF << base
            c._set_data_parity(ln, base, 4)
            if c.parity and errctl & EC_PO:
                pd = errctl & 0xF
                for k in range(4):
                    i = base + (3 - k if m.big_endian else k)
                    bit = 1 << i
                    ln.data_p = ln.data_p | bit if pd >> k & 1 else ln.data_p & ~bit
