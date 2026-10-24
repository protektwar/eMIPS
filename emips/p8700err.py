"""The MIPS P8700's cache and TLB errors (PLAN M25 phase 5e2; MD01502 rev.
1.84: 2.2 TLB ECC, 3.3.1.1-3.3.1.3 the L1 I-cache's EDC, 3.3.3 the L1
D-cache's ECC, 3.3.8 the L2's, 7.3.4 mipscacheerr, 7.3.5 mipserrctl, Table
7.4 the Cache Error exception, 7.3.9 mipsconfig5.ERL).

:class:`CoreErrors` is a core's mipscacheerr / mipserrctl ("per-core"
registers: every hart of the core reads the same ones).  mipserrctl.PE
(reset 0) turns the checking of the L1s and the FTLB on; the L2's is
GCR_ERR_CONTROL.L2_ECC_EN (reset 1, emips.p8700cps).

- **L1 D-cache** (:class:`ECCCache` "D"): a 7-bit ECC per 32-bit word and
  per tag.  A single-bit error is corrected and scrubbed (mipscacheerr
  STATE 01, ERROR_BITS = the bit); a multiple-bit data error in a *clean*
  line is "corrected by invalidating" it (ERROR_BITS 0x3F) — the line is
  fetched again; in a *dirty* line, or in a tag, it is uncorrectable
  (STATE 10): the access takes a Cache Error.  A sub-word store reads its
  word first (3.3.3.4: the read-modify-write checks it).  A dirty victim
  with an uncorrectable error is a Cache Error the refilling access takes
  (S: "a capacity replacement"; a tag error there: F, memory silently
  corrupted).
- **L1 I-cache** (:class:`ECCCache` "I"): any EDC error makes the IFU fetch
  the line again — software-transparent; recorded as corrected.
- **The L2** (emips.cache's SEC-DED per doubleword in the CM): a corrected
  error is recorded (ARRAY 6 / 7); an uncorrectable one is taken as an
  imprecise Cache Error by the core that read it (an instruction refill:
  at once).
- **The FTLB** (emips.p8700mmu): ECC on each entry's tag and data (Table
  2.2): corrected and scrubbed, or uncorrectable — a Cache Error.

A Cache Error is the P8700's custom exception 48 (through mipstvec, in
M-mode) and sets mipsconfig5.ERL.  While mipscacheerr holds an
uncorrectable error (STATE 10), a second one sets F2 and S and keeps the
first's details; corrected errors do not overwrite it.  Choices:
``sim.DEVIATIONS["P8700 cache errors"]``.
"""

from __future__ import annotations

from .cache import Cache, CacheError, Geometry, ecc_check, ecc_encode

#: mipscacheerr STATE (31:30)
NONE, CORRECTED, UNCORRECTABLE = 0, 1, 2
#: ARRAY (29:26)
IC_TAG, IC_DATA, DC_TAG, DC_DATA, FTLB_TAG, FTLB_DATA, L2_TAG, L2_DATA = range(8)
ARRAY_NAMES = ("L1 I-cache tag", "L1 I-cache data", "L1 D-cache tag", "L1 D-cache data",
               "FTLB tag", "FTLB data", "L2 tag", "L2 data")
#: the uncorrectable-error flags (23:20)
F2, FATAL, PERSISTENT, SCAPEGOAT = 1 << 23, 1 << 22, 1 << 21, 1 << 20
#: ERROR_BITS: "bit position cannot be determined" (a clean line invalidated)
UNKNOWN_BIT = 0x3F
#: mipserrctl (7.3.5): PE 31, BUSTIMEOUT 19:10
PE = 1 << 31
ERRCTL_MASK = PE | 0x3FF << 10
#: the custom cause (Table 7.4)
CACHE_ERROR = 48


class CoreErrors:
    """A core's mipscacheerr / mipserrctl; the arrays that report to it."""

    def __init__(self) -> None:
        self.cacheerr = 0
        self.errctl = 0
        #: the core's emips.cache.Caches and the MMUs of its harts (PE)
        self.caches = None
        self.mmus: list = []
        #: corrected errors counted (the views)
        self.corrected = 0

    @property
    def pe(self) -> bool:
        return bool(self.errctl & PE)

    def write_errctl(self, v: int) -> None:
        self.errctl = v & ERRCTL_MASK
        on = self.pe
        c = self.caches
        if c is not None:
            for cache in (c.i, c.d):
                if cache is not None:
                    cache.check = on
        for mmu in self.mmus:
            mmu.ecc_check = on

    def record(self, state: int, array: int, *, bits: int = UNKNOWN_BIT, way: int = 0,
               index: int = 0, word: int = 0, fatal: bool = False,
               scapegoat: bool = False) -> None:
        """7.3.4: an error found; an uncorrectable one held is kept (a second
        sets F2 and S)."""
        if self.cacheerr >> 30 == UNCORRECTABLE:
            if state == UNCORRECTABLE:
                self.cacheerr |= F2 | SCAPEGOAT
            return
        v = state << 30 | array << 26 | (way & 7) << 17 | (index & 0x1FFF) << 4 | (word & 0xF)
        if state == CORRECTED:
            v |= (bits & 0x3F) << 20
            self.corrected += 1
        else:
            v |= (FATAL if fatal else 0) | (SCAPEGOAT if scapegoat else 0)
        self.cacheerr = v

    def from_error(self, err: CacheError) -> None:
        """An uncorrectable error found by emips.cache (an L2 one) that was
        not recorded where it was found."""
        if getattr(err, "recorded", False):
            return
        c = err.cache
        array = {"I": IC_TAG, "D": DC_TAG}.get(c.kind, L2_TAG) + (0 if err.tag else 1)
        line = getattr(c, "line_shift", 6)
        self.record(UNCORRECTABLE, array, way=err.way, index=err.index >> line,
                    fatal=err.fatal)

    def describe(self) -> str:
        v = self.cacheerr
        state = v >> 30
        if state == NONE:
            return "mipscacheerr: no error"
        array = ARRAY_NAMES[v >> 26 & 0xF] if (v >> 26 & 0xF) < 8 else "reserved"
        where = f"way {v >> 17 & 7}, index {v >> 4 & 0x1FFF}, word {v & 0xF}"
        if state == CORRECTED:
            bit = v >> 20 & 0x3F
            how = "line invalidated" if bit == UNKNOWN_BIT else f"bit {bit}"
            return f"mipscacheerr: corrected {array} error ({how}), {where}"
        flags = [n for b, n in ((F2, "F2"), (FATAL, "F"), (PERSISTENT, "P"),
                                (SCAPEGOAT, "S")) if v & b]
        return (f"mipscacheerr: uncorrectable {array} error, {where}"
                + (f" ({' '.join(flags)})" if flags else ""))


def _bit(raw: bytes, fixed: bytes) -> int:
    """The data bit a correction flipped (a check bit: 0x3E, beyond the word)."""
    for k, (a, b) in enumerate(zip(raw, fixed)):
        if a != b:
            return 8 * k + (a ^ b).bit_length() - 1
    return 0x3E


class ECCCache(Cache):
    """A P8700 L1 cache with its check bits (see the module docstring):
    ``ecc`` per 32-bit word, ``tag_p`` the tag's check bits.  ``owner`` =
    the core's Caches (a refetch), ``errs`` = its :class:`CoreErrors`."""

    #: the word a check covers
    WORD = 4

    def __init__(self, kind: str, geometry: Geometry, owner, errs: CoreErrors) -> None:
        super().__init__(kind, geometry, parity=True)
        self.owner, self.errs = owner, errs
        self.words = self.line // self.WORD
        zero = ecc_encode(bytes(self.WORD))
        for row in self.lines:
            for ln in row:
                ln.ecc = [zero] * self.words
                self._seal(ln)

    @property
    def _tag_array(self) -> int:
        return IC_TAG if self.kind == "I" else DC_TAG

    # -- the check bits ------------------------------------------------------------

    @staticmethod
    def _tag_bytes(ln) -> bytes:
        return (ln.tag & ((1 << 40) - 1)).to_bytes(5, "little")

    def _set_data_parity(self, ln, start: int, count: int) -> None:
        g = self.WORD
        for k in range(start // g, min((start + count + g - 1) // g, self.words)):
            ln.ecc[k] = ecc_encode(bytes(ln.data[k * g:(k + 1) * g]))

    def _seal(self, ln) -> None:
        ln.tag_p = ecc_encode(self._tag_bytes(ln))
        ln.dirty_p = int(ln.dirty)

    def _random_check_bits(self, ln, rng) -> None:
        ln.ecc = [rng.getrandbits(8) for _ in range(self.words)]

    def inject(self, s: int, w: int, *, array: str = "data", byte: int = 0,
               bit: int = 0) -> None:
        if array == "ecc":
            self.lines[s][w].ecc[byte // self.WORD] ^= 1 << bit
        else:
            super().inject(s, w, array=array, byte=byte, bit=bit)

    # -- checks ----------------------------------------------------------------------

    def _check_tags(self, s: int) -> None:
        for w, ln in enumerate(self.lines[s]):
            if not ln.valid:
                continue
            raw = self._tag_bytes(ln)
            status, fixed = ecc_check(raw, ln.tag_p)
            if status == "ok":
                continue
            if status == "corrected":
                ln.tag = int.from_bytes(fixed, "little")
                self._seal(ln)                      # (scrubbed)
                self.errs.record(CORRECTED, self._tag_array, bits=_bit(raw, fixed), way=w,
                                 index=s)
            elif self.kind == "I":                  # (the IFU fetches it again)
                self._drop(s, w)
                self.errs.record(CORRECTED, IC_TAG, way=w, index=s)
            else:
                self._uncorrectable(s, w, tag=True)

    def _check_data(self, s: int, w: int, start: int, count: int, *,
                    fatal: bool = False) -> None:
        ln = self.lines[s][w]
        g = self.WORD
        for k in range(start // g, min((start + count + g - 1) // g, self.words)):
            raw = bytes(ln.data[k * g:(k + 1) * g])
            status, fixed = ecc_check(raw, ln.ecc[k])
            if status == "ok":
                continue
            data_array = DC_DATA if self.kind == "D" else IC_DATA
            if status == "corrected":
                ln.data[k * g:(k + 1) * g] = fixed
                ln.ecc[k] = ecc_encode(fixed)       # (scrubbed)
                self.errs.record(CORRECTED, data_array, bits=_bit(raw, fixed), way=w,
                                 index=s, word=k)
            elif self.kind == "I" or not (fatal or ln.dirty):
                self.errs.record(CORRECTED, data_array, way=w, index=s, word=k)
                self._refetch(s, w)                 # (a clean copy: fetched again)
                return
            else:
                self._uncorrectable(s, w, word=k, scapegoat=fatal)

    def _uncorrectable(self, s: int, w: int, *, tag: bool = False, word: int = 0,
                       scapegoat: bool = False, fatal: bool = False) -> None:
        self.errs.record(UNCORRECTABLE, self._tag_array if tag else self._tag_array + 1,
                         way=w, index=s, word=word, fatal=fatal, scapegoat=scapegoat)
        err = CacheError(self, data=not tag, tag=tag, fatal=fatal, way=w,
                         index=s << self.line_shift,
                         what=f"{self.name} uncorrectable ECC error, set {s} way {w}")
        err.recorded = True
        raise err

    def _drop(self, s: int, w: int) -> None:
        from .cache import Caches
        Caches._invalidate(self, s, w)

    def _refetch(self, s: int, w: int) -> None:
        """Fetch a clean line again in place (its MESI state kept)."""
        ln = self.lines[s][w]
        shared = ln.shared
        saved, self.check = self.check, False
        try:
            Cache.replace(self, s, w, ln.tag, self.owner)
        finally:
            self.check = saved
        ln.shared = shared

    def replace(self, s: int, w: int, tag: int, bus, *, zero: bool = False) -> None:
        """A dirty victim is checked as it is written back (3.3.3.5
        Evictions): corrected on the fly, or uncorrectable — a tag error
        there is F (memory silently corrupted)."""
        ln = self.lines[s][w]
        if self.check and ln.valid and ln.dirty:
            raw = self._tag_bytes(ln)
            status, fixed = ecc_check(raw, ln.tag_p)
            if status == "corrected":
                ln.tag = int.from_bytes(fixed, "little")
                self._seal(ln)
            elif status == "uncorrectable":
                self._uncorrectable(s, w, tag=True, fatal=True, scapegoat=True)
            self._check_data(s, w, 0, self.line, fatal=True)
        saved, self.check = self.check, False
        try:
            super().replace(s, w, tag, bus, zero=zero)
        finally:
            self.check = saved


def equip(caches, errs: CoreErrors) -> None:
    """Give a core's Caches the P8700's ECC L1s (before the Coherence
    Manager attaches it; the caches start invalid)."""
    for name in ("i", "d"):
        old = getattr(caches, name)
        if old is not None:
            new = ECCCache(old.kind, old.geometry, caches, errs)
            new.invalidate_all()
            setattr(caches, name, new)
    caches.errs = errs
    errs.caches = caches
