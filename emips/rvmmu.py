"""RISC-V address translation (PLAN M25 phase 2b; The RISC-V Instruction
Set Manual, Volume II: Privileged Architecture 20250508 ch. 12 "Supervisor-
Level ISA": 12.1.11 satp, 12.3-12.6 Sv32 / Sv39 / Sv48 / Sv57).

:class:`RVMMU` is the MMU of an :class:`emips.simrv.RVMachine` (the
``attach / xlate / peek / describe`` interface every eMIPS MMU has):

- **Which accesses translate** (12.1.11 / 3.1.6.4): an access whose
  effective privilege is S or U while satp.MODE is not Bare — the
  effective privilege of a load / store is mstatus.MPP when MPRV = 1 in
  M-mode, a fetch's is the hart's mode.  M-mode and Bare: PA = VA.
- **The walk** (12.3.2): from satp.PPN, one PTE per level (4 bytes on
  Sv32, 8 on the others), read from physical memory through the caches;
  V = 0, R = 0 with W = 1, reserved bits (63:54: no Svnapot / Svpbmt /
  Svrsw60t59b), non-leaf PTEs with D / A / U set, a misaligned superpage
  and a VA whose upper bits are not the sign of its top bit — a page
  fault (12 fetch, 13 load, 15 store / AMO); the permission checks with U
  / SUM / MXR; A = 0, or D = 0 on a store: a page fault (Svade — eMIPS
  has no Svadu, so the hardware never writes a PTE, as Spike by default).
  A PTE beyond the platform's physical memory: an access fault of the
  access's kind.
- **The TLB**: translations of valid, permitted leaf PTEs are cached by
  (ASID, VPN) with their level and G bit (``entries`` deep, LRU); a PTE
  change takes effect only after SFENCE.VMA (12.2.1: rs1 = a VA or all,
  rs2 = an ASID or all; global entries ignore the ASID).  A satp write
  flushes nothing (a new ASID simply does not match).
- **Views** (``peek`` / ``describe``) walk without touching the TLB, the
  counters or the A / D bits.

Choices: ``sim.DEVIATIONS["RISC-V"]``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .mmu import Translation

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
PAGE = 12
#: the platform's physical address space (emips.simrv's PMAs: 32 bits)
PA_BITS = 32
#: PTE bits (12.3.1)
V, R, W, X, U, G, A, D = (1 << k for k in range(8))
#: page faults by access kind (priv. Table 14)
FAULT = {"fetch": 12, "load": 13, "pcload": 13, "store": 15}
ACCESS_FAULT = {"fetch": 1, "load": 5, "pcload": 5, "store": 7}


def pma_cca(m, pa: int) -> int:
    """The platform's PMAs (priv. 3.6): I/O — RARS's MMIO page and the
    devices of emips.rvplatform (``m.io_regions``) — is uncached (2),
    memory cacheable write-back (3)."""
    if 0xFFFF0000 <= pa <= M32:
        return 2
    for lo, hi in m.io_regions:
        if lo <= pa < hi:
            return 2
    return 3


@dataclass(frozen=True)
class Scheme:
    """A translation scheme: Sv32 / Sv39 / Sv48 / Sv57 (12.3-12.6)."""
    name: str
    levels: int
    idx_bits: int
    pte_size: int
    va_bits: int


SCHEMES = {(32, 1): Scheme("Sv32", 2, 10, 4, 32),
           (64, 8): Scheme("Sv39", 3, 9, 8, 39), (64, 9): Scheme("Sv48", 4, 9, 8, 48),
           (64, 10): Scheme("Sv57", 5, 9, 8, 57)}


class PageFault(Exception):
    """A walk that ended in a fault (``cause``: page or access fault)."""

    def __init__(self, cause: int, why: str) -> None:
        super().__init__(why)
        self.cause = cause
        self.why = why


@dataclass
class TLBEntry:
    asid: int
    vpn: int                 # the VA's page number, cut to the leaf's level
    level: int
    pte: int
    ppn: int                 # the leaf's PPN (a superpage keeps the VA's low bits)


class RVMMU:
    """See the module docstring.  ``entries``: the TLB's size."""

    mt = 0
    name = "RISC-V MMU"
    supervisor = True

    def __init__(self, entries: int = 64) -> None:
        self.entries = entries
        self.tlb: OrderedDict[tuple[int, int, int], TLBEntry] = OrderedDict()
        self.hits = self.misses = self.walks = 0

    def attach(self, m) -> None:
        self.m = m

    # -- the state -------------------------------------------------------------------

    def _satp(self, m) -> tuple[Scheme | None, int, int]:
        """(the scheme or None for Bare, ASID, root PPN)."""
        s = m.csrs.satp
        if m.xlen == 32:
            mode, asid, ppn = s >> 31, (s >> 22) & 0x1FF, s & 0x3FFFFF
        else:
            mode, asid, ppn = s >> 60, (s >> 44) & 0xFFFF, s & ((1 << 44) - 1)
        return SCHEMES.get((m.xlen, mode)), asid, ppn

    @staticmethod
    def effective_priv(m, kind: str) -> int:
        """3.1.6.4: MPRV makes M-mode loads / stores use MPP's privilege."""
        c = m.csrs
        if m.dm_access is not None:             # a Debug Module access (Debug spec 3.7.1.3:
            return c.status_field(11, 3) if m.dm_access == "virtual" else 3   # aamvirtual)
        if kind != "fetch" and m.priv == 3 and c.status_bit(17):       # MPRV
            if m.debug_mode and not c.dcsr & 0x10:      # (Debug spec 4.1: dcsr.mprven)
                return 3
            return c.status_field(11, 3)
        return m.priv

    @staticmethod
    def _cca(m, pa: int) -> int:
        return pma_cca(m, pa)

    # -- the machine's interface -------------------------------------------------------

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        kind = "store" if kind == "store" else "fetch" if kind == "fetch" else "load"
        priv = self.effective_priv(m, kind)
        scheme, asid, root = self._satp(m)
        if priv == 3 or scheme is None:
            pa = va & m.amask
        else:
            try:
                pa = self._translate(m, va, kind, priv, scheme, asid, root, record=True)
            except PageFault as f:
                from .simrv import RVTrap
                raise RVTrap(f.cause, va) from None
        if pa >> PA_BITS:
            from .simrv import RVTrap
            raise RVTrap(ACCESS_FAULT[kind], va)
        return pa, self._cca(m, pa)

    def peek(self, m, va: int) -> tuple[int, int] | None:
        if m.priv == 3 and not m.csrs.status_bit(17) or self._satp(m)[0] is None:
            return None if va >> PA_BITS else (va, self._cca(m, va))     # (the fast path)
        t = self.describe(m, va)
        return None if t.pa is None or t.pa >> PA_BITS else (t.pa, t.cca)

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        kind = "store" if kind == "store" else "fetch" if kind == "fetch" else "load"
        priv = self.effective_priv(m, kind)
        scheme, asid, root = self._satp(m)
        if priv == 3 or scheme is None:
            pa = va & m.amask
            why = "M-mode: PA = VA" if priv == 3 else "satp.MODE = Bare: PA = VA"
        else:
            try:
                pa = self._translate(m, va, kind, priv, scheme, asid, root, record=False)
                why = f"{scheme.name}, ASID {asid}"
            except PageFault as f:
                return Translation(va, None, 2, scheme.name, True, f"{scheme.name}: {f.why}",
                                   fault="page fault")
        if pa >> PA_BITS:
            return Translation(va, None, 2, "bare", False,
                               f"no physical memory above {PA_BITS} bits", fault="access fault")
        return Translation(va, pa, self._cca(m, pa), scheme.name if scheme and priv < 3
                           else "bare", scheme is not None and priv < 3, why)

    # -- the walk ----------------------------------------------------------------------

    def _translate(self, m, va: int, kind: str, priv: int, sch: Scheme, asid: int,
                   root: int, *, record: bool) -> int:
        top = va >> (sch.va_bits - 1)                   # 12.3.2 / 12.4: canonical VAs
        if m.xlen == 64 and top not in (0, (1 << (64 - sch.va_bits + 1)) - 1):
            raise PageFault(FAULT[kind], "the VA's upper bits are not its sign")
        vpn = (va >> PAGE) & ((1 << (sch.levels * sch.idx_bits)) - 1)
        if record:
            e = self._lookup(asid, vpn, sch)
            if e is not None:
                self.hits += 1
                self._check(m, e.pte, kind, priv, record=True)
                return self._pa(e, va, sch)
            self.misses += 1
            self.walks += 1
        pte, level, ppn = self._walk(m, va, kind, sch, root)
        self._check(m, pte, kind, priv, record=record)
        if record:
            self._insert(TLBEntry(asid, vpn >> (level * sch.idx_bits), level, pte, ppn))
        return self._pa(TLBEntry(asid, 0, level, pte, ppn), va, sch)

    @staticmethod
    def _pa(e: TLBEntry, va: int, sch: Scheme) -> int:
        low = (1 << (PAGE + e.level * sch.idx_bits)) - 1
        return ((e.ppn << PAGE) & ~low) | (va & low)

    def _walk(self, m, va: int, kind: str, sch: Scheme, root: int) -> tuple[int, int, int]:
        """12.3.2 steps 1-6: the leaf PTE, its level and its PPN."""
        base = root << PAGE
        for level in range(sch.levels - 1, -1, -1):
            idx = (va >> (PAGE + level * sch.idx_bits)) & ((1 << sch.idx_bits) - 1)
            addr = base + idx * sch.pte_size
            if addr >> PA_BITS:
                raise PageFault(ACCESS_FAULT[kind], "a PTE beyond physical memory")
            pte = m.load_pte(addr, sch.pte_size)
            if pte is None:
                raise PageFault(ACCESS_FAULT[kind], "PMP refuses the PTE's address")
            if sch.pte_size == 8 and pte >> 54:
                raise PageFault(FAULT[kind], "reserved PTE bits set (63:54)")
            if not pte & V or (pte & (R | W)) == W:
                raise PageFault(FAULT[kind], "an invalid PTE" if not pte & V else "W without R")
            ppn = (pte >> 10) & ((1 << 44) - 1 if sch.pte_size == 8 else (1 << 22) - 1)
            if pte & (R | X):                       # a leaf
                if ppn & ((1 << (level * sch.idx_bits)) - 1):
                    raise PageFault(FAULT[kind], "a misaligned superpage")
                return pte, level, ppn
            if pte & (D | A | U):
                raise PageFault(FAULT[kind], "D / A / U set in a non-leaf PTE")
            base = ppn << PAGE
        raise PageFault(FAULT[kind], "no leaf PTE at level 0")

    @staticmethod
    def _check(m, pte: int, kind: str, priv: int, *, record: bool) -> None:
        """12.3.2 step 5 / 7 and 3.1.6.4: U, SUM, MXR, the permission, A /
        D (Svade)."""
        c = m.csrs
        if priv == 0 and not pte & U:
            raise PageFault(FAULT[kind], "a supervisor page in U-mode")
        if priv == 1 and pte & U and (kind == "fetch" or not c.status_bit(18)):   # SUM
            raise PageFault(FAULT[kind], "a user page in S-mode" + ("" if kind == "fetch"
                                                                   else " without SUM"))
        ok = pte & X if kind == "fetch" else pte & W if kind == "store" else \
            pte & R or (c.status_bit(19) and pte & X)                            # MXR
        if not ok:
            raise PageFault(FAULT[kind], f"no {'X' if kind == 'fetch' else 'W' if kind == 'store' else 'R'} permission")
        if not pte & A or (kind == "store" and not pte & D):
            raise PageFault(FAULT[kind], "A = 0" if not pte & A else "D = 0 on a store")

    # -- the TLB -------------------------------------------------------------------------

    def _lookup(self, asid: int, vpn: int, sch: Scheme) -> TLBEntry | None:
        for level in range(sch.levels):
            key_vpn = vpn >> (level * sch.idx_bits)
            for key in ((asid, key_vpn, level), (-1, key_vpn, level)):
                e = self.tlb.get(key)
                if e is not None:
                    self.tlb.move_to_end(key)
                    return e
        return None

    def _insert(self, e: TLBEntry) -> None:
        key = (-1 if e.pte & G else e.asid, e.vpn, e.level)
        self.tlb[key] = e
        self.tlb.move_to_end(key)
        while len(self.tlb) > self.entries:
            self.tlb.popitem(last=False)

    def sfence(self, m, va: int | None, asid: int | None) -> None:
        """SFENCE.VMA (12.2.1): ``va`` None = every address, ``asid`` None =
        every address space; an ASID-specific fence keeps global entries."""
        sch = self._satp(m)[0] or SCHEMES[(m.xlen, 1 if m.xlen == 32 else 8)]
        mask = (1 << (sch.levels * sch.idx_bits)) - 1
        vpn = None if va is None else (va >> PAGE) & mask
        for key in list(self.tlb):
            e = self.tlb[key]
            if asid is not None and (e.pte & G or e.asid != asid):
                continue
            if vpn is not None and vpn >> (e.level * sch.idx_bits) != e.vpn:
                continue
            del self.tlb[key]
