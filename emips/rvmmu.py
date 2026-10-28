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
  / SUM / MXR; A = 0, or D = 0 on a store: a page fault (Svade) — or with
  Svadu and xenvcfg.ADUE (LF8, priv. 16) a real PTE store sets them
  (through the caches, PMP-checked as an S-mode write; refused in an I/O
  region: an access fault).  A PTE beyond the platform's physical memory:
  an access fault of the access's kind.
- **LF8**: Svnapot (N with ppn[3:0] = 1000 at level 0: a 64 KB page; 13),
  Svpbmt (PBMT 1 NC = the "uncached accelerated" CCA 7, 2 IO = uncached,
  3 reserved; 14 — the G-stage's, then the VS-stage's), the reserved bits
  60:54, menvcfg / henvcfg PBMTE.
- **The TLB**: translations of valid, permitted leaf PTEs are cached by
  (ASID, VPN) with their level and G bit (``entries`` deep, LRU); a PTE
  change takes effect only after SFENCE.VMA (12.2.1: rs1 = a VA or all,
  rs2 = an ASID or all; global entries ignore the ASID).  A satp write
  flushes nothing (a new ASID simply does not match).
- **Views** (``peek`` / ``describe``) walk without touching the TLB, the
  counters, the caches (``load_pte(view=True)``) or the A / D bits.

Choices: ``sim.DEVIATIONS["RISC-V"]``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .mmu import Translation

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
PAGE = 12
#: PTE bits (12.3.1)
V, R, W, X, U, G, A, D = (1 << k for k in range(8))
#: xenvcfg.SSE (Zicfiss, 3.1.18 / 21.2.5): the stage's shadow-stack pages
SSE = 1 << 3
#: Svnapot's N, Svpbmt's PBMT (62:61), the reserved bits 60:54 (priv. 12.4, 13, 14)
N = 1 << 63
PBMT_SHIFT = 61
RESERVED = 0x7F << 54
#: the CCAs PBMT gives (14, Table 37): NC = uncached accelerated, IO = uncached
PBMT_CCA = {1: 7, 2: 2}
#: page faults by access kind (priv. Table 14)
#: (Zicfiss, priv. 22.2.3: a shadow-stack access — "ssload" / "ssstore",
#: :meth:`RVMMU.kind_of` — reports store faults, even SSPOPCHK's load)
FAULT = {"fetch": 12, "load": 13, "pcload": 13, "store": 15, "ssload": 15, "ssstore": 15}
ACCESS_FAULT = {"fetch": 1, "load": 5, "pcload": 5, "store": 7, "ssload": 7, "ssstore": 7}
#: the access kinds that need a page's D bit (12.3.2 step 9)
DIRTYING = ("store", "ssstore")
#: the TLB counters' access kinds (DTLB / ITLB events)
_STAT = {"fetch": "fetch", "load": "load", "store": "store", "ssload": "load",
         "ssstore": "store"}


def pma_cca(m, pa: int) -> int:
    """The platform's PMAs (priv. 3.6): I/O — RARS's MMIO page and the
    devices of emips.rvplatform (``m.io_regions``) — is uncached (2),
    memory cacheable write-back (3).  The P8700: its pmacfg / ERL first
    (emips.p8700); behind its Coherence Manager cacheable memory is
    coherent (MD01502 3.4, code 0: "Cacheable, coherent, write-back,
    write-allocate, read misses request shared" = the MIPS CCA 5)."""
    p = m.p8700
    if p is not None:
        cca = p.pma(pa)
        if cca is not None:
            return 5 if cca == 3 and m.caches is not None and m.caches.cm is not None else cca
    if 0xFFFF0000 <= pa <= M32:
        return 2
    for lo, hi in m.io_regions:
        if lo <= pa < hi:
            return 2
    if p is not None and m.caches is not None and m.caches.cm is not None:
        return 5
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
    napot: int = 0           # Svnapot: 4 = a 64 KB page (its PPN's low 4 bits 0)


class GuestFault(Exception):
    """A G-stage translation that failed (priv. 21.5.2): a guest-page fault
    of the access's kind for guest physical address ``gpa``; ``implicit``:
    while reading a VS-level page table (htinst / mtinst then get Table
    49's pseudoinstruction)."""

    def __init__(self, cause: int, gpa: int, implicit: bool, why: str,
                 write: bool = False) -> None:
        super().__init__(why)
        self.cause = cause
        self.gpa = gpa
        self.implicit = implicit
        self.why = why
        #: an implicit write (Svadu's update of a VS-level PTE: Table 49's 0x?020)
        self.write = write


#: the guest-page faults by access kind (21.6.1, Table 43)
GUEST_FAULT = {"fetch": 20, "load": 21, "store": 23, "ssload": 23, "ssstore": 23}


@dataclass
class GuestEntry:
    """A guest translation in the TLB (LF7's decision): both stages in one
    entry — guest virtual page -> supervisor physical page at the smaller
    of the two page sizes (``shift``) — tagged with hgatp.VMID, the VS
    ASID (-1: a global VS page or vsatp Bare) and both MODEs; the PTEs it
    came from (None: that stage is Bare), the guest physical page and the
    G-stage page it lies in (HFENCE.GVMA by address)."""
    vmid: int
    asid: int
    vs_mode: int
    g_mode: int
    shift: int
    vpn: int
    vs_pte: int | None
    g_pte: int | None
    gpa: int
    spa: int
    g_base: int
    g_shift: int


class RVMMU:
    """See the module docstring.  ``entries``: the TLB's size."""

    mt = 0
    name = "RISC-V MMU"
    supervisor = True

    def __init__(self, entries: int = 64) -> None:
        self.entries = entries
        self.tlb: OrderedDict[tuple, TLBEntry | GuestEntry] = OrderedDict()
        self.hits = self.misses = self.walks = 0
        #: hits / misses by access kind (fetch / load / store: the counters'
        #: DTLB / ITLB events, emips.rvhpm)
        self.kind_stats: dict[str, list[int]] = {"fetch": [0, 0], "load": [0, 0],
                                                 "store": [0, 0]}
        #: the page sizes (shifts) guest entries have been made with
        self._gshifts: set[int] = set()
        #: a 64 KB Svnapot entry is in the TLB
        self._napot = False
        #: the last translation's PBMT (0: the PMA; emips.rvmmu.PBMT_CCA)
        self._pbmt = 0

    def attach(self, m) -> None:
        self.m = m

    # -- the state -------------------------------------------------------------------

    def _satp(self, m) -> tuple[Scheme | None, int, int]:
        """(the scheme or None for Bare, ASID, root PPN)."""
        return self._atp(m, m.csrs.satp)

    @staticmethod
    def _atp(m, s: int) -> tuple[Scheme | None, int, int]:
        """satp / vsatp's fields: (scheme, ASID, root PPN)."""
        if m.xlen == 32:
            mode, asid, ppn = s >> 31, (s >> 22) & 0x1FF, s & 0x3FFFFF
        else:
            mode, asid, ppn = s >> 60, (s >> 44) & 0xFFFF, s & ((1 << 44) - 1)
        return SCHEMES.get((m.xlen, mode)), asid, ppn

    @staticmethod
    def _hgatp(m) -> tuple[Scheme | None, int, int]:
        """hgatp's fields (21.2.10): (the G-stage scheme — Sv32x4 ... — as
        its satp twin, or None for Bare; VMID; root PPN)."""
        s = m.csrs.hgatp
        if m.xlen == 32:
            mode, vmid, ppn = s >> 31, (s >> 22) & 0x7F, s & 0x3FFFFF
        else:
            mode, vmid, ppn = s >> 60, (s >> 44) & 0x3FFF, s & ((1 << 44) - 1)
        return SCHEMES.get((m.xlen, mode)), vmid, ppn

    @staticmethod
    def effective(m, kind: str) -> tuple[int, int]:
        """(privilege, V) of an access (3.1.6.4 / 21.4.1 Table 42): MPRV makes
        M-mode loads / stores use MPP's privilege and MPV's V (unless MPP
        = M); HLV / HLVX / HSV hstatus.SPVP's with V = 1 (21.3.1); a Debug
        Module access M-mode's or, with aamvirtual, MPRV's; a fetch the
        hart's mode."""
        c = m.csrs
        if m.dm_access is not None:             # a Debug Module access (Debug spec 3.7.1.3:
            if m.dm_access != "virtual":        # aamvirtual)
                return 3, 0
            mpp = c.status_field(11, 3)
            return mpp, (c.mstatus >> 39 & 1) if mpp != 3 else 0
        if kind != "fetch":
            if m.hyp_access is not None:
                return c.hstatus >> 8 & 1, 1    # (SPVP)
            if m.priv == 3 and c.status_bit(17) and not (           # MPRV (Smrnmi 8.3:
                    c.smrnmi and not c.mnstatus & 8):                 # not while NMIE = 0)
                if m.debug_mode and not c.dcsr & 0x10:      # (Debug spec 4.1: dcsr.mprven)
                    return 3, 0
                mpp = c.status_field(11, 3)
                return mpp, (c.mstatus >> 39 & 1) if mpp != 3 else 0
        return m.priv, m.v

    @staticmethod
    def effective_priv(m, kind: str) -> int:
        """The privilege of an access (:meth:`effective`)."""
        return RVMMU.effective(m, kind)[0]

    @staticmethod
    def _cca(m, pa: int) -> int:
        return pma_cca(m, pa)

    # -- the machine's interface -------------------------------------------------------

    @staticmethod
    def kind_of(m, kind: str) -> str:
        """fetch / load / store — or, during a Zicfiss instruction's access
        (``m.ss_access``), ssload / ssstore (priv. 22.2.3)."""
        if kind == "fetch":
            return kind
        if m.ss_access:
            return "ssstore" if kind == "store" else "ssload"
        return "store" if kind == "store" else "load"

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        kind = self.kind_of(m, kind)
        self._pbmt = 0
        priv, virt = self.effective(m, "store" if kind in DIRTYING else kind)
        if virt and priv < 3:
            pa = self._guest_xlate(m, va, kind, priv)
        else:
            scheme, asid, root = self._satp(m)
            if kind.startswith("ss") and (priv == 3 or scheme is None):
                from .simrv import RVTrap               # (22.2.3: Bare below M, any in M)
                raise RVTrap(ACCESS_FAULT[kind], va, gva=False)
            if priv == 3 or scheme is None:
                pa = va & m.amask
            else:
                try:
                    pa = self._translate(m, va, kind, priv, scheme, asid, root, record=True)
                except PageFault as f:
                    from .simrv import RVTrap
                    raise RVTrap(f.cause, va, gva=False) from None
        if pa >> m.pabits:
            from .simrv import RVTrap
            raise RVTrap(ACCESS_FAULT[kind], va, gva=bool(virt))
        return pa, PBMT_CCA.get(self._pbmt) or self._cca(m, pa)

    def _guest_xlate(self, m, va: int, kind: str, priv: int) -> int:
        """A V = 1 access: two-stage translation, its faults as traps
        (21.5.2: a VS-stage page fault, a G-stage guest-page fault with
        the GPA >> 2 for htval / mtval2, an access fault)."""
        from .simrv import RVTrap
        try:
            return self._guest(m, va, kind, priv, record=True)
        except PageFault as f:
            raise RVTrap(f.cause, va, gva=True) from None
        except GuestFault as g:
            tinst = ((0x3000 if m.xlen == 64 else 0x2000) | (0x20 if g.write else 0)) \
                if g.implicit else 0            # (Table 49: a VS-level PTE read / write)
            raise RVTrap(g.cause, va, tval2=g.gpa >> 2, tinst=tinst, gva=True) from None

    def peek(self, m, va: int) -> tuple[int, int] | None:
        priv, virt = self.effective(m, "load")
        if not virt and (priv == 3 or self._satp(m)[0] is None):
            return None if va >> m.pabits else (va, self._cca(m, va))     # (the fast path)
        t = self.describe(m, va)
        return None if t.pa is None or t.pa >> m.pabits else (t.pa, t.cca)

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        kind = kind if kind in ("ssload", "ssstore") else self.kind_of(m, kind)
        self._pbmt = 0
        priv, virt = self.effective(m, kind)
        if virt and priv < 3:
            return self._describe_guest(m, va, kind, priv)
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
        if pa >> m.pabits:
            return Translation(va, None, 2, "bare", False,
                               f"no physical memory above {m.pabits} bits", fault="access fault")
        cca = PBMT_CCA.get(self._pbmt) if scheme and priv < 3 else None
        if cca is not None:
            why += f", PBMT {'NC' if self._pbmt == 1 else 'IO'}"
        return Translation(va, pa, cca or self._cca(m, pa), scheme.name if scheme and priv < 3
                           else "bare", scheme is not None and priv < 3, why)

    def _describe_guest(self, m, va: int, kind: str, priv: int) -> Translation:
        """A view of a V = 1 translation (no TLB, counters, A / D bits):
        the stages, the GPA, which stage faulted."""
        vs, asid, _ = self._atp(m, m.csrs.vsatp)
        g, vmid, _ = self._hgatp(m)
        segment = f"{vs.name if vs else 'Bare'} + {g.name + 'x4' if g else 'Bare'}"
        stages = (f"VS-stage {vs.name + f', ASID {asid}' if vs else 'Bare'}; "
                  f"G-stage {g.name + 'x4' if g else 'Bare'}, VMID {vmid}")
        try:
            pa = self._guest(m, va, kind, priv, record=False, gpa_out=(out := []))
        except PageFault as f:
            stage = "VS-stage" if f.cause in (12, 13, 15) else "the walk"
            return Translation(va, None, 2, segment, True, f"{stage}: {f.why} ({stages})",
                               fault="page fault" if f.cause in (12, 13, 15) else "access fault")
        except GuestFault as gf:
            what = "a VS-level page table at " if gf.implicit else ""
            return Translation(va, None, 2, segment, True,
                               f"G-stage: {gf.why} ({what}GPA 0x{gf.gpa:x}; {stages})",
                               fault="guest-page fault")
        if pa >> m.pabits:
            return Translation(va, None, 2, segment, True,
                               f"no physical memory above {m.pabits} bits", fault="access fault")
        cca = PBMT_CCA.get(self._pbmt)
        return Translation(va, pa, cca or self._cca(m, pa), segment, True,
                           f"{stages}: GPA 0x{out[0]:x}"
                           + (f", PBMT {'NC' if self._pbmt == 1 else 'IO'}" if cca else ""))

    # -- the walk ----------------------------------------------------------------------

    @staticmethod
    def _env(m, vs: bool) -> int:
        """The xenvcfg a stage obeys (3.1.18 / 21.2.5): menvcfg for S-mode
        and the G-stage, henvcfg (as read) for the VS-stage."""
        c = m.csrs
        return c.henv() if vs else c.menvcfg

    def _translate(self, m, va: int, kind: str, priv: int, sch: Scheme, asid: int,
                   root: int, *, record: bool) -> int:
        top = va >> (sch.va_bits - 1)                   # 12.3.2 / 12.4: canonical VAs
        if m.xlen == 64 and top not in (0, (1 << (64 - sch.va_bits + 1)) - 1):
            raise PageFault(FAULT[kind], "the VA's upper bits are not its sign")
        vpn = (va >> PAGE) & ((1 << (sch.levels * sch.idx_bits)) - 1)
        env = m.csrs.menvcfg
        adue = bool(env & (1 << 61)) and "svadu" in m.extensions
        if record:
            e = self._lookup(asid, vpn, sch)
            if e is not None and not (adue and kind in DIRTYING and not e.pte & D):
                self.hits += 1                  # (a store to a clean page walks again:
                self.kind_stats[_STAT[kind]][0] += 1   # 12.3.2 step 9 may not use the TLB)
                self._check(m, e.pte, kind, priv, record=True)
                self._pbmt = e.pte >> PBMT_SHIFT & 3 if sch.pte_size == 8 else 0
                return self._pa(e, va, sch)
            self.misses += 1
            self.kind_stats[_STAT[kind]][1] += 1
            self.walks += 1
        pte, level, ppn, addr, napot = self._walk(m, va, kind, sch, root, view=not record,
                                                  env=env)
        pte = self._ad(m, pte, addr, kind, priv, sch, adue, record=record)
        self._pbmt = pte >> PBMT_SHIFT & 3 if sch.pte_size == 8 else 0
        e = TLBEntry(asid, vpn >> (level * sch.idx_bits + napot), level, pte, ppn, napot)
        if record:
            self._insert(e)
        return self._pa(e, va, sch)

    def _ad(self, m, pte: int, addr: int, kind: str, priv: int, sch: Scheme, adue: bool, *,
            record: bool, store_pte=None, **check) -> int:
        """12.3.2 steps 6-9: the permission checks, then A / D — a page
        fault (Svade) or, with ``adue`` (Svadu), the PTE written back with A
        (and D for a store) set; ``store_pte(addr, new)`` does the write
        (default: the hart's PTE store; False = refused: an access fault).
        Views (``record`` False) never write."""
        self._check(m, pte, kind, priv, record=record, ad=not adue, **check)
        need = A | (D if kind in DIRTYING else 0)
        if not adue or pte & need == need:
            return pte
        new = pte | need
        if record:
            ok = store_pte(addr, new) if store_pte is not None else \
                m.store_pte(addr, new, sch.pte_size)
            if not ok:
                raise PageFault(ACCESS_FAULT[kind], "Svadu's PTE update refused (PMP / an "
                                "I/O region)")
        return new

    @staticmethod
    def _pa(e: TLBEntry, va: int, sch: Scheme) -> int:
        low = (1 << (PAGE + e.level * sch.idx_bits + e.napot)) - 1
        return ((e.ppn << PAGE) & ~low) | (va & low)

    def _walk(self, m, va: int, kind: str, sch: Scheme, root: int,
              view: bool = False, pte_addr=None, env: int = 0
              ) -> tuple[int, int, int, int, int]:
        """12.3.2 steps 1-5 (Spike's order): the leaf PTE, its level, its
        PPN, its address (before ``pte_addr``: a VS-level PTE's guest
        physical address) and Svnapot's 4 (a 64 KB page) or 0.
        ``pte_addr``: a PTE's address -> the physical address to read (a
        VS-level page table: its G-stage translation, 21.5); ``env`` the
        xenvcfg of the stage (PBMTE)."""
        base = root << PAGE
        wide = sch.pte_size == 8
        for level in range(sch.levels - 1, -1, -1):
            idx = (va >> (PAGE + level * sch.idx_bits)) & ((1 << sch.idx_bits) - 1)
            where = base + idx * sch.pte_size
            addr = where if pte_addr is None else pte_addr(where)
            if addr >> m.pabits:
                raise PageFault(ACCESS_FAULT[kind], "a PTE beyond physical memory")
            pte = m.load_pte(addr, sch.pte_size, view)
            if pte is None:
                raise PageFault(ACCESS_FAULT[kind], "PMP refuses the PTE's address")
            if wide:
                self._bits_ok(m, pte, kind, env, PageFault, FAULT[kind])
            ss_page = pte & (R | W | X) == W and bool(env & SSE)    # (22.2.3: a shadow stack)
            if not pte & V or ((pte & (R | W)) == W and not ss_page):
                raise PageFault(FAULT[kind], "an invalid PTE" if not pte & V else
                                "W without R" + (" (xwr = 010 is reserved while the "
                                                 "stage's xenvcfg.SSE = 0)"
                                                 if pte & (R | W | X) == W else ""))
            ppn = (pte >> 10) & ((1 << 44) - 1 if wide else (1 << 22) - 1)
            if pte & (R | X) or ss_page:            # a leaf
                napot = self._napot_ok(pte, level, ppn, PageFault, FAULT[kind])
                if not napot and ppn & ((1 << (level * sch.idx_bits)) - 1):
                    raise PageFault(FAULT[kind], "a misaligned superpage")
                return pte, level, ppn & ~((1 << napot) - 1), where, napot
            if pte & (D | A | U) or (wide and pte & (N | 3 << PBMT_SHIFT)):
                raise PageFault(FAULT[kind], "D / A / U / N / PBMT set in a non-leaf PTE")
            base = ppn << PAGE
        raise PageFault(FAULT[kind], "no leaf PTE at level 0")

    @staticmethod
    def _bits_ok(m, pte: int, kind: str, env: int, exc, *args) -> None:
        """The bits 63:54 of an Sv39-Sv57 PTE (12.4, 13, 14): reserved 60:54,
        N without Svnapot, PBMT without xenvcfg.PBMTE (or Svpbmt), PBMT 3."""
        if pte & RESERVED:
            raise exc(*args, "reserved PTE bits set (60:54)")
        if pte & N and "svnapot" not in m.extensions:
            raise exc(*args, "N set without Svnapot")
        pbmt = pte >> PBMT_SHIFT & 3
        if pbmt and not (env & (1 << 62) and "svpbmt" in m.extensions):
            raise exc(*args, "PBMT set while xenvcfg.PBMTE = 0")
        if pbmt == 3:
            raise exc(*args, "PBMT = 3 (reserved)")

    @staticmethod
    def _napot_ok(pte: int, level: int, ppn: int, exc, *args) -> int:
        """Svnapot (13, Table 36): N on a level-0 leaf with ppn[3:0] = 1000 —
        a 64 KB page (4); any other encoding with N a page fault."""
        if not pte & N:
            return 0
        if level or ppn & 0xF != 0b1000:
            raise exc(*args, "a reserved Svnapot encoding")
        return 4

    @staticmethod
    def _check(m, pte: int, kind: str, priv: int, *, record: bool, sum_: int | None = None,
               mxr: int | None = None, hlvx: bool = False, ad: bool = True,
               env: int | None = None) -> None:
        """12.3.2 step 5 / 7 and 3.1.6.4: U, SUM, MXR, the permission, A /
        D (Svade; ``ad`` False: Svadu updates them instead).  The VS-stage
        (21.5): vsstatus's SUM, vsstatus.MXR OR the HS-level MXR (``sum_`` /
        ``mxr``); HLVX needs X instead of R.  Zicfiss (22.2.3, Spike's
        order): a shadow-stack page (xwr = 010; reserved — a page fault —
        while the stage's xenvcfg.SSE, ``env``, is 0) takes only
        shadow-stack stores; other stores (and CBOs: ``m.cbo_access``) are
        a store access fault, fetches an instruction access fault, loads
        read it whatever MXR; a shadow-stack access to any other page but a
        read-only one is a store access fault (read-only: the store page
        fault of the permission check)."""
        c = m.csrs
        sum_ = c.status_bit(18) if sum_ is None else sum_
        mxr = c.status_bit(19) if mxr is None else mxr
        if priv == 0 and not pte & U:
            raise PageFault(FAULT[kind], "a supervisor page in U-mode")
        if priv == 1 and pte & U and (kind == "fetch" or not sum_):           # SUM
            raise PageFault(FAULT[kind], "a user page in S-mode" + ("" if kind == "fetch"
                                                                   else " without SUM"))
        ss_page = pte & (R | W | X) == W
        ss = kind in ("ssload", "ssstore")
        if ss_page:
            env = c.menvcfg if env is None else env
            if not env & SSE:                   # (a cached SS page after SSE went 0)
                raise PageFault(FAULT[kind], "W without R (xwr = 010 is reserved while "
                                "the stage's xenvcfg.SSE = 0)")
            if kind == "store" or (getattr(m, "cbo_access", None) and not ss):
                raise PageFault(ACCESS_FAULT["store"], "a shadow-stack page: only shadow-"
                                "stack stores (22.2.3)")
            if kind == "fetch":
                raise PageFault(ACCESS_FAULT["fetch"], "a fetch from a shadow-stack page")
        elif ss and (pte & (R | W) == (R | W) or pte & X or (pte >> PBMT_SHIFT & 3 == 2
                                                            and m.xlen == 64)):
            raise PageFault(ACCESS_FAULT[kind], "a shadow-stack access to a page that is "
                            "neither a shadow-stack nor a read-only one (22.2.3)")
        ok = pte & X if kind == "fetch" or hlvx else pte & W if kind in DIRTYING or ss else \
            pte & R or (mxr and pte & X) or ss_page                            # MXR
        if not ok:
            need = "X" if kind == "fetch" or hlvx else "W" if kind in DIRTYING or ss else "R"
            raise PageFault(FAULT[kind], f"no {need} permission")
        if ad and (not pte & A or (kind in DIRTYING and not pte & D)):
            raise PageFault(FAULT[kind], "A = 0" if not pte & A else "D = 0 on a store")

    # -- two-stage translation (priv. 21.5) ----------------------------------------------

    def _guest(self, m, va: int, kind: str, priv: int, *, record: bool,
               gpa_out: list | None = None) -> int:
        """A V = 1 access (21.5): the VS-stage (vsatp; its page-table reads
        through the G-stage as implicit loads) to a guest physical address,
        then the G-stage (hgatp) to the supervisor physical one.  Svpbmt:
        the G-stage's PBMT, then the VS-stage's (14)."""
        c = m.csrs
        hlvx = m.hyp_access == "x" and kind == "load"
        vs, asid, vroot = self._atp(m, c.vsatp)
        g, vmid, groot = self._hgatp(m)
        if vs is None and kind in ("ssload", "ssstore"):    # (22.2.3: vsatp Bare)
            raise PageFault(ACCESS_FAULT[kind], "a shadow-stack access with vsatp Bare")
        if vs is None and g is None:
            if gpa_out is not None:
                gpa_out.append(va & m.amask)
            return va & m.amask                 # (both stages Bare)
        vs_mode = c.vsatp >> (31 if m.xlen == 32 else 60)
        g_mode = c.hgatp >> (31 if m.xlen == 32 else 60)
        vs_sum, hs_mxr = c.vsstatus >> 18 & 1, c.status_bit(19)
        vs_mxr = (c.vsstatus >> 19 & 1) | hs_mxr
        svadu = "svadu" in m.extensions
        vs_adue = svadu and bool(c.henv() & (1 << 61))
        g_adue = svadu and bool(c.menvcfg & (1 << 61))
        wide = m.xlen == 64
        if record:
            e = self._glookup(vmid, vs_mode, asid, g_mode, va)
            if e is not None and not (kind in DIRTYING and (
                    (vs_adue and e.vs_pte is not None and not e.vs_pte & D) or
                    (g_adue and e.g_pte is not None and not e.g_pte & D))):
                self.hits += 1
                self.kind_stats[_STAT[kind]][0] += 1
                if e.vs_pte is not None:
                    self._check(m, e.vs_pte, kind, priv, record=True, sum_=vs_sum, mxr=vs_mxr,
                                hlvx=hlvx, env=c.henv())
                gpa = e.gpa | (va & ((1 << e.shift) - 1))
                if e.g_pte is not None:
                    self._gcheck(e.g_pte, kind, gpa, hlvx=hlvx, mxr=hs_mxr, implicit=False)
                self._pbmt = self._final_pbmt(wide, e.vs_pte, e.g_pte)
                return e.spa | (va & ((1 << e.shift) - 1))
            self.misses += 1
            self.kind_stats[_STAT[kind]][1] += 1
            self.walks += 1
        if vs is None:
            gpa, vs_pte, vs_shift, global_ = va & m.amask, None, None, False
        else:
            top = va >> (vs.va_bits - 1)
            if m.xlen == 64 and top not in (0, (1 << (64 - vs.va_bits + 1)) - 1):
                raise PageFault(FAULT[kind], "the VA's upper bits are not its sign")

            def pte_addr(a: int) -> int:        # (an implicit load at the G-stage)
                return self._gstage(m, a, "load", kind, implicit=True, view=not record)[0]
            vs_pte, level, ppn, gaddr, napot = self._walk(m, va, kind, vs, vroot,
                                                          view=not record, pte_addr=pte_addr,
                                                          env=c.henv())

            def vs_store(a: int, new: int) -> bool:     # (21.5: the G-stage page of the
                spa = self._gstage(m, a, "store", kind, implicit=True, view=False,  # VS PTE
                                   gva_pte=True)[0]                     # must be writable)
                return m.store_pte(spa, new, vs.pte_size)
            vs_pte = self._ad(m, vs_pte, gaddr, kind, priv, vs, vs_adue, record=record,
                              store_pte=vs_store, sum_=vs_sum, mxr=vs_mxr, hlvx=hlvx,
                              env=c.henv())
            vs_shift = PAGE + level * vs.idx_bits + napot
            gpa = self._pa(TLBEntry(asid, 0, level, vs_pte, ppn, napot), va, vs)
            global_ = bool(vs_pte & G)
        if gpa_out is not None:
            gpa_out.append(gpa)
        spa, g_pte, g_shift = self._gstage(m, gpa, kind, kind, implicit=False, hlvx=hlvx,
                                           mxr=hs_mxr, view=not record)
        self._pbmt = self._final_pbmt(wide, vs_pte, g_pte)
        if record:
            shift = min(s for s in (vs_shift, g_shift) if s is not None)
            low = (1 << shift) - 1
            gs = g_shift if g_shift is not None else shift
            self._ginsert(GuestEntry(vmid, -1 if vs is None or global_ else asid, vs_mode,
                                     g_mode, shift, va >> shift, vs_pte, g_pte, gpa & ~low,
                                     spa & ~low, gpa & ~((1 << gs) - 1), gs))
        return spa

    @staticmethod
    def _final_pbmt(wide: bool, vs_pte: int | None, g_pte: int | None) -> int:
        """14: a non-zero VS-stage PBMT overrides the G-stage's, which
        overrides the PMA."""
        if not wide:
            return 0
        for pte in (vs_pte, g_pte):
            if pte is not None and pte >> PBMT_SHIFT & 3:
                return pte >> PBMT_SHIFT & 3
        return 0

    def _gstage(self, m, gpa: int, access: str, kind: str, *, implicit: bool,
                hlvx: bool = False, mxr: int = 0, view: bool, gva_pte: bool = False
                ) -> tuple[int, int | None, int | None]:
        """The G-stage (21.5.1): hgatp's Sv32x4 / Sv39x4 / Sv48x4 / Sv57x4 —
        the address 2 bits wider, a 16 KB root table; every access a user
        one (U must be set); ``access`` the permission it needs (an
        implicit VS page-table read: R, no MXR; ``gva_pte``: Svadu's write
        of a VS-level PTE: W), ``kind`` the fault's kind.  Svpbmt / Svnapot
        / Svadu as the S-stage with menvcfg.  Returns (SPA, the leaf PTE,
        the page's shift) — (gpa, None, None) for Bare."""
        sch, _, root = self._hgatp(m)
        if sch is None:
            return gpa, None, None
        c = m.csrs
        env = c.menvcfg
        adue = "svadu" in m.extensions and bool(env & (1 << 61))
        cause = GUEST_FAULT[kind]
        wide = sch.pte_size == 8
        if gpa >> (sch.va_bits + 2):
            raise GuestFault(cause, gpa, implicit, f"GPA bits above {sch.va_bits + 1} set",
                             gva_pte)
        base = root << PAGE
        for level in range(sch.levels - 1, -1, -1):
            width = sch.idx_bits + (2 if level == sch.levels - 1 else 0)
            idx = (gpa >> (PAGE + level * sch.idx_bits)) & ((1 << width) - 1)
            addr = base + idx * sch.pte_size
            if addr >> m.pabits:
                raise PageFault(ACCESS_FAULT[kind], "a G-stage PTE beyond physical memory")
            pte = m.load_pte(addr, sch.pte_size, view)
            if pte is None:
                raise PageFault(ACCESS_FAULT[kind], "PMP refuses a G-stage PTE's address")
            if wide:
                self._bits_ok(m, pte, kind, env, GuestFault, cause, gpa, implicit)
            if not pte & V or (pte & (R | W)) == W:
                raise GuestFault(cause, gpa, implicit, "an invalid G-stage PTE"
                                 if not pte & V else "W without R in a G-stage PTE", gva_pte)
            ppn = (pte >> 10) & ((1 << 44) - 1 if wide else (1 << 22) - 1)
            if pte & (R | X):
                napot = self._napot_ok(pte, level, ppn, GuestFault, cause, gpa, implicit)
                self._gcheck(pte, access, gpa, hlvx=hlvx, mxr=mxr, implicit=implicit,
                             cause=cause, ad=not adue, write=gva_pte)
                if not napot and ppn & ((1 << (level * sch.idx_bits)) - 1):
                    raise GuestFault(cause, gpa, implicit, "a misaligned G-stage superpage",
                                     gva_pte)
                need = A | (D if access in DIRTYING else 0)
                if adue and pte & need != need:
                    pte |= need
                    if not view and not m.store_pte(addr, pte, sch.pte_size):
                        raise PageFault(ACCESS_FAULT[kind], "Svadu's G-stage PTE update "
                                        "refused (PMP / an I/O region)")
                shift = PAGE + level * sch.idx_bits + napot
                low = (1 << shift) - 1
                return ((ppn << PAGE) & ~low) | (gpa & low), pte, shift
            if pte & (D | A | U) or (wide and pte & (N | 3 << PBMT_SHIFT)):
                raise GuestFault(cause, gpa, implicit,
                                 "D / A / U / N / PBMT set in a non-leaf G-stage PTE", gva_pte)
            base = ppn << PAGE
        raise GuestFault(cause, gpa, implicit, "no G-stage leaf PTE at level 0", gva_pte)

    @staticmethod
    def _gcheck(pte: int, access: str, gpa: int, *, hlvx: bool, mxr: int, implicit: bool,
                cause: int | None = None, ad: bool = True, write: bool = False) -> None:
        """A G-stage leaf's permission (21.5.1): U set (a user access), X for
        a fetch or HLVX, W for a store, R (or X with the HS-level MXR, not
        for page-table reads) for a load; A, and D for a store (Svade; ``ad``
        False: Svadu sets them)."""
        cause = GUEST_FAULT[access] if cause is None else cause
        if not pte & U:
            raise GuestFault(cause, gpa, implicit, "a G-stage page without U", write)
        if access in ("ssload", "ssstore"):     # (22.2.3: read-write at the G-stage)
            ok = pte & R and pte & W
        else:
            ok = pte & X if access == "fetch" or hlvx else pte & W if access == "store" else \
                pte & R or (mxr and not implicit and pte & X)
        if not ok:
            need = "X" if access == "fetch" or hlvx else "R and W" if access.startswith("ss") \
                else "W" if access == "store" else "R"
            raise GuestFault(cause, gpa, implicit, f"no G-stage {need} permission", write)
        if ad and (not pte & A or (access in DIRTYING and not pte & D)):
            raise GuestFault(cause, gpa, implicit, "A = 0 in a G-stage PTE" if not pte & A
                             else "D = 0 in a G-stage PTE on a store", write)

    # -- the TLB -------------------------------------------------------------------------

    def _lookup(self, asid: int, vpn: int, sch: Scheme) -> TLBEntry | None:
        sizes = [(level, 0) for level in range(sch.levels)]
        if self._napot:
            sizes.insert(1, (0, 4))             # (Svnapot: 64 KB pages)
        for level, napot in sizes:
            key_vpn = vpn >> (level * sch.idx_bits + napot)
            for key in ((asid, key_vpn, level, napot), (-1, key_vpn, level, napot)):
                e = self.tlb.get(key)
                if e is not None:
                    self.tlb.move_to_end(key)
                    return e
        return None

    def _insert(self, e: TLBEntry) -> None:
        if e.napot:
            self._napot = True
        key = (-1 if e.pte & G else e.asid, e.vpn, e.level, e.napot)
        self.tlb[key] = e
        self.tlb.move_to_end(key)
        while len(self.tlb) > self.entries:
            self.tlb.popitem(last=False)

    def _glookup(self, vmid: int, vs_mode: int, asid: int, g_mode: int,
                 va: int) -> GuestEntry | None:
        for shift in self._gshifts:
            for a in (asid, -1):
                key = ("g", vmid, vs_mode, a, g_mode, shift, va >> shift)
                e = self.tlb.get(key)
                if e is not None:
                    self.tlb.move_to_end(key)
                    return e
        return None

    def _ginsert(self, e: GuestEntry) -> None:
        self._gshifts.add(e.shift)
        key = ("g", e.vmid, e.vs_mode, e.asid, e.g_mode, e.shift, e.vpn)
        self.tlb[key] = e
        self.tlb.move_to_end(key)
        while len(self.tlb) > self.entries:
            self.tlb.popitem(last=False)

    def sfence(self, m, va: int | None, asid: int | None) -> None:
        """SFENCE.VMA (12.2.1): ``va`` None = every address, ``asid`` None =
        every address space; an ASID-specific fence keeps global entries.
        With V = 1 (21.5.3): the current VM's VS-level translations
        (:meth:`hfence_vvma`); with V = 0 only the HS-level ones."""
        if m.v:
            self.hfence_vvma(m, va, asid)
            return
        sch = self._satp(m)[0] or SCHEMES[(m.xlen, 1 if m.xlen == 32 else 8)]
        mask = (1 << (sch.levels * sch.idx_bits)) - 1
        vpn = None if va is None else (va >> PAGE) & mask
        for key in list(self.tlb):
            e = self.tlb[key]
            if isinstance(e, GuestEntry):
                continue
            if asid is not None and (e.pte & G or e.asid != asid):
                continue
            if vpn is not None and vpn >> (e.level * sch.idx_bits + e.napot) != e.vpn:
                continue
            del self.tlb[key]

    def hfence_vvma(self, m, va: int | None, asid: int | None) -> None:
        """HFENCE.VVMA (21.3.2): the guest translations of the current
        hgatp.VMID — of guest virtual address ``va`` (None: all), of VS ASID
        ``asid`` (None: all; global pages stay)."""
        vmid = self._hgatp(m)[1]
        if asid is not None:
            asid &= 0x1FF if m.xlen == 32 else 0xFFFF
        for key in list(self.tlb):
            e = self.tlb[key]
            if not isinstance(e, GuestEntry) or e.vmid != vmid:
                continue
            if asid is not None and e.asid != asid:
                continue
            if va is not None and (va & m.amask) >> e.shift != e.vpn:
                continue
            del self.tlb[key]

    def hfence_gvma(self, m, gpa: int | None, vmid: int | None) -> None:
        """HFENCE.GVMA (21.3.2): the guest translations of VMID ``vmid``
        (None: every VM) whose G-stage page holds guest physical address
        ``gpa`` (None: all)."""
        if vmid is not None:
            vmid &= 0x7F if m.xlen == 32 else 0x3FFF
        for key in list(self.tlb):
            e = self.tlb[key]
            if not isinstance(e, GuestEntry):
                continue
            if vmid is not None and e.vmid != vmid:
                continue
            if gpa is not None and not e.g_base <= gpa < e.g_base + (1 << e.g_shift):
                continue
            del self.tlb[key]


# ---------------------------------------------------------------- the views

_FLAGS = "VRWXUGAD"


def _flags(pte: int | None) -> str:
    """V R W X U G A D, then with Svnapot / Svpbmt bits N and the PBMT."""
    if pte is None:
        return "-"
    out = "".join(c if pte >> k & 1 else "-" for k, c in enumerate(_FLAGS))
    if pte >> 61:
        out += (" N" if pte & N else "") + {0: "", 1: " NC", 2: " IO", 3: " PBMT=3"}[
            pte >> PBMT_SHIFT & 3]
    return out


def _size(shift: int) -> str:
    size = 1 << shift
    return f"{size >> 10} KB" if size < 1 << 20 else f"{size >> 20} MB" if size < 1 << 30 \
        else f"{size >> 30} GB"


def entry_rows(m) -> list[tuple[str, ...]]:
    """The TLB's entries, most recent first (the TLB pane, the debugger's
    ``hyp``): (stage — "HS" an HS-level entry, "guest" a two-stage one —,
    VMID, ASID, virtual page, page size, guest physical page, physical page,
    the leaf PTE's flags (VS-stage for a guest), the G-stage PTE's flags)."""
    mmu = m.mmu
    out = []
    for e in reversed(list(getattr(mmu, "tlb", {}).values())):
        if isinstance(e, GuestEntry):
            out.append(("guest", str(e.vmid), "G" if e.asid < 0 else str(e.asid),
                        f"0x{e.vpn << e.shift:x}", _size(e.shift), f"0x{e.gpa:x}",
                        f"0x{e.spa:x}", _flags(e.vs_pte), _flags(e.g_pte)))
        else:
            shift = PAGE + e.level * (9 if m.xlen == 64 else 10) + e.napot
            out.append(("HS" if m.csrs.has_h else "S", "-",
                        "G" if e.pte & G else str(e.asid), f"0x{e.vpn << shift:x}",
                        _size(shift), "-", f"0x{(e.ppn << PAGE) & ~((1 << shift) - 1):x}",
                        _flags(e.pte), "-"))
    return out


def guest_lines(m, va: int | None = None) -> list[str]:
    """The hypervisor's state (priv. ch. 21) for the debugger's ``hyp``: the
    mode and V, hstatus, hgatp, vsatp, the delegations, the guest entries
    of the TLB; with ``va`` how a guest access (VS-mode's) translates it —
    both stages, or which faulted."""
    c = m.csrs
    if not c.has_h:
        return ["this hart has no hypervisor extension (--rv-ext ...h)"]
    vs, asid, vroot = RVMMU._atp(m, c.vsatp)
    g, vmid, groot = RVMMU._hgatp(m)
    h = c.hstatus
    out = [f"{m.mode_name}-mode (V = {m.v}); hstatus 0x{h:x}: SPV {h >> 7 & 1}, SPVP "
           f"{h >> 8 & 1}, HU {h >> 9 & 1}, VTVM {h >> 20 & 1}, VTW {h >> 21 & 1}, VTSR "
           f"{h >> 22 & 1}, GVA {h >> 6 & 1}",
           f"hgatp: {g.name + 'x4' if g else 'Bare'}, VMID {vmid}, root table at PA "
           f"0x{groot << PAGE:x} (16 KB)" if g else f"hgatp: Bare, VMID {vmid}",
           f"vsatp: {vs.name if vs else 'Bare'}, ASID {asid}, root table at GPA "
           f"0x{vroot << PAGE:x}" if vs else f"vsatp: Bare, ASID {asid}",
           f"hedeleg 0x{c.hedeleg:x}, hideleg 0x{c.hideleg & c.mideleg:x}; htval "
           f"0x{c.htval:x} (GPA 0x{c.htval << 2:x}), htinst 0x{c.htinst:x}"]
    rows = [r for r in entry_rows(m) if r[0] == "guest"]
    out.append(f"guest entries in the TLB: {len(rows)}")
    for r in rows:
        out.append(f"  VMID {r[1]:>5}  ASID {r[2]:>5}  {r[3]} ({r[4]}) -> GPA {r[5]} -> PA "
                   f"{r[6]}  VS {r[7]}  G {r[8]}")
    if va is not None:
        priv = c.hstatus >> 8 & 1
        t = m.mmu._describe_guest(m, va, "load", priv)
        where = f"PA 0x{t.pa:x}" if t.pa is not None else f"{t.fault}"
        out.append(f"guest VA 0x{va:x} as {'VS' if priv else 'VU'}-mode (hstatus.SPVP): "
                   f"{where} — {t.how}")
    return out
