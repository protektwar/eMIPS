"""The hardware page table walker (PLAN M9 phase 7).

Reference: *MIPS32 Architecture For Programmers Volume III* (MD00090,
Rev. 6.02) 4.12 "Hardware Page Table Walker" (the ``PageTableWalkRefill``
pseudo code), 9.18-9.20 / 9.22 (PWBase, PWField, PWSize, PWCtl), Tables
9.34 (pointer / PTE sizes), 9.37 / 9.38 (huge pages).

With ``Config3.PW = 1`` and ``PWCtl.PWEn = 1``, a TLB miss that would be a
TLB Refill first walks the page table in memory: up to three directory
levels (Global, Upper, Middle; PWSize.GDW / UDW / MDW index bits taken
from the faulting address at PWField.GDI / UDI / MDI) and the page table
(PWSize.PTW bits at PWField.PTI).  Each directory entry is a 32-bit
pointer; the page table holds the pair of PTEs, each shifted right by
PWField.PTEI - 2 (dropping the software bits) and rotated right by 2
(RI / XI into bits 31:30), so they become EntryLo0 / EntryLo1, written
into the TLB like a TLBWR.  If any read fails — an unmapped pointer, an
address no TLB entry maps, an address error — the walk stops silently and
the TLB Refill exception is taken as without a walker.  The walker never
changes the CP0 registers a refill handler uses (BadVAddr, EntryHi,
PageMask, EntryLo0 / 1, Context).

Huge pages (PWCtl.HugePg): a directory entry whose PTEvld bit
(PWCtl.Psn) is set is a PTE; a 2 x power-of-4 region gives two adjacent
pages from one entry, a power-of-4 region needs PWCtl.DPH and reads the
neighbouring entry for the other page (no PTEvld there: Machine Check,
PageGrain.MCCause = 5).

eMIPS choices (``sim.DEVIATIONS``): the walker runs only for a miss with
Status.EXL = 0 (a refill); it reads mapped tables when a valid TLB entry
maps them (no nested refill); PTEW 0 / 1 (4- or 8-byte PTEs) are
implemented, other widths are written as 0; with 8-byte PTEs and huge
pages the directory entries are 8 bytes too (Table 9.34); pages larger
than 256 MB or smaller than 4 KB abort the walk.

MIPS64 (Vol. III MIPS64 4.14, 9.16-9.20; the P6600's choices, MD01138
2.2.2.12-15 / 3.7): PWBase / PWField / PWSize are 64 bits; PWSize.PS
picks the pointer size — 0: 32-bit pointers (compatibility addresses,
sign-extended), only refills that would use the TLB Refill vector; 1:
64-bit pointers and PTEs, only XTLB Refills, and PWCtl.XK / XS / XU say
which of xuseg / xsseg / xkseg the walker serves, VA 63:62 joining the
Global Directory index (Table 9.26).  No Base Directory (PWCtl.PWDirExt
reads 0, PWField.BDI / PWSize.BDW read 0: the P6600's); PTEW = 1 only with
PS = 0 (setting PS clears it, the P6600's rule).  The PTE's 64-bit rotate
puts RI / XI at 63:62 — what the TLB keeps at 31:30 (``sim64.lo64``).
"""

from __future__ import annotations

from . import cp0 as C

M32 = 0xFFFFFFFF
PWBASE, PWFIELD, PWSIZE, PWCTL = (5, 5), (5, 6), (5, 7), (6, 6)
REGISTERS = frozenset({PWBASE, PWFIELD, PWSIZE, PWCTL})
#: Config3.PW
CFG3_PW = 1 << 24
PWCTL_EN, PWCTL_DPH, PWCTL_HUGEPG = 1 << 31, 1 << 7, 1 << 6
#: MIPS64: PWCtl.XK / XS / XU (28 / 27 / 26), PWSize.PS (30)
PWCTL_X = 7 << 26
PWSIZE_PS = 1 << 30
M64 = (1 << 64) - 1


class _Abort(Exception):
    """A walk read failed: give up, take the TLB Refill exception."""


def reset(m, release: str, p5600: bool = False) -> None:
    """Reset values: PWField GDI..PTI = 0 (Release 6 and the P5600 —
    MD01025 Tables 2.33 / 2.34: 0x0C30C302 —: 12), PTEI = 2; PWSize PTW = 0
    (Release 6, the P5600: 1); PWBase / PWCtl = 0."""
    r6 = release == "R6" or p5600
    m.cp0[PWBASE]["value"] = 0
    m.cp0[PWFIELD]["value"] = (12 << 24 | 12 << 18 | 12 << 12 | 12 << 6 | 2) if r6 else 2
    m.cp0[PWSIZE]["value"] = (1 << 6) if r6 else 0
    m.cp0[PWCTL]["value"] = 0


def write(m, key: tuple[int, int], value: int, release: str) -> int:
    """The value MTC0 leaves in a walker register (Vol. III 9.18-9.22;
    MIPS64: see the module docstring)."""
    r6 = release == "R6"
    wide = getattr(m.mmu, "wide", False)
    old = m.cp0[key]["value"]
    if key == PWBASE:
        return value & (M64 if wide else M32)
    if key == PWFIELD:
        new = value & 0x3FFFFFFF
        if r6 and any((new >> s) & 0x3F < 12 for s in (24, 18, 12, 6)):
            return old                       # R6: the whole write is dropped
        if (new & 0x3F) < 2:
            if r6:                           # R6: PTEI 0 / 1 not supported
                new = (new & ~0x3F) | (old & 0x3F)
        return new
    if key == PWSIZE:
        new = value & 0x7FFFFFFF & ~(0 if wide else PWSIZE_PS)  # PS: MIPS64 only
        if r6 and (new >> 6) & 0x3F == 0:
            return old                       # R6: PTW = 0 drops the write
        if (new & 0x3F) > 1 or (new & PWSIZE_PS and new & 0x3F):
            new &= ~0x3F                     # PTEW: 0 and 1 (1 only with PS = 0)
        return new
    if key == PWCTL:
        return value & (PWCTL_EN | PWCTL_DPH | PWCTL_HUGEPG | 0x3F
                        | (PWCTL_X if wide else 0))
    return old


def _rotate(t: int, ptei: int) -> int:
    """A PTE as EntryLo: shifted right by PTEI - 2, rotated right by 2."""
    t >>= max(ptei - 2, 0)
    t &= M32
    return ((t >> 2) | ((t & 3) << 30)) & M32


def _hi(mmu, va: int) -> int:
    """The EntryHi a walk writes: the faulting address's R / VPN2 (and bit
    12, for the 1 KB-page layout of the TLB entry)."""
    return va & (mmu._hi_vpn | 0x1000)


def _mask(page: int) -> int:
    """PageMask of a page size, or _Abort when the TLB has no such page."""
    if page < 4096 or page > (256 << 20) or page & (page - 1):
        raise _Abort
    return ((page - 1) >> 12) << 13


def walk(mmu, m, va: int) -> bool:
    """Try to refill the TLB for ``va`` from the page table (True when an
    entry was written)."""
    cp = m.cp0
    ctl = cp[PWCTL]["value"]
    if not ctl & PWCTL_EN:
        return False
    size, fld = cp[PWSIZE]["value"], cp[PWFIELD]["value"]
    gdw, udw, mdw = (size >> 24) & 0x3F, (size >> 18) & 0x3F, (size >> 12) & 0x3F
    ptw, ptew = (size >> 6) & 0x3F, size & 0x3F
    if not (gdw or udw or mdw):
        return False                         # no structure to walk
    gdi, udi, mdi = (fld >> 24) & 0x3F, (fld >> 18) & 0x3F, (fld >> 12) & 0x3F
    pti, ptei = (fld >> 6) & 0x3F, fld & 0x3F
    huge = bool(ctl & PWCTL_HUGEPG)
    psn = ctl & 0x3F
    wide = mmu.wide
    ps = wide and bool(size & PWSIZE_PS)
    prefix, pbits = 0, 0                     # VA 63:62 above the Global index
    if wide:
        if mmu._xtlb(m, va) != ps:           # PS: 32-bit regions or 64-bit, not both
            return False
        if ps:                               # PWCtl.XK / XS / XU (Table 9.26)
            sel, region = (ctl >> 26) & 7, va >> 62
            if sel == 0b001 and region:
                return False                 # xuseg only
            if sel == 0b011 and region == 3:
                return False                 # xuseg and xsseg
            if sel == 0b101 and region == 1:
                return False                 # xuseg and xkseg
            if sel not in (0b001, 0b011, 0b101, 0b111):
                return False                 # no segment (or a reserved setting)
            pbits = {0b001: 0, 0b011: 1, 0b101: 1, 0b111: 2}[sel]
            prefix = {0b011: region & 1, 0b101: region >> 1, 0b111: region}.get(sel, 0)
    nshift = 3 if ps else 2                  # the native pointer: 8 or 4 bytes
    dshift = nshift + (1 if huge and ptew == 1 else 0)
    pmask = M64 if wide else M32
    asid = cp[(10, 0)]["value"] & 0xFF

    def pointer(t: int) -> int:
        """A directory entry as the next table's address: 64-bit with PS,
        else a 32-bit (on MIPS64: compatibility, sign-extended) one."""
        if ps:
            return t & M64
        t &= M32
        return t | 0xFFFFFFFF00000000 if wide and t & 0x80000000 else t

    def read(addr: int, nbytes: int) -> int:
        addr &= pmask
        hit = mmu.walker_translation(m, addr)
        if hit is None:
            raise _Abort
        pa, cca = hit
        if nbytes == 8:                      # 8-byte entries: the whole value
            hi_lo = [m.load_pa(addr + 4 * i, [pa + 4 * i + k for k in range(4)], cca, 4)
                     for i in (0, 1)]
            return (hi_lo[0] << 32 | hi_lo[1]) if m.big_endian else (hi_lo[1] << 32 | hi_lo[0])
        return m.load_pa(addr, [pa + k for k in range(nbytes)], cca, nbytes)

    try:
        ptr = cp[PWBASE]["value"]
        for level, (width, lsb) in enumerate(((gdw, gdi), (udw, udi), (mdw, mdi))):
            if not width:
                continue
            index = (va >> lsb) & ((1 << width) - 1)
            if level == 0 and pbits:          # (MIPS64, PS = 1: VA 63:62 prepended)
                index |= prefix << width
            entry = (ptr | (index << dshift)) & pmask
            t = read(entry, 1 << dshift)
            if huge and t >> psn & 1:        # a huge page PTE in a directory
                lo = _rotate(t, ptei)
                if lsb & 1:                  # 2 x power of 4: adjacent pages
                    page = 1 << (lsb - 1)
                    bit = (1 << (lsb - 1)) >> 6
                    lo0, lo1 = lo & ~bit, lo | bit
                elif ctl & PWCTL_DPH:        # power of 4: the neighbour entry
                    page = 1 << lsb
                    t2 = read(entry ^ (1 << dshift), 1 << dshift)
                    if not t2 >> psn & 1:
                        from .sim import _Arch
                        raise _Arch(C.MCHECK, info={"mccause": 5, "why": "page walk: the "
                                                    "second huge-page PTE has no PTEvld"})
                    other = _rotate(t2, ptei)
                    lo0, lo1 = (other, lo) if (va >> lsb) & 1 else (lo, other)
                else:
                    raise _Abort
                mmu.write_random(m, _mask(page), _hi(mmu, va) | asid, lo0, lo1,
                                 how="page walk (huge page)")
                return True
            ptr = pointer(t)
        index = (va >> pti) & ((1 << ptw) - 1)
        off0 = (index >> 1) << (nshift + ptew + 1)
        off1 = off0 | (1 << (nshift + ptew))
        nbytes = (1 << nshift) << ptew
        lo0 = _rotate(read(ptr | off0, nbytes), ptei)
        lo1 = _rotate(read(ptr | off1, nbytes), ptei)
        page = 1 << (pti if pti % 2 == 0 else pti - 1)   # down to a power of 4
        mmu.write_random(m, _mask(page), _hi(mmu, va) | asid, lo0, lo1, how="page walk")
        return True
    except _Abort:
        return False
