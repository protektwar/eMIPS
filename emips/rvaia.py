"""The RISC-V Advanced Interrupt Architecture on a hart (LF9; The RISC-V
Advanced Interrupt Architecture 1.0, chs. 2, 3, 5-7).

- :class:`AIA` — Smaia / Ssaia's state of one hart (``CSRFile.aia``):
  mvien / mvip (5.3), the S-level interrupt-enable bits of virtual
  interrupts, hvien / hvip's bits 13-63 / hvictl / hviprio1-2 (6.3), the
  machine- and supervisor-level **iprio arrays** (5.2.1 / 5.4.1;
  IPRIOLEN 8 — Spike keeps them read-only zero, eMIPS implements them),
  and the three "top interrupt" computations :meth:`AIA.mtopi`,
  :meth:`AIA.stopi`, :meth:`AIA.vstopi` (Tables 5.3 / 5.5, 6.3.3): the
  hart takes the top of M, then of HS, then of VS (6.3.4); WFI wakes on
  any of them (5.5).
- :data:`DEFAULT_ORDER` — Table 5.1's default priority order with 5.1's
  proposed order for the standard local interrupts 16-23 / 32-47; eMIPS
  puts the custom interrupts 24-31 / 48-63 above all (lowest number first:
  Spike's rule for every number above 13 — which would put 35 above MEI,
  where Table 5.1 makes it the lowest), 14 / 15 after 13, 0 / 4 / 8 last.
- :class:`IntFile` / :class:`IMSIC` — an IMSIC (ch. 3): per hart an M
  file, with S-mode an S file and with H ``GEILEN`` guest files (Spike's
  31), 2047 identities each (eip / eie arrays, eidelivery 0 / 1,
  eithreshold, topei / claim); the files drive MEIP / SEIP, hgeip and
  VSEIP.  :class:`IMSICPages` is the memory-mapped side (3.5 / 3.6):
  seteipnum_le / _be in each file's 4 KB page, Spike's layout — M files
  at 0x24000000 + 0x1000·hart, S + guest files at 0x28000000 + 0x40000·hart
  (S, G1 ... G31, then read-only zero pages).

Choices: ``sim.DEVIATIONS["RISC-V AIA"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .rvcsr import CSRFile

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
#: interrupts 13-63 (the local and custom ones: mvien / mvip, hvien / hvip, 5.3 / 6.3.2)
HIGH = M64 & ~((1 << 13) - 1)
SSI, STI, SEI, MEI, SGEI, LCOFI = 1, 5, 9, 11, 12, 13
VSSI, VSTI, VSEI = 2, 6, 10

#: the default priority order, highest first (see the module docstring)
DEFAULT_ORDER = (list(range(24, 32)) + list(range(48, 64))
                 + [47, 23, 46, 45, 22, 44, 43, 21, 42, 41, 20, 40,
                    11, 3, 7, 9, 1, 5, 12, 10, 2, 6, 13, 14, 15,
                    39, 19, 38, 37, 18, 36, 35, 17, 34, 33, 16, 32, 0, 4, 8])
RANK = {n: k for k, n in enumerate(DEFAULT_ORDER)}
#: a priority number below every other (Tables 5.3 / 5.5: "Lowest")
LOWEST = 1 << 16
#: an external interrupt whose controller gives no priority number (the
#: console's wire, a PLIC): as an IMSIC identity above 255 (eMIPS's choice)
NO_PRIORITY = 256

#: IMSIC: identities a file (Spike's 32 64-bit registers), the guest files
#: (Spike's GEILEN), the pages (Spike's platform.h)
IDENTITIES = 2047
GEILEN = 31
IMSIC_M_BASE, IMSIC_S_BASE = 0x24000000, 0x28000000
PAGE = 0x1000
S_STRIDE = 64 * PAGE
#: hvictl (6.3.2; IID 6 bits as Spike)
VTI, IID_MASK, DPR, IPRIOM = 1 << 30, 0x3F << 16, 1 << 9, 1 << 8
HVICTL_MASK = VTI | IID_MASK | DPR | IPRIOM | 0xFF
#: hviprio1 / 2's fields: (register, byte) -> interrupt (6.3.1; the reserved
#: bytes for 0, 4 and 8 read 0)
HVIPRIO = {(0, 1): 1, (0, 3): 5, (0, 5): 13, (0, 6): 14, (0, 7): 15,
           **{(1, k): 16 + k for k in range(8)}}


def best(cands: dict[int, int], ref: int) -> tuple[int, int] | None:
    """The highest-priority interrupt of ``cands`` (interrupt -> its
    priority number, 0 = none) relative to the level's external interrupt
    ``ref`` (Tables 5.3 / 5.5): (interrupt, the number it is ordered by)."""
    top = None
    for n, p in cands.items():
        if n == ref:
            band = p
        elif p == 0:
            band = 0 if RANK.get(n, 999) < RANK[ref] else LOWEST
        else:
            band = p
        key = (band, RANK.get(n, 999))
        if top is None or key < top[0]:
            top = (key, n)
    if top is None:
        return None
    return top[1], top[0][0]


def iprio_field(band: int) -> int:
    """mtopi / stopi / vstopi's IPRIO for an ordering number (5.2.2)."""
    if 1 <= band <= 255:
        return band
    return 0 if band == 0 else 255


# ---------------------------------------------------------------- the IMSIC

class IntFile:
    """One IMSIC interrupt file (3.1 / 3.8): the eip / eie arrays (bit i =
    identity i; bit 0 read-only 0), eidelivery, eithreshold."""

    def __init__(self, imsic: "IMSIC", level: str, guest: int = 0) -> None:
        self.imsic = imsic
        self.level = level                      # "m", "s", "vs"
        self.guest = guest
        self.eidelivery = 0
        self.eithreshold = 0
        self.eip = 0
        self.eie = 0
        self.valid = (1 << (IDENTITIES + 1)) - 2

    def topei(self) -> int:
        """The lowest pending-and-enabled identity below eithreshold (if
        not zero), or 0 (3.9; Spike's: a threshold at or below the lowest
        such identity gives 0)."""
        ints = self.eip & self.eie
        if not ints:
            return 0
        iid = (ints & -ints).bit_length() - 1
        thd = self.eithreshold
        return 0 if thd and iid >= thd else iid

    def signal(self) -> bool:
        """The file's interrupt signal (3.10)."""
        return self.eidelivery == 1 and self.topei() != 0

    def pend(self, i: int) -> None:
        """seteipnum / an MSI (3.5): ignored unless an implemented identity."""
        if 0 < i <= IDENTITIES:
            self.eip |= 1 << i
            self.imsic.changed()

    def claim(self) -> int:
        """A write of *topei (3.9): clear the current top's pending bit."""
        t = self.topei()
        if t:
            self.eip &= ~(1 << t)
            self.imsic.changed()
        return t

    def reg(self, sel: int, xlen: int) -> tuple[Callable[[], int], Callable[[int], None]] | None:
        """The register *iselect ``sel`` (0x70-0xFF) reaches with the
        current XLEN: (read, write), None if it does not exist (an odd
        eip / eie with XLEN 64); the reserved ones read 0 (3.7)."""
        if sel == 0x70:
            return (lambda: self.eidelivery), lambda v: self._set("eidelivery", v & 1)
        if sel == 0x72:
            mask = M32 if xlen == 32 else M64   # (WLRL: kept whole, as Spike)
            return (lambda: self.eithreshold), lambda v: self._set("eithreshold", v & mask)
        if sel < 0x80:
            return (lambda: 0), lambda v: None
        k = sel & 0x3F
        if xlen == 64 and k & 1:
            return None
        name = "eip" if sel < 0xC0 else "eie"
        width = 32 if xlen == 32 else 64
        shift = 32 * k
        mask = ((1 << width) - 1) << shift & self.valid

        def read() -> int:
            return (getattr(self, name) & mask) >> shift

        def write(v: int) -> None:
            self._set(name, (getattr(self, name) & ~mask) | ((v << shift) & mask))
        return read, write

    def _set(self, name: str, v: int) -> None:
        setattr(self, name, v)
        self.imsic.changed()

    def lines(self) -> list[str]:
        ids = [i for i in range(1, IDENTITIES + 1) if (self.eip | self.eie) >> i & 1]
        shown = ", ".join(f"{i}{'P' if self.eip >> i & 1 else ''}{'E' if self.eie >> i & 1 else ''}"
                          for i in ids[:24]) + (" ..." if len(ids) > 24 else "")
        return [f"eidelivery {self.eidelivery}, eithreshold {self.eithreshold}, topei "
                f"{self.topei()}; identities (P pending, E enabled): {shown or 'none'}"]


class IMSIC:
    """A hart's IMSIC (ch. 3): its interrupt files and what they signal."""

    def __init__(self, m, hart: int) -> None:
        self.m = m
        self.hart = hart
        c = m.csrs
        self.mfile = IntFile(self, "m")
        self.sfile = IntFile(self, "s") if c.has_s else None
        self.guests = {g: IntFile(self, "vs", g) for g in range(1, GEILEN + 1)} \
            if c.has_h else {}
        #: hgeip (21.2.4): bit g = guest file g's signal
        self.hgeip = 0

    def file(self, level: str) -> IntFile | None:
        if level == "m":
            return self.mfile
        if level == "s":
            return self.sfile
        g = self.m.csrs.vgein()
        return self.guests.get(g)

    def changed(self) -> None:
        """A file changed: the hart's MEIP / SEIP lines and hgeip (3.10)."""
        m = self.m
        m.set_interrupt(MEI, self.mfile.signal())
        if self.sfile is not None:
            m.set_interrupt(SEI, self.sfile.signal())
        self.hgeip = sum(1 << g for g, f in self.guests.items() if f.signal())

    def lines(self) -> list[str]:
        out = [f"IMSIC of hart {self.hart}: M file at 0x{IMSIC_M_BASE + PAGE * self.hart:x}"]
        out += [f"  M: {s}" for s in self.mfile.lines()]
        if self.sfile is not None:
            out.append(f"  S file at 0x{IMSIC_S_BASE + S_STRIDE * self.hart:x}, guest files "
                       f"after it (GEILEN {len(self.guests)})")
            out += [f"  S: {s}" for s in self.sfile.lines()]
        for g, f in self.guests.items():
            if f.eip or f.eie or f.eidelivery:
                out += [f"  guest {g}: {s}" for s in f.lines()]
        if self.guests:
            out.append(f"  hgeip 0x{self.hgeip:x}")
        return out


class IMSICPages:
    """The IMSICs' memory-mapped pages (3.5 / 3.6) as one platform device:
    a naturally aligned 32-bit write of identity i to seteipnum_le (offset
    0; _be at 4: byte-swapped) pends i; reads return 0; other pages of the
    regions read 0 and ignore writes.  Accesses of other sizes are an
    access fault (``RVMachine._access``: Spike's)."""

    def __init__(self, imsics: list[IMSIC]) -> None:
        self.imsics = imsics
        n = len(imsics)
        self.m_end = IMSIC_M_BASE + PAGE * n
        self.s_end = IMSIC_S_BASE + S_STRIDE * n
        self.base, self.size = IMSIC_M_BASE, self.s_end - IMSIC_M_BASE
        self.regions = [(IMSIC_M_BASE, self.m_end), (IMSIC_S_BASE, self.s_end)]

    def covers(self, pa: int) -> bool:
        return IMSIC_M_BASE <= pa < self.m_end or IMSIC_S_BASE <= pa < self.s_end

    def tick(self, m, cycles: int) -> None:
        pass

    def target(self, pa: int) -> IntFile | None:
        """The interrupt file whose page holds ``pa`` (None: a zero page)."""
        if IMSIC_M_BASE <= pa < self.m_end:
            return self.imsics[(pa - IMSIC_M_BASE) // PAGE].mfile
        if IMSIC_S_BASE <= pa < self.s_end:
            h, page = divmod((pa - IMSIC_S_BASE) // PAGE, S_STRIDE // PAGE)
            im = self.imsics[h]
            return im.sfile if page == 0 else im.guests.get(page)
        return None

    def load(self, m, pa: int, size: int) -> int:
        return 0

    def store(self, m, pa: int, value: int, size: int) -> None:
        if size != 4 or pa & 3:
            return
        f = self.target(pa)
        off = pa & (PAGE - 1)
        if f is None or off > 4:
            return
        if off == 4:                            # seteipnum_be
            value = int.from_bytes((value & M32).to_bytes(4, "little"), "big")
        f.pend(value & M32)


# ---------------------------------------------------------------- the hart's CSRs

class AIA:
    """Smaia / Ssaia's state of one hart (see the module docstring)."""

    def __init__(self, c: "CSRFile", smaia: bool) -> None:
        self.c = c
        self.m = c.m
        self.smaia = smaia
        #: mvien (5.3): bits 1, 9 and 13-63 writable
        self.mvien = 0
        self.mvien_mask = ((1 << SSI) | (1 << SEI) | HIGH) if c.has_s else 0
        #: mvip's own bits: bit 1 while mvien.SSIP (5.3), 13-63; bit 9 is mip's
        #: software SEIP (one cell: 5.3), bit 5 mip.STIP
        self.mvip_ssip = 0
        self.mvip_hi = 0
        #: sie's bits for interrupts mvien makes virtual (5.3, Table 5.4)
        self.sie_v = 0
        #: hvien / hvip 13-63 / vsie's own bits (6.3.2, Table 6.1)
        self.hvien = 0
        self.hvip_hi = 0
        self.vsie_v = 0
        self.hvictl = 0
        self.hviprio = [0, 0]
        #: the iprio arrays (5.2.1 / 5.4.1): one priority number an interrupt
        self.iprio_m = [0] * 64
        self.iprio_s = [0] * 64

    # -- which iprio bytes exist ---------------------------------------------------

    def iprio_writable(self, level: str) -> int:
        """The interrupts whose iprio byte is writable: those whose mie /
        sie bit can be one, but not the level's external interrupt (5.2.1
        / 5.4.1) — nor, at machine level, those the hypervisor extension
        always delegates (5.2.1's note)."""
        c = self.c
        from .rvcsr import HS_INTERRUPTS
        if level == "m":
            return c.mie_mask() & ~(1 << MEI) & ~HS_INTERRUPTS
        return (((1 << SSI) | (1 << STI)) | (c.mip_extra & c.mie_mask())
                | (self.mvien_mask & HIGH)) & ~(1 << SEI)

    def iprio_reg(self, level: str, sel: int, xlen: int):
        """iprio register ``sel`` (0x30-0x3F) of ``level``: (read, write),
        None if it does not exist (odd with XLEN 64)."""
        k = sel - 0x30
        if xlen == 64 and k & 1:
            return None
        arr = self.iprio_m if level == "m" else self.iprio_s
        per = 4 if xlen == 32 else 8
        first = 4 * k
        wr = self.iprio_writable(level)

        def read() -> int:
            return sum(arr[first + j] << (8 * j) for j in range(per) if first + j < 64)

        def write(v: int) -> None:
            for j in range(per):
                n = first + j
                if n < 64 and wr >> n & 1:
                    arr[n] = v >> (8 * j) & 0xFF
        return read, write

    # -- mvip / sip / sie (5.3) ----------------------------------------------------

    def mvip_read(self) -> int:
        c = self.c
        from .rvcsr import STCE
        v = c.mip_sw & (1 << SEI)
        if not (c.sstc and c.menvcfg & STCE):
            v |= c.mip_sw & (1 << STI)
        v |= self.mvip_ssip << SSI if self.mvien >> SSI & 1 else c.mip_sw & (1 << SSI)
        return v | self.mvip_hi

    def mvip_write(self, v: int, mask: int = M64) -> None:
        c = self.c
        from .rvcsr import STCE
        mk = 0
        if not (c.sstc and c.menvcfg & STCE):
            mk |= (1 << STI)
        if not self.mvien >> SSI & 1:
            mk |= (1 << SSI)
        elif mask >> SSI & 1:
            self.mvip_ssip = v >> SSI & 1
        mk |= (1 << SEI)
        mk &= mask
        c.mip_sw = (c.mip_sw & ~mk) | (v & mk)
        hm = HIGH & self.mvien_mask & mask
        self.mvip_hi = (self.mvip_hi & ~hm) | (v & hm)

    def virtual_s(self) -> int:
        """The sip / sie bits mvien makes virtual (Table 5.4: mideleg 0,
        mvien 1)."""
        return ~self.c.mideleg & self.mvien & M64

    def sip(self, real: int) -> int:
        vm = self.virtual_s()
        return (real & ~vm) | (self.mvip_read() & vm)

    def sie(self, real: int) -> int:
        vm = self.virtual_s()
        return (real & ~vm) | (self.sie_v & vm)

    def sip_write(self, v: int, mask: int = M64) -> int:
        """The virtual part of a sip write (to mvip: bit 1 and 13-63 —
        sip.SEIP stays read-only); returns the bits left for the real sip."""
        vm = self.virtual_s() & mask
        self.mvip_write(v, vm & ((1 << SSI) | HIGH))
        return vm

    def sie_write(self, v: int, mask: int = M64) -> int:
        vm = self.virtual_s() & mask
        self.sie_v = (self.sie_v & ~vm) | (v & vm)
        return vm

    # -- vsip / vsie bits 13-63 (6.3.2, Table 6.1; hideleg's 13-63 read 0) ---------

    def vsip_hi(self) -> int:
        return self.hvip_hi & self.hvien

    def vsie_hi(self) -> int:
        return self.vsie_v & self.hvien

    # -- the top interrupts ----------------------------------------------------------

    def ext_priority(self, level: str) -> int:
        """The priority number the external interrupt controller gives the
        level's external interrupt (Tables 5.3 / 5.5): an IMSIC file's
        top identity, the APLIC's topi priority (``m.ext_priority``), else
        :data:`NO_PRIORITY`."""
        m = self.m
        im = getattr(m, "imsic", None)
        if im is not None:
            f = im.file(level)
            if f is not None and f.signal():
                return f.topei()
        hook = getattr(m, "ext_priority", None)
        if hook is not None:
            p = hook(level)
            if p:
                return p
        return NO_PRIORITY

    def mtopi_pair(self) -> tuple[int, int] | None:
        c = self.c
        pend = c.mip() & c.mie & ~c.mideleg & M64
        if not pend:
            return None
        cands = {n: (self.ext_priority("m") if n == MEI else self.iprio_m[n])
                 for n in range(64) if pend >> n & 1}
        return best(cands, MEI)

    def mtopi(self) -> int:
        t = self.mtopi_pair()
        return 0 if t is None else t[0] << 16 | iprio_field(t[1])

    def s_pending(self) -> int:
        """stopi's candidates (5.4.2): sip & sie, hip & hie, not delegated
        to VS level."""
        c = self.c
        from .rvcsr import HS_INTERRUPTS
        real = c.mip() & c.mideleg
        sp = self.sip(real & c.s_interrupts) & self.sie(c.mie & c.mideleg & c.s_interrupts)
        hp = real & c.mie & HS_INTERRUPTS
        return (sp | hp) & ~(c.hideleg & c.mideleg) & M64

    def stopi_pair(self) -> tuple[int, int] | None:
        pend = self.s_pending()
        if not pend:
            return None
        cands = {n: (self.ext_priority("s") if n == SEI else self.iprio_s[n])
                 for n in range(64) if pend >> n & 1}
        return best(cands, SEI)

    def stopi(self) -> int:
        t = self.stopi_pair()
        return 0 if t is None else t[0] << 16 | iprio_field(t[1])

    def _vs_prio(self, n: int) -> int:
        for (r, b), irq in HVIPRIO.items():
            if irq == n:
                return self.hviprio[r] >> (8 * b) & 0xFF
        return 0

    def vstopi_pair(self) -> tuple[int, int] | None:
        """vstopi's highest-priority candidate (6.3.3): (identity at VS
        level, the number it is ordered by).  The candidates: a virtual SEI
        (its number from the guest file, hvictl, or 256), then either the
        other pending-and-enabled bits of vsip / vsie (hviprio's numbers;
        VTI = 0) or hvictl's interrupt (VTI = 1, IID != 9: ordered against
        SEI by DPR alone)."""
        c = self.c
        h = c.mip() & c.mie & c.hideleg & c.mideleg
        std = (h >> 1) & ((1 << SSI) | (1 << STI) | (1 << SEI))   # (VSSI -> SSI ...)
        pend = std | (self.vsip_hi() & self.vsie_hi())
        hv = self.hvictl
        vti, iid, iprio = hv & VTI, (hv & IID_MASK) >> 16, hv & 0xFF
        cands = []                              # (band, rank, identity)
        sei_rank = RANK[SEI]
        if pend >> SEI & 1:
            g = c.vgein()
            im = getattr(self.m, "imsic", None)
            f = im.guests.get(g) if im is not None else None
            if f is not None and f.topei():
                p = f.topei()
            elif g == 0 and iid == SEI and iprio:
                p = iprio
            else:
                p = NO_PRIORITY
            cands.append((p, sei_rank, SEI))
        if not vti:
            for n in range(64):
                if pend >> n & 1 and n != SEI:
                    p = self._vs_prio(n)
                    r = RANK.get(n, 999)
                    cands.append((p if p else (0 if r < sei_rank else LOWEST), r, n))
        elif iid != SEI:
            below = bool(hv & DPR)
            band = iprio if iprio else (LOWEST if below else 0)
            cands.append((band, sei_rank + (0.5 if below else -0.5), iid))
        if not cands:
            return None
        band, _r, n = min(cands)
        return n, band

    def vstopi(self) -> int:
        t = self.vstopi_pair()
        if t is None:
            return 0
        prio = iprio_field(t[1]) if self.hvictl & IPRIOM else 1
        return t[0] << 16 | prio

    def lines(self) -> list[str]:
        """The AIA's state (the Interrupts tab, the debugger's ``aia``)."""
        c = self.c
        out = []
        if self.smaia:
            t = self.mtopi()
            out.append(f"mtopi 0x{t:x}" + (f": interrupt {t >> 16}, IPRIO {t & 0xFF}" if t
                                           else ": none pending and enabled"))
        if c.has_s:
            t = self.stopi()
            out.append(f"stopi 0x{t:x}" + (f": interrupt {t >> 16}, IPRIO {t & 0xFF}" if t
                                           else ": none pending and enabled"))
            out.append(f"mvien 0x{self.mvien:x}, mvip 0x{self.mvip_read():x}")
        if c.has_h:
            t = self.vstopi()
            out.append(f"vstopi 0x{t:x}" + (f": interrupt {t >> 16}" if t else ": none")
                       + f"; hvien 0x{self.hvien:x}, hvictl 0x{self.hvictl:x}, hviprio1 "
                       f"0x{self.hviprio[0]:x}, hviprio2 0x{self.hviprio[1]:x}, VGEIN "
                       f"{c.vgein()}")
        for lvl, arr in (("machine", self.iprio_m), ("supervisor", self.iprio_s)):
            set_ = [f"{n}: {p}" for n, p in enumerate(arr) if p]
            if set_:
                out.append(f"{lvl}-level iprio: " + ", ".join(set_))
        im = getattr(self.m, "imsic", None)
        if im is not None:
            out += im.lines()
        return out
