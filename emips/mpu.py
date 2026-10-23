"""The I7200's Memory Protection Unit (PLAN M23 phase 5d).

Reference: *MIPS I7200 Programmer's Guide* MD01232 rev. 01.20 chapter 4
(4.1 the default segments, Table 14; 4.2 / 4.6 the regions, Tables 20-23,
Figures 20-24 and the four example programs; 4.3 the CDMM, Figure 19,
Table 15; 4.4 MPU_ACSR / MPU_Config, Tables 16 / 17; 4.5 the segment
control registers, Tables 18 / 19; 4.7 Table 24, the TLB / FMT / MPU
builds), 6.4 Table 40 (MPUL / MPUS: ExcCode 29, at the priority of the
TLB refills); the datasheet MD01227 2.5.6 and Table 2 (8-32 regions by
4, 128 KB L1s with the MPU); the nanoMIPS PRA MD01251 ch. 5 / 9.44 (the
CDMM, CDMMBase, the ACSR's access rules).

An MPU does not translate: **PA = VA** over the whole 4 GB, for every
mode — no kernel / user address spaces, no supervisor mode.  Each 256 MB
*segment* (VA 31:28) has a cacheability attribute and three inhibits::

    RI  a data read  -> MPUL      WI  a data write -> MPUS
    XI  a fetch      -> MPUL      (CCA: the access's attribute)

*Regions* (8-32, as built) overlay that: a region is 16 equal
*subregions* of 2^Size bytes (Size 5-28: 32 bytes-256 MB), aligned to 16
subregions; BaseAddress names the first enabled subregion, Count how
many more follow (not past the region's last).  Where regions overlap,
the higher-numbered one wins; where none is enabled, the segment's
attributes hold.

The registers are memory mapped in the **Common Device Memory Map**
(CDMM): CDMMBase (CP0 15.2: PA 31:15, EN) places it; the Fast Debug
Channel's three 64-byte device register blocks come first, the MPU's at
offset 64 x 3::

    0x00 MPU_ACSR   DevType 1, DevSize, DevRev 0, Uw, Ur
    0x08 MPU_Config En, ExcR / ExcW / ExcX, Exc_Reg_Match, Exc_Reg_Num, NumRegions
    0x10 + 4 n      SegmentControl n: segments 4n..4n+3, a byte each (RI WI XI CCA)
    0x20 + 8 r      region r BaseAddress (31:5)
    0x24 + 8 r      region r Control: EN, Size, Count, RI, WI, XI, CCA

Table 24's builds: ``mpu[:N]`` (segments enabled permanently, N regions;
Config.K0 / KU / K23 read 0, Config5.CV 1) and ``fixed+mpu[:N]`` (the FMT
build: the fixed mapping and its Config fields, the regions overlaying
it; MPU_Config.En = 0).  With MT each VPE has its own MPU (the CDMM's
devices are per VPE, CDMMBase is shared: 4.3, MD01251 9.44).

eMIPS choices (``sim.DEVIATIONS["nanoMIPS MPU"]``): Config.MT reads 3
(Table 24: "the core operates in FMT mapping mode"; the PRA has no MPU
encoding); an MPUL / MPUS writes BadVAddr; a PC-relative load is a data
read (RI); CACHE and PREF are not checked (as the TLB's RI / XI); a Size
outside 5-28 makes the region match nothing; the FDC's blocks read 0
(not modelled); the reset segments are uncached with no inhibits; the
boot firmware places the CDMM at 0xBFC10000 (MPU) / PA 0x1FC10000 (FMT:
reached at 0xBFC10000 through kseg1) and makes the segments of kuseg /
kseg0 cacheable.
"""

from __future__ import annotations

from . import cp0 as C
from .mmu import CCA_UNCACHED, FixedMappingMMU, M32, Translation, segment

#: CDMMBase (CP0 15.2, MD01232 Table 15): PA 31:15 in 27:11, EN, CDMMSize
CDMMBASE = (15, 2)
CDMM_EN = 1 << 10
CDMM_ADDR = 0x0FFFF800
#: Config3.CDMM (MD01251 Table 9.67)
CONFIG3_CDMM = 1 << 3
#: Config5.CV (an MPU build: 1, read-only; Table 24)
CONFIG5 = (16, 5)
CFG5_CV = 1 << 29
#: the Cause code of MPUL / MPUS (MD01232 Table 40)
MPU_EXC = 29
#: a device register block, the FDC's blocks before the MPU's (Figure 19)
DRB = 64
FDC_DRBS = 3
MPU_OFFSET = DRB * FDC_DRBS
#: MPU registers (offsets in its block)
ACSR, CONFIG, SEGCTL, REGION = 0x00, 0x08, 0x10, 0x20
DEVTYPE = 0x01
ACSR_UW, ACSR_UR = 1 << 3, 1 << 2
CFG_EN, CFG_EXCR, CFG_EXCW, CFG_EXCX, CFG_MATCH = 1 << 31, 1 << 19, 1 << 18, 1 << 17, 1 << 16
#: an attribute byte (segment and region control, bits 5:0)
RI, WI, XI = 1 << 5, 1 << 4, 1 << 3
REG_EN = 1 << 15
#: the region counts the I7200 is built with (Table 23)
I7200_REGIONS = (8, 12, 16, 20, 24, 28, 32)
#: the boot firmware's CDMM (the guide's 0x1FC10000; with the MPU, VA = PA:
#: the same address kseg1 shows it at with the fixed mapping)
BOOT_CDMM = {True: 0xBFC10000, False: 0x1FC10000}


def flags(bits: int) -> str:
    return "".join(f for f, b in (("RI", RI), ("WI", WI), ("XI", XI)) if bits & b) or "-"


class MPU:
    """The I7200's MPU: ``segments`` = the MPU build (PA = VA, segment
    attributes), else the FMT build with MPU regions; ``regions`` 1-32."""

    mt = 3
    supervisor = False
    sc = False

    def __init__(self, regions: int = 16, *, segments: bool = True) -> None:
        if not 1 <= regions <= 32:
            raise ValueError("an MPU has 1-32 regions (MPU_Config.NumRegions)")
        self.n = regions
        self.segments = segments
        self.name = "MPU" if segments else "fixed mapping + MPU regions"
        self.fixed = FixedMappingMMU()
        #: segment attribute bytes (RI WI XI CCA), the regions' registers
        self.seg = [CCA_UNCACHED] * 16
        self.base = [0] * regions
        self.ctl = [0] * regions
        self.acsr = 0
        #: MPU_Config's exception fields (23:8)
        self.exc = 0
        #: enabled regions, highest number first: (lo, hi, bits, r)
        self.active: list[tuple[int, int, int, int]] = []
        self.drbs = -(-(REGION + 8 * regions) // DRB)
        self.size = MPU_OFFSET + self.drbs * DRB
        self.device = CDMM(self)
        self.REGISTERS = frozenset({CDMMBASE})
        self.address_ok = _anything if segments else None
        from .tlb import TLBStats
        self.stats = TLBStats()
        self.m = None

    # -- the Machine's side ------------------------------------------------------------

    def attach(self, m) -> None:
        if not getattr(m, "nano", False):
            raise ValueError("the MPU is the I7200's (MD01232 ch. 4): a nanoMIPS core "
                             "(--arch nanomips)")
        self.m = m
        cp = m.cp0
        k = (7 << 28) | (7 << 25) | 7
        config = cp[C.CONFIG]["value"] & ~(7 << 7)
        if self.segments:                       # K23 / KU / K0: RO 0 (Table 24)
            config &= ~k
            m._wmask[C.CONFIG] = m._wmask.get(C.CONFIG, M32) & ~k
            cp[CONFIG5]["value"] |= CFG5_CV
            m._wmask[CONFIG5] = m._wmask.get(CONFIG5, M32) & ~CFG5_CV
        cp[C.CONFIG]["value"] = config | (self.mt << 7)
        cp[C.CONFIG3]["value"] |= CONFIG3_CDMM
        cp[CDMMBASE]["value"] = self.cdmm_size
        m.cdmm = self.device

    @property
    def cdmm_size(self) -> int:
        """CDMMBase.CDMMSize: the blocks instantiated, less one (FDC + MPU)."""
        return FDC_DRBS + self.drbs - 1

    def cp0_write(self, m, key: tuple[int, int], value: int) -> None:
        if key == CDMMBASE:                     # PA 31:15, EN; CI, CDMMSize read-only
            m.cp0[key]["value"] = (value & (CDMM_ADDR | CDMM_EN)) | self.cdmm_size

    def attributes(self, va: int) -> tuple[int, int | None]:
        """(the attribute byte, the region) of ``va``: the highest-numbered
        enabled region holding it, else its segment (None: the FMT build
        and no region — the fixed mapping's attributes)."""
        for lo, hi, bits, r in self.active:
            if lo <= va < hi:
                return bits, r
        if self.segments:
            return self.seg[va >> 28], None
        return -1, None

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        va &= M32
        bits, r = self.attributes(va) if self.active or self.segments else (-1, None)
        if bits < 0:
            pa, cca = self.fixed.xlate(m, va, kind)
            self.stats.hits += 1
            return pa, cca
        pa = va if self.segments else self.fixed.pa(m.cp0[C.STATUS]["value"], 0, va)
        if bits & 0x38:
            what = (("XI", "MPUL", "fetch") if kind == "fetch" and bits & XI
                    else ("RI", "MPUL", "read") if kind in ("load", "pcload") and bits & RI
                    else ("WI", "MPUS", "write") if kind == "store" and bits & WI else None)
            if what is not None:
                inhibit, name, verb = what
                where = f"region {r}" if r is not None else f"segment {va >> 28}"
                self.stats.fault(name)
                from .sim import _Arch
                raise _Arch(MPU_EXC, badvaddr=va, info={
                    "why": f"{name}: MPU {where} inhibits a {verb} ({inhibit} = 1)",
                    "mpu": (inhibit, r)})
        self.stats.hits += 1
        return pa, bits & 7

    def exception_taken(self, m, what: tuple[str, int | None]) -> None:
        """MPU_Config's last protection exception (Table 17)."""
        inhibit, r = what
        self.exc = ({"RI": CFG_EXCR, "WI": CFG_EXCW, "XI": CFG_EXCX}[inhibit]
                    | (CFG_MATCH | (r << 8) if r is not None else 0))

    def exception_entry(self, m, va: int) -> None:
        """No TLB registers to fill."""

    def peek(self, m, va: int) -> tuple[int, int]:
        va &= M32
        bits, _r = self.attributes(va) if self.active or self.segments else (-1, None)
        if bits < 0:
            return self.fixed.xlate(m, va, "load")
        return (va if self.segments else self.fixed.pa(m.cp0[C.STATUS]["value"], 0, va),
                bits & 7)

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        va &= M32
        bits, r = self.attributes(va)
        if bits < 0:
            return self.fixed.describe(m, va, kind)
        where = f"MPU region {r}" if r is not None else f"MPU segment {va >> 28}"
        if self.segments:
            how, seg, mapped = f"{where}: PA = VA, {flags(bits)}", segment(va), False
            pa = va
        else:
            t = self.fixed.describe(m, va, kind)
            how, seg, mapped, pa = f"{t.how}; {where}: {flags(bits)}", t.segment, t.mapped, t.pa
        return Translation(va, pa, bits & 7, seg, mapped, how)

    def reset_stats(self) -> None:
        from .tlb import TLBStats
        self.stats = TLBStats()

    def boot(self, m, *, handler: bool = False) -> None:
        """The boot firmware: the CDMM enabled at :data:`BOOT_CDMM`; the MPU
        build's segments as the fixed mapping's firmware sets Config: kuseg /
        kseg0 cacheable (``boot_cca``), kseg1 / kseg2 / kseg3 uncached."""
        from .tlb import boot_cca
        m.cp0[CDMMBASE]["value"] = ((BOOT_CDMM[self.segments] >> 4) & CDMM_ADDR
                                    | CDMM_EN | self.cdmm_size)
        if self.segments:
            self.seg = [boot_cca(m)] * 10 + [CCA_UNCACHED] * 6

    # -- the registers (the CDMM device) -----------------------------------------------

    def config(self) -> int:
        return (CFG_EN if self.segments else 0) | self.exc | (self.n - 1)

    def get(self, off: int) -> int:
        if off == ACSR:
            return (DEVTYPE << 24) | ((self.drbs - 1) << 16) | self.acsr
        if off == CONFIG:
            return self.config()
        if SEGCTL <= off < REGION:
            k = (off - SEGCTL) // 4 * 4
            return sum(self.seg[k + i] << (8 * i) for i in range(4))
        r, part = divmod(off - REGION, 8)
        if off >= REGION and r < self.n:
            return self.base[r] if part == 0 else self.ctl[r] if part == 4 else 0
        return 0

    def put(self, off: int, value: int) -> None:
        if off == ACSR:
            self.acsr = value & (ACSR_UW | ACSR_UR)
        elif SEGCTL <= off < REGION:
            k = (off - SEGCTL) // 4 * 4
            for i in range(4):
                self.seg[k + i] = (value >> (8 * i)) & 0x3F
        elif off >= REGION:
            r, part = divmod(off - REGION, 8)
            if r >= self.n or part not in (0, 4):
                return
            if part == 0:
                self.base[r] = value & ~0x1F & M32
            else:
                self.ctl[r] = value & 0xFFFF
            self._rebuild()

    def _rebuild(self) -> None:
        out = []
        for r in reversed(range(self.n)):
            span = self.span(r)
            if span is not None:
                out.append((*span, self.ctl[r] & 0x3F, r))
        self.active = out

    def span(self, r: int) -> tuple[int, int] | None:
        """The enabled addresses of region ``r``: [lo, hi) — None when it
        is off (or its Size is not 5-28)."""
        ctl = self.ctl[r]
        size = (ctl >> 10) & 0x1F
        if not ctl & REG_EN or not 5 <= size <= 28:
            return None
        sub = 1 << size
        first = self.base[r] & ~(sub - 1)
        region = first & ~(16 * sub - 1)
        k = (first - region) // sub
        last = min(k + ((ctl >> 6) & 0xF), 15)
        return region + k * sub, region + (last + 1) * sub

    def lines(self) -> list[str]:
        """The MPU's state, for the debugger and the TLB pane."""
        m = self.m
        cdmm = m.cp0[CDMMBASE]["value"] if m is not None else 0
        base = (cdmm & CDMM_ADDR) << 4
        out = [f"{self.name} (MD01232 ch. 4): {self.n} regions; " + (
            "PA = VA, every mode the same rights" if self.segments else
            "the fixed mapping, MPU regions overlay its attributes")]
        out.append(f"  CDMMBase 0x{cdmm:08x}: " + (
            f"the CDMM at PA 0x{base:08x}, the MPU's registers at 0x{base + MPU_OFFSET:08x}"
            if cdmm & CDMM_EN else "the CDMM is off (EN = 0)"))
        cfg = self.config()
        last = ("none" if not cfg & (CFG_EXCR | CFG_EXCW | CFG_EXCX) else
                f"{'RI' if cfg & CFG_EXCR else 'WI' if cfg & CFG_EXCW else 'XI'} in "
                + (f"region {(cfg >> 8) & 0x1F}" if cfg & CFG_MATCH else "a segment"))
        out.append(f"  MPU_Config 0x{cfg:08x}: En {int(self.segments)}, last protection "
                   f"exception: {last}; MPU_ACSR Ur {int(bool(self.acsr & ACSR_UR))} "
                   f"Uw {int(bool(self.acsr & ACSR_UW))}")
        if self.segments:
            for s in range(16):
                b = self.seg[s]
                out.append(f"  segment {s:2} 0x{s << 28:08x}-0x{(s << 28) | 0x0FFFFFFF:08x} "
                           f"C{b & 7} {flags(b)}")
        for r in range(self.n):
            span = self.span(r)
            ctl = self.ctl[r]
            if span is None and not ctl & REG_EN:
                continue
            if span is None:
                out.append(f"  region {r:2}: enabled, Size {(ctl >> 10) & 0x1F}: "
                           "not 5-28, matches nothing")
                continue
            sub = 1 << ((ctl >> 10) & 0x1F)
            out.append(f"  region {r:2} 0x{span[0]:08x}-0x{span[1] - 1:08x}: "
                       f"{(span[1] - span[0]) // sub} subregion(s) of {_size(sub)}, "
                       f"C{ctl & 7} {flags(ctl)}")
        if not any(self.span(r) for r in range(self.n)):
            out.append("  no region enabled")
        return out


def _anything(m, addr: int) -> bool:
    """With the MPU every mode may use every address (MD01232 ch. 4)."""
    return True


def _size(n: int) -> str:
    for unit, k in (("MB", 20), ("KB", 10)):
        if n >= 1 << k:
            return f"{n >> k} {unit}"
    return f"{n} bytes"


class CDMM:
    """The Common Device Memory Map of one VPE (MD01251 ch. 5): a bus
    device at CDMMBase while EN = 1, its instantiated blocks — the FDC's
    (not modelled: read 0) and the MPU's.  ACSRs only in kernel mode; the
    MPU's other registers in user mode with MPU_ACSR.Ur / Uw (5.2)."""

    def __init__(self, mpu: MPU) -> None:
        self.mpu = mpu

    def base(self, m) -> int | None:
        cell = m.cp0[CDMMBASE]["value"]
        return (cell & CDMM_ADDR) << 4 if cell & CDMM_EN else None

    def covers(self, pa: int) -> bool:
        cell = self.mpu.m.cp0[CDMMBASE]["value"]
        if not cell & CDMM_EN:
            return False
        off = pa - ((cell & CDMM_ADDR) << 4)
        return 0 <= off < self.mpu.size

    def _word(self, m, off: int, write: bool) -> int | None:
        """The MPU register offset of a CDMM word, None when this mode may
        not reach it (or it is the FDC's)."""
        if off < MPU_OFFSET:
            return None
        off -= MPU_OFFSET
        if not m.kernel_mode:
            if off == ACSR or not self.mpu.acsr & (ACSR_UW if write else ACSR_UR):
                return None
        return off

    def _get(self, m, word: int) -> int:
        off = self._word(m, word, False)
        return 0 if off is None else self.mpu.get(off)

    def load(self, m, pa: int, size: int) -> int:
        off = pa - self.base(m)
        big = m.big_endian
        out = 0
        for i in range(size):
            o = off + i
            w = self._get(m, o & ~3)
            byte = (w >> (8 * (3 - (o & 3) if big else o & 3))) & 0xFF
            out = (out << 8) | byte if big else out | (byte << (8 * i))
        return out

    def store(self, m, pa: int, value: int, size: int) -> None:
        off = pa - self.base(m)
        big = m.big_endian
        words: dict[int, int] = {}
        for i in range(size):
            o = off + i
            word = o & ~3
            if word not in words:
                words[word] = self._get(m, word)
            byte = (value >> (8 * (size - 1 - i) if big else 8 * i)) & 0xFF
            shift = 8 * (3 - (o & 3)) if big else 8 * (o & 3)
            words[word] = (words[word] & ~(0xFF << shift)) | (byte << shift)
        for word, w in words.items():
            reg = self._word(m, word, True)
            if reg is not None:
                self.mpu.put(reg, w)

    def tick(self, m, cycles: int) -> None:
        pass


def make(spec_kind: str, rest: str, flags: list[str]) -> MPU:
    """``mpu[:N]`` / ``fixed+mpu[:N]`` (emips.mmu.make_mmu)."""
    try:
        n = int(rest) if rest else 16
    except ValueError:
        raise ValueError(f"bad MPU region count {rest!r}") from None
    return MPU(n, segments=spec_kind == "mpu")
