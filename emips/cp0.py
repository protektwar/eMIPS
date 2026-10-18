"""Coprocessor 0 model of the eMIPS core (PLAN M2).

Reference: *MIPS32 Architecture For Programmers Volume III: The MIPS32 and
microMIPS32 Privileged Resource Architecture* (MD00090), Release 2 —
chapter "CP0 Registers" (Status, Cause, EPC, EBase, Config0-3, ...) and
chapter "Interrupts and Exceptions" (general exception processing,
exception vector locations, Cause ExcCode values).

The register *list* (names and selectors) still comes from
``emips/data/cp0.json`` (the Ruby dump).  This module adds what the dump
lacks: the **reset value** and the **software-writable mask** of the
registers the exception model relies on, for the eMIPS core
configuration:

* MIPS32 Release 2, 32-bit, Fixed Mapping MMU by default (``Config.MT =
  3``, see :mod:`emips.mmu`): no TLB, no supervisor mode, ``Config.K0 /
  KU / K23`` writable (reset value UNDEFINED in the manual: 2 = uncached
  here); or the TLB (``Config.MT = 1``, :mod:`emips.tlb`, which sets its
  own registers, Config1.MMUSize, Config3 / Config4 and supervisor mode);
* big-endian by default (``Config.BE``), little-endian selectable;
* FPU present (``Config1.FP = 1``; executed from M1 slice 2), no COP2,
  no MIPS16e / microMIPS; the 24K's four watch register pairs
  (``Config1.WR = 1``);
* Release 2+: 1-16 GPR shadow sets (``SRSCtl.HSS``, the Machine's
  ``shadow_sets``; the 24K builds 1, 2 or 4), ``SRSMap``;
* no caches by default; with :class:`emips.cache.Caches` the caches of a
  24K-style core (Config1 / Config2 geometry, ErrCtl, CacheErr, TagLo /
  DataLo / DataHi, L23TagLo / L23DataLo / L23DataHi, Config7.HCI,
  Config.MM with the write buffer, Config2.L2B; see :func:`cache_masks`);
* interrupts (Vol. III 6.1): six hardware lines HW0..HW5 →
  ``Cause.IP2..IP7``, the timer ORed onto IP7 (``IntCtl.IPTI = 7``), two
  software interrupts ``Cause.IP0..IP1``; compatibility mode (``Cause.IV``
  moves the interrupt vector to offset 0x200) and, from Release 2,
  vectored interrupts (``Config3.VInt = 1``, as the 24K: ``IntCtl.VS``
  spacing, IP7..IP0 priority, :func:`vi_vector`);
* ``Count`` increments every second cycle (``CCRes = 2``);
* EJTAG Debug Mode as the 24K's (Debug / DEPC / DESAVE, :mod:`emips.ejtag`).

Registers not listed in :data:`RESET` / :data:`WRITE_MASK` keep the old
behavior: reset value 0, fully writable by ``MTC0``.
"""

from __future__ import annotations

M32 = 0xFFFFFFFF

# --------------------------------------------------------------- registers
STATUS = (12, 0)
INTCTL = (12, 1)
SRSCTL = (12, 2)
SRSMAP = (12, 3)
CAUSE = (13, 0)
EPC = (14, 0)
PRID = (15, 0)
EBASE = (15, 1)
CONFIG = (16, 0)
CONFIG1 = (16, 1)
CONFIG2 = (16, 2)
CONFIG3 = (16, 3)
#: CMGCRBase (CP0 15.3, a CPS core: the Coherence Manager's GCR base);
#: Config3.CMGCR says it is implemented
CMGCRBASE = (15, 3)
CONFIG3_CMGCR = 1 << 29
#: Config3.VInt / VEIC: vectored interrupts / an External Interrupt
#: Controller (Vol. III 9.44)
CONFIG3_VINT, CONFIG3_VEIC = 1 << 5, 1 << 6
BADVADDR = (8, 0)
BADINSTR, BADINSTRP = (8, 1), (8, 2)
COUNT = (9, 0)
COMPARE = (11, 0)
CONFIG7 = (16, 7)
HWRENA = (7, 0)
USERLOCAL = (4, 2)
ERRCTL = (26, 0)
CACHEERR = (27, 0)
ITAGLO, IDATALO, DTAGLO, DDATALO = (28, 0), (28, 1), (28, 2), (28, 3)
IDATAHI = (29, 1)
L23TAGLO, L23DATALO, L23DATAHI = (28, 4), (28, 5), (29, 5)
#: Config.MM (24K): write-through merging in the write buffer.
CONFIG_MM = 1 << 18
ERROREPC = (30, 0)
#: EJTAG (MD00047 2.7; emips.ejtag): Debug, DEPC, DESAVE
DEBUG, DEPC, DESAVE = (23, 0), (24, 0), (31, 0)
#: the 24K's watch registers (MD00343 6.2.26 / 6.2.27): pairs 0, 1 watch
#: instruction fetches, pairs 2, 3 loads / stores
WATCH_PAIRS = 4
WATCHLO = [(18, i) for i in range(WATCH_PAIRS)]
WATCHHI = [(19, i) for i in range(WATCH_PAIRS)]
WATCH_I, WATCH_R, WATCH_W = 4, 2, 1
WATCHHI_G = 1 << 30
WATCHHI_M = 1 << 31

#: Registers the Ruby dump lacks: (reg, sel) -> name.  Config7 is the
#: implementation register of the 24K (HCI = 0, WII = 0: eMIPS's WAIT only
#: wakes for an enabled interrupt).
EXTRA_CELLS: dict[tuple[int, int], str] = {
    CONFIG7: "Config7",
    # the performance counters (emips.perf; the dump has only select 0)
    (25, 1): "PerfCnt0", (25, 2): "PerfCtl1", (25, 3): "PerfCnt1",
    # the watch register pairs 1-3 (the dump has select 0)
    **{(18, i): f"WatchLo{i}" for i in range(1, 4)},
    **{(19, i): f"WatchHi{i}" for i in range(1, 4)}}

# ------------------------------------------------------------- Status bits
ST_IE = 1 << 0
ST_EXL = 1 << 1
ST_ERL = 1 << 2
ST_KSU_SHIFT = 3            # KSU = bits 4..3 (UM = bit 4 when SM unused)
ST_BEV = 1 << 22
#: Status.NMI / SR (19 / 20): the reset vector was entered for a
#: non-maskable interrupt / a soft reset; software may only clear them
ST_NMI, ST_SR = 1 << 19, 1 << 20
ST_CU0 = 1 << 28
ST_CU1 = 1 << 29
ST_CU2 = 1 << 30
ST_CU3 = 1 << 31

ST_IM_SHIFT = 8             # IM7..IM0 = bits 15..8

# -------------------------------------------------------------- Cause bits
CA_BD = 1 << 31
CA_TI = 1 << 30             # timer interrupt pending
CA_WP = 1 << 22             # a watch exception is deferred
CA_DC = 1 << 27             # disable Count
CA_IV = 1 << 23             # interrupts use the special vector (0x200)
CA_IP_SHIFT = 8             # IP7..IP0 = bits 15..8
CA_CE_SHIFT = 28            # CE = bits 29..28
CA_EXC_SHIFT = 2            # ExcCode = bits 6..2

# ----------------------------------------- Cause.ExcCode (Vol. III table)
INT, MOD, TLBL, TLBS = 0, 1, 2, 3
ADEL, ADES = 4, 5
IBE, DBE = 6, 7
SYS, BP, RI, CPU, OV, TR = 8, 9, 10, 11, 12, 13
FPE = 15
#: Release 3 RI / XI exceptions (PageGrain.IEC = 1), Machine Check
TLBRI, TLBXI = 19, 20
WATCH = 23
MCHECK = 24
#: The Cache Error exception has no ExcCode (it does not touch Cause);
#: eMIPS uses this number, outside the 5-bit field, to name it.
CACHE_ERROR = 32
#: ... and this one the Non Maskable Interrupt (Vol. III: through the
#: reset vector, no Cause change; a CPS core's comes from the GIC)
NMI = 33
#: ... and these the EJTAG debug exceptions (MD00047 2.3, emips.ejtag):
#: single step, debug interrupt, the imprecise data breaks, instruction
#: break, SDBBP, the precise data breaks — named by their Debug bit
DSS, DINT, DDBLIMPR, DDBSIMPR, DIB, DBP, DDBL, DDBS = range(40, 48)
DEBUG_EXCEPTIONS = frozenset(range(40, 48))

EXC_NAMES: dict[int, str] = {
    INT: "Int", MOD: "Mod", TLBL: "TLBL", TLBS: "TLBS", ADEL: "AdEL",
    ADES: "AdES", IBE: "IBE", DBE: "DBE", SYS: "Sys", BP: "Bp", RI: "RI",
    CPU: "CpU", OV: "Ov", TR: "Tr", FPE: "FPE", TLBRI: "TLBRI", TLBXI: "TLBXI",
    WATCH: "WATCH", MCHECK: "MCheck", CACHE_ERROR: "CacheErr", NMI: "NMI",
    DSS: "DSS", DINT: "DINT", DDBLIMPR: "DDBLImpr", DDBSIMPR: "DDBSImpr",
    DIB: "DIB", DBP: "DBp", DDBL: "DDBL", DDBS: "DDBS",
}

#: TLB Refill vector offset (while Status.EXL = 0).
REFILL_OFFSET = 0x000
#: General-exception vector offset (Vol. III "Exception Vector Locations").
GENERAL_OFFSET = 0x180
#: Interrupt vector offset when ``Cause.IV = 1`` (non-vectored).
INTERRUPT_OFFSET = 0x200
#: Cache Error vector offset.
CACHE_ERROR_OFFSET = 0x100
#: Count resolution: cycles per ``Count`` increment (RDHWR 3).
CCRES = 2
#: ``Cause.IP`` bit of the timer interrupt (``IntCtl.IPTI = 7``).
TIMER_IP = 7
#: IntCtl.VS encodings (Vol. III Table 9.35): spacing = VS × 32 bytes;
#: the others are reserved (UNDEFINED)
VS_VALUES = frozenset({0x00, 0x01, 0x02, 0x04, 0x08, 0x10})
#: SRSCtl fields (Vol. III 9.31): HSS 29..26, EICSS 21..18, ESS 15..12,
#: PSS 9..6, CSS 3..0
SRS_HSS, SRS_EICSS, SRS_ESS, SRS_PSS, SRS_CSS = 26, 18, 12, 6, 0
#: Vector base while ``Status.BEV = 1`` (bootstrap, kseg1 ROM area).
BEV_BASE = 0xBFC00200


#: Config.AR (architecture revision) per release: 0 = Release 1,
#: 1 = Release 2 (also R3 and R5), 2 = Release 6.
CONFIG_AR = {"R1": 0, "R2": 1, "R3": 1, "R5": 1, "R6": 2}


#: Config.K0 / KU / K23 after reset.  The manual leaves them UNDEFINED;
#: eMIPS resets them to 2 (uncached), the attribute every MIPS has.
RESET_CCA = 2


def config0(big_endian: bool, release: str = "R5") -> int:
    """Config: M=1 (Config1 present), K23 / KU (fixed mapping), BE, AT=0
    (MIPS32), AR (release), MT=3 (fixed mapping MMU), K0."""
    return ((1 << 31) | (RESET_CCA << 28) | (RESET_CCA << 25)
            | (int(big_endian) << 15) | (CONFIG_AR[release] << 10)
            | (3 << 7) | RESET_CCA)


#: Reset values (Vol. III "Reset Exception": BEV = ERL = 1; EBase base
#: 0x80000000).  ``Config`` depends on endianness; see :func:`reset_values`.
RESET: dict[tuple[int, int], int] = {
    STATUS: ST_BEV | ST_ERL,
    CAUSE: 0,
    EBASE: 0x80000000,
    CONFIG1: (1 << 31) | (1 << 3) | 1,   # M=1 (Config2 present), WR=1 (watch), FP=1; no C2/CA
    CONFIG2: 1 << 31,            # M=1 (Config3 present)
    CONFIG3: 0,                  # MIPS32 only; VInt from Release 2 (reset_values)
    PRID: 0,                     # CompanyID 0: not a licensed core
    INTCTL: (TIMER_IP << 29) | (7 << 26),   # IPTI = 7; IPPCI = 7 (emips.perf); VS = 0
    # the 24K's (MD00343 Table 6.36): DM = 0, NoDCR = 0, CountDM = 1,
    # EJTAGver = 3 (EJTAG 3.x), NoSSt = 0; the cause bits are UNDEFINED: 0
    DEBUG: (1 << 25) | (3 << 15),
    # WatchHi.M: another pair follows (pairs 0-2)
    **{WATCHHI[i]: WATCHHI_M for i in range(WATCH_PAIRS - 1)},
    # MIPS Trace (MD00343 6.2.29-6.2.33): TraceControl G = 1 without a TLB
    # (the Machine clears it with one), TraceControl2 ValidModes = 10 and
    # Mode bit 7 (PC tracing is always on on the 24K)
    (23, 1): 1 << 4,
    (23, 2): (2 << 5) | (1 << 7),
}

#: Bits ``MTC0`` may change; the rest are read-only (hardware-owned).
WRITE_MASK: dict[tuple[int, int], int] = {
    # CU1, CU0, RP, FR, BEV, IM7..0, KSU, ERL, EXL, IE.
    # CU2/CU3 read 0 (no COP2/COP3), RE = 0 (no reverse-endian).
    STATUS: (ST_CU1 | ST_CU0 | (1 << 27) | (1 << 26) | ST_BEV | 0xFF00
             | (3 << ST_KSU_SHIFT) | ST_ERL | ST_EXL | ST_IE),
    # IV, DC, WP (software clears it), IP1..0 (software interrupts).  BD,
    # CE, ExcCode, IP7..2: RO.
    CAUSE: (1 << 23) | (1 << 27) | CA_WP | 0x300,
    EBASE: 0x3FFFF000,           # ExceptionBase 29..12; 31..30 = 10, CPUNum RO
    BADVADDR: 0,
    PRID: 0,
    CONFIG: (7 << 28) | (7 << 25) | 7,   # K23, KU (fixed mapping), K0
    CONFIG1: 0,
    CONFIG2: 0,
    CONFIG3: 0,
    CMGCRBASE: 0,                # the GCR base: read-only (a CPS core)
    # 24K Config7 (MD00343 6.2.25): SL, BHT, RPS, BP, ULB, NBLSU, CPOOO are
    # writable (the 24K pipeline model uses them; the others ignore them)
    CONFIG7: 0x7F,
    INTCTL: 0,                   # VS from Release 2 (write_masks)
    SRSCTL: 0,                   # ESS / PSS with shadow sets (Machine)
    SRSMAP: 0,
    HWRENA: 0x2000000F,          # CPUNum, SYNCI_Step, CC, CCRes, ULR
    # Debug: LSNM, CountDM, IEXI, SSt (CacheEP / DBusEP are R/W1: the
    # Machine sets them); IBusEP / MCheckP read 0 on the 24K
    DEBUG: (1 << 28) | (1 << 25) | (1 << 20) | (1 << 8),
    # TraceControl: TS UT TB IO D E K U TFCR TLSM TIM On (S, ASID_M, ASID,
    # G: the Machine adds them with supervisor mode / a TLB); TraceControl2:
    # Mode 10:8 (bit 7 PC: always 1); UserTraceData; TraceIBPC IE + 4 x 3
    # bits; TraceDBPC DE + 2 x 3 bits
    (23, 1): (1 << 31) | (1 << 30) | (1 << 27) | (1 << 26) | (1 << 25) | (1 << 24)
    | (1 << 23) | (1 << 21) | 0xF,
    (23, 2): 0xF << 8,
    (23, 3): M32,
    (23, 4): (1 << 28) | 0xFFF,
    (23, 5): (1 << 28) | 0x3F,
    # WatchLo: VAddr and I (pairs 0, 1) / R, W (pairs 2, 3); WatchHi: G,
    # ASID, Mask (I / R / W are write-one-to-clear: the Machine)
    **{WATCHLO[i]: 0xFFFFFFF8 | (WATCH_I if i < 2 else WATCH_R | WATCH_W)
       for i in range(WATCH_PAIRS)},
    **{WATCHHI[i]: WATCHHI_G | (0xFF << 16) | 0xFF8 for i in range(WATCH_PAIRS)},
}


ST_FR = 1 << 26


def reset_values(big_endian: bool, release: str = "R5") -> dict[tuple[int, int], int]:
    values = dict(RESET)
    values[CONFIG] = config0(big_endian, release)
    if release != "R1":                 # vectored interrupts (the 24K: always)
        values[CONFIG3] |= CONFIG3_VINT
    values[CONFIG3] |= 1                # TL: MIPS Trace (emips.pdtrace)
    if release == "R6":                 # 64-bit FPRs only: FR = 1
        values[STATUS] |= ST_FR
        values[CONFIG3] |= 1 << 13      # ULRI: UserLocal is required in R6
    return values


def write_masks(release: str = "R5", *, supervisor: bool = False
                ) -> dict[tuple[int, int], int]:
    """MTC0 masks of the release: a Release 1 FPU is 32-bit only, so
    ``Status.FR`` is read-only 0.  Without supervisor mode (the Fixed
    Mapping MMU has none) ``Status`` bit 3 is R0: reads 0, ignores writes,
    and bit 4 is UM."""
    masks = dict(WRITE_MASK)
    if release in ("R1", "R6"):         # FR fixed: 0 (R1) / 1 (R6)
        masks[STATUS] &= ~ST_FR
    if not supervisor:
        masks[STATUS] &= ~(1 << ST_KSU_SHIFT)
    if release != "R1":                 # IntCtl.VS: VInt (and VEIC) implemented
        masks[INTCTL] = 0x1F << 5
    return masks


def ksu(status: int) -> int:
    """0 = kernel, 1 = supervisor, 2 = user (``Status.KSU``)."""
    return (status >> ST_KSU_SHIFT) & 3


def kernel_mode(status: int) -> bool:
    """Kernel mode: ``KSU = 0`` or ``EXL = 1`` or ``ERL = 1``."""
    return ksu(status) == 0 or bool(status & (ST_EXL | ST_ERL))


def address_ok(status: int, addr: int) -> bool:
    """Segment access check of a 32-bit virtual address (Vol. III "Virtual
    Memory": user mode reaches kuseg only; supervisor mode kuseg and sseg
    (0xC0000000-0xDFFFFFFF); kernel mode everything)."""
    if kernel_mode(status):
        return True
    addr &= M32
    if addr < 0x80000000:
        return True
    return ksu(status) == 1 and 0xC0000000 <= addr < 0xE0000000


def vector(status: int, ebase: int, *, interrupt_iv: bool = False,
           refill: bool = False, bev_base: int = BEV_BASE,
           offset: int | None = None) -> int:
    """General exception vector: BEV=1 -> 0xBFC00380, else EBase + 0x180.
    An interrupt with ``Cause.IV = 1`` uses offset 0x200 instead
    (0xBFC00400 / EBase + 0x200), a TLB Refill taken with EXL = 0 offset
    0x000 (0xBFC00200 / EBase + 0x000; Vol. III Table 6.9).  A core of a
    CPS has its BEV vectors at its reset base + 0x200 (``bev_base``,
    GCR_Cx_RESET_BASE, emips.cps).  ``offset``: a vectored interrupt's
    (:func:`vi_offset`)."""
    base = bev_base if status & ST_BEV else (ebase & 0xFFFFF000)
    if offset is None:
        offset = (REFILL_OFFSET if refill else
                  INTERRUPT_OFFSET if interrupt_iv else GENERAL_OFFSET)
    # "inhibiting a carry between bit 29 and bit 30" (EBase, Vol. III 9.41)
    return (base & 0xC0000000) | ((base + offset) & 0x3FFFFFFF)


def cache_error_vector(status: int, ebase: int, release: str = "R5", *,
                       cv: bool = False, bev_base: int = BEV_BASE) -> int:
    """Cache Error vector (Vol. III 6.2.15, 4.10.1.6): 0xBFC00300 while
    BEV = 1; else ``EBase[31:30] || 1 || EBase[28:12] || 0x100`` (bit 29
    forced to 1: kseg1, unmapped and uncached), or the full EBase with
    Segmentation Control and ``Config5.CV = 1``; 0xA0000100 in Release 1."""
    if status & ST_BEV:
        return bev_base + CACHE_ERROR_OFFSET
    if release == "R1":
        return 0xA0000000 + CACHE_ERROR_OFFSET
    if cv:
        return (ebase & 0xFFFFF000) + CACHE_ERROR_OFFSET
    return ((ebase & 0xDFFFF000) | (1 << 29)) + CACHE_ERROR_OFFSET


def cache_config2(l2, l3) -> int:
    """Config2 TS/TL/TA (bits 27..16) and SS/SL/SA (11..0) of the L3 / L2
    caches (:class:`emips.cache.Geometry`, ``None`` = no cache: line
    code 0); M (bit 31) is set by the reset value."""
    value = 0
    for geometry, shift in ((l3, 16), (l2, 0)):
        if geometry is not None:
            s, l_, a = geometry.config2()
            value |= (s << (shift + 8)) | (l_ << (shift + 4)) | (a << shift)
    return value


def cache_config1(icache, dcache) -> int:
    """Config1 IS/IL/IA (bits 24..16) and DS/DL/DA (15..7) of the caches
    (:class:`emips.cache.Geometry`, ``None`` = no cache: line code 0)."""
    value = 0
    for geometry, shift in ((icache, 16), (dcache, 7)):
        if geometry is not None:
            s, l_, a = geometry.config1()
            value |= (s << (shift + 6)) | (l_ << (shift + 3)) | (a << shift)
    return value


def cache_masks(parity: bool, *, ecc: bool = False, l2: bool = False,
                write_buffer: bool = False) -> dict[tuple[int, int], int]:
    """MTC0 masks of the cache registers (24K Tables 6.28-6.58): ErrCtl
    PE (only with parity), L2P (only with an ECC L2), PO, WST, SPR, PCO,
    FE, PCI, PI, PD; CacheErr read-only; I/DTagLo without the reserved
    bits 9:8 and 4:1 (L23TagLo: "a general 32b R/W register"); Config2.L2B
    with an L2; Config.MM with the write buffer."""
    errctl = ((1 << 30) | (1 << 29) | (1 << 28) | (1 << 27) | (1 << 19)
              | (0x3F << 13) | (0x1FF << 4) | 0xF)
    if parity:
        errctl |= 1 << 31
    if ecc:
        errctl |= 1 << 23
    taglo = M32 & ~(0x300 | 0x1E)
    masks = {ERRCTL: errctl, CACHEERR: 0, ITAGLO: taglo, DTAGLO: taglo,
             CONFIG7: 0x7F, CONFIG2: (1 << 12) if l2 else 0}
    if write_buffer:
        masks[CONFIG] = WRITE_MASK[CONFIG] | CONFIG_MM
    return masks


def interrupt_pending(status: int, cause: int) -> bool:
    """An interrupt is taken when some ``Cause.IP`` bit is enabled by its
    ``Status.IM`` bit, ``Status.IE = 1`` and ``EXL = ERL = 0``."""
    if not status & ST_IE or status & (ST_EXL | ST_ERL):
        return False
    return bool((cause >> CA_IP_SHIFT) & (status >> ST_IM_SHIFT) & 0xFF)


def interrupt_mode(status: int, cause: int, intctl: int, config3: int) -> str:
    """Vol. III Table 6.1: "compatibility", "vectored" (VI) or "eic"."""
    vs = (intctl >> 5) & 0x1F
    if status & ST_BEV or not cause & CA_IV or not vs:
        return "compatibility"
    if config3 & CONFIG3_VEIC:
        return "eic"
    return "vectored" if config3 & CONFIG3_VINT else "compatibility"


def vi_vector(status: int, cause: int) -> int:
    """The VI-mode priority encoder (Vol. III Table 6.3): the vector
    number 7..0 of the highest pending enabled interrupt, HW5 (IP7)
    first, SW0 (IP0) last; -1 when none."""
    return ((cause >> CA_IP_SHIFT) & (status >> ST_IM_SHIFT) & 0xFF).bit_length() - 1


def vi_offset(vecnum: int, intctl: int) -> int:
    """Vol. III 6.1.2: ``0x200 + vectorNumber × (IntCtl.VS || 0b00000)``."""
    return INTERRUPT_OFFSET + vecnum * (((intctl >> 5) & 0x1F) << 5)
