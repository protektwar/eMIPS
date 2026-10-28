"""Simulator options (GUI Configure > Settings, CLI flags), free of Qt.

:class:`Options` holds everything the Settings dialog edits besides the
release / byte order / mode / forwarding (which live on :class:`Session`
and the toolbar).  :meth:`Options.to_strings` / :meth:`Options.from_strings`
turn it into flat ``{key: str}`` pairs for ``QSettings``; unknown or
malformed values fall back to the defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields

from .cache import Caches, Geometry
from .malta import make_board
from .mmu import make_mmu
from .pipeline import LATENCY
from .timing import MEMORY_TIMING

BOOT_MODES = ("kernel", "user", "reset", "guest")   # (guest: a RISC-V hart with H, LF7)
#: the C compiler's optimisation levels (-O...)
C_OPTIMIZE = ("0", "1", "2", "3", "s")
EXCEPTION_MODES = ("cp0", "stop")
#: Pipeline models (pipeline mode): the textbook 5-stage pipeline, the MIPS 24K's
PIPELINE_MODELS = ("5-stage", "24k", "p8700")
#: Cache contents after reset: "random" (undefined, as on silicon) or
#: "hardware" (invalid: the 24K's Config7.HCI, simulation models)
CACHE_INIT_MODES = ("random", "hardware")
#: the most cores a system has (emips.cps: the P5600 / P6600's six,
#: MD01024 / MD01137)
MAX_CORES = 6
#: the most GPR sets (Vol. III: SRSCtl.HSS is 4 bits)
MAX_SHADOW_SETS = 16
#: the compressed instruction sets a core may have
COMPRESSED = ("auto", "none", "mips16e", "micromips", "micromips-only")
#: the DSP Module a core may have (emips.dsp, MD00374)
DSP_LEVELS = ("auto", "none", "dsp", "dspr2")
#: MSA, the MIPS SIMD Architecture (emips.msa, MD00866; PLAN M24)
MSA_LEVELS = ("auto", "none", "msa")
#: the architecture (PLAN M22): MIPS32, or MIPS64 (emips.sim64: 64-bit
#: registers; the 32-bit compatibility segments with the fixed mapping —
#: Config.AT = 1 —, the 64-bit address space with a TLB — AT = 2), or
#: nanoMIPS (PLAN M23, emips.simnano: nanoMIPS32 Release 6, its own ISA)
ARCHES = ("mips32", "mips64", "nanomips", "rv32", "rv64")
#: each architecture's name as the manuals write it
ARCH_TITLES = {"mips32": "MIPS32", "mips64": "MIPS64", "nanomips": "nanoMIPS",
               "rv32": "RV32", "rv64": "RV64"}
#: RISC-V (PLAN M25): the extensions of eMIPS's generic hart (emips.simrv;
#: RV64GC with Zba / Zbb, the P8700's set) and its privilege modes
RV_EXTENSIONS = "imafdc_zicsr_zifencei_zicntr_zihpm_zba_zbb"
RV_MODES = ("msu", "mu", "m")
#: a generic RISC-V hart's platform (LF9): RARS's console wired to MEIP, or
#: Spike's CLINT with a PLIC, an APLIC (direct delivery) or IMSICs (the
#: APLIC forwarding by MSIs)
RV_INTC = ("none", "plic", "aplic", "imsic")


def is_rv(arch: str) -> bool:
    """A RISC-V architecture (rv32 / rv64)."""
    return arch in ("rv32", "rv64")


def arch_title(arch: str) -> str:
    """``"nanomips"`` -> "nanoMIPS" (MD01247's spelling), ``"mips64"`` ->
    "MIPS64" ..."""
    return ARCH_TITLES.get(arch, arch.upper())


def gdb_program(arch: str) -> str:
    """The GDB that debugs this architecture: gdb-multiarch knows MIPS32 /
    MIPS64 and RISC-V, not nanoMIPS (its GDB is MediaTek's nanomips-elf-gdb)."""
    return "nanomips-elf-gdb" if arch == "nanomips" else "gdb-multiarch"

#: a nanoMIPS core's CRC32 instructions (Config5.CRCP) and global
#: invalidates (Config5.GI): "auto" = when the program has them
NANO_CRC = ("auto", "none", "crc")
NANO_GINV = ("auto", "none", "ginvi", "ginvt")
#: the instruction set C programs are compiled to (emips.cc)
C_ISAS = ("mips32", "mips16e", "micromips")



_RNMI_PART = re.compile(r"^(?:0[xX][0-9a-fA-F]+|[0-9]+|[A-Za-z_.$][\w.$]*)$")


#: the DSPRAM sizes (KB) a P8700 core may have (eMIPS: a power of two)
DSPRAM_KB = (0, 4, 8, 16, 32, 64, 128, 256)


def parse_waits(text: str) -> tuple[int, int, int] | None:
    """``--l2-waits T:W:D`` (each 0-3) -> (tag, way select, data), None when
    it is not that."""
    try:
        parts = tuple(int(x) for x in str(text).split(":"))
    except ValueError:
        return None
    return parts if len(parts) == 3 and all(0 <= x <= 3 for x in parts) else None


def parse_rnmi(text: str) -> list[str] | None:
    """``--rv-rnmi INT[,EXC]``'s parts (numbers or labels); [] for "", None
    when it is not that."""
    if not text:
        return []
    parts = [p.strip() for p in text.split(",")]
    if len(parts) > 2 or not all(_RNMI_PART.match(p) for p in parts):
        return None
    return parts

@dataclass
class Options:
    # assembler
    #: pseudo-instructions allowed (False = strict mode)
    pseudo: bool = True
    #: start every source in ``.set reorder`` (nop in each delay slot)
    reorder: bool = True
    #: Run / Step assemble the source first when it changed
    auto_assemble: bool = True
    #: C programs: the compiler's optimisation level (-O0 / -O1 / -O2 / -O3 / -Os);
    #: -O0 by default: every C line keeps its own code (stepping by lines)
    c_optimize: str = "0"
    #: C programs: "mips32", "mips16e" (cc --mips16) or "micromips" (cc
    #: --micromips, Release 3 / 5); crt0 stays MIPS32
    c_isa: str = "mips32"
    # processor
    #: Count increments every N cycles (CCRes: 1 or 2)
    count_rate: int = 2
    #: pipeline latencies (cycles), see :data:`emips.pipeline.LATENCY`
    latency: dict[str, int] = field(default_factory=lambda: dict(LATENCY))
    #: the pipeline model of pipeline mode, see PIPELINE_MODELS
    model: str = "5-stage"
    #: cores (1 to 6): more than one is a Coherent Processing System
    #: (emips.cps: the GCRs, the CPC, shared memory and devices)
    cores: int = 1
    #: an I/O Coherence Unit in the CPS (GCR_CONFIG.NUMIOCU = 1): DMA on a
    #: multi-core Malta is coherent with the data caches (emips.coherence)
    iocu: bool = False
    #: GPR shadow sets (SRSCtl.HSS + 1; Release 2+): 1 = the normal GPRs
    #: only; the 24K builds 1, 2 or 4, the architecture allows up to 16
    shadow_sets: int = 4
    #: the compressed instruction set the core has: "auto" = the core's
    #: own — MIPS16e from Release 1 to 5, as the 24K (it always has it),
    #: none in Release 6; "none"; "mips16e" (Config1.CA; emips.mips16);
    #: "micromips" (microMIPS32 and MIPS32, Config3.ISA = 2, Release 3+;
    #: emips.micromips); "micromips-only" (Config3.ISA = 1)
    compressed: str = "auto"
    #: an FPU (CP1: Config1.FP; off = a core without one, the 24Kc: Status.CU1
    #: reads 0, every FPU instruction is Coprocessor Unusable; C as soft-float)
    fpu: bool = True
    #: the DSP Module (emips.dsp): "auto" = the core has it (Revision 2)
    #: when the program has DSP instructions, else not (the 24K has none);
    #: "none"; "dsp" (Revision 1, the 24KE); "dspr2" (Revision 2: the 34K,
    #: 74K, 1004K, interAptiv, microAptiv; on Release 6 Revision 3)
    dsp: str = "auto"
    #: MSA (emips.msa, MD00866, Release 5 / 6): "auto" = the core has it when
    #: the program has MSA instructions (and no core preset says otherwise);
    #: "none"; "msa"
    msa: str = "auto"
    #: MSAIR.WRP: vector register partitioning with this many physical
    #: registers (1-32; 0 = none, as every real core: MD00866 3.4.3)
    msa_wrp: int = 0
    #: the MT Module (emips.mt, MD00378): "auto" = 2 VPEs and 5 TCs (a 34K)
    #: when the program has MT instructions, else none; "none"; "VPES:TCS"
    #: (1-3 VPEs, VPES-9 TCs: the 34K's 1-2, the I7200's 1-3; Release 2-5, nanoMIPS)
    mt: str = "auto"
    #: the policy manager of the MT processor's dispatch scheduler (MD00534
    #: 13.2): "rr" basic round-robin, "wrr" weighted round-robin (TCSchedule
    #: groups), "wrr2" with its throttles
    mt_pm: str = "rr"
    #: the MT processor's ITC (Inter-Thread Communication storage, MD00534
    #: ch. 12): "auto" = 4 FIFO + 8 semaphore cells, "none",
    #: "FIFOS:SEMAPHORES" (0, 1, 2, 4, 8 or 16 each)
    itc: str = "auto"
    #: Release 6 virtual processors a core (emips.vp: Vol. III 9.8, the
    #: I6400's 1, 2 or 4 — 4 at most, the VP numbers of a core); more than
    #: one makes a CM3 system (emips.cps: the CPC starts the VPs, DVP / EVP)
    vps: int = 1
    #: the ITU of a CM3 core (the I6400's gating storage, emips.itu): "auto"
    #: = 4 FIFO cells of 4 entries + 8 single-entry cells, "none",
    #: "FIFOS:SINGLES[:DEPTH]" (32 cells at most; FIFOs of 2, 4 or 8)
    itu: str = "auto"
    #: the core preset (emips.cores: "24kf", "m14k" ...): the release and the
    #: settings its manuals allow, its PRId; "" = the generic eMIPS core
    core: str = ""
    #: the architecture, see ARCHES: "mips64" = a MIPS64 core (64-bit GPRs,
    #: the doubleword instructions; the program is assembled for MIPS64);
    #: "nanomips" = a nanoMIPS32 core (Release 6; programs assembled with
    #: emips.nanoasm)
    arch: str = "mips32"
    #: a nanoMIPS core: the nanoMIPS subset (Config5.NMS = 1: the formats
    #: MD01247 marks "not in NMS" are Reserved Instructions; programs are
    #: assembled as GNU as -march=32r6s does)
    nms: bool = False
    #: nanoMIPS programs: the linker's relaxations (ld --relax: gold shrinks
    #: the instructions .linkrelax left relaxable once the addresses are known)
    relax: bool = False
    #: ... its CRC32 instructions (Config5.CRCP): "auto" = when the program
    #: has CRC32 instructions, "none", "crc"
    crc: str = "auto"
    #: ... its global invalidates (Config5.GI): "auto" = GI 3 when the
    #: program has GINVT, 2 when it has GINVI only; "none" (GI 0), "ginvi"
    #: (GI 2: GINVI), "ginvt" (GI 3: GINVI and GINVT)
    ginv: str = "auto"
    #: a RISC-V hart (PLAN M25): its extensions (the ISA string after
    #: rv32 / rv64; the programs are assembled for them too), its privilege
    #: modes ("msu", "mu", "m"), and the linker relaxation of its programs
    #: (GNU's default: as -mrelax, ld relaxing; off = -mno-relax)
    rv_extensions: str = RV_EXTENSIONS
    rv_modes: str = "msu"
    rv_relax: bool = True
    #: Smrnmi's handler addresses (priv. 8.5: implementation-defined; 0 as
    #: QEMU and Spike): "INT[,EXC]" — the RNMI handler and the RNMI
    #: exception handler (default: INT), numbers or the program's labels
    rv_rnmi: str = ""
    #: the platform's interrupt controller (RV_INTC; LF9): with one, --cores
    #: gives several harts on it
    rv_intc: str = "none"
    #: the P8700's L2 RAM wait states "TAG:WAYSELECT:DATA", 0-3 each (MD01502
    #: 5.14.1.20, "from configuration"; LF15)
    l2_waits: str = "0:0:0"
    #: the P8700's DSPRAM a core, KB (0: none; MD01502 7.3.7, LF15)
    dspram: int = 0
    # memory / boot
    #: "kernel" (MARS-like: kernel mode, FPU on, $sp/$gp set, handler at
    #: 0x80000180), "user" (the same in user mode), "reset" (the hardware
    #: reset state: BEV = ERL = 1, FPU off; PC = the entry)
    boot: str = "kernel"
    #: L1 caches: "none" or SIZE[:WAYS[:LINE]] (emips.cache.Geometry.parse;
    #: the 24K: "32K" = 32 KB, 4-way, 32-byte lines)
    icache: str = "none"
    dcache: str = "none"
    #: cache parity implemented (ErrCtl.PE writable, Cache Error exception)
    cache_parity: bool = False
    #: L2 / L3 caches: "none" or SIZE[:WAYS[:LINE]] (Config2; 8 ways and
    #: 32-byte lines by default, like a SOC-it L2: "256K"); L3 needs an L2
    l2: str = "none"
    l3: str = "none"
    #: L2 / L3 ECC implemented (ErrCtl.L2P writable)
    cache_ecc: bool = False
    #: the 24K write buffer (4 x 32 bytes: write-through merging,
    #: uncached accelerated gathering, emptied by SYNC)
    write_buffer: bool = False
    #: memory timing of the pipeline with caches (emips.timing.MEMORY_TIMING)
    mem_timing: dict[str, int] = field(default_factory=lambda: dict(MEMORY_TIMING))
    #: loads miss in the background (hit under miss) instead of blocking
    nonblocking: bool = False
    #: cache contents after reset, see CACHE_INIT_MODES
    cache_init: str = "random"
    #: the board: "flat" (every physical address is RAM, the MARS layout)
    #: or "malta[:RAM_MB[:SWITCHES]]" (the MIPS Malta: the GT-64120's map,
    #: bus errors, the board registers; emips.malta)
    board: str = "flat"
    #: the Malta's IDE disks: "[SLOT=]SOURCE,..." with SLOT pri0 / pri1 /
    #: sec0 / sec1 and SOURCE an image file or blank:SIZE (emips.ata; every
    #: run starts from the image, Session.save_disk keeps what it wrote)
    disks: str = ""
    #: the MMU (Config.MT): "fixed" (the Fixed Mapping MMU) or "tlb[:N]"
    #: (a TLB of N dual entries, default 32, with the 24K micro-TLBs)
    mmu: str = "fixed"
    # exceptions / console
    #: "cp0": take exceptions (jump to the handler); "stop": stop and show
    exceptions: str = "cp0"
    #: MARS / SPIM syscall services (off: SYSCALL takes the exception)
    syscalls: bool = True
    #: MMIO display busy / keyboard key spacing, in cycles
    tx_delay: int = 6
    rx_delay: int = 6
    # run
    #: Run stops after this many steps / cycles (0: no limit)
    step_limit: int = 0

    def to_strings(self) -> dict[str, str]:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool):
                out[f.name] = "1" if v else "0"
            elif isinstance(v, dict):
                out[f.name] = ",".join(f"{k}={n}" for k, n in v.items())
            else:
                out[f.name] = str(v)
        return out

    @classmethod
    def from_strings(cls, data: dict[str, str]) -> "Options":
        o = cls()
        for f in fields(cls):
            raw = data.get(f.name)
            if raw is None:
                continue
            default = getattr(o, f.name)
            try:
                if isinstance(default, bool):
                    value = raw == "1"
                elif isinstance(default, int):
                    value = int(raw)
                elif isinstance(default, dict):
                    value = dict(default)
                    least = 0 if f.name == "mem_timing" else 1
                    for item in raw.split(","):
                        k, _, n = item.partition("=")
                        if k in value and int(n) >= least:
                            value[k] = int(n)
                else:
                    value = raw
            except ValueError:
                continue
            setattr(o, f.name, value)
        return o.validated()

    def validated(self) -> "Options":
        """Out-of-range values back to their defaults."""
        d = Options()
        if self.boot not in BOOT_MODES:
            self.boot = d.boot
        if self.c_optimize not in C_OPTIMIZE:
            self.c_optimize = d.c_optimize
        if self.c_isa not in C_ISAS:
            self.c_isa = d.c_isa
        if self.exceptions not in EXCEPTION_MODES:
            self.exceptions = d.exceptions
        if self.count_rate not in (1, 2):
            self.count_rate = d.count_rate
        if self.cache_init not in CACHE_INIT_MODES:
            self.cache_init = d.cache_init
        if self.model not in PIPELINE_MODELS:
            self.model = d.model
        if not 1 <= self.cores <= MAX_CORES:
            self.cores = d.cores
        if not 1 <= self.shadow_sets <= MAX_SHADOW_SETS:
            self.shadow_sets = d.shadow_sets
        if self.compressed not in COMPRESSED:
            self.compressed = d.compressed
        if self.dsp not in DSP_LEVELS:
            self.dsp = d.dsp
        if self.msa not in MSA_LEVELS:
            self.msa = d.msa
        if not 0 <= self.msa_wrp <= 32:
            self.msa_wrp = d.msa_wrp
        if self.arch not in ARCHES:
            self.arch = d.arch
        if self.crc not in NANO_CRC:
            self.crc = d.crc
        if self.ginv not in NANO_GINV:
            self.ginv = d.ginv
        if self.rv_modes not in RV_MODES:
            self.rv_modes = d.rv_modes
        if not isinstance(self.rv_rnmi, str) or parse_rnmi(self.rv_rnmi) is None:
            self.rv_rnmi = d.rv_rnmi
        if self.rv_intc not in RV_INTC:
            self.rv_intc = d.rv_intc
        if parse_waits(self.l2_waits) is None:
            self.l2_waits = d.l2_waits
        if not isinstance(self.dspram, int) or self.dspram not in DSPRAM_KB:
            self.dspram = d.dspram
        try:
            from .riscv import RVISA, parse_extensions
            if "i" not in parse_extensions(self.rv_extensions):
                raise ValueError(self.rv_extensions)
            RVISA.for_core(64, self.rv_extensions)
        except (ValueError, KeyError):
            self.rv_extensions = d.rv_extensions
        self.mt = (self.mt or "").strip().lower()
        if self.mt != "auto":
            from .mt import parse_spec
            try:
                spec = parse_spec(self.mt)
                self.mt = "none" if spec is None else f"{spec[0]}:{spec[1]}"
            except ValueError:
                self.mt = d.mt
        from .mt import POLICY_MANAGERS
        if self.mt_pm not in POLICY_MANAGERS:
            self.mt_pm = d.mt_pm
        self.itc = (self.itc or "").strip().lower()
        if self.itc != "auto":
            from .itc import parse_spec as itc_spec
            try:
                spec = itc_spec(self.itc)
                self.itc = "none" if spec is None else f"{spec[0]}:{spec[1]}"
            except ValueError:
                self.itc = d.itc
        from .vp import MAX_VP
        if not 1 <= self.vps <= MAX_VP:
            self.vps = d.vps
        self.itu = (self.itu or "").strip().lower()
        if self.itu != "auto":
            from .itu import parse_spec as itu_spec
            try:
                spec = itu_spec(self.itu)
                self.itu = "none" if spec is None else ":".join(map(str, spec))
            except ValueError:
                self.itu = d.itu
        from .cores import CORES
        self.core = (self.core or "").lower()
        if self.core and self.core not in CORES:
            self.core = d.core
        for name, level in (("icache", 1), ("dcache", 1), ("l2", 2), ("l3", 3)):
            try:
                Geometry.parse(getattr(self, name), level)
            except ValueError:
                setattr(self, name, getattr(d, name))
        if Geometry.parse(self.l2, 2) is None:
            self.l3 = "none"                    # an L3 needs an L2
        try:
            make_mmu(self.mmu)
        except ValueError:
            self.mmu = d.mmu
        try:
            make_board(self.board)
        except ValueError:
            self.board = d.board
        try:
            from .malta import parse_disks
            parse_disks(self.disks)
        except ValueError:
            self.disks = d.disks
        for name in ("tx_delay", "rx_delay", "step_limit"):
            if getattr(self, name) < 0:
                setattr(self, name, getattr(d, name))
        return self

    def caches(self, *, core: int = 0, shared: bool = False) -> Caches | None:
        """The cache hierarchy these options describe (``None``: no
        caches and no write buffer).  ``shared``: a core of a CPS, whose L2
        / L3 belong to the Coherence Manager (:meth:`shared_levels`);
        ``core`` varies the undefined contents after reset."""
        i, d = Geometry.parse(self.icache), Geometry.parse(self.dcache)
        l2, l3 = self.shared_levels()
        if i is None and d is None and l2 is None and not self.write_buffer:
            return None
        if shared:
            l2 = l3 = None
        return Caches(i, d, parity=self.cache_parity,
                      hardware_init=self.cache_init == "hardware",
                      l2=l2, l3=l3, ecc=self.cache_ecc,
                      write_buffer=self.write_buffer, seed=0x24C + core)

    def shared_levels(self) -> tuple[Geometry | None, Geometry | None]:
        """The L2 / L3 geometries (an L3 needs an L2)."""
        l2, l3 = Geometry.parse(self.l2, 2), Geometry.parse(self.l3, 3)
        return l2, (l3 if l2 is not None else None)

    def make_mmu(self):
        """A fresh MMU as these options describe it."""
        return make_mmu(self.mmu)

    def make_board(self, *, big_endian: bool = True):
        """A fresh board as these options describe it (None: flat)."""
        return make_board(self.board, big_endian=big_endian)

    def segment_bases(self) -> dict[str, int] | None:
        """The assembler's default segments on this board (None: MARS)."""
        from .malta import SEGMENT_BASES, USER_SEGMENT_BASES
        if self.board.split(":")[0] != "malta":
            return None
        return dict(USER_SEGMENT_BASES if self.boot == "user" else SEGMENT_BASES)

    def assembler_differs(self, other: "Options") -> bool:
        """Do these options assemble a source differently from ``other``?"""
        return (self.pseudo, self.reorder, self.segment_bases(), self.c_optimize,
                self.c_isa, self.fpu, self.arch, self.nms, self.relax, self.rv_extensions,
                self.rv_relax) != \
            (other.pseudo, other.reorder, other.segment_bases(), other.c_optimize, other.c_isa,
             other.fpu, other.arch, other.nms, other.relax, other.rv_extensions,
             other.rv_relax)
