"""Simulator options (GUI Configure > Settings, CLI flags), free of Qt.

:class:`Options` holds everything the Settings dialog edits besides the
release / byte order / mode / forwarding (which live on :class:`Session`
and the toolbar).  :meth:`Options.to_strings` / :meth:`Options.from_strings`
turn it into flat ``{key: str}`` pairs for ``QSettings``; unknown or
malformed values fall back to the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from .cache import Caches, Geometry
from .malta import make_board
from .mmu import make_mmu
from .pipeline import LATENCY
from .timing import MEMORY_TIMING

BOOT_MODES = ("kernel", "user", "reset")
#: the C compiler's optimisation levels (-O...)
C_OPTIMIZE = ("0", "1", "2", "3", "s")
EXCEPTION_MODES = ("cp0", "stop")
#: Pipeline models (pipeline mode): the textbook 5-stage pipeline, the MIPS 24K's
PIPELINE_MODELS = ("5-stage", "24k")
#: Cache contents after reset: "random" (undefined, as on silicon) or
#: "hardware" (invalid: the 24K's Config7.HCI, simulation models)
CACHE_INIT_MODES = ("random", "hardware")
#: the most cores a system has (emips.cps)
MAX_CORES = 4
#: the most GPR sets (Vol. III: SRSCtl.HSS is 4 bits)
MAX_SHADOW_SETS = 16


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
    # processor
    #: Count increments every N cycles (CCRes: 1 or 2)
    count_rate: int = 2
    #: pipeline latencies (cycles), see :data:`emips.pipeline.LATENCY`
    latency: dict[str, int] = field(default_factory=lambda: dict(LATENCY))
    #: the pipeline model of pipeline mode, see PIPELINE_MODELS
    model: str = "5-stage"
    #: cores (1 to 4): more than one is a Coherent Processing System
    #: (emips.cps: the GCRs, the CPC, shared memory and devices)
    cores: int = 1
    #: an I/O Coherence Unit in the CPS (GCR_CONFIG.NUMIOCU = 1): DMA on a
    #: multi-core Malta is coherent with the data caches (emips.coherence)
    iocu: bool = False
    #: GPR shadow sets (SRSCtl.HSS + 1; Release 2+): 1 = the normal GPRs
    #: only; the 24K builds 1, 2 or 4, the architecture allows up to 16
    shadow_sets: int = 4
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
        return (self.pseudo, self.reorder, self.segment_bases(), self.c_optimize) != \
            (other.pseudo, other.reorder, other.segment_bases(), other.c_optimize)
