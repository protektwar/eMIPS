"""Performance counters (PLAN M10 phase 1).

References: *MIPS32 Architecture For Programmers Volume III* (MD00090,
Rev. 6.02) 9.61 "Performance Counter Register (CP0 Register 25)" and
9.24 Cause.PCI, 9.15 IntCtl.IPPCI; the **24K**'s counters (MD00343 6.2.35,
Tables 6.44-6.47; Config1.PC, 6.2.21).

Two counters, as the 24K: PerfCtl0 / PerfCnt0 at CP0 25 select 0 / 1 and
PerfCtl1 / PerfCnt1 at select 2 / 3 (PerfCtl0.M = 1).  A control register
holds Event (bits 11:5, the 24K's 7 bits), IE and the mode bits U, S, K,
EXL; a counter adds one per event::

    PerfCtl  | M | 0 ............ | Event (11:5) | IE | U | S | K | EXL |

Each counter has its own list of events (Table 6.46: event 2 is "branch
instructions" for counter 0 and "branch mispredictions" for counter 1).
An event marked *mode* in the table counts only in the modes whose bit is
set: U user, S supervisor, K kernel with EXL = ERL = 0, EXL with EXL = 1
and ERL = 0 (nothing with ERL = 1).  When bit 31 of a counter is 1 and its
IE is set, the counter interrupt is pending: Cause.PCI (Release 2+), ORed
into hardware interrupt 5 = Cause.IP7 (IntCtl.IPPCI = 7, the line the
timer uses; Release 1: "combined with hardware interrupt 5").  Writing
the counter with bit 31 = 0, or the counter wrapping to 0, clears it.

Where the events come from:

* **instruction completion** events (instructions, branches, loads, ...)
  are counted when the instruction completes — at once in the
  instruction-level Machine, at WB in the pipeline — in the mode it
  executed in (:meth:`Counters.retire`);
* **cache, TLB and exception** events come from the machine's own
  counters (cache statistics, micro-TLB counters, ...) as they change
  (:meth:`Counters.tick`, every clock);
* **stall** events come from the pipeline model, cycle by cycle
  (:meth:`Counters.add` from :mod:`emips.pipeline`).

Events of a structure a model does not have count nothing there: branch
mispredictions, the return prediction stack, the skewed ALU's stalls,
the fill-store buffer and load queue occupancy, the IFU fill buffer
come from the 24K pipeline model (``--model 24k``); the read latency
events from the memory timing (a pipeline with caches).  Not modelled
(count nothing): replay traps, FSB index conflicts, IFU fill-buffer-full
refetches.  CorExtend, CP2 and the scratchpads never happen on this
core.  eMIPS choices (``sim.DEVIATIONS`` "performance counters"):
counters and control fields reset to 0 (UNDEFINED: nothing counts until
software sets a mode bit); an event not marked *mode* counts in every mode
once any of U / S / K / EXL is set.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import cp0 as C

M32 = 0xFFFFFFFF
PERFCTL = ((25, 0), (25, 2))
PERFCNT = ((25, 1), (25, 3))
REGISTERS = frozenset(PERFCTL + PERFCNT)
NAMES = {(25, 0): "PerfCtl0", (25, 1): "PerfCnt0", (25, 2): "PerfCtl1", (25, 3): "PerfCnt1"}
CTL_M = 1 << 31
CTL_IE = 1 << 4
CTL_U, CTL_S, CTL_K, CTL_EXL = 1 << 3, 1 << 2, 1 << 1, 1
CTL_MODES = CTL_U | CTL_S | CTL_K | CTL_EXL
#: the 24K's writable control bits: Event 11:5, IE, U, S, K, EXL
CTL_MASK = (0x7F << 5) | CTL_IE | CTL_MODES
#: Cause.PCI (Release 2+), Config1.PC
CA_PCI = 1 << 26
CFG1_PC = 1 << 4
#: IntCtl.IPPCI: the counter interrupt shares hardware interrupt 5 (IP7)
IPPCI = 7


@dataclass(frozen=True)
class Event:
    key: str          # what eMIPS counts ("" = nothing in this core)
    name: str         # the 24K's name (Table 6.47)
    mode: bool        # counts only in the enabled modes (Table 6.46 "Mode")
    source: str = ""  # where eMIPS gets it


def _events() -> tuple[dict[int, Event], dict[int, Event]]:
    """Table 6.46 of the 24K: ``(counter 0 events, counter 1 events)``."""
    I = "instruction completion"            # noqa: E741
    M24 = "the 24K pipeline model"
    X = "not on this core"
    rows = {
        0: (("cycles", "Cycles", False, "the clock"), ("cycles", "Cycles", False, "the clock")),
        1: (("instructions", "Instructions completed", True, I),
            ("instructions", "Instructions completed", True, I)),
        2: (("branch", "Branch instructions", True, I),
            ("mispredict", "Branch mispredictions", True, M24)),
        3: (("jr31", "JR r31 (return) instructions", True, I),
            ("jr31_mispredict", "JR r31 mispredictions", True, M24)),
        4: (("jr", "JR (not r31) and JALR instructions", True, I),
            ("jr31_unpredicted", "JR r31 not predicted", True, M24)),
        5: (("itlb", "ITLB accesses", True, "the ITLB"), ("itlb_miss", "ITLB misses", True,
                                                           "the ITLB")),
        6: (("dtlb", "DTLB accesses", True, "the DTLB"), ("dtlb_miss", "DTLB misses", True,
                                                           "the DTLB")),
        7: (("jtlb_i", "JTLB instruction accesses", True, "the TLB"),
            ("jtlb_i_miss", "JTLB instruction misses", True, "the TLB")),
        8: (("jtlb_d", "JTLB data accesses", True, "the TLB"),
            ("jtlb_d_miss", "JTLB data misses", True, "the TLB")),
        9: (("icache", "Instruction cache accesses", True, "the I-cache"),
            ("icache_miss", "Instruction cache misses", True, "the I-cache")),
        10: (("dcache", "Data cache accesses", True, "the D-cache"),
             ("dcache_wb", "Data cache writebacks", True, "the D-cache")),
        11: (("dcache_miss", "Data cache misses", True, "the D-cache"),
             ("dcache_miss", "Data cache misses", True, "the D-cache")),
        13: (("store_miss", "Store misses", True, "the D-cache"),
             ("load_miss", "Load misses", True, "the D-cache")),
        14: (("integer", "Integer instructions completed", True, I),
             ("fpu", "FPU instructions completed", True, I)),
        15: (("load", "Loads completed", True, I), ("store", "Stores completed", True, I)),
        16: (("j", "J/JAL completed", True, I), ("mips16", "MIPS16 instructions completed",
                                                True, I)),
        17: (("nop", "No-ops completed", True, I),
             ("muldiv", "Integer multiply/divide completed", True, I)),
        18: (("stall", "Stall cycles", False, "the pipeline"),
             ("replay", "Replay traps (other than uTLB)", True, "not modelled")),
        19: (("sc", "SC instructions completed", True, I),
             ("sc_fail", "SC instructions failed", True, I)),
        20: (("pref", "Prefetch instructions completed", True, I),
             ("pref_hit", "Prefetch instructions completed with cache hit", True, I)),
        21: (("l2_wb", "L2 cache writebacks", False, "the L2"),
             ("l2", "L2 cache accesses", False, "the L2")),
        22: (("l2_miss", "L2 cache misses", False, "the L2"),
             ("l2_corrected", "L2 cache single bit errors corrected", False, "the L2")),
        23: (("exceptions", "Exceptions taken", True, "exception entry"), None),
        24: (("cache_fixup", "Cache fixup", True, "the pipeline (DTLB refills)"), None),
        25: (("ifu_stall", "IFU stall cycles", False, "the pipeline"),
             ("alu_stall", "ALU stall cycles", False, "the pipeline")),
        28: (None, ("", "Impl. specific Cp2 event", True, X)),
        29: (("", "Impl. specific ISPRAM event", True, X), ("", "Impl. specific DSPRAM event",
                                                              True, X)),
        30: (("", "Impl. specific CorExtend event", True, X), None),
        33: (("uncached_load", "Uncached loads", True, I),
             ("uncached_store", "Uncached stores", True, I)),
        35: (("", "CP2 arithmetic instructions completed", False, X),
             ("", "CP2 to/from instructions completed", True, X)),
        37: (("icache_stall", "I$ miss stall cycles", True, "the pipeline"),
             ("dcache_stall", "D$ miss stall cycles", True, "the pipeline")),
        38: (("sync_stall", "SYNC stall cycles", True, "the pipeline"),
             ("fsb_conflict", "FSB index conflicts", True, "not modelled")),
        39: (("dcache_miss_cycles", "D$ miss cycles", False, M24),
             ("l2_miss_cycles", "L2 miss cycles", False, M24)),
        40: (("uncached_stall", "Uncached stall cycles", True, "the pipeline"), None),
        41: (("mdu_stall", "MDU stall cycles", True, "the pipeline"),
             ("fpu_stall", "FPU stall cycles", True, "the pipeline")),
        42: (("", "CP2 stall cycles", True, X), ("", "CorExtend stall cycles", True, X)),
        43: (("", "ISPRAM stall cycles", True, X), ("", "DSPRAM stall cycles", True, X)),
        44: (("cacheop_stall", "CACHE instruction stall cycles", False, "the pipeline"), None),
        45: (("load_use_stall", "Load to use stall cycles", True, "the pipeline"),
             ("alu_agen_stall", "ALU to AGEN stall cycles", True, M24)),
        46: (("other_stall", "Other interlock stall cycles", True, M24),
             ("mispredict_stall", "Branch mispredict stall cycles", False, M24)),
        48: (("ifu_refetch", "IFU fill buffer full refetches", True, "not modelled"),
             ("ifu_fb", "IFU fill buffer allocated", False, M24)),
        49: (("", "EJTAG instruction triggerpoints", True, X),
             ("", "EJTAG data triggerpoints", True, X)),
        50: (("fsb_low", "FSB < 1/4 full", False, M24), ("fsb_mid", "FSB 1/4-1/2 full", False,
                                                          M24)),
        51: (("fsb_high", "FSB > 1/2 full", False, M24),
             ("fsb_full_stall", "FSB full pipeline stall cycles", False, "the pipeline")),
        52: (("ldq_low", "LDQ < 1/4 full", False, M24), ("ldq_mid", "LDQ 1/4-1/2 full", False,
                                                          M24)),
        53: (("ldq_high", "LDQ > 1/2 full", False, M24),
             ("ldq_full_stall", "LDQ full pipeline stall cycles", False, M24)),
        54: (("wbb_low", "WBB < 1/4 full", False, "the write buffer"),
             ("wbb_mid", "WBB 1/4-1/2 full", False, "the write buffer")),
        55: (("wbb_high", "WBB > 1/2 full", False, "the write buffer"),
             ("wbb_full_stall", "WBB full pipeline stall cycles", False, "the pipeline")),
        61: (("rr_latency", "Request latency to read response", False, "the memory timing"),
             ("rr_count", "Request count for RR latency", False, "the memory timing")),
    }
    for n in range(64, 68):
        rows[n] = (("", f"SI_PCEvent[{2 * (n - 64)}]", False, "no system events"),
                   ("", f"SI_PCEvent[{2 * (n - 64) + 1}]", False, "no system events"))
    out: tuple[dict[int, Event], dict[int, Event]] = ({}, {})
    for n, pair in rows.items():
        for i, row in enumerate(pair):
            if row is not None:
                out[i][n] = Event(*row)
    return out


#: Table 6.46: event number -> Event, for counter 0 and counter 1
EVENTS = _events()

#: events counted from the machine's own counters (read every clock)
SAMPLED = frozenset({"itlb", "itlb_miss", "dtlb", "dtlb_miss", "jtlb_i", "jtlb_i_miss",
                     "jtlb_d", "jtlb_d_miss", "icache", "icache_miss", "dcache",
                     "dcache_wb", "dcache_miss", "load_miss", "store_miss", "l2_wb", "l2",
                     "l2_miss", "l2_corrected", "rr_latency", "rr_count"})
#: events counted by occupancy, every clock
OCCUPANCY = frozenset({"wbb_low", "wbb_mid", "wbb_high"})


def mode_bit(status: int) -> int:
    """The PerfCtl mode bit of the current mode (0 with ERL = 1)."""
    if status & C.ST_ERL:
        return 0
    if status & C.ST_EXL:
        return CTL_EXL
    return (CTL_K, CTL_S, CTL_U, CTL_U)[C.ksu(status)]


# ------------------------------------------------------------- the machine's counters

def _kinds(cache, *kinds: str, hits: bool = True, misses: bool = True) -> int:
    if cache is None:
        return 0
    k = cache.stats.kinds
    return sum((k[x][0] if hits else 0) + (k[x][1] if misses else 0) for x in kinds if x in k)


def _tlb(m, what: str) -> int:
    mmu = m.mmu
    itlb = getattr(mmu, "itlb", None)
    if itlb is None:
        return 0
    dtlb, st = mmu.dtlb, mmu.stats
    return {"itlb": itlb.hits + itlb.misses, "itlb_miss": itlb.misses,
            "dtlb": dtlb.hits + dtlb.misses + st.unmapped_data, "dtlb_miss": dtlb.misses,
            "jtlb_i": itlb.misses, "jtlb_i_miss": st.jtlb_misses["fetch"],
            "jtlb_d": dtlb.misses, "jtlb_d_miss": st.jtlb_misses["data"]}[what]


def sample(m, key: str) -> int:
    """The machine's running total behind a sampled event."""
    if key[:4] in ("itlb", "dtlb", "jtlb"):
        return _tlb(m, key)
    c = m.caches
    if c is None:
        return 0
    if key == "rr_latency":
        return c.timing.read_latency
    if key == "rr_count":
        return c.timing.read_requests
    i, d, l2 = c.i, c.d, c.l2
    if key == "icache":
        return _kinds(i, "fetch")
    if key == "icache_miss":
        return _kinds(i, "fetch", hits=False)
    if key == "dcache":
        return _kinds(d, "load", "store")
    if key == "dcache_miss":
        return _kinds(d, "load", "store", hits=False)
    if key == "load_miss":
        return _kinds(d, "load", hits=False)
    if key == "store_miss":
        return _kinds(d, "store", hits=False)
    if key == "dcache_wb":
        return d.writebacks if d is not None else 0
    if l2 is None:
        return 0
    if key == "l2":
        return l2.stats.accesses
    if key == "l2_miss":
        return l2.stats.misses
    if key == "l2_wb":
        return l2.writebacks
    if key == "l2_corrected":
        return getattr(l2, "corrected", 0)
    raise KeyError(key)


# ------------------------------------------------------------- instruction classes

_MULDIV = frozenset({"MULT", "MULTU", "DIV", "DIVU", "MADD", "MADDU", "MSUB", "MSUBU", "MUL",
                     "MUL@R6", "MUH", "MULU", "MUHU", "DIV@R6", "MOD", "DIVU@R6", "MODU"})
_SC = frozenset({"SC", "SCE", "SC@R6"})
_PREF = frozenset({"PREF", "PREF@R6", "PREFE", "PREFX"})
_FP_MEM = frozenset({"LWC1", "LDC1", "SWC1", "SDC1", "LWXC1", "LDXC1", "LUXC1", "SWXC1",
                     "SDXC1", "SUXC1", "MOVF", "MOVT"})


def _static(dec) -> tuple[str, ...]:
    """The completion events an instruction always has (cached per word)."""
    from .pipeline import _RT_DEST
    from .sim import FPU_FAMILIES
    from .syntax import kind_of, operand_spec
    e = dec.entry
    name, fam = e["name"], e["family"]
    w = dec.word
    out = ["instructions"]
    if e.get("isa") in ("mips16e", "micromips"):    # no MIPS32 twin (emips.mips16 / micromips)
        out.append("integer")
        if fam in ("branch", "branch_fpu", "branch_compact"):
            out.append("branch")
        form = name.split(".", 1)[1]
        if form in ("JRRA", "JRCRA"):
            out.append("jr31")
        elif form in ("JR", "JALR", "JRC", "JALRC", "JR16", "JRC16", "JALR16", "JALRS16",
                      "JALRC16", "JALRS", "JALR.HB", "JALRS.HB", "JALRC.HB"):
            out.append("jr")
        if form in ("JAL", "JALX", "JALS", "J"):
            out.append("j")
        if fam == "load":
            out.append("load")
        if fam == "store":
            out.append("store")
        return tuple(out)
    fpu = fam in FPU_FAMILIES or name in _FP_MEM
    out.append("fpu" if fpu else "integer")
    if fam in ("branch", "branch_fpu", "branch_compact") or name.startswith("BC2"):
        out.append("branch")
    if name in ("JR", "JR.HB", "JALR", "JALR.HB", "JIC", "JIALC"):
        # a return: JR $31 (Release 6: JALR $zero, $31)
        ret = (w >> 21) & 31 == 31 and (name.startswith("JR") or (w >> 11) & 31 == 0)
        out.append("jr31" if ret else "jr")
    if name in ("J", "JAL"):
        out.append("j")
    if fam in ("load",) or name in ("LWC1", "LDC1", "LWXC1", "LDXC1", "LUXC1", "LL", "LLE",
                                     "LL@R6", "LWPC"):
        out.append("load")
    if fam in ("store",) or name in ("SWC1", "SDC1", "SWXC1", "SDXC1", "SUXC1") or name in _SC:
        out.append("store")
    if name in _MULDIV:
        out.append("muldiv")
    if name in _SC:
        out.append("sc")
    if name in ("NOP", "SSNOP", "EHB", "PAUSE"):   # sll $zero, $zero, n
        out.append("nop")
    for item in operand_spec(e):                   # a GPR destination that is $zero
        kind = kind_of(item)[0]
        field = {"rd": 11, "rsd": 21}.get(kind)
        if kind == "rt" and (fam == "imm" or name in _RT_DEST):
            field = 16
        if field is not None and (w >> field) & 31 == 0:
            out.append("nop")
            break
    return tuple(dict.fromkeys(out))


#: nanoMIPS's multiply / divide pages (no HI / LO: MD01247)
_NANO_MULDIV = frozenset(("MUL", "MUH", "MULU", "MUHU", "DIV", "DIVU", "MOD", "MODU"))


def _static_nano(i) -> tuple[str, ...]:
    """:func:`_static` of a nanoMIPS instruction (emips.simnano, PLAN M23):
    every PC-relative branch (BC, BALC, BRSC ... too) is a branch — there
    are no J-type jumps —, JRC / JALRC jumps to a register, a return
    (``jrc $ra``, RESTORE.JRC) jr31; "nop" = NOP / EHB / PAUSE or a $0
    destination."""
    from .simnano import _FAMILY, _USES, returns
    page = i.row.name
    fam = _FAMILY[page]
    out = ["instructions", "integer"]
    if fam == "branch_compact" or page in ("BC", "BALC", "MOVE.BALC", "BRSC", "BALRSC"):
        out.append("branch")
    if returns(i):
        out.append("jr31")
    elif page in ("JRC", "JALRC", "JALRC.HB"):
        out.append("jr")
    if fam == "load":
        out.append("load")
    if fam == "store":
        out.append("store")
    if page in _NANO_MULDIV:
        out.append("muldiv")
    if page == "SC":
        out.append("sc")
    writes = _USES[page][1](i.f)
    if page in ("NOP", "EHB", "PAUSE") or (writes and 0 in writes and fam != "store"):
        out.append("nop")
    return tuple(out)


# ------------------------------------------------------------- the counters

class Counters:
    """The two counter pairs of one core (``Machine.perf``)."""

    def __init__(self, m) -> None:
        self.m = m
        #: event key -> [(counter, mode-dependent, mode bits)]
        self.keys: dict[str, list[tuple[int, bool, int]]] = {}
        #: anything to count (the hooks look at this first)
        self.armed = False
        #: the counter interrupt is pending (Cause.PCI)
        self.pci = False
        self._prev: dict[str, int] = {}
        self._static: dict[int, tuple[str, ...]] = {}
        #: the current instruction's dynamic events (uncached accesses, PREF)
        self.pending: list[str] = []

    def reset(self) -> None:
        """Reset: the control registers 0 (no mode: nothing counts), PerfCtl0.M."""
        cp = self.m.cp0
        cp[PERFCTL[0]]["value"] = CTL_M
        cp[PERFCTL[1]]["value"] = 0
        for key in PERFCNT:
            cp[key]["value"] = 0
        self.m.cp0[C.CONFIG1]["value"] |= CFG1_PC
        self._rebuild()

    # -- MTC0 -----------------------------------------------------------------------

    def cp0_write(self, key: tuple[int, int], value: int) -> None:
        cp = self.m.cp0
        if key in PERFCTL:
            mask = CTL_MASK if self.m.mmu.supervisor else CTL_MASK & ~CTL_S
            cell = cp[key]
            cell["value"] = (cell["value"] & ~CTL_MASK) | (value & mask)
            self._rebuild()
        else:
            cp[key]["value"] = value & M32
        self._interrupt()

    def control(self, i: int) -> int:
        return self.m.cp0[PERFCTL[i]]["value"]

    def event(self, i: int) -> Event | None:
        return EVENTS[i].get((self.control(i) >> 5) & 0x7F)

    def _rebuild(self) -> None:
        keys: dict[str, list[tuple[int, bool, int]]] = {}
        for i in range(2):
            ctl = self.control(i)
            ev = self.event(i)
            if ev is None or not ev.key or not ctl & CTL_MODES:
                continue
            keys.setdefault(ev.key, []).append((i, ev.mode, ctl & CTL_MODES))
        for k in keys:
            if k in SAMPLED and k not in self.keys:
                self._prev[k] = sample(self.m, k)
        self.keys = keys
        self.armed = bool(keys)

    # -- counting ---------------------------------------------------------------------

    def add(self, key: str, n: int = 1, status: int | None = None) -> None:
        """``n`` events ``key`` (in the mode of ``status``, default now)."""
        hits = self.keys.get(key)
        if not hits or n <= 0:
            return
        bit = mode_bit(self.m.status if status is None else status)
        cp = self.m.cp0
        changed = False
        for i, mode, modes in hits:
            if mode and not modes & bit:
                continue
            cell = cp[PERFCNT[i]]
            cell["value"] = (cell["value"] + n) & M32
            changed = True
        if changed:
            self._interrupt()

    def tick(self, cycles: int) -> None:
        """``cycles`` clocks passed: cycles, the machine's counters, the
        write buffer occupancy."""
        keys = self.keys
        if "cycles" in keys:
            self.add("cycles", cycles)
        for k in keys:
            if k in SAMPLED:
                now = sample(self.m, k)
                d = now - self._prev.get(k, 0)
                self._prev[k] = now
                self.add(k, d if d >= 0 else now)
            elif k in OCCUPANCY:
                self._occupancy(k, cycles)

    def _occupancy(self, key: str, cycles: int) -> None:
        """Write buffer occupancy bins of the 24K (4 entries: < 1/4 = 0,
        1/4-1/2 = 1-2, > 1/2 = 3+)."""
        c = self.m.caches
        wbb = c.wbb if c is not None else None
        n = 0 if wbb is None else len(wbb.entries)
        bin_ = "wbb_low" if n == 0 else "wbb_mid" if n <= 2 else "wbb_high"
        if bin_ == key:
            self.add(key, cycles)

    # -- instruction completion -----------------------------------------------------------

    def classify(self, dec, word: int) -> tuple[str, ...]:
        """The completion events of the instruction just executed (with its
        dynamic ones: SC failed, PREF, uncached accesses)."""
        m = self.m
        if m.nano:                              # nanoMIPS: 2, 4 or 6 bytes
            key = (word, dec.name, m._insn_size)
            st = self._static.get(key)
            if st is None:
                st = self._static[key] = _static_nano(m._nm)
            rt = m._nm.f.get("rt", 0)
        else:
            key = word if "isa" not in dec.entry else (word, dec.name)
            st = self._static.get(key)
            if st is None:
                st = self._static[key] = _static(dec)
            rt = (word >> 16) & 31
        extra = self.pending
        if m.isa_mode and m.compressed == "mips16e":    # (the twins: by mode, not word)
            extra = extra + ["mips16"]
        if "sc" in st and m.read_reg(rt) == 0:
            extra = extra + ["sc_fail"]
        self.pending = []
        return st + tuple(extra) if extra else st

    def retire(self, events: tuple[str, ...], status: int) -> None:
        """An instruction completed: its events, in the mode it ran in."""
        for k in events:
            if k in self.keys:
                self.add(k, 1, status)

    # -- the interrupt ----------------------------------------------------------------------

    def _interrupt(self) -> None:
        cp = self.m.cp0
        pci = any(cp[PERFCNT[i]]["value"] >> 31 and self.control(i) & CTL_IE for i in range(2))
        if pci == self.pci:
            return
        self.pci = pci
        if self.m.release != "R1":
            cause = cp[C.CAUSE]["value"]
            cp[C.CAUSE]["value"] = cause | CA_PCI if pci else cause & ~CA_PCI
        self.m._update_ip()


# ------------------------------------------------------------- views and setup

def find(name: str) -> list[tuple[int, int]]:
    """``(counter, event number)`` of an event name (``dcache_miss``) or
    number (``11`` = both counters)."""
    name = name.strip().lower()
    if name.isdigit():
        n = int(name)
        return [(i, n) for i in range(2) if n in EVENTS[i]]
    return [(i, n) for i in range(2) for n, ev in EVENTS[i].items() if ev.key == name]


def configure(m, i: int, event: int, modes: str = "uk", ie: bool = False) -> None:
    """Set counter ``i`` as software does with MTC0: PerfCtl = event and
    mode bits (``u`` user, ``s`` supervisor, ``k`` kernel, ``x`` EXL), the
    counter 0."""
    bits = {"u": CTL_U, "s": CTL_S, "k": CTL_K, "x": CTL_EXL}
    ctl = (event << 5) | sum(bits[c] for c in set(modes)) | (CTL_IE if ie else 0)
    m.cp0_write(*PERFCTL[i], ctl)
    m.cp0_write(*PERFCNT[i], 0)


def modes_text(ctl: int) -> str:
    out = "".join(c for c, b in (("u", CTL_U), ("s", CTL_S), ("k", CTL_K), ("x", CTL_EXL))
                  if ctl & b)
    return out or "off"


def rows(m) -> list[tuple[int, int, int, Event | None]]:
    """``(counter, PerfCtl, PerfCnt, event)`` of both counters."""
    return [(i, m.cp0_read(*PERFCTL[i]), m.cp0_read(*PERFCNT[i]), m.perf.event(i))
            for i in range(2)]


def format_counters(m) -> str:
    out = []
    for i, ctl, cnt, ev in rows(m):
        num = (ctl >> 5) & 0x7F
        what = ev.name if ev is not None else "reserved"
        if ev is not None and not ev.key:
            what += " (never happens on this core)"
        elif ev is not None and ev.source == "the 24K pipeline model":
            what += " (the 24K pipeline model counts it)"
        elif ev is not None and ev.source == "not modelled":
            what += " (not modelled: counts nothing)"
        out.append(f"counter {i}: {cnt:>10d}  event {num:<3d} {what}; modes "
                   f"{modes_text(ctl)}{', interrupt on overflow' if ctl & CTL_IE else ''}")
    if m.perf.pci:
        out.append("Cause.PCI = 1: a counter interrupt is pending (IP7)")
    return "\n".join(out)


def format_events(counter: int | None = None) -> str:
    out = []
    for i in (range(2) if counter is None else [counter]):
        out.append(f"counter {i}:")
        for n, ev in sorted(EVENTS[i].items()):
            if not ev.key:
                continue
            tag = "" if ev.mode else "  (any mode)"
            later = ("  [24K model]" if ev.source == "the 24K pipeline model" else
                     "  [not modelled]" if ev.source == "not modelled" else "")
            out.append(f"  {n:3d} {ev.key:<20s} {ev.name}{tag}{later}")
    return "\n".join(out)
