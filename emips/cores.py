"""Core presets: real MIPS cores as their manuals let them be built (Qt-free).

A preset names one licensed core — the 24Kf, the M14K, the P5600 ... —
and says what each setting may be on it: the release it implements
(Config.AR / the manual), its compressed ISA, whether it has an FPU, the
MMUs, cache shapes, core counts, shadow sets and pipeline model its
build options offer.  :func:`apply` sets the release and every one of
those settings to the preset's own build (a choice inside the options:
the documents give no defaults), and the core's PRId.  After that the
settings stay editable within the core's options; a value outside them
makes the machine a generic eMIPS core again (``Options.core = ""``,
PRId 0) — the rule is in :mod:`emips.compat`.

What a core has that eMIPS does not model (the 34K / 1004K /
interAptiv pipelines' thread stage, the P5600's out-of-order pipeline ...) is listed
in :attr:`Core.missing`, and every preset keeps the 24K's
implementation-specific registers (Config7, the EJTAG 3.20 DCR, four
instruction / two data breakpoints): ``sim.DEVIATIONS["core presets"]``.

Sources (``docs``: the core's datasheet and Software User's Manual):
the 24K MD00346 / MD00354 / MD00343, the 34K MD00418 / MD00534, the 1004K
MD00584 / MD00622, the
interAptiv MD00903 / MD00904, the P5600 MD01024 / MD01025, the M14K
MD00666 / MD00668, the M14Kc MD00672 / MD00674, the microAptiv UC
MD00931 / MD00934, the microAptiv UP MD00942, the 4K MD00016, the nanoMIPS
I7200 MD01227 / MD01232.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .cache import Geometry
from .p8700 import EXTENSIONS as _P8700_EXTENSIONS

#: MIPS Technologies' Company ID (PRId 23:16; every manual here: 1)
COMPANY_MIPS = 1
#: the settings a preset decides (Options fields) and the release
FIELDS = ("release", "compressed", "fpu", "dsp", "msa", "mt", "mmu", "icache", "dcache", "l2",
          "l3", "cores", "vps", "shadow_sets", "model")
#: the settings a RISC-V core preset decides (PLAN M25 phase 5: the MIPS
#: settings do not apply to a RISC-V hart, emips.compat)
RV_FIELDS = ("rv_extensions", "rv_modes", "icache", "dcache", "l2", "l3", "cores", "vps",
             "model")


def _pow2(n: int) -> bool:
    return n > 0 and not n & (n - 1)


@dataclass(frozen=True)
class CacheChoice:
    """The shapes a cache may be built with: ``sizes`` (KB, 0 = no cache)
    with ``ways`` ways, plus the ``direct`` sizes built direct-mapped; or,
    with ``way_kb`` (the smallest and largest way, KB, a power of two),
    any number of ``ways`` of such ways up to ``max_kb``.  ``min_kb``: (line,
    the smallest size with that line) pairs (the I7200's L2: 32-byte lines
    from 128 KB); ``ways_kb``: (size, ways) pairs whose size has only those
    ways (the P8700's L2: 256 KB 8-way, the others 16-way)."""
    sizes: tuple[int, ...] = ()
    ways: tuple[int, ...] = (4,)
    lines: tuple[int, ...] = (32,)
    direct: tuple[int, ...] = ()
    way_kb: tuple[int, int] | None = None
    max_kb: int = 0
    min_kb: tuple[tuple[int, int], ...] = ()
    ways_kb: tuple[tuple[int, int], ...] = ()

    def allows(self, spec: str, level: int = 1) -> bool:
        try:
            g = Geometry.parse(spec, level)
        except ValueError:
            return False
        if g is None:
            return 0 in self.sizes or 0 in self.direct
        if g.line not in self.lines:
            return False
        kb, rest = divmod(g.size, 1024)
        if rest or kb < dict(self.min_kb).get(g.line, 0):
            return False
        if self.way_kb is not None:
            way, low, high = g.size // g.ways, self.way_kb[0] * 1024, self.way_kb[1] * 1024
            return (g.ways in self.ways and _pow2(way) and low <= way <= high
                    and kb <= self.max_kb)
        if g.ways == 1 and kb in self.direct:
            return True
        only = dict(self.ways_kb).get(kb)
        if only is not None:
            return g.ways == only and kb in self.sizes
        return g.ways in self.ways and kb in self.sizes and kb > 0

    def text(self) -> str:
        if self.sizes == (0,) and self.way_kb is None and not self.direct:
            return "no cache"
        if self.way_kb is not None:
            ways = "/".join(map(str, self.ways))
            return (f"0-{self.max_kb} KB, {ways}-way, {'/'.join(map(str, self.lines))}-byte "
                    f"lines")
        sizes = "/".join(map(str, self.sizes)).replace("0/", "none/", 1) \
            if self.sizes[:1] == (0,) else "/".join(map(str, self.sizes))
        out = (f"{sizes} KB, {'/'.join(map(str, self.ways))}-way, "
               f"{'/'.join(map(str, self.lines))}-byte lines")
        for line, kb in self.min_kb:
            out += f" ({line}-byte lines from {kb} KB)"
        for kb, ways in self.ways_kb:
            out += f" ({kb} KB: {ways}-way)"
        if self.direct:
            out += f" (or {'/'.join(map(str, self.direct))} KB direct-mapped)"
        return out


#: no cache at all (an SRAM interface: the M14K, the microAptiv UC)
NO_CACHE = CacheChoice(sizes=(0,))


def mmu_key(spec: str) -> str:
    """An MMU spec in one spelling: ``fixed``, ``bat``, ``tlb:N``,
    ``ftlb:N:SxW``, then ``+sc`` / ``+pw`` in that order; ``mpu:N``,
    ``fixed+mpu:N``."""
    text = (spec or "fixed").strip().lower().replace(" ", "")
    base, *flags = text.split("+")
    kind, _, rest = base.partition(":")
    parts = rest.split(":") if rest else []
    if kind == "mpu":
        return f"mpu:{parts[0] if parts and parts[0] else 16}"
    if kind == "fixed" and flags and flags[0].startswith("mpu"):
        n = flags[0].partition(":")[2]
        return f"fixed+mpu:{n or 16}"
    if kind == "tlb":
        base = f"tlb:{parts[0] if parts and parts[0] else 32}"
    elif kind == "ftlb":
        n = parts[0] if parts and parts[0] else "64"
        base = f"ftlb:{n}:{parts[1] if len(parts) > 1 else '128x4'}"
    return base + "".join(f"+{f}" for f in ("sc", "pw") if f in flags)


@dataclass(frozen=True)
class Core:
    #: the preset's name (``--core``)
    name: str
    #: what the user sees
    title: str
    release: str
    #: PRId Processor ID (15:8)
    processor_id: int
    compressed: tuple[str, ...]
    fpu: tuple[bool, ...]
    #: the DSP Module it may have: "none", "dsp" (Revision 1), "dspr2"
    dsp: tuple[str, ...]
    #: the MT Module it may have: "none" or "VPES:TCS" (emips.mt.parse_spec)
    mt: tuple[str, ...]
    #: the MMU specs it may have (:func:`mmu_key` spelling)
    mmu: tuple[str, ...]
    icache: CacheChoice
    dcache: CacheChoice
    #: None: no L2 cache
    l2: CacheChoice | None
    cores: tuple[int, ...]
    shadow_sets: tuple[int, ...]
    model: tuple[str, ...]
    #: the preset's own build (Options fields), inside the choices above
    build: dict[str, Any]
    #: where the choices come from
    source: str
    #: what the core has that eMIPS does not model
    missing: tuple[str, ...] = ()
    #: what eMIPS models differently on this core
    notes: tuple[str, ...] = field(default=())
    #: the architecture: "mips32", or "mips64" (a 64-bit core, PLAN M22)
    arch: str = "mips32"
    #: Release 6 virtual processors a core it may have (emips.vp)
    vps: tuple[int, ...] = (1,)
    #: its Coherence Manager with more than one core: "cm2" (the 1004K /
    #: interAptiv / P5600 / P6600 generation) or "cm3" (the I6400's)
    cm: str = "cm2"
    #: L1 caches of this size (KB) are built only with an MPU (the I7200's
    #: 128 KB, MD01227 Table 2); 0: none
    mpu_l1_kb: int = 0
    #: MSA it may have (PLAN M24): "none", "msa"
    msa: tuple[str, ...] = ("none",)
    #: when its element-misaligned MSA LD / ST trap (Machine.msa_ade)
    msa_ade: frozenset = frozenset()
    #: its FPU is IEEE 754-2008 (FCSR.NAN2008 / ABS2008 read-only 1) even in
    #: Release 5 (the P5600: MD01025 13.10.7)
    fpu2008: bool = False
    #: a RISC-V core (PLAN M25 phase 5): the extension sets and privilege
    #: modes it may have
    rv_extensions: tuple[str, ...] = ()
    rv_modes: tuple[str, ...] = ()

    @property
    def rv(self) -> bool:
        return self.arch in ("rv32", "rv64")

    @property
    def fields(self) -> tuple[str, ...]:
        """The settings this preset decides."""
        return RV_FIELDS if self.rv else FIELDS

    @property
    def prid(self) -> int:
        """PRId: Company ID 1 (MIPS Technologies), the Processor ID,
        Revision 0 (the revision is the silicon's: QEMU's models use 0)."""
        return (COMPANY_MIPS << 16) | (self.processor_id << 8)

    def allows(self, name: str, value) -> bool:
        """May this core be built with ``name`` = ``value``?"""
        if name not in self.fields:
            return True                         # (a RISC-V core: the MIPS settings)
        if name == "rv_extensions":
            from .riscv import parse_extensions
            try:
                return parse_extensions(value) in {parse_extensions(e)
                                                   for e in self.rv_extensions}
            except ValueError:
                return False
        if name == "release":
            return value == self.release
        if name in ("icache", "dcache"):
            return getattr(self, name).allows(value, 1)
        if name == "l2":
            if self.l2 is None:
                return _parses(value, 2) and Geometry.parse(value, 2) is None
            return self.l2.allows(value, 2)
        if name == "l3":
            return _parses(value, 3) and Geometry.parse(value, 3) is None
        if name == "mmu":
            try:
                return mmu_key(value) in self.mmu
            except (ValueError, IndexError):
                return False
        if name == "msa" and value == "auto":   # (what the program needs, within these)
            return True
        return value in getattr(self, name)

    def choices(self, name: str) -> str:
        """The options of ``name`` on this core, for a message."""
        if name == "rv_extensions":
            return " or ".join(self.rv_extensions)
        if name == "rv_modes":
            return " or ".join({"msu": "M, S and U-mode (msu)", "mu": "M and U-mode (mu)",
                                "m": "M-mode only (m)"}[x] for x in self.rv_modes)
        if name == "release":
            return f"Release {self.release[1:]}"
        if name in ("icache", "dcache"):
            return getattr(self, name).text()
        if name == "l2":
            return "no L2" if self.l2 is None else self.l2.text()
        if name == "l3":
            return "no L3"
        if name == "fpu":
            return {(True,): "an FPU", (False,): "no FPU"}.get(self.fpu, "an FPU or none")
        if name == "dsp":
            if self.arch == "nanomips":
                return " or ".join({"none": "no DSP ASE", "dspr2": "the nanoMIPS DSP ASE"}[d]
                                   for d in self.dsp)
            return " or ".join({"none": "no DSP Module", "dsp": "the DSP Module (Revision 1)",
                                "dspr2": "the DSP Module Revision 2"}[d] for d in self.dsp)
        if name == "msa":
            return " or ".join({"none": "no MSA", "msa": "MSA"}[x] for x in self.msa)
        if name == "mt":
            if self.mt == ("none",):
                return "no MT Module"
            if self.arch == "nanomips":
                vpes = sorted({int(x.split(":")[0]) for x in self.mt})
                tcs = sorted({int(x.split(":")[1]) for x in self.mt})
                return (f"the nanoMIPS MT ASE: {'/'.join(map(str, vpes))} VPE(s), "
                        f"{tcs[0]}-{tcs[-1]} TCs")
            vpes = sorted({int(x.split(":")[0]) for x in self.mt if x != "none"})
            tcs = sorted({int(x.split(":")[1]) for x in self.mt if x != "none"})
            one = all(int(a) == int(b) for a, b in (x.split(":") for x in self.mt if x != "none"))
            return ("the MT Module: " + "/".join(map(str, vpes)) + " VPE(s), "
                    + ("one TC per VPE" if one else f"{tcs[0]}-{tcs[-1]} TCs"))
        if name == "mmu":                     # (MPU region counts in one item)
            out: list[str] = []
            for spec in self.mmu:
                head, _, n = spec.rpartition(":")
                if "mpu" in head and out and out[-1].startswith(head + ":"):
                    out[-1] += "/" + n
                else:
                    out.append(spec)
            return ", ".join(out)
        if name == "vps" and self.rv:
            return "/".join(map(str, self.vps)) + " harts a core"
        if name == "vps":
            return ("one VP a core (no Release 6 multi-threading)" if self.vps == (1,)
                    else "/".join(map(str, self.vps)) + " VPs a core (Release 6, a CM3)")
        return ", ".join(str(v) for v in getattr(self, name))


def _parses(spec: str, level: int) -> bool:
    try:
        Geometry.parse(spec, level)
    except ValueError:
        return False
    return True


# -- the cores ---------------------------------------------------------------

_L1_24K = CacheChoice(sizes=(0, 8, 16, 32, 64))
_L2_SOCIT = CacheChoice(sizes=(0, 128, 256, 512, 1024, 2048, 4096, 8192), ways=(8,),
                        lines=(32, 64))
_TLB_24K = ("tlb:16", "tlb:32", "tlb:64", "fixed")
#: the I7200's MPU region counts (MD01232 Table 23)
_I7200_REGIONS = (8, 12, 16, 20, 24, 28, 32)
_MCU_L1 = CacheChoice(sizes=(0,), ways=(1, 2, 3, 4), lines=(16,), way_kb=(1, 16), max_kb=64)
_24K_MISSING = ("CorExtend (UDI)",)
#: the 34K's MT build options (MD00534 1.1: 1 or 2 VPEs, 1-9 TCs)
_MT_SPECS = tuple(f"{v}:{t}" for v in (1, 2) for t in range(v, 10))
_CPS_NOTE = ("eMIPS builds a CPS (the CM, CPC, GIC) only with 2 or more cores, and at "
             "most 6",)


#: the 5K's caches (MD00109 Table 6: 4 / 8 / 16 KB ways, direct-mapped to
#: 4-way, at most 64 KB, 32-byte lines)
_L1_5K = CacheChoice(sizes=(0,), ways=(1, 2, 3, 4), lines=(32,), way_kb=(4, 16), max_kb=64)
#: the I6400's L1 (MD01196 3.4: 32 or 64 KB, 4-way, 64-byte lines)
_L1_I6400 = CacheChoice(sizes=(32, 64), ways=(4,), lines=(64,))
#: the P8700's L1 I / D (MD01502 Table 3.1: 32 / 64 KB, 4-way, 64-byte lines)
_L1_P8700 = CacheChoice(sizes=(32, 64), ways=(4,), lines=(64,))


def _5k(name: str, title: str, fpu: bool) -> Core:
    """The MIPS64 5Kc / 5Kf (MD00109: the 5Kf datasheet; the 5Kc is the 5K
    without an FPU — GCC's 5k.md)."""
    return Core(
        name, title, "R1", 0x81, ("none",), (fpu,), ("none",), ("none",),
        ("tlb:16", "tlb:32", "tlb:48", "fixed"), _L1_5K, _L1_5K, None, (1,), (1,),
        ("5-stage",),
        build=dict(compressed="none", fpu=fpu, dsp="none", mt="none", mmu="tlb:32",
                   icache="32K:4:32", dcache="32K:4:32", l2="none", l3="none", cores=1,
                   shadow_sets=1, model="5-stage"),
        source="MD00109 (the 5Kf datasheet 01.03: MIPS64 Release 1, a 16 / 32 / 48 "
               "dual-entry JTLB or the FMT, caches 0-64 KB of 4-16 KB ways, direct-mapped "
               "to 4-way, 32-byte lines: Table 6; DMULT 9 / DDIV 67 cycles: Table 3), "
               "GCC's 5k.md (the 5Kc: no FPU), QEMU's 5Kc / 5Kf models (PRId 0x81)",
        missing=("the 5K's six-stage I D R E M W pipeline (eMIPS: the textbook 5-stage)"
                 + (" and the 5Kf's dual issue of FPU and integer instructions" if fpu
                    else ""),
                 "42-bit virtual / 36-bit physical addresses (eMIPS: 40 / 32)",
                 "the write-through, write-allocate cache attribute (CCA 1)",
                 "the COP2 interface", "MDMX"),
        arch="mips64")


def _24k(name: str, title: str, fpu: bool) -> Core:
    return Core(
        name, title, "R2", 0x93, ("mips16e",), (fpu,), ("none",), ("none",), _TLB_24K, _L1_24K,
        _L1_24K, _L2_SOCIT, (1,), (1, 2, 4), ("24k",),
        build=dict(compressed="mips16e", fpu=fpu, dsp="none", mt="none", mmu="tlb:32",
                   icache="32K:4:32", dcache="32K:4:32", l2="none", l3="none", cores=1,
                   shadow_sets=4, model="24k", write_buffer=True),
        source="MD00346 / MD00354 (24K datasheets: caches 0/8/16/32/64 KB 4-way, JTLB "
               "16/32/64 or FMT, 1/2/4 shadow sets, the 8-stage pipeline), MD00343 "
               "Table 6.26 (PRId 0x93), Table 6.18 (CU1 without the FPU), Table 6.32 "
               "(Config3.DSPP = 0: the DSP ASE is the 24KE's)",
        missing=_24K_MISSING)


def _34k(name: str, title: str, fpu: bool) -> Core:
    return Core(
        name, title, "R2", 0x95, ("mips16e", "none"), (fpu,), ("dsp",), _MT_SPECS, _TLB_24K,
        CacheChoice(sizes=(0, 8, 16, 32, 64), direct=(1, 2, 4, 8, 16)),
        CacheChoice(sizes=(0, 8, 16, 32, 64), direct=(1, 2, 4, 8, 16)),
        _L2_SOCIT, (1,), (1,), ("24k",),
        build=dict(compressed="mips16e", fpu=fpu, dsp="dsp", mt="2:5", mmu="tlb:32",
                   icache="32K:4:32", dcache="32K:4:32", l2="none", l3="none", cores=1,
                   shadow_sets=1, model="24k", write_buffer=True),
        source="MD00418 (34Kc datasheet: 1-2 VPEs, 1-9 TCs, caches 0/8/16/32/64 KB 4-way "
               "or direct-mapped 1-16 KB, JTLB 16/32/64 per VPE or FMT, MIPS16e optional), "
               "MD00534 Table 7.39 (PRId 0x95), 4.2 (the DSP ASE always present, "
               "Config3.DSPP; no DSP2P: Revision 1), 6.4 (no dedicated shadow sets: TCs "
               "through SRSConf0), ch. 3 (the 34Kf's FPU), ch. 12 (the ITC), ch. 13 (the "
               "policy managers)",
        missing=("CorExtend (UDI)", "the scratchpad RAMs"),
        notes=("pipeline mode: the 34K's pipeline (MD00534 ch. 2: the 24K's with the "
               "thread-select stage IT, per-TC instruction buffers, the dispatch scheduler)",))


def _1004k(name: str, title: str, fpu: bool) -> Core:
    return Core(
        name, title, "R2", 0x99, ("mips16e", "none"), (fpu,), ("dsp", "none"), ("1:1", "2:2"),
        _TLB_24K,
        CacheChoice(sizes=(8, 16, 32, 64), direct=(0, 1, 2, 4, 8, 16)),
        CacheChoice(sizes=(4, 8, 16, 32, 64), direct=(0, 1, 2, 4, 8, 16)),
        _L2_SOCIT, (1, 2, 3, 4), (1, 2, 3, 4), ("24k",),
        build=dict(compressed="mips16e", fpu=fpu, dsp="dsp", mt="2:2", mmu="tlb:32",
                   icache="32K:4:32", dcache="32K:4:32", l2="256K:8:32", l3="none", cores=2,
                   shadow_sets=1, model="24k", write_buffer=True),
        source="MD00584 (1004K datasheet: 1-4 coherent CPUs, caches, JTLB 16/32/64 or "
               "FMT per VPE, MIPS16e optional, the 1004Kf's FPU, the DSP ASE optional), "
               "MD00622 1.1 (the MT ASE: 1 or 2 VPEs, one TC per VPE, a multi-core ITC), "
               "Table 7.40 (PRId 0x99), 6.4 (no dedicated shadow sets: a free TC's "
               "registers through SRSConf0), 4.2 / Table 7.48 (the DSP ASE: Config3.DSPP, "
               "no DSP2P — Revision 1)",
        missing=("CorExtend (UDI)",),
        notes=("pipeline mode: the 34K's pipeline (MD00622 2.1: the same 8-9 stages, IT "
               "bypassed with one TC)",) + _CPS_NOTE)


CORES: dict[str, Core] = {c.name: c for c in (
    Core("4kc", "MIPS 4Kc", "R1", 0x80, ("none",), (False,), ("none",), ("none",), ("tlb:16",),
         CacheChoice(sizes=(0,), ways=(1, 2, 3, 4), lines=(16,), way_kb=(1, 4), max_kb=16),
         CacheChoice(sizes=(0,), ways=(1, 2, 3, 4), lines=(16,), way_kb=(1, 4), max_kb=16),
         None, (1,), (1,), ("5-stage",),
         build=dict(compressed="none", fpu=False, dsp="none", mt="none", mmu="tlb:16",
                    icache="16K:4:16", dcache="16K:4:16", l2="none", l3="none", cores=1,
                    shadow_sets=1, model="5-stage"),
         source="MD00016 (4K SUM: Release 1, PRId 0x80 (5.2.14), a 16-entry JTLB, caches "
                "0-16 KB 1-4 way 16-byte lines (Table 7-2), the 5-stage I E M A W "
                "pipeline, no FPU / MIPS16e)",
         notes=("the micro-TLBs are the 24K's 4 / 8 entries (the 4Kc's ITLB has 3)",
                "EJTAG 3.20 (the 4Kc has EJTAG 2.6)")),
    _24k("24kc", "MIPS 24Kc", fpu=False),
    _24k("24kf", "MIPS 24Kf", fpu=True),
    _34k("34kc", "MIPS 34Kc", fpu=False),
    _34k("34kf", "MIPS 34Kf", fpu=True),
    _1004k("1004kc", "MIPS 1004Kc", fpu=False),
    _1004k("1004kf", "MIPS 1004Kf", fpu=True),
    Core("interaptiv", "MIPS interAptiv", "R3", 0xA1, ("mips16e", "none"), (True, False),
         ("dspr2", "none"), _MT_SPECS,
         tuple(f"tlb:{n}+sc" for n in (16, 32, 48, 64)),
         CacheChoice(sizes=(4, 8, 16, 32, 64)), CacheChoice(sizes=(4, 8, 16, 32, 64)),
         CacheChoice(sizes=(0, 128, 256, 512, 1024, 2048, 4096, 8192), ways=(8,),
                     lines=(32, 64)),
         (1, 2, 3, 4), (1, 2, 3, 4), ("24k",),
         build=dict(compressed="mips16e", fpu=True, dsp="dspr2", mt="2:5", mmu="tlb:32+sc",
                    icache="32K:4:32", dcache="32K:4:32", l2="256K:8:32", l3="none", cores=2,
                    shadow_sets=1, model="24k", write_buffer=True),
         source="MD00903 (interAptiv datasheet: Release 3, 1-4 cores, L1 4-64 KB 4-way "
                "32-byte lines, the CM2's 8-way L2 of 0/128 KB-8 MB, EVA, the DSP ASE "
                "optional), MD00904 Table 2.10 (PRId 0xA1), the JTLB 16/32/48/64 per VPE, "
                "SegCtl0-2 (2.2.3), 15.3 (the DSP ASE Revision 2: DSPP, DSP2P), 1.1 (the "
                "MT ASE: 1 or 2 VPEs and 1-9 TCs a core), ch. 9 (the GIC's VPs)",
         missing=("MIPS16e2", "the MPU", "the CM2's per-TC priorities (GCR_Cx_TCID_n)"),
         notes=("pipeline mode: the 34K's pipeline (MD00903: an 8/9-stage pipeline, the "
                "dispatch scheduler of MD00904 ch. 10)",) + _CPS_NOTE),
    Core("p5600", "MIPS P5600", "R5", 0xA8, ("none",), (True, False), ("none",), ("none",),
         ("ftlb:64:128x4+sc", "ftlb:64:128x4+sc+pw", "tlb:64+sc", "tlb:64+sc+pw"),
         CacheChoice(sizes=(32, 64)), CacheChoice(sizes=(32, 64)),
         CacheChoice(sizes=(512, 1024, 2048, 4096, 8192), ways=(8,), lines=(32, 64)),
         (1, 2, 3, 4, 5, 6), (1,), ("5-stage", "24k"),
         build=dict(compressed="none", fpu=True, dsp="none", msa="msa", mt="none",
                    mmu="ftlb:64:128x4+sc", icache="32K:4:32",
                    dcache="32K:4:32", l2="512K:8:32", l3="none", cores=2, shadow_sets=1,
                    model="5-stage"),
         source="MD01024 (P5600 datasheet: up to 6 cores, L1 32 / 64 KB 4-way, the 8-way "
                "L2 of 512 KB-8 MB, VTLB 64 + FTLB 512 4-way), MD01025 1.1.2 (Release 5), "
                "Table 2.11 (PRId 0xA8), Table 2.3 (MT 1 / 4), Table 2.6 (SC = 1, no "
                "MIPS16e / microMIPS), Table 2.68 (one GPR set)",
         missing=("the out-of-order superscalar pipeline (no model: the 5-stage or the "
                  "24K's)", "the virtualization ASE (VZ)", "XPA",
                  "MSA's 128-bit EJTAG data breakpoints (DBCS / DBVS, ganged mode)"),
         notes=_CPS_NOTE + ("MSA with the FPU (\"FPU3\", MD01025 13.4); MSAIR = 0x300 (the "
                            "manual: Preset; QEMU's P5600 leaves 0)",),
         msa=("msa", "none"), msa_ade=frozenset({"uncached"}), fpu2008=True),
    Core("m14k", "MIPS M14K", "R2", 0x9B, ("micromips", "micromips-only", "none"), (False,),
         ("none",), ("none",), ("fixed",), NO_CACHE, NO_CACHE, None, (1,), (1, 2, 4, 8, 16), ("5-stage",),
         build=dict(compressed="micromips", fpu=False, dsp="none", mt="none", mmu="fixed",
                    icache="none",
                    dcache="none", l2="none", l3="none", cores=1, shadow_sets=4,
                    model="5-stage"),
         source="MD00666 (M14K datasheet: microMIPS / MIPS32 build options (Table 4), the "
                "5-stage I E M A W pipeline, SRAM interfaces), MD00668 Table 5.26 (AR = "
                "Release 2, MT = FMT), Table 5.23 (PRId 0x9B), Table 5.12 (1/2/4/8/16 "
                "shadow sets), Table 5.10 (no FPU: CU1 reads 0)",
         missing=("the MCU ASE (IACK, more interrupt inputs)", "the MPU",
                  "the SRAM interface timing (eMIPS: ideal memory)")),
    Core("m14kc", "MIPS M14Kc", "R2", 0x9C, ("micromips", "micromips-only", "none"),
         (False,), ("none",), ("none",), ("fixed", "tlb:16", "tlb:32"), _MCU_L1, _MCU_L1, None, (1,),
         (1, 2, 4, 8, 16), ("5-stage",),
         build=dict(compressed="micromips", fpu=False, dsp="none", mt="none", mmu="tlb:16",
                    icache="16K:4:16", dcache="16K:4:16", l2="none", l3="none", cores=1,
                    shadow_sets=4, model="5-stage"),
         source="MD00672 (M14Kc datasheet Table 5: caches 0-64 KB 1-4 way 16-byte lines, "
                "FMT or a 16 / 32-entry TLB, microMIPS options), MD00674 (PRId 0x9C, AR = "
                "Release 2, FP = 0)",
         missing=("the MCU ASE", "the scratchpad RAMs")),
    Core("microaptiv-uc", "MIPS microAptiv UC", "R2", 0x9D,
         ("micromips", "micromips-only", "none"), (False, True), ("dspr2", "none"), ("none",),
         ("fixed",), NO_CACHE,
         NO_CACHE, None, (1,), (1, 2, 4, 8, 16), ("5-stage",),
         build=dict(compressed="micromips", fpu=False, dsp="dspr2", mt="none", mmu="fixed",
                    icache="none", dcache="none", l2="none", l3="none", cores=1,
                    shadow_sets=4, model="5-stage"),
         source="MD00931 (microAptiv UC datasheet Table 7: microMIPS options, FMT, the "
                "UCF's FPU, the DSP Module Revision 2 optional), MD00934 (PRId 0x9D, AR = "
                "Release 2, Config1.FP preset)",
         missing=("the MCU ASE", "the MPU",
                  "the SRAM interface timing (eMIPS: ideal memory)")),
    Core("microaptiv-up", "MIPS microAptiv UP", "R2", 0x9E,
         ("micromips", "micromips-only", "none"), (False,), ("dspr2", "none"), ("none",),
         ("fixed", "tlb:16", "tlb:32"),
         _MCU_L1, _MCU_L1, None, (1,), (1, 2, 4, 8, 16), ("5-stage",),
         build=dict(compressed="micromips", fpu=False, dsp="dspr2", mt="none", mmu="tlb:16",
                    icache="16K:4:16", dcache="16K:4:16", l2="none", l3="none", cores=1,
                    shadow_sets=4, model="5-stage"),
         source="MD00942 (microAptiv UP SUM: PRId 0x9E, AR = Release 2, FMT or a 16 / "
                "32-entry TLB, caches 0-64 KB 1-4 way 16-byte lines, FP = 0, the DSP "
                "Module Revision 2 build-time configurable: 3.2)",
         missing=("the MCU ASE",)),
    # -- MIPS64 (PLAN M22 phase 5c) --
    _5k("5kc", "MIPS64 5Kc", fpu=False),
    _5k("5kf", "MIPS64 5Kf", fpu=True),
    Core("20kc", "MIPS64 20Kc", "R1", 0x82, ("none",), (True,), ("none",), ("none",),
         ("tlb:48",), CacheChoice(sizes=(32,), lines=(32,)), CacheChoice(sizes=(32,), lines=(32,)),
         None, (1,), (1,), ("5-stage",),
         build=dict(compressed="none", fpu=True, dsp="none", mt="none", mmu="tlb:48",
                    icache="32K:4:32", dcache="32K:4:32", l2="none", l3="none", cores=1,
                    shadow_sets=1, model="5-stage"),
         source="QEMU's 20Kc model (cpu-defs.c.inc: PRId 0x82, MIPS64 Release 1, a "
                "48-entry TLB, 32 KB 4-way caches with 32-byte lines, an FPU, SEGBITS 40 / "
                "PABITS 36), the 2002 announcements (a hard core: two 32 KB caches), GCC's "
                "20kc.md (its pipeline); the 20Kc user's manual was not available",
         missing=("the 20Kc's 7-stage dual-issue pipeline (eMIPS: the textbook 5-stage)",
                  "paired single and MIPS-3D (QEMU's FIR.PS / 3D)",
                  "36-bit physical addresses (eMIPS: 32)"),
         arch="mips64"),
    Core("p6600", "MIPS64 P6600", "R6", 0xA4, ("none",), (True, False), ("none",), ("none",),
         ("ftlb:64:128x4+sc", "ftlb:64:128x4+sc+pw"),
         CacheChoice(sizes=(32, 64)), CacheChoice(sizes=(32, 64)),
         CacheChoice(sizes=(512, 1024, 2048, 4096, 8192), ways=(8,), lines=(32, 64)),
         (1, 2, 3, 4, 5, 6), (1,), ("5-stage",),
         build=dict(compressed="none", fpu=True, dsp="none", msa="msa", mt="none",
                    mmu="ftlb:64:128x4+sc", icache="32K:4:32", dcache="32K:4:32",
                    l2="512K:8:32", l3="none", cores=2, shadow_sets=1, model="5-stage"),
         source="MD01137 (P6600 datasheet: up to six MIPS64 Release 6 cores, L1 32 / 64 KB "
                "4-way, the 8-way L2 of 512 KB-8 MB, VTLB 64 + FTLB 512 dual entries in "
                "128 sets x 4 ways, an optional FPU), MD01138 (Table 2.11 PRId 0xA4; Config1: "
                "32-byte L1 lines; Config2: L2 lines 32 / 64; Config3: SC = 1, PW by "
                "configuration, ISA = 0 and DSPP = 0: no microMIPS, no DSP Module; SRSCtl."
                "HSS = 0; ch. 6-8: the CM2, CPC and GIC, with 40-bit physical addresses)",
         missing=("the P6600's out-of-order superscalar pipeline (eMIPS: the textbook "
                  "5-stage)", "the virtualization ASE (VZ)", "the IOCU and its "
                  "IOMMU", "40-bit physical addresses (eMIPS: 32 — the CM's _UPPER registers "
                  "are kept, a block they move above 4 GB is out of reach)",
                  "48-bit virtual addresses (eMIPS: SEGBITS 40)"),
         notes=_CPS_NOTE,
         arch="mips64", msa=("msa", "none"), msa_ade=frozenset({"uncached", "page16k"})),
    Core("i6400", "MIPS I6400", "R6", 0xA9, ("none",), (True,), ("none",), ("none",),
         ("ftlb:16:64x4",), _L1_I6400, _L1_I6400,
         CacheChoice(sizes=(256,), ways=(8,), lines=(64,)), (1, 2, 3, 4, 5, 6), (1,),
         ("5-stage",),
         build=dict(compressed="none", fpu=True, dsp="none", msa="msa", mt="none",
                    mmu="ftlb:16:64x4", icache="64K:4:64", dcache="64K:4:64", l2="256K:8:64",
                    l3="none", cores=1, vps=1, shadow_sets=1, model="5-stage"),
         source="MD01196 (the I6400 Programmer's Guide 1.00: MIPS64 Release 6, 1.2; 16 "
                "VTLB entries per VP and the FTLB of 64 sets x 4 ways, 2.2.1-2.2.2 / "
                "Config4; L1 32 / 64 KB 4-way, 64-byte lines, 3.4 / Config1; the L2 of "
                "Table 3.1; up to 6 cores of 1, 2 or 4 VPs, 5.1 / 6.2.11 / 12.1; the CM3, "
                "CPC, GIC and ITU, ch. 5-7 / 11), QEMU's I6400 model (PRId 0xA9)",
         missing=("the I6400's simultaneous multi-threading, dual-issue pipeline (eMIPS: "
                  "the textbook 5-stage, a pipeline per VP)",
                  "the shared micro-TLBs of its VPs (eMIPS: the 24K's per VP)",
                  "the Debug Unit (DBU: the CPC keeps its power domain only)",
                  "clock ratios and the prescaler (stored: eMIPS has one clock)",
                  "the MMIO regions' ports and request limit (stored: one memory port)",
                  "the L2 prefetcher (GCR_L2_PFT_CONTROL.NPFT = 0)",
                  "the 16-way L2 of 512 KB-8 MB (eMIPS's L2 has 1-8 ways: Config2.SA)",
                  "64 KB FTLB pages (eMIPS: 4 / 16 KB)",
                  "the virtualization ASE (VZ)", "MMID and GINVT",
                  "48-bit virtual and physical addresses (eMIPS: 40 / 32)", "L1 ECC"),
         notes=("MD01196 1.3 calls the FTLB \"512 dual-entry\", 2.2.1 / Config4 64 sets x 4 "
                "ways: eMIPS follows the registers (256 entries)",
                "eMIPS builds the CM3 system (CM, CPC, GIC, ITU) with 2 or more cores or "
                "VPs"),
         arch="mips64", vps=(1, 2, 4), cm="cm3", msa=("msa",)),

    # -- nanoMIPS (PLAN M23 phase 5c) --
    Core("i7200", "MIPS I7200", "R6", 0x00, ("none",), (False,), ("dspr2", "none"),
         tuple(f"{v}:{t}" for v in (1, 2, 3) for t in range(v, 10)),
         ("tlb:16", "tlb:32", "tlb:64", "tlb:16+sc", "tlb:32+sc", "tlb:64+sc")
         + tuple(f"mpu:{n}" for n in _I7200_REGIONS)
         + tuple(f"fixed+mpu:{n}" for n in _I7200_REGIONS),
         CacheChoice(sizes=(4, 8, 16, 32, 64, 128)), CacheChoice(sizes=(4, 8, 16, 32, 64, 128)),
         CacheChoice(sizes=(0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192), ways=(8,),
                     lines=(32, 64), min_kb=((32, 128),)),
         (1, 2, 3, 4), (1,), ("5-stage",),
         build=dict(compressed="none", fpu=False, dsp="dspr2", mt="2:5", mmu="tlb:32",
                    icache="32K:4:32", dcache="32K:4:32", l2="256K:8:32", l3="none", cores=1,
                    shadow_sets=1, model="5-stage"),
         source="MD01227 (I7200 datasheet 01.20: nanoMIPS32 Release 6, 1.1; the DSP ASE "
                "optional; the MT ASE with 1-3 VPEs and 1-9 TCs a core; a JTLB of 16 / 32 / "
                "64 dual entries per VPE, an MPU or the FMT, EVA, Table 2; L1 4-64 KB "
                "4-way 32-byte lines, 128 KB with the MPU, Table 1; 1-4 cores on a CM 2.6 with the GIC, CPC, "
                "ITU, Table 2), MD01232 (Programmer's Guide 01.20: ch. 4 the MPU, 8-32 "
                "regions by 4; Table 30, the 8-way L2 of 32 / 64-byte lines; ch. 10 / 11 / "
                "14), QEMU's I7200 model (PRId "
                "0x00010000: neither manual gives the Processor ID, and Linux's MIPS "
                "cpu.h has no I7200)",
         missing=("the I7200's 9-stage dual-issue pipeline (eMIPS: the textbook 5-stage; "
                  "MT in instruction mode)", "the scratchpad RAMs (ISPRAM, "
                  "DSPRAM, USPRAM)", "the QoS policy manager (eMIPS: the 34K's round-robin "
                  "/ weighted round-robin)", "OCI debug and the Fast Debug Channel (eMIPS: "
                  "EJTAG 3.20)", "the CM 2.6's own registers (eMIPS: the CM2's)",
                  "more than 8 VPs in a CPS (4 cores of 3 VPEs: 12)"),
         notes=("the nanoMIPS DSP and MT ASEs (MD01249 / MD01255)",
                "the MPU (MD01232 ch. 4): mpu:N = the MPU build (PA = VA, 16 segments, N "
                "regions, 128 KB L1s allowed), fixed+mpu:N = the FMT build with N MPU "
                "regions") + _CPS_NOTE,
         arch="nanomips", mpu_l1_kb=128),

    # -- RISC-V (PLAN M25 phase 5) --
    Core("p8700", "MIPS P8700", "RV", 0x00, ("auto",), (True,), ("auto",), ("auto",),
         ("fixed",), _L1_P8700, _L1_P8700,
         CacheChoice(sizes=(256, 512, 1024, 2048, 4096, 8192), ways=(16,), lines=(64,),
                     ways_kb=((256, 8),)),
         (1, 2, 3, 4, 5, 6), (4,), ("p8700", "5-stage"),
         build=dict(rv_extensions=_P8700_EXTENSIONS, rv_modes="msu", cores=1, vps=1,
                    icache="64K:4:64", dcache="64K:4:64", l2="1M:16:64", l3="none",
                    model="p8700"),
         source="MD01502 (the P8700 / P8700-F Multiprocessing System Programmer's Guide "
                "1.80: RV64GC Zba Zbb, 1.2; the custom instructions, Appendix B; the CSRs "
                "and causes, ch. 7; the counters, 11.1; rev. 1.84: the MMU, ch. 2, the "
                "caches, ch. 3, the CM / CPC, ch. 5 / 6, the interrupt controller, ch. 8), "
                "QEMU's mips-p8700 (marchid "
                "0x8000000000000201, Sv48), OpenSBI's P8700 platform code, Linux's "
                "\"Zalrsc only\" patch (no AMOs)",
         missing=("MDIAGR / MDIAGW (undocumented)",),
         notes=("no AMO instructions: an AMO traps for M-mode emulation (mipsconfig6)",
                "the time CSR traps (Read Time, cause 26, or an illegal instruction)"),
         arch="rv64", rv_extensions=(_P8700_EXTENSIONS,), rv_modes=("msu",), vps=(1, 2)),
)}

#: the generic eMIPS core's name (no preset): any setting, PRId 0
GENERIC = ""


def get(name: str | None) -> Core | None:
    return CORES.get((name or "").lower()) if name else None


def conflict(o, release: str, name: str) -> str | None:
    """Why setting ``name`` (an Options field or "release") is not one
    the chosen core can be built with (None: it is, or no core)."""
    core = get(getattr(o, "core", ""))
    if core is None or name not in core.fields:
        return None
    value = release if name == "release" else getattr(o, name)
    if core.allows(name, value):
        return _mpu_l1(core, o) if name in ("icache", "dcache", "mmu") else None
    shown = {"fpu": "an FPU" if value else "no FPU"}.get(name, str(value))
    label = {"shadow_sets": "shadow sets", "l2": "L2", "l3": "L3",
             "release": "release", "rv_extensions": "the extensions",
             "rv_modes": "the privilege modes"}.get(name, name)
    return (f"the {core.title} is built with {label}: {core.choices(name)} "
            f"(not {shown}; {core.source.split(' (')[0]})")


def _mpu_l1(core: Core, o) -> str | None:
    """An L1 of :attr:`Core.mpu_l1_kb` without the MPU (the I7200's 128 KB)."""
    if not core.mpu_l1_kb or mmu_key(o.mmu).startswith("mpu"):
        return None
    for name in ("icache", "dcache"):
        try:
            g = Geometry.parse(getattr(o, name), 1)
        except ValueError:
            continue
        if g is not None and g.size == core.mpu_l1_kb * 1024:   # (larger: not at all)
            return (f"the {core.title} is built with {core.mpu_l1_kb} KB L1 caches only "
                    f"with the MPU (--mmu mpu:N; not {name} {getattr(o, name)} with "
                    f"--mmu {o.mmu}; {core.source.split(' (')[0]})")
    return None


def apply(o, name: str, keep: frozenset[str] | set[str] = frozenset()):
    """``o`` with core ``name``'s build (the fields not in ``keep``) and
    ``core`` set: (the options, the core's release — None for a RISC-V
    core).  ``name`` "" or
    "generic": the generic core (the settings stay)."""
    if not name or name.lower() == "generic":
        return replace(o, core=GENERIC), None
    core = get(name)
    if core is None:
        raise ValueError(f"unknown core {name!r} ({', '.join(CORES)})")
    changes = {k: v for k, v in core.build.items() if k not in keep}
    if "arch" not in keep:
        changes["arch"] = core.arch
    if core.rv:                                 # (RISC-V: no release; the session's stays)
        return replace(o, core=core.name, **changes), None
    if "vps" not in keep:
        changes.setdefault("vps", 1)
    if "msa" not in keep:                       # (the builds before M24 have none)
        changes.setdefault("msa", "none")
    return replace(o, core=core.name, **changes), core.release


def describe(core: Core) -> list[str]:
    """The core's settings, for the CLI / GUI."""
    from .options import arch_title
    if core.rv:
        from . import p8700 as P
        out = [f"{core.title}: {arch_title(core.arch)}, mvendorid 0x{P.MVENDORID:x}, marchid "
               f"0x{P.MARCHID:016x}, mimpid 0x{P.MIMPID:x}"]
    else:
        out = [f"{core.title}: {arch_title(core.arch)} Release {core.release[1:]}, "
               f"PRId 0x{core.prid:08x}"]
    for name in core.fields[1:] if not core.rv else core.fields:
        out.append(f"  {name.replace('_', ' ')}: {core.choices(name)}")
    out += [f"  not modelled: {m}" for m in core.missing]
    out += [f"  note: {n}" for n in core.notes]
    out.append(f"  source: {core.source}")
    return out
