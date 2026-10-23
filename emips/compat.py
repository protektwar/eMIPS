"""Which settings go together (Qt-free; the Settings dialog, the toolbar
and the command line use it).

A combination eMIPS cannot model correctly is never offered: each
:class:`Rule` names the settings it ties, why (the manual or the design
that says so), and how the *dependent* setting adjusts when a *primary*
one (the release, the board) changes.  The GUI disables the values a
rule forbids and shows the adjustment; the command line refuses a
conflicting combination given explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from . import cores as _cores
from .options import Options, arch_title

RELEASE_ORDER = ("R1", "R2", "R3", "R5", "R6")


def at_least(release: str, minimum: str) -> bool:
    return RELEASE_ORDER.index(release) >= RELEASE_ORDER.index(minimum)


def _mmu_flags(o: Options) -> set[str]:
    return set(o.mmu.split("+")[1:])


def _drop_flag(o: Options, flag: str) -> None:
    base, *flags = o.mmu.split("+")
    o.mmu = "+".join([base] + [f for f in flags if f != flag])


def _mt(o: Options) -> bool:
    """The MT Module asked for explicitly ("VPES:TCS")."""
    return o.mt not in ("auto", "none")


def _vpes(o: Options) -> int:
    from .mt import parse_spec
    try:
        spec = parse_spec(o.mt)
    except ValueError:
        return 1
    return spec[0] if spec else 1


#: the VPs a CPS's boot firmware serves (emips.cps.MAX_VPS)
_MAX_VPS = 8


def _malta(o: Options) -> bool:
    return o.board.split(":")[0] == "malta"


def _mips64(o: Options) -> bool:
    return o.arch == "mips64"


def _mmu_kind(o: Options) -> str:
    """"mpu" (the MPU build), "fixed+mpu" (the FMT build's regions), or the
    MMU's kind ("fixed", "tlb" ...)."""
    base, *flags = o.mmu.lower().split("+")
    kind = base.split(":")[0]
    return "fixed+mpu" if kind == "fixed" and flags and flags[0].startswith("mpu") else kind


def _nano(o: Options) -> bool:
    return o.arch == "nanomips"


def _rv(o: Options) -> bool:
    return o.arch in ("rv32", "rv64")


#: a RISC-V hart (PLAN M25) has none of the MIPS settings: (field, the value
#: it goes back to, why)
_RV_DEFAULTS = (
    ("board", "flat", "the Malta's core cards carry MIPS CPUs (MD00048); a RISC-V program "
                      "runs on the flat RARS memory map"),
    ("cores", 1, "a generic RISC-V core is one hart here; several harts on one platform "
                 "come with the P8700 preset (PLAN M25 phase 5)"),
    ("model", "5-stage", "the 24K is a MIPS32 core (MD00343)"),
    ("mmu", "fixed", "a RISC-V hart translates with satp (Sv32 / Sv39 / Sv48 / Sv57, the "
                     "privileged spec ch. 12), not a MIPS MMU"),
    ("compressed", "auto", "RISC-V has no ISA modes: its compressed instructions are the "
                           "C extension (Options.rv_extensions)"),
    ("dsp", "auto", "the DSP Module is a MIPS ASE"),
    ("msa", "auto", "MSA is a MIPS ASE (RISC-V's vectors are the V extension)"),
    ("mt", "auto", "the MT Module is a MIPS ASE"),
    ("vps", 1, "Release 6 VPs belong to MIPS cores"),
    ("nms", False, "the nanoMIPS subset belongs to nanoMIPS cores"),
    ("shadow_sets", 4, "shadow register sets belong to MIPS cores"),
    ("c_isa", "mips32", "MIPS16e / microMIPS are MIPS ISAs"),
)


#: why the nanoMIPS core has no FPU / ISA modes / ...
_NANO_FPU = ("MD01247 defines no nanoMIPS FPU instructions (the CP1 cells of its opcode map "
             "are reserved; the I7200 has no FPU)")


@dataclass(frozen=True)
class Rule:
    #: the setting that adjusts (the dependent)
    field: str
    #: what it depends on ("release", "board", ...)
    on: str
    #: True when the combination is not allowed
    broken: Callable[[Options, str], bool]
    #: the dependent's adjustment (to a value that works)
    fix: Callable[[Options], None]
    #: why, for the user (or a function of the options and the release)
    why: str | Callable[[Options, str], str]

    def reason(self, o: Options, release: str) -> str:
        return self.why(o, release) if callable(self.why) else self.why


RULES = (
    # first, RISC-V (PLAN M25): another architecture — none of the MIPS settings
    *(Rule(f, "arch", (lambda f, v: lambda o, r: _rv(o) and getattr(o, f) != v)(f, v),
           (lambda f, v: lambda o: setattr(o, f, v))(f, v), why)
      for f, v, why in _RV_DEFAULTS),
    # first, nanoMIPS (PLAN M23): an architecture of its own, Release 6 based —
    # what its cores do not have (the rules below see the settings they leave)
    Rule("arch", "release", lambda o, r: _nano(o) and r != "R6",
         lambda o: setattr(o, "arch", "mips32"),
         "nanoMIPS is Release 6 based (MD01247 ch. 1): there is no nanoMIPS Release 1-5"),
    Rule("fpu", "arch", lambda o, r: _nano(o) and o.fpu, lambda o: setattr(o, "fpu", False),
         _NANO_FPU),
    Rule("compressed", "arch", lambda o, r: _nano(o) and o.compressed not in ("auto", "none"),
         lambda o: setattr(o, "compressed", "auto"),
         "a nanoMIPS core has no ISA modes (MD01251 3.5.4): its 16-bit instructions are "
         "part of the one nanoMIPS ISA"),
    Rule("model", "arch", lambda o, r: _nano(o) and o.model == "24k",
         lambda o: setattr(o, "model", "5-stage"),
         "the 24K is a MIPS32 core (MD00343): a nanoMIPS program runs on the 5-stage pipeline"),
    Rule("mmu", "arch", lambda o, r: not _nano(o) and "mpu" in _mmu_kind(o),
         lambda o: setattr(o, "mmu", "fixed"),
         "the MPU is the I7200's (MD01232 ch. 4): a nanoMIPS core's"),
    Rule("mmu", "board", lambda o, r: _malta(o) and _mmu_kind(o) == "mpu",
         lambda o: setattr(o, "mmu", "fixed+mpu:" + (o.mmu.partition(":")[2] or "16")),
         "with the MPU PA = VA (MD01232 ch. 4): the Malta's SDRAM is at physical 0, where "
         "its kseg0 programs are not — the FMT build with MPU regions maps them there"),
    Rule("c_isa", "arch", lambda o, r: _nano(o) and o.c_isa != "mips32",
         lambda o: setattr(o, "c_isa", "mips32"),
         "C for a nanoMIPS core is nanoMIPS code (nanomips-elf-gcc): no MIPS16e / microMIPS"),
    # then MIPS64 (PLAN M22): the 64-bit core has none of these yet — the
    # rules below see the settings they leave
    Rule("compressed", "arch", lambda o, r: _mips64(o) and o.compressed.startswith("micromips")
         and not at_least(r, "R3"), lambda o: setattr(o, "compressed", "auto"),
         "microMIPS64 needs Release 3 or later (MD00594 3.1: microMIPS came with Release 3)"),
    Rule("c_isa", "arch", lambda o, r: _mips64(o) and o.c_isa == "micromips"
         and not at_least(r, "R3"), lambda o: setattr(o, "c_isa", "mips32"),
         "microMIPS64 needs Release 3 or later (MD00594 3.1)"),
    Rule("model", "arch", lambda o, r: _mips64(o) and o.model == "24k",
         lambda o: setattr(o, "model", "5-stage"),
         "the 24K is a MIPS32 core (MD00343): a MIPS64 program runs on the 5-stage pipeline"),
    Rule("core", "arch", lambda o, r: (c := _cores.get(o.core)) is not None and c.arch != o.arch,
         lambda o: setattr(o, "core", ""),
         lambda o, r: (f"the {_cores.get(o.core).title} is a "
                       f"{arch_title(_cores.get(o.core).arch)} core")),
    Rule("compressed", "release", lambda o, r: o.compressed == "mips16e" and r == "R6",
         lambda o: setattr(o, "compressed", "auto"),
         "Release 6 has no MIPS16e (it was removed; microMIPS R6 is its 16-bit ISA)"),
    Rule("compressed", "release",
         lambda o, r: o.compressed.startswith("micromips") and not at_least(r, "R2"),
         lambda o: setattr(o, "compressed", "auto"),
         "microMIPS32 needs Release 2 or later (MD00582 describes it for Release 3; the "
         "M14K is a Release 2 core with microMIPS, MD00668 Table 5.26)"),
    Rule("c_isa", "release", lambda o, r: o.c_isa == "mips16e" and r == "R6",
         lambda o: setattr(o, "c_isa", "mips32"),
         "Release 6 has no MIPS16e"),
    Rule("c_isa", "release", lambda o, r: o.c_isa == "micromips" and r not in ("R2", "R3", "R5"),
         lambda o: setattr(o, "c_isa", "mips32"),
         "microMIPS C needs Release 2, 3 or 5: Release 1 has no microMIPS core, and no "
         "compiler builds Release 6 microMIPS as the manual defines it (emips.cc)"),
    Rule("compressed", "c_isa", lambda o, r: o.c_isa == "micromips" and o.compressed in (
        "none", "mips16e", "micromips-only"), lambda o: setattr(o, "compressed", "auto"),
         "microMIPS C needs a core with microMIPS and MIPS32 (crt0 is MIPS32; auto: the "
         "program's code decides)"),
    Rule("compressed", "c_isa", lambda o, r: o.c_isa == "mips16e" and o.compressed in (
        "none", "micromips", "micromips-only"), lambda o: setattr(o, "compressed", "auto"),
         "MIPS16e C needs a core with MIPS16e"),
    Rule("model", "compressed", lambda o, r: o.model == "24k" and o.compressed.startswith("micromips"),
         lambda o: setattr(o, "model", "5-stage"),
         "the 24K has no microMIPS (MD00343 1.2.1.11: MIPS16e); the microMIPS cores (M14K, "
         "microAptiv) have 5-stage pipelines"),
    Rule("dsp", "release", lambda o, r: o.dsp in ("dsp", "dspr2") and not at_least(r, "R2"),
         lambda o: setattr(o, "dsp", "auto"),
         "the DSP Module needs Release 2 or later (MD00374 3.1)"),
    Rule("dsp", "release", lambda o, r: o.dsp == "dsp" and r == "R6",
         lambda o: setattr(o, "dsp", "dspr2"),
         "on Release 6 the DSP Module is Revision 3: Revision 2's instructions and "
         "BPOSGE32C (Config3.DSPP = DSP2P = 1, MD00374 3.11)"),
    Rule("msa", "release", lambda o, r: o.msa == "msa" and r not in ("R5", "R6"),
         lambda o: setattr(o, "msa", "auto"),
         "MSA needs Release 5 or later (MD00866 1.1: MIPS Release 5)"),
    Rule("msa", "fpu", lambda o, r: o.msa == "msa" and not o.fpu,
         lambda o: setattr(o, "msa", "auto"),
         "MSA needs the FPU: the vector registers extend its 64-bit registers (FR = 1; "
         "MD00866 3.1, Config3.MSAP needs Config1.FP)"),
    Rule("msa", "arch", lambda o, r: o.msa == "msa" and _nano(o),
         lambda o: setattr(o, "msa", "auto"),
         "nanoMIPS has no MSA (MD01247: no MSA instruction, no opcode space for it)"),
    Rule("msa", "mt", lambda o, r: o.msa == "msa" and _mt(o),
         lambda o: setattr(o, "msa", "auto"),
         "no MT core has MSA (the 34K, 1004K, interAptiv; MSA's partitioning registers "
         "per thread context are not modelled)"),
    Rule("model", "msa", lambda o, r: o.model == "24k" and o.msa == "msa",
         lambda o: setattr(o, "model", "5-stage"),
         "the 24K has no MSA (the MSA cores — P5600, P6600, I6400 — are modelled on the "
         "5-stage pipeline)"),
    Rule("mt", "release", lambda o, r: _mt(o) and r in ("R1", "R6") and not _nano(o),
         lambda o: setattr(o, "mt", "auto"),
         "the MT Module needs Release 2-5 (MD00378 1.1: Release 2; Release 6 replaced "
         "it with DVP / EVP and has none of its instructions)"),
    Rule("vps", "release", lambda o, r: o.vps > 1 and r != "R6",
         lambda o: setattr(o, "vps", 1),
         "virtual processors are Release 6 multi-threading (Vol. III 9.8, Config5.VP; "
         "before Release 6: the MT Module)"),
    Rule("shadow_sets", "release", lambda o, r: o.shadow_sets > 1 and not at_least(r, "R2"),
         lambda o: setattr(o, "shadow_sets", 1),
         "shadow register sets exist from Release 2 (SRSCtl / SRSMap); Release 1 has the "
         "normal GPRs only"),
    Rule("mmu", "release", lambda o, r: "sc" in _mmu_flags(o) and not at_least(r, "R3"),
         lambda o: _drop_flag(o, "sc"),
         "Segmentation Control and EVA are Release 3 features"),
    Rule("mmu", "release", lambda o, r: "pw" in _mmu_flags(o) and not at_least(r, "R5"),
         lambda o: _drop_flag(o, "pw"),
         "the hardware page table walker is a Release 5 feature"),
    Rule("cores", "release", lambda o, r: o.cores > 1 and not at_least(r, "R2"),
         lambda o: setattr(o, "cores", 1),
         "a multi-core system (CPS) is made of Release 2+ cores (EBase.CPUNum; the 1004K, "
         "interAptiv, P5600)"),
    Rule("cores", "mt", lambda o, r: _mt(o) and o.cores * _vpes(o) > _MAX_VPS,
         lambda o: setattr(o, "cores", _MAX_VPS // _vpes(o)),
         lambda o, r: (f"{o.cores} MT cores of {_vpes(o)} VPEs: the boot firmware's GIC "
                       f"routes the IPIs of {_MAX_VPS} VPs (Boot-MIPS init_gic; the MT CPSs, "
                       "the 1004K / interAptiv, have up to 4 cores of 2 VPEs)")),
    Rule("iocu", "cores", lambda o, r: o.iocu and o.cores < 2,
         lambda o: setattr(o, "iocu", False),
         "an I/O Coherence Unit is part of a multi-core system's Coherence Manager "
         "(GCR_CONFIG.NUMIOCU): it needs 2 or more cores"),
    Rule("iocu", "board", lambda o, r: o.iocu and not _malta(o),
         lambda o: setattr(o, "iocu", False),
         "the IOCU makes the Malta's PCI DMA coherent: the flat board has no DMA device"),
    Rule("boot", "board", lambda o, r: o.boot == "user" and _malta(o)
         and o.mmu.split("+")[0].split(":")[0] not in ("tlb", "ftlb"),
         lambda o: setattr(o, "boot", "kernel"),
         "user mode on the Malta needs a TLB: the boot firmware's refill handler maps "
         "kuseg into the SDRAM (the fixed mapping would put it beyond it)"),
    Rule("board", "core", lambda o, r: o.core == "20kc" and _malta(o),
         lambda o: setattr(o, "board", "flat"),
         "the 20Kc was on the Core20K card, whose system controller is the Bonito64 "
         "(MD00008 Table 1.1): eMIPS's Malta has the CoreLV's GT-64120, which carried "
         "4K and 5K-class CPUs (MD00007)"),
    Rule("disks", "board", lambda o, r: bool(o.disks.strip()) and not _malta(o),
         lambda o: setattr(o, "disks", ""),
         "the IDE disks are on the Malta's PIIX4 south bridge: the flat board has no "
         "devices but the MARS console"),
    Rule("l3", "l2", lambda o, r: o.l3 != "none" and o.l2 == "none",
         lambda o: setattr(o, "l3", "none"),
         "an L3 cache needs an L2 cache"),
) + tuple(
    # last: a core preset (emips.cores) holds only while every setting is one
    # its manuals allow; any other value makes it the generic eMIPS core
    Rule("core", name,
         lambda o, r, name=name: _cores.conflict(o, r, name) is not None,
         lambda o: setattr(o, "core", ""),
         lambda o, r, name=name: _cores.conflict(o, r, name) or "")
    for name in _cores.FIELDS)


def _shown(value) -> str:
    """A setting's value in a note: on / off, generic for no core preset."""
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value) if value != "" else "generic"


def problems(o: Options, release: str) -> list[Rule]:
    """The rules this combination breaks."""
    return [r for r in RULES if r.broken(o, release)]


def adjust(o: Options, release: str, keep: frozenset[str] | set[str] = frozenset()
           ) -> tuple[Options, list[str]]:
    """The dependents adjusted to what the primary settings allow: (the
    new options, a note for each change); the fields in ``keep`` (given
    on purpose) are not adjusted."""
    o = replace(o, latency=dict(o.latency), mem_timing=dict(o.mem_timing))
    notes = []
    for rule in RULES:
        if rule.field not in keep and rule.broken(o, release):
            old, why = getattr(o, rule.field), rule.reason(o, release)
            rule.fix(o)
            notes.append(f"{rule.field} {_shown(old)} -> {_shown(getattr(o, rule.field))}: "
                         f"{why}")
    return o, notes


def forbidden(field: str, value, o: Options, release: str) -> str | None:
    """Why ``field = value`` does not go with the rest (None: it does).
    ``field`` may be "release"."""
    if field == "release":
        cand, rel = o, value
    else:
        cand, rel = replace(o, **{field: value}), release
    for rule in RULES:
        if rule.broken(cand, rel) and field in (rule.field, rule.on):
            return rule.reason(cand, rel)
    return None


def program_problem(o: Options, kind: str, program=None) -> str | None:
    """A loaded program that cannot run with these options: ``kind`` "c" /
    "elf" (linked for the flat MARS layout, 0x00400000); ``program`` (an
    emips.assembler.Program) with MIPS16e / microMIPS code the chosen core
    does not have."""
    if kind in ("c", "elf") and _malta(o):
        return ("C programs and ELF executables are linked for the flat memory layout "
                "(code at 0x00400000): on the Malta that is not memory — choose the flat board")
    if kind == "elf" and getattr(program, "arch", o.arch) != o.arch:
        what = {"mips64": "an ELF64 (n64)", "nanomips": "a nanoMIPS ELF (EM_NANOMIPS)"}.get(
            program.arch, "an ELF32")
        return (f"the program is {what}: it runs on {arch_title(program.arch)} — open it "
                f"again, or choose {arch_title(program.arch)}")
    if (_mips64(o) and not o.mmu.startswith(("tlb", "ftlb"))
            and max(getattr(program, "image", None) or (0,)) > 0xFFFFFFFF):
        return ("the program has code or data above the 32-bit compatibility space (a "
                "64-bit address): that needs 64-bit addressing, Config.AT = 2 — a TLB MMU "
                f"(--mmu tlb; this is --mmu {o.mmu})")
    if _nano(o) and o.nms and "xnms" in getattr(program, "needs", ()):
        return ("the program is built for the full nanoMIPS ISA (.nanoMIPS.abiflags: "
                "NANOMIPS_ASE_xNMS, DN00179 5.2): this core has the nanoMIPS subset only "
                "(--nms) — build it with -march=32r6s (emips cc --nms)")
    if _nano(o) and o.mt == "none" and "mt" in getattr(program, "needs", ()):
        return ("the program has nanoMIPS MT ASE instructions (MD01255): the core has "
                "none (--mt none)")
    if _nano(o) and o.dsp == "none" and "dsp" in getattr(program, "needs", ()):
        return ("the program has nanoMIPS DSP ASE instructions (MD01249): the core has "
                "none (--dsp none)")
    core = _cores.get(o.core)
    if getattr(program, "msa", False) and (o.msa == "none" or core is not None
                                           and "msa" not in core.msa):
        return ("the program has MSA instructions: " + (
            f"this {core.title} has no MSA" if core is not None and "msa" not in core.msa
            else "the core has none (--msa none)"))
    if getattr(program, "msa", False) and not o.fpu:
        return ("the program has MSA instructions: the core has no FPU (--no-fpu), and "
                "MSA's vector registers extend the FPU's")
    if getattr(program, "mt", False) and o.mt == "none":
        return ("the program has MT Module instructions: " + (
            f"this {core.title} is built without the MT Module" if core is not None
            and core.mt == ("none",) else "the core has none (--mt none)"))
    need = getattr(program, "dsp", 0)
    have = {"dsp": 1, "dspr2": 2}.get(o.dsp, 0 if o.dsp == "none" or core else need)
    if need > have:
        rev = "DSP Module Revision 2" if need > 1 else "DSP Module"
        has = "only Revision 1" if have else ("no DSP Module" if core else "none")
        what = (f"this {core.title} has {has} (--dsp {o.dsp}; it is built with "
                f"{core.choices('dsp')})" if core else f"the core has {has} (--dsp {o.dsp})")
        return f"the program has {rev} instructions: {what}"
    if core is not None and program is not None:
        for code, isa, name in ((program.m16, "mips16e", "MIPS16e"),
                                (program.mm, "micromips", "microMIPS")):
            if code and not o.compressed.startswith(isa):
                return (f"the program has {name} code: this {core.title} has "
                        f"{'no compressed ISA' if o.compressed == 'none' else 'no ' + name} "
                        f"(--compressed {o.compressed}; it is built with "
                        f"{core.choices('compressed')})")
    return None


def explain(rules: list[Rule], o: Options | None = None, release: str = "R5") -> str:
    o = o if o is not None else Options()
    return "; ".join(f"{r.field} with this {r.on}: {r.reason(o, release)}" for r in rules)
