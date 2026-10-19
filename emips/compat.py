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

from .options import Options

RELEASE_ORDER = ("R1", "R2", "R3", "R5", "R6")


def at_least(release: str, minimum: str) -> bool:
    return RELEASE_ORDER.index(release) >= RELEASE_ORDER.index(minimum)


def _mmu_flags(o: Options) -> set[str]:
    return set(o.mmu.split("+")[1:])


def _drop_flag(o: Options, flag: str) -> None:
    base, *flags = o.mmu.split("+")
    o.mmu = "+".join([base] + [f for f in flags if f != flag])


def _malta(o: Options) -> bool:
    return o.board.split(":")[0] == "malta"


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
    #: why, for the user
    why: str


RULES = (
    Rule("compressed", "release", lambda o, r: o.compressed == "mips16e" and r == "R6",
         lambda o: setattr(o, "compressed", "auto"),
         "Release 6 has no MIPS16e (it was removed; microMIPS R6 is its 16-bit ISA)"),
    Rule("compressed", "release",
         lambda o, r: o.compressed.startswith("micromips") and not at_least(r, "R3"),
         lambda o: setattr(o, "compressed", "auto"),
         "microMIPS32 starts with Release 3 (MD00582 3.1)"),
    Rule("c_isa", "release", lambda o, r: o.c_isa == "mips16e" and r == "R6",
         lambda o: setattr(o, "c_isa", "mips32"),
         "Release 6 has no MIPS16e"),
    Rule("c_isa", "release", lambda o, r: o.c_isa == "micromips" and r not in ("R3", "R5"),
         lambda o: setattr(o, "c_isa", "mips32"),
         "microMIPS C needs Release 3 or 5: microMIPS starts with Release 3 (MD00582 3.1), "
         "and no compiler builds Release 6 microMIPS as the manual defines it (emips.cc)"),
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
    Rule("disks", "board", lambda o, r: bool(o.disks.strip()) and not _malta(o),
         lambda o: setattr(o, "disks", ""),
         "the IDE disks are on the Malta's PIIX4 south bridge: the flat board has no "
         "devices but the MARS console"),
    Rule("l3", "l2", lambda o, r: o.l3 != "none" and o.l2 == "none",
         lambda o: setattr(o, "l3", "none"),
         "an L3 cache needs an L2 cache"),
)


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
            old = getattr(o, rule.field)
            rule.fix(o)
            notes.append(f"{rule.field} {old} -> {getattr(o, rule.field)}: {rule.why}")
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
            return rule.why
    return None


def program_problem(o: Options, kind: str) -> str | None:
    """A loaded program that cannot run with these options: ``kind`` "c" /
    "elf" (linked for the flat MARS layout, 0x00400000)."""
    if kind in ("c", "elf") and _malta(o):
        return ("C programs and ELF executables are linked for the flat memory layout "
                "(code at 0x00400000): on the Malta that is not memory — choose the flat board")
    return None


def explain(rules: list[Rule]) -> str:
    return "; ".join(f"{r.field} with this {r.on}: {r.why}" for r in rules)
