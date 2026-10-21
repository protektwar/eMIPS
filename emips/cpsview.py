"""Views of a multi-core system (PLAN M11 phase 5), shared by the CLI
debugger (``cps``, ``cpu``, ``cm``, ``gic``) and the GUI's System pane.

Qt-free and side-effect free: everything is read from the
:class:`~emips.session.Session`, its :class:`~emips.cps.CPS`, the
Coherence Manager and the GIC — registers are read through their own
``get`` (which changes nothing), never by a load of a core.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import cp0 as C
from . import cps as P
from . import gic as G
from .cacheview import MESI_NAMES

M32 = 0xFFFFFFFF


# ------------------------------------------------------------------ the cores

@dataclass
class CoreRow:
    cpu: int
    state: str               # the CPC's SEQSTATE, or "exited" / "waiting"
    pc: int | None
    line: int | None
    mode: str                # kernel / supervisor / user; + EXL / ERL
    pins: str                # the hardware interrupt pins asserted (IP2..IP7)
    selected: bool
    running: bool


def _mode(m) -> str:
    st = m.status
    if st & C.ST_ERL:
        return "kernel (ERL)"
    if st & C.ST_EXL:
        return "kernel (EXL)"
    return {0: "kernel", 1: "supervisor", 2: "user"}.get((st >> 3) & 3, "kernel")


def _pins(lines: int) -> str:
    return ", ".join(f"IP{i + 2}" for i in range(6) if lines >> i & 1) or "-"


def core_rows(s) -> list[CoreRow]:
    """Every core: power state, position, mode, interrupt pins."""
    out = []
    for c in s.cores():
        m = s.machines[c["cpu"]]
        state = "exited" if c["halted"] else ("waiting (WAIT)" if c["waiting"] else c["state"])
        lines = (f"RIPL {m.irq_lines}" if m.cp0_read(*C.CONFIG3) & C.CONFIG3_VEIC
                 and s.cps.gic.vps[c["cpu"]].eic else _pins(m.irq_lines))
        out.append(CoreRow(c["cpu"], state, c["pc"], c["line"], _mode(m),
                           lines, c["selected"], c["running"]))
    return out


# ------------------------------------------------------------------ coherence

@dataclass
class DomainRow:
    cpu: int
    coherence: int           # GCR_Cx_COHERENCE
    coherent: bool           # its own bit: coherent mode
    hears: list[int]         # whose interventions it receives


def domain_rows(cps) -> list[DomainRow]:
    out = []
    for j in range(cps.n):
        dom = cps.gcr.domain(j)                  # (a CM3's COHEN: one domain)
        out.append(DomainRow(j, cps.gcr.core[j][P.CX_COHERENCE], bool(dom >> j & 1),
                             [i for i in range(cps.n) if i != j and dom >> i & 1]))
    return out


def last_line(s) -> tuple[int, str] | None:
    """``(line PA, how)`` of the selected core's last data-cache access."""
    for where, a in reversed(s.cache_accesses()):
        if a.level == "D":
            mask = s.machine.caches.d.line_mask
            return a.pa & ~mask & M32, f"{where}: {a.kind} 0x{a.pa:08x}"
    return None


def copies_rows(s, pa: int) -> list[tuple[int, str]]:
    """``(core, MESI name)`` of every data-cache copy of ``pa``'s line."""
    return [(j, MESI_NAMES[st]) for j, st in s.cps.cm.copies(pa)]


# ------------------------------------------------------------------ the GIC

@dataclass
class SourceRow:
    src: int
    kind: str
    enabled: bool
    pending: bool
    target: str
    vps: str


def gic_rows(gic) -> list[SourceRow]:
    """The enabled sources (and every pending, routed one)."""
    pend = gic.pending()
    out = []
    for src in range(gic.n):
        bit = 1 << src
        if not gic.mask & bit:
            continue
        kind = ("edge, " + ("both" if gic.dual & bit else
                            "rising" if gic.pol & bit else "falling")
                if gic.trig & bit else
                "level, active " + ("high" if gic.pol & bit else "low"))
        out.append(SourceRow(src, kind, True, bool(pend & bit), G._target(gic.map_pin[src],
                                                                   gic._eic_vps(gic.map_vp[src])),
                             G._vps(gic.map_vp[src])))
    return out


@dataclass
class LocalRow:
    """One local interrupt of a core (GIC_VL_PEND / MASK / its map)."""
    cpu: int
    num: int                 # its bit in PEND / MASK
    name: str
    pending: bool
    enabled: bool
    routable: bool           # GIC_VL_CTL (the core's timer, counters, software)
    target: str


def local_rows(gic) -> list[LocalRow]:
    """Every core's local interrupts: pending, enabled, where they go (a pin,
    an EIC vector, NMI) — the core's own timer / counters / software
    interrupts only when routable (else wired inside the core)."""
    out = []
    for v, vp in enumerate(gic.vps):
        pend = gic.local_pending(v)
        for i, name in enumerate(G.LOCAL_NAMES):
            if i == G.LOCAL_FDC:
                continue                              # no FDC
            bit = {G.LOCAL_TIMER: G.CTL_TIMER, G.LOCAL_PERFCTR: G.CTL_PERFCNT,
                   G.LOCAL_SWINT0: G.CTL_SWINT, G.LOCAL_SWINT1: G.CTL_SWINT}.get(i)
            routable = bit is None or bool(gic.ctl_routable & bit)
            target = (G._target(vp.maps[G._MAP_INDEX[i]], vp.eic) if routable
                      else "wired in the core")
            out.append(LocalRow(v, i, name, bool(pend >> i & 1), bool(vp.mask >> i & 1),
                                routable, target))
    return out


def gic_core_lines(gic) -> list[str]:
    out = []
    for v, vp in enumerate(gic.vps):
        pend = gic.local_pending(v)
        local = ", ".join(G.LOCAL_NAMES[i] + (" PENDING" if pend >> i & 1 else "")
                          for i in range(7) if vp.mask >> i & 1)
        compare = "max" if vp.compare == (1 << 64) - 1 else str(vp.compare)
        lines = (f"EIC mode, RIPL {vp.pins}" if vp.eic else f"pins {G._bits(vp.pins)}")
        out.append(f"core {v}: {lines}" + (", NMI" if vp.nmi else "")
                   + f"; local: {local or 'none'}; COMPARE {compare}")
    return out


# ------------------------------------------------------------------ the registers

@dataclass
class RegRow:
    block: str               # "GCR", "GCR core 1", "CPC core 0", "GIC", ...
    name: str
    offset: int
    value: int
    meaning: str


def _gcr_global_cm3(cps, m) -> list[RegRow]:
    """A CM3's global registers (MD01196 ch. 5): 64 bits, the low halves
    shown (the high ones: GCR_*_UPPER-like, 0 below 4 GB)."""
    g = cps.gcr
    rows = []
    for off, name, meaning in (
            (P.GCR_CONFIG, "GCR_CONFIG", "PCORES (7:0), NUMIOCU (11:8), regions (19:16), "
                                         "ITU (31)"),
            (P.GCR_BASE, "GCR_BASE", "the GCRs' address (47:15), CM_DEFAULT_TARGET (1:0)"),
            (P.GCR_CONTROL, "GCR_CONTROL", "SYNCCTL (16), SHST_EN (3), SPEC_READ_EN (0)"),
            (P.GCR_REV, "GCR_REV", "CM3 = 8.0"),
            (P.GCR_ERROR_CAUSE + 4, "GCR_ERROR_CAUSE[63:32]", cps.cm.describe_error()),
            (P.GCR_ERROR_ADDR, "GCR_ERROR_ADDR", ""),
            (P.GCR_GIC_BASE, "GCR_GIC_BASE", "the GIC (47:17), enable (0)"),
            (P.GCR_CPC_BASE, "GCR_CPC_BASE", "the CPC (47:15), enable (0)"),
            (P.GCR_GIC_STATUS, "GCR_GIC_STATUS", "a GIC is connected"),
            (P.GCR_CPC_STATUS, "GCR_CPC_STATUS", "a CPC is connected"),
            (P.GCR_ACCESS_CM3, "GCR_ACCESS", "cores (5:0) / IOCUs (17:16) that may write "
                                             "the GCRs and the CPC"),
            (P.GCR_L2_CONFIG, "GCR_L2_CONFIG", "the L2: sets (15:12), line (11:8), ways - 1 "
                                               "(7:0), bypass (20)"),
            (P.GCR_SYS_CONFIG2, "GCR_SYS_CONFIG2", "MAXVPW: VP numbers a core"),
            (P.GCR_SEM, "GCR_SEM", "the uncached semaphore: SEM_LOCK (31), the owner"),
            (P.GCR_BEV_BASE, "GCR_BEV_BASE", "the BEV base of powered-up cores (stored)")):
        rows.append(RegRow("GCR", name, off, g.get(m, off), meaning))
    return sorted(rows, key=lambda r: r.offset)


def _gcr_global(cps, m) -> list[RegRow]:
    if cps.cm3:
        return _gcr_global_cm3(cps, m)
    g = cps.gcr
    rows = []
    for off, name, meaning in (
            (P.GCR_CONFIG, "GCR_CONFIG", "PCORES = cores - 1 (7:0), regions (19:16)"),
            (P.GCR_BASE, "GCR_BASE", "the GCRs' address (31:15), default region"),
            (P.GCR_CONTROL, "GCR_CONTROL", "SYNCCTL (16), SHST_EN (3), SPEC_READ_EN (0)"),
            (P.GCR_ACCESS, "GCR_ACCESS", "cores that may write the GCRs"),
            (P.GCR_REV, "GCR_REV", "CM2 = 6.0"),
            (P.GCR_ERROR_CAUSE, "GCR_ERROR_CAUSE", cps.cm.describe_error()),
            (P.GCR_ERROR_ADDR, "GCR_ERROR_ADDR", ""),
            (P.GCR_ERROR_MULT, "GCR_ERROR_MULT", "a second error's type"),
            (P.GCR_L2_ONLY_SYNC_BASE, "GCR_L2_ONLY_SYNC_BASE", "(31:12), enable (0)"),
            (P.GCR_GIC_BASE, "GCR_GIC_BASE", "the GIC (31:17), enable (0)"),
            (P.GCR_CPC_BASE, "GCR_CPC_BASE", "the CPC (31:15), enable (0)"),
            (P.GCR_GIC_STATUS, "GCR_GIC_STATUS", "a GIC is connected"),
            (P.GCR_CPC_STATUS, "GCR_CPC_STATUS", "a CPC is connected")):
        rows.append(RegRow("GCR", name, off, g.get(m, off), meaning))
    for i, b in enumerate(P.GCR_REGS[::2]):
        rows.append(RegRow("GCR", f"GCR_REG{i}_BASE", b, g.get(m, b), ""))
        rows.append(RegRow("GCR", f"GCR_REG{i}_MASK", b + 8, g.get(m, b + 8),
                           "mask (31:16), CCA override (7:4), DROPL2 (2), target (1:0)"))
    return sorted(rows, key=lambda r: r.offset)


def _gcr_core(cps, m, j) -> list[RegRow]:
    regs = cps.gcr.core[j]
    blk = f"GCR core {j}"
    coh = regs[P.CX_COHERENCE]
    if cps.cm3:                                  # COHEN; REDIRECT / RESET_BASE per VP
        rows = [RegRow(blk, "GCR_Cx_COHERENCE", P.CX_COHERENCE, coh,
                       "COHEN: coherent" if coh & P.COHEN else "COHEN = 0: not coherent"),
                RegRow(blk, "GCR_Cx_CONFIG", P.CX_CONFIG, cps.vpes - 1, "PVPE = VPs - 1"),
                RegRow(blk, "GCR_Cx_ID", P.CX_ID, j, ""),
                RegRow(blk, "GCR_Cx_RESET_EXT_BASE", P.CX_RESET_EXT_BASE,
                       regs[P.CX_RESET_EXT_BASE], "UseExceptionBase (30)")]
        for k in range(cps.vpes):
            v = j * cps.vpes + k
            r = cps.gcr.vp[v]
            red = r[P.CX_OTHER]
            vb = f"GCR core {j} VP {cps.vp_id(v)}"
            rows += [RegRow(vb, "GCR_Cx_REDIRECT", P.CX_OTHER, red,
                            f"core-other block = core {(red >> 8) & 0x3F} VP {red & 7}"
                            + (f", block {(red >> 24) & 3}" if red >> 24 & 3 else "")),
                     RegRow(vb, "GCR_Cx_RESET_BASE", P.CX_RESET_BASE, r[P.CX_RESET_BASE],
                            f"starts at 0x{cps.reset_base(j, k):08x}"
                            + (" (RESET_BASE_MODE: anywhere)"
                               if r[P.CX_RESET_BASE] & P.RESET_BASE_MODE else ""))]
        return rows
    return [
        RegRow(blk, "GCR_Cx_COHERENCE", P.CX_COHERENCE, coh,
               ("coherent" if coh >> j & 1 else "not coherent")),
        RegRow(blk, "GCR_Cx_OTHER", P.CX_OTHER, regs[P.CX_OTHER],
               f"core-other block = core {(regs[P.CX_OTHER] >> 16) & 0xFFFF}"),
        RegRow(blk, "GCR_Cx_RESET_BASE", P.CX_RESET_BASE, regs[P.CX_RESET_BASE],
               "the reset / BEV vector base"),
        RegRow(blk, "GCR_Cx_ID", P.CX_ID, j, ""),
        RegRow(blk, "GCR_Cx_RESET_EXT_BASE", P.CX_RESET_EXT_BASE, regs[P.CX_RESET_EXT_BASE],
               "UseExceptionBase (30): " + ("RESET_BASE" if regs[P.CX_RESET_EXT_BASE] & P.EXT_UEB
                                            else "0xBFC00000")),
    ]


_CMD_NAMES = {1: "ClockOff", 2: "PwrDown", 3: "PwrUp", 4: "Reset"}


def _cpc(cps, m) -> list[RegRow]:
    if cps.cpc.base is None:
        return [RegRow("CPC", "(disabled)", 0, 0, "GCR_CPC_BASE.EN = 0")]
    if cps.cm3:                                  # (GCR_ACCESS gates the CPC too)
        rows = [RegRow("CPC", "CPC_REVISION", P.CPC_REVISION, cps.cpc.get(m, P.CPC_REVISION),
                       ""),
                RegRow("CPC", "CPC_ROCC_CTL", P.CPC_ROCC_CTL, cps.cpc.rocc,
                       "reset occurred: cores (5:0), CM (16), DBU (17), cause (30:29), "
                       "CPC (31)")]
    else:
        rows = [RegRow("CPC", "CPC_ACCESS", P.CPC_ACCESS, cps.cpc.get(m, P.CPC_ACCESS), ""),
                RegRow("CPC", "CPC_REVISION", P.CPC_REVISION, cps.cpc.get(m, P.CPC_REVISION),
                       "")]
    for j in range(cps.n):
        stat = cps.cpc.stat_conf(j)
        rows.append(RegRow(f"CPC core {j}", "CPC_Cx_CMD", P.CPC_CX_CMD, cps.last_cmd[j],
                           _CMD_NAMES.get(cps.last_cmd[j], "none yet")
                           + (" (the last one)" if cps.last_cmd[j] else "")))
        rows.append(RegRow(f"CPC core {j}", "CPC_Cx_STAT_CONF", P.CPC_CX_STAT_CONF, stat,
                           "SEQSTATE " + P.SEQ_NAMES.get(cps.seq_state(j), "?")
                           + (f", waiting: {_CMD_NAMES[cps.pending[j]]}"
                              if cps.pending[j] else "")))
        if cps.cm3:
            run = cps.vp_mask[j] if cps.state[j] in (P.SEQ_U5, P.SEQ_D2) else 0
            rows.append(RegRow(f"CPC core {j}", "CPC_Cx_VP_RUNNING", P.CPC_VP_RUNNING, run,
                               "VPs running: " + (", ".join(
                                   str(cps.vp_id(j * cps.vpes + k)) for k in range(cps.vpes)
                                   if run >> k & 1) or "none")))
    return rows


def _gic(cps, m) -> list[RegRow]:
    gic = cps.gic
    if not gic.enabled:
        return [RegRow("GIC", "(disabled)", 0, 0, "GCR_GIC_BASE.EN = 0")]
    rows = [RegRow("GIC", "GIC_SH_CONFIG", G.SH_CONFIG, gic.get(m, G.SH_CONFIG),
                   "COUNTSTOP 28, COUNTBITS, NUMINTERRUPTS, PVPS"),
            RegRow("GIC", "GIC_SH_COUNTER", G.SH_COUNTER_LO, gic.counter & M32,
                   f"{gic.counter} (64 bits)"),
            RegRow("GIC", "GIC_SH_MASK31_0", G.SH_MASK, gic.mask & M32, "enabled sources"),
            RegRow("GIC", "GIC_SH_MASK63_32", G.SH_MASK + 4, gic.mask >> 32, ""),
            RegRow("GIC", "GIC_SH_PEND31_0", G.SH_PEND, gic.pending() & M32, "pending"),
            RegRow("GIC", "GIC_SH_PEND63_32", G.SH_PEND + 4, gic.pending() >> 32, "")]
    for v, vp in enumerate(gic.vps):
        blk = (f"GIC VP {cps.vp_id(v)}" if cps.cm3 and cps.vpes > 1 else f"GIC core {v}")
        rows += [RegRow(blk, "GIC_VL_CTL", G.VL_CTL, gic._get_vp(v, G.VL_CTL),
                        ("EIC mode" if vp.eic else "not EIC") + ", routable: "
                        + (", ".join(n for b, n in ((G.CTL_TIMER, "timer"),
                                                    (G.CTL_PERFCNT, "counters"),
                                                    (G.CTL_SWINT, "software"))
                                     if gic.ctl_routable & b) or "none")),
                 RegRow(blk, "GIC_VL_PEND", G.VL_PEND, gic.local_pending(v), ""),
                 RegRow(blk, "GIC_VL_MASK", G.VL_MASK, vp.mask, ""),
                 RegRow(blk, "GIC_VL_COMPARE_MAP", G.VL_MAP + 4, vp.maps[1],
                        G._target(vp.maps[1], vp.eic)),
                 RegRow(blk, "GIC_VL_TIMER_MAP", G.VL_MAP + 8, vp.maps[2],
                        G._target(vp.maps[2], vp.eic)),
                 RegRow(blk, "GIC_VL_PERFCTR_MAP", G.VL_MAP + 0x10, vp.maps[4],
                        G._target(vp.maps[4], vp.eic)),
                 RegRow(blk, "GIC_VL_SWINT0_MAP", G.VL_MAP + 0x14, vp.maps[5],
                        G._target(vp.maps[5], vp.eic)),
                 RegRow(blk, "GIC_VL_SWINT1_MAP", G.VL_MAP + 0x18, vp.maps[6],
                        G._target(vp.maps[6], vp.eic))]
        rows += [RegRow(blk, f"GIC_VL_EIC_SHADOW_SET{k}", G.VL_EIC_SS + 4 * k, ss,
                        f"vector {k} -> shadow set {ss}")
                 for k, ss in enumerate(vp.eic_ss) if ss]
        rows.append(RegRow(blk, "GIC_VL_COMPARE", G.VL_COMPARE_LO, vp.compare & M32,
                           "max" if vp.compare == (1 << 64) - 1 else str(vp.compare)))
    return rows


def register_rows(s) -> list[RegRow]:
    """The CPS registers (GCR, its core blocks, CPC, GIC), as each core's
    own view would read them."""
    cps = s.cps
    m0 = s.machines[0]
    rows = _gcr_global(cps, m0)
    for j in range(cps.n):
        rows += _gcr_core(cps, s.machines[j], j)
    return rows + _cpc(cps, m0) + _gic(cps, m0)


# ------------------------------------------------------------------ text (the CLI)

def format_cps(s) -> str:
    out = []
    block = None
    for r in register_rows(s):
        if r.block != block:
            block = r.block
            out.append(f"{block}:")
        out.append(f"  {r.name:<24} +0x{r.offset:04x}  0x{r.value:08x}"
                   + (f"  {r.meaning}" if r.meaning else ""))
    return "\n".join(out)
