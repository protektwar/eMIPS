"""The MT Module at a glance, for the GUI's MT pane (Qt-free, side-effect
free): the VPEs, every thread context (one core, or each core of a CPS),
the policy manager and the ITC cells.  Everything is read from
:class:`emips.mt.Processor` without binding a TC or touching a cache."""

from __future__ import annotations

from dataclasses import dataclass

from . import cp0 as C
from . import mt as MT


@dataclass(frozen=True)
class VPERow:
    vp: int
    core: int | None
    vpe: int
    vpa: bool
    mvp: bool
    te: bool
    excpt: int
    yqmask: int
    exclusive: str          # "" or why only the XTC runs
    xtc: int


@dataclass(frozen=True)
class TCRow:
    key: int                # the index into processors() x tcs (the GUI's row id)
    core: int | None
    tc: int
    vp: int
    state: str
    pc: int
    line: int | None
    flags: str
    group: int | None
    issued: int
    waits: str
    next: bool


@dataclass(frozen=True)
class CellRow:
    cell: int
    kind: str
    address: int | None
    flags: str
    data: str


def processors(s) -> list[tuple[int | None, int, MT.Processor]]:
    """(core or None, the VP of its VPE 0, processor) for the session."""
    mt = s.mt
    if mt is None:
        return []
    if isinstance(mt, MT.Cluster):
        v = s.cps.vpes
        return [(c, c * v, p) for c, p in enumerate(mt.procs)]
    return [(None, 0, mt)]


def header(s) -> list[str]:
    out = []
    for core, _, p in processors(s):
        ctl = p.mvpcontrol
        who = "" if p.evp else (f" — EVP = 0: only TC{p.solo.n} runs" if p.solo else "")
        out.append((f"core {core}: " if core is not None else "")
                   + f"{len(p.vpes)} VPE(s), {len(p.tcs)} TC(s); MVPControl 0x{ctl:08x} "
                   f"(EVP {int(bool(ctl & MT.EVP))}, VPC {int(bool(ctl & MT.VPC))}, STLB "
                   f"{int(bool(ctl & MT.STLB))}, CPA {int(bool(ctl & MT.CPA))}){who}")
        out.append("  " + p.pm_line())
        out.append(f"  YIELD qualifier inputs 0x{p.yq:04x}"
                   + (f" (the GIC 0x{p.yq_gic:04x})" if p.yq_gic else "")
                   + f"; idle cycles {p.idle_cycles}")
    return out


def vpe_rows(s) -> list[VPERow]:
    out = []
    for core, base, p in processors(s):
        for v, m in enumerate(p.vpes):
            vc = p.vpecontrol[v]
            why = ""
            if p.exclusive(v):
                why = ("Debug Mode" if m._dm else "EXL / ERL"
                       if m.cp0[C.STATUS]["value"] & (C.ST_EXL | C.ST_ERL) else "TE = 0")
            out.append(VPERow(base + v, core, v, p.vpa[v], p.mvp[v], bool(vc & MT.TE),
                              (vc >> MT.EXCPT_SHIFT) & 7, p.yqmask[v], why, p.xtc[v]))
    return out


def tc_rows(s) -> list[TCRow]:
    out = []
    key = 0
    for core, base, p in processors(s):
        for r in p.rows():
            flags = " ".join(f for f, on in (("DA", r["da"]), ("IXMT", r["ixmt"]),
                                              ("DT", r["dt"]), ("XTC", r["xtc"]),
                                              ("STP", r["stp"])) if on)
            waits = r["gs"] or (f"YIELD 0x{r['yield']:04x}" if r["yield"] else "")
            src = s.source_line(r["pc"], base + r["vpe"]) if s.program is not None else None
            out.append(TCRow(key, core, r["tc"], base + r["vpe"], r["state"], r["pc"],
                             src.line if src else None, flags, r["group"], r["issued"],
                             waits, r["next"]))
            key += 1
    return out


def tc_of(s, key: int) -> tuple[MT.Processor, MT.TC, int] | None:
    """The processor, TC and its VP of a :class:`TCRow` key."""
    k = 0
    for _, base, p in processors(s):
        for tc in p.tcs:
            if k == key:
                return p, tc, base + tc.vpe
            k += 1
    return None


def cell_rows(s) -> list[CellRow]:
    procs = processors(s)
    itc = procs[0][2].itc if procs else None
    if itc is None:
        return []
    out = []
    for r in itc.rows():
        flags = "".join(f for f, on in (("E", r["empty"]), ("F", r["full"]),
                                        ("T", r["trap"])) if on) or "-"
        if r["kind"] == "FIFO":
            data = " ".join(f"{w:08x}" for w in r["data"][:r["count"]]) or "(empty)"
        else:
            data = f"{r['data'][0]:08x}"
        out.append(CellRow(r["cell"], r["kind"], r["address"], flags, data))
    return out


def itc_header(s) -> str:
    procs = processors(s)
    itc = procs[0][2].itc if procs else None
    if itc is None:
        return "no ITC (MVPConf0.GS = 0)"
    return itc.lines()[0]
