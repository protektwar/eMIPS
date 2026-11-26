"""The window's other panes as data, Qt-free (M29 W3): the register tabs
(PC / CP0, FPU, DSP, MSA, Vector, Interrupts, Debug), Caches, TLB, Counters,
Trace, Board, System, MT and the Tools — each a list of *blocks* (tables,
lines, the address split into fields, notes) and *controls* (a choice, a
check box, a button, a value to set), built from the modules the window's
panes read (``cacheview``, ``tlbview``, ``cpsview``, ``mtview``, ``perf``,
``tools``, the Session's ``*_lines`` / ``*_rows``).  The web page draws any
pane from them; :func:`act` does what a pane's buttons do (the window's
panes call the same Session methods).

Blocks: ``{"kind": "note", "text"}``, ``{"kind": "lines", "title", "lines"}``,
``{"kind": "table", "title", "heads", "rows", "fills", "bold", "tips"}`` (a
fill is a token: ``valid`` / ``hit`` / ``miss`` / ``current`` / ``error`` /
``band:ST``), ``{"kind": "fields", "title", "cells"}`` (name, where, bits,
value).  Controls: ``{"kind": "select" | "check" | "text" | "button", "key",
"label", "value", "items", "tip"}`` — a select / check / text sets the
pane's option ``key`` (``text`` with ``action``: a button beside it acts).
Views never change the machine (as the window's): only :func:`act` does.
"""

from __future__ import annotations

import struct

PANES = ("special", "fpu", "dsp", "msa", "vector", "interrupts", "debug", "caches", "tlb",
         "counters", "trace", "board", "system", "mt", "tools")
TITLES = {"special": "PC / CP0", "fpu": "FPU", "dsp": "DSP", "msa": "MSA", "vector": "Vector",
          "interrupts": "Interrupts", "debug": "Debug", "caches": "Caches", "tlb": "TLB",
          "counters": "Counters", "trace": "Trace", "board": "Board", "system": "System",
          "mt": "MT", "tools": "Tools"}


def note(text: str) -> dict:
    return {"kind": "note", "text": text}


def lines(title: str, rows: list[str]) -> dict:
    return {"kind": "lines", "title": title, "lines": list(rows)}


def table(title: str, heads: list[str], rows: list[list[str]], fills=None, bold=None,
          tips=None) -> dict:
    return {"kind": "table", "title": title, "heads": list(heads),
            "rows": [[str(c) for c in r] for r in rows],
            "fills": list(fills) if fills else [None] * len(rows),
            "bold": list(bold) if bold else [False] * len(rows),
            "tips": list(tips) if tips else None}


def available(s) -> list[str]:
    """The panes this session has (as the window shows its tabs / docks)."""
    m = s.machine
    out = ["special", "fpu", "interrupts", "debug", "caches", "tlb", "counters", "trace",
           "board", "system", "mt", "tools"]
    if m is not None and getattr(m, "dsp", None):
        out.insert(2, "dsp")
    if m is not None and getattr(m, "msa", False):
        out.insert(2, "msa")
    if m is not None and m.rv and getattr(m, "vector", None) is not None:
        out.insert(2, "vector")
    return out


def pane(s, name: str, opts: dict | None = None, memo: dict | None = None) -> dict:
    """Pane ``name``'s blocks and controls; ``opts`` its options (what the
    controls set), ``memo`` a dict the caller keeps between calls (the
    values of the last step: what changed is highlighted)."""
    opts = opts or {}
    memo = memo if memo is not None else {}
    if s.machine is None and name != "tools":
        return {"blocks": [note("No program: assemble or open one.")], "controls": []}
    return globals()[f"_{name}"](s, opts, memo)


def act(s, name: str, action: str, args: dict | None = None) -> str:
    """A pane's button (or a value set): what the window's pane does.  The
    text says what happened ("" = nothing to say)."""
    args = args or {}
    m = s.machine
    if action == "reset_cache_stats" and m.caches is not None:
        m.caches.reset_stats()
    elif action == "reset_tlb_stats":
        from . import tlbview as tv
        tv.reset_stats(m)
    elif action == "counter":
        i = int(args["i"])
        event = args.get("event")
        event = None if event in (None, "", "off") else int(event)
        modes = args.get("modes") or ("uk" if event is not None else "")
        s.configure_counter(i, event, modes, bool(args.get("ie")))
    elif action == "trace":
        _trace_apply(s, args)
    elif action == "board_nmi":
        s.board_nmi()
    elif action == "dint":
        s.debug_interrupt()
    elif action == "rnmi":
        s.rnmi(int(args.get("cause", 0) or 0))
    elif action == "core":
        s.select(int(args["cpu"]))
    elif action == "yq":
        s.yield_inputs(int(str(args["value"]).strip(), 0))
    elif action == "sys":
        s.pm_input(int(args["value"]))
    else:
        raise ValueError(f"no action {action!r} in pane {name!r}")
    return ""


# -- the register tabs ------------------------------------------------------------------------

def _special(s, opts, memo) -> dict:
    rows = s.special_registers()
    return {"blocks": [table("PC / CSRs" if s.rv else "PC / CP0", ["register", "value"],
                             [[n, f"0x{v:0{s.hex_digits(n)}x}"] for n, v in rows])],
            "controls": []}


def _fpu(s, opts, memo) -> dict:
    m = s.machine
    if m.rv:
        from .riscv import FPR_NAMES
        from .rvfpu import show
        if m.flen == 0:
            return {"blocks": [note("Zfinx: the floating-point values are in the x registers"
                                    if m.inx else "This hart has no F / D extension: no f "
                                    "registers")], "controls": []}
        return {"blocks": [table("f registers", ["register", "bits", "value"],
                                 [[f"f{i} ({FPR_NAMES[i]})", f"0x{raw:0{m.flen // 4}x}",
                                   show(raw, m.flen)] for i, raw in enumerate(m.fpr)])],
                "controls": []}
    if not m.has_fpu:
        return {"blocks": [note("This core has no FPU (Config1.FP = 0): Status.CU1 reads 0, "
                                "every FPU instruction is Coprocessor Unusable")], "controls": []}
    fpr = m.fpr
    fr64 = bool(m.status & (1 << 26))
    rows = []
    for i, raw in enumerate(fpr):
        lo = raw & 0xFFFFFFFF
        single = struct.unpack(">f", lo.to_bytes(4, "big"))[0]
        if fr64:
            dbl = struct.unpack(">d", raw.to_bytes(8, "big"))[0]
            rows.append([f"$f{i}", f"0x{raw:016x}", f"{single:.7g} / {dbl:.15g}"])
        elif i % 2:
            rows.append([f"$f{i}", f"0x{lo:08x}", f"{single:.7g} / (the high half of "
                                                  f"$f{i - 1})"])
        else:
            pair = ((fpr[i + 1] & 0xFFFFFFFF) << 32) | lo
            dbl = struct.unpack(">d", pair.to_bytes(8, "big"))[0]
            rows.append([f"$f{i}", f"0x{lo:08x}", f"{single:.7g} / {dbl:.17g} "
                                                  f"($f{i + 1}:$f{i})"])
    return {"blocks": [note(f"Status.FR = {int(fr64)}: " + ("64-bit FPRs" if fr64 else
                                                            "32-bit FPRs, a double in an "
                                                            "even / odd pair")),
                       table("FPRs", ["register", "bits", "single / double"], rows)],
            "controls": []}


def _changed(memo: dict, key, values: list) -> list:
    """The values of the step before (the last step's changes are
    highlighted, as the window's tabs do)."""
    step = memo.get("step")
    prev = memo.get(key)
    if prev is None or memo.get(f"{key}@") != step:
        memo[f"{key}.old"] = prev if prev is not None and len(prev) == len(values) else values
        memo[key] = values
        memo[f"{key}@"] = step
    return memo[f"{key}.old"]


def _dsp(s, opts, memo) -> dict:
    rows = s.dsp_rows()
    old = _changed(memo, "dsp", [v for _, v, _ in rows])
    cells, fills = [], []
    for (name, value, why), o in zip(rows, old):
        digits = 16 if s.wide and name.startswith("ac") else 8
        cells.append([name, f"0x{value:0{digits}x}" if not name.startswith("  ") else
                      (f"{value:04b}" if "ccond" in name else
                       f"0x{value:02x}" if "ouflag" in name else str(value)), why or ""])
        fills.append("current" if o != value else None)
    return {"blocks": [table("DSP Module", ["register", "value", ""], cells, fills)],
            "controls": []}


def _lanes(title, rows, bits, old, text_lines, view_key, views, view) -> dict:
    lanes = len(rows[0][2]) if rows else 0
    mask = (1 << bits) - 1
    cells, fills, bold = [], [], []
    for (name, value, texts), o in zip(rows, old):
        cells.append([name, *texts])
        hot = [(o >> (i * bits) & mask) != (value >> (i * bits) & mask)
               for i in range(len(texts))]
        fills.append("current" if any(hot) else None)
        bold.append(any(hot))
    return {"blocks": [lines("", text_lines),
                       table(title, ["reg"] + [f"[{i}]" for i in range(lanes)], cells, fills,
                             bold)],
            "controls": [{"kind": "select", "key": view_key, "label": "elements",
                          "items": [[k, k] for k in views], "value": view}]}


def _msa(s, opts, memo) -> dict:
    view = opts.get("view") or "w.x"
    if view not in s.MSA_VIEWS:
        view = "w.x"
    rows = s.msa_rows(view)
    old = _changed(memo, ("msa", view), [v for _, v, _ in rows])
    return _lanes("vector registers (element 0 first)", rows, s.MSA_VIEWS[view][0], old,
                  s.msa_control_lines(), "view", list(s.MSA_VIEWS), view)


def _vector(s, opts, memo) -> dict:
    from .sessionrv import VECTOR_VIEWS, vector_rows, vector_state_lines
    m = s.machine
    view = opts.get("view") or "e32.x"
    if view not in VECTOR_VIEWS:
        view = "e32.x"
    rows = vector_rows(m, view)
    old = _changed(memo, ("vec", view), [v for _, v, _ in rows])
    return _lanes("vector registers (element 0 first)", rows, VECTOR_VIEWS[view][0], old,
                  vector_state_lines(m), "view", list(VECTOR_VIEWS), view)


def _interrupts(s, opts, memo) -> dict:
    m = s.machine
    if s.rv:
        controls = []
        if m.csrs.smrnmi:
            controls.append({"kind": "button", "action": "rnmi", "label": "RNMI",
                             "tip": "A resumable non-maskable interrupt (Smrnmi)",
                             "disabled": m.rnmi_pending is not None or bool(s.halted)})
        return {"blocks": [lines("", [ln.strip() for ln in s.interrupt_lines()])],
                "controls": controls}
    st = s.interrupt_state()
    blocks = [lines("", [ln.strip() for ln in s.interrupt_lines() if "SRSMap:" not in ln])]
    controls = []
    n = st["sets"]
    if n > 1:
        k = int(opts.get("set", st["css"]))
        k = k if 0 <= k < n else st["css"]
        items = []
        for j in range(n):
            mark = ["current" if j == st["css"] else "",
                    "previous" if j == st["pss"] and j != st["css"] else ""]
            label = ", ".join(x for x in mark if x)
            items.append([str(j), f"set {j}" + (f" ({label})" if label else "")])
        controls.append({"kind": "select", "key": "set", "label": "shadow set",
                         "items": items, "value": str(k)})
        blocks.append(note(
            f"EIC mode: the controller gives the set (SRSCtl.EICSS {st['eicss']})"
            if st["mode"] == "eic" else
            "SRSMap, vector → set: " + " ".join(f"{v}→{j}" for v, j in enumerate(st["srsmap"]))))
        d = s.hex_digits()
        bits = 4 * d
        blocks.append(table(f"shadow set {k}", ["register", "value", "signed"],
                            [[name, f"0x{v:0{d}x}", str(v - (1 << bits) if v >> (bits - 1)
                                                        else v)]
                             for name, v in s.shadow_set(k)]))
    return {"blocks": blocks, "controls": controls}


def _debug(s, opts, memo) -> dict:
    m = s.machine
    controls = []
    if not s.rv:
        controls.append({"kind": "button", "action": "dint", "label": "Debug Interrupt",
                         "tip": "The DINT input: the core enters Debug Mode (EJTAG)",
                         "disabled": bool(m.debug_mode) or bool(s.halted)})
    return {"blocks": [lines("", [ln.strip() for ln in s.debug_lines()])],
            "controls": controls}


# -- caches ----------------------------------------------------------------------------------

def _caches(s, opts, memo) -> dict:
    from . import cacheview as cv
    m = s.machine
    caches = [] if m.caches is None else m.caches.all
    if not caches:
        return {"blocks": [note("No caches. Turn them on in Settings > Memory (the "
                                "program restarts).")], "controls": []}
    names = [c.name for c in caches]
    want = opts.get("which")
    c = caches[names.index(want)] if want in names else \
        caches[names.index("D-cache")] if "D-cache" in names else caches[0]
    acc = cv.last_access(s.cache_accesses(), c)
    focus = acc.set if acc is not None else None
    all_sets = bool(opts.get("all")) and c.sets <= cv.ALL_SETS_LIMIT
    blocks = [note(f"{c.name}: {c.geometry.describe()}")]
    if acc is None:
        blocks.append(note("not used in the last step — cells: tag (hex), a = age (0 = most "
                           "recently used), " + ("M / E / S = Modified / Exclusive / Shared "
                                                 "(MESI)" if c.mesi else "D = dirty")
                           + ", L = locked"))
    else:
        sp = cv.split(c, acc.va, acc.pa)
        cells = []
        for name, msb, lsb, value, src in sp.fields():
            if msb < lsb:
                continue
            addr = sp.pa if src == "PA" else sp.va
            cells.append([name, f"{src}[{msb}:{lsb}]",
                          format((addr >> lsb) & ((1 << (msb - lsb + 1)) - 1),
                                 f"0{msb - lsb + 1}b"), f"0x{value:x} ({value})"])
        blocks.append({"kind": "fields", "title": acc.describe(), "cells": cells})
    rows = cv.pane_sets(c, all_sets=all_sets, focus=focus)
    big = m.big_endian
    cells, fills, bold, tips = [], [], [], []
    for sidx in rows:
        row, rfill, rbold, rtip = [f"set {sidx}"], [None], [False], [""]
        for wv in cv.set_view(c, sidx):
            f = "valid" if wv.valid else None
            b = False
            if acc is not None and acc.set == sidx and acc.way == wv.way:
                f, b = ("hit" if acc.hit else "miss"), True
            row.append(cv.cell_text(c, wv))
            rfill.append(f)
            rbold.append(b)
            rtip.append(cv.cell_tip(c, sidx, wv, big))
        cells.append(row)
        fills.append(rfill)
        bold.append(rbold)
        tips.append(rtip)
    t = table("", [""] + [f"way {w}" for w in range(c.ways)], cells)
    t["cellfills"], t["cellbold"], t["celltips"] = fills, bold, tips
    blocks.append(t)
    text = cv.stats_lines(c)
    if c.kind == "D" and m.caches.wbb is not None:
        text += cv.wbb_lines(m.caches.wbb)
    cl = s.rvcps
    if cl is not None:
        core = m.p8700.core
        if c.kind == "D":
            text += [""] + cl.l1pf[core].lines()
            if cl.dspram[core] is not None:
                text += cl.dspram[core].lines()
        elif c.kind == "S" and cl.cm.l2pf is not None:
            text += [""] + cl.cm.l2pf.lines()
            text += [ln for ln in cl.cm.describe() if ln.startswith(("MMIO", "IOCU", "shared"))]
        text += cl.cdma.lines() if c.kind == "S" else []
    blocks.append(lines("statistics", text))
    return {"blocks": blocks, "controls": [
        {"kind": "select", "key": "which", "label": "cache", "items": [[n, n] for n in names],
         "value": c.name},
        {"kind": "check", "key": "all", "label": "all sets", "value": all_sets,
         "disabled": c.sets > cv.ALL_SETS_LIMIT,
         "tip": f"List every set (caches of up to {cv.ALL_SETS_LIMIT} sets)"},
        {"kind": "button", "action": "reset_cache_stats", "label": "Reset statistics"}]}


# -- TLB ---------------------------------------------------------------------------------------

def _tlb(s, opts, memo) -> dict:
    from . import tlbview as tv
    m = s.machine
    if m.rv:
        return _tlb_rv(s, m)
    kind = tv.kind(m)
    if kind == "fixed":
        return {"blocks": [note("No TLB: the fixed mapping MMU translates by fixed rules. "
                                "Pick a TLB in Settings > Processor > MMU (the program "
                                "restarts).")], "controls": []}
    reset = {"kind": "button", "action": "reset_tlb_stats", "label": "Reset statistics"}
    if kind == "mpu":
        mpu = m.mmu
        rows = []
        if mpu.segments:
            for k in range(16):
                rows.append([f"segment {k}", f"0x{k << 28:08x}-0x{(k << 28) | 0x0FFFFFFF:08x}",
                             f"C{mpu.seg[k] & 7}", tv.mpu_flags(mpu.seg[k])])
        fills = [None] * len(rows)
        for r in range(mpu.n):
            span = mpu.span(r)
            if span is not None:
                bits = mpu.ctl[r] & 0x3F
                rows.append([f"region {r}", f"0x{span[0]:08x}-0x{span[1] - 1:08x}",
                             f"C{bits & 7}", tv.mpu_flags(bits)])
                fills.append("valid")
        return {"blocks": [lines("", [x.strip() for x in mpu.lines()[:3]]),
                           table("", ["segment / region", "addresses", "CCA", "inhibits"],
                                 rows, fills),
                           lines("statistics", tv.stats_lines(m))], "controls": [reset]}
    accesses = s.tlb_accesses()
    acc = tv.last_access(accesses)
    blocks = [note(m.mmu.name)]
    ops = [a.note for _w, a in accesses if a.note]
    if acc is None:
        blocks.append(note("no translation through the TLB in the last step (unmapped "
                           "addresses, e.g. kseg0, are not translated)"))
    else:
        blocks.append(note(acc.describe()))
        if acc.page and acc.entry is not None:
            sp = tv.VASplit(acc.va, acc.page)
            cells = [[name, f"VA[{msb}:{lsb}]" if msb != lsb else f"VA[{msb}]",
                      tv.bits(acc.va, msb, lsb), f"0x{value:x}"]
                     for name, msb, lsb, value in sp.fields()]
            title = ""
            if acc.pa is not None:
                title = (f"{acc.name}: {tv.HALF[acc.half]} page -> PFN 0x{acc.pfn:x}; PA = "
                         f"0x{acc.frame:08x} + offset = 0x{acc.pa:08x}")
            blocks.append({"kind": "fields", "title": title, "cells": cells})
    if ops:
        blocks.append(lines("", ops))

    def mark(entry):
        if acc is not None and acc.entry == entry:
            return ("miss" if acc.fault else "hit"), True
        return None, False
    mmu = m.mmu
    if kind == "bat":
        rows, fills, bold = [], [], []
        for i, e in enumerate(mmu.entries):
            f, b = mark(i)
            if f is None and e.lo & tv.LO_V:
                f = "valid"
            rows.append([str(i), mmu.entry_how(i), f"0x{e.bounds << 12 | 0xFFF:08x}",
                         f"0x{e.pfn << 12:08x}", f"C{(e.lo >> 3) & 7} {tv.lo_flags(e.lo)}"])
            fills.append(f)
            bold.append(b)
        blocks.append(table("", ["entry", "use", "bounds (last VA)", "base", "attributes"],
                            rows, fills, bold))
    else:
        everything = bool(opts.get("all"))
        rows_ = tv.rows(mmu, m, everything=everything)
        if acc is not None and acc.entry is not None and acc.entry not in {r.index for r in rows_}:
            rows_ = sorted(rows_ + [tv.entry_row(mmu, m, acc.entry)], key=lambda r: r.index)
        rows, fills, bold, tips = [], [], [], []
        for r in rows_:
            f, b = mark(r.index)
            if f is None and r.used:
                f = "valid"
            marks = ("W" if r.wired else "") + ("R" if r.random else "")
            name = str(r.index) if not mmu.ftlb else (
                f"V{r.index}" if r.index < mmu.n else
                f"F{(r.index - mmu.n) % mmu.fsets}.{(r.index - mmu.n) // mmu.fsets}")
            if r.used:
                end = r.va + r.span - 1
                rows.append([name, marks, f"0x{r.va:08x}-0x{end & (0xFFFFFFFF if r.va >> 32 == 0 else (1 << 64) - 1):08x}",
                             "G" if r.g else f"ASID {r.asid}", tv.size_name(r.page),
                             tv.half_text(r, 0), tv.half_text(r, 1)])
            else:
                rows.append([name, marks, r.state, "", "", "", ""])
            fills.append(f)
            bold.append(b)
            tips.append(tv.format_entry(m, r.index))
        blocks.append(table("entries (W = wired, R = Random: the entry TLBWR writes next)",
                            ["entry", "", "VA (page pair)", "owner", "page",
                             "even page (EntryLo0)", "odd page (EntryLo1)"],
                            rows, fills, bold, tips))
        slots = [(u, i, e) for u in (mmu.itlb, mmu.dtlb) for i, e in enumerate(u.slots)]
        rows, fills, bold = [], [], []
        for u, i, e in slots:
            victim = tv.micro_victim(u) == i
            f, b = None, False
            if acc is not None and e is not None and acc.micro == u.name \
                    and e.matches(acc.va, acc.asid):
                f, b = ("hit" if acc.micro_hit else "miss"), True
            elif e is not None:
                f = "valid"
            rows.append([u.name, str(i) + (" *" if victim else ""), "-", "", "", "", ""]
                        if e is None else
                        [u.name, str(i) + (" *" if victim else ""), f"0x{e.vbase:08x}",
                         tv.size_name(e.size), "G" if e.g else f"ASID {e.asid}",
                         f"0x{e.pbase:08x} C{e.cca}", str(e.index)])
            fills.append(f)
            bold.append(b)
        blocks.append(table("micro-TLBs (* = replaced by the next miss)",
                            ["micro-TLB", "slot", "VA", "size", "owner", "PA", "from entry"],
                            rows, fills, bold))
    blocks.append(lines("statistics and registers", tv.stats_lines(m) + [""] +
                        [f"{n:<9} 0x{v:08x}" for n, v in tv.registers(m)]))
    controls = [reset]
    if kind != "bat":
        controls.insert(0, {"kind": "check", "key": "all", "label": "all entries",
                            "value": bool(opts.get("all"))})
    return {"blocks": blocks, "controls": controls}


def _tlb_rv(s, m) -> dict:
    mmu = m.mmu
    if not hasattr(mmu, "tlb"):
        return {"blocks": [note("No paging: this hart has no S-mode (Settings: RISC-V "
                                "privilege modes): PA = VA, PMP checks the accesses.")],
                "controls": []}
    scheme, asid, root = mmu._satp(m)
    blocks = [note(f"satp: {scheme.name if scheme is not None else 'Bare'}, ASID {asid}, root "
                   f"page table at PA 0x{root << 12:x} — M-mode and Bare translate PA = VA")]
    if hasattr(mmu, "rows"):                    # the P8700's VTLB / FTLB
        names = "VRWXUGAD"
        rows = [[where, asid_, f"0x{vpn:x}", size if leaf else f"{size} (non-leaf)",
                 f"0x{ppn:x}", "".join(c if pte >> k & 1 else "-" for k, c in enumerate(names))]
                for where, asid_, vpn, size, ppn, pte, leaf in mmu.rows(m)]
        blocks.append(table("", ["entry", "ASID", "VPN", "page", "PPN", "PTE flags"], rows))
        blocks.append(lines("", [mmu.lines(m)[0], mmu.lines(m)[-1].strip()]))
        return {"blocks": blocks, "controls": []}
    from .rvmmu import entry_rows
    h = m.csrs.has_h
    heads = ["ASID", "page", "size", "PA", "PTE flags"]
    if h:
        g, vmid, _groot = mmu._hgatp(m)
        vs, vasid, _vroot = mmu._atp(m, m.csrs.vsatp)
        blocks.append(note(f"{m.mode_name}-mode, V = {m.v}; hgatp: "
                           f"{g.name + 'x4' if g else 'Bare'}, VMID {vmid}; vsatp: "
                           f"{vs.name if vs else 'Bare'}, ASID {vasid} — a guest access "
                           "translates twice (VS-stage, G-stage)"))
        heads = ["stage", "VMID"] + heads[:3] + ["GPA", "PA", "VS / PTE flags", "G-stage flags"]
    entries = entry_rows(m)
    rows = [list(r) if h else [r[2], r[3], r[4], r[6], r[7]] for r in entries]
    blocks.append(table("the TLB (most recent first)", heads, rows,
                        ["valid"] + [None] * (len(rows) - 1) if rows else None))
    blocks.append(lines("", [f"TLB: {len(entries)} of {mmu.entries} entries",
                             f"hits {mmu.hits}   misses {mmu.misses}   page-table walks "
                             f"{mmu.walks}"]))
    return {"blocks": blocks, "controls": []}


# -- counters ----------------------------------------------------------------------------------

def _counters(s, opts, memo) -> dict:
    from . import perf
    m = s.machine
    if m.rv:
        c = m.csrs
        if m.p8700 is not None:
            text = [f"RISC-V: the counters are CSRs — cycle {c.read(0xB00):,}, instret "
                    f"{c.read(0xB02):,}.  The MIPS P8700's mhpmcounter3-6 (mhpmevent3-6: "
                    "csrw, or the debugger's perf N EVENT):"] + m.p8700.counters.lines()
        elif c.hpm is not None:
            text = [f"RISC-V: the counters are CSRs (Zicntr / Zihpm) — cycle {c.read(0xB00):,}, "
                    f"instret {c.read(0xB02):,}, time {m.mtime:,}.  mhpmcounter3-31 count the "
                    "SBI PMU event written to mhpmevent3-31 (csrw, or the debugger's perf N "
                    "EVENT; perf events lists them):"] + c.hpm.lines()
        else:
            text = [f"RISC-V: the counters are CSRs (Zicntr) — cycle {c.read(0xB00):,}, "
                    f"instret {c.read(0xB02):,}, time {m.mtime:,}; this hart has no "
                    "mhpmcounter3-31 (Zihpm)."]
        return {"blocks": [lines("", text)], "controls": []}
    rows, counters = [], []
    for i in range(2):
        ctl = m.cp0_read(*perf.PERFCTL[i])
        on = bool(ctl & perf.CTL_MODES)
        event = (ctl >> 5) & 0x7F
        modes = "".join(k for k, bit in (("u", perf.CTL_U), ("s", perf.CTL_S),
                                         ("k", perf.CTL_K), ("x", perf.CTL_EXL)) if ctl & bit)
        count = m.cp0_read(*perf.PERFCNT[i])
        ev = perf.EVENTS[i].get(event)
        rows.append([f"PerfCnt{i}", f"{event}: {ev.name}" if on and ev else "(stopped)",
                     modes.upper() or "-", "yes" if ctl & perf.CTL_IE else "no", f"{count:,}"])
        items = [["off", "(stopped)"]] + [
            [str(n), f"{n}: {e.name}" + {"the 24K pipeline model": " [24K model]",
                                          "not modelled": " [not modelled]"}.get(e.source, "")]
            for n, e in sorted(perf.EVENTS[i].items()) if e.key]
        counters.append({"kind": "counter", "i": i, "items": items,
                         "event": str(event) if on else "off", "modes": modes,
                         "ie": bool(ctl & perf.CTL_IE), "supervisor": bool(m.mmu.supervisor)})
    blocks = [table("", ["counter", "event", "counts in", "interrupt on overflow", "count"], rows)]
    ratio = _ratio(m)
    if ratio:
        blocks.append(note(ratio))
    blocks.append(note("Setting a counter here is what software does with mtc0 $25 (select "
                       "0-3); the setting is kept across Reset and Step Back. Events marked "
                       "[24K model] count in the 24K pipeline model only."))
    return {"blocks": blocks, "controls": counters}


def _ratio(m) -> str:
    from . import perf
    rows = perf.rows(m)
    on = [(i, cnt, ev) for i, ctl, cnt, ev in rows if ctl & perf.CTL_MODES and ev]
    text = ""
    if len(on) == 2 and on[0][1]:
        (_, a, ea), (_, b, eb) = on
        text = f"{eb.name} / {ea.name} = {b / a:.3f}"
        if {ea.key, eb.key} == {"cycles", "instructions"}:
            cyc = a if ea.key == "cycles" else b
            ins = b if ea.key == "cycles" else a
            if ins:
                text += f"  —  CPI {cyc / ins:.2f}"
    if m.perf.pci:
        text += ("  " if text else "") + "Cause.PCI: a counter interrupt is pending"
    return text


# -- trace -------------------------------------------------------------------------------------

_MIPS_TRACE = (("addr", "Addresses", "load and store addresses (TLA / TSA)"),
               ("data", "Data", "load and store data (TD)"),
               ("ca", "Cycle-accurate", "CA: a TF1 for every cycle without a completed "
                                        "instruction"),
               ("from", "Trace-From", "stop when the trace memory is full (default Trace-To: "
                                      "keep the newest)"),
               ("tb", "All branches", "TB: the PC of every branch, also the ones the program "
                                      "predicts"),
               ("io", "No overflow", "IO: stall the core instead of losing trace when the "
                                     "FIFO is full"))
_RV_TRACE = (("return", "Implicit return", "a return to the predicted address sends nothing"),
             ("predict", "Branch prediction", "E-Trace: correctly predicted branches as a "
                                              "count"),
             ("cache", "Jump cache", "E-Trace: uninferable targets as a jump target cache "
                                     "index"),
             ("htm", "HTM", "N-Trace: branch history (off: BTM, a message a taken branch)"),
             ("repeat", "Repeated history", "N-Trace: repeated HIST records counted"),
             ("sijump", "Sequential jumps", "LUI / AUIPC + JALR: inferable"),
             ("time", "Timestamps", "trTsEnable: the hart's cycles"))


def _trace(s, opts, memo) -> dict:
    from . import pdtrace as PDT
    m = s.machine
    if m.rv and s.rvtrace is None:
        return {"blocks": [note("The P8700 traces through its Hybrid Debug (the debugger's "
                                "dbu commands)")], "controls": []}
    if m.rv:
        e = s.rvtrace.encoders[s.cpu]
        on = e.enabled
        f = e.features
        state = {"return": bool(f & 8), "predict": bool(f & 0x10), "cache": bool(f & 0x20),
                 "repeat": bool(f & 0x100), "sijump": bool(f & 4),
                 "htm": (e.control >> 4 & 7) != 3, "time": e.ts.enabled} if on else \
            {k: k == "htm" for k, _, _ in _RV_TRACE}
        controls = [{"kind": "trace", "on": on, "rv": True,
                     "protocol": ("e", "n")[e.format] if on else "e",
                     "options": [[k, label, tip, state[k]] for k, label, tip in _RV_TRACE],
                     "packets": bool(opts.get("packets"))}]
        if opts.get("packets"):
            from .sessionrv import trace_packet_lines
            text = trace_packet_lines(s, 200)
        else:
            text = s.trace_lines(200)
        state_lines = s.trace_state()[:3]
    else:
        tcb = m.ejtag.tap.tcb
        st = m.pdt.settings()
        on = bool(st["on"] and tcb.enabled)
        state = {"addr": bool(st["mode"] & PDT.MODE_LA), "data": bool(st["mode"] & PDT.MODE_LD),
                 "ca": tcb.ca, "from": bool(tcb.control_b >> PDT.TB_TM_SHIFT & 1),
                 "tb": st["tb"], "io": st["io"]} if on else {k: False for k, _, _ in _MIPS_TRACE}
        controls = [{"kind": "trace", "on": on, "rv": False,
                     "options": [[k, label, tip, state[k]] for k, label, tip in _MIPS_TRACE]}]
        text = s.trace_lines(200)
        state_lines = s.trace_state()[:2]
    return {"blocks": [lines("", state_lines),
                       lines("", [ln.strip() for ln in text] if text else
                             ["Nothing traced yet: tick Trace, then run."])],
            "controls": controls}


def _trace_apply(s, args: dict) -> None:
    from . import pdtrace as PDT
    m = s.machine
    o = dict(args.get("options") or {})
    if m.rv:
        if s.rvtrace is None:
            return
        if args.get("on"):
            proto = args.get("protocol") or "e"
            opts = tuple(k for k in ("return", "sijump") if o.get(k))
            if proto == "e":
                opts += tuple(k for k in ("predict", "cache") if o.get(k))
            elif o.get("repeat"):
                opts += ("repeat",)
            s.rv_trace_start(protocol=proto, options=opts, htm=bool(o.get("htm")),
                             timestamps=bool(o.get("time")))
        elif any(e.enabled for e in s.rvtrace.encoders):
            s.trace_stop()
        return
    if args.get("on"):
        mode = PDT.MODE_PC | (PDT.MODE_LA | PDT.MODE_SA if o.get("addr") else 0) \
            | (PDT.MODE_LD | PDT.MODE_SD if o.get("data") else 0)
        s.trace_start(mode=mode, ca=bool(o.get("ca")), trace_from=bool(o.get("from")),
                      tb=bool(o.get("tb")), io=bool(o.get("io")))
    elif m.ejtag.tap.tcb.enabled or m.pdt.on:
        s.trace_stop()


# -- the board ---------------------------------------------------------------------------------

def _board(s, opts, memo) -> dict:
    board = s.board
    if board is None:
        return {"blocks": [note("Flat board: every physical address is RAM (the MARS "
                                "layout). Settings > Board and devices chooses the MIPS "
                                "Malta: its memory map, bus errors, LEDs and display.")],
                "controls": []}
    f = board.fpga
    blocks = [{"kind": "board", "leds": [bool(f.ledbar >> i & 1) for i in range(7, -1, -1)],
               "display": f.display, "switches": f"S2 = 0x{board.switches:02x}"},
              note("MIPS Malta, CoreLV card (GT-64120 system controller): SDRAM, the monitor "
                   "flash and the board registers; anything else is a bus error.")]
    text = s.board_lines()
    if board.disks or s.disk_notes:
        text += ["", "IDE disk commands:"] + s.disk_lines(states=False)
    blocks.append(lines("", text))
    return {"blocks": blocks, "controls": [
        {"kind": "button", "action": "board_nmi", "label": "ON/NMI",
         "tip": "The board's ON/NMI button: a Non Maskable Interrupt (NMISTATUS.ONNMI)"}]}


# -- the system (a CPS, or several harts) -----------------------------------------------------

def _system(s, opts, memo) -> dict:
    from . import cpsview as cv
    m = s.machine
    pick = {"kind": "core", "action": "core"}
    if s.cps is None and s.rv and len(s.machines) > 1:
        rows = s.cores()
        d = s.hex_digits("pc")
        cells = [[s.cpu_name(c["cpu"]), "exited" if c["halted"] else
                  "waiting (WFI)" if c["waiting"] else c["state"],
                  f"0x{c['pc']:0{d}x}" if c["pc"] is not None else "-",
                  str(c["line"]) if c["line"] else "-", s.machines[c["cpu"]].mode_name]
                 for c in rows]
        cm = getattr(getattr(s, "rvcps", None), "cm", None)
        return {"blocks": [note(f"{len(rows)} harts; every running hart steps together, hart 0 "
                                "first. Click one to show it in every pane."),
                           {**table("Cores", ["hart", "state", "pc", "line", "mode"], cells,
                                    ["valid" if c["selected"] else None for c in rows]), **pick},
                           lines("Coherence", cm.describe() if cm is not None else
                                 ["No Coherence Manager: the harts share one memory."])],
                "controls": []}
    if s.cps is None:
        return {"blocks": [note("One core: no Coherent Processing System. Choose 2 to 6 cores "
                                "in Settings > Processor.")], "controls": []}
    cps = s.cps
    rows = cv.core_rows(s)
    d = s.hex_digits("pc")
    blocks = [note(f"{len(rows)} cores; every running core steps together, core 0 first. "
                   "Click one to show it in every pane."),
              {**table("Cores", ["core", "state", "pc", "line", "mode", "interrupt pins"],
                       [[str(r.cpu), r.state, f"0x{r.pc:0{d}x}" if r.pc is not None else "-",
                         str(r.line) if r.line else "-", r.mode, r.pins] for r in rows],
                       ["valid" if r.selected else None for r in rows]), **pick}]
    dom = cv.domain_rows(cps)
    blocks.append(table("Coherence", ["core", "GCR_COHERENCE", "mode",
                                      "hears interventions from"],
                        [[str(x.cpu), f"0x{x.coherence:02x}",
                          "coherent" if x.coherent else "not coherent",
                          ", ".join(map(str, x.hears)) or "-"] for x in dom]))
    blocks.append(note(", ".join(f"{k} {v}" for k, v in cps.cm.counts.items())
                       + " — " + cps.cm.describe_error()))
    last = cv.last_line(s) if m.caches is not None and m.caches.d is not None else None
    if last is None:
        blocks.append(note(f"No data-cache access in the last step (of core {s.cpu})"))
    else:
        pa, how = last
        copies = cv.copies_rows(s, pa)
        fill = {"Modified": "miss", "Exclusive": "hit", "Shared": "valid"}
        blocks.append(table(f"Line 0x{pa:08x} ({how}), in every core's data cache",
                            ["core", "MESI state"],
                            [[str(j), st] for j, st in copies] or [["-", "no copy"]],
                            [fill.get(st) for _, st in copies] or None))
    gic = cps.gic
    if not gic.enabled:
        blocks.append(note("GIC disabled (GCR_GIC_BASE.GIC_EN = 0)"))
    else:
        src = cv.gic_rows(gic)
        blocks.append(table(f"GIC at 0x{gic.base:08x}: {gic.n} sources, counter {gic.counter}",
                            ["source", "trigger", "state", "goes to", "cores"],
                            [[str(r.src), r.kind, "PENDING" if r.pending else "idle", r.target,
                              r.vps] for r in src],
                            ["hit" if r.pending else None for r in src]))
        loc = cv.local_rows(gic)
        blocks.append(table("local interrupts", ["core", "local interrupt", "state", "enabled",
                                                 "goes to"],
                            [[str(r.cpu), r.name, "PENDING" if r.pending else "idle",
                              "yes" if r.enabled else "no", r.target] for r in loc],
                            ["hit" if r.pending and r.enabled else None for r in loc]))
        blocks.append(lines("", cv.gic_core_lines(gic)))
    regs = cv.register_rows(s)
    blocks.append(table("Registers", ["block", "register", "offset", "value", "meaning"],
                        [[r.block, r.name, f"+0x{r.offset:04x}", f"0x{r.value:08x}", r.meaning]
                         for r in regs]))
    return {"blocks": blocks, "controls": []}


# -- MT ----------------------------------------------------------------------------------------

def _mt(s, opts, memo) -> dict:
    from . import mtview as mv
    m = s.machine
    if s.mt is None:
        return {"blocks": [note("The MT Module is a MIPS ASE (MD00378): a RISC-V hart has none."
                                if m.rv else "No MT Module. Choose one in Settings > Processor "
                                "(MT Module), or use --mt VPES:TCS.")], "controls": []}
    vpes = mv.vpe_rows(s)
    blocks = [lines("", mv.header(s)),
              {**table("VPEs", ["VP", "core", "VPE", "VPA", "MVP", "TE", "EXCPT", "YQMask",
                                "runs only"],
                       [[str(r.vp), "-" if r.core is None else str(r.core), str(r.vpe),
                         str(int(r.vpa)), str(int(r.mvp)), str(int(r.te)), str(r.excpt),
                         f"0x{r.yqmask:04x}", f"TC{r.xtc} ({r.exclusive})" if r.exclusive
                         else "-"] for r in vpes],
                       ["valid" if r.vp == s.cpu else None for r in vpes]),
               "kind": "table", "action": "core", "pick": [r.vp for r in vpes]}]
    tcs = mv.tc_rows(s)
    d = s.hex_digits("pc")
    fill = {"running": "hit", "halted": None, "free": None}
    blocks.append({**table("Thread contexts (> issues next)",
                           ["TC", "core", "VP", "state", "pc", "line", "flags", "group",
                            "issued", "waits for"],
                           [[(">" if r.next else " ") + f"TC{r.tc}",
                             "-" if r.core is None else str(r.core), str(r.vp), r.state,
                             f"0x{r.pc:0{d}x}", str(r.line) if r.line else "-", r.flags or "-",
                             "-" if r.group is None else str(r.group), str(r.issued),
                             r.waits or "-"] for r in tcs],
                           [fill.get(r.state, "miss") for r in tcs]),
                   "action": "core", "pick": [r.vp for r in tcs]})
    tc = opts.get("tc")
    got = mv.tc_of(s, int(tc)) if tc not in (None, "") else None
    if got is not None and s.cpu == got[2]:
        blocks.append(lines(f"TC{got[1].n}", s.tc_lines(got[1].n)))
    blocks.append(lines("ITC", [mv.itc_header(s)]))
    cells = mv.cell_rows(s)
    if cells:
        blocks.append(table("", ["cell", "kind", "address", "E/F/T", "data (oldest first)"],
                            [[str(r.cell), r.kind,
                              f"0x{r.address:08x}" if r.address is not None else "-",
                              r.flags, r.data] for r in cells]))
    procs = mv.processors(s)
    p = procs[0][2] if procs else None
    controls = [{"kind": "text", "key": "yq", "label": "YIELD qualifier inputs",
                 "value": f"0x{p.yq_ext:04x}" if p else "0x0000", "action": "yq"},
                {"kind": "text", "key": "sys", "label": "PM_sys_avail (0-7)",
                 "value": str(p.sys_avail) if p else "7", "action": "sys"}]
    return {"blocks": blocks, "controls": controls}


# -- the Tools ---------------------------------------------------------------------------------

def _tools(s, opts, memo) -> dict:
    """The instruction reference (a search), the encoder / decoder and the
    number converter, in the session's release and architecture."""
    from . import tools
    release = s.release if s.machine is not None else "R5"
    arch = s.options.arch
    blocks = []
    which = opts.get("tool") or "reference"
    controls = [{"kind": "select", "key": "tool", "label": "tool",
                 "items": [["reference", "Instruction reference"],
                           ["encoder", "Encoder / decoder"], ["converter", "Number converter"]],
                 "value": which}]
    text = str(opts.get("q", ""))
    if which == "reference":
        controls.append({"kind": "text", "key": "q", "label": "search",
                         "value": text, "tip": "a mnemonic, a word of the description"})
        key = (release, arch, s.options.rv_extensions)
        if memo.get("ref_key") != key:
            memo["ref"] = tools.reference(release, arch, s.options.rv_extensions) \
                if arch in ("rv32", "rv64") else tools.reference(release, arch)
            memo["ref_key"] = key
        rows = tools.search(memo["ref"], text)[:200]
        blocks.append(table(f"{len(rows)} instructions" + (" (the first 200)"
                                                          if len(rows) == 200 else ""),
                            ["mnemonic", "syntax", "description", "format", "since"],
                            [[r.mnemonic, r.syntax, r.description, r.format,
                              r.since + (f", removed in {r.removed_in}" if r.removed_in else "")]
                             for r in rows]))
    elif which == "encoder":
        controls.append({"kind": "text", "key": "q", "label": "instruction or word",
                         "value": text, "tip": "an instruction (assembled) or a number "
                                               "(decoded): addiu $t0, $t0, 4 / 0x25080004"})
        if text.strip():
            try:
                mode, words = tools.encode_or_decode(
                    text, release=release, abi=True, arch=arch) if arch == "mips32" else \
                    tools.encode_or_decode(text, release=release, arch=arch)
                for d in words:
                    blocks.append({"kind": "fields",
                                   "title": f"{d.encoding(arch)}: {d.text} — {d.name}, "
                                            f"{d.description} ({d.format}, {d.execution})",
                                   "cells": [[f.name, f"[{f.msb}:{f.lsb}]", f.bits, f.meaning]
                                             for f in d.fields]})
                    blocks.append(lines("", [tools.group_bits(d.word, 8 * d.size)]))
            except Exception as exc:            # noqa: BLE001 (shown to the user)
                blocks.append(note(f"{type(exc).__name__}: {exc}"))
    else:
        controls.append({"kind": "text", "key": "q", "label": "value",
                         "value": text, "tip": "decimal, 0x..., 0b..., 'c', 1.5, -2e-3, inf, nan"})
        if text.strip():
            try:
                n = tools.convert(text, nan2008=release == "R6")
                rows = [["hex", n.hex], ["unsigned", str(n.unsigned)],
                        ["signed (two's complement)", str(n.signed)], ["binary", n.binary],
                        ["octal", n.octal], ["bytes as text", n.chars]]
                blocks.append(table(f"{n.source}, {n.width} bits", ["", ""], rows))
                for fp, title in ((n.single, "as an IEEE-754 single (float)"),
                                  (n.double, "as an IEEE-754 double"),
                                  (n.half, "as an IEEE-754 half (binary16)")):
                    if fp is not None:
                        blocks.append(lines(title, [
                            f"sign {fp.sign}, exponent {fp.exponent}, fraction 0x{fp.fraction:x}"
                            f" — {fp.kind}", fp.formula, f"= {tools.fmt_float(fp.value)}"]))
                if n.notes:
                    blocks.append(lines("", n.notes))
            except Exception as exc:            # noqa: BLE001
                blocks.append(note(f"{type(exc).__name__}: {exc}"))
    return {"blocks": blocks, "controls": controls}
