"""The Memory pane's and the pipeline chart's data, Qt-free: the window's
widgets (``gui/widgets.py``) and the web page (``web/bridge.py``) show the
same rows (PLAN M29: no view is computed in a Qt widget only).

Nothing here changes the machine: the memory words are the Session's
side-effect-free ``memory_words``, the chart reads the pipeline's history.
"""

from __future__ import annotations

#: the Memory pane's shortcuts (the window's combo box, the page's select)
MEMORY_SEGMENTS = [".data (0x10010000)", ".text (0x00400000)", "stack ($sp)",
                   "MMIO (0xffff0000)"]


def memory_start(s, segment: int, text: str = "", physical: bool = False) -> int:
    """Where the dump starts: a typed address (hex or decimal), else the
    segment shortcut's address — physical: the shortcut translated."""
    text = text.strip()
    if text:
        try:
            return int(text, 0)
        except ValueError:
            pass
    m = s.machine
    va = [0x10010000, 0x00400000, (m.read_reg(29) - 64) if m else 0, 0xFFFF0000][segment]
    if physical and m is not None:
        pa = m.translate(va).pa
        return va if pa is None else pa
    return va


def memory_rows_count(segment: int) -> int:
    return 1 if segment == 3 else 32               # (MMIO: one row of device registers)


def memory_layout(s, show_bytes: bool, doubles: bool) -> str:
    """"bytes" (16 a row, address order), "doubles" (MIPS64: 2 doublewords
    a row, as LD loads them) or "words" (4 a row, as LW loads them)."""
    return "bytes" if show_bytes else "doubles" if s.wide and doubles else "words"


def memory_heads(layout: str) -> list[str]:
    return ([f"+{k:x}" for k in range(16)] + ["ascii"] if layout == "bytes"
            else ["+0", "+8"] if layout == "doubles" else ["+0", "+4", "+8", "+c"])


def memory_rows(s, start: int, count: int, physical: bool, layout: str
                ) -> list[tuple[str, list[str]]]:
    """``(address text, cells)`` rows: ``--------`` where nothing maps the
    word (the TLB), ``--`` its bytes."""
    if s.machine is None:
        return []
    big = s.machine.big_endian
    out = []
    for addr, words in s.memory_words(start, count, physical=physical):
        if layout == "bytes":                    # (the bytes in address order)
            data = [None if w is None else b for w in words
                    for b in (w.to_bytes(4, "big" if big else "little")
                              if w is not None else (None,) * 4)]
            cells = ["--" if b is None else f"{b:02x}" for b in data]
            cells.append("".join(" " if b is None else chr(b) if 32 <= b < 127 else "."
                                 for b in data))
        elif layout == "doubles":                # as LD loads them (the byte order)
            cells = []
            for hi, lo in ((words[0], words[1]), (words[2], words[3])):
                if not big:
                    hi, lo = lo, hi
                cells.append("-" * 16 if hi is None or lo is None else f"{hi:08x}{lo:08x}")
        else:
            cells = ["--------" if w is None else f"{w:08x}" for w in words]
        out.append((f"0x{addr:016x}" if addr >> 32 else f"0x{addr:08x}", cells))
    return out


def pipeline_chart(s, window: int = 24, rows: int = 20
                   ) -> tuple[list[int], list[tuple[int, str, list[str]]]]:
    """The last ``window`` cycles: ``(cycles, [(pc, text, a cell a cycle)])``
    for the last ``rows`` instructions — a cell is the stage, ``--`` a stall
    in ID, ``xx`` a killed instruction (the 24K), "" none."""
    p = s.pipeline
    if p is None:
        return [], []
    hist = p.history[-window:]
    table: dict[int, tuple[int, str, dict[int, str]]] = {}
    cycles = [t.cycle for t in hist]
    slots = getattr(p, "slots", None)
    if slots is not None:                        # the 24K: every instruction's stages
        first = cycles[0] if cycles else 0
        for sl in reversed(slots):
            cells = {c: st for c, st in sl.timeline.items() if c >= first}
            if cells:
                table[sl.seq] = (sl.pc, sl.text, cells)
            elif sl.fetched < first - 64:
                break
    else:
        for t in hist:
            for st, v in t.stages.items():
                if v is not None:
                    cell = "--" if st == "ID" and t.stall else st
                    table.setdefault(v.seq, (v.pc, v.text, {}))[2][t.cycle] = cell
    seqs = sorted(table)[-rows:]
    return cycles, [(table[q][0], table[q][1], [table[q][2].get(c, "") for c in cycles])
                    for q in seqs]


def instruction(v, tr=lambda t: t) -> dict | None:
    """The instruction pane (what happened inside the last executed
    instruction, an InstrView): ``title``, ``format``, ``fields`` (name,
    msb, lsb, bits, meaning — MSB first), ``sections`` ((title, lines):
    ALU, Address, Branch, Effects), ``lanes`` (MSA / RISC-V V elements) and
    ``exception``.  Plain text: the window renders it as HTML, the page too."""
    if v is None:
        return None
    from .mmu import cca_name
    from .trace import digits_of
    w = getattr(v, "digits", 8)                     # a register's (MIPS64: 16)
    slot = " — delay slot" if v.in_delay_slot else ""
    own = v.format in ("MIPS16e", "microMIPS")      # no MIPS32 twin
    out: dict = {"title": f"0x{v.pc:0{w}x}: {v.source_text or v.text}", "format": []}
    if v.compressed:
        out["format"].append(f"{v.compressed}{slot if own else ''}")
    if not own:
        out["format"].append(f"{v.format}-format word 0x{v.word:08x}"
                             + (f" ({v.text})" if v.compressed else "") + slot)
    out["fields"] = [[f.name, f.msb, f.lsb, f.bits, f.meaning] for f in v.fields]
    sections = []
    if v.alu:
        a = v.alu
        lines = [a.op]
        if a.a:
            lines.append(f"a = 0x{a.a.value:0{w}x} from {a.a.source}")
        if a.b:
            lines.append(f"b = 0x{a.b.value:0{w}x} from {a.b.source}")
        if a.result is not None:
            lines.append(f"result = 0x{a.result:0{w}x}" + (" (zero)" if a.zero else "")
                         + (" OVERFLOW" if a.overflow else ""))
        sections.append(["ALU", lines])
    if v.address:
        ad = v.address
        lines = [f"base 0x{ad.base.value:0{w}x} + {ad.offset} = 0x{ad.address:0{w}x} — "
                 f"{ad.access} of {ad.size} byte(s), {ad.segment} ({ad.privilege}, "
                 f"{'mapped' if ad.mapped else 'unmapped'})"
                 + ("" if ad.aligned else " MISALIGNED")]
        if ad.physical is not None:
            lines.append(f"physical 0x{ad.physical:08x} — {ad.how}; attribute {ad.cca} = "
                         f"{cca_name(ad.cca)}")
        sections.append(["Address", lines])
    if v.branch:
        br = v.branch
        what = (f"taken → 0x{br.target:0{w}x}" if br.taken
                else "not taken" + (", delay slot nullified" if br.likely else ""))
        sections.append(["Branch", [f"{br.condition}: {what}"
                                    + (f"; link $ra = 0x{br.link:0{w}x}"
                                       if br.link is not None else "")]])
    eff = [f"memory read 0x{a:08x} = 0x{val:0{2 * n}x}" for a, n, val in v.mem_reads]
    eff += [f"memory write 0x{a:08x}: 0x{o:0{2 * n}x} → 0x{nw:0{2 * n}x}"
            for a, n, o, nw in v.mem_writes]
    eff += [f"{name}: 0x{o:0{d}x} → 0x{nw:0{d}x}"
            for name, o, nw in v.reg_writes
            for d in (v.vdigits if getattr(v, "rv", False) and name[0] == "v"
                      and name[1:].isdigit() else digits_of(name, w),)]
    if eff:
        sections.append(["Effects", eff])
    out["sections"] = sections
    lanes = getattr(v, "lanes", None)
    if lanes is not None and hasattr(lanes, "sew"):     # RISC-V V (tracerv.VectorLanes)
        out["lanes"] = {"title": f"{tr('Elements')} (e{lanes.sew}, {tr('element 0 first')})",
                        "lines": list(lanes.lines())}
    elif lanes is not None:                             # MSA (trace.LaneView)
        n = max(len(es) for _, _, _, es in lanes.rows)
        out["lanes"] = {"title": f"{tr('Elements')} ({lanes.size}-bit, "
                                 f"{tr('element 0 first')})",
                        "head": [f"[{i}]" for i in range(n)],
                        "rows": [[f"{role} $w{reg}", [f"{x:0{size // 4}x}" for x in es]]
                                 for role, reg, size, es in lanes.rows]}
    out["exception"] = v.exception or None
    return out
