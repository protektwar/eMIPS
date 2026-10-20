"""Views of the MIPS 24K pipeline model's own state (PLAN M10 phase 5),
shared by the CLI debugger (``core``) and the GUI's 24K datapath.

Qt-free and side-effect free: everything is read from the
:class:`~emips.pipe24k.Pipeline24K` — its fetch unit (the instruction
buffer, the branch history table, the return prediction stack), its units
(multiply / divide, FPU, load / store) and the CP0 writes still on their
way (the hazards).  Nothing here predicts, fetches or executes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pipe24k import (BHT_ENTRIES, FSB_ENTRIES, IB_ENTRIES, LDQ_ENTRIES, MUL_RATE,
                      RPS_ENTRIES, Pipeline24K, hazard_spacing)

M32 = 0xFFFFFFFF
#: the two-bit counter's states
BHT_STATES = ("strongly not taken", "weakly not taken", "weakly taken", "strongly taken")


def is_24k(p) -> bool:
    return isinstance(p, Pipeline24K)


# ------------------------------------------------------------------ the fetch unit

@dataclass
class BufferRow:
    seq: int
    pc: int
    text: str
    where: str                  # "IF" (still fetching) / "IS" / "IT" (in the buffer)
    ready_in: int               # cycles until it can enter RF (0: now)
    in_slot: bool
    pred: str                   # a branch / jump: how it was predicted


def _queues(p: Pipeline24K) -> list:
    """The instruction buffers: one, or the 34K's one per TC (its
    instructions carry their TC in their text)."""
    fronts = getattr(p, "fronts", None)
    if fronts is None:
        return list(p.queue)
    return sorted((s for tc in p.p.tcs for s in p._front(tc).queue), key=lambda s: s.seq)


def buffer_rows(p: Pipeline24K) -> list[BufferRow]:
    """The fetched instructions not in RF yet, oldest first (the 34K:
    every TC's)."""
    t = p.cycles
    out = []
    for s in _queues(p):
        where = s.timeline.get(t, "IT")
        out.append(BufferRow(s.seq, s.pc, s.text, where, max(0, s.avail - t - 1),
                             s.in_slot, s.pred))
    return out


def bht_rows(p: Pipeline24K) -> list[tuple[int, int, str]]:
    """``(index, counter, state)`` of the BHT entries used so far."""
    return [(i, p.bht[i], BHT_STATES[p.bht[i]]) for i in sorted(p.bht_used)]


def rps_rows(p: Pipeline24K) -> list[tuple[int, int, bool]]:
    """``(slot, return address, is the top)`` of the return stack."""
    return [(i, p.rps[i], i == p.rps_top) for i in range(RPS_ENTRIES)]


def fetch_state(p: Pipeline24K) -> str:
    """What the fetch unit is doing now (the 34K: each TC's)."""
    fronts = getattr(p, "fronts", None)
    if fronts is not None:
        parts = []
        for tc in p.p.tcs:
            f = p._front(tc)
            parts.append(f"TC{tc.n}: " + _fetch_text(f, p.cycles + 1, f.fetch_isa,
                                                      len(f.queue)))
        return "; ".join(parts)
    return _fetch_text(p, p.cycles + 1, p.fetch_isa, len(p.queue))


def _fetch_text(p, t: int, isa: int, queued: int) -> str:
    if not p.fetching:
        return "stopped (the program is ending)"
    if p.blocked:
        return "waiting for a jump to execute (no prediction)"
    if p.after_slot is not None:
        what, target, at = p.after_slot
        return ("fetching the delay slot, then waiting for the jump" if what == "wait" else
                f"fetching the delay slot, then 0x{target:08x} (from cycle {at})")
    m16 = " (MIPS16e: a 64-bit bundle every other cycle, recoded in IR / IK)" \
        if isa else ""
    if p.fetch_at > t:
        return (f"waiting until cycle {p.fetch_at} (a miss, a redirect"
                f"{' or the other cycle' if m16 else ''}), then 0x{p.fetch_pc:08x}{m16}")
    if queued > IB_ENTRIES + 2:
        return "the instruction buffer is full"
    return f"fetching 0x{p.fetch_pc:08x}{m16}"


# ------------------------------------------------------------------ the units

@dataclass
class UnitRow:
    unit: str
    busy: bool
    state: str


def unit_rows(p: Pipeline24K) -> list[UnitRow]:
    """The multiply / divide unit, the FPU, the load / store unit, the TLB
    interlock: busy or free, and why."""
    t = p.cycles + 1
    out = []
    n, window = MUL_RATE
    muls = sum(1 for c in p.mul_starts if c > t - window)
    div = p.div_free - t
    mdu = []
    if div > 0:
        mdu.append(f"divider busy for {div} more cycle{'s' if div != 1 else ''}")
    mdu.append(f"{muls} of {n} MULs in the last {window} cycles")
    out.append(UnitRow("MDU", div > 0 or muls >= n, "; ".join(mdu)))
    fpu = p.fpu_free - t
    out.append(UnitRow("FPU", fpu > 0, f"busy for {fpu} more cycle{'s' if fpu != 1 else ''}"
                       if fpu > 0 else "free"))
    ldq = sum(1 for c in p.ldq if c >= t)
    tm = p.timing
    fsb = sum(1 for c in tm.pending.values() if c >= t) if tm is not None else 0
    lsu = []
    if p.lsu_free > t:
        lsu.append(f"waiting for the DTLB ({p.lsu_free - t} more)")
    lsu.append(f"load data queue {ldq} / {LDQ_ENTRIES}")
    lsu.append(f"fill-store buffer {fsb} / {FSB_ENTRIES}")
    out.append(UnitRow("LSU", p.lsu_free > t or ldq >= LDQ_ENTRIES, "; ".join(lsu)))
    if p.tlb_until > t:
        out.append(UnitRow("TLB", True, f"a TLB write / read: loads, stores, CP0 wait "
                                        f"{p.tlb_until - t} more"))
    return out


@dataclass
class HazardRow:
    register: str
    old: int | None
    new: int | None
    producer: str
    visible: dict                # consumer class -> cycle from which it sees new


def hazard_rows(p: Pipeline24K) -> list[HazardRow]:
    """The CP0 writes some instruction may not see yet."""
    t = p.cycles + 1
    out = []
    for h in p.hazards:
        vis = {c: h.ex + hazard_spacing(h, h.key, c) + 1 for c in ("other", "lsu", "cop", "int")}
        if max(vis.values()) <= t:
            continue
        name = "TLB entries" if h.key == "tlb" else (
            p.m.cp0[h.key].get("name") or f"CP0 {h.key[0]}.{h.key[1]}")
        out.append(HazardRow(name, None if h.key == "tlb" else h.old,
                             None if h.key == "tlb" else h.new, h.text, vis))
    return out


# ------------------------------------------------------------------ text (the CLI)

def format_buffer(p: Pipeline24K) -> str:
    rows = buffer_rows(p)
    out = [f"fetch unit: {fetch_state(p)}",
           f"instruction buffer ({len(rows)} fetched, {IB_ENTRIES} entries + the IF / IS "
           "bundles):"]
    if not rows:
        out.append("  (empty)")
    for r in rows:
        note = []
        if r.in_slot:
            note.append("delay slot")
        if r.pred:
            note.append(f"predicted: {r.pred}")
        if r.ready_in:
            note.append(f"ready in {r.ready_in}")
        out.append(f"  {r.where}  0x{r.pc:08x}  {r.text:<28}" + ("  (" + ", ".join(note) + ")"
                                                              if note else ""))
    return "\n".join(out)


def format_predictors(p: Pipeline24K, *, limit: int = 32) -> str:
    rows = bht_rows(p)
    out = [f"BHT ({BHT_ENTRIES} two-bit counters, index = VA[11:3]): {len(rows)} used; "
           f"{p.predicted} branches / jumps resolved, {p.mispredicts} mispredicted"]
    for i, n, state in rows[:limit]:
        out.append(f"  {i:3d}  {n}  {state}")
    if len(rows) > limit:
        out.append(f"  ... {len(rows) - limit} more")
    out.append(f"return prediction stack ({RPS_ENTRIES} entries, > = top):")
    for i, addr, top in rps_rows(p):
        out.append(f"  {'>' if top else ' '} {i}  0x{addr:08x}")
    if p.last_prediction is not None:
        pc, text, how, nxt = p.last_prediction
        nxt_text = f"0x{nxt:08x}" if nxt is not None else "wait"
        out.append(f"last prediction: 0x{pc:08x} {text}: {how} -> {nxt_text}")
    if p.last_resolution is not None:
        pc, text, taken, right = p.last_resolution
        out.append(f"last resolved:   0x{pc:08x} {text}: {'taken' if taken else 'not taken'}, "
                   + ("predicted right" if right else "MISPREDICTED"))
    return "\n".join(out)


def format_units(p: Pipeline24K) -> str:
    out = [f"{r.unit:4s} {'BUSY' if r.busy else 'free'}  {r.state}" for r in unit_rows(p)]
    hz = hazard_rows(p)
    if hz:
        out.append("CP0 writes on their way (an instruction executing before the cycle "
                   "sees the old value):")
        for h in hz:
            vals = "" if h.old is None else f" 0x{h.old:08x} -> 0x{h.new:08x}"
            out.append(f"  {h.register}{vals} by {h.producer}: from cycle "
                       f"{h.visible['other']} ({h.visible['lsu']} for loads / stores, "
                       f"{h.visible['cop']} for the FPU)")
    else:
        out.append("no CP0 hazard pending")
    return "\n".join(out)


def format_core(p: Pipeline24K) -> str:
    return "\n".join([format_buffer(p), "", format_predictors(p, limit=8), "",
                      format_units(p)])
