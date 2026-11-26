"""The pipeline views of the 24K and the P8700, Qt-free (M29 W2): the 24K's
nine stages drawn (the layout and :func:`draw24k`, what one refresh shows,
with emips.datapathdraw's colour tokens) and the P8700's window table
(:func:`p8700_view`).  The window (``gui/datapath24k.py``,
``gui/core8700.py``) and the web page show the same thing.

Each cycle the 24K's stages holding an instruction are filled in their
colour (the colours of the textbook stage doing the same work), the stage
where an interlock holds the pipeline is red, a redirect of the fetch (a
misprediction, an exception, ERET) lights the redirect wire.  Everything
comes from :meth:`Session.stage_details`, the pipeline's last
:class:`~emips.pipeline.CycleTrace` and :mod:`emips.coreview` — nothing is
re-executed here."""

from __future__ import annotations

from . import coreview as cv
from .datapathinfo import PARTS_24K

W, H = 1500, 790
STAGES = ("IF", "IS", "IT", "RF", "AG", "EX", "MS", "ER", "WB")
COL = 166                                    # column pitch
X0 = 4
Y_PART, H_PART = 112, 96
Y_WIRE = Y_PART + H_PART // 2
IB_ROWS = 10

PARTS = {
    "IF": ("I-cache\nITLB, BHT",
           "IF: the I-cache and its tags are read, the ITLB translates the PC, the "
           "branch history table is read — for a 64-bit bundle (two instructions)"),
    "IS": ("hit / way\npredict",
           "IS: I-cache hit and way select; conditional branches are predicted from "
           "the BHT, jr $31 from the return stack"),
    "IT": ("instruction\nbuffer",
           "IT: the 8-entry instruction buffer (bypassed while empty); the branch "
           "target is computed here"),
    "RF": ("registers\ndecode",
           "RF: register file read, decode; a load / store waits here for its base "
           "register (the skewed ALU: an ALU result reaches AG a cycle late)"),
    "AG": ("address\nadder",
           "AG: the data address is added; an instruction waits here for its ALU "
           "operands (load data, MFC0, MUL / MFHI results) or a busy unit"),
    "EX": ("ALU, DTLB\nbranch",
           "EX: the ALU, the DTLB, the D-cache access starts; branches are resolved "
           "(a misprediction redirects the fetch)"),
    "MS": ("D-cache\nhit, align",
           "MS: D-cache hit and way select, load align; a waiting access holds the "
           "pipeline here"),
    "ER": ("exceptions\ncomplete",
           "ER: exceptions and interrupts are taken here; the instruction completes"),
    "WB": ("register\nwrite", "WB: the register file is written"),
}

#: where a stall kind holds the pipeline
STALL_STAGE = {"memory": "MS", "fetch": "IT", "alu-agen": "RF"}


def x_of(st: str) -> int:
    return X0 + STAGES.index(st) * COL


def layout24k() -> dict:
    """The 24K's static drawing: stage bands, the stage parts, the wires,
    the value places, the text blocks and the clickable regions (key ->
    the guide's number, title, badge position, region box)."""
    ex, iff, ag = x_of("EX") + COL // 2, x_of("IF") + COL // 2, x_of("AG") + COL // 2
    wires = {f"{a}-{b}": [[[x_of(a) + COL - 20, Y_WIRE], [x_of(b) + 12, Y_WIRE]], ""]
             for a, b in zip(STAGES, STAGES[1:])}
    wires["redirect"] = [[[ex, Y_PART], [ex, 80], [iff, 80], [iff, Y_PART]],
                         "redirect: a misprediction (from EX, fetched in the cycle after "
                         "MS), an exception (ER), ERET"]
    wires["bypass_ex"] = [[[x_of("EX") + 40, Y_PART + H_PART], [x_of("EX") + 40, 236],
                           [ag, 236], [ag, Y_PART + H_PART]],
                          "bypass EX -> AG: an ALU result for the next instruction's ALU "
                          "(no wait), its address a cycle later"]
    wires["bypass_ms"] = [[[x_of("MS") + 40, Y_PART + H_PART], [x_of("MS") + 40, 252],
                           [ag - 30, 252], [ag - 30, Y_PART + H_PART]],
                          "bypass MS -> AG: load data (one cycle to an ALU user, two to an "
                          "address)"]
    px, ux = x_of("RF"), x_of("EX")
    regions = {f"24k:{st}": [x_of(st) + 12, Y_PART, x_of(st), 0, COL - 8, 290]
               for st in STAGES}
    regions["24k:buffer"] = [X0 + 6, 300, X0 + 6, 300, 3 * COL - 30, 22 + IB_ROWS * 22]
    regions["24k:predict"] = [px + 6, 300, px + 6, 300, 2 * COL - 12, 160]
    regions["24k:units"] = [ux + 6, 300, ux + 6, 300, 4 * COL - 12, 160]
    regions["24k:hazards"] = [ux + 6, 470, ux + 6, 470, 4 * COL - 12, 110]
    return {
        "W": W, "H": H, "col": COL,
        "stages": {st: x_of(st) for st in STAGES},
        "parts": {st: [x_of(st) + 12, Y_PART, COL - 32, H_PART, label, tip]
                  for st, (label, tip) in PARTS.items()},
        "values": {**{st: [x_of(st) + 8, Y_PART + H_PART + 10, 8] for st in STAGES},
                   "redirect": [iff + 40, 62, 8]},
        "wires": wires,
        "ib": [[X0 + 6, 322 + i * 22, 3 * COL - 30, 20] for i in range(IB_ROWS)],
        # text blocks: key -> [x, y, size, bold, colour token]
        "texts": {"ib_head": [X0 + 6, 300, 9, True, None, "instruction buffer (IS / IT)"],
                  "fetch_note": [X0 + 6, 270, 8, False, "muted",
                                 "fetch unit (IFU): two instructions a cycle"],
                  "pipe_note": [x_of("RF") + 6, 270, 8, False, "muted",
                                "integer pipeline (ALU) and load / store unit (LSU)"],
                  "predict_head": [px + 6, 300, 9, True, None, "branch prediction"],
                  "units_head": [ux + 6, 300, 9, True, None, "units"],
                  "hazards_head": [ux + 6, 470, 9, True, None,
                                   "CP0 writes on their way (hazards)"],
                  "legend": [X0 + 6, H - 22, 8, False, "muted",
                             "stage colours: fetch (IF IS IT), decode (RF), execute (AG EX), "
                             "memory (MS), completion (ER WB); red: where the pipeline is "
                             "held, the redirect wire"]},
        # the refreshed texts: key -> [x, y, size, bold]
        "blocks": {"predict": [px + 6, 322, 8, False], "units": [ux + 6, 322, 8, False],
                   "hazards": [ux + 6, 492, 8, False], "status": [X0 + 6, 610, 10, True],
                   "events": [X0 + 6, 636, 8, False]},
        "regions": regions,
        "numbers": {k: [PARTS_24K[k].number, PARTS_24K[k].title] for k in regions},
    }


def draw24k(s) -> dict:
    """What the 24K's diagram shows for the session's current cycle."""
    d = {"bands": {st: "bubble" for st in STAGES}, "titles": {st: "" for st in STAGES},
         "parts": {st: [None, None] for st in STAGES},
         "values": {k: ["", None] for k in (*STAGES, "redirect")},
         "wires": {k: [None, None] for k in layout24k()["wires"]},
         "ib": [["", None] for _ in range(IB_ROWS)],
         "blocks": {"predict": ["", None], "units": ["", None], "hazards": ["", None],
                    "status": ["", None], "events": ["", "muted"]}}
    p = s.pipeline
    if p is None or not cv.is_24k(p):
        d["blocks"]["status"][0] = "pipeline mode with the 24K model shows it here"
        return d
    details = s.stage_details()
    ev = details.pop("_events", {})
    trace = p.history[-1] if p.history else None
    stall = ev.get("stall") or ""
    kind = stall.split(":", 1)[0] if stall else ""
    held = STALL_STAGE.get(kind, "AG") if stall else None
    for st in STAGES:
        dd = details.get(st) or {}
        if dd.get("pc") is None:
            d["titles"][st] = "(bubble)"
            continue
        d["bands"][st] = f"band:{st}"
        d["titles"][st] = _clip(dd.get("text", ""), 21)
        d["parts"][st] = [f"stage:{st}", None]
        d["values"][st] = [_value(st, dd), None]
    for a, b in zip(STAGES, STAGES[1:]):
        if (details.get(b) or {}).get("pc") is not None:
            d["wires"][f"{a}-{b}"] = [f"stage:{b}", None]
    if held is not None and held in d["parts"]:
        d["parts"][held] = ["red", "red_fill"]
        d["values"][held] = [_clip(stall.split(":", 1)[-1].strip(), 24), "red"]
    events = ev.get("events") or []
    redirect = next((e for e in events if "kill the younger" in e or
                     "fetch the handler" in e or "return to" in e), None)
    if redirect is not None:
        d["wires"]["redirect"] = ["red", None]
        d["values"]["redirect"] = [_clip(redirect, 60), "red"]
    if trace is not None:
        view = details.get("EX", {}).get("view")
        if view is not None and view.alu is not None and view.alu.a is not None:
            d["wires"]["bypass_ex"] = ["stage:EX", 1.6]
    rows = cv.buffer_rows(p)
    for i in range(min(IB_ROWS, len(rows))):
        r = rows[i]
        d["ib"][i] = [f"{r.where} {r.pc:08x} {_clip(r.text, 26)}"
                      + (" (slot)" if r.in_slot else ""), f"band:{r.where}"]
    d["blocks"]["predict"][0] = predict_text(p)
    d["blocks"]["units"][0] = "\n".join(f"{r.unit:4s} {'BUSY' if r.busy else 'free'}  "
                                        f"{_clip(r.state, 60)}" for r in cv.unit_rows(p))
    hz = cv.hazard_rows(p)
    d["blocks"]["hazards"][0] = "\n".join(
        f"{h.register} by {_clip(h.producer, 22)}: seen from cycle {h.visible['other']}"
        for h in hz[:5]) or "none"
    status = f"cycle {ev.get('cycle', p.cycles)}   {cv.fetch_state(p)}"
    if stall:
        status += f"   STALL — {stall}"
    d["blocks"]["status"] = [_clip(status, 150), "red" if stall else None]
    d["blocks"]["events"][0] = "\n".join(_clip(e, 150) for e in events[:6])
    return d


def _value(st: str, d: dict) -> str:
    v = d.get("view")
    if st == "IF":
        return f"PC {d['pc']:08x}"
    if v is None:
        return ""
    if st == "AG" and v.address is not None:
        return f"addr {v.address.address:08x}"
    if st == "EX":
        if v.exception:
            return f"EXCEPTION {v.exception}"
        if v.branch is not None:
            return "taken" if v.branch.taken else "not taken"
        if v.alu is not None and v.alu.result is not None:
            return f"= {v.alu.result:08x}"
    if st == "MS" and (v.mem_reads or v.mem_writes):
        if v.mem_reads:
            a, n, val = v.mem_reads[0]
            return f"read {val:0{2 * n}x}"
        a, n, _, new = v.mem_writes[0]
        return f"write {new:0{2 * n}x}"
    if st == "WB" and v.reg_writes:
        name, _, new = v.reg_writes[0]
        return f"{name} <- {new:08x}"
    return ""


def predict_text(p) -> str:
    lines = []
    if p.last_prediction is not None:
        pc, text, how, nxt = p.last_prediction
        lines.append(f"last: {_clip(text, 26)}")
        lines.append(f"  {how} -> " + (f"{nxt:08x}" if nxt is not None else "wait"))
    if p.last_resolution is not None:
        pc, text, taken, right = p.last_resolution
        lines.append(f"resolved: {_clip(text, 22)}")
        lines.append(f"  {'taken' if taken else 'not taken'}, "
                     + ("right" if right else "MISPREDICTED"))
    lines.append(f"{p.predicted} resolved, {p.mispredicts} mispredicted")
    rows = cv.bht_rows(p)
    lines.append(f"BHT: {len(rows)} of 512 used")
    for i, n, state in rows[-3:]:
        lines.append(f"  [{i:3d}] {state}")
    lines.append("return stack: " + " ".join(
        ("[" if top else "") + f"{a:08x}" + ("]" if top else "")
        for _, a, top in cv.rps_rows(p)))
    return "\n".join(lines)


# -- the P8700's window ----------------------------------------------------------------------

#: the stage colour a window entry gets by its state
STATE_STAGE = {"queued": "ID", "dispatched": "ID", "executing": "EX", "complete": "WB",
               "fetching": "IF", "buffered": "IF"}
P8700_COLUMNS = ("hart", "#", "pc", "instruction", "unit", "state")


def p8700_view(s, tr=lambda t: t) -> dict | None:
    """The P8700 core's view: ``title``, ``rows`` (the window, oldest first,
    then the instruction buffer: cells and a band token — ``error`` for a
    wrong-path instruction) and ``state`` (queues, units, front ends,
    predictors)."""
    p = getattr(s.pipeline, "core", None)
    if p is None:
        return None
    title = tr("P8700 core: cycle {c}, {n} graduated — the window (dispatched "
               "instructions, oldest first), then the instruction buffer").format(
                   c=p.cycles, n=p.retired)
    for h in [h for h in p.harts if not h.rob and not h.done]:   # (why it is empty)
        why = p.front_state(h) or tr("the front end fetches")
        title += "\n" + tr("hart {k}: the window is empty — {why}").format(k=h.k, why=why)
    rows = []
    for row in p.rows(buffer=True):
        state = tr(row["state"]) + (f" ({row['left']})" if row["left"] else "")
        flags = [tr("wrong path")] if row["ghost"] else []
        flags += [tr("mispredicted")] if row["mispredicted"] else []
        flags += [tr("bonded")] if row["bonded"] else []
        if flags:
            state += " — " + ", ".join(flags)
        rows.append([[str(row["hart"]), str(row["seq"]), f"{row['pc']:08x}", row["text"],
                      row["unit"], state],
                     "error" if row["ghost"] else f"band:{STATE_STAGE[row['state']]}"])
    state = "\n".join(p.lines("queues") + p.lines("predict")[1:]
                      + [ln for ln in p.lines("window")[1:] if ln.startswith("hart")])
    return {"title": title, "columns": [tr(c) for c in P8700_COLUMNS], "rows": rows,
            "state": state}


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"
