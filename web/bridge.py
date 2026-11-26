"""eMIPS in the browser: the page's Python side (it runs in Pyodide).

One Session, loaded as the command line loads a program (``python3 -m
emips debug FILE``: the program's ``# eMIPS:`` line, then the choices the
page adds as flags), and the state the page shows, as JSON.  The page
runs a program in slices (``run``) so that it stays responsive; the
debugger box sends its lines to a ``cli.Debugger`` on the same Session.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import shlex
import time

from emips.__main__ import build_parser
from emips.cli import Debugger, load_file, session_from_args

#: the program and its companion files (a lab's disk image ...) live here
WORK = "/work"

#: choices the others follow (the session adjusts them, as the window does:
#: nanoMIPS takes Release 6, a core preset its own release and options)
PRIMARY = {"arch", "core"}
#: the Settings dialog's sections: the window's tabs, as the debug command's
#: options (the toolbar keeps Release and Mode; files / commands are not settings)
SECTIONS = [
    ("Processor", ["arch", "endian", "core", "no_fpu", "compressed", "dsp", "msa", "msa_wrp",
                   "mt", "mt_pm", "itc", "cores", "vps", "itu", "shadow_sets", "crc", "ginv",
                   "nms"]),
    ("RISC-V", ["rv_extensions", "rv_modes", "rv_intc", "rv_rnmi", "l2_waits", "dspram"]),
    ("Pipeline", ["no_forwarding", "mem_timing", "nonblocking"]),
    ("Memory", ["mmu", "caches", "icache", "dcache", "cache_parity", "cache_init", "l2", "l3",
                "cache_ecc", "write_buffer"]),
    ("Board and devices", ["board", "iocu", "display"]),
    ("Exceptions and start", ["boot", "exceptions", "no_syscalls"]),
    ("Assembler", ["strict", "noreorder", "numeric", "no_header", "no_relax", "relax"]),
]


class Web:
    def __init__(self) -> None:
        self.s = None
        self.dbg: Debugger | None = None
        self.messages: list[str] = []
        self.shown = 0                          # (console text already sent)
        self.views: dict = {}                   # the panes the page shows (their data)
        self.pane_memo: dict = {}               # (each pane's values of the last step)
        self._devices_reset()

    # -- loading ------------------------------------------------------------------------

    def save(self, name: str, text: str) -> str:
        """A companion file (a disk image) next to the program."""
        os.makedirs(WORK, exist_ok=True)
        with open(os.path.join(WORK, os.path.basename(name)), "w", encoding="utf-8") as f:
            f.write(text)
        return "null"                           # (every method answers JSON)

    def open(self, name: str, text: str, flags: str = "") -> str:
        """Assemble ``text`` as ``name`` with the program's own ``# eMIPS:``
        settings and ``flags`` over them; the new state."""
        self.save(name, text)
        path = os.path.join(WORK, os.path.basename(name))
        self.messages = []
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                args = build_parser().parse_args(["debug", path, *shlex.split(flags)])
                s = session_from_args(args)
                ok = load_file(s, path, self.messages.append)
        except SystemExit as exc:               # (argparse: a bad flag: its last line;
            lines = out.getvalue().strip().splitlines()   #  the CLI's refusals: the reason)
            self.messages.append(exc.code if isinstance(exc.code, str) else
                                 lines[-1].split(": error: ")[-1] if lines else "bad settings")
            self.s = self.dbg = None
            return self.state()
        except Exception as exc:                # noqa: BLE001 (shown to the user)
            self.messages.append(f"{type(exc).__name__}: {exc}")
            self.s = self.dbg = None
            return self.state()
        if out.getvalue().strip():
            self.messages.append(out.getvalue().strip())
        if not ok:
            self.s = self.dbg = None
            return self.state()
        for note in [*s.adjusted, *s.program_notes, *s.assembler_messages()]:   # (the log)
            if note not in self.messages:
                self.messages.append(note)
        s.stop_at_end = True
        self.s = s
        self.dbg = Debugger(s, stdout=io.StringIO())
        self.shown = 0
        self._devices_reset()
        return self.state(listing=True)

    # -- settings -----------------------------------------------------------------------

    @staticmethod
    def _debug_parser():
        import argparse
        p = build_parser()
        sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        return sub.choices["debug"]

    def settings_schema(self) -> str:
        """The Settings dialog: its sections and fields, from the command
        line's own options (flag, kind, choices, help)."""
        import argparse
        acts = {a.dest: a for a in self._debug_parser()._actions}
        out = []
        for title, dests in SECTIONS:
            fields = []
            for d in dests:
                a = acts.get(d)
                if a is None:
                    continue
                flag = max(a.option_strings, key=len)
                if isinstance(a, (argparse._StoreTrueAction, argparse._StoreConstAction)):
                    kind, choices = "bool", None
                elif a.choices is not None:
                    kind, choices = "choice", [str(c) for c in a.choices]
                else:
                    kind, choices = "text", None
                fields.append({"dest": d, "flag": flag, "kind": kind, "choices": choices,
                               "help": (a.help or "").replace("%%", "%")})
            out.append({"title": title, "fields": fields})
        return json.dumps(out)

    def settings(self, flags: str = "") -> str:
        """For ``flags`` (the page's): each choice that does not go with the
        rest and why (compat.forbidden, as the window greys it)."""
        from dataclasses import replace
        from emips.cli import options_from_args
        from emips.compat import RULES
        from emips.options import Options
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                args = build_parser().parse_args(["debug", "x.asm", *shlex.split(flags)])
            o = options_from_args(args)
        except (SystemExit, ValueError) as exc:
            return json.dumps({"error": str(exc) or "bad settings"})
        release = args.release or ("R6" if o.arch == "nanomips" else "R5")
        fields = {f.name for f in __import__("dataclasses").fields(Options)}
        acts = {a.dest: a for a in self._debug_parser()._actions}
        out: dict = {}
        for _title, dests in SECTIONS:
            for d in dests:
                a = acts.get(d)
                if a is None or a.choices is None or d not in fields or d in PRIMARY:
                    continue
                why = {}
                for c in a.choices:
                    r = None                    # (only a dependent is greyed: a primary
                    try:                        #  choice makes its dependents follow)
                        cand = replace(o, **{d: c})
                        r = next((rule.reason(cand, release) for rule in RULES
                                  if rule.field == d and rule.broken(cand, release)), None)
                    except Exception:           # noqa: BLE001 (a value the rule cannot judge)
                        r = None
                    if r:
                        why[str(c)] = r
                if why:
                    out[d] = why
        return json.dumps({"forbidden": out})

    # -- running ------------------------------------------------------------------------

    def step(self) -> str:
        r = self.s.step()
        return self.state(result=r)

    def run(self, steps: int) -> str:
        """At most ``steps`` steps (one slice of Run)."""
        r = self.s.run(max_steps=steps)
        # (a slice the run goes on after: frames at most every FRAME_GAP s, no tab
        #  data; the slice that ends it: everything — the worker's stop rule)
        self.running = not (r.stopped or r.error or self.s.halted or self.s.needs_input())
        try:
            return self.state(result=r)
        finally:
            self.running = False

    def step_back(self) -> str:
        self.s.step_back()
        return self.state()

    def reset(self) -> str:
        self.s.reset()
        self.shown = 0
        self._devices_reset()
        return self.state()

    def give_input(self, line: str) -> str:
        """An input line — for an input dialog its answer (``!cancel`` =
        Cancel, ``!close`` = closed: as the command line)."""
        if self.s.pending_dialog() is not None:
            from emips.dialogs import typed
            line = typed(line)
        self.s.give_input(line)
        return self.state()

    def type_keys(self, text: str) -> str:
        self.s.type_keys(text)
        return self.state()

    def toggle_breakpoint(self, addr: int) -> str:
        self.s.toggle_breakpoint(addr)
        return self.state()

    def toggle_line(self, line: int) -> str:
        """A breakpoint on a source line (its first instruction, as the
        window's gutter)."""
        addr = self.s.line_address(line)
        if addr is not None:
            self.s.toggle_breakpoint(addr)
        return self.state()

    @staticmethod
    def datapath_layout() -> str:
        """The 5-stage datapath's drawing (emips.datapathdraw.layout)."""
        from emips import datapathdraw
        return json.dumps(datapathdraw.layout())

    @staticmethod
    def core_layout() -> str:
        """The 24K's drawing (emips.coredraw.layout24k)."""
        from emips import coredraw
        return json.dumps(coredraw.layout24k())

    def part_info(self, key: str, bit: int = 0) -> str:
        """What a datapath part is and does this cycle (the window's part
        dialog: emips.datapathinfo.part_info), with its gates
        (emips.partlogic: the PLAs, ALU bit ``bit``'s slice, their text)."""
        from emips import datapathinfo, partlogic
        info = datapathinfo.part_info(self.s, key)
        logic = partlogic.part_logic(self.s, key, int(bit)) if not key.startswith("24k:") \
            else partlogic.PartLogic()          # (the 24K's parts: text only, as the window)
        return json.dumps({"heading": info.heading, "stage": info.part.stage,
                           "instruction": info.instruction, "inside": info.part.inside,
                           "now": info.now, "html": logic.html, "width": logic.width,
                           "drawings": [[name, sc.to_json()] for name, sc in logic.drawings]})

    def memory_segments(self) -> str:
        from emips.viewdata import MEMORY_SEGMENTS
        return json.dumps(MEMORY_SEGMENTS)

    def _devices_reset(self) -> None:
        """What of the program's dialogs / notes / frames the page has had
        (a fresh program or a reset sends them all again)."""
        self.dialogs_seen = 0
        self.notes_seen = 0
        self.frame_digest = None
        self.frame_at = 0.0

    def pane_action(self, name: str, action: str, args: str = "{}") -> str:
        """A pane's button / value (emips.panedata.act: what the window's
        pane does); the new state."""
        from emips import panedata as P
        try:
            P.act(self.s, name, action, json.loads(args))
        except Exception as exc:                # noqa: BLE001 (shown to the user)
            self.messages = [f"{type(exc).__name__}: {exc}"]
            st = json.loads(self.state())
            self.messages = []
            return json.dumps(st)
        return self.state()

    def set_views(self, spec: str) -> str:
        """The panes the page shows now (JSON: {"memory": {segment, address,
        physical, bytes, doubles}, "chart": true ...}): each state carries
        their data."""
        self.views = json.loads(spec)
        return self.state()

    def debugger(self, line: str) -> str:
        """One debugger command; its output and the new state."""
        buf = io.StringIO()
        self.dbg.stdout = buf
        try:
            with contextlib.redirect_stdout(buf):
                self.dbg.onecmd(line)
        except Exception as exc:                # noqa: BLE001 (shown to the user)
            buf.write(f"{type(exc).__name__}: {exc}\n")
        state = json.loads(self.state())
        state["debugger"] = buf.getvalue()
        return json.dumps(state)

    # -- what the page shows ------------------------------------------------------------

    def state(self, result=None, listing: bool = False) -> str:
        st: dict = {"messages": [m.replace(WORK + "/", "") for m in self.messages],
                    "loaded": self.s is not None}
        s = self.s
        if s is None:
            return json.dumps(st)
        m = s.machine
        out = s.output
        st["output"] = out[self.shown:] if len(out) >= self.shown else out
        st["reset_output"] = len(out) < self.shown
        self.shown = len(out)
        st["steps"] = s.steps
        st["halted"] = bool(s.halted)
        st["needs_input"] = bool(s.needs_input())
        st["pipeline"] = s.mode == "pipeline"
        st["line"] = s.current_line()
        st["pc"] = m.pc & 0xFFFFFFFF if m is not None else None
        try:
            st["stages"] = {k: v for k, v in s.stage_pcs().items() if v is not None} \
                if st["pipeline"] else {}
        except Exception:                       # noqa: BLE001 (a view: never fatal)
            st["stages"] = {}
        st["breakpoints"] = sorted(s.breakpoints)
        digits = s.hex_digits()                 # (16: MIPS64, RV64)
        bits = digits * 4
        st["regs"] = [[name, f"0x{value:0{digits}x}",
                       value - (1 << bits) if value >> (bits - 1) else value, changed]
                      for name, value, changed in
                      ((n, v & ((1 << bits) - 1), c) for n, v, c in s.registers())]
        try:
            st["special"] = []
            for n, v in s.special_registers()[:6]:
                d = s.hex_digits(n)
                st["special"].append([n, f"0x{v & ((1 << d * 4) - 1):0{d}x}"])
        except Exception:                       # noqa: BLE001
            st["special"] = []
        if result is not None:
            st["stopped"] = result.stopped
            st["error"] = result.error
        if listing:
            st["listing"] = [[a, f"0x{a:08x}", f"{w:08x}" if isinstance(w, int) else str(w),
                              t, ln] for a, w, t, ln in s.disassembly()]
        self._view_data(st)
        self._device_data(st)
        return json.dumps(st)

    # -- checks (emips.check: the window's Run -> Check, the CLI's check) -------------------

    def check_start(self, name: str, text: str, flags: str = "") -> str:
        """The program's checks — its ``# check:`` lines, else ``NAME.check``
        saved next to it — each run on a fresh machine (the run's ``eMIPS:``
        flags over the page's); :meth:`check_advance` runs them."""
        from emips import check as K
        self.save(name, text)
        path = os.path.join(WORK, os.path.basename(name))
        self.checking = None
        try:
            found = K.checks_of(text, path)
        except K.CheckError as exc:
            where = os.path.basename(exc.source) if exc.source else os.path.basename(name)
            return json.dumps({"error": f"{where}: {exc}", "done": True})
        if not found.runs:
            stem = os.path.splitext(os.path.basename(name))[0]
            return json.dumps({"error": f"{os.path.basename(name)}: no checks — add "
                               f"'# check: ...' lines to the program (# check: $v0 == 3, "
                               f"# check: output == \"hello\\n\", # input: 5 ...) or open a "
                               f"{stem}.check file", "done": True})
        self.checking = {"runs": found.runs, "i": -1, "ex": None, "path": path,
                         "flags": flags, "lines": [], "results": [],
                         "source": os.path.basename(str(found.file)) if found.file else ""}
        return self._check_next()

    def _check_next(self) -> str:
        from emips import check as K
        c = self.checking
        c["i"] += 1
        if c["i"] >= len(c["runs"]):
            return self._check_state(done=True)
        run = c["runs"][c["i"]]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                args = build_parser().parse_args(
                    ["debug", c["path"], *shlex.split(c["flags"]), *(run.flags or [])])
                s = session_from_args(args)
                errors: list[str] = []
                if not load_file(s, c["path"], errors.append):
                    raise ValueError(errors[0] if errors else "cannot load the program")
        except SystemExit as exc:
            lines = out.getvalue().strip().splitlines()
            why = exc.code if isinstance(exc.code, str) else \
                lines[-1].split(": error: ")[-1] if lines else "bad flags"
            self.checking = None
            return json.dumps({"error": f"run {c['i'] + 1}: {why}", "done": True,
                               "lines": c["lines"]})
        except Exception as exc:                # noqa: BLE001 (shown to the user)
            self.checking = None
            return json.dumps({"error": f"run {c['i'] + 1}: {exc}", "done": True,
                               "lines": c["lines"]})
        c["ex"] = K.Execution(s, run, limit=K.default_limit(s))
        return self._check_state()

    def check_advance(self, steps: int) -> str:
        """Up to ``steps`` more steps of the current run; its outcomes when
        it is over, then the next run."""
        from emips import check as K
        c = getattr(self, "checking", None)
        if c is None:
            return json.dumps({"done": True})
        ex = c["ex"]
        if ex.advance(steps):
            res = ex.result()
            c["results"].append(res)
            many = len(c["runs"]) > 1 or res.run.name
            rows: list = []
            if many:
                rows.append(["head", f"run {c['i'] + 1}" + (f": {res.run.name}"
                                                            if res.run.name else ""), 0])
            say: list[str] = []
            K.report([res], say.append, header=False)
            for ln in say:
                kind = "pass" if ln.startswith("PASS") else "fail" if ln.startswith("FAIL") \
                    else "info"
                rows.append([kind, ln, 0])
            line_of = {} if c["source"] else \
                {o.check.text: o.check.line for o in res.outcomes}   # (a .check file's: not)
            for r in rows:                     # (the check's line: a click shows it)
                for text, ln in line_of.items():
                    if r[0] in ("pass", "fail") and text in r[1]:
                        r[2] = ln
            if res.output:
                rows.append(["output", res.output, 0])
            c["lines"] += rows
            return self._check_next()
        return self._check_state()

    def _check_state(self, done: bool = False) -> str:
        c = self.checking
        st = {"done": done, "lines": c["lines"], "runs": len(c["runs"]),
              "run": min(c["i"] + 1, len(c["runs"])),
              "source": c["source"]}
        if done:
            from emips import check as K
            st["passed"], st["total"] = K.report(c["results"], lambda _t: None)
            self.checking = None
        else:
            st["steps"] = c["ex"].done
        return json.dumps(st)

    # -- M27's devices: dialogs, the display, files, sound ---------------------------------

    FRAME_GAP = 0.1                             # s between frames sent during a run

    def _device_data(self, st: dict) -> None:
        """What the program showed / drew / played since the last state: the
        new dialogs, the input dialog it waits in, the display's frame (when
        it changed), the files, the new notes."""
        from emips import dialogs as D
        s = self.s
        dl = s.dialogs()
        if len(dl) < self.dialogs_seen:         # (a reset / step back: never shown again)
            self.dialogs_seen = len(dl)
        st["dialogs"] = [{"title": D.TITLES.get(d.service, "eMIPS"), "kind": d.kind,
                          "text": d.text, "icon": d.icon, "input": d.is_input,
                          "line": d.line()} for d in dl[self.dialogs_seen:]]
        self.dialogs_seen = len(dl)
        pending = s.pending_dialog() if st.get("needs_input") else None
        if pending is not None:
            service, kind, text = pending
            st["dialog"] = {"title": D.TITLES.get(service, "eMIPS"), "kind": kind,
                            "text": text}
        dev = s.display_device()
        if dev is not None:
            now = time.monotonic()
            if not getattr(self, "running", False) or now - self.frame_at >= self.FRAME_GAP:
                frame = s.display_frame()
                if frame is not None:
                    digest = hash((frame.rgb, frame.on))
                    if digest != self.frame_digest:
                        self.frame_digest, self.frame_at = digest, now
                        sp = frame.spec
                        st["display"] = {
                            "columns": sp.columns, "rows": sp.rows, "unit_w": sp.unit_w,
                            "unit_h": sp.unit_h, "describe": sp.describe(),
                            "rgb": base64.b64encode(frame.rgb).decode(),
                            "written": frame.written, "unmapped": frame.unmapped,
                            "on": frame.on}
        area = s.file_area
        if area is not None and (area.data or area.given):
            st["files"] = {"lines": s.file_lines(),
                           "names": sorted(set(area.data) | set(area.given))}
        notes = s.midi_events()
        if len(notes) < self.notes_seen:
            self.notes_seen = 0
        if len(notes) > self.notes_seen or st.get("reset_output"):
            from emips import midi
            st["notes"] = {"new": [[midi.start_ms(e), e.pitch, e.duration, e.instrument,
                                    e.volume] for e in notes[self.notes_seen:]],
                           "lines": midi.lines(notes), "count": len(notes)}
            self.notes_seen = len(notes)

    def display_word(self, column: int, row: int) -> str:
        """The word under the mouse on the display: its addresses and value."""
        w = self.s.display_word(column, row) if self.s is not None else None
        if w is None:
            return json.dumps("")
        va, pa, word = w
        where = f"0x{va:08x}" + ("" if pa is None or pa == va else f" (PA 0x{pa:08x})")
        return json.dumps(f"({column}, {row}): {where} = 0x{word:08x}")

    def file_bytes(self, name: str) -> str:
        """A file of the program as the run has left it, base64 (a download)."""
        return json.dumps(base64.b64encode(self.s.file_data(name)).decode())

    def midi_download(self, kind: str) -> str:
        """The notes as a WAV or a Standard MIDI File, base64."""
        from emips import midi
        notes = self.s.midi_events()
        return json.dumps(base64.b64encode(midi.wav(notes) if kind == "wav"
                                           else midi.midi_file(notes)).decode())

    @staticmethod
    def timbres() -> str:
        """The synthesizer's timbres (emips.midi.TIMBRES) for the page's
        live playback: family -> [harmonics, attack, decay, sustain,
        release, percussive]."""
        from emips import midi
        return json.dumps({k: list(v) for k, v in midi.TIMBRES.items()})

    def _view_data(self, st: dict) -> None:
        """The shown panes' data (emips.viewdata: the window's own rows)."""
        from emips import viewdata as V
        s = self.s
        if getattr(self, "running", False):     # (a Run slice: as the window, only the
            return                              #  registers and the console until it stops)
        mem = self.views.get("memory")
        if mem is not None:
            try:
                seg = int(mem.get("segment", 0))
                layout = V.memory_layout(s, bool(mem.get("bytes")), mem.get("doubles", True))
                start = V.memory_start(s, seg, mem.get("address", ""), bool(mem.get("physical")))
                st["memory"] = {"heads": V.memory_heads(layout), "wide": s.wide,
                                "rows": V.memory_rows(s, start, V.memory_rows_count(seg),
                                                      bool(mem.get("physical")), layout)}
            except Exception as exc:            # noqa: BLE001 (a view: never fatal)
                st["memory"] = {"error": f"{type(exc).__name__}: {exc}"}
        if self.views.get("instr"):
            try:
                st["instr"] = V.instruction(s.last_view)
            except Exception as exc:            # noqa: BLE001 (a view: never fatal)
                st["instr"] = {"error": f"{type(exc).__name__}: {exc}"}
        if self.views.get("datapath"):          # (by the pipeline model: as the window)
            from emips import coredraw, coreview, datapathdraw
            try:
                p = s.pipeline
                if p is not None and getattr(p, "name", "") == "p8700":
                    st["datapath"] = {"model": "p8700", "view": coredraw.p8700_view(s)}
                elif p is not None and coreview.is_24k(p):
                    st["datapath"] = {"model": "24k", "view": coredraw.draw24k(s)}
                else:
                    st["datapath"] = {"model": "5-stage",
                                      **datapathdraw.draw(s).to_json()}
            except Exception as exc:            # noqa: BLE001 (a view: never fatal)
                st["datapath"] = {"error": f"{type(exc).__name__}: {exc}"}
        if self.views.get("pane"):              # (W3: the window's other panes)
            from emips import panedata as P
            name = self.views["pane"]
            avail = P.available(s)
            if name not in avail:
                name = avail[0]
            memo = self.pane_memo.setdefault(name, {})
            memo["step"] = s.steps
            try:
                st["pane"] = {"name": name, "available": [[k, P.TITLES[k]] for k in avail],
                              **P.pane(s, name, self.views.get("pane_opts") or {}, memo)}
            except Exception as exc:            # noqa: BLE001 (a view: never fatal)
                st["pane"] = {"name": name, "available": [[k, P.TITLES[k]] for k in avail],
                              "blocks": [P.note(f"{type(exc).__name__}: {exc}")],
                              "controls": []}
        if self.views.get("chart"):
            cycles, rows = V.pipeline_chart(s)
            st["chart"] = {"cycles": cycles,
                           "rows": [[f"{pc:08x}", text, cells] for pc, text, cells in rows]}


WEB = Web()
