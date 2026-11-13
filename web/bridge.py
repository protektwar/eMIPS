"""eMIPS in the browser: the page's Python side (it runs in Pyodide).

One Session, loaded as the command line loads a program (``python3 -m
emips debug FILE``: the program's ``# eMIPS:`` line, then the choices the
page adds as flags), and the state the page shows, as JSON.  The page
runs a program in slices (``run``) so that it stays responsive; the
debugger box sends its lines to a ``cli.Debugger`` on the same Session.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shlex

from emips.__main__ import build_parser
from emips.cli import Debugger, load_file, session_from_args

#: the program and its companion files (a lab's disk image ...) live here
WORK = "/work"


class Web:
    def __init__(self) -> None:
        self.s = None
        self.dbg: Debugger | None = None
        self.messages: list[str] = []
        self.shown = 0                          # (console text already sent)

    # -- loading ------------------------------------------------------------------------

    def save(self, name: str, text: str) -> None:
        """A companion file (a disk image) next to the program."""
        os.makedirs(WORK, exist_ok=True)
        with open(os.path.join(WORK, os.path.basename(name)), "w", encoding="utf-8") as f:
            f.write(text)

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
        except SystemExit:                      # (argparse: a bad flag: its last line)
            lines = out.getvalue().strip().splitlines()
            self.messages.append(lines[-1].split(": error: ")[-1] if lines else "bad settings")
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
        s.stop_at_end = True
        self.s = s
        self.dbg = Debugger(s, stdout=io.StringIO())
        self.shown = 0
        return self.state(listing=True)

    # -- running ------------------------------------------------------------------------

    def step(self) -> str:
        r = self.s.step()
        return self.state(result=r)

    def run(self, steps: int) -> str:
        """At most ``steps`` steps (one slice of Run)."""
        r = self.s.run(max_steps=steps)
        return self.state(result=r)

    def step_back(self) -> str:
        self.s.step_back()
        return self.state()

    def reset(self) -> str:
        self.s.reset()
        self.shown = 0
        return self.state()

    def give_input(self, line: str) -> str:
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
        return json.dumps(st)


WEB = Web()
