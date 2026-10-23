"""The interactive debugger of the command line (PLAN M7):
``python3 -m emips debug prog.asm``.

A gdb / SPIM-style prompt over :class:`emips.session.Session` — the same
engine as the GUI, so stepping, step back, breakpoints, console input,
strict mode and the boot / exception options behave identically.  Commands
can also come from ``-c "cmd; cmd"`` or a file (``-x``), for handouts and
tests.  Type ``help`` at the prompt for the list.

Locations (``break``, ``until``, ``mem``, ``dis``, ``list``): ``0x...`` is
an address, a plain number is a source line, a name is a label, ``$t0`` a
register's value, ``pc`` the current instruction.
"""

from __future__ import annotations

import cmd
import shlex
import struct
import sys
from pathlib import Path
from typing import TextIO

from . import cp0 as C
from . import ejtag as EJ
from . import pdtrace as PDT
from . import tap as TAPM
from .assembler import AsmError
from . import tools
from .options import (ARCHES, RV_EXTENSIONS, RV_MODES, BOOT_MODES, COMPRESSED, DSP_LEVELS, MAX_CORES, MSA_LEVELS, NANO_CRC,
                      NANO_GINV, Options)
from .session import Session, StepResult
from .timing import MEMORY_TIMING
from .trace import format_view

M32 = 0xFFFFFFFF
STAGES = ("IF", "ID", "EX", "MEM", "WB")
RUN_CHUNK = 20_000


class CliError(Exception):
    """A bad command argument (printed, the prompt goes on)."""


class Debugger(cmd.Cmd):
    intro = ""
    prompt = "(emips) "

    def __init__(self, session: Session, *, stdin: TextIO | None = None,
                 stdout: TextIO | None = None, program_input: TextIO | None = None,
                 echo: bool = False) -> None:
        super().__init__(stdin=stdin, stdout=stdout)
        if stdin is not None:
            self.use_rawinput = False
        self.s = session
        #: where read_* syscalls get their lines when the program asks
        self.program_input = program_input
        #: print each command before running it (scripts)
        self.echo = echo
        if echo:
            self.prompt = ""                   # the echo shows it
        self._printed = 0                      # program output already shown
        #: program output is printed as it happens (Session.on_output) by
        #: the caller, not after each command (``run``)
        self.live_output = False
        self.prev_regs = self._gprs()

    # -- plumbing --------------------------------------------------------------

    def say(self, text: str = "") -> None:
        self.stdout.write(text + "\n")

    def precmd(self, line: str) -> str:
        if self.echo and line.strip() and line != "EOF":
            self.say(f"{Debugger.prompt}{line}")
        return line

    #: commands of MIPS hardware (CP0, EJTAG, the CPS, the 24K ...): on a
    #: RISC-V hart they say what to use instead
    MIPS_ONLY = frozenset(("srs", "ejtag", "probe", "board", "disk", "trace", "tap", "cp0",
                           "dsp", "msa", "mt", "vp", "tc", "translate", "cm", "cps", "gic",
                           "tlb", "mpu", "perf", "core"))

    def onecmd(self, line: str) -> bool:
        word = line.split()[0] if line.split() else ""
        if word in self.MIPS_ONLY and self.s.rv:
            self.say(f"error: {word} is a MIPS command; this is a RISC-V hart — "
                     "regs, csr, fpu, mem, dis show its state")
            return False
        try:
            return super().onecmd(line)
        except CliError as exc:
            self.say(f"error: {exc}")
        except KeyboardInterrupt:
            self.say("interrupted")
        return False

    def postcmd(self, stop: bool, line: str) -> bool:
        self._flush_output()
        return stop

    def emptyline(self) -> bool:                # Enter does nothing (no repeat)
        return False

    def default(self, line: str) -> bool:
        self.say(f"unknown command {line.split()[0]!r} — type help")
        return False

    def _flush_output(self, newline: bool = True) -> None:
        """Show new program output (``newline=False``: leave a prompt like
        "Enter a number: " open for the input)."""
        out = self.s.output[self._printed:]
        if out and not self.live_output:
            self.stdout.write(out if out.endswith("\n") or not newline else out + "\n")
            self.stdout.flush()
            self._printed = len(self.s.output)

    def _gprs(self) -> list[int]:
        m = self.s.machine
        return [m.read_reg(i) for i in range(32)] if m is not None else [0] * 32

    def _need_program(self) -> None:
        if self.s.machine is None:
            raise CliError("no program loaded (load FILE)")

    # -- locations -------------------------------------------------------------

    def resolve_all(self, text: str) -> list[int]:
        """Every address of a location: a source line of a compiled program
        can start in several places (a loop's condition, its increment)."""
        self._need_program()
        s, t = self.s, text.strip()
        if s.program.c_lines:
            file, _, num = t.rpartition(":")
            if num.isdigit() and (file or t.isdigit()) and not t.lower().startswith("0x"):
                if not file:                           # the current file (main's
                    file = s.default_source_file()     # before the program gets there)
                addrs = s.line_addresses(file, int(num))
                if not addrs:
                    raise CliError(f"no code on or after {t}")
                return addrs
        return [self.resolve(t)]

    def resolve(self, text: str) -> int:
        """An address from ``0x...`` / a line number / a label / ``$reg`` /
        ``pc``; for a compiled program a line is a C line (``17``,
        ``file.c:17``) and a function name works."""
        self._need_program()
        s, t = self.s, text.strip()
        if s.program is not None and s.program.c_lines and (
                t.isdigit() or ":" in t and t.rpartition(":")[2].isdigit()):
            return self.resolve_all(t)[0]
        if t.lower() == "pc":
            pc = s.current_pc()
            if pc is None:
                raise CliError("no instruction is about to execute")
            return pc
        if t.startswith("$") or (s.rv and t not in s.program.symbols
                                 and self._gpr(t) is not None):    # (RISC-V: a0, x5)
            n = self._gpr(t)
            if n is None:
                raise CliError(f"not a register: {t}")
            return s.machine.read_reg(n)
        if t.lower().startswith(("0x", "0b", "0o")):
            try:
                return int(t, 0) & s.machine.amask      # (MIPS64: 64-bit addresses)
            except ValueError:
                raise CliError(f"not an address: {t}") from None
        if t.isdigit():
            line = int(t)
            addrs = sorted(a for a, src in s.program.source_map.items() if src.line == line)
            if not addrs:
                raise CliError(f"no instruction on line {line}")
            return addrs[0]
        if t in s.program.symbols:
            return s.program.symbols[t] & M32
        raise CliError(f"unknown location {t!r} (0x address, line number, label, "
                       "$register, pc" + (", file.c:LINE, function"
                                          if s.program.c_lines else "") + ")")

    def _where(self, pc: int | None, marker: str = "=>", cpu: int | None = None) -> str:
        s = self.s
        if pc is None:
            return f"{marker} (pipeline empty)"
        src = s.source_line(pc, cpu)
        text = s.instruction_text(pc, cpu)
        where = f"   line {src.line}: {src.text.strip()}" if src and s.kind == "asm" else ""
        c = s.source_position(pc)
        if c is not None:                                   # a compiled program: its C line
            where = f"   {Path(c[0]).name}:{c[1]}: {c[2]}"
        if s.cps is not None and cpu is None:
            cpu = s.cpu
        core = f"{s.cpu_name(cpu)}  " if cpu is not None else ""
        if s.machines[s.cpu if cpu is None else cpu].debug_mode:
            core += "[debug] "
        return f"{marker} {core}0x{pc:08x}  {text:<30}{where}"

    # -- reporting after running -----------------------------------------------

    def _report(self, r: StepResult, show: bool = True) -> None:
        s = self.s
        self._flush_output()
        if r.error:
            self.say(f"stopped: {r.error}")
        elif r.stopped == "breakpoint":
            self.say("breakpoint")
        elif r.stopped == "cursor":
            pass
        elif r.stopped == "end":
            self.say("the program ran past its last instruction (end it with "
                     f"li ${'t4' if s.nano else 'v0'}, 10 / syscall)")
        elif r.stopped == "exit":
            code = s.services.exit_code if s.services is not None else None
            self.say("the program exited" + (f" with code {code}" if code else ""))
        elif r.stopped == "limit":
            self.say("step limit reached")
        elif r.stopped == "input":
            self.say("the program waits for input: give it with  input TEXT")
        if show:
            self._show_position(r.exceptions)

    def _exception_line(self, e: dict) -> str:
        """``exception TLBL (TLB refill: ...) at PC, BadVAddr -> vector``."""
        what = e["error"] + ((" (XTLB refill)" if e.get("xtlb") else " (TLB refill)")
                             if e.get("refill") else "")
        why = f": {e['why']}" if e.get("why") else ""
        bad = f", BadVAddr 0x{e['badvaddr']:08x}" if e.get("badvaddr") is not None else ""
        core = f"{self.s.cpu_name(e['cpu'])}: " if e.get("cpu") is not None else ""
        if e.get("vector_number") is not None:      # a vectored interrupt
            what += f" (vector {e['vector_number']})"
        to = f"0x{e['vector']:08x}" + (f", shadow set {e['shadow_set']}"
                                       if e.get("shadow_set") else "")
        if e.get("debug"):                          # EJTAG: into Debug Mode
            return f"   {core}debug exception {what} at 0x{e['addr']:08x} -> {to} (Debug Mode)"
        if e.get("debug_mode"):                     # in Debug Mode: the debug vector again
            return (f"   {core}exception {what} in Debug Mode at 0x{e['addr']:08x}{why} -> "
                    f"{to} (Debug.DExcCode {e['dexccode']})")
        return f"   {core}exception {what} at 0x{e['addr']:08x}{bad}{why} -> {to}"

    def _show_position(self, exceptions: list | None = None) -> None:
        s = self.s
        if s.machine is None:
            return
        if s.pipeline is not None:
            self.say(self._cycle_line())
            self._other_cores()
            return
        v = s.last_view
        if v is not None:
            effects = [f"{n} = 0x{new:08x}" for n, _o, new in v.reg_writes]
            effects += [f"[0x{a:08x}] = 0x{nw:0{2 * n}x}" for a, n, _o, nw in v.mem_writes]
            if v.branch is not None:
                effects.append("taken" if v.branch.taken else "not taken")
            if v.exception:
                effects.append(f"EXCEPTION {v.exception}")
            self.say((f"   0x{v.pc:08x}  {(v.source_text or v.text):<30}"
                      + ("  ; " + ", ".join(effects)
                                                           if effects else "")).rstrip())
        for e in exceptions or ():
            self.say(self._exception_line(e))
        if s.halted:
            self.say("   (ended)")
        else:
            self.say(self._where(s.current_pc()))
            self._other_cores()

    def _other_cores(self) -> None:
        """A multi-core system: where the other running cores are."""
        s = self.s
        if s.cps is None:
            return
        for i in s.running():
            if i != s.cpu:
                p = s.pipelines[i]
                pc = s.machines[i].pc if p is None else (
                    p.current_slot().pc if p.current_slot() is not None else None)
                self.say(self._where(pc, "  ", cpu=i))

    def _cycle_line(self) -> str:
        s = self.s
        d = s.stage_details()
        ev = d.pop("_events", {})
        names = s.pipeline.STAGES if s.pipeline is not None else STAGES
        cells = [f"{st} {(d.get(st) or {}).get('text') or '-'}" for st in names]
        line = f"cycle {ev.get('cycle', 0):<4} " + " | ".join(cells)
        extra = []
        if ev.get("stall"):
            extra.append(f"   stall: {ev['stall']}")
        extra += [f"   {e}" for e in ev.get("events", [])]
        return "\n".join([line] + extra)

    def _input_needed(self) -> bool:
        """The program waits for a line: read it from the program input
        stream (or the prompt) and go on.  False if there is none."""
        if self.program_input is None:
            return False
        if self.program_input is sys.stdin and sys.stdin.isatty() \
                and (not self.s.output or self.s.output.endswith("\n")):
            self.stdout.write("program input> ")    # the program printed no prompt
            self.stdout.flush()
        line = self.program_input.readline()
        if not line:
            return False
        self.s.give_input(line.rstrip("\n"))
        return True

    def _run(self, max_steps: int | None = None) -> StepResult:
        """Run (in chunks, Ctrl+C stops) until something stops it."""
        s, total = self.s, StepResult()
        while True:
            chunk = RUN_CHUNK if max_steps is None else min(RUN_CHUNK, max_steps - total.steps)
            if chunk <= 0:
                return total
            r = s.run(max_steps=chunk)
            total.steps += r.steps
            total.stopped, total.error = r.stopped, r.error
            # between chunks the output goes on as it is (slow serial text);
            # a stop ends its line so the debugger's message starts a new one
            self._flush_output(newline=bool(r.stopped or r.error) and r.stopped != "input")
            if r.stopped == "input" and self._input_needed():
                continue
            if r.stopped or r.error:
                return total

    # ============================================================== commands
    # running

    def do_step(self, arg: str) -> None:
        """step [N]   (s) one instruction — or one clock cycle in pipeline mode"""
        self._need_program()
        n = self._count(arg)
        r = StepResult()
        taken: list = []                        # every exception of the N steps
        for _ in range(n):
            r = self.s.step()
            if r.stopped == "input" and self._input_needed():
                r = self.s.step()
            taken += r.exceptions
            if r.stopped or r.error:
                break
            if self.s.pipeline is not None and n > 1:
                self.say(self._cycle_line())
        r.exceptions = taken
        self._report(r, show=not (self.s.pipeline is not None and n > 1))

    do_s = do_step

    def do_next(self, arg: str) -> None:
        """next [N]   (n) like step, but a call (jal, jalr, bal, balc ...) runs
        to its return as one step"""
        self._need_program()
        r = StepResult()
        for _ in range(self._count(arg)):
            r = self.s.step_over()
            if r.stopped or r.error:
                break
        self._report(r)

    do_n = do_next

    def do_back(self, arg: str) -> None:
        """back [N]   undo N steps (replays the program from the start)"""
        self._need_program()
        for _ in range(self._count(arg)):
            if not self.s.step_back():
                self.say("at the start")
                break
        self._printed = min(self._printed, len(self.s.output))
        self._show_position()

    def do_continue(self, arg: str) -> None:
        """continue [N]   (c, run, r) run until a breakpoint, the end, an error or
        an input request (at most N steps)"""
        self._need_program()
        self._report(self._run(self._count(arg) if arg.strip() else None))

    do_c = do_r = do_run = do_continue

    def do_until(self, arg: str) -> None:
        """until LOCATION   (u) run until the instruction at LOCATION is next"""
        self._need_program()
        if not arg.strip():
            raise CliError("until what? (an address, line, label)")
        self.s.run_to = self.resolve(arg)
        self._report(self._run())

    do_u = do_until

    def do_finish(self, arg: str) -> None:
        """finish   run until the current function returns to its caller (the
        call stack of bt: deeper calls of a recursive function returning to
        the same address do not stop it)"""
        self._need_program()
        self._report(self.s.finish())

    def do_stepline(self, arg: str) -> None:
        """stepline [N]   (sl) run to the next source line — a C line of a
        compiled program, a source line of an assembled one; into calls"""
        self._line_steps(arg, over=False)

    def do_nextline(self, arg: str) -> None:
        """nextline [N]   (nl) like stepline, but a call runs as one step"""
        self._line_steps(arg, over=True)

    do_sl, do_nl = do_stepline, do_nextline

    def _line_steps(self, arg: str, over: bool) -> None:
        self._need_program()
        r = StepResult()
        taken: list = []
        for _ in range(self._count(arg)):
            r = self.s.step_line(over=over)
            taken += r.exceptions
            if r.stopped or r.error:
                break
        r.exceptions = taken
        self._report(r)

    def do_bt(self, arg: str) -> None:
        """bt   (backtrace) the calls not returned yet, innermost first: the
        function, the call / the current instruction, its source line"""
        self._need_program()
        frames = self.s.backtrace()
        if not frames:
            self.say("no instruction is about to execute")
        for f in frames:
            where = ""
            if f["line"] is not None:
                where = f"   {Path(f['line'][0]).name}:{f['line'][1]}: {f['line'][2]}"
            elif f["source"] is not None and self.s.kind == "asm":
                where = f"   line {f['source'].line}: {f['source'].text.strip()}"
            self.say(f"#{f['level']}  0x{f['pc']:08x} in {f['function'] or '??'}{where}")

    do_backtrace = do_bt

    def do_reset(self, arg: str) -> None:
        """reset   restart the program (breakpoints and given input are kept)"""
        self._need_program()
        self.s.reset()
        self._printed = 0
        self._show_position()

    # -- breakpoints -------------------------------------------------------------

    def do_break(self, arg: str) -> None:
        """break [LOCATION]   (b) set a breakpoint; without a location: list them"""
        if not arg.strip():
            bps = sorted(self.s.breakpoints)
            if not bps:
                self.say("no breakpoints")
            for a in bps:
                self.say(self._where(a, " *"))
            return
        for addr in self.resolve_all(arg):
            if self.s.program is not None and addr not in self.s.program.source_map:
                self.say(f"warning: 0x{addr:08x} is not an instruction of the program")
            self.s.breakpoints.add(addr)
            self.say(f"breakpoint at {self._where(addr, '').strip()}")

    do_b = do_break

    def do_delete(self, arg: str) -> None:
        """delete LOCATION | all   (d) remove a breakpoint (or all)"""
        if arg.strip() == "all":
            self.s.breakpoints.clear()
            self.say("all breakpoints deleted")
            return
        wanted = self.resolve_all(arg)
        addrs = [a for a in wanted if a in self.s.breakpoints]
        if not addrs:
            raise CliError(f"no breakpoint at 0x{wanted[0]:08x}")
        self.s.breakpoints.difference_update(addrs)

    do_d = do_delete

    # -- looking -----------------------------------------------------------------

    def do_regs(self, arg: str) -> None:
        """regs [NAMES]   the 32 GPRs (changed by the last command marked *), or
        only some: regs $t0 $sp"""
        self._need_program()
        s = self.s
        rows = s.registers()
        if arg.strip():
            want = {self._gpr(t) for t in arg.split()}
            if None in want:
                raise CliError("regs takes register names: regs $t0 $sp")
            rows = [r for i, r in enumerate(rows) if i in want]
            w = s.hex_digits()
            for name, value, changed in rows:
                self.say(f"{name:>6} = 0x{value:0{w}x}  {self._signed(value, 4 * w)}"
                         + ("  *" if changed else ""))
            return
        w = s.hex_digits()
        for i in range(0, 32, 4):
            self.say("  ".join(f"{n:>5}{'*' if ch else ' '}0x{v:0{w}x}"
                               for n, v, ch in rows[i:i + 4]))
        m = s.machine
        if m.rv:                                # (no HI / LO, no shadow sets)
            self.say(f"   pc  0x{m.pc:0{w}x}    {_RV_MODE[m.priv]}-mode")
            return
        hilo = "" if m.nano else f"    hi  0x{m.hi:0{w}x}    lo  0x{m.lo:0{w}x}"   # (none)
        self.say(f"   pc  0x{m.pc:0{w}x}{hilo}"
                 + (f"    shadow set {m.srs_field(C.SRS_CSS)} of 0-{m.hss}"
                    if m.srs_field(C.SRS_CSS) else ""))

    def do_srs(self, arg: str) -> None:
        """srs [N]   the interrupt mode and the GPR shadow sets (SRSCtl: the
        current set CSS, the previous PSS, the exceptions' ESS; SRSMap:
        the set of each interrupt vector); srs N: set N's registers"""
        self._need_program()
        s = self.s
        if arg.strip():
            try:
                rows = s.shadow_set(int(arg, 0))
            except ValueError as exc:
                raise CliError(str(exc)) from None
            for i in range(0, 32, 4):
                self.say("  ".join(f"{n:>5} 0x{v:08x}" for n, v in rows[i:i + 4]))
            return
        for line in s.interrupt_lines():
            self.say(line)

    def do_ejtag(self, arg: str) -> None:
        """ejtag [dint | ib ... | db ... | watch ... | clear]   EJTAG debug
        state: Debug Mode, the Debug register (the last debug exception,
        single step SSt, LSNM, CountDM, IEXI), DEPC, DESAVE, the DCR (drseg
        0xFF300000), the debug vector, the hardware breakpoints and watch
        registers.  The debugger sets them as a probe would:
          ejtag dint                     a debug interrupt (the DINT input)
          ejtag ib N LOC [mask M] [asid A] [tc T] [trigger]
                                         instruction breakpoint N (0-3): a DIB
                                         exception there (trigger: only IBS.BS)
          ejtag db N LOC [load|store] [mask M] [value V] [asid A] [tc T]
                [trigger]
                                         data breakpoint N (0-1): DDBL / DDBS
                                         (precise); with a value: DDBLImpr /
                                         DDBSImpr after the access (imprecise)
                                         (tc T: only thread context T, the
                                         MT Module's TCuse)
          ejtag watch N LOC [i|r|w|rw] [mask M] [asid A]
                                         watch registers N (0, 1: fetches; 2,
                                         3: loads / stores): a Watch exception
          ejtag ib|db|watch N off        turn one off
          ejtag clear                    clear IBS / DBS.BS, WatchHi, Cause.WP"""
        self._need_program()
        words = arg.split()
        if not words:
            for line in self.s.debug_lines():
                self.say(line)
            return
        cmd = words[0]
        if cmd == "dint" and len(words) == 1:
            if self.s.machine.debug_mode:
                self.say("Debug Mode: a debug interrupt is ignored (MD00047 2.3.10)")
                return
            self.s.debug_interrupt()
            self.say("debug interrupt requested: taken before the next instruction")
            return
        if cmd == "clear" and len(words) == 1:
            s = self.s
            s.debug_write("drseg", EJ.IBS, 0)
            s.debug_write("drseg", EJ.DBS, 0)
            for key in C.WATCHHI:
                s.debug_write("cp0", key, 7)
            s.debug_write("cp0", C.CAUSE, s.machine.cp0_read(*C.CAUSE) & ~C.CA_WP)
            self.say("IBS.BS, DBS.BS, WatchHi I / R / W and Cause.WP cleared")
            return
        if cmd in ("ib", "db", "watch") and len(words) >= 3:
            self._ejtag_set(cmd, words[1:])
            for line in self.s.breakpoint_lines():
                self.say(line)
            return
        raise CliError("ejtag [dint | ib N LOC ... | db N LOC ... | watch N LOC ... | "
                       "ib|db|watch N off | clear] (help ejtag)")

    def do_probe(self, arg: str) -> None:
        """probe [on [trap] [boot] | off | load FILE | break | fastdata WORD... |
        profile [on|off]]   an EJTAG probe on the cores' TAPs (the debugger as
        one): it serves dmseg (0xff200000-0xff2fffff) through processor
        accesses.
          probe on [trap] [boot]   connect it (ECR.ProbEn); trap: the debug
                                   handler comes from its memory at 0xff200200
                                   (ProbTrap); boot: the EJTAGBOOT instruction
                                   (a debug interrupt right after reset)
          probe off                disconnect it (connecting / loading restarts
                                   the program)
          probe load FILE          assemble FILE and put what lies in dmseg
                                   (.ktext 0xff200200 ...) in its memory
          probe break              EjtagBrk: a debug interrupt through the ECR
          probe fastdata W ...     words for FASTDATA downloads (0xff200000-0f)
          probe profile [on|off]   read the PC samples after every step / show
                                   where they fell
          probe mem ADDR [WORDS]   the probe's EJTAG memory (dmseg address)
          probe rate N             PC samples every 2^(5+N) cycles (DCR.PCR, 0-7)
        Without arguments: the TAP, the ECR, the probe's last accesses."""
        self._need_program()
        s = self.s
        words = arg.split()
        if not words:
            for line in s.probe_lines():
                self.say(line)
            return
        cmd = words[0]
        if cmd == "on":
            extra = set(words[1:])
            if extra - {"trap", "boot"}:
                raise CliError("probe on [trap] [boot]")
            s.set_probe(True, trap="trap" in extra, boot="boot" in extra)
            self.say("probe connected; the program restarted")
        elif cmd == "off" and len(words) == 1:
            s.set_probe(False)
            self.say("probe disconnected; the program restarted")
        elif cmd == "load" and len(words) == 2:
            if s.probe is None:
                raise CliError("no probe: probe on first")
            try:
                text = Path(words[1]).read_text(encoding="utf-8")
                n = s.probe_load(text)
            except OSError as exc:
                raise CliError(f"cannot read {words[1]}: {exc.strerror or exc}") from None
            except (ValueError, AsmError) as exc:
                raise CliError(str(exc)) from None
            self.say(f"{n} bytes in the probe's EJTAG memory; the program restarted")
        elif cmd == "break" and len(words) == 1:
            if s.probe is None:
                raise CliError("no probe: probe on first")
            s.probe_event("ejtagbrk")
            self.say("EjtagBrk: a debug interrupt before the next instruction")
        elif cmd == "fastdata" and len(words) > 1:
            if s.probe is None:
                raise CliError("no probe: probe on first")
            try:
                values = [int(w, 0) & 0xFFFFFFFF for w in words[1:]]
            except ValueError:
                raise CliError("fastdata takes words") from None
            s.probe_event("fastdata", None, values)
            self.say(f"{len(values)} word{'s' if len(values) > 1 else ''} queued for FASTDATA")
        elif cmd == "rate" and len(words) == 2:
            try:
                n = int(words[1], 0)
            except ValueError:
                n = -1
            if not 0 <= n <= 7:
                raise CliError("probe rate 0-7 (every 32 ... 4096 cycles)")
            dcr = s.machine.ejtag.dcr
            s.debug_write("drseg", 0, (dcr & ~(7 << EJ.DCR_PCR_SHIFT)) | (n << EJ.DCR_PCR_SHIFT))
            self.say(f"PC sampling every {s.machine.ejtag.sample_period} cycles")
        elif cmd == "mem" and len(words) in (2, 3):
            if s.probe is None:
                raise CliError("no probe: probe on first")
            try:
                addr = int(words[1], 0) & ~3
                n = int(words[2], 0) if len(words) == 3 else 4
            except ValueError:
                raise CliError("probe mem ADDR [WORDS]") from None
            if not TAPM.DMSEG <= addr < TAPM.DMSEG + TAPM.DMSEG_SIZE:
                raise CliError("dmseg is 0xff200000-0xff2fffff")
            order = "big" if s.machine.big_endian else "little"
            for row in range(0, n, 4):
                a = addr + 4 * row
                ws = [int.from_bytes(bytes(s.probe.mem.get(a + 4 * i + k - TAPM.DMSEG, 0)
                                           for k in range(4)), order)
                      for i in range(min(4, n - row))]
                self.say(f"0x{a:08x}  " + " ".join(f"{w:08x}" for w in ws))
        elif cmd == "profile" and len(words) <= 2:
            if s.probe is None:
                raise CliError("no probe: probe on first")
            if len(words) == 2:
                if words[1] not in ("on", "off"):
                    raise CliError("probe profile [on|off]")
                s.probe.profile = words[1] == "on"
                self.say(f"PC sample collection {words[1]}")
                return
            rows = s.profile_rows()
            if not rows:
                self.say("no PC samples collected (probe profile on, then run)")
            total = len(s.probe.samples)
            for n, pc, where in rows:
                self.say(f"{n:6} {100 * n / total:5.1f}%  0x{pc:08x}  {where}")
        else:
            raise CliError("probe [on [trap] [boot] | off | load FILE | break | "
                           "fastdata WORD... | profile [on|off] | mem ADDR] (help probe)")

    def do_board(self, arg: str) -> None:
        """board   the board (--board / set board): flat, or the MIPS Malta —
        its SDRAM, LEDs and ASCII display, the GT-64120's interrupt cause and
        error registers, and its address decode (what answers where)."""
        self._need_program()
        if arg.strip():
            raise CliError("board (no arguments; set board flat|malta[:RAM_MB[:SWITCHES]])")
        for line in self.s.board_lines():
            self.say(line)

    def do_disk(self, arg: str) -> None:
        """disk [save SLOT [FILE] | show SLOT LBA [N]]   the Malta's IDE disks
        (--disk / set disks): each one's state, registers and last commands;
          disk save SLOT [FILE]   write the disk as the program left it (default:
                                  its image file; the next load starts from it)
          disk show SLOT LBA [N]  hex dump of N sectors (default 1) from LBA"""
        self._need_program()
        parts = arg.split()
        try:
            if not parts:
                for line in self.s.disk_lines():
                    self.say(line)
            elif parts[0] == "save" and len(parts) in (2, 3):
                self.say(self.s.save_disk(parts[1], parts[2] if len(parts) == 3 else None))
            elif parts[0] == "show" and len(parts) in (3, 4):
                for line in self.s.disk_sector(parts[1], int(parts[2], 0),
                                               int(parts[3], 0) if len(parts) == 4 else 1):
                    self.say(line)
            else:
                raise CliError("disk [save SLOT [FILE] | show SLOT LBA [N]] (help disk)")
        except (ValueError, OSError) as exc:
            raise CliError(str(exc)) from None

    def do_trace(self, arg: str) -> None:
        """trace [on [OPTIONS] | off | show [N] | words [N] | formats [N] |
        trigger N SOURCE TYPE [tf6 INFO] [once] [out] | trigger N off | in]
        MIPS Trace (PDtrace + the TCB, 24K 11.7-11.13), set up through the
        selected core's TAP as a trace probe does:
          trace on [OPTIONS]       trace into the on-chip trace memory; OPTIONS:
                                   modes U K S E D (default all), asid N (else
                                   any ASID), la sa ld sd / addresses / data /
                                   all (what besides the PC), ca (cycle-accurate:
                                   stall cycles too), from (Trace-From: stop when
                                   full; default Trace-To: keep the newest), tb
                                   (every branch's PC), io (stall rather than
                                   overflow), sync N (a full PC every 2^(5+N)
                                   cycles), tlsif (call / I-miss / D-miss bits),
                                   keep (do not empty the trace memory), wait
                                   (record only after a start trigger)
          trace off                stop (the last trace word is written)
          trace show [N]           the regenerated instructions (the last N, 20)
          trace words [N]          the trace words (64 bits: Type + Trace)
          trace formats [N]        the trace formats TF1-TF6 in them
          trace trigger N SRC TYPE TCBTRIGn: SRC dm (Debug Mode entry) | probe
                                   (TR_TRIGIN), TYPE start | end | about | info;
                                   tf6 INFO: a TF6 into the trace, once: fire one
                                   time, out: strobe the probe's TR_TRIGOUT
          trace trigger N off      clear it
          trace in                 the probe drives TR_TRIGIN (a trigger input)
        Without arguments: the settings, the TCB and its triggers."""
        self._need_program()
        s = self.s
        words = arg.split()
        if not words:
            for line in s.trace_state():
                self.say(line)
            return
        cmd, rest = words[0], words[1:]
        if cmd == "on":
            modes, mode, opts, asid, sync = "", PDT.MODE_PC, set(), None, 0
            it = iter(rest)
            for w in it:
                lw = w.lower()
                if lw in ("u", "k", "s", "e", "d"):
                    modes += lw.upper()
                elif lw in ("la", "sa", "ld", "sd"):
                    mode |= {"la": PDT.MODE_LA, "sa": PDT.MODE_SA, "ld": PDT.MODE_LD,
                             "sd": PDT.MODE_SD}[lw]
                elif lw == "addresses":
                    mode |= PDT.MODE_LA | PDT.MODE_SA
                elif lw == "data":
                    mode |= PDT.MODE_LD | PDT.MODE_SD
                elif lw == "all":
                    mode |= 0x1F
                elif lw in ("asid", "sync"):
                    try:
                        v = int(next(it), 0)
                    except (StopIteration, ValueError):
                        raise CliError(f"trace on {lw} N") from None
                    if lw == "asid":
                        asid = v & 0xFF
                    elif not 0 <= v <= 7:
                        raise CliError("trace on sync 0-7 (every 32 ... 4096 cycles)")
                    else:
                        sync = v
                elif lw in ("ca", "from", "to", "tb", "io", "tlsif", "keep", "wait"):
                    opts.add(lw)
                else:
                    raise CliError(f"trace on: unknown option {w} (help trace)")
            try:
                s.trace_start(modes=modes or "UKSED", mode=mode, ca="ca" in opts,
                              trace_from="from" in opts, asid=asid, tb="tb" in opts,
                              io="io" in opts, sync=sync, tlsif="tlsif" in opts,
                              keep="keep" in opts, wait="wait" in opts)
            except ValueError as exc:               # (nanoMIPS: no MIPS Trace)
                raise CliError(str(exc)) from None
            for line in s.trace_state()[:2]:
                self.say(line)
        elif cmd == "off" and not rest:
            s.trace_stop()
            self.say(s.trace_state()[1])
        elif cmd in ("show", "words", "formats") and len(rest) <= 1:
            try:
                n = int(rest[0], 0) if rest else 20
            except ValueError:
                raise CliError(f"trace {cmd} [N]") from None
            if cmd == "show":
                lines = s.trace_lines(n)
                for line in lines or ["   nothing traced (trace on, then run)"]:
                    self.say(line)
                return
            if cmd == "words":
                ws = s.trace_words()
                for i, w in list(enumerate(ws))[-n:]:
                    self.say(f"   {i:4}  0x{w:016x}  Type {w & 0xF}")
                if not ws:
                    self.say("   the trace memory is empty")
                return
            for f in s.trace_formats()[-n:]:
                self.say("   " + PDT.format_text(f))
        elif cmd == "in" and not rest:
            s.probe_event("trigger-in")
            self.say("TR_TRIGIN: the triggers on the probe's input fire this cycle")
        elif cmd == "trigger" and len(rest) >= 2:
            try:
                n = int(rest[0], 0)
            except ValueError:
                n = -1
            if not 0 <= n < PDT.TRIGGERS:
                raise CliError(f"trace trigger 0-{PDT.TRIGGERS - 1} ...")
            if rest[1] == "off" and len(rest) == 2:
                s.trace_trigger(n, 0)
                self.say(f"trigger {n} off")
                return
            if len(rest) < 3 or rest[1] not in ("dm", "probe") or \
                    rest[2] not in ("start", "end", "about", "info"):
                raise CliError("trace trigger N dm|probe start|end|about|info "
                               "[tf6 INFO] [once] [out]")
            v = (PDT.TR_DM if rest[1] == "dm" else PDT.TR_PDTRI) | \
                (("start", "end", "about", "info").index(rest[2]) << PDT.TR_TYPE_SHIFT)
            it = iter(rest[3:])
            for w in it:
                if w == "tf6":
                    try:
                        v |= PDT.TR_TRACE | ((int(next(it), 0) & 0xFF) << 24)
                    except (StopIteration, ValueError):
                        raise CliError("trace trigger ... tf6 INFO") from None
                elif w == "once":
                    v |= PDT.TR_FO
                elif w == "out":
                    v |= PDT.TR_PDTRO
                else:
                    raise CliError(f"trace trigger: unknown option {w}")
            s.trace_trigger(n, v)
            for line in s.trace_state()[3:]:
                if line.startswith(f"   trigger {n}:"):
                    self.say(line.strip())
        else:
            raise CliError("trace [on [OPTIONS] | off | show [N] | words [N] | formats [N] | "
                           "trigger N ... | in] (help trace)")

    def do_tap(self, arg: str) -> None:
        """tap ir CODE | tap dr VALUE [BITS] | tap reset   scan the selected core's
        EJTAG TAP by hand, bit by bit through its states (IEEE 1149.1):
          tap ir 0x0a              shift an instruction (IDCODE 1, IMPCODE 3,
                                   ADDRESS 8, DATA 9, CONTROL 0xa, ALL 0xb,
                                   EJTAGBOOT 0xc, NORMALBOOT 0xd, FASTDATA 0xe,
                                   TCBCONTROLA-C 0x10 / 0x11 / 0x13, TCBDATA
                                   0x12, PCSAMPLE 0x14, BYPASS 0x1f; or the name)
          tap dr 0x8000 [32]       shift VALUE through the selected data register
                                   (its own length by default): prints what came
                                   out (the captured value)
          tap reset                five TCKs with TMS = 1: Test-Logic-Reset"""
        self._need_program()
        s = self.s
        words = arg.split()
        tap = s.machine.ejtag.tap
        if tap is None or not words:
            raise CliError("tap ir CODE | tap dr VALUE [BITS] | tap reset (help tap)")
        if words[0] == "reset" and len(words) == 1:
            s.probe_event("tap-reset")
            self.say(f"TAP: {tap.state}, IR IDCODE")
            return
        names = {v: k for k, v in TAPM.INSTRUCTIONS.items()}
        try:
            if words[0] == "ir" and len(words) == 2:
                code = names.get(words[1].upper())
                code = int(words[1], 0) if code is None else code
                out = s.probe_event("tap-ir", None, code & 0x1F)
                self.say(f"IR <- 0x{code & 0x1F:02x} "
                         f"({TAPM.INSTRUCTIONS.get(code & 0x1F, 'BYPASS')}); out 0b{out:05b}")
                return
            if words[0] == "dr" and len(words) in (2, 3):
                value = int(words[1], 0)
                bits = int(words[2], 0) if len(words) == 3 else tap._dr_length()
                out = s.probe_event("tap-dr", bits, value)
                self.say(f"DR ({TAPM.INSTRUCTIONS.get(tap.ir, 'BYPASS')}, {bits} bits): "
                         f"out 0x{out:0{(bits + 3) // 4}x}, in 0x{value:x}")
                return
        except ValueError:
            raise CliError("tap: numbers like 0x0a") from None
        raise CliError("tap ir CODE | tap dr VALUE [BITS] | tap reset (help tap)")

    def _ejtag_set(self, kind: str, words: list[str]) -> None:
        s = self.s
        count = {"ib": EJ.IB_COUNT, "db": EJ.DB_COUNT, "watch": C.WATCH_PAIRS}[kind]
        try:
            n = int(words[0])
        except ValueError:
            n = -1
        if not 0 <= n < count:
            raise CliError(f"{kind} takes a number 0-{count - 1}")
        base = (0x1100 if kind == "ib" else 0x2100) + 0x100 * n
        if words[1] == "off":
            if kind == "ib":
                s.debug_write("drseg", base + EJ.IBC, 0)
            elif kind == "db":
                s.debug_write("drseg", base + EJ.DBC, 0)
            else:
                s.debug_write("cp0", C.WATCHLO[n], 0)
            return
        addr = self.resolve(words[1])
        opts: dict[str, int | bool] = {}
        rest = words[2:]
        while rest:
            w = rest.pop(0)
            if w in ("mask", "value", "asid", "tc"):
                if not rest:
                    raise CliError(f"{w} needs a number")
                try:
                    opts[w] = int(rest.pop(0), 0) & 0xFFFFFFFF
                except ValueError:
                    raise CliError(f"{w} needs a number") from None
            elif w in ("trigger", "load", "store", "i", "r", "w", "rw"):
                opts[w] = True
            else:
                raise CliError(f"unknown option {w!r} (help ejtag)")
        asid = opts.get("asid")
        if asid is not None and kind != "watch" and not s.machine.ejtag.asid_sup:
            raise CliError("ASID compares need a TLB (IBS / DBS.ASIDsup = 0 with this MMU)")
        tc = opts.get("tc")
        if tc is not None and (kind == "watch" or s.machine.mt is None):
            raise CliError("tc T: a hardware breakpoint of a core with the MT Module (TCuse)")
        enable = EJ.BC_TE if opts.get("trigger") else EJ.BC_BE
        use = (EJ.BC_ASIDUSE if asid is not None else 0) | (
            EJ.BC_TCUSE | (tc & 0xFF) << EJ.BC_TC_SHIFT if tc is not None else 0)
        if kind == "ib":
            s.debug_write("drseg", base + EJ.IBA, addr)
            s.debug_write("drseg", base + EJ.IBM, opts.get("mask", 0))
            s.debug_write("drseg", base + EJ.IBASID, asid or 0)
            s.debug_write("drseg", base + EJ.IBC, enable | use)
        elif kind == "db":
            no = 0
            if opts.get("load") and not opts.get("store"):
                no = EJ.DBC_NOSB
            elif opts.get("store") and not opts.get("load"):
                no = EJ.DBC_NOLB
            value = opts.get("value")
            blm = 0 if value is not None else 0xFF       # 0xFF: no value compare
            s.debug_write("drseg", base + EJ.DBA, addr)
            s.debug_write("drseg", base + EJ.DBM, opts.get("mask", 0))
            s.debug_write("drseg", base + EJ.DBASID, asid or 0)
            s.debug_write("drseg", base + EJ.DBV, value or 0)
            s.debug_write("drseg", base + EJ.DBC, (blm << EJ.DBC_BLM_SHIFT) | no | enable | use)
        else:
            fetch = n < 2
            kinds = (C.WATCH_I if fetch else
                     (C.WATCH_R if opts.get("r") or opts.get("rw") else 0)
                     | (C.WATCH_W if opts.get("w") or opts.get("rw") else 0))
            if fetch and (opts.get("r") or opts.get("w") or opts.get("rw")) or \
                    not fetch and (opts.get("i") or not kinds):
                raise CliError("watch 0 and 1 watch instruction fetches (i), 2 and 3 "
                               "loads / stores (r, w, rw)")
            hi = (opts.get("mask", 0) & 0xFF8) | (C.WATCHHI_G if asid is None
                                                    else (asid & 0xFF) << 16)
            s.debug_write("cp0", C.WATCHHI[n], hi | 7)  # (I / R / W: write 1 = clear)
            s.debug_write("cp0", C.WATCHLO[n], (addr & ~7) | kinds)

    def do_cp0(self, arg: str) -> None:
        """cp0   PC, HI, LO and the CP0 registers (Status, Cause, EPC ...), FCSR"""
        self._need_program()
        for name, value in self.s.special_registers():
            self.say(f"{name:>9} = 0x{value:0{self.s.hex_digits(name)}x}")

    def do_csr(self, arg: str) -> None:
        """csr [NAME|NUMBER ...]   RISC-V: the hart's CSRs (default: the trap
        and status ones — mstatus, mtvec, mepc, mcause ...)"""
        self._need_program()
        s, m = self.s, self.s.machine
        if not m.rv:
            raise CliError("csr is a RISC-V command (MIPS: cp0)")
        if not arg.strip():
            for name, value in s.special_registers():
                self.say(f"{name:>9} = 0x{value:0{s.hex_digits(name)}x}")
            return
        from . import rvcsr
        by_name = {v: k for k, v in rvcsr.NAMES.items()}
        for tok in arg.split():
            num = by_name.get(tok.lower())
            if num is None:
                try:
                    num = int(tok, 0)
                except ValueError:
                    raise CliError(f"no CSR {tok!r}") from None
            try:
                value = m.csrs.read(num)
            except Exception as exc:                        # noqa: BLE001
                raise CliError(f"CSR 0x{num:03x}: {exc}") from None
            name = rvcsr.NAMES.get(num, f"0x{num:03x}")
            self.say(f"{name:>9} = 0x{value:0{s.hex_digits()}x}")

    def do_fpu(self, arg: str) -> None:
        """fpu   the 32 FPU registers (as single / double) and FCSR (RISC-V: f0-f31,
        fcsr's frm / fflags)"""
        self._need_program()
        m = self.s.machine
        if m.rv:
            if not m.flen:
                self.say("  this hart has no F / D extension")
                return
            from .riscv import FPR_NAMES
            for i, raw in enumerate(m.fpr):
                single = struct.unpack("<f", (raw & M32).to_bytes(4, "little"))[0]
                boxed = m.flen == 32 or raw >> 32 == M32
                dbl = struct.unpack("<d", raw.to_bytes(8, "little"))[0] if m.flen == 64 else None
                self.say(f"  {FPR_NAMES[i]:<4} 0x{raw:0{m.flen // 4}x}  "
                         + (f"{single:<14.7g}" if boxed else " " * 14)
                         + (f" {dbl:.15g}" if dbl is not None else ""))
            self.say(f"  fcsr 0x{m.fcsr:08x}  frm {m.fcsr >> 5 & 7}  fflags "
                     f"0b{m.fcsr & 31:05b} (NV DZ OF UF NX)")
            return
        if not m.has_fpu:
            self.say("  this core has no FPU (Config1.FP = 0): Status.CU1 reads 0, every "
                     "FPU instruction is Coprocessor Unusable")
            return
        for i, raw in enumerate(m.fpr):
            single = struct.unpack(">f", (raw & M32).to_bytes(4, "big"))[0]
            dbl = struct.unpack(">d", raw.to_bytes(8, "big"))[0]
            self.say(f"  $f{i:<2} 0x{raw:016x}  {single:<14.7g} {dbl:.15g}")
        self.say(f"  fcsr 0x{m.fcsr:08x}")
        if m.msa:
            self.say("  (MSA: $fN is bits 63..0 of the vector register $wN — `msa` shows "
                     "all 128 bits)")

    def do_dsp(self, arg: str) -> None:
        """dsp   the DSP Module: its revision, Status.MX, DSPControl (pos,
        scount, c, EFI, ccond, ouflag) and the accumulators ac0-ac3"""
        self._need_program()
        for line in self.s.dsp_lines():
            self.say(f"  {line}")

    def do_msa(self, arg: str) -> None:
        """msa [$wN ...] [b|h|w|d|w.f|d.f]   MSA: Config5.MSAEn / Status.FR,
        MSAIR, MSACSR (RM, Flags, Enables, Cause, NX, FS), the partitioning
        registers, and the vector registers (every non-zero one, or those
        named) as 128-bit values or element by element in the format given
        (element 0 first; w.f / d.f as floating point)"""
        self._need_program()
        regs, fmt = [], ""
        for tok in arg.split():
            t = tok.lower()
            if t in Session.MSA_FORMATS:
                fmt = t
                continue
            n = t.lstrip("$").lstrip("w")
            if not t.startswith(("$w", "w")) or not n.isdigit() or int(n) > 31:
                raise CliError(f"msa: $w0 ... $w31 or a format (b h w d w.f d.f), not {tok}")
            regs.append(int(n))
        for line in self.s.msa_lines(regs or None, fmt):
            self.say(f"  {line}")

    def do_mt(self, arg: str) -> None:
        """mt [yq BITS | sys N | itc]   the MT Module: MVPControl, each VPE
        (VPEControl, VPEConf0), the policy manager and each thread context
        (state, PC, DA / IXMT / DT, group; > issues next); yq BITS sets the
        YIELD qualifier inputs (the 34K's 16 pins) a blocked YIELD waits
        for; sys N the policy manager's PM_sys_avail input (0-7, WRR2's
        throttles); itc lists the ITC cells (gating storage)"""
        self._need_program()
        parts = arg.split()
        if parts and parts[0] == "yq":
            if len(parts) != 2:
                raise CliError("mt yq BITS, e.g. mt yq 0x1")
            self.s.yield_inputs(int(parts[1], 0))
        elif parts and parts[0] == "sys":
            if len(parts) != 2 or not 0 <= int(parts[1], 0) <= 7:
                raise CliError("mt sys N (0-7: free entries of the system buffer)")
            self.s.pm_input(int(parts[1], 0))
        elif parts == ["itc"]:
            for line in self.s.itc_lines():
                self.say(f"  {line}")
            return
        elif parts:
            raise CliError("mt [yq BITS | sys N | itc]")
        for line in self.s.mt_lines():
            self.say(f"  {line}")

    def do_vp(self, arg: str) -> None:
        """vp [itu]   Release 6 virtual processors (a CM3 system: --vps N, the
        I6400): per core VP_RUNNING, the DVP that holds the others, SAAR, and
        every VP (GlobalNumber, VPControl, PC); itu lists the selected VP's
        core's ITU cells (gating storage)"""
        self._need_program()
        parts = arg.split()
        if parts == ["itu"]:
            lines = self.s.itu_lines()
        elif parts:
            raise CliError("vp [itu]")
        else:
            lines = self.s.vp_lines()
        for line in lines:
            self.say(f"  {line}")

    def do_tc(self, arg: str) -> None:
        """tc N   thread context N: its TC registers and its GPRs (MFTR's view)"""
        self._need_program()
        if not arg.strip():
            raise CliError("tc N")
        try:
            lines = self.s.tc_lines(int(arg, 0))
        except ValueError as exc:
            raise CliError(str(exc)) from None
        for line in lines:
            self.say(f"  {line}")

    def do_mem(self, arg: str) -> None:
        """mem [-p] LOCATION [WORDS]   (x) memory as words + text, from
        LOCATION (default 16 words).  -p: LOCATION is a physical address
        and the dump shows physical memory (see translate)"""
        self._need_program()
        parts = arg.split()
        physical = bool(parts) and parts[0] == "-p"
        if physical:
            parts = parts[1:]
        if not parts:
            raise CliError("mem where? e.g. mem 0x10010000, mem nums 8, mem $sp, "
                           "mem -p 0x50010000")
        start = self.resolve(parts[0]) & ~3
        words = int(parts[1], 0) if len(parts) > 1 else 16
        rows = (words + 3) // 4
        for addr, ws in self.s.memory_words(start, rows, physical=physical):
            text = "".join(chr(b) if 32 <= b < 127 else "."
                           for w in ws for b in (w or 0).to_bytes(4, "big"))
            self.say(f"0x{addr:08x}  " + " ".join("--------" if w is None else f"{w:08x}"
                                                for w in ws) + f"  {text}")

    do_x = do_mem

    def do_translate(self, arg: str) -> None:
        """translate LOCATION   (t) the physical address of a virtual one,
        as the MMU translates it now: segment, rule, cacheability"""
        self._need_program()
        if not arg.strip():
            raise CliError("translate what? e.g. translate 0x10010000, translate $sp")
        from .mmu import cca_name
        t = self.s.machine.translate(self.resolve(arg.strip()))
        if t.pa is None:
            what = ("an Address Error" if t.fault == "address error"
                    else f"a TLB {t.fault}")
            self.say(f"0x{t.va:08x} {t.segment} -> no physical address  ({t.how}: "
                     f"an access would take {what} exception)")
            return
        self.say(f"0x{t.va:08x} {t.segment} -> physical 0x{t.pa:08x}  ({t.how}; "
                 f"attribute {t.cca} = {cca_name(t.cca)})")

    do_t = do_translate

    def do_cache(self, arg: str) -> None:
        """cache [WHAT]   the caches (Settings > Caches / --caches):
          cache               every cache: shape, valid lines, hit rate
          cache stats [reset] hits / misses per access kind, the 3 C's
                              (compulsory, capacity, conflict) + invalidated,
                              evictions, the write buffer; reset: start again
          cache last          the cache accesses of the last step
          cache I|D|L2|L3     the sets holding valid lines
          cache D set N       one set: ways, tags, LRU age, data
          cache D LOC         how LOC is cut (tag | index | offset) + its set
          cache LOC           where the line of LOC is in every cache
          cache wb            the write buffer
        Nothing here touches the caches (no LRU update, no counting)."""
        from . import cacheview as cv
        self._need_program()
        s = self.s
        caches = s.machine.caches
        if caches is None or not caches.all:
            raise CliError("no caches: start with --caches (or --icache / --dcache / "
                           "--l2), or set dcache 32K")
        parts = arg.split()
        if not parts:
            for c in caches.all:
                st = c.stats
                valid = sum(ln.valid for row in c.lines for ln in row)
                self.say(f"{cv.short_name(c)}  {c.geometry.describe()}: {valid} lines "
                         f"valid; {st.accesses} accesses, hits {cv.pct(st.hits, st.accesses)}")
            if caches.wbb is not None:
                self.say(cv.wbb_lines(caches.wbb)[0])
            return
        what = parts[0].lower()
        if what == "stats":
            if parts[1:] == ["reset"]:
                caches.reset_stats()
                self.say("cache statistics reset")
                return
            self.say(cv.format_stats(caches))
            return
        if what == "last":
            acc = s.cache_accesses()
            if not acc:
                self.say("no cache access in the last step")
            for where, a in acc:
                self.say(f"{where:<13} {a.describe()}")
            return
        if what in ("wb", "wbb"):
            if caches.wbb is None:
                raise CliError("no write buffer (--write-buffer)")
            self.say("\n".join(cv.wbb_lines(caches.wbb)))
            return
        c = cv.cache_named(caches, what) if what in cv.CACHE_KEYS else None
        if c is None and what in cv.CACHE_KEYS:
            raise CliError(f"there is no {parts[0]} cache ({caches.describe()})")
        if c is None:
            self.say(cv.locate(s.machine, caches, self.resolve(parts[0])))
            return
        if len(parts) == 1:
            self.say(cv.format_cache(c))
        elif parts[1].lower() == "set":
            if len(parts) < 3:
                raise CliError(f"which set? 0 .. {c.sets - 1}")
            n = int(parts[2], 0)
            if not 0 <= n < c.sets:
                raise CliError(f"set {n}: {c.name} has sets 0 .. {c.sets - 1}")
            self.say(cv.format_set(c, n))
        else:
            va = self.resolve(parts[1])
            pa = s.machine.translate(va).pa
            if pa is None:
                raise CliError(f"0x{va & 0xFFFFFFFF:08x}: no TLB entry maps it now")
            self.say(cv.format_split(c, va, pa))
            self.say(cv.format_set(c, cv.split(c, va, pa).index))

    def do_cm(self, arg: str) -> None:
        """cm [LOC]   the Coherence Manager of a multi-core system (--cores N):
          cm        the coherence domain (GCR_Cx_COHERENCE of every core),
                    what the CM did (requests, interventions, cache-to-cache
                    transfers, invalidations, upgrades, writebacks,
                    globalized cache operations), the shared L2, its error
          cm LOC    the MESI state of LOC's line in every core's data cache"""
        self._need_program()
        s = self.s
        if s.cps is None:
            raise CliError("one core: no Coherence Manager (set cores 2..6)")
        cm = s.cps.cm
        if not arg.strip():
            self.say("\n".join(cm.describe()))
            return
        va = self.resolve(arg.strip())
        xl = s.machine.translate(va)
        if xl.pa is None:
            raise CliError(f"0x{va & 0xFFFFFFFF:08x}: no TLB entry maps it now")
        from .cacheview import MESI_NAMES
        copies = cm.copies(xl.pa)
        self.say(f"0x{va & 0xFFFFFFFF:08x}: PA 0x{xl.pa:08x}, attribute {xl.cca}: "
                 + (", ".join(f"core {j} {MESI_NAMES[st]}" for j, st in copies)
                    if copies else "in no data cache"))

    def do_cps(self, arg: str) -> None:
        """cps   the registers of a multi-core system (--cores N): the
        Coherence Manager's GCRs (global, each core's block), the Cluster
        Power Controller, the GIC — read without side effects"""
        from .cpsview import format_cps
        self._need_program()
        if self.s.cps is None:
            raise CliError("one core: no CPS registers (set cores 2..6)")
        self.say(format_cps(self.s))

    def do_gic(self, arg: str) -> None:
        """gic   the Global Interrupt Controller of a multi-core system
        (--cores N): enabled or not, its counter, every enabled source
        (trigger, polarity, pending, where it goes), every core's local
        interrupts, pins and GIC COMPARE"""
        self._need_program()
        s = self.s
        if s.cps is None:
            raise CliError("one core: no GIC (set cores 2..6)")
        self.say("\n".join(s.cps.gic.describe()))

    def do_tlb(self, arg: str) -> None:
        """tlb [WHAT]   the MMU's entries (--mmu / set mmu tlb | ftlb | bat):
          tlb                 the entries in use, Wired, Random, the ASID
          tlb all             every entry (also never written / invalidated)
          tlb entry N         entry N: tag, both pages, what TLBR would read
          tlb set N           (FTLB) the ways of FTLB set N
          tlb micro           the ITLB / DTLB copies and their counters
          tlb stats [reset]   lookups, hits / misses, exceptions, page walks
          tlb last            the translations of the last step
          tlb LOC             how LOC translates now: VPN2 | even/odd |
                              offset -> entry -> PFN -> PA
        With the I7200's MPU (--mmu mpu:N | fixed+mpu:N; also `mpu`): `tlb`
        lists CDMMBase, MPU_Config, the segments and the enabled regions,
        `tlb LOC` the region or segment of LOC and what it inhibits.
        Nothing here touches the TLB (no micro-TLB refill, no counting)."""
        from . import tlbview as tv
        self._need_program()
        s = self.s
        m = s.machine
        parts = arg.split()
        what = parts[0].lower() if parts else ""
        if what == "stats":
            if parts[1:] == ["reset"]:
                tv.reset_stats(m)
                self.say("TLB statistics reset")
                return
            self.say(tv.format_stats(m))
            return
        if what == "last":
            acc = s.tlb_accesses()
            if not acc:
                self.say("no translation through the TLB in the last step"
                         + ("" if tv.kind(m) != "fixed" else f" ({tv.no_tlb(m)})"))
            for where, a in acc:
                self.say(f"{where:<11} {a.describe()}")
            return
        if what and what not in ("all", "entry", "set", "micro"):
            self.say(tv.explain(m, self.resolve(arg.strip())))
            return
        if tv.kind(m) == "fixed":
            raise CliError(tv.no_tlb(m))
        if tv.kind(m) == "mpu" and what in ("entry", "set"):
            raise CliError("the MPU has no TLB entries: `tlb` lists its segments and regions")
        if what == "micro":
            self.say(tv.format_micro(m))
        elif what == "entry":
            n = self._entry_number(parts, m.mmu)
            if tv.kind(m) == "bat":
                self.say(tv.format_bat(m).splitlines()[n + 1].strip())
            else:
                self.say(tv.format_entry(m, n))
        elif what == "set":
            mmu = m.mmu
            if tv.kind(m) != "tlb" or not mmu.ftlb:
                raise CliError("tlb set N: only an FTLB has sets (--mmu ftlb)")
            if len(parts) < 2 or not parts[1].isdigit():
                raise CliError(f"which set? 0 .. {mmu.fsets - 1}")
            n = int(parts[1], 0)
            if not 0 <= n < mmu.fsets:
                raise CliError(f"set {n}: the FTLB has sets 0 .. {mmu.fsets - 1}")
            self.say(tv.format_ftlb_set(m, n))
        else:
            self.say(tv.format_tlb(m, everything=what == "all"))

    def do_mpu(self, arg: str) -> None:
        """mpu [LOC]   the I7200's MPU (--mmu mpu:N | fixed+mpu:N): CDMMBase,
        MPU_Config, the segments and the enabled regions; with LOC, the
        region or segment of LOC and what it inhibits (as `tlb`)."""
        from . import tlbview as tv
        self._need_program()
        if tv.kind(self.s.machine) != "mpu":
            raise CliError(f"no MPU: this core has the {self.s.machine.mmu.name} "
                           "(--mmu mpu:N on a nanoMIPS core)")
        self.do_tlb(arg)

    @staticmethod
    def _entry_number(parts: list[str], mmu) -> int:
        top = len(mmu.entries) - 1
        if len(parts) < 2:
            raise CliError(f"which entry? 0 .. {top}")
        try:
            n = int(parts[1], 0)
        except ValueError:
            raise CliError(f"tlb entry N: N is a number 0 .. {top}") from None
        if not 0 <= n <= top:
            raise CliError(f"entry {n}: the entries are 0 .. {top}")
        return n

    def do_perf(self, arg: str) -> None:
        """perf [WHAT]   the performance counters (CP0 25, as on a 24K):
          perf                      both counters: count, event, modes
          perf events [N]           the events counter N (0 / 1) can count
          perf N EVENT [MODES] [ie] count EVENT (a name or number) on counter N
                                    in MODES: u user, s supervisor, k kernel,
                                    x exception level (default uk); ie: the
                                    overflow interrupt; the count starts at 0
          perf N off                stop counter N (its mode bits 0)
          perf N = VALUE            write counter N
        Software does the same with mtc0 to $25 (select 0-3)."""
        from . import perf
        self._need_program()
        m = self.s.machine
        parts = arg.split()
        if not parts:
            self.say(perf.format_counters(m))
            return
        if parts[0] == "events":
            n = int(parts[1]) if len(parts) > 1 and parts[1] in ("0", "1") else None
            self.say(perf.format_events(n))
            return
        if parts[0] not in ("0", "1") or len(parts) < 2:
            raise CliError("perf N EVENT [MODES] [ie] with N = 0 or 1 (help perf)")
        i = int(parts[0])
        if parts[1] == "off":
            self.s.configure_counter(i, None)
        elif parts[1] == "=":
            if len(parts) < 3:
                raise CliError("perf N = VALUE")
            try:
                value = int(parts[2], 0)
            except ValueError:
                value = self.resolve(parts[2])          # $reg
            m.cp0_write(*perf.PERFCNT[i], value)
        else:
            where = [n for c, n in perf.find(parts[1]) if c == i]
            if not where:
                other = perf.find(parts[1])
                hint = (f": it is event {other[0][1]} of counter {other[0][0]}" if other
                        else " (perf events)")
                raise CliError(f"counter {i} cannot count {parts[1]!r}{hint}")
            rest = [p.lower() for p in parts[2:]]
            ie = "ie" in rest
            modes = next((p for p in rest if p != "ie"), "uk")
            if set(modes) - set("uskx"):
                raise CliError("modes are letters of u, s, k, x (e.g. uk)")
            self.s.configure_counter(i, where[0], modes, ie)
        self.say(perf.format_counters(m).splitlines()[i])

    def do_core(self, arg: str) -> None:
        """core [buffer|predict|units]   inside the MIPS 24K pipeline model
        (set model 24k, mode pipeline):
          core            all of it
          core buffer     the fetch unit: what it does, the instruction buffer
          core predict    the branch history table, the return stack, the
                          last prediction and resolution
          core units      multiply / divide, FPU, load / store unit, and the
                          CP0 writes still on their way (hazards)"""
        from . import coreview as cv
        self._need_program()
        p = self.s.pipeline
        if not cv.is_24k(p):
            raise CliError("not the 24K pipeline: set model 24k, then mode pipeline")
        what = arg.strip().lower()
        text = {"": cv.format_core, "buffer": cv.format_buffer,
                "predict": cv.format_predictors, "units": cv.format_units}.get(what)
        if text is None:
            raise CliError("core [buffer|predict|units]")
        self.say(text(p))

    def do_cpu(self, arg: str) -> None:
        """cpu [N]   the cores of a multi-core system (set cores N) — or the
        VPEs of the MT Module: their power state (the CPC's SEQSTATE) and
        position; cpu N selects the core / VPE the other commands show
        (regs, cp0, mem, view, chart, ...)"""
        self._need_program()
        s = self.s
        if arg.strip():
            try:
                s.select(int(arg, 0))
            except ValueError as exc:
                raise CliError(str(exc) if "core" in str(exc) else
                               f"not a core number: {arg.strip()}") from None
        if len(s.machines) < 2:
            self.say(f"one core (core 0); set cores 2..{MAX_CORES} for a multi-core system")
            return
        d = s.hex_digits("pc")
        for c in s.cores():
            where = (f", pc 0x{c['pc']:0{d}x}" if c["pc"] is not None else "") + (
                f" (line {c['line']})" if c["line"] else "") + (
                " (exited)" if c["halted"] else " (waiting)" if c["waiting"] else "")
            self.say(("=> " if c["selected"] else "   ") + f"{s.cpu_name(c['cpu'])}: "
                     f"{c['state']}{where}")

    def do_dis(self, arg: str) -> None:
        """dis [LOCATION] [COUNT]   disassembly (default: around the current
        instruction); => marks it, * marks breakpoints"""
        self._need_program()
        s = self.s
        parts = arg.split()
        cur = s.current_pc()
        rows = s.disassembly()
        if not rows:
            return
        addrs = [a for a, *_ in rows]
        count = int(parts[1], 0) if len(parts) > 1 else 12
        center = self.resolve(parts[0]) if parts else cur
        i = addrs.index(center) if center in addrs else 0
        lo = max(0, i - (count // 3 if not parts else 0))
        prog = s.program
        for a, w, text, line in rows[lo:lo + count]:
            mark = "=>" if a == cur else " *" if a in s.breakpoints else "  "
            enc = encoding(w, prog.size_at(a), prog.arch) \
                if prog.arch in ("nanomips", "rv32", "rv64") else f"{w:08x}"
            self.say(f"{mark} 0x{a:08x}  {enc}  {text:<30} {line:>4}")

    def do_list(self, arg: str) -> None:
        """list [LINE|LOCATION]   (l) source lines around the current line (a
        compiled program: its C file — list 17, list file.c:17, list main)"""
        self._need_program()
        s = self.s
        if s.program.c_lines:
            self._list_c(arg.strip())
            return
        src = (s.source if s.kind == "asm" else s.listing).splitlines()
        cur = s.current_line()
        if arg.strip():
            t = arg.strip()
            center = int(t) if t.isdigit() else \
                s.program.source_map.get(self.resolve(t), None)
            center = center if isinstance(center, int) else (center.line if center else 1)
        else:
            center = cur or 1
        bp_lines = {s.program.source_map[a].line for a in s.breakpoints
                    if a in s.program.source_map}
        for n in range(max(1, center - 5), min(len(src), center + 5) + 1):
            mark = "=>" if n == cur else " *" if n in bp_lines else "  "
            self.say(f"{mark} {n:4}  {src[n - 1]}")

    do_l = do_list

    def _list_c(self, t: str) -> None:
        s = self.s
        cur_pc = s.current_pc()
        cur = s.source_position(cur_pc) if cur_pc is not None else None
        if t:
            addr = self.resolve_all(t)[0]
            pos = s.source_position(addr)
            if pos is None:
                raise CliError(f"no source line for {t}")
            file, center = pos[0], pos[1]
        elif cur is not None:
            file, center = cur[0], cur[1]
        else:
            got = s.source_file()
            if got is None:
                raise CliError("no source file")
            file, center = got[0], 1
        got = s.source_file(next(iter(s.line_addresses(file, 1)), None))
        text = got[1] if got and got[0] == file else []
        bps = {s.program.c_lines.get(a) for a in s.breakpoints}
        if not text:
            self.say(f"({file}: cannot read it)")
            return
        self.say(f"{Path(file).name}:")
        for n in range(max(1, center - 5), min(len(text), center + 5) + 1):
            mark = ("=>" if cur and (cur[0], cur[1]) == (file, n) else
                    " *" if (file, n) in bps else "  ")
            self.say(f"{mark} {n:4}  {text[n - 1]}")

    def do_where(self, arg: str) -> None:
        """where   the current position, steps / cycles, CPI"""
        self._need_program()
        s = self.s
        if s.pipeline is not None:
            cpi = f", CPI {s.pipeline.cycles / s.pipeline.retired:.2f}" \
                if s.pipeline.retired else ""
            n = s.pipeline.retired
            own = f" (core {s.cpu}: {s.pipeline.cycles} since it started)" \
                if s.cps is not None and s.pipeline.cycles != s.steps else ""
            self.say(f"{s.steps} cycles{own}, {n} instruction{'s' if n != 1 else ''} "
                     f"retired{cpi}")
            self.say(self._cycle_line())
        else:
            self.say(f"{s.steps} instructions executed")
        self.say("(ended)" if s.halted else self._where(s.current_pc()))

    def do_view(self, arg: str) -> None:
        """view   what happened inside the last executed instruction: fields,
        ALU, address, branch, effects"""
        self._need_program()
        v = self.s.last_view
        self.say(format_view(v) if v is not None else "nothing executed yet")

    def do_chart(self, arg: str) -> None:
        """chart [CYCLES]   the pipeline chart of the last CYCLES cycles
        (default 24): instructions x cycles, -- = a stall in ID"""
        p = self.s.pipeline
        if p is None:
            raise CliError("not in pipeline mode (mode pipeline)")
        self.say(p.chart(width=self._count(arg) if arg.strip() else 24, last=True))
        if p.stalls:
            self.say("stall cycles: " + ", ".join(f"{k} {v}" for k, v in sorted(p.stalls.items())))

    def do_print(self, arg: str) -> None:
        """print VALUE   (p) a register, label, number or float in every form:
        print $t0, print nums, print 0x41, print 1.5"""
        t = arg.strip()
        if not t:
            raise CliError("print what?")
        as_float = False
        fpr = None
        if self.s.rv and self.s.machine is not None:      # RISC-V: f10, fa0 (a single)
            from .riscv import FPR_NAMES
            name = t.lstrip("$").lower()
            fpr = FPR_NAMES.index(name) if name in FPR_NAMES else \
                int(name[1:]) if name[:1] == "f" and name[1:].isdigit() and int(name[1:]) < 32 \
                else None
        if fpr is not None:
            t, as_float = f"0x{self.s.machine.fpr[fpr] & M32:08x}", True
        elif t.startswith("$f") and t[2:].isdigit() and self.s.machine is not None:
            t, as_float = f"0x{self.s.machine.fpr[int(t[2:]) % 32] & M32:08x}", True
        elif t.startswith("$") or (self.s.program is not None and t in self.s.program.symbols) \
                or (self.s.rv and self.s.machine is not None and self._gpr(t) is not None):
            t = f"0x{self.resolve(t):08x}"
        else:
            try:
                int(t, 0)
            except ValueError:
                as_float = not t.startswith("'")    # 1.5, -2e-3, inf, nan
        try:
            n = tools.convert(t, nan2008=self.s.release == "R6")
        except ValueError as exc:
            raise CliError(str(exc)) from None
        self.say(f"hex {n.hex}   unsigned {n.unsigned}   signed {n.signed}   "
                 f"text {n.chars!r}")
        self.say(f"bin {n.binary}")
        for fp in (n.single, n.double) if as_float or n.width == 64 else ():
            if fp is not None:
                formula = f": {fp.formula}" if fp.kind in ("normal", "subnormal") else ""
                self.say(f"{'single' if fp.width == 32 else 'double'} "
                         f"{tools.fmt_float(fp.value)}   ({fp.kind}{formula})")
        for note in n.notes:
            self.say(f"note: {note}")

    do_p = do_print

    # -- tools ---------------------------------------------------------------------

    def do_decode(self, arg: str) -> None:
        """decode WORD   a 32-bit instruction word: fields and disassembly"""
        self._tool(arg, decode=True)

    def do_encode(self, arg: str) -> None:
        """encode INSTRUCTION   the machine word(s) of an instruction"""
        self._tool(arg, decode=False)

    def _tool(self, arg: str, decode: bool) -> None:
        s = self.s
        arch = s.options.arch
        rv = {"extensions": s.options.rv_extensions} if s.rv else {}
        try:
            if decode and arch == "nanomips":
                raw, size = tools.parse_nano(arg)
                words = [tools.decode_word(raw, size=size, abi=s.abi, arch=arch)]
            elif decode:
                word = tools.parse_rv(arg) if s.rv else tools.parse_word(arg)
                words = [tools.decode_word(word, release=s.release, abi=s.abi, arch=arch,
                                           **rv)]
            else:
                words = tools.encode_text(arg, release=s.release, abi=s.abi,
                                          pseudo=s.options.pseudo, arch=arch, **rv)
        except Exception as exc:                            # noqa: BLE001
            raise CliError(f"{type(exc).__name__}: {exc}") from None
        for d in words:
            if s.rv:
                self.say(f"{d.encoding(arch)}  {d.text}    ({d.description}: {d.format})")
            elif arch == "nanomips":
                size = 2 if d.word < 1 << 16 else 4 if d.word < 1 << 32 else 6
                self.say(f"{encoding(d.word, size, arch).rstrip()}  {d.text}    "
                         f"({d.description}: {d.format})")
            else:
                self.say(f"0x{d.word:08x}  {d.text}    ({d.description}, {d.format}-format)")
            self.say("  " + "  ".join(f"{f.name}[{f.msb}:{f.lsb}]={f.bits} ({f.meaning})"
                                      for f in d.fields))

    def do_ref(self, arg: str) -> None:
        """ref TEXT   the instruction reference: instructions matching TEXT"""
        from .assembler import PSEUDO_HINTS
        word = arg.strip().lower()
        if word in PSEUDO_HINTS:
            self.say(f"{word} is a pseudo-instruction; the assembler writes: "
                     f"{PSEUDO_HINTS[word]}")
        rows = tools.search(tools.reference(self.s.release, arch=self.s.options.arch,
                                            extensions=self.s.options.rv_extensions), arg)
        if not rows and word not in PSEUDO_HINTS:
            self.say("nothing found")
        for r in rows[:40]:
            notes = ([f"since {r.since}"] if not (self.s.nano or self.s.rv) else []) + (
            [f"removed in {r.removed_in}"] if r.removed_in else [])
            notes += (["delay slot"] if r.delay_slot else []) + \
                (["privileged"] if r.privileged else [])
            self.say(f"{r.syntax:<32} {r.description}"
                     + (f"  [{', '.join(notes)}]" if notes else ""))
        if len(rows) > 40:
            self.say(f"... {len(rows) - 40} more")

    # -- console -------------------------------------------------------------------

    def do_input(self, arg: str) -> None:
        """input TEXT   a line for the program's next read_int / read_string ..."""
        self.s.give_input(arg)

    def do_keys(self, arg: str) -> None:
        """keys TEXT   type TEXT on the memory-mapped keyboard (\\n = Enter)"""
        self.s.type_keys(arg.encode().decode("unicode_escape"))

    # -- configuration ---------------------------------------------------------------

    def do_mode(self, arg: str) -> None:
        """mode instruction | pipeline   (restarts the program)"""
        m = arg.strip()
        if m not in ("instruction", "pipeline"):
            raise CliError("mode instruction | mode pipeline")
        self.s.set_mode(m)
        self._printed = 0
        model = self.s.options.model
        self.say(f"mode {m}" + ((", the 24K pipeline" if model == "24k" else
                                 ", forwarding " + ("on" if self.s.forwarding else "off"))
                                if m == "pipeline" else ""))

    def do_set(self, arg: str) -> None:
        """set OPTION VALUE   release R1..R6 | endian big|little | forwarding on|off |
        strict on|off | reorder on|off | boot kernel|user|reset |
        exceptions cp0|stop | syscalls on|off | names abi|numeric |
        icache|dcache|l2|l3 SIZE[:WAYS[:LINE]]|none | write-buffer on|off |
        mmu fixed|bat|tlb[:N]|ftlb[:N[:SxW]][+sc][+pw] | model 5-stage|24k |
        cores 1..6 | shadow-sets 1..16 | board flat|malta[:RAM_MB[:SWITCHES]] |
        disks [SLOT=]IMAGE[,...]|none (the Malta's IDE disks) | iocu on|off |
        compressed auto|none|mips16e|micromips|micromips-only | fpu on|off |
        dsp auto|none|dsp|dspr2 (the DSP Module) | msa auto|none|msa (MSA, SIMD) |
        msa-wrp 0..32 (MSA vector register partitioning: physical registers) |
        mt auto|none|VPES:TCS (the MT Module: 1-3 VPEs, up to 9 TCs) |
        mt-pm rr|wrr|wrr2 (its policy manager) | itc auto|none|FIFOS:SEMAPHORES |
        vps 1..4 (Release 6 VPs a core: a CM3 system) |
        itu auto|none|FIFOS:SINGLES[:DEPTH] (a CM3 core's ITU) |
        core NAME|generic (a real core's settings: python3 -m emips cores) |
        arch mips32|mips64|nanomips|rv32|rv64 (MIPS64: 64-bit registers, the
        doubleword instructions; nanoMIPS: nanoMIPS32 Release 6; RISC-V: a hart) |
        nms on|off | crc auto|none|crc | ginv auto|none|ginvi|ginvt (a nanoMIPS
        core: the nanoMIPS subset, CRC32, global invalidate) | relax on|off
        (nanoMIPS: the linker's relaxations, ld --relax; RISC-V: as -mrelax and
        ld's relaxation, on by default) | rv-ext EXTS | rv-modes msu|mu|m (a
        RISC-V hart's extensions and privilege modes)
        (changes reassemble or restart the program; a setting the chosen core
        is not built with makes it the generic core)"""
        parts = arg.split()
        if len(parts) != 2:
            raise CliError("set OPTION VALUE — see help set")
        key, value = parts
        s, o = self.s, self.s.options
        on = value in ("on", "1", "yes", "true")
        if key == "release" and value.upper() in ("R1", "R2", "R3", "R5", "R6"):
            if s.nano and value.upper() != "R6":
                raise CliError("nanoMIPS is Release 6 based (MD01247 ch. 1): set arch "
                               "mips32 first")
            s.release = value.upper()
        elif key == "endian" and value in ("big", "little"):
            s.endian = value
        elif key == "forwarding":
            s.forwarding = on
        elif key == "strict":
            o.pseudo = not on
        elif key == "reorder":
            o.reorder = on
        elif key == "boot" and value in BOOT_MODES:
            o.boot = value
        elif key == "exceptions" and value in ("cp0", "stop"):
            o.exceptions = value
        elif key == "syscalls":
            o.syscalls = on
        elif key == "names" and value in ("abi", "numeric"):
            s.abi = value == "abi"
            return
        elif key in ("icache", "dcache", "l2", "l3"):
            from .cache import Geometry
            level = 1 if key.endswith("cache") else int(key[1])
            try:
                Geometry.parse(value, level)
            except ValueError as exc:
                raise CliError(str(exc)) from None
            if key == "l3" and value.lower() != "none" and o.l2 == "none":
                raise CliError("an L3 cache needs an L2 cache (set l2 256K first)")
            setattr(o, key, value)
        elif key == "write-buffer":
            o.write_buffer = on
        elif key == "model" and value.lower() in ("5-stage", "24k"):
            o.model = value.lower()
        elif key == "iocu":
            if on and (o.cores < 2 or o.board.split(":")[0] != "malta"):
                raise CliError("an IOCU needs a multi-core Malta: set board malta, set cores 2")
            o.iocu = on
        elif key == "cores" and value.isdigit() and 1 <= int(value) <= MAX_CORES:
            o.cores = int(value)
        elif key == "vps":
            if not (value.isdigit() and 1 <= int(value) <= 4):
                raise CliError("set vps 1..4")
            if int(value) > 1 and self.s.release != "R6":
                raise CliError("virtual processors are Release 6's (set release R6)")
            o.vps = int(value)
        elif key == "itu":
            from .itu import parse_spec as itu_spec
            v = value.lower()
            if v != "auto":
                try:
                    spec = itu_spec(v)
                except ValueError as exc:
                    raise CliError(str(exc)) from None
                v = "none" if spec is None else ":".join(map(str, spec))
            o.itu = v
        elif key == "shadow-sets" and value.isdigit() and 1 <= int(value) <= 16:
            o.shadow_sets = int(value)
        elif key == "compressed" and value.lower() in COMPRESSED:
            if value.lower() == "mips16e" and self.s.release == "R6":
                raise CliError("Release 6 has no MIPS16e")
            if value.lower().startswith("micromips") and self.s.release == "R1":
                raise CliError("microMIPS32 needs Release 2 or later (set release R2)")
            o.compressed = value.lower()
        elif key == "fpu":
            o.fpu = on
        elif key == "dsp" and value.lower() in DSP_LEVELS:
            if value.lower() in ("dsp", "dspr2") and self.s.release == "R1":
                raise CliError("the DSP Module needs Release 2 or later (set release R2)")
            if value.lower() == "dsp" and self.s.release == "R6":
                raise CliError("Release 6 has only DSP Revision 3: set dsp dspr2")
            o.dsp = value.lower()
        elif key == "msa" and value.lower() in MSA_LEVELS:
            if value.lower() == "msa" and self.s.release not in ("R5", "R6"):
                raise CliError("MSA needs Release 5 or later (set release R5)")
            o.msa = value.lower()
        elif key == "msa-wrp" and value.isdigit() and 0 <= int(value) <= 32:
            o.msa_wrp = int(value)
        elif key == "mt":
            from .mt import parse_spec
            v = value.lower()
            if v != "auto":
                try:
                    spec = parse_spec(v)
                except ValueError as exc:
                    raise CliError(str(exc)) from None
                if spec is not None and self.s.release in ("R1", "R6"):
                    raise CliError("the MT Module needs Release 2-5 (set release R5)")
                v = "none" if spec is None else f"{spec[0]}:{spec[1]}"
            o.mt = v
        elif key == "mt-pm":
            from .mt import POLICY_MANAGERS
            if value.lower() not in POLICY_MANAGERS:
                raise CliError("set mt-pm rr|wrr|wrr2")
            o.mt_pm = value.lower()
        elif key == "itc":
            from .itc import parse_spec as itc_spec
            v = value.lower()
            if v != "auto":
                try:
                    spec = itc_spec(v)
                except ValueError as exc:
                    raise CliError(str(exc)) from None
                v = "none" if spec is None else f"{spec[0]}:{spec[1]}"
            o.itc = v
        elif key == "core":
            from . import cores
            try:
                built, release = cores.apply(o, value)
            except ValueError as exc:
                raise CliError(str(exc)) from None
            s.options = built
            if release:
                s.release = release
            core = cores.get(built.core)
            self.say(f"core: {core.title if core else 'the generic eMIPS core'}"
                     + (f" (Release {core.release[1:]}, PRId 0x{core.prid:08x})" if core else ""))
        elif key == "mmu":
            from .mmu import make_mmu
            try:
                make_mmu(value)
            except ValueError as exc:
                raise CliError(str(exc)) from None
            o.mmu = value.lower()
        elif key == "board":
            from .malta import make_board
            try:
                make_board(value)
            except ValueError as exc:
                raise CliError(str(exc)) from None
            o.board = value.lower()
        elif key == "arch" and value.lower() in ARCHES:
            o.arch = value.lower()
            if o.arch == "nanomips":
                s.release = "R6"
        elif key == "nms":
            o.nms = on
        elif key == "relax":                    # (RISC-V: as -mrelax + ld's relaxation)
            if s.rv:
                o.rv_relax = on
            else:
                o.relax = on
        elif key == "rv-ext":
            from .riscv import RVISA, parse_extensions
            try:
                if "i" not in parse_extensions(value):
                    raise ValueError("an RV32I / RV64I base (i) is needed")
                RVISA.for_core(64, value)
            except (ValueError, KeyError) as exc:
                raise CliError(f"rv-ext {value}: {exc}") from None
            o.rv_extensions = value.lower()
        elif key == "rv-modes" and value.lower() in RV_MODES:
            o.rv_modes = value.lower()
        elif key == "crc" and value.lower() in NANO_CRC:
            o.crc = value.lower()
        elif key == "ginv" and value.lower() in NANO_GINV:
            o.ginv = value.lower()
        elif key == "disks":
            from .malta import parse_disks
            value = "" if value.lower() == "none" else value
            try:
                parse_disks(value)
            except ValueError as exc:
                raise CliError(str(exc)) from None
            if value and o.board.split(":")[0] != "malta":
                raise CliError("the IDE disks are on the Malta: set board malta first")
            o.disks = value
            s._media = {}
        else:
            raise CliError(f"unknown option or value: set {arg}")
        had = s.options.core
        self._reload()
        if had and not s.options.core:
            for note in s.adjusted:
                if note.startswith("core "):
                    self.say(f"adjusted: {note}")

    def _reload(self) -> None:
        s = self.s
        if not s.source:                       # nothing loaded yet
            return
        ok = (s.load_hex(s.source) if s.kind == "hex" else
              s.load_c(s.c_text, Path(s.c_file).name, optimize=f"-O{s.options.c_optimize}")
              if s.c_file is not None else
              s.load_elf(s.elf_data, s.source) if s.kind == "elf" else s.load_source(s.source))
        self._printed = 0
        if not ok:
            self.say("error: " + s.errors[0])
            return
        self._show_position()

    def do_load(self, arg: str) -> None:
        """load FILE   an assembly source (.asm / .s) or an Intel HEX image"""
        if not arg.strip():
            raise CliError("load FILE")
        if not load_file(self.s, shlex.split(arg)[0], self.say):
            return
        self._printed = 0
        self._show_position()

    def do_quit(self, arg: str) -> bool:
        """quit   (q, Ctrl+D) leave"""
        return True

    do_q = do_exit = do_quit

    def do_EOF(self, arg: str) -> bool:
        if self.stdin is sys.stdin and sys.stdin.isatty():
            self.say()
        return True

    # -- help --------------------------------------------------------------------------

    def do_help(self, arg: str) -> None:
        """help [COMMAND]   the commands, or one command"""
        if arg:
            fn = getattr(self, f"do_{arg}", None)
            if fn is None or not fn.__doc__:
                raise CliError(f"no help for {arg!r}")
            self.say(_dedent(fn.__doc__))
            return
        self.say(HELP)

    @staticmethod
    def _count(arg: str) -> int:
        try:
            n = int(arg, 0) if arg.strip() else 1
        except ValueError:
            raise CliError(f"not a number: {arg.strip()}") from None
        if n < 1:
            raise CliError("the count must be at least 1")
        return n

    def _gpr(self, tok: str) -> int | None:
        """A GPR name in the core's ABI (MIPS64: the n64 names, $t0 = $12;
        nanoMIPS: the p32 ones, $t4 = $2, $a4 = $8, and GNU as's $r0 ...)."""
        if self.s.rv:                           # x5, t0, fp, $t0
            from .sessionrv import gpr_number
            return gpr_number(tok)
        from .syntax import gpr_number
        return gpr_number(tok, "p32" if self.s.nano else "n64" if self.s.wide else True)

    @staticmethod
    def _signed(v: int, bits: int = 32) -> int:
        return v - (1 << bits) if v >> (bits - 1) else v


_RV_MODE = {3: "M", 1: "S", 0: "U"}

HELP = """\
running      step [N] (s)   next [N] (n, over calls)   back [N]   continue [N] (c, run)
             until LOC (u)  finish   reset
breakpoints  break [LOC] (b)   delete LOC|all (d)
looking      regs [$r ...]   cp0   csr (RISC-V)   fpu   mem [-p] LOC [WORDS] (x)   translate LOC (t)
             dis [LOC] [N]   list [LOC] (l)   cache [stats|last|I|D|L2 [set N|LOC]|LOC]
             tlb [all|entry N|set N|micro|stats|last|LOC]   perf [events|N EVENT]
             ejtag [dint|ib|db|watch|clear] (Debug Mode, breakpoints)   srs (interrupts)
             probe [on|off|load|break|fastdata|profile] (EJTAG probe)   tap ir|dr|reset
             core [buffer|predict|units] (the 24K model)   cpu [N] (the cores)   cm [LOC]   gic   cps
             where   view (inside the last instruction)   chart (pipeline)   print VALUE (p)
             board (the Malta)   disk [save|show] (its IDE disks)   trace [on|show|...]
tools        decode WORD   encode INSTRUCTION   ref TEXT
console      input TEXT (a line for read_*)   keys TEXT (MMIO keyboard)
setup        mode instruction|pipeline   set OPTION VALUE (help set)   load FILE
             quit (q)
LOC: 0x... address, a line number, a label, $reg (its value) or pc.
help COMMAND for details."""


def encoding(word: int, size: int, arch: str = "mips32") -> str:
    """An instruction's encoding as a listing shows it: 8 digits; a 16-bit
    MIPS16e / microMIPS one 4 (and 4 spaces); a nanoMIPS one its halfwords
    as objdump prints them (``d205``, ``00a4 0007``, ``6080 5678 1234``)."""
    if arch == "nanomips":
        halves = [(word >> (16 * k)) & 0xFFFF for k in range(size // 2 - 1, -1, -1)]
        return " ".join(f"{h:04x}" for h in halves).ljust(14)
    return f"{word:04x}    " if size == 2 else f"{word:08x}"


def _dedent(doc: str) -> str:
    lines = doc.splitlines()
    return "\n".join([lines[0]] + [ln[8:] if ln.startswith("        ") else ln.strip()
                                   for ln in lines[1:]])


def load_file(s: Session, path: str, say) -> bool:
    """Load ``path`` (an ELF executable by its magic; a C file is built
    first with emips.cc; else by suffix: Intel HEX or assembly) into the
    session; errors are said."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        say(f"error: cannot open {path}: {exc.strerror or exc}")
        return False
    text = raw.decode("utf-8", errors="replace")
    if path.lower().endswith(".c"):                    # C: built first (emips.cc)
        ok = s.load_c(text, Path(path).name, optimize=f"-O{s.options.c_optimize}")
        if not ok:
            say(f"error: {path}: {s.errors[0]}")
        return ok
    if raw[:4] == b"\x7fELF":                          # an ELF executable (emips.elf)
        ok = s.load_elf(raw, path)
    else:
        ok = s.load_hex(text) if path.lower().endswith((".hex", ".ihex")) else \
            s.load_source(text)
    if not ok:
        say(f"error: {path}: {s.errors[0]}")
    return ok


def options_from_args(args) -> Options:
    """Session options from the shared command-line flags."""
    default = "32K" if getattr(args, "caches", False) else "none"
    icache, dcache = getattr(args, "icache", None), getattr(args, "dcache", None)
    return Options(pseudo=not getattr(args, "strict", False),
                   reorder=not getattr(args, "noreorder", False),
                   boot=getattr(args, "boot", "kernel"),
                   exceptions=getattr(args, "exceptions", "cp0"),
                   syscalls=not getattr(args, "no_syscalls", False),
                   icache=icache if icache is not None else default,
                   dcache=dcache if dcache is not None else default,
                   cache_parity=getattr(args, "cache_parity", False),
                   cache_init=getattr(args, "cache_init", "random"),
                   l2=getattr(args, "l2", None) or "none",
                   l3=getattr(args, "l3", None) or "none",
                   cache_ecc=getattr(args, "cache_ecc", False),
                   write_buffer=getattr(args, "write_buffer", False),
                   mem_timing=dict(MEMORY_TIMING, **(getattr(args, "mem_timing", None) or {})),
                   nonblocking=getattr(args, "nonblocking", False),
                   mmu=getattr(args, "mmu", None) or "fixed",
                   board=getattr(args, "board", None) or "flat",
                   disks=",".join(getattr(args, "disk", None) or []),
                   model=getattr(args, "model", None) or "5-stage",
                   cores=getattr(args, "cores", None) or 1,
                   iocu=getattr(args, "iocu", False),
                   shadow_sets=getattr(args, "shadow_sets", None) or 4,
                   compressed=getattr(args, "compressed", None) or "auto",
                   fpu=not getattr(args, "no_fpu", False),
                   dsp=getattr(args, "dsp", None) or "auto",
                   msa=getattr(args, "msa", None) or "auto",
                   msa_wrp=getattr(args, "msa_wrp", None) or 0,
                   mt=getattr(args, "mt", None) or "auto",
                   mt_pm=getattr(args, "mt_pm", None) or "rr",
                   itc=getattr(args, "itc", None) or "auto",
                   vps=getattr(args, "vps", None) or 1,
                   itu=getattr(args, "itu", None) or "auto",
                   arch=getattr(args, "arch", None) or "mips32",
                   nms=getattr(args, "nms", False),
                   relax=getattr(args, "relax", False),
                   crc=getattr(args, "crc", None) or "auto",
                   ginv=getattr(args, "ginv", None) or "auto",
                   rv_extensions=getattr(args, "rv_extensions", None) or RV_EXTENSIONS,
                   rv_modes=getattr(args, "rv_modes", None) or "msu",
                   rv_relax=not getattr(args, "no_relax", False))


def explicit_fields(args) -> set[str]:
    """The Options fields the command line (or the program's header) gave
    on purpose: a core preset (``--core``) sets only the others."""
    out = {"icache", "dcache"} if getattr(args, "caches", False) else set()
    for name in ("icache", "dcache", "l2", "l3", "mmu", "model", "cores", "shadow_sets",
                 "compressed", "dsp", "msa", "msa_wrp", "mt", "mt_pm", "itc", "vps", "itu",
                 "arch", "crc", "ginv", "rv_extensions", "rv_modes"):
        if getattr(args, name, None) is not None:
            out.add(name)
    for name, flag in (("fpu", "no_fpu"), ("write_buffer", "write_buffer"), ("nms", "nms")):
        if getattr(args, flag, False):
            out.add(name)
    return out


def apply_header(args) -> str | None:
    """The program's ``eMIPS:`` header (emips.profile) supplies the flags
    the command line did not give; returns what it applied (None: no
    header)."""
    from .profile import read
    path = getattr(args, "file", None)
    if not path or getattr(args, "no_header", False):
        return None
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        prof = read(text, Path(path).resolve().parent)
    except OSError:
        return None
    except ValueError as exc:
        sys.exit(f"{path}: {exc}")
    if prof is None:
        return None
    defaults = {"endian": "big", "boot": "kernel"}
    for key, value in prof.given.items():
        if key == "little":
            if getattr(args, "endian", "big") == "big":
                args.endian = "little"
            continue
        if not hasattr(args, key):
            continue                            # e.g. --pipeline for pipe
        cur = getattr(args, key)
        if cur in (None, False) or cur == defaults.get(key):
            setattr(args, key, value)
    return prof.describe()


def session_from_args(args) -> Session:
    from . import cores
    from .compat import adjust, explain, problems
    apply_header(args)
    options = options_from_args(args)
    release = args.release
    core = getattr(args, "core", None)
    if core:                                    # its build, under what was given
        options, core_release = cores.apply(options, core, keep=explicit_fields(args))
        release = release or core_release
    release = release or ("R6" if options.arch == "nanomips" else "R5")
    given = {f for f, a in (("shadow_sets", "shadow_sets"), ("cores", "cores"), ("mmu", "mmu"),
                            ("l3", "l3"), ("boot", "boot"), ("disks", "disk"), ("iocu", "iocu"),
                            ("model", "model"), ("compressed", "compressed"), ("dsp", "dsp"),
                            ("msa", "msa"), ("mt", "mt"), ("vps", "vps"), ("arch", "arch"), ("board", "board"))
             if getattr(args, a, None) not in (None, "kernel")}
    if options.core:
        given |= {"core"} | explicit_fields(args)
    options, _notes = adjust(options, release, keep=given)
    bad = problems(options, release)
    if bad:                                     # given together on purpose: refuse
        sys.exit("these settings do not go together: " + explain(bad, options, release))
    s = Session(release=release, endian=args.endian,
                mode="pipeline" if getattr(args, "pipeline", False) else "instruction",
                forwarding=not getattr(args, "no_forwarding", False),
                abi=not getattr(args, "numeric", False),
                options=options)
    # pipe --events / --views print every cycle: keep them all
    s.full_history = bool(getattr(args, "events", False) or getattr(args, "views", False))
    return s
