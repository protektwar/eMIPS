"""One debugging session — the engine of the GUI (M8) and the CLI debugger
(M7), free of Qt.

A :class:`Session` owns the program, the machine and (in pipeline mode) the
pipeline; it steps / runs / steps back, keeps breakpoints, routes console
I/O, and exposes the state the views draw.  The Qt widgets only read it
and call its methods, so everything here is unit-tested without a display.

*Step back* replays from reset to the previous step: the emulator is
deterministic, and the console input consumed so far is recorded and fed
again in the same order.

A program comes from assembly source (:meth:`Session.load_source`) or from
an Intel HEX image (:meth:`Session.load_hex`); for HEX the session builds a
disassembly listing (:attr:`Session.listing`) whose lines play the role of
source lines (highlighting, breakpoints).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cp0 as C
from . import ejtag as EJ
from .assembler import BOOT_SP, AsmError, Program, SourceLine, assemble
from .console import MmioConsole, SyscallServices
from .cps import CPS
from .cache import Caches
from .disasm import disassemble
from .exceptions import SimException
from .hexload import DecodeFile, HexLoadError
from .isa import ISA, ISADecodeError
from .memory import PMemory
from .pipe24k import Pipeline24K
from .pipeline import CycleTrace, Pipeline
from .sim import Machine
from .syntax import gpr_name
from . import tap as TAPM
from . import pdtrace as PDT
from . import perf
from .options import Options

M32 = 0xFFFFFFFF
READ_SERVICES = {5, 6, 7, 8, 12}
#: the stack of core N starts N x 64 KB below core 0's (the boot firmware)
CORE_STACK = 0x10000


#: calls stepped over by :meth:`Session.step_over` (they write a return
#: address); the compact R6 ones return to PC + 4 (no delay slot)
_COMPACT_CALLS = {"BALC", "JIALC", "BEQZALC", "BNEZALC", "BLEZALC", "BGEZALC",
                  "BGTZALC", "BLTZALC"}
_CALLS = {"JAL", "JALR", "JALR.HB", "JALX", "BAL",
          "BGEZAL", "BLTZAL", "BGEZALL", "BLTZALL"}


@dataclass
class Frame:
    """A call not returned yet: the calling instruction, the address it
    jumped to, where it returns, $sp at the call."""
    site: int
    target: int
    ret: int
    sp: int


class InputNeeded(Exception):
    """The program waits for a line of input (a read_* syscall)."""


@dataclass
class StepResult:
    """What one step / run did, for the status bar and the views."""

    steps: int = 0
    stopped: str = ""                   # "", "breakpoint", "exit", "halt", "input", ...
    error: str | None = None
    trace: CycleTrace | None = None
    #: the exceptions taken by this step (Machine.exception_log entries)
    exceptions: list = field(default_factory=list)


class _SourceFiles:
    """The text of source files named by a line table (read once; a file
    that cannot be read gives empty lines)."""

    _shared: "_SourceFiles | None" = None

    def __init__(self) -> None:
        self.files: dict[str, list[str]] = {}

    @classmethod
    def shared(cls) -> "_SourceFiles":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def line(self, path: str, n: int) -> str:
        text = self.files.get(path)
        if text is None:
            try:
                text = Path(path).read_text(errors="replace").splitlines()
            except OSError:
                text = []
            self.files[path] = text
        return text[n - 1].strip() if 0 < n <= len(text) else ""


@dataclass
class Session:
    release: str = "R5"
    endian: str = "big"
    mode: str = "instruction"           # or "pipeline"
    forwarding: bool = True
    syscalls: bool = True
    #: register names: ABI ($t0) or numbers ($8)
    abi: bool = True
    #: "asm" (source assembled) or "hex" (Intel HEX image)
    kind: str = "asm"
    #: the bytes of a loaded ELF executable (kind "elf"; source = its path)
    elf_data: bytes = b""
    source: str = ""
    program: Program | None = None
    errors: list[str] = field(default_factory=list)
    breakpoints: set[int] = field(default_factory=set)
    #: performance counters set from the debugger / GUI (not by the
    #: program): counter -> (event, modes, ie); applied again at every
    #: reset, so Reset and Step Back keep them
    counters: dict = field(default_factory=dict)
    #: views of executed instructions by sequence (pipeline) / step number
    views: dict[int, Any] = field(default_factory=dict)
    #: Settings-dialog options (assembler, boot, exceptions, console, ...)
    options: Options = field(default_factory=Options)

    def __post_init__(self) -> None:
        self.machine: Machine | None = None
        self.pipeline: Pipeline | None = None
        self.console: MmioConsole | None = None
        self.services: SyscallServices | None = None
        self.output = ""
        self.steps = 0
        self._inputs: list[str] = []        # lines given so far (replay)
        self._input_pos = 0
        self._keys = ""                     # MMIO keyboard text given so far
        #: debug interrupts given: (step, core) — replayed by a step back
        self._dints: list[tuple[int, int]] = []
        #: debug registers the debugger set: (step, core, "drseg" | "cp0",
        #: offset | (reg, sel), value) — replayed by a step back
        self._pokes: list[tuple] = []
        #: EJTAG: each core's TAP (fresh at every reset: power-up), the
        #: probe on them (configuration: kept across resets) and whether
        #: it gives the EJTAGBOOT instruction; PC samples it collected
        self.taps: list[TAPM.TAP] = []
        self.probe: TAPM.Probe | None = None
        self.probe_boot = False
        self.prev_regs: list[int] = [0] * 32
        self.last_view: Any = None
        #: run-to-cursor target: :meth:`run` stops when it reaches it
        self.run_to: int | None = None
        #: HEX mode: the disassembly listing shown instead of source
        self.listing = ""
        #: steps :meth:`run` may still take (Settings > Run step limit;
        #: None = no limit); the GUI sets it when Run starts
        self.budget: int | None = None
        #: :meth:`run` stops ("end") before an instruction outside the
        #: program (fell off its end) instead of fetching garbage (CLI)
        self.stop_at_end = False
        #: called with every piece of program output (live printing, CLI)
        self.on_output = None
        #: the cores (one Machine / pipeline each); :attr:`machine` /
        #: :attr:`pipeline` are the selected core's (:attr:`cpu`, what the
        #: views show); :attr:`cps` = the platform with more than one core
        self.machines: list[Machine] = []
        self.pipelines: list = []
        self.cps: CPS | None = None
        self.cpu = 0
        self._views: list[dict] = []
        self._last_views: list = []
        self._prev: list[list[int]] = []
        #: the last cycle's trace of every core (pipeline mode)
        self.traces: list = []
        self._code: set[int] | None = None

    # -- program --------------------------------------------------------------

    def load_source(self, text: str) -> bool:
        """Assemble ``text``; on success reset to the new program."""
        self.source = text
        self.errors = []
        try:
            self.program = assemble(text, endian=self.endian, release=self.release,
                                    pseudo=self.options.pseudo,
                                    reorder=self.options.reorder)
        except AsmError as exc:
            self.errors = [str(exc)]
            self._drop_program()
            return False
        self.kind = "asm"
        self._inputs, self._keys, self._dints, self._pokes = [], "", [], []
        self.reset()
        return True

    def load_hex(self, text: str) -> bool:
        """Load an Intel HEX image: the code (the run holding the entry, and
        words below 0x10000000 or in kseg0 0x80000000-0x8fffffff, where
        MARS / GNU put text and the exception handler) is disassembled into
        :attr:`listing`; every other word is data.  The PC starts at the
        start-address record (05) or the lowest address."""
        self.source = text
        self.errors = []
        loader = DecodeFile(PMemory())
        try:
            loader.load_text(text)
        except (HexLoadError, ValueError) as exc:
            self.errors = [f"Intel HEX: {exc}"]
            self._drop_program()
            return False
        image = {int(a, 16): int(v, 16)
                 for a, v in loader._memory.get_memory().items()}
        if not image:
            self.errors = ["Intel HEX: no data records"]
            self._drop_program()
            return False
        entry = loader.start_address if loader.has_start else min(image)
        prog = Program(endian=self.endian, release=self.release, image=image,
                       entry=entry)
        words = sorted({a & ~3 for a in image
                        if all((a & ~3) + i in image for i in range(4))})
        runs: list[list[int]] = []
        for a in words:
            if runs and runs[-1][-1] + 4 == a:
                runs[-1].append(a)
            else:
                runs.append([a])

        def is_code(run: list[int]) -> bool:
            a = run[0]
            return (run[0] <= entry <= run[-1] or a < 0x10000000
                    or 0x80000000 <= a < 0x90000000)

        isa = ISA.for_release(self.release)
        code = [a for r in runs if is_code(r) for a in r]
        data = [a for r in runs if not is_code(r) for a in r]
        lines = [f"# Intel HEX image: {len(image)} bytes, entry 0x{entry:08x}",
                 f"# MIPS32 {self.release}, {self.endian}-endian — read-only "
                 "disassembly (address, word, instruction)", "", "# code"]
        for a in code:
            w = prog.word(a)
            text_ = f"{a:08x}  {w:08x}  {disassemble(w, a, isa=isa, abi=self.abi)}"
            lines.append(text_)
            prog.source_map[a] = SourceLine(len(lines), text_)
        if data:
            lines += ["", "# data"]
            lines += [f"{a:08x}  {prog.word(a):08x}" for a in data]
        self.listing = "\n".join(lines) + "\n"
        self.program = prog
        self.kind = "hex"
        self._inputs, self._keys, self._dints, self._pokes = [], "", [], []
        self.reset()
        return True

    def load_elf(self, data: bytes, path: str = "") -> bool:
        """Load an ELF32 executable (emips.elf): its segments, entry and
        symbols; the release and byte order it was built for (EF_MIPS_ARCH,
        EI_DATA) become the session's.  :attr:`listing` is a read-only
        disassembly with each source line (DWARF line table; the text when
        the file can be read) above its instructions."""
        from .elf import ElfError, parse_elf
        self.source = path
        self.elf_data = data
        self.errors = []
        try:
            img = parse_elf(data, path or "<elf>")
        except ElfError as exc:
            self.errors = [f"ELF: {exc}"]
            self._drop_program()
            return False
        self.release, self.endian = img.release, img.endian
        symbols = {}
        for sym in sorted(img.symbols, key=lambda x: x.kind != "func"):
            symbols.setdefault(sym.name, sym.value)
        prog = Program(endian=img.endian, release=img.release, image=img.image,
                       symbols=symbols, entry=img.entry, c_lines=img.source_lines(),
                       c_stmts=img.statement_starts(),
                       functions=sorted((sym.value, sym.value + max(sym.size, 4), sym.name)
                                        for sym in img.symbols if sym.kind == "func"))
        for name, addr, size, _x in img.sections:
            prog.segments[name] = (addr, addr + size)
        labels = {}
        for sym in img.symbols:
            if sym.kind == "func":
                labels.setdefault(sym.value, sym.name)
        isa = ISA.for_release(img.release)
        texts = _SourceFiles()
        entry_name = labels.get(img.entry, "")
        lines = [f"# ELF executable {Path(path).name if path else ''}: {img.arch_name}, "
                 f"{img.endian}-endian, entry 0x{img.entry:08x}"
                 + (f" ({entry_name})" if entry_name else ""),
                 "# read-only listing: each source line (DWARF line table) above its "
                 "instructions"]
        lines += [f"# note: {n}" for n in img.notes]
        last = None
        for start, stop in img.code_ranges():
            lines.append("")
            for a in range(start & ~3, stop, 4):
                if a in labels:
                    lines += ([] if lines[-1] == "" else [""]) + [f"{labels[a]}:"]
                    last = None
                where = prog.c_lines.get(a)
                if where is not None and where != last:
                    f, n = where
                    lines.append(f"# {Path(f).name}:{n}  {texts.line(f, n)}".rstrip())
                    last = where
                w = prog.word(a)
                text_ = f"{a:08x}  {w:08x}  {disassemble(w, a, isa=isa, abi=self.abi)}"
                lines.append(text_)
                prog.source_map[a] = SourceLine(len(lines), text_)
        data_sections = [(n, a, sz) for n, a, sz, x in img.sections if not x]
        if data_sections:
            lines += ["", "# data"] + [f"# {n:<12} 0x{a:08x}  {sz} bytes"
                                       for n, a, sz in data_sections]
        self.listing = "\n".join(lines) + "\n"
        self.program = prog
        self.kind = "elf"
        self._inputs, self._keys, self._dints, self._pokes = [], "", [], []
        self.reset()
        return True

    def source_position(self, addr: int) -> tuple[str, int, str] | None:
        """``(file, line, text)`` of a compiled program's instruction (its
        DWARF line), or ``None``."""
        if self.program is None or not self.program.c_lines:
            return None
        where = self.program.c_lines.get(addr)
        if where is None:
            return None
        return where[0], where[1], _SourceFiles.shared().line(*where)

    def _drop_program(self) -> None:
        """A load failed: nothing is runnable (the old program must not go
        on running as if the new source were loaded)."""
        self.program = None
        self.machine = self.pipeline = self.console = self.services = None
        self.machines, self.pipelines, self.cps = [], [], None
        self.views, self.last_view, self.steps = {}, None, 0

    def unload(self) -> None:
        """No program (File > New)."""
        self.program = None
        self.machine = self.pipeline = self.console = self.services = None
        self.machines, self.pipelines, self.cps = [], [], None
        self.source, self.listing, self.kind = "", "", "asm"
        self.errors, self.views = [], {}
        self.output, self.steps, self.last_view, self.run_to = "", 0, None, None
        self.breakpoints = set()

    def reset(self) -> None:
        """Fresh machine (and pipeline), program loaded, inputs kept;
        applies :attr:`options` (boot state, exceptions, console, timer,
        latencies).  With more than one core: a CPS (emips.cps) on one
        physical memory; the kernel / user boot starts every core at the
        entry (the firmware's job), the reset boot only core 0 (the others
        are powered down until the program starts them with the CPC)."""
        o = self.options
        self.output = ""
        self.console = MmioConsole(on_output=self._emit, tx_delay=o.tx_delay,
                                   rx_delay=o.rx_delay)
        self.console.feed(self._keys)
        self._input_pos = 0
        self.services = None
        if self.syscalls and o.syscalls:
            self.services = SyscallServices(read_line=self._read_line, write=self._emit)
        mem = PMemory()
        self.taps = []
        if self.probe is not None:
            self.probe.power_up()
        n = o.cores
        if n == 1:
            self.cps = None
            self.machines = [self._make_core(None, 0, mem)]
        else:
            l2, l3 = o.shared_levels()
            self.cps = CPS(n, lambda cps, i: self._make_core(cps, i, mem), self._core_started,
                           l2=l2, l3=l3, ecc=o.cache_ecc,
                           hardware_init=o.cache_init == "hardware")
            self.machines = self.cps.cores
            self.console.sink = self.cps.gic.set_input     # TX, RX: GIC sources 0, 1
        boot = o.boot != "reset"
        if self.program is not None:
            for i in range(n) if boot else (0,):
                m = self.machines[i]
                self.program.load(m, boot=boot)
                if o.boot == "user":            # KSU = user (Status.UM)
                    m._cp0_set(C.STATUS, m.status | (2 << C.ST_KSU_SHIFT))
                if boot and i:                  # its own stack
                    m.write_reg(29, BOOT_SP - i * CORE_STACK)
            if boot and self.cps is not None:
                self.cps.boot()
        self.pipelines = [self._make_pipeline(m) for m in self.machines]
        self._code = None
        self.steps = 0
        self.run_to = None
        self._views = [{} for _ in range(n)]
        self._last_views = [None] * n
        self._prev = [[m.read_reg(i) for i in range(32)] for m in self.machines]
        #: per core, the calls not returned yet (the debugger's call stack)
        self.calls = [[] for _ in self.machines]
        for cpu, m in enumerate(self.machines):
            self._watch_calls(cpu, m)
        self.traces = [None] * n
        for m in self.machines:
            self._apply_counters(m)
        self.select(min(self.cpu, n - 1))

    def _make_core(self, cps: CPS | None, cpu: int, mem: PMemory) -> Machine:
        """A core in its hardware reset state on the shared memory and
        devices (the console's interrupts go to core 0); in a CPS its
        caches join the Coherence Manager (the shared L2 / L3)."""
        o = self.options
        caches = o.caches(core=cpu, shared=cps is not None)
        if cps is not None and caches is None and cps.cm.l2 is not None:
            caches = Caches(None, None)               # only the shared L2 / L3
        if caches is not None and cps is not None:
            cps.cm.attach(cpu, caches)
        m = Machine(memory=mem, endian=self.endian, release=self.release,
                    exceptions="raise" if o.exceptions == "stop" else "cp0",
                    caches=caches, mmu=o.make_mmu(), cpu=cpu,
                    shadow_sets=o.shadow_sets)
        m.tracing = True
        m.ccres = o.count_rate
        if cpu == 0:
            self.console.attach(m)
        else:
            self.console.share(m)
        m.syscall_handler = self.services
        while len(self.taps) <= cpu:            # the core's TAP (power-up)
            t = TAPM.TAP()
            t.ejtagboot = self.probe_boot
            self.taps.append(t)
        tap = self.taps[cpu]
        tap.processor_reset(m)                  # Rocc, EJTAGBOOT's ECR values
        if self.probe is not None:              # the probe sees Rocc: it takes the TAP
            self.probe.connect(tap)
        return m

    def _make_pipeline(self, m: Machine):
        if self.mode != "pipeline":
            return None
        o = self.options
        cls = Pipeline24K if o.model == "24k" else Pipeline
        return cls(m, forwarding=self.forwarding, latency=o.latency,
                   memory=o.mem_timing, nonblocking=o.nonblocking)

    def _apply_counters(self, m: Machine) -> None:
        for i, (event, modes, ie) in sorted(self.counters.items()):
            perf.configure(m, i, event, modes, ie)

    def _core_started(self, cpu: int, m: Machine) -> None:
        """The CPC / GCR powered a core up or reset it: a fresh core (and
        pipeline) at its reset vector."""
        self._apply_counters(m)
        self.pipelines[cpu] = self._make_pipeline(m)
        self._prev[cpu] = [m.read_reg(i) for i in range(32)]
        if len(self.calls) > cpu:
            self.calls[cpu] = []
            self._watch_calls(cpu, m)
        self._last_views[cpu] = None
        self.traces[cpu] = None
        if cpu == self.cpu:
            self.select(cpu)

    def select(self, cpu: int) -> None:
        """Show core ``cpu`` (the views, the debugger's commands)."""
        if not 0 <= cpu < len(self.machines):
            raise ValueError(f"there is no core {cpu} (cores 0-{len(self.machines) - 1})")
        self.cpu = cpu
        self.machine = self.machines[cpu]
        self.pipeline = self.pipelines[cpu]
        self.views = self._views[cpu]
        self.last_view = self._last_views[cpu]
        self.prev_regs = self._prev[cpu]

    def running(self) -> list[int]:
        """The cores that execute (one: always core 0)."""
        if self.cps is None:
            return [0] if self.machines else []
        return self.cps.running()

    def configure_counter(self, i: int, event: int | None, modes: str = "uk",
                          ie: bool = False) -> None:
        """Set performance counter ``i`` as software would (MTC0) and keep
        the setting for the next resets; ``event=None`` stops it."""
        if event is None:
            self.counters.pop(i, None)
            for m in self.machines:
                m.cp0_write(*perf.PERFCTL[i], m.cp0_read(*perf.PERFCTL[i]) & ~perf.CTL_MODES)
            return
        self.counters[i] = (event, modes, ie)
        for m in self.machines:
            perf.configure(m, i, event, modes, ie)

    def set_mode(self, mode: str, *, forwarding: bool | None = None) -> None:
        self.mode = mode
        if forwarding is not None:
            self.forwarding = forwarding
        if self.program is not None:            # nothing to reset before a load
            self.reset()

    # -- console --------------------------------------------------------------

    def _emit(self, text: str) -> None:
        self.output += text
        if self.on_output is not None:
            self.on_output(text)

    def _read_line(self) -> str:
        if self._input_pos >= len(self._inputs):
            raise InputNeeded()
        line = self._inputs[self._input_pos]
        self._input_pos += 1
        return line

    def give_input(self, line: str) -> None:
        """A line for the next read_* syscall."""
        self._inputs.append(line)

    def type_keys(self, text: str) -> None:
        """Keystrokes for the memory-mapped keyboard."""
        self._keys += text
        if self.console is not None:
            self.console.feed(text)

    def needs_input(self) -> bool:
        """Is the next instruction to execute (on any core) a read_*
        syscall with no input available yet?"""
        if (not self.machines or self.services is None
                or self._input_pos < len(self._inputs)):
            return False
        return any(self._core_needs_input(i) for i in self.running())

    def _core_needs_input(self, cpu: int) -> bool:
        m = self.machines[cpu]
        pc = self._next_executing_pc(cpu)
        if pc is None:
            return False
        try:
            word = m.code_word(pc)
            if word & 0xFC00003F != 0x0000000C:  # not SYSCALL: no decode needed
                return False
            dec = m.isa.decode(word)
        except (SimException, ISADecodeError):
            return False
        return dec.name == "SYSCALL" and m.read_reg(2) in READ_SERVICES

    def _next_executing_pc(self, cpu: int | None = None) -> int | None:
        cpu = self.cpu if cpu is None else cpu
        p = self.pipelines[cpu]
        if p is None:
            return self.machines[cpu].pc
        nxt = p.next_slot()
        return nxt.pc if nxt is not None else None

    # -- stepping -------------------------------------------------------------

    @property
    def halted(self) -> bool:
        """The program ended: a core exited (and its pipeline drained), or
        no core runs any more."""
        if not self.machines:
            return True
        for m, p in zip(self.machines, self.pipelines):
            if m.halted and (p is None or p.empty()):
                return True
        return not self.running()

    def step(self) -> StepResult:
        """One instruction (instruction mode) or one clock cycle (pipeline)
        — of every running core, core 0 first."""
        res = StepResult()
        if self.machine is None or self.halted:
            res.stopped = "exit" if self.machine is not None else "no program"
            return res
        if self.needs_input():
            res.stopped = "input"
            return res
        for k, i in self._dints:                # replayed debug interrupts
            if k == self.steps and i < len(self.machines):
                self.machines[i].debug_interrupt()
        for k, i, kind, key, value in self._pokes:    # and debug register writes
            if k == self.steps and i < len(self.machines):
                self._poke(self.machines[i], kind, key, value)
        cores = self.running()
        logged = [(m.exception_log, len(m.exception_log)) for m in self.machines]
        for i in cores:
            m = self.machines[i]
            self._prev[i] = [m.read_reg(r) for r in range(32)]
        self.prev_regs = self._prev[self.cpu]
        cpu = None
        if self.cps is not None:
            self.cps.clock = self.steps         # one cycle (instruction) per step
        try:
            for cpu in cores:
                self._step_core(cpu)
        except InputNeeded:
            res.stopped = "input"
            return res
        except (SimException, ISADecodeError) as exc:
            if cpu != self.cpu:
                self.select(cpu)                # show the core that stopped
            where = f"core {cpu}: " if self.cps is not None else ""
            res.error = f"{where}{type(exc).__name__}: {exc}" + self._no_handler_hint(exc) \
                + self._past_end_hint(exc) + self._moved_code_hint(exc)
            res.stopped = "halt"
            return res
        self.select(self.cpu)                   # a core may have restarted
        if self.probe is not None and self.probe.profile:     # the probe reads PCsample
            for i in cores:
                if i < len(self.taps):
                    got = self.probe.read_sample(self.taps[i])
                    if got is not None:
                        self.probe.samples.append(got)
        res.trace = self.traces[self.cpu]
        self.steps += 1
        res.steps = 1
        for i, m in enumerate(self.machines):
            log, n = logged[i]
            start = n if m.exception_log is log else 0      # 0: a fresh core
            for e in m.exception_log[start:]:
                if "vector" in e:
                    res.exceptions.append(dict(e, cpu=i) if self.cps is not None else e)
        if self.halted:
            res.stopped = "exit"
        return res

    def _step_core(self, cpu: int) -> None:
        m, p = self.machines[cpu], self.pipelines[cpu]
        views = self._views[cpu]
        if p is None:
            m.last_view = None
            m.fetch_tlb = []
            m.step()
            view = m.last_view
            self._last_views[cpu] = view
            if view is not None:
                views[self.steps] = view
            return
        trace = p.cycle()
        self.traces[cpu] = trace
        for st in ("EX", "ID"):
            view = getattr(trace, f"{st.lower()}_view")
            if view is not None:
                sv = trace.stages.get(st)
                key = sv.seq if sv is not None else -1
                views[key] = view
                self._last_views[cpu] = view

    #: exception vectors (BEV = 0 / 1, general and interrupt offsets)
    VECTORS = {0x80000180, 0x80000200, 0xBFC00380, 0xBFC00400,
               0x80000000, 0xBFC00200}           # + the TLB Refill vectors

    def _no_handler_hint(self, exc: Exception) -> str:
        """An exception was taken but nothing is at the vector: say which
        exception and how to handle it."""
        from .exceptions import FetchError, ProbeWaitError
        m = self.machine
        if isinstance(exc, ProbeWaitError) or not isinstance(exc, FetchError) or m is None:
            return ""
        if m.debug_mode and m.pc == m.debug_vector():
            st = self.debug_state()
            why = ", ".join(st["causes"]) or f"an exception in Debug Mode (DExcCode " \
                f"{st['dexccode']})"
            return (f" — the CPU took a debug exception ({why}, DEPC 0x{st['depc']:08x}) "
                    f"and jumped to the debug vector 0x{m.pc:08x}, but there is no debug "
                    f"handler there: write one (.ktext 0x{m.pc:08x}) that ends with deret")
        if m.pc not in self.VECTORS:
            return ""
        code = (m.cp0_read(*C.CAUSE) >> C.CA_EXC_SHIFT) & 31
        name = C.EXC_NAMES.get(code, str(code))
        return (f" — the CPU took exception {name} (Cause.ExcCode {code}, EPC "
                f"0x{m.cp0_read(*C.EPC):08x}) and jumped to the exception vector "
                f"0x{m.pc:08x}, but there is no handler there: write one "
                f"(.ktext 0x{m.pc:08x}), or choose Settings > Exceptions: stop the "
                "simulation (--exceptions stop)")

    def _past_end_hint(self, exc: Exception) -> str:
        from .exceptions import FetchError
        m, p = self.machine, self.program
        if not isinstance(exc, FetchError) or m is None or p is None or self.kind != "asm":
            return ""
        text = p.segments.get("text")
        if text is None or m.pc != text[1]:
            return ""
        return (" — the program ran past its last instruction: end it with "
                "li $v0, 10 / syscall (exit)")

    def _moved_code_hint(self, exc: Exception) -> str:
        """The program has code at this kuseg address, but not at the
        physical address it now maps to: Status.ERL changed since the
        program was loaded (the fixed mapping moves kuseg by 1 GB)."""
        from .exceptions import FetchError
        m = self.machine
        if (not isinstance(exc, FetchError) or m is None or m.pc >= 0x80000000
                or m.pc not in self.code_addresses()):
            return ""
        erl = 1 if m.status & C.ST_ERL else 0
        return (f" — the program is loaded at this address, but with the fixed "
                f"mapping MMU kuseg is at PA = VA + 0x40000000 while Status.ERL = 0 "
                f"and at PA = VA while ERL = 1: ERL is {erl} now and was "
                f"{1 - erl} when the program was loaded (an eret in the reset "
                "state clears ERL)")

    def code_addresses(self) -> set[int]:
        """Addresses of the program's instructions (.text and .ktext) and
        of the boot firmware's (the TLB refill handler)."""
        if self.program is None:
            return set()
        if self._code is None:
            fw = self.machine.firmware if self.machine is not None else None
            self._code = set(self.program.source_map) | (set(fw.source_map) if fw else set())
        return self._code

    def program_address(self, pc: int | None, cpu: int | None = None) -> int | None:
        """``pc`` if the program has an instruction there; else the program
        address of the same physical word as core ``cpu`` maps them now (a
        core running the program through kseg0 / kseg1, or kuseg moved by
        the fixed mapping); else None."""
        if pc is None or self.program is None:
            return None
        code = self.code_addresses()
        if pc in code:
            return pc
        m = self.machines[self.cpu if cpu is None else cpu]
        try:
            pa = m.physical(pc)
        except (SimException, ValueError):
            return None
        for va in (pa, (pa - 0x40000000) & M32, pa | 0x80000000, pa | 0xA0000000):
            if va in code:
                try:
                    if m.physical(va) == pa:
                        return va
                except (SimException, ValueError):
                    pass
        return None

    def source_line(self, pc: int | None, cpu: int | None = None) -> SourceLine | None:
        """The source line of the instruction at ``pc`` (or of the program
        address it aliases, :meth:`program_address`)."""
        a = self.program_address(pc, cpu)
        return self.program.source_map.get(a) if a is not None else None

    def at_end(self) -> bool:
        """The next instruction to execute (on a running core) is outside
        the program (and not a delay slot / exception handler entry
        already decided)."""
        if self.machine is None or self.program is None:
            return False

        def outside(pc: int, cpu: int) -> bool:   # a vector: the hint explains
            return pc not in self.VECTORS and self.program_address(pc, cpu) is None

        for i in self.running():
            m, p = self.machines[i], self.pipelines[i]
            if p is None:
                if not m.slot_armed and not m.halted and outside(m.pc, i):
                    return True
            else:
                nxt = p.entering_ex()
                if nxt is not None and outside(nxt.pc, i):
                    return True
        return False

    def step_over(self) -> StepResult:
        """Step, but run a whole call (JAL / JALR / BAL / BGEZAL ... /
        R6 BALC, JIALC ...) as one step: stops at the return address."""
        pc = self._stop_pc()
        ret = self._return_address(pc) if pc is not None else None
        if ret is None:
            return self.step()
        saved, self.run_to = self.run_to, ret
        r = self.run()
        if r.stopped == "cursor":
            r.stopped = ""
        self.run_to = saved if r.stopped != "" else None
        return r

    def _return_address(self, pc: int) -> int | None:
        m = self.machine
        try:
            dec = m.isa.decode(m.code_word(pc))
        except (SimException, ISADecodeError):
            return None
        name = dec.name
        if name in _COMPACT_CALLS:
            return (pc + 4) & M32                 # compact: no delay slot
        if name in _CALLS:
            return (pc + 8) & M32                 # after the delay slot
        return None

    # -- source-level debugging (M13 phase 3) -----------------------------

    def _watch_calls(self, cpu: int, m: Machine) -> None:
        """Keep core ``cpu``'s call stack: a linking jump / branch that was
        taken pushes a frame, ``jr $ra`` (R6: ``jalr $zero, $ra``, ``jrc
        $ra``) pops one — what the code does, not a guess from the stack
        memory."""
        def hook(pc: int, dec, word: int, result, cpu=cpu) -> None:
            name = dec.name
            stack = self.calls[cpu]
            if name in _CALLS or name in _COMPACT_CALLS:
                if result is not None and result >= 0:         # taken
                    ret = (pc + (4 if name in _COMPACT_CALLS else 8)) & M32
                    stack.append(Frame(pc, result & M32, ret, m.read_reg(29)))
                    if len(stack) > 10_000:
                        del stack[:1000]
            elif stack and (word >> 21) & 31 == 31 and (
                    name in ("JR", "JR.HB") or (name in ("JALR", "JALR.HB", "JALR@R6")
                                                and (word >> 11) & 31 == 0)
                    or name == "JIC" and not word & 0xFFFF):
                stack.pop()
        m.call_hook = hook

    def _line_key(self, addr: int | None) -> tuple[str, int] | None:
        """The source line of an instruction: (file, line) of a compiled
        program, ("", listing / source line) of an assembled one."""
        a = self.program_address(addr) if addr is not None else None
        if a is None:
            return None
        if self.program.c_lines:
            return self.program.c_lines.get(a)
        src = self.program.source_map.get(a)
        return ("", src.line) if src is not None else None

    def _stmt_start(self, addr: int) -> bool:
        """Does a statement (a line, or an assembly source line) start at
        ``addr``?"""
        a = self.program_address(addr)
        if a is None:
            return False
        if self.program.c_stmts:
            key = self.program.c_lines.get(a)
            return key is not None and a in self.program.c_stmts.get(key, ())
        prev = self.program.source_map.get((a - 4) & M32)
        here = self.program.source_map.get(a)
        return here is not None and (prev is None or prev.line != here.line)

    def step_line(self, over: bool = False, limit: int = 1_000_000) -> StepResult:
        """Run until the next source line starts (gdb's ``step`` /
        ``next``): the PC leaves the addresses of the current line and
        reaches the start of a statement — of any line, the same one too (a
        loop).  ``over``: a call runs as one step.  Code without line
        information (the firmware, a handler, library code built without
        -g) is run through."""
        start_pc = self._stop_pc()
        key = self._line_key(start_pc)
        total = StepResult()
        if start_pc is None:
            return self.step()
        here = {start_pc}
        while total.steps < limit:
            r = self.step_over() if over else self.step()
            total.steps += r.steps
            total.trace = r.trace or total.trace
            total.exceptions += r.exceptions
            if r.stopped or r.error:
                total.stopped, total.error = r.stopped, r.error
                return total
            pc = self._stop_pc()
            if pc is None:
                continue                        # the pipeline is filling
            k = self._line_key(pc)
            if k is None:
                continue
            if k == key and pc in here:
                continue
            if k == key and not self._stmt_start(pc):
                here.add(pc)
                continue
            if self._stmt_start(pc) or k != key:
                return total
        total.stopped = "limit"
        return total

    def finish(self, limit: int = 10_000_000) -> StepResult:
        """Run until the current function returns to its caller (its frame
        on the call stack is popped and the caller's next instruction is
        reached).  Without a frame (main called by crt0 is one; code
        reached without a call): an ordinary run."""
        stack = self.calls[self.cpu]
        if not stack:
            return self.run(limit)
        depth = len(stack)
        ret = stack[-1].ret
        total = StepResult()
        saved = self.run_to
        while total.steps < limit:
            self.run_to = ret
            r = self.run(limit - total.steps)
            total.steps += r.steps
            total.exceptions += r.exceptions
            total.trace = r.trace or total.trace
            if r.stopped == "cursor" and len(self.calls[self.cpu]) >= depth:
                continue                        # the same return address, deeper
            total.stopped = "" if r.stopped == "cursor" else r.stopped
            total.error = r.error
            break
        self.run_to = saved if total.stopped else None
        return total

    def function_at(self, addr: int | None) -> str | None:
        """The function an instruction belongs to: an ELF's STT_FUNC symbol,
        else the closest label at or before it in the code."""
        if addr is None or self.program is None:
            return None
        a = self.program_address(addr)
        a = addr if a is None else a
        for start, end, name in self.program.functions:
            if start <= a < end:
                return name
        if self.program.functions:
            return None
        best = None
        code = self.code_addresses()
        for name, value in self.program.symbols.items():
            if value in code and value <= a and (best is None or value > best[1]):
                best = (name, value)
        return best[0] if best else None

    def backtrace(self, cpu: int | None = None) -> list[dict[str, Any]]:
        """The call stack, innermost first: for each frame the function, the
        PC (the next instruction / the call site), its source line."""
        cpu = self.cpu if cpu is None else cpu
        out = []
        pc = self._stop_pc(cpu)
        frames = [pc] + [f.site for f in reversed(self.calls[cpu])] if pc is not None else []
        for k, a in enumerate(frames):
            out.append({"level": k, "pc": a, "function": self.function_at(a),
                        "line": self.source_position(self.program_address(a, cpu) or a)
                        if self.program is not None and self.program.c_lines else None,
                        "source": self.source_line(a, cpu)})
        return out

    def line_addresses(self, file: str, line: int) -> list[int]:
        """Where a source line's statements start (a breakpoint at
        ``file:line``); the next line with code when it has none.  ``file``
        matches by name (``""``: the only / first file)."""
        prog = self.program
        if prog is None:
            return []
        if not prog.c_stmts:
            a = self.line_address(line)
            return [a] if a is not None else []
        name = Path(file).name
        lines = sorted(k for k in prog.c_stmts if not name or Path(k[0]).name == name)
        for f, n in lines:
            if n >= line and (not name or Path(f).name == name):
                same = [k for k in lines if k[1] == n]
                return sorted(a for k in same for a in prog.c_stmts[k])
        return []

    def default_source_file(self) -> str:
        """The file a bare line number means: the one the program is in —
        main's while it is still in the start-up code (as gdb)."""
        if self.program is None or not self.program.c_lines:
            return ""
        pc = self._stop_pc()
        main = self.program.symbols.get("main")
        entry_fn = self.function_at(self.program.entry)
        where = self._line_key(pc) if pc is not None else None
        if main is not None and (where is None or self.function_at(pc) == entry_fn):
            where = self.program.c_lines.get(main, where)
        return where[0] if where else ""

    def source_file(self, addr: int | None = None) -> tuple[str, list[str]] | None:
        """The source file of a compiled program's instruction (default: the
        selected core's next one) and its text, for a source view."""
        if self.program is None or not self.program.c_lines:
            return None
        pc = self._stop_pc() if addr is None else addr
        where = self._line_key(pc) if pc is not None else None
        if where is None:
            files = sorted({f for f, _n in self.program.c_lines.values()})
            if not files:
                return None
            main = self.program.symbols.get("main")
            f0 = self.program.c_lines.get(main, (files[0], 0))[0] if main else files[0]
            where = (f0, 0)
        sf = _SourceFiles.shared()
        sf.line(where[0], 1)
        return where[0], sf.files.get(where[0], [])

    def run(self, max_steps: int = 100_000) -> StepResult:
        """Step until a breakpoint, exit, error, input request or
        ``max_steps`` (the GUI calls this in slices)."""
        total = StepResult()
        for i in range(max_steps):
            if self.budget is not None:
                if self.budget <= 0:
                    total.stopped = "limit"
                    self.budget = None
                    return total
                self.budget -= 1
            if self.stop_at_end and self.at_end():
                total.stopped = "end"
                return total
            r = self.step()
            total.steps += r.steps
            total.trace = r.trace or total.trace
            if r.stopped or r.error:
                total.stopped, total.error = r.stopped, r.error
                return total
            if self.at_breakpoint():
                pc = self._stop_pc()
                total.stopped = "cursor" if pc == self.run_to else "breakpoint"
                self.run_to = None
                return total
        return total

    def _stop_pc(self, cpu: int | None = None) -> int | None:
        """The instruction a breakpoint is checked against: the next to
        execute (instruction mode) / the one about to enter EX."""
        if self.machine is None:
            return None
        cpu = self.cpu if cpu is None else cpu
        p = self.pipelines[cpu]
        if p is None:
            return self.machines[cpu].pc
        nxt = p.entering_ex()
        return nxt.pc if nxt is not None else None

    def _entering_ex(self):
        """Pipeline mode: the instruction that executes in the next cycle
        (5-stage: in EX, not executed yet; 24K: in AG), or None."""
        return self.pipeline.entering_ex()

    def at_breakpoint(self) -> bool:
        """A running core is at a breakpoint (or the run-to target): that
        core becomes the selected one."""
        if self.machine is None:
            return False
        for i in [self.cpu] + [c for c in self.running() if c != self.cpu]:
            pc = self._stop_pc(i)
            if pc is None:
                continue
            a = self.program_address(pc, i)
            if (pc in self.breakpoints or pc == self.run_to
                    or a is not None and a != pc and (a in self.breakpoints or a == self.run_to)):
                if i != self.cpu:
                    self.select(i)
                return True
        return False

    def line_address(self, line: int) -> int | None:
        """First instruction address of source line ``line``, or of the
        next line that has code (run to cursor)."""
        if self.program is None:
            return None
        best = None
        for a, src in self.program.source_map.items():
            if src.line >= line and (best is None or (src.line, a) < best):
                best = (src.line, a)
        return best[1] if best else None

    def step_back(self) -> bool:
        """Undo one step by replaying from reset (deterministic)."""
        if self.steps == 0:
            return False
        target = self.steps - 1
        self.reset()
        for _ in range(target):
            if self.step().steps == 0:
                break
        return True

    def toggle_breakpoint(self, addr: int) -> bool:
        if addr in self.breakpoints:
            self.breakpoints.discard(addr)
            return False
        self.breakpoints.add(addr)
        return True

    # -- state for the views -----------------------------------------------------

    def registers(self) -> list[tuple[str, int, bool]]:
        """``(name, value, changed by the last step)`` for the 32 GPRs."""
        m = self.machine
        return [(gpr_name(i, self.abi), m.read_reg(i), m.read_reg(i) != self.prev_regs[i])
                for i in range(32)]

    def shadow_set(self, n: int) -> list[tuple[str, int]]:
        """``(name, value)`` of GPR set ``n`` of the selected core (set 0 =
        the normal GPRs; ``SRSCtl.CSS`` is the one instructions use)."""
        m = self.machine
        if not 0 <= n < len(m.shadow):
            raise ValueError(f"there is no shadow set {n} (sets 0-{len(m.shadow) - 1})")
        bank = m.shadow[n]
        return [(gpr_name(i, self.abi), bank[i] & 0xFFFFFFFF if i else 0) for i in range(32)]

    def interrupt_state(self) -> dict[str, Any]:
        """The interrupt system of the selected core (Vol. III 6.1): the
        mode, IntCtl.VS spacing, the vector the next interrupt would take,
        the shadow set fields and the SRSMap mapping."""
        m = self.machine
        status, cause = m.status, m.cp0_read(*C.CAUSE)
        intctl = m.cp0_read(*C.INTCTL)
        mode = m.interrupt_mode()
        pending = (cause >> 8) & (status >> 8) & 0xFF
        vec = (C.vi_vector(status, cause) if mode == "vectored" else
               m.irq_lines or -1 if mode == "eic" else None)
        srsmap = m.cp0_read(*C.SRSMAP)
        return {"mode": mode, "spacing": ((intctl >> 5) & 0x1F) << 5,
                "pending": pending, "enabled": bool(status & C.ST_IE)
                and not status & (C.ST_EXL | C.ST_ERL)
                and (mode != "eic" or m.irq_lines > (status >> 10) & 0x3F),
                # EIC mode: what the controller requests now, what was taken last
                "eic_present": bool(m.cp0_read(*C.CONFIG3) & C.CONFIG3_VEIC),
                "ripl": m.irq_lines if mode == "eic" else None, "eic_ss": m.eic_ss,
                "ipl": (status >> 10) & 0x3F,
                "cause_ripl": (cause >> 10) & 0x3F, "eicss": m.srs_field(C.SRS_EICSS),
                "vector_number": vec if vec is not None and vec >= 0 else None,
                "vector": (C.vector(status, m.cp0_read(*C.EBASE),
                                    offset=C.vi_offset(vec, intctl), bev_base=m.bev_base)
                           if vec is not None and vec >= 0 else None),
                "sets": len(m.shadow), "css": m.srs_field(C.SRS_CSS),
                "pss": m.srs_field(C.SRS_PSS), "ess": m.srs_field(C.SRS_ESS),
                "srsmap": [(srsmap >> (4 * v)) & 0xF for v in range(8)]}

    # -- EJTAG (emips.ejtag) ---------------------------------------------------------

    def debug_interrupt(self) -> None:
        """A request on the selected core's DINT input (as a probe's
        EjtagBrk): a Debug Interrupt exception before its next
        instruction (ignored in Debug Mode).  Recorded: a step back
        replays it."""
        if self.machine is None:
            return
        self._dints.append((self.steps, self.cpu))
        self.machine.debug_interrupt()

    def debug_write(self, kind: str, key: Any, value: int) -> None:
        """The debugger sets a debug register of the selected core, as a
        probe does (through a debug handler in dmseg): a drseg register
        (``kind`` "drseg", ``key`` = its offset: the hardware breakpoints)
        or a CP0 register ("cp0", ``key`` = (reg, sel): the watch
        registers).  Recorded: a step back replays it."""
        if self.machine is None:
            return
        self._pokes.append((self.steps, self.cpu, kind, key, value))
        self._poke(self.machine, kind, key, value)

    @staticmethod
    def _poke(m: Machine, kind: str, key: Any, value: int) -> int:
        tap = m.ejtag.tap
        if kind == "drseg":
            m.ejtag.set_register(key, value)
        elif kind == "cp0":
            m.cp0_write(*key, value)
        elif kind == "ejtagbrk":                # the probe's EjtagBrk
            if tap is not None and tap.probe is not None:
                tap.probe.debug_break(tap)
        elif kind == "fastdata":                # words for FASTDATA downloads
            if tap is not None and tap.probe is not None:
                tap.probe.download.extend(value)
        elif kind == "tap-ir":
            return tap.scan_ir(value)
        elif kind == "tap-dr":
            return tap.scan_dr(value, key)
        elif kind == "tap-reset":
            tap.reset_logic()
        elif kind == "trigger-in":              # the probe's trigger to TR_TRIGIN
            tap.tcb.probe_trigger()
            m.trace_settings_changed()
        return 0

    # -- the probe (emips.tap) ------------------------------------------------------

    def set_probe(self, on: bool, *, trap: bool = False, boot: bool = False) -> None:
        """Connect (or remove) an EJTAG probe on every core's TAP: ProbEn,
        with ``trap`` ProbTrap (the debug handler comes from the probe's
        memory at 0xFF200200), with ``boot`` the EJTAGBOOT instruction (a
        debug interrupt right after reset).  The program restarts; what
        was loaded into the probe's memory stays."""
        if on:
            image = self.probe.image if self.probe is not None else {}
            self.probe = TAPM.Probe(trap=trap or boot)
            self.probe.image = image
            self.probe_boot = boot
        else:
            self.probe = None
            self.probe_boot = False
        if self.program is not None:
            self.reset()

    def probe_load(self, source: str) -> int:
        """Assemble ``source`` and put its bytes in dmseg (0xFF200000 -
        0xFF2FFFFF, e.g. ``.ktext 0xff200200``) into the probe's memory;
        the program restarts.  Returns the number of bytes."""
        if self.probe is None:
            raise ValueError("no probe (probe on first)")
        prog = assemble(source, endian=self.endian, release=self.release,
                        pseudo=self.options.pseudo, reorder=self.options.reorder)
        n = self.probe.load(prog.image)
        if not n:
            raise ValueError("nothing in dmseg (0xff200000-0xff2fffff): put the handler "
                             "at .ktext 0xff200200")
        if self.program is not None:
            self.reset()
        return n

    def probe_event(self, kind: str, key: Any = None, value: Any = 0) -> int:
        """Something the probe does now on the selected core's TAP —
        "ejtagbrk", "fastdata" (words), or a raw scan "tap-ir" / "tap-dr"
        (``key`` = length) / "tap-reset"; recorded for step back.  Returns
        what a scan shifted out."""
        if self.machine is None:
            return 0
        self._pokes.append((self.steps, self.cpu, kind, key, value))
        return self._poke(self.machine, kind, key, value)

    def probe_lines(self) -> list[str]:
        """The TAP and the probe of the selected core (the debugger's
        ``probe``, the Debug tab)."""
        m = self.machine
        tap = m.ejtag.tap
        if tap is None:
            return ["no TAP"]
        ecr = tap.ecr()
        bits = [n for b, n in ((TAPM.ECR_ROCC, "Rocc"), (TAPM.ECR_PRACC, "PrAcc"),
                               (TAPM.ECR_PROBEN, "ProbEn"), (TAPM.ECR_PROBTRAP, "ProbTrap"),
                               (TAPM.ECR_EJTAGBRK, "EjtagBrk"), (TAPM.ECR_DOZE, "Doze"),
                               (TAPM.ECR_HALT, "Halt"), (TAPM.ECR_DM, "DM")) if ecr & b]
        out = [f"TAP: {tap.state}, IR {TAPM.INSTRUCTIONS.get(tap.ir, 'BYPASS')} "
               f"(0x{tap.ir:02x}), {tap.tck} TCK; IDCODE 0x{TAPM.DEVICE_ID:08x}, "
               f"IMPCODE 0x{tap.impcode():08x}"
               + (", EJTAGBOOT given" if tap.ejtagboot else ""),
               f"ECR 0x{ecr:08x}: " + (", ".join(bits) or "-")]
        p = self.probe
        if p is None:
            out.append("probe: none (dmseg accesses wait forever; probe on)")
        else:
            out.append(f"probe: connected{' (ProbTrap: the debug handler at 0xff200200)' if p.trap else ''}"
                       f"{', EJTAGBOOT' if self.probe_boot else ''}; EJTAG memory "
                       f"{len(p.image)} bytes loaded, {p.accesses} processor accesses")
            for kind, va, size, data, *how in list(p.log)[-4:]:
                out.append(f"   {kind:<5} 0x{va:08x} ({size} byte{'s' if size > 1 else ''}) "
                           f"data 0x{data:08x}" + (" FASTDATA" if how else ""))
            if p.download or p.upload:
                out.append(f"   FASTDATA: {len(p.download)} words to download, uploaded "
                           + (" ".join(f"0x{w:08x}" for w in p.upload[-6:]) or "none"))
        ej = m.ejtag
        new = "new" if ej.pc_sample & 1 else "read"
        out.append(f"PC sampling every {ej.sample_period} cycles (DCR.PCR): last sample "
                   f"0x{(ej.pc_sample >> 1) & 0xFFFFFFFF:08x} ({new})"
                   + (f", {len(p.samples)} collected" if p is not None and p.samples else ""))
        return out

    def profile_rows(self, top: int = 10) -> list[tuple[int, int, str]]:
        """The PC samples the probe collected (``probe.profile``), most
        frequent first: (count, address, where)."""
        if self.probe is None:
            return []
        counts: dict[int, int] = {}
        for pc, _asid in self.probe.samples:
            counts[pc] = counts.get(pc, 0) + 1
        rows = []
        for pc, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
            src = self.source_line(pc)
            c = self.source_position(pc)
            where = (f"{Path(c[0]).name}:{c[1]}" if c is not None else
                     f"line {src.line}" if src is not None else "")
            fn = self.function_at(pc)
            rows.append((n, pc, f"{fn or '?'}  {where}".rstrip()))
        return rows

    # -- MIPS Trace (emips.pdtrace): the TCB through TAP scans ---------------------------

    def _tcb_scan(self, instruction: int, value: int) -> int:
        self.probe_event("tap-ir", None, instruction)
        return self.probe_event("tap-dr", None, value)

    def trace_start(self, *, modes: str = "UKSED", mode: int = PDT.MODE_PC, ca: bool = False,
                    trace_from: bool = False, asid: int | None = None, tb: bool = False,
                    io: bool = False, sync: int = 0, tlsif: bool = False,
                    keep: bool = False, wait: bool = False) -> None:
        """Turn tracing on in the selected core's TCB, as a trace probe does
        (TAP scans, recorded for step back): TCBCONTROLC.Mode (``mode``:
        PC + MODE_LA / SA / LD / SD), TCBCONTROLA (On, the ``modes`` traced
        — U S K E D —, ``asid`` or G, TB, IO, SyP = ``sync``), TCBCONTROLB
        (EN — off with ``wait``: a start trigger turns it on —, CA, TM
        Trace-To or ``trace_from``, TLSIF; TR empties the trace memory unless
        ``keep``)."""
        a = PDT.TA_ON | ((sync & 7) << PDT.TA_SYP_SHIFT)
        for k, bit in (("U", PDT.TA_U), ("S", PDT.TA_S), ("K", PDT.TA_K), ("E", PDT.TA_E),
                       ("D", PDT.TA_D)):
            if k in modes.upper():
                a |= bit
        a |= PDT.TA_G if asid is None else (asid & 0xFF) << PDT.TA_ASID_SHIFT
        if tb:
            a |= PDT.TA_TB
        if io:
            a |= PDT.TA_IO
        if tlsif:
            a |= PDT.TA_TFCR | PDT.TA_TIM | PDT.TA_TLSM
        b = PDT.TB_WE | (0 if wait else PDT.TB_EN) | (int(trace_from) << PDT.TB_TM_SHIFT)
        if ca:
            b |= PDT.TB_CA
        if tlsif:
            b |= PDT.TB_TLSIF
        if not keep:
            b |= PDT.TB_TR
        self._tcb_scan(PDT.TCBCONTROLC, (mode & 0x1F) << PDT.TC_MODE_SHIFT)
        self._tcb_scan(PDT.TCBCONTROLA, a)
        self._tcb_scan(PDT.TCBCONTROLB, b)

    def trace_stop(self) -> None:
        """TCBCONTROLB.EN = 0 (the last trace word is written) and
        TCBCONTROLA.On = 0."""
        tcb = self.machine.ejtag.tap.tcb
        b = tcb.control_b & ~(PDT.TB_EN | PDT.TB_TR | PDT.TB_RM | PDT.TB_WR | PDT.TB_BF)
        self._tcb_scan(PDT.TCBCONTROLB, b | PDT.TB_WE)
        self._tcb_scan(PDT.TCBCONTROLA, tcb.control_a & ~PDT.TA_ON)

    def trace_trigger(self, n: int, value: int) -> None:
        """Write TCBTRIG``n`` (TCBCONTROLB.REG = 16 + n, WR, then TCBDATA)."""
        tcb = self.machine.ejtag.tap.tcb
        keep = tcb.control_b & ~((0x1F << PDT.TB_REG_SHIFT) | PDT.TB_WR | PDT.TB_TR
                                 | PDT.TB_RM | PDT.TB_BF)
        self._tcb_scan(PDT.TCBCONTROLB, keep | PDT.TB_WE | ((16 + n) << PDT.TB_REG_SHIFT)
                       | PDT.TB_WR)
        self._tcb_scan(PDT.TCBDATA, value)
        self._tcb_scan(PDT.TCBCONTROLB, keep | PDT.TB_WE)

    def trace_words(self) -> list[int]:
        """The trace memory, oldest word first, and the word the TCB is
        filling (what EN = 0 would write) — read directly, as views do
        (reading TCBTW through the TAP moves TCBRDP)."""
        tcb = self.machine.ejtag.tap.tcb
        n = len(tcb.words)
        words = [tcb.words[(tcb.stp // 8 + i) % n] for i in range(tcb.count)]
        if tcb.used and not tcb.control_b & PDT.TB_BF:
            words.append(tcb.tw | tcb.type)
        return words

    def trace_formats(self) -> list[PDT.Format]:
        tcb = self.machine.ejtag.tap.tcb
        return PDT.decode_words(self.trace_words(), ca=tcb.ca, opt=tcb.tlsif)

    def trace_steps(self) -> list[Any]:
        """Trace regeneration over the trace words and the program image:
        :class:`emips.pdtrace.Step` per instruction, strings for events."""
        m = self.machine
        st = m.pdt.settings()

        def word(a: int) -> int | None:
            try:
                return m.code_word(a)
            except SimException:
                return None
        return PDT.regenerate(self.trace_formats(), word, m.isa, tb=st["tb"],
                              mode=st["mode"], vector=PDT.exception_vectors(m))

    def trace_state(self) -> list[str]:
        """The trace settings and the TCB (the debugger's ``trace``)."""
        m = self.machine
        tcb = m.ejtag.tap.tcb
        st = m.pdt.settings()
        what = "+".join(n for b, n in ((PDT.MODE_PC, "PC"), (PDT.MODE_LA, "load addresses"),
                                       (PDT.MODE_SA, "store addresses"),
                                       (PDT.MODE_LD, "load data"), (PDT.MODE_SD, "store data"))
                        if st["mode"] & b)
        who = "software (TraceControl.TS)" if st["ts"] else "the TCB (TCBCONTROLA)"
        out = [f"trace {'on' if st['on'] else 'off'}, controlled by {who}: modes "
               f"{st['modes'] or '-'}, " + ("any ASID" if st["g"] else f"ASID {st['asid']}")
               + f"; {what}" + (", TB" if st["tb"] else "") + (", IO" if st["io"] else "")
               + f"; sync every {st['sync']} cycles"]
        tm = ("Trace-To", "Trace-From")[(tcb.control_b >> PDT.TB_TM_SHIFT) & 1]
        out.append(f"TCB: {'recording' if tcb.enabled else 'not recording'} (EN), {tm}"
                   + (", cycle-accurate" if tcb.ca else "") + (", TLSIF" if tcb.tlsif else "")
                   + f"; {tcb.count} of {len(tcb.words)} trace words"
                   + (" (full: BF)" if tcb.control_b & PDT.TB_BF else "")
                   + f", {tcb.formats} formats" + (f", {tcb.used} bits in the open word"
                                                  if tcb.used else ""))
        out.append(f"   TCBCONTROLA 0x{tcb.control_a:08x}  TCBCONTROLB 0x{tcb.control_b:08x}  "
                   f"TCBCONTROLC 0x{tcb.control_c:08x}  TCBCONFIG 0x{tcb.config():08x}")
        for i in range(PDT.TRIGGERS):
            v = tcb._trig_value(i)
            if not v & (PDT.TR_DM | PDT.TR_PDTRI | PDT.TR_CHTRI):
                continue
            src = "+".join(n for b, n in ((PDT.TR_DM, "Debug Mode entry"),
                                          (PDT.TR_PDTRI, "the probe (TR_TRIGIN)"),
                                          (PDT.TR_CHTRI, "the chip")) if v & b)
            kind = ("start", "end", "about", "info")[(v >> PDT.TR_TYPE_SHIFT) & 3]
            out.append(f"   trigger {i}: {src} -> {kind}"
                       + (f", TF6 info 0x{v >> 24:02x}" if v & PDT.TR_TRACE else "")
                       + (", once" if v & PDT.TR_FO else "")
                       + (", to the probe (TR_TRIGOUT)" if v & PDT.TR_PDTRO else "")
                       + (" — fired" if v & PDT.TR_TR else ""))
        if m.pdt.fifo:
            out.append(f"   PDtrace FIFO: {len(m.pdt.fifo)} records waiting")
        return out

    def trace_lines(self, last: int = 20) -> list[str]:
        """The regenerated instructions (the last ``last``) with their
        addresses, data and the trace events (the debugger's ``trace
        show``)."""
        items = self.trace_steps()
        if last:
            items = items[-last:]
        m = self.machine
        out = []
        for it in items:
            if isinstance(it, str):
                out.append(f"   -- {it}")
                continue
            try:
                word = m.code_word(it.pc)
                text = disassemble(word, it.pc, isa=m.isa)
            except (SimException, ISADecodeError):
                text = "?"
            src = self.source_line(it.pc)
            line = f"   0x{it.pc:08x}  {PDT.INSCOMP_NAMES[it.inscomp]:<3} {text:<28}"
            extra = []
            if it.load is not None:
                extra.append(f"load 0x{it.load:08x}")
            if it.store is not None:
                extra.append(f"store 0x{it.store:08x}")
            extra += [f"data 0x{d:x}" for d in it.data]
            if it.fcr:
                extra.append("call/return")
            if it.im:
                extra.append("I-miss")
            if src is not None and not extra:
                extra.append(f"line {src.line}")
            out.append((line + "  ".join(extra)).rstrip())
        return out

    def breakpoint_state(self) -> dict[str, Any]:
        """The hardware breakpoints (drseg) and the watch registers of the
        selected core: the enabled ones and the status bits."""
        m = self.machine
        ej = m.ejtag
        ib = [{"n": n, "address": ej.iba[n], "mask": ej.ibm[n],
               "asid": ej.ibasid[n] if ej.ibc[n] & EJ.BC_ASIDUSE else None,
               "break": bool(ej.ibc[n] & EJ.BC_BE), "trigger": bool(ej.ibc[n] & EJ.BC_TE),
               "hit": bool(ej.ibs_bs >> n & 1)}
              for n in range(EJ.IB_COUNT) if ej.ibc[n] & (EJ.BC_BE | EJ.BC_TE)]
        db = []
        for n in range(EJ.DB_COUNT):
            c = ej.dbc[n]
            if not c & (EJ.BC_BE | EJ.BC_TE):
                continue
            blm, bai = (c >> EJ.DBC_BLM_SHIFT) & 0xFF, (c >> EJ.DBC_BAI_SHIFT) & 0xFF
            db.append({"n": n, "address": ej.dba[n], "mask": ej.dbm[n],
                       "asid": ej.dbasid[n] if c & EJ.BC_ASIDUSE else None,
                       "load": not c & EJ.DBC_NOLB, "store": not c & EJ.DBC_NOSB,
                       "value": None if (blm | bai) & 0xF == 0xF else ej.dbv[n],
                       "blm": blm, "bai": bai,
                       "break": bool(c & EJ.BC_BE), "trigger": bool(c & EJ.BC_TE),
                       "hit": bool(ej.dbs_bs >> n & 1)})
        watch = []
        for i, (lo_key, hi_key) in enumerate(zip(C.WATCHLO, C.WATCHHI)):
            lo, hi = m.cp0_read(*lo_key), m.cp0_read(*hi_key)
            if lo & 7:
                watch.append({"n": i, "address": lo & ~7 & 0xFFFFFFFF, "mask": hi & 0xFF8,
                              "kinds": "".join(k for b, k in ((4, "I"), (2, "R"), (1, "W"))
                                               if lo & b),
                              "asid": None if hi & C.WATCHHI_G else (hi >> 16) & 0xFF,
                              "hit": "".join(k for b, k in ((4, "I"), (2, "R"), (1, "W"))
                                             if hi & b)})
        return {"ib": ib, "ibs_bs": ej.ibs_bs, "db": db, "dbs_bs": ej.dbs_bs,
                "watch": watch, "wp": bool(m.cp0_read(*C.CAUSE) & C.CA_WP),
                "impr": m.ddb_impr}

    def breakpoint_lines(self) -> list[str]:
        """:meth:`breakpoint_state` as text (the debugger's ``ejtag``, the
        Debug tab)."""
        st = self.breakpoint_state()
        out = []
        if st["ib"] or st["ibs_bs"]:
            out.append(f"instruction breakpoints (IBS.BS 0x{st['ibs_bs']:x}):")
            for b in st["ib"]:
                out.append(f"   IB{b['n']} 0x{b['address']:08x}"
                           + (f" mask 0x{b['mask']:08x}" if b["mask"] else "")
                           + (f" ASID {b['asid']}" if b["asid"] is not None else "")
                           + " " + "+".join(x for x, on in (("break", b["break"]),
                                                            ("trigger", b["trigger"])) if on)
                           + ("  (matched)" if b["hit"] else ""))
        if st["db"] or st["dbs_bs"]:
            out.append(f"data breakpoints (DBS.BS 0x{st['dbs_bs']:x}):")
            for b in st["db"]:
                kinds = "load/store" if b["load"] and b["store"] else \
                    "load" if b["load"] else "store" if b["store"] else "never"
                out.append(f"   DB{b['n']} 0x{b['address']:08x}"
                           + (f" mask 0x{b['mask']:08x}" if b["mask"] else "")
                           + f" {kinds}"
                           + (f" value 0x{b['value']:08x} (imprecise)" if b["value"] is not None
                              else "")
                           + (f" ASID {b['asid']}" if b["asid"] is not None else "")
                           + " " + "+".join(x for x, on in (("break", b["break"]),
                                                            ("trigger", b["trigger"])) if on)
                           + ("  (matched)" if b["hit"] else ""))
        if st["impr"]:
            out.append("an imprecise data break is due (DDBLImpr / DDBSImpr)")
        for w in st["watch"]:
            out.append(f"watch {w['n']}: 0x{w['address']:08x}"
                       + (f" mask 0x{w['mask']:03x}" if w["mask"] else "")
                       + f" {w['kinds']}"
                       + (f" ASID {w['asid']}" if w["asid"] is not None else "")
                       + (f"  (matched {w['hit']})" if w["hit"] else ""))
        if st["wp"]:
            out.append("Cause.WP: a deferred watch exception waits for EXL = ERL = 0")
        if not out:
            out.append(f"no hardware breakpoints ({EJ.IB_COUNT} instruction, {EJ.DB_COUNT} "
                       f"data in drseg) nor watch registers ({C.WATCH_PAIRS} pairs) set")
        return out

    def debug_state(self) -> dict[str, Any]:
        """The EJTAG state of the selected core (MD00047 2.7, ch. 3)."""
        m = self.machine
        d = m.cp0_read(*C.DEBUG)
        dcr = m.ejtag.read_dcr()
        return {"dm": m.debug_mode, "debug": d, "depc": m.cp0_read(*C.DEPC),
                "desave": m.cp0_read(*C.DESAVE), "dcr": dcr,
                "causes": [name for bit, name in EJ.CAUSE_NAMES if d & bit],
                "dexccode": (d >> EJ.DEXC_SHIFT) & 0x1F, "dbd": bool(d & EJ.DBD),
                "sst": bool(d & EJ.SST), "lsnm": bool(d & EJ.LSNM),
                "iexi": bool(d & EJ.IEXI), "countdm": bool(d & EJ.COUNTDM),
                "halt": bool(d & EJ.HALT), "doze": bool(d & EJ.DOZE),
                "held": [name for bit, name in ((EJ.DBUSEP, "data bus error"),
                                                (EJ.CACHEEP, "cache error")) if d & bit],
                "int_enabled": bool(dcr & EJ.DCR_INTE), "nmi_enabled": bool(dcr & EJ.DCR_NMIE),
                "nmi_pending": bool(dcr & EJ.DCR_NMIP), "probe": bool(dcr & EJ.DCR_PE),
                "vector": m.debug_vector(), "step_due": m.ss_pending,
                "dint_pending": m.dint_pending}

    def debug_lines(self) -> list[str]:
        """:meth:`debug_state` as text (the debugger's ``ejtag``, the
        register pane's Debug tab)."""
        st = self.debug_state()
        slot = " (in a delay slot: DEPC is its branch, Debug.DBD)" if st["dbd"] else ""
        if st["dm"]:
            out = ["Debug Mode (Debug.DM = 1): kernel rights, dseg at 0xFF200000-0xFF3FFFFF, "
                   "interrupts and NMIs wait"]
            if st["causes"]:
                out.append(f"   entered by {', '.join(st['causes'])} (DEPC 0x{st['depc']:08x})"
                           + slot)
            else:
                code = st["dexccode"]
                name = "CacheErr" if code == EJ.DEXC_CACHEERR else C.EXC_NAMES.get(code, str(code))
                out.append(f"   re-entered by an exception in Debug Mode: DExcCode {code} "
                           f"({name}), DEPC 0x{st['depc']:08x}{slot}")
            if st["halt"] or st["doze"]:
                out.append("   the core was asleep (WAIT): Debug.Halt / Doze")
        else:
            out = ["Non-Debug Mode (Debug.DM = 0)"]
            if st["causes"]:
                out.append(f"   last debug exception: {', '.join(st['causes'])} (DEPC "
                           f"0x{st['depc']:08x}){slot}")
        flags = [f"single step {'on' if st['sst'] else 'off'} (SSt)",
                 f"Count {'runs' if st['countdm'] else 'stops'} in Debug Mode (CountDM)"]
        if st["lsnm"]:
            flags.append("dseg loads / stores go to memory (LSNM)")
        if st["iexi"]:
            flags.append("imprecise errors held (IEXI)")
        out.append(f"Debug 0x{st['debug']:08x}: " + ", ".join(flags))
        if st["held"]:
            out.append("   held until IEXI = 0: " + ", ".join(st["held"]) + " (DBusEP / CacheEP)")
        out.append(f"DEPC 0x{st['depc']:08x}   DESAVE 0x{st['desave']:08x}   "
                   f"debug vector 0x{st['vector']:08x}")
        out.append(f"DCR 0x{st['dcr']:08x}: interrupts "
                   f"{'enabled' if st['int_enabled'] else 'masked'} (IntE), NMI "
                   f"{'enabled' if st['nmi_enabled'] else 'masked'} (NMIE)"
                   + (", an NMI is pending" if st["nmi_pending"] else "")
                   + ", probe " + ("present" if st["probe"] else "none (ProbEn = 0)"))
        due = (["a Debug Single Step"] if st["step_due"] else []) + \
            (["a Debug Interrupt"] if st["dint_pending"] else [])
        if due:
            out.append("before the next instruction: " + ", ".join(due))
        return out + self.breakpoint_lines() + self.probe_lines()

    def interrupt_lines(self) -> list[str]:
        """:meth:`interrupt_state` as text (the debugger's ``srs``, the GUI's
        Interrupts tab)."""
        st = self.interrupt_state()
        mode = {"compatibility": "interrupt compatibility mode",
                "vectored": "vectored interrupts (VI)",
                "eic": "external interrupt controller (EIC)"}[st["mode"]]
        out = [f"interrupts: {mode}" + (f", vector spacing {st['spacing']} bytes"
                                        if st["spacing"] else "")]
        if st["mode"] == "eic":
            out.append(f"   the controller requests RIPL {st['ripl']}"
                       + (f" (shadow set {st['eic_ss']})" if st["ripl"] else "")
                       + f", Status.IPL {st['ipl']}; last taken: Cause.RIPL "
                       f"{st['cause_ripl']}, SRSCtl.EICSS {st['eicss']}")
        else:
            if st["eic_present"]:
                out.append("   an External Interrupt Controller is present (Config3.VEIC "
                           "= 1): EIC mode with Cause.IV = 1 and IntCtl.VS != 0")
            if st["pending"]:
                out.append("   pending and enabled (Cause.IP & Status.IM): "
                           + ", ".join(f"IP{i}" for i in range(7, -1, -1)
                                       if st["pending"] >> i & 1))
        if st["vector_number"] is not None:
            out.append(f"   next: vector {st['vector_number']} -> 0x{st['vector']:08x}"
                       + ("" if st["enabled"] else " (not taken now)"))
        if st["sets"] == 1:
            out.append("GPR shadow sets: none (the normal GPRs only; set shadow-sets N)")
            return out
        out.append(f"GPR shadow sets 0-{st['sets'] - 1}: current (CSS) {st['css']}, "
                   f"previous (PSS) {st['pss']}, exceptions (ESS) {st['ess']}")
        if st["mode"] != "eic":                 # EIC mode: the controller's sets
            out.append("   SRSMap: " + "  ".join(f"vector {v} -> {ss}"
                                                 for v, ss in enumerate(st["srsmap"])))
        return out

    def special_registers(self) -> list[tuple[str, int]]:
        m = self.machine
        rows = [("pc", m.pc), ("hi", m.hi), ("lo", m.lo)]
        for name, key in (("Status", (12, 0)), ("Cause", (13, 0)), ("EPC", (14, 0)),
                          ("BadVAddr", (8, 0)), ("Count", (9, 0)), ("Compare", (11, 0))):
            rows.append((name, m.cp0_read(*key)))
        if m.release != "R1":                       # interrupts / shadow sets (R2+)
            rows += [("IntCtl", m.cp0_read(*C.INTCTL)), ("SRSCtl", m.cp0_read(*C.SRSCTL))]
            if m.hss:
                rows.append(("SRSMap", m.cp0_read(*C.SRSMAP)))
        if self.cps is not None:                    # CPUNum
            rows.append(("EBase", m.cp0_read(*C.EBASE)))
        if m.debug_mode:                            # EJTAG (the ejtag command: all)
            rows += [("Debug", m.cp0_read(*C.DEBUG)), ("DEPC", m.cp0_read(*C.DEPC)),
                     ("DESAVE", m.cp0_read(*C.DESAVE))]
        mt = getattr(m.mmu, "mt", 3)
        if mt == 2:                                 # BAT: Index, bounds, base
            rows += [(name, m.cp0_read(*key)) for name, key in (
                ("Index", (0, 0)), ("EntryHi", (10, 0)), ("EntryLo0", (2, 0)))]
        if mt in (1, 4):                            # the TLB's registers
            mmu = m.mmu
            tlb = [("EntryHi", (10, 0)), ("Index", (0, 0)), ("Random", (1, 0)),
                   ("EntryLo0", (2, 0)), ("EntryLo1", (3, 0)), ("PageMask", (5, 0)),
                   ("Wired", (6, 0)), ("Context", (4, 0))]
            if mmu.pagegrain:
                tlb.append(("PageGrain", (5, 1)))
            if mmu.sc:
                tlb += [("SegCtl0", (5, 2)), ("SegCtl1", (5, 3)), ("SegCtl2", (5, 4)),
                        ("Config5", (16, 5))]
            if mmu.pw:
                tlb += [("PWBase", (5, 5)), ("PWField", (5, 6)), ("PWSize", (5, 7)),
                        ("PWCtl", (6, 6))]
            if m.has_bad_instr:
                tlb += [("BadInstr", (8, 1)), ("BadInstrP", (8, 2))]
            rows += [(name, m.cp0_read(*key)) for name, key in tlb]
        elif m.has_bad_instr:
            rows += [("BadInstr", m.cp0_read(8, 1)), ("BadInstrP", m.cp0_read(8, 2))]
        rows += [(name, m.cp0_read(*key)) for key, name in sorted(perf.NAMES.items())]
        rows.append(("FCSR", m.fcsr))
        return rows

    def current_pc(self, cpu: int | None = None) -> int | None:
        """The instruction about to execute (instruction mode) / in EX, of
        the selected core (or core ``cpu``)."""
        if self.machine is None:
            return None
        cpu = self.cpu if cpu is None else cpu
        p = self.pipelines[cpu]
        if p is None:
            return self.machines[cpu].pc
        ex = p.current_slot()
        return ex.pc if ex is not None else None

    def cores(self) -> list[dict[str, Any]]:
        """One row per core: ``cpu``, ``state`` (the CPC's SEQSTATE name),
        ``running``, ``pc`` (:meth:`current_pc`), ``line``, ``halted``,
        ``waiting`` (in WAIT), ``selected``, ``coherent``."""
        from .cps import SEQ_NAMES
        running = set(self.running())
        out = []
        for i, m in enumerate(self.machines):
            if self.cps is not None:
                st = self.cps.seq_state(i)
                state, coherent = SEQ_NAMES.get(st, str(st)), self.cps.coherent(i)
            else:
                state, coherent = "running", False
            pc = self.current_pc(i) if i in running else None
            src = self.source_line(pc, i) if pc is not None else None
            out.append({"cpu": i, "state": state, "running": i in running, "pc": pc,
                        "line": src.line if src else None, "halted": m.halted,
                        "waiting": m.waiting and i in running,
                        "selected": i == self.cpu, "coherent": coherent})
        return out

    def current_line(self) -> int | None:
        """Source line of :meth:`current_pc` (1-based) if known."""
        src = self.source_line(self.current_pc())
        return src.line if src else None

    def stage_pcs(self) -> dict[str, int]:
        """Pipeline mode: the address in each stage during the last cycle
        (what the datapath and the chart show)."""
        if self.pipeline is None:
            return {}
        if not self.pipeline.history:
            slot = self.pipeline.stage.get("IF")
            return {"IF": slot.pc} if slot is not None else {}
        return {st: sv.pc for st, sv in self.pipeline.history[-1].stages.items()
                if sv is not None}

    def stage_lines(self) -> dict[str, int]:
        """Pipeline mode: source line of the instruction in each stage
        during the last cycle."""
        if self.program is None:
            return {}
        out = {}
        for st, pc in self.stage_pcs().items():
            src = self.source_line(pc)
            if src is not None:
                out[st] = src.line
        return out

    def memory_words(self, start: int, count: int, *, physical: bool = False
                     ) -> list[tuple[int, list[int | None]]]:
        """``(address, [4 words])`` rows of a hex dump: virtual addresses
        (translated as the CPU would now, the data cache's copy where it
        holds the line, without touching the caches; ``None`` where no TLB
        entry maps the word), or ``physical`` ones (the RAM itself,
        bypassing the caches; devices are not read)."""
        from .exceptions import UnmappedAddress
        m = self.machine
        start &= ~0xF
        if physical:
            def word(pa: int) -> int | None:
                data = [m.phys_read_byte(pa + i) for i in range(4)]
                return int.from_bytes(bytes(data), "big" if m.big_endian else "little")
        else:
            def word(va: int) -> int | None:
                try:
                    return m.peek_word(va)
                except UnmappedAddress:
                    return None
        return [(start + 16 * r, [word(start + 16 * r + 4 * i) for i in range(4)])
                for r in range(count)]

    def disassembly(self) -> list[tuple[int, int, str, int | None]]:
        """``(address, word, text, source line)`` of the program's code."""
        if self.program is None:
            return []
        return [(a, w, t, self.program.source_map[a].line)
                for a, w, t, _ in self.program.listing(abi=self.abi)]

    def cache_accesses(self) -> list[tuple[str, Any]]:
        """The cache accesses of the last step, as ``(where, CacheAccess)``:
        instruction mode "fetch" / "data" of the executed instruction;
        pipeline mode "IF" (the fetch of the instruction in IF, with
        "IF (waiting)" while the line is still coming) and "MEM" (the
        access of the instruction in MEM)."""
        out: list[tuple[str, Any]] = []
        if self.machine is None or self.machine.caches is None:
            return out
        if self.pipeline is None:
            v = self.last_view
            if v is not None:
                out += [("fetch", a) for a in v.fetch] + [("data", a) for a in v.cache]
            return out
        hist = self.pipeline.history
        if not hist:
            return out
        last = hist[-1]
        sv = last.stages.get("IF")
        if sv is not None:
            where = "IF (waiting)" if sv.fetch_wait else "IF"
            out += [(where, a) for a in sv.fetch_cache]
        mem = self.pipeline.MEM_STAGE
        sv = last.stages.get(mem)
        if sv is not None:
            v = self.views.get(sv.seq)
            if v is not None:
                where = f"{mem} (waiting)" if sv.mem_wait else mem
                out += [(where, a) for a in v.cache]
        return out

    def tlb_accesses(self) -> list[tuple[str, Any]]:
        """The translations of the last step through a TLB or BAT MMU, as
        ``(where, TLBAccess)``: instruction mode "fetch" / "data" (and
        "instruction" for what TLBWI / TLBWR / TLBP / TLBR / TLBINV did),
        also for a fetch that failed (a TLB Refill on the fetch); pipeline
        mode "IF" (the fetch of the instruction in IF), "EX" (an access
        that raised its exception in EX: it never reaches MEM) and "MEM"
        (the instruction in MEM)."""
        m = self.machine
        out: list[tuple[str, Any]] = []
        if m is None or not hasattr(m.mmu, "watch"):
            return out

        def own(v) -> list[tuple[str, Any]]:
            return [("instruction" if a.note else "data", a) for a in v.tlb]

        if self.pipeline is None:
            v = self.last_view
            if v is not None:
                return [("fetch", a) for a in v.fetch_tlb] + own(v)
            return [("fetch", a) for a in m.fetch_tlb]
        hist = self.pipeline.history
        if not hist:
            return out
        last = hist[-1]
        sv = last.stages.get("IF")
        if sv is not None:
            out += [("IF", a) for a in sv.fetch_tlb]
        v = last.ex_view                    # a faulting access never reaches MEM
        if v is not None and v.exception:
            out += [("EX" if w == "data" else w, a) for w, a in own(v)]
        mem = self.pipeline.MEM_STAGE
        sv = last.stages.get(mem)
        if sv is not None:
            v = self.views.get(sv.seq)
            if v is not None:
                out += [(mem if w == "data" else w, a) for w, a in own(v)]
        return out

    def stage_details(self) -> dict[str, dict[str, Any]]:
        """Pipeline mode: what each stage did in the last cycle (its
        instruction, the ID control signals, and the InstrView recorded
        when the instruction executed: ALU in EX, memory access in MEM,
        register write in WB), plus that cycle's events."""
        out: dict[str, dict[str, Any]] = {}
        if self.pipeline is None:
            return out
        hist = self.pipeline.history
        if not hist:
            slot = self.pipeline.stage.get("IF")
            out["IF"] = {"text": slot.text if slot else "(bubble)",
                         "pc": slot.pc if slot else None}
            return out
        last = hist[-1]
        for st, sv in last.stages.items():
            if sv is None:
                out[st] = {"text": "(bubble)", "pc": None, "seq": None,
                           "control": None, "view": None}
                continue
            out[st] = {"text": sv.text, "pc": sv.pc, "seq": sv.seq,
                       "control": sv.control, "view": self.views.get(sv.seq)}
            if st == "IF":                      # the caches it went through
                out[st]["cache"], out[st]["wait"] = sv.fetch_cache, sv.fetch_wait
                out[st]["tlb"] = sv.fetch_tlb
            elif st == self.pipeline.MEM_STAGE and out[st]["view"] is not None:
                out[st]["cache"], out[st]["wait"] = out[st]["view"].cache, sv.mem_wait
                out[st]["tlb"] = out[st]["view"].tlb
        out["_events"] = {"events": last.events, "stall": last.stall,
                          "cycle": last.cycle}
        return out
