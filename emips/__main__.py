"""The eMIPS command line (PLAN M7).

    python3 -m emips asm   prog.asm [--hex prog.hex] [--numeric]
    python3 -m emips dis   prog.hex|prog.elf [--numeric]
    python3 -m emips run   prog.asm|prog.hex|prog.elf [--pipeline] [--steps N]
                           [--break LOC ...] [--until LOC] [--trace]
                           [--cp0] [--fpu] [--mem LOC[:WORDS] ...] [--cache-stats]
                           [--tlb-stats] [--perf EVENT[,EVENT]]
    python3 -m emips pipe  prog.asm [--cycles N] [--events] [--views] [--cache-stats]
                           [--tlb-stats] [--perf EVENT[,EVENT]]
    python3 -m emips debug prog.asm|prog.hex|prog.elf [-c "cmd; cmd"] [-x FILE]
    python3 -m emips gui   [prog.asm|prog.hex|prog.elf]   (needs PyQt6 or PySide6)
    python3 -m emips cc    prog.c [more.c] [-o prog.elf] [--release R2] [--little]

Shared options: ``--release R1|R2|R3|R5|R6``, ``--little``, ``--strict``
(no pseudo-instructions), ``--noreorder`` (no automatic delay-slot nops),
``--boot kernel|user|reset``, ``--exceptions cp0|stop``, ``--no-syscalls``,
``--keys TEXT`` (MMIO keyboard), ``--no-forwarding`` (pipeline),
``--caches`` (the 24K's L1 caches) or ``--icache`` / ``--dcache
SIZE[:WAYS[:LINE]]``, ``--cache-parity``, ``--cache-init random|hardware``,
``--l2`` / ``--l3 SIZE[:WAYS[:LINE]]``, ``--cache-ecc``, ``--write-buffer``,
``--mem-timing KEY=N,...`` and ``--nonblocking`` (memory timing of the
pipeline with caches), ``--mmu fixed|bat|tlb[:N]|ftlb[...][+sc][+pw]``,
``--model 5-stage|24k`` (the pipeline model: textbook or the MIPS 24K),
``--cores N`` (a multi-core system of N cores, emips.cps), ``--board
flat|malta[:RAM_MB[:SWITCHES]]`` (the MIPS Malta board, emips.malta),
``--disk [SLOT=]IMAGE`` (an IDE disk on the Malta, emips.ata).

``asm`` prints the listing (address, word, disassembly, source line) and
optionally writes Intel HEX; ``dis`` disassembles the words of a HEX file.
``run`` runs the program until it exits, leaves its code, hits a
breakpoint / ``--until``, or ``--steps``; program output is printed live
and read_* syscalls read stdin; then the registers (and ``--cp0``,
``--fpu``, ``--mem``, ``--cache-stats``: hits, misses and the 3 C's of
every cache; ``--tlb-stats``: TLB lookups, misses and exceptions;
``--perf``: the performance counters, set to count those events).
``pipe`` = ``run`` on the pipeline (``--model``) with the chart, CPI and
stalls (``--events``: forwarding, branch decisions, flushes per cycle;
``--views``: the datapath view of EX / ID); ``run --pipeline`` prints
the same.  ``debug`` is the interactive debugger
(:mod:`emips.cli`; ``help`` at its prompt).  Everything runs on
:class:`emips.session.Session`, the engine of the GUI.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from .assembler import AsmError, assemble
from .cli import CliError, Debugger, load_file, session_from_args
from .disasm import disassemble
from .hexload import DecodeFile
from .memory import PMemory
from .cache import Geometry
from .timing import MEMORY_TIMING
from .options import BOOT_MODES, CACHE_INIT_MODES, Options
from .session import Session, StepResult
from .sim import Machine
from .trace import format_view

RELEASES = ["R1", "R2", "R3", "R5", "R6"]


def _assemble(path: str, endian: str, release: str | None = None, args=None):
    pseudo = not getattr(args, "strict", False)
    reorder = not getattr(args, "noreorder", False)
    board = getattr(args, "board", None) or "flat"
    try:
        return assemble(Path(path).read_text(), endian=endian, release=release,
                        pseudo=pseudo, reorder=reorder,
                        bases=Options(board=board).segment_bases())
    except AsmError as exc:
        sys.exit(f"{path}: {exc}")


def _cmd_asm(args) -> None:
    prog = _assemble(args.file, args.endian, args.release, args)
    for addr, word, text, src in prog.listing(abi=not args.numeric):
        print(f"{addr:08x}  {word:08x}  {text:32} # {src}")
    if args.hex:
        Path(args.hex).write_text(prog.to_ihex())


def _cmd_cc(args) -> None:
    from .cc import BuildError, build
    out = args.output or str(Path(args.sources[0]).with_suffix(".elf"))
    try:
        log = build(args.sources, out, release=args.release, endian=args.endian,
                    optimize=f"-O{args.opt}")
    except BuildError as exc:
        sys.exit(str(exc))
    if args.verbose:
        print(log)
    print(f"wrote {out}: run it with  python3 -m emips run {out}")


def _cmd_dis(args) -> None:
    if Path(args.file).read_bytes()[:4] == b"\x7fELF":     # an ELF: its listing
        s = Session(abi=not args.numeric)
        if not s.load_elf(Path(args.file).read_bytes(), args.file):
            sys.exit(f"{args.file}: {s.errors[0]}")
        print(s.listing, end="")
        return
    mem = PMemory()
    DecodeFile(mem).load(args.file)
    m = Machine(memory=mem, endian=args.endian, release=args.release)
    seen = set()
    for a in sorted(int(k, 16) for k in mem.get_memory()):
        w = a & ~3
        if w in seen or any(mem.read_from_memory(w + i) is None for i in range(4)):
            continue
        seen.add(w)
        word = m.read_word(w)
        print(f"{w:08x}  {word:08x}  "
              f"{disassemble(word, w, abi=not args.numeric, isa=m.isa)}")


# ------------------------------------------------------------------- run

def _session(args):
    s = session_from_args(args)
    errors: list[str] = []
    if not load_file(s, args.file, errors.append):
        sys.exit(errors[0] if errors else f"cannot load {args.file}")
    s.type_keys(_keys(args))
    s.stop_at_end = True
    return s


def _keys(args) -> str:
    """--keys TEXT, with \\n, \\t ... understood (as the debugger's keys)."""
    text = getattr(args, "keys", "") or ""
    return text.encode("latin-1", "backslashreplace").decode("unicode_escape")


def _run_to_stop(d: Debugger, max_steps: int, trace: bool) -> StepResult:
    """Run like the debugger's continue; with ``trace`` print what happened
    inside every executed instruction."""
    s = d.s
    if not trace:
        return d._run(max_steps)
    total = StepResult()
    while total.steps < max_steps:
        r = d._run(1)
        total.steps += r.steps
        total.stopped, total.error = r.stopped, r.error
        if r.steps and s.pipeline is None and s.last_view is not None:
            print(format_view(s.last_view))
        if r.stopped or r.error or not r.steps:
            break
    return total


def _cmd_run(args) -> None:
    s = _session(args)
    d = Debugger(s, stdout=sys.stdout, program_input=sys.stdin)
    s.on_output = lambda text: (sys.stdout.write(text), sys.stdout.flush())
    d.live_output = True
    try:
        for loc in args.breaks or []:
            s.breakpoints.add(d.resolve(loc))
        if args.until:
            s.run_to = d.resolve(args.until)
        if getattr(args, "perf", None):
            _setup_perf(s, args.perf)
    except CliError as exc:
        sys.exit(f"error: {exc}")
    steps = args.steps
    if steps is None:
        steps = 1_000_000 if s.board is not None else 100_000
    r = _run_to_stop(d, steps, args.trace)
    s.on_output = None
    if s.output and not s.output.endswith("\n"):
        print()
    if r.error:
        print(f"stopped: {r.error}")
    elif r.stopped in ("breakpoint", "cursor"):
        print("stopped at " + d._where(s.current_pc(), "").strip())
    elif r.stopped == "input":
        print("stopped: the program waits for input (stdin is empty)")
    if s.pipeline is not None:
        if r.stopped == "end":
            s.pipeline.drain(s.code_addresses())
        _print_pipeline(s, args)
    else:
        m = s.machine
        print(f"\n--- {s.steps} steps, pc = 0x{m.pc:08x}, {m.cycles} cycles"
              + (" (exited)" if s.halted else ""))
        d.do_regs("")
    if s.cps is not None:
        print("--- cores")
        d.do_cpu("")
    _extras(d, args)
    for i, m in enumerate(s.machines):
        for e in m.exception_log:
            if "vector" in e and s.cps is not None:
                e = dict(e, cpu=i)
            print(d._exception_line(e).strip() if "vector" in e else f"exception: {e}")


def _setup_perf(s, spec: str) -> None:
    """--perf EVENT[,EVENT]: each event on a counter that can count it, in
    user and kernel mode (as ``perf N EVENT``)."""
    from itertools import product

    from . import perf
    names = [x.strip() for x in spec.split(",") if x.strip()]
    for name in names:
        if not perf.find(name):
            raise CliError(f"--perf: no counter can count {name!r} (perf events in "
                           "the debugger lists the events)")
    for choice in product(*(perf.find(name) for name in names)):
        if len({i for i, _ in choice}) == len(choice):    # one event per counter
            for i, n in choice:
                s.configure_counter(i, n, "uk")
            return
    raise CliError(f"--perf: the two counters cannot count {', '.join(names)} together")


def _extras(d: Debugger, args) -> None:
    try:
        if getattr(args, "cp0", False):
            d.do_cp0("")
        if getattr(args, "fpu", False):
            d.do_fpu("")
        for spec in getattr(args, "mem", None) or []:
            loc, _, n = spec.partition(":")
            print(f"--- memory {loc}")
            d.do_mem(f"{loc} {n or 16}")
        if getattr(args, "cache_stats", False):
            print("--- caches")
            d.do_cache("stats")
        if getattr(args, "tlb_stats", False):
            print("--- TLB")
            d.do_tlb("stats")
        if getattr(args, "perf", None):
            print("--- performance counters")
            d.do_perf("")
    except CliError as exc:
        print(f"error: {exc}")


def _print_pipeline(s, args) -> None:
    p = s.pipeline
    print()
    print(p.chart(width=args.width))
    cpi = p.cycles / p.retired if p.retired else float("nan")
    print(f"\n{p.cycles} cycles, {p.retired} instructions retired, CPI {cpi:.2f}")
    if p.stalls:
        print("stall cycles:", ", ".join(f"{k} {v}" for k, v in sorted(p.stalls.items())))
    if s.cps is not None:                       # the other cores (the chart: core s.cpu)
        for i, q in enumerate(s.pipelines):
            if i != s.cpu and q is not None:
                qcpi = f", CPI {q.cycles / q.retired:.2f}" if q.retired else ""
                print(f"core {i}: {q.cycles} cycles, {q.retired} instructions retired{qcpi}")
    if getattr(args, "events", False) or getattr(args, "views", False):
        for t in p.history:
            for e in ([t.stall] if t.stall else []) + t.events:
                print(f"{t.cycle:5}  {e}")
            if getattr(args, "views", False):
                for stage, v in (("ID", t.id_view), ("EX", t.ex_view)):
                    if v is not None:
                        body = format_view(v).replace("\n", "\n" + " " * 13)
                        print(f"{t.cycle:5}  [{stage}] {body}")


def _cmd_pipe(args) -> None:
    args.pipeline = True
    args.steps = args.cycles
    args.trace = False
    _cmd_run(args)


# ------------------------------------------------------------------ debug

def _cmd_debug(args) -> None:
    s = session_from_args(args)
    s.stop_at_end = args.stop_at_end
    script: list[str] = []
    for c in args.command or []:
        script += [x.strip() for x in c.split(";") if x.strip()]
    if args.script:
        script += [ln.strip() for ln in Path(args.script).read_text().splitlines()
                   if ln.strip() and not ln.lstrip().startswith("#")]
    scripted = bool(script)
    d = Debugger(s, stdin=io.StringIO("\n".join(script) + "\n") if scripted else None,
                 stdout=sys.stdout, program_input=sys.stdin, echo=scripted)
    if args.file:
        if not load_file(s, args.file, d.say):
            sys.exit(1)
        s.type_keys(_keys(args))
    if not scripted:
        d.intro = (f"eMIPS debugger — MIPS32 {s.release}, {s.endian}-endian, "
                   f"{s.mode} mode. Type help.")
        if s.machine is not None:
            d.intro += "\n" + d._where(s.current_pc())
    while True:
        try:
            d.cmdloop()
            return
        except KeyboardInterrupt:              # Ctrl+C at the prompt: keep going
            d.say("^C  (quit or Ctrl+D to leave)")
            d.intro = ""


def _cmd_gui(args) -> None:
    try:
        from .gui.main import main as gui_main
    except ImportError as exc:
        sys.exit("the GUI needs PyQt6 (or PySide6): install your system's package "
                 "(Debian / Ubuntu: sudo apt install python3-pyqt6) or, in a virtual "
                 f"environment, pip install PyQt6  ({exc})")
    sys.exit(gui_main([args.file] if args.file else []))



def _geometry(text: str, level: int = 1) -> str:
    """argparse type of --icache / --dcache: a valid cache geometry."""
    try:
        Geometry.parse(text, level)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text


def _geometry2(text: str) -> str:
    """argparse type of --l2 / --l3 (Config2 shapes)."""
    return _geometry(text, 2)


def _mmu(text: str) -> str:
    """argparse type of --mmu (emips.mmu.make_mmu)."""
    from .mmu import make_mmu
    try:
        make_mmu(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text.lower()


def _disk(text: str) -> str:
    """argparse type of --disk (emips.malta.parse_disks)."""
    from .malta import parse_disks
    try:
        parse_disks(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text


def _board(text: str) -> str:
    """argparse type of --board (emips.malta.make_board)."""
    from .malta import make_board
    try:
        make_board(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text.lower()


def _mem_timing(text: str) -> dict[str, int]:
    """argparse type of --mem-timing: KEY=N[,KEY=N] (emips.timing)."""
    out = {}
    for item in text.split(","):
        key, _, n = item.partition("=")
        key = key.strip().upper()
        if key not in MEMORY_TIMING or not n.strip().isdigit():
            raise argparse.ArgumentTypeError(
                f"{item!r}: expected KEY=N with KEY one of {', '.join(MEMORY_TIMING)}")
        out[key] = int(n)
    return out

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m emips", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (
            ("asm", _cmd_asm, "assemble: listing, Intel HEX"),
            ("dis", _cmd_dis, "disassemble an Intel HEX file"),
            ("run", _cmd_run, "run a program, print the registers"),
            ("pipe", _cmd_pipe, "run on the 5-stage pipeline: chart, CPI, stalls"),
            ("debug", _cmd_debug, "interactive debugger (help at its prompt)")):
        p = sub.add_parser(name, help=helptext)
        if name == "debug":
            p.add_argument("file", nargs="?")
        else:
            p.add_argument("file")
        p.add_argument("--little", dest="endian", action="store_const",
                       const="little", default="big", help="little-endian")
        p.add_argument("--release", choices=RELEASES, default=None,
                       help="MIPS32 release (default R5)")
        p.set_defaults(fn=fn)
        if name in ("asm", "dis", "debug"):
            p.add_argument("--numeric", action="store_true",
                           help="print $8 instead of $t0")
        if name != "dis":
            p.add_argument("--strict", action="store_true",
                           help="strict mode: no pseudo-instructions")
            p.add_argument("--noreorder", action="store_true",
                           help="start in .set noreorder (fill delay slots yourself)")
        if name == "asm":
            p.add_argument("--hex", help="write Intel HEX here")
            p.add_argument("--board", metavar="SPEC", type=_board, default=None,
                           help="assemble for this board's memory layout (malta: "
                                "the segments in kseg0 SDRAM)")
        if name in ("run", "pipe", "debug"):
            p.add_argument("--keys", default="", help="type this on the MMIO keyboard (\\n = Enter)")
            p.add_argument("--no-header", action="store_true",
                           help="ignore the program's '# eMIPS: ...' settings line")
            p.add_argument("--no-syscalls", action="store_true",
                           help="SYSCALL raises the exception (no OS emulation)")
            p.add_argument("--boot", choices=BOOT_MODES, default="kernel",
                           help="start state: kernel (MARS-like), user, reset")
            p.add_argument("--exceptions", choices=["cp0", "stop"], default="cp0",
                           help="take exceptions (cp0) or stop the simulation")
            p.add_argument("--no-forwarding", action="store_true",
                           help="pipeline without forwarding paths")
            p.add_argument("--caches", action="store_true",
                           help="24K L1 caches: 32 KB I-cache and D-cache, 4-way, 32-byte lines")
            p.add_argument("--icache", metavar="SIZE[:WAYS[:LINE]]", default=None,
                           type=_geometry, help="instruction cache, e.g. 16K or 8K:2:32 (none: off)")
            p.add_argument("--dcache", metavar="SIZE[:WAYS[:LINE]]", default=None,
                           type=_geometry, help="data cache, e.g. 32K (none: off)")
            p.add_argument("--cache-parity", action="store_true",
                           help="caches with parity (ErrCtl.PE, Cache Error exception)")
            p.add_argument("--cache-init", choices=CACHE_INIT_MODES, default="random",
                           help="cache contents after reset: random (undefined) or "
                                "hardware (invalid, Config7.HCI)")
            p.add_argument("--l2", metavar="SIZE[:WAYS[:LINE]]", default=None,
                           type=_geometry2, help="L2 cache, e.g. 256K (8-way, 32-byte lines) "
                                                 "or 512K:4:64")
            p.add_argument("--l3", metavar="SIZE[:WAYS[:LINE]]", default=None,
                           type=_geometry2, help="L3 cache (needs --l2), e.g. 2M")
            p.add_argument("--cache-ecc", action="store_true",
                           help="L2 / L3 with ECC (ErrCtl.L2P, imprecise Cache Error)")
            p.add_argument("--write-buffer", action="store_true",
                           help="24K write buffer: write-through merging, uncached "
                                "accelerated gathering, SYNC empties it")
            p.add_argument("--mem-timing", metavar="KEY=N[,KEY=N]", type=_mem_timing,
                           default=None, help="pipeline memory cycles with caches: "
                           + ", ".join(f"{k} (default {v})" for k, v in MEMORY_TIMING.items()))
            p.add_argument("--nonblocking", action="store_true",
                           help="pipeline: loads miss in the background (hit under miss)")
            p.add_argument("--mmu", metavar="SPEC", type=_mmu, default=None,
                           help="MMU: fixed (default), bat, tlb[:N] (the 24K's TLB, "
                                "default 32 entries), ftlb[:N[:SETSxWAYS]] (VTLB + "
                                "FTLB, default 64 + 128x4); a TLB takes +sc "
                                "(segmentation control, EVA) and +pw (page table "
                                "walker), e.g. ftlb+sc+pw")
            p.add_argument("--board", metavar="SPEC", type=_board, default=None,
                           help="the board: flat (default: every physical address is "
                                "RAM, the MARS layout) or malta[:RAM_MB[:SWITCHES]] "
                                "(the MIPS Malta: the GT-64120's memory map, bus "
                                "errors, the board registers; programs in kseg0)")
            p.add_argument("--disk", metavar="[SLOT=]IMAGE", action="append", type=_disk,
                           default=None,
                           help="an IDE disk on the Malta (with --board malta): SLOT "
                                "pri0 (default, then pri1 sec0 sec1), IMAGE a file or "
                                "blank:SIZE (e.g. blank:1M); repeatable; every run "
                                "starts from the image (the debugger's disk save keeps "
                                "what it wrote)")
            p.add_argument("--model", choices=("5-stage", "24k"), default=None,
                           help="pipeline model: 5-stage (the textbook pipeline, "
                                "default) or 24k (the MIPS 24K's 8 stages, branch "
                                "prediction, skewed ALU)")
            p.add_argument("--cores", type=int, choices=(1, 2, 3, 4), default=None,
                           help="cores (a CPS: shared memory, the CM's GCRs, the "
                                "CPC); kernel / user boot starts them all at the "
                                "entry, reset boot core 0 only")
            p.add_argument("--iocu", action="store_true",
                           help="with --board malta --cores N: an I/O Coherence Unit "
                                "(GCR_CONFIG.NUMIOCU = 1) — the PCI DMA is coherent "
                                "with the data caches")
            p.add_argument("--shadow-sets", type=int, choices=range(1, 17),
                           default=None, metavar="N",
                           help="GPR shadow sets, 1-16 (SRSCtl.HSS + 1; default 4, "
                                "as a 24K built with 4)")
        if name in ("run", "debug"):
            p.add_argument("--pipeline", action="store_true",
                           help="pipeline mode (--model): a step is a clock cycle")
        if name in ("run", "pipe"):
            p.add_argument("--break", dest="breaks", action="append", metavar="LOC",
                           help="breakpoint: 0x address, line number or label")
            p.add_argument("--until", metavar="LOC", help="run until LOC is next")
            p.add_argument("--width", type=int, default=40, help="chart width (cycles)")
            p.add_argument("--cp0", action="store_true", help="print CP0 registers")
            p.add_argument("--fpu", action="store_true", help="print FPU registers")
            p.add_argument("--mem", action="append", metavar="LOC[:WORDS]",
                           help="print memory (e.g. nums:8, 0x10010000)")
            p.add_argument("--cache-stats", action="store_true",
                           help="print the cache statistics: hits, misses, the 3 C's")
            p.add_argument("--tlb-stats", action="store_true",
                           help="print the TLB statistics: lookups, misses, exceptions")
            p.add_argument("--perf", metavar="EVENT[,EVENT]",
                           help="count these events on the performance counters "
                                "(user + kernel mode) and print them, e.g. "
                                "instructions,dcache_miss")
        if name == "run":
            p.add_argument("--steps", type=int, default=None,
                           help="stop after this many steps (default 100000; on the "
                                "Malta 1000000: its 38400-baud serial output takes "
                                "thousands of instructions a character)")
            p.add_argument("--trace", action="store_true",
                           help="print what happens inside every instruction")
        if name == "pipe":
            p.add_argument("--cycles", type=int, default=10_000,
                           help="stop after this many cycles (default 10000)")
            p.add_argument("--events", action="store_true",
                           help="forwarding, branch decisions, flushes per cycle")
            p.add_argument("--views", action="store_true",
                           help="events + the datapath view of EX / ID")
        if name == "debug":
            p.add_argument("-c", "--command", action="append",
                           help='commands instead of the prompt: -c "break loop; c; regs"')
            p.add_argument("-x", "--script", help="a file of commands (one per line)")
            p.add_argument("--stop-at-end", action="store_true",
                           help="stop when the program runs past its last instruction")
    c = sub.add_parser("cc", help="build a C program for eMIPS (clang or a MIPS GCC)")
    c.add_argument("sources", nargs="+", help="C (.c) or assembly (.S) files")
    c.add_argument("-o", "--output", help="the ELF to write (default: the first "
                                          "source's name .elf)")
    c.add_argument("--release", choices=RELEASES, default="R2",
                   help="MIPS32 release to build for (default R2)")
    c.add_argument("--little", dest="endian", action="store_const", const="little",
                   default="big", help="little-endian")
    c.add_argument("-O", dest="opt", choices=["0", "1", "2", "3", "s"], default="1",
                   help="optimisation level (default 1)")
    c.add_argument("-v", "--verbose", action="store_true", help="print the commands")
    c.set_defaults(fn=_cmd_cc)
    g = sub.add_parser("gui", help="graphical debugger / datapath view")
    g.add_argument("file", nargs="?")
    g.set_defaults(fn=_cmd_gui)
    args = ap.parse_args(argv)
    if getattr(args, "l3", None) and not getattr(args, "l2", None):
        ap.error("--l3 needs an L2 cache (--l2)")
    args.fn(args)


if __name__ == "__main__":
    main()
