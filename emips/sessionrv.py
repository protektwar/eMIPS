"""RISC-V in the Session (PLAN M25 phase 3c): what :class:`emips.session.
Session` does differently for a RISC-V hart, kept here so the Session's
MIPS paths have one test (``Session.rv``) each, not RISC-V cases inside.

* The hart (``make_core``): ``emips.simrv.RVMachine`` with the options'
  extensions and modes, RARS's services (``emips.rvservices``), the MMIO
  console of RARS (= MARS's keyboard and display at 0xffff0000) — its two
  interrupt lines ORed into the machine external interrupt (mip.MEIP: no
  PLIC on the flat board; ``sim.DEVIATIONS["RARS services"]``).
* The boot (decision 11): "kernel" = the program in M-mode at its entry
  (``Program.load``: sp, gp, mstatus.FS), "user" = the same, then the hart
  in U-mode as an mret would leave it, with mcounteren / scounteren letting
  it read cycle / time / instret (what OpenSBI grants); "reset" = the
  hart's reset state at the entry.
* The views a debugger needs: the instruction text, whether the next
  instruction is a reading ``ecall``, the return address of a call, the
  call stack by the return-address-stack hints of the unprivileged spec
  2.5.1 (x1 / x5 are link registers), the trap vectors, the hints for a
  trap with no handler.
"""

from __future__ import annotations

from functools import lru_cache

from .exceptions import FetchError, SimException
from .riscv import RVISA, Disassembler, RVDecodeError, size_of
from .rvservices import READ_SERVICES

M32 = 0xFFFFFFFF
#: the link registers (unprivileged 2.5.1: ra, t0)
LINK = (1, 5)
ECALL = 0x00000073

def make_core(session, cpu: int, mem, caches):
    """A RISC-V hart in its reset state on the session's memory, console
    and services."""
    from .sim import Machine
    o = session.options
    m = Machine(memory=mem, arch=o.arch, extensions=o.rv_extensions, modes=o.rv_modes,
                exceptions="raise" if o.exceptions == "stop" else "cp0",
                caches=caches, cpu=cpu)
    m.tracing = True
    m.agent_watch = session.watchpoints
    if cpu == 0:
        session.console.attach(m)
    else:
        session.console.share(m)
    m.syscall_handler = session.services
    return m


def console_sink(session):
    """The console's interrupt lines -> the hart's MEIP (bit 11)."""
    levels = {}

    def sink(line: int, on: bool) -> None:
        levels[line] = on
        if session.machines:
            session.machines[0].set_interrupt(11, any(levels.values()))
    return sink


def boot(session, m) -> None:
    """The user boot: U-mode (as mret leaves it, MPP = U), the counters
    readable there."""
    if session.options.boot != "user":
        return
    if 0 not in m.modes:
        note = ("boot user -> machine: the hart has M-mode only "
                f"(--rv-modes {session.options.rv_modes})")
        if note not in session.program_notes:
            session.program_notes.append(note)
            session.adjusted.append(note)
        return
    m.priv = 0
    m.csrs.mcounteren = 0x7                     # CY, TM, IR
    if 1 in m.modes:
        m.csrs.scounteren = 0x7


def code(m, pc: int) -> tuple[int, int] | None:
    """(raw, size) of the instruction at ``pc`` as a fetch would see it
    now (a view: no side effects), or None."""
    try:
        h = m.code_half(pc)
        size = size_of(h)
        if size == 4:
            return h | m.code_half((pc + 2) & m.amask) << 16, 4
        return (h, 2) if size == 2 else None
    except (SimException, ValueError):
        return None


@lru_cache(maxsize=16)
def _disassembler(xlen: int, exts: frozenset, numeric: bool) -> Disassembler:
    return Disassembler(xlen, exts, numeric=numeric)


def instruction_text(m, pc: int, abi: bool = True, symbols=None) -> str:
    c = code(m, pc)
    if c is None:
        return "?"
    d = _disassembler(m.xlen, m.extensions, not abi)
    return d.text(c[0], c[1], pc)


def needs_input(m, pc: int) -> bool:
    """The next instruction is an ecall of a service that reads input."""
    a7 = m.read_reg(17) & M32
    if a7 not in READ_SERVICES and not (a7 == 63 and m.read_reg(10) == 0):
        return False
    c = code(m, pc)
    return c is not None and c == (ECALL, 4)


def _decode(m, pc: int):
    c = code(m, pc)
    if c is None:
        return None
    try:
        return m.rvisa.decode(*c)
    except RVDecodeError:
        return None


def return_address(m, pc: int) -> int | None:
    """Where a call at ``pc`` returns (jal / jalr / c.jal / c.jalr with a
    link register as rd), else None."""
    i = _decode(m, pc)
    if i is None or i.base.name not in ("jal", "jalr") or i.f.get("rd") not in LINK:
        return None
    return (pc + i.size) & m.amask


def call_hook(session, cpu: int, m):
    """The call stack (unprivileged 2.5.1's return-address-stack hints):
    rd a link register pushes, rs1 a link register (not rd) pops — both:
    pop then push; jal pushes when rd links."""
    from .session import Frame

    def hook(pc: int, i, raw: int, nxt: int) -> None:
        name = i.base.name
        if name not in ("jal", "jalr"):
            return
        rd = i.f.get("rd", 0)
        stack = session.calls[cpu]
        if name == "jalr":
            rs1 = i.f.get("rs1", 0)
            if rs1 in LINK and (rd not in LINK or rd != rs1) and stack:
                stack.pop()
        if rd in LINK:
            stack.append(_frame(Frame, pc, nxt, (pc + i.size) & m.amask, m.read_reg(2)))
            if len(stack) > 10_000:
                del stack[:1000]
    return hook


def _frame(cls, site, target, ret, sp):
    return cls(site, target, ret, sp)


def trap_vector(m, pc: int) -> str | None:
    """"m" / "s" when ``pc`` is where a trap enters (mtvec / stvec: the base,
    or base + 4 x cause in vectored mode), else None."""
    for kind, value in (("m", m.csrs.mtvec), ("s", getattr(m.csrs, "stvec", None))):
        if value is None:
            continue
        base = value & ~3 & m.amask
        if pc == base or (value & 3 == 1 and base < pc <= base + 4 * 64 and not (pc - base) % 4):
            return kind
    return None


def no_handler_hint(session, exc: Exception) -> str:
    m = session.machine
    if not isinstance(exc, FetchError) or m is None or not getattr(m, "rv", False):
        return ""
    kind = trap_vector(m, m.pc)
    log = m.exception_log
    if kind is None or len(log) < 2 or log[-2].get("vector") != m.pc:
        return ""                               # (no trap brought it here)
    from .simrv import CAUSE_NAMES, INTERRUPT_NAMES
    cause = m.csrs.mcause if kind == "m" else m.csrs.scause
    epc = m.csrs.mepc if kind == "m" else m.csrs.sepc
    interrupt = bool(cause >> (m.xlen - 1))
    num = cause & ~(1 << (m.xlen - 1))
    name = (INTERRUPT_NAMES if interrupt else CAUSE_NAMES).get(num, str(num))
    digits = 8 if m.xlen == 32 else 16
    return (f" — the hart took {'interrupt' if interrupt else 'exception'} {name} "
            f"({kind}cause {num}, {kind}epc 0x{epc:0{digits}x}) to {kind}tvec = "
            f"0x{m.pc:0{digits}x}, but there is no trap handler there: write one and point "
            f"{kind}tvec at it (csrw {kind}tvec, handler), or choose Settings > Exceptions: "
            "stop the simulation (--exceptions stop)")


EXIT_HINT = "li a7, 10 / ecall"


def gpr_number(tok: str) -> int | None:
    """A RISC-V register name (the ABI's, fp, x0-x31; a leading $ is
    allowed), else None."""
    from .riscv import GPR_NAMES
    t = tok.strip().lstrip("$").lower()
    if t == "fp":
        return 8
    if t in GPR_NAMES:
        return GPR_NAMES.index(t)
    if t.startswith("x") and t[1:].isdigit() and int(t[1:]) < 32:
        return int(t[1:])
    return None


def hex_listing(session, prog, image: dict[int, int], entry: int) -> str:
    """load_hex's listing of a RISC-V image: each run of halfwords that
    holds the entry or lies below 0x10000000 is code, decoded instruction
    by instruction from its start (prog.rv, the source map: listing lines
    act as source lines); the other halfwords are data."""
    from .assembler import SourceLine
    from .cli import encoding
    o = session.options
    xlen = 32 if o.arch == "rv32" else 64
    d = Disassembler(xlen, o.rv_extensions, numeric=not session.abi)
    isa = RVISA.for_core(xlen, o.rv_extensions)
    halves = sorted({a & ~1 for a in image if (a & ~1) in image and (a | 1) in image})
    runs: list[list[int]] = []
    for a in halves:
        if runs and runs[-1][-1] + 2 == a:
            runs[-1].append(a)
        else:
            runs.append([a])
    half = lambda a: image[a] | image[a + 1] << 8        # noqa: E731
    digits = 8 if xlen == 32 else 16
    lines = [f"# Intel HEX image: {len(image)} bytes, entry 0x{entry:0{digits}x}",
             f"# RV{xlen} {o.rv_extensions}, little-endian — read-only disassembly "
             "(address, encoding, instruction)", "", "# code"]
    data = []
    for run in runs:
        a, end = run[0], run[-1] + 2
        if not (a <= entry < end or a < 0x10000000):
            data += run
            continue
        while a < end:
            h = half(a)
            size = size_of(h) or 4
            if a + size > end:
                data += list(range(a, end, 2))
                break
            raw = h | (half(a + 2) << 16 if size == 4 else 0)
            try:
                isa.decode(raw, size)
                text = d.text(raw, size, a)
            except RVDecodeError:
                text = "(illegal)"
            line = f"{a:08x}  {encoding(raw, size, o.arch)}  {text}"
            lines.append(line)
            prog.rv[a] = size
            prog.source_map[a] = SourceLine(len(lines), line)
            a += size
    if data:
        lines += ["", "# data"]
        lines += [f"{a:08x}  {half(a):04x}" for a in data]
    prog.rv_extensions = o.rv_extensions
    return "\n".join(lines) + "\n"


def missing_extensions(hart: str, code: str) -> list[str]:
    """The extensions ``code`` was built for that the hart lacks (both
    closed under what they imply)."""
    from .riscv import closure, parse_extensions
    have = closure(parse_extensions(hart) | {"i"})
    need = closure(parse_extensions(code) | {"i"})
    return sorted(need - have)


def listing_row(session, prog, a: int, stop: int, labels) -> tuple[int, int, str]:
    """(size, encoding, text) of the instruction an ELF has at ``a`` (the
    load_elf listing: the hart's and the code's extensions decode)."""
    xlen = 32 if prog.arch == "rv32" else 64
    exts = session.options.rv_extensions + "_" + (prog.rv_extensions or "i")
    h = prog.half(a)
    size = size_of(h) or 4
    if a + size > stop:
        return 2, h, "(a halfword)"
    raw = h | (prog.half(a + 2) << 16 if size == 4 else 0)
    key = (xlen, exts, session.abi, id(labels))
    d = getattr(session, "_rv_listing", (None, None))
    if d[0] != key:                             # (one per listing: objdump's walk)
        d = (key, Disassembler(xlen, exts, numeric=not session.abi, symbols=labels))
        session._rv_listing = d
    try:
        RVISA.for_core(xlen, exts).decode(raw, size)
        return size, raw, d[1].text(raw, size, a)
    except RVDecodeError:
        return size, raw, "(illegal)"


#: mip / mie bit names (priv. 3.1.9 / Table 14)
IRQ_BITS = ((11, "MEI"), (9, "SEI"), (7, "MTI"), (5, "STI"), (3, "MSI"), (1, "SSI"))


def _irq_names(v: int) -> str:
    return " ".join(n for b, n in IRQ_BITS if v >> b & 1) or "-"


def interrupt_lines(m) -> list[str]:
    """The hart's interrupts (priv. 3.1.6.1 / 3.1.8 / 3.1.9 / 3.1.7): the
    global enables, mip / mie, delegation, xtvec's mode, what is taken next
    (the debugger's ``srs`` on MIPS, the register pane's Interrupts tab)."""
    from . import rvcsr as CSR
    from .simrv import INTERRUPT_NAMES
    c = m.csrs
    out = [f"mstatus.MIE {c.status_bit(CSR.MIE)}"
           + (f", SIE {c.status_bit(CSR.SIE)}" if c.has_s else "")
           + f"; the hart runs in {({3: 'M', 1: 'S', 0: 'U'})[m.priv]}-mode",
           f"mip {_irq_names(c.mip())}   (0x{c.mip():x})",
           f"mie {_irq_names(c.mie)}   (0x{c.mie:x})"]
    if c.has_s:
        out.append(f"mideleg {_irq_names(c.mideleg)} (to S-mode), medeleg 0x{c.medeleg:x}")
    for name, tvec in (("mtvec", c.mtvec),) + ((("stvec", c.stvec),) if c.has_s else ()):
        out.append(f"{name} 0x{tvec & ~3 & m.amask:x}: "
                   + ("vectored (BASE + 4 x cause for interrupts)" if tvec & 3 == 1
                      else "direct"))
    req = m.interrupt_request()
    out.append("taken before the next instruction: "
               + (f"{INTERRUPT_NAMES.get(req[0], req[0])} -> {'S' if req[1] == 1 else 'M'}-mode"
                  if req is not None else "none")
               + ("  (WFI: waiting)" if m.waiting else ""))
    return out


def debug_lines(m) -> list[str]:
    """The hart's debug state (Debug spec ch. 4-5): Debug Mode, dcsr, dpc,
    the triggers (the debugger's ``ejtag`` on MIPS, the Debug tab)."""
    from .rvdebug import CAUSE_SHIFT, DISABLED, EBREAKM, EBREAKS, EBREAKU, STEP
    c = m.csrs
    causes = {1: "ebreak", 2: "a trigger", 3: "a halt request", 4: "a single step",
              5: "a reset halt request"}
    if m.debug_mode:
        out = [f"Debug Mode (Debug spec 4.1): entered by "
               f"{causes.get(c.dcsr >> CAUSE_SHIFT & 7, '?')}, dpc 0x{c.dpc:x}; the Debug "
               "Module drives the hart"]
    else:
        out = ["not in Debug Mode"]
    ebreak = [k for bit, k in ((EBREAKM, "M"), (EBREAKS, "S"), (EBREAKU, "U")) if c.dcsr & bit]
    out.append(f"dcsr 0x{c.dcsr:08x}: prv {c.dcsr & 3}, step {int(bool(c.dcsr & STEP))}, "
               "ebreak enters Debug Mode in " + (", ".join(ebreak) or "no mode"))
    used = [k for k, t in enumerate(m.tm.triggers) if t.type != DISABLED]
    out.append(f"triggers: {m.tm.count} (Sdtrig), "
               + (f"in use: {', '.join(str(k) for k in used)}" if used else "none in use"))
    if m.debug_module is not None:
        out.append("a Debug Module is attached (the JTAG DTM)")
    return out
