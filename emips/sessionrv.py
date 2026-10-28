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

from dataclasses import replace
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
    p8700 = o.core == "p8700"                   # (the P8700: emips.p8700, 64 PMP entries)
    m = Machine(memory=mem, arch=o.arch, extensions=o.rv_extensions, modes=o.rv_modes,
                exceptions="raise" if o.exceptions == "stop" else "cp0",
                caches=caches, cpu=cpu, core="p8700" if p8700 else None,
                pmp=P8700_PMP if p8700 else 0)
    m.tracing = True
    m.agent_watch = session.watchpoints
    if cpu == 0:
        session.console.attach(m)
    else:
        session.console.share(m)
    m.syscall_handler = session.services
    rnmi_vectors(session, m)
    return m


def rnmi_vectors(session, m) -> None:
    """``--rv-rnmi INT[,EXC]``: the hart's RNMI handler and RNMI exception
    handler addresses (priv. 8.5; numbers or the program's labels; EXC
    defaults to INT); a label the program lacks or a hart without Smrnmi:
    a note, the addresses stay 0."""
    from .options import parse_rnmi
    parts = parse_rnmi(session.options.rv_rnmi) or []
    if not parts:
        return
    note = None
    if not m.csrs.smrnmi:
        note = f"--rv-rnmi {session.options.rv_rnmi}: the hart has no Smrnmi (--rv-ext)"
    else:
        symbols = getattr(session.program, "symbols", {}) or {}
        vals = []
        for p in parts:
            try:
                vals.append(int(p, 0))
            except ValueError:
                if p not in symbols:
                    note = f"--rv-rnmi: no label {p!r} in the program (the handler stays at 0)"
                    break
                vals.append(symbols[p])
        else:
            m.rnmi_vector = vals[0] & m.amask
            m.rnmi_exc_vector = vals[-1] & m.amask
    if note is not None and note not in session.program_notes:
        session.program_notes.append(note)


def cluster_config(session) -> tuple[int, int] | None:
    """(cores, harts a core) of the P8700's cluster (emips.p8700cps): the
    preset always has its Coherence Manager and CPC, even with one hart."""
    o = session.options
    if o.core != "p8700":
        return None
    return o.cores, o.vps


def make_cluster(session, mem):
    """The P8700 cluster on the session's memory: every hart as
    :func:`make_core` makes it, the reset PC = the program's entry page (the
    GCR block below it, every RESET_BASE at it)."""
    from .p8700cps import Cluster
    o = session.options
    cores, harts = cluster_config(session)
    entry = getattr(session.program, "entry", None)
    reset_pc = (entry if entry is not None else 0x00400000) & ~0xFFF

    def factory(cl, core, hart, pc, caches):
        m = make_core(session, core * harts + hart, mem,
                      caches if caches is not None else None)
        m.pc = m.reset_pc = pc
        return m
    hw = replace(o, cache_init="hardware")   # (MD01502 3.5: no software can initialize them)
    from .options import parse_waits
    return Cluster(cores, harts, factory, reset_pc, timer_div=o.count_rate,
                   on_start=session._core_started,
                   caches=lambda c: hw.caches(core=c, shared=True), l2=o.shared_levels()[0],
                   iocu=1 if o.iocu else 0, l2_waits=parse_waits(o.l2_waits) or (0, 0, 0),
                   dspram=o.dspram * 1024)


def platform_config(session) -> str | None:
    """A generic hart's platform controller (LF9: ``--rv-intc``), or None."""
    o = session.options
    if o.core == "p8700" or o.rv_intc == "none":
        return None
    return o.rv_intc


def make_platform(session, machines):
    """Spike's CLINT / SSWI and the chosen controller on ``machines``
    (emips.rvplatform); the console's TX / RX lines are its sources 1 / 2."""
    from .rvplatform import Platform
    o = session.options
    return Platform(machines, controller=o.rv_intc, timer_div=o.count_rate)


#: the console's lines on a platform: TX ready -> source 1, RX ready -> 2
CONSOLE_SOURCES = {0: 1, 1: 2}


def console_sink(session):
    """The console's interrupt lines -> the hart's MEIP (bit 11); on a
    platform (``--rv-intc``) sources 1 (TX) / 2 (RX) of its PLIC / APLIC; on
    a P8700 the APLIC's source 4 (the Boston board's UART, QEMU's)."""
    levels = {}

    def sink(line: int, on: bool) -> None:
        levels[line] = on
        cl = session.rvcps
        if cl is not None and cl.ic is not None:    # (the P8700: the APLIC's UART source)
            from .p8700ic import UART_SOURCE
            cl.ic.set_input(UART_SOURCE, any(levels.values()))
        elif session.rvplat is not None:
            session.rvplat.set_input(CONSOLE_SOURCES[line], on)
        elif session.machines:
            session.machines[0].set_interrupt(11, any(levels.values()))
    return sink


def boot(session, m) -> None:
    """The kernel / user boot of a P8700 (:func:`p8700_firmware`); the
    user boot: U-mode (as mret leaves it, MPP = U), the counters readable
    there (cycle, time, instret and the hpmcounters the hart has)."""
    if session.options.boot == "reset":
        return
    if m.p8700 is not None:
        p8700_firmware(m)
    if session.options.boot == "guest":
        guest_boot(m)
        return
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
    m.csrs.mcounteren = m.csrs.counter_mask()   # CY, TM, IR (and the hpmcounters)
    if 1 in m.modes:
        m.csrs.scounteren = m.csrs.counter_mask()


def guest_boot(m) -> None:
    """LF7's guest boot: what a hypervisor does to start a guest that owns
    its memory one to one (priv. ch. 21) — hgatp and vsatp Bare, the
    counters enabled (mcounteren, hcounteren, scounteren), vsstatus.FS /
    VS Initial beside mstatus's (21.2.11: both must be on), then MRET's
    entry into VS-mode (V = 1, S)."""
    from . import rvcsr as CSR
    c = m.csrs
    c.mcounteren = c.hcounteren = c.scounteren = c.counter_mask()
    c.vsstatus |= c.mstatus & ((3 << CSR.FS_SHIFT) | (3 << CSR.VS_SHIFT))
    m.priv, m.v = 1, 1


#: the P8700's PMP entries (its 64 PMAs: MD01502 7.3.16)
P8700_PMP = 64


def _napot(base: int, size: int) -> int | None:
    """pmpaddr of a NAPOT region (priv. 3.7.1.1), None when it is not one."""
    if size < 8 or size & (size - 1) or base & (size - 1):
        return None
    return (base >> 2) | ((size >> 3) - 1)


def p8700_firmware(m) -> None:
    """What the P8700's boot firmware sets up — OpenSBI's (``platform/
    generic/mips/boston.c`` ``boston_nascent_init`` and ``p8700.c``
    ``mips_p8700_pmp_set``): PMP entries with their PMAs — every I/O
    region uncached (the platform's devices, RARS's MMIO page), the last
    entry all of memory cacheable with speculation, read / write /
    execute for S and U —, and mipsconfig6.AMO_II / RDTM_II (an AMO or a
    ``time`` read is an illegal instruction: OpenSBI emulates them in its
    handler; a program here sees the trap).  Misaligned accesses stay in
    hardware (OpenSBI sets mipsconfig7.DMALN and emulates them)."""
    from . import p8700 as P
    from .rvpmp import A_SHIFT, NAPOT, R, TOR, W, X
    p, pmp = m.p8700, m.pmp
    rwx = R | W | X
    k = 0
    regions = [(0xFFFF0000, 0x10000)] + [(lo, hi - lo) for lo, hi in m.io_regions]
    for base, size in regions:
        a = _napot(base, size)
        if a is not None and k < P8700_PMP - 1:
            pmp.cfg[k] = rwx | NAPOT << A_SHIFT
            pmp.addr[k] = a
            p.pmacfg[k] = 2                     # (uncached)
            k += 1
        elif k < P8700_PMP - 2:                 # (TOR: two entries)
            pmp.addr[k] = base >> 2
            pmp.addr[k + 1] = (base + size) >> 2
            pmp.cfg[k + 1] = rwx | TOR << A_SHIFT
            p.pmacfg[k + 1] = 2
            k += 2
    last = P8700_PMP - 1
    pmp.cfg[last] = rwx | NAPOT << A_SHIFT
    pmp.addr[last] = pmp.addr_mask              # (all of the address space)
    p.pmacfg[last] = 0x08                       # (cacheable, speculation enabled)
    pmp.version += 1
    p._pma_version += 1
    p.config6 |= P.AMO_II | P.RDTM_II
    # boston_nascent_init's cluster and core set-up (LF15): the L2 prefetcher
    # (4 KB pages, PFTEN; WRI_MODE 2, code prefetch, every port), every L1
    # prefetch tracker on, each with an age prescaler of 8
    cl = m.cluster
    if cl is not None and cl.cm.l2pf is not None:
        cl.cm.l2pf.put(0, 0xFFFFF110, 0xFFFFFFFF)
        cl.cm.l2pf.put(1, 0x15FF, 0xFFFFFFFF)
    pf = p.l1pf()
    if pf is not None:
        pf.pfen = 0xFF
        from .p8700mem import PRESCALER
        for k in range(len(pf.trackers)):
            pf.write(k, PRESCALER, pf.read(k, PRESCALER) | 8)


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
    or base + 4 x cause in vectored mode), "mips" for the P8700's
    mipstvec, else None."""
    tvecs = [("m", m.csrs.mtvec), ("s", getattr(m.csrs, "stvec", None))]
    if m.csrs.has_h:                            # (LF7: a trap into VS-mode)
        tvecs.append(("vs", m.csrs.vstvec))
    if m.p8700 is not None:                     # (the custom exceptions: MD01502 7.3.1)
        tvecs.append(("mips", m.p8700.mipstvec))
    for kind, value in tvecs:
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
    if str(log[-2].get("name", "")).endswith("(mipstvec)"):
        kind = "mips"                           # (a P8700 custom exception)
    elif log[-2].get("virt"):
        kind = "vs"                             # (into VS-mode: priv. 21.6.2)
    elif log[-2].get("to") in (1, 3):
        kind = "s" if log[-2]["to"] == 1 else "m"
    tvec = f"{kind}tvec"
    if kind == "mips":
        kind = "m"
    c = m.csrs
    cause, epc = {"m": (c.mcause, c.mepc), "s": (c.scause, c.sepc),
                  "vs": (c.vscause, c.vsepc)}[kind]
    interrupt = bool(cause >> (m.xlen - 1))
    num = cause & ~(1 << (m.xlen - 1))
    name = (INTERRUPT_NAMES if interrupt else CAUSE_NAMES).get(num, str(num))
    if tvec == "mipstvec":
        from .p8700 import CAUSE_NAMES as P
        name = P.get(num, name)
    digits = 8 if m.xlen == 32 else 16
    why = ""
    if m.p8700 is not None and not interrupt and num in (2, 26):
        i = log[-2].get("word")
        if num == 26:
            why = " (the P8700 has no time CSR: a read traps for the firmware to emulate)"
        elif i is not None and i & 0x7F == 0x2F and i >> 27 != 2 and i >> 27 != 3:
            why = (" (the P8700 has no AMO instructions, only LR / SC: an AMO traps for the "
                   "firmware to emulate — use an lr / sc loop)")
    return (f" — the hart took {'interrupt' if interrupt else 'exception'} {name}{why} "
            f"({kind}cause {num}, {kind}epc 0x{epc:0{digits}x}) to {tvec} = "
            f"0x{m.pc:0{digits}x}, but there is no trap handler there: write one and point "
            f"{tvec} at it (csrw {'0x7c0' if tvec == 'mipstvec' else tvec}, handler"
            + ("; in the guest: csrw stvec, handler" if kind == "vs" else "") + "), "
            "or choose Settings > Exceptions: "
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
IRQ_BITS = ((11, "MEI"), (9, "SEI"), (7, "MTI"), (5, "STI"), (3, "MSI"), (1, "SSI"),
            (12, "SGEI"), (10, "VSEI"), (6, "VSTI"), (2, "VSSI"))


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
           + f"; the hart runs in {m.mode_name}-mode",
           f"mip {_irq_names(c.mip())}   (0x{c.mip():x})",
           f"mie {_irq_names(c.mie)}   (0x{c.mie:x})"]
    if c.has_s:
        out.append(f"mideleg {_irq_names(c.mideleg)} (to {'HS' if c.has_h else 'S'}-mode), "
                   f"medeleg 0x{c.medeleg:x}")
    if c.has_h:                                 # (21.2.2-21.2.3, 21.2.12)
        out.append(f"hideleg {_irq_names(c.hideleg & c.mideleg)} (to VS-mode), hedeleg "
                   f"0x{c.hedeleg:x}; hvip {_irq_names(c.hvip | c.mip_sw & 4)}; "
                   f"vsstatus.SIE {c.vsstatus >> 1 & 1}")
    tvecs = (("mtvec", c.mtvec),) + ((("stvec", c.stvec),) if c.has_s else ()) \
        + ((("vstvec", c.vstvec),) if c.has_h else ())
    for name, tvec in tvecs:
        out.append(f"{name} 0x{tvec & ~3 & m.amask:x}: "
                   + ("vectored (BASE + 4 x cause for interrupts)" if tvec & 3 == 1
                      else "direct"))
    if c.smrnmi:                                # (priv. ch. 8)
        s = c.mnstatus
        rnmi = "not requested" if m.rnmi_pending is None else f"pending, cause {m.rnmi_pending}"
        out.append(f"mnstatus.NMIE {s >> 3 & 1}"
                   + (" (every interrupt masked: RNMI handler or reset; MNRET sets it)"
                      if not s & CSR.NMIE else "")
                   + f"; RNMI {rnmi}; handlers 0x{m.rnmi_vector:x} / exceptions "
                   f"0x{m.rnmi_exc_vector:x}")
    if c.aia is not None:                       # (the AIA: the tops, its CSRs, the IMSIC)
        out += c.aia.lines()
    out += cfi_lines(m)
    if c.hpm is not None and c.sscofpmf:        # (Sscofpmf 20.1-20.2)
        ovf = c.hpm.overflowed()
        out.append("counter overflow (LCOFI 13): "
                   + (", ".join(f"mhpmcounter{n}" for n in range(3, 32) if ovf >> n & 1)
                      + " (OF)" if ovf else "no OF bit set"))
    req = m.interrupt_request()
    out.append("taken before the next instruction: "
               + (f"{INTERRUPT_NAMES.get(req[0], req[0])} -> "
                  f"{({-1: 'VS', 1: 'HS' if c.has_h else 'S', 3: 'M'})[req[1]]}-mode"
                  if req is not None else "none")
               + ("" if not m.waiting else "  (WFI: waiting)" if m.wrs_until is None
                  else "  (WRS: waiting on the reservation set)"))
    return out


def cfi_lines(m) -> list[str]:
    """Control-flow integrity (LF10; unpriv. ch. 33, priv. ch. 22): ELP and
    each mode's xLPE (Zicfilp), the shadow stack's ssp and each mode's
    xSSE (Zicfiss) — the Interrupts tab, the debugger's ``cfi``."""
    from . import rvcsr as CSR
    c = m.csrs
    out = []
    modes = [(3, 0, "M")] + ([(1, 0, "S" if not c.has_h else "HS")] if c.has_s else []) \
        + ([(0, 0, "U")] if c.has_u else []) + ([(1, 1, "VS"), (0, 1, "VU")] if c.has_h else [])
    if c.zicfilp:
        on = [n for p, v, n in modes if c.xlpe(p, v)]
        pelp = [n for n, bit in (("MPELP", c.mstatus >> CSR.MPELP & 1),
                                 ("SPELP", c.mstatus >> CSR.SPELP & 1 if c.has_s else 0),
                                 ("vsstatus.SPELP", c.vsstatus >> CSR.SPELP & 1
                                  if c.has_h else 0)) if bit]
        elp = "LP_EXPECTED (the next instruction must be an LPAD)" if c.elp \
            else "NO_LP_EXPECTED"
        out.append(f"Zicfilp: ELP {elp}; landing pads enforced in "
                   f"{', '.join(on) or 'no mode'}-mode"
                   + (f"; {', '.join(pelp)} set" if pelp else ""))
    if c.zicfiss:
        prv, v = m.priv, m.v
        on = []
        for p, vv, n in modes:                  # (xSSE per mode: Table 53)
            m.priv, m.v = p, vv
            if c.xsse():
                on.append(n)
        m.priv, m.v = prv, v
        out.append(f"Zicfiss: ssp 0x{c.ssp:x}; the shadow stack is active in "
                   f"{', '.join(on) or 'no mode'}-mode (now: "
                   f"{'active' if c.xsse() else 'inactive: SSPUSH / SSPOPCHK / SSRDP are MOPs'})")
    return out


def debug_lines(m) -> list[str]:
    """The hart's debug state (Debug spec ch. 4-5): Debug Mode, dcsr, dpc,
    the triggers (the debugger's ``ejtag`` on MIPS, the Debug tab)."""
    from .rvdebug import CAUSE_SHIFT, DISABLED, EBREAKM, EBREAKS, EBREAKU, STEP
    c = m.csrs
    causes = {1: "ebreak", 2: "a trigger", 3: "a halt request", 4: "a single step",
              5: "a reset halt request"}
    if m.debug_mode and m.hybrid is not None:
        out = [f"Debug Mode (MIPS Hybrid Debug): entered by "
               f"{causes.get(c.dcsr >> CAUSE_SHIFT & 7, '?')}, dpc 0x{c.dpc:x}; the debug "
               f"handler runs (vector 0x{m.hybrid.vector():x})"]
    elif m.debug_mode:
        out = [f"Debug Mode (Debug spec 4.1): entered by "
               f"{causes.get(c.dcsr >> CAUSE_SHIFT & 7, '?')}, dpc 0x{c.dpc:x}; the Debug "
               "Module drives the hart"]
    else:
        out = ["not in Debug Mode"]
    from .rvdebug import DCSR_V, EBREAKVS, EBREAKVU
    ebreak = [k for bit, k in ((EBREAKM, "M"), (EBREAKS, "S"), (EBREAKU, "U"),
                               (EBREAKVS, "VS"), (EBREAKVU, "VU")) if c.dcsr & bit]
    out.append(f"dcsr 0x{c.dcsr:08x}: prv {c.dcsr & 3}"
               + (f", v {int(bool(c.dcsr & DCSR_V))}" if c.has_h else "")
               + f", step {int(bool(c.dcsr & STEP))}, "
               "ebreak enters Debug Mode in " + (", ".join(ebreak) or "no mode"))
    used = [k for k, t in enumerate(m.tm.triggers) if t.type != DISABLED]
    out.append(f"triggers: {m.tm.count} (Sdtrig), "
               + (f"in use: {', '.join(str(k) for k in used)}" if used else "none in use"))
    if m.debug_module is not None:
        out.append("a Debug Module is attached (the JTAG DTM)")
    if m.hybrid is not None:                    # (the P8700: LF16)
        out += m.hybrid.lines()
        if m.cluster is not None:
            out += m.cluster.dbu.lines()
    return out


# -- the P8700's Debug Unit and probe (LF16: emips.p8700dbu) ----------------------------

def dbu_action(cl, op: str, args: tuple) -> int:
    """A probe's action on the cluster's Debug Unit — TAP scans only (the
    Session records them: Reset and Step Back replay them).  Returns what it
    read (0 otherwise)."""
    from . import ejtag as EJ
    from .p8700dbu import CMD_READ, CMD_RESUME, CMD_STEP, CMD_WRITE, Probe
    p = Probe(cl.dbu)
    if op == "probe":
        p.load_monitor()
        p.attach(trap=bool(args[0]) if args else True)
    elif op == "detach":
        p.detach()
    elif op == "halt":
        p.halt(args[0])
    elif op == "dint-all":
        from . import tap as T
        p.control(p._keep() | T.ECR_EJTAGBRK)
    elif op in ("resume", "step"):
        p.command(args[0], CMD_RESUME if op == "resume" else CMD_STEP)
    elif op == "read":
        p.command(args[0], CMD_READ, args[1])
    elif op == "write":
        p.command(args[0], CMD_WRITE, args[1], args[2])
    elif op == "reg":
        p.set_register(args[0], args[1], args[2])
    elif op == "pc":
        p.set_pc(args[0], args[1])
    elif op == "ib":                            # (k, address or None, hart or None)
        k, addr, hart = args
        base = 0x1100 + 0x100 * k
        if addr is None:
            p.drseg(0, base + EJ.IBC, 0)
        else:
            p.drseg(0, base + EJ.IBA, addr)
            p.drseg(0, base + EJ.IBM, 0)
            tc = EJ.BC_TCUSE | hart << EJ.BC_TC_SHIFT if hart is not None else 0
            p.drseg(0, base + EJ.IBC, EJ.BC_BE | tc)
    elif op == "db":                            # (k, address or None, kind, value, hart)
        k, addr, kind, value, hart = args
        base = 0x2100 + 0x100 * k
        if addr is None:
            p.drseg(0, base + EJ.DBC, 0)
        else:
            p.drseg(0, base + EJ.DBA, addr)
            p.drseg(0, base + EJ.DBM, 0)
            if value is not None:
                p.drseg(0, base + EJ.DBV, value)
            c = EJ.BC_BE | (0 if value is not None else 0xFF << EJ.DBC_BLM_SHIFT) \
                | (EJ.DBC_NOLB if kind == "store" else EJ.DBC_NOSB if kind == "load" else 0) \
                | (EJ.BC_TCUSE | hart << EJ.BC_TC_SHIFT if hart is not None else 0)
            p.drseg(0, base + EJ.DBC, c)
    elif op == "fdc-send":                      # (the probe sends: the TAP's FDC register)
        from .p8700dbu import FDC as IR_FDC
        p.tap.scan_ir(IR_FDC)
        p.tap.scan_dr(1 << 37 | (args[0] & 0xF) << 32 | args[1] & 0xFFFFFFFF)
    elif op == "fdc-take":
        from .p8700dbu import FDC as IR_FDC
        p.tap.scan_ir(IR_FDC)
        v = p.tap.scan_dr(0)
        if v >> 37 & 1:
            p.tap.scan_dr(1 << 36)
        return v
    else:
        raise ValueError(f"unknown Debug Unit action {op!r}")
    return p.tap.data if op == "read" else 0


def dbu_hart_lines(cl, i: int) -> list[str]:
    """Hart ``i`` as the probe's monitor reports it (a view of the DBU's RAM:
    the mailbox, the registers it saved)."""
    from .p8700dbu import (MAILBOX, MAILBOX_SIZE, MB_DATA, MB_DPC, MB_ENTRIES, MB_ERROR,
                           MB_REGS, MB_STATE, MB_COMMAND)
    from .riscv import GPR_NAMES
    d = cl.dbu
    c, h = divmod(i, cl.harts)
    b = MAILBOX + MAILBOX_SIZE * (2 * c + h)
    rd = lambda off: d.ram_read(b + off, 8)    # noqa: E731
    st, cmd = rd(MB_STATE), rd(MB_COMMAND)
    out = [f"hart {i}: " + ("in the monitor" if st and not cmd else
                            "running a command" if st else "not in the monitor")
           + f" (entered {rd(MB_ENTRIES)} time(s)), pc 0x{rd(MB_DPC):x}"
           + (f", the last command faulted (exception {rd(MB_ERROR) - 1})" if rd(MB_ERROR)
              else "") + f", data 0x{rd(MB_DATA):x}"]
    if st:
        regs = [f"{GPR_NAMES[n]:>4} 0x{rd(MB_REGS + 8 * n):016x}" for n in range(1, 32)]
        for k in range(0, 31, 4):
            out.append("   " + "  ".join(regs[k:k + 4]))
    return out


def dbu_lines(cl) -> list[str]:
    """The Debug Unit, the FDC, the Trace Funnel and the CM's counters (the
    debugger's ``dbu``)."""
    out = cl.dbu.lines()
    out += [f"  breakpoints: {cl.bp.ib_count} instruction, {cl.bp.db_count} data (shared)"]
    out += cl.fdc.lines() + cl.trf.lines() + cl.cmpc.lines()
    return out


# -- the P8700's own registers (PLAN M25 phase 5b; emips.p8700) -------------------

def configure_counter(m, i: int, event: int | None, modes: str) -> None:
    """mhpmevent(3 + i) = ``event`` counting in ``modes`` (letters u / s /
    m; k = m), as software writes it — the P8700's (MD01502 11.1.2) or,
    with Zihpm, the generic hart's (SBI event_idx; the modes as Sscofpmf's
    xINH bits — VS / VU follow S / U — when it has them); None: event 0."""
    p = m.p8700
    hpm = m.csrs.hpm
    modes = modes.lower().replace("k", "m")
    value = 0 if event is None else event
    if p is not None:
        if not 0 <= i < p.counters.N:
            return
        from . import p8700 as P
        for letter, bit in (("m", P.MINH), ("s", P.SINH), ("u", P.UINH)):
            if event is not None and letter not in modes:
                value |= bit
        p.counters.write_event(i, value)
        if event is not None:
            p.counters.count[i] = 0
        return
    if hpm is None or not 0 <= i < 32 - 3:
        return
    from . import rvhpm as H
    if event is not None and hpm.sscofpmf:
        for letter, bits in (("m", H.MINH), ("s", H.SINH | H.VSINH), ("u", H.UINH | H.VUINH)):
            if letter not in modes:
                value |= bits
    hpm.write_event(3 + i, value)
    if event is not None:
        hpm.write_count(3 + i, 0)


def find_event(name: str, m=None) -> int | None:
    """An event by number or name: the P8700's (Table 11.4), or with ``m``
    a generic hart the SBI event_idx eMIPS counts (emips.rvhpm)."""
    if m is not None and m.p8700 is None:
        from .rvhpm import EVENTS as GENERIC
        try:
            n = int(name, 0)
        except ValueError:
            want = name.lower().replace("_", " ")
            n = next((k for k, ev in GENERIC.items() if ev.lower() == want), None)
        return n if n in GENERIC else None
    from .p8700 import EVENTS
    try:
        n = int(name, 0)
    except ValueError:
        n = next((k for k, (ev, _) in EVENTS.items() if ev.lower() == name.lower()), None)
    return n if n in EVENTS else None


def event_lines(m=None) -> list[str]:
    if m is not None and m.p8700 is None:
        from .rvhpm import EVENTS as GENERIC, never
        return [f"0x{n:05x} {name}" + ("   (never counts on eMIPS)" if never(n) else "")
                for n, name in sorted(GENERIC.items())]
    from .p8700 import EVENTS
    src = {"retire": "each instruction", "cycle": "each clock", "trap": "each trap",
           "pipeline": "the pipeline (the P8700's or the 5-stage)",
           "cache": "the caches' and TLBs' counters", "prefetch": "the L1 prefetcher",
           "": "not modelled: never counts"}
    return [f"{n:4} {name:36} {src[how]}" for n, (name, how) in sorted(EVENTS.items())]


def p8700_lines(m) -> list[str]:
    """The P8700's registers (the debugger's ``p8700``; MD01502 ch. 7)."""
    p = m.p8700
    if p is None:
        return ["this hart is not a P8700 (--core p8700)"]
    from . import p8700 as P
    d = 16
    routes = []
    for n, (std, custom) in sorted(P.ROUTES.items(), reverse=True):
        name = ("MEI", "MSI", "MTI", "SEI", "STI", "VSEI")[5 - n]
        routes.append(f"{name} -> mip[{std if p.mipsintctl >> n & 1 else custom}]")
    tvec = p.mipstvec
    out = [f"mhartid 0x{p.hartid:x} (cluster {p.cluster}, core {p.core}, hart {p.hart}); "
           f"mvendorid 0x{P.MVENDORID:x}, marchid 0x{P.MARCHID:x}",
           f"mipstvec  0x{tvec:0{d}x} ({'vectored' if tvec & 3 == 1 else 'direct'}: the "
           "custom exceptions)",
           f"mipsintctl 0x{p.mipsintctl:02x}: " + ", ".join(routes),
           f"mipsconfig5 0x{p.config5:x}: ERL {p.config5 & 1}"
           f"{', MDIAGL' if p.config5 & P.MDIAGL else ''}",
           f"mipsconfig6 0x{p.config6:x}: AMO_II {p.config6 >> 2 & 1} (an AMO: "
           f"{'illegal instruction' if p.config6 & P.AMO_II else 'mipstvec, mcause 2'}), "
           f"RDTM_II {p.config6 >> 1 & 1} (time: "
           f"{'illegal instruction' if p.config6 & P.RDTM_II else 'mipstvec, mcause 26'})",
           f"mipsconfig7 0x{p.config7:x}: misaligned loads / stores "
           f"{'trap' if p.misaligned_traps else 'in hardware'}",
           f"mipsconfig1 0x{p.config1():08x}",
           f"mipserrctl 0x{p.errors().errctl:08x}: ECC checks "
           f"{'on' if p.errors().pe else 'off'} (PE); " + p.errors().describe()]
    pmas = []
    names = {0: "cacheable", 2: "uncached", 3: "uncached accelerated"}
    for k in range(m.pmp.entries):
        rng = m.pmp._range(k)
        if rng is not None:
            cfg = p.pmacfg[k]
            pmas.append(f"  pma{k}: 0x{rng[0]:x}-0x{rng[1] - 1:x} "
                        f"{names.get(cfg & 7, 'reserved (uncached)')}"
                        f"{', speculative' if cfg & 8 else ''}")
    out.append("PMAs (pmacfg, by PMP entry):" if pmas else
               "PMAs: no PMP entry configured (the platform's attributes)")
    out += pmas
    if p.erl:
        out.append("  ERL = 1: every access uncached, no interrupts")
    out += p.counters.lines()
    if hasattr(m.mmu, "lines"):                 # (its TLBs: emips.p8700mmu)
        out += m.mmu.lines(m)
    if m.cluster is not None:                   # (the cluster: emips.p8700cps)
        cl = m.cluster
        out += cl.lines()
        if cl.ic is not None:
            out += cl.ic.lines()
        out += cl.l1pf[p.core].lines()          # (LF15: emips.p8700mem, emips.axicdma)
        if cl.dspram[p.core] is not None:
            out += cl.dspram[p.core].lines()
        out += cl.cdma.lines()
    return out


# ------------------------------------------------------------ the vector unit (LF6)

#: the vector views: "e8" ... "e64" with x (hex), s (signed), u (unsigned)
#: or f (IEEE binary16 / 32 / 64)
VECTOR_VIEWS = {f"e{w}.{how}": (w, how) for w in (8, 16, 32, 64) for how in ("x", "s", "u")}
VECTOR_VIEWS.update({"e16.f": (16, "f"), "e32.f": (32, "f"), "e64.f": (64, "f")})


def _element_text(e: int, bits: int, how: str) -> str:
    if how == "x":
        return f"{e:0{bits // 4}x}"
    if how == "u":
        return str(e)
    if how == "s":
        return str(e - (1 << bits) if e >> (bits - 1) & 1 else e)
    import struct
    fmt = {16: ">e", 32: ">f", 64: ">d"}[bits]
    return f"{struct.unpack(fmt, e.to_bytes(bits // 8, 'big'))[0]:.7g}"


def vector_state_lines(m) -> list[str]:
    """The vector unit's configuration (LF6): VLEN / ELEN and the FP
    element widths, mstatus.VS, vtype decoded, vl / VLMAX, vstart, vxrm,
    vxsat — the head of the debugger's ``vec`` and the V tab."""
    vu = m.vector
    if vu is None:
        return ["this hart has no vector unit (--rv-ext with v, or zve32x / zve32f / "
                "zve64x / zve64f / zve64d)"]
    vs = ("Off", "Initial", "Clean", "Dirty")[m.csrs.status_field(9, 3)]
    fp = ", ".join(f"e{w}" for w in sorted(vu.fp)) or "none"
    rm = ("rnu", "rne", "rdn", "rod")[vu.vxrm]
    return [f"VLEN {vu.vlen}, ELEN {vu.elen}; FP elements: {fp}; mstatus.VS {vs}"
            + (" — vector instructions are illegal (set mstatus.VS)" if vs == "Off" else ""),
            f"vtype 0x{vu.vtype:x}: {vu.describe_vtype()}; vl {vu.vl} of VLMAX {vu.vlmax}; "
            f"vstart {vu.vstart}; vxrm {vu.vxrm} ({rm}); vxsat {vu.vxsat}; vlenb {vu.vlenb}"]


def vector_lines(m, regs: list[int] | None, view: str) -> list[str]:
    """The debugger's ``vec``: the state, then the registers ``regs``
    (default: every one not zero) as VLEN-bit values, or element by
    element (element 0 first) in ``view`` (:data:`VECTOR_VIEWS`)."""
    lines = vector_state_lines(m)
    vu = m.vector
    if vu is None:
        return lines
    if regs is None:
        regs = [n for n in range(32) if vu.register(n)] or [0]
    for n in regs:
        v = vu.register(n)
        if not view:
            lines.append(f"v{n:<2} 0x{v:0{2 * vu.vlenb}x}")
            continue
        bits, how = VECTOR_VIEWS[view]
        es = [v >> (bits * k) & ((1 << bits) - 1) for k in range(vu.vlen // bits)]
        lines.append(f"v{n:<2} {view}: " + " ".join(
            f"[{k}] {_element_text(e, bits, how)}" for k, e in enumerate(es)))
    return lines


def vector_rows(m, view: str = "e32.x") -> list[tuple[str, int, list[str]]]:
    """``(name, value, element texts)`` of v0-v31 in ``view`` — the
    register pane's V tab (empty without a vector unit)."""
    vu = m.vector if m is not None else None
    if vu is None:
        return []
    bits, how = VECTOR_VIEWS[view]
    rows = []
    for n in range(32):
        v = vu.register(n)
        rows.append((f"v{n}", v, [_element_text(v >> (bits * k) & ((1 << bits) - 1), bits, how)
                                  for k in range(vu.vlen // bits)]))
    return rows


# -- RISC-V trace (LF11: emips.rvtrace, etrace, ntrace) ---------------------------------

#: trTeInstFeatures bits by option name (the control interface's Table
#: "Trace Instruction Features"; E-Trace: 0-5, N-Trace: 2, 3, 6-10)
TRACE_FEATURES = {"full": 1, "notrapaddr": 2, "sijump": 4, "return": 8, "predict": 0x10,
                  "cache": 0x20, "repeat": 0x100, "alljumps": 0x200, "msb": 0x400}
TRACE_OPTION_HELP = {
    "full": "E-Trace: full addresses (trTeInstNoAddrDiff)",
    "notrapaddr": "E-Trace: implicit exception (trTeInstNoTrapAddr)",
    "sijump": "sequentially inferable jumps (trTeInstEnSequentialJump)",
    "return": "implicit return (trTeInstEnImplicitReturn)",
    "predict": "E-Trace: branch prediction (trTeInstEnBranchPrediction)",
    "cache": "E-Trace: jump target cache (trTeInstEnJumpTargetCache)",
    "repeat": "N-Trace: repeated history / RepeatBranch (trTeInstEnRepeatedHistory)",
    "alljumps": "N-Trace: direct jumps as taken branches (trTeInstEnAllJumps)",
    "msb": "N-Trace: extended address MSBs (trTeInstExtendAddrMSB)",
}
SYNC_MODES = {"off": 0, "packets": 1, "cycles": 2, "halfwords": 3}


def make_trace(session, machines):
    """The trace components on generic harts (the P8700 has its own: LF16)."""
    from .rvtrace import TraceSystem
    return TraceSystem(machines)


def trace_writes(session, *, protocol: str = "e", options: tuple = (), htm: bool = True,
                 ir_mode: int = 3, sync: str = "packets", sync_max: int = 4,
                 timestamps: bool = False, prescale: int = 0, stall: bool = False,
                 triggers: bool = True, smem: tuple[int, int] | None = None,
                 stop_on_wrap: bool = False, async_freq: int = 0, context: bool = False,
                 tracing: bool = True) -> list[tuple[int, int]]:
    """The register writes that start tracing on every hart, in the control
    interface's order (Enabling and Disabling: the sink, the funnel, then
    the encoders, trTeEnable last).  ``protocol`` "e" (E-Trace) or "n"
    (N-Trace, ``htm`` or BTM, implicit return mode ``ir_mode``); ``smem``:
    (address, bytes) of a buffer in system memory instead of the SRAM."""
    from .rvtrace import MAP
    ts = session.rvtrace
    n = len(ts.encoders)
    out: list[tuple[int, int]] = []
    if getattr(ts, "trf", None) is not None:    # (the P8700: the Trace Funnel, LF16)
        trf = p8700_trace_address(ts, None, 0)
        out += [(trf, 1 << 31 | 1 << 15), (trf, 1 << 31 | 1)]   # (WE + TR, then WE + EN)
    sink = MAP["ramsink"]
    ctl = 1 | (1 << 4 if smem else 0) | (1 << 8 if stop_on_wrap else 0) | (async_freq & 7) << 12
    if getattr(ts, "trf", None) is None:
        out += [(sink, 0), (sink, 1), (sink, ctl)]
    if getattr(ts, "trf", None) is not None:
        pass
    elif smem:
        base, size = smem
        out += [(sink + 0x10, base & 0xFFFFFFFF), (sink + 0x14, base >> 32),
                (sink + 0x18, (base + size - 4) & 0xFFFFFFFF), (sink + 0x1C, (base + size - 4) >> 32)]
        out += [(sink + 0x20, base & 0xFFFFFFFF), (sink + 0x24, base >> 32)]
    else:
        out.append((sink + 0x20, 0))
    if getattr(ts, "trf", None) is None:
        out.append((sink, ctl | 2))
    if ts.funnel is not None:
        f = MAP["funnel"]
        out += [(f, 1), (f, 3)]
    fmt = 0 if protocol == "e" else 1
    feats = 0
    for o in options:
        feats |= TRACE_FEATURES[o]
    if fmt == 1 and feats & 8:
        feats |= (ir_mode & 3) << 6
    src_bits = (n - 1).bit_length() if n > 1 else 0
    mode = 7 if fmt == 0 else (6 if htm else 3)
    for k in range(n):
        e = MAP["encoder"] + 0x1000 * k if getattr(ts, "trf", None) is None \
            else p8700_trace_address(ts, k, 0)
        base = 1 | mode << 4 | (fmt and context) << 9 | int(triggers) << 11 \
            | int(stall) << 13 | SYNC_MODES[sync] << 16 | (sync_max & 15) << 20 | fmt << 24
        out += [(e, 0), (e, 1), (e, 1 | fmt << 24)]
        out.append((e + 0x008, feats | (k << 16 if src_bits else 0) | src_bits << 28))
        out += [(e + 0x00C, 0), (e + 0x054, 0x4 << 12), (e + 0x058, 0)]
        if timestamps:
            out.append((e + 0x040, 1 | 2 | 3 << 4 | (prescale & 3) << 8 | 1 << 15))
        out.append((e, base | 2 | (4 if tracing else 0)))
    return out


def trace_stop_writes(session) -> list[tuple[int, int]]:
    """Disable and flush: the encoders (trTeEnable 0), the funnel, the sink."""
    from .rvtrace import MAP
    ts = session.rvtrace
    out = []
    if getattr(ts, "trf", None) is not None:    # (the P8700: LF16)
        out += [(p8700_trace_address(ts, k, 0), e.control & ~6)
                for k, e in enumerate(ts.encoders)]
        return out + [(p8700_trace_address(ts, None, 0),
                       1 << 31 | ts.trf.control & ~1 & 0x7FFFFFFF)]
    for k, e in enumerate(ts.encoders):
        out.append((MAP["encoder"] + 0x1000 * k, e.control & ~6))
    if ts.funnel is not None:
        out.append((MAP["funnel"], 1))
    out.append((MAP["ramsink"], ts.sink.control & ~2))
    return out


#: a P8700 trace register's address in trace_writes: a DBU master address
#: (emips.p8700dbu) marked by bit 63 — the probe reaches them through the DBU
P8700_TRACE = 1 << 63


def p8700_trace_address(ts, k: int | None, off: int) -> int:
    """Hart ``k``'s encoder register ``off`` (its drseg at 0x8000), or with
    ``k`` None the Trace Funnel's (GCR 0x3F100)."""
    from .p8700dbg import TRACE_BASE
    from .p8700dbu import HART_SHIFT, RING_GCR
    from .p8700oci import TRF_BASE
    if k is None:
        return P8700_TRACE | RING_GCR << 56 | TRF_BASE + off
    cl = ts.cl
    c, h = divmod(k, cl.harts)
    return P8700_TRACE | c << 56 | h << HART_SHIFT | TRACE_BASE + off


def trace_write(m, pa: int, value: int) -> None:
    """A trace register write as the debugger does it (System Bus Access;
    the P8700's: through its Debug Unit)."""
    if pa & P8700_TRACE:
        a = pa & ~P8700_TRACE
        m.cluster.dbu.master_write(a, value, 8 if a >> 56 == 0x22 else 4)
        return
    bus = m.trace_encoder.sys.bus
    bus.store(m, pa, value & 0xFFFFFFFF, 4)


def trace_bytes(session) -> bytes:
    return session.rvtrace.sink.contents()


def _trap_vector(m):
    """Implicit exception mode's handler address from the cause and the
    handler's privilege (the packet's): today's xtvec (direct, or vectored
    for an interrupt)."""
    def vector(cause: int, interrupt: bool, priv: int) -> int:
        c = m.csrs
        tvec = c.mtvec if priv == 3 else c.vstvec if priv == 6 else c.stvec
        base = tvec & ~3
        return base + 4 * cause if interrupt and tvec & 3 == 1 else base
    return vector


def trace_decode(session) -> dict[int, list]:
    """Each hart's regenerated flow from the sink's bytes: {hart: events}
    (emips.etrace.Event / emips.ntrace.Event) — the decoders read the
    program as a fetch would see it now."""
    from . import etrace as E
    from . import ntrace as N
    from .rvtrace import decapsulate, sync_sequence
    ts = session.rvtrace
    data = trace_bytes(session)
    out: dict[int, list] = {}
    encs = ts.encoders
    e0 = encs[0]
    if ts.sink.wrap and data:                   # (the oldest bytes are gone)
        if e0.format == 0:
            seq = sync_sequence(*ts.encap)
            k = data.find(seq)
            data = data[k + len(seq):] if k >= 0 else b""
        else:
            k = next((i for i, b in enumerate(data) if b & 3 == 3), len(data))
            data = data[k + 1:]
    if e0.format == 0:
        sbits = e0._src()[1]
        decs = {k: E.Decoder(e.params, (lambda m_: lambda a: _decode(m_, a))(e.m),
                             word=(lambda m_: lambda a, n: _word(m_, a, n))(e.m),
                             trap_vector=_trap_vector(e.m), jvt=e.m.csrs.jvt
                             if hasattr(e.m.csrs, "jvt") else 0, wait_sync=True)
                for k, e in enumerate(encs)}
        for sid, _ts, payload in decapsulate(data, sbits, e0.ts.enabled):
            d = decs.get(sid)
            if d is None:
                continue
            try:
                d.feed(payload)
            except E.DecodeError as exc:
                d.events.append(E.Event("error", info=str(exc)))
                decs[sid] = E.Decoder(d.p, d.instr, word=d.word, trap_vector=d.trap_vector,
                                      jvt=d.jvt, wait_sync=True)
                decs[sid].events = d.events
        for k, d in decs.items():
            out[k] = d.events
        return out
    mode = e0.nmode()
    decs_n = {k: N.Decoder(e.xlen, (lambda m_: lambda a: _decode(m_, a))(e.m), mode=e.nmode(),
                           word=(lambda m_: lambda a, n: _word(m_, a, n))(e.m))
              for k, e in enumerate(encs)}
    from .ntrace import _Reader, unpack
    for raw in _Reader(data).messages():
        msg = unpack(raw, src_bits=mode.src_bits, tstamp=mode.timestamps)
        d = decs_n.get(msg.src or 0)
        if d is None:
            continue
        try:
            d.message(msg)
        except N.DecodeError as exc:
            d.events.append(N.Event("error", info=str(exc)))
            d.pc = None
    for k, d in decs_n.items():
        out[k] = d.events
    return out


def _word(m, a: int, n: int) -> int:
    try:
        return int.from_bytes(bytes(m.read_byte(a + k) for k in range(n)), "little")
    except SimException:
        return 0


def trace_lines(session, last: int = 20, hart: int | None = None) -> list[str]:
    """The regenerated instructions of a hart (the last ``last``) and the
    trace's events (the debugger's ``trace show``)."""
    k = session.cpu if hart is None else hart
    events = trace_decode(session).get(k, [])
    if last:
        events = events[-last:]
    m = session.machines[k]
    out = []
    for e in events:
        if e.kind == "pc":
            out.append(f"   {e.addr:0{m.xlen // 4}x}  {instruction_text(m, e.addr)}")
        elif e.kind == "trap":
            what = "interrupt" if getattr(e, "interrupt", False) or e.info == "interrupt" \
                else "exception"
            cause = getattr(e, "cause", None)
            out.append(f"   -- {what}" + (f" (cause {cause})" if cause is not None and
                                          hasattr(e, "cause") else "")
                       + (f" at 0x{e.addr:x}" if e.addr else ""))
        else:
            out.append(f"   -- {e.kind}" + (f": {e.info}" if e.info else "")
                       + (f" (0x{e.addr:x})" if e.addr and e.kind != "lost" else ""))
    return out


def trace_packet_lines(session, last: int = 20, hart: int | None = None) -> list[str]:
    """The packets / messages the selected hart's encoder sent, with why
    (the debugger's ``trace packets``)."""
    k = session.cpu if hart is None else hart
    e = session.rvtrace.encoders[k]
    out = []
    for p in e.packets[-last:] if last else e.packets:
        raw = getattr(p, "raw", b"")
        if hasattr(p, "tcode"):
            fields = ", ".join(f"{a} {v:#x}" for a, v in p.fields.items())
            out.append(f"   {p.name:<22} {fields}  [{raw.hex(' ')}]  ({p.reason})")
        else:
            fields = ", ".join(f"{a} {v:#x}" for a, v, b in p.fields[1:] if b)
            out.append(f"   {p.name():<13} {fields}  [{raw.hex(' ')}]  ({p.reason})")
    return out


def trace_state(session) -> list[str]:
    """The trace components (the debugger's ``trace``)."""
    from .rvtrace import MAP
    ts = session.rvtrace
    out = []
    p8700 = getattr(ts, "trf", None) is not None
    for k, e in enumerate(ts.encoders):
        c = e.control
        if not c & 1:
            where = f"hart {k}'s drseg 0x8000" if p8700 else \
                f"0x{MAP['encoder'] + 0x1000 * k:08x}"
            out.append(f"encoder {k} ({where}): off (trTeActive 0)")
            continue
        fmt = ("E-Trace", "N-Trace")[e.format] if e.format < 2 else f"format {e.format}"
        mode = (c >> 4) & 7
        mname = {3: "BTM", 6: "HTM", 7: "", 0: "no instruction trace"}.get(mode, str(mode))
        feats = [n for n, b in TRACE_FEATURES.items() if e.features & b]
        sync = ("off", "packets", "cycles", "halfwords")[c >> 16 & 3]
        out.append(f"encoder {k}: {fmt}{' ' + mname if mname else ''}, "
                   f"{'enabled' if c & 2 else 'disabled'}, instruction trace "
                   f"{'on' if c & 4 else 'off'}" + (f"; {', '.join(feats)}" if feats else "")
                   + f"; sync by {sync} (2^{(c >> 20 & 15) + 4})"
                   + (", trigger actions" if c >> 11 & 1 else "")
                   + (", stall" if c >> 13 & 1 else "")
                   + (", OVERFLOW (trace lost)" if c >> 12 & 1 else "")
                   + (", timestamps" if e.ts.enabled else "")
                   + f"; {len(e.packets)} {'packets' if e.format == 0 else 'messages'}")
    if p8700:
        return out + ts.trf.lines()
    s = ts.sink
    if s.control & 1:
        where = (f"system memory 0x{s.start:x}-0x{s.limit + 3:x}" if s.smem
                 else f"its SRAM ({len(s.sram) // 1024} KB)")
        used = len(s.contents())
        out.append(f"RAM sink: {'enabled' if s.enabled else 'disabled'}, {where}; "
                   f"{used} bytes" + (" (wrapped: the oldest are overwritten)" if s.wrap
                                      else "") + (", stops on wrap" if s.control >> 8 & 1
                                                  else ""))
    else:
        out.append("RAM sink: off (trRamActive 0)")
    if ts.funnel is not None:
        f = ts.funnel
        out.append(f"funnel: {'enabled' if f.control & 2 else 'disabled'}"
                   + (f", inputs disabled 0x{f.disinput:x}" if f.disinput else ""))
    return out


# -- the P8700's own pipeline (LF14: emips.pipe8700) ---------------------------------

def _harts_a_core(session) -> int:
    return session.rvcps.harts if session.rvcps is not None else 1


def p8700_pipeline(session, m):
    """Hart ``m``'s view of its core's P8700 pipeline (one a core, shared
    by the core's harts; made with the core's first hart, a restarted
    hart takes its place)."""
    from .pipe8700 import HartPipeline, P8700Pipeline
    o = session.options
    n = _harts_a_core(session)
    cpu = m.cpu_num
    core, k = divmod(cpu, n)
    p = session._p8700_cores.get(core)
    if p is None:
        harts = session.machines[core * n:(core + 1) * n] or [m]
        p = P8700Pipeline(harts, latency=o.latency, memory=o.mem_timing,
                          nonblocking=o.nonblocking)
        cl = session.rvcps
        if cl is not None:
            p.runs = lambda j, base=core * n: cl.runs(base + j)
        p.debug_stop = lambda pc, j, base=core * n: session._stops_at(pc, base + j)
        session._keep_history(p)
        session._p8700_cores[core] = p
    elif p.harts[k].m is not m:
        p.attach(k, m)
    return HartPipeline(p, k)


def p8700_issuing(session) -> list[int]:
    """The cores that cycle: each core's first running hart."""
    n = _harts_a_core(session)
    first: dict[int, int] = {}
    for c in session.running():
        first.setdefault(c // n, c)
    return sorted(first.values())


def p8700_views(session, cpu: int, p, trace) -> None:
    """After a P8700 core's cycle: every hart of the core gets the
    trace, each the views of the instructions it executed."""
    n = _harts_a_core(session)
    base = cpu - cpu % n
    for j in range(len(p.core.harts)):
        if base + j < len(session.traces):
            session.traces[base + j] = trace
    for k, seq, view in p.core.views():
        c = base + k
        views = session._views[c]
        views[seq] = view
        session._last_views[c] = view
        while len(views) > session.KEPT_VIEWS:
            del views[next(iter(views))]
