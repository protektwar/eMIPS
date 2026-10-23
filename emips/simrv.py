"""The RISC-V machine (PLAN M25 phase 1c).

``Machine(arch="rv32" | "rv64")`` makes an :class:`RVMachine`: a hart of
XLEN 32 or 64 running the instructions of its extensions (``extensions``,
default ``"imafdc_zicsr_zifencei_zicntr_zba_zbb"``) as The RISC-V
Instruction Set Manual, Volume I: Unprivileged Architecture 20250508
defines them (sections cited are that manual's; "priv." = Volume II,
Privileged Architecture 20250508).

What it takes from :class:`emips.sim.Machine` is what is not architecture:
physical memory, the caches, devices, the cycle count.  Everything
architectural is its own: ``x0``-``x31`` (``regs``; x0 reads 0), ``pc``,
the CSRs (:mod:`emips.rvcsr`), the privilege mode (``priv``; phase 1: M
only), the instruction set (``rvisa``, :mod:`emips.riscv`), traps.
Addresses go through ``mmu`` — :class:`BareMMU` (satp.MODE = Bare,
priv. 12.1.11: physical = virtual) until phase 2.

- **Instructions** (:data:`EXECUTORS`, one per 32-bit row; a compressed
  instruction runs its expansion's, with its own size for the link and
  the next PC): RV32I / RV64I (2, 4), M (13), A (14: LR / SC with one
  reservation, the AMOs), Zicsr (6, the CSR access rules of priv. 2.1),
  Zifencei (6.1: the I-cache made coherent), Zicntr (8), Zba / Zbb (29),
  F / D (21, 22: :mod:`emips.rvfpu`).
- **Misaligned accesses** (``m25-RISC-V.md`` decision 7): ordinary loads
  and stores are done in hardware; LR / SC / AMOs raise address
  misaligned (4 / 6) — 14.2 / 14.4.
- **Traps** (phase 1 = M-mode only, priv. 3.1.6 / 3.1.14-3.1.17): mepc,
  mcause, mtval (the faulting address; an illegal instruction: its bits;
  an ``ebreak``: the PC), mstatus.MPIE / MIE / MPP, PC = mtvec.BASE.  An
  ``ecall`` asks ``syscall_handler`` first (RARS's services, decision 10:
  the service number in a7).  ``exceptions="raise"``: a trap raises
  :class:`RVException` and changes nothing.
- **Platform** (phase 1): a 32-bit physical address space — an access
  above 4 GB (RV64, bare) is an access fault (a PMA: nothing there);
  ``0xFFFF0000``-``0xFFFFFFFF`` (RARS's MMIO page) is uncached.  The
  timer behind ``time``: :attr:`RVMachine.mtime` counts every
  ``ccres`` cycles (decision 9; the ACLINT's in phase 2).
- **Debug** (phase 2e, The RISC-V Debug Specification 1.0): the trigger
  module (``tm``, :mod:`emips.rvdebug`; ``triggers=`` of them, 4 by
  default) is asked before each instruction (icount), for each fetch
  parcel / load / store, and after each trap (itrigger / etrigger); a
  trigger that fires raises a breakpoint exception or enters Debug Mode.
  Debug Mode (Sdext, ch. 4): :meth:`RVMachine.enter_debug` (ebreak with
  dcsr.ebreakm / s / u, a trigger with action 1, dcsr.step, a halt
  request), :meth:`RVMachine.resume` (DRET's work), and
  :meth:`RVMachine.execute_debug` — one instruction executed in Debug
  Mode as from a program buffer (4.1's rules).  The Debug Module that
  drives them through abstract commands: :mod:`emips.rvdm` (phase 2f).
"""

from __future__ import annotations

from typing import Any, Callable

from . import rvcsr as CSR
from . import sim as S
from .exceptions import SimException
from .mmu import Translation
from .riscv import RV, RVDecodeError, RVISA, size_of
from .rvdebug import (BREAKPOINT as ACTION_BREAKPOINT, CAUSE_EBREAK, CAUSE_HALTREQ,
                      CAUSE_STEP, CAUSE_TRIGGER, EBREAKM, EBREAKS, EBREAKU, MPTE, MTE, STEP,
                      Fired, Triggers)

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

#: exception codes (priv. Table 3.6)
INSN_MISALIGNED, INSN_ACCESS, ILLEGAL, BREAKPOINT = 0, 1, 2, 3
LOAD_MISALIGNED, LOAD_ACCESS, STORE_MISALIGNED, STORE_ACCESS = 4, 5, 6, 7
ECALL_U, ECALL_S, ECALL_M = 8, 9, 11
#: the interrupts (priv. Table 14)
INTERRUPT_NAMES = {1: "supervisor software interrupt", 3: "machine software interrupt",
                   5: "supervisor timer interrupt", 7: "machine timer interrupt",
                   9: "supervisor external interrupt", 11: "machine external interrupt"}
#: the extensions of the default hart (Zihpm: hpmcounter3-31, read 0)
DEFAULT_EXTENSIONS = "imafdc_zicsr_zifencei_zicntr_zihpm_zba_zbb"
CAUSE_NAMES = {
    0: "instruction address misaligned", 1: "instruction access fault",
    2: "illegal instruction", 3: "breakpoint", 4: "load address misaligned",
    5: "load access fault", 6: "store/AMO address misaligned", 7: "store/AMO access fault",
    8: "environment call from U-mode", 9: "environment call from S-mode",
    11: "environment call from M-mode", 12: "instruction page fault",
    13: "load page fault", 15: "store/AMO page fault",
}
#: the cacheability the bare platform gives memory / the MMIO page
CCA_CACHED, CCA_UNCACHED = 3, 2
MMIO_BASE = 0xFFFF0000
from .rvmmu import PA_BITS, RVMMU, pma_cca  # noqa: E402  (32-bit physical space, the PMAs)
from . import tracerv  # noqa: E402  (the views of the instructions)


class RVTrap(Exception):
    """A synchronous exception an instruction raises (``cause``, ``tval``)."""

    def __init__(self, cause: int, tval: int = 0) -> None:
        super().__init__(CAUSE_NAMES.get(cause, f"cause {cause}"))
        self.cause = cause
        self.tval = tval


class EnterDebug(Exception):
    """An instruction enters Debug Mode (an ``ebreak`` with dcsr.ebreakX)."""

    def __init__(self, cause: int) -> None:
        super().__init__(cause)
        self.cause = cause


class DebugModeTrap(Exception):
    """An exception of an instruction executed in Debug Mode: no trap is
    taken (Debug spec 4.1 rule 4); the Debug Module reports it."""

    def __init__(self, cause: int, tval: int = 0) -> None:
        super().__init__(cause, tval)
        self.cause = cause
        self.tval = tval


class RVException(SimException):
    """A trap in ``exceptions="raise"`` mode (a debugger stop): nothing
    changed.  ``exc_code`` = mcause."""

    def __init__(self, cause: int, tval: int, pc: int, name: str | None = None,
                 word: int | None = None) -> None:
        super().__init__(f"{CAUSE_NAMES.get(cause, cause)} at 0x{pc:x}", name=name, word=word,
                         addr=pc)
        self.exc_code = cause
        self.tval = tval


class BareMMU:
    """satp.MODE = Bare (priv. 12.1.11): the physical address is the
    virtual address; the platform's PMAs decide what is there (see the
    module docstring)."""

    mt = 0
    name = "bare"
    supervisor = False

    def attach(self, m) -> None:
        self.m = m

    @staticmethod
    def effective_priv(m, kind: str) -> int:
        """MPRV (an M + U hart has it: priv. 3.1.6.4) and the Debug
        Module's accesses, as with S-mode (emips.rvmmu)."""
        return RVMMU.effective_priv(m, kind)

    @staticmethod
    def _cca(m, pa: int) -> int:
        return pma_cca(m, pa)

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        if va >> PA_BITS:
            raise RVTrap(INSN_ACCESS if kind == "fetch" else
                         STORE_ACCESS if kind == "store" else LOAD_ACCESS, va)
        return va, pma_cca(m, va)

    def peek(self, m, va: int) -> tuple[int, int] | None:
        return None if va >> PA_BITS else (va, pma_cca(m, va))

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        if va >> PA_BITS:
            return Translation(va, None, CCA_UNCACHED, "bare", False,
                               f"bare: no physical memory above {PA_BITS} bits",
                               fault="access fault")
        cca = pma_cca(m, va)
        how = "bare: PA = VA" + (", I/O (uncached)" if cca == CCA_UNCACHED else "")
        return Translation(va, va, cca, "bare", False, how)


def _s32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v >> 31 else v


def _s64(v: int) -> int:
    v &= M64
    return v - (1 << 64) if v >> 63 else v


class RVMachine(S.Machine):
    """A RISC-V hart (see the module docstring)."""

    rv = True
    names = "riscv"
    #: RISC-V Debug Mode (an attribute: it hides sim.Machine's EJTAG property)
    debug_mode = False
    #: the operands the pipeline bypassed for the instruction it issues
    #: (emips.rvpipeline; the views name the path)
    rv_sources: dict | None = None

    def __init__(self, memory=None, pc: int = 0, *, arch: str = "rv64",
                 extensions: str = DEFAULT_EXTENSIONS, modes: str = "msu",
                 exceptions: str = "cp0", mmu: Any = None, caches=None, cpu: int = 0,
                 pmp: int = 0, triggers: int = 4, **_ignored: Any) -> None:
        if arch not in ("rv32", "rv64"):
            raise ValueError(f"arch must be 'rv32' or 'rv64', not {arch!r}")
        if modes not in ("m", "mu", "msu"):
            raise ValueError(f"modes must be 'm', 'mu' or 'msu', not {modes!r}")
        if mmu is None:                         # S-mode: page-based translation (2b)
            from .rvmmu import RVMMU
            mmu = RVMMU() if "s" in modes else BareMMU()
        super().__init__(memory=memory, pc=0, endian="little", exceptions=exceptions,
                         release="R6", mmu=mmu, caches=caches, cpu=cpu, fpu=False)
        self.arch = arch
        #: XLEN (priv. 3.1.1: MXLEN = XLEN here)
        self.xlen = 32 if arch == "rv32" else 64
        self.amask = M32 if self.xlen == 32 else M64
        self.vamask = self.amask
        self.wide = self.xlen == 64
        self.sgn = _s32 if self.xlen == 32 else _s64
        #: the instructions (emips.riscv; S-mode brings SRET / SFENCE.VMA,
        #: Sdext DRET — Debug spec 4.1: illegal outside Debug Mode)
        self.rvisa = RVISA.for_core(self.xlen, extensions + ("_s" if "s" in modes else "")
                                    + "_sdext")
        self.extensions = self.rvisa.extensions
        #: IALIGN (1.5): 16 with C, else 32
        self.ialign = 2 if "c" in self.extensions else 4
        #: the privilege modes the hart has (priv. 1.2: M, M + U, M + S + U)
        #: and the current one: 3 = M, 1 = S, 0 = U (reset: M, priv. 3.4)
        self.modes = frozenset({"m": 3, "s": 1, "u": 0}[k] for k in modes)
        self.priv = 3
        #: address translation (satp, priv. 12.1.11; emips.rvmmu): the MODEs
        #: satp accepts (0 = Bare; Sv32 = 1; Sv39 / Sv48 / Sv57 = 8-10) and
        #: whether a page-based mode exists
        self.paging = "s" in modes
        self.satp_modes = frozenset({0} | ({1} if self.xlen == 32 else {8, 9, 10})
                                    if self.paging else {0})
        #: WFI is waiting for an interrupt (priv. 3.3.3)
        self.waiting = False
        #: the platform's timer (emips.rvplatform.CLINT) and I/O regions (PMAs)
        self.timer = None
        self.io_regions: list[tuple[int, int]] = []
        #: the hart's number (mhartid)
        self.hartid = cpu
        #: the counters a CSR write set during this instruction (priv.
        #: 3.1.10: the write takes effect after the instruction completes)
        self.counter_written: set[str] = set()
        #: the LR reservation: (address, size) or None (14.2)
        self.reservation: tuple[int, int] | None = None
        #: FLEN (22.2): 64 with D, 32 with F only, 0 without F
        self.flen = 64 if "d" in self.extensions else 32 if "f" in self.extensions else 0
        #: fcsr = frm 7:5 | fflags 4:0 (21.2); the f registers (FLEN bits;
        #: emips.rvfpu NaN-boxes single values when FLEN = 64)
        self.fcsr = 0
        self.fpr = [0] * 32
        #: Physical Memory Protection (priv. 3.7; emips.rvpmp): ``pmp`` entries
        from .rvpmp import PMP
        self.pmp = PMP(self.xlen, pmp)
        #: Debug Mode (Debug spec ch. 4): in it, a halt request pending, the
        #: hart resumed with dcsr.step (one instruction, then Debug Mode)
        self.debug_mode = False
        self.halt_request = False
        self._stepping = False
        #: the Debug Module the hart is attached to (emips.rvdm; it hears
        #: of every halt: halt groups) and an access it makes for an
        #: abstract command ("physical" / "virtual": aamvirtual)
        self.debug_module = None
        self.dm_access: str | None = None
        #: the trigger module (Debug spec ch. 5; emips.rvdebug)
        self.tm = Triggers(self, triggers)
        #: the CSRs (emips.rvcsr)
        self.csrs = CSR.CSRFile(self)
        self.pc = pc & self.amask
        #: where a reset starts the hart (reset_hart: the platform's vector)
        self.reset_pc = self.pc
        #: the last instruction executed (an RV) — the views
        self.last_insn: RV | None = None
        #: the F / D registers exist (the views; sim.Machine's MIPS FPU flag)
        self.has_fpu = self.flen > 0

    # -- registers -------------------------------------------------------------

    def read_reg(self, index: int) -> int:
        index &= 31
        return 0 if index == 0 else self.regs[index] & self.amask

    def write_reg(self, index: int, value: int) -> None:
        index &= 31
        if index:
            self.regs[index] = value & self.amask

    write_dreg = write_reg

    def to_pc(self, addr: int) -> int:
        return addr & getattr(self, "amask", M64)

    @property
    def mtime(self) -> int:
        """The platform timer: the ACLINT MTIMER's mtime (emips.rvplatform),
        or without a platform cycles / CCRes (decision 9)."""
        if self.timer is not None:
            return self.timer.mtime()
        return self.cycles // self.ccres

    # -- the Machine's MIPS paths a RISC-V hart never takes --------------------

    def interrupt_request(self, ixmt=None) -> tuple[int, int] | None:
        """The interrupt the hart takes now, as (cause, the mode it traps
        to), or None (priv. 3.1.9 / 10.1.3: pending in mip and enabled in
        mie; to M unless delegated, when M-mode is below M or MIE; to S
        when delegated and the hart is in U, or in S with SIE; the
        priority MEI, MSI, MTI, SEI, SSI, STI)."""
        c = self.csrs
        pend = c.mip() & c.mie
        if not pend:
            return None
        m_ok = self.priv < 3 or c.status_bit(CSR.MIE)
        s_ok = self.priv < 1 or (self.priv == 1 and c.status_bit(CSR.SIE))
        for bit in CSR.PRIORITY:
            if pend >> bit & 1:
                if c.mideleg >> bit & 1:
                    if s_ok:
                        return bit, 1
                elif m_ok:
                    return bit, 3
        return None

    def set_interrupt(self, bit: int, level: bool) -> None:
        """A platform interrupt line (MSIP / MTIP / MEIP, the external
        SEIP): mip bit ``bit`` follows it."""
        if level:
            self.irq_lines |= 1 << bit
        else:
            self.irq_lines &= ~(1 << bit)

    def pending_cache_error(self):
        return None

    def advance(self, cycles: int = 1) -> None:
        self.cycles += cycles
        if "mcycle" not in self.counter_written:
            self.csrs.tick(cycles)
        for dev in self.devices:
            dev.tick(self, cycles)
        if self.caches is not None and self.caches.fsb:
            self.caches.tick()

    def address_ok(self, addr: int) -> bool:
        return True

    def kernel_mode(self) -> bool:
        return self.priv == 3

    # -- execution ---------------------------------------------------------------

    def _half(self, addr: int) -> int:
        """A fetched halfword; a fault names this halfword's address (priv.
        3.1.16: the part of the instruction that faulted).  PMP checks each
        16-bit parcel (as Spike fetches)."""
        try:
            if self.pmp.entries:
                pa, _ = self._xlate(self, addr & self.amask, "fetch")
                if not self.pmp.allows(pa, 2, "fetch", self.priv):
                    raise RVTrap(INSN_ACCESS, addr)
            return self.fetch_half(addr)
        except RVTrap as t:
            raise RVTrap(t.cause, addr) from None

    def _access(self, addr: int, size: int, kind: str) -> tuple[list[int], int]:
        """Translate an access (sim.Machine) and check PMP on its physical
        bytes (priv. 3.7.1; fetches: per parcel in :meth:`_half`)."""
        pas, cca = super()._access(addr, size, kind)
        p = self.pmp
        if p.entries and kind != "fetch":
            k = "store" if kind == "store" else "load"
            priv = self.mmu.effective_priv(self, k) if hasattr(self.mmu, "effective_priv") \
                else self.priv
            start = 0
            for j in range(1, size + 1):           # (one check per physically contiguous run)
                if j == size or pas[j] != pas[j - 1] + 1:
                    if not p.allows(pas[start], j - start, k, priv):
                        raise RVTrap(STORE_ACCESS if k == "store" else LOAD_ACCESS, addr)
                    start = j
        return pas, cca

    def load(self, addr: int, size: int, kind: str = "load", pre: bool = True) -> int:
        """sim.Machine's load, with the load triggers: the address before
        the access (``pre``; LR / AMOs check it themselves, ahead of their
        alignment), the data after it (Debug spec 5.5, Table 13)."""
        tm = self.tm
        if not tm.loads:
            return super().load(addr, size, kind)
        if pre:
            tm.memory("load", addr, size)
        v = super().load(addr, size, kind)
        tm.memory("load", addr, size, v)
        return v

    def store(self, addr: int, value: int, size: int, *, gather: bool = True,
              atomic: bool = False) -> bool | None:
        """sim.Machine's store, after the store triggers (address and data)."""
        if self.tm.stores:
            self.tm.memory("store", addr, size, value & ((1 << 8 * size) - 1))
        return super().store(addr, value, size, gather=gather, atomic=atomic)

    def trigger_check(self, op: str, addr: int, size: int) -> None:
        """The address triggers of an access (``op`` load / store) that
        are checked ahead of its alignment (LR, AMOs)."""
        if self.tm.loads if op == "load" else self.tm.stores:
            self.tm.memory(op, addr, size)

    def load_pte(self, pa: int, size: int) -> int:
        """A page-table entry at physical address ``pa`` (emips.rvmmu; through
        the caches, as an implicit load); None when PMP refuses it (an
        S-mode read, priv. 3.7.2)."""
        if not self.pmp.allows(pa, size, "load", 1):
            return None
        return self.load_pa(pa, [pa + k for k in range(size)], pma_cca(self, pa), size)

    def fetch(self, pc: int, triggers: bool = True) -> RV:
        """Fetch and decode the instruction at ``pc`` (its first halfword
        tells the size: 1.5); an illegal one raises :class:`RVTrap`.
        Execute triggers look at each parcel's address before it is
        fetched, then at the whole instruction (Debug spec Table 13) —
        ``triggers=False``: not (the pipeline's fetch ahead; the execute
        triggers look when the instruction issues)."""
        tm = self.tm if triggers else _NO_TRIGGERS
        if tm.fetch:
            tm.memory("execute", pc, 2)
        if pc & (self.ialign - 1):
            raise RVTrap(INSN_MISALIGNED, pc)
        h = self._half(pc)
        size = size_of(h)
        if size == 4:
            if tm.fetch:
                tm.memory("execute", (pc + 2) & self.amask, 2)
            raw = h | self._half((pc + 2) & self.amask) << 16
        else:
            raw = h
        if tm.fetch and size:
            tm.memory("execute", pc, size, raw)
        try:
            return self.rvisa.decode(raw, size or 4)
        except RVDecodeError:
            raise RVTrap(ILLEGAL, raw if size else h) from None

    def step(self) -> RV | None:
        """Execute one instruction (one clock cycle in this model);
        returns it (``None`` when the fetch or decode trapped)."""
        if self.halted:
            return None
        if self.debug_mode:                     # halted (Debug spec 4.1): the Debug
            self.advance(1)                     # Module drives the hart (phase 2f)
            return None
        if self.halt_request:                   # haltreq (4.7)
            self.halt_request = False
            self.enter_debug(CAUSE_HALTREQ)
            return None
        pc = self.pc
        # dcsr.stepie = 0: no interrupt while stepping (4.9.1)
        req = None if self._stepping else self.interrupt_request()
        if req is not None:
            self.advance(1)
            self.take_interrupt(req, pc)
            return None
        if self.waiting:                        # WFI: until mip & mie (3.3.3)
            if self.csrs.mip() & self.csrs.mie:
                self.waiting = False
            else:
                self.advance(1)
                return None
        return self.issue(pc)

    def take_interrupt(self, req: tuple[int, int], pc: int) -> None:
        """Take the interrupt ``req`` (:meth:`interrupt_request`) before
        the instruction at ``pc`` (xepc = pc)."""
        tm = self.tm
        tm.after = None
        self.waiting = False
        if tm.icounts:                          # (a trap from an enabled mode counts)
            tm.icount_trap(self.priv)
        prev = self.priv
        self._enter(req[0], 0, pc, req[1], interrupt=True)
        self._trap_triggers(req[0], True, prev)

    def issue(self, pc: int, i: RV | None = None, fault: Exception | None = None, *,
              clock: bool = True) -> RV | None:
        """Execute the instruction at ``pc``: fetch it (or take ``i``, fetched
        earlier — the pipeline —, or raise ``fault``, what its fetch found),
        take its trap or the trigger it fires, retire it; ``self.pc`` is then
        the next instruction (the next one, a jump's target, a trap vector).
        ``clock``: one clock cycle passes (instruction mode; the pipeline
        clocks the hart itself and clears :attr:`counter_written` after)."""
        tm = self.tm
        tm.after = None
        tick = self.advance if clock else (lambda n: None)
        v = tracerv.begin(self, pc) if self.tracing else None   # (the view: emips.tracerv)
        try:
            if tm.icounts:
                f = tm.icount()
                if f is not None:
                    raise f
            if fault is not None:
                raise fault
            if i is None:
                i = self.fetch(pc)
            if v is not None:
                tracerv.fetched(self, v, i)
            nxt = EXECUTORS[i.base.name](self, i, pc)
        except Fired as f:
            tick(1)
            self._fire(f, pc)
            self._stepped()
            if v is not None:
                self.last_view = tracerv.finish(self, v, i, None, "a trigger fired")
            return i
        except EnterDebug as e:
            tick(1)
            self.enter_debug(e.cause, pc)
            if v is not None:
                self.last_view = tracerv.finish(self, v, i, None, "Debug Mode")
            return i
        except RVTrap as t:
            tick(1)
            self._trap(t, pc, i)
            self._stepped()
            if v is not None:
                self.last_view = tracerv.finish(self, v, i, None,
                                                CAUSE_NAMES.get(t.cause, f"cause {t.cause}"))
            return i
        except SimException as exc:             # (a simulator halt)
            self._mem_log = None
            self.exception_log.append({"name": getattr(exc, "name", None),
                                       "word": getattr(exc, "word", None),
                                       "addr": getattr(exc, "addr", pc),
                                       "error": type(exc).__name__})
            raise
        if "minstret" not in self.counter_written:
            self.csrs.retire()
        self.last_insn = i
        if clock:
            self.advance(1)
            self.counter_written.clear()
        self.pc = (pc + i.size if nxt is None else nxt) & self.amask
        if self.call_hook is not None:          # (the debugger's call stack)
            self.call_hook(pc, i, i.raw, self.pc)
        if v is not None:
            self.last_view = tracerv.finish(self, v, i, self.pc)
        if tm.after is not None:                # a timing-after trigger (5.7.12 hit = 3)
            f, tm.after = tm.after, None
            self._fire(f, self.pc)
        self._stepped()
        return i

    def _trap(self, t: RVTrap, pc: int, i: RV | None, triggers: bool = True) -> None:
        """Take a synchronous exception (priv. 3.1.8: to S-mode when
        medeleg delegates it and the hart is in S or U); then the
        etriggers (``triggers``: not for a trigger's own breakpoint)."""
        if self.exceptions == "raise":
            raise RVException(t.cause, t.tval, pc, name=i.name if i else None,
                              word=i.raw if i else None)
        prev = self.priv
        to = 1 if self.priv <= 1 and self.csrs.medeleg >> t.cause & 1 else 3
        self.exception_log.append({"name": CAUSE_NAMES.get(t.cause), "cause": t.cause,
                                   "tval": t.tval, "epc": pc, "word": i.raw if i else None,
                                   "priv": self.priv, "to": to})
        self._enter(t.cause, t.tval, pc, to, interrupt=False)
        if triggers:
            self._trap_triggers(t.cause, False, prev)

    def _trap_triggers(self, cause: int, interrupt: bool, prev: int) -> None:
        """Debug spec 5.7.14 / 5.7.15: an itrigger / etrigger fires after
        the trap, before the handler's first instruction (tval 0)."""
        if self.tm.traps:
            f = self.tm.trap(cause, interrupt, prev)
            if f is not None:
                self._fire(f, self.pc)

    def _fire(self, f: Fired, epc: int) -> None:
        """A trigger's action (Table 12): a breakpoint exception (epc:
        the instruction that matched, or the next one for timing after)
        or Debug Mode (cause 2, dpc = epc)."""
        if f.action == ACTION_BREAKPOINT:
            self._trap(RVTrap(BREAKPOINT, f.tval), epc, None, triggers=False)
        else:
            self.enter_debug(CAUSE_TRIGGER, epc)

    def _stepped(self) -> None:
        """dcsr.step (4.5.1): after the instruction (or the trap it took,
        the PC at the handler) the hart re-enters Debug Mode, cause 4."""
        if self._stepping:
            self._stepping = False
            if not self.debug_mode:
                self.enter_debug(CAUSE_STEP)

    def _enter(self, cause: int, tval: int, pc: int, to: int, *, interrupt: bool) -> None:
        """Trap entry (priv. 3.1.6.1, 3.1.7, 3.1.14-3.1.17; 10.1.1-10.1.9
        for S-mode): xEPC, xCAUSE (the interrupt bit XLEN-1), xTVAL, the
        xPIE / xIE / xPP stack, the PC from xtvec (vectored: BASE +
        4 * cause for an interrupt); into M-mode tcontrol.MPTE = MTE, MTE
        = 0 (Debug spec 5.7.6)."""
        c = self.csrs
        code = cause | ((1 << (self.xlen - 1)) if interrupt else 0)
        if interrupt:
            self.exception_log.append({"name": INTERRUPT_NAMES.get(cause), "cause": code,
                                       "epc": pc, "priv": self.priv, "to": to})
        if to == 1:
            c.sepc = pc & ~1
            c.scause = code
            c.stval = tval & self.amask
            c.set_status(spie=c.status_bit(CSR.SIE), sie=0, spp=self.priv)
            tvec = c.stvec
        else:
            c.mepc = pc & ~1
            c.mcause = code
            c.mtval = tval & self.amask
            c.set_status(mpie=c.status_bit(CSR.MIE), mie=0, mpp=self.priv)
            tvec = c.mtvec
            tm = self.tm
            if tm.count and not c.has_s:
                tm.tcontrol = MPTE if tm.tcontrol & MTE else 0
        self.priv = to                          # (the LR reservation survives a trap:
        base = tvec & ~3                        # priv. 3.3.2 lets only xRET drop it)
        self.pc = (base + 4 * cause if interrupt and tvec & 3 == 1 else base) & self.amask
        if self.exception_log:                  # (the entry _trap / the interrupt made:
            e = self.exception_log[-1]          # the debugger's exception line)
            e.setdefault("error", e.get("name") or f"cause {cause}")
            e.update(addr=pc, vector=self.pc,
                     why=f"{'s' if to == 1 else 'm'}cause 0x{code & self.amask:x}, "
                         f"{'s' if to == 1 else 'm'}tval 0x{tval & self.amask:x}")

    # -- Debug Mode (Debug spec ch. 4) ---------------------------------------------------

    def enter_debug(self, cause: int, dpc: int | None = None) -> None:
        """4.7: dcsr.cause and prv, dpc = the next instruction (``dpc``, or
        the PC), M-mode privilege; WFI ends."""
        c = self.csrs
        dpc = self.pc if dpc is None else dpc
        self.exception_log.append({"name": "Debug Mode", "debug": cause, "epc": dpc,
                                   "priv": self.priv})
        c.dcsr = (c.dcsr & ~(7 << 6 | 3)) | cause << 6 | self.priv
        c.dpc = dpc & self.amask & ~1
        self.pc = dpc & self.amask
        self.priv = 3
        self.debug_mode = True
        self.waiting = False
        self._stepping = False
        if self.debug_module is not None:
            self.debug_module.hart_halted(self)

    def reset_hart(self) -> None:
        """A hart reset (the Debug Module's hartreset / ndmreset; priv.
        3.4): M-mode at ``reset_pc``, the CSRs, PMP, triggers and the
        TLB at their reset values, no reservation, out of Debug Mode and
        WFI.  The x / f registers keep their (UNSPECIFIED) values."""
        from .rvpmp import PMP
        self.priv = 3
        self.pc = self.reset_pc & self.amask
        self.pmp = PMP(self.xlen, self.pmp.entries)
        self.tm = Triggers(self, self.tm.count)
        self.csrs = CSR.CSRFile(self)
        self.debug_mode = self.waiting = self._stepping = self.halt_request = False
        self.reservation = None
        self.dm_access = None
        flush = getattr(self.mmu, "sfence", None)
        if flush is not None:
            flush(self, None, None)

    def resume(self) -> None:
        """4.8 (DRET; the Debug Module's resume): PC = dpc, the mode =
        dcsr.prv (below M: MPRV = 0), out of Debug Mode; dcsr.step: one
        instruction, then Debug Mode again."""
        c = self.csrs
        self.pc = c.read(0x7B1)
        self.priv = c.dcsr & 3
        if self.priv < 3:
            c.set_status(mprv=0)
        self.debug_mode = False
        self._stepping = bool(c.dcsr & STEP)

    def execute_debug(self, raw: int) -> str | None:
        """Execute one instruction in Debug Mode, as from a program buffer
        (4.1): M-mode rights (MPRV only with dcsr.mprven), no interrupts,
        no triggers; an exception takes no trap — :class:`DebugModeTrap`
        (rule 4).  ``ebreak`` ends the program ("ebreak"), DRET resumes
        ("dret"), WFI is a nop (rule 8); control transfers, AUIPC (rules
        10-12) and the instructions that change the mode (rule 9: ECALL,
        MRET, SRET) are illegal here."""
        if not self.debug_mode:
            raise ValueError("the hart is not in Debug Mode")
        size = 4 if raw & 3 == 3 else 2
        try:
            i = self.rvisa.decode(raw, size)
        except RVDecodeError:
            raise DebugModeTrap(ILLEGAL, raw) from None
        name = i.base.name
        if name == "ebreak":
            return "ebreak"
        if name == "dret":
            self.resume()
            return "dret"
        if name in _DEBUG_ILLEGAL:
            raise DebugModeTrap(ILLEGAL, raw)
        if name != "wfi":
            try:
                EXECUTORS[name](self, i, self.pc)
            except RVTrap as t:
                raise DebugModeTrap(t.cause, t.tval) from None
        if "minstret" not in self.counter_written:
            self.csrs.retire()
        self.advance(1)
        self.counter_written.clear()
        return None

    def target(self, pc: int, offset: int) -> int:
        """A branch / jump target, checked for IALIGN (2.5: an
        instruction-address-misaligned exception on the branch)."""
        t = (pc + offset) & self.amask
        if t & (self.ialign - 1):
            raise RVTrap(INSN_MISALIGNED, t)
        return t


class _NoTriggers:
    fetch = False


_NO_TRIGGERS = _NoTriggers()


# ------------------------------------------------------------------ executors

Executor = Callable[[RVMachine, RV, int], "int | None"]
EXECUTORS: dict[str, Executor] = {}


def _x(name: str):
    def deco(fn: Executor) -> Executor:
        EXECUTORS[name] = fn
        return fn
    return deco


def _reg(m: RVMachine, n: int) -> int:
    return 0 if n == 0 else m.regs[n] & m.amask


def _set(m: RVMachine, n: int, v: int) -> None:
    if n:
        m.regs[n] = v & m.amask


def _w(v: int) -> int:
    """A W instruction's 32-bit result, sign-extended (4.2)."""
    return _s32(v)


# -- RV32I / RV64I (2.4-2.8, 4.2-4.4) ----------------------------------------

@_x("lui")
def _lui(m, i, pc):
    _set(m, i.f["rd"], i.f["imm"])


@_x("auipc")
def _auipc(m, i, pc):
    _set(m, i.f["rd"], pc + i.f["imm"])


@_x("jal")
def _jal(m, i, pc):
    t = m.target(pc, i.f["imm"])
    _set(m, i.f["rd"], pc + i.size)
    return t


@_x("jalr")
def _jalr(m, i, pc):
    t = (_reg(m, i.f["rs1"]) + i.f["imm"]) & m.amask & ~1
    if t & (m.ialign - 1):
        raise RVTrap(INSN_MISALIGNED, t)
    _set(m, i.f["rd"], pc + i.size)
    return t


def _branch(name: str, cond: Callable[[RVMachine, int, int], bool]) -> None:
    def ex(m, i, pc):
        if cond(m, _reg(m, i.f["rs1"]), _reg(m, i.f["rs2"])):
            return m.target(pc, i.f["imm"])
        return None
    EXECUTORS[name] = ex


_branch("beq", lambda m, a, b: a == b)
_branch("bne", lambda m, a, b: a != b)
_branch("blt", lambda m, a, b: m.sgn(a) < m.sgn(b))
_branch("bge", lambda m, a, b: m.sgn(a) >= m.sgn(b))
_branch("bltu", lambda m, a, b: a < b)
_branch("bgeu", lambda m, a, b: a >= b)


def _addr(m: RVMachine, i: RV) -> int:
    return (_reg(m, i.f["rs1"]) + i.f["imm"]) & m.amask


def _load(name: str, size: int, signed: bool) -> None:
    def ex(m, i, pc):
        v = m.load(_addr(m, i), size)
        if signed and v >> (8 * size - 1):
            v -= 1 << (8 * size)
        _set(m, i.f["rd"], v)
    EXECUTORS[name] = ex


for _n, _sz, _sg in (("lb", 1, True), ("lh", 2, True), ("lw", 4, True), ("ld", 8, True),
                     ("lbu", 1, False), ("lhu", 2, False), ("lwu", 4, False)):
    _load(_n, _sz, _sg)


def _store(name: str, size: int) -> None:
    def ex(m, i, pc):
        addr = _addr(m, i)
        m.store(addr, _reg(m, i.f["rs2"]), size)
        _clear_reservation(m, addr, size)
    EXECUTORS[name] = ex


for _n, _sz in (("sb", 1), ("sh", 2), ("sw", 4), ("sd", 8)):
    _store(_n, _sz)


def _alu(name: str, op: Callable[[RVMachine, int, int], int], imm: bool = False,
         word: bool = False) -> None:
    def ex(m, i, pc):
        a = _reg(m, i.f["rs1"])
        b = (i.f["shamt"] if "shamt" in i.f else i.f["imm"]) if imm else _reg(m, i.f["rs2"])
        v = op(m, a, b & m.amask)
        _set(m, i.f["rd"], _w(v) if word else v)
    EXECUTORS[name] = ex


def _shamt(m: RVMachine, b: int) -> int:
    return b & (m.xlen - 1)


for _n, _op in (
        ("add", lambda m, a, b: a + b), ("sub", lambda m, a, b: a - b),
        ("sll", lambda m, a, b: a << _shamt(m, b)),
        ("slt", lambda m, a, b: int(m.sgn(a) < m.sgn(b))),
        ("sltu", lambda m, a, b: int(a < b)),
        ("xor", lambda m, a, b: a ^ b),
        ("srl", lambda m, a, b: a >> _shamt(m, b)),
        ("sra", lambda m, a, b: m.sgn(a) >> _shamt(m, b)),
        ("or", lambda m, a, b: a | b), ("and", lambda m, a, b: a & b)):
    _alu(_n, _op)
    if _n not in ("sub",):
        _alu(f"{_n}i" if _n not in ("sll", "srl", "sra", "sltu") else
             {"sll": "slli", "srl": "srli", "sra": "srai", "sltu": "sltiu"}[_n], _op, imm=True)
for _n, _op in (
        ("addw", lambda m, a, b: a + b), ("subw", lambda m, a, b: a - b),
        ("sllw", lambda m, a, b: (a & M32) << (b & 31)),
        ("srlw", lambda m, a, b: (a & M32) >> (b & 31)),
        ("sraw", lambda m, a, b: _s32(a) >> (b & 31))):
    _alu(_n, _op, word=True)
for _n, _base in (("addiw", "addw"), ("slliw", "sllw"), ("srliw", "srlw"), ("sraiw", "sraw")):
    _alu(_n, {"addw": lambda m, a, b: a + b, "sllw": lambda m, a, b: (a & M32) << (b & 31),
              "srlw": lambda m, a, b: (a & M32) >> (b & 31),
              "sraw": lambda m, a, b: _s32(a) >> (b & 31)}[_base], imm=True, word=True)


@_x("fence")
def _fence(m, i, pc):
    """2.7: orders memory; the write buffer drains (a SYNC's job)."""
    if m.caches is not None:
        m.caches.sync(0)


@_x("fence.i")
def _fence_i(m, i, pc):
    """6.1: later fetches see earlier stores — the D-cache's dirty lines
    written back, the I-cache invalidated."""
    if m.caches is not None:
        m.caches.fence_i()


@_x("ecall")
def _ecall(m, i, pc):
    h = m.syscall_handler
    if h is not None and h(m, _reg(m, 17)):
        return None
    raise RVTrap({0: ECALL_U, 1: ECALL_S, 3: ECALL_M}[m.priv], 0)


@_x("ebreak")
def _ebreak(m, i, pc):
    """A breakpoint exception, or Debug Mode when dcsr.ebreakm / ebreaks /
    ebreaku has it for this mode (Debug spec 4.9.1)."""
    if m.csrs.dcsr & {3: EBREAKM, 1: EBREAKS, 0: EBREAKU}[m.priv]:
        raise EnterDebug(CAUSE_EBREAK)
    raise RVTrap(BREAKPOINT, pc)


@_x("mret")
def _mret(m, i, pc):
    """priv. 3.3.2: MIE = MPIE, MPIE = 1, priv = MPP, MPP = the least
    privileged mode; MPRV = 0 when MPP is not M."""
    if m.priv != 3:
        raise RVTrap(ILLEGAL, i.raw)
    c = m.csrs
    mpp = c.status_field(CSR.MPP_SHIFT, 3)
    c.set_status(mie=c.status_bit(CSR.MPIE), mpie=1, mpp=c.lowest_priv)
    if mpp != 3:
        c.set_status(mprv=0)
    m.priv = mpp
    tm = m.tm
    if tm.count and not c.has_s:                # Debug spec 5.7.6: MTE = MPTE
        tm.tcontrol = MPTE | MTE if tm.tcontrol & MPTE else 0
    return c.read(0x341)


@_x("sret")
def _sret(m, i, pc):
    """priv. 3.3.2 / 10.1.1: S-mode or M-mode only, not in S-mode with
    TSR; SIE = SPIE, SPIE = 1, priv = SPP, SPP = U, MPRV = 0."""
    c = m.csrs
    if not c.has_s or m.priv < 1 or (m.priv == 1 and c.status_bit(CSR.TSR)):
        raise RVTrap(ILLEGAL, i.raw)
    spp = c.status_bit(CSR.SPP)
    c.set_status(sie=c.status_bit(CSR.SPIE), spie=1, spp=0, mprv=0)
    m.priv = spp
    return c.read(0x141)


@_x("wfi")
def _wfi(m, i, pc):
    """priv. 3.3.3: wait until an interrupt is pending and enabled in mie
    (whatever the global enables).  Below M-mode with TW = 1, or in U-mode
    when S-mode exists, an illegal instruction (eMIPS's time limit is
    0, as Spike's)."""
    c = m.csrs
    if (m.priv < 3 and c.status_bit(CSR.TW)) or (m.priv == 0 and c.has_s):
        raise RVTrap(ILLEGAL, i.raw)
    if not c.mip() & c.mie and not m._stepping:     # (stepped: a nop, Debug spec 4.5.1)
        m.waiting = True


@_x("sfence.vma")
def _sfence_vma(m, i, pc):
    """priv. 10.2.1: S-mode or M-mode; in S-mode with TVM, illegal (the
    translation caches it flushes: phase 2b)."""
    c = m.csrs
    if not c.has_s or m.priv < 1 or (m.priv == 1 and c.status_bit(CSR.TVM)):
        raise RVTrap(ILLEGAL, i.raw)
    flush = getattr(m.mmu, "sfence", None)
    if flush is not None:
        flush(m, _reg(m, i.f["rs1"]) if i.f["rs1"] else None,
              _reg(m, i.f["rs2"]) if i.f["rs2"] else None)


EXECUTORS["dret"] = lambda m, i, pc: (_ for _ in ()).throw(RVTrap(ILLEGAL, i.raw))
#: (DRET outside Debug Mode: illegal — Debug spec 4.1; in it: execute_debug)
#: what Debug Mode refuses (Debug spec 4.1 rules 9-12: eMIPS's choice)
_DEBUG_ILLEGAL = frozenset({"jal", "jalr", "beq", "bne", "blt", "bge", "bltu", "bgeu",
                            "auipc", "ecall", "mret", "sret"})


# -- Zicsr (6.1; priv. 2.1) ---------------------------------------------------

def _csr(name: str, imm: bool, op: str) -> None:
    def ex(m, i, pc):
        f = i.f
        num, rd = f["csr"], f["rd"]
        src = f["zimm"] if imm else _reg(m, f["rs1"])
        writes = op == "w" or (f["zimm"] if imm else f["rs1"]) != 0
        reads = op != "w" or rd != 0
        c = m.csrs
        c.check(num, write=writes, insn=i.raw)
        old = c.read(num) if reads else 0       # (CSRRW with rd = x0: no read)
        if writes:
            base = c.read_for_write(num) if op != "w" else 0
            new = src if op == "w" else (base | src) if op == "s" else (base & ~src)
            c.write(num, new & m.amask)
        _set(m, rd, old)
    EXECUTORS[name] = ex


for _n, _imm, _op in (("csrrw", False, "w"), ("csrrs", False, "s"), ("csrrc", False, "c"),
                      ("csrrwi", True, "w"), ("csrrsi", True, "s"), ("csrrci", True, "c")):
    _csr(_n, _imm, _op)


# -- M (13) --------------------------------------------------------------------

def _div(a: int, b: int, bits: int, signed: bool, rem: bool) -> int:
    """13.2: division by zero and overflow (no trap; Table 13.1)."""
    mask = (1 << bits) - 1
    a &= mask
    b &= mask
    if signed:
        a = a - (1 << bits) if a >> (bits - 1) else a
        b = b - (1 << bits) if b >> (bits - 1) else b
    if b == 0:
        return a if rem else mask
    if signed and a == -(1 << (bits - 1)) and b == -1:
        return 0 if rem else a
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return a - q * b if rem else q


for _n, _fn in (
        ("mul", lambda m, a, b: a * b),
        ("mulh", lambda m, a, b: (m.sgn(a) * m.sgn(b)) >> m.xlen),
        ("mulhsu", lambda m, a, b: (m.sgn(a) * b) >> m.xlen),
        ("mulhu", lambda m, a, b: (a * b) >> m.xlen),
        ("div", lambda m, a, b: _div(a, b, m.xlen, True, False)),
        ("divu", lambda m, a, b: _div(a, b, m.xlen, False, False)),
        ("rem", lambda m, a, b: _div(a, b, m.xlen, True, True)),
        ("remu", lambda m, a, b: _div(a, b, m.xlen, False, True))):
    _alu(_n, _fn)
for _n, _fn in (
        ("mulw", lambda m, a, b: a * b),
        ("divw", lambda m, a, b: _div(a, b, 32, True, False)),
        ("divuw", lambda m, a, b: _div(a, b, 32, False, False)),
        ("remw", lambda m, a, b: _div(a, b, 32, True, True)),
        ("remuw", lambda m, a, b: _div(a, b, 32, False, True))):
    _alu(_n, _fn, word=True)


# -- A (14) ----------------------------------------------------------------------

def _clear_reservation(m: RVMachine, addr: int, size: int) -> None:
    """A store of this hart to the reserved bytes: the reservation stays
    (14.2 allows it); other harts' stores clear it (the CPS, phase 5)."""


def _aligned(m: RVMachine, addr: int, size: int, store: bool) -> None:
    if addr % size:
        raise RVTrap(STORE_MISALIGNED if store else LOAD_MISALIGNED, addr)


def _lr(size: int):
    def ex(m, i, pc):
        addr = _reg(m, i.f["rs1"])
        m.trigger_check("load", addr, size)     # (Debug spec Table 13: before alignment)
        _aligned(m, addr, size, False)
        v = m.load(addr, size, pre=False)
        m.reservation = (addr, size)
        _set(m, i.f["rd"], _s32(v) if size == 4 else v)
    return ex


def _sc(size: int):
    def ex(m, i, pc):
        addr = _reg(m, i.f["rs1"])
        _aligned(m, addr, size, True)
        ok = m.reservation == (addr, size)
        m.reservation = None                    # 14.2: SC invalidates it either way
        if ok:
            m.store(addr, _reg(m, i.f["rs2"]), size, atomic=True)
        else:                                   # 13.2: "no SC.W shall retire unless it
            m._access(addr, size, "store")      # passes memory permission checks"
        _set(m, i.f["rd"], 0 if ok else 1)
    return ex


EXECUTORS.update({"lr.w": _lr(4), "lr.d": _lr(8), "sc.w": _sc(4), "sc.d": _sc(8)})


def _amo(name: str, size: int, op: Callable[[int, int, int], int]) -> None:
    bits = 8 * size

    def ex(m, i, pc):
        addr = _reg(m, i.f["rs1"])
        m.trigger_check("store", addr, size)    # (Spike: the store's address first)
        _aligned(m, addr, size, True)           # 14.4: store / AMO misaligned
        old = m.load(addr, size, "store")
        new = op(old, _reg(m, i.f["rs2"]) & ((1 << bits) - 1), bits)
        m.store(addr, new, size, atomic=True)
        _set(m, i.f["rd"], _s32(old) if size == 4 else old)
    EXECUTORS[name] = ex


def _sb(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


for _sfx, _size in (("w", 4), ("d", 8)):
    for _n, _op in (
            ("amoswap", lambda a, b, n: b), ("amoadd", lambda a, b, n: a + b),
            ("amoxor", lambda a, b, n: a ^ b), ("amoand", lambda a, b, n: a & b),
            ("amoor", lambda a, b, n: a | b),
            ("amomin", lambda a, b, n: a if _sb(a, n) <= _sb(b, n) else b),
            ("amomax", lambda a, b, n: a if _sb(a, n) >= _sb(b, n) else b),
            ("amominu", lambda a, b, n: min(a, b)), ("amomaxu", lambda a, b, n: max(a, b))):
        _amo(f"{_n}.{_sfx}", _size, _op)


# -- Zba / Zbb (29) ----------------------------------------------------------------

def _clz(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return bits - v.bit_length()


def _ctz(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return bits if v == 0 else (v & -v).bit_length() - 1


def _rol(v: int, n: int, bits: int) -> int:
    v &= (1 << bits) - 1
    n %= bits
    return ((v << n) | (v >> (bits - n))) & ((1 << bits) - 1)


for _n, _fn in (
        ("sh1add", lambda m, a, b: (a << 1) + b), ("sh2add", lambda m, a, b: (a << 2) + b),
        ("sh3add", lambda m, a, b: (a << 3) + b),
        ("add.uw", lambda m, a, b: (a & M32) + b),
        ("sh1add.uw", lambda m, a, b: ((a & M32) << 1) + b),
        ("sh2add.uw", lambda m, a, b: ((a & M32) << 2) + b),
        ("sh3add.uw", lambda m, a, b: ((a & M32) << 3) + b),
        ("andn", lambda m, a, b: a & ~b), ("orn", lambda m, a, b: a | ~b),
        ("xnor", lambda m, a, b: ~(a ^ b)),
        ("max", lambda m, a, b: a if m.sgn(a) >= m.sgn(b) else b),
        ("maxu", lambda m, a, b: max(a, b)),
        ("min", lambda m, a, b: a if m.sgn(a) <= m.sgn(b) else b),
        ("minu", lambda m, a, b: min(a, b)),
        ("rol", lambda m, a, b: _rol(a, b, m.xlen)),
        ("ror", lambda m, a, b: _rol(a, -b % m.xlen, m.xlen))):
    _alu(_n, _fn)
_alu("slli.uw", lambda m, a, b: (a & M32) << b, imm=True)
_alu("rori", lambda m, a, b: _rol(a, -b % m.xlen, m.xlen), imm=True)
for _n, _fn in (("rolw", lambda m, a, b: _rol(a, b & 31, 32)),
                ("rorw", lambda m, a, b: _rol(a, -(b & 31) % 32, 32))):
    _alu(_n, _fn, word=True)
_alu("roriw", lambda m, a, b: _rol(a, -(b & 31) % 32, 32), imm=True, word=True)


def _unary(name: str, fn: Callable[[RVMachine, int], int], word: bool = False) -> None:
    def ex(m, i, pc):
        v = fn(m, _reg(m, i.f["rs1"]))
        _set(m, i.f["rd"], _w(v) if word else v)
    EXECUTORS[name] = ex


def _orc_b(m: RVMachine, v: int) -> int:
    out = 0
    for k in range(m.xlen // 8):
        if (v >> (8 * k)) & 0xFF:
            out |= 0xFF << (8 * k)
    return out


def _rev8(m: RVMachine, v: int) -> int:
    return int.from_bytes(v.to_bytes(m.xlen // 8, "little"), "big")


for _n, _fn in (
        ("clz", lambda m, v: _clz(v, m.xlen)), ("ctz", lambda m, v: _ctz(v, m.xlen)),
        ("cpop", lambda m, v: bin(v).count("1")),
        ("sext.b", lambda m, v: _sb(v & 0xFF, 8)), ("sext.h", lambda m, v: _sb(v & 0xFFFF, 16)),
        ("zext.h", lambda m, v: v & 0xFFFF), ("orc.b", _orc_b), ("rev8", _rev8)):
    _unary(_n, _fn)
for _n, _fn in (("clzw", lambda m, v: _clz(v, 32)), ("ctzw", lambda m, v: _ctz(v, 32)),
                ("cpopw", lambda m, v: bin(v & M32).count("1"))):
    _unary(_n, _fn)


# -- F / D (21, 22): emips.rvfpu -------------------------------------------------

from . import rvfpu as _rvfpu  # noqa: E402  (it imports this module's helpers)

EXECUTORS.update(_rvfpu._executors())
