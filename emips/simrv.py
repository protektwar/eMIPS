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
  the next PC): RV32I / RV64I (2, 4), M (12), A (13: LR / SC with one
  reservation, the AMOs), Zicsr (6, the CSR access rules of priv. 2.1),
  Zifencei (5: the I-cache made coherent), Zicntr (7), Zba / Zbb / Zbs /
  Zbc / Zbkb / Zbkc / Zbkx (29.4), Zicond (11), Zawrs (14: WRS
  waits on the reservation), Zimop / Zcmop (10), F / D (20, 21:
  :mod:`emips.rvfpu`).
- **Misaligned accesses** (``m25-RISC-V.md`` decision 7): ordinary loads
  and stores are done in hardware; LR / SC / AMOs raise address
  misaligned (4 / 6) — 13.2 / 13.4.
- **Traps** (phase 1 = M-mode only, priv. 3.1.6 / 3.1.14-3.1.17): mepc,
  mcause, mtval (the faulting address; an illegal instruction: its bits;
  an ``ebreak``: the PC), mstatus.MPIE / MIE / MPP, PC = mtvec.BASE.  An
  ``ecall`` asks ``syscall_handler`` first (RARS's services, decision 10:
  the service number in a7).  ``exceptions="raise"``: a trap raises
  :class:`RVException` and changes nothing.
- **Platform** (phase 1; LF13): physical addresses of ``pabits`` bits —
  34 on RV32 (Sv32's 22-bit PPN), 56 on RV64 (Sv39 / Sv48's 44-bit one),
  48 on the P8700 (MD01502) —, all RAM but for the devices; an access
  beyond them is an access fault (a PMA: nothing there);
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

from . import riscv as R
from . import rvcsr as CSR
from . import sim as S
from .exceptions import SimException
from .mmu import Translation
from .riscv import RV, RVDecodeError, RVISA, size_of
from .rvdebug import (BREAKPOINT as ACTION_BREAKPOINT, CAUSE_EBREAK, CAUSE_HALTREQ,
                      CAUSE_STEP, CAUSE_TRIGGER, DCSR_V, EBREAKM, EBREAKS, EBREAKU,
                      EBREAKVS, EBREAKVU, MPTE, MTE, STEP, Fired, Triggers)

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

#: exception codes (priv. Table 3.6)
INSN_MISALIGNED, INSN_ACCESS, ILLEGAL, BREAKPOINT = 0, 1, 2, 3
LOAD_MISALIGNED, LOAD_ACCESS, STORE_MISALIGNED, STORE_ACCESS = 4, 5, 6, 7
ECALL_U, ECALL_S, ECALL_M = 8, 9, 11
#: the hypervisor extension's (priv. Table 43): a VS ecall, the guest-page
#: faults, the virtual instruction
ECALL_VS, INSN_GUEST_PAGE, LOAD_GUEST_PAGE, VIRTUAL, STORE_GUEST_PAGE = 10, 20, 21, 22, 23
#: the software-check exception (Zicfilp / Zicfiss; priv. 3.1.16: xtval 2 =
#: a landing-pad fault, 3 = a shadow-stack fault)
SOFTWARE_CHECK = CSR.SOFTWARE_CHECK
#: the exceptions that write a guest virtual address to xtval when the
#: access was virtual (21.2.1 GVA; Spike's ``gva`` flags)
GVA_CAUSES = frozenset({0, 1, 3, 4, 5, 6, 7, 12, 13, 15, 20, 21, 23})
#: an NMI's request (interrupt_request; the P8700: emips.p8700ic)
NMI = -1
#: a resumable NMI's request (Smrnmi, priv. ch. 8)
RNMI = -2
#: the interrupts (priv. Table 14)
INTERRUPT_NAMES = {NMI: "non-maskable interrupt", RNMI: "resumable non-maskable interrupt",
                   1: "supervisor software interrupt", 3: "machine software interrupt",
                   5: "supervisor timer interrupt", 7: "machine timer interrupt",
                   9: "supervisor external interrupt", 11: "machine external interrupt",
                   2: "virtual supervisor software interrupt",
                   6: "virtual supervisor timer interrupt",
                   10: "virtual supervisor external interrupt",
                   13: "local counter-overflow interrupt",
                   12: "supervisor guest external interrupt",
                   # the AIA's (Table 5.2): RAS events, other local / custom ones
                   35: "low-priority RAS event interrupt",
                   43: "high-priority RAS event interrupt",
                   **{n: f"local interrupt {n}" for n in (14, 15, *range(16, 24),
                                                          *range(32, 35), *range(36, 43),
                                                          *range(44, 48))},
                   **{n: f"custom interrupt {n}" for n in (*range(24, 32), *range(48, 64))}}
#: the extensions of the default hart (Zihpm: hpmcounter3-31, read 0)
DEFAULT_EXTENSIONS = "imafdc_zicsr_zifencei_zicntr_zihpm_zba_zbb"
CAUSE_NAMES = {
    0: "instruction address misaligned", 1: "instruction access fault",
    2: "illegal instruction", 3: "breakpoint", 4: "load address misaligned",
    5: "load access fault", 6: "store/AMO address misaligned", 7: "store/AMO access fault",
    8: "environment call from U-mode", 9: "environment call from S-mode",
    11: "environment call from M-mode", 12: "instruction page fault",
    13: "load page fault", 15: "store/AMO page fault",
    10: "environment call from VS-mode", 20: "instruction guest-page fault",
    21: "load guest-page fault", 22: "virtual instruction", 23: "store/AMO guest-page fault",
    18: "software check",
}
#: the privilege modes' names (with H: VU / VS when V = 1 — 21.1, Table 38)
MODE_NAMES = {(3, 0): "M", (1, 0): "S", (0, 0): "U", (1, 1): "VS", (0, 1): "VU"}
#: the cacheability the bare platform gives memory / the MMIO page
CCA_CACHED, CCA_UNCACHED = 3, 2
MMIO_BASE = 0xFFFF0000
from .rvmmu import RVMMU, pma_cca  # noqa: E402  (the PMAs)
from . import tracerv  # noqa: E402  (the views of the instructions)


#: the extensions that need S-mode (LF8; Spike's isa_parser requirements)
S_EXTENSIONS = frozenset({"svadu", "svpbmt", "svnapot", "svinval", "sstc", "sscofpmf",
                          "sscsrind", "ssccfg", "smcdeleg", "ssaia"})


class RVTrap(Exception):
    """A synchronous exception an instruction raises (``cause``, ``tval``);
    ``custom``: one of the P8700's custom M-mode exceptions, taken through
    mipstvec (emips.p8700).  The hypervisor extension's (21.6): ``tval2``
    (htval / mtval2: a guest physical address >> 2), ``tinst`` (htinst /
    mtinst), ``gva`` (tval is a guest virtual address; None: the hart's V
    for :data:`GVA_CAUSES`)."""

    def __init__(self, cause: int, tval: int = 0, custom: bool = False, *, tval2: int = 0,
                 tinst: int = 0, gva: bool | None = None) -> None:
        super().__init__(cause_name(cause, custom))
        self.cause = cause
        self.tval = tval
        self.custom = custom
        self.tval2 = tval2
        self.tinst = tinst
        self.gva = gva


def cause_name(cause: int, custom: bool = False) -> str:
    """An exception's name (priv. Table 3.6; the P8700's Table 7.4)."""
    if custom:
        from .p8700 import CAUSE_NAMES as P
        return f"{P.get(cause) or CAUSE_NAMES.get(cause, f'cause {cause}')} (mipstvec)"
    return CAUSE_NAMES.get(cause, f"cause {cause}")


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
                 word: int | None = None, custom: bool = False) -> None:
        super().__init__(f"{cause_name(cause, custom)} at 0x{pc:x}", name=name, word=word,
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
    def effective(m, kind: str) -> tuple[int, int]:
        return RVMMU.effective(m, kind)

    @staticmethod
    def _cca(m, pa: int) -> int:
        return pma_cca(m, pa)

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        if va >> m.pabits:
            raise RVTrap(INSN_ACCESS if kind == "fetch" else
                         STORE_ACCESS if kind == "store" else LOAD_ACCESS, va)
        return va, pma_cca(m, va)

    def peek(self, m, va: int) -> tuple[int, int] | None:
        return None if va >> m.pabits else (va, pma_cca(m, va))

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        if va >> m.pabits:
            return Translation(va, None, CCA_UNCACHED, "bare", False,
                               f"bare: no physical memory above {m.pabits} bits",
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
    #: the P8700's Hybrid Debug (emips.p8700dbg.HartDebug): its Debug Mode
    #: executes a handler (MIPS EJTAG's) — None: the Debug spec's halted hart
    hybrid = None
    _step_next = False
    #: the trace encoder (emips.rvtrace) and its stall request (LF11)
    trace_encoder = None
    trace_stall = False
    #: where the trigger module's external outputs go (actions 8 / 9)
    tm_ext_output = None
    #: the operands the pipeline bypassed for the instruction it issues
    #: (emips.rvpipeline; the views name the path)
    rv_sources: dict | None = None
    #: the MIPS P8700's own state (emips.p8700) on a P8700 hart
    p8700 = None
    #: the 64-byte granule of the LR reservation of a P8700 cluster's hart
    #: (another hart's store there clears it; ``cluster``: sim.Machine)
    lr_granule: int | None = None
    #: the NMI input is asserted (a P8700's APLIC / watchdog)
    nmi_pending = False
    #: the vector unit (emips.rvvector; None without V / Zve*)
    vector = None
    #: the virtualization mode V (priv. 21.1: 1 = VS / VU-mode)
    v = 0
    #: an HLV / HLVX / HSV's access in progress ("v" / "x"; 21.3.1)
    hyp_access: str | None = None
    #: a Zicfiss instruction's shadow-stack access in progress / a CBO's
    #: translation (priv. 22.2.3: emips.rvmmu's ssload / ssstore kinds)
    ss_access = False
    cbo_access = False
    #: Smrnmi (priv. ch. 8): the RNMI input (None or the RNMI's cause code)
    #: and the implementation-defined handler addresses (QEMU's
    #: rnmi-interrupt-vector / rnmi-exception-vector; 0 as QEMU and Spike)
    rnmi_pending: int | None = None
    rnmi_vector = 0
    rnmi_exc_vector = 0
    #: the AIA (LF9): the hart's IMSIC (emips.rvaia, a platform's), its pages
    #: (only aligned 32-bit accesses: AIA 3.5), the external interrupt
    #: priority a direct-mode APLIC gives (level "m" / "s" -> number or 0)
    imsic = None
    imsic_regions: tuple = ()
    ext_priority = None
    #: what the last CBO did and its block (VA, PA, CCA): the views
    cmo_note: str | None = None
    cmo_access: tuple[int, int, int] | None = None
    #: the mode (priv, V) the instruction being executed started in: the
    #: counters count its cycle and its retirement there (Smcntrpmf 7.3)
    count_mode: tuple[int, int] | None = None

    def __init__(self, memory=None, pc: int = 0, *, arch: str = "rv64",
                 extensions: str = DEFAULT_EXTENSIONS, modes: str = "msu",
                 exceptions: str = "cp0", mmu: Any = None, caches=None, cpu: int = 0,
                 pmp: int = 0, triggers: int = 4, core: str | None = None,
                 **_ignored: Any) -> None:
        if arch not in ("rv32", "rv64"):
            raise ValueError(f"arch must be 'rv32' or 'rv64', not {arch!r}")
        if core not in (None, "", "p8700"):
            raise ValueError(f"core must be None or 'p8700', not {core!r}")
        if core == "p8700" and arch != "rv64":
            raise ValueError("the P8700 is an RV64 core (MD01502 1.2)")
        if modes not in ("m", "mu", "msu"):
            raise ValueError(f"modes must be 'm', 'mu' or 'msu', not {modes!r}")
        from .riscv import parse_extensions
        if "h" in parse_extensions(extensions):  # (priv. 21: S-mode and paging)
            if modes != "msu":
                raise ValueError("the hypervisor extension needs M, S and U modes "
                                 "(priv. ch. 21)")
            if core == "p8700":
                raise ValueError("the P8700 has no hypervisor extension (MD01502 1.2)")
        s_only = parse_extensions(extensions) & S_EXTENSIONS
        if s_only and "s" not in modes:        # (LF8: supervisor extensions; Spike's
            raise ValueError(f"{', '.join(sorted(s_only))}: supervisor-level, they need "
                             "S-mode (--rv-modes msu)")         # isa_parser)
        if mmu is None:                         # S-mode: page-based translation (2b)
            from .rvmmu import RVMMU
            if core == "p8700" and "s" in modes:    # (its TLBs and walkers: 5c)
                from .p8700mmu import P8700MMU
                mmu = P8700MMU()
            else:
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
        self.ialign = 2 if "zca" in self.extensions else 4     # (C or Zca: 28.5)
        #: the privilege modes the hart has (priv. 1.2: M, M + U, M + S + U)
        #: and the current one: 3 = M, 1 = S, 0 = U (reset: M, priv. 3.4)
        self.modes = frozenset({"m": 3, "s": 1, "u": 0}[k] for k in modes)
        self.priv = 3
        #: address translation (satp, priv. 12.1.11; emips.rvmmu): the MODEs
        #: satp accepts (0 = Bare; Sv32 = 1; Sv39 / Sv48 / Sv57 = 8-10) and
        #: whether a page-based mode exists
        self.paging = "s" in modes
        #: the physical address space (LF13): Sv32's 22-bit PPN gives 34 bits,
        #: Sv39 / Sv48 / Sv57's 44-bit PPN 56 (priv. 12.3-12.6), the P8700 48
        #: (MD01502 1.2: "48-bit virtual and physical addresses") — all of it
        #: RAM but for the platform's devices; beyond it an access fault
        self.pabits = 48 if core == "p8700" else 34 if self.xlen == 32 else 56
        self.pamask = (1 << self.pabits) - 1
        if caches is not None:
            caches.set_pamask(self.pamask)
        self.satp_modes = frozenset({0} | ({1} if self.xlen == 32 else {8, 9, 10})
                                    if self.paging else {0})
        if core == "p8700":                     # (Sv39 / Sv48: QEMU's mips-p8700, 2.7)
            from .p8700 import P8700
            from .p8700dbg import HartDebug
            self.p8700 = P8700(self)
            self.hybrid = HartDebug(self)       # (LF16: MIPS EJTAG's Debug Mode)
            self.satp_modes = frozenset({0, 8, 9} if self.paging else {0})
        #: WFI is waiting for an interrupt (priv. 3.3.3), or a WRS (Zawrs)
        self.waiting = False
        #: a WRS's wait: None (WFI), -1 (WRS.NTO) or the cycle WRS.STO's
        #: timeout ends
        self.wrs_until: int | None = None
        #: the platform's timer (emips.rvplatform.CLINT) and I/O regions (PMAs)
        self.timer = None
        self.io_regions: list[tuple[int, int]] = []
        #: the hart's number (mhartid)
        self.hartid = cpu
        #: the counters a CSR write set during this instruction (priv.
        #: 3.1.10: the write takes effect after the instruction completes)
        self.counter_written: set[str] = set()
        #: the LR reservation: (address, size) or None (13.2)
        self.reservation: tuple[int, int] | None = None
        #: FLEN (21.2, 22): 128 with Q, 64 with D, 32 with F only, 0 without F
        #: (Zfinx & co.: no f registers either)
        e = self.extensions
        self.flen = 128 if "q" in e else 64 if "d" in e else 32 if "f" in e else 0
        #: Zfinx & co. (ch. 26): floating point on the x registers
        self.inx = "zfinx" in e
        #: fcsr = frm 7:5 | fflags 4:0 (20.2); the f registers (FLEN bits;
        #: emips.rvfpu NaN-boxes narrower values)
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
        #: the trace ingress port (emips.rvingress: LF11), None = no encoder
        self.ingress = None
        self.dm_access: str | None = None
        #: the trigger module (Debug spec ch. 5; emips.rvdebug)
        self.tm = Triggers(self, triggers)
        #: the vector unit (LF6, unpriv. ch. 30; emips.rvvector): with V or a Zve*
        from .rvvector import VectorUnit
        self.vector = VectorUnit(self, e) if "zve32x" in e else None
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
    def mode_name(self) -> str:
        """The privilege mode's name (priv. 21.1, Table 38: with H, S is
        HS; VS / VU when V = 1)."""
        from .tracerv import mode_name
        return mode_name(self.priv, self.v, self.csrs.has_h)

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
        to: 3 M, 1 HS, -1 VS), or None (priv. 3.1.9 / 10.1.3 / 21.2.3:
        pending in mip and enabled in mie; M-level ones first — not
        delegated, when the hart is below M or MIE —, then HS-level ones
        — delegated by mideleg but not hideleg, when V = 1, below S or in
        HS with SIE —, then with V = 1 the VS-level ones — hideleg, below
        VS or in VS with vsstatus.SIE —; in each the priority MEI, MSI,
        MTI, SEI, SSI, STI, SGEI, VSEI, VSSI, VSTI)."""
        if self.debug_mode:                     # (Debug spec 4.1: none in Debug Mode)
            return None
        hy = self.hybrid
        if self.nmi_pending and (hy is None or hy.nmie):
            return NMI, 3                       # (non-maskable: before every interrupt)
        c = self.csrs
        if c.smrnmi:                            # 8.3: NMIE = 0 masks every interrupt
            if not c.mnstatus & CSR.NMIE:
                return None
            if self.rnmi_pending is not None:
                return RNMI, 3                  # (8.1: above every other interrupt)
        if hy is not None and not hy.inte:      # (the P8700's DCR.IntE: MD00047 3.1)
            return None
        if c.aia is not None:
            return self._aia_request()
        pend = c.mip() & c.mie
        if not pend:
            return None
        if self.p8700 is not None and self.p8700.erl:
            return None                         # (mipsconfig5.ERL: none but NMI)
        priv, v = self.priv, self.v
        to = 3
        ints = pend & ~c.mideleg if priv < 3 or c.status_bit(CSR.MIE) else 0
        if not ints:
            to, hideleg = 1, c.hideleg & c.mideleg
            if v or priv < 1 or (priv == 1 and c.status_bit(CSR.SIE)):
                ints = pend & c.mideleg & ~hideleg
            if not ints and v and (priv < 1 or c.vsstatus >> CSR.SIE & 1):
                to, ints = -1, pend & hideleg
        if not ints:
            return None
        for bit in c.priority:
            if ints >> bit & 1:
                return bit, to
        return None

    def _aia_request(self) -> tuple[int, int] | None:
        """With the AIA (AIA 5.2.2 / 5.4.2 / 6.3.4): the top of machine
        level (mtopi) when M-level interrupts are enabled, then of HS level
        (stopi), then with V = 1 of VS level (vstopi) — each the highest by
        the iprio arrays / hviprio / hvictl and the default order.  A VS
        interrupt comes back as its HS twin's number for 1 / 5 / 9 (2 / 6
        / 10: _enter's convention), others as themselves."""
        c = self.csrs
        a = c.aia
        priv, v = self.priv, self.v
        if priv < 3 or c.status_bit(CSR.MIE):
            t = a.mtopi_pair()
            if t is not None:
                return t[0], 3
        if v or priv < 1 or (priv == 1 and c.status_bit(CSR.SIE)):
            t = a.stopi_pair()
            if t is not None:
                return t[0], 1
        if v and (priv < 1 or c.vsstatus >> CSR.SIE & 1) and c.has_h:
            t = a.vstopi_pair()
            if t is not None:
                n = t[0]
                return (n + 1 if n in (1, 5, 9) else n), -1
        return None

    def set_interrupt(self, bit: int, level: bool) -> None:
        """A platform interrupt line (MSIP / MTIP / MEIP, the external
        SEIP): mip bit ``bit`` follows it (the P8700: where mipsintctl
        routes it)."""
        if self.p8700 is not None:
            self.irq_lines = self.p8700.set_line(bit, level)
            return
        if level:
            self.irq_lines |= 1 << bit
        else:
            self.irq_lines &= ~(1 << bit)

    def pending_cache_error(self):
        return None

    def _cache_arch(self, err) -> "RVTrap":
        """An error found in a cache (sim.Machine's loads / stores / fetches
        catch emips.cache.CacheError): the P8700's Cache Error, custom
        exception 48 (emips.p8700err)."""
        p = self.p8700
        if p is None:                           # (no RISC-V cache error exists)
            raise SimException(f"a cache error on a hart without one: {err}")
        p.errors().from_error(err)
        from .p8700err import CACHE_ERROR
        return RVTrap(CACHE_ERROR, 0, custom=True)

    def take_cache_error(self, pc: int) -> bool:
        """An imprecise L2 error waiting (emips.cache.Caches.pending): the
        Cache Error before the instruction at ``pc`` (xepc = pc)."""
        c = self.caches
        if c is None or c.pending is None or self.p8700 is None:
            return False
        hy = self.hybrid
        if hy is not None:                      # (LF16: deferred while IEXI — CACHEEP)
            if self.debug_mode and hy.defers():
                return False
            hy.cacheep = False
        self.waiting, self.wrs_until = False, None
        self._trap(self._cache_arch(c.take_pending()), pc, None)
        return True

    def advance(self, cycles: int = 1) -> None:
        self.cycles += cycles
        self.csrs.tick(cycles, self.count_mode, skip_cycle="mcycle" in self.counter_written)
        p = self.p8700
        if p is not None and p.counters.armed:
            p.counters.tick(cycles)
        for dev in self.devices:
            dev.tick(self, cycles)
        if self.caches is not None and self.caches.fsb:
            self.caches.tick()

    def address_ok(self, addr: int) -> bool:
        return True

    # -- dseg in the views (the P8700's Hybrid Debug, LF16) ----------------------------

    def _dseg_view(self, addr: int) -> bool:
        hy = self.hybrid
        return hy is not None and self.debug_mode and hy.covers(addr & self.amask)

    def translate(self, addr: int) -> Translation:
        if self._dseg_view(addr):
            va = addr & self.amask
            from .p8700dbg import DRSEG_BASE
            seg = "drseg" if va >= DRSEG_BASE else "dmseg"
            return Translation(va, None, CCA_UNCACHED, seg, False,
                               "Hybrid Debug " + ("registers" if seg == "drseg" else
                                                  "memory (the DBU's RAM)")
                               + ": no physical address")
        return super().translate(addr)

    def read_byte(self, addr: int) -> int:
        if self._dseg_view(addr):
            v = self.hybrid.peek(addr & self.amask, 1)
            if v is not None:
                return v
        return super().read_byte(addr)

    def code_half(self, addr: int) -> int:
        if self._dseg_view(addr):
            v = self.hybrid.peek(addr & self.amask, 2)
            if v is None:
                from .exceptions import UnmappedAddress
                raise UnmappedAddress(f"0x{addr & self.amask:x}: dmseg, no probe serves it",
                                      name=None, word=None, addr=addr)
            return v
        return super().code_half(addr)

    def kernel_mode(self) -> bool:
        return self.priv == 3

    # -- execution ---------------------------------------------------------------

    def _half(self, addr: int) -> int:
        """A fetched halfword; a fault names this halfword's address (priv.
        3.1.16: the part of the instruction that faulted).  PMP checks each
        16-bit parcel (as Spike fetches)."""
        if self.debug_mode and self.hybrid is not None and self.hybrid.covers(addr & self.amask):
            return self.hybrid.fetch_half(addr & self.amask)    # (dseg: LF16)
        try:
            if self.pmp.entries:
                pa, _ = self._xlate(self, addr & self.amask, "fetch")
                if not self.pmp.allows(pa, 2, "fetch", self.priv):
                    raise RVTrap(INSN_ACCESS, addr)
            return self.fetch_half(addr)
        except RVTrap as t:
            if t.tval == addr:
                raise
            u = RVTrap(t.cause, addr, t.custom)         # (a P8700 TLB miss keeps its facts)
            if hasattr(t, "tlb_miss"):
                u.tlb_miss = t.tlb_miss
            raise u from None

    def _access(self, addr: int, size: int, kind: str) -> tuple[list[int], int]:
        """Translate an access (sim.Machine) and check PMP on its physical
        bytes (priv. 3.7.1; fetches: per parcel in :meth:`_half`)."""
        pas, cca = super()._access(addr, size, kind)
        if self.imsic_regions and kind != "fetch" and (size != 4 or pas[0] & 3):
            for lo, hi in self.imsic_regions:   # (AIA 3.5: an IMSIC page takes aligned
                if lo <= pas[0] < hi:            # words only — Spike: an access fault)
                    raise RVTrap(STORE_ACCESS if kind == "store" else LOAD_ACCESS, addr)
        if self.ss_access and kind != "fetch" and pma_cca(self, pas[0]) == CCA_UNCACHED:
            virt = self.mmu.effective(self, "store")[1] if hasattr(self.mmu, "effective") \
                else 0                          # (22.2.3: idempotent memory only)
            raise RVTrap(STORE_ACCESS, addr, gva=bool(virt))
        p = self.pmp
        if p.entries and kind != "fetch":
            k = "store" if kind == "store" or self.ss_access else "load"
            priv, virt = self.mmu.effective(self, k) if hasattr(self.mmu, "effective") \
                else (self.priv, self.v)
            if k == "load" and self.hyp_access == "x":
                k = "hlvx"                      # (21.3.1: read and execute)
            start = 0
            for j in range(1, size + 1):           # (one check per physically contiguous run)
                if j == size or pas[j] != pas[j - 1] + 1:
                    if not p.allows(pas[start], j - start, k, priv):
                        raise RVTrap(STORE_ACCESS if k == "store" else LOAD_ACCESS, addr,
                                     gva=bool(virt))     # (a shadow-stack access: a store's)
                    start = j
        return pas, cca

    def load(self, addr: int, size: int, kind: str = "load", pre: bool = True) -> int:
        """sim.Machine's load, with the load triggers: the address before
        the access (``pre``; LR / AMOs check it themselves, ahead of their
        alignment), the data after it (Debug spec 5.5, Table 13)."""
        hy = self.hybrid
        if hy is not None:                      # (the P8700's Hybrid Debug: LF16)
            if self.debug_mode:
                if hy.covers(addr & self.amask):
                    return self._dseg_load(addr & self.amask, size)
            elif hy.bp.db_on:
                hy.data(addr & self.amask, size, False)
                if hy._vc is not None:
                    v = self._load(addr, size, kind, pre)
                    hy.loaded(addr & self.amask, size, v)
                    return v
        return self._load(addr, size, kind, pre)

    def _load(self, addr: int, size: int, kind: str, pre: bool) -> int:
        tm = self.tm
        if not tm.loads:
            return super().load(addr, size, kind)
        if pre:
            tm.memory("load", addr, size)
        v = super().load(addr, size, kind)
        tm.memory("load", addr, size, v)
        return v

    def _dseg_load(self, va: int, size: int) -> int:
        v = self.hybrid.load(va, size)
        if self._mem_log is not None:
            self._log("read", va, size, v)
        return v

    def store(self, addr: int, value: int, size: int, *, gather: bool = True,
              atomic: bool = False) -> bool | None:
        """sim.Machine's store, after the store triggers (address and data)."""
        hy = self.hybrid
        if hy is not None:                      # (the P8700's Hybrid Debug: LF16)
            if self.debug_mode:
                va = addr & self.amask
                if hy.covers(va):
                    value &= (1 << 8 * size) - 1
                    if self._mem_log is not None:
                        self._log("write", va, size, hy.peek(va, size) or 0, value)
                    hy.store(va, value, size)
                    return None
            elif hy.bp.db_on:
                hy.data(addr & self.amask, size, True, value & ((1 << 8 * size) - 1))
        if self.tm.stores:
            self.tm.memory("store", addr, size, value & ((1 << 8 * size) - 1))
        cl = self.cluster
        if cl is not None and any(o.reservation is not None for o in cl.machines if o is not self):
            pa = self.translate(addr).pa        # (MD01502 3.2: the other harts' granule)
            if pa is not None:
                cl.stored(self, pa)
        return super().store(addr, value, size, gather=gather, atomic=atomic)

    def trigger_check(self, op: str, addr: int, size: int) -> None:
        """The address triggers of an access (``op`` load / store) that
        are checked ahead of its alignment (LR, AMOs)."""
        if self.tm.loads if op == "load" else self.tm.stores:
            self.tm.memory(op, addr, size)

    def load_pte(self, pa: int, size: int, view: bool = False) -> int:
        """A page-table entry at physical address ``pa`` (emips.rvmmu; through
        the caches, as an implicit load — not one of the instruction's own
        accesses in its view); None when PMP refuses it (an S-mode read,
        priv. 3.7.2).  ``view``: as the views read, with no side effect."""
        if not self.pmp.allows(pa, size, "load", 1):
            return None
        cca = pma_cca(self, pa)
        if view:
            cached = self._uses_dcache(cca)
            return sum(self._view_byte(pa, pa + k, cached) << (8 * k) for k in range(size))
        log, self._mem_log = self._mem_log, None
        try:
            return self.load_pa(pa, [pa + k for k in range(size)], cca, size)
        finally:
            self._mem_log = log

    def store_pte(self, pa: int, value: int, size: int) -> bool:
        """Svadu's PTE update (priv. 12.3.2 step 9, LF8): ``value`` stored at
        physical ``pa`` — PMP-checked as an S-mode write (3.7.2), refused in
        an I/O region (page tables need the "hardware page-table write"
        PMA), through the caches as an atomic store, not one of the
        instruction's own accesses in its view.  False: refused (the
        caller's access fault).  It is atomic with the walk's read: nothing
        else runs within one instruction step."""
        if not self.pmp.allows(pa, size, "store", 1):
            return False
        cca = pma_cca(self, pa)
        if cca == 2:
            return False
        pas = [pa + k for k in range(size)]
        data = (value & ((1 << 8 * size) - 1)).to_bytes(size, "little")
        cl = self.cluster
        if cl is not None:
            cl.stored(self, pa)                 # (another hart's LR granule)
        log, self._mem_log = self._mem_log, None
        try:
            c = self.caches
            if c is not None and c.cacheable and cca in self.cached_cca:
                c.write(pa & M32, pas, data, cca, True)
            else:
                for k, b in enumerate(data):
                    self.phys_write_byte(pa + k, b)
        finally:
            self._mem_log = log
        return True

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
        if self.debug_mode and self.hybrid is None:     # halted (Debug spec 4.1): the
            self.advance(1)                     # Debug Module drives the hart (phase 2f)
            return None
        if self.halt_request and not self.debug_mode:   # haltreq (4.7)
            self.halt_request = False
            self.enter_debug(CAUSE_HALTREQ)
            return None
        if self.trace_stall and not self.trace_encoder.drain():
            self.advance(1)                     # (the trace encoder stalls the hart:
            return None                         # trTeInstStallEna, LF11)
        pc = self.pc
        if self.caches is not None and self.caches.pending is not None \
                and self.take_cache_error(pc):  # (an imprecise L2 error, MD01502 3.3.8)
            self.advance(1)
            return None
        # dcsr.stepie = 0: no interrupt while stepping (4.9.1)
        req = None if self._stepping or self._step_next else self.interrupt_request()
        if req is not None:
            self.advance(1)
            self.take_interrupt(req, pc)
            return None
        if self.waiting:                        # WFI / WRS: until mip & mie (3.3.3)
            if self.still_waiting():
                self.advance(1)
                return None
        return self.issue(pc)

    def still_waiting(self) -> bool:
        """A WFI / WRS waits on (False: it has just ended): until an
        interrupt is pending and enabled in mie (priv. 3.3.3); a WRS also
        until the reservation set is gone or WRS.STO's timeout (Zawrs 14.1)."""
        c = self.csrs
        a = c.aia
        if a is not None:                       # (AIA 5.5: a top at any level)
            end = a.mtopi_pair() is not None or (c.has_s and a.stopi_pair() is not None) \
                or (c.has_h and a.vstopi_pair() is not None)
        else:
            end = bool(c.mip() & c.mie)
        if self.wrs_until is not None:
            end = end or self.reservation is None or 0 <= self.wrs_until <= self.cycles
        if end:
            self.waiting = False
            self.wrs_until = None
        return not end

    def take_interrupt(self, req: tuple[int, int], pc: int) -> None:
        """Take the interrupt ``req`` (:meth:`interrupt_request`) before
        the instruction at ``pc`` (xepc = pc)."""
        tm = self.tm
        tm.after = None
        self.waiting, self.wrs_until = False, None
        if req[0] == NMI:
            self.nmi_pending = False
            self._take_nmi(pc)
            return
        if req[0] == RNMI:
            self._take_rnmi(pc)
            return
        if tm.icounts:                          # (a trap from an enabled mode counts)
            tm.icount_trap(self.priv, self.v)
        prev = (self.priv, self.v)
        p = self.p8700
        if p is not None and p.counters.armed:
            p.counters.trap()
        to = req[1]
        self._enter(req[0], 0, pc, 1 if to == -1 else to, interrupt=True, virt=to == -1)
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
        if self._step_next:                     # (a DRET with dcsr.step: this one steps)
            self._stepping, self._step_next = True, False
        tick = self.advance if clock else (lambda n: None)
        mode = (self.priv, self.v)              # (the counters: the mode it started in)
        self.count_mode = mode
        v = tracerv.begin(self, pc) if self.tracing else None   # (the view: emips.tracerv)
        try:
            if tm.ext_pending:                  # (a tmexttrigger input: 5.7.16)
                f = tm.external()
                if f is not None:
                    raise f
            if tm.icounts:
                f = tm.icount()
                if f is not None:
                    raise f
            hy = self.hybrid
            if hy is not None and hy.bp.ib_on and not self.debug_mode:
                hy.instruction(pc)              # (the P8700's breakpoints: LF16)
            if self.csrs.elp:                   # (Zicfilp: the target must be an LPAD)
                i = self._landing_pad(pc, i, fault)
            if fault is not None:
                raise fault
            if i is None:
                i = self.fetch(pc)
            if v is not None:
                tracerv.fetched(self, v, i)
            nxt = EXECUTORS[i.base.name](self, i, pc)
        except Fired as f:
            tick(1)
            self.count_mode = None
            self._fire(f, pc)
            self._stepped()
            if v is not None:
                self.last_view = tracerv.finish(self, v, i, None, "a trigger fired")
            return i
        except EnterDebug as e:
            tick(1)
            self.count_mode = None
            self.enter_debug(e.cause, pc)
            if v is not None:
                self.last_view = tracerv.finish(self, v, i, None, "Debug Mode")
            return i
        except RVTrap as t:
            tick(1)
            self.count_mode = None
            self._trap(t, pc, i)
            self._stepped()
            if v is not None:
                self.last_view = tracerv.finish(self, v, i, None, cause_name(t.cause, t.custom))
            return i
        except SimException as exc:             # (a simulator halt)
            self._mem_log = None
            self.count_mode = None
            self.exception_log.append({"name": getattr(exc, "name", None),
                                       "word": getattr(exc, "word", None),
                                       "addr": getattr(exc, "addr", pc),
                                       "error": type(exc).__name__})
            raise
        if "minstret" not in self.counter_written:
            self.csrs.retire(mode)
        self.last_insn = i
        npc = (pc + i.size if nxt is None else nxt) & self.amask
        p = self.p8700
        if p is not None and p.counters.armed:
            p.counters.retired(i, pc, npc)
        hpm = self.csrs.hpm
        if hpm is not None and hpm.armed:
            hpm.retired(i, pc, npc, mode)
        if clock:
            self.advance(1)
            self.counter_written.clear()
        self.count_mode = None
        self.pc = npc
        if self.ingress is not None:            # (the trace encoder: E-Trace 4.2)
            self.ingress.retire(pc, i, npc, mode)
        if self.call_hook is not None:          # (the debugger's call stack)
            self.call_hook(pc, i, i.raw, self.pc)
        if v is not None:
            self.last_view = tracerv.finish(self, v, i, self.pc)
        if tm.after is not None:                # a timing-after trigger (5.7.12 hit = 3)
            f, tm.after = tm.after, None
            self._fire(f, self.pc)
        self._stepped()
        return i

    def _landing_pad(self, pc: int, i: RV | None, fault: Exception | None) -> RV | None:
        """unpriv. 33.1.1 / priv. 22.1.2: with ELP = LP_EXPECTED the next
        instruction must be an LPAD — else a software-check exception
        (xtval 2), after the fetch's own faults (an instruction access
        fault outranks it) and before decoding (it outranks an illegal
        instruction).  The LPAD itself checks its alignment and label."""
        if fault is not None:
            if isinstance(fault, RVTrap) and fault.cause == ILLEGAL:
                raise RVTrap(SOFTWARE_CHECK, CSR.LANDING_PAD_FAULT)
            return i
        if i is None:
            try:
                i = self.fetch(pc)
            except RVTrap as t:
                if t.cause == ILLEGAL:
                    raise RVTrap(SOFTWARE_CHECK, CSR.LANDING_PAD_FAULT) from None
                raise
        if i.base.name != "lpad":
            raise RVTrap(SOFTWARE_CHECK, CSR.LANDING_PAD_FAULT)
        return i

    def _trap(self, t: RVTrap, pc: int, i: RV | None, triggers: bool = True) -> None:
        """Take a synchronous exception (priv. 3.1.8: to S-mode when
        medeleg delegates it and the hart is in S or U); then the
        etriggers (``triggers``: not for a trigger's own breakpoint)."""
        if self.exceptions == "raise":
            raise RVException(t.cause, t.tval, pc, name=i.name if i else None,
                              word=i.raw if i else None, custom=t.custom)
        if self.debug_mode and self.hybrid is not None:
            self.hybrid.exception(t.cause, pc)  # (MD00047 2.4.3: dpc, DEXCCODE, the vector)
            return
        prev = (self.priv, self.v)
        custom = t.custom
        c = self.csrs
        to, virt = 3, False                     # (3.1.8 / 21.6.2: M unless medeleg, then
        if not custom and self.priv <= 1 and t.cause < self.xlen and \
                c.medeleg >> t.cause & 1:       # HS unless V = 1 and hedeleg: VS)
            to, virt = 1, bool(self.v and c.hedeleg >> t.cause & 1)
        self.exception_log.append({"name": cause_name(t.cause, custom), "cause": t.cause,
                                   "tval": t.tval, "epc": pc, "word": i.raw if i else None,
                                   "priv": self.priv, "v": self.v, "to": to, "virt": virt})
        p = self.p8700
        if p is not None and p.counters.armed:
            p.counters.trap()
        miss = getattr(t, "tlb_miss", None)
        if miss is not None:                    # (a software-walk TLB miss: 7.3.2)
            p.tlb_miss(miss)
        gva = t.gva if t.gva is not None else bool(self.v) and t.cause in GVA_CAUSES
        self._enter(t.cause, t.tval, pc, to, interrupt=False,
                    vector=p.vector(t.cause) if custom else None, virt=virt,
                    tval2=t.tval2, tinst=t.tinst, gva=gva)
        if custom and t.cause == 48:            # (a Cache Error sets ERL: MD01502 7.3.9)
            from .p8700 import ERL
            p.config5 |= ERL
        if triggers:
            self._trap_triggers(t.cause, False, prev)

    def _trap_triggers(self, cause: int, interrupt: bool, prev: tuple[int, int]) -> None:
        """Debug spec 5.7.14 / 5.7.15: an itrigger / etrigger fires after
        the trap, before the handler's first instruction (tval 0); ``prev``
        = (the mode, V) the trap came from."""
        if self.tm.traps:
            f = self.tm.trap(cause, interrupt, prev)
            if f is not None:
                self._fire(f, self.pc)

    def _fire(self, f: Fired, epc: int) -> None:
        """A trigger's action (Table 12): a breakpoint exception (epc:
        the instruction that matched, or the next one for timing after)
        or Debug Mode (cause 2, dpc = epc)."""
        if f.action == ACTION_BREAKPOINT:
            self._trap(RVTrap(BREAKPOINT, f.tval, gva=f.gva), epc, None, triggers=False)
        else:
            self.enter_debug(CAUSE_TRIGGER, epc)

    def _stepped(self) -> None:
        """dcsr.step (4.5.1): after the instruction (or the trap it took,
        the PC at the handler) the hart re-enters Debug Mode, cause 4."""
        if self._stepping:
            self._stepping = False
            if not self.debug_mode:
                self.enter_debug(CAUSE_STEP)

    def _enter(self, cause: int, tval: int, pc: int, to: int, *, interrupt: bool,
               vector: int | None = None, virt: bool = False, tval2: int = 0,
               tinst: int = 0, gva: bool = False) -> None:
        """Trap entry (priv. 3.1.6.1, 3.1.7, 3.1.14-3.1.17; 10.1.1-10.1.9
        for S-mode; 21.6.2 with H): xEPC, xCAUSE (the interrupt bit
        XLEN-1), xTVAL, the xPIE / xIE / xPP stack, the PC from xtvec
        (vectored: BASE + 4 * cause for an interrupt) or ``vector`` (the
        P8700's mipstvec); into M-mode tcontrol.MPTE = MTE, MTE = 0 (Debug
        spec 5.7.6).  ``to`` 1 with ``virt``: into VS-mode (vsepc ...,
        vsstatus, V stays 1; a VS interrupt's cause is its S twin's, 21.2.2);
        into HS / M: htval / mtval2 = ``tval2``, htinst / mtinst = ``tinst``,
        hstatus SPV / SPVP / GVA or mstatus MPV / GVA, V = 0."""
        c = self.csrs
        elp, c.elp = c.elp, 0                   # (22.1.2: xPELP = ELP, ELP = NO_LP_EXPECTED)
        if virt and interrupt and cause in (CSR.VSSI, CSR.VSTI, CSR.VSEI):
            cause -= 1                          # (VSSI -> SSI, VSTI -> STI, VSEI -> SEI)
        code = cause | ((1 << (self.xlen - 1)) if interrupt else 0)
        if interrupt:
            name = INTERRUPT_NAMES.get(cause)
            if name is None and self.p8700 is not None:
                from .p8700 import INTERRUPT_NAMES as P
                name = P.get(cause)
            self.exception_log.append({"name": name, "cause": code, "epc": pc,
                                       "priv": self.priv, "v": self.v, "to": to,
                                       "virt": virt})
        if to == 1 and virt:
            c.vsepc = pc & ~1
            c.vscause = code
            c.vstval = tval & self.amask
            vs = c.vsstatus & ~((1 << CSR.SPIE) | (1 << CSR.SIE) | (1 << CSR.SPP))
            c.vsstatus = vs | (c.vsstatus >> CSR.SIE & 1) << CSR.SPIE | (self.priv & 1) << CSR.SPP
            if c.zicfilp:
                c.vsstatus = (c.vsstatus & ~(1 << CSR.SPELP)) | elp << CSR.SPELP
            tvec = c.vstvec
        elif to == 1:
            c.sepc = pc & ~1
            c.scause = code
            c.stval = tval & self.amask
            c.htval, c.htinst = tval2 & self.amask, tinst & self.amask
            c.set_status(spie=c.status_bit(CSR.SIE), sie=0, spp=self.priv)
            if c.zicfilp:
                c.mstatus = (c.mstatus & ~(1 << CSR.SPELP)) | elp << CSR.SPELP
            if c.has_h:                         # (21.2.1: SPV, SPVP when V was 1, GVA)
                h = c.hstatus & ~((1 << CSR.H_SPV) | (1 << CSR.H_GVA))
                if self.v:
                    h = (h & ~(1 << CSR.H_SPVP)) | (self.priv & 1) << CSR.H_SPVP
                c.hstatus = h | self.v << CSR.H_SPV | int(gva) << CSR.H_GVA
            self.v = 0
            tvec = c.stvec
        else:
            c.mepc = pc & ~1
            c.mcause = code
            c.mtval = tval & self.amask
            if self.p8700 is not None:          # (MPPPS / MTW: MD01502 7.3.9)
                self.p8700.m_trap(vector is not None, c.status_field(CSR.MPP_SHIFT, 3))
            c.set_status(mpie=c.status_bit(CSR.MIE), mie=0, mpp=self.priv)
            if c.zicfilp:
                c.mstatus = (c.mstatus & ~(1 << CSR.MPELP)) | elp << CSR.MPELP
            if c.has_h:                         # (21.4.1: MPV, GVA; 21.4.4-5)
                c.mtval2, c.mtinst = tval2 & self.amask, tinst & self.amask
                c.mstatus = (c.mstatus & ~((1 << CSR.MPV) | (1 << CSR.GVA))) \
                    | self.v << CSR.MPV | int(gva) << CSR.GVA
            self.v = 0
            tvec = c.mtvec
            tm = self.tm
            if tm.count and not c.has_s:
                tm.tcontrol = MPTE if tm.tcontrol & MTE else 0
        from_m = self.priv == 3
        mode = (self.priv, self.v)
        self.priv = to                          # (the LR reservation survives a trap:
        base = tvec & ~3                        # priv. 3.3.2 lets only xRET drop it)
        self.pc = (vector if vector is not None else
                   base + 4 * cause if interrupt and tvec & 3 == 1 else base) & self.amask
        if to == 3 and from_m and c.smrnmi and not c.mnstatus & CSR.NMIE:
            self.pc = self.rnmi_exc_vector & self.amask     # (8.5: an exception in the
            #                                                 RNMI handler)
        if self.exception_log:                  # (the entry _trap / the interrupt made:
            e = self.exception_log[-1]          # the debugger's exception line)
            e.setdefault("error", e.get("name") or f"cause {cause}")
            x = "vs" if virt else "s" if to == 1 else "m"
            why = f"{x}cause 0x{code & self.amask:x}, {x}tval 0x{tval & self.amask:x}"
            if tval2 and not virt:
                why += f", {'h' if to == 1 else 'm'}tval{'' if to == 1 else '2'} " \
                       f"0x{tval2 & self.amask:x} (GPA 0x{tval2 << 2:x})"
            e.update(addr=pc, vector=self.pc, why=why)
        if self.ingress is not None:            # (E-Trace 4.2: a trap block)
            self.ingress.trap(pc, cause, tval, interrupt, mode)

    def _take_nmi(self, pc: int) -> None:
        """An NMI (priv. 3.4: M-mode, mepc = the interrupted instruction,
        mcause 0 = "unknown" — the P8700's Table 7.6): the M-mode trap's
        MPIE / MIE / MPP stack, mipsconfig5.ERL = 1 (7.3.9: no interrupt
        until MRET / SRET), the vector = the hart's RESET_BASE (the MIPS
        way: DEVIATIONS)."""
        c = self.csrs
        self.exception_log.append({"name": INTERRUPT_NAMES[NMI], "cause": 0, "epc": pc,
                                   "priv": self.priv, "to": 3})
        c.mepc, c.mcause, c.mtval = pc & ~1, 0, 0
        p = self.p8700
        if p is not None:
            p.m_trap(False, c.status_field(CSR.MPP_SHIFT, 3))
            p.config5 |= 1                      # (ERL)
        c.set_status(mpie=c.status_bit(CSR.MIE), mie=0, mpp=self.priv)
        mode = (self.priv, self.v)
        self.priv = 3
        self.v = 0
        cl = self.cluster
        if cl is not None and p is not None:
            self.pc = cl.reset_base[cl.index(p.core, p.hart)] & self.amask
        else:
            self.pc = self.reset_pc & self.amask
        e = self.exception_log[-1]
        e.update(error=e["name"], addr=pc, vector=self.pc, why="mcause 0x0 (NMI)")
        if self.ingress is not None:
            self.ingress.trap(pc, 0, 0, True, mode)

    def _take_rnmi(self, pc: int) -> None:
        """A resumable NMI (priv. 8.5): mnepc = the interrupted PC, mncause =
        the interrupt bit and the RNMI's cause, mnstatus.MNPP / MNPV = the
        mode, NMIE = 0; M-mode at the RNMI handler's address.  mepc /
        mcause / mstatus keep their values (that is what makes it
        resumable)."""
        c = self.csrs
        cause, self.rnmi_pending = self.rnmi_pending or 0, None
        c.mnepc = pc & ~1
        c.mncause = 1 << (self.xlen - 1) | cause
        c.mnstatus = (c.mnstatus & ~(CSR.NMIE | CSR.MNPV | 3 << CSR.MNPP_SHIFT)) \
            | self.priv << CSR.MNPP_SHIFT | (CSR.MNPV if self.v and c.has_h else 0)
        if c.zicfilp:                           # (8.5: MNPELP = ELP, ELP = 0)
            c.mnstatus = (c.mnstatus & ~CSR.MNPELP) | (CSR.MNPELP if c.elp else 0)
            c.elp = 0
        self.exception_log.append({"name": INTERRUPT_NAMES[RNMI], "cause": c.mncause,
                                   "epc": pc, "priv": self.priv, "v": self.v, "to": 3})
        mode = (self.priv, self.v)
        self.priv, self.v = 3, 0
        self.pc = self.rnmi_vector & self.amask
        e = self.exception_log[-1]
        e.update(error=e["name"], addr=pc, vector=self.pc,
                 why=f"mncause 0x{c.mncause & self.amask:x}, mnepc 0x{c.mnepc:x} (NMIE = 0)")
        if self.ingress is not None:
            self.ingress.trap(pc, cause, 0, True, mode)

    # -- Debug Mode (Debug spec ch. 4) ---------------------------------------------------

    def enter_debug(self, cause: int, dpc: int | None = None) -> None:
        """4.7: dcsr.cause and prv, dpc = the next instruction (``dpc``, or
        the PC), M-mode privilege; WFI ends."""
        c = self.csrs
        dpc = self.pc if dpc is None else dpc
        hy = self.hybrid
        if hy is not None and self.debug_mode:  # (an ebreak in Debug Mode: an exception)
            hy.exception(BREAKPOINT, dpc)
            return
        self.exception_log.append({"name": "Debug Mode", "debug": cause, "epc": dpc,
                                   "priv": self.priv})
        c.dcsr = (c.dcsr & ~(7 << 6 | 3 | DCSR_V)) | cause << 6 | self.priv \
            | (DCSR_V if self.v else 0)         # (4.9.1: v, the mode's V)
        if c.zicfilp:                           # (4.9.1 / priv. 22.1.2: pelp = ELP)
            c.dcsr = (c.dcsr & ~CSR.DCSR_PELP) | (CSR.DCSR_PELP if c.elp else 0)
            c.elp = 0
        c.dpc = dpc & self.amask & ~1
        self.pc = dpc & self.amask
        self.priv = 3
        self.v = 0
        self.debug_mode = True
        self.waiting, self.wrs_until = False, None
        self._stepping = False
        if hy is not None:                      # (LF16: the handler runs — the vector)
            self.pc = hy.entered(cause)
        if self.ingress is not None:            # (E-Trace Table 4.7: halted)
            self.ingress.halted(True)
        if self.debug_module is not None:
            self.debug_module.hart_halted(self)

    def reset_hart(self) -> None:
        """A hart reset (the Debug Module's hartreset / ndmreset; priv.
        3.4): M-mode at ``reset_pc``, the CSRs, PMP, triggers and the
        TLB at their reset values, no reservation, out of Debug Mode and
        WFI.  The x / f registers keep their (UNSPECIFIED) values."""
        from .rvpmp import PMP
        self.priv = 3
        self.v = 0
        self.pc = self.reset_pc & self.amask
        self.pmp = PMP(self.xlen, self.pmp.entries)
        self.tm = Triggers(self, self.tm.count)
        self.csrs = CSR.CSRFile(self)
        self.debug_mode = self.waiting = self._stepping = self.halt_request = False
        self.wrs_until = None
        self.reservation = None
        self.dm_access = None
        if self.vector is not None:             # (30.3.11: vill, vl = 0; v0-v31 keep theirs)
            v = self.vector
            v.vstart = v.vxrm = v.vxsat = 0
            v.set_vl(0, 0, 0, self.amask)
        if self.p8700 is not None:
            self.p8700.reset()
            self.irq_lines = 0
            if hasattr(self.mmu, "config7_written"):
                self.mmu.config7_written(self)  # (FTLB64 back to 0)
        flush = getattr(self.mmu, "sfence", None)
        if flush is not None:
            flush(self, None, None)
        if self.ingress is not None:            # (E-Trace Table 4.7: reset)
            self.ingress.reset()

    def resume(self) -> None:
        """4.8 (DRET; the Debug Module's resume): PC = dpc, the mode =
        dcsr.prv (below M: MPRV = 0), out of Debug Mode; dcsr.step: one
        instruction, then Debug Mode again."""
        c = self.csrs
        self.pc = c.read(0x7B1)
        self.priv = c.dcsr & 3
        self.v = int(bool(c.dcsr & DCSR_V) and self.priv < 3)    # (4.9.1: v)
        if c.zicfilp:                           # (22.1.2: ELP = pelp if the mode's xLPE)
            c.elp = int(bool(c.dcsr & CSR.DCSR_PELP) and c.xlpe(self.priv, self.v))
            c.dcsr &= ~CSR.DCSR_PELP
        if self.priv < 3:
            c.set_status(mprv=0)
        self.debug_mode = False
        self._stepping = bool(c.dcsr & STEP)
        if self.hybrid is not None:
            self.hybrid.left()
        if self.ingress is not None:
            self.ingress.halted(False)

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
    rs1 = i.f["rs1"]
    t = (_reg(m, rs1) + i.f["imm"]) & m.amask & ~1
    if t & (m.ialign - 1):
        raise RVTrap(INSN_MISALIGNED, t)
    _set(m, i.f["rd"], pc + i.size)
    c = m.csrs                                  # (Zicfilp, Listing 4: an indirect call /
    if c.zicfilp and rs1 not in (1, 5, 7) and c.xlpe(m.priv, m.v):     # jump — JALR,
        c.elp = 1                               # C.JR, C.JALR — expects a landing pad)
    return t


@_x("lpad")
def _lpad(m, i, pc):
    """33.1.2 (Listing 5): with xLPE and ELP = LP_EXPECTED, a software-check
    exception (xtval 2) unless the PC is 4-byte aligned and the label is 0
    or x7[31:12]; then ELP = NO_LP_EXPECTED.  Else a no-op (AUIPC x0)."""
    c = m.csrs
    if c.elp and c.xlpe(m.priv, m.v):
        lpl = i.f["imm"] & 0xFFFFF000
        if pc & 3 or (lpl and (_reg(m, 7) & 0xFFFFF000) != lpl):
            raise RVTrap(SOFTWARE_CHECK, CSR.LANDING_PAD_FAULT)
        c.elp = 0


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


def misaligned(m: RVMachine, addr: int, size: int, store: bool) -> None:
    """A misaligned load / store when the hart refuses them (the P8700's
    mipsconfig7.DMALN / DULS): address misaligned (4 / 6)."""
    p = m.p8700
    if p is not None and addr % size and p.misaligned_traps:
        raise RVTrap(STORE_MISALIGNED if store else LOAD_MISALIGNED, addr)


def _load(name: str, size: int, signed: bool) -> None:
    def ex(m, i, pc):
        addr = _addr(m, i)
        misaligned(m, addr, size, False)
        v = m.load(addr, size)
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
        misaligned(m, addr, size, True)
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
    """5: later fetches see earlier stores — the D-cache's dirty lines
    written back, the I-cache invalidated."""
    if m.caches is not None:
        m.caches.fence_i()


@_x("ecall")
def _ecall(m, i, pc):
    h = m.syscall_handler
    if h is not None and h(m, _reg(m, 17)):
        return None
    raise RVTrap(ECALL_VS if m.v and m.priv == 1 else       # (21.6.1: VS = 10)
                 {0: ECALL_U, 1: ECALL_S, 3: ECALL_M}[m.priv], 0)


@_x("ebreak")
def _ebreak(m, i, pc):
    """A breakpoint exception, or Debug Mode when dcsr.ebreakm / ebreaks /
    ebreaku (VS / VU: ebreakvs / ebreakvu) has it for this mode (Debug
    spec 4.9.1)."""
    bit = {1: EBREAKVS, 0: EBREAKVU}[m.priv] if m.v else \
        {3: EBREAKM, 1: EBREAKS, 0: EBREAKU}[m.priv]
    if m.csrs.dcsr & bit:
        raise EnterDebug(CAUSE_EBREAK)
    raise RVTrap(BREAKPOINT, pc)


@_x("mret")
def _mret(m, i, pc):
    """priv. 3.3.2: MIE = MPIE, MPIE = 1, priv = MPP, MPP = the least
    privileged mode; MPRV = 0 when MPP is not M; V = MPV unless MPP = M,
    MPV = 0 (21.6.4)."""
    if m.priv != 3:                             # (not HS-qualified: illegal, 21.6.1)
        raise RVTrap(ILLEGAL, i.raw)
    c = m.csrs
    mpp = c.status_field(CSR.MPP_SHIFT, 3)
    mpv = c.mstatus >> CSR.MPV & 1
    c.set_status(mie=c.status_bit(CSR.MPIE), mpie=1, mpp=c.lowest_priv)
    if mpp != 3:
        c.set_status(mprv=0)
    c.mstatus &= ~(1 << CSR.MPV)
    m.priv = mpp
    m.v = mpv if mpp != 3 else 0
    if c.zicfilp:                               # (22.1.2: ELP = MPELP if yLPE; MPELP = 0)
        c.elp = (c.mstatus >> CSR.MPELP & 1) if c.xlpe(m.priv, m.v) else 0
        c.mstatus &= ~(1 << CSR.MPELP)
    if m.p8700 is not None and m.p8700.mret():  # (MD01502 7.3.9: ERL, MTW, MPPPS)
        c.set_status(mpp=1)
    tm = m.tm
    if tm.count and not c.has_s:                # Debug spec 5.7.6: MTE = MPTE
        tm.tcontrol = MPTE | MTE if tm.tcontrol & MPTE else 0
    return c.read(0x341)


@_x("mnret")
def _mnret(m, i, pc):
    """Smrnmi 8.4: M-mode only; PC = mnepc, the mode = mnstatus.MNPP (V =
    MNPV), MPRV = 0 when that is below M, NMIE = 1."""
    if m.priv != 3:
        raise RVTrap(ILLEGAL, i.raw)
    c = m.csrs
    s = c.mnstatus
    prv, virt = (s >> CSR.MNPP_SHIFT) & 3, int(bool(s & CSR.MNPV))
    if prv != 3:
        c.set_status(mprv=0)
    c.mnstatus = s | CSR.NMIE
    m.priv = prv
    m.v = virt if prv != 3 else 0
    if c.zicfilp:                               # (8.4: ELP = MNPELP AND yLPE)
        c.elp = int(bool(s & CSR.MNPELP) and c.xlpe(m.priv, m.v))
        c.mnstatus &= ~CSR.MNPELP
    return c.read(0x741)


@_x("sret")
def _sret(m, i, pc):
    """priv. 3.3.2 / 10.1.1: S-mode or M-mode only, not in S-mode with
    TSR; SIE = SPIE, SPIE = 1, priv = SPP, SPP = U, MPRV = 0; with H
    (21.6.4) V = hstatus.SPV, SPV = 0 — and in VS-mode (a virtual-
    instruction exception in VU or with VTSR) vsstatus's stack, vsepc."""
    c = m.csrs
    if m.v:
        if m.priv == 0 or c.hstatus >> CSR.H_VTSR & 1:
            raise RVTrap(VIRTUAL, i.raw)
        epc = c.read(0x141)                     # (vsepc)
        vs = c.vsstatus
        spp = vs >> CSR.SPP & 1
        vs &= ~((1 << CSR.SIE) | (1 << CSR.SPIE) | (1 << CSR.SPP))
        c.vsstatus = vs | (c.vsstatus >> CSR.SPIE & 1) << CSR.SIE | 1 << CSR.SPIE
        m.priv = spp
        if c.zicfilp:                           # (22.1.2: the VS-level SPELP)
            c.elp = (c.vsstatus >> CSR.SPELP & 1) if c.xlpe(m.priv, 1) else 0
            c.vsstatus &= ~(1 << CSR.SPELP)
        return epc
    if not c.has_s or m.priv < 1 or (m.priv == 1 and c.status_bit(CSR.TSR)):
        raise RVTrap(ILLEGAL, i.raw)
    epc = c.read(0x141)
    spp = c.status_bit(CSR.SPP)
    c.set_status(sie=c.status_bit(CSR.SPIE), spie=1, spp=0, mprv=0)
    m.priv = spp
    if c.has_h:
        m.v = c.hstatus >> CSR.H_SPV & 1
        c.hstatus &= ~(1 << CSR.H_SPV)
    if c.zicfilp:                               # (22.1.2: ELP = SPELP if yLPE; SPELP = 0)
        c.elp = (c.mstatus >> CSR.SPELP & 1) if c.xlpe(m.priv, m.v) else 0
        c.mstatus &= ~(1 << CSR.SPELP)
    if m.p8700 is not None:
        m.p8700.config5 &= ~1
    return epc


@_x("wfi")
def _wfi(m, i, pc):
    """priv. 3.3.3: wait until an interrupt is pending and enabled in mie
    (whatever the global enables).  Below M-mode with TW = 1, or in U-mode
    when S-mode exists, an illegal instruction (eMIPS's time limit is
    0, as Spike's); with V = 1 and TW = 0, in VU-mode or VS-mode with
    hstatus.VTW, a virtual-instruction exception (21.2.1 / 21.6.1)."""
    c = m.csrs
    if m.priv < 3 and c.status_bit(CSR.TW):
        raise RVTrap(ILLEGAL, i.raw)
    if m.v:
        if m.priv == 0 or c.hstatus >> CSR.H_VTW & 1:
            raise RVTrap(VIRTUAL, i.raw)
    elif m.priv == 0 and c.has_s:
        raise RVTrap(ILLEGAL, i.raw)
    if not c.mip() & c.mie and not m._stepping:     # (stepped: a nop, Debug spec 4.5.1)
        m.waiting = True


@_x("sfence.vma")
def _sfence_vma(m, i, pc):
    """priv. 10.2.1: S-mode or M-mode; in S-mode with TVM, illegal (the
    translation caches it flushes: phase 2b); with V = 1 (21.5.3): VS-mode
    — VU-mode or hstatus.VTVM: a virtual-instruction exception — and the
    current VM's VS-level translations only."""
    c = m.csrs
    if m.v:
        if m.priv == 0 or c.hstatus >> CSR.H_VTVM & 1:
            raise RVTrap(VIRTUAL, i.raw)
    elif not c.has_s or m.priv < 1 or (m.priv == 1 and c.status_bit(CSR.TVM)):
        raise RVTrap(ILLEGAL, i.raw)
    flush = getattr(m.mmu, "sfence", None)
    if flush is not None:
        flush(m, _reg(m, i.f["rs1"]) if i.f["rs1"] else None,
              _reg(m, i.f["rs2"]) if i.f["rs2"] else None)


# -- the hypervisor extension (priv. 21.3) -------------------------------------------

def _hyp_check(m: RVMachine, i: RV, hu: bool) -> None:
    """21.3.1 / 21.3.2: with V = 1 a virtual-instruction exception; in
    U-mode illegal (HLV / HLVX / HSV: unless hstatus.HU)."""
    if m.v:
        raise RVTrap(VIRTUAL, i.raw)
    if m.priv == 0 and not (hu and m.csrs.hstatus >> CSR.H_HU & 1):
        raise RVTrap(ILLEGAL, i.raw)


def _hlv(name: str, size: int, signed: bool, x: bool = False) -> None:
    """HLV.width / HLVX.HU / HLVX.WU (21.3.1): a load as VS / VU-mode would
    make it (the privilege hstatus.SPVP, two-stage translation, vsstatus's
    SUM; HLVX: execute permission instead of read) — ``hyp_access``."""
    def ex(m, i, pc):
        _hyp_check(m, i, True)
        m.hyp_access = "x" if x else "v"
        try:
            v = m.load(_reg(m, i.f["rs1"]), size)
        finally:
            m.hyp_access = None
        if signed and v >> (8 * size - 1):
            v -= 1 << (8 * size)
        _set(m, i.f["rd"], v)
    EXECUTORS[name] = ex


def _hsv(name: str, size: int) -> None:
    def ex(m, i, pc):
        _hyp_check(m, i, True)
        addr = _reg(m, i.f["rs1"])
        m.hyp_access = "v"
        try:
            m.store(addr, _reg(m, i.f["rs2"]), size)
        finally:
            m.hyp_access = None
        _clear_reservation(m, addr, size)
    EXECUTORS[name] = ex


for _n, _sz, _sg in (("hlv.b", 1, True), ("hlv.h", 2, True), ("hlv.w", 4, True),
                     ("hlv.d", 8, True), ("hlv.bu", 1, False), ("hlv.hu", 2, False),
                     ("hlv.wu", 4, False)):
    _hlv(_n, _sz, _sg)
_hlv("hlvx.hu", 2, False, x=True)
_hlv("hlvx.wu", 4, False, x=True)
for _n, _sz in (("hsv.b", 1), ("hsv.h", 2), ("hsv.w", 4), ("hsv.d", 8)):
    _hsv(_n, _sz)


@_x("hfence.vvma")
def _hfence_vvma(m, i, pc):
    """21.3.2: M or HS-mode; the VS-level translations of the current VMID
    (rs1 a guest virtual address, rs2 an ASID; x0 = all)."""
    _hyp_check(m, i, False)
    flush = getattr(m.mmu, "hfence_vvma", None)
    if flush is not None:
        flush(m, _reg(m, i.f["rs1"]) if i.f["rs1"] else None,
              _reg(m, i.f["rs2"]) if i.f["rs2"] else None)


@_x("hfence.gvma")
def _hfence_gvma(m, i, pc):
    """21.3.2: M-mode, or HS-mode with mstatus.TVM = 0; the guest-physical
    translations (rs1 a guest physical address >> 2, rs2 a VMID)."""
    _hyp_check(m, i, False)
    if m.priv == 1 and m.csrs.status_bit(CSR.TVM):
        raise RVTrap(ILLEGAL, i.raw)
    flush = getattr(m.mmu, "hfence_gvma", None)
    if flush is not None:
        flush(m, _reg(m, i.f["rs1"]) << 2 if i.f["rs1"] else None,
              _reg(m, i.f["rs2"]) if i.f["rs2"] else None)


# -- Svinval (priv. ch. 15) -----------------------------------------------------------

EXECUTORS["sinval.vma"] = _sfence_vma           # (15: "simpler implementations" — the
EXECUTORS["hinval.vvma"] = _hfence_vvma         # same invalidations, the same traps)
EXECUTORS["hinval.gvma"] = _hfence_gvma


def _sfence_inval(m, i, pc):
    """SFENCE.W.INVAL / SFENCE.INVAL.IR (15): ordering only — a no-op on
    eMIPS's in-order hart; U-mode illegal, VU-mode virtual; never TVM /
    VTVM."""
    if m.v and m.priv == 0:
        raise RVTrap(VIRTUAL, i.raw)
    if m.priv == 0:
        raise RVTrap(ILLEGAL, i.raw)


EXECUTORS["sfence.w.inval"] = EXECUTORS["sfence.inval.ir"] = _sfence_inval


# -- the cache-block operations (unpriv. ch. 19) --------------------------------------

#: the cache block of the CMOs (19.3.1: uniform; Spike's --blocksz default)
CBO_BLOCK = 64
#: a management CBO's load-type fault becomes the store-type one (19.4.2.2)
_TO_STORE = {LOAD_ACCESS: STORE_ACCESS, 13: 15, LOAD_GUEST_PAGE: STORE_GUEST_PAGE}


def _cbo_enable(m: RVMachine, i: RV, shift: int, mask: int) -> tuple[int, int, int]:
    """19.5's rules for xenvcfg's field at ``shift``: below M menvcfg's, in
    U senvcfg's (illegal), in VS / VU henvcfg's and in VU senvcfg's
    (virtual); returns (m, s, h) fields."""
    c = m.csrs
    mf, sf, hf = (c.menvcfg >> shift) & mask, (c.senvcfg >> shift) & mask, \
        (c.henvcfg >> shift) & mask
    if (m.priv != 3 and not mf) or (c.has_s and m.priv == 0 and not m.v and not sf):
        raise RVTrap(ILLEGAL, i.raw)
    if m.v and (not hf or (m.priv == 0 and not sf)):
        raise RVTrap(VIRTUAL, i.raw)
    return mf, sf, hf


def _cbo_target(m: RVMachine, i: RV, kind: str) -> tuple[int, int, int, int]:
    """The block of a CBO: (rs1, its block's VA, PA, CCA) — translated as a
    load (management: A checked or set, not D) or a store (zero), the
    faults store-type with tval = rs1 (19.4.2.2); PMP and the PMAs on the
    whole block (a management CBO: a load or a store allowed; zero: a
    store; an I/O region: a store access fault); the store triggers on the
    block (19.4.2.4)."""
    ea = _reg(m, i.f["rs1"])
    base = ea & ~(CBO_BLOCK - 1) & m.amask
    if m.tm.stores:
        m.tm.memory("store", base, CBO_BLOCK)
    m.cbo_access = True                         # (22.2.3: no CBO on a shadow-stack page)
    try:
        pa, cca = m._xlate(m, base, kind)
    except RVTrap as t:
        raise RVTrap(_TO_STORE.get(t.cause, t.cause), ea, t.custom, tval2=t.tval2,
                     tinst=t.tinst, gva=t.gva) from None
    finally:
        m.cbo_access = False
    priv, virt = m.mmu.effective(m, kind) if hasattr(m.mmu, "effective") else (m.priv, m.v)
    p = m.pmp
    if p.entries and not (p.allows(pa, CBO_BLOCK, "store", priv) or
                          kind == "load" and p.allows(pa, CBO_BLOCK, "load", priv)):
        raise RVTrap(STORE_ACCESS, ea, gva=bool(virt))
    if pma_cca(m, pa) == 2:                     # (an I/O region: not main memory)
        raise RVTrap(STORE_ACCESS, ea, gva=bool(virt))
    return ea, base, pa, cca


def _cbo_manage(op: str):
    def ex(m, i, pc):
        if op == "inval":
            mf, sf, hf = _cbo_enable(m, i, 4, 3)
            if (m.priv != 3 and mf == 1) or (not m.v and m.priv == 0 and m.csrs.has_s
                                             and sf == 1) or \
                    (m.v and (hf == 1 or (m.priv == 0 and sf == 1))):
                kind = "flush"                  # (19.5: CBIE = 01 makes it a flush)
            else:
                kind = "inval"
        else:
            _cbo_enable(m, i, 6, 1)
            kind = op
        ea, base, pa, cca = _cbo_target(m, i, "load")
        m.cmo_access = (base, pa, cca)
        done = m.caches.cbo(kind, base, pa, CBO_BLOCK, cca) if m.caches is not None else \
            f"cbo.{kind}: no caches"
        m.cmo_note = ("as a flush (xenvcfg.CBIE = 01): " if kind != op else "") + done
        cl = m.cluster
        if cl is not None:
            cl.stored(m, pa)                    # (19.4.3: the others' reservations)
    return ex


EXECUTORS["cbo.clean"] = _cbo_manage("clean")
EXECUTORS["cbo.flush"] = _cbo_manage("flush")
EXECUTORS["cbo.inval"] = _cbo_manage("inval")


@_x("cbo.zero")
def _cbo_zero(m, i, pc):
    """CBO.ZERO (19.6.2): the block's bytes are zeroed by stores — a
    write-back D-cache allocates its lines zeroed, else doublewords /
    words are stored (the store triggers already looked at the block)."""
    _cbo_enable(m, i, 7, 1)
    ea, base, pa, cca = _cbo_target(m, i, "store")
    m.cmo_access = (base, pa, cca)
    c = m.caches
    if c is not None and c.cacheable and cca in m.cached_cca and \
            c.cbo_zero(base, pa, CBO_BLOCK, cca):
        how = "the D-cache's lines allocated zeroed"
    else:
        step = 8 if m.xlen == 64 else 4
        for k in range(0, CBO_BLOCK, step):
            S.Machine.store(m, base + k, 0, step)
        how = f"{CBO_BLOCK // step} stores of zero"
    m.cmo_note = f"cbo.zero: {how}"
    cl = m.cluster
    if cl is not None:
        cl.stored(m, pa)


_ORI = EXECUTORS["ori"]
#: Zicbop's hints (19.6.3): ORI x0, rs1, imm with imm[4:0] = 0 / 1 / 3
_PREFETCH = {0: "fetch", 1: "load", 3: "store"}


@_x("ori")
def _ori(m, i, pc):
    """ORI — and with Zicbop and rd = x0 its PREFETCH.I / .R / .W hints
    (19.6.3): the block at rs1 + imm[11:5] << 5 brought into the I- / D-
    cache when a fetch / load / store may access it (never a fault, the A
    / D bits neither checked nor set: a view's translation)."""
    f = i.f
    if f["rd"] == 0 and "zicbop" in m.extensions and f["imm"] & 0x1F in _PREFETCH \
            and m.caches is not None:
        kind = _PREFETCH[f["imm"] & 0x1F]
        va = (_reg(m, f["rs1"]) + (f["imm"] & ~0x1F)) & m.amask & ~(CBO_BLOCK - 1)
        t = m.mmu.describe(m, va, kind)
        if t.pa is not None and t.cca in m.cached_cca:
            priv = m.mmu.effective(m, kind)[0] if hasattr(m.mmu, "effective") else m.priv
            if m.pmp.allows(t.pa, 1, kind, priv):
                c = m.caches
                if kind == "fetch":             # (a hit refreshes the line, a miss fills)
                    if c.i is not None:
                        if c.cm is not None:
                            c.cm.claim(c)
                        c.timing.begin("pref")
                        c._lookup(c.i, va, t.pa, "pref", t.cca, None)
                else:
                    c.prefetch(0 if kind == "load" else 1, va, t.pa, t.cca)
    return _ORI(m, i, pc)


def _dret(m, i, pc):
    """DRET: illegal outside Debug Mode (Debug spec 4.1; a halted hart's
    runs through execute_debug); the P8700's executed Debug Mode resumes
    (MIPS DERET's role, LF16)."""
    if not m.debug_mode or m.hybrid is None:
        raise RVTrap(ILLEGAL, i.raw)
    m.resume()
    if m._stepping:                             # (dcsr.step: the next instruction, then
        m._stepping, m._step_next = False, True  # Debug Mode — not this DRET)
    return m.pc


EXECUTORS["dret"] = _dret
#: (DRET outside Debug Mode: illegal — Debug spec 4.1; in it: execute_debug)
#: what Debug Mode refuses (Debug spec 4.1 rules 9-12: eMIPS's choice)
_DEBUG_ILLEGAL = frozenset({"jal", "jalr", "beq", "bne", "blt", "bge", "bltu", "bgeu",
                            "auipc", "ecall", "mret", "sret", "mnret"})


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


# -- M (12) --------------------------------------------------------------------

def _div(a: int, b: int, bits: int, signed: bool, rem: bool) -> int:
    """12.2: division by zero and overflow (no trap; Table 11)."""
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


# -- A (13) ----------------------------------------------------------------------

def _clear_reservation(m: RVMachine, addr: int, size: int) -> None:
    """A store of this hart to the reserved bytes: the reservation stays
    (13.2 allows it); other harts' stores clear it (the CPS, phase 5)."""


def _aligned(m: RVMachine, addr: int, size: int, store: bool) -> None:
    if addr % size:                             # (GVA: the access's V — MPRV with MPV)
        virt = m.mmu.effective(m, "store" if store else "load")[1] \
            if hasattr(m.mmu, "effective") else 0
        raise RVTrap(STORE_MISALIGNED if store else LOAD_MISALIGNED, addr, gva=bool(virt))


def _lr(size: int):
    def ex(m, i, pc):
        addr = _reg(m, i.f["rs1"])
        m.trigger_check("load", addr, size)     # (Debug spec Table 13: before alignment)
        _aligned(m, addr, size, False)
        v = m.load(addr, size, pre=False)
        m.reservation = (addr, size)
        if m.cluster is not None:
            pa = m.translate(addr).pa
            m.lr_granule = None if pa is None else pa >> 6
        _set(m, i.f["rd"], _s32(v) if size == 4 else v)
    return ex


def _sc(size: int):
    def ex(m, i, pc):
        addr = _reg(m, i.f["rs1"])
        _aligned(m, addr, size, True)
        ok = m.reservation == (addr, size)
        m.reservation = None                    # 13.2: SC invalidates it either way
        if ok:
            m.store(addr, _reg(m, i.f["rs2"]), size, atomic=True)
        else:                                   # 13.2: "no SC.W shall retire unless it
            m._access(addr, size, "store")      # passes memory permission checks"
        _set(m, i.f["rd"], 0 if ok else 1)
    return ex


EXECUTORS.update({"lr.w": _lr(4), "lr.d": _lr(8), "sc.w": _sc(4), "sc.d": _sc(8)})


def _amo_start(m: RVMachine, i: RV, size: int) -> int:
    """An AMO's address after the checks every AMO makes first (the store
    trigger, natural alignment; the P8700's emulation trap)."""
    addr = _reg(m, i.f["rs1"])
    m.trigger_check("store", addr, size)        # (Spike: the store's address first)
    _aligned(m, addr, size, True)               # 13.4: store / AMO misaligned
    if m.p8700 is not None:                     # no AMOs: emulated (MD01502 7.3.10)
        m._access(addr, min(size, 8), "store")
        m.p8700.amo(i.raw)
    return addr


def _amocas(size: int):
    """AMOCAS.B / H / W / D / Q (15.1, 16.1): load, compare with rd, store
    rs2 when equal (no write when not: 15.1 allows either), rd = the old
    value (sign-extended); a 2·XLEN operand is an even / odd register pair
    (x0: zero, not written).  Write permission always (a load of kind
    store)."""
    def ex(m, i, pc):
        addr = _amo_start(m, i, size)
        half = m.xlen // 8
        f = i.f
        pair = size > half

        def value(n):
            if pair:
                return 0 if n == 0 else _reg(m, n) | _reg(m, n + 1) << (8 * half)
            return _reg(m, n) & ((1 << 8 * size) - 1)
        comp, swap = value(f["rd"]), value(f["rs2"])
        if pair:
            m._access((addr + half) & m.amask, half, "store")    # (both halves writable)
            old = m.load(addr, half, "store") | m.load((addr + half) & m.amask, half,
                                                        "store") << (8 * half)
        else:
            old = m.load(addr, size, "store")
        if old == comp:
            if pair:
                m.store(addr, swap & ((1 << 8 * half) - 1), half, atomic=True)
                m.store((addr + half) & m.amask, swap >> (8 * half), half, atomic=True)
            else:
                m.store(addr, swap, size, atomic=True)
        if pair:
            if f["rd"]:
                _set(m, f["rd"], old & ((1 << 8 * half) - 1))
                _set(m, f["rd"] + 1, old >> (8 * half))
        else:
            _set(m, f["rd"], old if size == half else _sb(old, 8 * size))
    return ex


for _n, _sz in (("amocas.b", 1), ("amocas.h", 2), ("amocas.w", 4), ("amocas.d", 8),
                ("amocas.q", 16)):
    EXECUTORS[_n] = _amocas(_sz)


def _amo(name: str, size: int, op: Callable[[int, int, int], int]) -> None:
    bits = 8 * size

    def ex(m, i, pc):
        addr = _amo_start(m, i, size)
        old = m.load(addr, size, "store")
        new = op(old, _reg(m, i.f["rs2"]) & ((1 << bits) - 1), bits)
        m.store(addr, new, size, atomic=True)
        _set(m, i.f["rd"], old if size == m.xlen // 8 else _sb(old, bits))   # (sign-extended)
    EXECUTORS[name] = ex


def _sb(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


for _sfx, _size in (("b", 1), ("h", 2), ("w", 4), ("d", 8)):     # (b / h: Zabha, 16.1)
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


# -- Zbs (29.4.4: single-bit instructions; index = rs2 / shamt mod XLEN) ---------------

for _n, _fn in (("bclr", lambda m, a, b: a & ~(1 << (b & (m.xlen - 1)))),
                ("bext", lambda m, a, b: (a >> (b & (m.xlen - 1))) & 1),
                ("binv", lambda m, a, b: a ^ (1 << (b & (m.xlen - 1)))),
                ("bset", lambda m, a, b: a | (1 << (b & (m.xlen - 1))))):
    _alu(_n, _fn)
    _alu(_n + "i", _fn, imm=True)


# -- Zbc (29.4.3) and Zbkc (29.4.6): carry-less multiplication --------------------------

def _clmul(a: int, b: int) -> int:
    """The 2·XLEN-bit carry-less product of a and b."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


for _n, _fn in (("clmul", lambda m, a, b: _clmul(a, b)),
                ("clmulh", lambda m, a, b: _clmul(a, b) >> m.xlen),
                ("clmulr", lambda m, a, b: _clmul(a, b) >> (m.xlen - 1))):
    _alu(_n, _fn)


# -- Zbkb / Zbkx (29.4.5, 29.4.7): the crypto bit manipulations -------------------------

def _pack(m: RVMachine, a: int, b: int) -> int:
    half = m.xlen // 2
    mask = (1 << half) - 1
    return (a & mask) | (b & mask) << half


def _brev8(m: RVMachine, v: int) -> int:
    out = 0
    for k in range(m.xlen // 8):
        byte = (v >> (8 * k)) & 0xFF
        out |= int(f"{byte:08b}"[::-1], 2) << (8 * k)
    return out


def _zip(m: RVMachine, v: int) -> int:
    """RV32 zip: bit i of the low half -> bit 2i, of the high half -> 2i + 1."""
    out = 0
    for k in range(16):
        out |= ((v >> k) & 1) << (2 * k) | ((v >> (k + 16)) & 1) << (2 * k + 1)
    return out


def _unzip(m: RVMachine, v: int) -> int:
    """RV32 unzip: the even bits -> the low half, the odd ones -> the high."""
    out = 0
    for k in range(16):
        out |= ((v >> (2 * k)) & 1) << k | ((v >> (2 * k + 1)) & 1) << (k + 16)
    return out


def _xperm(bits: int):
    """xperm4 / xperm8: each element of rs2 selects an element of rs1
    (out of range: 0)."""
    def op(m: RVMachine, a: int, b: int) -> int:
        n, mask = m.xlen // bits, (1 << bits) - 1
        out = 0
        for k in range(n):
            sel = (b >> (bits * k)) & mask
            if sel < n:
                out |= ((a >> (bits * sel)) & mask) << (bits * k)
        return out
    return op


_alu("pack", _pack)
_alu("packh", lambda m, a, b: (a & 0xFF) | (b & 0xFF) << 8)
_alu("packw", lambda m, a, b: (a & 0xFFFF) | (b & 0xFFFF) << 16, word=True)
_alu("xperm4", _xperm(4))
_alu("xperm8", _xperm(8))
for _n, _fn in (("brev8", _brev8), ("zip", _zip), ("unzip", _unzip)):
    _unary(_n, _fn)


# -- Zicond (11): conditional zero -------------------------------------------------

_alu("czero.eqz", lambda m, a, b: 0 if b == 0 else a)
_alu("czero.nez", lambda m, a, b: 0 if b != 0 else a)


# -- Zcmp / Zcmt (28.13 / 28.14): push / pop, moves, table jumps ---------------------

def _push_pop(kind: str):
    """CM.PUSH / CM.POP / CM.POPRET / CM.POPRETZ (28.13.7-28.13.10): the
    registers (s11 ... s0, ra) stored below sp / loaded from the top of the
    frame, then — only when every access succeeded — sp moved, a0 = 0
    (POPRETZ) and ret (POPRET / POPRETZ)."""
    def ex(m, i, pc):
        bytes_ = m.xlen // 8
        regs = R.rlist_regs(i.f["rlist"])
        adj = R.sp_base(i.f["rlist"], m.xlen) + 16 * i.f["spimm"]
        sp = _reg(m, 2)
        if kind == "push":
            addr = sp
            for r in reversed(regs):            # (x27 ... x8, x1: the highest first)
                addr = (addr - bytes_) & m.amask
                m.store(addr, _reg(m, r), bytes_)
            _set(m, 2, sp - adj)
            return None
        addr = sp + adj
        values = []
        for r in reversed(regs):
            addr -= bytes_
            v = m.load(addr & m.amask, bytes_)
            values.append((r, _s32(v) if bytes_ == 4 else v))
        for r, v in values:                     # (a fault above left the registers as they were)
            _set(m, r, v)
        if kind == "popretz":
            _set(m, 10, 0)
        _set(m, 2, sp + adj)
        if kind != "pop":
            return _reg(m, 1) & ~1 & m.amask
        return None
    return ex


for _n in ("push", "pop", "popret", "popretz"):
    EXECUTORS[f"cm.{_n}"] = _push_pop(_n)


@_x("cm.mvsa01")
def _cm_mvsa01(m, i, pc):
    """28.13.11: a0 -> the first s register, a1 -> the second."""
    a0, a1 = _reg(m, 10), _reg(m, 11)
    _set(m, R.sreg(i.f["sreg1"]), a0)
    _set(m, R.sreg(i.f["sreg2"]), a1)


@_x("cm.mva01s")
def _cm_mva01s(m, i, pc):
    """28.13.12: the two s registers -> a0, a1."""
    s1, s2 = _reg(m, R.sreg(i.f["sreg1"])), _reg(m, R.sreg(i.f["sreg2"]))
    _set(m, 10, s1)
    _set(m, 11, s2)


@_x("cm.jalt")
def _cm_jalt(m, i, pc):
    """CM.JT (index < 32) / CM.JALT (28.14.4 / 28.14.5): the target is the
    XLEN-bit entry jvt.base + index x XLEN/8, read as an instruction fetch
    (execute permission; a fault names its address, epc = this
    instruction); CM.JALT links ra."""
    index = i.f["index"]
    base = (m.csrs.jvt & ~0x3F) + index * (m.xlen // 8)
    target = 0
    for k in range(m.xlen // 16):                # (little-endian parcels)
        target |= m._half((base + 2 * k) & m.amask) << (16 * k)
    if index >= 32:
        _set(m, 1, pc + 2)
    return target & ~1 & m.amask


# -- Zimop / Zcmop (10): may-be-operations ------------------------------------------
# MOP.R.n / MOP.RR.n write 0 to rd (they read nothing: "no syntactic
# dependency"); C.MOP.n runs its expansion, an addi to x0 (riscv.EXPAND).

for _n in ("mop.r.N", "mop.rr.N"):
    EXECUTORS[_n] = lambda m, i, pc: _set(m, i.f["rd"], 0)


# -- Zicfiss (unpriv. 33.2, priv. 22.2): the shadow stack -----------------------------
# SSPUSH / SSPOPCHK / SSRDP (and C.SSPUSH / C.SSPOPCHK, their expansions) are
# the MOPs they are encoded as while xSSE = 0: nothing, or rd = 0 for SSRDP.

def _ss_access(m: RVMachine, fn):
    """A shadow-stack access (emips.rvmmu's ssload / ssstore: store faults,
    a shadow-stack page, a store's PMP, idempotent memory)."""
    m.ss_access = True
    try:
        return fn()
    finally:
        m.ss_access = False


def _ss_address(m: RVMachine, addr: int, size: int) -> int:
    """22.2.3: an ssp that is not XLEN-aligned is a store access fault."""
    if addr % size:
        raise RVTrap(STORE_ACCESS, addr)
    return addr


def _sspush(m, i, pc):
    """33.2.4 (Listing 6): ssp - XLEN/8 = the link register, then ssp
    decremented (only once the store has completed)."""
    c = m.csrs
    if not c.xsse():
        return None
    size = m.xlen // 8
    addr = _ss_address(m, (c.ssp - size) & m.amask, size)
    value = _reg(m, i.f["rs2"])
    _ss_access(m, lambda: m.store(addr, value, size))
    c.ssp = addr
    return None


def _sspopchk(m, i, pc):
    """33.2.5 (Listing 7): load from ssp, compare with the link register —
    unequal: a software-check exception (xtval 3), ssp unchanged — else
    ssp incremented."""
    c = m.csrs
    if not c.xsse():
        return None
    size = m.xlen // 8
    addr = _ss_address(m, c.ssp & m.amask, size)
    value = _ss_access(m, lambda: m.load(addr, size))
    if value != _reg(m, i.f["rs1"]):
        raise RVTrap(SOFTWARE_CHECK, CSR.SHADOW_STACK_FAULT)
    c.ssp = (addr + size) & m.amask
    return None


for _n in ("sspush.x1", "sspush.x5"):
    EXECUTORS[_n] = _sspush
for _n in ("sspopchk.x1", "sspopchk.x5"):
    EXECUTORS[_n] = _sspopchk


@_x("ssrdp")
def _ssrdp(m, i, pc):
    """33.2.6 (Listing 8): rd = ssp, or 0 (MOP.R.28) while xSSE = 0."""
    c = m.csrs
    _set(m, i.f["rd"], c.ssp if c.xsse() else 0)


def _ssamoswap(size: int):
    def ex(m, i, pc):
        """33.2.7 (Listings 9 / 10): illegal below M without menvcfg.SSE,
        without S-mode, in U-mode without senvcfg.SSE; virtual in VS-mode
        without henvcfg.SSE, in VU-mode without senvcfg.SSE (which then
        reads 0 also without henvcfg.SSE); then an atomic swap on a
        shadow-stack page (22.2.3) — rd = the old value (.W on RV64:
        sign-extended).  Misaligned: a store access fault (Spike's choice of
        the AMO "exception options")."""
        c = m.csrs
        priv, v = m.priv, m.v
        if priv != 3 and not c.menvcfg & CSR.SSE:
            raise RVTrap(ILLEGAL, i.raw)
        if not c.has_s:
            raise RVTrap(ILLEGAL, i.raw)
        if priv == 0 and not v and not c.senv() & CSR.SSE:
            raise RVTrap(ILLEGAL, i.raw)
        if v and priv == 1 and not c.henv() & CSR.SSE:
            raise RVTrap(VIRTUAL, i.raw)
        if v and priv == 0 and not c.senv() & CSR.SSE:
            raise RVTrap(VIRTUAL, i.raw)
        addr = _reg(m, i.f["rs1"])
        m.trigger_check("store", addr, size)
        if addr % size:
            raise RVTrap(STORE_ACCESS, addr)
        swap = _reg(m, i.f["rs2"])

        def go():
            old = m.load(addr, size, "store")
            m.store(addr, swap, size, atomic=True)
            return old
        old = _ss_access(m, go)
        _set(m, i.f["rd"], old if size == m.xlen // 8 else _sb(old, 8 * size))
    return ex


EXECUTORS["ssamoswap.w"] = _ssamoswap(4)
EXECUTORS["ssamoswap.d"] = _ssamoswap(8)


# -- Zawrs (14): wait on the reservation set ----------------------------------------

#: WRS.STO's "short" timeout in cycles ("10 to 100 times an on-chip cache
#: miss latency": 10 x emips.timing's 50-cycle memory access)
WRS_STO_CYCLES = 500


def _wrs(short: bool):
    def ex(m, i, pc):
        """14.1: stall while the reservation set is valid and no interrupt
        is pending (WFI's rule: mip & mie); WRS.STO for at most
        WRS_STO_CYCLES.  Below M-mode with mstatus.TW = 1, WRS.NTO is an
        illegal instruction (eMIPS's time limit is 0, as for WFI and
        Spike).  Stepped or in Debug Mode: no stall."""
        c = m.csrs
        if not short and m.priv < 3 and c.status_bit(CSR.TW):
            raise RVTrap(ILLEGAL, i.raw)
        if m.reservation is None or m._stepping or m.debug_mode or c.mip() & c.mie:
            return
        m.waiting = True
        m.wrs_until = m.cycles + WRS_STO_CYCLES if short else -1
    return ex


EXECUTORS["wrs.nto"] = _wrs(False)
EXECUTORS["wrs.sto"] = _wrs(True)


# -- the P8700's custom instructions (MD01502 Appendix B) -----------------------
# binutils 2.45's vendor extensions Xmipscmov / Xmipslsp / Xmipscbop /
# Xmipsexectl: a hart without them decodes none of these (its HART.udi).

@_x("mips.ccmov")
def _mips_ccmov(m, i, pc):
    """CCMOV: rd = rs1 if rs2 != 0 else rs3."""
    f = i.f
    _set(m, f["rd"], _reg(m, f["rs1"]) if _reg(m, f["rs2"]) else _reg(m, f["rs3"]))


def _pair_load(size: int):
    def ex(m, i, pc):
        """LDP / LWP: two consecutive doublewords / words from rs1 + offset
        (an unsigned offset; misaligned pairs are done in hardware, the
        P8700's choice — CONFIG.pair_aligned 0); rd gets the lower one, rd2
        the upper (little-endian); words are sign-extended."""
        va = (_reg(m, i.f["rs1"]) + i.f["imm"]) & m.amask
        misaligned(m, va, size, False)
        lo = m.load(va, size)
        hi = m.load((va + size) & m.amask, size)
        if size == 4:
            lo, hi = _sb(lo, 32), _sb(hi, 32)
        _set(m, i.f["rd"], lo)                  # (rd = rd2: UNKNOWN — the upper one stays)
        _set(m, i.f["rd2"], hi)
    return ex


def _pair_store(size: int):
    def ex(m, i, pc):
        """SDP / SWP: rs2 to rs1 + offset, rs3 to the next doubleword / word."""
        va = (_reg(m, i.f["rs1"]) + i.f["imm"]) & m.amask
        misaligned(m, va, size, True)
        m.store(va, _reg(m, i.f["rs2"]), size)
        m.store((va + size) & m.amask, _reg(m, i.f["rs3"]), size)
        _clear_reservation(m, va, 2 * size)
    return ex


EXECUTORS["mips.ldp"] = _pair_load(8)
EXECUTORS["mips.lwp"] = _pair_load(4)
EXECUTORS["mips.sdp"] = _pair_store(8)
EXECUTORS["mips.swp"] = _pair_store(4)

#: PREF's prefetch types into the D-cache -> emips.cache's PREF hints (load,
#: store, prepare for store: a zeroed line on a miss)
_PREF_TYPES = {0: 0, 1: 1, 7: 30}


@_x("mips.pref")
def _mips_pref(m, i, pc):
    """PREF hint, imm(rs1): never an exception (a fault: nothing done);
    hint[4:3] the cache (I / D / L2 / L3), hint[2:0] the type (0 load, 1
    store, 7 prepare for store; the rest reserved: a nop).  eMIPS prefetches
    into the D-cache; the other targets do nothing — "any action, including
    doing nothing, is permitted"."""
    hint = i.f["hint"]
    if m.caches is None or hint >> 3 != 1 or (hint & 7) not in _PREF_TYPES:
        return
    va = (_reg(m, i.f["rs1"]) + i.f["imm"]) & m.amask
    kind = "load" if hint & 7 == 0 else "store"
    try:
        pa, cca = m._xlate(m, va, kind)
        if m.pmp.entries and not m.pmp.allows(pa, 1, kind, m.priv):
            return
    except (RVTrap, SimException):
        return                                  # (an address exception: a nop)
    if cca in m.cached_cca:
        m.caches.prefetch(_PREF_TYPES[hint & 7], va, pa, cca)


@_x("mips.mtlbwr")
def _mips_mtlbwr(m, i, pc):
    """MTLBWR rs1, level (MD01502 rev. 1.84 Appendix B, Xmipsstw): M-mode
    only; the level immediate must be 0 (the P8700 has non-leaf TLB
    entries: mipsconfig5.LEVEL gives the level); the PTE in rs1 for the VA
    in mtval, translation satp (XATP 0), G ORed with mipsconfig5.G — into
    the VTLB / FTLB as the walker writes (emips.p8700mmu)."""
    if m.priv < 3 or i.f["level"] != 0 or not hasattr(m.mmu, "write") or m.p8700 is None:
        raise RVTrap(ILLEGAL, i.raw)
    p = m.p8700
    _, asid, _ = m.mmu._satp(m)
    m.mmu.write(m, m.csrs.mtval, _reg(m, i.f["rs1"]), (p.config5 >> 7) & 7, asid,
                bool(p.config5 & (1 << 14)))


for _n in ("mips.ehb", "mips.ihb", "mips.pause"):
    # EHB / IHB: hazard barriers (the pipeline: IHB fetches the instructions
    # after it again); PAUSE a hint — nothing architectural
    EXECUTORS[_n] = lambda m, i, pc: None


# -- F / D (20, 21): emips.rvfpu -------------------------------------------------

from . import rvfpu as _rvfpu  # noqa: E402  (it imports this module's helpers)

EXECUTORS.update(_rvfpu._executors())

# -- scalar cryptography (31): emips.rvcrypto ------------------------------------

from . import rvcrypto as _rvcrypto  # noqa: E402

EXECUTORS.update(_rvcrypto._executors())

from . import rvvector as _rvvector  # noqa: E402

EXECUTORS.update(_rvvector.executors())
