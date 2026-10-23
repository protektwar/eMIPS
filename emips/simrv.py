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
"""

from __future__ import annotations

from typing import Any, Callable

from . import rvcsr as CSR
from . import sim as S
from .exceptions import SimException
from .mmu import Translation
from .riscv import RV, RVDecodeError, RVISA, size_of

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

#: exception codes (priv. Table 3.6)
INSN_MISALIGNED, INSN_ACCESS, ILLEGAL, BREAKPOINT = 0, 1, 2, 3
LOAD_MISALIGNED, LOAD_ACCESS, STORE_MISALIGNED, STORE_ACCESS = 4, 5, 6, 7
ECALL_U, ECALL_S, ECALL_M = 8, 9, 11
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
#: the platform's physical address space (phase 1: 32 bits)
PA_BITS = 32


class RVTrap(Exception):
    """A synchronous exception an instruction raises (``cause``, ``tval``)."""

    def __init__(self, cause: int, tval: int = 0) -> None:
        super().__init__(CAUSE_NAMES.get(cause, f"cause {cause}"))
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
    def _cca(pa: int) -> int:
        return CCA_UNCACHED if MMIO_BASE <= pa <= M32 else CCA_CACHED

    def xlate(self, m, va: int, kind: str) -> tuple[int, int]:
        if va >> PA_BITS:
            raise RVTrap(INSN_ACCESS if kind == "fetch" else
                         STORE_ACCESS if kind == "store" else LOAD_ACCESS, va)
        return va, self._cca(va)

    def peek(self, m, va: int) -> tuple[int, int] | None:
        return None if va >> PA_BITS else (va, self._cca(va))

    def describe(self, m, va: int, kind: str = "load") -> Translation:
        if va >> PA_BITS:
            return Translation(va, None, CCA_UNCACHED, "bare", False,
                               f"bare: no physical memory above {PA_BITS} bits",
                               fault="access fault")
        how = "bare: PA = VA" + (", the MMIO page (uncached)" if self._cca(va) == CCA_UNCACHED
                                 else "")
        return Translation(va, va, self._cca(va), "bare", False, how)


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

    def __init__(self, memory=None, pc: int = 0, *, arch: str = "rv64",
                 extensions: str = "imafdc_zicsr_zifencei_zicntr_zba_zbb",
                 exceptions: str = "cp0", mmu: Any = None, caches=None, cpu: int = 0,
                 **_ignored: Any) -> None:
        if arch not in ("rv32", "rv64"):
            raise ValueError(f"arch must be 'rv32' or 'rv64', not {arch!r}")
        super().__init__(memory=memory, pc=0, endian="little", exceptions=exceptions,
                         release="R6", mmu=mmu if mmu is not None else BareMMU(),
                         caches=caches, cpu=cpu, fpu=False)
        self.arch = arch
        #: XLEN (priv. 3.1.1: MXLEN = XLEN here)
        self.xlen = 32 if arch == "rv32" else 64
        self.amask = M32 if self.xlen == 32 else M64
        self.vamask = self.amask
        self.wide = self.xlen == 64
        self.sgn = _s32 if self.xlen == 32 else _s64
        #: the instructions (emips.riscv)
        self.rvisa = RVISA.for_core(self.xlen, extensions)
        self.extensions = self.rvisa.extensions
        #: IALIGN (1.5): 16 with C, else 32
        self.ialign = 2 if "c" in self.extensions else 4
        #: the privilege mode: 3 = M, 1 = S, 0 = U (phase 1: M only)
        self.priv = 3
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
        #: the CSRs (emips.rvcsr)
        self.csrs = CSR.CSRFile(self)
        self.pc = pc & self.amask
        #: the last instruction executed (an RV) — the views
        self.last_insn: RV | None = None

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
        """The platform timer (phase 1; decision 9)."""
        return self.cycles // self.ccres

    # -- the Machine's MIPS paths a RISC-V hart never takes --------------------

    def interrupt_request(self, ixmt=None):         # interrupts: phase 2
        return None

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

    def fetch(self, pc: int) -> RV:
        """Fetch and decode the instruction at ``pc`` (its first halfword
        tells the size: 1.5); an illegal one raises :class:`RVTrap`."""
        h = self.fetch_half(pc)
        size = size_of(h)
        if size == 4:
            raw = h | self.fetch_half((pc + 2) & self.amask) << 16
        else:
            raw = h
        try:
            return self.rvisa.decode(raw, size or 4)
        except RVDecodeError:
            raise RVTrap(ILLEGAL, raw if size else h) from None

    def step(self) -> RV | None:
        """Execute one instruction (one clock cycle in this model);
        returns it (``None`` when the fetch or decode trapped)."""
        if self.halted:
            return None
        pc = self.pc
        i: RV | None = None
        try:
            if pc & (self.ialign - 1):
                raise RVTrap(INSN_MISALIGNED, pc)
            i = self.fetch(pc)
            nxt = EXECUTORS[i.base.name](self, i, pc)
        except RVTrap as t:
            self.advance(1)
            self._trap(t, pc, i)
            return i
        except SimException as exc:
            self.exception_log.append({"name": getattr(exc, "name", None),
                                       "word": getattr(exc, "word", None),
                                       "addr": getattr(exc, "addr", pc),
                                       "error": type(exc).__name__})
            raise
        if "minstret" not in self.counter_written:
            self.csrs.retire()
        self.last_insn = i
        self.advance(1)
        self.counter_written.clear()
        self.pc = (pc + i.size if nxt is None else nxt) & self.amask
        return i

    def _trap(self, t: RVTrap, pc: int, i: RV | None) -> None:
        """Take a synchronous exception into M-mode (priv. 3.1.6.1,
        3.1.7, 3.1.14-3.1.17)."""
        if self.exceptions == "raise":
            raise RVException(t.cause, t.tval, pc, name=i.name if i else None,
                              word=i.raw if i else None)
        self.exception_log.append({"name": CAUSE_NAMES.get(t.cause), "cause": t.cause,
                                   "tval": t.tval, "epc": pc, "word": i.raw if i else None,
                                   "priv": self.priv})
        c = self.csrs
        c.mepc = pc & ~1
        c.mcause = t.cause
        c.mtval = t.tval & self.amask
        c.set_status(mpie=c.status_bit(CSR.MIE), mie=0, mpp=self.priv)
        self.priv = 3
        self.reservation = None
        self.pc = c.mtvec & ~3 & self.amask

    def target(self, pc: int, offset: int) -> int:
        """A branch / jump target, checked for IALIGN (2.5: an
        instruction-address-misaligned exception on the branch)."""
        t = (pc + offset) & self.amask
        if t & (self.ialign - 1):
            raise RVTrap(INSN_MISALIGNED, t)
        return t


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
    raise RVTrap(BREAKPOINT, pc)


@_x("mret")
def _mret(m, i, pc):
    """priv. 3.3.2: MIE = MPIE, MPIE = 1, priv = MPP, MPP = U (M when no U)."""
    if m.priv != 3:
        raise RVTrap(ILLEGAL, i.raw)
    c = m.csrs
    mpp = c.status_field(CSR.MPP_SHIFT, 3)
    c.set_status(mie=c.status_bit(CSR.MPIE), mpie=1, mpp=c.lowest_priv)
    if mpp != 3:
        c.set_status(mprv=0)
    m.priv = mpp
    return c.mepc & m.amask & ~(m.ialign - 1 if m.ialign == 4 else 1)


@_x("wfi")
def _wfi(m, i, pc):
    """priv. 3.3.3: WFI may be a NOP (phase 1: no interrupt can wake it)."""


for _n in ("sret", "sfence.vma", "dret"):          # (no S-mode / Debug Mode: phase 2)
    EXECUTORS[_n] = lambda m, i, pc: (_ for _ in ()).throw(RVTrap(ILLEGAL, i.raw))


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
            new = src if op == "w" else (old | src) if op == "s" else (old & ~src)
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
        _aligned(m, addr, size, False)
        v = m.load(addr, size)
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
        _set(m, i.f["rd"], 0 if ok else 1)
    return ex


EXECUTORS.update({"lr.w": _lr(4), "lr.d": _lr(8), "sc.w": _sc(4), "sc.d": _sc(8)})


def _amo(name: str, size: int, op: Callable[[int, int, int], int]) -> None:
    bits = 8 * size

    def ex(m, i, pc):
        addr = _reg(m, i.f["rs1"])
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
