"""The RISC-V hart's CSRs (PLAN M25 phases 1c / 2a; The RISC-V Instruction
Set Manual, Volume II: Privileged Architecture 20250508 — "priv." — and
Volume I's Zicsr / Zicntr / Zihpm / F chapters).

:class:`CSRFile` holds the CSRs of one :class:`emips.simrv.RVMachine`:
:data:`CSRS` lists every one eMIPS implements with the condition under
which it exists (the hart's modes — M, M + U, M + S + U —, its XLEN and
extensions), how it reads and how it is written (WARL / WLRL legalised
here).  :meth:`CSRFile.check` applies priv. 2.1: a CSR that does not
exist, needs more privilege (number bits 9:8), is read-only (bits 11:10 =
11) and written, a counter its counter-enable bits hide (3.1.11 /
4.1.5), satp / SFENCE.VMA with mstatus.TVM, or fflags / frm / fcsr with
mstatus.FS = Off is an illegal instruction.

Where the specification leaves a choice, eMIPS takes Spike's (the
reference simulator, ``riscv/csr_init.cc`` / ``csrs.cc``): MPP / SPP
legalise an unsupported mode to the least privileged one; mtvec / stvec
drop MODE bit 1 (2 / 3 are reserved); mcounteren / scounteren /
mcountinhibit have one bit per implemented counter; mideleg delegates
SSI / STI / SEI, medeleg every exception S-mode can take (not the M-mode
ECALL; instruction-address-misaligned only without C); mip's SSIP / STIP
/ SEIP are writable with S-mode, SEIP read = the written bit OR the
external line; misa is read-only; mvendorid / marchid / mimpid read 0;
mhpmcounter3-31 / mhpmevent3-31 (and with Zihpm hpmcounter3-31) read 0.
The trigger registers reach :mod:`emips.rvdebug`; dcsr / dpc / dscratch0-1
exist only in Debug Mode (Debug spec 4.9).
Choices: ``sim.DEVIATIONS["RISC-V"]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .rvdebug import DCSR_DEBUGVER, MPTE, MTE, TINFO

if TYPE_CHECKING:
    from .simrv import RVMachine

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

# mstatus (priv. 3.1.6)
SIE, MIE, SPIE, UBE, MPIE, SPP = 1, 3, 5, 6, 7, 8
MPP_SHIFT, FS_SHIFT, XS_SHIFT = 11, 13, 15
MPRV, SUM, MXR, TVM, TW, TSR = 17, 18, 19, 20, 21, 22
UXL_SHIFT, SXL_SHIFT = 32, 34
#: mip / mie bits (priv. 3.1.9)
SSI, MSI, STI, MTI, SEI, MEI = 1, 3, 5, 7, 9, 11
#: the interrupt priority (3.1.9: MEI, MSI, MTI, SEI, SSI, STI)
PRIORITY = (MEI, MSI, MTI, SEI, SSI, STI)
S_INTERRUPTS = (1 << SSI) | (1 << STI) | (1 << SEI)
M_INTERRUPTS = (1 << MSI) | (1 << MTI) | (1 << MEI)


@dataclass(frozen=True)
class CSR:
    """One CSR: ``exists(c)`` on this hart, ``read(c)``, ``write(c, v)``
    (None: read-only)."""
    num: int
    name: str
    exists: Callable[["CSRFile"], bool]
    read: Callable[["CSRFile"], int]
    write: Callable[["CSRFile", int], None] | None = None


class CSRFile:
    """The CSRs of one hart (see the module docstring)."""

    def __init__(self, m: "RVMachine") -> None:
        self.m = m
        #: the hart's privilege modes (3 M, 1 S, 0 U)
        self.modes = frozenset(m.modes)
        self.has_s = 1 in self.modes
        self.has_u = 0 in self.modes
        #: the least-privileged mode (an xRET's xPP afterwards, MPP's
        #: legalisation)
        self.lowest_priv = min(self.modes)
        self.zihpm = "zihpm" in m.extensions
        # mstatus: reset MPP = the lowest mode (Spike), UXL / SXL = XLEN
        self.mstatus = self.lowest_priv << MPP_SHIFT
        if m.xlen == 64:
            if self.has_u:
                self.mstatus |= 2 << UXL_SHIFT
            if self.has_s:
                self.mstatus |= 2 << SXL_SHIFT
        self.mtvec = self.stvec = 0
        self.mepc = self.sepc = 0
        self.mcause = self.scause = 0
        self.mtval = self.stval = 0
        self.mscratch = self.sscratch = 0
        self.mie = 0
        #: mip's software-writable bits (SSIP, STIP, SEIP's own bit)
        self.mip_sw = 0
        self.medeleg = self.mideleg = 0
        self.mcounteren = self.scounteren = 0
        self.mcountinhibit = 0
        self.menvcfg = self.senvcfg = 0
        self.satp = 0
        #: mcycle / minstret (64-bit on every XLEN, priv. 3.1.10)
        self.mcycle = 0
        self.minstret = 0
        #: Sdext (Debug spec 4.9): dcsr (debugver 4, prv 3 at reset), dpc,
        #: dscratch0 / 1 — Debug Mode only
        self.dcsr = DCSR_DEBUGVER | 3
        self.dpc = 0
        self.dscratch0 = self.dscratch1 = 0
        self.table = {c.num: c for c in CSRS if c.exists(self)}

    # -- mstatus helpers -------------------------------------------------------------

    def status_bit(self, bit: int) -> int:
        return (self.mstatus >> bit) & 1

    def status_field(self, shift: int, mask: int) -> int:
        return (self.mstatus >> shift) & mask

    def legal_mode(self, mode: int) -> int:
        """A privilege mode an xPP field may hold (Spike's legalisation:
        an unsupported mode becomes the least privileged one)."""
        return mode if mode in self.modes else self.lowest_priv

    def set_status(self, **fields: int) -> None:
        """Set mstatus bits by name: mie mpie sie spie mprv (one bit), mpp
        (2 bits, legalised), spp (one bit)."""
        s = self.mstatus
        bits = {"mie": MIE, "mpie": MPIE, "sie": SIE, "spie": SPIE, "mprv": MPRV, "spp": SPP}
        for name, v in fields.items():
            if name == "mpp":
                s = (s & ~(3 << MPP_SHIFT)) | (self.legal_mode(v) << MPP_SHIFT)
            else:
                s = (s & ~(1 << bits[name])) | ((v & 1) << bits[name])
        self.mstatus = s

    def mstatus_value(self) -> int:
        """mstatus as read (SD = FS or XS Dirty)."""
        s = self.mstatus
        if (s >> FS_SHIFT) & 3 == 3:
            s |= 1 << (self.m.xlen - 1) if self.m.xlen == 32 else 1 << 63
        return s

    def mstatus_mask(self) -> int:
        """mstatus's writable bits on this hart."""
        mask = (1 << MIE) | (1 << MPIE) | (3 << MPP_SHIFT)
        if self.has_s:
            mask |= (1 << SIE) | (1 << SPIE) | (1 << SPP) | (1 << TSR)
        if self.has_u:
            mask |= (1 << MPRV) | (1 << TW)
        if self.has_s and self.m.paging:
            mask |= (1 << SUM) | (1 << MXR) | (1 << TVM)
        if self.m.flen:
            mask |= 3 << FS_SHIFT
        return mask

    def write_mstatus(self, v: int, mask: int | None = None) -> None:
        mask = self.mstatus_mask() if mask is None else mask
        mpp = self.legal_mode((v >> MPP_SHIFT) & 3)
        v = (v & ~(3 << MPP_SHIFT)) | (mpp << MPP_SHIFT)
        self.mstatus = (self.mstatus & ~mask) | (v & mask)

    def sstatus_mask(self) -> int:
        """sstatus's writable bits (priv. 10.1.1)."""
        mask = (1 << SIE) | (1 << SPIE) | (1 << SPP)
        if self.m.paging:
            mask |= (1 << SUM) | (1 << MXR)
        if self.m.flen:
            mask |= 3 << FS_SHIFT
        return mask

    def sstatus_value(self) -> int:
        read = self.sstatus_mask() | (3 << XS_SHIFT) | (1 << UBE)
        if self.m.xlen == 64:
            read |= 3 << UXL_SHIFT | 1 << 63
        else:
            read |= 1 << 31
        return self.mstatus_value() & read

    # -- interrupts ------------------------------------------------------------------

    def mip(self) -> int:
        """mip as read: the platform's lines (MSIP, MTIP, MEIP, the
        external SEIP) and the software-writable bits."""
        return self.m.irq_lines & (M_INTERRUPTS | (1 << SEI) if self.has_s else M_INTERRUPTS) \
            | self.mip_sw

    def mip_mask(self) -> int:
        return S_INTERRUPTS if self.has_s else 0

    def mie_mask(self) -> int:
        return M_INTERRUPTS | (S_INTERRUPTS if self.has_s else 0)

    def medeleg_mask(self) -> int:
        mask = sum(1 << c for c in (1, 2, 3, 4, 5, 6, 7, 8, 9))
        if self.m.ialign == 4:
            mask |= 1                           # (with C it cannot happen: Spike)
        if self.m.paging:
            mask |= (1 << 12) | (1 << 13) | (1 << 15)
        return mask

    def counter_mask(self) -> int:
        return 0x7 | (0xFFFFFFF8 if self.zihpm else 0)

    # -- access --------------------------------------------------------------------

    def exists(self, num: int) -> bool:
        return num in self.table

    def check(self, num: int, *, write: bool, insn: int) -> None:
        """priv. 2.1 and the per-CSR rules (see the module docstring)."""
        from .simrv import ILLEGAL, RVTrap
        m = self.m
        if num not in self.table or (num >> 8) & 3 > m.priv or (write and num >> 10 == 3):
            raise RVTrap(ILLEGAL, insn)
        if 0x7B0 <= num <= 0x7BF and not m.debug_mode:
            raise RVTrap(ILLEGAL, insn)         # Debug spec 4.9: Debug Mode only
        if num in (0x001, 0x002, 0x003) and self.status_field(FS_SHIFT, 3) == 0:
            raise RVTrap(ILLEGAL, insn)         # F 21.2 / priv. 3.1.6.6: FS = Off
        if 0xC00 <= num <= 0xC1F or 0xC80 <= num <= 0xC9F:
            bit = 1 << (num & 31)               # 3.1.11 / 10.1.5: the counter enables
            if m.priv < 3 and not self.mcounteren & bit:
                raise RVTrap(ILLEGAL, insn)
            if m.priv == 0 and self.has_s and not self.scounteren & bit:
                raise RVTrap(ILLEGAL, insn)
        if num == 0x180 and m.priv == 1 and self.status_bit(TVM):
            raise RVTrap(ILLEGAL, insn)         # 3.1.6.6: TVM

    def read(self, num: int) -> int:
        return self.table[num].read(self) & self.m.amask

    def read_for_write(self, num: int) -> int:
        """The value CSRRS / CSRRC modify (mip: the software-writable
        SEIP, not the external line — priv. 3.1.9)."""
        if num in (0x344, 0x144):
            v = (self.mip() & ~(1 << SEI)) | (self.mip_sw & (1 << SEI))
            return v if num == 0x344 else v & self.mideleg
        return self.read(num)

    def write(self, num: int, value: int) -> None:
        w = self.table[num].write
        if w is not None:
            w(self, value & self.m.amask)

    # -- the counters (priv. 3.1.10 / Zicntr) -------------------------------------------

    def tick(self, cycles: int) -> None:
        if not self.mcountinhibit & 1:
            self.mcycle = (self.mcycle + cycles) & M64

    def retire(self) -> None:
        if not self.mcountinhibit & 4:
            self.minstret = (self.minstret + 1) & M64

    def misa(self) -> int:
        mxl = 1 if self.m.xlen == 32 else 2
        bits = 0
        for e in self.m.extensions:
            if len(e) == 1 and e.isalpha():
                bits |= 1 << (ord(e) - ord("a"))
        if self.has_u:
            bits |= 1 << (ord("u") - ord("a"))
        if self.has_s:
            bits |= 1 << (ord("s") - ord("a"))
        return mxl << (self.m.xlen - 2) | bits


# ------------------------------------------------------------------ the table

def _always(c: CSRFile) -> bool:
    return True


def _rv32(c: CSRFile) -> bool:
    return c.m.xlen == 32


def _s(c: CSRFile) -> bool:
    return c.has_s


def _u(c: CSRFile) -> bool:
    return c.has_u


def _f(c: CSRFile) -> bool:
    return c.m.flen > 0


def _attr(name: str) -> Callable[[CSRFile], int]:
    return lambda c: getattr(c, name)


def _set(name: str, mask: int | Callable[[CSRFile], int] = M64) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        mk = mask(c) if callable(mask) else mask
        setattr(c, name, (getattr(c, name) & ~mk) | (v & mk))
    return w


def _counter(name: str, half: str) -> Callable[[CSRFile], int]:
    def read(c: CSRFile) -> int:
        v = c.m.mtime if name == "time" else getattr(c, name) if name else 0
        if c.m.xlen == 32:
            return (v >> 32 if half == "h" else v) & M32
        return v
    return read


def _w_counter(name: str, half: str) -> Callable[[CSRFile, int], None]:
    def write(c: CSRFile, v: int) -> None:
        old = getattr(c, name)
        if c.m.xlen == 32:
            new = (old & M32) | (v & M32) << 32 if half == "h" else (old & ~M32) | (v & M32)
        else:
            new = v
        setattr(c, name, new & M64)
        c.m.counter_written.add(name)           # (takes effect after this instruction)
    return write


def _w_fcsr(part: str) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        m = c.m
        if part == "fflags":
            m.fcsr = (m.fcsr & ~0x1F) | (v & 0x1F)
        elif part == "frm":
            m.fcsr = (m.fcsr & 0x1F) | (v & 7) << 5
        else:
            m.fcsr = v & 0xFF
        c.mstatus |= 3 << FS_SHIFT              # (FP state written: Dirty)
    return w


def _w_tvec(name: str) -> Callable[[CSRFile, int], None]:
    return lambda c, v: setattr(c, name, v & ~2)     # MODE 2 / 3 reserved (Spike: ~2)


def _r_epc(name: str) -> Callable[[CSRFile], int]:
    return lambda c: getattr(c, name) & ~(3 if c.m.ialign == 4 else 1)


def _w_mip(c: CSRFile, v: int) -> None:
    mk = c.mip_mask()
    c.mip_sw = (c.mip_sw & ~mk) | (v & mk)


def _w_sip(c: CSRFile, v: int) -> None:
    mk = (1 << SSI) & c.mideleg                 # 10.1.3: only SSIP is writable in sip
    c.mip_sw = (c.mip_sw & ~mk) | (v & mk)


def _w_sie(c: CSRFile, v: int) -> None:
    mk = S_INTERRUPTS & c.mideleg
    c.mie = (c.mie & ~mk) | (v & mk)


def _w_satp(c: CSRFile, v: int) -> None:
    """priv. 10.1.11: a write of an unsupported MODE has no effect."""
    m = c.m
    mode = v >> 31 if m.xlen == 32 else v >> 60
    if mode in m.satp_modes:
        c.satp = v


def _w_dcsr(c: CSRFile, v: int) -> None:
    """Debug spec 4.9.1: ebreakm / ebreaks / ebreaku (with S / U), step,
    mprven and prv (an unsupported mode: the least privileged, as MPP)
    are writable; stepie, stopcount, stoptime read 0 (Spike); debugver,
    cause read-only."""
    from . import rvdebug as D
    mk = D.EBREAKM | D.STEP | D.MPRVEN | (D.EBREAKS if c.has_s else 0) \
        | (D.EBREAKU if c.has_u else 0)
    c.dcsr = (c.dcsr & ~(mk | 3)) | (v & mk) | c.legal_mode(v & 3)


def _trig(c: CSRFile) -> bool:
    return c.m.tm.count > 0


def _menvcfg_mask(c: CSRFile) -> int:
    return 1 if c.has_s else 0                  # FIOM (3.1.18; may be RO 0 without S)


def _half(base: Callable[[CSRFile], int], half: str) -> Callable[[CSRFile], int]:
    return (lambda c: base(c) >> 32 & M32) if half == "h" else (lambda c: base(c) & M32 if
                                                               c.m.xlen == 32 else base(c))


def _w64(name: str, mask: Callable[[CSRFile], int], half: str) -> Callable[[CSRFile, int], None]:
    """A 64-bit register's write (RV32: one half)."""
    def w(c: CSRFile, v: int) -> None:
        mk = mask(c)
        if c.m.xlen == 32:
            mk &= (M32 << 32) if half == "h" else M32
            v = (v & M32) << 32 if half == "h" else v & M32
        setattr(c, name, (getattr(c, name) & ~mk) | (v & mk))
    return w


CSRS: list[CSR] = [
    # -- unprivileged (Zicsr / F 21.2 / Zicntr / Zihpm) --
    CSR(0x001, "fflags", _f, lambda c: c.m.fcsr & 0x1F, _w_fcsr("fflags")),
    CSR(0x002, "frm", _f, lambda c: c.m.fcsr >> 5 & 7, _w_fcsr("frm")),
    CSR(0x003, "fcsr", _f, lambda c: c.m.fcsr & 0xFF, _w_fcsr("fcsr")),
    CSR(0xC00, "cycle", _always, _counter("mcycle", "")),
    CSR(0xC01, "time", _always, _counter("time", "")),
    CSR(0xC02, "instret", _always, _counter("minstret", "")),
    CSR(0xC80, "cycleh", _rv32, _counter("mcycle", "h")),
    CSR(0xC81, "timeh", _rv32, _counter("time", "h")),
    CSR(0xC82, "instreth", _rv32, _counter("minstret", "h")),
    *(CSR(0xC00 + n, f"hpmcounter{n}", lambda c: c.zihpm, lambda c: 0) for n in range(3, 32)),
    *(CSR(0xC80 + n, f"hpmcounter{n}h", lambda c: c.zihpm and c.m.xlen == 32, lambda c: 0)
      for n in range(3, 32)),
    # -- supervisor (priv. 10.1) --
    CSR(0x100, "sstatus", _s, CSRFile.sstatus_value,
        lambda c, v: c.write_mstatus(v, c.sstatus_mask())),
    CSR(0x104, "sie", _s, lambda c: c.mie & c.mideleg & S_INTERRUPTS, _w_sie),
    CSR(0x105, "stvec", _s, _attr("stvec"), _w_tvec("stvec")),
    CSR(0x106, "scounteren", _s, _attr("scounteren"), _set("scounteren", CSRFile.counter_mask)),
    CSR(0x10A, "senvcfg", _s, _attr("senvcfg"), _set("senvcfg", 1)),
    CSR(0x140, "sscratch", _s, _attr("sscratch"), _set("sscratch")),
    CSR(0x141, "sepc", _s, _r_epc("sepc"), lambda c, v: setattr(c, "sepc", v & ~1)),
    CSR(0x142, "scause", _s, _attr("scause"), _set("scause")),
    CSR(0x143, "stval", _s, _attr("stval"), _set("stval")),
    CSR(0x144, "sip", _s, lambda c: c.mip() & c.mideleg & S_INTERRUPTS, _w_sip),
    CSR(0x180, "satp", _s, _attr("satp"), _w_satp),
    # -- machine (priv. 3.1) --
    CSR(0xF11, "mvendorid", _always, lambda c: 0),
    CSR(0xF12, "marchid", _always, lambda c: 0),
    CSR(0xF13, "mimpid", _always, lambda c: 0),
    CSR(0xF14, "mhartid", _always, lambda c: c.m.hartid),
    CSR(0xF15, "mconfigptr", _always, lambda c: 0),
    CSR(0x300, "mstatus", _always, CSRFile.mstatus_value,
        lambda c, v: c.write_mstatus(v if c.m.xlen == 64 else (c.mstatus & ~M32) | v)),
    CSR(0x301, "misa", _always, CSRFile.misa, lambda c, v: None),
    CSR(0x302, "medeleg", _s, lambda c: c.medeleg & (M32 if c.m.xlen == 32 else M64),
        _w64("medeleg", CSRFile.medeleg_mask, "")),
    CSR(0x303, "mideleg", _s, _attr("mideleg"), _set("mideleg", lambda c: S_INTERRUPTS)),
    CSR(0x304, "mie", _always, _attr("mie"), _set("mie", CSRFile.mie_mask)),
    CSR(0x305, "mtvec", _always, _attr("mtvec"), _w_tvec("mtvec")),
    CSR(0x306, "mcounteren", _u, _attr("mcounteren"), _set("mcounteren", CSRFile.counter_mask)),
    CSR(0x30A, "menvcfg", _u, _half(_attr("menvcfg"), ""), _w64("menvcfg", _menvcfg_mask, "")),
    CSR(0x310, "mstatush", _rv32, lambda c: c.mstatus >> 32 & M32, lambda c, v: None),
    CSR(0x312, "medelegh", lambda c: c.has_s and c.m.xlen == 32, lambda c: c.medeleg >> 32,
        _w64("medeleg", CSRFile.medeleg_mask, "h")),
    CSR(0x31A, "menvcfgh", lambda c: c.has_u and c.m.xlen == 32, lambda c: c.menvcfg >> 32,
        _w64("menvcfg", _menvcfg_mask, "h")),
    CSR(0x320, "mcountinhibit", _always, _attr("mcountinhibit"),
        _set("mcountinhibit", lambda c: c.counter_mask() & ~2)),
    *(CSR(0x320 + n, f"mhpmevent{n}", _always, lambda c: 0, lambda c, v: None)
      for n in range(3, 32)),
    CSR(0x340, "mscratch", _always, _attr("mscratch"), _set("mscratch")),
    CSR(0x341, "mepc", _always, _r_epc("mepc"), lambda c, v: setattr(c, "mepc", v & ~1)),
    CSR(0x342, "mcause", _always, _attr("mcause"), _set("mcause")),
    CSR(0x343, "mtval", _always, _attr("mtval"), _set("mtval")),
    CSR(0x344, "mip", _always, CSRFile.mip, _w_mip),
    CSR(0xB00, "mcycle", _always, _counter("mcycle", ""), _w_counter("mcycle", "")),
    CSR(0xB02, "minstret", _always, _counter("minstret", ""), _w_counter("minstret", "")),
    CSR(0xB80, "mcycleh", _rv32, _counter("mcycle", "h"), _w_counter("mcycle", "h")),
    CSR(0xB82, "minstreth", _rv32, _counter("minstret", "h"), _w_counter("minstret", "h")),
    # -- PMP (3.7.1): every pmpaddr / pmpcfg exists; RV64 has only the even pmpcfgs --
    *(CSR(0x3A0 + n, f"pmpcfg{n}", lambda c, n=n: c.m.pmp.cfg_exists(n),
          lambda c, n=n: c.m.pmp.read_cfg(n), lambda c, v, n=n: c.m.pmp.write_cfg(n, v))
      for n in range(16)),
    *(CSR(0x3B0 + i, f"pmpaddr{i}", _always, lambda c, i=i: c.m.pmp.read_addr(i),
          lambda c, v, i=i: c.m.pmp.write_addr(i, v)) for i in range(64)),
    *(CSR(0xB00 + n, f"mhpmcounter{n}", _always, lambda c: 0, lambda c, v: None)
      for n in range(3, 32)),
    # -- the trigger module (Debug spec 5.7; emips.rvdebug) --
    CSR(0x7A0, "tselect", _trig, lambda c: c.m.tm.tselect, lambda c, v: c.m.tm.write_tselect(v)),
    CSR(0x7A1, "tdata1", _trig, lambda c: c.m.tm.read1(), lambda c, v: c.m.tm.write1(v)),
    CSR(0x7A2, "tdata2", _trig, lambda c: c.m.tm.read2(), lambda c, v: c.m.tm.write2(v)),
    CSR(0x7A3, "tdata3", _trig, lambda c: c.m.tm.read3(), lambda c, v: c.m.tm.write3(v)),
    CSR(0x7A4, "tinfo", _trig, lambda c: TINFO, lambda c, v: None),
    CSR(0x7A5, "tcontrol", lambda c: _trig(c) and not c.has_s, lambda c: c.m.tm.tcontrol,
        lambda c, v: setattr(c.m.tm, "tcontrol", v & (MTE | MPTE))),
    CSR(0x7A8, "mcontext", _trig, lambda c: c.m.tm.mcontext,
        lambda c, v: setattr(c.m.tm, "mcontext", v & ((1 << (6 if c.m.xlen == 32 else 13)) - 1))),
    CSR(0x5A8, "scontext", lambda c: _trig(c) and c.has_s, lambda c: c.m.tm.scontext,
        lambda c, v: setattr(c.m.tm, "scontext", v & ((1 << (16 if c.m.xlen == 32 else 32)) - 1))),
    # -- Debug Mode (Debug spec 4.9; Sdext) --
    CSR(0x7B0, "dcsr", _always, _attr("dcsr"), _w_dcsr),
    CSR(0x7B1, "dpc", _always, _r_epc("dpc"), lambda c, v: setattr(c, "dpc", v & ~1)),
    CSR(0x7B2, "dscratch0", _always, _attr("dscratch0"), _set("dscratch0")),
    CSR(0x7B3, "dscratch1", _always, _attr("dscratch1"), _set("dscratch1")),
    *(CSR(0xB80 + n, f"mhpmcounter{n}h", _rv32, lambda c: 0, lambda c, v: None)
      for n in range(3, 32)),
]
NAMES = {c.num: c.name for c in CSRS}
