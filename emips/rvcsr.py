"""The RISC-V hart's CSRs (PLAN M25 phase 1c; The RISC-V Instruction Set
Manual, Volume II: Privileged Architecture 20250508 — "priv." — and
Volume I's Zicsr / Zicntr / F chapters).

:class:`CSRFile` holds the CSRs of one :class:`emips.simrv.RVMachine`
and applies priv. 2.1's access rules (:meth:`CSRFile.check`): a CSR the
hart does not have, one whose number's bits 9:8 name a higher privilege
than the hart's, or a write to a read-only one (bits 11:10 = 11) is an
illegal instruction.

Phase 1 (an M-mode-only hart): the unprivileged counters (cycle / time /
instret, their ``h`` halves on RV32), the machine information registers,
mstatus (MIE, MPIE, MPP read-only 3 without U-mode, FS with F, SD),
misa (read-only: the extensions), mtvec (MODE 0 / 1; the values 2 / 3
are reserved — WARL: bit 1 is not kept, as Spike does), mepc (bit 0
always 0; bit 1 too without C), mcause, mtval, mscratch, mie / mip (the
M-level bits; nothing drives mip before phase 2), mcycle / minstret (and
``h``), mcountinhibit (CY, IR), mhpmcounter3-31 / mhpmevent3-31
(read-only 0: priv. 3.1.10 allows it), mconfigptr (0); fflags / frm /
fcsr with F (phase 1d).  Phase 2a completes the M-mode set (and S / U).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .simrv import RVMachine

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

# mstatus bits (priv. 3.1.6)
SIE, MIE, SPIE, MPIE = 1, 3, 5, 7
MPP_SHIFT = 11
FS_SHIFT = 13
MPRV = 17

#: the CSR numbers phase 1 implements -> name (riscv-opcodes csrs.csv names)
NAMES = {
    0x001: "fflags", 0x002: "frm", 0x003: "fcsr",
    0xC00: "cycle", 0xC01: "time", 0xC02: "instret",
    0xC80: "cycleh", 0xC81: "timeh", 0xC82: "instreth",
    0xF11: "mvendorid", 0xF12: "marchid", 0xF13: "mimpid", 0xF14: "mhartid",
    0xF15: "mconfigptr",
    0x300: "mstatus", 0x301: "misa", 0x304: "mie", 0x305: "mtvec", 0x310: "mstatush",
    0x320: "mcountinhibit",
    0x340: "mscratch", 0x341: "mepc", 0x342: "mcause", 0x343: "mtval", 0x344: "mip",
    0xB00: "mcycle", 0xB02: "minstret", 0xB80: "mcycleh", 0xB82: "minstreth",
    **{0xB00 + n: f"mhpmcounter{n}" for n in range(3, 32)},
    **{0xB80 + n: f"mhpmcounter{n}h" for n in range(3, 32)},
    **{0x320 + n: f"mhpmevent{n}" for n in range(3, 32)},
}
_RV32_ONLY = {0xC80, 0xC81, 0xC82, 0x310, 0xB80, 0xB82, *range(0xB83, 0xBA0)}
_FP = {0x001, 0x002, 0x003}


class CSRFile:
    """The CSRs of one hart (see the module docstring)."""

    def __init__(self, m: "RVMachine") -> None:
        self.m = m
        self.mstatus = 0
        self.mtvec = 0
        self.mepc = 0
        self.mcause = 0
        self.mtval = 0
        self.mscratch = 0
        self.mie = 0
        self.mcountinhibit = 0
        #: mcycle / minstret (64-bit on every XLEN, priv. 3.1.10)
        self.mcycle = 0
        self.minstret = 0
        #: the privilege modes the hart has (phase 1: M)
        self.modes = {3}
        #: MPP after an MRET: the least-privileged mode (priv. 3.3.2)
        self.lowest_priv = min(self.modes)
        # FS = Off at reset (phase 1d decides); MPP: read-only M without U-mode
        self.mstatus |= 3 << MPP_SHIFT

    # -- mstatus helpers -----------------------------------------------------------

    def status_bit(self, bit: int) -> int:
        return (self.mstatus >> bit) & 1

    def status_field(self, shift: int, mask: int) -> int:
        return (self.mstatus >> shift) & mask

    def set_status(self, *, mie: int | None = None, mpie: int | None = None,
                   mpp: int | None = None, mprv: int | None = None) -> None:
        s = self.mstatus
        for bit, v in ((MIE, mie), (MPIE, mpie), (MPRV, mprv)):
            if v is not None:
                s = (s & ~(1 << bit)) | ((v & 1) << bit)
        if mpp is not None:
            if mpp not in self.modes:           # WARL: an unsupported mode
                mpp = self.lowest_priv
            s = (s & ~(3 << MPP_SHIFT)) | (mpp << MPP_SHIFT)
        self.mstatus = s

    # -- access --------------------------------------------------------------------

    def exists(self, num: int) -> bool:
        if num not in NAMES:
            return False
        if num in _RV32_ONLY and self.m.xlen != 32:
            return False
        if num in _FP and "f" not in self.m.extensions:
            return False
        return True

    def check(self, num: int, *, write: bool, insn: int) -> None:
        """priv. 2.1: an illegal instruction for a CSR that does not exist,
        needs more privilege, or is read-only and written."""
        from .simrv import ILLEGAL, RVTrap
        if not self.exists(num) or (num >> 8) & 3 > self.m.priv or \
                (write and num >> 10 == 3):
            raise RVTrap(ILLEGAL, insn)
        if num in _FP and self.status_field(FS_SHIFT, 3) == 0:
            raise RVTrap(ILLEGAL, insn)         # F 21.2 / priv. 3.1.6.6: FS = Off

    def read(self, num: int) -> int:
        m = self.m
        r = _READ.get(num)
        if r is not None:
            return r(self) & m.amask
        if 0xB03 <= num <= 0xB1F or 0xB83 <= num <= 0xB9F or 0x323 <= num <= 0x33F:
            return 0                            # mhpmcounter / mhpmevent: read-only 0
        raise KeyError(f"CSR 0x{num:03x}")

    def write(self, num: int, value: int) -> None:
        w = _WRITE.get(num)
        if w is not None:
            w(self, value & self.m.amask)

    # -- the counters (priv. 3.1.10 / Zicntr) ----------------------------------------

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
        if 0 in self.modes:
            bits |= 1 << (ord("u") - ord("a"))
        if 1 in self.modes:
            bits |= 1 << (ord("s") - ord("a"))
        return mxl << (self.m.xlen - 2) | bits

    def sd_status(self) -> int:
        s = self.mstatus
        if (s >> FS_SHIFT) & 3 == 3:
            s |= 1 << (self.m.xlen - 1)
        return s


def _lo(v: int) -> int:
    return v & M32


def _hi(v: int) -> int:
    return v >> 32 & M32


def _counter(name: str, half: str) -> Callable[[CSRFile], int]:
    def read(c: CSRFile) -> int:
        v = c.m.mtime if name == "time" else getattr(c, name)
        if c.m.xlen == 32:
            return _hi(v) if half == "h" else _lo(v)
        return v
    return read


_READ: dict[int, Callable[[CSRFile], int]] = {
    0x001: lambda c: c.m.fcsr & 0x1F,
    0x002: lambda c: c.m.fcsr >> 5 & 7,
    0x003: lambda c: c.m.fcsr & 0xFF,
    0xC00: _counter("mcycle", ""), 0xC01: _counter("time", ""),
    0xC02: _counter("minstret", ""),
    0xC80: _counter("mcycle", "h"), 0xC81: _counter("time", "h"),
    0xC82: _counter("minstret", "h"),
    0xB00: _counter("mcycle", ""), 0xB02: _counter("minstret", ""),
    0xB80: _counter("mcycle", "h"), 0xB82: _counter("minstret", "h"),
    0xF11: lambda c: 0, 0xF12: lambda c: 0, 0xF13: lambda c: 0,
    0xF14: lambda c: c.m.hartid, 0xF15: lambda c: 0,
    0x300: CSRFile.sd_status,
    0x301: CSRFile.misa,
    0x304: lambda c: c.mie,
    0x305: lambda c: c.mtvec,
    0x310: lambda c: 0,
    0x320: lambda c: c.mcountinhibit,
    0x340: lambda c: c.mscratch,
    0x341: lambda c: c.mepc & ~(3 if c.m.ialign == 4 else 1),
    0x342: lambda c: c.mcause,
    0x343: lambda c: c.mtval,
    0x344: lambda c: 0,
}


def _w_mstatus(c: CSRFile, v: int) -> None:
    keep = (1 << MIE) | (1 << MPIE)
    if "f" in c.m.extensions:
        keep |= 3 << FS_SHIFT
    c.mstatus = (c.mstatus & ~keep) | (v & keep)
    c.set_status(mpp=(v >> MPP_SHIFT) & 3)


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


_WRITE: dict[int, Callable[[CSRFile, int], None]] = {
    0x001: lambda c, v: setattr(c.m, "fcsr", (c.m.fcsr & ~0x1F) | (v & 0x1F)),
    0x002: lambda c, v: setattr(c.m, "fcsr", (c.m.fcsr & 0x1F) | (v & 7) << 5),
    0x003: lambda c, v: setattr(c.m, "fcsr", v & 0xFF),
    0x300: _w_mstatus,
    0x304: lambda c, v: setattr(c, "mie", v & 0x888),          # MSIE, MTIE, MEIE
    0x305: lambda c, v: setattr(c, "mtvec", v & ~2),           # MODE 2 / 3 reserved
    0x320: lambda c, v: setattr(c, "mcountinhibit", v & 5),    # CY, IR
    0x340: lambda c, v: setattr(c, "mscratch", v),
    0x341: lambda c, v: setattr(c, "mepc", v & ~1),
    0x342: lambda c, v: setattr(c, "mcause", v),
    0x343: lambda c, v: setattr(c, "mtval", v),
    0xB00: _w_counter("mcycle", ""), 0xB02: _w_counter("minstret", ""),
    0xB80: _w_counter("mcycle", "h"), 0xB82: _w_counter("minstret", "h"),
}
