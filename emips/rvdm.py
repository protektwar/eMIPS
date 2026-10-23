"""The RISC-V Debug Module and its JTAG Debug Transport Module (PLAN M25
phase 2f), after The RISC-V Debug Specification 1.0 chapters 3 and 6
(``docs`` keeps the PDF).

:class:`DebugModule` is a version-1.0 DM (dmstatus.version 3) for the
harts of one platform (:class:`emips.simrv.RVMachine`): its registers
on the Debug Module Interface (:meth:`DebugModule.dmi_read` /
:meth:`dmi_write`, 7 address bits), run control (haltreq, resumereq,
halt-on-reset, hartreset, ndmreset, ackhavereset), the hart array mask
(hasel, hawindowsel / hawindow), halt and resume groups (dmcs2; no
external triggers), halt summaries, the abstract commands Access
Register (GPRs, FPRs, CSRs — dpc is the PC —, postexec, postincrement),
Quick Access and Access Memory (physical or as M-mode with MPRV:
aamvirtual, postincrement), abstractauto, the Program Buffer (with an
implicit ebreak), and System Bus Access (sbversion 1, 8- to 64-bit,
autoincrement, read on address / data).

The implementation is a *direct* one (the spec leaves it open, 4.1): the
DM performs an abstract command on the hart itself and runs the program
buffer through :meth:`emips.simrv.RVMachine.execute_debug` — so the
data registers are not shadowed in the hart (hartinfo.dataaccess = 0,
datasize 0; both dscratch registers are the debugger's), control
transfers and AUIPC in the program buffer are illegal (4.1 rules
10-12), and every command completes at once (busy is never seen).
Spike, the oracle, is an execution-based DM (its harts run a debug ROM);
where the two must differ is listed in ``sim.DEVIATIONS["RISC-V"]``.

:class:`JtagDTM` is the DTM of 6.1 on the IEEE 1149.1 state machine of
:mod:`emips.tap`: IR 5 bits (IDCODE 0x01 after Test-Logic-Reset, dtmcs
0x10, dmi 0x11, BYPASS for every other instruction), dmi = abits 7 +
data 32 + op 2; an access starts at Update-DR and its result is captured
by the next Capture-DR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .tap import (CAP_DR, CAP_IR, IDLE, NEXT, SHIFT_DR, SHIFT_IR, TLR, UPD_DR, UPD_IR)

if TYPE_CHECKING:
    from .simrv import RVMachine

M32 = 0xFFFFFFFF

# DMI addresses (Table 6)
DATA0, DMCONTROL, DMSTATUS, HARTINFO, HALTSUM1 = 0x04, 0x10, 0x11, 0x12, 0x13
HAWINDOWSEL, HAWINDOW, ABSTRACTCS, COMMAND, ABSTRACTAUTO = 0x14, 0x15, 0x16, 0x17, 0x18
CONFSTRPTR0, NEXTDM, PROGBUF0, AUTHDATA, DMCS2 = 0x19, 0x1D, 0x20, 0x30, 0x32
HALTSUM2, HALTSUM3, SBADDRESS3, SBCS = 0x34, 0x35, 0x37, 0x38
SBADDRESS0, SBADDRESS1, SBADDRESS2 = 0x39, 0x3A, 0x3B
SBDATA0, SBDATA1, SBDATA2, SBDATA3, HALTSUM0 = 0x3C, 0x3D, 0x3E, 0x3F, 0x40

# abstractcs.cmderr (3.14.6)
ERR_NONE, ERR_BUSY, ERR_NOTSUP, ERR_EXCEPTION, ERR_HALTRESUME, ERR_BUS, ERR_OTHER = \
    0, 1, 2, 3, 4, 5, 7
# dcsr.cause for a halt (4.9.1)
CAUSE_HALTREQ, CAUSE_RESETHALTREQ, CAUSE_GROUP = 3, 5, 6
#: the JTAG DTM (6.1)
IR_IDCODE, IR_DTMCS, IR_DMI, IR_BYPASS = 0x01, 0x10, 0x11, 0x1F
ABITS = 7
#: eMIPS's IDCODE: version 0, part 0, no JEDEC manufacturer (bit 0 = 1),
#: as its EJTAG TAP (emips.tap.DEVICE_ID)
IDCODE = 0x00000001


def _width(n: int) -> int:
    """The bits needed for the indices 0 .. n - 1 (HARTSELLEN)."""
    return (n - 1).bit_length()


class _Hart:
    """The DM's view of one hart (3.5): halt request, resume ack,
    halt-on-reset, havereset; the groups (3.6); held in reset."""

    def __init__(self, m: "RVMachine") -> None:
        self.m = m
        self.haltreq = False
        self.resumeack = False
        self.resethaltreq = False
        self.havereset = True                   # (the power-on reset is a reset)
        self.haltgroup = 0
        self.resumegroup = 0
        self.in_reset = False


class DebugModule:
    """A Debug Module for ``harts`` (see the module docstring)."""

    #: the highest halt / resume group (dmcs2.group is 5 bits)
    GROUPS = 31

    def __init__(self, harts: list["RVMachine"], *, progbufsize: int = 2,
                 datacount: int = 4, sba: int = 64, impebreak: bool = True) -> None:
        if not 1 <= datacount <= 12 or not 0 <= progbufsize <= 16:
            raise ValueError("datacount must be 1-12, progbufsize 0-16")
        if progbufsize == 1 and not impebreak:
            raise ValueError("a 1-word program buffer needs impebreak (3.8)")
        self.harts = [_Hart(m) for m in harts]
        for m in harts:
            m.debug_module = self
        self.progbufsize = progbufsize
        self.datacount = datacount
        self.sba = sba
        self.impebreak = impebreak
        self.hartsellen = _width(len(harts))
        self._reset_dm()
        self.dmactive = False

    # -- the DM's own state --------------------------------------------------------

    def _reset_dm(self) -> None:
        """dmactive = 0 (3.14.2): the DM's registers take their reset values;
        harts go to group 0 (3.6)."""
        self.dmactive = False
        self.hartsel = 0
        self.hasel = False
        self.ndmreset = False
        self.hartreset = False
        self.mask = [False] * len(self.harts)
        self.hawindowsel = 0
        self.data = [0] * self.datacount
        self.progbuf = [0] * self.progbufsize
        self.command = 0
        self.cmderr = ERR_NONE
        self.autoexecdata = 0
        self.autoexecprogbuf = 0
        self.grouptype = 0
        self.sbcs_access = 2
        self.sbreadonaddr = self.sbreadondata = self.sbautoincrement = False
        self.sberror = 0
        self.sbbusyerror = False
        self.sbaddress = 0
        self.sbdata = [0, 0, 0, 0]
        for h in self.harts:
            h.haltreq = False
            h.haltgroup = h.resumegroup = 0

    def _selected(self) -> list[_Hart]:
        """The currently selected harts (3.3): hartsel's (if it exists) and,
        with hasel, those of the hart array mask."""
        out = []
        for k, h in enumerate(self.harts):
            if k == self.hartsel or (self.hasel and self.mask[k]):
                out.append(h)
        return out

    def _hart(self) -> _Hart | None:
        return self.harts[self.hartsel] if self.hartsel < len(self.harts) else None

    def available(self, h: _Hart) -> bool:
        return not h.in_reset

    # -- the hart's side -----------------------------------------------------------

    def hart_halted(self, m: "RVMachine") -> None:
        """A hart entered Debug Mode (any cause): the other running harts of
        its halt group halt too (3.6, cause 6)."""
        h = next((x for x in self.harts if x.m is m), None)
        if h is None or not h.haltgroup:
            return
        for other in self.harts:
            if other is not h and other.haltgroup == h.haltgroup and not other.m.debug_mode \
                    and self.available(other):
                other.m.enter_debug(CAUSE_GROUP)

    def _halt(self, h: _Hart, cause: int = CAUSE_HALTREQ) -> None:
        if not h.m.debug_mode and self.available(h):
            h.m.enter_debug(cause)

    def _resume(self, h: _Hart) -> None:
        """3.5: a halted hart resumes (DRET's work) and sets its resume ack;
        the other halted harts of its resume group follow (3.6)."""
        group = [h]
        if h.resumegroup:
            group += [x for x in self.harts if x is not h and x.resumegroup == h.resumegroup]
        for x in group:
            if x.m.debug_mode and self.available(x):
                x.m.resume()
                x.resumeack = True

    def _reset(self, h: _Hart, held: bool) -> None:
        """hartreset / ndmreset (3.2): held while the bit is 1; released, the
        hart starts from reset and halts at once with haltreq or
        resethaltreq (cause 5 / 3)."""
        if held:
            if not h.in_reset:
                h.in_reset = True
                h.m.reset_hart()
            return
        if h.in_reset:
            h.in_reset = False
            h.havereset = True
            h.resumeack = False
            if h.resethaltreq:
                h.m.enter_debug(CAUSE_RESETHALTREQ)
            elif h.haltreq:
                h.m.enter_debug(CAUSE_HALTREQ)

    # -- the DMI ---------------------------------------------------------------------

    def dmi_read(self, addr: int) -> int:
        """A DMI read (3.14; unimplemented registers read 0)."""
        if not self.dmactive and addr != DMCONTROL:
            return self._dmstatus() if addr == DMSTATUS else 0
        if DATA0 <= addr < DATA0 + self.datacount:
            v = self.data[addr - DATA0]
            self._auto(self.autoexecdata >> (addr - DATA0) & 1)
            return v
        if PROGBUF0 <= addr < PROGBUF0 + self.progbufsize:
            v = self.progbuf[addr - PROGBUF0]
            self._auto(self.autoexecprogbuf >> (addr - PROGBUF0) & 1)
            return v
        if addr == DMCONTROL:
            return (self.hartreset << 29 | self.hasel << 26 | (self.hartsel & 0x3FF) << 16
                    | (self.hartsel >> 10) << 6 | self.ndmreset << 1 | self.dmactive)
        if addr == DMSTATUS:
            return self._dmstatus()
        if addr == HARTINFO:
            return 2 << 20                      # nscratch 2; dataaccess 0, datasize 0
        if addr == ABSTRACTCS:
            return self.progbufsize << 24 | self.cmderr << 8 | self.datacount
        if addr == COMMAND:
            return 0
        if addr == ABSTRACTAUTO:
            return self.autoexecprogbuf << 16 | self.autoexecdata
        if addr == HAWINDOWSEL:
            return self.hawindowsel
        if addr == HAWINDOW:
            base = self.hawindowsel * 32
            return sum(1 << i for i in range(32)
                       if base + i < len(self.harts) and self.mask[base + i])
        if addr == DMCS2:
            h = self._hart()
            group = 0 if h is None else (h.resumegroup if self.grouptype else h.haltgroup)
            return self.grouptype << 11 | group << 2
        if addr in (HALTSUM0, HALTSUM1, HALTSUM2, HALTSUM3):
            return self._haltsum(addr)
        if addr == SBCS:
            return self._sbcs()
        if addr == SBADDRESS0:
            return self.sbaddress & M32
        if addr == SBADDRESS1:
            return self.sbaddress >> 32 & M32
        if addr in (SBDATA0, SBDATA1, SBDATA2, SBDATA3):
            k = (SBDATA0, SBDATA1, SBDATA2, SBDATA3).index(addr)
            v = self.sbdata[k]
            if k == 0 and self.sbreadondata and self._sb_ok():
                self._sb_read()
            return v
        return 0

    def dmi_write(self, addr: int, value: int) -> None:
        """A DMI write (3.14)."""
        value &= M32
        if addr == DMCONTROL:
            self._write_dmcontrol(value)
            return
        if not self.dmactive:
            return
        if DATA0 <= addr < DATA0 + self.datacount:
            self.data[addr - DATA0] = value
            self._auto(self.autoexecdata >> (addr - DATA0) & 1)
        elif PROGBUF0 <= addr < PROGBUF0 + self.progbufsize:
            self.progbuf[addr - PROGBUF0] = value
            self._auto(self.autoexecprogbuf >> (addr - PROGBUF0) & 1)
        elif addr == ABSTRACTCS:
            self.cmderr &= ~(value >> 8 & 7)    # R/W1C
        elif addr == COMMAND:
            if self.cmderr == ERR_NONE:
                self.command = value
                self._execute()
        elif addr == ABSTRACTAUTO:
            self.autoexecdata = value & ((1 << self.datacount) - 1)
            self.autoexecprogbuf = value >> 16 & ((1 << self.progbufsize) - 1)
        elif addr == HAWINDOWSEL:
            self.hawindowsel = value & ((1 << max(0, _width(len(self.harts)) - 5)) - 1)
        elif addr == HAWINDOW:
            base = self.hawindowsel * 32
            for i in range(32):
                if base + i < len(self.harts):
                    self.mask[base + i] = bool(value >> i & 1)
        elif addr == DMCS2:
            self._write_dmcs2(value)
        elif addr == SBCS:
            self._write_sbcs(value)
        elif addr in (SBADDRESS0, SBADDRESS1, SBADDRESS2, SBADDRESS3):
            if addr == SBADDRESS0:
                self.sbaddress = (self.sbaddress & ~M32) | value
                if self.sbreadonaddr and self._sb_ok():
                    self._sb_read()
            elif addr == SBADDRESS1 and self.sba_asize > 32:
                self.sbaddress = (self.sbaddress & M32) | value << 32
        elif addr in (SBDATA0, SBDATA1, SBDATA2, SBDATA3):
            k = (SBDATA0, SBDATA1, SBDATA2, SBDATA3).index(addr)
            self.sbdata[k] = value
            if k == 0 and self._sb_ok():
                self._sb_write()

    # -- dmcontrol / dmstatus ----------------------------------------------------------------

    def _write_dmcontrol(self, value: int) -> None:
        if not value & 1:                       # dmactive = 0: the DM resets
            self._reset_dm()
            return
        if not self.dmactive:                   # (activation: the other bits ignored)
            self.dmactive = True
            return
        sel = (value >> 16 & 0x3FF) | (value >> 6 & 0x3FF) << 10
        self.hartsel = sel & ((1 << self.hartsellen) - 1)
        self.hasel = bool(value >> 26 & 1) and len(self.harts) > 1
        haltreq, resumereq = value >> 31 & 1, value >> 30 & 1
        hartreset, ndmreset = value >> 29 & 1, value >> 1 & 1
        for h in self._selected():
            if value >> 28 & 1:                 # ackhavereset
                h.havereset = False
            if value >> 3 & 1 and not value >> 2 & 1:
                h.resethaltreq = True           # setresethaltreq
            if value >> 2 & 1:
                h.resethaltreq = False          # clrresethaltreq
            h.haltreq = bool(haltreq)
            if haltreq:
                self._halt(h)
            elif resumereq and h.m.debug_mode:  # (ignored while haltreq is set)
                h.resumeack = False
                self._resume(h)
        held = list(self.harts) if ndmreset else self._selected() if hartreset else []
        for h in self.harts:
            self._reset(h, h in held)
        self.hartreset, self.ndmreset = bool(hartreset), bool(ndmreset)

    def _dmstatus(self) -> int:
        if not self.dmactive:
            return 3                            # (version stays readable)
        exists = self._selected()
        nonexistent = self.hartsel >= len(self.harts)
        avail = [h for h in exists if self.available(h)]
        unavail = [h for h in exists if not self.available(h)]
        halted = [h for h in avail if h.m.debug_mode]
        running = [h for h in avail if not h.m.debug_mode]

        def all_any(flags: list[bool]) -> tuple[int, int]:
            return int(bool(flags) and all(flags)), int(any(flags))

        v = 3 | 1 << 7 | 1 << 5                 # version 1.0, authenticated, hasresethaltreq
        v |= self.impebreak << 22 | self.ndmreset << 24
        a, n = all_any([h.havereset for h in exists])
        v |= a << 19 | n << 18
        a, n = all_any([h.resumeack for h in exists])
        v |= a << 17 | n << 16
        v |= int(not exists and nonexistent) << 15 | int(nonexistent) << 14
        v |= int(bool(exists) and len(unavail) == len(exists)) << 13 | int(bool(unavail)) << 12
        v |= int(bool(exists) and len(running) == len(exists)) << 11 | int(bool(running)) << 10
        v |= int(bool(exists) and len(halted) == len(exists)) << 9 | int(bool(halted)) << 8
        return v

    def _haltsum(self, addr: int) -> int:
        """3.14.18-3.14.21: one bit per hart (haltsum0) or per group of 32 /
        1024 / 32768 harts, around hartsel's."""
        span = {HALTSUM0: 1, HALTSUM1: 32, HALTSUM2: 1024, HALTSUM3: 32768}[addr]
        base = (self.hartsel // (span * 32)) * span * 32
        v = 0
        for bit in range(32):
            lo = base + bit * span
            if any(self.harts[k].m.debug_mode and self.available(self.harts[k])
                   for k in range(lo, min(lo + span, len(self.harts)))):
                v |= 1 << bit
        return v

    def _write_dmcs2(self, value: int) -> None:
        """3.14.17: halt groups (grouptype 0) and resume groups (1); no DM
        external triggers (hgselect, dmexttrigger tied to 0)."""
        self.grouptype = value >> 11 & 1
        if value >> 1 & 1 and not value & 1:   # hgwrite, for harts
            group = value >> 2 & 0x1F
            for h in self._selected():
                if self.grouptype:
                    h.resumegroup = group
                else:
                    h.haltgroup = group

    # -- abstract commands (3.7) ---------------------------------------------------------------

    def _auto(self, bit: int) -> None:
        """abstractauto (3.14.8): an access to that data / progbuf word acts
        as if command were written again."""
        if bit and self.cmderr == ERR_NONE:
            self._execute()

    def _execute(self) -> None:
        cmd = self.command
        kind = cmd >> 24
        h = self._hart()
        if h is None or not self.available(h):
            self.cmderr = ERR_HALTRESUME
            return
        if kind == 0:
            self.cmderr = self._access_register(h, cmd)
        elif kind == 1:
            self.cmderr = self._quick_access(h, cmd)
        elif kind == 2:
            self.cmderr = self._access_memory(h, cmd)
        else:
            self.cmderr = ERR_NOTSUP

    def _arg(self, n: int, width: int) -> int:
        """Argument ``n`` of ``width`` bits (Table 2)."""
        words = width // 32
        return sum(self.data[n * words + k] << (32 * k) for k in range(words))

    def _set_arg(self, n: int, width: int, value: int) -> None:
        words = width // 32
        for k in range(words):
            self.data[n * words + k] = value >> (32 * k) & M32

    def _access_register(self, h: _Hart, cmd: int) -> int:
        """3.7.1.1: transfer regno <-> arg0, postincrement, postexec."""
        from .simrv import RVTrap
        m = h.m
        size, postinc = cmd >> 20 & 7, cmd >> 19 & 1
        postexec, transfer, write = cmd >> 18 & 1, cmd >> 17 & 1, cmd >> 16 & 1
        regno = cmd & 0xFFFF
        if cmd >> 23 & 1 or (postexec and not self.progbufsize):
            return ERR_NOTSUP
        if not m.debug_mode:
            return ERR_HALTRESUME
        if transfer:
            if size not in (2, 3):
                return ERR_NOTSUP
            width = 32 << (size - 2)
            if width // 32 > self.datacount:
                return ERR_NOTSUP
            if regno < 0x1000:                  # CSRs
                c = m.csrs
                if width > m.xlen:
                    return ERR_NOTSUP
                try:
                    c.check(regno, write=bool(write), insn=0)
                except RVTrap:
                    return ERR_EXCEPTION
                if write:
                    c.write(regno, self._sext(self._arg(0, width), width, m.xlen))
                else:
                    self._set_arg(0, width, c.read(regno))
            elif regno < 0x1020:                # GPRs
                if width > m.xlen:
                    return ERR_NOTSUP
                r = regno - 0x1000
                if write:
                    m.write_reg(r, self._sext(self._arg(0, width), width, m.xlen))
                else:
                    self._set_arg(0, width, m.read_reg(r))
            elif regno < 0x1040:                # FPRs
                if not m.flen:
                    return ERR_EXCEPTION        # (the register does not exist)
                if width > m.flen:
                    return ERR_NOTSUP
                r = regno - 0x1020
                if write:
                    v = self._arg(0, width)
                    m.fpr[r] = (v | ((1 << m.flen) - 1) ^ ((1 << width) - 1)
                                if width < m.flen else v)        # (NaN-boxed)
                else:
                    self._set_arg(0, width, m.fpr[r])
            else:
                return ERR_EXCEPTION            # (no such register)
            if postinc:
                self.command = (cmd & ~0xFFFF) | (regno + 1) & 0xFFFF
        if postexec:
            return self._run_progbuf(m)
        return ERR_NONE

    @staticmethod
    def _sext(v: int, width: int, xlen: int) -> int:
        """A narrower write to a wider register: sign-extended (Spike's lw)."""
        if width < xlen and v >> (width - 1) & 1:
            v -= 1 << width
        return v

    def _program(self) -> bytes:
        words = list(self.progbuf) + ([0x00100073] if self.impebreak else [])
        return b"".join(w.to_bytes(4, "little") for w in words)

    def _run_progbuf(self, m: "RVMachine") -> int:
        """3.8: the Program Buffer, until ebreak (or the implicit one); an
        exception ends it with cmderr 3, the hart staying in Debug Mode;
        running off the end without impebreak ends it too."""
        from .simrv import DebugModeTrap
        code = self._program()
        pos = 0
        while pos + 2 <= len(code):
            h = int.from_bytes(code[pos:pos + 2], "little")
            size = 4 if h & 3 == 3 else 2
            if pos + size > len(code):
                return ERR_EXCEPTION
            raw = int.from_bytes(code[pos:pos + size], "little")
            try:
                r = m.execute_debug(raw)
            except DebugModeTrap:
                return ERR_EXCEPTION
            if r is not None:                   # ebreak / dret
                return ERR_NONE
            pos += size
        return ERR_NONE

    def _quick_access(self, h: _Hart, cmd: int) -> int:
        """3.7.1.2: halt the running hart, run the Program Buffer, resume it
        (an exception: cmderr 3, the hart stays halted with cause 3)."""
        m = h.m
        if cmd & 0xFFFFFF or not self.progbufsize:
            return ERR_NOTSUP
        if m.debug_mode:
            return ERR_HALTRESUME
        m.enter_debug(CAUSE_HALTREQ)
        err = self._run_progbuf(m)
        if err != ERR_NONE:
            return err
        if m.debug_mode:
            m.resume()
        return ERR_NONE

    def _access_memory(self, h: _Hart, cmd: int) -> int:
        """3.7.1.3: arg0 <-> memory at arg1, physical or virtual (as M-mode
        with MPRV), postincrement; the hart's view and permissions."""
        from .simrv import RVTrap
        m = h.m
        virtual, size = cmd >> 23 & 1, cmd >> 20 & 7
        postinc, write = cmd >> 19 & 1, cmd >> 16 & 1
        if cmd >> 17 & 3 or cmd >> 14 & 3 or cmd & 0x3FFF:
            return ERR_NOTSUP
        if not m.debug_mode:
            return ERR_HALTRESUME
        width = m.xlen
        if size > (3 if m.xlen == 64 else 2) or self.datacount < 2 * width // 32:
            return ERR_NOTSUP
        n = 1 << size
        addr = self._arg(1, width)
        m.dm_access = "virtual" if virtual else "physical"
        try:
            if write:
                m.store(addr, self._arg(0, width) & ((1 << 8 * n) - 1), n)
            else:
                self._set_arg(0, width, m.load(addr, n))
        except RVTrap:
            return ERR_EXCEPTION
        finally:
            m.dm_access = None
        if postinc:
            self._set_arg(1, width, (addr + n) & ((1 << width) - 1))
        return ERR_NONE

    # -- System Bus Access (3.10, 3.14.22-30) ----------------------------------------------------

    @property
    def sba_asize(self) -> int:
        from .rvmmu import PA_BITS
        return PA_BITS if self.sba else 0

    def _sbcs(self) -> int:
        v = (1 << 29 if self.sba else 0) | self.sbbusyerror << 22 | self.sbreadonaddr << 20
        v |= self.sbcs_access << 17 | self.sbautoincrement << 16 | self.sbreadondata << 15
        v |= self.sberror << 12 | self.sba_asize << 5
        for k, bits in enumerate((8, 16, 32, 64, 128)):
            if self.sba >= bits:
                v |= 1 << k
        return v

    def _write_sbcs(self, value: int) -> None:
        if value >> 22 & 1:
            self.sbbusyerror = False            # R/W1C
        self.sbreadonaddr = bool(value >> 20 & 1)
        self.sbcs_access = value >> 17 & 7
        self.sbautoincrement = bool(value >> 16 & 1)
        self.sbreadondata = bool(value >> 15 & 1)
        self.sberror &= ~(value >> 12 & 7)      # R/W1C

    def _sb_ok(self) -> bool:
        return bool(self.sba) and not self.sberror and not self.sbbusyerror

    def _sb_target(self) -> tuple[int, int] | None:
        """(address, bytes) of the access, or None with sberror set: an
        unsupported size (4), a misaligned address (3), nothing there (2)."""
        from .rvmmu import PA_BITS
        n = 1 << self.sbcs_access
        if self.sbcs_access > 3 or 8 * n > self.sba:
            self.sberror = 4
            return None
        a = self.sbaddress
        if a % n:
            self.sberror = 3
            return None
        if a + n > 1 << PA_BITS:
            self.sberror = 2
            return None
        return a, n

    def _bus(self) -> "RVMachine":
        return self.harts[0].m

    def _sb_read(self) -> None:
        t = self._sb_target()
        if t is None:
            return
        a, n = t
        m = self._bus()
        for dev in m.devices:
            if dev.covers(a):
                v = dev.load(m, a, n)
                break
        else:
            v = int.from_bytes(bytes(m.phys_read_byte(a + k) for k in range(n)), "little")
        self.sbdata[0] = v & M32
        if n == 8:
            self.sbdata[1] = v >> 32
        self._sb_increment(n)

    def _sb_write(self) -> None:
        t = self._sb_target()
        if t is None:
            return
        a, n = t
        v = (self.sbdata[0] | self.sbdata[1] << 32) & ((1 << 8 * n) - 1)
        m = self._bus()
        for dev in m.devices:
            if dev.covers(a):
                dev.store(m, a, v, n)
                break
        else:
            for k in range(n):
                m.phys_write_byte(a + k, v >> (8 * k) & 0xFF)
        self._sb_increment(n)

    def _sb_increment(self, n: int) -> None:
        if self.sbautoincrement:
            self.sbaddress = (self.sbaddress + n) & ((1 << 64) - 1)

    # -- views -------------------------------------------------------------------------------

    def lines(self) -> list[str]:
        out = [f"Debug Module: {'active' if self.dmactive else 'inactive'}, hartsel "
               f"{self.hartsel}{' + mask' if self.hasel else ''}, cmderr {self.cmderr}, "
               f"progbuf {self.progbufsize}, data {self.datacount}"]
        for k, h in enumerate(self.harts):
            state = "unavailable (reset)" if h.in_reset else \
                "halted" if h.m.debug_mode else "running"
            flags = [n for n, b in (("haltreq", h.haltreq), ("resumeack", h.resumeack),
                                    ("resethaltreq", h.resethaltreq),
                                    ("havereset", h.havereset)) if b]
            out.append(f"hart {k}: {state}" + (f" [{', '.join(flags)}]" if flags else "")
                       + (f" halt group {h.haltgroup}" if h.haltgroup else "")
                       + (f" resume group {h.resumegroup}" if h.resumegroup else ""))
        return out


class JtagDTM:
    """The JTAG Debug Transport Module (Debug spec 6.1) of ``dm``."""

    def __init__(self, dm: DebugModule) -> None:
        self.dm = dm
        self.state = TLR
        self.ir = IR_IDCODE
        self.shift = 0
        self.length = 0
        self.dmi = 0                            # address | data | op (status)
        self.sticky = 0                         # a failed / busy op stays (6.1.5)

    def clock(self, tms: int, tdi: int = 0) -> int:
        """One TCK (as emips.tap.TAP.clock): the current state's action,
        then the move TMS selects.  Returns TDO."""
        st = self.state
        tdo = 0
        if st == TLR:
            self.ir = IR_IDCODE
        elif st == CAP_IR:
            self.shift, self.length = 0b00001, 5
        elif st == CAP_DR:
            self.shift, self.length = self._capture()
        elif st in (SHIFT_IR, SHIFT_DR) and self.length:
            tdo = self.shift & 1
            self.shift = (self.shift >> 1) | ((tdi & 1) << (self.length - 1))
        elif st == UPD_IR:
            self.ir = self.shift & 0x1F
        elif st == UPD_DR:
            self._update(self.shift)
        self.state = NEXT[st][tms & 1]
        if self.state == TLR:
            self.ir = IR_IDCODE
        return tdo

    def _dtmcs(self) -> int:
        return self.sticky << 10 | ABITS << 4 | 1      # dmistat, abits 7, version 1 (1.0)

    def _capture(self) -> tuple[int, int]:
        if self.ir == IR_IDCODE:
            return IDCODE, 32
        if self.ir == IR_DTMCS:
            return self._dtmcs(), 32
        if self.ir == IR_DMI:
            return self.dmi, ABITS + 34
        return 0, 1                             # BYPASS (every other instruction)

    def _update(self, value: int) -> None:
        if self.ir == IR_DTMCS:
            if value >> 16 & 1 or value >> 17 & 1:     # dmireset / dtmhardreset
                self.sticky = 0
                self.dmi &= ~3
            return
        if self.ir != IR_DMI or self.sticky:
            return
        op, data, addr = value & 3, value >> 2 & M32, value >> 34 & ((1 << ABITS) - 1)
        self.dmi = value & ~3
        if op == 1:
            self.dmi = addr << 34 | self.dm.dmi_read(addr) << 2
        elif op == 2:
            self.dm.dmi_write(addr, data)
        elif op == 3:                           # reserved: a failed operation
            self.sticky = 2
            self.dmi |= 2

    # -- a debugger's side ----------------------------------------------------------------------

    def reset_logic(self) -> None:
        for _ in range(5):
            self.clock(1)
        self.clock(0)

    def _scan(self, select: list[int], value: int, length: int) -> int:
        if self.state != IDLE:
            self.reset_logic()
        for tms in select:
            self.clock(tms)
        self.clock(0)
        out = 0
        for i in range(length):
            out |= self.clock(1 if i == length - 1 else 0, value >> i & 1) << i
        self.clock(1)
        self.clock(0)
        return out

    def scan_ir(self, instruction: int) -> int:
        return self._scan([1, 1, 0], instruction, 5)

    def scan_dr(self, value: int, length: int) -> int:
        return self._scan([1, 0], value, length)

    def dmi_access(self, addr: int, data: int = 0, op: int = 1) -> int:
        """One DMI operation through scans (a debugger's view): select dmi,
        scan the request, scan a nop to capture its result; returns the
        data (a read) — raises RuntimeError on a failed / busy status."""
        if self.ir != IR_DMI:
            self.scan_ir(IR_DMI)
        self.scan_dr(addr << 34 | (data & M32) << 2 | op, ABITS + 34)
        out = self.scan_dr(0, ABITS + 34)
        if out & 3:
            raise RuntimeError(f"DMI status {out & 3}")
        return out >> 2 & M32
