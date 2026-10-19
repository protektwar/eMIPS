"""The EJTAG Test Access Port and a probe (PLAN M14 phase 3).

References: *MIPS EJTAG Specification* (MD00047 rev. 5.06) chapter 4 "EJTAG
Test Access Port" and chapter 7 "PC Sampling"; the 24K's TAP (MD00343
11.3-11.6: Tables 11.22-11.27, Figure 11.20).

* :class:`TAP` — one per core.  A real IEEE 1149.1 controller: the 16
  states driven by TMS on every TCK (:meth:`TAP.clock`, Figure 4.2), a
  5-bit Instruction register (Capture-IR loads 00001), and the EJTAG
  data registers of the 24K: Device ID, Implementation, Address, Data,
  EJTAG Control (ECR), ALL, Fastdata, PCsample, Bypass (unused and TCB
  instructions select Bypass: no Trace Control Block).  :meth:`scan_ir`
  / :meth:`scan_dr` walk the states bit by bit as a probe does.
* Processor accesses (4.6.4, 11.5): a fetch / load / store to dmseg in
  Debug Mode with ECR.ProbEn = 1 latches the Address (its low bits per
  Psz) and Data (bytes as on the bus, by byte order) registers, sets
  PrAcc / PRnW / Psz and waits until the probe writes PrAcc = 0 — the
  probe (:class:`Probe`) is called then and services it through scans.
  An access no probe finishes waits forever (:class:`ProbeWaitError`).
* The ECR (Table 11.25): Rocc (set at a processor reset; until it is
  written 0 the ECR ignores updates), PrAcc (R/W0), ProbEn, ProbTrap
  (the debug vector moves to 0xFF200200 in dmseg), EjtagBrk (a debug
  interrupt), PrRst (resets the processor through ``on_reset``), PerRst
  (a pin with no device on it), Doze / Halt / DM (status).  EJTAGBOOT /
  NORMALBOOT set the reset value of EjtagBrk, ProbEn and ProbTrap; the
  indication survives processor resets, Test-Logic-Reset clears it.
* PC sampling (chapter 7, 24K 11.6): the EJTAG unit samples the PC of
  the graduating instruction every 2^(5 + DCR.PCR) cycles into PCsample
  (New, PC, ASID); in WAIT the New bit is set again when the probe
  clears it.
"""

from __future__ import annotations

from collections import deque

from .exceptions import ProbeWaitError
from . import pdtrace as PDT

M32 = 0xFFFFFFFF

# ------------------------------------------------------------ TAP states
TLR, IDLE = "Test-Logic-Reset", "Run-Test/Idle"
SEL_DR, CAP_DR, SHIFT_DR, EXIT1_DR, PAUSE_DR, EXIT2_DR, UPD_DR = (
    "Select-DR-Scan", "Capture-DR", "Shift-DR", "Exit1-DR", "Pause-DR", "Exit2-DR",
    "Update-DR")
SEL_IR, CAP_IR, SHIFT_IR, EXIT1_IR, PAUSE_IR, EXIT2_IR, UPD_IR = (
    "Select-IR-Scan", "Capture-IR", "Shift-IR", "Exit1-IR", "Pause-IR", "Exit2-IR",
    "Update-IR")
#: state -> (next with TMS = 0, next with TMS = 1) (MD00047 Figure 4.2)
NEXT = {
    TLR: (IDLE, TLR), IDLE: (IDLE, SEL_DR),
    SEL_DR: (CAP_DR, SEL_IR), CAP_DR: (SHIFT_DR, EXIT1_DR), SHIFT_DR: (SHIFT_DR, EXIT1_DR),
    EXIT1_DR: (PAUSE_DR, UPD_DR), PAUSE_DR: (PAUSE_DR, EXIT2_DR),
    EXIT2_DR: (SHIFT_DR, UPD_DR), UPD_DR: (IDLE, SEL_DR),
    SEL_IR: (CAP_IR, TLR), CAP_IR: (SHIFT_IR, EXIT1_IR), SHIFT_IR: (SHIFT_IR, EXIT1_IR),
    EXIT1_IR: (PAUSE_IR, UPD_IR), PAUSE_IR: (PAUSE_IR, EXIT2_IR),
    EXIT2_IR: (SHIFT_IR, UPD_IR), UPD_IR: (IDLE, SEL_DR),
}

# ------------------------------------------------------ TAP instructions
IR_LENGTH = 5
IDCODE, IMPCODE, ADDRESS, DATA, CONTROL, ALL = 0x01, 0x03, 0x08, 0x09, 0x0A, 0x0B
EJTAGBOOT, NORMALBOOT, FASTDATA, PCSAMPLE, BYPASS = 0x0C, 0x0D, 0x0E, 0x14, 0x1F
INSTRUCTIONS = {IDCODE: "IDCODE", IMPCODE: "IMPCODE", ADDRESS: "ADDRESS", DATA: "DATA",
                CONTROL: "CONTROL", ALL: "ALL", EJTAGBOOT: "EJTAGBOOT",
                NORMALBOOT: "NORMALBOOT", FASTDATA: "FASTDATA", PCSAMPLE: "PCSAMPLE",
                BYPASS: "BYPASS", **PDT.TCB_INSTRUCTIONS}
#: data register lengths (the rest: the 1-bit Bypass register)
DR_LENGTH = {IDCODE: 32, IMPCODE: 32, ADDRESS: 32, DATA: 32, CONTROL: 32, ALL: 96,
             FASTDATA: 33, PCSAMPLE: 41}

# ----------------------------------------------- EJTAG Control register
ECR_ROCC = 1 << 31
ECR_PSZ_SHIFT = 29
ECR_DOZE = 1 << 22
ECR_HALT = 1 << 21
ECR_PERRST = 1 << 20
ECR_PRNW = 1 << 19
ECR_PRACC = 1 << 18
ECR_PRRST = 1 << 16
ECR_PROBEN = 1 << 15
ECR_PROBTRAP = 1 << 14
ECR_ISAONDEBUG = 1 << 13
ECR_EJTAGBRK = 1 << 12
ECR_DM = 1 << 3

#: the Implementation register (24K Table 11.24): EJTAGver 3, DINTsup,
#: ASIDsize 2 with a TLB (bits 22:21), MIPS16e (bit 16) with the ASE, NoDMA
IMPCODE_BASE = (3 << 29) | (1 << 24) | (1 << 14)
IMPCODE_MIPS16 = 1 << 16
#: the Device ID register: Version, PartNumber and ManufID are core inputs
#: (EJ_Version / EJ_PartNumber / EJ_ManufID); eMIPS ties them to 0, bit 0 = 1
DEVICE_ID = 0x00000001

#: dmseg and its Fastdata area (0xFF200000 - 0xFF20000F)
DMSEG, DMSEG_SIZE = 0xFF200000, 0x100000
FASTDATA_END = 0xFF20000F


class TAP:
    """A core's EJTAG Test Access Port (see the module docstring)."""

    def __init__(self) -> None:
        self.m = None
        #: what the probe services processor accesses with (Probe.service)
        self.probe: "Probe | None" = None
        #: PrRst: a callable that resets the processor (the Session's)
        self.on_reset = None
        self.ejtagboot = False
        self.tck = 0
        #: the Trace Control Block (MIPS Trace, emips.pdtrace)
        self.tcb = PDT.TCB()
        self.tcb.on_change = self._tcb_changed
        self.power_up()

    # -- reset -------------------------------------------------------------------

    def power_up(self) -> None:
        """TRST* / power-up: Test-Logic-Reset, IDCODE, no EJTAGBOOT; the
        ECR as after a reset (Rocc = 1)."""
        self.state = TLR
        self.ir = IDCODE
        self.shift = 0
        self.length = 0
        self.ejtagboot = False
        self.tcb.power_up()
        self._ecr_reset()

    def _tcb_changed(self) -> None:
        if self.m is not None:
            self.m.trace_settings_changed()

    def _ecr_reset(self) -> None:
        self.rocc = True
        self.per_rst = self.pr_rst = False
        self.prob_en = self.prob_trap = self.ejtagbrk = self.ejtagboot
        #: ECR.ISAOnDebug as written (Table 4.9: reset 0; read-only 0
        #: without microMIPS, 1 on a microMIPS-only core)
        self.isa_on_debug = False
        self.pending: dict | None = None
        self.address = 0
        self.data = 0

    def processor_reset(self, m) -> None:
        """A processor reset (a new Machine for this core): the ECR's reset
        values — EjtagBrk, ProbEn, ProbTrap from the EJTAGBOOT indication,
        Rocc set, a pending processor access dropped (Table 11.25)."""
        self.m = m
        m.ejtag.tap = self
        m.pdt.tcb = self.tcb
        m.trace_settings_changed()
        self._ecr_reset()
        if self.ejtagbrk:                       # EJTAGBOOT: a debug interrupt at once
            m.debug_interrupt()

    # -- the controller ---------------------------------------------------------------

    def clock(self, tms: int, tdi: int = 0) -> int:
        """One TCK: the action of the current state on the rising edge, then
        the move TMS selects.  Returns TDO (the bit shifted out)."""
        self.tck += 1
        st = self.state
        tdo = 0
        if st == TLR:
            self.ir = IDCODE
            self.ejtagboot = False
            self.tcb.power_up()                 # 11.9.11: the TCB resets too
        elif st == CAP_IR:
            self.shift, self.length = 0b00001, IR_LENGTH
        elif st == CAP_DR:
            self.shift, self.length = self._capture(), self._dr_length()
        elif st in (SHIFT_IR, SHIFT_DR) and self.length:
            tdo = self.shift & 1
            self.shift = (self.shift >> 1) | ((tdi & 1) << (self.length - 1))
        elif st == UPD_IR:
            self._update_ir(self.shift & ((1 << IR_LENGTH) - 1))
        elif st == UPD_DR:
            self._update_dr(self.shift)
        self.state = NEXT[st][tms & 1]
        if self.state == TLR:
            self.ir = IDCODE
            self.ejtagboot = False
        return tdo

    def reset_logic(self) -> None:
        """Five TCKs with TMS = 1: Test-Logic-Reset from any state, then
        Run-Test/Idle."""
        for _ in range(5):
            self.clock(1)
        self.clock(0)

    def _scan(self, select: list[int], value: int, length: int) -> int:
        """From Run-Test/Idle: the select TMS sequence, Capture, ``length``
        Shift clocks (TDI = value LSB first), Exit1, Update, back to Idle."""
        if self.state != IDLE:
            self.reset_logic()
        for tms in select:
            self.clock(tms)
        self.clock(0)                           # Capture -> Shift
        out = 0
        for i in range(length):
            out |= self.clock(1 if i == length - 1 else 0, value >> i & 1) << i
        self.clock(1)                           # Exit1 -> Update
        self.clock(0)                           # Update -> Run-Test/Idle
        return out

    def scan_ir(self, instruction: int) -> int:
        return self._scan([1, 1, 0], instruction, IR_LENGTH)

    def scan_dr(self, value: int, length: int | None = None) -> int:
        return self._scan([1, 0], value, self._dr_length() if length is None else length)

    # -- the registers -----------------------------------------------------------------

    def _dr_length(self) -> int:
        if self.ir in PDT.TCB_INSTRUCTIONS:
            return self.tcb.length(self.ir)
        return DR_LENGTH.get(self.ir, 1)

    def ecr(self) -> int:
        """The EJTAG Control register as the probe reads it (Capture-DR)."""
        m = self.m
        v = 0
        if self.rocc:
            v |= ECR_ROCC
        p = self.pending
        if p is not None:
            v |= (p["psz"] << ECR_PSZ_SHIFT) | ECR_PRACC | (ECR_PRNW if p["write"] else 0)
        if m is not None:
            if m.waiting:
                v |= ECR_DOZE | ECR_HALT
            elif m.status & (1 << 27):
                v |= ECR_DOZE
            if m.debug_mode:
                v |= ECR_DM
            if self.ejtagbrk and m.dint_pending:
                v |= ECR_EJTAGBRK
        if self.per_rst:
            v |= ECR_PERRST
        if self.pr_rst:
            v |= ECR_PRRST
        if self.prob_en:
            v |= ECR_PROBEN
        if self.prob_trap:
            v |= ECR_PROBTRAP
        if m is not None and self.isa_on_debug_bit(m):
            v |= ECR_ISAONDEBUG
        return v

    def isa_on_debug_bit(self, m) -> bool:
        """ECR.ISAOnDebug (MD00047 Table 4.9): the debug handler's ISA in
        the probe's memory — 1 = microMIPS."""
        if getattr(m, "compressed", None) != "micromips":
            return False
        return self.isa_on_debug or not m.mips32

    def impcode(self) -> int:
        m = self.m
        tlb = m is not None and getattr(m.mmu, "mt", 3) in (1, 4)
        m16 = m is not None and getattr(m, "compressed", None) == "mips16e"
        return IMPCODE_BASE | ((2 << 21) if tlb else 0) | (IMPCODE_MIPS16 if m16 else 0)

    def pcsample(self) -> int:
        ej = self.m.ejtag if self.m is not None else None
        return ej.pc_sample if ej is not None else 0

    def _capture(self) -> int:
        ir = self.ir
        if ir == IDCODE:
            return DEVICE_ID
        if ir == IMPCODE:
            return self.impcode()
        if ir == ADDRESS:
            return self.address
        if ir == DATA:
            return self.data
        if ir == CONTROL:
            return self.ecr()
        if ir == ALL:
            return (self.address << 64) | (self.data << 32) | self.ecr()
        if ir == FASTDATA:
            return (self.data << 1) | int(self._fastdata_ok())
        if ir == PCSAMPLE:
            return self.pcsample()
        if ir in PDT.TCB_INSTRUCTIONS:
            return self.tcb.capture(ir)
        return 0                                # Bypass

    def _update_ir(self, value: int) -> None:
        self.ir = value
        if value == EJTAGBOOT:
            self.ejtagboot = True
        elif value == NORMALBOOT:
            self.ejtagboot = False

    def _update_dr(self, value: int) -> None:
        ir = self.ir
        if ir == DATA:
            self.data = value & M32
        elif ir == CONTROL:
            self._write_ecr(value & M32)
        elif ir == ALL:
            self.data = (value >> 32) & M32
            self._write_ecr(value & M32)
        elif ir == FASTDATA:
            self._fastdata(value)
        elif ir in PDT.TCB_INSTRUCTIONS:
            self.tcb.update(ir, value)
        elif ir == PCSAMPLE and self.m is not None and not value & 1:
            ej = self.m.ejtag
            if not self.m.waiting:              # in WAIT New is set again (7.2.1)
                ej.pc_sample &= ~1

    def _write_ecr(self, value: int) -> None:
        """Update-DR of the ECR: ignored while Rocc = 1 unless Rocc is
        written 0 (Table 11.25)."""
        if self.rocc and value & ECR_ROCC:
            return
        self.rocc = self.rocc and bool(value & ECR_ROCC)    # R/W0
        self.per_rst = bool(value & ECR_PERRST)
        self.prob_en = bool(value & ECR_PROBEN)
        self.prob_trap = bool(value & ECR_PROBTRAP)
        self.isa_on_debug = bool(value & ECR_ISAONDEBUG)
        m = self.m
        if value & ECR_EJTAGBRK and m is not None and not m.debug_mode:
            self.ejtagbrk = True                # R/W1: a debug interrupt
            m.debug_interrupt()
        if not value & ECR_PRACC and self.pending is not None:
            self.pending["done"] = True         # the processor access finishes
        if value & ECR_PRRST:
            self.pr_rst = True
            if self.on_reset is not None:       # the reset clears the bit again
                self.on_reset()

    # -- processor accesses (4.6.4; 24K 11.5) -------------------------------------------

    def _fastdata_ok(self) -> bool:
        p = self.pending
        return p is not None and DMSEG <= p["va"] <= FASTDATA_END

    def _fastdata(self, value: int) -> None:
        """Update-DR of Data + Fastdata (Table 11.27): SPrAcc = 0 completes
        a pending access to the Fastdata area (a download takes the data
        shifted in)."""
        if not value & 1 and self._fastdata_ok():
            if not self.pending["write"]:
                self.data = (value >> 1) & M32
            self.pending["done"] = True

    def processor_access(self, kind: str, va: int, size: int, value: int = 0) -> int:
        """A fetch / load / store to dmseg (Debug Mode): latch Address and
        Data, set PrAcc, let the probe service it.  Returns the value read
        (the bytes of the Data register at this access's lanes)."""
        if not self.prob_en:
            raise ProbeWaitError(
                f"{kind} at dmseg 0x{va:08x} with ECR.ProbEn = 0: no probe serves EJTAG "
                "memory, the access waits forever (MD00047 2.2.2.1)",
                name=None, word=None, addr=va)
        big = self.m.big_endian
        k = va & 3
        shift = 8 * (4 - size - k) if big else 8 * k
        if size == 1:
            psz, addr = 0, va
        elif size == 2:
            psz, addr = 1, va & ~1
        elif size == 4:
            psz, addr = 2, va & ~3
        else:                                   # 3 bytes (LWL / LWR / SWL / SWR): triple
            psz, addr = 3, (va & ~3) | (1 if k else 0)
        write = kind == "store"
        self.address = addr & M32
        if write:
            self.data = (value << shift) & M32
        self.pending = {"kind": kind, "va": va, "size": size, "psz": psz, "write": write,
                        "done": False}
        if self.probe is not None:
            self.probe.service(self)
        p, self.pending = self.pending, None
        if p is None or not p["done"]:
            what = {"fetch": "fetch from", "load": "load from", "store": "store to"}[kind]
            raise ProbeWaitError(
                f"{what} dmseg 0x{va:08x}: the probe did not finish the processor access "
                "(ECR.PrAcc stays 1), the core waits", name=None, word=None, addr=va)
        if write:
            return 0
        return (self.data >> shift) & ((1 << 8 * size) - 1)


class Probe:
    """An EJTAG probe on the TAPs of the cores (the debugger acting as
    one): it holds the EJTAG memory dmseg maps (1 MB), services processor
    accesses through TAP scans (poll the ECR, read Address, read / write
    Data, write PrAcc = 0; the Fastdata area through FASTDATA), and reads
    PC samples.  One probe serves every core's TAP."""

    def __init__(self, *, trap: bool = False) -> None:
        #: ProbTrap: the debug handler comes from the probe (0xFF200200)
        self.trap = trap
        #: the EJTAG memory: dmseg offset -> byte (what was loaded)
        self.image: dict[int, int] = {}
        self.mem: dict[int, int] = {}
        #: FASTDATA: words for downloads (processor loads), uploaded words
        self.download: deque[int] = deque()
        self.upload: list[int] = []
        #: the last processor accesses (kind, address, size, value), PC samples
        self.log: deque[tuple] = deque(maxlen=64)
        self.samples: list[tuple[int, int]] = []
        self.accesses = 0
        #: read PCsample after every step (profiling)
        self.profile = False

    def power_up(self) -> None:
        """A fresh run: the EJTAG memory as loaded, empty logs."""
        self.mem = dict(self.image)
        self.upload = []
        self.log.clear()
        self.samples = []
        self.accesses = 0

    def load(self, image: dict[int, int]) -> int:
        """Put the bytes of an assembled image that lie in dmseg into the
        EJTAG memory; returns how many."""
        n = 0
        for addr, byte in image.items():
            if DMSEG <= addr < DMSEG + DMSEG_SIZE:
                self.image[addr - DMSEG] = byte & 0xFF
                n += 1
        self.mem = dict(self.image)
        return n

    def connect(self, tap: TAP) -> None:
        """Take the TAP: acknowledge the reset (Rocc = 0), set ProbEn (and
        ProbTrap) — through a CONTROL scan."""
        tap.probe = self
        tap.scan_ir(CONTROL)
        tap.scan_dr(self._ecr_value(rocc=False))

    def _ecr_value(self, *, rocc: bool = True, pracc: bool = True, brk: bool = False) -> int:
        return ((ECR_ROCC if rocc else 0) | (ECR_PRACC if pracc else 0) | ECR_PROBEN
                | (ECR_PROBTRAP if self.trap else 0) | (ECR_EJTAGBRK if brk else 0))

    def debug_break(self, tap: TAP) -> None:
        """EjtagBrk: a debug interrupt through the ECR."""
        tap.scan_ir(CONTROL)
        tap.scan_dr(self._ecr_value(brk=True))

    def service(self, tap: TAP) -> None:
        """Poll the ECR; serve the pending processor access (MD00047
        4.6.4.1 / 4.6.4.2)."""
        tap.scan_ir(CONTROL)
        ecr = tap.scan_dr(self._ecr_value())             # PrAcc written 1: keep it pending
        if ecr & ECR_ROCC:                               # acknowledge a reset first
            tap.scan_dr(self._ecr_value(rocc=False))
            ecr = tap.scan_dr(self._ecr_value())
        if not ecr & ECR_PRACC:
            return
        write = bool(ecr & ECR_PRNW)
        psz = (ecr >> ECR_PSZ_SHIFT) & 3
        tap.scan_ir(ADDRESS)
        addr = tap.scan_dr(0)
        size, offsets = _psz_bytes(psz, addr)
        big = tap.m.big_endian
        va = tap.pending["va"]
        fast = DMSEG <= va <= FASTDATA_END
        if fast:                                         # FASTDATA (Table 11.27)
            tap.scan_ir(FASTDATA)
            if write:
                out = tap.scan_dr(0)                     # SPrAcc 0: complete
                self.upload.append((out >> 1) & M32)
                self.log.append(("store", va, size, (out >> 1) & M32, "fastdata"))
            else:
                if not self.download:
                    return                               # nothing to give: the core waits
                value = self.download.popleft() & M32
                tap.scan_dr(value << 1)
                self.log.append(("load", va, size, value, "fastdata"))
            self.accesses += 1
            return
        base = (addr & ~3) - DMSEG
        tap.scan_ir(DATA)
        if write:
            data = tap.scan_dr(0)
            for k in offsets:                            # Figure 11.20: the lanes by
                lane = 3 - k if big else k               # byte order
                self.mem[base + k] = (data >> 8 * lane) & 0xFF
            self.log.append(("store", va, size, data))
        else:
            data = 0                                     # unused lanes: 0 (11.4.3.1)
            for k in offsets:
                lane = 3 - k if big else k
                data |= self.mem.get(base + k, 0) << 8 * lane
            tap.scan_dr(data)
            self.log.append(("fetch" if tap.pending["kind"] == "fetch" else "load", va,
                             size, data))
        tap.scan_ir(CONTROL)
        tap.scan_dr(self._ecr_value(pracc=False))        # finish: PrAcc = 0
        self.accesses += 1

    def read_sample(self, tap: TAP) -> tuple[int, int] | None:
        """A PCSAMPLE scan: (PC, ASID) if New, clearing New."""
        tap.scan_ir(PCSAMPLE)
        v = tap.scan_dr(0)                               # New written 0
        if not v & 1:
            return None
        return (v >> 1) & M32, (v >> 33) & 0xFF


def _psz_bytes(psz: int, addr: int) -> tuple[int, tuple[int, ...]]:
    """Size and the bytes (offsets in the word) of a pending processor
    access, from Psz and Address[1:0] (Table 11.25)."""
    a = addr & 3
    if psz == 0:
        return 1, (a,)
    if psz == 1:
        return 2, (a, a + 1)
    if psz == 2:
        return 4, (0, 1, 2, 3)
    return 3, ((0, 1, 2) if a == 0 else (1, 2, 3))
