"""Console I/O for student programs: a memory-mapped terminal device and
MARS / SPIM ``syscall`` services.

**Memory-mapped terminal** (:class:`MmioConsole`) — the MARS "Keyboard and
Display MMIO Simulator" / SPIM terminal (Patterson & Hennessy, appendix A):

==========  ==================  =========================================
0xFFFF0000  receiver control    bit 0 Ready (a key is waiting), bit 1 IE
0xFFFF0004  receiver data       low byte: the key; reading it clears Ready
0xFFFF0008  transmitter control bit 0 Ready (can take a character), bit 1 IE
0xFFFF000C  transmitter data    writing the low byte displays it
==========  ==================  =========================================

After a character is written the transmitter is busy for ``tx_delay``
cycles (Ready = 0); a key fed with :meth:`MmioConsole.feed` arrives
``rx_delay`` cycles after the previous one was read.  With IE set, a ready
receiver raises hardware interrupt line 1 (Cause.IP3) and a ready
transmitter line 0 (Cause.IP2), level-triggered, as in SPIM.  The
registers are in kseg3, so they need kernel mode (``Program.load`` boots in
kernel mode).

**Syscall services** (:class:`SyscallServices`) — the SPIM / MARS
convention: service number in ``$v0``, arguments in ``$a0..$a3`` /
``$f12``, results in ``$v0`` / ``$f0``.  This is *OS emulation*, not
hardware: installed as ``machine.syscall_handler`` it services the
``SYSCALL`` instead of the exception; an unknown service is declined and
the real Sys exception is taken.
"""

from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .files import FileArea, FileTable
from . import dialogs as D
from .rvservices import CYCLES_PER_MS, JavaRandom, ServiceError

M32 = 0xFFFFFFFF


class MmioConsole:
    """MARS-style memory-mapped keyboard and display."""

    BASE = 0xFFFF0000
    RX_CTRL, RX_DATA, TX_CTRL, TX_DATA = 0, 1, 2, 3
    RX_IRQ, TX_IRQ = 1, 0                     # hardware lines (IP3 / IP2)

    def __init__(self, *, tx_delay: int = 6, rx_delay: int = 6,
                 on_output: Callable[[str], None] | None = None) -> None:
        self.tx_delay, self.rx_delay = tx_delay, rx_delay
        self.on_output = on_output
        #: everything displayed so far
        self.output = ""
        self._pending: deque[int] = deque()   # keys not yet delivered
        self.rx_ready = False
        self.rx_data = 0
        self.rx_ie = False
        self.tx_ready = True
        self.tx_ie = False
        self._rx_wait = 0
        self._tx_wait = 0
        #: the core whose interrupt lines the console drives and whose clock
        #: ticks it (the first attached; other cores only share the bus)
        self.owner = None
        #: a CPS: ``sink(line, level)`` — the lines are GIC sources (TX 0,
        #: RX 1, emips.gic) instead of the owner's pins
        self.sink = None
        self._levels = 0

    # -- wiring -------------------------------------------------------------

    def attach(self, machine) -> "MmioConsole":
        """Put the console on ``machine``'s bus and wire its interrupt
        lines to it (the owner: its clock ticks the console)."""
        machine.devices.append(self)
        self.owner = machine
        self._update_irq(machine)
        return self

    def share(self, machine) -> "MmioConsole":
        """Put the console on another core's bus too (a multi-core
        system): it reads and writes the same registers, but the interrupts
        go to the owner (until the GIC routes them, PLAN M11 phase 4)."""
        machine.devices.append(self)
        return self

    def feed(self, text: str) -> None:
        """Queue keystrokes (delivered one at a time)."""
        self._pending.extend(text.encode("latin-1", "replace"))

    def covers(self, addr: int) -> bool:
        return self.BASE <= addr < self.BASE + 16

    # -- bus ------------------------------------------------------------------

    def _reg(self, i: int) -> int:
        if i == self.RX_CTRL:
            return int(self.rx_ready) | (int(self.rx_ie) << 1)
        if i == self.RX_DATA:
            return self.rx_data
        if i == self.TX_CTRL:
            return int(self.tx_ready) | (int(self.tx_ie) << 1)
        return 0                              # transmitter data reads 0

    @staticmethod
    def _shift(machine, k: int) -> int:
        """Bit position of byte ``k`` (0 = lowest address) of a word."""
        return 8 * (3 - k) if machine.big_endian else 8 * k

    def load(self, machine, addr: int, size: int) -> int:
        value = 0
        touched_lsb = False
        for i in range(size):
            a = addr + i
            reg, k = (a - self.BASE) // 4, (a - self.BASE) & 3
            byte = (self._reg(reg) >> self._shift(machine, k)) & 0xFF
            value = (value << 8) | byte if machine.big_endian else value | (byte << (8 * i))
            if reg == self.RX_DATA and self._shift(machine, k) == 0:
                touched_lsb = True
        if touched_lsb and self.rx_ready:      # reading the key consumes it
            self.rx_ready = False
            self._rx_wait = self.rx_delay
            self._update_irq(machine)
        return value

    def store(self, machine, addr: int, value: int, size: int) -> None:
        for i in range(size):
            a = addr + i
            reg, k = (a - self.BASE) // 4, (a - self.BASE) & 3
            shift = 8 * (size - 1 - i) if machine.big_endian else 8 * i
            byte = (value >> shift) & 0xFF
            if self._shift(machine, k) != 0:
                continue                      # only the low byte matters
            if reg == self.RX_CTRL:
                self.rx_ie = bool(byte & 2)
            elif reg == self.TX_CTRL:
                self.tx_ie = bool(byte & 2)
            elif reg == self.TX_DATA and self.tx_ready:
                ch = chr(byte)
                self.output += ch
                if self.on_output:
                    self.on_output(ch)
                self.tx_ready = False
                self._tx_wait = self.tx_delay
        self._update_irq(machine)

    def tick(self, machine, cycles: int) -> None:
        if self.owner is not None and machine is not self.owner:
            return                            # one clock: the owner's
        if not self.tx_ready:
            self._tx_wait -= cycles
            if self._tx_wait <= 0:
                self.tx_ready = True
        if not self.rx_ready and self._pending:
            self._rx_wait -= cycles
            if self._rx_wait <= 0:
                self.rx_data = self._pending.popleft()
                self.rx_ready = True
        self._update_irq(machine)

    def _update_irq(self, machine) -> None:
        if self.sink is not None:
            for line, on in ((self.RX_IRQ, self.rx_ready and self.rx_ie),
                             (self.TX_IRQ, self.tx_ready and self.tx_ie)):
                if bool(self._levels >> line & 1) != bool(on):
                    self._levels ^= 1 << line
                    self.sink(line, bool(on))
            return
        if self.owner is not None:
            machine = self.owner
        lines = machine.irq_lines               # (every tick: usually nothing to change)
        if (bool(lines >> self.RX_IRQ & 1) == bool(self.rx_ready and self.rx_ie)
                and bool(lines >> self.TX_IRQ & 1) == bool(self.tx_ready and self.tx_ie)):
            return
        for line, on in ((self.RX_IRQ, self.rx_ready and self.rx_ie),
                         (self.TX_IRQ, self.tx_ready and self.tx_ie)):
            if on:
                if not machine.irq_lines >> line & 1:
                    machine.assert_irq(line)
            elif machine.irq_lines >> line & 1:
                machine.clear_irq(line)


@dataclass(frozen=True)
class MidiEvent:
    """One MIDI note (MARS 31 / 33, RARS 31 / 33): when (the core's cycle)
    and what — out-of-range values already replaced by ToneGenerator's
    defaults (pitch 60, 1000 ms, instrument 0, volume 100)."""
    cycle: int
    cpu: int
    pitch: int
    duration: int                             # ms
    instrument: int
    volume: int
    sync: bool                                # 33: the program waited for it

    @classmethod
    def of(cls, m, args: list[int], sync: bool) -> "MidiEvent":
        pitch, duration, instrument, volume = args
        ok = range(128)
        return cls(m.cycles + getattr(m, "clock_offset", 0), getattr(m, "cpu", 0),
                   pitch if pitch in ok else 60, duration if duration >= 0 else 1000,
                   instrument if instrument in ok else 0, volume if volume in ok else 100,
                   sync)


class SyscallServices:
    """MARS / SPIM syscall services (install as ``machine.syscall_handler``).

    ``read_line()`` supplies input lines (default: none -> empty input);
    ``write(text)`` receives output (default: appended to :attr:`output`).
    On a MIPS64 core (PLAN M22) the integer services use the whole 64-bit
    register: print_int (1) prints it signed, read_int (5) reads 64 bits,
    34 / 35 / 36 print 16 hex / 64 binary digits / the unsigned value.  On
    a nanoMIPS core (PLAN M23) the registers are the same numbers — the
    service and the result in $2 (p32's $t4), the arguments in $4-$6
    ($a0-$a2), as UHI keeps them on nanoMIPS (MD01069 ch. 2) — and the
    FPU services (2, 3, 6, 7, 43, 44) are not offered: it has no FPRs.

    MARS 4.5's time, sound and random services (PLAN M27 phase S1;
    MARS 4.5's ``mars/mips/instructions/syscalls/``), as RARS's are on RISC-V
    (``emips.rvservices``) and with the same choices, because a run must
    replay exactly (``sim.DEVIATIONS["MARS services"]``): time (30) counts
    the core's cycles at 1 MHz from 0 (MARS: the host's clock), sleep (32)
    lets that time pass, MIDI (31 / 33: pitch, duration, instrument,
    volume in $a0-$a3, each out of 0-127 replaced by ToneGenerator's
    default) is recorded in :attr:`midi` as an event at the core's cycle
    (:class:`MidiEvent`) and 33 waits its duration; the random streams
    (40-44) are ``java.util.Random`` — a seeded one draws what MARS draws,
    an unseeded one starts from seed 0 (MARS: the host's clock) — results
    in $a0 / $f0; a bound below 1 for 42 stops the program
    (:class:`~emips.rvservices.ServiceError`, MARS's ProcessingException).
    """

    HEAP_BASE = 0x10040000                    # MARS heap

    def __init__(self, *, read_line: Callable[[], str] | None = None,
                 write: Callable[[str], None] | None = None) -> None:
        self.read_line = read_line or (lambda: "")
        self._write = write
        self.output = ""
        self.heap = self.HEAP_BASE
        self.exit_code: int | None = None
        self.streams: dict[int, JavaRandom] = {}
        self.midi: list[MidiEvent] = []
        self.files = FileTable(FileArea())    # (the Session gives the program's files)
        #: the dialogs shown so far (emips.dialogs.Dialog; 50-59)
        self.dialogs: list = []
        #: ``on_dialog(dialog)``: told of each one as it is shown
        self.on_dialog = None

    def write(self, text: str) -> None:
        self.output += text
        if self._write:
            self._write(text)

    def _stream(self, index: int) -> JavaRandom:
        s = self.streams.get(index)
        if s is None:                         # (MARS: new Random(), seeded by the clock)
            s = self.streams[index] = JavaRandom(0)
        return s

    def __call__(self, m, code: int) -> bool:
        v0 = m.read_reg(2)
        if m.nano and v0 in (2, 3, 6, 7, 43, 44):   # nanoMIPS has no FPRs: an unknown service
            return False
        a0, a1 = m.read_reg(4), m.read_reg(5)
        s32 = lambda x: x - (1 << 32) if x & 0x80000000 else x   # noqa: E731
        wide = m.wide                         # MIPS64: 64-bit registers
        if v0 == 1:                           # print_int
            self.write(str(m.sgn(a0)))
        elif v0 == 2:                         # print_float ($f12)
            self.write(_fmt(struct.unpack(">f", (m.fpr[12] & M32).to_bytes(4, "big"))[0],
                            single=True))
        elif v0 == 3:                         # print_double ($f12/$f13)
            from .sim import fpr_get
            self.write(_fmt(struct.unpack(">d", fpr_get(m, 12, "D").to_bytes(8, "big"))[0]))
        elif v0 == 4:                         # print_string
            self.write(_cstring(m, a0))
        elif v0 == 5:                         # read_int
            if wide:
                m.write_dreg(2, _int(self.read_line()))
            else:
                m.write_reg(2, _int(self.read_line()) & M32)
        elif v0 == 6:                         # read_float -> $f0
            bits = struct.unpack(">I", struct.pack(">f", _float(self.read_line())))[0]
            m.fpr[0] = (m.fpr[0] & ~M32) | bits
        elif v0 == 7:                         # read_double -> $f0/$f1
            from .sim import fpr_set
            bits = struct.unpack(">Q", struct.pack(">d", _float(self.read_line())))[0]
            fpr_set(m, 0, "L", bits)
        elif v0 == 8:                         # read_string (a0 buffer, a1 size)
            # fgets semantics (SPIM / MARS): at most a1 - 1 characters, the
            # newline included when it fits, then a NUL
            if s32(a1) >= 1:
                line = self.read_line() + "\n"
                text = line[: s32(a1) - 1].encode("latin-1", "replace")
                for i, b in enumerate(text + b"\0"):
                    m.write_byte(a0 + i, b)
        elif v0 == 9:                         # sbrk
            m.write_reg(2, self.heap)
            self.heap = (self.heap + (a0 & M32) + 3) & ~3
        elif v0 in (10, 17):                  # exit / exit2
            self.exit_code = s32(a0) if v0 == 17 else 0
            m.halted = True
        elif v0 == 11:                        # print_char
            self.write(chr(a0 & 0xFF))
        elif v0 == 12:                        # read_char
            line = self.read_line()
            m.write_reg(2, ord(line[0]) if line else 0)
        elif v0 == 34:                        # MARS: print hex
            self.write(f"0x{a0:016x}" if wide else f"0x{a0:08x}")
        elif v0 == 35:                        # MARS: print binary
            self.write(f"{a0:064b}" if wide else f"{a0:032b}")
        elif v0 == 36:                        # MARS: print unsigned
            self.write(str(a0))
        elif v0 == 30:                        # time: ms in $a0 (low) / $a1 (high)
            ms = (m.cycles + getattr(m, "clock_offset", 0)) // CYCLES_PER_MS
            m.write_reg(4, ms & M32)
            m.write_reg(5, (ms >> 32) & M32)
        elif v0 == 32:                        # sleep $a0 ms
            if s32(a0 & M32) > 0:
                m.advance(s32(a0 & M32) * CYCLES_PER_MS)
        elif v0 in (31, 33):                  # MIDI out (33: synchronous)
            ev = MidiEvent.of(m, [s32(m.read_reg(r) & M32) for r in (4, 5, 6, 7)], v0 == 33)
            self.midi.append(ev)
            if ev.sync and ev.duration > 0:
                m.advance(ev.duration * CYCLES_PER_MS)
        elif v0 == 40:                        # random seed ($a0 stream, $a1 seed)
            s = self.streams.get(s32(a0 & M32))
            if s is None:
                self.streams[s32(a0 & M32)] = JavaRandom(s32(a1 & M32))
            else:
                s.set_seed(s32(a1 & M32))
        elif v0 == 41:                        # random int -> $a0
            m.write_reg(4, self._stream(s32(a0 & M32)).next_int() & M32)
        elif v0 == 42:                        # random int in [0, $a1) -> $a0
            bound = s32(a1 & M32)
            if bound <= 0:
                raise ServiceError("Upper bound of range cannot be negative (syscall 42)")
            m.write_reg(4, self._stream(s32(a0 & M32)).next_int(bound))
        elif v0 == 43:                        # random float -> $f0
            bits = struct.unpack(">I", struct.pack(">f", self._stream(s32(a0 & M32))
                                                   .next_float()))[0]
            m.fpr[0] = (m.fpr[0] & ~M32) | bits
        elif v0 == 44:                        # random double -> $f0 / $f1
            from .sim import fpr_set
            bits = struct.unpack(">Q", struct.pack(">d", self._stream(s32(a0 & M32))
                                                   .next_double()))[0]
            fpr_set(m, 0, "L", bits)
        elif v0 == 13:                        # open ($a0 name, $a1 flags) -> $v0 fd
            m.write_reg(2, self.files.open(_cstring(m, a0), s32(a1 & M32)) & M32)
        elif v0 == 14:                        # read ($a0 fd, $a1 buffer, $a2 length)
            m.write_reg(2, self._read_fd(m, s32(a0 & M32), a1,
                                         s32(m.read_reg(6) & M32)) & M32)
        elif v0 == 15:                        # write ($a0 fd, $a1 buffer, $a2 length)
            m.write_reg(2, self._write_fd(m, s32(a0 & M32), a1,
                                          s32(m.read_reg(6) & M32)) & M32)
        elif v0 == 16:                        # close ($a0 fd)
            self.files.close(s32(a0 & M32))
        elif 50 <= v0 <= 59:                  # the dialogs (emips.dialogs)
            if m.nano and v0 in D.FLOAT_SERVICES:
                return False
            self._dialog(m, v0, a0, s32(a1 & M32))
        else:
            return False                      # not ours: take the exception
        return True

    def _show(self, d) -> None:
        self.dialogs.append(d)
        if self.on_dialog is not None:
            self.on_dialog(d)

    def peek_dialog(self, m):
        """The input dialog the next syscall would show, as (service, kind,
        text), or None (a view: no side effects)."""
        v0 = m.read_reg(2) & M32
        if v0 not in D.INPUT or m.nano and v0 in D.FLOAT_SERVICES:
            return None
        return v0, D.INPUT[v0], _cstring(m, m.read_reg(4))

    def _dialog(self, m, v0: int, a0: int, a1: int) -> None:
        """MARS 4.5's dialogs (SyscallConfirmDialog ... MessageDialogString)."""
        from .rvservices import java_string
        from .sim import fpr_get, fpr_set
        text = _cstring(m, a0)
        if v0 not in D.INPUT:                 # 55-59: shown, nothing returned
            icon = D.ICONS.get(a1, "plain") if v0 == 55 else "information"
            if v0 == 56:
                text += str(a1)
            elif v0 == 57:
                text += java_string(struct.unpack(">f", (m.fpr[12] & M32).to_bytes(4, "big"))[0],
                                    single=True)
            elif v0 == 58:
                text += java_string(struct.unpack(">d", fpr_get(m, 12, "D").to_bytes(8, "big"))[0])
            elif v0 == 59:
                text += _cstring(m, m.read_reg(5))
            self._show(D.Dialog(v0, "message", text, icon))
            return
        answer = self.read_line()             # (InputNeeded: the window asks)
        self._show(D.Dialog(v0, D.INPUT[v0], text, answer=answer))
        set_a0 = (lambda v: m.write_reg(4, v & M32))
        set_a1 = (lambda v: m.write_reg(5, v & M32))
        if v0 == 50:
            set_a0(D.confirm_value(answer, rars=False))
            return
        if v0 == 51:
            if answer == D.CANCEL:
                set_a0(0), set_a1(-2)
            elif answer == "":
                set_a0(0), set_a1(-3)
            else:
                try:
                    set_a0(D.parse_int(answer)), set_a1(0)
                except ValueError:
                    set_a0(0), set_a1(-1)
            return
        if v0 in (52, 53):
            single = v0 == 52

            def put(x: float) -> None:
                if single:
                    bits = struct.unpack(">I", struct.pack(">f", x))[0]
                    m.fpr[0] = (m.fpr[0] & ~M32) | bits
                else:
                    fpr_set(m, 0, "L", struct.unpack(">Q", struct.pack(">d", x))[0])
            put(0.0)
            if answer == D.CANCEL:
                set_a1(-2)
            elif answer == "":
                set_a1(-3)
            else:
                try:
                    put(D.parse_float(answer)), set_a1(0)
                except (ValueError, OverflowError):
                    set_a1(-1)
            return
        size = m.read_reg(6) & M32                            # 54: string
        buf, size = m.read_reg(5), size - (1 << 32) if size >> 31 else size
        if answer == D.CANCEL:
            set_a1(-2)
        elif answer == "":
            set_a1(-3)
        else:
            stores, flag = D.string_bytes(bytes(ord(c) & 0xFF for c in answer), size, False)
            for off, b in stores:
                m.write_byte((buf + off) & m.amask, b)
            set_a1(flag)

    def _read_fd(self, m, fd: int, buf: int, length: int) -> int:
        """MARS readFromFile: the bytes read, 0 at the end, -1 (not open for
        reading, a negative length); fd 0 = the console's next input line."""
        if length < 0:
            return -1
        if fd == 0:
            data = (self.read_line() + "\n").encode("latin-1", "replace")[:length]
        else:
            data = self.files.read(fd, length)
            if data is None:
                return -1
        for i, b in enumerate(data):
            m.write_byte((buf + i) & m.amask, b)
        return len(data)

    def _write_fd(self, m, fd: int, buf: int, length: int) -> int:
        """MARS writeToFile: the length written or -1; fd 1 / 2 = the console."""
        if length < 0:
            return -1
        data = bytes(m.read_byte((buf + i) & m.amask) for i in range(length))
        if fd in (1, 2):
            self.write(data.decode("latin-1"))
            return length
        return self.files.write(fd, data)


def _cstring(m, addr: int, limit: int = 1 << 16) -> str:
    out = bytearray()
    for i in range(limit):
        b = m.read_byte(addr + i)
        if b == 0:
            break
        out.append(b)
    return out.decode("latin-1")


def _int(text: str) -> int:
    try:
        return int(text.strip() or "0", 0)
    except ValueError:
        return 0


def _float(text: str) -> float:
    try:
        return float(text.strip() or "0")
    except ValueError:
        return 0.0


def _fmt(x: float, single: bool = False) -> str:
    """Shortest text that reads back as the same value (as MARS prints)."""
    if x != x:
        return "NaN"
    if x in (float("inf"), float("-inf")):
        return "Infinity" if x > 0 else "-Infinity"
    if single:
        bits = struct.pack(">f", x)
        for digits in range(1, 10):
            text = f"{x:.{digits}g}"
            if struct.pack(">f", float(text)) == bits:
                break
    else:
        text = repr(x)
    if "e" not in text and "." not in text and "inf" not in text:
        text += ".0"
    return text
