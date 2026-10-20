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
from typing import Callable

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


class SyscallServices:
    """MARS / SPIM syscall services (install as ``machine.syscall_handler``).

    ``read_line()`` supplies input lines (default: none -> empty input);
    ``write(text)`` receives output (default: appended to :attr:`output`).
    """

    HEAP_BASE = 0x10040000                    # MARS heap

    def __init__(self, *, read_line: Callable[[], str] | None = None,
                 write: Callable[[str], None] | None = None) -> None:
        self.read_line = read_line or (lambda: "")
        self._write = write
        self.output = ""
        self.heap = self.HEAP_BASE
        self.exit_code: int | None = None

    def write(self, text: str) -> None:
        self.output += text
        if self._write:
            self._write(text)

    def __call__(self, m, code: int) -> bool:
        v0 = m.read_reg(2)
        a0, a1 = m.read_reg(4), m.read_reg(5)
        s32 = lambda x: x - (1 << 32) if x & 0x80000000 else x   # noqa: E731
        if v0 == 1:                           # print_int
            self.write(str(s32(a0)))
        elif v0 == 2:                         # print_float ($f12)
            self.write(_fmt(struct.unpack(">f", (m.fpr[12] & M32).to_bytes(4, "big"))[0],
                            single=True))
        elif v0 == 3:                         # print_double ($f12/$f13)
            from .sim import fpr_get
            self.write(_fmt(struct.unpack(">d", fpr_get(m, 12, "D").to_bytes(8, "big"))[0]))
        elif v0 == 4:                         # print_string
            self.write(_cstring(m, a0))
        elif v0 == 5:                         # read_int
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
            self.heap = (self.heap + a0 + 3) & ~3
        elif v0 in (10, 17):                  # exit / exit2
            self.exit_code = s32(a0) if v0 == 17 else 0
            m.halted = True
        elif v0 == 11:                        # print_char
            self.write(chr(a0 & 0xFF))
        elif v0 == 12:                        # read_char
            line = self.read_line()
            m.write_reg(2, ord(line[0]) if line else 0)
        elif v0 == 34:                        # MARS: print hex
            self.write(f"0x{a0:08x}")
        elif v0 == 35:                        # MARS: print binary
            self.write(f"{a0:032b}")
        elif v0 == 36:                        # MARS: print unsigned
            self.write(str(a0))
        else:
            return False                      # not ours: take the exception
        return True


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
