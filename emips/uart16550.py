"""A 16550 UART (PLAN M15 phase 2): the Malta's two Super I/O serial ports
and its CBUS UART (tty2).

Reference: *PC16550D Universal Asynchronous Receiver/Transmitter with
FIFOs* (National 1995 / TI SNLS378C):
the registers (8.6.1, Tables 1-3: reset values), the FIFOs and their
trigger levels, the interrupts in priority order (receiver line status,
received data / character timeout, transmitter holding register empty,
modem status), the character timeout (4 character times), loopback; the
SMSC FDC37M81x datasheet for how a PC-style port gates its IRQ with
MCR.OUT2.

Time is the core's cycles: ``cycles_per_bit`` = the CPU clock / (the
UART clock / 16 / the divisor).  A character takes start + data + parity
+ stop bits; it leaves the transmitter (``on_tx``) when its last bit is
sent, and a character typed on the terminal (:meth:`feed`) arrives one
character time after the previous one.  The terminal on the other end
asserts DCD, DSR and CTS (it is connected and ready); RI stays off.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

# register numbers (address bits A2:A0)
RBR = THR = DLL = 0
IER = DLM = 1
IIR = FCR = 2
LCR, MCR, LSR, MSR, SCR = 3, 4, 5, 6, 7
# IER
IER_RDA, IER_THRE, IER_RLS, IER_MS = 1, 2, 4, 8
# IIR identifications (bits 3:0)
IIR_NONE, IIR_RLS, IIR_RDA, IIR_TIMEOUT, IIR_THRE, IIR_MS = 0x1, 0x6, 0x4, 0xC, 0x2, 0x0
# FCR
FCR_ENABLE, FCR_RX_RESET, FCR_TX_RESET = 1, 2, 4
TRIGGERS = (1, 4, 8, 14)
# LCR
LCR_DLAB, LCR_BREAK, LCR_PEN, LCR_STB = 0x80, 0x40, 0x08, 0x04
# MCR
MCR_DTR, MCR_RTS, MCR_OUT1, MCR_OUT2, MCR_LOOP = 1, 2, 4, 8, 16
# LSR
LSR_DR, LSR_OE, LSR_PE, LSR_FE, LSR_BI, LSR_THRE, LSR_TEMT, LSR_FIFOERR = \
    1, 2, 4, 8, 16, 32, 64, 128
# MSR
MSR_DCTS, MSR_DDSR, MSR_TERI, MSR_DDCD, MSR_CTS, MSR_DSR, MSR_RI, MSR_DCD = \
    1, 2, 4, 8, 16, 32, 64, 128
FIFO_SIZE = 16


class UART16550:
    """One 16550 (see the module docstring).  ``clock``: its crystal (Hz);
    ``cpu_hz``: the core's clock; ``gate_out2``: the IRQ output is enabled
    by MCR.OUT2 (a PC-style port); ``on_irq(level)``; ``on_tx(char)``."""

    def __init__(self, name: str, *, clock: int = 1843200, cpu_hz: int = 100_000_000,
                 gate_out2: bool = True) -> None:
        self.name = name
        self.clock = clock
        self.cpu_hz = cpu_hz
        self.gate_out2 = gate_out2
        self.on_irq: Callable[[bool], None] | None = None
        self.on_tx: Callable[[str], None] | None = None
        self.pending: deque[int] = deque()      # typed, not arrived yet
        self.output = ""                        # everything transmitted
        self.reset()

    def reset(self) -> None:
        """Master reset (Table 3); the divisor and the scratch register
        are not reset (undefined: eMIPS keeps 0)."""
        self.ier = 0
        self.fcr = 0
        self.lcr = 0
        self.mcr = 0
        self.lsr = LSR_THRE | LSR_TEMT
        self.msr_delta = 0
        self.dll = getattr(self, "dll", 0)
        self.dlm = getattr(self, "dlm", 0)
        self.scr = getattr(self, "scr", 0)
        self.rx: deque[int] = deque()
        self.rx_err: deque[int] = deque()        # LSR error bits per received byte
        self.tx: deque[int] = deque()
        self.shift: int | None = None            # the character being sent
        self.now = 0
        self.tx_done = 0                         # when the shift register empties
        self.rx_next = 0                         # when the next typed character may arrive
        self.rx_last = 0                         # last receive / read (character timeout)
        self.thre_pending = False                # THRE interrupt (cleared by IIR read / THR write)
        self.timeout = False
        self.level = False
        self._update()

    # -- timing ---------------------------------------------------------------------

    @property
    def divisor(self) -> int:
        return self.dll | self.dlm << 8

    def bits_per_char(self) -> int:
        data = 5 + (self.lcr & 3)
        stop = 1 if not self.lcr & LCR_STB else (2 if data > 5 else 1)   # 1.5: rounded down
        return 1 + data + (1 if self.lcr & LCR_PEN else 0) + stop

    def char_cycles(self) -> int | None:
        """Cycles one character takes on the line (None: no baud clock)."""
        if self.divisor == 0:
            return None
        return max(1, (self.bits_per_char() * 16 * self.divisor * self.cpu_hz) // self.clock)

    @property
    def baud(self) -> float:
        return self.clock / 16 / self.divisor if self.divisor else 0.0

    def next_due(self) -> int | None:
        """The next cycle anything happens by itself (None: nothing)."""
        t = []
        if (self.shift is not None or self.tx) and self.char_cycles() is not None:
            t.append(self.tx_done)                # (no baud clock: nothing moves)
        if self.pending and self.char_cycles() is not None:
            t.append(self.rx_next)
        if self.rx and self.fcr & FCR_ENABLE and not self.timeout and self.char_cycles():
            t.append(self.rx_last + 4 * self.char_cycles())
        return min(t) if t else None

    def advance(self, now: int) -> None:
        """Run the UART up to cycle ``now``."""
        step = self.char_cycles()
        while True:
            due = self.next_due()
            if due is None or due > now:
                break
            self.now = due
            if (self.shift is not None or self.tx) and self.tx_done <= due:
                self._tx_step(step)
            elif self.pending and step is not None and self.rx_next <= due:
                self._receive(self.pending.popleft())
                self.rx_next = due + step
            elif self.rx and self.fcr & FCR_ENABLE and not self.timeout and step:
                self.timeout = True
                self._update()
        self.now = now

    def _tx_step(self, step: int | None) -> None:
        """The shift register finished a character: send it and load the
        next one."""
        if self.shift is not None:
            ch = self.shift
            self.shift = None
            if self.mcr & MCR_LOOP:              # loopback: back into the receiver
                self._receive(ch)
            else:
                text = chr(ch & self._data_mask())
                self.output += text
                if self.on_tx is not None:
                    self.on_tx(text)
        if self.tx and step is not None:
            self._load_shift(step)
        else:
            self.lsr |= LSR_TEMT
        self._update()

    def _load_shift(self, step: int) -> None:
        self.shift = self.tx.popleft()
        self.tx_done = self.now + step
        if not self.tx:                          # the holding register / FIFO empties
            self.lsr |= LSR_THRE
            self.thre_pending = True

    def _data_mask(self) -> int:
        return (1 << (5 + (self.lcr & 3))) - 1

    # -- the terminal ---------------------------------------------------------------

    def feed(self, data: bytes) -> None:
        """Characters typed on the terminal (they arrive at the line rate)."""
        if not self.pending:
            self.rx_next = max(self.rx_next, self.now + (self.char_cycles() or 0))
        self.pending.extend(data)

    def _receive(self, byte: int) -> None:
        cap = FIFO_SIZE if self.fcr & FCR_ENABLE else 1
        if len(self.rx) >= cap:                  # overrun: the new character is lost
            self.lsr |= LSR_OE
        else:
            self.rx.append(byte & self._data_mask())
            self.rx_err.append(0)
            self.lsr |= LSR_DR
        self.rx_last = self.now
        self.timeout = False
        self._update()

    # -- the registers --------------------------------------------------------------

    def read(self, reg: int) -> int:
        reg &= 7
        if self.lcr & LCR_DLAB and reg in (DLL, DLM):
            return self.dll if reg == DLL else self.dlm
        if reg == RBR:
            value = self.rx.popleft() if self.rx else 0
            if self.rx_err:
                self.rx_err.popleft()
            if not self.rx:
                self.lsr &= ~LSR_DR
            self.rx_last = self.now
            self.timeout = False
            self._update()
            return value
        if reg == IER:
            return self.ier
        if reg == IIR:
            iid = self._iid()
            if iid == IIR_THRE:                  # reading IIR clears a THRE interrupt
                self.thre_pending = False
                self._update()
            return iid | (0xC0 if self.fcr & FCR_ENABLE else 0)
        if reg == LCR:
            return self.lcr
        if reg == MCR:
            return self.mcr
        if reg == LSR:
            value = self.lsr
            if self.fcr & FCR_ENABLE and any(self.rx_err):
                value |= LSR_FIFOERR
            self.lsr &= ~(LSR_OE | LSR_PE | LSR_FE | LSR_BI)   # cleared by reading LSR
            self._update()
            return value
        if reg == MSR:
            value = self._msr_inputs() | self.msr_delta
            self.msr_delta = 0
            self._update()
            return value
        return self.scr

    def peek(self, reg: int) -> int:
        """A register as a view sees it (no side effects)."""
        reg &= 7
        if self.lcr & LCR_DLAB and reg in (DLL, DLM):
            return self.dll if reg == DLL else self.dlm
        return {RBR: self.rx[0] if self.rx else 0, IER: self.ier,
                IIR: self._iid() | (0xC0 if self.fcr & FCR_ENABLE else 0),
                LCR: self.lcr, MCR: self.mcr, LSR: self.lsr,
                MSR: self._msr_inputs() | self.msr_delta, SCR: self.scr}[reg]

    def write(self, reg: int, value: int) -> None:
        reg &= 7
        value &= 0xFF
        if self.lcr & LCR_DLAB and reg in (DLL, DLM):
            if reg == DLL:
                self.dll = value
            else:
                self.dlm = value
            return
        if reg == THR:
            cap = FIFO_SIZE if self.fcr & FCR_ENABLE else 1
            if len(self.tx) < cap:               # a full FIFO drops the character
                self.tx.append(value)
            self.lsr &= ~(LSR_THRE | LSR_TEMT)
            self.thre_pending = False
            if self.shift is None and self.char_cycles() is not None:
                self._load_shift(self.char_cycles())
        elif reg == IER:
            old = self.ier
            self.ier = value & 0x0F
            if self.ier & IER_THRE and not old & IER_THRE and self.lsr & LSR_THRE:
                self.thre_pending = True         # enabling THRE with it empty interrupts
        elif reg == FCR:
            if (value ^ self.fcr) & FCR_ENABLE:  # changing the mode empties both FIFOs
                self.rx.clear(), self.rx_err.clear(), self.tx.clear()
            if value & FCR_RX_RESET:
                self.rx.clear()
                self.rx_err.clear()
                self.lsr &= ~LSR_DR
            if value & FCR_TX_RESET:
                self.tx.clear()
                self.lsr |= LSR_THRE
            self.fcr = value & 0xC9
        elif reg == LCR:
            self.lcr = value
        elif reg == MCR:
            old = self._msr_inputs()
            self.mcr = value & 0x1F
            self._modem_changed(old)
        elif reg == SCR:
            self.scr = value
        self._update()

    def _msr_inputs(self) -> int:
        """CTS / DSR / RI / DCD: the terminal (connected, ready), or in
        loopback the outputs RTS / DTR / OUT1 / OUT2."""
        if self.mcr & MCR_LOOP:
            m = self.mcr
            return ((MSR_CTS if m & MCR_RTS else 0) | (MSR_DSR if m & MCR_DTR else 0)
                    | (MSR_RI if m & MCR_OUT1 else 0) | (MSR_DCD if m & MCR_OUT2 else 0))
        return MSR_CTS | MSR_DSR | MSR_DCD

    def _modem_changed(self, old: int) -> None:
        new = self._msr_inputs()
        diff = old ^ new
        if diff & MSR_CTS:
            self.msr_delta |= MSR_DCTS
        if diff & MSR_DSR:
            self.msr_delta |= MSR_DDSR
        if diff & MSR_DCD:
            self.msr_delta |= MSR_DDCD
        if old & MSR_RI and not new & MSR_RI:    # trailing edge of RI
            self.msr_delta |= MSR_TERI

    # -- the interrupt ----------------------------------------------------------------

    def _iid(self) -> int:
        """The highest-priority pending interrupt (Table 4)."""
        ier = self.ier
        if ier & IER_RLS and self.lsr & (LSR_OE | LSR_PE | LSR_FE | LSR_BI):
            return IIR_RLS
        if ier & IER_RDA:
            if self.fcr & FCR_ENABLE:
                if len(self.rx) >= TRIGGERS[self.fcr >> 6]:
                    return IIR_RDA
                if self.timeout and self.rx:
                    return IIR_TIMEOUT
            elif self.rx:
                return IIR_RDA
        if ier & IER_THRE and self.thre_pending:
            return IIR_THRE
        if ier & IER_MS and self.msr_delta:
            return IIR_MS
        return IIR_NONE

    def _update(self) -> None:
        level = self._iid() != IIR_NONE
        if self.gate_out2 and not self.mcr & MCR_OUT2:
            level = False                         # a PC port: OUT2 enables the IRQ
        if level != self.level:
            self.level = level
            if self.on_irq is not None:
                self.on_irq(level)

    def describe(self) -> str:
        parity = "n" if not self.lcr & LCR_PEN else ("e" if self.lcr & 0x10 else "o")
        stop = 2 if self.lcr & LCR_STB else 1
        rate = f"{self.baud:.0f} baud" if self.divisor else "no baud clock (divisor 0)"
        return (f"{self.name}: {rate}, {5 + (self.lcr & 3)}{parity}{stop}, "
                f"IER 0x{self.ier:02x}, LSR 0x{self.lsr:02x}, MCR 0x{self.mcr:02x}, "
                f"FIFO {'on' if self.fcr & FCR_ENABLE else 'off'}, rx {len(self.rx)}, "
                f"tx {len(self.tx) + (self.shift is not None)}, IRQ {'on' if self.level else 'off'}")
