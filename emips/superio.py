"""The SMSC FDC37M81x Super I/O chip on the Malta's ISA bus (PLAN M15
phase 2): its configuration ports and its two serial ports.

Reference: *FDC37M81x PC98/99 Compliant Enhanced Super I/O Controller*
(SMSC):
"Configuration" — the CONFIG / INDEX / DATA ports (3F0h / 3F1h with
SYSOPT = 0), the config keys 55h / AAh, Table 51 (the registers and
their reset values: after a hard reset every logical device is
disabled), logical devices 4 and 5 (serial ports 1 and 2: Activate 30h,
base 60h / 61h, IRQ 70h); the serial ports are 16550-compatible with
their IRQ enabled by MCR.OUT2.  The floppy, parallel and keyboard
devices are not modelled (their configuration registers are kept).

On the Malta the serial ports are tty0 (3F8h, IRQ 4) and tty1 (2F8h,
IRQ 3) — YAMON sets them up so (MD00048 Table 5.2; the boot firmware
does the same).
"""

from __future__ import annotations

from typing import Callable

from .uart16550 import UART16550

CONFIG_PORT = 0x3F0
DEVICE_ID = 0x4D
LD_SERIAL1, LD_SERIAL2 = 4, 5


class SuperIO:
    """See the module docstring.  ``set_irq(irq, level)``: the ISA IRQ
    lines (the PIIX4's interrupt controller)."""

    def __init__(self, cpu_hz: int) -> None:
        self.set_irq: Callable[[int, bool], None] | None = None
        self.uarts = [UART16550("tty0 (Super I/O serial port 1)", cpu_hz=cpu_hz),
                      UART16550("tty1 (Super I/O serial port 2)", cpu_hz=cpu_hz)]
        for i, u in enumerate(self.uarts):
            u.on_irq = lambda level, i=i: self._uart_irq(i, level)
        self.reset()

    def reset(self) -> None:
        """Hard reset (Table 51): Run state, every logical device off."""
        self.config_mode = False
        self.index = 0
        self.global_regs = {0x02: 0, 0x03: 0x03, 0x07: 0, 0x20: DEVICE_ID, 0x21: 0x01,
                            0x22: 0, 0x23: 0, 0x24: 0x04, 0x26: 0xF0, 0x27: 0x03}
        self.ld: dict[int, dict[int, int]] = {n: {0x30: 0, 0x60: 0, 0x61: 0, 0x70: 0}
                                              for n in range(9)}
        self.ld[0].update({0x60: 0x03, 0x61: 0xF0, 0x70: 0x06, 0x74: 0x02})
        self.ld[LD_SERIAL2][0xF1] = 0x02
        self.irq_levels = [False, False]
        for u in self.uarts:
            u.reset()

    # -- where the serial ports are --------------------------------------------------

    def _serial(self, i: int) -> tuple[int, int] | None:
        """(base, irq) of serial port ``i`` when it is activated."""
        regs = self.ld[LD_SERIAL1 + i]
        if not regs[0x30] & 1:
            return None
        return (regs[0x60] << 8 | regs[0x61]) & 0xFFF8, regs[0x70] & 0xF

    def _port_uart(self, port: int) -> tuple[UART16550, int] | None:
        for i in range(2):
            s = self._serial(i)
            if s is not None and s[0] <= port < s[0] + 8:
                return self.uarts[i], port - s[0]
        return None

    @property
    def config_port(self) -> int:
        return self.global_regs[0x27] << 8 | self.global_regs[0x26]

    def claims(self, port: int) -> bool:
        if port in (self.config_port, self.config_port + 1):
            return True
        return self._port_uart(port) is not None

    # -- the bus --------------------------------------------------------------------------

    def io_read(self, port: int) -> int:
        hit = self._port_uart(port)
        if hit is not None and not (self.config_mode and port in (self.config_port,
                                                                  self.config_port + 1)):
            u, reg = hit
            return u.read(reg)
        if not self.config_mode:
            return 0xFF
        if port == self.config_port:
            return self.index
        return self._config_read(self.index)

    def io_peek(self, port: int) -> int:
        hit = self._port_uart(port)
        if hit is not None:
            return hit[0].peek(hit[1])
        if self.config_mode and port == self.config_port + 1:
            return self._config_read(self.index)
        return 0xFF

    def io_write(self, port: int, value: int) -> None:
        if port == self.config_port:
            if not self.config_mode:
                if value == 0x55:                 # the config key
                    self.config_mode = True
                    return
            elif value == 0xAA:                   # exit: the Run state
                self.config_mode = False
                return
            else:
                self.index = value
                return
        if self.config_mode and port == self.config_port + 1:
            self._config_write(self.index, value)
            return
        hit = self._port_uart(port)
        if hit is not None:
            hit[0].write(hit[1], value)

    def _config_read(self, index: int) -> int:
        if index < 0x30:
            return self.global_regs.get(index, 0)
        return self.ld[self.global_regs[0x07] % 9].get(index, 0)

    def _config_write(self, index: int, value: int) -> None:
        if index < 0x30:
            if index in (0x20, 0x21):
                return                            # hard-wired
            if index == 0x02 and value & 1:       # soft reset: the logical devices
                for regs in self.ld.values():
                    regs[0x30] = 0
                return
            self.global_regs[index] = value
            return
        before = [self._serial(i) for i in range(2)]
        self.ld[self.global_regs[0x07] % 9][index] = value
        for i in range(2):                        # the IRQ moves with the port
            if before[i] != self._serial(i) and self.irq_levels[i] and before[i]:
                if self.set_irq is not None:
                    self.set_irq(before[i][1], False)
                self._uart_irq(i, self.uarts[i].level)

    def _uart_irq(self, i: int, level: bool) -> None:
        self.irq_levels[i] = level
        s = self._serial(i)
        if s is not None and s[1] and self.set_irq is not None:
            self.set_irq(s[1], level)

    # -- time -----------------------------------------------------------------------------

    def next_due(self) -> int | None:
        t = [d for d in (u.next_due() for u in self.uarts) if d is not None]
        return min(t) if t else None

    def advance(self, cycle: int) -> None:
        for u in self.uarts:
            u.advance(cycle)

    def configure(self, ld: int, base: int, irq: int) -> None:
        """What the firmware does (enter, select, base, IRQ, activate, exit)."""
        self.ld[ld].update({0x60: base >> 8, 0x61: base & 0xFF, 0x70: irq, 0x30: 1})
        i = ld - LD_SERIAL1
        if 0 <= i < 2:
            self._uart_irq(i, self.uarts[i].level)

    def describe(self) -> list[str]:
        out = []
        for i, u in enumerate(self.uarts):
            s = self._serial(i)
            where = f"at 0x{s[0]:03x}, IRQ {s[1]}" if s else "disabled (logical device off)"
            out.append(f"{u.describe()} — {where}")
        return out
