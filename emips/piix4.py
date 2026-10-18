"""The Intel 82371AB PIIX4 south bridge of the Malta (PLAN M15 phase 2).

Reference: *82371AB PCI-TO-ISA / IDE XCELERATOR (PIIX4)* datasheet
(Intel 290562-001, 1997): 4.2.2 the interrupt
controller registers (ICW / OCW, ELCR1 / ELCR2 at 4D0h / 4D1h), 4.2.3
the timer, 4.2.4 NMISC (61h), 4.2.5 the RTC index / data ports (70h-73h),
8.6 the two cascaded 82C59s, 8.8 the 82C54 timer / counters, 8.9 the
real-time clock (an MC146818A: the Motorola datasheet in the same folder
for its registers); the PCI configuration header of each function.

ISA I/O ports reach the PIIX4 through the GT-64120's PCI I/O window
(0x18000000 + port on a Malta set up by YAMON).  It claims every ISA
port subtractively: one no ISA device answers reads 0xFF (the floating
bus).  Its interrupt controller's INTR is the Malta's CPU INT0; the CPU
gets the vector with a PCI interrupt acknowledge cycle (the GT-64120's
IACK register, 0xC34 — as Linux's ``mips_pcibios_iack`` does) or with
an OCW3 poll.

Time comes from the core's cycles: the timer counts at 1.193182 MHz, the
RTC at 32.768 kHz, the core at the board's clock (``cpu_hz``).
"""

from __future__ import annotations

import datetime
from typing import Callable

from . import ata as ATA
from .busmaster import BusMasterChannel

#: the core's clock by default on the eMIPS Malta (an eMIPS choice: 10 MHz,
#: so that serial characters and timer ticks come within a few thousand
#: instructions; the board spec's MHZ field changes it)
CPU_HZ = 10_000_000
PIT_HZ = 1_193_182
RTC_HZ = 32_768
#: the RTC's date when the board is powered (an eMIPS choice: deterministic)
RTC_EPOCH = datetime.datetime(2026, 1, 1, 0, 0, 0)


def ticks(cycles: int, hz: int, cpu_hz: int = CPU_HZ) -> int:
    """Whole ``hz`` clock ticks in ``cycles`` core cycles."""
    return cycles * hz // cpu_hz


def cycles_for(t: int, hz: int, cpu_hz: int = CPU_HZ) -> int:
    """The first core cycle at which ``t`` ticks of ``hz`` have passed."""
    return -(-t * cpu_hz // hz)


# ================================================================ the 82C59 pair

class I8259:
    """One 82C59A (PIIX4 8.6, 4.2.2): IRR, ISR, IMR, the ICW sequence,
    OCW2 (EOI, rotation), OCW3 (read IRR / ISR, poll, special mask)."""

    def __init__(self, name: str, master: bool) -> None:
        self.name = name
        self.master = master
        self.reset()

    def reset(self) -> None:
        self.irr = 0
        self.isr = 0
        self.imr = 0xFF                          # masked until initialised (eMIPS)
        self.base = 0x08 if self.master else 0x70
        self.icw3 = 0x04 if self.master else 0x02
        self.icw4 = 0
        self.init_step = 0                       # 0: running, 2..4: next ICW expected
        self.icw4_needed = False
        self.read_isr = False
        self.poll = False
        self.smm = False                         # special mask mode
        self.rotate_aeoi = False
        self.lowest = 7                          # lowest-priority IRQ (rotation)
        self.lines = 0                           # the input levels
        self.elcr = 0                            # level-triggered inputs (PIIX4 ELCR)
        self.initialised = False

    # -- the inputs ---------------------------------------------------------------------

    def set_line(self, irq: int, level: bool) -> None:
        bit = 1 << irq
        old = self.lines & bit
        if level:
            self.lines |= bit
            if not old or self.elcr & bit:       # edge: a rising edge; level: the level
                self.irr |= bit
        else:
            self.lines &= ~bit
            if self.elcr & bit:                  # a level request goes when the line does
                self.irr &= ~bit

    # -- priority ----------------------------------------------------------------------

    def _order(self) -> list[int]:
        return [(self.lowest + 1 + i) & 7 for i in range(8)]

    def _highest(self, bits: int) -> int | None:
        for irq in self._order():
            if bits >> irq & 1:
                return irq
        return None

    def pending(self, slave_int: bool = False) -> int | None:
        """The IRQ this controller would request now (None: no INT)."""
        if not self.initialised:
            return None
        req = self.irr & ~self.imr
        if self.smm:
            req &= ~self.isr                     # special mask: lower in-service ones too
            irq = self._highest(req)
            return irq
        for irq in self._order():
            if self.isr >> irq & 1:              # in service: nothing at or below it
                if self.master and self.icw4 & 0x10 and irq == 2 and slave_int:
                    return 2                     # special fully nested: the slave again
                return None
            if req >> irq & 1:
                return irq
        return None

    def acknowledge(self, irq: int) -> None:
        """The INTA pulses: in service, the request taken (edge)."""
        bit = 1 << irq
        if not self.elcr & bit:
            self.irr &= ~bit
        if self.icw4 & 2:                        # AEOI
            if self.rotate_aeoi:
                self.lowest = irq
        else:
            self.isr |= bit

    # -- the registers -----------------------------------------------------------------

    def write(self, a0: int, value: int) -> None:
        value &= 0xFF
        if a0 == 0:
            if value & 0x10:                     # ICW1
                self.init_step = 2
                self.icw4_needed = bool(value & 1)
                self.imr = 0
                self.isr = 0
                self.irr &= self.elcr & self.lines
                self.lowest = 7
                self.smm = False
                self.read_isr = False
                self.icw4 = 0
                self.initialised = False
            elif value & 0x08:                   # OCW3
                if value & 0x04:
                    self.poll = True
                if value & 0x02:
                    self.read_isr = bool(value & 1)
                if value & 0x40:
                    self.smm = bool(value & 0x20)
            else:                                # OCW2
                self._ocw2(value)
            return
        if self.init_step == 2:                  # ICW2: the vector base
            self.base = value & 0xF8
            self.init_step = 3
        elif self.init_step == 3:                # ICW3 (cascade)
            self.icw3 = value
            self.init_step = 4 if self.icw4_needed else 0
            if not self.icw4_needed:
                self.initialised = True
        elif self.init_step == 4:                # ICW4
            self.icw4 = value & 0x1F
            self.init_step = 0
            self.initialised = True
        else:                                    # OCW1: the mask
            self.imr = value

    def _ocw2(self, value: int) -> None:
        cmd, level = value >> 5, value & 7
        if cmd in (1, 5):                        # non-specific EOI (5: and rotate)
            irq = self._highest(self.isr)
            if irq is not None:
                self.isr &= ~(1 << irq)
                if cmd == 5:
                    self.lowest = irq
        elif cmd in (3, 7):                      # specific EOI (7: and rotate)
            self.isr &= ~(1 << level)
            if cmd == 7:
                self.lowest = level
        elif cmd == 4:
            self.rotate_aeoi = True
        elif cmd == 0:
            self.rotate_aeoi = False
        elif cmd == 6:                           # set priority
            self.lowest = level

    def read(self, a0: int) -> int:
        if a0 == 1:
            return self.imr
        return self.isr if self.read_isr else self.irr


class PICs:
    """CNTRL-1 (master, 20h) and CNTRL-2 (slave, A0h, on the master's IRQ2)
    with ELCR1 / ELCR2 (4D0h / 4D1h); ``on_intr(level)``: INTR."""

    def __init__(self) -> None:
        self.master = I8259("CNTRL-1", True)
        self.slave = I8259("CNTRL-2", False)
        self.on_intr: Callable[[bool], None] | None = None
        self.intr = False

    def reset(self) -> None:
        self.master.reset()
        self.slave.reset()
        self.update()

    def set_irq(self, irq: int, level: bool) -> None:
        if irq < 8:
            self.master.set_line(irq, level)
        else:
            self.slave.set_line(irq - 8, level)
        self.update()

    def update(self) -> None:
        self.master.set_line(2, self.slave.pending() is not None)
        level = self.master.pending(self.slave.pending() is not None) is not None
        if level != self.intr:
            self.intr = level
            if self.on_intr is not None:
                self.on_intr(level)

    def inta(self) -> int:
        """An interrupt acknowledge cycle: the vector (a spurious IRQ 7 /
        15 when nothing requests any more)."""
        m = self.master
        irq = m.pending(self.slave.pending() is not None)
        if irq is None:
            self.update()
            return m.base | 7
        m.acknowledge(irq)
        if irq == 2 and m.icw3 & 4:
            s = self.slave
            sirq = s.pending()
            if sirq is None:
                vector = s.base | 7
            else:
                s.acknowledge(sirq)
                vector = s.base | sirq
        else:
            vector = m.base | irq
        self.update()
        return vector

    def poll(self, ctl: I8259) -> int:
        """OCW3 poll: the next read of the controller is an acknowledge."""
        ctl.poll = False
        irq = ctl.pending(ctl is self.master and self.slave.pending() is not None)
        if irq is None:
            return 0
        ctl.acknowledge(irq)
        self.update()
        return 0x80 | irq

    def read(self, port: int) -> int:
        ctl = self.master if port < 0x80 else self.slave
        if port & 1 == 0 and ctl.poll:
            return self.poll(ctl)
        return ctl.read(port & 1)

    def write(self, port: int, value: int) -> None:
        ctl = self.master if port < 0x80 else self.slave
        ctl.write(port & 1, value)
        self.update()

    def describe(self) -> list[str]:
        out = []
        for c, first in ((self.master, 0), (self.slave, 8)):
            state = "initialised" if c.initialised else "not initialised (ICW1-4)"
            out.append(f"{c.name}: {state}, vectors 0x{c.base:02x}, IRR 0x{c.irr:02x} "
                       f"ISR 0x{c.isr:02x} IMR 0x{c.imr:02x} ELCR 0x{c.elcr:02x} "
                       f"(IRQ {first}-{first + 7})")
        return out


# ================================================================ the 82C54

class Counter:
    """One 82C54 counter (PIIX4 8.8): its count follows the elapsed timer
    ticks; modes 0-5, binary or BCD."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.mode = 0
        self.rw = 3                              # LSB then MSB
        self.bcd = False
        self.reload = 0                          # the programmed count (0 = 65536)
        self.loaded_at: int | None = None        # the tick the count was loaded
        self.triggered_at: int | None = None     # modes 1 / 5: the gate trigger
        self.write_lsb: int | None = None
        self.latch: list[int] = []
        self.status_latch: int | None = None
        self.read_msb = False
        self.null_count = True
        self.gate = n != 2                        # counters 0 / 1: tied high

    @property
    def period(self) -> int:
        if self.reload:
            return _from_bcd(self.reload) if self.bcd else self.reload
        return 10000 if self.bcd else 0x10000

    def start(self, t: int) -> int | None:
        """The tick counting starts (None: not running)."""
        if self.loaded_at is None:
            return None
        if self.mode in (1, 5):
            return self.triggered_at
        if not self.gate and self.mode in (0, 2, 3, 4):
            return None
        return self.loaded_at + 1

    def count(self, t: int) -> int:
        n = self.period
        s = self.start(t)
        if s is None or t < s:
            return self.reload
        e = t - s
        if self.mode in (2, 6):
            c = n - (e % n)
        elif self.mode in (3, 7):
            c = n - 2 * (e % (n // 2 or 1))
        else:
            c = (n - e) % (10000 if self.bcd else 0x10000)
        return _to_bcd(c) if self.bcd else c & 0xFFFF

    def out(self, t: int) -> bool:
        n = self.period
        s = self.start(t)
        if s is None:
            return self.mode != 0 or self.loaded_at is None
        if t < s:
            return self.mode != 0
        e = t - s
        if self.mode == 0:
            return e >= n
        if self.mode == 1:
            return e >= n
        if self.mode in (2, 6):
            return e % n != n - 1
        if self.mode in (3, 7):
            return e % n < (n + 1) // 2
        if self.mode in (4, 5):
            return e != n
        return True

    def next_rise(self, t: int) -> int | None:
        """The first tick after ``t`` where OUT goes low -> high."""
        n = self.period
        s = self.start(t)
        if s is None:
            return None
        e = t - s
        if e < 0:
            return s + n if self.mode in (0, 1) else (s + n if self.mode not in (4, 5)
                                                      else s + n + 1)
        if self.mode in (0, 1):                  # OUT goes high at the terminal count
            return s + n if e < n else None
        if self.mode in (2, 3, 6, 7):            # a rising edge every period
            return s + (e // n + 1) * n
        if self.mode in (4, 5):                  # low for one tick at count 0
            return s + n + 1 if e <= n else None
        return None


def _to_bcd(v: int) -> int:
    return int(str(v % 10000), 16)


def _from_bcd(v: int) -> int:
    try:
        return int(f"{v:x}") or 10000
    except ValueError:
        return 10000


class PIT:
    """The three counters (40h-43h); counter 0's OUT is IRQ0, counter 2's
    is NMISC bit 5 and gated by NMISC bit 0."""

    def __init__(self) -> None:
        self.counters = [Counter(i) for i in range(3)]

    def reset(self) -> None:
        self.counters = [Counter(i) for i in range(3)]

    def write(self, port: int, value: int, t: int) -> None:
        if port == 3:
            sc, rw, mode, bcd = value >> 6, (value >> 4) & 3, (value >> 1) & 7, value & 1
            if sc == 3:                          # read-back
                for i in range(3):
                    if value >> (i + 1) & 1:
                        c = self.counters[i]
                        if not value & 0x20 and not c.latch:
                            c.latch = self._bytes(c, c.count(t))
                        if not value & 0x10 and c.status_latch is None:
                            c.status_latch = (int(c.out(t)) << 7 | int(c.null_count) << 6
                                              | c.rw << 4 | c.mode << 1 | int(c.bcd))
                return
            c = self.counters[sc]
            if rw == 0:                          # counter latch
                if not c.latch:
                    c.latch = self._bytes(c, c.count(t))
                return
            c.rw, c.mode, c.bcd = rw, (mode if mode < 6 else mode - 4), bool(bcd)
            c.loaded_at = None
            c.write_lsb = None
            c.null_count = True
            return
        c = self.counters[port]
        if c.rw == 1:
            c.reload = value
        elif c.rw == 2:
            c.reload = value << 8
        elif c.write_lsb is None:
            c.write_lsb = value
            return
        else:
            c.reload = c.write_lsb | value << 8
            c.write_lsb = None
        c.loaded_at = t
        c.null_count = False
        if c.mode in (1, 5):
            c.triggered_at = None

    def _bytes(self, c: Counter, count: int) -> list[int]:
        if c.rw == 1:
            return [count & 0xFF]
        if c.rw == 2:
            return [count >> 8 & 0xFF]
        return [count & 0xFF, count >> 8 & 0xFF]

    def read(self, port: int, t: int) -> int:
        c = self.counters[port]
        if c.status_latch is not None:
            v = c.status_latch
            c.status_latch = None
            return v
        if c.latch:
            return c.latch.pop(0)
        count = c.count(t)
        if c.rw == 1:
            return count & 0xFF
        if c.rw == 2:
            return count >> 8 & 0xFF
        c.read_msb = not c.read_msb
        return count >> 8 & 0xFF if not c.read_msb else count & 0xFF

    def set_gate2(self, level: bool, t: int) -> None:
        c = self.counters[2]
        if level and not c.gate and c.mode in (1, 5) and c.loaded_at is not None:
            c.triggered_at = t + 1
        c.gate = level


# ================================================================ the MC146818 RTC

RTC_SEC, RTC_SEC_ALARM, RTC_MIN, RTC_MIN_ALARM, RTC_HOUR, RTC_HOUR_ALARM = range(6)
RTC_DOW, RTC_DOM, RTC_MONTH, RTC_YEAR = 6, 7, 8, 9
RTC_A, RTC_B, RTC_C, RTC_D = 0x0A, 0x0B, 0x0C, 0x0D
A_UIP = 0x80
B_SET, B_PIE, B_AIE, B_UIE, B_SQWE, B_DM, B_24H, B_DSE = 0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1
C_IRQF, C_PF, C_AF, C_UF = 0x80, 0x40, 0x20, 0x10
D_VRT = 0x80
#: the update-in-progress window before each update (MC146818A: 244 us)
UIP_US = 244


class RTC:
    """The PIIX4's MC146818A-compatible clock with 256 bytes of CMOS RAM
    (8.9): the time and alarm registers (BCD or binary, 12 / 24 hour),
    registers A-D, the periodic / alarm / update-ended interrupts on IRQ8.
    It runs from :data:`RTC_EPOCH` when the board is powered."""

    def __init__(self, cpu_hz: int = CPU_HZ) -> None:
        self.on_irq: Callable[[bool], None] | None = None
        self.ram = bytearray(256)
        self.hz = cpu_hz
        self.reset()

    def reset(self) -> None:
        """Power-on: the battery-backed clock runs (DV = 010), 24-hour BCD
        as a PC BIOS leaves it; periodic rate 6 (1024 Hz), no interrupts."""
        self.when = RTC_EPOCH                    # the time at cycle ``since``
        self.since = 0
        self.a = 0x26
        self.b = B_24H
        self.c = 0
        self.alarm = [0, 0, 0]                   # sec, min, hour (register values)
        self.index = 0
        self.ext_index = 0
        self.nmi_disabled = True
        self.last_update = 0                     # the cycle of the last update
        self.next_periodic = None
        self.level = False
        self._schedule(0)

    # -- time ---------------------------------------------------------------------------

    def running(self) -> bool:
        return (self.a >> 4) & 7 == 2 and not self.b & B_SET

    def now(self, cycle: int) -> datetime.datetime:
        if not self.running():
            return self.when
        return self.when + datetime.timedelta(seconds=(cycle - self.since) // self.hz)

    def _settle(self, cycle: int) -> None:
        """Fold the elapsed time into ``when`` (before a change)."""
        self.when = self.now(cycle)
        if self.running():
            self.since += (cycle - self.since) // self.hz * self.hz
        else:
            self.since = cycle

    def periodic_cycles(self) -> int | None:
        rs = self.a & 0xF
        if rs == 0:
            return None
        if rs < 3:
            rs += 7                              # 32.768 kHz base: 1, 2 = 8, 9
        return (1 << (rs - 1)) * self.hz // RTC_HZ

    def _schedule(self, cycle: int) -> None:
        p = self.periodic_cycles()
        self.next_periodic = None if p is None or (self.a >> 4) & 7 != 2 else cycle + p

    def next_due(self, cycle: int) -> int | None:
        t = []
        if self.next_periodic is not None:
            t.append(self.next_periodic)
        if self.running():
            t.append(self.since + ((cycle - self.since) // self.hz + 1) * self.hz)
        return min(t) if t else None

    def advance(self, cycle: int) -> None:
        while True:
            due = self.next_due(self.last_update)
            if due is None or due > cycle:
                break
            if self.next_periodic is not None and due == self.next_periodic:
                self.c |= C_PF
                self.next_periodic += self.periodic_cycles()
            if self.running() and (due - self.since) % self.hz == 0 and due > self.since:
                self.c |= C_UF                   # the update cycle: a new second
                t = self.now(due)
                if self._alarm_hit(t):
                    self.c |= C_AF
            self.last_update = due
            self._irq()
        self.last_update = max(self.last_update, cycle)

    def _alarm_hit(self, t: datetime.datetime) -> bool:
        vals = (t.second, t.minute, t.hour)
        for k, (reg, v) in enumerate(zip(self.alarm, vals)):
            if reg & 0xC0 == 0xC0:               # "don't care"
                continue
            if self._decode(reg, hour=k == 2) != v:
                return False
        return True

    # -- register encoding ---------------------------------------------------------------

    def _encode(self, v: int, *, hour: bool = False) -> int:
        pm = 0
        if hour and not self.b & B_24H:
            pm = 0x80 if v >= 12 else 0
            v = v % 12 or 12
        if not self.b & B_DM:
            v = (v // 10) << 4 | v % 10
        return v | pm

    def _decode(self, r: int, *, hour: bool = False) -> int:
        pm = hour and not self.b & B_24H and r & 0x80
        r &= 0x7F if hour else 0xFF
        v = r if self.b & B_DM else (r >> 4) * 10 + (r & 0xF)
        if hour and not self.b & B_24H:
            v = v % 12 + (12 if pm else 0)
        return v

    def read_reg(self, reg: int, cycle: int) -> int:
        self.advance(cycle)
        if reg <= RTC_YEAR:
            t = self.now(cycle)
            return {RTC_SEC: self._encode(t.second), RTC_MIN: self._encode(t.minute),
                    RTC_HOUR: self._encode(t.hour, hour=True),
                    RTC_DOW: self._encode(t.isoweekday() % 7 + 1),
                    RTC_DOM: self._encode(t.day), RTC_MONTH: self._encode(t.month),
                    RTC_YEAR: self._encode(t.year % 100),
                    RTC_SEC_ALARM: self.alarm[0], RTC_MIN_ALARM: self.alarm[1],
                    RTC_HOUR_ALARM: self.alarm[2]}[reg]
        if reg == RTC_A:
            uip = 0
            if self.running():
                into = (cycle - self.since) % self.hz
                if into >= self.hz - UIP_US * self.hz // 1_000_000:
                    uip = A_UIP
            return self.a | uip
        if reg == RTC_B:
            return self.b
        if reg == RTC_C:                         # reading C clears the flags
            v = self.c | (C_IRQF if self._irqf() else 0)
            self.c = 0
            self._irq()
            return v
        if reg == RTC_D:
            return D_VRT
        return self.ram[reg]

    def write_reg(self, reg: int, value: int, cycle: int) -> None:
        self.advance(cycle)
        value &= 0xFF
        if reg in (RTC_SEC_ALARM, RTC_MIN_ALARM, RTC_HOUR_ALARM):
            self.alarm[(reg - 1) // 2] = value
            return
        if reg <= RTC_YEAR:
            self._settle(cycle)
            t = self.when
            fields = {"second": t.second, "minute": t.minute, "hour": t.hour,
                      "day": t.day, "month": t.month, "year": t.year}
            if reg == RTC_SEC:
                fields["second"] = min(self._decode(value), 59)
            elif reg == RTC_MIN:
                fields["minute"] = min(self._decode(value), 59)
            elif reg == RTC_HOUR:
                fields["hour"] = min(self._decode(value, hour=True), 23)
            elif reg == RTC_DOM:
                fields["day"] = max(1, min(self._decode(value), 31))
            elif reg == RTC_MONTH:
                fields["month"] = max(1, min(self._decode(value), 12))
            elif reg == RTC_YEAR:
                fields["year"] = 2000 + self._decode(value) % 100
            try:
                self.when = datetime.datetime(**fields)
            except ValueError:                   # an impossible date: kept as it was
                pass
            return                               # day of week: computed from the date
        if reg == RTC_A:
            self._settle(cycle)
            self.a = value & 0x7F
            self._schedule(cycle)
        elif reg == RTC_B:
            self._settle(cycle)
            self.b = value & 0xFF
            if self.b & B_SET:
                self.b &= ~B_UIE
            self._irq()
        elif reg in (RTC_C, RTC_D):
            return                               # read-only
        else:
            self.ram[reg] = value

    def _irqf(self) -> bool:
        return bool(self.c & self.b & (C_PF | C_AF | C_UF))

    def _irq(self) -> None:
        level = self._irqf()
        if level != self.level:
            self.level = level
            if self.on_irq is not None:
                self.on_irq(level)

    def describe(self, cycle: int) -> str:
        t = self.now(cycle)
        p = self.periodic_cycles()
        rate = f"periodic every {p * 1_000_000 // self.hz} us" if p else "no periodic interrupt"
        return (f"RTC {t:%Y-%m-%d %H:%M:%S}{'' if self.running() else ' (stopped)'}, "
                f"{'binary' if self.b & B_DM else 'BCD'}, "
                f"{'24' if self.b & B_24H else '12'}-hour, {rate}, A 0x{self.a:02x} "
                f"B 0x{self.b:02x} C 0x{self.c:02x}, IRQ8 {'on' if self.level else 'off'}")


# ================================================================ PCI configuration

class PCIFunction:
    """A PCI function's configuration space: the header's IDs read-only;
    with ``masks`` (byte offset -> writable bits) only those bits are
    written (the rest read as reset) and ``w1c`` bits clear when written
    with 1; without, everything else is kept as written."""

    def __init__(self, name: str, vendor: int, device: int, cls: int, rev: int = 0,
                 header: int = 0, defaults: dict[int, int] | None = None,
                 masks: dict[int, int] | None = None,
                 w1c: dict[int, int] | None = None) -> None:
        self.name = name
        self.space = bytearray(256)
        self.space[0:2] = vendor.to_bytes(2, "little")
        self.space[2:4] = device.to_bytes(2, "little")
        self.space[8] = rev
        self.space[9:12] = cls.to_bytes(3, "little")
        self.space[0x0E] = header
        for off, value in (defaults or {}).items():
            self.space[off] = value
        self.masks = masks
        self.w1c = w1c or {}
        self.on_write: Callable[[], None] | None = None
        self.reset_space = bytes(self.space)

    def reset(self) -> None:
        self.space[:] = self.reset_space

    def read(self, reg: int) -> int:
        return int.from_bytes(self.space[reg & 0xFC: (reg & 0xFC) + 4], "little")

    def write(self, reg: int, value: int, mask: int = 0xFFFFFFFF) -> None:
        reg &= 0xFC
        for k in range(4):
            if mask >> (8 * k) & 0xFF == 0:
                continue
            off = reg + k
            if off < 0x04 or 0x08 <= off < 0x0C or off == 0x0E:
                continue                         # IDs, class, header: read-only
            byte = value >> (8 * k) & 0xFF
            if self.masks is None:
                self.space[off] = byte
                continue
            m = self.masks.get(off, 0)
            c = self.w1c.get(off, 0)
            self.space[off] = (self.space[off] & ~m | byte & m) & ~(byte & c)
        if self.on_write is not None:
            self.on_write()


#: the IDE function's header (5.1): PCICMD IOSE / BME, PCISTS 0280h with
#: MAS / RTA / STA write-1-to-clear, MLT 7:4, BMIBA 15:4 (bit 0: I/O),
#: IDETIM / SIDETIM / UDMACTL / UDMATIM (the reserved bits read 0)
IDE_DEFAULTS = {0x06: 0x80, 0x07: 0x02, 0x20: 0x01}
IDE_MASKS = {0x04: 0x05, 0x0D: 0xF0, 0x20: 0xF0, 0x21: 0xFF,
             0x40: 0xFF, 0x41: 0xF3, 0x42: 0xFF, 0x43: 0xF3, 0x44: 0xFF,
             0x48: 0x0F, 0x4A: 0x33, 0x4B: 0x33}
IDE_W1C = {0x07: 0x38}


# ================================================================ the PIIX4

class PIIX4:
    """The ISA side of the PIIX4: the port decode and the devices on it
    (the PICs, the timer, NMISC, the RTC, ELCR) plus the Super I/O chip
    on its ISA bus; its PCI functions (device 10)."""

    def __init__(self, cpu_hz: int = CPU_HZ) -> None:
        self.hz = cpu_hz
        self.pics = PICs()
        self.pit = PIT()
        self.rtc = RTC(cpu_hz)
        self.rtc.on_irq = lambda level: self.pics.set_irq(8, level)
        self.isa_devices: list = []              # the Super I/O (ports it claims)
        self.functions = [
            PCIFunction("PIIX4 PCI-to-ISA bridge", 0x8086, 0x7110, 0x060100, rev=0x01,
                        header=0x80, defaults={0x04: 0x07, 0x60: 0x80, 0x61: 0x80,
                                               0x62: 0x80, 0x63: 0x80, 0x69: 0x02}),
            PCIFunction("PIIX4 IDE", 0x8086, 0x7111, 0x010180, rev=0x01,
                        defaults=IDE_DEFAULTS, masks=IDE_MASKS, w1c=IDE_W1C),
            PCIFunction("PIIX4 USB", 0x8086, 0x7112, 0x0C0300, rev=0x01),
            PCIFunction("PIIX4 power management", 0x8086, 0x7113, 0x068000, rev=0x02),
        ]
        self.nmisc = 0
        self.irq0 = False
        self.now = 0
        #: the IDE channels (9.2): primary (IRQ 14), secondary (IRQ 15)
        self.ide = [ATA.IDEChannel("primary IDE", 14, lambda: self.now),
                    ATA.IDEChannel("secondary IDE", 15, lambda: self.now)]
        #: the bus master IDE channels (5.2, 9.4; emips.busmaster)
        self.bm = [BusMasterChannel(n, ch, cpu_hz) for n, ch in
                   (("primary", self.ide[0]), ("secondary", self.ide[1]))]
        ide = self.functions[1].space
        for i, (ch, bm) in enumerate(zip(self.ide, self.bm)):
            ch.on_irq = lambda irq, level, bm=bm: (bm.irq_changed(level),
                                                   self.pics.set_irq(irq, level))
            bm.now = lambda: self.now
            bm.bme = lambda: bool(ide[0x04] & 0x04)
            bm.udma_enabled = lambda n, i=i: bool(ide[0x48] >> (2 * i + n) & 1)
            bm.on_master_abort = self._master_abort

    def _master_abort(self) -> None:
        self.functions[1].space[0x07] |= 0x20    # PCISTS.MAS (5.1.4)

    def reset(self) -> None:
        for bm in self.bm:
            bm.reset()
        self.pics.reset()
        self.pit.reset()
        self.rtc.reset()
        for f in self.functions:
            f.reset()
        self.nmisc = 0
        self.irq0 = False

    # -- time -----------------------------------------------------------------------------

    def next_due(self) -> int | None:
        t = []
        rise = self.pit.counters[0].next_rise(ticks(self.now, PIT_HZ, self.hz))
        if rise is not None:
            t.append(cycles_for(rise, PIT_HZ, self.hz))
        r = self.rtc.next_due(self.rtc.last_update)
        if r is not None:
            t.append(r)
        for dev in [*self.isa_devices, *self.ide, *self.bm]:
            d = dev.next_due()
            if d is not None:
                t.append(d)
        return min(t) if t else None

    def _ide_due(self) -> int | None:
        t = [d for d in (x.next_due() for x in (*self.ide, *self.bm)) if d is not None]
        return min(t) if t else None

    def advance(self, cycle: int) -> None:
        start = self.now
        while True:                               # the disks and their DMA, in time order
            t = self._ide_due()
            if t is None or t > cycle:
                break
            self.now = max(t, start)
            for x in (*self.ide, *self.bm):
                x.advance(self.now)
        self.now = start
        old = ticks(self.now, PIT_HZ, self.hz)
        self.now = cycle
        self.rtc.advance(cycle)
        t = ticks(cycle, PIT_HZ, self.hz)
        c0 = self.pit.counters[0]
        rise = c0.next_rise(old)
        if rise is not None and old < rise <= t:   # a rising edge went by: IRQ0's edge
            self.pics.set_irq(0, False)
            self.pics.set_irq(0, True)
        out = c0.out(t)
        if out != self.irq0 or (rise is not None and old < rise <= t):
            self.irq0 = out
            self.pics.set_irq(0, out)
        for dev in self.isa_devices:
            dev.advance(cycle)

    # -- the IDE ports (9.2) --------------------------------------------------------------

    def ide_port(self, port: int) -> tuple[ATA.IDEChannel, str, int] | None:
        """(channel, "cmd" | "ctl", offset) for a port the IDE function
        decodes: PCICMD.IOSE and the channel's IDETIM.IDE set (5.1.3,
        5.1.10); the control block's byte 3 goes to ISA (the floppy)."""
        f = self.functions[1].space
        if not f[0x04] & 1:
            return None
        for i, (cmd, ctl) in enumerate(((0x1F0, 0x3F4), (0x170, 0x374))):
            if not f[0x41 + 2 * i] & 0x80:
                continue
            if cmd <= port < cmd + 8:
                return self.ide[i], "cmd", port - cmd
            if ctl <= port < ctl + 3:
                return self.ide[i], "ctl", port - ctl
        return None

    def bm_port(self, port: int) -> tuple[BusMasterChannel, int] | None:
        """(channel, offset) for a port in the bus master block: BMIBA's
        16 bytes (5.1.9) with PCICMD.IOSE."""
        f = self.functions[1].space
        if not f[0x04] & 1:
            return None
        base = (f[0x20] | f[0x21] << 8) & 0xFFF0
        if base and base <= port < base + 16:
            return self.bm[(port - base) >> 3], (port - base) & 7
        return None

    def read_bytes(self, port: int, size: int, *, peek: bool = False) -> bytes:
        """A PCI I/O read of ``size`` bytes at ``port`` (the bytes in
        address order).  The IDE Data register is 16 bits: a 16-bit access
        is one word (DD7:0 at the lower address), a 32-bit one two words
        back to back (9.3), a byte access one word's DD7:0.  The bus
        master registers take byte, word or dword accesses (5.2)."""
        port &= 0xFFFF
        if self.bm_port(port) is not None:
            return bytes(self.bm_port(port + i)[0].read(self.bm_port(port + i)[1])
                         if self.bm_port(port + i) else 0xFF for i in range(size))
        hit = self.ide_port(port)
        if hit is not None and hit[1] == "cmd" and hit[2] == ATA.DATA:
            ch = hit[0]
            out = bytearray()
            for _ in range(max(1, size // 2)):
                w = ch.read_data(peek=peek)
                out += bytes((w & 0xFF, w >> 8))
            return bytes(out[:size])
        out = bytearray()
        for i in range(size):
            p = (port + i) & 0xFFFF
            hit = self.ide_port(p)
            if hit is None:
                out.append(self.io_peek(p) if peek else self._isa_read(p))
            elif hit[1] == "cmd":
                out.append(hit[0].read(hit[2], peek=peek))
            else:
                out.append(hit[0].read_alt_status() if hit[2] == 2 else ATA.FLOAT)
        return bytes(out)

    def write_bytes(self, port: int, data: bytes) -> None:
        port &= 0xFFFF
        if self.bm_port(port) is not None:
            for i, byte in enumerate(data):
                hit = self.bm_port(port + i)
                if hit is not None:
                    hit[0].write(hit[1], byte)
            return
        hit = self.ide_port(port)
        if hit is not None and hit[1] == "cmd" and hit[2] == ATA.DATA:
            ch = hit[0]
            if len(data) == 1:
                ch.write_data(data[0] | 0xFF00)  # DD15:8 undriven: read high
            for k in range(0, len(data) - 1, 2):
                ch.write_data(data[k] | data[k + 1] << 8)
            return
        for i, byte in enumerate(data):
            p = (port + i) & 0xFFFF
            hit = self.ide_port(p)
            if hit is None:
                self._isa_write(p, byte)
            elif hit[1] == "cmd":
                hit[0].write(hit[2], byte)
            elif hit[2] == 2:
                hit[0].write_control(byte)

    # -- ISA ports -------------------------------------------------------------------------

    def io_read(self, port: int) -> int:
        """One byte at ``port`` (as :meth:`read_bytes`)."""
        return self.read_bytes(port, 1)[0]

    def io_write(self, port: int, value: int) -> None:
        self.write_bytes(port, bytes((value & 0xFF,)))

    def _isa_read(self, port: int) -> int:
        port &= 0xFFFF
        t = ticks(self.now, PIT_HZ, self.hz)
        if port in (0x20, 0x21, 0xA0, 0xA1):
            return self.pics.read(port)
        if 0x40 <= port <= 0x43:
            return 0xFF if port == 0x43 else self.pit.read(port - 0x40, t)
        if port == 0x61:
            c2 = self.pit.counters[2]
            return ((self.nmisc & 0x0F) | (0x20 if c2.out(t) else 0)
                    | (0x10 if (t // 18) & 1 else 0))   # refresh toggles every ~15 us
        if port == 0x71:
            return self.rtc.read_reg(self.rtc.index, self.now)
        if port == 0x73:
            return self.rtc.ram[0x80 | self.rtc.ext_index]
        if port == 0x72:
            return self.rtc.ext_index
        if port == 0x4D0:
            return self.pics.master.elcr
        if port == 0x4D1:
            return self.pics.slave.elcr
        for dev in self.isa_devices:
            if dev.claims(port):
                return dev.io_read(port)
        return 0xFF                              # nothing on the ISA bus answers

    def _isa_write(self, port: int, value: int) -> None:
        port &= 0xFFFF
        value &= 0xFF
        t = ticks(self.now, PIT_HZ, self.hz)
        if port in (0x20, 0x21, 0xA0, 0xA1):
            self.pics.write(port, value)
        elif 0x40 <= port <= 0x43:
            self.pit.write(port - 0x40, value, t)
            self.advance(self.now)
        elif port == 0x61:
            self.nmisc = value & 0x0F
            self.pit.set_gate2(bool(value & 1), t)
        elif port == 0x70:
            self.rtc.index = value & 0x7F
            self.rtc.nmi_disabled = bool(value & 0x80)
        elif port == 0x71:
            self.rtc.write_reg(self.rtc.index, value, self.now)
        elif port == 0x72:
            self.rtc.ext_index = value & 0x7F
        elif port == 0x73:
            self.rtc.ram[0x80 | self.rtc.ext_index] = value
        elif port == 0x4D0:
            self.pics.master.elcr = value & 0xF8
            self.pics.update()
        elif port == 0x4D1:
            self.pics.slave.elcr = value & 0xDE
            self.pics.update()
        else:
            for dev in self.isa_devices:
                if dev.claims(port):
                    dev.io_write(port, value)
                    return

    def io_peek(self, port: int) -> int:
        """A port as a view sees it (no side effects: status registers,
        never a FIFO or an acknowledge)."""
        port &= 0xFFFF
        hit = self.ide_port(port)
        if hit is not None:
            return self.read_bytes(port, 1, peek=True)[0]
        for dev in self.isa_devices:
            if dev.claims(port):
                return dev.io_peek(port)
        if port in (0x21, 0xA1):
            return (self.pics.master if port < 0x80 else self.pics.slave).imr
        if port == 0x4D0:
            return self.pics.master.elcr
        if port == 0x4D1:
            return self.pics.slave.elcr
        return 0xFF
