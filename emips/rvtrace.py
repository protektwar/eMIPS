"""RISC-V trace: the components of the RISC-V Trace Control Interface 1.0
(LF11; the specification's ``RISC-V-Trace-Control-Interface.adoc``) on a generic hart's system bus, driving the E-Trace and
N-Trace encoders (:mod:`emips.etrace`, :mod:`emips.ntrace`) from each
hart's ingress port (:mod:`emips.rvingress`).

**Components** (4 KB each, 32-bit registers, :data:`MAP`): a Trace
Encoder per hart (Table "Trace Encoder Registers": trTeControl, trTeImpl,
trTeInstFeatures, trTeInstFilters, trTeDataControl — no data trace:
trTeDataImplemented 0 —, the timestamp unit trTsControl / trTsCounter,
the trigger controls, 4 filters and 4 comparators at 0x400 / 0x600), a
Trace Funnel when there are several harts, and one Trace RAM Sink (16 KB
of SRAM read through trRamRP / trRamData, or SMEM mode: a buffer in
system memory the sink writes as a bus master — below the caches, as a
DMA engine).  Sinks store bytes LSB first in 32-bit words; packets are
atomic (a component never sends a partial one).  No PIB sink or ATB
bridge: nothing in eMIPS sits on their pins.

**The encoder** follows the control interface's rules: trTeActive resets
it (registers to reset values), trTeEnable 0 -> 1 starts the protocol
(E-Trace: a support packet), 1 -> 0 flushes it; trTeInstTracing turns
instruction trace on / off (by software, by trace-on / trace-off actions
with trTeInstTrigEnable — Debug spec Table 12, E-Trace Table 4.9 — or by
external triggers); trTeFormat chooses E-Trace (0) or N-Trace (1); the
filters qualify instructions (privilege, exception cause / interrupt from
a trap to its return, comparators on iaddr / context / tval with the
four match modes, a primary / secondary notify = a watchpoint).  Back
pressure: a sink that cannot take a packet (stopped on wrap, disabled)
loses it — trTeInstStallOrOverflow, then E-Trace's *trace_lost* support
packet or N-Trace's Error message and a SYNC 7 restart — or, with
trTeInstStallEna, stalls the hart until it can.

**Timestamps** (the timestamp unit, trTsMode 3 internal core: the hart's
cycles, or 2 internal system: hart 0's; prescale, trTsRunInDebug, 48
bits): E-Trace puts them in the encapsulation (the Unformatted Trace &
Diagnostic Data Packet Encapsulation 1.0: header length / flow / extend,
srcID, a timestamp of T = 6 bytes), N-Trace in each message's TSTAMP.

**Cross triggers** (:data:`WIRING`: eMIPS's — the specifications leave
them to the platform): a hart's trigger module outputs 0 / 1 (actions 8
/ 9) -> its encoder's external inputs 0 / 1; encoder output 0 -> the
hart's tmexttrigger input 0; output 1 -> the Debug Module's external
trigger 0 (its halt group halts); the DM's external trigger 1 (a group
halted / resumed) -> every encoder's input 2; the probe (the debugger's
``trace in``) -> input 3.

Choices: ``sim.DEVIATIONS["RISC-V trace"]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import etrace as E
from . import ntrace as N
from . import rvingress as I
from .rvplatform import Device

M32 = 0xFFFFFFFF
#: the component base addresses (the control interface leaves them to the
#: platform: below the CLINT, Spike / QEMU virt have nothing there)
MAP = {"encoder": 0x03000000, "funnel": 0x03008000, "ramsink": 0x03009000}
MAX_ENCODERS = 8
SRAM_SIZE = 16 * 1024
TS_WIDTH = 48
FILTERS, COMPARATORS = 4, 4
EXT_INPUTS, EXT_OUTPUTS = 4, 2
WIRING = {
    "tm_out": "hart h's trigger module output k (action 8 + k) -> encoder h input k",
    "te_out0": "encoder h output 0 -> hart h's tmexttrigger input 0",
    "te_out1": "encoder h output 1 -> the Debug Module's external trigger 0",
    "dm_ext1": "the Debug Module's external trigger 1 -> every encoder's input 2",
    "probe": "the probe (debugger 'trace in') -> every encoder's input 3",
}
# component types (trXxImpl)
COMP_ENCODER, COMP_FUNNEL, COMP_RAMSINK = 0x1, 0x8, 0x9
# trTeControl bits
ACTIVE, ENABLE, INST_TRACING, EMPTY = 1, 2, 4, 8
CTX, INST_TRIG_ENABLE, STALL_OR_OVERFLOW, STALL_ENA, INHIBIT_SRC = 9, 11, 12, 13, 15


def _f(v: int, lo: int, n: int = 1) -> int:
    return v >> lo & ((1 << n) - 1)


# -- filters and comparators -----------------------------------------------------------


@dataclass
class Comparator:
    """trTeComp_j (the control interface's Table "Comparator Control")."""
    control: int = 0
    pmatch: int = 0
    smatch: int = 0
    latch: bool = False

    WRITABLE = 0x3 | 0xC | 0x70 | 0x700 | 0x3000 | 0xC000

    def _input(self, sel: int, b: I.Block) -> int | None:
        if sel == 0:
            return b.iaddr if b.iretire else None
        if sel == 1:
            return b.context
        if sel == 2:
            return b.tval if b.itype == I.EXCEPTION else None
        return None                             # (3: daddr — no data trace)

    @staticmethod
    def _cmp(fn: int, v: int | None, ref: int) -> bool:
        if fn == 7:
            return True
        if fn == 6:
            return False
        if v is None:
            return False
        return (v == ref, v != ref, v < ref, v <= ref, v > ref, v >= ref)[fn]

    def evaluate(self, b: I.Block) -> tuple[bool, bool]:
        """(the comparator's output, a notification) for block ``b``."""
        c = self.control
        pin, sin = _f(c, 0, 2), _f(c, 2, 2)
        pf, sf = _f(c, 4, 3), _f(c, 8, 3)
        mode = _f(c, 12, 2)
        pv, sv = self._input(pin, b), self._input(sin, b)
        if sf == 6 and pv is not None:          # (S is P's mask)
            p = self._cmp(pf, pv & self.smatch, self.pmatch & self.smatch)
            s = True
        else:
            p = self._cmp(pf, pv, self.pmatch)
            s = self._cmp(6 if sf == 6 else sf, sv, self.smatch) if sf != 6 else True
        if pf == 6 and mode == 3:
            self.latch = True                   # (6 primes the latch)
        if mode == 0:
            out = p
        elif mode == 1:
            out = p and s
        elif mode == 2:
            out = not (p and s)
        else:                                   # (3: set on P, until the one after S)
            out = self.latch or p
            self.latch = (self.latch or p) and not s
        notify = (_f(c, 14) and pin == 0 and p) or (_f(c, 15) and sin == 0 and s
                                                     and sf not in (6, 7))
        return out, bool(notify)


@dataclass
class Filter:
    """trTeFilter_i (Table "Filter Control" and its match registers)."""
    control: int = 0
    inst: int = 0
    ecause_lo: int = 0
    ecause_hi: int = 0
    impdef_value: int = 0
    impdef_mask: int = 0
    #: the trap-to-return window of MatchEcause / MatchInterrupt (a nesting count)
    nest: int = 0

    WRITABLE = 0x1FFFF | 0x3000000

    def matches(self, b: I.Block, comps: list[bool]) -> bool:
        c = self.control
        if not c & 1:
            return False
        ok = True
        if _f(c, 1):
            ok &= bool(self.inst >> b.priv & 1)
        trap = b.itype in (I.EXCEPTION, I.INTERRUPT)
        if _f(c, 2) or _f(c, 3):
            if trap:
                if self.nest:
                    self.nest += 1
                else:
                    want = True
                    if _f(c, 2):
                        choice = self.ecause_lo | self.ecause_hi << 32
                        want &= bool(choice >> b.cause & 1)
                    if _f(c, 3):
                        want &= (b.itype == I.INTERRUPT) == bool(_f(self.inst, 8))
                    if want:
                        self.nest = 1
            active = self.nest > 0
            if b.itype == I.TRAP_RETURN and self.nest:
                self.nest -= 1                  # (the return ends it: traced itself)
            ok &= active
        for k, (en, sel) in enumerate(((4, 5), (8, 9), (12, 13))):
            if _f(c, en):
                j = _f(c, sel, 3)
                ok &= j < len(comps) and comps[j]
        if _f(c, 16):
            ok &= (0 & self.impdef_mask) == (self.impdef_value & self.impdef_mask)
        return ok


# -- the timestamp unit ---------------------------------------------------------------


@dataclass
class Timestamp:
    """trTsControl / trTsCounter (the control interface's Timestamp Unit)."""
    control: int = 0
    base: int = 0
    since: int = 0                              # (the clock when it last started)
    debug_stop: int | None = None

    MODES = (0, 2, 3)

    def write(self, v: int, clock: int) -> None:
        old = self.control
        self.value(clock)
        self.base = self.value(clock)
        self.since = clock
        if v & 4:                               # (trTsReset: W1)
            self.base = 0
        mode = _f(v, 4, 3)
        if mode not in self.MODES:
            mode = _f(old, 4, 3)
        keep = 0x1 | 0x2 | 0x8 | 0x300 | 0x8000
        self.control = (v & keep & ~0x70) | mode << 4 | TS_WIDTH << 24
        if not v & 1:
            self.control = TS_WIDTH << 24

    def running(self) -> bool:
        c = self.control
        return bool(c & 1 and c & 2 and _f(c, 4, 3) in (2, 3))

    def value(self, clock: int) -> int:
        if not self.running():
            return self.base
        pre = 2 * _f(self.control, 8, 2)
        return (self.base + ((clock - self.since) >> pre)) & ((1 << TS_WIDTH) - 1)

    @property
    def enabled(self) -> bool:
        return bool(self.control & 0x8000 and self.control & 1)


# -- the encoder component ------------------------------------------------------------


class TraceEncoder:
    """A hart's Trace Encoder: registers (``get`` / ``put``) and the
    consumer of its ingress port (``block`` / ``halted`` / ``reset``)."""

    def __init__(self, system: TraceSystem, index: int, m: Any) -> None:
        self.sys = system
        self.index = index
        self.m = m
        self.xlen = m.xlen
        self.params = E.params_for(self.xlen)
        self.lost = False
        self.queue: list[tuple[bytes, str]] = []
        self.ext_pending = 0                    # (external trigger inputs this cycle)
        self.packets: list[Any] = []            # (what it sent: E.Packet / N.Message)
        self.reset_registers()

    # -- registers ---------------------------------------------------------------

    def reset_registers(self) -> None:
        self.control = 0
        self.features = 0
        self.filters_reg = 0
        self.dbg_control = 0
        self.ext_in = 0
        self.ext_out = 0
        self.filt = [Filter() for _ in range(FILTERS)]
        self.comp = [Comparator() for _ in range(COMPARATORS)]
        self.ts = Timestamp()
        self.enc: Any = None
        self.lost = False
        self.queue = []
        self.m.trace_stall = False

    @property
    def format(self) -> int:
        return _f(self.control, 24, 3)

    @property
    def active(self) -> bool:
        return bool(self.control & ACTIVE)

    @property
    def enabled(self) -> bool:
        return bool(self.control & ACTIVE and self.control & ENABLE)

    @property
    def tracing(self) -> bool:
        return bool(self.control & INST_TRACING)

    def impl(self) -> int:
        major, minor = (0, 0) if self.format == 0 else (1, 0)
        return 1 | 0 << 4 | COMP_ENCODER << 8 | major << 16 | minor << 20

    def get(self, off: int) -> int:
        if off == 0x000:
            c = self.control
            empty = 0 if self.queue else EMPTY
            return (c & ~EMPTY) | empty
        if off == 0x004:
            return self.impl()
        if not self.active:
            return 0
        if off == 0x008:
            return self.features
        if off == 0x00C:
            return self.filters_reg
        if off == 0x010:
            return 0                            # (trTeDataImplemented 0)
        if off == 0x040:
            return self.ts.control
        if off in (0x048, 0x04C):
            v = self.ts.value(self._clock())
            return v & M32 if off == 0x048 else v >> 32
        if off == 0x050:
            return self.dbg_control
        if off == 0x054:
            return self.ext_in
        if off == 0x058:
            return self.ext_out
        if 0x400 <= off < 0x400 + 0x20 * FILTERS:
            f = self.filt[(off - 0x400) >> 5]
            return {0: f.control, 4: f.inst, 8: f.ecause_lo, 0xC: f.ecause_hi,
                    0x10: f.impdef_value, 0x14: f.impdef_mask}.get(off & 0x1F, 0)
        if 0x600 <= off < 0x600 + 0x20 * COMPARATORS:
            c = self.comp[(off - 0x600) >> 5]
            r = off & 0x1F
            return {0: c.control, 0x10: c.pmatch & M32, 0x14: c.pmatch >> 32,
                    0x18: c.smatch & M32, 0x1C: c.smatch >> 32}.get(r, 0)
        return 0

    def _legal_control(self, v: int) -> int:
        """trTeControl's WARL fields: trTeInstMode (E-Trace: 7; N-Trace 0, 3
        or 6), trTeInstSyncMode 0-3, SyncMax 0-15, trTeFormat 0 / 1."""
        fmt = _f(v, 24, 3)
        if fmt not in (0, 1):
            fmt = self.format
        mode = _f(v, 4, 3)
        if fmt == 0:
            mode = 7
        elif mode not in (0, 3, 6):
            mode = 6
        ctx = _f(v, CTX) if fmt == 1 else 0
        keep = ACTIVE | ENABLE | INST_TRACING | 1 << INST_TRIG_ENABLE | 1 << STALL_ENA \
            | 1 << INHIBIT_SRC | 3 << 16 | 0xF << 20
        return (v & keep) | mode << 4 | ctx << CTX | fmt << 24

    def put(self, off: int, v: int) -> None:
        v &= M32
        if off == 0x000:
            self._write_control(v)
            return
        if not self.active:
            return
        if off == 0x008:
            self.features = self._legal_features(v)
        elif off == 0x00C:
            self.filters_reg = v & ((1 << FILTERS) - 1)
        elif off == 0x040:
            self.ts.write(v, self._clock())
        elif off == 0x050:
            self.dbg_control = 0                # (no vendor trigger setup)
        elif off == 0x054:
            self.ext_in = sum((_f(v, 4 * k, 4) if _f(v, 4 * k, 4) in (0, 2, 3, 4) else 0)
                              << 4 * k for k in range(EXT_INPUTS))
        elif off == 0x058:
            self.ext_out = sum((_f(v, 4 * k, 4) & 3) << 4 * k for k in range(EXT_OUTPUTS))
        elif 0x400 <= off < 0x400 + 0x20 * FILTERS:
            f = self.filt[(off - 0x400) >> 5]
            r = off & 0x1F
            if r == 0:
                f.control = v & Filter.WRITABLE & ~0x3000000     # (no data trace)
            elif r == 4:
                f.inst = v & (0x7B | 0x100)     # (privileges 0, 1, 3, 4, 5, 6)
            elif r == 8:
                f.ecause_lo = v
            elif r == 0xC:
                f.ecause_hi = v
            elif r == 0x10:
                f.impdef_value = v
            elif r == 0x14:
                f.impdef_mask = v
        elif 0x600 <= off < 0x600 + 0x20 * COMPARATORS:
            c = self.comp[(off - 0x600) >> 5]
            r = off & 0x1F
            if r == 0:
                pin, sin = _f(v, 0, 2), _f(v, 2, 2)
                pin = pin if pin != 3 else 0        # (daddr: no data trace)
                sin = sin if sin != 3 else 0
                c.control = (v & Comparator.WRITABLE & ~0xF) | pin | sin << 2
                c.latch = False
            elif r == 0x10:
                c.pmatch = (c.pmatch & ~M32) | v
            elif r == 0x14:
                c.pmatch = (c.pmatch & M32) | v << 32 if self.xlen == 64 else c.pmatch
            elif r == 0x18:
                c.smatch = (c.smatch & ~M32) | v
            elif r == 0x1C:
                c.smatch = (c.smatch & M32) | v << 32 if self.xlen == 64 else c.smatch

    def _legal_features(self, v: int) -> int:
        """trTeInstFeatures: E-Trace has bits 0-5 (N-Trace's 6-10 read 0),
        N-Trace bits 2, 3, 6-10 (E-Trace's 0, 1, 4, 5 read 0, 8.1); SrcBits
        0-12."""
        if self.format == 0:
            bits = v & 0x3F
        else:
            bits = v & (0x4 | 0x8 | 0xC0 | 0x100 | 0x200 | 0x400)
            if bits & 0x8 and not bits & 0xC0:
                bits |= 0xC0                    # (implicit return: full addresses)
        src_bits = min(_f(v, 28, 4), 12)
        src_id = _f(v, 16, 12) & ((1 << src_bits) - 1) if src_bits else 0
        return bits | src_id << 16 | src_bits << 28

    def _write_control(self, v: int) -> None:
        old = self.control
        if not v & ACTIVE:                      # (trTeActive 0: the component resets)
            if old & ENABLE:
                self._disable()
            self.reset_registers()
            self.sys.attach(self.index)
            return
        new = self._legal_control(v)
        if not old & ACTIVE:                    # (released from reset: defaults)
            new &= ACTIVE | 1 << 24 | 0x70 | 0xF << 20 | 3 << 16
        new |= old & (1 << STALL_OR_OVERFLOW)
        if v & 1 << STALL_OR_OVERFLOW:          # (W1C)
            new &= ~(1 << STALL_OR_OVERFLOW)
        was_enabled, was_tracing = self.enabled, self.tracing and self.enabled
        if not was_enabled and new & ENABLE:
            new &= ~(1 << STALL_OR_OVERFLOW)    # (cleared when trace is enabled)
        self.control = new
        if not was_enabled and self.enabled:
            self._enable()
        elif was_enabled and not self.enabled:
            self._disable()
        elif self.enabled and was_tracing != self.tracing:
            self._tracing_changed(was_tracing)

    # -- the protocols ------------------------------------------------------------------

    def _clock(self) -> int:
        if _f(self.ts.control, 4, 3) == 2:
            return self.sys.harts[0].cycles
        return self.m.cycles

    def _src(self) -> tuple[int, int]:
        f = self.features
        bits = 0 if self.control >> INHIBIT_SRC & 1 else _f(f, 28, 4)
        return _f(f, 16, 12), bits

    def emode(self) -> E.Mode:
        f, c = self.features, self.control
        return E.Mode(full_address=bool(f & 1), implicit_exception=bool(f & 2),
                      sijump=bool(f & 4), implicit_return=bool(f & 8),
                      branch_prediction=bool(f & 0x10), jump_target_cache=bool(f & 0x20),
                      sync_mode=_f(c, 16, 2), sync_max=_f(c, 20, 4))

    def nmode(self) -> N.Mode:
        f, c = self.features, self.control
        sid, sbits = self._src()
        return N.Mode(htm=_f(c, 4, 3) == 6, implicit_return=_f(f, 6, 2) if f & 8 else 0,
                      repeated_history=bool(f & 0x100), sijump=bool(f & 4),
                      all_jumps=bool(f & 0x200), extend_msb=bool(f & 0x400),
                      context=bool(_f(c, CTX)), sync_mode=_f(c, 16, 2),
                      sync_max=_f(c, 20, 4), src_bits=sbits, src_id=sid,
                      timestamps=self.ts.enabled)

    def _enable(self) -> None:
        self.lost = False
        self.packets = []
        self.sys.attach(self.index)
        if self.format == 0:
            self.enc = E.Encoder(self.params, self.emode(), self._out_e)
            self.enc.start()
        else:
            self.enc = N.Encoder(self.xlen, self.nmode(), self._out_n,
                                 now=lambda: self.ts.value(self._clock()),
                                 has_h=self.m.csrs.has_h)
            if not self.tracing or _f(self.control, 4, 3) == 0:
                self.enc.tracing = True         # (trace starts disabled: EVCODE 4)
                self.enc.sync_due = None
                self.enc._emit(N.CORRELATION, [("EVCODE", N.EV_DISABLED, 4),
                                               ("CDF", 1 if self.enc.mode.htm else 0, 2),
                                               ("ICNT", 0, 0)] +
                               ([("HIST", 1, 0)] if self.enc.mode.htm else []),
                               "trace disabled")
                self.enc.tracing = False
            else:
                self.enc.start(N.SYNC_ENABLE)

    def _disable(self) -> None:
        if self.enc is None:
            return
        if self.format == 0:
            self.enc.disable()
        else:
            self.enc.stop(N.EV_DISABLED, self.m.pc)
        self.sys.flush()
        self.sys.attach(self.index)

    def _tracing_changed(self, was: bool) -> None:
        """trTeInstTracing changed between blocks (software, a trigger)."""
        if self.enc is None:
            return
        if self.format == 0:
            if was:
                self.enc.stop()
        else:
            if was:
                self.enc.stop(N.EV_DISABLED, self.m.pc)
            elif _f(self.control, 4, 3):
                self.enc.start(N.SYNC_ENABLE)
        self._ext_out(started=not was)

    def _ext_out(self, *, started: bool) -> None:
        for k in range(EXT_OUTPUTS):
            ev = _f(self.ext_out, 4 * k, 4)
            if ev & (1 if started else 2):
                self.sys.encoder_output(self, k)

    # -- output -----------------------------------------------------------------------

    def _out_e(self, pkt: E.Packet) -> None:
        payload = E.encode_payload(pkt)
        sid, sbits = self._src()
        ts = self.ts.value(self._clock()) if self.ts.enabled else None
        data = encapsulate(payload, sid, sbits, ts)
        pkt.raw = data
        self.packets.append(pkt)
        self._send(data, "e")

    def _out_n(self, msg: N.Message) -> None:
        self.packets.append(msg)
        self._send(msg.raw, "n")

    def _send(self, data: bytes, kind: str) -> None:
        if self.lost:
            return                              # (dropped until there is room again)
        if self.queue or not self.sys.accept(self, data, kind):
            if self.control >> STALL_ENA & 1:
                self.queue.append((data, kind))
                self.m.trace_stall = True
                self.control |= 1 << STALL_OR_OVERFLOW
                return
            self.control |= 1 << STALL_OR_OVERFLOW
            self.lost = True

    def drain(self) -> bool:
        """Retry the stalled packets (the hart's step while it stalls): True
        when they all went."""
        while self.queue:
            data, kind = self.queue[0]
            if not self.sys.accept(self, data, kind):
                return False
            self.queue.pop(0)
        self.m.trace_stall = False
        return True

    def _recover(self) -> None:
        """After a loss, once the sink takes packets again (2's rule: a
        synchronisation after a gap)."""
        if not (self.lost and self.sys.can_accept(self)):
            return
        self.lost = False
        if self.format == 0:
            self.enc.lost()
        else:
            self.enc.overflow()
            if self.tracing:
                self.enc.start(N.SYNC_OVERRUN)

    # -- the ingress side ------------------------------------------------------------------

    def block(self, b: I.Block) -> None:
        if not self.enabled or self.enc is None:
            return
        self._recover()
        c = self.control
        pulses = b.trigger
        ext = self.ext_pending
        self.ext_pending = 0
        on = off = False
        notify_ext = False
        if c >> INST_TRIG_ENABLE & 1:
            on = bool(pulses & I.TRACE_ON)
            off = bool(pulses & I.TRACE_OFF)
        for k in range(EXT_INPUTS):
            if ext >> k & 1:
                act = _f(self.ext_in, 4 * k, 4)
                if act == 2 and c >> INST_TRIG_ENABLE & 1:
                    on = True
                elif act == 3 and c >> INST_TRIG_ENABLE & 1:
                    off = True
                elif act == 4:
                    notify_ext = True
        was = self.tracing
        traced = was or on
        if on and not was:
            self.control |= INST_TRACING
            if self.format == 1:
                self.enc.start(N.SYNC_ENABLE)
            self._ext_out(started=True)
        comps = [cm.evaluate(b) for cm in self.comp]
        enabled_filters = [f for k, f in enumerate(self.filt) if self.filters_reg >> k & 1]
        qualified = traced and _f(c, 4, 3) != 0
        if enabled_filters:
            res = [f.matches(b, [o for o, _ in comps]) for f in enabled_filters]
            qualified = qualified and any(res)
        notify = bool(pulses & I.TRACE_NOTIFY) or any(n for _, n in comps)
        if self.format == 0:
            self.enc.push(b, qualified, notify or notify_ext)
        else:
            n = self.enc
            if qualified and not n.tracing:
                n.start(N.SYNC_ENABLE)
            elif not qualified and n.tracing:
                n.stop(N.EV_DISABLED, b.iaddr)
            if qualified:
                code = N.SYNC_EVENT if notify else N.SYNC_EXTERNAL if notify_ext else None
                n.push(b, code)
        if off and self.tracing:
            self.control &= ~INST_TRACING       # (traced through this one, then off)
            if self.format == 0:
                self.enc.stop()
            else:
                self.enc.stop(N.EV_DISABLED, b.npc)
            self._ext_out(started=False)

    def halted(self, on: bool) -> None:
        """E-Trace Table 4.7's *halted*: the last instruction, a support
        packet (E) / ProgTraceCorrelation EVCODE 0 (N); after Debug Mode the
        trace restarts with a sync (N: SYNC 3)."""
        if not self.enabled or self.enc is None:
            return
        if on:
            if self.format == 0:
                self.enc.stop()
            else:
                self.enc.stop(N.EV_DEBUG, self.m.pc)
        elif self.format == 1 and self.tracing:
            self.enc.start(N.SYNC_DEBUG)

    def reset(self) -> None:
        if not self.enabled or self.enc is None:
            return
        if self.format == 0:
            self.enc.stop()
        else:
            self.enc.stop(N.EV_DISABLED, self.m.pc)
            if self.tracing:
                self.enc.start(N.SYNC_RESET)


def encapsulate(payload: bytes, src_id: int = 0, src_bits: int = 0,
                timestamp: int | None = None, flow: int = 0) -> bytes:
    """The Unformatted Trace & Diagnostic Data Packet Encapsulation 1.0 ch. 2
    (normal encapsulation): header (length 4:0, flow 6:5, extend 7), srcID
    (``src_bits``), timestamp (T = 6 bytes when given), payload — a bit
    stream after the header, LSB first; length = ceil((payload bits +
    srcID bits % 8) / 8)."""
    bits = 0
    n = 0
    bits |= src_id << n
    n += src_bits
    if timestamp is not None:
        bits |= (timestamp & ((1 << 48) - 1)) << n
        n += 48
    bits |= int.from_bytes(payload, "little") << n
    n += 8 * len(payload)
    total = (n + 7) // 8
    if payload and payload[-1] & 0x80 and n % 8:  # (the padding: the payload's sign —
        bits |= ((1 << (8 * total)) - 1) & ~((1 << n) - 1)   # "don't care", 2.2.4)
    length = total - src_bits // 8 - (6 if timestamp is not None else 0)
    header = length | flow << 5 | (0x80 if timestamp is not None else 0)
    return bytes([header]) + bits.to_bytes(total, "little")


def decapsulate(data: bytes, src_bits: int = 0, timestamps: bool = False) -> list[
        tuple[int, int | None, bytes]]:
    """The packets in a byte stream: (srcID, timestamp, payload); null
    packets (length 0) are skipped."""
    out = []
    i = 0
    s_bytes = src_bits // 8
    t = 6 if timestamps else 0
    while i < len(data):
        h = data[i]
        length = h & 0x1F
        i += 1
        if length == 0:
            continue
        ext = h >> 7 & 1
        total = s_bytes + (t if ext else 0) + length
        body = int.from_bytes(data[i:i + total], "little")
        i += total
        sid = body & ((1 << src_bits) - 1)
        body >>= src_bits
        ts = None
        if ext:
            ts = body & ((1 << 48) - 1)
            body >>= 48
        pbits = 8 * total - src_bits - (48 if ext else 0)
        nbytes = pbits // 8                     # (a whole-byte payload: the rest is padding)
        payload = (body & ((1 << (8 * nbytes)) - 1)).to_bytes(nbytes, "little")
        out.append((sid, ts, payload))
    return out


def sync_sequence(src_bits: int = 0, timestamps: bool = False) -> bytes:
    """2.3: N + 1 null bytes (N = 31 + T + S), the last a null.alignment —
    the first non-null byte after it starts a packet."""
    n = 31 + (6 if timestamps else 0) + src_bits // 8
    return bytes(n) + b"\x80"


# -- the funnel and the RAM sink ---------------------------------------------------------


class Funnel:
    """trFunnelControl / Impl / DisInput: the encoders' packets, in arrival
    order (each is atomic), to the sink."""

    def __init__(self) -> None:
        self.control = 0
        self.disinput = 0

    def get(self, off: int) -> int:
        if off == 0:
            return self.control | EMPTY
        if off == 4:
            return 1 | COMP_FUNNEL << 8
        if off == 8 and self.control & ACTIVE:
            return self.disinput
        return 0

    def put(self, off: int, v: int) -> None:
        if off == 0:
            self.control = v & (ACTIVE | ENABLE) if v & ACTIVE else 0
            if not self.control & ACTIVE:
                self.disinput = 0
        elif off == 8 and self.control & ACTIVE:
            self.disinput = v & ((1 << MAX_ENCODERS) - 1)

    def passes(self, index: int) -> bool | None:
        """None: the input is disabled (read and discarded); False: the
        funnel is off (nothing passes)."""
        if not (self.control & ACTIVE and self.control & ENABLE):
            return False
        return None if self.disinput >> index & 1 else True


class RamSink:
    """The Trace RAM Sink (Table "Trace RAM Sink Registers"): SRAM (16 KB,
    trRamStart 0) or SMEM (a range of system memory written below the
    caches)."""

    def __init__(self, system: TraceSystem) -> None:
        self.sys = system
        self.sram = bytearray(SRAM_SIZE)
        self.reset()

    def reset(self) -> None:
        self.control = 0
        self.start = 0
        self.limit = SRAM_SIZE - 4
        self.wp = 0
        self.rp = 0
        self.wrap = False
        self.partial = b""                      # (bytes not yet a whole word)
        self.since_sync = 0
        self.kind = "e"

    @property
    def smem(self) -> bool:
        return bool(self.control >> 4 & 1)

    @property
    def enabled(self) -> bool:
        return bool(self.control & ACTIVE and self.control & ENABLE)

    def get(self, off: int) -> int:
        if off == 0x00:
            return (self.control & ~EMPTY) | (0 if self.partial else EMPTY)
        if off == 0x04:
            return 1 | COMP_RAMSINK << 8 | 1 << 12 | 1 << 13
        if not self.control & ACTIVE:
            return 0
        if off == 0x10:
            return self.start & M32
        if off == 0x14:
            return self.start >> 32
        if off == 0x18:
            return self.limit & M32
        if off == 0x1C:
            return self.limit >> 32
        if off == 0x20:
            return (self.wp & M32 & ~3) | int(self.wrap)
        if off == 0x24:
            return self.wp >> 32
        if off == 0x28:
            return self.rp & M32
        if off == 0x2C:
            return self.rp >> 32
        if off == 0x40:
            return self.word(self.rp)
        return 0

    def read_data(self) -> int:
        """trRamData with its side effect (RP advances, wraps at Limit)."""
        v = self.word(self.rp)
        self.rp = self.start if self.rp >= self.limit else self.rp + 4
        return v

    def word(self, addr: int) -> int:
        if self.smem:
            return self.sys.phys_word(addr)
        a = addr % SRAM_SIZE
        return int.from_bytes(self.sram[a:a + 4], "little")

    def put(self, off: int, v: int) -> None:
        v &= M32
        if off == 0x00:
            if not v & ACTIVE:
                self.reset()
                return
            was = self.enabled
            mode = _f(v, 4)
            keep = ACTIVE | ENABLE | 1 << 8 | 7 << 12
            self.control = (v & keep) | mode << 4
            if self.smem != bool(mode):
                pass
            if not self.smem:
                self.start = 0
            if was and not self.enabled:
                self.flush()
            return
        if not self.control & ACTIVE:
            return
        if self.enabled and off in (0x10, 0x14, 0x18, 0x1C, 0x20, 0x24):
            return                              # (not while trace is on)
        if off == 0x10 and self.smem:
            self.start = (self.start & ~M32) | (v & ~3)
        elif off == 0x14 and self.smem:
            self.start = (self.start & M32) | v << 32
        elif off == 0x18:
            lim = v & ~3
            self.limit = min(lim, SRAM_SIZE - 4) if not self.smem else \
                (self.limit & ~M32) | lim
        elif off == 0x1C and self.smem:
            self.limit = (self.limit & M32) | v << 32
        elif off == 0x20:
            self.wp = (self.wp & ~M32) | (v & ~3)
            self.wrap = bool(v & 1)
        elif off == 0x24 and self.smem:
            self.wp = (self.wp & M32) | v << 32
        elif off == 0x28:
            self.rp = (self.rp & ~M32) | (v & ~3)
            if not self.smem:
                self.rp = min(self.rp, self.limit)
        elif off == 0x2C:
            self.rp = (self.rp & M32) | v << 32

    # -- writing ----------------------------------------------------------------------

    def _space(self) -> int:
        """Bytes that fit before a stop-on-wrap stops the sink."""
        if not self.control >> 8 & 1:
            return 1 << 62
        return self.limit + 4 - self.wp - len(self.partial)

    def can_accept(self, n: int = 1) -> bool:
        return self.enabled and self._space() >= n

    @property
    def stopped(self) -> bool:
        """Stopped on wrap: full, nothing overwritten."""
        return bool(self.control >> 8 & 1 and self.wrap and not self.control & ENABLE)

    def write(self, data: bytes, kind: str) -> bool:
        if not self.enabled:
            return False
        freq = _f(self.control, 12, 3)
        if freq and kind == "e" and self.since_sync >= 1 << (freq + 6):
            data = sync_sequence(*self.sys.encap) + data
            self.since_sync = 0
        if self._space() < len(data):
            self.control &= ~ENABLE             # (trRamStopOnWrap: full)
            self.flush()
            return False
        self.since_sync += len(data)
        buf = self.partial + data
        self.kind = kind
        while len(buf) >= 4:
            self._store(buf[:4])
            buf = buf[4:]
        self.partial = buf
        return True

    def _store(self, w: bytes) -> None:
        if self.smem:
            self.sys.phys_write(self.wp, w)
        else:
            a = self.wp % SRAM_SIZE
            self.sram[a:a + 4] = w
        if self.wp >= self.limit:
            self.wp = self.start
            self.wrap = True
            if self.control >> 8 & 1:           # (trRamStopOnWrap: full — the sink stops)
                self.control &= ~ENABLE
        else:
            self.wp += 4

    def flush(self) -> None:
        """Disabling flushes the last word, padded with idle bytes (E-Trace's
        null.idle 0x00, N-Trace's 0xFF)."""
        if self.partial:
            pad = b"\x00" if self.kind == "e" else b"\xff"
            self._store(self.partial + pad * (4 - len(self.partial)))
            self.partial = b""

    def contents(self) -> bytes:
        """The trace in the buffer, oldest byte first (a view: no side
        effect) — with the bytes not yet a whole word."""
        def rd(a: int, n: int) -> bytes:
            if self.smem:
                return self.sys.phys_read(a, n)
            a %= SRAM_SIZE
            return bytes(self.sram[a:a + n])
        if self.wrap:
            data = rd(self.wp, self.limit + 4 - self.wp) + rd(self.start, self.wp - self.start)
        else:
            data = rd(self.start, self.wp - self.start)
        return data + self.partial


# -- the system ------------------------------------------------------------------------


class TraceBus(Device):
    """The trace components on every hart's bus (byte-granular, the
    hart's little-endian order: Device's load / store)."""

    def __init__(self, system: TraceSystem) -> None:
        self.sys = system
        self.base = MAP["encoder"]
        self.size = MAP["ramsink"] + 0x1000 - MAP["encoder"]

    def _target(self, off: int):
        s = self.sys
        a = self.base + off
        if a >= MAP["ramsink"]:
            return s.sink, a - MAP["ramsink"]
        if a >= MAP["funnel"]:
            return (s.funnel, a - MAP["funnel"]) if s.funnel is not None else (None, 0)
        k = off >> 12
        return (s.encoders[k], off & 0xFFF) if k < len(s.encoders) else (None, 0)

    def reg_read(self, m, off: int, side_effects: bool = True) -> int:
        c, r = self._target(off)
        if c is None:
            return 0
        if c is self.sys.sink and r == 0x40 and side_effects:
            return c.read_data()
        return c.get(r)

    def reg_write(self, m, off: int, value: int, size: int) -> None:
        c, r = self._target(off)
        if c is not None:
            c.put(r, value)


class TraceSystem:
    """The trace components of a set of harts (see the module docstring):
    ``encoders`` (one a hart), ``funnel`` (with several), ``sink``."""

    def __init__(self, harts: list, *, dm: Any = None) -> None:
        self.harts = list(harts)
        self.dm = dm
        self.encoders = [TraceEncoder(self, k, m) for k, m in enumerate(self.harts)]
        self.funnel = Funnel() if len(self.harts) > 1 else None
        self.sink = RamSink(self)
        self.bus = TraceBus(self)
        self.encap = (0, False)
        if dm is not None:
            dm.on_ext_trigger = self.dm_output
        #: each hart's ingress port — on the hart (m.ingress) only while its
        #: encoder is enabled: a hart nobody traces pays nothing
        self.ports = []
        for k, m in enumerate(self.harts):
            port = m.ingress if m.ingress is not None else I.IngressPort(m)
            port.consumers.append(self.encoders[k])
            self.ports.append(port)
            self.attach(k)
            m.devices.insert(0, self.bus)
            m.io_regions = list(getattr(m, "io_regions", []) or []) + [
                (self.bus.base, self.bus.base + self.bus.size)]
            m.trace_encoder = self.encoders[k]
            m.tm_ext_output = (lambda k_: (lambda out: self._tm_output(k_, out)))(k)

    def attach(self, k: int) -> None:
        """Put hart ``k``'s ingress port on it when its encoder is enabled
        (or the port has other consumers / a record), else take it off."""
        port = self.ports[k]
        m = self.harts[k]
        others = [c for c in port.consumers if c is not self.encoders[k]]
        want = self.encoders[k].enabled or bool(others) or port.log is not None
        m.ingress = port if want else None

    # -- routing ----------------------------------------------------------------------

    def accept(self, enc: TraceEncoder, data: bytes, kind: str) -> bool:
        if self.funnel is not None:
            p = self.funnel.passes(enc.index)
            if p is None:
                return True                     # (a disabled input: discarded)
            if not p:
                return False
        sid, sbits = enc._src()
        self.encap = (sbits, enc.ts.enabled)
        return self.sink.write(data, kind)

    def can_accept(self, enc: TraceEncoder) -> bool:
        if self.funnel is not None and self.funnel.passes(enc.index) is False:
            return False
        return self.sink.can_accept(64)

    def flush(self) -> None:
        pass

    def phys_write(self, pa: int, data: bytes) -> None:
        m = self.harts[0]
        for k, b in enumerate(data):
            m.phys_write_byte(pa + k, b)

    def phys_read(self, pa: int, n: int) -> bytes:
        m = self.harts[0]
        return bytes(m.phys_read_byte(pa + k) for k in range(n))

    def phys_word(self, pa: int) -> int:
        return int.from_bytes(self.phys_read(pa, 4), "little")

    # -- cross triggers (WIRING) ---------------------------------------------------------

    def _tm_output(self, hart: int, out: int) -> None:
        """A trigger module's external output (actions 8 / 9)."""
        if out < 2:
            self.encoders[hart].ext_pending |= 1 << out

    def encoder_output(self, enc: TraceEncoder, out: int) -> None:
        if out == 0:
            enc.m.tm.ext_input(0)
        elif out == 1 and self.dm is not None:
            self.dm.external_trigger(0)

    def dm_output(self, trigger: int) -> None:
        """The Debug Module notified its external trigger ``trigger``."""
        if trigger == 1:
            for e in self.encoders:
                e.ext_pending |= 1 << 2

    def probe_input(self) -> None:
        for e in self.encoders:
            e.ext_pending |= 1 << 3
