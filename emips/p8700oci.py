"""The rest of the P8700's On-Chip Instrumentation (LF16; MD01502 rev. 1.84
5.14.2 GCR.Debug, 5.14.6 the Fast Debug Channel, 5.14.7 the Trace Funnel,
1.2.2 "MIPS EJTAG with RISC-V Trace"; the P5600's MD01025 15.1.3 for the
Coherence Manager's performance-counter events).

**Trace** (:class:`P8700Trace`, ``Cluster.trace``): each hart has LF11's
RISC-V trace encoder (:class:`emips.rvtrace.TraceEncoder`: E-Trace or
N-Trace, the Trace Control Interface's encoder registers) in its drseg at
0x8000 (emips.p8700dbg; "sampling registers" — the place is assumed); their
messages, in the RISC-V Trace Encapsulation (srcID = the hart's index), go
to the cluster's Trace Funnel.

**The Trace Funnel** (:class:`TraceFunnel`, 5.14.7, GCR_BASE + 0x3F100):
CONTROL (WE gates every other field of a write; EN; TM reads 0 —
Trace-To: the on-chip memory wraps, BF set —; TR resets the pointers and
BF; RM reads the line at READPTR into DATA[0-7]; TL / TO the program /
probe lock; IDLE 1: words reach the SRAM at once; OfC 0 — on-chip only;
CR / CAL / STCE stored), CONFIG (16 KB: SZ 6, ONT 1, OfT 0, SRC_B 3 — 4-bit
sources —, TAG_B 0, REV 1), WRITEPTR / READPTR (byte addresses of 64-byte
lines; reading DATA[7] reads the next line), DATA[0-7], STUSER / STENABLE
(stored: no system trace source in eMIPS).  The CM3's own PDtrace
(GCR.Debug TCBCONTROLD / E / PERFCNTR) "only exists if the CM3 is
configured with PDTrace": this one is not — they read 0.

**The FDC** (:class:`FDC`, 5.14.6, GCR_BASE + 0x3F000): transmit and
receive FIFOs of 16 entries (a channel 0-15 and 32 bits each), FDACSR,
FDCFG (Tx / Rx_IntThresh: an interrupt while the FIFO is full / not empty /
almost full — the table's encodings for both —, on APLIC source
:data:`FDC_SOURCE`, assumed), FDSTAT, FDRX (pops; empty: 0), FDTX0-15 (the
channel = the register; full: dropped).  The probe's side is the DBU TAP's
FDC register (38 bits, assumed: Capture — 37 a transmit entry is there, 36
the receive FIFO is full, 35:32 its channel, 31:0 its data; Update — 37
the probe sends 35:32 / 31:0, 36 it took the entry shown).

**The CM3's performance counters** (:class:`CMCounters`, 5.14.2.4-11,
GCR_BASE + 0x900): PC_CTL (per counter COUNTON / RESET, the cycle counter's,
PERF_OVF_STOP, PERF_INT_EN; PERF_NUM_CNT 2), PC_OV (RW1C), PC_EVENT,
PC_CYCL, PC_QUAL0 / 1, PC_CNT0 / 1 (32 bits; an overflow at 0xFFFFFFFF ->
0); the events of MD01025 Table 15.1 the model sees: 0 Request_Count (the
request port, the CCA group by coherence, the command and the target
qualifiers of Table 15.2; burst lengths: every eMIPS request is a line,
the group's bits are ignored), 8 L2_Util (Table 15.8: a new request
started / a start taken), 9 L2_Hit (Table 15.9: read hit / miss, full /
partial write hit / miss, allocation, the source port); the others never
count.  An overflow with PERF_INT_EN raises APLIC source
:data:`CM_PERF_SOURCE` (assumed) until PC_OV is cleared.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from . import rvingress as I
from .rvtrace import TraceEncoder, TraceSystem

M32, M64 = 0xFFFFFFFF, (1 << 64) - 1
#: GCR offsets (Table 5.24)
FDC_BASE, TRF_BASE, OCI_END = 0x3F000, 0x3F100, 0x3F200
DEBUG_BASE, DEBUG_END = 0x0800, 0x0A00
#: the APLIC sources (assumed: 1-5 are the console's / the CDMA's)
FDC_SOURCE, CM_PERF_SOURCE = 6, 7

# -- the Trace Funnel (5.14.7) ------------------------------------------------------------

TRF_SIZE, TRF_SZ = 16 * 1024, 6                 # (2^(SZ + 8) bytes)
LINE = 64                                       # (DATA[0-7]: 8 doublewords)
WE, STCE, IDLE, TL, TO, RM, TR, BF = (1 << 31, 1 << 20, 1 << 19, 1 << 18, 1 << 17,
                                      1 << 16, 1 << 15, 1 << 14)
CR_MASK, CAL, PIB_MCP, OFC, EN = 7 << 5, 1 << 4, 1 << 2, 1 << 1, 1 << 0
#: CONFIG: TAG_B 0 (5 bits), SRC_B 3 (4 bits), SZ, ONT, OfT 0, REV 1
TRF_CONFIG = 3 << 21 | TRF_SZ << 17 | 1 << 5 | 1
CONTROL, CONFIG, WRITEPTR, READPTR, DATA0, STUSER, STENABLE = (
    0x00, 0x08, 0x10, 0x18, 0x20, 0x60, 0x68)


class TraceFunnel:
    """The cluster's Trace Funnel with its on-chip trace memory."""

    def __init__(self) -> None:
        self.sram = bytearray(TRF_SIZE)
        self.control = 0
        self.lock_owner: int | None = None
        self.wp = self.rp = 0
        self.data = [0] * 8
        self.stuser = self.stenable = 0
        self.partial = b""
        self.kind = "e"
        self.since_sync = 0
        self.encap = (0, False)

    @property
    def enabled(self) -> bool:
        return bool(self.control & EN)

    @property
    def wrap(self) -> bool:
        """The memory wrapped (BF): the oldest words are overwritten."""
        return bool(self.control & BF)

    # -- registers --------------------------------------------------------------------

    def get(self, off: int) -> int:
        if off == CONTROL:
            v = self.control | IDLE
            if self.lock_owner is not None:
                v |= TL | (TO if self.lock_owner else 0)
            return v
        if off == CONFIG:
            return TRF_CONFIG
        if off == WRITEPTR:
            return self.wp
        if off == READPTR:
            return self.rp
        if DATA0 <= off < DATA0 + 64:
            return self.data[(off - DATA0) >> 3]
        if off == STUSER:
            return self.stuser
        if off == STENABLE:
            return self.stenable
        return 0

    def read(self, off: int) -> int:
        """A read with its side effect: DATA[7] reads the next line."""
        v = self.get(off)
        if off == DATA0 + 56:
            self.rp = (self.rp + LINE) % TRF_SIZE
            self._read_line()
        return v

    def _read_line(self) -> None:
        a = self.rp
        self.data = [int.from_bytes(self.sram[a + 8 * k:a + 8 * k + 8], "little")
                     for k in range(8)]

    def put(self, off: int, v: int, by: int = 0) -> None:
        """A write (``by``: the requester for the lock — 0 the program, 1 the
        probe)."""
        if off == CONTROL:
            if not v & WE:
                return
            if v & TL:                          # (the lock: TL with the owner in TO)
                if self.lock_owner is None:
                    self.lock_owner = 1 if v & TO else 0
            elif self.lock_owner is not None and self.lock_owner == (1 if v & TO else 0):
                self.lock_owner = None
            keep = STCE | CR_MASK | CAL | EN
            was = self.enabled
            self.control = (self.control & ~keep) | (v & keep)
            if was and not self.enabled:
                self.flush()
            if v & TR:
                self.wp = self.rp = 0
                self.control &= ~BF
                self.partial = b""
            if v & RM:
                self._read_line()
        elif off == WRITEPTR:
            self.wp = v & (TRF_SIZE - LINE)
        elif off == READPTR:
            self.rp = v & (TRF_SIZE - LINE)
        elif DATA0 <= off < DATA0 + 64:
            self.data[(off - DATA0) >> 3] = v & M64
        elif off == STUSER:
            self.stuser = v & M32
        elif off == STENABLE:
            self.stenable = v & M32

    # -- the sink -------------------------------------------------------------------------

    def can_accept(self, n: int = 1) -> bool:
        return self.enabled

    def write(self, data: bytes, kind: str) -> bool:
        if not self.enabled:
            return False
        self.kind = kind
        buf = self.partial + data
        while len(buf) >= 8:
            self._store(buf[:8])
            buf = buf[8:]
        self.partial = buf
        return True

    def _store(self, w: bytes) -> None:
        self.sram[self.wp:self.wp + 8] = w
        self.wp += 8
        if self.wp >= TRF_SIZE:                 # (Trace-To: wrap, BF)
            self.wp = 0
            self.control |= BF

    def flush(self) -> None:
        if self.partial:
            pad = b"\x00" if self.kind == "e" else b"\xff"
            self._store(self.partial + pad * (8 - len(self.partial)))
            self.partial = b""

    def contents(self) -> bytes:
        """The trace in the memory, oldest byte first (a view)."""
        if self.control & BF:
            data = bytes(self.sram[self.wp:]) + bytes(self.sram[:self.wp])
        else:
            data = bytes(self.sram[:self.wp])
        return data + self.partial

    def lines(self) -> list[str]:
        lock = "unlocked" if self.lock_owner is None else \
            f"locked by the {'probe' if self.lock_owner else 'program'}"
        return [f"Trace Funnel: {'enabled' if self.enabled else 'disabled'}, {lock}, "
                f"WRITEPTR 0x{self.wp:x}{' (wrapped: BF)' if self.control & BF else ''}, "
                f"READPTR 0x{self.rp:x}, {TRF_SIZE // 1024} KB on chip"]


class P8700Trace(TraceSystem):
    """The cluster's harts' trace encoders into the Trace Funnel."""

    def __init__(self, cl: Any) -> None:
        self.cl = cl
        self.harts = list(cl.machines)
        self.dm = None
        self.funnel = None
        self.trf = self.sink = TraceFunnel()
        self.bus = None
        self.encap = (0, False)
        self.encoders: list[TraceEncoder] = []
        self.ports: list[I.IngressPort] = []
        for k, m in enumerate(self.harts):
            self.encoders.append(None)          # (filled by rehart)
            self.ports.append(None)
            self.rehart(k, m)

    def rehart(self, k: int, m: Any) -> None:
        """Hart ``k`` (made afresh: a reset, VP_RUN) gets a new encoder (at
        its reset values) and ingress port."""
        if k < len(self.harts):
            self.harts[k] = m
        enc = TraceEncoder(self, k, m)
        self.encoders[k] = enc
        port = m.ingress if m.ingress is not None else I.IngressPort(m)
        port.consumers.append(enc)
        self.ports[k] = port
        self.attach(k)
        m.trace_encoder = enc
        m.tm_ext_output = (lambda k_: (lambda out: self._tm_output(k_, out)))(k)
        if m.hybrid is not None:
            m.hybrid.trace = enc

    def accept(self, enc: TraceEncoder, data: bytes, kind: str) -> bool:
        self.encap = self.trf.encap = (enc._src()[1], enc.ts.enabled)
        return self.trf.write(data, kind)

    def can_accept(self, enc: TraceEncoder) -> bool:
        return self.trf.can_accept(64)

    def encoder_output(self, enc: TraceEncoder, out: int) -> None:
        if out == 0:
            enc.m.tm.ext_input(0)

    def phys_write(self, pa: int, data: bytes) -> None:     # (no SMEM sink)
        pass


# -- the Fast Debug Channel (5.14.6) -------------------------------------------------------

FIFO = 16
FDACSR, FDCFG, FDSTAT, FDRX, FDTX0 = 0x00, 0x08, 0x10, 0x18, 0x20
FDACSR_VALUE = 253 << 24 | 2 << 16 | 1 << 12


class FDC:
    """The cluster's Fast Debug Channel (see the module docstring)."""

    def __init__(self, cl: Any) -> None:
        self.cl = cl
        self.tx: deque[tuple[int, int]] = deque()
        self.rx: deque[tuple[int, int]] = deque()
        self.cfg = 0
        self.level = False
        #: what the probe received (channel, data), in order (the views)
        self.received: list[tuple[int, int]] = []

    def get(self, off: int) -> int:
        if off == FDACSR:
            return FDACSR_VALUE
        if off == FDCFG:
            return self.cfg | FIFO << 8 | FIFO
        if off == FDSTAT:
            v = len(self.tx) << 24 | len(self.rx) << 16
            if self.rx:
                v |= self.rx[0][0] << 4
            return v | (8 if not self.rx else 0) | (4 if len(self.rx) == FIFO else 0) \
                | (2 if not self.tx else 0) | (1 if len(self.tx) == FIFO else 0)
        if off == FDRX:
            return self.rx[0][1] if self.rx else 0
        return 0

    def read(self, off: int) -> int:
        v = self.get(off)
        if off == FDRX and self.rx:
            self.rx.popleft()
            self._irq()
        return v

    def put(self, off: int, v: int) -> None:
        if off == FDCFG:
            self.cfg = v & (0xF << 16)
            self._irq()
        elif FDTX0 <= off < FDTX0 + 8 * FIFO and not off & 7:
            if len(self.tx) < FIFO:
                self.tx.append(((off - FDTX0) >> 3, v & M32))
            self._irq()

    @staticmethod
    def _cond(thresh: int, n: int) -> bool:
        return (thresh == 1 and n == FIFO) or (thresh == 2 and n > 0) or \
            (thresh == 3 and n >= FIFO - 1)

    def _irq(self) -> None:
        lvl = self._cond(self.cfg >> 18 & 3, len(self.tx)) or \
            self._cond(self.cfg >> 16 & 3, len(self.rx))
        if lvl != self.level:
            self.level = lvl
            ic = self.cl.ic
            if ic is not None:
                ic.set_input(FDC_SOURCE, lvl)

    # -- the probe's side (the DBU TAP's FDC register) ------------------------------------

    def tap_capture(self) -> int:
        v = (1 << 36) if len(self.rx) == FIFO else 0
        if self.tx:
            ch, d = self.tx[0]
            v |= 1 << 37 | ch << 32 | d
        return v

    def tap_update(self, v: int) -> None:
        if v >> 36 & 1 and self.tx:             # (the probe took the entry it saw)
            self.received.append(self.tx.popleft())
        if v >> 37 & 1 and len(self.rx) < FIFO:
            self.rx.append((v >> 32 & 0xF, v & M32))
        self._irq()

    def lines(self) -> list[str]:
        return [f"FDC: transmit {len(self.tx)}/{FIFO}, receive {len(self.rx)}/{FIFO}, "
                f"thresholds Tx {self.cfg >> 18 & 3} Rx {self.cfg >> 16 & 3}"
                + (", interrupt" if self.level else "")
                + (f"; the probe got {len(self.received)}" if self.received else "")]


# -- the CM3's performance counters (5.14.2.4-11; MD01025 15.1.3) ---------------------------

PC_CTL, PC_OV, PC_EVENT, PC_CYCL = 0x900, 0x920, 0x930, 0x980
PC_QUAL0, PC_CNT0, PC_QUAL1, PC_CNT1 = 0x990, 0x998, 0x9A0, 0x9A8
#: Table 15.2's command bits
CMD_BITS = {"legacy_wr": 15, "legacy_rd": 14, "read_share": 13, "read_own": 12,
            "read_discard": 11, "upgrade": 10, "writeback": 9, "write_inval": 8,
            "invalidate": 6, "copyback": 5, "copyback_inv": 4, "sync": 3}
EVENT_NAMES = {0: "Request_Count", 1: "Coh_Req_Resp", 2: "L2_WR_Data_Util",
               3: "L2_Cmd_Util", 4: "L2_RD_Data_Util", 5: "Sharing_Miss", 6: "RSU_Util",
               8: "L2_Util", 9: "L2_Hit", 16: "IOCU_Request", 17: "IOCU1_Request"}


class CMCounters:
    """The CM3's two performance counters and its cycle counter."""

    def __init__(self, cl: Any) -> None:
        self.cl = cl
        self.ctl = 2                            # (PERF_NUM_CNT)
        self.ov = 0
        self.event = 0
        self.cycl = 0
        self.qual = [0, 0]
        self.cnt = [0, 0]
        self.level = False
        self._since = 0

    def _on(self, k: int) -> bool:
        return bool(self.ctl >> (6 + 2 * k) & 1)

    def _event(self, k: int) -> int:
        return self.event >> 8 * k & 0xFF

    def get(self, off: int) -> int:
        if off == PC_CTL:
            return self.ctl
        if off == PC_OV:
            return self.ov
        if off == PC_EVENT:
            return self.event
        if off == PC_CYCL:
            self.tick()
            return self.cycl
        if off in (PC_QUAL0, PC_QUAL1):
            return self.qual[off == PC_QUAL1]
        if off in (PC_CNT0, PC_CNT1):
            return self.cnt[off == PC_CNT1]
        return 0                                # (TCBCONTROLD / E / PERFCNTR: no CM PDTrace)

    def put(self, off: int, v: int, mask: int) -> None:
        if off == PC_CTL:
            self.tick()
            v = (self.ctl & ~mask) | (v & mask)
            for k, on, reset in ((0, 6, 7), (1, 8, 9)):
                if v >> on & 1 and v >> reset & 1:
                    self.cnt[k] = 0
                    self.ov &= ~(2 << k)
            if v >> 4 & 1 and v >> 5 & 1:
                self.cycl = 0
                self.ov &= ~1
            self.ctl = (v & (3 << 29 | 0x15 << 4)) | 2     # (the RESET bits read 0)
            self._irq()
        elif off == PC_OV:
            self.ov &= ~(v & mask & 7)
            self._irq()
        elif off == PC_EVENT:
            self.event = ((self.event & ~mask) | (v & mask)) & 0xFFFF
        elif off == PC_CYCL:
            self.tick()
            self.cycl = ((self.cycl & ~mask) | (v & mask)) & M32
        elif off in (PC_QUAL0, PC_QUAL1):
            k = off == PC_QUAL1
            self.qual[k] = ((self.qual[k] & ~mask) | (v & mask)) & M64
        elif off in (PC_CNT0, PC_CNT1):
            k = off == PC_CNT1
            self.cnt[k] = ((self.cnt[k] & ~mask) | (v & mask)) & M32

    # -- counting ---------------------------------------------------------------------------

    def tick(self) -> None:
        """The cycle counter follows the cluster's clock."""
        now = self.cl.clock
        n, self._since = now - self._since, now
        if n > 0 and self.ctl >> 4 & 1:
            self._add_cycles(n)

    def _add_cycles(self, n: int) -> None:
        v = self.cycl + n
        if v > M32:
            self.ov |= 1
            v &= M32
            self._overflow()
        self.cycl = v

    def _bump(self, k: int) -> None:
        if self.cnt[k] == M32:
            self.cnt[k] = 0
            self.ov |= 2 << k
            self._overflow()
        else:
            self.cnt[k] += 1

    def _overflow(self) -> None:
        if self.ctl >> 29 & 1:                  # (PERF_OVF_STOP: every counter stops)
            self.ctl &= ~(1 << 4 | 1 << 6 | 1 << 8)
        self._irq()

    def _irq(self) -> None:
        lvl = bool(self.ctl >> 30 & 1 and self.ov)
        if lvl != self.level:
            self.level = lvl
            ic = self.cl.ic
            if ic is not None:
                ic.set_input(CM_PERF_SOURCE, lvl)

    @property
    def armed(self) -> bool:
        return bool(self.ctl & (1 << 6 | 1 << 8))

    def request(self, port: int, cmd: str, coherent: bool, target: str = "memory") -> None:
        """Event 0: a request reached the CM (Table 15.2's groups: port, CCA,
        command, target)."""
        for k in (0, 1):
            if not self._on(k) or self._event(k) != 0:
                continue
            q = self.qual[k]
            if not q >> (24 + port) & 1:
                continue
            if cmd not in ("sync",):            # (the CCA group: not for cache ops / syncs)
                cca = (3 << 19) if coherent else 1 << 21
                if not q & cca:
                    continue
            if not q >> CMD_BITS[cmd] & 1:
                continue
            if not q >> {"memory": 2, "gcr": 1, "mmio": 0}[target] & 1:
                continue
            self._bump(k)

    def l2(self, port: int, kind: str, hit: bool, full: bool = True,
           allocated: bool = False) -> None:
        """Events 8 / 9: an L2 lookup (``kind`` read / write)."""
        for k in (0, 1):
            if not self._on(k):
                continue
            ev, q = self._event(k), self.qual[k]
            if ev == 8:
                if q & 0x11:                    # (a new request started / a start taken)
                    self._bump(k)
            elif ev == 9:
                if not q >> port & 1:
                    continue
                bit = (9 if hit else 8) if kind == "read" else \
                    (14 if full else 13) if hit else (12 if full else 11)
                if not q >> bit & 1:
                    continue
                if not hit and q & (3 << 18) and not q >> (19 if allocated else 18) & 1:
                    continue
                self._bump(k)

    def lines(self) -> list[str]:
        self.tick()
        out = [f"CM performance counters: cycles {self.cycl}"
               f"{' (counting)' if self.ctl >> 4 & 1 else ''}"]
        for k in (0, 1):
            ev = self._event(k)
            out.append(f"  counter {k}: {self.cnt[k]}, event {ev} "
                       f"{EVENT_NAMES.get(ev, '(reserved)')}, qualifier 0x{self.qual[k]:x}"
                       f"{' (counting)' if self._on(k) else ''}"
                       f"{' (overflowed)' if self.ov >> (k + 1) & 1 else ''}")
        return out
