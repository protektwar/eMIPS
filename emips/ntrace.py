"""RISC-V N-Trace (Nexus-based Trace) 1.0 (LF11): the encoder, its
messages and a decoder.  The task group's repository keeps the
reference C code (``refcode/c``:
NexRv — its decoder is the oracle of ``tests/test_ntrace.py``).

**Transport** (ch. 4): each byte is MDO[5:0] << 2 | MSEO[1:0]; a message
starts with MSEO 00, a variable-length field ends on a byte with MSEO 01,
the message's last byte has 11, idle bytes are 0xFF.  Fields go LSB first:
fixed-length fields pack across bytes, a variable-length field starts in
the bits left in the current byte (a new byte when none) and ends at a byte
boundary (its MSB zeros are not sent; at least one bit).  :func:`pack` /
:func:`unpack` do this from :data:`MESSAGES` (the fields of ch. 6 Table 6,
the same as NexRvMsg.h).

**The encoder** (:class:`Encoder`, ch. 9's rules) counts retired halfwords
in I-CNT and follows the ingress blocks (emips.rvingress; 3's Tables 2 /
3):

* trace start: ProgTraceSync (SYNC 5 trace enable, 3 after Debug Mode, 1
  after a reset, 7 after a FIFO overrun — then an Error message first)
  with I-CNT 0 and F-ADDR = the first traced instruction;
* BTM (trTeInstMode 3): a taken branch -> DirectBranch; an uninferable
  jump or a trap -> IndirectBranch with B-TYPE 0 / 2 (exception) / 3
  (interrupt) and U-ADDR (the target XOR the last address, 7.1), emitted
  when the target is known (the next block); HTM (6): branches go into
  HIST (a 1 for taken, after a stop bit), the jumps / traps ->
  IndirectBranchHist (IndirectBranch when HIST is empty, 9's rule 4);
* ResourceFull (6.8) when HIST is full (31 branches: RCODE 1) or I-CNT
  reaches its limit (RCODE 0);
* periodic sync (trTeInstSyncMode / Max: messages, cycles or halfwords)
  and notifications (Trace-notify: SYNC 6, an external trigger: 0) as
  ProgTraceSync (BTM) / IndirectBranchHistSync with B-TYPE 0 (HTM) at the
  next retired instruction (its address, I-CNT before it, 7.5);
* trace stop: ProgTraceCorrelation with EVCODE 0 (Debug Mode) or 4 (trace
  disabled), the I-CNT and in HTM the HIST (CDF 1);
* Ownership (6.1, trTeContext): the V / PRV after a trap or trap return
  that changes them, scontext (and first hcontext) after its write and
  after each synchronising message;
* optional (8): implicit return (trTeInstImplicitReturnMode 1 counting,
  2 partial — the low 16 bits —, 3 full addresses; a stack of 32: a
  return to the predicted address needs no message), repeated history
  (trTeInstEnRepeatedHistory: RepeatBranch for identical branch messages,
  ResourceFull RCODE 2 with HREPEAT for identical full HIST records),
  sequential jumps (sijump), all jumps (trTeInstEnAllJumps: a direct jump
  is reported as a taken branch), extended address MSBs
  (trTeInstExtendAddrMSB: identical most significant bits skipped in
  multiples of 6, 7.6);
* SRC (trTeSrcBits, unless trTeInhibitSrc) after TCODE, TSTAMP last
  (trTsEnable: absolute in synchronising messages, else relative).

**The decoder** (:class:`Decoder`, ch. 10's principles) walks the program:
HIST bits decide the branches first, the rest of I-CNT runs to the
message's end (a BTM DirectBranch's last instruction is a taken branch),
then F-ADDR / U-ADDR give the next PC; ResourceFull adds to the next
I-CNT (RCODE 0) or walks its HIST (1 / 2 HREPEAT times); RepeatBranch
repeats the last branch message.  Choices: ``sim.DEVIATIONS["RISC-V
trace"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import rvingress as I

# TCODEs (Table 6)
OWNERSHIP, DIRECT, INDIRECT, ERROR, SYNC_MSG, DIRECT_SYNC, INDIRECT_SYNC = 2, 3, 4, 8, 9, 11, 12
RESOURCE_FULL, INDIRECT_HIST, INDIRECT_HIST_SYNC, REPEAT, CORRELATION = 27, 28, 29, 30, 33
NAMES = {2: "Ownership", 3: "DirectBranch", 4: "IndirectBranch", 8: "Error",
         9: "ProgTraceSync", 11: "DirectBranchSync", 12: "IndirectBranchSync",
         27: "ResourceFull", 28: "IndirectBranchHist", 29: "IndirectBranchHistSync",
         30: "RepeatBranch", 33: "ProgTraceCorrelation"}
#: the fields after TCODE / SRC: (name, bits) — bits 0 = variable length
MESSAGES: dict[int, tuple[tuple[str, int], ...]] = {
    OWNERSHIP: (("PROCESS", 0),),
    DIRECT: (("ICNT", 0),),
    INDIRECT: (("BTYPE", 2), ("ICNT", 0), ("UADDR", 0)),
    ERROR: (("ETYPE", 4), ("ECODE", 0)),
    SYNC_MSG: (("SYNC", 4), ("ICNT", 0), ("FADDR", 0)),
    DIRECT_SYNC: (("SYNC", 4), ("ICNT", 0), ("FADDR", 0)),
    INDIRECT_SYNC: (("SYNC", 4), ("BTYPE", 2), ("ICNT", 0), ("FADDR", 0)),
    RESOURCE_FULL: (("RCODE", 4), ("RDATA", 0), ("HREPEAT", 0)),
    INDIRECT_HIST: (("BTYPE", 2), ("ICNT", 0), ("UADDR", 0), ("HIST", 0)),
    INDIRECT_HIST_SYNC: (("SYNC", 4), ("BTYPE", 2), ("ICNT", 0), ("FADDR", 0),
                         ("HIST", 0)),
    REPEAT: (("BCNT", 0),),
    CORRELATION: (("EVCODE", 4), ("CDF", 2), ("ICNT", 0), ("HIST", 0)),
}
SYNC_CODES = (OWNERSHIP, SYNC_MSG, DIRECT_SYNC, INDIRECT_SYNC, INDIRECT_HIST_SYNC)
# SYNC (Table 9)
SYNC_EXTERNAL, SYNC_RESET, SYNC_PERIODIC, SYNC_DEBUG, SYNC_ICNT = 0, 1, 2, 3, 4
SYNC_ENABLE, SYNC_EVENT, SYNC_OVERRUN = 5, 6, 7
SYNC_NAMES = {0: "external trigger", 1: "exit from reset", 2: "periodic",
              3: "exit from Debug Mode", 4: "I-CNT full", 5: "trace enable",
              6: "trace event", 7: "FIFO overrun", 9: "exit from power-down"}
# B-TYPE (Table 7)
BT_INDIRECT, BT_TRAP, BT_EXCEPTION, BT_INTERRUPT = 0, 1, 2, 3
# EVCODE (6.12)
EV_DEBUG, EV_DISABLED = 0, 4
#: N-Trace's limits (Table 8) and eMIPS's encoder's sizes
HIST_BITS = 31                      # (HIST: 32 bits with the stop bit, as NexRv)
ICNT_LIMIT = 1 << 16                # (NexRv's enco_opt_limICNT)
MAX_REPEAT = (1 << 18) - 1          # (NTRACE_MAX_BCNT / _HREPEAT)
CALL_STACK = 32                     # (8.2: never deeper than 32)


def _var_bits(v: int) -> int:
    return max(1, v.bit_length())


def pack(tcode: int, fields: list[tuple[str, int, int]]) -> bytes:
    """One message: TCODE then ``fields`` = (name, value, bits — 0 for a
    variable-length field, -N for one sent with N bits: an address with its
    most significant bits, 7.6)."""
    out: list[int] = [tcode << 2]
    used = 6                                # (bits of the current byte's MDO in use)
    allf = list(fields)
    for name, value, bits in allf:
        if bits > 0:                        # fixed: across bytes
            for k in range(bits):
                if used == 6:
                    out.append(0)
                    used = 0
                out[-1] |= (value >> k & 1) << (2 + used)
                used += 1
            continue
        v = value
        if used == 6:
            out.append(0)
            used = 0
        room = 6 - used
        out[-1] |= (v & ((1 << room) - 1)) << (2 + used)
        v >>= room
        left = -bits - room if bits < 0 else 0
        while v or left > 0:
            out.append((v & 0x3F) << 2)
            v >>= 6
            left -= 6
        out[-1] |= 1                        # (MSEO 01: the end of the field)
        used = 6
    out[-1] |= 3                            # (MSEO 11: the end of the message)
    return bytes(out)


@dataclass
class Message:
    tcode: int
    fields: dict[str, int] = field(default_factory=dict)
    src: int | None = None
    tstamp: int | None = None
    raw: bytes = b""
    reason: str = ""
    #: the bits a variable-length field was sent with (7.6's extension)
    widths: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return NAMES.get(self.tcode, f"TCODE {self.tcode}")

    def get(self, k: str, default: int = 0) -> int:
        return self.fields.get(k, default)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.i = 0

    def messages(self) -> list[list[int]]:
        out, cur = [], []
        for b in self.data:
            if not cur and b == 0xFF:
                continue                    # (idle)
            cur.append(b)
            if b & 3 == 3:
                out.append(cur)
                cur = []
        return out


def unpack(raw: list[int] | bytes, *, src_bits: int = 0, tstamp: bool = False) -> Message:
    """A message's fields (``src_bits``: the SRC field's width, ``tstamp``:
    TSTAMP enabled)."""
    raw = list(raw)
    tcode = raw[0] >> 2
    pos = [0, 6]                            # (byte, bit within its MDO)

    def fixed(bits: int) -> int:
        v = 0
        for k in range(bits):
            if pos[1] == 6:
                pos[0] += 1
                pos[1] = 0
            v |= (raw[pos[0]] >> (2 + pos[1]) & 1) << k
            pos[1] += 1
        return v

    width = [0]

    def var() -> int:
        if pos[1] == 6:
            pos[0] += 1
            pos[1] = 0
        v = raw[pos[0]] >> (2 + pos[1])
        n = 6 - pos[1]
        while raw[pos[0]] & 3 == 0:
            pos[0] += 1
            v |= (raw[pos[0]] >> 2) << n
            n += 6
        pos[1] = 6
        width[0] = n
        return v

    m = Message(tcode, raw=bytes(raw))
    if src_bits:
        m.src = fixed(src_bits)
    defs = MESSAGES.get(tcode, ())
    for name, bits in defs:
        if name == "HREPEAT" and m.fields.get("RCODE") != 2:
            continue
        if name == "HIST" and tcode == CORRELATION and m.fields.get("CDF") != 1:
            continue
        m.fields[name] = fixed(bits) if bits else var()
        if not bits:
            m.widths[name] = width[0]
    if tstamp and not (raw[pos[0]] & 3 == 3 and pos[1] == 6 and pos[0] == len(raw) - 1):
        m.tstamp = var()
    return m


# -- the encoder ----------------------------------------------------------------------


@dataclass
class Mode:
    """trTeControl / trTeInstFeatures / trTsControl as N-Trace reads them."""
    htm: bool = True                    # trTeInstMode 6 (False: 3, BTM)
    implicit_return: int = 0            # trTeInstImplicitReturnMode (0: off)
    repeated_history: bool = False      # trTeInstEnRepeatedHistory
    sijump: bool = False                # trTeInstEnSequentialJump
    all_jumps: bool = False             # trTeInstEnAllJumps
    extend_msb: bool = False            # trTeInstExtendAddrMSB
    context: bool = False               # trTeContext: Ownership messages
    sync_mode: int = 1                  # trTeInstSyncMode
    sync_max: int = 4                   # trTeInstSyncMax
    src_bits: int = 0                   # trTeSrcBits (0 or inhibited: no SRC)
    src_id: int = 0                     # trTeSrcID
    timestamps: bool = False            # trTsEnable


class Encoder:
    """The N-Trace encoder.  ``out`` gets each :class:`Message` (``raw``:
    its bytes); ``now()`` gives the timestamp (trTs)."""

    def __init__(self, xlen: int, mode: Mode, out: Callable[[Message], None],
                 now: Callable[[], int] | None = None, has_h: bool = False) -> None:
        self.xlen = xlen
        self.amask = (1 << xlen) - 1
        self.mode = mode
        self.out = out
        self.now = now or (lambda: 0)
        self.has_h = has_h
        self.tracing = False
        self.icnt = 0
        self.hist = 1
        self.last_addr = 0
        self.pending: tuple[int, int, str] | None = None   # (tcode, btype, reason)
        self.sync_due: int | None = None        # (a SYNC code to send at the next one)
        self.stack: list[int] = []
        self.calls = 0
        self.prev: I.Block | None = None
        self.last_ts = 0
        self.sync_count = 0
        self.repeat: tuple | None = None        # (the last branch message's key)
        self.bcnt = 0
        self.hrepeat: int | None = None         # (the last full HIST record)
        self.hcount = 0
        self.lost = False
        self.messages = 0
        self.ownership_due = False
        self.context = (0, 0)
        self.synced_at: int | None = None

    # -- emitting -----------------------------------------------------------------

    def _emit(self, tcode: int, fields: list[tuple[str, int, int]], reason: str,
              sync: bool = False, flush: bool = True) -> None:
        if flush and (self.bcnt or self.hcount):
            self._flush_repeat()                # (what was counted comes first)
        m = self.mode
        pre: list[tuple[str, int, int]] = []
        if m.src_bits:
            pre.append(("SRC", m.src_id, m.src_bits))
        fl = pre + fields
        ts = None
        if m.timestamps:
            t = self.now()
            ts = t if sync or tcode in SYNC_CODES else t - self.last_ts
            self.last_ts = t
            fl = fl + [("TSTAMP", ts, 0)]
        raw = pack(tcode, fl)
        msg = Message(tcode, {k: v for k, v, _ in fields}, m.src_id if m.src_bits else None,
                      ts, raw, reason)
        self.messages += 1
        if m.sync_mode == 1:
            self.sync_count += 1
        self.out(msg)

    def _addr(self, name: str, a: int) -> tuple[str, int, int]:
        """An F-ADDR / U-ADDR field (7.1: without bit 0).  With
        trTeInstExtendAddrMSB (7.6) identical most significant ones are not
        sent — the field's top bit (a whole number of MDOs) is then 1 and
        extended by the decoder; a value whose top MDO bit is 1 otherwise
        gets one more MDO of zeros."""
        n = self.xlen - 1
        v = (a & self.amask) >> 1
        if not self.mode.extend_msb:
            return name, v, 0
        if v >> (n - 1) & 1:
            k = n
            while k > 1 and v >> (k - 2) & 1:
                k -= 1                          # (k bits: the run's last 1 on top)
            bits = -(-k // 6) * 6
            return name, v & ((1 << bits) - 1), -bits
        bits = -(-_var_bits(v) // 6) * 6
        if v >> (bits - 1) & 1:
            bits += 6
        return name, v, -bits

    def _flush_repeat(self) -> None:
        if self.bcnt:
            n, self.bcnt = self.bcnt, 0
            self._emit(REPEAT, [("BCNT", n, 0)], "repeated branch", flush=False)
        if self.hcount:
            n, self.hcount = self.hcount, 0
            if n > 1:
                self._emit(RESOURCE_FULL, [("RCODE", 2, 4), ("RDATA", self.hrepeat, 0),
                                           ("HREPEAT", n, 0)], "repeated history",
                           flush=False)
            else:
                self._emit(RESOURCE_FULL, [("RCODE", 1, 4), ("RDATA", self.hrepeat, 0)],
                           "HIST full", flush=False)
        self.hrepeat = None

    def _branch_msg(self, tcode: int, fields: list, key: tuple, reason: str) -> None:
        """A branch message, or (repeated history) a count of it (6.11)."""
        if self.mode.repeated_history and self.repeat == key and self.bcnt < MAX_REPEAT \
                and not self.hcount:
            self.bcnt += 1
            return
        self._emit(tcode, fields, reason)
        self.repeat = key

    def _sync(self, code: int, addr: int, reason: str) -> None:
        """A synchronising message at ``addr`` (I-CNT before it): it resets the
        state (7.5)."""
        self._flush_repeat()
        if self.mode.htm and self.hist != 1:
            self._emit(INDIRECT_HIST_SYNC, [("SYNC", code, 4), ("BTYPE", BT_INDIRECT, 2),
                                            ("ICNT", self.icnt, 0),
                                            self._addr("FADDR", addr),
                                            ("HIST", self.hist, 0)], reason, True)
        else:
            self._emit(SYNC_MSG, [("SYNC", code, 4), ("ICNT", self.icnt, 0),
                                  self._addr("FADDR", addr)], reason, True)
        self.synced_at = addr
        self.icnt, self.hist = 0, 1
        self.last_addr = addr
        self.stack, self.calls = [], 0
        self.repeat = None
        self.sync_count = 0
        if self.mode.context:
            self.ownership_due = True

    def _ownership(self, b: I.Block, kind: int) -> None:
        """6.1: PROCESS = CONTEXT << 5 | V << 4 | PRV << 2 | FORMAT (format 0:
        V / PRV changed, 2: scontext, 3: hcontext — sent before scontext)."""
        v = 1 if b.priv in (I.PRIV_VS, I.PRIV_VU) else 0
        prv = {I.PRIV_U: 0, I.PRIV_S: 1, I.PRIV_M: 3, I.PRIV_VS: 1, I.PRIV_VU: 0}.get(b.priv, 3)
        base = v << 4 | prv << 2
        if kind == 0:
            self._emit(OWNERSHIP, [("PROCESS", base, 0)], "privilege")
            return
        if self.has_h:
            self._emit(OWNERSHIP, [("PROCESS", b.hcontext << 5 | base | 3, 0)], "hcontext")
        self._emit(OWNERSHIP, [("PROCESS", b.context << 5 | base | 2, 0)], "scontext")

    # -- the ingress side -----------------------------------------------------------

    def start(self, code: int = SYNC_ENABLE) -> None:
        """Tracing starts: the next traced block gets a ProgTraceSync."""
        self.tracing = True
        self.sync_due = code
        self.icnt, self.hist = 0, 1
        self.pending = None
        self.prev = None
        if self.lost:
            self._emit(ERROR, [("ETYPE", 0, 4), ("ECODE", 0b100, 0)], "messages lost")
            self.lost = False
            self.sync_due = SYNC_OVERRUN

    def push(self, b: I.Block, notify: int | None = None) -> None:
        """One traced block (``notify``: a SYNC code for a notification on it —
        6 trace event, 0 external trigger)."""
        m = self.mode
        if not self.tracing:
            return
        addr = b.iaddr
        self.synced_at = None
        if self.sync_due is not None:
            code, self.sync_due = self.sync_due, None
            self.pending = None
            self._sync(code, addr, SYNC_NAMES.get(code, "sync"))
        elif self.pending is not None:
            tcode, btype, reason = self.pending
            self.pending = None
            self._indirect(tcode, btype, addr, reason)
        if self.ownership_due and b.iretire:
            self.ownership_due = False
            self._ownership(b, 1)
            self.context = (b.context, b.hcontext)
        elif m.context and self.prev is not None and b.iretire:
            if b.priv != self.prev.priv:
                self._ownership(b, 0)
            if (b.context, b.hcontext) != self.context:
                self._ownership(b, 1)
                self.context = (b.context, b.hcontext)
        if b.iretire and self._periodic_due(b):
            self._sync(SYNC_PERIODIC, addr, "periodic")
        if notify is not None and b.iretire:
            self._sync(notify, addr, "notify")
        self.prev = b
        if not b.iretire:                       # (a trap: 3's Table 2, itype 1 / 2)
            self.pending = (INDIRECT_HIST if m.htm else INDIRECT,
                            BT_INTERRUPT if b.itype == I.INTERRUPT else BT_EXCEPTION,
                            "interrupt" if b.itype == I.INTERRUPT else "exception")
            return
        prev_icnt = self.icnt
        self.icnt += b.halfwords
        it = b.itype
        if m.all_jumps and it in (I.INFERABLE_CALL, I.INFERABLE_JUMP, I.OTHER_INFERABLE):
            branch_like = True
        else:
            branch_like = it in (I.NOT_TAKEN, I.TAKEN)
        taken = it != I.NOT_TAKEN
        if branch_like:
            if m.htm:
                self.hist = self.hist << 1 | int(taken)
                if self.hist >> HIST_BITS & 1:
                    self._hist_full()
            elif taken:
                self._branch_msg(DIRECT, [("ICNT", self.icnt, 0)], ("D", self.icnt),
                                 "taken branch")
                self.icnt = 0
        elif self._uninferable(b):
            self.pending = (INDIRECT_HIST if m.htm else INDIRECT, BT_INDIRECT,
                            I.ITYPE_NAMES.get(it, "jump"))
        self._calls(b)
        if self.icnt >= ICNT_LIMIT:             # (6.8 RCODE 0: the count before)
            self._emit(RESOURCE_FULL, [("RCODE", 0, 4), ("RDATA", prev_icnt, 0)],
                       "I-CNT full")
            self.icnt -= prev_icnt

    def _hist_full(self) -> None:
        """HIST holds 31 branches: ResourceFull RCODE 1 (or, with repeated
        history, counted while it repeats: RCODE 2)."""
        h = self.hist
        self.hist = 1
        if self.mode.repeated_history:
            if self.bcnt:
                n, self.bcnt = self.bcnt, 0
                self._emit(REPEAT, [("BCNT", n, 0)], "repeated branch", flush=False)
            if self.hrepeat == h and self.hcount < MAX_REPEAT:
                self.hcount += 1
                return
            self._flush_repeat()
            self.hrepeat, self.hcount = h, 1
            return
        self._emit(RESOURCE_FULL, [("RCODE", 1, 4), ("RDATA", h, 0)], "HIST full")

    def _uninferable(self, b: I.Block) -> bool:
        it = b.itype
        m = self.mode
        if it == I.TRAP_RETURN:
            return True
        if it in (I.RETURN, I.COROUTINE_SWAP) and m.implicit_return:
            if self._predicted(b):
                return False
            return True
        if it in I.UNINFERABLE or it == I.RETURN:
            # (8.1: the LUI / AUIPC must be in the same stretch the decoder walks
            # — not before a synchronising message at the jump)
            if m.sijump and b.sijump and self.synced_at != b.iaddr:
                return False
            return True
        return False

    def _predicted(self, b: I.Block) -> bool:
        """8.2: the return goes where the stack says (mode 1: any non-empty
        count; 2: the low 16 bits match; 3: the whole address)."""
        mode = self.mode.implicit_return
        if mode == 1:
            return self.calls > 0
        if not self.stack:
            return False
        top = self.stack[-1]
        if mode == 2:
            return (top ^ b.npc) & 0xFFFF == 0
        return top == b.npc

    def _calls(self, b: I.Block) -> None:
        mode = self.mode.implicit_return
        if not mode:
            return
        it = b.itype
        if it in (I.RETURN, I.COROUTINE_SWAP):
            if mode == 1:
                if self.calls:
                    self.calls -= 1
            elif self.stack:
                self.stack.pop()
        if it in I.CALLS or it == I.COROUTINE_SWAP:
            if mode == 1:
                self.calls = min(self.calls + 1, CALL_STACK)
            else:
                if len(self.stack) == CALL_STACK:
                    self.stack.pop(0)
                self.stack.append(b.link)

    def _indirect(self, tcode: int, btype: int, addr: int, reason: str) -> None:
        m = self.mode
        ua = self._addr("UADDR", addr ^ self.last_addr)
        u = ua[1]
        if tcode == INDIRECT_HIST and self.hist == 1:
            tcode = INDIRECT                    # (9's rule 4: an empty HIST)
        fields = [("BTYPE", btype, 2), ("ICNT", self.icnt, 0), ua]
        if tcode == INDIRECT_HIST:
            fields.append(("HIST", self.hist, 0))
        key = (tcode, btype, self.icnt, u, self.hist)
        if btype != BT_INDIRECT:
            self._emit(tcode, fields, reason)
            self.repeat = None
        else:
            self._branch_msg(tcode, fields, key, reason)
        self.icnt, self.hist = 0, 1
        self.last_addr = addr
        del m

    def _periodic_due(self, b: I.Block) -> bool:
        m = self.mode
        if m.sync_mode == 0:
            return False
        if m.sync_mode == 2:
            if self.prev is not None:
                self.sync_count += max(0, b.cycle - self.prev.cycle)
        elif m.sync_mode == 3:
            self.sync_count += b.halfwords
        return self.sync_count >= 1 << (m.sync_max + 4)

    def stop(self, evcode: int = EV_DISABLED, npc: int | None = None) -> None:
        """Tracing stops: a pending indirect message (its target ``npc``),
        then ProgTraceCorrelation (EVCODE, I-CNT, HIST in HTM)."""
        if not self.tracing:
            return
        if self.pending is not None:
            tcode, btype, reason = self.pending
            self.pending = None
            if npc is not None:
                self._indirect(tcode, btype, npc, reason)
        if self.sync_due is not None:           # (nothing traced since the start)
            self.sync_due = None
            self.tracing = False
            return
        fields = [("EVCODE", evcode, 4), ("CDF", 1 if self.mode.htm else 0, 2),
                  ("ICNT", self.icnt, 0)]
        if self.mode.htm:
            fields.append(("HIST", self.hist, 0))
        self._emit(CORRELATION, fields, "Debug Mode" if evcode == EV_DEBUG else
                   "trace stopped")
        self.icnt, self.hist = 0, 1
        self.tracing = False

    def overflow(self) -> None:
        """Messages could not be sent: an Error message and a sync with SYNC 7
        once the trace goes on."""
        self.lost = True
        self.tracing = False


# -- the decoder ----------------------------------------------------------------------


class DecodeError(Exception):
    pass


@dataclass
class Event:
    kind: str
    addr: int = 0
    info: str = ""


class Decoder:
    """Ch. 10: rebuild the flow from the messages (``instr(addr)``: the
    instruction there, ``word``: memory for CM.JT's table)."""

    def __init__(self, xlen: int, instr: Callable[[int], Any], *, mode: Mode | None = None,
                 word: Callable[[int, int], int] | None = None, jvt: int = 0) -> None:
        self.xlen = xlen
        self.amask = (1 << xlen) - 1
        self.instr = instr
        self.word = word
        self.jvt = jvt
        self.mode = mode or Mode()
        self.pc: int | None = None
        self.last_addr = 0
        self.adjust = 0
        self.events: list[Event] = []
        self.stack: list[int] = []
        self.last_branch: Message | None = None
        self.last_hist_full: int | None = None
        self.prev_ts = 0

    def pcs(self) -> list[int]:
        return [e.addr for e in self.events if e.kind == "pc"]

    def feed(self, data: bytes) -> list[Message]:
        msgs = []
        for raw in _Reader(data).messages():
            m = unpack(raw, src_bits=self.mode.src_bits, tstamp=self.mode.timestamps)
            self.message(m)
            msgs.append(m)
        return msgs

    def _address(self, msg: Message, name: str, full: bool) -> int:
        """7.1 / 7.6: the field's value without bit 0, extended (with the
        MSB extension) when its last MDO's top bit is 1."""
        v = msg.get(name)
        a = v << 1
        if self.mode.extend_msb:
            n = msg.widths.get(name, 0)
            if n and v >> (n - 1) & 1:
                a |= ~((1 << (n + 1)) - 1)
        a &= self.amask
        return a if full else a ^ self.last_addr

    def _get(self, a: int):
        i = self.instr(a)
        if i is None:
            raise DecodeError(f"no instruction at 0x{a:x}")
        return i

    def _walk(self, icnt: int | None, hist: int, last_taken: bool = False) -> None:
        """Walk ``icnt`` halfwords (None: until HIST is used up) from pc: HIST's
        bits (after its stop bit) decide the branches; a BTM DirectBranch's
        last instruction is a taken branch (``last_taken``)."""
        if self.pc is None:
            return
        if icnt is not None:
            icnt += self.adjust
            self.adjust = 0
            if icnt < 0:
                raise DecodeError("I-CNT below what ResourceFull messages walked")
        bits = []
        if hist > 1:
            n = hist.bit_length() - 1
            bits = [hist >> (n - 1 - k) & 1 for k in range(n)]
        done = 0
        m = self.mode
        guard = 1 << 24
        while True:
            if icnt is not None and done >= icnt:
                break
            if icnt is None and not bits:
                break
            guard -= 1
            if guard < 0:
                raise DecodeError("the walk does not end")
            pc = self.pc
            i = self._get(pc)
            it = I.classify(i)
            self.events.append(Event("pc", pc))
            done += i.size // 2
            last = icnt is not None and done >= icnt
            name = i.base.name
            nxt = (pc + i.size) & self.amask
            branch_like = name in I.BRANCHES or (m.all_jumps and it in (
                I.INFERABLE_CALL, I.INFERABLE_JUMP, I.OTHER_INFERABLE))
            if branch_like:
                taken = bool(bits.pop(0)) if bits else (last and last_taken)
                if taken:
                    nxt = self._target(i, pc)
            elif it in (I.INFERABLE_CALL, I.INFERABLE_JUMP, I.OTHER_INFERABLE):
                nxt = self._target(i, pc)
            elif self._sequential(i, it, pc):
                nxt = self._seq_target
            elif it in (I.RETURN, I.COROUTINE_SWAP) and m.implicit_return and not last \
                    and self.stack:
                nxt = self.stack.pop()
            elif it in I.UNINFERABLE or it in (I.RETURN, I.TRAP_RETURN):
                if not last:
                    raise DecodeError(f"an uninferable jump at 0x{pc:x} inside I-CNT")
                if m.implicit_return and it in (I.RETURN, I.COROUTINE_SWAP) and self.stack:
                    self.stack.pop()
                nxt = None
            if m.implicit_return and (it in I.CALLS or it == I.COROUTINE_SWAP):
                if len(self.stack) == CALL_STACK:
                    self.stack.pop(0)
                self.stack.append((pc + i.size) & self.amask)
            self._prev_insn = (pc, i)
            self.pc = nxt
            if nxt is None:
                break
        if icnt is not None and done != icnt:
            raise DecodeError(f"I-CNT {icnt} ends inside an instruction")

    def _target(self, i, pc: int) -> int:
        if i.name == "cm.jalt":
            base = (self.jvt & ~0x3F) + i.f["index"] * (self.xlen // 8)
            return self.word(base & self.amask, self.xlen // 8) & ~1 & self.amask
        if i.base.name == "jalr":
            return i.f.get("imm", 0) & ~1 & self.amask
        return (pc + i.f["imm"]) & self.amask

    _prev_insn: tuple | None = None
    _seq_target = 0

    def _sequential(self, i, it: int, pc: int) -> bool:
        if not (self.mode.sijump and it in I.UNINFERABLE) or self._prev_insn is None:
            return False
        ppc, prev = self._prev_insn
        if prev.base.name not in ("lui", "auipc") or (ppc + prev.size) & self.amask != pc:
            return False
        rd = prev.f.get("rd", 0)
        if rd == 0 or rd != i.f.get("rs1"):
            return False
        from .etrace import sx
        t = sx(prev.f["imm"], 32) + (ppc if prev.base.name == "auipc" else 0)
        self._seq_target = (t + i.f.get("imm", 0)) & ~1 & self.amask
        return True

    def message(self, msg: Message) -> None:
        t = msg.tcode
        f = msg.fields
        if t == OWNERSHIP:
            self.events.append(Event("ownership", 0, f"PROCESS 0x{f.get('PROCESS', 0):x}"))
            return
        if t == ERROR:
            self.events.append(Event("lost"))
            self.pc = None
            return
        if t in (SYNC_MSG, DIRECT_SYNC, INDIRECT_SYNC, INDIRECT_HIST_SYNC):
            if self.pc is not None:
                self._walk(f.get("ICNT", 0), f.get("HIST", 1),
                           last_taken=t == DIRECT_SYNC)
            else:
                self.adjust = 0
            a = self._address(msg, "FADDR", True)
            self.pc = a
            self.last_addr = a
            self.stack = []
            self._prev_insn = None
            self.last_branch = None
            self.events.append(Event("sync", a, SYNC_NAMES.get(f.get("SYNC", 0), "")))
            return
        if self.pc is None:
            return                              # (not synchronised yet)
        if t == RESOURCE_FULL:
            rc = f.get("RCODE", 0)
            if rc == 0:                         # (6.8: added to the next I-CNT)
                self._icnt_full += f.get("RDATA", 0)
                return
            for _ in range(f.get("HREPEAT", 1) if rc == 2 else 1):
                before = self._count_events()
                self._walk(None, f.get("RDATA", 1))
                self.adjust -= self._halfwords_since(before)
            return
        if t == REPEAT:
            last = self.last_branch
            if last is None:
                raise DecodeError("RepeatBranch without a branch message before it")
            for _ in range(f.get("BCNT", 0)):
                self._branch(last)
            return
        if t == CORRELATION:
            self._walk(f.get("ICNT", 0) + self._take_full(), f.get("HIST", 1) if
                       f.get("CDF") == 1 else 1)
            self.events.append(Event("end", self.pc or 0,
                                     "Debug Mode" if f.get("EVCODE") == EV_DEBUG else
                                     "trace stopped"))
            self.pc = None
            return
        if t in (DIRECT, INDIRECT, INDIRECT_HIST):
            self._branch(msg)
            self.last_branch = msg
            return

    _icnt_full = 0

    def _take_full(self) -> int:
        n, self._icnt_full = self._icnt_full, 0
        return n

    def _count_events(self) -> int:
        return len(self.events)

    def _halfwords_since(self, k: int) -> int:
        n = 0
        for e in self.events[k:]:
            if e.kind == "pc":
                i = self.instr(e.addr)
                n += i.size // 2 if i is not None else 1
        return n

    def _branch(self, msg: Message) -> None:
        f = msg.fields
        icnt = f.get("ICNT", 0) + self._take_full()
        if msg.tcode == DIRECT:
            self._walk(icnt, 1, last_taken=True)
            return
        self._walk(icnt, f.get("HIST", 1))
        a = self._address(msg, "UADDR", False)
        bt = f.get("BTYPE", 0)
        if bt in (BT_TRAP, BT_EXCEPTION, BT_INTERRUPT):
            self.events.append(Event("trap", a, ("trap", "trap", "exception",
                                                 "interrupt")[bt]))
        self.pc = a
        self.last_addr = a
