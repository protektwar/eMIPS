"""Efficient Trace for RISC-V (E-Trace) 2.0 (LF11): the instruction trace
encoder of ch. 7 / 9, its packets (ch. 7: te_inst formats 0-3 with
sign-based compression) and a decoder (ch. 11's pseudocode, extended to
every optional mode the encoder has).  The specification's repository
keeps the Siemens reference flow (``referenceFlow/scripts``: encoder_model.py / decoder_model.py —
the oracle of ``tests/test_etrace.py``, which implement the baseline
only).

The encoder (:class:`Encoder`) reads the hart's ingress blocks
(:mod:`emips.rvingress`) through the 3-stage pipeline of ch. 9 (previous,
current, next) and follows Figure 9.1 step by step, with the names of the
reference model (``prev_updiscon``, ``er_n``, ``ppccd_br``, ``rpt_br`` ...)
for the baseline.  The optional modes (8.2, the *ioptions* of the support
packet, :data:`OPTIONS`):

* **full address** (8.2.2): every address absolute;
* **implicit exception** (8.2.3): no trap handler address in a format 3
  subformat 1 packet with *thaddr* 1 (the decoder takes the vector from
  the cause: :class:`Decoder`'s ``trap_vector``);
* **sequentially inferable jumps** (8.2.4): a jump the hart marks with
  *sijump* is inferable when the LUI / AUIPC before it was traced in the
  same packet's range (not when the jump itself starts a format 3);
* **implicit return** (8.2.5): a stack of 2^return_stack_size_p return
  addresses (or, with call_counter_size_p, a nested call counter): a
  return whose target is the stack's top is not reported; a mismatch is,
  with *irreport* / *irdepth*; 7.6.2's two cases report the depth on the
  packet of an instruction before a trap, privilege change or resync; a
  co-routine swap pops and pushes;
* **branch prediction** (8.2.6): 2^bpred_size_p 2-bit counters indexed by
  the branch address; 31 or more correctly predicted branches in a row are
  reported as a count (format 0 subformat 0, 7.8);
* **jump target cache** (8.2.7): 2^cache_size_p direct-mapped targets of
  uninferable jumps; a hit is reported as its index (format 0 subformat
  1) unless format 1 / 2 is shorter.

Resynchronisation (9.2) counts packets, hart cycles or retired halfwords
(trTeInstSyncMode 1-3) up to 2^(trTeInstSyncMax + 4).  A support packet
(format 3 subformat 3) starts the trace and follows its end (qualification
lost: *ended_rep* / *ended_ntr*, 7.5.1) or a loss (*trace_lost*).

Packets are lists of fields (:class:`Packet`); :func:`encode_payload`
packs them LSB first and removes the identical most significant bits
(ch. 7: "sign based compression", at least one kept), sign-extending to a
whole byte; :func:`decode_payload` reads them back from the payload's
bytes (the transport — the encapsulation, emips.rvtrace — gives the
length).  Choices: ``sim.DEVIATIONS["RISC-V trace"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import rvingress as I

# te_inst formats / subformats (ch. 7)
F_EXT, F_BRANCH, F_ADDR, F_SYNC = 0, 1, 2, 3
SF_START, SF_TRAP, SF_CONTEXT, SF_SUPPORT = 0, 1, 2, 3
F0_COUNT, F0_CACHE = 0, 1
#: qual_status (Table 7.4)
NO_CHANGE, ENDED_REP, TRACE_LOST, ENDED_NTR = 0, 1, 2, 3
QUAL_NAMES = {0: "no change", 1: "ended (reported)", 2: "trace lost",
              3: "ended (not reported)"}
MAX_BRANCHES = 31
#: the ioptions bits (7.5: implementation-defined order — the reference
#: decoder's, then sijump): name -> bit
OPTIONS = ("implicit_return", "implicit_exception", "full_address",
           "jump_target_cache", "branch_prediction", "sijump")


@dataclass(frozen=True)
class Params:
    """The encoder's parameters (Table 10.1) as the decoder discovers them
    (10.1: Table 10.3).  eMIPS's encoder: XLEN-wide addresses with
    compressed instructions (iaddress_lsb_p 1), 6-bit causes (the custom
    causes reach 63), 3-bit privilege (VS / VU), scontext as context, no
    time field (timestamps go in the encapsulation), an 8-entry return
    stack, a 64-entry branch predictor, a 16-entry jump target cache,
    sijump."""
    iaddress_width_p: int = 64
    iaddress_lsb_p: int = 1
    ecause_width_p: int = 6
    privilege_width_p: int = 3
    context_width_p: int = 32
    nocontext_p: int = 0
    time_width_p: int = 0
    notime_p: int = 1
    call_counter_size_p: int = 0
    return_stack_size_p: int = 3
    bpred_size_p: int = 6
    cache_size_p: int = 4
    sijump_p: int = 1
    #: format 0's subformat field (7.8.1: 1 bit for the two optional formats)
    f0s_width_p: int = 1
    #: the support packet's *encoder_mode* width (E) and data-trace fields
    #: (*denable* / *dloss* / *doptions*: only when data trace exists —
    #: eMIPS has none; the reference model sends 1 + 1 + 4 bits anyway)
    encoder_mode_width: int = 1
    data_fields: bool = False
    doptions_width: int = 4

    @property
    def address_bits(self) -> int:
        return self.iaddress_width_p - self.iaddress_lsb_p

    @property
    def irdepth_bits(self) -> int:
        rss = self.return_stack_size_p
        return rss + (1 if rss else 0) + self.call_counter_size_p

    def supports(self, option: str) -> bool:
        return {"implicit_return": bool(self.return_stack_size_p or
                                        self.call_counter_size_p),
                "branch_prediction": self.bpred_size_p > 0,
                "jump_target_cache": self.cache_size_p > 0,
                "sijump": bool(self.sijump_p)}.get(option, True)

    @property
    def options(self) -> tuple[str, ...]:
        """The ioptions bits this encoder sends (one a supported mode; the
        five of the reference decoder always, sijump when supported)."""
        return OPTIONS[:5] + (("sijump",) if self.sijump_p else ())


def params_for(xlen: int) -> Params:
    return Params(iaddress_width_p=xlen)


@dataclass
class Packet:
    """One te_inst: its fields in transmission order (name, value, bits)."""
    fields: list[tuple[str, int, int]] = field(default_factory=list)
    #: why the encoder sent it (the flowchart's decision: views)
    reason: str = ""

    def add(self, name: str, value: int, bits: int) -> None:
        if bits:
            self.fields.append((name, value & ((1 << bits) - 1), bits))

    def get(self, name: str, default: int | None = None) -> int | None:
        for n, v, _ in self.fields:
            if n == name:
                return v
        return default

    @property
    def format(self) -> int:
        return self.fields[0][1]

    @property
    def subformat(self) -> int | None:
        return self.get("subformat")

    def bits(self) -> tuple[int, int]:
        """(value, length) LSB first."""
        v = n = 0
        for _, value, bits in self.fields:
            v |= value << n
            n += bits
        return v, n

    def name(self) -> str:
        f, sf = self.format, self.subformat
        if f == F_SYNC:
            return ("sync", "trap", "context", "support")[sf]
        if f == F_EXT:
            return ("branch count", "jump cache")[sf or 0]
        return "branch" if f == F_BRANCH else "address"


def compress(value: int, n: int) -> bytes:
    """Ch. 7's sign-based compression: drop the most significant bits equal
    to the MSB (one stays), sign-extend to a whole byte; LSB byte first."""
    msb = value >> (n - 1) & 1
    keep = n
    while keep > 1 and (value >> (keep - 2) & 1) == msb:
        keep -= 1
    nbytes = (keep + 7) // 8
    value &= (1 << keep) - 1
    if msb:
        value |= ((1 << (8 * nbytes)) - 1) & ~((1 << keep) - 1)
    return value.to_bytes(nbytes, "little")


def encode_payload(p: Packet) -> bytes:
    return compress(*p.bits())


class _Bits:
    """Read fields LSB first from a compressed payload (sign-extended past
    its end)."""

    def __init__(self, data: bytes) -> None:
        self.v = int.from_bytes(data, "little")
        self.n = 8 * len(data)
        self.sign = self.v >> (self.n - 1) & 1 if self.n else 0
        self.pos = 0

    def take(self, bits: int) -> int:
        if not bits:
            return 0
        v = self.v >> self.pos
        if self.pos + bits > self.n and self.sign:
            v |= ~((1 << max(0, self.n - self.pos)) - 1)
        self.pos += bits
        return v & ((1 << bits) - 1)

    def peek_msb(self, value: int, bits: int) -> int:
        return value >> (bits - 1) & 1


def _map_bits(branches: int) -> tuple[int, bool]:
    """Format 1's branch_map length and whether an address follows (7.7)."""
    if branches == 0:
        return 31, False
    for n in (1, 3, 7, 15):
        if branches <= n:
            return n, True
    return 31, True


def decode_payload(data: bytes, p: Params, options: dict[str, bool]) -> Packet:
    """The te_inst in ``data`` (a payload as :func:`encode_payload` made it);
    ``options`` = the last support packet's (they decide format 0's
    subformat when f0s_width_p is 0)."""
    b = _Bits(data)
    out = Packet()

    def get(name: str, bits: int) -> int:
        v = b.take(bits)
        if bits:
            out.fields.append((name, v, bits))
        return v

    fmt = get("format", 2)
    ab = p.address_bits
    irb = p.irdepth_bits

    def status_fields(prev_bit: int) -> None:
        get("notify", 1)
        get("updiscon", 1)
        get("irreport", 1)
        get("irdepth", irb)

    if fmt == F_SYNC:
        sf = get("subformat", 2)
        if sf in (SF_START, SF_TRAP, SF_CONTEXT):
            if sf != SF_CONTEXT:
                get("branch", 1)
            get("privilege", p.privilege_width_p)
            get("time", 0 if p.notime_p else p.time_width_p)
            get("context", 0 if p.nocontext_p else p.context_width_p)
            if sf == SF_START:
                get("address", ab)
            elif sf == SF_TRAP:
                get("ecause", p.ecause_width_p)
                intr = get("interrupt", 1)
                thaddr = get("thaddr", 1)
                if not (thaddr and options.get("implicit_exception")):
                    get("address", ab)
                if not intr:
                    get("tval", p.iaddress_width_p)
        else:
            get("ienable", 1)
            get("encoder_mode", p.encoder_mode_width)
            get("qual_status", 2)
            get("ioptions", len(p.options))
            if p.data_fields:
                get("denable", 1)
                get("dloss", 1)
                get("doptions", p.doptions_width)
    elif fmt == F_ADDR:
        get("address", ab)
        status_fields(0)
    elif fmt == F_BRANCH:
        br = get("branches", 5)
        bits, has_addr = _map_bits(br)
        get("branch_map", bits)
        if has_addr:
            get("address", ab)
            status_fields(0)
    else:
        if p.f0s_width_p:
            sf = get("subformat", p.f0s_width_p)
        else:
            sf = F0_CACHE if options.get("jump_target_cache") else F0_COUNT
            out.fields.append(("subformat", sf, 0))
        if sf == F0_COUNT:
            get("branch_count", 32)
            bf = get("branch_fmt", 2)
            if bf:
                get("address", ab)
                status_fields(0)
        else:
            get("index", p.cache_size_p)
            br = get("branches", 5)
            if br:
                get("branch_map", _map_bits(br)[0])
            get("irreport", 1)
            get("irdepth", irb)
    return out


# -- the encoder ----------------------------------------------------------------------


@dataclass(slots=True)
class _Slot:
    """A pipeline stage: an ingress block and what the encoder decided
    about it."""
    b: I.Block | None               # None: an unqualified cycle
    qualified: bool = False
    notify: bool = False
    #: an uninferable PC discontinuity (decided when the next block is known)
    updiscon: bool = False
    #: a return the implicit-return stack mispredicted: the depth it had
    ir_fail: int | None = None
    #: reported by a format 3 packet (sijump needs the LUI before it seen)
    synced: bool = False
    #: a branch the predictor got wrong
    bp_fail: bool = False
    #: the predictor counted it (part of a correctly predicted run)
    bp_ok: bool = False

    @property
    def itype(self) -> int:
        return self.b.itype if self.b is not None else 0

    @property
    def exc_only(self) -> bool:
        b = self.b
        return b is not None and b.iretire == 0 and b.itype in (I.EXCEPTION, I.INTERRUPT)

    @property
    def exception(self) -> bool:
        return self.b is not None and self.b.itype in (I.EXCEPTION, I.INTERRUPT)

    @property
    def is_branch(self) -> bool:
        return self.b is not None and self.b.itype in (I.NOT_TAKEN, I.TAKEN)


_UNQ = _Slot(None)


@dataclass
class Mode:
    """The run-time options (trTeInstFeatures, trTeControl's sync fields)."""
    full_address: bool = False
    implicit_exception: bool = False
    sijump: bool = False
    implicit_return: bool = False
    branch_prediction: bool = False
    jump_target_cache: bool = False
    #: trTeInstSyncMode: 0 off, 1 packets, 2 hart cycles, 3 halfwords
    sync_mode: int = 1
    #: trTeInstSyncMax: resync after 2^(sync_max + 4)
    sync_max: int = 1

    def ioptions(self, p: Params) -> int:
        return sum(1 << k for k, name in enumerate(p.options) if getattr(self, name))


class Encoder:
    """The E-Trace instruction trace encoder (ch. 9's algorithm).  ``out``
    gets each packet (:class:`Packet`) as it is made."""

    def __init__(self, params: Params, mode: Mode, out: Callable[[Packet], None]) -> None:
        self.p = params
        self.mode = mode
        self.out = out
        self.prev = self.curr = self.next = _UNQ
        self.branches = 0
        self.branch_map = 0                     # bit k: the k-th oldest, 0 = taken
        self.resync_count = 0
        self.resync_state = 1                   # (9.2's FSM, modes 2 / 3)
        self.trap_reported = False
        self.last_address: int | None = None
        self.amask = (1 << params.iaddress_width_p) - 1
        # optional modes
        self.stack: list[int] = []              # implicit return: links (or a count)
        self.calls = 0
        self.ret_since_call = False
        self.branch_since_ret = False
        self.pbc = 0                            # correctly predicted branches in a row
        self.bp_table = [1] * (1 << params.bpred_size_p) if params.bpred_size_p else []
        self.cache: list[int | None] = [None] * (1 << params.cache_size_p) \
            if params.cache_size_p else []
        self.cci_pending = False
        self.last_ctype_context = 0
        self.enabled = False
        self.packets = 0
        self.force_sync = False
        self.flags: dict[str, bool] = {}

    # -- the ingress side ---------------------------------------------------------

    def start(self) -> None:
        """The encoder is enabled: the support packet (7.5)."""
        self.enabled = True
        self._support(NO_CHANGE)

    def push(self, b: I.Block | None, qualified: bool = True, notify: bool = False) -> None:
        """One ingress cycle (None / ``qualified`` False: an instruction not
        traced)."""
        s = _Slot(b, qualified and b is not None, notify)
        self.prev, self.curr, self.next = self.curr, self.next, s
        if self.curr.qualified:
            self._encode()

    def stop(self) -> None:
        """Tracing stops (trTeInstTracing 0, a trace-off trigger, halted,
        reset, the filter): the last traced instruction is reported, then the
        support packet (7.5) — as if an unqualified instruction followed."""
        if self.next.qualified:
            self.push(None, False)

    def lost(self) -> None:
        """Packets were lost (the sink did not take them): a support packet
        with *trace_lost*; the next traced instruction starts with a sync."""
        self._support(TRACE_LOST)
        self.force_sync = True

    def disable(self) -> None:
        """trTeEnable 1 -> 0: the end of the trace and a support packet with
        *ienable* 0."""
        self.enabled = False
        if self.next.qualified:
            self.push(None, False)
        else:
            self._support(NO_CHANGE)

    # -- packets ------------------------------------------------------------------

    def _send(self, pkt: Packet, clears: bool = True) -> None:
        """Send a packet; ``clears``: it reports the branch map (formats 0-2)
        or resynchronises (format 3 subformats 0 / 1) — a context or support
        packet leaves the unreported branches alone."""
        self.packets += 1
        if self.mode.sync_mode == 1:
            self.resync_count += 1
        if clears:
            self.branches = 0
            self.branch_map = 0
        self.out(pkt)

    def _support(self, qual: int) -> None:
        p = self.p
        pkt = Packet(reason="support")
        pkt.add("format", F_SYNC, 2)
        pkt.add("subformat", SF_SUPPORT, 2)
        pkt.add("ienable", int(self.enabled), 1)
        pkt.add("encoder_mode", 0, p.encoder_mode_width)
        pkt.add("qual_status", qual, 2)
        pkt.add("ioptions", self.mode.ioptions(p), len(p.options))
        if p.data_fields:
            pkt.add("denable", 0, 1)
            pkt.add("dloss", 0, 1)
            pkt.add("doptions", 0, p.doptions_width)
        self._send(pkt, clears=False)

    def _sync_reset(self) -> None:
        """A format 3 packet resynchronises: resync counter, implicit-return
        stack, predictor (all entries 01), jump target cache (8.2)."""
        self.resync_count = 0
        self.resync_state = 1
        self.stack = []
        self.calls = 0
        self.ret_since_call = self.branch_since_ret = False
        self.pbc = 0
        if self.bp_table:
            self.bp_table = [1] * len(self.bp_table)
        if self.cache:
            self.cache = [None] * len(self.cache)

    def _sync(self, sf: int, curr: _Slot, *, thaddr: int = 0, exc: _Slot | None = None,
              reason: str) -> None:
        p = self.p
        b = curr.b
        pkt = Packet(reason=reason)
        pkt.add("format", F_SYNC, 2)
        pkt.add("subformat", sf, 2)
        pkt.add("branch", 0 if b.itype == I.TAKEN else 1, 1)
        pkt.add("privilege", b.priv, p.privilege_width_p)
        if not p.notime_p:
            pkt.add("time", b.cycle, p.time_width_p)
        if not p.nocontext_p:
            pkt.add("context", b.context, p.context_width_p)
        self.trap_reported = False
        if sf == SF_START:
            pkt.add("address", b.iaddr >> p.iaddress_lsb_p, p.address_bits)
            self.last_address = b.iaddr
        else:
            e = exc.b
            self.trap_reported = thaddr == 0 and (self.flags["prev_updiscon"] or
                                                  self.flags["next_exception"])
            pkt.add("ecause", e.cause, p.ecause_width_p)
            intr = int(e.itype == I.INTERRUPT)
            pkt.add("interrupt", intr, 1)
            pkt.add("thaddr", thaddr, 1)
            if not (thaddr and self.mode.implicit_exception):
                pkt.add("address", b.iaddr >> p.iaddress_lsb_p, p.address_bits)
            self.last_address = b.iaddr
            if not intr:
                pkt.add("tval", e.tval, p.iaddress_width_p)
        curr.synced = True
        self._send(pkt)
        self._sync_reset()

    def _diff(self, addr: int) -> int:
        if self.mode.full_address or self.last_address is None:
            return addr >> self.p.iaddress_lsb_p
        return ((addr - self.last_address) & self.amask) >> self.p.iaddress_lsb_p

    def _status(self, pkt: Packet, msb: int, irreport: bool, irdepth: int) -> None:
        """7.6.1 / 7.6.2: *notify*, *updiscon*, *irreport* encoded against the
        bit before them, *irdepth* all that bit unless *irreport* differs."""
        notify = int(self.flags["notify"]) ^ msb
        upd = int(self.flags["updiscon"]) ^ notify
        irr = int(irreport) ^ upd
        pkt.add("notify", notify, 1)
        pkt.add("updiscon", upd, 1)
        pkt.add("irreport", irr, 1)
        n = self.p.irdepth_bits
        if n:
            pkt.add("irdepth", irdepth if irr != upd else (-upd & ((1 << n) - 1)), n)

    def _address_packet(self, curr: _Slot, reason: str, *, target: bool = False,
                        irreport: bool = False, irdepth: int = 0) -> None:
        """Format 0 / 1 / 2 with curr's address (9.1's "format selection")."""
        if self.resync_state == 2:              # (9.2: the resync follows this one)
            self.resync_state = 3
        p = self.p
        addr = curr.b.iaddr
        ab = p.address_bits
        hit = None
        if target and self.mode.jump_target_cache and self.cache:
            idx = (addr >> p.iaddress_lsb_p) & (len(self.cache) - 1)
            if self.cache[idx] == addr:
                hit = idx
            else:                               # (8.2.7: a miss replaces the entry)
                self.cache[idx] = addr
        if self.mode.branch_prediction and self.pbc >= MAX_BRANCHES:
            pkt = Packet(reason=reason)
            pkt.add("format", F_EXT, 2)
            pkt.add("subformat", F0_COUNT, p.f0s_width_p)
            pkt.add("branch_count", self.pbc - MAX_BRANCHES, 32)
            pkt.add("branch_fmt", 3 if curr.bp_fail else 2, 2)
            d = self._diff(addr)
            pkt.add("address", d, ab)
            self._status(pkt, d >> (ab - 1) & 1, irreport, irdepth)
            self.last_address = addr
            self.pbc = 0
            self._send(pkt)
            return
        pkt = Packet(reason=reason)
        if self.branches:
            pkt.add("format", F_BRANCH, 2)
            pkt.add("branches", self.branches, 5)
            pkt.add("branch_map", self.branch_map, _map_bits(self.branches)[0])
        else:
            pkt.add("format", F_ADDR, 2)
        d = self._diff(addr)
        pkt.add("address", d, ab)
        self._status(pkt, d >> (ab - 1) & 1, irreport, irdepth)
        if hit is not None and not self.flags["notify"] and not self.flags["updiscon"]:
            idx = hit
            if True:
                alt = Packet(reason=reason + " (jump target cache)")
                alt.add("format", F_EXT, 2)
                alt.add("subformat", F0_CACHE, p.f0s_width_p)
                alt.add("index", idx, p.cache_size_p)
                alt.add("branches", self.branches, 5)
                if self.branches:
                    alt.add("branch_map", self.branch_map, _map_bits(self.branches)[0])
                prev_bit = alt.fields[-1][1] >> (alt.fields[-1][2] - 1) & 1
                irr = int(irreport) ^ prev_bit
                alt.add("irreport", irr, 1)
                n = p.irdepth_bits
                if n:
                    alt.add("irdepth", irdepth if irr != prev_bit
                            else (-prev_bit & ((1 << n) - 1)), n)
                if len(encode_payload(alt)) <= len(encode_payload(pkt)):
                    self.pbc = 0
                    self.last_address = addr    # (the index reports the address)
                    self._send(alt)
                    return
        self.last_address = addr
        self.pbc = 0
        self._send(pkt)

    def _branch_only(self, curr: _Slot) -> None:
        """rpt_br: a full branch map without an address (format 1, branches
        0) or, after 31+ correct predictions, the count (format 0)."""
        if self.mode.branch_prediction and self.pbc >= MAX_BRANCHES:
            pkt = Packet(reason="prediction failed")
            pkt.add("format", F_EXT, 2)
            pkt.add("subformat", F0_COUNT, self.p.f0s_width_p)
            pkt.add("branch_count", self.pbc - MAX_BRANCHES, 32)
            pkt.add("branch_fmt", 0, 2)
        else:
            pkt = Packet(reason="branch map full")
            pkt.add("format", F_BRANCH, 2)
            pkt.add("branches", 0, 5)
            pkt.add("branch_map", self.branch_map, 31)
        self.pbc = 0
        self._send(pkt)

    # -- the algorithm --------------------------------------------------------------

    @property
    def resync_max(self) -> int:
        return 1 << (self.mode.sync_max + 4)

    def _resync_flags(self, curr: _Slot) -> tuple[bool, bool, bool]:
        """(resync_br, resync_exceeded, the next instruction will be resynced)
        — 9.2.  Counting packets (trTeInstSyncMode 1) as the reference
        model does (resync_br at the maximum, the sync once it is passed);
        cycles / halfwords (2 / 3) with 9.2's FSM: state 2 reports the
        current instruction with its address — also without branches
        (DEVIATIONS: so the format 3 follows that packet at once, the case
        *updiscon* covers; a sync several instructions after an inferable
        address would leave a decoder at the address's first occurrence) —,
        state 3 the sync."""
        m = self.mode
        if m.sync_mode == 0:
            return False, False, False
        mx = self.resync_max
        pending = self._pending()
        if m.sync_mode == 1:
            c = self.resync_count
            return c == mx and pending != 0, c > mx, c == mx
        del pending
        st = self.resync_state
        return st == 2 and curr.b.iretire > 0, st == 3, st == 2

    def _pending(self) -> int:
        """The branches not reported yet: the map's and, counting correct
        predictions, those."""
        if self.mode.branch_prediction and self.pbc >= MAX_BRANCHES:
            return self.branches + self.pbc
        return self.branches

    def _branch(self, curr: _Slot) -> None:
        """Update branch map / predictor (Figure 9.1's first box)."""
        b = curr.b
        taken = b.itype == I.TAKEN
        m = self.mode
        if m.branch_prediction and self.bp_table:
            idx = (b.iaddr >> self.p.iaddress_lsb_p) & (len(self.bp_table) - 1)
            st = self.bp_table[idx]
            ok = (st >> 1) == int(taken)
            self.bp_table[idx] = _bp_next(st, taken)
            if self.pbc >= MAX_BRANCHES:        # (counting: no branch map)
                if ok:
                    self.pbc += 1
                else:
                    curr.bp_fail = True
                return
            if ok and self.pbc >= 0:
                self.pbc += 1
            elif not ok:
                self.pbc = -1                   # (the run is broken: map it)
        self.branch_map |= (0 if taken else 1) << self.branches
        self.branches += 1
        if m.branch_prediction and self.pbc == MAX_BRANCHES:
            self.branches = 0                   # (31 predicted correctly: count on)
            self.branch_map = 0

    def _encode(self) -> None:
        prev, curr, nxt = self.prev, self.curr, self.next
        b = curr.b
        m = self.mode
        if m.sync_mode in (2, 3):
            if m.sync_mode == 3:
                self.resync_count += b.halfwords
            elif prev.b is not None:
                self.resync_count += max(0, b.cycle - prev.b.cycle)
            if self.resync_state == 1 and self.resync_count >= self.resync_max:
                self.resync_state = 2
        if curr.is_branch:
            self._branch(curr)
            self.branch_since_ret = True
        flags = self.flags = {}
        prev_upd = prev.qualified and prev.updiscon
        flags["prev_updiscon"] = prev_upd
        resync_br, resync_exceeded, next_resync = self._resync_flags(curr)
        nb = nxt.b
        flags["next_exception"] = nxt.exception
        flags["updiscon"] = prev_upd and (nxt.exception or (nb is not None and
                                                            nb.priv != b.priv)
                                          or next_resync)
        notify = (curr.notify or bool(b.trigger & I.TRACE_NOTIFY)) and \
            b.iretire > 0                       # (a notification names a retired one)
        # (7.6.1: the notify bit marks a packet sent only for the notification;
        # one sent anyway — a jump target — keeps it equal to the address MSB)
        flags["notify"] = False
        prev_exception = prev.qualified and prev.exception
        prev_reported = prev_exception and self.trap_reported
        self.trap_reported = False
        first = not prev.qualified or self.force_sync
        self.force_sync = False
        pb = prev.b
        ppccd = pb is not None and prev.qualified and (b.priv != pb.priv or (
            b.context != pb.context and (b.ctype == 2 or prev_upd)))
        ppccd_br = self._pending() != 0 and nb is not None and (nb.priv != b.priv or (
            nb.context != b.context and (nb.ctype == 2 or self._decide(curr, nxt)[0])))
        er_n = (b.iretire > 0 and curr.exception) or notify
        if b.ctype == 1 and pb is not None and b.context != pb.context:
            self.cci_pending = True
        counting = m.branch_prediction and self.pbc >= MAX_BRANCHES
        count_full = counting and self.pbc - MAX_BRANCHES >= 0xFFFFFFFF
        rpt_br = (self.branches == MAX_BRANCHES and not counting) or \
            (counting and curr.bp_fail)
        ended_ntr = False
        if prev_exception:
            if curr.exc_only:
                self._sync(SF_TRAP, curr, thaddr=0, exc=prev, reason="trap, trap")
            elif prev_reported:
                self._sync(SF_START, curr, reason="handler (trap reported)")
            else:
                self._sync(SF_TRAP, curr, thaddr=1, exc=prev, reason="trap")
        elif curr.exc_only and (first or prev_upd):
            # (7.4.1: a trap at the first traced instruction or at an uninferable
            # jump's target — the EPC cannot be inferred: thaddr 0; a trap-only
            # block is no instruction, so no sync start for it)
            self._sync(SF_TRAP, curr, thaddr=0, exc=curr, reason="trap at a jump target"
                       if prev_upd else "trap (first traced)")
        elif not curr.exc_only and (first or ppccd or resync_exceeded):
            self._sync(SF_START, curr, reason="first traced" if first else
                       "privilege / context" if ppccd else "resync")
        elif prev_upd:
            self._address_packet(curr, "uninferable jump target", target=True,
                                 irreport=prev.ir_fail is not None,
                                 irdepth=prev.ir_fail or 0)
            ended_ntr = True
        elif resync_br or er_n or count_full:
            flags["notify"] = notify
            self._address_packet(curr, "resync" if resync_br else "notify" if er_n
                                 else "branch count full", **self._ir_report(prev))
        elif curr.exc_only and not nxt.qualified:
            self._sync(SF_TRAP, curr, thaddr=0, exc=curr, reason="trap (trace ends)")
        elif nxt.exc_only or ppccd_br or not nxt.qualified:
            self._address_packet(curr, "before a trap" if nxt.exc_only else
                                 "before a privilege change" if ppccd_br else
                                 "last traced", **self._ir_report(prev))
        elif rpt_br:
            self._branch_only(curr)
        elif self.cci_pending and b.iretire:
            pkt = Packet(reason="context")
            pkt.add("format", F_SYNC, 2)
            pkt.add("subformat", SF_CONTEXT, 2)
            pkt.add("privilege", b.priv, self.p.privilege_width_p)
            if not self.p.notime_p:
                pkt.add("time", b.cycle, self.p.time_width_p)
            if not self.p.nocontext_p:
                pkt.add("context", b.context, self.p.context_width_p)
            self.cci_pending = False
            self._send(pkt, clears=False)
        if not nxt.qualified:                   # (curr was the last traced: 7.5.1)
            self._support(ENDED_NTR if ended_ntr else ENDED_REP)
        # (after the packet: a sync for curr empties the return stack first)
        curr.updiscon, curr.ir_fail = self._decide(curr, nxt)
        self._stack(curr)

    def _ir_report(self, prev: _Slot) -> dict[str, Any]:
        """7.6.2: the stack depth (or call count) on the packet of an
        instruction reached without an uninferable jump (before a trap,
        privilege change, resync, a notification, the end): after a
        correctly predicted return — also when the depth is 0 (the address
        may occur inside the call that returned, at another depth; a
        decoder never takes a return at depth 0 for an implicit one, so
        reporting 0 is safe: DEVIATIONS) —, otherwise (any depth) unless no
        return came since the last call or a branch since the last return."""
        if not self.mode.implicit_return:
            return {}
        depth = self._depth()
        if prev.qualified and prev.itype in (I.RETURN, I.COROUTINE_SWAP) and \
                prev.ir_fail is None and prev.b is not None and prev.b.iretire:
            return {"irreport": True, "irdepth": depth}
        if self.ret_since_call and not self.branch_since_ret:
            return {"irreport": True, "irdepth": depth}
        return {}

    def _depth(self) -> int:
        return len(self.stack) if self.p.return_stack_size_p else self.calls

    def _decide(self, curr: _Slot, nxt: _Slot) -> tuple[bool, int | None]:
        """(curr is an uninferable PC discontinuity, the depth of a failed
        implicit-return prediction): 9.1's *updiscon* — the uninferable
        jumps, xRET, the returns (with implicit return only when the stack is
        empty or its top is not the target) —, unless sijump makes the jump
        inferable."""
        b = curr.b
        it = b.itype
        m = self.mode
        if not b.iretire:
            return False, None
        if it == I.TRAP_RETURN:
            return True, None
        if it in (I.RETURN, I.COROUTINE_SWAP) and m.implicit_return:
            depth = self._depth()
            if depth == 0:
                return True, None
            if self.p.return_stack_size_p:
                target = nxt.b.iaddr if nxt.b is not None else None
                if target is not None and self.stack[-1] != target:
                    return True, depth
            return False, None
        if it in I.UNINFERABLE or it == I.RETURN:
            if m.sijump and b.sijump and self.prev.qualified and not curr.synced:
                return False, None
            return True, None
        return False, None

    def _stack(self, curr: _Slot) -> None:
        """curr's call / return on the implicit-return stack (after its
        packet: the decoder pushes and pops as it walks past it)."""
        m = self.mode
        b = curr.b
        if not (m.implicit_return and b.iretire):
            return
        it = b.itype
        p = self.p
        if it in (I.RETURN, I.COROUTINE_SWAP):
            if p.return_stack_size_p:
                if self.stack:
                    self.stack.pop()
            elif self.calls:
                self.calls -= 1
            if it == I.RETURN:
                self.ret_since_call = True
                self.branch_since_ret = False
        if it in I.CALLS or it == I.COROUTINE_SWAP:
            if p.return_stack_size_p:
                if len(self.stack) == 1 << p.return_stack_size_p:
                    self.stack.pop(0)
                self.stack.append(b.link)
            elif self.calls < (1 << p.call_counter_size_p):
                self.calls += 1
            self.ret_since_call = False


def _bp_next(st: int, taken: bool) -> int:
    """8.2.6's 2-bit counter: 00 -> 01 on a failure; 01 -> 00 on success, else
    11; 11 -> 10 on a failure; 10 -> 11 on success, else 00."""
    ok = (st >> 1) == int(taken)
    return {0: 0 if ok else 1, 1: 0 if ok else 3, 3: 3 if ok else 2,
            2: 3 if ok else 0}[st]


# -- the decoder ----------------------------------------------------------------------


class DecodeError(Exception):
    """The packets do not fit the program (ch. 11's ERROR cases)."""


@dataclass
class Event:
    """What the decoder reconstructs: a retired instruction ("pc"), a trap
    ("trap": cause, interrupt, tval, epc), the trace's start / end
    ("start" / "end" with the qual_status), a loss ("lost"), a context."""
    kind: str
    addr: int = 0
    cause: int = 0
    interrupt: bool = False
    tval: int = 0
    info: str = ""


def sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


class Decoder:
    """Ch. 11's decoder with the encoder's optional modes: it follows the
    program from each reported address, taking branch outcomes from the
    branch maps (or, with branch prediction, from its own copy of the
    predictor), return addresses from its own return stack, uninferable
    targets from the packets (or the jump target cache).  ``instr(addr)``
    returns the decoded instruction at ``addr`` (an :class:`emips.riscv.RV`;
    None: no code there), ``word(addr, bytes)`` reads memory (CM.JT's
    table), ``trap_vector(cause, interrupt, privilege)`` the handler
    address (implicit exception mode: the packet's privilege is the
    handler's), ``jvt`` the jvt CSR (Zcmt's table base); ``wait_sync``:
    packets before the first format 3 are skipped (a trace that wrapped)
    instead of being an error."""

    def __init__(self, params: Params, instr: Callable[[int], Any], *,
                 word: Callable[[int, int], int] | None = None,
                 trap_vector: Callable[[int, bool, int], int] | None = None,
                 jvt: int = 0, wait_sync: bool = False) -> None:
        self.p = params
        self.instr = instr
        self.word = word
        self.trap_vector = trap_vector
        self.jvt = jvt
        self.amask = (1 << params.iaddress_width_p) - 1
        self.xlen = params.iaddress_width_p
        self.events: list[Event] = []
        self.options: dict[str, bool] = {k: False for k in OPTIONS}
        self.pc: int | None = None
        self.last_pc: int | None = None
        self.address = 0
        self.privilege = 0
        self.start_of_trace = True
        self.inferred_address = False
        self.stop_at_last_branch = False
        # the branch outcomes still to use, oldest first: ["bit", taken, carried]
        # (a branch map's, or a format 3's — carried: the predictor skips it),
        # ["count", n] (n correct predictions), ["fail"] (the next one failed)
        self.queue: list[list] = []
        self.stack: list[int] = []
        self.bp_table = [1] * (1 << params.bpred_size_p) if params.bpred_size_p else []
        self.cache: list[int | None] = [None] * (1 << params.cache_size_p) \
            if params.cache_size_p else []
        self.pkt: Packet | None = None
        self.flags: dict[str, bool] = {}
        self.steps_left = 0
        self.wait_sync = wait_sync

    # -- reporting ------------------------------------------------------------------

    def _report(self, pc: int) -> None:
        self.events.append(Event("pc", pc))
        self.steps_left -= 1
        if self.steps_left < 0:
            raise DecodeError("no end to the walk: the packets do not fit the program")

    def pcs(self) -> list[int]:
        return [e.addr for e in self.events if e.kind == "pc"]

    # -- branch outcomes -------------------------------------------------------------

    @property
    def branches(self) -> int:
        return sum(q[1] if q[0] == "count" else 1 for q in self.queue)

    def _bits(self, bm: int, n: int) -> None:
        self.queue += [["bit", not (bm >> k & 1), False] for k in range(n)]

    def _is_branch(self, addr: int) -> bool:
        i = self.instr(addr)
        return i is not None and i.base.name in I.BRANCHES

    def _unprocessed(self, addr: int) -> bool:
        return self.branches != (1 if self._is_branch(addr) else 0)

    def _bp_index(self, addr: int) -> int:
        return (addr >> self.p.iaddress_lsb_p) & (len(self.bp_table) - 1)

    def _taken(self, addr: int) -> bool:
        if not self.queue:
            raise DecodeError(f"cannot resolve the branch at 0x{addr:x}")
        q = self.queue[0]
        bp = self.bp_table and self.options["branch_prediction"]
        if q[0] == "bit":
            self.queue.pop(0)
            if bp and not q[2]:
                k = self._bp_index(addr)
                self.bp_table[k] = _bp_next(self.bp_table[k], q[1])
            return q[1]
        if not self.bp_table:
            raise DecodeError(f"cannot resolve the branch at 0x{addr:x}")
        k = self._bp_index(addr)
        pred = bool(self.bp_table[k] >> 1)
        if q[0] == "count":
            q[1] -= 1
            if q[1] == 0:
                self.queue.pop(0)
            taken = pred
        else:
            self.queue.pop(0)
            taken = not pred
        self.bp_table[k] = _bp_next(self.bp_table[k], taken)
        return taken

    def _reset_sync(self) -> None:
        self.stack = []
        if self.bp_table:
            self.bp_table = [1] * len(self.bp_table)
        if self.cache:
            self.cache = [None] * len(self.cache)

    # -- the program --------------------------------------------------------------

    def _get(self, addr: int):
        i = self.instr(addr)
        if i is None:
            raise DecodeError(f"no instruction at 0x{addr:x}")
        return i

    def _jvt_target(self, i) -> int:
        base = (self.jvt & ~0x3F) + i.f["index"] * (self.xlen // 8)
        return self.word(base & self.amask, self.xlen // 8) & ~1 & self.amask

    def _sequential(self, i, it: int) -> int | None:
        """8.2.4: an uninferable jump whose rs1 the LUI / C.LUI / AUIPC just
        before it wrote: its target."""
        if not (self.options["sijump"] and it in I.UNINFERABLE) or self.last_pc is None \
                or self.last_pc == self.pc:
            return None
        prev = self.instr(self.last_pc)
        if prev is None or prev.base.name not in ("lui", "auipc"):
            return None
        rd = prev.f.get("rd", 0)
        if rd == 0 or rd != i.f.get("rs1"):
            return None
        t = sx(prev.f["imm"], 32)
        if prev.base.name == "auipc":
            t += self.last_pc
        return (t + i.f.get("imm", 0)) & ~1 & self.amask

    def _implicit(self, it: int, address: int) -> bool:
        """A return taken from the stack (11's is_implicit_return): not when
        the packet reports a failed prediction at this depth (*irreport*,
        *irdepth*) — unless the stack's top is the reported address itself
        or branches are left to process: a failed return's target is not its
        prediction, and it ends the walk with every branch processed (the
        same *irreport* also reports the depth before a trap: 7.6.2)."""
        if not self.options["implicit_return"] or it not in (I.RETURN, I.COROUTINE_SWAP):
            return False
        if not self.stack:
            return False
        f = self.flags
        if f.get("irreport") and self.pkt is not None and \
                self.pkt.get("irdepth", 0) == len(self.stack):
            if self.stack[-1] == address or self.branches:
                return True
            return False
        return True

    def _push(self, link: int) -> None:
        p = self.p
        size = p.return_stack_size_p or p.call_counter_size_p
        if len(self.stack) == 1 << size:
            self.stack.pop(0)
        self.stack.append(link)

    def next_pc(self, address: int) -> bool:
        """Ch. 11's next_pc: step past the instruction at pc; True when it
        is an uninferable discontinuity (pc = the reported ``address``)."""
        pc = self.pc
        i = self._get(pc)
        it = I.classify(i)
        stop = False
        size = i.size
        name = i.base.name
        seq = self._sequential(i, it)
        ir = self.options["implicit_return"]
        if it in (I.INFERABLE_CALL, I.INFERABLE_JUMP, I.OTHER_INFERABLE):
            if i.name == "cm.jalt":
                self.pc = self._jvt_target(i)
            elif name == "jal":
                self.pc = (pc + i.f["imm"]) & self.amask
            else:                               # (JALR rs1 = x0: the immediate)
                self.pc = i.f.get("imm", 0) & ~1 & self.amask
            stop = self.pc == pc                # (a jump to itself: the walk ends)
        elif seq is not None:
            self.pc = seq
        elif self._implicit(it, address):
            self.pc = self.stack.pop()
        elif it in I.UNINFERABLE or it == I.RETURN or it == I.TRAP_RETURN:
            if self.stop_at_last_branch:
                raise DecodeError(f"an uninferable jump at 0x{pc:x} before the last branch")
            if ir and it in (I.RETURN, I.COROUTINE_SWAP) and self.stack:
                self.stack.pop()                # (the encoder popped it too)
            self.pc = address
            stop = True
            if self.cache and self.options["jump_target_cache"]:
                self.cache[(address >> self.p.iaddress_lsb_p) & (len(self.cache) - 1)] \
                    = address
        elif name in I.BRANCHES:
            if self._taken(pc):
                self.pc = (pc + i.f["imm"]) & self.amask
                stop = self.pc == pc
            else:
                self.pc = (pc + size) & self.amask
        else:
            self.pc = (pc + size) & self.amask
        if ir and (it in I.CALLS or it == I.COROUTINE_SWAP):
            self._push((pc + size) & self.amask)
        self.last_pc = pc
        return stop

    # -- packets ------------------------------------------------------------------

    def feed(self, payload: bytes, budget: int = 1 << 22) -> Packet:
        """Process one te_inst (its payload bytes)."""
        pkt = decode_payload(payload, self.p, self.options)
        self.packet(pkt, budget)
        return pkt

    def _status(self, pkt: Packet) -> None:
        f = self.flags = {}
        if pkt.get("notify") is None:
            if pkt.format == F_EXT and pkt.subformat == F0_CACHE:
                br = pkt.get("branches", 0)     # (7.8.3: against the bit before it)
                prev = pkt.get("branch_map", 0) >> (_map_bits(br)[0] - 1) & 1 if br else 0
                f["irreport"] = pkt.get("irreport") != prev
            return
        ab = self.p.address_bits
        msb = pkt.get("address", 0) >> (ab - 1) & 1
        f["notify"] = pkt.get("notify") != msb
        f["updiscon"] = pkt.get("updiscon") != pkt.get("notify")
        f["irreport"] = pkt.get("irreport") != pkt.get("updiscon")

    def _new_address(self, pkt: Packet) -> None:
        lsb = self.p.iaddress_lsb_p
        a = pkt.get("address", 0)
        if self.options["full_address"]:
            self.address = (a << lsb) & self.amask
        else:
            self.address = (self.address + (sx(a, self.p.address_bits) << lsb)) & self.amask

    def packet(self, pkt: Packet, budget: int = 1 << 22) -> None:
        self.pkt = pkt
        self.steps_left = budget
        self._status(pkt)
        p = self.p
        fmt = pkt.format
        if fmt == F_SYNC:
            sf = pkt.subformat
            if sf == SF_SUPPORT:
                self._support(pkt)
                return
            if sf == SF_CONTEXT:
                self.privilege = pkt.get("privilege", 0)
                self.events.append(Event("context", info=f"context 0x{pkt.get('context', 0):x}"))
                return
            if sf == SF_TRAP:
                intr = bool(pkt.get("interrupt"))
                epc = None if intr else self._exception_address(pkt)
                self.events.append(Event("trap", epc or 0, pkt.get("ecause", 0), intr,
                                         pkt.get("tval", 0)))
                if not pkt.get("thaddr"):
                    return
            self.inferred_address = False
            if sf == SF_TRAP and pkt.get("address") is None:
                if self.trap_vector is None:
                    raise DecodeError("implicit exception mode needs the trap vectors")
                self.address = self.trap_vector(pkt.get("ecause", 0),
                                                bool(pkt.get("interrupt")),
                                                pkt.get("privilege", 3)) & self.amask
            else:
                self.address = (pkt.get("address", 0) << p.iaddress_lsb_p) & self.amask
            if sf == SF_TRAP or self.start_of_trace:
                self.queue = []
            if self._is_branch(self.address):   # (1 unprocessed branch: this one)
                self.queue.append(["bit", pkt.get("branch") == 0, True])
            if sf == SF_START and not self.start_of_trace:
                self._follow(self.address, pkt)
            else:
                if self.start_of_trace:         # (the marker before its instruction)
                    self.events.append(Event("start", self.address))
                self.pc = self.address
                self._report(self.pc)
            # (a jump a format 3 reports is never sequentially inferred — the
            # encoder cannot know the decoder saw the LUI / AUIPC before it)
            self.last_pc = self.pc
            self.privilege = pkt.get("privilege", 0)
            self.start_of_trace = False
            self._reset_sync()
            return
        if self.start_of_trace:
            if self.wait_sync:
                return
            raise DecodeError("the trace must start with a format 3 packet")
        cached = None
        if fmt == F_EXT and pkt.subformat == F0_CACHE:
            cached = pkt.get("index", 0)        # (resolved once the walk to the
            self.stop_at_last_branch = False    # previous address is complete)
            br = pkt.get("branches", 0)
            if br:
                self._bits(pkt.get("branch_map", 0), br)
        elif fmt == F_EXT:
            self.queue.append(["count", pkt.get("branch_count", 0) + MAX_BRANCHES])
            bf = pkt.get("branch_fmt", 0)
            if bf == 0:
                self.queue.append(["fail"])
                self.stop_at_last_branch = True
            else:
                self.stop_at_last_branch = False
                self._new_address(pkt)
                if bf == 3:
                    self.queue.append(["fail"])
        else:
            if fmt == F_ADDR or pkt.get("branches"):
                self.stop_at_last_branch = False
                self._new_address(pkt)
            if fmt == F_BRANCH:
                br = pkt.get("branches", 0)
                self.stop_at_last_branch = br == 0
                self._bits(pkt.get("branch_map", 0), br or MAX_BRANCHES)
        self._follow(self.address, pkt, cached)

    def _support(self, pkt: Packet) -> None:
        opts = pkt.get("ioptions", 0)
        for k, name in enumerate(self.p.options):
            self.options[name] = bool(opts >> k & 1)
        q = pkt.get("qual_status", 0)
        if q == TRACE_LOST:
            self.events.append(Event("lost"))
        if q != NO_CHANGE:
            if self.pc is not None and not self.start_of_trace:
                if q == ENDED_NTR and self.inferred_address:
                    prev = self.pc
                    self.inferred_address = False
                    while not self.next_pc(prev):
                        self._report(self.pc)
                    self._report(self.pc)
                self.events.append(Event("end", self.pc or 0, info=QUAL_NAMES[q]))
            self.start_of_trace = True

    def _exception_address(self, pkt: Packet) -> int | None:
        """The exception's EPC (11's exception_address): the reported address
        when it follows an uninferable jump, else the instruction after the
        last one reported (ECALL / EBREAK do not retire: they are that
        instruction)."""
        if self.pc is None or self.start_of_trace:
            return None
        i = self.instr(self.pc)
        if i is None:
            return None
        it = I.classify(i)
        if (it in I.UNINFERABLE or it in (I.RETURN, I.TRAP_RETURN)) and not pkt.get("thaddr"):
            return (pkt.get("address", 0) << self.p.iaddress_lsb_p) & self.amask
        save = (self.pc, self.last_pc)
        try:
            self.next_pc(self.pc)
            return self.pc
        except DecodeError:
            return None
        finally:
            self.pc, self.last_pc = save

    def _follow(self, address: int, pkt: Packet, cached: int | None = None) -> None:
        """Ch. 11's follow_execution_path (``cached``: a jump target cache
        index — the address, known once the walk to the previous one ends)."""
        if self.pc is None:
            raise DecodeError("no PC to start from")
        prev_address = self.pc
        f = self.flags
        while True:
            if self.inferred_address:
                stop = self.next_pc(prev_address)
                self._report(self.pc)
                if stop:
                    self.inferred_address = False
                continue
            if cached is not None:
                a = self.cache[cached] if self.cache else None
                if a is None:
                    raise DecodeError(f"jump target cache entry {cached} is empty")
                address = self.address = a
                cached = None
            stop = self.next_pc(address)
            self._report(self.pc)
            pc = self.pc
            if self.branches == 1 and self._is_branch(pc) and self.stop_at_last_branch:
                self.stop_at_last_branch = False
                return
            if stop:
                if self._unprocessed(pc):
                    raise DecodeError(f"unprocessed branches at 0x{pc:x}")
                return
            if pkt.format != F_SYNC and pc == address and not self.stop_at_last_branch \
                    and f.get("notify") and not self._unprocessed(pc) and (
                        not f.get("irreport") or pkt.get("irdepth", 0) == len(self.stack)):
                return
            if pkt.format != F_SYNC and pc == address and not self.stop_at_last_branch \
                    and not f.get("updiscon") \
                    and not self._unprocessed(pc) and (
                        not f.get("irreport") or pkt.get("irdepth", 0) == len(self.stack)):
                self.inferred_address = True
                return
            if pkt.format == F_SYNC and pc == address and not self._unprocessed(pc):
                return
