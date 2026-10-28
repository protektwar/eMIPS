"""The trace ingress port of a RISC-V hart (LF11): what the hart tells a
trace encoder about every instruction it retires and every trap it takes
— Efficient Trace for RISC-V 2.0 ch. 4 (the Instruction Trace Interface,
Tables 4.1-4.6), which the N-Trace 1.0 specification uses unchanged (its
ch. 3 maps each instruction to its *itype*).

The hart retires one instruction a cycle (single retirement, 4.2.1: the
SR signal group, ``iretire`` 0 or 1, ``ilastsize`` = log2 of its size in
halfwords) and gives the encoder one :class:`Block` per retired
instruction and one per trap.  A trap is a block of its own with
``iretire`` = 0 and ``itype`` 1 (exception) or 2 (interrupt): the
trapping instruction did not retire (priv. 3.3.1 — ECALL / EBREAK do not
retire either, as N-Trace's Table 1 says; ``minstret`` does not count
them), ``iaddr`` is the instruction the trap is taken at (xepc), ``cause``
/ ``tval`` what the trap writes to xcause / xtval, ``priv`` the mode it
was taken from (the handler's mode starts with its first instruction).

*itype* (Table 4.4, 4 bits: implicit return needs the full classes) comes
from the instruction's encoding and, for a branch, whether it was taken —
:func:`classify`, shared with the decoders (``emips.etrace`` /
``emips.ntrace``), so the hart and a decoder never disagree:

* JAL rd (and C.J / C.JAL, which expand to it): a call (9) when rd is a
  link register (x1 / x5), a jump without linkage (11) when rd is x0, else
  another jump with linkage (15) — all inferable;
* JALR rd, rs1 with rs1 = x0: the target is the immediate, a constant in
  the opcode — inferable by E-Trace's strict definition (4.1.1; the
  reference flow's ``is_inferrable_jump``): 9 / 11 / 15 as JAL;
* JALR rd, rs1 otherwise (and C.JR / C.JALR): N-Trace's Table 1 — an
  uninferable call (8: rd a link, rs1 not, or rd = rs1 = a link), a
  co-routine swap (12: both links, different), a return (13: rs1 a link,
  rd not), an uninferable jump (10: rd = x0) or another uninferable jump
  with linkage (14);
* BEQ ... BGEU (and C.BEQZ / C.BNEZ): 4 not taken, 5 taken;
* MRET, SRET, MNRET: 3 (DRET runs in Debug Mode, which is never traced);
* CM.POPRET / CM.POPRETZ: a return (13); CM.JT / CM.JALT: an inferable
  jump (11) / call (9) — N-Trace's Table 1: jump tables are static and
  known to the decoder;
* everything else 0.

With a 3-bit *itype* (``itype_width`` 3) the uninferable classes 8, 10,
12, 13, 14 become 6 and the inferable ones 0 (4.2: "itype = 0 is used for
inferrable calls").

``sijump`` (4.1.1, sequentially inferable jumps): an uninferable jump
(6, 8, 10, 12, 14) whose rs1 the instruction retired just before it wrote
with LUI / C.LUI / AUIPC.  ``priv`` (Table 4.5, 3 bits): U 0, S / HS 1,
M 3, VU 5, VS 6 (D = 4: Debug Mode is not traced — the hart signals
``halted`` instead, 4.2.4).  ``context`` is ``scontext`` (Sdtrig 5.7.8:
what an operating system writes to name the process it runs — the
information E-Trace's 4.2 and N-Trace's Ownership message carry), with
``ctype`` 1 (report imprecisely, Table 4.6) on the first block after it
changed; ``hcontext`` is :attr:`Block.hcontext` for N-Trace's second
Ownership message.  ``trigger`` (Table 4.7) holds the trigger module's
trace actions this cycle (Debug spec Table 12: 2 trace-on = bit 0,
3 trace-off = bit 1, 4 trace-notify = bit 2).

Choices: ``sim.DEVIATIONS["RISC-V trace"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .riscv import RV
    from .simrv import RVMachine

# itype (Table 4.4)
NONE, EXCEPTION, INTERRUPT, TRAP_RETURN, NOT_TAKEN, TAKEN = 0, 1, 2, 3, 4, 5
UNINFERABLE_JUMP3 = 6                       # (3-bit itype: every uninferable jump)
UNINFERABLE_CALL, INFERABLE_CALL, UNINFERABLE_JUMP, INFERABLE_JUMP = 8, 9, 10, 11
COROUTINE_SWAP, RETURN, OTHER_UNINFERABLE, OTHER_INFERABLE = 12, 13, 14, 15
ITYPE_NAMES = {0: "none", 1: "exception", 2: "interrupt", 3: "trap return",
               4: "not-taken branch", 5: "taken branch", 6: "uninferable jump",
               8: "uninferable call", 9: "inferable call", 10: "uninferable jump",
               11: "inferable jump", 12: "co-routine swap", 13: "return",
               14: "other uninferable jump", 15: "other inferable jump"}
#: the uninferable jumps (sijump applies to them: 4.2 Table 4.1)
UNINFERABLE = frozenset((UNINFERABLE_JUMP3, UNINFERABLE_CALL, UNINFERABLE_JUMP,
                         COROUTINE_SWAP, OTHER_UNINFERABLE))
CALLS = frozenset((UNINFERABLE_CALL, INFERABLE_CALL))
BRANCHES = frozenset(("beq", "bne", "blt", "bge", "bltu", "bgeu"))
TRAP_RETURNS = frozenset(("mret", "sret", "mnret"))
_LINK = (1, 5)
#: priv (Table 4.5)
PRIV_U, PRIV_S, PRIV_M, PRIV_D, PRIV_VU, PRIV_VS = 0, 1, 3, 4, 5, 6
PRIV_NAMES = {0: "U", 1: "S", 3: "M", 4: "D", 5: "VU", 6: "VS"}
#: trigger (Table 4.7): the trace actions 2-4 (Debug spec Table 12)
TRACE_ON, TRACE_OFF, TRACE_NOTIFY = 1, 2, 4


def priv_code(priv: int, v: int) -> int:
    """Table 4.5's priv for the hart's (privilege, V)."""
    if v:
        return PRIV_VS if priv == 1 else PRIV_VU
    return priv


def classify(i: RV) -> int:
    """The *itype* of a retired instruction with a 4-bit itype, without
    the branch outcome: a branch is :data:`NOT_TAKEN` (the caller makes it
    :data:`TAKEN`).  Shared by the hart and the decoders."""
    name = i.base.name
    if name in BRANCHES:
        return NOT_TAKEN
    if name == "jal" or name == "jalr":
        rd = i.f.get("rd", 0)
        rs = i.f.get("rs1", 0) if name == "jalr" else 0
        if name == "jal" or rs == 0:            # (a constant target: inferable)
            return INFERABLE_CALL if rd in _LINK else INFERABLE_JUMP if rd == 0 \
                else OTHER_INFERABLE
        if rd in _LINK:
            if rs in _LINK and rd != rs:
                return COROUTINE_SWAP
            return UNINFERABLE_CALL
        if rs in _LINK:
            return RETURN
        return UNINFERABLE_JUMP if rd == 0 else OTHER_UNINFERABLE
    if name in TRAP_RETURNS:
        return TRAP_RETURN
    rn = i.name
    if rn in ("cm.popret", "cm.popretz"):
        return RETURN
    if rn == "cm.jalt":                         # (CM.JT: index < 32)
        return INFERABLE_CALL if i.f.get("index", 0) >= 32 else INFERABLE_JUMP
    return NONE


def to_3bit(itype: int) -> int:
    """4.2: with ``itype_width_p`` 3 the uninferable jumps are 6 and the
    inferable ones 0."""
    if itype >= 8:
        return UNINFERABLE_JUMP3 if itype in UNINFERABLE or itype == RETURN else NONE
    return itype


def link_of(i: RV, pc: int, xlen_mask: int) -> int:
    """The return address a call pushes: the instruction after it."""
    return (pc + i.size) & xlen_mask


@dataclass(slots=True)
class Block:
    """One cycle on the ingress port (Tables 4.1 / 4.3)."""
    itype: int
    iaddr: int
    iretire: int                    # 0 or 1 (single retirement)
    ilastsize: int                  # log2 of the size in halfwords (0: 16-bit, 1: 32-bit)
    priv: int
    cause: int = 0
    tval: int = 0
    context: int = 0
    ctype: int = 0
    sijump: int = 0
    trigger: int = 0                # TRACE_ON / TRACE_OFF / TRACE_NOTIFY bits
    #: the hart's clock cycle (timestamps, the cycle-counting resync mode)
    cycle: int = 0
    #: not ingress signals: for the views, the decoders' checks and N-Trace's
    #: Ownership (hcontext) — the next PC, the link a call pushes
    npc: int = 0
    hcontext: int = 0
    link: int = 0

    @property
    def size(self) -> int:
        """The retired instruction's size in bytes (0 for a trap block)."""
        return (2 << self.ilastsize) if self.iretire else 0

    @property
    def halfwords(self) -> int:
        return (1 << self.ilastsize) if self.iretire else 0


class Consumer(Protocol):
    """What the port drives: a trace encoder (emips.rvtrace)."""

    def block(self, b: Block) -> None: ...
    def halted(self, on: bool) -> None: ...
    def reset(self) -> None: ...


@dataclass
class IngressPort:
    """A hart's ingress port (``RVMachine.ingress``): it turns the hart's
    retirements and traps into :class:`Block` s for its ``consumers``
    (and, with ``log``, keeps them: the tests' and views' record)."""
    m: RVMachine
    itype_width: int = 4
    consumers: list[Any] = field(default_factory=list)
    log: list[Block] | None = None
    #: the trigger module's trace actions not yet given to a block
    pulses: int = 0
    _last: tuple[str, int] | None = None    # (base name, rd) of the last retired
    _context: int = 0
    _ctype: int = 0

    def _emit(self, b: Block) -> None:
        if self.log is not None:
            self.log.append(b)
        for c in self.consumers:
            c.block(b)

    def _context_now(self) -> tuple[int, int]:
        tm = self.m.tm
        ctx = tm.scontext
        if ctx != self._context:
            self._context, self._ctype = ctx, 1
        ctype, self._ctype = self._ctype, 0
        return ctx, ctype

    def retire(self, pc: int, i: RV, npc: int, mode: tuple[int, int]) -> None:
        """The instruction ``i`` at ``pc`` retired (in ``mode``, the
        (privilege, V) it ran in); the hart goes on at ``npc``."""
        m = self.m
        it = classify(i)
        if it == NOT_TAKEN and npc != (pc + i.size) & m.amask:
            it = TAKEN
        sij = 0
        if it in UNINFERABLE:
            last = self._last
            if last is not None and last[0] in ("lui", "auipc") and \
                    last[1] == i.f.get("rs1", -1) and last[1] != 0:
                sij = 1
        self._last = (i.base.name, i.f.get("rd", 0))
        ctx, ctype = self._context_now()
        if self.itype_width == 3:
            it = to_3bit(it)
        pulses, self.pulses = self.pulses, 0
        link = (pc + i.size) & m.amask
        self._emit(Block(it, pc & m.amask, 1, 0 if i.size == 2 else 1,
                         priv_code(*mode), context=ctx, ctype=ctype, sijump=sij,
                         trigger=pulses, cycle=m.cycles, npc=npc & m.amask,
                         hcontext=m.tm.mcontext, link=link))

    def trap(self, epc: int, cause: int, tval: int, interrupt: bool,
             mode: tuple[int, int]) -> None:
        """A trap taken at ``epc`` from ``mode`` (before the instruction
        there retires; an interrupt: before it executes)."""
        m = self.m
        self._last = None
        ctx, ctype = self._context_now()
        # (trace-notify names a retired instruction: it waits for the next one)
        pulses, self.pulses = self.pulses & ~TRACE_NOTIFY, self.pulses & TRACE_NOTIFY
        self._emit(Block(INTERRUPT if interrupt else EXCEPTION, epc & m.amask, 0, 0,
                         priv_code(*mode), cause=cause, tval=0 if interrupt else
                         tval & m.amask, context=ctx, ctype=ctype, trigger=pulses,
                         cycle=m.cycles, npc=m.pc & m.amask, hcontext=m.tm.mcontext))

    def halted(self, on: bool) -> None:
        """Table 4.7's ``halted``: Debug Mode entered (on) / left."""
        self._last = None
        for c in self.consumers:
            c.halted(on)

    def reset(self) -> None:
        """Table 4.7's ``reset``: the hart was reset."""
        self._last = None
        self.pulses = 0
        for c in self.consumers:
            c.reset()
