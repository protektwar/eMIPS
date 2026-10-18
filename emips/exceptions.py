"""Simulator-level exceptions (M1).

``emips`` core stays zero-dependency, so these are plain Python exceptions
(no external library).  They carry enough context for the step loop to log
what happened and for tests to assert the *kind* of failure.

Two kinds of failure exist since M2:

* **Architectural exceptions** (address error, overflow, trap, syscall,
  reserved instruction, coprocessor unusable, ...).  By default the
  :class:`~emips.sim.Machine` handles them like the hardware does (CP0
  ``EPC``/``Cause``/``BadVAddr``/``Status.EXL`` + jump to the exception
  vector) and raises nothing.  Only in ``exceptions="raise"`` (debug) mode
  are they raised as :class:`MipsException` (or the more specific legacy
  classes below), each carrying ``exc_code`` (``Cause.ExcCode``).
* **Simulator conditions** that real hardware has no equivalent for —
  fetching from memory that holds no code (:class:`FetchError`), a row not
  executed yet (:class:`UnimplementedError`), a CP0 register the table does
  not model (:class:`CP0RegisterError`).  These always raise.

The eMIPS error *codes* that pre-date M1 (``OPCODE_INSTRUCTION_NOT_FOUND`` =
201, the file-load 100/101/102) live in :mod:`emips.instruction` /
:mod:`emips.hexload`.  The codes here are *sim-local* tags used to identify
the failure class; hardware-style trap/service codes are exposed on the more
specific subclasses (:attr:`TrapException.trapcode`,
:attr:`SyscallException.service`).
"""

from __future__ import annotations


class SimException(Exception):
    """Base class for every error the eMIPS step loop can raise."""

    #: sim-local failure tag (0 = generic); subclasses may override.
    code: int = 0

    def __init__(
        self,
        message: str,
        *,
        name: str | None = None,
        word: int | None = None,
        addr: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.name = name
        self.word = word
        self.addr = addr


class MipsException(SimException):
    """An architectural exception raised in ``exceptions="raise"`` mode.

    ``exc_code`` is the ``Cause.ExcCode`` value (see :mod:`emips.cp0`),
    ``badvaddr`` the faulting address for address errors, ``ce`` the
    coprocessor number for Coprocessor Unusable.
    """

    def __init__(
        self,
        message: str,
        *,
        name: str | None = None,
        word: int | None = None,
        addr: int | None = None,
        exc_code: int | None = None,
        badvaddr: int | None = None,
        ce: int = 0,
    ) -> None:
        super().__init__(message, name=name, word=word, addr=addr)
        self.exc_code = exc_code
        self.badvaddr = badvaddr
        self.ce = ce


class FetchError(SimException):
    """An instruction fetch read a byte that was never written to memory
    (simulator halt: there is no program there, e.g. no exception handler
    loaded at the vector)."""

    code = 1


class UnmappedAddress(FetchError):
    """A view (debugger, loader, syscall service) asked for a virtual
    address no TLB entry maps now: there is no physical address to show.
    A real access would take a TLB exception instead."""

    code = 2


class ProbeWaitError(FetchError):
    """Debug Mode software accessed dmseg (EJTAG memory, 0xFF200000 -
    0xFF2FFFFF) while no probe services it (DCR.ProbEn = 0): the access
    "hangs until it is satisfied by the probe" (MD00047 2.2.2.1), so the
    core never goes on (simulator halt)."""

    code = 3


class BusError(FetchError):
    """A physical address the board decodes as nothing (PLAN M15: the
    Malta's GT-64120, :mod:`emips.malta`).  On a real read the core takes
    a Bus Error exception (IBE on a fetch, DBE on a load: the Machine turns
    this into one); a view or a loader that meets it has nothing to show."""

    code = 4

    def __init__(self, pa: int, kind: str = "read", message: str | None = None) -> None:
        super().__init__(message or f"bus error: nothing at physical 0x{pa & 0xFFFFFFFF:08x}",
                         addr=pa)
        self.pa = pa
        self.kind = kind


class IllegalInstructionError(MipsException):
    """A Reserved Instruction (``exc_code`` 10) for a word the table
    decodes but the core does not implement (paired-single with
    ``Status.CU1 = 1``, ``JALX`` without MIPS16e, ``DERET`` outside Debug
    Mode), raised in ``exceptions="raise"`` mode.

    This is the *architecturally invalid* case, as opposed to
    :class:`UnimplementedError` which is "the simulator has not wired this
    row yet".
    """

    code = 8  # informational: MIPS "illegal instruction" (0x08)


class UnimplementedError(SimException):
    """A row the M1 scheduler has not wired yet.

    Used for (a) the COP1 FPU family in Slice 1 (wired in Slice 2), (b) the
    overlay CP0/TLB rows deferred to M2 (``TLB*``/``DERET``), (c) the
    unambiguous-to-decode-but-condition-ambiguous PIC32 conditional moves.
    Carries the instruction *name* so the CLI/tests can report it precisely.
    """

    code = 2


class TrapException(MipsException):
    """An explicit trap (``TEQ/TNE/T-*``, ``exc_code`` 13 = Tr) or a
    ``BREAK`` (``exc_code`` 9 = Bp), raised in ``exceptions="raise"`` mode.
    ``trapcode`` is the instruction's code field (0 for the immediate
    traps).
    """

    code = 0  # per-trap code carried in ``trapcode``

    def __init__(
        self,
        message: str,
        *,
        name: str | None = None,
        word: int | None = None,
        addr: int | None = None,
        trapcode: int | None = None,
        exc_code: int = 13,
    ) -> None:
        super().__init__(message, name=name, word=word, addr=addr, exc_code=exc_code)
        self.trapcode = trapcode


class SyscallException(MipsException):
    """``SYSCALL`` (``exc_code`` 8 = Sys) raised in ``exceptions="raise"``
    mode; ``service`` is the instruction's 20-bit code field.  When
    ``machine.syscall_handler`` is installed it services the call instead
    and no exception happens (OS emulation, as in SPIM/MARS)."""

    code = 9  # sim-local tag (pre-M2 value, kept)

    def __init__(
        self,
        message: str,
        *,
        name: str | None = None,
        word: int | None = None,
        addr: int | None = None,
        service: int | None = None,
    ) -> None:
        super().__init__(message, name=name, word=word, addr=addr, exc_code=8)
        self.service = service


class CP0RegisterError(SimException):
    """``MFC0``/``MTC0`` named a ``(register, selector)`` pair that the
    loaded CP0 table does not define (e.g. registers 20/21/22, which the
    PIC32 leaves absent, or the malformed ``(10, 7)`` entry).

    Real hardware would raise an "invalid physical address" / CP0
    exception; here we carry the offending pair so tests/CLI can report it.
    """

    code = 3  # sim-local: invalid CP0 register/selector

    def __init__(
        self,
        message: str,
        *,
        name: str | None = None,
        word: int | None = None,
        addr: int | None = None,
        reg: int | None = None,
        sel: int | None = None,
    ) -> None:
        super().__init__(message, name=name, word=word, addr=addr)
        self.reg = reg
        self.sel = sel
