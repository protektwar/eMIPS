"""MARS's / RARS's dialog services (PLAN M27 phase S6): 50 ConfirmDialog,
51-54 the input dialogs (int, float, double, string), 55-59 the message
dialogs (RARS: 55, 56, 58 double, 59 string, 60 float) — each service's
``simulate`` from MARS 4.5's ``mars/mips/instructions/syscalls/`` and RARS
1.6's ``rars/riscv/syscalls/`` (``JOptionPane``).

A run must replay exactly (Step Back, checks), so a dialog's answer is an
input line like read_int's: the window asks with a real dialog and gives
the answer (:meth:`Session.give_input`), the command line reads it from
the program's input (a ``# input:`` / checks line too); Cancel is the
line :data:`CANCEL`, a confirm dialog closed without a choice
:data:`CLOSED` — typed ``!cancel`` / ``!close`` on the command line
(:func:`typed`).  The message dialogs need no answer: they are recorded
(:class:`Dialog`, the services' ``dialogs``), shown by the window and
printed by the command line — never part of the program's output.
"""

from __future__ import annotations

from dataclasses import dataclass

CANCEL = "\x18"                             # the Cancel button (an input dialog)
CLOSED = "\x1b"                             # a confirm dialog closed (Escape)
#: input dialogs by service number (the same on MARS and RARS)
INPUT = {50: "confirm", 51: "int", 52: "float", 53: "double", 54: "string"}
MARS_MESSAGES = {55: "message", 56: "int", 57: "float", 58: "double", 59: "string"}
RARS_MESSAGES = {55: "message", 56: "int", 58: "double", 59: "string", 60: "float"}
FLOAT_SERVICES = {52, 53, 57, 58, 60}       # (none on a core without FPRs)
#: MessageDialog's $a1: JOptionPane's message types (else plain)
ICONS = {0: "error", 1: "information", 2: "warning", 3: "question"}
TITLES = {50: "ConfirmDialog", 51: "InputDialogInt", 52: "InputDialogFloat",
          53: "InputDialogDouble", 54: "InputDialogString", 55: "MessageDialog",
          56: "MessageDialogInt", 57: "MessageDialogFloat", 58: "MessageDialogDouble",
          59: "MessageDialogString", 60: "MessageDialogFloat"}


@dataclass(frozen=True)
class Dialog:
    """One dialog the program showed: its service, kind ("confirm", "int"
    ... for input dialogs; "message" for 55-60), the text, the icon of a
    message dialog and an input dialog's answer (:data:`CANCEL` ...)."""
    service: int
    kind: str
    text: str
    icon: str = "information"
    answer: str | None = None

    @property
    def is_input(self) -> bool:
        return self.answer is not None

    def line(self) -> str:
        """For the command line and the window's Messages."""
        head = f"[{TITLES.get(self.service, 'dialog')}]"
        text = self.text.replace("\n", "\\n")
        if self.answer is None:
            return f"{head} {text}" + (f" ({self.icon})" if self.icon not in
                                       ("information", "plain") else "")
        answer = ("Cancel" if self.answer == CANCEL else "closed" if self.answer == CLOSED
                  else repr(self.answer))
        return f"{head} {text} -> {answer}"


def typed(line: str) -> str:
    """A line typed for a dialog on the command line: ``!cancel`` = Cancel,
    ``!close`` = closed."""
    t = line.strip().lower()
    return CANCEL if t == "!cancel" else CLOSED if t == "!close" else line


def confirm_value(answer: str, rars: bool) -> int:
    """JOptionPane.showConfirmDialog: 0 Yes, 1 No, 2 Cancel, -1 closed
    (RARS turns closed into Cancel).  Typed: y / yes / 0, n / no / 1,
    anything else Cancel."""
    if answer == CLOSED:
        return 2 if rars else -1
    a = answer.strip().lower()
    return 0 if a in ("y", "yes", "0") else 1 if a in ("n", "no", "1") else 2


def parse_int(text: str) -> int:
    """Integer.parseInt (no trimming: " 5" is not a number); ValueError."""
    body = text[1:] if text[:1] in "+-" else text
    if not body or not all("0" <= c <= "9" for c in body):
        raise ValueError(text)
    v = int(text)
    if not -(1 << 31) <= v < 1 << 31:
        raise ValueError(text)
    return v


def parse_float(text: str) -> float:
    """Float.parseFloat / Double.parseDouble (trimmed; a trailing f / d,
    Infinity, NaN); ValueError."""
    from .rvservices import _java_float
    return float(_java_float(text))


def string_bytes(answer: bytes, max_length: int, utf8: bool) -> tuple[list[tuple[int, int]], int]:
    """InputDialogString's stores ((offset, byte) ...) and its $a1 / a1
    flag: at most max - 1 characters, a newline when it fits, a NUL; -4
    when the answer was longer.  MARS counts characters, RARS UTF-8 bytes;
    MARS writes the NUL at min(len + 1, max - 1) (so with no room for the
    newline the string ends at max - 1)."""
    n = len(answer)
    stores = [(i, answer[i]) for i in range(min(n, max_length - 1))]
    if utf8:                                    # RARS
        k = min(max_length - 1, n)
        if k < max_length - 1:
            stores.append((k, 0x0A))
            k += 1
        stores.append((k, 0))
    else:                                       # MARS
        if n < max_length - 1:
            stores.append((min(n, max_length - 2), 0x0A))
        stores.append((min(n + 1, max_length - 1), 0))
    return stores, -4 if n > max_length - 1 else 0
