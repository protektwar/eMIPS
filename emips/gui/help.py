"""Help > ... pages (HTML shown in a dialog).  Texts go through
:func:`~emips.gui.i18n.tr` paragraph by paragraph so they can be
translated."""

from __future__ import annotations

import html
import platform

from .. import __version__
from ..syntax import GPR_NAMES
from .i18n import N_, tr
from .qt import API, QtCore

E = html.escape

AUTHOR = "Alexandru Herlas"
AUTHOR_EMAIL = "45262264+protektwar@users.noreply.github.com"
HOMEPAGE = "https://github.com/protektwar/eMIPS"


def quick_start() -> str:
    steps = [
        tr("Write a program in the editor, open one (File > Open: .asm / .s source "
           "or an Intel HEX image), or pick one from File > Examples."),
        tr("Assemble it (Ctrl+B). Errors go to the Messages pane and the line is "
           "marked in red."),
        tr("Step (F10) one instruction at a time — or one clock cycle in the "
           "5-stage pipeline mode. Step back (Shift+F10) undoes a step. Run (F5) "
           "runs until a breakpoint, the end of the program, an error or an input "
           "request; Pause (F6) stops it."),
        tr("Breakpoints: click a line number or press F9 on a line. Run to cursor "
           "(Ctrl+F10) runs up to the line the cursor is on."),
        tr("Watch: the Datapath (what every stage does), the Pipeline chart "
           "(instructions x cycles, stalls), the Instruction details (bit fields, ALU, "
           "address, branch), Registers and Memory. The amber line in the editor is "
           "the next instruction to execute."),
        tr("Console: program output; type a line there when the program reads input "
           "(read_int ...), or type on the memory-mapped keyboard."),
        tr("Tools: the instruction reference (every instruction of the selected "
           "release, its encoding and syntax), the encoder / decoder (a word in, its "
           "fields out — or an instruction in, its machine words out) and the number "
           "converter (hex, binary, signed, IEEE-754)."),
        tr("Panes can be moved, tabbed together or detached (drag the title bar); "
           "a detached pane is a window of its own (minimize, maximize, resize) "
           "with a Dock button that puts it back. View > Reset layout restores "
           "the default arrangement."),
        tr("The whole user guide — every pane, the command line, the computer "
           "architecture behind it and the labs — is under Help > User Guide (F1)."),
    ]
    return (f"<h2>{E(tr('Quick start'))}</h2><ol>"
            + "".join(f"<li>{E(s)}</li>" for s in steps) + "</ol>")


SYSCALLS = [
    (1, "print_int", "$a0 = integer"),
    (2, "print_float", "$f12 = float"),
    (3, "print_double", "$f12 = double"),
    (4, "print_string", "$a0 = address of a NUL-terminated string"),
    (5, "read_int", "-> $v0"),
    (6, "read_float", "-> $f0"),
    (7, "read_double", "-> $f0"),
    (8, "read_string", "$a0 = buffer, $a1 = size"),
    (9, "sbrk", "$a0 = bytes -> $v0 = address"),
    (10, "exit", ""),
    (11, "print_char", "$a0 = character"),
    (12, "read_char", "-> $v0"),
    (17, "exit2", "$a0 = exit code"),
    (34, "print_hex", "$a0 = integer"),
    (35, "print_binary", "$a0 = integer"),
    (36, "print_unsigned", "$a0 = integer"),
]


def syscalls() -> str:
    rows = "".join(f"<tr><td align=right>{n}</td><td><tt>{name}</tt></td>"
                   f"<td><tt>{E(args)}</tt></td></tr>" for n, name, args in SYSCALLS)
    return (f"<h2>{E(tr('System calls'))}</h2><p>"
            + E(tr("Put the service number in $v0, the arguments in $a0..$a3 (or "
                   "$f12), then execute syscall. These are the MARS / SPIM services "
                   "(operating-system emulation); any other number takes the real "
                   "System Call exception."))
            + f"</p><table cellpadding=3><tr><th>$v0</th><th>{E(tr('service'))}</th>"
            f"<th>{E(tr('arguments / result'))}</th></tr>{rows}</table>")


REG_USE = {0: N_("constant 0"), 1: N_("assembler temporary"), 2: N_("results"),
           4: N_("arguments"), 8: N_("temporaries"), 16: N_("saved"),
           24: N_("temporaries"), 26: N_("kernel"), 28: N_("global pointer"),
           29: N_("stack pointer"), 30: N_("frame pointer"), 31: N_("return address")}
_SPAN = {2: 2, 4: 4, 8: 8, 16: 8, 24: 2, 26: 2}


def registers() -> str:
    rows = []
    for first, use in REG_USE.items():
        n = _SPAN.get(first, 1)
        names = ", ".join(f"${GPR_NAMES[i]}" for i in range(first, first + n))
        nums = f"${first}" if n == 1 else f"${first}-${first + n - 1}"
        saved = tr("yes") if first in (16, 28, 29, 30, 31) else ""
        rows.append(f"<tr><td><tt>{nums}</tt></td><td><tt>{names}</tt></td>"
                    f"<td>{E(tr(use))}</td><td>{saved}</td></tr>")
    return (f"<h2>{E(tr('Registers'))}</h2><p>"
            + E(tr("The 32 general-purpose registers and their use in the MIPS "
                   "calling convention (o32). \"Preserved\" registers must have the "
                   "same value after a function call."))
            + f"</p><table cellpadding=3><tr><th>{E(tr('number'))}</th><th>"
            f"{E(tr('name'))}</th><th>{E(tr('use'))}</th><th>{E(tr('preserved'))}"
            "</th></tr>" + "".join(rows) + "</table><p>"
            + E(tr("Also: PC, HI and LO (multiply / divide), the CP0 registers (Status, "
                   "Cause, EPC, BadVAddr, Count, Compare ...) and the FPU registers "
                   "$f0-$f31 with FCSR."))
            + "</p>")


def about() -> str:
    qt_version = QtCore.qVersion()
    return (f"<h2>eMIPS {E(__version__)}</h2><p>"
            + E(tr("A MIPS32 teaching emulator: assembler, disassembler, the "
                   "instruction set of every MIPS32 release (R1-R6), CP0 exceptions and "
                   "interrupts, an IEEE-754 FPU, caches, TLBs, the classic 5-stage "
                   "pipeline with its datapath, forwarding and stalls, the MIPS "
                   "24K's 8-stage pipeline with branch prediction, and systems of up to "
                   "four cores (the Coherence Manager, the Cluster Power Controller, the "
                   "Global Interrupt Controller)."))
            + "</p><p>" + E(tr("Behaviour follows the MIPS32 Architecture for "
                               "Programmers manuals."))
            + "</p><p>" + E(tr("Developed by {name}")).format(
                name=f"<b>{E(AUTHOR)}</b>")
            + f" &lt;{E(AUTHOR_EMAIL)}&gt;<br><a href=\"{E(HOMEPAGE)}\">{E(HOMEPAGE)}</a></p><p>"
            + E(tr("eMIPS began in 2021 as a Ruby library and a Discord bot; in 2026 it "
                   "was rewritten in Python and grew into this teaching emulator. It was "
                   "developed with the help of Claude (Anthropic), used as an AI "
                   "programming assistant."))
            + "</p><p>" + E(tr("License: GPL-3.0-or-later. Copyright © 2021-2026 {name}."))
            .format(name=E(AUTHOR))
            + f"</p><p>Python {E(platform.python_version())}, {E(API)} / Qt "
            f"{E(qt_version)}</p>")
