"""Example programs for File > Examples (each ``.asm`` here, with the
release it needs).  ``tests/test_gui_session.py`` runs every one to
completion and checks what it prints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DIR = Path(__file__).parent


@dataclass(frozen=True)
class Example:
    """An example; the settings it needs are its own ``# eMIPS:`` line
    (emips.profile)."""
    file: str
    title: str
    keys: str = ""                      # MMIO keyboard input (tests)
    input: tuple[str, ...] = ()         # console lines (tests)
    output: str = ""                    # what the program prints

    @property
    def path(self) -> Path:
        return DIR / self.file

    def source(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def profile(self):
        """What its ``# eMIPS:`` line asks for (the defaults without one)."""
        from ...profile import Profile, read
        return read(self.source(), DIR) or Profile(args=[])


EXAMPLES = [
    Example("sum_array.asm", "Sum of an array (loop, load, syscalls)", output="sum = 18"),
    Example("hello_input.asm", "Console input / output (read_int)", input=("7",),
            output="Enter a number: squared = 49\n"),
    Example("factorial.asm", "Function calls and the stack (recursive factorial)",
            output="120"),
    Example("pipeline_hazards.asm", "Pipeline hazards: stalls and forwarding"),
    Example("delay_slots.asm", "Branch delay slots and branch-likely", output="50"),
    Example("mmio_echo.asm", "Memory-mapped I/O: keyboard echo", keys="hi\n",
            output="hi\n"),
    Example("timer_interrupt.asm", "Interrupts: CP0 timer and exception handler",
            output="timer interrupts: 3"),
    Example("vectored_interrupts.asm", "Vectored interrupts and shadow register sets",
            output="timer ticks: 3"),
    Example("debug_monitor.asm", "EJTAG Debug Mode: a single-step monitor (SDBBP, DERET)",
            output="10 in 14 steps"),
    Example("float_circle.asm", "Floating point: area of a circle",
            output="area = 19.634954"),
    Example("r6_compact_branches.asm", "Release 6: compact branches (R6)",
            output="250"),
    Example("tlb_refill.asm", "TLB: a refill handler reading a page table",
            output="42 99"),
    Example("copy_on_write.asm", "TLB: copy-on-write with the TLB Modified exception",
            output="1 1 2 "),
    Example("mips64_factorial.asm", "MIPS64: 64-bit registers (factorial of 20)",
            output="2432902008176640000"),
]
