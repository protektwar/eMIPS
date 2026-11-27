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
    Example("rv_sum_array.asm", "RISC-V: the sum of an array (RV32)", output="sum = 31"),
    Example("rv_factorial.asm", "RISC-V: a recursive function and the stack (RV64)",
            output="-2102132736\n120"),
    Example("rv_trap.asm", "RISC-V: a trap handler, an exception and an interrupt",
            output="2"),
    Example("bitmap_gradient.asm", "The bitmap display: a gradient (MARS's Bitmap Display)"),
    Example("read_file.asm", "Files: read a file given with the program",
            output="Hello from a file.\nThis text was given to the program with --file;\n"
                   "it reaches no other file on your computer.\n"),
    Example("play_scale.asm", "Sound: a scale (MIDI notes, the Sound pane)"),
    Example("dialogs.asm", "Dialogs: ask and answer (MARS's 50-59)",
            input=("19", "Ada", "y"), output="Hello, Ada\n"),
]


#: "MIPS by Example in eMIPS" (PLAN M28): each episode's programs in
#: recipes/ — the assembly, its C version, a checks file, a RISC-V twin for
#: some (File > Examples > MIPS by Example in eMIPS; tests/test_recipes.py)
RECIPE_DIR = DIR / "recipes"


@dataclass(frozen=True)
class Recipe:
    """An episode: its number, title and file stem (``ep01_strings``)."""
    number: int
    title: str
    stem: str

    def files(self) -> list[tuple[str, Path]]:
        """(kind, path) of the episode's programs that exist: "Assembly",
        "C", "RISC-V", its challenge's solution ("Challenge solved",
        "Challenge solved, C") and the files they ``.include`` ("Library",
        "Challenge solved, library": no checks of their own)."""
        out = []
        for kind, name in (("Assembly", f"{self.stem}.asm"), ("C", f"{self.stem}.c"),
                           ("RISC-V", f"{self.stem}_rv.asm"),
                           ("Challenge solved", f"{self.stem}_challenge.asm"),
                           ("Challenge solved, C", f"{self.stem}_challenge.c"),
                           ("Library", f"{self.stem}_lib.asm"),
                           ("Challenge solved, library", f"{self.stem}_challenge_lib.asm")):
            if (RECIPE_DIR / name).is_file():
                out.append((kind, RECIPE_DIR / name))
        return out


RECIPES = [
    Recipe(1, "Functions done right: the stack frame", "ep01_frames"),
    Recipe(2, "Arguments and recursion", "ep02_recursion"),
    Recipe(3, "Strings in memory: length and the byte order", "ep03_strings"),
    Recipe(4, "Comparing two strings", "ep04_strcmp"),
    Recipe(5, "Copy, concatenate, and the overflow", "ep05_strcat"),
    Recipe(6, "Reading a line; upper and lower case", "ep06_case"),
    Recipe(7, "Reverse in place and palindromes", "ep07_reverse"),
    Recipe(8, "Counting letters: the most frequent character", "ep08_letters"),
    Recipe(9, "Finding a substring", "ep09_find"),
    Recipe(10, "Numbers to text: decimal, binary, hex", "ep10_itoa"),
    Recipe(11, "Text to numbers: parse a line, min and max", "ep11_parse"),
    Recipe(12, "A tiny sprintf", "ep12_sprintf"),
    Recipe(13, "Splitting a line into words", "ep13_split"),
    Recipe(14, "Sorting an array of strings", "ep14_sort"),
    Recipe(15, "First decisions: the largest of three, even and odd", "ep15_decide"),
    Recipe(16, "Arithmetic: the four operations, shifts, area and perimeter", "ep16_arith"),
    Recipe(17, "Division without div; counting 1 bits three ways", "ep17_bits"),
    Recipe(18, "GCD and the Sieve of Eratosthenes", "ep18_primes"),
    Recipe(19, "Multiples and statistics", "ep19_stats"),
    Recipe(20, "Floating point: a circle's area with doubles", "ep20_circle"),
    Recipe(21, "Arrays: sum, search, reverse, two score arrays into grades", "ep21_arrays"),
    Recipe(22, "Sorting numbers: bubble, insertion, into a second array", "ep22_sort"),
    Recipe(23, "Binary search, loop and recursive", "ep23_bsearch"),
    Recipe(24, "Matrices: add and multiply", "ep24_matrix"),
    Recipe(25, "A linked list with sbrk", "ep25_list"),
    Recipe(26, "Balanced brackets with the stack", "ep26_brackets"),
    Recipe(27, "A calculator: jump tables and function pointers", "ep27_calc"),
    Recipe(28, "Recursion: from n to 1 and back, all binary patterns", "ep28_recursion"),
    Recipe(29, "Towers of Hanoi", "ep29_hanoi"),
    Recipe(30, "A simulation with nested calls", "ep30_savings"),
    Recipe(31, "A menu-driven program with dialogs", "ep31_menu"),
    Recipe(32, "A library of your own", "ep32_arrays"),
    Recipe(33, "A guessing game", "ep33_guess"),
    Recipe(34, "Drawing a house on the bitmap display", "ep34_house"),
    Recipe(35, "Animation without flicker", "ep35_bounce"),
    Recipe(36, "A maze game", "ep36_maze"),
    Recipe(37, "Playing a tune", "ep37_tune"),
    Recipe(38, "Your own exception handler", "ep38_exceptions"),
    Recipe(39, "An interrupt-driven keyboard", "ep39_keyboard"),
    Recipe(40, "A timer interrupt: a clock", "ep40_clock"),
    Recipe(41, "Quicksort", "ep41_quicksort"),
    Recipe(42, "Merge sort", "ep42_mergesort"),
    Recipe(43, "A hash table: counting words", "ep43_hashtable"),
    Recipe(44, "From text to a float", "ep44_atof"),
    Recipe(45, "A tiny text adventure", "ep45_adventure"),
    Recipe(46, "Vectors: MSA from assembly and C", "ep46_vectors"),
    Recipe(47, "Two cores and a lock", "ep47_lock"),
    Recipe(48, "A whole program in RISC-V: a calculator", "ep48_calc"),
]
