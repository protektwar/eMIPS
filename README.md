# eMIPS

**A MIPS32 emulator for learning computer architecture.**  Write or load
a program, run it on a model that behaves like a real MIPS32 processor,
and watch what happens inside: how each instruction is decoded, what the
ALU computes, how the pipeline stages overlap, where and why the
pipeline stalls, what the caches and the TLB do, how cores share memory.

![eMIPS: the editor, the 5-stage datapath and the registers](docs/guide/img/01-pipeline-mode.png)

The MIPS32 *Architecture for Programmers* manuals are the reference:
eMIPS behaves one to one like a MIPS, and where the manuals leave a
choice open, it says which one it made.

## What you can do with it

- **Write assembly and run it** — an editor with breakpoints, a two-pass
  assembler (GNU as / MARS syntax, pseudo-instructions), a disassembler,
  Intel HEX, ELF executables and C programs with source-level debugging;
  the MARS / SPIM system calls and a memory-mapped console.
- **See inside the processor** — the classic 5-stage datapath drawn live
  (control signals, forwarding, stalls, the ALU down to its gates), the
  pipeline chart, every instruction's bit fields, operands and addresses.
- **Every MIPS32 release, R1 to R6** — the integer instruction set, the
  IEEE 754 FPU, Release 6 compact branches; big and little endian; the
  compressed instruction sets MIPS16e and microMIPS (Releases 2 to 6), in
  the assembler, the pipelines, C and GDB.
- **MIPS64, first steps** — 64-bit registers and every doubleword
  instruction of Releases 1 to 6, `dli` / `dla`, the n64 register names,
  run instruction by instruction or on the 5-stage pipeline, in the 32-bit
  compatibility address space or with a TLB in the whole 64-bit one — xuseg, xsseg, xkphys,
  xkseg, the XTLB Refill; every MMU: the BAT, the FTLB, Segmentation
  Control and EVA, the page table walker (`--arch mips64 [--mmu ...]`);
  C for the n64 ABI and ELF64 executables (`cc --arch mips64`), GDB with
  64-bit registers; microMIPS64, MIPS16e64, the DSP and MT Modules on
  MIPS64; a 64-bit CPU on the MIPS Malta board (`--board malta --core 5kc`).
- **nanoMIPS, first steps** — the I7200's instruction set: 16-, 32- and
  48-bit instructions, Release 6 based, no delay slots, the nanoMIPS DSP
  and MT ASEs; an assembler and
  linker that give the bytes of the nanoMIPS toolchain's `as` and `ld`
  (link-time relaxation included), the views of every format, every MMU
  and the I7200's memory protection unit (`--mmu mpu:N`), caches and cores, instruction by instruction or on the 5-stage
  pipeline, from the command line and the debugger (`--arch nanomips
  [--nms]`); nanoMIPS ELF executables, C with the nanoMIPS toolchain's
  GCC (`cc --arch nanomips`) and GDB; the Malta board with a nanoMIPS boot
  monitor; in the window, with its datapath, panes and tools.
- **The DSP Module** — SIMD on bytes and halfwords, saturating and
  fixed-point arithmetic, four accumulators (Revisions 1 to 3), in
  MIPS32 and microMIPS, on both pipelines, from C and in GDB.
- **MSA, the MIPS SIMD Architecture** — 32 vector registers of 128 bits
  (16 bytes, 8 halfwords, 4 words or 2 doublewords at once): integer,
  saturating, fixed-point and IEEE floating-point vectors, MSACSR, the MSA
  Disabled exception, on MIPS32, MIPS64 and microMIPS, on the 5-stage
  pipeline, from C (GCC's auto-vectorizer) and in GDB.
- **The MT Module** — hardware threads: thread contexts and VPEs, FORK /
  YIELD, the policy managers, the ITC's gating storage, on the 34K's
  pipeline and on multi-core 1004K / interAptiv systems, on MIPS64 too;
  EJTAG breakpoints for one thread; a GDB thread per thread context.
- **The privileged architecture** — CP0, exceptions, interrupts
  (compatibility, vectored, EIC), shadow register sets, kernel /
  supervisor / user mode, the timer.
- **The memory system** — L1 / L2 / L3 caches, the write buffer, memory
  timing; the TLB and the other MMUs (BAT, VTLB + FTLB, segmentation
  control, the page table walker; on nanoMIPS the I7200's memory
  protection unit).
- **Real MIPS cores** — presets for the 4Kc, 24Kc / 24Kf, 34Kc / 34Kf,
  1004Kc / 1004Kf, interAptiv, P5600, M14K / M14Kc, microAptiv UC / UP,
  the MIPS64 5Kc / 5Kf, 20Kc, I6400 and P6600 and the nanoMIPS I7200:
  each core's release, compressed ISA, FPU, DSP Module, MT Module, MSA, MMU,
  caches and PRId as its manuals allow.
- **A real core's pipeline** — a model of the MIPS 24K's 8-stage pipeline
  with branch prediction, CP0 hazards and performance counters, and the
  34K's, which adds the thread contexts.
- **More than one core** — up to six MIPS32 or MIPS64 cores with coherent
  caches (MESI), memory ordering, the Cluster Power Controller and the
  Global Interrupt Controller; the I6400's CM3 with Release 6 virtual
  processors (DVP / EVP, VPs started by the CPC, the ITU's gating
  storage).
- **Debugging hardware** — EJTAG Debug Mode, hardware breakpoints, watch
  registers, a probe on the TAP, PC sampling and MIPS Trace.
- **Exercise checks** — what a program must do (its output, registers,
  memory, cycles, stalls), written in the program or in a teacher's file;
  Run → Check in the window or `python3 -m emips check` says which checks
  pass (with an exit status for scripts).

<p>
<img alt="Forwarding in the 5-stage datapath" src="docs/guide/img/03-datapath-forwarding.png" width="49%">
<img alt="A branch misprediction in the MIPS 24K pipeline" src="docs/guide/img/03-datapath-24k.png" width="49%">
</p>

## Running it

You need **Python 3.10 or newer** (on Windows type `py` where this says
`python3`).  The simulator needs nothing else; the window needs **PyQt6**
(or PySide6 — either works).

```bash
git clone https://github.com/protektwar/eMIPS.git
cd eMIPS
```

### Straight from the folder — nothing to install

```bash
python3 -m emips run docs/guide/labs/trace_me.asm    # the terminal works right away
```

For the window, install Qt for Python from your system's packages:

| System | Command |
|---|---|
| Debian / Ubuntu / Mint | `sudo apt install python3-pyqt6` |
| Fedora | `sudo dnf install python3-pyqt6` |
| Arch / Manjaro | `sudo pacman -S python-pyqt6` |

```bash
python3 -m emips gui
```

Run the `python3 -m emips ...` commands from inside the `eMIPS` folder.

### In a virtual environment — any system, with pip

Most Linux distributions refuse `pip install` into the system's Python
(*error: externally-managed-environment*); a virtual environment is the
place for pip — and the way on macOS and Windows:

```bash
python3 -m venv .venv                 # Debian / Ubuntu: sudo apt install python3-venv first
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install ".[gui]"                  # eMIPS + PyQt6
emips-gui                             # the window; "emips run prog.asm" in the terminal
```

Activate it again (`source .venv/bin/activate`) in every new terminal.
Already have PyQt6 from your system?  `python3 -m venv
--system-site-packages .venv` lets the environment use it.

### As an application — pipx

```bash
pipx install ".[gui]"                 # Debian / Ubuntu: sudo apt install pipx
emips-gui
```

### First steps

In the window: **File → Examples**, **Step** (F10), **Run** (F5) — and
**Help → User Guide** (F1).  In the terminal:

```bash
python3 -m emips run prog.asm            # run it (system calls + console), show the registers
python3 -m emips asm prog.asm            # the listing: address, word, disassembly, source
python3 -m emips pipe prog.asm --events  # the 5-stage pipeline chart, stalls, forwarding
python3 -m emips debug prog.asm          # the debugger (type help)
python3 -m emips gdb prog.asm            # let GDB debug it: gdb-multiarch -ex 'target remote :1234'
python3 -m emips check prog.asm          # run its '# check:' lines: pass / fail
python3 -m emips --help                  # everything else
```

(Installed with pip or pipx, `emips` is the same command: `emips run
prog.asm`.)  Options shared by the commands include `--release R1..R6`,
`--arch mips64`, `--little`, `--model 5-stage|24k`, `--caches`, `--mmu tlb`,
`--cores N`.

### From Python

```python
from emips import Machine, assemble

prog = assemble("""
        .data
nums:   .word 3, 4, 5
        .text
main:   la    $a0, nums
        lw    $t0, 4($a0)
        addiu $t0, $t0, 10
""")
m = Machine()
prog.load(m)
for _ in range(4):
    m.step()
print(m.read_reg(8))         # 14
```

## The user guide

[**docs/guide/index.html**](docs/guide/index.html) — open it in a web
browser, or in eMIPS with **Help → User Guide** (F1).  Six phases: getting
started, writing programs, the window in depth, the command line, the
computer architecture behind every number on the screen (with 72 labs
you check with eMIPS itself), and a reference.

## Author

**Alexandru Herlas** — [github.com/protektwar](https://github.com/protektwar)
(45262264+protektwar@users.noreply.github.com)

eMIPS began in 2021 as a Ruby library and a Discord bot; in 2026 it was
rewritten in Python and grew into this teaching emulator.  It was
developed with the help of Claude (Anthropic), used as an AI programming
assistant.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).  Copyright © 2021–2026
Alexandru Herlas.
