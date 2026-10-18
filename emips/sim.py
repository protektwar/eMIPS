"""Instruction-level MIPS32 simulator core (M1 slice 1 + M2 semantics).

Steps the decoded ISA (:mod:`emips.isa`, the M2-corrected 313-row table)
over byte-addressed memory (:mod:`emips.memory`), one instruction per
:meth:`Machine.step`.  This is the **architectural reference model**: the
M5 pipeline must reach the same registers, memory and CP0 state.

Semantics follow the *MIPS32 Architecture For Programmers* manuals
(Vol. II-A instruction pages, Vol. III exceptions/CP0); any difference is
listed in :data:`DEVIATIONS`.  Highlights:

* **Dispatch by row name.**  ``ISA.decode(word)`` identifies the row,
  :func:`classify` buckets it (EXEC / HINT / ILLEGAL / DEFER) and a
  per-name executor runs it.  Execution never re-reads ``funct`` numbers.
* **Branch delay slots** after every branch and jump (pre-Release 6),
  taken or not (a fault in the slot of a not-taken branch also reports
  ``Cause.BD``).  A branch-likely that is not taken *nullifies* its delay
  slot.  Branch
  targets are ``PC + 4 + sign_extend(offset << 2)``; jump targets take
  ``PC + 4``'s upper 4 bits.  Linking branches/jumps write ``PC + 8``.
* **Precise exceptions, handled by CP0** (default ``exceptions="cp0"``):
  ``EPC`` (or the branch's address with ``Cause.BD = 1`` when the faulting
  instruction is in a delay slot), ``Cause.ExcCode``/``CE``, ``BadVAddr``
  for address errors, ``Status.EXL = 1``, then the PC jumps to the general
  exception vector (``0xBFC00380`` while ``Status.BEV = 1``, else
  ``EBase + 0x180``).  The faulting instruction has no effect.  ``ERET``
  returns.  ``exceptions="raise"`` (debug mode) raises a
  :class:`~emips.exceptions.MipsException` instead and changes no state.
* **Reset state** per Vol. III: ``Status.BEV = Status.ERL = 1`` (kernel
  mode), ``CU0 = CU1 = 0`` — so FPU instructions raise Coprocessor
  Unusable until software sets ``Status.CU1``.
* **Interrupts** (Vol. III, interrupt compatibility mode): software
  ``Cause.IP0..1``, hardware lines HW0..HW5 (:meth:`Machine.assert_irq`)
  on ``IP2..IP7``, and the timer (``Count == Compare`` sets ``Cause.TI``
  on IP7).  Checked between instructions; taken as an ExcCode 0
  exception with ``EPC`` = the next instruction (its branch when it is a
  delay slot), at offset 0x200 when ``Cause.IV = 1``.  ``WAIT`` sleeps
  until one is taken.  Time comes from :meth:`Machine.advance` — one
  cycle per step here, ``Count`` every second cycle.
* **Memory** is byte-addressed, **big-endian** by default
  (``endian="little"`` for e.g. PIC32).  Unwritten data bytes read 0;
  fetching an unwritten code byte is a simulator halt
  (:class:`~emips.exceptions.FetchError`), not an architectural exception.
  Every address is **virtual** and is translated to a physical address
  by the MMU (:mod:`emips.mmu`; the Fixed Mapping MMU by default, the
  TLB of :mod:`emips.tlb` with ``mmu=TLBMMU(N)``) after the segment
  access rights are checked; :attr:`Machine.mem` holds the physical
  memory, memory-mapped devices sit at physical addresses.

``emips`` core stays zero-runtime-dependency (stdlib only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from . import cp0 as C
from . import ejtag as EJ
from . import pdtrace as PDT
from . import fpu as F
from . import perf
from .exceptions import (
    BusError,
    CP0RegisterError,
    FetchError,
    IllegalInstructionError,
    MipsException,
    SimException,
    SyscallException,
    TrapException,
    UnimplementedError,
    UnmappedAddress,
    WatchpointHit,
)
from .cache import CACHED_CCA, CACHED_CCA_CPS, CCA_24K, CCA_UCA, COHERENT_CCA, CacheError, Caches
from .isa import ISA, Decode, ISADecodeError
from .memory import PMemory
from .mmu import CCA_UNCACHED, SUPPORTED_CCA, Translation, default_mmu

__all__ = [
    "BUCKET_DEFER",
    "BUCKET_EXEC",
    "BUCKET_HINT",
    "BUCKET_ILLEGAL",
    "CP0RegisterError",
    "DEFER_NAMES",
    "DEVIATIONS",
    "EXEC_NAMES",
    "FPU_FAMILIES",
    "HINT_NAMES",
    "Machine",
    "PRIVILEGED_NAMES",
    "classify",
]

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

#: Documented choices where the manual leaves behavior UNPREDICTABLE /
#: implementation-dependent, or where the simulator is knowingly partial.
DEVIATIONS: dict[str, str] = {
    "DIV/DIVU by zero": "UNPREDICTABLE in the manual; HI = LO = 0xFFFFFFFF "
                        "here (no exception, as on real cores).",
    "branch in a delay slot": "UNPREDICTABLE; here the inner taken branch "
                              "replaces the outer target.",
    "INS/EXT with msb < lsb or pos + size > 32": "UNPREDICTABLE; computed "
                                                 "with the bit fields masked.",
    "SC address": "the manual only requires LLbit; SC here also requires the "
                  "address of the preceding LL.",
    "MUL HI/LO": "UNPREDICTABLE after MUL; left unchanged.",
    "Count": "advances every Machine.ccres cycles (CCRes, 2 by default, 1 "
             "selectable); the instruction-level Machine counts one cycle "
             "per instruction, the 5-stage pipeline the real cycles.",
    "COP2": "no COP2 is present, so Status.CU2 reads 0 and every COP2 "
            "instruction raises Coprocessor Unusable (CE = 2). (QEMU raises "
            "Reserved Instruction instead.)",
    "paired single": "the FPU has no PS format (FIR.PS = 0): with Status.CU1 "
                     "= 1, PS instructions raise Reserved Instruction.",
    "FPU": "full IEEE in hardware: no Unimplemented Operation (FCSR.E) for "
           "denormals; FS = 1 flushes tiny results to zero; trapped "
           "Underflow follows the untrapped rule (tiny and inexact); "
           "RECIP/RSQRT are 1/x and 1/sqrt(x) correctly rounded per step.",
    "FR = 0 odd register": "a 64-bit (D/L) operand in an odd FPR with "
                           "Status.FR = 0 is UNPREDICTABLE; Reserved "
                           "Instruction here (QEMU does so for arithmetic).",
    "FR = 0 L format": "L-format operations (CVT.L, ROUND.L, ..., CVT.S/D.L) "
                       "and LUXC1/SUXC1 with Status.FR = 0 are UNPREDICTABLE; "
                       "Reserved Instruction here (as QEMU).",
    "CVT.S.D NaN": "a quiet NaN whose payload narrows to zero becomes the "
                   "default NaN (Linux sp_fdp.c); QEMU returns +inf.",
    "FCCR/FEXR/FENR": "CTC1 to them with non-zero reserved bits is "
                      "UNPREDICTABLE; the defined bits are written here "
                      "(QEMU ignores such a write).",
    "ABS/NEG NaN": "legacy ABS/NEG are arithmetic (manual; Linux "
                   "232b6ec5df87); this QEMU version treats them as sign-bit "
                   "operations.",
    "R6 divide by zero": "DIV/MOD/DIVU/MODU by zero are UNPREDICTABLE; the "
                         "result is 0 here (as QEMU).",
    "R6 misaligned access": "ordinary loads/stores to misaligned addresses "
                            "are handled in hardware (the manual allows "
                            "hardware or an Address Error); LL/SC must be "
                            "aligned.",
    "R6 FCCR": "Release 6 has no FPU condition codes: CTC1 to FCCR is "
               "ignored (as QEMU), FCSR bits 31..25/23 read 0.",
    "SYNCI_Step": "RDHWR 1 returns the smallest L1 line size (32 on the "
                  "24K), 0 without an instruction cache.",
    "R6 NAL": "NAL decodes with any offset (QEMU); binutils only accepts 0.",
    "FPE state": "a trapping FP operation leaves FCSR.Cause set (the handler "
                 "reads it) and writes no destination; a CTC1 that enables a "
                 "pending Cause traps after its write (manual).",
    "ELF programs": "emips.elf / emips.cc (M13): executables (ET_EXEC) only, "
                    "ELF32 either byte order; EF_MIPS_ARCH MIPS32 Release 2 "
                    "binaries run as R5 (every later pre-R6 release keeps their "
                    "instructions); position-independent (abicalls) code is "
                    "loaded with a warning (nothing sets $gp / $t9 for it); "
                    "PT_LOAD segments go where the MMU maps their virtual "
                    "addresses (as assembled programs), .MIPS.abiflags and the "
                    "ELF's FP ABI are not checked (the FPU follows the release "
                    "and Status.FR); no C library, no dynamic linking.",
    "JALX": "no MIPS16e / microMIPS (Config1.CA = 0, Config3.ISA = 0): "
            "Reserved Instruction.",
    "EJTAG": "Debug Mode as the 24K's EJTAG 3.20 (MD00343 ch. 11, MD00047 "
             "ch. 2 / 3): Debug.EJTAGver = 3, NoDCR = 0 (dseg present), "
             "CountDM R/W (reset 1), IBusEP / MCheckP read 0 (precise on the "
             "24K), no DebugVectorAddr / Debug2 (EJTAG 4 / 5). The TAP "
             "(emips.tap): IDCODE = 1 (the core's Version / PartNumber / "
             "ManufID inputs tied to 0), IMPCODE MIPS16 = 0 (no MIPS16e in "
             "eMIPS), the TCB's instructions 0x10-0x13 (MIPS Trace); PrRst / "
             "PerRst drive pins nothing is connected to (the bits read back); "
             "a dmseg access completes within its instruction / cycle (a "
             "real probe takes some hundred TCKs while the core waits) and "
             "an access no probe finishes is a simulator halt "
             "(ProbeWaitError). One probe serves every core's TAP; it "
             "connects at reset (a fresh TAP: power-up) and, when "
             "profiling, reads PCsample after every step. PC sampling as the "
             "24K (11.6): always on, DCR.PCSe reads 0, the PC of the "
             "instruction completing in the cycle the counter overflows (or "
             "the next one); in WAIT the WAIT's. Reading / writing "
             "Debug, DEPC, DESAVE from Non-Debug Mode is UNPREDICTABLE / "
             "UNDEFINED: eMIPS gives the real register (as cores allow 'for "
             "test purposes'). DERET outside Debug Mode (UNDEFINED) is "
             "Reserved Instruction; in a delay slot (UNDEFINED) it returns. "
             "A drseg access other than a word (UNDEFINED) reads / writes "
             "those bytes of the register. Random is not frozen in Debug "
             "Mode (2.2.3.2 leaves it to the implementation); the "
             "performance counters count in Debug Mode by Status as "
             "elsewhere (the 24K manual says nothing). Halt = Doze = 1 when "
             "a debug exception wakes a WAIT (the 24K stops its clock), "
             "Doze also with Status.RP.",
    "hardware breakpoints": "the 24K's 4 instruction / 2 data breakpoints "
                            "(MD00343 11.2): ASIDsup only with a TLB MMU (TLB / "
                            "FTLB; the BAT and the fixed mapping have no ASID); "
                            "no MIPS16e, so IBA / IBM bit 0 (ISA / ISAM) compare "
                            "with ISAMode 0. Value compares are imprecise, as "
                            "the 24K's (DDBLImpr / DDBSImpr before the next "
                            "instruction); a store that matches a precise "
                            "breakpoint leaves the value-compare BS bits "
                            "unchanged (MD00047 Table 5.8's option; the 24K "
                            "sets them, Table 11.6, but the store data is not "
                            "known there in eMIPS); an SC whose LLbit is 0 is "
                            "still compared (implementation-dependent). drseg "
                            "writes take effect at once (no SYNC spacing); an "
                            "R6 misaligned access spanning two words uses the "
                            "lanes of Table 5.5 and compares the value it "
                            "moved (UNPREDICTABLE).",
    "Malta board": "the MIPS Malta with a CoreLV card (emips.malta, "
                   "emips.gt64120; MD00048, MD00007, the GT-64120 datasheet rev. "
                   "0.2): the real Core24K card's MSC01 controller was never "
                   "documented publicly, so the 24K sits on the CoreLV's "
                   "GT-64120, as in QEMU. Bus errors follow the datasheet (4.5) "
                   "and the 24K (5.8.13): a read nothing decodes is a bus error "
                   "(IBE precise, DBE imprecise: the load's register gets 0), a "
                   "write only the GT-64120 interrupt (QEMU raises DBE on "
                   "stores); a write-allocate refill that fails is a DBE, the "
                   "store lost. The GT-64120's CPU-side remap registers, timers, "
                   "DMA engines and PCI_1 are stored but not used (the PCI target "
                   "BARs, bank sizes, remaps and enables are); its internal "
                   "space answers 4 KB; SDRAM decoded beyond the installed size "
                   "reads 0; the registers are always little-endian (the CoreLV "
                   "strapping; CPU Configuration bit 12 and PCI_0 Command bit 0 "
                   "choose only the PCI byte lanes). The boot "
                   "firmware programs the decoders as YAMON's map says (MD00008 "
                   "Table 3.1); BootCS covers 0x1E000000-0x1FFFFFFF so the "
                   "monitor flash answers (the CoreLV manual's 0xF0 would leave "
                   "it out; YAMON's own values are unpublished). The CBUS "
                   "answers 0 where no Malta device sits; the monitor flash is "
                   "read-only (no CFI commands); SOFTRES ends the program (Run "
                   "→ Reset starts it again); the MARS MMIO console is not on "
                   "the board (the system calls stay). PCI memory has no device "
                   "(master abort: all ones). The south bridge (emips.piix4, "
                   "superio, uart16550): the core runs at the board's clock "
                   "(default 10 MHz, so serial characters and timer ticks come "
                   "within thousands of instructions; malta:RAM:SW:MHZ changes "
                   "it); the RTC powers up at 2026-01-01 00:00:00 running in BCD "
                   "24-hour mode (deterministic replay), its writes take effect at "
                   "once, DSE is ignored; the 8259s are masked until their ICW "
                   "sequence (their reset state is undefined); the PIIX4's USB "
                   "and power-management functions have their PCI headers only, "
                   "the Super I/O's floppy, parallel and keyboard "
                   "devices their configuration registers only; ISA ports nothing "
                   "answers read 0xFF; the serial terminal asserts CTS / DSR / DCD "
                   "and never sends a parity / framing error or a break, RTS / CTS "
                   "flow control is not enforced; 1.5 stop bits count as 1. The "
                   "firmware sets the serial ports as YAMON does (38400 8N1, "
                   "FIFOs on, OUT2 off) and routes PIRQA-D to IRQ 10, 10, 11, 11.",
    "DMA": "the PIIX4's bus master IDE (emips.busmaster) through the "
           "GT-64120's PCI target (MaltaBoard.dma_access): below every cache "
           "level and the write buffer — not coherent, as on the Malta (Linux "
           "uses software coherency there). eMIPS choices: a transfer moves up "
           "to a sector per step (the PIIX4's 64-byte line buffer and PCI "
           "bursts are not modelled; the destination is checked before data "
           "leaves the disk), at the rate of the disk's DMA mode (multiword "
           "480 / 150 / 120 ns per word, Ultra DMA 16.7 / 25 / 33.3 MB/s; Ultra "
           "DMA only with the drive's UDMACTL bit, else multiword 2), the "
           "power-on mode multiword DMA 2; the alignment rules of PRDs and "
           "regions (dword, no 64 KB / 4 KB crossing) are not checked (bit 0 "
           "of an address and a count is ignored); a direction that does not "
           "match the disk's command moves nothing; a target retry is never "
           "given up (the PIIX4 retries forever). The firmware puts the bus "
           "master block at I/O 1000h and opens the GT-64120's BAR0 over the "
           "whole SDRAM at PCI 0 (Memory / Master Enable; SCS[3:2] and the CS "
           "BARs disabled), so a PCI address is the physical address. A "
           "multi-core Malta (cores > 1) puts the CPS — the CM, CPC and GIC at "
           "Linux's Malta addresses, decoded before the GT-64120 — on the "
           "CoreLV's GT-64120, as QEMU does (the real multi-core Malta cards, "
           "CoreFPGA with the ROC-it controller, were never documented "
           "publicly); the 8259s' INTR is GIC shared interrupt 3 (Linux "
           "malta.dts, QEMU), routed by the firmware to core 0's pin 0; the "
           "board's other lines go to core 0 directly, ORed with the GIC's pins "
           "(not seen in EIC mode). The IOCU option (GCR_CONFIG.NUMIOCU = 1) "
           "puts an I/O Coherence Unit behind the GT-64120's PCI target — a "
           "pairing no real card had (the IOCU came with ROC-it): DMA to the "
           "SDRAM becomes coherent (reads as CohReadDiscard: a Modified copy "
           "supplies its line, nothing changes state; writes invalidate every "
           "copy, a Modified one written back first), snooping the cores whose "
           "COH_DOMAIN_EN has bit 4; the cores' write buffers are emptied "
           "before each IOCU request; DMA with reversed byte lanes does not use "
           "it; CPU accesses to PCI still go to the GT-64120 (no address region "
           "targets the IOCU's MMIO port).",
    "Boot monitor": "the Malta's eMIPS monitor (emips/firmware/monitor.asm, "
                    "emips.monitor) gives YAMON's application interface (MD00008 "
                    "ch. 5: the context of Table 5.1, the function table at "
                    "0x1fc00500 with yamon_api.h's offsets) — it is not YAMON: no "
                    "shell, no load / go commands (eMIPS's loader plays them and "
                    "writes the application descriptor at 0x8000c000), no "
                    "environment in flash, no SPD SDRAM detection (the image is "
                    "built for the board). Its functions use the caller's stack "
                    "(YAMON switches to its own); print / print_count / getchar "
                    "use tty0 whatever the port, no CR is added; syscon_read "
                    "knows ids 34-36 only (the other ids of YAMON's syscon_api.h "
                    "are unpublished: -1); register_esr's retfunc ends the "
                    "application; an interrupt or exception nobody registered is "
                    "masked / ends the application with a message (YAMON: its "
                    "register dump); exit waits for tty0 to send its FIFO and "
                    "ends the run with the eMIPS exit service (YAMON: its "
                    "prompt). The vectors go to RAM only where the application "
                    "has no code (0x80000180 / 0x80000200, the TLB refill at "
                    "0x80000000); $gp is set as for eMIPS programs (YAMON: not). "
                    "The kernel boot sets the same state in Python (the tests "
                    "compare them); --boot reset runs the code.",
    "IDE disks": "ATA/ATAPI-4 devices (T13 1153D rev. 18) on the PIIX4's IDE "
                 "channels (emips.ata). eMIPS choices where ATA-4 leaves it to "
                 "the device: the busy times (ata.TIMING_US: microseconds, not "
                 "a real disk's milliseconds and seconds — command 5, first "
                 "sector 100, further sectors 10, a written block 100, reset "
                 "400, power-on 2000, spin-up from Standby 1000); the address "
                 "registers end at the last sector transferred and Sector Count "
                 "counts the sectors left (ATA-4: not applicable); a command "
                 "written while DRQ is set abandons the transfer (7.4.3: "
                 "indeterminate); both devices latch the DEV bit even while BSY "
                 "(the selection never differs between them); while BSY the "
                 "registers read as stored; a Data read without DRQ reads "
                 "FFFFh; an undriven cable reads 7Fh (DD7 pulled down, the "
                 "other lines high); Status reads 50h (DRDY, DSC) when ready. "
                 "No write cache (SET FEATURES 02h / 82h and 05h / 85h abort; "
                 "FLUSH CACHE succeeds at once); the default CHS translation "
                 "is 16 heads x 63 sectors (fewer for a small disk); multiple "
                 "mode is off after power-on; no SMART, security, host "
                 "protected area, removable media or queued commands. The "
                 "PIIX4's IDE timing registers are kept, not used (the disk's "
                 "times are the times). Every run starts from the disk image "
                 "(what a program writes stays in the run until saved: Reset "
                 "and Step Back replay the same disk). The firmware enables "
                 "the IDE function's I/O space and bus master, decodes both "
                 "channels with compatible timing and waits for the disks.",
    "MIPS Trace": "PDtrace 4.x and a TCB as the 24K's options (MD00439 6.16, "
                  "MD00343 11.7-11.13; emips.pdtrace). eMIPS choices: the "
                  "PDtrace FIFO holds 8 records (unpublished); the TCB has 4 "
                  "triggers and 8 KB of on-chip memory (TCBCONFIG.SZ 5) and "
                  "no off-chip interface (PiN 0, no TR_DATA / PIB, CR "
                  "ignored); records leave in order one a cycle and data in "
                  "order (TF4 DataOrder 0, PendL 0); PCs and addresses are "
                  "never shortened by leading-bit chopping (only the delta / "
                  "full choice of 6.1 / 6.2); a TPC is a delta when it fits "
                  "in 16 bits; IO stalls the instruction-level Machine a cycle "
                  "at a time while the pipelines let the FIFO grow (no "
                  "back-pressure into a pipeline stage); trace triggers from "
                  "breakpoints are event-based (TraceIBPC / TraceDBPC 001 "
                  "start, 000 stop, kept until the other) with the 24K's "
                  "TriggerOn latch, and a matching instruction is traced "
                  "with its TF6 1001 after it; Im (TLSIF) is the "
                  "instruction's fetch missing the I-cache, LSm a TD of an "
                  "access that missed the D-cache; an MTC0 to the trace "
                  "registers acts from the next instruction. Triggers: DM "
                  "entry and the probe's TR_TRIGIN (no chip-level trigger "
                  "inputs); an 'about' trigger keeps 512 more trace words. "
                  "Views read the trace memory directly; the decoder knows "
                  "an exception at a place the program predicts by its TPC "
                  "being an exception vector sent in its own cycle "
                  "(regeneration software's rule, not the hardware's).",
    "watch": "the 24K's 4 WatchLo / WatchHi pairs (MD00343 6.2.26 / 6.2.27): "
             "0 / 1 instruction fetches, 2 / 3 loads / stores; a match with "
             "EXL / ERL set sets Cause.WP (the Deferred Watch is taken before "
             "the next instruction once EXL = ERL = 0, Table 5.1 order); "
             "never in Debug Mode; PREF, CACHE and SYNCI do not trigger "
             "(implementation-dependent); an SC does even when it fails "
             "(Vol. III 6.2.25). The 24K's fetch hazard of 10 instructions "
             "after an MTC0 WatchLo / WatchHi (Table 2.7) is modelled in the "
             "24K pipeline only.",
    "interrupts": "compatibility mode and, from Release 2, vectored "
                  "interrupts (Config3.VInt = 1 as on the 24K; Vol. III "
                  "6.1.1.2: IntCtl.VS spacing, IP7..IP0 priority, vector "
                  "0x200 + n x spacing); EIC mode (6.1.1.3) with an External "
                  "Interrupt Controller: only the GIC of a CPS is one "
                  "(GIC_VPEi_CTL.EIC_MODE drives Config3.VEIC), EIC option 1 "
                  "as the 24K (the vector is RIPL; MD00343 5.3.1.3); an "
                  "EICSS above HSS (UNDEFINED) selects set 0. Cause.RIPL "
                  "keeps the last RIPL taken while in EIC mode; outside it "
                  "Cause.IP7..2 follow the six lines again. A reserved "
                  "IntCtl.VS value (UNDEFINED) is not written. The HW lines "
                  "are driven by the MMIO console (keyboard IP3, display "
                  "IP2), the GIC (a CPS) and Machine.assert_irq.",
    "shadow register sets": "1-16 GPR sets (Machine(shadow_sets=N), Options "
                            "shadow_sets, default 4; the 24K builds 1, 2 or 4; "
                            "Release 1: none). Vol. III 6.2.3 / 9.31: on an "
                            "exception with EXL = 0 and BEV = 0, PSS <- CSS and "
                            "CSS <- SRSMap[vector] (a vectored interrupt) or "
                            "ESS; ERET with ERL = 0 and BEV = 0: CSS <- PSS. "
                            "A set above HSS written to ESS / PSS / SRSMap "
                            "(UNDEFINED) leaves that field unchanged; SRSMap "
                            "without shadow sets (UNPREDICTABLE) reads 0. QEMU "
                            "does not switch CSS (its ERET keeps the set) and "
                            "Unicorn delivers no interrupts, so this is pinned "
                            "to the manual (tests/test_vectored.py).",
    "WAIT": "the core sleeps until an interrupt is *pending and enabled* "
            "(IM & IP, IE=1, EXL=ERL=0); masked interrupts do not wake it.",
    "Config.K0/KU/K23": "UNDEFINED after reset in the manual; 2 (uncached) "
                        "here.  The boot firmware (Program.load(boot=True)) "
                        "sets K0 = KU = 3 (cacheable), K23 = 2 (kseg2/kseg3 "
                        "uncached: the MMIO console lives in kseg3).",
    "physical memory": "every physical address is RAM: never-written bytes "
                       "read 0, a fetch from one is a simulator halt "
                       "(FetchError), no Bus Error is raised.",
    "caches": "none by default (Config1 IL = DL = 0, Config2 SL = 0: every "
              "access goes to memory).  Machine(caches=Caches(...)) adds L1 "
              "caches as the 24K's (MD00343 ch. 8): VIPT, LRU, write-back / "
              "write-through, line locking, optional parity; any Config1 "
              "geometry with ways of 1 KB or more (TagLo holds PA[31:10]); "
              "optional L2 / L3 and write buffer (see their entries).",
    "reserved cacheability attributes": "1, 4, 5, 6 are reserved on the 24K; "
                                        "here they bypass the caches like 2. "
                                        "7 (uncached accelerated) is uncached; "
                                        "with the write buffer its word / "
                                        "doubleword stores are gathered.",
    "L2 / L3 caches": "as the 24K's SOC-it L2 (MD00355 3.4.1; its own manual "
                      "MD00525 is not public): physically indexed and tagged, "
                      "write-back without write allocate, every cacheable refill "
                      "read-allocates (the I-side as write-back, MD00343 8.5.2), "
                      "uncached / UCA traffic bypasses it; neither inclusive nor "
                      "exclusive (the 24K has no snoop port to invalidate L1 "
                      "lines).  Any Config2 shape; true LRU up to 4 ways, a 7-bit "
                      "tree pseudo-LRU above.  eMIPS's own layouts: L23TagLo = "
                      "PA[31:IndexBit] | V 7 | D 6 | L 5 | P 0; WS test mode "
                      "LRU 14:8, dirty 23:16, dirty parity 31:24.  The 24K has no "
                      "L3: an L3 follows the L2's rules (CACHE target T).  "
                      "Config2.L2B takes effect at once.",
    "L2 / L3 ECC": "SEC-DED, 8 check bits per doubleword (Hamming + overall "
                   "parity), a parity bit per tag and per dirty bit (detect "
                   "only); checked while ErrCtl.PE != ErrCtl.L2P.  A 1-bit data "
                   "error is corrected on the fly and counted, the array keeps "
                   "it.  Index Store ECC (ErrCtl.WST) writes L23DataLo[7:0] "
                   "(MD00343 says DDataLo); the check bits cannot be read back.  "
                   "CacheErr.EM is 0.",
    "L2 / L3 errors": "imprecise (MD00343 5.8.12): an uncorrectable error found "
                      "by a data access, a write-back or a CACHE operation lets "
                      "the instruction complete (a load writes the bad data) and "
                      "is taken before the next one (ErrorEPC = that one); one "
                      "found by an instruction refill is taken at that fetch.  A "
                      "second error before the exception sets CacheErr.EF.",
    "write buffer": "Caches(write_buffer=True), as the 24K's (MD00343 8.4): "
                    "one write-through merge entry (Config.MM) and one uncached "
                    "accelerated gather entry at a time; evictions and uncached "
                    "stores, \"flushed immediately\", reach the next level at "
                    "once (their cycles and the 4-entry limit come with the "
                    "memory timing, M9 phase 4).  UNPREDICTABLE UCA cases: a "
                    "doubleword into a word position sends the gathered block "
                    "and goes on its own; sub-word, SWL / SWR and SC stores go "
                    "on their own and leave the block alone.",
    "SYNC": "every stype empties the write buffer's merge / gather entries "
            "(24K: \"the BIU will stop merging on all WBB entries\"); the "
            "completion barriers (0, the CM's 2 / 3, reserved stypes) also "
            "perform the fill-store buffer's waiting stores and wait for every "
            "write and fill; the ordering barriers (4, 0x10-0x13) wait for "
            "nothing (24K: \"the core will not wait for pending "
            "transactions\"), SYNC_WMB / MB / RELEASE perform the waiting "
            "stores first (younger stores must not pass them; loads are "
            "processed in order). No external SYNC transaction (Config7.ES is "
            "not implemented); stype 0 in a CPS follows GCR_CONTROL.SYNCCTL "
            "(MD01025 Table 11.23): 0 (reset) an intervention sync — the "
            "waiting stores are performed, no wait for memory —, 1 a memory "
            "sync (the completion barrier above).",
    "memory ordering": "a CPS core in the pipeline with non-blocking caches "
                       "(the 24K model, or --nonblocking): a coherent store "
                       "needing the CM (a miss, or a Shared line) waits in the "
                       "fill-store buffer (8 lines) and is globally performed "
                       "INTV cycles later, oldest first, when the CM orders it "
                       "(eMIPS performs the request then: interventions, the "
                       "bytes into the line); younger loads and stores that hit "
                       "go on, its own loads of those bytes are forwarded, of "
                       "other bytes of the line wait for the line (C2C / L2 / "
                       "memory) — so other cores can see a core's stores late "
                       "and out of order with its loads (store buffering), as "
                       "on weakly ordered MIPS systems (Linux asm/sync.h). SC "
                       "never waits in the buffer (it needs the line). A "
                       "barrier performs waiting stores at once (they become "
                       "visible no later than the hardware would make them). "
                       "Load values are taken when the load executes (a miss's "
                       "data is the line as it is then). The instruction-level "
                       "machine and blocking caches are sequentially "
                       "consistent. One system bus behind the CM for every "
                       "core (refills and writes queue on it in the system's "
                       "clock).",
    "memory timing": "only in the pipeline (the instruction-level machine "
                     "takes one cycle per instruction) and only with caches "
                     "(without them memory is ideal, one cycle, like on-chip "
                     "SRAM).  emips.timing: L2 12 / memory 50 cycles to the "
                     "first doubleword and 1 per further one (MD00343 2.2.4, "
                     "MD00355 3.4.1), L3 25 and a write 10 are eMIPS's "
                     "assumptions; CACHE costs from MD00355 Table 3.5.  "
                     "Blocking by default (the textbook); nonblocking = the "
                     "24K's hit under miss with 8 Fill-Store Buffer lines.  "
                     "Refills share one bus (data beats in turn); writes go "
                     "out one after another; reads are not held behind "
                     "unrelated writes.  The L2's own write-backs, refills "
                     "of a longer L2 line and an L2 / L3 CACHE operation's work "
                     "cost nothing beyond their write-buffer entry.",
    "cache contents after reset": "undefined in the manual: random tags, "
                                  "state bits, data, parity and LRU from a "
                                  "fixed seed (runs repeat); Caches("
                                  "hardware_init=True) is the 24K's Config7.HCI "
                                  "(all invalid).  The boot firmware "
                                  "(Program.load(boot=True)) invalidates them.",
    "CACHE": "an operation on a missing cache or a reserved one, an address "
             "operation on an uncached address and ErrCtl.SPR (no scratchpad "
             "RAM) do nothing (the Release 6 rule; UNDEFINED before).  An index "
             "operation naming a way that does not exist does nothing.  Two "
             "valid ways with the same tag (made by Index Store Tag): the "
             "lowest one hits.",
    "LRU state": "an Index Store WS value that names no single oldest way "
                 "replaces the candidate older than the most others, lowest "
                 "way first; with more than 4 ways WS test mode reaches the "
                 "first 6 pair bits and the dirty bits of ways 0-3 only.",
    "precode bits": "the 24K does not publish its encoding: eMIPS stores 3 "
                    "bits per instruction (branch/jump, register jump, link); "
                    "they do not affect execution.",
    "Cache Error exception": "taken precisely, by the instruction whose "
                             "access read the bad array (the 24K may be "
                             "imprecise); ES = EE = EB = SP = 0 (no external "
                             "requests, no scratchpad).",
    "PREF": "the 24K's hints (MD00343 Table 12.1); a prefetch hit on a "
            "resident line only updates the LRU (not for the streamed "
            "hints 4/5); PrepareForStore (30) needs a write-back attribute.",
    "MMIO through a cacheable segment": "a line fill / write-back reads / "
                                        "writes device registers byte by "
                                        "byte, with their side effects.",
    "TLB": "Machine(mmu=TLBMMU(N)) (emips.tlb), the 24K's (MD00343 4.3-4.4): "
           "1..64 dual entries, pages 4 KB-256 MB (+ 1 KB with PageGrain.ESP, "
           "Release 2-5), a hidden written bit per entry (cleared at reset: an "
           "uninitialised entry never matches; TLBR reads its undefined, "
           "seeded-random contents), write-compare: entries matching a TLBWI / "
           "TLBWR are disabled, or a Machine Check aborts the write when both "
           "are valid (Table 4.9) — so TLBP / TLBR never see multiple matches. "
           "32-bit physical addresses (PFN 25:6; 27:6 with 1 KB pages); no XPA "
           "(Config3.LPA = 0), no ContextConfig (CTXTC = 0), Config4.KScrExist "
           "= 0.",
    "CPS": "emips.cps (M11 phase 1; the CM2 GCR / CPC of the 1004K, "
           "interAptiv, P5600: Boot-MIPS MD00901, Linux mips-cm.h / mips-cpc.h "
           "/ smp-cps.c; checked after M12 against the P5600 manual MD01025 ch. "
           "10 / 11). eMIPS choices: the CPC sequences at once (no SEQDEL / RAIL "
           "/ RESETLEN delays, no transitional states, a command is never "
           "preempted); ClockOff of a running core: D2 (PwrUp resumes it), of a "
           "powered-down one: U2 (PwrUp / Reset reset it); a CM2 has no "
           "GCR_Cx_RESET_RELEASE (offset 0 is reserved): cores start through "
           "the CPC; the reset base is taken when the core starts "
           "(RESET_EXT_BASE.UseExceptionBase, reset 1: GCR_Cx_RESET_BASE with "
           "bits 31:30 = 10; 0: 0xBFC00000; the BEV overlay fields are stored "
           "only; the BEV vectors: base + 0x200); GCR_REGn steer the shared L2 "
           "only (see Coherence Manager); CPC_Cx_OTHER.CORENUM is 23:16 (Linux; "
           "the P5600: 18:16); one VPE per core; no IOCU; no CM power domain "
           "(the GIC: see GIC). The firmware "
           "(kernel / user boot) gives every core write access, enables the "
           "CPC at 0x1BDE0000, puts every core in the coherence domain, maps "
           "cacheable memory coherent (CCA 5, as Boot-MIPS does for coherent "
           "cores) and starts them all at the entry with $sp 64 KB apart "
           "(eMIPS's MARS-like boot); the reset boot runs core 0 only. Each "
           "step runs every running core one instruction / cycle, core 0 "
           "first; L1 caches per core, the L2 / L3 shared behind the CM.",
    "Coherence Manager": "emips.coherence (M11 phase 2; the CPS training notes, "
                         "Linux mips-cm.h / mips-cm.c / c-r4k.c / cps-vec.S). MESI "
                         "between the L1 data caches for CCA 4 / 5 of cores in "
                         "coherent mode (GCR_Cx_COHERENCE own bit; bit i = "
                         "interventions from core i). eMIPS choices: a request "
                         "completes at once, in execution order (core 0 first); an "
                         "Exclusive / Modified copy is sent cache to cache (C2C "
                         "cycles), a Shared one comes from the L2 / memory; Read Own "
                         "of a Modified line hands the dirty data over (no "
                         "writeback), Read Share writes it back (with GCR_CONTROL."
                         "SHST_EN a lone copy is installed Shared too); an LL asks "
                         "Shared even with CCA 4 (P5600 SUM); instruction refills "
                         "are non-coherent reads (Legacy RD: the L2 / memory, never "
                         "another core's D-cache — new code needs a Hit Writeback / "
                         "SYNCI, MD01025); Hit Invalidate / Hit Writeback "
                         "(Invalidate) / SYNCI on a coherent address are done in "
                         "every core of the domain; PREF 30 and stores of a core "
                         "without a data cache write a Modified copy back before "
                         "invalidating it; a store others can see (any but a "
                         "non-coherent one kept in the D-cache) clears their LLbit "
                         "for its 32-byte block, an intervention invalidating the "
                         "LL's line does too; a core leaving the domain keeps its "
                         "lines (software must flush, as the notes say); CCA 4 / 5 "
                         "of a core not in coherent mode act as 3; DTagLo holds "
                         "the state as the 1004K (MD00622 Table 9.1: V = 0, L = 1 "
                         "is Shared; V = 0, L = 1, D = 1, reserved, stores an "
                         "invalid line), locked lines are not coherent (no "
                         "interventions); inconsistent copies record an "
                         "intervention error (MD01025 Table 11.11: 18 for a read — "
                         "it needs a response —, 17 for an upgrade; the info has "
                         "each core's state and SResp, the SC bit) and the request "
                         "goes on; a line fill of a GCR / GIC / CPC register (a "
                         "cached load, a write-back store miss) records GC_RD (2) "
                         "or COH_RD (4) and does nothing (a load returns 0), a "
                         "single beat (UCA, a write-through store, no D-cache) "
                         "reaches the register; a request that needs a response "
                         "and whose GCR_ERROR_MASK bit is 0 (the reset value "
                         "0x000A002A) gets an ERROR response: the core takes a data "
                         "bus error, imprecisely (before its next instruction, as "
                         "the 24K, MD00343 5.8.13); a set mask bit raises the CM's "
                         "error interrupt instead (wired to GIC source 2: the "
                         "manual leaves it to the system); GCR_ERROR_CAUSE / MULT "
                         "are R/W (a type stays until written 0), ERROR_ADDR 2:0 "
                         "read 0; a region's CCA override chooses how the shared L2 "
                         "treats a line (uncached / UCA: bypass, 0: write-through) "
                         "only when its target is memory, DROP_L2 drops the "
                         "region's L2 / L3 Hit-type CACHE operations; IOCU targets "
                         "act as memory; an uncached "
                         "write to the L2-only SYNC region is swallowed (the L2 has "
                         "no internal reordering to barrier); one system bus "
                         "for all cores (see memory ordering).",
    "GIC": "emips.gic (M11 phase 4; checked against the P5600 manual MD01025 "
           "ch. 12 after M12: offsets, reset values — MAPn_PIN = pin 0 of no "
           "core, the local mask 0x7F, the local maps WD -> NMI, TIMER / "
           "PERFCTR / FDC -> pin 5, the others pin 0 — and the watchdog). eMIPS "
           "choices: 40 sources (Boot-MIPS's example system); the shared "
           "counter and the watchdogs count system cycles (the rate is "
           "implementation-specific / not given); the core's timer, "
           "performance counter and software interrupts are routable "
           "(GIC_VPEi_CTL 3:1 = 1, the 1004K-generation build option; the "
           "P5600 has no such bits, its local interrupts always go through "
           "the GIC; no FDC): a routable timer / counter is not wired to IP7 "
           "(their reset maps put them on pin 5 = IP7), the software "
           "interrupts go through the GIC only in EIC mode (MD01025 "
           "12.3.8.7); EIC mode (12.5.3.15: the MAP field is one less than "
           "the EIC level): a map field n is level n + 1 (63: 63), the "
           "highest wins, its shadow set is GIC_VPEi_EIC_SHADOW_SET[vector] "
           "(the 1004K generation; the P5600 has no shadow sets); a source "
           "mapped to several cores (MAP_CORE: \"only one bit may be set\") "
           "or to pin and NMI at once (\"only one ... at any one time\") "
           "goes to all of them; the watchdog (12.3.7.2, 12.6.4): TYPE 0 one "
           "trip, 1 second countdown (the second expiry resets every core "
           "that is up, WDRESET survives; the watchdog stops), 2 interval "
           "timer, 3-7 (reserved) as one trip; it stops while its core "
           "sleeps (WAIT, a CPC low-power state) unless WD_CONFIG.WAIT, a "
           "one-trip expiry leaves WD_START as written; no yield qualifiers "
           "(MAP_TO_YQ is stored, does nothing); REVISIONID reads 0; "
           "unconnected sources are tied low (so an active-low level source "
           "is pending — masked at reset); the COUNTER is written only while "
           "COUNTSTOP (12.5.3.2: else unpredictable); a WEDGE to a level "
           "source does nothing (undefined in 12.5.3.9). The console's TX / "
           "RX are sources 0 / 1; the boot firmware routes them to core 0's "
           "pins 0 / 1 and sets up sources 24-39 as Boot-MIPS's IPIs.",
    "NMI": "Vol. III Non Maskable Interrupt: taken before the next "
           "instruction (the pipelines: when it would enter EX, like an "
           "interrupt), ErrorEPC = the restart PC, Status.BEV / ERL / NMI = 1, "
           "SR = TS = 0, PC = the reset vector (a CPS core's reset base); "
           "Cause untouched; Status.NMI / SR can only be cleared by software. "
           "Its only source is the GIC (MAP_TO_NMI): an edge on the input.",
    "control logic as gates": "emips.controllogic / alulogic (the Datapath's part "
                              "window): MIPS's own decoder circuits are not published, so "
                              "the main control and the ALU control are drawn as PLAs as "
                              "Patterson & Hennessy draw the textbook subset's, generated "
                              "for every MIPS32 instruction of the release from the "
                              "emulator's decode (Quine-McCluskey minimized, checked against "
                              "every instruction row): an opcode PLA, secondary PLAs on the "
                              "field that tells an opcode's instructions apart (R6 compact "
                              "branches: comparators), a 4-bit ALUOp (the textbook's 2 bits "
                              "cannot name MIPS32's ALU work) and an ALU whose 3-bit "
                              "Operation adds XOR and the barrel shifter to the textbook "
                              "slice. The gate values are derived from the traced operands "
                              "and agree with the traced result.",
    "24K pipeline": "emips.pipe24k (MD00343 chapter 2): results from the "
                    "Machine's executors in EX, in program order; exceptions "
                    "and interrupts are taken in ER (an interrupt goes with the "
                    "instruction about to enter EX). eMIPS choices: the fetch "
                    "unit holds the 8-entry instruction buffer plus the IF / IS "
                    "bundles (12 instructions); each word of a bundle is its "
                    "own I-cache / ITLB access (the 24K: one per bundle); the "
                    "sequential fetches a predicted-taken branch kills are not "
                    "made; the BHT resets to weakly not taken (UNDEFINED); the "
                    "return prediction stack is circular, pushed by JAL, JALR "
                    "$31, BGEZAL / BAL (2.2.2) and the R6 BALC / JIALC, and not "
                    "repaired after a misprediction; a fetch fault stops the "
                    "fetch until a redirect; R6 instructions take the timing of "
                    "the nearest 24K class. Units: a waiting access (blocking "
                    "miss with Config7.SL, full write or fill-store buffer, "
                    "SYNC, CACHE) holds MS and everything behind (the 24K slips "
                    "a blocking load in ER); a non-blocking miss delays only its "
                    "users and takes a load data queue entry (9); a DTLB miss "
                    "holds the LSU only (Config7.NBLSU: the pipes); a write to a "
                    "register a missing load has yet to deliver waits until 3 "
                    "cycles before the data; an MDU instruction writing HI / LO "
                    "after a divide starts one cycle before the divide ends; a "
                    "MUL's write-port arbitration is not modelled; every FPU "
                    "instruction except the stores passes M1 (repeat rates), at "
                    "the integer clock (Config7.FPR = 0); Config7.CPOOO has no "
                    "effect. Not modelled: replay traps, FSB index conflicts, "
                    "IFU fill-buffer-full refetches. CP0 hazards (Table 2.7): a "
                    "CP0 write (MTC0, EI / DI, TLBR / TLBP / TLBW*) is seen by an "
                    "instruction executing at least spacing + 1 cycles after "
                    "it (2; 3 for EntryHi / Status / TLB entries -> loads and "
                    "stores; 4 for Status -> coprocessor instructions and "
                    "Compare -> the timer interrupt), the earlier ones see the "
                    "old value; EHB waits until 3 cycles after the last write "
                    "and clears them, ERET / JR.HB / JALR.HB clear them for what "
                    "follows (ERET itself still reads an EPC written too "
                    "close); instruction hazards (Table 2.8) come from the "
                    "fetch running ahead: what the fetch unit already holds was "
                    "fetched with the old state (up to 12 instructions, not "
                    "the table's 10); MTC0 SRSCtl.PSS -> RDPGPR / WRPGPR: 1; "
                    "Watch, CorExtend hazards: no such hardware. The Machine and the 5-stage model have no "
                    "hazards (CP0 writes are seen at once). The forwarding "
                    "option is the 5-stage model's: the 24K always has its "
                    "bypasses; no MIPS16e (IR / IK stages). Config7's 24K bits "
                    "(SL, BHT, RPS, BP, ULB, NBLSU, CPOOO) are writable in every "
                    "model; only the 24K model uses them.",
    "performance counters": "two counters as the 24K (MD00343 6.2.35: "
                            "PerfCtl0/1, PerfCnt0/1, its 7-bit Event field and "
                            "event lists, Table 6.46); reset: controls and counts "
                            "0 (UNDEFINED; no mode bit: nothing counts), an event "
                            "not marked Mode in Table 6.46 counts in every mode "
                            "once any of U / S / K / EXL is set; IntCtl.IPPCI = 7 "
                            "(the timer's line; Release 1: hardware interrupt 5). "
                            "Instruction events count at completion (the pipeline: "
                            "WB) in the mode the instruction ran in; cache / TLB "
                            "events follow the caches' and micro-TLBs' counters "
                            "(per instruction fetch, not per 64-bit fetch bundle); "
                            "stall events come from the pipeline model in use; "
                            "events of structures a model lacks (branch "
                            "prediction, IFU buffers, FSB / LDQ occupancy) count "
                            "nothing until the 24K pipeline model; DTLB / ITLB "
                            "accesses count 0 with the fixed mapping and BAT MMUs "
                            "(no micro-TLBs).",
    "TLB Random": "a clock-rate down-counter from N-1 to Wired (MD00355 "
                  "3.7.3: one step per cycle, so per instruction in the "
                  "Machine and per clock in the pipeline) with the 24K's "
                  "10-bit LFSR that skips a decrement now and then against "
                  "livelocks (MD00343 6.2.2); its taps are not published: "
                  "eMIPS uses x^10 + x^7 + 1 and skips when the low 4 state "
                  "bits are set (about 1 cycle in 16, tlb.LFSR_SKIPS).  "
                  "Release 6: the register reads 0, TLBWR uses the hidden "
                  "counter.",
    "TLB UNDEFINED values": "pre-R6, an Index or Random value >= N selects "
                            "entry value mod N (the low index bits), Wired "
                            ">= N keeps Random at N-1; a PageMask that is "
                            "no page size is stored one bit per pair (the "
                            "upper bit of each pair, as the 24K's compressed "
                            "mask) — Release 6 ignores such writes and an "
                            "out-of-range Index.  The VPN2 / PFN bits under "
                            "the mask are kept on a write and read back.",
    "TLB exceptions": "EntryLo0 / EntryLo1 are UNPREDICTABLE after one: left "
                      "unchanged; EntryHi.EHINV is kept.  BadInstr / BadInstrP "
                      "without an instruction word (a fetch fault) are "
                      "UNPREDICTABLE: left unchanged.  A miss with EXL = 1 "
                      "takes TLBL / TLBS at the general vector (the 24K calls "
                      "it TLB Invalid, MD00343 5.8.11).",
    "micro-TLBs": "4-entry ITLB (true LRU) and 8-entry DTLB (tree pseudo-LRU) "
                  "of 4 KB / 1 MB (sub)pages (1 KB pages are kept whole); "
                  "entries carry the ASID and G (the manual does not say "
                  "whether an ASID change flushes them: never visible); a "
                  "JTLB write drops the copies of that entry, TLBINV / TLBINVF "
                  "all of them; a miss costs the pipeline MEMORY_TIMING['UTLB'] "
                  "= 3 cycles (MD00355 3.7.2) in IF or MEM.  Only mapped "
                  "addresses use them.",
    "TLB and fetch": "the pipeline fetches ahead with the mapping of its "
                     "fetch cycle, as hardware does (an instruction hazard): "
                     "change the mapping of code, then ERET or JR.HB to it.",
    "BAT": "Machine(mmu=BATMMU()) (emips.bat, Vol. III A.2): all six entries "
           "(I / D x kuseg, kseg2, kseg3), no supervisor mode; Random, "
           "EntryLo1, Context, PageMask and Wired read 0 and ignore writes (the "
           "Release 6 rule; UNDEFINED before); TLBP / TLBWR are Reserved "
           "Instructions; an out-of-bounds or invalid access is TLB Invalid "
           "(general vector) and writes only BadVAddr; the boot firmware "
           "loads the fixed mapping's map.",
    "VTLB + FTLB": "TLBMMU(N, ftlb=(sets, ways)) — the P5600's MMU (Vol. III "
                   "A.3; MD01024 sizes: 64-entry VTLB, 512-entry 4-way FTLB; "
                   "checked against the P5600 manual MD01025 ch. 3): the FTLB is "
                   "off after reset (Config6.FTLBEn = 0: Config.MT = 1, "
                   "Config4.MMUExtDef = 1, Index 6 bits; FTLBEn = 1: MT = 4, "
                   "MMUExtDef = 3) and the boot firmware turns it on (as "
                   "Linux); its contents across an FTLBEn change are kept "
                   "(UNDEFINED in the manual); 4 KB or 16 KB FTLB pages; the "
                   "set is VA[19:13] ^ VA[26:20] (16 KB: [21:15] ^ [28:22]; "
                   "other set counts: the same rule); TLBWR of the FTLB page "
                   "size goes to the FTLB (Config6.FTLBP: 1 in 16 / 8 / 4 to "
                   "the VTLB), its way chosen by a cycle counter (the manual: "
                   "\"indexing is performed exclusively in hardware\" — an "
                   "eMIPS choice); Index = {way, set} + VIndex; TLBWI checks "
                   "the page size and the set (Machine Check, MCCause 3 / 4); "
                   "a write invalidates older duplicates in the VTLB and its "
                   "FTLB set without a Machine Check, a lookup / TLBP that "
                   "finds several takes one (MCCause 1); Config4.IE = 2: "
                   "TLBINV / TLBINVF act on the VTLB or one FTLB set by Index; "
                   "Random moves only on VTLB writes (a 10-bit LFSR: the "
                   "\"pseudo-random LRU\" is unpublished) and Wired writes; "
                   "Config4.M = 1, KScrExist = 0x1C; only VTLB entries can be "
                   "wired; no 1 KB pages; the 24K's micro-TLBs (the P5600's "
                   "are 16 / 32 entries), no FTLB parity, configurable sizes "
                   "beyond the P5600's (eMIPS choices).",
    "Segmentation Control": "TLBMMU(..., sc=True) (Vol. III 4.10, Release "
                            "3+; emips.segctl): reset to the legacy map with "
                            "the values QEMU's P5600 model uses; no overlay "
                            "segments (Solution 1: CFG2 stays unmapped and "
                            "uncached after reset); AM = 6 behaves as UK; "
                            "Config5.K = 0 gives the kseg0 region (CFG3) "
                            "Config.K0 when unmapped, and keeps the "
                            "exception vectors in kseg0 / kseg1 (EBase[31:30] "
                            "read as 10, MD01025 3.6.3.2); 32-bit physical "
                            "addresses (high PA bits dropped); EBase.WG "
                            "write gate; Config5.CV for the Cache Error "
                            "vector.  EVA instructions exist only with it "
                            "(Config5.EVA), else Reserved Instruction.",
    "page table walker": "TLBMMU(..., pw=True) (Vol. III 4.12, Release 5+; "
                         "emips.ptw): runs for a TLB miss with Status.EXL = 0 "
                         "only; reads through the caches with kernel rights, "
                         "mapped tables only when a valid TLB entry maps them "
                         "(no nested refill); PTEW 0 / 1, HugePg and DPH "
                         "implemented (a dual-page pair without PTEvld is a "
                         "Machine Check, MCCause 5); with 8-byte PTEs and huge "
                         "pages directory entries are 8 bytes (Table 9.34); "
                         "pages above 256 MB or below 4 KB abort the walk; "
                         "with the P5600's MMU (ftlb) PWField / PWSize reset to "
                         "its values (MD01025: 0x0C30C302, PTW = 1).",
    "boot firmware with a TLB":"Program.load(boot=True) writes every entry "
                                "unique and invalid (Vol. III 4.9.3 InitTLB) "
                                "and, unless the program has code at "
                                "0x80000000-0x800000FF, installs a refill "
                                "handler there (emips.tlb.FIRMWARE_REFILL) "
                                "that maps kuseg at VA + 1 GB (cacheable) and "
                                "kseg2 / kseg3 at PA = VA (uncached), global "
                                "4 KB pages — what the fixed mapping does; the "
                                "views and the loader use the same mapping for "
                                "a page not yet in the TLB.",
    "GDB stub": "emips gdb / Run > GDB Server (emips.gdbstub) is a debug agent, "
                "not part of the core: it reads and writes registers and memory "
                "through the debugger's side-effect-free views (no exception, no "
                "cache or TLB counting; an unmapped address or a bus error is an E "
                "reply) and keeps breakpoints itself (Z0 and Z1 alike, no BREAK "
                "written, so one in a delay slot works). Watchpoints (Z2-Z4) are "
                "checked before every load / store of the program "
                "(Machine.agent_watch, before the EJTAG / watch-register checks; "
                "never for views, fetches, CACHE / PREF or DMA) and stop before the "
                "access, as a precise Watch exception; the instruction runs again "
                "(a delay slot keeps its branch). GDB steps MIPS in software (a "
                "temporary breakpoint after the delay slot). Threads = cores in "
                "all-stop mode: every running core steps together (the Session's "
                "lockstep) unless GDB resumes only some threads (vCont;c:1, "
                "stepping over a breakpoint: the other cores stay stopped); a "
                "watch stop lets the other cores finish their step and costs the "
                "stopped core that step. Pipeline mode: stops are "
                "precise (the instruction about to execute; all older executed); "
                "the 5-stage pipeline holds a branch with a breakpoint in ID (it "
                "reaches EX unexecuted and is fetched again when the run goes on: "
                "a few cycles per stop) and a watched access holds EX for a cycle "
                "(the 24K: AG); writing the PC flushes the pipeline. The exception "
                "signals follow MIPS Linux's (AdE / TLB SIGSEGV, IBE / DBE SIGBUS, "
                "RI / CpU SIGILL, Ov / FPE SIGFPE). An assembly program gets a "
                "written ELF (emips.elfwrite: labels as NOTYPE symbols, a DWARF 2 "
                "line table, no frame information: GDB's backtrace in assembly is "
                "its own heuristic).",
}

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

BUCKET_EXEC = "EXEC"
BUCKET_HINT = "HINT"
BUCKET_ILLEGAL = "ILLEGAL"
BUCKET_DEFER = "DEFER"

#: HINT rows: hints, barriers and cache maintenance: they change no
#: register.  CACHE / PREF / SYNCI run :data:`_HINT_EXECUTORS` (they act
#: on the caches when the machine has some); the others are no-ops here.
HINT_NAMES: frozenset[str] = frozenset(
    {"CACHE", "CACHEE", "PREF", "PREFE", "PREFX", "SYNC", "SSNOP",
     "EHB", "PAUSE", "SYNCI", "CACHE@R6", "PREF@R6"}
)

#: FPU-family names: gated by ``Status.CU1`` (paired-single rows are
#: ILLEGAL: the FPU has no PS format, ``FIR.PS = 0``).
FPU_FAMILIES: frozenset[str] = frozenset(
    {"fpu_arith", "fpu_cmp", "fpu_cvt", "fpu_dsp", "cop1_move", "branch_fpu"}
)

#: Rows not executed yet (simulator limitation, always raises
#: :class:`UnimplementedError`): none since EJTAG Debug Mode (M14).
DEFER_NAMES: frozenset[str] = frozenset()

#: The EVA rows (Vol. III 4.11, Table 4.6): kernel accesses with user
#: rights, implemented only with Segmentation Control (``Config5.EVA``),
#: else Reserved Instruction.
EVA_NAMES: frozenset[str] = frozenset(
    {"LBE", "LBUE", "LHE", "LHUE", "LWE", "LWLE", "LWRE", "LLE",
     "SBE", "SHE", "SWE", "SWLE", "SWRE", "SCE", "PREFE", "CACHEE"})

#: Rows that need kernel mode or ``Status.CU0`` (else Coprocessor
#: Unusable, CE = 0): CP0 access, cache ops and the EVA family.
PRIVILEGED_NAMES: frozenset[str] = frozenset(
    {"MFC0", "MTC0", "ERET", "DERET", "WAIT", "DI", "EI", "RDPGPR", "WRPGPR",
     "TLBP", "TLBR", "TLBWI", "TLBWR", "TLBINV", "TLBINVF", "CACHE", "CACHEE",
     "LBE", "LBUE", "LHE", "LHUE", "LWE", "LWLE", "LWRE", "LLE",
     "SBE", "SHE", "SWE", "SWLE", "SWRE", "SCE", "PREFE", "CACHE@R6"}
)


def has_delay_slot(entry: Mapping[str, Any]) -> bool:
    """Branches and jumps (pre-Release 6): the next instruction is their
    delay slot whether or not they are taken (``ERET`` has none)."""
    return entry["family"] in ("branch", "branch_fpu", "jump") or \
        entry["name"].startswith("BC2")


def is_cti(entry: Mapping[str, Any]) -> bool:
    """Control transfer instruction: any branch or jump (R6: may not be
    placed in a delay slot or a forbidden slot)."""
    return entry["family"] in ("branch", "branch_fpu", "jump", "branch_compact",
                               "jump_compact") or entry["name"].startswith("BC2")


def classify(entry: Mapping[str, Any]) -> str:
    """Bucket a decoded :class:`~emips.isa` row (:attr:`Decode.entry`).

    Order: HINT → ILLEGAL (``status == 'unimpl'`` or a tie-break loser)
    → DEFER (:data:`DEFER_NAMES`) → EXEC.
    """
    name = entry["name"]
    if name in HINT_NAMES:
        return BUCKET_HINT
    if entry["status"] == "unimpl" or not entry["decode"]["lead"]:
        return BUCKET_ILLEGAL
    if name in DEFER_NAMES:
        return BUCKET_DEFER
    return BUCKET_EXEC


# ---------------------------------------------------------------------------
# Small bit helpers
# ---------------------------------------------------------------------------


def _s16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def _s32(value: int) -> int:
    value &= M32
    return value - 0x100000000 if value & 0x80000000 else value


def _s9(value: int) -> int:
    value &= 0x1FF
    return value - 0x200 if value & 0x100 else value


def rs(word: int) -> int:
    return (word >> 21) & 31


def rt(word: int) -> int:
    return (word >> 16) & 31


def rd(word: int) -> int:
    return (word >> 11) & 31


def sa(word: int) -> int:
    return (word >> 6) & 31


def imm16(word: int) -> int:
    return word & 0xFFFF


def off9(word: int) -> int:
    """Signed 9-bit ``offset`` of the EVA rows (bits 15..7)."""
    return _s9((word >> 7) & 0x1FF)


def _br_target(pc: int, word: int) -> int:
    """Branch target: ``PC + 4 + sign_extend(offset || 00)``."""
    return (pc + 4 + (_s16(imm16(word)) << 2)) & M32


def _j_target(pc: int, word: int) -> int:
    """J/JAL target: ``(PC + 4)[31:28] || instr_index || 00``."""
    return ((pc + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2)


# ---------------------------------------------------------------------------
# CP0 table (names from emips/data/cp0.json, reset/masks from emips.cp0)
# ---------------------------------------------------------------------------

_CP0_PATH = Path(__file__).resolve().parent / "data" / "cp0.json"


def _load_cp0_table() -> dict[tuple[int, int], dict[str, Any]]:
    """Valid ``(reg, sel)`` CP0 cells.  A cell is valid iff the JSON has an
    entry with a ``value`` key for that pair; the known gaps (registers
    20/21/22, register 10 selector 7) stay absent, as in the data."""
    raw: Any = json.loads(_CP0_PATH.read_text(encoding="utf-8"))
    table: dict[tuple[int, int], dict[str, Any]] = {}
    for reg, sels in raw.items():
        if not isinstance(sels, Mapping):
            continue
        for sel, info in sels.items():
            if isinstance(info, dict) and "value" in info:
                cell = dict(info)
                cell["value"] = int(info["value"]) & M32
                table[(int(reg), int(sel))] = cell
    return table


class _Arch(Exception):
    """Internal signal: an executor raised an architectural exception."""

    def __init__(self, code: int, *, badvaddr: int | None = None,
                 ce: int = 0, info: dict[str, Any] | None = None) -> None:
        super().__init__(C.EXC_NAMES.get(code, str(code)))
        self.code = code
        self.badvaddr = badvaddr
        self.ce = ce
        self.info = info or {}


#: Executor return value: a branch-likely was not taken — skip its delay slot.
NULLIFY = -1

# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------


class Machine:
    """Instruction-level MIPS32 machine.

    Holds the architectural state — 32 GPRs, PC, HI/LO, CP0, the LL/SC
    LLbit, a stub FPR file — and steps it.  Parameters:

    ``endian``      ``"big"`` (default) or ``"little"``.
    ``exceptions``  ``"cp0"`` (default): exceptions are handled by CP0 and
                    the exception vector, like the hardware.  ``"raise"``:
                    debug mode, raise a :class:`MipsException` instead.
    ``caches``      ``None`` (default, no caches) or an
                    :class:`emips.cache.Caches` (L1 instruction / data).
    ``shadow_sets`` GPR sets, 1 (default: the normal GPRs only) to 16
                    (``SRSCtl.HSS + 1``; Release 2+, Vol. III 9.31; the 24K
                    builds 1, 2 or 4).  :attr:`regs` is the current set.
    """

    def __init__(self, isa: ISA | None = None, memory: PMemory | None = None,
                 pc: int = 0, *, endian: str = "big",
                 exceptions: str = "cp0", release: str | None = None,
                 mmu: Any = None, caches: Caches | None = None, cpu: int = 0,
                 shadow_sets: int = 1) -> None:
        if endian not in ("big", "little"):
            raise ValueError(f"endian must be 'big' or 'little', not {endian!r}")
        if exceptions not in ("cp0", "raise"):
            raise ValueError(f"exceptions must be 'cp0' or 'raise', not {exceptions!r}")
        if isa is not None and release is not None and isa.release != release:
            raise ValueError(f"isa is {isa.release}, release={release!r}")
        self.isa = isa if isa is not None else ISA(release=release)
        #: MIPS32 release of this core (ISA, Config.AR, FPU width)
        self.release = self.isa.release
        #: Count increments every ``ccres`` cycles (the implementation's
        #: CCRes, read by RDHWR 3; typically 2, :data:`emips.cp0.CCRES`)
        self.ccres = C.CCRES
        #: address translation (Config.MT): emips.mmu.FixedMappingMMU
        self.mmu = mmu if mmu is not None else default_mmu()
        self._wmask = C.write_masks(self.release, supervisor=self.mmu.supervisor)
        #: physical memory, keyed by physical address
        self.mem = memory if memory is not None else PMemory()
        self.big_endian = endian == "big"
        self.exceptions = exceptions
        self.pc = pc & M32
        if not 1 <= shadow_sets <= 16:
            raise ValueError(f"shadow_sets must be 1..16, not {shadow_sets}")
        #: the GPR sets (Release 1: only the normal GPRs); ``regs`` is the
        #: current one (SRSCtl.CSS), set 0 = the normal GPRs
        self.shadow: list[list[int]] = [
            [0] * 32 for _ in range(shadow_sets if self.release != "R1" else 1)]
        self.regs: list[int] = self.shadow[0]
        self.hi = 0
        self.lo = 0
        #: FPU: 32 64-bit FPRs (see fpr_* for the FR=0 / FR=1 views),
        #: FCSR (FCR31) and the read-only FIR (FCR0).
        self.fpr: list[int] = [0] * 32
        self.fcsr: int = 0
        self.fir: int = FIR if self.release != "R1" else FIR_R1
        if self.release == "R6":                   # IEEE 754-2008 FPU
            self.fir |= FIR_HAS2008
            self.fcsr = FCSR_RO_2008
        #: CP0 registers, keyed ``(register, selector)``.
        self.cp0: dict[tuple[int, int], dict[str, Any]] = _load_cp0_table()
        for key, name in C.EXTRA_CELLS.items():
            self.cp0.setdefault(key, {"name": name, "value": 0})
        for key, value in C.reset_values(self.big_endian, self.release).items():
            self.cp0[key]["value"] = value
        #: SRSCtl.HSS: the highest shadow set (0 = the normal GPRs only)
        self.hss = len(self.shadow) - 1
        self.cp0[C.SRSCTL]["value"] = self.hss << C.SRS_HSS
        if self.hss:                            # ESS, PSS; SRSMap (Vol. III 9.31 / 9.32)
            self._wmask[C.SRSCTL] = (0xF << C.SRS_ESS) | (0xF << C.SRS_PSS)
            self._wmask[C.SRSMAP] = M32
        else:                                   # SRSMap UNPREDICTABLE: reads 0
            self._wmask[C.SRSMAP] = 0
        #: this core's number in a multi-core system (emips.cps): EBase.CPUNum
        #: (9:0, read-only), RDHWR 0, GCR_CL_ID
        self.cpu_num = cpu
        self.cp0[C.EBASE]["value"] |= cpu & 0x3FF
        #: the multi-core system this core belongs to (emips.cps.CPS) or None
        self.system: Any = None
        #: the BEV = 1 vector base (0xBFC00200; a CPS core: its reset base
        #: GCR_Cx_RESET_BASE + 0x200)
        self.bev_base = C.BEV_BASE
        #: L1 caches (emips.cache.Caches) or None
        self.caches = caches
        #: the attributes that go through the caches (a CPS core: + the
        #: coherent 4 and 5, emips.coherence)
        self.cached_cca = caches.cached_cca if caches is not None else CACHED_CCA
        #: cacheability attributes Config.K0 / KU / K23 implement
        self.supported_cca = SUPPORTED_CCA
        if caches is not None:
            caches.m = self
            self.supported_cca = CCA_24K
            self._wmask.update(C.cache_masks(caches.parity, ecc=caches.ecc,
                                             l2=caches.l2 is not None,
                                             write_buffer=caches.wbb is not None))
            self.cp0[C.CONFIG1]["value"] |= C.cache_config1(
                caches.i and caches.i.geometry, caches.d and caches.d.geometry)
            self.cp0[C.CONFIG2]["value"] |= C.cache_config2(
                caches.l2 and caches.l2.geometry, caches.l3 and caches.l3.geometry)
            self.cp0[C.CONFIG7]["value"] |= int(caches.hardware_init) << 18
            if caches.wbb is not None:          # Config.MM: merging allowed
                self.cp0[C.CONFIG]["value"] |= C.CONFIG_MM
        self.mmu.attach(self)
        if getattr(self.mmu, "mt", 3) in (1, 4):  # TraceControl ASID_M / ASID / G (6.2.29)
            self._wmask[(23, 1)] |= (0xFF << 13) | (0xFF << 5) | (1 << 4)
            self.cp0[(23, 1)]["value"] = 0
        if self.mmu.supervisor:
            self._wmask[(23, 1)] |= 1 << 22
        #: the performance counters (CP0 25; emips.perf)
        self.perf = perf.Counters(self)
        self.perf.reset()
        #: completion events of the instruction just executed and the
        #: Status it ran with (set only while the counters count)
        self.last_events: tuple | None = None
        #: what the last PREF did: None, or True / False = hit / miss
        self.pref_hit: bool | None = None
        #: BadInstr / BadInstrP implemented (Config3.BI / BP)
        self.has_bad_instr = self.release == "R6" or getattr(self.mmu, "bad_instr", False)
        if self.has_bad_instr:
            self.cp0[C.CONFIG3]["value"] |= (1 << 27) | (1 << 26)
        #: the MMU's access translation, (m, va, kind) -> (pa, cca)
        self._xlate = self.mmu.xlate
        #: registers the MMU owns (MTC0 goes to it), its clock (Random)
        self._mmu_regs = getattr(self.mmu, "REGISTERS", None) or frozenset()
        self._mmu_tick = getattr(self.mmu, "tick", None)
        #: segment rights from the MMU (Segmentation Control), else fixed
        self._mmu_address_ok = getattr(self.mmu, "address_ok", None)
        #: an EVA instruction is running: its accesses use user rights
        self._eva = False
        #: LL/SC state: LLbit and the LL address.
        self.lr_addr: int | None = None
        self.lr_valid: bool = False
        #: the LL's physical address (another core's store to its line
        #: clears LLbit, emips.coherence / emips.cps)
        self.lr_pa: int | None = None
        #: Optional ``(machine, code)`` callable: services SYSCALL instead
        #: of taking the exception (OS emulation, as in SPIM/MARS); it may
        #: return False to decline, then the Sys exception is taken.
        self.syscall_handler: Callable[[Machine, int], Any] | None = None
        #: Memory-mapped devices (emips.console.MmioConsole, ...): objects
        #: with covers(addr), load(m, addr, size), store(m, addr, v, size),
        #: tick(m, cycles).
        self.devices: list[Any] = []
        #: Set by an "exit" service: step() then does nothing.
        self.halted: bool = False
        #: One entry per exception / simulator error, in order.
        self.exception_log: list[dict[str, Any]] = []
        #: Armed delay slot: the next step executes the slot, then the PC
        #: becomes ``slot_target``.
        self.slot_armed: bool = False
        self.slot_target: int | None = None
        self._redirect: int | None = None
        #: set by a not-taken conditional compact branch (R6 forbidden slot)
        self._forbidden: bool = False
        #: tracing (emips.trace): build an InstrView per executed instruction
        self.tracing: bool = False
        #: called after every instruction that completed (pc, Decode, word,
        #: the executor's result: a branch target ...): the debugger's
        #: call stack (Session)
        self.call_hook: Callable[[int, Decode, int, int | None], None] | None = None
        #: tracing with caches: the cache accesses of the last fetch
        self.fetch_accesses: list = []
        #: tracing with a TLB / BAT: the translations of the last fetch
        #: (emips.tlb.TLBAccess)
        self.fetch_tlb: list = []
        self.last_view = None
        self._mem_log: list | None = None
        #: Clock: cycles elapsed, and the half-step towards the next Count.
        self.cycles: int = 0
        self._count_phase: int = 0
        #: Hardware interrupt lines HW0..HW5 (bit n = line n, level); with
        #: an External Interrupt Controller in EIC mode the six lines carry
        #: the encoded RIPL instead (Vol. III 6.1.1.3)
        self.irq_lines: int = 0
        #: the shadow set the EIC sends with its request (SRSCtl.EICSS)
        self.eic_ss: int = 0
        #: the timer / performance counter interrupts go out to an
        #: interrupt controller instead of being wired to IntCtl.IPTI /
        #: IPPCI (a CPS core whose GIC routes them, emips.gic)
        self.local_routed: bool = False
        #: called when the interrupt requests the core sends out change:
        #: Cause.TI, Cause.PCI, Cause.IP0, IP1 (the GIC's local inputs)
        self.on_irq_outputs: Callable[[], None] | None = None
        self._irq_out: tuple = (False, False, False, False)
        #: a data bus error waiting to be taken (imprecise: a CM error
        #: response, emips.coherence; a read the board decodes as nothing)
        self.bus_error_pending: bool = False
        #: the board (emips.malta) — None: every physical address is RAM
        self.board = None
        #: an LL is loading (a coherent CCA 4 load then asks for Shared)
        self._linked: bool = False
        #: a Non Maskable Interrupt is waiting (an edge on the NMI input:
        #: a CPS core's GIC, emips.gic)
        self.nmi_pending: bool = False
        #: True after WAIT until an interrupt is taken.
        self.waiting: bool = False
        #: EJTAG (emips.ejtag): the debug unit (drseg, dmseg), Debug.DM,
        #: Debug.SSt, a Debug Single Step due (an execution step completed
        #: with SSt = 1), a debug interrupt request (the DINT input)
        self.ejtag = EJ.EJTAG(self)
        #: MIPS Trace (emips.pdtrace): the PDtrace module; what the last
        #: executed instruction leaves for its completion cycle; the
        #: accesses / breakpoint triggers of the instruction being traced;
        #: whether the last fetch missed the I-cache (the Im bit)
        self.pdt = PDT.PDtrace(self)
        self.pdt_packet: PDT.Packet | None = None
        self._pdt_acc: list | None = None
        self._trig: list = []
        self.fetch_missed = False
        self._dm: bool = False
        self._sst: bool = False
        self.ss_pending: bool = False
        self.dint_pending: bool = False
        #: Debug.DBusEP / CacheEP set: an imprecise error is held
        self._ep: bool = False
        #: hardware breakpoints / watch registers armed for instructions /
        #: data (debug_hw_changed); the value compares an access still
        #: owes (mask, lanes, bus width); imprecise data breaks due
        #: (Debug.DDBLImpr / DDBSImpr bits)
        self._ihw: bool = False
        self._dhw: bool = False
        #: a debug agent's watchpoints (the GDB stub; objects with addr,
        #: length, load, store), checked before every load / store — not
        #: hardware: the program cannot see them
        self.agent_watch: list = []
        self._vc: tuple | None = None
        self.ddb_impr: int = 0
        #: a WAIT was ended to take the exception being taken (wake)
        self.slept: bool = False
        #: the last WAIT (a PC sample while asleep)
        self.wait_pc: int = 0
        #: what the last CACHE instruction did (for the views)
        self.last_cache_op: str = ""
        #: code the boot firmware put in memory (emips.assembler.Program,
        #: e.g. the TLB refill handler) or None
        self.firmware = None

    # -- registers -----------------------------------------------------------

    def read_reg(self, index: int) -> int:
        index &= 31
        return 0 if index == 0 else self.regs[index] & M32

    def write_reg(self, index: int, value: int) -> None:
        index &= 31
        if index != 0:
            self.regs[index] = value & M32

    def cp0_read(self, reg: int, sel: int = 0) -> int:
        return self.cp0[(reg, sel)]["value"]

    def cp0_write(self, reg: int, sel: int = 0, value: int = 0) -> None:
        """Software write (``MTC0``): only the writable bits change.  In
        Release 6 a write of an unsupported cacheability attribute to
        ``Config.K0 / KU / K23`` leaves that field unchanged (Vol. III
        Config); before R6 using one is UNDEFINED and it is just stored."""
        if (reg, sel) == (10, 0):               # EntryHi: a TMOAS with a TLB (3.2)
            self.pdt.entryhi_written()
        if (reg, sel) in self._mmu_regs:
            self.mmu.cp0_write(self, (reg, sel), value)
            return
        if reg == 25 and (reg, sel) in perf.REGISTERS:
            self.perf.cp0_write((reg, sel), value)
            return
        cell = self.cp0[(reg, sel)]
        mask = self._wmask.get((reg, sel), M32)
        if reg == 19 and sel < C.WATCH_PAIRS:   # WatchHi I / R / W: write one to clear
            cell["value"] &= ~(value & 7)
        if (reg, sel) == C.DEBUG:               # CacheEP / DBusEP: write 1 to set
            cell["value"] |= value & EJ.W1_BITS
            self._sst = bool(value & EJ.SST)
            self._ep = bool(cell["value"] & EJ.W1_BITS)
        if (reg, sel) == C.CONFIG and self.release == "R6":
            for shift in (0, 25, 28):
                if (value >> shift) & 7 not in self.supported_cca:
                    mask &= ~(7 << shift)
        old = cell["value"]
        if (reg, sel) in (C.SRSCTL, C.SRSMAP):  # a set above HSS: UNDEFINED
            shifts = ((C.SRS_ESS, C.SRS_PSS) if (reg, sel) == C.SRSCTL
                      else range(0, 32, 4))     # -> that field keeps its value
            for shift in shifts:
                if (value >> shift) & 0xF > self.hss:
                    mask &= ~(0xF << shift)
        if (reg, sel) == C.INTCTL and (value >> 5) & 0x1F not in C.VS_VALUES:
            mask &= ~(0x1F << 5)                # a reserved VS: UNDEFINED -> ignored
        cell["value"] = (cell["value"] & ~mask | value & mask) & M32
        if (reg, sel) == C.STATUS:              # NMI / SR: software can only clear
            cell["value"] &= ~((C.ST_NMI | C.ST_SR) & old & ~value)
        if (reg, sel) == C.ERRCTL and self.caches is not None:
            self.caches.set_checking(cell["value"])
        if (reg, sel) == C.CONFIG2 and self.caches is not None:
            self.caches.set_config2(cell["value"])
        if (reg, sel) in (C.CAUSE, C.STATUS, C.INTCTL):   # IP1..0, the interrupt mode
            self._update_ip()
        if reg == 18 and sel < C.WATCH_PAIRS:   # WatchLo I / R / W: what to check
            self.debug_hw_changed()
        if reg == 23 and 1 <= sel <= 5:         # MIPS Trace
            if (reg, sel) == PDT.USERTRACEDATA:
                self.pdt.user_data(cell["value"])
            self.trace_settings_changed()

    def _cp0_set(self, key: tuple[int, int], value: int) -> None:
        """Hardware write (exception entry, ERET, DI/EI): no mask."""
        self.cp0[key]["value"] = value & M32

    @property
    def status(self) -> int:
        return self.cp0[C.STATUS]["value"]

    # -- shadow register sets (Release 2, Vol. III 9.31) ----------------------

    def srs_field(self, shift: int) -> int:
        """A 4-bit SRSCtl field (``C.SRS_CSS``, ``SRS_PSS``, ``SRS_ESS`` ...)."""
        return (self.cp0[C.SRSCTL]["value"] >> shift) & 0xF

    def _set_css(self, new: int, *, save: bool) -> None:
        """Hardware changes the current set: exception entry (``save``:
        PSS ← CSS first) or ERET (CSS ← PSS)."""
        srs = self.cp0[C.SRSCTL]["value"]
        if save:
            srs = (srs & ~(0xF << C.SRS_PSS)) | ((srs & 0xF) << C.SRS_PSS)
        self.cp0[C.SRSCTL]["value"] = (srs & ~0xF) | new
        self.regs = self.shadow[new]

    def interrupt_mode(self) -> str:
        """"compatibility", "vectored" or "eic" (Vol. III Table 6.1)."""
        return C.interrupt_mode(self.status, self.cp0_read(*C.CAUSE),
                                self.cp0_read(*C.INTCTL), self.cp0_read(*C.CONFIG3))

    @property
    def kernel_mode(self) -> bool:
        """Kernel mode — or Debug Mode, which has its rights (MD00047 2.2)."""
        return self._dm or C.kernel_mode(self.status)

    @property
    def debug_mode(self) -> bool:
        """EJTAG Debug Mode (``Debug.DM``)."""
        return self._dm

    def debug_interrupt(self) -> None:
        """A request on the DINT input (or the probe's EjtagBrk): a Debug
        Interrupt exception as soon as possible — ignored in Debug Mode
        (MD00047 2.3.10)."""
        if not self._dm:
            self.dint_pending = True

    @property
    def debug_due(self) -> bool:
        """A debug exception waits for the next instruction (single step,
        debug interrupt, imprecise data break): the 5-stage pipeline does
        not resolve a branch in ID then (it is taken on the branch)."""
        return self.ss_pending or bool(self.ddb_impr) or (self.dint_pending and not self._dm)

    def trace_settings_changed(self) -> None:
        """A trace control changed (TraceControl*, the TCB): TraceControl2's
        TCB fields follow (SyP from TCBCONTROLA, TBU = TCBCONTROLB.OfC,
        TBI = 0: one trace buffer) and PDtrace rechecks its work."""
        t = self.pdt.tcb
        cell = self.cp0[PDT.TRACECONTROL2]
        syp = (t.control_a >> PDT.TA_SYP_SHIFT) & 7 if t is not None else 0
        tbu = int(bool(t.control_b & PDT.TB_OFC)) if t is not None else 0
        cell["value"] = (cell["value"] & ~0x1F) | (tbu << 3) | syp
        self.pdt.refresh()

    def debug_hw_changed(self) -> None:
        """A breakpoint or watch register changed: what to check (kept
        in flags, so an unarmed core pays one test per access)."""
        lo = [self.cp0[k]["value"] for k in C.WATCHLO]
        self._ihw = self.ejtag.ib_armed or any(v & C.WATCH_I for v in lo)
        self._dhw = self.ejtag.db_armed or any(v & (C.WATCH_R | C.WATCH_W) for v in lo)

    def _asid(self) -> int:
        return self.cp0[(10, 0)]["value"] & 0xFF

    def _watch(self, start: int, n: int, kind: int) -> bool:
        """The watch registers (Vol. III 9.51 / 9.52; the 24K's 4 pairs):
        a pair whose enable ``kind`` (I / R / W) is set matches when a
        byte of the access lies in its doubleword (VAddr 31:3, Mask 11:3
        bits ignored) and G = 1 or its ASID is EntryHi's; WatchHi records
        the kind.  Returns True if a Watch exception is taken now; with
        EXL or ERL set it is deferred instead (Cause.WP)."""
        hit = False
        end = (start + n - 1) & M32
        for lo_key, hi_key in zip(C.WATCHLO, C.WATCHHI):
            lo = self.cp0[lo_key]["value"]
            if not lo & kind:
                continue
            hi = self.cp0[hi_key]["value"]
            if not hi & C.WATCHHI_G and (hi >> 16) & 0xFF != self._asid():
                continue
            care = ~(hi & 0xFF8 | 7) & M32
            if (start ^ lo) & care and (end ^ lo) & care:
                continue
            self.cp0[hi_key]["value"] = hi | kind
            hit = True
        if not hit:
            return False
        if self.status & (C.ST_EXL | C.ST_ERL):
            self._cp0_set(C.CAUSE, self.cp0_read(*C.CAUSE) | C.CA_WP)
            return False
        return True

    def instruction_checks(self, pc: int) -> None:
        """Before an instruction executes in Non-Debug Mode (MD00343
        Table 5.1): an instruction hardware breakpoint (DIB; the BS bits
        of every matching break / triggerpoint), then a fetch watch.
        Evaluated on the PC even when the fetch itself will fault."""
        if self._dm:
            return
        ej = self.ejtag
        hits, brk = ej.instruction_match(pc, self._asid()) if ej.ib_armed else (0, False)
        if hits:
            ej.ibs_bs |= hits
            self._trace_triggers(hits, False)
        if brk:
            raise _Arch(C.DIB, info={"why": f"instruction breakpoint (IBS.BS 0x{hits:x})"})
        if self._watch(pc, 4, C.WATCH_I):
            raise _Arch(C.WATCH, info={"why": "instruction fetch"})

    def _trace_triggers(self, hits: int, data: bool) -> None:
        """Hardware breakpoints used as trace triggers (TE): TraceIBPC /
        TraceDBPC (IE / DE) say start (001) or stop (000) for each
        (MD00343 6.2.32 / 6.2.33, 11.10.1)."""
        ej = self.ejtag
        bpc = self.cp0[PDT.TRACEDBPC if data else PDT.TRACEIBPC]["value"]
        if not bpc & PDT.IBPC_IE:
            return
        ctl = ej.dbc if data else ej.ibc
        for n in range(len(ctl)):
            if hits >> n & 1 and ctl[n] & EJ.BC_TE:
                code = (bpc >> (3 * n)) & 7
                if code in (0, 1):
                    self._trig.append((int(data), code, n))

    def _data_checks(self, addr: int, size: int, store: bool,
                     span: tuple[int, int] | None) -> None:
        """A load / store in Non-Debug Mode (MD00047 5.3.2, MD00343
        11.2.5.2 / Table 5.1): data breakpoints without a value compare
        are precise (DDBL / DDBS before any address error or TLB
        exception: the access does not happen); those with one are
        compared when the data is on the bus (load / store) and are
        imprecise on the 24K; then a data watch.  ``span``: the bytes an
        LWL / LWR / SWL / SWR really accesses."""
        self._vc = None
        ej = self.ejtag
        if ej.db_armed:
            width = 8 if size == 8 else 4
            start, n = span if span is not None else (addr & ~(size - 1), size)
            lanes = 0
            for a in range(start, start + n):
                k = a % width
                lanes |= 1 << (width - 1 - k if self.big_endian else k)
            lanes &= (1 << width) - 1
            precise, brk, value = ej.data_match(addr, lanes, width, store, self._asid())
            if precise:
                ej.dbs_bs |= precise
                self._trace_triggers(precise, True)
            if brk:                             # Table 11.6: a load's value compares unknown
                raise _Arch(C.DDBS if store else C.DDBL,
                            info={"why": f"data breakpoint (DBS.BS 0x{precise:x})"})
            if value:
                self._vc = (value, lanes, width)
        start, n = span if span is not None else (addr, size)
        if self._watch(start, n, C.WATCH_W if store else C.WATCH_R):
            raise _Arch(C.WATCH, info={"why": "store" if store else "load"})

    def _value_checks(self, addr: int, size: int, value: int, store: bool) -> None:
        """The data value of an access the breakpoints are waiting for
        (DB_value_match on the data bus): matches set BS; a breaking one
        makes an imprecise debug exception due (Debug.DDBLImpr / DDBSImpr)."""
        mask, lanes, width = self._vc
        self._vc = None
        k = addr % width
        shift = 8 * (width - size - k) if self.big_endian else 8 * k
        data = (value << shift) & ((1 << 8 * width) - 1) if shift >= 0 else value
        ej = self.ejtag
        hits = brk = 0
        for n in range(EJ.DB_COUNT):
            if mask >> n & 1 and ej.value_match(n, lanes, width, data):
                hits |= 1 << n
                brk |= ej.dbc[n] & EJ.BC_BE
        ej.dbs_bs |= hits
        if hits:
            self._trace_triggers(hits, True)
        if brk:
            self.ddb_impr |= EJ.DDBSIMPR if store else EJ.DDBLIMPR

    def _trace_cycle(self, completed: bool, *, asleep: bool = False) -> None:
        """The cycle's PDtrace output (``completed``: the instruction of
        :attr:`pdt_packet` completed); with Inhibit Overflow a full FIFO
        stalls the core a cycle at a time (11.12.2)."""
        packet, self.pdt_packet = (self.pdt_packet if completed else None), None
        pdt = self.pdt
        pdt.clock(packet, asleep=asleep)
        while pdt.backpressure:
            self.advance(1)
            pdt.clock(None)

    def wake(self) -> None:
        """An interrupt or exception ends a WAIT (the debug exception
        entry still sees that the core slept: Debug.Halt / Doze)."""
        self.slept = self.slept or self.waiting
        self.waiting = False

    def debug_vector(self) -> int:
        """The debug exception vector (MD00047 Table 2.9; MD00343 Tables
        5.5 / 5.6): 0xBFC00480, a CPS core's reset base + 0x480; in the
        probe's memory (0xFF200200) with ECR.ProbTrap = 1."""
        if self.ejtag.prob_trap:
            return EJ.PROBE_VECTOR
        return (self.bev_base + EJ.DEBUG_VECTOR_OFFSET) & M32

    # -- a core of a CPS (emips.cps) -------------------------------------------

    def join_system(self, cps) -> None:
        """Make this machine a core of ``cps``: the coherent attributes 4 / 5
        (Config.K0 / KU / K23, TLB pages), Config3.CMGCR and CMGCRBase (CP0
        15.3: the GCRs' physical address bits 35:15 in 29:11, read-only)."""
        self.system = cps
        self.supported_cca = self.supported_cca | COHERENT_CCA
        self.cached_cca = CACHED_CCA_CPS
        self.cp0[C.CONFIG3]["value"] |= C.CONFIG3_CMGCR
        self.cmgcr_moved(cps.gcr.base)

    def cmgcr_moved(self, base: int) -> None:
        """GCR_BASE changed: CMGCRBase follows it."""
        self.cp0[C.CMGCRBASE]["value"] = (base >> 4) & 0x3FFFF800

    # -- clock, timer, interrupt lines ----------------------------------------

    def advance(self, cycles: int = 1) -> None:
        """Let ``cycles`` clock cycles pass: ``Count`` increments every
        :attr:`ccres` cycles (unless ``Cause.DC``), and reaching
        ``Compare`` raises the timer interrupt (``Cause.TI`` → IP7)."""
        self.cycles += cycles
        for dev in self.devices:
            dev.tick(self, cycles)
        if self.caches is not None and self.caches.fsb:
            self.caches.tick()                  # waiting stores whose line came
        if self._mmu_tick is not None:
            self._mmu_tick(self, cycles)
        if self.ejtag.tap is not None:          # PC sampling (EJTAG ch. 7)
            self.ejtag.tick(cycles)
        if self.perf.armed:
            self.perf.tick(cycles)
        if self.cp0_read(*C.CAUSE) & C.CA_DC:
            return
        if self._dm and not self.cp0[C.DEBUG]["value"] & EJ.COUNTDM:
            return                              # Count stopped in Debug Mode
        self._count_phase += cycles
        count = self.cp0_read(*C.COUNT)
        compare = self.cp0_read(*C.COMPARE)
        hit = False
        while self._count_phase >= self.ccres:
            self._count_phase -= self.ccres
            count = (count + 1) & M32
            hit = hit or count == compare
        self._cp0_set(C.COUNT, count)
        if hit:
            self._cp0_set(C.CAUSE, self.cp0_read(*C.CAUSE) | C.CA_TI)
            self._update_ip()

    def nmi(self) -> None:
        """An edge on the NMI input: taken before the next instruction,
        whatever Status says (Vol. III "Non Maskable Interrupt")."""
        self.nmi_pending = True

    def interrupt_request(self) -> "_Arch | None":
        """What the core takes before its next instruction (MD00343 Table
        5.1): a Debug Single Step, a Debug Interrupt, an NMI, or an
        enabled interrupt (``Status.IE``, ``EXL = ERL = 0``; IM & IP, or in
        EIC mode the controller's RIPL above ``Status.IPL``).  In Debug
        Mode none (MD00047 2.5); the DCR's IntE / NMIE mask interrupts /
        NMIs in Non-Debug Mode (the NMI stays pending)."""
        if self.ss_pending:
            return _Arch(C.DSS)
        if self._dm:
            return None
        if self.dint_pending:
            return _Arch(C.DINT)
        if self.ddb_impr:                       # an imprecise data break
            return _Arch(C.DDBLIMPR if self.ddb_impr & EJ.DDBLIMPR else C.DDBSIMPR)
        dcr = self.ejtag.dcr
        if self.nmi_pending and dcr & EJ.DCR_NMIE:
            return _Arch(C.NMI)
        status = self.status
        cause = self.cp0_read(*C.CAUSE)
        if dcr & EJ.DCR_INTE:
            if self.interrupt_mode() == "eic":
                if (status & C.ST_IE and not status & (C.ST_EXL | C.ST_ERL)
                        and self.irq_lines > (status >> C.ST_IM_SHIFT + 2) & 0x3F):
                    return _Arch(C.INT)
            elif C.interrupt_pending(status, cause):
                return _Arch(C.INT)
        if cause & C.CA_WP and not status & (C.ST_EXL | C.ST_ERL):
            return _Arch(C.WATCH, info={"why": "deferred (Cause.WP)"})
        return None

    def set_irq_lines(self, value: int, eic_ss: int = 0) -> None:
        """An interrupt controller drives the six interrupt lines at once:
        HW0..HW5 as a bit mask, or in EIC mode the encoded RIPL with the
        shadow set for it (Vol. III 6.1.1.3)."""
        self.irq_lines = value & 0x3F
        self.eic_ss = eic_ss & 0xF
        self._update_ip()

    def set_eic_present(self, present: bool) -> None:
        """The SI_EICPresent input (Config3.VEIC, read-only; a CPS core:
        GIC_VPEi_CTL.EIC_MODE).  Release 2+."""
        if self.release == "R1":
            return
        cfg3 = self.cp0_read(*C.CONFIG3)
        self._cp0_set(C.CONFIG3, cfg3 | C.CONFIG3_VEIC if present else cfg3 & ~C.CONFIG3_VEIC)
        self._update_ip()

    def assert_irq(self, line: int) -> None:
        """Raise hardware interrupt line HW0..HW5 (→ ``Cause.IP2..IP7``)."""
        if not 0 <= line <= 5:
            raise ValueError(f"hardware interrupt line must be 0..5, not {line}")
        self.irq_lines |= 1 << line
        self._update_ip()

    def clear_irq(self, line: int) -> None:
        if not 0 <= line <= 5:
            raise ValueError(f"hardware interrupt line must be 0..5, not {line}")
        self.irq_lines &= ~(1 << line)
        self._update_ip()

    def _update_ip(self) -> None:
        """Recompute the read-only ``Cause.IP7..IP2`` from the lines, TI
        and the performance counter interrupt (both on IP7, unless an
        interrupt controller routes them); in EIC mode those bits are
        RIPL, loaded only when an interrupt is taken.  Then tell the
        controller what the core requests (TI, PCI, IP1..0)."""
        cause = self.cp0_read(*C.CAUSE)
        if self.interrupt_mode() != "eic":
            hw = self.irq_lines
            if not self.local_routed and (cause & C.CA_TI or self.perf.pci):
                hw |= 1 << 5
            cause = (cause & ~(0x3F << (C.CA_IP_SHIFT + 2))) | (hw << (C.CA_IP_SHIFT + 2))
            self._cp0_set(C.CAUSE, cause)
        out = (bool(cause & C.CA_TI), self.perf.pci, bool(cause & 0x100), bool(cause & 0x200))
        if out != self._irq_out:
            self._irq_out = out
            if self.on_irq_outputs is not None:
                self.on_irq_outputs()

    def irq_outputs(self) -> tuple[bool, bool, bool, bool]:
        """What the core sends an interrupt controller: (timer, performance
        counter, software 0, software 1) requests."""
        return self._irq_out

    # -- memory --------------------------------------------------------------

    def translate(self, addr: int) -> Translation:
        """How the MMU translates the virtual address *addr* right now
        (physical address — ``None`` when no TLB entry maps it —,
        cacheability attribute, segment, TLB entry); no side effects.  dseg in Debug Mode: no
        physical address (MD00047 Table 2.2)."""
        addr &= M32
        if self._dm and EJ.in_dseg(addr):
            return Translation(addr, None, CCA_UNCACHED, EJ.segment(addr), False,
                               "EJTAG " + ("debug registers" if addr >= EJ.DRSEG_BASE
                                           else "probe memory") + ": no physical address")
        return self.mmu.describe(self, addr)

    def _peek(self, addr: int) -> tuple[int, int]:
        """``(pa, cca)`` of a virtual address for a view (no side effects);
        :class:`UnmappedAddress` when nothing maps it (dseg in Debug
        Mode: no physical address)."""
        if self._dm and EJ.in_dseg(addr & M32):
            raise UnmappedAddress(f"0x{addr & M32:08x} is EJTAG {EJ.segment(addr & M32)} "
                                  "(Debug Mode): no physical address",
                                  name=None, word=None, addr=addr & M32)
        hit = self.mmu.peek(self, addr & M32)
        if hit is None:
            raise UnmappedAddress(f"no TLB entry maps 0x{addr & M32:08x} now",
                                  name=None, word=None, addr=addr & M32)
        return hit

    def physical(self, addr: int) -> int:
        """The physical address of the virtual address *addr* (views and
        the loader: no side effects; :class:`UnmappedAddress` if none)."""
        return self._peek(addr)[0]

    def _access(self, addr: int, size: int, kind: str) -> tuple[list[int], int]:
        """Translate an access (``kind``: fetch / load / pcload / store /
        cacheop / pref): the physical address of each byte and the
        cacheability attribute, or the MMU's exception (TLB).  Within one
        1 KB block (the smallest MIPS page) one translation serves all
        bytes; an access crossing it is translated byte by byte."""
        addr &= M32
        pa, cca = self._xlate(self, addr, kind)
        if (addr & 0x3FF) + size <= 0x400:
            return [pa + i for i in range(size)], cca
        return [pa] + [self._xlate(self, (addr + i) & M32, kind)[0]
                       for i in range(1, size)], cca

    def _view_pas(self, addr: int, size: int) -> list[int]:
        first = self._peek(addr)[0]
        if (addr & 0x3FF) + size <= 0x400:
            return [first + i for i in range(size)]
        return [self.physical(addr + i) for i in range(size)]

    def phys_read_byte(self, pa: int) -> int:
        """Byte at physical address *pa*; never-written bytes read **0**."""
        raw = self.mem.read_from_memory(pa & M32)
        return 0 if raw is None else int(raw, 16)

    def phys_write_byte(self, pa: int, value: int) -> None:
        self.mem.write_to_memory(pa & M32, format(value & 0xFF, "02x"))

    def _cache_arch(self, err: CacheError) -> _Arch:
        """The Cache Error exception for a parity error found in a cache."""
        return _Arch(C.CACHE_ERROR, info={"cacheerr": err.cacheerr(), "what": err.what,
                                          "cache": err.cache.kind})

    def _cca(self, addr: int) -> int:
        """Cacheability attribute of *addr* now (views; uncached if unmapped)."""
        hit = self.mmu.peek(self, addr & M32)
        return CCA_UNCACHED if hit is None else hit[1]

    def _uses_dcache(self, cca: int) -> bool:
        c = self.caches
        return c is not None and c.cacheable and cca in self.cached_cca

    def _cached(self, addr: int) -> bool:
        """Does a data access at *addr* go through the caches now?"""
        return self._uses_dcache(self._cca(addr))

    def _view_byte(self, addr: int, pa: int, cached: bool | None = None) -> int:
        """The byte a load at *addr* (physical *pa*) would return now,
        without side effects (the caches, the write buffer, memory)."""
        if self.caches is not None:
            if cached is None:
                cached = self._cached(addr)
            coherent = cached and self.caches.cm is not None and \
                self.caches.coherent(self._cca(addr))
            byte = self.caches.view_data(addr & M32, pa, cached, coherent)
            return 0 if byte is None else byte
        if self.board is not None and self.board.bus.covers(pa):
            return self._board_peek(pa, 1)
        return self.phys_read_byte(pa)

    def _board_peek(self, pa: int, size: int) -> int:
        """A view of a board address (no side effects); nothing there: a
        :class:`BusError`."""
        value = self.board.bus.peek(self, pa, size)
        if value is None:
            raise BusError(pa, "view")
        return value

    def read_byte(self, addr: int) -> int:
        """Data byte at virtual address *addr* as a load would see it now
        (the data cache's copy when it holds the line), with no side
        effects; unwritten bytes read **0** (not an error); an address no
        TLB entry maps raises :class:`UnmappedAddress`."""
        addr &= M32
        if self._dm and EJ.in_dseg(addr) and not self.cp0[C.DEBUG]["value"] & EJ.LSNM:
            byte = self.ejtag.peek(addr, 1)
            if byte is not None:
                return byte
        pa, cca = self._peek(addr)
        return self._view_byte(addr, pa, self._uses_dcache(cca))

    def write_byte(self, addr: int, value: int) -> None:
        """Write the byte at virtual address *addr* as a debugger or
        loader does: into memory and into every cached or buffered copy
        of it (no line is allocated, no dirty bit changes)."""
        pa = self.physical(addr)
        self.phys_write_byte(pa if self.board is None else self.board.storage(pa), value)
        if self.caches is not None:
            self.caches.backdoor_write(pa, value)

    def _log(self, kind: str, addr: int, *entry: int) -> None:
        self._mem_log.append((kind, addr & M32, *entry, self.translate(addr)))

    def load(self, addr: int, size: int, kind: str = "load") -> int:
        """``size`` bytes at virtual address *addr* as an unsigned value, in
        the machine's byte order (no alignment / access checks); through
        the data cache when the address is cacheable.  ``kind = "pcload"``
        for a PC-relative load (RI / XI rules, Vol. III 4.9.4).  In Debug
        Mode dseg goes to the EJTAG unit (unless ``Debug.LSNM``)."""
        if self._dm and EJ.in_dseg(addr & M32) and not self.cp0[C.DEBUG]["value"] & EJ.LSNM:
            value = self.ejtag.load(addr & M32, size)
            if self._mem_log is not None:
                self._log("read", addr, size, value)
            return value
        pas, cca = self._access(addr, size, kind)
        if self.perf.armed and cca not in self.cached_cca:
            self.perf.pending.append("uncached_load")
        if self._vc is None and self._pdt_acc is None:
            return self.load_pa(addr, pas, cca, size)
        value = self.load_pa(addr, pas, cca, size)
        if self._vc is not None:
            self._value_checks(addr & M32, size, value, False)
        if self._pdt_acc is not None:               # MIPS Trace: the load
            self._pdt_acc.append(("load", addr & M32, size, value))
        return value

    def load_pa(self, addr: int, pas: list[int], cca: int, size: int) -> int:
        """The data part of a load whose translation is done: the bytes at
        physical ``pas`` (virtual ``addr``) with attribute ``cca`` — through
        the data cache, the write buffer or a device, like any load.  Also
        the hardware page table walker's reads (emips.ptw)."""
        c = self.caches
        if self.system is not None and cca != CCA_UNCACHED and self.system.registers_at(pas[0]) \
                and self.system.register_error(self, pas[0], cca, False):
            return 0                            # a line fill of a register: a CM error
        if c is not None and c.cacheable and cca in self.cached_cca \
                and not (self.system is not None and self.system.registers_at(pas[0])):
            try:
                data = c.read(addr & M32, pas, size, cca)
            except CacheError as err:
                raise self._cache_arch(err) from None
            except BusError:                    # the refill: a data bus error
                return self._read_bus_error(addr, size)
            value = int.from_bytes(data, "big" if self.big_endian else "little")
            if self._mem_log is not None:
                self._log("read", addr, size, value)
            return value
        if c is not None:                       # write buffer before the read
            c.read_request(pas[0], size, "load", no_cache=cca in self.cached_cca)
        for dev in self.devices:
            if dev.covers(pas[0]):
                try:
                    value = dev.load(self, pas[0], size)
                except BusError:
                    return self._read_bus_error(addr, size)
                if self._mem_log is not None:
                    self._log("read", addr, size, value)
                return value
        data = [self.phys_read_byte(pa) for pa in pas]
        if not self.big_endian:
            data.reverse()
        value = 0
        for byte in data:
            value = (value << 8) | byte
        if self._mem_log is not None:
            self._log("read", addr, size, value)
        return value

    def _read_bus_error(self, addr: int, size: int) -> int:
        """A read the board decodes as nothing: the Data Bus Error is
        imprecise on the 24K (MD00343 5.8.13: taken before the next
        instruction; the target register may be corrupted — eMIPS loads
        0)."""
        self.bus_error_pending = True
        if self._mem_log is not None:
            self._log("read", addr, size, 0)
        return 0

    def store(self, addr: int, value: int, size: int, *, gather: bool = True,
              atomic: bool = False) -> None:
        """Store the low ``size`` bytes of *value* at virtual address
        *addr*, machine order; through the data cache when cacheable, the
        write buffer when there is one (``gather = False``: a partial or
        conditional store, never gathered as uncached accelerated)."""
        if self._dm and EJ.in_dseg(addr & M32) and not self.cp0[C.DEBUG]["value"] & EJ.LSNM:
            value &= (1 << (8 * size)) - 1
            if self._mem_log is not None:
                self._log("write", addr, size, self.ejtag.peek(addr & M32, size) or 0, value)
            self.ejtag.store(addr & M32, value, size)
            return
        pas, cca = self._access(addr, size, "store")
        if self.perf.armed and cca not in self.cached_cca:
            self.perf.pending.append("uncached_store")
        value &= (1 << (8 * size)) - 1
        if self._vc is not None:
            self._value_checks(addr & M32, size, value, True)
        if self._pdt_acc is not None:               # MIPS Trace: the store
            self._pdt_acc.append(("store", addr & M32, size, value))
        if self.system is not None:
            if self.system.store_check(self, pas[0], size, cca, value):
                return                          # a register error / an L2-only SYNC
        c = self.caches
        cached = self._uses_dcache(cca)
        if c is not None and not cached:
            order = "big" if self.big_endian else "little"
            old = None
            if self._mem_log is not None:
                old = bytes(self._view_byte((addr + i) & M32, pa, False)
                            for i, pa in enumerate(pas))
            if c.uncached_store(pas[0], value.to_bytes(size, order), cca, gather):
                if old is not None:
                    self._log("write", addr, size, int.from_bytes(old, order), value)
                return
        if cached:
            order = "big" if self.big_endian else "little"
            old = None
            if self._mem_log is not None:
                old = bytes(self._view_byte((addr + i) & M32, pa, True)
                            for i, pa in enumerate(pas))
            try:
                c.write(addr & M32, pas, value.to_bytes(size, order), cca, atomic)
            except CacheError as err:
                raise self._cache_arch(err) from None
            except BusError:                    # a write-allocate refill: DBE, the
                self.bus_error_pending = True   # store data lost (24K ErrCtl.WABE)
                return
            if old is not None:
                self._log("write", addr, size, int.from_bytes(old, order), value)
            return
        for dev in self.devices:
            if dev.covers(pas[0]):
                if self._mem_log is not None:
                    self._log("write", addr, size, 0, value)
                dev.store(self, pas[0], value, size)
                return
        if self._mem_log is not None:
            old = [self.phys_read_byte(pa) for pa in pas]
            if not self.big_endian:
                old.reverse()
            self._log("write", addr, size, int.from_bytes(bytes(old), "big"), value)
        for i, pa in enumerate(pas):
            shift = 8 * (size - 1 - i) if self.big_endian else 8 * i
            self.phys_write_byte(pa, value >> shift)

    def read_word(self, addr: int) -> int:
        return self.load(addr, 4)

    def write_word(self, addr: int, value: int) -> None:
        self.store(addr, value & M32, 4)

    def write_bytes(self, addr: int, value: int, count: int) -> None:
        self.store(addr, value, count)

    def peek_word(self, addr: int) -> int:
        """The word at virtual address *addr* as a load would see it, for
        the views: no cache side effects (device registers are still read
        through the device, as before)."""
        addr &= M32
        if self._dm and EJ.in_dseg(addr) and not self.cp0[C.DEBUG]["value"] & EJ.LSNM:
            word = self.ejtag.peek(addr, 4)
            if word is not None:
                return word
        pas = self._view_pas(addr, 4)
        cached = self._cached(addr)
        if not cached:
            if self.board is not None and self.board.bus.covers(pas[0]):
                if not any(d is not self.board.bus and d.covers(pas[0]) for d in self.devices):
                    return self._board_peek(pas[0], 4)
            for dev in self.devices:
                if dev.covers(pas[0]):
                    return dev.load(self, pas[0], 4)
        data = bytes(self._view_byte((addr + i) & M32, pa, cached) for i, pa in enumerate(pas))
        return int.from_bytes(data, "big" if self.big_endian else "little")

    @staticmethod
    def _no_code(addr: int, i: int, pa: int) -> FetchError:
        va = (addr + i) & M32
        where = f"0x{va:08x}" if pa == va else f"0x{va:08x} (physical 0x{pa:08x})"
        return FetchError(f"code fetch failed: no byte at {where}",
                          name=None, word=None, addr=addr)

    def _icached(self, cca: int) -> bool:
        """Does an instruction fetch with attribute *cca* go through the
        I-cache?  (Write-through pages too: the 24K I-cache always fills
        as write-back, MD00343 8.5.2.)"""
        c = self.caches
        return (c is not None and (c.i is not None or c.l2 is not None)
                and cca in self.cached_cca)

    def fetch_word(self, addr: int) -> int:
        """Fetch the instruction word at virtual address *addr* (through the
        instruction cache when cacheable); **any** unwritten byte is a
        :class:`FetchError` (simulator halt).  While :attr:`tracing`, the
        cache accesses it made are left in :attr:`fetch_accesses`, its
        translations (TLB / BAT) in :attr:`fetch_tlb`."""
        if self.pdt.active and self.caches is not None and self.caches.i is not None:
            stats = self.caches.i.stats             # the Im bit of MIPS Trace
            before = stats.misses
            try:
                return self._fetch_word(addr) if not self.tracing else \
                    self._fetch_watched(addr)
            finally:
                self.fetch_missed = stats.misses != before
        if not self.tracing:
            return self._fetch_word(addr)
        return self._fetch_watched(addr)

    def _fetch_watched(self, addr: int) -> int:
        c = self.caches
        cw = c is not None and c.log is None
        mmu = self.mmu if hasattr(self.mmu, "watch") and self.mmu.log is None else None
        if cw:
            c.watch()
        if mmu is not None:
            mmu.watch()
        try:
            return self._fetch_word(addr)
        finally:
            if cw:
                self.fetch_accesses = c.taken()
            if mmu is not None:
                self.fetch_tlb = mmu.taken()

    def _fetch_word(self, addr: int) -> int:
        addr &= M32
        if self._dm and EJ.in_dseg(addr):       # a fetch in dseg: always dseg
            return self.ejtag.fetch(addr)
        pas, cca = self._access(addr, 4, "fetch")
        c = self.caches
        if c is not None and self._icached(cca):
            try:
                data, known = c.fetch(addr, pas[0], cca)
            except CacheError as err:
                raise self._cache_arch(err) from None
            except BusError:                    # IBE: precise (MD00343 5.8.13)
                raise _Arch(C.IBE, info={"why": "instruction fetch bus error"}) from None
            if known != 0xF:
                i = next(k for k in range(4) if not known >> k & 1)
                raise self._no_code(addr, i, pas[0] + i)
            return int.from_bytes(data, "big" if self.big_endian else "little")
        if c is not None:                       # write buffer before the read
            c.read_request(pas[0], 4, "fetch", no_cache=cca in self.cached_cca)
        if self.board is not None and self.board.bus.covers(pas[0]):
            try:
                return self.board.bus.load(self, pas[0], 4)
            except BusError:                    # IBE: precise (MD00343 5.8.13)
                raise _Arch(C.IBE, info={"why": "instruction fetch bus error"}) from None
        mem = self.mem.get_memory()
        raw = [mem.get(format(pa, "x")) for pa in pas]
        for i, (pa, byte) in enumerate(zip(pas, raw)):
            if byte is None:
                raise self._no_code(addr, i, pa)
        for dev in self.devices:
            if dev.covers(pas[0]):
                return dev.load(self, pas[0], 4)
        if not self.big_endian:
            raw.reverse()
        return int("".join(raw), 16)

    def code_word(self, addr: int) -> int:
        """The instruction word a fetch at *addr* would get now (the
        I-cache's copy when it holds the line), for the views and the
        debugger: no side effects.  Unwritten bytes: :class:`FetchError`
        (an address no TLB entry maps: :class:`UnmappedAddress`)."""
        addr &= M32
        if self._dm and EJ.in_dseg(addr):       # the probe's memory (dmseg)
            word = self.ejtag.peek(addr, 4) if addr < EJ.DRSEG_BASE else None
            if word is None:
                raise UnmappedAddress(f"0x{addr:08x}: EJTAG memory, no probe serves it",
                                      name=None, word=None, addr=addr)
            return word
        order = "big" if self.big_endian else "little"
        pa, cca = self._peek(addr)
        if self.caches is not None:
            cached = self._icached(cca)
            c = self.caches.i
            ln = c.probe(addr, pa) if cached and c is not None else None
            if ln is not None:
                off = addr & c.line_mask
                known = ln.known >> off & 0xF
                if known != 0xF:
                    i = next(k for k in range(4) if not known >> k & 1)
                    raise self._no_code(addr, i, pa + i)
                return int.from_bytes(ln.data[off:off + 4], order)
            pas = self._view_pas(addr, 4)
            coherent = self.caches.coherent(cca)
            data = [self.caches.view_data(addr, p, cached, coherent) if coherent
                    else self.caches.view_below(p, cached) for p in pas]
            for i, (p, byte) in enumerate(zip(pas, data)):
                if byte is None:
                    raise self._no_code(addr, i, p)
            return int.from_bytes(bytes(data), order)
        pas = self._view_pas(addr, 4)
        if self.board is not None and self.board.bus.covers(pas[0]):
            return self._board_peek(pas[0], 4)
        mem = self.mem.get_memory()
        raw = [mem.get(format(pa, "x")) for pa in pas]
        for i, (pa, byte) in enumerate(zip(pas, raw)):
            if byte is None:
                raise self._no_code(addr, i, pa)
        if not self.big_endian:
            raw.reverse()
        return int("".join(raw), 16)

    # -- cache maintenance (CACHE, SYNCI, PREF) --------------------------------

    def cache_op(self, op: int, addr: int) -> str:
        """``CACHE op, addr``: the kernel-segment check (AdEL, never for
        alignment), the translation (a TLB Refill / Invalid can happen on
        any operation, Vol. II CACHE), then the operation
        (:meth:`emips.cache.Caches.cache_op`).  Returns what was done
        (also kept in :attr:`last_cache_op`)."""
        addr &= M32
        if not self.address_ok(addr):
            raise _Arch(C.ADEL, badvaddr=addr)
        pa, cca = self._xlate(self, addr, "cacheop")
        if self.caches is None:
            done = "no caches: no operation"
        else:
            try:
                done = self.caches.cache_op(op, addr, pa, cca)
            except CacheError as err:
                raise self._cache_arch(err) from None
        self.last_cache_op = done
        return done

    def synci(self, addr: int) -> None:
        """``SYNCI addr`` (Release 2): write the data-cache line back and
        invalidate the instruction-cache line holding *addr*.  A kernel
        address in user mode is an Address Error (the manual leaves it to
        the implementation); the address is translated (TLB Refill /
        Invalid)."""
        addr &= M32
        if not self.address_ok(addr):
            raise _Arch(C.ADEL, badvaddr=addr)
        pa, cca = self._xlate(self, addr, "cacheop")
        if self.caches is not None and cca in self.cached_cca:
            try:
                self.caches.synci(addr, pa, cca)
            except CacheError as err:
                raise self._cache_arch(err) from None

    def prefetch(self, hint: int, addr: int) -> None:
        """``PREF hint, addr``: never an address or TLB exception — an
        address that would raise one is not prefetched; nothing for an
        uncached address (Vol. II-A PREF)."""
        addr &= M32
        if self.caches is None or not self.address_ok(addr):
            return
        try:
            pa, cca = self._xlate(self, addr, "pref")
        except _Arch:
            return
        if hint == 25 and cca == CCA_UCA:       # Nudge: flush a gathered block
            self.caches.nudge(pa)
        if cca in self.cached_cca:
            d = self.caches.d
            hits = d.stats.hits if d is not None else 0
            try:
                self.caches.prefetch(hint, addr, pa, cca)
            except CacheError as err:
                raise self._cache_arch(err) from None
            if self.perf.armed and d is not None:    # attempted: the 24K's PREF events
                self.perf.pending += ["pref"] + (["pref_hit"] if d.stats.hits > hits else [])

    def sync(self, stype: int = 0) -> None:
        """``SYNC stype``: every barrier type empties the write buffer's
        merging / gathering entries (24K SYNC: completion barriers flush
        the pending writes, ordering barriers stop merging); their cost in
        cycles comes with the memory timing (M9 phase 4)."""
        if self.caches is not None:
            self.caches.sync(stype)

    def pending_cache_error(self) -> _Arch | None:
        """An imprecise L2 / L3 error or data bus error waiting to be taken
        (as the next instruction starts), or ``None``.  A bus error: a
        read the Coherence Manager answered with an ERROR response (its
        GCR_ERROR_MASK bit 0) — the 24K takes data bus errors imprecisely
        (MD00343 5.8.13).  A Debug Single Step / Debug Interrupt goes
        first; with ``Debug.IEXI = 1`` (the start of a debug handler) the
        error waits, shown in ``Debug.DBusEP`` / ``CacheEP``, until IEXI
        is cleared (MD00047 2.4.2) — software may also set those bits."""
        cache = self.caches is not None and self.caches.pending is not None
        if not (cache or self.bus_error_pending or self._ep):
            return None
        if self.debug_due:
            return None
        cell = self.cp0[C.DEBUG]
        dbg = cell["value"]
        if dbg & EJ.IEXI:
            if self.bus_error_pending:
                cell["value"] |= EJ.DBUSEP
            if cache:
                cell["value"] |= EJ.CACHEEP
            self._ep = bool(cell["value"] & EJ.W1_BITS)
            return None
        if self.bus_error_pending or dbg & EJ.DBUSEP:
            self.bus_error_pending = False
            cell["value"] = dbg & ~EJ.DBUSEP
            exc = _Arch(C.DBE)
        elif cache:
            cell["value"] = dbg & ~EJ.CACHEEP
            exc = self._cache_arch(self.caches.take_pending())
        else:                                   # CacheEP set by software: CacheErr as it is
            cell["value"] = dbg & ~EJ.CACHEEP
            exc = _Arch(C.CACHE_ERROR, info={"cacheerr": self.cp0_read(*C.CACHEERR),
                                             "what": "Debug.CacheEP set by software"})
        self._ep = bool(cell["value"] & EJ.W1_BITS)
        return exc

    def address_ok(self, addr: int) -> bool:
        """May the current mode use virtual address *addr*?  The fixed
        segments (Vol. III "Virtual Memory"), or the MMU's segment
        configuration (Segmentation Control, an EVA access with user
        rights).  Debug Mode has kernel rights."""
        if self._dm and not self._eva:
            return True
        if self._mmu_address_ok is not None:
            return self._mmu_address_ok(self, addr & M32)
        return C.address_ok(self.status, addr)

    def data_address(self, addr: int, size: int, store: bool, *,
                     atomic: bool = False, span: tuple[int, int] | None = None) -> int:
        """Address check of a data access: alignment to ``size`` and the
        segment rights of the current mode, else Address Error (AdEL for
        loads, AdES for stores) with ``BadVAddr = addr``.  Release 6
        handles misaligned ordinary loads/stores in hardware (here); only
        LL/SC (``atomic``) must stay aligned.  First the data breakpoints
        and watch registers, which come before an address error."""
        addr &= M32
        if self.agent_watch:
            self._agent_watch(addr, size, store, span)
        if self._dhw and not self._dm:
            self._data_checks(addr, size, store, span)
        misaligned = addr % size and (self.release != "R6" or atomic)
        if misaligned or not self.address_ok(addr):
            raise _Arch(C.ADES if store else C.ADEL, badvaddr=addr)
        return addr

    def _agent_watch(self, addr: int, size: int, store: bool,
                     span: tuple[int, int] | None) -> None:
        """A debugger watchpoint on any byte of this access: stop before
        it (:class:`WatchpointHit`; ``step`` makes the instruction run
        again)."""
        start, n = span if span is not None else (addr, size)
        for w in self.agent_watch:
            if (w.store if store else w.load) and start < w.addr + w.length \
                    and w.addr < start + n:
                raise WatchpointHit(max(start, w.addr), w, store)

    def r6_slot_check(self, dec: Decode, in_slot: bool, forbidden: bool) -> None:
        """Release 6: a branch or jump in a delay slot, or in the forbidden
        slot after a not-taken conditional compact branch, is Reserved
        Instruction."""
        if self.release == "R6" and (in_slot or forbidden) and is_cti(dec.entry):
            raise _Arch(C.RI)

    # -- stepping ------------------------------------------------------------

    def step(self) -> Decode | None:
        """Execute one instruction (one clock cycle in this model).

        0. If an enabled interrupt is pending, take it instead (EPC = this
           PC, or its branch when it is a delay slot) and return ``None``;
           while sleeping after ``WAIT``, only let a cycle pass.
        1. Capture and disarm the delay slot; check the PC (alignment and
           segment rights → AdEL) and fetch the word (unwritten bytes →
           :class:`FetchError`, simulator halt).
        2. Decode; no match → Reserved Instruction.
        3. Gate and dispatch by bucket (:func:`classify`):
           FPU families (and the COP1X PREFX) need ``Status.CU1`` and
           COP2 rows need ``CU2``
           (Coprocessor Unusable), :data:`PRIVILEGED_NAMES` need kernel mode
           or ``CU0``; then HINT = no-op, ILLEGAL = Reserved Instruction,
           DEFER = :class:`UnimplementedError`, EXEC = the executor.
        4. Advance: a taken branch arms the delay slot; a not-taken
           branch-likely skips its slot (``PC + 8``); otherwise ``PC + 4``,
           or the armed target when this instruction was a delay slot.

        An architectural exception goes to CP0 (or is raised in
        ``exceptions="raise"`` mode); see :meth:`_take_exception`.
        Returns the executed :class:`Decode` (``None`` when the word did
        not decode).
        """
        if self.halted:
            return None
        in_slot = self.slot_armed
        outer_target = self.slot_target
        forbidden, self._forbidden = self._forbidden, False
        self.slot_armed = False
        self.slot_target = None
        self._redirect = None
        pc = self.pc
        word: int | None = None
        dec: Decode | None = None
        late = self.pending_cache_error()
        if late is not None:                    # imprecise: before this instruction
            self.wake()
            self.advance(1)
            if self.pdt.active:
                self._trace_cycle(False)
            self._take_exception(late, pc, None, None, in_slot)
            return None
        req = self.interrupt_request()
        if req is not None:
            self.wake()
            self.advance(1)
            if self.pdt.active:
                self._trace_cycle(False)
            self._take_exception(req, pc, None, None, in_slot)
            return None
        if self.waiting:
            self.advance(1)
            if self.ejtag.sample_due:           # asleep: the WAIT is what graduated
                self.ejtag.sample(self.wait_pc)
            if self.pdt.active:
                self._trace_cycle(False, asleep=True)
            return None
        try:
            if self._ihw:
                self.instruction_checks(pc)
            if pc & 3 or not self.address_ok(pc):
                raise _Arch(C.ADEL, badvaddr=pc)
            word = self.fetch_word(pc)
            try:
                dec = self.isa.decode(word)
            except ISADecodeError as exc:
                if self.exceptions == "raise":
                    raise
                raise _Arch(C.RI) from exc
            self.r6_slot_check(dec, in_slot, forbidden)
            result = self._traced(pc, word, dec, in_slot)
            if self.last_events is not None:    # completed: count it
                self.perf.retire(*self.last_events)
                self.last_events = None
        except _Arch as exc:
            self.advance(1)
            if self.pdt.active:
                self._trig = []
                self._trace_cycle(False)
            self._take_exception(exc, pc, word, dec, in_slot)
            return dec
        except WatchpointHit:                   # a debugger stop: this instruction again
            self.slot_armed, self.slot_target, self._forbidden = in_slot, outer_target, forbidden
            self.last_events = None
            raise
        except (SimException, ISADecodeError) as exc:
            self.exception_log.append({
                "name": getattr(exc, "name", None),
                "word": getattr(exc, "word", word),
                "addr": getattr(exc, "addr", pc),
                "error": type(exc).__name__,
            })
            raise

        self.advance(1)
        if self.ejtag.sample_due:                   # PC sampling: this one graduated
            self.ejtag.sample(pc)
        if self.pdt.active or self.pdt_packet is not None:   # MIPS Trace
            self._trace_cycle(True)
        if self._redirect is not None:              # ERET: no delay slot
            self.pc = self._redirect & M32
        elif result == NULLIFY:
            self.pc = (pc + 8) & M32
        elif result is not None or has_delay_slot(dec.entry):
            # taken: the slot runs, then the target; not taken: the slot is
            # still a delay slot (BD on a fault there), then PC + 8
            self.slot_armed = True
            self.slot_target = (result if result is not None else pc + 8) & M32
            self.pc = (pc + 4) & M32
        else:
            self.pc = (outer_target if in_slot else pc + 4) & M32
        return dec

    # -- execution unit interface (used by step() and emips.pipeline) ---------

    def execute(self, pc: int, word: int, dec: Decode, *, in_slot: bool = False,
                sources: dict | None = None) -> int | None:
        """Functionally execute one decoded instruction located at ``pc``:
        no fetch, no clock tick, no PC update.  Returns the executor result
        (branch/jump target, :data:`NULLIFY` or ``None``); ``ERET``'s
        target is left in ``_redirect``.  Raises :class:`_Arch` for an
        architectural exception, before any state change."""
        self.pc = pc & M32
        self._redirect = None
        return self._traced(pc, word, dec, in_slot, sources)

    def _traced(self, pc: int, word: int, dec: Decode, in_slot: bool,
                sources: dict | None = None) -> int | None:
        """Run the executor; with :attr:`tracing` on, record what happened
        inside it in :attr:`last_view` (see :mod:`emips.trace`); then tell
        :attr:`call_hook` (the debugger's call stack) it completed."""
        if self.call_hook is None and not self._sst and not self.pdt.active:
            return self._run_instruction(pc, word, dec, in_slot, sources)
        dm = self._dm
        if self.pdt.active:                     # MIPS Trace: what it completes with
            result = self._run_traced(pc, word, dec, in_slot, sources)
        else:
            result = self._run_instruction(pc, word, dec, in_slot, sources)
        if self._sst and not dm and (in_slot or result == NULLIFY
                                     or not has_delay_slot(dec.entry)):
            # an execution step in Non-Debug Mode is done: the instruction,
            # or a branch with its delay slot (MD00047 2.3.9)
            self.ss_pending = True
        if self.call_hook is not None:
            self.call_hook(pc, dec, word, result)
        return result

    def _run_traced(self, pc: int, word: int, dec: Decode, in_slot: bool,
                    sources: dict | None) -> int | None:
        """Execute with PDtrace capturing: the mode it runs in, its loads /
        stores, a D-cache miss (LSm), then the packet for its completion."""
        mode = PDT.mode_of(self)
        asid = self.cp0[(10, 0)]["value"] & 0xFF
        controls = self.pdt._controls()         # an MTC0 to them acts after itself
        self._pdt_acc = acc = []
        d = self.caches.d if self.caches is not None else None
        misses = d.stats.misses if d is not None else 0
        try:
            result = self._run_instruction(pc, word, dec, in_slot, sources)
        finally:
            self._pdt_acc = None
        triggers, self._trig = self._trig, []
        packet = self.pdt.capture(pc, dec, word, result, acc, in_slot, mode, asid, triggers,
                                  controls)
        if packet is not None:
            packet.im = int(self.fetch_missed)
            if d is not None and d.stats.misses != misses and packet.records:
                for r in packet.records:
                    if r.ttype == PDT.TD:
                        r.lsm = 1
        self.pdt_packet = packet
        return result

    def _run_instruction(self, pc: int, word: int, dec: Decode, in_slot: bool,
                         sources: dict | None) -> int | None:
        if self.perf.armed:
            return self._counted(pc, word, dec, in_slot, sources)
        if not self.tracing:
            return self._dispatch(dec, word)
        return self._viewed(pc, word, dec, in_slot, sources)

    def _counted(self, pc: int, word: int, dec: Decode, in_slot: bool,
                 sources: dict | None) -> int | None:
        """Execute with the performance counters counting: on completion
        :attr:`last_events` = the instruction's events and its Status."""
        status = self.status
        self.perf.pending = []
        self.last_events = None
        result = (self._viewed(pc, word, dec, in_slot, sources) if self.tracing
                  else self._dispatch(dec, word))
        self.last_events = (self.perf.classify(dec, word), status)
        return result

    def _viewed(self, pc: int, word: int, dec: Decode, in_slot: bool,
                sources: dict | None) -> int | None:
        from . import trace
        view = trace.begin(self, pc, word, dec, in_slot, sources)
        self.last_view = view
        try:
            result = self._dispatch(dec, word)
        except Exception as exc:
            trace.finish(self, view, exception=exc)
            raise
        trace.finish(self, view, result=result)
        return result

    def take_exception(self, exc: _Arch, pc: int, word: int | None,
                       dec: Decode | None, in_slot: bool) -> None:
        """Enter the exception handler for ``exc`` raised by the
        instruction at ``pc`` (see :meth:`_take_exception`)."""
        self.slot_armed = False
        self.slot_target = None
        self._take_exception(exc, pc, word, dec, in_slot)

    def _dispatch(self, dec: Decode, word: int) -> int | None:
        entry = dec.entry
        name, family = entry["name"], entry["family"]
        status = self.status
        if family in FPU_FAMILIES or name in ("MOVF", "MOVT", "PREFX"):
            if not status & C.ST_CU1:
                raise _Arch(C.CPU, ce=1)
        if family == "cop2":
            raise _Arch(C.CPU, ce=2)                # CU2 is hardwired 0
        if name in EVA_NAMES:
            if not getattr(self.mmu, "eva", False):  # Config5.EVA = 0
                raise _Arch(C.RI)
            if not (self.kernel_mode or status & C.ST_CU0):
                raise _Arch(C.CPU, ce=0)
            self._eva = True
            try:
                hint = _HINT_EXECUTORS.get(name)
                return hint(self, word) if hint is not None else _EXECUTORS[name](self, word)
            finally:
                self._eva = False
        if name in PRIVILEGED_NAMES and not (self.kernel_mode
                                             or status & C.ST_CU0):
            raise _Arch(C.CPU, ce=0)
        bucket = classify(entry)
        if bucket == BUCKET_HINT:
            hint = _HINT_EXECUTORS.get(name)
            return hint(self, word) if hint is not None else None
        if bucket == BUCKET_ILLEGAL:
            raise _Arch(C.RI)
        if bucket == BUCKET_DEFER:
            raise UnimplementedError(
                f"not executed yet: {name}", name=name, word=word, addr=self.pc)
        return _EXECUTORS[name](self, word)

    def _take_exception(self, exc: _Arch, pc: int, word: int | None,
                        dec: Decode | None, in_slot: bool) -> None:
        """General exception processing (Vol. III), or raise in debug mode."""
        name = dec.name if dec is not None else None
        entry: dict[str, Any] = {
            "name": name, "word": word, "addr": pc,
            "error": C.EXC_NAMES.get(exc.code, str(exc.code)),
            "exc_code": exc.code, "badvaddr": exc.badvaddr,
        }
        if self.exceptions == "raise":
            self.exception_log.append(entry)
            raise _python_exception(exc, name, word, pc)

        status = self.status
        if self.perf.armed:
            self.perf.add("exceptions", 1, status)
        if self.caches is not None:             # flushes a gathered UCA block
            self.caches.exception_entry()
        slept = self.waiting or self.slept
        self.waiting = self.slept = False
        if exc.code in C.DEBUG_EXCEPTIONS and not self._dm:
            self._take_debug(exc, pc, in_slot, entry, slept)
            return
        if self._dm:                            # an exception in Debug Mode
            self._take_debug_mode(exc, pc, in_slot, entry)
            return
        if self._sst:                           # a DSS on the handler's first instruction
            self.ss_pending = True
        if exc.code == C.CACHE_ERROR:
            self._take_cache_error(exc, pc, in_slot, entry)
            return
        if exc.code == C.NMI:
            self._take_nmi(pc, in_slot, entry)
            return
        cause = self.cp0[C.CAUSE]["value"]
        offset = None
        if not status & C.ST_EXL:
            epc = (pc - 4) & M32 if in_slot else pc
            self._cp0_set(C.EPC, epc)
            cause = cause | C.CA_BD if in_slot else cause & ~C.CA_BD
            # Vol. III 6.2.3: the vector of a vectored interrupt, the new set
            new_set = self.srs_field(C.SRS_ESS)
            mode = self.interrupt_mode() if exc.code == C.INT else None
            if mode == "vectored":
                vecnum = C.vi_vector(status, cause)
                new_set = (self.cp0_read(*C.SRSMAP) >> (4 * vecnum)) & 0xF
                offset = C.vi_offset(vecnum, self.cp0_read(*C.INTCTL))
                entry["vector_number"] = vecnum
            elif mode == "eic":                 # option 1: the vector is RIPL
                vecnum = self.irq_lines         # Cause.RIPL, SRSCtl.EICSS latched
                cause = (cause & ~(0x3F << (C.CA_IP_SHIFT + 2))) | (vecnum << (C.CA_IP_SHIFT + 2))
                new_set = self.eic_ss if self.eic_ss <= self.hss else 0
                srs = self.cp0[C.SRSCTL]["value"]
                if self.hss:
                    self._cp0_set(C.SRSCTL, (srs & ~(0xF << C.SRS_EICSS))
                                  | (new_set << C.SRS_EICSS))
                offset = C.vi_offset(vecnum, self.cp0_read(*C.INTCTL))
                entry["vector_number"] = vecnum
                entry["ripl"] = vecnum
            if self.hss and not status & C.ST_BEV:
                self._set_css(new_set, save=True)
                entry["shadow_set"] = new_set
        cause &= ~((3 << C.CA_CE_SHIFT) | (0x1F << C.CA_EXC_SHIFT))
        cause |= (exc.ce << C.CA_CE_SHIFT) | (exc.code << C.CA_EXC_SHIFT)
        self._cp0_set(C.CAUSE, cause)
        if exc.badvaddr is not None:
            self._cp0_set(C.BADVADDR, exc.badvaddr)
        tlb = exc.info.get("tlb", False)
        if tlb:                                 # Context.BadVPN2, EntryHi.VPN2
            self.mmu.exception_entry(self, exc.badvaddr)
        self._bad_instr(exc, pc, word, in_slot)
        self._cp0_set(C.STATUS, status | C.ST_EXL)
        if exc.code == C.MCHECK:                # Status.TS, PageGrain.MCCause
            self.mmu.machine_check(self, exc.info.get("mccause", 1))
        refill = tlb and exc.info.get("refill", False) and not status & C.ST_EXL
        ebase = self.cp0[C.EBASE]["value"]
        if getattr(self.mmu, "sc", False) and not self.cp0_read(16, 5) & (1 << 30):
            # Config5.K = 0 (the legacy configuration): EBase[31:30] read as
            # 10 — the vectors stay in kseg0 / kseg1 (MD01025 3.6.3.2)
            ebase = (ebase & 0x3FFFFFFF) | 0x80000000
        self.pc = C.vector(status, ebase,
                           interrupt_iv=exc.code == C.INT and bool(cause & C.CA_IV),
                           refill=refill, bev_base=self.bev_base, offset=offset)
        entry.update(epc=self.cp0[C.EPC]["value"], bd=bool(cause & C.CA_BD),
                     vector=self.pc, refill=refill)
        if exc.info.get("why"):
            entry["why"] = exc.info["why"]
        self.exception_log.append(entry)

    #: exceptions that do not set BadInstr / BadInstrP (Vol. III 9.25):
    #: not synchronous to an instruction
    _NO_BADINSTR = frozenset({C.INT, C.MCHECK, C.IBE, C.DBE, C.CACHE_ERROR})

    def _bad_instr(self, exc: _Arch, pc: int, word: int | None, in_slot: bool) -> None:
        """BadInstr = the faulting instruction word, BadInstrP = the branch
        before it when it is in a delay slot (Vol. III 9.25 / 9.26; Release
        3+ with the TLB, required in Release 6).  Without an instruction
        word (a fetch fault) they are UNPREDICTABLE: left unchanged."""
        if not self.has_bad_instr or exc.code in self._NO_BADINSTR or word is None:
            return
        self._cp0_set(C.BADINSTR, word)
        if in_slot:
            try:
                self._cp0_set(C.BADINSTRP, self.code_word((pc - 4) & M32))
            except FetchError:
                pass

    def _take_debug(self, exc: _Arch, pc: int, in_slot: bool, entry: dict,
                    slept: bool) -> None:
        """A debug exception (MD00047 2.3.4): DEPC = the restart PC (the
        branch for a delay slot: DBD), the exception's Debug bit (the
        others cleared), Halt / Doze = the core slept (WAIT; Doze also
        RP), IEXI = DM = 1, PC = the debug vector.  Pending debug
        interrupt / single step requests are cleared; nothing else
        changes (Status, Cause, EPC, SRSCtl stay)."""
        cell = self.cp0[C.DEBUG]
        depc = (pc - 4) & M32 if in_slot else pc
        dbg = cell["value"] & ~(EJ.DBD | EJ.CAUSE_BITS | EJ.DOZE | EJ.HALT)
        dbg |= EJ.EXC_BIT[exc.code] | self.ddb_impr | EJ.IEXI | EJ.DM
        self.ddb_impr = 0
        if in_slot:
            dbg |= EJ.DBD
        if slept:
            dbg |= EJ.DOZE | EJ.HALT
        elif self.status & (1 << 27):           # Status.RP: reduced power
            dbg |= EJ.DOZE
        cell["value"] = dbg
        self._cp0_set(C.DEPC, depc)
        self._dm = True
        self.ss_pending = self.dint_pending = False
        self.pc = self.debug_vector()
        entry.update(epc=depc, bd=in_slot, vector=self.pc, debug=True)
        self.exception_log.append(entry)

    def _take_debug_mode(self, exc: _Arch, pc: int, in_slot: bool, entry: dict) -> None:
        """An exception in Debug Mode (MD00047 2.4.3): only DEPC and
        Debug change — DBD, the cause bits cleared, DExcCode = the
        exception's code (Bp 9 for SDBBP, CacheErr 30 for a cache
        error), IEXI = 1 — and the debug vector is entered again, still
        in Debug Mode."""
        code = exc.code
        dexc = (EJ.DEXC_BP if code == C.DBP else EJ.DEXC_CACHEERR if code == C.CACHE_ERROR
                else code & 0x1F)
        cell = self.cp0[C.DEBUG]
        depc = (pc - 4) & M32 if in_slot else pc
        dbg = cell["value"] & ~(EJ.DBD | EJ.CAUSE_BITS | (0x1F << EJ.DEXC_SHIFT))
        dbg |= (dexc << EJ.DEXC_SHIFT) | EJ.IEXI
        if in_slot:
            dbg |= EJ.DBD
        cell["value"] = dbg
        self._cp0_set(C.DEPC, depc)
        self.waiting = False
        self.pc = self.debug_vector()
        entry.update(epc=depc, bd=in_slot, vector=self.pc, debug_mode=True, dexccode=dexc)
        self.exception_log.append(entry)

    def _take_nmi(self, pc: int, in_slot: bool, entry: dict) -> None:
        """Vol. III "Non Maskable Interrupt (NMI) Exception": ErrorEPC = the
        restart PC (the branch for a delay slot), Status BEV = ERL = NMI = 1,
        SR = TS = 0, the reset vector (a CPS core's reset base); Cause is
        not touched."""
        self.nmi_pending = False
        self.waiting = False
        epc = (pc - 4) & M32 if in_slot else pc
        self._cp0_set(C.ERROREPC, epc)
        status = self.status & ~(C.ST_SR | (1 << 21))
        self._cp0_set(C.STATUS, status | C.ST_BEV | C.ST_ERL | C.ST_NMI)
        self.pc = (self.bev_base - 0x200) & M32
        entry.update(epc=epc, bd=in_slot, vector=self.pc)
        self.exception_log.append(entry)

    def _take_cache_error(self, exc: _Arch, pc: int, in_slot: bool,
                          entry: dict[str, Any]) -> None:
        """Cache Error exception (Vol. III 6.2.15; 24K 5.8.12): CacheErr,
        ErrorEPC = the restart PC (the branch for a delay slot), Status.ERL
        = 1, PC = the Cache Error vector (kseg1: unmapped, uncached).
        ErrCtl.FE marks the first error; another one while FE is still set
        sets SE.  Not maskable; Cause is untouched."""
        status = self.status
        self._cp0_set(C.CACHEERR, exc.info["cacheerr"])
        errctl = self.cp0_read(*C.ERRCTL)
        errctl = (errctl | (1 << 20)) if errctl & (1 << 19) else (errctl | (1 << 19)) & ~(1 << 20)
        self._cp0_set(C.ERRCTL, errctl)
        epc = (pc - 4) & M32 if in_slot else pc
        self._cp0_set(C.ERROREPC, epc)
        self._cp0_set(C.STATUS, status | C.ST_ERL)
        self.pc = C.cache_error_vector(status, self.cp0[C.EBASE]["value"], self.release,
                                       cv=getattr(self.mmu, "sc", False)
                                       and bool(self.cp0_read(16, 5) & (1 << 29)),
                                       bev_base=self.bev_base)
        entry.update(epc=epc, bd=in_slot, vector=self.pc, cacheerr=exc.info["cacheerr"],
                     what=exc.info["what"])
        self.exception_log.append(entry)


def _python_exception(exc: _Arch, name: str | None, word: int | None,
                      pc: int) -> MipsException:
    """The ``exceptions="raise"`` form of an architectural exception."""
    label = C.EXC_NAMES.get(exc.code, str(exc.code))
    msg = f"{label} exception" + (f" in {name}" if name else "") + f" at 0x{pc:08x}"
    if exc.code == C.CACHE_ERROR:
        msg = f"Cache Error exception ({exc.info['what']})" + msg[len(label) + 10:]
    kw = dict(name=name, word=word, addr=pc)
    if exc.code == C.SYS:
        return SyscallException(msg, service=exc.info.get("code"), **kw)
    if exc.code in (C.TR, C.BP):
        return TrapException(msg, trapcode=exc.info.get("code"),
                             exc_code=exc.code, **kw)
    if exc.code == C.RI:
        return IllegalInstructionError(msg, exc_code=C.RI, **kw)
    return MipsException(msg, exc_code=exc.code, badvaddr=exc.badvaddr,
                         ce=exc.ce, **kw)


# ---------------------------------------------------------------------------
# Executors (one per EXEC row; dispatched by name, never by ``funct``).
# An executor returns ``None`` (fall through), a branch/jump target (arms
# the delay slot) or NULLIFY; it raises _Arch for an exception, before
# changing any state.
# ---------------------------------------------------------------------------

Executor = Callable[[Machine, int], int | None]
_M = Machine


# -- R-type ALU -------------------------------------------------------------


def _ov(r: int) -> int:
    """Integer Overflow when a signed result does not fit in 32 bits
    (ADD/ADDI/SUB); otherwise the result."""
    if not -(1 << 31) <= r < (1 << 31):
        raise _Arch(C.OV)
    return r


def _e_add(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), _ov(_s32(m.read_reg(rs(w))) + _s32(m.read_reg(rt(w)))))
    return None


def _e_addu(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rs(w)) + m.read_reg(rt(w)))
    return None


def _e_sub(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), _ov(_s32(m.read_reg(rs(w))) - _s32(m.read_reg(rt(w)))))
    return None


def _e_subu(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rs(w)) - m.read_reg(rt(w)))
    return None


def _e_and(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rs(w)) & m.read_reg(rt(w)))
    return None


def _e_or(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rs(w)) | m.read_reg(rt(w)))
    return None


def _e_xor(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rs(w)) ^ m.read_reg(rt(w)))
    return None


def _e_nor(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), ~(m.read_reg(rs(w)) | m.read_reg(rt(w))))
    return None


def _e_slt(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), int(_s32(m.read_reg(rs(w))) < _s32(m.read_reg(rt(w)))))
    return None


def _e_sltu(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), int(m.read_reg(rs(w)) < m.read_reg(rt(w))))
    return None


def _e_movz(m: _M, w: int) -> int | None:
    if m.read_reg(rt(w)) == 0:
        m.write_reg(rd(w), m.read_reg(rs(w)))
    return None


def _e_movn(m: _M, w: int) -> int | None:
    if m.read_reg(rt(w)) != 0:
        m.write_reg(rd(w), m.read_reg(rs(w)))
    return None


# -- Shifts / rotates -------------------------------------------------------


def _rotr(x: int, n: int) -> int:
    n &= 31
    return ((x >> n) | (x << (32 - n))) & M32 if n else x & M32


def _e_sll(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rt(w)) << sa(w))
    return None


def _e_srl(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rt(w)) >> sa(w))
    return None


def _e_sra(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), _s32(m.read_reg(rt(w))) >> sa(w))
    return None


def _e_rotr(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), _rotr(m.read_reg(rt(w)), sa(w)))
    return None


def _e_sllv(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rt(w)) << (m.read_reg(rs(w)) & 31))
    return None


def _e_srlv(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rt(w)) >> (m.read_reg(rs(w)) & 31))
    return None


def _e_srav(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), _s32(m.read_reg(rt(w))) >> (m.read_reg(rs(w)) & 31))
    return None


def _e_rotrv(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), _rotr(m.read_reg(rt(w)), m.read_reg(rs(w))))
    return None


# -- Count leading, bit fields, byte shuffles -------------------------------


def _e_clo(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), 32 - (~m.read_reg(rs(w)) & M32).bit_length())
    return None


def _e_clz(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), 32 - m.read_reg(rs(w)).bit_length())
    return None


def _e_ins(m: _M, w: int) -> int | None:
    """INS rt, rs, pos, size: ``rt[msb..lsb] ← rs[msb-lsb..0]``
    (``msb`` = rd field, ``lsb`` = sa field)."""
    msb, lsb = rd(w), sa(w)
    width = max(msb - lsb + 1, 0)
    mask = (((1 << width) - 1) << lsb) & M32
    src = (m.read_reg(rs(w)) << lsb) & mask
    m.write_reg(rt(w), (m.read_reg(rt(w)) & ~mask) | src)
    return None


def _e_ext(m: _M, w: int) -> int | None:
    """EXT rt, rs, pos, size: ``rt ← 0 || rs[pos+size-1..pos]``
    (``msbd`` = rd field = size - 1, ``lsb`` = sa field = pos)."""
    size, pos = rd(w) + 1, sa(w)
    m.write_reg(rt(w), (m.read_reg(rs(w)) >> pos) & ((1 << size) - 1))
    return None


def _e_seb(m: _M, w: int) -> int | None:
    v = m.read_reg(rt(w)) & 0xFF
    m.write_reg(rd(w), v - 0x100 if v & 0x80 else v)
    return None


def _e_seh(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), _s16(m.read_reg(rt(w))))
    return None


def _e_wsbh(m: _M, w: int) -> int | None:
    v = m.read_reg(rt(w))
    m.write_reg(rd(w), ((v & 0x00FF00FF) << 8) | ((v >> 8) & 0x00FF00FF))
    return None


# -- MUL / HI:LO family -----------------------------------------------------


def _prod64(a: int, b: int, signed: bool) -> int:
    p = _s32(a) * _s32(b) if signed else (a & M32) * (b & M32)
    return p & M64


def _hi_lo(m: _M) -> int:
    return (m.hi << 32) | (m.lo & M32)


def _set_hi_lo(m: _M, value: int) -> None:
    value &= M64
    m.hi, m.lo = value >> 32, value & M32


def _e_mul(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), _prod64(m.read_reg(rs(w)), m.read_reg(rt(w)), True))
    return None


def _e_mult(m: _M, w: int) -> int | None:
    _set_hi_lo(m, _prod64(m.read_reg(rs(w)), m.read_reg(rt(w)), True))
    return None


def _e_multu(m: _M, w: int) -> int | None:
    _set_hi_lo(m, _prod64(m.read_reg(rs(w)), m.read_reg(rt(w)), False))
    return None


def _e_madd(m: _M, w: int) -> int | None:
    _set_hi_lo(m, _hi_lo(m) + _prod64(m.read_reg(rs(w)), m.read_reg(rt(w)), True))
    return None


def _e_maddu(m: _M, w: int) -> int | None:
    _set_hi_lo(m, _hi_lo(m) + _prod64(m.read_reg(rs(w)), m.read_reg(rt(w)), False))
    return None


def _e_msub(m: _M, w: int) -> int | None:
    _set_hi_lo(m, _hi_lo(m) - _prod64(m.read_reg(rs(w)), m.read_reg(rt(w)), True))
    return None


def _e_msubu(m: _M, w: int) -> int | None:
    _set_hi_lo(m, _hi_lo(m) - _prod64(m.read_reg(rs(w)), m.read_reg(rt(w)), False))
    return None


def _e_mfhi(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.hi)
    return None


def _e_mflo(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.lo)
    return None


def _e_mthi(m: _M, w: int) -> int | None:
    m.hi = m.read_reg(rs(w))
    return None


def _e_mtlo(m: _M, w: int) -> int | None:
    m.lo = m.read_reg(rs(w))
    return None


def _e_div(m: _M, w: int) -> int | None:
    """DIV: quotient (toward zero) → LO, remainder (dividend's sign) → HI.
    ÷0 is UNPREDICTABLE (see :data:`DEVIATIONS`)."""
    a, b = _s32(m.read_reg(rs(w))), _s32(m.read_reg(rt(w)))
    if b == 0:
        m.hi = m.lo = M32
        return None
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    m.lo, m.hi = q & M32, (a - q * b) & M32
    return None


def _e_divu(m: _M, w: int) -> int | None:
    a, b = m.read_reg(rs(w)), m.read_reg(rt(w))
    if b == 0:
        m.hi = m.lo = M32
    else:
        m.lo, m.hi = a // b, a % b
    return None


# -- I-form ALU -------------------------------------------------------------


def _e_addi(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), _ov(_s32(m.read_reg(rs(w))) + _s16(imm16(w))))
    return None


def _e_addiu(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), m.read_reg(rs(w)) + _s16(imm16(w)))
    return None


def _e_andi(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), m.read_reg(rs(w)) & imm16(w))
    return None


def _e_ori(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), m.read_reg(rs(w)) | imm16(w))
    return None


def _e_xori(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), m.read_reg(rs(w)) ^ imm16(w))
    return None


def _e_slti(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), int(_s32(m.read_reg(rs(w))) < _s16(imm16(w))))
    return None


def _e_sltiu(m: _M, w: int) -> int | None:
    """SLTIU: unsigned compare against the *sign-extended* immediate."""
    m.write_reg(rt(w), int(m.read_reg(rs(w)) < (_s16(imm16(w)) & M32)))
    return None


def _e_lui(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), imm16(w) << 16)
    return None


# -- Branches ---------------------------------------------------------------
# Link register is always GPR 31 (= PC + 8), written whether or not the
# branch is taken.  The "likely" forms nullify the delay slot when not taken.


def _branch(m: _M, w: int, cond: bool, *, link: bool = False,
            likely: bool = False) -> int | None:
    if link:
        m.write_reg(31, m.pc + 8)
    if cond:
        return _br_target(m.pc, w)
    return NULLIFY if likely else None


def _e_beq(m: _M, w: int) -> int | None:
    return _branch(m, w, m.read_reg(rs(w)) == m.read_reg(rt(w)))


def _e_beql(m: _M, w: int) -> int | None:
    return _branch(m, w, m.read_reg(rs(w)) == m.read_reg(rt(w)), likely=True)


def _e_bne(m: _M, w: int) -> int | None:
    return _branch(m, w, m.read_reg(rs(w)) != m.read_reg(rt(w)))


def _e_bnel(m: _M, w: int) -> int | None:
    return _branch(m, w, m.read_reg(rs(w)) != m.read_reg(rt(w)), likely=True)


def _gez(m: _M, w: int) -> bool:
    return _s32(m.read_reg(rs(w))) >= 0


def _gtz(m: _M, w: int) -> bool:
    return _s32(m.read_reg(rs(w))) > 0


def _e_bgez(m: _M, w: int) -> int | None:
    return _branch(m, w, _gez(m, w))


def _e_bgezl(m: _M, w: int) -> int | None:
    return _branch(m, w, _gez(m, w), likely=True)


def _e_bgezal(m: _M, w: int) -> int | None:
    return _branch(m, w, _gez(m, w), link=True)


def _e_bgezall(m: _M, w: int) -> int | None:
    return _branch(m, w, _gez(m, w), link=True, likely=True)


def _e_bltz(m: _M, w: int) -> int | None:
    return _branch(m, w, not _gez(m, w))


def _e_bltzl(m: _M, w: int) -> int | None:
    return _branch(m, w, not _gez(m, w), likely=True)


def _e_bltzal(m: _M, w: int) -> int | None:
    return _branch(m, w, not _gez(m, w), link=True)


def _e_bltzall(m: _M, w: int) -> int | None:
    return _branch(m, w, not _gez(m, w), link=True, likely=True)


def _e_bgtz(m: _M, w: int) -> int | None:
    return _branch(m, w, _gtz(m, w))


def _e_bgtzl(m: _M, w: int) -> int | None:
    return _branch(m, w, _gtz(m, w), likely=True)


def _e_blez(m: _M, w: int) -> int | None:
    return _branch(m, w, not _gtz(m, w))


def _e_blezl(m: _M, w: int) -> int | None:
    return _branch(m, w, not _gtz(m, w), likely=True)


# B = BEQ r0, r0; BAL = BGEZAL r0; NAL = BLTZAL r0 (never taken, links).
_e_b = _e_beq
_e_bal = _e_bgezal
_e_nal = _e_bltzal


# -- Jumps ------------------------------------------------------------------
# JR/JALR jump to GPR[rs] as is: a misaligned target faults (AdEL) when it
# is fetched, as there is no MIPS16e / microMIPS mode to switch to.


def _e_j(m: _M, w: int) -> int | None:
    return _j_target(m.pc, w)


def _e_jal(m: _M, w: int) -> int | None:
    m.write_reg(31, m.pc + 8)
    return _j_target(m.pc, w)


def _e_jr(m: _M, w: int) -> int | None:
    return m.read_reg(rs(w))


def _e_jalr(m: _M, w: int) -> int | None:
    target = m.read_reg(rs(w))
    m.write_reg(rd(w), m.pc + 8)
    return target


_e_jr_hb = _e_jr          # + hazard barrier: no hazards in this model
_e_jalr_hb = _e_jalr


def _e_jalx(m: _M, w: int) -> int | None:
    raise _Arch(C.RI)     # no MIPS16e / microMIPS (see DEVIATIONS)


# -- Loads / stores ---------------------------------------------------------


def _addr_i(m: _M, w: int) -> int:
    return (m.read_reg(rs(w)) + _s16(imm16(w))) & M32


def _addr_e(m: _M, w: int) -> int:
    return (m.read_reg(rs(w)) + off9(w)) & M32


def _sx(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _load(m: _M, w: int, addr: int, size: int, signed: bool) -> None:
    v = m.load(m.data_address(addr, size, store=False), size)
    m.write_reg(rt(w), _sx(v, 8 * size) if signed else v)


def _store(m: _M, w: int, addr: int, size: int) -> None:
    m.store(m.data_address(addr, size, store=True), m.read_reg(rt(w)), size)


def _e_lb(m: _M, w: int) -> int | None:
    _load(m, w, _addr_i(m, w), 1, True)
    return None


def _e_lbu(m: _M, w: int) -> int | None:
    _load(m, w, _addr_i(m, w), 1, False)
    return None


def _e_lh(m: _M, w: int) -> int | None:
    _load(m, w, _addr_i(m, w), 2, True)
    return None


def _e_lhu(m: _M, w: int) -> int | None:
    _load(m, w, _addr_i(m, w), 2, False)
    return None


def _e_lw(m: _M, w: int) -> int | None:
    _load(m, w, _addr_i(m, w), 4, False)
    return None


def _e_sb(m: _M, w: int) -> int | None:
    _store(m, w, _addr_i(m, w), 1)
    return None


def _e_sh(m: _M, w: int) -> int | None:
    _store(m, w, _addr_i(m, w), 2)
    return None


def _e_sw(m: _M, w: int) -> int | None:
    _store(m, w, _addr_i(m, w), 4)
    return None


# -- Unaligned word access (LWL/LWR/SWL/SWR and the EVA forms) --------------
# Vol. II-A LWL/LWR/SWL/SWR: the bytes from the effective address to the
# end (LWL/SWL, big-endian) or start (LWR/SWR) of its aligned word are
# merged into the most / least significant end of the register.  In
# little-endian mode the roles of the two ends swap.  ``k = addr & 3``.


def _span_left(m: _M, addr: int) -> tuple[int, int]:
    """The bytes LWL / SWL access (the breakpoints' byte lanes)."""
    k = addr & 3
    return (addr, 4 - k) if m.big_endian else (addr & ~3 & M32, k + 1)


def _span_right(m: _M, addr: int) -> tuple[int, int]:
    """The bytes LWR / SWR access."""
    k = addr & 3
    return (addr & ~3 & M32, k + 1) if m.big_endian else (addr, 4 - k)


def _lw_left(m: _M, w: int, addr: int) -> None:
    addr = m.data_address(addr, 1, store=False, span=_span_left(m, addr & M32))
    k, base, old = addr & 3, addr & ~3, m.read_reg(rt(w))
    if m.big_endian:                    # bytes addr..base+3 -> MS end
        n = 4 - k
        v = m.load(addr, n) << (8 * (4 - n))
    else:                               # bytes base..addr -> MS end
        n = k + 1
        v = m.load(base, n) << (8 * (4 - n))
    keep = (1 << (8 * (4 - n))) - 1
    m.write_reg(rt(w), v | (old & keep))


def _lw_right(m: _M, w: int, addr: int) -> None:
    addr = m.data_address(addr, 1, store=False, span=_span_right(m, addr & M32))
    k, base, old = addr & 3, addr & ~3, m.read_reg(rt(w))
    if m.big_endian:                    # bytes base..addr -> LS end
        n = k + 1
        v = m.load(base, n)
    else:                               # bytes addr..base+3 -> LS end
        n = 4 - k
        v = m.load(addr, n)
    keep = M32 ^ ((1 << (8 * n)) - 1)
    m.write_reg(rt(w), v | (old & keep))


def _sw_left(m: _M, w: int, addr: int) -> None:
    addr = m.data_address(addr, 1, store=True, span=_span_left(m, addr & M32))
    k, base, r = addr & 3, addr & ~3, m.read_reg(rt(w))
    if m.big_endian:                    # MS (4-k) bytes -> addr..base+3
        n = 4 - k
        m.store(addr, r >> (8 * (4 - n)), n, gather=False)
    else:                               # MS (k+1) bytes -> base..addr
        n = k + 1
        m.store(base, r >> (8 * (4 - n)), n, gather=False)


def _sw_right(m: _M, w: int, addr: int) -> None:
    addr = m.data_address(addr, 1, store=True, span=_span_right(m, addr & M32))
    k, base, r = addr & 3, addr & ~3, m.read_reg(rt(w))
    if m.big_endian:                    # LS (k+1) bytes -> base..addr
        m.store(base, r, k + 1, gather=False)
    else:                               # LS (4-k) bytes -> addr..base+3
        m.store(addr, r, 4 - k, gather=False)


def _e_lwl(m: _M, w: int) -> int | None:
    _lw_left(m, w, _addr_i(m, w))
    return None


def _e_lwr(m: _M, w: int) -> int | None:
    _lw_right(m, w, _addr_i(m, w))
    return None


def _e_swl(m: _M, w: int) -> int | None:
    _sw_left(m, w, _addr_i(m, w))
    return None


def _e_swr(m: _M, w: int) -> int | None:
    _sw_right(m, w, _addr_i(m, w))
    return None


# -- LL / SC ----------------------------------------------------------------
# LL sets LLbit; SC stores only while LLbit is set (and, here, the address
# matches the LL's), writes 1/0 to rt and clears LLbit.  ERET clears LLbit.


def _ll(m: _M, w: int, addr: int) -> None:
    addr = m.data_address(addr, 4, store=False, atomic=True)
    m._linked = True
    try:
        m.write_reg(rt(w), m.load(addr, 4))
    finally:
        m._linked = False
    m.lr_addr, m.lr_valid = addr, True
    if m.system is not None:                   # the line other cores' stores watch
        m.lr_pa = m._peek(addr)[0]


def _sc(m: _M, w: int, addr: int) -> None:
    addr = m.data_address(addr, 4, store=True, atomic=True)
    ok = m.lr_valid and m.lr_addr == addr
    if ok:                                     # atomic: never waits in the FSB
        m.store(addr, m.read_reg(rt(w)), 4, gather=False, atomic=True)
    m.write_reg(rt(w), int(ok))
    m.lr_valid = False


def _e_ll(m: _M, w: int) -> int | None:
    _ll(m, w, _addr_i(m, w))
    return None


def _e_sc(m: _M, w: int) -> int | None:
    _sc(m, w, _addr_i(m, w))
    return None


# -- EVA forms (kernel accesses with user-mode rights; 9-bit offset) ---------
# Machine._dispatch runs them with Machine._eva set: the MMU then checks
# and translates their addresses as a user-mode access would be (Vol. III
# Table 4.7); without EVA (Config5.EVA = 0) they are Reserved Instructions.


def _e_lbe(m: _M, w: int) -> int | None:
    _load(m, w, _addr_e(m, w), 1, True)
    return None


def _e_lbue(m: _M, w: int) -> int | None:
    _load(m, w, _addr_e(m, w), 1, False)
    return None


def _e_lhe(m: _M, w: int) -> int | None:
    _load(m, w, _addr_e(m, w), 2, True)
    return None


def _e_lhue(m: _M, w: int) -> int | None:
    _load(m, w, _addr_e(m, w), 2, False)
    return None


def _e_lwe(m: _M, w: int) -> int | None:
    _load(m, w, _addr_e(m, w), 4, False)
    return None


def _e_sbe(m: _M, w: int) -> int | None:
    _store(m, w, _addr_e(m, w), 1)
    return None


def _e_she(m: _M, w: int) -> int | None:
    _store(m, w, _addr_e(m, w), 2)
    return None


def _e_swe(m: _M, w: int) -> int | None:
    _store(m, w, _addr_e(m, w), 4)
    return None


def _e_lwle(m: _M, w: int) -> int | None:
    _lw_left(m, w, _addr_e(m, w))
    return None


def _e_lwre(m: _M, w: int) -> int | None:
    _lw_right(m, w, _addr_e(m, w))
    return None


def _e_swle(m: _M, w: int) -> int | None:
    _sw_left(m, w, _addr_e(m, w))
    return None


def _e_swre(m: _M, w: int) -> int | None:
    _sw_right(m, w, _addr_e(m, w))
    return None


def _e_lle(m: _M, w: int) -> int | None:
    _ll(m, w, _addr_e(m, w))
    return None


def _e_sce(m: _M, w: int) -> int | None:
    _sc(m, w, _addr_e(m, w))
    return None


# -- CP0 / privileged -------------------------------------------------------


def _cp0_cell(m: _M, w: int, name: str) -> tuple[int, int]:
    reg, sel = rd(w), w & 7
    if (reg, sel) not in m.cp0:
        raise CP0RegisterError(
            f"CP0 register not modeled: {name} reg={reg} sel={sel}",
            name=name, word=w, addr=m.pc, reg=reg, sel=sel,
        )
    return reg, sel


def _e_mfc0(m: _M, w: int) -> int | None:
    reg, sel = _cp0_cell(m, w, "MFC0")
    m.write_reg(rt(w), m.cp0_read(reg, sel))
    return None


def _e_mtc0(m: _M, w: int) -> int | None:
    reg, sel = _cp0_cell(m, w, "MTC0")
    m.cp0_write(reg, sel, m.read_reg(rt(w)))
    if (reg, sel) == C.COMPARE:             # writing Compare clears TI
        m._cp0_set(C.CAUSE, m.cp0_read(*C.CAUSE) & ~C.CA_TI)
        m._update_ip()
    return None


def _e_wait(m: _M, w: int) -> int | None:
    """WAIT: sleep until an interrupt is taken; its EPC is the instruction
    after the WAIT."""
    m.waiting = True
    m.wait_pc = m.pc
    return None


def _e_di(m: _M, w: int) -> int | None:
    status = m.status
    m.write_reg(rt(w), status)
    m._cp0_set(C.STATUS, status & ~C.ST_IE)
    return None


def _e_ei(m: _M, w: int) -> int | None:
    status = m.status
    m.write_reg(rt(w), status)
    m._cp0_set(C.STATUS, status | C.ST_IE)
    return None


def _e_eret(m: _M, w: int) -> int | None:
    """ERET: return from exception (ERL → ErrorEPC, else EPC), clearing
    ERL / EXL and LLbit; from EXL with shadow sets and BEV = 0, CSS ←
    PSS (Vol. III 9.31).  No delay slot."""
    status = m.status
    if status & C.ST_ERL:
        m._redirect = m.cp0_read(*C.ERROREPC)
        m._cp0_set(C.STATUS, status & ~C.ST_ERL)
    else:
        m._redirect = m.cp0_read(*C.EPC)
        m._cp0_set(C.STATUS, status & ~C.ST_EXL)
        if m.hss and not status & C.ST_BEV:
            m._set_css(m.srs_field(C.SRS_PSS), save=False)
    m.lr_valid = False
    return None


def _tlb_op(op: str) -> Executor:
    """TLBP / TLBR / TLBWI / TLBWR / TLBINV / TLBINVF: done by the TLB MMU
    (:mod:`emips.tlb`).  Without a TLB (the fixed mapping) they are
    Reserved Instructions (Vol. III A.1.3), as TLBINV / TLBINVF are
    without ``Config4.IE``."""
    def run(m: _M, w: int) -> int | None:
        fn = getattr(m.mmu, op, None)
        if fn is None or op.startswith("tlbinv") and not m.mmu.invalidate:
            raise _Arch(C.RI)
        fn(m)
        return None
    run.__name__ = f"_e_{op}"
    return run


def _e_deret(m: _M, w: int) -> int | None:
    """DERET (MD00047 2.8): DM ← 0, IEXI ← 0, return to DEPC; no delay
    slot, LLbit kept, a hazard barrier.  Outside Debug Mode UNDEFINED:
    Reserved Instruction here (DEVIATIONS "EJTAG")."""
    if not m._dm:
        raise _Arch(C.RI)
    cell = m.cp0[C.DEBUG]
    cell["value"] &= ~(EJ.DM | EJ.IEXI)
    m._dm = False
    m._redirect = m.cp0_read(*C.DEPC)
    return None


def _e_sdbbp(m: _M, w: int) -> int | None:
    """SDBBP code (MD00047 2.8): a Debug Breakpoint exception — in Debug
    Mode a Debug Mode Breakpoint exception (DExcCode = Bp).  The code
    field is for the handler only (it loads the word at DEPC)."""
    raise _Arch(C.DBP, info={"code": (w >> 6) & 0xFFFFF})


def _e_rdpgpr(m: _M, w: int) -> int | None:
    """RDPGPR rd, rt: GPR[rd] ← SGPR[SRSCtl.PSS, rt] (with one register
    set the previous set is the current one)."""
    r = rt(w)
    m.write_reg(rd(w), m.shadow[m.srs_field(C.SRS_PSS)][r] & M32 if r else 0)
    return None


def _e_wrpgpr(m: _M, w: int) -> int | None:
    """WRPGPR rd, rt: SGPR[SRSCtl.PSS, rd] ← GPR[rt]."""
    r = rd(w)
    if r:
        m.shadow[m.srs_field(C.SRS_PSS)][r] = m.read_reg(rt(w))
    return None


def _e_rdhwr(m: _M, w: int) -> int | None:
    """RDHWR rt, rd: hardware registers 0 CPUNum, 1 SYNCI_Step (the L1
    line size; 0 without caches), 2 CC (Count), 3 CCRes (2), 5 XNP (Release 6: 1, no paired
    LL/SC), 29 UserLocal (Release 6, ULRI).  Needs kernel mode or the
    ``HWREna`` bit; anything else is Reserved Instruction."""
    num = rd(w)
    values = {0: m.cp0_read(*C.EBASE) & 0x3FF,
              1: m.caches.synci_step if m.caches is not None else 0,
              2: m.cp0_read(*C.COUNT), 3: m.ccres}
    if m.cp0_read(*C.CONFIG3) >> 13 & 1:          # ULRI (Release 6)
        values[29] = m.cp0_read(*C.USERLOCAL)
    if m.release == "R6":
        values[5] = 1                             # XNP: no LLWP/SCWP
    if num not in values or not (m.kernel_mode
                                 or m.cp0_read(*C.HWRENA) >> num & 1):
        raise _Arch(C.RI)
    m.write_reg(rt(w), values[num])
    return None


# -- Traps / BREAK / SYSCALL ------------------------------------------------


def _trap(cond: bool, code: int = 0) -> int | None:
    if cond:
        raise _Arch(C.TR, info={"code": code})
    return None


def _tcode(w: int) -> int:
    return (w >> 6) & 0x3FF


def _simm(w: int) -> int:
    return _s16(imm16(w)) & M32


def _e_teq(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) == m.read_reg(rt(w)), _tcode(w))


def _e_tne(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) != m.read_reg(rt(w)), _tcode(w))


def _e_tlt(m: _M, w: int) -> int | None:
    return _trap(_s32(m.read_reg(rs(w))) < _s32(m.read_reg(rt(w))), _tcode(w))


def _e_tltu(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) < m.read_reg(rt(w)), _tcode(w))


def _e_tge(m: _M, w: int) -> int | None:
    return _trap(_s32(m.read_reg(rs(w))) >= _s32(m.read_reg(rt(w))), _tcode(w))


def _e_tgeu(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) >= m.read_reg(rt(w)), _tcode(w))


def _e_teqi(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) == _simm(w))


def _e_tnei(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) != _simm(w))


def _e_tgei(m: _M, w: int) -> int | None:
    return _trap(_s32(m.read_reg(rs(w))) >= _s16(imm16(w)))


def _e_tlti(m: _M, w: int) -> int | None:
    return _trap(_s32(m.read_reg(rs(w))) < _s16(imm16(w)))


def _e_tgeiu(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) >= _simm(w))


def _e_tltiu(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) < _simm(w))


def _e_break(m: _M, w: int) -> int | None:
    raise _Arch(C.BP, info={"code": (w >> 6) & 0xFFFFF})


def _e_syscall(m: _M, w: int) -> int | None:
    code = (w >> 6) & 0xFFFFF
    if m.syscall_handler is not None and m.syscall_handler(m, code) is not False:
        return None
    raise _Arch(C.SYS, info={"code": code})


def _e_nop(m: _M, w: int) -> int | None:
    return None


# -- FPU (COP1) ---------------------------------------------------------------
# Vol. I-A "Floating-Point Unit" + Vol. II-A COP1 pages; arithmetic in
# emips.fpu (exact IEEE soft-float, legacy MIPS NaNs).

#: FIR (FCR0): F64, L, W, D, S implemented; no PS / 3D; legacy NaNs.
FIR = (1 << 22) | (1 << 21) | (1 << 20) | (1 << 17) | (1 << 16)
#: Release 1 FPU: 32-bit (no F64, no L format): W, D, S.
FIR_R1 = (1 << 20) | (1 << 17) | (1 << 16)
#: FCSR bits software may write: FCC7..1, FS, FCC0, Cause, Enables, Flags, RM.
FCSR_MASK = 0xFF83FFFF
#: FCSR.ABS2008 | NAN2008: read-only, 1 in Release 6
FCSR_RO_2008 = (1 << 19) | (1 << 18)
#: Release 6 has no FPU condition codes: FCC7..0 (bits 31..25, 23) read 0
FCSR_MASK_R6 = FCSR_MASK & ~0xFE800000
#: FIR.Has2008
FIR_HAS2008 = 1 << 23
_ST_FR = 1 << 26


def _fs(w: int) -> int:
    return (w >> 11) & 31


def _ft(w: int) -> int:
    return (w >> 16) & 31


def _fd(w: int) -> int:
    return (w >> 6) & 31


def _fr(w: int) -> int:
    return (w >> 21) & 31


def _fr64(m: _M) -> bool:
    return bool(m.status & _ST_FR)


def _even(m: _M, n: int) -> None:
    """A 64-bit operand needs an even FPR when Status.FR = 0."""
    if n & 1 and not _fr64(m):
        raise _Arch(C.RI)


def fpr_get(m: _M, n: int, f: str) -> int:
    """Read FPR ``n`` as format ``f`` ('S'/'W' 32-bit, 'D'/'L' 64-bit)."""
    if f in ("S", "W"):
        return m.fpr[n] & M32
    _even(m, n)
    if _fr64(m):
        return m.fpr[n]
    return ((m.fpr[n + 1] & M32) << 32) | (m.fpr[n] & M32)


def fpr_set(m: _M, n: int, f: str, v: int) -> None:
    if f in ("S", "W"):
        m.fpr[n] = (m.fpr[n] & ~M32) | (v & M32)     # FR=1: high half kept
        return
    _even(m, n)
    if _fr64(m):
        m.fpr[n] = v & M64
    else:
        m.fpr[n] = (m.fpr[n] & ~M32) | (v & M32)
        m.fpr[n + 1] = (m.fpr[n + 1] & ~M32) | ((v >> 32) & M32)


def _fmt(f: str) -> F.Fmt:
    return F.S if f == "S" else F.D


def _rm(m: _M) -> int:
    return m.fcsr & 3


def _flush(m: _M) -> bool:
    return bool(m.fcsr >> 24 & 1)


def _fp_commit(m: _M, flags: set[str], write: Callable[[], None]) -> None:
    """Set FCSR.Cause to this operation's exceptions; trap (FPE) when one
    is enabled — destination not written, Flags unchanged — else write the
    result and accumulate Flags."""
    cause = sum(1 << F.FLAG_BITS[x] for x in flags)
    m.fcsr = (m.fcsr & ~(0x3F << 12)) | (cause << 12)
    if cause & (m.fcsr >> 7) & 0x1F:
        raise _Arch(C.FPE)
    m.fcsr |= cause << 2
    write()


def _fcc(m: _M, cc: int) -> int:
    return (m.fcsr >> (23 if cc == 0 else 24 + cc)) & 1


def _set_fcc(m: _M, cc: int, bit: bool) -> None:
    pos = 23 if cc == 0 else 24 + cc
    m.fcsr = (m.fcsr & ~(1 << pos)) | (int(bit) << pos)


def _ex_fp2(op: Callable, f: str) -> Executor:          # fd <- op(fs, ft)
    def ex(m: _M, w: int) -> int | None:
        a, b = fpr_get(m, _fs(w), f), fpr_get(m, _ft(w), f)
        if f == "D":
            _even(m, _fd(w))
        bits, flags = op(a, b, _fmt(f), _rm(m), _flush(m), _n8(m))
        _fp_commit(m, flags, lambda: fpr_set(m, _fd(w), f, bits))
        return None
    return ex


def _n8(m: _M) -> bool:
    """FCSR.NAN2008: IEEE 754-2008 NaNs / ABS (Release 6)."""
    return bool(m.fcsr >> 18 & 1)


def _ex_fp1(op: Callable, f: str) -> Executor:          # fd <- op(fs)
    def ex(m: _M, w: int) -> int | None:
        a = fpr_get(m, _fs(w), f)
        if f == "D":
            _even(m, _fd(w))
        bits, flags = op(a, _fmt(f), m)
        _fp_commit(m, flags, lambda: fpr_set(m, _fd(w), f, bits))
        return None
    return ex


def _madd(neg_product_sum: bool, subtract: bool, f: str) -> Executor:
    """MADD/MSUB/NMADD/NMSUB.fmt fd, fr, fs, ft: (fs*ft) ± fr, negated for
    the N forms — two roundings (not fused in Release 2)."""
    def ex(m: _M, w: int) -> int | None:
        fmt = _fmt(f)
        a, b, c = (fpr_get(m, _fs(w), f), fpr_get(m, _ft(w), f),
                   fpr_get(m, _fr(w), f))
        if f == "D":
            _even(m, _fd(w))
        p, fl1 = F.mul(a, b, fmt, _rm(m), _flush(m), _n8(m))
        r, fl2 = F.add(p, c, fmt, _rm(m), _flush(m), subtract=subtract,
                       nan2008=_n8(m))
        if neg_product_sum:
            r ^= fmt.sign_bit
        _fp_commit(m, fl1 | fl2, lambda: fpr_set(m, _fd(w), f, r))
        return None
    return ex


def _ex_cvt(src: str, dst: str, rm_fixed: int | None = None) -> Executor:
    def ex(m: _M, w: int) -> int | None:
        if "L" in (src, dst) and not _fr64(m):
            raise _Arch(C.RI)                   # L format needs FR = 1
        a = fpr_get(m, _fs(w), src)
        if dst in ("D", "L"):
            _even(m, _fd(w))
        rm = _rm(m) if rm_fixed is None else rm_fixed
        if src in ("W", "L"):
            bits, flags = F.from_int(a, 32 if src == "W" else 64, _fmt(dst), rm)
        elif dst in ("W", "L"):
            bits, flags = F.to_int(a, _fmt(src), 32 if dst == "W" else 64, rm,
                                   nan2008=_n8(m))
        else:
            bits, flags = F.convert(a, _fmt(src), _fmt(dst), rm, _flush(m),
                                    nan2008=_n8(m))
        _fp_commit(m, flags, lambda: fpr_set(m, _fd(w), dst, bits))
        return None
    return ex


def _ex_cmp(cond: int, f: str) -> Executor:
    def ex(m: _M, w: int) -> int | None:
        a, b = fpr_get(m, _fs(w), f), fpr_get(m, _ft(w), f)
        res, flags = F.compare(a, b, _fmt(f), cond, nan2008=_n8(m))
        _fp_commit(m, flags, lambda: _set_fcc(m, (w >> 8) & 7, res))
        return None
    return ex


def _ex_mov(f: str, cond: Callable[[_M, int], bool] | None = None) -> Executor:
    """MOV / MOVF / MOVT / MOVZ / MOVN.fmt: a copy, no exceptions."""
    def ex(m: _M, w: int) -> int | None:
        v = fpr_get(m, _fs(w), f)
        if f == "D":
            _even(m, _fd(w))
        if cond is None or cond(m, w):
            fpr_set(m, _fd(w), f, v)
        return None
    return ex


def _cc_tf(m: _M, w: int) -> bool:                  # MOVF/MOVT: cc 20..18, tf 16
    return _fcc(m, (w >> 18) & 7) == (w >> 16) & 1


def _e_movci(m: _M, w: int) -> int | None:          # integer MOVF / MOVT
    if _cc_tf(m, w):
        m.write_reg(rd(w), m.read_reg(rs(w)))
    return None


def _e_bc1(m: _M, w: int) -> int | None:
    """BC1F/BC1T(L): cc 20..18, nd (likely) bit 17, tf bit 16."""
    taken = _fcc(m, (w >> 18) & 7) == (w >> 16) & 1
    return _branch(m, w, taken, likely=bool(w >> 17 & 1))


def _e_mfc1(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), fpr_get(m, _fs(w), "W"))
    return None


def _e_mtc1(m: _M, w: int) -> int | None:
    fpr_set(m, _fs(w), "W", m.read_reg(rt(w)))
    return None


def _e_mfhc1(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), fpr_get(m, _fs(w), "D") >> 32)
    return None


def _e_mthc1(m: _M, w: int) -> int | None:
    n = _fs(w)
    old = fpr_get(m, n, "D")
    fpr_set(m, n, "D", (old & M32) | (m.read_reg(rt(w)) << 32))
    return None


def _e_cfc1(m: _M, w: int) -> int | None:
    """CFC1 rt, fs: FIR (0), FCCR (25), FEXR (26), FENR (28), FCSR (31)."""
    fcsr, n = m.fcsr, _fs(w)
    views = {
        0: m.fir,
        25: ((fcsr >> 24) & 0xFE) | ((fcsr >> 23) & 1),
        26: fcsr & 0x0003F07C,
        28: (fcsr & 0xF80) | ((fcsr >> 22) & 4) | (fcsr & 3),
        31: fcsr,
    }
    if n not in views:
        raise _Arch(C.RI)
    m.write_reg(rt(w), views[n])
    return None


def _e_ctc1(m: _M, w: int) -> int | None:
    """CTC1 rt, fs (25, 26, 28, 31).  A write that leaves an enabled Cause
    bit set traps (FPE) *after* the write (Vol. II-A CTC1)."""
    v, n, fcsr = m.read_reg(rt(w)), _fs(w), m.fcsr
    if n == 25:
        if m.release != "R6":                   # R6 has no FCCs: ignored
            fcsr = (fcsr & ~(0xFE << 24) & ~(1 << 23)) | ((v & 0xFE) << 24) | ((v & 1) << 23)
    elif n == 26:
        fcsr = (fcsr & ~0x0003F07C) | (v & 0x0003F07C)
    elif n == 28:
        fcsr = (fcsr & ~(0xF83 | (1 << 24))) | (v & 0xF83) | ((v & 4) << 22)
    elif n == 31:
        mask = FCSR_MASK_R6 if m.release == "R6" else FCSR_MASK
        fcsr = (v & mask) | (fcsr & FCSR_RO_2008)   # NAN2008/ABS2008 RO
    else:
        raise _Arch(C.RI)
    m.fcsr = fcsr
    if (fcsr >> 12) & 0x20 or (fcsr >> 12) & (fcsr >> 7) & 0x1F:
        raise _Arch(C.FPE)
    return None


def _fp_load(size: int, indexed: bool = False, unaligned: bool = False) -> Executor:
    def ex(m: _M, w: int) -> int | None:
        base = m.read_reg(rs(w))
        addr = (base + m.read_reg(rt(w))) if indexed else (base + _s16(imm16(w)))
        addr &= M32
        if unaligned:
            if not _fr64(m):
                raise _Arch(C.RI)               # LUXC1 needs FR = 1
            addr &= ~7
        reg = _fd(w) if indexed else _ft(w)
        f = "W" if size == 4 else "L"
        if f == "L":
            _even(m, reg)
        fpr_set(m, reg, f, m.load(m.data_address(addr, size, store=False), size))
        return None
    return ex


def _fp_store(size: int, indexed: bool = False, unaligned: bool = False) -> Executor:
    def ex(m: _M, w: int) -> int | None:
        base = m.read_reg(rs(w))
        addr = (base + m.read_reg(rt(w))) if indexed else (base + _s16(imm16(w)))
        addr &= M32
        if unaligned:
            if not _fr64(m):
                raise _Arch(C.RI)               # SUXC1 needs FR = 1
            addr &= ~7
        reg = _fs(w) if indexed else _ft(w)
        v = fpr_get(m, reg, "W" if size == 4 else "L")
        m.store(m.data_address(addr, size, store=True), v, size)
        return None
    return ex


_C_CONDS = ["F", "UN", "EQ", "UEQ", "OLT", "ULT", "OLE", "ULE",
            "SF", "NGLE", "SEQ", "NGL", "LT", "NGE", "LE", "NGT"]
_RM_OPS = {"ROUND": F.RN, "TRUNC": F.RZ, "CEIL": F.RP, "FLOOR": F.RM}


def _fpu_executors() -> dict[str, Executor]:
    ex: dict[str, Executor] = {}
    for f in ("S", "D"):
        ex[f"ADD.{f}"] = _ex_fp2(
            lambda a, b, fm, rm, fl, n8: F.add(a, b, fm, rm, fl, nan2008=n8), f)
        ex[f"SUB.{f}"] = _ex_fp2(
            lambda a, b, fm, rm, fl, n8: F.add(a, b, fm, rm, fl, subtract=True,
                                               nan2008=n8), f)
        ex[f"MUL.{f}"] = _ex_fp2(F.mul, f)
        ex[f"DIV.{f}"] = _ex_fp2(F.div, f)
        ex[f"SQRT.{f}"] = _ex_fp1(
            lambda a, fm, m: F.sqrt(a, fm, _rm(m), _flush(m), _n8(m)), f)
        ex[f"ABS.{f}"] = _ex_fp1(
            lambda a, fm, m: F.neg_abs(a, fm, negate=False, nan2008=_n8(m)), f)
        ex[f"NEG.{f}"] = _ex_fp1(
            lambda a, fm, m: F.neg_abs(a, fm, negate=True, nan2008=_n8(m)), f)

        def recip(a, fm, m):
            one = F.round_pack(F.Fraction(1), fm, F.RN)[0]
            return F.div(one, a, fm, _rm(m), _flush(m), _n8(m))

        def rsqrt(a, fm, m):
            r, fl1 = F.sqrt(a, fm, _rm(m), _flush(m), _n8(m))
            one = F.round_pack(F.Fraction(1), fm, F.RN)[0]
            q, fl2 = F.div(one, r, fm, _rm(m), _flush(m), _n8(m))
            return q, fl1 | fl2

        ex[f"RECIP.{f}"] = _ex_fp1(recip, f)
        ex[f"RSQRT.{f}"] = _ex_fp1(rsqrt, f)
        ex[f"MADD.{f}"] = _madd(False, False, f)
        ex[f"MSUB.{f}"] = _madd(False, True, f)
        ex[f"NMADD.{f}"] = _madd(True, False, f)
        ex[f"NMSUB.{f}"] = _madd(True, True, f)
        ex[f"MOV.{f}"] = _ex_mov(f)
        ex[f"MOVF.{f}"] = _ex_mov(f, _cc_tf)
        ex[f"MOVT.{f}"] = _ex_mov(f, _cc_tf)
        ex[f"MOVZ.{f}"] = _ex_mov(f, lambda m, w: m.read_reg(rt(w)) == 0)
        ex[f"MOVN.{f}"] = _ex_mov(f, lambda m, w: m.read_reg(rt(w)) != 0)
        for i, c in enumerate(_C_CONDS):
            ex[f"C.{c}.{f}"] = _ex_cmp(i, f)
        for to in ("W", "L"):
            ex[f"CVT.{to}.{f}"] = _ex_cvt(f, to)
            for op, rm in _RM_OPS.items():
                ex[f"{op}.{to}.{f}"] = _ex_cvt(f, to, rm)
        for frm in ("W", "L"):
            ex[f"CVT.{f}.{frm}"] = _ex_cvt(frm, f)
    ex["CVT.D.S"] = _ex_cvt("S", "D")
    ex["CVT.S.D"] = _ex_cvt("D", "S")
    for name in ("BC1F", "BC1T", "BC1FL", "BC1TL"):
        ex[name] = _e_bc1
    ex.update({
        "MOVF": _e_movci, "MOVT": _e_movci,
        "MFC1": _e_mfc1, "MTC1": _e_mtc1, "MFHC1": _e_mfhc1, "MTHC1": _e_mthc1,
        "CFC1": _e_cfc1, "CTC1": _e_ctc1,
        "LWC1": _fp_load(4), "LDC1": _fp_load(8),
        "SWC1": _fp_store(4), "SDC1": _fp_store(8),
        "LWXC1": _fp_load(4, True), "LDXC1": _fp_load(8, True),
        "LUXC1": _fp_load(8, True, True),
        "SWXC1": _fp_store(4, True), "SDXC1": _fp_store(8, True),
        "SUXC1": _fp_store(8, True, True),
    })
    return ex




# -- Release 6 (integer) --------------------------------------------------------
# Vol. II-A Release 6.  Compact branches have no delay slot: a taken one
# leaves its target in ``_redirect``; a not-taken conditional one marks the
# next instruction as its forbidden slot.  Links are PC + 4, written whether
# or not the branch is taken.


def _compact(m: _M, cond: bool, target: int, *, link: bool = False,
             conditional: bool = True) -> int | None:
    if link:
        m.write_reg(31, m.pc + 4)
    if cond:
        m._redirect = target & M32
    elif conditional:
        m._forbidden = True
    return None


def _off16(m: _M, w: int) -> int:
    return m.pc + 4 + (_s16(imm16(w)) << 2)


def _off21(m: _M, w: int) -> int:
    return m.pc + 4 + (_sx(w & 0x1FFFFF, 21) << 2)


def _off26(m: _M, w: int) -> int:
    return m.pc + 4 + (_sx(w & 0x3FFFFFF, 26) << 2)


def _ovf(a: int, b: int) -> bool:
    r = _s32(a) + _s32(b)
    return not -(1 << 31) <= r < (1 << 31)


def _r6_branches() -> dict[str, Executor]:
    R = lambda m, w, f: m.read_reg(f(w))                    # noqa: E731
    S = lambda m, w, f: _s32(m.read_reg(f(w)))              # noqa: E731
    two = {
        "BEQC": lambda m, w: R(m, w, rs) == R(m, w, rt),
        "BNEC": lambda m, w: R(m, w, rs) != R(m, w, rt),
        "BGEC": lambda m, w: S(m, w, rs) >= S(m, w, rt),
        "BLTC": lambda m, w: S(m, w, rs) < S(m, w, rt),
        "BGEUC": lambda m, w: R(m, w, rs) >= R(m, w, rt),
        "BLTUC": lambda m, w: R(m, w, rs) < R(m, w, rt),
        "BOVC": lambda m, w: _ovf(R(m, w, rs), R(m, w, rt)),
        "BNVC": lambda m, w: not _ovf(R(m, w, rs), R(m, w, rt)),
    }
    one = {                                     # condition on rt; link?
        "BEQZALC": (lambda v: v == 0, True), "BNEZALC": (lambda v: v != 0, True),
        "BLEZALC": (lambda v: v <= 0, True), "BGEZALC": (lambda v: v >= 0, True),
        "BGTZALC": (lambda v: v > 0, True), "BLTZALC": (lambda v: v < 0, True),
        "BLEZC": (lambda v: v <= 0, False), "BGEZC": (lambda v: v >= 0, False),
        "BGTZC": (lambda v: v > 0, False), "BLTZC": (lambda v: v < 0, False),
    }
    ex: dict[str, Executor] = {}
    for n, cond in two.items():
        ex[n] = (lambda c: lambda m, w: _compact(m, c(m, w), _off16(m, w)))(cond)
    for n, (cond, link) in one.items():
        ex[n] = (lambda c, l: lambda m, w: _compact(
            m, c(_s32(m.read_reg(rt(w)))), _off16(m, w), link=l))(cond, link)
    ex["BEQZC"] = lambda m, w: _compact(m, m.read_reg(rs(w)) == 0, _off21(m, w))
    ex["BNEZC"] = lambda m, w: _compact(m, m.read_reg(rs(w)) != 0, _off21(m, w))
    ex["BC"] = lambda m, w: _compact(m, True, _off26(m, w), conditional=False)
    ex["BALC"] = lambda m, w: _compact(m, True, _off26(m, w), link=True,
                                        conditional=False)
    ex["JIC"] = lambda m, w: _compact(m, True, m.read_reg(rt(w)) + _s16(imm16(w)),
                                       conditional=False)
    ex["JIALC"] = lambda m, w: _compact(m, True, m.read_reg(rt(w)) + _s16(imm16(w)),
                                         link=True, conditional=False)
    return ex


def _e_aui(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), m.read_reg(rs(w)) + (imm16(w) << 16))
    return None


def _e_auipc(m: _M, w: int) -> int | None:
    m.write_reg(rs(w), m.pc + (imm16(w) << 16))
    return None


def _e_aluipc(m: _M, w: int) -> int | None:
    m.write_reg(rs(w), (m.pc + (imm16(w) << 16)) & ~0xFFFF)
    return None


def _pc19(m: _M, w: int) -> int:
    return (m.pc + (_sx(w & 0x7FFFF, 19) << 2)) & M32


def _e_addiupc(m: _M, w: int) -> int | None:
    m.write_reg(rs(w), _pc19(m, w))
    return None


def _e_lwpc(m: _M, w: int) -> int | None:
    m.write_reg(rs(w), m.load(m.data_address(_pc19(m, w), 4, store=False), 4, "pcload"))
    return None


def _e_align(m: _M, w: int) -> int | None:
    bp = (w >> 6) & 3
    a, b = m.read_reg(rt(w)), m.read_reg(rs(w))
    m.write_reg(rd(w), a if bp == 0 else (a << (8 * bp)) | (b >> (32 - 8 * bp)))
    return None


def _e_bitswap(m: _M, w: int) -> int | None:
    v = m.read_reg(rt(w))
    out = 0
    for i in range(4):
        byte = (v >> (8 * i)) & 0xFF
        out |= int(f"{byte:08b}"[::-1], 2) << (8 * i)
    m.write_reg(rd(w), out)
    return None


def _e_lsa(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), (m.read_reg(rs(w)) << (((w >> 6) & 3) + 1)) + m.read_reg(rt(w)))
    return None


def _e_seleqz(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rs(w)) if m.read_reg(rt(w)) == 0 else 0)
    return None


def _e_selnez(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rs(w)) if m.read_reg(rt(w)) != 0 else 0)
    return None


def _r6_muldiv() -> dict[str, Executor]:
    def mk(fn):
        def ex(m: _M, w: int) -> int | None:
            m.write_reg(rd(w), fn(m.read_reg(rs(w)), m.read_reg(rt(w))))
            return None
        return ex

    def sdiv(a: int, b: int, mod: bool) -> int:
        a, b = _s32(a), _s32(b)
        if b == 0:
            return 0                       # UNPREDICTABLE (QEMU: 0)
        q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
        return a - q * b if mod else q

    return {
        "MUL@R6": mk(lambda a, b: _s32(a) * _s32(b)),
        "MUH": mk(lambda a, b: (_s32(a) * _s32(b)) >> 32),
        "MULU": mk(lambda a, b: a * b),
        "MUHU": mk(lambda a, b: (a * b) >> 32),
        "DIV@R6": mk(lambda a, b: sdiv(a, b, False)),
        "MOD": mk(lambda a, b: sdiv(a, b, True)),
        "DIVU@R6": mk(lambda a, b: a // b if b else 0),
        "MODU": mk(lambda a, b: a % b if b else 0),
    }


def _e_sigrie(m: _M, w: int) -> int | None:
    raise _Arch(C.RI)


def _e_ll_r6(m: _M, w: int) -> int | None:
    _ll(m, w, _addr_e(m, w))
    return None


def _e_sc_r6(m: _M, w: int) -> int | None:
    _sc(m, w, _addr_e(m, w))
    return None


def _r6_executors() -> dict[str, Executor]:
    ex = _r6_branches() | _r6_muldiv()
    ex.update({
        "AUI": _e_aui, "AUIPC": _e_auipc, "ALUIPC": _e_aluipc,
        "ADDIUPC": _e_addiupc, "LWPC": _e_lwpc, "ALIGN": _e_align,
        "BITSWAP": _e_bitswap, "LSA": _e_lsa, "SELEQZ": _e_seleqz,
        "SELNEZ": _e_selnez, "CLZ@R6": _e_clz, "CLO@R6": _e_clo,
        "SIGRIE": _e_sigrie, "LL@R6": _e_ll_r6, "SC@R6": _e_sc_r6,
        "RDHWR@R6": _e_rdhwr,
    })
    return ex


def _r6_fpu() -> dict[str, Executor]:
    """Release 6 FPU: CMP.cond (mask result), SEL*, fused MADDF/MSUBF,
    MIN/MAX(A), RINT, CLASS, BC1EQZ/NEZ (delay slot; bit 0 of ft)."""
    ex: dict[str, Executor] = {}
    conds = {"AF": 0, "UN": 1, "EQ": 2, "UEQ": 3, "LT": 4, "ULT": 5, "LE": 6,
             "ULE": 7, "SAF": 8, "SUN": 9, "SEQ": 10, "SUEQ": 11, "SLT": 12,
             "SULT": 13, "SLE": 14, "SULE": 15, "OR": 17, "UNE": 18, "NE": 19,
             "SOR": 25, "SUNE": 26, "SNE": 27}

    def cmp(cond: int, f: str) -> Executor:
        def e(m: _M, w: int) -> int | None:
            a, b = fpr_get(m, _fs(w), f), fpr_get(m, _ft(w), f)
            res, flags = F.cmp_r6(a, b, _fmt(f), cond, _n8(m))
            ones = M32 if f == "S" else M64
            _fp_commit(m, flags, lambda: fpr_set(m, _fd(w), "W" if f == "S" else "L",
                                                 ones if res else 0))
            return None
        return e

    def sel(kind: str, f: str) -> Executor:
        def e(m: _M, w: int) -> int | None:
            fs_, ft_ = fpr_get(m, _fs(w), f), fpr_get(m, _ft(w), f)
            if kind == "SEL":
                v = ft_ if fpr_get(m, _fd(w), f) & 1 else fs_
            elif kind == "SELEQZ":
                v = fs_ if not ft_ & 1 else 0
            else:
                v = fs_ if ft_ & 1 else 0
            fpr_set(m, _fd(w), f, v)
            return None
        return e

    def fused(subtract: bool, f: str) -> Executor:
        def e(m: _M, w: int) -> int | None:
            a, b, c = (fpr_get(m, _fs(w), f), fpr_get(m, _ft(w), f),
                       fpr_get(m, _fd(w), f))
            r, flags = F.fma(a, b, c, _fmt(f), _rm(m), _flush(m),
                             subtract=subtract, nan2008=_n8(m))
            _fp_commit(m, flags, lambda: fpr_set(m, _fd(w), f, r))
            return None
        return e

    def mm(op: str, f: str) -> Executor:
        return _ex_fp2(lambda a, b, fm, rm, fl, n8: F.minmax(a, b, fm, op, n8), f)

    for f in ("S", "D"):
        for cn, c in conds.items():
            ex[f"CMP.{cn}.{f}"] = cmp(c, f)
        for k in ("SEL", "SELEQZ", "SELNEZ"):
            ex[f"{k}.{f}"] = sel(k, f)
        ex[f"MADDF.{f}"] = fused(False, f)
        ex[f"MSUBF.{f}"] = fused(True, f)
        for op in ("MIN", "MAX", "MINA", "MAXA"):
            ex[f"{op}.{f}"] = mm(op, f)
        ex[f"RINT.{f}"] = _ex_fp1(lambda a, fm, m: F.rint(a, fm, _rm(m), _n8(m)), f)
        ex[f"CLASS.{f}"] = _ex_fp1(lambda a, fm, m: (F.classify(a, fm, _n8(m)), set()), f)
    ex["BC1EQZ"] = lambda m, w: _branch(m, w, not m.fpr[_ft(w)] & 1)
    ex["BC1NEZ"] = lambda m, w: _branch(m, w, bool(m.fpr[_ft(w)] & 1))
    return ex

# ---------------------------------------------------------------------------
# Dispatch table (name → executor)
# ---------------------------------------------------------------------------

# -- Cache maintenance: CACHE, SYNCI, PREF (HINT rows) ------------------------


def _h_cache(m: _M, w: int) -> int | None:
    """CACHE op, offset(base): op = bits 20..16 (rt field)."""
    m.cache_op(rt(w), _addr_i(m, w))
    return None


def _h_cache9(m: _M, w: int) -> int | None:
    """CACHE (Release 6) / CACHEE: 9-bit offset in bits 15..7."""
    m.cache_op(rt(w), _addr_e(m, w))
    return None


def _h_synci(m: _M, w: int) -> int | None:
    m.synci(_addr_i(m, w))
    return None


def _h_pref(m: _M, w: int) -> int | None:
    """PREF hint, offset(base): hint = bits 20..16."""
    m.prefetch(rt(w), _addr_i(m, w))
    return None


def _h_pref9(m: _M, w: int) -> int | None:
    """PREF (Release 6) / PREFE: 9-bit offset."""
    m.prefetch(rt(w), _addr_e(m, w))
    return None


def _h_prefx(m: _M, w: int) -> int | None:
    """PREFX hint, index(base): hint = bits 15..11."""
    m.prefetch(rd(w), (m.read_reg(rs(w)) + m.read_reg(rt(w))) & M32)
    return None


def _h_sync(m: _M, w: int) -> int | None:
    """SYNC stype (bits 10..6)."""
    m.sync((w >> 6) & 31)
    return None


#: HINT rows that do something (on the caches): the others are no-ops.
_HINT_EXECUTORS: dict[str, Executor] = {
    "CACHE": _h_cache, "CACHE@R6": _h_cache9, "CACHEE": _h_cache9,
    "SYNCI": _h_synci, "PREF": _h_pref, "PREF@R6": _h_pref9, "PREFE": _h_pref9,
    "PREFX": _h_prefx, "SYNC": _h_sync,
}

_EXECUTORS: dict[str, Executor] = {
    "ADD": _e_add, "ADDI": _e_addi, "ADDIU": _e_addiu, "ADDU": _e_addu,
    "AND": _e_and, "ANDI": _e_andi,
    "B": _e_b, "BAL": _e_bal, "BEQ": _e_beq, "BEQL": _e_beql,
    "BGEZ": _e_bgez, "BGEZAL": _e_bgezal, "BGEZALL": _e_bgezall,
    "BGEZL": _e_bgezl, "BGTZ": _e_bgtz, "BGTZL": _e_bgtzl,
    "BLEZ": _e_blez, "BLEZL": _e_blezl, "BLTZ": _e_bltz,
    "BLTZAL": _e_bltzal, "BLTZALL": _e_bltzall, "BLTZL": _e_bltzl,
    "BNE": _e_bne, "BNEL": _e_bnel, "BREAK": _e_break,
    "CLO": _e_clo, "CLZ": _e_clz, "DERET": _e_deret, "DI": _e_di,
    "DIV": _e_div, "DIVU": _e_divu, "EI": _e_ei, "ERET": _e_eret,
    "TLBP": _tlb_op("tlbp"), "TLBR": _tlb_op("tlbr"), "TLBWI": _tlb_op("tlbwi"),
    "TLBWR": _tlb_op("tlbwr"), "TLBINV": _tlb_op("tlbinv"),
    "TLBINVF": _tlb_op("tlbinvf"),
    "EXT": _e_ext, "INS": _e_ins,
    "J": _e_j, "JAL": _e_jal, "JALR": _e_jalr, "JALR.HB": _e_jalr_hb,
    "JALX": _e_jalx, "JR": _e_jr, "JR.HB": _e_jr_hb,
    "LB": _e_lb, "LBE": _e_lbe, "LBU": _e_lbu, "LBUE": _e_lbue,
    "LH": _e_lh, "LHE": _e_lhe, "LHU": _e_lhu, "LHUE": _e_lhue,
    "LL": _e_ll, "LLE": _e_lle, "LUI": _e_lui, "LW": _e_lw, "LWE": _e_lwe,
    "LWL": _e_lwl, "LWLE": _e_lwle, "LWR": _e_lwr, "LWRE": _e_lwre,
    "MADD": _e_madd, "MADDU": _e_maddu, "MFC0": _e_mfc0, "MFHI": _e_mfhi,
    "MFLO": _e_mflo, "MOVN": _e_movn, "MOVZ": _e_movz, "MSUB": _e_msub,
    "MSUBU": _e_msubu, "MTC0": _e_mtc0, "MTHI": _e_mthi, "MTLO": _e_mtlo,
    "MUL": _e_mul, "MULT": _e_mult, "MULTU": _e_multu,
    "NAL": _e_nal, "NOP": _e_nop, "NOR": _e_nor, "OR": _e_or, "ORI": _e_ori,
    "RDHWR": _e_rdhwr, "RDPGPR": _e_rdpgpr, "ROTR": _e_rotr,
    "ROTRV": _e_rotrv,
    "SB": _e_sb, "SBE": _e_sbe, "SC": _e_sc, "SCE": _e_sce, "SEB": _e_seb,
    "SEH": _e_seh, "SH": _e_sh, "SHE": _e_she, "SLL": _e_sll,
    "SLLV": _e_sllv, "SLT": _e_slt, "SLTI": _e_slti, "SLTIU": _e_sltiu,
    "SLTU": _e_sltu, "SRA": _e_sra, "SRAV": _e_srav, "SRL": _e_srl,
    "SRLV": _e_srlv, "SUB": _e_sub, "SUBU": _e_subu, "SW": _e_sw,
    "SWE": _e_swe, "SWL": _e_swl, "SWLE": _e_swle, "SWR": _e_swr,
    "SWRE": _e_swre, "SYSCALL": _e_syscall, "SDBBP": _e_sdbbp, "SDBBP@R6": _e_sdbbp,
    "TEQ": _e_teq, "TEQI": _e_teqi, "TGE": _e_tge, "TGEI": _e_tgei,
    "TGEIU": _e_tgeiu, "TGEU": _e_tgeu, "TLT": _e_tlt, "TLTI": _e_tlti,
    "TLTIU": _e_tltiu, "TLTU": _e_tltu, "TNE": _e_tne, "TNEI": _e_tnei,
    "WAIT": _e_wait, "WRPGPR": _e_wrpgpr, "WSBH": _e_wsbh, "XOR": _e_xor, "XORI": _e_xori,
}

_EXECUTORS.update(_fpu_executors())
_EXECUTORS.update(_r6_executors())
_EXECUTORS.update(_r6_fpu())

#: Row names with an executor (dispatched by name, never by ``funct``).
EXEC_NAMES: frozenset[str] = frozenset(_EXECUTORS)
