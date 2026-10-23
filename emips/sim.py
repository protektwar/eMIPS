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
    Blocked,
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
from . import dsp as DSP
from . import msa as _V
from . import mt as MT
from .itc import ERRCTL_ITC as ITC_ERRCTL
from . import micromips as MM
from . import mips16 as M16
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
#: the full pre-R6 table: the MIPS32 expansions of MIPS16e instructions
#: (SEB / SEH exist in MIPS16e on a Release 1 core too)
_ISA_R5 = ISA.for_release("R5")

#: Documented choices where the manual leaves behavior UNPREDICTABLE /
#: implementation-dependent, or where the simulator is knowingly partial.
DEVIATIONS: dict[str, str] = {
    "DIV/DIVU by zero": "UNPREDICTABLE in the manual; HI = LO = 0xFFFFFFFF "
                        "here (no exception, as on real cores); on MIPS64 all "
                        "ones (sign-extended), and DDIV / DDIVU the same.",
    "MIPS64 addressing": "the MIPS64 core (emips.sim64) with the fixed mapping "
                         "has Config.AT = 1 (Vol. III A.1: a fixed-mapping MIPS64 "
                         "is assumed to have 32-bit addresses): only the 32-bit "
                         "compatibility segments; Status.KX / SX / UX read 0 and "
                         "ignore writes (9.30), so any other 64-bit address is an "
                         "Address Error. With a TLB it has Config.AT = 2: all of "
                         "Table 4.2, SEGBITS = 40 (VPN2 39:13, the manual's "
                         "example) and PABITS = 32 (the EntryLo PFN of the 24K "
                         "TLB eMIPS models; physical addresses stay 32 bits, so "
                         "each xkphys window is 4 GB). Code in Kernel / Supervisor "
                         "Mode whose effective address carries out of bit 31 with "
                         "64-bit addressing off is UNPREDICTABLE (4.10): an "
                         "Address Error here (the address is not a compatibility "
                         "one). xkuseg while Status.ERL = 1 is implementation "
                         "dependent (Table 4.2, 4.8): only the 2 GB kuseg becomes "
                         "unmapped, the rest of xuseg stays mapped.",
    "MIPS64 TLB": "the 64-bit TLB keeps the 24K's MIPS32 TLB (entries, micro-"
                  "TLBs, Random, PageMask) with 64-bit EntryHi (R 63:62, VPN2 "
                  "39:13, Fill 61:40 reads 0), Context (PTEBase 63:23) and "
                  "XContext (PTEBase 63:33, R 32:31, BadVPN2 30:4); EntryLo "
                  "keeps RI / XI where MIPS32 has them (31 / 30, what MFC0 / "
                  "MTC0 see) and DMFC0 / DMTC0 move them to 63 / 62; the PFN "
                  "is PA 31:12 (bits 29:26 and PFNX are 0).",
    "MIPS64 pipeline": "a MIPS64 program runs on the 5-stage pipeline; the 24K "
                       "model is a MIPS32 core and becomes the 5-stage one. "
                       "DMULT / DMULTU busy the MDU 9 cycles, DDIV / DDIVU 67: the "
                       "MIPS64 5K's latencies (MD00109, the 5Kf datasheet, Table "
                       "3: its 32-bit DIV is 35, eMIPS's DIV; GCC's 5k.md counts "
                       "them one less); Release 6's DMUL / DMUH ... hold "
                       "EX for DMULT's latency, DDIV / DMOD ... for DDIV's (the "
                       "5K has no Release 6). The pipeline takes its next PC from "
                       "the Machine (fetch_va / seq_pc / branch_pc: the "
                       "compatibility fetch, Release 6's wrap).",
    "MIPS64 ELF and C": "an ELF64 (the n64 ABI) runs on MIPS64, an ELF32 on "
                        "MIPS32: the ELF's class sets the session's architecture "
                        "(a real MIPS64 also runs o32 code). n32 (an ELF32 with "
                        "64-bit code, EF_MIPS_ABI2) is refused. Addresses in the "
                        "32-bit compatibility space are kept as their 32-bit "
                        "form (as the -msym32 assembler's), so xuseg "
                        "0x80000000-0xFFFFFFFF (which that form would make "
                        "kseg0) cannot be loaded; above 4 GB needs a TLB "
                        "(Config.AT = 2). C is n64 with -msym32 (GCC; clang has "
                        "no -msym32 and builds 64-bit address sequences): "
                        "crt0 loads addresses with lui + addiu, because -msym32's "
                        "lui + daddiu of __stack_top 0x7FFFEFF0 (%hi = 0x8000) "
                        "gives 0xFFFFFFFF7FFFEFF0.",
    "MIPS64 EJTAG and trace": "a 64-bit core's drseg registers are 64 bits "
                              "and take doublewords; another size (UNDEFINED, "
                              "MD00047 2.2.2.2) reads / writes those bytes of "
                              "the register. Its TAP's Address register stays "
                              "32 bits (eMIPS's physical addresses; 4.5.4: the "
                              "PA's width); PCsample is 73 bits (81 with the MT "
                              "Module's TC: see MT Module). MIPS Trace sends a 64-bit "
                              "PC / address that is a sign-extended word as one "
                              "32-bit part with an early TEnd (MD00439 6.4, "
                              "optional), any other in two; data values always "
                              "whole (6.3: no compression). The decoder never "
                              "lets an instruction before an overflow (TF5) "
                              "take a record sent after it.",
    "MIPS64 MMUs": "the BAT on MIPS64 keeps 32-bit addresses (Config.AT = 1: "
                   "App. A.2 describes entries for useg / kseg2 / kseg3 only). "
                   "The FTLB indexes the 64-bit VA by the P5600 / P6600 hash "
                   "(MD01138 3.10: VPN2 26:20 ^ 19:13). Segmentation Control: "
                   "SegCtl1.XAM = 6 (no encoding in Table 9.17; the pseudo "
                   "code's XKP) is a kernel-only unmapped mode, as eMIPS's "
                   "AM = 6; EBase is 64 bits (WG) only with Segmentation "
                   "Control, as the write gate. The walker follows the "
                   "P6600 (MD01138 2.2.2.12-15): no Base Directory "
                   "(PWDirExt, BDI, BDW read 0), PTEW = 1 only with PS = 0; "
                   "PS = 0 walks only refills that use the TLB Refill vector "
                   "and PS = 1 only XTLB Refills (Table 9.22, the P6600) — "
                   "the 4.14.3 pseudo code's early 'no segment to map' exit "
                   "would stop every PS = 0 walk, against its own condition "
                   "text; the XK / XS / XU selection and the VA 63:62 prefix "
                   "apply with PS = 1; 32-bit pointers are compatibility "
                   "addresses (sign-extended).",
    "MIPS64 boot firmware": "on a MIPS64 core the boot firmware sets Status.FR "
                            "(the n64 ABI's 64-bit FPRs) and, with a TLB, KX / "
                            "SX / UX — what a 64-bit Linux kernel does (arch/"
                            "mips/kernel/traps.c configure_status); its XTLB "
                            "Refill handler (EBase + 0x080) maps as the TLB "
                            "Refill one by VA 31:0: bit 63 set (kseg2 / kseg3, "
                            "xkseg) PA = VA 31:0 uncached, else VA 31:0 + 1 GB "
                            "cacheable (xuseg / xsseg addresses that differ "
                            "above bit 31 share pages). $sp starts at "
                            "0x7FFFEFF0, 16-byte aligned for the n64 ABI "
                            "(MARS's 0x7FFFEFFC, MIPS32's, is not even "
                            "doubleword-aligned: SD to the stack would fault).",
    "MIPS64 NotWordValue": "a 32-bit instruction whose operand is not a sign-"
                           "extended word (Vol. II-A64: ADD, ADDU, SUB, SRL, "
                           "ROTR, CLZ, EXT, INS, SEB, WSBH, MULT, DIV, MUL ...) "
                           "is UNPREDICTABLE; here it uses the low word. QEMU "
                           "does too, except that its ADD / ADDI / SUB judge "
                           "overflow on bit 63.",
    "MIPS64 LWR": "an LWR that does not load bit 31 may keep GPR[rt][63:32] or "
                  "sign-extend bit 31 (Vol. II-A64 LWR); it sign-extends here, "
                  "as QEMU.",
    "MIPS64 DMFC0 of a 32-bit register": "the manual does not say: DMFC0 "
                                         "sign-extends it (as QEMU), DMTC0 "
                                         "writes the low word.",
    "MIPS64 Status.PX": "Undefined at reset (Vol. III 9.30): 0 here; the user-"
                        "mode boot sets it (64-bit operations in User Mode).",
    "microMIPS64": "Machine(arch=\"mips64\", compressed=\"micromips\") from "
                   "Release 3 (MD00594 3.1: microMIPS came with Release 3; the "
                   "M14K's Release 2 microMIPS is a microMIPS32 core).  The "
                   "recoded instructions run as their MIPS64 twins; LDP / SDP "
                   "/ LDM / SDM / LWUPC have their own executors (64-bit "
                   "operations disabled: Reserved Instruction; LDP rd = $ra "
                   "or = base, any of them in a delay slot: UNDEFINED — RI, "
                   "as LWP).  Release 6 has no oracle: binutils has no "
                   "microMIPS R6, LLVM's microMIPS64r6 back end (MicroMips64r6"
                   "InstrInfo.td) was removed in 2018 — the 6.05 pages decide, and where "
                   "the manual contradicts itself: DLSA is where Table 11.14 "
                   "puts it (POOL32S bits 8:6 = 100, 5:0 = 000000; the page "
                   "and LLVM give 5:0 = 000100, DINSM's minor), DALIGN has rt "
                   "in 25:21 as the page and ALIGN (LLVM swaps rs / rt), LLD "
                   "keeps the page's 12-bit offset (the twin is the Release "
                   "5 LLD; SCD's page has 9 bits), LWUPC's address is (PC & "
                   "~3) + offset as LWPC's (its page omits the & ~3 but "
                   "calls the access naturally aligned).  The assembler: GNU "
                   "as encodes addu rd, rs, $zero / addiu rd, rs, 0 as MOVE16, "
                   "a 64-bit copy — eMIPS keeps ADDU / ADDIU (the low word, "
                   "sign-extended).",
    "MIPS16e64": "MD00077 (MIPS64 Vol. IV-a) rev. 2.60 with the MIPS64-only "
                 "fix MD00076 rev. 2.61 records (an extended SD ra, offset(sp) "
                 "has 0 in bits 7..5 — 2.60's page still shows ra there — else "
                 "Reserved, as the other fixed-zero fields): "
                 "Machine(arch=\"mips64\", compressed=\"mips16e\"), Releases 1-5. "
                 "The MIPS64-only forms run as their MIPS64 twins (DSLL / DSRL / "
                 "DSRA by 32-63 = the *32 forms) or M16.LDPC / DADDIUPC / SEW / "
                 "ZEW (ZEW has no MIPS64 Release 1 instruction, SEW's SLL rx, "
                 "rx, 0 is a 32-bit one), and need 64-bit operations enabled (⊥: "
                 "Reserved Instruction in User Mode with Status.PX = UX = 0). "
                 "The 32-bit forms as MIPS32's on a MIPS64 core: word results "
                 "sign-extended, MOVE / AND / OR / XOR / NOT / CMP / LI / ZEB / "
                 "MFHI ... whole registers; a NotWordValue operand "
                 "(UNPREDICTABLE: ADDIU, ADDU, SUBU, NEG, SLL ..., MULT, DIV, "
                 "SEW, ZEW) gives the result of its low word.  PC-relative "
                 "bases are the 64-bit PC (ADDIUPC / LWPC: & ~3, then a word "
                 "result; LDPC: & ~7; DADDIUPC: all 64 bits); JAL / JALX keep PC "
                 "63..28; links are 64-bit.  SAVE / RESTORE move the low words "
                 "(RESTORE sign-extends them), sp adds / subtracts all 64 bits. "
                 "Options.compressed \"auto\" on MIPS64 gives MIPS16e only to a "
                 "program with MIPS16e code (no MIPS64 core eMIPS models has "
                 "it: the 5K's datasheet MD00109 lists none).  C: GCC builds "
                 "MIPS16 n64 code soft-float only (cc --mips16 --arch mips64: "
                 "the whole program -msoft-float).  No QEMU model has MIPS16e "
                 "on MIPS64: the oracle is the GNU simulator (sim/mips, "
                 "mipsisa64r2-elf), where it agrees with the manual.",
    "RISC-V": "The RISC-V Instruction Set Manual 20250508, Volumes I and II "
              "(emips.riscv, emips.simrv, emips.rvcsr; PLAN M25 phase 1).  Where "
              "the specifications leave a choice: misaligned ordinary loads / "
              "stores are done in hardware (Zicclsm; the P8700 and QEMU do the "
              "same), LR / SC / AMOs raise address misaligned; mtval gets the "
              "faulting address, an illegal instruction's bits (all 16 / 32 of "
              "them) and an EBREAK's PC (Spike's and QEMU's choices); mtvec's "
              "reserved MODE values 2 / 3 lose bit 1 (WARL, as Spike); misa is "
              "read-only; mvendorid / marchid / mimpid are 0 (not implemented); "
              "mhpmcounter3-31 / mhpmevent3-31 read 0 (priv. 3.1.10 allows it); "
              "an SC does not clear its reservation on this hart's own stores "
              "(14.2 allows keeping it; another hart's store will clear it), "
              "nor does a trap or an xRET (priv. 3.3.2; Spike: single-stepping "
              "an LR / SC sequence works); the reservation set is the bytes the "
              "LR read (Spike's is larger: an SC elsewhere in it fails here); "
              "FENCE drains the write buffer, FENCE.I also writes the D-cache "
              "back and invalidates the I-cache (no cycles charged).  The "
              "privilege modes (phase 2a, emips.rvcsr's docstring lists the "
              "WARL choices: Spike's): an unsupported xPP mode becomes the "
              "least privileged one, reset MPP = the least privileged mode, "
              "the WFI time limit is 0 (TW / U-mode WFI illegal at once), "
              "medeleg bit 19 read-only 0.  Virtual memory (phase 2b, "
              "emips.rvmmu): Svade (no hardware A / D updates: a page fault, "
              "as Spike by default), a 64-entry TLB flushed only by SFENCE.VMA "
              "(a satp write flushes nothing: a new ASID does not match), a "
              "satp write with an unsupported MODE changes nothing (Spike "
              "writes ASID / PPN anyway).  PMP (phase 2c, emips.rvpmp): 0 "
              "entries unless asked, granularity 4 bytes, pmpaddr of 54 / 32 "
              "bits, R = 0 with W = 1 drops W (Spike); every PMP CSR exists "
              "even with no entries (Spike: illegal); fetches are checked per "
              "16-bit parcel.  The platform (phase 2d, emips.rvplatform): "
              "Spike's map; mtimecmp resets to all ones (unknown in the "
              "ACLINT), mtime counts every 2 core cycles of hart 0; PLIC 31 "
              "sources / 4 priority bits / contexts 2h, 2h + 1 (Spike's); the "
              "APLIC delivers directly only (DM, BE read-only 0), IPRIOLEN 8; "
              "interrupts are taken before the next instruction.  Debug "
              "(phase 2e, emips.rvdebug): 4 triggers, each of every type "
              "Spike's have (mcontrol, icount, itrigger, etrigger, mcontrol6), "
              "tinfo version 1; mcontrol has no NAPOT match, mcontrol6's "
              "maskmax6 = XLEN - 1, size 0 only, uncertain / hit timing as "
              "Spike; the trace actions 2-4 set hit and nothing else (no trace "
              "encoder), other actions read 0; textra mhselect 0 / 4, sselect "
              "0-2; 5.4's re-entrancy rule 1 with S-mode, tcontrol without; "
              "execute triggers per 16-bit parcel then the whole instruction, "
              "loads address before / data after, an AMO's store address "
              "first, icount counts an instruction as it starts (its tdata1 "
              "reads the count from before), an interrupt counts without "
              "firing a pending icount (Spike drops it); dcsr: stepie, "
              "stopcount, stoptime read 0, prv legalised as MPP, mprven "
              "writable; in Debug Mode control transfers, AUIPC, ECALL and "
              "xRET are refused (4.1 rules 9-12 allow it), counters and time "
              "keep running.  The Debug Module (phase 2f, emips.rvdm): a direct "
              "1.0 DM (abstract commands done by the DM, the program buffer "
              "through Debug Mode execution: hartinfo dataaccess 0 / datasize 0 "
              "/ nscratch 2, no jumps or AUIPC there, commands and bus accesses "
              "complete at once), progbufsize 2, datacount 4, impebreak; every "
              "optional feature but authentication and DM external triggers "
              "(halt and resume groups 0-31, halt-on-reset, hart array mask, "
              "haltsum0-3, Quick Access, register postincrement, aamvirtual as "
              "MPRV with MPP regardless of mprven, abstractauto, SBA 8-64-bit "
              "with sbasize 32 and errors 2 / 3 / 4); a narrower register write "
              "sign-extends, a narrower memory read zero-extends arg0, an "
              "oversized aarsize is not supported (2); hartreset / ndmreset hold "
              "the harts unavailable until released; activating the DM ignores "
              "the rest of that dmcontrol write.  The JTAG DTM: IDCODE 1, dtmcs "
              "version 1, abits 7, idle 0, no errinfo.  Physical addresses "
              "of 32 bits (an "
              "access above 4 GB on RV64 is an access fault), 0xFFFF0000-"
              "0xFFFFFFFF uncached (RARS's MMIO page), the timer "
              "behind time = cycles / CCRes (the ACLINT's mtime: phase 2d).  The decoder follows the "
              "specification where binutils' objdump does not: c.addi16sp with "
              "nzimm = 0 and the RV32 shifts by 32-63 are illegal, FENCE / "
              "FENCE.I execute with their reserved fields set.  F / D (emips.rvfpu): "
              "mstatus.FS is Off at reset (the boot decides: phase 2); a "
              "reserved rounding mode is an illegal instruction on every "
              "instruction with an rm field, the exact conversions too (Spike "
              "and QEMU agree).",
    "nanoMIPS": "MD01247 rev. 01.01 / MD01251 rev. 06.09 (emips.nanomips, "
                "emips.simnano; PLAN M23 phases 1-2).  The x fields are ignored as A.4.3 says: NOP[16] / "
                "NOP[32] / EHB / PAUSE with x bits set are those instructions "
                "(binutils' objdump wants them 0 and prints addiu zero / sll "
                "zero).  BBEQZC / BBNEZC decode with bit 32-63 (their Operation "
                "raises RI on a 32-bit core; objdump does not decode them).  "
                "ADD is not in the nanoMIPS subset: its page says so, the "
                "Instruction Summary leaves it out.  The opcode map's DSP and "
                "MT cells (MD01249 / MD01255) are reserved until M23 phase 5; "
                "its CP1, CP2, UDI, VZ, MCU and MIPS64 cells have no nanoMIPS32 "
                "definition and stay reserved (the error names the cell).  The "
                "text follows objdump's default aliases (li, move, lapc, beqzc "
                "...; text(aliases=False) = -M no-aliases), with $ before register "
                "names and CP0 registers by number; a SAVE / RESTORE list that "
                "starts below $fp and wraps past $31 is written as its runs "
                "(binutils reads past its register names there).  The Machine "
                "(Machine(arch='nanomips')): a Release 6 core with no FPU "
                "(MD01247 has no nanoMIPS FPU instructions), no ISA mode, the "
                "fixed mapping by default; CP0 instructions need Kernel / Debug "
                "Mode or Status.CU0 as MD01251 3.5.3 says (MD01247's "
                "IsCoprocessor0Enabled leaves CU0 out from Release 2; QEMU "
                "and MIPS32 Release 6 grant it).  A misaligned ordinary load "
                "/ store completes (read_memory_at_va's 'optional' is the "
                "implementation's choice and the I7200 manuals do not say; "
                "QEMU raises AdE for its indexed, PC-relative and LWM / SWM "
                "forms, the GNU simulator stops); SAVE / RESTORE and the "
                "linked accesses check alignment as their pages do.  BBEQZC "
                "/ BBNEZC with bit 32-63 raise RI only in User Mode (PX / UX "
                "read 0) and otherwise test bit 31 (A.3.5: GPR values are "
                "signed).  UNPREDICTABLE: MOVEP reads both inputs first, EXT "
                "with pos + size > 32 reads zeros above bit 31, INS with size "
                "< 1 is RI (QEMU), LWM reads its base once, RESTORE of $29 "
                "ends with $29 + u, LLWP with rt = ru leaves ru's word, an SC "
                "/ SCWP that does not match the last LL / LLWP (address or "
                "size) fails.  MFC0 / MTC0 of a missing register read 0 / "
                "ignore the write (their pages); with no extended registers "
                "MFHC0 reads 0 and MTHC0 writes nothing.  PREF hints 24-30 "
                "are NOPs (MD01247: not MIPS Release 6's RI).  CRC32 "
                "(Config5.CRCP), GINVI / GINVT (Config5.GI) and NMS are core "
                "options, off by default; LLWP / SCWP are present (Config5.XNP "
                "= 0, RDHWR 5).  BadInstr holds a 16-bit instruction in bits "
                "31:16, a 48-bit one's last halfword goes to BadInstrX (none on "
                "an NMS core); there is no BadInstrP.  GINVI invalidates this "
                "core's I-cache, GINVT its TLB (other cores: M23 phase 5).",
    "nanoMIPS DSP ASE": "MD01249 rev. 0.04 (PLAN M23 phase 5a) is a preliminary "
                        "document: its 154 formats come from binutils' D32 "
                        "entries, checked against objdump, GNU as -mdsp, QEMU's "
                        "I7200 and the diagrams eMIPS could read (CMP.cond.PH's "
                        "prints the major opcode 0010000, seven digits).  Each "
                        "page runs the MIPS DSP Module's operation (MD00374, "
                        "emips.dsp, Revision 3: Config3.DSPP = DSP2P = 1) on "
                        "nanoMIPS's fields — the same UNPREDICTABLE choices as on "
                        "MIPS32.  BPOSGE32C's Operation text is MIPS32's (a "
                        "17-bit offset, a slot at PC + 4): eMIPS follows its "
                        "diagram (s[13:1] s[14]) and nanoMIPS's branch rule, the "
                        "next instruction + the byte offset, as binutils does; "
                        "QEMU decodes the offset unscaled and unsigned, and its "
                        "MTHLIP takes the accumulator from bits 15..11 "
                        "(tests/test_nanomips_dsp_unicorn.py compares those with "
                        "the manual instead).  The registers each instruction "
                        "reads and writes are its MIPS32 twin's (binutils' "
                        "RD / WR flags are wrong for MTHLIP and others).",
    "nanoMIPS MT ASE": "MD01255 rev. 1.17 (PLAN M23 phase 5b) is the MIPS MT Module "
                       "(MD00378, emips.mt) on nanoMIPS32 — its chapters 4 and 6-10 "
                       "differ from MD00378's only in typography, an "
                       "implementation field in MVPControl (17:16, read 0 here) and "
                       "a note on pre-fetching for suspended TCs — with the eight "
                       "instructions in nanoMIPS encodings (from binutils' MT32 "
                       "entries) and a 5-bit sel and no rx in MFTR / MTTR (Cop2 "
                       "registers: n = rs).  The choices of 'MT Module' apply.  In "
                       "pipeline mode an MT nanoMIPS core runs in instruction mode: "
                       "the multithreaded pipeline is the 34K's (a MIPS32 core); "
                       "the I7200's own 9-stage dual-issue pipeline is not "
                       "modelled.  No oracle: QEMU's I7200 runs one TC.",
    "nanoMIPS MPU": "The I7200's Memory Protection Unit (PLAN M23 phase 5d, "
                    "emips.mpu; MD01232 ch. 4, Table 40): --mmu mpu:N (the MPU "
                    "build: PA = VA for every mode, 16 segments, N regions) or "
                    "fixed+mpu:N (Table 24's FMT build: the fixed mapping with N "
                    "MPU regions), on nanoMIPS cores only.  Neither manual gives "
                    "Config.MT for the MPU and the PRA has no encoding for it: it "
                    "reads 3 (Table 24: 'the core operates in FMT mapping mode').  "
                    "MPUL / MPUS (ExcCode 29) write BadVAddr, as the TLB "
                    "exceptions do (the guide does not say); MPU_Config's last "
                    "exception fields change when one is taken.  A PC-relative "
                    "load is a data read (RI); CACHE and PREF are not checked "
                    "(as the TLB's RI / XI).  In the FMT build regions are "
                    "matched on the virtual address, as the segments are.  A "
                    "region whose Size is not 5-28 matches nothing; a BaseAddress "
                    "below the subregion size's alignment is aligned down.  The "
                    "reset segments (Preset) are uncached with no inhibits; the "
                    "unwritten region fields read 0.  The CDMM covers its "
                    "instantiated blocks only (CDMMSize + 1); the Fast Debug "
                    "Channel's three blocks before the MPU's read 0 and ignore "
                    "writes (the FDC is not modelled); a cached access to the "
                    "CDMM (UNPREDICTABLE, MD01251 ch. 5) fills a line from the "
                    "registers.  In Debug Mode dseg (drseg and dmseg) is never "
                    "checked; outside it the dseg addresses are ordinary memory "
                    "(MD00047).  The boot firmware enables the CDMM at 0xBFC10000 "
                    "(the MPU build: VA = PA, an uncached segment) or PA "
                    "0x1FC10000 (the FMT build: reached at 0xBFC10000 through "
                    "kseg1 — the guide's examples use that PA) and makes the MPU "
                    "build's segments 0-9 cacheable, 10-15 uncached (as its "
                    "Config.K0 / KU / K23); the Malta takes the FMT build only "
                    "(its SDRAM is at physical 0).",
    "nanoMIPS session": "PLAN M23 phase 4a: --arch nanomips / Options.arch "
                        "'nanomips' (the release follows: R6).  The generic "
                        "nanoMIPS core keeps the 24K's implementation registers "
                        "(Config7, the micro-TLBs, EJTAG 3.20, the 24K's counter "
                        "events) as every core preset does; it takes every MMU "
                        "MD01251 defines (fixed, BAT, TLB / FTLB, Segmentation "
                        "Control and EVA, the walker: its boot firmware's refill "
                        "handler is nanoMIPS code), caches, a CPS and Release 6 "
                        "VPs (MD01251 9.5 / 9.8).  Programs are assembled with "
                        "every ASE on and the core's NMS: an instruction the core "
                        "lacks assembles and is a Reserved Instruction when it "
                        "runs (CRC32 / GINVI / GINVT: Options.crc / ginv 'auto' = "
                        "the core has them when the program uses them).  The "
                        "MARS syscall services keep their register numbers "
                        "(the service and the result in $2 = p32's $t4, the "
                        "arguments in $4-$6 = $a0-$a2), as UHI does on nanoMIPS "
                        "(MD01069 ch. 2, DN00179); the FPU services 2 / 3 / 6 / "
                        "7 are unknown services (no FPRs).  MIPS Trace is "
                        "refused: MD00439 6.16 has no nanoMIPS ISAM and the I7200 "
                        "debugs through OCI (MD01227 3.7).  The counters' events: "
                        "every PC-relative branch (BC, BALC, BRSC ...) is a "
                        "branch, JRC / JALRC a jump to a register, jrc $ra / "
                        "RESTORE.JRC a return (jr31), a $0 destination a nop.  "
                        "The views' decode fields are the format's tokens (the "
                        "major opcode named by Tables B.2 / B.3, the decode "
                        "pseudocode's arguments); a probe's handler is "
                        "assembled with .ktext at 0xff200200 (GNU as's .ktext N "
                        "is a subsection).  The 5-stage pipeline (phase 4b): one "
                        "instruction a fetch whatever its size (2, 4 or 6 bytes); "
                        "every branch is compact and resolves in ID (taken: the "
                        "fetched instruction is squashed; no forbidden slots); LWM "
                        "/ SWM, SAVE / RESTORE, LLWP / SCWP move one word a cycle "
                        "in EX (the 32-bit data path); RESTORE.JRC jumps from EX "
                        "after its first word ($ra) and the fetch at its target "
                        "overlaps the other words; MUL ... MODU hold EX as Release "
                        "6's do.  The 24K model is a MIPS32 core (refused).  "
                        "ELF / C / GDB (phase 4c): EM_NANOMIPS executables of the "
                        "p32 ABI (DN00179 ch. 5); nanoMIPS64 / p64 and ELFs whose "
                        ".nanoMIPS.abiflags name the DSP / MT / MCU / VZ / MSA "
                        "ASEs are refused; xNMS (built for the full ISA) refuses "
                        "an --nms core, CRC / GINV switch the core's Config5.CRCP "
                        "/ GI on ('auto'; GINV gives GINVI and GINVT).  C is "
                        "nanomips-elf-gcc (GCC 11.2 of release nanoMIPS-2022.03-01: "
                        "the later releases ship no compiler) and gold, soft float, "
                        "-march=32r6 (32r6s on an NMS core), the nanoMIPS "
                        "assembler's layout (code at 0x00400000; data at "
                        "0x10010000, _gp and the gp-relative small data after it) "
                        "and crt0_nano.S; that release installs libgcc for "
                        "little-endian -march=32r6 only, so other builds that call "
                        "its routines (soft float, 64-bit division) are refused "
                        "with the reason.  GDB: nanoMIPS GDB's description (r0-r31, "
                        "pc 32, status / badvaddr / cause 33-35; no fpu feature: "
                        "the core has no FPU).  The Malta (phase 4d): see 'Boot "
                        "monitor'.  The window (phase 4e): the 5-stage "
                        "datapath shows a nanoMIPS instruction's halfwords, the "
                        "registers its format reads (emips.simnano.uses), the "
                        "immediate its decode steps make and a +2 / 4 / 6 adder; "
                        "the main-control signals are the textbook's, from what "
                        "the instruction does (ALUSrc from its page's ALU "
                        "operand, RegDst the register it writes).  Its gates (phase "
                        "4f, emips.nanocontrol) are eMIPS's design: no nanoMIPS "
                        "decoder is published; the PLAs follow MD01247's opcode "
                        "map (the major opcode, then a pool's fixed bits), "
                        "RegDst names the format's destination field (3 bits) "
                        "and ALUOp the ALU function (nanoMIPS has no common funct "
                        "field).",
    "RISC-V assembler": "emips.rvasm + emips.rvld (PLAN M25 phase 3a) are GNU as "
                        "2.42 and ld 2.42 for one source file, ported from their "
                        "sources: the bytes riscv64-unknown-elf-as -mno-relax + "
                        "ld give (tests/test_rvasm_gnu.py compares them: random "
                        "instruction texts of every row, the macros, branch "
                        "relaxation, alignment, data, .insn; and with gas's default "
                        "-mrelax ld's relaxation: calls, gp- / x0-relative "
                        "accesses, c.lui, R_RISCV_ALIGN, random programs).  "
                        "assemble(relax=, link_relax=) = -mrelax / -mno-relax "
                        "and ld's default / --no-relax.  The relaxation keeps "
                        "ld's own behaviour: a .align assembled under .option "
                        "norvc after code ld turns into 2-byte instructions "
                        "(the object's RVC flag never clears) fails with ld's "
                        "\"bytes required for alignment\"; an .align in code "
                        "assembled without relaxation in a file that ends with "
                        "relaxation on is filled with zero bytes (gas's "
                        "riscv_handle_align reads the final option), as is a "
                        "code section's end padding under -mrelax.  A local "
                        "symbol equated to another (x = y + 4) relocates "
                        "against y + 4 (gas's adjust_reloc_syms) and a local "
                        "absolute one becomes a constant without "
                        "R_RISCV_RELAX (fixup_segment); a global equate is "
                        "relaxed as an absolute symbol.  "
                        "eMIPS lays the sections out itself, with RARS's "
                        "addresses: .text (then .rodata) at 0x00400000, .data, "
                        ".sdata (.srodata first), .sbss, .bss at 0x10010000; "
                        "__global_pointer$ is ld's default script's MIN("
                        "__SDATA_BEGIN__ + 0x800, MAX(__DATA_BEGIN__ + 0x800, "
                        "__BSS_END__ - 0x800)).  A static link of position-"
                        "dependent code: TLS and GOT relocations (%tprel_*, "
                        "%got_pcrel_hi, la under .option pic) are refused by the "
                        "linker.  CSR names are gas's default privileged spec "
                        "(1.11).  Where gas aborts (an .insn cb / cj that must "
                        "grow to reach its target) eMIPS refuses the line with a "
                        "message.  Not modelled, refused by name: .macro, .rept / "
                        ".irp, conditional assembly, .include / .incbin, .org, "
                        ".comm / .lcomm, .uleb128 / .sleb128, .octa, .reloc, "
                        ".pushsection / .popsection, .float16; .cfi_* / "
                        ".attribute / .type / .size are accepted and ignored "
                        "(no .eh_frame, no attributes section: eMIPS links no "
                        "ELF object).",
    "RARS services": "emips.rvservices (PLAN M25 phase 3c, decision 10) is RARS "
                     "1.6's ecall services (rars/riscv/syscalls/*.java): a7 the "
                     "service, a0-a3 / fa0 the arguments, registers read as Java "
                     "ints (the low 32 bits, also on RV64) and written "
                     "sign-extended, Java's number formats and java.util.Random.  "
                     "eMIPS has no host clock, sound, dialogs or files in a "
                     "program's reach and must replay a run exactly: time (30) "
                     "counts the hart's cycles at 1 MHz from 0, sleep (32) and "
                     "the synchronous MIDI (33) let that time pass, MIDI (31) is "
                     "silent, an unseeded random stream starts from seed 0, "
                     "GetCWD (17) gives \"/\", open (1024) fails (-1), read / "
                     "write / close / lseek know descriptors 0-2 only (read of "
                     "0 = one input line and its newline), the dialogs (50-60) "
                     "are not offered (an environment call exception).  RARS's "
                     "MMIO keyboard and display (= MARS's at 0xffff0000) raise "
                     "the machine external interrupt (mip.MEIP: the flat board "
                     "has no PLIC — RARS itself uses its own user-level "
                     "interrupt cause).  The boots (decision 11): kernel = "
                     "M-mode at the entry with sp, gp = __global_pointer$ and "
                     "mstatus.FS = Initial; user = the same then U-mode (as mret "
                     "leaves it) with mcounteren / scounteren granting cycle / "
                     "time / instret; reset = the hart's reset state at the "
                     "entry.  Strict mode (assemble pseudo=False) refuses GNU's "
                     "macros (li of a large constant, la, call ...) and names "
                     "their expansion; aliases (mv, ret, li of a small one: "
                     "c.li) are real encodings and stay.",
    "P8700 custom instructions": "MD01502 Appendix B as binutils 2.45 names them "
                                 "(mips.ccmov, mips.ldp / lwp / sdp / swp, mips.pref, "
                                 "mips.ehb / ihb / pause; PLAN M25 phase 5a): a hart has "
                                 "them with the extensions xmipscmov / xmipslsp / "
                                 "xmipscbop / xmipsexectl (App. B's HART.udi).  The pairs "
                                 "follow the P8700 (CONFIG.pair_aligned = 0): any "
                                 "alignment works, as two accesses (the 16- / 8-byte "
                                 "atomicity of an aligned pair is not observable on one "
                                 "hart); LDP / LWP with rd = rd2 (UNKNOWN) leaves the upper "
                                 "value.  mips.pref prefetches into the D-cache (hint "
                                 "8-15) — load (0), store (1), prepare for store (7: a "
                                 "zero-filled line, as App. B's fill_cache allows); the "
                                 "I-cache / L2 / L3 targets and reserved types do nothing "
                                 "(\"any action, including doing nothing\"); no exception "
                                 "ever.  mips.ihb serializes the pipeline (its "
                                 "instructions are fetched again); mips.ehb / pause are "
                                 "no-ops.  GNU as 2.45.50 answers a vendor instruction "
                                 "without its extension with \"internal: unreachable "
                                 "INSN_CLASS_*\"; eMIPS names the extension.",
    "P8700 hart": "emips.p8700 (PLAN M25 phase 5b; MD01502 ch. 7 / 11): mimpid reads 0 "
                  "and marchid UARCH 1 (\"from configuration\": QEMU's mips-p8700); "
                  "\"Undefined\" reset values are 0, except mipsintctl = 0x3F, the "
                  "standard routing (OpenSBI's P8700 code never writes it, so the hardware "
                  "must come up so); mipstvec in vectored mode keeps BASE as written "
                  "(HART.vectored_int_align is not given: 4 bytes); the custom exceptions' "
                  "mtval is the instruction (as an illegal instruction's; the software "
                  "walk's TLB misses: the VA, \"P8700 MMU\").  The P8700 has no AMOs (MD01502 rev. 1.84 1.10.2.2): an AMO "
                  "checks alignment, translation and PMP before it traps "
                  "(\"no addressing related exceptions\"); LR / SC are not affected.  The "
                  "PMAs: pmacfg k belongs to PMP entry k (OpenSBI's mips_p8700_pmp_set), "
                  "the lowest-numbered entry whose range holds the address (whatever its "
                  "permissions) decides, reserved CCA codes are uncached, no entry = the "
                  "platform's attributes, the speculation bit changes nothing (no "
                  "speculation is modelled); 64 PMP entries (one per PMA).  ERL makes every "
                  "access uncached and masks every interrupt; xRET clears it.  Custom "
                  "interrupts (mip 16-20, 6, 10 when routed there, 13 = the counters' "
                  "overflow as Sscofpmf's LCOFI) are M-mode interrupts (mie writable, "
                  "LCOFI delegable), in the AIA's default priority order.  mipsconfig7's "
                  "performance bits change nothing but DMALN / DULS (misaligned loads and "
                  "stores trap); the prefetch trackers' registers are stored (no "
                  "prefetching); mipsconfig9 writes the selected register, mipsconfig10 "
                  "reads it.  The counters: an event counts when it happens in the hart "
                  "(\"edge counting\" per occurrence); a cycle event per clock; eMIPS's "
                  "hart issues and graduates an instruction in one cycle (the 5-stage "
                  "pipeline: in ID or EX), so the *_isu events equal the *_grad ones; "
                  "mispredictions are the 5-stage pipeline's (predicted not taken: every "
                  "taken branch, every JALR); the events eMIPS has no source for never "
                  "count (perf events lists them).  mipsdsprambase / mipscacheerr / "
                  "mipserrctl are stored (no DSPRAM; caches: phase 5e).  The boot firmware "
                  "(kernel / user boot) is OpenSBI's setup: PMP + PMAs (I/O uncached, the "
                  "rest cacheable) and mipsconfig6 AMO_II / RDTM_II (an AMO or a time read "
                  "is an illegal instruction); unlike OpenSBI it leaves DMALN clear, as it "
                  "emulates nothing.  GNU as 2.45 has no names for these CSRs: write "
                  "their numbers (csrw 0x7c0, t0) or .equ names.",
    "P8700 MMU": "emips.p8700mmu (PLAN M25 phase 5c; MD01502 rev. 1.84 ch. 2, 7.3, App. B): the "
                 "sizes are the largest (VTLB 64 dual entries, ITLB 16, DTLB 32 dual entries, "
                 "FTLB 128 x 4); physical addresses stay eMIPS's 32 bits (the P8700: 48).  "
                 "Choices where the manual is silent: the FTLB set = the VPN above the pair, "
                 "modulo 128 (no hash given); the micro-TLBs hold 4 KB / 64 KB pieces (a larger "
                 "page: 64 KB pieces), LRU, and every miss costs UTLB cycles in the pipeline; "
                 "the walker and MTLBWR fill one half of a dual entry (the PTE they have); "
                 "FTLBP counts writes (one in N, deterministic); a change of FTLB64 empties "
                 "the FTLB; the hardware walker caches no non-leaf entry and writes no A / D "
                 "bit (Svade, as the generic hart).  The software walk: the P8700 has "
                 "non-leaf TLB entries (HART.tlb_non_leaf: mipsconfig5's fields are "
                 "described, MTLBWR's level must be 0), so a miss with no non-leaf entry is "
                 "satpTLBMiss (60, the root PTE) and one below a non-leaf entry is "
                 "Instruction / Load / Store TLB Miss (24 / 25 / 27); MTLBWR of an invalid PTE "
                 "writes an entry whose use is the standard page fault (a MIPS refill "
                 "handler's way); XATP reads 0 (no H extension) and LEVEL is WARL 0-3.  MTW "
                 "makes M-mode loads / stores physical with PMP's S-mode rights (priv. 3.7.2: "
                 "the walker's); MPPPS / MTW / MPTW follow 7.3.9 on every M-mode trap, MRET "
                 "restores them and clears MPPPS.  Svnapot (64 KB pages) is on: the TLBs "
                 "and walker have 64 KB pages, and Svnapot is how RISC-V page tables hold "
                 "them.  TLB ECC (mipscacheerr 4 / 5): phase 5e.",
    "RISC-V pipeline": "emips.rvpipeline (PLAN M25 phase 4a) is Patterson & Hennessy's "
                       "RISC-V edition (ch. 4) with its 4.8 choice: branches, jal and "
                       "jalr resolve in ID, predicted not taken (one bubble when taken), "
                       "forwarding EX/MEM and MEM/WB -> EX and EX/MEM -> ID; the "
                       "hazard unit stalls a load's user (a store of the loaded value "
                       "too: its data is needed in EX), a branch / jalr one cycle after "
                       "an ALU result, two after a load.  eMIPS's choices: multiplies, "
                       "divides and the F / D arithmetic hold EX for the MIPS "
                       "pipeline's latencies (LATENCY: MUL 4, DIV 35, FP_* 4 / 17 / 32; "
                       "the textbook has one-cycle units), the FPU is a blocking EX "
                       "unit; CSR writes, fence.i, sfence.vma, xRET, ecall, ebreak and "
                       "wfi serialize (the instructions behind them are fetched "
                       "again, as Rocket replays after a CSR write) — so every fetch "
                       "sees the privilege, translation, PMP and triggers the "
                       "instruction model sees; the execute triggers look at an "
                       "instruction when it issues, not when it is fetched ahead; code "
                       "stored without fence.i may run stale (Zifencei allows it); "
                       "traps are taken in EX, a branch's (a misaligned target) in ID "
                       "— everything older has executed; interrupts before the next "
                       "instruction that would execute; Debug Mode drops what was "
                       "fetched and refetches at dpc.  The caches' memory timing is the "
                       "MIPS pipeline's (emips.timing).",
    "RISC-V C and GDB": "emips.cc / emips/c (PLAN M25 phase 3d): RISC-V C runs "
                        "freestanding on RARS's map with RARS's ecalls (emips.h): "
                        "print_long prints the low 32 bits (RARS reads a Java int), "
                        "exit is Exit2 (93); -march is the hart's extensions "
                        "(zicntr / zihpm included: GCC 13 accepts them), the ABI "
                        "ilp32 / lp64 with d or f when the hart has D or F, medlow "
                        "addresses, the GCC's libgcc multilib for them (clang links "
                        "none: soft float / division without M are refused with "
                        "the reason).  An ELF built for extensions the hart lacks "
                        "loads with a note; its instructions trap as illegal.  The "
                        "GDB stub's CSR feature lists every CSR the hart has "
                        "(reads / writes as a csrr / csrw in M-mode would, with no "
                        "privilege check; a read-only one ignores the write), "
                        "priv (4161) writes only a mode the hart has; dcsr / dpc "
                        "are listed though only Debug Mode reaches them.",
    "nanoMIPS assembler": "emips.nanoasm (PLAN M23 phase 3) is GNU as's nanoMIPS "
                          "port + gold for one source file: the bytes "
                          "nanomips-elf-as + ld give (tests/test_nanomips_asm.py "
                          "compares them).  Where gas cannot assemble a line "
                          "eMIPS does not either, with gas's message -- except "
                          "sh $r, label with the label defined later: gas gives "
                          "the gp-relative SH the byte format and stops with an "
                          "internal error; eMIPS emits the lui + sh it emits for "
                          "a label defined earlier (an expansion that needs an "
                          "instruction the core lacks, such as div $r, $s, -1's "
                          "NEG on an NMS core, is refused: gas's internal error).  "
                          "eMIPS lays the sections out itself (gas has no "
                          "addresses): code at the text base, then at the data "
                          "base .data, .rdata, .rodata, ALIGN(8), _gp, .ssdata / "
                          ".ssbss / .sdata, ALIGN(4), .sbss, .bss (uhi32.ld's "
                          "order, .data first as the MIPS32 assembler has it), "
                          ".ktext / .kdata (eMIPS's kernel segments) at theirs; "
                          "subsections (.text 1) are refused.  .asciiz is "
                          ".asciz (MARS).  Not modelled: PIC / TLS relocation "
                          "operators (%got ..., %tprel ...), the balc stubs "
                          "(off by default in gas), .jumptable, .insn, .reloc, "
                          "macros and conditional assembly (.macro, .if).  "
                          "Strict mode (pseudo=False) refuses gas's macros and "
                          "names what they expand to.  .linkrelax (phase 3b, "
                          "emips.nanold: gold's expansions and, with link= "
                          "{'relax': True}, its relaxations) keeps gold's own "
                          "behaviour: a .set nolinkrelax takes effect after the "
                          "instruction it precedes (gas sorts the NORELAX after "
                          "that instruction's relocation) and a .set linkrelax "
                          "one instruction late; only the first .align after a "
                          "transformed instruction is padded again (a later one "
                          "keeps its padding even when that misaligns what "
                          "follows); an .align N, 0 in code loses its fill "
                          "(BFD writes the absolute 0 as symbol 0) and ld "
                          "crashes in write_insn when it must pad it again -- "
                          "eMIPS stops with that error.  gold's options not "
                          "modelled: the GOT / TLS / GP-setup and "
                          "save / restore transformations (no PIC code), "
                          "--finalize-relocs, sorting by reference.",
    "DSP Module on MIPS64":"MD00375 rev. 3.02: MIPS32's DSP instructions on "
                            "64-bit registers and LDX; the 64-bit SIMD "
                            "instructions (octal bytes, quad halfwords, DSHILO, "
                            "BPOSGE64 ...) were deprecated in rev. 2.40 and are "
                            "not modelled (Reserved Instruction; binutils and "
                            "QEMU still decode them).  Where the manual "
                            "contradicts itself: DSPControl is Table 3.4's "
                            "(ccond 27:24, pos 5:0; some pages' pseudocode and "
                            "QEMU's MIPS64 still use the deprecated "
                            "instructions' ccond 31:24 and pos 6:0, so MTHLIP "
                            "from pos 32 gives 0 here), EXTP sign-extends its "
                            "result (rev. 2.40) while EXTPV / EXTPDP / EXTPDPV "
                            "keep their pages' zero extension, SHILO's page "
                            "writes 192 bits — each half sign-extended, as "
                            "MADD's.  MAQ_SA saturates against the 64-bit LO "
                            "(sat32AccumulateQ31: HI 63..0 || LO 63..0 + a), as "
                            "QEMU.  QEMU departs where the pages are clear: its "
                            "MFHI / MFLO / MTHI / MTLO of ac1-ac3 keep only the "
                            "low word (MD00375: the whole register), SUBUH.QB "
                            "zero-extends, SHILO by 0 leaves the halves alone.  "
                            "A NotWordValue operand is UNPREDICTABLE: the low "
                            "word is used.",
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
    "R6 divide by zero": "DIV/MOD/DIVU/MODU (and MIPS64's DDIV/DMOD/DDIVU/"
                         "DMODU) by zero are UNPREDICTABLE; the result is 0 "
                         "here (QEMU: 0 for MOD/MODU, the dividend for DIV/"
                         "DIVU — it divides by 1).",
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
    "JALX": "without MIPS16e (Machine() alone, a Release 6 core, or "
            "Options.compressed none: Config1.CA = 0) Reserved Instruction, and a "
            "JR / JALR to an odd address faults on the fetch (AdEL); with it (the "
            "Session's default from Release 1 to 5) see MIPS16e.",
    "MIPS16e": "Vol. IV-a rev. 2.63 as the 24K's (MD00343 1.2.1.11: a recoder "
               "turns MIPS16e into MIPS32): Machine(compressed=\"mips16e\"), "
               "Options.compressed \"auto\" (the default: from Release 1 to 5, "
               "as the 24K always has it; none in Release 6), \"none\" (a MIPS32 "
               "core without it: a stray odd JR target is an address error), "
               "\"mips16e\"; a program with MIPS16e code gets it anyway. Instructions with a "
               "MIPS32 twin run as it (emips.mips16.expand: T = GPR 24, LI = ORI, "
               "CMP = XOR, ZEB = ANDI ...), the others by M16.* executors. "
               "Reserved: the MIPS64-only encodings on a MIPS32 core (MIPS64: "
               "see MIPS16e64) and the reserved ones, ASMACRO (no "
               "implementation-defined macros), EXTEND before a non-extensible "
               "instruction, an extended instruction whose fixed-zero fields are "
               "not 0 (MIPS16e2 reuses them), SAVE / RESTORE aregs 1111 "
               "(UNPREDICTABLE). An extended, branch or jump instruction in a jump "
               "delay slot (UNPREDICTABLE) runs whole, as in MIPS32 code. SAVE / "
               "RESTORE write sp last (a fault restarts them). The pipelines: the "
               "5-stage one fetches one instruction (2 or 4 bytes) per IF, its "
               "branches resolve in ID like MIPS32 ones (a taken compact branch "
               "squashes the instruction fetched after it); the 24K adds IR / IK "
               "(see \"24K pipeline\"); SAVE / RESTORE hold EX (the 24K: EX "
               "after AG) one cycle per word they move (MD00343 1.2.1.11 calls "
               "them macro instructions and gives no timing). The "
               "assembler: GNU as syntax (binutils mips16-opc.c), the short form "
               "when it fits else EXTEND, branches relaxed; .set reorder makes JR "
               "/ JALR compact (JRC / JALRC) and puts a NOP in JAL / JALX's slot "
               "(GNU as moves an earlier instruction there when it can); a "
               "2-operand DIV / DIVU is the machine instruction (GNU as: its "
               "divide macro; the disassembler prints div $zero, rx, ry); the "
               "address of a MIPS16e label taken as data (.word, la, li) has bit 0 "
               "set, as the linker does for a MIPS16 symbol.",
    "microMIPS": "MD00582 rev. 5.04 (Release 3 / 5) and rev. 6.05 (Release 6): "
                 "Machine(compressed=\"micromips\") = microMIPS32 and MIPS32 "
                 "(Config3.ISA = 2: MIPS32 out of reset, ISAOnExc R/W, reset 0), "
                 "\"micromips-only\" (Config3.ISA = 1, ISAOnExc 1 read-only, the core "
                 "starts in microMIPS; a jump to an even address faults on the fetch, "
                 "JALX is Reserved Instruction); Release 2+ (MD00582 describes "
                 "Release 3, but the M14K / M14Kc / microAptiv are Release 2 cores "
                 "with microMIPS: MD00668 Table 5.26, AR = Release 2; their microMIPS "
                 "has the Release 2 instructions: EVA twins are Reserved); "
                 "Options.compressed the same, and a program with microMIPS code gets "
                 "it (not with a core preset). Recoded "
                 "instructions run as their MIPS32 twin (emips.micromips.expand; "
                 "Release 6's CACHE keeps a 12-bit offset: its twin is the Release 5 "
                 "row), branches / jumps / LWP / SWP / LWM / SWM / LWXS / MOVEP / "
                 "ADDIUPC / LWPC / JRADDIUSP by MM.* executors. The handlers' ISA mode: "
                 "Config3.ISAOnExc for the vectors at EBase offsets, Config3.ISA bit 0 "
                 "(the reset ISA) for the BEV ones, NMI and reset (Vol. III Table 9.67 "
                 "says ISAOnExc applies to the EBase vectors; QEMU uses it for all); "
                 "debug exceptions: ECR.ISAOnDebug in the probe's memory, else the reset "
                 "ISA (MD00047 2.3.3). A delay slot of the wrong size for JAL / JALS ... "
                 "(UNPREDICTABLE) runs whole; the link is still past the size the "
                 "instruction names. UNPREDICTABLE, Reserved Instruction as QEMU: LWP "
                 "/ SWP / LWM / SWM in a delay slot, LWP with rd = base or rd = 31. LWM "
                 "loads every word before writing a register; SWM / SWP store in order "
                 "(a fault after the first word leaves those written: the instruction "
                 "restarts). Unicorn's QEMU departs (pinned to the manual in "
                 "test_micromips.py): SWM stores 16 bits of each register, LWM / SWM "
                 "check no alignment, microMIPS R6 ADDIUPC / LWPC / AUIPC / ALUIPC are "
                 "Reserved Instruction, BOVC $zero, $zero writes a link. Release 6: no "
                 "delay slots (no forbidden slots either), compact branches only; "
                 "B*ZALC write the link before testing rt (6.05: rt = 31 compares the "
                 "new link). The assembler: GNU as's choices (binutils "
                 "micromips-opc.c order: the 16-bit form when the operands fit; "
                 "macros from 32-bit forms; an operand with a label: the 32-bit form; "
                 "noreorder: a 32-bit delay slot after JAL / JALR16 / BGEZAL / JALX, a "
                 "16-bit one after JALS / JALRS / BGEZALS; reorder: B / BEQZ / BNEZ / "
                 "JR become BC / BEQZC / BNEZC / JRC, the other branches get a NOP of "
                 "their slot's size, nothing is moved into a slot); Release 6 (GNU as "
                 "has none): the 16-bit compact form when the target / registers fit "
                 "(relaxed like Release 5's), MIPS32 syntax for recoded instructions "
                 "(lsa rd, rs, rt, sa where 6.05 writes LSA rt, rs, rd, sa). "
                 "Release 6 compact branch offsets count halfwords (6.05's Operation: "
                 "offset || 0, as QEMU; its Description says shifted left 2 bits, as "
                 "LLVM 18 assembles them). The 5-stage pipeline runs microMIPS (the "
                 "M14K's pipeline has five stages): one instruction (2 or 4 bytes) "
                 "per IF, branches in ID, a delay slot of either size, LWM / SWM / "
                 "LWP / SWP hold EX one cycle per word; the 24K has none (MD00343 "
                 "1.2.1.11): Pipeline24K refuses a microMIPS core, Options / the CLI "
                 "refuse the pair, a microMIPS program in 24K mode runs on the "
                 "5-stage model (Session.program_notes says so). C: cc --micromips "
                 "for Release 2 / 3 / 5 (crt0 MIPS32); Release 6 is refused (GNU as has "
                 "no microMIPS R6, clang 18's la / li macros emit the MIPS32 LUI "
                 "encoding there and its compact branches use the other offset "
                 "scale).",
    "MSA": "emips.msa (MD00866 / MD00868 rev. 1.12; PLAN M24, the working plan "
           "m24-MSA-SIMD.md). A scalar FPU write leaves bits 127..64 of the vector "
           "register as they were (UNPREDICTABLE: QEMU does the same), so does a "
           "change of Status.FR. MSAIR = 0x00000300 (WRP 0, ProcessorID 3: QEMU's "
           "I6400 value; the core manuals say only 'Preset'); MSACSR.Impl (22:21) "
           "reads 0. LD / ST are done in hardware at any address (one or two "
           "translations, both before a byte is stored), unless a core preset "
           "traps an element-misaligned one (Machine.msa_ade: the P5600 / P6600 "
           "for uncached memory, the P6600 across a 16 KB page; MAAR regions are "
           "not modelled). The Unimplemented Operation cause (E) is never raised: "
           "subnormals are handled in hardware. FRCP / FRSQRT are 1 / x and 1 / "
           "sqrt(x) rounded at each step (the manual allows 1 / 2 ULP) and, as "
           "the pages allow, signal Inexact alone unless the operand is infinite, "
           "the result a NaN, or Invalid / Divide by Zero arise (QEMU's rule). "
           "DIV / MOD by zero (UNPREDICTABLE) give QEMU's results: DIV_S -1 / 1 by "
           "the dividend's sign, DIV_U all ones, MOD the dividend; DIV_S of the "
           "most negative by -1 the most negative, MOD_S 0. A subnormal "
           "floating-point result signals Underflow (Cause U when enabled, else "
           "only when inexact: Figure 3.26), not for FMIN / FMAX / FMIN_A / "
           "FMAX_A. Where QEMU 2.1.4 departs from the manual eMIPS follows the "
           "manual: FLOG2 is IEEE logB (negatives, -inf, ±0 with Divide by Zero), "
           "16-bit values are never flushed by FS, SUBS_S.D saturates only real "
           "overflows, SAT_S.D / SAT_U.D with m = 63 keep the value, tininess is "
           "detected after rounding (Table 3.3; QEMU: before), FMADD / FMSUB of "
           "inf x 0 give the default NaN even with a NaN wd (QEMU: wd's NaN), and "
           "their Underflow follows the manual (QEMU's host-FPU muladd raises it "
           "for normal results) "
           "(tests/test_msa_unicorn.py DEPARTURES). The 16-byte accesses meet the "
           "24K's EJTAG 3.20 (data breakpoints over the 16 bytes, value compares "
           "of 64 bits): the P5600's 128-bit DBCS / DBVS and its ganged even / odd "
           "breakpoints (MD01025 14.8.5.3) are not modelled. Vector register "
           "partitioning (MSAIR.WRP, Machine(msa_wrp=N): no real core has it) is "
           "one thread context's: N physical registers, MSAMap takes the lowest "
           "free one, an unmapped physical register keeps its value; MSARequest "
           "holds the registers that were missing or unsaved. MSA with the MT "
           "Module is refused (no MT core has MSA; per-TC MSACSR / partitions are "
           "not modelled); each Release 6 VP has its own vector registers. "
           "Pipelines: the 5-stage model holds EX for the P6600's MSA latencies "
           "(MD01138 Table 15.3: pipeline.LATENCY MSA ...; the P5600's manual "
           "gives none); its 'variable' divides use the FPU's divide / square "
           "root latencies and, for DIV_S ... MOD_U, the 32-bit MDU divide's; "
           "FMADD / FMSUB hold EX for the add operand's 4 (8 to a multiply "
           "operand is not modelled); LD.df is ready after MEM like a load "
           "(Table 15.3's 10 assumes the P6600's out-of-order pipeline). Every "
           "floating-point MSA instruction but FCLASS reads and writes MSACSR "
           "(a forwarded resource). The 24K model has no MSA. GDB: no GDB has an "
           "MSA feature; the stub's own org.gnu.gdb.mips.msa sends w0-w31 as ST.D "
           "stores them (Linux's NT_MIPS_MSA layout), so its union has the "
           "doubleword views in both byte orders and the word / halfword / byte / "
           "single views on little-endian only (on big-endian `monitor msa $wN w` "
           "shows them). ELF: o32 MSA code for an FR = 0 FP ABI is refused; legacy "
           "NaN code on a 2008 FPU only gets a note.",
    "DSP Module": "emips.dsp (MD00374 rev. 3.01): Revision 1 / 2 and, on Release "
                  "6, Revision 3. Where the manual leaves a result UNPREDICTABLE "
                  "eMIPS does what QEMU does: a .PH compare leaves ccond 27..26; EXTP / "
                  "EXTPV / EXTPDP / EXTPDPV that fail (EFI = 1) write 0 to rt; INSV "
                  "with pos / size outside 0 <= pos < 32, 0 < size <= 32, pos + size "
                  "<= 32 leaves rt; MTHLIP with pos > 32 leaves pos; BALIGN with bp 0 "
                  "/ 2 leaves rt; the GPR-targeting multiplies (MUL.PH, MULQ_*, "
                  "MULEQ_S, MULEU_S) leave HI / LO (ac0) as base MUL does. MAQ_SA.W."
                  "PHL / PHR's page swaps its two limits (bit 32 = 0 gives "
                  "0x80000000): eMIPS saturates a positive overflow to 0x7FFFFFFF, as "
                  "DPAQX_SA's page and QEMU. RDDSP / WRDSP decode the manual's 10-bit "
                  "mask (bits 6-9 ignored; binutils accepts only 6 bits). The one-"
                  "operand idiom rddsp rd / wrdsp rs is GNU as's mask 0x3FF (the "
                  "manual says 31: the EFI bit is included). The accumulator forms of "
                  "MULT ... MTLO with ac = 0 are the base instructions before Release "
                  "6 (no DSP State Disabled); on Release 6 the accumulators belong to "
                  "the DSP Module and every form needs Status.MX. LWX / LHX / LBUX "
                  "stay aligned on Release 6 (their pages: Address Error). microMIPS: "
                  "MD00764's encodings are pre-Release 6 (Releases 2-5); on Release 6 "
                  "microMIPS has none. The 24K model times the DSP Module as the 1004K "
                  "(MD00622 Table 2.5: the 24KE's manual is not available); an RDDSP "
                  "after a multiply-accumulate waits 4 cycles (2.4.2 says only that it "
                  "waits for the MDU). The 5-stage model has no Status.MX hazard (it "
                  "has no CP0 hazards).",
    "MT Module": "emips.mt (MD00378 rev. 1.12, the 34K MD00534): 1-3 VPEs (the 34K's "
                 "2, the I7200's 3), 1-9 TCs, "
                 "Releases 2-5; on a CPS per core (see CPS). Dispatch: one instruction "
                 "of one runnable "
                 "TC a cycle, chosen by the 34K's policy managers (MD00534 13.2: basic "
                 "round-robin — the schedule registers read -1 —, WRR, WRR2); inside a "
                 "group the TCs take turns after the group's last issuer; a rotation "
                 "begins when the previous one's hold ends, its hold = the runnable TCs "
                 "of the winning group then (at least 1, also when nothing runs); "
                 "TCScheFBack counts completed instructions (not a blocked one, not one "
                 "that took an exception). WRR2's inputs: PM_wbb_avail = 4 - the open "
                 "write buffer entries, PM_fsb_avail = 8 - the fills in flight with the "
                 "memory timing, PM_ldq_avail = 7 in instruction mode, PM_sys_avail is "
                 "set from the debugger (7 at reset); VPEScheFBack shows SYSS 0, WBBS 4, "
                 "FSBS 8, LDQS 9. Pipeline mode: the 34K pipeline (see 34K pipeline). "
                 "A WAIT ends when an interrupt signal is active on the VPE, taken or "
                 "not (the 34K's Config7.WII = 1, read-only). A blocked instruction (a YIELD waiting for its "
                 "qualifiers, a gating storage access) has had no effect and issues "
                 "again, as itself (in its delay slot), when it can complete. SI_Vpe0MaxTC: VPE 1 gets the last TC "
                 "at reset, its XTC (UNDEFINED) is that TC. Configuration (MVPControl."
                 "VPC): TCBind.CurVPE rebinds a TC; VPEConf1 / Config1 (the FPU, the "
                 "TLB sizes) follow the configuration: VPEConf1.NCP1 moves the FPU "
                 "context with its registers (it starts on VPE 0; with NCP1 0 on both "
                 "it is kept for the next binding), MVPControl.STLB joins the JTLBs "
                 "into one array of at most 64 entries (VPE 0's first) and clearing it "
                 "gives each VPE its entries back as they are then; a shared TLBWR "
                 "that would hit the other VPE's Index entry (P = 0) writes the entry "
                 "Random takes next instead of stalling; the micro-TLBs stay per VPE. "
                 "MVPControl.CPA + VPEOpt.IWX / DWX: the refilling VPE's excluded ways "
                 "are not allocated (all excluded: no exclusion; a locked line still "
                 "counts as in 8.5.4). The YIELD qualifier inputs are set from the "
                 "debugger. The "
                 "TCStatus layout is the 34K's (no TFR: Status.FR stays per VPE). A "
                 "TC halted with a delay slot next shows TCRestart = the branch and "
                 "TDS = 1, and when released without a TCRestart write goes on at the "
                 "slot (the branch's effect is not redone). A reserved negative YIELD "
                 "rs is treated as -1. A TCStatus write by the TC itself keeps "
                 "the DT it writes. An MTTR to an FPU control register does not take "
                 "the FPE it may enable (6.29: the target TC services it). The "
                 "performance counters and the EJTAG TAP stay per VPE (the 34K: per "
                 "processor). The MARS console interrupts VPE 0. microMIPS (MD00768): "
                 "the eight instructions as the manual's pages; its FORK diagram gives "
                 "bits 6:0 a 6-bit value 100111 — eMIPS takes bit 6 = 0 (no binutils, "
                 "LLVM or QEMU encodes microMIPS MT to compare with). EJTAG with the "
                 "MT Module (MD00047 1.11 / 5.3 / 7.2, the 34K's MD00534 ch. 11): IBCn "
                 "/ DBCn TC (31:24) and TCuse (22) are R/W and match the issuing TC's "
                 "number; PCsample has the TC field above the ASID (48:41, MIPS64 "
                 "80:73: 49 / 81 bits — always, the 34K's DCR has no PCnoTCID); "
                 "ECR.VPED = the VPE cannot run (VPEConf0.VPA = 0, or EVP = 0 and the "
                 "one TC that runs is another VPE's): Table 11.25's reset value 1 is "
                 "not followed for VPE 0, which runs at reset. MIPS64 (MD00378 with "
                 "GPRLEN = 64 — its MFTR / MTTR pseudocode knows Config.AT —; no MIPS64 "
                 "core had the module, so no implementation decides): a TC's GPRs, "
                 "HI / LO and accumulator halves are 64 bits; TCRestart and TCContext "
                 "hold 64 bits (6.14 / 6.16: 'the same width as the processor GPRs'; "
                 "DMFC0 / DMTC0 whole), the other MT registers 32 (DMFC0 sign-extends); "
                 "MFTR sign-extends a narrower source (a 32-bit CP0 register, "
                 "DSPControl, an FPCR), reads a CP0 register as DMFC0 (EntryLo's RI / "
                 "XI at 63 / 62) and an FPR whole (AT = 1 / 2: the widths match); MTTR "
                 "writes the GPR whole (not sign-extended), a CP0 register as DMTC0; "
                 "h = 1 (UNPREDICTABLE when the widths match): an FPR's high word, a "
                 "CP0 register 0 / no effect, as on MIPS32; YIELD and FORK take whole "
                 "registers; Status.PX / KX / SX / UX stay per VPE (MD00378 3.1: only "
                 "CU3..0, MX, KSU per TC); the ITC cells are 64 bits (see ITC). The "
                 "MT Module on MIPS64 runs in instruction mode (pipeline mode asked "
                 "for: a note) — eMIPS's multithreaded pipeline is the 34K's, a "
                 "MIPS32 core.",
    "ITC": "emips.itc (MD00534 ch. 12, MD00378 9.1 / App. A): the reference ITU of "
           "one MT processor, 4 FIFO + 8 semaphore cells by default. The block decodes "
           "physical addresses ahead of the caches and the bus, whatever the CCA "
           "(MD00427 3.3.1: 'all accesses to this physical address range will go to "
           "ITC'). The CACHE pseudo-tag accesses (ErrCtl.ITC, Index Load / Store Tag D) "
           "act on address bit 3 before any translation. MD00534 Table 12.6 says "
           "AddrMask 0x3f = 128 KB: eMIPS follows the field's 7 bits (0x7f = 128 KB, "
           "MD00427). FIFOPtr = the entries (MD00534; MD00427 calls it the oldest's "
           "index). Reset (UNDEFINED): every cell's data and E / F / T bits 0, "
           "ITCAddressMap0 / 1 0. The cells are 32 bits (address bit 2 ignored, "
           "MD00378 A.2) — 64 on MIPS64 (A.1's 64-bit cells: LD / SD move a whole "
           "cell, a word is one half by address bit 2 in the CPU's byte order, as "
           "the I6400's ITU, MD01196 11.2.4). An access narrower than the cell "
           "(undefined) acts as a whole-cell access on its bytes: a load returns "
           "them, a store writes them in their place with the other bytes 0 (one "
           "size at one address reads back what it wrote, A.2); an address "
           "inside the block with no cell reads 0, writes are dropped (undefined); "
           "P/V views of a FIFO cell act on its oldest entry without changing E / F "
           "(undefined); the reserved views read 0, ignore writes; SC: success = the "
           "ITC's answer on every view (the E/F try view fails on a full cell), "
           "independent of the LLbit (MD00378 7.2); the P/V counter is bits 15:0, "
           "the upper bits are kept. A blocked access has had no effect and is issued "
           "again when the cell lets it complete (a TCHalt or an exception on the TC "
           "aborts it: nothing to undo). No ITC performance events (the presets keep "
           "the 24K's events).",
    "core presets": "emips.cores: a preset sets what the core's datasheet and "
                    "Software User's Manual let it be built with (the release, the "
                    "compressed ISA, the FPU, the MMU, the cache shapes, the cores, "
                    "the shadow sets, the pipeline model) and its PRId (Company 1, "
                    "the Processor ID, Revision 0 as QEMU's models: the revision is "
                    "the silicon's; FIR 15:0 repeats it, MD00343 Table 3.6). The "
                    "documents name no default build: the preset's own (e.g. the 24Kf "
                    "with 32 KB caches and 32 TLB entries, 4 shadow sets) is an eMIPS "
                    "choice inside the options. Every preset keeps the 24K's "
                    "implementation registers (Config7, the EJTAG 3.20 DCR with 4 "
                    "instruction / 2 data breakpoints, the micro-TLBs of 4 / 8 "
                    "entries, the 24K's cache line replacement, CCAs and performance "
                    "events); what a core has that eMIPS does not model is listed in "
                    "Core.missing (the MCU ASE, MIPS16e2, MSA, the I7200's MPU, "
                    "VZ, XPA, the P5600's out-of-order pipeline: its preset offers the "
                    "5-stage or the 24K model), and eMIPS builds a CPS only with 2-6 "
                    "cores (a 1-core 1004K / interAptiv / P5600 has no CM here). The "
                    "34K / 1004K / interAptiv use the 24K model, with the MT Module "
                    "the 34K pipeline (its thread-select stage IT; bypassed with one "
                    "thread context, MD00584). Not offered: the "
                    "M5100 / M5150 (VZ always present, a 2008 FPU), the M6250 (no PRId "
                    "in its datasheet), the 74K / proAptiv (out of order). The MIPS64 "
                    "presets (5Kc / 5Kf, 20Kc, I6400) keep eMIPS's SEGBITS 40 / PABITS "
                    "32 (the 5K has 42 / 36, the I6400 48 / 48) and the 5-stage model; "
                    "the 20Kc's build is QEMU's model's (its manual was not found), "
                    "the I6400's FTLB the 64 x 4 of its Config4 (MD01196 1.3 says 512 "
                    "dual entries), its PRIds QEMU's. A core "
                    "without an FPU (Config1.FP = 0): Status.CU1 / FR read 0 and ignore "
                    "writes (MD00343 Table 6.18 / 6.12), every FPU instruction is "
                    "Coprocessor Unusable 1, the boot firmware leaves CU1 0, GDB sees "
                    "the FPU registers as unavailable (its MIPS description requires "
                    "the feature), C is built -msoft-float (the soft-float routines "
                    "need a soft-float libgcc).  The I7200 (M23 phase 5c): PRId "
                    "0x00010000, QEMU's model's (neither MD01227 nor MD01232 gives "
                    "the Processor ID, Linux's cpu.h has no I7200); 1-3 VPEs, so "
                    "an MT processor holds up to 3 VPEs (the TCs above SI_Vpe0MaxTC "
                    "one per further VPE at reset, the rest to the last).",
    "EJTAG": "Debug Mode as the 24K's EJTAG 3.20 (MD00343 ch. 11, MD00047 "
             "ch. 2 / 3): Debug.EJTAGver = 3, NoDCR = 0 (dseg present), "
             "CountDM R/W (reset 1), IBusEP / MCheckP read 0 (precise on the "
             "24K), no DebugVectorAddr / Debug2 (EJTAG 4 / 5). The TAP "
             "(emips.tap): IDCODE = 1 (the core's Version / PartNumber / "
             "ManufID inputs tied to 0), IMPCODE MIPS16 = 1 with MIPS16e "
             "(MD00047 Table 4.4), the TCB's instructions 0x10-0x13 (MIPS Trace); PrRst / "
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
                   "FIFOs on, OUT2 off) and routes PIRQA-D to IRQ 10, 10, 11, 11. "
                   "A 64-bit CPU (M22 phase 6g): the CoreLV carried MIPS64 5K LVs "
                   "on its 64-bit SysAD bus (MD00007; YAMON's Table 1.1), so the "
                   "5Kc / 5Kf presets are real; eMIPS also puts the generic MIPS64 "
                   "cores (Releases 1-6) and the MIPS64 presets other than the 20Kc "
                   "(the Core20K card's Bonito64 controller: refused) on it, as QEMU's "
                   "64-bit Malta does — no such cards existed — and multi-core "
                   "64-bit Maltas paired as the MIPS32 ones; REVISION keeps CORID 1 "
                   "(CoreLV). A doubleword access to the board covers two 32-bit "
                   "words, each as a word access sees it (the board registers on "
                   "64-bit boundaries: the second word of a pair reads 0 and is "
                   "dropped; GT-64120 registers: two registers); a doubleword to PCI "
                   "I/O is two 32-bit PCI cycles, with the byte lanes reversed all 8 "
                   "bytes swap (GT-64120 Table 7). Physical addresses stay 32-bit; "
                   "the kseg0 / kseg1 segments are sign-extended (the -msym32 "
                   "assembler); the user window is the same through the 32-bit and "
                   "the XTLB refill handlers.",
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
                    "compare them); --boot reset runs the code. On a 64-bit CPU "
                    "(phase 6g) the same MIPS32 code runs, assembled for MIPS64 "
                    "with the o32 temporaries as register numbers: the exception "
                    "frame keeps doublewords (sd / ld, HI / LO too; at 0x8000c600, "
                    "above the strings), its RAM addresses are sign-extended, an "
                    "unhandled exception prints the 64-bit EPC / BadVAddr, the "
                    "application starts with Status.FR and, with 64-bit addresses "
                    "(Config.AT = 2), KX / SX / UX as the kernel boot's boot64(), "
                    "and the XTLB refill handler goes to 0x80000080 too; YAMON's "
                    "interface stays 32-bit (the descriptor's words, the table's "
                    "kseg0 pointers, a registered ISR / ESR: a compatibility "
                    "address) — its functions use only o32's temporaries, all "
                    "caller-saved in n64 too. On a nanoMIPS CPU (M23 phase 4d; "
                    "QEMU's Malta has a nanoMIPS boot loader for its I7200, "
                    "write_bootloader_nanomips, no YAMON) the same code is "
                    "assembled as nanoMIPS for the nanoMIPS subset (GNU as's "
                    "port takes the MIPS mnemonics as their compact forms; the "
                    "registers as $rN, MIPS's lui as li, .balign to the fixed "
                    "addresses); YAMON's table stays at 0x1fc00500, its functions "
                    "follow the p32 convention (no YAMON for nanoMIPS exists: "
                    "the arguments in $a0-$a3 as in o32, the result in $a0 — and "
                    "in $2); the vector stub and the TLB refill handler are "
                    "nanoMIPS code; a returning application's value is its $a0; "
                    "$gp = _gp, the start of the small data (p32).",
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
           "the P5600: 18:16); MT cores (the 1004K / interAptiv, emips.mt "
           "Cluster): every core has the same VPEs / TCs, VP = core * VPEs + VPE "
           "(MD00904 Table 9.1) = EBase.CPUNum, the CPC powers / resets a core "
           "with all its VPEs (a fresh processor: TC 0 at the reset vector), one "
           "ITC for the CPS (the 1004K's multi-core ITC), the GIC's MAP_TO_YQ "
           "drives the YIELD qualifier inputs of the cores MAPn_VP names (ORed "
           "with the debugger's), the CM sees a core (its caches) whichever VPE "
           "accesses, a cycle issues one TC on every running core; no IOCU; no "
           "CM power domain "
           "(the GIC: see GIC). The firmware "
           "(kernel / user boot) gives every core write access, enables the "
           "CPC at 0x1BDE0000, puts every core in the coherence domain, maps "
           "cacheable memory coherent (CCA 5, as Boot-MIPS does for coherent "
           "cores) and starts them all at the entry with $sp 64 KB apart "
           "(eMIPS's MARS-like boot); the reset boot runs core 0 only. Each "
           "step runs every running core one instruction / cycle, core 0 "
           "first; L1 caches per core, the L2 / L3 shared behind the CM. 2-6 "
           "cores (the P5600 / P6600: up to six, MD01024 / MD01137), at most 8 VPs "
           "(cores x VPEs: the boot firmware routes Boot-MIPS's IPIs, sources 24 + "
           "n / 32 + n, to 8 VPs; an MT CPS with 'auto' gets one VPE a core above "
           "4). MIPS64 cores (the P6600, MD01138 ch. 6-8: the P5600's CM2): "
           "GCR_Cx_RESET_BASE is a 32-bit virtual address, sign-extended; "
           "CMGCRBase is 64 bits (PA 39:15 in 35:11); the _UPPER registers of the "
           "40-bit physical addresses (Table 6.5, on MIPS32 too: GCR_BASE_UPPER 4:0, "
           "the others 7:0) are R/W (reset 0), and a block or region they move "
           "above 4 GB is out of reach of eMIPS's 32-bit physical addresses; "
           "GCR_CUSTOM_BASE keeps 31:16, GGU_EN reads 0 (GCR_CUSTOM_STATUS.GGU_EX "
           "= 0: no custom block); a 64-bit access to the 32-bit registers moves "
           "two of them in the byte order (the manual: 32-bit accesses only).",
    "CM3": "emips.cps with cm3 (M22 phase 6f; the I6400: MD01196 ch. 5-7, Linux "
           "mips-cm.h / mips-cpc.h / smp-cps.c / cps-vec.S for the CM3 fields, QEMU's "
           "Boston CPS and the P8700's CM3.5 RESET_BASE / VP_RUN pages where the guide "
           "points at its unavailable register companion). The registers are 64 bits: "
           "a doubleword access reaches a whole register, a word access its low "
           "(offset + 0) or high (offset + 4) half in either byte order (what Linux's "
           "32- and 64-bit accessors rely on; the high halves are the CM2's _UPPER "
           "registers, 16 bits: PA 47:32, out of reach above 4 GB). GCR_REV 8.0; "
           "GCR_ACCESS at 0x120 (cores 5:0, IOCUs 17:16, reset 0x3003F; it gates the "
           "CPC too: no CPC_ACCESS; 0x020 reads 0); GCR_CONFIG bit 31 = an ITU, 4 "
           "regions; the CM2's GCR_REGn are gone (0x90-0xCF read 0: the CM3's "
           "address regions are the MMIO ones, 0x6F8-0x738, stored: one memory port, "
           "no request limit); GCR_BASE keeps CM_DEFAULT_TARGET (1:0) only; "
           "GCR_L2_CONFIG (0x130) describes the L2 as Config2 does (Linux's fields; "
           "BYPASS without one), read-only; GCR_SYS_CONFIG2.MAXVPW = 4; "
           "GCR_L2_PFT_CONTROL / _B stored (NPFT 0: no prefetcher); GCR_SEM (0x640) a "
           "lock write is inhibited while SEM_LOCK is set; GCR_BEV_BASE (0x680) "
           "stored, not used; GCR_Cx_COHERENCE is COHEN (bit 0): every core with it "
           "set is in one domain (the directory); GCR_CL_REDIRECT (0x018) and "
           "GCR_CL_RESET_BASE (0x020) are per VP: REDIRECT's CORE 13:8 / VP 2:0 / "
           "BLOCK 25:24 choose the core-other window (BLOCK 1 = the global block, 2 / "
           "3 and a core or VP that does not exist read 0) for the GCR's and the "
           "CPC's core-other blocks, RESET_BASE's RESET_BASE_MODE (bit 1, R/W, reset "
           "0): 1 = BEVExcBase anywhere in the 4 GB (MD01196 5.5.2), 0 = as a CM2's "
           "UseExceptionBase (31:30 = 10); RESET_EXT_BASE per core as a CM2's. Errors "
           "(Table 5.4): a cached access to a register is MP_REQUEST_DECODE_ERR (2) "
           "with the CM2's request bits as ERR_INFO, ERR_TYPE in 63:58; the response "
           "is an error whatever GCR_ERROR_MASK (5.10), the mask only interrupts; "
           "writing the recorded ERR_TYPE back clears CAUSE and MULT; the directory "
           "reports no intervention error (no such type). The CPC: 64-bit registers "
           "as the GCR's; another domain through the accessing VP's GCR_CL_REDIRECT "
           "(a core, the CM 0x20, the DBU 0x23, an IOCU 0x10, memory 0x1A — Table "
           "5.1); the CM's domain reads U5 while a core is up, its commands are "
           "ignored; the DBU's takes the commands (its state only: no DBU); clock "
           "ratios (CC_CTL 0x18 per domain, the global CC_CTL 0x28 whose change "
           "sequence ends at once, the prescaler 0x48 whose ratio reads back in "
           "23:16) and RAM_SLEEP (0x50) are stored: eMIPS has one clock; PWRUP_CTL "
           "writes are ignored (the CM runs while a core does, 6.2.9); ROCC_CTL "
           "(Figure 6.4): the occurred bits write-1-to-clear (eMIPS's choice), at "
           "power-on the CPC, the CM and core 0; RESET_CAUSE 2 after a watchdog "
           "reset. 2-6 cores of 1-4 VPs, at most 24 VPs (MAPn_VP's 24 bits).",
    "Release 6 VPs": "emips.vp (M22 phase 6f; Vol. II / III 6.06 / 6.03, MD01196 ch. "
                     "12). Each VP is a whole Machine (its GPRs, CP0, FPU, VTLB, "
                     "exceptions, debug); a core's VPs share its L1 caches (and the "
                     "shared L2 behind the CM), its FTLB (a write on one VP is the "
                     "others' too) and its ITU / SAAR. VPNum = core x 4 + VPId "
                     "(Table 9.13's non-contiguous numbering, Max-VP 4: the I6400's GIC) "
                     "= EBase.CPUNum and the GIC's VP number; GlobalNumber = CoreNum "
                     "11:8, VPId 7:0. Each step runs every running VP one instruction "
                     "(or a pipeline cycle: a pipeline per VP — the I6400 issues two a "
                     "cycle from up to four VPs, not modelled). The CPC: after a core "
                     "reset VP 0 runs (si_core_vc_run_init; 'the value in this "
                     "register is reset whenever the associated core is reset', 6.2.11); "
                     "VP_RUN of a stopped VP starts it from its reset base as a fresh "
                     "VP (the P8700's 'from the reset vector', QEMU's cpu_reset), "
                     "VP_STOP suspends it; on a core that does not execute both do "
                     "nothing (its reset decides). DVP: the core's other VPs fetch "
                     "nothing and take no interrupt until this VP's EVP (their clock "
                     "runs); with VPControl.DIS = 1 it still returns VPControl (the "
                     "pseudocode; the programming notes' nesting needs it); a VP whose "
                     "DVP holds the others and starts again releases them. The kernel / "
                     "user boot starts every VP at the entry as every core (stacks 64 "
                     "KB apart); the reset boot core 0's VP 0. microMIPS DVP / EVP: the "
                     "6.05 manual's rt in 25:21 (LLVM 18 puts it in 20:16).",
    "ITU": "emips.itu (M22 phase 6f; MD01196 ch. 11, QEMU's mips_itu.c): one ITU a "
           "core, 64-bit cells (FIFOs of 2 / 4 / 8 entries, then single-entry "
           "cells: 4 FIFOs of 4 + 8 by default), at SAAR's base (bits 43:13 = PA "
           "47:17; 128 KB, SIZE read-only 17; EN only with an ITU: SAARI writes "
           "other than 0 are dropped, SAAR is the core's, one for its VPs). Decoded "
           "on the physical address ahead of the caches, whatever the CCA (11.3: "
           "'always uncached'); the cell index from bit 7 + ICR0.BLK_GRAIN, the "
           "bits above it to 15 and below it to 7 select nothing; a word is one half "
           "of a cell by address bit 2 in the CPU's byte order. Errors are error "
           "responses (a data bus error, imprecise): a 1- or 2-byte access "
           "(ERROR_AXI), a cell past the configured ones, the ICR view of a cell "
           "but 0 and the E / F / P/V state errors of 11.5.3 (ERROR_EXEC); a "
           "misaligned access is an address error (11.3) on Release 6 too. A "
           "blocking access issues again until it completes (the VP stalls: no "
           "thread exception — Release 6 has no MT Module); Bypass of a FIFO "
           "reads the oldest entry, writes the newest; a Control write of E = 1 "
           "empties a FIFO (QEMU); P/V loads return the 16-bit count they "
           "decrement (zero-extended); the reserved views 6-14 act as Bypass "
           "(Table 11.7).",
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
           "pins 0 / 1 and sets up sources 24-39 as Boot-MIPS's IPIs. A CM3's GIC "
           "(the I6400, MD01196 ch. 7): 64 sources (multiples of 8), 64-bit shared "
           "registers (a doubleword reaches 64 sources, a word 32 as the CM2's), "
           "MAPn_VP a bit per VP number (core x 4 + VP, Table 7.3), GIC_VL_IDENT "
           "the VP number, GIC_VL_OTHER.VPNUM (5:0, Linux) the other VP's — the "
           "guide's GIC_REDIRECT_EN through GCR_CL_REDIRECT is a CM3.5 field "
           "(Linux), not modelled; the boot firmware's IPIs are the guide's, source "
           "32 + VPNum rising-edge to that VP (7.2.7), pin 0.",
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
    "34K pipeline": "emips.pipe34k (MD00534 ch. 2 / 13): the 24K pipeline model with "
                    "the MT Module's TCs — used in pipeline mode whenever the MT Module "
                    "is on (the 1004K / interAptiv too: their thread-select stage is the "
                    "34K's). Per TC: the fetch state, an 8-entry instruction buffer (the "
                    "6-entry option is not offered), the register scoreboard; per VPE: "
                    "the CP0 hazards and the micro-TLBs (the 34K's ITLB of 3 shared + 1 "
                    "entry per TC is the 24K's 4-entry ITLB per VPE). Before IF one TC "
                    "fetches a cycle, round-robin among those that can (runnable or in "
                    "WAIT, buffer not full, not waiting for a jump); an I-cache miss "
                    "holds that TC only. IT (several TCs): RF at IF + 3 at the earliest, "
                    "the dispatch scheduler chooses among the TCs with an instruction "
                    "ready by the policy manager. The skid buffer: an instruction whose "
                    "operand comes from a load that missed (its data later than EX + 2) "
                    "goes back with its TC's younger issued instructions, the TC "
                    "dispatches again when the data is there; the MDU / FPU 'long "
                    "stall' replays are not modelled (they hold AG as on the 24K). A TC "
                    "that is not runnable (WAIT, a blocked YIELD / ITC access, its VPE "
                    "exclusive to another TC, TCHalt) loses its issued instructions the "
                    "same way; a blocked instruction re-issues as itself. Software "
                    "stopping a TC (TCHalt, TCStatus.A = 0) sets its PC to the oldest "
                    "instruction it has not executed (a delay slot: armed, the next "
                    "fetched one after it); a restart / release refetches from its PC. "
                    "An MFTR of a running TC's TCRestart reads the last instruction it "
                    "executed. A WAIT ends when an interrupt signal is active on the VPE "
                    "even if IE / IXMT / EXL keep it from being taken (Config7.WII = 1), "
                    "at the TC's next dispatch; the interrupt goes with whichever TC's "
                    "instruction enters EX first. The return prediction stack is used "
                    "only while one TC is runnable (2.2.2), else a JR $31 waits. No "
                    "performance counter events of the 34K's (Replays, Single Threaded "
                    "mode ... are kept as Pipeline34K.replays only). The exit service "
                    "drains every TC.",
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
                    "Watch, CorExtend hazards: no such hardware. MIPS16e "
                    "(2.1.3, 2.1.4, 2.2.5): IF reads a 64-bit bundle every "
                    "other cycle, the instructions that start in it (one "
                    "crossing its end is fetched whole); the recoder (IR, the "
                    "cycle after IS, then IK) takes two 16-bit instructions a "
                    "cycle, one when it or the next is extended / JAL(X) (Table "
                    "2.1); prediction in IR as in IS (the BHT by VA 11:3 for "
                    "BEQZ / BNEZ / BTEQZ / BTNEZ, B taken, JAL / JALX taken, JR "
                    "$ra / JRC $ra by the return stack, JAL / JALX / JALR / "
                    "JALRC push), the target fetched 3 cycles after the branch's "
                    "IF; a taken branch's bundle-mates are not queued (IK). The Machine and the 5-stage model have no "
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
                "not part of the core (with the MT Module GDB's threads are the "
                "TCs, core by core: a packet that reads registers / memory binds "
                "the thread's TC to its VPE for that moment, a stop names the TC "
                "about to issue on the selected VP, vCont;s:N runs cycles until "
                "TC N issued one instruction, vCont;c:N lets only the named TCs "
                "issue): it reads and writes registers and memory "
                "through the debugger's side-effect-free views (no exception, no "
                "cache or TLB counting; an unmapped address or a bus error is an E "
                "reply) and keeps breakpoints itself (Z0 and Z1 alike, no BREAK "
                "written, so one in a delay slot works). Watchpoints (Z2-Z4) are "
                "checked before every load / store of the program "
                "(Machine.agent_watch, before the EJTAG / watch-register checks; "
                "never for views, fetches, CACHE / PREF or DMA) and stop before the "
                "access, as a precise Watch exception; the instruction runs again "
                "(a delay slot keeps its branch). GDB steps MIPS in software (a "
                "temporary breakpoint after the delay slot). A MIPS64 core: "
                "every register 64 bits (GDB's mips64-*.xml), the 32-bit ones "
                "(Status, Cause, FCSR, FIR) sign-extended and an FPR with FR = 0 "
                "its word zero-extended, as QEMU's stub (GDB shows 64-bit FPRs "
                "whole: its FR = 0 handling is disabled); a breakpoint at a "
                "sign-extended compatibility address is kept in its 32-bit form, "
                "a watchpoint in the form the core compares (Config.AT = 1: "
                "32-bit). Threads = cores in "
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
                "its own heuristic; the labels of MIPS16e / microMIPS code get "
                "STO_MIPS16 / STO_MICROMIPS and e_flags the ASE, as GNU ld writes "
                "them). In 16-bit code $pc has bit 0 set (QEMU's stub, GDB's "
                "mips-tdep.c is_compact_addr); a written $pc's bit 0 is the ISA mode "
                "(an even one at compressed code of the program keeps the ISA); a "
                "Z0 / Z1 address's bit 0 is dropped (kinds 2 / 3 / 5).",
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
#: :class:`UnimplementedError`): none since EJTAG Debug Mode (M14) but MSA's
#: (M24, until phase 2: set at the end of the module).
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
     "SBE", "SHE", "SWE", "SWLE", "SWRE", "SCE", "PREFE", "CACHE@R6",
     # the MT Module (MD00378 5.1: "privileged Cop0 instructions")
     "DMT", "EMT", "DVPE", "EVPE", "MFTR", "MTTR",
     # MIPS64 (PLAN M22)
     "DMFC0", "DMTC0",
     # Release 6 multi-threading (Vol. II 6.06: Coprocessor Unusable)
     "DVP", "EVP"}
)


def has_delay_slot(entry: Mapping[str, Any]) -> bool:
    """Branches and jumps (pre-Release 6): the next instruction is their
    delay slot whether or not they are taken (``ERET`` has none); MSA's
    BZ / BNZ keep theirs in Release 6 (MD00866 BNZ.V)."""
    return entry["family"] in ("branch", "branch_fpu", "jump", "msa_branch") or \
        entry["name"].startswith("BC2")


def is_cti(entry: Mapping[str, Any]) -> bool:
    """Control transfer instruction: any branch or jump (R6: may not be
    placed in a delay slot or a forbidden slot)."""
    return entry["family"] in ("branch", "branch_fpu", "jump", "branch_compact",
                               "jump_compact", "msa_branch") or entry["name"].startswith("BC2")


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


def _s64(value: int) -> int:
    value &= M64
    return value - (1 << 64) if value >> 63 else value


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


def _br_target(pc: int, word: int, mask: int = M32) -> int:
    """Branch target: ``PC + 4 + sign_extend(offset || 00)`` (``mask``:
    the PC's width, MIPS64 64 bits)."""
    return (pc + 4 + (_s16(imm16(word)) << 2)) & mask


def _j_target(pc: int, word: int, mask: int = M32) -> int:
    """J/JAL target: ``(PC + 4)[GPRLEN-1:28] || instr_index || 00``."""
    return ((pc + 4) & ~0x0FFFFFFF & mask) | ((word & 0x3FFFFFF) << 2)


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

#: emips.trace, imported on the first traced instruction
_trace = None

#: id(row) -> (the row, its executor) for the rows Machine._dispatch runs
#: without a check (no Coprocessor Unusable / DSPDis / EVA / privilege
#: gate, an EXEC row): found by its first dispatch
_PLAIN: dict[int, tuple] = {}

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
    ``compressed``  ``None`` (default), ``"mips16e"``: the MIPS16e ASE
                    (Config1.CA; Vol. IV-a, :mod:`emips.mips16`), or
                    ``"micromips"``: microMIPS32 and MIPS32 (Config3.ISA =
                    2, Release 3+; :mod:`emips.micromips`), or
                    ``"micromips-only"`` (Config3.ISA = 1) — JALX and JR /
                    JALR with bit 0 set switch :attr:`isa_mode`.
    ``arch``        ``"mips32"`` (default) or ``"mips64"``: a MIPS64 core
                    (:class:`emips.sim64.Machine64`, PLAN M22).
    """

    #: the architecture (MIPS64: :class:`emips.sim64.Machine64`)
    arch = "mips32"
    #: the width of the GPRs, HI / LO, the PC and virtual addresses
    amask = M32
    #: 64-bit registers (MIPS64)
    wide = False
    #: a GPR's value as a signed number
    sgn = staticmethod(_s32)
    #: the width of the virtual addresses the MMU sees (a MIPS64 core with
    #: 64-bit addressing, Config.AT = 2: 64 bits)
    vamask = M32
    #: a nanoMIPS core (emips.simnano: 16 / 32 / 48-bit instructions only)
    nano = False
    #: a RISC-V hart (emips.simrv: not a MIPS core at all; PLAN M25)
    rv = False
    #: the ABI's register names (emips.syntax.gpr_name): o32's, "n64"
    #: (MIPS64), "p32" (nanoMIPS), "riscv" (emips.riscv.GPR_NAMES)
    names: bool | str = True
    #: the bytes the last LL linked (nanoMIPS LLWP: 8; emips.simnano)
    lr_size = 4

    def __new__(cls, *args: Any, arch: str = "mips32", **kwargs: Any) -> "Machine":
        if arch not in ("mips32", "mips64", "nanomips", "rv32", "rv64"):
            raise ValueError("arch must be 'mips32', 'mips64', 'nanomips', 'rv32' or "
                             f"'rv64', not {arch!r}")
        if arch == "mips64" and cls is Machine:
            cls = _S64.Machine64
        elif arch == "nanomips" and cls is Machine:
            cls = _SN.NanoMachine
        elif arch in ("rv32", "rv64") and cls is Machine:
            from . import simrv
            cls = simrv.RVMachine
        return super().__new__(cls)

    def __init__(self, isa: ISA | None = None, memory: PMemory | None = None,
                 pc: int = 0, *, endian: str = "big",
                 exceptions: str = "cp0", release: str | None = None,
                 mmu: Any = None, caches: Caches | None = None, cpu: int = 0,
                 shadow_sets: int = 1, compressed: str | None = None,
                 fpu: bool = True, prid: int = 0, dsp: int | None = None,
                 mt: bool = False, arch: str = "mips32", msa: bool | None = None,
                 msa_wrp: int = 0, fpu2008: bool = False) -> None:
        if endian not in ("big", "little"):
            raise ValueError(f"endian must be 'big' or 'little', not {endian!r}")
        if exceptions not in ("cp0", "raise"):
            raise ValueError(f"exceptions must be 'cp0' or 'raise', not {exceptions!r}")
        if isa is not None and release is not None and isa.release != release:
            raise ValueError(f"isa is {isa.release}, release={release!r}")
        if dsp is None:
            dsp = isa.dsp if isa is not None else 0
        if isa is not None and isa.dsp != dsp:
            raise ValueError(f"isa has DSP level {isa.dsp}, dsp={dsp!r}")
        if dsp and (isa.release if isa is not None else release) == "R1":
            raise ValueError("the DSP Module needs Release 2+ (MD00374 3.1)")
        if dsp == 1 and (isa.release if isa is not None else release) == "R6":
            raise ValueError("Release 6 has only DSP Revision 3 (DSPP = DSP2P = 1): dsp=2")
        if isa is not None and mt and not isa.mt and not self.nano:
            raise ValueError("isa has no MT Module rows, mt=True")
        if mt and not self.nano and (isa.release if isa is not None else release
                                     or ISA.DEFAULT_RELEASE) in ("R1", "R6"):
            raise ValueError("the MT Module needs Release 2-5 (MD00378 1.1; Release 6 "
                             "has none)")
        if msa is None:
            msa = isa.msa if isa is not None else False
        if isa is not None and isa.msa != bool(msa):
            raise ValueError(f"isa {'has' if isa.msa else 'has no'} MSA rows, msa={msa!r}")
        if msa and (self.nano or (isa.release if isa is not None else release
                                  or ISA.DEFAULT_RELEASE) not in ("R5", "R6")):
            raise ValueError("MSA needs MIPS32 / MIPS64 Release 5+ (MD00866 1.1; "
                             "nanoMIPS has none)")
        if msa and mt:
            raise ValueError("MSA with the MT Module: no MT core has MSA (the 34K, "
                             "1004K, interAptiv); per-TC MSA contexts are not modelled")
        if msa_wrp and not msa:
            raise ValueError("vector register partitioning (msa_wrp) needs MSA")
        # (the shared table of the release: decoding is a pure function)
        self.isa = isa if isa is not None else ISA.for_release(
            release, dsp, mt and not self.nano, arch="mips32" if self.nano else self.arch,
            msa=bool(msa))
        # (a nanoMIPS core keeps MIPS32 Release 6's table: nothing decodes with it)
        if self.isa.arch != (self.arch if not self.nano else "mips32"):
            raise ValueError(f"a {self.arch} machine needs a {self.arch} ISA, "
                             f"not {self.isa.arch}")
        #: MIPS32 release of this core (ISA, Config.AR, FPU width)
        self.release = self.isa.release
        #: Count increments every ``ccres`` cycles (the implementation's
        #: CCRes, read by RDHWR 3; typically 2, :data:`emips.cp0.CCRES`)
        self.ccres = C.CCRES
        #: address translation (Config.MT): emips.mmu.FixedMappingMMU
        self.mmu = mmu if mmu is not None else default_mmu()
        #: the CDMM's devices of an MPU (emips.mpu; set by its attach)
        self.cdmm: Any = None
        self._wmask = C.write_masks(self.release, supervisor=self.mmu.supervisor)
        #: physical memory, keyed by physical address
        self.mem = memory if memory is not None else PMemory()
        self.big_endian = endian == "big"
        self.exceptions = exceptions
        self.pc = self.to_pc(pc)
        if not 1 <= shadow_sets <= 16:
            raise ValueError(f"shadow_sets must be 1..16, not {shadow_sets}")
        if compressed not in (None, "mips16e", "micromips", "micromips-only"):
            raise ValueError("compressed must be None, 'mips16e', 'micromips' or "
                             f"'micromips-only', not {compressed!r}")
        if compressed == "mips16e" and self.release == "R6":
            raise ValueError("Release 6 has no MIPS16e (Vol. II rev. 6: removed)")
        if compressed in ("micromips", "micromips-only") and self.release == "R1":
            raise ValueError("microMIPS32 needs Release 2+ (the M14K: Release 2, "
                             "MD00668 Table 5.26; MD00582: Release 3)")
        #: the compressed instruction set the core implements: "mips16e"
        #: (Config1.CA), "micromips" (Config3.ISA) or None
        self.compressed = "micromips" if compressed == "micromips-only" else compressed
        #: the MIPS32 encoding is implemented (not a microMIPS-only core)
        self.mips32 = compressed != "micromips-only"
        #: the ISA mode: 0 = MIPS32, 1 = MIPS16e / microMIPS (Vol. IV-a 3.7,
        #: MD00582 3.5)
        self.isa_mode = 0 if self.mips32 else 1
        #: the microMIPS instruction being executed (its MM.* executor)
        self._mm = None
        #: the compressed instruction's own encoding (BadInstr), and the
        #: branch's whose delay slot runs (BadInstrP)
        self._insn_raw: int | None = None
        self._slot_raw: int | None = None
        #: the size of the branch / jump whose delay slot runs (EPC = it)
        self._slot_back = 4
        #: the MIPS16e instruction being executed (its M16.* executor)
        self._m16 = None
        #: the base address of a PC-relative MIPS16e instruction (Table 3.14)
        self._m16_base = 0
        #: the table a MIPS16e instruction's twin decodes with (Release 5:
        #: SEB / SEH ... are MIPS16e instructions on a Release 1 core too;
        #: Machine64: its own MIPS64 table)
        self._isa16 = _ISA_R5
        #: the GPR sets (Release 1: only the normal GPRs); ``regs`` is the
        #: current one (SRSCtl.CSS), set 0 = the normal GPRs
        self.shadow: list[list[int]] = [
            [0] * 32 for _ in range(shadow_sets if self.release != "R1" else 1)]
        self.regs: list[int] = self.shadow[0]
        self.hi = 0
        self.lo = 0
        #: the DSP Module (emips.dsp, MD00374): 0 = none, 1 = Revision 1,
        #: 2 = Revision 2 (Revision 3 on Release 6); DSPControl and the
        #: accumulators ac1-ac3 (index 0 unused: ac0 = hi / lo)
        self.dsp = dsp
        self.dspctl = 0
        self.dsp_hi: list[int] = [0] * 4
        self.dsp_lo: list[int] = [0] * 4
        #: MSA (emips.msa, MD00866): the vector registers' bits 127..64
        #: (63..0 are ``fpr``), MSACSR, MSAIR, the partitioning registers
        #: (WRP; phase 2f)
        self.msa = bool(msa)
        self.wh: list[int] = [0] * 32
        self.msacsr = 0
        self.msair = _V.MSAIR | (_V.MSAIR_WRP if msa_wrp else 0)
        #: MSAIR.WRP's partitioning (emips.msa.Partitions; ``msa_wrp`` physical
        #: vector registers, 3.4.3-3.4.8 / Appendix A) or None
        self.msa_wrp: Any = _V.Partitions(msa_wrp) if msa_wrp else None
        #: when an element-misaligned LD / ST traps (decision 9; a core
        #: preset's): "uncached" (the P5600, MD01025 8.8.9; the P6600
        #: 3.13.4), "page16k" (one crossing a 16 KB boundary: the P6600
        #: 3.13.3); empty = done in hardware, as MD00866 allows
        self.msa_ade: frozenset[str] = frozenset()
        #: FPU: 32 64-bit FPRs (see fpr_* for the FR=0 / FR=1 views),
        #: FCSR (FCR31) and the read-only FIR (FCR0).
        #: an FPU (CP1) is implemented (Config1.FP; the 24Kf, not the 24Kc:
        #: without one Status.CU1 / FR read 0 and ignore writes, MD00343
        #: Table 6.12, so every FPU instruction is Coprocessor Unusable 1)
        self.has_fpu = fpu
        self.fpr: list[int] = [0] * 32
        self.fcsr: int = 0
        self.fir: int = FIR if self.release != "R1" else FIR_R1
        if self.release == "R6" or msa or fpu2008:  # IEEE 754-2008 FPU (an MSA
            self.fir |= FIR_HAS2008                 # core's too: MD00866 3.3.2;
            self.fcsr = FCSR_RO_2008                # the P5600's: MD01025 13.10.7)
        #: PRId (CP0 15.0): Company ID 23:16, Processor ID 15:8, Revision 7:0
        #: (0: no licensed core); FIR 15:0 repeats Processor ID / Revision
        #: (MD00343 Table 3.6)
        self.prid = prid & M32
        self.fir |= self.prid & 0xFFFF
        #: CP0 registers, keyed ``(register, selector)``.
        self.cp0: dict[tuple[int, int], dict[str, Any]] = _load_cp0_table()
        #: the 64-bit ones (MIPS64: emips.sim64); the others are 32 bits
        self._wide_cp0: frozenset[tuple[int, int]] = frozenset()
        for key, name in C.EXTRA_CELLS.items():
            self.cp0.setdefault(key, {"name": name, "value": 0})
        for key, value in C.reset_values(self.big_endian, self.release).items():
            self.cp0[key]["value"] = value
        self.cp0[C.PRID]["value"] = self.prid
        if not fpu:                             # Config1.FP = 0; CU1, FR: R0
            self.cp0[C.CONFIG1]["value"] &= ~1
            self.cp0[C.STATUS]["value"] &= ~(C.ST_CU1 | C.ST_FR)
            self._wmask[C.STATUS] &= ~(C.ST_CU1 | C.ST_FR)
        #: the MT processor this Machine is a VPE of (emips.mt.Processor; it
        #: attaches itself), the VPE number, the bound TC is interrupt exempt
        #: (TCStatus.IXMT)
        self.mt: Any = None
        self.vpe = 0
        self._ixmt = False
        #: CP0 registers another object holds (MFC0 / MTC0 go to its read /
        #: write): the MT processor's, a Release 6 core's shared ones
        self._ext_regs: frozenset = frozenset()
        self._ext: Any = None
        #: the Release 6 core this Machine is a virtual processor of
        #: (emips.vp.VPCore; it attaches itself) and its VPId, or None
        self.vpcore: Any = None
        self.vp_id = 0
        if mt:                                  # Config3.MT, ULRI (MD00378 6.25.7)
            self.cp0[C.CONFIG3]["value"] |= C.CFG3_MT | C.CFG3_ULRI
        # Config3.MSAP, Config5.MSAEn: R/W with MSA, else R0 (MD00866 3.2)
        cfg5 = self._wmask.get((16, 5), M32)
        self._wmask[(16, 5)] = cfg5 | _V.CFG5_MSAEN if msa else cfg5 & ~_V.CFG5_MSAEN
        if msa:
            self.cp0[C.CONFIG3]["value"] |= _V.CFG3_MSAP | 1 << 31      # (M: Config4)
            self.cp0[(16, 4)]["value"] |= 1 << 31                       # (M: Config5)
        if dsp:                                 # Config3.DSPP / DSP2P, Status.MX R/W
            self.cp0[C.CONFIG3]["value"] |= C.CFG3_DSPP | (C.CFG3_DSP2P if dsp > 1 else 0)
            self._wmask[C.STATUS] |= C.ST_MX    # (MD00374 3.11; reset 0: MD00622 Table 7.32)
        if compressed == "mips16e":
            self.cp0[C.CONFIG1]["value"] |= C.CFG1_CA
        elif self.compressed == "micromips":    # Config3.ISA; ISAOnExc RW with both
            if self.mips32:
                self.cp0[C.CONFIG3]["value"] |= 2 << C.CFG3_ISA_SHIFT
                self._wmask[C.CONFIG3] = self._wmask.get(C.CONFIG3, 0) | C.CFG3_ISAONEXC
            else:
                self.cp0[C.CONFIG3]["value"] |= (1 << C.CFG3_ISA_SHIFT) | C.CFG3_ISAONEXC
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
        #: the core it is in (a VPE of an MT core in a CPS: the core's
        #: number; the GCR / CPC core-local blocks)
        self.core_num = cpu
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
        #: tick(m, cycles).  First: the core's own CDMM (an MPU's registers).
        self.devices: list[Any] = [self.cdmm] if self.cdmm is not None else []
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
        #: the last step's instruction could not complete yet (the MT
        #: Module: emips.exceptions.Blocked); the scheduler takes it
        self.blocked = None
        #: the MT processor's gating storage (emips.itc.ITC) or None
        self.itc = None
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
        """Write a 32-bit result (MIPS64: sign-extended to 64 bits)."""
        index &= 31
        if index != 0:
            self.regs[index] = value & M32

    #: write a GPR-width result (MIPS64: all 64 bits; here = write_reg)
    write_dreg = write_reg

    def _sxw(self, value: int) -> int:
        """A 32-bit result as a register holds it (MIPS64: sign-extended)."""
        return value & M32

    def to_pc(self, addr: int) -> int:
        """A 32-bit address as the PC holds it (MIPS64: sign-extended, the
        compatibility segments)."""
        return addr & M32

    def ops64(self) -> bool:
        """64-bit operations are enabled (MIPS64 only)."""
        return False

    def fetch_va(self, pc: int) -> int:
        """The virtual address an instruction fetch at ``pc`` uses (the
        pipelines; MIPS64 with 32-bit addresses: the compatibility address,
        or Address Error)."""
        return pc

    def seq_pc(self, addr: int) -> int:
        """The next sequential PC (a pipeline's fetch; MIPS64 Release 6:
        it may wrap at bit 31, Vol. III 6.03 4.11)."""
        return addr & self.amask

    def branch_pc(self, addr: int) -> int:
        """A branch / jump / ERET target the last executed instruction
        produced, as the PC takes it (MIPS64 Release 6: 4.11's wrap)."""
        return addr & self.amask

    def _bp_va(self, va: int) -> int:
        """A data address as the breakpoints and watch registers compare
        it (MIPS64 with 32-bit addresses: sign-extended)."""
        return va

    def data_wrap(self) -> bool:
        """Data addresses are sign-extended from bit 31 now (MIPS64's 32-bit
        compatibility addressing, Vol. III 4.10 / 4.11)."""
        return False

    def cp0_read(self, reg: int, sel: int = 0) -> int:
        if self._ext_regs and (reg, sel) in self._ext_regs:
            return self._ext.read(self, (reg, sel))
        return self.cp0[(reg, sel)]["value"]

    def cp0_view(self, reg: int, sel: int = 0) -> int:
        """A CP0 register as the views show it: its architectural value
        (MIPS64: EntryLo with RI / XI at 63 / 62, as DMFC0 reads it)."""
        return self.cp0_read(reg, sel)

    def cp0_dread(self, reg: int, sel: int = 0) -> int:
        """A CP0 register as a GPR-width move takes it (MIPS64: DMFC0, an
        MFTR; here MFC0's value)."""
        return self.cp0_read(reg, sel)

    def cp0_dwrite(self, reg: int, sel: int, value: int) -> None:
        """A GPR-width move to a CP0 register (MIPS64: DMTC0, an MTTR;
        here MTC0)."""
        self.mtc0(reg, sel, value)

    def mtc0(self, reg: int, sel: int, value: int) -> None:
        """What an MTC0 does (an MTTR to a CP0 register too): the write,
        and a Compare write clears the timer interrupt."""
        self.cp0_write(reg, sel, value)
        if (reg, sel) == C.COMPARE:             # writing Compare clears TI
            self._cp0_set(C.CAUSE, self.cp0_read(*C.CAUSE) & ~C.CA_TI)
            self._update_ip()

    def cp0_write(self, reg: int, sel: int = 0, value: int = 0) -> None:
        """Software write (``MTC0``): only the writable bits change.  In
        Release 6 a write of an unsupported cacheability attribute to
        ``Config.K0 / KU / K23`` leaves that field unchanged (Vol. III
        Config); before R6 using one is UNDEFINED and it is just stored."""
        if self._ext_regs and (reg, sel) in self._ext_regs:
            self._ext.write(self, (reg, sel), value)
            return
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
        cell["value"] = (cell["value"] & ~mask | value & mask) & (
            M64 if (reg, sel) in self._wide_cp0 else M32)
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
        self.cp0[key]["value"] = value & (M64 if key in self._wide_cp0 else M32)

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
        # (an MT VPE's sets can shrink under it, emips.mt: UNDEFINED -> set 0)
        self.regs = self.shadow[new] if new < len(self.shadow) else self.shadow[0]

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

    def _tc_id(self) -> int:
        """The thread context issuing (the MT Module: its TC number, as the
        EJTAG breakpoints' TCuse and PCsample see it; 0 without)."""
        return self.mt.current[self.vpe].n if self.mt is not None else 0

    def _watch(self, start: int, n: int, kind: int) -> bool:
        """The watch registers (Vol. III 9.51 / 9.52; the 24K's 4 pairs):
        a pair whose enable ``kind`` (I / R / W) is set matches when a
        byte of the access lies in its doubleword (VAddr 31:3 — MIPS64
        with 64-bit addresses: 63:3 —, Mask 11:3 bits ignored) and G = 1
        or its ASID is EntryHi's; WatchHi records the kind.  Returns True
        if a Watch exception is taken now; with EXL or ERL set it is
        deferred instead (Cause.WP)."""
        hit = False
        vamask = self.amask
        end = (start + n - 1) & vamask
        for lo_key, hi_key in zip(C.WATCHLO, C.WATCHHI):
            lo = self.cp0[lo_key]["value"]
            if not lo & kind:
                continue
            hi = self.cp0[hi_key]["value"]
            if not hi & C.WATCHHI_G and (hi >> 16) & 0xFF != self._asid():
                continue
            care = ~(hi & 0xFF8 | 7) & vamask
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
        hits, brk = (ej.instruction_match(pc, self._asid(), self._tc_id()) if ej.ib_armed
                     else (0, False))
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
        addr = self._bp_va(addr)                # (MIPS64: the 64-bit address)
        if span is not None:
            span = (self._bp_va(span[0]), span[1])
        if ej.db_armed:                         # (a 64-bit core: a doubleword bus)
            width = 8 if size == 8 or self.wide else 4
            start, n = span if span is not None else (addr & ~(size - 1), size)
            lanes = 0
            for a in range(start, start + n):
                k = a % width
                lanes |= 1 << (width - 1 - k if self.big_endian else k)
            lanes &= (1 << width) - 1
            precise, brk, value = ej.data_match(addr, lanes, width, store, self._asid(),
                                                self._tc_id())
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
        if self.wide:                           # a 64-bit register (MD01138 2.2.12.2)
            self._wide_cp0 = self._wide_cp0 | {C.CMGCRBASE}
        self.cmgcr_moved(cps.gcr.full_base)

    def cmgcr_moved(self, base: int) -> None:
        """GCR_BASE (or GCR_BASE_UPPER) changed: CMGCRBase follows it —
        MIPS32 PA 35:15 in 29:11, MIPS64 PA 39:15 in 35:11 (the P6600,
        MD01138 Table 2.85)."""
        self.cp0[C.CMGCRBASE]["value"] = (base >> 4) & (
            0xFFFFFFF800 if self.wide else 0x3FFFF800)

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
        cp0 = self.cp0                          # (Cause / Count / Compare: never MT's)
        if cp0[C.CAUSE]["value"] & C.CA_DC:
            return
        if self._dm and not cp0[C.DEBUG]["value"] & EJ.COUNTDM:
            return                              # Count stopped in Debug Mode
        self._count_phase += cycles
        if self._count_phase < self.ccres:
            return                              # (no tick this cycle)
        cell = cp0[C.COUNT]
        count = cell["value"]
        compare = cp0[C.COMPARE]["value"]
        hit = False
        while self._count_phase >= self.ccres:
            self._count_phase -= self.ccres
            count = (count + 1) & M32
            hit = hit or count == compare
        cell["value"] = count
        if hit:
            self._cp0_set(C.CAUSE, self.cp0_read(*C.CAUSE) | C.CA_TI)
            self._update_ip()

    def nmi(self) -> None:
        """An edge on the NMI input: taken before the next instruction,
        whatever Status says (Vol. III "Non Maskable Interrupt")."""
        self.nmi_pending = True

    def interrupt_request(self, ixmt: bool | None = None) -> "_Arch | None":
        """What the core takes before its next instruction (MD00343 Table
        5.1): a Debug Single Step, a Debug Interrupt, an NMI, or an
        enabled interrupt (``Status.IE``, ``EXL = ERL = 0``; IM & IP, or in
        EIC mode the controller's RIPL above ``Status.IPL``).  In Debug
        Mode none (MD00047 2.5); the DCR's IntE / NMIE mask interrupts /
        NMIs in Non-Debug Mode (the NMI stays pending).  An MT VPE's TC
        with TCStatus.IXMT (``ixmt``; default: the bound TC's) takes no
        interrupt (MD00378 6.12)."""
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
        cp0 = self.cp0
        status = cp0[C.STATUS]["value"]
        cause = cp0[C.CAUSE]["value"]
        if (status & C.ST_IE and not status & (C.ST_EXL | C.ST_ERL)   # (both modes need
                and dcr & EJ.DCR_INTE                                  # these: first)
                and not (self._ixmt if ixmt is None else ixmt)):
            if self.interrupt_mode() == "eic":
                if self.irq_lines > (status >> C.ST_IM_SHIFT + 2) & 0x3F:
                    return _Arch(C.INT)
            elif C.interrupt_pending(status, cause):
                return _Arch(C.INT)
        if cause & C.CA_WP and not status & (C.ST_EXL | C.ST_ERL):
            return _Arch(C.WATCH, info={"why": "deferred (Cause.WP)"})
        return None

    def interrupt_signal(self) -> bool:
        """An interrupt signal is active on this core / VPE, whether or not
        Status.IE, EXL / ERL or TCStatus.IXMT let it be taken: Cause.IP &
        Status.IM (EIC mode: RIPL above Status.IPL), an NMI, a debug
        interrupt — what ends an MT TC's WAIT on the 34K (Config7.WII,
        MD00534 10.2.1)."""
        if self.nmi_pending or self.dint_pending:
            return True
        status = self.status
        if self.interrupt_mode() == "eic":
            return self.irq_lines > (status >> C.ST_IM_SHIFT + 2) & 0x3F
        cause = self.cp0_read(*C.CAUSE)
        return bool((status >> 8) & (cause >> 8) & 0xFF)

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
        addr &= self.vamask
        if self._dm and self._in_dseg(addr):
            return Translation(addr, None, CCA_UNCACHED, EJ.segment(addr), False,
                               "EJTAG " + ("debug registers" if addr >= EJ.DRSEG_BASE
                                           else "probe memory") + ": no physical address")
        return self.mmu.describe(self, addr)

    def _peek(self, addr: int) -> tuple[int, int]:
        """``(pa, cca)`` of a virtual address for a view (no side effects);
        :class:`UnmappedAddress` when nothing maps it (dseg in Debug
        Mode: no physical address)."""
        addr &= self.vamask
        if self._dm and self._in_dseg(addr):
            raise UnmappedAddress(f"0x{addr:08x} is EJTAG {EJ.segment(addr & M32)} "
                                  "(Debug Mode): no physical address",
                                  name=None, word=None, addr=addr)
        hit = self.mmu.peek(self, addr)
        if hit is None:
            raise UnmappedAddress(f"no TLB entry maps 0x{addr:08x} now",
                                  name=None, word=None, addr=addr)
        return hit

    def _in_dseg(self, addr: int) -> bool:
        """In EJTAG's dseg (0xFF200000-0xFF3FFFFF; MIPS64: sign-extended)."""
        return EJ.in_dseg(addr & M32)

    def _xlate_peek(self, addr: int) -> int:
        """The physical address of *addr* now, -1 when nothing maps it
        (no side effects)."""
        try:
            return self._peek(addr)[0]
        except UnmappedAddress:
            return -1

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
        addr &= self.vamask
        pa, cca = self._xlate(self, addr, kind)
        if (addr & 0x3FF) + size <= 0x400:
            return [pa + i for i in range(size)], cca
        return [pa] + [self._xlate(self, (addr + i) & self.vamask, kind)[0]
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
        hit = self.mmu.peek(self, addr & self.vamask)
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
        if self.itc is not None and self.itc.covers(pa):
            return self._itc_part(pa, self.itc.peek(pa), 1)
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
        addr &= self.vamask
        if self._dm and self._in_dseg(addr) and not self.cp0[C.DEBUG]["value"] & EJ.LSNM:
            byte = self.ejtag.peek(addr & M32, 1)
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
        self._mem_log.append((kind, addr & self.vamask, *entry, self.translate(addr)))

    def load(self, addr: int, size: int, kind: str = "load") -> int:
        """``size`` bytes at virtual address *addr* as an unsigned value, in
        the machine's byte order (no alignment / access checks); through
        the data cache when the address is cacheable.  ``kind = "pcload"``
        for a PC-relative load (RI / XI rules, Vol. III 4.9.4).  In Debug
        Mode dseg goes to the EJTAG unit (unless ``Debug.LSNM``)."""
        if self._dm and self._in_dseg(addr) and not self.cp0[C.DEBUG]["value"] & EJ.LSNM:
            value = self.ejtag.load(addr & M32, size)
            if self._mem_log is not None:
                self._log("read", addr, size, value)
            return value
        pas, cca = self._access(addr, size, kind)
        if self.itc is not None and self.itc.covers(pas[0]):
            if addr % size and self.vpcore is not None:     # the ITU: aligned only
                raise _Arch(C.ADEL, badvaddr=addr)           # (MD01196 11.3)
            return self._itc_load(addr, pas[0], size)
        if self.perf.armed and cca not in self.cached_cca:
            self.perf.pending.append("uncached_load")
        if self._vc is None and self._pdt_acc is None:
            return self.load_pa(addr, pas, cca, size)
        value = self.load_pa(addr, pas, cca, size)
        if self._vc is not None:
            self._value_checks(addr & M32, size, value, False)
        if self._pdt_acc is not None:               # MIPS Trace: the load
            self._pdt_acc.append(("load", addr & self.vamask, size, value))
        return value

    def _itc_shift(self, pa: int, size: int) -> int:
        """Where the ``size`` bytes at ``pa`` sit in an ITC cell's value
        (32 bits: address bit 2 ignored, MD00378 A.2; 64 bits: a doubleword
        in this byte order)."""
        n = self.itc.bytes
        k = pa & (n - 1)
        return max(0, 8 * (n - size - k) if self.big_endian else 8 * k)

    def _itc_part(self, pa: int, word: int, size: int) -> int:
        """The ``size`` bytes at ``pa`` of an ITC cell's value."""
        return (word >> self._itc_shift(pa, size)) & ((1 << (8 * size)) - 1)

    def _itc_load(self, addr: int, pa: int, size: int) -> int:
        """A load from gating storage (emips.itc, emips.itu): uncached, may
        block; an ITU error response is a data bus error."""
        try:
            value = self._itc_part(pa, self.itc.load(self, pa, size), size)
        except BusError:
            return self._read_bus_error(addr, size)
        if self._mem_log is not None:
            self._log("read", addr, size, value)
        if self._pdt_acc is not None:
            self._pdt_acc.append(("load", addr & self.vamask, size, value))
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
        pa0 = pas[0]
        w = self.mem.word(pa0) if size == 4 and not pa0 & 3 and pas[3] == pa0 + 3 else None
        if w is not None:                       # (one written word: the cached copy)
            value = w if self.big_endian else int.from_bytes(w.to_bytes(4, "big"), "little")
        else:
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
              atomic: bool = False) -> bool | None:
        """Store the low ``size`` bytes of *value* at virtual address
        *addr*, machine order; through the data cache when cacheable, the
        write buffer when there is one (``gather = False``: a partial or
        conditional store, never gathered as uncached accelerated).  A
        store to gating storage returns whether it succeeded (emips.itc:
        the E/F try view of a full cell drops it)."""
        if self._dm and self._in_dseg(addr) and not self.cp0[C.DEBUG]["value"] & EJ.LSNM:
            value &= (1 << (8 * size)) - 1
            if self._mem_log is not None:
                self._log("write", addr, size, self.ejtag.peek(addr & M32, size) or 0, value)
            self.ejtag.store(addr & M32, value, size)
            return
        pas, cca = self._access(addr, size, "store")
        value &= (1 << (8 * size)) - 1
        if self.itc is not None and self.itc.covers(pas[0]):
            if addr % size and self.vpcore is not None:     # the ITU: aligned only
                raise _Arch(C.ADES, badvaddr=addr)
            if self._mem_log is not None:
                self._log("write", addr, size, self._itc_part(pas[0], self.itc.peek(pas[0]),
                                                               size), value)
            if self._pdt_acc is not None:
                self._pdt_acc.append(("store", addr & self.vamask, size, value))
            # gating storage: may block; the bytes in their place (a store /
            # load pair of one size at one address sees the same data, A.2)
            try:
                return self.itc.store(self, pas[0], value << self._itc_shift(pas[0], size),
                                      size)
            except BusError:                    # an ITU error response (MD01196 11.5)
                self.bus_error_pending = True
                return False
        if self.perf.armed and cca not in self.cached_cca:
            self.perf.pending.append("uncached_store")
        if self._vc is not None:
            self._value_checks(addr & M32, size, value, True)
        if self._pdt_acc is not None:               # MIPS Trace: the store
            self._pdt_acc.append(("store", addr & self.vamask, size, value))
        if self.system is not None:
            if self.system.store_check(self, pas[0], size, cca, value):
                return                          # a register error / an L2-only SYNC
        c = self.caches
        cached = self._uses_dcache(cca)
        if c is not None and not cached:
            order = "big" if self.big_endian else "little"
            old = None
            if self._mem_log is not None:
                old = bytes(self._view_byte((addr + i) & self.vamask, pa, False)
                            for i, pa in enumerate(pas))
            if c.uncached_store(pas[0], value.to_bytes(size, order), cca, gather):
                if old is not None:
                    self._log("write", addr, size, int.from_bytes(old, order), value)
                return
        if cached:
            order = "big" if self.big_endian else "little"
            old = None
            if self._mem_log is not None:
                old = bytes(self._view_byte((addr + i) & self.vamask, pa, True)
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
        addr &= self.vamask
        if self._dm and self._in_dseg(addr) and not self.cp0[C.DEBUG]["value"] & EJ.LSNM:
            word = self.ejtag.peek(addr & M32, 4)
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
        data = bytes(self._view_byte((addr + i) & self.vamask, pa, cached)
                     for i, pa in enumerate(pas))
        return int.from_bytes(data, "big" if self.big_endian else "little")

    @staticmethod
    def _no_code(addr: int, i: int, pa: int) -> FetchError:
        va = (addr + i) & M64 if addr > M32 else (addr + i) & M32
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

    def fetch_half(self, addr: int) -> int:
        """Fetch the MIPS16e halfword at *addr* (Vol. IV-a 3.16: the
        halfword an LH of that address would read, in either byte order):
        the aligned word's fetch, needing only these two bytes.  (MIPS64:
        the address the fetch uses — Config.AT = 2: all 64 bits.)"""
        addr &= self.vamask
        off = addr & 2
        word = self.fetch_word(addr & ~3, need=0x3 << off)
        return word >> 16 if (off == 0) == self.big_endian else word & 0xFFFF

    def code_half(self, addr: int) -> int:
        """The halfword a fetch at *addr* would get now (views)."""
        addr &= self.amask
        try:
            word = self.code_word(addr & ~3)
        except FetchError:
            hi, lo = self.read_byte(addr), self.read_byte(addr + 1)
            return (hi << 8 | lo) if self.big_endian else (lo << 8 | hi)
        off = addr & 2
        return word >> 16 if (off == 0) == self.big_endian else word & 0xFFFF

    def fetch_word(self, addr: int, need: int = 0xF) -> int:
        """Fetch the instruction word at virtual address *addr* (through the
        instruction cache when cacheable); **any** unwritten byte is a
        :class:`FetchError` (simulator halt).  While :attr:`tracing`, the
        cache accesses it made are left in :attr:`fetch_accesses`, its
        translations (TLB / BAT) in :attr:`fetch_tlb`."""
        if self.pdt.active and self.caches is not None and self.caches.i is not None:
            stats = self.caches.i.stats             # the Im bit of MIPS Trace
            before = stats.misses
            try:
                return self._fetch_word(addr, need) if not self.tracing else \
                    self._fetch_watched(addr, need)
            finally:
                self.fetch_missed = stats.misses != before
        if not self.tracing:
            return self._fetch_word(addr, need)
        return self._fetch_watched(addr, need)

    def _fetch_watched(self, addr: int, need: int = 0xF) -> int:
        c = self.caches
        cw = c is not None and c.log is None
        mmu = self.mmu if hasattr(self.mmu, "watch") and self.mmu.log is None else None
        if cw:
            c.watch()
        if mmu is not None:
            mmu.watch()
        try:
            return self._fetch_word(addr, need)
        finally:
            if cw:
                self.fetch_accesses = c.taken()
            if mmu is not None:
                self.fetch_tlb = mmu.taken()

    def _fetch_word(self, addr: int, need: int = 0xF) -> int:
        """``need``: the bytes that must hold code (bit k = byte k of the
        word; a MIPS16e halfword needs two)."""
        addr &= self.vamask
        if self._dm and self._in_dseg(addr):    # a fetch in dseg: always dseg
            return self.ejtag.fetch(addr & M32)
        pas, cca = self._access(addr, 4, "fetch")
        c = self.caches
        if c is not None and self._icached(cca):
            try:
                data, known = c.fetch(addr, pas[0], cca)
            except CacheError as err:
                raise self._cache_arch(err) from None
            except BusError:                    # IBE: precise (MD00343 5.8.13)
                raise _Arch(C.IBE, info={"why": "instruction fetch bus error"}) from None
            if known & need != need:
                i = next(k for k in range(4) if need >> k & 1 and not known >> k & 1)
                raise self._no_code(addr, i, pas[0] + i)
            return int.from_bytes(data, "big" if self.big_endian else "little")
        if c is not None:                       # write buffer before the read
            c.read_request(pas[0], 4, "fetch", no_cache=cca in self.cached_cca)
        if self.board is not None and self.board.bus.covers(pas[0]):
            try:
                return self.board.bus.load(self, pas[0], 4)
            except BusError:                    # IBE: precise (MD00343 5.8.13)
                raise _Arch(C.IBE, info={"why": "instruction fetch bus error"}) from None
        pa0 = pas[0]
        if not pa0 & 3 and pas[3] == pa0 + 3:   # (one word: the cached copy)
            w = self.mem.word(pa0)
            if w is not None:
                for dev in self.devices:
                    if dev.covers(pa0):
                        return dev.load(self, pa0, 4)
                return w if self.big_endian else int.from_bytes(w.to_bytes(4, "big"), "little")
        mem = self.mem.get_memory()
        raw = [mem.get(format(pa, "x")) for pa in pas]
        for i, (pa, byte) in enumerate(zip(pas, raw)):
            if byte is None:
                if need >> i & 1:
                    raise self._no_code(addr, i, pa)
                raw[i] = "00"
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
        addr &= self.vamask
        if self._dm and self._in_dseg(addr):    # the probe's memory (dmseg)
            word = self.ejtag.peek(addr & M32, 4) if addr & M32 < EJ.DRSEG_BASE else None
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
        pa0 = pas[0]
        if not pa0 & 3 and pas[3] == pa0 + 3:   # (one word: the cached copy)
            w = self.mem.word(pa0)
            if w is not None:
                return w if self.big_endian else int.from_bytes(w.to_bytes(4, "big"), "little")
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
        addr &= self.vamask
        if not self.address_ok(addr):
            raise _Arch(C.ADEL, badvaddr=addr)
        if self.itc is not None and self.cp0_read(*C.ERRCTL) & ITC_ERRCTL:
            done = self.itc.cache_op(self, op, addr)   # the ITC's address maps
            if done is not None:
                self.last_cache_op = done
                return done
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
        addr &= self.vamask
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
        addr &= self.vamask
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
                     atomic: bool = False, span: tuple[int, int] | None = None,
                     aligned: bool = False) -> int:
        """Address check of a data access: alignment to ``size`` and the
        segment rights of the current mode, else Address Error (AdEL for
        loads, AdES for stores) with ``BadVAddr = addr``.  Release 6
        handles misaligned ordinary loads/stores in hardware (here); only
        LL/SC (``atomic``) and the DSP Module's indexed loads (``aligned``:
        MD00374 LWX / LHX) must stay aligned.  First the data breakpoints
        and watch registers, which come before an address error."""
        addr &= self.vamask
        if self.agent_watch:
            self._agent_watch(addr, size, store, span)
        if self._dhw and not self._dm:
            self._data_checks(addr, size, store, span)
        misaligned = addr % size and (self.release != "R6" or atomic or aligned)
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
        if self.waiting and self.mt is not None and self.interrupt_signal():
            self.wake()                         # Config7.WII: the WAIT is over (MD00534
            #                                     10.2.1), the interrupt is not taken
        if self.waiting:
            self.advance(1)
            if self.ejtag.sample_due:           # asleep: the WAIT is what graduated
                self.ejtag.sample(self.wait_pc)
            if self.pdt.active:
                self._trace_cycle(False, asleep=True)
            return None
        size = 4
        try:
            # MIPS64: the virtual address the fetch uses (Config.AT = 1: the
            # compatibility address, AdEL outside those segments)
            va = self._compat(pc) if self.wide else pc
            if self._ihw:                       # (the PC as it is: MIPS64 64-bit)
                self.instruction_checks(pc)
            self._insn_raw = None
            if self.nano:                       # nanoMIPS: 16 / 32 / 48 bits (MD01247)
                size, word, dec = self._fetch_nm(pc)
            elif self.isa_mode and self.compressed == "micromips":
                size, word, dec = self._fetch_mm(pc)
            elif self.isa_mode:                     # MIPS16e: 16 / 32-bit instructions
                size, word, dec = self._fetch16(pc)
                self._m16_base = (pc - self._slot_back) & self.amask if in_slot else pc
            else:
                if va & 3 or not self.address_ok(va) or not self.mips32:
                    raise _Arch(C.ADEL, badvaddr=pc)
                word = self.fetch_word(va)
                try:
                    dec = self.isa.decode(word)
                except ISADecodeError as exc:
                    if self.exceptions == "raise":
                        raise
                    raise self.reserved(word) from exc
                self._m16 = self._mm = None
            self.r6_slot_check(dec, in_slot, forbidden)
            if in_slot and dec.name in _MM_NO_SLOT:
                raise _Arch(C.RI)               # UNPREDICTABLE in a delay slot: QEMU's RI
            result = self._traced(pc, word, dec, in_slot)
            if self.last_events is not None:    # completed: count it
                self.perf.retire(*self.last_events)
                self.last_events = None
        except _Arch as exc:
            self.advance(1)
            if self.pdt.active:
                self._trig = []
                self._trace_cycle(False)
            if word is None and self._insn_raw is not None:
                word = self._insn_raw           # an RI found decoding what was fetched
            self._take_exception(exc, pc, word, dec, in_slot)
            return dec
        except WatchpointHit:                   # a debugger stop: this instruction again
            self.slot_armed, self.slot_target, self._forbidden = in_slot, outer_target, forbidden
            self.last_events = None
            raise
        except Blocked as b:                    # MT: it issues again when it can complete
            self.slot_armed, self.slot_target, self._forbidden = in_slot, outer_target, forbidden
            self.last_events = None
            self.blocked = b
            self.advance(1)
            if self.pdt.active:
                self._trig = []
                self._trace_cycle(False)
            return dec
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
        if self._redirect is not None:              # ERET, a compact jump: no delay slot
            self._jump(self._redirect)
        elif result == NULLIFY:
            self.pc = (pc + 8) & self.amask
        elif result is not None or has_delay_slot(dec.entry):
            # taken: the slot runs, then the target; not taken: the slot is
            # still a delay slot (BD on a fault there), then PC + 8 — or,
            # microMIPS, the instruction after the slot (16 or 32 bits): None
            self.slot_armed = True
            amask = self.amask
            self.slot_target = result & amask if result is not None else \
                None if self.isa_mode else (pc + 8) & amask
            self._slot_back = size
            self._slot_raw = self._insn_raw
            self.pc = (pc + size) & amask
        elif in_slot and outer_target is not None:
            self._jump(outer_target)
        else:
            self.pc = (pc + size) & self.amask
        return dec

    def _jump(self, target: int) -> None:
        """Go to ``target``.  With MIPS16e its bit 0 is the ISA mode there
        (Vol. IV-a 3.7.4: JR / JALR / JRC / JALRC load it, JALX inverts it,
        the MIPS16e branches keep 1; ERET / DERET: EPC / DEPC bit 0); bit 0
        of the PC is 0.  Without, the address as is (an odd one faults on
        the fetch)."""
        target &= self.amask
        if self.compressed is not None:
            self.isa_mode = target & 1
            target &= ~1
        self.pc = target

    def _fetch16(self, pc: int) -> tuple[int, int, Decode]:
        """MIPS16e: fetch and decode the instruction at ``pc`` (2 or 4
        bytes); return (size, the word the executor gets, its Decode) — the
        MIPS32 expansion where there is one (emips.mips16.expand), else the
        ``M16.*`` entry with the raw encoding."""
        if not self.address_ok(pc):
            raise _Arch(C.ADEL, badvaddr=pc)
        h = self.fetch_half(pc)
        got: list[int] = []

        def second() -> int:
            if not self.address_ok(pc + 2):
                raise _Arch(C.ADEL, badvaddr=(pc + 2) & self.amask)
            got.append(self._second_half(pc + 2))
            return got[0]
        try:
            i = M16.decode(h, second, arch=self.arch)
        except ISADecodeError as exc:
            # (BadInstr: the encoding fetched, Vol. III 9.25)
            self._insn_raw = h << 16 | got[0] if got else h
            if self.exceptions == "raise":
                raise
            raise _Arch(C.RI) from exc
        self._m16 = i
        self._insn_raw = i.raw
        word = M16.expand(i)
        if word is None:
            return i.size, i.raw, Decode(i.raw, M16.ENTRIES[i.form])
        return i.size, word, self._isa16.decode(word)

    def _fetch_mm(self, pc: int) -> tuple[int, int, Decode]:
        """microMIPS: fetch and decode the instruction at ``pc`` (2 or 4
        bytes); return (size, the word the executor gets, its Decode) — the
        MIPS32 twin where there is one (emips.micromips.expand), else the
        ``MM.*`` entry with the raw encoding."""
        if not self.address_ok(pc):
            raise _Arch(C.ADEL, badvaddr=pc)
        h = self.fetch_half(pc)
        got: list[int] = []

        def second() -> int:
            if not self.address_ok(pc + 2):
                raise _Arch(C.ADEL, badvaddr=(pc + 2) & self.amask)
            got.append(self._second_half(pc + 2))
            return got[0]
        try:
            i = MM.decode(h, second, release=self.release, dsp=self.dsp,
                          mt=self.mt is not None, arch=self.arch, msa=self.msa)
            word = MM.expand(i)
            dec = Decode(i.raw, MM.ENTRIES[i.form]) if word is None else \
                self._twin_isa(i).decode(word)
        except ISADecodeError as exc:
            # (BadInstr: the encoding fetched, Vol. III 9.25)
            self._insn_raw = h << 16 | got[0] if got else h
            if self.exceptions == "raise":
                raise
            raise self.reserved(self._insn_raw, micromips=True) from exc
        self._mm = i
        self._insn_raw = i.raw
        return i.size, (i.raw if word is None else word), dec

    def _second_half(self, addr: int) -> int:
        """The second halfword of a 32-bit compressed instruction: its
        fetch's cache accesses / translations join the first's (views)."""
        acc, tlb, missed = self.fetch_accesses, self.fetch_tlb, self.fetch_missed
        h = self.fetch_half(addr)
        if self.fetch_accesses is not acc:
            self.fetch_accesses = acc + self.fetch_accesses
        if self.fetch_tlb is not tlb:
            self.fetch_tlb = tlb + self.fetch_tlb
        self.fetch_missed = self.fetch_missed or missed
        return h

    def fetch_compressed(self, pc: int) -> tuple[int, int, Decode, Any]:
        """The pipelines' fetch in MIPS16e / microMIPS mode: (size, the
        word the executor gets, its Decode, the compressed instruction)
        — as :meth:`step` fetches (raises :class:`_Arch`, or
        :class:`ISADecodeError` with ``exceptions="raise"``)."""
        if self.compressed == "micromips":
            size, word, dec = self._fetch_mm(pc)
            return size, word, dec, self._mm
        size, word, dec = self._fetch16(pc)
        return size, word, dec, self._m16

    def enter(self, pc: int, isa: int, insn: Any, in_slot: bool, back: int,
              slot_raw: int | None) -> None:
        """The pipelines, before an instruction executes or takes its
        exception: the per-instruction state :meth:`step` keeps — the ISA
        mode it runs in, its compressed form (None: MIPS32), the size of
        the branch whose delay slot it is (EPC, the MIPS16e PC-relative
        base) and that branch's compressed encoding (BadInstrP)."""
        self.isa_mode = isa
        mm = self.compressed == "micromips"
        self._mm = insn if mm else None
        self._m16 = None if mm else insn
        self._insn_raw = insn.raw if insn is not None else None
        self._slot_back = back
        self._slot_raw = slot_raw
        self._m16_base = (pc - back) & self.amask if in_slot else pc

    def _twin_isa(self, i) -> ISA:
        """The table a microMIPS instruction's MIPS32 (MIPS64) twin decodes
        with: the core's (a Release 6 row with a wider field than MIPS32 /
        MIPS64 R6 has twins to the Release 5 row of the same operation)."""
        if not getattr(i.row, "r5_twin", False):
            return self.isa
        return _ISA_R5 if self.arch == "mips32" else ISA.for_release("R5", arch=self.arch)

    def _handler_isa(self, ebase_relative: bool) -> int:
        """The ISA mode a handler starts in: MIPS32 with MIPS16e (Vol.
        IV-a 3.7.3); microMIPS: Config3.ISAOnExc for the vectors at EBase
        offsets, else (BEV = 1, NMI, reset) the reset ISA, Config3.ISA bit
        0 (Vol. III Table 9.67; MD00582 3.2; MD00047 2.3.3)."""
        if self.compressed != "micromips":
            return 0
        cfg3 = self.cp0[C.CONFIG3]["value"]
        return (cfg3 >> 16) & 1 if ebase_relative else (cfg3 >> C.CFG3_ISA_SHIFT) & 1

    def _debug_isa(self) -> int:
        """The ISA mode of the debug handler (MD00047 2.3.3): in the probe's
        memory (ProbTrap) ECR.ISAOnDebug, else the reset ISA."""
        if self.compressed != "micromips":
            return 0
        if self.ejtag.prob_trap:
            return int(self.ejtag.tap.isa_on_debug_bit(self))
        return self._handler_isa(False)

    def _restart_pc(self, pc: int, in_slot: bool) -> int:
        """EPC / ErrorEPC / DEPC: the instruction, or in a delay slot the
        branch (``_slot_back`` bytes before), with the ISA mode in bit 0
        (Vol. III 9.30, Vol. IV-a 3.7.3)."""
        base = (pc - self._slot_back) if in_slot else pc
        return (base | self.isa_mode) & self.amask

    # -- execution unit interface (used by step() and emips.pipeline) ---------

    def execute(self, pc: int, word: int, dec: Decode, *, in_slot: bool = False,
                sources: dict | None = None) -> int | None:
        """Functionally execute one decoded instruction located at ``pc``:
        no fetch, no clock tick, no PC update.  Returns the executor result
        (branch/jump target, :data:`NULLIFY` or ``None``); ``ERET``'s
        target is left in ``_redirect``.  Raises :class:`_Arch` for an
        architectural exception, before any state change."""
        self.pc = pc & self.amask
        self._redirect = None
        if in_slot and dec.name in _MM_NO_SLOT:
            raise _Arch(C.RI)                   # (as step)
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
        global _trace
        trace = _trace
        if trace is None:
            from . import trace
            _trace = trace
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
        plain = _PLAIN.get(id(entry))
        if plain is not None and plain[0] is entry:     # no gate: its executor
            return plain[1](self, word)
        name, family = entry["name"], entry["family"]
        status = self.status
        if entry.get("ase") in ("dsp", "dspr2") and not status & C.ST_MX:
            raise _Arch(C.DSPDIS)               # MD00374 Table 3.7
        if entry.get("ase") == "msa" and family != "alu":
            self.msa_gate()                     # (LSA / DLSA: Config3.MSAP only)
            if self.msa_wrp is not None:
                return self._partitioned(entry, word)
        if family in FPU_FAMILIES or name in ("MOVF", "MOVT", "PREFX"):
            if not status & C.ST_CU1:
                raise _Arch(C.CPU, ce=1)
            if self.msa_wrp is not None and classify(entry) == BUCKET_EXEC:
                return self._partitioned(entry, word)       # (FPRs = vector registers)
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
        run = _EXECUTORS[name]
        if entry.get("ase") not in ("dsp", "dspr2", "msa") and family not in FPU_FAMILIES \
                and name not in ("MOVF", "MOVT", "PREFX") and family != "cop2" \
                and name not in EVA_NAMES and name not in PRIVILEGED_NAMES:
            _PLAIN[id(entry)] = (entry, run)    # (the checks above never apply to it)
        return run(self, word)

    def msa_gate(self) -> None:
        """MSA's checks before an MSA instruction runs (MD00866 3.2, 3.3.2,
        3.5; the Answers' order, QEMU's P5600 / I6400 agree): RI when the
        FPU is usable in 32-bit mode (Status.CU1 = 1, FR = 0), MSA Disabled
        when Config5.MSAEn = 0 (Config3.MSAP = 0: the word does not decode,
        RI)."""
        status = self.status
        if status & C.ST_CU1 and not status & C.ST_FR:
            raise _Arch(C.RI)
        if not self.cp0[(16, 5)]["value"] & _V.CFG5_MSAEN:
            raise _Arch(C.MSADIS)

    def _partitioned(self, entry: Mapping[str, Any], word: int) -> int | None:
        """An MSA / FPU instruction with MSAIR.WRP: MSA Disabled while one of
        its vector registers is not mapped or not saved (MSARequest), then
        the instruction, then MSAModify |= what it wrote (3.4.5 / 3.4.6)."""
        accessed, written = _V.registers_of(entry, word, self)
        self.msa_wrp.check(self, accessed)
        result = _EXECUTORS[entry["name"]](self, word)
        self.msa_wrp.done(written)
        return result

    def reserved(self, word: int, *, micromips: bool = False) -> _Arch:
        """The exception of a word that decodes to no instruction: RI — but
        on an MSA core a word of MSA's opcode space takes MSA's gate first
        (MD00866 3.5, 1.11: "reserved MSA opcodes" raise MSA Disabled while
        Config5.MSAEn = 0)."""
        if self.msa and _V.in_space(word, micromips, self.arch):
            try:
                self.msa_gate()
            except _Arch as exc:
                return exc
        return _Arch(C.RI)

    def msa_address(self, addr: int, *, store: bool) -> int:
        """LD / ST.df's address check: the segment rights, data breakpoints
        and watches of the 16 bytes; no alignment restriction (MD00866 LD /
        ST: misaligned in hardware, decision 9)."""
        addr &= self.vamask
        if self.agent_watch:
            self._agent_watch(addr, 16, store, None)
        if self._dhw and not self._dm:
            self._data_checks(addr, 16, store, None)
        if not self.address_ok(addr) or not self.address_ok((addr + 15) & self.vamask):
            raise _Arch(C.ADES if store else C.ADEL, badvaddr=addr)
        return addr

    def msa_misaligned(self, addr: int, *, store: bool) -> None:
        """An element-misaligned LD / ST on a core that traps some (the
        preset's :attr:`msa_ade`): Address Error."""
        if "page16k" in self.msa_ade and (addr & 0x3FFF) + 16 > 0x4000 or \
                "uncached" in self.msa_ade and self._cca(addr) not in self.cached_cca:
            raise _Arch(C.ADES if store else C.ADEL, badvaddr=addr)

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
        if "mpu" in exc.info:                   # MPU_Config: the last protection exception
            self.mmu.exception_taken(self, exc.info["mpu"])

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
        epc = self._restart_pc(pc, in_slot)
        self.isa_mode = self._handler_isa(not status & C.ST_BEV)
        if not status & C.ST_EXL:
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
        if refill and exc.info.get("xtlb"):     # MIPS64: XTLB Refill (Vol. III Table
            offset = 0x080                      # 6.9: 64-bit addresses enabled)
            entry["xtlb"] = True
        ebase = self.cp0[C.EBASE]["value"]
        if getattr(self.mmu, "sc", False) and not self.cp0_read(16, 5) & (1 << 30):
            # Config5.K = 0 (the legacy configuration): EBase[31:30] read as
            # 10 — the vectors stay in kseg0 / kseg1 (MD01025 3.6.3.2)
            ebase = (ebase & 0x3FFFFFFF) | 0x80000000
        self.pc = self.to_pc(C.vector(
            status, ebase, interrupt_iv=exc.code == C.INT and bool(cause & C.CA_IV),
            refill=refill, bev_base=self.bev_base, offset=offset))
        entry.update(epc=self.cp0[C.EPC]["value"], bd=bool(cause & C.CA_BD),
                     vector=self.pc, refill=refill)
        if exc.info.get("why"):
            entry["why"] = exc.info["why"]
        self.exception_log.append(entry)

    #: exceptions that do not set BadInstr / BadInstrP (Vol. III 6.02
    #: 9.25: not synchronous to an instruction, and Watch)
    _NO_BADINSTR = frozenset({C.INT, C.MCHECK, C.IBE, C.DBE, C.CACHE_ERROR, C.WATCH})

    def _bad_instr(self, exc: _Arch, pc: int, word: int | None, in_slot: bool) -> None:
        """BadInstr = the faulting instruction word, BadInstrP = the branch
        before it when it is in a delay slot (Vol. III 9.25 / 9.26; Release
        3+ with the TLB, required in Release 6).  Without an instruction
        word (a fetch fault) they are UNPREDICTABLE: left unchanged."""
        if not self.has_bad_instr or exc.code in self._NO_BADINSTR or word is None:
            return
        compressed = self._insn_raw is not None     # its own encoding (9.25)
        self._cp0_set(C.BADINSTR, self._insn_raw if compressed else word)
        if in_slot:
            if compressed and self._slot_raw is not None:
                self._cp0_set(C.BADINSTRP, self._slot_raw)
                return
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
        depc = self._restart_pc(pc, in_slot)
        self.isa_mode = self._debug_isa()
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
        self.pc = self.to_pc(self.debug_vector())
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
        depc = self._restart_pc(pc, in_slot)
        self.isa_mode = self._debug_isa()
        dbg = cell["value"] & ~(EJ.DBD | EJ.CAUSE_BITS | (0x1F << EJ.DEXC_SHIFT))
        dbg |= (dexc << EJ.DEXC_SHIFT) | EJ.IEXI
        if in_slot:
            dbg |= EJ.DBD
        cell["value"] = dbg
        self._cp0_set(C.DEPC, depc)
        self.waiting = False
        self.pc = self.to_pc(self.debug_vector())
        entry.update(epc=depc, bd=in_slot, vector=self.pc, debug_mode=True, dexccode=dexc)
        self.exception_log.append(entry)

    def _take_nmi(self, pc: int, in_slot: bool, entry: dict) -> None:
        """Vol. III "Non Maskable Interrupt (NMI) Exception": ErrorEPC = the
        restart PC (the branch for a delay slot), Status BEV = ERL = NMI = 1,
        SR = TS = 0, the reset vector (a CPS core's reset base); Cause is
        not touched."""
        self.nmi_pending = False
        self.waiting = False
        epc = self._restart_pc(pc, in_slot)
        self.isa_mode = self._handler_isa(False)
        self._cp0_set(C.ERROREPC, epc)
        status = self.status & ~(C.ST_SR | (1 << 21))
        self._cp0_set(C.STATUS, status | C.ST_BEV | C.ST_ERL | C.ST_NMI)
        self.pc = self.to_pc(self.bev_base - 0x200)
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
        epc = self._restart_pc(pc, in_slot)
        self.isa_mode = self._handler_isa(not status & C.ST_BEV)
        self._cp0_set(C.ERROREPC, epc)
        self._cp0_set(C.STATUS, status | C.ST_ERL)
        self.pc = self.to_pc(C.cache_error_vector(
            status, self.cp0[C.EBASE]["value"], self.release,
            cv=(getattr(self.mmu, "sc", False) or getattr(self.mmu, "segments", False))
            and bool(self.cp0_read(16, 5) & (1 << 29)),
            bev_base=self.bev_base))
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
    m.write_dreg(rd(w), m.read_reg(rs(w)) & m.read_reg(rt(w)))
    return None


def _e_or(m: _M, w: int) -> int | None:
    m.write_dreg(rd(w), m.read_reg(rs(w)) | m.read_reg(rt(w)))
    return None


def _e_xor(m: _M, w: int) -> int | None:
    m.write_dreg(rd(w), m.read_reg(rs(w)) ^ m.read_reg(rt(w)))
    return None


def _e_nor(m: _M, w: int) -> int | None:
    m.write_dreg(rd(w), ~(m.read_reg(rs(w)) | m.read_reg(rt(w))))
    return None


def _e_slt(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), int(m.sgn(m.read_reg(rs(w))) < m.sgn(m.read_reg(rt(w)))))
    return None


def _e_sltu(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), int(m.read_reg(rs(w)) < m.read_reg(rt(w))))
    return None


def _e_movz(m: _M, w: int) -> int | None:
    if m.read_reg(rt(w)) == 0:
        m.write_dreg(rd(w), m.read_reg(rs(w)))
    return None


def _e_movn(m: _M, w: int) -> int | None:
    if m.read_reg(rt(w)) != 0:
        m.write_dreg(rd(w), m.read_reg(rs(w)))
    return None


# -- Shifts / rotates -------------------------------------------------------


def _rotr(x: int, n: int) -> int:
    x &= M32
    n &= 31
    return ((x >> n) | (x << (32 - n))) & M32 if n else x & M32


def _e_sll(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), m.read_reg(rt(w)) << sa(w))
    return None


def _e_srl(m: _M, w: int) -> int | None:
    m.write_reg(rd(w), (m.read_reg(rt(w)) & M32) >> sa(w))
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
    m.write_reg(rd(w), (m.read_reg(rt(w)) & M32) >> (m.read_reg(rs(w)) & 31))
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
    m.write_reg(rd(w), 32 - (m.read_reg(rs(w)) & M32).bit_length())
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
    return ((m.hi & M32) << 32) | (m.lo & M32)


def _set_hi_lo(m: _M, value: int) -> None:
    """HI, LO = the 64-bit result's halves (MIPS64: each sign-extended)."""
    value &= M64
    m.hi, m.lo = m._sxw(value >> 32), m._sxw(value)


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
    m.write_dreg(rd(w), m.hi)
    return None


def _e_mflo(m: _M, w: int) -> int | None:
    m.write_dreg(rd(w), m.lo)
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
        m.hi = m.lo = m._sxw(M32)
        return None
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    m.lo, m.hi = m._sxw(q), m._sxw(a - q * b)
    return None


def _e_divu(m: _M, w: int) -> int | None:
    a, b = m.read_reg(rs(w)) & M32, m.read_reg(rt(w)) & M32
    if b == 0:
        m.hi = m.lo = m._sxw(M32)
    else:
        m.lo, m.hi = m._sxw(a // b), m._sxw(a % b)
    return None


# -- I-form ALU -------------------------------------------------------------


def _e_addi(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), _ov(_s32(m.read_reg(rs(w))) + _s16(imm16(w))))
    return None


def _e_addiu(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), m.read_reg(rs(w)) + _s16(imm16(w)))
    return None


def _e_andi(m: _M, w: int) -> int | None:
    m.write_dreg(rt(w), m.read_reg(rs(w)) & imm16(w))
    return None


def _e_ori(m: _M, w: int) -> int | None:
    m.write_dreg(rt(w), m.read_reg(rs(w)) | imm16(w))
    return None


def _e_xori(m: _M, w: int) -> int | None:
    m.write_dreg(rt(w), m.read_reg(rs(w)) ^ imm16(w))
    return None


def _e_slti(m: _M, w: int) -> int | None:
    m.write_reg(rt(w), int(m.sgn(m.read_reg(rs(w))) < _s16(imm16(w))))
    return None


def _e_sltiu(m: _M, w: int) -> int | None:
    """SLTIU: unsigned compare against the *sign-extended* immediate."""
    m.write_reg(rt(w), int(m.read_reg(rs(w)) < (_s16(imm16(w)) & m.amask)))
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
        m.write_dreg(31, m.pc + 8)
    if cond:
        return _br_target(m.pc, w, m.amask)
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
    return m.sgn(m.read_reg(rs(w))) >= 0


def _gtz(m: _M, w: int) -> bool:
    return m.sgn(m.read_reg(rs(w))) > 0


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
    return _j_target(m.pc, w, m.amask)


def _e_jal(m: _M, w: int) -> int | None:
    m.write_dreg(31, m.pc + 8)
    return _j_target(m.pc, w, m.amask)


def _e_jr(m: _M, w: int) -> int | None:
    return m.read_reg(rs(w))


def _e_jalr(m: _M, w: int) -> int | None:
    target = m.read_reg(rs(w))
    m.write_dreg(rd(w), m.pc + 8)
    return target


_e_jr_hb = _e_jr          # + hazard barrier: no hazards in this model
_e_jalr_hb = _e_jalr


def _e_jalx(m: _M, w: int) -> int | None:
    """JALX (Vol. IV-a "JALX target, MIPS32 with MIPS16e"): JAL to the
    other ISA.  Without MIPS16e / microMIPS: Reserved Instruction."""
    if m.compressed is None:
        raise _Arch(C.RI)
    m.write_dreg(31, m.pc + 8)
    return _j_target(m.pc, w, m.amask) | 1


# -- Loads / stores ---------------------------------------------------------


def _addr_i(m: _M, w: int) -> int:
    return (m.read_reg(rs(w)) + _s16(imm16(w))) & m.amask


def _addr_e(m: _M, w: int) -> int:
    return (m.read_reg(rs(w)) + off9(w)) & m.amask


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
    return (addr, 4 - k) if m.big_endian else (addr & ~3, k + 1)


def _span_right(m: _M, addr: int) -> tuple[int, int]:
    """The bytes LWR / SWR access."""
    k = addr & 3
    return (addr & ~3, k + 1) if m.big_endian else (addr, 4 - k)


def _lw_left(m: _M, w: int, addr: int) -> None:
    addr = m.data_address(addr, 1, store=False, span=_span_left(m, addr))
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
    addr = m.data_address(addr, 1, store=False, span=_span_right(m, addr))
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
    addr = m.data_address(addr, 1, store=True, span=_span_left(m, addr))
    k, base, r = addr & 3, addr & ~3, m.read_reg(rt(w))
    if m.big_endian:                    # MS (4-k) bytes -> addr..base+3
        n = 4 - k
        m.store(addr, r >> (8 * (4 - n)), n, gather=False)
    else:                               # MS (k+1) bytes -> base..addr
        n = k + 1
        m.store(base, r >> (8 * (4 - n)), n, gather=False)


def _sw_right(m: _M, w: int, addr: int) -> None:
    addr = m.data_address(addr, 1, store=True, span=_span_right(m, addr))
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
    if m.itc is not None and m.itc.enabled and m.itc.covers(m._xlate_peek(addr) & M32):
        # gating storage: success is the ITC's answer, whatever the LLbit
        # (MD00378 7.2; the E/F try view fails on a full cell, A.2)
        ok = bool(m.store(addr, m.read_reg(rt(w)), 4, gather=False, atomic=True))
    elif ok:                                   # atomic: never waits in the FSB
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
    m.mtc0(reg, sel, m.read_reg(rt(w)))
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


def _e_dvp(m: _M, w: int) -> int | None:
    """DVP rt (Release 6, Vol. II 6.06): GPR[rt] <- VPControl (sign-
    extended); the core's other VPs stop until this VP's EVP (emips.vp).
    Without Config5.VP a NOP."""
    core = m.vpcore
    if core is not None:
        m.write_reg(rt(w), core.dvp(m))
    return None


def _e_evp(m: _M, w: int) -> int | None:
    """EVP rt (Release 6, Vol. II 6.06): GPR[rt] <- VPControl; the other
    VPs go on.  Without Config5.VP a NOP."""
    core = m.vpcore
    if core is not None:
        m.write_reg(rt(w), core.evp(m))
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
    m.write_dreg(rd(w), m.shadow[m.srs_field(C.SRS_PSS)][r] if r else 0)
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
    # MIPS64: UserLocal is 64 bits wide when 64-bit operations are enabled
    # (sign_extend_if_32bit_op), the others sign-extended
    (m.write_dreg if num == 29 and m.ops64() else m.write_reg)(rt(w), values[num])
    return None


# -- Traps / BREAK / SYSCALL ------------------------------------------------


def _trap(cond: bool, code: int = 0) -> int | None:
    if cond:
        raise _Arch(C.TR, info={"code": code})
    return None


def _tcode(w: int) -> int:
    return (w >> 6) & 0x3FF


def _simm(m: _M, w: int) -> int:
    return _s16(imm16(w)) & m.amask


def _e_teq(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) == m.read_reg(rt(w)), _tcode(w))


def _e_tne(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) != m.read_reg(rt(w)), _tcode(w))


def _e_tlt(m: _M, w: int) -> int | None:
    return _trap(m.sgn(m.read_reg(rs(w))) < m.sgn(m.read_reg(rt(w))), _tcode(w))


def _e_tltu(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) < m.read_reg(rt(w)), _tcode(w))


def _e_tge(m: _M, w: int) -> int | None:
    return _trap(m.sgn(m.read_reg(rs(w))) >= m.sgn(m.read_reg(rt(w))), _tcode(w))


def _e_tgeu(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) >= m.read_reg(rt(w)), _tcode(w))


def _e_teqi(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) == _simm(m, w))


def _e_tnei(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) != _simm(m, w))


def _e_tgei(m: _M, w: int) -> int | None:
    return _trap(m.sgn(m.read_reg(rs(w))) >= _s16(imm16(w)))


def _e_tlti(m: _M, w: int) -> int | None:
    return _trap(m.sgn(m.read_reg(rs(w))) < _s16(imm16(w)))


def _e_tgeiu(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) >= _simm(m, w))


def _e_tltiu(m: _M, w: int) -> int | None:
    return _trap(m.read_reg(rs(w)) < _simm(m, w))


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
        m.write_dreg(rd(w), m.read_reg(rs(w)))
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


def fcr_read(m: _M, n: int) -> int | None:
    """FPU control register ``n`` as CFC1 reads it: FIR (0), FCCR (25),
    FEXR (26), FENR (28), FCSR (31); None for any other number."""
    fcsr = m.fcsr
    views = {
        0: m.fir,
        25: ((fcsr >> 24) & 0xFE) | ((fcsr >> 23) & 1),
        26: fcsr & 0x0003F07C,
        28: (fcsr & 0xF80) | ((fcsr >> 22) & 4) | (fcsr & 3),
        31: fcsr,
    }
    return views.get(n)


def fcr_write(m: _M, n: int, v: int) -> bool | None:
    """CTC1 to FPU control register ``n`` (25, 26, 28, 31): None for any
    other number, else whether an enabled Cause bit is now set (the FPE
    CTC1 then takes)."""
    fcsr = m.fcsr
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
        return None
    m.fcsr = fcsr
    return bool((fcsr >> 12) & 0x20 or (fcsr >> 12) & (fcsr >> 7) & 0x1F)


def _e_cfc1(m: _M, w: int) -> int | None:
    """CFC1 rt, fs: FIR (0), FCCR (25), FEXR (26), FENR (28), FCSR (31)."""
    value = fcr_read(m, _fs(w))
    if value is None:
        raise _Arch(C.RI)
    m.write_reg(rt(w), value)
    return None


def _e_ctc1(m: _M, w: int) -> int | None:
    """CTC1 rt, fs (25, 26, 28, 31).  A write that leaves an enabled Cause
    bit set traps (FPE) *after* the write (Vol. II-A CTC1)."""
    trap = fcr_write(m, _fs(w), m.read_reg(rt(w)))
    if trap is None:
        raise _Arch(C.RI)
    if trap:
        raise _Arch(C.FPE)
    return None


def _fp_load(size: int, indexed: bool = False, unaligned: bool = False) -> Executor:
    def ex(m: _M, w: int) -> int | None:
        base = m.read_reg(rs(w))
        addr = (base + m.read_reg(rt(w))) if indexed else (base + _s16(imm16(w)))
        addr &= m.amask
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
        addr &= m.amask
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
        m.write_dreg(31, m.pc + 4)
    if cond:
        m._redirect = target & m.amask
    elif conditional:
        m._forbidden = True
    return None


def _off16(m: _M, w: int) -> int:
    return m.pc + 4 + (_s16(imm16(w)) << 2)


def _off21(m: _M, w: int) -> int:
    return m.pc + 4 + (_sx(w & 0x1FFFFF, 21) << 2)


def _off26(m: _M, w: int) -> int:
    return m.pc + 4 + (_sx(w & 0x3FFFFFF, 26) << 2)


def _ovf(m: _M, a: int, b: int) -> bool:
    """BOVC / BNVC: the 32-bit sum overflows — or (MIPS64) an operand is
    not a sign-extended word (``input_overflow``, Vol. II-A64 6.06)."""
    r = _s32(a) + _s32(b)
    return not -(1 << 31) <= r < (1 << 31) or m.wide and (
        _s32(a) & M64 != a or _s32(b) & M64 != b)


def _r6_branches() -> dict[str, Executor]:
    R = lambda m, w, f: m.read_reg(f(w))                    # noqa: E731
    S = lambda m, w, f: m.sgn(m.read_reg(f(w)))             # noqa: E731
    two = {
        "BEQC": lambda m, w: R(m, w, rs) == R(m, w, rt),
        "BNEC": lambda m, w: R(m, w, rs) != R(m, w, rt),
        "BGEC": lambda m, w: S(m, w, rs) >= S(m, w, rt),
        "BLTC": lambda m, w: S(m, w, rs) < S(m, w, rt),
        "BGEUC": lambda m, w: R(m, w, rs) >= R(m, w, rt),
        "BLTUC": lambda m, w: R(m, w, rs) < R(m, w, rt),
        "BOVC": lambda m, w: _ovf(m, R(m, w, rs), R(m, w, rt)),
        "BNVC": lambda m, w: not _ovf(m, R(m, w, rs), R(m, w, rt)),
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
            m, c(m.sgn(m.read_reg(rt(w)))), _off16(m, w), link=l))(cond, link)
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
    m.write_dreg(rs(w), m.pc + (_s16(imm16(w)) << 16))
    return None


def _e_aluipc(m: _M, w: int) -> int | None:
    m.write_dreg(rs(w), (m.pc + (_s16(imm16(w)) << 16)) & ~0xFFFF)
    return None


def _pc19(m: _M, w: int) -> int:
    return (m.pc + (_sx(w & 0x7FFFF, 19) << 2)) & m.amask


def _e_addiupc(m: _M, w: int) -> int | None:
    m.write_dreg(rs(w), _pc19(m, w))
    return None


def _e_lwpc(m: _M, w: int) -> int | None:
    m.write_reg(rs(w), m.load(m.data_address(_pc19(m, w), 4, store=False), 4, "pcload"))
    return None


def _e_align(m: _M, w: int) -> int | None:
    bp = (w >> 6) & 3
    a, b = m.read_reg(rt(w)), m.read_reg(rs(w))
    m.write_reg(rd(w), a if bp == 0 else (a << (8 * bp)) | ((b & M32) >> (32 - 8 * bp)))
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
    m.write_dreg(rd(w), m.read_reg(rs(w)) if m.read_reg(rt(w)) == 0 else 0)
    return None


def _e_selnez(m: _M, w: int) -> int | None:
    m.write_dreg(rd(w), m.read_reg(rs(w)) if m.read_reg(rt(w)) != 0 else 0)
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
            return 0                       # UNPREDICTABLE (DEVIATIONS)
        q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
        return a - q * b if mod else q

    return {
        "MUL@R6": mk(lambda a, b: _s32(a) * _s32(b)),
        "MUH": mk(lambda a, b: (_s32(a) * _s32(b)) >> 32),
        "MULU": mk(lambda a, b: (a & M32) * (b & M32)),
        "MUHU": mk(lambda a, b: ((a & M32) * (b & M32)) >> 32),
        "DIV@R6": mk(lambda a, b: sdiv(a, b, False)),
        "MOD": mk(lambda a, b: sdiv(a, b, True)),
        "DIVU@R6": mk(lambda a, b: (a & M32) // (b & M32) if b & M32 else 0),
        "MODU": mk(lambda a, b: (a & M32) % (b & M32) if b & M32 else 0),
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
    m.prefetch(rd(w), (m.read_reg(rs(w)) + m.read_reg(rt(w))) & m.amask)
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

# -- MIPS16e (emips.mips16): the instructions without a MIPS32 twin ---------
# (the others run as their MIPS32 expansion).  m._m16 is the instruction,
# m.pc its address, m._m16_base the base of a PC-relative one (Table 3.14).


def _e16_addiupc(m: _M, w: int) -> int | None:
    f = m._m16.f                                # (a word, sign-extended on MIPS64)
    m.write_reg(f["rx"], (m._m16_base & ~3) + f["imm"])
    return None


def _e16_lwpc(m: _M, w: int) -> int | None:
    f = m._m16.f
    addr = ((m._m16_base & ~3) + f["imm"]) & m.amask
    m.write_reg(f["rx"], m.load(m.data_address(addr, 4, store=False), 4, "pcload"))
    return None


def _e16_ldpc(m: _M, w: int) -> int | None:
    """LD ry, offset(pc) (MD00077): the base with its 3 low bits cleared;
    extended, an unaligned sum is an Address Error."""
    f = m._m16.f
    addr = ((m._m16_base & ~7) + f["imm"]) & m.amask
    m.write_dreg(f["ry"], m.load(m.data_address(addr, 8, store=False), 8, "pcload"))
    return None


def _e16_daddiupc(m: _M, w: int) -> int | None:
    f = m._m16.f                                # DADDIU ry, pc, imm: all 64 bits
    m.write_dreg(f["ry"], ((m._m16_base & ~3) + f["imm"]) & m.amask)
    return None


def _e16_sew(m: _M, w: int) -> int | None:
    rx = m._m16.f["rx"]                         # SEW (MD00077): rx 31..0 sign-extended
    m.write_reg(rx, m.read_reg(rx))
    return None


def _e16_zew(m: _M, w: int) -> int | None:
    rx = m._m16.f["rx"]                         # ZEW (MD00077): 0^32 || rx 31..0
    m.write_dreg(rx, m.read_reg(rx) & M32)
    return None


def _e16_branch(cond: Callable[[_M, dict], bool]) -> Executor:
    """B / BEQZ / BNEZ / BTEQZ / BTNEZ: no delay slot (Vol. IV-a 3.13)."""
    def run(m: _M, w: int) -> int | None:
        i = m._m16
        if cond(m, i.f):
            m._redirect = M16.branch_target(i, m.pc, m.amask) | 1
        return None
    return run


def _e16_jr(reg: int | None) -> Executor:
    def run(m: _M, w: int) -> int | None:
        return m.read_reg(m._m16.f["rx"] if reg is None else reg)
    return run


def _e16_jalr(m: _M, w: int) -> int | None:
    target = m.read_reg(m._m16.f["rx"])
    m.write_dreg(31, ((m.pc + 4) | 1) & m.amask)    # past the jump and its delay slot
    return target


def _e16_jrc(reg: int | None) -> Executor:
    def run(m: _M, w: int) -> int | None:
        m._redirect = m.read_reg(m._m16.f["rx"] if reg is None else reg)
        return None
    return run


def _e16_jalrc(m: _M, w: int) -> int | None:
    target = m.read_reg(m._m16.f["rx"])
    m.write_dreg(31, ((m.pc + 2) | 1) & m.amask)
    m._redirect = target
    return None


def _e16_jal(exchange: bool) -> Executor:
    def run(m: _M, w: int) -> int | None:
        m.write_dreg(31, ((m.pc + 6) | 1) & m.amask)    # past the 32-bit jump and its slot
        target = M16.jump_target(m._m16, m.pc, m.amask)
        return target if exchange else target | 1
    return run


def _m16_frame_regs(f: dict, statics: int) -> list[int]:
    """SAVE / RESTORE's registers from the frame's top down: ra, the extra
    s-registers (GPR 30 highest), s1, s0, the statics GPR 7 down."""
    return ([31] if f["ra"] else []) + list(reversed(M16.XSREGS[:f["xsregs"]])) + \
        ([17] if f["s1"] else []) + ([16] if f["s0"] else []) + [7 - k for k in range(statics)]


def _e16_save(m: _M, w: int) -> int | None:
    """SAVE (Vol. IV-a): the incoming arguments into the caller's area,
    then the registers below sp, then sp -= the frame.  sp changes last:
    a fault restarts the instruction from the start.  On MIPS64 the low
    word of each register (MD00077 SAVE), sp a 64-bit subtraction."""
    f = m._m16.f
    sp = m.read_reg(29)
    args, statics = M16.args_statics(f["aregs"])
    amask = m.amask

    def put(addr: int, reg: int) -> None:
        m.store(m.data_address(addr & amask, 4, store=True), m.read_reg(reg) & M32, 4)
    for k in range(args):
        put(sp + 4 * k, 4 + k)
    at = sp
    for reg in _m16_frame_regs(f, statics):
        at -= 4
        put(at, reg)
    m.write_dreg(29, (sp - f["frame"]) & amask)
    return None


def _e16_restore(m: _M, w: int) -> int | None:
    """RESTORE: the same registers from the frame's top down, then sp +=
    the frame (the arguments stay in the caller's area); on MIPS64 each
    word sign-extended (MD00077 RESTORE: LoadStackWord)."""
    f = m._m16.f
    amask = m.amask
    top = (m.read_reg(29) + f["frame"]) & amask
    _args, statics = M16.args_statics(f["aregs"])
    at = top
    for reg in _m16_frame_regs(f, statics):
        at -= 4
        m.write_reg(reg, m.load(m.data_address(at & amask, 4, store=False), 4))
    m.write_dreg(29, top)
    return None


_M16_EXECUTORS: dict[str, Executor] = {
    "M16.ADDIUPC": _e16_addiupc, "M16.LWPC": _e16_lwpc,
    "M16.B": _e16_branch(lambda m, f: True),
    "M16.BEQZ": _e16_branch(lambda m, f: m.read_reg(f["rx"]) == 0),
    "M16.BNEZ": _e16_branch(lambda m, f: m.read_reg(f["rx"]) != 0),
    "M16.BTEQZ": _e16_branch(lambda m, f: m.read_reg(24) == 0),
    "M16.BTNEZ": _e16_branch(lambda m, f: m.read_reg(24) != 0),
    "M16.JR": _e16_jr(None), "M16.JRRA": _e16_jr(31), "M16.JALR": _e16_jalr,
    "M16.JRC": _e16_jrc(None), "M16.JRCRA": _e16_jrc(31), "M16.JALRC": _e16_jalrc,
    "M16.JAL": _e16_jal(False), "M16.JALX": _e16_jal(True),
    "M16.SAVE": _e16_save, "M16.RESTORE": _e16_restore,
    "M16.LDPC": _e16_ldpc, "M16.DADDIUPC": _e16_daddiupc, "M16.SEW": _e16_sew,
    "M16.ZEW": _e16_zew,
}


# -- microMIPS (emips.micromips): the instructions without a MIPS32 twin ----
# (the others run as their MIPS32 twin).  m._mm is the instruction, m.pc
# its address.  Branch / jump targets keep bit 0 = 1 (microMIPS), the
# links carry it too (MD00582 3.5); offsets count halfwords.  microMIPS64
# (MD00594): the same, with GPR-width addresses, links and compares.


def _emm_branch(cond: Callable[[_M, dict], bool], *, linked: bool = False) -> Executor:
    """BEQ / BNE / B*Z / B*ZAL(S) / BC1F ... / B16 / BEQZ16 / BNEZ16: a
    delay slot of either size, the link past it (5.4 / 5.5)."""
    def run(m: _M, w: int) -> int | None:
        i = m._mm
        taken = cond(m, i.f)
        if linked:
            m.write_dreg(31, MM.link(i, m.pc) & m.amask)
        return MM.branch_target(i, m.pc, m.amask) | 1 if taken else None
    return run


def _emm_compact(cond: Callable[[_M, dict], bool]) -> Executor:
    """BEQZC / BNEZC: no delay slot."""
    def run(m: _M, w: int) -> int | None:
        i = m._mm
        if cond(m, i.f):
            m._redirect = MM.branch_target(i, m.pc, m.amask) | 1
        return None
    return run


def _emm_jalr(m: _M, w: int) -> int | None:
    """JALR / JALRS (.HB) rt, rs; JR = JALR $0 (the link to $0 vanishes)."""
    f = m._mm.f
    target = m.read_reg(f["s"])
    m.write_dreg(f["t"], MM.link(m._mm, m.pc) & m.amask)
    return target


def _emm_jr16(link: bool) -> Executor:
    def run(m: _M, w: int) -> int | None:
        target = m.read_reg(m._mm.f["s"])
        if link:
            m.write_dreg(31, MM.link(m._mm, m.pc) & m.amask)
        return target
    return run


def _emm_jrc(m: _M, w: int) -> int | None:
    m._redirect = m.read_reg(m._mm.f["s"])
    return None


def _emm_jraddiusp(m: _M, w: int) -> int | None:
    """JRADDIUSP: PC = $ra, $sp += the immediate; no delay slot."""
    m._redirect = m.read_reg(31)
    m.write_reg(29, (m.read_reg(29) + m._mm.f["i"]) & M32)
    return None


def _emm_jump(link: bool, exchange: bool = False) -> Executor:
    """J / JAL / JALS (stay in microMIPS) / JALX (to MIPS32)."""
    def run(m: _M, w: int) -> int | None:
        i = m._mm
        if exchange and not m.mips32:
            raise _Arch(C.RI)                   # MD00582 JALX: no MIPS32 to go to
        if link:
            m.write_dreg(31, MM.link(i, m.pc) & m.amask)
        target = MM.jump_target(i, m.pc, m.amask)
        return target if exchange else target | 1
    return run


def _emm_addiupc(m: _M, w: int) -> int | None:
    """ADDIUPC rs: the instruction's address with bits 1:0 cleared + the
    immediate (also in a delay slot, unlike MIPS16e; 5.4)."""
    f = m._mm.f
    m.write_reg(f["t"], ((m.pc & ~3) + f["i"]) & M32)
    return None


def _emm_lwxs(m: _M, w: int) -> int | None:
    """LWXS rd, index(base): base + index * 4."""
    f = m._mm.f
    addr = (m.read_reg(f["b"]) + (m.read_reg(f["t"]) << 2)) & m.amask
    m.write_reg(f["d"], m.load(m.data_address(addr, 4, store=False), 4))
    return None


def _emm_words(m: _M, regs: list[int], base: int, store: bool, size: int = 4) -> None:
    """LWM / SWM / LWP / SWP: consecutive aligned words (5.4; the address
    must be word-aligned, even in Release 6); LDM / SDM / LDP / SDP the
    same with doublewords (MD00594).  Loads read every word before
    writing a register (a fault leaves the registers as they were);
    stores go in order (a fault after the first leaves the first ones
    written: the instruction restarts and writes them again)."""
    m.data_address(base, size, store=store, atomic=True)
    mask = m.amask
    if store:
        for k, reg in enumerate(regs):
            m.store(m.data_address((base + size * k) & mask, size, store=True),
                    m.read_reg(reg), size)
        return
    vals = [m.load(m.data_address((base + size * k) & mask, size, store=False), size)
            for k in range(len(regs))]
    write = m.write_reg if size == 4 else m.write_dreg
    for reg, v in zip(regs, vals):
        write(reg, v)


def _emm_pair(store: bool, size: int = 4) -> Executor:
    def run(m: _M, w: int) -> int | None:
        f = m._mm.f
        if f["t"] == 31 or not store and f["t"] == f["b"]:
            raise _Arch(C.RI)                   # UNPREDICTABLE: QEMU's RI (DEVIATIONS)
        _emm_words(m, [f["t"], f["t"] + 1], (m.read_reg(f["b"]) + f["o"]) & m.amask, store,
                   size)
        return None
    return run


def _emm_multiple(store: bool, size: int = 4) -> Executor:
    def run(m: _M, w: int) -> int | None:
        i = m._mm
        base = 29 if i.form in ("LWM16", "SWM16") else i.f["b"]
        _emm_words(m, MM.reglist(i), (m.read_reg(base) + i.f["o"]) & m.amask, store, size)
        return None
    return run


def _emm_movep(m: _M, w: int) -> int | None:
    """MOVEP rd, re, rs, rt: both sources read first."""
    f = m._mm.f
    rd, re = MM.MOVEP_DEST[f["e"]]
    a, b = m.read_reg(f["s"]), m.read_reg(f["t"])
    m.write_dreg(rd, a)
    m.write_dreg(re, b)
    return None


# -- microMIPS Release 6 (MD00582 6.05): compact branches and jumps only;
# the target is the instruction after the branch + the offset (halfwords),
# the link that address with bit 0 set (QEMU's mips32r6 agrees).

def _emm6(cond: Callable[[_M, dict], bool], *, link: bool = False) -> Executor:
    """The link first, then the condition (6.05 B*ZALC: with rt = 31 the
    new link is compared)."""
    def run(m: _M, w: int) -> int | None:
        i = m._mm
        if link:
            m.write_dreg(31, ((m.pc + i.size) | 1) & m.amask)
        taken = cond(m, i.f)
        if taken:
            m._redirect = MM.branch_target(i, m.pc, m.amask) | 1
        return None
    return run


def _emm6_jump(link: str | None, offset: bool = False) -> Executor:
    """JRC16 / JALRC16 / JALRC (.HB) / JIC / JIALC: to GPR (+ the offset),
    its bit 0 the ISA mode; ``link`` the field of the link register."""
    def run(m: _M, w: int) -> int | None:
        i = m._mm
        f = i.f
        target = m.read_reg(f["t"] if offset else f["s"])
        if offset:
            target = (target + f["i"]) & m.amask
        if link is not None:
            m.write_dreg(31 if link == "ra" else f[link], ((m.pc + i.size) | 1) & m.amask)
        m._redirect = target
        return None
    return run


def _emm6_bc1(nonzero: bool) -> Executor:
    def run(m: _M, w: int) -> int | None:
        if not m.status & C.ST_CU1:
            raise _Arch(C.CPU, ce=1)
        i = m._mm
        if bool(m.fpr[i.f["T"]] & 1) == nonzero:
            m._redirect = MM.branch_target(i, m.pc, m.amask) | 1
        return None
    return run


def _emm6_lwpc(m: _M, w: int) -> int | None:
    """LWPC rt: the word at (PC with bits 1:0 cleared) + the offset."""
    f = m._mm.f
    addr = ((m.pc & ~3) + f["o"]) & m.amask
    m.write_reg(f["t"], m.load(m.data_address(addr, 4, store=False), 4, "pcload"))
    return None


def _emm6_lwupc(m: _M, w: int) -> int | None:
    """LWUPC rt (microMIPS64 Release 6): the word at (PC with bits 1:0
    cleared, as LWPC: DEVIATIONS "microMIPS64") + the offset, zero-
    extended."""
    f = m._mm.f
    addr = ((m.pc & ~3) + f["o"]) & m.amask
    m.write_dreg(f["t"], m.load(m.data_address(addr, 4, store=False), 4, "pcload"))
    return None


def _sg(m: _M, f: dict, k: str) -> int:
    return m.sgn(m.read_reg(f[k]))


_MM6_EXECUTORS: dict[str, Executor] = {
    "MM.BC16": _emm6(lambda m, f: True), "MM.BC": _emm6(lambda m, f: True),
    "MM.BALC": _emm6(lambda m, f: True, link=True),
    "MM.BEQZC16": _emm6(lambda m, f: m.read_reg(f["s"]) == 0),
    "MM.BNEZC16": _emm6(lambda m, f: m.read_reg(f["s"]) != 0),
    "MM.BEQC": _emm6(lambda m, f: m.read_reg(f["s"]) == m.read_reg(f["t"])),
    "MM.BNEC": _emm6(lambda m, f: m.read_reg(f["s"]) != m.read_reg(f["t"])),
    "MM.BLTC": _emm6(lambda m, f: _sg(m, f, "s") < _sg(m, f, "t")),
    "MM.BGEC": _emm6(lambda m, f: _sg(m, f, "s") >= _sg(m, f, "t")),
    "MM.BLTUC": _emm6(lambda m, f: m.read_reg(f["s"]) < m.read_reg(f["t"])),
    "MM.BGEUC": _emm6(lambda m, f: m.read_reg(f["s"]) >= m.read_reg(f["t"])),
    "MM.BOVC": _emm6(lambda m, f: _ovf(m, m.read_reg(f["s"]), m.read_reg(f["t"]))),
    "MM.BNVC": _emm6(lambda m, f: not _ovf(m, m.read_reg(f["s"]), m.read_reg(f["t"]))),
    "MM.BLTZC": _emm6(lambda m, f: _sg(m, f, "t") < 0),
    "MM.BLEZC": _emm6(lambda m, f: _sg(m, f, "t") <= 0),
    "MM.BGEZC": _emm6(lambda m, f: _sg(m, f, "t") >= 0),
    "MM.BGTZC": _emm6(lambda m, f: _sg(m, f, "t") > 0),
    "MM.BEQZALC": _emm6(lambda m, f: m.read_reg(f["t"]) == 0, link=True),
    "MM.BNEZALC": _emm6(lambda m, f: m.read_reg(f["t"]) != 0, link=True),
    "MM.BLTZALC": _emm6(lambda m, f: _sg(m, f, "t") < 0, link=True),
    "MM.BLEZALC": _emm6(lambda m, f: _sg(m, f, "t") <= 0, link=True),
    "MM.BGEZALC": _emm6(lambda m, f: _sg(m, f, "t") >= 0, link=True),
    "MM.BGTZALC": _emm6(lambda m, f: _sg(m, f, "t") > 0, link=True),
    "MM.BC1EQZC": _emm6_bc1(False), "MM.BC1NEZC": _emm6_bc1(True),
    "MM.JRC16": _emm6_jump(None), "MM.JALRC16": _emm6_jump("ra"),
    "MM.JALRC": _emm6_jump("t"), "MM.JALRC.HB": _emm6_jump("t"),
    "MM.JIC": _emm6_jump(None, offset=True), "MM.JIALC": _emm6_jump("ra", offset=True),
    "MM.JRCADDIUSP": _emm_jraddiusp, "MM.LWPC": _emm6_lwpc, "MM.LWUPC": _emm6_lwupc,
}

#: UNPREDICTABLE in a delay slot (MD00582 5.4): Reserved Instruction, as QEMU
_MM_NO_SLOT = frozenset({"MM.LWP", "MM.SWP", "MM.LWM16", "MM.SWM16", "MM.LWM32", "MM.SWM32",
                         "MM.LDP", "MM.SDP", "MM.LDM", "MM.SDM"})


_MM_EXECUTORS: dict[str, Executor] = {
    "MM.B16": _emm_branch(lambda m, f: True),
    "MM.BEQZ16": _emm_branch(lambda m, f: m.read_reg(f["s"]) == 0),
    "MM.BNEZ16": _emm_branch(lambda m, f: m.read_reg(f["s"]) != 0),
    "MM.BEQ": _emm_branch(lambda m, f: m.read_reg(f["s"]) == m.read_reg(f["t"])),
    "MM.BPOSGE32": _emm_branch(lambda m, f: (m.dspctl & 0x3F) >= 32),   # MD00764
    "MM.BNE": _emm_branch(lambda m, f: m.read_reg(f["s"]) != m.read_reg(f["t"])),
    "MM.BLTZ": _emm_branch(lambda m, f: _sg(m, f, "s") < 0),
    "MM.BGEZ": _emm_branch(lambda m, f: _sg(m, f, "s") >= 0),
    "MM.BLEZ": _emm_branch(lambda m, f: _sg(m, f, "s") <= 0),
    "MM.BGTZ": _emm_branch(lambda m, f: _sg(m, f, "s") > 0),
    "MM.BLTZAL": _emm_branch(lambda m, f: _sg(m, f, "s") < 0, linked=True),
    "MM.BGEZAL": _emm_branch(lambda m, f: _sg(m, f, "s") >= 0, linked=True),
    "MM.BLTZALS": _emm_branch(lambda m, f: _sg(m, f, "s") < 0, linked=True),
    "MM.BGEZALS": _emm_branch(lambda m, f: _sg(m, f, "s") >= 0, linked=True),
    "MM.BC1F": _emm_branch(lambda m, f: _fcc(m, f["c"]) == 0),
    "MM.BC1T": _emm_branch(lambda m, f: _fcc(m, f["c"]) == 1),
    "MM.BEQZC": _emm_compact(lambda m, f: m.read_reg(f["s"]) == 0),
    # MSA (MD00870 / MD00872): no delay slot, a halfword offset
    **{f"MM.{name}": _emm_compact((lambda c: lambda m, f: c(_V.vec(m, f["V"])))(cond))
       for name, cond in _V.CONDITIONS.items()},
    "MM.BNEZC": _emm_compact(lambda m, f: m.read_reg(f["s"]) != 0),
    "MM.JALR": _emm_jalr, "MM.JALR.HB": _emm_jalr, "MM.JALRS": _emm_jalr,
    "MM.JALRS.HB": _emm_jalr,
    "MM.JR16": _emm_jr16(False), "MM.JALR16": _emm_jr16(True), "MM.JALRS16": _emm_jr16(True),
    "MM.JRC": _emm_jrc, "MM.JRADDIUSP": _emm_jraddiusp,
    "MM.J": _emm_jump(False), "MM.JAL": _emm_jump(True), "MM.JALS": _emm_jump(True),
    "MM.JALX": _emm_jump(True, exchange=True),
    "MM.ADDIUPC": _emm_addiupc, "MM.LWXS": _emm_lwxs, "MM.MOVEP": _emm_movep,
    "MM.LWP": _emm_pair(False), "MM.SWP": _emm_pair(True),
    "MM.LWM16": _emm_multiple(False), "MM.SWM16": _emm_multiple(True),
    "MM.LWM32": _emm_multiple(False), "MM.SWM32": _emm_multiple(True),
    "MM.LDP": _emm_pair(False, 8), "MM.SDP": _emm_pair(True, 8),     # microMIPS64
    "MM.LDM": _emm_multiple(False, 8), "MM.SDM": _emm_multiple(True, 8),
    **_MM6_EXECUTORS,
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
    "DVP": _e_dvp, "EVP": _e_evp,
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

_EXECUTORS.update(_M16_EXECUTORS)
_EXECUTORS.update(_MM_EXECUTORS)
_EXECUTORS.update(_fpu_executors())
_EXECUTORS.update(_r6_executors())
_EXECUTORS.update(_r6_fpu())
_EXECUTORS.update(DSP.EXECUTORS)
_EXECUTORS.update(MT.EXECUTORS)

# MIPS64 (PLAN M22): the 64-bit machine and the doubleword instructions
from . import sim64 as _S64  # noqa: E402  (it builds on everything above)
_EXECUTORS.update(_S64.EXECUTORS)
from . import simnano as _SN  # noqa: E402  (nanoMIPS: emips.simnano)
_EXECUTORS.update(_SN.EXECUTORS)
_EXECUTORS.update(_V.EXECUTORS)                 # MSA (PLAN M24)
_EXECUTORS["LSA@MSA"] = _EXECUTORS["LSA"]       # MSA's LSA / DLSA before Release 6
_EXECUTORS["DLSA@MSA"] = _EXECUTORS["DLSA"]
PRIVILEGED_NAMES = PRIVILEGED_NAMES | _SN.PRIVILEGED
EVA_NAMES = EVA_NAMES | _SN.EVA

#: Row names with an executor (dispatched by name, never by ``funct``).
#: the MIPS32 rows with an executor (MIPS16e's own M16.* executors are not rows)
EXEC_NAMES: frozenset[str] = frozenset(n for n in _EXECUTORS
                                      if not n.startswith(("M16.", "MM.", "NM.")))
#: MSA's rows without an executor yet (PLAN M24 phase 2 adds them; no core
#: decodes MSA before Machine(msa=) exists)
DEFER_NAMES = DEFER_NAMES | frozenset(
    e["name"] for e in ISA.for_release("R6", arch="mips64", msa=True).instructions
    if e.get("ase") == "msa" and e["name"] not in _EXECUTORS)
