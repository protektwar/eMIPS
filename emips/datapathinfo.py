"""What each part of the 5-stage datapath does — inside, and in the cycle
on the screen (the GUI's Datapath: click a part).  Qt-free.

The parts carry the numbers of the guide's figure (Phase 3, 3.4): 1 PC
mux, 2 PC, 3 +4, 4 instruction memory, 5 control, 6 registers, 7
sign-extend, 8 branch target, 9 comparator, 10 hazard detection, 11
forwarding muxes A and B, 12 ALUSrc, 13 ALU, 14 RegDst, 15 forwarding
unit, 16 data memory, 17 the write-back mux, 18 the FPU / MSA unit (the
coprocessor beside the ALU: the textbook figure has none); the level
below the caches has no number.

"What happens now" comes from the per-cycle trace only
(:meth:`Session.stage_details` / :attr:`Session.last_view`, the cycle's
events) and the decoded instruction word (its bit fields, the control
signals) — nothing is executed or recomputed here.

A nanoMIPS core (PLAN M23 phase 4e) runs on the same five stages with
2-, 4- and 6-byte instructions: :data:`NANO_PARTS` says what differs
inside (the +2 / 4 / 6 adder, the halfword fetch, compact branches from
the address of the next instruction, the immediates the format's decode
pseudocode assembles), and the descriptions read the instruction's
fields from its encoding diagram (:func:`emips.tracenano.fields`) and its
registers from :func:`emips.simnano.uses`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .syntax import GPR_NAMES, GPR_NAMES_N64, GPR_NAMES_P32, gpr_name

STAGES = ("IF", "ID", "EX", "MEM", "WB")
_GPRS = {f"${n}" for names in (GPR_NAMES, GPR_NAMES_N64, GPR_NAMES_P32) for n in names}
#: hex digits of a register value in the descriptions (MIPS64: 16; set by
#: part_info from the session)
_DIGITS = [8]


@dataclass
class Part:
    key: str
    number: int | None
    title: str
    stage: str
    inside: str                  # how the part works (the same every cycle)


#: the parts, in the order of the guide's figure
PARTS: dict[str, Part] = {p.key: p for p in (
    Part("pcmux", 1, "PC mux (PCSrc)", "IF",
         "A 2-input multiplexer in front of the PC.  Its inputs are PC + 4 (from the "
         "+4 adder) and the branch / jump target computed in ID; its select line, "
         "PCSrc, is 1 when the branch in ID is taken or the instruction there is a "
         "jump.  Its output becomes the PC at the end of the cycle.  (An exception "
         "or an eret loads the PC another way: the handler's vector / EPC.)"),
    Part("pc", 2, "PC (program counter)", "IF",
         "A 32-bit register holding the address of the instruction being fetched.  "
         "It is written at the end of every cycle with what the PC mux chooses — "
         "unless the hazard detection unit stalls the pipeline (PCWrite = 0): then "
         "the same instruction is fetched again."),
    Part("add4", 3, "+4 adder", "IF",
         "A 32-bit adder with one input fixed at 4: instructions are 4 bytes, so "
         "PC + 4 is the next one in memory.  Its result goes to the PC mux and, "
         "through IF/ID, to ID (the branch target and the link address are "
         "computed from it)."),
    Part("imem", 4, "Instruction memory", "IF",
         "Read-only for the pipeline: the address is the PC, the output the 32-bit "
         "instruction word, latched into IF/ID.  With caches it is the I-cache: a "
         "hit delivers in the cycle, a miss fills the line from the level below "
         "first (IF waits).  With a TLB the address is translated first (the "
         "ITLB)."),
    Part("control", 5, "Main control unit", "ID",
         "Combinational logic: from the opcode (bits 31..26) and, for R-type, the "
         "funct field (5..0), it makes the control signals.  They travel with the "
         "instruction through ID/EX, EX/MEM and MEM/WB, each stage using its own: "
         "RegDst and ALUSrc / ALUOp in EX, MemRead / MemWrite in MEM, MemtoReg and "
         "RegWrite in WB; Branch and Jump in ID (branches are decided there)."),
    Part("regs", 6, "Register file", "ID / WB",
         "32 registers of 32 bits ($zero always reads 0) with two read ports and "
         "one write port.  In ID it reads rs and rt (5-bit numbers from the "
         "instruction); WB writes one register (number and data from MEM/WB) in the "
         "first half of the cycle, so an instruction in ID reads a value WB writes "
         "in the same cycle."),
    Part("sext", 7, "Sign-extend", "ID",
         "Makes the 16-bit immediate 32 bits wide: bit 15 is copied into bits "
         "31..16 (sign extension: -1 stays -1).  The logical immediates (andi, "
         "ori, xori) are zero-extended instead; lui puts it in the upper half.  "
         "The result goes to ID/EX (operand b) and, shifted left by 2, to the "
         "branch-target adder."),
    Part("btarget", 8, "Branch-target adder", "ID",
         "An adder: PC + 4 + (sign-extended offset << 2) — the offset counts "
         "instructions, so it is multiplied by 4.  For j / jal the target is "
         "instead (PC + 4)[31:28] with the 26-bit index << 2 (a 256 MB region).  "
         "In this pipeline branches are decided in ID, so the target is ready "
         "for the PC mux one cycle after the fetch: one delay slot."),
    Part("cmp", 9, "Branch comparator (=)", "ID",
         "Compares the two registers read in ID (beq / bne: equal?; bgez, bltz ...: "
         "the sign of rs).  Its output, with Branch, drives PCSrc.  An operand still "
         "being computed in EX/MEM can be forwarded to it; one still in EX "
         "makes the hazard unit stall."),
    Part("hazard", 10, "Hazard detection unit", "ID",
         "Compares the registers the instruction in ID reads with the destinations "
         "of the instructions ahead: a load in EX whose result ID needs (load-use), "
         "a branch needing a result still in EX (or a load in EX / MEM), a busy "
         "multiply / divide unit, no forwarding at all.  Then it stalls: PC and "
         "IF/ID keep their values and a bubble (all control signals 0) goes into "
         "EX."),
    Part("fwda", 11, "Forwarding mux A", "EX",
         "A 3-input multiplexer for ALU operand a: the value read from the register "
         "file in ID (from ID/EX), the result of the instruction one ahead "
         "(EX/MEM), or of the one two ahead (MEM/WB).  The forwarding unit "
         "chooses: a result not yet written back is used directly — no stall."),
    Part("fwdb", 11, "Forwarding mux B", "EX",
         "A 3-input multiplexer for the second register operand (rt): the value "
         "read from the register file in ID (from ID/EX), or the result forwarded "
         "from EX/MEM or MEM/WB, as the forwarding unit chooses.  Its output goes "
         "to the ALUSrc mux and, for a store, is the data written to memory."),
    Part("alusrc", 12, "ALUSrc mux", "EX",
         "Chooses ALU operand b: the register value (from mux B) for R-type and "
         "branches, or the extended immediate for I-type arithmetic, loads and "
         "stores (the address offset).  Control signal ALUSrc."),
    Part("alu", 13, "ALU", "EX",
         "The arithmetic-logic unit: adder / subtractor, and, or, xor, nor, "
         "set-on-less-than, the shifter.  The ALU control picks the function from "
         "ALUOp (the instruction class) and the funct field.  Outputs: the result "
         "(to EX/MEM: a register value, or a memory address) and Zero; add / sub "
         "also detect signed overflow (a trap for add, addi, sub)."),
    Part("regdst", 14, "RegDst mux", "EX",
         "Chooses the number of the register the instruction will write: rt "
         "(I-type: bits 20..16), rd (R-type: bits 15..11), 31 ($ra: jal, bal, "
         "...) or, on an MSA core, bits 10..6 (COPY_S / COPY_U / CFCMSA put the GPR "
         "they write there).  The number travels to WB with the result; the hazard and "
         "forwarding units compare against it."),
    Part("fwdu", 15, "Forwarding unit", "EX",
         "Compares the rs / rt of the instruction in EX with the destination "
         "registers in EX/MEM and MEM/WB (when their RegWrite is 1 and the "
         "register is not $zero); the nearest match wins (EX/MEM before MEM/WB).  "
         "It drives the select lines of muxes A and B."),
    Part("dmem", 16, "Data memory", "MEM",
         "Loads read and stores write at the address the ALU computed (base + "
         "offset); MemRead / MemWrite.  Bytes and halfwords use the low address "
         "bits; a misaligned access is an address error.  With caches it is the "
         "D-cache (a miss: the line comes from the level below while MEM waits), "
         "with a TLB the address is translated first (the DTLB)."),
    Part("wbmux", 17, "Write-back mux (MemtoReg)", "WB",
         "Chooses the value written back to the register file: the data loaded in "
         "MEM (MemtoReg = 1) or the ALU result.  Together with RegWrite and the "
         "destination number it drives the register file's write port."),
    Part("vunit", 18, "FPU / MSA unit", "EX",
         "The coprocessor beside the ALU (the textbook figure has none): the FPU's "
         "and MSA's own register file (32 registers: $fN = the low 64 bits of "
         "MSA's 128-bit $wN) and execution units — FP add / multiply / divide, "
         "and MSA's integer, fixed-point and FP units working on every element "
         "(16 bytes, 8 halfwords, 4 words or 2 doublewords) at once.  It holds EX "
         "for its latency (the FPU's; MSA's from the P6600, MD01138 Table 15.3).  "
         "The integer datapath only hands it GPR operands (MTC1, FILL, INSERT), "
         "takes GPR results back (MFC1, COPY_S, CFCMSA) and computes the address "
         "of its loads and stores (LWC1, LD.df: the vector data goes to / from "
         "the data memory)."),
    Part("below_if", None, "The level below the I-cache", "IF",
         "Where an instruction-cache miss is served: the L2 (or L3, or memory).  "
         "One unified level, drawn under each L1 cache."),
    Part("below_mem", None, "The level below the D-cache", "MEM",
         "Where a data-cache miss is served and dirty lines are written back: the "
         "L2 (or L3, or memory)."),
)}


#: what is different inside on a nanoMIPS core (MD01247: 16 / 32 / 48-bit
#: instructions, compact branches only, no delay slots): (title, inside)
NANO_PARTS: dict[str, tuple[str, str]] = {
    "pcmux": ("PC mux (PCSrc)",
              "A 2-input multiplexer in front of the PC.  Its inputs are the address of "
              "the next instruction (PC + 2, 4 or 6, from the adder) and the branch / "
              "jump target computed in ID; PCSrc is 1 when the branch in ID is taken or "
              "the instruction there is a jump.  nanoMIPS has no delay slots: a taken "
              "branch replaces the instruction fetched behind it."),
    "add4": ("+2 / 4 / 6 adder",
             "An adder of the instruction's size: a nanoMIPS instruction is 16, 32 or "
             "48 bits, and its first halfword says which (MD01247 3.1: bit 12 set = 16 "
             "bits; major opcode P48I = 48).  PC + size is the next instruction: to the "
             "PC mux and through IF/ID to ID (the base of PC-relative branches and the "
             "link address)."),
    "imem": ("Instruction memory",
             "Read-only for the pipeline: the address is the PC (2-byte aligned), the "
             "output one instruction of 1, 2 or 3 halfwords, latched into IF/ID.  A "
             "32-bit instruction is two halfwords, the first one more significant, in "
             "either byte order (MD01247 A.2).  With caches it is the I-cache, with a "
             "TLB the address is translated first (the ITLB)."),
    "control": ("Main control unit",
                "Combinational logic: from the major opcode (bits 15..10 of the first "
                "halfword: the format, or the pool of formats, MD01247 Tables B.2 / "
                "B.3) and the pool's own fixed bits it makes the control signals.  They "
                "travel with the instruction through ID/EX, EX/MEM and MEM/WB.  Drawn "
                "as PLAs: the major opcode decoder makes the signals of a format, or "
                "enables its pool's decoder.  RegDst (3 bits) names the field of the "
                "register written (each format has its own), ALUOp (4 bits) the ALU's "
                "function itself: nanoMIPS has no funct field common to its formats."),
    "sext": ("Immediate decode (sign-extend)",
             "A nanoMIPS immediate can be 2 to 32 bits, its bits scattered over the "
             "encoding: the format's decode pseudocode reassembles, sign- or zero-"
             "extends and scales it (``sx(s, 15)``, ``shift(u, 2)``; MD01247 ch. 3).  "
             "The result goes to ID/EX (operand b) and to the branch-target adder."),
    "btarget": ("Branch-target adder",
                "An adder: the address of the next instruction (PC + 2, 4 or 6) + the "
                "offset, which counts bytes (always even).  BC / BALC reach ±32 MB, "
                "conditional branches less.  Every nanoMIPS branch is compact: decided "
                "in ID, no delay slot.  Register jumps (JRC, JALRC) take the target "
                "from the register file."),
    "cmp": ("Branch comparator",
            "Compares what the branch in ID asks for: two registers (BEQC, BNEC, BLTC, "
            "BGEUC ...), a register with an immediate (BEQIC, BLTIC ...), a register "
            "with zero (BEQZC, BNEZC) or one bit of a register (BBEQZC, BBNEZC).  Its "
            "output, with Branch, drives PCSrc; operands can be forwarded to it."),
    "alu": ("ALU",
            "The arithmetic-logic unit: adder / subtractor, and, or, xor, nor, "
            "set-on-less-than, the shifter.  The ALU control turns ALUOp (the function "
            "the main control names) and ALUSrc (a shift by a register's amount) into "
            "the ALU's lines.  Outputs: the result (to EX/MEM: a register value, or a "
            "memory address) and Zero; add / sub also detect signed overflow (a trap "
            "for ADD, SUB)."),
    "alusrc": ("ALUSrc mux",
               "Chooses ALU operand b: a register (ADDU, the index of LWX ...) or the "
               "decoded immediate (ADDIU, a load / store offset).  Control signal "
               "ALUSrc."),
    "regdst": ("RegDst mux",
               "Chooses the register the instruction will write.  Each nanoMIPS format "
               "puts it in its own field (rt, rd, a 3-bit rt3 through MD01247 4.7's "
               "register map, $ra for the links, a list for RESTORE / LWM): the "
               "control's RegDst names it."),
}


@dataclass
class PartInfo:
    part: Part
    instruction: str | None      # what is in the part's stage
    now: list[str] = field(default_factory=list)

    @property
    def heading(self) -> str:
        num = f"{self.part.number} · " if self.part.number is not None else ""
        return f"{num}{self.part.title} — {self.part.stage}"


# ------------------------------------------------------------- the cycle's data

def details(s) -> tuple[dict, dict]:
    """``(stage -> detail dict, events)`` of the cycle on the screen: the
    pipeline's stages, or the one executed instruction in every stage."""
    if s.machine is None:
        return {}, {}
    if s.pipeline is not None:
        d = s.stage_details()
        return d, d.pop("_events", {})
    v = s.last_view
    if v is None:
        return {}, {"label": "step to see the executed instruction"}
    one = {"text": v.text, "pc": v.pc, "view": v}
    out = {st: one for st in STAGES}
    out["IF"] = dict(one, cache=v.fetch, tlb=v.fetch_tlb)
    out["MEM"] = dict(one, cache=v.cache, tlb=v.tlb)
    return out, {"label": f"executed: {v.text}", "events": []}


def word_of(s, d: dict) -> int | None:
    """The instruction word of a stage (from its view, else memory);
    MIPS16e / microMIPS: the MIPS32 word it is recoded to (None: a form
    without one)."""
    if d.get("compressed"):
        return d.get("word")
    v = d.get("view")
    if v is not None:
        return v.word
    try:
        return s.machine.code_word(d["pc"])
    except Exception:                                   # noqa: BLE001
        return None


def nano_of(s, d: dict):
    """The nanoMIPS instruction (emips.nanomips.NM) of a stage, or None (not
    a nanoMIPS core, a bubble, no code there)."""
    m = s.machine
    if m is None or not getattr(m, "nano", False) or d.get("pc") is None:
        return None
    return s._nano_insn(m, d["pc"])


def _nano_decode(i):
    from .sim import Decode
    from .simnano import ENTRIES, entry_name
    return Decode(i.raw, ENTRIES[entry_name(i.row)])


def control_of(s, d: dict) -> dict[str, Any] | None:
    """The main-control signals of a stage's instruction (recorded for ID;
    they travel with it, so the same in every stage)."""
    from .pipeline import analyze, control_signals
    if d.get("control"):
        return d["control"]
    i = nano_of(s, d)
    if i is not None:
        dec = _nano_decode(i)
        return control_signals(dec, analyze(dec, insn=i))
    word = word_of(s, d)
    if word is None:
        return None
    try:
        dec = s.machine.isa.decode(word)
        return control_signals(dec, analyze(dec))
    except Exception:                                   # noqa: BLE001
        return None


def fields_of(s, d: dict) -> list:
    v = d.get("view")
    if v is not None:
        return v.fields
    i = nano_of(s, d)
    if i is not None:
        from .tracenano import fields
        return fields(i, "p32")
    from .trace import decode_fields
    word = word_of(s, d)
    if word is None:
        return []
    try:
        return decode_fields(s.machine.isa.decode(word))
    except Exception:                                   # noqa: BLE001
        return []


def dest_of(d: dict, names: Any = None) -> tuple[str, int | None] | None:
    """``(register, value written)`` if the stage's instruction writes a
    GPR (RegWrite), else None (the trace lists only changed registers: an
    unchanged write takes the ALU result).  ``names``: the core's register
    names (Machine.names; default: by the view's width)."""
    v, ctl = d.get("view"), d.get("control") or {}
    gprs = [w for w in (v.reg_writes if v is not None else []) if w[0] in _GPRS]
    if gprs:
        return gprs[0][0], gprs[0][2]
    if not ctl.get("RegWrite") or v is None or v.exception:
        return None
    w, dst = v.word, ctl.get("RegDst")
    if isinstance(dst, str) and dst[1:2].isdigit():     # a number: compressed / nanoMIPS
        reg = int(dst[1:].split(",")[0])
    else:
        reg = {"rd": (w >> 11) & 31, "$ra": 31, "rd6": (w >> 6) & 31}.get(dst, (w >> 16) & 31)
    if reg == 0:
        return None
    value = None if ctl.get("MemtoReg") else (v.alu.result if v.alu is not None else None)
    if names is None:
        names = "n64" if getattr(v, "digits", 8) == 16 else True
    return gpr_name(reg, names), value


def taken(d: dict, ev: dict) -> bool:
    v = d.get("view")
    return any("taken ->" in e for e in ev.get("events") or []) or (
        v is not None and v.branch is not None and v.branch.taken)


def tlb_lines(translations: list) -> list[str]:
    """What the micro-TLB / TLB (or BAT) did for an access."""
    lines = []
    for a in translations:
        if a.note:
            continue
        if a.micro:
            where = f"TLB entry {a.entry}" if a.entry is not None else "no TLB entry"
            lines.append(f"{a.micro} hit" if a.micro_hit else f"{a.micro} MISS: {where}")
        elif a.entry is not None:
            lines.append(a.name)
        if a.fault:
            lines.append(f"  {a.fault}")
    return lines


# ------------------------------------------------------------- per part

def alu_view(s, key: str):
    """The InstrView whose ALU operation part ``key`` (the ALU, or the
    24K's EX) shows this cycle, or None."""
    if key not in ("alu", "24k:EX") or s.machine is None:
        return None
    det, _ = details(s)
    return (det.get("EX") or {}).get("view")


def control_word(s, key: str) -> int | None:
    """The instruction word the main control decodes for part ``key``
    (Control in ID, the 24K's RF), or None."""
    if key not in ("control", "24k:RF") or s.machine is None:
        return None
    det, _ = details(s)
    d = det.get("ID" if key == "control" else "RF") or {}
    if d.get("pc") is None:
        return None
    return word_of(s, d)


def control_nano(s, key: str):
    """The nanoMIPS instruction (emips.nanomips.NM) the main control decodes
    for part ``key`` (Control, in ID), or None."""
    if key != "control" or s.machine is None or not getattr(s.machine, "nano", False):
        return None
    det, _ = details(s)
    return nano_of(s, det.get("ID") or {})


def _stage_of(key: str) -> str:
    st = PARTS[key].stage
    return "ID" if st == "ID / WB" else st


def _field(fields: list, *names: str):
    return next((f for f in fields if f.name in names), None)


def _h(v: int | None) -> str:
    return "?" if v is None else f"0x{v:0{_DIGITS[0]}x}"


def _mask() -> int:
    return (1 << 4 * _DIGITS[0]) - 1


def part_info(s, key: str) -> PartInfo:
    """What part ``key`` is and does in the cycle on the screen."""
    _DIGITS[0] = s.hex_digits() if s.machine is not None else 8
    if key.startswith("24k:"):
        return _24k(s, key)
    part = PARTS[key]
    if key in NANO_PARTS and getattr(s.machine, "nano", False):
        title, inside = NANO_PARTS[key]
        part = replace(part, title=title, inside=inside)
    det, ev = details(s)
    st = _stage_of(key)
    d = det.get(st) or {}
    text = d.get("text") if d.get("pc") is not None else None
    info = PartInfo(part, text)
    if s.machine is None:
        info.now = ["No program: assemble or open one, then step."]
        return info
    if not det:
        info.now = [ev.get("label") or "Step to see what happens."]
        return info
    if text is None and key not in ("hazard", "fwdu", "regs"):
        info.now = [f"{st} holds a bubble this cycle: nothing to do."]
        return info
    pipeline = s.pipeline is not None
    nano = _NANO_BUILDERS.get(key) if getattr(s.machine, "nano", False) else None
    lines = (nano or _BUILDERS[key])(s, d, det, ev, pipeline)
    info.now = [ln for ln in lines if ln is not None] or [
        f"{st} holds {text}: no instruction word to decode"]
    return info


def _pcmux(s, d, det, ev, pipeline):
    pc = d.get("pc")
    idd = det.get("ID") or {}
    target = None
    v = idd.get("view")
    if v is not None and v.branch is not None:
        target = v.branch.target
    out = [f"input 0, PC + 4: {_h(pc + 4 if pc is not None else None)}",
           f"input 1, branch / jump target from ID: {_h(target)}"
           if target is not None else "input 1, branch / jump target: (no branch in ID)"]
    stall = ev.get("stall")
    if pipeline and stall and not stall.startswith(("memory", "I-miss")):
        out.append("stall: the PC is not written (PCWrite = 0) — the same fetch again")
    elif taken(idd, ev) if pipeline else taken(d, ev):
        out.append(f"PCSrc = 1 (taken): the next PC is the target {_h(target)}"
                   + ("" if pipeline else " (after the delay slot)"))
    else:
        out.append(f"PCSrc = 0: the next PC is {_h(pc + 4 if pc is not None else None)}")
    if any(e.startswith("exception") for e in ev.get("events") or []):
        out.append("an exception this cycle: the PC is loaded with the handler's vector")
    return out


def _pc(s, d, det, ev, pipeline):
    out = [f"holds {_h(d.get('pc'))}: the address of {d.get('text')}"]
    stall = ev.get("stall")
    if pipeline and stall and not stall.startswith(("memory", "I-miss")):
        out.append("held this cycle: the hazard detection unit stalls (PCWrite = 0)")
    if d.get("wait"):
        out.append(f"the fetch waits for the I-cache: {d['wait']} more cycle(s)")
    return out


def _add4(s, d, det, ev, pipeline):
    pc = d.get("pc")
    return [f"{_h(pc)} + 4 = {_h((pc + 4) & _mask())}",
            "to the PC mux (input 0) and to IF/ID (PC + 4 for ID)"]


def _imem(s, d, det, ev, pipeline):
    word = word_of(s, d)
    out = [f"address (PC): {_h(d.get('pc'))}",
           f"reads the word {_h(word)}: {d.get('text')}"]
    for t in tlb_lines(d.get("tlb") or []):
        out.append(f"translation: {t.strip()}")
    for a in d.get("cache") or []:
        out.append(a.describe())
    if d.get("wait"):
        out.append(f"waiting for the line: {d['wait']} more cycle(s)")
    return out


_SIGNALS = {
    "RegDst": "which register is written: rt, rd, $ra or rd6 (MSA: bits 10..6); - = no GPR",
    "ALUSrc": "ALU operand b: the register (reg) or the immediate (imm)",
    "MemtoReg": "write back the loaded data (1) or the ALU result (0)",
    "RegWrite": "a register is written in WB",
    "MemRead": "MEM reads data memory (a load)",
    "MemWrite": "MEM writes data memory (a store)",
    "Branch": "a branch: decided in ID",
    "Jump": "a jump: the target goes to the PC",
    "ALUOp": "what the ALU does (from the opcode / funct)",
}


def _control(s, d, det, ev, pipeline):
    fields = fields_of(s, d)
    out = []
    op = _field(fields, "op")
    fn = _field(fields, "funct")
    if op is not None:
        out.append(f"opcode (31..26) = {op.bits} ({op.meaning})"
                   + (f", funct (5..0) = {fn.bits} ({fn.meaning})" if fn is not None else ""))
    ctl = control_of(s, d) or {}
    for k, v in ctl.items():
        val = ("1" if v else "0") if isinstance(v, bool) else str(v)
        out.append(f"{k} = {val}: {_SIGNALS.get(k, '')}")
    return out


def _regs(s, d, det, ev, pipeline):
    out = []
    if d.get("pc") is not None:
        fields = fields_of(s, d)
        rs, rt = _field(fields, "rs", "base"), _field(fields, "rt")
        reads = [f"{f.name} = {f.meaning} (#{f.value})" for f in (rs, rt) if f is not None]
        out.append("read port 1 / 2 (ID, " + str(d.get("text")) + "): "
                   + (", ".join(reads) if reads else "no register operands"))
        v = d.get("view")
        if v is not None and v.alu is not None:
            for name, op in (("a", v.alu.a), ("b", v.alu.b)):
                if op is not None and op.source.startswith("$"):
                    out.append(f"  read {op.source}: {_h(op.value)}")
        elif pipeline:
            out.append("  the values go into ID/EX: see muxes A / B next cycle "
                       "(a forwarded value may replace them)")
    else:
        out.append("ID holds a bubble: nothing is read")
    wb = det.get("WB") or {}
    dest = dest_of(dict(wb, control=control_of(s, wb))) if wb.get("pc") is not None else None
    if dest is not None:
        name, value = dest
        out.append(f"write port (WB, {wb.get('text')}): {name} <- {_h(value)} "
                   "(first half of the cycle)")
    else:
        out.append("write port: nothing written this cycle")
    return out


def _sext(s, d, det, ev, pipeline):
    imm = _field(fields_of(s, d), "immediate", "offset")
    if imm is None or imm.msb - imm.lsb != 15:
        return [f"not used: {d.get('text')} has no 16-bit immediate"]
    raw = imm.value
    name = (d.get("view").name if d.get("view") is not None else
            s.machine.isa.decode(word_of(s, d)).name)
    if name in ("ANDI", "ORI", "XORI"):
        how, val = "zero-extended (a logical immediate)", raw
    elif name == "LUI":
        how, val = "lui: placed in the upper half", raw << 16
    else:
        how, val = "sign-extended (bit 15 = " + str(raw >> 15) + ")", \
            raw - 0x10000 if raw & 0x8000 else raw
    return [f"{imm.name} (bits {imm.msb}..{imm.lsb}) = {imm.bits} = 0x{raw:04x}",
            f"{how}: {_h(val & _mask())} = {val}"]


def _btarget(s, d, det, ev, pipeline):
    ctl = control_of(s, d) or {}
    if not (ctl.get("Branch") or ctl.get("Jump")):
        return [f"not used: {d.get('text')} is not a branch or jump"]
    v = d.get("view")
    pc = d.get("pc")
    out = [f"PC + 4 = {_h((pc + 4) & _mask())}"]
    f = _field(fields_of(s, d), "offset", "instr_index")
    if f is not None:
        out.append(f"{f.name} = {f.meaning}")
    if v is not None and v.branch is not None:
        out.append(f"target = {_h(v.branch.target)}")
    elif ctl.get("Jump"):
        out.append("the target comes from a register (jr / jalr): see the operands")
    return out


def _cmp(s, d, det, ev, pipeline):
    ctl = control_of(s, d) or {}
    if not ctl.get("Branch"):
        return [f"not used: {d.get('text')} is not a conditional branch"]
    v = d.get("view")
    out = []
    if v is not None and v.branch is not None:
        out.append(f"condition: {v.branch.condition}")
        out.append("taken" if v.branch.taken else "not taken"
                   + (" (likely: the delay slot is squashed)" if v.branch.likely else ""))
    fwd = [e for e in ev.get("events") or [] if "-> ID" in e]
    out += [f"forwarded: {e}" for e in fwd]
    if v is None and pipeline:
        out.append("decided when the operands are ready (the hazard unit may stall it)")
    return out


def _hazard(s, d, det, ev, pipeline):
    if not pipeline:
        return ["instruction mode: no pipeline — nothing overlaps, nothing to stall"]
    stall = ev.get("stall")
    idd = det.get("ID") or {}
    out = [f"checks {idd.get('text') or '(bubble)'} in ID against "
           f"{(det.get('EX') or {}).get('text') or '(bubble)'} in EX and "
           f"{(det.get('MEM') or {}).get('text') or '(bubble)'} in MEM"]
    if stall and not stall.startswith(("memory", "I-miss")):
        out.append(f"STALL: {stall}")
        out.append("PC and IF/ID hold, a bubble enters EX")
    elif stall:
        out.append(f"the pipeline waits for memory ({stall}) — not this unit")
    else:
        out.append("no hazard: ID goes on")
    return out


def _operand(s, det, which: str, pipeline: bool) -> list[str]:
    exd = det.get("EX") or {}
    v = exd.get("view")
    out = []
    if pipeline:
        mem, wb = det.get("MEM") or {}, det.get("WB") or {}
        for label, sd in (("EX/MEM", mem), ("MEM/WB", wb)):
            dest = (dest_of(dict(sd, control=control_of(s, sd)), s.machine.names)
                    if sd.get("pc") is not None else None)
            out.append(f"input {label}: " + (f"{dest[0]} = {_h(dest[1])} ({sd.get('text')})"
                                             if dest else "no register result"))
    else:
        out.append("instruction mode: no instruction ahead — always the register file")
    if v is None or v.alu is None:
        return out + ["no ALU operand this cycle"]
    op = v.alu.a if which == "a" else v.alu.b
    if op is None:
        return out + [f"operand {which}: not used"]
    if which == "b" and not op.source.startswith("$") and getattr(s.machine, "nano", False):
        out.insert(0, f"ALUSrc takes the {op.source} for operand b"
                      + ("; mux B carries the store data" if v.mem_writes else ""))
        return out
    if which == "b" and not op.source.startswith("$"):
        rt = _field(v.fields, "rt")
        out.insert(0, f"ALUSrc takes the {op.source.split(' ')[0]} for operand b; mux B "
                      f"carries rt ({rt.meaning if rt else '-'})"
                      + (": the store data" if v.mem_writes else ""))
        return out
    out.insert(0, f"chosen: {_h(op.value)} from {op.source}")
    return out


def _fwda(s, d, det, ev, pipeline):
    return _operand(s, det, "a", pipeline)


def _fwdb(s, d, det, ev, pipeline):
    return _operand(s, det, "b", pipeline)


def _alusrc(s, d, det, ev, pipeline):
    ctl = control_of(s, d) or {}
    v = d.get("view")
    out = [f"ALUSrc = {ctl.get('ALUSrc', '?')}"]
    if v is not None and v.alu is not None and v.alu.b is not None:
        out.append(f"operand b = {_h(v.alu.b.value)} ({v.alu.b.source})")
    return out


def coprocessor_reads(s, word: int | None) -> tuple[str, list[str]] | None:
    """An FPU / MSA instruction word in ID: ("fpu" | "msa", the registers it
    reads — the coprocessor's own and the GPRs it hands over), else None."""
    if word is None or s.machine is None:
        return None
    try:
        dec = s.machine.isa.decode(word)
    except Exception:                                    # noqa: BLE001
        return None
    e = dec.entry
    if e.get("ase") == "msa" and e["family"] != "alu":
        unit = "msa"
    elif e["family"].startswith("fpu"):
        unit = "fpu"
    else:
        return None
    from .pipeline import analyze
    info = analyze(dec)
    names = []
    for r in sorted(info.reads_ex | info.reads_id, key=str):
        if isinstance(r, tuple) and r[0] == "f":
            names.append(f"${'w' if unit == 'msa' else 'f'}{r[1]}")
        elif isinstance(r, tuple) and r[0] == "r":
            names.append(gpr_name(r[1], s.machine.names))
        elif isinstance(r, str):
            names.append(r.upper() if r != "fcc" else "FCC")
    return unit, names


def coprocessor_of(s, v) -> str | None:
    """"fpu" / "msa" when the FPU / MSA unit executes the instruction."""
    if v is None:
        return None
    if getattr(v, "lanes", None) is not None:
        return "msa"
    try:
        e = s.machine.isa.get(v.name)
    except Exception:                                    # noqa: BLE001
        return None
    if e is None:
        return None
    if e.get("ase") == "msa" and e["family"] != "alu":
        return "msa"
    return "fpu" if e["family"].startswith("fpu") else None


def _alu(s, d, det, ev, pipeline):
    v = d.get("view")
    if v is None or v.alu is None:
        unit = coprocessor_of(s, v)
        if unit is not None:
            return [f"{d.get('text')}: the ALU does no operation for it — the "
                    f"{'MSA unit' if unit == 'msa' else 'FPU'} (18) executes it"]
        return [f"{d.get('text')}: the ALU does no operation for it"]
    a = v.alu
    out = [f"operation: {a.op}"]
    if a.a is not None:
        out.append(f"a = {_h(a.a.value)} ({a.a.source})")
    if a.b is not None:
        out.append(f"b = {_h(a.b.value)} ({a.b.source})")
    if a.result is not None:
        bits = 4 * _DIGITS[0]
        out.append(f"result = {_h(a.result)} "
                   f"({a.result - (1 << bits) if a.result >> (bits - 1) & 1 else a.result})")
    if a.zero is not None:
        out.append(f"Zero = {int(a.zero)}")
    if a.overflow:
        out.append("OVERFLOW: signed overflow")
    if v.exception:
        out.append(f"EXCEPTION: {v.exception}")
    return out


def _regdst(s, d, det, ev, pipeline):
    ctl = control_of(s, d) or {}
    fields = fields_of(s, d)
    rt, rd = _field(fields, "rt"), _field(fields, "rd")
    out = [f"RegDst = {ctl.get('RegDst', '?')}; inputs: rt = "
           f"{rt.meaning if rt else '-'}, rd = {rd.meaning if rd else '-'}, 31 = $ra"
           + (", rd6 = bits 10..6 (MSA)" if getattr(s.machine, "msa", False) else "")]
    dest = dest_of(dict(d, control=ctl))
    out.append(f"destination: {dest[0]}" if dest else "RegWrite = 0: no register written")
    return out


def _fwdu(s, d, det, ev, pipeline):
    if not pipeline:
        return ["instruction mode: no pipeline — nothing to forward"]
    exd = det.get("EX") or {}
    fields = fields_of(s, exd) if exd.get("pc") is not None else []
    rs, rt = _field(fields, "rs", "base"), _field(fields, "rt")
    reads = ", ".join(f.meaning for f in (rs, rt) if f is not None) or "-"
    out = [f"EX ({exd.get('text') or '(bubble)'}) reads: {reads}"]
    for label, st in (("EX/MEM", "MEM"), ("MEM/WB", "WB")):
        sd = det.get(st) or {}
        dest = dest_of(dict(sd, control=control_of(s, sd))) if sd.get("pc") is not None else None
        out.append(f"{label} writes: {dest[0] if dest else 'nothing'}")
    fwd = [e for e in ev.get("events") or [] if e.startswith("forward")]
    out += fwd or ["no match: the operands come from the register file"]
    return out


def _dmem(s, d, det, ev, pipeline):
    v = d.get("view")
    if v is None or not (v.mem_reads or v.mem_writes or v.address):
        return [f"not used: {d.get('text')} does not access memory "
                "(its ALU result goes around it to MEM/WB)"]
    out = []
    a = v.address
    if a is not None:
        out.append(f"address = {a.base.source} {_h(a.base.value)} + {a.offset} = "
                   f"{_h(a.address)} ({a.size} bytes, {a.access})")
        out.append(f"{a.segment}, {a.privilege}, "
                   + ("mapped" if a.mapped else "unmapped") + ", "
                   + ("cached" if a.cached else "uncached")
                   + ("" if a.aligned else ", MISALIGNED"))
        if a.physical is not None:
            out.append(f"physical address {_h(a.physical)}" + (f" ({a.how})" if a.how else ""))
    for addr, n, val in v.mem_reads:
        out.append(f"read  {_h(addr)} = 0x{val:0{2 * n}x}")
    for addr, n, old, new in v.mem_writes:
        out.append(f"write {_h(addr)}: 0x{old:0{2 * n}x} -> 0x{new:0{2 * n}x}")
    for t in tlb_lines(d.get("tlb") or []):
        out.append(f"translation: {t.strip()}")
    for acc in d.get("cache") or []:
        out.append(acc.describe())
    if d.get("wait"):
        out.append(f"waiting: {d['wait']} more cycle(s)")
    return out


def _wbmux(s, d, det, ev, pipeline):
    ctl = control_of(s, d) or {}
    dest = dest_of(dict(d, control=ctl))
    if dest is None:
        return [f"RegWrite = 0: {d.get('text')} writes no register"]
    v = d.get("view")
    load = bool(ctl.get("MemtoReg")) or (v is not None and bool(v.mem_reads)
                                          and not v.mem_writes)
    return [f"MemtoReg = {int(load)}: " + ("the loaded data" if load else "the ALU result"),
            f"{dest[0]} <- {_h(dest[1])}"]


def _below(s, d, det, ev, pipeline):
    out = []
    for acc in d.get("cache") or []:
        if acc.level in ("S", "T"):
            out.append(acc.describe())
    return out or ["not used this cycle (the L1 cache served the access, or no "
                   "access)"]


def _vunit(s, d, det, ev, pipeline):
    v = d.get("view")
    unit = coprocessor_of(s, v)
    if unit is None:
        return [f"{d.get('text')}: not used (an integer instruction)"]
    out = []
    if unit == "msa" and v.lanes is not None:
        out.append(f"MSA: {d.get('text')}, {v.lanes.size}-bit elements, element 0 first")
        for role, reg, size, es in v.lanes.rows:
            shown = " ".join(f"{x:0{size // 4}x}" for x in es[:8])
            out.append(f"  {role} $w{reg}: {shown}" + (" …" if len(es) > 8 else ""))
    else:
        out.append(f"{'MSA' if unit == 'msa' else 'FPU'}: {d.get('text')}")
    for name, old, new in v.reg_writes:
        if name.startswith(("$f", "$w")) or name in ("fcsr", "msacsr"):
            digits = 32 if name.startswith("$w") else 8 if name in ("fcsr", "msacsr") else 16
            out.append(f"{name}: 0x{old:0{digits}x} -> 0x{new:0{digits}x}")
    if v.exception:
        out.append(f"EXCEPTION: {v.exception}")
    return out


_BUILDERS = {"pcmux": _pcmux, "pc": _pc, "add4": _add4, "imem": _imem,
             "control": _control, "regs": _regs, "sext": _sext, "btarget": _btarget,
             "cmp": _cmp, "hazard": _hazard, "fwda": _fwda, "fwdb": _fwdb,
             "alusrc": _alusrc, "alu": _alu, "regdst": _regdst, "fwdu": _fwdu,
             "dmem": _dmem, "wbmux": _wbmux, "below_if": _below, "below_mem": _below,
             "vunit": _vunit}


# ------------------------------------------------------------- nanoMIPS

def _names(s) -> Any:
    """The core's register names (the datapath always shows names)."""
    return s.machine.names


def _halfwords(i) -> str:
    return " ".join(f"{(i.raw >> (16 * k)) & 0xFFFF:04x}"
                    for k in range(i.size // 2 - 1, -1, -1))


def _nano_target(s, d: dict) -> int | None:
    """The branch / jump target of a stage's instruction: what it did (its
    view), else what its encoding says (PC-relative only)."""
    v = d.get("view")
    if v is not None and v.branch is not None and v.branch.target is not None:
        return v.branch.target
    i = nano_of(s, d)
    if i is None:
        return None
    from .nanomips import target
    from .tracenano import _BRANCHES
    return target(i, d["pc"]) if i.row.name in _BRANCHES else None


def _nano_pcmux(s, d, det, ev, pipeline):
    pc, i = d.get("pc"), nano_of(s, d)
    nxt = (pc + i.size) & _mask() if i is not None and pc is not None else None
    idd = det.get("ID") or {} if pipeline else d
    target = _nano_target(s, idd)
    out = [f"input 0, PC + {i.size if i else '?'}: {_h(nxt)}",
           f"input 1, branch / jump target from ID: {_h(target)}"
           if target is not None else "input 1, branch / jump target: (no branch in ID)"]
    stall = ev.get("stall")
    if pipeline and stall and not stall.startswith(("memory", "I-miss")):
        out.append("stall: the PC is not written (PCWrite = 0) — the same fetch again")
    elif taken(idd, ev):
        out.append(f"PCSrc = 1 (taken): the next PC is the target {_h(target)}"
                   + (" — the instruction fetched behind the branch is dropped (no "
                      "delay slot)" if pipeline else ""))
    else:
        out.append(f"PCSrc = 0: the next PC is {_h(nxt)}")
    if any(e.startswith("exception") for e in ev.get("events") or []):
        out.append("an exception this cycle: the PC is loaded with the handler's vector")
    return out


def _nano_add4(s, d, det, ev, pipeline):
    pc, i = d.get("pc"), nano_of(s, d)
    if i is None:
        return [f"{_h(pc)}: no nanoMIPS instruction to size"]
    h0 = i.raw >> (16 * (i.size // 2 - 1))
    why = ("bit 12 = 1: 16 bits" if i.size == 2 else
           "major opcode 011000 (P48I): 48 bits" if i.size == 6 else
           "bit 12 = 0: 32 bits")
    return [f"first halfword 0x{h0:04x}: {why} (MD01247 3.1)",
            f"{_h(pc)} + {i.size} = {_h((pc + i.size) & _mask())}",
            "to the PC mux (input 0) and to IF/ID (the next instruction's address for ID)"]


def _nano_imem(s, d, det, ev, pipeline):
    i = nano_of(s, d)
    out = [f"address (PC): {_h(d.get('pc'))}"]
    if i is not None:
        out.append(f"reads {i.size // 2} halfword(s) {_halfwords(i)}: a {8 * i.size}-bit "
                   f"{i.row.form}, {d.get('text')}")
    else:
        out.append(f"reads: {d.get('text')}")
    for t in tlb_lines(d.get("tlb") or []):
        out.append(f"translation: {t.strip()}")
    for a in d.get("cache") or []:
        out.append(a.describe())
    if d.get("wait"):
        out.append(f"waiting for the line: {d['wait']} more cycle(s)")
    return out


def _nano_control(s, d, det, ev, pipeline):
    fields = fields_of(s, d)
    out = []
    major = _field(fields, "major")
    if major is not None:
        out.append(f"major opcode (15..10 of the first halfword) = {major.bits} "
                   f"({major.meaning})")
    ops = [f for f in fields if f.name == "op"]
    if ops:
        out.append("the pool's fixed bits: " + ", ".join(
            f"{f.msb}..{f.lsb} = {f.bits}" if f.msb != f.lsb else f"{f.msb} = {f.bits}"
            for f in ops) + (f" -> {ops[-1].meaning}" if ops[-1].meaning else ""))
    for k, v in (control_of(s, d) or {}).items():
        val = ("1" if v else "0") if isinstance(v, bool) else str(v)
        if k == "RegDst" and val[1:2].isdigit():       # "$4, $5" -> "$4 = $a0, ..."
            val = ", ".join(f"{r} = {gpr_name(int(r.strip()[1:]), _names(s))}"
                            for r in val.split(","))
        out.append(f"{k} = {val}: {_NANO_SIGNALS.get(k) or _SIGNALS.get(k, '')}")
    return out


_NANO_SIGNALS = {
    "RegDst": "the register written (named by the format's field)",
    "ALUSrc": "ALU operand b: a register (reg), the decoded immediate (imm), none (-)",
    "ALUOp": "what the ALU does (the instruction page)",
}


def _nano_regs(s, d, det, ev, pipeline):
    from .simnano import uses
    out = []
    i = nano_of(s, d)
    if d.get("pc") is not None and i is not None:
        reads = [gpr_name(n, _names(s)) for n in uses(i)[0]]
        out.append(f"read ports (ID, {d.get('text')}): "
                   + (", ".join(reads) if reads else "no register operands")
                   + (" — two a cycle: a list (SAVE, SWM ...) is read a word a cycle"
                      if len(reads) > 2 else ""))
        v = d.get("view")
        if v is not None and v.alu is not None:
            for op in (v.alu.a, v.alu.b):
                if op is not None and op.source.startswith("$"):
                    out.append(f"  read {op.source}: {_h(op.value)}")
        elif pipeline:
            out.append("  the values go into ID/EX: see muxes A / B next cycle "
                       "(a forwarded value may replace them)")
    else:
        out.append("ID holds a bubble: nothing is read")
    wb = det.get("WB") or {}
    dest = (dest_of(dict(wb, control=control_of(s, wb)), _names(s))
            if wb.get("pc") is not None else None)
    if dest is not None:
        out.append(f"write port (WB, {wb.get('text')}): {dest[0]} <- {_h(dest[1])} "
                   "(first half of the cycle)")
    else:
        out.append("write port: nothing written this cycle")
    return out


def _nano_immediates(s, d) -> list:
    """The fields of a stage's instruction that make immediates (not
    registers, not opcode bits)."""
    from .tracenano import _REGISTERS, _derived
    i = nano_of(s, d)
    if i is None:
        return []
    made, _ = _derived(i.row)
    out = []
    for f in fields_of(s, d):
        if f.name in ("major", "op", "x") or f.name in _REGISTERS:
            continue
        derived = made.get(f.name)
        if derived and all(x in _REGISTERS for x in derived):
            continue                                # a register field (rt3 -> rt)
        out.append(f)
    return out


def _nano_sext(s, d, det, ev, pipeline):
    imm = _nano_immediates(s, d)
    if not imm:
        return [f"not used: {d.get('text')} has no immediate"]
    return [f"{f.name} (bits {f.msb}..{f.lsb}) = {f.bits}" + (f": {f.meaning}" if f.meaning
                                                               else "") for f in imm]


def _nano_btarget(s, d, det, ev, pipeline):
    ctl = control_of(s, d) or {}
    if not (ctl.get("Branch") or ctl.get("Jump")):
        return [f"not used: {d.get('text')} is not a branch or jump"]
    pc, i = d.get("pc"), nano_of(s, d)
    if i is None:
        return []
    out = [f"the next instruction: PC + {i.size} = {_h((pc + i.size) & _mask())}"]
    if "offset" in i.f:
        out.append(f"offset = {i.f['offset']} bytes")
    target = _nano_target(s, d)
    if target is not None:
        out.append(f"target = {_h(target)}")
    else:
        out.append("the target comes from a register (JRC / JALRC / BRSC): see the "
                   "operands")
    return out


def _nano_cmp(s, d, det, ev, pipeline):
    from .tracenano import _cond
    ctl = control_of(s, d) or {}
    i = nano_of(s, d)
    if not ctl.get("Branch") or i is None:
        return [f"not used: {d.get('text')} is not a conditional branch"]
    out = [f"condition: {_cond(i.row.name, i.f, lambda n: gpr_name(n, _names(s)))}"]
    v = d.get("view")
    if v is not None and v.branch is not None:
        out.append("taken" if v.branch.taken else "not taken")
    fwd = [e for e in ev.get("events") or [] if "-> ID" in e]
    out += [f"forwarded: {e}" for e in fwd]
    if v is None and pipeline:
        out.append("decided when the operands are ready (the hazard unit may stall it)")
    return out


def _nano_regdst(s, d, det, ev, pipeline):
    from .simnano import uses
    ctl = control_of(s, d) or {}
    i = nano_of(s, d)
    written = [gpr_name(n, _names(s)) for n in uses(i)[1]] if i is not None else []
    out = [f"RegDst = {ctl.get('RegDst', '?')}"
           + (f": {', '.join(written)}" if written else "")]
    dest = dest_of(dict(d, control=ctl), _names(s))
    out.append(f"destination: {dest[0]}" if dest else "RegWrite = 0: no register written")
    if i is not None and dest:
        from . import nanocontrol as nc
        code = nc.regdst(i.row)
        if code is not None:
            out.append(f"the main control's RegDst2..0 = {code:03b}: {nc.REGDST[code]}")
    return out


def _nano_fwdu(s, d, det, ev, pipeline):
    from .simnano import uses
    if not pipeline:
        return ["instruction mode: no pipeline — nothing to forward"]
    exd = det.get("EX") or {}
    i = nano_of(s, exd)
    reads = ", ".join(gpr_name(n, _names(s)) for n in uses(i)[0]) if i is not None else ""
    out = [f"EX ({exd.get('text') or '(bubble)'}) reads: {reads or '-'}"]
    for label, st in (("EX/MEM", "MEM"), ("MEM/WB", "WB")):
        sd = det.get(st) or {}
        dest = (dest_of(dict(sd, control=control_of(s, sd)), _names(s))
                if sd.get("pc") is not None else None)
        out.append(f"{label} writes: {dest[0] if dest else 'nothing'}")
    fwd = [e for e in ev.get("events") or [] if e.startswith("forward")]
    out += fwd or ["no match: the operands come from the register file"]
    return out


def _nano_wbmux(s, d, det, ev, pipeline):
    ctl = control_of(s, d) or {}
    dest = dest_of(dict(d, control=ctl), _names(s))
    if dest is None:
        return [f"RegWrite = 0: {d.get('text')} writes no register"]
    v = d.get("view")
    load = bool(ctl.get("MemtoReg")) or (v is not None and bool(v.mem_reads)
                                          and not v.mem_writes)
    return [f"MemtoReg = {int(load)}: " + ("the loaded data" if load else "the ALU result"),
            f"{dest[0]} <- {_h(dest[1])}"]


#: the nanoMIPS versions of the builders (the others serve both)
_NANO_BUILDERS = {"pcmux": _nano_pcmux, "add4": _nano_add4, "imem": _nano_imem,
                  "control": _nano_control, "regs": _nano_regs, "sext": _nano_sext,
                  "btarget": _nano_btarget, "cmp": _nano_cmp, "regdst": _nano_regdst,
                  "fwdu": _nano_fwdu, "wbmux": _nano_wbmux}


# ------------------------------------------------------------- the MIPS 24K

#: the 24K's drawing (the Datapath with the 24K model): the nine stages and
#: the fetch / execution units around them, numbered 1-13
PARTS_24K: dict[str, Part] = {p.key: p for p in (
    Part("24k:IF", 1, "IF — instruction fetch", "IF",
         "The I-cache data and tags are read with the PC of a 64-bit bundle (two "
         "instructions, one at an odd word), the ITLB translates the PC, and the "
         "branch history table (512 two-bit counters, index = PC[11:3]) is read "
         "for the branches in the bundle."),
    Part("24k:IS", 2, "IS — instruction select", "IS",
         "The tags are compared (hit, way select); a conditional branch is "
         "predicted from its BHT counter, jr $31 from the 4-entry return "
         "prediction stack, j / jal / b / bal always taken; a predicted-taken "
         "branch redirects the fetch to its target (after the delay slot)."),
    Part("24k:IT", 3, "IT — instruction buffer", "IT",
         "The 8-entry instruction buffer decouples the fetch (two a cycle) from "
         "the pipeline (one a cycle); it is bypassed while empty.  The branch "
         "target is computed here."),
    Part("24k:RF", 4, "RF — register fetch", "RF",
         "The register file is read and the instruction decoded.  A load or store "
         "waits here for its base register when an ALU result is not ready for the "
         "address adder (the skewed ALU: an ALU result reaches AG one cycle "
         "late)."),
    Part("24k:AG", 5, "AG — address generation", "AG",
         "The data address of a load / store is added (base + offset).  Any "
         "instruction waits here for operands still on their way (load data, "
         "MFC0, MUL / MFHI results) or for a busy unit (divider, FPU, the load / "
         "store unit, a CP0 hazard barrier)."),
    Part("24k:EX", 6, "EX — execute", "EX",
         "The ALU computes, the DTLB translates the data address and the D-cache "
         "access starts; branches are resolved — a misprediction kills the younger "
         "instructions and redirects the fetch (4 cycles lost)."),
    Part("24k:MS", 7, "MS — memory select", "MS",
         "The D-cache tags are compared (hit, way select) and load data is aligned "
         "and sign / zero-extended.  A waiting access (a blocking miss, a full write "
         "buffer, SYNC, CACHE) holds the pipeline here."),
    Part("24k:ER", 8, "ER — exception resolution", "ER",
         "Exceptions and interrupts are taken here: the older instructions have "
         "completed, the younger are killed and the fetch goes to the vector; the "
         "instruction completes."),
    Part("24k:WB", 9, "WB — write back", "WB",
         "The register file is written (the result was already available to "
         "younger instructions through the bypasses)."),
    Part("24k:buffer", 10, "Instruction buffer", "IS / IT",
         "The instructions fetched but not yet in RF, oldest first: where each is "
         "(IS / IT), when it can enter RF, the branch predictions made at fetch."),
    Part("24k:predict", 11, "Branch prediction", "IS / EX",
         "The branch history table (512 two-bit counters: strongly / weakly taken "
         "or not taken) and the 4-entry return prediction stack (jal pushes, "
         "jr $31 pops); predictions are made in IS and checked when the branch "
         "executes in EX."),
    Part("24k:units", 12, "Units", "AG / EX / MS",
         "The multiply / divide unit (MUL issues 3 per 5 cycles, the divider is "
         "iterative), the FPU (its repeat rates), the load / store unit (its load "
         "data queue of 9, the fill-store buffer, the DTLB): an instruction waits "
         "in AG while the unit it needs is busy."),
    Part("24k:hazards", 13, "CP0 hazards", "EX",
         "A CP0 write (mtc0, ei / di, the TLB instructions) takes effect a few "
         "cycles later: an instruction executing too soon still sees the old value "
         "(MD00343 Table 2.7).  ehb waits for them; eret, jr.hb and jalr.hb clear "
         "them for what follows."),
)}


def _24k(s, key: str) -> PartInfo:
    from . import coreview as cv
    part = PARTS_24K[key]
    p = s.pipeline
    if p is None or not cv.is_24k(p):
        return PartInfo(part, None, ["pipeline mode with the 24K model shows it"])
    det = s.stage_details()
    ev = det.pop("_events", {})
    name = key.split(":", 1)[1]
    stall = ev.get("stall") or ""
    if name in STAGES_24K:
        d = det.get(name) or {}
        text = d.get("text") if d.get("pc") is not None else None
        lines = [] if text is not None else [f"{name} holds a bubble this cycle"]
        if text is not None:
            lines += _stage24(s, name, d, ev, p)
        held = {"memory": "MS", "fetch": "IT", "alu-agen": "RF"}.get(
            stall.split(":", 1)[0], "AG") if stall else None
        if held == name:
            lines.append(f"HELD: {stall}")
        return PartInfo(part, text, lines)
    if name == "buffer":
        lines = [f"fetch unit: {cv.fetch_state(p)}"]
        rows = cv.buffer_rows(p)
        lines += [f"{r.where} 0x{r.pc:08x} {r.text}"
                  + (" (delay slot)" if r.in_slot else "")
                  + (f", predicted {r.pred}" if r.pred else "")
                  + (f", ready in {r.ready_in}" if r.ready_in else "") for r in rows]
        if not rows:
            lines.append("empty")
        return PartInfo(part, None, lines)
    if name == "predict":
        return PartInfo(part, None, cv.format_predictors(p, limit=8).splitlines())
    if name == "units":
        return PartInfo(part, None, [f"{r.unit}: {'BUSY' if r.busy else 'free'} — {r.state}"
                                     for r in cv.unit_rows(p)])
    hz = cv.hazard_rows(p)
    return PartInfo(part, None, [
        f"{h.register}" + ("" if h.old is None else f" 0x{h.old:08x} -> 0x{h.new:08x}")
        + f" by {h.producer}: seen from cycle {h.visible['other']} "
        f"(loads / stores {h.visible['lsu']}, FPU {h.visible['cop']})" for h in hz]
        or ["no CP0 write on its way"])


STAGES_24K = ("IF", "IS", "IT", "RF", "AG", "EX", "MS", "ER", "WB")


def _stage24(s, st: str, d: dict, ev: dict, p) -> list[str]:
    v = d.get("view")
    out = [f"0x{d['pc']:08x}: {d['text']}"]
    if st == "IF":
        for t in tlb_lines(d.get("tlb") or []):
            out.append(f"translation: {t.strip()}")
        out += [a.describe() for a in d.get("cache") or []]
        if d.get("wait"):
            out.append(f"waiting for the line: {d['wait']} more cycle(s)")
    if st in ("IS", "IT") and p.last_prediction is not None \
            and p.last_prediction[0] == d["pc"]:
        _, _, how, nxt = p.last_prediction
        out.append(f"predicted: {how} -> " + (f"0x{nxt:08x}" if nxt is not None else "wait"))
    if st == "RF":
        fields = fields_of(s, d)
        regs = [f"{f.name} = {f.meaning}" for f in fields
                if f.name in ("rs", "rt", "base", "index", "fs", "ft")]
        out.append("reads: " + (", ".join(regs) or "no register operands"))
    if st == "AG" and v is not None and v.address is not None:
        a = v.address
        out.append(f"address = {a.base.source} 0x{a.base.value:08x} + {a.offset} = "
                   f"0x{a.address:08x}")
    if st == "EX" and v is not None:
        if v.alu is not None:
            out.append(f"ALU: {v.alu.op}"
                       + (f" = 0x{v.alu.result:08x}" if v.alu.result is not None else ""))
            for name, op in (("a", v.alu.a), ("b", v.alu.b)):
                if op is not None:
                    out.append(f"  {name} = 0x{op.value:08x} ({op.source})")
        if v.branch is not None:
            out.append(f"branch: {v.branch.condition}: "
                       + ("taken" if v.branch.taken else "not taken"))
            if p.last_resolution is not None and p.last_resolution[0] == d["pc"]:
                out.append("predicted right" if p.last_resolution[3] else "MISPREDICTED: "
                           "the younger instructions are killed, the fetch redirected")
        if v.exception:
            out.append(f"EXCEPTION {v.exception} (taken when it reaches ER)")
    if st == "MS":
        if v is not None:
            for addr, n, val in v.mem_reads:
                out.append(f"read  0x{addr:08x} = 0x{val:0{2 * n}x}")
            for addr, n, old, new in v.mem_writes:
                out.append(f"write 0x{addr:08x}: 0x{old:0{2 * n}x} -> 0x{new:0{2 * n}x}")
        for t in tlb_lines(d.get("tlb") or []):
            out.append(f"translation: {t.strip()}")
        out += [a.describe() for a in d.get("cache") or []]
        if d.get("wait"):
            out.append(f"waiting: {d['wait']} more cycle(s)")
    if st == "ER":
        out += [e for e in ev.get("events") or [] if "exception" in e or "interrupt" in e
                or "NMI" in e]
    if st == "WB" and v is not None:
        out += [f"{name} <- 0x{new:08x}" for name, _old, new in v.reg_writes]
    return out
