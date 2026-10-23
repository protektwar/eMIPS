"""The control logic of the datapath as gates (the GUI: click Control or
the ALU): the **main control unit** and the **ALU control**, as PLAs for
the full MIPS32 instruction set of a release.  Qt-free.

Real MIPS cores' decoders are not published; Patterson & Hennessy draw
the textbook subset's as PLAs (Appendix C/D).  eMIPS builds the same kind
of logic for every MIPS32 instruction, generated from the emulator's own
decode — the signals the pipeline uses (:func:`emips.pipeline.
control_signals`) and the ALU functions (:mod:`emips.alulogic`) — and
minimized (:mod:`emips.logic`); ``tests/test_controllogic.py`` checks the
gates against every instruction row of every release.

Main control, as the opcode map is laid out: an **opcode PLA** (op5..op0)
makes the signals of every opcode whose instructions all share them, and
for the others (SPECIAL, REGIMM, SPECIAL2 / 3, COP0 / 1, R6's POP groups
...) an *escape* line that enables a **secondary PLA** on the field
that tells them apart (funct, rt, rs, ...; R6's compact branches by
comparators: rs = 0, rt = 0, rs = rt, rs < rt).  Signals: RegDst1 / 0
(00 rt, 01 rd, 10 $ra), ALUSrc, MemtoReg, RegWrite, MemRead, MemWrite,
Branch, Jump and a 4-bit ALUOp (the textbook's 2 bits cannot name all of
MIPS32's ALU work):

====  ===========================  ====  ========================
0000  add (loads, stores, addiu)   0101  and (andi)
0001  sub (beq, bne)               0110  or  (ori)
0010  R-type: funct decides        0111  xor (xori)
0011  slt  (slti)                  1000  lui (shift left 16)
0100  sltu (sltiu)                 1111  not the ALU (another unit)
====  ===========================  ====  ========================

ALU control: a PLA from ALUOp, funct and the two rotate bits (ROTR: bit
21, ROTRV: bit 6) to Ainvert, Bnegate, Operation2..0 (000 AND, 001 OR,
010 add, 011 Less, 100 XOR, 101 the shifter), Unsigned (sltu's Less),
the shifter's ShRight / ShArith / ShRotate / ShVar / Lui, and OtherUnit
(multiply / divide, bit fields, jumps ...).

MIPS64 (``arch="mips64"``, PLAN M22): the same logic for the MIPS64 rows
of the release, for a 64-bit ALU.  Its ALU control has two more lines:
**Word** (a 32-bit operation — ADDU, SUBU, the word shifts, LUI: the
result is bit 31 sign-extended, an overflow is bit 31's; MIPS64 Vol. II
"the 32-bit result is sign-extended") and **Sh32** (DSLL32 ...: the
shift amount + 32); ALUOp 1001 = add word (ADDI, ADDIU, AUI — the
address of a load / store, DADDIU and DADDI add 64 bits: 0000).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .isa import ISA
from .logic import PLA, build

M32 = 0xFFFFFFFF
#: alias rows that refine another row's encoding (decode picks them first)
ALIASES = {"B", "BAL", "NAL", "NOP", "SSNOP", "EHB", "PAUSE"}
FIELDS = {"rs": (21, 5), "rt": (16, 5), "rd": (11, 5), "sa": (6, 5), "funct": (0, 6)}
SIGNALS = ["RegDst1", "RegDst0", "ALUSrc", "MemtoReg", "RegWrite", "MemRead", "MemWrite",
           "Branch", "Jump", "ALUOp3", "ALUOp2", "ALUOp1", "ALUOp0"]
ESCAPE_NAMES = {0b000000: "SPECIAL", 0b000001: "REGIMM", 0b010000: "COP0",
                0b010001: "COP1", 0b010010: "COP2", 0b010011: "COP1X",
                0b011100: "SPECIAL2", 0b011111: "SPECIAL3"}
#: comparators of R6's compact-branch groups (derived inputs)
COMPARATORS = ["rs=0", "rt=0", "rs=rt", "rs<rt"]
#: the probe patterns for the bits no PLA looks at (registers not $0)
_PATTERNS = (0x0A5294A5, 0x10842108, 0x07FFFFFF, 0x00000000)

# -- the ALU functions --------------------------------------------------------------

#: ALUOp code -> ALU function (0010: the funct field)
ALUOP_FUNCTION = {0b0000: "add", 0b0001: "sub", 0b0011: "slt", 0b0100: "sltu",
                  0b0101: "and", 0b0110: "or", 0b0111: "xor", 0b1000: "lui",
                  0b1111: "other"}
#: ... of a MIPS64 core: 1001 = add word (ADDI / ADDIU / AUI)
ALUOP_FUNCTION64 = {**ALUOP_FUNCTION, 0b1001: "addw"}
_R_FUNCTION = {"ADD": "add", "ADDU": "add", "SUB": "sub", "SUBU": "sub", "AND": "and",
               "OR": "or", "XOR": "xor", "NOR": "nor", "SLT": "slt", "SLTU": "sltu",
               "SLL": "sll", "SRL": "srl", "SRA": "sra", "ROTR": "rotr",
               "SLLV": "sllv", "SRLV": "srlv", "SRAV": "srav", "ROTRV": "rotrv"}
ALU_OUTPUTS = ["Ainvert", "Bnegate", "Op2", "Op1", "Op0", "Unsigned", "ShRight", "ShArith",
               "ShRotate", "ShVar", "Lui", "OtherUnit"]
#: ... of a MIPS64 core's 64-bit ALU: + Word (a 32-bit operation, its result
#: bit 31 sign-extended) and Sh32 (the shift amount + 32)
ALU_OUTPUTS64 = ALU_OUTPUTS + ["Word", "Sh32"]
#: R-type rows -> ALU function on MIPS64: the 32-bit operations are word
#: ones ("addw", "sll.w" ...), the doubleword ones the 64-bit ALU's own
_R_FUNCTION64 = {
    **{k: v + ".w" if v in ("sll", "srl", "sra", "rotr", "sllv", "srlv", "srav", "rotrv")
       else v for k, v in _R_FUNCTION.items()},
    "ADD": "addw", "ADDU": "addw", "SUB": "subw", "SUBU": "subw",
    "DADD": "add", "DADDU": "add", "DSUB": "sub", "DSUBU": "sub",
    "DSLL": "sll", "DSRL": "srl", "DSRA": "sra", "DROTR": "rotr",
    "DSLL32": "sll32", "DSRL32": "srl32", "DSRA32": "sra32", "DROTR32": "rotr32",
    "DSLLV": "sllv", "DSRLV": "srlv", "DSRAV": "srav", "DROTRV": "rotrv"}
#: the word forms of the ALU functions
_WORD = {"addw": "add", "subw": "sub"}
#: ALU function -> its control lines (Ainvert Bnegate Op2..0 Unsigned, shifter flags)
_LINES = {
    "and": (0, 0, 0b000), "or": (0, 0, 0b001), "add": (0, 0, 0b010), "sub": (0, 1, 0b010),
    "slt": (0, 1, 0b011), "sltu": (0, 1, 0b011), "nor": (1, 1, 0b000), "xor": (0, 0, 0b100),
}
_SHIFTS = {"sll": (0, 0, 0, 0), "srl": (1, 0, 0, 0), "sra": (1, 1, 0, 0), "rotr": (1, 0, 1, 0),
           "sllv": (0, 0, 0, 1), "srlv": (1, 0, 0, 1), "srav": (1, 1, 0, 1),
           "rotrv": (1, 0, 1, 1), "lui": (0, 0, 0, 0)}


def alu_lines(function: str, wide: bool = False) -> dict[str, bool]:
    """The ALU control's outputs for an ALU function (``wide``: a MIPS64
    core's, with Word and Sh32)."""
    word = sh32 = False
    if function in _WORD:
        function, word = _WORD[function], True
    elif function.endswith(".w"):
        function, word = function[:-2], True
    elif function.endswith("32") and function[:-2] in _SHIFTS:
        function, sh32 = function[:-2], True
    word = word or (wide and function == "lui")      # (lui: a word, sign-extended)
    out = {o: False for o in (ALU_OUTPUTS64 if wide else ALU_OUTPUTS)}
    if wide:
        out.update(Word=word, Sh32=sh32)
    if function in _LINES:
        ainv, bneg, op = _LINES[function]
        out.update(Ainvert=bool(ainv), Bnegate=bool(bneg), Op2=bool(op & 4),
                   Op1=bool(op & 2), Op0=bool(op & 1), Unsigned=function == "sltu")
    elif function in _SHIFTS:
        r, a, rot, var = _SHIFTS[function]
        out.update(Op2=True, Op0=True, ShRight=bool(r), ShArith=bool(a), ShRotate=bool(rot),
                   ShVar=bool(var), Lui=function == "lui")
    else:
        out["OtherUnit"] = True
    return out


_ADD = {"ADDI", "ADDIU", "AUI"}
#: MIPS64's 64-bit immediate adds (ALUOp add; the 32-bit ones are add word)
_DADD = {"DADDI", "DADDIU", "DAUI"}
_IMM_CODE = {"SLTI": 0b0011, "SLTIU": 0b0100, "ANDI": 0b0101, "ORI": 0b0110, "XORI": 0b0111,
             "LUI": 0b1000}


def aluop_code(dec, wide: bool = False) -> int:
    """The main control's ALUOp for a decoded instruction (``wide``: on a
    MIPS64 core, where ADDI / ADDIU / AUI add a word)."""
    name, fam = dec.name, dec.family
    if dec.word >> 26 == 0:
        return 0b0010
    if name in _IMM_CODE:
        return _IMM_CODE[name]
    if wide and name in _ADD:
        return 0b1001
    if name in _ADD or wide and name in _DADD or fam in ("load", "store", "msa_mem") or name in (
            "LL", "SC", "LL@R6", "SC@R6", "PREF", "CACHE", "LWC1", "SWC1", "LDC1", "SDC1",
            "LWC2", "SWC2", "LDC2", "SDC2"):
        return 0b0000
    if name in ("BEQ", "BNE", "BEQL", "BNEL"):
        return 0b0001
    return 0b1111


def main_signals(isa: ISA, dec) -> dict[str, bool]:
    """The main control's signals for a decoded instruction: those of its
    instruction row (the pipeline's control signals + ALUOp), the same
    whatever its register numbers (a write to $0 is still RegWrite)."""
    return _row_signals(isa, dec.name)


@lru_cache(maxsize=None)
def _probe_word(isa: ISA, name: str) -> int:
    """A word of row ``name`` with non-zero register fields."""
    e = isa.get(name)
    mask, value = e["decode"]["mask"], e["decode"]["value"]
    for pattern in _PATTERNS + (0x0042_1084, 0x0022_1084, 0x0062_1084):
        w = value | (pattern & ~mask & M32)
        try:
            if isa.decode(w).name == name:
                return w
        except Exception:                                # noqa: BLE001
            pass
    for rs in range(1, 32):                              # R6: rs / rt relations
        for rt in range(1, 32):
            w = (value & ~((31 << 21) | (31 << 16))) | (rs << 21) | (rt << 16)
            w = w | (_PATTERNS[0] & ~mask & ~((31 << 21) | (31 << 16)) & M32)
            try:
                if isa.decode(w).name == name:
                    return w
            except Exception:                            # noqa: BLE001
                pass
    raise ValueError(f"no word decodes to {name}")


@lru_cache(maxsize=None)
def _row_signals(isa: ISA, name: str) -> dict[str, bool]:
    from .pipeline import analyze, control_signals
    dec = isa.decode(_probe_word(isa, name))
    c = control_signals(dec, analyze(dec))
    dst = {"rt": 0, "rd": 1, "$ra": 2, "rd6": 3, "-": 0}[c["RegDst"]]   # (rd6: MSA's 10..6)
    code = aluop_code(dec, isa.arch == "mips64")
    return {"RegDst1": bool(dst & 2), "RegDst0": bool(dst & 1),
            "ALUSrc": c["ALUSrc"] == "imm", "MemtoReg": bool(c["MemtoReg"]),
            "RegWrite": bool(c["RegWrite"]), "MemRead": bool(c["MemRead"]),
            "MemWrite": bool(c["MemWrite"]), "Branch": bool(c["Branch"]),
            "Jump": bool(c["Jump"]), "ALUOp3": bool(code & 8), "ALUOp2": bool(code & 4),
            "ALUOp1": bool(code & 2), "ALUOp0": bool(code & 1)}


def _decode(isa: ISA, word: int):
    try:
        dec = isa.decode(word)
    except Exception:                                    # noqa: BLE001
        return None
    return None if dec.name in ALIASES else dec


def _bits(word: int, fields: list[str]) -> int:
    """The input vector of ``fields`` (or comparators) of a word."""
    x = 0
    for f in fields:
        if f in FIELDS:
            lsb, n = FIELDS[f]
            x = (x << n) | ((word >> lsb) & ((1 << n) - 1))
        else:
            rs, rt = (word >> 21) & 31, (word >> 16) & 31
            bit = {"rs=0": rs == 0, "rt=0": rt == 0, "rs=rt": rs == rt,
                   "rs<rt": rs < rt}[f]
            x = (x << 1) | int(bit)
    return x


def input_names(fields: list[str]) -> list[str]:
    out = []
    for f in fields:
        if f in FIELDS:
            n = FIELDS[f][1]
            out += [f"{'F' if f == 'funct' else f}{i}" for i in range(n - 1, -1, -1)]
        else:
            out.append(f)
    return out


# -- the main control ------------------------------------------------------------------

@dataclass
class Secondary:
    escape: str
    opcode: int
    fields: list[str]
    pla: PLA


@dataclass
class MainControl:
    release: str
    opcode: PLA
    secondary: dict[str, Secondary]
    arch: str = "mips32"


#: the fields tried, in order, to tell an escape opcode's instructions apart
_CANDIDATES = (["funct"], ["rt"], ["rs"], ["rs", "funct"], ["rt", "funct"],
               ["sa", "funct"], COMPARATORS, ["funct"] + COMPARATORS,
               ["rs", "rt", "funct"])        # (MSA: the minor opcode + ELM's 25..16)


def _field_values(rows: list, field: str) -> set[int]:
    """Values a field takes in probe words: the ones the templates of
    ``rows`` fix (their free bits from the patterns), and the patterns'."""
    lsb, n = FIELDS[field]
    fmask = ((1 << n) - 1) << lsb
    p = _PATTERNS[0]
    out = {(p & fmask) >> lsb}
    if field in ("rs", "rt"):
        out.add(0)                                       # $zero: R6's relations
    for e in rows:
        mask, value = e["decode"]["mask"], e["decode"]["value"]
        if mask & fmask:
            out.add((((value & mask) | (p & ~mask)) & fmask) >> lsb)
    return out


def _secondary_table(isa: ISA, op: int, fields: list[str], rows: list
                     ) -> dict[int, dict[str, bool]] | None:
    """The truth of an escape opcode over ``fields`` (None when they do not
    tell its instructions apart): every value of the input fields, with
    every value the instructions' templates give the other fields."""
    from itertools import product
    comparators = any(f not in FIELDS for f in fields)
    inputs = [f for f in fields if f in FIELDS]
    if comparators:
        inputs += [f for f in ("rs", "rt") if f not in inputs]
    others = [f for f in FIELDS if f not in inputs]
    choices = [[(f, v) for v in range(1 << FIELDS[f][1])] for f in inputs] + \
              [[(f, v) for v in sorted(_field_values(rows, f))] for f in others]
    table: dict[int, dict[str, bool]] = {}
    for combo in product(*choices):
        w = op << 26
        for f, v in combo:
            w |= v << FIELDS[f][0]
        dec = _decode(isa, w)
        if dec is None:
            continue
        sig = main_signals(isa, dec)
        x = _bits(w, fields)
        old = table.get(x)
        if old is not None and old != sig:
            return None
        table[x] = sig
    return table


@lru_cache(maxsize=None)
def main_control(release: str = "R5", arch: str = "mips32", msa: bool = False) -> MainControl:
    """The main control unit of a release (``arch="mips64"``: of MIPS64,
    the doubleword rows too; ``msa``: with MSA's rows), as PLAs."""
    isa = ISA.for_release(release, arch=arch, msa=msa)
    rows = [e for e in isa.leads if isa.in_release(e["name"]) and e["name"] not in ALIASES]
    by_op: dict[int, list] = {}
    for e in rows:
        by_op.setdefault(e["decode"]["value"] >> 26, []).append(e)
    # the signals of every opcode (probe words): one set, or an escape
    opcode_sig: dict[int, dict[str, bool]] = {}
    escapes: dict[int, str] = {}
    for op, es in sorted(by_op.items()):
        seen = []
        for e in es:
            for pattern in _PATTERNS:
                w = e["decode"]["value"] | (pattern & ~e["decode"]["mask"] & M32)
                dec = _decode(isa, w)
                if dec is not None and dec.name == e["name"]:
                    seen.append(main_signals(isa, dec))
                    break
        if not seen:
            continue
        if all(s == seen[0] for s in seen):
            opcode_sig[op] = seen[0]
        else:
            escapes[op] = ESCAPE_NAMES.get(op, f"OP{op:06b}")
    names = [escapes[op] for op in sorted(escapes)]
    table = {}
    for op in range(64):
        if op in escapes:                    # the secondary makes the signals
            table[op] = dict({o: False for o in SIGNALS}, **{n: n == escapes[op] for n in names})
        elif op in opcode_sig:
            table[op] = dict(opcode_sig[op], **{n: False for n in names})
    opcode = build("opcode decoder", [f"op{i}" for i in range(5, -1, -1)],
                   SIGNALS + names, table)
    secondary = {}
    for op, name in sorted(escapes.items()):
        constrained = any(e.get("constraint") for e in by_op[op])
        for fields in _CANDIDATES:
            if constrained and not any(f in COMPARATORS for f in fields):
                continue                       # rs / rt relations: comparators
            t = _secondary_table(isa, op, list(fields), by_op[op])
            if t is not None:
                break
        else:                                            # pragma: no cover
            raise ValueError(f"no field set tells opcode {op:06b} apart")
        pla = build(f"{name} decoder", input_names(list(fields)), SIGNALS, t)
        secondary[name] = Secondary(name, op, list(fields), pla)
    return MainControl(release, opcode, secondary, arch)


@dataclass
class MainEval:
    word: int
    opcode_terms: list[bool]
    opcode_out: dict[str, bool]
    escape: Secondary | None
    secondary_terms: list[bool]
    secondary_out: dict[str, bool]
    signals: dict[str, bool]            # the final signals (opcode OR secondary)

    @property
    def aluop(self) -> int:
        return sum(int(self.signals[f"ALUOp{i}"]) << i for i in range(4))


def evaluate_main(mc: MainControl, word: int) -> MainEval:
    op = word >> 26
    hit, out = mc.opcode.evaluate(op)
    esc = next((s for s in mc.secondary.values() if out.get(s.escape)), None)
    s_hit: list[bool] = []
    s_out = {o: False for o in SIGNALS}
    if esc is not None:
        s_hit, s_out = esc.pla.evaluate(_bits(word, esc.fields))
    final = {o: out[o] or s_out.get(o, False) for o in SIGNALS}
    return MainEval(word, hit, out, esc, s_hit, s_out, final)


# -- the ALU control ------------------------------------------------------------------

ALU_INPUTS = ["ALUOp3", "ALUOp2", "ALUOp1", "ALUOp0", "F5", "F4", "F3", "F2", "F1", "F0",
              "B21", "B6"]


def alu_input(aluop: int, word: int) -> int:
    return (aluop << 8) | ((word & 0x3F) << 2) | (((word >> 21) & 1) << 1) | ((word >> 6) & 1)


def alu_function(isa: ISA, aluop: int, word: int) -> str | None:
    """The ALU function an (ALUOp, word) asks for (None: don't care)."""
    wide = isa.arch == "mips64"
    if aluop == 0b0010:
        dec = _decode(isa, word)
        if dec is None:
            return None
        return (_R_FUNCTION64 if wide else _R_FUNCTION).get(dec.name, "other")
    return (ALUOP_FUNCTION64 if wide else ALUOP_FUNCTION).get(aluop)


@lru_cache(maxsize=None)
def alu_control(release: str = "R5", arch: str = "mips32", msa: bool = False) -> PLA:
    """The ALU control of a release (``arch="mips64"``: a MIPS64 core's,
    with Word and Sh32; ``msa``: MSA's LSA / DLSA before Release 6), as a
    PLA."""
    isa = ISA.for_release(release, arch=arch, msa=msa)
    wide = arch == "mips64"
    functions = ALUOP_FUNCTION64 if wide else ALUOP_FUNCTION
    r_function = _R_FUNCTION64 if wide else _R_FUNCTION
    table: dict[int, dict[str, bool]] = {}
    for aluop in functions:                      # independent of funct / B21 / B6
        lines = alu_lines(functions[aluop], wide)
        for rest in range(256):
            table[(aluop << 8) | rest] = lines
    special = [e for e in isa.leads if isa.in_release(e["name"]) and
               e["name"] not in ALIASES and e["decode"]["value"] >> 26 == 0]
    for e in special:                             # R-type: the funct field (+ R bits)
        mask, base = e["decode"]["mask"], _probe_word(isa, e["name"])
        for b21 in (0, 1):
            for b6 in (0, 1):
                w = (base & ~((1 << 21) | (1 << 6))) | (b21 << 21) | (b6 << 6)
                if (w ^ base) & mask:
                    continue                      # the template fixes that bit
                dec = _decode(isa, w)
                if dec is None or dec.name != e["name"]:
                    continue
                x = alu_input(0b0010, w)
                lines = alu_lines(r_function.get(dec.name, "other"), wide)
                if table.get(x, lines) != lines:              # pragma: no cover
                    raise ValueError(f"ALU control: {dec.name} is ambiguous")
                table[x] = lines
    return build("ALU control", ALU_INPUTS, ALU_OUTPUTS64 if wide else ALU_OUTPUTS, table)


def operation_name(out: dict[str, bool]) -> str:
    op = (int(out["Op2"]) << 2) | (int(out["Op1"]) << 1) | int(out["Op0"])
    if out["OtherUnit"]:
        return "not the ALU: another unit"
    return {0: "AND", 1: "OR", 2: "add", 3: "Less (set on less than)", 4: "XOR",
            5: "the shifter"}.get(op, "?")
