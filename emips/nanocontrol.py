"""The control logic of a nanoMIPS core's datapath as gates (PLAN M23
phase 4f; the GUI: click Control or the ALU): the **main control unit**
and the **ALU control**, as PLAs, generated from the formats of
:mod:`emips.nanomips` as :mod:`emips.controllogic` does for MIPS32.  Qt-free.

The I7200's decoder is not published either.  Its opcode map (MD01247
App. B) is laid out for one: the **major opcode** — bits 15..10 of the
first halfword, Tables B.2 / B.3 — names a format or a *pool* of formats,
and each pool tells its formats apart by more fixed bits (Tables B.4 ...)
and, for a few, a condition on a field (A.4's "with ...": MOVE "with
rt != 0", BEQC[16] "with rs3 < rt3 and u != 0", PREF "with hint != 31").
So the main control is an **opcode PLA** on op5..op0 making the signals
of every major opcode whose formats share them, and an *escape* line for
each pool that enables a **secondary PLA**; a pool's inputs are the fixed
bits its formats use (named by the instruction's bit numbers, the first
halfword the most significant: ``b2`` = bit 2 of a 32-bit instruction,
in its second halfword) and a comparator per condition.  A pool's table
is too large to list point by point (up to 20 inputs), so it is built as
cubes — each format's fixed bits, minus the more specific formats the
decoder tries first (:func:`emips.logic.sharp`) — and minimized against
them (:func:`emips.logic.build_cubes`); undefined encodings are don't
cares, as in the MIPS32 logic.

Signals (those of :func:`emips.pipeline.control_signals`):

* **RegDst2..0** — which field names the register written: there is no
  single rd / rt pair, each format has its own (``REGDST``): 000 rt (bits
  9..5 of the first halfword, in every size), 001 rd (32-bit: bits
  15..11), 010 rt3 / 011 rd3 (16-bit: 3 bits through MD01247 4.7's
  ``gpr3`` map), 100 rt4 (``gpr4``), 101 $ra (the links), 110 two
  registers (MOVEP, MOVE.BALC, LLWP), 111 a register list (SAVE /
  RESTORE / LWM: the load / store unit walks it);
* ALUSrc (operand b an immediate), MemtoReg, RegWrite, MemRead, MemWrite,
  Branch, Jump;
* **ALUOp3..0** — the ALU function itself (``ALUOP_FUNCTION``): nanoMIPS
  has no funct field common to its register formats (P32A's are bits
  9..3, P16C's bits 1..0 ...), so the pools' PLAs name the operation and
  the ALU control only turns it into the ALU's lines.

ALU control: a PLA from ALUOp and ALUSrc (a shift by a register: ShVar)
to the lines of :mod:`emips.alulogic`'s ALU (``controllogic.ALU_OUTPUTS``).
``tests/test_nanocontrol.py`` checks both against every format.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

from . import nanomips as NM
from .controllogic import ALU_OUTPUTS, alu_lines
from .logic import PLA, Term, build, build_cubes, sharp

SIGNALS = ["RegDst2", "RegDst1", "RegDst0", "ALUSrc", "MemtoReg", "RegWrite", "MemRead",
           "MemWrite", "Branch", "Jump", "ALUOp3", "ALUOp2", "ALUOp1", "ALUOp0"]
#: RegDst code -> the field naming the register written
REGDST = {0: "rt", 1: "rd", 2: "rt3", 3: "rd3", 4: "rt4", 5: "$ra", 6: "two registers",
          7: "a register list"}
_REGDST_CODE = {v: k for k, v in REGDST.items()}
#: ALUOp code -> the ALU function (controllogic.alu_lines' names)
ALUOP_FUNCTION = {0b0000: "add", 0b0001: "sub", 0b0010: "and", 0b0011: "or",
                  0b0100: "xor", 0b0101: "nor", 0b0110: "slt", 0b0111: "sltu",
                  0b1000: "sll", 0b1001: "srl", 0b1010: "sra", 0b1011: "rotr",
                  0b1111: "other"}
#: ALUOp code -> what the main control asks for
ALUOP_WHY = {0b0000: "add: ADDU, ADDIU, ADDIUPC, every load / store address",
             0b0001: "subtract: SUBU, SUB", 0b0010: "and: AND, ANDI",
             0b0011: "or: OR, ORI; LI / LUI (a = 0: the immediate passes)",
             0b0100: "xor: XOR, XORI", 0b0101: "nor: NOR, NOT (b = 0)",
             0b0110: "slt: SLT, SLTI", 0b0111: "sltu: SLTU, SLTIU",
             0b1000: "shift left: SLL, SLLV", 0b1001: "shift right logical: SRL, SRLV",
             0b1010: "shift right arithmetic: SRA, SRAV", 0b1011: "rotate right: ROTR, ROTRV",
             0b1111: "not the ALU"}
_FUNCTION = {"ADD": "add", "ADDU": "add", "ADDIU": "add", "ADDIUPC": "add",
             "SUB": "sub", "SUBU": "sub", "AND": "and", "ANDI": "and",
             "OR": "or", "ORI": "or", "LI": "or", "LUI": "or", "XOR": "xor", "XORI": "xor",
             "NOR": "nor", "NOT": "nor", "SLT": "slt", "SLTI": "slt", "SLTU": "sltu",
             "SLTIU": "sltu", "SLL": "sll", "SLLV": "sll", "SRL": "srl", "SRLV": "srl",
             "SRA": "sra", "SRAV": "sra", "ROTR": "rotr", "ROTRV": "rotr"}
_CODE = {f: c for c, f in ALUOP_FUNCTION.items()}
#: the pages whose registers written are a list
_LISTS = frozenset(("SAVE", "RESTORE", "LWM", "UALWM"))
_PATTERNS = (0x0A5294A5A5A5, 0x108421084210, 0x07FFFFFFFFFF, 0x3C3C3C3C3C3C,
             0x5A5A5A5A5A5A)


def first_halfword(raw: int, size: int) -> int:
    return (raw >> (8 * size - 16)) & 0xFFFF


def major(raw: int, size: int) -> int:
    """The major opcode: bits 15..10 of the first halfword."""
    return first_halfword(raw, size) >> 10


# -- a format's signals -----------------------------------------------------------

@lru_cache(maxsize=None)
def probe(row: NM.Row) -> NM.NM:
    """An instruction of format ``row`` with non-zero register fields."""
    full = (1 << (8 * row.size)) - 1
    rng = random.Random(row.form)
    for k in range(400):
        p = _PATTERNS[k] if k < len(_PATTERNS) else rng.getrandbits(48)
        raw = row.value | (p & ~row.mask & full)
        hit = NM.TABLE.match(raw, row.size)
        if hit is not None and hit[0] is row:
            return NM.decode_raw(raw, row.size)
    raise ValueError(f"no encoding decodes to {row.form}")      # pragma: no cover


class _Seen(dict):
    """The decoded fields, recording which ones are read."""

    def __init__(self, f: dict) -> None:
        super().__init__(f)
        self.read: list[str] = []

    def __getitem__(self, key):
        self.read.append(key)
        return super().__getitem__(key)


def _roots(row: NM.Row, arg: str) -> set[str]:
    """The encoded fields an argument is decoded from (empty: a constant)."""
    roots: dict[str, set[str]] = {n: {n} for n, _ in row.fields}
    for target, _fn, a, const in row.steps:
        src: set[str] = set()
        for x in (a, const):
            if isinstance(x, str):
                src |= roots.get(x, set())
        roots[target] = src
    return roots.get(arg, set())


def regdst(row: NM.Row) -> int | None:
    """The RegDst code of a format (None: it writes no register)."""
    from .simnano import _DSP_TWINS, _USES
    _reads, writes = _USES[row.name]
    if row.name in _DSP_TWINS:                  # the DSP ASE: by the register numbers
        regs = writes(NM.decode_raw(NM.encode(row, {"rt": 5, "rs": 6, "rd": 7}), 4).f)
        return _REGDST_CODE[{(5,): "rt", (7,): "rd"}[regs]] if regs else None
    f = _Seen(probe(row).f)
    regs = writes(f)
    if not regs:
        return None
    if row.name in _LISTS:
        return _REGDST_CODE["a register list"]
    if len(regs) > 1:
        return _REGDST_CODE["two registers"]
    fields = set().union(set(), *(_roots(row, a) for a in f.read))
    if not fields and regs == (31,):
        return _REGDST_CODE["$ra"]                  # (a constant: dst = 31)
    if len(fields) != 1:                            # pragma: no cover
        raise ValueError(f"{row.form}: the register written comes from {sorted(fields)}")
    name = fields.pop()
    return _REGDST_CODE[{"rtz4": "rt4"}.get(name, name)]


def aluop(row: NM.Row) -> int:
    """The ALU function's code of a format (``ALUOP_FUNCTION``)."""
    from .tracenano import _MEMORY
    if row.name in _MEMORY:
        return _CODE["add"]                     # the effective address
    return _CODE[_FUNCTION.get(row.name, "other")]


@lru_cache(maxsize=None)
def row_signals(row: NM.Row) -> dict[str, bool]:
    """The main control's signals for a format — the pipeline's control
    signals of an instance (:func:`emips.pipeline.control_signals`), the
    same whatever its register numbers (a write to $0 is still RegWrite)."""
    from .pipeline import LATENCY, _analyze_nano, control_signals
    from .sim import Decode
    from .simnano import ENTRIES, entry_name
    i = probe(row)
    c = control_signals(Decode(i.raw, ENTRIES[entry_name(row)]), _analyze_nano(i, LATENCY))
    dst = regdst(row)
    code = aluop(row)
    return {"RegDst2": bool((dst or 0) & 4), "RegDst1": bool((dst or 0) & 2),
            "RegDst0": bool((dst or 0) & 1), "ALUSrc": c["ALUSrc"] == "imm",
            "MemtoReg": bool(c["MemtoReg"]), "RegWrite": dst is not None,
            "MemRead": bool(c["MemRead"]), "MemWrite": bool(c["MemWrite"]),
            "Branch": bool(c["Branch"]), "Jump": bool(c["Jump"]),
            "ALUOp3": bool(code & 8), "ALUOp2": bool(code & 4), "ALUOp1": bool(code & 2),
            "ALUOp0": bool(code & 1)}


def signals_of(sig: dict[str, bool]) -> tuple[int | None, int]:
    """(RegDst, ALUOp) of a signal set (RegDst None when RegWrite is 0)."""
    dst = (int(sig["RegDst2"]) << 2) | (int(sig["RegDst1"]) << 1) | int(sig["RegDst0"])
    code = sum(int(sig[f"ALUOp{k}"]) << k for k in range(4))
    return (dst if sig["RegWrite"] else None), code


# -- the main control ------------------------------------------------------------------

@dataclass(frozen=True)
class Input:
    """A pool PLA's input: bit ``bit`` of the instruction, or the
    condition ``cond`` of format ``row`` (a comparator)."""
    name: str
    bit: int | None = None
    cond: str | None = None
    row: NM.Row | None = None

    def value(self, raw: int) -> int:
        if self.bit is not None:
            return (raw >> self.bit) & 1
        return int(bool(self.row.when(self.row.extract(raw))))


@dataclass
class Secondary:
    escape: str                 # the pool's name (MD01247 Tables B.2 / B.3)
    major: int
    size: int
    inputs: list[Input]
    pla: PLA

    @property
    def fields(self) -> list[str]:
        return [i.name for i in self.inputs]

    def x(self, raw: int) -> int:
        """The PLA's input vector for an instruction (input 0 = the MSB)."""
        v = 0
        for i in self.inputs:
            v = (v << 1) | i.value(raw)
        return v


@dataclass
class MainControl:
    opcode: PLA
    secondary: dict[str, Secondary]
    release: str = "R6"
    arch: str = "nanomips"


def _pool(name: str, maj: int, size: int, rows: list[NM.Row]) -> Secondary:
    """The secondary PLA of a pool: its fixed bits below the major opcode
    and a comparator per condition; the formats' cubes in the decoder's
    order (most specific first: :class:`emips.nanomips._Table`)."""
    width = 8 * size
    below = (1 << (width - 6)) - 1
    used = 0
    for r in rows:
        used |= r.mask & below
    inputs = [Input(f"b{b}", bit=b) for b in range(width - 7, -1, -1) if used >> b & 1]
    conds: list[Input] = []
    for r in rows:
        if r.cond is not None and all(c.cond != r.cond for c in conds):
            conds.append(Input(r.cond, cond=r.cond, row=r))
    inputs += conds
    n = len(inputs)

    def cube(r: NM.Row) -> Term:
        value = mask = 0
        for k, inp in enumerate(inputs):
            pos = 1 << (n - 1 - k)
            if inp.bit is not None and r.mask >> inp.bit & 1:
                mask |= pos
                value |= pos if r.value >> inp.bit & 1 else 0
            elif inp.cond is not None and inp.cond == r.cond:
                mask |= pos
                value |= pos
        return value, mask

    table: list[tuple[Term, dict[str, bool]]] = []
    before: list[Term] = []
    for r in rows:
        c = cube(r)
        region = [c]
        for b in before:
            region = [piece for part in region for piece in sharp(part, b)]
        before.append(c)
        sig = row_signals(r)
        table += [(piece, sig) for piece in region]
    pla = build_cubes(f"{name} decoder", [i.name for i in inputs], SIGNALS, table)
    return _prune(Secondary(name, maj, size, inputs, pla))


def _prune(sec: Secondary) -> Secondary:
    """Drop the inputs no product term reads."""
    pla, n = sec.pla, sec.pla.n
    keep = [k for k in range(n) if any(t[1] >> (n - 1 - k) & 1 for t in pla.terms)]
    if len(keep) == n:
        return sec

    def squeeze(x: int) -> int:
        v = 0
        for k in keep:
            v = (v << 1) | (x >> (n - 1 - k) & 1)
        return v

    terms = [(squeeze(v), squeeze(m)) for v, m in pla.terms]
    new = PLA(pla.name, [pla.inputs[k] for k in keep], pla.outputs, terms, pla.uses)
    return Secondary(sec.escape, sec.major, sec.size, [sec.inputs[k] for k in keep], new)


@lru_cache(maxsize=None)
def main_control() -> MainControl:
    """The main control unit of a nanoMIPS core (every format of
    :data:`emips.nanomips.ROWS`), as PLAs."""
    from .tracenano import MAJOR
    escapes: dict[int, tuple[str, int, list[NM.Row]]] = {}
    opcode_sig: dict[int, dict[str, bool]] = {}
    for (size, maj), rows in sorted(NM.TABLE.by_major.items()):
        sigs = [row_signals(r) for r in rows]
        if all(s == sigs[0] for s in sigs):
            opcode_sig[maj] = sigs[0]
        else:
            escapes[maj] = (MAJOR[maj], size, rows)
    names = [escapes[m][0] for m in sorted(escapes)]
    table = {}
    for maj in range(64):
        if maj in escapes:                       # the pool's PLA makes the signals
            table[maj] = dict({o: False for o in SIGNALS},
                              **{n: n == escapes[maj][0] for n in names})
        elif maj in opcode_sig:
            table[maj] = dict(opcode_sig[maj], **{n: False for n in names})
    opcode = build("major opcode decoder", [f"op{i}" for i in range(5, -1, -1)],
                   SIGNALS + names, table)
    secondary = {name: _pool(name, maj, size, rows)
                 for maj, (name, size, rows) in sorted(escapes.items())}
    return MainControl(opcode, secondary)


@dataclass
class MainEval:
    word: int
    size: int
    opcode_terms: list[bool]
    opcode_out: dict[str, bool]
    escape: Secondary | None
    x: int | None                       # the pool PLA's input vector
    secondary_terms: list[bool]
    secondary_out: dict[str, bool]
    signals: dict[str, bool]            # the final signals (opcode OR pool)

    @property
    def aluop(self) -> int:
        return sum(int(self.signals[f"ALUOp{i}"]) << i for i in range(4))

    @property
    def regdst(self) -> int:
        return sum(int(self.signals[f"RegDst{i}"]) << i for i in range(3))


def evaluate_main(mc: MainControl, raw: int, size: int) -> MainEval:
    maj = major(raw, size)
    hit, out = mc.opcode.evaluate(maj)
    esc = next((s for s in mc.secondary.values() if out.get(s.escape)), None)
    s_hit: list[bool] = []
    s_out = {o: False for o in SIGNALS}
    x = None
    if esc is not None:
        x = esc.x(raw)
        s_hit, s_out = esc.pla.evaluate(x)
    final = {o: out[o] or s_out.get(o, False) for o in SIGNALS}
    return MainEval(raw, size, hit, out, esc, x, s_hit, s_out, final)


# -- the ALU control ------------------------------------------------------------------

ALU_INPUTS = ["ALUOp3", "ALUOp2", "ALUOp1", "ALUOp0", "ALUSrc"]
_SHIFTS = ("sll", "srl", "sra", "rotr")


def alu_function(code: int, alusrc: bool) -> str | None:
    """The ALU function of (ALUOp, ALUSrc): a shift by a register (ALUSrc
    0) is the variable one (ShVar)."""
    f = ALUOP_FUNCTION.get(code)
    if f in _SHIFTS and not alusrc:
        return f + "v"
    return f


def alu_input(code: int, alusrc: bool) -> int:
    return (code << 1) | int(alusrc)


@lru_cache(maxsize=None)
def alu_control() -> PLA:
    """The ALU control of a nanoMIPS core, as a PLA."""
    table = {}
    for code in ALUOP_FUNCTION:
        for src in (False, True):
            table[alu_input(code, src)] = alu_lines(alu_function(code, src))
    return build("ALU control", ALU_INPUTS, ALU_OUTPUTS, table)


def control_for(raw: int, size: int):
    """The ALU control of an instruction (an
    :class:`emips.alulogic.AluControl`): the main control's ALUOp and
    ALUSrc, then the ALU control PLA's lines."""
    from .alulogic import AluControl
    ev = evaluate_main(main_control(), raw, size)
    _, lines = alu_control().evaluate(alu_input(ev.aluop, ev.signals["ALUSrc"]))
    return AluControl(ev.aluop, None, lines, ALUOP_WHY.get(ev.aluop, ""))
