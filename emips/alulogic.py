"""Inside the ALU: the gates of an operation the trace recorded (the GUI:
click the ALU).  Qt-free.

The textbook MIPS ALU (Patterson & Hennessy, Appendix A.5 / Fig. 4.12):
32 one-bit slices, each an AND gate, an OR gate and a full adder whose
inputs can be inverted (Ainvert, Bnegate — Bnegate is also the carry into
bit 0, so a - b = a + ~b + 1), and a 4-input mux choosing AND, OR, the sum
or Less (Operation, 2 bits).  Bit 31's slice also makes Set (the sign of
a - b corrected by overflow, wired to bit 0's Less: slt) and Overflow
(carry into bit 31 XOR carry out); Zero is the NOR of all result bits.
The ALU control turns the main control's ALUOp and the funct field into
those 4 lines.

eMIPS's ALU for the full MIPS32 set extends the slice with an XOR gate
and the barrel shifter's output: Operation has 3 bits (000 AND, 001 OR,
010 add, 011 Less, 100 XOR, 101 the shifter), Unsigned makes Less come
from the carry (sltu); its control lines come from the ALU control PLA
(:mod:`emips.controllogic`), which the main control's ALUOp drives.

What is shown is derived from the operands *the trace recorded* (a, b)
and cross-checked against the result it recorded (:attr:`AluLogic.matches`)
— the view explains the hardware, it never decides a result.  Beyond the
textbook ALU, real MIPS cores have an unsigned compare (sltu: Less = NOT
the carry out of a - b), xor gates, and a barrel shifter (5 stages of 2:1
muxes: 16, 8, 4, 2, 1 places); multiply / divide and the bit-field
instructions are other units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

M32 = 0xFFFFFFFF

#: Operation (3 bits) -> what the slice's result mux passes on
OPERATION = {0: "AND", 1: "OR", 2: "add (the adder's sum)", 3: "Less (set on less than)",
             4: "XOR", 5: "the shifter"}
#: ALUOp code -> what the main control asks for
ALUOP_WHY = {0b0000: "add: a load, a store, addi / addiu",
             0b0001: "subtract: beq / bne",
             0b0010: "R-type: the funct field decides",
             0b0011: "slti", 0b0100: "sltiu", 0b0101: "andi", 0b0110: "ori",
             0b0111: "xori", 0b1000: "lui: shift left 16",
             0b1111: "not the ALU"}


@dataclass
class AluControl:
    aluop: int                   # the main control's 4-bit ALUOp
    funct: str | None            # R-type: bits 5..0
    lines: dict                  # the ALU control's outputs (controllogic.ALU_OUTPUTS)

    @property
    def aluop_why(self) -> str:
        return ALUOP_WHY.get(self.aluop, "")

    @property
    def ainvert(self) -> int:
        return int(self.lines["Ainvert"])

    @property
    def bnegate(self) -> int:
        return int(self.lines["Bnegate"])

    @property
    def operation(self) -> int:
        lines = self.lines
        return (int(lines["Op2"]) << 2) | (int(lines["Op1"]) << 1) | int(lines["Op0"])

    @property
    def unsigned(self) -> int:
        return int(self.lines["Unsigned"])

    @property
    def other(self) -> bool:
        return bool(self.lines["OtherUnit"])

    @property
    def text(self) -> str:
        """``Ainvert Bnegate Op2 Op1 Op0`` as bits."""
        return f"{self.ainvert}{self.bnegate}{self.operation:03b}"


@dataclass
class Slice:
    """One bit of the ALU: its gates' inputs and outputs."""
    i: int
    a: int
    b: int
    a_in: int                    # after Ainvert
    b_in: int                    # after Bnegate
    and_: int
    or_: int
    xor_: int
    cin: int
    sum_: int
    cout: int
    less: int
    shift: int                   # the shifter's output bit
    result: int


@dataclass
class AluLogic:
    instruction: str
    control: AluControl | None
    a: int
    b: int
    result: int | None           # what the gates give
    traced: int | None           # what the trace recorded
    slices: list[Slice] = field(default_factory=list)
    zero: int = 0
    overflow: int | None = None
    set_: int | None = None
    #: the barrel shifter: (stage, value after it)
    shifter: list[tuple[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def matches(self) -> bool:
        return self.result is not None and self.result == self.traced


def _bits(x: int) -> list[int]:
    return [(x >> i) & 1 for i in range(32)]


def control_for(word: int, release: str = "R5") -> AluControl:
    """The ALU control of an instruction word: the main control's ALUOp,
    then the ALU control PLA's lines (:mod:`emips.controllogic`)."""
    from . import controllogic as cl
    ev = cl.evaluate_main(cl.main_control(release), word)
    _, lines = cl.alu_control(release).evaluate(cl.alu_input(ev.aluop, word))
    funct = format(word & 0x3F, "06b") if ev.aluop == 0b0010 else None
    return AluControl(ev.aluop, funct, lines)


def alu_logic(view, release: str = "R5") -> AluLogic | None:
    """The gates of the instruction ``view`` (an :class:`emips.trace.InstrView`)
    executed, or None when it used no ALU operation."""
    if view is None or view.alu is None:
        return None
    alu = view.alu
    a = alu.a.value & M32 if alu.a is not None else 0
    b = alu.b.value & M32 if alu.b is not None else 0
    ctl = control_for(view.word, release)
    out = AluLogic(view.text, ctl, a, b, None, alu.result)
    if ctl.other:
        out.notes.append(f"{view.text}: {alu.op} — done by another unit (the multiply / "
                         "divide unit, the bit-field or count logic ...): the ALU control "
                         "says OtherUnit")
        return out
    shifted = _shift(out, ctl, a, b) if ctl.operation == 5 else None
    _slices(out, ctl, a, b, shifted)
    return out


def _slices(out: AluLogic, ctl: AluControl, a: int, b: int, shifted: int | None) -> None:
    ab, bb = _bits(a), _bits(b)
    sb = _bits(shifted or 0)
    carry = ctl.bnegate                              # carry into bit 0
    slices = []
    for i in range(32):
        ai = ab[i] ^ ctl.ainvert
        bi = bb[i] ^ ctl.bnegate
        s = ai ^ bi ^ carry
        cout = (ai & bi) | (ai & carry) | (bi & carry)
        slices.append(Slice(i, ab[i], bb[i], ai, bi, ai & bi, ai | bi, ai ^ bi, carry, s,
                            cout, 0, sb[i], 0))
        carry = cout
    msb = slices[31]
    overflow = msb.cin ^ msb.cout
    set_ = (1 - msb.cout) if ctl.unsigned else msb.sum_ ^ overflow
    slices[0].less = set_
    result = 0
    for sl in slices:
        sl.result = (sl.and_, sl.or_, sl.sum_, sl.less, sl.xor_, sl.shift, 0, 0)[ctl.operation]
        result |= sl.result << sl.i
    out.slices, out.result = slices, result
    out.zero = int(result == 0)
    out.overflow, out.set_ = overflow, set_
    if ctl.operation == 2:
        traps = out.instruction.split()[0] in ("add", "addi", "sub")
        out.notes.append("Overflow = carry into bit 31 XOR carry out of bit 31 = "
                         f"{overflow}" + (" — this instruction traps on it" if traps else
                                          " (ignored: this instruction does not trap)"))
    if ctl.operation == 3 and ctl.unsigned:
        out.notes.append("Unsigned (sltu): Less = NOT the carry out of a - b (a borrow "
                         "means a < b unsigned) instead of the sign")


def _shift(out: AluLogic, ctl: AluControl, a: int, b: int) -> int:
    """The barrel shifter: 5 stages of 32 2:1 muxes, shifting by 16, 8, 4,
    2, 1 places when that bit of the amount is 1."""
    lines = ctl.lines
    if lines["Lui"]:
        value, amount, kind = b, 16, "left"
        out.notes.append("lui: the immediate shifted left 16 places")
    else:
        value, amount = a, b & 31
        kind = ("rotate right" if lines["ShRotate"] else
                "right, arithmetic (sign bits fill)" if lines["ShArith"] else
                "right, logical (zeros fill)" if lines["ShRight"] else "left")
    src = ("16 (Lui)" if lines["Lui"] else "rs (ShVar)" if lines["ShVar"]
           else "the sa field")
    out.notes.append(f"shift amount {amount} = {amount:05b} from {src}: {kind}")
    sign = value >> 31
    for k in (16, 8, 4, 2, 1):
        if amount & k:
            if kind == "left":
                value = (value << k) & M32
            elif kind.startswith("rotate"):
                value = ((value >> k) | (value << (32 - k))) & M32
            elif kind.startswith("right, arith"):
                value = (value >> k) | ((((1 << k) - 1) << (32 - k)) if sign else 0)
            else:
                value >>= k
            out.shifter.append((f"stage {k:2d}: shift {k}", value))
        else:
            out.shifter.append((f"stage {k:2d}: pass", value))
    return value


def binary(x: int) -> str:
    """32 bits, bit 31 first, in groups of 4."""
    s = format(x & M32, "032b")
    return " ".join(s[i:i + 4] for i in range(0, 32, 4))


def rows(logic: AluLogic) -> list[tuple[str, int, bool]]:
    """The 32-bit rows of the slices (label, value, is it the one the
    Operation mux passes on)."""
    if not logic.slices:
        return []
    sl = logic.slices
    v = lambda attr: sum(getattr(s, attr) << s.i for s in sl)  # noqa: E731
    op = logic.control.operation
    carries = sum(s.cout << s.i for s in sl)
    out = [("a", logic.a, False), ("b", logic.b, False),
           (f"a' (Ainvert {logic.control.ainvert})", v("a_in"), False),
           (f"b' (Bnegate {logic.control.bnegate})", v("b_in"), False),
           ("AND   a' & b'", v("and_"), op == 0),
           ("OR    a' | b'", v("or_"), op == 1),
           ("XOR   a' ^ b'", v("xor_"), op == 4),
           ("carry out of each bit", carries, False),
           ("SUM   a' + b' + carry in", v("sum_"), op == 2),
           ("Less  (bit 0 = Set)", v("less"), op == 3)]
    if op == 5:
        out.append(("shifter output", v("shift"), True))
    return out + [("result", logic.result or 0, False)]
