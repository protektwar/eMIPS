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

A nanoMIPS core (``arch="nanomips"``, PLAN M23) has the same 32-bit ALU;
its main control names the ALU function itself (:mod:`emips.nanocontrol`).

A RISC-V hart (``arch="rv32"`` / ``"rv64"``, PLAN M25 phase 4c) has the
same ALU, 32 or 64 bits wide; its controls come from :mod:`emips.rvcontrol`
(Word: RV64's *W operations; andn / orn / xnor are and / or / xor with
Bnegate); RISC-V's add and sub never trap.

A MIPS64 core (``arch="mips64"``, PLAN M22) has a 64-bit ALU: 64 slices
(Overflow and Set at bit 63) and a 6-stage barrel shifter (32, 16, 8,
4, 2, 1; Sh32 adds 32 to the amount: DSLL32 ...).  Its **Word** line
makes a 32-bit operation of MIPS64 Vol. II: the slices' result bit 31 is
sign-extended into bits 63..32 and an overflow is bit 31's (ADDU, ADD,
SUBU ...), a word shift shifts the low word in the 5-stage part and
sign-extends it (SLL ... ROTRV), LUI's result is a word.  An address in
the 32-bit compatibility space (Vol. III 4.10 / 4.11) is then
sign-extended from bit 31 by the address path, after the adder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1

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
#: ... on a MIPS64 core (64-bit adds; ADDI / ADDIU add a word)
ALUOP_WHY64 = {**ALUOP_WHY, 0b0000: "add (64 bits): a load, a store, daddi / daddiu",
               0b1000: "lui: shift left 16, a word (bit 31 sign-extended)",
               0b1001: "add word: addi / addiu (the 32-bit sum, sign-extended)"}


def aluop_why(code: int, wide: bool = False) -> str:
    """What the main control's ALUOp ``code`` asks for (``wide``: MIPS64)."""
    return (ALUOP_WHY64 if wide else ALUOP_WHY).get(code, "")


@dataclass
class AluControl:
    aluop: int                   # the main control's 4-bit ALUOp
    funct: str | None            # R-type: bits 5..0
    lines: dict                  # the ALU control's outputs (controllogic.ALU_OUTPUTS)
    #: what ALUOp asks for, when not MIPS32's / MIPS64's code (nanoMIPS, RISC-V)
    why: str | None = None
    #: ALUOp's width (RISC-V's: 3 bits)
    aluop_bits: int = 4

    @property
    def aluop_why(self) -> str:
        if self.why is not None:
            return self.why
        return aluop_why(self.aluop, "Word" in self.lines)

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
    def word(self) -> int:
        """MIPS64: a 32-bit operation (its result bit 31 sign-extended)."""
        return int(self.lines.get("Word", False))

    @property
    def sh32(self) -> int:
        """MIPS64: the shift amount + 32 (DSLL32 ...)."""
        return int(self.lines.get("Sh32", False))

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
    #: the ALU's width (MIPS64: 64)
    width: int = 32
    #: the bit Overflow and Set come from (31; MIPS64: 63, a Word operation 31)
    top: int = 31
    #: MIPS64: what the address path makes of the sum (a compatibility
    #: address: bit 31 sign-extended), when it differs
    address: int | None = None

    @property
    def matches(self) -> bool:
        return self.result is not None and self.traced in (self.result, self.address)


def _bits(x: int, n: int = 32) -> list[int]:
    return [(x >> i) & 1 for i in range(n)]


def _sx32(x: int) -> int:
    """Bit 31 sign-extended into bits 63..32."""
    x &= M32
    return x | (M64 ^ M32) if x & 0x80000000 else x


def control_for(word: int, release: str = "R5", arch: str = "mips32",
                msa: bool = False) -> AluControl:
    """The ALU control of an instruction word: the main control's ALUOp,
    then the ALU control PLA's lines (:mod:`emips.controllogic`; ``msa``:
    a core with MSA — LSA / DLSA are its rows before Release 6)."""
    from . import controllogic as cl
    ev = cl.evaluate_main(cl.main_control(release, arch, msa), word)
    _, lines = cl.alu_control(release, arch, msa).evaluate(cl.alu_input(ev.aluop, word))
    funct = format(word & 0x3F, "06b") if ev.aluop == 0b0010 else None
    return AluControl(ev.aluop, funct, lines)


def alu_logic(view, release: str = "R5", arch: str = "mips32",
              msa: bool = False, extensions: str = "") -> AluLogic | None:
    """The gates of the instruction ``view`` (an :class:`emips.trace.InstrView`)
    executed, or None when it used no ALU operation (``arch="mips64"``: the
    64-bit ALU)."""
    if view is None or view.alu is None:
        return None
    alu = view.alu
    rv = arch in ("rv32", "rv64")
    width = 64 if arch in ("mips64", "rv64") else 32
    mask = (1 << width) - 1
    a = alu.a.value & mask if alu.a is not None else 0
    b = alu.b.value & mask if alu.b is not None else 0
    if rv:                                       # (emips.rvcontrol: the expanded word)
        from . import rvcontrol as rc
        ctl = rc.control_for(rc.view_word(view, width, extensions), width, extensions)
    elif arch == "nanomips":                     # (emips.nanocontrol; a 32-bit ALU)
        from .nanocontrol import control_for as nano_control
        ctl = nano_control(view.word, view.size)
    else:
        ctl = control_for(view.word, release, arch, msa)
    top = 31 if width == 32 or ctl.word else 63
    out = AluLogic(view.text, ctl, a, b, None, alu.result, width=width, top=top)
    if ctl.other:
        out.notes.append(f"{view.text}: {alu.op} — done by another unit (the multiply / "
                         "divide unit, the bit-field or count logic ...): the ALU control "
                         "says OtherUnit")
        return out
    shifted = _shift(out, ctl, a, b) if ctl.operation == 5 else None
    _slices(out, ctl, a, b, shifted, traps=not rv)
    if arch == "mips64" and view.address is not None and out.result is not None \
            and alu.result == _sx32(out.result) != out.result:
        out.address = _sx32(out.result)                 # (Vol. III 4.10 / 4.11)
        out.notes.append("the address path then sign-extends bit 31: a 32-bit "
                         "compatibility address (MIPS64 Vol. III 4.10 / 4.11)")
    return out


def _slices(out: AluLogic, ctl: AluControl, a: int, b: int, shifted: int | None,
            traps: bool = True) -> None:
    n = out.width
    ab, bb = _bits(a, n), _bits(b, n)
    sb = _bits(shifted or 0, n)
    carry = ctl.bnegate                              # carry into bit 0
    slices = []
    for i in range(n):
        ai = ab[i] ^ ctl.ainvert
        bi = bb[i] ^ ctl.bnegate
        s = ai ^ bi ^ carry
        cout = (ai & bi) | (ai & carry) | (bi & carry)
        slices.append(Slice(i, ab[i], bb[i], ai, bi, ai & bi, ai | bi, ai ^ bi, carry, s,
                            cout, 0, sb[i], 0))
        carry = cout
    msb = slices[out.top]                            # (bit 31; MIPS64 bit 63, a Word 31)
    overflow = msb.cin ^ msb.cout
    set_ = (1 - msb.cout) if ctl.unsigned else msb.sum_ ^ overflow
    slices[0].less = set_
    result = 0
    for sl in slices:
        sl.result = (sl.and_, sl.or_, sl.sum_, sl.less, sl.xor_, sl.shift, 0, 0)[ctl.operation]
        result |= sl.result << sl.i
    if ctl.word and ctl.operation != 5:              # MIPS64 Word: bit 31 sign-extended
        out.notes.append("Word: a 32-bit operation — result bit 31 "
                         f"({(result >> 31) & 1}) is copied into bits 63..32")
        result = _sx32(result)
    out.slices, out.result = slices, result
    out.zero = int(result == 0)
    out.overflow, out.set_ = overflow, set_
    if ctl.operation == 2:
        t = out.top
        traps = traps and out.instruction.split()[0] in ("add", "addi", "sub", "dadd",
                                                           "daddi", "dsub")
        out.notes.append(f"Overflow = carry into bit {t} XOR carry out of bit {t} = "
                         f"{overflow}" + (" — this instruction traps on it" if traps else
                                          " (ignored: this instruction does not trap)"))
    if ctl.operation == 3 and ctl.unsigned:
        out.notes.append("Unsigned (sltu): Less = NOT the carry out of a - b (a borrow "
                         "means a < b unsigned) instead of the sign")


def _shift(out: AluLogic, ctl: AluControl, a: int, b: int) -> int:
    """The barrel shifter: 5 stages of 32 2:1 muxes, shifting by 16, 8, 4,
    2, 1 places when that bit of the amount is 1 (MIPS64: 6 stages of 64,
    from 32; a Word shift uses the low word's 5, then bit 31 is
    sign-extended)."""
    lines = ctl.lines
    n = 32 if out.width == 32 or ctl.word else 64      # the shifter's width
    mask = (1 << n) - 1
    if lines["Lui"]:
        value, amount, kind = b & mask, 16, "left"
        out.notes.append("lui: the immediate shifted left 16 places")
    else:
        value = a & mask
        amount = (b & (n - 1)) + (32 if ctl.sh32 else 0)
        kind = ("rotate right" if lines["ShRotate"] else
                "right, arithmetic (sign bits fill)" if lines["ShArith"] else
                "right, logical (zeros fill)" if lines["ShRight"] else "left")
    src = ("16 (Lui)" if lines["Lui"] else "rs (ShVar)" if lines["ShVar"]
           else "the sa field + 32 (Sh32)" if ctl.sh32 else "the sa field")
    bits = 6 if n == 64 else 5
    out.notes.append(f"shift amount {amount} = {amount:0{bits}b} from {src}: {kind}"
                     + (" — on the low word (Word)" if ctl.word and out.width == 64
                        and not lines["Lui"] else ""))
    sign = value >> (n - 1)
    for k in ((32,) if n == 64 else ()) + (16, 8, 4, 2, 1):
        if amount & k:
            if kind == "left":
                value = (value << k) & mask
            elif kind.startswith("rotate"):
                value = ((value >> k) | (value << (n - k))) & mask
            elif kind.startswith("right, arith"):
                value = (value >> k) | ((((1 << k) - 1) << (n - k)) if sign else 0)
            else:
                value >>= k
            out.shifter.append((f"stage {k:2d}: shift {k}", value))
        else:
            out.shifter.append((f"stage {k:2d}: pass", value))
    if out.width == 64 and ctl.word:                    # MIPS64 Word: sign-extend
        value = _sx32(value)
        out.shifter.append(("Word: bit 31 into 63..32", value))
    return value


def binary(x: int, width: int = 32) -> str:
    """``width`` bits (32; MIPS64: 64), the top one first, in groups of 4."""
    s = format(x & ((1 << width) - 1), f"0{width}b")
    return " ".join(s[i:i + 4] for i in range(0, width, 4))


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
