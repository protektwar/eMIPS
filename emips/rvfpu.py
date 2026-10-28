"""RISC-V's floating point (PLAN M25 phase 1d; the unprivileged manual
20250508 ch. 20 "F", ch. 21 "D"; LF2: ch. 22 "Q", ch. 23 "Zfh" / "Zfhmin",
ch. 25 "Zfa", ch. 26 "Zfinx" / "Zdinx" / "Zhinx" / "Zhinxmin").

The arithmetic is :mod:`emips.fpu`'s exact soft-float; what is RISC-V's
own (``m25-RISC-V.md`` decision 8):

- **NaNs:** every NaN result is the canonical NaN (0x7FC00000 /
  0x7FF8000000000000, 20.3: "no NaN payload propagation"); a signaling
  NaN operand signals NV.
- **NaN-boxing** (21.2; 22 / 23 recursively): a value narrower than FLEN
  (16 / 32 / 64 / 128) has every bit above it set; an operand that is not
  properly boxed reads as the canonical NaN; loads, FMV.*.X and results
  are boxed.  FSH / FSW / FSD and FMV.X.* take the low bits as they are.
- **Zfinx & co.** (ch. 26): the operands are x registers (``i.inx``):
  w-bit operands use bits w-1:0, w-bit results are sign-extended to XLEN
  (26.1); RV32 Zdinx's doubles are even / odd register pairs (an odd
  number is reserved: an illegal instruction; x0 reads 0 and is not
  written, 26.3); mstatus.FS is 0 and gates nothing (26.6).
- **No traps:** the exception flags accrue in fflags (20.2).
- **Rounding** (20.2 Table 25): RNE / RTZ / RDN / RUP / RMM, 7 =
  dynamic (frm); a reserved value (5, 6, or a dynamic one with frm 5-7) is
  an illegal instruction — on every instruction with an rm field (Spike
  and QEMU check it on the exact conversions too).
- **FMIN / FMAX** (20.6): IEEE 754-2019 minimumNumber / maximumNumber
  (one NaN: the other operand; signaling: NV), -0 < +0.
- **FCVT to an integer** (20.7 Table 28): NaN and too-large values give
  the largest value, too-small the smallest; NV; NX when rounded.
- **FEQ** signals NV only for a signaling NaN, **FLT / FLE** for any NaN.
- **FCLASS** (Table 29): ten bits, -inf ... qNaN.
- **The fused multiply-adds** (20.6) signal NV for inf * 0 even with a
  quiet NaN addend.
- **mstatus.FS** (priv. 3.1.6.6): Off makes every F / D instruction (and
  fflags / frm / fcsr) illegal; an instruction that writes an f register
  or raises a flag sets it Dirty.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from . import fpu as F

if TYPE_CHECKING:
    from .riscv import RV
    from .simrv import RVMachine

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
BOX = 0xFFFFFFFF00000000

#: RISC-V rm -> emips.fpu's rounding modes (RDN = toward -inf = fpu.RM)
_RM = {0: F.RN, 1: F.RZ, 2: F.RM, 3: F.RP, 4: F.RMM}
_FLAGS = {"I": 1, "U": 2, "O": 4, "Z": 8, "V": 16}


def canonical(f: F.Fmt) -> int:
    return F.default_nan(f, True)


def rounding(m: "RVMachine", i: "RV") -> int:
    """The instruction's rounding mode (an illegal instruction for a
    reserved one)."""
    from .simrv import ILLEGAL, RVTrap
    rm = i.f["rm"]
    if rm == 7:
        rm = m.fcsr >> 5 & 7
    if rm not in _RM:
        raise RVTrap(ILLEGAL, i.raw)
    return _RM[rm]


def require_fp(m: "RVMachine", i: "RV") -> None:
    """mstatus.FS = Off: an illegal instruction (not with Zfinx: 26.6);
    with Zfinx and Smstateen, the FCSR bit of the stateen CSRs (priv. 4.2:
    every FP instruction, as though it accessed fcsr — Spike's require_fp)."""
    from .simrv import ILLEGAL, RVTrap
    if not i.inx:
        if not m.csrs.fs_on():                  # (with V = 1 vsstatus.FS too: 21.2.11)
            raise RVTrap(ILLEGAL, i.raw)
    elif m.csrs.stateen:
        from .rvcsr import SE_FCSR
        m.csrs.stateen_check(SE_FCSR, i.raw, user=True)


def dirty(m: "RVMachine") -> None:
    if m.flen:                                  # (Zfinx: FS is hardwired to 0)
        m.csrs.fs_dirty()


def accrue(m: "RVMachine", flags: set[str]) -> None:
    if flags:
        m.fcsr |= sum(_FLAGS[x] for x in flags)
        dirty(m)


#: the formats by their letter (Table 30 / 31: fmt 00 S, 01 D, 10 H, 11 Q)
FMTS = {"h": F.H, "s": F.S, "d": F.D, "q": F.Q}


def box(v: int, width: int, flen: int) -> int:
    """``v`` (``width`` bits) NaN-boxed in an ``flen``-bit register."""
    return (v & ((1 << width) - 1)) | (((1 << flen) - 1) ^ ((1 << width) - 1))


def unbox(v: int, f: F.Fmt, flen: int) -> int:
    """An f register as an ``f`` operand: the canonical NaN unless boxed."""
    w = f.bits
    if w < flen and v >> w != (1 << (flen - w)) - 1:
        return canonical(f)
    return v & ((1 << w) - 1)


def read_s(m: "RVMachine", n: int) -> int:
    """A single operand (21.2: not NaN-boxed = the canonical NaN)."""
    return unbox(m.fpr[n], F.S, m.flen)


def write_s(m: "RVMachine", n: int, v: int) -> None:
    m.fpr[n] = box(v, 32, m.flen)
    dirty(m)


def read_d(m: "RVMachine", n: int) -> int:
    return unbox(m.fpr[n], F.D, m.flen)


def write_d(m: "RVMachine", n: int, v: int) -> None:
    m.fpr[n] = box(v, 64, m.flen)
    dirty(m)


def number(bits: int, f: F.Fmt) -> float:
    """A value of format ``f`` as a Python float (a quad rounded to the
    nearest double; out of its range: infinity)."""
    if F.is_nan(bits, f):
        return float("nan")
    if F.is_inf(bits, f):
        return float("-inf") if F.sign(bits, f) else float("inf")
    v = F.value(bits, f)
    try:
        d = float(v)
    except OverflowError:
        d = float("inf") if v > 0 else float("-inf")
    return -0.0 if v == 0 and F.sign(bits, f) else d


def _text(bits: int, f: F.Fmt) -> str:
    special = F.is_nan(bits, f) or F.is_inf(bits, f)
    if f is not F.Q or special or F.value(bits, f) == 0:
        return f"{number(bits, f):.{ {16: 5, 32: 7, 64: 15}.get(f.bits, 17)}g}"
    exact = F.value(bits, f)                    # (a quad: 36 significant digits)
    from decimal import Context, Decimal
    c = Context(prec=36)
    return format(c.divide(Decimal(exact.numerator), Decimal(exact.denominator)), "g")


def show(raw: int, flen: int) -> str:
    """An f register for the views: the narrowest format it is NaN-boxed
    in (21.2), and its value."""
    for name, f in (("half", F.H), ("single", F.S), ("double", F.D), ("quad", F.Q)):
        if f.bits > flen:
            break
        if f.bits == flen or raw >> f.bits == (1 << (flen - f.bits)) - 1:
            text = _text(raw & ((1 << f.bits) - 1), f)
            return text if f.bits == flen else f"{text} ({name}, NaN-boxed)"
    return "?"


def operand_formats(name: str) -> dict[str, str]:
    """The floating-point operands of a row (its 32-bit name) and their
    formats (``h`` / ``s`` / ``d`` / ``q``); the other operands are
    integers (x registers) or immediates (``fli``'s rs1)."""
    if not name.startswith("f") or name in ("fence", "fence.i", "fence.tso"):
        return {}
    parts = name.split(".")
    op, last = parts[0], parts[-1]
    if op in ("flh", "flw", "fld", "flq"):
        return {"rd": "hsdq"["hwdq".index(op[2])]}
    if op in ("fsh", "fsw", "fsd", "fsq"):
        return {"rs2": "hsdq"["hwdq".index(op[2])]}
    if op == "fmv":                             # fmv.x.h / fmv.h.x ... ("w": a single)
        fmt = {"w": "s"}.get(parts[1] if parts[2] == "x" else parts[2])
        fmt = fmt or (parts[1] if parts[2] == "x" else parts[2])
        return {"rd": fmt} if parts[2] == "x" else {"rs1": fmt}
    if op == "fmvh":                            # fmvh.x.d / fmvh.x.q
        return {"rs1": last}
    if op == "fmvp":                            # fmvp.d.x / fmvp.q.x
        return {"rd": parts[1]}
    if op == "fli":
        return {"rd": last}
    if op in ("fcvt", "fcvtmod"):
        out = {}
        if parts[1] in FMTS:
            out["rd"] = parts[1]
        if parts[2] in FMTS:
            out["rs1"] = parts[2]
        return out
    if op in ("feq", "flt", "fle", "fleq", "fltq"):
        return {"rs1": last, "rs2": last}
    if op == "fclass":
        return {"rs1": last}
    if op in ("fsqrt", "fround", "froundnx"):
        return {"rd": last, "rs1": last}
    return {"rd": last, "rs1": last, "rs2": last, "rs3": last}


def _illegal(i: "RV"):
    from .simrv import ILLEGAL, RVTrap
    return RVTrap(ILLEGAL, i.raw)


def get(m: "RVMachine", i: "RV", field: str, f: F.Fmt) -> int:
    """Operand ``field`` of ``i`` in format ``f``: an f register (boxing
    checked) or, with Zfinx & co., an x register (pair) — ch. 26."""
    n = i.f[field]
    if not i.inx:
        return unbox(m.fpr[n], f, m.flen)
    if f.bits > i.inx:                          # RV32 Zdinx: an even / odd pair (26.3)
        if n & 1:
            raise _illegal(i)
        return 0 if n == 0 else (m.regs[n] & M32) | (m.regs[n + 1] & M32) << 32
    return m.regs[n] & ((1 << f.bits) - 1)       # (26.1: bits XLEN-1:w ignored)


def put(m: "RVMachine", i: "RV", field: str, v: int, f: F.Fmt) -> None:
    """Write result ``v`` (format ``f``) to operand ``field``: NaN-boxed in
    an f register, or sign-extended in an x register (pair)."""
    from .simrv import _set
    n = i.f[field]
    if not i.inx:
        m.fpr[n] = box(v, f.bits, m.flen)
        dirty(m)
        return
    if f.bits > i.inx:
        if n & 1:
            raise _illegal(i)
        if n:                                   # (x0: the whole write has no effect)
            _set(m, n, v & M32)
            _set(m, n + 1, (v >> 32) & M32)
        return
    w = f.bits
    v &= (1 << w) - 1
    _set(m, n, v - (1 << w) if v >> (w - 1) else v)


def _canon(bits: int, f: F.Fmt) -> int:
    return canonical(f) if F.is_nan(bits, f) else bits


# -------------------------------------------------------------- operations

def fma(a: int, b: int, c: int, f: F.Fmt, rm: int, *, neg_product: bool,
        neg_addend: bool) -> tuple[int, set[str]]:
    """(±a * b) ± c, rounded once (20.6)."""
    flags: set[str] = set()
    inf0 = (F.is_inf(a, f) and not F.is_nan(b, f) and not F.is_inf(b, f) and F.value(b, f) == 0) \
        or (F.is_inf(b, f) and not F.is_nan(a, f) and not F.is_inf(a, f) and F.value(a, f) == 0)
    if inf0:
        return canonical(f), {"V"}
    if any(F.is_snan(x, f, True) for x in (a, b, c)):
        return canonical(f), {"V"}
    if any(F.is_nan(x, f) for x in (a, b, c)):
        return canonical(f), flags
    psign = bool(F.sign(a, f) ^ F.sign(b, f)) ^ neg_product
    csign = bool(F.sign(c, f)) ^ neg_addend
    if F.is_inf(a, f) or F.is_inf(b, f):
        if F.is_inf(c, f) and csign != psign:
            return canonical(f), {"V"}
        return (f.sign_bit if psign else 0) | f.inf, flags
    if F.is_inf(c, f):
        return (f.sign_bit if csign else 0) | f.inf, flags
    p = F.value(a, f) * F.value(b, f)
    p = -p if neg_product else p
    cv = F.value(c, f)
    cv = -cv if neg_addend else cv
    v = p + cv
    if v == 0:
        if p == 0 and cv == 0 and psign == csign:
            neg_zero = psign
        else:
            neg_zero = rm == F.RM
        return F.round_pack(v, f, rm, neg_zero=neg_zero)
    return F.round_pack(v, f, rm)


def minmax(a: int, b: int, f: F.Fmt, op: str) -> tuple[int, set[str]]:
    flags = {"V"} if F.is_snan(a, f, True) or F.is_snan(b, f, True) else set()
    na, nb = F.is_nan(a, f), F.is_nan(b, f)
    if na and nb:
        return canonical(f), flags
    if na:
        return b, flags
    if nb:
        return a, flags
    va, vb = F._ordered(a, f), F._ordered(b, f)
    if va == vb:                                 # -0 < +0
        neg_a = bool(F.sign(a, f))
        pick_a = neg_a if op == "min" else not neg_a
        return (a if pick_a else b), flags
    return (a if (va < vb) == (op == "min") else b), flags


def to_int(a: int, f: F.Fmt, bits: int, signed: bool, rm: int) -> tuple[int, set[str]]:
    """FCVT.W / WU / L / LU (Table 28); the raw ``bits``-wide result."""
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if F.is_nan(a, f):
        return hi & ((1 << bits) - 1), {"V"}
    if F.is_inf(a, f):
        return (lo if F.sign(a, f) else hi) & ((1 << bits) - 1), {"V"}
    v = F.value(a, f)
    neg = v < 0
    q = F._round_int(-v if neg else v, neg, rm)
    r = -q if neg else q
    if r < lo:
        return lo & ((1 << bits) - 1), {"V"}
    if r > hi:
        return hi & ((1 << bits) - 1), {"V"}
    return r & ((1 << bits) - 1), ({"I"} if r != v else set())


def from_int(n: int, bits: int, signed: bool, f: F.Fmt, rm: int) -> tuple[int, set[str]]:
    n &= (1 << bits) - 1
    if signed and n >> (bits - 1):
        n -= 1 << bits
    return F.round_pack(Fraction(n), f, rm)


def compare(a: int, b: int, f: F.Fmt, op: str) -> tuple[int, set[str]]:
    if F.is_nan(a, f) or F.is_nan(b, f):
        sig = op != "eq" or F.is_snan(a, f, True) or F.is_snan(b, f, True)
        return 0, ({"V"} if sig else set())
    va, vb = F._ordered(a, f), F._ordered(b, f)
    return int(va == vb if op == "eq" else va < vb if op == "lt" else va <= vb), set()


def classify(a: int, f: F.Fmt) -> int:
    """Table 29."""
    if F.is_nan(a, f):
        return 1 << (8 if F.is_snan(a, f, True) else 9)
    neg = F.sign(a, f)
    if F.is_inf(a, f):
        return 1 << (0 if neg else 7)
    if F._exp(a, f) == 0:
        if F._frac(a, f) == 0:
            return 1 << (3 if neg else 4)
        return 1 << (2 if neg else 5)
    return 1 << (1 if neg else 6)


# -------------------------------------------------------------- Zfa (ch. 25)

#: FLI's constants (Table 36): the values, "min" (the format's smallest
#: normal), inf and the canonical NaN; 2^16 is infinity in half precision
FLI = (-1, "min", Fraction(1, 1 << 16), Fraction(1, 1 << 15), Fraction(1, 256),
       Fraction(1, 128), Fraction(1, 16), Fraction(1, 8), Fraction(1, 4), Fraction(5, 16),
       Fraction(3, 8), Fraction(7, 16), Fraction(1, 2), Fraction(5, 8), Fraction(3, 4),
       Fraction(7, 8), 1, Fraction(5, 4), Fraction(3, 2), Fraction(7, 4), 2, Fraction(5, 2),
       3, 4, 8, 16, 128, 256, 1 << 15, 1 << 16, "inf", "nan")


def fli(k: int, f: F.Fmt) -> int:
    v = FLI[k]
    if v == "min":
        return 1 << f.fbits
    if v == "inf":
        return f.inf
    if v == "nan":
        return canonical(f)
    bits, _ = F.round_pack(Fraction(v), f, F.RN)
    return bits                                 # (exact, or 2^16 -> inf in half: no flags)


def minmaxm(a: int, b: int, f: F.Fmt, op: str) -> tuple[int, set[str]]:
    """FMINM / FMAXM (25.2): FMIN / FMAX but a NaN operand gives the
    canonical NaN."""
    if F.is_nan(a, f) or F.is_nan(b, f):
        sig = F.is_snan(a, f, True) or F.is_snan(b, f, True)
        return canonical(f), ({"V"} if sig else set())
    return minmax(a, b, f, op)


def fround(a: int, f: F.Fmt, rm: int, exact: bool) -> tuple[int, set[str]]:
    """FROUND / FROUNDNX (25.3): to an integral value in rm; zero and
    infinity unchanged; a signaling NaN signals NV; FROUNDNX also NX."""
    if F.is_nan(a, f):
        return canonical(f), ({"V"} if F.is_snan(a, f, True) else set())
    bits, flags = F.rint(a, f, rm, nan2008=True)
    return bits, (flags if exact else set())


def fcvtmod(a: int) -> tuple[int, set[str]]:
    """FCVTMOD.W.D (25.4): truncated, modulo 2^32 (sign-extended later);
    NaN / infinity -> 0; the flags of FCVT.W.D with the input."""
    _, flags = to_int(a, F.D, 32, True, F.RZ)
    if F.is_nan(a, F.D) or F.is_inf(a, F.D):
        return 0, flags
    v = F.value(a, F.D)
    t = int(v)                                  # (toward zero)
    return t & M32, flags


# -------------------------------------------------------------- executors

def _executors() -> dict:
    from .simrv import _reg, _set, _s32, misaligned
    ex: dict = {}

    def fp(fn):
        def run(m, i, pc):
            require_fp(m, i)
            return fn(m, i, pc)
        return run

    for fmt, f in FMTS.items():
        def arith(op, f=f):
            def run(m, i, pc):
                rm = rounding(m, i)
                a = get(m, i, "rs1", f)
                b = get(m, i, "rs2", f) if op != "sqrt" else 0
                if op == "add":
                    v, fl = F.add(a, b, f, rm, nan2008=True)
                elif op == "sub":
                    v, fl = F.add(a, b, f, rm, subtract=True, nan2008=True)
                elif op == "mul":
                    v, fl = F.mul(a, b, f, rm, nan2008=True)
                elif op == "div":
                    v, fl = F.div(a, b, f, rm, nan2008=True)
                else:
                    v, fl = F.sqrt(a, f, rm, nan2008=True)
                put(m, i, "rd", _canon(v, f), f)
                accrue(m, fl)
            return fp(run)
        for op in ("add", "sub", "mul", "div", "sqrt"):
            ex[f"f{op}.{fmt}"] = arith(op)

        def fused(neg_product, neg_addend, f=f):
            def run(m, i, pc):
                rm = rounding(m, i)
                v, fl = fma(get(m, i, "rs1", f), get(m, i, "rs2", f), get(m, i, "rs3", f), f,
                            rm, neg_product=neg_product, neg_addend=neg_addend)
                put(m, i, "rd", _canon(v, f), f)
                accrue(m, fl)
            return fp(run)
        ex[f"fmadd.{fmt}"] = fused(False, False)
        ex[f"fmsub.{fmt}"] = fused(False, True)
        ex[f"fnmsub.{fmt}"] = fused(True, False)
        ex[f"fnmadd.{fmt}"] = fused(True, True)

        def sgnj(kind, f=f):
            def run(m, i, pc):
                a, b = get(m, i, "rs1", f), get(m, i, "rs2", f)
                sb = f.sign_bit
                s = (b & sb) if kind == "j" else (~b & sb) if kind == "jn" else ((a ^ b) & sb)
                put(m, i, "rd", (a & ~sb) | s, f)
            return fp(run)
        for kind in ("j", "jn", "jx"):
            ex[f"fsgn{kind}.{fmt}"] = sgnj(kind)

        def mm(op, nan_any, f=f):
            def run(m, i, pc):
                a, b = get(m, i, "rs1", f), get(m, i, "rs2", f)
                v, fl = minmaxm(a, b, f, op) if nan_any else minmax(a, b, f, op)
                put(m, i, "rd", v, f)
                accrue(m, fl)
            return fp(run)
        ex[f"fmin.{fmt}"] = mm("min", False)
        ex[f"fmax.{fmt}"] = mm("max", False)
        ex[f"fminm.{fmt}"] = mm("min", True)
        ex[f"fmaxm.{fmt}"] = mm("max", True)

        def cmp(op, quiet, f=f):
            def run(m, i, pc):
                a, b = get(m, i, "rs1", f), get(m, i, "rs2", f)
                v, fl = compare(a, b, f, op)
                if quiet and not (F.is_snan(a, f, True) or F.is_snan(b, f, True)):
                    fl = set()                  # (FLEQ / FLTQ: quiet NaNs signal nothing)
                _set(m, i.f["rd"], v)
                accrue(m, fl)
            return fp(run)
        for op in ("eq", "lt", "le"):
            ex[f"f{op}.{fmt}"] = cmp(op, False)
        ex[f"fleq.{fmt}"] = cmp("le", True)
        ex[f"fltq.{fmt}"] = cmp("lt", True)

        def cls(f=f):
            def run(m, i, pc):
                _set(m, i.f["rd"], classify(get(m, i, "rs1", f), f))
            return fp(run)
        ex[f"fclass.{fmt}"] = cls()

        def rnd(exact, f=f):
            def run(m, i, pc):
                rm = rounding(m, i)
                v, fl = fround(get(m, i, "rs1", f), f, rm, exact)
                put(m, i, "rd", v, f)
                accrue(m, fl)
            return fp(run)
        ex[f"fround.{fmt}"] = rnd(False)
        ex[f"froundnx.{fmt}"] = rnd(True)

        def load_imm(f=f):
            def run(m, i, pc):
                put(m, i, "rd", fli(i.f["rs1"], f), f)
            return fp(run)
        ex[f"fli.{fmt}"] = load_imm()

        for ity, bits, signed in (("w", 32, True), ("wu", 32, False), ("l", 64, True),
                                  ("lu", 64, False)):
            def f2i(bits=bits, signed=signed, f=f):
                def run(m, i, pc):
                    rm = rounding(m, i)
                    v, fl = to_int(get(m, i, "rs1", f), f, bits, signed, rm)
                    _set(m, i.f["rd"], _s32(v) if bits == 32 else v)   # (W results sign-extend)
                    accrue(m, fl)
                return fp(run)

            def i2f(bits=bits, signed=signed, f=f):
                def run(m, i, pc):
                    rm = rounding(m, i)
                    v, fl = from_int(_reg(m, i.f["rs1"]), bits, signed, f, rm)
                    put(m, i, "rd", v, f)
                    accrue(m, fl)
                return fp(run)
            ex[f"fcvt.{ity}.{fmt}"] = f2i()
            ex[f"fcvt.{fmt}.{ity}"] = i2f()

        for src, g in FMTS.items():             # format to format (exact when widening)
            if src == fmt:
                continue

            def f2f(f=f, g=g):
                def run(m, i, pc):
                    rm = rounding(m, i)
                    v, fl = F.convert(get(m, i, "rs1", g), g, f, rm, nan2008=True)
                    put(m, i, "rd", _canon(v, f), f)
                    accrue(m, fl)
                return fp(run)
            ex[f"fcvt.{fmt}.{src}"] = f2f()

    def fcvtmod_w_d(m, i, pc):                  # (25.4: rm = RTZ is fixed in the encoding)
        v, fl = fcvtmod(get(m, i, "rs1", F.D))
        _set(m, i.f["rd"], _s32(v))
        accrue(m, fl)
    ex["fcvtmod.w.d"] = fp(fcvtmod_w_d)

    def _ea(m, i, size, store):
        addr = (_reg(m, i.f["rs1"]) + i.f["imm"]) & m.amask
        misaligned(m, addr, size, store)        # (the P8700's DMALN / DULS)
        return addr

    for fmt, size in (("h", 2), ("w", 4), ("d", 8)):
        f = FMTS["s" if fmt == "w" else fmt]

        def load(size=size, f=f):
            def run(m, i, pc):
                put(m, i, "rd", m.load(_ea(m, i, size, False), size), f)
            return fp(run)

        def store(size=size):
            def run(m, i, pc):
                m.store(_ea(m, i, size, True), m.fpr[i.f["rs2"]] & ((1 << 8 * size) - 1), size)
            return fp(run)
        ex[f"fl{fmt}"] = load()
        ex[f"fs{fmt}"] = store()

    def flq(m, i, pc):                          # (22.1: two doublewords, not atomic)
        addr = _ea(m, i, 16, False)
        lo = m.load(addr, 8)
        hi = m.load((addr + 8) & m.amask, 8)
        put(m, i, "rd", lo | hi << 64, F.Q)

    def fsq(m, i, pc):
        addr = _ea(m, i, 16, True)
        hi_addr = (addr + 8) & m.amask
        m._access(hi_addr, 8, "store")          # (both halves allowed before either is written)
        v = m.fpr[i.f["rs2"]]
        m.store(addr, v & M64, 8)
        m.store(hi_addr, (v >> 64) & M64, 8)
    ex["flq"] = fp(flq)
    ex["fsq"] = fp(fsq)

    def fmv_x(width):
        def run(m, i, pc):
            v = m.fpr[i.f["rs1"]] & ((1 << width) - 1)
            _set(m, i.f["rd"], v - (1 << width) if v >> (width - 1) else v)
        return fp(run)

    def fmv_f(f):
        def run(m, i, pc):
            put(m, i, "rd", _reg(m, i.f["rs1"]), f)
        return fp(run)
    ex.update({"fmv.x.h": fmv_x(16), "fmv.x.w": fmv_x(32), "fmv.x.d": fmv_x(64),
               "fmv.h.x": fmv_f(F.H), "fmv.w.x": fmv_f(F.S), "fmv.d.x": fmv_f(F.D)})

    def fmvh(width):                            # 25.5: RV32 D / RV64 Q, the upper half
        def run(m, i, pc):
            v = (m.fpr[i.f["rs1"]] >> width) & ((1 << width) - 1)
            _set(m, i.f["rd"], v)
        return fp(run)

    def fmvp(f, width):
        def run(m, i, pc):
            v = (_reg(m, i.f["rs1"]) & ((1 << width) - 1)) | \
                (_reg(m, i.f["rs2"]) & ((1 << width) - 1)) << width
            put(m, i, "rd", v, f)
        return fp(run)
    ex.update({"fmvh.x.d": fmvh(32), "fmvh.x.q": fmvh(64),
               "fmvp.d.x": fmvp(F.D, 32), "fmvp.q.x": fmvp(F.Q, 64)})
    return ex
