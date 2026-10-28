"""Exact IEEE 754 soft-float for the MIPS32 FPU (M1 slice 2).

Every operation is computed **exactly** (as a rational number) and then
rounded once to the destination format in the requested rounding mode, so
results and exception flags are those of an IEEE 754 binary32/binary64
unit.  MIPS-specific choices follow the *MIPS32 Architecture For
Programmers* Vol. I-A (FPU chapter) and Vol. II-A for a Release 2 FPU
with **legacy (pre-2008) NaN encoding** (``FIR.Has2008 = 0``,
``FCSR.NAN2008 = ABS2008 = 0``):

* quiet NaN = fraction MSB **0**, signaling NaN = fraction MSB **1**;
  default quiet NaN ``0x7FBFFFFF`` / ``0x7FF7FFFFFFFFFFFF``;
* an operation with a signaling-NaN operand signals Invalid and returns
  the default NaN; otherwise a NaN operand is propagated (first operand
  first);
* invalid float→integer conversions (NaN, ±∞, out of range) signal Invalid
  and return ``2^31 - 1`` / ``2^63 - 1``;
* ``ABS``/``NEG`` are arithmetic (Linux commit 232b6ec5df87, "ABS.fmt and
  NEG.fmt ... specified as arithmetic"): a quiet NaN is propagated
  unchanged, a signaling NaN signals Invalid;
* ``CVT.S.D``/``CVT.D.S`` propagate a quiet NaN's payload (shifted), and
  return the default NaN when narrowing would leave no NaN (Linux
  ``arch/mips/math-emu/sp_fdp.c``);
* tininess is detected **after rounding**; Underflow is flagged when the
  result is tiny **and** inexact (untrapped), or tiny (trapped);
* ``FCSR.FS = 1`` flushes tiny results to zero (Underflow + Inexact).

**Release 6 / IEEE 754-2008 mode** (``nan2008=True``, i.e. FCSR.NAN2008 =
ABS2008 = 1): quiet NaN = fraction MSB **1** (default ``0x7FC00000``), a
signaling NaN operand signals Invalid and is *quieted and propagated*
(signaling NaNs first, then first operand first), invalid conversions give
0 for NaN and saturate otherwise, ``ABS``/``NEG`` only change the sign
bit; plus the Release 6 operations :func:`cmp_r6`, :func:`fma`,
:func:`minmax`, :func:`rint` and :func:`classify` (checked against QEMU's
mips32r6-generic CPU).

Flags are a set of the letters ``V Z O U I`` (Invalid, divide-by-Zero,
Overflow, Underflow, Inexact), matching the FCSR Cause/Enable/Flag bit
order ``E V Z O U I`` (bits 17..12 / -..7 / 6..2).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

# rounding modes (FCSR.RM); RMM = round to nearest, ties away from zero
# (IEEE 754-2008 roundTiesToAway: RISC-V's rm 4, PLAN M25 — no MIPS FPU has it)
RN, RZ, RP, RM, RMM = 0, 1, 2, 3, 4
#: round to odd (truncate, then set the LSB when inexact; overflow gives the
#: largest finite value): RISC-V's vfncvt.rod.f.f.w (unpriv. 30.13.18),
#: softfloat's softfloat_round_odd
ROD = 5

#: FCSR Cause / Enable / Flag bit index of each flag (relative: I = 0).
FLAG_BITS = {"I": 0, "U": 1, "O": 2, "Z": 3, "V": 4}


@dataclass(frozen=True)
class Fmt:
    name: str
    ebits: int
    fbits: int

    @property
    def bits(self) -> int:
        return 1 + self.ebits + self.fbits

    @property
    def bias(self) -> int:
        return (1 << (self.ebits - 1)) - 1

    @property
    def emax(self) -> int:
        return self.bias

    @property
    def emin(self) -> int:
        return 1 - self.bias

    @property
    def p(self) -> int:                     # precision incl. hidden bit
        return self.fbits + 1

    @property
    def exp_mask(self) -> int:
        return (1 << self.ebits) - 1

    @property
    def sign_bit(self) -> int:
        return 1 << (self.bits - 1)

    @property
    def default_nan(self) -> int:
        """Legacy MIPS default quiet NaN (fraction MSB clear)."""
        return (self.exp_mask << self.fbits) | ((1 << (self.fbits - 1)) - 1)

    @property
    def max_finite(self) -> int:
        return ((self.exp_mask - 1) << self.fbits) | ((1 << self.fbits) - 1)

    @property
    def inf(self) -> int:
        return self.exp_mask << self.fbits


S = Fmt("S", 8, 23)
D = Fmt("D", 11, 52)
#: binary16 and binary128 (MSA's halves; RISC-V's Zfh and Q)
H = Fmt("H", 5, 10)
Q = Fmt("Q", 15, 112)


# ------------------------------------------------------------ classification

def sign(bits: int, f: Fmt) -> int:
    return (bits >> (f.bits - 1)) & 1


def _exp(bits: int, f: Fmt) -> int:
    return (bits >> f.fbits) & f.exp_mask


def _frac(bits: int, f: Fmt) -> int:
    return bits & ((1 << f.fbits) - 1)


def is_nan(bits: int, f: Fmt) -> bool:
    return _exp(bits, f) == f.exp_mask and _frac(bits, f) != 0


def is_snan(bits: int, f: Fmt, nan2008: bool = False) -> bool:
    """Signaling NaN: fraction MSB **set** (legacy MIPS) / **clear** (2008)."""
    return is_nan(bits, f) and bool(_frac(bits, f) >> (f.fbits - 1)) != nan2008


def default_nan(f: Fmt, nan2008: bool = False) -> int:
    if nan2008:
        return (f.exp_mask << f.fbits) | (1 << (f.fbits - 1))
    return f.default_nan


def quiet(bits: int, f: Fmt) -> int:
    """2008 encoding: a signaling NaN made quiet (payload kept)."""
    return bits | (1 << (f.fbits - 1))


def is_inf(bits: int, f: Fmt) -> bool:
    return _exp(bits, f) == f.exp_mask and _frac(bits, f) == 0


def value(bits: int, f: Fmt) -> Fraction:
    """Exact value of a finite operand (sign included)."""
    e, m = _exp(bits, f), _frac(bits, f)
    if e == 0:
        v = Fraction(m, 1 << (f.fbits - f.emin))
    else:
        v = Fraction((1 << f.fbits) | m) * Fraction(2) ** (e - f.bias - f.fbits)
    return -v if sign(bits, f) else v


# ----------------------------------------------------------------- rounding

def _floor_log2(v: Fraction) -> int:
    """floor(log2(v)) for v > 0, exactly."""
    n, d = v.numerator, v.denominator
    e = n.bit_length() - d.bit_length()
    if (n << max(0, -e)) < (d << max(0, e)):
        e -= 1
    return e


def _round_int(m: Fraction, neg: bool, rm: int) -> int:
    """Round a non-negative magnitude to an integer per ``rm``."""
    q = m.numerator // m.denominator
    rem = m - q
    if rem == 0:
        return q
    if rm == RN:
        half = Fraction(1, 2)
        return q + 1 if rem > half or (rem == half and q & 1) else q
    if rm == RMM:
        return q + 1 if rem >= Fraction(1, 2) else q
    if rm == RZ:
        return q
    if rm == ROD:
        return q | 1
    if rm == RP:
        return q if neg else q + 1
    return q + 1 if neg else q              # RM


def round_pack(v: Fraction, f: Fmt, rm: int, *, neg_zero: bool = False,
               flush: bool = False) -> tuple[int, set[str]]:
    """Round the exact value ``v`` to format ``f``.  ``neg_zero`` gives the
    sign of an exact zero.  ``flush``: True flushes results tiny after
    rounding to zero (FCSR.FS); ``"before"`` those tiny *before* rounding
    (MSACSR.FS, MD00866 3.5.4) — both with Underflow and Inexact.  Returns
    ``(bits, flags)``."""
    flags: set[str] = set()
    if v == 0:
        return (f.sign_bit if neg_zero else 0), flags
    neg = v < 0
    mag = -v if neg else v
    sgn = f.sign_bit if neg else 0
    if flush == "before" and mag < Fraction(2) ** f.emin:
        return sgn, {"U", "I"}
    e = max(_floor_log2(mag), f.emin)
    scaled = mag / Fraction(2) ** (e - f.fbits)          # [1, 2^p) or less
    q = _round_int(scaled, neg, rm)
    if q != scaled:
        flags.add("I")
    if q >> f.p:                                         # carried to 2^p
        q >>= 1
        e += 1
    if e > f.emax:                                       # overflow
        flags |= {"O", "I"}
        to_inf = rm in (RN, RMM) or (rm == RP and not neg) or (rm == RM and neg)
        return sgn | (f.inf if to_inf else f.max_finite), flags
    if q >> f.fbits:                                     # normal
        return sgn | ((e + f.bias) << f.fbits) | (q & ((1 << f.fbits) - 1)), flags
    # tiny after rounding (subnormal or zero)
    if flush:
        return sgn, flags | {"U", "I"}
    if "I" in flags:
        flags.add("U")
    return sgn | q, flags


# ------------------------------------------------------------- operations

def _nan_result(f: Fmt, *ops: int, nan2008: bool = False) -> tuple[int, set[str]] | None:
    """NaN handling of an arithmetic operation, or ``None`` when no operand
    is a NaN."""
    for x in ops:
        if is_snan(x, f, nan2008):
            return (quiet(x, f) if nan2008 else f.default_nan), {"V"}
    for x in ops:
        if is_nan(x, f):
            return x, set()
    return None


def add(a: int, b: int, f: Fmt, rm: int, flush: bool = False,
        subtract: bool = False, nan2008: bool = False) -> tuple[int, set[str]]:
    if subtract:
        b ^= f.sign_bit if not is_nan(b, f) else 0
    nan = _nan_result(f, a, b, nan2008=nan2008)
    if nan:
        return nan
    if is_inf(a, f) or is_inf(b, f):
        if is_inf(a, f) and is_inf(b, f) and sign(a, f) != sign(b, f):
            return default_nan(f, nan2008), {"V"}
        return (a if is_inf(a, f) else b), set()
    v = value(a, f) + value(b, f)
    if v == 0:                       # exact zero: IEEE 754 sign rules
        if value(a, f) == 0 and value(b, f) == 0 and sign(a, f) == sign(b, f):
            neg_zero = bool(sign(a, f))
        else:
            neg_zero = rm == RM
        return round_pack(v, f, rm, neg_zero=neg_zero)
    return round_pack(v, f, rm, flush=flush)


def mul(a: int, b: int, f: Fmt, rm: int, flush: bool = False,
        nan2008: bool = False) -> tuple[int, set[str]]:
    nan = _nan_result(f, a, b, nan2008=nan2008)
    if nan:
        return nan
    neg = bool(sign(a, f) ^ sign(b, f))
    if is_inf(a, f) or is_inf(b, f):
        other = b if is_inf(a, f) else a
        if not is_inf(other, f) and value(other, f) == 0:
            return default_nan(f, nan2008), {"V"}
        return (f.sign_bit if neg else 0) | f.inf, set()
    return round_pack(value(a, f) * value(b, f), f, rm, neg_zero=neg, flush=flush)


def div(a: int, b: int, f: Fmt, rm: int, flush: bool = False,
        nan2008: bool = False) -> tuple[int, set[str]]:
    nan = _nan_result(f, a, b, nan2008=nan2008)
    if nan:
        return nan
    neg = bool(sign(a, f) ^ sign(b, f))
    sg = f.sign_bit if neg else 0
    if is_inf(a, f):
        return (default_nan(f, nan2008), {"V"}) if is_inf(b, f) else (sg | f.inf, set())
    if is_inf(b, f):
        return sg, set()
    va, vb = value(a, f), value(b, f)
    if vb == 0:
        return (default_nan(f, nan2008), {"V"}) if va == 0 else (sg | f.inf, {"Z"})
    return round_pack(va / vb, f, rm, neg_zero=neg, flush=flush)


def sqrt(a: int, f: Fmt, rm: int, flush: bool = False,
         nan2008: bool = False) -> tuple[int, set[str]]:
    nan = _nan_result(f, a, nan2008=nan2008)
    if nan:
        return nan
    if is_inf(a, f):
        return (default_nan(f, nan2008), {"V"}) if sign(a, f) else (a, set())
    v = value(a, f)
    if v == 0:
        return a, set()                                  # sqrt(±0) = ±0
    if v < 0:
        return default_nan(f, nan2008), {"V"}
    # exact enough: sqrt to 2p+2 bits plus a sticky bit decides rounding
    scale = 2 * (f.p + 4) - _floor_log2(v)
    scale += scale & 1
    n = v * (Fraction(2) ** scale)
    r = isqrt(n.numerator // n.denominator)
    exact = r * r == n
    approx = Fraction(r, 1) if exact else Fraction(2 * r + 1, 2)  # sticky
    return round_pack(approx / Fraction(2) ** (scale // 2), f, rm, flush=flush)


def neg_abs(a: int, f: Fmt, *, negate: bool, nan2008: bool = False) -> tuple[int, set[str]]:
    """NEG / ABS (legacy: arithmetic — a signaling NaN signals Invalid;
    2008 / ABS2008: only the sign bit changes, never an exception)."""
    if nan2008:
        return ((a ^ f.sign_bit) if negate else (a & ~f.sign_bit)), set()
    if is_snan(a, f):
        return f.default_nan, {"V"}
    if is_nan(a, f):
        return a, set()
    return (a ^ f.sign_bit) if negate else (a & ~f.sign_bit), set()


def convert(a: int, src: Fmt, dst: Fmt, rm: int, flush: bool = False,
            nan2008: bool = False) -> tuple[int, set[str]]:
    """CVT.S.D / CVT.D.S."""
    if is_nan(a, src):
        flags = set()
        if is_snan(a, src, nan2008):
            if not nan2008:
                return dst.default_nan, {"V"}
            a, flags = quiet(a, src), {"V"}
        shift = dst.fbits - src.fbits
        frac = _frac(a, src) << shift if shift >= 0 else _frac(a, src) >> -shift
        if frac == 0 or (nan2008 and not frac >> (dst.fbits - 1)):
            return default_nan(dst, nan2008), flags
        return ((dst.sign_bit if sign(a, src) else 0) | dst.inf | frac), flags
    if is_inf(a, src):
        return (dst.sign_bit if sign(a, src) else 0) | dst.inf, set()
    return round_pack(value(a, src), dst, rm, neg_zero=bool(sign(a, src)),
                      flush=flush)


def from_int(n: int, bits: int, dst: Fmt, rm: int) -> tuple[int, set[str]]:
    """CVT.S/D.W/L: ``n`` is the raw ``bits``-wide two's complement."""
    n &= (1 << bits) - 1
    if n >> (bits - 1):
        n -= 1 << bits
    return round_pack(Fraction(n), dst, rm)


def to_int(a: int, src: Fmt, bits: int, rm: int,
           nan2008: bool = False) -> tuple[int, set[str]]:
    """CVT/ROUND/TRUNC/CEIL/FLOOR to W (32) or L (64).  Invalid results:
    legacy 2^(n-1)-1; 2008: 0 for NaN, the nearest bound otherwise."""
    hi, lo = (1 << (bits - 1)) - 1, 1 << (bits - 1)          # lo = min as raw
    if is_nan(a, src):
        return (0 if nan2008 else hi), {"V"}
    if is_inf(a, src):
        return (lo if nan2008 and sign(a, src) else hi), {"V"}
    v = value(a, src)
    neg = v < 0
    q = _round_int(-v if neg else v, neg, rm)
    r = -q if neg else q
    if not -(1 << (bits - 1)) <= r < (1 << (bits - 1)):
        return (lo if nan2008 and r < 0 else hi), {"V"}
    return r & ((1 << bits) - 1), ({"I"} if r != v else set())


def compare(a: int, b: int, f: Fmt, cond: int, nan2008: bool = False) -> tuple[bool, set[str]]:
    """C.cond.fmt: ``cond`` bit 0 = unordered, 1 = equal, 2 = less,
    3 = signal Invalid on (quiet) unordered."""
    if is_nan(a, f) or is_nan(b, f):
        signal = is_snan(a, f, nan2008) or is_snan(b, f, nan2008) or bool(cond & 8)
        return bool(cond & 1), ({"V"} if signal else set())
    va, vb = _ordered(a, f), _ordered(b, f)
    return bool((cond & 2 and va == vb) or (cond & 4 and va < vb)), set()


def _ordered(x: int, f: Fmt) -> Fraction | float:
    if is_inf(x, f):
        return float("-inf") if sign(x, f) else float("inf")
    return value(x, f)


# ------------------------------------------------------------ Release 6 ops

def cmp_r6(a: int, b: int, f: Fmt, cond: int, nan2008: bool = True) -> tuple[bool, set[str]]:
    """CMP.cond.fmt (``cond`` = the function field: 0-15 like C.cond, 17 OR,
    18 UNE, 19 NE; +8 / 25-27 = signaling)."""
    signaling = bool(cond & 8)
    unord = is_nan(a, f) or is_nan(b, f)
    flags: set[str] = set()
    if unord and (signaling or is_snan(a, f, nan2008) or is_snan(b, f, nan2008)):
        flags.add("V")
    eq = lt = False
    if not unord:
        va, vb = _ordered(a, f), _ordered(b, f)
        eq, lt = va == vb, va < vb
    base = cond & ~8
    if base < 8:
        res = bool(base & 1 and unord or base & 2 and eq or base & 4 and lt)
    else:
        res = {17: not unord, 18: unord or not eq, 19: not unord and not eq}[base]
    return res, flags


def fma(a: int, b: int, c: int, f: Fmt, rm: int, flush: bool = False, *,
        subtract: bool = False, nan2008: bool = True) -> tuple[int, set[str]]:
    """MADDF / MSUBF: ``c ± a*b`` rounded once (fused)."""
    nan = _nan_result(f, a, b, c, nan2008=nan2008)
    if nan:
        return nan
    psign = bool(sign(a, f) ^ sign(b, f)) ^ subtract
    if is_inf(a, f) or is_inf(b, f):
        other = b if is_inf(a, f) else a
        if not is_inf(other, f) and value(other, f) == 0:
            return default_nan(f, nan2008), {"V"}
        if is_inf(c, f) and bool(sign(c, f)) != psign:
            return default_nan(f, nan2008), {"V"}
        return (f.sign_bit if psign else 0) | f.inf, set()
    if is_inf(c, f):
        return c, set()
    p = value(a, f) * value(b, f)
    p = -p if subtract else p
    v = value(c, f) + p
    if v == 0:
        cz = value(c, f) == 0
        if p == 0 and cz and bool(sign(c, f)) == psign:
            neg_zero = psign
        else:
            neg_zero = rm == RM
        return round_pack(v, f, rm, neg_zero=neg_zero)
    return round_pack(v, f, rm, flush=flush)


def minmax(a: int, b: int, f: Fmt, op: str, nan2008: bool = True) -> tuple[int, set[str]]:
    """MIN / MAX / MINA / MAXA.fmt: a quiet NaN loses to a number; a
    signaling NaN signals Invalid (quieted NaN result); ``MINA/MAXA``
    compare magnitudes, ties resolved like ``MIN/MAX``."""
    if is_snan(a, f, nan2008) or is_snan(b, f, nan2008):
        return _nan_result(f, a, b, nan2008=nan2008)
    if is_nan(a, f) and is_nan(b, f):
        return a, set()
    if is_nan(a, f):
        return b, set()
    if is_nan(b, f):
        return a, set()
    va, vb = _ordered(a, f), _ordered(b, f)

    def pick_min(x, y, vx, vy):
        if vx != vy:
            return x if vx < vy else y
        return x if sign(x, f) else y            # -0 < +0
    def pick_max(x, y, vx, vy):
        if vx != vy:
            return x if vx > vy else y
        return y if sign(x, f) else x
    if op in ("MINA", "MAXA"):
        ma, mb = abs(va), abs(vb)
        if ma != mb:
            return ((a if ma < mb else b) if op == "MINA" else (a if ma > mb else b)), set()
        op = op[:3]
    return (pick_min(a, b, va, vb) if op == "MIN" else pick_max(a, b, va, vb)), set()


def rint(a: int, f: Fmt, rm: int, nan2008: bool = True) -> tuple[int, set[str]]:
    """RINT.fmt: round to an integral value in the same format."""
    nan = _nan_result(f, a, nan2008=nan2008)
    if nan:
        return nan
    if is_inf(a, f):
        return a, set()
    v = value(a, f)
    neg = sign(a, f)
    q = _round_int(-v if v < 0 else v, v < 0, rm)
    r = Fraction(-q if v < 0 else q)
    bits, _ = round_pack(r, f, rm, neg_zero=bool(neg))
    return bits, ({"I"} if r != v else set())


#: CLASS.fmt result bits
CLASS_BITS = {"snan": 0, "qnan": 1, "-inf": 2, "-normal": 3, "-subnormal": 4,
              "-zero": 5, "+inf": 6, "+normal": 7, "+subnormal": 8, "+zero": 9}


def classify(a: int, f: Fmt, nan2008: bool = True) -> int:
    if is_nan(a, f):
        return 1 << CLASS_BITS["snan" if is_snan(a, f, nan2008) else "qnan"]
    sg = "-" if sign(a, f) else "+"
    if is_inf(a, f):
        kind = "inf"
    elif _exp(a, f) == 0:
        kind = "zero" if _frac(a, f) == 0 else "subnormal"
    else:
        kind = "normal"
    return 1 << CLASS_BITS[sg + kind]
