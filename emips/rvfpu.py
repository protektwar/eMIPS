"""RISC-V's F and D extensions (PLAN M25 phase 1d; the unprivileged manual
20250508 ch. 21 "F", ch. 22 "D").

The arithmetic is :mod:`emips.fpu`'s exact soft-float; what is RISC-V's
own (``m25-RISC-V.md`` decision 8):

- **NaNs:** every NaN result is the canonical NaN (0x7FC00000 /
  0x7FF8000000000000, 21.3: "no NaN payload propagation"); a signaling
  NaN operand signals NV.
- **NaN-boxing** (22.2): a single value in a 64-bit register has bits
  63:32 all ones; an operand that is not properly boxed reads as the
  canonical NaN; FLW / FMV.W.X / single results are boxed.  FSW and
  FMV.X.W take bits 31:0 as they are.
- **No traps:** the exception flags accrue in fflags (21.2).
- **Rounding** (21.2 Table 21.1): RNE / RTZ / RDN / RUP / RMM, 7 =
  dynamic (frm); a reserved value (5, 6, or a dynamic one with frm 5-7) is
  an illegal instruction — on every instruction with an rm field (Spike
  and QEMU check it on the exact conversions too).
- **FMIN / FMAX** (21.6): IEEE 754-2019 minimumNumber / maximumNumber
  (one NaN: the other operand; signaling: NV), -0 < +0.
- **FCVT to an integer** (21.7 Table 21.2): NaN and too-large values give
  the largest value, too-small the smallest; NV; NX when rounded.
- **FEQ** signals NV only for a signaling NaN, **FLT / FLE** for any NaN.
- **FCLASS** (Table 21.3): ten bits, -inf ... qNaN.
- **The fused multiply-adds** (21.6) signal NV for inf * 0 even with a
  quiet NaN addend.
- **mstatus.FS** (priv. 3.1.6.6): Off makes every F / D instruction (and
  fflags / frm / fcsr) illegal; an instruction that writes an f register
  or raises a flag sets it Dirty.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from . import fpu as F
from . import rvcsr as CSR

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
    from .simrv import ILLEGAL, RVTrap
    if m.csrs.status_field(CSR.FS_SHIFT, 3) == 0:
        raise RVTrap(ILLEGAL, i.raw)


def dirty(m: "RVMachine") -> None:
    m.csrs.mstatus |= 3 << CSR.FS_SHIFT


def accrue(m: "RVMachine", flags: set[str]) -> None:
    if flags:
        m.fcsr |= sum(_FLAGS[x] for x in flags)
        dirty(m)


def read_s(m: "RVMachine", n: int) -> int:
    """A single operand (22.2: not NaN-boxed = the canonical NaN)."""
    v = m.fpr[n]
    if m.flen == 64 and v & BOX != BOX:
        return canonical(F.S)
    return v & M32


def write_s(m: "RVMachine", n: int, v: int) -> None:
    m.fpr[n] = (v & M32) | (BOX if m.flen == 64 else 0)
    dirty(m)


def read_d(m: "RVMachine", n: int) -> int:
    return m.fpr[n] & M64


def write_d(m: "RVMachine", n: int, v: int) -> None:
    m.fpr[n] = v & M64
    dirty(m)


def _canon(bits: int, f: F.Fmt) -> int:
    return canonical(f) if F.is_nan(bits, f) else bits


# -------------------------------------------------------------- operations

def fma(a: int, b: int, c: int, f: F.Fmt, rm: int, *, neg_product: bool,
        neg_addend: bool) -> tuple[int, set[str]]:
    """(±a * b) ± c, rounded once (21.6)."""
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
    """FCVT.W / WU / L / LU (Table 21.2); the raw ``bits``-wide result."""
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
    """Table 21.3."""
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


# -------------------------------------------------------------- executors

def _executors() -> dict:
    from .simrv import _reg, _set, _s32, misaligned
    ex: dict = {}

    def fp(fn):
        def run(m, i, pc):
            require_fp(m, i)
            return fn(m, i, pc)
        return run

    for fmt, f, rd_, wr_ in (("s", F.S, read_s, write_s), ("d", F.D, read_d, write_d)):
        def arith(op, f=f, rd_=rd_, wr_=wr_):
            def run(m, i, pc):
                rm = rounding(m, i)
                a, b = rd_(m, i.f["rs1"]), rd_(m, i.f.get("rs2", 0))
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
                wr_(m, i.f["rd"], _canon(v, f))
                accrue(m, fl)
            return fp(run)
        for op in ("add", "sub", "mul", "div", "sqrt"):
            ex[f"f{op}.{fmt}"] = arith(op)

        def fused(neg_product, neg_addend, f=f, rd_=rd_, wr_=wr_):
            def run(m, i, pc):
                rm = rounding(m, i)
                v, fl = fma(rd_(m, i.f["rs1"]), rd_(m, i.f["rs2"]), rd_(m, i.f["rs3"]), f, rm,
                            neg_product=neg_product, neg_addend=neg_addend)
                wr_(m, i.f["rd"], _canon(v, f))
                accrue(m, fl)
            return fp(run)
        ex[f"fmadd.{fmt}"] = fused(False, False)
        ex[f"fmsub.{fmt}"] = fused(False, True)
        ex[f"fnmsub.{fmt}"] = fused(True, False)
        ex[f"fnmadd.{fmt}"] = fused(True, True)

        def sgnj(kind, f=f, rd_=rd_, wr_=wr_):
            def run(m, i, pc):
                a, b = rd_(m, i.f["rs1"]), rd_(m, i.f["rs2"])
                sb = f.sign_bit
                s = (b & sb) if kind == "j" else (~b & sb) if kind == "jn" else ((a ^ b) & sb)
                wr_(m, i.f["rd"], (a & ~sb) | s)
            return fp(run)
        for kind in ("j", "jn", "jx"):
            ex[f"fsgn{kind}.{fmt}"] = sgnj(kind)

        def mm(op, f=f, rd_=rd_, wr_=wr_):
            def run(m, i, pc):
                v, fl = minmax(rd_(m, i.f["rs1"]), rd_(m, i.f["rs2"]), f, op)
                wr_(m, i.f["rd"], v)
                accrue(m, fl)
            return fp(run)
        ex[f"fmin.{fmt}"] = mm("min")
        ex[f"fmax.{fmt}"] = mm("max")

        def cmp(op, f=f, rd_=rd_):
            def run(m, i, pc):
                v, fl = compare(rd_(m, i.f["rs1"]), rd_(m, i.f["rs2"]), f, op)
                _set(m, i.f["rd"], v)
                accrue(m, fl)
            return fp(run)
        for op in ("eq", "lt", "le"):
            ex[f"f{op}.{fmt}"] = cmp(op)

        def cls(f=f, rd_=rd_):
            def run(m, i, pc):
                _set(m, i.f["rd"], classify(rd_(m, i.f["rs1"]), f))
            return fp(run)
        ex[f"fclass.{fmt}"] = cls()

        for ity, bits, signed in (("w", 32, True), ("wu", 32, False), ("l", 64, True),
                                  ("lu", 64, False)):
            def f2i(bits=bits, signed=signed, f=f, rd_=rd_):
                def run(m, i, pc):
                    rm = rounding(m, i)
                    v, fl = to_int(rd_(m, i.f["rs1"]), f, bits, signed, rm)
                    _set(m, i.f["rd"], _s32(v) if bits == 32 else v)   # (W results sign-extend)
                    accrue(m, fl)
                return fp(run)

            def i2f(bits=bits, signed=signed, f=f, wr_=wr_):
                def run(m, i, pc):
                    rm = rounding(m, i)
                    v, fl = from_int(_reg(m, i.f["rs1"]), bits, signed, f, rm)
                    wr_(m, i.f["rd"], v)
                    accrue(m, fl)
                return fp(run)
            ex[f"fcvt.{ity}.{fmt}"] = f2i()
            ex[f"fcvt.{fmt}.{ity}"] = i2f()

    def s2d(m, i, pc):
        rounding(m, i)
        a = read_s(m, i.f["rs1"])
        v, fl = F.convert(a, F.S, F.D, F.RN, nan2008=True)
        write_d(m, i.f["rd"], _canon(v, F.D))
        accrue(m, fl)

    def d2s(m, i, pc):
        rm = rounding(m, i)
        v, fl = F.convert(read_d(m, i.f["rs1"]), F.D, F.S, rm, nan2008=True)
        write_s(m, i.f["rd"], _canon(v, F.S))
        accrue(m, fl)
    ex["fcvt.d.s"] = fp(s2d)
    ex["fcvt.s.d"] = fp(d2s)

    def _ea(m, i, size, store):
        addr = (_reg(m, i.f["rs1"]) + i.f["imm"]) & m.amask
        misaligned(m, addr, size, store)        # (the P8700's DMALN / DULS)
        return addr

    def flw(m, i, pc):
        write_s(m, i.f["rd"], m.load(_ea(m, i, 4, False), 4))

    def fsw(m, i, pc):
        m.store(_ea(m, i, 4, True), m.fpr[i.f["rs2"]] & M32, 4)

    def fld(m, i, pc):
        write_d(m, i.f["rd"], m.load(_ea(m, i, 8, False), 8))

    def fsd(m, i, pc):
        m.store(_ea(m, i, 8, True), m.fpr[i.f["rs2"]] & M64, 8)

    def fmv_x_w(m, i, pc):
        _set(m, i.f["rd"], _s32(m.fpr[i.f["rs1"]]))

    def fmv_w_x(m, i, pc):
        write_s(m, i.f["rd"], _reg(m, i.f["rs1"]))

    def fmv_x_d(m, i, pc):
        _set(m, i.f["rd"], m.fpr[i.f["rs1"]])

    def fmv_d_x(m, i, pc):
        write_d(m, i.f["rd"], _reg(m, i.f["rs1"]))
    ex.update({"flw": fp(flw), "fsw": fp(fsw), "fld": fp(fld), "fsd": fp(fsd),
               "fmv.x.w": fp(fmv_x_w), "fmv.w.x": fp(fmv_w_x),
               "fmv.x.d": fp(fmv_x_d), "fmv.d.x": fp(fmv_d_x)})
    return ex
