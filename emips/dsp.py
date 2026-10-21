"""The MIPS DSP Module (PLAN M18): its executors and state.

Source: *MIPS Architecture for Programmers Volume IV-e: The MIPS DSP
Module for the MIPS32 Architecture* (MD00374 rev. 3.01): chapter 3 (the
state, Config3.DSPP / DSP2P, Status.MX, the DSP State Disabled exception,
Tables 3.4-3.8), chapter 6 (every instruction's operation).  Revision 1
(the 24KE), Revision 2 (the 34K / 74K / 1004K / interAptiv / microAptiv)
and, on Release 6, Revision 3 (BPOSGE32C; the accumulators belong to the
DSP Module alone).

State on the :class:`emips.sim.Machine`: ``dspctl`` (DSPControl,
:data:`DSPCONTROL_MASK`: pos 5..0, scount 12..7, c 13, EFI 14, ouflag
23..16, ccond 27..24), ``dsp_hi`` / ``dsp_lo`` (ac1-ac3; index 0 unused —
ac0 is the base ``hi`` / ``lo``).  The executors here are reached through
``emips.sim._dispatch``, which raises DSP State Disabled (ExcCode 26)
while Status.MX = 0.

Element order: the manual names elements by position in the register
(left / right halfword, bytes D C B A from bit 31 down), never by memory
order, so nothing here depends on the byte order (3.9).

UNPREDICTABLE results take QEMU's choice (``sim.DEVIATIONS["DSP
Module"]``): ccond bits a .PH compare does not write stay, EXTP* that
fail write 0, INSV out of range leaves rt, MTHLIP with pos > 32 keeps
pos, BALIGN bp 0 / 2 leaves rt, the GPR-targeting multiplies leave ac0.
"""

from __future__ import annotations

from typing import Callable

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1

#: the DSPControl bits that exist on MIPS32 (Figure 3.5; 31..28, 15, 6 read 0)
DSPCONTROL_MASK = 0x0FFF7FBF
POS, SCOUNT_SHIFT, C_BIT, EFI_BIT = 0x3F, 7, 13, 14
OUFLAG_SHIFT, CCOND_SHIFT = 16, 24
#: RDDSP / WRDSP mask bit -> the DSPControl field it selects
MASK_FIELDS = (0x3F, 0x3F << 7, 1 << 13, 0xFF << 16, 0xFF << 24, 1 << 14)

Executor = Callable[["object", int], "int | None"]


# --------------------------------------------------------------- helpers

def _rs(w: int) -> int:
    return (w >> 21) & 31


def _rt(w: int) -> int:
    return (w >> 16) & 31


def _rd(w: int) -> int:
    return (w >> 11) & 31


def _ac(w: int) -> int:
    return (w >> 11) & 3


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def acc(m, n: int) -> int:
    """Accumulator ``n`` as an unsigned 64-bit value (ac0 = HI / LO)."""
    if n == 0:
        return ((m.hi & M32) << 32) | (m.lo & M32)
    return ((m.dsp_hi[n] & M32) << 32) | (m.dsp_lo[n] & M32)


def set_acc(m, n: int, v: int) -> None:
    """Write accumulator ``n`` from a 64-bit value: HI = bits 63..32, LO =
    bits 31..0 (MIPS64, MD00375: each sign-extended)."""
    v &= M64
    hi, lo = m._sxw(v >> 32), m._sxw(v)
    if n == 0:
        m.hi, m.lo = hi, lo
    else:
        m.dsp_hi[n], m.dsp_lo[n] = hi, lo


def _half(m, n: int, high: bool) -> int:
    """HI[ac] / LO[ac] as the register holds it (MIPS64: all 64 bits)."""
    if n == 0:
        return m.hi if high else m.lo
    return (m.dsp_hi if high else m.dsp_lo)[n]


def _set_half(m, n: int, high: bool, v: int) -> None:
    v &= m.amask
    if n == 0:
        if high:
            m.hi = v
        else:
            m.lo = v
    else:
        (m.dsp_hi if high else m.dsp_lo)[n] = v


def _flag(m, bit: int) -> None:
    """Set a sticky DSPControl bit (an ouflag bit 16..23)."""
    m.dspctl |= 1 << bit


def _halves(v: int, signed: bool = True) -> tuple[int, int]:
    """(left, right) halfwords: bits 31..16, 15..0."""
    hi, lo = (v >> 16) & 0xFFFF, v & 0xFFFF
    return (_sx(hi, 16), _sx(lo, 16)) if signed else (hi, lo)


def _bytes(v: int, signed: bool = False) -> tuple[int, int, int, int]:
    """(D, C, B, A): bits 31..24, 23..16, 15..8, 7..0."""
    out = tuple((v >> s) & 0xFF for s in (24, 16, 8, 0))
    return tuple(_sx(b, 8) for b in out) if signed else out  # type: ignore[return-value]


def _ph(b: int, a: int) -> int:
    return ((b & 0xFFFF) << 16) | (a & 0xFFFF)


def _qb(d: int, c: int, b: int, a: int) -> int:
    return ((d & 0xFF) << 24) | ((c & 0xFF) << 16) | ((b & 0xFF) << 8) | (a & 0xFF)


def _r(m, w: int, which: Callable[[int], int]) -> int:
    """GPR[x]31..0: what every DSP operation reads (MIPS64, MD00375: the
    low word; a NotWordValue operand is UNPREDICTABLE)."""
    return m.read_reg(which(w)) & M32


def _sat(m, v: int, lo: int, hi: int, bit: int) -> int:
    """Clamp to [lo, hi], setting ouflag ``bit`` when it clamps."""
    if v > hi:
        _flag(m, bit)
        return hi
    if v < lo:
        _flag(m, bit)
        return lo
    return v


def _wrap(m, v: int, lo: int, hi: int, bit: int) -> int:
    """Modular result; ouflag ``bit`` when it did not fit."""
    if not lo <= v <= hi:
        _flag(m, bit)
    return v


# ------------------------------------------------- arithmetic (Table 3.5 bit 20)

def _pairwise_ph(op: Callable[[object, int, int], int], signed: bool = True) -> Executor:
    def ex(m, w: int) -> None:
        sb, sa = _halves(_r(m, w, _rs), signed)
        tb, ta = _halves(_r(m, w, _rt), signed)
        m.write_reg(_rd(w), _ph(op(m, sb, tb), op(m, sa, ta)))
    return ex


def _pairwise_qb(op: Callable[[object, int, int], int], signed: bool = False) -> Executor:
    def ex(m, w: int) -> None:
        s, t = _bytes(_r(m, w, _rs), signed), _bytes(_r(m, w, _rt), signed)
        m.write_reg(_rd(w), _qb(*(op(m, a, b) for a, b in zip(s, t))))
    return ex


def _word(op: Callable[[object, int, int], int]) -> Executor:
    def ex(m, w: int) -> None:
        m.write_reg(_rd(w), op(m, _sx(_r(m, w, _rs), 32), _sx(_r(m, w, _rt), 32)))
    return ex


def _abs(bits: int) -> Callable[[object, int], int]:
    lo = -(1 << (bits - 1))

    def f(m, a: int) -> int:
        if a == lo:                         # satAbs: the one value without a positive
            _flag(m, 20)
            return -lo - 1
        return -a if a < 0 else a
    return f


def _e_absq_s_ph(m, w: int) -> None:
    b, a = _halves(_r(m, w, _rt))
    f = _abs(16)
    m.write_reg(_rd(w), _ph(f(m, b), f(m, a)))


def _e_absq_s_qb(m, w: int) -> None:
    f = _abs(8)
    m.write_reg(_rd(w), _qb(*(f(m, x) for x in _bytes(_r(m, w, _rt), True))))


def _e_absq_s_w(m, w: int) -> None:
    m.write_reg(_rd(w), _abs(32)(m, _sx(_r(m, w, _rt), 32)))


def _add(bits: int, sat: bool, sub: bool = False, signed: bool = True):
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)

    def f(m, a: int, b: int) -> int:
        v = a - b if sub else a + b
        return _sat(m, v, lo, hi, 20) if sat else _wrap(m, v, lo, hi, 20)
    return f


def _halve(sub: bool, rnd: bool) -> Callable[[object, int, int], int]:
    """(a ± b [+ 1]) >> 1 without overflow (ADDQH / SUBUH ...)."""
    def f(m, a: int, b: int) -> int:
        return ((a - b if sub else a + b) + (1 if rnd else 0)) >> 1
    return f


def _e_addsc(m, w: int) -> None:
    v = _r(m, w, _rs) + _r(m, w, _rt)
    m.dspctl = (m.dspctl & ~(1 << C_BIT)) | ((v >> 32) << C_BIT)
    m.write_reg(_rd(w), v)


def _e_addwc(m, w: int) -> None:
    v = _sx(_r(m, w, _rs), 32) + _sx(_r(m, w, _rt), 32) + ((m.dspctl >> C_BIT) & 1)
    m.write_reg(_rd(w), _wrap(m, v, -(1 << 31), (1 << 31) - 1, 20))


def _e_modsub(m, w: int) -> None:
    rs, rt = _r(m, w, _rs), _r(m, w, _rt)
    m.write_reg(_rd(w), (rt >> 8) & 0xFFFF if rs == 0 else rs - (rt & 0xFF))


def _e_raddu_w_qb(m, w: int) -> None:
    m.write_reg(_rd(w), sum(_bytes(_r(m, w, _rs))))


# ------------------------------------------------------------- compare / pick

_CMP = {"EQ": lambda a, b: a == b, "LT": lambda a, b: a < b, "LE": lambda a, b: a <= b}


def _ccond(m, bits: int, n: int) -> None:
    """Write the low ``n`` ccond bits (the others stay: QEMU's choice for
    the UNPREDICTABLE ones of a .PH compare)."""
    mask = ((1 << n) - 1) << CCOND_SHIFT
    m.dspctl = (m.dspctl & ~mask) | (bits << CCOND_SHIFT)


def _cc_qb(m, w: int, cond: str) -> int:
    s, t = _bytes(_r(m, w, _rs)), _bytes(_r(m, w, _rt))
    return sum(int(_CMP[cond](a, b)) << (3 - i) for i, (a, b) in enumerate(zip(s, t)))


def _cmpu(cond: str) -> Executor:
    def ex(m, w: int) -> None:
        _ccond(m, _cc_qb(m, w, cond), 4)
    return ex


def _cmpgu(cond: str, dsp: bool) -> Executor:
    def ex(m, w: int) -> None:
        cc = _cc_qb(m, w, cond)
        if dsp:                              # CMPGDU: DSPControl too
            _ccond(m, cc, 4)
        m.write_reg(_rd(w), cc)
    return ex


def _cmp_ph(cond: str) -> Executor:
    def ex(m, w: int) -> None:
        (sb, sa), (tb, ta) = _halves(_r(m, w, _rs)), _halves(_r(m, w, _rt))
        _ccond(m, (int(_CMP[cond](sb, tb)) << 1) | int(_CMP[cond](sa, ta)), 2)
    return ex


def _e_pick_qb(m, w: int) -> None:
    s, t = _bytes(_r(m, w, _rs)), _bytes(_r(m, w, _rt))
    cc = m.dspctl >> CCOND_SHIFT
    m.write_reg(_rd(w), _qb(*(s[i] if cc >> (3 - i) & 1 else t[i] for i in range(4))))


def _e_pick_ph(m, w: int) -> None:
    s, t = _halves(_r(m, w, _rs), False), _halves(_r(m, w, _rt), False)
    cc = m.dspctl >> CCOND_SHIFT
    m.write_reg(_rd(w), _ph(s[0] if cc & 2 else t[0], s[1] if cc & 1 else t[1]))


def _e_packrl_ph(m, w: int) -> None:
    m.write_reg(_rd(w), _ph(_r(m, w, _rs), _r(m, w, _rt) >> 16))


# ------------------------------------------------------------------ shifts

def _sa(w: int, bits: int) -> int:
    return (w >> 21) & ((1 << bits) - 1)


def _shl(bits: int, sat: bool, signed: bool = True) -> Callable[[object, int, int], int]:
    """shift8Left (unsigned bytes) / shift16Left / sat16ShiftLeft /
    sat32ShiftLeft: the bits shifted out must be all copies of the sign
    (all zero for bytes), else ouflag 22 (and saturate)."""
    mask = (1 << bits) - 1

    def f(m, a: int, s: int) -> int:
        if s == 0:
            return a & mask
        if signed:
            discard = _sx(a, bits) >> (bits - 1 - s)          # the s bits out + the sign
            bad = discard not in (0, -1)
        else:
            bad = (a & mask) >> (bits - s) != 0
        if bad:
            _flag(m, 22)
            if sat:
                return (1 << (bits - 1)) - 1 if _sx(a, bits) >= 0 else 1 << (bits - 1)
        return (a << s) & mask
    return f


def _shra(bits: int, rnd: bool) -> Callable[[object, int, int], int]:
    def f(m, a: int, s: int) -> int:
        a = _sx(a, bits)
        if rnd and s:
            return ((a >> (s - 1)) + 1) >> 1
        return a >> s
    return f


def _shrl(bits: int) -> Callable[[object, int, int], int]:
    def f(m, a: int, s: int) -> int:
        return (a & ((1 << bits) - 1)) >> s
    return f


def _shift_qb(op, imm: bool) -> Executor:
    def ex(m, w: int) -> None:
        s = _sa(w, 3) if imm else _r(m, w, _rs) & 7
        m.write_reg(_rd(w), _qb(*(op(m, x, s) for x in _bytes(_r(m, w, _rt)))))
    return ex


def _shift_ph(op, imm: bool) -> Executor:
    def ex(m, w: int) -> None:
        s = _sa(w, 4) if imm else _r(m, w, _rs) & 15
        b, a = _halves(_r(m, w, _rt), False)
        m.write_reg(_rd(w), _ph(op(m, b, s), op(m, a, s)))
    return ex


def _shift_w(op, imm: bool) -> Executor:
    def ex(m, w: int) -> None:
        s = _sa(w, 5) if imm else _r(m, w, _rs) & 31
        m.write_reg(_rd(w), op(m, _r(m, w, _rt), s))
    return ex


# ------------------------------------------------------------- multiplies

def _mul_q15(m, a: int, b: int, bit: int) -> int:
    """multiplyQ15Q15: (a * b) << 1, -1 * -1 = 0x7FFFFFFF with ouflag ``bit``."""
    if a == b == -0x8000:
        _flag(m, bit)
        return 0x7FFFFFFF
    return (a * b) << 1


def _e_muleq_s_w_ph(left: bool) -> Executor:
    def ex(m, w: int) -> None:
        s, t = _halves(_r(m, w, _rs)), _halves(_r(m, w, _rt))
        i = 0 if left else 1
        m.write_reg(_rd(w), _mul_q15(m, s[i], t[i], 21))
    return ex


def _e_muleu_s_ph_qb(left: bool) -> Executor:
    def ex(m, w: int) -> None:
        q = _bytes(_r(m, w, _rs))
        b8, a8 = (q[0], q[1]) if left else (q[2], q[3])
        tb, ta = _halves(_r(m, w, _rt), False)
        m.write_reg(_rd(w), _ph(_sat(m, b8 * tb, 0, 0xFFFF, 21),
                                _sat(m, a8 * ta, 0, 0xFFFF, 21)))
    return ex


def _q15_round(m, a: int, b: int) -> int:
    """rndQ15MultiplyQ15Q15."""
    if a == b == -0x8000:
        _flag(m, 21)
        return 0x7FFF
    return (((a * b) << 1) + 0x8000) >> 16


def _q15_sat(m, a: int, b: int) -> int:
    """sat16MultiplyQ15Q15."""
    if a == b == -0x8000:
        _flag(m, 21)
        return 0x7FFF
    return ((a * b) << 1) >> 16


def _e_mulq_rs_ph(m, w: int) -> None:
    (sb, sa), (tb, ta) = _halves(_r(m, w, _rs)), _halves(_r(m, w, _rt))
    m.write_reg(_rd(w), _ph(_q15_round(m, sb, tb), _q15_round(m, sa, ta)))


def _e_mulq_s_ph(m, w: int) -> None:
    (sb, sa), (tb, ta) = _halves(_r(m, w, _rs)), _halves(_r(m, w, _rt))
    m.write_reg(_rd(w), _ph(_q15_sat(m, sb, tb), _q15_sat(m, sa, ta)))


def _mulq_w(rnd: bool) -> Executor:
    def ex(m, w: int) -> None:
        a, b = _sx(_r(m, w, _rs), 32), _sx(_r(m, w, _rt), 32)
        if a == b == -(1 << 31):
            _flag(m, 21)
            m.write_reg(_rd(w), 0x7FFFFFFF)
            return
        m.write_reg(_rd(w), (((a * b) << 1) + (0x80000000 if rnd else 0)) >> 32)
    return ex


def _mul_ph(sat: bool) -> Executor:
    def ex(m, w: int) -> None:
        (sb, sa), (tb, ta) = _halves(_r(m, w, _rs)), _halves(_r(m, w, _rt))
        f = _sat if sat else _wrap
        m.write_reg(_rd(w), _ph(f(m, sb * tb, -0x8000, 0x7FFF, 21),
                                f(m, sa * ta, -0x8000, 0x7FFF, 21)))
    return ex


# -------------------------------------------- accumulator: multiply-accumulate

def _acc_signed(m, n: int) -> int:
    return _sx(acc(m, n), 64)


def _dot(kind: str, sub: bool, cross: bool = False) -> Executor:
    """DPA / DPS / MULSA / DPAQ / DPSQ / MULSAQ / DPAU / DPSU / DPAX ..."""
    def ex(m, w: int) -> None:
        n = _ac(w)
        rs, rt = _r(m, w, _rs), _r(m, w, _rt)
        if kind in ("u_l", "u_r"):                       # DPAU / DPSU.H.QBL / QBR
            s, t = _bytes(rs), _bytes(rt)
            i = 0 if kind == "u_l" else 2
            dotp = s[i] * t[i] + s[i + 1] * t[i + 1]
        else:
            (sb, sa), (tb, ta) = _halves(rs), _halves(rt)
            if cross:
                tb, ta = ta, tb
            if kind.startswith("w"):                     # integer halfwords
                pb, pa = sb * tb, sa * ta
            else:                                        # Q15: multiplyQ15Q15
                pb, pa = _mul_q15(m, sb, tb, 16 + n), _mul_q15(m, sa, ta, 16 + n)
            dotp = pb - pa if kind.endswith("sa") else pb + pa
        v = acc(m, n) - dotp if sub else acc(m, n) + dotp
        set_acc(m, n, v)
    return ex


def _dot_sat32(sub: bool) -> Executor:
    """DPAQX_SA.W.PH / DPSQX_SA.W.PH: the accumulator saturates to Q31."""
    def ex(m, w: int) -> None:
        n = _ac(w)
        (sb, sa), (tb, ta) = _halves(_r(m, w, _rs)), _halves(_r(m, w, _rt))
        dotp = _mul_q15(m, sb, ta, 16 + n) + _mul_q15(m, sa, tb, 16 + n)
        v = _sx(_acc_signed(m, n) + (-dotp if sub else dotp), 64)
        set_acc(m, n, _sat(m, v, -(1 << 31), (1 << 31) - 1, 16 + n))
    return ex


def _dpq_sa_l_w(sub: bool) -> Executor:
    """DPAQ_SA.L.W / DPSQ_SA.L.W: a Q31 product into a saturating Q63."""
    def ex(m, w: int) -> None:
        n = _ac(w)
        a, b = _sx(_r(m, w, _rs), 32), _sx(_r(m, w, _rt), 32)
        if a == b == -(1 << 31):
            _flag(m, 16 + n)
            dotp = (1 << 63) - 1
        else:
            dotp = (a * b) << 1
        v = _acc_signed(m, n) + (-dotp if sub else dotp)
        set_acc(m, n, _sat(m, v, -(1 << 63), (1 << 63) - 1, 16 + n))
    return ex


def _maq(left: bool, sat: bool) -> Executor:
    """MAQ_S[A].W.PHL / PHR."""
    def ex(m, w: int) -> None:
        n = _ac(w)
        i = 0 if left else 1
        p = _mul_q15(m, _halves(_r(m, w, _rs))[i], _halves(_r(m, w, _rt))[i], 16 + n)
        # MIPS64 (MD00375 sat32AccumulateQ31: HI[acc]63..0 || LO[acc]63..0
        # + a): bits 32 / 31 of the sum are those of the 64-bit LO + a
        v = (_half(m, n, False) if sat and m.wide else acc(m, n)) + p
        if sat:                                          # sat32AccumulateQ31
            # only bits 32 and 31 of the sum are compared; the page has the
            # limits swapped (temp32 = 0 -> 0x80000000): eMIPS saturates as
            # DPAQX_SA and QEMU do (sim.DEVIATIONS["DSP Module"])
            b32, b31 = (v >> 32) & 1, (v >> 31) & 1
            if b32 != b31:
                _flag(m, 16 + n)
                v = 0x7FFFFFFF if b32 == 0 else -(1 << 31)
            else:
                v = _sx(v, 32)
        set_acc(m, n, v)
    return ex


def _base_ac(op: str) -> Executor:
    """MULT / MULTU / MADD / MADDU / MSUB / MSUBU with ac (the DSP forms)."""
    signed = not op.endswith("U")

    def ex(m, w: int) -> None:
        n = _ac(w)
        a, b = _r(m, w, _rs), _r(m, w, _rt)
        p = _sx(a, 32) * _sx(b, 32) if signed else a * b
        if op.startswith("MULT"):
            set_acc(m, n, p)
        elif op.startswith("MADD"):
            set_acc(m, n, acc(m, n) + p)
        else:
            set_acc(m, n, acc(m, n) - p)
    return ex


def _e_mfhi(m, w: int) -> None:
    """MFHI rd, ac (MIPS64: GPR[rd]63..0 <- HI[ac]63..0)."""
    m.write_dreg(_rd(w), _half(m, (w >> 21) & 3, True))


def _e_mflo(m, w: int) -> None:
    m.write_dreg(_rd(w), _half(m, (w >> 21) & 3, False))


def _e_mthi(m, w: int) -> None:
    """MTHI rs, ac (MIPS64: HI[ac]63..0 <- GPR[rs]63..0)."""
    _set_half(m, _ac(w), True, m.read_reg(_rs(w)))


def _e_mtlo(m, w: int) -> None:
    _set_half(m, _ac(w), False, m.read_reg(_rs(w)))


# ------------------------------------------------------ accumulator: extract

def _shift_acc_65(m, n: int, s: int) -> int:
    """_shiftShortAccRightArithmetic: (ac >> s) with one more bit below
    (a 65-bit value whose bit 0 is the rounding bit)."""
    a = _acc_signed(m, n)
    return (a << 1) if s == 0 else a >> (s - 1)


def _extr_w(mode: str, var: bool) -> Executor:
    """EXTR[_R|_RS].W, EXTRV[_R|_RS].W (ouflag 23)."""
    def ex(m, w: int) -> None:
        n = _ac(w)
        s = (_r(m, w, _rs) if var else w >> 21) & 31
        t = _shift_acc_65(m, n, s)
        if not -(1 << 32) <= t < (1 << 32):          # temp64..32 not all equal
            _flag(m, 23)
        if mode == "":
            m.write_reg(_rt(w), t >> 1)
            return
        t += 1
        if not -(1 << 32) <= t < (1 << 32):
            _flag(m, 23)
            if mode == "rs":
                m.write_reg(_rt(w), 0x7FFFFFFF if t >= 0 else 0x80000000)
                return
        m.write_reg(_rt(w), t >> 1)
    return ex


def _extr_s_h(var: bool) -> Executor:
    def ex(m, w: int) -> None:
        n = _ac(w)
        s = (_r(m, w, _rs) if var else w >> 21) & 31
        m.write_reg(_rt(w), _sat(m, _acc_signed(m, n) >> s, -0x8000, 0x7FFF, 23))
    return ex


def _extp(var: bool, dp: bool) -> Executor:
    """EXTP / EXTPV / EXTPDP / EXTPDPV: size + 1 bits from DSPControl.pos."""
    def ex(m, w: int) -> None:
        n = _ac(w)
        size = (_r(m, w, _rs) if var else w >> 21) & 31
        pos = m.dspctl & POS
        if pos - (size + 1) >= -1:
            v = (acc(m, n) >> (pos - size)) & ((1 << (size + 1)) - 1)
            if var or dp:                   # MIPS64 (MD00375): zero-extended,
                m.write_dreg(_rt(w), v)     # EXTP alone sign-extends (rev. 2.40)
            else:
                m.write_reg(_rt(w), v)
            m.dspctl &= ~(1 << EFI_BIT)
            if dp:
                m.dspctl = (m.dspctl & ~POS) | ((pos - (size + 1)) & POS)
        else:
            m.dspctl |= 1 << EFI_BIT
            m.write_reg(_rt(w), 0)                  # UNPREDICTABLE: QEMU writes 0
    return ex


def _shilo(var: bool) -> Executor:
    def ex(m, w: int) -> None:
        n = _ac(w)
        s = _sx(_r(m, w, _rs) if var else w >> 20, 6)
        set_acc(m, n, acc(m, n) >> s if s >= 0 else acc(m, n) << -s)
    return ex


def _e_mthlip(m, w: int) -> None:
    n = _ac(w)
    set_acc(m, n, ((acc(m, n) & M32) << 32) | _r(m, w, _rs))
    pos = m.dspctl & POS
    if pos <= 32:                                   # > 32: UNPREDICTABLE, kept (QEMU)
        m.dspctl = (m.dspctl & ~POS) | (pos + 32)


# --------------------------------------------------- DSPControl, bits, misc

def _mask_bits(mask: int) -> int:
    return sum(f for i, f in enumerate(MASK_FIELDS) if mask >> i & 1)


def _e_rddsp(m, w: int) -> None:
    m.write_reg(_rd(w), m.dspctl & _mask_bits(w >> 16) & DSPCONTROL_MASK)


def _e_wrdsp(m, w: int) -> None:
    sel = _mask_bits(w >> 11) & DSPCONTROL_MASK
    m.dspctl = (m.dspctl & ~sel) | (_r(m, w, _rs) & sel)


def _e_insv(m, w: int) -> None:
    pos, size = m.dspctl & POS, (m.dspctl >> SCOUNT_SHIFT) & 0x3F
    if not (pos < 32 and 0 < size <= 32 and pos + size <= 32):
        return                                      # UNPREDICTABLE: rt kept (QEMU)
    field = ((1 << size) - 1) << pos
    rt = _r(m, w, _rt)
    m.write_reg(_rt(w), (rt & ~field) | ((_r(m, w, _rs) << pos) & field))


def _e_bitrev(m, w: int) -> None:
    v = _r(m, w, _rt) & 0xFFFF
    m.write_reg(_rd(w), int(f"{v:016b}"[::-1], 2))


def _e_append(m, w: int) -> None:
    s = _rd(w)
    m.write_reg(_rt(w), (_r(m, w, _rt) << s) | (_r(m, w, _rs) & ((1 << s) - 1)))


def _e_prepend(m, w: int) -> None:
    s = _rd(w)
    m.write_reg(_rt(w), ((_r(m, w, _rs) & ((1 << s) - 1)) << (32 - s)) | (_r(m, w, _rt) >> s)
                if s else _r(m, w, _rt))


def _e_balign(m, w: int) -> None:
    bp = (w >> 11) & 3
    if bp in (0, 2):
        return                                      # UNPREDICTABLE: rt kept
    m.write_reg(_rt(w), (_r(m, w, _rt) << (8 * bp)) | (_r(m, w, _rs) >> (8 * (4 - bp))))


def _e_repl_qb(m, w: int) -> None:
    m.write_reg(_rd(w), ((w >> 16) & 0xFF) * 0x01010101)


def _e_repl_ph(m, w: int) -> None:
    v = _sx(w >> 16, 10) & 0xFFFF
    m.write_reg(_rd(w), _ph(v, v))


def _e_replv_qb(m, w: int) -> None:
    m.write_reg(_rd(w), (_r(m, w, _rt) & 0xFF) * 0x01010101)


def _e_replv_ph(m, w: int) -> None:
    v = _r(m, w, _rt)
    m.write_reg(_rd(w), _ph(v, v))


# ------------------------------------------------------- precision expand / reduce

def _preceq_w(left: bool) -> Executor:
    def ex(m, w: int) -> None:
        h = _halves(_r(m, w, _rt), False)[0 if left else 1]
        m.write_reg(_rd(w), h << 16)
    return ex


def _prece(pick: tuple[int, int], frac: bool) -> Executor:
    """PRECEQU.PH.QB* (byte << 7) / PRECEU.PH.QB* (byte): two of the four
    bytes (D C B A = 0 1 2 3)."""
    def ex(m, w: int) -> None:
        q = _bytes(_r(m, w, _rt))
        b, a = q[pick[0]], q[pick[1]]
        m.write_reg(_rd(w), _ph(b << 7, a << 7) if frac else _ph(b, a))
    return ex


def _e_precrq_qb_ph(m, w: int) -> None:
    rs, rt = _r(m, w, _rs), _r(m, w, _rt)
    m.write_reg(_rd(w), _qb(rs >> 24, rs >> 8, rt >> 24, rt >> 8))


def _e_precr_qb_ph(m, w: int) -> None:
    rs, rt = _r(m, w, _rs), _r(m, w, _rt)
    m.write_reg(_rd(w), _qb(rs >> 16, rs, rt >> 16, rt))


def _e_precrq_ph_w(m, w: int) -> None:
    m.write_reg(_rd(w), _ph(_r(m, w, _rs) >> 16, _r(m, w, _rt) >> 16))


def _e_precrq_rs_ph_w(m, w: int) -> None:
    def f(a: int) -> int:                           # trunc16Sat16Round
        v = _sx(a, 32) + 0x8000
        if v > 0x7FFFFFFF:
            _flag(m, 22)
            v = 0x7FFFFFFF
        return v >> 16
    m.write_reg(_rd(w), _ph(f(_r(m, w, _rs)), f(_r(m, w, _rt))))


def _e_precrqu_s_qb_ph(m, w: int) -> None:
    def f(h: int) -> int:                           # sat8ReducePrecision
        if h & 0x8000:
            _flag(m, 22)
            return 0
        if h > 0x7F80:
            _flag(m, 22)
            return 0xFF
        return h >> 7
    rs, rt = _halves(_r(m, w, _rs), False), _halves(_r(m, w, _rt), False)
    m.write_reg(_rd(w), _qb(f(rs[0]), f(rs[1]), f(rt[0]), f(rt[1])))


def _precr_sra(rnd: bool) -> Executor:
    def ex(m, w: int) -> None:
        s = _rd(w)
        rt, rs = _sx(_r(m, w, _rt), 32), _sx(_r(m, w, _rs), 32)
        if rnd and s:
            b, a = ((rt >> (s - 1)) + 1) >> 1, ((rs >> (s - 1)) + 1) >> 1
        else:
            b, a = rt >> s, rs >> s
        m.write_reg(_rt(w), _ph(b, a))
    return ex


# -------------------------------------------------------- loads and branches

def _lx(size: int, signed: bool) -> Executor:
    """LWX / LHX / LBUX / LDX (MIPS64, MD00375): rd <- memory[base + index],
    aligned (AdEL)."""
    def ex(m, w: int) -> None:
        addr = (m.read_reg(_rs(w)) + m.read_reg(_rt(w))) & m.amask
        v = m.load(m.data_address(addr, size, store=False, aligned=True), size)
        if size == 8:
            m.write_dreg(_rd(w), v)
        else:
            m.write_reg(_rd(w), _sx(v, 8 * size) if signed else v)
    return ex


def _bposge32(compact: bool) -> Executor:
    def ex(m, w: int) -> int | None:
        target = (m.pc + 4 + (_sx(w, 16) << 2)) & m.amask
        taken = (m.dspctl & POS) >= 32
        if compact:                                 # BPOSGE32C: no delay slot
            if taken:
                m._redirect = target
            else:
                m._forbidden = True
            return None
        return target if taken else None
    return ex


# ---------------------------------------------------------------- the table

EXECUTORS: dict[str, Executor] = {
    "ABSQ_S.PH": _e_absq_s_ph, "ABSQ_S.QB": _e_absq_s_qb, "ABSQ_S.W": _e_absq_s_w,
    "ADDQ.PH": _pairwise_ph(_add(16, False)), "ADDQ_S.PH": _pairwise_ph(_add(16, True)),
    "ADDQ_S.W": _word(_add(32, True)),
    "ADDQH.PH": _pairwise_ph(_halve(False, False)), "ADDQH_R.PH": _pairwise_ph(_halve(False, True)),
    "ADDQH.W": _word(_halve(False, False)), "ADDQH_R.W": _word(_halve(False, True)),
    "ADDSC": _e_addsc, "ADDWC": _e_addwc,
    "ADDU.QB": _pairwise_qb(_add(8, False, signed=False)),
    "ADDU_S.QB": _pairwise_qb(_add(8, True, signed=False)),
    "ADDU.PH": _pairwise_ph(_add(16, False, signed=False), False),
    "ADDU_S.PH": _pairwise_ph(_add(16, True, signed=False), False),
    "ADDUH.QB": _pairwise_qb(_halve(False, False)), "ADDUH_R.QB": _pairwise_qb(_halve(False, True)),
    "SUBQ.PH": _pairwise_ph(_add(16, False, True)), "SUBQ_S.PH": _pairwise_ph(_add(16, True, True)),
    "SUBQ_S.W": _word(_add(32, True, True)),
    "SUBQH.PH": _pairwise_ph(_halve(True, False)), "SUBQH_R.PH": _pairwise_ph(_halve(True, True)),
    "SUBQH.W": _word(_halve(True, False)), "SUBQH_R.W": _word(_halve(True, True)),
    "SUBU.QB": _pairwise_qb(_add(8, False, True, signed=False)),
    "SUBU_S.QB": _pairwise_qb(_add(8, True, True, signed=False)),
    "SUBU.PH": _pairwise_ph(_add(16, False, True, signed=False), False),
    "SUBU_S.PH": _pairwise_ph(_add(16, True, True, signed=False), False),
    "SUBUH.QB": _pairwise_qb(_halve(True, False)), "SUBUH_R.QB": _pairwise_qb(_halve(True, True)),
    "MODSUB": _e_modsub, "RADDU.W.QB": _e_raddu_w_qb,
    **{f"CMPU.{c}.QB": _cmpu(c) for c in _CMP},
    **{f"CMPGU.{c}.QB": _cmpgu(c, False) for c in _CMP},
    **{f"CMPGDU.{c}.QB": _cmpgu(c, True) for c in _CMP},
    **{f"CMP.{c}.PH": _cmp_ph(c) for c in _CMP},
    "PICK.QB": _e_pick_qb, "PICK.PH": _e_pick_ph, "PACKRL.PH": _e_packrl_ph,
    "SHLL.QB": _shift_qb(_shl(8, False, False), True),
    "SHLLV.QB": _shift_qb(_shl(8, False, False), False),
    "SHLL.PH": _shift_ph(_shl(16, False), True), "SHLLV.PH": _shift_ph(_shl(16, False), False),
    "SHLL_S.PH": _shift_ph(_shl(16, True), True), "SHLLV_S.PH": _shift_ph(_shl(16, True), False),
    "SHLL_S.W": _shift_w(_shl(32, True), True), "SHLLV_S.W": _shift_w(_shl(32, True), False),
    "SHRL.QB": _shift_qb(_shrl(8), True), "SHRLV.QB": _shift_qb(_shrl(8), False),
    "SHRL.PH": _shift_ph(_shrl(16), True), "SHRLV.PH": _shift_ph(_shrl(16), False),
    "SHRA.QB": _shift_qb(_shra(8, False), True), "SHRAV.QB": _shift_qb(_shra(8, False), False),
    "SHRA_R.QB": _shift_qb(_shra(8, True), True), "SHRAV_R.QB": _shift_qb(_shra(8, True), False),
    "SHRA.PH": _shift_ph(_shra(16, False), True), "SHRAV.PH": _shift_ph(_shra(16, False), False),
    "SHRA_R.PH": _shift_ph(_shra(16, True), True), "SHRAV_R.PH": _shift_ph(_shra(16, True), False),
    "SHRA_R.W": _shift_w(_shra(32, True), True), "SHRAV_R.W": _shift_w(_shra(32, True), False),
    "MULEQ_S.W.PHL": _e_muleq_s_w_ph(True), "MULEQ_S.W.PHR": _e_muleq_s_w_ph(False),
    "MULEU_S.PH.QBL": _e_muleu_s_ph_qb(True), "MULEU_S.PH.QBR": _e_muleu_s_ph_qb(False),
    "MULQ_RS.PH": _e_mulq_rs_ph, "MULQ_S.PH": _e_mulq_s_ph,
    "MULQ_RS.W": _mulq_w(True), "MULQ_S.W": _mulq_w(False),
    "MUL.PH": _mul_ph(False), "MUL_S.PH": _mul_ph(True),
    "DPA.W.PH": _dot("w", False), "DPS.W.PH": _dot("w", True),
    "DPAX.W.PH": _dot("w", False, True), "DPSX.W.PH": _dot("w", True, True),
    "MULSA.W.PH": _dot("wsa", False),
    "DPAQ_S.W.PH": _dot("q", False), "DPSQ_S.W.PH": _dot("q", True),
    "DPAQX_S.W.PH": _dot("q", False, True), "DPSQX_S.W.PH": _dot("q", True, True),
    "MULSAQ_S.W.PH": _dot("qsa", False),
    "DPAQX_SA.W.PH": _dot_sat32(False), "DPSQX_SA.W.PH": _dot_sat32(True),
    "DPAU.H.QBL": _dot("u_l", False), "DPAU.H.QBR": _dot("u_r", False),
    "DPSU.H.QBL": _dot("u_l", True), "DPSU.H.QBR": _dot("u_r", True),
    "DPAQ_SA.L.W": _dpq_sa_l_w(False), "DPSQ_SA.L.W": _dpq_sa_l_w(True),
    "MAQ_S.W.PHL": _maq(True, False), "MAQ_S.W.PHR": _maq(False, False),
    "MAQ_SA.W.PHL": _maq(True, True), "MAQ_SA.W.PHR": _maq(False, True),
    **{f"{op}@DSP": _base_ac(op) for op in ("MULT", "MULTU", "MADD", "MADDU", "MSUB", "MSUBU")},
    "MFHI@DSP": _e_mfhi, "MFLO@DSP": _e_mflo, "MTHI@DSP": _e_mthi, "MTLO@DSP": _e_mtlo,
    "EXTR.W": _extr_w("", False), "EXTR_R.W": _extr_w("r", False),
    "EXTR_RS.W": _extr_w("rs", False), "EXTRV.W": _extr_w("", True),
    "EXTRV_R.W": _extr_w("r", True), "EXTRV_RS.W": _extr_w("rs", True),
    "EXTR_S.H": _extr_s_h(False), "EXTRV_S.H": _extr_s_h(True),
    "EXTP": _extp(False, False), "EXTPV": _extp(True, False),
    "EXTPDP": _extp(False, True), "EXTPDPV": _extp(True, True),
    "SHILO": _shilo(False), "SHILOV": _shilo(True), "MTHLIP": _e_mthlip,
    "RDDSP": _e_rddsp, "WRDSP": _e_wrdsp, "INSV": _e_insv, "BITREV": _e_bitrev,
    "APPEND": _e_append, "PREPEND": _e_prepend, "BALIGN": _e_balign,
    "REPL.QB": _e_repl_qb, "REPL.PH": _e_repl_ph,
    "REPLV.QB": _e_replv_qb, "REPLV.PH": _e_replv_ph,
    "PRECEQ.W.PHL": _preceq_w(True), "PRECEQ.W.PHR": _preceq_w(False),
    "PRECEQU.PH.QBL": _prece((0, 1), True), "PRECEQU.PH.QBR": _prece((2, 3), True),
    "PRECEQU.PH.QBLA": _prece((0, 2), True), "PRECEQU.PH.QBRA": _prece((1, 3), True),
    "PRECEU.PH.QBL": _prece((0, 1), False), "PRECEU.PH.QBR": _prece((2, 3), False),
    "PRECEU.PH.QBLA": _prece((0, 2), False), "PRECEU.PH.QBRA": _prece((1, 3), False),
    "PRECRQ.QB.PH": _e_precrq_qb_ph, "PRECR.QB.PH": _e_precr_qb_ph,
    "PRECRQ.PH.W": _e_precrq_ph_w, "PRECRQ_RS.PH.W": _e_precrq_rs_ph_w,
    "PRECRQU_S.QB.PH": _e_precrqu_s_qb_ph,
    "PRECR_SRA.PH.W": _precr_sra(False), "PRECR_SRA_R.PH.W": _precr_sra(True),
    "LWX": _lx(4, False), "LHX": _lx(2, True), "LBUX": _lx(1, False),
    "LDX": _lx(8, False),                               # MIPS64 (MD00375)
    "BPOSGE32": _bposge32(False), "BPOSGE32C": _bposge32(True),
}


# ------------------------------------------------- what the pipelines see

def _names(text: str) -> frozenset[str]:
    return frozenset(text.split())


#: multiply-accumulate: read and write their accumulator (MD00622 Table
#: 2.5 "MAC's"); *_SA saturate after accumulating
MACS = _names("""DPA.W.PH DPS.W.PH DPAX.W.PH DPSX.W.PH MULSA.W.PH DPAQ_S.W.PH DPSQ_S.W.PH
    DPAQX_S.W.PH DPSQX_S.W.PH MULSAQ_S.W.PH DPAQX_SA.W.PH DPSQX_SA.W.PH DPAU.H.QBL
    DPAU.H.QBR DPSU.H.QBL DPSU.H.QBR DPAQ_SA.L.W DPSQ_SA.L.W MAQ_S.W.PHL MAQ_S.W.PHR
    MAQ_SA.W.PHL MAQ_SA.W.PHR MADD@DSP MADDU@DSP MSUB@DSP MSUBU@DSP""")
SATURATING_MACS = frozenset(n for n in MACS if "_SA." in n)
#: write the whole accumulator without reading it
ACC_WRITERS = _names("MULT@DSP MULTU@DSP")
#: read the accumulator (and write a GPR): Table 2.5's EXT* and MFHI / MFLO
ACC_READERS = _names("""EXTR.W EXTR_R.W EXTR_RS.W EXTR_S.H EXTRV.W EXTRV_R.W EXTRV_RS.W
    EXTRV_S.H EXTP EXTPV EXTPDP EXTPDPV""")
#: shift / move within the accumulator: read and write it
ACC_SHIFTS = _names("SHILO SHILOV MTHLIP")
#: the multiplies whose result is a GPR (Table 2.5's MUL*)
GPR_MULTIPLIES = _names("""MUL.PH MUL_S.PH MULEQ_S.W.PHL MULEQ_S.W.PHR MULEU_S.PH.QBL
    MULEU_S.PH.QBR MULQ_RS.PH MULQ_S.PH MULQ_RS.W MULQ_S.W""")
#: write DSPControl (Tables 3.5 and 3.8)
DSPCONTROL_WRITERS = _names("""ABSQ_S.QB ABSQ_S.PH ABSQ_S.W ADDQ.PH ADDQ_S.PH ADDQ_S.W ADDSC
    ADDU.QB ADDU_S.QB ADDU.PH ADDU_S.PH ADDWC SUBQ.PH SUBQ_S.PH SUBQ_S.W SUBU.QB SUBU_S.QB
    SUBU.PH SUBU_S.PH CMPU.EQ.QB CMPU.LT.QB CMPU.LE.QB CMPGDU.EQ.QB CMPGDU.LT.QB
    CMPGDU.LE.QB CMP.EQ.PH CMP.LT.PH CMP.LE.PH MUL.PH MUL_S.PH MULEQ_S.W.PHL MULEQ_S.W.PHR
    MULEU_S.PH.QBL MULEU_S.PH.QBR MULQ_RS.PH MULQ_S.PH MULQ_RS.W MULQ_S.W PRECRQ_RS.PH.W
    PRECRQU_S.QB.PH SHLL.QB SHLLV.QB SHLL.PH SHLLV.PH SHLL_S.PH SHLLV_S.PH SHLL_S.W
    SHLLV_S.W EXTR.W EXTR_R.W EXTR_RS.W EXTR_S.H EXTRV.W EXTRV_R.W EXTRV_RS.W EXTRV_S.H
    EXTP EXTPV EXTPDP EXTPDPV MTHLIP WRDSP DPAQ_S.W.PH DPSQ_S.W.PH DPAQX_S.W.PH
    DPSQX_S.W.PH MULSAQ_S.W.PH DPAQX_SA.W.PH DPSQX_SA.W.PH DPAQ_SA.L.W DPSQ_SA.L.W
    MAQ_S.W.PHL MAQ_S.W.PHR MAQ_SA.W.PHL MAQ_SA.W.PHR""")
#: read DSPControl (Table 3.8)
DSPCONTROL_READERS = _names("""RDDSP BPOSGE32 BPOSGE32C EXTP EXTPV EXTPDP EXTPDPV INSV
    ADDWC PICK.QB PICK.PH MTHLIP""")
#: rows whose destination is rt (the others: rd), and those that also read it
RT_DEST = ACC_READERS | _names("INSV APPEND PREPEND BALIGN PRECR_SRA.PH.W PRECR_SRA_R.PH.W")
LOADS = _names("LWX LHX LBUX LDX")


def accumulator(n: int, half: str) -> object:
    """The pipelines' name of accumulator ``n``'s HI or LO (ac0: "hi" /
    "lo", as the base MULT / MFHI)."""
    return half if n == 0 else (half, n)


def resources(name: str, word: int) -> tuple[set, set]:
    """``(reads, writes)`` beyond the GPRs: the accumulators' halves and
    "dspctl" (the pipelines' hazards)."""
    reads: set = set()
    writes: set = set()
    if name in ("MFHI@DSP", "MFLO@DSP"):
        reads.add(accumulator((word >> 21) & 3, "hi" if name == "MFHI@DSP" else "lo"))
    elif name in ("MTHI@DSP", "MTLO@DSP"):
        writes.add(accumulator(_ac(word), "hi" if name == "MTHI@DSP" else "lo"))
    else:
        both = {accumulator(_ac(word), "hi"), accumulator(_ac(word), "lo")}
        if name in MACS or name in ACC_SHIFTS:
            reads |= both
            writes |= both
        elif name in ACC_WRITERS:
            writes |= both
        elif name in ACC_READERS:
            reads |= both
    if name in DSPCONTROL_READERS:
        reads.add("dspctl")
    if name in DSPCONTROL_WRITERS:
        writes.add("dspctl")
    return reads, writes
