"""MSA, the MIPS SIMD Architecture (PLAN M24 phase 2).

Reference: *MIPS Architecture for Programmers Volume IV-j: The MIPS32 /
MIPS64 SIMD Architecture Module* (MD00866 / MD00868) rev. 1.12 — 3.2
detection and enabling, 3.3 the vector registers and their FPU overlay, 3.4
the control registers, 3.5 the exceptions, ch. 4 every instruction;
MD00870 / MD00872 for microMIPS.  The cores' choices: the P5600 (MD01025
ch. 13), the I6400 (MD01196 ch. 9), the P6600 (MD01138).  The working plan
``m24-MSA-SIMD.md`` has the decisions; ``sim.DEVIATIONS["MSA"]`` the
choices where the manual leaves one open.

State: vector register ``n`` = ``m.wh[n] << 64 | m.fpr[n]`` (3.3.2: its low
64 bits are FPU register ``n`` when Status.FR = 1); ``m.msacsr``; the
core's ``m.msair``.  Element ``i`` of a vector in format ``df`` is bits
``(i + 1) * size - 1 .. i * size`` (Figure 3-1), whatever the byte order —
memory order comes in only through LD / ST (Table 3.1).

Executors run *after* the MSA gate in ``sim.Machine._dispatch`` (RI with
Config3.MSAP = 0 or CU1 = 1 and FR = 0, then MSA Disabled with
Config5.MSAEn = 0).
"""

from __future__ import annotations

from typing import Callable

from . import cp0 as C
from . import fpu as F

M32 = 0xFFFF_FFFF
M64 = (1 << 64) - 1
M128 = (1 << 128) - 1
#: the vector register width (3.1)
WRLEN = 128
#: element size in bits by data format (3.1)
SIZE = {"B": 8, "H": 16, "W": 32, "D": 64}

# ------------------------------------------------------------ MSACSR (3.4.2)

CSR_FS = 1 << 24            # flush to zero
CSR_NX = 1 << 18            # non-trapping exception mode
CAUSE_SHIFT, ENABLES_SHIFT, FLAGS_SHIFT = 12, 7, 2
CAUSE_E = 1 << 17           # Unimplemented Operation (Cause only, always enabled)
#: the bits software can write: FS, NX, Cause, Enables, Flags, RM (Impl,
#: 22:21, is "available to control implementation dependent features":
#: none here, R0 — DEVIATIONS "MSA")
MSACSR_MASK = CSR_FS | CSR_NX | 0x3FFFF
#: MSAIR (3.4.1): WRP 0, ProcessorID 3, Revision 0 — QEMU's I6400 / I6500 /
#: mips32r6-generic value; the core manuals give only "Preset" (decision 8)
MSAIR = 0x00000300
#: MSAIR.WRP: vector register partitioning (3.4.3; phase 2f)
MSAIR_WRP = 1 << 16

# ---------------------------------------------------------- Config3 / Config5

#: Config3.MSAP (Vol. III 9.42, MD00866 3.2)
CFG3_MSAP = 1 << 28
#: Config5.MSAEn (Vol. III 9.44, MD00866 3.2)
CFG5_MSAEN = 1 << 27


# ------------------------------------------------------------ the opcode space

#: COP1 rs values of the branches (Table 3.17): BZ.V, BNZ.V, BZ.df, BNZ.df
BRANCH_RS = frozenset({0x0B, 0x0F, *range(0x18, 0x20)})
_MM_MINORS: frozenset[int] | None = None


def in_space(word: int, micromips: bool = False, arch: str = "mips32") -> bool:
    """Is ``word`` in MSA's opcode space (Tables 3.17 / 3.18: opcode 30 and
    the COP1 branch encodings; microMIPS, MD00870: POOL32S with an MSA
    minor opcode, POOL32D's branch rs values)?  A reserved word there is MSA
    Disabled while Config5.MSAEn = 0 (MD00866 1.11 history)."""
    global _MM_MINORS
    if not micromips:
        op = word >> 26
        return op == 0x1E or op == 0x11 and (word >> 21) & 31 in BRANCH_RS
    if word >> 16 == 0 and word < 0x10000:      # (a 16-bit instruction)
        return False
    major = (word >> 26) & 63
    if major == 0x20:
        return (word >> 21) & 31 in BRANCH_RS
    if major != 0x16:
        return False
    if _MM_MINORS is None:
        from . import micromips as MM
        _MM_MINORS = frozenset(r.value & 63 for r in MM._MSA32 + MM._MSA64
                               if r.value >> 26 == 0x16 and r.form != "DLSA@MSA")
    return word & 63 in _MM_MINORS


# ------------------------------------------------------------ the registers

def vec(m, n: int) -> int:
    """Vector register ``n`` as a 128-bit value."""
    return m.wh[n] << 64 | m.fpr[n]


def set_vec(m, n: int, v: int) -> None:
    m.fpr[n] = v & M64
    m.wh[n] = (v >> 64) & M64


def lanes(v: int, size: int) -> list[int]:
    """The elements of ``v`` (unsigned), element 0 first."""
    mask = (1 << size) - 1
    return [(v >> (i * size)) & mask for i in range(WRLEN // size)]


def pack(values, size: int) -> int:
    """A vector from its elements (element 0 first; each cut to ``size``)."""
    mask = (1 << size) - 1
    v = 0
    for i, e in enumerate(values):
        v |= (e & mask) << (i * size)
    return v


def sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def df_of(name: str) -> int:
    """The element size of row ``name`` (``ADDV.H`` -> 16)."""
    return SIZE[name.rsplit(".", 1)[-1]]


def _wd(w: int) -> int:
    return (w >> 6) & 31


def _ws(w: int) -> int:
    return (w >> 11) & 31


def _wt(w: int) -> int:
    return (w >> 16) & 31


def _n(w: int, size: int) -> int:
    """ELM's element index n: the low 4 / 3 / 2 / 1 bits of df/n."""
    return (w >> 16) & ((WRLEN // size) - 1)


# ------------------------------------------------------------ the control registers

def control_read(m, cs: int) -> int:
    """CFCMSA's value of MSA control register ``cs`` (MD00866 CFCMSA):
    MSAIR, MSACSR; the partitioning registers only with MSAIR.WRP (phase
    2f); a reserved register reads 0."""
    if cs == 0:
        return m.msair
    if cs == 1:
        return m.msacsr
    if m.msair & MSAIR_WRP and 2 <= cs <= 7:
        return m.msa_wrp.read(m, cs)
    return 0


def control_write(m, cd: int, v: int) -> None:
    """CTCMSA: MSACSR (then the MSA Floating-Point exception when a Cause
    bit and its Enable — or Cause.E — are set: the register is written
    first, EPC = the CTCMSA); the writable partitioning registers with WRP;
    every other write is ignored."""
    v &= M32
    if cd == 1:
        m.msacsr = v & MSACSR_MASK
        cause = (m.msacsr >> CAUSE_SHIFT) & 0x3F
        if cause & (0x20 | (m.msacsr >> ENABLES_SHIFT) & 0x1F):
            from .sim import _Arch
            raise _Arch(C.MSAFPE)
    elif m.msair & MSAIR_WRP and cd in (3, 4, 6, 7):
        m.msa_wrp.write(m, cd, v)


def _privileged(m, n: int) -> None:
    """CFCMSA / CTCMSA of MSAAccess ... MSAUnmap (only with WRP) need
    Coprocessor 0 access: else Coprocessor Unusable 0 (3.4.3-3.4.8)."""
    if 2 <= n <= 7 and m.msair & MSAIR_WRP and not (m.kernel_mode
                                                   or m.status & C.ST_CU0):
        from .sim import _Arch
        raise _Arch(C.CPU, ce=0)


def _e_cfcmsa(m, w: int) -> None:
    _privileged(m, _ws(w))
    v = control_read(m, _ws(w))
    m.write_reg((w >> 6) & 31, v)               # (sign-extended on MIPS64)


def _e_ctcmsa(m, w: int) -> None:
    _privileged(m, _wd(w))
    control_write(m, _wd(w), m.read_reg(_ws(w)))


# ------------------------------------------------------------ data movement (ch. 4)

def _e_move_v(m, w: int) -> None:
    set_vec(m, _wd(w), vec(m, _ws(w)))


def _copy(signed: bool, size: int) -> Callable:
    """COPY_S / COPY_U.df rd, ws[n]: the element, sign- / zero-extended to
    the GPR (MIPS64: COPY_S.W sign-extends, COPY_U.W zero-extends)."""
    def run(m, w: int) -> None:
        e = lanes(vec(m, _ws(w)), size)[_n(w, size)]
        v = sx(e, size) & m.amask if signed else e
        m.write_dreg((w >> 6) & 31, v)
    return run


def _fill(size: int) -> Callable:
    """FILL.df wd, rs: every element = the low bits of GPR rs."""
    def run(m, w: int) -> None:
        g = m.regs[_ws(w)]
        set_vec(m, _wd(w), pack([g] * (WRLEN // size), size))
    return run


def _insert(size: int) -> Callable:
    """INSERT.df wd[n], rs: element n = the low bits of GPR rs."""
    def run(m, w: int) -> None:
        es = lanes(vec(m, _wd(w)), size)
        es[_n(w, size)] = m.regs[_ws(w)]
        set_vec(m, _wd(w), pack(es, size))
    return run


def _insve(size: int) -> Callable:
    """INSVE.df wd[n], ws[0]: element n = element 0 of ws."""
    def run(m, w: int) -> None:
        es = lanes(vec(m, _wd(w)), size)
        es[_n(w, size)] = lanes(vec(m, _ws(w)), size)[0]
        set_vec(m, _wd(w), pack(es, size))
    return run


def _splat(size: int) -> Callable:
    """SPLAT.df wd, ws[rt]: every element = element GPR[rt] mod the count."""
    def run(m, w: int) -> None:
        k = m.regs[_wt(w)] % (WRLEN // size)
        e = lanes(vec(m, _ws(w)), size)[k]
        set_vec(m, _wd(w), pack([e] * (WRLEN // size), size))
    return run


def _splati(size: int) -> Callable:
    """SPLATI.df wd, ws[n]."""
    def run(m, w: int) -> None:
        e = lanes(vec(m, _ws(w)), size)[_n(w, size)]
        set_vec(m, _wd(w), pack([e] * (WRLEN // size), size))
    return run


def _ldi(size: int) -> Callable:
    """LDI.df wd, s10: every element = s10 sign-extended (a byte takes its
    low 8 bits)."""
    def run(m, w: int) -> None:
        v = sx((w >> 11) & 0x3FF, 10)
        set_vec(m, _wd(w), pack([v] * (WRLEN // size), size))
    return run


# ------------------------------------------------------------ LD / ST (MI10)

def _address(m, w: int, size: int) -> int:
    """LD / ST's address: GPR[rs] + s10 x the element size (bytes)."""
    return m.regs[_ws(w)] + sx((w >> 16) & 0x3FF, 10) * (size // 8)


def _order(m, raw: bytes, size: int) -> int:
    """16 bytes in memory order -> the vector: element i at byte i * size / 8,
    each element in the core's byte order (Table 3.1)."""
    if not m.big_endian:
        return int.from_bytes(raw, "little")
    n = size // 8
    return pack([int.from_bytes(raw[i:i + n], "big") for i in range(0, 16, n)], size)


def _unorder(m, v: int, size: int) -> bytes:
    if not m.big_endian:
        return v.to_bytes(16, "little")
    n = size // 8
    return b"".join(e.to_bytes(n, "big") for e in lanes(v, size))


def _ld(size: int) -> Callable:
    """LD.df wd, s10(rs): 16 bytes, no alignment restriction (1.12: atomic
    per element only when element-aligned; misaligned in hardware, decision
    9), one or two translations, a fault on either before the register is
    written."""
    def run(m, w: int) -> None:
        addr = m.msa_address(_address(m, w, size), store=False)
        if m.msa_ade and addr % (size // 8):
            m.msa_misaligned(addr, store=False)
        value = m.load(addr, 16)
        raw = value.to_bytes(16, "big" if m.big_endian else "little")
        set_vec(m, _wd(w), _order(m, raw, size))
    return run


def _st(size: int) -> Callable:
    """ST.df wd, s10(rs): both pages translated before a byte is written."""
    def run(m, w: int) -> None:
        addr = m.msa_address(_address(m, w, size), store=True)
        if m.msa_ade and addr % (size // 8):
            m.msa_misaligned(addr, store=True)
        raw = _unorder(m, vec(m, _wd(w)), size)
        m.store(addr, int.from_bytes(raw, "big" if m.big_endian else "little"), 16)
    return run


# ------------------------------------------------------------ branches (COP1)

def _zero_elem(size: int) -> Callable[[int], bool]:
    return lambda v: any(e == 0 for e in lanes(v, size))


#: the branch conditions: BZ.V (all of wt zero), BNZ.V (any bit set), BZ.df
#: (some element zero), BNZ.df (no element zero)
CONDITIONS: dict[str, Callable[[int], bool]] = {
    "BZ.V": lambda v: v == 0,
    "BNZ.V": lambda v: v != 0,
    **{f"BZ.{df}": _zero_elem(s) for df, s in SIZE.items()},
    **{f"BNZ.{df}": (lambda z: lambda v: not z(v))(_zero_elem(s)) for df, s in SIZE.items()},
}


def _branch(cond: Callable[[int], bool]) -> Callable:
    """BZ / BNZ wt, s16: a delay slot (Release 6 too), the target from the
    delay slot (the s16 offset: see the BZ / BNZ corrections' why)."""
    def run(m, w: int) -> int | None:
        if cond(vec(m, _wt(w))):
            return (m.pc + 4 + (sx(w, 16) << 2)) & m.amask
        return None
    return run


# ------------------------------------------------------------ integer (phase 2b)
#
# Each element operation follows its page's Operation and the helper
# functions there (abs, sat_s, sat_u, ave, srar ...), written on Python
# integers: a saturation of the exact result is what the pseudocode's
# n + 1-bit arithmetic computes.

def _s(v: int, n: int) -> int:
    return v - (1 << n) if v >> (n - 1) & 1 else v


def _clamp_s(v: int, n: int) -> int:
    return max(-(1 << (n - 1)), min((1 << (n - 1)) - 1, v))


def _clamp_u(v: int, n: int) -> int:
    return max(0, min((1 << n) - 1, v))


def _abs(v: int, n: int) -> int:
    """abs(tt, n): the n-bit magnitude (abs of the most negative = itself,
    read unsigned: 2^(n-1))."""
    return (-_s(v, n)) & ((1 << n) - 1) if v >> (n - 1) & 1 else v


def _div(a: int, b: int) -> int:
    """Integer division truncated toward zero (DIV_S: the C / MIPS rule)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _mod(a: int, b: int) -> int:
    """The remainder with the dividend's sign (MOD_S)."""
    return a - _div(a, b) * b


#: 3R element operations (ws element, wt element, element size) -> result
#: (any integer: cut to the element size)
_3R: dict[str, Callable[[int, int, int], int]] = {
    "ADDV": lambda a, b, n: a + b,
    "SUBV": lambda a, b, n: a - b,
    "MAX_S": lambda a, b, n: a if _s(b, n) < _s(a, n) else b,
    "MAX_U": lambda a, b, n: a if b < a else b,
    "MIN_S": lambda a, b, n: a if _s(b, n) > _s(a, n) else b,
    "MIN_U": lambda a, b, n: a if b > a else b,
    "MAX_A": lambda a, b, n: a if _abs(b, n) < _abs(a, n) else b,
    "MIN_A": lambda a, b, n: a if _abs(b, n) > _abs(a, n) else b,
    "CEQ": lambda a, b, n: -1 if a == b else 0,
    "CLT_S": lambda a, b, n: -1 if _s(a, n) < _s(b, n) else 0,
    "CLT_U": lambda a, b, n: -1 if a < b else 0,
    "CLE_S": lambda a, b, n: -1 if _s(a, n) <= _s(b, n) else 0,
    "CLE_U": lambda a, b, n: -1 if a <= b else 0,
    "ADD_A": lambda a, b, n: _abs(a, n) + _abs(b, n),
    "ADDS_A": lambda a, b, n: _clamp_s(_abs(a, n) + _abs(b, n), n),
    "ADDS_S": lambda a, b, n: _clamp_s(_s(a, n) + _s(b, n), n),
    "ADDS_U": lambda a, b, n: _clamp_u(a + b, n),
    "AVE_S": lambda a, b, n: (_s(a, n) + _s(b, n)) >> 1,
    "AVE_U": lambda a, b, n: (a + b) >> 1,
    "AVER_S": lambda a, b, n: (_s(a, n) + _s(b, n) + 1) >> 1,
    "AVER_U": lambda a, b, n: (a + b + 1) >> 1,
    "SUBS_S": lambda a, b, n: _clamp_s(_s(a, n) - _s(b, n), n),
    "SUBS_U": lambda a, b, n: _clamp_u(a - b, n),
    "SUBSUS_U": lambda a, b, n: _clamp_u(a - _s(b, n), n),
    "SUBSUU_S": lambda a, b, n: _clamp_s(a - b, n),
    "ASUB_S": lambda a, b, n: abs(_s(a, n) - _s(b, n)),
    "ASUB_U": lambda a, b, n: abs(a - b),
    "MULV": lambda a, b, n: a * b,
    # division by zero: no exception; the result is UNPREDICTABLE (DIV_S
    # ... pages) — eMIPS gives QEMU's: see _3R_DIV
    "SLL": lambda a, b, n: a << (b % n),
    "SRA": lambda a, b, n: _s(a, n) >> (b % n),
    "SRL": lambda a, b, n: a >> (b % n),
    "SRAR": lambda a, b, n: _srar(a, b % n, n),
    "SRLR": lambda a, b, n: _srlr(a, b % n, n),
    "BCLR": lambda a, b, n: a & ~(1 << (b % n)),
    "BSET": lambda a, b, n: a | 1 << (b % n),
    "BNEG": lambda a, b, n: a ^ 1 << (b % n),
}


def _srar(a: int, k: int, n: int) -> int:
    """srar(ts, k, n): shift right arithmetic, rounding (bit k - 1 added)."""
    return _s(a, n) if k == 0 else (_s(a, n) >> k) + (a >> (k - 1) & 1)


def _srlr(a: int, k: int, n: int) -> int:
    return a if k == 0 else (a >> k) + (a >> (k - 1) & 1)


#: DIV / MOD by zero (UNPREDICTABLE: DEVIATIONS "MSA") — QEMU's choices
#: (msa_helper.c msa_div_s_df ...): DIV_S -> -1 if the dividend is >= 0
#: else 1, DIV_U -> all ones, MOD_* -> the dividend; and DIV_S of the most
#: negative by -1 -> the most negative, MOD_S -> 0
def _div_s(a: int, b: int, n: int) -> int:
    x, y = _s(a, n), _s(b, n)
    if y == 0:
        return -1 if x >= 0 else 1
    if x == -(1 << (n - 1)) and y == -1:
        return x
    return _div(x, y)


def _mod_s(a: int, b: int, n: int) -> int:
    x, y = _s(a, n), _s(b, n)
    if y == 0:
        return x
    if x == -(1 << (n - 1)) and y == -1:
        return 0
    return _mod(x, y)


_3R.update({
    "DIV_S": _div_s,
    "DIV_U": lambda a, b, n: a // b if b else (1 << n) - 1,
    "MOD_S": _mod_s,
    "MOD_U": lambda a, b, n: a % b if b else a,
})

#: 3R operations that also read wd (MADDV, MSUBV, BINSL, BINSR), as
#: (wd element, ws element, wt element, n) -> result
_3RD: dict[str, Callable[[int, int, int, int], int]] = {
    "MADDV": lambda d, a, b, n: d + a * b,
    "MSUBV": lambda d, a, b, n: d - a * b,
    "BINSL": lambda d, a, b, n: _binsl(d, a, b % n + 1, n),
    "BINSR": lambda d, a, b, n: _binsr(d, a, b % n + 1, n),
}


def _binsl(d: int, a: int, k: int, n: int) -> int:
    """BINSL: the k left (most significant) bits of a into d."""
    mask = ((1 << k) - 1) << (n - k)
    return d & ~mask | a & mask


def _binsr(d: int, a: int, k: int, n: int) -> int:
    mask = (1 << k) - 1
    return d & ~mask | a & mask


def _3r(op: Callable, size: int) -> Callable:
    def run(m, w: int) -> None:
        a, b = lanes(vec(m, _ws(w)), size), lanes(vec(m, _wt(w)), size)
        set_vec(m, _wd(w), pack([op(x, y, size) for x, y in zip(a, b)], size))
    return run


def _3rd(op: Callable, size: int) -> Callable:
    def run(m, w: int) -> None:
        d = lanes(vec(m, _wd(w)), size)
        a, b = lanes(vec(m, _ws(w)), size), lanes(vec(m, _wt(w)), size)
        set_vec(m, _wd(w), pack([op(z, x, y, size) for z, x, y in zip(d, a, b)], size))
    return run


def _wide(op: Callable[[int, int, int], int], size: int, *, acc: int = 0) -> Callable:
    """DOTP / DPADD / DPSUB / HADD / HSUB.df: elements of ``size`` from
    pairs of ``size / 2``-bit elements (op(odd / even halves ...)); ``acc``
    +1 / -1 adds / subtracts the product to / from wd."""
    h = size // 2

    def run(m, w: int) -> None:
        a, b = lanes(vec(m, _ws(w)), h), lanes(vec(m, _wt(w)), h)
        d = lanes(vec(m, _wd(w)), size)
        out = []
        for i in range(WRLEN // size):
            r = op(a[2 * i], a[2 * i + 1], b[2 * i], b[2 * i + 1], h)
            out.append(d[i] + acc * r if acc else r)
        set_vec(m, _wd(w), pack(out, size))
    return run


def _dotp_s(a0, a1, b0, b1, h):
    return _s(a1, h) * _s(b1, h) + _s(a0, h) * _s(b0, h)


def _dotp_u(a0, a1, b0, b1, h):
    return a1 * b1 + a0 * b0


#: HADD / HSUB: the odd element of ws with the even one of wt
_HORIZONTAL = {
    "HADD_S": lambda a0, a1, b0, b1, h: _s(a1, h) + _s(b0, h),
    "HADD_U": lambda a0, a1, b0, b1, h: a1 + b0,
    "HSUB_S": lambda a0, a1, b0, b1, h: _s(a1, h) - _s(b0, h),
    "HSUB_U": lambda a0, a1, b0, b1, h: a1 - b0,
}

#: I5 rows: (the 3R operation, the immediate is s5 — else u5)
_I5 = {"ADDVI": ("ADDV", False), "SUBVI": ("SUBV", False), "MAXI_S": ("MAX_S", True),
       "MAXI_U": ("MAX_U", False), "MINI_S": ("MIN_S", True), "MINI_U": ("MIN_U", False),
       "CEQI": ("CEQ", True), "CLTI_S": ("CLT_S", True), "CLTI_U": ("CLT_U", False),
       "CLEI_S": ("CLE_S", True), "CLEI_U": ("CLE_U", False)}


def _i5(base: str, size: int, signed: bool) -> Callable:
    op = _3R[base]

    def run(m, w: int) -> None:
        imm = (w >> 16) & 31
        if signed:
            imm = sx(imm, 5)
        imm &= (1 << size) - 1
        a = lanes(vec(m, _ws(w)), size)
        set_vec(m, _wd(w), pack([op(x, imm, size) for x in a], size))
    return run


def _bit(base: str, size: int) -> Callable:
    """BIT format: m = the low log2(size) bits of df/m."""
    def run(m, w: int) -> None:
        k = (w >> 16) & (size - 1)
        a = lanes(vec(m, _ws(w)), size)
        if base in ("BINSLI", "BINSRI"):
            d = lanes(vec(m, _wd(w)), size)
            f = _binsl if base == "BINSLI" else _binsr
            out = [f(z, x, k + 1, size) for z, x in zip(d, a)]
        elif base == "SAT_S":
            out = [_clamp_s(_s(x, size), k + 1) for x in a]
        elif base == "SAT_U":
            out = [_clamp_u(x, k + 1) for x in a]
        else:
            op = _3R[base[:-1]]                 # SLLI -> SLL ...
            out = [op(x, k, size) for x in a]
        set_vec(m, _wd(w), pack(out, size))
    return run


def _count(f: Callable[[int, int], int], size: int) -> Callable:
    def run(m, w: int) -> None:
        set_vec(m, _wd(w), pack([f(x, size) for x in lanes(vec(m, _ws(w)), size)], size))
    return run


def _nlzc(x: int, n: int) -> int:
    return n - x.bit_length()


def _nloc(x: int, n: int) -> int:
    return _nlzc(~x & ((1 << n) - 1), n)


def _pcnt(x: int, n: int) -> int:
    return bin(x).count("1")


# the logical operations (VEC, I8): whole vectors / bytes
def _vec(op: Callable[[int, int, int], int]) -> Callable:
    """AND.V ... BSEL.V: op(wd, ws, wt) on the 128-bit values."""
    def run(m, w: int) -> None:
        set_vec(m, _wd(w), op(vec(m, _wd(w)), vec(m, _ws(w)), vec(m, _wt(w))) & M128)
    return run


_VEC = {
    "AND.V": lambda d, s, t: s & t,
    "OR.V": lambda d, s, t: s | t,
    "NOR.V": lambda d, s, t: ~(s | t),
    "XOR.V": lambda d, s, t: s ^ t,
    "BMNZ.V": lambda d, s, t: s & t | d & ~t,      # ws where wt is 1
    "BMZ.V": lambda d, s, t: s & ~t | d & t,       # ws where wt is 0
    "BSEL.V": lambda d, s, t: t & d | s & ~d,      # wt where wd is 1, else ws
}


def _i8(op: Callable[[int, int, int], int]) -> Callable:
    """ANDI.B ... BSELI.B: op(wd, ws, the immediate in every byte)."""
    def run(m, w: int) -> None:
        imm = int.from_bytes(bytes([(w >> 16) & 0xFF]) * 16, "little")
        set_vec(m, _wd(w), op(vec(m, _wd(w)), vec(m, _ws(w)), imm) & M128)
    return run


_I8 = {
    "ANDI.B": lambda d, s, i: s & i,
    "ORI.B": lambda d, s, i: s | i,
    "NORI.B": lambda d, s, i: ~(s | i),
    "XORI.B": lambda d, s, i: s ^ i,
    "BMNZI.B": lambda d, s, i: s & i | d & ~i,
    "BMZI.B": lambda d, s, i: s & ~i | d & i,
    "BSELI.B": lambda d, s, i: i & d | s & ~d,
}


def _integer_executors() -> dict[str, Callable]:
    ex: dict[str, Callable] = {}
    for df, size in SIZE.items():
        for base, op in _3R.items():
            ex[f"{base}.{df}"] = _3r(op, size)
        for base, op in _3RD.items():
            ex[f"{base}.{df}"] = _3rd(op, size)
        for name, (base, signed) in _I5.items():
            ex[f"{name}.{df}"] = _i5(base, size, signed)
        for base in ("SLLI", "SRAI", "SRLI", "SRARI", "SRLRI", "BCLRI", "BSETI", "BNEGI",
                     "BINSLI", "BINSRI", "SAT_S", "SAT_U"):
            ex[f"{base}.{df}"] = _bit(base, size)
        ex[f"NLZC.{df}"] = _count(_nlzc, size)
        ex[f"NLOC.{df}"] = _count(_nloc, size)
        ex[f"PCNT.{df}"] = _count(_pcnt, size)
    for df, size in (("H", 16), ("W", 32), ("D", 64)):
        ex[f"DOTP_S.{df}"] = _wide(_dotp_s, size)
        ex[f"DOTP_U.{df}"] = _wide(_dotp_u, size)
        ex[f"DPADD_S.{df}"] = _wide(_dotp_s, size, acc=1)
        ex[f"DPADD_U.{df}"] = _wide(_dotp_u, size, acc=1)
        ex[f"DPSUB_S.{df}"] = _wide(_dotp_s, size, acc=-1)
        ex[f"DPSUB_U.{df}"] = _wide(_dotp_u, size, acc=-1)
        for name, op in _HORIZONTAL.items():
            ex[f"{name}.{df}"] = _wide(op, size)
    for name, op in _VEC.items():
        ex[name] = _vec(op)
    for name, op in _I8.items():
        ex[name] = _i8(op)
    return ex


# ------------------------------------------------------------ permutes (phase 2c)

def _slide(d: int, s: int, n: int, size: int) -> int:
    """SLD / SLDI's slide (the pages' Operation): the vector seen as
    ``size / 8`` rows of ``WRLEN / size`` bytes (row k = bytes k x columns
    ...); in each row, wd's bytes then ws's are one ``2 x columns``-byte
    string (wd the high part), and the result's byte i is its byte i + n."""
    cols = WRLEN // size                      # bytes in a row = the element count
    db, sb = d.to_bytes(16, "little"), s.to_bytes(16, "little")
    out = bytearray(16)
    for k in range(size // 8):
        row = sb[k * cols:(k + 1) * cols] + db[k * cols:(k + 1) * cols]   # (ws low)
        for i in range(cols):
            out[k * cols + i] = row[i + n]
    return int.from_bytes(out, "little")


def _sld(size: int) -> Callable:
    def run(m, w: int) -> None:
        n = m.regs[_wt(w)] % (WRLEN // size)
        set_vec(m, _wd(w), _slide(vec(m, _wd(w)), vec(m, _ws(w)), n, size))
    return run


def _sldi(size: int) -> Callable:
    def run(m, w: int) -> None:
        set_vec(m, _wd(w), _slide(vec(m, _wd(w)), vec(m, _ws(w)), _n(w, size), size))
    return run


def _vshf(size: int) -> Callable:
    """VSHF.df: wd's elements choose from ws || wt (wt = elements 0 ...):
    bits 5..0 modulo twice the count; bit 6 or 7 set -> 0."""
    count = WRLEN // size

    def run(m, w: int) -> None:
        src = lanes(vec(m, _wt(w)), size) + lanes(vec(m, _ws(w)), size)
        out = [0 if c & 0xC0 else src[(c & 63) % (2 * count)]
               for c in lanes(vec(m, _wd(w)), size)]
        set_vec(m, _wd(w), pack(out, size))
    return run


def _shf(size: int) -> Callable:
    """SHF.df: every set of 4 elements shuffled by i8 (2 bits an element)."""
    def run(m, w: int) -> None:
        i8 = (w >> 16) & 0xFF
        a = lanes(vec(m, _ws(w)), size)
        set_vec(m, _wd(w), pack([a[i - i % 4 + (i8 >> 2 * (i % 4) & 3)]
                                 for i in range(len(a))], size))
    return run


def _pair(pick: Callable[[list, list, int], list]) -> Callable[[int], Callable]:
    """PCKEV / PCKOD / ILVL / ILVR / ILVEV / ILVOD: pick(ws, wt, count)."""
    def make(size: int) -> Callable:
        def run(m, w: int) -> None:
            a, b = lanes(vec(m, _ws(w)), size), lanes(vec(m, _wt(w)), size)
            set_vec(m, _wd(w), pack(pick(a, b, len(a)), size))
        return run
    return make


def _interleave(x: list, y: list) -> list:
    """wd[2i] = x[i], wd[2i + 1] = y[i]."""
    return [e for pair in zip(x, y) for e in pair]


_PERMUTES = {
    "PCKEV": _pair(lambda a, b, c: b[0::2] + a[0::2]),     # right half: wt's
    "PCKOD": _pair(lambda a, b, c: b[1::2] + a[1::2]),
    "ILVL": _pair(lambda a, b, c: _interleave(b[c // 2:], a[c // 2:])),
    "ILVR": _pair(lambda a, b, c: _interleave(b[:c // 2], a[:c // 2])),
    "ILVEV": _pair(lambda a, b, c: _interleave(b[0::2], a[0::2])),
    "ILVOD": _pair(lambda a, b, c: _interleave(b[1::2], a[1::2])),
}


def _permute_executors() -> dict[str, Callable]:
    ex: dict[str, Callable] = {}
    for df, size in SIZE.items():
        ex[f"SLD.{df}"] = _sld(size)
        ex[f"SLDI.{df}"] = _sldi(size)
        ex[f"VSHF.{df}"] = _vshf(size)
        for name, make in _PERMUTES.items():
            ex[f"{name}.{df}"] = make(size)
    for df in "BHW":
        ex[f"SHF.{df}"] = _shf(SIZE[df])
    return ex


# ------------------------------------------------------------ fixed point (phase 2d)
#
# Q15 / Q31 (.H / .W): the pages' q_mul / q_mulr / q_madd / q_maddr /
# q_msub / q_msubr — the product of two n-bit fractions is 2n bits with
# two sign bits; the result keeps bits 2n-2 .. n-1 (rounded: + 1 << (n-2)),
# saturated.  -1 x -1 is the one product that overflows: q_mul returns the
# largest fraction.

def _q_mul(a: int, b: int, n: int, rnd: bool) -> int:
    if a == b == 1 << (n - 1):
        return (1 << (n - 1)) - 1
    p = _s(a, n) * _s(b, n) + ((1 << (n - 2)) if rnd else 0)
    return p >> (n - 1)


def _q_acc(d: int, a: int, b: int, n: int, sub: bool, rnd: bool) -> int:
    p = _s(a, n) * _s(b, n)
    acc = (_s(d, n) << (n - 1)) + (-p if sub else p) + ((1 << (n - 2)) if rnd else 0)
    return _clamp_s(acc >> (n - 1), n)


def _fixed_executors() -> dict[str, Callable]:
    ex: dict[str, Callable] = {}
    for df, size in (("H", 16), ("W", 32)):
        ex[f"MUL_Q.{df}"] = _3r(lambda a, b, n: _q_mul(a, b, n, False), size)
        ex[f"MULR_Q.{df}"] = _3r(lambda a, b, n: _q_mul(a, b, n, True), size)
        for name, sub, rnd in (("MADD_Q", False, False), ("MADDR_Q", False, True),
                               ("MSUB_Q", True, False), ("MSUBR_Q", True, True)):
            ex[f"{name}.{df}"] = _3rd((lambda sb, rd: lambda d, a, b, n:
                                       _q_acc(d, a, b, n, sb, rd))(sub, rnd), size)
    return ex


# ------------------------------------------------------------ floating point (phase 2e)
#
# IEEE 754-2008 arithmetic from emips.fpu (exact, one rounding) with
# MSACSR in place of FCSR: RM, FS (flush subnormal inputs — Inexact — and
# results tiny *before* rounding — Underflow + Inexact, 3.5.4), the 2008
# NaNs (3.3.2: an MSA core is a NAN2008 core).  Each element gives a result
# and its exceptions c; _Fp.element is Figure 3.26 (the Cause update, the
# NX signaling NaN), _Fp.finish Figure 3.27 (Flags, or the MSA FP
# exception with the destination unwritten).

#: binary16 (FEXDO.H, FEXUPL / FEXUPR.W: 3.5.3, Table 3.7)
H = F.Fmt("H", 5, 10)
FMT = {16: H, 32: F.S, 64: F.D}
#: the Cause / Enable / Flag bits (E V Z O U I = 5..0)
_BIT = {"E": 0x20, "V": 0x10, "Z": 0x08, "O": 0x04, "U": 0x02, "I": 0x01}
#: the signaling NaN NX writes, before its low 6 bits (Table 3.7)
_NX_NAN = {16: 0x7C00, 32: 0x7F800000, 64: 0x7FF0000000000000}


def _bits(flags: set[str]) -> int:
    c = 0
    for x in flags:
        c |= _BIT[x]
    return c


class _Fp:
    """One floating-point instruction's MSACSR bookkeeping (Figures 3.26 /
    3.27): Cause cleared first, each element's exceptions folded in, then
    the Flags update or the trap."""

    def __init__(self, m) -> None:
        self.m = m
        csr = m.msacsr
        self.rm = csr & 3
        self.fs = bool(csr & CSR_FS)
        self.nx = bool(csr & CSR_NX)
        self.enable = ((csr >> ENABLES_SHIFT) & 0x1F) | 0x20     # E always enabled
        self.cause = 0
        self.trap = False

    @property
    def flush(self):
        """emips.fpu's flush mode: tiny results flushed before rounding."""
        return "before" if self.fs else False

    def input(self, x: int, size: int, *, compare: bool = False) -> tuple[int, set[str]]:
        """FS: a subnormal input -> zero of its sign (Inexact, but not for a
        compare); 16-bit values are never flushed."""
        f = FMT[size]
        if self.fs and size != 16 and (x >> f.fbits) & ((1 << f.ebits) - 1) == 0 \
                and x & ((1 << f.fbits) - 1):
            return x & f.sign_bit, (set() if compare else {"I"})
        return x, set()

    def element(self, r: int, flags: set[str], size: int, *, tiny: bool = False) -> int:
        """Figure 3.26: the value written for an element with result ``r``
        (the default result of a disabled exception, as emips.fpu gives it)
        and exceptions ``flags``; ``tiny``: an exact tiny result (U only
        if enabled)."""
        c = _bits(flags)
        if tiny and not c & 0x02:
            c |= 0x02
        enable = self.enable
        if c & 0x04 and not enable & 0x04:      # O not enabled: I
            c |= 0x01
        if c & 0x02 and not enable & 0x02 and not c & 0x01:
            c ^= 0x02                           # exact underflow, U not enabled
        if not c & enable:
            self.cause |= c
            return r
        if not self.nx:
            self.cause |= c
            self.trap = True
            return r
        return _NX_NAN[size] | c                # ((e >> 6) << 6) | c

    def finish(self) -> bool:
        """Figure 3.27: write MSACSR; True when the destination may be
        written (else the MSA Floating-Point exception follows)."""
        m = self.m
        m.msacsr = (m.msacsr & ~(0x3F << CAUSE_SHIFT)) | (self.cause << CAUSE_SHIFT)
        if self.cause & self.enable:
            from .sim import _Arch
            raise _Arch(C.MSAFPE)
        m.msacsr |= (self.cause & 0x1F) << FLAGS_SHIFT
        return True


def _tiny(r: int, size: int) -> bool:
    """A subnormal result (tiny, exact or not)."""
    f = FMT[size]
    return (r >> f.fbits) & ((1 << f.ebits) - 1) == 0 and bool(r & ((1 << f.fbits) - 1))


def _fp_run(m, w: int, size: int, out_size: int, compute) -> None:
    """Run ``compute(fp)`` -> the destination's elements (each through
    fp.element), finish, write wd."""
    fp = _Fp(m)
    values = compute(fp)
    if fp.finish():
        set_vec(m, _wd(w), pack(values, out_size))


def _arith(op, size: int, *, nargs: int = 2, reads_wd: bool = False, tiny: bool = True,
           ints: int = 0, reciprocal: bool = False) -> Callable:
    """FADD / FSUB / FMUL / FDIV / FSQRT / FMADD / FMSUB ...: ``op(fp, xs,
    size)`` -> (bits, flags) per element, inputs flushed per FS (but the
    last ``ints`` operands: FEXP2's integer wt); ``tiny``: a subnormal result
    signals Underflow (not for FMIN / FMAX ..., which compute nothing);
    ``reciprocal``: FRCP / FRSQRT's approximation rule (below)."""
    def run(m, w: int) -> None:
        srcs = [lanes(vec(m, _ws(w)), size)]
        if nargs > 1:
            srcs.append(lanes(vec(m, _wt(w)), size))
        if reads_wd:
            srcs.insert(0, lanes(vec(m, _wd(w)), size))

        def compute(fp: _Fp) -> list[int]:
            out = []
            for xs in zip(*srcs):
                flags: set[str] = set()
                ins = []
                for k, x in enumerate(xs):
                    if k < len(xs) - ints:
                        x, fl = fp.input(x, size)
                        flags |= fl
                    ins.append(x)
                r, fl = op(fp, ins, size)
                flags |= fl
                if reciprocal and not flags & {"V", "Z"} and not (
                        F.is_inf(ins[0], FMT[size]) or F.is_nan(r, FMT[size])):
                    # "allowed to not signal Overflow / Underflow", Inexact
                    # "if there is a chance the approximated result may
                    # differ" (FRCP / FRSQRT): Inexact alone unless the
                    # operand is infinite or the result a NaN, as QEMU
                    flags = {"I"}
                out.append(fp.element(r, flags, size, tiny=tiny and not reciprocal
                                      and _tiny(r, size)))
            return out
        _fp_run(m, w, size, size, compute)
    return run


def _scaleb(fp: _Fp, xs: list[int], size: int) -> tuple[int, set[str]]:
    """FEXP2: ws x 2^wt (IEEE scaleB; wt a signed integer element)."""
    a, n = xs[0], sx(xs[1], size)
    f = FMT[size]
    nan = F._nan_result(f, a, nan2008=True)
    if nan:
        return nan
    if F.is_inf(a, f) or F.value(a, f) == 0:
        return a, set()
    from fractions import Fraction
    lim = 2 * f.bias + f.p + 2              # beyond: overflow / underflow either way
    v = F.value(a, f) * Fraction(2) ** max(-lim, min(lim, n))
    return F.round_pack(v, f, fp.rm, flush=fp.flush)


def _logb(fp: _Fp, xs: list[int], size: int) -> tuple[int, set[str]]:
    """FLOG2: IEEE logB — the integral exponent of the value, as a float
    (exact); logB(±0) = -inf with Divide by Zero, logB(±inf) = +inf."""
    a, f = xs[0], FMT[size]
    nan = F._nan_result(f, a, nan2008=True)
    if nan:
        return nan
    if F.is_inf(a, f):
        return f.inf, set()
    v = F.value(a, f)
    if v == 0:
        return f.sign_bit | f.inf, {"Z"}
    from fractions import Fraction
    return F.round_pack(Fraction(F._floor_log2(abs(v))), f, fp.rm)


_ONE = {32: 0x3F800000, 64: 0x3FF0000000000000}

_FP_ARITH: dict[str, tuple] = {
    # name: (op(fp, xs, size), nargs, reads wd)
    "FADD": (lambda fp, x, n: F.add(x[0], x[1], FMT[n], fp.rm, fp.flush, nan2008=True), 2, False),
    "FSUB": (lambda fp, x, n: F.add(x[0], x[1], FMT[n], fp.rm, fp.flush, subtract=True,
                                    nan2008=True), 2, False),
    "FMUL": (lambda fp, x, n: F.mul(x[0], x[1], FMT[n], fp.rm, fp.flush, nan2008=True), 2, False),
    "FDIV": (lambda fp, x, n: F.div(x[0], x[1], FMT[n], fp.rm, fp.flush, nan2008=True), 2, False),
    "FSQRT": (lambda fp, x, n: F.sqrt(x[0], FMT[n], fp.rm, fp.flush, nan2008=True), 1, False),
    # wd + / - ws x wt, fused (the operands as (wd, ws, wt); a NaN result
    # takes the first NaN "from left to right as described by the
    # instruction format", 3.5.3: wd, ws, wt)
    "FMADD": (lambda fp, x, n: _fma(fp, x, n, False), 2, True),
    "FMSUB": (lambda fp, x, n: _fma(fp, x, n, True), 2, True),
    "FEXP2": (_scaleb, 2, False),
    "FLOG2": (_logb, 1, False),
    # FRCP / FRSQRT: exact (decision: as QEMU — 1 / x, 1 / sqrt(x) rounded
    # at each step; the manual allows 1 / 2 ULP)
    "FRCP": (lambda fp, x, n: F.div(_ONE[n], x[0], FMT[n], fp.rm, fp.flush, nan2008=True),
             1, False),
    "FRSQRT": (lambda fp, x, n: _rsqrt(fp, x[0], n), 1, False),
    "FRINT": (lambda fp, x, n: F.rint(x[0], FMT[n], fp.rm, nan2008=True), 1, False),
    "FMIN": (lambda fp, x, n: F.minmax(x[0], x[1], FMT[n], "MIN"), 2, False),
    "FMAX": (lambda fp, x, n: F.minmax(x[0], x[1], FMT[n], "MAX"), 2, False),
    "FMIN_A": (lambda fp, x, n: F.minmax(x[0], x[1], FMT[n], "MINA"), 2, False),
    "FMAX_A": (lambda fp, x, n: F.minmax(x[0], x[1], FMT[n], "MAXA"), 2, False),
}


def _fma(fp: _Fp, x: list[int], n: int, sub: bool) -> tuple[int, set[str]]:
    f = FMT[n]
    for a, b in ((x[1], x[2]), (x[2], x[1])):   # "The multiplication between an
        if F.is_inf(a, f) and not F.is_nan(b, f) and F.value(b, f) == 0:   # infinity
            return F.default_nan(f, True), {"V"}    # and a zero signals Invalid"
    nan = F._nan_result(f, x[0], x[1], x[2], nan2008=True)
    if nan:
        return nan
    return F.fma(x[1], x[2], x[0], FMT[n], fp.rm, fp.flush, subtract=sub, nan2008=True)


def _rsqrt(fp: _Fp, a: int, n: int) -> tuple[int, set[str]]:
    r, f1 = F.sqrt(a, FMT[n], fp.rm, fp.flush, nan2008=True)
    q, f2 = F.div(_ONE[n], r, FMT[n], fp.rm, fp.flush, nan2008=True)
    return q, f1 | f2


#: compares: (unordered, equal, less, signaling) of each name (MD00866
#: FCAF ... FSUNE): the result is all ones when true
_CMP = {"AF": (0, 0, 0), "UN": (1, 0, 0), "EQ": (0, 1, 0), "UEQ": (1, 1, 0),
        "LT": (0, 0, 1), "ULT": (1, 0, 1), "LE": (0, 1, 1), "ULE": (1, 1, 1),
        "OR": "OR", "UNE": "UNE", "NE": "NE"}


def _compare(kind, signaling: bool, size: int) -> Callable:
    f = FMT[size]

    def run(m, w: int) -> None:
        a, b = lanes(vec(m, _ws(w)), size), lanes(vec(m, _wt(w)), size)

        def compute(fp: _Fp) -> list[int]:
            out = []
            for x, y in zip(a, b):
                x, _ = fp.input(x, size, compare=True)
                y, _ = fp.input(y, size, compare=True)
                unord = F.is_nan(x, f) or F.is_nan(y, f)
                flags = {"V"} if unord and (signaling or F.is_snan(x, f, True)
                                            or F.is_snan(y, f, True)) else set()
                eq = lt = False
                if not unord:
                    vx, vy = F._ordered(x, f), F._ordered(y, f)
                    eq, lt = vx == vy, vx < vy
                if kind == "OR":
                    res = not unord
                elif kind == "UNE":
                    res = unord or not eq
                elif kind == "NE":
                    res = not unord and not eq
                else:
                    u, e, l_ = kind
                    res = bool(u and unord or e and eq or l_ and lt)
                out.append(fp.element(-1 if res else 0, flags, size))
            return out
        _fp_run(m, w, size, size, compute)
    return run


def _class(size: int) -> Callable:
    """FCLASS: the 10-bit class mask (never flushed, no exception)."""
    def run(m, w: int) -> None:
        set_vec(m, _wd(w), pack([F.classify(x, FMT[size]) for x in lanes(vec(m, _ws(w)), size)],
                                size))
    return run


def _to_int(size: int, signed: bool, trunc: bool) -> Callable:
    """FTINT_S / FTINT_U (RM), FTRUNC_S / FTRUNC_U (toward zero): NaN ->
    0, out of range -> the nearest bound, both Invalid; inexact -> I."""
    f = FMT[size]
    lo, hi = (-(1 << (size - 1)), (1 << (size - 1)) - 1) if signed else (0, (1 << size) - 1)

    def run(m, w: int) -> None:
        a = lanes(vec(m, _ws(w)), size)

        def compute(fp: _Fp) -> list[int]:
            out = []
            for x in a:
                x, flags = fp.input(x, size)
                if F.is_nan(x, f):
                    r, flags = 0, flags | {"V"}
                elif F.is_inf(x, f):
                    r, flags = (lo if F.sign(x, f) else hi), flags | {"V"}
                else:
                    v = F.value(x, f)
                    neg = v < 0
                    q = F._round_int(-v if neg else v, neg, F.RZ if trunc else fp.rm)
                    q = -q if neg else q
                    if not lo <= q <= hi:
                        r, flags = (lo if q < lo else hi), flags | {"V"}
                    else:
                        r = q
                        if q != v:
                            flags = flags | {"I"}
                out.append(fp.element(r, flags, size))
            return out
        _fp_run(m, w, size, size, compute)
    return run


def _from_int(size: int, signed: bool) -> Callable:
    """FFINT_S / FFINT_U: the integer, rounded per RM."""
    def run(m, w: int) -> None:
        a = lanes(vec(m, _ws(w)), size)

        def compute(fp: _Fp) -> list[int]:
            from fractions import Fraction
            out = []
            for x in a:
                v = sx(x, size) if signed else x
                r, flags = F.round_pack(Fraction(v), FMT[size], fp.rm)
                out.append(fp.element(r, flags, size))
            return out
        _fp_run(m, w, size, size, compute)
    return run


def _fexdo(size: int) -> Callable:
    """FEXDO.H / .W: ws's elements (2 x size) down-converted to the left
    half, wt's to the right half (16-bit results never flushed)."""
    src = 2 * size

    def run(m, w: int) -> None:
        a, b = lanes(vec(m, _ws(w)), src), lanes(vec(m, _wt(w)), src)

        def compute(fp: _Fp) -> list[int]:
            out = []
            for x in b + a:                         # right half = wt
                x, flags = fp.input(x, src)
                r, fl = F.convert(x, FMT[src], FMT[size], fp.rm,
                                  fp.flush if size != 16 else False, nan2008=True)
                flags |= fl
                out.append(fp.element(r, flags, size, tiny=_tiny(r, size)))
            return out
        _fp_run(m, w, src, size, compute)
    return run


def _fexup(size: int, left: bool) -> Callable:
    """FEXUPL / FEXUPR.W / .D: the left / right half of ws up-converted
    (exact; 16-bit inputs never flushed)."""
    src = size // 2

    def run(m, w: int) -> None:
        a = lanes(vec(m, _ws(w)), src)
        half = a[len(a) // 2:] if left else a[:len(a) // 2]

        def compute(fp: _Fp) -> list[int]:
            out = []
            for x in half:
                x, flags = fp.input(x, src)
                r, fl = F.convert(x, FMT[src], FMT[size], fp.rm, fp.flush, nan2008=True)
                out.append(fp.element(r, flags | fl, size))
            return out
        _fp_run(m, w, src, size, compute)
    return run


def _ffq(size: int, left: bool) -> Callable:
    """FFQL / FFQR.W / .D: Q15 / Q31 -> float (exact: no exception)."""
    src = size // 2

    def run(m, w: int) -> None:
        from fractions import Fraction
        a = lanes(vec(m, _ws(w)), src)
        half = a[len(a) // 2:] if left else a[:len(a) // 2]
        out = [F.round_pack(Fraction(sx(x, src), 1 << (src - 1)), FMT[size], F.RN)[0]
               for x in half]
        set_vec(m, _wd(w), pack(out, size))
    return run


def _ftq(size: int) -> Callable:
    """FTQ.H / .W: ws's floats (2 x size) to Q15 / Q31 in the left half,
    wt's in the right: scaled by 2^(size-1), rounded per RM; outside the
    range -> the bound with Overflow + Inexact, NaN -> 0 with Invalid."""
    src = 2 * size
    f = FMT[src]
    lo, hi = -(1 << (size - 1)), (1 << (size - 1)) - 1

    def run(m, w: int) -> None:
        a, b = lanes(vec(m, _ws(w)), src), lanes(vec(m, _wt(w)), src)

        def compute(fp: _Fp) -> list[int]:
            out = []
            for x in b + a:
                x, flags = fp.input(x, src)
                if F.is_nan(x, f):
                    r, flags = 0, flags | {"V"}
                elif F.is_inf(x, f):
                    r, flags = (lo if F.sign(x, f) else hi), flags | {"O", "I"}
                else:
                    v = F.value(x, f) * (1 << (size - 1))
                    neg = v < 0
                    q = F._round_int(-v if neg else v, neg, fp.rm)
                    q = -q if neg else q
                    if not lo <= q <= hi:
                        r, flags = (lo if q < lo else hi), flags | {"O", "I"}
                    else:
                        r = q
                        if q != v:
                            flags = flags | {"I"}
                out.append(fp.element(r, flags, size))
            return out
        _fp_run(m, w, src, size, compute)
    return run


def _float_executors() -> dict[str, Callable]:
    ex: dict[str, Callable] = {}
    for df, size in (("W", 32), ("D", 64)):
        for name, (op, nargs, reads_wd) in _FP_ARITH.items():
            ex[f"{name}.{df}"] = _arith(op, size, nargs=nargs, reads_wd=reads_wd,
                                        tiny=not name.startswith(("FMIN", "FMAX")),
                                        ints=1 if name == "FEXP2" else 0,
                                        reciprocal=name in ("FRCP", "FRSQRT"))
        for cond, kind in _CMP.items():
            ex[f"FC{cond}.{df}"] = _compare(kind, False, size)
            ex[f"FS{cond}.{df}"] = _compare(kind, True, size)
        ex[f"FCLASS.{df}"] = _class(size)
        ex[f"FTINT_S.{df}"] = _to_int(size, True, False)
        ex[f"FTINT_U.{df}"] = _to_int(size, False, False)
        ex[f"FTRUNC_S.{df}"] = _to_int(size, True, True)
        ex[f"FTRUNC_U.{df}"] = _to_int(size, False, True)
        ex[f"FFINT_S.{df}"] = _from_int(size, True)
        ex[f"FFINT_U.{df}"] = _from_int(size, False)
        ex[f"FEXUPL.{df}"] = _fexup(size, True)
        ex[f"FEXUPR.{df}"] = _fexup(size, False)
        ex[f"FFQL.{df}"] = _ffq(size, True)
        ex[f"FFQR.{df}"] = _ffq(size, False)
    for df, size in (("H", 16), ("W", 32)):
        ex[f"FEXDO.{df}"] = _fexdo(size)
        ex[f"FTQ.{df}"] = _ftq(size)
    return ex


# ------------------------------------------------------------ vector register partitioning (phase 2f)

#: template letters naming vector registers: MSA's wt / ws / wd, the FPU's
#: fs / ft / fd / fr (corrections.json letters)
_VREG_LETTERS = {"V": 16, "W": 11, "Y": 6, "S": 11, "T": 16, "D": 6, "R": 21}


#: the rows (by mnemonic without the format) whose result depends on wd's
#: value before (MD00866: "the destination ... is also a source"): the
#: insert / bit-select / slide / shuffle forms and every accumulation;
#: tests/test_msa.py re-derives it from the executors (the 3b hazards,
#: the trace's lanes)
READS_WD = frozenset((
    "BINSL", "BINSLI", "BINSR", "BINSRI", "BMNZ", "BMNZI", "BMZ", "BMZI", "BSEL", "BSELI",
    "DPADD_S", "DPADD_U", "DPSUB_S", "DPSUB_U", "FMADD", "FMSUB", "INSERT", "INSVE",
    "MADDR_Q", "MADDV", "MADD_Q", "MSUBR_Q", "MSUBV", "MSUB_Q", "SLD", "SLDI", "VSHF"))


def registers_of(entry: dict, word: int, m=None) -> tuple[int, int]:
    """(accessed, written) vector registers of an MSA or FPU instruction as
    32-bit masks (MSARequest / MSAModify, 3.4.5 / 3.4.6): its register
    fields; written = its destination (MSA: wd unless a store / branch /
    CTCMSA; the FPU: fd, a load's ft, MTC1's fs)."""
    if "encode" not in entry:                   # a microMIPS MSA branch (MM.BZ.V ...)
        return 1 << m._mm.f["V"], 0
    enc, name = entry["encode"], entry["name"]
    accessed = written = 0
    for ch, lsb in _VREG_LETTERS.items():
        if ch in enc:
            accessed |= 1 << ((word >> lsb) & 31)
    if entry.get("ase") == "msa":
        if "Y" in enc and not name.startswith("ST.") and entry["family"] != "msa_branch":
            written = 1 << ((word >> 6) & 31)
    elif "D" in enc and not name.startswith(("C.", "BC1")):
        written = 1 << ((word >> 6) & 31)
    elif name in ("LWC1", "LDC1", "LWC1@R6", "LDC1@R6"):
        written = 1 << ((word >> 16) & 31)
    elif name in ("MTC1", "MTHC1", "DMTC1"):
        written = 1 << ((word >> 11) & 31)
    return accessed, written


class Partitions:
    """MSAIR.WRP = 1 (3.4.3-3.4.8, Appendix A): ``phys`` physical vector
    registers shared by the architectural $w0 ... $w31 of this thread
    context; MSAMap / MSAUnmap map and free them (a physical register keeps
    its value while unmapped: mapping it again shows that value), MSAAccess
    says which are mapped, MSASave which still hold another software
    context's value, MSARequest what the last refused instruction needed,
    MSAModify what instructions wrote (ORed by hardware)."""

    def __init__(self, phys: int = 32) -> None:
        if not 1 <= phys <= 32:
            raise ValueError(f"physical vector registers 1..32, not {phys}")
        self.store = [0] * phys            # the physical registers' values
        self.owner: dict[int, int] = {}    # architectural n -> physical index
        self.access = self.save = self.modify = self.request = 0
        self.map = self.unmap = 0

    def read(self, m, cs: int) -> int:
        return {2: self.access, 3: self.save, 4: self.modify, 5: self.request,
                6: self.map, 7: self.unmap}[cs]

    def write(self, m, cd: int, v: int) -> None:
        if cd == 3:
            self.save = v
        elif cd == 4:
            self.modify = v
        elif cd == 6:
            self.map = n = v & 31
            if n not in self.owner:
                free = [p for p in range(len(self.store)) if p not in self.owner.values()]
                if free:                    # (none left: MSAAccess does not change)
                    self.owner[n] = free[0]
                    set_vec(m, n, self.store[free[0]])
                    self.access |= 1 << n
        elif cd == 7:
            self.unmap = n = v & 31
            if n in self.owner:
                self.store[self.owner.pop(n)] = vec(m, n)
                self.access &= ~(1 << n)

    def check(self, m, accessed: int) -> None:
        """Before an MSA / FPU instruction: MSARequest = its registers that
        are not mapped or not saved; any -> MSA Disabled (3.5)."""
        self.request = accessed & (~self.access | self.save) & M32
        if self.request:
            from .sim import _Arch
            raise _Arch(C.MSADIS)

    def done(self, written: int) -> None:
        self.modify |= written


# ------------------------------------------------------------ the table

def _executors() -> dict[str, Callable]:
    ex: dict[str, Callable] = {"CFCMSA": _e_cfcmsa, "CTCMSA": _e_ctcmsa, "MOVE.V": _e_move_v}
    for df, size in SIZE.items():
        ex[f"COPY_S.{df}"] = _copy(True, size)
        ex[f"FILL.{df}"] = _fill(size)
        ex[f"INSERT.{df}"] = _insert(size)
        ex[f"INSVE.{df}"] = _insve(size)
        ex[f"SPLAT.{df}"] = _splat(size)
        ex[f"SPLATI.{df}"] = _splati(size)
        ex[f"LDI.{df}"] = _ldi(size)
        ex[f"LD.{df}"] = _ld(size)
        ex[f"ST.{df}"] = _st(size)
    for df, size in (("B", 8), ("H", 16), ("W", 32)):
        ex[f"COPY_U.{df}"] = _copy(False, size)
    for name, cond in CONDITIONS.items():
        ex[name] = _branch(cond)
    ex.update(_integer_executors())
    ex.update(_permute_executors())
    ex.update(_fixed_executors())
    ex.update(_float_executors())
    return ex


#: row name -> executor (into ``sim._EXECUTORS``)
EXECUTORS: dict[str, Callable] = _executors()
