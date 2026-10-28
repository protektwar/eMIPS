"""The RISC-V vector extension (LF6; The RISC-V Instruction Set Manual,
Volume I: Unprivileged Architecture 20250508, chapter 30 "V" — sections
cited are that chapter's).

:class:`VectorUnit` is a hart's vector state (30.3): ``v0``-``v31``
(:attr:`VectorUnit.regs`, VLEN bits each, elements packed from the least
significant byte up and a register group's registers one after another:
30.4 — so element ``i`` of the group at ``vd`` is at byte ``vd * VLENB +
i * EEW / 8``), ``vtype`` / ``vl`` (set only by ``vset{i}vl{i}`` and a
fault-only-first load), ``vstart``, ``vxrm``, ``vxsat`` and the read-only
``vlenb``.  The hart's ``extensions`` choose its parameters (30.2, 30.18):
VLEN = the largest Zvl*b (V brings Zvl128b, Zve32* Zvl32b, Zve64* Zvl64b),
ELEN = 64 with a Zve64*, else 32; floating-point elements need Zve32f
(32-bit) / Zve64d (64-bit) / Zvfh (16-bit).

The executors (:func:`executors`, joined to ``emips.simrv.EXECUTORS``)
follow the chapter; where it leaves a choice (a *reserved* encoding, a
policy) eMIPS takes Spike's (``riscv/v_ext_macros.h``,
``vector_unit.cc``), so Spike can check it:

- a reserved encoding — a misaligned register group, an EMUL out of 1/8-8,
  a forbidden overlap, a masked instruction writing ``v0`` — is an illegal
  instruction (30.5.2: "reserved");
- an unsupported ``vtype`` sets vill (30.6.1.1), and so does ``vsetvli
  x0, x0`` changing VLMAX (30.6.2: reserved, "may set vill"); ``vl`` =
  min(AVL, VLMAX) (30.6.3's rule 2 lets an implementation choose);
- tail- and mask-agnostic elements keep their values (30.3.4.3: "a simple
  in-order implementation can ignore the settings");
- a vector instruction that cannot be interrupted part-way (arithmetic)
  with ``vstart`` ≠ 0 is an illegal instruction (30.3.7 permits it: eMIPS
  never produces such a vstart); loads and stores restart at ``vstart``,
  and a trap on element *i* leaves vstart = *i* with the elements before
  it written (30.17.1, precise traps);
- indexed accesses with 64-bit indices on RV32 are illegal (30.7.2: "an
  implementation must raise an illegal-instruction exception if the EEW is
  not supported for offset elements");
- misaligned elements are done in hardware, as the hart's scalar accesses
  (30.8; ``m25-RISC-V.md`` decision 7);
- every vector instruction that passes its checks makes mstatus.VS Dirty
  (30.3.2: "may also change ... at any time").

Choices: ``sim.DEVIATIONS["RISC-V vector"]``.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import TYPE_CHECKING, Callable

from . import rvcsr as CSR

if TYPE_CHECKING:
    from .riscv import RV
    from .simrv import RVMachine

#: mstatus.VS (priv. 3.1.6.6; 30.3.2)
VS_SHIFT = 9
ILLEGAL = 2


class VectorUnit:
    """One hart's vector registers and CSRs (see the module docstring)."""

    def __init__(self, m: "RVMachine", extensions: frozenset) -> None:
        self.m = m
        e = extensions
        #: ELEN / VLEN (30.2, 30.18.1-3)
        self.elen = 64 if {"zve64x", "zve64f", "zve64d"} & e else 32
        self.vlen = max(int(x[3:-1]) for x in e if re.fullmatch(r"zvl\d+b", x))
        self.vlenb = self.vlen // 8
        #: the floating-point element widths (30.18: Zve32f, Zve64d, Zvfh /
        #: Zvfhmin's conversions)
        self.fp = {w for w, x in ((32, "zve32f"), (64, "zve64d"), (16, "zvfh")) if x in e}
        self.fp16min = "zvfhmin" in e
        #: the whole V (30.18.3: the Zve64* subsets have no 64-bit vmulh* /
        #: vsmul)
        self.full = "v" in e
        self.regs = bytearray(32 * self.vlenb)
        self.vstart = 0
        self.vxrm = 0
        self.vxsat = 0
        self.vl = 0
        self.vtype = 0
        self.vlmax = 0
        self.sew = 8
        self.lmul = Fraction(1)
        self.vta = self.vma = 0
        self.vill = False
        self.set_vl(0, 0, 0, self.m.amask)      # (reset: vill, vl = 0 — Spike, 30.3.11)

    # -- vtype / vl (30.6) --------------------------------------------------------------

    def set_vl(self, rd: int, rs1: int, avl: int, new: int) -> int:
        """vset{i}vl{i}'s work (30.6; Spike's ``set_vl``): ``rs1`` -1 for
        vsetivli; returns the new vl."""
        m = self.m
        new &= m.amask
        if new != self.vtype:
            old_vlmax = self.vlmax
            vlmul = (new & 7) - 8 if new & 4 else new & 7
            self.sew = 8 << (new >> 3 & 7)
            self.lmul = Fraction(1 << vlmul) if vlmul >= 0 else Fraction(1, 1 << -vlmul)
            self.vlmax = int(Fraction(self.vlen // self.sew) * self.lmul)
            self.vta, self.vma = new >> 6 & 1, new >> 7 & 1
            self.vill = (not Fraction(1, 8) <= self.lmul <= 8
                         or self.sew > min(self.lmul, 1) * self.elen
                         or new >> 8 != 0           # (bits XLEN-2:8 reserved, vill itself)
                         or (rd == 0 and rs1 == 0 and old_vlmax != self.vlmax))
            if self.vill:
                self.vlmax = 0
                self.vtype = 1 << (m.xlen - 1)
            else:
                self.vtype = new
        if self.vlmax == 0:
            self.vl = 0
        elif rd == 0 and rs1 == 0:
            pass                                # (keep vl: 30.6.2)
        elif rs1 == 0:
            self.vl = self.vlmax                # (AVL = ~0)
        else:
            self.vl = min(avl, self.vlmax)
        return self.vl

    # -- elements (30.4) ----------------------------------------------------------------

    def get(self, reg: int, i: int, size: int) -> int:
        """Element ``i`` (``size`` bytes, unsigned) of the group at ``reg``."""
        o = reg * self.vlenb + i * size
        return int.from_bytes(self.regs[o:o + size], "little")

    def put(self, reg: int, i: int, size: int, v: int) -> None:
        o = reg * self.vlenb + i * size
        self.regs[o:o + size] = (v & ((1 << 8 * size) - 1)).to_bytes(size, "little")

    def mask(self, i: int, reg: int = 0) -> int:
        """Mask bit ``i`` of ``reg`` (30.4.5: bit i of the register)."""
        return self.regs[reg * self.vlenb + (i >> 3)] >> (i & 7) & 1

    def set_mask(self, i: int, v: int, reg: int) -> None:
        o = reg * self.vlenb + (i >> 3)
        self.regs[o] = self.regs[o] & ~(1 << (i & 7)) | (v & 1) << (i & 7)

    def register(self, n: int) -> int:
        """``v<n>`` as one VLEN-bit integer (the views, GDB)."""
        return int.from_bytes(self.regs[n * self.vlenb:(n + 1) * self.vlenb], "little")

    def set_register(self, n: int, v: int) -> None:
        self.regs[n * self.vlenb:(n + 1) * self.vlenb] = \
            (v & ((1 << self.vlen) - 1)).to_bytes(self.vlenb, "little")

    @property
    def vcsr(self) -> int:
        return self.vxrm << 1 | self.vxsat

    def describe_vtype(self) -> str:
        """vtype as vsetvli's operands (``e32, m1, ta, ma``) or ``vill``."""
        if self.vill:
            return "vill"
        lm = f"m{self.lmul}" if self.lmul >= 1 else f"mf{self.lmul.denominator}"
        return f"e{self.sew}, {lm}, {'ta' if self.vta else 'tu'}, {'ma' if self.vma else 'mu'}"


# --------------------------------------------------------------- the checks


class _Illegal(Exception):
    pass


def _ill(m: "RVMachine", i: "RV"):
    from .simrv import RVTrap
    return RVTrap(ILLEGAL, i.raw)


def vs_on(m: "RVMachine") -> bool:
    return m.csrs.vs_on()                       # (with V = 1 vsstatus.VS too: 21.2.11)


def dirty(m: "RVMachine") -> None:
    m.csrs.vs_dirty()


def require(m: "RVMachine", i: "RV", ok: bool) -> None:
    if not ok:
        raise _ill(m, i)


def require_vector(m: "RVMachine", i: "RV", alu: bool) -> "VectorUnit":
    """Spike's ``require_vector``: VS on, vtype legal (30.3.4.4), vstart 0
    for an instruction that is never interrupted part-way (VS becomes
    Dirty when the instruction runs: :func:`executors`)."""
    vu = m.vector
    require(m, i, vu is not None and vs_on(m) and not vu.vill and not (alu and vu.vstart))
    return vu


def require_novtype(m: "RVMachine", i: "RV") -> "VectorUnit":
    """VS on (vset{i}vl{i}, whole-register moves / loads / stores: they do
    not depend on vtype, 30.3.4.4)."""
    vu = m.vector
    require(m, i, vu is not None and vs_on(m))
    return vu


def aligned(reg: int, emul: Fraction) -> bool:
    """A register group's base is a multiple of EMUL (30.3.4.2; < 1: any)."""
    pos = int(emul)
    return pos == 0 or reg & (pos - 1) == 0


def overlapped(a: int, asize: Fraction, b: int, bsize: Fraction) -> bool:
    """Spike's ``is_overlapped`` (sizes below one register count as one)."""
    asize, bsize = max(int(asize), 1), max(int(bsize), 1)
    return max(a + asize, b + bsize) - min(a, b) < asize + bsize


def overlapped_widen(a: int, asize: Fraction, b: int, bsize: Fraction) -> bool:
    """Spike's ``is_overlapped_widen``: the source may overlap the highest
    part of the destination (30.5.2)."""
    asize, bsize = max(int(asize), 1), max(int(bsize), 1)
    if a < b and overlapped(a, asize, b, bsize) and not overlapped(a, asize, b + bsize, bsize):
        return False
    return max(a + asize, b + bsize) - min(a, b) < asize + bsize


def _vm(i: "RV") -> int:
    """The vm bit (30.5.3.1: inst[25], also where the row fixes it)."""
    return i.raw >> 25 & 1


def require_vm(m: "RVMachine", i: "RV") -> None:
    """A masked instruction cannot write v0 (30.5.3: reserved)."""
    require(m, i, _vm(i) == 1 or i.f.get("vd", i.f.get("vs3", 1)) != 0)


# ------------------------------------------------------- loads and stores (30.7)


def _active(vu: VectorUnit, i: "RV", k: int) -> bool:
    return _vm(i) == 1 or vu.mask(k) == 1


def _xreg(m: "RVMachine", n: int) -> int:
    return m.read_reg(n)


def _check_unit(m: "RVMachine", i: "RV", eew: int | None, nf: int, load: bool) -> Fraction:
    """VI_CHECK_STORE / VI_CHECK_LOAD (30.7.3, 30.7.8): ``eew`` None = a
    mask load / store (EEW 8, EMUL 1); returns EMUL."""
    vu = require_vector(m, i, False)
    if eew is None:
        veew, vemul = 1, Fraction(1)
    else:
        veew, vemul = eew, Fraction(eew, vu.sew) * vu.lmul
    emul = max(int(vemul), 1)
    vd = i.f.get("vd", i.f.get("vs3"))
    require(m, i, Fraction(1, 8) <= vemul <= 8 and aligned(vd, vemul)
            and nf * emul <= 8 and vd + nf * emul <= 32 and veew <= vu.elen)
    if load:
        require_vm(m, i)
    return Fraction(emul)


def _check_index(m: "RVMachine", i: "RV", eew: int, nf: int, load: bool) -> Fraction:
    """VI_CHECK_ST_INDEX / VI_CHECK_LD_INDEX (30.7.3, 30.7.8.3): the index
    group's EMUL and the data group's (SEW, LMUL)."""
    vu = require_vector(m, i, False)
    require(m, i, eew <= min(vu.elen, m.xlen))
    vemul = Fraction(eew, vu.sew) * vu.lmul
    require(m, i, Fraction(1, 8) <= vemul <= 8)
    flmul = max(int(vu.lmul), 1)
    vd = i.f.get("vd", i.f.get("vs3"))
    vs2 = i.f["vs2"]
    require(m, i, aligned(vd, vu.lmul) and aligned(vs2, vemul)
            and nf * flmul <= 8 and vd + nf * flmul <= 32)
    if load:
        for fn in range(nf):
            seg = vd + flmul * fn
            if eew > vu.sew:
                if seg != vs2:
                    require(m, i, not overlapped(seg, vu.lmul, vs2, vemul))
            elif eew < vu.sew:
                if vemul < 1:
                    require(m, i, not overlapped(seg, vu.lmul, vs2, vemul))
                else:
                    require(m, i, not overlapped_widen(seg, vu.lmul, vs2, vemul))
            if nf >= 2:                         # (30.7.8.3: no overlap with the index)
                require(m, i, not overlapped(seg, vu.lmul, vs2, vemul))
        require_vm(m, i)
    return vemul


def _loads(m: "RVMachine", i: "RV", count: int, addr: Callable[[int, int], int],
           size: int, dst: Callable[[int, int], tuple[int, int]], ff: bool = False) -> None:
    """The element loop of a load: elements ``vstart``..``count`` - 1 that
    are active; field ``fn`` of element ``k`` from ``addr(k, fn)`` into
    ``dst(k, fn)`` = (register, element index).  A trap on element k leaves
    vstart = k (30.17.1); fault-only-first trims vl instead (30.7.7)."""
    from .simrv import RVTrap
    vu = m.vector
    nf = i.f.get("nf", 0) + 1
    amask = m.amask
    for k in range(vu.vstart, count):
        if not _active(vu, i, k):
            continue
        vu.vstart = k
        for fn in range(nf):                    # (a segment's fields before a fault stay
            try:                                #  written: implementation-defined, Spike)
                v = m.load(addr(k, fn) & amask, size)
            except RVTrap:
                if ff and k > 0:                # (only element 0 traps: vl = k)
                    vu.vl = k
                    vu.vstart = 0
                    return
                if ff:
                    vu.vstart = 0
                raise
            reg, idx = dst(k, fn)
            vu.put(reg, idx, size, v)
    vu.vstart = 0


def _stores(m: "RVMachine", i: "RV", count: int, addr: Callable[[int, int], int],
            size: int, src: Callable[[int, int], tuple[int, int]]) -> None:
    from .simrv import _clear_reservation
    vu = m.vector
    nf = i.f.get("nf", 0) + 1
    amask = m.amask
    for k in range(vu.vstart, count):
        if not _active(vu, i, k):
            continue
        vu.vstart = k
        for fn in range(nf):
            reg, idx = src(k, fn)
            a = addr(k, fn) & amask
            m.store(a, vu.get(reg, idx, size), size)
            _clear_reservation(m, a, size)
    vu.vstart = 0


def _unit(eew: int | None, store: bool, ff: bool = False):
    """vle<eew>.v / vse<eew>.v / vlseg / vsseg, vle<eew>ff.v, vlm.v / vsm.v
    (30.7.4, 30.7.7, 30.7.8.1)."""
    size = 1 if eew is None else eew // 8

    def ex(m, i, pc):
        nf = i.f.get("nf", 0) + 1
        emul = int(_check_unit(m, i, eew, nf, not store))
        vu = m.vector
        base = _xreg(m, i.f["rs1"])
        count = (vu.vl + 7) // 8 if eew is None else vu.vl
        vd = i.f.get("vd", i.f.get("vs3"))
        addr = lambda k, fn: base + (k * nf + fn) * size
        reg = lambda k, fn: (vd + fn * emul, k)
        if store:
            _stores(m, i, count, addr, size, reg)
        else:
            _loads(m, i, count, addr, size, reg, ff)
    return ex


def _strided(eew: int, store: bool):
    """vlse<eew>.v / vsse<eew>.v and their segment forms (30.7.5, 30.7.8.2)."""
    size = eew // 8

    def ex(m, i, pc):
        nf = i.f.get("nf", 0) + 1
        emul = int(_check_unit(m, i, eew, nf, not store))
        vu = m.vector
        base, stride = _xreg(m, i.f["rs1"]), _xreg(m, i.f["rs2"])
        vd = i.f.get("vd", i.f.get("vs3"))
        addr = lambda k, fn: base + k * stride + fn * size
        reg = lambda k, fn: (vd + fn * emul, k)
        (_stores if store else _loads)(m, i, vu.vl, addr, size, reg)
    return ex


def _indexed(eew: int, store: bool):
    """vl{u,o}xei<eew>.v / vs{u,o}xei<eew>.v and their segment forms
    (30.7.6, 30.7.8.3): the offsets zero-extended, the data SEW wide."""
    isize = eew // 8

    def ex(m, i, pc):
        nf = i.f.get("nf", 0) + 1
        _check_index(m, i, eew, nf, not store)
        vu = m.vector
        size = vu.sew // 8
        flmul = max(int(vu.lmul), 1)
        base = _xreg(m, i.f["rs1"])
        vs2 = i.f["vs2"]
        vd = i.f.get("vd", i.f.get("vs3"))
        addr = lambda k, fn: base + vu.get(vs2, k, isize) + fn * size
        reg = lambda k, fn: (vd + fn * flmul, k)
        (_stores if store else _loads)(m, i, vu.vl, addr, size, reg)
    return ex


def _whole_load(eew: int):
    """vl<nf>re<eew>.v (30.7.9): evl = NFIELDS * VLEN / EEW, no vtype."""
    size = eew // 8

    def ex(m, i, pc):
        vu = require_novtype(m, i)
        nf = (i.raw >> 29) + 1
        vd = i.f["vd"]
        require(m, i, size * 8 <= vu.elen and aligned(vd, Fraction(nf)))
        base = _xreg(m, i.f["rs1"])
        for k in range(vu.vstart, nf * vu.vlenb // size):
            vu.vstart = k
            vu.put(vd, k, size, m.load((base + k * size) & m.amask, size))
        vu.vstart = 0
    return ex


def _whole_store(m, i, pc):
    """vs<nf>r.v (30.7.9): byte by byte (vstart counts bytes, as Spike)."""
    from .simrv import _clear_reservation
    vu = require_novtype(m, i)
    nf = (i.raw >> 29) + 1
    vs3 = i.f["vs3"]
    require(m, i, aligned(vs3, Fraction(nf)))
    base = _xreg(m, i.f["rs1"])
    for k in range(vu.vstart, nf * vu.vlenb):
        vu.vstart = k
        a = (base + k) & m.amask
        m.store(a, vu.get(vs3, k, 1), 1)
        _clear_reservation(m, a, 1)
    vu.vstart = 0


# ----------------------------------------------------- configuration (30.6)


def _vsetvli(m, i, pc):
    vu = require_novtype(m, i)
    rs1 = i.f["rs1"]
    m.write_reg(i.f["rd"], vu.set_vl(i.f["rd"], rs1, m.read_reg(rs1), i.f["vtypei"]))
    vu.vstart = 0


def _vsetivli(m, i, pc):
    vu = require_novtype(m, i)
    m.write_reg(i.f["rd"], vu.set_vl(i.f["rd"], -1, i.f["zimm"], i.f["vtypei"]))
    vu.vstart = 0


def _vsetvl(m, i, pc):
    vu = require_novtype(m, i)
    rs1 = i.f["rs1"]
    m.write_reg(i.f["rd"], vu.set_vl(i.f["rd"], rs1, m.read_reg(rs1), m.read_reg(i.f["rs2"])))
    vu.vstart = 0


# ------------------------------------- integer arithmetic (30.11-30.16)


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _ones(bits: int) -> int:
    return (1 << bits) - 1


def _scalar(m: "RVMachine", n: int, sew: int) -> int:
    """x[n] as a SEW-bit operand (30.5.1: sign-extended when XLEN < SEW,
    else truncated)."""
    return _sx(m.read_reg(n), m.xlen) & _ones(sew)


def _body(vu: VectorUnit, i: "RV") -> range | list[int]:
    """The active elements (30.5.4): vstart..vl - 1 where the mask is set."""
    if _vm(i):
        return range(vu.vstart, vu.vl)
    return [k for k in range(vu.vstart, vu.vl) if vu.mask(k)]


def _src(m: "RVMachine", vu: VectorUnit, i: "RV", sew: int) -> Callable[[int], int]:
    """The second operand of element k: vs1's, x[rs1], simm5 (sign-extended)
    or uimm5 — SEW bits."""
    f = i.f
    if "vs1" in f:
        r, n = f["vs1"], sew // 8
        return lambda k: vu.get(r, k, n)
    if "rs1" in f:
        v = _scalar(m, f["rs1"], sew)
    elif "imm" in f:
        v = f["imm"] & _ones(sew)
    else:
        v = f["zimm"]
    return lambda k: v


def _round(v: int, d: int, xrm: int) -> int:
    """(v >> d) + r with vxrm's increment r (30.3.8, Table 47)."""
    if d <= 0:
        return v
    bit = v >> (d - 1) & 1
    if xrm == 0:                                # rnu
        r = bit
    elif xrm == 1:                              # rne
        r = bit & (1 if v & ((1 << (d - 1)) - 1) or v >> d & 1 else 0)
    elif xrm == 2:                              # rdn
        r = 0
    else:                                       # rod
        r = 1 if not v >> d & 1 and v & ((1 << d) - 1) else 0
    return (v >> d) + r


def _sat(vu: VectorUnit, v: int, lo: int, hi: int) -> int:
    """Clip to lo..hi, setting vxsat when it clips (30.12)."""
    if v < lo:
        vu.vxsat = 1
        return lo
    if v > hi:
        vu.vxsat = 1
        return hi
    return v


def _check_sss(m: "RVMachine", i: "RV", vu: VectorUnit) -> None:
    """VI_CHECK_SSS: no v0 under a mask, the groups aligned to LMUL."""
    require_vm(m, i)
    if vu.lmul > 1:
        f = i.f
        require(m, i, aligned(f["vd"], vu.lmul) and aligned(f.get("vs2", 0), vu.lmul)
                and aligned(f.get("vs1", 0), vu.lmul))


def _check_mss(m: "RVMachine", i: "RV", vu: VectorUnit) -> None:
    """VI_CHECK_MSS: a mask result may overlap a source only in its
    lowest register (30.5.2)."""
    f = i.f
    vd = f["vd"]
    for key in ("vs2", "vs1"):
        if key in f:
            r = f[key]
            if vd != r:
                require(m, i, not overlapped(vd, Fraction(1), r, vu.lmul))
            require(m, i, aligned(r, vu.lmul))


def _signed(w: int, fn: Callable[[int, int], int]) -> Callable[..., int]:
    return lambda a, b, vu, d, w=w: fn(_sx(a, w), _sx(b, w))


def _div(a: int, b: int, w: int, signed: bool, rem: bool) -> int:
    """30.11.11 (Table 52's special cases, as M's)."""
    if signed:
        a, b = _sx(a, w), _sx(b, w)
    if b == 0:
        return a if rem else -1
    if signed and a == -(1 << (w - 1)) and b == -1:
        return 0 if rem else a
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return a - q * b if rem else q


def _sat_ops(name: str) -> Callable[..., int]:
    def op(a, b, vu, d, w):
        if name == "vsaddu":
            return _sat(vu, a + b, 0, _ones(w))
        if name == "vssubu":
            return _sat(vu, a - b, 0, _ones(w))
        lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
        if name == "vsadd":
            return _sat(vu, _sx(a, w) + _sx(b, w), lo, hi)
        return _sat(vu, _sx(a, w) - _sx(b, w), lo, hi)
    return op


def _avg(name: str) -> Callable[..., int]:
    def op(a, b, vu, d, w):
        if name in ("vaadd", "vasub"):
            a, b = _sx(a, w), _sx(b, w)
        return _round(a + b if name.startswith("vaadd") else a - b, 1, vu.vxrm)
    return op


def _vsmul(a, b, vu, d, w):
    a, b = _sx(a, w), _sx(b, w)
    lo = -(1 << (w - 1))
    if a == b == lo:
        vu.vxsat = 1
        return -lo - 1
    return _round(a * b, w - 1, vu.vxrm)


#: the single-width operations: op(a = vs2, b = vs1 / rs1 / imm, vu, d = the
#: old vd, w = SEW) -> the result (any integer: it is truncated to SEW)
SINGLE: dict[str, Callable[..., int]] = {
    "vadd": lambda a, b, vu, d, w: a + b,
    "vsub": lambda a, b, vu, d, w: a - b,
    "vrsub": lambda a, b, vu, d, w: b - a,
    "vand": lambda a, b, vu, d, w: a & b,
    "vor": lambda a, b, vu, d, w: a | b,
    "vxor": lambda a, b, vu, d, w: a ^ b,
    "vsll": lambda a, b, vu, d, w: a << (b & (w - 1)),
    "vsrl": lambda a, b, vu, d, w: a >> (b & (w - 1)),
    "vsra": lambda a, b, vu, d, w: _sx(a, w) >> (b & (w - 1)),
    "vminu": lambda a, b, vu, d, w: min(a, b),
    "vmaxu": lambda a, b, vu, d, w: max(a, b),
    "vmin": lambda a, b, vu, d, w: a if _sx(a, w) <= _sx(b, w) else b,
    "vmax": lambda a, b, vu, d, w: a if _sx(a, w) >= _sx(b, w) else b,
    "vmul": lambda a, b, vu, d, w: a * b,
    "vmulh": lambda a, b, vu, d, w: _sx(a, w) * _sx(b, w) >> w,
    "vmulhu": lambda a, b, vu, d, w: a * b >> w,
    "vmulhsu": lambda a, b, vu, d, w: _sx(a, w) * b >> w,
    "vdivu": lambda a, b, vu, d, w: _div(a, b, w, False, False),
    "vdiv": lambda a, b, vu, d, w: _div(a, b, w, True, False),
    "vremu": lambda a, b, vu, d, w: _div(a, b, w, False, True),
    "vrem": lambda a, b, vu, d, w: _div(a, b, w, True, True),
    "vmacc": lambda a, b, vu, d, w: d + a * b,
    "vnmsac": lambda a, b, vu, d, w: d - a * b,
    "vmadd": lambda a, b, vu, d, w: d * b + a,
    "vnmsub": lambda a, b, vu, d, w: a - d * b,
    "vsaddu": _sat_ops("vsaddu"), "vsadd": _sat_ops("vsadd"),
    "vssubu": _sat_ops("vssubu"), "vssub": _sat_ops("vssub"),
    "vaaddu": _avg("vaaddu"), "vaadd": _avg("vaadd"),
    "vasubu": _avg("vasubu"), "vasub": _avg("vasub"),
    "vsmul": _vsmul,
    "vssrl": lambda a, b, vu, d, w: _round(a, b & (w - 1), vu.vxrm),
    "vssra": lambda a, b, vu, d, w: _round(_sx(a, w), b & (w - 1), vu.vxrm),
}
#: no 64-bit form on Zve64* (30.18.3)
_NOT_ZVE64 = frozenset({"vmulh", "vmulhu", "vmulhsu", "vsmul"})


def _single(op: Callable[..., int], name: str):
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _check_sss(m, i, vu)
        w = vu.sew
        require(m, i, vu.full or w < 64 or name not in _NOT_ZVE64)
        n, mk = w // 8, _ones(w)
        f = i.f
        vd, vs2 = f["vd"], f["vs2"]
        b = _src(m, vu, i, w)
        for k in _body(vu, i):
            vu.put(vd, k, n, op(vu.get(vs2, k, n), b(k), vu, vu.get(vd, k, n), w) & mk)
    return ex


#: the integer compares (30.11.8): (a = vs2, b) -> bool
COMPARE: dict[str, Callable[[int, int, int], bool]] = {
    "vmseq": lambda a, b, w: a == b,
    "vmsne": lambda a, b, w: a != b,
    "vmsltu": lambda a, b, w: a < b,
    "vmslt": lambda a, b, w: _sx(a, w) < _sx(b, w),
    "vmsleu": lambda a, b, w: a <= b,
    "vmsle": lambda a, b, w: _sx(a, w) <= _sx(b, w),
    "vmsgtu": lambda a, b, w: a > b,
    "vmsgt": lambda a, b, w: _sx(a, w) > _sx(b, w),
}


def _compare(cmp: Callable[[int, int, int], bool]):
    def ex(m, i, pc):
        _check_mss(m, i, require_vector(m, i, True))
        vu = m.vector
        w = vu.sew
        n = w // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        b = _src(m, vu, i, w)
        res = [(k, cmp(vu.get(vs2, k, n), b(k), w)) for k in _body(vu, i)]
        for k, r in res:                        # (vd may be a source: read first)
            vu.set_mask(k, int(r), vd)
    return ex


def _carry(name: str):
    """vadc / vsbc (the carry / borrow from v0) and vmadc / vmsbc (the
    carry / borrow out, a mask; with or without a carry in) — 30.11.4."""
    out = name.startswith("vm")
    sub = "sbc" in name

    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        (_check_mss if out else _check_sss)(m, i, vu)
        w = vu.sew
        n, mk = w // 8, _ones(w)
        vd, vs2 = i.f["vd"], i.f["vs2"]
        cin = not out or _vm(i) == 0
        b = _src(m, vu, i, w)
        res = []
        for k in range(vu.vstart, vu.vl):
            a, c = vu.get(vs2, k, n), vu.mask(k) if cin else 0
            v = a - b(k) - c if sub else a + b(k) + c
            res.append((k, (1 if v < 0 or v > mk else 0) if out else v & mk))
        for k, v in res:
            if out:
                vu.set_mask(k, v, vd)
            else:
                vu.put(vd, k, n, v)
    return ex


def _merge(m, i, pc):
    """vmerge.v[vxi]m (the mask picks vs1 / rs1 / imm or vs2) and
    vmv.v.[vxi] (vm = 1: the operand) — 30.11.15-16, every body element."""
    vu = require_vector(m, i, True)
    _check_sss(m, i, vu)
    w = vu.sew
    n = w // 8
    f = i.f
    vd, vs2, sel = f["vd"], f.get("vs2", 0), _vm(i) == 0
    b = _src(m, vu, i, w)
    for k in range(vu.vstart, vu.vl):
        vu.put(vd, k, n, b(k) if not sel or vu.mask(k) else vu.get(vs2, k, n))


def _wide_check(m: "RVMachine", i: "RV", vu: VectorUnit, wide_src: bool) -> None:
    """VI_CHECK_DSS / VI_CHECK_DDS (30.5.2's overlap rules for a 2 * SEW
    destination)."""
    require(m, i, vu.lmul <= 4 and vu.sew * 2 <= vu.elen and aligned(i.f["vd"], 2 * vu.lmul))
    require_vm(m, i)
    vd = i.f["vd"]
    for key in ("vs2", "vs1"):
        if key not in i.f:
            continue
        r = i.f[key]
        if key == "vs2" and wide_src:
            require(m, i, aligned(r, 2 * vu.lmul))
            continue
        require(m, i, aligned(r, vu.lmul))
        if vu.lmul < 1:
            require(m, i, not overlapped(vd, 2 * vu.lmul, r, vu.lmul))
        else:
            require(m, i, not overlapped_widen(vd, 2 * vu.lmul, r, vu.lmul))


#: widening operations (30.11.2 / 30.11.12 / 30.11.13): (vs2 signed, operand
#: signed, op(a, b, d)) — .w forms read vs2 at 2 * SEW
WIDEN: dict[str, tuple[bool, bool, Callable[[int, int, int], int]]] = {
    "vwaddu": (False, False, lambda a, b, d: a + b),
    "vwadd": (True, True, lambda a, b, d: a + b),
    "vwsubu": (False, False, lambda a, b, d: a - b),
    "vwsub": (True, True, lambda a, b, d: a - b),
    "vwmulu": (False, False, lambda a, b, d: a * b),
    "vwmul": (True, True, lambda a, b, d: a * b),
    "vwmulsu": (True, False, lambda a, b, d: a * b),
    "vwmaccu": (False, False, lambda a, b, d: d + a * b),
    "vwmacc": (True, True, lambda a, b, d: d + a * b),
    "vwmaccsu": (False, True, lambda a, b, d: d + a * b),
    "vwmaccus": (True, False, lambda a, b, d: d + a * b),
}


def _widen(name: str, wide_src: bool):
    sa, sb, op = WIDEN[name]

    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _wide_check(m, i, vu, wide_src)
        w = vu.sew
        n = w // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        b = _src(m, vu, i, w)
        res = []
        for k in _body(vu, i):
            if wide_src:
                a = vu.get(vs2, k, 2 * n)
                a = _sx(a, 2 * w) if sa else a
            else:
                a = vu.get(vs2, k, n)
                a = _sx(a, w) if sa else a
            bk = _sx(b(k), w) if sb else b(k)
            res.append((k, op(a, bk, vu.get(vd, k, 2 * n))))
        for k, v in res:
            vu.put(vd, k, 2 * n, v)
    return ex


def _ext(div: int, signed: bool):
    """vzext.vf2 / 4 / 8, vsext.vf2 / 4 / 8 (30.11.3)."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        frm = vu.sew // div
        f = i.f
        vd, vs2 = f["vd"], f["vs2"]
        emul = vu.lmul / div
        require(m, i, frm >= 8 and vd != vs2 and Fraction(1, 8) <= emul <= 8
                and aligned(vd, vu.lmul) and aligned(vs2, emul))
        require_vm(m, i)
        require(m, i, not (overlapped(vd, vu.lmul, vs2, emul) if emul < 1
                           else overlapped_widen(vd, vu.lmul, vs2, emul)))
        n, fn = vu.sew // 8, frm // 8
        for k in _body(vu, i):
            v = vu.get(vs2, k, fn)
            vu.put(vd, k, n, _sx(v, frm) if signed else v)
    return ex


def _narrow_check(m: "RVMachine", i: "RV", vu: VectorUnit) -> None:
    """VI_CHECK_SDS: a SEW destination from a 2 * SEW vs2 (30.5.2)."""
    f = i.f
    vd, vs2 = f["vd"], f["vs2"]
    require(m, i, vu.lmul <= 4 and vu.sew * 2 <= vu.elen and aligned(vs2, 2 * vu.lmul)
            and aligned(vd, vu.lmul))
    require_vm(m, i)
    if vd != vs2:
        require(m, i, not overlapped(vd, vu.lmul, vs2, 2 * vu.lmul))
    if "vs1" in f:
        require(m, i, aligned(f["vs1"], vu.lmul))


def _narrow(name: str):
    """vnsrl / vnsra (30.11.7), vnclipu / vnclip (30.12.5)."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _narrow_check(m, i, vu)
        w = vu.sew
        n = w // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        b = _src(m, vu, i, w)
        res = []
        for k in _body(vu, i):
            a = vu.get(vs2, k, 2 * n)
            sh = b(k) & (2 * w - 1)
            if name == "vnsrl":
                v = a >> sh
            elif name == "vnsra":
                v = _sx(a, 2 * w) >> sh
            elif name == "vnclipu":
                v = _sat(vu, _round(a, sh, vu.vxrm), 0, _ones(w))
            else:
                v = _sat(vu, _round(_sx(a, 2 * w), sh, vu.vxrm), -(1 << (w - 1)),
                         (1 << (w - 1)) - 1)
            res.append((k, v))
        for k, v in res:
            vu.put(vd, k, n, v)
    return ex


#: the integer reductions (30.14.1): (signed, op)
REDUCE: dict[str, tuple[bool, Callable[[int, int], int]]] = {
    "vredsum": (False, lambda a, b: a + b),
    "vredand": (False, lambda a, b: a & b),
    "vredor": (False, lambda a, b: a | b),
    "vredxor": (False, lambda a, b: a ^ b),
    "vredminu": (False, min), "vredmaxu": (False, max),
    "vredmin": (True, min), "vredmax": (True, max),
}


def _reduce(name: str, wide: bool):
    """vred* (30.14.1) and vwredsum[u] (30.14.2): vd[0] = op(vs1[0], the
    active vs2 elements), in element order (unordered is allowed in any)."""
    if wide:
        signed, op = name == "vwredsum", lambda a, b: a + b
    else:
        signed, op = REDUCE[name]

    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        require(m, i, (not wide or vu.sew * 2 <= vu.elen) and aligned(i.f["vs2"], vu.lmul))
        w = vu.sew
        n = w // 8
        dw = 2 * w if wide else w
        f = i.f
        acc = vu.get(f["vs1"], 0, dw // 8)
        acc = _sx(acc, dw) if signed else acc
        for k in _body(vu, i):
            v = vu.get(f["vs2"], k, n)
            acc = op(acc, _sx(v, w) if signed else v)
        if vu.vl > 0:
            vu.put(f["vd"], 0, dw // 8, acc)
    return ex


#: the mask-register logical instructions (30.15.1)
MASK_LOGIC: dict[str, Callable[[int, int], int]] = {
    "vmand": lambda a, b: a & b, "vmnand": lambda a, b: 1 - (a & b),
    "vmandn": lambda a, b: a & (1 - b), "vmxor": lambda a, b: a ^ b,
    "vmor": lambda a, b: a | b, "vmnor": lambda a, b: 1 - (a | b),
    "vmorn": lambda a, b: a | (1 - b), "vmxnor": lambda a, b: 1 - (a ^ b),
}


def _mask_logic(op: Callable[[int, int], int]):
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        f = i.f
        res = [(k, op(vu.mask(k, f["vs2"]), vu.mask(k, f["vs1"])))
               for k in range(vu.vstart, vu.vl)]
        for k, v in res:
            vu.set_mask(k, v, f["vd"])
    return ex


def _vcpop(m, i, pc):
    vu = require_vector(m, i, True)
    vs2 = i.f["vs2"]
    m.write_reg(i.f["rd"], sum(vu.mask(k, vs2) for k in _body(vu, i)))


def _vfirst(m, i, pc):
    vu = require_vector(m, i, True)
    vs2 = i.f["vs2"]
    m.write_reg(i.f["rd"], next((k for k in _body(vu, i) if vu.mask(k, vs2)), -1))


def _set_first(kind: str):
    """vmsbf.m / vmsif.m / vmsof.m (30.15.5-7)."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        require_vm(m, i)
        f = i.f
        require(m, i, f["vd"] != f["vs2"])
        seen = False
        for k in _body(vu, i):
            bit = vu.mask(k, f["vs2"])
            if kind == "bf":
                r = not seen and not bit
            elif kind == "if":
                r = not seen
            else:
                r = not seen and bit
            seen = seen or bool(bit)
            vu.set_mask(k, int(r), f["vd"])
    return ex


def _viota(m, i, pc):
    """viota.m (30.15.8): each active element = the set bits of vs2 below it
    among the active elements."""
    vu = require_vector(m, i, True)
    require_vm(m, i)
    f = i.f
    vd, vs2 = f["vd"], f["vs2"]
    require(m, i, aligned(vd, vu.lmul) and not overlapped(vd, vu.lmul, vs2, Fraction(1)))
    n = vu.sew // 8
    count = 0
    for k in _body(vu, i):
        bit = vu.mask(k, vs2)
        vu.put(vd, k, n, count)
        count += bit


def _vid(m, i, pc):
    vu = require_vector(m, i, True)
    require(m, i, aligned(i.f["vd"], vu.lmul))
    require_vm(m, i)
    n = vu.sew // 8
    for k in _body(vu, i):
        vu.put(i.f["vd"], k, n, k)


def _vmv_x_s(m, i, pc):
    """vmv.x.s (30.16.1): element 0 sign-extended (or truncated) to XLEN,
    even when vl = 0 or vstart >= vl (30.5.4)."""
    vu = require_vector(m, i, True)
    w = vu.sew
    m.write_reg(i.f["rd"], _sx(vu.get(i.f["vs2"], 0, w // 8), w))


def _vmv_s_x(m, i, pc):
    vu = require_vector(m, i, True)
    if vu.vl > 0 and vu.vstart < vu.vl:
        w = vu.sew
        vu.put(i.f["vd"], 0, w // 8, _scalar(m, i.f["rs1"], w))


def _slide(kind: str):
    """vslideup / vslidedown (30.16.3), vslide1up / vslide1down (30.16.3);
    a source element past VLMAX reads 0."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        f = i.f
        vd, vs2 = f["vd"], f["vs2"]
        require(m, i, aligned(vs2, vu.lmul) and aligned(vd, vu.lmul))
        require_vm(m, i)
        if kind in ("up", "1up"):
            require(m, i, vd != vs2)
        w = vu.sew
        n = w // 8
        if kind in ("up", "down"):
            off = m.read_reg(f["rs1"]) if "rs1" in f else f["zimm"]
        else:
            x = _scalar(m, f["rs1"], w)
        for k in _body(vu, i):
            if kind == "up":
                if k >= off:
                    vu.put(vd, k, n, vu.get(vs2, k - off, n))
            elif kind == "down":
                vu.put(vd, k, n, vu.get(vs2, k + off, n) if k + off < vu.vlmax else 0)
            elif kind == "1up":
                vu.put(vd, k, n, vu.get(vs2, k - 1, n) if k else x)
            else:
                vu.put(vd, k, n, vu.get(vs2, k + 1, n) if k != vu.vl - 1 else x)
    return ex


def _gather(kind: str):
    """vrgather.vv / .vx / .vi, vrgatherei16.vv (30.16.4): an index past
    VLMAX reads 0."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        f = i.f
        vd, vs2 = f["vd"], f["vs2"]
        require(m, i, aligned(vd, vu.lmul) and aligned(vs2, vu.lmul) and vd != vs2)
        w = vu.sew
        n = w // 8
        if kind == "vv":
            require(m, i, aligned(f["vs1"], vu.lmul) and vd != f["vs1"])
            index = lambda k: vu.get(f["vs1"], k, n)
        elif kind == "ei16":
            emul = Fraction(16, w) * vu.lmul
            require(m, i, Fraction(1, 8) <= emul <= 8 and aligned(f["vs1"], emul)
                    and not overlapped(vd, vu.lmul, f["vs1"], emul))
            index = lambda k: vu.get(f["vs1"], k, 2)
        else:
            x = m.read_reg(f["rs1"]) if kind == "vx" else f["zimm"]
            index = lambda k: x
        require_vm(m, i)
        res = []
        for k in _body(vu, i):
            j = index(k)
            res.append((k, vu.get(vs2, j, n) if j < vu.vlmax else 0))
        for k, v in res:
            vu.put(vd, k, n, v)
    return ex


def _vcompress(m, i, pc):
    """vcompress.vm (30.16.5): vs2's elements whose vs1 mask bit is set,
    packed from element 0."""
    vu = require_vector(m, i, True)
    f = i.f
    vd, vs2, vs1 = f["vd"], f["vs2"], f["vs1"]
    require(m, i, aligned(vd, vu.lmul) and aligned(vs2, vu.lmul) and vd != vs2
            and not overlapped(vd, vu.lmul, vs1, Fraction(1)))
    n = vu.sew // 8
    pos = 0
    for k in range(vu.vstart, vu.vl):
        if vu.mask(k, vs1):
            vu.put(vd, pos, n, vu.get(vs2, k, n))
            pos += 1


def _vmv_nr(nr: int):
    """vmv<nr>r.v (30.16.6): whole registers."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        vd, vs2 = i.f["vd"], i.f["vs2"]
        require(m, i, aligned(vd, Fraction(nr)) and aligned(vs2, Fraction(nr)))
        if vd != vs2:
            b = vu.vlenb
            vu.regs[vd * b:(vd + nr) * b] = vu.regs[vs2 * b:(vs2 + nr) * b]
    return ex


# ------------------------------------------------ floating point (30.13-30.14)

from . import fpu as F                          # noqa: E402
from . import rvfpu as RF                       # noqa: E402

#: SEW -> the element format (16 = Zvfh's binary16)
FMT = {16: F.H, 32: F.S, 64: F.D}
_FRM = {0: F.RN, 1: F.RZ, 2: F.RM, 3: F.RP, 4: F.RMM}


def _require_fp(m: "RVMachine", i: "RV") -> None:
    """mstatus.FS on (30.13: the vector FP instructions use fcsr)."""
    require(m, i, m.inx or m.csrs.fs_on())


def _frm(m: "RVMachine", i: "RV") -> int:
    """fcsr.frm (30.13.1: the dynamic mode; a reserved one is illegal)."""
    rm = m.fcsr >> 5 & 7
    require(m, i, rm in _FRM)
    return _FRM[rm]


def _fp_common(m: "RVMachine", i: "RV", vu: VectorUnit, sew: int | None = None) -> F.Fmt:
    """VI_VFP_COMMON: FS on, a legal vtype, frm not reserved (every vector
    FP instruction reads it: Spike's VFP_RM), SEW one of the hart's FP
    widths."""
    _require_fp(m, i)
    _frm(m, i)
    w = vu.sew if sew is None else sew
    require(m, i, w in vu.fp)
    return FMT[w]


def _fscalar(m: "RVMachine", n: int, f: F.Fmt) -> int:
    """f[n] as an element of format f (NaN-boxing checked: 21.2; Zfinx:
    the x register's low bits)."""
    if m.inx:
        return m.read_reg(n) & ((1 << f.bits) - 1)
    return RF.unbox(m.fpr[n], f, m.flen)


def _canon(bits: int, f: F.Fmt) -> int:
    return RF.canonical(f) if F.is_nan(bits, f) else bits


def _fop(name: str) -> Callable[[int, int, F.Fmt, int], tuple[int, set[str]]]:
    """op(vs2's element a, the other operand b, fmt, rm) -> (bits, flags)."""
    def op(a, b, f, rm):
        if name in ("add", "sub", "rsub"):
            x, y = (b, a) if name == "rsub" else (a, b)
            return F.add(x, y, f, rm, subtract=name != "add", nan2008=True)
        if name == "mul":
            return F.mul(a, b, f, rm, nan2008=True)
        if name in ("div", "rdiv"):
            x, y = (b, a) if name == "rdiv" else (a, b)
            return F.div(x, y, f, rm, nan2008=True)
        if name in ("min", "max"):
            return RF.minmax(a, b, f, name)
        sb = f.sign_bit                         # sgnj / sgnjn / sgnjx: no flags
        s_ = b & sb if name == "sgnj" else ~b & sb if name == "sgnjn" else (a ^ b) & sb
        return (a & ~sb) | s_, set()
    return op


#: the single-width FP operations (30.13.2-30.13.13)
FSINGLE = {"vfadd": "add", "vfsub": "sub", "vfrsub": "rsub", "vfmul": "mul",
           "vfdiv": "div", "vfrdiv": "rdiv", "vfmin": "min", "vfmax": "max",
           "vfsgnj": "sgnj", "vfsgnjn": "sgnjn", "vfsgnjx": "sgnjx"}
#: the fused multiply-adds (30.13.6 / 30.13.7): (the addend is vd —
#: vd = ±(vs1 * vs2) ± vd — or vs2 — vd = ±(vs1 * vd) ± vs2, negate the
#: product, negate the addend)
FMA = {"vfmacc": (True, False, False), "vfnmacc": (True, True, True),
       "vfmsac": (True, False, True), "vfnmsac": (True, True, False),
       "vfmadd": (False, False, False), "vfnmadd": (False, True, True),
       "vfmsub": (False, False, True), "vfnmsub": (False, True, False)}


def _fb(m: "RVMachine", vu: VectorUnit, i: "RV", f: F.Fmt, n: int) -> Callable[[int], int]:
    """The second operand: vs1's element or f[rs1]."""
    if "vs1" in i.f:
        r = i.f["vs1"]
        return lambda k: vu.get(r, k, n)
    v = _fscalar(m, i.f["rs1"], f)
    return lambda k: v


def _fsingle(op_name: str):
    op = _fop(op_name)
    canon = op_name not in ("sgnj", "sgnjn", "sgnjx", "min", "max")

    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _check_sss(m, i, vu)
        f = _fp_common(m, i, vu)
        rm = _frm(m, i) if canon else F.RN
        n = vu.sew // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        b = _fb(m, vu, i, f, n)
        flags: set[str] = set()
        for k in _body(vu, i):
            v, fl = op(vu.get(vs2, k, n), b(k), f, rm)
            flags |= fl
            vu.put(vd, k, n, _canon(v, f) if canon else v)
        RF.accrue(m, flags)
    return ex


def _ffma(spec: tuple[bool, bool, bool], wide: bool = False):
    acc_vd, neg_p, neg_a = spec

    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        if wide:
            _wide_check(m, i, vu, False)
        else:
            _check_sss(m, i, vu)
        f = _fp_common(m, i, vu)
        if wide:
            require(m, i, 2 * vu.sew in vu.fp)
        g = FMT[2 * vu.sew] if wide else f
        rm = _frm(m, i)
        n = vu.sew // 8
        dn = 2 * n if wide else n
        vd, vs2 = i.f["vd"], i.f["vs2"]
        b = _fb(m, vu, i, f, n)
        flags: set[str] = set()
        for k in _body(vu, i):
            x, y, d = b(k), vu.get(vs2, k, n), vu.get(vd, k, dn)
            if wide:                            # (exact: the operands widened first)
                x, f1 = F.convert(x, f, g, rm, nan2008=True)
                y, f2 = F.convert(y, f, g, rm, nan2008=True)
                flags |= f1 | f2
            if acc_vd:
                v, fl = RF.fma(x, y, d, g, rm, neg_product=neg_p, neg_addend=neg_a)
            else:
                v, fl = RF.fma(x, d, y, g, rm, neg_product=neg_p, neg_addend=neg_a)
            flags |= fl
            vu.put(vd, k, dn, _canon(v, g))
        RF.accrue(m, flags)
    return ex


#: widening FP adds / subtracts / multiplies (30.13.3 / 30.13.5)
FWIDEN = {"vfwadd": "add", "vfwsub": "sub", "vfwmul": "mul"}


def _fwiden(op_name: str, wide_src: bool):
    op = _fop(op_name)

    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _wide_check(m, i, vu, wide_src)
        f = _fp_common(m, i, vu)
        require(m, i, 2 * vu.sew in vu.fp)
        g = FMT[2 * vu.sew]
        rm = _frm(m, i)
        n = vu.sew // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        b = _fb(m, vu, i, f, n)
        flags: set[str] = set()
        for k in _body(vu, i):
            if wide_src:
                a = vu.get(vs2, k, 2 * n)
            else:
                a, fl = F.convert(vu.get(vs2, k, n), f, g, rm, nan2008=True)
                flags |= fl
            y, fl = F.convert(b(k), f, g, rm, nan2008=True)
            v, fl2 = op(a, y, g, rm)
            flags |= fl | fl2
            vu.put(vd, k, 2 * n, _canon(v, g))
        RF.accrue(m, flags)
    return ex


def _funary(kind: str):
    """vfsqrt.v, vfrsqrt7.v, vfrec7.v, vfclass.v (30.13.8-30.13.14)."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _check_sss(m, i, vu)
        f = _fp_common(m, i, vu)
        rm = _frm(m, i) if kind in ("sqrt", "rec7") else F.RN
        n = vu.sew // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        flags: set[str] = set()
        for k in _body(vu, i):
            a = vu.get(vs2, k, n)
            if kind == "sqrt":
                v, fl = F.sqrt(a, f, rm, nan2008=True)
                v = _canon(v, f)
            elif kind == "rsqrt7":
                v, fl = rsqrt7(a, f)
            elif kind == "rec7":
                v, fl = rec7(a, f, rm)
            else:
                v, fl = RF.classify(a, f), set()
            flags |= fl
            vu.put(vd, k, n, v)
        RF.accrue(m, flags)
    return ex


#: 30.13.9 Table 58 (index: exp[0] then the six significand MSBs) and
#: 30.13.10 Table 59 (the seven significand MSBs) — the manual's tables
RSQRT7 = (52, 51, 50, 48, 47, 46, 44, 43, 42, 41, 40, 39, 38, 36, 35, 34, 33, 32, 31, 30, 30,
          29, 28, 27, 26, 25, 24, 23, 23, 22, 21, 20, 19, 19, 18, 17, 16, 16, 15, 14, 14, 13,
          12, 12, 11, 10, 10, 9, 9, 8, 7, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 127, 125,
          123, 121, 119, 118, 116, 114, 113, 111, 109, 108, 106, 105, 103, 102, 100, 99, 97,
          96, 95, 93, 92, 91, 90, 88, 87, 86, 85, 84, 83, 82, 80, 79, 78, 77, 76, 75, 74, 73,
          72, 71, 70, 70, 69, 68, 67, 66, 65, 64, 63, 63, 62, 61, 60, 59, 59, 58, 57, 56, 56,
          55, 54, 53)
REC7 = (127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99,
        97, 96, 94, 93, 91, 90, 88, 87, 85, 84, 83, 81, 80, 79, 77, 76, 75, 74, 72, 71, 70,
        69, 68, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
        47, 46, 45, 44, 43, 42, 41, 40, 40, 39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 31, 30,
        29, 28, 28, 27, 26, 25, 25, 24, 23, 23, 22, 21, 21, 20, 19, 19, 18, 17, 17, 16, 15,
        15, 14, 14, 13, 12, 12, 11, 11, 10, 9, 9, 8, 8, 7, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1,
        1, 0)


def _normalized(a: int, f: F.Fmt) -> tuple[int, int]:
    """(normalized exponent, normalized significand without its leading 1)
    — 30.13.9 / 30.13.10's rule for subnormal inputs."""
    e, sig = a >> f.fbits & ((1 << f.ebits) - 1), a & ((1 << f.fbits) - 1)
    if e == 0:
        while not sig >> (f.fbits - 1) & 1:
            e -= 1
            sig <<= 1
        sig = (sig << 1) & ((1 << f.fbits) - 1)
    return e, sig


def rsqrt7(a: int, f: F.Fmt) -> tuple[int, set[str]]:
    """vfrsqrt7.v's estimate (30.13.9)."""
    neg = F.sign(a, f)
    if F.is_nan(a, f):
        return RF.canonical(f), {"V"} if F.is_snan(a, f, True) else set()
    if F.value(a, f) == 0 and not F.is_inf(a, f):
        return (f.sign_bit if neg else 0) | f.inf, {"Z"}
    if neg:
        return RF.canonical(f), {"V"}
    if F.is_inf(a, f):
        return 0, set()
    e, sig = _normalized(a, f)
    out = RSQRT7[(e & 1) << 6 | sig >> (f.fbits - 6)] << (f.fbits - 7)
    bias = (1 << (f.ebits - 1)) - 1
    oe = (3 * bias - 1 - e) // 2
    return oe << f.fbits | out, set()


def rec7(a: int, f: F.Fmt, rm: int) -> tuple[int, set[str]]:
    """vfrec7.v's estimate (30.13.10)."""
    neg = F.sign(a, f)
    sgn = f.sign_bit if neg else 0
    if F.is_nan(a, f):
        return RF.canonical(f), {"V"} if F.is_snan(a, f, True) else set()
    if F.is_inf(a, f):
        return sgn, set()
    if F.value(a, f) == 0:
        return sgn | f.inf, {"Z"}
    e, sig = _normalized(a, f)
    bias = (1 << (f.ebits - 1)) - 1
    oe = 2 * bias - 1 - e
    if oe > 2 * bias:                           # (a tiny subnormal: overflow)
        to_max = rm == F.RZ or (rm == F.RM and not neg) or (rm == F.RP and neg)
        return sgn | (f.max_finite if to_max else f.inf), {"I", "O"}
    out = REC7[sig >> (f.fbits - 7)] << (f.fbits - 7)
    if oe <= 0:                                 # (a subnormal result, denormalized)
        out = out >> 1 | 1 << (f.fbits - 1)
        if oe < 0:
            out >>= 1
        oe = 0
    return sgn | oe << f.fbits | out, set()


#: the FP compares (30.13.14): (op, swap operands, negate)
FCOMPARE = {"vmfeq": ("eq", False, False), "vmfne": ("eq", False, True),
            "vmflt": ("lt", False, False), "vmfle": ("le", False, False),
            "vmfgt": ("lt", True, False), "vmfge": ("le", True, False)}


def _fcompare(spec: tuple[str, bool, bool]):
    op, swap, neg = spec

    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _check_mss(m, i, vu)
        f = _fp_common(m, i, vu)
        n = vu.sew // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        b = _fb(m, vu, i, f, n)
        flags: set[str] = set()
        res = []
        for k in _body(vu, i):
            x, y = vu.get(vs2, k, n), b(k)
            r, fl = RF.compare(y, x, f, op) if swap else RF.compare(x, y, f, op)
            flags |= fl
            res.append((k, r ^ neg))
        for k, r in res:
            vu.set_mask(k, r, vd)
        RF.accrue(m, flags)
    return ex


def _fmerge(m, i, pc):
    """vfmerge.vfm / vfmv.v.f (30.13.15-16): every body element."""
    vu = require_vector(m, i, True)
    _check_sss(m, i, vu)
    f = _fp_common(m, i, vu)
    n = vu.sew // 8
    vd, vs2, sel = i.f["vd"], i.f.get("vs2", 0), _vm(i) == 0
    x = _fscalar(m, i.f["rs1"], f)
    for k in range(vu.vstart, vu.vl):
        vu.put(vd, k, n, x if not sel or vu.mask(k) else vu.get(vs2, k, n))


def _vfmv_f_s(m, i, pc):
    """vfmv.f.s (30.16.2): element 0, NaN-boxed into f[rd]."""
    vu = require_vector(m, i, True)
    f = _fp_common(m, i, vu)
    v = vu.get(i.f["vs2"], 0, vu.sew // 8)
    if m.inx:
        m.write_reg(i.f["rd"], _sx(v, vu.sew))
    else:
        m.fpr[i.f["rd"]] = RF.box(v, vu.sew, m.flen)
        RF.dirty(m)
    del f


def _vfmv_s_f(m, i, pc):
    vu = require_vector(m, i, True)
    f = _fp_common(m, i, vu)
    if vu.vl > 0 and vu.vstart < vu.vl:
        vu.put(i.f["vd"], 0, vu.sew // 8, _fscalar(m, i.f["rs1"], f))


def _fslide(up: bool):
    """vfslide1up.vf / vfslide1down.vf (30.16.3)."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        f = i.f
        vd, vs2 = f["vd"], f["vs2"]
        require(m, i, aligned(vs2, vu.lmul) and aligned(vd, vu.lmul))
        require_vm(m, i)
        if up:
            require(m, i, vd != vs2)
        fmt = _fp_common(m, i, vu)
        n = vu.sew // 8
        x = _fscalar(m, f["rs1"], fmt)
        for k in _body(vu, i):
            if up:
                vu.put(vd, k, n, vu.get(vs2, k - 1, n) if k else x)
            else:
                vu.put(vd, k, n, vu.get(vs2, k + 1, n) if k != vu.vl - 1 else x)
    return ex


def _freduce(kind: str, wide: bool):
    """vfred[ou]sum / vfredmin / vfredmax (30.14.3-4), vfwred[ou]sum
    (30.14.5): in element order (an unordered sum may use any; Spike's)."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        require(m, i, (not wide or vu.sew * 2 <= vu.elen) and aligned(i.f["vs2"], vu.lmul))
        f = _fp_common(m, i, vu)
        if wide:
            require(m, i, 2 * vu.sew in vu.fp)
        g = FMT[2 * vu.sew] if wide else f
        rm = _frm(m, i)
        n = vu.sew // 8
        gn = g.bits // 8
        acc = vu.get(i.f["vs1"], 0, gn)
        flags: set[str] = set()
        active = False
        for k in _body(vu, i):
            active = True
            x = vu.get(i.f["vs2"], k, n)
            if wide:
                x, fl = F.convert(x, f, g, rm, nan2008=True)
                flags |= fl
            if kind == "sum":
                acc, fl = F.add(acc, x, g, rm, nan2008=True)
                acc = _canon(acc, g)
            else:
                acc, fl = RF.minmax(acc, x, g, kind)
            flags |= fl
        if vu.vl > 0:
            if kind == "sum" and i.row.name.startswith(("vfredusum", "vfwredusum")) \
                    and not active and F.is_nan(acc, g):
                if F.is_snan(acc, g, True):     # (unordered, nothing active: Spike's
                    flags.add("V")              #  canonical NaN for a NaN vs1[0])
                acc = RF.canonical(g)
            vu.put(i.f["vd"], 0, gn, acc)
        RF.accrue(m, flags)
    return ex


def _fcvt(kind: str, src: str, dst: str):
    """vfcvt / vfwcvt / vfncvt (30.13.17-19): ``kind`` "" (same width),
    "w" (widening) or "n" (narrowing); ``src`` / ``dst`` "f", "x", "xu",
    "rtz.x", "rtz.xu", "rod.f"."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _require_fp(m, i)
        _frm(m, i)
        w = vu.sew
        if kind == "w":
            _wide_check(m, i, vu, False)
            sw, dw = w, 2 * w
        elif kind == "n":
            _narrow_check(m, i, vu)
            sw, dw = 2 * w, w
        else:
            _check_sss(m, i, vu)
            sw = dw = w
        # the FP format(s) involved must exist (Zvfhmin: only f <-> f)
        if src == "f" and dst.endswith("f"):
            need = {sw, dw} - ({16} if vu.fp16min and dst == "f" and 16 in (sw, dw) else set())
        else:
            need = {sw if src == "f" else dw}
        require(m, i, need <= vu.fp and sw <= 64 and dw <= 64 and min(sw, dw) >= 8)
        rm = F.ROD if dst == "rod.f" else F.RZ if dst.startswith("rtz") else _frm(m, i)
        sn, dn = sw // 8, dw // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        flags: set[str] = set()
        res = []
        for k in _body(vu, i):
            a = vu.get(vs2, k, sn)
            if src == "f" and dst.endswith("f"):
                v, fl = F.convert(a, FMT[sw], FMT[dw], rm, nan2008=True)
                v = _canon(v, FMT[dw])
            elif src == "f":
                v, fl = RF.to_int(a, FMT[sw], dw, not dst.endswith("xu"), rm)
            else:
                v, fl = RF.from_int(a, sw, src == "x", FMT[dw], rm)
            flags |= fl
            res.append((k, v))
        for k, v in res:
            vu.put(vd, k, dn, v)
        RF.accrue(m, flags)
    return ex


# ------------------------------------------------ vector cryptography (ch. 32)

from . import rvcrypto as RC                    # noqa: E402

M32 = 0xFFFFFFFF


def _brev8(v: int, nbytes: int) -> int:
    """Each byte's bits reversed (vbrev8.v, GHASH's byte order)."""
    out = 0
    for k in range(nbytes):
        b = v >> (8 * k) & 0xFF
        out |= int(f"{b:08b}"[::-1], 2) << (8 * k)
    return out


def _rev8(v: int, nbytes: int) -> int:
    return int.from_bytes(v.to_bytes(nbytes, "little"), "big")


def _zvk(ext: str | tuple):
    """require_zvk*: VS on, a legal vtype, vstart 0 (32.1.5: vector crypto
    instructions are never interrupted part-way here), the extension."""
    exts = (ext,) if isinstance(ext, str) else ext

    def check(m: "RVMachine", i: "RV") -> VectorUnit:
        vu = require_vector(m, i, True)
        require(m, i, any(e in m.extensions for e in exts))
        return vu
    return check


#: Zvbb / Zvkb single-width operations (32.4): op(a, b, w) as SINGLE's
ZVBB = {
    "vandn": ("zvkb", lambda a, b, vu, d, w: a & ~b),
    "vrol": ("zvkb", lambda a, b, vu, d, w: (a << (b & (w - 1)) | a >> (-b & (w - 1)))),
    "vror": ("zvkb", lambda a, b, vu, d, w: (a >> (b & (w - 1)) | a << (-b & (w - 1)))),
    "vclmul": ("zvbc", lambda a, b, vu, d, w: RC_clmul(a, b)),
    "vclmulh": ("zvbc", lambda a, b, vu, d, w: RC_clmul(a, b) >> 64),
}


def RC_clmul(a: int, b: int) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _zvbb_single(name: str):
    ext, op = ZVBB[name]
    check = _zvk(ext)

    def ex(m, i, pc):
        vu = check(m, i)
        if name.startswith("vclmul"):
            require(m, i, vu.sew == 64)         # (32.1.5: vclmul[h] at SEW = 64)
        _check_sss(m, i, vu)
        w = vu.sew
        n, mk = w // 8, _ones(w)
        f = i.f
        vd, vs2 = f["vd"], f["vs2"]
        b = _src(m, vu, i, w)                    # (vror.vi: imm is the 6-bit uimm)
        for k in _body(vu, i):
            vu.put(vd, k, n, op(vu.get(vs2, k, n), b(k), vu, 0, w) & mk)
    return ex


#: the unary Zvbb / Zvkb operations
ZVBB_UNARY = {
    "vbrev": ("zvbb", lambda a, w: int(f"{a:0{w}b}"[::-1], 2)),
    "vbrev8": ("zvkb", lambda a, w: _brev8(a, w // 8)),
    "vrev8": ("zvkb", lambda a, w: _rev8(a, w // 8)),
    "vclz": ("zvbb", lambda a, w: w - a.bit_length()),
    "vctz": ("zvbb", lambda a, w: (a & -a).bit_length() - 1 if a else w),
    "vcpop": ("zvbb", lambda a, w: bin(a).count("1")),
}


def _zvbb_unary(name: str):
    ext, op = ZVBB_UNARY[name]
    check = _zvk(ext)

    def ex(m, i, pc):
        vu = check(m, i)
        _check_sss(m, i, vu)
        w = vu.sew
        n = w // 8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        for k in _body(vu, i):
            vu.put(vd, k, n, op(vu.get(vs2, k, n), w))
    return ex


def _vwsll(m, i, pc):
    """vwsll.v[vxi] (32.4: zero-extended, shifted by the low log2(2 * SEW)
    bits)."""
    vu = _zvk("zvbb")(m, i)
    _wide_check(m, i, vu, False)
    w = vu.sew
    n = w // 8
    vd, vs2 = i.f["vd"], i.f["vs2"]
    b = _src(m, vu, i, w)
    res = [(k, vu.get(vs2, k, n) << (b(k) & (2 * w - 1))) for k in _body(vu, i)]
    for k, v in res:
        vu.put(vd, k, 2 * n, v)


def _egs(m: "RVMachine", i: "RV", vu: VectorUnit, egs: int, egw: int, sew: int | tuple):
    """32.1.5's element-group constraints: SEW, EGW <= VLEN * LMUL (even
    when vl = 0), vl and vstart multiples of EGS, unmasked."""
    sews = (sew,) if isinstance(sew, int) else sew
    require(m, i, vu.sew in sews and egw <= vu.vlen * vu.lmul
            and vu.vl % egs == 0 and vu.vstart % egs == 0 and _vm(i) == 1)


def _vs_check(m: "RVMachine", i: "RV", vu: VectorUnit, egw: int) -> None:
    """A .vs instruction's scalar element group (32.1.5): vd aligned to
    LMUL, vs2 to EGW / VLEN, vs2 not overlapping vd's register group."""
    vd, vs2 = i.f["vd"], i.f["vs2"]
    eg = Fraction(egw, vu.vlen)
    g = max(int(vu.lmul), 1)
    require(m, i, aligned(vd, vu.lmul) and (eg <= 1 or aligned(vs2, eg))
            and vd // g != vs2 // g)


def _groups(vu: VectorUnit, egs: int) -> range:
    return range(vu.vstart // egs, vu.vl // egs)


def _words(v: int, n: int, size: int = 32) -> list[int]:
    mk = (1 << size) - 1
    return [v >> (size * k) & mk for k in range(n)]


def _pack(ws: list[int], size: int = 32) -> int:
    mk = (1 << size) - 1
    return sum((w & mk) << (size * k) for k, w in enumerate(ws))


def _gf128(y: int, h: int) -> int:
    """GHASH's multiply (32.4 vghsh / vgmul; the bit-reflected field of
    NIST SP 800-38D): operands and result byte-wise bit-reversed."""
    mk = (1 << 128) - 1
    h = _brev8(h, 16)
    z = 0
    for bit in range(128):
        if y >> bit & 1:
            z ^= h
        reduce = h >> 127 & 1
        h = (h << 1) & mk
        if reduce:
            h ^= 0x87
    return _brev8(z, 16)


def _zvkg(kind: str):
    def ex(m, i, pc):
        vu = _zvk("zvkg")(m, i)
        _egs(m, i, vu, 4, 128, 32)
        _check_sss(m, i, vu)
        vd, vs2 = i.f["vd"], i.f["vs2"]
        for g in _groups(vu, 4):
            y, h = vu.get(vd, g, 16), vu.get(vs2, g, 16)
            if kind == "vghsh":
                y ^= vu.get(i.f["vs1"], g, 16)
            vu.put(vd, g, 16, _gf128(_brev8(y, 16), h))
    return ex


def _aes_state(v: int) -> list[int]:
    return list(v.to_bytes(16, "little"))


def _shift_rows(st: list[int], inverse: bool) -> list[int]:
    return [st[4 * ((c - r) % 4 if inverse else (c + r) % 4) + r]
            for c in range(4) for r in range(4)]


def _mix(st: list[int], inverse: bool) -> list[int]:
    out = []
    for c in range(4):
        word = RC._word(st[4 * c:4 * c + 4])
        out += RC._bytes32(RC.mixcolumn_inv(word) if inverse else RC.mixcolumn_fwd(word))
    return out


def _aes_round(state: int, key: int, kind: str) -> int:
    """vaes{e,d}{m,f} (32.4): one AES round (middle or final), encrypting
    or decrypting, with round key ``key``."""
    st = _aes_state(state)
    if kind[0] == "e":
        st = [RC.AES_SBOX[b] for b in st]
        st = _shift_rows(st, False)
        if kind == "em":
            st = _mix(st, False)
        st = [b ^ k for b, k in zip(st, _aes_state(key))]
    else:
        st = _shift_rows(st, True)
        st = [RC.AES_SBOX_INV[b] for b in st]
        st = [b ^ k for b, k in zip(st, _aes_state(key))]
        if kind == "dm":
            st = _mix(st, True)
    return int.from_bytes(bytes(st), "little")


def _vaes(kind: str, scalar: bool):
    """vaesef / vaesem / vaesdf / vaesdm .vv / .vs, vaesz.vs (32.4)."""
    def ex(m, i, pc):
        vu = _zvk("zvkned")(m, i)
        _egs(m, i, vu, 4, 128, 32)
        if scalar:
            _vs_check(m, i, vu, 128)
        else:
            _check_sss(m, i, vu)
        vd, vs2 = i.f["vd"], i.f["vs2"]
        key0 = vu.get(vs2, 0, 16)
        for g in _groups(vu, 4):
            key = key0 if scalar else vu.get(vs2, g, 16)
            state = vu.get(vd, g, 16)
            vu.put(vd, g, 16, state ^ key if kind == "z" else _aes_round(state, key, kind))
    return ex


def _subword_rot(temp: int, rot: bool) -> int:
    if rot:
        temp = (temp >> 8 | temp << 24) & M32
    return RC.subword(temp, RC.AES_SBOX)


def _vaeskf(two: bool):
    """vaeskf1.vi (AES-128) / vaeskf2.vi (AES-256) key schedule (32.4):
    an out-of-range round number has bit 3 toggled (as the manual)."""
    def ex(m, i, pc):
        vu = _zvk("zvkned")(m, i)
        _egs(m, i, vu, 4, 128, 32)
        _check_sss(m, i, vu)
        r = i.f["zimm"] & 0xF
        lo, hi = (2, 14) if two else (1, 10)
        if not lo <= r <= hi:
            r ^= 0x8
        vd, vs2 = i.f["vd"], i.f["vs2"]
        for g in _groups(vu, 4):
            cur = _words(vu.get(vs2, g, 16), 4)
            if two:
                even = r & 1 == 0
                temp = _subword_rot(cur[3], even) ^ (RC.RCON[(r >> 1) - 1] if even else 0)
                prev = _words(vu.get(vd, g, 16), 4)
            else:
                temp = _subword_rot(cur[3], True) ^ RC.RCON[r - 1]
                prev = cur
            w0 = prev[0] ^ temp
            w1 = prev[1] ^ w0
            w2 = prev[2] ^ w1
            w3 = prev[3] ^ w2
            vu.put(vd, g, 16, _pack([w0, w1, w2, w3]))
    return ex


def _sha2_funcs(w: int):
    if w == 32:
        ror, mk = RC.ror32, M32
        s0 = lambda x: ror(x, 2) ^ ror(x, 13) ^ ror(x, 22)
        s1 = lambda x: ror(x, 6) ^ ror(x, 11) ^ ror(x, 25)
        g0 = lambda x: ror(x, 7) ^ ror(x, 18) ^ x >> 3
        g1 = lambda x: ror(x, 17) ^ ror(x, 19) ^ x >> 10
    else:
        ror, mk = RC.ror64, (1 << 64) - 1
        s0 = lambda x: ror(x, 28) ^ ror(x, 34) ^ ror(x, 39)
        s1 = lambda x: ror(x, 14) ^ ror(x, 18) ^ ror(x, 41)
        g0 = lambda x: ror(x, 1) ^ ror(x, 8) ^ x >> 7
        g1 = lambda x: ror(x, 19) ^ ror(x, 61) ^ x >> 6
    return mk, s0, s1, g0, g1


def _vsha2(kind: str):
    """vsha2ms / vsha2ch / vsha2cl (32.4): SHA-256 at SEW = 32 (Zvknha /
    Zvknhb), SHA-512 at SEW = 64 (Zvknhb)."""
    def ex(m, i, pc):
        vu = require_vector(m, i, True)
        _check_sss(m, i, vu)
        f = i.f
        vd, vs1, vs2 = f["vd"], f["vs1"], f["vs2"]
        require(m, i, vu.sew in (32, 64) and vd != vs1 and vd != vs2)
        w = vu.sew
        if w == 32:
            require(m, i, "zvknha" in m.extensions or "zvknhb" in m.extensions)
        else:
            require(m, i, "zvknhb" in m.extensions)
        egw = 4 * w
        _egs(m, i, vu, 4, egw, w)
        mk, s0, s1, g0, g1 = _sha2_funcs(w)
        nb = egw // 8
        for g in _groups(vu, 4):
            d = _words(vu.get(vd, g, nb), 4, w)
            b = _words(vu.get(vs2, g, nb), 4, w)
            a1 = _words(vu.get(vs1, g, nb), 4, w)
            if kind == "ms":
                w0, w1, w2, w3 = d
                w4, w9, w10, w11 = b
                w12, _w13, w14, w15 = a1
                sch = lambda x14, x9, x1, x0: (g1(x14) + x9 + g0(x1) + x0) & mk
                w16 = sch(w14, w9, w1, w0)
                w17 = sch(w15, w10, w2, w1)
                w18 = sch(w16, w11, w3, w2)
                w19 = sch(w17, w12, w4, w3)
                vu.put(vd, g, nb, _pack([w16, w17, w18, w19], w))
                continue
            h, g_, d_, c = d                    # vd = {c, d, g, h} (high to low)
            f_, e, b_, a = b                    # vs2 = {a, b, e, f}
            kws = (a1[2], a1[3]) if kind == "ch" else (a1[0], a1[1])
            for kw in kws:
                t1 = (h + s1(e) + ((e & f_) ^ (~e & g_)) + kw) & mk
                t2 = (s0(a) + ((a & b_) ^ (a & c) ^ (b_ & c))) & mk
                h, g_, f_, e = g_, f_, e, (d_ + t1) & mk
                d_, c, b_, a = c, b_, a, (t1 + t2) & mk
            vu.put(vd, g, nb, _pack([f_, e, b_, a], w))
    return ex


#: SM4's CK constants (GB/T 32907-2016, 32.4 vsm4k)
SM4_CK = tuple(sum(((4 * k + j) * 7 % 256) << (24 - 8 * j) for j in range(4)) for k in range(32))


def _sm4_sub(x: int) -> int:
    return RC.subword(x, RC.SM4_SBOX)


def _vsm4k(m, i, pc):
    vu = _zvk("zvksed")(m, i)
    _egs(m, i, vu, 4, 128, 32)
    _check_sss(m, i, vu)
    rnd = i.f["zimm"] & 7
    vd, vs2 = i.f["vd"], i.f["vs2"]
    rol = RC.rol32
    for g in _groups(vu, 4):
        rk = _words(vu.get(vs2, g, 16), 4)
        for j in range(4):
            s = _sm4_sub(rk[-3] ^ rk[-2] ^ rk[-1] ^ SM4_CK[4 * rnd + j])
            rk.append(rk[-4] ^ s ^ rol(s, 13) ^ rol(s, 23))
        vu.put(vd, g, 16, _pack(rk[4:]))


def _sm4_rounds(x: list[int], rk: list[int]) -> list[int]:
    rol = RC.rol32
    x = list(x)
    for j in range(4):
        s = _sm4_sub(x[-3] ^ x[-2] ^ x[-1] ^ rk[j])
        x.append(x[-4] ^ s ^ rol(s, 2) ^ rol(s, 10) ^ rol(s, 18) ^ rol(s, 24))
    return x[4:]


def _vsm4r(scalar: bool):
    def ex(m, i, pc):
        vu = _zvk("zvksed")(m, i)
        _egs(m, i, vu, 4, 128, 32)
        if scalar:
            _vs_check(m, i, vu, 128)
        else:
            _check_sss(m, i, vu)
        vd, vs2 = i.f["vd"], i.f["vs2"]
        key0 = _words(vu.get(vs2, 0, 16), 4)
        for g in _groups(vu, 4):
            rk = key0 if scalar else _words(vu.get(vs2, g, 16), 4)
            vu.put(vd, g, 16, _pack(_sm4_rounds(_words(vu.get(vd, g, 16), 4), rk)))
    return ex


def _bswap_words(v: int) -> list[int]:
    """An SM3 element group: 8 words, each byte-swapped (32.4: big-endian
    words in memory order)."""
    return [_rev8(x, 4) for x in _words(v, 8)]


def _from_bswap(ws: list[int]) -> int:
    return _pack([_rev8(x, 4) for x in ws])


def _vsm3me(m, i, pc):
    vu = _zvk("zvksh")(m, i)
    _egs(m, i, vu, 8, 256, 32)
    _check_sss(m, i, vu)
    vd, vs1, vs2 = i.f["vd"], i.f["vs1"], i.f["vs2"]
    require(m, i, vd != vs2)
    rol = RC.rol32
    p1 = lambda x: x ^ rol(x, 15) ^ rol(x, 23)
    for g in _groups(vu, 8):
        w = _bswap_words(vu.get(vs1, g, 32)) + _bswap_words(vu.get(vs2, g, 32))
        for j in range(16, 24):
            w.append(p1(w[j - 16] ^ w[j - 9] ^ rol(w[j - 3], 15)) ^ rol(w[j - 13], 7) ^ w[j - 6])
        vu.put(vd, g, 32, _from_bswap(w[16:24]))


def _vsm3c(m, i, pc):
    vu = _zvk("zvksh")(m, i)
    _egs(m, i, vu, 8, 256, 32)
    _check_sss(m, i, vu)
    vd, vs2 = i.f["vd"], i.f["vs2"]
    require(m, i, vd != vs2)
    rnd = i.f["zimm"]
    rol = RC.rol32
    p0 = lambda x: x ^ rol(x, 9) ^ rol(x, 17)
    for g in _groups(vu, 8):
        a, b, c, d, e, f_, g_, h = _bswap_words(vu.get(vd, g, 32))
        w = _bswap_words(vu.get(vs2, g, 32))
        out = []
        for j, wj, xj in ((2 * rnd, w[0], w[0] ^ w[4]), (2 * rnd + 1, w[1], w[1] ^ w[5])):
            t = 0x79CC4519 if j <= 15 else 0x7A879D8A
            ss1 = rol((rol(a, 12) + e + rol(t, j % 32)) & M32, 7)
            ss2 = ss1 ^ rol(a, 12)
            ff = a ^ b ^ c if j <= 15 else (a & b) | (a & c) | (b & c)
            gg = e ^ f_ ^ g_ if j <= 15 else (e & f_) | (~e & g_)
            tt1 = (ff + d + ss2 + xj) & M32
            tt2 = (gg + h + ss1 + wj) & M32
            d, c, b, a = c, rol(b, 9), a, tt1
            h, g_, f_, e = g_, rol(f_, 19), e, p0(tt2)
            out.append((a, c, e, g_))
        (a1, c1, e1, g1), (a2, c2, e2, g2) = out
        vu.put(vd, g, 32, _from_bswap([a2, a1, c2, c1, e2, e1, g2, g1]))


# --------------------------------------------- the pipeline's view (LF6f)


def is_vector(i: "RV") -> bool:
    """A vector instruction (rv_v / the vector cryptography rows)."""
    return i.row.file.startswith(("rv_v", "rv_zv"))


#: the element-group sizes (32.1.4) by mnemonic prefix
_EGS = {"vaes": 4, "vsha2": 4, "vgh": 4, "vgm": 4, "vsm4": 4, "vsm3": 8}


def resources(i: "RV", vu: VectorUnit | None, latency: dict[str, int]) \
        -> tuple[set, set, int, bool]:
    """(registers read, registers written, EX cycles, a load) of a vector
    instruction for the 5-stage pipeline (``emips.rvpipeline``) under the
    hart's vtype: ``("v", n)`` for each register of a group (EMUL from
    30.5.2 / 30.7.3 / 32.1.4), ``("r", n)`` / ``("f", n)`` the scalars;
    the old vd is read (inactive and tail elements are kept: eMIPS's
    agnostic policy), v0 under a mask.  Timing: one element (one element
    group, one memory access) a cycle through EX, plus the unit's latency
    for multiplies / divides (not pipelined: per element) / FP."""
    name, f = i.row.name, i.f
    reads: set = set()
    writes: set = set()
    if vu is None:
        return reads, writes, 1, False
    lmul, sew, vl = vu.lmul, vu.sew, vu.vl
    if vu.vill and not name.startswith("vset"):
        lmul, sew, vl = Fraction(1), 8, 0       # (illegal: it traps in EX)

    def group(n: int, emul: Fraction) -> set:
        return {("v", (n + k) & 31) for k in range(max(int(emul), 1))}

    stem = name.split(".")[0]
    if name.startswith("vset"):
        writes = {("r", f["rd"])} if f.get("rd") else set()
        reads = {("r", f["rs1"])} if f.get("rs1") and name != "vsetivli" else set()
        if name == "vsetvl" and f.get("rs2"):
            reads.add(("r", f["rs2"]))
        return reads, writes, 1, False
    if "rs1" in f and f["rs1"]:
        fp_scalar = name.endswith((".vf", ".wf", ".vfm")) or name in ("vfmv.v.f", "vfmv.s.f")
        reads.add(("f" if fp_scalar and not i.inx else "r", f["rs1"]))
    if "rs2" in f and f["rs2"]:
        reads.add(("r", f["rs2"]))
    if "rd" in f and f["rd"]:
        writes.add(("f" if name == "vfmv.f.s" and not i.inx else "r", f["rd"]))
    if _vm(i) == 0:
        reads.add(("v", 0))
    mem = name.startswith(("vl", "vs")) and not name.startswith(("vslide", "vsll", "vsra",
                                                                 "vsrl", "vsadd", "vssub",
                                                                 "vsmul", "vssr", "vsbc",
                                                                 "vsext", "vsm3", "vsm4",
                                                                 "vsha2"))
    if mem:
        nf = f.get("nf", 0) + 1
        m = re.match(r"v[ls](\d)re?(\d+)?r?\.v|v[ls]m\.v|v[ls]\w*?e(?:i)?(\d+)", name)
        whole = re.match(r"v[ls](\d)r", name)
        load = name.startswith("vl")
        data = f.get("vd", f.get("vs3"))
        if whole:
            n = int(whole.group(1))
            (writes if load else reads).update(group(data, Fraction(n)))
            eew = int(re.search(r"e(\d+)", name).group(1)) if load else 8
            return reads, writes, max(1, n * vu.vlenb * 8 // eew), load
        if name in ("vlm.v", "vsm.v"):
            (writes if load else reads).add(("v", data))
            if load:
                reads.add(("v", data))
            return reads, writes, max(1, (vl + 7) // 8), load
        eew = int(re.search(r"(\d+)", name.split(".")[0][2:]).group(1)) if m else sew
        if "ei" in name:
            reads |= group(f["vs2"], Fraction(eew, sew) * lmul)
            demul = lmul
        else:
            demul = Fraction(eew, sew) * lmul
        g = max(int(demul), 1)
        regs = set()
        for k in range(nf):
            regs |= group(data + k * g, demul)
        (writes if load else reads).update(regs)
        if load:
            reads |= regs
        return reads, writes, max(1, vl * nf), load
    # arithmetic: the operands' EMUL
    wide_d = name.startswith(("vw", "vfw")) and not name.startswith(("vwred", "vfwred"))
    wide_s2 = name.startswith(("vn", "vfn")) or ".w" in name
    mask_d = stem.startswith(("vms", "vmf", "vmadc", "vmsbc")) or name.endswith(".mm") \
        or stem in ("vmsbf", "vmsif", "vmsof")
    red = name.endswith(".vs") and not stem.startswith(("vaes", "vsm4"))
    ext = re.search(r"vf(\d)$", name)
    e_d = Fraction(1) if mask_d or red or name in ("vmv.s.x", "vfmv.s.f") else \
        2 * lmul if wide_d else lmul
    nr = re.match(r"vmv(\d)r", name)
    if nr:
        e_d = Fraction(int(nr.group(1)))
    if "vd" in f:
        writes |= group(f["vd"], e_d)
        if not nr:
            reads |= group(f["vd"], e_d)
    for key in ("vs1", "vs2"):
        if key not in f:
            continue
        e = lmul
        if key == "vs2" and wide_s2 or (key == "vs1" and red and wide_d is False and
                                        name.startswith(("vwred", "vfwred"))):
            e = 2 * lmul
        if key == "vs2" and ext:
            e = lmul / int(ext.group(1))
        if key == "vs1" and (red or name in ("vcompress.vm",) or name.endswith(".mm")):
            e = Fraction(1)
        if key == "vs2" and (name.endswith((".mm", ".m")) or stem in ("vmsbf", "vmsif", "vmsof",
                                                                     "viota", "vcpop", "vfirst")
                             or name in ("vmv.x.s", "vfmv.f.s")):
            e = Fraction(1)
        if key == "vs2" and name.endswith(".vs") and stem.startswith(("vaes", "vsm4")):
            e = Fraction(128, vu.vlen)
        if key == "vs1" and name == "vrgatherei16.vv":
            e = Fraction(16, sew) * lmul
        if nr:
            e = Fraction(int(nr.group(1)))
        reads |= group(f[key], e)
    if wide_d and name.startswith(("vw", "vfw")) and ".w" not in name:
        pass
    egs = next((v for k, v in _EGS.items() if stem.startswith(k)), 1)
    work = max(1, vl // egs)
    if stem.startswith(("vdiv", "vrem", "vfdiv", "vfrdiv", "vfsqrt")):
        unit = latency["DIV"] if not stem.startswith("vf") else \
            latency["FP_DIV_D" if sew == 64 else "FP_DIV_S"]
        return reads, writes, max(1, vl) * unit, False
    if stem.startswith(("vmul", "vwmul", "vmacc", "vnmsac", "vmadd", "vnmsub", "vwmacc",
                        "vsmul", "vclmul")):
        lat = latency["MUL"]
    elif stem.startswith(("vfma", "vfnma", "vfms", "vfnms", "vfwma", "vfwnma", "vfwms",
                          "vfwnms")):
        lat = latency["FP_MADD"]
    elif stem.startswith(("vfmul", "vfwmul")):
        lat = latency["FP_MUL"]
    elif stem.startswith(("vfadd", "vfsub", "vfrsub", "vfwadd", "vfwsub", "vfred", "vfwred")):
        lat = latency["FP_ADD"]
    elif stem.startswith(("vfcvt", "vfwcvt", "vfncvt")):
        lat = latency["FP_CVT"]
    else:
        lat = 1
    return reads, writes, work + lat - 1, False


# ----------------------------------------------------------------- the table


def executors() -> dict[str, Callable]:
    ex: dict[str, Callable] = {"vsetvli": _vsetvli, "vsetivli": _vsetivli, "vsetvl": _vsetvl,
                               "vlm.v": _unit(None, False), "vsm.v": _unit(None, True)}
    for eew in (8, 16, 32, 64):
        ex[f"vle{eew}.v"] = _unit(eew, False)
        ex[f"vse{eew}.v"] = _unit(eew, True)
        ex[f"vle{eew}ff.v"] = _unit(eew, False, ff=True)
        ex[f"vlse{eew}.v"] = _strided(eew, False)
        ex[f"vsse{eew}.v"] = _strided(eew, True)
        for o in "uo":
            ex[f"vl{o}xei{eew}.v"] = _indexed(eew, False)
            ex[f"vs{o}xei{eew}.v"] = _indexed(eew, True)
        for nf in (1, 2, 4, 8):
            ex[f"vl{nf}re{eew}.v"] = _whole_load(eew)
    for nf in (1, 2, 4, 8):
        ex[f"vs{nf}r.v"] = _whole_store
        ex[f"vmv{nf}r.v"] = _vmv_nr(nf)
    for name, op in SINGLE.items():
        for form in ("vv", "vx", "vi"):
            ex[f"{name}.{form}"] = _single(op, name)
    for name, cmp in COMPARE.items():
        for form in ("vv", "vx", "vi"):
            ex[f"{name}.{form}"] = _compare(cmp)
    for name in ("vadc", "vsbc", "vmadc", "vmsbc"):
        for form in ("vvm", "vxm", "vim", "vv", "vx", "vi"):
            ex[f"{name}.{form}"] = _carry(name)
    for form in ("vvm", "vxm", "vim"):
        ex[f"vmerge.{form}"] = _merge
    for form in ("v.v", "v.x", "v.i"):
        ex[f"vmv.{form}"] = _merge
    for name in WIDEN:
        for form in ("vv", "vx"):
            ex[f"{name}.{form}"] = _widen(name, False)
        for form in ("wv", "wx"):
            ex[f"{name}.{form}"] = _widen(name, True)
    for div in (2, 4, 8):
        ex[f"vzext.vf{div}"] = _ext(div, False)
        ex[f"vsext.vf{div}"] = _ext(div, True)
    for name in ("vnsrl", "vnsra", "vnclipu", "vnclip"):
        for form in ("wv", "wx", "wi"):
            ex[f"{name}.{form}"] = _narrow(name)
    for name in REDUCE:
        ex[f"{name}.vs"] = _reduce(name, False)
    ex["vwredsum.vs"] = _reduce("vwredsum", True)
    ex["vwredsumu.vs"] = _reduce("vwredsumu", True)
    for name, op in MASK_LOGIC.items():
        ex[f"{name}.mm"] = _mask_logic(op)
    ex.update({"vcpop.m": _vcpop, "vfirst.m": _vfirst, "vmsbf.m": _set_first("bf"),
               "vmsif.m": _set_first("if"), "vmsof.m": _set_first("of"), "viota.m": _viota,
               "vid.v": _vid, "vmv.x.s": _vmv_x_s, "vmv.s.x": _vmv_s_x,
               "vslideup.vx": _slide("up"), "vslideup.vi": _slide("up"),
               "vslidedown.vx": _slide("down"), "vslidedown.vi": _slide("down"),
               "vslide1up.vx": _slide("1up"), "vslide1down.vx": _slide("1down"),
               "vrgather.vv": _gather("vv"), "vrgather.vx": _gather("vx"),
               "vrgather.vi": _gather("vi"), "vrgatherei16.vv": _gather("ei16"),
               "vcompress.vm": _vcompress})
    for name, op in FSINGLE.items():
        for form in ("vv", "vf"):
            ex[f"{name}.{form}"] = _fsingle(op)
    for name, spec in FMA.items():
        for form in ("vv", "vf"):
            ex[f"{name}.{form}"] = _ffma(spec)
    for name, spec in (("vfwmacc", FMA["vfmacc"]), ("vfwnmacc", FMA["vfnmacc"]),
                       ("vfwmsac", FMA["vfmsac"]), ("vfwnmsac", FMA["vfnmsac"])):
        for form in ("vv", "vf"):
            ex[f"{name}.{form}"] = _ffma(spec, wide=True)
    for name, op in FWIDEN.items():
        for form in ("vv", "vf"):
            ex[f"{name}.{form}"] = _fwiden(op, False)
        for form in ("wv", "wf"):
            ex[f"{name}.{form}"] = _fwiden(op, True)
    for kind in ("sqrt", "rsqrt7", "rec7", "class"):
        ex[f"vf{kind}.v"] = _funary(kind)
    for name, spec in FCOMPARE.items():
        for form in ("vv", "vf"):
            ex[f"{name}.{form}"] = _fcompare(spec)
    ex.update({"vfmerge.vfm": _fmerge, "vfmv.v.f": _fmerge, "vfmv.f.s": _vfmv_f_s,
               "vfmv.s.f": _vfmv_s_f, "vfslide1up.vf": _fslide(True),
               "vfslide1down.vf": _fslide(False),
               "vfredosum.vs": _freduce("sum", False), "vfredusum.vs": _freduce("sum", False),
               "vfredmin.vs": _freduce("min", False), "vfredmax.vs": _freduce("max", False),
               "vfwredosum.vs": _freduce("sum", True), "vfwredusum.vs": _freduce("sum", True)})
    for name in ZVBB:
        for form in ("vv", "vx", "vi"):
            ex[f"{name}.{form}"] = _zvbb_single(name)
    for name in ZVBB_UNARY:
        ex[f"{name}.v"] = _zvbb_unary(name)
    for form in ("vv", "vx", "vi"):
        ex[f"vwsll.{form}"] = _vwsll
    ex.update({"vghsh.vv": _zvkg("vghsh"), "vgmul.vv": _zvkg("vgmul"),
               "vaesz.vs": _vaes("z", True), "vaeskf1.vi": _vaeskf(False),
               "vaeskf2.vi": _vaeskf(True), "vsha2ms.vv": _vsha2("ms"),
               "vsha2ch.vv": _vsha2("ch"), "vsha2cl.vv": _vsha2("cl"), "vsm4k.vi": _vsm4k,
               "vsm4r.vv": _vsm4r(False), "vsm4r.vs": _vsm4r(True), "vsm3me.vv": _vsm3me,
               "vsm3c.vi": _vsm3c})
    for kind in ("ef", "em", "df", "dm"):
        ex[f"vaes{kind}.vv"] = _vaes(kind, False)
        ex[f"vaes{kind}.vs"] = _vaes(kind, True)
    for kind, tail in (("", "v"), ("w", "v"), ("n", "w")):
        pre = f"vf{kind}cvt" if kind else "vfcvt"
        for dst, src in (("xu", "f"), ("x", "f"), ("rtz.xu", "f"), ("rtz.x", "f"),
                         ("f", "xu"), ("f", "x"), ("f", "f"), ("rod.f", "f")):
            ex[f"{pre}.{dst}.{src}.{tail}"] = _fcvt(kind, src, dst)
    return {k: _dirtying(v) for k, v in ex.items()}


def _dirtying(fn: Callable) -> Callable:
    """Run a vector instruction; one that ran (or trapped part-way, its
    vstart written) leaves mstatus.VS Dirty (30.3.2) — an illegal one does
    not (Spike)."""
    def ex(m, i, pc):
        try:
            r = fn(m, i, pc)
        except Exception as t:
            if getattr(t, "cause", ILLEGAL) != ILLEGAL:
                dirty(m)
            raise
        dirty(m)
        return r
    return ex


__all__ = ["VectorUnit", "executors", "VS_SHIFT", "CSR"]
